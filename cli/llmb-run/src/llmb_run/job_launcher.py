# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Job launching functionality for different launcher types."""

import fcntl
import importlib
import json
import logging
import os
import re
import subprocess
import tempfile
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

import yaml
from rich.console import Console

from llmb_run.config_manager import ClusterConfig, ExecutionTarget
from llmb_run.constants import (
    DEFAULT_NVL_DOMAIN_BY_GPU,
    GPU_TYPE_TO_NUM_GPUS,
    MBRIDGE_GPU_TYPE_MAP,
    SLURM_OUTPUT_PATTERN,
)
from llmb_run.env_args import (
    apply_mbridge_extra_args_contract,
    apply_nemo_explicit_env_contract,
    apply_nemo_workload_args_contract,
    apply_sbatch_explicit_env_contract,
)
from llmb_run.job_history import record_job_submission
from llmb_run.metadata_utils import RunConfig
from llmb_run.nsys_mount_handler import get_tool_mounts
from llmb_run.run_config import create_llmb_config, resolve_container_images
from llmb_run.slurm_args import (
    ADDITIONAL_SLURM_PARAMS_KEY,
    SlurmArgs,
    validate_no_additional_slurm_params_conflict,
)
from llmb_run.slurm_utils import SlurmJob, get_cluster_name, parse_slurm_job_id
from llmb_run.tasks import format_task_output

logger = logging.getLogger('llmb_run.job_launcher')
console = Console()

# Pre-compiled regex patterns for filtering Nemo2 status instructions
NEMO2_PYTHON_API_PATTERN = re.compile(
    r'# The experiment was run with the following tasks.*?' r'experiment\.cancel\([^)]*\)[^\n]*', re.DOTALL
)

NEMO2_CLI_PATTERN = re.compile(
    r'# You can inspect this experiment at a later point in time using the CLI as well:.*?'
    r'nemo experiment cancel [^\n]*',
    re.DOTALL,
)

NEMO2_CLEANUP_NEWLINES_PATTERN = re.compile(r'\n\s*\n\s*\n+')


def sanitize_srt_submission_environment(env: dict[str, str]) -> dict[str, str]:
    """Remove login-host Python environments before SRT submits compute jobs."""
    sanitized = dict(env)
    environment_roots = {
        os.path.normpath(value)
        for name in ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT")
        if (value := sanitized.get(name))
    }
    path_entries = sanitized.get("PATH", "").split(os.pathsep)
    sanitized["PATH"] = os.pathsep.join(
        entry
        for entry in path_entries
        if entry and all(os.path.normpath(entry) != os.path.join(root, "bin") for root in environment_roots)
    )
    for name in (
        "VIRTUAL_ENV",
        "CONDA_PREFIX",
        "CONDA_DEFAULT_ENV",
        "UV_PROJECT_ENVIRONMENT",
        "PYTHONHOME",
        "PYTHONPATH",
    ):
        sanitized.pop(name, None)
    return sanitized


def should_enable_vboost(config):
    """Determine if VBoost should be automatically enabled based on cluster configuration.

    Args:
        config: Cluster configuration dictionary

    Returns:
        bool: True if VBoost should be enabled, False otherwise
    """
    # Check if cluster_name is explicitly configured
    cluster_name = config.cluster_name

    # If not configured, try to detect from SLURM
    if not cluster_name:
        cluster_name = get_cluster_name()

    # Enable VBoost for 'eos' cluster
    if cluster_name and cluster_name.lower() == 'eos':
        return True

    return False


def _compute_segment_size(num_nodes: int, domain_nodes: int) -> int:
    """Return the greatest divisor of num_nodes that is <= domain_nodes.

    domain_nodes is the NVLink domain (or Slurm base block) expressed in nodes,
    so the result is the largest segment that both tiles num_nodes and fits in
    a single domain.
    """
    for d in range(min(num_nodes, domain_nodes), 0, -1):
        if num_nodes % d == 0:
            return d
    return 1


def filter_nemo2_status_instructions(output: str) -> str:
    """
    Filter out Nemo2 experiment status instructions from launcher output.

    Removes the sections that tell users how to check job status using
    experiment.status(), experiment.logs(), etc. and 'nemo experiment' CLI commands.
    Each section is filtered independently to handle ordering changes.

    Args:
        output: The raw output from the Nemo2 launcher

    Returns:
        Filtered output with status instructions removed
    """
    filtered_output = output

    # Remove Python API section using pre-compiled pattern
    filtered_output = NEMO2_PYTHON_API_PATTERN.sub('', filtered_output)

    # Remove CLI section using pre-compiled pattern
    filtered_output = NEMO2_CLI_PATTERN.sub('', filtered_output)

    # Clean up multiple consecutive newlines using pre-compiled pattern
    filtered_output = NEMO2_CLEANUP_NEWLINES_PATTERN.sub('\n\n', filtered_output)

    return filtered_output.strip()


def parse_nemo2_output(output: str) -> tuple[str, str, str]:
    """Parse Nemo launcher output to extract job ID and local directory.

    Args:
        output: Complete output from the Nemo launcher

    Returns:
        tuple: (job_id, local_directory, error_message)
               job_id and local_directory are None if not found or error occurred
               error_message is None if no error, otherwise contains error description
    """
    job_id = None
    local_dir = None
    error_msg = None

    # Find all Task headers to validate we have the expected structure
    task_headers = re.findall(r'^Task (\d+):', output, re.MULTILINE)

    if not task_headers:
        error_msg = "No Task headers found in output"
        return None, None, error_msg

    if len(task_headers) > 1:
        error_msg = f"Multiple tasks found ({len(task_headers)} tasks: {task_headers}). Expected only Task 0."
        return None, None, error_msg

    # Extract the Task 0 section for parsing
    task_0_match = re.search(r'Task 0:.*?(?=\n\n|\nTask \d+:|\n#|$)', output, re.DOTALL)
    if not task_0_match:
        error_msg = "Could not extract Task 0 section from output"
        return None, None, error_msg

    task_0_section = task_0_match.group(0)

    # Look for job ID pattern within Task 0 section: "- Job id: 3530909"
    job_id_match = re.search(r'- Job id:\s+(\d+)', task_0_section)
    if job_id_match:
        job_id = job_id_match.group(1)

    # Look for local directory pattern within Task 0 section: "- Local Directory: /path/to/directory"
    local_dir_match = re.search(r'- Local Directory:\s+(.+)', task_0_section)
    if local_dir_match:
        local_dir = local_dir_match.group(1).strip()

    # Validate that if we found a job_id, we should also have a local_dir
    if job_id and not local_dir:
        error_msg = f"Found job ID {job_id} but no local directory - this should not happen"
        return None, None, error_msg

    return job_id, local_dir, None


def get_venv_environment(venv_path: str, env_type: str = 'venv') -> dict:
    """Prepare environment variables for running commands in a virtual environment.

    Args:
        venv_path: Path to the virtual environment
        env_type: Type of virtual environment ('venv' or 'conda')

    Returns:
        dict: Modified environment variables for use with subprocess

    Raises:
        ValueError: If python3 executable is not found in the virtual environment
    """
    env = os.environ.copy()

    bin_dir = os.path.join(venv_path, 'bin')
    python_path = os.path.join(bin_dir, 'python3')
    if not os.path.exists(python_path):
        raise ValueError(f"Invalid virtual environment: python3 executable not found at {python_path}")

    env['PATH'] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop('PYTHONHOME', None)

    if env_type == 'venv' or env_type == 'uv':
        env['VIRTUAL_ENV'] = venv_path
    elif env_type == 'conda':
        env['CONDA_PREFIX'] = venv_path

    return env


def create_experiment_directory(
    workload_dir: str,
    workload_key: str,
    model_size: str,
    dtype: str,
    scale: int,
    max_retries: int = 3,
    description: str | None = None,
) -> str:
    """Create experiment directory with collision handling.

    Args:
        workload_dir: Base workload directory path
        workload_key: Workload identifier
        model_size: Model size (e.g., "671b")
        dtype: Data type (e.g., "bf16")
        scale: Number of GPUs
        max_retries: Maximum retry attempts for timestamp collision
        description: Optional launcher-specific directory description. The
            default preserves the existing GPU-scale naming for all current
            launchers.

    Returns:
        str: Full path to created experiment directory

    Raises:
        RuntimeError: If directory creation fails after max_retries
    """
    desc = description or f"{workload_key}_{model_size}_{dtype}_gpus{scale}"
    experiments_base = os.path.join(workload_dir, "experiments", desc)

    for attempt in range(max_retries):
        timestamp = str(int(time.time()))
        experiment_dir = os.path.join(experiments_base, timestamp)

        try:
            os.makedirs(experiment_dir)
            return experiment_dir
        except FileExistsError:
            # Directory already exists (collision), wait and retry
            if attempt < max_retries - 1:
                logger.debug(f"Directory {experiment_dir} exists, retrying...")
                time.sleep(1)
        except OSError as e:
            # Other OS errors (permission denied, disk full, etc.) - fail immediately
            raise RuntimeError(f"Failed to create experiment directory {experiment_dir}: {e}") from e

    raise RuntimeError(f"Failed to create unique experiment directory after {max_retries} attempts")


def parse_srt_slurm_json_output(output: str) -> dict:
    """Parse the single-submission contract from ``srtctl apply --json``."""
    records = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON from srtctl apply: {line!r}") from e
        if not isinstance(record, dict):
            raise ValueError("Invalid JSON from srtctl apply: expected an object per line.")
        records.append(record)

    if len(records) != 1:
        raise ValueError(f"Expected exactly one SRT-Slurm submission, received {len(records)}.")

    record = records[0]
    if record.get("status") != "submitted":
        detail = record.get("error") or record.get("status") or "unknown error"
        raise ValueError(f"SRT-Slurm submission failed: {detail}")

    missing = [key for key in ("slurm_job_id", "output_dir") if not record.get(key)]
    if missing:
        raise ValueError(f"SRT-Slurm submission response is missing: {', '.join(missing)}")
    return record


class JobLauncher(ABC):
    """Abstract base class for job launchers."""

    def __init__(self, config: ClusterConfig, workloads):
        self.config = config
        self.workloads = workloads

    @abstractmethod
    def launch(self, task) -> SlurmJob:
        """Launch a single task.

        Args:
            task: WorkloadTask to launch

        Returns:
            SlurmJob: SlurmJob object containing, job_workdir may be None in some cases.
        """
        pass

    def get_launch_script(self, task):
        """Return the launch script name from workload metadata, defaulting to 'launch.sh'."""
        metadata = self.workloads[task.workload_key].get('metadata', {})
        return metadata.get('run', {}).get('launch_script', 'launch.sh')

    def get_gpu_type(self):
        """Determine GPU type for a task from cluster config only."""
        return self.config.gpu_type

    def get_gpus_per_node(self, gpu_type: str) -> int:
        """GPUs per node: GPUS_PER_NODE env override, else the GPU-type table."""
        if env_override := os.environ.get('GPUS_PER_NODE'):
            return int(env_override)
        if gpu_type not in GPU_TYPE_TO_NUM_GPUS:
            raise ValueError(
                f"Invalid GPU type specified: '{gpu_type}'. "
                f"Valid types in 'llmb-run' modules are: {', '.join(GPU_TYPE_TO_NUM_GPUS)}"
            )
        return GPU_TYPE_TO_NUM_GPUS[gpu_type]

    def get_nvl_domain(self, gpu_type: str, gpus_per_node: int) -> int:
        """NVLink domain size in GPUs: config value, else per-type default, else one node."""
        if self.config.nvl_domain:
            return self.config.nvl_domain
        return DEFAULT_NVL_DOMAIN_BY_GPU.get(gpu_type, gpus_per_node)

    def get_segment_domain_nodes(self, gpu_type: str, gpus_per_node: int) -> int:
        """Domain size in nodes for --segment auto-detection.

        A segment must fit inside the smallest Slurm block recorded at install
        (on NVL576 that is still one rack); without recorded blocks, fall back
        to the NVLink domain expressed in nodes.
        """
        if self.config.nvl_block_sizes:
            return min(self.config.nvl_block_sizes)
        return self.get_nvl_domain(gpu_type, gpus_per_node) // gpus_per_node

    def get_mbridge_gpu_type(self, gpu_type: str) -> str:
        """GPU type name Megatron-Bridge expects for this LLMB GPU type."""
        return MBRIDGE_GPU_TYPE_MAP.get(gpu_type, gpu_type)

    def resolve_task_slurm_args(self, task, *, workload_config=None) -> SlurmArgs | None:
        """Resolve per-task canonical Slurm args in one place."""
        cli_args = task.slurm_args
        if not cli_args:
            return None

        workload_environment = {}
        if workload_config:
            workload_environment = workload_config.get('environment', {})

        validate_no_additional_slurm_params_conflict(
            cli_args=cli_args,
            cluster_environment=self.config.environment,
            workload_environment=workload_environment,
            task_environment=task.env_overrides,
        )
        return cli_args


class SbatchLauncher(JobLauncher):
    """Launcher for SLURM sbatch jobs.

    TODO: Needs work if used for new sbatch jobs."""

    def launch(self, task):
        """Launch a task using the legacy sbatch method."""
        workload_config = self.config.workload_config(task.workload_key)
        job = {
            "workload": task.workload_key,
            "LLMB_INSTALL": self.config.llmb_install,
            "dir": self.workloads[task.workload_key]["dir"],
            "MODEL_SIZE": task.model_size,
            "DTYPE": task.dtype,
        }

        # We need to pass node counts and ntasks-per-node to sbatch jobs.
        gpu_type = self.get_gpu_type()
        gpus_per_node = self.get_gpus_per_node(gpu_type)
        nvl_domain = self.get_nvl_domain(gpu_type, gpus_per_node)
        # Ensure we have at least one node for jobs that use less than all GPUs on a node.
        num_nodes = (task.scale + gpus_per_node - 1) // gpus_per_node
        num_nodes = str(num_nodes)
        # If the job uses less than all GPUs on a node, set ntasks-per-node to the number of GPUs used.
        ntasks_per_node = str(gpus_per_node) if task.scale >= gpus_per_node else str(task.scale)

        llmb_workload = f"{self.config.llmb_install}/workloads/{job['workload']}"
        try:
            slurm_args = self.resolve_task_slurm_args(task, workload_config=workload_config)
        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)

        cmd = [
            "sbatch",
            "--parsable",
            f"--output={llmb_workload}/{SLURM_OUTPUT_PATTERN}",
            "-J",
            f"{job['workload']}_{job['MODEL_SIZE']}_{job['DTYPE']}",
            "-N",
            num_nodes,
            f"--ntasks-per-node={ntasks_per_node}",
        ]

        if slurm_args:
            cmd.extend(slurm_args.to_sbatch_args())

        cmd.append(self.get_launch_script(task))

        env = os.environ.copy()

        # Set basic job environment variables
        env["LLMB_INSTALL"] = job["LLMB_INSTALL"]
        env["LLMB_REPO"] = self.config.llmb_repo
        env["LLMB_WORKLOAD"] = llmb_workload
        env["LLMB_RECIPE_ROOT"] = job["dir"]
        env["MODEL_SIZE"] = job["MODEL_SIZE"]
        env["DTYPE"] = job["DTYPE"]
        env["JOB_TOTAL_GPUS"] = str(task.scale)
        env["GPU_TYPE"] = gpu_type
        env["NVL_DOMAIN"] = str(nvl_domain)
        env["MBRIDGE_GPU_TYPE"] = self.get_mbridge_gpu_type(gpu_type)

        # Add SLURM environment variables
        slurm_env = self.config.slurm.env()
        env.update(slurm_env)

        if task.profile:
            env['ENABLE_PROFILE'] = 'true'

        # Automatically enable VBoost for 'eos' cluster if not explicitly set
        if (
            should_enable_vboost(self.config)
            and 'ENABLE_VBOOST' not in env
            and 'ENABLE_VBOOST' not in task.env_overrides
        ):
            env['ENABLE_VBOOST'] = 'true'
            logger.debug("Automatically enabled VBoost for 'eos' cluster")

        # Handle environment variables from config
        if self.config.environment:
            env_vars = {k: str(v) for k, v in self.config.environment.items()}
            env.update(env_vars)

        # Convert all task override values to strings
        task_env = {k: str(v) for k, v in task.env_overrides.items()}
        env.update(task_env)
        apply_sbatch_explicit_env_contract(env, task.explicit_env_overrides)

        # Handle model parameter overrides
        if task.model_overrides:
            env.update({k.upper(): str(v) for k, v in task.model_overrides.items()})

        try:
            logger.debug(f"Command: {cmd}")
            result = subprocess.run(cmd, capture_output=True, check=True, text=True, env=env, cwd=job['dir'])
            job_id = parse_slurm_job_id(result.stdout)
            logger.info(format_task_output(task, prefix="SUBMITTED: ", suffix=f"jobid={job_id}"))

            # Create llmb-config.yaml file in the current directory
            config_path = create_llmb_config(task, str(job_id), None, self.config, self.workloads)

            # TODO: This job directory is not correct.
            return SlurmJob(job_id=job_id, job_workdir=None, llmb_config_path=config_path)
        except subprocess.CalledProcessError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"error={e.stderr.strip()}"))
            return SlurmJob(job_id=None, job_workdir=None)
        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)


class ConfiguredSbatchLauncher(JobLauncher):
    """Launcher for SLURM sbatch jobs with managed experiment directories.

    This launcher creates experiment directories before job submission and passes experiment directory path via LLMB_EXPERIMENT_DIR environment variable.
    """

    def resolve_slurm_target(self, task) -> str:
        """'gpu' unless the workload's metadata says `run.requires_gpus: false`, then 'cpu'.

        cluster_config resolves the cpu target to a copy of the gpu one when slurm.cpu is not set,
        so a GPU-less workload lands on the CPU partition when there is one and on the GPU
        partition otherwise; a `--env SBATCH_PARTITION=...` override on the task is applied later."""
        metadata = self.workloads.get(task.workload_key, {}).get('metadata', {})
        return 'gpu' if metadata.get('run', {}).get('requires_gpus', True) else 'cpu'

    def launch(self, task):
        """Launch a task using sbatch with a pre-created experiment directory."""
        workload_config = self.config.workload_config(task.workload_key)
        # Working directory for sbatch (where the launch script lives)
        script_dir = self.workloads[task.workload_key]["dir"]
        # Base directory for experiments (under $LLMB_INSTALL/workloads/)
        llmb_workload = f"{self.config.llmb_install}/workloads/{task.workload_key}"

        # Create experiment directory
        try:
            experiment_dir = create_experiment_directory(
                workload_dir=llmb_workload,
                workload_key=task.workload_key,
                model_size=task.model_size,
                dtype=task.dtype,
                scale=task.scale,
            )
        except RuntimeError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)

        # Build sbatch command
        job_name = f"{task.workload_key}_{task.model_size}_{task.dtype}"

        # Determine node configuration
        gpu_type = self.get_gpu_type()
        gpus_per_node = self.get_gpus_per_node(gpu_type)
        nvl_domain = self.get_nvl_domain(gpu_type, gpus_per_node)

        num_nodes = (task.scale + gpus_per_node - 1) // gpus_per_node
        ntasks_per_node = str(gpus_per_node) if task.scale >= gpus_per_node else str(task.scale)
        try:
            slurm_args = self.resolve_task_slurm_args(task, workload_config=workload_config)
        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)

        cmd = [
            "sbatch",
            "--parsable",
            f"--output={experiment_dir}/slurm-%j.out",
            "-J",
            job_name,
            "-N",
            str(num_nodes),
            f"--ntasks-per-node={ntasks_per_node}",
        ]

        segment_override = slurm_args.get_named_param('segment') if slurm_args else None
        if slurm_args:
            cmd.extend(slurm_args.to_sbatch_args())

        cmd.append(self.get_launch_script(task))

        # Set up environment
        env = os.environ.copy()
        env["LLMB_INSTALL"] = self.config.llmb_install
        env["LLMB_REPO"] = self.config.llmb_repo
        env["LLMB_WORKLOAD"] = llmb_workload
        env["LLMB_RECIPE_ROOT"] = script_dir
        env["LLMB_EXPERIMENT_DIR"] = experiment_dir
        env["MODEL_SIZE"] = task.model_size
        env["DTYPE"] = task.dtype
        env["JOB_TOTAL_GPUS"] = str(task.scale)
        env["GPU_TYPE"] = gpu_type
        env["NVL_DOMAIN"] = str(nvl_domain)
        env["MBRIDGE_GPU_TYPE"] = self.get_mbridge_gpu_type(gpu_type)

        # Add SLURM environment variables (partition, account, GPUs per node)
        slurm_env = self.config.slurm.env(self.resolve_slurm_target(task))
        env.update(slurm_env)

        if task.profile:
            env['ENABLE_PROFILE'] = 'true'

        # Automatically enable VBoost for 'eos' cluster if not explicitly set
        if (
            should_enable_vboost(self.config)
            and 'ENABLE_VBOOST' not in env
            and 'ENABLE_VBOOST' not in task.env_overrides
        ):
            env['ENABLE_VBOOST'] = 'true'
            logger.debug("Automatically enabled VBoost for 'eos' cluster")

        # Handle environment variables from config
        if self.config.environment:
            env_vars = {k: str(v) for k, v in self.config.environment.items()}
            env.update(env_vars)

        # Convert all task override values to strings
        task_env = {k: str(v) for k, v in task.env_overrides.items()}
        env.update(task_env)
        apply_sbatch_explicit_env_contract(env, task.explicit_env_overrides)

        # Handle model parameter overrides
        if task.model_overrides:
            env.update({k.upper(): str(v) for k, v in task.model_overrides.items()})

        # Resolve segment as a --segment flag. CLI override is already in cmd
        # via to_sbatch_args(). Otherwise, check for SBATCH_SEGMENT_SIZE from any
        # env source, or auto-detect when the NVLink domain spans multiple nodes.
        # NOTE: Newer Slurm versions support SBATCH_SEGMENT_SIZE as an env var
        # natively; if we ever drop support for older versions, we could set the
        # env var instead of the flag.
        if segment_override is None:
            env_segment = env.get('SBATCH_SEGMENT_SIZE')
            if env_segment is not None:
                cmd.insert(-1, f"--segment={env_segment}")
            else:
                domain_nodes = self.get_segment_domain_nodes(gpu_type, gpus_per_node)
                if domain_nodes > 1:
                    computed = _compute_segment_size(num_nodes, domain_nodes)
                    cmd.insert(-1, f"--segment={computed}")
                    logger.debug(
                        f"Auto-detected segment size {computed} for {gpu_type} "
                        f"(NVL{nvl_domain}, domain_nodes={domain_nodes})"
                    )

        # Remove SBATCH_SEGMENT_SIZE from the subprocess env — we pass --segment
        # as a flag instead.
        env.pop('SBATCH_SEGMENT_SIZE', None)

        try:
            logger.debug(f"Command: {cmd}")
            result = subprocess.run(cmd, capture_output=True, check=True, text=True, env=env, cwd=script_dir)
            job_id = parse_slurm_job_id(result.stdout)
            logger.info(
                format_task_output(task, prefix="SUBMITTED: ", suffix=f"jobid={job_id} workdir={experiment_dir}")
            )

            # Create llmb-config.yaml file in the experiment directory
            config_path = create_llmb_config(task, str(job_id), experiment_dir, self.config, self.workloads)

            return SlurmJob(job_id=job_id, job_workdir=experiment_dir, llmb_config_path=config_path)
        except subprocess.CalledProcessError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"error={e.stderr.strip()}"))
            return SlurmJob(job_id=None, job_workdir=None)
        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)


class SrtSlurmLauncher(JobLauncher):
    """Submit an LLMB-owned SRT-Slurm recipe without a dispatcher allocation."""

    def _run_config(self, task) -> dict:
        metadata = self.workloads[task.workload_key].get("metadata", {})
        config = metadata.get("run", {}).get("srt_slurm")
        if not isinstance(config, dict):
            raise ValueError("Missing required 'run.srt_slurm' mapping in workload metadata.")
        return config

    @staticmethod
    def _artifact_aliases(metadata: dict, llmb_install: Path, llmb_workload: Path) -> tuple[dict, dict]:
        """Resolve LLMB-provisioned model and container aliases for SRT-Slurm."""
        model_paths: dict[str, str] = {}
        repositories = metadata.get("downloads", {}).get("huggingface", {}).get("repos", []) or []
        for repository in repositories:
            if not isinstance(repository, dict) or repository.get("target") != "workload":
                continue
            alias = repository.get("name")
            if not isinstance(alias, str) or not alias:
                raise ValueError("SRT-Slurm Hugging Face downloads must declare a non-empty 'name' alias.")
            path = llmb_workload / alias
            if not path.exists():
                raise ValueError(f"SRT-Slurm model alias '{alias}' does not exist: {path}")
            model_paths[alias] = str(path)

        containers: dict[str, str] = {}
        images = metadata.get("container", {}).get("images", []) or []
        if not isinstance(images, list):
            raise ValueError("SRT-Slurm container.images must resolve to a list.")
        for image in images:
            if not isinstance(image, dict):
                raise ValueError("SRT-Slurm container images must declare explicit 'name' aliases.")
            alias = image.get("name")
            if not isinstance(alias, str) or not alias:
                raise ValueError("SRT-Slurm container images must declare a non-empty 'name' alias.")
            path = llmb_install / "images" / f"{alias}.sqsh"
            if not path.is_file():
                raise ValueError(f"SRT-Slurm container alias '{alias}' does not exist: {path}")
            containers[alias] = str(path)

        if not model_paths:
            raise ValueError("SRT-Slurm workload metadata declares no workload-local model aliases.")
        if not containers:
            raise ValueError("SRT-Slurm workload metadata declares no container aliases.")
        return model_paths, containers

    def _write_srtslurm_config(self, srt_root: Path, metadata: dict, llmb_workload: Path) -> Path:
        """Refresh workload aliases and current cluster settings atomically at launch time."""
        gpu_env = self.config.slurm.env("gpu")
        account = gpu_env.get("SBATCH_ACCOUNT") or gpu_env.get("SLURM_ACCOUNT")
        partition = self.config.slurm.partition("gpu")
        if not account:
            raise ValueError("Slurm GPU target does not define an account.")
        if not partition:
            raise ValueError("Slurm GPU target does not define a partition.")

        configured_gpus = self.config.slurm.gpus_per_node("gpu")
        if configured_gpus:
            try:
                gpus_per_node = int(configured_gpus)
            except ValueError as e:
                raise ValueError("Slurm GPU target gpus-per-node must be an integer.") from e
            if gpus_per_node <= 0:
                raise ValueError("Slurm GPU target gpus-per-node must be positive.")
        else:
            gpus_per_node = self.get_gpus_per_node(self.get_gpu_type())

        model_paths, containers = self._artifact_aliases(metadata, Path(self.config.llmb_install), llmb_workload)
        config_path = srt_root / "srtslurm.yaml"
        lock_path = srt_root / ".srtslurm.yaml.lock"
        with lock_path.open("a+", encoding="utf-8") as lock_file:
            fcntl.flock(lock_file, fcntl.LOCK_EX)
            try:
                existing = yaml.safe_load(config_path.read_text()) if config_path.exists() else None
            except (OSError, yaml.YAMLError) as e:
                raise ValueError(f"Unable to read SRT-Slurm configuration {config_path}: {e}") from e
            if existing is None:
                config: dict = {}
            elif isinstance(existing, dict):
                config = dict(existing)
            else:
                raise ValueError(f"SRT-Slurm configuration must contain a mapping: {config_path}")

            merged_models = dict(config.get("model_paths") or {})
            merged_models.update(model_paths)
            merged_containers = dict(config.get("containers") or {})
            merged_containers.update(containers)
            config.update(
                {
                    "default_account": account,
                    "default_partition": partition,
                    "gpus_per_node": gpus_per_node,
                    "srtctl_root": str(srt_root),
                    "model_paths": merged_models,
                    "containers": merged_containers,
                }
            )
            if configured_gpus:
                config.setdefault("use_gpus_per_node_directive", True)
            else:
                config["use_gpus_per_node_directive"] = False
            config.setdefault("default_time_limit", "04:00:00")
            config.setdefault("network_interface", "")
            config.setdefault("use_segment_sbatch_directive", False)
            config.setdefault("use_exclusive_sbatch_directive", True)

            fd, temporary_name = tempfile.mkstemp(prefix=".srtslurm.", suffix=".yaml", dir=srt_root)
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as stream:
                    yaml.safe_dump(config, stream, sort_keys=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary_name, config_path)
            finally:
                if os.path.exists(temporary_name):
                    os.unlink(temporary_name)
        return config_path

    @staticmethod
    def _resolve_relative_path(base: Path, value: str, field: str) -> Path:
        candidate = Path(value)
        if candidate.is_absolute():
            raise ValueError(f"'{field}' must be relative to the installed workload directory.")
        resolved_base = base.resolve()
        resolved = (resolved_base / candidate).resolve()
        try:
            resolved.relative_to(resolved_base)
        except ValueError as e:
            raise ValueError(f"'{field}' must stay within {resolved_base}.") from e
        return resolved

    @staticmethod
    def _recipe_gpu_count(recipe_path: Path, selector: str | None = None) -> int:
        """Return the physical GPU count requested by an SRT-Slurm recipe."""
        try:
            document = yaml.safe_load(recipe_path.read_text())
        except (OSError, yaml.YAMLError) as e:
            raise ValueError(f"Unable to read SRT-Slurm recipe topology from {recipe_path}: {e}") from e
        if not isinstance(document, dict):
            raise ValueError(f"SRT-Slurm recipe is not a mapping: {recipe_path}")

        base = document.get("base")
        if base is not None and not isinstance(base, dict):
            raise ValueError(f"SRT-Slurm recipe 'base' must be a mapping: {recipe_path}")
        recipe = base if isinstance(base, dict) else document
        resources = recipe.get("resources")
        if not isinstance(resources, dict):
            raise ValueError(f"SRT-Slurm recipe has no valid 'resources' mapping: {recipe_path}")
        resources = dict(resources)

        # SRT selectors are overlays on base. Account for any resource values
        # changed by the selected overlay before validating LLMB --scale.
        if selector:
            if base is None:
                raise ValueError(f"SRT-Slurm recipe selector '{selector}' requires a recipe with a 'base' mapping.")
            overlay = document.get(selector)
            if not isinstance(overlay, dict):
                raise ValueError(f"SRT-Slurm recipe selector '{selector}' is missing or is not a mapping.")
            overlay_resources = overlay.get("resources")
            if overlay_resources is not None:
                if not isinstance(overlay_resources, dict):
                    raise ValueError(f"SRT-Slurm recipe selector '{selector}.resources' must be a mapping.")
                resources.update(overlay_resources)

        def positive_int(name: str) -> int:
            value = resources.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"SRT-Slurm recipe resource '{name}' must be a positive integer in {recipe_path}.")
            return value

        gpus_per_node = positive_int("gpus_per_node")
        if any(name in resources for name in ("prefill_nodes", "decode_nodes", "agg_nodes")):
            nodes = sum(
                positive_int(name) for name in ("prefill_nodes", "decode_nodes", "agg_nodes") if name in resources
            )
        else:
            nodes = positive_int("nodes")
        return nodes * gpus_per_node

    @classmethod
    def _validate_recipe_scale(cls, recipe_path: Path, selector: str | None, scale: int) -> None:
        recipe_gpu_count = cls._recipe_gpu_count(recipe_path, selector)
        if scale != recipe_gpu_count:
            raise ValueError(
                f"Selected SRT-Slurm recipe requests {recipe_gpu_count} GPUs, "
                f"but LLMB --scale is {scale}. Submit this recipe with "
                f"'--scale {recipe_gpu_count}'."
            )

    def launch(self, task):
        llmb_workload = Path(self.config.llmb_install) / "workloads" / task.workload_key
        workload_recipe_dir = Path(self.workloads[task.workload_key]["dir"])
        workload_config = self.config.workload_config(task.workload_key)
        metadata = self.workloads[task.workload_key].get("metadata", {})

        try:
            srt_config = self._run_config(task)
            srt_root = llmb_workload / "srt-slurm"

            recipe_value = srt_config.get("recipe")
            recipe_env = srt_config.get("recipe_env")
            selector_value = srt_config.get("selector")

            env = os.environ.copy()
            env.update(self.config.slurm.env())
            if self.config.environment:
                env.update({key: str(value) for key, value in self.config.environment.items()})
            env.update({key: str(value) for key, value in workload_config.get("environment", {}).items()})
            env.update({key: str(value) for key, value in task.env_overrides.items()})
            apply_sbatch_explicit_env_contract(env, task.explicit_env_overrides)

            if recipe_env and env.get(str(recipe_env)):
                recipe_value = env[str(recipe_env)]
            if not recipe_value:
                raise ValueError("Missing required 'run.srt_slurm.recipe' metadata value.")

            recipe_path = self._resolve_relative_path(workload_recipe_dir, str(recipe_value), "run.srt_slurm.recipe")
            if not recipe_path.is_file():
                raise ValueError(f"SRT-Slurm recipe does not exist: {recipe_path}")
            recipe_argument = str(recipe_path)
            if selector_value:
                selector = str(selector_value)
                if not re.fullmatch(r"[A-Za-z0-9_.-]+", selector):
                    raise ValueError("'run.srt_slurm.selector' contains unsupported characters.")
                recipe_argument = f"{recipe_argument}:{selector}"

            self._validate_recipe_scale(
                recipe_path,
                str(selector_value) if selector_value else None,
                task.scale,
            )

            venv_path = workload_config.get("venv_path")
            if not venv_path:
                raise ValueError(f"venv_path not found in config for workload {task.workload_key}")
            srtctl = Path(venv_path) / "bin" / "srtctl"
            if not srtctl.is_file() or not os.access(srtctl, os.X_OK):
                raise ValueError(f"Pinned SRT-Slurm executable not found or not executable: {srtctl}")
            if not srt_root.is_dir():
                raise ValueError(f"Pinned SRT-Slurm checkout does not exist: {srt_root}")
            srtslurm_config = self._write_srtslurm_config(srt_root, metadata, llmb_workload)

            experiment_dir = Path(
                create_experiment_directory(
                    workload_dir=str(llmb_workload),
                    workload_key=task.workload_key,
                    model_size=task.model_size,
                    dtype=task.dtype,
                    scale=task.scale,
                    description=(
                        f"{task.workload_key}_{task.model_size}_{task.dtype}_recipe_"
                        f"{re.sub(r'[^A-Za-z0-9_.-]+', '-', recipe_path.stem).strip('-')}"
                    ),
                )
            )
            env.update(
                {
                    "LLMB_INSTALL": self.config.llmb_install,
                    "LLMB_REPO": self.config.llmb_repo,
                    "LLMB_WORKLOAD": str(llmb_workload),
                    "LLMB_RECIPE_ROOT": str(workload_recipe_dir),
                    "LLMB_EXPERIMENT_DIR": str(experiment_dir),
                    "MODEL_SIZE": task.model_size,
                    "DTYPE": task.dtype,
                    "JOB_TOTAL_GPUS": str(task.scale),
                    "GPU_TYPE": self.get_gpu_type(),
                    "SRTSLURM_CONFIG": str(srtslurm_config),
                }
            )
            if task.model_overrides:
                env.update({key.upper(): str(value) for key, value in task.model_overrides.items()})

            # srtctl itself is invoked by absolute path below. Do not propagate
            # a login-host Python environment into jobs that may run on a
            # different CPU architecture.
            env = sanitize_srt_submission_environment(env)

            command = [
                str(srtctl),
                "apply",
                "-f",
                recipe_argument,
                "-o",
                str(experiment_dir),
                "-y",
                "--json",
            ]
            logger.debug("Command: %s", command)
            result = subprocess.run(
                command,
                cwd=srt_root,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                error_detail = result.stderr.strip() or "no error output"
                raise ValueError(f"srtctl apply exited with status {result.returncode}: {error_detail}")
            if result.stderr.strip():
                logger.info("SRT-Slurm: %s", result.stderr.strip())
            record = parse_srt_slurm_json_output(result.stdout)

            job_id = parse_slurm_job_id(record["slurm_job_id"])
            resolved_output = Path(record["output_dir"]).resolve()
            try:
                resolved_output.relative_to(experiment_dir.resolve())
            except ValueError as e:
                raise ValueError(
                    f"SRT-Slurm returned an output directory outside the LLMB experiment: {resolved_output}"
                ) from e
            output_dir = str(resolved_output)
            config_path = create_llmb_config(task, str(job_id), output_dir, self.config, self.workloads)
            if not config_path:
                logger.warning(
                    "SRT-Slurm job %s was submitted, but LLMB config generation failed; "
                    "the job will still be recorded without a config path.",
                    job_id,
                )
            logger.info(format_task_output(task, prefix="SUBMITTED: ", suffix=f"jobid={job_id} workdir={output_dir}"))
            return SlurmJob(job_id=job_id, job_workdir=output_dir, llmb_config_path=config_path)
        except (OSError, ValueError) as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)


class Nemo2Launcher(JobLauncher):
    """Launcher for Nemo2 jobs with virtual environment support."""

    def launch(self, task):
        """Launch a task using the Nemo2 method with venv activation."""
        workload_config = self.config.workload_config(task.workload_key)
        venv_path = workload_config.get('venv_path')
        venv_type = workload_config.get('venv_type')

        if not venv_path:
            logger.error(f"venv_path not found in config for workload {task.workload_key}")
            return SlurmJob(job_id=None, job_workdir=None)

        if not venv_type:
            logger.error(f"venv_type not found in config for workload {task.workload_key}")
            return SlurmJob(job_id=None, job_workdir=None)

        try:
            slurm_args = self.resolve_task_slurm_args(task, workload_config=workload_config)
        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
            return SlurmJob(job_id=None, job_workdir=None)

        try:
            workload_metadata = self.workloads[task.workload_key].get('metadata', {})
            launcher_type = workload_metadata.get('run', {}).get('launcher_type')

            # Get venv environment with the correct type
            env = get_venv_environment(venv_path, venv_type)

            # Set basic job environment variables
            gpu_type = self.get_gpu_type()
            gpus_per_node = self.get_gpus_per_node(gpu_type)
            env["LLMB_INSTALL"] = self.config.llmb_install
            env["LLMB_REPO"] = self.config.llmb_repo
            env["LLMB_WORKLOAD"] = f"{self.config.llmb_install}/workloads/{task.workload_key}"
            env["LLMB_RECIPE_ROOT"] = self.workloads[task.workload_key]["dir"]
            env["MODEL_SIZE"] = task.model_size
            env["DTYPE"] = task.dtype
            env["JOB_TOTAL_GPUS"] = str(task.scale)
            env["GPU_TYPE"] = gpu_type
            env["NVL_DOMAIN"] = str(self.get_nvl_domain(gpu_type, gpus_per_node))
            env["MBRIDGE_GPU_TYPE"] = self.get_mbridge_gpu_type(gpu_type)

            # Add SLURM environment variables
            slurm_env = self.config.slurm.env()
            env.update(slurm_env)

            if task.profile:
                env['ENABLE_PROFILE'] = 'true'
                if env.get('ENABLE_GPU_METRICS', 'false').lower() == 'true':
                    env['GPU_METRICS_NODES'] = os.getenv('GPU_METRICS_NODES', '0')

            if task.proxy:
                env['PROXY_WORKLOAD'] = 'true'

            # Automatically enable VBoost for 'eos' cluster if not explicitly set
            if (
                should_enable_vboost(self.config)
                and 'ENABLE_VBOOST' not in env
                and 'ENABLE_VBOOST' not in task.env_overrides
            ):
                env['ENABLE_VBOOST'] = 'true'
                logger.debug("Automatically enabled VBoost for 'eos' cluster")

            # Handle environment variables from config
            if self.config.environment:
                env_vars = {k: str(v) for k, v in self.config.environment.items()}
                env.update(env_vars)

            # Handle workload-specific environment variables
            workload_env = workload_config.get('environment', {})
            workload_env_str = {k: str(v) for k, v in workload_env.items()}
            env.update(workload_env_str)

            # Convert all task override values to strings
            task_env = {k: str(v) for k, v in task.env_overrides.items()}
            env.update(task_env)
            if launcher_type == 'megatron_bridge':
                # Dual-write established workload and environment arguments until
                # MBridge recipes no longer require CONFIG_OVERRIDES.
                apply_nemo_workload_args_contract(env, task.extra_workload_args)
                apply_mbridge_extra_args_contract(
                    env,
                    task.mbridge_args,
                    compatibility_args=task.extra_workload_args,
                    env_overrides=task.explicit_env_overrides,
                )
            apply_nemo_explicit_env_contract(env, task.explicit_env_overrides)

            # Handle model parameter overrides
            if task.model_overrides:
                env.update({k.upper(): str(v) for k, v in task.model_overrides.items()})

            if slurm_args:
                env[ADDITIONAL_SLURM_PARAMS_KEY] = slurm_args.to_additional_slurm_params()

            # Handle custom tool mounts if needed (workarounds for container bugs during profiling)
            try:
                tool_mounts = get_tool_mounts(
                    llmb_install=self.config.llmb_install,
                    workload_metadata=workload_metadata,
                    gpu_type=self.get_gpu_type(),
                    arch=self.config.install.node_architecture,
                    profiling_enabled=task.profile,
                )

                if tool_mounts:
                    # Append to RUN_CONF_MOUNTS if it exists, otherwise create it
                    if 'RUN_CONF_MOUNTS' in env:
                        env['RUN_CONF_MOUNTS'] += ',' + tool_mounts
                    else:
                        env['RUN_CONF_MOUNTS'] = tool_mounts
            except FileNotFoundError as e:
                # Fatal error: required tool version missing for profiling
                logger.error(format_task_output(task, suffix=f"{e}"))
                return SlurmJob(job_id=None, job_workdir=None)
            except Exception as e:
                # Log but don't fail for unexpected errors in mount handling
                logger.warning(format_task_output(task, prefix="WARNING: ", suffix=f"nsys mount handler error: {e}"))

            # Run launch script in the workload directory
            workload_dir = self.workloads[task.workload_key]["dir"]
            launch_script = self.get_launch_script(task)

            logger.debug(f"Running {launch_script} in {workload_dir} with venv {venv_path} (type: {venv_type})")

            # Create spinner with task info
            spinner_text = (
                f"Launching {task.workload_key} (model_size={task.model_size}, dtype={task.dtype}, scale={task.scale})"
            )

            job_id = None
            local_dir = None
            parse_error = None

            with console.status(spinner_text, spinner="bouncingBall"):
                result = subprocess.run(
                    [f"./{launch_script}"], env=env, cwd=workload_dir, capture_output=True, text=True
                )

                # Parse output for job ID and local directory
                job_id, local_dir, parse_error = parse_nemo2_output(result.stdout)

            # Display the filtered output after spinner completes
            console.print("\n[bold green]Launch Output:[/bold green]")
            filtered_output = filter_nemo2_status_instructions(result.stdout)
            console.print(filtered_output)

            if result.returncode == 0:
                if parse_error:
                    console.print(f"\n[bold red]Parse Error:[/bold red] {parse_error}")
                    logger.error(
                        format_task_output(task, prefix="ERROR: ", suffix=f"output parsing failed: {parse_error}")
                    )
                    return SlurmJob(job_id=None, job_workdir=None)
                else:
                    try:
                        parsed_job_id = parse_slurm_job_id(job_id)
                    except ValueError as e:
                        logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
                        return SlurmJob(job_id=None, job_workdir=None)
                    logger.info(format_task_output(task, prefix="LAUNCHED: ", suffix=f"jobid={job_id}"))
                    logger.info(f"JobID: {job_id}, Workdir: {local_dir}")

                    # Create llmb-config.yaml file in the experiment directory
                    config_path = create_llmb_config(task, str(parsed_job_id), local_dir, self.config, self.workloads)

                    return SlurmJob(job_id=parsed_job_id, job_workdir=local_dir, llmb_config_path=config_path)
            else:
                console.print("\n[bold red]Launch Error:[/bold red]")
                console.print(result.stderr)
                logger.error(
                    format_task_output(
                        task, prefix="ERROR: ", suffix=f"launch failed with return code {result.returncode}"
                    )
                )
                return SlurmJob(job_id=None, job_workdir=None)

        except ValueError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"venv error: {e}"))
            return SlurmJob(job_id=None, job_workdir=None)
        except subprocess.CalledProcessError as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"launch error: {e}"))
            return SlurmJob(job_id=None, job_workdir=None)
        except Exception as e:
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"unexpected error: {e}"))
            return SlurmJob(job_id=None, job_workdir=None)


_synthetic_id_lock = threading.Lock()
_last_synthetic_id = 0


def _synthetic_job_id() -> int:
    """A unique numeric id for a launch that has no Slurm job id (nvcre).

    Microsecond resolution, and strictly increasing within this process, so two launches
    finishing in the same second cannot overwrite each other's job-history row.
    """
    global _last_synthetic_id
    with _synthetic_id_lock:
        _last_synthetic_id = max(time.time_ns() // 1000, _last_synthetic_id + 1)
        return _last_synthetic_id


class NVCreLauncher(JobLauncher):
    """Launcher for Kubernetes/nvcre workloads.

    Routes tasks through the workload's launch.sh exactly like Nemo2Launcher,
    but injects CLUSTER_TYPE=kubernetes and NVCRE_* env vars (sourced from
    the cluster config's ``kubernetes:`` block) instead of SLURM env vars.
    The launch.sh detects CLUSTER_TYPE and calls setup_experiment.py with
    --nvcre_* flags rather than --account/--partition.
    """

    def launch(self, task) -> SlurmJob:
        workload_config = self.config.workload_config(task.workload_key)
        venv_path = workload_config.get('venv_path')
        venv_type = workload_config.get('venv_type')

        if not venv_path:
            logger.error(f"venv_path not found in config for workload {task.workload_key}")
            return SlurmJob(job_id=None, job_workdir=None)

        if not venv_type:
            logger.error(f"venv_type not found in config for workload {task.workload_key}")
            return SlurmJob(job_id=None, job_workdir=None)

        # Same venv activation as Nemo2Launcher: setup_experiment.py runs on the
        # launch node from the workload's venv (created by the standard install
        # flow), so launch.sh's plain `python3` resolves inside it.
        env = get_venv_environment(venv_path, venv_type)

        # Standard job env vars (same as other launchers)
        env['LLMB_INSTALL'] = self.config.llmb_install
        env['MODEL_SIZE'] = task.model_size
        env['DTYPE'] = task.dtype
        env['JOB_TOTAL_GPUS'] = str(task.scale)
        env['GPU_TYPE'] = self.get_gpu_type()

        # launch.sh reads these to apply the profiling and reduced-size proxy overrides, as on Slurm.
        if task.profile:
            env['ENABLE_PROFILE'] = 'true'
        if task.proxy:
            env['PROXY_WORKLOAD'] = 'true'

        # Signal to launch.sh which executor path to take
        env['CLUSTER_TYPE'] = 'kubernetes'

        # Override the image so launch.sh uses an OCI registry reference instead
        # of the default Slurm .sqsh path.  Metadata stores images as
        # "nvcr.io#org/image:tag" (# replaces the first / to avoid path issues).
        workload_metadata = self.workloads[task.workload_key].get('metadata', {})
        # `container.images` is a list, or a by_gpu mapping ({default: [...], h100: [...]}); resolve it
        # for this cluster's GPU type instead of indexing it as a list.
        container_images = resolve_container_images(
            workload_metadata.get('container', {}).get('images', []), self.get_gpu_type()
        )
        if container_images and 'RUN_CONF_IMAGE' not in env:
            env['RUN_CONF_IMAGE'] = container_images[0].replace('#', '/', 1)

        # nvcre-specific vars — read from cluster_config.yaml kubernetes: block
        k8s = self.config.extras.get('kubernetes') or {}
        env['NVCRE_NAMESPACE'] = str(k8s.get('namespace', 'default'))
        pull_secrets = k8s.get('image_pull_secrets') or []
        env['NVCRE_IMAGE_PULL_SECRET'] = str(pull_secrets[0]) if pull_secrets else ''
        env['NVCRE_WORKDIR_PVC'] = str(k8s.get('workdir_pvc') or '')
        env['NVCRE_NODE_SELECTOR_JSON'] = str(k8s.get('node_selector_json') or '')
        env['NVCRE_TIMEOUT_PER_JOB'] = str(k8s.get('timeout_per_job') or '12h')
        if k8s.get('gang_scheduler_name'):
            env['NVCRE_GANG_SCHEDULER_NAME'] = str(k8s['gang_scheduler_name'])
        if k8s.get('hf_token_secret_name'):
            env['NVCRE_HF_TOKEN_SECRET_NAME'] = str(k8s['hf_token_secret_name'])
            env['NVCRE_HF_TOKEN_SECRET_KEY'] = str(k8s.get('hf_token_secret_key') or 'token')

        # Cluster-level environment overrides
        if self.config.environment:
            env.update({k: str(v) for k, v in self.config.environment.items()})

        # Per-workload environment
        workload_env = workload_config.get('environment', {})
        env.update({k: str(v) for k, v in workload_env.items()})

        # Task-level overrides
        env.update({k: str(v) for k, v in task.env_overrides.items()})
        workload_metadata = self.workloads[task.workload_key].get('metadata', {})
        launcher_type = workload_metadata.get('run', {}).get('launcher_type')
        if launcher_type == 'megatron_bridge':
            apply_nemo_workload_args_contract(env, task.extra_workload_args)
            apply_mbridge_extra_args_contract(
                env,
                task.mbridge_args,
                compatibility_args=task.extra_workload_args,
                env_overrides=task.explicit_env_overrides,
            )
        apply_nemo_explicit_env_contract(env, task.explicit_env_overrides)

        if task.model_overrides:
            env.update({k.upper(): str(v) for k, v in task.model_overrides.items()})

        workload_dir = self.workloads[task.workload_key]['dir']
        launch_script = self.get_launch_script(task)

        spinner_text = (
            f"Launching {task.workload_key} (model_size={task.model_size}, "
            f"dtype={task.dtype}, scale={task.scale}) via nvcre"
        )

        job_id = None
        local_dir = None
        parse_error = None

        from datetime import datetime as _datetime

        _submit_time = _datetime.now().isoformat(timespec='seconds')

        with console.status(spinner_text, spinner="bouncingBall"):
            result = subprocess.run([f'./{launch_script}'], env=env, cwd=workload_dir, capture_output=True, text=True)
            job_id, local_dir, parse_error = parse_nemo2_output(result.stdout)

        console.print("\n[bold green]Launch Output:[/bold green]")
        filtered_output = filter_nemo2_status_instructions(result.stdout)
        console.print(filtered_output)

        if result.returncode != 0:
            console.print("\n[bold red]Launch Error:[/bold red]")
            console.print(result.stderr)
            logger.error(
                format_task_output(task, prefix="ERROR: ", suffix=f"launch failed with return code {result.returncode}")
            )
            return SlurmJob(job_id=None, job_workdir=None)

        if parse_error:
            console.print(f"\n[bold red]Parse Error:[/bold red] {parse_error}")
            logger.error(format_task_output(task, prefix="ERROR: ", suffix=f"output parsing failed: {parse_error}"))
            return SlurmJob(job_id=None, job_workdir=None)

        # nvcre jobs do not have a numeric Slurm job ID — the NeMo-Run
        # app handle is a string like "nvcre://nvcre/<app_id>".
        # parse_nemo2_output therefore returns job_id=None even on success.
        # When we have a valid local_dir but no numeric ID, synthesise one
        # from the submission timestamp so downstream config naming works.
        if job_id is None and local_dir is not None:
            parsed_job_id = _synthetic_job_id()
            logger.info(
                format_task_output(
                    task,
                    prefix="LAUNCHED: ",
                    suffix=f"nvcre workload submitted (synthetic id={parsed_job_id})",
                )
            )
        else:
            try:
                parsed_job_id = parse_slurm_job_id(job_id)
            except ValueError as e:
                logger.error(format_task_output(task, prefix="ERROR: ", suffix=str(e)))
                return SlurmJob(job_id=None, job_workdir=None)
            logger.info(format_task_output(task, prefix="LAUNCHED: ", suffix=f"jobid={job_id}"))

        config_path = create_llmb_config(task, str(parsed_job_id), local_dir, self.config, self.workloads)
        return SlurmJob(
            job_id=parsed_job_id, job_workdir=local_dir, llmb_config_path=config_path, submit_time=_submit_time
        )


def create_launcher(launcher_type, config, workloads):
    """Factory function to create the appropriate launcher."""
    if launcher_type == 'sbatch':
        return SbatchLauncher(config, workloads)
    elif launcher_type == 'configured_sbatch':
        return ConfiguredSbatchLauncher(config, workloads)
    elif launcher_type == 'srt_slurm':
        return SrtSlurmLauncher(config, workloads)
    elif launcher_type == 'nemo':
        return Nemo2Launcher(config, workloads)
    elif launcher_type == 'megatron_bridge':
        # Megatron Bridge can reuse the existing Nemo2Launcher
        return Nemo2Launcher(config, workloads)
    elif launcher_type == 'nvcre':
        return NVCreLauncher(config, workloads)
    else:
        raise ValueError(f"Unknown launcher type: {launcher_type}")


def _login_for_uploads(config, *, pp_upload=False, upload_env=None):
    """Sign in to whatever the configured upload backend needs, if anything.

    Called once per invocation, before any task is launched, so a bulk
    submission prompts once rather than per task, and a failed login stops
    the run before jobs are queued that could not upload.

    `pp_upload`/`upload_env` come from the CLI's `--pp-upload`/`--pp-process`/
    `--upload-env` flags, folded into `config` before the login check so that
    `--pp-upload` alone is enough to select the vault_ssa backend, without a
    `cluster_config.yaml` edit, wherever this module has built-in defaults.

    Vault being unreachable is the one failure that does not stop the run:
    `vault_ssa` already retries it, so this just logs a warning and returns
    `None`, letting the training job submit without an upload job. Any other
    failure (bad config, incomplete login, permission denied) still stops the run.
    """
    try:
        auth_module = importlib.import_module("llmb_run.internal.vault_ssa")
    except ModuleNotFoundError as e:
        if e.name != "llmb_run.internal.vault_ssa":
            # vault_ssa.py exists but one of its own imports is missing: a real
            # packaging problem, not "internal extensions aren't installed" — must
            # not be silently treated as "no login needed".
            raise
        return None  # Internal extensions unavailable – safe to ignore.

    auth_module.apply_cli_overrides(config, pp_upload=pp_upload, upload_env=upload_env)
    try:
        return auth_module.login_if_required(config)
    except Exception as e:
        # `VaultUnreachableError` is looked up on the (dynamically imported) module
        # rather than a static import, matching how this function already treats
        # `vault_ssa` as an optional internal extension elsewhere.
        unreachable_error = getattr(auth_module, 'VaultUnreachableError', None)
        if unreachable_error is not None and isinstance(e, unreachable_error):
            logger.warning(f"Could not reach Vault for the upload login; submitting without an upload job: {e}")
            return None
        raise


def _has_post_processing_eligible_task(task_list, workloads) -> bool:
    """Whether any task in `task_list` would actually reach the post-processing pipeline.

    Mirrors the pipeline's own eligibility checks so `run_tests` can decide,
    before signing in for uploads, whether anything it queues could use that
    session — an invocation of only ineligible tasks (e.g. all
    inference/microbenchmark) should not force an interactive Vault login.
    Treated as ineligible, not an error, when the pipeline module is unavailable.
    This runs ahead of `run_tests`'s per-task try/except, so a malformed workload's
    metadata is caught here too — one bad task must not abort the whole submission
    before anything launches.
    """
    try:
        pipeline = importlib.import_module("llmb_run.internal.post_processing_pipeline")
    except ModuleNotFoundError:
        return False

    for task in task_list:
        metadata = workloads.get(task.workload_key, {}).get('metadata')
        if not metadata:
            continue
        launcher_type = metadata.get('run', {}).get('launcher_type')
        if launcher_type not in pipeline.SUPPORTED_PP_LAUNCHERS:
            continue
        try:
            if pipeline.workload_supports_post_processing(metadata):
                return True
        except Exception as e:
            logger.warning(f"Could not check post-processing eligibility for {task.workload_key}: {e}")
    return False


def _effective_pp_upload(pp_process: bool, pp_upload: bool) -> bool:
    """`--pp-process` implies `--pp-upload`; the reverse does not hold."""
    return pp_upload or pp_process


def run_tests(config, task_list, workloads, *, pp_process=False, pp_upload=False, upload_env=None):
    """Run tests using the appropriate launcher for each task.

    Without explicit post-processing flags, the existing results parsing and rclone
    upload flow remains enabled. `pp_process` selects full Vault/SSA post-processing
    and implies `pp_upload`; `pp_upload` alone queues uploader jobs without results
    parsing.
    When the cluster config has ``execution_target: kubernetes``,
    every task is routed through NVCreLauncher regardless of the workload's
    own ``run.launcher_type``. A workload may only run on the cluster's
    ``execution_target`` when it declares that target in ``run.execution_targets``
    (unset means slurm only); the rest fail up front instead of submitting a
    Slurm-shaped command to the cluster.
    """
    launchers = {}
    failed_tasks = []
    pp_upload = _effective_pp_upload(pp_process, pp_upload)
    upload_auth = None
    if pp_upload and _has_post_processing_eligible_task(task_list, workloads):
        upload_auth = _login_for_uploads(config, pp_upload=pp_upload, upload_env=upload_env)

    # Cluster-level override: if execution_target is kubernetes, every task
    # uses NVCreLauncher regardless of per-workload launcher_type.
    cluster_override = 'nvcre' if config.execution_target == ExecutionTarget.KUBERNETES else None

    for task in task_list:
        # Get launcher type from workload metadata
        metadata = workloads[task.workload_key]['metadata']
        workload_run = RunConfig.model_validate(metadata.get('run', {}))
        if not workload_run.supports(config.execution_target, config.gpu_type):
            supported_targets = workload_run.targets_for(config.gpu_type)
            logger.error(
                f"Workload {task.workload_key} does not support execution target "
                f"'{config.execution_target}' on GPU {config.gpu_type} "
                f"(supported: {[target.value for target in supported_targets]})."
            )
            failed_tasks.append(task)
            continue
        launcher_type = cluster_override or metadata.get('run', {}).get('launcher_type')
        if not launcher_type:
            logger.error(f"Missing required 'run.launcher_type' field in metadata for workload {task.workload_key}")
            failed_tasks.append(task)
            continue

        # Create launcher if not already created
        if launcher_type not in launchers:
            try:
                launchers[launcher_type] = create_launcher(launcher_type, config, workloads)
            except ValueError as e:
                logger.error(f"Failed to create launcher for {task.workload_key}: {e}")
                failed_tasks.append(task)
                continue

        # Launch the task
        slurm_job = launchers[launcher_type].launch(task)

        if not slurm_job.job_id:
            failed_tasks.append(task)
            continue

        record_job_submission(config, task, slurm_job, workloads)

        # Run post-processing pipeline (resparse, upload, workload inspector).
        # nvcre jobs run synchronously and have no Slurm dependency mechanism,
        # so the Slurm-based pipeline does not apply.
        if slurm_job.job_id and slurm_job.job_workdir and launcher_type != 'nvcre':
            try:
                pipeline = importlib.import_module("llmb_run.internal.post_processing_pipeline")

                results = pipeline.run_post_processing_pipeline(
                    config=config,
                    launcher_type=launcher_type,
                    slurm_job=slurm_job,
                    task=task,
                    workload_metadata=workloads[task.workload_key].get('metadata'),
                    upload_auth=upload_auth,
                    pp_process=pp_process,
                    pp_upload=pp_upload,
                )

                # Log any errors
                if results.has_errors:
                    for error in results.errors:
                        logger.warning(f"Post-processing pipeline error: {error}")

            except ModuleNotFoundError:
                # Internal extensions unavailable – safe to ignore.
                pass
            except Exception as e:
                logger.warning(f"Post-processing pipeline failed: {e}")

    if failed_tasks:
        raise RuntimeError(f"Failed to launch {len(failed_tasks)} out of {len(task_list)} tasks.")
