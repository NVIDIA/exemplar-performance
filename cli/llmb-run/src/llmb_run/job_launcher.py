# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Job launching functionality for different launcher types."""

import importlib
import logging
import os
import re
import subprocess
import time
from abc import ABC, abstractmethod

from rich.console import Console

from llmb_run.config_manager import ClusterConfig
from llmb_run.constants import (
    GPU_TYPE_TO_NUM_GPUS,
    NVLINK_DOMAIN_SIZE,
    SLURM_OUTPUT_PATTERN,
)
from llmb_run.env_args import (
    apply_mbridge_extra_args_contract,
    apply_nemo_explicit_env_contract,
    apply_nemo_workload_args_contract,
    apply_sbatch_explicit_env_contract,
)
from llmb_run.job_history import record_job_submission
from llmb_run.nsys_mount_handler import get_tool_mounts
from llmb_run.run_config import create_llmb_config
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


def _compute_segment_size(num_nodes: int) -> int:
    """Compute SBATCH_SEGMENT_SIZE for GB200/GB300 NVLink systems.

    Returns the greatest divisor of num_nodes that is <= NVLINK_DOMAIN_SIZE.
    For powers of 2 this is min(num_nodes, 16); for multiples of 18 it is 18.
    """
    for d in range(min(num_nodes, NVLINK_DOMAIN_SIZE), 0, -1):
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
    workload_dir: str, workload_key: str, model_size: str, dtype: str, scale: int, max_retries: int = 3
) -> str:
    """Create experiment directory with collision handling.

    Args:
        workload_dir: Base workload directory path
        workload_key: Workload identifier
        model_size: Model size (e.g., "671b")
        dtype: Data type (e.g., "bf16")
        scale: Number of GPUs
        max_retries: Maximum retry attempts for timestamp collision

    Returns:
        str: Full path to created experiment directory

    Raises:
        RuntimeError: If directory creation fails after max_retries
    """
    desc = f"{workload_key}_{model_size}_{dtype}_gpus{scale}"
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
        if os.environ.get('GPUS_PER_NODE'):
            gpus_per_node = int(os.environ.get('GPUS_PER_NODE'))
        else:
            gpu_type = self.get_gpu_type()
            if gpu_type not in GPU_TYPE_TO_NUM_GPUS:
                raise ValueError(
                    f"Invalid GPU type specified: '{gpu_type}'. Valid types in 'llmb-run' modules are: {', '.join(GPU_TYPE_TO_NUM_GPUS.keys())}"
                )
            gpus_per_node = GPU_TYPE_TO_NUM_GPUS[gpu_type]
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
        env["MODEL_SIZE"] = job["MODEL_SIZE"]
        env["DTYPE"] = job["DTYPE"]
        env["JOB_TOTAL_GPUS"] = str(task.scale)
        env["GPU_TYPE"] = self.get_gpu_type()

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
        if os.environ.get('GPUS_PER_NODE'):
            gpus_per_node = int(os.environ.get('GPUS_PER_NODE'))
        else:
            if gpu_type not in GPU_TYPE_TO_NUM_GPUS:
                raise ValueError(
                    f"Invalid GPU type specified: '{gpu_type}'. Valid types in 'llmb-run' modules are: {', '.join(GPU_TYPE_TO_NUM_GPUS.keys())}"
                )
            gpus_per_node = GPU_TYPE_TO_NUM_GPUS[gpu_type]

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
        env["LLMB_EXPERIMENT_DIR"] = experiment_dir
        env["MODEL_SIZE"] = task.model_size
        env["DTYPE"] = task.dtype
        env["JOB_TOTAL_GPUS"] = str(task.scale)
        env["GPU_TYPE"] = gpu_type

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

        # Resolve segment as a --segment flag. CLI override is already in cmd
        # via to_sbatch_args(). Otherwise, check for SBATCH_SEGMENT_SIZE from any
        # env source, or auto-detect for GB200/GB300.
        # NOTE: Newer Slurm versions support SBATCH_SEGMENT_SIZE as an env var
        # natively; if we ever drop support for older versions, we could set the
        # env var instead of the flag.
        if segment_override is None:
            env_segment = env.get('SBATCH_SEGMENT_SIZE')
            if env_segment is not None:
                cmd.insert(-1, f"--segment={env_segment}")
            elif gpu_type in {'gb200', 'gb300'}:
                computed = _compute_segment_size(num_nodes)
                cmd.insert(-1, f"--segment={computed}")
                logger.debug(f"Auto-detected segment size {computed} for {gpu_type}")

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
            env["LLMB_INSTALL"] = self.config.llmb_install
            env["MODEL_SIZE"] = task.model_size
            env["DTYPE"] = task.dtype
            env["JOB_TOTAL_GPUS"] = str(task.scale)
            env["GPU_TYPE"] = self.get_gpu_type()

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


def create_launcher(launcher_type, config, workloads):
    """Factory function to create the appropriate launcher."""
    if launcher_type == 'sbatch':
        return SbatchLauncher(config, workloads)
    elif launcher_type == 'configured_sbatch':
        return ConfiguredSbatchLauncher(config, workloads)
    elif launcher_type == 'nemo':
        return Nemo2Launcher(config, workloads)
    elif launcher_type == 'megatron_bridge':
        # Megatron Bridge can reuse the existing Nemo2Launcher
        return Nemo2Launcher(config, workloads)
    else:
        raise ValueError(f"Unknown launcher type: {launcher_type}")


def run_tests(config, task_list, workloads):
    """Run tests using the appropriate launcher for each task."""
    launchers = {}
    failed_tasks = []

    for task in task_list:
        # Get launcher type from workload metadata
        metadata = workloads[task.workload_key]['metadata']
        launcher_type = metadata.get('run', {}).get('launcher_type')
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

        # Run post-processing pipeline (resparse, upload, workload inspector)
        if slurm_job.job_id and slurm_job.job_workdir:
            try:
                pipeline = importlib.import_module("llmb_run.internal.post_processing_pipeline")

                results = pipeline.run_post_processing_pipeline(
                    config=config,
                    launcher_type=launcher_type,
                    slurm_job=slurm_job,
                    task=task,
                    workload_metadata=workloads[task.workload_key].get('metadata'),
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
