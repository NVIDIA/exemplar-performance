# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Workload management functions for LLMB installation.

This module provides functions for finding, parsing, and processing workload metadata files,
as well as executing workload setup tasks and installation scripts.
"""

import logging
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml

from llmb_install.cluster.slurm import augment_env_for_job_type
from llmb_install.config.models import EnvironmentPrompt
from llmb_install.environment.detector import get_clean_environment_for_subprocess, get_host_architecture
from llmb_install.environment.venv_manager import get_venv_environment

logger = logging.getLogger(__name__)

HF_TOKEN_PROMPT = EnvironmentPrompt(
    name="HF_TOKEN",
    description=(
        "Most workloads require a Hugging Face token to download models and datasets "
        "(https://huggingface.co/settings/tokens)."
    ),
    required=False,
)


def find_metadata_files(root_dir: str) -> List[Path]:
    """Find all metadata.yaml files in the given directory and its subdirectories, excluding deprecated folder."""
    metadata_files = []
    for path in Path(root_dir).rglob("metadata.yaml"):
        # TODO: deprecated/ was removed from the repo in 26.10. Drop this skip once all
        # releases that still ship it fall outside the support window (~6 months).
        if 'deprecated' not in path.parts:
            metadata_files.append(path)
    return metadata_files


def parse_metadata_file(file_path: Path) -> Dict[str, Any]:
    """Parse a metadata.yaml file and return its contents."""
    with open(file_path, 'r') as f:
        return yaml.safe_load(f)


def build_workload_dict(root_dir: str) -> Dict[str, Dict[str, Any]]:
    """
    Build a dictionary of available workloads from metadata.yaml files.
    Key format: 'workload_type'_'workload'
    """
    workload_dict = {}

    metadata_files = find_metadata_files(root_dir)

    for file_path in metadata_files:
        try:
            metadata = parse_metadata_file(file_path)

            general = metadata.get('general', {})
            workload_type = general.get('workload_type')
            workload = general.get('workload')

            if workload_type and workload:
                key = f"{workload_type}_{workload}"

                # Store all metadata and add installer-specific path field
                # This future-proofs against new metadata fields (e.g., downloads, models, etc.)
                workload_dict[key] = {
                    **metadata,
                    'path': str(file_path.parent),  # Installer-specific field
                }

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

    return workload_dict


def filter_tools_from_workload_list(workloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Filter tools from the workload list."""
    tools = {}
    for key, workload_data in workloads.items():
        if workload_data.get('general', {}).get('workload_type') == 'tools':
            tools[key] = workload_data
    return tools


def get_setup_tasks(workload_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Return the list of setup tasks for a workload.

    Behaviour:
    • If `setup.tasks` exists, return that list preserving order.
    • Otherwise, return an empty list.
    """
    setup_cfg: Dict[str, Any] = workload_data.get("setup", {}) or {}
    tasks: List[Dict[str, Any]] = setup_cfg.get("tasks", []) or []

    if tasks:
        return tasks

    # No explicit tasks defined
    return []


def get_environment_prompts(workload_data: Dict[str, Any]) -> List[EnvironmentPrompt]:
    """Return the environment variables a recipe declares under ``environment_prompts``."""
    raw = workload_data.get("environment_prompts", []) or []
    prompts: List[EnvironmentPrompt] = []
    for entry in raw:
        try:
            prompts.append(EnvironmentPrompt.model_validate(entry))
        except Exception as e:
            logger.warning(
                f"Ignoring malformed environment_prompts entry {entry!r} in {workload_data.get('path', '?')}: {e}"
            )
    return prompts


def collect_environment_prompts(
    workloads: Dict[str, Dict[str, Any]], selected_keys: Iterable[str]
) -> List[EnvironmentPrompt]:
    """Merge ``environment_prompts`` declarations across the selected workloads."""
    # HF_TOKEN remains an installer-wide prompt until recipes declare it explicitly.
    merged_prompts = {HF_TOKEN_PROMPT.name: HF_TOKEN_PROMPT.model_copy(deep=True)}
    for key in selected_keys:
        workload_data = workloads.get(key)
        if not workload_data:
            continue
        for prompt in get_environment_prompts(workload_data):
            if prompt.name not in merged_prompts:
                merged_prompts[prompt.name] = prompt.model_copy(update={"workloads": [key]})
            else:
                existing_prompt = merged_prompts[prompt.name]
                if key not in existing_prompt.workloads:
                    existing_prompt.workloads.append(key)
                existing_prompt.required = existing_prompt.required or prompt.required
                if not existing_prompt.description and prompt.description:
                    existing_prompt.description = prompt.description
    return list(merged_prompts.values())


def missing_environment_prompts(
    prompts: List[EnvironmentPrompt], env_vars: Optional[Dict[str, str]]
) -> List[EnvironmentPrompt]:
    """Return required prompts that have no non-blank value in ``env_vars``."""
    env_vars = env_vars or {}
    return [p for p in prompts if p.required and not (env_vars.get(p.name) or "").strip()]


def run_setup_tasks(
    workload_key: str,
    workload_data: Dict[str, Any],
    venv_path: Optional[str],
    venv_type: str,
    install_path: str,
    slurm_info: Dict[str, Any],
    gpu_type: str,
    node_architecture: str,
):
    """Execute setup tasks defined for a workload in serial order.

    Args:
        workload_key: Identifier such as "finetune_llama4-maverick".
        workload_data: Metadata dict for the workload.
        venv_path: Path to the venv to activate for this workload (may be None).
        venv_type: Environment type for venv_path ('uv', 'venv', or 'conda').
        install_path: Base installation path ($LLMB_INSTALL).
        slurm_info: Cluster SLURM config as gathered earlier.
        gpu_type: GPU type (e.g., 'h100', 'gb200').
        node_architecture: Canonical compute-node architecture from cluster config
            ('x86_64' or 'aarch64').  When this differs from the login-node arch,
            sbatch tasks run with venv stripped so that login-arch binaries are not
            propagated to compute nodes of a different architecture.
    """
    tasks = get_setup_tasks(workload_data)
    if not tasks:
        return

    workload_dir = workload_data["path"]

    # Determine once whether login-node and compute-node architectures differ.
    # When they do, sbatch tasks must not propagate the login-arch venv to the
    # compute node (sbatch defaults to --export=ALL).
    host_arch = get_host_architecture()
    arch_mismatch = bool(host_arch and node_architecture and host_arch != node_architecture)

    for idx, task in enumerate(tasks, start=1):
        name = task.get("name", f"task_{idx}")
        cmd = task.get("cmd")
        if not cmd:
            print(f"Skipping task '{name}' - no cmd provided.")
            continue
        job_type = task.get("job_type", "local").lower()
        requires_gpus = task.get("requires_gpus", False)
        # Kubernetes installs pass an empty slurm_info: there is no Slurm to
        # submit to, so Slurm-backed setup tasks cannot run. Skip them with a
        # warning instead of crashing on the missing SLURM configuration.
        if not slurm_info and job_type in ("sbatch", "srun", "nemo2"):
            print(
                f"Skipping setup task '{name}' for {workload_key}: job_type '{job_type}' requires Slurm, "
                f"which is not available on this install target."
            )
            continue
        # Ensure all env values are strings to avoid type-related subprocess errors
        task_env_extra_raw = task.get("env", {}) or {}
        task_env_extra: Dict[str, str] = {k: str(v) for k, v in task_env_extra_raw.items()}

        # Compose environment.
        # For sbatch tasks on mismatched architectures: strip both the installer's
        # own venv (present in os.environ) and any workload venv so that login-arch
        # binaries are not forwarded to compute nodes via sbatch --export=ALL.
        # For all other cases: activate the workload venv (if any) or inherit as-is.
        if job_type == "sbatch" and arch_mismatch:
            env = get_clean_environment_for_subprocess()
            print(
                f"  Note: login arch ({host_arch}) differs from compute arch ({node_architecture}). "
                f"Running sbatch task '{name}' with venv stripped to avoid arch mismatch."
            )
        elif venv_path:
            env = get_venv_environment(venv_path, venv_type)
        else:
            env = os.environ.copy()

        env["LLMB_INSTALL"] = install_path
        env["LLMB_WORKLOAD"] = os.path.join(install_path, "workloads", workload_key)
        env["MANUAL_INSTALL"] = "false"
        env["GPU_TYPE"] = gpu_type
        # Architectures, so a task can decide where to build: cross-compile here
        # for the compute nodes' CPU architecture, or fall back to srun only when the two differ.
        env["LLMB_NODE_ARCHITECTURE"] = node_architecture or ""
        env["LLMB_HOST_ARCHITECTURE"] = host_arch or ""
        # task-specific overrides
        env.update(task_env_extra)
        # SLURM augmentation (no-op on targets without Slurm, e.g. Kubernetes)
        if slurm_info:
            augment_env_for_job_type(env, job_type, slurm_info, requires_gpus)

        banner = f"Running setup task [{workload_key}] - {name} (type: {job_type})"
        print("\n" + banner)
        print("-" * len(banner))

        try:
            if job_type == "sbatch":
                # Submit the sbatch job. Assume `cmd` contains the sbatch script path (and optional args).
                # Force log output to the workload's install dir; %x expands to --job-name if the script sets one.
                log_path = os.path.join(env["LLMB_WORKLOAD"], "%x-%j.out")
                full_cmd = ["sbatch", f"--output={log_path}"] + shlex.split(cmd)
                result = subprocess.run(full_cmd, check=True, capture_output=True, text=True, cwd=workload_dir, env=env)

                # Attempt to extract job-id from output (handles both --parsable and default formats)
                stdout = result.stdout.strip()
                job_id_match = re.search(r"(\d+)$", stdout)
                job_id = job_id_match.group(1) if job_id_match else None
                if job_id:
                    print(f"✓ Submitted SBATCH job (id={job_id}) for task '{name}'.")
                else:
                    print(f"✓ Submitted SBATCH job for task '{name}': {stdout or '(no output)'}")
            else:
                # local, nemo2, srun are run inline - split command safely to avoid shell injection
                cmd_list = shlex.split(cmd)
                subprocess.run(cmd_list, check=True, cwd=workload_dir, env=env, capture_output=False)
                print(f"✓ Finished task '{name}' successfully.")

        except subprocess.CalledProcessError as e:
            e.add_note(f"Setup task '{name}' for {workload_key} failed.")
            raise
