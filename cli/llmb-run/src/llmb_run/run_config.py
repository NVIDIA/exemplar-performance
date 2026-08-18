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

"""Configuration file generation for individual test runs."""

import getpass
import hashlib
import importlib
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional

import yaml

from llmb_run.config_manager import ClusterConfig
from llmb_run.constants import USER_ENV_VARS
from llmb_run.slurm_utils import get_cluster_name

logger = logging.getLogger('llmb_run.run_config')

# internal mode
_internal_modules_available = False
try:
    internal_module = importlib.import_module("llmb_run.internal")
    _internal_modules_available = True
except ModuleNotFoundError:
    # Internal extensions unavailable – safe to ignore.
    pass


def _get_current_user() -> str:
    """Resolve the human user to attribute a job to.

    Checks USER_ENV_VARS (CI-provided identity, since the OS/container user
    in a runner is generic, e.g. root; SUDO_USER covers the interactive sudo
    case) before falling back to the OS user. getpass.getuser() can raise on
    minimal containers with no matching passwd entry, so it's wrapped rather
    than allowed to crash config generation.
    """
    for var in USER_ENV_VARS:
        val = os.environ.get(var)
        if val:
            return val
    try:
        return getpass.getuser()
    except Exception:
        return 'unknown'


def load_llmb_version(repo_root: str) -> Optional[str]:
    """Load llmb_version from release.yaml at repo root.

    Args:
        repo_root: Path to repository root directory.

    Returns:
        Version string, or None if not found
    """
    release_yaml_path = os.path.join(repo_root, 'release.yaml')
    if os.path.exists(release_yaml_path):
        try:
            with open(release_yaml_path, 'r') as f:
                release_data = yaml.safe_load(f)
                if version := release_data.get('llmb_version'):
                    logger.debug(f"Loaded llmb_version '{version}' from {release_yaml_path}")
                    return version
        except Exception as e:
            logger.warning(f"Failed to load release.yaml from {release_yaml_path}: {e}")

    return None


def find_experiment_config_file(experiment_dir: str) -> Optional[str]:
    """Find the experiment config file in the configs directory.

    Args:
        experiment_dir: Directory where experiment files are located

    Returns:
        Path to the config file, or None if not found
    """
    configs_dir = os.path.join(experiment_dir, 'configs')
    if not os.path.exists(configs_dir):
        logger.warning(f"Configs directory not found: {configs_dir}")
        return None

    try:
        for filename in os.listdir(configs_dir):
            if filename.endswith('_config.yaml'):
                config_path = os.path.join(configs_dir, filename)
                logger.debug(f"Found experiment config file: {config_path}")
                return config_path
    except Exception as e:
        logger.warning(f"Error searching for config file in {configs_dir}: {e}")

    logger.warning(f"No *_config.yaml file found in {configs_dir}")
    return None


def remove_nsys_callback_from_config(config_data: Dict[str, Any]) -> None:
    """Remove NsysCallback entries from the trainer callbacks section.

    Modifies the config_data in place since we own this data structure.

    Args:
        config_data: Parsed YAML configuration data to modify
    """
    # Navigate to trainer.callbacks if it exists
    if 'trainer' in config_data and isinstance(config_data['trainer'], dict):
        if 'callbacks' in config_data['trainer'] and isinstance(config_data['trainer']['callbacks'], list):
            # Filter out NsysCallback entries
            filtered_callbacks = []
            for callback in config_data['trainer']['callbacks']:
                if isinstance(callback, dict) and '_target_' in callback:
                    if 'nsys.NsysCallback' not in callback['_target_']:
                        filtered_callbacks.append(callback)
                else:
                    # Keep non-dict callbacks or those without _target_
                    filtered_callbacks.append(callback)

            config_data['trainer']['callbacks'] = filtered_callbacks
            logger.debug(f"Removed NsysCallback entries, {len(filtered_callbacks)} callbacks remaining")


def remove_checkpointing_from_config(config_data: Dict[str, Any]) -> None:
    """Remove checkpointing from the config data.

    Removes:
    - resume.* if exists
    - trainer.enable_checkpointing

    Args:
        config_data: Parsed YAML configuration data to modify
    """
    # Remove top-level resume section if it exists
    if 'resume' in config_data:
        del config_data['resume']
        logger.debug("Removed top-level 'resume' section from config")

    # Remove trainer.enable_checkpointing if it exists
    if 'trainer' in config_data and isinstance(config_data['trainer'], dict):
        if 'enable_checkpointing' in config_data['trainer']:
            del config_data['trainer']['enable_checkpointing']
            logger.debug("Removed 'enable_checkpointing' from trainer section")


def config_to_bytes(config_data: Dict[str, Any]) -> bytes:
    """Convert config data to normalized bytes for hashing.

    Args:
        config_data: Configuration dictionary

    Returns:
        Normalized bytes representation
    """
    # Convert to JSON with sorted keys for consistent ordering
    normalized_json = json.dumps(config_data, sort_keys=True, separators=(',', ':'), default=str)
    return normalized_json.encode('utf-8')


def generate_experiment_id(experiment_dir: str, fw_version: str) -> Optional[str]:
    """Generate experiment ID based on normalized config file hash and fw_version.

    Args:
        experiment_dir: Directory where experiment files are located
        fw_version: Framework version to include in hash (required)

    Returns:
        SHA256 hash of the config and fw_version, or None if config not found
    """
    config_file_path = find_experiment_config_file(experiment_dir)
    if not config_file_path:
        return None

    try:
        # Load the config file
        with open(config_file_path, 'r') as f:
            config_data = yaml.safe_load(f)

        # Remove NsysCallback entries (modifies in place)
        remove_nsys_callback_from_config(config_data)

        # Remove checkpointing configuration (modifies in place)
        remove_checkpointing_from_config(config_data)

        # Convert config to normalized bytes
        config_bytes = config_to_bytes(config_data)

        # Convert fw_version to bytes
        fw_version_bytes = fw_version.encode('utf-8')

        # Concatenate config bytes and fw_version bytes for hashing
        hash_input = config_bytes + fw_version_bytes

        # Generate SHA256 hash
        hash_object = hashlib.sha256(hash_input)
        experiment_id = hash_object.hexdigest()

        logger.debug(
            f"Generated experiment_id: {experiment_id} for config: {config_file_path}, fw_version: {fw_version}"
        )
        return experiment_id

    except Exception as e:
        logger.warning(f"Failed to generate experiment_id from {config_file_path}: {e}")
        return None


def resolve_container_images(images_field: Any, gpu_type: str) -> list:
    """Resolve container images from various formats to a list of image strings.

    Args:
        images_field: The container.images field from metadata, which can be:
            - A list of strings: ['nvcr.io#nvidia/nemo:25.04.01']
            - A list of dicts: [{'url': 'nvcr.io#...', 'name': '...'}]
            - A dict with by_gpu structure: {'by_gpu': {'h100': [...], 'default': [...]}}
        gpu_type: The GPU type to use for by_gpu resolution

    Returns:
        List of resolved image strings (full paths like 'nvcr.io#nvidia/nemo:25.07')
        Returns empty list if no valid images found
    """
    if not images_field:
        return []

    # Handle by_gpu dictionary structure
    if isinstance(images_field, dict) and 'by_gpu' in images_field:
        by_gpu = images_field['by_gpu']
        # Try to get images for the specific gpu_type, fall back to 'default'
        if gpu_type in by_gpu:
            images_field = by_gpu[gpu_type]
        elif 'default' in by_gpu:
            images_field = by_gpu['default']
        else:
            # No matching GPU type and no default
            logger.warning(f"No images found for gpu_type '{gpu_type}' and no 'default' key in by_gpu structure")
            return []

    # Handle list format
    if isinstance(images_field, list):
        resolved_images = []
        for item in images_field:
            if isinstance(item, str):
                # Simple string format
                resolved_images.append(item)
            elif isinstance(item, dict) and 'url' in item:
                # Dict with 'url' key
                resolved_images.append(item['url'])
        return resolved_images

    # Unexpected format
    logger.warning(f"Unexpected images_field format: {type(images_field)}")
    return []


def create_llmb_config(task, job_id, workdir, config: ClusterConfig, workloads):
    """Create llmb-config.yaml file in the experiment's folder.

    Args:
        task: WorkloadTask object containing job parameters
        job_id: Job ID returned from the launcher
        workdir: Working directory path (may be None for some launchers)
        config: Cluster configuration
        workloads: Workloads dictionary

    Returns:
        str: Path to the created config file, or None if creation failed
    """
    # Setup: determine paths and get workload metadata
    try:
        if workdir:
            # For Nemo2 launcher, use the workdir returned by the launcher
            config_dir = workdir
        else:
            # For other launchers, use the installed workload directory
            llmb_install = config.llmb_install
            workload_key = task.workload_key
            config_dir = os.path.join(llmb_install, 'workloads', workload_key)

        config_file_path = os.path.join(config_dir, f'llmb-config_{job_id}.yaml')
        os.makedirs(config_dir, exist_ok=True)

        # Get workload metadata
        workload_info = workloads[task.workload_key]
        metadata = workload_info['metadata']
    except KeyError as e:
        logger.error(f"Failed to get workload metadata: {e}. workload_key={task.workload_key}")
        return None
    except Exception as e:
        logger.error(f"Failed to setup config directory: {type(e).__name__}: {e}. workdir={workdir}")
        return None

    # Resolve container images and extract framework version
    gpu_type = config.gpu_type
    images_field = metadata.get('container', {}).get('images', [])
    resolved_images = resolve_container_images(images_field, gpu_type)

    if resolved_images:
        try:
            # Extract fw_version from the first image
            fw_version = resolved_images[0].split(':')[-1] if ':' in resolved_images[0] else 'unknown'
        except (IndexError, AttributeError) as e:
            logger.error(f"Failed to parse image version: {e}. resolved_images={resolved_images}")
            fw_version = 'unknown'
    else:
        logger.debug(f"No container images resolved for gpu_type '{gpu_type}', images_field={images_field}")
        fw_version = 'unknown'

    # Validate cluster name
    try:
        cluster_name = config.cluster_name
        if not cluster_name:
            cluster_name = get_cluster_name()
            if cluster_name:
                logger.warning(
                    f"\n{'='*80}\n"
                    f"⚠️  WARNING: cluster_name not configured!\n"
                    f"   Using '{cluster_name}' detected from SLURM configuration.\n"
                    f"   To fix this, add the following to your cluster_config.yaml:\n"
                    f"   \n"
                    f"   cluster_name: {cluster_name}\n"
                    f"{'='*80}"
                )
            if _internal_modules_available and not cluster_name:
                logger.error("Cluster name not configured and unable to auto-detect.")
                return None
    except Exception as e:
        logger.error(f"Failed to determine cluster name: {type(e).__name__}: {e}")
        return None

    # Generate experiment_id (only for nemo2 workloads)
    framework = metadata.get('general', {}).get('framework', 'unknown')
    if framework == 'nemo2':
        experiment_id = generate_experiment_id(config_dir, fw_version)
    else:
        logger.debug(f"Skipping experiment_id generation for {framework} workload")
        experiment_id = None

    # Get llmb_version from release.yaml (new format) with fallback to metadata.yaml (legacy)
    repo_root = config.llmb_repo
    llmb_version = load_llmb_version(repo_root) or metadata.get('general', {}).get('gsw_version') or 'unknown'

    # Build configuration dictionary
    gpu_slurm_env = config.slurm.env(target='gpu')
    llmb_config = {
        'job_info': {
            'job_id': job_id,
            'user': _get_current_user(),
            'launcher_type': metadata.get('run', {}).get('launcher_type', ''),
            'launch_time': datetime.now().isoformat(),
            'experiment_id': experiment_id,
        },
        'workload_info': {
            'framework': framework,
            'gsw_version': llmb_version,  # Keep field name for backward compatibility
            'fw_version': fw_version,
            'workload_type': workload_info.get('workload_type', ''),
            'synthetic_dataset': workload_info.get('synthetic_dataset', True),
        },
        'model_info': {
            'model_name': metadata.get('general', {}).get('model', workload_info.get('workload', '')),
            'model_size': task.model_size,
            'dtype': task.dtype,
            'scale': task.scale,
            'gpu_type': gpu_type,
        },
        'cluster_info': {
            'cluster_name': cluster_name,
            'gpus_per_node': config.slurm.gpus_per_node(target='gpu'),
            'llmb_install': config.llmb_install,
            'llmb_repo': config.llmb_repo,
            'slurm_account': gpu_slurm_env.get('SBATCH_ACCOUNT', ''),
            'slurm_gpu_partition': config.slurm.partition(target='gpu'),
            'slurm_cpu_partition': config.slurm.partition(target='cpu'),
        },
        'container_info': {
            'images': resolved_images,
        },
        'job_config': {
            'profile_enabled': task.profile,
            'proxy': task.proxy,
            'strong_scaling': os.getenv('STRONG_SCALING', 'false').lower() == 'true',
            'env_overrides': task.env_overrides,
            'model_overrides': task.model_overrides,
            'mbridge_args': list(task.mbridge_args),
        },
    }

    # Write configuration to file
    try:
        with open(config_file_path, 'w') as f:
            yaml.dump(llmb_config, f, default_flow_style=False, sort_keys=False)
        logger.debug(f"Created llmb-config.yaml at: {config_file_path}")
        return config_file_path
    except Exception as e:
        logger.error(
            f"Failed to write llmb-config.yaml: {type(e).__name__}: {e}. " f"config_file_path={config_file_path}"
        )
        return None
