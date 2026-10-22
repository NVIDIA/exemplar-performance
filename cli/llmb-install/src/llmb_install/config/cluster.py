# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Cluster configuration management for LLMB installer."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from llmb_install.cluster.slurm import get_cluster_name
from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)


def load_cluster_config(install_path: str) -> Optional[Dict[str, Any]]:
    """Load cluster_config.yaml from an installation directory.

    Args:
        install_path: Path to the installation directory

    Returns:
        Dict containing config, or None if not found/invalid
    """
    cluster_config_path = Path(install_path) / "cluster_config.yaml"
    if not cluster_config_path.exists():
        return None

    try:
        with open(cluster_config_path, 'r') as f:
            config = yaml.safe_load(f)
            return normalize_cluster_config(config) if config else None
    except (yaml.YAMLError, OSError) as e:
        logger.warning(f"Could not load cluster config: {e}")
        return None


def normalize_cluster_config(config: dict) -> dict:
    """Normalize a loaded cluster config to v2 format.

    Detects v1 by presence of 'launcher' key (v1 configs have no schema_version).
    If already v2 (no 'launcher'), returns as-is.
    """
    if 'launcher' not in config:
        return config  # Already v2

    normalized = dict(config)
    launcher = normalized.pop('launcher')

    # Flatten launcher fields to top level
    normalized['schema_version'] = 2
    for key in ('llmb_repo', 'llmb_install', 'gpu_type', 'cluster_name'):
        if key in launcher:
            normalized[key] = launcher[key]

    # Move launcher.node_architecture → install.node_architecture if needed
    if 'node_architecture' in launcher:
        install = normalized.get('install', {}).copy()
        if 'node_architecture' not in install:
            install['node_architecture'] = launcher['node_architecture']
            normalized['install'] = install

    # Restructure flat slurm keys into gpu/cpu blocks
    slurm = normalized.get('slurm', {})
    if 'gpu_partition' in slurm or 'cpu_partition' in slurm:
        new_slurm = {}
        if 'account' in slurm:
            new_slurm['account'] = slurm['account']
        new_slurm['gpu'] = {
            'partition': slurm.get('gpu_partition', ''),
            'gres': slurm.get('gpu_partition_gres'),
        }
        new_slurm['cpu'] = {
            'partition': slurm.get('cpu_partition', ''),
            'gres': slurm.get('cpu_partition_gres'),
        }
        normalized['slurm'] = new_slurm

    return normalized


def create_cluster_config(
    install_path: str,
    root_dir: str,
    selected_workloads: List[str],
    slurm_info: dict,
    env_vars: dict,
    gpu_type: str,
    venv_type: str,
    workload_venvs: Dict[str, str],
    node_architecture: str = 'x86_64',
    install_method: str = 'local',
    image_folder: Optional[str] = None,
    existing_cluster_config: Optional[Dict[str, Any]] = None,
) -> None:
    """Create or update cluster_config.yaml file with all installation configuration.

    Args:
        install_path: Base installation directory
        root_dir: Path to the LLMB repository root
        selected_workloads: List of selected workload keys
        slurm_info: SLURM configuration dictionary
        env_vars: Environment variables dictionary
        gpu_type: Selected GPU type
        venv_type: Type of virtual environment used ('venv' or 'conda')
        workload_venvs: Dictionary mapping workload keys to their venv paths
        node_architecture: Node architecture (e.g., 'x86_64', 'arm64')
        install_method: Installation method ('local' or 'slurm')
        image_folder: Optional path to container image folder
        existing_cluster_config: Optional existing config for incremental updates
    """
    # If updating, merge with existing workloads
    if existing_cluster_config:
        existing_workloads = existing_cluster_config.get('workloads', {}).get('installed', [])
        all_workloads = list(set(existing_workloads + selected_workloads))

        # Merge venv configs - existing configs take precedence
        existing_configs = existing_cluster_config.get('workloads', {}).get('config', {})
        merged_workload_venvs = dict(existing_configs)
        # Add new workload venvs
        for wl_name in selected_workloads:
            if wl_name not in merged_workload_venvs:
                venv_path = workload_venvs.get(wl_name)
                if venv_path:
                    merged_workload_venvs[wl_name] = {
                        'venv_path': venv_path,
                        'venv_type': venv_type,
                    }
    else:
        all_workloads = selected_workloads
        merged_workload_venvs = {}
        for workload_key in selected_workloads:
            venv_path = workload_venvs.get(workload_key)
            if venv_path:
                merged_workload_venvs[workload_key] = {
                    'venv_path': venv_path,
                    'venv_type': venv_type,
                }

    # Add cluster name if available
    # For incremental installs, preserve existing cluster_name if current detection fails
    cluster_name = get_cluster_name()
    if not cluster_name and existing_cluster_config:
        cluster_name = existing_cluster_config.get('cluster_name')

    # Build install metadata section
    # For incremental installs, merge with existing install section to preserve all fields
    if existing_cluster_config:
        # Start with existing install section (preserves all fields including image_folder)
        install_section = existing_cluster_config.get('install', {}).copy()
        # Update with new values (these should match existing for incremental, but update anyway)
        install_section['venv_type'] = venv_type
        install_section['method'] = install_method
        install_section['node_architecture'] = node_architecture
        # Update image_folder if explicitly provided (allows override via -i flag)
        if image_folder is not None:
            install_section['image_folder'] = image_folder
        # If image_folder is None and not in existing, don't add it
    else:
        # New installation - create fresh install section
        install_section = {
            'venv_type': venv_type,
            'method': install_method,
            'node_architecture': node_architecture,  # Canonical location
        }
        if image_folder:
            install_section['image_folder'] = image_folder

    cluster_config = {
        'schema_version': 2,
        'llmb_repo': root_dir,
        'llmb_install': install_path,
        'gpu_type': gpu_type,
        **({'cluster_name': cluster_name} if cluster_name else {}),
        'install': install_section,
        'environment': env_vars,
        'slurm': {
            'account': slurm_info['slurm']['account'],
            'gpu': {
                'partition': slurm_info['slurm']['gpu_partition'],
                'gres': slurm_info['slurm'].get('gpu_partition_gres'),
            },
            'cpu': {
                'partition': slurm_info['slurm']['cpu_partition'],
                'gres': slurm_info['slurm'].get('cpu_partition_gres'),
            },
        },
        'workloads': {'installed': all_workloads, 'config': merged_workload_venvs},
    }

    # Write the cluster config file
    config_path = os.path.join(install_path, "cluster_config.yaml")
    with open(config_path, 'w') as f:
        yaml.dump(cluster_config, f, default_flow_style=False, sort_keys=False)

    logger.debug(f"Created cluster configuration: {config_path}")
