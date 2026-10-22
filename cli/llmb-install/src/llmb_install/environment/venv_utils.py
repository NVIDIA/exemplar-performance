# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Utilities for virtual environment management and reuse."""

import os
import re
from typing import Dict, Optional


def extract_venv_hash(venv_path: str) -> Optional[str]:
    """Extract the 12-character dependency hash from a venv path.

    Args:
        venv_path: Path to virtual environment (e.g., "/path/venvs/venv_abc123def456")

    Returns:
        The 12-character hex hash if found, None otherwise

    Examples:
        >>> extract_venv_hash("/install/venvs/venv_abc123def456")
        'abc123def456'
        >>> extract_venv_hash("/install/venvs/myenv")
        None
    """
    match = re.search(r'venv_([a-f0-9]{12})', venv_path)
    return match.group(1) if match else None


def build_venv_hash_mapping(workload_venvs: Dict[str, str]) -> Dict[str, str]:
    """Build a reverse mapping from dependency hash to venv path.

    This enables quick lookup of existing venvs by their dependency hash
    for reuse during incremental installs.

    Args:
        workload_venvs: Dictionary mapping workload names to venv paths

    Returns:
        Dictionary mapping 12-char hashes to venv paths

    Examples:
        >>> venvs = {'wl1': '/venvs/venv_abc123def456', 'wl2': '/venvs/venv_123abc456def'}
        >>> mapping = build_venv_hash_mapping(venvs)
        >>> mapping['abc123def456']
        '/venvs/venv_abc123def456'
    """
    hash_to_venv = {}
    for venv_path in workload_venvs.values():
        venv_hash = extract_venv_hash(venv_path)
        if venv_hash:
            hash_to_venv[venv_hash] = venv_path
    return hash_to_venv


def should_reuse_venv(
    dependency_hash: str, existing_venv_mapping: Dict[str, str], validate_exists: bool = True
) -> Optional[str]:
    """Determine if an existing venv can be reused for a dependency group.

    Args:
        dependency_hash: Full dependency hash for the new workload group
        existing_venv_mapping: Mapping of 12-char hashes to venv paths
        validate_exists: If True, verify the venv path actually exists

    Returns:
        Path to reusable venv if found and valid, None otherwise

    Examples:
        >>> mapping = {'abc123def456': '/venvs/venv_abc123def456'}
        >>> should_reuse_venv('abc123def456789', mapping, validate_exists=False)
        '/venvs/venv_abc123def456'
        >>> should_reuse_venv('fedcba987654321', mapping, validate_exists=False)
        None
    """
    # Truncate to 12 chars for comparison
    short_hash = dependency_hash[:12]

    # Check if we have a matching venv
    venv_path = existing_venv_mapping.get(short_hash)
    if not venv_path:
        return None

    # Optionally validate the path exists
    if validate_exists and not os.path.exists(venv_path):
        return None

    return venv_path


def extract_venv_type_from_config(cluster_config: Dict) -> Optional[str]:
    """Extract venv_type from cluster config with backward compatibility.

    Tries multiple locations in order of preference:
    1. cluster_config.install.venv_type (new format)
    2. First workload's venv_type in workloads.config (old format)

    Args:
        cluster_config: Cluster configuration dictionary

    Returns:
        venv_type string if found, None otherwise
    """
    # Try new install section first
    install_section = cluster_config.get('install', {})
    venv_type = install_section.get('venv_type')
    if venv_type:
        return venv_type

    # Fallback to extracting from existing workload configs
    workload_configs = cluster_config.get('workloads', {}).get('config', {})
    for wl_config in workload_configs.values():
        venv_type = wl_config.get('venv_type')
        if venv_type:
            return venv_type

    return None
