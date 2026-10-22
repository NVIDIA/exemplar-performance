# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""GPU detection and configuration utilities for LLMB installer."""

import copy
import re
from typing import Any, Dict, List

from llmb_install.constants import (
    ARCHITECTURES,
    DEFAULT_ARCHITECTURE_BY_GPU,
    GPU_PREFIX_PRIORITY,
    SUPPORTED_GPU_TYPES,
)


def _select_from_by_gpu(mapping: Any, gpu_type: str, item_label: str) -> Any:
    """Select a value from a {'by_gpu': {...}} mapping with optional 'default'.

    If mapping does not contain 'by_gpu', it is returned unchanged.
    Raises ValueError when selection is impossible.
    """
    if not isinstance(mapping, dict) or 'by_gpu' not in mapping:
        return mapping
    table = mapping.get('by_gpu') or {}
    if not isinstance(table, dict):
        raise ValueError(f"Invalid by_gpu mapping for {item_label}")
    if gpu_type in table:
        return table[gpu_type]
    if 'default' in table:
        return table['default']
    raise ValueError(f"No {item_label} defined for GPU '{gpu_type}' and no 'default' provided.")


def _resolve_images_for_gpu(images_field: Any, gpu_type: str) -> List[Any]:
    """Normalize container.images for a specific GPU type.

    Supports:
      - list[str | {url,name}]
      - { by_gpu: { h100|b200|gb200|default: list[str|{url,name}] } }
    Returns a list regardless of input form.
    """
    resolved = _select_from_by_gpu(images_field, gpu_type, "container.images")
    if isinstance(resolved, list):
        return resolved
    return [resolved]


def _resolve_repositories_for_gpu(repos_field: Any, gpu_type: str) -> Dict[str, Dict[str, str]]:
    """Resolve repositories for a specific GPU type.

    Supports a flat mapping {repo_key: {url, commit}} or a top-level
    { by_gpu: { h100|b200|gb200|default: { repo_key: {url, commit} } } }.
    """
    # If empty or None, return empty mapping
    if not repos_field:
        return {}

    # If top-level by_gpu wrapper exists, select for this GPU
    selected = _select_from_by_gpu(repos_field, gpu_type, "repositories")

    if not isinstance(selected, dict):
        raise ValueError("repositories must resolve to a mapping of repo_key -> {url, commit}")

    # Validate and normalize entries
    out: Dict[str, Dict[str, str]] = {}
    for name, entry in selected.items():
        if not isinstance(entry, dict) or 'url' not in entry or 'commit' not in entry:
            raise ValueError(f"Invalid repository entry for '{name}' after GPU resolution")
        out[name] = {'url': entry['url'], 'commit': entry['commit'], 'submodules': entry.get('submodules', False)}
    return out


def _resolve_tools_for_gpu(tools_field: Any, gpu_type: str) -> Dict[str, str]:
    """Resolve tools for a specific GPU type.

    Three-level resolution:
    1. If GPU type explicitly in by_gpu mapping → return that version
    2. Else if 'default' key exists in by_gpu → return default version
    3. Else → return empty dict (use container version, no download)

    Supports:
      - Simple string: {tool_name: "version"}
      - GPU conditional: {tool_name: {by_gpu: {h100: "v1", default: "v2"}}}

    Returns:
        Dict[str, str]: Mapping of tool_name -> version, or empty dict if using container
    """
    if not tools_field:
        return {}

    if not isinstance(tools_field, dict):
        return {}

    resolved: Dict[str, str] = {}

    for tool_name, tool_spec in tools_field.items():
        # Simple string format: tool_name: "version"
        if isinstance(tool_spec, str):
            resolved[tool_name] = tool_spec
        # GPU-conditional format: tool_name: {by_gpu: {...}}
        elif isinstance(tool_spec, dict) and 'by_gpu' in tool_spec:
            by_gpu_table = tool_spec.get('by_gpu')
            if not isinstance(by_gpu_table, dict):
                raise ValueError(f"Invalid by_gpu mapping for tool '{tool_name}'")

            # Check if GPU type is explicitly listed
            if gpu_type in by_gpu_table:
                version = by_gpu_table[gpu_type]
                if isinstance(version, str):
                    resolved[tool_name] = version
            # Check for default fallback
            elif 'default' in by_gpu_table:
                version = by_gpu_table['default']
                if isinstance(version, str):
                    resolved[tool_name] = version
            # Otherwise: GPU not specified and no default = use container version (don't add to resolved)

    return resolved


def resolve_gpu_overrides(workloads: Dict[str, Dict[str, Any]], gpu_type: str) -> Dict[str, Dict[str, Any]]:
    """Resolve GPU-specific image, repository, and tool overrides for each workload.

    Returns a deep-copied structure where:
      - container.images is a concrete list
      - repositories is a flat {repo_key: {url, commit}} mapping
      - tools is a flat {tool_name: version} mapping (or empty if using container)
    """
    specialized: Dict[str, Dict[str, Any]] = {}
    for key, workload in workloads.items():
        wd_copy = copy.deepcopy(workload)

        # Resolve images
        container_cfg = wd_copy.get('container', {}) or {}
        if 'images' in container_cfg:
            container_cfg['images'] = _resolve_images_for_gpu(container_cfg['images'], gpu_type)
            wd_copy['container'] = container_cfg

        # Resolve repositories (top-level by_gpu wrapper supported)
        repos_cfg = wd_copy.get('repositories', {}) or {}
        if repos_cfg:
            wd_copy['repositories'] = _resolve_repositories_for_gpu(repos_cfg, gpu_type)

        # Resolve setup (top-level by_gpu wrapper supported)
        setup_cfg = wd_copy.get('setup', {}) or {}
        if setup_cfg:
            wd_copy['setup'] = _select_from_by_gpu(setup_cfg, gpu_type, "setup")

        # Resolve tools (top-level by_gpu wrapper supported)
        tools_cfg = wd_copy.get('tools', {}) or {}
        if tools_cfg:
            wd_copy['tools'] = _resolve_tools_for_gpu(tools_cfg, gpu_type)

        specialized[key] = wd_copy
    return specialized


def get_supported_gpu_types(workloads: Dict[str, Dict[str, Any]]) -> set[str]:
    """Get all supported GPU types from workload metadata files."""
    gpu_types = set()

    for workload_data in workloads.values():
        run_config = workload_data.get('run', {})
        gpu_configs = run_config.get('gpu_configs', {})
        gpu_types.update(gpu_configs.keys())

    return gpu_types


def filter_workloads_by_gpu_type(workloads: Dict[str, Dict[str, Any]], gpu_type: str) -> Dict[str, Dict[str, Any]]:
    """Filter workloads to only include those that support the specified GPU type."""
    filtered_workloads = {}

    for key, workload_data in workloads.items():
        run_config = workload_data.get('run', {})
        gpu_configs = run_config.get('gpu_configs', {})

        # Only include workloads that have configuration for the selected GPU type
        if gpu_type in gpu_configs:
            filtered_workloads[key] = workload_data

    return filtered_workloads


def _parse_gpu_type(gpu_type: str) -> tuple[str, int]:
    """Parse GPU type into prefix and model number.

    Expected format: <alphabetic_prefix><numeric_model>
    Examples: 'gb300' -> ('gb', 300), 'h100' -> ('h', 100)

    Args:
        gpu_type: GPU type string (e.g., 'gb300', 'h100')

    Returns:
        tuple[str, int]: (prefix, model_number)

    Raises:
        ValueError: If GPU type doesn't match expected format
    """
    match = re.match(r'^([a-z]+)(\d+)$', gpu_type.lower())
    if not match:
        raise ValueError(
            f"GPU type '{gpu_type}' does not match expected format '<prefix><number>' "
            f"(e.g., 'gb300', 'h100'). GPU types must consist of lowercase letters "
            f"followed by digits with no other characters."
        )
    prefix, number_str = match.groups()
    return prefix, int(number_str)


def _gpu_sort_key(gpu_type: str) -> tuple[int, int]:
    """Generate sort key for GPU types.

    Sorts by prefix priority (lower priority number = higher precedence),
    then by model number descending within the same prefix.

    Example ordering with current priorities:
        gb300 (priority=1, model=-300)
        gb200 (priority=1, model=-200)
        b300  (priority=2, model=-300)
        b200  (priority=2, model=-200)
        h100  (priority=3, model=-100)

    Args:
        gpu_type: GPU type string (e.g., 'gb300', 'h100')

    Returns:
        tuple[int, int]: (prefix_priority, -model_number) for sorting

    Raises:
        ValueError: If GPU prefix is not in GPU_PREFIX_PRIORITY map
    """
    prefix, model_number = _parse_gpu_type(gpu_type)

    if prefix not in GPU_PREFIX_PRIORITY:
        known_prefixes = sorted(GPU_PREFIX_PRIORITY.keys())
        raise ValueError(
            f"GPU prefix '{prefix}' not found in GPU_PREFIX_PRIORITY. "
            f"Add '{prefix}' to GPU_PREFIX_PRIORITY in constants.py. "
            f"Known prefixes: {known_prefixes}"
        )

    priority = GPU_PREFIX_PRIORITY[prefix]
    # Negative model number for descending sort (higher models first)
    return (priority, -model_number)


def get_available_gpu_choices(workloads: Dict[str, Dict[str, Any]]) -> List[Dict[str, str]]:
    """Get available GPU type choices based on workloads.

    GPU types are sorted by prefix priority (gb > b > h), then by model number
    descending within each prefix. This ensures premium/latest GPUs appear first.

    Args:
        workloads: Dictionary of workload metadata

    Returns:
        List[Dict[str, str]]: List of choice dictionaries with 'name' and 'value' keys,
                              sorted by display priority

    Raises:
        ValueError: If no supported GPU types found, or if a GPU type has an unknown prefix
    """
    # Get supported GPU types from workloads, but limit to known types
    supported_types = get_supported_gpu_types(workloads)
    available_types = list(supported_types.intersection(SUPPORTED_GPU_TYPES))

    if not available_types:
        raise ValueError("No supported GPU types found in workload metadata.")

    # Sort by prefix priority, then by model number (descending)
    # This will raise ValueError if any GPU type has an unknown prefix
    available_types.sort(key=_gpu_sort_key)

    return [{'name': gpu_type.upper(), 'value': gpu_type} for gpu_type in available_types]


def get_architecture_choices(gpu_type: str) -> List[Dict[str, str]]:
    """Get architecture choices for a given GPU type.

    Args:
        gpu_type: The selected GPU type (h100, gb200, b200)

    Returns:
        List[Dict[str, str]]: List of architecture choice dictionaries
    """
    return [{'name': "x86_64", 'value': 'x86_64'}, {'name': "aarch64", 'value': 'aarch64'}]


def should_auto_select_architecture(gpu_type: str) -> bool:
    """Check if architecture should be auto-selected for this GPU type.

    Args:
        gpu_type: The selected GPU type

    Returns:
        bool: True if should auto-select aarch64 (GB200), False otherwise
    """
    return DEFAULT_ARCHITECTURE_BY_GPU.get(gpu_type, {'fixed_arch': False}).get('fixed_arch')


def get_default_architecture(gpu_type: str) -> str:
    """Get default architecture for a GPU type.

    Args:
        gpu_type: The selected GPU type

    Returns:
        str: Default architecture ('x86_64' or 'aarch64')
    """
    return DEFAULT_ARCHITECTURE_BY_GPU.get(gpu_type, {'arch': ARCHITECTURES['x86_64']}).get('arch')
