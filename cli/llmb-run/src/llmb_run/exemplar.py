# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exemplar certification command functionality.

Loads workload configurations from exemplar.yaml, validates against metadata,
and enforces strict install gating.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from llmb_run.config_manager import ClusterConfig
from llmb_run.metadata_utils import model_size_to_billions, normalize_model_dtype_config, parse_workload_name
from llmb_run.task_generation import ValidationError
from llmb_run.tasks import WorkloadTask

logger = logging.getLogger('llmb_run.exemplar')


def load_exemplar_yaml(cluster_config: ClusterConfig) -> Dict[str, Any]:
    """Load and parse exemplar.yaml from llmb_repo.

    Args:
        cluster_config: Cluster configuration containing llmb_repo

    Returns:
        Parsed YAML contents as a dictionary

    Raises:
        ValidationError: If file is missing or cannot be parsed
    """
    llmb_repo = cluster_config.llmb_repo
    if not llmb_repo:
        raise ValidationError("cluster_config.llmb_repo is not set")

    exemplar_path = Path(llmb_repo) / "exemplar.yaml"

    if not exemplar_path.exists():
        raise ValidationError(
            f"exemplar.yaml not found at computed path: {exemplar_path}\n"
            f"(llmb_repo: {llmb_repo})\n"
            "This file is required for 'llmb-run exemplar'."
        )

    try:
        with open(exemplar_path, 'r') as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValidationError(f"Failed to parse exemplar.yaml at {exemplar_path}: {e}") from e
    except Exception as e:
        raise ValidationError(f"Failed to read exemplar.yaml at {exemplar_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValidationError(
            f"exemplar.yaml at {exemplar_path} must contain a YAML mapping, got {type(data).__name__}"
        )

    return data


def validate_exemplar_yaml_schema(yaml_data: Dict[str, Any], gpu_type: str) -> None:
    """Validate the required structure of exemplar.yaml.

    Schema rules:
    - Top-level 'config' and 'workloads' keys are required
    - workloads[gpu_type] must exist and be non-empty
    - Each workload entry must be a single-key mapping
    - Each entry must have 'dtypes' as a non-empty list of strings
    - No duplicate workload+size combinations under the same gpu_type

    Args:
        yaml_data: Parsed YAML data
        gpu_type: GPU type from cluster config

    Raises:
        ValidationError: If schema validation fails
    """
    # Check top-level keys
    if 'config' not in yaml_data:
        raise ValidationError("exemplar.yaml missing required top-level key: 'config'")

    if 'workloads' not in yaml_data:
        raise ValidationError("exemplar.yaml missing required top-level key: 'workloads'")

    workloads = yaml_data['workloads']
    if not isinstance(workloads, dict):
        raise ValidationError(f"exemplar.yaml 'workloads' must be a mapping, got {type(workloads).__name__}")

    # Check gpu_type exists
    if gpu_type not in workloads:
        raise ValidationError(
            f"exemplar.yaml does not contain workloads for gpu_type '{gpu_type}'.\n"
            f"Available gpu_types: {', '.join(sorted(workloads.keys()))}"
        )

    gpu_workloads = workloads[gpu_type]
    if not isinstance(gpu_workloads, list):
        raise ValidationError(
            f"exemplar.yaml workloads['{gpu_type}'] must be a list, got {type(gpu_workloads).__name__}"
        )

    if not gpu_workloads:
        raise ValidationError(f"exemplar.yaml workloads['{gpu_type}'] is empty (must contain at least one workload)")

    # Validate each workload entry
    seen_workloads = set()
    for idx, entry in enumerate(gpu_workloads):
        if not isinstance(entry, dict):
            raise ValidationError(
                f"exemplar.yaml workloads['{gpu_type}'][{idx}] must be a mapping, got {type(entry).__name__}"
            )

        # Each entry must be a single-key mapping
        if len(entry) != 1:
            keys = list(entry.keys()) if entry else []
            raise ValidationError(
                f"exemplar.yaml workloads['{gpu_type}'][{idx}] must have exactly one key (the workload name), "
                f"got {len(entry)} keys: {keys}"
            )

        workload_name = list(entry.keys())[0]
        workload_config = entry[workload_name]

        if not isinstance(workload_config, dict):
            raise ValidationError(
                f"exemplar.yaml workloads['{gpu_type}'][{idx}] entry '{workload_name}' "
                f"must have a mapping as its value, got {type(workload_config).__name__}"
            )

        # Check for required 'dtypes' key
        if 'dtypes' not in workload_config:
            raise ValidationError(
                f"exemplar.yaml workloads['{gpu_type}'][{idx}] entry '{workload_name}' "
                f"missing required key 'dtypes'"
            )

        dtypes = workload_config['dtypes']
        if not isinstance(dtypes, list):
            raise ValidationError(
                f"exemplar.yaml workloads['{gpu_type}'][{idx}] entry '{workload_name}' "
                f"'dtypes' must be a list, got {type(dtypes).__name__}"
            )

        if not dtypes:
            raise ValidationError(
                f"exemplar.yaml workloads['{gpu_type}'][{idx}] entry '{workload_name}' "
                f"'dtypes' list is empty (must contain at least one dtype)"
            )

        # Check for duplicates
        if workload_name in seen_workloads:
            raise ValidationError(f"exemplar.yaml workloads['{gpu_type}'] contains duplicate entry: '{workload_name}'")
        seen_workloads.add(workload_name)


def parse_exemplar_workload_name(workload_name: str) -> Tuple[str, str]:
    """Parse workload name into workload_key and model_size.

    The model_size is the suffix after the last underscore and must match the pattern:
    - \\d+(\\.\\d+)?[bt] (e.g., '7b', '70b', '3.5b', '1t')

    Args:
        workload_name: Full workload name (e.g., 'pretrain_llama3.1_70b')

    Returns:
        Tuple of (workload_key, model_size)

    Raises:
        ValidationError: If workload_name has no underscore or model_size format is invalid
    """
    workload_key, model_size = parse_workload_name(workload_name)

    if model_size is None:
        raise ValidationError(
            f"Invalid workload name '{workload_name}': must contain at least one underscore "
            f"with model size suffix (e.g., 'pretrain_llama3.1_70b' or 'pretrain_kimi-k2_1t')"
        )

    return workload_key, model_size


def get_exemplar_configs_from_yaml(yaml_data: Dict[str, Any], gpu_type: str) -> List[Tuple[str, str, str]]:
    """Extract workload configurations from exemplar.yaml for the specified gpu_type.

    Returns a list of (workload_key, model_size, dtype) tuples.
    Deduplicates exact duplicates to avoid accidental double-submits.

    Args:
        yaml_data: Parsed and validated exemplar.yaml data
        gpu_type: GPU type to extract configs for

    Returns:
        List of unique (workload_key, model_size, dtype) tuples

    Raises:
        ValidationError: If workload name parsing fails
    """
    workloads = yaml_data['workloads'][gpu_type]
    configs = []

    for entry in workloads:
        workload_name = list(entry.keys())[0]
        workload_config = entry[workload_name]

        # Parse workload name to extract workload_key and model_size
        workload_key, model_size = parse_exemplar_workload_name(workload_name)

        # Generate configs for each dtype
        for dtype in workload_config['dtypes']:
            configs.append((workload_key, model_size, dtype))

    # Deduplicate exact duplicates
    unique_configs = list(dict.fromkeys(configs))
    if len(unique_configs) < len(configs):
        logger.debug(f"Deduplicated {len(configs) - len(unique_configs)} duplicate configs")

    return unique_configs


def _extract_numeric_model_size(model_size: str) -> float:
    """Extract model size in billions for sorting.

    Examples:
        "7b" -> 7.0
        "70b" -> 70.0
        "340b" -> 340.0
        "405b" -> 405.0
        "1t" -> 1000.0

    Args:
        model_size: Model size string (e.g., "7b", "70b", "340b", "1t")

    Returns:
        Numeric value for sorting (defaults to 0.0 if no match)
    """
    return model_size_to_billions(model_size)


def validate_yaml_config_against_metadata(
    workload_key: str,
    model_size: str,
    dtype: str,
    scale: int,
    gpu_type: str,
    workloads: Dict,
) -> None:
    """Validate that a YAML-requested config is supported by metadata.

    Validates:
    1. workload_key exists in workloads
    2. gpu_type is supported for this workload
    3. model_size exists for this gpu_type
    4. dtype exists for this model_size
    5. scale is explicitly present in the dtype's scales list

    Args:
        workload_key: Workload identifier (e.g., 'pretrain_llama3.1')
        model_size: Model size (e.g., '70b')
        dtype: Data type (e.g., 'fp8', 'bf16')
        scale: Scale to validate (e.g., 512)
        gpu_type: GPU type (e.g., 'h100', 'gb200')
        workloads: Dictionary of all workloads from get_workloads()

    Raises:
        ValidationError: If validation fails with detailed error message
    """
    # Check workload exists
    if workload_key not in workloads:
        raise ValidationError(
            f"Workload '{workload_key}' requested in exemplar.yaml does not exist.\n"
            f"Available workloads: {', '.join(sorted(workloads.keys()))}"
        )

    workload_data = workloads[workload_key]
    metadata = workload_data.get('metadata', {})
    gpu_configs = metadata.get('run', {}).get('gpu_configs', {})

    # Check gpu_type is supported
    if gpu_type not in gpu_configs:
        available_gpus = ', '.join(sorted(gpu_configs.keys())) if gpu_configs else 'none'
        raise ValidationError(
            f"Workload '{workload_key}' does not support gpu_type '{gpu_type}'.\n"
            f"Available gpu_types for this workload: {available_gpus}"
        )

    gpu_config = gpu_configs[gpu_type]
    model_configs = gpu_config.get('model_configs', [])

    # Find the model_config for this model_size
    matching_model_config = None
    available_sizes = []
    for model_config in model_configs:
        size = model_config.get('model_size')
        if size:
            available_sizes.append(size)
            if size == model_size:
                matching_model_config = model_config
                break

    if not matching_model_config:
        raise ValidationError(
            f"Workload '{workload_key}' does not have model_size '{model_size}' for gpu_type '{gpu_type}'.\n"
            f"Available model_sizes: {', '.join(sorted(available_sizes))}"
        )

    # Use normalize_model_dtype_config to get per-dtype scales
    dtype_map = normalize_model_dtype_config(matching_model_config)

    # Check dtype exists
    if dtype not in dtype_map:
        available_dtypes = ', '.join(sorted(dtype_map.keys())) if dtype_map else 'none'
        raise ValidationError(
            f"Workload '{workload_key}' model_size '{model_size}' does not support dtype '{dtype}' for gpu_type '{gpu_type}'.\n"
            f"Available dtypes: {available_dtypes}"
        )

    # Check scale is explicitly present
    dtype_config = dtype_map[dtype]
    dtype_scales = dtype_config.get('scales', [])
    if scale not in dtype_scales:
        scales_str = ', '.join(map(str, sorted(dtype_scales))) if dtype_scales else 'none'
        raise ValidationError(
            f"Workload '{workload_key}' model_size '{model_size}' dtype '{dtype}' does not explicitly support scale {scale} for gpu_type '{gpu_type}'.\n"
            f"Available scales for this dtype: {scales_str}"
        )


def get_eligible_exemplar_configs_from_yaml(
    yaml_data: Dict[str, Any], gpu_type: str, workloads: Dict
) -> List[Tuple[str, str, str]]:
    """Get eligible configs from exemplar.yaml and validate against metadata.

    For each YAML-requested config:
    1. Parse workload name into workload_key and model_size
    2. Validate against metadata (workload exists, gpu_type supported, model_size exists,
       dtype exists, config.scale explicitly present in scales)
    3. Return validated configs

    Args:
        yaml_data: Parsed and validated exemplar.yaml data
        gpu_type: GPU type from cluster config
        workloads: Dictionary of all workloads from get_workloads()

    Returns:
        List of validated (workload_key, model_size, dtype) tuples

    Raises:
        ValidationError: If any YAML-requested config fails metadata validation
    """
    # Get scale from config (default to 512 if omitted)
    scale = yaml_data.get('config', {}).get('scale', 512)

    # Get configs from YAML
    yaml_configs = get_exemplar_configs_from_yaml(yaml_data, gpu_type)

    # Validate each config against metadata
    for workload_key, model_size, dtype in yaml_configs:
        validate_yaml_config_against_metadata(workload_key, model_size, dtype, scale, gpu_type, workloads)

    return yaml_configs


def validate_strict_installs(eligible_configs: List[Tuple[str, str, str]], cluster_config: ClusterConfig) -> None:
    """Validate that all eligible workloads are installed, with strict error handling.

    Args:
        eligible_configs: List of (workload_key, model_size, dtype) tuples
        cluster_config: Cluster configuration dictionary

    Raises:
        ValidationError: If eligible_configs is empty or if any eligible workloads are missing
    """
    # Check if eligible universe is empty (shouldn't happen with YAML, but defensive)
    if not eligible_configs:
        gpu_type = cluster_config.gpu_type
        msg = (
            f"No eligible workloads found for exemplar certification (gpu_type={gpu_type}).\n"
            "This may indicate an empty or invalid exemplar.yaml configuration."
        )
        raise ValidationError(msg)

    # Get unique eligible workloads
    eligible_workload_keys = {cfg[0] for cfg in eligible_configs}

    # Get installed workloads
    installed_workloads = cluster_config.workloads.installed

    # Edge case: if installed is missing or empty, treat as no workloads installed
    if not installed_workloads:
        missing = sorted(eligible_workload_keys)
        error_lines = [
            f"No workloads are installed. The following {len(missing)} eligible workloads must be installed:"
        ]
        for workload_key in missing:
            error_lines.append(f"  - {workload_key}")
        error_lines.append("\nTo install workloads, run:")
        error_lines.append("  cd $LLMB_INSTALL && llmb-install")
        error_lines.append("Then select the missing workloads listed above.")
        raise ValidationError("\n".join(error_lines))

    # Check for missing workloads
    installed_set = set(installed_workloads)
    missing = sorted(eligible_workload_keys - installed_set)

    if missing:
        error_lines = [
            f"The following {len(missing)} eligible workload(s) are not installed but required for exemplar certification:"
        ]
        for workload_key in missing:
            error_lines.append(f"  - {workload_key}")
        error_lines.append("\nTo install missing workloads, run:")
        error_lines.append("  cd $LLMB_INSTALL && llmb-install")
        error_lines.append("Then select the missing workloads listed above.")
        raise ValidationError("\n".join(error_lines))

    # All eligible workloads are installed
    logger.debug(
        f"All {len(eligible_workload_keys)} eligible workloads are installed: {', '.join(sorted(eligible_workload_keys))}"
    )


def compute_and_validate_eligible_configs(
    workloads: Dict, cluster_config: ClusterConfig
) -> Tuple[List[Tuple[str, str, str]], int, int, bool]:
    """Load YAML configs, validate against metadata, and validate strict install gating.

    This is the main entry point for exemplar preflight checks.

    Args:
        workloads: Dictionary of all workloads from get_workloads()
        cluster_config: Cluster configuration dictionary

    Returns:
        Tuple of (eligible configs list, scale from YAML, repeats from YAML, profile from YAML)
        - eligible configs: List of (workload_key, model_size, dtype) tuples
        - scale: Scale value from YAML config (defaults to 512)
        - repeats: Number of repeats from YAML config (defaults to 1)
        - profile: If True, include exactly one profiled run per config (last repeat). If False, no profiling.

    Raises:
        ValidationError: If validation fails (YAML missing, metadata mismatch, empty universe, or missing installs)
    """
    gpu_type = cluster_config.gpu_type
    if not gpu_type:
        raise ValidationError("No GPU type specified in cluster configuration.")

    # Load and validate YAML
    yaml_data = load_exemplar_yaml(cluster_config)
    validate_exemplar_yaml_schema(yaml_data, gpu_type)

    # Extract config values from YAML
    config = yaml_data.get('config', {})
    scale = config.get('scale', 512)
    repeats = config.get('repeats', 1)
    profile = config.get('profile', False)

    # Get eligible configs from YAML and validate against metadata
    eligible_configs = get_eligible_exemplar_configs_from_yaml(yaml_data, gpu_type, workloads)

    # Validate strict installs
    validate_strict_installs(eligible_configs, cluster_config)

    return eligible_configs, scale, repeats, profile


def generate_exemplar_tasks(
    workloads: Dict, cluster_config: ClusterConfig, repeats: Optional[int] = None
) -> List[WorkloadTask]:
    """Generate WorkloadTask list for exemplar certification.

    Behavior:
    - Load configs from exemplar.yaml for the cluster gpu_type
    - Validate each config against metadata (workload exists, gpu_type supported,
      model_size exists, dtype exists, scale explicitly in dtype's scales list)
    - Validate strict install gating
    - For each eligible (workload_key, model_size, dtype), generate repeats at
      config.scale (default 512)
    - If config.profile is True, the last repeat is profiled and earlier repeats are non-profiled
      (e.g., repeats=3 -> [False, False, True]); if config.profile is False, all repeats are non-profiled
    - Deterministic ordering and clustered by workload_key for submission speed;
      repeats contiguous

    Args:
        workloads: Dictionary of all workloads from get_workloads()
        cluster_config: Cluster configuration dictionary
        repeats: Number of repeats per configuration. If None, uses value from YAML config.repeats (default: 1)

    Returns:
        List of WorkloadTask objects, deterministically ordered:
        - Clustered by workload_key (sorted)
        - Within workload_key, sorted by model_size, then dtype
        - Repeats are contiguous for each (workload_key, model_size, dtype)

    Raises:
        ValidationError: If validation fails (YAML missing, metadata mismatch, empty universe, or missing installs)
    """
    # Get eligible configs, scale, yaml_repeats, and profile (validates YAML, metadata, and installs)
    eligible_configs, scale, yaml_repeats, profile = compute_and_validate_eligible_configs(workloads, cluster_config)

    # Use CLI repeats if provided, otherwise use YAML repeats
    effective_repeats = repeats if repeats is not None else yaml_repeats
    if isinstance(effective_repeats, bool) or not isinstance(effective_repeats, int) or effective_repeats < 1:
        raise ValidationError(
            f"Invalid exemplar repeats value: {effective_repeats!r}. " "Expected an integer greater than or equal to 1."
        )

    # Sort configs for deterministic ordering:
    # - Clustered by workload_key
    # - Within workload_key, sort by model_size (numerically), then dtype
    eligible_configs.sort(key=lambda x: (x[0], _extract_numeric_model_size(x[1]), x[2]))

    # Generate tasks with contiguous repeats
    task_list = []
    for workload_key, model_size, dtype in eligible_configs:
        for repeat_idx in range(effective_repeats):
            # Exemplar profiling policy: at most one profiling run per workload config,
            # and when present it is always the last repeat for deterministic behavior.
            run_profile = bool(profile and repeat_idx == (effective_repeats - 1))
            task_list.append(
                WorkloadTask(
                    workload_key=workload_key,
                    model_size=model_size,
                    dtype=dtype,
                    scale=scale,
                    profile=run_profile,
                )
            )

    profiled_runs = sum(1 for task in task_list if task.profile)
    logger.debug(
        f"Generated {len(task_list)} exemplar tasks "
        f"({len(eligible_configs)} configs × {effective_repeats} repeats, {profiled_runs} profiled)"
    )
    return task_list
