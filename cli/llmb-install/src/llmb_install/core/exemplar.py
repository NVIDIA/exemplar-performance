# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exemplar workload selection from exemplar.yaml."""

import re
from pathlib import Path
from typing import Any, Dict, List

import yaml


def get_exemplar_workloads(llmb_repo: Path, gpu_type: str) -> List[str]:
    r"""
    Extract workload base keys from exemplar.yaml for a given GPU type.

    Performs lenient parsing of the YAML workloads list:
    - String items are treated as workload_full_name
    - Dict items: all string keys are treated as workload_full_name candidates
    - Other item types are ignored

    Each workload_full_name is parsed by splitting on the last '_':
    - Prefix becomes the base_key (e.g., 'pretrain_llama3.1')
    - Suffix must match ^\d+(\.\d+)?[bt]$ (case-insensitive, e.g., '70b', '3.5b', '1t')

    Args:
        llmb_repo: Path to the LLMB repository root
        gpu_type: GPU type to select workloads for (case-insensitive)

    Returns:
        Sorted list of unique workload base keys (without size suffix)

    Raises:
        ValueError: If YAML file is missing/invalid, gpu_type section is missing/empty,
                   or any workload has an invalid size suffix
    """
    yaml_path = llmb_repo / "exemplar.yaml"

    # Load YAML file
    try:
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise ValueError(f"exemplar.yaml not found at expected path: {yaml_path}\n(llmb_repo={llmb_repo})") from e
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in {yaml_path}: {e}") from e
    except Exception as e:
        raise ValueError(f"Failed to read {yaml_path}: {e}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Invalid exemplar.yaml: root must be a dict, got {type(data).__name__}")

    # Get workloads section
    workloads_section = data.get('workloads')
    if not isinstance(workloads_section, dict):
        raise ValueError(f"Invalid exemplar.yaml: 'workloads' must be a dict, got {type(workloads_section).__name__}")

    # Find GPU type section (case-insensitive)
    gpu_type_lower = gpu_type.lower()
    matched_key = None
    gpu_workloads = None

    for key, value in workloads_section.items():
        if isinstance(key, str) and key.lower() == gpu_type_lower:
            matched_key = key
            gpu_workloads = value
            break

    if matched_key is None:
        available = ', '.join(sorted(workloads_section))
        raise ValueError(
            f"GPU type '{gpu_type}' not found in exemplar.yaml workloads section.\n" f"Available GPU types: {available}"
        )

    if not isinstance(gpu_workloads, list):
        raise ValueError(
            f"Invalid exemplar.yaml: workloads[{matched_key}] must be a list, " f"got {type(gpu_workloads).__name__}"
        )

    if not gpu_workloads:
        raise ValueError(f"workloads[{matched_key}] is empty in exemplar.yaml")

    # Extract workload_full_names with lenient parsing
    workload_full_names = []
    for item in gpu_workloads:
        if isinstance(item, str):
            workload_full_names.append(item)
        elif isinstance(item, dict):
            # Dict keys that are strings are workload_full_name candidates
            workload_full_names.extend(key for key in item if isinstance(key, str))

    if not workload_full_names:
        raise ValueError(
            f"No valid workload entries found in workloads[{matched_key}]. "
            f"Expected string items or dicts with string keys."
        )

    # Parse workload_full_names to extract base keys
    base_keys = []
    size_suffix_pattern = re.compile(r'^\d+(\.\d+)?[bt]$')

    for full_name in workload_full_names:
        if '_' not in full_name:
            raise ValueError(
                f"Invalid workload name '{full_name}' in workloads[{matched_key}]: "
                f"expected format '<workload_key>_<size><unit>' (e.g., 'pretrain_llama3.1_70b' or "
                f"'pretrain_kimi-k2_1t')"
            )

        base_key, size_suffix = full_name.rsplit('_', 1)

        if not base_key:
            raise ValueError(
                f"Invalid workload name '{full_name}' in workloads[{matched_key}]: "
                f"workload key cannot be empty (e.g., '_70b' is invalid)"
            )

        # Validate size suffix
        if not size_suffix_pattern.match(size_suffix.lower()):
            raise ValueError(
                f"Invalid size suffix in '{full_name}' (workloads[{matched_key}]): "
                f"'{size_suffix}' does not match pattern '^\\d+(\\.\\d+)?[bt]$'. "
                f"Examples: '70b', '3.5b', '405b', '1t'"
            )

        base_keys.append(base_key)

    return sorted(set(base_keys))


def validate_exemplar_workloads(base_keys: List[str], available_workloads: Dict[str, Any], gpu_type: str) -> List[str]:
    """Validate that exemplar workloads exist in available workloads.

    Args:
        base_keys: Workload keys from exemplar.yaml
        available_workloads: Dictionary of available workloads
        gpu_type: GPU type for error messages

    Returns:
        List of validated workload keys

    Raises:
        ValueError: If any workloads are missing or list is empty
    """
    missing_keys = [key for key in base_keys if key not in available_workloads]
    if missing_keys:
        available_list = '\n  - '.join(sorted(available_workloads.keys()))
        missing_list = '\n  - '.join(missing_keys)
        raise ValueError(
            f"The following workloads from exemplar.yaml are not available for {gpu_type}:\n"
            f"  - {missing_list}\n\n"
            f"Available workloads:\n"
            f"  - {available_list}"
        )

    if not base_keys:
        raise ValueError(f"No workloads found in exemplar.yaml for GPU type: {gpu_type}")

    return base_keys
