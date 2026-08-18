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

"""Workload validation and discovery functionality.

Workload Naming Convention:
---------------------------
Throughout this codebase, workloads use a consistent naming structure:

- workload_key: Full identifier in format "{workload_type}_{workload}"
  Example: "pretrain_<workload>"

- workload_type: Category of workload (e.g., "pretrain", "finetune", "inference")
  Example: "pretrain"

- workload: Base workload name
  Example: "<workload>"

The relationship is: workload_key = f"{workload_type}_{workload}"
This format is established when loading metadata and used consistently throughout the tool.
"""

import glob
import logging
import pathlib
from enum import Enum

import yaml

from llmb_run.config_manager import ClusterConfig
from llmb_run.constants import EXCLUDE_WORKLOADS, METADATA_FILE_PATTERN
from llmb_run.metadata_utils import model_size_to_billions, normalize_model_dtype_config

logger = logging.getLogger('llmb_run.workload_validator')


class ValidationErrorType(Enum):
    WORKLOAD_NOT_FOUND = "workload_not_found"
    WORKLOAD_NOT_INSTALLED = "workload_not_installed"
    GPU_TYPE_NOT_SUPPORTED = "gpu_type_not_supported"
    MODEL_SIZE_NOT_SUPPORTED = "model_size_not_supported"
    DTYPE_NOT_SUPPORTED = "dtype_not_supported"
    SCALE_NOT_SUPPORTED = "scale_not_supported"


def get_workloads(config: ClusterConfig):
    """Get a list of all workload metadata files in the repo. Extract relevant data."""
    workloads = {}
    # Get list of metadata files using the new launcher config structure
    llmb_repo = config.llmb_repo
    metadata_files = glob.glob(f"{llmb_repo}/{METADATA_FILE_PATTERN}", recursive=True)

    for meta_file in metadata_files:
        try:
            with open(meta_file, 'r') as f:
                metadata = yaml.safe_load(f)

            # Extract workload name and type from the 'general' section
            workload = metadata.get('general', {}).get('workload')
            workload_type = metadata.get('general', {}).get('workload_type')
            if not workload or not workload_type:
                logger.warning(
                    f"Metadata file {meta_file} is missing 'general.workload' or 'general.workload_type' field. Skipping."
                )
                continue

            # Create the full workload key in format workload_type_workload
            workload_key = f"{workload_type}_{workload}"

            if workload in EXCLUDE_WORKLOADS:
                logger.debug(f"Excluding workload {workload} based on exclude_workloads list.")
                continue

            # Store the entire parsed metadata AND the directory path
            workloads[workload_key] = {
                "metadata": metadata,
                "dir": str(pathlib.Path(meta_file).parent),
                "workload": workload,
                "workload_type": workload_type,
            }

        except FileNotFoundError:
            logger.error(f"Metadata file not found: {meta_file}")
            continue
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML file {meta_file}: {e}")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred processing {meta_file}: {e}")
            continue

    return workloads


def validate_workload_with_details(
    workloads, workload_key, model_size, cluster_gpu_type, cluster_config, dtype=None, scale=None, proxy=False
):
    """Validate workload and return detailed results for better error reporting.

    Args:
        workloads: Dictionary of all workloads
        workload_key: Workload identifier
        model_size: Model size string
        cluster_gpu_type: GPU type of the cluster
        cluster_config: Cluster configuration
        dtype: Optional data type to validate
        scale: Optional scale to validate
        proxy: If True, validate against proxy_scales instead of production scales

    Returns:
        tuple: (is_valid: bool, error_type: ValidationErrorType, error_message: str, suggestions: list)
    """
    installed_workloads = cluster_config.workloads.installed

    if workload_key not in workloads:
        # Filter suggestions by installed workloads
        available_workloads = [w for w in sorted(workloads.keys()) if w in installed_workloads]
        suggestions = [
            w for w in available_workloads if workload_key.lower() in w.lower() or w.lower() in workload_key.lower()
        ]
        if not suggestions:
            suggestions = available_workloads[:5]  # Show first 5 as examples

        error_msg = f"Workload '{workload_key}' not found."
        return False, ValidationErrorType.WORKLOAD_NOT_FOUND, error_msg, suggestions

    if installed_workloads and workload_key not in installed_workloads:
        error_msg = f"Workload '{workload_key}' is not installed on this cluster."
        return False, ValidationErrorType.WORKLOAD_NOT_INSTALLED, error_msg, installed_workloads

    metadata = workloads[workload_key]['metadata']

    gpu_configs = metadata.get('run', {}).get('gpu_configs', {})

    if cluster_gpu_type not in gpu_configs:
        supported_gpu_types = list(gpu_configs.keys())
        error_msg = f"GPU type '{cluster_gpu_type}' not supported for workload '{workload_key}'."
        return False, ValidationErrorType.GPU_TYPE_NOT_SUPPORTED, error_msg, supported_gpu_types

    model_configs = gpu_configs.get(cluster_gpu_type, {}).get('model_configs', [])

    model_config = None
    for config_entry in model_configs:
        if config_entry.get('model_size') == model_size:
            model_config = config_entry
            break

    if not model_config:
        available_sizes = sorted(
            {config.get('model_size') for config in model_configs if config.get('model_size')},
            key=lambda size: (model_size_to_billions(size), size),
        )
        gpu_info = f" for GPU type '{cluster_gpu_type}'"
        error_msg = f"Model size '{model_size}' not supported for workload '{workload_key}'{gpu_info}."
        return False, ValidationErrorType.MODEL_SIZE_NOT_SUPPORTED, error_msg, available_sizes

    # Determine supported dtypes and resolve per-dtype configuration if available.
    dtypes_mapping = normalize_model_dtype_config(model_config)
    if dtypes_mapping:
        supported_dtypes = list(dtypes_mapping.keys())
    else:
        dtypes_value = model_config.get('dtypes', [])
        supported_dtypes = (
            dtypes_value if isinstance(dtypes_value, list) else [dtypes_value] if isinstance(dtypes_value, str) else []
        )
    if dtype is not None:
        if dtype not in supported_dtypes:
            error_msg = f"Data type '{dtype}' not supported for {workload_key}_{model_size}."
            return False, ValidationErrorType.DTYPE_NOT_SUPPORTED, error_msg, supported_dtypes

    if scale is not None:
        # Determine which scale field to validate against based on mode
        scale_field = 'proxy_scales' if proxy else 'scales'

        # Get scales from per-dtype config if available, otherwise from model-level config
        if dtype is not None and dtypes_mapping:
            dtype_cfg = dtypes_mapping.get(dtype, {})
            supported_scales = dtype_cfg.get(scale_field, [])
            exact_scales = True if proxy else dtype_cfg.get('exact_scales', model_config.get('exact_scales', False))
        else:
            supported_scales = model_config.get(scale_field, [])
            exact_scales = True if proxy else model_config.get('exact_scales', False)

        # Error if proxy mode but no proxy scales defined
        if proxy and not supported_scales:
            error_msg = f"No proxy scales defined for {workload_key}_{model_size}."
            return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, []

        try:
            scale_int = int(scale)
            supported_scales_int = [int(s) for s in supported_scales]

            if exact_scales:
                # Exact mode: only allow scales that are explicitly listed
                if scale_int not in supported_scales_int:
                    error_msg = f"Scale {scale} not supported for {workload_key}_{model_size}."
                    return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, supported_scales
            else:
                # Default mode: allow exact matches and scales above maximum (with warning)
                if supported_scales_int:  # Only validate if scales are defined
                    min_scale = min(supported_scales_int)
                    max_tested_scale = max(supported_scales_int)

                    if scale_int < min_scale:
                        error_msg = f"Scale {scale} is below minimum supported scale for {workload_key}_{model_size}."
                        return (
                            False,
                            ValidationErrorType.SCALE_NOT_SUPPORTED,
                            error_msg,
                            [f"minimum: {min_scale}"] + supported_scales,
                        )
                    elif scale_int in supported_scales_int:
                        # Exact match - valid
                        pass
                    elif scale_int > max_tested_scale:
                        logger.warning(
                            f"Scale {scale} for {workload_key}_{model_size} exceeds maximum tested scale of {max_tested_scale}. This configuration has not been validated."
                        )
                    else:
                        error_msg = f"Scale {scale} not supported for {workload_key}_{model_size}."
                        return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, supported_scales

        except (ValueError, TypeError):
            error_msg = f"Invalid scale format '{scale}' for {workload_key}_{model_size}. Scale must be a number."
            return False, ValidationErrorType.SCALE_NOT_SUPPORTED, error_msg, supported_scales

    # If we reached here, validation passed
    logger.debug(f"Validation successful for {workload_key}_{model_size} (dtype: {dtype}, scale: {scale}).")
    return True, None, "", []


def format_validation_error(error_type, error_msg, suggestions):
    """Format a user-friendly validation error message with suggestions."""
    lines = [error_msg]

    if suggestions:
        if error_type == ValidationErrorType.WORKLOAD_NOT_FOUND:
            lines.append(f"Available workloads: {', '.join(suggestions[:10])}")
            if len(suggestions) > 10:
                lines.append("  (use 'llmb-run list' to see all installed workloads)")
        elif error_type == ValidationErrorType.WORKLOAD_NOT_INSTALLED:
            lines.append(f"Installed workloads: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.GPU_TYPE_NOT_SUPPORTED:
            lines.append(f"Supported GPU types: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.MODEL_SIZE_NOT_SUPPORTED:
            lines.append(f"Available model sizes: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.DTYPE_NOT_SUPPORTED:
            lines.append(f"Supported data types: {', '.join(suggestions)}")
        elif error_type == ValidationErrorType.SCALE_NOT_SUPPORTED:
            lines.append(f"Supported scales: {', '.join(map(str, suggestions))}")

    return "\n".join(lines)


def print_avail_workloads(workloads, cluster_config, cluster_gpu_type=None, verbose=False, workload_filter=None):
    """Print available workloads with their supported configurations.

    Args:
        workloads: Dictionary of all workloads
        cluster_config: Cluster configuration
        cluster_gpu_type: Optional GPU type filter
        verbose: Whether to show detailed configuration
        workload_filter: Optional workload key(s) to show - can be string, list, or None
    """
    if not workloads:
        logger.info("No workloads found.")
        return True

    installed_workloads = cluster_config.workloads.installed

    # Filter by workload_filter if specified
    if workload_filter:
        # Normalize to list for consistent handling
        filter_list = [workload_filter] if isinstance(workload_filter, str) else workload_filter

        filtered_workloads = {}
        for wf in filter_list:
            if wf not in workloads:
                logger.error(f"Workload '{wf}' not found.")
                return False
            if installed_workloads and wf not in installed_workloads:
                logger.error(f"Workload '{wf}' is not installed on this cluster.")
                logger.info(f"Installed workloads: {', '.join(installed_workloads)}")
                return False
            filtered_workloads[wf] = workloads[wf]
    else:
        filtered_workloads = {k: v for k, v in workloads.items() if k in installed_workloads}

    if not filtered_workloads:
        logger.info("No installed workloads found.")
        return True

    logger.info("Available workloads:")
    logger.info("=" * 50)

    for workload_key in sorted(filtered_workloads.keys()):
        metadata = filtered_workloads[workload_key]['metadata']
        gpu_configs = metadata.get('run', {}).get('gpu_configs', {})

        # Filter by cluster GPU type if specified
        if cluster_gpu_type and cluster_gpu_type in gpu_configs:
            relevant_configs = {cluster_gpu_type: gpu_configs[cluster_gpu_type]}
        else:
            relevant_configs = gpu_configs

        if not relevant_configs:
            continue

        logger.info(f"\n{workload_key}:")

        # Collect all model sizes, dtypes, and scales across GPU types
        all_model_sizes = set()
        model_details = {}

        for gpu_type, config in relevant_configs.items():
            model_configs = config.get('model_configs', [])
            for model_config in model_configs:
                model_size = model_config.get('model_size')
                if model_size:
                    all_model_sizes.add(model_size)
                    if model_size not in model_details:
                        model_details[model_size] = {
                            'dtypes': set(),
                            'per_dtype_scales': {},  # dtype -> set[int]
                            'per_dtype_exact': {},  # dtype -> bool
                            'per_dtype_proxy_scales': {},  # dtype -> set[int]
                            'gpu_types': set(),
                        }

                    # Normalize per-dtype info for listing
                    dtype_map = normalize_model_dtype_config(model_config)
                    if dtype_map:
                        model_details[model_size]['dtypes'].update(dtype_map.keys())
                        # Track per-dtype scales, proxy_scales, and exactness for display
                        for _dt, cfg in dtype_map.items():
                            scales_ints = [int(s) for s in cfg.get('scales', [])]
                            proxy_scales_ints = [int(s) for s in cfg.get('proxy_scales', [])]
                            model_details[model_size]['per_dtype_scales'].setdefault(_dt, set()).update(scales_ints)
                            model_details[model_size]['per_dtype_proxy_scales'].setdefault(_dt, set()).update(
                                proxy_scales_ints
                            )
                            model_details[model_size]['per_dtype_exact'][_dt] = bool(cfg.get('exact_scales', False))
                    else:
                        dvalues = model_config.get('dtypes', [])
                        if isinstance(dvalues, str):
                            dvalues = [dvalues]
                        model_details[model_size]['dtypes'].update(dvalues)
                        for _dt in dvalues:
                            scales_ints = [int(s) for s in model_config.get('scales', [])]
                            proxy_scales_ints = [int(s) for s in model_config.get('proxy_scales', [])]
                            model_details[model_size]['per_dtype_scales'].setdefault(_dt, set()).update(scales_ints)
                            model_details[model_size]['per_dtype_proxy_scales'].setdefault(_dt, set()).update(
                                proxy_scales_ints
                            )
                            model_details[model_size]['per_dtype_exact'][_dt] = bool(
                                model_config.get('exact_scales', False)
                            )
                    model_details[model_size]['gpu_types'].add(gpu_type)

        if verbose:
            for model_size in sorted(all_model_sizes, key=lambda size: (model_size_to_billions(size), size)):
                details = model_details[model_size]
                logger.info(f"  {model_size}:")
                if details['dtypes']:
                    dtype_list = sorted(details['dtypes'])
                    logger.info(f"    Data types: {', '.join(dtype_list)}")
                    # Decide whether to aggregate or split per dtype
                    per_scales = details['per_dtype_scales']
                    per_exact = details['per_dtype_exact']
                    per_proxy_scales = details['per_dtype_proxy_scales']
                    unique_scales = {tuple(sorted(per_scales.get(dt, set()))) for dt in dtype_list}
                    unique_exact = {bool(per_exact.get(dt, False)) for dt in dtype_list}
                    unique_proxy_scales = {tuple(sorted(per_proxy_scales.get(dt, set()))) for dt in dtype_list}

                    if len(unique_scales) == 1 and len(unique_exact) == 1:
                        # Aggregated output for production scales
                        common_scales_tuple = next(iter(unique_scales))
                        common_scales = sorted(common_scales_tuple, key=int)
                        if not common_scales:
                            logger.info("    Scales: None specified")
                        else:
                            scales_str = ', '.join(map(str, common_scales))
                            if next(iter(unique_exact)):
                                logger.info(f"    Scales: {scales_str} (exact matches only)")
                            else:
                                logger.info(f"    Scales: {scales_str}")

                        # Show proxy scales if available and consistent across dtypes
                        if len(unique_proxy_scales) == 1:
                            common_proxy_scales_tuple = next(iter(unique_proxy_scales))
                            common_proxy_scales = sorted(common_proxy_scales_tuple, key=int)
                            if common_proxy_scales:
                                proxy_scales_str = ', '.join(map(str, common_proxy_scales))
                                logger.info(f"    Proxy scales: {proxy_scales_str}")
                    else:
                        # Split per dtype for clarity
                        logger.info("    Scales by dtype:")
                        for dt in dtype_list:
                            scales_for_dt = sorted(per_scales.get(dt, set()), key=int)
                            if not scales_for_dt:
                                continue
                            exact_suffix = " (exact)" if per_exact.get(dt, False) else ""
                            logger.info(f"      {dt}: {', '.join(map(str, scales_for_dt))}{exact_suffix}")

                        # Show proxy scales per dtype if any exist
                        any_proxy_scales = any(per_proxy_scales.get(dt, set()) for dt in dtype_list)
                        if any_proxy_scales:
                            logger.info("    Proxy scales by dtype:")
                            for dt in dtype_list:
                                proxy_scales_for_dt = sorted(per_proxy_scales.get(dt, set()), key=int)
                                if proxy_scales_for_dt:
                                    logger.info(f"      {dt}: {', '.join(map(str, proxy_scales_for_dt))}")
                else:
                    logger.info("    Data types: None specified")
                if len(details['gpu_types']) > 1 or not cluster_gpu_type:
                    logger.info(f"    GPU types: {', '.join(sorted(details['gpu_types']))}")
        else:
            model_sizes = sorted(all_model_sizes, key=lambda size: (model_size_to_billions(size), size))
            logger.info(f"  Model sizes: {', '.join(model_sizes)}")

    if not verbose:
        logger.info("\nUse --verbose to see detailed configuration for each workload.")
    logger.info("")

    return True
