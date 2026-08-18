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

"""Unified task generation logic for llmb-run."""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from llmb_run.config_manager import ClusterConfig
from llmb_run.metadata_utils import normalize_model_dtype_config, parse_workload_name
from llmb_run.task_loader import (
    flatten_yaml_tasks,
    gen_tasks,
    get_tasks_wrapper,
)
from llmb_run.tasks import WorkloadTask
from llmb_run.workload_validator import (
    format_validation_error,
    validate_workload_with_details,
)

logger = logging.getLogger('llmb_run.task_generation')

if TYPE_CHECKING:
    from llmb_run.slurm_args import SlurmArgs


class ValidationError(Exception):
    """Custom exception for validation errors during task generation."""

    pass


@dataclass
class TaskGenerationRequest:
    """Encapsulates all task generation parameters."""

    workloads: Dict[str, Any]
    cluster_config: ClusterConfig

    # Specification methods
    workload: Optional[str] = None  # Single or comma-separated
    model_size: Optional[str] = None  # Only for explicit mode
    dtype: Optional[str] = None  # Single or comma-separated
    scale: Optional[str] = None  # Single or comma-separated
    max_scale: Optional[int] = None  # For discovery
    min_scale: bool = False  # For discovery
    exact_scales: bool = False  # For discovery - prevent power-of-2 expansion
    file_path: Optional[str] = None  # For file mode

    # Modifiers
    repeats: int = 1
    profile: bool = False
    proxy: bool = False
    force: bool = False
    slurm_args: Optional['SlurmArgs'] = None
    explicit_env_overrides: dict[str, str] = field(default_factory=dict)
    # Legacy Megatron-Bridge argument channel carried through CONFIG_OVERRIDES.
    # Replaced by mbridge_args once all supported launch scripts consume
    # LLMB_MBRIDGE_EXTRA_ARGS.
    extra_workload_args: tuple[str, ...] = ()
    mbridge_args: tuple[str, ...] = ()

    def validate(self) -> None:
        """Validate parameter combinations."""
        # File mode restrictions
        if self.file_path:
            if any([self.workload, self.model_size, self.max_scale, self.min_scale]):
                raise ValidationError(
                    "Cannot mix --file with workload specifications\n"
                    "Either use: llmb-run submit -f my-tests.yaml\n"
                    "Or specify: llmb-run submit -w X -s Y -d Z --scale N"
                )

        # Model size restrictions
        if self.model_size:
            self.model_size = self.model_size.strip().lower()

            if not self.workload:
                raise ValidationError("--model-size requires --workload")

            # If model_size provided, only one workload allowed
            if ',' in (self.workload or ''):
                raise ValidationError(
                    "Global '--model-size' / '-s' cannot be used with multiple workloads.\n\n"
                    "To target specific model sizes, append them to the workload name.\n"
                    "Workloads without a suffix will run ALL available sizes.\n\n"
                    "Example:\n"
                    "  llmb-run submit -w pretrain_llama3.1_70b,pretrain_kimi-k2_1t,pretrain_nemotron-h"
                )

            # Strip redundant size suffix from workload name when -s is also provided.
            # e.g., -w pretrain_llama_7b -s 7b -> workload becomes "pretrain_llama"
            parsed_key, parsed_size = parse_workload_name(self.workload)
            if parsed_size:
                if parsed_size == self.model_size:
                    self.workload = parsed_key
                else:
                    raise ValidationError(
                        f"Workload name implies size '{parsed_size}' but --model-size is '{self.model_size}'.\n"
                        f"Use either: -w {parsed_key} -s {self.model_size}\n"
                        f"        or: -w {self.workload}"
                    )

        # Scale mutual exclusivity
        if self.scale and (self.max_scale or self.min_scale):
            raise ValidationError(
                "Cannot use --scale with --max-scale or --min-scale\n"
                "Either specify exact scale(s): --scale 128,256,512\n"
                "Or use discovery: --max-scale 512"
            )

        # Scale requirement: Always required except in file mode
        if not self.file_path:
            if not any([self.scale, self.max_scale, self.min_scale]):
                raise ValidationError(
                    "Must specify scale parameter\n"
                    "Either specify exact scale(s): --scale 128,256,512\n"
                    "Or use discovery: --max-scale 512 or --min-scale"
                )

        if self.force:
            # Resolve model_size from workload name if not provided explicitly
            has_model_size = self.model_size
            if not has_model_size and self.workload:
                _, parsed_size = parse_workload_name(self.workload)
                has_model_size = parsed_size

            if (
                self.file_path
                or self.max_scale
                or self.min_scale
                or not all((self.workload, has_model_size, self.dtype, self.scale))
            ):
                raise ValidationError(
                    "--force is only supported for a single explicit task.\n"
                    "Use: llmb-run submit -w WORKLOAD -s MODEL_SIZE --dtype DTYPE --scale SCALE --force\n"
                    "  or: llmb-run submit -w WORKLOAD_SIZE --dtype DTYPE --scale SCALE --force"
                )

            multi_value_flags = []
            if len(parse_comma_list(self.workload)) != 1:
                multi_value_flags.append("--workload")
            if self.model_size and len(parse_comma_list(self.model_size)) != 1:
                multi_value_flags.append("--model-size")
            if len(parse_comma_list(self.dtype)) != 1:
                multi_value_flags.append("--dtype")
            if len(parse_comma_list(self.scale)) != 1:
                multi_value_flags.append("--scale")

            if multi_value_flags:
                raise ValidationError(
                    "--force only supports a single explicit task.\n"
                    f"These options must be single values: {', '.join(multi_value_flags)}"
                )


def generate_tasks(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks from unified request specification."""
    try:
        request.validate()
    except ValidationError as e:
        # Re-raise as ValueError to be compatible with typical main.py handling or catch explicitly
        raise ValueError(str(e)) from e

    if request.file_path:
        tasks = _generate_from_file(request)
    elif request.force:
        tasks = _generate_forced_explicit_task(request)
    elif request.model_size:
        # Has explicit model size
        # Always use discovery/targeted mode logic which supports metadata-backed
        # generation, implicit dtypes, and various scale specifications.
        tasks = _generate_explicit_workload_with_scale_discovery(request)
    else:
        # Discovery mode: workload names include size
        tasks = _generate_discovery_tasks(request)

    return _apply_task_generation_modifiers(tasks, request)


def parse_comma_list(value: Optional[str]) -> List[str]:
    """Parse comma-separated list, handling spaces."""
    if not value:
        return []
    # Strip spaces and filter empty strings
    return [item.strip() for item in value.split(',') if item.strip()]


def _generate_explicit_workload_with_scale_discovery(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks for explicit workload with scale discovery.

    Example: llmb-run submit -w pretrain_kimi-k2 -s 1t -d fp8 --max-scale 512
    Generates: pretrain_kimi-k2_1t at all supported scales up to 512
    """
    workload_key = request.workload
    model_size = request.model_size

    # Filter by the specific workload_modelsize combo
    # generate_submit_all_tasks supports filtering by "workload_key" or "workload_key_model_size"
    workload_filter = [f"{workload_key}_{model_size}"]

    dtype_filter = parse_comma_list(request.dtype) if request.dtype else None

    # Handle specific scales if provided
    specific_scales = parse_comma_list(request.scale)
    if specific_scales:
        specific_scales = [int(s) for s in specific_scales]
    else:
        specific_scales = None

    tasks = generate_submit_all_tasks(
        request.workloads,
        request.cluster_config,
        request.max_scale,
        request.repeats,
        request.profile,
        min_scale=request.min_scale,
        exact_scales=request.exact_scales,
        dtype_filter=dtype_filter,
        workload_filter=workload_filter,
        specific_scales=specific_scales,
        slurm_args=request.slurm_args,
        proxy=request.proxy,
    )
    return tasks


def _generate_discovery_tasks(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks using discovery mode (workload names include size)."""
    # Reuse existing generate_submit_all_tasks logic
    workload_filter = parse_comma_list(request.workload) if request.workload else None
    dtype_filter = parse_comma_list(request.dtype) if request.dtype else None

    # If request.scale is provided (specific scales), we pass it.
    # Discovery mode supports --scale 128,256 too (applied to filtered workloads).
    specific_scales = parse_comma_list(request.scale)
    if specific_scales:
        specific_scales = [int(s) for s in specific_scales]
    else:
        specific_scales = None

    tasks = generate_submit_all_tasks(
        request.workloads,
        request.cluster_config,
        request.max_scale,
        request.repeats,
        request.profile,
        min_scale=request.min_scale,
        exact_scales=request.exact_scales,
        dtype_filter=dtype_filter,
        workload_filter=workload_filter,
        specific_scales=specific_scales,
        slurm_args=request.slurm_args,
        proxy=request.proxy,
    )
    return tasks


def _generate_from_file(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate tasks from file specification."""
    # Reuse existing get_tasks_wrapper logic
    tasks_parsed = get_tasks_wrapper(request.workloads, request.file_path, request.cluster_config)

    if request.file_path.endswith(('.yaml', '.yml')):
        tasks = flatten_yaml_tasks(tasks_parsed)
    else:
        tasks = gen_tasks(tasks_parsed)

    if request.slurm_args:
        for task in tasks:
            task.slurm_args = request.slurm_args

    return tasks


def _generate_forced_explicit_task(request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Generate one explicit task while bypassing dtype/scale compatibility checks."""
    if request.model_size:
        workload_key = parse_comma_list(request.workload)[0]
        model_size = parse_comma_list(request.model_size)[0]
    else:
        # Resolve from workload_size name (e.g., "pretrain_foo_1t" -> "pretrain_foo", "1t")
        workload_key, model_size = parse_workload_name(parse_comma_list(request.workload)[0])
    dtype = parse_comma_list(request.dtype)[0]

    try:
        scale = int(parse_comma_list(request.scale)[0])
    except ValueError as e:
        raise ValueError(f"Invalid scale format '{request.scale}'. Scale must be a number.") from e

    cluster_gpu_type = request.cluster_config.gpu_type
    is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
        request.workloads,
        workload_key,
        model_size,
        cluster_gpu_type,
        request.cluster_config,
    )
    if not is_valid:
        raise ValueError(format_validation_error(error_type, error_msg, suggestions))

    tasks = [
        WorkloadTask(
            workload_key=workload_key,
            model_size=model_size,
            dtype=dtype,
            scale=scale,
            profile=request.profile,
            proxy=request.proxy,
            slurm_args=request.slurm_args,
        )
        for _ in range(request.repeats)
    ]
    return tasks


def _apply_task_generation_modifiers(tasks: List[WorkloadTask], request: TaskGenerationRequest) -> List[WorkloadTask]:
    """Apply request-level modifiers to generated tasks."""
    tasks = _apply_explicit_env_overrides(tasks, request.explicit_env_overrides)
    tasks = _apply_extra_workload_args(tasks, request.extra_workload_args)
    tasks = _apply_mbridge_args(tasks, request.mbridge_args)
    return tasks


def _apply_explicit_env_overrides(tasks: List[WorkloadTask], overrides: dict[str, str]) -> List[WorkloadTask]:
    """Apply explicit CLI env vars to generated tasks."""
    if not overrides:
        return tasks

    for task in tasks:
        task.env_overrides = {**task.env_overrides, **overrides}
        task.explicit_env_overrides = {**task.explicit_env_overrides, **overrides}

    return tasks


def _apply_extra_workload_args(tasks: List[WorkloadTask], args: tuple[str, ...]) -> List[WorkloadTask]:
    """Apply request-level extra workload args to generated tasks."""
    if not args:
        return tasks

    for task in tasks:
        task.extra_workload_args = (*task.extra_workload_args, *args)

    return tasks


def _apply_mbridge_args(tasks: List[WorkloadTask], args: tuple[str, ...]) -> List[WorkloadTask]:
    """Apply request-level Megatron-Bridge arguments to generated tasks."""
    if not args:
        return tasks

    for task in tasks:
        task.mbridge_args = (*task.mbridge_args, *args)

    return tasks


def generate_submit_all_tasks(
    workloads,
    cluster_config: ClusterConfig,
    max_scale,
    repeats=1,
    profile=False,
    min_scale=False,
    exact_scales=False,
    dtype_filter=None,
    workload_filter=None,
    specific_scales=None,
    slurm_args: Optional['SlurmArgs'] = None,
    proxy=False,
):
    """Generate tasks for all installed workloads up to max_scale.

    By default (Discovery Mode), only 'pretrain' and 'finetune' workloads are included.
    Other types (e.g., 'inference', 'microbenchmark') can be included by explicitly
    requesting them via `workload_filter`.

    Args:
        workloads: Dictionary of available workloads from get_workloads()
        cluster_config: Cluster configuration dictionary
        max_scale: Maximum scale (number of GPUs) to test up to, or None for metadata scales.
        repeats: Number of repeats for each configuration (default: 1)
        profile: Whether to enable profiling for all tasks (default: False)
        min_scale: If True, only run minimum scale per metadata (default: False)
        exact_scales: If True, only use scales from metadata (no power-of-2 expansion) (default: False)
        dtype_filter: List of dtypes to filter by, or None for all (default: None)
        workload_filter: List of workloads to filter by, or None for all (default: None)
        specific_scales: List of specific scales to run, or None to use max_scale/min_scale logic (default: None)
        slurm_args: Optional canonical Slurm submit args to apply to jobs.
        proxy: If True, use proxy_scales instead of production scales (default: False)

    Returns:
        list: List of WorkloadTask objects for all valid configurations
    """
    # Get configuration details
    installed_workloads = cluster_config.workloads.installed
    cluster_gpu_type = cluster_config.gpu_type

    if not cluster_gpu_type:
        logger.error("No GPU type specified in cluster configuration")
        return []

    max_scale_str = max_scale if max_scale is not None else "metadata scales"
    logger.debug(
        f"Discovering tasks for installed workloads (max_scale: {max_scale_str}, repeats: {repeats}, profile: {profile}, min_scale: {min_scale})"
    )
    if dtype_filter:
        logger.debug(f"Filtering dtypes: {dtype_filter}")
    if workload_filter:
        logger.debug(f"Filtering workloads: {workload_filter}")

    task_list = []
    filtered_workloads = {}

    # Filter workloads by installation status and type
    allowed_types = ['pretrain', 'finetune']
    for workload_key, workload_data in workloads.items():
        if workload_key not in installed_workloads:
            continue

        workload_type = workload_data.get('workload_type', '')
        if not workload_filter and workload_type not in allowed_types:
            logger.debug(f"Skipping {workload_key}: workload_type '{workload_type}' not in {allowed_types}")
            continue

        # Apply workload filter if specified
        if workload_filter:
            # Match a base workload filter or a model-size-specific filter for the same workload.
            workload_matches = False
            for filter_item in workload_filter:
                if workload_key == filter_item or filter_item.startswith(workload_key + '_'):
                    workload_matches = True
                    break
            if not workload_matches:
                logger.debug(f"Skipping {workload_key}: not in workload filter {workload_filter}")
                continue

        filtered_workloads[workload_key] = workload_data

    if not filtered_workloads:
        logger.info("No installed pretrain/finetuning workloads found")
        return []

    logger.debug(f"Found {len(filtered_workloads)} eligible workloads: {', '.join(filtered_workloads.keys())}")

    # Generate tasks for each eligible workload
    for workload_key, workload_data in filtered_workloads.items():
        _generate_workload_tasks(
            workload_key,
            workload_data,
            cluster_gpu_type,
            max_scale,
            repeats,
            profile,
            task_list,
            min_scale,
            runtime_exact_scales=exact_scales,
            dtype_filter=dtype_filter,
            workload_filter=workload_filter,
            specific_scales=specific_scales,
            slurm_args=slurm_args,
            proxy=proxy,
        )

    logger.debug(f"Generated {len(task_list)} tasks across {len(filtered_workloads)} workloads")
    return task_list


def _generate_workload_tasks(
    workload_key,
    workload_data,
    cluster_gpu_type,
    max_scale,
    repeats,
    profile,
    task_list,
    min_scale=False,
    runtime_exact_scales=False,
    dtype_filter=None,
    workload_filter=None,
    specific_scales=None,
    slurm_args: Optional['SlurmArgs'] = None,
    proxy=False,
):
    """Generate tasks for a single workload and add them to task_list.

    Args:
        workload_key: The workload identifier
        workload_data: Workload metadata and configuration
        cluster_gpu_type: GPU type of the cluster
        max_scale: Maximum scale to test up to, or None for metadata scales.
        repeats: Number of repeats per configuration
        profile: Whether to enable profiling
        task_list: List to append generated tasks to
        min_scale: If True, only run minimum scale per metadata (default: False)
        runtime_exact_scales: If True, only use scales from metadata (no power-of-2 expansion) (default: False)
        dtype_filter: List of dtypes to filter by, or None for all (default: None)
        workload_filter: List of workload filters, may include workload_modelsize (default: None)
        specific_scales: List of specific scales to run, or None to use max_scale/min_scale logic (default: None)
        slurm_args: Optional canonical Slurm submit args to apply to jobs.
        proxy: If True, use proxy_scales instead of production scales (default: False)
    """
    metadata = workload_data['metadata']
    gpu_configs = metadata.get('run', {}).get('gpu_configs', {})

    # Check if workload supports the cluster's GPU type
    if cluster_gpu_type not in gpu_configs:
        logger.warning(f"Skipping {workload_key}: no configuration for GPU type '{cluster_gpu_type}'")
        return

    gpu_config = gpu_configs[cluster_gpu_type]
    model_configs = gpu_config.get('model_configs', [])

    # Generate tasks for each model configuration
    for model_config in model_configs:
        model_size = model_config.get('model_size')
        if not model_size:
            continue

        # Apply workload_modelsize filter if specified
        if workload_filter:
            workload_modelsize = f"{workload_key}_{model_size}"
            model_matches = False
            for filter_item in workload_filter:
                if filter_item == workload_key or filter_item == workload_modelsize:
                    model_matches = True
                    break
            if not model_matches:
                logger.debug(f"Skipping {workload_modelsize}: not in workload filter {workload_filter}")
                continue

        # Normalize dtypes to a mapping of dtype -> {scales, exact_scales}
        dtype_map = normalize_model_dtype_config(model_config)

        if not dtype_map:
            logger.warning(f"Skipping {workload_key}_{model_size}: no dtypes defined")
            continue

        # Create tasks for permutations per dtype respecting per-dtype scales
        for dtype, cfg in dtype_map.items():
            # Apply dtype filter if specified
            if dtype_filter and dtype not in dtype_filter:
                logger.debug(f"Skipping {workload_key}_{model_size} dtype={dtype}: not in dtype filter {dtype_filter}")
                continue

            # Select scales based on proxy mode
            if proxy:
                # In proxy mode, use proxy_scales and treat as exact (no power-of-2 expansion)
                dtype_scales = cfg.get('proxy_scales', [])
                effective_exact_scales = True  # Proxy scales are always treated as exact

                # Skip if no proxy_scales defined for this dtype
                if not dtype_scales:
                    logger.debug(f"Skipping {workload_key}_{model_size} dtype={dtype}: no proxy_scales defined")
                    continue
            else:
                # In production mode, use regular scales
                dtype_scales = cfg.get('scales', [])
                metadata_exact_scales = cfg.get('exact_scales', model_config.get('exact_scales', False))
                # Logical OR: if runtime flag is set OR metadata says exact, then use exact scales
                effective_exact_scales = runtime_exact_scales or metadata_exact_scales

                if not dtype_scales:
                    logger.warning(f"Skipping {workload_key}_{model_size} dtype={dtype}: no scales defined")
                    continue

            if specific_scales is not None:
                # Use specific scales, but only those supported by the workload
                scales_to_test = []
                for requested_scale in specific_scales:
                    if effective_exact_scales:
                        # For exact scales, only include scales that are explicitly supported
                        if requested_scale in dtype_scales:
                            scales_to_test.append(requested_scale)
                        else:
                            logger.debug(
                                f"Skipping scale {requested_scale} for {workload_key}_{model_size} dtype={dtype}: not in supported exact scales {dtype_scales}"
                            )
                    else:
                        # For non-exact scales, follow the same logic as max_scale validation
                        if dtype_scales:  # Only validate if scales are defined
                            min_supported_scale = min(dtype_scales)
                            max_tested_scale = max(dtype_scales)

                            if requested_scale < min_supported_scale:
                                logger.debug(
                                    f"Skipping scale {requested_scale} for {workload_key}_{model_size} dtype={dtype}: below minimum supported scale {min_supported_scale}"
                                )
                            elif requested_scale in dtype_scales or requested_scale > max_tested_scale:
                                # Either exact match or above max tested (will get warning in validation)
                                scales_to_test.append(requested_scale)
                            else:
                                logger.debug(
                                    f"Skipping scale {requested_scale} for {workload_key}_{model_size} dtype={dtype}: not supported"
                                )
                        else:
                            # No scale restrictions defined, accept the requested scale
                            scales_to_test.append(requested_scale)
            elif min_scale:
                # If min_scale flag is set, only use the minimum scale (optionally constrained by max_scale)
                if max_scale is not None:
                    min_valid_scale = (
                        min(scale for scale in dtype_scales if scale <= max_scale)
                        if any(scale <= max_scale for scale in dtype_scales)
                        else None
                    )
                    if min_valid_scale is None:
                        logger.debug(
                            f"No valid min scale for {workload_key}_{model_size} dtype={dtype} within max_scale={max_scale}"
                        )
                        continue
                    scales_to_test = [min_valid_scale]
                else:
                    # No max_scale limit, just use the minimum scale from metadata
                    scales_to_test = [min(dtype_scales)]
            else:
                scales_to_test = _generate_scales_up_to_max(dtype_scales, max_scale, effective_exact_scales)

            if not scales_to_test:
                max_scale_str = max_scale if max_scale is not None else "metadata scales"
                logger.debug(
                    f"No valid scales for {workload_key}_{model_size} dtype={dtype} within max_scale={max_scale_str}"
                )
                continue

            for scale in scales_to_test:
                for _ in range(repeats):
                    task_list.append(
                        WorkloadTask(
                            workload_key=workload_key,
                            model_size=model_size,
                            dtype=dtype,
                            scale=scale,
                            profile=profile,
                            proxy=proxy,
                            slurm_args=slurm_args,
                        )
                    )


def _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales):
    """Generate list of scales to test up to max_scale.

    Args:
        metadata_scales: List of scales from metadata file
        max_scale: Maximum scale (number of GPUs) to test, or None for metadata scales.
        exact_scales: If True, only use scales from metadata (up to max)

    Returns:
        list: Sorted list of scales to test
    """
    if not metadata_scales:
        return []

    metadata_scales_int = sorted([int(s) for s in metadata_scales])

    if exact_scales:
        # Only use scales from metadata (optionally up to max)
        if max_scale is not None:
            return [s for s in metadata_scales_int if s <= max_scale]
        else:
            return metadata_scales_int

    # Use all metadata scales up to max, plus power-of-2 scales beyond max metadata scale
    if max_scale is not None:
        scales_to_test = [s for s in metadata_scales_int if s <= max_scale]

        # If max_scale is greater than the highest metadata scale, add power-of-2 scales
        max_metadata_scale = max(metadata_scales_int)
        if max_scale > max_metadata_scale:
            # Find next power of 2 after max_metadata_scale
            next_power = 1
            while next_power <= max_metadata_scale:
                next_power *= 2

            # Add power-of-2 scales up to max_scale
            while next_power <= max_scale:
                scales_to_test.append(next_power)
                next_power *= 2
    else:
        # No max limit, just use metadata scales
        scales_to_test = metadata_scales_int

    return sorted(set(scales_to_test))
