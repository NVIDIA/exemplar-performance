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

"""Task parsing logic for workload execution."""

import ast
import itertools
import logging
import re

import yaml

from llmb_run.config_manager import ClusterConfig
from llmb_run.env_args import validate_env_key, validate_shell_safe_env_value
from llmb_run.metadata_utils import parse_workload_name
from llmb_run.tasks import WorkloadTask
from llmb_run.workload_validator import (
    format_validation_error,
    validate_workload_with_details,
)

logger = logging.getLogger('llmb_run.task_loader')


def merge_dicts(default, override):
    """Helper to merge two dictionaries (override takes precedence)."""
    result = default.copy()
    result.update(override)
    return result


def get_tasks_simple(workloads, input_file, cluster_config: ClusterConfig | None = None):
    """Parse a simple format task file into workload configurations.

    The simple format is designed for quick task specification with minimal syntax.
    Each workload section starts with a header and contains one or more task lines.

    Format:
    workload_modelsize:
    (dtype_list, scale_list, repeats, profile=False)

    Example:
    pretrain_llama3.1_70b:
    ('bf16', [128, 256], 3)
    # With profiling enabled
    ('bf16', [512], 1, True)

    Note: Inline trailing comments on task lines are not supported.
    Put comments on their own lines instead.

    Args:
        workloads: Dictionary of available workloads
        input_file: Path to the task specification file
        cluster_config: Optional cluster configuration for validation

    Returns:
        dict: Nested dictionary of workload tasks

    Raises:
        FileNotFoundError: If input_file does not exist
        ValueError: If task format is invalid
    """
    header_re = re.compile(r'^([\w\.]+)_([\w\.]+):$')  # Capture workload and model size
    task_re = re.compile(r'^\s*\((.*)\)\s*$')  # Match entire task line

    workload_tasks = {}
    current_workload_key = None
    current_model_size = None
    current_tasks = []

    try:
        with open(input_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:  # Empty line indicates end of current workload section
                    if current_workload_key:
                        workload_tasks.setdefault(current_workload_key, {})[current_model_size] = current_tasks
                    current_workload_key = current_model_size = None
                    current_tasks = []
                    continue

                if header_match := header_re.match(line):
                    workload_key, model_size = header_match.groups()

                    cluster_gpu_type = cluster_config.gpu_type if cluster_config else None
                    is_valid, error_type, error_msg, suggestions = validate_workload_with_details(
                        workloads, workload_key, model_size, cluster_gpu_type, cluster_config
                    )
                    if not is_valid:
                        user_error = format_validation_error(error_type, error_msg, suggestions)
                        logger.error(user_error)
                        logger.error(f"Skipping invalid workload specification: {workload_key}_{model_size}")
                        # Reset state to skip tasks for this invalid workload
                        current_workload_key = None
                        current_model_size = None
                        current_tasks = []
                        continue

                    current_workload_key = workload_key
                    current_model_size = model_size
                    current_tasks = []

                elif task_match := task_re.match(line):
                    if not current_workload_key:
                        # Skip tasks for invalid workloads
                        continue
                    try:
                        task_data = ast.literal_eval(task_match.group(1))
                        current_tasks.append((current_workload_key,) + task_data)
                    except (SyntaxError, ValueError) as e:
                        logger.error(f"Invalid task format in line: {line}")
                        raise ValueError(f"Invalid task format: {e}") from e

        # Handle the last workload if file doesn't end with newline
        if current_workload_key:
            workload_tasks.setdefault(current_workload_key, {})[current_model_size] = current_tasks
    except FileNotFoundError:
        logger.error(f"Input file not found: {input_file}")
        raise
    except Exception as e:
        logger.error(f"Error processing input file: {e}")
        raise
    return workload_tasks


def get_tasks_yaml(input_file):
    """Parse an advanced YAML workload file.

    The expected YAML format is described in the README.md file.

    Args:
        input_file: Path to YAML file
    Returns a dictionary in the same nested format:
      { workload: { model_size: [ list of tuples ] } }
    where each tuple is:
      (workload, model_size, dtype, scale, profile, proxy, env_overrides, model_overrides)
    """
    with open(input_file, 'r') as f:
        data = yaml.safe_load(f)

    workload_tasks = {}
    for header, spec in data.items():
        workload_key, model_size = parse_workload_name(header)

        if model_size is None:
            raise ValueError(
                f"Invalid YAML header: '{header}'\n"
                f"\n"
                f"Missing or invalid model size suffix.\n"
                f"\n"
                f"Expected format: workload_modelsize\n"
                f"  Examples:\n"
                f"    • pretrain_llama3.1_70b:\n"
                f"    • pretrain_nemotron-h_56b:\n"
                f"    • pretrain_kimi-k2_1t:\n"
                f"\n"
                f"Model size must match pattern: _<digits>(b|t) or _<digits>.<digits>(b|t)\n"
                f"\n"
                f"To see available workloads:\n"
                f"  llmb-run list"
            )

        tasks_list = []
        defaults = spec.get("defaults", {})
        default_env = defaults.get("env", {})
        default_params = defaults.get("params", {})

        # Get top level settings
        top_dtypes = spec.get("dtypes")
        top_scales = spec.get("scales")
        top_repeats = spec.get("repeats", 1)
        top_add_profile = spec.get("add_profile", False)

        for task in spec.get("tasks", []):
            # Get task-specific settings, falling back to top level if not specified
            dtypes = task.get("dtypes", top_dtypes)
            if dtypes is None:
                raise ValueError(f"Missing required field 'dtypes' in task: {task}")
            dtypes = [dtypes] if isinstance(dtypes, str) else dtypes
            if not isinstance(dtypes, list):
                raise ValueError(f"'dtypes' must be a string or list, got {type(dtypes)}")

            scales = task.get("scales", top_scales)
            if scales is None:
                raise ValueError(f"Missing required field 'scales' in task: {task}")
            if isinstance(scales, int):
                scales = [scales]
            elif not isinstance(scales, list):
                raise ValueError(f"'scales' must be an integer or list, got {type(scales)}")

            repeats = task.get("repeats", top_repeats)
            profile = task.get("profile", False)
            add_profile = task.get("add_profile", top_add_profile)
            proxy = task.get("proxy", False)

            # Validate that only one profiling mode is set
            if profile and add_profile:
                raise ValueError("Cannot specify both 'profile' and 'add_profile' in the same task")

            overrides = task.get("overrides", {})

            # Env Overrides
            task_env = merge_dicts(default_env, overrides.get("env", {}))
            normalized_task_env = {}
            for _env_key, _env_value in task_env.items():
                env_key = validate_env_key(_env_key, source='YAML env')
                if env_key in normalized_task_env:
                    raise ValueError(f"Duplicate YAML env variable '{env_key}' was specified more than once.")
                validate_shell_safe_env_value(env_key, str(_env_value))
                normalized_task_env[env_key] = _env_value
            task_env = normalized_task_env

            # Model Specific Overrides
            param_overrides = overrides.get("params", {})

            # Find parameters that need to be swept
            regular_params = {}
            sweep_params = {}
            for key, value in param_overrides.items():
                if isinstance(value, list):
                    sweep_params[key] = value
                else:
                    regular_params[key] = value

            if sweep_params:
                # Generate all combinations of sweep parameters
                param_names = list(sweep_params.keys())
                param_values = [sweep_params[name] for name in param_names]
                for param_combination in itertools.product(*param_values):
                    # Create a copy of the base params
                    current_params = merge_dicts(default_params, regular_params)

                    # Apply the current combination of sweep parameters
                    for name, value in zip(param_names, param_combination, strict=True):
                        current_params[name] = value

                    # Generate tasks for this combination
                    for dtype, scale in itertools.product(dtypes, scales):
                        # Add regular performance runs
                        for _r in range(repeats):
                            tasks_list.append(
                                (workload_key, model_size, dtype, scale, profile, proxy, task_env, current_params)
                            )
                        # Add one profiling run if requested
                        if add_profile:
                            tasks_list.append(
                                (workload_key, model_size, dtype, scale, True, proxy, task_env, current_params)
                            )
            else:
                # No sweeps needed, proceed as before
                current_params = default_params.copy()
                current_params.update(regular_params)
                for dtype, scale in itertools.product(dtypes, scales):
                    # Add regular performance runs
                    for _r in range(repeats):
                        tasks_list.append(
                            (workload_key, model_size, dtype, scale, profile, proxy, task_env, current_params)
                        )
                    # Add one profiling run if requested
                    if add_profile:
                        tasks_list.append(
                            (workload_key, model_size, dtype, scale, True, proxy, task_env, current_params)
                        )

        workload_tasks.setdefault(workload_key, {})[model_size] = tasks_list
    return workload_tasks


def get_tasks_wrapper(workloads, input_file, cluster_config: ClusterConfig | None = None):
    """Dispatcher for task parsing based on file extension."""
    if input_file.endswith(('.yaml', '.yml')):
        return get_tasks_yaml(input_file)
    else:
        return get_tasks_simple(workloads, input_file, cluster_config)


def gen_tasks(simple_tasks):
    """Convert parsed simple workload spec into a list of WorkloadTasks.
    This function expects that each task in simple_tasks is a tuple of the form:
      (workload_key, dtype_list, scale_list, repeat_count, *optional_params)
    """
    task_list = []
    for workload_key in simple_tasks:
        for model_size in simple_tasks[workload_key]:
            for task in simple_tasks[workload_key][model_size]:
                workload_key_from_task, dtype, scales, repeats, *params = task
                if isinstance(dtype, str):
                    dtype = [dtype]
                for dt, scale in itertools.product(dtype, scales):
                    for _r in range(repeats):
                        # If optional params exist, the first is profile.
                        profile = params[0] if params else False
                        task_list.append(WorkloadTask(workload_key_from_task, model_size, dt, scale, profile))
    return task_list


def flatten_yaml_tasks(advanced_tasks):
    """Flatten tasks from YAML advanced format to WorkloadTask objects."""
    task_list = []
    for workload in advanced_tasks:
        for model_size in advanced_tasks[workload]:
            for t in advanced_tasks[workload][model_size]:
                # t is a tuple: (workload, model_size, dtype, scale, profile, proxy, env_overrides, model_overrides)
                w, m, dt, scale, profile, proxy, env_overrides, model_overrides = t
                task_list.append(
                    WorkloadTask(
                        w,
                        m,
                        dt,
                        scale,
                        profile,
                        proxy,
                        env_overrides=env_overrides,
                        explicit_env_overrides=env_overrides,
                        model_overrides=model_overrides,
                    )
                )
    return task_list
