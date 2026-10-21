# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private protocol shared by the ATIF profile launcher and replay controller.

The replay controller needs Python-specific environment overrides to run
inside arbitrary task images. Recorded workload commands must not inherit
those overrides, so the launcher preserves the image values under this
private namespace and the replay process restores them before each action.
"""

CONTROLLER_ENV_VARS_MARKER = "TASKTRACE_CONTROLLER_ENV_VARS"
WORKLOAD_ENV_PREFIX = "TASKTRACE_WORKLOAD_ENV_"
WORKLOAD_ENV_PRESENT_PREFIX = "TASKTRACE_WORKLOAD_ENV_PRESENT_"


def workload_environment_variable(name: str) -> str:
    """Return the private variable used to preserve one task-image value."""
    return f"{WORKLOAD_ENV_PREFIX}{name}"


def workload_environment_presence_variable(name: str) -> str:
    """Return the marker that distinguishes an empty value from an unset one."""
    return f"{WORKLOAD_ENV_PRESENT_PREFIX}{name}"
