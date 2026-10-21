# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility import for TaskTrace Capture's Harbor Apptainer environment.

Harbor configuration and existing integrations refer to
``tasktrace_capture.apptainer_env:ApptainerEnvironment``. The implementation lives in
the :mod:`tasktrace_capture.apptainer` package, while this module keeps that public path
stable and re-exports the small set of helpers used by runtime preflight and
cleanup code.
"""

from tasktrace_capture.apptainer.config import (
    APPTAINER_INSTANCE_PREFIX,
    apptainer_binary,
    apptainer_runtime_root,
    clean_apptainer_environment,
)
from tasktrace_capture.apptainer.config import (
    sanitize_instance_name as _sanitize_instance_name,
)
from tasktrace_capture.apptainer.environment import ApptainerEnvironment
from tasktrace_capture.apptainer.installation import (
    ensure_apptainer_installation,
    inspect_apptainer_installation,
)
from tasktrace_capture.apptainer.mounts import mount_spec as _mount_spec

__all__ = [
    "APPTAINER_INSTANCE_PREFIX",
    "ApptainerEnvironment",
    "_mount_spec",
    "_sanitize_instance_name",
    "apptainer_binary",
    "apptainer_runtime_root",
    "clean_apptainer_environment",
    "ensure_apptainer_installation",
    "inspect_apptainer_installation",
]
