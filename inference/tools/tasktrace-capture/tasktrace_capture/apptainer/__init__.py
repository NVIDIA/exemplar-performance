# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Apptainer runtime components.

The public Harbor plugin remains available from :mod:`tasktrace_capture.apptainer_env`.
Internal modules separate image preparation from container lifecycle so either
side can be replaced independently.
"""

from tasktrace_capture.apptainer.environment import ApptainerEnvironment

__all__ = ["ApptainerEnvironment"]
