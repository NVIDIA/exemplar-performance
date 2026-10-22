# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
LLMB Install - Modular installer for LLM benchmarking workloads.
"""


def _get_version():
    """Get package version from installed package metadata.

    Requires the package to be installed (via pip install or pip install -e .).
    This is the recommended approach for Python 3.8+ applications.
    """
    try:
        from importlib.metadata import PackageNotFoundError, version

        return version("llmb-install")
    except PackageNotFoundError:
        raise RuntimeError(
            "Package 'llmb-install' is not installed. " "Please install it using 'pip install .' "
        ) from None


__version__ = _get_version()
