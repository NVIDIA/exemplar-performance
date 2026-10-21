# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Cache directory management utilities for LLMB installer."""

import os
from pathlib import Path

from llmb_install.environment.detector import is_uv_installed


def setup_cache_directories(install_path: str, venv_type: str) -> None:
    """Configure cache directories for the selected environment type with fallback defaults.

    Args:
        install_path: The LLMB installation path
        venv_type: The selected virtual environment type ('uv', 'venv', or 'conda')
    """

    # Default cache locations
    default_pip_cache = os.path.join(install_path, '.cache', 'pip')
    default_uv_cache = os.path.join(install_path, '.cache', 'uv')

    # Always handle PIP_CACHE_DIR since all environment types may use pip
    pip_cache_dir = os.environ.get('PIP_CACHE_DIR')
    if not pip_cache_dir:
        os.environ['PIP_CACHE_DIR'] = default_pip_cache
        os.makedirs(default_pip_cache, exist_ok=True)
        print(f"Set PIP_CACHE_DIR to: {default_pip_cache}")
    elif _is_under_home_directory(pip_cache_dir):
        print(f"\nWARNING: PIP_CACHE_DIR is under home directory: {pip_cache_dir}")
        print("This may cause space issues. Consider using a more performant storage.")

    # Handle UV_CACHE_DIR if uv is available (regardless of primary venv_type)
    if venv_type == 'uv' or is_uv_installed():
        uv_cache_dir = os.environ.get('UV_CACHE_DIR')
        if not uv_cache_dir:
            os.environ['UV_CACHE_DIR'] = default_uv_cache
            os.makedirs(default_uv_cache, exist_ok=True)
            print(f"Set UV_CACHE_DIR to: {default_uv_cache}")
        elif _is_under_home_directory(uv_cache_dir):
            print(f"\nWARNING: UV_CACHE_DIR is under home directory: {uv_cache_dir}")
            print("This may cause space issues. Consider using a more performant storage.")


def _is_under_home_directory(path: str) -> bool:
    """Check if a path is under the user's home directory.

    Args:
        path: Path to check

    Returns:
        True if path is under home directory, False otherwise
    """
    try:
        home_dir = Path.home()
        check_path = Path(path).resolve()
        return home_dir in check_path.parents or check_path == home_dir
    except (OSError, ValueError):
        # Fallback to simple string check if Path operations fail
        return path.startswith('/home')
