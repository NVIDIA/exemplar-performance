# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Virtual environment management utilities for LLMB installer."""

import os
import shutil
import subprocess

from llmb_install.constants import MIN_PYTHON_VERSION
from llmb_install.environment.detector import (
    get_clean_environment_for_subprocess,
)

UV_PYTHON_PREFERENCE_VARS = ('UV_MANAGED_PYTHON', 'UV_NO_MANAGED_PYTHON', 'UV_PYTHON_PREFERENCE')


def setup_uv_python_preference() -> None:
    """Default uv to managed python unless the user set one of uv's own preference variables.

    Conflicting preference settings are an error.
    Applied to os.environ so venv creation, setup tasks and uvx all inherit the same preference.
    """
    if os.environ.get('LLMB_DISABLE_MANAGED_PYTHON', '').lower() in ('1', 'true', 'yes'):
        print("WARNING: LLMB_DISABLE_MANAGED_PYTHON is deprecated; set UV_NO_MANAGED_PYTHON=1 instead.")
        for var in UV_PYTHON_PREFERENCE_VARS:
            os.environ.pop(var, None)
        os.environ['UV_NO_MANAGED_PYTHON'] = '1'

    set_preference_vars = [f"{var}={os.environ[var]}" for var in UV_PYTHON_PREFERENCE_VARS if os.environ.get(var)]
    if len(set_preference_vars) > 1:
        raise ValueError(
            "Conflicting uv python preference variables set: "
            f"{', '.join(set_preference_vars)}. "
            "Set only one (in your shell or the playfile environment_vars)."
        )

    if not any(os.environ.get(var) for var in UV_PYTHON_PREFERENCE_VARS):
        os.environ['UV_MANAGED_PYTHON'] = '1'
    elif os.environ.get('UV_NO_MANAGED_PYTHON') or os.environ.get('UV_PYTHON_PREFERENCE') in ('system', 'only-system'):
        print("WARNING: uv python preference set in environment; managed python not enforced")


def create_virtual_environment(venv_path: str, venv_type: str) -> None:
    """Create a virtual environment at the specified path using the given environment type.

    Args:
        venv_path: Path where the virtual environment should be created
        venv_type: Type of virtual environment to create ('venv', 'conda', or 'uv')

    Raises:
        ValueError: If venv_type is not supported
        subprocess.CalledProcessError: If virtual environment creation fails.
    """
    print(f"Creating virtual environment at {venv_path}...")

    if venv_type == 'venv':
        # Use system python3 instead of current environment's python
        env = get_clean_environment_for_subprocess()
        subprocess.run(
            ['python3', '-m', 'venv', '--clear', venv_path], env=env, check=True, capture_output=True, text=True
        )

    elif venv_type == 'conda':
        # If the conda env path already exists, remove it to prevent errors
        if os.path.exists(venv_path):
            print(f"  Removing existing conda environment at {venv_path}")
            shutil.rmtree(venv_path)

        # Create conda environment with minimum required Python version
        subprocess.run(
            ['conda', 'create', '-p', venv_path, f'python={MIN_PYTHON_VERSION}', '--yes'],
            check=True,
            capture_output=True,
            text=True,
        )

    elif venv_type == 'uv':
        # If the venv path already exists, remove it to prevent errors
        # uv's --clear flag can fail on some filesystems (e.g. shared/network mounts)
        if os.path.exists(venv_path):
            shutil.rmtree(venv_path)

        subprocess.run(
            ['uv', 'venv', '--clear', '--python', MIN_PYTHON_VERSION, venv_path],
            check=True,
            capture_output=True,
            text=True,
        )

        # uv does not install pip in new venvs. Installing it manually handles legacy scripts where 'pip install' is used.
        uv_venv = get_venv_environment(venv_path, 'uv')
        subprocess.run(
            [
                'uv',
                'pip',
                'install',
                'pip',
            ],
            env=uv_venv,
            check=True,
            capture_output=True,
            text=True,
        )
    else:
        raise ValueError(f"Unsupported environment type: {venv_type}")

    print(f"Created virtual environment at: {venv_path}")


def get_venv_environment(venv_path: str, venv_type: str) -> dict:
    """Prepare environment variables for running commands in a virtual environment.

    Args:
        venv_path: Path to the virtual environment
        venv_type: Type of virtual environment ('venv', 'conda', or 'uv')

    Returns:
        dict: Modified environment variables for use with subprocess

    Raises:
        ValueError: If python3 executable is not found in the virtual environment
    """
    env = os.environ.copy()

    bin_dir = os.path.join(venv_path, 'bin')
    python_path = os.path.join(bin_dir, 'python3')
    if not os.path.exists(python_path):
        raise ValueError(f"Invalid virtual environment: python3 executable not found at {python_path}")

    env['PATH'] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop('PYTHONHOME', None)

    if venv_type == 'venv' or venv_type == 'uv':
        env['VIRTUAL_ENV'] = venv_path
    elif venv_type == 'conda':
        env['CONDA_PREFIX'] = venv_path
        # UV uses this to distinguish activated conda envs from system conda installations
        env['CONDA_DEFAULT_ENV'] = venv_path

    return env
