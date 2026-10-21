# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for venv_manager functions."""

import pytest

from llmb_install.environment.venv_manager import get_venv_environment


class TestGetVenvEnvironment:
    """Tests for get_venv_environment function."""

    def test_conda_sets_required_env_vars(self, tmp_path):
        """Test that conda venv type sets CONDA_PREFIX and CONDA_DEFAULT_ENV."""
        # Create mock venv with python3
        venv_path = tmp_path / "test_conda_env"
        bin_dir = venv_path / "bin"
        bin_dir.mkdir(parents=True)
        python_exe = bin_dir / "python3"
        python_exe.touch()

        env = get_venv_environment(str(venv_path), 'conda')

        # Verify CONDA_PREFIX is set
        assert 'CONDA_PREFIX' in env
        assert env['CONDA_PREFIX'] == str(venv_path)

        # Verify CONDA_DEFAULT_ENV is set (required for UV to recognize activated conda env)
        assert 'CONDA_DEFAULT_ENV' in env
        assert env['CONDA_DEFAULT_ENV'] == str(venv_path)

        # Verify PATH includes venv bin directory
        assert str(bin_dir) in env['PATH']

    def test_venv_sets_virtual_env(self, tmp_path):
        """Test that venv type sets VIRTUAL_ENV."""
        venv_path = tmp_path / "test_venv"
        bin_dir = venv_path / "bin"
        bin_dir.mkdir(parents=True)
        python_exe = bin_dir / "python3"
        python_exe.touch()

        env = get_venv_environment(str(venv_path), 'venv')

        assert 'VIRTUAL_ENV' in env
        assert env['VIRTUAL_ENV'] == str(venv_path)
        assert 'CONDA_PREFIX' not in env
        assert 'CONDA_DEFAULT_ENV' not in env

    def test_uv_sets_virtual_env(self, tmp_path):
        """Test that uv type sets VIRTUAL_ENV."""
        venv_path = tmp_path / "test_uv_env"
        bin_dir = venv_path / "bin"
        bin_dir.mkdir(parents=True)
        python_exe = bin_dir / "python3"
        python_exe.touch()

        env = get_venv_environment(str(venv_path), 'uv')

        assert 'VIRTUAL_ENV' in env
        assert env['VIRTUAL_ENV'] == str(venv_path)
        assert 'CONDA_PREFIX' not in env
        assert 'CONDA_DEFAULT_ENV' not in env

    def test_missing_python_raises_error(self, tmp_path):
        """Test that missing python3 executable raises ValueError."""
        venv_path = tmp_path / "incomplete_env"
        venv_path.mkdir()

        with pytest.raises(ValueError, match="python3 executable not found"):
            get_venv_environment(str(venv_path), 'conda')
