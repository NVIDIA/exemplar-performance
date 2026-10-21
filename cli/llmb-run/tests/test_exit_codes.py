# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from unittest.mock import patch


def test_success_exit_code(mock_dependencies, runner, mock_workloads, mocker):
    """Test that a successful command returns exit code 0."""
    from llmb_run.main import app

    mocker.patch('llmb_run.main.run_tests')
    result = runner.invoke(app, ["submit", "-w", "training_foo", "-s", "7b", "-d", "fp16", "--scale", "8"])
    assert result.exit_code == 0


def test_user_error_invalid_workload(mock_dependencies, runner):
    """Test that an invalid workload returns EXIT_VALIDATION_ERROR (1)."""
    from llmb_run.main import EXIT_VALIDATION_ERROR, app

    result = runner.invoke(app, ["submit", "-w", "nonexistent", "-s", "7b", "-d", "fp16", "--scale", "8"])
    assert result.exit_code == EXIT_VALIDATION_ERROR


def test_user_error_missing_config(runner, mocker):
    """Test that missing configuration returns EXIT_VALIDATION_ERROR (1)."""
    from llmb_run.main import EXIT_VALIDATION_ERROR, app

    mocker.patch('llmb_run.main.get_cluster_config', side_effect=FileNotFoundError("Config not found"))
    result = runner.invoke(app, ["list"])
    assert result.exit_code == EXIT_VALIDATION_ERROR


def test_user_error_slurm_env(mock_dependencies, runner):
    """Test that running inside SLURM returns EXIT_VALIDATION_ERROR (1)."""
    from llmb_run.main import EXIT_VALIDATION_ERROR, app

    with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
        result = runner.invoke(app, ["list"])
        assert result.exit_code == EXIT_VALIDATION_ERROR


def test_system_error_launch_failure(mock_dependencies, runner, mock_workloads, mocker):
    """Test that a launch failure returns EXIT_SYSTEM_ERROR (2)."""
    from llmb_run.main import EXIT_SYSTEM_ERROR, app

    # Mock run_tests to raise RuntimeError (which we added to job_launcher.py)
    mocker.patch('llmb_run.main.run_tests', side_effect=RuntimeError("Launch failed"))
    result = runner.invoke(app, ["submit", "-w", "training_foo", "-s", "7b", "-d", "fp16", "--scale", "8"])
    assert result.exit_code == EXIT_SYSTEM_ERROR


def test_list_nonexistent_workload_returns_user_error(mock_dependencies, runner):
    """Test that listing a nonexistent workload returns EXIT_VALIDATION_ERROR (1)."""
    from llmb_run.main import EXIT_VALIDATION_ERROR, app

    result = runner.invoke(app, ["list", "-w", "nonexistent"])
    assert result.exit_code == EXIT_VALIDATION_ERROR
