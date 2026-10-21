# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for installation state management (resume functionality)."""

import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.config.system import (
    InstallStateManager,
    clear_install_state,
    install_state_exists,
    load_install_state,
    save_install_state,
)


@pytest.fixture
def temp_config_dir():
    """Create a temporary directory for config tests."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_install_config():
    """Create a sample InstallConfig for testing."""
    slurm_config = SlurmConfig(
        account="test_account",
        gpu_partition="gpu_partition",
        cpu_partition="cpu_partition",
        gpu_partition_gres=8,
        cpu_partition_gres=None,
    )

    return InstallConfig(
        install_path="/test/install/path",
        venv_type="conda",
        gpu_type="h100",
        node_architecture="x86_64",
        slurm=slurm_config,
        selected_workloads=["workload1", "workload2", "workload3"],
        install_method="slurm",
        environment_vars={"HF_TOKEN": "test_token"},
        cache_dirs_configured=True,
        llmb_repo="/test/install/path/llmb_repo",
    )


class TestInstallStateManager:
    """Test the InstallStateManager class for resume functionality."""

    def test_save_install_state(self, temp_config_dir, sample_install_config):
        """Test saving installation state."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/install/path/venvs/shared_venv_abc123"}

        # Save state
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Verify file was created
        assert state_file.exists()

        # Verify contents
        with open(state_file) as f:
            state_data = yaml.safe_load(f)

        assert "install_config" in state_data
        assert "completed_workloads" in state_data
        assert "workload_venvs" in state_data
        assert "timestamp" in state_data

        assert state_data["completed_workloads"] == ["workload1"]
        assert state_data["workload_venvs"] == {"workload1": "/test/install/path/venvs/shared_venv_abc123"}
        assert state_data["install_config"]["install_path"] == "/test/install/path"

    def test_load_install_state(self, temp_config_dir, sample_install_config):
        """Test loading installation state."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        # Use a real temporary directory for install path
        install_path = temp_config_dir / "install"
        install_path.mkdir(parents=True, exist_ok=True)
        sample_install_config.install_path = str(install_path)

        completed_workloads = ["workload1", "workload2"]
        workload_venvs = {
            "workload1": f"{install_path}/venvs/shared_venv_abc123",
            "workload2": f"{install_path}/venvs/shared_venv_abc123",
        }

        # Save state first
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Load state
        result = manager.load_install_state()

        assert result is not None
        loaded_config, loaded_completed, loaded_venvs, loaded_existing_config = result

        # Verify loaded config
        assert isinstance(loaded_config, InstallConfig)
        assert loaded_config.install_path == str(install_path)
        assert loaded_config.gpu_type == sample_install_config.gpu_type
        assert loaded_config.selected_workloads == sample_install_config.selected_workloads

        # Verify completed workloads
        assert loaded_completed == ["workload1", "workload2"]

        # Verify workload venvs
        assert loaded_venvs == workload_venvs

        # Verify existing_cluster_config (should be None for normal install)
        assert loaded_existing_config is None

    def test_load_nonexistent_state(self, temp_config_dir):
        """Test loading state when file doesn't exist."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        result = manager.load_install_state()
        assert result is None

    def test_state_expiration(self, temp_config_dir, sample_install_config):
        """Test that state older than 7 days is considered expired."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/venvs/venv1"}

        # Save state
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Manually modify timestamp to 8 days ago
        with open(state_file) as f:
            state_data = yaml.safe_load(f)

        old_timestamp = datetime.now() - timedelta(days=8)
        state_data["timestamp"] = old_timestamp.isoformat()

        with open(state_file, "w") as f:
            yaml.dump(state_data, f)

        # Try to load - should return None due to expiration
        result = manager.load_install_state()
        assert result is None

    def test_clear_install_state(self, temp_config_dir, sample_install_config):
        """Test clearing installation state."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/venvs/venv1"}

        # Save state
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)
        assert state_file.exists()

        # Clear state
        manager.clear_install_state()

        # Verify file was deleted
        assert not state_file.exists()

    def test_state_exists(self, temp_config_dir, sample_install_config):
        """Test checking if state exists."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        # Initially doesn't exist
        assert not manager.exists()

        # Save state
        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/venvs/venv1"}
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Now exists
        assert manager.exists()

    def test_atomic_save(self, temp_config_dir, sample_install_config):
        """Test that save is atomic (uses temp file + rename)."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/venvs/venv1"}

        # Save state
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Verify no temp files left behind
        temp_files = list(temp_config_dir.glob("install_state.yaml.*"))
        assert len(temp_files) == 0

    def test_install_path_validation(self, temp_config_dir, sample_install_config):
        """Test that state is cleared if install directory no longer exists."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        # Use a non-existent install path
        sample_install_config.install_path = "/nonexistent/install/path"

        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/venvs/venv1"}

        # Save state
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Try to load - should return None because install directory doesn't exist
        result = manager.load_install_state()
        assert result is None

        # State file should have been deleted
        assert not state_file.exists()


class TestConvenienceFunctions:
    """Test the convenience functions for install state management."""

    def test_save_and_load_install_state(self, temp_config_dir, sample_install_config):
        """Test save and load convenience functions."""
        completed_workloads = ["workload1", "workload2"]
        workload_venvs = {
            "workload1": "/test/venvs/shared_venv_abc123",
            "workload2": "/test/venvs/shared_venv_abc123",
        }

        # Mock the module-level singleton manager
        with patch("llmb_install.config.system._install_state_manager") as mock_manager:
            # Test save (now includes 4th parameter: existing_cluster_config=None)
            save_install_state(sample_install_config, completed_workloads, workload_venvs)
            mock_manager.save_install_state.assert_called_once_with(
                sample_install_config, completed_workloads, workload_venvs, None
            )

            # Test load (4-tuple with None for existing_cluster_config)
            mock_manager.load_install_state.return_value = (
                sample_install_config,
                completed_workloads,
                workload_venvs,
                None,
            )
            result = load_install_state()

            assert result is not None
            loaded_config, loaded_completed, loaded_venvs, loaded_existing_config = result
            assert loaded_config == sample_install_config
            assert loaded_completed == completed_workloads
            assert loaded_venvs == workload_venvs
            assert loaded_existing_config is None

    def test_install_state_exists(self, temp_config_dir):
        """Test install_state_exists convenience function."""
        with patch("llmb_install.config.system._install_state_manager") as mock_manager:
            mock_manager.exists.return_value = True

            result = install_state_exists()
            assert result is True

            mock_manager.exists.assert_called_once()

    def test_clear_install_state_func(self, temp_config_dir):
        """Test clear_install_state convenience function."""
        with patch("llmb_install.config.system._install_state_manager") as mock_manager:
            clear_install_state()
            mock_manager.clear_install_state.assert_called_once()


class TestWorkloadCompletionTracking:
    """Test workload completion tracking logic."""

    def test_incremental_completion_tracking(self, temp_config_dir, sample_install_config):
        """Test that workloads can be marked complete incrementally."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        # Use a real temporary directory for install path
        install_path = temp_config_dir / "install"
        install_path.mkdir(parents=True, exist_ok=True)
        sample_install_config.install_path = str(install_path)

        # Start with first workload complete
        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": f"{install_path}/venvs/shared_venv_abc123"}
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Load and verify
        result = manager.load_install_state()
        assert result is not None
        _, loaded_completed, _, _ = result
        assert loaded_completed == ["workload1"]

        # Add second workload
        completed_workloads.append("workload2")
        workload_venvs["workload2"] = f"{install_path}/venvs/shared_venv_abc123"
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Load and verify again
        result = manager.load_install_state()
        assert result is not None
        _, loaded_completed, loaded_venvs, _ = result
        assert loaded_completed == ["workload1", "workload2"]
        assert len(loaded_venvs) == 2

    def test_remaining_workloads_calculation(self, sample_install_config):
        """Test calculating remaining workloads from completed list."""
        selected_workloads = sample_install_config.selected_workloads
        completed_workloads = ["workload1"]

        # Calculate remaining
        remaining = [w for w in selected_workloads if w not in completed_workloads]

        assert remaining == ["workload2", "workload3"]
        assert len(remaining) == 2

    def test_all_workloads_completed(self, sample_install_config):
        """Test edge case where all workloads are completed."""
        selected_workloads = sample_install_config.selected_workloads
        completed_workloads = ["workload1", "workload2", "workload3"]

        # Calculate remaining
        remaining = [w for w in selected_workloads if w not in completed_workloads]

        assert remaining == []
        assert len(remaining) == 0


class TestStateIntegrity:
    """Test state file integrity and error handling."""

    def test_corrupted_state_file(self, temp_config_dir):
        """Test handling of corrupted state file."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        # Create corrupted YAML file
        with open(state_file, "w") as f:
            f.write("invalid: yaml: content:\n  - broken")

        # Loading corrupted file should return None
        result = manager.load_install_state()
        assert result is None

    def test_missing_required_fields(self, temp_config_dir):
        """Test handling of state file with missing required fields."""
        state_file = temp_config_dir / "install_state.yaml"

        # Create state file missing required fields
        incomplete_state = {
            "timestamp": datetime.now().isoformat(),
            # Missing install_config, completed_workloads, workload_venvs
        }

        with open(state_file, "w") as f:
            yaml.dump(incomplete_state, f)

        manager = InstallStateManager(str(state_file))
        result = manager.load_install_state()

        # Should return None for invalid state
        assert result is None

    def test_file_permissions(self, temp_config_dir, sample_install_config):
        """Test that state file has correct permissions."""
        state_file = temp_config_dir / "install_state.yaml"
        manager = InstallStateManager(str(state_file))

        completed_workloads = ["workload1"]
        workload_venvs = {"workload1": "/test/venvs/venv1"}

        # Save state
        manager.save_install_state(sample_install_config, completed_workloads, workload_venvs)

        # Check file permissions (should be readable by user)
        assert state_file.exists()
        assert os.access(state_file, os.R_OK)
        assert os.access(state_file, os.W_OK)
