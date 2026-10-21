# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for system configuration persistence."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.config.system import (
    SystemConfig,
    SystemConfigManager,
    load_system_config,
    save_system_config,
    system_config_exists,
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
        selected_workloads=["nemotron", "llama3.1"],
        image_download_method="slurm",
        environment_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
        cache_dirs_configured=True,
    )


@pytest.fixture
def sample_system_config():
    """Create a sample SystemConfig for testing."""
    slurm_config = SlurmConfig(
        account="test_account",
        gpu_partition="gpu_partition",
        cpu_partition="cpu_partition",
        gpu_partition_gres=8,
        cpu_partition_gres=None,
    )

    return SystemConfig(
        venv_type="conda",
        image_download_method="slurm",
        gpu_type="h100",
        node_architecture="x86_64",
        slurm=slurm_config,
        environment_vars={"CUDA_VISIBLE_DEVICES": "0,1,2,3"},
    )


class TestSystemConfig:
    """Test SystemConfig data model."""

    def test_from_install_config_sanitizes_fields(self, sample_install_config):
        """Test that SystemConfig.from_install_config excludes changing fields."""
        system_config = SystemConfig.from_install_config(sample_install_config)

        # Should include stable fields
        assert system_config.venv_type == "conda"
        assert system_config.image_download_method == "slurm"
        assert system_config.gpu_type == "h100"
        assert system_config.node_architecture == "x86_64"
        assert system_config.slurm.account == "test_account"
        assert system_config.environment_vars == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}

        # Should not have changing fields (these aren't in SystemConfig model)
        # This is enforced by the model structure itself

    def test_to_dict_serialization(self, sample_system_config):
        """Test SystemConfig serialization to dictionary."""
        data = sample_system_config.model_dump()

        expected_keys = {
            'venv_type',
            'image_download_method',
            'gpu_type',
            'node_architecture',
            'slurm',
            'kubernetes',
            'environment_vars',
            'image_folder',
            'workload_selection_mode',
        }
        assert set(data.keys()) == expected_keys

        assert data['venv_type'] == "conda"
        assert data['slurm']['account'] == "test_account"
        assert data['environment_vars'] == {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}

    def test_from_dict_deserialization(self, sample_system_config):
        """Test SystemConfig deserialization from dictionary."""
        data = sample_system_config.model_dump()
        restored_config = SystemConfig.model_validate(data)

        assert restored_config.venv_type == sample_system_config.venv_type
        assert restored_config.image_download_method == sample_system_config.image_download_method
        assert restored_config.gpu_type == sample_system_config.gpu_type
        assert restored_config.slurm.account == sample_system_config.slurm.account
        assert restored_config.environment_vars == sample_system_config.environment_vars

    def test_roundtrip_serialization(self, sample_system_config):
        """Test that serialization and deserialization preserves data."""
        data = sample_system_config.model_dump()
        restored_config = SystemConfig.model_validate(data)
        data2 = restored_config.model_dump()

        assert data == data2


class TestSystemConfigManager:
    """Test SystemConfigManager functionality."""

    def test_save_and_load_config(self, temp_config_dir, sample_install_config):
        """Test saving and loading system config."""
        config_path = temp_config_dir / "system_config.yaml"
        manager = SystemConfigManager(str(config_path))

        # Save config
        manager.save(sample_install_config)
        assert config_path.exists()

        # Load config
        loaded_config = manager.load()
        assert loaded_config is not None
        assert loaded_config.venv_type == "conda"
        assert loaded_config.gpu_type == "h100"
        assert loaded_config.slurm.account == "test_account"

    def test_load_nonexistent_config(self, temp_config_dir):
        """Test loading when config file doesn't exist."""
        config_path = temp_config_dir / "nonexistent.yaml"
        manager = SystemConfigManager(str(config_path))

        loaded_config = manager.load()
        assert loaded_config is None

    def test_exists_method(self, temp_config_dir, sample_install_config):
        """Test the exists() method."""
        config_path = temp_config_dir / "system_config.yaml"
        manager = SystemConfigManager(str(config_path))

        # Should not exist initially
        assert not manager.exists()

        # Should exist after saving
        manager.save(sample_install_config)
        assert manager.exists()

    def test_delete_method(self, temp_config_dir, sample_install_config):
        """Test the delete() method."""
        config_path = temp_config_dir / "system_config.yaml"
        manager = SystemConfigManager(str(config_path))

        # Save config
        manager.save(sample_install_config)
        assert config_path.exists()

        # Delete config
        manager.delete()
        assert not config_path.exists()

    def test_file_permissions(self, temp_config_dir, sample_install_config):
        """Test that saved config has proper permissions (0600)."""
        config_path = temp_config_dir / "system_config.yaml"
        manager = SystemConfigManager(str(config_path))

        manager.save(sample_install_config)

        # Check file permissions (should be 0600 = owner read/write only)
        file_stat = os.stat(config_path)
        permissions = file_stat.st_mode & 0o777
        assert permissions == 0o600

    def test_atomic_save(self, temp_config_dir, sample_install_config):
        """Test that save operation is atomic (uses temporary file)."""
        config_path = temp_config_dir / "system_config.yaml"
        manager = SystemConfigManager(str(config_path))

        # Save config - should not leave temporary files
        manager.save(sample_install_config)

        # Check that no .tmp files remain
        tmp_files = list(temp_config_dir.glob("*.tmp"))
        assert len(tmp_files) == 0

        # Main file should exist
        assert config_path.exists()


class TestConvenienceFunctions:
    """Test the convenience functions that use global manager."""

    @patch('llmb_install.config.system._system_config_manager')
    def test_save_system_config(self, mock_manager, sample_install_config):
        """Test save_system_config convenience function."""
        save_system_config(sample_install_config)
        mock_manager.save.assert_called_once_with(sample_install_config)

    @patch('llmb_install.config.system._system_config_manager')
    def test_load_system_config(self, mock_manager):
        """Test load_system_config convenience function."""
        load_system_config()
        mock_manager.load.assert_called_once()

    @patch('llmb_install.config.system._system_config_manager')
    def test_system_config_exists(self, mock_manager):
        """Test system_config_exists convenience function."""
        system_config_exists()
        mock_manager.exists.assert_called_once()


class TestSystemConfigIntegration:
    """Integration tests for system config functionality."""

    def test_full_save_load_cycle(self, temp_config_dir, sample_install_config):
        """Test complete save/load cycle with real files."""
        config_path = temp_config_dir / "system_config.yaml"

        # Replace the global manager with a test instance
        test_manager = SystemConfigManager(str(config_path))
        with patch('llmb_install.config.system._system_config_manager', test_manager):

            # Save using convenience function
            save_system_config(sample_install_config)

            # Load using convenience function
            loaded_config = load_system_config()

            # Verify data integrity
            assert loaded_config.venv_type == sample_install_config.venv_type
            assert loaded_config.gpu_type == sample_install_config.gpu_type
            assert loaded_config.image_download_method == sample_install_config.image_download_method

            # Verify SLURM config
            assert loaded_config.slurm.account == sample_install_config.slurm.account
            assert loaded_config.slurm.gpu_partition == sample_install_config.slurm.gpu_partition

            # Verify environment variables
            assert loaded_config.environment_vars == sample_install_config.environment_vars

    def test_config_excludes_changing_fields(self, temp_config_dir, sample_install_config):
        """Test that changing fields are excluded from system config."""
        config_path = temp_config_dir / "system_config.yaml"
        manager = SystemConfigManager(str(config_path))

        # Save config
        manager.save(sample_install_config)

        # Load and verify excluded fields are not present
        loaded_config = manager.load()

        # SystemConfig should not have these fields from InstallConfig
        assert not hasattr(loaded_config, 'install_path')
        assert not hasattr(loaded_config, 'selected_workloads')
        assert not hasattr(loaded_config, 'venv_path')
        assert not hasattr(loaded_config, 'cache_dirs_configured')
