# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for express mode functionality."""

import argparse
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llmb_install.cli import parse_arguments
from llmb_install.config.models import InstallConfig, SlurmConfig
from llmb_install.config.system import SystemConfig
from llmb_install.constants import EXIT_CANCELLED, EXIT_ERROR
from llmb_install.core.installer import Installer


@pytest.fixture(autouse=True)
def _uv_available(monkeypatch):
    """Express mode validates uv before collecting fresh install config."""
    monkeypatch.setattr("llmb_install.ui.prompts.environment.is_uv_installed", lambda: True)


@pytest.fixture
def sample_system_config():
    """Create a sample system config."""
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


class TestExpressModeValidation:
    """Test express mode validation and requirements."""

    def test_express_mode_requires_system_config(self):
        """Test that express mode errors when system config doesn't exist."""
        installer = Installer()
        installer.root_dir = "/test/repo"
        installer.workloads = {"test": {"name": "Test"}}

        # Create mock args for express command
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None
        args.install_path_flag = "/test/install"
        args.workloads = "test"
        args.image_folder = None

        # Mock system config to not exist
        with patch('llmb_install.config.system.system_config_exists', return_value=False):
            with patch('llmb_install.config.system.get_system_config_path', return_value=Path("/test/path")):
                with pytest.raises(SystemExit) as exc_info:
                    installer._run_express_mode(args)

                assert exc_info.value.code == 1

    def test_express_mode_succeeds_with_system_config(self, sample_system_config, tmp_path):
        """Test that express mode works when system config exists."""
        installer = Installer()
        installer.root_dir = "/test/repo"
        installer.workloads = {"test": {"name": "Test"}}

        # Create mock args for express command
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None
        args.install_path_flag = str(tmp_path / "install")
        args.workloads = "test"
        args.image_folder = None

        # Create a mock config to return from _collect_express_configuration
        mock_config = InstallConfig(
            install_path=str(tmp_path / "install"),
            gpu_type=sample_system_config.gpu_type,
            node_architecture=sample_system_config.node_architecture,
            venv_type=sample_system_config.venv_type,
            slurm=sample_system_config.slurm,
            environment_vars=sample_system_config.environment_vars,
            selected_workloads=["test"],
            image_download_method=sample_system_config.image_download_method,
            ui_mode='simple',  # Express mode uses simple UI
        )

        # Mock system config to exist and mock configuration collection
        with patch('llmb_install.config.system.system_config_exists', return_value=True):
            with patch.object(installer, '_collect_express_configuration', return_value=mock_config):
                with patch.object(installer, '_confirm_configuration', return_value=True):
                    with patch.object(installer, '_prepare_workloads', return_value=installer.workloads):
                        with patch.object(installer, '_perform_installation') as mock_perform_installation:
                            # Should not raise SystemExit
                            installer._run_express_mode(args)
                            mock_perform_installation.assert_called_once_with(
                                mock_config, installer.workloads, is_resume=False
                            )

    def test_run_express_mode_no_workloads_selected_exits_cancelled(self):
        """User exiting workload selection should return cancellation exit code."""
        installer = Installer()
        args = argparse.Namespace(command='express', list_workloads=False, image_folder=None)

        with patch('llmb_install.config.system.system_config_exists', return_value=True):
            with patch.object(
                installer, '_collect_express_configuration', return_value=SimpleNamespace(selected_workloads=[])
            ):
                with pytest.raises(SystemExit) as exc_info:
                    installer._run_express_mode(args)

        assert exc_info.value.code == EXIT_CANCELLED


class TestExpressModeConfiguration:
    """Test express mode configuration collection."""

    @pytest.fixture
    def mock_args(self, tmp_path):
        """Create mock arguments for express mode."""
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None
        args.install_path_flag = str(tmp_path / "install")
        args.workloads = "pretrain_nemotron-h,pretrain_llama3.1"
        args.image_folder = None
        return args

    def test_express_mode_uses_cli_args(self, mock_args, sample_system_config):
        """Test that express mode uses CLI arguments when provided."""
        installer = Installer()
        installer.root_dir = "/test/repo"
        installer.workloads = {
            "pretrain_nemotron-h": {"name": "Nemotron-H"},
            "pretrain_llama3.1": {"name": "Llama 3.1"},
        }

        with patch('llmb_install.core.installer.load_system_config', return_value=sample_system_config):
            with patch('llmb_install.config.system.system_config_exists', return_value=True):
                with patch.object(installer, '_handle_repository_setup', return_value="/test/repo"):
                    with patch(
                        'llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads
                    ):
                        with patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}):
                            with patch(
                                'llmb_install.core.installer.resolve_gpu_overrides', return_value=installer.workloads
                            ):
                                with patch(
                                    'llmb_install.core.installer.prompt_environment_variables',
                                    return_value=sample_system_config.environment_vars,
                                ) as mock_prompt_environment:
                                    config = installer._collect_express_configuration(mock_args)

        # Should use CLI arguments
        assert config.install_path == mock_args.install_path_flag
        assert config.selected_workloads == ["pretrain_nemotron-h", "pretrain_llama3.1"]

        # Should use system config values, except fresh installs force uv for recipe environments
        assert config.gpu_type == "h100"
        assert config.venv_type == "uv"
        assert config.image_download_method == "slurm"
        assert config.node_architecture == "x86_64"
        mock_prompt_environment.assert_called_once()
        prompt_call = mock_prompt_environment.call_args
        assert prompt_call.kwargs["defaults"] == sample_system_config.environment_vars
        assert prompt_call.kwargs["express_mode"] is True
        assert [prompt.name for prompt in prompt_call.kwargs["prompts"]] == ["HF_TOKEN"]

    def test_express_mode_workloads_all(self, sample_system_config, tmp_path):
        """Test that --workloads all selects all available workloads."""
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None
        args.install_path_flag = str(tmp_path / "install")
        args.workloads = "all"
        args.image_folder = None

        installer = Installer()
        installer.root_dir = "/test/repo"  # Set root_dir for repository operations
        installer.workloads = {
            "pretrain_nemotron-h": {"name": "Nemotron-H"},
            "pretrain_llama3.1": {"name": "Llama 3.1"},
            "grok1": {"name": "Grok"},
        }

        # Ensure filter_workloads_by_gpu_type returns the workloads for the GPU type
        filtered_workloads = dict(installer.workloads.items())  # Return all workloads for test

        with patch('llmb_install.core.installer.load_system_config', return_value=sample_system_config):
            with patch('llmb_install.config.system.system_config_exists', return_value=True):
                with patch.object(installer, '_handle_repository_setup', return_value="/test/repo"):
                    with patch(
                        'llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=filtered_workloads
                    ):
                        with patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}):
                            with patch(
                                'llmb_install.core.installer.resolve_gpu_overrides', return_value=filtered_workloads
                            ):
                                with patch(
                                    'llmb_install.core.installer.prompt_environment_variables',
                                    return_value=sample_system_config.environment_vars,
                                ):
                                    config = installer._collect_express_configuration(args)

        # Should select all workloads
        assert set(config.selected_workloads) == {"pretrain_nemotron-h", "pretrain_llama3.1", "grok1"}

    def test_express_mode_prompts_when_no_cli_args(self, sample_system_config, tmp_path):
        """Test that express mode prompts when CLI args are not provided."""
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None  # No positional arg
        args.install_path_flag = None  # No flag arg
        args.workloads = None  # No CLI arg
        args.image_folder = None

        installer = Installer()
        installer.root_dir = "/test/repo"  # Set root_dir for repository operations
        installer.workloads = {"pretrain_nemotron-h": {"name": "Nemotron-H"}}

        prompted_path = str(tmp_path / "prompted_install")

        with patch('llmb_install.core.installer.load_system_config', return_value=sample_system_config):
            with patch('llmb_install.config.system.system_config_exists', return_value=True):
                with patch('llmb_install.core.installer.prompt_install_location', return_value=prompted_path):
                    with patch(
                        'llmb_install.core.installer.prompt_workload_selection',
                        return_value=(["pretrain_nemotron-h"], "custom"),
                    ):
                        with patch.object(installer, '_handle_repository_setup', return_value="/test/repo"):
                            with patch(
                                'llmb_install.core.installer.filter_workloads_by_gpu_type',
                                return_value=installer.workloads,
                            ):
                                with patch(
                                    'llmb_install.core.installer.filter_tools_from_workload_list', return_value={}
                                ):
                                    with patch(
                                        'llmb_install.core.installer.resolve_gpu_overrides',
                                        return_value=installer.workloads,
                                    ):
                                        with patch(
                                            'llmb_install.core.installer.prompt_environment_variables',
                                            return_value=sample_system_config.environment_vars,
                                        ):
                                            config = installer._collect_express_configuration(args)

        # Should use prompted values
        assert config.install_path == prompted_path
        assert config.selected_workloads == ["pretrain_nemotron-h"]

    def test_collect_express_configuration_prompt_cancel_exits_cancelled(self, sample_system_config):
        """Cancelling install-path prompt should return cancellation exit code."""
        installer = Installer()
        installer.root_dir = "/test/repo"
        installer.workloads = {"pretrain_nemotron-h": {"name": "Nemotron-H"}}

        args = argparse.Namespace(
            command='express',
            install_path_pos=None,
            install_path_flag=None,
            workloads='pretrain_nemotron-h',
            image_folder=None,
            dev_mode=False,
        )

        with patch('llmb_install.core.installer.load_system_config', return_value=sample_system_config):
            with patch('llmb_install.core.installer.prompt_install_location', return_value=None):
                with pytest.raises(SystemExit) as exc_info:
                    installer._collect_express_configuration(args)

        assert exc_info.value.code == EXIT_CANCELLED

    def test_collect_express_configuration_existing_install_exits_error(self, sample_system_config, tmp_path):
        """Express mode should exit with error when pointed at an existing installation."""
        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {"pretrain_nemotron-h": {"name": "Nemotron-H"}}

        args = argparse.Namespace(
            command='express',
            install_path_pos=None,
            install_path_flag=str(tmp_path / "install"),
            workloads='pretrain_nemotron-h',
            image_folder=None,
            dev_mode=False,
        )

        with patch('llmb_install.core.installer.load_system_config', return_value=sample_system_config):
            with patch.object(installer, '_detect_incremental_install', return_value={"workloads": {"installed": []}}):
                with pytest.raises(SystemExit) as exc_info:
                    installer._collect_express_configuration(args)

        assert exc_info.value.code == EXIT_ERROR


class TestCLIIntegration:
    """Test CLI argument parsing for express mode."""

    def test_express_mode_cli_flags(self):
        """Test that express subcommand CLI flags are parsed correctly."""
        # Test express subcommand with --install-path flag
        test_args = [
            'llmb-install',
            'express',
            '--install-path',
            '/test/path',
            '--workloads',
            'pretrain_nemotron-h,pretrain_llama3.1',
        ]

        with patch('sys.argv', test_args):
            args = parse_arguments()

        assert args.command == 'express'
        assert args.install_path_flag == '/test/path'
        assert args.workloads == 'pretrain_nemotron-h,pretrain_llama3.1'

    def test_express_mode_positional_args(self):
        """Test that express subcommand positional argument is parsed correctly."""
        # Test express subcommand with positional install path
        test_args = ['llmb-install', 'express', '/test/path', '--workloads', 'pretrain_nemotron-h,pretrain_llama3.1']

        with patch('sys.argv', test_args):
            args = parse_arguments()

        assert args.command == 'express'
        assert args.install_path_pos == '/test/path'
        assert args.workloads == 'pretrain_nemotron-h,pretrain_llama3.1'

    def test_express_mode_help_text(self, capsys):
        """Test that help text includes express mode information."""
        test_args = ['llmb-install', '--help']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

        captured = capsys.readouterr()

        # Should mention express subcommand in main help
        assert 'express' in captured.out
        assert 'Express installation using saved system configuration' in captured.out

    def test_express_subcommand_help_text(self, capsys):
        """Test that express subcommand help shows specific options."""
        test_args = ['llmb-install', 'express', '--help']

        with patch('sys.argv', test_args):
            with pytest.raises(SystemExit):
                parse_arguments()

        captured = capsys.readouterr()

        # Should show express-specific help
        assert 'Express mode performs minimal-prompt installation' in captured.out
        assert '--install-path' in captured.out
        assert '--workloads' in captured.out
        assert 'install_path_pos' in captured.out


class TestWorkloadSelection:
    """Test workload selection in express mode."""

    def test_comma_separated_workload_parsing(self):
        """Test parsing comma-separated workload lists."""
        workload_string = "pretrain_nemotron-h, pretrain_llama3.1 , grok1,  mistral"

        # This is the logic from _collect_express_configuration
        selected = [w.strip() for w in workload_string.split(',') if w.strip()]

        assert selected == ["pretrain_nemotron-h", "pretrain_llama3.1", "grok1", "mistral"]

    def test_workloads_all_case_insensitive(self):
        """Test that 'all' is case insensitive."""
        test_cases = ["all", "ALL", "All", "aLL"]

        for workloads_arg in test_cases:
            # This mimics the logic in _collect_express_configuration
            is_all = workloads_arg.lower() == 'all'
            assert is_all, f"'{workloads_arg}' should be treated as 'all'"


class TestSystemConfigDefaults:
    """Test that system config provides proper defaults."""

    def test_system_config_used_as_defaults(self, sample_system_config, tmp_path):
        """Test that system config values are used throughout express mode."""
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None
        args.install_path_flag = str(tmp_path / "install")
        args.workloads = "nemotron"
        args.image_folder = None

        installer = Installer()
        installer.root_dir = "/test/repo"  # Set root_dir for repository operations
        installer.workloads = {"nemotron": {"name": "Nemotron"}}

        with patch('llmb_install.core.installer.load_system_config', return_value=sample_system_config):
            with patch('llmb_install.config.system.system_config_exists', return_value=True):
                with patch.object(installer, '_handle_repository_setup', return_value="/test/repo"):
                    with patch(
                        'llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads
                    ):
                        with patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}):
                            with patch(
                                'llmb_install.core.installer.resolve_gpu_overrides',
                                return_value=installer.workloads,
                            ):
                                with patch(
                                    'llmb_install.core.installer.prompt_environment_variables',
                                    return_value=sample_system_config.environment_vars,
                                ):
                                    config = installer._collect_express_configuration(args)

        # System config values should be preserved, except stale venv_type defaults
        # are ignored for fresh express installs.
        assert config.gpu_type == sample_system_config.gpu_type
        assert config.node_architecture == sample_system_config.node_architecture
        assert config.venv_type == "uv"
        assert config.image_download_method == sample_system_config.image_download_method
        assert config.slurm.account == sample_system_config.slurm.account
        assert config.slurm.gpu_partition == sample_system_config.slurm.gpu_partition
        # Environment variables should be merged from system config when prompts return empty values
        assert config.environment_vars == sample_system_config.environment_vars
        assert config.ui_mode == 'simple'  # Express mode uses simple UI


class TestExpressModeExemplar:
    """Test express mode with exemplar workload selection."""

    def _create_exemplar_args(self, tmp_path):
        """Helper to create standard args for exemplar mode testing."""
        args = argparse.Namespace()
        args.command = 'express'
        args.install_path_pos = None
        args.install_path_flag = str(tmp_path / "install")
        args.workloads = None
        args.exemplar = True
        args.image_folder = None
        return args

    def _setup_installer_with_patches(self, installer, tmp_path, system_config):
        """Context manager helper for common installer patches."""
        from contextlib import ExitStack

        filtered_workloads = dict(installer.workloads.items())

        stack = ExitStack()
        stack.enter_context(patch('llmb_install.core.installer.load_system_config', return_value=system_config))
        stack.enter_context(patch('llmb_install.config.system.system_config_exists', return_value=True))
        stack.enter_context(patch.object(installer, '_handle_repository_setup', return_value=str(tmp_path)))
        stack.enter_context(
            patch('llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=filtered_workloads)
        )
        stack.enter_context(patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}))
        stack.enter_context(patch('llmb_install.core.installer.resolve_gpu_overrides', return_value=filtered_workloads))
        stack.enter_context(
            patch(
                'llmb_install.core.installer.prompt_environment_variables',
                return_value=system_config.environment_vars,
            )
        )

        return stack

    def test_express_mode_workloads_exemplar(self, sample_system_config, tmp_path):
        """Test that --exemplar flag uses exemplar.yaml."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {
            "pretrain_llama3.1": {"name": "Llama 3.1"},
            "pretrain_nemotron": {"name": "Nemotron"},
        }

        # Create exemplar.yaml
        (tmp_path / "exemplar.yaml").write_text("""
workloads:
  h100:
    - pretrain_llama3.1_70b
""")

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            config = installer._collect_express_configuration(args)

        assert config.selected_workloads == ["pretrain_llama3.1"]

    def test_express_mode_exemplar_deduplicates_sizes(self, sample_system_config, tmp_path):
        """Test that exemplar workload selection deduplicates size variants."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {"pretrain_llama3.1": {"name": "Llama 3.1"}}

        # Create exemplar.yaml with multiple size variants
        (tmp_path / "exemplar.yaml").write_text("""
workloads:
  h100:
    - pretrain_llama3.1_70b
    - pretrain_llama3.1_405b
""")

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            config = installer._collect_express_configuration(args)

        assert config.selected_workloads == ["pretrain_llama3.1"]

    def test_express_mode_exemplar_ignores_values(self, sample_system_config, tmp_path):
        """Test that exemplar selection ignores YAML values (dtypes, etc.)."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {
            "pretrain_model_a": {"name": "Model A"},
            "pretrain_model_b": {"name": "Model B"},
        }

        # Create exemplar.yaml with various value formats
        (tmp_path / "exemplar.yaml").write_text("""
workloads:
  h100:
    - pretrain_model_a_70b:
        dtypes: []
    - pretrain_model_b_8b:
        garbage: "ignored"
""")

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            config = installer._collect_express_configuration(args)

        assert sorted(config.selected_workloads) == ["pretrain_model_a", "pretrain_model_b"]

    def test_express_mode_exemplar_missing_yaml_exits(self, sample_system_config, tmp_path):
        """Test that express mode exits with error when exemplar.yaml is missing."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)  # No exemplar.yaml created
        installer.workloads = {"pretrain_llama3.1": {"name": "Llama 3.1"}}

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            with pytest.raises(SystemExit) as exc_info:
                installer._collect_express_configuration(args)

        assert exc_info.value.code == 1

    def test_express_mode_exemplar_size_suffix_case_insensitive(self, sample_system_config, tmp_path):
        """Test that express mode accepts case-insensitive size suffixes (70B treated as 70b)."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {"pretrain_llama3.1": {"name": "Llama 3.1"}}

        # Create exemplar.yaml with uppercase suffix (should be accepted)
        (tmp_path / "exemplar.yaml").write_text("""
workloads:
  h100:
    - pretrain_llama3.1_70B
""")

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            config = installer._collect_express_configuration(args)

        assert config.selected_workloads == ["pretrain_llama3.1"]

    def test_express_mode_exemplar_invalid_size_format_exits(self, sample_system_config, tmp_path):
        """Test that express mode exits with error when size suffix format is completely invalid."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {"pretrain_llama3.1": {"name": "Llama 3.1"}}

        # Create exemplar.yaml with completely invalid size suffix
        (tmp_path / "exemplar.yaml").write_text("""
workloads:
  h100:
    - pretrain_llama3.1_invalid
""")

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            with pytest.raises(SystemExit) as exc_info:
                installer._collect_express_configuration(args)

        assert exc_info.value.code == 1

    def test_express_mode_exemplar_workload_not_available_exits(self, sample_system_config, tmp_path):
        """Test that express mode exits when exemplar workload is not in available workloads."""
        args = self._create_exemplar_args(tmp_path)

        installer = Installer()
        installer.root_dir = str(tmp_path)
        installer.workloads = {"pretrain_llama3.1": {"name": "Llama 3.1"}}

        # Create exemplar.yaml with workload not in available workloads
        (tmp_path / "exemplar.yaml").write_text("""
workloads:
  h100:
    - pretrain_nonexistent_70b
""")

        with self._setup_installer_with_patches(installer, tmp_path, sample_system_config):
            with pytest.raises(SystemExit) as exc_info:
                installer._collect_express_configuration(args)

        assert exc_info.value.code == 1
