# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the exemplar certification command."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from llmb_run.config_manager import ClusterConfig


@pytest.fixture
def mock_exemplar_cluster_config(tmp_path):
    """Cluster configuration for exemplar tests with llmb_repo pointing to tmp_path."""
    return {
        "launcher": {
            "llmb_repo": str(tmp_path),
            "llmb_install": str(tmp_path),
            "gpu_type": "h100",
        },
        "slurm": {
            "gpu_partition": "batch",
        },
        "workloads": {
            "installed": ["pretrain_workload_a", "pretrain_workload_b"],
        },
    }


@pytest.fixture
def mock_exemplar_workloads():
    """Workloads with dtype-specific scale configurations for exemplar tests.

    pretrain_workload_a:
        - model_size: 7b
            - fp8: scales [256, 512, 1024]
            - bf16: scales [256, 1024]  # Note: 512 NOT included

    pretrain_workload_b:
        - model_size: 13b
            - fp8: scales [512]
            - bf16: scales [512]
    """
    return {
        "pretrain_workload_a": {
            "metadata": {
                "run": {
                    "launcher_type": "sbatch",
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": {
                                        "fp8": {"scales": [256, 512, 1024]},
                                        "bf16": {"scales": [256, 1024]},  # 512 NOT included
                                    },
                                }
                            ]
                        }
                    },
                }
            },
            "dir": "/tmp",
            "workload": "workload_a",
            "workload_type": "pretrain",
        },
        "pretrain_workload_b": {
            "metadata": {
                "run": {
                    "launcher_type": "sbatch",
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "13b",
                                    "dtypes": {
                                        "fp8": {"scales": [512]},
                                        "bf16": {"scales": [512]},
                                    },
                                }
                            ]
                        }
                    },
                }
            },
            "dir": "/tmp",
            "workload": "workload_b",
            "workload_type": "pretrain",
        },
    }


def write_exemplar_yaml(tmp_path, yaml_content):
    """Helper to write exemplar.yaml to tmp_path."""
    exemplar_path = tmp_path / "exemplar.yaml"
    with open(exemplar_path, 'w') as f:
        yaml.dump(yaml_content, f)


@pytest.fixture
def mock_exemplar_dependencies(mock_exemplar_cluster_config, mock_exemplar_workloads):
    """Patch dependencies for exemplar tests.

    Uses side_effect (not return_value) so that tests which mutate mock_exemplar_cluster_config
    before invoking the CLI command get a freshly built ClusterConfig that reflects those mutations.
    """

    def _always_valid(*a, **k):
        return (True, None, "", [])

    def _get_cluster_config():
        return ClusterConfig.from_dict(mock_exemplar_cluster_config)

    with (
        patch('llmb_run.main.get_cluster_config', side_effect=_get_cluster_config),
        patch('llmb_run.main.get_workloads', return_value=mock_exemplar_workloads),
        patch('llmb_run.main.validate_workload_with_details', side_effect=_always_valid),
    ):
        yield


@pytest.mark.usefixtures("mock_exemplar_dependencies")
class TestExemplarCommand:
    """Tests for the exemplar certification command."""

    def test_exemplar_missing_yaml_exits_with_path(self, runner, caplog, mock_exemplar_cluster_config):
        """Test missing exemplar.yaml -> exit 1 with expected path."""
        from llmb_run.main import app

        tmp_path = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])

        # Don't create exemplar.yaml - it should be missing
        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "exemplar.yaml not found" in caplog.text
        # Check that the expected path is mentioned
        expected_path = tmp_path / "exemplar.yaml"
        assert str(expected_path) in caplog.text
        assert "llmb_repo" in caplog.text

    def test_exemplar_missing_config_key_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test missing 'config' top-level key -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
            # Missing 'config' key
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "missing required top-level key: 'config'" in caplog.text

    def test_exemplar_missing_workloads_key_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test missing 'workloads' top-level key -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            # Missing 'workloads' key
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "missing required top-level key: 'workloads'" in caplog.text

    def test_exemplar_missing_gpu_type_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test missing workloads[gpu_type] -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"gb200": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},  # Only gb200, not h100
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "does not contain workloads for gpu_type 'h100'" in caplog.text

    def test_exemplar_empty_gpu_workloads_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test empty workloads[gpu_type] list -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": []},  # Empty list
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "workloads['h100'] is empty" in caplog.text

    def test_exemplar_missing_dtypes_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test missing dtypes key -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {}}]},  # Missing 'dtypes' key
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "missing required key 'dtypes'" in caplog.text

    def test_exemplar_empty_dtypes_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test empty dtypes list -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": []}}]},  # Empty dtypes list
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "'dtypes' list is empty" in caplog.text

    def test_exemplar_multiple_keys_in_entry_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test list item has multiple keys -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {
                "h100": [
                    {
                        "pretrain_workload_a_7b": {"dtypes": ["fp8"]},
                        "pretrain_workload_b_13b": {"dtypes": ["bf16"]},  # Multiple keys in same entry
                    }
                ]
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "must have exactly one key" in caplog.text

    def test_exemplar_duplicate_workload_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test duplicate workload+size under gpu -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {
                "h100": [
                    {"pretrain_workload_a_7b": {"dtypes": ["fp8"]}},
                    {"pretrain_workload_a_7b": {"dtypes": ["bf16"]}},  # Duplicate workload name
                ]
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "duplicate entry" in caplog.text
        assert "pretrain_workload_a_7b" in caplog.text

    def test_exemplar_workload_name_no_underscore_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test workload name without underscore -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"workloadwithoutsize": {"dtypes": ["fp8"]}}]},  # No underscore
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "must contain at least one underscore" in caplog.text

    def test_exemplar_invalid_model_size_suffix_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test invalid model size suffix (after last _) -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {
                "h100": [{"pretrain_workload_a_7billion": {"dtypes": ["fp8"]}}]  # Invalid suffix: '7billion' not '7b'
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        # Check that error mentions the invalid workload name and suffix requirement
        assert "Invalid workload name" in caplog.text or "Invalid model size suffix" in caplog.text
        assert "7billion" in caplog.text
        assert "model size suffix" in caplog.text

    def test_exemplar_workload_not_found_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test workload not in metadata -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_nonexistent_7b": {"dtypes": ["fp8"]}}]},  # Workload doesn't exist
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "does not exist" in caplog.text
        assert "pretrain_nonexistent" in caplog.text

    def test_exemplar_gpu_type_not_supported_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test workload exists but doesn't support cluster gpu_type -> exit 1."""
        from llmb_run.main import app

        # Change cluster to a100, but workloads only support h100
        mock_exemplar_cluster_config["launcher"]["gpu_type"] = "a100"

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"a100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "does not support gpu_type 'a100'" in caplog.text

    def test_exemplar_model_size_not_found_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test model_size not in metadata -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_99b": {"dtypes": ["fp8"]}}]},  # 99b doesn't exist
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "does not have model_size '99b'" in caplog.text

    def test_exemplar_dtype_not_in_metadata_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test dtype not in metadata -> exit 1."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp32"]}}]},  # fp32 not supported
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "does not support dtype 'fp32'" in caplog.text

    def test_exemplar_scale_not_in_dtype_scales_exits_with_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test config.scale not explicitly listed for dtype -> exit 1."""
        from llmb_run.main import app

        # pretrain_workload_a 7b bf16 has scales [256, 1024] but NOT 512
        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["bf16"]}}]},  # bf16 doesn't have 512
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit 1, got {result.exit_code}"
        assert "does not explicitly support scale 512" in caplog.text
        assert "bf16" in caplog.text

    def test_exemplar_missing_installs_exits_with_error(
        self, runner, caplog, mock_exemplar_cluster_config, mock_exemplar_workloads
    ):
        """Test strict install gating: YAML lists workload not installed -> exit 1 with install instructions."""
        from llmb_run.main import app

        # Modify cluster config to have only pretrain_workload_a installed, not pretrain_workload_b
        mock_exemplar_cluster_config["workloads"]["installed"] = ["pretrain_workload_a"]

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {
                "h100": [
                    {"pretrain_workload_a_7b": {"dtypes": ["fp8"]}},
                    {"pretrain_workload_b_13b": {"dtypes": ["fp8"]}},  # Not installed
                ]
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit code 1, got {result.exit_code}"
        assert "not installed but required for exemplar certification" in caplog.text
        assert "pretrain_workload_b" in caplog.text
        assert "cd $LLMB_INSTALL && llmb-install" in caplog.text

    def test_exemplar_happy_path_submits_yaml_workloads(self, runner, caplog, mock_exemplar_cluster_config):
        """Test happy path: only YAML-listed workloads are submitted with expected repeat/profile mix."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 2, "profile": True},
            "workloads": {
                "h100": [
                    {"pretrain_workload_a_7b": {"dtypes": ["fp8"]}},  # This one has 512
                    {"pretrain_workload_b_13b": {"dtypes": ["fp8", "bf16"]}},  # Both have 512
                ]
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {caplog.text}"

        # Expected: 3 configs (a_7b fp8, b_13b fp8, b_13b bf16) × 2 repeats = 6 tasks
        assert len(recorded_tasks) == 6, f"Expected 6 tasks, got {len(recorded_tasks)}"

        # Verify all tasks have correct scale
        for task in recorded_tasks:
            assert task.scale == 512, f"Task {task} should have scale=512"

        # profile=True means one profiled run (the last repeat) per workload/model/dtype config
        profile_tasks = [t for t in recorded_tasks if t.profile]
        assert len(profile_tasks) == 3, f"Expected 3 profiled runs (1 per config), got {len(profile_tasks)}"

        grouped_profiles = {}
        for task in recorded_tasks:
            key = (task.workload_key, task.model_size, task.dtype)
            grouped_profiles.setdefault(key, []).append(task.profile)

        assert grouped_profiles[("pretrain_workload_a", "7b", "fp8")] == [False, True]
        assert grouped_profiles[("pretrain_workload_b", "13b", "bf16")] == [False, True]
        assert grouped_profiles[("pretrain_workload_b", "13b", "fp8")] == [False, True]

        # Verify specific workload/dtype combinations
        workload_a_tasks = [t for t in recorded_tasks if t.workload_key == "pretrain_workload_a"]
        assert len(workload_a_tasks) == 2, "Expected 2 tasks for workload_a (1 config × 2 repeats)"
        assert all(t.dtype == "fp8" and t.model_size == "7b" for t in workload_a_tasks)

        workload_b_tasks = [t for t in recorded_tasks if t.workload_key == "pretrain_workload_b"]
        assert len(workload_b_tasks) == 4, "Expected 4 tasks for workload_b (2 configs × 2 repeats)"

        # Verify output mentions jobs
        assert "Exemplar Certification Jobs" in caplog.text
        assert "(6)" in caplog.text

    def test_exemplar_happy_path_profile_false(self, runner, caplog, mock_exemplar_cluster_config):
        """Test that profile=False from YAML is respected."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 1, "profile": False},  # profile=False
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {caplog.text}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].profile is False, "Task should have profile=False from YAML"

    def test_exemplar_omitted_repeat_and_profile_use_non_profiled_single_repeat(
        self, runner, caplog, mock_exemplar_cluster_config
    ):
        """Test that omitted repeat/profile config falls back to one non-profiled task."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {caplog.text}"
        assert len(recorded_tasks) == 1
        assert recorded_tasks[0].profile is False

    def test_exemplar_cli_repeats_overrides_yaml(self, runner, caplog, mock_exemplar_cluster_config):
        """Test that CLI --repeats overrides YAML config.repeats."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 2, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["exemplar", "-r", "5"])  # CLI override: 5 repeats

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {caplog.text}"

        # Expected: 1 config × 5 repeats = 5 tasks (CLI overrides YAML's 2)
        assert len(recorded_tasks) == 5, f"Expected 5 tasks (CLI override), got {len(recorded_tasks)}"
        assert [t.profile for t in recorded_tasks] == [False, False, False, False, True]

    def test_exemplar_cli_repeats_zero_exits_with_validation_error(self, runner, mock_exemplar_cluster_config):
        """Test that CLI --repeats 0 is rejected by argument parsing."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar", "-r", "0"])

        assert result.exit_code == 2, f"Expected exit code 2, got {result.exit_code}"
        assert "Invalid value for '-r' / '--repeats'" in result.output

    def test_exemplar_yaml_repeats_zero_exits_with_validation_error(self, runner, caplog, mock_exemplar_cluster_config):
        """Test that YAML config.repeats=0 is rejected with a clear validation error."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 0, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 1, f"Expected exit code 1, got {result.exit_code}"
        assert "Invalid exemplar repeats value" in caplog.text

    def test_exemplar_dry_run_does_not_submit(self, runner, caplog, mock_exemplar_cluster_config):
        """Test that --dry-run does not call run_tests."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {"h100": [{"pretrain_workload_a_7b": {"dtypes": ["fp8"]}}]},
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        run_tests_called = False

        def _fake_run_tests(*_args, **_kwargs):
            nonlocal run_tests_called
            run_tests_called = True

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["exemplar", "--dry-run"])

        assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {caplog.text}"
        assert run_tests_called is False, "run_tests should not be called in dry-run mode"
        assert "Dry run enabled" in caplog.text

    def test_exemplar_deterministic_ordering(self, runner, mock_exemplar_cluster_config):
        """Test that tasks are generated in deterministic order: workload -> model_size (numeric) -> dtype."""
        from llmb_run.main import app

        # Mock workloads with varied sizes and dtypes to test sorting
        mock_workloads = {
            "workload_b": {
                "workload_type": "pretrain",
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "8b",
                                        "dtypes": {"fp8": {"scales": [512]}},
                                    }
                                ]
                            }
                        }
                    }
                },
            },
            "workload_a": {
                "workload_type": "pretrain",
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "70b",
                                        "dtypes": {"fp8": {"scales": [512]}},
                                    },
                                    {
                                        "model_size": "1t",
                                        "dtypes": {"fp8": {"scales": [512]}},
                                    },
                                    {
                                        "model_size": "7b",
                                        "dtypes": {
                                            "bf16": {"scales": [512]},
                                            "fp8": {"scales": [512]},
                                        },
                                    },
                                ]
                            }
                        }
                    }
                },
            },
        }

        mock_exemplar_cluster_config["workloads"]["installed"] = ["workload_a", "workload_b"]

        yaml_content = {
            "config": {"scale": 512, "repeats": 1, "profile": True},
            "workloads": {
                "h100": [
                    {"workload_a_7b": {"dtypes": ["bf16", "fp8"]}},
                    {"workload_a_70b": {"dtypes": ["fp8"]}},
                    {"workload_a_1t": {"dtypes": ["fp8"]}},
                    {"workload_b_8b": {"dtypes": ["fp8"]}},
                ]
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        def _always_valid(*a, **k):
            return (True, None, "", [])

        with (
            patch(
                'llmb_run.main.get_cluster_config',
                return_value=ClusterConfig.from_dict(mock_exemplar_cluster_config),
            ),
            patch('llmb_run.main.get_workloads', return_value=mock_workloads),
            patch('llmb_run.main.validate_workload_with_details', side_effect=_always_valid),
            patch('llmb_run.main.run_tests', _fake_run_tests),
        ):
            result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 0

        # Expected order:
        # 1. workload_a (7b bf16) - workload_a, numeric 7.0, bf16
        # 2. workload_a (7b fp8)  - workload_a, numeric 7.0, fp8
        # 3. workload_a (70b fp8) - workload_a, numeric 70.0, fp8
        # 4. workload_a (1t fp8)  - workload_a, numeric 1000.0, fp8
        # 5. workload_b (8b fp8)  - workload_b, numeric 8.0, fp8

        assert len(recorded_tasks) == 5

        t1 = recorded_tasks[0]
        assert t1.workload_key == "workload_a" and t1.model_size == "7b" and t1.dtype == "bf16"

        t2 = recorded_tasks[1]
        assert t2.workload_key == "workload_a" and t2.model_size == "7b" and t2.dtype == "fp8"

        t3 = recorded_tasks[2]
        assert t3.workload_key == "workload_a" and t3.model_size == "70b" and t3.dtype == "fp8"

        t4 = recorded_tasks[3]
        assert t4.workload_key == "workload_a" and t4.model_size == "1t" and t4.dtype == "fp8"

        t5 = recorded_tasks[4]
        assert t5.workload_key == "workload_b" and t5.model_size == "8b" and t5.dtype == "fp8"

    def test_exemplar_repeats_are_contiguous(self, runner, mock_exemplar_cluster_config):
        """Test that repeats for the same config are contiguous in task list."""
        from llmb_run.main import app

        yaml_content = {
            "config": {"scale": 512, "repeats": 3, "profile": True},
            "workloads": {
                "h100": [
                    {"pretrain_workload_a_7b": {"dtypes": ["fp8"]}},
                    {"pretrain_workload_b_13b": {"dtypes": ["fp8"]}},
                ]
            },
        }

        llmb_repo = Path(mock_exemplar_cluster_config['launcher']['llmb_repo'])
        write_exemplar_yaml(llmb_repo, yaml_content)

        recorded_tasks = []

        def _fake_run_tests(_cfg, tasks, _workloads, **_kwargs):
            recorded_tasks.extend(tasks)

        with patch('llmb_run.main.run_tests', _fake_run_tests):
            result = runner.invoke(app, ["exemplar"])

        assert result.exit_code == 0
        assert len(recorded_tasks) == 6  # 2 configs × 3 repeats

        # First 3 should all be workload_a 7b fp8
        for i in range(3):
            t = recorded_tasks[i]
            assert t.workload_key == "pretrain_workload_a"
            assert t.model_size == "7b"
            assert t.dtype == "fp8"
        assert [t.profile for t in recorded_tasks[:3]] == [False, False, True]

        # Next 3 should all be workload_b 13b fp8
        for i in range(3, 6):
            t = recorded_tasks[i]
            assert t.workload_key == "pretrain_workload_b"
            assert t.model_size == "13b"
            assert t.dtype == "fp8"
        assert [t.profile for t in recorded_tasks[3:6]] == [False, False, True]
