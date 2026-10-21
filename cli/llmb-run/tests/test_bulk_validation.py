# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmb_run.main import validate_bulk_tasks
from llmb_run.task_loader import flatten_yaml_tasks, get_tasks_yaml
from llmb_run.tasks import WorkloadTask
from llmb_run.workload_validator import ValidationErrorType


@pytest.fixture
def workloads_fixture():
    """Return a comprehensive workloads dictionary for testing."""
    return {
        "pretrain_nemotron-h": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "56b",
                                    "dtypes": "fp8",  # Only fp8 supported
                                    "scales": [32, 64, 128, 256, 512, 1024, 2048],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/nemotron-h",
            "workload": "nemotron-h",
            "workload_type": "pretrain",
        },
        "pretrain_llama": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": ["fp8", "bf16", "fp16"],
                                    "scales": [8, 16, 32, 64],
                                },
                                {
                                    "model_size": "70b",
                                    "dtypes": ["bf16", "fp16"],  # No fp8 for 70b
                                    "scales": [64, 128, 256],
                                },
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/llama",
            "workload": "llama",
            "workload_type": "pretrain",
        },
    }


@pytest.fixture
def cluster_config_fixture(cluster_config_factory):
    """Return cluster configuration for testing."""
    return cluster_config_factory({"workloads": {"installed": ["pretrain_nemotron-h", "pretrain_llama"]}})


@pytest.fixture
def yaml_file(tmp_path):
    """Factory fixture to create temporary YAML files for testing."""

    def _create(content):
        path = tmp_path / "tasks.yaml"
        path.write_text(content)
        return str(path)

    return _create


class TestBulkValidation:
    """Test bulk job validation functionality."""

    def test_all_tasks_valid(self, workloads_fixture, cluster_config_fixture):
        """Test bulk validation when all tasks are valid."""
        tasks = [
            WorkloadTask("pretrain_llama", "7b", "fp8", 32, False),
            WorkloadTask("pretrain_llama", "7b", "bf16", 16, False),
            WorkloadTask("pretrain_llama", "70b", "bf16", 128, False),
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        assert len(validated_tasks) == 3
        assert len(error_summary) == 0

    def test_mixed_valid_invalid_tasks(self, workloads_fixture, cluster_config_fixture):
        """Test bulk validation with mix of valid and invalid tasks."""
        tasks = [
            WorkloadTask("pretrain_nemotron-h", "56b", "fp8", 128, False),  # Valid
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 128, False),  # Invalid dtype
            WorkloadTask("pretrain_llama", "7b", "fp8", 32, False),  # Valid
            WorkloadTask("pretrain_llama", "70b", "fp8", 128, False),  # Invalid dtype for 70b
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        assert len(validated_tasks) == 2  # Only 2 valid tasks
        assert len(error_summary) == 2  # 2 different error types

        # Check error grouping
        error_types = [err['error_type'] for err in error_summary.values()]
        assert ValidationErrorType.DTYPE_NOT_SUPPORTED in error_types

    def test_error_grouping_by_type(self, workloads_fixture, cluster_config_fixture):
        """Test that identical errors are properly grouped."""
        tasks = [
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 128, False),  # Invalid dtype
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 256, False),  # Same error
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 512, False),  # Same error
            WorkloadTask("pretrain_llama", "70b", "fp8", 128, False),  # Different invalid dtype
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        assert len(validated_tasks) == 0
        assert len(error_summary) == 2  # Two different error groups

        # Find the nemotron-h error group
        nemotron_errors = [err for err in error_summary.values() if "nemotron-h" in err['example_task'].workload_key]
        assert len(nemotron_errors) == 1
        assert nemotron_errors[0]['count'] == 3  # 3 bf16 errors grouped together

    def test_scale_validation_error(self, workloads_fixture, cluster_config_fixture):
        """Test validation with invalid scale values."""
        tasks = [
            WorkloadTask("pretrain_llama", "7b", "fp8", 4, False),  # Scale too small (below min 8)
            WorkloadTask("pretrain_llama", "7b", "bf16", 4, False),  # Same scale, different dtype
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        assert len(validated_tasks) == 0
        assert len(error_summary) == 1  # Both tasks should have same scale error

        error = list(error_summary.values())[0]
        assert error['error_type'] == ValidationErrorType.SCALE_NOT_SUPPORTED
        assert error['count'] == 2  # Both tasks failed with same error

    def test_workload_not_found_error(self, workloads_fixture, cluster_config_fixture):
        """Test validation with non-existent workload."""
        tasks = [
            WorkloadTask("pretrain_nonexistent", "7b", "fp8", 32, False),
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        assert len(validated_tasks) == 0
        assert len(error_summary) == 1

        error = list(error_summary.values())[0]
        assert error['error_type'] == ValidationErrorType.WORKLOAD_NOT_FOUND


class TestBulkYamlParsing:
    """Test bulk job parsing and validation integration."""

    def test_yaml_parsing_with_validation(self, workloads_fixture, cluster_config_fixture, yaml_file):
        """Test end-to-end YAML parsing with validation."""
        path = yaml_file("""
pretrain_nemotron-h_56b:
  dtypes: ['fp8', 'bf16']  # bf16 should fail validation
  scales: 128
  repeats: 1
  tasks:
    - overrides:
        env:
          GBS: 96
    - overrides:
        env:
          GBS: 192
""")

        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        validated_tasks, error_summary = validate_bulk_tasks(task_list, workloads_fixture, cluster_config_fixture)

        # Should have 2 valid fp8 tasks and 2 invalid bf16 tasks
        assert len(validated_tasks) == 2  # Only fp8 tasks
        assert len(error_summary) == 1  # bf16 errors grouped

        bf16_error = list(error_summary.values())[0]
        assert bf16_error['count'] == 2  # 2 bf16 tasks failed
        assert bf16_error['error_type'] == ValidationErrorType.DTYPE_NOT_SUPPORTED
        assert 'fp8' in bf16_error['suggestions']

    def test_complex_yaml_multiple_workloads(self, workloads_fixture, cluster_config_fixture, yaml_file):
        """Test complex YAML with multiple workloads and error types."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: ['fp8', 'fp16']
  scales: [32, 128]  # 128 too large for 7b
  repeats: 1
  tasks:
    - {}

pretrain_nemotron-h_56b:
  dtypes: 'bf16'  # Not supported
  scales: 256
  repeats: 1
  tasks:
    - {}
""")

        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        validated_tasks, error_summary = validate_bulk_tasks(task_list, workloads_fixture, cluster_config_fixture)

        # Should have some valid llama fp8/fp16 tasks and various errors
        assert len(validated_tasks) >= 1  # At least some valid tasks
        assert len(error_summary) >= 1  # Multiple error types

    def test_error_suggestions_quality(self, workloads_fixture, cluster_config_fixture):
        """Test that error suggestions are helpful and accurate."""
        tasks = [
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 128, False),
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        error = list(error_summary.values())[0]
        assert error['error_type'] == ValidationErrorType.DTYPE_NOT_SUPPORTED
        assert 'fp8' in error['suggestions']  # Should suggest the valid dtype
        assert 'bf16' not in error['suggestions']  # Should not suggest invalid dtype

    def test_yaml_header_missing_model_size(self, yaml_file):
        """Test error when model size suffix is missing."""
        path = yaml_file("""
pretrain_nemotron-h:
  tasks:
    - dtypes: fp8
      scales: 128
""")

        with pytest.raises(ValueError) as exc_info:
            get_tasks_yaml(path)

        error_msg = str(exc_info.value)
        assert "pretrain_nemotron-h" in error_msg
        assert "model size" in error_msg.lower() or "modelsize" in error_msg.lower()
        assert any(pattern in error_msg for pattern in ["_<digits>(b|t)", "Example", "format"])
        assert "llmb-run list" in error_msg

    def test_yaml_header_invalid_model_size_format(self, yaml_file):
        """Test error when model size doesn't match required pattern."""
        path = yaml_file("""
pretrain_llama_7x:
  tasks:
    - dtypes: fp8
      scales: 128
""")

        with pytest.raises(ValueError) as exc_info:
            get_tasks_yaml(path)

        error_msg = str(exc_info.value)
        assert "pretrain_llama_7x" in error_msg
        assert "model size" in error_msg.lower() or "modelsize" in error_msg.lower()

    def test_yaml_header_with_decimal_version(self, yaml_file):
        """Test that decimal versions in workload names parse correctly."""
        path = yaml_file("""
pretrain_llama3.1_70b:
  tasks:
    - dtypes: fp8
      scales: 128
      repeats: 1
""")

        tasks_parsed = get_tasks_yaml(path)
        assert "pretrain_llama3.1" in tasks_parsed
        assert "70b" in tasks_parsed["pretrain_llama3.1"]

    def test_yaml_header_with_hyphens(self, yaml_file):
        """Test that hyphens in workload names parse correctly."""
        path = yaml_file("""
pretrain_nemotron-h_56b:
  tasks:
    - dtypes: fp8
      scales: 128
      repeats: 1
""")

        tasks_parsed = get_tasks_yaml(path)
        assert "pretrain_nemotron-h" in tasks_parsed
        assert "56b" in tasks_parsed["pretrain_nemotron-h"]

    def test_yaml_header_with_trillion_suffix(self, yaml_file):
        """Test that trillion suffixes in workload names parse correctly."""
        path = yaml_file("""
pretrain_kimi-k2_1t:
  tasks:
    - dtypes: fp8
      scales: 1024
      repeats: 1
""")

        tasks_parsed = get_tasks_yaml(path)
        assert "pretrain_kimi-k2" in tasks_parsed
        assert "1t" in tasks_parsed["pretrain_kimi-k2"]

    def test_yaml_error_propagates_to_submit_command(self, tmp_path, cluster_config_factory):
        """Test that YAML parse errors surface properly in submit command."""
        yaml_content = """
pretrain_invalid:
  tasks:
    - dtypes: fp8
      scales: 128
"""
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text(yaml_content)

        # Test that the error surfaces through task_generation.py
        from llmb_run.task_generation import TaskGenerationRequest, generate_tasks

        request = TaskGenerationRequest(
            workloads={},  # Empty for this test
            cluster_config=cluster_config_factory(),
            file_path=str(yaml_file),
        )

        with pytest.raises(ValueError) as exc_info:
            generate_tasks(request)

        error_msg = str(exc_info.value)
        assert "pretrain_invalid" in error_msg
        assert "model size" in error_msg.lower()


class TestMainIntegration:
    """Test integration with main validation flow."""

    def test_validation_summary_output_format(self, workloads_fixture, cluster_config_fixture):
        """Test that validation summary provides clear, actionable output."""
        tasks = [
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 128, False),
            WorkloadTask("pretrain_nemotron-h", "56b", "bf16", 256, False),
            WorkloadTask("pretrain_llama", "70b", "fp8", 128, False),
        ]

        validated_tasks, error_summary = validate_bulk_tasks(tasks, workloads_fixture, cluster_config_fixture)

        assert len(validated_tasks) == 0
        assert len(error_summary) == 2

        # Check that errors have the required fields for nice output formatting
        for error in error_summary.values():
            assert 'count' in error
            assert 'error_type' in error
            assert 'error_msg' in error
            assert 'suggestions' in error
            assert 'example_task' in error
            assert error['count'] > 0


class TestYamlProxyParsing:
    """Test YAML parsing with proxy field."""

    def test_yaml_with_proxy_field(self, yaml_file):
        """Test YAML parsing with proxy: true in task."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: ['fp8']
  scales: [32]
  repeats: 1
  tasks:
    - proxy: true
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        assert len(task_list) == 1
        assert task_list[0].proxy is True

    def test_yaml_proxy_at_workload_level(self, yaml_file):
        """Test YAML with proxy: true propagating to multiple tasks."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: ['fp8', 'bf16']
  scales: [32]
  repeats: 1
  tasks:
    - proxy: true
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        # Should create 2 tasks (fp8 and bf16), both with proxy=True
        assert len(task_list) == 2
        for task in task_list:
            assert task.proxy is True


class TestYamlAddProfile:
    """Test YAML add_profile feature."""

    def test_add_profile_creates_extra_profiling_run(self, yaml_file):
        """add_profile should add one profiling run per dtype/scale combo."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: ['fp8']
  scales: [32]
  repeats: 2
  add_profile: true
  tasks:
    - {}
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        # 2 repeats + 1 profiling = 3 tasks
        assert len(task_list) == 3
        profile_tasks = [t for t in task_list if t.profile is True]
        non_profile = [t for t in task_list if t.profile is False]
        assert len(profile_tasks) == 1
        assert len(non_profile) == 2

    def test_add_profile_with_multiple_dtypes_and_scales(self, yaml_file):
        """add_profile adds one profile run per dtype/scale combination."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: ['fp8', 'bf16']
  scales: [32, 64]
  repeats: 1
  add_profile: true
  tasks:
    - {}
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        # 2 dtypes * 2 scales = 4 combos; each gets 1 repeat + 1 profile = 8 total
        assert len(task_list) == 8
        profile_tasks = [t for t in task_list if t.profile is True]
        assert len(profile_tasks) == 4

    def test_profile_and_add_profile_conflict(self, yaml_file):
        """Setting both profile and add_profile should raise an error."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: ['fp8']
  scales: [32]
  repeats: 1
  tasks:
    - profile: true
      add_profile: true
""")
        with pytest.raises(ValueError, match="Cannot specify both"):
            get_tasks_yaml(path)


class TestYamlDefaults:
    """Test YAML defaults feature."""

    def test_defaults_env_propagates_to_tasks(self, yaml_file):
        """Default env overrides propagate to all tasks."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: fp8
  scales: 32
  repeats: 1
  defaults:
    env:
      GBS: 96
  tasks:
    - {}
    - overrides:
        env:
          SEQ_LEN: 4096
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        assert len(task_list) == 2
        # First task gets only defaults
        assert task_list[0].env_overrides == {"GBS": 96}
        # Second task gets defaults merged with overrides
        assert task_list[1].env_overrides == {"GBS": 96, "SEQ_LEN": 4096}

    def test_defaults_env_overridden_by_task(self, yaml_file):
        """Task-level env overrides take precedence over defaults."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: fp8
  scales: 32
  repeats: 1
  defaults:
    env:
      GBS: 96
  tasks:
    - overrides:
        env:
          GBS: 192
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        assert len(task_list) == 1
        assert task_list[0].env_overrides["GBS"] == 192

    def test_defaults_params_propagates(self, yaml_file):
        """Default model params propagate to tasks."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: fp8
  scales: 32
  repeats: 1
  defaults:
    params:
      tp: 4
  tasks:
    - {}
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        assert task_list[0].model_overrides == {"tp": 4}


class TestYamlSweepParams:
    """Test YAML sweep parameter feature."""

    def test_sweep_single_param(self, yaml_file):
        """Sweep over a single parameter generates correct task count."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: fp8
  scales: 32
  repeats: 1
  tasks:
    - overrides:
        params:
          tp: [1, 2, 4]
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        # 3 tp values * 1 dtype * 1 scale * 1 repeat = 3 tasks
        assert len(task_list) == 3
        tp_values = {t.model_overrides["tp"] for t in task_list}
        assert tp_values == {1, 2, 4}

    def test_sweep_multiple_params(self, yaml_file):
        """Sweep over multiple parameters generates cartesian product."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: fp8
  scales: 32
  repeats: 1
  tasks:
    - overrides:
        params:
          tp: [1, 2]
          pp: [1, 4]
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        # 2 tp * 2 pp * 1 dtype * 1 scale * 1 repeat = 4 tasks
        assert len(task_list) == 4
        combos = {(t.model_overrides["tp"], t.model_overrides["pp"]) for t in task_list}
        assert combos == {(1, 1), (1, 4), (2, 1), (2, 4)}

    def test_sweep_with_add_profile(self, yaml_file):
        """Sweep combined with add_profile generates extra profiling runs."""
        path = yaml_file("""
pretrain_llama_7b:
  dtypes: fp8
  scales: 32
  repeats: 1
  tasks:
    - add_profile: true
      overrides:
        params:
          tp: [1, 2]
""")
        tasks_parsed = get_tasks_yaml(path)
        task_list = flatten_yaml_tasks(tasks_parsed)

        # 2 tp values * (1 repeat + 1 profile) = 4 tasks
        assert len(task_list) == 4
        profile_tasks = [t for t in task_list if t.profile is True]
        assert len(profile_tasks) == 2


if __name__ == "__main__":
    pytest.main([__file__])
