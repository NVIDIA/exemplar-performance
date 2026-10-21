# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmb_run.config_manager import ClusterConfig
from llmb_run.task_generation import (
    TaskGenerationRequest,
    ValidationError,
    _generate_scales_up_to_max,
    generate_submit_all_tasks,
    generate_tasks,
    parse_comma_list,
)


def _build_cluster_config(*, gpu_type: str = "h100", installed: list[str] | None = None) -> ClusterConfig:
    return ClusterConfig.from_dict(
        {
            "launcher": {
                "llmb_repo": "/tmp",
                "llmb_install": "/tmp",
                "gpu_type": gpu_type,
            },
            "slurm": {
                "gpu_partition": "batch",
            },
            "workloads": {
                "installed": installed or [],
            },
        }
    )


@pytest.fixture
def workloads_fixture():
    """Return comprehensive workloads dictionary for submit-all testing."""
    return {
        "pretrain_nemotron": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "340b",
                                    "dtypes": ["fp8", "bf16"],
                                    "scales": [128, 256, 512],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/nemotron",
            "workload": "nemotron",
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
                                    "dtypes": ["fp8", "bf16"],
                                    "scales": [16, 32, 64],
                                },
                                {
                                    "model_size": "70b",
                                    "dtypes": "bf16",  # Only bf16 for 70b
                                    "scales": [256, 512],
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
        "inference_test": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": ["fp8"],
                                    "scales": [8],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/inference",
            "workload": "test",
            "workload_type": "inference",
        },
        "pretrain_exact_scales": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "100b",
                                    "dtypes": ["fp8"],
                                    "scales": [64, 128],
                                    "exact_scales": True,
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/exact",
            "workload": "exact_scales",
            "workload_type": "pretrain",
        },
        "finetune_model": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "13b",
                                    "dtypes": ["bf16"],
                                    "scales": [32, 64],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/finetune",
            "workload": "model",
            "workload_type": "finetune",
        },
    }


@pytest.fixture
def cluster_config_fixture():
    """Return cluster configuration for testing."""
    return _build_cluster_config(
        installed=[
            "pretrain_nemotron",
            "pretrain_llama",
            "inference_test",  # Should be excluded by workload_type
            "pretrain_exact_scales",
            "finetune_model",
        ]
    )


class TestScaleGeneration:
    """Test scale generation logic for submit-all functionality."""

    def test_exact_scales_only_uses_metadata_scales(self):
        """Test that exact_scales=True only uses scales from metadata up to max."""
        metadata_scales = [64, 128, 256]
        max_scale = 512
        exact_scales = True

        result = _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales)

        assert result == [64, 128, 256]

    def test_exact_scales_respects_max_limit(self):
        """Test that exact_scales=True respects max_scale limit."""
        metadata_scales = [64, 128, 256, 512]
        max_scale = 200
        exact_scales = True

        result = _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales)

        assert result == [64, 128]

    def test_non_exact_scales_adds_power_of_2(self):
        """Test that non-exact scales adds power-of-2 scales beyond metadata max."""
        metadata_scales = [16, 32, 64]
        max_scale = 512
        exact_scales = False

        result = _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales)

        # Should include metadata scales + power-of-2 scales: 128, 256, 512
        expected = [16, 32, 64, 128, 256, 512]
        assert result == expected

    def test_non_exact_scales_no_extension_when_max_low(self):
        """Test that non-exact scales doesn't add scales when max_scale <= max metadata."""
        metadata_scales = [128, 256, 512]
        max_scale = 400
        exact_scales = False

        result = _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales)

        assert result == [128, 256]

    def test_power_of_2_sequence_correct(self):
        """Test that power-of-2 sequence is generated correctly."""
        metadata_scales = [32]  # Max is 32
        max_scale = 1024
        exact_scales = False

        result = _generate_scales_up_to_max(metadata_scales, max_scale, exact_scales)

        # Should be [32, 64, 128, 256, 512, 1024]
        expected = [32, 64, 128, 256, 512, 1024]
        assert result == expected

    def test_empty_metadata_scales(self):
        """Test handling of empty metadata scales."""
        result = _generate_scales_up_to_max([], 512, False)
        assert result == []


class TestSubmitAllTaskGeneration:
    """Test submit-all task generation functionality."""

    def test_basic_task_generation(self, workloads_fixture, cluster_config_fixture):
        """Test basic task generation for installed workloads."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=128, repeats=1, profile=False
        )

        # Should generate tasks for pretrain and finetune workloads only
        workload_keys = {task.workload_key for task in tasks}
        expected_workloads = {"pretrain_nemotron", "pretrain_llama", "pretrain_exact_scales", "finetune_model"}
        assert workload_keys == expected_workloads

        # Should not include inference workloads
        assert "inference_test" not in workload_keys

    def test_max_scale_limit_respected(self, workloads_fixture, cluster_config_fixture):
        """Test that max_scale limit is respected."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=100, repeats=1, profile=False
        )

        # No task should have scale > 100
        for task in tasks:
            assert task.scale <= 100

    def test_repeats_parameter(self, workloads_fixture, cluster_config_fixture):
        """Test that repeats parameter creates multiple tasks."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=256, repeats=3, profile=False
        )

        # Count tasks for a specific configuration (use scale 128 which is in metadata)
        nemotron_340b_fp8_128_tasks = [
            task
            for task in tasks
            if (
                task.workload_key == "pretrain_nemotron"
                and task.model_size == "340b"
                and task.dtype == "fp8"
                and task.scale == 128
            )
        ]

        assert len(nemotron_340b_fp8_128_tasks) == 3

    def test_exact_scales_handling(self, workloads_fixture, cluster_config_fixture):
        """Test that exact_scales=True is handled correctly."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=512, repeats=1, profile=False
        )

        # Find tasks for the exact_scales workload
        exact_tasks = [task for task in tasks if task.workload_key == "pretrain_exact_scales"]
        scales = {task.scale for task in exact_tasks}

        # Should only include metadata scales (64, 128), not extended scales
        assert scales == {64, 128}

    def test_power_of_2_extension(self, workloads_fixture, cluster_config_fixture):
        """Test that power-of-2 scales are added when max_nodes > max metadata scale."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=1024, repeats=1, profile=False
        )

        # Find tasks for nemotron (scales: [128, 256, 512], no exact_scales)
        nemotron_tasks = [task for task in tasks if task.workload_key == "pretrain_nemotron"]
        scales = {task.scale for task in nemotron_tasks}

        # Should include original scales + power-of-2 extension: 1024
        assert 128 in scales
        assert 256 in scales
        assert 512 in scales
        assert 1024 in scales

    def test_per_dtype_scales_mapping(self, cluster_config_fixture):
        """Tasks should reflect per-dtype scales when dtypes is a mapping."""
        workloads = {
            "pretrain_custom": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "15b",
                                        "dtypes": {
                                            "fp8": [128, 256],
                                            "bf16": {"scales": [256, 512], "exact_scales": True},
                                        },
                                    }
                                ]
                            }
                        }
                    }
                },
                "workload_type": "pretrain",
            }
        }

        cluster_config = _build_cluster_config(installed=["pretrain_custom"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=1024, repeats=1, profile=False)

        # Collect scales per dtype
        fp8_scales = {t.scale for t in tasks if t.dtype == "fp8"}
        bf16_scales = {t.scale for t in tasks if t.dtype == "bf16"}

        # fp8 should include [128, 256] and extend power-of-2 up to max (512, 1024)
        assert fp8_scales == {128, 256, 512, 1024}

        # bf16 exact_scales: True, should be exactly the provided scales
        assert bf16_scales == {256, 512}

    def test_list_mode_handles_mapping(self, monkeypatch, capsys, caplog):
        """Ensure list mode aggregates dtypes/scales correctly with mapping form."""
        workloads = {
            "pretrain_nemotron4": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "15b",
                                        "dtypes": {
                                            "fp8": [16, 32],
                                            "bf16": {"scales": [32, 64]},
                                        },
                                    }
                                ]
                            }
                        }
                    }
                },
                "dir": "/fake/nemotron4",
                "workload": "nemotron4",
                "workload_type": "pretrain",
            }
        }
        cluster_config = _build_cluster_config(installed=["pretrain_nemotron4"])

        # Call listing function directly to avoid CLI patching ambiguity
        from llmb_run.workload_validator import print_avail_workloads

        caplog.clear()
        caplog.set_level("INFO", logger="llmb_run.workload_validator")
        print_avail_workloads(workloads, cluster_config, cluster_gpu_type="h100", verbose=True)
        logged = caplog.text
        assert "pretrain_nemotron4:" in logged
        assert "15b:" in logged
        # Should list both dtypes and include aggregated scales
        assert ("Data types: bf16, fp8" in logged) or ("Data types: fp8, bf16" in logged)
        # Per-dtype scales should be present (new formatting)
        assert "fp8: 16, 32" in logged
        assert "bf16: 32, 64" in logged

    def test_dtype_permutations(self, workloads_fixture, cluster_config_fixture):
        """Test that all dtype permutations are generated."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=256, repeats=1, profile=False
        )

        # Find tasks for nemotron 340b at scale 256
        nemotron_340b_256_tasks = [
            task
            for task in tasks
            if (task.workload_key == "pretrain_nemotron" and task.model_size == "340b" and task.scale == 256)
        ]

        dtypes = {task.dtype for task in nemotron_340b_256_tasks}
        assert dtypes == {"fp8", "bf16"}

    def test_model_size_permutations(self, workloads_fixture, cluster_config_fixture):
        """Test that all model size permutations are generated."""
        tasks = generate_submit_all_tasks(workloads_fixture, cluster_config_fixture, max_scale=512, repeats=1)

        # Find all llama tasks
        llama_tasks = [task for task in tasks if task.workload_key == "pretrain_llama"]
        model_sizes = {task.model_size for task in llama_tasks}

        assert model_sizes == {"7b", "70b"}

    def test_gpu_type_filtering(self, workloads_fixture):
        """Test that workloads without cluster GPU type are skipped."""
        cluster_config = _build_cluster_config(
            gpu_type="a100",  # Different GPU type
            installed=["pretrain_nemotron", "pretrain_llama"],
        )

        tasks = generate_submit_all_tasks(workloads_fixture, cluster_config, max_scale=256, repeats=1, profile=False)

        # Should generate no tasks since workloads don't support a100
        assert len(tasks) == 0

    def test_no_installed_workloads(self, workloads_fixture):
        """Test handling when no workloads are installed."""
        cluster_config = _build_cluster_config(installed=[])

        tasks = generate_submit_all_tasks(workloads_fixture, cluster_config, max_scale=256, repeats=1, profile=False)

        assert len(tasks) == 0

    def test_workload_type_filtering(self, workloads_fixture, cluster_config_fixture):
        """Test that only pretrain and finetune workloads are included."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=256, repeats=1, profile=False
        )

        workload_keys = {task.workload_key for task in tasks}

        # Should include pretrain and finetune
        assert "pretrain_nemotron" in workload_keys
        assert "pretrain_llama" in workload_keys
        assert "finetune_model" in workload_keys

        # Should not include inference
        assert "inference_test" not in workload_keys


class TestSubmitAllFilteringFeatures:
    """Test new filtering features for submit-all command."""

    def test_workload_filter(self, workloads_fixture, cluster_config_fixture):
        """Test filtering by specific workloads."""
        # Test filtering to only nemotron workload
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_nemotron"],
        )

        workload_keys = {task.workload_key for task in tasks}
        assert workload_keys == {"pretrain_nemotron"}

        # Test filtering to multiple workloads
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_nemotron", "finetune_model"],
        )

        workload_keys = {task.workload_key for task in tasks}
        assert workload_keys == {"pretrain_nemotron", "finetune_model"}

    def test_dtype_filter(self, workloads_fixture, cluster_config_fixture):
        """Test filtering by specific dtypes."""
        # Test filtering to only fp8
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=256, repeats=1, profile=False, dtype_filter=["fp8"]
        )

        dtypes = {task.dtype for task in tasks}
        assert dtypes == {"fp8"}

        # Test filtering to only bf16
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=256, repeats=1, profile=False, dtype_filter=["bf16"]
        )

        dtypes = {task.dtype for task in tasks}
        assert dtypes == {"bf16"}

        # Test filtering to multiple dtypes (should be same as no filter in this case)
        tasks_filtered = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            dtype_filter=["fp8", "bf16"],
        )

        tasks_unfiltered = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=256, repeats=1, profile=False
        )

        assert len(tasks_filtered) == len(tasks_unfiltered)

    def test_min_scale_flag(self, workloads_fixture, cluster_config_fixture):
        """Test min_scale flag to only run minimum scales."""
        # Test with min_scale=True
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=512, repeats=1, profile=False, min_scale=True
        )

        # Group tasks by workload and dtype to find min scales
        task_scales = {}
        for task in tasks:
            key = (task.workload_key, task.model_size, task.dtype)
            if key not in task_scales:
                task_scales[key] = []
            task_scales[key].append(task.scale)

        # Each workload/model/dtype combination should only have one scale (the minimum)
        for key, scales in task_scales.items():
            assert len(set(scales)) == 1, f"Multiple scales found for {key}: {scales}"

        # Verify some specific minimum scales based on fixture data
        nemotron_scales = {task.scale for task in tasks if task.workload_key == "pretrain_nemotron"}
        assert 128 in nemotron_scales  # Min scale for nemotron is 128
        assert 256 not in nemotron_scales and 512 not in nemotron_scales  # Should not have larger scales

    def test_min_scale_with_max_scale_interaction(self, workloads_fixture, cluster_config_fixture):
        """Test that min_scale respects max_scale limit."""
        # Test where max_scale is smaller than some min scales in metadata
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=100, repeats=1, profile=False, min_scale=True
        )

        # No task should have scale > max_scale
        for task in tasks:
            assert task.scale <= 100

        # Nemotron has min scale 128, so it should be excluded entirely
        nemotron_tasks = [task for task in tasks if task.workload_key == "pretrain_nemotron"]
        assert len(nemotron_tasks) == 0

    def test_combined_filters(self, workloads_fixture, cluster_config_fixture):
        """Test combining multiple filters together."""
        # Test workload + dtype filter
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_nemotron", "pretrain_llama"],
            dtype_filter=["fp8"],
        )

        workload_keys = {task.workload_key for task in tasks}
        assert workload_keys.issubset({"pretrain_nemotron", "pretrain_llama"})

        dtypes = {task.dtype for task in tasks}
        assert dtypes == {"fp8"}

        # Test all three filters together
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=512,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_nemotron"],
            dtype_filter=["fp8"],
            min_scale=True,
        )

        # Should only have nemotron tasks with fp8 dtype at minimum scale
        assert len(tasks) > 0
        for task in tasks:
            assert task.workload_key == "pretrain_nemotron"
            assert task.dtype == "fp8"
            assert task.scale == 128  # Min scale for nemotron

    def test_nonexistent_workload_filter(self, workloads_fixture, cluster_config_fixture):
        """Test filtering with nonexistent workload names."""
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            workload_filter=["nonexistent_workload"],
        )

        assert len(tasks) == 0

    def test_nonexistent_dtype_filter(self, workloads_fixture, cluster_config_fixture):
        """Test filtering with nonexistent dtype."""
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            dtype_filter=["fp32"],  # Not in fixture data
        )

        assert len(tasks) == 0

    def test_min_scale_only_mode(self, workloads_fixture, cluster_config_fixture):
        """Test min_scale mode without max_scale (no upper limit)."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=None, repeats=1, profile=False, min_scale=True
        )

        # Should have tasks with minimum scales for each workload/dtype combination
        assert len(tasks) > 0

        # Group tasks by workload and dtype to verify only min scales
        task_scales = {}
        for task in tasks:
            key = (task.workload_key, task.model_size, task.dtype)
            if key not in task_scales:
                task_scales[key] = []
            task_scales[key].append(task.scale)

        # Each workload/model/dtype combination should only have one scale (the minimum)
        for key, scales in task_scales.items():
            assert len(set(scales)) == 1, f"Multiple scales found for {key}: {scales}"

        # Check specific minimum scales - nemotron should use its minimum (128)
        nemotron_scales = {task.scale for task in tasks if task.workload_key == "pretrain_nemotron"}
        assert 128 in nemotron_scales  # Min scale for nemotron 340b

    def test_workload_modelsize_filter(self, workloads_fixture, cluster_config_fixture):
        """Test filtering by specific workload_modelsize combinations."""
        # Test filtering to specific model size
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_nemotron_340b"],
        )

        # Should only have nemotron 340b tasks, not 15b
        workload_model_combinations = {f"{task.workload_key}_{task.model_size}" for task in tasks}
        assert workload_model_combinations == {"pretrain_nemotron_340b"}

        # Test filtering to multiple specific model sizes
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=256,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_llama_7b", "finetune_model_13b"],
        )

        workload_model_combinations = {f"{task.workload_key}_{task.model_size}" for task in tasks}
        assert workload_model_combinations == {"pretrain_llama_7b", "finetune_model_13b"}

    def test_mixed_workload_and_modelsize_filter(self, workloads_fixture, cluster_config_fixture):
        """Test mixing workload-only and workload_modelsize filters."""
        # Mix full workload and specific model size
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=512,
            repeats=1,
            profile=False,
            workload_filter=["pretrain_nemotron", "pretrain_llama_7b"],  # Full nemotron + specific llama size
        )

        workload_keys = {task.workload_key for task in tasks}
        model_combinations = {f"{task.workload_key}_{task.model_size}" for task in tasks}

        # Should have all nemotron models
        assert "pretrain_nemotron" in workload_keys
        # Should only have llama 7b, not 70b
        assert "pretrain_llama_7b" in model_combinations
        assert "pretrain_llama_70b" not in model_combinations

    def test_realistic_nemotron4_scenario(self):
        """Test realistic scenario with nemotron4 workload similar to user example."""
        # Create a workload similar to the nemotron metadata provided by user
        workload_data = {
            "pretrain_nemotron4": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "15b",
                                        "dtypes": ["fp8", "bf16"],
                                        "scales": [16, 32, 64, 128, 256, 512],
                                    },
                                    {"model_size": "340b", "dtypes": ["fp8", "bf16"], "scales": [256, 512, 1024]},
                                ]
                            }
                        }
                    }
                },
                "workload_type": "pretrain",
            }
        }

        cluster_config = _build_cluster_config(installed=["pretrain_nemotron4"])

        # Test the exact scenario from user example: --max-scale 256 --min-scale -w pretrain_nemotron4
        tasks = generate_submit_all_tasks(
            workload_data,
            cluster_config,
            max_scale=256,
            repeats=1,
            profile=False,
            min_scale=True,
            workload_filter=["pretrain_nemotron4"],
        )

        # Should generate 4 tasks as mentioned in user example:
        # 15b fp8 scale=16, 15b bf16 scale=16, 340b fp8 scale=256, 340b bf16 scale=256
        assert len(tasks) == 4

        # Check specific tasks
        task_details = [(t.model_size, t.dtype, t.scale) for t in tasks]
        expected_tasks = [
            ("15b", "fp8", 16),  # min scale for 15b
            ("15b", "bf16", 16),  # min scale for 15b
            ("340b", "fp8", 256),  # min scale for 340b (within max_scale 256)
            ("340b", "bf16", 256),  # min scale for 340b (within max_scale 256)
        ]

        for expected in expected_tasks:
            assert expected in task_details, f"Expected task {expected} not found in {task_details}"


@pytest.fixture
def simple_workload_factory():
    """Factory for creating test workloads with specified scales."""

    def _create(workload_name, scales, dtypes="fp8", exact_scales=None):
        config = {
            "model_size": "10b",
            "dtypes": dtypes if isinstance(dtypes, dict) else [dtypes] if isinstance(dtypes, str) else dtypes,
        }
        if scales is not None:
            config["scales"] = scales
        if exact_scales is not None:
            config["exact_scales"] = exact_scales

        return {
            workload_name: {
                "metadata": {"run": {"gpu_configs": {"h100": {"model_configs": [config]}}}},
                "workload_type": "pretrain",
            }
        }

    return _create


class TestExactScalesFlag:
    """Test the --exact-scales flag behavior (TDD - these tests should fail initially)."""

    def test_exact_scales_prevents_power_of_2_expansion(self, workloads_fixture, cluster_config_fixture):
        """With metadata [128, 256, 512] and max_scale=1024, exact_scales prevents expansion to 1024."""
        tasks = generate_submit_all_tasks(
            workloads_fixture,
            cluster_config_fixture,
            max_scale=1024,
            repeats=1,
            workload_filter=["pretrain_nemotron"],
            exact_scales=True,
        )

        scales = {task.scale for task in tasks}
        assert scales == {128, 256, 512}
        assert 1024 not in scales

    def test_exact_scales_with_smaller_max_scale(self, simple_workload_factory, cluster_config_fixture):
        """With metadata [128, 256] and max_scale=512, exact_scales prevents expansion to 512."""
        workloads = simple_workload_factory("pretrain_test", scales=[128, 256])
        cluster_config = _build_cluster_config(installed=["pretrain_test"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=512, repeats=1, exact_scales=True)

        scales = {task.scale for task in tasks}
        assert scales == {128, 256}
        assert 512 not in scales

    def test_exact_scales_with_specific_scales(self, simple_workload_factory, cluster_config_fixture):
        """With metadata [128, 256] and specific_scales=[256, 512], exact_scales skips 512."""
        workloads = simple_workload_factory("pretrain_test", scales=[128, 256])
        cluster_config = _build_cluster_config(installed=["pretrain_test"])

        tasks = generate_submit_all_tasks(
            workloads,
            cluster_config,
            max_scale=None,
            repeats=1,
            specific_scales=[256, 512],
            exact_scales=True,
        )

        scales = {task.scale for task in tasks}
        assert scales == {256}
        assert 512 not in scales

    def test_exact_scales_overrides_per_workload_setting(self, simple_workload_factory, cluster_config_fixture):
        """Global exact_scales=True overrides workload metadata exact_scales=False."""
        workloads = simple_workload_factory("pretrain_test", scales=[64, 128], exact_scales=False)
        cluster_config = _build_cluster_config(installed=["pretrain_test"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=512, repeats=1, exact_scales=True)

        scales = {task.scale for task in tasks}
        assert scales == {64, 128}
        assert 256 not in scales and 512 not in scales

    def test_exact_scales_respects_max_scale_limit(self, simple_workload_factory, cluster_config_fixture):
        """With metadata [64, 128, 256, 512] and max_scale=200, only returns [64, 128]."""
        workloads = simple_workload_factory("pretrain_test", scales=[64, 128, 256, 512])
        cluster_config = _build_cluster_config(installed=["pretrain_test"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=200, repeats=1, exact_scales=True)

        scales = {task.scale for task in tasks}
        assert scales == {64, 128}
        assert 256 not in scales and 512 not in scales

    def test_exact_scales_with_per_dtype_scales(self, simple_workload_factory, cluster_config_fixture):
        """With per-dtype scales (fp8:[128,256], bf16:[256,512]), exact_scales prevents expansion."""
        dtypes = {"fp8": [128, 256], "bf16": [256, 512]}
        workloads = simple_workload_factory("pretrain_test", scales=None, dtypes=dtypes)
        cluster_config = _build_cluster_config(installed=["pretrain_test"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=1024, repeats=1, exact_scales=True)

        fp8_scales = {t.scale for t in tasks if t.dtype == "fp8"}
        bf16_scales = {t.scale for t in tasks if t.dtype == "bf16"}
        all_scales = {t.scale for t in tasks}

        assert fp8_scales == {128, 256}
        assert bf16_scales == {256, 512}
        assert 1024 not in all_scales


class TestSubmitAllEdgeCases:
    """Test edge cases and error conditions for submit-all."""

    def test_missing_gpu_type_in_config(self, workloads_fixture):
        """Test handling when GPU type is missing from cluster config."""
        cluster_config = _build_cluster_config(gpu_type="", installed=["pretrain_nemotron"])

        tasks = generate_submit_all_tasks(workloads_fixture, cluster_config, max_scale=256, repeats=1, profile=False)

        assert len(tasks) == 0

    def test_workload_without_scales(self, cluster_config_fixture):
        """Test handling workload with missing scales."""
        workloads = {
            "pretrain_broken": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp8"],
                                        # Missing scales
                                    }
                                ]
                            }
                        }
                    }
                },
                "workload_type": "pretrain",
            }
        }

        cluster_config = _build_cluster_config(installed=["pretrain_broken"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=256, repeats=1, profile=False)

        assert len(tasks) == 0

    def test_single_dtype_string_handling(self, cluster_config_fixture):
        """Test that single dtype as string is handled correctly."""
        workloads = {
            "pretrain_single_dtype": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": "fp8",  # String instead of list
                                        "scales": [32],
                                        "exact_scales": True,  # Only use exact scales to avoid power-of-2 extension
                                    }
                                ]
                            }
                        }
                    }
                },
                "workload_type": "pretrain",
            }
        }

        cluster_config = _build_cluster_config(installed=["pretrain_single_dtype"])

        tasks = generate_submit_all_tasks(workloads, cluster_config, max_scale=256, repeats=1, profile=False)

        assert len(tasks) == 1
        assert tasks[0].dtype == "fp8"

    def test_profiling_parameter(self, workloads_fixture, cluster_config_fixture):
        """Test that profiling parameter is correctly applied to all tasks."""
        tasks = generate_submit_all_tasks(
            workloads_fixture, cluster_config_fixture, max_scale=128, repeats=1, profile=True
        )

        # All tasks should have profiling enabled
        for task in tasks:
            assert task.profile is True


class TestParseCommaList:
    """Tests for parse_comma_list utility."""

    def test_simple_list(self):
        assert parse_comma_list("fp8,bf16") == ["fp8", "bf16"]

    def test_strips_whitespace(self):
        assert parse_comma_list(" fp8 , bf16 ") == ["fp8", "bf16"]

    def test_single_value(self):
        assert parse_comma_list("fp8") == ["fp8"]

    def test_empty_string(self):
        assert parse_comma_list("") == []

    def test_none(self):
        assert parse_comma_list(None) == []

    def test_trailing_comma(self):
        assert parse_comma_list("fp8,bf16,") == ["fp8", "bf16"]


class TestTaskGenerationRequestValidate:
    """Tests for TaskGenerationRequest.validate() rules."""

    def _req(self, **kwargs):
        """Helper to create a request with minimal defaults."""
        defaults = {"workloads": {}, "cluster_config": _build_cluster_config(installed=[])}
        defaults.update(kwargs)
        return TaskGenerationRequest(**defaults)

    def test_file_mode_rejects_workload_mix(self):
        req = self._req(file_path="tasks.yaml", workload="pretrain_foo")
        with pytest.raises(ValidationError, match="Cannot mix --file"):
            req.validate()

    def test_file_mode_rejects_max_scale_mix(self):
        req = self._req(file_path="tasks.yaml", max_scale=128)
        with pytest.raises(ValidationError, match="Cannot mix --file"):
            req.validate()

    def test_file_mode_rejects_min_scale_mix(self):
        req = self._req(file_path="tasks.yaml", min_scale=True)
        with pytest.raises(ValidationError, match="Cannot mix --file"):
            req.validate()

    def test_model_size_requires_workload(self):
        req = self._req(model_size="7b", scale="128")
        with pytest.raises(ValidationError, match="--model-size requires --workload"):
            req.validate()

    def test_model_size_rejects_multiple_workloads(self):
        req = self._req(workload="pretrain_a,pretrain_b", model_size="7b", scale="128")
        with pytest.raises(ValidationError, match="cannot be used with multiple workloads"):
            req.validate()

    def test_scale_mutual_exclusivity(self):
        req = self._req(workload="pretrain_foo", scale="128", max_scale=256)
        with pytest.raises(ValidationError, match="Cannot use --scale with --max-scale"):
            req.validate()

    def test_scale_required_outside_file_mode(self):
        req = self._req(workload="pretrain_foo")
        with pytest.raises(ValidationError, match="Must specify scale parameter"):
            req.validate()

    def test_valid_explicit_mode(self):
        """Valid explicit mode should not raise."""
        req = self._req(workload="pretrain_foo", model_size="7b", scale="128")
        req.validate()  # Should not raise

    def test_valid_discovery_mode(self):
        """Valid discovery mode should not raise."""
        req = self._req(workload="pretrain_foo_7b", max_scale=256)
        req.validate()  # Should not raise

    def test_valid_file_mode(self):
        """Valid file mode should not raise."""
        req = self._req(file_path="tasks.yaml")
        req.validate()  # Should not raise

    def test_valid_min_scale_mode(self):
        """Valid min_scale mode should not raise."""
        req = self._req(workload="pretrain_foo", min_scale=True)
        req.validate()  # Should not raise

    def test_force_requires_single_explicit_task(self):
        req = self._req(workload="pretrain_foo_7b", max_scale=256, force=True)
        with pytest.raises(ValidationError, match="only supported for a single explicit task"):
            req.validate()

    def test_force_rejects_multi_value_arguments(self):
        req = self._req(workload="pretrain_foo", model_size="7b", dtype="fp8,bf16", scale="128", force=True)
        with pytest.raises(ValidationError, match="must be single values: --dtype"):
            req.validate()

    def test_force_accepts_workload_size_without_model_size(self):
        req = self._req(workload="pretrain_foo_7b", model_size=None, dtype="fp8", scale="128", force=True)
        req.validate()  # Should not raise

    def test_force_rejects_workload_size_without_parseable_size(self):
        req = self._req(workload="pretrain_foo", model_size=None, dtype="fp8", scale="128", force=True)
        with pytest.raises(ValidationError, match="only supported for a single explicit task"):
            req.validate()

    def test_redundant_size_suffix_stripped_when_model_size_provided(self):
        """'-w pretrain_foo_7b -s 7b' should normalize workload to 'pretrain_foo'."""
        req = self._req(workload="pretrain_foo_7b", model_size="7b", scale="128")
        req.validate()
        assert req.workload == "pretrain_foo"

    def test_redundant_trillion_size_suffix_stripped_when_model_size_provided(self):
        """'-w pretrain_foo_1T -s 1T' should normalize workload and model size."""
        req = self._req(workload="pretrain_foo_1T", model_size="1T", scale="128")
        req.validate()
        assert req.workload == "pretrain_foo"
        assert req.model_size == "1t"

    def test_mismatched_size_suffix_rejected(self):
        """'-w pretrain_foo_7b -s 13b' should be rejected as conflicting."""
        req = self._req(workload="pretrain_foo_7b", model_size="13b", scale="128")
        with pytest.raises(ValidationError, match="implies size '7b' but --model-size is '13b'"):
            req.validate()


class TestGenerateTasksIntegration:
    """Integration tests for generate_tasks() dispatch with real workload fixtures.

    These tests exercise the full path from TaskGenerationRequest through to
    WorkloadTask output without any mocking of internal logic.
    """

    @pytest.fixture
    def workloads(self):
        return {
            "pretrain_llama": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp8", "bf16"],
                                        "scales": [8, 16, 32],
                                    },
                                    {
                                        "model_size": "70b",
                                        "dtypes": ["bf16"],
                                        "scales": [64, 128],
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
    def cluster_config(self):
        return _build_cluster_config(installed=["pretrain_llama"])

    def test_explicit_mode_with_model_size(self, workloads, cluster_config):
        """Explicit mode: -w pretrain_llama -s 7b -d fp8 --scale 16"""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                model_size="7b",
                dtype="fp8",
                scale="16",
            )
        )
        assert len(tasks) == 1
        assert tasks[0].workload_key == "pretrain_llama"
        assert tasks[0].model_size == "7b"
        assert tasks[0].dtype == "fp8"
        assert tasks[0].scale == 16

    def test_explicit_mode_multiple_scales(self, workloads, cluster_config):
        """Explicit mode with comma-separated scales: --scale 8,16,32"""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                model_size="7b",
                dtype="fp8",
                scale="8,16,32",
            )
        )
        scales = {t.scale for t in tasks}
        assert scales == {8, 16, 32}

    def test_discovery_mode_max_scale(self, workloads, cluster_config):
        """Discovery mode: -w pretrain_llama_7b --max-scale 16"""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama_7b",
                max_scale=16,
            )
        )
        # Should get fp8 and bf16 at scales 8 and 16
        assert len(tasks) == 4
        scales = {t.scale for t in tasks}
        assert scales == {8, 16}
        dtypes = {t.dtype for t in tasks}
        assert dtypes == {"fp8", "bf16"}

    def test_discovery_mode_min_scale(self, workloads, cluster_config):
        """Discovery mode: -w pretrain_llama --min-scale"""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                min_scale=True,
            )
        )
        # 7b: min=8 (fp8 + bf16) + 70b: min=64 (bf16 only) = 3 tasks
        assert len(tasks) == 3
        for t in tasks:
            if t.model_size == "7b":
                assert t.scale == 8
            elif t.model_size == "70b":
                assert t.scale == 64

    def test_discovery_mode_dtype_filter(self, workloads, cluster_config):
        """Discovery with dtype filter: -w pretrain_llama --max-scale 32 -d fp8"""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                max_scale=32,
                dtype="fp8",
            )
        )
        # Only fp8 tasks, only 7b (70b min scale is 64 > 32)
        # 7b fp8 at scales 8, 16, 32 = 3 tasks
        assert len(tasks) == 3
        for t in tasks:
            assert t.dtype == "fp8"
            assert t.model_size == "7b"

    def test_file_mode_yaml(self, workloads, cluster_config, tmp_path):
        """File mode with YAML spec."""
        yaml_content = """
pretrain_llama_7b:
  dtypes: fp8
  scales: 16
  repeats: 1
  tasks:
    - {}
"""
        spec = tmp_path / "tasks.yaml"
        spec.write_text(yaml_content)

        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                file_path=str(spec),
            )
        )
        assert len(tasks) == 1
        assert tasks[0].workload_key == "pretrain_llama"
        assert tasks[0].model_size == "7b"
        assert tasks[0].dtype == "fp8"
        assert tasks[0].scale == 16

    def test_repeats(self, workloads, cluster_config):
        """Repeats multiply task count."""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                model_size="7b",
                dtype="fp8",
                scale="8",
                repeats=3,
            )
        )
        assert len(tasks) == 3
        assert all(t.scale == 8 for t in tasks)

    def test_proxy_flag(self, cluster_config):
        """Proxy flag propagates to tasks and uses proxy_scales."""
        workloads_with_proxy = {
            "pretrain_llama": {
                "metadata": {
                    "run": {
                        "gpu_configs": {
                            "h100": {
                                "model_configs": [
                                    {
                                        "model_size": "7b",
                                        "dtypes": ["fp8"],
                                        "scales": [8, 16, 32],
                                        "proxy_scales": [8],
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
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads_with_proxy,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                model_size="7b",
                dtype="fp8",
                scale="8",
                proxy=True,
            )
        )
        assert len(tasks) == 1
        assert tasks[0].proxy is True

    def test_force_preserves_requested_unsupported_scale(self, workloads, cluster_config):
        """Force mode should keep an explicitly requested scale even if metadata would skip it."""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                model_size="7b",
                dtype="fp8",
                scale="24",
                force=True,
            )
        )
        assert len(tasks) == 1
        assert tasks[0].scale == 24
        assert tasks[0].dtype == "fp8"

    def test_force_preserves_requested_unsupported_dtype(self, workloads, cluster_config):
        """Force mode should keep an explicitly requested dtype even if metadata would skip it."""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama",
                model_size="7b",
                dtype="fp16",
                scale="16",
                force=True,
            )
        )
        assert len(tasks) == 1
        assert tasks[0].scale == 16
        assert tasks[0].dtype == "fp16"

    def test_force_still_requires_supported_model_selection(self, workloads, cluster_config):
        """Force mode should still reject workload/model combinations missing from metadata."""
        with pytest.raises(ValueError, match="Model size '13b' not supported"):
            generate_tasks(
                TaskGenerationRequest(
                    workloads=workloads,
                    cluster_config=cluster_config,
                    workload="pretrain_llama",
                    model_size="13b",
                    dtype="fp16",
                    scale="16",
                    force=True,
                )
            )

    def test_force_with_workload_size_style(self, workloads, cluster_config):
        """Force mode should work with -w workload_size style (no -s)."""
        tasks = generate_tasks(
            TaskGenerationRequest(
                workloads=workloads,
                cluster_config=cluster_config,
                workload="pretrain_llama_7b",
                dtype="fp8",
                scale="24",
                force=True,
            )
        )
        assert len(tasks) == 1
        assert tasks[0].workload_key == "pretrain_llama"
        assert tasks[0].model_size == "7b"
        assert tasks[0].scale == 24
        assert tasks[0].dtype == "fp8"

    def test_validation_error_propagates(self, workloads, cluster_config):
        """Invalid request raises ValueError from validate()."""
        with pytest.raises(ValueError, match="Cannot mix --file"):
            generate_tasks(
                TaskGenerationRequest(
                    workloads=workloads,
                    cluster_config=cluster_config,
                    file_path="tasks.yaml",
                    workload="pretrain_llama",
                )
            )


if __name__ == "__main__":
    pytest.main([__file__])
