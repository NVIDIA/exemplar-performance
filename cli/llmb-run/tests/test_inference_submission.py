# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmb_run.task_generation import generate_submit_all_tasks


@pytest.fixture
def inference_workload_fixture():
    return {
        "inference_test_job": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "70b",
                                    "dtypes": ["fp8"],
                                    "scales": [8],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/inference",
            "workload": "test_job",
            "workload_type": "inference",
        },
        "pretrain_standard": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "8b",
                                    "dtypes": ["bf16"],
                                    "scales": [8],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/fake/pretrain",
            "workload": "standard",
            "workload_type": "pretrain",
        },
    }


@pytest.fixture
def cluster_config_fixture(cluster_config_factory):
    return cluster_config_factory({"workloads": {"installed": ["inference_test_job", "pretrain_standard"]}})


def test_inference_excluded_by_default(inference_workload_fixture, cluster_config_fixture):
    """Test that inference workloads are strictly excluded when no filter is applied (Discovery Mode)."""
    tasks = generate_submit_all_tasks(
        inference_workload_fixture, cluster_config_fixture, max_scale=None, workload_filter=None  # Default behavior
    )

    workload_keys = {t.workload_key for t in tasks}
    assert "pretrain_standard" in workload_keys
    assert "inference_test_job" not in workload_keys


def test_inference_included_explicitly(inference_workload_fixture, cluster_config_fixture):
    """Test that inference workloads are allowed when explicitly requested via filter."""
    tasks = generate_submit_all_tasks(
        inference_workload_fixture,
        cluster_config_fixture,
        max_scale=None,
        workload_filter=["inference_test_job"],  # Explicit request
    )

    workload_keys = {t.workload_key for t in tasks}
    assert "inference_test_job" in workload_keys
    assert len(tasks) > 0
    assert tasks[0].workload_key == "inference_test_job"
