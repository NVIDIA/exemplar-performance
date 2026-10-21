# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

# Ensure the package under tests (located in `src/`) is discoverable without installation.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from llmb_run.config_manager import ClusterConfig  # noqa: E402


@pytest.fixture
def mock_cluster_config():
    """Return a minimal, valid cluster configuration for tests."""
    return build_cluster_config(
        {
            "workloads": {
                "installed": ["training_foo"],
            }
        }
    )


def _deep_merge(base: dict, override: dict) -> dict:
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_merge(base[key], value)
        else:
            base[key] = value
    return base


def build_cluster_config(override: dict | None = None) -> ClusterConfig:
    raw = {
        "launcher": {
            "llmb_repo": "/tmp",
            "llmb_install": "/tmp",
            "gpu_type": "h100",
        },
        "slurm": {
            "account": "test_account",
            "gpu_partition": "batch",
            "gpu_gres": "8",
        },
        "workloads": {
            "installed": [],
            "config": {},
        },
        "environment": {},
        "cwd": Path.cwd(),
    }

    if override:
        raw = _deep_merge(deepcopy(raw), override)

    return ClusterConfig.from_dict(raw)


@pytest.fixture
def cluster_config_factory():
    return build_cluster_config


@pytest.fixture
def mock_workloads():
    """Return a single synthetic workload entry that matches the dummy config."""
    return {
        "training_foo": {
            "metadata": {
                "run": {
                    "launcher_type": "sbatch",
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": ["fp16", "bf16"],
                                    "scales": [8, 16, 32],
                                    "proxy_scales": [4, 8],
                                }
                            ]
                        }
                    },
                }
            },
            "dir": "/tmp",
            "workload": "foo",
            "workload_type": "pretrain",
        },
    }


@pytest.fixture
def runner():
    """Create a CLI runner for tests."""
    return CliRunner()


@pytest.fixture
def mock_dependencies(mock_cluster_config, mock_workloads):
    """Patch all dependencies before importing the app.

    This fixture automatically mocks:
    - llmb_run.main.get_cluster_config
    - llmb_run.main.get_workloads
    - llmb_run.main.validate_workload_with_details (always valid)
    - llmb_run.task_loader.validate_workload_with_details (always valid)
    """

    def _always_valid(*a, **k):
        return (True, None, "", [])

    with (
        patch('llmb_run.main.get_cluster_config', return_value=mock_cluster_config),
        patch('llmb_run.main.get_workloads', return_value=mock_workloads),
        patch('llmb_run.main.validate_workload_with_details', side_effect=_always_valid),
        patch('llmb_run.task_loader.validate_workload_with_details', side_effect=_always_valid),
    ):
        yield
