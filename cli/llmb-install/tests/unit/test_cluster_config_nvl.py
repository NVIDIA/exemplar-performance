# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

import yaml

from llmb_install.config.cluster import create_cluster_config


def _write_config(tmp_path, **kwargs):
    slurm_info = {
        "slurm": {
            "account": "acct",
            "gpu_partition": "gpu",
            "cpu_partition": "cpu",
            "gpu_partition_gres": 4,
            "cpu_partition_gres": None,
        }
    }
    with patch("llmb_install.config.cluster.get_cluster_name", return_value=None):
        create_cluster_config(
            str(tmp_path),
            "/repo",
            ["workload"],
            slurm_info,
            {},
            "gb200",
            "uv",
            {},
            nvl_domain=kwargs.get("nvl_domain"),
            nvl_block_sizes=kwargs.get("nvl_block_sizes"),
            existing_cluster_config=kwargs.get("existing_cluster_config"),
        )
    return yaml.safe_load((tmp_path / "cluster_config.yaml").read_text())


def test_cluster_config_writes_nvlink_fields(tmp_path):
    config = _write_config(tmp_path, nvl_domain=72, nvl_block_sizes=[18])
    assert config["nvl_domain"] == 72
    assert config["nvl_block_sizes"] == [18]


def test_cluster_config_omits_unset_nvlink_fields(tmp_path):
    config = _write_config(tmp_path)
    assert "nvl_domain" not in config
    assert "nvl_block_sizes" not in config


def test_cluster_config_preserves_nvlink_fields_incrementally(tmp_path):
    existing = {"nvl_domain": 576, "nvl_block_sizes": [9, 144], "workloads": {"installed": [], "config": {}}}
    config = _write_config(tmp_path, existing_cluster_config=existing)
    assert config["nvl_domain"] == 576
    assert config["nvl_block_sizes"] == [9, 144]
