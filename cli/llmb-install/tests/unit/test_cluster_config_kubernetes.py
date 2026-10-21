# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""cluster_config.yaml must carry the Kubernetes settings llmb-run reads."""

import yaml

from llmb_install.config.cluster import create_cluster_config
from llmb_install.config.models import KubernetesConfig


def _write(tmp_path, kubernetes=None, existing=None):
    create_cluster_config(
        str(tmp_path),
        "/repo",
        ["w1"],
        {},
        {},
        "gb300",
        "uv",
        {},
        node_architecture="aarch64",
        existing_cluster_config=existing,
        kubernetes_info=kubernetes.model_dump() if kubernetes else None,
    )
    path = tmp_path / "cluster_config.yaml"
    return yaml.safe_load(path.read_text()), path


def test_launcher_settings_are_written_to_the_kubernetes_block(tmp_path):
    config, _ = _write(
        tmp_path,
        KubernetesConfig(
            namespace="ns",
            workdir_pvc="my-pvc",
            node_selector_json='{"gpu":"gb300"}',
            timeout_per_job="6h",
            kubeconfig="/etc/kube/config",
        ),
    )

    assert config["kubernetes"] == {
        "namespace": "ns",
        "kubeconfig": "/etc/kube/config",
        "workdir_pvc": "my-pvc",
        "node_selector_json": '{"gpu":"gb300"}',
        "timeout_per_job": "6h",
    }


def test_install_time_secrets_and_flags_are_not_written(tmp_path):
    kube = KubernetesConfig(namespace="ns", ngc_api_key="nvapi-SECRET", run_nvcrectl_init=True)

    config, path = _write(tmp_path, kube)

    assert "nvapi-SECRET" not in path.read_text()
    assert "ngc_api_key" not in kube.model_dump()  # not saved to system config / install state either
    assert kube.ngc_api_key == "nvapi-SECRET"  # but still available to create the pull secrets
    assert "run_nvcrectl_init" not in config["kubernetes"]


def test_incremental_install_keeps_recorded_settings_that_the_new_run_leaves_unset(tmp_path):
    existing = {"kubernetes": {"namespace": "ns", "workdir_pvc": "my-pvc", "node_selector_json": '{"a":"b"}'}}
    config, _ = _write(tmp_path, KubernetesConfig(namespace="ns"), existing=existing)

    assert config["kubernetes"]["workdir_pvc"] == "my-pvc"
    assert config["kubernetes"]["node_selector_json"] == '{"a":"b"}'
