# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Workloads are offered only for the execution targets their launch script declares."""

import pytest
from pydantic import ValidationError

from llmb_install.cluster.gpu import filter_workloads_by_gpu_type
from llmb_install.core.installer import Installer

WORKLOADS = {
    "k8s_ready": {
        "run": {
            "execution_targets": {"by_gpu": {"default": ["slurm", "kubernetes"], "h100": ["slurm"]}},
            "gpu_configs": {"gb300": {}, "h100": {}},
        }
    },
    "slurm_only": {"run": {"gpu_configs": {"gb300": {}}}},  # e.g. the MLPerf recipes
    "k8s_ready_h100_only": {"run": {"execution_targets": ["slurm", "kubernetes"], "gpu_configs": {"h100": {}}}},
    "k8s_only": {"run": {"execution_targets": ["kubernetes"], "gpu_configs": {"gb300": {}}}},
}


def test_slurm_filtering_keeps_slurm_capable_workloads():
    assert set(filter_workloads_by_gpu_type(WORKLOADS, "gb300")) == {"k8s_ready", "slurm_only"}


def test_kubernetes_filtering_drops_workloads_that_do_not_declare_support():
    assert set(filter_workloads_by_gpu_type(WORKLOADS, "gb300", execution_target="kubernetes")) == {
        "k8s_ready",
        "k8s_only",
    }


def test_gpu_override_keeps_h100_slurm_while_hiding_its_kubernetes_variant():
    assert "k8s_ready" in filter_workloads_by_gpu_type(WORKLOADS, "h100", execution_target="slurm")
    assert "k8s_ready" not in filter_workloads_by_gpu_type(WORKLOADS, "h100", execution_target="kubernetes")
    assert "k8s_ready" in filter_workloads_by_gpu_type(WORKLOADS, "gb300", execution_target="kubernetes")


def test_selecting_h100_kubernetes_override_reports_target_not_gpu(capsys):
    installer = Installer()
    installer.workloads = WORKLOADS

    with pytest.raises(SystemExit):
        installer._prepare_workloads("h100", ["k8s_ready"], execution_target="kubernetes")

    out = capsys.readouterr().out
    assert "do not support execution target 'kubernetes'" in out
    assert "do not support gpu_type 'h100'" not in out


def test_slurm_filtering_drops_kubernetes_only_workloads():
    assert "k8s_only" not in filter_workloads_by_gpu_type(WORKLOADS, "gb300", execution_target="slurm")


def test_an_undeclared_execution_targets_defaults_to_slurm():
    workloads = {"w": {"run": {"gpu_configs": {"gb300": {}}}}}

    assert set(filter_workloads_by_gpu_type(workloads, "gb300", execution_target="slurm")) == {"w"}
    assert filter_workloads_by_gpu_type(workloads, "gb300", execution_target="kubernetes") == {}


def test_a_scalar_execution_targets_is_rejected():
    workloads = {"w": {"run": {"execution_targets": "kubernetes", "gpu_configs": {"gb300": {}}}}}

    with pytest.raises(ValidationError):
        filter_workloads_by_gpu_type(workloads, "gb300", execution_target="kubernetes")


def test_an_unknown_execution_target_value_is_rejected():
    workloads = {"w": {"run": {"execution_targets": ["k8s"], "gpu_configs": {"gb300": {}}}}}

    with pytest.raises(ValidationError):
        filter_workloads_by_gpu_type(workloads, "gb300", execution_target="kubernetes")


def test_an_empty_execution_targets_is_rejected():
    workloads = {"w": {"run": {"execution_targets": [], "gpu_configs": {"gb300": {}}}}}

    with pytest.raises(ValidationError):
        filter_workloads_by_gpu_type(workloads, "gb300", execution_target="kubernetes")


def test_selecting_a_slurm_only_workload_on_kubernetes_says_so(capsys):
    installer = Installer()
    installer.workloads = WORKLOADS

    with pytest.raises(SystemExit) as exc_info:
        installer._prepare_workloads("gb300", ["slurm_only"], execution_target="kubernetes")

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "do not support execution target 'kubernetes'" in out
    assert "slurm_only" in out
    assert "gpu_type" not in out  # the GPU is fine; it is the target that is not


def test_selecting_a_kubernetes_only_workload_on_slurm_says_so(capsys):
    installer = Installer()
    installer.workloads = WORKLOADS

    with pytest.raises(SystemExit) as exc_info:
        installer._prepare_workloads("gb300", ["k8s_only"], execution_target="slurm")

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "do not support execution target 'slurm'" in out
    assert "k8s_only" in out
    assert "gpu_type" not in out  # the GPU is fine; it is the target that is not


def test_a_kubernetes_ready_workload_is_accepted():
    installer = Installer()
    installer.workloads = WORKLOADS

    prepared = installer._prepare_workloads("gb300", ["k8s_ready"], execution_target="kubernetes")

    assert "k8s_ready" in prepared


def test_workload_listing_on_a_kubernetes_install_hides_slurm_only_workloads(monkeypatch, capsys):
    from llmb_install.config.models import KubernetesConfig
    from llmb_install.config.system import SystemConfig

    system_config = SystemConfig(
        venv_type="uv",
        gpu_type="gb300",
        node_architecture="aarch64",
        kubernetes=KubernetesConfig(namespace="ns"),
    )
    monkeypatch.setattr("llmb_install.config.system.system_config_exists", lambda: True)
    monkeypatch.setattr("llmb_install.config.system.load_system_config", lambda: system_config)
    installer = Installer()
    installer.workloads = WORKLOADS

    installer._list_available_workloads()

    out = capsys.readouterr().out
    assert "k8s_ready" in out
    assert "slurm_only" not in out
