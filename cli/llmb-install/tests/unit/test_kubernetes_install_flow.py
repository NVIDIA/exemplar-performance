# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A Kubernetes install provisions images, HuggingFace assets and tools in-cluster, not locally."""

from pathlib import Path
from unittest.mock import Mock

import pytest
import yaml

from llmb_install.config.models import InstallConfig, KubernetesConfig, SlurmConfig
from llmb_install.core.installer import Installer

LOCAL_SETUP = (
    "setup_compute_uv_binaries",
    "get_required_images",
    "fetch_container_images",
    "download_huggingface_files_for_workloads",
    "fetch_and_install_tools",
)


def _config(tmp_path, kubernetes=None):
    return InstallConfig(
        install_path=str(tmp_path / "install"),
        venv_type="uv",
        gpu_type="gb300",
        node_architecture="aarch64",
        selected_workloads=["w1"],
        kubernetes=kubernetes,
        slurm=(
            None if kubernetes else SlurmConfig(account="a", gpu_partition="g", cpu_partition="c", gpu_partition_gres=4)
        ),
        nvl_domain=72,
        dev_mode=True,
    )


@pytest.fixture
def local_setup(monkeypatch):
    """Patch everything a local install would do, and return the mocks that must not run."""
    mocks = {name: Mock(return_value={}) for name in LOCAL_SETUP}
    # A local install would try these for real (enroot, GitHub downloads).
    mocks["get_required_tools"] = Mock(return_value=[("nsys", "1.0")])
    for name, mock in mocks.items():
        monkeypatch.setattr(f"llmb_install.core.installer.{name}", mock)
    for name in (
        "install_state_exists",
        "clear_install_state",
        "save_install_state",
        "create_llmb_run_symlink",
        "print_dependency_group_summary",
        "run_setup_tasks",
        "save_system_config",
        "create_virtual_environment",
    ):
        monkeypatch.setattr(f"llmb_install.core.installer.{name}", Mock(return_value=False))
    monkeypatch.setattr("llmb_install.core.installer.check_kubernetes_prerequisites", Mock(return_value=True))
    monkeypatch.setattr("llmb_install.core.installer.get_venv_environment", Mock(return_value={}))
    return mocks


def test_kubernetes_install_does_not_download_images_assets_or_tools(tmp_path, local_setup):
    config = _config(tmp_path, KubernetesConfig(namespace="ns", workdir_pvc="pvc"))

    Installer()._perform_installation(config, {"w1": {"name": "w1", "path": "/repo/w1"}})

    for name, mock in local_setup.items():
        assert not mock.called, f"{name} must not run for a Kubernetes install"


def test_kubernetes_install_runs_the_standard_venv_and_clone_flow(tmp_path, local_setup, monkeypatch):
    """Only images/HF/tools are skipped: venvs and git clones go through the standard machinery."""
    create_venv = Mock()
    install_deps = Mock()
    clone = Mock()
    monkeypatch.setattr("llmb_install.core.installer.create_virtual_environment", create_venv)
    monkeypatch.setattr("llmb_install.core.installer.install_dependencies", install_deps)
    monkeypatch.setattr("llmb_install.core.installer.clone_git_repos", clone)

    config = _config(tmp_path, KubernetesConfig(namespace="ns"))
    workloads = {
        "w1": {
            "name": "w1",
            "path": "/repo/w1",
            "setup": {
                "dependencies": {
                    "git": {"megatron_bridge": {"url": "https://example.com/mb.git", "commit": "abc"}},
                    "pip": [{"package": "nemo_run", "url": "https://example.com/run.git", "commit": "def"}],
                }
            },
        }
    }

    Installer()._perform_installation(config, workloads)

    create_venv.assert_called_once()
    install_deps.assert_called_once()
    clone.assert_called_once()

    # The venv is created under the install path and both the dependency
    # install and the recorded workload config point at it, so the launcher
    # can activate it.
    venv_path = create_venv.call_args.args[0]
    assert venv_path.startswith(str(config.install_path) + "/venvs/")
    assert install_deps.call_args.args[0] == venv_path
    cluster_config = yaml.safe_load((Path(config.install_path) / "cluster_config.yaml").read_text())
    assert cluster_config["workloads"]["config"]["w1"]["venv_path"] == venv_path
    assert cluster_config["execution_target"] == "kubernetes"
    # and the local downloads are still skipped
    for name, mock in local_setup.items():
        assert not mock.called, f"{name} must not run for a Kubernetes install"


def test_slurm_install_still_downloads_images_assets_and_tools(tmp_path, local_setup):
    config = _config(tmp_path)

    Installer()._perform_installation(config, {"w1": {"name": "w1", "path": "/repo/w1"}})

    for name in LOCAL_SETUP:
        assert local_setup[name].called, f"{name} should run for a Slurm install"


def test_ngc_key_from_the_prompt_reaches_image_pull_secret_creation(tmp_path, local_setup, monkeypatch):
    create_secrets = Mock()
    monkeypatch.setattr("llmb_install.cluster.kubernetes.create_image_pull_secrets", create_secrets)
    config = _config(
        tmp_path,
        KubernetesConfig(namespace="ns", image_pull_secrets=["ngc-secret"], ngc_api_key="nvapi-KEY"),
    )

    Installer()._perform_installation(config, {"w1": {"name": "w1", "path": "/repo/w1"}})

    create_secrets.assert_called_once()
    assert create_secrets.call_args.kwargs["ngc_api_key"] == "nvapi-KEY"
    assert create_secrets.call_args.kwargs["secret_names"] == ["ngc-secret"]


def _resume_edit(tmp_path, resume_config, monkeypatch):
    """Run the resume 'edit' prompts with every Slurm prompt wired to fail."""

    def slurm_prompt_must_not_run(*args, **kwargs):
        raise AssertionError("a Kubernetes install must not be asked for Slurm settings")

    monkeypatch.setattr("llmb_install.ui.prompts.slurm.prompt_slurm_info", slurm_prompt_must_not_run)
    monkeypatch.setattr("llmb_install.core.installer.prompt_image_download_method", slurm_prompt_must_not_run)
    monkeypatch.setattr(
        "llmb_install.ui.prompts.environment.prompt_environment_variables", lambda ui, **kw: {"HF_TOKEN": "hf_x"}
    )
    installer = Installer()
    installer.workloads = {}
    return installer._collect_limited_configuration(
        ui_mode="simple", resume_config=resume_config, completed_workloads=["w1"], args=None
    )


def test_resume_edit_of_a_kubernetes_install_keeps_the_kubernetes_target(tmp_path, monkeypatch):
    resume = _config(tmp_path, KubernetesConfig(namespace="ns", workdir_pvc="pvc"))

    edited = _resume_edit(tmp_path, resume, monkeypatch)

    assert edited.kubernetes == resume.kubernetes
    assert edited.slurm is None
    assert edited.image_download_method == "local"


def test_resume_edit_of_a_slurm_install_still_prompts_for_slurm(tmp_path, monkeypatch):
    resume = _config(tmp_path)
    prompted = []
    monkeypatch.setattr(
        "llmb_install.ui.prompts.slurm.prompt_slurm_info",
        lambda ui, defaults, express_mode: (
            prompted.append(defaults) or {"slurm": {"account": "a", "gpu_partition": "g", "cpu_partition": "c"}}
        ),
    )
    monkeypatch.setattr("llmb_install.core.installer.prompt_image_download_method", lambda ui, d, express_mode: "slurm")
    monkeypatch.setattr("llmb_install.ui.prompts.environment.prompt_environment_variables", lambda ui, **kw: {})
    installer = Installer()
    installer.workloads = {}

    edited = installer._collect_limited_configuration(
        ui_mode="simple", resume_config=resume, completed_workloads=["w1"], args=None
    )

    assert prompted and edited.slurm.account == "a"
    assert edited.kubernetes is None
    assert edited.image_download_method == "slurm"
