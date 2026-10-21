# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for shared-venv workload cloning in installer flows."""

import os
from unittest.mock import Mock

import pytest

from llmb_install.config.models import InstallConfig
from llmb_install.core.installer import Installer

DEP_HASH = "0123456789abcdef"


def make_git_dep(url: str, commit: str, install_type: str = "clone"):
    """Build a git dependency map entry for tests."""
    return {
        "repo": {
            "url": url,
            "commit": commit,
            "install_method": {"type": install_type},
        }
    }


def make_workload(name: str):
    """Build a minimal workload payload for installer tests."""
    return {
        "name": name,
        "path": f"/repo/{name}",
        "setup": {"dependencies": {"pip": ["numpy==1.24.0"]}},
    }


def make_install_config(
    install_path: str, selected_workloads: list[str], environment_vars: dict[str, str] | None = None
) -> InstallConfig:
    """Build a minimal install config for _perform_installation tests."""
    return InstallConfig(
        install_path=install_path,
        venv_type="venv",
        gpu_type="h100",
        node_architecture="x86_64",
        selected_workloads=selected_workloads,
        install_method="local",
        environment_vars=environment_vars or {},
        dev_mode=True,
    )


def setup_common_mocks(monkeypatch):
    """Patch non-target side effects so tests focus on shared-venv behavior."""
    # Downloads uv tarballs from github.com otherwise: ~15 MB per test run,
    # and an outright failure on networks that block GitHub.
    monkeypatch.setattr("llmb_install.core.installer.setup_compute_uv_binaries", Mock())
    monkeypatch.setattr("llmb_install.core.installer.install_state_exists", Mock(return_value=False))
    monkeypatch.setattr("llmb_install.core.installer.clear_install_state", Mock())
    monkeypatch.setattr("llmb_install.core.installer.save_install_state", Mock())
    monkeypatch.setattr("llmb_install.core.installer.create_llmb_run_symlink", Mock())
    monkeypatch.setattr("llmb_install.core.installer.get_required_images", Mock(return_value={}))
    monkeypatch.setattr("llmb_install.core.installer.fetch_container_images", Mock())
    monkeypatch.setattr("llmb_install.core.installer.download_huggingface_files_for_workloads", Mock())
    monkeypatch.setattr("llmb_install.core.installer.get_required_tools", Mock(return_value=[]))
    monkeypatch.setattr("llmb_install.core.installer.fetch_and_install_tools", Mock())
    monkeypatch.setattr("llmb_install.core.installer.print_dependency_group_summary", Mock())
    monkeypatch.setattr("llmb_install.core.installer.run_setup_tasks", Mock())
    monkeypatch.setattr("llmb_install.core.installer.save_system_config", Mock())
    monkeypatch.setattr("llmb_install.core.installer.create_virtual_environment", Mock())
    monkeypatch.setattr("llmb_install.core.installer.get_venv_environment", Mock(return_value={}))
    monkeypatch.setattr("llmb_install.environment.venv_utils.build_venv_hash_mapping", Mock(return_value={}))


@pytest.fixture
def shared_workloads():
    """Provide two workloads in a single dependency group."""
    return {
        "w1": make_workload("w1"),
        "w2": make_workload("w2"),
    }


def test_shared_group_clones_per_workload_new_venv(monkeypatch, tmp_path, shared_workloads):
    """Shared groups should clone each workload's git deps even when venv is newly created."""
    setup_common_mocks(monkeypatch)

    monkeypatch.setattr(
        "llmb_install.core.installer.group_workloads_by_dependencies", Mock(return_value={DEP_HASH: ["w1", "w2"]})
    )
    monkeypatch.setattr("llmb_install.environment.venv_utils.should_reuse_venv", Mock(return_value=None))

    resolve_map = {
        "w1": {"git": make_git_dep("https://example.com/repo.git", "aaaa1111")},
        "w2": {"git": make_git_dep("https://example.com/repo.git", "bbbb2222")},
    }

    mock_resolve = Mock(side_effect=lambda workload: resolve_map[workload["name"]])
    mock_clone = Mock()
    mock_install_deps = Mock()
    mock_cluster_config = Mock()

    monkeypatch.setattr("llmb_install.core.installer._resolve_dependencies", mock_resolve)
    monkeypatch.setattr("llmb_install.core.installer.clone_git_repos", mock_clone)
    monkeypatch.setattr("llmb_install.core.installer.install_dependencies", mock_install_deps)
    monkeypatch.setattr("llmb_install.core.installer.create_cluster_config", mock_cluster_config)

    installer = Installer()
    installer.root_dir = "/repo"

    install_path = str(tmp_path / "install")
    install_config = make_install_config(install_path, ["w1", "w2"])

    installer._perform_installation(install_config, shared_workloads)

    assert mock_clone.call_count == 2
    mock_clone.assert_any_call(resolve_map["w1"]["git"], f"{install_path}/workloads/w1", install_path)
    mock_clone.assert_any_call(resolve_map["w2"]["git"], f"{install_path}/workloads/w2", install_path)

    assert mock_install_deps.call_count == 1
    _, _, install_dependencies_arg, first_workload_dir, _ = mock_install_deps.call_args.args
    assert install_dependencies_arg == resolve_map["w1"]
    assert first_workload_dir == f"{install_path}/workloads/w1"

    workload_venvs = mock_cluster_config.call_args.args[7]
    assert workload_venvs["w1"] == f"{install_path}/venvs/venv_{DEP_HASH[:12]}"
    assert workload_venvs["w2"] == f"{install_path}/venvs/venv_{DEP_HASH[:12]}"


def test_shared_group_clones_per_workload_reused_venv(monkeypatch, tmp_path, shared_workloads):
    """Shared groups should clone each workload's git deps when reusing an existing venv."""
    setup_common_mocks(monkeypatch)

    monkeypatch.setattr(
        "llmb_install.core.installer.group_workloads_by_dependencies", Mock(return_value={DEP_HASH: ["w1", "w2"]})
    )

    reused_venv_path = str(tmp_path / "install" / "venvs" / "existing")
    (tmp_path / "install" / "venvs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "install" / "venvs" / "existing").mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr("llmb_install.environment.venv_utils.should_reuse_venv", Mock(return_value=reused_venv_path))

    resolve_map = {
        "w1": {"git": make_git_dep("https://example.com/repo.git", "aaaa1111")},
        "w2": {"git": make_git_dep("https://example.com/repo.git", "bbbb2222")},
    }

    mock_resolve = Mock(side_effect=lambda workload: resolve_map[workload["name"]])
    mock_clone = Mock()
    mock_install_deps = Mock()
    mock_create_venv = Mock()
    mock_cluster_config = Mock()

    monkeypatch.setattr("llmb_install.core.installer._resolve_dependencies", mock_resolve)
    monkeypatch.setattr("llmb_install.core.installer.clone_git_repos", mock_clone)
    monkeypatch.setattr("llmb_install.core.installer.install_dependencies", mock_install_deps)
    monkeypatch.setattr("llmb_install.core.installer.create_virtual_environment", mock_create_venv)
    monkeypatch.setattr("llmb_install.core.installer.create_cluster_config", mock_cluster_config)

    installer = Installer()
    installer.root_dir = "/repo"

    install_path = str(tmp_path / "install")
    install_config = make_install_config(install_path, ["w1", "w2"])

    installer._perform_installation(
        install_config,
        shared_workloads,
        existing_workload_venvs={"prev": reused_venv_path},
    )

    assert mock_clone.call_count == 2
    mock_clone.assert_any_call(resolve_map["w1"]["git"], f"{install_path}/workloads/w1", install_path)
    mock_clone.assert_any_call(resolve_map["w2"]["git"], f"{install_path}/workloads/w2", install_path)

    mock_create_venv.assert_not_called()
    mock_install_deps.assert_not_called()

    workload_venvs = mock_cluster_config.call_args.args[7]
    assert workload_venvs["w1"] == reused_venv_path
    assert workload_venvs["w2"] == reused_venv_path


@pytest.mark.parametrize("reuse_existing", [False, True])
def test_shared_group_handles_missing_workload_deps_without_clone(
    monkeypatch, tmp_path, shared_workloads, reuse_existing
):
    """Workloads with no resolved deps should skip clone while still getting mapped to group venv."""
    setup_common_mocks(monkeypatch)

    monkeypatch.setattr(
        "llmb_install.core.installer.group_workloads_by_dependencies", Mock(return_value={DEP_HASH: ["w1", "w2"]})
    )

    if reuse_existing:
        reused_venv_path = str(tmp_path / "install" / "venvs" / "existing")
        (tmp_path / "install" / "venvs").mkdir(parents=True, exist_ok=True)
        (tmp_path / "install" / "venvs" / "existing").mkdir(parents=True, exist_ok=True)
        monkeypatch.setattr(
            "llmb_install.environment.venv_utils.should_reuse_venv", Mock(return_value=reused_venv_path)
        )
    else:
        reused_venv_path = None
        monkeypatch.setattr("llmb_install.environment.venv_utils.should_reuse_venv", Mock(return_value=None))

    resolve_map = {
        "w1": {"git": make_git_dep("https://example.com/repo.git", "aaaa1111")},
        "w2": None,
    }

    mock_resolve = Mock(side_effect=lambda workload: resolve_map[workload["name"]])
    mock_clone = Mock()
    mock_cluster_config = Mock()
    mock_install_deps = Mock()

    monkeypatch.setattr("llmb_install.core.installer._resolve_dependencies", mock_resolve)
    monkeypatch.setattr("llmb_install.core.installer.clone_git_repos", mock_clone)
    monkeypatch.setattr("llmb_install.core.installer.install_dependencies", mock_install_deps)
    monkeypatch.setattr("llmb_install.core.installer.create_cluster_config", mock_cluster_config)

    installer = Installer()
    installer.root_dir = "/repo"

    install_path = str(tmp_path / "install")
    install_config = make_install_config(install_path, ["w1", "w2"])

    installer._perform_installation(
        install_config,
        shared_workloads,
        existing_workload_venvs={"prev": reused_venv_path} if reuse_existing else None,
    )

    assert mock_clone.call_count == 1
    mock_clone.assert_called_once_with(resolve_map["w1"]["git"], f"{install_path}/workloads/w1", install_path)

    workload_venvs = mock_cluster_config.call_args.args[7]
    expected_venv_path = reused_venv_path if reuse_existing else f"{install_path}/venvs/venv_{DEP_HASH[:12]}"
    assert workload_venvs["w1"] == expected_venv_path
    assert workload_venvs["w2"] == expected_venv_path


def test_no_dependency_workload_skips_venv_and_runs_tasks(monkeypatch, tmp_path):
    """No-dependency workloads should install as a no-op setup without legacy warnings."""
    setup_common_mocks(monkeypatch)

    workload = {
        "w1": {
            "name": "w1",
            "path": "/repo/w1",
            "setup": {"tasks": [{"name": "prepare", "cmd": "echo prepare", "job_type": "local"}]},
        }
    }

    monkeypatch.setattr(
        "llmb_install.core.installer.group_workloads_by_dependencies", Mock(return_value={None: ["w1"]})
    )
    mock_create_venv = Mock()
    mock_install_deps = Mock()
    mock_run_tasks = Mock()
    mock_cluster_config = Mock()

    monkeypatch.setattr("llmb_install.core.installer.create_virtual_environment", mock_create_venv)
    monkeypatch.setattr("llmb_install.core.installer.install_dependencies", mock_install_deps)
    monkeypatch.setattr("llmb_install.core.installer.run_setup_tasks", mock_run_tasks)
    monkeypatch.setattr("llmb_install.core.installer.create_cluster_config", mock_cluster_config)

    installer = Installer()
    installer.root_dir = "/repo"

    install_path = str(tmp_path / "install")
    install_config = make_install_config(install_path, ["w1"])

    installer._perform_installation(install_config, workload)

    mock_create_venv.assert_not_called()
    mock_install_deps.assert_not_called()
    mock_run_tasks.assert_called_once_with(
        "w1",
        workload["w1"],
        None,
        "venv",
        install_path,
        {},
        "h100",
        "x86_64",
    )

    workload_venvs = mock_cluster_config.call_args.args[7]
    assert "w1" not in workload_venvs


@pytest.mark.parametrize("shell_value", [None, "from_shell"])
def test_environment_vars_are_applied_to_process_environment(monkeypatch, tmp_path, shell_value):
    """Playfile environment variables override shell values and reach cluster config."""
    setup_common_mocks(monkeypatch)
    monkeypatch.setattr(
        "llmb_install.core.installer.group_workloads_by_dependencies", Mock(return_value={None: ["w1"]})
    )
    mock_cluster_config = Mock()
    monkeypatch.setattr("llmb_install.core.installer.create_cluster_config", mock_cluster_config)

    # setenv first so monkeypatch records the original state and restores it on teardown
    # even when the installer itself later writes the variable into os.environ.
    monkeypatch.setenv("LLMB_TEST_PLAYFILE_VAR", shell_value or "")
    if shell_value is None:
        monkeypatch.delenv("LLMB_TEST_PLAYFILE_VAR")

    installer = Installer()
    installer.root_dir = "/repo"
    install_path = str(tmp_path / "install")
    install_config = make_install_config(
        install_path,
        ["w1"],
        environment_vars={"LLMB_TEST_PLAYFILE_VAR": "from_playfile"},
    )

    installer._perform_installation(install_config, {"w1": make_workload("w1")})

    assert os.environ["LLMB_TEST_PLAYFILE_VAR"] == "from_playfile"
    assert mock_cluster_config.call_args.args[4] == install_config.environment_vars


def test_no_dependency_workload_honors_venv_req(monkeypatch, tmp_path):
    """A no-dependency workload can still request an empty venv for setup tasks."""
    setup_common_mocks(monkeypatch)

    workload = {
        "w1": {
            "name": "w1",
            "path": "/repo/w1",
            "setup": {"venv_req": True, "tasks": [{"name": "prepare", "cmd": "echo prepare", "job_type": "local"}]},
        }
    }

    monkeypatch.setattr(
        "llmb_install.core.installer.group_workloads_by_dependencies", Mock(return_value={None: ["w1"]})
    )
    mock_create_venv = Mock()
    mock_run_tasks = Mock()
    mock_cluster_config = Mock()

    monkeypatch.setattr("llmb_install.core.installer.create_virtual_environment", mock_create_venv)
    monkeypatch.setattr("llmb_install.core.installer.run_setup_tasks", mock_run_tasks)
    monkeypatch.setattr("llmb_install.core.installer.create_cluster_config", mock_cluster_config)

    installer = Installer()
    installer.root_dir = "/repo"

    install_path = str(tmp_path / "install")
    install_config = make_install_config(install_path, ["w1"])
    expected_venv_path = f"{install_path}/venvs/w1_venv"

    installer._perform_installation(install_config, workload)

    mock_create_venv.assert_called_once_with(expected_venv_path, "venv")
    assert mock_run_tasks.call_args.args[2] == expected_venv_path

    workload_venvs = mock_cluster_config.call_args.args[7]
    assert workload_venvs["w1"] == expected_venv_path
