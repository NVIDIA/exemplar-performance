# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import platform
import subprocess
from pathlib import Path

import pytest

import llmb_install.core.workload as workload_module
from llmb_install.cluster.slurm import augment_env_for_job_type
from llmb_install.core.workload import get_setup_tasks, run_setup_tasks
from llmb_install.environment.detector import normalize_architecture


@pytest.fixture()
def slurm_info():
    return {
        "slurm": {
            "account": "acct",
            "gpu_partition": "gpu-part",
            "cpu_partition": "cpu-part",
            "gpu_partition_gres": 8,
            "cpu_partition_gres": 4,
        }
    }


def _dummy_completed(stdout=""):
    return subprocess.CompletedProcess(args=[], returncode=0, stdout=stdout, stderr="")


def _create_workload_data(workdir: str, task_name: str, cmd: str, job_type: str, requires_gpus: bool = None):
    """Helper to create workload data structure."""
    task = {"name": task_name, "cmd": cmd, "job_type": job_type}
    if requires_gpus is not None:
        task["requires_gpus"] = requires_gpus

    return {"path": workdir, "setup": {"tasks": [task]}}


def _setup_mock_run(monkeypatch, stdout=""):
    """Helper to setup subprocess.run mock and return captured data."""
    captured = {}

    def fake_run(cmd, *args, **kwargs):
        captured["cmd"] = cmd
        captured["shell"] = kwargs.get("shell", False)
        captured["env"] = kwargs.get("env", {})
        return _dummy_completed(stdout)

    monkeypatch.setattr(subprocess, "run", fake_run)
    return captured


def _host_arch() -> str:
    """Return the canonical architecture of the current host."""
    return normalize_architecture(platform.machine())


def _run_task_test(
    tmp_path: Path,
    slurm_info,
    monkeypatch,
    task_name: str,
    cmd: str,
    job_type: str,
    requires_gpus: bool = None,
    stdout: str = "",
    node_architecture: str = "",
):
    """Helper to run a task test and return captured data.

    node_architecture defaults to the current host arch so existing tests never
    see a mismatch and their behaviour is unchanged.
    """
    if not node_architecture:
        node_architecture = _host_arch()

    wdir = tmp_path / f"dummy_{task_name}"
    wdir.mkdir()
    workload_key = f"tool_{task_name}"

    workload_data = _create_workload_data(str(wdir), task_name, cmd, job_type, requires_gpus)
    captured = _setup_mock_run(monkeypatch, stdout)

    run_setup_tasks(
        workload_key, workload_data, None, "venv", str(tmp_path / "install"), slurm_info, "h100", node_architecture
    )

    return captured


def test_setup_task_env_exports_architectures(tmp_path, slurm_info, monkeypatch):
    captured = _run_task_test(
        tmp_path, slurm_info, monkeypatch, "arch", "echo hi", "local", node_architecture="aarch64"
    )
    assert captured["env"]["LLMB_NODE_ARCHITECTURE"] == "aarch64"
    assert captured["env"]["LLMB_HOST_ARCHITECTURE"] == _host_arch()


def _make_fake_venv(tmp_path: Path, name: str = "fake_venv") -> Path:
    """Create a minimal venv directory with bin/python3 so get_venv_environment doesn't raise."""
    venv = tmp_path / name
    bin_dir = venv / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python3"
    python.touch()
    return venv


def _assert_slurm_env(env, expected_partition: str, expected_gres: str, job_type: str):
    """Helper to assert SLURM environment variables."""
    if job_type == "sbatch":
        assert env["SBATCH_ACCOUNT"] == "acct"
        assert env["SBATCH_PARTITION"] == expected_partition
        assert env["SBATCH_GPUS_PER_NODE"] == expected_gres
    elif job_type == "srun":
        assert env["SLURM_ACCOUNT"] == "acct"
        assert env["SLURM_PARTITION"] == expected_partition
        assert env["SLURM_GPUS_PER_NODE"] == expected_gres


# Original tests (kept for backward compatibility)
def test_local_task(tmp_path: Path, slurm_info, monkeypatch):
    captured = _run_task_test(tmp_path, slurm_info, monkeypatch, "local", "echo hello", "local")

    assert captured["shell"] is False  # Changed from shell=True to shlex.split()
    assert captured["cmd"] == ["echo", "hello"]  # Now a list due to shlex.split()
    assert captured["env"]["LLMB_WORKLOAD"].endswith("tool_local")


def test_srun_env_injection(tmp_path: Path, slurm_info, monkeypatch):
    captured = _run_task_test(tmp_path, slurm_info, monkeypatch, "srun", "echo srun", "srun")
    _assert_slurm_env(captured["env"], "cpu-part", "4", "srun")


def test_sbatch_job_submission(tmp_path: Path, slurm_info, monkeypatch):
    captured = _run_task_test(
        tmp_path, slurm_info, monkeypatch, "sbatch", "dummy_script.sh", "sbatch", stdout="Submitted batch job 12345\n"
    )

    assert captured["cmd"][0] == "sbatch"
    assert any(arg.startswith("--output=") and arg.endswith("%x-%j.out") for arg in captured["cmd"])
    assert "dummy_script.sh" in captured["cmd"]
    _assert_slurm_env(captured["env"], "cpu-part", "4", "sbatch")


def test_get_setup_tasks_empty():
    data = {"setup": {}}
    assert get_setup_tasks(data) == []


# Direct function tests
@pytest.mark.parametrize(
    "job_type,requires_gpus,expected_partition,expected_gres",
    [
        ("sbatch", False, "cpu-part", "4"),
        ("sbatch", True, "gpu-part", "8"),
        ("srun", False, "cpu-part", "4"),
        ("srun", True, "gpu-part", "8"),
    ],
)
def test_augment_env_for_job_type(slurm_info, job_type, requires_gpus, expected_partition, expected_gres):
    """Test augment_env_for_job_type directly with different configurations."""
    env = {}
    augment_env_for_job_type(env, job_type, slurm_info, requires_gpus)
    _assert_slurm_env(env, expected_partition, expected_gres, job_type)


def test_augment_env_overwrite(slurm_info):
    """Test that augment_env_for_job_type overwrites existing values."""
    env = {"SBATCH_ACCOUNT": "old"}
    augment_env_for_job_type(env, "sbatch", slurm_info)
    assert env["SBATCH_ACCOUNT"] == "acct"
    assert env["SBATCH_PARTITION"] == "cpu-part"


# GPU requirement tests
@pytest.mark.parametrize(
    "job_type,requires_gpus,expected_partition,expected_gres",
    [
        ("sbatch", True, "gpu-part", "8"),
        ("sbatch", False, "cpu-part", "4"),
        ("sbatch", None, "cpu-part", "4"),  # Test default behavior
        ("srun", True, "gpu-part", "8"),
    ],
)
def test_gpu_requirements(
    tmp_path: Path, slurm_info, monkeypatch, job_type, requires_gpus, expected_partition, expected_gres
):
    """Test that tasks with different GPU requirements use correct partitions."""
    task_name = f"{job_type}_{'gpu' if requires_gpus else 'cpu'}"
    stdout = "Submitted batch job 12345\n" if job_type == "sbatch" else ""

    captured = _run_task_test(
        tmp_path, slurm_info, monkeypatch, task_name, f"echo {task_name}", job_type, requires_gpus, stdout
    )

    _assert_slurm_env(captured["env"], expected_partition, expected_gres, job_type)


def test_local_task_ignores_gpu_requirement(tmp_path: Path, slurm_info, monkeypatch):
    """Test that local tasks ignore requires_gpus setting."""
    captured = _run_task_test(
        tmp_path, slurm_info, monkeypatch, "local_gpu", "echo local gpu", "local", requires_gpus=True
    )

    # Verify command was executed without shell (not sbatch) - now using shlex.split()
    assert captured["cmd"] == ["echo", "local", "gpu"]

    # Verify augment_env_for_job_type doesn't add SLURM vars for local tasks
    test_env = {}
    augment_env_for_job_type(test_env, "local", slurm_info, True)

    slurm_vars = [
        "SBATCH_ACCOUNT",
        "SBATCH_PARTITION",
        "SBATCH_GPUS_PER_NODE",
        "SLURM_ACCOUNT",
        "SLURM_PARTITION",
        "SLURM_GPUS_PER_NODE",
    ]
    for var in slurm_vars:
        assert var not in test_env


# ---------------------------------------------------------------------------
# Arch-mismatch venv stripping tests
# ---------------------------------------------------------------------------


def _make_mismatched_archs():
    """Return (host_arch, compute_arch) where the two are different."""
    host = _host_arch()
    other = "aarch64" if host == "x86_64" else "x86_64"
    return host, other


def test_sbatch_arch_mismatch_strips_venv(tmp_path: Path, slurm_info, monkeypatch):
    """sbatch tasks on a mismatched arch strip both the active installer venv and
    the workload venv from the subprocess environment.

    sbatch defaults to --export=ALL, so any VIRTUAL_ENV / venv bin already in
    os.environ (the installer's own active venv) would be forwarded to compute
    nodes verbatim.  get_clean_environment_for_subprocess() must remove them.
    """
    host_arch, compute_arch = _make_mismatched_archs()
    monkeypatch.setattr(workload_module, "get_host_architecture", lambda: host_arch)

    # Simulate the installer's own active venv in the current process environment —
    # this is what sbatch --export=ALL would propagate to the compute node.
    fake_installer_venv = tmp_path / "installer_venv"
    installer_bin = str(fake_installer_venv / "bin")
    monkeypatch.setenv("VIRTUAL_ENV", str(fake_installer_venv))
    monkeypatch.setenv("PATH", installer_bin + os.pathsep + os.environ.get("PATH", ""))

    # Workload venv — passed as venv_path but must also not appear in the result.
    fake_workload_venv = _make_fake_venv(tmp_path, "workload_venv")
    workload_bin = str(fake_workload_venv / "bin")

    wdir = tmp_path / "dummy_sbatch_mismatch"
    wdir.mkdir()
    workload_data = _create_workload_data(str(wdir), "sbatch_mismatch", "dummy_script.sh", "sbatch")
    captured = _setup_mock_run(monkeypatch, stdout="Submitted batch job 99\n")

    run_setup_tasks(
        "wl_sbatch_mismatch",
        workload_data,
        str(fake_workload_venv),
        "venv",
        str(tmp_path / "install"),
        slurm_info,
        "gb200",
        compute_arch,
    )

    env = captured["env"]
    # The installer's active venv must be stripped from os.environ before forwarding.
    assert "VIRTUAL_ENV" not in env, "Installer venv (VIRTUAL_ENV) must be stripped on arch mismatch"
    assert installer_bin not in env.get("PATH", "").split(
        os.pathsep
    ), "Installer venv bin must be removed from PATH on arch mismatch"
    # The workload venv must also not be activated.
    assert workload_bin not in env.get("PATH", "").split(
        os.pathsep
    ), "Workload venv bin must not be on PATH on arch mismatch"


def test_srun_arch_mismatch_keeps_venv(tmp_path: Path, slurm_info, monkeypatch):
    """srun tasks keep the workload venv even when architectures differ.

    The user is responsible for the srun invocation; stripping is out of scope
    for this change and deferred to the full multi-arch revamp.
    """
    host_arch, compute_arch = _make_mismatched_archs()
    monkeypatch.setattr(workload_module, "get_host_architecture", lambda: host_arch)

    fake_venv = _make_fake_venv(tmp_path, "fake_venv_srun")
    venv_bin = str(fake_venv / "bin")

    wdir = tmp_path / "dummy_srun_mismatch"
    wdir.mkdir()
    workload_data = _create_workload_data(str(wdir), "srun_mismatch", "echo srun cmd", "srun")
    captured = _setup_mock_run(monkeypatch)

    run_setup_tasks(
        "wl_srun_mismatch",
        workload_data,
        str(fake_venv),
        "venv",
        str(tmp_path / "install"),
        slurm_info,
        "gb200",
        compute_arch,
    )

    env = captured["env"]
    assert env.get("VIRTUAL_ENV") == str(fake_venv), "Workload venv should be activated for srun tasks"
    assert venv_bin in env.get("PATH", ""), "Venv bin dir should be on PATH for srun tasks"


def test_sbatch_arch_match_keeps_venv(tmp_path: Path, slurm_info, monkeypatch):
    """sbatch tasks with matching arch activate the workload venv as normal."""
    host_arch = _host_arch()
    monkeypatch.setattr(workload_module, "get_host_architecture", lambda: host_arch)

    fake_venv = _make_fake_venv(tmp_path, "fake_venv_sbatch_match")
    venv_bin = str(fake_venv / "bin")

    wdir = tmp_path / "dummy_sbatch_match"
    wdir.mkdir()
    workload_data = _create_workload_data(str(wdir), "sbatch_match", "dummy_script.sh", "sbatch")
    captured = _setup_mock_run(monkeypatch, stdout="Submitted batch job 42\n")

    run_setup_tasks(
        "wl_sbatch_match",
        workload_data,
        str(fake_venv),
        "venv",
        str(tmp_path / "install"),
        slurm_info,
        "h100",
        host_arch,  # compute arch == host arch → no mismatch
    )

    env = captured["env"]
    assert env.get("VIRTUAL_ENV") == str(fake_venv), "Workload venv should be activated when arches match"
    assert venv_bin in env.get("PATH", ""), "Venv bin dir should be on PATH when arches match"
