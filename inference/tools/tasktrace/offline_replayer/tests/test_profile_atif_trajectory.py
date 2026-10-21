# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from offline_replayer.adapters import atif_profile
from offline_replayer.adapters.atif_profile import (
    HIDE_PREPARED_PYTHON3_COMMAND,
    HOST_RUNTIME_EXECUTABLE,
    HOST_RUNTIME_LOADER,
    HOST_RUNTIME_MOUNT,
    REPLAY_PYTHON,
    RUST_CONTROLLER_DIGEST_ENV,
    RUST_CONTROLLER_ENV,
    RUST_CONTROLLER_EXECUTABLE,
    STAGE_REPLAY_PYTHON_COMMAND,
    AtifTrajectoryProfileConfig,
    _current_task_environment,
    build_container_profile_config,
    preflight_atif_trajectory_profile,
    run_atif_trajectory_profile,
)
from offline_replayer.measurement import current, pipeline
from offline_replayer.measurement.container import ProfileError
from offline_replayer.measurement.network_policy import NetworkMode, NetworkPolicy


def _trajectory(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "session_id": "session",
                "agent": {"name": "codex"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-05-24T01:00:00+00:00",
                        "tool_calls": [
                            {
                                "function_name": "exec_command",
                                "arguments": {"cmd": "true"},
                            }
                        ],
                        "observation": {"content": "Process exited with code 0"},
                    }
                ],
            }
        )
    )
    return path


def test_build_container_profile_wraps_atif_replay(tmp_path: Path) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=1.0,
        cpuset_cpus=None,
        allocated_cpus=1.0,
        pacing="recorded",
        timeout_sec=120.0,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        capture_action_outputs=True,
        network="none",
        network_policy=NetworkPolicy(NetworkMode.NO_NETWORK),
        network_policy_proxy_image="python@sha256:test-proxy",
        network_policy_external_network="egress-test",
        coordinated_baseline=True,
    )

    profile, trajectory = build_container_profile_config(config)

    assert trajectory.schema_version == "ATIF-v1.5"
    assert profile.cpu_limit == 1.0
    assert profile.allocated_cpus == 1.0
    assert profile.network == "none"
    assert profile.network_policy == NetworkPolicy(NetworkMode.NO_NETWORK)
    assert profile.network_policy_proxy_image == "python@sha256:test-proxy"
    assert profile.network_policy_external_network == "egress-test"
    assert profile.coordinated_baseline is True
    assert profile.terminate_descendants_at_finish is True
    assert "offline_replayer.adapters.atif_replay" in profile.command
    assert REPLAY_PYTHON in profile.command
    assert "--pacing recorded" in profile.command
    assert "--timeout-sec 120.0" in profile.command
    assert "--cwd /app" in profile.command
    assert "--patch-root /tmp" in profile.command
    assert "--action-output-dir /tasktrace/action_outputs" in profile.command
    assert 'TASKTRACE_WORKLOAD_ENV_PYTHONPATH="${PYTHONPATH-}"' in profile.command
    assert 'TASKTRACE_WORKLOAD_ENV_PRESENT_PYTHONPATH="${PYTHONPATH+1}"' in profile.command
    assert "TASKTRACE_CONTROLLER_ENV_VARS=PYTHONPATH" in profile.command
    assert any("dst=/tasktrace-input,readonly" in mount for mount in profile.mounts)
    source_mount = next(mount for mount in profile.mounts if "dst=/tasktrace-source" in mount)
    assert f"src={Path(__file__).resolve().parents[2]}" in source_mount
    assert profile.prepare_command == STAGE_REPLAY_PYTHON_COMMAND
    assert profile.workload_identity["task_name"] == trajectory.task_name
    assert profile.source_identity["format"] == "atif"
    assert profile.source_identity["sha256"] == hashlib.sha256(config.trajectory.read_bytes()).hexdigest()
    assert profile.expected_outcome["action_count"] == 1


def test_build_container_profile_passes_scheduler_configuration(tmp_path: Path) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="recorded",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        scheduler_concurrency=True,
        scheduler_backend="perf",
        scheduler_bcc_buffer_pages=4096,
        scheduler_perf_binary="/usr/bin/perf",
        scheduler_perf_mmap_pages=1024,
        memory_profile=True,
        memory_sample_interval_sec=0.05,
        memory_limit_bytes=1_073_741_824,
        memory_swap_total_limit_bytes=1_073_741_824,
        profile_sets=("cpu-compare",),
    )

    profile, trajectory_result = build_container_profile_config(config)

    assert profile.scheduler_concurrency is True
    assert profile.scheduler_backend == "perf"
    assert profile.scheduler_bcc_buffer_pages == 4096
    assert profile.scheduler_perf_binary == "/usr/bin/perf"
    assert profile.scheduler_perf_mmap_pages == 1024
    assert profile.memory_profile is True
    assert profile.memory_sample_interval_sec == 0.05
    assert profile.memory_limit_bytes == 1_073_741_824
    assert profile.memory_swap_total_limit_bytes == 1_073_741_824
    assert profile.profile_sets == ("cpu-compare",)


def test_atif_preflight_builds_exact_profile_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    trajectory = _trajectory(tmp_path / "trajectory.json")
    output = tmp_path / "output"
    config = AtifTrajectoryProfileConfig(
        trajectory=trajectory,
        image="registry.example/task@sha256:" + "a" * 64,
        output_dir=output,
        cpu_limit=None,
        cpuset_cpus="20-23",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=900.0,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        task_name="declared-task",
        profile_sets=("cpu-compare",),
    )
    captured = []

    def fake_preflight(profile):
        captured.append(profile)
        return {
            "schema_version": "tasktrace-profile-preflight.v1",
            "ready": True,
            "runtime": "docker",
            "runtime_identity": {"runtime": "docker"},
            "image_identity": {"id": "sha256:" + "b" * 64},
            "profile_sets": ["cpu-compare"],
            "cpuset_cpus": "20-23",
            "allocated_cpus": None,
            "effective_allocated_cpus": 4.0,
            "checks": [{"name": "profile_config", "status": "passed"}],
            "limitations": ["dynamic attach remains pending"],
            "configuration": {},
        }

    monkeypatch.setattr(atif_profile, "preflight_profile", fake_preflight)

    report = preflight_atif_trajectory_profile(config)

    assert report["ready"] is True
    assert report["source_sha256"] == hashlib.sha256(trajectory.read_bytes()).hexdigest()
    assert report["runtime"] == "docker"
    assert report["image_identity"]["id"] == "sha256:" + "b" * 64
    assert report["profile_sets"] == ["cpu-compare"]
    assert report["cpuset_cpus"] == "20-23"
    assert report["allocated_cpus"] is None
    assert report["task_name"] == "declared-task"
    assert report["checks"][0]["name"] == "atif_source"
    assert captured[0].source_identity["sha256"] == report["source_sha256"]
    assert captured[0].output_dir == output / "container_profile"
    assert not output.exists()


def test_build_container_profile_stages_named_controller_after_user_preparation(
    tmp_path: Path,
) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command="apt-get install -y python3",
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
    )

    profile, _trajectory_result = build_container_profile_config(config)

    assert profile.prepare_command == (f"(apt-get install -y python3) && {STAGE_REPLAY_PYTHON_COMMAND}")


def test_build_container_profile_can_hide_python_added_for_private_controller(
    tmp_path: Path,
) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="16-23",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command="apt-get install -y python3",
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        hide_prepared_python3=True,
    )

    profile, _trajectory_result = build_container_profile_config(config)

    assert profile.prepare_command == (
        f"(apt-get install -y python3) && {STAGE_REPLAY_PYTHON_COMMAND} && " f"{HIDE_PREPARED_PYTHON3_COMMAND}"
    )


def test_build_container_profile_can_mount_hidden_host_controller_runtime(
    tmp_path: Path,
) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="16-23",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        controller_runtime="host-mount",
    )

    profile, _trajectory_result = build_container_profile_config(config)

    assert profile.prepare_command is None
    assert HOST_RUNTIME_EXECUTABLE in profile.command
    assert "/tasktrace-controller-runtime/bin/python3" in profile.command
    assert HOST_RUNTIME_LOADER in profile.command
    assert "--library-path" in profile.command
    assert "-m offline_replayer.adapters.atif_replay" in profile.command
    assert 'TASKTRACE_WORKLOAD_ENV_PYTHONHOME="${PYTHONHOME-}"' in profile.command
    assert 'TASKTRACE_WORKLOAD_ENV_PRESENT_PYTHONHOME="${PYTHONHOME+1}"' in profile.command
    assert f"PYTHONHOME={HOST_RUNTIME_MOUNT}" in profile.command
    assert "TASKTRACE_CONTROLLER_ENV_VARS=PYTHONPATH:PYTHONHOME" in profile.command
    assert any(f"dst={HOST_RUNTIME_EXECUTABLE},readonly" in mount for mount in profile.mounts)
    assert any(f"dst={HOST_RUNTIME_LOADER},readonly" in mount for mount in profile.mounts)


def test_build_container_profile_mounts_content_bound_rust_controller(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    executable = tmp_path / "tasktrace"
    executable.write_bytes(b"native-tasktrace-test-binary")
    executable.chmod(0o755)
    monkeypatch.setenv(RUST_CONTROLLER_ENV, str(executable))
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="16-23",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=30.0,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        controller_runtime="rust-host-mount",
        capture_action_outputs=True,
    )

    profile, _trajectory_result = build_container_profile_config(config)

    digest = hashlib.sha256(executable.read_bytes()).hexdigest()
    assert profile.prepare_command is None
    assert f"{RUST_CONTROLLER_EXECUTABLE} replay atif" in profile.command
    assert "offline_replayer.adapters.atif_replay" not in profile.command
    assert REPLAY_PYTHON not in profile.command
    assert f"{RUST_CONTROLLER_DIGEST_ENV}={digest}" in profile.command
    assert f"TASKTRACE_CONTROLLER_ENV_VARS={RUST_CONTROLLER_DIGEST_ENV}" in profile.command
    assert "--action-output-dir /tasktrace/action_outputs" in profile.command
    assert "--timeout-sec 30.0" in profile.command
    assert any(
        f"src={executable.resolve()},dst={RUST_CONTROLLER_EXECUTABLE},readonly" in mount for mount in profile.mounts
    )
    assert not any("dst=/tasktrace-source" in mount for mount in profile.mounts)
    assert profile.source_identity["rust_controller_binary_sha256"] == digest


def test_rust_host_mount_fails_closed_without_native_dispatcher_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(RUST_CONTROLLER_ENV, raising=False)
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        controller_runtime="rust-host-mount",
    )

    with pytest.raises(ProfileError, match="native tasktrace dispatcher"):
        build_container_profile_config(config)


def test_rust_host_mount_can_hide_image_python_without_staging(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    executable = tmp_path / "tasktrace"
    executable.write_bytes(b"native-tasktrace-test-binary")
    executable.chmod(0o755)
    monkeypatch.setenv(RUST_CONTROLLER_ENV, str(executable))
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        controller_runtime="rust-host-mount",
        hide_prepared_python3=True,
    )

    profile, _trajectory_result = build_container_profile_config(config)

    assert profile.prepare_command == HIDE_PREPARED_PYTHON3_COMMAND
    assert STAGE_REPLAY_PYTHON_COMMAND not in profile.prepare_command


def test_host_mounted_controller_uses_one_runtime_when_launcher_python_differs(tmp_path: Path, monkeypatch) -> None:
    system_python = tmp_path / "python3.12"
    system_stdlib = tmp_path / "python3.12-lib"
    system_library_root = tmp_path / "usr-lib"
    system_library_dir = system_library_root / "x86_64-linux-gnu"
    system_loader = system_library_dir / "ld-linux-x86-64.so.2"
    system_python.touch()
    system_stdlib.mkdir()
    system_library_dir.mkdir(parents=True)
    system_loader.touch()

    monkeypatch.setattr(atif_profile, "HOST_RUNTIME_SOURCE_PYTHON", system_python)
    monkeypatch.setattr(atif_profile, "HOST_RUNTIME_LIBRARY_ROOT", system_library_root)
    monkeypatch.setattr(
        atif_profile.subprocess,
        "run",
        lambda *args, **kwargs: type(
            "Completed",
            (),
            {"stdout": json.dumps({"stdlib": str(system_stdlib), "multiarch": "x86_64-linux-gnu"})},
        )(),
    )

    runtime = atif_profile._host_controller_runtime()

    assert runtime.python == system_python.resolve()
    assert runtime.stdlib == system_stdlib.resolve()
    assert runtime.multiarch == "x86_64-linux-gnu"
    assert runtime.library_dir == system_library_dir
    assert runtime.loader == system_loader


def test_run_profile_preserves_atif_pipeline_report_contract(tmp_path: Path, monkeypatch) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="16-23",
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        task_name="instance_owner__repo-full-commit",
        controller_runtime="host-mount",
    )

    captured_cancel_requested = []

    def fake_run(_profile, *, cancel_requested=None):
        captured_cancel_requested.append(cancel_requested)
        replay = config.output_dir / "replay" / "replay_report.json"
        replay.write_text(json.dumps({"summary": {"source_outcome_mismatch_count": 0}}))
        return {
            "image_identity": {"id": "sha256:image"},
            "cpu": {"average_logical_cores_used": 1.0},
            "scheduler_concurrency": None,
            "proc_stat_comparison": None,
            "strace": {"mode": "none"},
            "warning": None,
            "status": "valid",
            "profile_provenance": {"mode": "baseline", "profile_sets": []},
        }

    monkeypatch.setattr(pipeline, "run_profile", fake_run)

    def cancel_requested():
        return False

    report = run_atif_trajectory_profile(config, cancel_requested=cancel_requested)

    assert report == {
        "schema_version": "container-atif-trajectory-profile.v1",
        "task_name": "instance_owner__repo-full-commit",
        "source_trial": config.trajectory.parent.parent.name,
        "source_trajectory": str(config.trajectory.resolve()),
        "source_trajectory_sha256": hashlib.sha256(config.trajectory.read_bytes()).hexdigest(),
        "source_trajectory_schema_version": "ATIF-v1.5",
        "source_agent_original_format": None,
        "agent": {"name": "codex"},
        "image": "alexgshaw/task:20251031",
        "image_identity": {"id": "sha256:image"},
        "network": None,
        "controller_runtime": "host-mount",
        "hide_prepared_python3": False,
        "patch_roots": ["/tmp"],
        "action_output_capture": {
            "enabled": False,
            "directory": None,
        },
        "pacing": "unpaced",
        "replay_report": str((config.output_dir / "replay" / "replay_report.json").resolve()),
        "container_profile_report": str((config.output_dir / "container_profile" / "profile_report.json").resolve()),
        "replay_summary": {"source_outcome_mismatch_count": 0},
        "container_cpu": {"average_logical_cores_used": 1.0},
        "scheduler_concurrency": None,
        "proc_stat_comparison": None,
        "strace": {"mode": "none"},
        "warning": None,
        "status": "valid",
        "profile_provenance": {"mode": "baseline", "profile_sets": []},
    }
    assert captured_cancel_requested == [cancel_requested]
    assert json.loads((config.output_dir / "pipeline_report.json").read_text()) == report


def test_host_mount_can_hide_image_python3_without_staging(tmp_path: Path) -> None:
    # host-mount runs the driver off-PATH, so hiding the image's python3 needs no
    # in-image staging -- the prepare step is just the removal command.
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image="alexgshaw/task:20251031",
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus="0-3",
        allocated_cpus=None,
        pacing="llm-gaps",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        controller_runtime="host-mount",
        hide_prepared_python3=True,
        runtime="enroot",
    )

    profile, _ = build_container_profile_config(config)

    assert profile.prepare_command == HIDE_PREPARED_PYTHON3_COMMAND
    assert STAGE_REPLAY_PYTHON_COMMAND not in (profile.prepare_command or "")
    assert profile.runtime == "enroot"


@pytest.mark.parametrize("pacing", ["recorded", "llm-gaps", "unpaced"])
def test_build_current_profile_runs_atif_directly_with_all_pacing_modes(tmp_path: Path, pacing: str) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image=None,
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus=None,
        allocated_cpus=None,
        pacing=pacing,
        timeout_sec=120.0,
        prepare_command="echo prepared",
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir=str(tmp_path),
        runtime="current",
        current_container_exclusive=True,
    )

    profile, trajectory = build_container_profile_config(config)

    assert trajectory.schema_version == "ATIF-v1.5"
    assert profile.runtime == "current"
    assert profile.image is None
    assert profile.mounts == ()
    assert profile.prepare_command == "echo prepared"
    assert profile.command.startswith(str(Path(sys.executable)))
    assert " -I -c " in profile.command
    assert "runpy.run_module" in profile.command
    assert str(Path(atif_profile.__file__).resolve().parents[2]) in profile.command
    assert str(config.trajectory.resolve()) in profile.command
    assert str((config.output_dir / "replay" / "replay_report.json").resolve()) in profile.command
    assert f"--pacing {pacing}" in profile.command
    assert "--timeout-sec 120.0" in profile.command


def test_current_task_environment_removes_only_controller_paths(tmp_path: Path) -> None:
    source = tmp_path / "source"
    virtualenv = source / ".venv"
    task_python = tmp_path / "task-python"
    task_bin = tmp_path / "task-bin"

    result = _current_task_environment(
        {
            "PATH": os.pathsep.join([str(virtualenv / "bin"), str(task_bin)]),
            "PYTHONPATH": os.pathsep.join([".", str(virtualenv / "deps"), str(task_python)]),
            "VIRTUAL_ENV": str(virtualenv),
            "VIRTUAL_ENV_PROMPT": "(tasktrace)",
            "UV_RUN_RECURSION_DEPTH": "1",
            "KEEP_VISIBLE": "yes",
        },
        controller_source=source,
        controller_executable=virtualenv / "bin" / "python3",
        controller_prefix=virtualenv,
        controller_base_prefix=tmp_path / "base-python",
        invocation_cwd=source,
    )

    assert result.assignments == (
        f"PATH={task_bin}",
        f"PYTHONPATH={task_python}",
    )
    assert result.unset_names == (
        "PYTHONPATH",
        "UV_RUN_RECURSION_DEPTH",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
    )


def test_current_task_environment_detects_unactivated_controller_virtualenv(
    tmp_path: Path,
) -> None:
    virtualenv = tmp_path / "controller-venv"

    result = _current_task_environment(
        {"PATH": f"{virtualenv / 'bin'}:/usr/bin:/bin"},
        controller_source=virtualenv / "lib/python/site-packages",
        controller_executable=virtualenv / "bin/python3",
        controller_prefix=virtualenv,
        controller_base_prefix=Path("/usr"),
        invocation_cwd=tmp_path,
    )

    assert result.assignments == ("PATH=/usr/bin:/bin",)
    assert result.unset_names == ()


def test_current_task_environment_always_removes_uv_controller_marker(
    tmp_path: Path,
) -> None:
    result = _current_task_environment(
        {
            "PATH": "/usr/bin:/bin",
            "UV_RUN_RECURSION_DEPTH": "1",
        },
        controller_source=tmp_path / "source",
        controller_executable=Path("/usr/bin/python3"),
        controller_prefix=Path("/usr"),
        controller_base_prefix=Path("/usr"),
        invocation_cwd=tmp_path,
    )

    assert result.assignments == ()
    assert result.unset_names == ("UV_RUN_RECURSION_DEPTH",)


def test_current_profile_commands_cannot_see_controller_virtualenv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    virtualenv = tmp_path / "controller-venv"
    controller_bin = virtualenv / "bin"
    controller_bin.mkdir(parents=True)
    task_bin = tmp_path / "task-bin"
    task_bin.mkdir()
    controller_python = controller_bin / "python3"
    controller_python.symlink_to(Path(sys._base_executable))
    controller_pip = controller_bin / "pip"
    controller_pip.write_text("#!/bin/sh\nexit 0\n")
    controller_pip.chmod(0o755)
    task_pip = task_bin / "pip"
    task_pip.write_text("#!/bin/sh\nexit 0\n")
    task_pip.chmod(0o755)
    monkeypatch.setenv("VIRTUAL_ENV", str(virtualenv))
    monkeypatch.setenv("VIRTUAL_ENV_PROMPT", "(controller)")
    monkeypatch.setenv("UV_RUN_RECURSION_DEPTH", "1")
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join([str(controller_bin), str(task_bin), "/usr/bin", "/bin"]),
    )
    monkeypatch.setenv(
        "PYTHONPATH",
        os.pathsep.join(
            [
                str(Path(atif_profile.__file__).resolve().parents[2]),
                str(tmp_path / "task-python"),
            ]
        ),
    )
    monkeypatch.setattr(atif_profile.sys, "prefix", str(virtualenv))
    monkeypatch.setattr(atif_profile.sys, "base_prefix", "/usr")
    monkeypatch.setattr(atif_profile.sys, "executable", str(controller_python))
    trajectory = _trajectory(tmp_path / "trajectory.json")
    payload = json.loads(trajectory.read_text())
    payload["steps"][0]["tool_calls"][0]["arguments"]["cmd"] = (
        "printf '%s\\n' "
        "\"VIRTUAL_ENV=${VIRTUAL_ENV-}\" "
        "\"PATH=$PATH\" "
        "\"PYTHONPATH=${PYTHONPATH-}\" "
        "\"PYTHON=$(command -v python3 || true)\" "
        "\"PIP=$(command -v pip || true)\" "
        "> environment.txt"
    )
    trajectory.write_text(json.dumps(payload))
    config = AtifTrajectoryProfileConfig(
        trajectory=trajectory,
        image=None,
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus=None,
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir=str(tmp_path),
        runtime="current",
        current_container_exclusive=True,
    )

    profile, _ = build_container_profile_config(config)
    environment = current._command_environment(profile.env, profile.unset_env)
    completed = subprocess.run(
        ["/bin/bash", "-c", profile.command],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    workload_environment = (tmp_path / "environment.txt").read_text()
    workload_values = dict(line.split("=", 1) for line in workload_environment.splitlines())
    assert "VIRTUAL_ENV=\n" in workload_environment
    assert str(controller_bin) not in workload_values["PATH"].split(os.pathsep)
    assert str(task_bin) in workload_values["PATH"].split(os.pathsep)
    assert f"PYTHONPATH={tmp_path / 'task-python'}" in workload_environment
    assert f"PYTHON={controller_python}" not in workload_environment
    assert f"PIP={task_pip}" in workload_environment


def test_current_source_checkout_import_survives_external_workdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    system_python = Path("/usr/bin/python3")
    if not system_python.is_file():
        pytest.skip("/usr/bin/python3 is unavailable")
    source = Path(atif_profile.__file__).resolve().parents[2]
    monkeypatch.setattr(atif_profile.sys, "executable", str(system_python))
    monkeypatch.setattr(atif_profile.sys, "prefix", "/usr")
    monkeypatch.setattr(atif_profile.sys, "base_prefix", "/usr")
    monkeypatch.delenv("VIRTUAL_ENV", raising=False)
    monkeypatch.setenv("PYTHONPATH", ".")
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image=None,
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus=None,
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir=str(tmp_path),
        runtime="current",
        current_container_exclusive=True,
    )

    profile, _ = build_container_profile_config(config)
    environment = current._command_environment(profile.env, profile.unset_env)
    completed = subprocess.run(
        ["/bin/bash", "-c", profile.command],
        cwd=tmp_path,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )

    assert source != tmp_path
    assert "PYTHONPATH" in profile.unset_env
    assert completed.returncode == 0, completed.stderr
    assert (config.output_dir / "replay" / "replay_report.json").is_file()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"image": "task:latest"}, "does not accept --image"),
        ({"cpu_limit": 1.0}, "cannot enforce --cpu-limit"),
        ({"cpuset_cpus": "0-3"}, "cannot enforce --cpuset-cpus"),
        ({"scheduler_concurrency": True}, "scheduler-concurrency"),
        ({"scheduler_bcc_buffer_pages": 4096}, "scheduler-bcc-buffer-pages"),
        ({"scheduler_attach_timeout_sec": 21.0}, "scheduler profiling options"),
        (
            {"scheduler_cpu_tolerance_fraction": 0.03},
            "scheduler profiling options",
        ),
        ({"scheduler_cpu_tolerance_sec": 0.006}, "scheduler profiling options"),
        ({"scheduler_backend": "perf"}, "scheduler profiling options"),
        ({"scheduler_perf_binary": "/custom/perf"}, "scheduler profiling options"),
        ({"scheduler_perf_mmap_pages": 1024}, "scheduler profiling options"),
        ({"proc_stat_comparison": True}, "proc-stat-comparison"),
        ({"proc_stat_interval_sec": 0.25}, "proc-stat sampling options"),
        ({"proc_stat_baseline_sec": 1.0}, "proc-stat sampling options"),
        ({"docker": "/custom/docker"}, "runtime executable overrides"),
        ({"enroot": "/custom/enroot"}, "runtime executable overrides"),
        ({"network": "none"}, "network"),
        (
            {"network_policy": NetworkPolicy(NetworkMode.PUBLIC)},
            "network policy controls",
        ),
        ({"memory_profile": True}, "memory profiling"),
        ({"memory_limit_bytes": 268_435_456}, "memory profiling"),
        ({"profile_sets": ("cpu-compare",)}, "--profile-set"),
        ({"coordinated_baseline": True}, "--coordinated-baseline"),
        ({"verification_command": "true"}, "--verification-command"),
        ({"controller_runtime": "host-mount"}, "separate controller runtime"),
        ({"hide_prepared_python3": True}, "cannot hide"),
        ({"allocated_cpus": 0.0}, "allocated CPU count"),
        ({"strace_mode": "invalid"}, "unsupported strace mode"),
        ({"string_limit": 0}, "string limit"),
    ],
)
def test_current_programmatic_config_rejects_unsupported_settings(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    values: dict[str, object] = {
        "trajectory": _trajectory(tmp_path / "trajectory.json"),
        "image": None,
        "output_dir": tmp_path / "output",
        "cpu_limit": None,
        "cpuset_cpus": None,
        "allocated_cpus": None,
        "pacing": "unpaced",
        "timeout_sec": None,
        "prepare_command": None,
        "strace_mode": "none",
        "string_limit": 4096,
        "shell": "/bin/bash",
        "workdir": str(tmp_path),
        "runtime": "current",
        "current_container_exclusive": True,
    }
    values.update(overrides)

    with pytest.raises(ProfileError, match=message):
        build_container_profile_config(AtifTrajectoryProfileConfig(**values))  # type: ignore[arg-type]


def test_current_profile_pipeline_records_current_runtime_metadata(tmp_path: Path, monkeypatch) -> None:
    config = AtifTrajectoryProfileConfig(
        trajectory=_trajectory(tmp_path / "trajectory.json"),
        image=None,
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus=None,
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir=str(tmp_path),
        runtime="current",
    )

    def fake_run(_profile):
        replay = config.output_dir / "replay" / "replay_report.json"
        replay.write_text(json.dumps({"summary": {"source_outcome_mismatch_count": 0}}))
        return {
            "runtime": "current",
            "scope": "current-container",
            "environment_identity": {"hostname": "sandbox"},
            "task_environment": {
                "unset_variables": ["VIRTUAL_ENV"],
                "overridden_variables": ["PATH"],
            },
            "measurement_validity": {"valid_for_average_cpu_reporting": True},
            "image_identity": None,
            "cpu": {"average_logical_cores_used": 1.0},
            "scheduler_concurrency": None,
            "proc_stat_comparison": None,
            "strace": {"mode": "none"},
            "warning": "exclusive sandbox required",
        }

    monkeypatch.setattr(pipeline, "run_profile", fake_run)

    report = run_atif_trajectory_profile(config)

    assert report["runtime"] == "current"
    assert report["measurement_scope"] == "current-container"
    assert report["environment_identity"] == {"hostname": "sandbox"}
    assert report["task_environment"] == {
        "unset_variables": ["VIRTUAL_ENV"],
        "overridden_variables": ["PATH"],
    }
    assert report["measurement_validity"]["valid_for_average_cpu_reporting"] is True
    assert report["image"] is None


def test_current_profile_captures_action_outputs(tmp_path: Path) -> None:
    trajectory = _trajectory(tmp_path / "trajectory.json")
    payload = json.loads(trajectory.read_text())
    payload["steps"][0]["tool_calls"][0]["arguments"]["cmd"] = "printf captured"
    trajectory.write_text(json.dumps(payload))
    config = AtifTrajectoryProfileConfig(
        trajectory=trajectory,
        image=None,
        output_dir=tmp_path / "output",
        cpu_limit=None,
        cpuset_cpus=None,
        allocated_cpus=None,
        pacing="unpaced",
        timeout_sec=None,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir=str(tmp_path),
        capture_action_outputs=True,
        runtime="current",
        current_container_exclusive=True,
    )

    profile, _ = build_container_profile_config(config)

    assert "--action-output-dir" in profile.command
    assert str((config.output_dir / "replay" / "action_outputs").resolve()) in profile.command
    (config.output_dir / "replay").mkdir(parents=True)
    completed = subprocess.run(
        ["/bin/bash", "-c", profile.command],
        cwd=tmp_path,
        env=current._command_environment(profile.env, profile.unset_env),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert (config.output_dir / "replay" / "action_outputs" / "action-0001.stdout.txt").read_text() == "captured"


@pytest.mark.parametrize("status, expected_exit", [("valid", 0), ("invalid", 1)])
def test_current_runtime_cli_needs_no_image_or_allocation(
    tmp_path: Path, monkeypatch, capsys, status: str, expected_exit: int
) -> None:
    trajectory = _trajectory(tmp_path / "trajectory.json")
    captured: list[AtifTrajectoryProfileConfig] = []

    def fake_profile(config: AtifTrajectoryProfileConfig):
        captured.append(config)
        return {
            "status": status,
            "task_name": "task",
            "source_trial": "trial",
            "replay_summary": {
                "actions_executed": 1,
                "actions_recorded": 1,
                "actions_timed_out": 0,
                "source_outcome_mismatch_count": 0,
            },
            "container_cpu": {
                "wall_sec": 1.0,
                "usage_sec": 1.0,
                "average_logical_cores_used": 1.0,
                "utilization_percent_of_allocation": 100.0,
                "allocated_logical_cpus": 1.0,
            },
            "scheduler_concurrency": None,
        }

    monkeypatch.setattr(atif_profile, "run_atif_trajectory_profile", fake_profile)

    assert (
        atif_profile.main(
            [
                str(trajectory),
                "--runtime",
                "current",
                "--pacing",
                "llm-gaps",
                "--workdir",
                str(tmp_path),
                "--current-container-exclusive",
                "--output-dir",
                str(tmp_path / "output"),
            ]
        )
        == expected_exit
    )

    assert captured[0].runtime == "current"
    assert captured[0].image is None
    assert captured[0].allocated_cpus is None
    assert captured[0].pacing == "llm-gaps"
    assert captured[0].current_container_exclusive is True
    assert captured[0].network_policy is None
    assert "Average logical cores used: 1.000000" in capsys.readouterr().out


def test_cli_forwards_internal_run_cancellation_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trajectory = _trajectory(tmp_path / "trajectory.json")
    cancellation = tmp_path / "cancel-requested"
    cancellation.write_text("cancel\n")
    observed: list[bool] = []

    def fake_profile(config: AtifTrajectoryProfileConfig, *, cancel_requested=None) -> dict[str, object]:
        assert cancel_requested is not None
        assert config.task_name == "manifest-task"
        observed.append(cancel_requested())
        return {
            "status": "valid",
            "task_name": "task",
            "source_trial": "trial",
            "replay_summary": {
                "actions_executed": 0,
                "actions_recorded": 0,
                "actions_timed_out": 0,
                "source_outcome_mismatch_count": 0,
            },
            "container_cpu": {
                "wall_sec": 1.0,
                "usage_sec": 0.0,
                "average_logical_cores_used": 0.0,
                "utilization_percent_of_allocation": 0.0,
                "allocated_logical_cpus": 1.0,
            },
            "scheduler_concurrency": None,
        }

    monkeypatch.setenv(atif_profile.RUN_CANCEL_FILE_ENV, str(cancellation))
    monkeypatch.setattr(atif_profile, "run_atif_trajectory_profile", fake_profile)

    assert (
        atif_profile.main(
            [
                str(trajectory),
                "--image",
                "task:latest",
                "--cpuset-cpus",
                "0",
                "--output-dir",
                str(tmp_path / "output"),
                "--task-name",
                "manifest-task",
            ]
        )
        == 0
    )
    assert observed == [True]


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--image", "task:latest"], "does not accept --image"),
        (["--cpuset-cpus", "0-3"], "cannot enforce --cpuset-cpus"),
        (["--cpu-limit", "1"], "cannot enforce --cpu-limit"),
        (["--scheduler-concurrency"], "not supported"),
        (["--scheduler-bcc-buffer-pages", "4096"], "not supported"),
        (["--scheduler-attach-timeout-sec", "21"], "not supported"),
        (["--scheduler-cpu-tolerance-fraction", "0.03"], "not supported"),
        (["--scheduler-cpu-tolerance-sec", "0.006"], "not supported"),
        (["--scheduler-backend", "perf"], "requires --scheduler-concurrency"),
        (["--scheduler-perf-binary", "/custom/perf"], "not supported"),
        (["--scheduler-perf-mmap-pages", "1024"], "not supported"),
        (["--proc-stat-interval-sec", "0.25"], "not supported"),
        (["--proc-stat-baseline-sec", "1"], "not supported"),
        (["--docker", "/custom/docker"], "not supported"),
        (["--enroot", "/custom/enroot"], "not supported"),
        (["--network", "none"], "not supported"),
        (["--network-policy", "public"], "not supported"),
        (["--memory-profile"], "not supported"),
        (["--memory-limit-bytes", "268435456"], "not supported"),
        (["--profile-set", "cpu-compare"], "not supported"),
        (["--coordinated-baseline"], "not supported"),
        (["--verification-command", "true"], "not supported"),
        (["--controller-runtime", "host-mount"], "not applicable"),
        (["--hide-prepared-python3"], "unsafe"),
    ],
)
def test_current_runtime_cli_rejects_nested_runtime_controls(
    tmp_path: Path, arguments: list[str], message: str, capsys
) -> None:
    trajectory = _trajectory(tmp_path / "trajectory.json")

    with pytest.raises(SystemExit) as raised:
        atif_profile.main(
            [
                str(trajectory),
                "--runtime",
                "current",
                "--workdir",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "output"),
                *arguments,
            ]
        )

    assert raised.value.code == 2
    assert message in capsys.readouterr().err
