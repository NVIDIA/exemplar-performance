# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import time
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from offline_replayer.measurement import container as container_module
from offline_replayer.measurement import current as current_module
from offline_replayer.measurement.container import (
    ENROOT_PROCESS_CONTAINMENT,
    ProfileConfig,
    ProfileError,
    _cgroup_subtree_pids,
    _run_enroot_image,
    _signal_cgroup_descendants,
    _wait_for_cgroup_descendants_to_drain,
    build_enroot_image_command,
    build_existing_container_exec_command,
    build_image_command,
    build_report,
    cpu_summary_lines,
    parse_cpuset_cpu_count,
    parse_cpuset_cpu_ids,
    preflight_profile,
    run_profile,
)
from offline_replayer.measurement.enroot_network import (
    BACKEND,
    SCHEMA_VERSION,
    validate_evidence,
)
from offline_replayer.measurement.network_policy import NetworkPolicy


def _config(tmp_path: Path, **overrides: object) -> ProfileConfig:
    values: dict[str, object] = {
        "command": "make -j8",
        "prepare_command": None,
        "output_dir": tmp_path,
        "image": "swebench/task:latest",
        "container": None,
        "cpuset_cpus": "0-7",
        "cpu_limit": None,
        "allocated_cpus": None,
        "workdir": "/workdir",
        "user": None,
        "env": (),
        "mounts": (),
        "strace_mode": "execve",
        "string_limit": 4096,
        "shell": "/bin/bash",
    }
    values.update(overrides)
    return ProfileConfig(**values)  # type: ignore[arg-type]


def _write_enroot_timing_evidence(output_dir: Path, *, clock_source: str = "linux-proc-uptime") -> None:
    for filename, content in {
        "clock_source": clock_source,
        "clock_name": "CLOCK_BOOTTIME",
        "clock_resolution_ns": "10000000",
        "clock_boot_id": "boot-123",
        "clock_time_namespace": "time:[42]",
        "clock_time_namespace_boottime_offset": "0 0",
        "initial_effective_cpu_affinity": "0",
    }.items():
        (output_dir / filename).write_text(content + "\n")
    lifecycle = {
        "schema_version": "tasktrace-enroot-lifecycle.v1",
        "clock_reference": {
            "clock_id": "CLOCK_BOOTTIME",
            "clock_name": "CLOCK_BOOTTIME",
            "resolution_ns": 1,
        },
        "process_containment": ENROOT_PROCESS_CONTAINMENT,
        "container_name": "tasktrace-profile-test",
        "container_created": True,
        "window_start_ns": 900_000_000,
        "window_end_ns": 3_100_000_000,
        "phases": [
            {
                "phase": "start",
                "start_ns": 900_000_000,
                "end_ns": 3_100_000_000,
                "duration_sec": 2.2,
                "returncode": 0,
            }
        ],
        "cleanup": {"container_absent": True, "valid": True},
    }
    (output_dir / "enroot_lifecycle.json").write_text(json.dumps(lifecycle))


def test_parse_cpuset_cpu_count_supports_ranges_and_duplicates() -> None:
    assert parse_cpuset_cpu_count("0-3,6,8-9,3") == 7
    assert parse_cpuset_cpu_ids("3,1-2,3") == (1, 2, 3)


def test_parse_cpuset_cpu_count_rejects_invalid_range() -> None:
    with pytest.raises(ProfileError, match="invalid CPU set range"):
        parse_cpuset_cpu_count("4-2")


def test_profile_preflight_is_side_effect_free_secret_safe_and_honest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "new-output"
    image_metadata = [
        {
            "Id": "sha256:" + "a" * 64,
            "RepoDigests": ["registry.example/task@sha256:" + "b" * 64],
            "Created": "2026-07-01T00:00:00Z",
            "Architecture": "amd64",
            "Os": "linux",
            "Variant": None,
            "Size": 1234,
            "RootFS": {"Type": "layers", "Layers": ["sha256:" + "c" * 64]},
            "Config": {
                "Cmd": ["--api-token", "do-not-persist"],
                "Env": ["API_TOKEN=image-secret"],
            },
        }
    ]

    def fake_run(command, *, check=False):
        assert check is False
        if command[1] == "version":
            return subprocess.CompletedProcess(command, 0, "27.0.0\n", "")
        if command[1] == "info":
            return subprocess.CompletedProcess(command, 0, "2\n", "")
        assert command[1:3] == ["image", "inspect"]
        return subprocess.CompletedProcess(command, 0, json.dumps(image_metadata), "")

    monkeypatch.setattr(container_module, "_run", fake_run)
    monkeypatch.setattr(container_module.shutil, "which", lambda command: f"/usr/bin/{Path(command).name}")
    monkeypatch.setattr(container_module, "_online_cpu_ids", lambda: tuple(range(8)))
    monkeypatch.setattr(
        container_module,
        "_preflight_cgroup_v2",
        lambda **_kwargs: {
            "name": "cgroup_v2",
            "status": "passed",
            "required": True,
        },
    )
    monkeypatch.setattr(container_module.os, "geteuid", lambda: 0)
    monkeypatch.setattr(container_module, "_require_bcc_bindings", lambda: None)
    config = _config(
        output,
        cpuset_cpus="2-3",
        env=("API_TOKEN=host-secret",),
        strace_mode="execve",
        profile_sets=("cpu-compare",),
    )

    report = preflight_profile(config)

    assert report["ready"] is True
    assert report["runtime"] == "docker"
    assert report["image_identity"]["id"] == "sha256:" + "a" * 64
    assert "config" not in report["image_identity"]
    assert report["profile_sets"] == ["cpu-compare"]
    assert report["cpuset_cpus"] == "2-3"
    assert report["allocated_cpus"] is None
    checks = {item["name"]: item for item in report["checks"]}
    assert checks["scheduler_backend_static"]["status"] == "passed"
    assert checks["scheduler_dynamic_attach"]["status"] == "pending"
    assert checks["memory_cgroup_dynamic_scope"]["status"] == "pending"
    encoded = json.dumps(report, sort_keys=True)
    assert "host-secret" not in encoded
    assert "image-secret" not in encoded
    assert "do-not-persist" not in encoded
    assert not output.exists()


def test_docker_preflight_requires_an_immutable_image_id(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = _config(tmp_path / "output")
    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {"architecture": "amd64", "id": None},
    )

    with pytest.raises(ProfileError, match="immutable sha256 image ID"):
        container_module._docker_preflight_identity(config, "task:latest")


def test_enroot_file_preflight_records_content_identity_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "task.sqsh"
    image.write_bytes(b"immutable enroot image")
    output = tmp_path / "output"
    monkeypatch.setattr(container_module.shutil, "which", lambda _command: "/usr/bin/enroot")
    monkeypatch.setattr(
        container_module,
        "_run",
        lambda command, **_kwargs: subprocess.CompletedProcess(command, 0, "enroot version 3.5.0\n", ""),
    )
    monkeypatch.setattr(
        container_module,
        "_preflight_cgroup_v2",
        lambda **_kwargs: {
            "name": "cgroup_v2",
            "status": "passed",
            "required": False,
        },
    )
    config = _config(
        output,
        runtime="enroot",
        image=str(image),
        cpuset_cpus=None,
        allocated_cpus=1.0,
        strace_mode="none",
    )

    report = preflight_profile(config)

    assert report["ready"] is True
    assert report["image_identity"]["sha256"] == hashlib.sha256(image.read_bytes()).hexdigest()
    assert report["image_identity"]["immutable"] is True
    assert any("wall-time-only" in item for item in report["limitations"])
    assert not output.exists()


def test_profile_preflight_rejects_invalid_cpuset_without_creating_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(container_module, "_online_cpu_ids", lambda: (0, 1))

    with pytest.raises(ProfileError, match="unavailable logical CPUs"):
        preflight_profile(_config(output, cpuset_cpus="20-23"))

    assert not output.exists()


def test_current_preflight_remains_generic_and_reports_no_image_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "output"
    monkeypatch.setattr(current_module, "resolve_current_cgroup", lambda: object())
    config = _config(
        output,
        runtime="current",
        image=None,
        container=None,
        cpuset_cpus=None,
        allocated_cpus=1.0,
        strace_mode="none",
        workdir=str(tmp_path),
    )

    report = preflight_profile(config)

    assert report["ready"] is True
    assert report["runtime"] == "current"
    assert report["image_identity"] is None
    assert any("no independently inspectable" in item for item in report["limitations"])
    assert not output.exists()


def test_cgroup_subtree_pids_includes_nested_descendants(tmp_path: Path) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (tmp_path / "cgroup.procs").write_text("100\n")
    (nested / "cgroup.procs").write_text("101\n102\n")

    assert _cgroup_subtree_pids(tmp_path) == [100, 101, 102]


def test_signal_cgroup_descendants_revalidates_identity_and_excludes_root(tmp_path: Path, monkeypatch) -> None:
    sent: list[tuple[int, signal.Signals]] = []
    monkeypatch.setattr(container_module, "_cgroup_subtree_pids", lambda _path: [100, 101])
    monkeypatch.setattr(
        container_module,
        "_process_snapshot",
        lambda pid: {"pid": pid, "start_time_ticks": pid + 1000},
    )
    monkeypatch.setattr(container_module.os, "kill", lambda pid, sig: sent.append((pid, sig)))

    signaled = _signal_cgroup_descendants(
        tmp_path,
        root_tid=100,
        sig=signal.SIGTERM,
    )

    assert signaled == [101]
    assert sent == [(101, signal.SIGTERM)]


def test_wait_for_cgroup_descendants_allows_transient_exit(tmp_path: Path) -> None:
    procs = tmp_path / "cgroup.procs"
    procs.write_text("100\n101\n")

    class Tracer:
        def poll(self, *, timeout_ms: int) -> None:
            assert timeout_ms > 0
            procs.write_text("100\n")

    (
        remaining,
        waited_sec,
        initial_processes,
        remaining_processes,
    ) = _wait_for_cgroup_descendants_to_drain(
        tmp_path,
        root_tid=100,
        tracer=Tracer(),
        timeout_sec=0.1,
    )

    assert remaining == []
    assert waited_sec >= 0
    assert [process["pid"] for process in initial_processes] == [101]
    assert remaining_processes == []


def test_wait_for_cgroup_descendants_rejects_persistent_process(
    tmp_path: Path,
) -> None:
    (tmp_path / "cgroup.procs").write_text("100\n101\n")

    class Tracer:
        def poll(self, *, timeout_ms: int) -> None:
            raise AssertionError("zero grace should not poll")

    (
        remaining,
        _waited_sec,
        initial_processes,
        remaining_processes,
    ) = _wait_for_cgroup_descendants_to_drain(
        tmp_path,
        root_tid=100,
        tracer=Tracer(),
        timeout_sec=0,
    )

    assert remaining == [101]
    assert [process["pid"] for process in initial_processes] == [101]
    assert [process["pid"] for process in remaining_processes] == [101]


def test_build_image_command_configures_isolated_trace_run(tmp_path: Path) -> None:
    command = build_image_command(
        _config(
            tmp_path,
            env=("MODE=test",),
            mounts=("type=bind,src=/state,dst=/workdir",),
            prepare_command="apt-get install -y strace",
        )
    )

    assert command[:2] == ["docker", "run"]
    assert ["--cpuset-cpus", "0-7"] == command[command.index("--cpuset-cpus") : command.index("--cpuset-cpus") + 2]
    assert "SYS_PTRACE" in command
    assert "seccomp=unconfined" in command
    assert "TASKTRACE_PROFILE_COMMAND=make -j8" in command
    assert "TASKTRACE_PROFILE_SHELL=/bin/bash" in command
    assert "TASKTRACE_PROFILE_PREPARE_COMMAND=apt-get install -y strace" in command
    assert "type=bind,src=/state,dst=/workdir" in command
    assert command[-3:] == ["/bin/sh", "swebench/task:latest", "/tasktrace-profile-entrypoint.sh"]


def test_build_enroot_image_command_configures_image_run(tmp_path: Path) -> None:
    command = build_enroot_image_command(
        _config(
            tmp_path,
            runtime="enroot",
            env=("MODE=test",),
            mounts=("type=bind,src=/state,dst=/workdir,readonly",),
            strace_mode="none",
        )
    )

    assert command[:8] == [
        "unshare",
        "--user",
        "--map-current-user",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount-proc",
        "--",
    ]
    assert command.index("taskset") < command.index("enroot")
    enroot = command.index("enroot")
    # the host home hook is disabled on the enroot process itself, never via --env
    assert command[enroot - 2 : enroot + 2] == ["env", "ENROOT_MOUNT_HOME=no", "enroot", "start"]
    assert not any(
        a == "--env" and b.startswith("ENROOT_MOUNT_HOME") for a, b in zip(command, command[1:], strict=False)
    )
    mount = "/state:/workdir:none:x-create=auto,bind,ro"
    assert ["--mount", mount] == command[command.index(mount) - 1 : command.index(mount) + 1]
    assert "TASKTRACE_PROFILE_COMMAND=make -j8" in command
    assert "TASKTRACE_PROFILE_SHELL=/bin/bash" in command
    assert command[-5:] == [
        "--",
        "swebench/task:latest",
        "/bin/sh",
        "-c",
        "cd /workdir && exec /bin/sh /tasktrace-profile-entrypoint.sh",
    ]


def test_enroot_command_requests_cpu_stat_skip(tmp_path: Path) -> None:
    command = build_enroot_image_command(_config(tmp_path, runtime="enroot", strace_mode="none"))

    assert "TASKTRACE_PROFILE_SKIP_CPU_STATS=1" in command


def test_enroot_no_network_wraps_start_with_evidence_helper(tmp_path: Path) -> None:
    evidence = tmp_path / "private" / "evidence.json"

    command = build_enroot_image_command(
        _config(
            tmp_path,
            runtime="enroot",
            network_policy=NetworkPolicy("no-network"),
            strace_mode="none",
        ),
        network_evidence_path=evidence,
    )

    assert command[:8] == [
        "unshare",
        "--user",
        "--map-current-user",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount-proc",
        "--net",
    ]
    assert any(part.endswith("/enroot_network.py") for part in command)
    assert command[command.index("--evidence-path") + 1] == str(evidence.resolve())
    assert command.index("taskset") < command.index("enroot")


def test_enroot_network_evidence_validation_is_fail_closed() -> None:
    uid, gid = os.geteuid(), os.getegid()
    payload = {
        "schema_version": SCHEMA_VERSION,
        "state": "enforced",
        "backend": BACKEND,
        "policy_effective_mode": "no-network",
        "valid": True,
        "parent_netns_inode": 101,
        "netns_inode": 202,
        "netns_differs": True,
        "controller_euid": uid,
        "controller_egid": gid,
        "namespace_euid": uid,
        "namespace_egid": gid,
        "interfaces": ["lo"],
        "ipv4_routes": [],
        "ipv6_addresses": [],
    }

    assert (
        validate_evidence(
            payload,
            parent_netns_inode=101,
            controller_euid=uid,
            controller_egid=gid,
        )
        == payload
    )
    payload["interfaces"] = ["eth0"]
    with pytest.raises(RuntimeError, match="evidence is invalid"):
        validate_evidence(
            payload,
            parent_netns_inode=101,
            controller_euid=uid,
            controller_egid=gid,
        )


def test_run_enroot_image_creates_extracts_and_removes_container(tmp_path: Path, monkeypatch) -> None:
    image = tmp_path / "task.sqsh"
    image.write_text("squashfs")
    calls: list[list[str]] = []

    def fake_run(command, check: bool = False):
        calls.append(list(command))
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(container_module, "_run", fake_run)

    _run_enroot_image(_config(tmp_path, runtime="enroot", image=str(image), strace_mode="none"))

    # A .sqsh path is extracted to a uniquely named container first (no squashfuse),
    # the disposable run uses that name, and the container is removed afterwards.
    assert calls[0][:3] == ["enroot", "create", "--name"]
    container_name = calls[0][3]
    assert calls[0][4] == str(image)
    start = next(c for c in calls if "start" in c)
    assert container_name in start and str(image) not in start
    remove = next(c for c in calls if c[1:3] == ["remove", "-f"])
    assert remove[3] == container_name
    assert calls[-1] == ["enroot", "list"]
    lifecycle = json.loads((tmp_path / "enroot_lifecycle.json").read_text())
    assert [phase["phase"] for phase in lifecycle["phases"]] == [
        "create",
        "start",
        "remove",
        "list",
    ]
    assert lifecycle["clock_reference"]["clock_name"] == "CLOCK_BOOTTIME"
    assert lifecycle["cleanup"] == {
        "container_absent": True,
        "errors": [],
        "list_returncode": 0,
        "remove_returncode": 0,
        "valid": True,
    }


def test_run_enroot_image_cleans_partial_container_after_create_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "task.sqsh"
    image.write_text("squashfs")
    calls: list[list[str]] = []

    def fake_run(command, check: bool = False):
        calls.append(list(command))
        returncode = 1 if command[1:2] == ["create"] else 0
        stderr = "create failed" if returncode else ""
        return subprocess.CompletedProcess(list(command), returncode, "", stderr)

    monkeypatch.setattr(container_module, "_run", fake_run)

    with pytest.raises(ProfileError, match="enroot create failed"):
        _run_enroot_image(
            _config(
                tmp_path,
                runtime="enroot",
                image=str(image),
                strace_mode="none",
            )
        )

    container_name = calls[0][3]
    assert ["enroot", "remove", "-f", container_name] in calls
    assert ["enroot", "list"] in calls
    lifecycle = json.loads((tmp_path / "enroot_lifecycle.json").read_text())
    assert lifecycle["cleanup"]["container_absent"] is True


def test_run_enroot_image_cleans_partial_container_when_create_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "task.sqsh"
    image.write_text("squashfs")
    calls: list[list[str]] = []

    def fake_run(command, check: bool = False):
        calls.append(list(command))
        if command[1:2] == ["create"]:
            raise OSError("create interrupted")
        return subprocess.CompletedProcess(list(command), 0, "", "")

    monkeypatch.setattr(container_module, "_run", fake_run)

    with pytest.raises(OSError, match="create interrupted"):
        _run_enroot_image(
            _config(
                tmp_path,
                runtime="enroot",
                image=str(image),
                strace_mode="none",
            )
        )

    container_name = calls[0][3]
    assert calls[1:] == [
        ["enroot", "remove", "-f", container_name],
        ["enroot", "list"],
    ]
    lifecycle = json.loads((tmp_path / "enroot_lifecycle.json").read_text())
    assert lifecycle["phases"][0]["phase"] == "create"
    assert lifecycle["phases"][0]["returncode"] is None
    assert lifecycle["cleanup"]["valid"] is True


def test_run_enroot_image_fails_closed_on_cleanup_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "task.sqsh"
    image.write_text("squashfs")

    def fake_run(command, check: bool = False):
        returncode = 1 if command[1:3] == ["remove", "-f"] else 0
        return subprocess.CompletedProcess(list(command), returncode, "", "busy")

    monkeypatch.setattr(container_module, "_run", fake_run)

    with pytest.raises(ProfileError, match="enroot remove failed"):
        _run_enroot_image(_config(tmp_path, runtime="enroot", image=str(image), strace_mode="none"))

    lifecycle = json.loads((tmp_path / "enroot_lifecycle.json").read_text())
    assert lifecycle["cleanup"]["valid"] is False
    assert lifecycle["cleanup"]["remove_returncode"] == 1


def test_run_enroot_image_fails_when_exact_container_name_survives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    image = tmp_path / "task.sqsh"
    image.write_text("squashfs")
    container_name: str | None = None

    def fake_run(command, check: bool = False):
        nonlocal container_name
        if command[1:3] == ["create", "--name"]:
            container_name = command[3]
        stdout = f"other\n{container_name}\n" if command[1:] == ["list"] else ""
        return subprocess.CompletedProcess(list(command), 0, stdout, "")

    monkeypatch.setattr(container_module, "_run", fake_run)

    with pytest.raises(ProfileError, match="survived removal"):
        _run_enroot_image(_config(tmp_path, runtime="enroot", image=str(image), strace_mode="none"))


def test_cpu_summary_lines_handles_wall_only_enroot_report() -> None:
    lines = cpu_summary_lines(
        {
            "wall_sec": 2.0,
            "usage_sec": None,
            "average_logical_cores_used": None,
            "utilization_percent_of_allocation": None,
            "allocated_logical_cpus": 1.0,
        }
    )

    assert lines == [
        "Wall time: 2.000000000 sec",
        "CPU time: n/a (cgroup cpu.stat unavailable)",
        "Average logical cores used: n/a (cgroup cpu.stat unavailable)",
    ]


def test_build_report_reports_wall_only_when_cpu_stats_skipped(tmp_path: Path) -> None:
    (tmp_path / "start_ns").write_text("1000000000\n")
    (tmp_path / "end_ns").write_text("3000000000\n")
    (tmp_path / "exit_code").write_text("0\n")
    _write_enroot_timing_evidence(tmp_path)

    report = build_report(
        _config(tmp_path, runtime="enroot", image="task.sqsh", cpuset_cpus="0", strace_mode="none"),
        scope="disposable-image-container",
        docker_returncode=0,
    )

    assert report["cpu"]["method"] == "wall-clock-only-no-cgroup"
    assert report["cpu"]["wall_clock_source"] == "linux-proc-uptime"
    assert report["cpu"]["wall_sec"] == 2.0
    assert report["cpu"]["usage_sec"] is None
    assert report["cpu"]["average_logical_cores_used"] is None
    assert report["cpu"]["allocated_logical_cpus"] == 1.0
    assert report["cpu"]["allocated_cpu_source"] == "taskset-requested-affinity"
    assert report["cpu"]["cpu_affinity_enforced"] is False
    assert report["cpu"]["initial_effective_cpu_affinity"] == "0"
    assert report["cpu"]["initial_effective_logical_cpu_ids"] == [0]
    assert report["cpu"]["window_start_ns"] == 1_000_000_000
    assert report["cpu"]["window_end_ns"] == 3_000_000_000
    assert report["cpu"]["clock_reference"] == {
        "accuracy_claim": None,
        "boot_id": "boot-123",
        "clock_id": "enroot-task-boottime",
        "clock_name": "CLOCK_BOOTTIME",
        "resolution_ns": 10_000_000,
        "source": "linux-proc-uptime",
        "storage_precision_ns": 1,
        "time_namespace": "time:[42]",
        "time_namespace_boottime_offset_ns": 0,
        "timestamp_unit": "nanoseconds",
    }
    assert report["enroot_lifecycle"]["cleanup"]["container_absent"] is True


@pytest.mark.parametrize("clock_source", [None, "clock-realtime"])
def test_build_report_rejects_missing_or_nonmonotonic_enroot_clock(
    tmp_path: Path,
    clock_source: str | None,
) -> None:
    (tmp_path / "start_ns").write_text("1000000000\n")
    (tmp_path / "end_ns").write_text("3000000000\n")
    (tmp_path / "exit_code").write_text("0\n")
    if clock_source is not None:
        _write_enroot_timing_evidence(tmp_path, clock_source=clock_source)

    with pytest.raises(ProfileError, match="clock_source|wall clock source"):
        build_report(
            _config(
                tmp_path,
                runtime="enroot",
                image="task.sqsh",
                cpuset_cpus="0",
                strace_mode="none",
            ),
            scope="disposable-image-container",
            docker_returncode=0,
        )


def test_build_existing_container_command_does_not_mutate_cpu_limits(
    tmp_path: Path,
) -> None:
    config = _config(
        tmp_path,
        image=None,
        container="task-running",
        cpuset_cpus=None,
        allocated_cpus=8.0,
    )

    command = build_existing_container_exec_command(config, "/tmp/profile", "/tmp/profile/entrypoint.sh")

    assert command[:2] == ["docker", "exec"]
    assert "--cpuset-cpus" not in command
    assert command[-3:] == ["task-running", "/bin/sh", "/tmp/profile/entrypoint.sh"]


def test_untraced_image_baseline_does_not_enable_ptrace_permissions(
    tmp_path: Path,
) -> None:
    command = build_image_command(_config(tmp_path, strace_mode="none"))

    assert "SYS_PTRACE" not in command
    assert "seccomp=unconfined" not in command


def test_image_run_can_apply_docker_cpu_quota_without_changing_cpuset(
    tmp_path: Path,
) -> None:
    command = build_image_command(_config(tmp_path, cpuset_cpus=None, allocated_cpus=2.0, cpu_limit=2.0))

    assert ["--cpus", "2.0"] == command[command.index("--cpus") : command.index("--cpus") + 2]
    assert "--cpuset-cpus" not in command


def test_image_run_can_disable_container_networking(tmp_path: Path) -> None:
    command = build_image_command(_config(tmp_path, network="none"))

    assert ["--network", "none"] == command[command.index("--network") : command.index("--network") + 2]


@pytest.mark.parametrize(
    ("policy", "expected_network"),
    [
        (NetworkPolicy("public"), None),
        (NetworkPolicy("no-network"), "none"),
        (NetworkPolicy("allowlist"), "none"),
    ],
)
def test_typed_network_policy_builds_native_docker_network_modes(
    tmp_path: Path,
    policy: NetworkPolicy,
    expected_network: str | None,
) -> None:
    command = build_image_command(_config(tmp_path, network_policy=policy, strace_mode="none"))

    if expected_network is None:
        assert "--network" not in command
    else:
        index = command.index("--network")
        assert command[index : index + 2] == ["--network", expected_network]
    assert "NET_RAW" not in command


def test_prepared_allowlist_builds_internal_attachment_and_proxy_environment(
    tmp_path: Path,
) -> None:
    policy = NetworkPolicy("allowlist", ("pypi.org",))
    proxy_url = "http://tasktrace-egress-proxy:3128"
    enforcement = {
        "state": "active",
        "backend": "docker-forward-proxy-v1",
        "docker_network_override": "tasktrace-policy-test",
        "policy": policy.to_report_payload(),
        "proxy": {"url": proxy_url},
        "external_network_identity": {"configuration_sha256": "d" * 64},
        "validation": {
            "external_network_inspected": True,
            "internal_network_inspected": True,
            "proxy_container_inspected": True,
            "proxy_ready": True,
            "proxy_bound_to_internal_address": True,
            "state_disjoint_from_workload_mounts": True,
            "proxy_healthy_at_cleanup": None,
        },
    }
    command = build_image_command(
        _config(
            tmp_path,
            network_policy=policy,
            network_attachment="tasktrace-policy-test",
            network_policy_enforcement=enforcement,
            env=(
                f"HTTP_PROXY={proxy_url}",
                f"HTTPS_PROXY={proxy_url}",
                f"http_proxy={proxy_url}",
                f"https_proxy={proxy_url}",
            ),
            strace_mode="none",
        )
    )

    network_index = command.index("--network")
    assert command[network_index : network_index + 2] == [
        "--network",
        "tasktrace-policy-test",
    ]
    cap_index = command.index("--cap-drop")
    assert command[cap_index : cap_index + 2] == ["--cap-drop", "NET_RAW"]
    assert "HTTPS_PROXY=http://tasktrace-egress-proxy:3128" in command
    assert "https_proxy=http://tasktrace-egress-proxy:3128" in command


def test_unprepared_allowlist_fails_closed_before_docker_command(
    tmp_path: Path,
) -> None:
    with pytest.raises(ProfileError, match="not prepared"):
        build_image_command(
            _config(
                tmp_path,
                network_policy=NetworkPolicy("allowlist", ("pypi.org",)),
                strace_mode="none",
            )
        )


@pytest.mark.parametrize(
    "enforcement",
    [
        None,
        {"state": "closed"},
        {
            "state": "active",
            "docker_network_override": "different-network",
            "validation": {
                "external_network_inspected": True,
                "internal_network_inspected": True,
                "proxy_container_inspected": True,
                "proxy_ready": True,
            },
        },
    ],
)
def test_allowlist_attachment_without_active_validated_enforcement_fails_closed(
    tmp_path: Path, enforcement: dict[str, object] | None
) -> None:
    with pytest.raises(ProfileError, match="not prepared"):
        build_image_command(
            _config(
                tmp_path,
                network_policy=NetworkPolicy("allowlist", ("pypi.org",)),
                network_attachment="tasktrace-policy-test",
                network_policy_enforcement=enforcement,
                strace_mode="none",
            )
        )


@pytest.mark.parametrize(
    "source",
    [
        "/var/run/docker.sock",
        "/run/docker.sock",
        "/var/run/../run/docker.sock",
        "/run/user/1000/docker.sock",
        "/tmp/podman.sock",
        "/var/run",
    ],
)
def test_allowlist_rejects_runtime_socket_or_parent_mounts(tmp_path: Path, source: str) -> None:
    config = _config(
        tmp_path,
        network_policy=NetworkPolicy("allowlist", ("pypi.org",)),
        mounts=(f"type=bind,src={source},dst=/runtime",),
        strace_mode="none",
    )

    with pytest.raises(ProfileError, match="container-runtime socket"):
        container_module.validate_profile_config(config)


def test_current_runtime_rejects_explicit_public_network_policy(
    tmp_path: Path,
) -> None:
    inherited_network = _config(
        tmp_path / "inherited",
        runtime="current",
        image=None,
        cpuset_cpus=None,
        strace_mode="none",
        workdir=str(tmp_path),
    )
    container_module.validate_profile_config(inherited_network)

    with pytest.raises(ProfileError, match="network policy controls"):
        container_module.validate_profile_config(
            replace(
                inherited_network,
                output_dir=tmp_path / "explicit",
                network_policy=NetworkPolicy("public"),
            )
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {"scheduler_attach_timeout_sec": 21.0},
        {"scheduler_cpu_tolerance_fraction": 0.03},
        {"scheduler_cpu_tolerance_sec": 0.006},
        {"scheduler_backend": "perf"},
        {"scheduler_bcc_buffer_pages": 4096},
        {"scheduler_perf_binary": "/custom/perf"},
        {"scheduler_perf_mmap_pages": 1024},
        {"proc_stat_interval_sec": 0.25},
        {"proc_stat_baseline_sec": 1.0},
    ],
)
def test_current_runtime_rejects_unused_profiler_tuning(tmp_path: Path, overrides: dict[str, object]) -> None:
    config = _config(
        tmp_path,
        runtime="current",
        image=None,
        cpuset_cpus=None,
        strace_mode="none",
        **overrides,
    )

    with pytest.raises(ProfileError, match="profiling options|sampling options"):
        container_module.validate_profile_config(config)


def _allowlist_enforcement(
    policy: NetworkPolicy, *, ephemeral_name: str, allowed_ports: tuple[int, ...]
) -> dict[str, object]:
    comparable = {
        "schema_version": "tasktrace-docker-network-enforcement-comparability.v1",
        "policy": policy.to_comparability_payload(),
        "backend": "docker-forward-proxy-v1",
        "allowed_ports": list(allowed_ports),
        "external_network": "bridge",
        "external_network_identity": {"configuration_sha256": "d" * 64},
        "proxy_source_sha256": "a" * 64,
        "proxy_image_identity": {
            "id": "sha256:" + "b" * 64,
            "repo_digests": ["python@sha256:" + "c" * 64],
        },
    }
    return {
        "state": "closed",
        "backend": "docker-forward-proxy-v1",
        "allowed_ports": list(allowed_ports),
        "ephemeral_internal_network": ephemeral_name,
        "proxy": {
            "container_name": ephemeral_name + "-proxy",
            "source_sha256": "a" * 64,
        },
        "comparability": comparable,
    }


def test_policy_report_retains_runtime_evidence_but_comparability_excludes_ephemera(
    tmp_path: Path,
) -> None:
    policy = NetworkPolicy("allowlist", ("pypi.org",))
    first_enforcement = _allowlist_enforcement(
        policy,
        ephemeral_name="tasktrace-policy-first",
        allowed_ports=(80, 443),
    )
    second_enforcement = _allowlist_enforcement(
        policy,
        ephemeral_name="tasktrace-policy-second",
        allowed_ports=(80, 443),
    )
    different_ports = _allowlist_enforcement(
        policy,
        ephemeral_name="tasktrace-policy-third",
        allowed_ports=(443,),
    )
    first = _config(
        tmp_path / "first",
        network_policy=policy,
        network_policy_enforcement=first_enforcement,
        strace_mode="none",
    )
    second = replace(
        first,
        output_dir=tmp_path / "second",
        network_policy_enforcement=second_enforcement,
    )
    third = replace(
        first,
        output_dir=tmp_path / "third",
        network_policy_enforcement=different_ports,
    )

    policy_report = container_module._network_policy_report(first)
    comparable = container_module._network_policy_comparability(first)
    assert policy_report["requested_mode"] == "allowlist"
    assert policy_report["effective_mode"] == "allowlist"
    assert policy_report["allowed_hosts"] == ["pypi.org"]
    assert policy_report["enforcement"]["ephemeral_internal_network"] == ("tasktrace-policy-first")
    assert comparable is not None
    assert comparable["enforcement"]["allowed_ports"] == [80, 443]
    assert "tasktrace-policy-first" not in json.dumps(comparable)
    assert container_module._legacy_network_projection(first) == "tasktrace-allowlist"

    identity = {"id": "sha256:" + "d" * 64}
    first_provenance = container_module._memory_execution_provenance(first, identity)
    second_provenance = container_module._memory_execution_provenance(second, identity)
    third_provenance = container_module._memory_execution_provenance(third, identity)
    assert first_provenance["comparability_sha256"] == second_provenance["comparability_sha256"]
    assert first_provenance["comparability_sha256"] != third_provenance["comparability_sha256"]


def test_run_profile_wires_allowlist_session_proxy_environment_and_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from offline_replayer.measurement import network_enforcement

    output = tmp_path / "profile"
    policy = NetworkPolicy("allowlist", ("pypi.org",))
    captured_commands: list[list[str]] = []

    class FakeNetworkSession:
        instances: list["FakeNetworkSession"] = []

        def __init__(
            self,
            received_policy: NetworkPolicy,
            output_dir: Path,
            **kwargs: object,
        ) -> None:
            self.policy = received_policy
            self.output_dir = output_dir
            self.kwargs = kwargs
            self.docker_network = "tasktrace-policy-fixture"
            self.proxy_environment = {
                "HTTP_PROXY": "http://tasktrace-egress-proxy:3128",
                "HTTPS_PROXY": "http://tasktrace-egress-proxy:3128",
                "http_proxy": "http://tasktrace-egress-proxy:3128",
                "https_proxy": "http://tasktrace-egress-proxy:3128",
            }
            self.closed = False
            type(self).instances.append(self)

        def __enter__(self) -> "FakeNetworkSession":
            return self

        def close(self) -> list[str]:
            self.closed = True
            return []

        def report_payload(self) -> dict[str, object]:
            proxy_url = "http://tasktrace-egress-proxy:3128"
            return {
                "state": "closed" if self.closed else "active",
                "backend": "docker-forward-proxy-v1",
                "docker_network_override": self.docker_network,
                "policy": self.policy.to_report_payload(),
                "proxy": {"url": proxy_url},
                "external_network_identity": {"configuration_sha256": "d" * 64},
                "validation": {
                    "external_network_inspected": True,
                    "internal_network_inspected": True,
                    "proxy_container_inspected": True,
                    "proxy_ready": True,
                    "proxy_bound_to_internal_address": True,
                    "state_disjoint_from_workload_mounts": True,
                    "proxy_healthy_at_cleanup": True if self.closed else None,
                },
                "comparability": {
                    "schema_version": ("tasktrace-docker-network-enforcement-comparability.v1"),
                    "policy": self.policy.to_comparability_payload(),
                    "backend": "docker-forward-proxy-v1",
                    "allowed_ports": [80, 443],
                    "external_network": self.kwargs["external_network"],
                    "external_network_identity": {"configuration_sha256": "d" * 64},
                    "proxy_source_sha256": "a" * 64,
                    "proxy_image_identity": {"id": "sha256:" + "b" * 64},
                },
            }

    def fake_run(command: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        captured_commands.append(list(command))  # type: ignore[arg-type]
        output.mkdir(parents=True, exist_ok=True)
        for filename, content in {
            "cpu.stat.before": "usage_usec 0\nuser_usec 0\nsystem_usec 0\n",
            "cpu.stat.after": "usage_usec 1000\nuser_usec 700\nsystem_usec 300\n",
            "start_ns": "1000000000\n",
            "end_ns": "2000000000\n",
            "exit_code": "0\n",
        }.items():
            (output / filename).write_text(content)
        return subprocess.CompletedProcess(list(command), 0, "", "")  # type: ignore[arg-type]

    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {"id": "sha256:" + "f" * 64},
    )
    monkeypatch.setattr(container_module, "_run", fake_run)
    monkeypatch.setattr(network_enforcement, "DockerNetworkPolicySession", FakeNetworkSession)

    report = run_profile(
        _config(
            output,
            network_policy=policy,
            network_policy_proxy_image="proxy:test",
            network_policy_external_network="fixture-egress",
            env=(
                "TOKEN=kept",
                "HTTPS_PROXY=http://attacker.invalid:9",
                "ALL_PROXY=socks5://attacker.invalid:9",
            ),
            strace_mode="none",
        )
    )

    assert len(FakeNetworkSession.instances) == 1
    session = FakeNetworkSession.instances[0]
    assert session.policy == policy
    assert session.kwargs == {
        "docker": "docker",
        "proxy_image": "proxy:test",
        "external_network": "fixture-egress",
        "workload_mount_sources": (),
    }
    assert session.closed is True
    command = captured_commands[0]
    network_index = command.index("--network")
    assert command[network_index : network_index + 2] == [
        "--network",
        "tasktrace-policy-fixture",
    ]
    assert "NET_RAW" in command
    assert "TOKEN=kept" in command
    assert "HTTPS_PROXY=http://tasktrace-egress-proxy:3128" in command
    assert "https_proxy=http://tasktrace-egress-proxy:3128" in command
    assert "HTTPS_PROXY=http://attacker.invalid:9" not in command
    assert "ALL_PROXY=socks5://attacker.invalid:9" not in command
    assert "sha256:" + "f" * 64 in command
    assert report["network"] == "tasktrace-allowlist"
    assert report["network_policy"]["enforcement"]["state"] == "closed"
    assert report["network_policy"]["enforcement"]["comparability"]["allowed_ports"] == [80, 443]


def test_run_profile_cleans_network_session_when_execution_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from offline_replayer.measurement import network_enforcement

    closed: list[bool] = []

    class FakeNetworkSession:
        docker_network = None
        proxy_environment: dict[str, str] = {}

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def __enter__(self) -> "FakeNetworkSession":
            return self

        def close(self) -> list[str]:
            closed.append(True)
            return []

        @staticmethod
        def report_payload() -> dict[str, object]:
            return {"state": "closed", "comparability": {}}

    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {"id": "sha256:" + "f" * 64},
    )
    monkeypatch.setattr(
        container_module,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("launch failed")),
    )
    monkeypatch.setattr(network_enforcement, "DockerNetworkPolicySession", FakeNetworkSession)

    with pytest.raises(RuntimeError, match="launch failed"):
        run_profile(
            _config(
                tmp_path / "profile",
                network_policy=NetworkPolicy("public"),
                strace_mode="none",
            )
        )

    assert closed == [True]


def test_scheduler_concurrency_image_run_enables_container_gate(tmp_path: Path) -> None:
    command = build_image_command(_config(tmp_path, strace_mode="none", scheduler_concurrency=True))

    assert "TASKTRACE_SCHEDULER_GATE_DIR=/tasktrace-profile-output" in command
    assert "TASKTRACE_SCHEDULER_TRACING=1" in command


def test_proc_stat_comparison_image_run_enables_gate_without_scheduler(
    tmp_path: Path,
) -> None:
    command = build_image_command(_config(tmp_path, strace_mode="none", proc_stat_comparison=True))

    assert "TASKTRACE_SCHEDULER_GATE_DIR=/tasktrace-profile-output" in command


def test_memory_profile_enables_lifecycle_and_command_gates(tmp_path: Path) -> None:
    command = build_image_command(
        _config(
            tmp_path,
            strace_mode="none",
            memory_profile=True,
            memory_limit_bytes=512 * 1024 * 1024,
            memory_swap_total_limit_bytes=512 * 1024 * 1024,
        )
    )

    assert "TASKTRACE_RESOURCE_GATE_DIR=/tasktrace-profile-output" in command
    assert "TASKTRACE_SCHEDULER_GATE_DIR=/tasktrace-profile-output" in command
    assert "TASKTRACE_SCHEDULER_TRACING=1" not in command
    assert ["--memory", str(512 * 1024 * 1024)] == command[command.index("--memory") : command.index("--memory") + 2]
    assert ["--memory-swap", str(512 * 1024 * 1024)] == command[
        command.index("--memory-swap") : command.index("--memory-swap") + 2
    ]


def test_build_report_calculates_utilization_from_cgroup_delta(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat.before").write_text("usage_usec 1000000\nuser_usec 600000\nsystem_usec 400000\n")
    (tmp_path / "cpu.stat.after").write_text("usage_usec 21000000\nuser_usec 15600000\nsystem_usec 5400000\n")
    (tmp_path / "start_ns").write_text("1000000000\n")
    (tmp_path / "end_ns").write_text("11000000000\n")
    (tmp_path / "exit_code").write_text("0\n")
    (tmp_path / "process.strace.log").write_text("execve(...)\n")

    report = build_report(
        _config(tmp_path),
        scope="disposable-image-container",
        docker_returncode=0,
        image_identity={"id": "sha256:abc", "repo_digests": [], "created": "today"},
    )

    assert report["cpu"]["usage_sec"] == 20.0
    assert report["cpu"]["average_logical_cores_used"] == 2.0
    assert report["cpu"]["allocated_logical_cpus"] == 8.0
    assert report["cpu"]["utilization_percent_of_allocation"] == 25.0
    assert report["cpu"]["enforced_logical_cpu_limit"] is None
    assert report["image_identity"]["id"] == "sha256:abc"
    assert report["strace"]["log_path"] == str(tmp_path / "process.strace.log")
    assert report["strace"]["structured_report_path"] is None
    assert report["process_exec"] is None
    assert not (tmp_path / "process_exec_report.json").exists()


def _coordinated_cpu_report(**overrides: object) -> dict[str, object]:
    report: dict[str, object] = {
        "schema_version": "cgroup-cpu-coordinated.v1",
        "method": "container-cgroup-v2-cpu.stat",
        "scope_id": "docker-cgroup-fixture",
        "window_start_ns": 2_000_000_000,
        "window_end_ns": 6_000_000_000,
        "wall_sec": 4.0,
        "usage_sec": 6.0,
        "user_sec": 4.5,
        "system_sec": 1.5,
        "average_logical_cores": 1.5,
        "cpu_stat_delta": {
            "usage_usec": 6_000_000,
            "user_usec": 4_500_000,
            "system_usec": 1_500_000,
        },
        "leaf_cpu": {
            "valid_for_cpu_performance": True,
            "finite_limit_present": False,
            "throttling_observed": False,
        },
        "ancestor_cpu": {
            "valid_for_cpu_performance": True,
            "finite_limits_present": False,
            "throttling_observed": False,
        },
        "raw_source_snapshot_schema_version": ("container-cgroup-cpu-source-snapshots.v1"),
        "raw_source_snapshot_read_count": 12,
        "valid_for_cpu_performance": True,
        "invalid_reasons": [],
    }
    report.update(overrides)
    return report


@pytest.mark.parametrize("legacy_cpu_stat", ["missing", "conflicting"])
def test_build_report_uses_coordinated_cpu_and_preserves_public_aliases(
    tmp_path: Path,
    legacy_cpu_stat: str,
) -> None:
    (tmp_path / "start_ns").write_text("1000000000\n")
    (tmp_path / "end_ns").write_text("11000000000\n")
    (tmp_path / "exit_code").write_text("0\n")
    if legacy_cpu_stat == "conflicting":
        (tmp_path / "cpu.stat.before").write_text("usage_usec 0\nuser_usec 0\nsystem_usec 0\n")
        (tmp_path / "cpu.stat.after").write_text("usage_usec 99000000\nuser_usec 88000000\nsystem_usec 11000000\n")
    coordinated = _coordinated_cpu_report()

    report = build_report(
        _config(
            tmp_path,
            cpuset_cpus="0-3",
            cpu_limit=3.0,
            strace_mode="none",
        ),
        scope="disposable-image-container",
        docker_returncode=0,
        coordinated_cpu_report=coordinated,
    )

    cpu = report["cpu"]
    assert cpu["wall_clock_source"] == "host-monotonic"
    assert cpu["wall_sec"] == 4.0
    assert cpu["usage_sec"] == 6.0
    assert cpu["user_sec"] == 4.5
    assert cpu["system_sec"] == 1.5
    assert cpu["average_logical_cores"] == 1.5
    assert cpu["average_logical_cores_used"] == 1.5
    assert cpu["allocated_logical_cpus"] == 4.0
    assert cpu["utilization_fraction_of_allocation"] == 0.375
    assert cpu["utilization_percent_of_allocation"] == 37.5
    assert cpu["enforced_logical_cpu_limit"] == 3.0
    assert cpu["enforced_cpu_limit_source"] == "docker --cpus"
    assert cpu["scope_id"] == "docker-cgroup-fixture"
    assert cpu["window_start_ns"] == 2_000_000_000
    assert cpu["window_end_ns"] == 6_000_000_000
    assert cpu["leaf_cpu"] == coordinated["leaf_cpu"]
    assert cpu["ancestor_cpu"] == coordinated["ancestor_cpu"]
    assert cpu["valid_for_cpu_performance"] is True
    assert cpu["invalid_reasons"] == []
    assert cpu["raw_source_snapshot_schema_version"] == ("container-cgroup-cpu-source-snapshots.v1")


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("wall_sec", None),
        ("usage_sec", "not-a-number"),
        ("user_sec", object()),
        ("system_sec", None),
        ("average_logical_cores", "invalid"),
    ],
)
def test_build_report_rejects_malformed_coordinated_cpu_numerics(
    tmp_path: Path,
    field: str,
    malformed: object,
) -> None:
    (tmp_path / "start_ns").write_text("1000000000\n")
    (tmp_path / "end_ns").write_text("2000000000\n")
    (tmp_path / "exit_code").write_text("0\n")
    coordinated = _coordinated_cpu_report(**{field: malformed})

    with pytest.raises(
        ProfileError,
        match="coordinated cgroup CPU report is missing required numeric fields",
    ):
        build_report(
            _config(tmp_path, strace_mode="none"),
            scope="disposable-image-container",
            docker_returncode=0,
            coordinated_cpu_report=coordinated,
        )


def test_build_report_attaches_deterministic_structured_process_evidence(
    tmp_path: Path,
) -> None:
    for filename, content in {
        "cpu.stat.before": "usage_usec 1000000\nuser_usec 600000\nsystem_usec 400000\n",
        "cpu.stat.after": "usage_usec 3000000\nuser_usec 1600000\nsystem_usec 1400000\n",
        "start_ns": "1000000000\n",
        "end_ns": "3000000000\n",
        "exit_code": "0\n",
    }.items():
        (tmp_path / filename).write_text(content)
    trace_path = tmp_path / "process.strace.log"
    trace_path.write_text("""\
100 1700000000.000000001 execve("/bin/bash", ["bash", "-c", "make"], 0x1) = 0
100 1700000000.010000000 chdir("project") = 0
100 1700000000.020000000 clone(child_stack=NULL, flags=SIGCHLD) = 101
101 1700000000.030000000 execve("/usr/bin/make", ["make", "all"], 0x1) = 0
101 1700000000.040000000 +++ exited with 0 +++
100 1700000000.050000000 +++ exited with 0 +++
""")
    config = _config(tmp_path, strace_mode="process")

    first = build_report(config, scope="disposable-image-container", docker_returncode=0)
    structured_path = tmp_path / "process_exec_report.json"
    first_bytes = structured_path.read_bytes()
    second = build_report(config, scope="disposable-image-container", docker_returncode=0)

    assert first["cpu"]["usage_sec"] == 2.0
    assert first["process_exec"]["valid"] is True
    assert first["process_exec"]["scope"]["root_pid"] == 100
    assert first["process_exec"]["summary"]["process_count"] == 2
    assert first["process_exec"]["artifacts"] == {
        "raw_trace_path": str(trace_path),
        "report_path": str(structured_path),
    }
    assert first["strace"] == {
        "mode": "process",
        "log_path": str(trace_path),
        "structured_report_path": str(structured_path),
        "string_limit": 4096,
    }
    child = next(process for process in first["process_exec"]["processes"] if process["pid"] == 101)
    assert child["parent_pid"] == 100
    assert child["binary"] == "/usr/bin/make"
    assert child["argv"] == ["make", "all"]
    assert child["cwd"] == "/workdir/project"
    assert json.loads(first_bytes) == first["process_exec"]
    assert structured_path.read_bytes() == first_bytes
    assert second["process_exec"] == first["process_exec"]


def test_build_report_retains_invalid_process_evidence_without_failing_cpu(
    tmp_path: Path,
) -> None:
    for filename, content in {
        "cpu.stat.before": "usage_usec 0\nuser_usec 0\nsystem_usec 0\n",
        "cpu.stat.after": "usage_usec 1000000\nuser_usec 800000\nsystem_usec 200000\n",
        "start_ns": "0\n",
        "end_ns": "1000000000\n",
        "exit_code": "0\n",
    }.items():
        (tmp_path / filename).write_text(content)
    (tmp_path / "process.strace.log").write_text("""\
200 1700000100.000000000 execve("/bin/tool", ["tool", ...], 0x1) = 0
200 1700000100.100000000 clone(child_stack=NULL, flags=SIGCHLD) = 201
""")

    report = build_report(
        _config(tmp_path, strace_mode="process"),
        scope="disposable-image-container",
        docker_returncode=0,
    )

    assert report["command_exit_code"] == 0
    assert report["cpu"]["usage_sec"] == 1.0
    assert report["process_exec"]["valid"] is False
    assert report["process_exec"]["trace_complete"] is False
    codes = {item["code"] for item in report["process_exec"]["anomalies"]}
    assert "truncated_or_invalid_argv" in codes
    assert "unresolved_process_lifecycle" in codes
    assert report["strace"]["structured_report_path"] == str(tmp_path / "process_exec_report.json")
    assert json.loads((tmp_path / "process_exec_report.json").read_text()) == report["process_exec"]


def test_existing_container_report_warns_about_concurrent_container_work(
    tmp_path: Path,
) -> None:
    for filename, content in {
        "cpu.stat.before": "usage_usec 0\n",
        "cpu.stat.after": "usage_usec 1000000\n",
        "start_ns": "0\n",
        "end_ns": "1000000000\n",
        "exit_code": "0\n",
    }.items():
        (tmp_path / filename).write_text(content)
    config = _config(
        tmp_path,
        image=None,
        container="already-running",
        cpuset_cpus=None,
        allocated_cpus=8.0,
    )

    report = build_report(config, scope="existing-container", docker_returncode=0)

    assert "other concurrent work" in report["warning"]


def test_scheduler_report_warns_about_privileged_trace_overhead(tmp_path: Path) -> None:
    for filename, content in {
        "cpu.stat.before": "usage_usec 0\n",
        "cpu.stat.after": "usage_usec 1000000\n",
        "start_ns": "0\n",
        "end_ns": "1000000000\n",
        "exit_code": "0\n",
    }.items():
        (tmp_path / filename).write_text(content)

    report = build_report(
        _config(tmp_path, strace_mode="none", scheduler_concurrency=True),
        scope="disposable-image-container",
        docker_returncode=0,
    )

    assert "bcc scheduler-tracing overhead" in report["warning"]


def test_perf_scheduler_report_names_selected_backend(tmp_path: Path) -> None:
    for filename, content in {
        "cpu.stat.before": "usage_usec 0\n",
        "cpu.stat.after": "usage_usec 1000000\n",
        "start_ns": "0\n",
        "end_ns": "1000000000\n",
        "exit_code": "0\n",
    }.items():
        (tmp_path / filename).write_text(content)

    report = build_report(
        _config(
            tmp_path,
            strace_mode="none",
            scheduler_concurrency=True,
            scheduler_backend="perf",
        ),
        scope="disposable-image-container",
        docker_returncode=0,
    )

    assert "perf scheduler-tracing overhead" in report["warning"]


def test_proc_stat_comparison_report_does_not_claim_task_peak(tmp_path: Path) -> None:
    for filename, content in {
        "cpu.stat.before": "usage_usec 0\n",
        "cpu.stat.after": "usage_usec 1000000\n",
        "start_ns": "0\n",
        "end_ns": "1000000000\n",
        "exit_code": "0\n",
    }.items():
        (tmp_path / filename).write_text(content)

    report = build_report(
        _config(tmp_path, strace_mode="none", proc_stat_comparison=True),
        scope="disposable-image-container",
        docker_returncode=0,
    )

    assert "sampled host /proc/stat context" in report["warning"]
    assert "not task peak concurrency" in report["warning"]


def test_run_profile_rejects_nonempty_output_directory(tmp_path: Path) -> None:
    (tmp_path / "stale-process.strace.log").write_text("old")

    with pytest.raises(ProfileError, match="new or empty"):
        run_profile(_config(tmp_path, strace_mode="none"))


def test_memory_gate_preserves_artifacts_after_prepare_failure(tmp_path: Path) -> None:
    output = tmp_path / "profile"
    output.mkdir()
    environment = {
        **os.environ,
        "TASKTRACE_PROFILE_DIR": str(output),
        "TASKTRACE_PROFILE_COMMAND": "true",
        "TASKTRACE_PROFILE_PREPARE_COMMAND": "false",
        "TASKTRACE_PROFILE_SHELL": "/bin/sh",
        "TASKTRACE_STRACE_MODE": "none",
        "TASKTRACE_RESOURCE_GATE_DIR": str(output),
        "TASKTRACE_SCHEDULER_GATE_DIR": str(output),
    }
    process = subprocess.Popen(
        ["/bin/sh", str(container_module.ENTRYPOINT)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )

    def wait_for(path: Path) -> None:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if path.exists():
                return
            if process.poll() is not None:
                _stdout, stderr = process.communicate()
                raise AssertionError(f"entrypoint exited early: {stderr}")
            time.sleep(0.01)
        raise AssertionError(f"timed out waiting for {path}")

    try:
        wait_for(output / "sandbox_attach_ready")
        container_module._release_gate(output / "sandbox.start.gate")
        wait_for(output / "scheduler_attach_ready")
        container_module._release_gate(output / "scheduler.start.gate")
        wait_for(output / "scheduler_measurement_done")

        assert (output / "prepare_exit_code").read_text().strip() == "1"
        assert (output / "exit_code").read_text().strip() == "1"
        start_ns = int((output / "start_ns").read_text())
        end_ns = int((output / "end_ns").read_text())
        assert start_ns > 0
        assert end_ns >= start_ns
        assert (output / "cpu.stat.before").is_file()
        assert (output / "cpu.stat.after").is_file()
        assert "prepare command failed" in (output / "command.stderr.txt").read_text()

        container_module._release_gate(output / "scheduler.finish.gate")
        _stdout, _stderr = process.communicate(timeout=5)
        assert process.returncode == 1
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


def test_scheduler_entrypoint_avoids_in_window_timestamp_helpers() -> None:
    script = container_module.ENTRYPOINT.read_text()

    assert 'timestamp="$(date +%s%N 2>/dev/null || true)"' in script
    assert 'timestamp="$(timestamp_ns' not in script
    assert 'printf \'0\\n\' > "$TASKTRACE_PROFILE_DIR/start_ns"' in script
    assert 'printf \'0\\n\' > "$TASKTRACE_PROFILE_DIR/end_ns"' in script
    assert 'chmod a+rw "$TASKTRACE_PROFILE_DIR/start_ns"' in script
    assert 'chmod a+rw "$TASKTRACE_PROFILE_DIR/end_ns"' in script


def test_enroot_entrypoint_keeps_one_monotonic_clock_when_date_changes(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    fake_date = fake_bin / "date"
    fake_date.write_text("#!/bin/sh\nprintf 'unsupported-N\\n'\n")
    fake_date.chmod(0o755)
    quoted_date = shlex.quote(str(fake_date))
    command = (
        f"printf '%s\\n' '#!/bin/sh' \"printf '1700000000000000000\\\\n'\" "
        f"> {quoted_date}; chmod +x {quoted_date}; sleep 0.05"
    )
    environment = {
        **os.environ,
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "TASKTRACE_PROFILE_DIR": str(output),
        "TASKTRACE_PROFILE_COMMAND": command,
        "TASKTRACE_PROFILE_SHELL": "/bin/sh",
        "TASKTRACE_PROFILE_SKIP_CPU_STATS": "1",
        "TASKTRACE_STRACE_MODE": "none",
    }

    completed = subprocess.run(
        ["/bin/sh", str(container_module.ENTRYPOINT)],
        text=True,
        capture_output=True,
        env=environment,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    start_ns = int((output / "start_ns").read_text())
    end_ns = int((output / "end_ns").read_text())
    assert 0 < end_ns - start_ns < 1_000_000_000_000
    assert (output / "clock_source").read_text().strip() == "linux-proc-uptime"
    assert (output / "clock_name").read_text().strip() == "CLOCK_BOOTTIME"
    assert int((output / "clock_resolution_ns").read_text()) > 0
    assert (output / "clock_boot_id").read_text().strip()
    assert (output / "clock_time_namespace").read_text().strip().startswith("time:[")


def test_entrypoint_invokes_post_measurement_verifier_exactly_once() -> None:
    script = container_module.ENTRYPOINT.read_text()

    assert script.count('"$command_shell" -c "$verification_command"') == 1


def test_process_entrypoint_requests_cwd_events_and_pid_prefixed_shared_output() -> None:
    script = container_module.ENTRYPOINT.read_text()

    assert 'execve)\n        strace_filter="execve"' in script
    assert 'process)\n' in script
    assert 'strace_filter="%process,chdir,fchdir"' in script
    assert 'strace -f -ttt -s "$string_limit"' in script
    assert '-o "$TASKTRACE_PROFILE_DIR/process.strace.log"' in script
    assert "strace -ff" not in script


def test_run_profile_attaches_optional_memory_report(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "profile"
    config = _config(
        output,
        strace_mode="none",
        memory_profile=True,
        memory_limit_bytes=512 * 1024 * 1024,
    )
    calls = 0
    executed_images: list[str | None] = []

    def fake_observers(observer_config: ProfileConfig):
        nonlocal calls
        calls += 1
        executed_images.append(observer_config.image)
        target = observer_config.output_dir
        for filename, content in {
            "cpu.stat.before": "usage_usec 0\n",
            "cpu.stat.after": "usage_usec 1000\n",
            "start_ns": "1000000000\n",
            "end_ns": "2000000000\n",
            "exit_code": "0\n",
        }.items():
            (target / filename).write_text(content)
        completed = subprocess.CompletedProcess(["docker"], 0, "", "")
        memory = {
            "instrumentation_valid": True,
            "instrumentation_invalid_reasons": [],
            "peak_valid": True,
            "natural_peak_trustworthy": True,
            "natural_peak_invalid_reasons": [],
            "censoring": "none",
            "lifetime": {"peak_bytes": 1234},
        }
        return completed, None, None, memory

    monkeypatch.setattr(
        container_module,
        "_inspect_image_identity",
        lambda _config: {"id": "sha256:image"},
    )
    monkeypatch.setattr(container_module, "_run_image_with_host_observers", fake_observers)

    report = run_profile(config)
    second_report = run_profile(replace(config, output_dir=tmp_path / "profile-second"))
    traced_report = run_profile(
        replace(
            config,
            output_dir=tmp_path / "profile-traced",
            strace_mode="process",
        )
    )

    assert calls == 3
    assert executed_images == ["sha256:image", "sha256:image", "sha256:image"]
    assert "profile_run_id" not in report
    uuid.UUID(report["run_id"])
    uuid.UUID(report["memory"]["profile_run_id"])
    assert report["memory"]["profile_run_id"] == report["run_id"]
    assert second_report["run_id"] != report["run_id"]
    assert report["profile_started_at_utc"].endswith("+00:00")
    assert report["memory"]["profile_started_at_utc"].endswith("+00:00")
    assert len(report["memory"]["controller_source_sha256"]) == 64
    assert report["memory"]["lifetime"]["peak_bytes"] == 1234
    provenance = report["memory"]["execution_provenance"]
    assert provenance["runtime"] == "docker"
    assert provenance["image"] == "swebench/task:latest"
    assert provenance["image_identity"] == {"id": "sha256:image"}
    assert provenance["host_platform"]["kernel_release"]
    assert provenance["host_platform"]["machine"]
    assert provenance["host_platform"]["cgroup_version"] == "v2"
    assert provenance["host_platform"]["page_size_bytes"] > 0
    assert provenance["scheduler_concurrency"] is False
    assert provenance["proc_stat_comparison"] is False
    assert len(provenance["controller_source_sha256"]) == 64
    assert provenance["controller_source_sha256"] == report["memory"]["controller_source_sha256"]
    assert len(provenance["comparability_sha256"]) == 64
    assert json.loads((output / "profile_report.json").read_text()) == report
    assert traced_report["memory"]["natural_peak_trustworthy"] is False
    assert any(
        "strace ran inside the measured cgroup" in reason
        for reason in traced_report["memory"]["natural_peak_invalid_reasons"]
    )


def test_memory_provenance_normalizes_run_specific_mount_sources(
    tmp_path: Path,
) -> None:
    first = _config(
        tmp_path / "first",
        env=("B=2", "A=1"),
        mounts=("type=bind,src=/run/one,dst=/tasktrace",),
        memory_limit_bytes=512,
        memory_swap_total_limit_bytes=512,
    )
    second = _config(
        tmp_path / "second",
        env=("A=1", "B=2"),
        mounts=("type=bind,src=/run/two,dst=/tasktrace",),
        memory_limit_bytes=1024,
        memory_swap_total_limit_bytes=2048,
    )

    first_provenance = container_module._memory_execution_provenance(first, {"id": "sha256:image"})
    second_provenance = container_module._memory_execution_provenance(second, {"id": "sha256:image"})

    assert first_provenance["raw_mounts"] != second_provenance["raw_mounts"]
    assert first_provenance["normalized_mounts"] == [{"type": "bind", "destination": "/tasktrace", "readonly": False}]
    assert first_provenance["configured_swap_policy"] == "disabled"
    assert second_provenance["configured_swap_policy"] == "finite-enabled"
    assert first_provenance["comparability_excludes_memory_limits"] is True
    assert first_provenance["memory_sample_interval_sec"] == 0.1
    assert set(container_module.EXECUTION_PROVENANCE_FIELDS).union(
        container_module.EXECUTION_PROVENANCE_OPTIONAL_FIELDS
    ) == set(first_provenance)
    assert container_module.execution_comparability_sha256(first_provenance) == first_provenance["comparability_sha256"]
    assert first_provenance["comparability_sha256"] == second_provenance["comparability_sha256"]


def test_memory_provenance_hashes_environment_values_without_leaking_them(
    tmp_path: Path,
) -> None:
    secret = "sk-sensitive-value-that-must-not-be-archived"
    first = _config(
        tmp_path / "first",
        env=(f"OPENAI_API_KEY={secret}", "EMPTY=", "FLAG_WITHOUT_EQUALS"),
    )
    second = replace(
        first,
        output_dir=tmp_path / "second",
        env=("OPENAI_API_KEY=different", "EMPTY=", "FLAG_WITHOUT_EQUALS"),
    )

    first_provenance = container_module._memory_execution_provenance(first, {"id": "sha256:image"})
    second_provenance = container_module._memory_execution_provenance(second, {"id": "sha256:image"})
    encoded = json.dumps(first_provenance, sort_keys=True)

    assert secret not in encoded
    assert "OPENAI_API_KEY" in encoded
    assert all("value" not in item for item in first_provenance["environment"])
    assert all(len(item["value_sha256"]) == 64 for item in first_provenance["environment"])
    assert first_provenance["comparability_sha256"] != second_provenance["comparability_sha256"]


def test_immutable_execution_config_uses_inspected_docker_image_id(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)

    resolved = container_module._immutable_execution_config(config, {"id": "sha256:immutable"})

    assert config.image == "swebench/task:latest"
    assert resolved.image == "sha256:immutable"
    with pytest.raises(ProfileError, match="immutable sha256"):
        container_module._immutable_execution_config(config, {"id": "mutable"})


def test_inspect_image_identity_retains_platform_and_safe_config_provenance(tmp_path: Path, monkeypatch) -> None:
    image_id = "sha256:" + "a" * 64
    source_digest = "example.invalid/task@sha256:" + "b" * 64
    layer_digests = ["sha256:" + "c" * 64, "sha256:" + "d" * 64]
    secret = "must-not-be-archived"
    metadata = {
        "Id": image_id,
        "RepoDigests": [source_digest],
        "Created": "2026-06-30T12:34:56Z",
        "Architecture": "arm64",
        "Os": "linux",
        "Variant": "v8",
        "Size": 123456789,
        "RootFS": {"Type": "layers", "Layers": layer_digests},
        "Config": {
            "User": "1000:1000",
            "WorkingDir": "/workspace",
            "Entrypoint": ["/usr/bin/tini", "--"],
            "Cmd": ["/bin/sh"],
            "Shell": ["/bin/bash", "-c"],
            "Env": [f"API_KEY={secret}"],
            "Labels": {"untrusted.example/secret": secret},
        },
    }

    monkeypatch.setattr(
        container_module,
        "_run",
        lambda command: subprocess.CompletedProcess(command, 0, json.dumps([metadata]), ""),
    )

    identity = container_module._inspect_image_identity(_config(tmp_path))

    assert identity == {
        "id": image_id,
        "repo_digests": [source_digest],
        "created": "2026-06-30T12:34:56Z",
        "architecture": "arm64",
        "os": "linux",
        "variant": "v8",
        "size_bytes": 123456789,
        "rootfs": {"type": "layers", "layers": layer_digests},
        "config": {
            "user": "1000:1000",
            "working_dir": "/workspace",
            "entrypoint": ["/usr/bin/tini", "--"],
            "cmd": ["/bin/sh"],
            "shell": ["/bin/bash", "-c"],
        },
    }
    assert secret not in json.dumps(identity, sort_keys=True)


def test_inspect_image_identity_preserves_legacy_fields_when_optional_metadata_absent(
    tmp_path: Path, monkeypatch
) -> None:
    metadata = {
        "Id": "sha256:" + "a" * 64,
        "RepoDigests": [],
        "Created": "2026-06-30T12:34:56Z",
    }
    monkeypatch.setattr(
        container_module,
        "_run",
        lambda command: subprocess.CompletedProcess(command, 0, json.dumps([metadata]), ""),
    )

    identity = container_module._inspect_image_identity(_config(tmp_path))

    assert identity["id"] == metadata["Id"]
    assert identity["repo_digests"] == []
    assert identity["created"] == metadata["Created"]
    assert identity["architecture"] is None
    assert identity["os"] is None
    assert identity["variant"] is None
    assert identity["rootfs"] == {"type": None, "layers": []}
    assert identity["config"] == {
        "user": None,
        "working_dir": None,
        "entrypoint": None,
        "cmd": None,
        "shell": None,
    }


@pytest.mark.parametrize("payload", ["{}", "[]", "[null]", "[{}, {}]"])
def test_inspect_image_identity_rejects_non_single_image_metadata(tmp_path: Path, monkeypatch, payload: str) -> None:
    monkeypatch.setattr(
        container_module,
        "_run",
        lambda command: subprocess.CompletedProcess(command, 0, payload, ""),
    )

    with pytest.raises(ProfileError, match="invalid Docker image inspection output"):
        container_module._inspect_image_identity(_config(tmp_path))


def test_coordinated_baseline_is_explicit_unprofiled_disposable_docker_mode(
    tmp_path: Path,
) -> None:
    baseline = _config(tmp_path, coordinated_baseline=True)

    container_module.validate_profile_config(baseline)
    environment = container_module._profile_environment(baseline, "/tasktrace-profile-output")
    assert "TASKTRACE_RESOURCE_GATE_DIR=/tasktrace-profile-output" in environment
    assert "TASKTRACE_SCHEDULER_GATE_DIR=/tasktrace-profile-output" in environment
    assert "TASKTRACE_SCHEDULER_TRACING=1" not in environment

    with pytest.raises(ProfileError, match="mutually exclusive"):
        container_module.validate_profile_config(replace(baseline, profile_sets=("cpu-compare",)))
    with pytest.raises(ProfileError, match="cannot enable"):
        container_module.validate_profile_config(replace(baseline, memory_profile=True))
    with pytest.raises(ProfileError, match="disposable --image"):
        container_module.validate_profile_config(replace(baseline, image=None, container="existing"))


@pytest.mark.parametrize(
    ("events_delta", "expected_status"),
    [
        ({"high": 1, "oom_kill": 0, "oom_group_kill": 0}, 0),
        ({"high": 0, "oom_kill": 1, "oom_group_kill": 0}, 1),
    ],
)
def test_standalone_cli_accepts_pressure_but_fails_on_oom_kill(
    tmp_path: Path,
    monkeypatch,
    events_delta: dict[str, int],
    expected_status: int,
) -> None:
    captured: list[ProfileConfig] = []

    def fake_run_profile(config: ProfileConfig) -> dict[str, object]:
        captured.append(config)
        return {
            "command_exit_code": 0,
            "cpu": {
                "wall_sec": 1.0,
                "usage_sec": 1.0,
                "average_logical_cores_used": 1.0,
                "allocated_logical_cpus": 1.0,
                "utilization_percent_of_allocation": 100.0,
            },
            "memory": {
                "lifetime": {"peak_bytes": 1024},
                "command_window": {"sampled_peak_bytes": 1024},
                "samples_path": str(tmp_path / "samples.json"),
                "instrumentation_valid": True,
                "peak_valid": True,
                "natural_peak_trustworthy": False,
                "timeline_valid_for_shape": True,
                "observed_limit_margin": {},
                "events_delta": events_delta,
            },
        }

    monkeypatch.setattr(container_module, "run_profile", fake_run_profile)

    status = container_module.main(
        [
            "--image",
            "task:local",
            "--command",
            "true",
            "--output-dir",
            str(tmp_path / "profile"),
            "--allocated-cpus",
            "1",
            "--strace-mode",
            "none",
            "--memory-profile",
        ]
    )

    assert status == expected_status
    assert captured[0].memory_profile is True


def test_standalone_cli_passes_repeated_profile_sets_and_hardware_options(tmp_path: Path, monkeypatch) -> None:
    captured: list[ProfileConfig] = []

    def fake_run_profile(config: ProfileConfig) -> dict[str, object]:
        captured.append(config)
        return {
            "command_exit_code": 0,
            "cpu": {
                "wall_sec": 1.0,
                "usage_sec": 1.0,
                "average_logical_cores_used": 1.0,
                "allocated_logical_cpus": 1.0,
                "utilization_percent_of_allocation": 100.0,
            },
            "profiling_valid": True,
            "evidence_bundle": {"path": str(tmp_path / "evidence_bundle.json")},
        }

    monkeypatch.setattr(container_module, "run_profile", fake_run_profile)

    status = container_module.main(
        [
            "--image",
            "task:local",
            "--command",
            "true",
            "--output-dir",
            str(tmp_path / "profile"),
            "--allocated-cpus",
            "1",
            "--profile-set",
            "software-deep",
            "--profile-set",
            "hardware-deep",
            "--hardware-perf-event",
            "task-clock",
            "--hardware-perf-minimum-running-ratio",
            "0.99",
        ]
    )

    assert status == 0
    assert captured[0].profile_sets == ("software-deep", "hardware-deep")
    assert captured[0].hardware_perf_events == ("task-clock",)
    assert captured[0].hardware_perf_minimum_running_ratio == 0.99


def test_standalone_cli_passes_explicit_coordinated_baseline(tmp_path: Path, monkeypatch) -> None:
    captured: list[ProfileConfig] = []

    def fake_run_profile(config: ProfileConfig) -> dict[str, object]:
        captured.append(config)
        return {
            "command_exit_code": 0,
            "cpu": {
                "wall_sec": 1.0,
                "usage_sec": 0.5,
                "average_logical_cores_used": 0.5,
                "allocated_logical_cpus": 1.0,
                "utilization_percent_of_allocation": 50.0,
            },
            "profiling_valid": True,
            "evidence_bundle": {"path": str(tmp_path / "evidence_bundle.json")},
        }

    monkeypatch.setattr(container_module, "run_profile", fake_run_profile)

    status = container_module.main(
        [
            "--image",
            "task:local",
            "--command",
            "true",
            "--output-dir",
            str(tmp_path / "baseline"),
            "--allocated-cpus",
            "1",
            "--coordinated-baseline",
        ]
    )

    assert status == 0
    assert len(captured) == 1
    assert captured[0].coordinated_baseline is True
    assert captured[0].profile_sets == ()
