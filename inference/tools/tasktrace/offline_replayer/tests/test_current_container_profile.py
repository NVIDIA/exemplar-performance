# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest
from offline_replayer.measurement import current
from offline_replayer.measurement.container import ProfileConfig, ProfileError
from offline_replayer.measurement.current import (
    CgroupV2Location,
    parse_cpu_max,
    resolve_current_cgroup,
    run_current_profile,
)


def _config(tmp_path: Path, **overrides: object) -> ProfileConfig:
    values: dict[str, object] = {
        "command": "true",
        "prepare_command": None,
        "output_dir": tmp_path / "profile",
        "image": None,
        "container": None,
        "cpuset_cpus": None,
        "cpu_limit": None,
        "allocated_cpus": None,
        "workdir": str(tmp_path),
        "user": None,
        "env": (),
        "mounts": (),
        "strace_mode": "none",
        "string_limit": 4096,
        "shell": "/bin/bash",
        "runtime": "current",
        "current_container_exclusive": True,
    }
    values.update(overrides)
    return ProfileConfig(**values)  # type: ignore[arg-type]


def _fake_cgroup(tmp_path: Path) -> CgroupV2Location:
    cgroup = tmp_path / "cgroup"
    cgroup.mkdir()
    (cgroup / "cpu.stat").write_text(
        "usage_usec 1000000\n"
        "user_usec 600000\n"
        "system_usec 400000\n"
        "nr_periods 10\n"
        "nr_throttled 1\n"
        "throttled_usec 100000\n"
    )
    (cgroup / "cpu.max").write_text("100000 100000\n")
    (cgroup / "cpuset.cpus.effective").write_text("0-3\n")
    (cgroup / "cgroup.procs").write_text(f"{os.getpid()}\n")
    return CgroupV2Location(
        mount_point=cgroup,
        mount_root="/",
        membership_path="/",
        path=cgroup,
    )


def test_resolve_current_cgroup_supports_private_namespace_root(tmp_path: Path) -> None:
    mount_point = tmp_path / "cgroup"
    mount_point.mkdir()
    (mount_point / "cpu.stat").write_text("usage_usec 0\n")
    (mount_point / "cgroup.procs").write_text(f"{os.getpid()}\n")
    proc_cgroup = tmp_path / "cgroup.txt"
    proc_cgroup.write_text("0::/\n")
    mountinfo = tmp_path / "mountinfo.txt"
    mountinfo.write_text(f"29 23 0:26 / {mount_point} rw,nosuid,nodev,noexec,relatime " "- cgroup2 cgroup rw\n")

    location = resolve_current_cgroup(
        proc_self_cgroup=proc_cgroup,
        proc_self_mountinfo=mountinfo,
    )

    assert location.path == mount_point
    assert location.membership_path == "/"


def test_resolve_current_cgroup_supports_host_relative_membership(tmp_path: Path) -> None:
    mount_point = tmp_path / "cgroup"
    scope = mount_point / "kubepods.slice" / "pod.scope"
    scope.mkdir(parents=True)
    (scope / "cpu.stat").write_text("usage_usec 0\n")
    (scope / "cgroup.procs").write_text(f"{os.getpid()}\n")
    proc_cgroup = tmp_path / "cgroup.txt"
    proc_cgroup.write_text("0::/kubepods.slice/pod.scope\n")
    mountinfo = tmp_path / "mountinfo.txt"
    mountinfo.write_text(f"29 23 0:26 / {mount_point} rw,nosuid,nodev,noexec,relatime " "- cgroup2 cgroup rw\n")

    location = resolve_current_cgroup(
        proc_self_cgroup=proc_cgroup,
        proc_self_mountinfo=mountinfo,
    )

    assert location.path == scope


def test_resolve_current_cgroup_maps_membership_relative_to_mount_root(
    tmp_path: Path,
) -> None:
    mount_point = tmp_path / "cgroup"
    child = mount_point / "container"
    child.mkdir(parents=True)
    (child / "cpu.stat").write_text("usage_usec 0\n")
    (child / "cgroup.procs").write_text(f"{os.getpid()}\n")
    proc_cgroup = tmp_path / "cgroup.txt"
    proc_cgroup.write_text("0::/kubepods.slice/pod.scope/container\n")
    mountinfo = tmp_path / "mountinfo.txt"
    mountinfo.write_text(
        f"29 23 0:26 /kubepods.slice/pod.scope {mount_point} " "rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n"
    )

    location = resolve_current_cgroup(
        proc_self_cgroup=proc_cgroup,
        proc_self_mountinfo=mountinfo,
    )

    assert location.path == child
    assert location.mount_root == "/kubepods.slice/pod.scope"


def test_resolve_current_cgroup_requires_cpu_stat(tmp_path: Path) -> None:
    mount_point = tmp_path / "cgroup"
    mount_point.mkdir()
    proc_cgroup = tmp_path / "cgroup.txt"
    proc_cgroup.write_text("0::/\n")
    mountinfo = tmp_path / "mountinfo.txt"
    mountinfo.write_text(f"29 23 0:26 / {mount_point} rw - cgroup2 cgroup rw\n")

    with pytest.raises(ProfileError, match="requires readable cgroup v2"):
        resolve_current_cgroup(
            proc_self_cgroup=proc_cgroup,
            proc_self_mountinfo=mountinfo,
        )


def test_parse_cpu_max_handles_quota_and_unlimited() -> None:
    assert parse_cpu_max("100000 100000") == 1.0
    assert parse_cpu_max("250000 100000") == 2.5
    assert parse_cpu_max("max 100000") is None


def test_allocation_uses_tightest_visible_ancestor_quota(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    mount = tmp_path / "cgroup"
    child = mount / "child"
    child.mkdir(parents=True)
    (mount / "cpu.max").write_text("200000 100000\n")
    (child / "cpu.max").write_text("max 100000\n")
    (child / "cpuset.cpus.effective").write_text("0-3\n")
    location = CgroupV2Location(
        mount_point=mount,
        mount_root="/",
        membership_path="/child",
        path=child,
    )
    monkeypatch.setattr(current.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})

    allocation = current.read_allocation_state(location)

    assert allocation.cpu_max == "max 100000"
    assert allocation.quota_cores == 2.0
    assert [item["value"] for item in allocation.cpu_max_hierarchy] == [
        "max 100000",
        "200000 100000",
    ]


def test_allocation_metadata_read_errors_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    location = _fake_cgroup(tmp_path)
    target = location.path / "cpu.max"
    original_read_text = Path.read_text

    def fail_target(path: Path, *args, **kwargs):
        if path == target:
            raise PermissionError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_target)

    with pytest.raises(ProfileError, match="allocation metadata"):
        current.read_allocation_state(location)


def test_allocation_affinity_read_errors_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    location = _fake_cgroup(tmp_path)

    def fail_affinity(_pid: int) -> set[int]:
        raise PermissionError("denied")

    monkeypatch.setattr(current.os, "sched_getaffinity", fail_affinity)

    with pytest.raises(ProfileError, match="CPU affinity"):
        current.read_allocation_state(location)


def test_descendant_process_read_errors_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    location = _fake_cgroup(tmp_path)
    child = location.path / "child"
    child.mkdir()
    target = child / "cgroup.procs"
    target.write_text("123\n")
    original_read_text = Path.read_text

    def fail_target(path: Path, *args, **kwargs):
        if path == target:
            raise PermissionError("denied")
        return original_read_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", fail_target)

    with pytest.raises(ProfileError, match="descendant cgroup"):
        current._cgroup_pids(location.path)


def test_descendant_process_enumeration_errors_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    location = _fake_cgroup(tmp_path)

    def fail_walk(*_args, **_kwargs):
        raise PermissionError("denied")

    monkeypatch.setattr(current.os, "walk", fail_walk)

    with pytest.raises(ProfileError, match="enumerate descendant cgroups"):
        current._cgroup_pids(location.path)


def test_invalid_cgroup_process_ids_fail_closed(tmp_path: Path) -> None:
    location = _fake_cgroup(tmp_path)
    (location.path / "cgroup.procs").write_text("not-a-pid\n")

    with pytest.raises(ProfileError, match="invalid process ID"):
        current._cgroup_pids(location.path)


def test_timed_cpu_stat_uses_read_midpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "cpu.stat"
    path.write_text("usage_usec 10\n")
    times = iter((100, 200))
    monkeypatch.setattr(current.time, "monotonic_ns", lambda: next(times))

    raw, stats, midpoint_ns, read_duration_ns = current._read_timed_cpu_stat(path)

    assert raw == "usage_usec 10\n"
    assert stats == {"usage_usec": 10}
    assert midpoint_ns == 150
    assert read_duration_ns == 100


@pytest.mark.parametrize(
    ("cpuset_ids", "affinity_ids", "expected_source"),
    [
        ((0, 1), (), "cpuset"),
        ((), (0, 1), "affinity"),
        ((0, 1), (0, 1), "cpuset+affinity"),
    ],
)
def test_capacity_names_only_observed_cpu_sources(
    cpuset_ids: tuple[int, ...],
    affinity_ids: tuple[int, ...],
    expected_source: str,
) -> None:
    allocation = current.AllocationState(
        cpu_max=None,
        cpu_max_hierarchy=(),
        quota_cores=None,
        cpuset=None,
        cpuset_ids=cpuset_ids,
        affinity_ids=affinity_ids,
    )

    capacity, source, effective_ids = current._capacity(allocation, None)

    assert capacity == 2.0
    assert source == expected_source
    assert effective_ids == (0, 1)


def test_run_current_profile_calculates_cgroup_cpu_and_throttling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _fake_cgroup(tmp_path)
    monkeypatch.setattr(current, "resolve_current_cgroup", lambda: location)
    monkeypatch.setattr(current.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    times = iter(
        (
            1_000_000_000,
            1_000_000_000,
            3_000_000_000,
            3_000_000_000,
        )
    )
    monkeypatch.setattr(current.time, "monotonic_ns", lambda: next(times))

    def fake_run(command, **kwargs):
        (location.path / "cpu.stat").write_text(
            "usage_usec 3000000\n"
            "user_usec 2100000\n"
            "system_usec 900000\n"
            "nr_periods 30\n"
            "nr_throttled 6\n"
            "throttled_usec 1600000\n"
        )
        return subprocess.CompletedProcess(command, 0, "replayed\n", "")

    monkeypatch.setattr(current.subprocess, "run", fake_run)

    report = run_current_profile(_config(tmp_path))

    assert report["runtime"] == "current"
    assert report["scope"] == "current-container"
    assert report["cpu"]["usage_sec"] == 2.0
    assert report["cpu"]["average_logical_cores_used"] == 1.0
    assert report["cpu"]["allocated_logical_cpus"] == 1.0
    assert report["cpu"]["allocated_cpu_source"] == "cgroup-v2-cpu.max+cpuset+affinity"
    assert report["cpu"]["utilization_percent_of_allocation"] == 100.0
    assert report["cpu"]["throttling"] == {
        "periods": 20,
        "throttled_periods": 5,
        "throttled_sec": 1.5,
        "burst_count": 0,
        "burst_sec": 0.0,
    }
    assert report["cpu"]["snapshot_read_duration_ns"] == {
        "before": 0,
        "after": 0,
    }
    assert report["measurement_validity"]["valid_for_average_cpu_reporting"] is True
    assert (tmp_path / "profile" / "cpu.stat.before").is_file()
    assert (tmp_path / "profile" / "cpu.stat.after").is_file()
    assert (tmp_path / "profile" / "current_environment.json").is_file()


def test_run_current_profile_flags_other_processes_in_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    location = _fake_cgroup(tmp_path)
    unrelated_pid = 999999
    (location.path / "cgroup.procs").write_text(f"{os.getpid()}\n{unrelated_pid}\n")
    monkeypatch.setattr(current, "resolve_current_cgroup", lambda: location)
    monkeypatch.setattr(current.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    times = iter(
        (
            1_000_000_000,
            1_000_000_000,
            2_000_000_000,
            2_000_000_000,
        )
    )
    monkeypatch.setattr(current.time, "monotonic_ns", lambda: next(times))

    def fake_run(command, **kwargs):
        (location.path / "cpu.stat").write_text(
            "usage_usec 2000000\n"
            "user_usec 1200000\n"
            "system_usec 800000\n"
            "nr_periods 20\n"
            "nr_throttled 1\n"
            "throttled_usec 100000\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(current.subprocess, "run", fake_run)

    report = run_current_profile(_config(tmp_path))

    validity = report["measurement_validity"]
    assert validity["valid_for_average_cpu_reporting"] is False
    assert validity["potential_unrelated_processes_before"] == [{"pid": unrelated_pid, "name": None}]
    assert "Potential unrelated processes" in report["warning"]


def test_surviving_processes_are_scanned_before_artifact_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _fake_cgroup(tmp_path)
    unrelated_pid = 999999
    profile_dir = tmp_path / "profile"
    monkeypatch.setattr(current, "resolve_current_cgroup", lambda: location)
    monkeypatch.setattr(current.os, "sched_getaffinity", lambda _pid: {0})
    times = iter(
        (
            1_000_000_000,
            1_000_000_000,
            2_000_000_000,
            2_000_000_000,
        )
    )
    monkeypatch.setattr(current.time, "monotonic_ns", lambda: next(times))

    def fake_run(command, **kwargs):
        (location.path / "cpu.stat").write_text(
            "usage_usec 2000000\n"
            "user_usec 1200000\n"
            "system_usec 800000\n"
            "nr_periods 20\n"
            "nr_throttled 1\n"
            "throttled_usec 100000\n"
        )
        (location.path / "cgroup.procs").write_text(f"{os.getpid()}\n{unrelated_pid}\n")
        return subprocess.CompletedProcess(command, 0, "", "")

    original_write_text = Path.write_text

    def remove_survivor_after_first_artifact(path: Path, data: str, *args, **kwargs) -> int:
        written = original_write_text(path, data, *args, **kwargs)
        if path.parent == profile_dir and path.name == "cpu.stat.before":
            original_write_text(
                location.path / "cgroup.procs",
                f"{os.getpid()}\n",
            )
        return written

    monkeypatch.setattr(current.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "write_text", remove_survivor_after_first_artifact)

    report = run_current_profile(_config(tmp_path))

    assert report["measurement_validity"]["valid_for_average_cpu_reporting"] is False
    assert report["measurement_validity"]["processes_outside_controller_ancestry_after"] == [
        {"pid": unrelated_pid, "name": None}
    ]


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"cpu_limit": 1.0}, "cannot enforce --cpu-limit"),
        ({"scheduler_bcc_buffer_pages": 4096}, "scheduler-bcc-buffer-pages"),
        ({"terminate_descendants_at_finish": True}, "descendant termination"),
        ({"network": "none"}, "network"),
    ],
)
def test_run_current_profile_rejects_unsupported_runtime_options(
    tmp_path: Path, overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(ProfileError, match=message):
        run_current_profile(_config(tmp_path, **overrides))


def test_prepare_and_measured_command_use_sanitized_task_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _fake_cgroup(tmp_path)
    monkeypatch.setenv("VIRTUAL_ENV", "/controller")
    monkeypatch.setenv("PATH", "/controller/bin:/usr/bin:/bin")
    monkeypatch.setattr(current, "resolve_current_cgroup", lambda: location)
    monkeypatch.setattr(current.os, "sched_getaffinity", lambda _pid: {0})
    times = iter(
        (
            1_000_000_000,
            1_000_000_000,
            2_000_000_000,
            2_000_000_000,
        )
    )
    monkeypatch.setattr(current.time, "monotonic_ns", lambda: next(times))
    environments: list[dict[str, str]] = []

    def fake_run(command, **kwargs):
        environments.append(kwargs["env"])
        if len(environments) == 2:
            (location.path / "cpu.stat").write_text(
                "usage_usec 2000000\n"
                "user_usec 1200000\n"
                "system_usec 800000\n"
                "nr_periods 20\n"
                "nr_throttled 1\n"
                "throttled_usec 100000\n"
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(current.subprocess, "run", fake_run)

    report = run_current_profile(
        _config(
            tmp_path,
            prepare_command="true",
            env=("PATH=/usr/bin:/bin",),
            unset_env=("VIRTUAL_ENV",),
        )
    )

    assert len(environments) == 2
    assert all(environment.get("VIRTUAL_ENV") is None for environment in environments)
    assert all(environment["PATH"] == "/usr/bin:/bin" for environment in environments)
    assert report["task_environment"] == {
        "unset_variables": ["VIRTUAL_ENV"],
        "overridden_variables": ["PATH"],
    }


def test_current_profile_is_diagnostic_without_exclusivity_assertion(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    location = _fake_cgroup(tmp_path)
    monkeypatch.setattr(current, "resolve_current_cgroup", lambda: location)
    monkeypatch.setattr(current.os, "sched_getaffinity", lambda _pid: {0})
    times = iter(
        (
            1_000_000_000,
            1_000_000_000,
            2_000_000_000,
            2_000_000_000,
        )
    )
    monkeypatch.setattr(current.time, "monotonic_ns", lambda: next(times))

    def fake_run(command, **kwargs):
        (location.path / "cpu.stat").write_text(
            "usage_usec 2000000\n"
            "user_usec 1200000\n"
            "system_usec 800000\n"
            "nr_periods 20\n"
            "nr_throttled 1\n"
            "throttled_usec 100000\n"
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(current.subprocess, "run", fake_run)

    report = run_current_profile(_config(tmp_path, current_container_exclusive=False))

    assert report["measurement_validity"]["valid_for_average_cpu_reporting"] is False
    assert "exclusivity was not asserted" in report["warning"]
