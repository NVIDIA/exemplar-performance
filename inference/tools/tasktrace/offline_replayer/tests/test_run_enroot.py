# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest
from offline_replayer.measurement.enroot_network import BACKEND, SCHEMA_VERSION
from offline_replayer.measurement.network_policy import NetworkPolicy
from offline_replayer.runs import enroot as enroot_module
from offline_replayer.runs.enroot import (
    ENROOT_TASK_CLOCK,
    _plan_digest,
    invalidate_changed_images,
    invalidate_duration_bounds,
    project,
    runtime_provenance,
    wave_overall,
)
from offline_replayer.runs.spec import RunDefaultsSpec, RunLaunchSpec, RunTaskSpec

CONTROLLER_SOURCE_SHA256 = "c" * 64
RUNTIME_PROVENANCE = {"comparability_sha256": "e" * 64}


def _projection_inputs(tmp_path: Path) -> tuple[dict, RunTaskSpec, RunDefaultsSpec, RunLaunchSpec]:
    image = tmp_path / "task.sqsh"
    image.write_bytes(b"squashfs")
    stat = image.stat()
    source_sha256 = "a" * 64
    task = RunTaskSpec(
        task_id="task",
        atif=tmp_path / "trajectory.json",
        source_sha256=source_sha256,
        image=str(image),
        expected_image_id=None,
        action_timeout_sec=None,
    )
    defaults = RunDefaultsSpec(
        pacing="llm-gaps",
        network_policy="no-network",
        allowed_hosts=(),
        profile_sets=(),
        controller_runtime="host-mount",
        controller_cpus="0",
        controller_cpu_ids=(0,),
        action_timeout_sec=900,
        runtime="enroot",
    )
    launch = RunLaunchSpec(task_id="task", cpus="4", cpu_ids=(4,))
    parent_netns_inode = Path("/proc/self/ns/net").stat().st_ino
    report = {
        "status": "valid",
        "source_trajectory_sha256": source_sha256,
        "image": str(image),
        "image_identity": {
            "runtime": "enroot",
            "path": str(image.resolve()),
            "size_bytes": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
        },
        "pacing": "llm-gaps",
        "controller_runtime": "host-mount",
        "profile_provenance": {"profile_sets": []},
        "network_policy": {
            "comparability_sha256": NetworkPolicy("no-network").comparability_digest,
            "enforcement": {
                "schema_version": SCHEMA_VERSION,
                "state": "enforced",
                "backend": BACKEND,
                "policy_effective_mode": "no-network",
                "valid": True,
                "parent_netns_inode": parent_netns_inode,
                "netns_inode": parent_netns_inode + 1,
                "netns_differs": True,
                "controller_euid": os.geteuid(),
                "controller_egid": os.getegid(),
                "namespace_euid": os.geteuid(),
                "namespace_egid": os.getegid(),
                "interfaces": ["lo"],
                "ipv4_routes": [],
                "ipv6_addresses": [],
                "publication": {"source": "controller-private-temporary-state", "published_after_enroot_exit": True},
            },
        },
        "container_cpu": {
            "wall_sec": 2.0,
            "method": "wall-clock-only-no-cgroup",
            "wall_clock_source": ENROOT_TASK_CLOCK,
            "allocated_cpu_source": "taskset-requested-affinity",
            "allocated_logical_cpus": 1.0,
            "requested_cpu_affinity": "4",
            "initial_effective_cpu_affinity": "4",
            "initial_effective_logical_cpu_ids": [4],
            "window_start_ns": 1_000_000_000,
            "window_end_ns": 3_000_000_000,
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "source": ENROOT_TASK_CLOCK,
                "resolution_ns": 10_000_000,
                "boot_id": "boot-id",
                "time_namespace": "time:[1]",
            },
        },
        "enroot_lifecycle": {
            "process_containment": {
                "backend": "linux-user-pid-namespace-v1",
                "namespace_init_exit_kills_remaining_processes": True,
                "outer_exit_kills_namespace_init": True,
            },
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "boot_id": "boot-id",
                "time_namespace": "time:[1]",
            },
            "phases": [
                {"phase": "create", "start_ns": 0, "end_ns": 500_000_000},
                {"phase": "start", "start_ns": 500_000_000, "end_ns": 3_500_000_000},
                {"phase": "remove", "start_ns": 3_500_000_000, "end_ns": 3_700_000_000},
                {"phase": "list", "start_ns": 3_700_000_000, "end_ns": 3_800_000_000},
            ],
            "cleanup": {"valid": True, "container_absent": True},
        },
        "replay_summary": {"actions_timed_out": 0, "source_outcome_mismatch_count": 0},
    }
    return report, task, defaults, launch


def test_enroot_projection_and_wave_summary_are_wall_only(tmp_path: Path) -> None:
    report, task, defaults, launch = _projection_inputs(tmp_path)

    result = project(
        report,
        task=task,
        defaults=defaults,
        launch=launch,
        preflight_image_id="sha256:" + "b" * 64,
        controller_source_sha256=CONTROLLER_SOURCE_SHA256,
        runtime_provenance=RUNTIME_PROVENANCE,
    )
    overall = wave_overall(
        [
            {
                "attempt_index": 0,
                "status": "completed",
                "duration_valid": result.valid,
                "duration_sec": result.duration_sec,
                "window_start_ns": result.window_start_ns,
                "window_end_ns": result.window_end_ns,
                "clock_reference": result.clock_reference,
            }
        ],
        interrupted=False,
        outer_wall_sec=4.0,
        expected_attempt_indexes={0},
    )

    assert result.valid and result.duration_sec == 2.0
    assert result.cleanup_valid is True
    assert result.window_start_ns == 1_000_000_000
    assert result.window_end_ns == 3_000_000_000
    assert result.clock_reference == report["container_cpu"]["clock_reference"]
    assert result.cpu_usage_sec is result.average_logical_cores is None
    assert overall["wall_only_throughput_tasks_per_sec"] == 0.5
    assert overall["lifecycle_inclusive_throughput_tasks_per_sec"] == 0.25
    assert overall["total_cgroup_cpu_sec"] is None
    assert overall["pool_utilization_percent"] is None
    report["network_policy"]["enforcement"]["interfaces"] = ["eth0"]
    assert not project(
        report,
        task=task,
        defaults=defaults,
        launch=launch,
        preflight_image_id="sha256:" + "b" * 64,
        controller_source_sha256=CONTROLLER_SOURCE_SHA256,
        runtime_provenance=RUNTIME_PROVENANCE,
    ).valid


def test_runtime_provenance_is_contextual_secret_safe_and_content_bound(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bin_dir = tmp_path / "bin"
    library = tmp_path / "lib"
    system = tmp_path / "system"
    user = tmp_path / "user"
    states = {name: tmp_path / name for name in ("cache", "data", "run", "temp")}
    for directory in (bin_dir, library, system, *states.values()):
        directory.mkdir()
    enroot = bin_dir / "enroot"
    taskset = bin_dir / "taskset"
    unshare = bin_dir / "unshare"
    unsquashfs = bin_dir / "unsquashfs"
    enroot.write_text("enroot executable")
    taskset.write_text("taskset executable")
    unshare.write_text("unshare executable")
    unsquashfs.write_text("unsquashfs executable")
    (bin_dir / "enroot-nsenter").write_text("nsenter executable")
    runtime = library / "runtime.sh"
    runtime.write_text("runtime v1")
    (system / "enroot.conf").write_text("ENROOT_REMAP_ROOT=y\n")

    def which(name: str) -> str | None:
        return {
            "enroot": str(enroot),
            "taskset": str(taskset),
            "unshare": str(unshare),
            "unsquashfs": str(unsquashfs),
        }.get(name)

    def run(command, **_kwargs):
        cpus, action = command[2], command[-1]
        if action == "version":
            output = "4.0.1\n"
        else:
            output = "\n".join(
                (
                    f"  ENROOT_LIBRARY_PATH={library}",
                    f"  ENROOT_SYSCONF_PATH={system}",
                    f"  ENROOT_CONFIG_PATH={user}",
                    f"  ENROOT_CACHE_PATH={states['cache']}",
                    f"  ENROOT_DATA_PATH={states['data']}",
                    f"  ENROOT_RUNTIME_PATH={states['run']}",
                    f"  ENROOT_TEMP_PATH={states['temp']}",
                    f"  ENROOT_MAX_PROCESSORS={4 if cpus == '0-3' else 1}",
                    "  ENROOT_SECRET=must-not-be-published",
                )
            )
        return subprocess.CompletedProcess(command, 0, output, "")

    monkeypatch.setattr(enroot_module.shutil, "which", which)
    monkeypatch.setattr(enroot_module.subprocess, "run", run)
    first = runtime_provenance(controller_cpus="0-3", launch_cpu_sets=("4", "5"))
    second = runtime_provenance(controller_cpus="0-3", launch_cpu_sets=("4", "5"))

    assert first == second
    assert first["effective_config_sha256"]["controller"]["sha256"] != first["effective_config_sha256"]["launches"]["4"]
    assert "must-not-be-published" not in json.dumps(first)
    runtime.write_text("runtime v2")
    changed = runtime_provenance(controller_cpus="0-3", launch_cpu_sets=("4", "5"))
    assert changed["comparability_sha256"] != first["comparability_sha256"]


def test_enroot_projection_rejects_non_monotonic_wall_clock(tmp_path: Path) -> None:
    report, task, defaults, launch = _projection_inputs(tmp_path)
    report["container_cpu"]["wall_clock_source"] = "clock-realtime"

    result = project(
        report,
        task=task,
        defaults=defaults,
        launch=launch,
        preflight_image_id="sha256:" + "b" * 64,
        controller_source_sha256=CONTROLLER_SOURCE_SHA256,
        runtime_provenance=RUNTIME_PROVENANCE,
    )

    assert result.valid is False
    assert "not from one monotonic clock" in result.reason


@pytest.mark.parametrize(
    ("mutation", "reason"),
    [
        (
            lambda report: report["enroot_lifecycle"]["cleanup"].update({"valid": False}),
            "cleanup evidence is invalid",
        ),
        (
            lambda report: report["container_cpu"].update({"allocated_cpu_source": "cpuset-cpus"}),
            "requested task affinity evidence is missing",
        ),
        (
            lambda report: report["container_cpu"].update({"initial_effective_logical_cpu_ids": [5]}),
            "initial effective task affinity differs",
        ),
        (
            lambda report: report["enroot_lifecycle"].pop("process_containment"),
            "PID-namespace containment evidence is invalid",
        ),
        (
            lambda report: report["container_cpu"].update({"window_end_ns": 4_000_000_000}),
            "wall duration disagrees with raw task timestamps",
        ),
        (
            lambda report: report["container_cpu"]["clock_reference"].update({"time_namespace": "time:[other]"}),
            "lack an identity mapping",
        ),
        (
            lambda report: (
                report["container_cpu"]["clock_reference"].update({"boot_id": None}),
                report["enroot_lifecycle"]["clock_reference"].update({"boot_id": None}),
            ),
            "clock reference is invalid",
        ),
        (
            lambda report: (
                report["container_cpu"]["clock_reference"].update({"time_namespace": None}),
                report["enroot_lifecycle"]["clock_reference"].update({"time_namespace": None}),
            ),
            "clock reference is invalid",
        ),
        (
            lambda report: report["enroot_lifecycle"]["phases"][1].update({"start_ns": 2_000_000_000}),
            "falls outside the start lifecycle phase",
        ),
    ],
)
def test_enroot_projection_rejects_cleanup_and_clock_boundary_drift(tmp_path: Path, mutation, reason: str) -> None:
    report, task, defaults, launch = _projection_inputs(tmp_path)
    mutation(report)

    result = project(
        report,
        task=task,
        defaults=defaults,
        launch=launch,
        preflight_image_id="sha256:" + "b" * 64,
        controller_source_sha256=CONTROLLER_SOURCE_SHA256,
        runtime_provenance=RUNTIME_PROVENANCE,
    )

    assert result.valid is False
    assert reason in result.reason


def test_enroot_wave_requires_achieved_all_launch_overlap() -> None:
    rows = [
        {
            "attempt_index": 1,
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 1.0,
            "window_start_ns": 1_000_000_000,
            "window_end_ns": 2_000_000_000,
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "boot_id": "boot-id",
                "time_namespace": "time:[1]",
            },
        },
        {
            "attempt_index": 2,
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 1.0,
            "window_start_ns": 2_000_000_000,
            "window_end_ns": 3_000_000_000,
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "boot_id": "boot-id",
                "time_namespace": "time:[1]",
            },
        },
    ]

    overall = wave_overall(
        rows,
        interrupted=False,
        outer_wall_sec=4.0,
        expected_attempt_indexes={1, 2},
        require_full_overlap=True,
    )

    assert overall["valid_for_throughput"] is False
    assert overall["throughput_tasks_per_sec"] is None
    assert overall["lifecycle_inclusive_throughput_tasks_per_sec"] is None
    assert overall["observed_lifecycle_inclusive_throughput_tasks_per_sec"] == 0.5
    assert "replay commands did not achieve all-launch overlap" in "; ".join(overall["throughput_invalid_reasons"])


def test_enroot_wave_reports_positive_overlap_and_separate_boundaries() -> None:
    rows = [
        {
            "attempt_index": 0,
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 3.0,
            "window_start_ns": 1_000_000_000,
            "window_end_ns": 4_000_000_000,
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "boot_id": "boot-id",
                "time_namespace": "time:[1]",
            },
        },
        {
            "attempt_index": 1,
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 3.0,
            "window_start_ns": 2_000_000_000,
            "window_end_ns": 5_000_000_000,
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "boot_id": "boot-id",
                "time_namespace": "time:[1]",
            },
        },
    ]

    overall = wave_overall(
        rows,
        interrupted=False,
        outer_wall_sec=10.0,
        expected_attempt_indexes={0, 1},
        require_full_overlap=True,
    )

    assert overall["valid_for_throughput"] is True
    assert overall["replay_command_batch_wall_sec"] == 4.0
    assert overall["throughput_tasks_per_sec"] == 0.5
    assert overall["lifecycle_inclusive_wall_sec"] == 10.0
    assert overall["lifecycle_inclusive_throughput_tasks_per_sec"] == 0.2
    assert overall["achieved_peak_measured_attempts"] == 2
    assert overall["start_spread_sec"] == 1.0
    assert overall["all_launch_overlap_sec"] == 2.0
    assert overall["time_sec_by_measured_attempts"] == {"1": 2.0, "2": 2.0}


def test_enroot_wave_requires_one_common_monotonic_clock_identity() -> None:
    rows = [
        {
            "attempt_index": index,
            "status": "completed",
            "duration_valid": True,
            "duration_sec": 2.0,
            "window_start_ns": 1_000_000_000,
            "window_end_ns": 3_000_000_000,
            "clock_reference": {
                "clock_name": "CLOCK_BOOTTIME",
                "boot_id": f"boot-{index}",
                "time_namespace": "time:[1]",
            },
        }
        for index in (1, 2)
    ]

    overall = wave_overall(
        rows,
        interrupted=False,
        outer_wall_sec=4.0,
        expected_attempt_indexes={1, 2},
        require_full_overlap=True,
    )

    assert overall["valid_for_throughput"] is False
    assert overall["throughput_tasks_per_sec"] is None
    assert "replay windows do not share one monotonic clock identity" in (overall["throughput_invalid_reasons"])


def test_changed_enroot_image_is_hashed_once_for_shared_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    image = tmp_path / "shared.sqsh"
    image.write_bytes(b"shared image")
    current_image_id = "sha256:" + hashlib.sha256(b"shared image").hexdigest()
    tasks = [
        RunTaskSpec(
            task_id=task_id,
            atif=tmp_path / f"{task_id}.json",
            source_sha256=character * 64,
            image=str(image),
            expected_image_id=None,
            action_timeout_sec=None,
        )
        for task_id, character in (("current", "a"), ("stale", "b"))
    ]
    rows = [
        {
            "task_id": "current",
            "image_id": current_image_id,
            "duration_valid": True,
            "reason": "",
        },
        {
            "task_id": "stale",
            "image_id": "sha256:" + "0" * 64,
            "duration_valid": True,
            "reason": "first reason",
        },
        {
            "task_id": "stale",
            "image_id": "sha256:" + "0" * 64,
            "duration_valid": True,
            "reason": "",
        },
    ]
    original_open = Path.open
    opens = 0

    def counted_open(path: Path, *args, **kwargs):
        nonlocal opens
        if path == image.resolve():
            opens += 1
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", counted_open)

    invalidate_changed_images(tasks, rows)

    assert opens == 1
    assert rows[0]["duration_valid"] is True
    assert rows[0]["reason"] == ""
    assert all(row["duration_valid"] is False for row in rows[1:])
    assert all("Enroot image content changed after preflight" in row["reason"] for row in rows[1:])
    assert rows[1]["reason"].startswith("first reason; ")


def test_duration_larger_than_outer_wave_is_rejected() -> None:
    rows = [
        {
            "duration_valid": True,
            "duration_sec": 2.1,
            "reason": "existing evidence",
        }
    ]

    invalidate_duration_bounds(rows, outer_wall_sec=2.0)

    assert rows[0]["duration_valid"] is False
    assert rows[0]["reason"] == ("existing evidence; task duration exceeds the lifecycle-inclusive outer wave bound")


def test_execution_plan_digest_covers_enroot_comparison_identity(
    tmp_path: Path,
) -> None:
    _report, task, defaults, launch = _projection_inputs(tmp_path)
    image = "sha256:" + "b" * 64

    def digest(
        task,
        defaults,
        launch,
        image,
        controller=CONTROLLER_SOURCE_SHA256,
        runtime="e" * 64,
    ):
        return _plan_digest(task, defaults, launch, image, controller, runtime)

    baseline = digest(task, defaults, launch, image)
    variant_digests = [
        digest(
            task,
            replace(defaults, controller_cpus="1", controller_cpu_ids=(1,)),
            launch,
            image,
        ),
        digest(
            task,
            replace(defaults, network_policy="public"),
            launch,
            image,
        ),
        digest(
            task,
            replace(
                defaults,
                network_policy="allowlist",
                allowed_hosts=("example.com",),
            ),
            launch,
            image,
        ),
        digest(
            task,
            replace(defaults, pacing="unpaced"),
            launch,
            image,
        ),
        digest(
            task,
            replace(defaults, action_timeout_sec=901.0),
            launch,
            image,
        ),
        digest(replace(task, action_timeout_sec=902.0), defaults, launch, image),
        digest(replace(task, source_sha256="c" * 64), defaults, launch, image),
        digest(
            task,
            defaults,
            replace(launch, cpus="5", cpu_ids=(5,)),
            image,
        ),
        digest(task, defaults, launch, "sha256:" + "d" * 64),
        digest(task, defaults, launch, image, "d" * 64),
        digest(
            task,
            defaults,
            launch,
            image,
            CONTROLLER_SOURCE_SHA256,
            "f" * 64,
        ),
    ]

    assert baseline not in variant_digests
    assert len(set(variant_digests)) == len(variant_digests)
