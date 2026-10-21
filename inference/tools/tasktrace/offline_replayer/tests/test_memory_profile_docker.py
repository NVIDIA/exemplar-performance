# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-Docker validation for cgroup memory profiling.

Run with, for example:

    TASKTRACE_DOCKER_TEST_IMAGE=python:3.12-slim \
      uv run --extra test python -m pytest -q \
      offline_replayer/tests/test_memory_profile_docker.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from offline_replayer.measurement.container import ProfileConfig, run_profile

IMAGE = os.environ.get("TASKTRACE_DOCKER_TEST_IMAGE")
pytestmark = pytest.mark.skipif(
    not IMAGE,
    reason="set TASKTRACE_DOCKER_TEST_IMAGE to run real-Docker memory tests",
)
MIB = 1024 * 1024


def _config(tmp_path: Path, command: str, **overrides: object) -> ProfileConfig:
    values: dict[str, object] = {
        "command": command,
        "prepare_command": None,
        "output_dir": tmp_path,
        "image": IMAGE,
        "container": None,
        "cpuset_cpus": None,
        "cpu_limit": None,
        "allocated_cpus": 1.0,
        "workdir": "/tmp",
        "user": None,
        "env": (),
        "mounts": (),
        "strace_mode": "none",
        "string_limit": 4096,
        "shell": "/bin/bash",
        "memory_profile": True,
        "memory_sample_interval_sec": 0.05,
        "docker": "docker",
    }
    values.update(overrides)
    return ProfileConfig(**values)  # type: ignore[arg-type]


def test_steady_anonymous_allocation_is_visible_in_peak_and_timeline(
    tmp_path: Path,
) -> None:
    report = run_profile(
        _config(
            tmp_path / "steady",
            "python -c 'import time; x=bytearray(64*1024*1024); " "x[::4096]=b\"x\"*(len(x)//4096); time.sleep(0.5)'",
        )
    )

    memory = report["memory"]
    assert memory["peak_valid"] is True
    assert memory["natural_peak_trustworthy"] is True
    assert memory["observation_boundary"] == "controlled-finish-gate"
    assert memory["observation_boundary_complete"] is True
    assert memory["lifetime"]["exact_through_controlled_finish_gate"] is True
    assert memory["lifetime"]["exact_for_fresh_cgroup_lifetime"] is False
    assert memory["lifetime"]["peak_bytes"] >= 56 * MIB
    assert memory["command_window"]["sampled_peak_bytes"] >= 56 * MIB
    assert memory["command_window"]["memory_stat_at_sampled_peak"]["anon"] >= 48 * MIB
    assert memory["command_window"]["sample_count"] >= 5


def test_kernel_peak_retains_a_sub_interval_memory_spike(tmp_path: Path) -> None:
    report = run_profile(
        _config(
            tmp_path / "spike",
            "python -c 'import time; x=bytearray(96*1024*1024); "
            "x[::4096]=b\"x\"*(len(x)//4096); del x; time.sleep(0.05)'",
            memory_sample_interval_sec=0.5,
        )
    )

    memory = report["memory"]
    assert memory["lifetime"]["peak_bytes"] >= 80 * MIB
    assert memory["lifetime"]["peak_bytes"] > memory["command_window"]["sampled_peak_bytes"]


def test_descendant_allocation_is_included_by_cgroup_scope(tmp_path: Path) -> None:
    report = run_profile(
        _config(
            tmp_path / "descendant",
            "python -c 'import os,time; p=os.fork(); "
            "(os.waitpid(p,0) if p else (lambda x: (x.__setitem__(slice(None,None,4096), "
            "b\"x\"*(len(x)//4096)), time.sleep(0.4), os._exit(0)))(bytearray(48*1024*1024)))'",
        )
    )

    assert report["memory"]["lifetime"]["peak_bytes"] >= 40 * MIB
    assert report["memory"]["remaining_descendant_pids_at_finish_gate"] == []


def test_page_cache_is_counted_and_not_mislabeled_as_rss(tmp_path: Path) -> None:
    report = run_profile(
        _config(
            tmp_path / "page-cache",
            "python -c 'import os,time; f=open(\"cache.bin\",\"wb\"); "
            "chunk=b\"x\"*(1024*1024); [f.write(chunk) for _ in range(64)]; "
            "f.flush(); os.fsync(f.fileno()); time.sleep(0.4)'",
        )
    )

    peak_stat = report["memory"]["command_window"]["memory_stat_at_sampled_peak"]
    assert peak_stat["file"] >= 32 * MIB
    assert "not process RSS" in report["memory"]["scope"]


def test_memory_limit_events_are_retained_as_censored_evidence(
    tmp_path: Path,
) -> None:
    report = run_profile(
        _config(
            tmp_path / "oom",
            "python -c 'x=bytearray(256*1024*1024); x[::4096]=b\"x\"*(len(x)//4096)'",
            memory_limit_bytes=96 * MIB,
            memory_swap_total_limit_bytes=96 * MIB,
        )
    )

    memory = report["memory"]
    assert report["command_exit_code"] != 0
    assert memory["censoring"] == "oom"
    assert memory["events_delta"]["oom"] > 0
    assert memory["instrumentation_valid"] is True
    assert memory["timeline_valid_for_shape"] is True
    assert isinstance(memory["lifetime"]["peak_bytes"], int)
    assert memory["peak_valid"] is True
    assert memory["natural_peak_trustworthy"] is False


def test_prepare_failure_still_writes_memory_evidence(tmp_path: Path) -> None:
    report = run_profile(
        _config(
            tmp_path / "prepare-failure",
            "true",
            prepare_command="exit 42",
        )
    )

    assert report["command_exit_code"] == 42
    assert report["memory"]["censoring"] == "functional_failure"
    assert report["memory"]["sample_count"] >= 2
    assert report["memory"]["natural_peak_trustworthy"] is False
    assert (tmp_path / "prepare-failure" / "profile_report.json").is_file()


def test_low_privilege_cpu_window_remains_foreground_command_only(
    tmp_path: Path,
) -> None:
    command = (
        "python -c 'import subprocess,time; "
        "subprocess.Popen([\"python\",\"-c\","
        "\"import time; e=time.monotonic()+0.6; "
        "exec(\\\"while time.monotonic() < e: pass\\\")\"]); "
        "e=time.monotonic()+0.2; "
        "exec(\"while time.monotonic() < e: pass\")'"
    )
    report = run_profile(_config(tmp_path / "drain-window", command))

    wall_sec = report["cpu"]["wall_sec"]
    usage_sec = report["cpu"]["usage_sec"]
    command_sampled_cpu = report["memory"]["command_window"]["sampled_cpu_usage_sec"]
    observation_sampled_cpu = report["memory"]["observation_window"]["sampled_cpu_usage_sec"]
    assert 0.15 <= wall_sec < 0.5
    assert usage_sec >= 0.15
    assert abs(command_sampled_cpu - usage_sec) < 0.05
    assert observation_sampled_cpu - command_sampled_cpu >= 0.15
    assert report["memory"]["remaining_descendant_pids_at_finish_gate"] == []


def test_successful_finite_limit_is_reported_without_recommendation_policy(
    tmp_path: Path,
) -> None:
    limit = 256 * MIB
    report = run_profile(
        _config(
            tmp_path / "finite-limit",
            "python -c 'x=bytearray(8*1024*1024); x[::4096]=b\"x\"*(len(x)//4096)'",
            memory_limit_bytes=limit,
            memory_swap_total_limit_bytes=limit,
        )
    )

    memory = report["memory"]
    assert memory["limits"]["memory_max_bytes"] == limit
    assert memory["lifetime"]["peak_bytes"] < limit
    assert memory["censoring"] == "none"
    assert memory["peak_valid"] is True
    assert memory["natural_peak_trustworthy"] is True
    margin = memory["observed_limit_margin"]
    assert margin["configured_limit_state"] == "finite"
    assert margin["configured_limit_bytes"] == limit
    assert margin["observed_peak_bytes"] == memory["lifetime"]["peak_bytes"]
    assert margin["remaining_to_limit_bytes"] == (limit - memory["lifetime"]["peak_bytes"])
    assert margin["peak_fraction_of_limit"] == pytest.approx(memory["lifetime"]["peak_bytes"] / limit)
