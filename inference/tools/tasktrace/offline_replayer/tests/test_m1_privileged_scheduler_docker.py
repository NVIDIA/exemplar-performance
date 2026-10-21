# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real privileged BCC/cgroup scheduler smoke test.

Run this test from a root-equivalent host context with BCC available, for
example the documented transient privileged chroot sidecar::

    TASKTRACE_M1_PRIVILEGED_SCHEDULER=1 \
    TASKTRACE_M1_DOCKER_TEST_IMAGE=<immutable-image-id> \
    TASKTRACE_M1_CPUSET=16-19 \
      pytest -q offline_replayer/tests/test_m1_privileged_scheduler_docker.py
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from offline_replayer.measurement.container import ProfileConfig, run_profile
from offline_replayer.measurement.evidence import read_bundle, validate_raw_artifacts

IMAGE = os.environ.get("TASKTRACE_M1_DOCKER_TEST_IMAGE")
CPUSET = os.environ.get("TASKTRACE_M1_CPUSET")
ENABLED = os.environ.get("TASKTRACE_M1_PRIVILEGED_SCHEDULER") == "1"
SOURCE_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(
    not ENABLED or not IMAGE or not CPUSET,
    reason=("set TASKTRACE_M1_PRIVILEGED_SCHEDULER=1, " "TASKTRACE_M1_DOCKER_TEST_IMAGE, and TASKTRACE_M1_CPUSET"),
)


def test_real_bcc_scheduler_is_loss_free_strict_and_does_not_delay_window(
    tmp_path: Path,
) -> None:
    output = tmp_path / "privileged-bcc"
    report = run_profile(
        ProfileConfig(
            command=(
                "PYTHONPATH=/tasktrace-source python -m "
                "offline_replayer.examples.m1_labeled_phase "
                "--phase-log /tasktrace-profile-output/phases.jsonl "
                "--sentinel /tasktrace-profile-output/executed.once "
                "--serial-cpu-sec 0.35 --memory-mib 64 "
                "--memory-hold-sec 0.45 --parallel-workers 4 "
                "--parallel-cpu-sec 0.45 --teardown-hold-sec 0.25"
            ),
            prepare_command=None,
            verification_command=(
                "test -s /tasktrace-profile-output/executed.once && " "test -s /tasktrace-profile-output/phases.jsonl"
            ),
            output_dir=output,
            image=IMAGE,
            container=None,
            cpuset_cpus=CPUSET,
            cpu_limit=None,
            allocated_cpus=4,
            workdir="/tmp",
            user=None,
            env=(),
            mounts=(f"type=bind,src={SOURCE_ROOT},dst=/tasktrace-source,readonly",),
            strace_mode="none",
            string_limit=4096,
            shell="/bin/bash",
            memory_sample_interval_sec=0.1,
            network="none",
            terminate_descendants_at_finish=True,
            profile_sets=("cpu-compare",),
            scheduler_backend="bcc",
        )
    )

    assert report["command_exit_code"] == 0
    assert report["profiling_valid"] is True
    assert report["profiling_invalid_reasons"] == []
    assert report["verification"]["status"] == "passed"
    assert report["verification"]["measured_counters_include_verification"] is False

    scheduler = report["scheduler_concurrency"]
    validation = scheduler["cgroup_validation"]
    cpu_usage = report["cpu"]["usage_sec"]
    strict_tolerance = max(0.005, 0.02 * cpu_usage)
    assert scheduler["method"] == "ebpf-bcc-finish-task-switch-cgroup-subtree"
    assert scheduler["trace_complete"] is True
    assert scheduler["lost_perf_events"] == 0
    assert scheduler["anomalies"] == []
    assert scheduler["events_observed"] > 0
    assert scheduler["peak_simultaneous_logical_cores"] == 4
    assert abs(scheduler["scheduled_runtime_sec"] - cpu_usage) <= strict_tolerance
    assert validation["within_tolerance"] is True
    assert validation["valid_for_concurrency_reporting"] is True
    assert validation["allowed_delta_sec"] == pytest.approx(strict_tolerance)
    assert validation["event_scaled_tolerance_applied"] is False

    memory = report["memory"]
    assert memory["instrumentation_valid"] is True
    assert memory["peak_valid"] is True
    assert memory["natural_peak_trustworthy"] is True
    assert memory["timeline_valid_for_shape"] is True
    assert memory["timeline_invalid_reasons"] == []
    assert memory["sample_count"] >= 10
    assert memory["lifetime"]["peak_bytes"] >= 56 * 1024 * 1024
    assert memory["ancestor_memory"]["evidence_complete"] is True
    assert memory["ancestor_memory"]["natural_demand_trustworthy"] is True

    markers = {
        marker["name"]: marker["canonical_timestamp_ns"]
        for marker in report["canonical_lifecycle"]["lifecycle_markers"]
    }
    assert markers["measured_work_start"] < markers["measured_work_end"]
    assert markers["measured_work_end"] < markers["verification_start"]
    assert markers["verification_start"] < markers["verification_end"]

    bundle_path = Path(report["evidence_bundle"]["path"])
    bundle = read_bundle(bundle_path)
    validate_raw_artifacts(bundle, bundle_path.parent)
    assert bundle.extensions["evidence_status"] == "valid"
    assert bundle.extensions["workload_verification_valid"] is True
    assert {manifest.profiler_id: manifest.status for manifest in bundle.profilers} == {
        "cgroup-cpu": "valid",
        "cgroup-memory": "valid",
        "scheduler-concurrency": "valid",
    }
