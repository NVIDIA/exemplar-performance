# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from offline_replayer.measurement.scheduler_bcc import (
    SchedulerConcurrencyTracer,
    SchedulerEvent,
    SchedulerTraceError,
    add_cgroup_validation,
    summarize_scheduler_events,
)


def test_scheduler_summary_integrates_time_at_each_concurrency() -> None:
    second = 1_000_000_000
    report = summarize_scheduler_events(
        [
            SchedulerEvent(1 * second, 0, 100, True),
            SchedulerEvent(2 * second, 1, 101, True),
            SchedulerEvent(4 * second, 0, 100, False),
            SchedulerEvent(5 * second, 1, 101, False),
        ],
        window_start_ns=0,
        window_end_ns=10 * second,
    )

    assert report["scheduled_runtime_sec"] == 6.0
    assert report["average_logical_cores_used"] == 0.6
    assert report["peak_simultaneous_logical_cores"] == 2
    assert report["time_sec_by_simultaneous_logical_cores"] == {
        "0": 6.0,
        "1": 2.0,
        "2": 2.0,
    }
    assert report["runtime_sec_by_logical_cpu"] == {"0": 3.0, "1": 3.0}
    assert report["trace_complete"] is True


def test_scheduler_peak_ignores_zero_duration_cross_cpu_transition() -> None:
    report = summarize_scheduler_events(
        [
            SchedulerEvent(10, 0, 100, True),
            SchedulerEvent(20, 0, 100, False),
            SchedulerEvent(20, 1, 101, True),
            SchedulerEvent(30, 1, 101, False),
        ],
        window_start_ns=0,
        window_end_ns=40,
    )

    assert report["peak_simultaneous_logical_cores"] == 1
    assert report["trace_complete"] is True


def test_scheduler_summary_carries_active_cpu_across_window_start() -> None:
    report = summarize_scheduler_events(
        [
            SchedulerEvent(5, 1, 101, True),
            SchedulerEvent(20, 1, 101, False),
        ],
        window_start_ns=10,
        window_end_ns=30,
    )

    assert report["logical_cpus_active_at_window_start"] == [1]
    assert report["scheduled_runtime_sec"] == 0.00000001
    assert report["peak_simultaneous_logical_cores"] == 1
    assert report["anomalies"] == []
    assert report["trace_complete"] is True


def test_scheduler_validation_rejects_lost_events_or_cpu_mismatch() -> None:
    report = summarize_scheduler_events(
        [
            SchedulerEvent(0, 0, 100, True),
            SchedulerEvent(1_000_000_000, 0, 100, False),
        ],
        window_start_ns=0,
        window_end_ns=1_000_000_000,
        lost_events=1,
    )

    validated = add_cgroup_validation(
        report,
        cgroup_usage_sec=0.5,
        tolerance_fraction=0.01,
        tolerance_sec=0.001,
    )

    assert validated["trace_complete"] is False
    assert validated["cgroup_validation"]["within_tolerance"] is False
    assert validated["cgroup_validation"]["valid_for_concurrency_reporting"] is False


def test_scheduler_validation_does_not_scale_tolerance_with_event_count() -> None:
    report = {
        "scheduled_runtime_sec": 81.0,
        "events_observed": 10_000_000,
        "trace_complete": True,
    }

    validated = add_cgroup_validation(
        report,
        cgroup_usage_sec=100.0,
        tolerance_fraction=0.02,
        tolerance_sec=0.005,
        per_observed_event_tolerance_sec=0.000002,
    )

    validation = validated["cgroup_validation"]
    assert validation["fractional_or_absolute_tolerance_sec"] == 2.0
    assert validation["event_scaled_tolerance_sec"] == 20.0
    assert validation["allowed_negative_delta_sec"] == 2.0
    assert validation["event_scaled_tolerance_applied"] is False
    assert validation["within_tolerance"] is False
    assert validation["valid_for_concurrency_reporting"] is False


def test_scheduler_validation_accepts_exact_strict_boundary() -> None:
    report = {
        "scheduled_runtime_sec": 98.0,
        "events_observed": 10_000_000,
        "trace_complete": True,
    }

    validated = add_cgroup_validation(
        report,
        cgroup_usage_sec=100.0,
        tolerance_fraction=0.02,
        tolerance_sec=0.005,
        per_observed_event_tolerance_sec=0.000002,
    )

    validation = validated["cgroup_validation"]
    assert validation["allowed_delta_sec"] == 2.0
    assert validation["within_tolerance"] is True
    assert validation["valid_for_concurrency_reporting"] is True


def test_scheduler_validation_event_bound_does_not_allow_positive_overcount() -> None:
    report = {
        "scheduled_runtime_sec": 119.0,
        "events_observed": 10_000_000,
        "trace_complete": True,
    }

    validated = add_cgroup_validation(
        report,
        cgroup_usage_sec=100.0,
        tolerance_fraction=0.02,
        tolerance_sec=0.005,
        per_observed_event_tolerance_sec=0.000002,
    )

    validation = validated["cgroup_validation"]
    assert validation["allowed_negative_delta_sec"] == 2.0
    assert validation["allowed_positive_delta_sec"] == 2.0
    assert validation["within_tolerance"] is False
    assert validation["valid_for_concurrency_reporting"] is False


def test_scheduler_tracer_counts_lost_events_for_bcc_callback_signatures() -> None:
    tracer = SchedulerConcurrencyTracer()

    tracer._record_lost_events(7)
    tracer._record_lost_events(0, 5)

    assert tracer._lost_events == 12


def test_scheduler_tracer_requires_power_of_two_buffer_pages() -> None:
    with pytest.raises(SchedulerTraceError, match="positive power of two"):
        SchedulerConcurrencyTracer(buffer_pages=1000)
