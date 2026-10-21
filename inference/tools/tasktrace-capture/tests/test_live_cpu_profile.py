# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
from tasktrace_capture.live_cpu_profile import (
    LiveCpuProfileError,
    _cpu_window,
    parse_cpuset_cpu_ids,
    validate_profile_cpuset_cpus,
)
from tasktrace_capture.scheduler_concurrency import (
    add_cgroup_validation,
    parse_perf_script_switch_events,
    summarize_scheduler_events,
)


def test_cpu_window_reports_average_logical_cores():
    report = _cpu_window(
        "agent_execution",
        before={"usage_usec": 100, "user_usec": 50, "system_usec": 50},
        after={"usage_usec": 2_000_100, "user_usec": 1_000_050, "system_usec": 1_000_050},
        start_ns=1_000_000_000,
        end_ns=2_000_000_000,
    )
    assert report["usage_sec"] == 2.0
    assert report["average_logical_cores_used"] == 2.0


def test_perf_switch_events_produce_validated_peak_two():
    decoded = "\n".join(
        [
            " x 10/10 [000] 1.000000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0",
            " y 11/11 [001] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0",
            " x 10/10 [000] 1.600000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0",
            " y 11/11 [001] 2.000000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0",
        ]
    )
    events, lost, anomalies = parse_perf_script_switch_events(decoded)
    report = summarize_scheduler_events(
        events,
        window_start_ns=1_000_000_000,
        window_end_ns=2_000_000_000,
        lost_events=lost,
        decoder_anomalies=anomalies,
    )
    validated = add_cgroup_validation(report, cgroup_usage_sec=1.5)
    assert validated["peak_simultaneous_logical_cores"] == 2
    assert validated["scheduled_runtime_sec"] == 1.5
    assert validated["cgroup_validation"]["valid_for_concurrency_reporting"] is True


def test_perf_switch_events_accept_opaque_tid_on_multiple_cpus():
    decoded = "\n".join(
        [
            " x 10/-1 [000] 1.000000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0",
            " x 10/-1 [001] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0",
            " x 10/-1 [000] 1.600000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0",
            " x 10/-1 [001] 2.000000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0",
        ]
    )
    events, lost, anomalies = parse_perf_script_switch_events(decoded)
    report = summarize_scheduler_events(
        events,
        window_start_ns=1_000_000_000,
        window_end_ns=2_000_000_000,
        lost_events=lost,
        decoder_anomalies=anomalies,
    )
    validated = add_cgroup_validation(report, cgroup_usage_sec=1.5)
    assert anomalies == []
    assert validated["peak_simultaneous_logical_cores"] == 2
    assert validated["cgroup_validation"]["valid_for_concurrency_reporting"] is True


def test_perf_switch_events_coalesce_internal_switch_and_exact_duplicate():
    decoded = "\n".join(
        [
            " a 10/10 [000] 1.000000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0",
            " a 10/10 [000] 1.500000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 10/11",
            " b 10/11 [000] 1.500010000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 10/10",
            " b 10/11 [000] 2.000000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0",
            " b 10/11 [000] 2.000000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0",
        ]
    )
    events, lost, anomalies = parse_perf_script_switch_events(decoded)
    report = summarize_scheduler_events(
        events,
        window_start_ns=1_000_000_000,
        window_end_ns=2_000_000_000,
        lost_events=lost,
        decoder_anomalies=anomalies,
    )
    validated = add_cgroup_validation(report, cgroup_usage_sec=1.0)
    assert len(events) == 2
    assert validated["scheduled_runtime_sec"] == 1.0
    assert validated["cgroup_validation"]["valid_for_concurrency_reporting"] is True


def test_parse_cpuset_cpu_ids_handles_ranges_and_duplicates():
    assert parse_cpuset_cpu_ids("8-15,12,17") == tuple(range(8, 16)) + (17,)


def test_parse_cpuset_cpu_ids_rejects_descending_range():
    with pytest.raises(LiveCpuProfileError, match="invalid CPU set range"):
        parse_cpuset_cpu_ids("15-8")


def test_scheduler_profile_requires_named_cpuset():
    with pytest.raises(LiveCpuProfileError, match="explicitly choose"):
        validate_profile_cpuset_cpus(
            profile_scheduler="required",
            requested_cpuset=None,
            override_cpus=4,
        )


def test_named_scheduler_cpuset_must_match_allocation_width():
    with pytest.raises(LiveCpuProfileError, match="width must match"):
        validate_profile_cpuset_cpus(
            profile_scheduler="required",
            requested_cpuset="8-15",
            override_cpus=4,
        )
