# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from offline_replayer.measurement.proc_stat import (
    ProcStatComparisonSampler,
    add_reference_comparison,
    calculate_interval,
    parse_proc_stat,
    summarize_intervals,
)


def _stat(cpu0_user: int, cpu0_idle: int, cpu1_user: int, cpu1_idle: int) -> dict:
    return parse_proc_stat(f"cpu0 {cpu0_user} 0 0 {cpu0_idle} 0 0 0 0\n" f"cpu1 {cpu1_user} 0 0 {cpu1_idle} 0 0 0 0\n")


def test_proc_interval_matches_tasktrace_active_cpu_definition() -> None:
    interval = calculate_interval(
        _stat(0, 100, 0, 100),
        _stat(25, 175, 0, 200),
        cpu_ids=(0, 1),
        duration_sec=0.5,
        active_threshold_percent=3.0,
    )

    assert interval.utilization_percent_by_logical_cpu == {"0": 25.0, "1": 0.0}
    assert interval.busy_logical_cores == 0.25
    assert interval.active_logical_cpus == 1


def test_proc_interval_summary_weights_terminal_partial_bucket() -> None:
    first = calculate_interval(
        _stat(0, 100, 0, 100),
        _stat(25, 175, 0, 200),
        cpu_ids=(0, 1),
        duration_sec=0.5,
        active_threshold_percent=3.0,
    )
    terminal = calculate_interval(
        _stat(25, 175, 0, 200),
        _stat(25, 275, 25, 200),
        cpu_ids=(0, 1),
        duration_sec=0.25,
        active_threshold_percent=3.0,
    )

    summary = summarize_intervals([first, terminal])

    assert summary["duration_sec"] == 0.75
    assert summary["average_busy_logical_cores"] == 0.5
    assert summary["peak_active_logical_cpus"] == 1


def test_proc_sampler_reports_baseline_subtracted_estimate_and_reference_delta(
    monkeypatch,
) -> None:
    clock_value = [0.0]

    def clock() -> float:
        return clock_value[0]

    def sleep(seconds: float) -> None:
        clock_value[0] += seconds

    snapshots = iter(
        [
            _stat(0, 100, 0, 100),
            _stat(0, 150, 0, 150),
            _stat(0, 150, 0, 150),
            _stat(50, 150, 0, 200),
            _stat(75, 175, 0, 225),
        ]
    )
    sampler = ProcStatComparisonSampler(
        cpu_ids=(0, 1),
        interval_sec=0.5,
        baseline_sec=0.5,
        clock=clock,
        sleep=sleep,
    )
    monkeypatch.setattr(sampler, "_read", lambda: next(snapshots))

    sampler.collect_baseline()
    sampler.start_measurement()
    clock_value[0] += 0.5
    sampler.poll()
    clock_value[0] += 0.25
    report = sampler.finish()
    report = add_reference_comparison(
        report,
        cgroup_average_logical_cores=0.75,
        scheduler_peak_simultaneous_logical_cores=1,
        scheduler_peak_valid=True,
    )

    assert report["baseline"]["average_busy_logical_cores"] == 0.0
    assert report["measurement"]["average_busy_logical_cores"] == 0.833333333
    assert report["baseline_subtracted_estimate"]["average_busy_logical_cores"] == 0.833333333
    assert report["reference_comparison"]["raw_proc_minus_cgroup_average_logical_cores"] == 0.083333333
    assert report["reference_comparison"]["peak_values_are_equivalent"] is False


def test_proc_baseline_can_poll_a_faster_resource_observer(monkeypatch) -> None:
    clock_value = [0.0]
    callback_times: list[float] = []

    def clock() -> float:
        return clock_value[0]

    def sleep(seconds: float) -> None:
        clock_value[0] += seconds

    snapshots = iter([_stat(0, 100, 0, 100), _stat(0, 150, 0, 150)])
    sampler = ProcStatComparisonSampler(
        cpu_ids=(0, 1),
        interval_sec=0.5,
        baseline_sec=0.5,
        clock=clock,
        sleep=sleep,
    )
    monkeypatch.setattr(sampler, "_read", lambda: next(snapshots))

    sampler.collect_baseline(
        poll_callback=lambda: callback_times.append(clock()),
        poll_interval_sec=0.1,
    )

    assert len(callback_times) == 5
    assert callback_times[-1] == 0.5
