#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Low-privilege sampled host-CPU comparison for profiled command windows.

This deliberately matches the TaskTrace monitor's `/proc/stat` definition of
per-CPU busy time: user + nice + system + irq + softirq, divided by that busy
time plus idle + iowait.  It projects the host samples onto the profiled
container's allocated CPU IDs, but it cannot attribute activity to that
container.  Cgroup accounting and validated scheduler tracing remain the
authoritative measurements for task average and peak concurrency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


class ProcStatSamplingError(RuntimeError):
    """Raised when `/proc/stat` sampling cannot produce comparable output."""


@dataclass(frozen=True)
class ProcCpuCounters:
    """Counters needed by the TaskTrace-style CPU-utilization formula."""

    user: int
    system: int
    idle: int
    iowait: int


@dataclass(frozen=True)
class ProcStatInterval:
    """One sampled interval projected onto allocated logical CPU IDs."""

    duration_sec: float
    busy_logical_cores: float
    active_logical_cpus: int
    utilization_percent_by_logical_cpu: dict[str, float]


def parse_proc_stat(text: str) -> dict[int, ProcCpuCounters]:
    """Parse per-logical-CPU lines from Linux `/proc/stat` content."""
    counters: dict[int, ProcCpuCounters] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields or not fields[0].startswith("cpu") or not fields[0][3:].isdigit():
            continue
        if len(fields) < 9:
            raise ProcStatSamplingError(f"short /proc/stat CPU row: {line!r}")
        cpu = int(fields[0][3:])
        values = [int(value) for value in fields[1:9]]
        counters[cpu] = ProcCpuCounters(
            user=values[0] + values[1],
            system=values[2] + values[5] + values[6],
            idle=values[3],
            iowait=values[4],
        )
    if not counters:
        raise ProcStatSamplingError("no logical CPU rows found in /proc/stat")
    return counters


def calculate_interval(
    previous: dict[int, ProcCpuCounters],
    current: dict[int, ProcCpuCounters],
    *,
    cpu_ids: tuple[int, ...],
    duration_sec: float,
    active_threshold_percent: float,
) -> ProcStatInterval:
    """Calculate one TaskTrace-style utilization sample on selected CPU IDs."""
    if duration_sec <= 0:
        raise ProcStatSamplingError("sample interval must have positive duration")
    utilization: dict[str, float] = {}
    for cpu in cpu_ids:
        if cpu not in previous or cpu not in current:
            raise ProcStatSamplingError(f"logical CPU {cpu} missing from /proc/stat")
        before, after = previous[cpu], current[cpu]
        delta_user = after.user - before.user
        delta_system = after.system - before.system
        delta_idle = after.idle - before.idle
        delta_iowait = after.iowait - before.iowait
        deltas = (delta_user, delta_system, delta_idle, delta_iowait)
        if any(value < 0 for value in deltas):
            raise ProcStatSamplingError(f"logical CPU {cpu} counters moved backwards")
        total = sum(deltas)
        busy = delta_user + delta_system
        percent = round(100 * busy / total, 1) if total else 0.0
        utilization[str(cpu)] = percent
    return ProcStatInterval(
        duration_sec=duration_sec,
        busy_logical_cores=round(sum(utilization.values()) / 100, 9),
        active_logical_cpus=sum(value > active_threshold_percent for value in utilization.values()),
        utilization_percent_by_logical_cpu=utilization,
    )


def summarize_intervals(intervals: list[ProcStatInterval]) -> dict[str, Any]:
    """Summarize fixed-interval samples, weighting any terminal partial bucket."""
    duration_sec = sum(interval.duration_sec for interval in intervals)
    if duration_sec <= 0:
        return {
            "sample_count": 0,
            "duration_sec": 0.0,
            "average_busy_logical_cores": None,
            "average_active_logical_cpus": None,
            "peak_active_logical_cpus": None,
            "peak_busy_logical_cores_in_sample": None,
        }
    return {
        "sample_count": len(intervals),
        "duration_sec": round(duration_sec, 9),
        "average_busy_logical_cores": round(
            sum(interval.busy_logical_cores * interval.duration_sec for interval in intervals) / duration_sec,
            9,
        ),
        "average_active_logical_cpus": round(
            sum(interval.active_logical_cpus * interval.duration_sec for interval in intervals) / duration_sec,
            9,
        ),
        "peak_active_logical_cpus": max(interval.active_logical_cpus for interval in intervals),
        "peak_busy_logical_cores_in_sample": max(interval.busy_logical_cores for interval in intervals),
    }


class ProcStatComparisonSampler:
    """Collect baseline and measured `/proc/stat` samples without blocking a run."""

    def __init__(
        self,
        *,
        cpu_ids: tuple[int, ...],
        interval_sec: float = 0.5,
        baseline_sec: float = 5.0,
        active_threshold_percent: float = 3.0,
        proc_stat_path: Path = Path("/proc/stat"),
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not cpu_ids:
            raise ProcStatSamplingError("at least one logical CPU ID is required")
        if interval_sec <= 0 or baseline_sec <= 0:
            raise ProcStatSamplingError("sampling interval and baseline duration must be positive")
        if active_threshold_percent < 0:
            raise ProcStatSamplingError("active CPU threshold must not be negative")
        self.cpu_ids = cpu_ids
        self.interval_sec = interval_sec
        self.baseline_sec = baseline_sec
        self.active_threshold_percent = active_threshold_percent
        self.proc_stat_path = proc_stat_path
        self.clock = clock
        self.sleep = sleep
        self.baseline_intervals: list[ProcStatInterval] = []
        self.measurement_intervals: list[ProcStatInterval] = []
        self._previous: dict[int, ProcCpuCounters] | None = None
        self._previous_time: float | None = None
        self._next_sample_time: float | None = None

    def _read(self) -> dict[int, ProcCpuCounters]:
        return parse_proc_stat(self.proc_stat_path.read_text())

    def _append_interval(
        self,
        target: list[ProcStatInterval],
        previous: dict[int, ProcCpuCounters],
        previous_time: float,
        current: dict[int, ProcCpuCounters],
        current_time: float,
    ) -> None:
        if current_time <= previous_time:
            return
        target.append(
            calculate_interval(
                previous,
                current,
                cpu_ids=self.cpu_ids,
                duration_sec=current_time - previous_time,
                active_threshold_percent=self.active_threshold_percent,
            )
        )

    def collect_baseline(
        self,
        *,
        poll_callback: Callable[[], None] | None = None,
        poll_interval_sec: float | None = None,
    ) -> None:
        """Collect an idle-context baseline before the gated command is released."""
        if poll_interval_sec is not None and poll_interval_sec <= 0:
            raise ProcStatSamplingError("baseline callback interval must be positive")
        previous = self._read()
        previous_time = self.clock()
        deadline = previous_time + self.baseline_sec
        while previous_time < deadline:
            sample_deadline = min(previous_time + self.interval_sec, deadline)
            while self.clock() < sample_deadline:
                remaining = sample_deadline - self.clock()
                self.sleep(min(poll_interval_sec or remaining, remaining))
                if poll_callback is not None:
                    poll_callback()
            current_time = self.clock()
            current = self._read()
            self._append_interval(self.baseline_intervals, previous, previous_time, current, current_time)
            previous, previous_time = current, current_time

    def start_measurement(self) -> None:
        """Capture the beginning of the released command measurement window."""
        self._previous = self._read()
        self._previous_time = self.clock()
        self._next_sample_time = self._previous_time + self.interval_sec

    def poll(self) -> None:
        """Append an interval when the normal profiler loop reaches its deadline."""
        if self._previous is None or self._previous_time is None or self._next_sample_time is None:
            raise ProcStatSamplingError("measurement sampling was not started")
        current_time = self.clock()
        if current_time < self._next_sample_time:
            return
        current = self._read()
        self._append_interval(
            self.measurement_intervals,
            self._previous,
            self._previous_time,
            current,
            current_time,
        )
        self._previous, self._previous_time = current, current_time
        self._next_sample_time = current_time + self.interval_sec

    def finish(self) -> dict[str, Any]:
        """Finish the terminal partial bucket and return comparison-only metrics."""
        if self._previous is None or self._previous_time is None:
            raise ProcStatSamplingError("measurement sampling was not started")
        current_time = self.clock()
        current = self._read()
        self._append_interval(
            self.measurement_intervals,
            self._previous,
            self._previous_time,
            current,
            current_time,
        )
        baseline = summarize_intervals(self.baseline_intervals)
        measurement = summarize_intervals(self.measurement_intervals)
        baseline_average = baseline["average_busy_logical_cores"]
        measured_average = measurement["average_busy_logical_cores"]
        baseline_peak = baseline["peak_active_logical_cpus"]
        measured_peak = measurement["peak_active_logical_cpus"]
        return {
            "method": "tasktrace-style-proc-stat-sampling",
            "scope": (
                "Host /proc/stat activity projected onto allocated logical CPU IDs; "
                "not container- or task-attributed."
            ),
            "allocated_logical_cpu_ids": list(self.cpu_ids),
            "sample_interval_sec": self.interval_sec,
            "active_threshold_percent": self.active_threshold_percent,
            "baseline": baseline,
            "measurement": measurement,
            "baseline_subtracted_estimate": {
                "average_busy_logical_cores": (
                    round(max(0.0, measured_average - baseline_average), 9)
                    if measured_average is not None and baseline_average is not None
                    else None
                ),
                "peak_active_logical_cpus": (
                    max(0, measured_peak - baseline_peak)
                    if measured_peak is not None and baseline_peak is not None
                    else None
                ),
            },
            "meaning": (
                "Sampled host-pressure context only. The active-CPU threshold can "
                "overstate true simultaneous task execution after CPU migration or "
                "unrelated host work."
            ),
        }

    def write_samples(self, path: Path) -> None:
        """Persist raw interval evidence for later inspection."""
        payload = {
            "baseline": [interval.__dict__ for interval in self.baseline_intervals],
            "measurement": [interval.__dict__ for interval in self.measurement_intervals],
        }
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def add_reference_comparison(
    proc_report: dict[str, Any],
    *,
    cgroup_average_logical_cores: float | None,
    scheduler_peak_simultaneous_logical_cores: int | None,
    scheduler_peak_valid: bool,
) -> dict[str, Any]:
    """Attach differences to authoritative same-window measurements."""
    report = dict(proc_report)
    measured_average = report["measurement"]["average_busy_logical_cores"]
    subtracted_average = report["baseline_subtracted_estimate"]["average_busy_logical_cores"]
    sampled_peak = report["measurement"]["peak_active_logical_cpus"]
    report["reference_comparison"] = {
        "cgroup_average_logical_cores": cgroup_average_logical_cores,
        "raw_proc_minus_cgroup_average_logical_cores": (
            round(measured_average - cgroup_average_logical_cores, 9)
            if measured_average is not None and cgroup_average_logical_cores is not None
            else None
        ),
        "baseline_subtracted_proc_minus_cgroup_average_logical_cores": (
            round(subtracted_average - cgroup_average_logical_cores, 9)
            if subtracted_average is not None and cgroup_average_logical_cores is not None
            else None
        ),
        "scheduler_peak_simultaneous_logical_cores": (
            scheduler_peak_simultaneous_logical_cores if scheduler_peak_valid else None
        ),
        "sampled_proc_peak_active_logical_cpus": sampled_peak,
        "scheduler_peak_valid": scheduler_peak_valid,
        "peak_values_are_equivalent": False,
    }
    return report
