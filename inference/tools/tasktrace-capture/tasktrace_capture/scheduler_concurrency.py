# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Decode and validate native-perf scheduler concurrency measurements."""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from itertools import groupby
from typing import Any


class SchedulerTraceError(RuntimeError):
    """Raised when scheduler records cannot support a concurrency result."""


@dataclass(frozen=True)
class SchedulerEvent:
    timestamp_ns: int
    cpu: int
    tid: int
    entering: bool


@dataclass(frozen=True)
class _PerfSwitchRecord:
    timestamp_ns: int
    cpu: int
    pid: int
    tid: int
    entering: bool
    peer_pid: int
    peer_tid: int


_PERF_RECORD_LINE = re.compile(
    r"^\s*.*?\s+(?P<pid>-?\d+)/(?P<tid>-?\d+)\s+"
    r"\[(?P<cpu>\d+)\]\s+(?P<timestamp>[0-9]+\.[0-9]+):\s+"
    r"PERF_RECORD_(?P<event>SWITCH(?:_CPU_WIDE)?|EXIT)\b(?P<details>.*)$"
)
_PERF_SWITCH_DIRECTION = re.compile(r"^\s+(?P<direction>IN|OUT)\b")
_PERF_SWITCH_PEER = re.compile(r"\b(?:prev|next)\s+pid/tid:\s+(?P<pid>-?\d+)/(?P<tid>-?\d+)")
_LOST_COUNT = re.compile(r"\blost(?:\s+events?)?\s*[:=]?\s*(?P<count>\d+)", re.IGNORECASE)


def _timestamp_ns(timestamp_sec: str) -> int:
    try:
        return int(Decimal(timestamp_sec) * Decimal(1_000_000_000))
    except (InvalidOperation, ValueError) as exc:
        raise SchedulerTraceError(f"invalid perf script timestamp: {timestamp_sec!r}") from exc


def _same_perf_task(first_pid: int, first_tid: int, second_pid: int, second_tid: int) -> bool:
    """Compare perf task identities while accepting an unavailable TID."""
    return first_pid == second_pid and (first_tid == second_tid or first_tid < 0 or second_tid < 0)


def parse_perf_script_switch_events(
    text: str,
) -> tuple[list[SchedulerEvent], int, list[str]]:
    """Convert cgroup-scoped ``perf script`` records to occupancy transitions.

    CPU-wide perf output may emit ``tid=-1`` after a task exits, so it cannot
    be used as a stable cross-CPU identity. It also emits both sides of a
    switch between two tasks in the measured cgroup. That pair represents
    uninterrupted cgroup occupancy and must not create an artificial idle gap.
    """
    records: list[_PerfSwitchRecord] = []
    seen_records: set[_PerfSwitchRecord] = set()
    lost_events = 0
    for line in text.splitlines():
        if "PERF_RECORD_LOST" in line:
            count = _LOST_COUNT.search(line)
            lost_events += int(count.group("count")) if count else 1
        match = _PERF_RECORD_LINE.match(line)
        if match is None or not match.group("event").startswith("SWITCH"):
            continue
        direction = _PERF_SWITCH_DIRECTION.match(match.group("details"))
        if direction is None:
            raise SchedulerTraceError(f"unable to decode perf switch direction: {line!r}")
        peer = _PERF_SWITCH_PEER.search(match.group("details"))
        if peer is None:
            raise SchedulerTraceError(f"unable to decode perf switch peer: {line!r}")
        record = _PerfSwitchRecord(
            _timestamp_ns(match.group("timestamp")),
            int(match.group("cpu")),
            int(match.group("pid")),
            int(match.group("tid")),
            direction.group("direction") == "IN",
            int(peer.group("pid")),
            int(peer.group("tid")),
        )
        # perf can print the same terminal switch twice. Treating both OUT
        # records as occupancy changes would manufacture an impossible exit.
        if record not in seen_records:
            records.append(record)
            seen_records.add(record)

    sorted_records = sorted(
        records,
        key=lambda item: (item.timestamp_ns, item.cpu, item.entering, item.tid),
    )
    records_by_cpu: dict[int, list[tuple[int, _PerfSwitchRecord]]] = defaultdict(list)
    for index, record in enumerate(sorted_records):
        records_by_cpu[record.cpu].append((index, record))

    # A cgroup remains scheduled when CPU N switches directly from one of its
    # tasks to another. Perf prints an OUT/IN pair for that handoff; omitting
    # the pair prevents a fake idle interval from accumulating at high rates.
    internal_switch_indices: set[int] = set()
    for cpu_records in records_by_cpu.values():
        for (out_index, outgoing), (in_index, incoming) in zip(cpu_records, cpu_records[1:], strict=False):
            if outgoing.entering or not incoming.entering:
                continue
            if _same_perf_task(
                outgoing.peer_pid,
                outgoing.peer_tid,
                incoming.pid,
                incoming.tid,
            ) and _same_perf_task(
                incoming.peer_pid,
                incoming.peer_tid,
                outgoing.pid,
                outgoing.tid,
            ):
                internal_switch_indices.update((out_index, in_index))

    events = [
        SchedulerEvent(record.timestamp_ns, record.cpu, record.tid, record.entering)
        for index, record in enumerate(sorted_records)
        if index not in internal_switch_indices
    ]
    return events, lost_events, []


def summarize_scheduler_events(
    events: list[SchedulerEvent],
    *,
    window_start_ns: int,
    window_end_ns: int,
    lost_events: int = 0,
    decoder_anomalies: list[str] | None = None,
) -> dict[str, Any]:
    """Integrate event-timed simultaneous execution over one bounded window."""
    if window_end_ns <= window_start_ns:
        raise SchedulerTraceError("scheduler window must have positive duration")

    active_started: dict[int, int] = {}
    for event in sorted(
        (event for event in events if event.timestamp_ns < window_start_ns),
        key=lambda item: (item.timestamp_ns, not item.entering, item.cpu, item.tid),
    ):
        if event.entering:
            active_started[event.cpu] = window_start_ns
        else:
            active_started.pop(event.cpu, None)

    selected = sorted(
        (event for event in events if window_start_ns <= event.timestamp_ns <= window_end_ns),
        key=lambda item: (item.timestamp_ns, not item.entering, item.cpu, item.tid),
    )
    runtime_ns_by_cpu: dict[int, int] = defaultdict(int)
    duration_ns_by_concurrency: dict[int, int] = defaultdict(int)
    anomalies = list(decoder_anomalies or [])
    start_active = sorted(active_started)
    cursor_ns = window_start_ns
    peak = 0

    for timestamp_ns, grouped in groupby(selected, key=lambda item: item.timestamp_ns):
        interval_ns = timestamp_ns - cursor_ns
        if interval_ns > 0:
            width = len(active_started)
            duration_ns_by_concurrency[width] += interval_ns
            peak = max(peak, width)
        timestamp_events = list(grouped)
        for event in (event for event in timestamp_events if not event.entering):
            started_ns = active_started.pop(event.cpu, None)
            if started_ns is None:
                anomalies.append(f"logical CPU {event.cpu} left the measured cgroup without an active interval")
            else:
                runtime_ns_by_cpu[event.cpu] += event.timestamp_ns - started_ns
        for event in (event for event in timestamp_events if event.entering):
            if event.cpu in active_started:
                anomalies.append(f"logical CPU {event.cpu} entered the measured cgroup while already active")
            else:
                active_started[event.cpu] = event.timestamp_ns
        cursor_ns = timestamp_ns

    tail_ns = window_end_ns - cursor_ns
    if tail_ns > 0:
        width = len(active_started)
        duration_ns_by_concurrency[width] += tail_ns
        peak = max(peak, width)
    for cpu, started_ns in active_started.items():
        runtime_ns_by_cpu[cpu] += window_end_ns - started_ns

    window_ns = window_end_ns - window_start_ns
    scheduled_runtime_ns = sum(width * duration_ns for width, duration_ns in duration_ns_by_concurrency.items())
    per_cpu_runtime_ns = sum(runtime_ns_by_cpu.values())
    if scheduled_runtime_ns != per_cpu_runtime_ns:
        anomalies.append(
            "scheduler concurrency integral differs from per-CPU runtime total "
            f"({scheduled_runtime_ns} ns versus {per_cpu_runtime_ns} ns)"
        )
    return {
        "backend": "perf",
        "method": "perf-record-cgroup-native-switch-events",
        "scope": "native perf switch records restricted to the measured cgroup slice",
        "window_wall_sec": round(window_ns / 1_000_000_000, 9),
        "events_observed": len(selected),
        "events_ignored_before_window": sum(event.timestamp_ns < window_start_ns for event in events),
        "events_ignored_after_window": sum(event.timestamp_ns > window_end_ns for event in events),
        "logical_cpus_active_at_window_start": start_active,
        "lost_perf_events": lost_events,
        "scheduled_runtime_sec": round(scheduled_runtime_ns / 1_000_000_000, 9),
        "average_logical_cores_used": round(scheduled_runtime_ns / window_ns, 9),
        "peak_simultaneous_logical_cores": peak,
        "time_sec_by_simultaneous_logical_cores": {
            str(width): round(duration_ns / 1_000_000_000, 9)
            for width, duration_ns in sorted(duration_ns_by_concurrency.items())
        },
        "runtime_sec_by_logical_cpu": {
            str(cpu): round(duration_ns / 1_000_000_000, 9) for cpu, duration_ns in sorted(runtime_ns_by_cpu.items())
        },
        "anomalies": anomalies,
        "trace_complete": lost_events == 0 and not anomalies,
        "meaning": (
            "Time-integrated simultaneous scheduled execution on logical CPUs; " "this is not physical-core use."
        ),
    }


def add_cgroup_validation(
    scheduler_report: dict[str, Any],
    *,
    cgroup_usage_sec: float,
    tolerance_fraction: float = 0.02,
    tolerance_sec: float = 0.005,
) -> dict[str, Any]:
    """Require scheduler runtime to agree with the scoped cgroup CPU truth."""
    result = dict(scheduler_report)
    delta_sec = float(result["scheduled_runtime_sec"]) - cgroup_usage_sec
    allowed_sec = max(tolerance_sec, abs(cgroup_usage_sec) * tolerance_fraction)
    within = abs(delta_sec) <= allowed_sec
    result["cgroup_validation"] = {
        "authoritative_cgroup_usage_sec": round(cgroup_usage_sec, 9),
        "scheduler_minus_cgroup_usage_sec": round(delta_sec, 9),
        "allowed_delta_sec": round(allowed_sec, 9),
        "tolerance_fraction": tolerance_fraction,
        "tolerance_sec": tolerance_sec,
        "within_tolerance": within,
        "valid_for_concurrency_reporting": bool(result["trace_complete"]) and within,
    }
    return result
