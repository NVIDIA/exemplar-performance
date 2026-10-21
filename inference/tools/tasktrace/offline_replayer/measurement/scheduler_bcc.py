#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Privileged scheduler tracing for exact simultaneous logical-CPU occupancy.

This module is intentionally separate from process-exit tracers.  A task can
consume accurate aggregate CPU seconds without revealing how many logical CPUs
ran it at the same time.  Tracking CPU transitions into and out of the target
container cgroup provides that concurrency timeline.

The runtime BCC dependency is imported only when tracing starts so report
analysis and unit tests remain usable without eBPF privileges.
"""

from __future__ import annotations

import ctypes
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path
from typing import Any


class SchedulerTraceError(RuntimeError):
    """Raised when scheduler tracing cannot start or cannot produce valid data."""


@dataclass(frozen=True)
class SchedulerEvent:
    """One transition of the measured scope onto or off a logical CPU."""

    timestamp_ns: int
    cpu: int
    tid: int
    entering: bool


BPF_PROGRAM = r"""
#include <uapi/linux/ptrace.h>

struct event_t {
    u64 timestamp_ns;
    u32 cpu;
    u32 tid;
    u8 entering;
};

struct cpu_state_t {
    u32 tid;
    u8 active;
};

BPF_ARRAY(target_cgroup_id, u64, 1);
BPF_ARRAY(target_cgroup_level, u32, 1);
BPF_PERCPU_ARRAY(cpu_state, struct cpu_state_t, 1);
BPF_PERF_OUTPUT(events);

static __always_inline void emit_transition(
    void *ctx,
    u64 timestamp_ns,
    u32 tid,
    u8 entering
)
{
    struct event_t event = {};
    event.timestamp_ns = timestamp_ns;
    event.cpu = bpf_get_smp_processor_id();
    event.tid = tid;
    event.entering = entering;
    events.perf_submit(ctx, &event, sizeof(event));
}

int trace_finish_task_switch(struct pt_regs *ctx)
{
    u64 timestamp_ns = bpf_ktime_get_ns();
    u32 key = 0;
    u64 *target = target_cgroup_id.lookup(&key);
    u32 *target_level = target_cgroup_level.lookup(&key);
    struct cpu_state_t *state = cpu_state.lookup(&key);
    if (!target || !target_level || !state)
        return 0;

    u64 current_cgroup_id = bpf_get_current_ancestor_cgroup_id(*target_level);
    u32 current_tid = (u32)bpf_get_current_pid_tgid();
    u8 active = current_cgroup_id == *target;

    if (state->active && !active) {
        emit_transition(ctx, timestamp_ns, state->tid, 0);
    } else if (!state->active && active) {
        emit_transition(ctx, timestamp_ns, current_tid, 1);
    }

    state->active = active;
    state->tid = active ? current_tid : 0;
    return 0;
}
"""


def summarize_scheduler_events(
    events: list[SchedulerEvent],
    *,
    window_start_ns: int,
    window_end_ns: int,
    lost_events: int = 0,
    method: str = "scheduler-transition-integration",
    scope: str = "configured measured scheduler scope",
) -> dict[str, Any]:
    """Integrate a scheduler transition stream into a concurrency histogram.

    Events are sorted across CPU-local perf buffers before integration.  Peak
    concurrency is counted only when it persists for positive time, avoiding a
    zero-duration spike when same-timestamp transitions are delivered in an
    arbitrary cross-CPU order.
    """
    if window_end_ns <= window_start_ns:
        raise SchedulerTraceError("scheduler measurement window must have positive duration")

    selected = sorted(
        (event for event in events if window_start_ns <= event.timestamp_ns <= window_end_ns),
        key=lambda event: (event.timestamp_ns, not event.entering, event.cpu, event.tid),
    )
    ignored_before = sum(event.timestamp_ns < window_start_ns for event in events)
    ignored_after = sum(event.timestamp_ns > window_end_ns for event in events)
    active_cpu_started: dict[int, int] = {}
    for event in sorted(
        (event for event in events if event.timestamp_ns < window_start_ns),
        key=lambda event: (event.timestamp_ns, not event.entering, event.cpu, event.tid),
    ):
        if event.entering:
            active_cpu_started[event.cpu] = window_start_ns
        else:
            active_cpu_started.pop(event.cpu, None)
    active_cpus_at_window_start = sorted(active_cpu_started)
    runtime_ns_by_cpu: dict[int, int] = defaultdict(int)
    duration_ns_by_concurrency: dict[int, int] = defaultdict(int)
    anomalies: list[str] = []
    peak = 0
    cursor_ns = window_start_ns

    for timestamp_ns, timestamp_events_iter in groupby(selected, key=lambda event: event.timestamp_ns):
        interval_ns = timestamp_ns - cursor_ns
        if interval_ns > 0:
            running = len(active_cpu_started)
            duration_ns_by_concurrency[running] += interval_ns
            peak = max(peak, running)
        timestamp_events = list(timestamp_events_iter)
        for event in (event for event in timestamp_events if not event.entering):
            started_ns = active_cpu_started.pop(event.cpu, None)
            if started_ns is None:
                anomalies.append(f"logical CPU {event.cpu} left the measured scope without an active interval")
            else:
                runtime_ns_by_cpu[event.cpu] += event.timestamp_ns - started_ns
        for event in (event for event in timestamp_events if event.entering):
            if event.cpu in active_cpu_started:
                anomalies.append(f"logical CPU {event.cpu} entered the measured scope while already active")
            else:
                active_cpu_started[event.cpu] = event.timestamp_ns
        cursor_ns = timestamp_ns

    tail_ns = window_end_ns - cursor_ns
    if tail_ns > 0:
        running = len(active_cpu_started)
        duration_ns_by_concurrency[running] += tail_ns
        peak = max(peak, running)
    if active_cpu_started:
        for cpu, started_ns in sorted(active_cpu_started.items()):
            runtime_ns_by_cpu[cpu] += window_end_ns - started_ns
        anomalies.append(
            "measurement window ended while target tasks were still scheduled on "
            + ", ".join(str(cpu) for cpu in sorted(active_cpu_started))
        )

    window_ns = window_end_ns - window_start_ns
    scheduled_runtime_ns = sum(
        concurrency * duration_ns for concurrency, duration_ns in duration_ns_by_concurrency.items()
    )
    per_cpu_runtime_ns = sum(runtime_ns_by_cpu.values())
    if scheduled_runtime_ns != per_cpu_runtime_ns:
        anomalies.append(
            "scheduler concurrency integral differs from per-CPU runtime total "
            f"({scheduled_runtime_ns} ns versus {per_cpu_runtime_ns} ns)"
        )

    duration_sec = {
        str(concurrency): round(duration_ns / 1_000_000_000, 9)
        for concurrency, duration_ns in sorted(duration_ns_by_concurrency.items())
    }
    duration_fraction = {
        str(concurrency): round(duration_ns / window_ns, 9)
        for concurrency, duration_ns in sorted(duration_ns_by_concurrency.items())
    }
    return {
        "method": method,
        "scope": scope,
        "window_wall_sec": round(window_ns / 1_000_000_000, 9),
        "events_observed": len(selected),
        "events_ignored_before_window": ignored_before,
        "events_ignored_after_window": ignored_after,
        "logical_cpus_active_at_window_start": active_cpus_at_window_start,
        "lost_perf_events": lost_events,
        "scheduled_runtime_sec": round(scheduled_runtime_ns / 1_000_000_000, 9),
        "average_logical_cores_used": round(scheduled_runtime_ns / window_ns, 9),
        "peak_simultaneous_logical_cores": peak,
        "time_sec_by_simultaneous_logical_cores": duration_sec,
        "fraction_of_window_by_simultaneous_logical_cores": duration_fraction,
        "runtime_sec_by_logical_cpu": {
            str(cpu): round(runtime_ns / 1_000_000_000, 9) for cpu, runtime_ns in sorted(runtime_ns_by_cpu.items())
        },
        "anomalies": anomalies,
        "trace_complete": lost_events == 0 and not anomalies,
        "meaning": (
            "Time-integrated simultaneous scheduled execution of the profiled "
            "measured container scope on logical CPUs; this is not physical-core use."
        ),
    }


def add_cgroup_validation(
    scheduler_report: dict[str, Any],
    *,
    cgroup_usage_sec: float,
    tolerance_fraction: float = 0.02,
    tolerance_sec: float = 0.005,
    per_observed_event_tolerance_sec: float = 0.0,
) -> dict[str, Any]:
    """Cross-check scheduler integration against cgroup CPU accounting."""
    scheduler_usage_sec = float(scheduler_report["scheduled_runtime_sec"])
    delta_sec = scheduler_usage_sec - cgroup_usage_sec
    base_tolerance_sec = max(tolerance_sec, abs(cgroup_usage_sec) * tolerance_fraction)
    event_count = int(scheduler_report.get("events_observed", 0))
    # Event volume is useful diagnostic context, but it must never relax the
    # completeness gate.  Otherwise a busy trace can appear valid while
    # missing an arbitrarily large amount of scheduled runtime.  Apply the
    # documented symmetric max(fractional, absolute) bound in both directions.
    event_scaled_tolerance_sec = event_count * per_observed_event_tolerance_sec
    within_tolerance = -base_tolerance_sec <= delta_sec <= base_tolerance_sec
    result = dict(scheduler_report)
    result["cgroup_validation"] = {
        "authoritative_cgroup_usage_sec": cgroup_usage_sec,
        "scheduler_minus_cgroup_usage_sec": round(delta_sec, 9),
        "allowed_delta_sec": round(base_tolerance_sec, 9),
        "allowed_negative_delta_sec": round(base_tolerance_sec, 9),
        "allowed_positive_delta_sec": round(base_tolerance_sec, 9),
        "fractional_or_absolute_tolerance_sec": round(base_tolerance_sec, 9),
        "per_observed_event_tolerance_sec": per_observed_event_tolerance_sec,
        "event_scaled_tolerance_sec": round(event_scaled_tolerance_sec, 9),
        "event_scaled_tolerance_applied": False,
        "tolerance_fraction": tolerance_fraction,
        "tolerance_sec": tolerance_sec,
        "within_tolerance": within_tolerance,
        "valid_for_concurrency_reporting": (bool(scheduler_report.get("trace_complete")) and within_tolerance),
    }
    return result


class SchedulerConcurrencyTracer:
    """Collect scheduler transitions for one disposable container cgroup."""

    def __init__(self, *, buffer_pages: int | None = None) -> None:
        if buffer_pages is not None and (buffer_pages <= 0 or buffer_pages & (buffer_pages - 1)):
            raise SchedulerTraceError("BCC perf-buffer pages must be a positive power of two")
        self._bpf: Any | None = None
        self._events: list[SchedulerEvent] = []
        self._lost_events = 0
        self._buffer_pages = buffer_pages
        self.root_tid: int | None = None
        self.target_cgroup_id: int | None = None
        self.target_cgroup_level: int | None = None
        self.kprobe_symbol: str | None = None

    def start(self, root_tid: int, *, cgroup_path: Path) -> None:
        if os.geteuid() != 0:
            raise SchedulerTraceError("--scheduler-concurrency requires root privileges to load eBPF programs")
        try:
            from bcc import BPF
        except ImportError as exc:
            raise SchedulerTraceError("--scheduler-concurrency requires Python BCC bindings") from exc
        try:
            self._bpf = BPF(text=BPF_PROGRAM)
            target_cgroup_id = cgroup_path.stat().st_ino
            try:
                target_cgroup_level = len(cgroup_path.relative_to("/sys/fs/cgroup").parts)
            except ValueError as exc:
                raise SchedulerTraceError(f"cgroup path is outside the cgroup v2 hierarchy: {cgroup_path}") from exc
            target_map = self._bpf["target_cgroup_id"]
            target_map[ctypes.c_int(0)] = ctypes.c_ulonglong(target_cgroup_id)
            target_level_map = self._bpf["target_cgroup_level"]
            target_level_map[ctypes.c_int(0)] = ctypes.c_uint(target_cgroup_level)
            buffer_options: dict[str, Any] = {"lost_cb": self._record_lost_events}
            if self._buffer_pages is not None:
                buffer_options["page_cnt"] = self._buffer_pages
            self._bpf["events"].open_perf_buffer(self._record_event, **buffer_options)
            symbols = BPF.get_kprobe_functions(b"^finish_task_switch(\\.isra\\.[0-9]+)?$")
            if not symbols:
                raise SchedulerTraceError("finish_task_switch kprobe symbol is unavailable")
            symbol = sorted(symbols)[0]
            self._bpf.attach_kprobe(
                event=symbol,
                fn_name="trace_finish_task_switch",
            )
        except Exception as exc:
            raise SchedulerTraceError(f"unable to attach scheduler eBPF tracer: {exc}") from exc
        self.root_tid = root_tid
        self.target_cgroup_id = target_cgroup_id
        self.target_cgroup_level = target_cgroup_level
        self.kprobe_symbol = symbol.decode(errors="replace")

    def _record_event(self, _cpu: int, data: Any, _size: int) -> None:
        assert self._bpf is not None
        event = self._bpf["events"].event(data)
        self._events.append(
            SchedulerEvent(
                timestamp_ns=int(event.timestamp_ns),
                cpu=int(event.cpu),
                tid=int(event.tid),
                entering=bool(event.entering),
            )
        )

    def _record_lost_events(self, *args: int) -> None:
        """Record perf-buffer drops across BCC callback signature variants."""
        if args:
            self._lost_events += int(args[-1])

    def poll(self, timeout_ms: int = 100) -> None:
        if self._bpf is None:
            raise SchedulerTraceError("scheduler tracer has not been started")
        self._bpf.perf_buffer_poll(timeout=timeout_ms)

    def timestamp_ns(self) -> int:
        """Return the clock used by BPF ``bpf_ktime_get_ns`` timestamps."""
        return time.monotonic_ns()

    def finish(self, *, window_start_ns: int, window_end_ns: int) -> dict[str, Any]:
        if self._bpf is None:
            raise SchedulerTraceError("scheduler tracer has not been started")
        self.poll(timeout_ms=0)
        report = summarize_scheduler_events(
            self._events,
            window_start_ns=window_start_ns,
            window_end_ns=window_end_ns,
            lost_events=self._lost_events,
            method="ebpf-bcc-finish-task-switch-cgroup-subtree",
            scope=("all tasks scheduled from the measured container cgroup " "or descendant cgroups"),
        )
        report["root_tid"] = self.root_tid
        report["backend"] = "bcc"
        report["perf_buffer_pages"] = self._buffer_pages
        report["target_cgroup_id"] = self.target_cgroup_id
        report["target_cgroup_level"] = self.target_cgroup_level
        report["kprobe_symbol"] = self.kprobe_symbol
        self.close()
        return report

    def write_events(self, path: Path) -> None:
        with path.open("w") as output:
            for event in sorted(self._events, key=lambda item: item.timestamp_ns):
                output.write(
                    json.dumps(
                        {
                            "timestamp_ns": event.timestamp_ns,
                            "cpu": event.cpu,
                            "tid": event.tid,
                            "entering": event.entering,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )

    def close(self) -> None:
        if self._bpf is None:
            return
        cleanup = getattr(self._bpf, "cleanup", None)
        if cleanup is not None:
            cleanup()
        self._bpf = None
