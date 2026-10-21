#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Cgroup-scoped portable hardware counters collected with ``perf stat``.

The collector is deliberately separate from scheduler tracing.  It starts
disabled, is enabled and disabled through perf's acknowledged control pipe,
and is restricted to the measured container cgroup and configured CPU set.
Unsupported or multiplexed events are retained as explicit invalid evidence;
they are never reported as zero.
"""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import select
import signal
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable

HARDWARE_PERF_SCHEMA_VERSION = "cgroup-perf-hardware.v1"
PORTABLE_HARDWARE_EVENTS = (
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
)
MIN_RUNNING_RATIO = 0.95
PERF_EVENT_SOURCE_ROOT = Path("/sys/bus/event_source/devices")


class HardwarePerfError(RuntimeError):
    """Raised when perf collection itself cannot be trusted."""


def collect_hardware_perf_provenance(
    *,
    perf: str,
    event_source_root: Path = PERF_EVENT_SOURCE_ROOT,
    command_runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    kernel_release: str | None = None,
    machine: str | None = None,
) -> dict[str, Any]:
    """Collect compact host/perf identity without dumping volatile sysfs data."""
    runner = command_runner or subprocess.run
    errors: list[str] = []
    perf_version: str | None = None
    try:
        completed = runner(
            [perf, "--version"],
            text=True,
            capture_output=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        errors.append(f"unable to query perf version: {type(exc).__name__}: {exc}")
    else:
        output = (completed.stdout or completed.stderr or "").strip()
        if completed.returncode == 0 and output:
            perf_version = output.splitlines()[0][:512]
        else:
            diagnostic = output.splitlines()[0][:512] if output else "no diagnostic"
            errors.append(f"perf --version exited {completed.returncode}: {diagnostic}")

    event_source_names: list[str] = []
    try:
        event_source_names = sorted(entry.name for entry in event_source_root.iterdir() if entry.is_dir())
    except OSError as exc:
        errors.append("unable to enumerate perf event sources: " f"{type(exc).__name__}: {exc}")

    return {
        "perf_binary": perf,
        "perf_version": perf_version,
        "kernel_release": kernel_release or platform.release(),
        "machine": machine or platform.machine(),
        "event_source_root": str(event_source_root),
        "event_source_names": event_source_names,
        "provenance_valid": not errors,
        "provenance_errors": errors,
    }


@dataclass(frozen=True)
class PerfControlBounds:
    """Host-monotonic bounds around one acknowledged perf control action."""

    action: str
    request_monotonic_ns: int
    acknowledgement_monotonic_ns: int

    def __post_init__(self) -> None:
        if self.action not in {"enable", "disable"}:
            raise HardwarePerfError(f"unsupported perf control action: {self.action!r}")
        if (
            not isinstance(self.request_monotonic_ns, int)
            or isinstance(self.request_monotonic_ns, bool)
            or self.request_monotonic_ns < 0
            or not isinstance(self.acknowledgement_monotonic_ns, int)
            or isinstance(self.acknowledgement_monotonic_ns, bool)
            or self.acknowledgement_monotonic_ns < self.request_monotonic_ns
        ):
            raise HardwarePerfError("perf control bounds are not monotonic")

    @property
    def uncertainty_ns(self) -> int:
        return self.acknowledgement_monotonic_ns - self.request_monotonic_ns

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "request_monotonic_ns": self.request_monotonic_ns,
            "acknowledgement_monotonic_ns": self.acknowledgement_monotonic_ns,
            "uncertainty_ns": self.uncertainty_ns,
            "clock": "CLOCK_MONOTONIC",
        }


@dataclass(frozen=True)
class PerfCounterResult:
    """One requested perf event, including explicit invalid states."""

    event: str
    namespace: str
    status: str
    value: int | float | None
    unit: str | None
    time_enabled_ns: int | None
    time_running_ns: int | None
    time_running_fraction: float | None
    valid: bool
    invalid_reason: str | None
    observed_cgroup: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "namespace": self.namespace,
            "status": self.status,
            "value": self.value,
            "unit": self.unit,
            "time_enabled_ns": self.time_enabled_ns,
            "time_running_ns": self.time_running_ns,
            "time_running_fraction": self.time_running_fraction,
            "valid": self.valid,
            "invalid_reason": self.invalid_reason,
            "observed_cgroup": self.observed_cgroup,
        }


def _number(raw: str) -> int | float:
    value = float(raw)
    if not math.isfinite(value) or value < 0:
        raise ValueError(raw)
    integer = int(value)
    return integer if value == integer else value


def _event_namespace(event: str) -> str:
    # Only the deliberately reviewed generic set may flow into cross-ISA
    # portable comparisons. Arbitrary perf aliases/raw encodings remain
    # visible, but cannot be mistaken for semantically portable evidence.
    return "portable" if event in PORTABLE_HARDWARE_EVENTS else "custom-unclassified"


def _invalid_counter(
    event: str,
    *,
    status: str,
    reason: str,
    observed_cgroup: str | None = None,
) -> PerfCounterResult:
    return PerfCounterResult(
        event=event,
        namespace=_event_namespace(event),
        status=status,
        value=None,
        unit=None,
        time_enabled_ns=None,
        time_running_ns=None,
        time_running_fraction=None,
        valid=False,
        invalid_reason=reason,
        observed_cgroup=observed_cgroup,
    )


def _normalized_cgroup(value: str) -> str:
    """Normalize perf's cgroup display without weakening scope validation."""
    stripped = value.strip()
    if not stripped:
        raise ValueError("empty cgroup")
    return "/" + stripped.strip("/") if stripped != "/" else "/"


def parse_perf_stat_csv(
    text: str,
    *,
    requested_events: Iterable[str] = PORTABLE_HARDWARE_EVENTS,
    minimum_running_ratio: float = MIN_RUNNING_RATIO,
    expected_cgroup: str | None = None,
) -> tuple[list[PerfCounterResult], list[str]]:
    """Parse stable ``perf stat -x ';'`` output without hiding bad counters.

    ``perf stat -G`` inserts a cgroup field between the event and runtime
    fields.  The collector always supplies ``expected_cgroup`` so that this
    parser both uses the correct layout and verifies attribution.  Standalone
    callers may omit it when parsing output produced without ``-G``.
    """
    if not 0 < minimum_running_ratio <= 1:
        raise HardwarePerfError("minimum running ratio must be in (0, 1]")
    requested = tuple(requested_events)
    if (
        not requested
        or any(not isinstance(event, str) or not event.strip() for event in requested)
        or len(set(requested)) != len(requested)
    ):
        raise HardwarePerfError("requested perf events must be nonempty and unique")
    normalized_expected_cgroup: str | None = None
    if expected_cgroup is not None:
        try:
            normalized_expected_cgroup = _normalized_cgroup(expected_cgroup)
        except (AttributeError, ValueError) as exc:
            raise HardwarePerfError("expected perf cgroup must be a nonempty path") from exc
    parsed: dict[str, PerfCounterResult] = {}
    anomalies: list[str] = []
    for line_number, row in enumerate(csv.reader(text.splitlines(), delimiter=";"), 1):
        if not row or not any(field.strip() for field in row):
            continue
        if row[0].lstrip().startswith("#"):
            continue
        required_fields = 6 if normalized_expected_cgroup is not None else 5
        if len(row) < required_fields:
            anomalies.append(f"perf stat row {line_number} has fewer than {required_fields} fields")
            continue
        raw_value, raw_unit, raw_event = (field.strip() for field in row[:3])
        observed_cgroup: str | None = None
        if normalized_expected_cgroup is None:
            raw_enabled, raw_percent = (field.strip() for field in row[3:5])
        else:
            raw_cgroup, raw_enabled, raw_percent = (field.strip() for field in row[3:6])
            try:
                observed_cgroup = _normalized_cgroup(raw_cgroup)
            except ValueError:
                observed_cgroup = raw_cgroup or None
        event = raw_event.split(":", 1)[0]
        if event not in requested:
            continue
        if normalized_expected_cgroup is not None and observed_cgroup != normalized_expected_cgroup:
            reason = f"perf attributed {event} to cgroup {raw_cgroup!r}, expected " f"{expected_cgroup!r}"
            anomalies.append(reason)
            parsed[event] = _invalid_counter(
                event,
                status="scope_mismatch",
                reason=reason,
                observed_cgroup=observed_cgroup,
            )
            continue
        if event in parsed:
            reason = f"perf stat emitted duplicate event {event!r}"
            anomalies.append(reason)
            parsed[event] = replace(
                parsed[event],
                status="duplicate",
                valid=False,
                invalid_reason=reason,
            )
            continue
        if raw_value in {"<not supported>", "<not counted>"}:
            status = "unsupported" if raw_value == "<not supported>" else "not_counted"
            parsed[event] = _invalid_counter(
                event,
                status=status,
                reason=f"perf reported {raw_value} for {event}",
                observed_cgroup=observed_cgroup,
            )
            continue
        try:
            value = _number(raw_value)
            enabled_ns = int(raw_enabled)
            running_fraction = float(raw_percent) / 100
        except ValueError:
            anomalies.append(f"perf stat emitted malformed values for {event!r}")
            parsed[event] = _invalid_counter(
                event,
                status="malformed",
                reason="perf emitted a malformed count or runtime field",
                observed_cgroup=observed_cgroup,
            )
            continue
        if enabled_ns <= 0 or not math.isfinite(running_fraction) or not 0 <= running_fraction <= 1:
            anomalies.append(f"perf stat emitted invalid runtime metadata for {event!r}")
            parsed[event] = _invalid_counter(
                event,
                status="malformed",
                reason="perf emitted invalid enabled/running time metadata",
                observed_cgroup=observed_cgroup,
            )
            continue
        running_ns = round(enabled_ns * running_fraction)
        valid = running_fraction >= minimum_running_ratio
        parsed[event] = PerfCounterResult(
            event=event,
            namespace=_event_namespace(event),
            status="valid" if valid else "multiplexed",
            value=value,
            unit=raw_unit or None,
            time_enabled_ns=enabled_ns,
            time_running_ns=running_ns,
            time_running_fraction=round(running_fraction, 9),
            valid=valid,
            invalid_reason=(
                None
                if valid
                else (
                    f"time_running/time_enabled {running_fraction:.6f} is below "
                    f"the required {minimum_running_ratio:.6f}"
                )
            ),
            observed_cgroup=observed_cgroup,
        )
    results: list[PerfCounterResult] = []
    for event in requested:
        results.append(
            parsed.get(event)
            or _invalid_counter(
                event,
                status="missing",
                reason="perf stat output omitted the requested event",
            )
        )
    return results, anomalies


class PerfCgroupHardwareProfiler:
    """Collect portable PMU counters for one already-created cgroup."""

    def __init__(
        self,
        *,
        output_dir: Path,
        cpuset_cpus: str | None,
        perf: str = "perf",
        events: tuple[str, ...] = PORTABLE_HARDWARE_EVENTS,
        attach_timeout_sec: float = 20.0,
        minimum_running_ratio: float = MIN_RUNNING_RATIO,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        provenance_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        if attach_timeout_sec <= 0:
            raise HardwarePerfError("perf attach timeout must be positive")
        if (
            not events
            or any(not isinstance(event, str) or not event.strip() for event in events)
            or len(set(events)) != len(events)
        ):
            raise HardwarePerfError("perf event list must be nonempty and unique")
        if not 0 < minimum_running_ratio <= 1:
            raise HardwarePerfError("minimum running ratio must be in (0, 1]")
        self.output_dir = output_dir
        self.cpuset_cpus = cpuset_cpus
        self.perf = perf
        self.events = events
        self.attach_timeout_sec = attach_timeout_sec
        self.minimum_running_ratio = minimum_running_ratio
        self.clock_ns = clock_ns
        self.provenance_provider = provenance_provider or (lambda: collect_hardware_perf_provenance(perf=self.perf))
        self._stats_path = output_dir / "hardware_perf_stat.csv"
        self._stderr_path = output_dir / "hardware_perf.stderr.txt"
        self._process: subprocess.Popen[bytes] | None = None
        self._control_write_fd: int | None = None
        self._ack_read_fd: int | None = None
        self._ack_buffer = b""
        self._perf_cgroup: str | None = None
        self._attach_ready_bounds: PerfControlBounds | None = None
        self._enable_bounds: PerfControlBounds | None = None
        self._disable_bounds: PerfControlBounds | None = None
        self._enabled = False
        self._provenance: dict[str, Any] | None = None

    @property
    def stats_path(self) -> Path:
        """Raw perf-stat output path, including after an early failure."""
        return self._stats_path

    @property
    def stderr_path(self) -> Path:
        """Raw perf diagnostic path, including after an early failure."""
        return self._stderr_path

    @property
    def stderr_text(self) -> str:
        return self._stderr()

    @property
    def provenance(self) -> dict[str, Any]:
        """Compact collected provenance, retained after close or failure."""
        return dict(self._provenance or {})

    def _stderr(self) -> str:
        return self._stderr_path.read_text(errors="replace").strip() if self._stderr_path.is_file() else ""

    def _close_fds(self) -> None:
        for fd in (self._control_write_fd, self._ack_read_fd):
            if fd is not None:
                try:
                    os.close(fd)
                except OSError:
                    pass
        self._control_write_fd = None
        self._ack_read_fd = None

    def _control(self, action: str) -> PerfControlBounds:
        if action not in {"enable", "disable"}:
            raise HardwarePerfError(f"unsupported perf control action: {action!r}")
        if self._process is None or self._control_write_fd is None or self._ack_read_fd is None:
            raise HardwarePerfError("perf hardware profiler has not been started")
        if self._process.poll() is not None:
            raise HardwarePerfError(f"perf hardware profiler exited before {action}: {self._stderr()}")
        request_ns = self.clock_ns()
        try:
            os.write(self._control_write_fd, f"{action}\n".encode())
        except OSError as exc:
            raise HardwarePerfError(f"unable to send perf {action}: {exc}") from exc
        deadline = time.monotonic() + self.attach_timeout_sec
        while True:
            if b"\n" in self._ack_buffer:
                response, self._ack_buffer = self._ack_buffer.split(b"\n", 1)
                if response.strip(b"\x00 \t\r") != b"ack":
                    raise HardwarePerfError(f"unexpected perf control response: {response!r}")
                return PerfControlBounds(
                    action=action,
                    request_monotonic_ns=request_ns,
                    acknowledgement_monotonic_ns=self.clock_ns(),
                )
            if self._process.poll() is not None:
                raise HardwarePerfError(
                    f"perf hardware profiler exited before {action} acknowledgement: " f"{self._stderr()}"
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise HardwarePerfError(f"timed out waiting for perf {action} acknowledgement")
            readable, _writable, _exceptional = select.select([self._ack_read_fd], [], [], min(remaining, 0.1))
            if readable:
                chunk = os.read(self._ack_read_fd, 4096)
                if not chunk:
                    raise HardwarePerfError(f"perf control pipe closed before {action} acknowledgement")
                self._ack_buffer += chunk

    def attach(self, *, cgroup_path: Path) -> None:
        """Launch perf disabled and wait for an acknowledged readiness probe."""
        if self._process is not None:
            raise HardwarePerfError("perf hardware profiler is already attached")
        self._attach_ready_bounds = None
        self._enable_bounds = None
        self._disable_bounds = None
        self._enabled = False
        try:
            self._provenance = dict(self.provenance_provider())
        except Exception as exc:
            self._provenance = {
                "perf_binary": self.perf,
                "perf_version": None,
                "kernel_release": platform.release(),
                "machine": platform.machine(),
                "event_source_root": str(PERF_EVENT_SOURCE_ROOT),
                "event_source_names": [],
                "provenance_valid": False,
                "provenance_errors": ["hardware provenance provider failed: " f"{type(exc).__name__}: {exc}"],
            }
        try:
            relative = cgroup_path.relative_to("/sys/fs/cgroup")
        except ValueError as exc:
            raise HardwarePerfError(f"perf cgroup is outside /sys/fs/cgroup: {cgroup_path}") from exc
        self._perf_cgroup = "/" + relative.as_posix()
        control_read_fd, control_write_fd = os.pipe()
        ack_read_fd, ack_write_fd = os.pipe()
        command = [
            self.perf,
            "stat",
            "-D",
            "-1",
            "--control",
            f"fd:{control_read_fd},{ack_write_fd}",
            "-x",
            ";",
            "--no-big-num",
            # Generic hardware aliases expand once per core PMU on hybrid
            # x86 and big.LITTLE systems.  Merge those rows back to the
            # requested portable event name so the result remains one
            # cgroup-scoped aggregate rather than silently disappearing from
            # the parser as cpu_core/... and cpu_atom/... rows.
            "--hybrid-merge",
            "-o",
            str(self._stats_path),
        ]
        for event in self.events:
            command.extend(["-e", event])
        command.extend(["-G", self._perf_cgroup, "-a"])
        if self.cpuset_cpus is not None:
            command.extend(["-C", self.cpuset_cpus])
        try:
            stderr = self._stderr_path.open("wb")
            try:
                self._process = subprocess.Popen(
                    command,
                    stdout=subprocess.DEVNULL,
                    stderr=stderr,
                    pass_fds=(control_read_fd, ack_write_fd),
                )
            finally:
                stderr.close()
        except OSError as exc:
            os.close(control_read_fd)
            os.close(control_write_fd)
            os.close(ack_read_fd)
            os.close(ack_write_fd)
            raise HardwarePerfError(f"unable to start perf hardware profiler: {exc}") from exc
        os.close(control_read_fd)
        os.close(ack_write_fd)
        self._control_write_fd = control_write_fd
        self._ack_read_fd = ack_read_fd
        try:
            # --delay=-1 starts disabled.  An acknowledged idempotent disable
            # proves perf has opened its events and entered the control loop,
            # keeping process startup outside the measured workload window.
            self._attach_ready_bounds = self._control("disable")
        except HardwarePerfError:
            self.close()
            raise

    def enable(self) -> PerfControlBounds:
        """Begin counting and return the host-monotonic request/ACK bounds."""
        if self._process is None or self._perf_cgroup is None:
            raise HardwarePerfError("perf hardware profiler has not been attached")
        if self._enabled or self._enable_bounds is not None:
            raise HardwarePerfError("perf hardware profiler is already enabled")
        bounds = self._control("enable")
        self._enable_bounds = bounds
        self._enabled = True
        return bounds

    def start(self, *, cgroup_path: Path, enable: bool = True) -> None:
        """Backward-compatible attach, optionally followed by explicit enable."""
        self.attach(cgroup_path=cgroup_path)
        if enable:
            self.enable()

    def poll(self, timeout_ms: int = 0) -> None:
        if self._process is None:
            raise HardwarePerfError("perf hardware profiler has not been started")
        if self._process.poll() is not None:
            raise HardwarePerfError("perf hardware profiler exited before measurement completed: " f"{self._stderr()}")
        if timeout_ms > 0:
            time.sleep(timeout_ms / 1000)

    def finish(self, *, window_start_ns: int, window_end_ns: int) -> dict[str, Any]:
        if self._process is None or self._perf_cgroup is None:
            raise HardwarePerfError("perf hardware profiler has not been started")
        if not self._enabled or self._enable_bounds is None:
            raise HardwarePerfError("perf hardware profiler was never enabled")
        if window_end_ns <= window_start_ns:
            raise HardwarePerfError("hardware profiling window ends before it starts")
        self._disable_bounds = self._control("disable")
        self._enabled = False
        if self._process.poll() is None:
            self._process.send_signal(signal.SIGINT)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired as exc:
            self._process.kill()
            self._process.wait()
            raise HardwarePerfError("timed out stopping perf hardware profiler") from exc
        self._close_fds()
        if self._process.returncode not in (0, -signal.SIGINT):
            raise HardwarePerfError(f"perf hardware profiler failed while stopping: {self._stderr()}")
        if not self._stats_path.is_file():
            raise HardwarePerfError("perf hardware profiler produced no stat output")
        counters, anomalies = parse_perf_stat_csv(
            self._stats_path.read_text(errors="replace"),
            requested_events=self.events,
            minimum_running_ratio=self.minimum_running_ratio,
            expected_cgroup=self._perf_cgroup,
        )
        assert self._attach_ready_bounds is not None
        assert self._disable_bounds is not None
        enable_bounds = self._enable_bounds
        disable_bounds = self._disable_bounds
        outer_window_start_ns = enable_bounds.request_monotonic_ns
        outer_window_end_ns = disable_bounds.acknowledgement_monotonic_ns
        definite_start = enable_bounds.acknowledgement_monotonic_ns
        definite_end = disable_bounds.request_monotonic_ns
        definite_window_start_ns: int | None = definite_start
        definite_window_end_ns: int | None = definite_end
        if outer_window_end_ns <= outer_window_start_ns:
            raise HardwarePerfError("perf control window did not have positive duration")
        if definite_end < definite_start:
            anomalies.append("perf enable/disable acknowledgement bounds overlap; no definite active window")
            definite_window_start_ns = None
            definite_window_end_ns = None
        return {
            "schema_version": HARDWARE_PERF_SCHEMA_VERSION,
            "method": "perf-stat-cgroup-cpu-wide",
            "scope": "measured container cgroup on configured logical CPUs",
            "perf_cgroup": self._perf_cgroup,
            "cpuset_cpus": self.cpuset_cpus,
            # The outer window contains the complete counter-active interval;
            # endpoint uncertainty is the corresponding request-to-ACK span.
            # Caller lifecycle markers remain provenance, not asserted perf
            # activation timestamps.
            "window_start_ns": outer_window_start_ns,
            "window_end_ns": outer_window_end_ns,
            "window_duration_sec": round((outer_window_end_ns - outer_window_start_ns) / 1_000_000_000, 9),
            "window_start_alignment_error_ns": enable_bounds.uncertainty_ns,
            "window_end_alignment_error_ns": disable_bounds.uncertainty_ns,
            "requested_lifecycle_window_start_ns": window_start_ns,
            "requested_lifecycle_window_end_ns": window_end_ns,
            "definite_active_window_start_ns": definite_window_start_ns,
            "definite_active_window_end_ns": definite_window_end_ns,
            "control_bounds": {
                "attach_ready": self._attach_ready_bounds.to_dict(),
                "enable": enable_bounds.to_dict(),
                "disable": disable_bounds.to_dict(),
            },
            "minimum_running_ratio": self.minimum_running_ratio,
            "hybrid_pmu_counts_merged": True,
            "provenance": dict(self._provenance or {}),
            "counters": [counter.to_dict() for counter in counters],
            "anomalies": anomalies,
            "instrumentation_valid": not anomalies,
            "all_requested_metrics_valid": not anomalies and all(counter.valid for counter in counters),
            "stats_path": str(self.stats_path),
            "stderr_path": str(self.stderr_path),
        }

    def write_report(self, report: dict[str, Any], path: Path) -> None:
        path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")

    def close(self) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
        self._close_fds()
        self._process = None
        self._enabled = False
        self._attach_ready_bounds = None
        self._enable_bounds = None
        self._disable_bounds = None
