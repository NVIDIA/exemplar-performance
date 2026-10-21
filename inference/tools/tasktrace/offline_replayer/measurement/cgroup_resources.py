# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Task-scoped cgroup-v2 CPU and memory time-series sampling.

The sampler runs in the host profiler, outside the measured container cgroup.
``memory.current`` is the total memory charged to the cgroup subtree; it is not
process RSS and intentionally includes file cache and kernel memory.  Cumulative
``cpu.stat`` counters are sampled on the same clock so reports can show memory
and interval-average logical-core demand on one aligned timeline.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

TIMESERIES_SCHEMA_VERSION = "container-cgroup-resource-timeseries.v1"
MEMORY_REPORT_SCHEMA_VERSION = "container-cgroup-memory.v2"
MEMORY_SOURCE_SNAPSHOT_SCHEMA_VERSION = "container-cgroup-memory-source-snapshots.v1"
ANCESTOR_CPU_SOURCE_SNAPSHOT_SCHEMA_VERSION = "container-cgroup-ancestor-cpu-source-snapshots.v1"
MEMORY_STAT_SAMPLE_KEYS = (
    "anon",
    "file",
    "kernel",
    "kernel_stack",
    "pagetables",
    "percpu",
    "sock",
    "shmem",
    "zswap",
)
REQUIRED_MEMORY_EVENT_KEYS = (
    "high",
    "max",
    "oom",
    "oom_kill",
    "oom_group_kill",
)
ANCESTOR_CONSTRAINT_EVENT_KEYS = REQUIRED_MEMORY_EVENT_KEYS
REQUIRED_ANCESTOR_CPU_STAT_KEYS = (
    "usage_usec",
    "user_usec",
    "system_usec",
    "nr_periods",
    "nr_throttled",
    "throttled_usec",
)
ANCESTOR_CPU_THROTTLE_KEYS = (
    "nr_throttled",
    "throttled_usec",
)
PRODUCTION_CGROUP_ROOT = Path("/sys/fs/cgroup")
MEMORY_SOURCE_FILES = frozenset(
    {
        "memory.current",
        "memory.peak",
        "memory.swap.current",
        "memory.swap.peak",
        "memory.high",
        "memory.max",
        "memory.swap.high",
        "memory.swap.max",
        "memory.events",
        "memory.events.local",
        "memory.pressure",
        "memory.stat",
    }
)


class CgroupResourceSamplingError(RuntimeError):
    """Raised when a requested cgroup resource measurement is unusable."""


def parse_flat_keyed(text: str, *, source: str) -> dict[str, int]:
    """Parse a cgroup flat-keyed integer file without assuming key order."""
    values: dict[str, int] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 2:
            raise CgroupResourceSamplingError(f"invalid {source} row: {line!r}")
        key, raw_value = fields
        try:
            value = int(raw_value)
        except ValueError as exc:
            raise CgroupResourceSamplingError(f"invalid {source} value for {key!r}: {raw_value!r}") from exc
        if value < 0:
            raise CgroupResourceSamplingError(f"negative {source} value for {key!r}: {value}")
        values[key] = value
    return values


def parse_pressure(text: str, *, source: str = "memory.pressure") -> dict[str, dict[str, float | int]]:
    """Parse PSI rows while preserving future numeric fields."""
    result: dict[str, dict[str, float | int]] = {}
    for line in text.splitlines():
        fields = line.split()
        if not fields:
            continue
        kind = fields[0]
        row: dict[str, float | int] = {}
        for token in fields[1:]:
            key, separator, raw_value = token.partition("=")
            if not separator:
                raise CgroupResourceSamplingError(f"invalid {source} token: {token!r}")
            try:
                row[key] = int(raw_value) if key == "total" else float(raw_value)
            except ValueError as exc:
                raise CgroupResourceSamplingError(f"invalid {source} value for {kind}.{key}: {raw_value!r}") from exc
        result[kind] = row
    return result


def parse_cpu_max(text: str, *, source: str = "cpu.max") -> tuple[int | None, int, str]:
    """Parse one cgroup-v2 CPU bandwidth limit without losing its period."""
    fields = text.split()
    if len(fields) != 2:
        raise CgroupResourceSamplingError(f"invalid {source} value: expected quota and period, got {text.strip()!r}")
    raw_quota, raw_period = fields
    try:
        period_usec = int(raw_period)
    except ValueError as exc:
        raise CgroupResourceSamplingError(f"invalid {source} period: {raw_period!r}") from exc
    if period_usec <= 0:
        raise CgroupResourceSamplingError(f"nonpositive {source} period: {period_usec}")
    if raw_quota == "max":
        return None, period_usec, "unlimited"
    try:
        quota_usec = int(raw_quota)
    except ValueError as exc:
        raise CgroupResourceSamplingError(f"invalid {source} quota: {raw_quota!r}") from exc
    if quota_usec <= 0:
        raise CgroupResourceSamplingError(f"nonpositive {source} quota: {quota_usec}")
    return quota_usec, period_usec, "finite"


def _counter_delta(
    before: Mapping[str, int],
    after: Mapping[str, int],
    *,
    source: str,
) -> tuple[dict[str, int], list[str]]:
    deltas: dict[str, int] = {}
    anomalies: list[str] = []
    for key in sorted(set(before) | set(after)):
        start = before.get(key, 0)
        end = after.get(key, 0)
        if end < start:
            anomalies.append(f"{source} counter {key!r} moved backwards: {start} -> {end}")
            continue
        deltas[key] = end - start
    return deltas, anomalies


@dataclass(frozen=True)
class CgroupResourceSample:
    """One same-clock observation of cgroup CPU and memory counters."""

    timestamp_ns: int
    acquisition_start_ns: int
    acquisition_end_ns: int
    memory_current_bytes: int
    memory_swap_current_bytes: int | None
    memory_stat: dict[str, int]
    cpu_usage_usec: int | None
    cpu_user_usec: int | None
    cpu_system_usec: int | None


def _time_weighted_average_bytes(samples: list[CgroupResourceSample]) -> int | None:
    """Trapezoidal sampled average; it is descriptive, not an exact integral."""
    if not samples:
        return None
    if len(samples) == 1:
        return samples[0].memory_current_bytes
    weighted_twice_byte_ns = 0
    duration_ns = samples[-1].timestamp_ns - samples[0].timestamp_ns
    if duration_ns <= 0:
        return samples[-1].memory_current_bytes
    for previous, current in zip(samples, samples[1:], strict=False):
        interval_ns = current.timestamp_ns - previous.timestamp_ns
        if interval_ns <= 0:
            continue
        weighted_twice_byte_ns += (previous.memory_current_bytes + current.memory_current_bytes) * interval_ns
    denominator = 2 * duration_ns
    return (weighted_twice_byte_ns + denominator // 2) // denominator


def summarize_sample_window(
    samples: list[CgroupResourceSample],
    *,
    start_ns: int,
    end_ns: int,
) -> dict[str, Any]:
    """Summarize samples bounded by forced phase-marker observations."""
    selected = [sample for sample in samples if start_ns <= sample.timestamp_ns <= end_ns]
    if not selected:
        return {
            "sample_count": 0,
            "duration_sec": 0.0,
            "start_bytes": None,
            "end_bytes": None,
            "sampled_peak_bytes": None,
            "sampled_peak_elapsed_sec": None,
            "sampled_time_weighted_average_bytes": None,
            "sampled_peak_swap_bytes": None,
            "max_sample_gap_sec": None,
            "sampled_cpu_usage_sec": None,
            "sampled_cpu_average_logical_cores": None,
        }
    peak = max(selected, key=lambda sample: sample.memory_current_bytes)
    gaps = [
        (current.timestamp_ns - previous.timestamp_ns) / 1_000_000_000
        for previous, current in zip(selected, selected[1:], strict=False)
    ]
    swap_values = [
        sample.memory_swap_current_bytes for sample in selected if sample.memory_swap_current_bytes is not None
    ]
    cpu_usage_usec = None
    cpu_average_logical_cores = None
    if (
        selected[0].cpu_usage_usec is not None
        and selected[-1].cpu_usage_usec is not None
        and selected[-1].cpu_usage_usec >= selected[0].cpu_usage_usec
    ):
        cpu_usage_usec = selected[-1].cpu_usage_usec - selected[0].cpu_usage_usec
        window_duration_sec = (end_ns - start_ns) / 1_000_000_000
        if window_duration_sec > 0:
            cpu_average_logical_cores = (cpu_usage_usec / 1_000_000) / window_duration_sec
    return {
        "sample_count": len(selected),
        "duration_sec": round((end_ns - start_ns) / 1_000_000_000, 9),
        "start_bytes": selected[0].memory_current_bytes,
        "end_bytes": selected[-1].memory_current_bytes,
        "sampled_peak_bytes": peak.memory_current_bytes,
        "sampled_peak_elapsed_sec": round((peak.timestamp_ns - start_ns) / 1_000_000_000, 9),
        "sampled_time_weighted_average_bytes": _time_weighted_average_bytes(selected),
        "sampled_peak_swap_bytes": max(swap_values) if swap_values else None,
        "max_sample_gap_sec": round(max(gaps), 9) if gaps else 0.0,
        "sampled_cpu_usage_sec": (round(cpu_usage_usec / 1_000_000, 9) if cpu_usage_usec is not None else None),
        "sampled_cpu_average_logical_cores": (
            round(cpu_average_logical_cores, 9) if cpu_average_logical_cores is not None else None
        ),
        "memory_stat_at_sampled_peak": dict(peak.memory_stat),
    }


class AncestorCpuObserver:
    """Gate-time ancestor CPU policy/accounting observer independent of memory.

    ``start()`` is intended for the pre-work gate and ``finish()`` for the
    controlled finish gate. The observer retains the exact ``cpu.max`` and
    ``cpu.stat`` text it parses, so both profiled and coordinated-baseline CPU
    paths can publish the same fail-closed contract without enabling memory
    sampling.
    """

    def __init__(
        self,
        *,
        cgroup_path: Path,
        cgroup_root: Path,
        clock_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.cgroup_path = cgroup_path.resolve()
        self.cgroup_root = cgroup_root.resolve()
        try:
            self.cgroup_path.relative_to(self.cgroup_root)
        except ValueError as exc:
            raise CgroupResourceSamplingError(
                f"cgroup path {self.cgroup_path} is outside configured " f"cgroup root {self.cgroup_root}"
            ) from exc
        self.clock_ns = clock_ns
        self.ancestor_paths = self._ordered_ancestor_paths()
        self._before: list[dict[str, Any]] = []
        self._raw_reads: list[dict[str, Any]] = []
        self._phase: str | None = None
        self._started = False
        self._finished = False

    def _ordered_ancestor_paths(self) -> tuple[Path, ...]:
        if self.cgroup_path == self.cgroup_root:
            return ()
        ancestors: list[Path] = []
        current = self.cgroup_path.parent
        while True:
            ancestors.append(current)
            if current == self.cgroup_root:
                return tuple(ancestors)
            parent = current.parent
            if parent == current:
                raise CgroupResourceSamplingError(
                    f"unable to reach configured cgroup root {self.cgroup_root} " f"from {self.cgroup_path}"
                )
            current = parent

    def _relative_path(self, path: Path) -> str:
        return path.relative_to(self.cgroup_root).as_posix()

    def _source_scope(self, path: Path) -> str:
        relative = path.parent.resolve().relative_to(self.cgroup_root).as_posix()
        return f"ancestor:{relative or '.'}"

    def _read_source_text(self, path: Path) -> str:
        acquisition_start_ns = self.clock_ns()
        try:
            text = path.read_text()
        except OSError as exc:
            acquisition_end_ns = self.clock_ns()
            self._raw_reads.append(
                {
                    "read_index": len(self._raw_reads),
                    "phase": self._phase,
                    "source_scope": self._source_scope(path),
                    "filename": path.name,
                    "acquisition_start_ns": acquisition_start_ns,
                    "acquisition_end_ns": acquisition_end_ns,
                    "available": False,
                    "raw_text": None,
                    "error_type": type(exc).__name__,
                }
            )
            raise
        acquisition_end_ns = self.clock_ns()
        self._raw_reads.append(
            {
                "read_index": len(self._raw_reads),
                "phase": self._phase,
                "source_scope": self._source_scope(path),
                "filename": path.name,
                "acquisition_start_ns": acquisition_start_ns,
                "acquisition_end_ns": acquisition_end_ns,
                "available": True,
                "raw_text": text,
                "error_type": None,
            }
        )
        return text

    def _read_cpu_max(self, path: Path) -> tuple[int | None, int | None, str]:
        source = path / "cpu.max"
        try:
            text = self._read_source_text(source)
        except OSError:
            return None, None, "unavailable"
        quota_usec, period_usec, state = parse_cpu_max(text, source=str(source))
        return quota_usec, period_usec, state

    def _read_cpu_stat(self, path: Path) -> tuple[dict[str, int], bool]:
        source = path / "cpu.stat"
        try:
            text = self._read_source_text(source)
        except OSError:
            return {}, False
        return parse_flat_keyed(text, source=str(source)), True

    def _snapshot(self, path: Path) -> dict[str, Any]:
        acquisition_start_ns = self.clock_ns()
        errors: list[str] = []
        try:
            quota_usec, period_usec, cpu_max_state = self._read_cpu_max(path)
        except CgroupResourceSamplingError as exc:
            quota_usec, period_usec, cpu_max_state = None, None, "invalid"
            errors.append(str(exc))
        if cpu_max_state == "unavailable":
            if path == self.cgroup_root:
                cpu_max_state = "not-applicable-root"
            else:
                errors.append(f"required ancestor CPU limit is unavailable: {path / 'cpu.max'}")

        try:
            cpu_stat, cpu_stat_available = self._read_cpu_stat(path)
        except CgroupResourceSamplingError as exc:
            cpu_stat, cpu_stat_available = {}, False
            errors.append(str(exc))
        if not cpu_stat_available:
            errors.append(f"required ancestor CPU accounting is unavailable: {path / 'cpu.stat'}")
        else:
            missing_keys = [key for key in REQUIRED_ANCESTOR_CPU_STAT_KEYS if key not in cpu_stat]
            if missing_keys:
                errors.append(f"ancestor {path / 'cpu.stat'} lacks required counters: " + ", ".join(missing_keys))

        acquisition_end_ns = self.clock_ns()
        if acquisition_end_ns < acquisition_start_ns:
            errors.append(
                f"ancestor CPU snapshot clock moved backwards for {path}: "
                f"{acquisition_start_ns} -> {acquisition_end_ns}"
            )
        return {
            "path": self._relative_path(path),
            "acquisition_start_ns": acquisition_start_ns,
            "acquisition_end_ns": acquisition_end_ns,
            "cpu_max_quota_usec": quota_usec,
            "cpu_max_period_usec": period_usec,
            "cpu_max_state": cpu_max_state,
            "cpu_stat": cpu_stat,
            "cpu_stat_available": cpu_stat_available,
            "errors": list(dict.fromkeys(errors)),
        }

    def start(self) -> None:
        if self._started:
            raise CgroupResourceSamplingError("ancestor CPU observation already started")
        self._raw_reads = []
        self._phase = "observer_start"
        self._before = [self._snapshot(path) for path in self.ancestor_paths]
        self._phase = None
        self._started = True

    def finish(self) -> dict[str, Any]:
        if not self._started or self._finished:
            raise CgroupResourceSamplingError("ancestor CPU observation is not active")
        self._phase = "controlled_finish_gate"
        after = [self._snapshot(path) for path in self.ancestor_paths]
        self._phase = None
        self._finished = True
        return self._build_report(after)

    def _build_report(self, after_snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        ancestors: list[dict[str, Any]] = []
        aggregate_evidence_errors: list[str] = []
        aggregate_invalid_reasons: list[str] = []
        for before, after in zip(self._before, after_snapshots, strict=False):
            relative_path = str(before["path"])
            evidence_errors = [
                *[str(error) for error in before.get("errors", [])],
                *[str(error) for error in after.get("errors", [])],
            ]
            if after.get("path") != relative_path:
                evidence_errors.append(
                    f"ancestor CPU chain changed from {relative_path!r} to " f"{after.get('path')!r}"
                )

            cpu_stat_delta, counter_anomalies = _counter_delta(
                before.get("cpu_stat", {}),
                after.get("cpu_stat", {}),
                source=f"ancestor {relative_path!r} cpu.stat",
            )
            evidence_errors.extend(counter_anomalies)
            counter_regression_observed = bool(counter_anomalies)
            before_limit = (
                before.get("cpu_max_state"),
                before.get("cpu_max_quota_usec"),
                before.get("cpu_max_period_usec"),
            )
            after_limit = (
                after.get("cpu_max_state"),
                after.get("cpu_max_quota_usec"),
                after.get("cpu_max_period_usec"),
            )
            cpu_max_changed = before_limit != after_limit
            if cpu_max_changed:
                evidence_errors.append(
                    f"ancestor {relative_path!r} cpu.max changed during observation: "
                    f"{before_limit} -> {after_limit}"
                )
            throttle_deltas = {
                key: cpu_stat_delta.get(key, 0) for key in ANCESTOR_CPU_THROTTLE_KEYS if cpu_stat_delta.get(key, 0) > 0
            }
            finite_limit_present = before.get("cpu_max_state") == "finite" or after.get("cpu_max_state") == "finite"
            evidence_errors = list(dict.fromkeys(evidence_errors))
            invalid_reasons = list(evidence_errors)
            if throttle_deltas:
                rendered = ", ".join(f"{key}={value}" for key, value in sorted(throttle_deltas.items()))
                invalid_reasons.append(
                    f"ancestor cgroup {relative_path!r} recorded CPU throttling "
                    f"({rendered}); CPU performance evidence fails closed"
                )
            if finite_limit_present:
                invalid_reasons.append(
                    f"ancestor cgroup {relative_path!r} has a finite cpu.max; "
                    "unconstrained CPU performance is not proven"
                )
            invalid_reasons = list(dict.fromkeys(invalid_reasons))
            aggregate_evidence_errors.extend(evidence_errors)
            aggregate_invalid_reasons.extend(invalid_reasons)
            ancestors.append(
                {
                    "path": relative_path,
                    "start": before,
                    "finish": after,
                    "cpu_stat_delta": cpu_stat_delta,
                    "cpu_max_changed": cpu_max_changed,
                    "counter_regression_observed": counter_regression_observed,
                    "finite_limit_present": finite_limit_present,
                    "throttle_deltas": throttle_deltas,
                    "evidence_valid": not evidence_errors,
                    "policy_stable": not cpu_max_changed,
                    "throttling_observed": bool(throttle_deltas),
                    "valid_for_cpu_performance": not invalid_reasons,
                    "invalid_reasons": invalid_reasons,
                }
            )
        if len(self._before) != len(after_snapshots):
            message = "ancestor CPU cgroup chain length changed during observation"
            aggregate_evidence_errors.append(message)
            aggregate_invalid_reasons.append(message)
        aggregate_evidence_errors = list(dict.fromkeys(aggregate_evidence_errors))
        aggregate_invalid_reasons = list(dict.fromkeys(aggregate_invalid_reasons))
        return {
            "cgroup_root": str(self.cgroup_root),
            "cgroup_root_cpu_max_applicable": (self.cgroup_root / "cpu.max").exists(),
            "leaf_path": self._relative_path(self.cgroup_path),
            "chain_order": "leaf-parent-to-root",
            "ancestor_count": len(ancestors),
            "ancestors": ancestors,
            "evidence_complete": not aggregate_evidence_errors,
            "limits_stable": not any(ancestor["cpu_max_changed"] for ancestor in ancestors),
            "counter_regression_observed": any(ancestor["counter_regression_observed"] for ancestor in ancestors),
            "finite_limits_present": any(ancestor["finite_limit_present"] for ancestor in ancestors),
            "throttling_observed": any(ancestor["throttling_observed"] for ancestor in ancestors),
            "valid_for_cpu_performance": not aggregate_invalid_reasons,
            "evidence_errors": aggregate_evidence_errors,
            "invalid_reasons": aggregate_invalid_reasons,
            "raw_source_read_count": len(self._raw_reads),
            "meaning": (
                "Every ancestor cgroup is sampled at observer start and controlled "
                "finish. Missing/malformed sources, cpu.max changes, cpu.stat "
                "regressions, finite bandwidth limits, or positive nr_throttled/"
                "throttled_usec deltas fail closed for unconstrained CPU performance."
            ),
        }

    def raw_source_snapshot_payload(self) -> dict[str, Any]:
        if not self._finished:
            raise CgroupResourceSamplingError(
                "ancestor CPU source snapshots are incomplete before observation finishes"
            )
        return {
            "schema_version": ANCESTOR_CPU_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "scope": {
                "kind": "cgroup-v2-ancestor-chain",
                "leaf_path": self._relative_path(self.cgroup_path),
                "path_representation": "relative-to-configured-cgroup-root",
            },
            "phases": ["observer_start", "controlled_finish_gate"],
            "reads": list(self._raw_reads),
            "meaning": (
                "Exact ancestor cpu.max and cpu.stat text consumed at both gates; "
                "derived deltas and validity decisions live in ancestor_cpu."
            ),
        }

    def write_raw_source_snapshots(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                self.raw_source_snapshot_payload(),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )


class CgroupResourceSampler:
    """Poll one cgroup-v2 subtree and preserve aligned CPU/memory evidence."""

    def __init__(
        self,
        *,
        cgroup_path: Path,
        cgroup_root: Path | None = None,
        interval_sec: float = 0.1,
        clock_ns: Callable[[], int] = time.monotonic_ns,
        observe_ancestor_cpu: bool = True,
    ) -> None:
        if interval_sec <= 0:
            raise CgroupResourceSamplingError("sample interval must be positive")
        if not isinstance(observe_ancestor_cpu, bool):
            raise CgroupResourceSamplingError("observe_ancestor_cpu must be boolean")
        self.cgroup_path = cgroup_path
        self._resolved_cgroup_path = cgroup_path.resolve()
        if cgroup_root is None:
            production_root = PRODUCTION_CGROUP_ROOT.resolve()
            try:
                self._resolved_cgroup_path.relative_to(production_root)
            except ValueError:
                # Unit tests and embedders commonly provide a synthetic leaf
                # with no surrounding hierarchy. Treat that leaf as its root
                # unless the caller explicitly supplies the synthetic root.
                resolved_cgroup_root = self._resolved_cgroup_path
            else:
                resolved_cgroup_root = production_root
        else:
            resolved_cgroup_root = cgroup_root.resolve()
        try:
            self._resolved_cgroup_path.relative_to(resolved_cgroup_root)
        except ValueError as exc:
            raise CgroupResourceSamplingError(
                f"cgroup path {self._resolved_cgroup_path} is outside configured " f"cgroup root {resolved_cgroup_root}"
            ) from exc
        self.cgroup_root = resolved_cgroup_root
        self._ancestor_paths = self._ordered_ancestor_paths()
        self.ancestor_cpu_observer = (
            AncestorCpuObserver(
                cgroup_path=self._resolved_cgroup_path,
                cgroup_root=self.cgroup_root,
                clock_ns=clock_ns,
            )
            if observe_ancestor_cpu
            else None
        )
        self.interval_sec = interval_sec
        self.interval_ns = round(interval_sec * 1_000_000_000)
        self.clock_ns = clock_ns
        self.samples: list[CgroupResourceSample] = []
        self.markers: list[dict[str, Any]] = []
        self.anomalies: list[str] = []
        self.missed_sample_deadlines = 0
        self.max_sample_lateness_ns = 0
        self._started = False
        self._finished = False
        self._start_ns: int | None = None
        self._next_sample_ns: int | None = None
        self._events_before: dict[str, int] = {}
        self._events_local_before: dict[str, int] = {}
        self._pressure_before: dict[str, dict[str, float | int]] = {}
        self._stat_before: dict[str, int] = {}
        self._limits: dict[str, Any] = {}
        self._capabilities: dict[str, bool] = {}
        self._peak_at_observer_start: int | None = None
        self._ancestor_before: list[dict[str, Any]] = []
        self._raw_memory_source_phase: str | None = None
        self._raw_memory_source_reads: list[dict[str, Any]] = []

    def _ordered_ancestor_paths(self) -> tuple[Path, ...]:
        """Return the leaf parent through the configured root, in that order."""
        if self._resolved_cgroup_path == self.cgroup_root:
            return ()
        ancestors: list[Path] = []
        current = self._resolved_cgroup_path.parent
        while True:
            if current == self.cgroup_root:
                # The cgroup-v2 root cannot itself be resource controlled and
                # many kernels intentionally omit memory.current/high/max and
                # memory.events.local there.  It is not a hidden finite parent;
                # record that capability below without treating absent root
                # controller files as incomplete evidence. Synthetic roots
                # with controller files remain observable in unit tests.
                if (current / "memory.current").exists():
                    ancestors.append(current)
                return tuple(ancestors)
            ancestors.append(current)
            parent = current.parent
            if parent == current:
                raise CgroupResourceSamplingError(
                    f"unable to reach configured cgroup root {self.cgroup_root} " f"from {self._resolved_cgroup_path}"
                )
            current = parent

    def _read_required_int(self, name: str, *, cgroup_path: Path | None = None) -> int:
        path = (cgroup_path or self.cgroup_path) / name
        try:
            raw = self._read_source_text(path).strip()
            value = int(raw)
        except (OSError, ValueError) as exc:
            raise CgroupResourceSamplingError(f"unable to read required cgroup counter {path}: {exc}") from exc
        if value < 0:
            raise CgroupResourceSamplingError(f"negative cgroup counter in {path}")
        return value

    def _read_optional_int(self, name: str) -> int | None:
        value, _available = self._read_optional_int_with_status(name)
        return value

    def _read_optional_int_with_status(self, name: str, *, cgroup_path: Path | None = None) -> tuple[int | None, bool]:
        path = (cgroup_path or self.cgroup_path) / name
        try:
            raw = self._read_source_text(path).strip()
        except OSError:
            return None, False
        if raw == "max":
            return None, True
        try:
            value = int(raw)
        except ValueError as exc:
            raise CgroupResourceSamplingError(f"invalid integer in cgroup file {path}: {raw!r}") from exc
        if value < 0:
            raise CgroupResourceSamplingError(f"negative cgroup counter in {path}")
        return value, True

    def _read_optional_flat(self, name: str) -> dict[str, int]:
        values, _available = self._read_optional_flat_with_status(name)
        return values

    def _read_optional_flat_with_status(
        self, name: str, *, cgroup_path: Path | None = None
    ) -> tuple[dict[str, int], bool]:
        path = (cgroup_path or self.cgroup_path) / name
        try:
            text = self._read_source_text(path)
        except OSError:
            return {}, False
        return parse_flat_keyed(text, source=str(path)), True

    def _read_pressure(self) -> dict[str, dict[str, float | int]]:
        pressure, _available = self._read_pressure_with_status()
        return pressure

    def _read_pressure_with_status(
        self, *, cgroup_path: Path | None = None
    ) -> tuple[dict[str, dict[str, float | int]], bool]:
        path = (cgroup_path or self.cgroup_path) / "memory.pressure"
        try:
            text = self._read_source_text(path)
        except OSError:
            return {}, False
        return parse_pressure(text, source=str(path)), True

    def _read_limit(self, name: str, *, cgroup_path: Path | None = None) -> tuple[int | None, str]:
        path = (cgroup_path or self.cgroup_path) / name
        try:
            raw = self._read_source_text(path).strip()
        except OSError:
            return None, "unavailable"
        if raw == "max":
            return None, "unlimited"
        try:
            value = int(raw)
        except ValueError as exc:
            raise CgroupResourceSamplingError(f"invalid limit in cgroup file {path}: {raw!r}") from exc
        if value < 0:
            raise CgroupResourceSamplingError(f"negative cgroup limit in {path}")
        return value, "finite"

    def _source_scope(self, path: Path) -> str:
        parent = path.parent.resolve()
        if parent == self._resolved_cgroup_path:
            return "leaf"
        try:
            relative = parent.relative_to(self.cgroup_root).as_posix()
        except ValueError:
            return "outside-configured-cgroup-root"
        return f"ancestor:{relative or '.'}"

    def _read_source_text(self, path: Path) -> str:
        """Read a cgroup file and retain the exact memory source bytes at gates.

        The parsed memory report is useful interpretation, but it is not the
        kernel source evidence. During the observer-start and controlled-finish
        gates this records the exact text consumed by the parser, including
        repeated reads of a source in acquisition order. Paths are represented
        relative to the measured leaf/configured hierarchy so the artifact does
        not depend on a host-specific cgroup mount path.
        """
        try:
            text = path.read_text()
        except OSError as exc:
            if self._raw_memory_source_phase is not None and path.name in MEMORY_SOURCE_FILES:
                self._raw_memory_source_reads.append(
                    {
                        "read_index": len(self._raw_memory_source_reads),
                        "phase": self._raw_memory_source_phase,
                        "source_scope": self._source_scope(path),
                        "filename": path.name,
                        "available": False,
                        "raw_text": None,
                        "error_type": type(exc).__name__,
                    }
                )
            raise
        if self._raw_memory_source_phase is not None and path.name in MEMORY_SOURCE_FILES:
            self._raw_memory_source_reads.append(
                {
                    "read_index": len(self._raw_memory_source_reads),
                    "phase": self._raw_memory_source_phase,
                    "source_scope": self._source_scope(path),
                    "filename": path.name,
                    "available": True,
                    "raw_text": text,
                    "error_type": None,
                }
            )
        return text

    def _ancestor_relative_path(self, path: Path) -> str:
        return path.relative_to(self.cgroup_root).as_posix()

    def _snapshot_ancestor(self, path: Path) -> dict[str, Any]:
        """Read one ancestor without letting bad evidence disappear the run."""
        acquisition_start_ns = self.clock_ns()
        errors: list[str] = []

        try:
            memory_current_bytes = self._read_required_int("memory.current", cgroup_path=path)
        except CgroupResourceSamplingError as exc:
            memory_current_bytes = None
            errors.append(str(exc))

        limits: dict[str, Any] = {}
        for key, filename in (
            ("memory_high", "memory.high"),
            ("memory_max", "memory.max"),
        ):
            try:
                value, state = self._read_limit(filename, cgroup_path=path)
            except CgroupResourceSamplingError as exc:
                value, state = None, "invalid"
                errors.append(str(exc))
            if state == "unavailable":
                errors.append(f"required ancestor cgroup limit is unavailable: {path / filename}")
            limits[f"{key}_bytes"] = value
            limits[f"{key}_state"] = state

        try:
            events_local, events_local_available = self._read_optional_flat_with_status(
                "memory.events.local", cgroup_path=path
            )
        except CgroupResourceSamplingError as exc:
            events_local, events_local_available = {}, False
            errors.append(str(exc))
        missing_event_keys = [key for key in REQUIRED_MEMORY_EVENT_KEYS if key not in events_local]
        if not events_local_available:
            errors.append(f"required ancestor cgroup counter is unavailable: " f"{path / 'memory.events.local'}")
        elif missing_event_keys:
            errors.append(
                f"ancestor {path / 'memory.events.local'} lacks required counters: " + ", ".join(missing_event_keys)
            )

        try:
            pressure, pressure_available = self._read_pressure_with_status(cgroup_path=path)
        except CgroupResourceSamplingError as exc:
            pressure, pressure_available = {}, False
            errors.append(str(exc))
        if pressure_available:
            for kind in ("some", "full"):
                total = pressure.get(kind, {}).get("total")
                if not isinstance(total, int) or isinstance(total, bool) or total < 0:
                    errors.append(f"ancestor {path / 'memory.pressure'} lacks a valid " f"{kind}.total counter")

        acquisition_end_ns = self.clock_ns()
        if acquisition_end_ns < acquisition_start_ns:
            errors.append(
                f"ancestor snapshot clock moved backwards for {path}: "
                f"{acquisition_start_ns} -> {acquisition_end_ns}"
            )
        return {
            "path": self._ancestor_relative_path(path),
            "acquisition_start_ns": acquisition_start_ns,
            "acquisition_end_ns": acquisition_end_ns,
            "memory_current_bytes": memory_current_bytes,
            **limits,
            "events_local": events_local,
            "pressure": pressure,
            "pressure_available": pressure_available,
            "errors": errors,
        }

    def _capture(self, timestamp_ns: int) -> CgroupResourceSample:
        memory_stat = self._read_optional_flat("memory.stat")
        cpu_stat = self._read_optional_flat("cpu.stat")
        memory_current_bytes = self._read_required_int("memory.current")
        memory_swap_current_bytes = self._read_optional_int("memory.swap.current")
        acquisition_end_ns = self.clock_ns()
        if acquisition_end_ns < timestamp_ns:
            raise CgroupResourceSamplingError("sampling clock moved backwards during acquisition")
        sample = CgroupResourceSample(
            timestamp_ns=timestamp_ns,
            acquisition_start_ns=timestamp_ns,
            acquisition_end_ns=acquisition_end_ns,
            memory_current_bytes=memory_current_bytes,
            memory_swap_current_bytes=memory_swap_current_bytes,
            memory_stat={key: memory_stat[key] for key in MEMORY_STAT_SAMPLE_KEYS if key in memory_stat},
            cpu_usage_usec=cpu_stat.get("usage_usec"),
            cpu_user_usec=cpu_stat.get("user_usec"),
            cpu_system_usec=cpu_stat.get("system_usec"),
        )
        if self.samples and timestamp_ns < self.samples[-1].timestamp_ns:
            raise CgroupResourceSamplingError("sampling clock moved backwards")
        if self.samples and timestamp_ns == self.samples[-1].timestamp_ns:
            self.samples[-1] = sample
        else:
            self.samples.append(sample)
        return sample

    def _add_marker(self, name: str, timestamp_ns: int) -> None:
        assert self._start_ns is not None
        self.markers.append(
            {
                "name": name,
                "timestamp_ns": timestamp_ns,
                "elapsed_sec": round((timestamp_ns - self._start_ns) / 1_000_000_000, 9),
            }
        )

    def start(self) -> int:
        """Capture an immediate observation before releasing sandbox work."""
        if self._started:
            raise CgroupResourceSamplingError("resource sampling already started")
        self._raw_memory_source_reads = []
        self._raw_memory_source_phase = "observer_start"
        self._events_before, events_available = self._read_optional_flat_with_status("memory.events")
        self._events_local_before, events_local_available = self._read_optional_flat_with_status("memory.events.local")
        self._pressure_before = self._read_pressure()
        self._stat_before = self._read_optional_flat("memory.stat")
        self._limits = {}
        for key, filename in (
            ("memory_high", "memory.high"),
            ("memory_max", "memory.max"),
            ("swap_high", "memory.swap.high"),
            ("swap_max", "memory.swap.max"),
        ):
            value, state = self._read_limit(filename)
            self._limits[f"{key}_bytes"] = value
            self._limits[f"{key}_state"] = state
        self._peak_at_observer_start, memory_peak_available = self._read_optional_int_with_status("memory.peak")
        _swap_current, swap_current_available = self._read_optional_int_with_status("memory.swap.current")
        _swap_peak, swap_peak_available = self._read_optional_int_with_status("memory.swap.peak")
        self._capabilities = {
            "memory_events_at_observer_start": events_available,
            "memory_events_local_at_observer_start": events_local_available,
            "memory_peak_at_observer_start": memory_peak_available,
            "memory_swap_current_at_observer_start": swap_current_available,
            "memory_swap_peak_at_observer_start": swap_peak_available,
            "memory_swap_limit": self._limits["swap_max_state"] != "unavailable",
            "memory_limit": self._limits["memory_max_state"] != "unavailable",
        }
        self._ancestor_before = [self._snapshot_ancestor(path) for path in self._ancestor_paths]
        if self.ancestor_cpu_observer is not None:
            self.ancestor_cpu_observer.start()
        timestamp_ns = self.clock_ns()
        self._start_ns = timestamp_ns
        self._capture(timestamp_ns)
        self._raw_memory_source_phase = None
        self._add_marker("sandbox_observation_start", timestamp_ns)
        self._next_sample_ns = timestamp_ns + self.interval_ns
        self._started = True
        return timestamp_ns

    def poll(self) -> bool:
        """Capture at most one observation when the profiler loop reaches a deadline."""
        if not self._started or self._finished or self._next_sample_ns is None:
            raise CgroupResourceSamplingError("resource sampling is not active")
        timestamp_ns = self.clock_ns()
        if timestamp_ns < self._next_sample_ns:
            return False
        lateness_ns = timestamp_ns - self._next_sample_ns
        self.max_sample_lateness_ns = max(self.max_sample_lateness_ns, lateness_ns)
        self.missed_sample_deadlines += lateness_ns // self.interval_ns
        self._capture(timestamp_ns)
        self._next_sample_ns = timestamp_ns + self.interval_ns
        return True

    def mark(self, name: str) -> int:
        """Force a same-clock sample at a lifecycle boundary."""
        if not self._started or self._finished:
            raise CgroupResourceSamplingError("resource sampling is not active")
        timestamp_ns = self.clock_ns()
        self._capture(timestamp_ns)
        self._add_marker(name, timestamp_ns)
        self._next_sample_ns = timestamp_ns + self.interval_ns
        return timestamp_ns

    def _pressure_delta_between(
        self,
        before: Mapping[str, Mapping[str, float | int]],
        after: Mapping[str, Mapping[str, float | int]],
        *,
        duration_sec: float,
        source: str,
    ) -> tuple[dict[str, Any], list[str]]:
        result: dict[str, Any] = {}
        anomalies: list[str] = []
        for kind in sorted(set(before) | set(after)):
            before_total = before.get(kind, {}).get("total")
            after_total = after.get(kind, {}).get("total")
            if not isinstance(before_total, int) or not isinstance(after_total, int):
                continue
            if after_total < before_total:
                anomalies.append(f"{source} {kind!r} total moved backwards: " f"{before_total} -> {after_total}")
                continue
            delta_usec = after_total - before_total
            result[kind] = {
                "total_stall_usec": delta_usec,
                "stall_fraction_of_wall": (
                    round(delta_usec / (duration_sec * 1_000_000), 9) if duration_sec > 0 else None
                ),
            }
        return result, anomalies

    def _pressure_delta(
        self,
        after: Mapping[str, Mapping[str, float | int]],
        *,
        duration_sec: float,
    ) -> tuple[dict[str, Any], list[str]]:
        return self._pressure_delta_between(
            self._pressure_before,
            after,
            duration_sec=duration_sec,
            source="memory.pressure",
        )

    def _ancestor_memory_report(
        self,
        after_snapshots: list[dict[str, Any]],
        *,
        duration_sec: float,
    ) -> dict[str, Any]:
        """Combine ancestor snapshots and fail closed on ambiguous pressure."""
        ancestors: list[dict[str, Any]] = []
        aggregate_invalid_reasons: list[str] = []
        aggregate_evidence_errors: list[str] = []
        for before, after in zip(self._ancestor_before, after_snapshots, strict=False):
            relative_path = str(before["path"])
            evidence_errors = [
                *[str(error) for error in before.get("errors", [])],
                *[str(error) for error in after.get("errors", [])],
            ]
            if after.get("path") != relative_path:
                evidence_errors.append(f"ancestor chain changed from {relative_path!r} to " f"{after.get('path')!r}")

            events_delta, event_anomalies = _counter_delta(
                before.get("events_local", {}),
                after.get("events_local", {}),
                source=f"ancestor {relative_path!r} memory.events.local",
            )
            evidence_errors.extend(event_anomalies)
            for key in ("memory_high", "memory_max"):
                before_limit = (
                    before.get(f"{key}_state"),
                    before.get(f"{key}_bytes"),
                )
                after_limit = (
                    after.get(f"{key}_state"),
                    after.get(f"{key}_bytes"),
                )
                if before_limit != after_limit:
                    evidence_errors.append(
                        f"ancestor {relative_path!r} {key.replace('_', '.')} "
                        f"changed during observation: {before_limit} -> {after_limit}"
                    )

            pressure_delta: dict[str, Any] = {}
            pressure_anomalies: list[str] = []
            if before.get("pressure_available") and after.get("pressure_available"):
                pressure_delta, pressure_anomalies = self._pressure_delta_between(
                    before.get("pressure", {}),
                    after.get("pressure", {}),
                    duration_sec=duration_sec,
                    source=f"ancestor {relative_path!r} memory.pressure",
                )
                evidence_errors.extend(pressure_anomalies)

            constraint_event_deltas = {
                key: events_delta.get(key, 0) for key in ANCESTOR_CONSTRAINT_EVENT_KEYS if events_delta.get(key, 0) > 0
            }
            positive_pressure = {
                kind: int(values["total_stall_usec"])
                for kind, values in pressure_delta.items()
                if isinstance(values, dict)
                and isinstance(values.get("total_stall_usec"), int)
                and values["total_stall_usec"] > 0
            }
            finite_limits_present = any(before.get(f"{key}_state") == "finite" for key in ("memory_high", "memory_max"))
            attribution_ambiguous = bool(constraint_event_deltas or positive_pressure)

            invalid_reasons = list(dict.fromkeys(evidence_errors))
            if constraint_event_deltas:
                rendered = ", ".join(f"{key}={value}" for key, value in sorted(constraint_event_deltas.items()))
                invalid_reasons.append(
                    f"ancestor cgroup {relative_path!r} recorded constraint/OOM "
                    f"events ({rendered}); the events may come from the measured "
                    "child or sibling cgroups, so attribution is ambiguous and "
                    "the measured leaf's unconstrained demand cannot be inferred"
                )
            if positive_pressure:
                rendered = ", ".join(f"{kind}={value} usec" for kind, value in sorted(positive_pressure.items()))
                invalid_reasons.append(
                    f"ancestor cgroup {relative_path!r} recorded memory pressure "
                    f"({rendered}); the stalls may come from the measured child or "
                    "sibling cgroups, so attribution is ambiguous and "
                    "the measured leaf's unconstrained demand cannot be inferred"
                )

            evidence_errors = list(dict.fromkeys(evidence_errors))
            invalid_reasons = list(dict.fromkeys(invalid_reasons))
            aggregate_evidence_errors.extend(evidence_errors)
            aggregate_invalid_reasons.extend(invalid_reasons)
            ancestors.append(
                {
                    "path": relative_path,
                    "start": before,
                    "finish": after,
                    "events_local_delta": events_delta,
                    "pressure_delta": pressure_delta,
                    "constraint_event_deltas": constraint_event_deltas,
                    "positive_pressure_total_usec": positive_pressure,
                    "finite_limits_present": finite_limits_present,
                    "evidence_valid": not evidence_errors,
                    "attribution_ambiguous": attribution_ambiguous,
                    "natural_demand_trustworthy": not invalid_reasons,
                    "invalid_reasons": invalid_reasons,
                }
            )

        if len(self._ancestor_before) != len(after_snapshots):
            aggregate_evidence_errors.append("ancestor cgroup chain length changed during observation")
            aggregate_invalid_reasons.append("ancestor cgroup chain length changed during observation")
        aggregate_evidence_errors = list(dict.fromkeys(aggregate_evidence_errors))
        aggregate_invalid_reasons = list(dict.fromkeys(aggregate_invalid_reasons))
        return {
            "cgroup_root": str(self.cgroup_root),
            "cgroup_root_memory_controller_files_available": (self.cgroup_root / "memory.current").exists(),
            "leaf_path": self._ancestor_relative_path(self._resolved_cgroup_path),
            "chain_order": "leaf-parent-to-root",
            "ancestor_count": len(ancestors),
            "ancestors": ancestors,
            "evidence_complete": not aggregate_evidence_errors,
            "finite_limits_present": any(ancestor["finite_limits_present"] for ancestor in ancestors),
            "constraint_events_observed": any(bool(ancestor["constraint_event_deltas"]) for ancestor in ancestors),
            "pressure_observed": any(bool(ancestor["positive_pressure_total_usec"]) for ancestor in ancestors),
            "attribution_ambiguous": any(ancestor["attribution_ambiguous"] for ancestor in ancestors),
            "natural_demand_trustworthy": not aggregate_invalid_reasons,
            "natural_demand_untrustworthy": bool(aggregate_invalid_reasons),
            "evidence_errors": aggregate_evidence_errors,
            "invalid_reasons": aggregate_invalid_reasons,
            "meaning": (
                "Ancestor-local counters and pressure cover the measured leaf plus "
                "possible sibling cgroups. Positive events or stalls therefore "
                "invalidate unconstrained-demand interpretation even when attribution is ambiguous; "
                "finite unused ancestor limits are provenance only."
            ),
        }

    def finish(self) -> dict[str, Any]:
        """Capture terminal counters and build the memory summary."""
        if not self._started or self._finished or self._start_ns is None:
            raise CgroupResourceSamplingError("resource sampling is not active")
        self._raw_memory_source_phase = "controlled_finish_gate"
        end_ns = self.mark("sandbox_observation_end")
        events_after, events_after_available = self._read_optional_flat_with_status("memory.events")
        events_local_after, events_local_after_available = self._read_optional_flat_with_status("memory.events.local")
        pressure_after = self._read_pressure()
        stat_after = self._read_optional_flat("memory.stat")
        events_delta, event_anomalies = _counter_delta(self._events_before, events_after, source="memory.events")
        events_local_delta, local_event_anomalies = _counter_delta(
            self._events_local_before,
            events_local_after,
            source="memory.events.local",
        )
        duration_sec = (end_ns - self._start_ns) / 1_000_000_000
        pressure_delta, pressure_anomalies = self._pressure_delta(pressure_after, duration_sec=duration_sec)
        ancestor_after = [self._snapshot_ancestor(path) for path in self._ancestor_paths]
        ancestor_memory = self._ancestor_memory_report(ancestor_after, duration_sec=duration_sec)
        ancestor_cpu = self.ancestor_cpu_observer.finish() if self.ancestor_cpu_observer is not None else None
        self.anomalies.extend(
            event_anomalies + local_event_anomalies + pressure_anomalies + ancestor_memory["evidence_errors"]
        )
        lifetime_peak, memory_peak_after_available = self._read_optional_int_with_status("memory.peak")
        lifetime_swap_peak, swap_peak_after_available = self._read_optional_int_with_status("memory.swap.peak")
        _swap_current_after, swap_current_after_available = self._read_optional_int_with_status("memory.swap.current")
        self._raw_memory_source_phase = None
        self._capabilities.update(
            {
                "memory_events_at_finish_gate": events_after_available,
                "memory_events_local_at_finish_gate": events_local_after_available,
                "memory_peak_at_finish_gate": memory_peak_after_available,
                "memory_swap_current_at_finish_gate": swap_current_after_available,
                "memory_swap_peak_at_finish_gate": swap_peak_after_available,
                "ancestor_memory_evidence_complete": ancestor_memory["evidence_complete"],
            }
        )
        if ancestor_cpu is not None:
            self._capabilities.update(
                {
                    "ancestor_cpu_evidence_complete": ancestor_cpu["evidence_complete"],
                    "ancestor_cpu_valid_for_cpu_performance": ancestor_cpu["valid_for_cpu_performance"],
                }
            )
        sampled_peak = max(sample.memory_current_bytes for sample in self.samples)
        peak_invalid_reasons: list[str] = []
        if lifetime_peak is not None and lifetime_peak < sampled_peak:
            reason = "memory.peak was smaller than an observed memory.current sample"
            self.anomalies.append(reason)
            peak_invalid_reasons.append(reason)
        if lifetime_peak is None:
            peak_invalid_reasons.append("memory.peak is unavailable")
        marker_times = {marker["name"]: marker["timestamp_ns"] for marker in self.markers}
        observation = summarize_sample_window(self.samples, start_ns=self._start_ns, end_ns=end_ns)
        command_window = None
        if "command_start" in marker_times and "command_end" in marker_times:
            command_window = summarize_sample_window(
                self.samples,
                start_ns=marker_times["command_start"],
                end_ns=marker_times["command_end"],
            )
        censoring = "none"
        if any(events_after.get(key, 0) > 0 for key in ("oom", "oom_kill", "oom_group_kill")):
            censoring = "oom"
        elif events_after.get("max", 0) > 0:
            censoring = "memory_max"
        elif events_after.get("high", 0) > 0:
            censoring = "memory_high"
        elif (lifetime_swap_peak is not None and lifetime_swap_peak > 0) or (
            observation["sampled_peak_swap_bytes"] or 0
        ) > 0:
            # A swap-backed run may materially understate no-swap memory
            # demand. Keep the observed evidence, but do not call the peak an
            # unconstrained-demand observation.
            censoring = "swap_activity"
        instrumentation_invalid_reasons = list(self.anomalies)
        natural_peak_invalid_reasons = [
            *instrumentation_invalid_reasons,
            *ancestor_memory["invalid_reasons"],
        ]
        missing_event_keys = [
            key for key in REQUIRED_MEMORY_EVENT_KEYS if key not in self._events_before or key not in events_after
        ]
        events_observable = (
            self._capabilities["memory_events_at_observer_start"] and events_after_available and not missing_event_keys
        )
        if not events_observable:
            reason = "memory.events was unavailable or lacked required high/max/OOM counters"
            instrumentation_invalid_reasons.append(reason)
            natural_peak_invalid_reasons.append(reason)
        swap_limit_state = self._limits["swap_max_state"]
        swap_disabled = swap_limit_state == "finite" and self._limits["swap_max_bytes"] == 0
        swap_peak_observable = self._capabilities["memory_swap_peak_at_observer_start"] and swap_peak_after_available
        swap_observable = self._capabilities["memory_swap_limit"] and (swap_disabled or swap_peak_observable)
        if not swap_observable:
            natural_peak_invalid_reasons.append(
                "swap activity was not provably observable: require readable "
                "memory.swap.max and either disabled swap or memory.swap.peak"
            )
        if not self._capabilities["memory_limit"]:
            natural_peak_invalid_reasons.append("memory.max was unavailable")
        if lifetime_peak is None:
            natural_peak_invalid_reasons.append("memory.peak is unavailable")
        # Docker removes its child cgroup when init exits. The low-privilege
        # replay observation boundary therefore ends at the controlled finish gate:
        # startup, preparation, command work, descendants/drain, and terminal
        # in-container report writes are complete; optional post-measurement
        # verification, FIFO unblocking, and shell/init exit are excluded. Do
        # not call this a full cgroup lifetime.
        observation_boundary_complete = True
        if censoring != "none":
            natural_peak_invalid_reasons.append(
                (
                    "swap activity occurred; reproduce the production swap policy "
                    "explicitly before interpreting unconstrained demand"
                )
                if censoring == "swap_activity"
                else (
                    f"memory demand is constrained by {censoring}; the observed "
                    "peak remains exact but is not unconstrained demand"
                )
            )
        instrumentation_valid = not self.anomalies and bool(self.samples) and events_observable
        peak_valid = lifetime_peak is not None and lifetime_peak >= sampled_peak
        natural_peak_trustworthy = (
            instrumentation_valid
            and peak_valid
            and censoring == "none"
            and swap_observable
            and self._capabilities["memory_limit"]
            and observation_boundary_complete
            and ancestor_memory["natural_demand_trustworthy"]
        )
        timeline_invalid_reasons: list[str] = []
        if len(self.samples) < 2:
            timeline_invalid_reasons.append("fewer than two resource samples")
        max_gap = observation.get("max_sample_gap_sec")
        if isinstance(max_gap, (int, float)) and max_gap > 2 * self.interval_sec:
            timeline_invalid_reasons.append(f"maximum sample gap {max_gap:.6f}s exceeded twice the requested interval")
        self._finished = True
        instrumentation_invalid_reasons = list(dict.fromkeys(instrumentation_invalid_reasons))
        natural_peak_invalid_reasons = list(dict.fromkeys(natural_peak_invalid_reasons))
        memory_max = self._limits.get("memory_max_bytes")
        memory_max_state = self._limits.get("memory_max_state")
        finite_limit = (
            memory_max
            if memory_max_state == "finite" and isinstance(memory_max, int) and not isinstance(memory_max, bool)
            else None
        )
        remaining_to_limit = (
            max(0, finite_limit - lifetime_peak) if finite_limit is not None and lifetime_peak is not None else None
        )
        peak_fraction_of_limit = (
            lifetime_peak / finite_limit
            if finite_limit is not None and finite_limit > 0 and lifetime_peak is not None
            else None
        )
        max_sample_acquisition_duration_ns = max(
            sample.acquisition_end_ns - sample.acquisition_start_ns for sample in self.samples
        )
        report = {
            "schema_version": MEMORY_REPORT_SCHEMA_VERSION,
            "method": "container-cgroup-v2-memory-and-cpu-stat-sampling",
            "scope": (
                "Measured cgroup v2 subtree, including descendant anonymous, "
                "file-cache, and kernel memory charges. This is not process RSS."
            ),
            "clock": "host-monotonic",
            "sample_interval_sec": self.interval_sec,
            "sample_count": len(self.samples),
            "raw_source_snapshot_schema_version": (MEMORY_SOURCE_SNAPSHOT_SCHEMA_VERSION),
            "raw_source_snapshot_read_count": len(self._raw_memory_source_reads),
            "missed_sample_deadlines": self.missed_sample_deadlines,
            "max_sample_lateness_sec": round(self.max_sample_lateness_ns / 1_000_000_000, 9),
            "max_sample_acquisition_duration_sec": round(max_sample_acquisition_duration_ns / 1_000_000_000, 9),
            "markers": list(self.markers),
            "observation_window": observation,
            "command_window": command_window,
            "lifetime": {
                "peak_bytes": lifetime_peak,
                "peak_at_observer_start_bytes": self._peak_at_observer_start,
                "peak_source": "memory.peak-since-cgroup-creation-read-at-finish-gate",
                "exact_through_controlled_finish_gate": peak_valid,
                "exact_for_fresh_cgroup_lifetime": False,
                "container_init_exit_observed": False,
            },
            "swap": {
                "lifetime_peak_bytes": lifetime_swap_peak,
                "sampled_peak_bytes": observation["sampled_peak_swap_bytes"],
            },
            "limits": dict(self._limits),
            "observed_limit_margin": {
                "limit_scope": "leaf-cgroup-memory.max-at-observer-start",
                "configured_limit_state": memory_max_state,
                "configured_limit_bytes": finite_limit,
                "observed_peak_bytes": lifetime_peak,
                "remaining_to_limit_bytes": remaining_to_limit,
                "peak_fraction_of_limit": peak_fraction_of_limit,
                "arithmetic_valid": finite_limit is not None and peak_valid,
                "meaning": (
                    "Descriptive arithmetic for this run only: configured leaf "
                    "memory.max minus the observed exact peak. This is not "
                    "recommended safety headroom and does not account for a "
                    "constrained ancestor or future-run variance."
                ),
            },
            "capabilities": dict(self._capabilities),
            "observation_boundary": "controlled-finish-gate",
            "observation_boundary_complete": observation_boundary_complete,
            "full_cgroup_lifetime_observed": False,
            "events_at_observer_start": self._events_before,
            "events_lifetime": events_after,
            "events_delta": events_delta,
            "events_local_at_observer_start": self._events_local_before,
            "events_local_lifetime": events_local_after,
            "events_local_delta": events_local_delta,
            "pressure_delta": pressure_delta,
            "ancestor_memory": ancestor_memory,
            "memory_stat_before": self._stat_before,
            "memory_stat_after": stat_after,
            "censoring": censoring,
            "instrumentation_valid": instrumentation_valid,
            "instrumentation_invalid_reasons": instrumentation_invalid_reasons,
            "peak_valid": peak_valid,
            "peak_invalid_reasons": peak_invalid_reasons,
            "natural_peak_trustworthy": natural_peak_trustworthy,
            "natural_peak_invalid_reasons": natural_peak_invalid_reasons,
            "timeline_valid_for_shape": not timeline_invalid_reasons,
            "timeline_invalid_reasons": timeline_invalid_reasons,
            "meaning": (
                "memory.peak through the controlled finish gate is the exact "
                "observed maximum charge. It covers init startup, preparation, command "
                "work, descendants/drain, and terminal artifacts, but excludes "
                "post-measurement verification, the final FIFO unblock, and "
                "shell/init exit. Sampled memory and interval CPU values explain "
                "shape and may miss short spikes. A constrained run retains its "
                "observed peak while natural_peak_trustworthy becomes false."
            ),
        }
        if ancestor_cpu is not None:
            report["raw_ancestor_cpu_source_read_count"] = ancestor_cpu["raw_source_read_count"]
            report["ancestor_cpu"] = ancestor_cpu
        else:
            report["ancestor_cpu_observer"] = "disabled-owned-by-mandatory-cgroup-cpu-profiler"
        return report

    def raw_memory_source_snapshot_payload(self) -> dict[str, Any]:
        """Return exact gate-time cgroup memory and ancestor CPU source text.

        The compatibility method/artifact name predates ancestor CPU validation;
        a nested append-only source payload now retains each ancestor ``cpu.max``
        and ``cpu.stat`` consumed by the report.
        """
        if not self._finished:
            raise CgroupResourceSamplingError("memory source snapshots are incomplete before sampling finishes")
        payload = {
            "schema_version": MEMORY_SOURCE_SNAPSHOT_SCHEMA_VERSION,
            "scope": {
                "kind": "cgroup-v2-subtree",
                "leaf_path": self._ancestor_relative_path(self._resolved_cgroup_path),
                "path_representation": "relative-to-configured-cgroup-root",
            },
            "phases": ["observer_start", "controlled_finish_gate"],
            "controllers": (["cpu", "memory"] if self.ancestor_cpu_observer is not None else ["memory"]),
            "reads": list(self._raw_memory_source_reads),
            "meaning": (
                "Exact text consumed from cgroup memory controller files is in "
                "reads; exact ancestor cpu.max/cpu.stat source evidence is in "
                "ancestor_cpu when enabled. Derived deltas and validity decisions live in the "
                "parsed report, not this raw source."
            ),
        }
        if self.ancestor_cpu_observer is not None:
            payload["ancestor_cpu"] = self.ancestor_cpu_observer.raw_source_snapshot_payload()
        return payload

    def write_raw_memory_source_snapshots(self, path: Path) -> None:
        path.write_text(
            json.dumps(
                self.raw_memory_source_snapshot_payload(),
                indent=2,
                sort_keys=True,
            )
            + "\n"
        )

    def timeseries_payload(self) -> dict[str, Any]:
        """Return raw samples with derived interval-average CPU demand."""
        if not self._started or self._start_ns is None:
            raise CgroupResourceSamplingError("resource sampling was not started")
        rows: list[dict[str, Any]] = []
        previous: CgroupResourceSample | None = None
        for sample in self.samples:
            row = asdict(sample)
            acquisition_duration_ns = sample.acquisition_end_ns - sample.acquisition_start_ns
            row["elapsed_sec"] = round((sample.timestamp_ns - self._start_ns) / 1_000_000_000, 9)
            row["acquisition_start_elapsed_sec"] = round(
                (sample.acquisition_start_ns - self._start_ns) / 1_000_000_000,
                9,
            )
            row["acquisition_end_elapsed_sec"] = round(
                (sample.acquisition_end_ns - self._start_ns) / 1_000_000_000,
                9,
            )
            row["acquisition_duration_sec"] = round(acquisition_duration_ns / 1_000_000_000, 9)
            row["timestamp_alignment_uncertainty_sec"] = round(acquisition_duration_ns / 1_000_000_000, 9)
            row["interval_sec"] = None
            row["cpu_average_logical_cores"] = None
            if previous is not None:
                interval_sec = (sample.timestamp_ns - previous.timestamp_ns) / 1_000_000_000
                row["interval_sec"] = round(interval_sec, 9)
                if (
                    interval_sec > 0
                    and sample.cpu_usage_usec is not None
                    and previous.cpu_usage_usec is not None
                    and sample.cpu_usage_usec >= previous.cpu_usage_usec
                ):
                    row["cpu_average_logical_cores"] = round(
                        (sample.cpu_usage_usec - previous.cpu_usage_usec) / 1_000_000 / interval_sec,
                        9,
                    )
            rows.append(row)
            previous = sample
        return {
            "schema_version": TIMESERIES_SCHEMA_VERSION,
            "scope": str(self.cgroup_path),
            "clock": "host-monotonic",
            "timestamp_semantics": "acquisition-start",
            "timestamp_alignment_uncertainty_semantics": (
                "The cgroup files are read serially after timestamp_ns. Each value "
                "was acquired no later than acquisition_end_ns; the full acquisition "
                "duration is the conservative alignment uncertainty."
            ),
            "sample_interval_sec": self.interval_sec,
            "max_sample_acquisition_duration_sec": round(
                max(sample.acquisition_end_ns - sample.acquisition_start_ns for sample in self.samples) / 1_000_000_000,
                9,
            ),
            "markers": list(self.markers),
            "samples": rows,
        }

    def write_timeseries(self, path: Path) -> None:
        """Persist raw evidence outside the compact profile-report envelope."""
        path.write_text(json.dumps(self.timeseries_payload(), indent=2, sort_keys=True) + "\n")
