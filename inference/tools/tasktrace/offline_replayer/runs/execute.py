# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Execute one small, content-bound offline-replay run.

``tasktrace-run.v1`` preserves the original one-at-a-time contract.
``tasktrace-run.v2`` adds bounded shared-cpuset concurrency: every in-flight
task still gets a fresh Docker sandbox and independent cgroup accounting, but
all sandboxes are eligible to run on the same declared CPU pool.
``tasktrace-run.v3`` expands compact worker-pool placement (or accepts explicit
launches), then concurrently submits each ordered wave and waits for its full
completion before dispatching the next one. Enroot retains the measured replay
windows so achieved overlap can be validated without claiming aligned starts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import deque
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from ..adapters.atif_profile import (
    AtifTrajectoryProfileConfig,
    preflight_atif_trajectory_profile,
    run_atif_trajectory_profile,
)
from ..adapters.pacing import UNPACED
from ..measurement.container import controller_source_sha256
from ..measurement.network_policy import NetworkPolicy
from . import enroot as enroot_runtime
from .host import HostIdentityError, collect_host_cpu_identity
from .spec import (
    SCHEMA_VERSION_V1,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    OrderedRunSpec,
    RunLaunchSpec,
    RunSpec,
    RunSpecError,
    RunTaskSpec,
    RunWaveSpec,
    load_run_spec,
)

SUMMARY_SCHEMA_VERSION_V1 = "tasktrace-run-summary.v1"
SUMMARY_SCHEMA_VERSION_V2 = "tasktrace-run-summary.v2"
SUMMARY_SCHEMA_VERSION_V3 = "tasktrace-run-summary.v3"
# Backward-compatible name for readers/tests of the original summary.
SUMMARY_SCHEMA_VERSION = SUMMARY_SCHEMA_VERSION_V1
RESULT_COLUMNS = (
    "task_id",
    "repetition",
    "status",
    "duration_sec",
    "duration_valid",
    "reason",
    "profile_dir",
    "source_sha256",
    "image_id",
    "execution_plan_digest",
)
RESULT_COLUMNS_V2 = (
    "attempt_index",
    "completion_index",
    *RESULT_COLUMNS,
    "window_start_ns",
    "window_end_ns",
    "cpu_usage_sec",
    "average_logical_cores",
)
RESULT_COLUMNS_V3 = (
    "attempt_index",
    "completion_index",
    "wave_index",
    "wave_id",
    "wave_scored",
    "launch_index",
    "cpus",
    *RESULT_COLUMNS,
    "window_start_ns",
    "window_end_ns",
    "cpu_usage_sec",
    "average_logical_cores",
)
ENROOT_STATE_ENVIRONMENT = frozenset({"ENROOT_CACHE_PATH", "ENROOT_DATA_PATH", "ENROOT_RUNTIME_PATH"})
ENROOT_FORBIDDEN_ENVIRONMENT = frozenset({"BASH_ENV", "PYTHONHOME"})
# Cgroup CPU is retained in integer microseconds.  Permit one counter quantum
# per attempt plus a one-part-per-billion float allowance when checking the
# aggregate against the declared CPU-pool capacity.
_CPU_CAPACITY_COUNTER_TOLERANCE_SEC_PER_ATTEMPT = 1e-6
_CPU_CAPACITY_FLOAT_TOLERANCE_FRACTION = 1e-9


class RunExecutionError(RuntimeError):
    """A run could not be started without ambiguity."""


@dataclass(frozen=True)
class RunExecutionResult:
    exit_code: int
    rows: tuple[Mapping[str, Any], ...]
    summary: Mapping[str, Any]


@dataclass(frozen=True)
class _Preflight:
    image_id: str
    controller_source_sha256: str | None = None
    enroot_runtime_provenance: Mapping[str, Any] | None = None


@dataclass(frozen=True)
class _Attempt:
    attempt_index: int
    repetition: int
    task: RunTaskSpec


@dataclass(frozen=True)
class _WaveAttempt:
    attempt_index: int
    wave_index: int
    launch_index: int
    wave: RunWaveSpec
    launch: RunLaunchSpec
    task: RunTaskSpec


@dataclass(frozen=True)
class _ProjectedDuration:
    duration_sec: float | None
    valid: bool
    reason: str
    image_id: str
    execution_plan_digest: str | None
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    cpu_usage_sec: float | None = None
    average_logical_cores: float | None = None


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _profile_config(
    spec: RunSpec | OrderedRunSpec,
    task: RunTaskSpec,
    output_dir: Path,
    *,
    cpus: str | None = None,
) -> AtifTrajectoryProfileConfig:
    if isinstance(spec, OrderedRunSpec):
        defaults = spec.defaults
        selected_cpus = cpus
        runtime = defaults.runtime
        pacing = defaults.pacing
        timeout_sec = task.action_timeout_sec if task.action_timeout_sec is not None else defaults.action_timeout_sec
        controller_runtime = defaults.controller_runtime
        profile_sets = defaults.profile_sets
        network_policy = NetworkPolicy(
            mode=defaults.network_policy,
            allowed_hosts=defaults.allowed_hosts,
        )
        workdir = defaults.workdir if defaults.workdir is not None else "/app"
    else:
        selected_cpus = spec.cpus
        runtime = "docker"
        pacing = UNPACED
        timeout_sec = task.action_timeout_sec
        controller_runtime = "host-mount"
        profile_sets = ("cpu-average",)
        network_policy = None
        workdir = "/app"
    return AtifTrajectoryProfileConfig(
        trajectory=task.atif,
        image=task.image,
        output_dir=output_dir,
        cpu_limit=None,
        cpuset_cpus=selected_cpus,
        allocated_cpus=None,
        pacing=pacing,
        timeout_sec=timeout_sec,
        prepare_command=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir=workdir,
        task_name=task.task_id,
        capture_action_outputs=False,
        controller_runtime=controller_runtime,
        network_policy=network_policy,
        runtime=runtime,
        profile_sets=profile_sets,
    )


def _image_id(report: Mapping[str, Any]) -> str | None:
    identity = report.get("image_identity")
    if not isinstance(identity, Mapping):
        return None
    enroot_id = enroot_runtime.image_id(identity)
    if enroot_id is not None:
        return enroot_id
    value = identity.get("id")
    return value if isinstance(value, str) and value else None


def _ensure_output_absent(output: Path) -> None:
    if output.exists():
        raise RunExecutionError(f"output path must not already exist: {output}")


def _validate_ordered_runtime_environment(spec: OrderedRunSpec) -> None:
    """Reject ambient configuration that can silently alter Enroot behavior."""
    if spec.defaults.runtime != "enroot":
        return
    inherited = sorted(
        name
        for name in os.environ
        if (name.startswith("ENROOT_") and name not in ENROOT_STATE_ENVIRONMENT) or name in ENROOT_FORBIDDEN_ENVIRONMENT
    )
    if inherited:
        raise RunExecutionError(
            "unsealed ambient Enroot/runtime configuration is not supported: " + ", ".join(inherited)
        )


def _set_current_thread_affinity(cpu_ids: Sequence[int]) -> tuple[int, ...]:
    requested = set(cpu_ids)
    try:
        os.sched_setaffinity(0, requested)
        observed = set(os.sched_getaffinity(0))
    except (AttributeError, OSError) as exc:
        raise RunExecutionError(f"unable to enforce controller CPU affinity: {exc}") from exc
    if observed != requested:
        raise RunExecutionError(
            "controller CPU affinity differs after enforcement: "
            f"requested={sorted(requested)}, observed={sorted(observed)}"
        )
    return tuple(sorted(observed))


def _ordered_wall_clock_ns(runtime: str) -> int:
    """Use the same suspend-aware clock as Enroot task/lifecycle evidence."""
    if runtime == "enroot":
        clock_id = getattr(time, "CLOCK_BOOTTIME", None)
        if clock_id is None:
            raise RunExecutionError("CLOCK_BOOTTIME is unavailable on this platform")
        return time.clock_gettime_ns(clock_id)
    return time.monotonic_ns()


@contextmanager
def _controller_affinity(
    spec: OrderedRunSpec,
) -> Any:
    """Pin host-side dispatch while preserving the caller's prior affinity.

    Linux affinity is per thread, so worker threads reapply this set before
    spawning runtime helpers.  The in-container replay controller starts with
    its launch's requested taskset affinity, and caller affinity is restored.
    """
    try:
        previous = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError) as exc:
        raise RunExecutionError(f"unable to inspect controller CPU affinity: {exc}") from exc
    requested = spec.defaults.controller_cpu_ids
    missing = sorted(set(requested) - set(previous))
    if missing:
        raise RunExecutionError(
            "controller CPU set is outside the caller's available affinity: " + ", ".join(map(str, missing))
        )
    observed = _set_current_thread_affinity(requested)
    evidence: dict[str, Any] = {
        "requested_cpus": spec.defaults.controller_cpus,
        "requested_logical_cpu_ids": list(requested),
        "observed_main_thread_logical_cpu_ids": list(observed),
        "caller_affinity_before": list(previous),
        "caller_affinity_restored": False,
        "scope": (
            "host runner threads and inherited Enroot launch helpers; not the "
            "taskset-pinned in-container replay controller or kernel work"
            if spec.defaults.runtime == "enroot"
            else "host runner threads and inherited Docker-client helpers; not the "
            "in-container replay controller, Docker daemon, or kernel work"
        ),
    }
    try:
        yield evidence
    finally:
        try:
            os.sched_setaffinity(0, set(previous))
            restored = tuple(sorted(os.sched_getaffinity(0)))
        except (AttributeError, OSError) as exc:
            raise RunExecutionError(f"unable to restore caller CPU affinity: {exc}") from exc
        if restored != previous:
            raise RunExecutionError(
                "caller CPU affinity differs after restoration: "
                f"expected={list(previous)}, observed={list(restored)}"
            )
        evidence["caller_affinity_restored"] = True


def _preflight_all(spec: RunSpec) -> dict[str, _Preflight]:
    results: dict[str, _Preflight] = {}
    errors: list[str] = []
    for task in spec.tasks:
        output_dir = spec.output / "tasks" / task.task_id / "rep-0001" / "profile"
        try:
            report = preflight_atif_trajectory_profile(_profile_config(spec, task, output_dir))
        except Exception as exc:
            errors.append(f"{task.task_id}: {exc}")
            continue
        if not isinstance(report, Mapping):
            errors.append(f"{task.task_id}: profile preflight returned a non-object")
            continue
        reasons: list[str] = []
        if report.get("ready") is not True:
            reasons.append("profile preflight did not report ready=true")
        if report.get("source_sha256") != task.source_sha256:
            reasons.append("preflight source digest differs from the sealed ATIF")
        if report.get("cpuset_cpus") != spec.cpus:
            reasons.append("preflight CPU set differs from the requested CPU set")
        if report.get("profile_sets") != ["cpu-average"]:
            reasons.append("preflight did not select exactly the cpu-average profile set")
        image_id = _image_id(report)
        if image_id is None:
            reasons.append("preflight did not resolve an image ID")
        elif task.expected_image_id is not None and image_id != task.expected_image_id:
            reasons.append(f"resolved image ID {image_id!r} differs from expected " f"{task.expected_image_id!r}")
        if reasons:
            errors.append(f"{task.task_id}: " + "; ".join(reasons))
        else:
            assert image_id is not None
            results[task.task_id] = _Preflight(image_id=image_id)
    if errors:
        raise RunExecutionError("preflight failed: " + " | ".join(errors))
    return results


def _preflight_ordered(
    spec: OrderedRunSpec,
    attempts: Sequence[_WaveAttempt],
    *,
    controller_digest: str | None = None,
    enroot_provenance: Mapping[str, Any] | None = None,
) -> dict[tuple[str, str], _Preflight]:
    """Preflight every distinct task/cpuset placement before any launch."""
    results: dict[tuple[str, str], _Preflight] = {}
    image_ids_by_task: dict[str, str] = {}
    enroot_by_task: dict[str, _Preflight] = {}
    errors: list[str] = []
    for attempt in attempts:
        key = (attempt.task.task_id, attempt.launch.cpus)
        if key in results:
            continue
        if spec.defaults.runtime == "enroot":
            try:
                available = set(os.sched_getaffinity(0))
            except (AttributeError, OSError) as exc:
                errors.append(
                    f"{attempt.task.task_id} on {attempt.launch.cpus}: "
                    f"unable to inspect available CPU placement: {exc}"
                )
                continue
            missing = sorted(set(attempt.launch.cpu_ids) - available)
            if missing:
                errors.append(
                    f"{attempt.task.task_id} on {attempt.launch.cpus}: "
                    "requested CPU set contains unavailable logical CPUs: " + ", ".join(map(str, missing))
                )
                continue
            cached = enroot_by_task.get(attempt.task.task_id)
            if cached is not None:
                results[key] = cached
                continue
        output_dir = _wave_attempt_profile_path(spec, attempt)
        try:
            report = preflight_atif_trajectory_profile(
                _profile_config(
                    spec,
                    attempt.task,
                    output_dir,
                    cpus=attempt.launch.cpus,
                )
            )
        except Exception as exc:
            errors.append(f"{attempt.task.task_id} on {attempt.launch.cpus}: {exc}")
            continue
        if not isinstance(report, Mapping):
            errors.append(
                f"{attempt.task.task_id} on {attempt.launch.cpus}: " "profile preflight returned a non-object"
            )
            continue
        reasons: list[str] = []
        if report.get("ready") is not True:
            reasons.append("profile preflight did not report ready=true")
        if report.get("source_sha256") != attempt.task.source_sha256:
            reasons.append("preflight source digest differs from the sealed ATIF")
        if report.get("cpuset_cpus") != attempt.launch.cpus:
            reasons.append("preflight CPU set differs from the launch CPU set")
        if report.get("profile_sets") != list(spec.defaults.profile_sets):
            reasons.append("preflight profile sets differ from defaults.profile_sets")
        image_id = _image_id(report)
        if image_id is None:
            reasons.append("preflight did not resolve an image ID")
        elif attempt.task.expected_image_id is not None and image_id != attempt.task.expected_image_id:
            reasons.append(
                f"resolved image ID {image_id!r} differs from expected " f"{attempt.task.expected_image_id!r}"
            )
        previous_image_id = image_ids_by_task.get(attempt.task.task_id)
        if image_id is not None and previous_image_id is not None and image_id != previous_image_id:
            reasons.append("resolved image ID changed across launch placements")
        if reasons:
            errors.append(f"{attempt.task.task_id} on {attempt.launch.cpus}: " + "; ".join(reasons))
            continue
        assert image_id is not None
        image_ids_by_task[attempt.task.task_id] = image_id
        preflight = _Preflight(
            image_id=image_id,
            controller_source_sha256=controller_digest,
            enroot_runtime_provenance=enroot_provenance,
        )
        results[key] = preflight
        if spec.defaults.runtime == "enroot":
            enroot_by_task[attempt.task.task_id] = preflight
    if errors:
        raise RunExecutionError("preflight failed: " + " | ".join(errors))
    return results


def _positive_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number > 0 else None


def _nonnegative_number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) and number >= 0 else None


def _nonnegative_monotonic_ns(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _nonnegative_integer(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


def _sha256_digest(value: Any) -> str | None:
    if not isinstance(value, str) or len(value) != 64:
        return None
    if not all(character in "0123456789abcdef" for character in value):
        return None
    return value


def _project_duration(
    report: Mapping[str, Any],
    *,
    task: RunTaskSpec,
    preflight: _Preflight,
    require_common_window: bool = False,
    ordered_spec: OrderedRunSpec | None = None,
    launch: RunLaunchSpec | None = None,
) -> _ProjectedDuration:
    """Validate one attempt without discarding its diagnostic evidence.

    Missing or inconsistent replay, timing, cgroup, source, image, or plan
    evidence invalidates the projected duration.  The caller still retains the
    row, but must exclude it from headline performance metrics.
    """
    if ordered_spec is not None and ordered_spec.defaults.runtime == "enroot" and launch is not None:
        return enroot_runtime.project(
            report,
            task=task,
            preflight_image_id=preflight.image_id,
            controller_source_sha256=preflight.controller_source_sha256,
            runtime_provenance=preflight.enroot_runtime_provenance,
            defaults=ordered_spec.defaults,
            launch=launch,
        )

    reasons: list[str] = []
    cpu = report.get("container_cpu")
    if not isinstance(cpu, Mapping):
        cpu = {}
        reasons.append("container_cpu evidence is missing")
    duration = _positive_number(cpu.get("wall_sec"))
    if duration is None:
        reasons.append("container_cpu.wall_sec is not a finite positive number")
    if cpu.get("valid_for_cpu_performance") is not True:
        reasons.append("container CPU evidence is not valid for performance comparison")
    window_start_ns = _nonnegative_monotonic_ns(cpu.get("window_start_ns"))
    window_end_ns = _nonnegative_monotonic_ns(cpu.get("window_end_ns"))
    cpu_usage_sec = _nonnegative_number(cpu.get("usage_sec"))
    average_logical_cores = _nonnegative_number(cpu.get("average_logical_cores"))
    if ordered_spec is not None and launch is not None:
        if cpu.get("allocated_cpu_source") != "cpuset-cpus":
            reasons.append("container CPU allocation is not sourced from cpuset-cpus")
        allocated_logical_cpus = _positive_number(cpu.get("allocated_logical_cpus"))
        if allocated_logical_cpus != float(len(launch.cpu_ids)):
            reasons.append("container CPU allocation width differs from the launch CPU set")

        execution_plan = report.get("execution_plan")
        extensions = execution_plan.get("extensions") if isinstance(execution_plan, Mapping) else None
        if not isinstance(extensions, Mapping):
            reasons.append("execution plan extensions are missing")
        elif extensions.get("cpuset_cpus") != launch.cpus:
            reasons.append("execution plan CPU set differs from the launch CPU set")

        defaults = ordered_spec.defaults
        if report.get("pacing") != defaults.pacing:
            reasons.append("profile pacing differs from defaults.pacing")
        if report.get("controller_runtime") != defaults.controller_runtime:
            reasons.append("profile controller runtime differs from defaults.controller_runtime")
        provenance = report.get("profile_provenance")
        if not isinstance(provenance, Mapping):
            reasons.append("profile provenance is missing")
        elif provenance.get("profile_sets") != list(defaults.profile_sets):
            reasons.append("profile sets differ from defaults.profile_sets")
        network_policy = report.get("network_policy")
        expected_network = NetworkPolicy(
            mode=defaults.network_policy,
            allowed_hosts=defaults.allowed_hosts,
        )
        if not isinstance(network_policy, Mapping):
            reasons.append("network policy evidence is missing")
        elif network_policy.get("comparability_sha256") != expected_network.comparability_digest:
            reasons.append("network policy differs from defaults.network_policy")
    if require_common_window:
        if window_start_ns is None:
            reasons.append("container_cpu.window_start_ns is missing or invalid")
        if window_end_ns is None:
            reasons.append("container_cpu.window_end_ns is missing or invalid")
        if window_start_ns is not None and window_end_ns is not None and window_end_ns <= window_start_ns:
            reasons.append("container CPU common-window timestamps are not increasing")
        if cpu_usage_sec is None:
            reasons.append("container_cpu.usage_sec is missing or invalid")
        if average_logical_cores is None:
            reasons.append("container_cpu.average_logical_cores is missing or invalid")
        if (
            duration is not None
            and window_start_ns is not None
            and window_end_ns is not None
            and not math.isclose(
                (window_end_ns - window_start_ns) / 1_000_000_000,
                duration,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            reasons.append("container CPU wall time disagrees with common-window timestamps")
        if (
            duration is not None
            and cpu_usage_sec is not None
            and average_logical_cores is not None
            and not math.isclose(
                cpu_usage_sec / duration,
                average_logical_cores,
                rel_tol=1e-9,
                abs_tol=1e-9,
            )
        ):
            reasons.append("container CPU average cores disagrees with CPU/wall time")

    replay = report.get("replay_summary")
    if not isinstance(replay, Mapping):
        replay = {}
        reasons.append("replay summary is missing")
    timed_out = _nonnegative_integer(replay.get("actions_timed_out"))
    mismatches = _nonnegative_integer(replay.get("source_outcome_mismatch_count"))
    if timed_out is None:
        reasons.append("replay timeout count is missing or invalid")
    elif timed_out:
        reasons.append(f"{timed_out} replay action(s) timed out")
    if mismatches is None:
        reasons.append("replay outcome-mismatch count is missing or invalid")
    elif mismatches:
        reasons.append(f"{mismatches} replay source-outcome mismatch(es)")

    if report.get("profiling_valid") is not True:
        reasons.append("profiling evidence is not explicitly valid")
    if report.get("status") != "valid":
        reasons.append("pipeline status is not valid")
    if report.get("source_trajectory_sha256") != task.source_sha256:
        reasons.append("profile source digest differs from the sealed ATIF")
    execution_plan_digest = _sha256_digest(report.get("execution_plan_digest"))
    if execution_plan_digest is None:
        reasons.append("execution plan digest is missing or invalid")

    observed_image_id = _image_id(report)
    image_id = observed_image_id or preflight.image_id
    if observed_image_id is None:
        reasons.append("profile did not retain an observed image ID")
    elif observed_image_id != preflight.image_id:
        reasons.append("profile image ID differs from preflight")
    if task.expected_image_id is not None and image_id != task.expected_image_id:
        reasons.append("profile image ID differs from expected image ID")

    valid = duration is not None and not reasons
    return _ProjectedDuration(
        duration_sec=duration,
        valid=valid,
        reason="; ".join(dict.fromkeys(reasons)),
        image_id=image_id,
        execution_plan_digest=execution_plan_digest,
        window_start_ns=window_start_ns,
        window_end_ns=window_end_ns,
        cpu_usage_sec=cpu_usage_sec,
        average_logical_cores=average_logical_cores,
    )


def _failed_row(
    *,
    task: RunTaskSpec,
    repetition: int,
    profile_dir: str,
    image_id: str,
    reason: str,
    runtime: str,
    attempt_index: int | None = None,
    ordered_attempt: _WaveAttempt | None = None,
) -> dict[str, Any]:
    row = {
        "task_id": task.task_id,
        "repetition": repetition,
        "status": "failed",
        "duration_sec": None,
        "duration_valid": False,
        "reason": reason,
        "profile_dir": profile_dir,
        "source_sha256": task.source_sha256,
        "image_id": image_id,
        "execution_plan_digest": None,
    }
    if runtime == "enroot":
        row.update({"cleanup_valid": None, "clock_reference": None})
    if attempt_index is not None:
        row.update(
            {
                "attempt_index": attempt_index,
                "completion_index": None,
                "window_start_ns": None,
                "window_end_ns": None,
                "cpu_usage_sec": None,
                "average_logical_cores": None,
            }
        )
    if ordered_attempt is not None:
        row.update(
            {
                "wave_index": ordered_attempt.wave_index,
                "wave_id": ordered_attempt.wave.wave_id,
                "wave_scored": ordered_attempt.wave.scored,
                "launch_index": ordered_attempt.launch_index,
                "cpus": ordered_attempt.launch.cpus,
            }
        )
    return row


def _write_csv(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    columns: Sequence[str] = RESULT_COLUMNS,
) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in columns})
    temporary.replace(path)


def _measured_attempt_overlap(
    intervals: Sequence[tuple[int, int]],
) -> dict[str, Any]:
    """Integrate measured-command overlap without calling it CPU concurrency."""
    if not intervals:
        return {
            "window_start_ns": None,
            "window_end_ns": None,
            "window_sec": None,
            "peak_measured_attempts": 0,
            "time_sec_by_measured_attempts": {},
        }
    changes: dict[int, int] = {}
    for start_ns, end_ns in intervals:
        changes[start_ns] = changes.get(start_ns, 0) + 1
        changes[end_ns] = changes.get(end_ns, 0) - 1
    ordered = sorted(changes)
    cursor_ns = ordered[0]
    active = 0
    peak = 0
    duration_ns_by_active: dict[int, int] = {}
    for timestamp_ns in ordered:
        if timestamp_ns > cursor_ns:
            duration_ns_by_active[active] = duration_ns_by_active.get(active, 0) + timestamp_ns - cursor_ns
        active += changes[timestamp_ns]
        peak = max(peak, active)
        cursor_ns = timestamp_ns
    if active != 0:
        raise RunExecutionError("measured-attempt overlap intervals did not close")
    start_ns = ordered[0]
    end_ns = ordered[-1]
    return {
        "window_start_ns": start_ns,
        "window_end_ns": end_ns,
        "window_sec": (end_ns - start_ns) / 1_000_000_000,
        "peak_measured_attempts": peak,
        "time_sec_by_measured_attempts": {
            str(count): duration_ns / 1_000_000_000 for count, duration_ns in sorted(duration_ns_by_active.items())
        },
    }


def _v2_overall(
    spec: RunSpec | None,
    rows: Sequence[Mapping[str, Any]],
    *,
    interrupted: bool,
    outer_run_wall_sec: float,
    expected_attempt_count: int | None = None,
    expected_attempt_indexes: set[int] | None = None,
    pool_cpu_ids: Sequence[int] | None = None,
    requested_concurrency: int | None = None,
) -> dict[str, Any]:
    """Compute batch metrics only after all declared attempts are retained.

    Completeness and attempt evidence are checked before capacity math.  An
    impossible aggregate CPU value invalidates the batch rather than being
    clipped into an apparently valid utilization result.
    """
    if spec is not None:
        expected_attempt_count = len(spec.tasks) * spec.repetitions
        expected_attempt_indexes = set(range(1, expected_attempt_count + 1))
        pool_cpu_ids = spec.cpu_ids
        requested_concurrency = spec.concurrency
    if (
        expected_attempt_count is None
        or expected_attempt_indexes is None
        or pool_cpu_ids is None
        or requested_concurrency is None
    ):
        raise RunExecutionError("concurrent summary inputs are incomplete")
    valid_rows = [row for row in rows if row.get("duration_valid") is True]
    valid_durations = [float(row["duration_sec"]) for row in valid_rows]
    measured_window_rows = [
        row
        for row in rows
        if isinstance(row.get("window_start_ns"), int)
        and not isinstance(row.get("window_start_ns"), bool)
        and isinstance(row.get("window_end_ns"), int)
        and not isinstance(row.get("window_end_ns"), bool)
        and int(row["window_start_ns"]) >= 0
        and int(row["window_end_ns"]) > int(row["window_start_ns"])
    ]
    measured_intervals = [(int(row["window_start_ns"]), int(row["window_end_ns"])) for row in measured_window_rows]
    valid_intervals = [
        (int(row["window_start_ns"]), int(row["window_end_ns"]))
        for row in valid_rows
        if isinstance(row.get("window_start_ns"), int)
        and not isinstance(row.get("window_start_ns"), bool)
        and isinstance(row.get("window_end_ns"), int)
        and not isinstance(row.get("window_end_ns"), bool)
        and int(row["window_start_ns"]) >= 0
        and int(row["window_end_ns"]) > int(row["window_start_ns"])
    ]
    overlap = _measured_attempt_overlap(measured_intervals)
    observed_attempt_indexes = [
        row.get("attempt_index")
        for row in rows
        if isinstance(row.get("attempt_index"), int) and not isinstance(row.get("attempt_index"), bool)
    ]
    cpu_usage_complete = all(
        isinstance(row.get("cpu_usage_sec"), (int, float))
        and not isinstance(row.get("cpu_usage_sec"), bool)
        and math.isfinite(float(row["cpu_usage_sec"]))
        and float(row["cpu_usage_sec"]) >= 0
        for row in valid_rows
    )
    invalid_reasons: list[str] = []
    if interrupted:
        invalid_reasons.append("run was interrupted")
    if len(rows) != expected_attempt_count:
        invalid_reasons.append(f"retained {len(rows)} of {expected_attempt_count} declared attempts")
    if (
        len(observed_attempt_indexes) != len(rows)
        or set(observed_attempt_indexes) != expected_attempt_indexes
        or len(set(observed_attempt_indexes)) != len(observed_attempt_indexes)
    ):
        invalid_reasons.append("retained attempt indexes are incomplete or duplicated")
    if len(valid_rows) != expected_attempt_count:
        invalid_reasons.append(
            f"only {len(valid_rows)} of {expected_attempt_count} attempts have valid duration evidence"
        )
    if len(valid_intervals) != expected_attempt_count:
        invalid_reasons.append(
            f"only {len(valid_intervals)} of {expected_attempt_count} valid attempts have usable measured windows"
        )
    if not cpu_usage_complete or len(valid_rows) != expected_attempt_count:
        invalid_reasons.append("not every declared attempt has valid cgroup CPU usage")
    if int(overlap["peak_measured_attempts"]) > requested_concurrency:
        invalid_reasons.append(
            "measured-attempt overlap exceeds requested concurrency: "
            f"peak={overlap['peak_measured_attempts']}, requested={requested_concurrency}"
        )

    base_complete = not invalid_reasons
    candidate_batch_wall_sec = float(overlap["window_sec"]) if base_complete else None
    candidate_total_cpu_sec = math.fsum(float(row["cpu_usage_sec"]) for row in valid_rows) if base_complete else None
    cpu_capacity_sec = candidate_batch_wall_sec * len(pool_cpu_ids) if candidate_batch_wall_sec is not None else None
    cpu_capacity_tolerance_sec = (
        max(
            expected_attempt_count * _CPU_CAPACITY_COUNTER_TOLERANCE_SEC_PER_ATTEMPT,
            cpu_capacity_sec * _CPU_CAPACITY_FLOAT_TOLERANCE_FRACTION,
        )
        if cpu_capacity_sec is not None
        else None
    )
    if (
        candidate_total_cpu_sec is not None
        and cpu_capacity_sec is not None
        and cpu_capacity_tolerance_sec is not None
        and candidate_total_cpu_sec > cpu_capacity_sec + cpu_capacity_tolerance_sec
    ):
        invalid_reasons.append(
            "aggregate cgroup CPU exceeds declared pool capacity: "
            f"usage={candidate_total_cpu_sec:.9f}s, capacity={cpu_capacity_sec:.9f}s, "
            f"tolerance={cpu_capacity_tolerance_sec:.9f}s"
        )
    complete = not invalid_reasons
    batch_wall_sec = candidate_batch_wall_sec if complete else None
    total_cpu_sec = candidate_total_cpu_sec if complete else None
    aggregate_average_cores = (
        total_cpu_sec / batch_wall_sec
        if total_cpu_sec is not None and batch_wall_sec is not None and batch_wall_sec > 0
        else None
    )
    requested_concurrency_sec = float(overlap["time_sec_by_measured_attempts"].get(str(requested_concurrency), 0.0))
    return {
        "attempt_count": len(rows),
        "declared_attempt_count": expected_attempt_count,
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "valid_duration_count": len(valid_rows),
        "summed_service_time_sec": (math.fsum(valid_durations) if valid_durations else None),
        "interrupted": interrupted,
        "valid_for_throughput": complete,
        "throughput_invalid_reasons": invalid_reasons,
        "outer_run_wall_sec": outer_run_wall_sec,
        "batch_window_start_ns": overlap["window_start_ns"],
        "batch_window_end_ns": overlap["window_end_ns"],
        "measured_window_count": len(measured_window_rows),
        "measured_window_span_sec": overlap["window_sec"],
        "batch_wall_sec": batch_wall_sec,
        "throughput_tasks_per_sec": (
            expected_attempt_count / batch_wall_sec if batch_wall_sec is not None and batch_wall_sec > 0 else None
        ),
        "throughput_tasks_per_hour": (
            3600 * expected_attempt_count / batch_wall_sec
            if batch_wall_sec is not None and batch_wall_sec > 0
            else None
        ),
        "total_cgroup_cpu_sec": total_cpu_sec,
        "cpu_pool_capacity_sec": cpu_capacity_sec,
        "cpu_pool_capacity_tolerance_sec": cpu_capacity_tolerance_sec,
        "aggregate_average_logical_cores": aggregate_average_cores,
        "pool_utilization_percent": (
            100 * aggregate_average_cores / len(pool_cpu_ids) if aggregate_average_cores is not None else None
        ),
        "requested_concurrency": requested_concurrency,
        "achieved_peak_measured_attempts": overlap["peak_measured_attempts"],
        "time_sec_by_measured_attempts": overlap["time_sec_by_measured_attempts"],
        "requested_concurrency_sec": requested_concurrency_sec,
        "requested_concurrency_fraction": (
            requested_concurrency_sec / batch_wall_sec if batch_wall_sec is not None and batch_wall_sec > 0 else None
        ),
    }


def _summary(
    spec: RunSpec,
    rows: Sequence[Mapping[str, Any]],
    *,
    host_cpu: Mapping[str, Any],
    interrupted: bool = False,
    outer_run_wall_sec: float | None = None,
) -> dict[str, Any]:
    tasks: list[dict[str, Any]] = []
    for task in spec.tasks:
        task_rows = [row for row in rows if row["task_id"] == task.task_id]
        valid_durations = [
            float(row["duration_sec"])
            for row in task_rows
            if row.get("duration_valid") is True and isinstance(row.get("duration_sec"), (int, float))
        ]
        image_ids = sorted(
            {row["image_id"] for row in task_rows if isinstance(row.get("image_id"), str) and row["image_id"]}
        )
        execution_plan_digests = sorted(
            {
                row["execution_plan_digest"]
                for row in task_rows
                if isinstance(row.get("execution_plan_digest"), str) and row["execution_plan_digest"]
            }
        )
        complete = len(task_rows) == spec.repetitions and len(valid_durations) == spec.repetitions
        tasks.append(
            {
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "image_ids": image_ids,
                "execution_plan_digests": execution_plan_digests,
                "attempt_count": len(task_rows),
                "completed": sum(row["status"] == "completed" for row in task_rows),
                "failed": sum(row["status"] == "failed" for row in task_rows),
                "valid_duration_count": len(valid_durations),
                "median_duration_sec": (statistics.median(valid_durations) if complete else None),
            }
        )
    valid_durations = [
        float(row["duration_sec"])
        for row in rows
        if row.get("duration_valid") is True and isinstance(row.get("duration_sec"), (int, float))
    ]
    total = math.fsum(valid_durations)
    complete = bool(rows) and len(valid_durations) == len(rows)
    summary = {
        "schema_version": (
            SUMMARY_SCHEMA_VERSION_V1 if spec.schema_version == SCHEMA_VERSION_V1 else SUMMARY_SCHEMA_VERSION_V2
        ),
        "manifest": str(spec.manifest_path),
        "manifest_sha256": spec.manifest_sha256,
        "cpus": spec.cpus,
        "host_cpu": dict(host_cpu),
        "controller_runtime": "host-mount",
        "controller_runtime_content_bound": False,
        "controller_runtime_limitation": (
            "Host Python and system-library contents are not fully content-bound; "
            "use matching host runtime installations."
        ),
        "duration_clock": "host-monotonic",
        "duration_boundary": "measured replay command",
        "repetitions": spec.repetitions,
        "results_csv": "results.csv",
        "tasks": tasks,
        "overall": {
            "attempt_count": len(rows),
            "completed": sum(row["status"] == "completed" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "valid_duration_count": len(valid_durations),
            "valid_duration_total_sec": total if valid_durations else None,
            "replay_tasks_per_sec": (len(rows) / total if complete and total > 0 else None),
        },
    }
    if spec.schema_version == SCHEMA_VERSION_V2:
        if outer_run_wall_sec is None:
            raise RunExecutionError("v2 summary requires outer run wall time")
        summary.update(
            {
                "concurrency": spec.concurrency,
                "scheduling": {
                    "mode": "bounded-fifo-shared-cpuset",
                    "container_cpu_placement": "all attempts share cpus",
                    "completion_index_semantics": "worker-return-order",
                    "cpu_exclusivity_verified": False,
                },
                "overall": _v2_overall(
                    spec,
                    rows,
                    interrupted=interrupted,
                    outer_run_wall_sec=outer_run_wall_sec,
                ),
            }
        )
    return summary


def _ordered_summary(
    spec: OrderedRunSpec,
    rows: Sequence[dict[str, Any]],
    *,
    host_cpu: Mapping[str, Any],
    interrupted: bool,
    outer_run_wall_sec: float,
    wave_outer_wall_sec: Mapping[int, float],
    controller_affinity: Mapping[str, Any],
    controller_source_digest: str | None = None,
    enroot_runtime_provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize each wave independently and fail its headline metrics closed.

    A wave's requested lane set is the union of its launch CPU IDs. Unlike
    waves are deliberately not collapsed into one throughput number, and
    invalid launch evidence keeps the affected wave out of headline analysis.
    """
    if spec.defaults.runtime == "enroot":
        for wave_index in range(1, len(spec.waves) + 1):
            wave_rows = [row for row in rows if row.get("wave_index") == wave_index]
            enroot_runtime.invalidate_duration_bounds(
                wave_rows,
                outer_wall_sec=wave_outer_wall_sec.get(wave_index, 0.0),
            )
    task_summaries: list[dict[str, Any]] = []
    for task in spec.tasks:
        task_rows = [row for row in rows if row["task_id"] == task.task_id]
        task_summaries.append(
            {
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "image_ids": sorted(
                    {
                        str(row["image_id"])
                        for row in task_rows
                        if isinstance(row.get("image_id"), str) and row["image_id"]
                    }
                ),
                "execution_plan_digests": sorted(
                    {
                        str(row["execution_plan_digest"])
                        for row in task_rows
                        if isinstance(row.get("execution_plan_digest"), str) and row["execution_plan_digest"]
                    }
                ),
                "attempt_count": len(task_rows),
                "completed": sum(row["status"] == "completed" for row in task_rows),
                "failed": sum(row["status"] == "failed" for row in task_rows),
                "valid_duration_count": sum(row.get("duration_valid") is True for row in task_rows),
            }
        )

    wave_summaries: list[dict[str, Any]] = []
    first_attempt_index = 1
    for wave_index, wave in enumerate(spec.waves, start=1):
        wave_rows = [row for row in rows if row.get("wave_index") == wave_index]
        attempt_indexes = set(range(first_attempt_index, first_attempt_index + len(wave.launches)))
        first_attempt_index += len(wave.launches)
        pool_cpu_ids = tuple(sorted({cpu_id for launch in wave.launches for cpu_id in launch.cpu_ids}))
        if spec.defaults.runtime == "enroot":
            overall = enroot_runtime.wave_overall(
                wave_rows,
                interrupted=interrupted,
                outer_wall_sec=wave_outer_wall_sec.get(wave_index, 0.0),
                expected_attempt_indexes=attempt_indexes,
                require_full_overlap=True,
            )
        else:
            overall = _v2_overall(
                None,
                wave_rows,
                interrupted=interrupted,
                outer_run_wall_sec=wave_outer_wall_sec.get(wave_index, 0.0),
                expected_attempt_count=len(wave.launches),
                expected_attempt_indexes=attempt_indexes,
                pool_cpu_ids=pool_cpu_ids,
                requested_concurrency=len(wave.launches),
            )
        valid_durations = [
            float(row["duration_sec"])
            for row in wave_rows
            if row.get("duration_valid") is True and isinstance(row.get("duration_sec"), (int, float))
        ]
        if overall["valid_for_throughput"] is True and valid_durations:
            overall["mean_task_duration_sec"] = statistics.fmean(valid_durations)
            overall["median_task_duration_sec"] = statistics.median(valid_durations)
            if spec.defaults.runtime == "enroot":
                overall["mean_batch_relative_completion_sec"] = None
            else:
                batch_start_ns = overall["batch_window_start_ns"]
                assert isinstance(batch_start_ns, int)
                completions = [(int(row["window_end_ns"]) - batch_start_ns) / 1_000_000_000 for row in wave_rows]
                overall["mean_batch_relative_completion_sec"] = statistics.fmean(completions)
        else:
            overall["mean_task_duration_sec"] = None
            overall["median_task_duration_sec"] = None
            overall["mean_batch_relative_completion_sec"] = None
        wave_summaries.append(
            {
                "wave_index": wave_index,
                "wave_id": wave.wave_id,
                "scored": wave.scored,
                "metadata": dict(wave.metadata),
                "cpus": ",".join(map(str, pool_cpu_ids)),
                "logical_cpu_count": len(pool_cpu_ids),
                **(
                    {
                        "requested_cpu_lanes": ",".join(map(str, pool_cpu_ids)),
                        "requested_logical_cpu_lane_count": len(pool_cpu_ids),
                    }
                    if spec.defaults.runtime == "enroot"
                    else {}
                ),
                "launch_count": len(wave.launches),
                "overall": overall,
            }
        )

    declared_attempt_count = sum(len(wave.launches) for wave in spec.waves)
    all_rows_valid = len(rows) == declared_attempt_count and all(row.get("duration_valid") is True for row in rows)
    all_waves_valid = all(wave["overall"]["valid_for_throughput"] is True for wave in wave_summaries)
    valid_for_analysis = not interrupted and all_rows_valid and all_waves_valid
    return {
        "schema_version": SUMMARY_SCHEMA_VERSION_V3,
        "manifest": str(spec.manifest_path),
        "manifest_sha256": spec.manifest_sha256,
        "input_root": str(spec.input_root),
        **({"runtime": "enroot"} if spec.defaults.runtime == "enroot" else {}),
        "host_cpu": dict(host_cpu),
        "controller_affinity": dict(controller_affinity),
        "controller_runtime": spec.defaults.controller_runtime,
        **({"controller_source_sha256": controller_source_digest} if spec.defaults.runtime == "enroot" else {}),
        **(
            {"enroot_runtime_provenance": dict(enroot_runtime_provenance or {})}
            if spec.defaults.runtime == "enroot"
            else {}
        ),
        "controller_runtime_content_bound": False,
        "controller_runtime_limitation": (
            "Host Python and system-library contents are not fully content-bound; "
            "use matching host runtime installations."
        ),
        "duration_clock": (enroot_runtime.ENROOT_TASK_CLOCK if spec.defaults.runtime == "enroot" else "host-monotonic"),
        "duration_boundary": "measured replay command",
        "pacing": spec.defaults.pacing,
        "network_policy": {
            "mode": spec.defaults.network_policy,
            "allowed_hosts": list(spec.defaults.allowed_hosts),
        },
        "profile_sets": list(spec.defaults.profile_sets),
        "results_csv": "results.csv",
        "scheduling": {
            "mode": "ordered-waves",
            "authoring_mode": ("compact-worker-pool" if spec.defaults.placement is not None else "explicit-launches"),
            "worker_cpus": spec.defaults.worker_cpus,
            "placement": spec.defaults.placement,
            "within_wave_dispatch": "concurrent-submit",
            "measured_start_alignment": (
                "observed-not-synchronized" if spec.defaults.runtime == "enroot" else "not-enforced"
            ),
            "wave_barrier": "all launches return before the next wave is dispatched",
            "container_cpu_placement": (
                "taskset-requested-per-launch-affinity" if spec.defaults.runtime == "enroot" else "per-launch cpuset"
            ),
            **(
                {
                    "throughput_wall_clock": "CLOCK_BOOTTIME",
                    "replay_throughput_boundary": (
                        "earliest measured replay-command start through latest " "measured replay-command end"
                    ),
                    "lifecycle_throughput_boundary": (
                        "host wave dispatch through Enroot create, preparation, " "replay, remove, and worker return"
                    ),
                }
                if spec.defaults.runtime == "enroot"
                else {}
            ),
            "completion_index_semantics": "worker-return-order",
            "cpu_exclusivity_verified": False,
            **({"independent_performance_unit": "wave"} if spec.defaults.runtime == "enroot" else {}),
        },
        "tasks": task_summaries,
        "waves": wave_summaries,
        "overall": {
            "attempt_count": len(rows),
            "declared_attempt_count": declared_attempt_count,
            "completed": sum(row["status"] == "completed" for row in rows),
            "failed": sum(row["status"] == "failed" for row in rows),
            "valid_duration_count": sum(row.get("duration_valid") is True for row in rows),
            "wave_count": len(spec.waves),
            "valid_wave_count": sum(wave["overall"]["valid_for_throughput"] is True for wave in wave_summaries),
            "interrupted": interrupted,
            "outer_run_wall_sec": outer_run_wall_sec,
            **(
                {
                    "analysis_mode": "wall-only",
                    "valid_for_wall_only_analysis": valid_for_analysis,
                    "valid_for_cpu_analysis": False,
                }
                if spec.defaults.runtime == "enroot"
                else {}
            ),
            "valid_for_analysis": valid_for_analysis,
        },
    }


def _invalidate_plan_drift(rows: Sequence[dict[str, Any]]) -> None:
    """Reject aggregation if one task's immutable execution plan changed."""
    groups = {
        (
            str(row["task_id"]),
            str(row.get("cpus")) if "wave_id" in row else None,
        )
        for row in rows
    }
    for task_id, cpus in groups:
        task_rows = [
            row
            for row in rows
            if row["task_id"] == task_id and ((str(row.get("cpus")) if "wave_id" in row else None) == cpus)
        ]
        digests = {
            row["execution_plan_digest"] for row in task_rows if isinstance(row.get("execution_plan_digest"), str)
        }
        if len(digests) <= 1:
            continue
        for row in task_rows:
            row["duration_valid"] = False
            drift_reason = "execution plan changed across repetitions"
            row["reason"] = "; ".join(reason for reason in (str(row.get("reason") or ""), drift_reason) if reason)


def _write_json(payload: Mapping[str, Any], path: Path) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def _resolved_atif_path(task: RunTaskSpec, *, input_root: Path) -> str:
    """Prefer a portable input-root-relative source path when it is safe."""
    try:
        return task.atif.relative_to(input_root).as_posix()
    except ValueError:
        return task.atif.as_posix()


def _resolved_ordered_payload(
    spec: OrderedRunSpec,
    preflights: Mapping[tuple[str, str], _Preflight],
) -> dict[str, Any]:
    """Serialize the normalized v3 plan without authoring shorthand."""
    image_ids_by_task: dict[str, str] = {}
    for (task_id, _cpus), preflight in preflights.items():
        previous = image_ids_by_task.setdefault(task_id, preflight.image_id)
        if previous != preflight.image_id:
            raise RunExecutionError(f"resolved image ID changed across placements for task {task_id!r}")

    tasks: list[dict[str, Any]] = []
    for task in spec.tasks:
        image_id = image_ids_by_task.get(task.task_id)
        if image_id is None:
            raise RunExecutionError(f"resolved plan is missing preflight image ID for task {task.task_id!r}")
        item: dict[str, Any] = {
            "id": task.task_id,
            "atif": _resolved_atif_path(task, input_root=spec.input_root),
            "expected_source_sha256": task.source_sha256,
            "image": task.image,
            "expected_image_id": image_id,
        }
        if task.action_timeout_sec is not None:
            item["action_timeout_sec"] = task.action_timeout_sec
        tasks.append(item)

    defaults: dict[str, Any] = {
        "pacing": spec.defaults.pacing,
        "network_policy": spec.defaults.network_policy,
        "allowed_hosts": list(spec.defaults.allowed_hosts),
        "profile_sets": list(spec.defaults.profile_sets),
        "controller_runtime": spec.defaults.controller_runtime,
        "controller_cpus": spec.defaults.controller_cpus,
        "action_timeout_sec": spec.defaults.action_timeout_sec,
    }
    if spec.defaults.runtime == "enroot":
        defaults["runtime"] = "enroot"

    return {
        "schema_version": SCHEMA_VERSION_V3,
        "output": spec.output.as_posix(),
        "input_root": spec.input_root.as_posix(),
        "defaults": defaults,
        "tasks": tasks,
        "waves": [
            {
                "id": wave.wave_id,
                "scored": wave.scored,
                "metadata": dict(wave.metadata),
                "launches": [{"task": launch.task_id, "cpus": launch.cpus} for launch in wave.launches],
            }
            for wave in spec.waves
        ],
    }


def _write_resolved_ordered_manifest(
    spec: OrderedRunSpec,
    preflights: Mapping[tuple[str, str], _Preflight],
) -> tuple[str, str]:
    """Write the immutable, explicit execution plan retained with results."""
    filename = "resolved-run.yaml"
    path = spec.output / filename
    temporary = path.with_name(f".{path.name}.tmp")
    source_manifest = json.dumps(str(spec.manifest_path), ensure_ascii=False)
    controller_digests = {
        preflight.controller_source_sha256
        for preflight in preflights.values()
        if preflight.controller_source_sha256 is not None
    }
    controller_header = (
        f"# controller_source_sha256: {next(iter(controller_digests))}\n" if len(controller_digests) == 1 else ""
    )
    runtime_digests = {
        provenance.get("comparability_sha256")
        for preflight in preflights.values()
        if isinstance((provenance := preflight.enroot_runtime_provenance), Mapping)
    }
    runtime_header = (
        f"# enroot_runtime_provenance_sha256: {next(iter(runtime_digests))}\n" if len(runtime_digests) == 1 else ""
    )
    header = (
        "# AUTO-GENERATED: fully resolved tasktrace-run.v3 execution plan.\n"
        "# DO NOT EDIT: edit the source manifest and run it again.\n"
        f"# source_manifest: {source_manifest}\n"
        f"# source_manifest_sha256: {spec.manifest_sha256}\n" + controller_header + runtime_header
    )
    body = yaml.safe_dump(
        _resolved_ordered_payload(spec, preflights),
        sort_keys=False,
        allow_unicode=True,
    )
    try:
        temporary.write_text(header + body)
        temporary.replace(path)
    except OSError as exc:
        raise RunExecutionError(f"unable to write resolved run manifest {path}: {exc}") from exc
    return filename, _file_sha256(path)


def _attempts(spec: RunSpec) -> tuple[_Attempt, ...]:
    attempts: list[_Attempt] = []
    index = 0
    for repetition in range(1, spec.repetitions + 1):
        for task in spec.tasks:
            index += 1
            attempts.append(
                _Attempt(
                    attempt_index=index,
                    repetition=repetition,
                    task=task,
                )
            )
    return tuple(attempts)


def _wave_attempts(spec: OrderedRunSpec) -> tuple[_WaveAttempt, ...]:
    task_by_id = {task.task_id: task for task in spec.tasks}
    attempts: list[_WaveAttempt] = []
    attempt_index = 0
    for wave_index, wave in enumerate(spec.waves, start=1):
        for launch_index, launch in enumerate(wave.launches, start=1):
            attempt_index += 1
            attempts.append(
                _WaveAttempt(
                    attempt_index=attempt_index,
                    wave_index=wave_index,
                    launch_index=launch_index,
                    wave=wave,
                    launch=launch,
                    task=task_by_id[launch.task_id],
                )
            )
    return tuple(attempts)


def _attempt_profile_path(spec: RunSpec, attempt: _Attempt) -> Path:
    return spec.output / "tasks" / attempt.task.task_id / f"rep-{attempt.repetition:04d}" / "profile"


def _wave_attempt_profile_path(
    spec: OrderedRunSpec,
    attempt: _WaveAttempt,
) -> Path:
    return (
        spec.output
        / "waves"
        / f"{attempt.wave_index:04d}-{attempt.wave.wave_id}"
        / f"launch-{attempt.launch_index:04d}-{attempt.task.task_id}"
        / "profile"
    )


def _run_attempt(
    spec: RunSpec | OrderedRunSpec,
    attempt: _Attempt | _WaveAttempt,
    preflight: _Preflight,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    task = attempt.task
    ordered_attempt = attempt if isinstance(attempt, _WaveAttempt) else None
    profile = (
        _wave_attempt_profile_path(spec, ordered_attempt)
        if isinstance(spec, OrderedRunSpec) and ordered_attempt is not None
        else _attempt_profile_path(spec, attempt)  # type: ignore[arg-type]
    )
    profile_relative = profile.relative_to(spec.output).as_posix()
    attempt_index = attempt.attempt_index if spec.schema_version != SCHEMA_VERSION_V1 else None
    runtime = spec.defaults.runtime if isinstance(spec, OrderedRunSpec) else "docker"
    repetition = attempt.repetition if isinstance(attempt, _Attempt) else 1
    try:
        current_source_sha256 = _file_sha256(task.atif)
    except OSError as exc:
        return _failed_row(
            task=task,
            repetition=repetition,
            profile_dir=profile_relative,
            image_id=preflight.image_id,
            reason=f"unable to re-read ATIF source: {exc}",
            runtime=runtime,
            attempt_index=attempt_index,
            ordered_attempt=ordered_attempt,
        )
    if current_source_sha256 != task.source_sha256:
        return _failed_row(
            task=task,
            repetition=repetition,
            profile_dir=profile_relative,
            image_id=preflight.image_id,
            reason=(
                "ATIF source changed after the run file was loaded: "
                f"expected {task.source_sha256}, got {current_source_sha256}"
            ),
            runtime=runtime,
            attempt_index=attempt_index,
            ordered_attempt=ordered_attempt,
        )
    try:
        config = _profile_config(
            spec,
            task,
            profile,
            cpus=(ordered_attempt.launch.cpus if ordered_attempt is not None else None),
        )
        report = (
            run_atif_trajectory_profile(config)
            if cancel_requested is None or config.runtime != "docker"
            else run_atif_trajectory_profile(
                config,
                cancel_requested=cancel_requested,
            )
        )
    except Exception as exc:
        return _failed_row(
            task=task,
            repetition=repetition,
            profile_dir=profile_relative,
            image_id=preflight.image_id,
            reason=f"profile failed: {exc}",
            runtime=runtime,
            attempt_index=attempt_index,
            ordered_attempt=ordered_attempt,
        )
    if not isinstance(report, Mapping):
        return _failed_row(
            task=task,
            repetition=repetition,
            profile_dir=profile_relative,
            image_id=preflight.image_id,
            reason="profile returned a non-object report",
            runtime=runtime,
            attempt_index=attempt_index,
            ordered_attempt=ordered_attempt,
        )
    projection = _project_duration(
        report,
        task=task,
        preflight=preflight,
        require_common_window=spec.schema_version != SCHEMA_VERSION_V1,
        ordered_spec=spec if isinstance(spec, OrderedRunSpec) else None,
        launch=ordered_attempt.launch if ordered_attempt is not None else None,
    )
    row: dict[str, Any] = {
        "task_id": task.task_id,
        "repetition": repetition,
        "status": "completed",
        "duration_sec": projection.duration_sec,
        "duration_valid": projection.valid,
        "reason": projection.reason,
        "profile_dir": profile_relative,
        "source_sha256": task.source_sha256,
        "image_id": projection.image_id,
        "execution_plan_digest": projection.execution_plan_digest,
    }
    if config.runtime == "enroot":
        row.update(
            {
                "cleanup_valid": getattr(projection, "cleanup_valid", None),
                "clock_reference": getattr(projection, "clock_reference", None),
            }
        )
    if spec.schema_version != SCHEMA_VERSION_V1:
        row.update(
            {
                "attempt_index": attempt.attempt_index,
                "completion_index": None,
                "window_start_ns": projection.window_start_ns,
                "window_end_ns": projection.window_end_ns,
                "cpu_usage_sec": projection.cpu_usage_sec,
                "average_logical_cores": projection.average_logical_cores,
            }
        )
    if ordered_attempt is not None:
        row.update(
            {
                "wave_index": ordered_attempt.wave_index,
                "wave_id": ordered_attempt.wave.wave_id,
                "wave_scored": ordered_attempt.wave.scored,
                "launch_index": ordered_attempt.launch_index,
                "cpus": ordered_attempt.launch.cpus,
            }
        )
    return row


def _execute_v1(
    spec: RunSpec,
    attempts: Sequence[_Attempt],
    preflights: Mapping[str, _Preflight],
) -> list[dict[str, Any]]:
    return [_run_attempt(spec, attempt, preflights[attempt.task.task_id]) for attempt in attempts]


def _execute_v2(
    spec: RunSpec,
    attempts: Sequence[_Attempt],
    preflights: Mapping[str, _Preflight],
) -> tuple[list[dict[str, Any]], bool]:
    """Run a repetition-major FIFO with bounded refill and safe interruption.

    Rows are returned in declared-attempt order while ``completion_index``
    records worker-return order.  Ctrl-C broadcasts cooperative cancellation,
    drains active workers through normal cleanup, and creates invalid rows for
    attempts that never started so a partial batch cannot look complete.
    """
    pending = deque(attempts)
    cancel_requested = threading.Event()
    futures: dict[Future[dict[str, Any]], _Attempt] = {}
    rows_by_attempt: dict[int, dict[str, Any]] = {}
    worker_rows: dict[int, dict[str, Any]] = {}
    completion_indices: dict[int, int] = {}
    completion_lock = threading.Lock()
    completion_index = 0
    interrupted = False
    executor: ThreadPoolExecutor | None = None

    def drive_attempt(attempt: _Attempt) -> dict[str, Any]:
        nonlocal completion_index
        try:
            row = _run_attempt(
                spec,
                attempt,
                preflights[attempt.task.task_id],
                cancel_requested=cancel_requested.is_set,
            )
            with completion_lock:
                worker_rows[attempt.attempt_index] = row
            return row
        finally:
            with completion_lock:
                completion_index += 1
                completion_indices[attempt.attempt_index] = completion_index

    def submit_one() -> None:
        if executor is None or not pending:
            raise RunExecutionError("concurrent attempt dispatch state is invalid")
        attempt = pending[0]
        future = executor.submit(
            drive_attempt,
            attempt,
        )
        futures[future] = attempt
        removed = pending.popleft()
        if removed is not attempt:
            raise RunExecutionError("concurrent FIFO attempt queue changed during dispatch")

    def retain_future(future: Future[dict[str, Any]], attempt: _Attempt) -> None:
        if attempt.attempt_index in rows_by_attempt:
            return
        if future.cancelled():
            profile = _attempt_profile_path(spec, attempt)
            row = _failed_row(
                task=attempt.task,
                repetition=attempt.repetition,
                profile_dir=profile.relative_to(spec.output).as_posix(),
                image_id=preflights[attempt.task.task_id].image_id,
                reason="run interrupted before attempt start",
                runtime="docker",
                attempt_index=attempt.attempt_index,
            )
        else:
            try:
                row = future.result()
            except Exception as exc:
                profile = _attempt_profile_path(spec, attempt)
                row = _failed_row(
                    task=attempt.task,
                    repetition=attempt.repetition,
                    profile_dir=profile.relative_to(spec.output).as_posix(),
                    image_id=preflights[attempt.task.task_id].image_id,
                    reason=(
                        f"profile interrupted: {exc}" if cancel_requested.is_set() else f"profile worker failed: {exc}"
                    ),
                    runtime="docker",
                    attempt_index=attempt.attempt_index,
                )
        with completion_lock:
            row["completion_index"] = completion_indices.get(attempt.attempt_index)
        rows_by_attempt[attempt.attempt_index] = row

    try:
        executor = ThreadPoolExecutor(
            max_workers=spec.concurrency,
            thread_name_prefix="tasktrace-run",
        )
        for _index in range(min(spec.concurrency, len(pending))):
            submit_one()
        while futures:
            completed, _not_done = wait(
                tuple(futures),
                return_when=FIRST_COMPLETED,
            )
            for future in sorted(
                completed,
                key=lambda item: (
                    completion_indices.get(futures[item].attempt_index, math.inf),
                    futures[item].attempt_index,
                ),
            ):
                attempt = futures.pop(future)
                retain_future(future, attempt)
                if pending and not cancel_requested.is_set():
                    submit_one()
    except KeyboardInterrupt:
        interrupted = True
        cancel_requested.set()
        for future in tuple(futures):
            future.cancel()
    except BaseException:
        cancel_requested.set()
        for future in tuple(futures):
            future.cancel()
        raise
    finally:
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)

    if interrupted:
        for future, attempt in sorted(
            futures.items(),
            key=lambda item: (
                completion_indices.get(item[1].attempt_index, math.inf),
                item[1].attempt_index,
            ),
        ):
            retain_future(future, attempt)
        for attempt in attempts:
            if attempt.attempt_index in rows_by_attempt:
                continue
            with completion_lock:
                recovered_row = worker_rows.get(attempt.attempt_index)
                recovered_completion_index = completion_indices.get(attempt.attempt_index)
            if recovered_row is not None:
                recovered_row["completion_index"] = recovered_completion_index
                rows_by_attempt[attempt.attempt_index] = recovered_row
                continue
            profile = _attempt_profile_path(spec, attempt)
            row = _failed_row(
                task=attempt.task,
                repetition=attempt.repetition,
                profile_dir=profile.relative_to(spec.output).as_posix(),
                image_id=preflights[attempt.task.task_id].image_id,
                reason="run interrupted before attempt dispatch",
                runtime="docker",
                attempt_index=attempt.attempt_index,
            )
            row["completion_index"] = recovered_completion_index
            rows_by_attempt[attempt.attempt_index] = row

    rows = [rows_by_attempt[index] for index in sorted(rows_by_attempt)]
    return rows, interrupted


def _invalidate_rows(rows: Sequence[dict[str, Any]], reason: str) -> None:
    for row in rows:
        row["duration_valid"] = False
        row["reason"] = "; ".join(value for value in (str(row.get("reason") or ""), reason) if value)


def _execute_v3(
    spec: OrderedRunSpec,
    attempts: Sequence[_WaveAttempt],
    preflights: Mapping[tuple[str, str], _Preflight],
) -> tuple[list[dict[str, Any]], bool, dict[int, float]]:
    """Dispatch each declared wave together and enforce a full wave barrier."""
    cancel_requested = threading.Event()
    rows_by_attempt: dict[int, dict[str, Any]] = {}
    worker_rows: dict[int, dict[str, Any]] = {}
    completion_indices: dict[int, int] = {}
    completion_lock = threading.Lock()
    completion_index = 0
    interrupted = False
    wave_outer_wall_sec: dict[int, float] = {}
    attempts_by_wave = {
        wave_index: [attempt for attempt in attempts if attempt.wave_index == wave_index]
        for wave_index in range(1, len(spec.waves) + 1)
    }

    def failed_attempt(attempt: _WaveAttempt, reason: str) -> dict[str, Any]:
        profile = _wave_attempt_profile_path(spec, attempt)
        return _failed_row(
            task=attempt.task,
            repetition=1,
            profile_dir=profile.relative_to(spec.output).as_posix(),
            image_id=preflights[(attempt.task.task_id, attempt.launch.cpus)].image_id,
            reason=reason,
            runtime=spec.defaults.runtime,
            attempt_index=attempt.attempt_index,
            ordered_attempt=attempt,
        )

    def drive_attempt(attempt: _WaveAttempt) -> dict[str, Any]:
        nonlocal completion_index
        try:
            # sched_setaffinity targets the calling thread.  Each pool worker
            # must therefore reapply host dispatch affinity before it spawns
            # Docker-client helpers for this attempt.
            _set_current_thread_affinity(spec.defaults.controller_cpu_ids)
            row = _run_attempt(
                spec,
                attempt,
                preflights[(attempt.task.task_id, attempt.launch.cpus)],
                cancel_requested=cancel_requested.is_set,
            )
            with completion_lock:
                worker_rows[attempt.attempt_index] = row
            return row
        finally:
            with completion_lock:
                completion_index += 1
                completion_indices[attempt.attempt_index] = completion_index

    def retain_future(future: Future[dict[str, Any]], attempt: _WaveAttempt) -> None:
        if attempt.attempt_index in rows_by_attempt:
            return
        if future.cancelled():
            row = failed_attempt(attempt, "run interrupted before attempt start")
        else:
            try:
                row = future.result()
            except Exception as exc:
                row = failed_attempt(
                    attempt,
                    (f"profile interrupted: {exc}" if cancel_requested.is_set() else f"profile worker failed: {exc}"),
                )
        with completion_lock:
            row["completion_index"] = completion_indices.get(attempt.attempt_index)
        rows_by_attempt[attempt.attempt_index] = row

    for wave_index in range(1, len(spec.waves) + 1):
        wave_attempts = attempts_by_wave[wave_index]
        wave_started_ns = _ordered_wall_clock_ns(spec.defaults.runtime)
        futures: dict[Future[dict[str, Any]], _WaveAttempt] = {}
        executor: ThreadPoolExecutor | None = None
        try:
            executor = ThreadPoolExecutor(
                max_workers=len(wave_attempts),
                thread_name_prefix=f"tasktrace-wave-{wave_index}",
            )
            for attempt in wave_attempts:
                future = executor.submit(drive_attempt, attempt)
                futures[future] = attempt
            wait(tuple(futures))
        except KeyboardInterrupt:
            interrupted = True
            cancel_requested.set()
            for future in tuple(futures):
                future.cancel()
        except BaseException:
            cancel_requested.set()
            for future in tuple(futures):
                future.cancel()
            raise
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
            wave_outer_wall_sec[wave_index] = (
                _ordered_wall_clock_ns(spec.defaults.runtime) - wave_started_ns
            ) / 1_000_000_000

        for future, attempt in sorted(
            futures.items(),
            key=lambda item: (
                completion_indices.get(item[1].attempt_index, math.inf),
                item[1].attempt_index,
            ),
        ):
            retain_future(future, attempt)
        if interrupted:
            break

    if interrupted:
        for attempt in attempts:
            if attempt.attempt_index in rows_by_attempt:
                continue
            with completion_lock:
                recovered_row = worker_rows.get(attempt.attempt_index)
                recovered_completion_index = completion_indices.get(attempt.attempt_index)
            if recovered_row is not None:
                recovered_row["completion_index"] = recovered_completion_index
                rows_by_attempt[attempt.attempt_index] = recovered_row
            else:
                row = failed_attempt(
                    attempt,
                    "run interrupted before attempt dispatch",
                )
                row["completion_index"] = recovered_completion_index
                rows_by_attempt[attempt.attempt_index] = row

    rows = [rows_by_attempt[index] for index in sorted(rows_by_attempt)]
    return rows, interrupted, wave_outer_wall_sec


def _execute_ordered_run(spec: OrderedRunSpec) -> RunExecutionResult:
    _validate_ordered_runtime_environment(spec)
    _ensure_output_absent(spec.output)
    attempts = _wave_attempts(spec)
    selected_cpu_ids = tuple(
        sorted(
            set(spec.defaults.controller_cpu_ids).union(
                cpu_id for wave in spec.waves for launch in wave.launches for cpu_id in launch.cpu_ids
            )
        )
    )
    host_cpu: Mapping[str, Any] | None = None
    preflights: dict[tuple[str, str], _Preflight] | None = None
    controller_digest: str | None = None
    enroot_provenance: Mapping[str, Any] | None = None
    if spec.defaults.runtime == "enroot":
        try:
            controller_digest = controller_source_sha256()
            enroot_provenance = enroot_runtime.runtime_provenance(
                controller_cpus=spec.defaults.controller_cpus,
                launch_cpu_sets=tuple(dict.fromkeys(launch.cpus for wave in spec.waves for launch in wave.launches)),
            )
        except (OSError, RuntimeError) as exc:
            raise RunExecutionError(f"runtime identity preflight failed: {exc}") from exc
        try:
            host_cpu = collect_host_cpu_identity(selected_cpu_ids)
        except HostIdentityError as exc:
            raise RunExecutionError(f"host CPU identity preflight failed: {exc}") from exc
        preflights = _preflight_ordered(
            spec,
            attempts,
            controller_digest=controller_digest,
            enroot_provenance=enroot_provenance,
        )

    controller_affinity: dict[str, Any]
    with _controller_affinity(spec) as controller_affinity:
        if host_cpu is None or preflights is None:
            try:
                host_cpu = collect_host_cpu_identity(selected_cpu_ids)
            except HostIdentityError as exc:
                raise RunExecutionError(f"host CPU identity preflight failed: {exc}") from exc
            preflights = _preflight_ordered(
                spec,
                attempts,
                controller_digest=controller_digest,
                enroot_provenance=enroot_provenance,
            )
        try:
            spec.output.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            raise RunExecutionError(f"output path was claimed while preflight ran: {spec.output}") from exc
        except OSError as exc:
            raise RunExecutionError(f"unable to create output directory {spec.output}: {exc}") from exc

        resolved_manifest, resolved_manifest_sha256 = _write_resolved_ordered_manifest(spec, preflights)
        run_started_ns = _ordered_wall_clock_ns(spec.defaults.runtime)
        rows, interrupted, wave_outer_wall_sec = _execute_v3(
            spec,
            attempts,
            preflights,
        )
        outer_run_wall_sec = (_ordered_wall_clock_ns(spec.defaults.runtime) - run_started_ns) / 1_000_000_000

    assert host_cpu is not None
    if spec.defaults.runtime == "enroot":
        enroot_runtime.invalidate_changed_images(spec.tasks, rows)
        try:
            final_controller_digest = controller_source_sha256()
        except (OSError, RuntimeError) as exc:
            _invalidate_rows(rows, f"unable to re-read controller source: {exc}")
        else:
            if final_controller_digest != controller_digest:
                _invalidate_rows(rows, "controller source changed during the run")
        try:
            final_enroot_provenance = enroot_runtime.runtime_provenance(
                controller_cpus=spec.defaults.controller_cpus,
                launch_cpu_sets=tuple(dict.fromkeys(launch.cpus for wave in spec.waves for launch in wave.launches)),
            )
        except (OSError, RuntimeError) as exc:
            _invalidate_rows(rows, f"unable to re-read Enroot runtime identity: {exc}")
        else:
            if final_enroot_provenance != enroot_provenance:
                _invalidate_rows(rows, "Enroot runtime identity changed during the run")
    _invalidate_plan_drift(rows)
    summary = _ordered_summary(
        spec,
        rows,
        host_cpu=host_cpu,
        interrupted=interrupted,
        outer_run_wall_sec=outer_run_wall_sec,
        wave_outer_wall_sec=wave_outer_wall_sec,
        controller_affinity=controller_affinity,
        controller_source_digest=controller_digest,
        enroot_runtime_provenance=enroot_provenance,
    )
    summary["resolved_manifest"] = resolved_manifest
    summary["resolved_manifest_sha256"] = resolved_manifest_sha256
    _write_csv(rows, spec.output / "results.csv", columns=RESULT_COLUMNS_V3)
    _write_json(summary, spec.output / "summary.json")
    exit_code = 130 if interrupted else (0 if summary["overall"]["valid_for_analysis"] is True else 1)
    return RunExecutionResult(
        exit_code=exit_code,
        rows=tuple(rows),
        summary=summary,
    )


def execute_run(spec: RunSpec | OrderedRunSpec) -> RunExecutionResult:
    """Preflight every task, then execute the declared run contract."""
    if isinstance(spec, OrderedRunSpec):
        return _execute_ordered_run(spec)
    _ensure_output_absent(spec.output)
    try:
        host_cpu = collect_host_cpu_identity(spec.cpu_ids)
    except HostIdentityError as exc:
        raise RunExecutionError(f"host CPU identity preflight failed: {exc}") from exc
    preflights = _preflight_all(spec)
    try:
        # Exclusive final-component creation claims this output after preflight.
        # A concurrent run using the same path can never mix its artifacts here.
        spec.output.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise RunExecutionError(f"output path was claimed while preflight ran: {spec.output}") from exc
    except OSError as exc:
        raise RunExecutionError(f"unable to create output directory {spec.output}: {exc}") from exc

    run_started_ns = time.monotonic_ns()
    attempts = _attempts(spec)
    interrupted = False
    if spec.schema_version == SCHEMA_VERSION_V1:
        rows = _execute_v1(spec, attempts, preflights)
    else:
        rows, interrupted = _execute_v2(spec, attempts, preflights)
    outer_run_wall_sec = (time.monotonic_ns() - run_started_ns) / 1_000_000_000

    _invalidate_plan_drift(rows)
    summary = _summary(
        spec,
        rows,
        host_cpu=host_cpu,
        interrupted=interrupted,
        outer_run_wall_sec=outer_run_wall_sec,
    )
    _write_csv(
        rows,
        spec.output / "results.csv",
        columns=(RESULT_COLUMNS if spec.schema_version == SCHEMA_VERSION_V1 else RESULT_COLUMNS_V2),
    )
    _write_json(summary, spec.output / "summary.json")
    all_attempts_valid = all(row["duration_valid"] is True for row in rows)
    if spec.schema_version == SCHEMA_VERSION_V2:
        all_attempts_valid = all_attempts_valid and summary["overall"]["valid_for_throughput"] is True
    exit_code = 130 if interrupted else (0 if all_attempts_valid else 1)
    return RunExecutionResult(
        exit_code=exit_code,
        rows=tuple(rows),
        summary=summary,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run ATIF tasks from a strict YAML file in fresh sandboxes "
            "(v1 runs sequentially; v2 is bounded concurrent; "
            "v3 uses ordered waves and may select Docker or Enroot)."
        )
    )
    parser.add_argument("run_file", metavar="FILE.yaml", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    try:
        spec = load_run_spec(args.run_file)
        result = execute_run(spec)
    except (RunSpecError, RunExecutionError, OSError) as exc:
        parser.error(str(exc))
    overall = result.summary["overall"]
    print(
        f"Completed: {overall['completed']}; failed: {overall['failed']}; "
        f"valid durations: {overall['valid_duration_count']}"
    )
    print(f"Results: {spec.output / 'results.csv'}")
    print(f"Summary: {spec.output / 'summary.json'}")
    if spec.schema_version == SCHEMA_VERSION_V2:
        throughput = overall["throughput_tasks_per_sec"]
        utilization = overall["pool_utilization_percent"]
        if throughput is not None and utilization is not None:
            print(f"Throughput: {throughput:.6f} tasks/sec; " f"pool utilization: {utilization:.2f}%")
        else:
            print("Throughput: unavailable because the concurrent run was invalid")
    elif spec.schema_version == SCHEMA_VERSION_V3:
        detail = (
            "per-wave wall latency and throughput are in summary.json"
            if isinstance(spec, OrderedRunSpec) and spec.defaults.runtime == "enroot"
            else "per-wave latency, throughput, and utilization are in summary.json"
        )
        print(f"Waves valid: {overall['valid_wave_count']}/{overall['wave_count']}; " + detail)
    return result.exit_code


if __name__ == "__main__":
    sys.exit(main())
