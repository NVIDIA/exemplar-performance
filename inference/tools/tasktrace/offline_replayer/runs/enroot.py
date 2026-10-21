# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Wall-only result projection for ordered Enroot runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..measurement.enroot_network import validate_evidence
from ..measurement.network_policy import NetworkPolicy
from .spec import RunDefaultsSpec, RunLaunchSpec, RunTaskSpec

ENROOT_TASK_CLOCK = "linux-proc-uptime"
# /proc/uptime is normally exposed at centisecond precision.  Leave enough
# room for quantization at both task boundaries when comparing clock domains.
OUTER_WALL_TOLERANCE_SEC = 0.05
_RUNTIME_STATE_KEYS = (
    "ENROOT_CACHE_PATH",
    "ENROOT_DATA_PATH",
    "ENROOT_RUNTIME_PATH",
    "ENROOT_TEMP_PATH",
)


@dataclass(frozen=True)
class EnrootProjection:
    duration_sec: float | None
    valid: bool
    reason: str
    image_id: str
    execution_plan_digest: str
    cleanup_valid: bool = False
    clock_reference: Mapping[str, Any] | None = None
    window_start_ns: int | None = None
    window_end_ns: int | None = None
    cpu_usage_sec: None = None
    average_logical_cores: None = None


def _content_tree_sha256(root: Path, *, exclude_credentials: bool = False) -> str | None:
    """Hash a runtime/config tree without publishing file contents."""
    if not root.exists():
        return None
    digest = hashlib.sha256()
    paths = [root] if root.is_file() else sorted(root.rglob("*"))
    for path in paths:
        relative = path.name if root.is_file() else path.relative_to(root).as_posix()
        if exclude_credentials and Path(relative).name == ".credentials":
            continue
        if path.is_symlink():
            payload = os.readlink(path).encode()
            kind = b"symlink"
        elif path.is_file():
            payload = path.read_bytes()
            kind = b"file"
        else:
            continue
        digest.update(relative.encode())
        digest.update(b"\0" + kind + b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def _mount_filesystem_type(path: Path) -> str | None:
    """Resolve the filesystem type from Linux mountinfo using the longest mount."""
    try:
        lines = Path("/proc/self/mountinfo").read_text().splitlines()
    except OSError:
        return None
    resolved = path.resolve()
    best: tuple[int, str] | None = None
    for line in lines:
        fields = line.split()
        try:
            separator = fields.index("-")
        except ValueError:
            continue
        if separator + 1 >= len(fields) or len(fields) < 5:
            continue
        mount_point = Path(
            fields[4].replace("\\040", " ").replace("\\011", "\t").replace("\\012", "\n").replace("\\134", "\\")
        )
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        candidate = (len(mount_point.parts), fields[separator + 1])
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1] if best is not None else None


def runtime_provenance(
    *,
    controller_cpus: str,
    launch_cpu_sets: Sequence[str],
    enroot: str = "enroot",
) -> dict[str, Any]:
    """Capture compact, secret-safe Enroot and state-filesystem identity."""
    executable = shutil.which(enroot)
    if executable is None:
        raise RuntimeError(f"Enroot executable not found: {enroot}")
    taskset = shutil.which("taskset")
    if taskset is None:
        raise RuntimeError("taskset executable not found")
    required_helpers = {name: shutil.which(name) for name in ("unshare", "unsquashfs")}
    if any(path is None for path in required_helpers.values()):
        raise RuntimeError("required Enroot launch helper not found")
    environment = dict(os.environ)
    environment["LC_ALL"] = "C"

    def inspect(cpus: str, command: str) -> subprocess.CompletedProcess[str]:
        try:
            return subprocess.run(
                [taskset, "-c", cpus, executable, command],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
                env=environment,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeError(f"unable to inspect Enroot {command} on CPUs {cpus}: {exc}") from exc

    def parse_settings(output: str) -> dict[str, str]:
        return {
            key: value
            for line in output.splitlines()
            if (entry := line.strip()).startswith("ENROOT_") and "=" in entry
            for key, value in (entry.split("=", 1),)
        }

    def settings_sha256(settings: Mapping[str, str]) -> str:
        encoded = "\n".join(f"{key}={settings[key]}" for key in sorted(settings)).encode()
        return hashlib.sha256(encoded).hexdigest()

    cpu_sets = tuple(dict.fromkeys((controller_cpus, *launch_cpu_sets)))
    try:
        version_run = inspect(controller_cpus, "version")
        settings_by_cpus = {cpus: parse_settings(inspect(cpus, "info").stdout) for cpus in cpu_sets}
    except RuntimeError:
        raise
    version = version_run.stdout.strip()
    settings = settings_by_cpus[controller_cpus]
    required = {
        "ENROOT_LIBRARY_PATH",
        "ENROOT_SYSCONF_PATH",
        "ENROOT_CONFIG_PATH",
        *_RUNTIME_STATE_KEYS,
    }
    if not version or not required.issubset(settings):
        raise RuntimeError("Enroot info omitted required runtime configuration")
    for cpus, contextual in settings_by_cpus.items():
        if not required.issubset(contextual) or any(contextual[key] != settings[key] for key in required):
            raise RuntimeError(f"Enroot runtime paths changed with CPU affinity context {cpus}")

    executable_path = Path(executable).resolve()
    library_path = Path(settings["ENROOT_LIBRARY_PATH"]).resolve()
    runtime_digest = hashlib.sha256()
    runtime_paths = sorted(
        {
            *executable_path.parent.glob("enroot*"),
            Path(taskset).resolve(),
            *(Path(path).resolve() for path in required_helpers.values() if path),
            library_path,
        }
    )
    for path in runtime_paths:
        content_digest = _content_tree_sha256(path)
        if content_digest is None:
            raise RuntimeError(f"Enroot runtime path is missing: {path}")
        runtime_digest.update(str(path).encode())
        runtime_digest.update(b"\0" + content_digest.encode() + b"\0")

    state_paths: dict[str, Any] = {}
    for key in _RUNTIME_STATE_KEYS:
        path = Path(settings[key]).resolve()
        try:
            stat = path.stat()
            statvfs = os.statvfs(path)
        except OSError as exc:
            raise RuntimeError(f"unable to inspect {key}: {exc}") from exc
        state_paths[key] = {
            "path": str(path),
            "filesystem_type": _mount_filesystem_type(path),
            "device_major": os.major(stat.st_dev),
            "device_minor": os.minor(stat.st_dev),
            "filesystem_id": str(getattr(statvfs, "f_fsid", "")),
        }

    provenance: dict[str, Any] = {
        "schema_version": "tasktrace-enroot-runtime.v1",
        "version": version,
        "executable": str(executable_path),
        "runtime_content_sha256": runtime_digest.hexdigest(),
        "effective_config_sha256": {
            "controller": {
                "cpus": controller_cpus,
                "sha256": settings_sha256(settings_by_cpus[controller_cpus]),
            },
            "launches": {cpus: settings_sha256(settings_by_cpus[cpus]) for cpus in dict.fromkeys(launch_cpu_sets)},
        },
        "system_config_sha256": _content_tree_sha256(Path(settings["ENROOT_SYSCONF_PATH"]).resolve()),
        "user_config_sha256": _content_tree_sha256(
            Path(settings["ENROOT_CONFIG_PATH"]).resolve(),
            exclude_credentials=True,
        ),
        "state_paths": state_paths,
    }
    encoded = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode()
    provenance["comparability_sha256"] = hashlib.sha256(encoded).hexdigest()
    return provenance


def image_id(identity: Any) -> str | None:
    if not isinstance(identity, Mapping) or identity.get("runtime") != "enroot":
        return None
    digest = identity.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        return None
    if any(character not in "0123456789abcdef" for character in digest):
        return None
    return f"sha256:{digest}"


def _nonnegative_integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else None


def _plan_digest(
    task: RunTaskSpec,
    defaults: RunDefaultsSpec,
    launch: RunLaunchSpec,
    resolved_image_id: str,
    controller_source_sha256: str,
    runtime_provenance_sha256: str,
) -> str:
    payload = {
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "image_id": resolved_image_id,
        "controller_source_sha256": controller_source_sha256,
        "enroot_runtime_provenance_sha256": runtime_provenance_sha256,
        "cpus": launch.cpus,
        "pacing": defaults.pacing,
        "network_policy": defaults.network_policy,
        "allowed_hosts": list(defaults.allowed_hosts),
        "controller_runtime": defaults.controller_runtime,
        "controller_cpus": defaults.controller_cpus,
        "effective_action_timeout_sec": (
            task.action_timeout_sec if task.action_timeout_sec is not None else defaults.action_timeout_sec
        ),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def project(
    report: Mapping[str, Any],
    *,
    task: RunTaskSpec,
    preflight_image_id: str,
    controller_source_sha256: str | None,
    runtime_provenance: Mapping[str, Any] | None,
    defaults: RunDefaultsSpec,
    launch: RunLaunchSpec,
) -> EnrootProjection:
    """Validate one Enroot report without claiming cgroup CPU evidence."""
    reasons: list[str] = []
    if not isinstance(controller_source_sha256, str) or len(controller_source_sha256) != 64:
        reasons.append("controller source identity is missing or invalid")
    runtime_provenance_sha256 = (
        runtime_provenance.get("comparability_sha256") if isinstance(runtime_provenance, Mapping) else None
    )
    if not isinstance(runtime_provenance_sha256, str) or len(runtime_provenance_sha256) != 64:
        reasons.append("Enroot runtime provenance is missing or invalid")
    cpu = report.get("container_cpu")
    if not isinstance(cpu, Mapping):
        cpu = {}
        reasons.append("container_cpu evidence is missing")
    wall = cpu.get("wall_sec")
    duration = (
        float(wall)
        if isinstance(wall, (int, float))
        and not isinstance(wall, bool)
        and math.isfinite(float(wall))
        and float(wall) > 0
        else None
    )
    if duration is None:
        reasons.append("container_cpu.wall_sec is not a finite positive number")
    if cpu.get("method") != "wall-clock-only-no-cgroup":
        reasons.append("Enroot CPU method is not wall-clock-only-no-cgroup")
    if cpu.get("wall_clock_source") != ENROOT_TASK_CLOCK:
        reasons.append("Enroot wall duration is not from one monotonic clock")
    if cpu.get("allocated_cpu_source") != "taskset-requested-affinity":
        reasons.append("Enroot requested task affinity evidence is missing")
    if cpu.get("allocated_logical_cpus") != float(len(launch.cpu_ids)):
        reasons.append("Enroot allocation width differs from the launch CPU set")
    if cpu.get("requested_cpu_affinity") != launch.cpus:
        reasons.append("Enroot requested task affinity differs from the launch CPU set")
    initial_effective_ids = cpu.get("initial_effective_logical_cpu_ids")
    if initial_effective_ids != list(launch.cpu_ids):
        reasons.append("Enroot initial effective task affinity differs from the launch CPU set")
    raw_start = _nonnegative_integer(cpu.get("window_start_ns"))
    raw_end = _nonnegative_integer(cpu.get("window_end_ns"))
    clock = cpu.get("clock_reference")
    resolution_ns: int | None = None
    if not isinstance(clock, Mapping):
        reasons.append("Enroot task clock reference is missing")
    else:
        resolution_ns = _nonnegative_integer(clock.get("resolution_ns"))
        if (
            clock.get("clock_name") != "CLOCK_BOOTTIME"
            or clock.get("source") != ENROOT_TASK_CLOCK
            or resolution_ns is None
            or resolution_ns <= 0
            or not isinstance(clock.get("boot_id"), str)
            or not clock.get("boot_id")
            or not isinstance(clock.get("time_namespace"), str)
            or not clock.get("time_namespace")
        ):
            reasons.append("Enroot task clock reference is invalid")
    if raw_start is None or raw_end is None or raw_end <= raw_start:
        reasons.append("Enroot task window is missing or non-increasing")
    elif (
        duration is not None
        and resolution_ns is not None
        and not math.isclose(
            (raw_end - raw_start) / 1_000_000_000,
            duration,
            rel_tol=0.0,
            abs_tol=2 * resolution_ns / 1_000_000_000,
        )
    ):
        reasons.append("Enroot wall duration disagrees with raw task timestamps")

    if report.get("image") != task.image:
        reasons.append("profile image reference differs from the declared task image")
    if report.get("pacing") != defaults.pacing:
        reasons.append("profile pacing differs from defaults.pacing")
    if report.get("controller_runtime") != defaults.controller_runtime:
        reasons.append("profile controller runtime differs from defaults.controller_runtime")
    provenance = report.get("profile_provenance")
    if not isinstance(provenance, Mapping) or provenance.get("profile_sets") != []:
        reasons.append("profile sets differ from defaults.profile_sets")
    policy = report.get("network_policy")
    expected_policy = NetworkPolicy(mode=defaults.network_policy, allowed_hosts=defaults.allowed_hosts)
    if not isinstance(policy, Mapping) or policy.get("comparability_sha256") != expected_policy.comparability_digest:
        reasons.append("network policy differs from defaults.network_policy")
    elif defaults.network_policy == "no-network":
        enforcement = policy.get("enforcement")
        publication = enforcement.get("publication") if isinstance(enforcement, Mapping) else None
        try:
            if not isinstance(enforcement, Mapping):
                raise RuntimeError("evidence is missing")
            validate_evidence(
                enforcement,
                parent_netns_inode=Path("/proc/self/ns/net").stat().st_ino,
                controller_euid=os.geteuid(),
                controller_egid=os.getegid(),
            )
            if (
                not isinstance(publication, Mapping)
                or publication.get("source") != "controller-private-temporary-state"
                or publication.get("published_after_enroot_exit") is not True
            ):
                raise RuntimeError("publication evidence is invalid")
        except (OSError, RuntimeError):
            reasons.append("Enroot no-network enforcement evidence is invalid")

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
    if report.get("status") != "valid":
        reasons.append("pipeline status is not valid")
    if report.get("source_trajectory_sha256") != task.source_sha256:
        reasons.append("profile source digest differs from the sealed ATIF")

    identity = report.get("image_identity")
    path = Path(task.image).resolve()
    try:
        stat = path.stat()
    except OSError as exc:
        reasons.append(f"unable to revalidate Enroot image after replay: {exc}")
    else:
        if not isinstance(identity, Mapping):
            reasons.append("profile image identity is missing")
        elif (
            identity.get("runtime") != "enroot"
            or identity.get("path") != str(path)
            or identity.get("size_bytes") != stat.st_size
            or identity.get("mtime_ns") != stat.st_mtime_ns
        ):
            reasons.append("profile image identity differs from the declared local sqsh")

    lifecycle = report.get("enroot_lifecycle")
    cleanup_valid = False
    if not isinstance(lifecycle, Mapping):
        reasons.append("Enroot lifecycle evidence is missing")
    else:
        lifecycle_clock = lifecycle.get("clock_reference")
        process_containment = lifecycle.get("process_containment")
        cleanup = lifecycle.get("cleanup")
        phases = lifecycle.get("phases")
        if (
            not isinstance(lifecycle_clock, Mapping)
            or lifecycle_clock.get("clock_name") != "CLOCK_BOOTTIME"
            or not isinstance(clock, Mapping)
            or not isinstance(lifecycle_clock.get("boot_id"), str)
            or not lifecycle_clock.get("boot_id")
            or not isinstance(lifecycle_clock.get("time_namespace"), str)
            or not lifecycle_clock.get("time_namespace")
            or lifecycle_clock.get("boot_id") != clock.get("boot_id")
            or lifecycle_clock.get("time_namespace") != clock.get("time_namespace")
        ):
            reasons.append("Enroot task and lifecycle clocks lack an identity mapping")
        if process_containment != {
            "backend": "linux-user-pid-namespace-v1",
            "namespace_init_exit_kills_remaining_processes": True,
            "outer_exit_kills_namespace_init": True,
        }:
            reasons.append("Enroot PID-namespace containment evidence is invalid")
        if (
            not isinstance(cleanup, Mapping)
            or cleanup.get("valid") is not True
            or cleanup.get("container_absent") is not True
        ):
            reasons.append("Enroot cleanup evidence is invalid")
        else:
            cleanup_valid = True
        start_phases = (
            [phase for phase in phases if isinstance(phase, Mapping) and phase.get("phase") == "start"]
            if isinstance(phases, list)
            else []
        )
        if len(start_phases) != 1:
            reasons.append("Enroot start lifecycle boundary is missing")
        elif raw_start is not None and raw_end is not None:
            phase_start = _nonnegative_integer(start_phases[0].get("start_ns"))
            phase_end = _nonnegative_integer(start_phases[0].get("end_ns"))
            tolerance = resolution_ns or 0
            if (
                phase_start is None
                or phase_end is None
                or raw_start + tolerance < phase_start
                or raw_end > phase_end + tolerance
            ):
                reasons.append("Enroot task window falls outside the start lifecycle phase")

    return EnrootProjection(
        duration_sec=duration,
        valid=duration is not None and not reasons,
        reason="; ".join(dict.fromkeys(reasons)),
        image_id=preflight_image_id,
        execution_plan_digest=_plan_digest(
            task,
            defaults,
            launch,
            preflight_image_id,
            controller_source_sha256 or "",
            runtime_provenance_sha256 or "",
        ),
        cleanup_valid=cleanup_valid,
        clock_reference=dict(clock) if isinstance(clock, Mapping) else None,
        window_start_ns=raw_start,
        window_end_ns=raw_end,
    )


def invalidate_changed_images(
    tasks: Sequence[RunTaskSpec],
    rows: Sequence[dict[str, Any]],
) -> None:
    """Invalidate completed rows if a local image changed after preflight."""
    observed: dict[Path, str | OSError] = {}
    for task in tasks:
        path = Path(task.image).resolve()
        if path not in observed:
            try:
                digest = hashlib.sha256()
                with path.open("rb") as image:
                    for block in iter(lambda: image.read(1024 * 1024), b""):
                        digest.update(block)
                observed[path] = f"sha256:{digest.hexdigest()}"
            except OSError as exc:
                observed[path] = exc
        current = observed[path]
        for row in rows:
            if row.get("task_id") != task.task_id:
                continue
            if isinstance(current, str) and row.get("image_id") == current:
                continue
            row["duration_valid"] = False
            reason = (
                f"unable to re-read Enroot image after replay: {current}"
                if isinstance(current, OSError)
                else "Enroot image content changed after preflight"
            )
            row["reason"] = "; ".join(filter(None, (row.get("reason"), reason)))


def invalidate_duration_bounds(
    rows: Sequence[dict[str, Any]],
    *,
    outer_wall_sec: float,
) -> None:
    """Reject task durations that cannot fit inside the monotonic wave wall."""
    if not math.isfinite(outer_wall_sec) or outer_wall_sec <= 0:
        return
    upper_bound = outer_wall_sec + OUTER_WALL_TOLERANCE_SEC
    for row in rows:
        duration = row.get("duration_sec")
        if row.get("duration_valid") is not True or (
            isinstance(duration, (int, float))
            and not isinstance(duration, bool)
            and math.isfinite(float(duration))
            and 0 < float(duration) <= upper_bound
        ):
            continue
        row["duration_valid"] = False
        reason = "task duration exceeds the lifecycle-inclusive outer wave bound"
        row["reason"] = "; ".join(filter(None, (str(row.get("reason") or ""), reason)))


def wave_overall(
    rows: Sequence[Mapping[str, Any]],
    *,
    interrupted: bool,
    outer_wall_sec: float,
    expected_attempt_indexes: set[int],
    require_full_overlap: bool = False,
) -> dict[str, Any]:
    """Summarize replay-command and lifecycle boundaries independently."""
    expected = len(expected_attempt_indexes)
    outer_wall_valid = math.isfinite(outer_wall_sec) and outer_wall_sec > 0
    valid_rows = [row for row in rows if row.get("duration_valid") is True]
    indexes = [
        value
        for row in rows
        for value in (row.get("attempt_index"),)
        if isinstance(value, int) and not isinstance(value, bool)
    ]
    reasons: list[str] = []
    if interrupted:
        reasons.append("run was interrupted")
    if len(rows) != expected:
        reasons.append(f"retained {len(rows)} of {expected} declared attempts")
    if len(indexes) != len(rows) or set(indexes) != expected_attempt_indexes or len(set(indexes)) != len(indexes):
        reasons.append("retained attempt indexes are incomplete or duplicated")
    if len(valid_rows) != expected:
        reasons.append(f"only {len(valid_rows)} of {expected} attempts have valid wall-only duration evidence")
    intervals = [
        (int(row["window_start_ns"]), int(row["window_end_ns"]))
        for row in valid_rows
        if isinstance(row.get("window_start_ns"), int)
        and not isinstance(row.get("window_start_ns"), bool)
        and isinstance(row.get("window_end_ns"), int)
        and not isinstance(row.get("window_end_ns"), bool)
        and int(row["window_end_ns"]) > int(row["window_start_ns"])
    ]
    if len(intervals) != expected:
        reasons.append(f"only {len(intervals)} of {expected} attempts have usable absolute replay windows")
    clock_identities = {
        (
            clock.get("clock_name"),
            clock.get("boot_id"),
            clock.get("time_namespace"),
        )
        for row in valid_rows
        if isinstance((clock := row.get("clock_reference")), Mapping)
    }
    if len(clock_identities) != 1:
        reasons.append("replay windows do not share one monotonic clock identity")
    changes: dict[int, int] = {}
    for start_ns, end_ns in intervals:
        changes[start_ns] = changes.get(start_ns, 0) + 1
        changes[end_ns] = changes.get(end_ns, 0) - 1
    active = peak = 0
    cursor: int | None = None
    duration_by_active: dict[int, int] = {}
    for timestamp, delta in sorted(changes.items()):
        if cursor is not None and timestamp > cursor:
            duration_by_active[active] = duration_by_active.get(active, 0) + timestamp - cursor
        active += delta
        peak = max(peak, active)
        cursor = timestamp
    if active != 0:
        reasons.append("replay overlap intervals did not close")
    if require_full_overlap and peak != expected:
        reasons.append(f"replay commands did not achieve all-launch overlap: peak={peak}, expected={expected}")
    if not outer_wall_valid:
        reasons.append("lifecycle-inclusive outer wave wall time is not finite and positive")
    replay_start = min((start for start, _end in intervals), default=None)
    replay_end = max((end for _start, end in intervals), default=None)
    replay_wall = (
        (replay_end - replay_start) / 1_000_000_000
        if replay_start is not None and replay_end is not None and not reasons
        else None
    )
    replay_throughput = expected / replay_wall if replay_wall else None
    observed_lifecycle_throughput = expected / outer_wall_sec if outer_wall_valid else None
    lifecycle_throughput = observed_lifecycle_throughput if not reasons else None
    starts = [start for start, _end in intervals]
    start_spread_sec = (max(starts) - min(starts)) / 1_000_000_000 if starts else None
    all_launch_overlap_sec = duration_by_active.get(expected, 0) / 1_000_000_000
    durations = [float(row["duration_sec"]) for row in valid_rows]
    return {
        "attempt_count": len(rows),
        "declared_attempt_count": expected,
        "completed": sum(row["status"] == "completed" for row in rows),
        "failed": sum(row["status"] == "failed" for row in rows),
        "valid_duration_count": len(valid_rows),
        "summed_service_time_sec": math.fsum(durations) if durations else None,
        "interrupted": interrupted,
        "valid_for_throughput": not reasons,
        "valid_for_wall_only_throughput": not reasons,
        "throughput_invalid_reasons": reasons,
        "measurement_mode": "wall-only-replay-command",
        "outer_run_wall_sec": outer_wall_sec,
        "lifecycle_inclusive_wall_sec": outer_wall_sec if outer_wall_valid else None,
        "observed_lifecycle_inclusive_throughput_tasks_per_sec": (observed_lifecycle_throughput),
        "lifecycle_inclusive_throughput_tasks_per_sec": lifecycle_throughput,
        "lifecycle_inclusive_throughput_tasks_per_hour": (
            3600 * lifecycle_throughput if lifecycle_throughput else None
        ),
        "batch_window_start_ns": replay_start,
        "batch_window_end_ns": replay_end,
        "measured_window_count": len(intervals),
        "measured_window_span_sec": (
            (replay_end - replay_start) / 1_000_000_000 if replay_start is not None and replay_end is not None else None
        ),
        "batch_wall_sec": replay_wall,
        "replay_command_batch_wall_sec": replay_wall,
        "throughput_tasks_per_sec": replay_throughput,
        "throughput_tasks_per_hour": (3600 * replay_throughput if replay_throughput else None),
        "wall_only_throughput_tasks_per_sec": replay_throughput,
        "wall_only_throughput_tasks_per_hour": (3600 * replay_throughput if replay_throughput else None),
        "total_cgroup_cpu_sec": None,
        "aggregate_average_logical_cores": None,
        "pool_utilization_percent": None,
        "requested_concurrency": expected,
        "achieved_peak_measured_attempts": peak,
        "start_spread_sec": start_spread_sec,
        "all_launch_overlap_sec": all_launch_overlap_sec,
        "time_sec_by_measured_attempts": {
            str(count): duration_ns / 1_000_000_000 for count, duration_ns in sorted(duration_by_active.items())
        },
    }
