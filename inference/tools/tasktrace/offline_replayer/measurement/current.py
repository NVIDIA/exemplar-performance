# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile a command inside the process's current cgroup-v2 container scope."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .container import STRACE_MODES, ProfileConfig, ProfileError, parse_cpuset_cpu_ids

PROC_SELF_CGROUP = Path("/proc/self/cgroup")
PROC_SELF_MOUNTINFO = Path("/proc/self/mountinfo")
CGROUP_REQUIRED_FILES = ("cpu.stat", "cgroup.procs")
KUBERNETES_IDENTITY_ENV = (
    "HOSTNAME",
    "KUBERNETES_SERVICE_HOST",
    "KUBERNETES_SERVICE_PORT",
    "POD_NAME",
    "POD_NAMESPACE",
    "POD_UID",
    "CONTAINER_NAME",
    "NODE_NAME",
)


@dataclass(frozen=True)
class CgroupV2Location:
    mount_point: Path
    mount_root: str
    membership_path: str
    path: Path


@dataclass(frozen=True)
class AllocationState:
    cpu_max: str | None
    cpu_max_hierarchy: tuple[dict[str, Any], ...]
    quota_cores: float | None
    cpuset: str | None
    cpuset_ids: tuple[int, ...]
    affinity_ids: tuple[int, ...]


def _decode_mountinfo_path(value: str) -> str:
    replacements = {
        "\\040": " ",
        "\\011": "\t",
        "\\012": "\n",
        "\\134": "\\",
    }
    for escaped, decoded in replacements.items():
        value = value.replace(escaped, decoded)
    return value


def _unified_membership(path: Path = PROC_SELF_CGROUP) -> str:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ProfileError(f"unable to read cgroup membership from {path}: {exc}") from exc
    for line in lines:
        if line.startswith("0::"):
            membership = line.split("::", 1)[1]
            return membership if membership.startswith("/") else f"/{membership}"
    raise ProfileError(f"cgroup v2 membership not found in {path}")


def _cgroup2_mounts(path: Path = PROC_SELF_MOUNTINFO) -> list[tuple[str, Path]]:
    try:
        lines = path.read_text().splitlines()
    except OSError as exc:
        raise ProfileError(f"unable to read mount information from {path}: {exc}") from exc
    mounts: list[tuple[str, Path]] = []
    for line in lines:
        if " - " not in line:
            continue
        before, after = line.split(" - ", 1)
        after_fields = after.split()
        before_fields = before.split()
        if not after_fields or after_fields[0] != "cgroup2" or len(before_fields) < 5:
            continue
        mount_root = _decode_mountinfo_path(before_fields[3])
        mount_point = Path(_decode_mountinfo_path(before_fields[4]))
        mounts.append((mount_root, mount_point))
    if not mounts:
        raise ProfileError(f"cgroup v2 mount not found in {path}")
    return mounts


def _mount_candidates(mount_root: str, mount_point: Path, membership: str) -> list[Path]:
    root = PurePosixPath(mount_root)
    member = PurePosixPath(membership)
    relative_paths: list[PurePosixPath] = []
    try:
        relative_paths.append(member.relative_to(root))
    except ValueError:
        pass
    relative_paths.append(PurePosixPath(membership.lstrip("/")))

    candidates: list[Path] = []
    for relative in relative_paths:
        candidate = mount_point / Path(str(relative))
        if candidate not in candidates:
            candidates.append(candidate)
    return candidates


def resolve_current_cgroup(
    *,
    proc_self_cgroup: Path = PROC_SELF_CGROUP,
    proc_self_mountinfo: Path = PROC_SELF_MOUNTINFO,
) -> CgroupV2Location:
    membership = _unified_membership(proc_self_cgroup)
    attempted: list[Path] = []
    for mount_root, mount_point in _cgroup2_mounts(proc_self_mountinfo):
        for candidate in _mount_candidates(mount_root, mount_point, membership):
            attempted.append(candidate)
            if all((candidate / filename).is_file() for filename in CGROUP_REQUIRED_FILES):
                return CgroupV2Location(
                    mount_point=mount_point,
                    mount_root=mount_root,
                    membership_path=membership,
                    path=candidate,
                )
    attempted_text = ", ".join(str(path) for path in attempted)
    raise ProfileError(
        "current-container profiling requires readable cgroup v2 cpu.stat and "
        "cgroup.procs; "
        f"checked: {attempted_text}"
    )


def read_cpu_stat(path: Path) -> tuple[str, dict[str, int]]:
    try:
        raw = path.read_text()
    except OSError as exc:
        raise ProfileError(f"unable to read cgroup CPU statistics from {path}: {exc}") from exc
    stats: dict[str, int] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) != 2:
            continue
        try:
            stats[fields[0]] = int(fields[1])
        except ValueError as exc:
            raise ProfileError(f"invalid cgroup CPU statistic in {path}: {line!r}") from exc
    if "usage_usec" not in stats:
        raise ProfileError(f"cgroup CPU statistics have no usage_usec counter: {path}")
    return raw, stats


def _read_timed_cpu_stat(
    path: Path,
) -> tuple[str, dict[str, int], int, int]:
    read_started_ns = time.monotonic_ns()
    raw, stats = read_cpu_stat(path)
    read_finished_ns = time.monotonic_ns()
    if read_finished_ns < read_started_ns:
        raise ProfileError("monotonic clock moved backward while reading cgroup CPU statistics")
    midpoint_ns = read_started_ns + (read_finished_ns - read_started_ns) // 2
    return raw, stats, midpoint_ns, read_finished_ns - read_started_ns


def parse_cpu_max(value: str | None) -> float | None:
    if value is None:
        return None
    fields = value.split()
    if len(fields) != 2:
        raise ProfileError(f"invalid cgroup cpu.max value: {value!r}")
    quota_raw, period_raw = fields
    try:
        period = int(period_raw)
    except ValueError as exc:
        raise ProfileError(f"invalid cgroup cpu.max period: {value!r}") from exc
    if period <= 0:
        raise ProfileError(f"invalid cgroup cpu.max period: {value!r}")
    if quota_raw == "max":
        return None
    try:
        quota = int(quota_raw)
    except ValueError as exc:
        raise ProfileError(f"invalid cgroup cpu.max quota: {value!r}") from exc
    if quota <= 0:
        raise ProfileError(f"invalid cgroup cpu.max quota: {value!r}")
    return quota / period


def _read_optional(path: Path) -> str | None:
    try:
        value = path.read_text().strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ProfileError(f"unable to read cgroup allocation metadata from {path}: {exc}") from exc
    return value or None


def _read_inherited(location: CgroupV2Location, filename: str) -> str | None:
    current = location.path
    while current == location.mount_point or current.is_relative_to(location.mount_point):
        value = _read_optional(current / filename)
        if value is not None:
            return value
        if current == location.mount_point:
            break
        current = current.parent
    return None


def _cpu_max_hierarchy(location: CgroupV2Location) -> tuple[dict[str, Any], ...]:
    values: list[dict[str, Any]] = []
    current = location.path
    while current == location.mount_point or current.is_relative_to(location.mount_point):
        raw = _read_optional(current / "cpu.max")
        if raw is not None:
            values.append(
                {
                    "path": str(current / "cpu.max"),
                    "value": raw,
                    "quota_cores": parse_cpu_max(raw),
                }
            )
        if current == location.mount_point:
            break
        current = current.parent
    return tuple(values)


def read_allocation_state(location: CgroupV2Location) -> AllocationState:
    cpu_max_values = _cpu_max_hierarchy(location)
    cpu_max = cpu_max_values[0]["value"] if cpu_max_values else None
    finite_quotas = [float(item["quota_cores"]) for item in cpu_max_values if item["quota_cores"] is not None]
    cpuset = _read_inherited(location, "cpuset.cpus.effective")
    cpuset_ids = parse_cpuset_cpu_ids(cpuset) if cpuset is not None else ()
    try:
        affinity_ids = tuple(sorted(os.sched_getaffinity(0)))
    except (AttributeError, OSError) as exc:
        raise ProfileError(f"unable to read current process CPU affinity: {exc}") from exc
    if not affinity_ids:
        raise ProfileError("current process CPU affinity is empty")
    return AllocationState(
        cpu_max=cpu_max,
        cpu_max_hierarchy=cpu_max_values,
        quota_cores=min(finite_quotas) if finite_quotas else None,
        cpuset=cpuset,
        cpuset_ids=cpuset_ids,
        affinity_ids=affinity_ids,
    )


def _capacity(allocation: AllocationState, override: float | None) -> tuple[float | None, str | None, tuple[int, ...]]:
    if allocation.cpuset_ids and allocation.affinity_ids:
        effective_ids = tuple(sorted(set(allocation.cpuset_ids).intersection(allocation.affinity_ids)))
        cpu_set_source = "cpuset+affinity"
    elif allocation.cpuset_ids:
        effective_ids = allocation.cpuset_ids
        cpu_set_source = "cpuset"
    elif allocation.affinity_ids:
        effective_ids = allocation.affinity_ids
        cpu_set_source = "affinity"
    else:
        effective_ids = ()
        cpu_set_source = None
    if override is not None:
        return override, "user-supplied", effective_ids
    effective_count = float(len(effective_ids)) if effective_ids else None
    if allocation.quota_cores is not None and effective_count is not None:
        return (
            min(allocation.quota_cores, effective_count),
            f"cgroup-v2-cpu.max+{cpu_set_source}",
            effective_ids,
        )
    if allocation.quota_cores is not None:
        return allocation.quota_cores, "cgroup-v2-cpu.max", effective_ids
    if effective_count is not None:
        return effective_count, cpu_set_source, effective_ids
    return None, None, effective_ids


def _read_cgroup_pids(path: Path, *, scope: str) -> set[int]:
    try:
        values = path.read_text().split()
    except OSError as exc:
        raise ProfileError(f"unable to inspect processes in {scope} from {path}: {exc}") from exc
    pids: set[int] = set()
    for value in values:
        try:
            pid = int(value)
        except ValueError as exc:
            raise ProfileError(f"invalid process ID in {path}: {value!r}") from exc
        if pid <= 0:
            raise ProfileError(f"invalid process ID in {path}: {value!r}")
        pids.add(pid)
    return pids


def _cgroup_pids(path: Path) -> list[int]:
    root_process_file = path / "cgroup.procs"
    pids = _read_cgroup_pids(root_process_file, scope="current cgroup")

    def raise_walk_error(error: OSError) -> None:
        raise error

    try:
        process_files = sorted(
            Path(directory) / "cgroup.procs"
            for directory, _subdirectories, filenames in os.walk(path, onerror=raise_walk_error)
            if "cgroup.procs" in filenames and Path(directory) / "cgroup.procs" != root_process_file
        )
    except OSError as exc:
        raise ProfileError(f"unable to enumerate descendant cgroups under {path}: {exc}") from exc
    for process_file in process_files:
        pids.update(_read_cgroup_pids(process_file, scope="descendant cgroup"))
    return sorted(pids)


def _parent_pid(pid: int) -> int | None:
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    closing = stat.rfind(")")
    if closing < 0:
        return None
    fields = stat[closing + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        return int(fields[1])
    except ValueError:
        return None


def _ancestor_pids(pid: int) -> set[int]:
    ancestors: set[int] = set()
    current = pid
    while current > 0 and current not in ancestors:
        ancestors.add(current)
        parent = _parent_pid(current)
        if parent is None:
            break
        current = parent
    return ancestors


def _process_details(pids: list[int]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for pid in pids:
        try:
            name = Path(f"/proc/{pid}/comm").read_text().strip()
        except OSError:
            name = None
        details.append({"pid": pid, "name": name})
    return details


def _environment_identity(location: CgroupV2Location) -> dict[str, Any]:
    kubernetes = {key: os.environ[key] for key in KUBERNETES_IDENTITY_ENV if os.environ.get(key)}
    os_release: dict[str, str] = {}
    try:
        for line in Path("/etc/os-release").read_text().splitlines():
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key in {"ID", "VERSION_ID", "PRETTY_NAME"}:
                os_release[key.lower()] = value.strip().strip('"')
    except OSError:
        pass
    return {
        "runtime": "current",
        "hostname": socket.gethostname(),
        "python_executable": sys.executable,
        "python_version": sys.version.split()[0],
        "cgroup_v2": {
            "mount_point": str(location.mount_point),
            "mount_root": location.mount_root,
            "membership_path": location.membership_path,
            "resolved_path": str(location.path),
            "private_namespace_root": location.membership_path == "/",
        },
        "kubernetes": kubernetes or None,
        "os_release": os_release or None,
    }


def _command_environment(assignments: tuple[str, ...], unset_names: tuple[str, ...]) -> dict[str, str]:
    environment = os.environ.copy()
    for name in unset_names:
        if not name or "=" in name:
            raise ProfileError(f"invalid environment variable name to unset: {name!r}")
        environment.pop(name, None)
    for assignment in assignments:
        if "=" not in assignment:
            raise ProfileError(f"environment assignment must use KEY=VALUE: {assignment!r}")
        key, value = assignment.split("=", 1)
        if not key:
            raise ProfileError(f"environment assignment has an empty key: {assignment!r}")
        environment[key] = value
    return environment


def _run_prepare(config: ProfileConfig, environment: dict[str, str]) -> None:
    if config.prepare_command is None:
        return
    completed = subprocess.run(
        [config.shell, "-c", config.prepare_command],
        cwd=config.workdir,
        env=environment,
        text=True,
        errors="replace",
        capture_output=True,
        check=False,
    )
    (config.output_dir / "prepare.stdout.txt").write_text(completed.stdout)
    (config.output_dir / "prepare.stderr.txt").write_text(completed.stderr)
    (config.output_dir / "prepare_exit_code").write_text(f"{completed.returncode}\n")
    if completed.returncode != 0:
        raise ProfileError(
            "current-container prepare command failed with exit code "
            f"{completed.returncode}: {completed.stderr.strip()}"
        )


def _execution_command(config: ProfileConfig, environment: dict[str, str]) -> list[str]:
    if config.strace_mode == "none":
        return [config.shell, "-c", config.command]
    strace = shutil.which("strace", path=environment.get("PATH"))
    if strace is None:
        raise ProfileError("strace is not installed in the current container")
    trace_filter = "execve" if config.strace_mode == "execve" else "%process"
    return [
        strace,
        "-f",
        "-ttt",
        "-s",
        str(config.string_limit),
        "-e",
        f"trace={trace_filter}",
        "-o",
        str(config.output_dir / "process.strace.log"),
        config.shell,
        "-c",
        config.command,
    ]


def validate_current_profile_config(config: ProfileConfig, environment: Mapping[str, str] | None = None) -> None:
    """Reject requests the read-only current-container backend cannot honor."""
    if config.runtime != "current":
        raise ProfileError(f"current-container backend received runtime {config.runtime!r}")
    if config.image is not None or config.container is not None:
        raise ProfileError("current-container runtime does not accept --image or --container")
    if config.mounts:
        raise ProfileError("current-container runtime does not accept additional mounts")
    if config.cpuset_cpus is not None:
        raise ProfileError("current-container runtime cannot enforce --cpuset-cpus; configure the sandbox allocation")
    if config.cpu_limit is not None:
        raise ProfileError("current-container runtime cannot enforce --cpu-limit; configure the sandbox allocation")
    if config.user is not None:
        raise ProfileError("current-container runtime cannot change the executing user")
    if config.scheduler_concurrency:
        raise ProfileError("--scheduler-concurrency is not supported by current-container runtime")
    if config.scheduler_bcc_buffer_pages is not None:
        raise ProfileError("--scheduler-bcc-buffer-pages is not supported by current-container runtime")
    if config.proc_stat_comparison:
        raise ProfileError("--proc-stat-comparison is not supported by current-container runtime")
    if config.terminate_descendants_at_finish:
        raise ProfileError("descendant termination is not supported by current-container runtime")
    if config.network is not None:
        raise ProfileError("--network is not supported by current-container runtime")
    if config.strace_mode not in STRACE_MODES:
        raise ProfileError(f"unsupported strace mode: {config.strace_mode!r}")
    if config.string_limit <= 0:
        raise ProfileError("strace string limit must be positive")
    if config.allocated_cpus is not None and config.allocated_cpus <= 0:
        raise ProfileError("allocated CPU count must be positive")
    if config.workdir is None:
        raise ProfileError("current-container runtime requires --workdir")
    workdir = Path(config.workdir)
    if not workdir.is_dir():
        raise ProfileError(f"current-container workdir does not exist: {workdir}")
    search_path = environment.get("PATH") if environment is not None else None
    if shutil.which(config.shell, path=search_path) is None:
        raise ProfileError(f"current-container shell is not executable: {config.shell}")


def _cpu_delta(before: dict[str, int], after: dict[str, int]) -> dict[str, int]:
    delta = {key: after.get(key, 0) - before.get(key, 0) for key in sorted(set(before) | set(after))}
    decreased = [key for key, value in delta.items() if value < 0]
    if decreased:
        raise ProfileError("current cgroup CPU counters decreased during the measured window: " + ", ".join(decreased))
    return delta


def _task_environment_summary(config: ProfileConfig) -> dict[str, list[str]]:
    return {
        "unset_variables": list(config.unset_env),
        "overridden_variables": [assignment.split("=", 1)[0] for assignment in config.env],
    }


def run_current_profile(config: ProfileConfig) -> dict[str, Any]:
    environment = _command_environment(config.env, config.unset_env)
    validate_current_profile_config(config, environment)
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ProfileError(f"profile output directory must be new or empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    _run_prepare(config, environment)

    location_before = resolve_current_cgroup()
    allocation_before = read_allocation_state(location_before)
    identity = _environment_identity(location_before)
    pids_before = _cgroup_pids(location_before.path)
    controller_ancestors_before = _ancestor_pids(os.getpid())
    unrelated_before = sorted(set(pids_before) - controller_ancestors_before)

    raw_before, stats_before, start_ns, before_read_duration_ns = _read_timed_cpu_stat(
        location_before.path / "cpu.stat"
    )
    completed = subprocess.run(
        _execution_command(config, environment),
        cwd=config.workdir,
        env=environment,
        text=True,
        errors="replace",
        capture_output=True,
        check=False,
    )
    raw_after, stats_after, end_ns, after_read_duration_ns = _read_timed_cpu_stat(location_before.path / "cpu.stat")
    if end_ns <= start_ns:
        raise ProfileError("current-container measured window has non-positive duration")

    location_after = resolve_current_cgroup()
    pids_after = _cgroup_pids(location_after.path)
    controller_ancestors_after = _ancestor_pids(os.getpid())
    unrelated_after = sorted(set(pids_after) - controller_ancestors_after)
    allocation_after = read_allocation_state(location_after)

    (config.output_dir / "cpu.stat.before").write_text(raw_before)
    (config.output_dir / "cpu.stat.after").write_text(raw_after)
    (config.output_dir / "command.stdout.txt").write_text(completed.stdout)
    (config.output_dir / "command.stderr.txt").write_text(completed.stderr)
    (config.output_dir / "start_ns").write_text(f"{start_ns}\n")
    (config.output_dir / "end_ns").write_text(f"{end_ns}\n")
    (config.output_dir / "exit_code").write_text(f"{completed.returncode}\n")
    (config.output_dir / "strace_mode").write_text(f"{config.strace_mode}\n")

    delta = _cpu_delta(stats_before, stats_after)
    wall_sec = (end_ns - start_ns) / 1_000_000_000
    usage_sec = delta["usage_usec"] / 1_000_000
    average_cores = usage_sec / wall_sec if wall_sec > 0 else None
    capacity, capacity_source, effective_cpu_ids = _capacity(allocation_before, config.allocated_cpus)
    utilization = (
        average_cores / capacity if average_cores is not None and capacity is not None and capacity > 0 else None
    )
    location_stable = location_before == location_after
    allocation_stable = allocation_before == allocation_after
    valid = (
        config.current_container_exclusive
        and location_stable
        and allocation_stable
        and not unrelated_before
        and not unrelated_after
    )
    warnings = [
        "Current-container cgroup accounting includes every process sharing this "
        "container. Comparable results require a fresh, single-task sandbox."
    ]
    if unrelated_before:
        warnings.append("Potential unrelated processes existed before replay.")
    if unrelated_after:
        warnings.append("Processes outside the controller ancestry remained after replay.")
    if not location_stable:
        warnings.append("The process changed cgroup scope during replay.")
    if not allocation_stable:
        warnings.append("The cgroup CPU allocation changed during replay.")
    if config.strace_mode != "none":
        warnings.append("This run includes strace overhead.")
    if not config.current_container_exclusive:
        warnings.append(
            "Container exclusivity was not asserted; CPU data is diagnostic rather than " "approved for comparison."
        )

    report = {
        "schema_version": "container-command-profile.v1",
        "runtime": "current",
        "scope": "current-container",
        "image": None,
        "image_identity": None,
        "container": None,
        "environment_identity": identity,
        "task_environment": _task_environment_summary(config),
        "measurement_validity": {
            "valid_for_average_cpu_reporting": valid,
            "requires_exclusive_single_task_container": True,
            "exclusive_single_task_container_asserted": config.current_container_exclusive,
            "cgroup_path_stable": location_stable,
            "allocation_stable": allocation_stable,
            "potential_unrelated_processes_before": _process_details(unrelated_before),
            "processes_outside_controller_ancestry_after": _process_details(unrelated_after),
        },
        "command": config.command,
        "prepare_command": config.prepare_command,
        "shell": config.shell,
        "strace": {
            "mode": config.strace_mode,
            "log_path": (
                str(config.output_dir / "process.strace.log")
                if (config.output_dir / "process.strace.log").exists()
                else None
            ),
            "string_limit": config.string_limit,
        },
        "command_exit_code": completed.returncode,
        "runtime_returncode": completed.returncode,
        "docker_returncode": None,
        "cpu": {
            "method": "current-container-cgroup-v2-cpu.stat",
            "scope_path": str(location_before.path),
            "wall_sec": round(wall_sec, 9),
            "usage_sec": round(usage_sec, 9),
            "user_sec": round(delta.get("user_usec", 0) / 1_000_000, 9),
            "system_sec": round(delta.get("system_usec", 0) / 1_000_000, 9),
            "average_logical_cores_used": (round(average_cores, 9) if average_cores is not None else None),
            "allocated_logical_cpus": capacity,
            "allocated_cpu_source": capacity_source,
            "effective_logical_cpu_ids": list(effective_cpu_ids),
            "effective_logical_cpu_count": (len(effective_cpu_ids) if effective_cpu_ids else None),
            "cgroup_cpuset": allocation_before.cpuset,
            "cgroup_cpuset_logical_cpu_ids": list(allocation_before.cpuset_ids),
            "process_affinity_logical_cpu_ids": list(allocation_before.affinity_ids),
            "enforced_logical_cpu_limit": allocation_before.quota_cores,
            "enforced_cpu_limit_source": ("cgroup-v2-cpu.max" if allocation_before.quota_cores is not None else None),
            "cpu_max": allocation_before.cpu_max,
            "cpu_max_hierarchy": list(allocation_before.cpu_max_hierarchy),
            "utilization_fraction_of_allocation": (round(utilization, 9) if utilization is not None else None),
            "utilization_percent_of_allocation": (round(100 * utilization, 6) if utilization is not None else None),
            "throttling": {
                "periods": delta.get("nr_periods", 0),
                "throttled_periods": delta.get("nr_throttled", 0),
                "throttled_sec": round(delta.get("throttled_usec", 0) / 1_000_000, 9),
                "burst_count": delta.get("nr_bursts", 0),
                "burst_sec": round(delta.get("burst_usec", 0) / 1_000_000, 9),
            },
            "raw_cpu_stat_delta": delta,
            "snapshot_read_duration_ns": {
                "before": before_read_duration_ns,
                "after": after_read_duration_ns,
            },
        },
        "warning": " ".join(warnings),
    }
    (config.output_dir / "current_environment.json").write_text(
        json.dumps(
            {
                "environment_identity": identity,
                "task_environment": _task_environment_summary(config),
                "allocation_before": {
                    "cpu_max": allocation_before.cpu_max,
                    "cpu_max_hierarchy": list(allocation_before.cpu_max_hierarchy),
                    "quota_cores": allocation_before.quota_cores,
                    "cpuset": allocation_before.cpuset,
                    "cpuset_ids": list(allocation_before.cpuset_ids),
                    "affinity_ids": list(allocation_before.affinity_ids),
                },
                "allocation_after": {
                    "cpu_max": allocation_after.cpu_max,
                    "cpu_max_hierarchy": list(allocation_after.cpu_max_hierarchy),
                    "quota_cores": allocation_after.quota_cores,
                    "cpuset": allocation_after.cpuset,
                    "cpuset_ids": list(allocation_after.cpuset_ids),
                    "affinity_ids": list(allocation_after.affinity_ids),
                },
                "pids_before": _process_details(pids_before),
                "pids_after": _process_details(pids_after),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    (config.output_dir / "profile_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return report
