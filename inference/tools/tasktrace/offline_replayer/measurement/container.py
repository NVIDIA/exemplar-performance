#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run one shell command with scoped CPU and optional memory accounting.

This is a standalone diagnostic utility for replay experiments. Fresh image
mode is suitable for comparable runs; existing-container mode is useful for
inspection but its cgroup counters also include other work in that container.

The profiler uses container cgroup counters for aggregate CPU use and optional
``strace`` only to identify launched processes. It intentionally avoids
requiring perf/eBPF privileges for its normal measurement path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from ..timebase import boottime_clock_reference
from .cgroup_resources import (
    CgroupResourceSampler,
    CgroupResourceSamplingError,
)
from .enroot_network import EVIDENCE_FILENAME, validate_evidence
from .network_policy import (
    LEGACY_NETWORK_UNSET,
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyError,
    resolve_cli_network_policy,
    resolve_network_policy,
)
from .proc_stat import (
    ProcStatComparisonSampler,
    ProcStatSamplingError,
    add_reference_comparison,
)
from .process_exec import load_process_exec_trace, write_process_exec_report
from .scheduler_bcc import (
    SchedulerConcurrencyTracer,
    SchedulerTraceError,
    add_cgroup_validation,
)
from .scheduler_perf import PerfSchedulerConcurrencyTracer

ENTRYPOINT = Path(__file__).with_name("container_profile_entrypoint.sh")
STRACE_MODES = ("execve", "process", "none")
SCHEDULER_BACKENDS = ("bcc", "perf")
SCHEDULER_DESCENDANT_DRAIN_GRACE_SEC = 1.0
BCC_SCHEDULER_EVENT_BOUNDARY_TOLERANCE_SEC = 0.000002
CONTAINER_RUNTIMES = ("docker", "enroot")
PROFILE_RUNTIMES = (*CONTAINER_RUNTIMES, "current")
PROFILE_SETS = (
    "cpu-average",
    "cpu-peak",
    "memory-detail",
    "cpu-compare",
    "software-deep",
    "hardware-deep",
)
PROFILE_PREFLIGHT_SCHEMA_VERSION = "tasktrace-profile-preflight.v1"
DEFAULT_HARDWARE_PERF_EVENTS = (
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
)
DEFAULT_NETWORK_POLICY_PROXY_IMAGE = "python:3.12-slim"
ENROOT_LIFECYCLE_FILENAME = "enroot_lifecycle.json"
ENROOT_PROCESS_CONTAINMENT = {
    "backend": "linux-user-pid-namespace-v1",
    "namespace_init_exit_kills_remaining_processes": True,
    "outer_exit_kills_namespace_init": True,
}
EXECUTION_COMPARABILITY_FIELDS = (
    "runtime",
    "image",
    "image_identity",
    "command",
    "prepare_command",
    "workdir",
    "shell",
    "user",
    "cpuset_cpus",
    "allocated_cpus",
    "cpu_limit",
    "network",
    "environment",
    "normalized_mounts",
    "strace_mode",
    "scheduler_concurrency",
    "scheduler_backend",
    "proc_stat_comparison",
    "terminate_descendants_at_finish",
    "host_platform",
    "controller_source_sha256",
)
EXECUTION_LIMIT_FIELDS = (
    "memory_limit_bytes",
    "memory_swap_total_limit_bytes",
)
EXECUTION_PROVENANCE_FIELDS = (
    *EXECUTION_COMPARABILITY_FIELDS,
    *EXECUTION_LIMIT_FIELDS,
    "configured_swap_policy",
    "memory_sample_interval_sec",
    "raw_mounts",
    "comparability_sha256",
    "comparability_excludes_raw_host_mount_sources",
    "comparability_excludes_memory_limits",
)
EXECUTION_PROVENANCE_OPTIONAL_FIELDS = ("network_policy",)


class ProfileError(RuntimeError):
    """Raised when a profile run cannot be executed or interpreted safely."""


@dataclass(frozen=True)
class ProfileConfig:
    command: str
    prepare_command: str | None
    output_dir: Path
    image: str | None
    container: str | None
    cpuset_cpus: str | None
    cpu_limit: float | None
    allocated_cpus: float | None
    workdir: str | None
    user: str | None
    env: tuple[str, ...]
    mounts: tuple[str, ...]
    strace_mode: str
    string_limit: int
    shell: str
    verification_command: str | None = None
    scheduler_concurrency: bool = False
    scheduler_attach_timeout_sec: float = 20.0
    scheduler_cpu_tolerance_fraction: float = 0.02
    scheduler_cpu_tolerance_sec: float = 0.005
    scheduler_backend: str = "bcc"
    scheduler_bcc_buffer_pages: int | None = None
    scheduler_perf_binary: str = "perf"
    scheduler_perf_mmap_pages: int | None = None
    proc_stat_comparison: bool = False
    proc_stat_interval_sec: float = 0.5
    proc_stat_baseline_sec: float = 5.0
    memory_profile: bool = False
    memory_sample_interval_sec: float = 0.1
    memory_limit_bytes: int | None = None
    memory_swap_total_limit_bytes: int | None = None
    terminate_descendants_at_finish: bool = False
    # ``network`` remains the backward-compatible raw Docker option. New code
    # should use ``network_policy``; runtime-created attachment names are kept
    # separate so random per-run network IDs never enter comparability hashes.
    network: str | None = None
    network_policy: NetworkPolicy | None = None
    network_policy_proxy_image: str = DEFAULT_NETWORK_POLICY_PROXY_IMAGE
    network_policy_external_network: str = "bridge"
    network_attachment: str | None = None
    network_policy_enforcement: Mapping[str, Any] | None = None
    runtime: str = "docker"
    docker: str = "docker"
    enroot: str = "enroot"
    current_container_exclusive: bool = False
    unset_env: tuple[str, ...] = ()
    profile_sets: tuple[str, ...] = ()
    coordinated_baseline: bool = False
    render_unified_view: bool = True
    hardware_perf_binary: str = "perf"
    hardware_perf_events: tuple[str, ...] = DEFAULT_HARDWARE_PERF_EVENTS
    hardware_perf_attach_timeout_sec: float = 20.0
    hardware_perf_minimum_running_ratio: float = 0.95
    workload_identity: Mapping[str, Any] | None = None
    source_identity: Mapping[str, Any] | None = None
    expected_outcome: Mapping[str, Any] | None = None


def resolved_network_policy(config: ProfileConfig) -> NetworkPolicy | None:
    """Resolve semantic policy while preserving legacy named Docker networks.

    ``None`` means the caller deliberately used the old arbitrary Docker
    network escape hatch. It remains supported for compatibility but is not a
    portable TaskTrace network policy.
    """
    try:
        if config.network_policy is not None:
            return resolve_network_policy(
                policy=config.network_policy,
                legacy_network=(config.network if config.network is not None else LEGACY_NETWORK_UNSET),
            )
        if config.network in (None, "none"):
            return NetworkPolicy.from_legacy_network(config.network)
        return None
    except NetworkPolicyError as exc:
        raise ProfileError(str(exc)) from exc


_CONTAINER_RUNTIME_SOCKET_NAMES = frozenset(
    {
        "containerd.sock",
        "cri-dockerd.sock",
        "crio.sock",
        "docker.sock",
        "podman.sock",
    }
)
_CONTAINER_RUNTIME_SOCKET_PATHS = tuple(
    Path(path)
    for path in (
        "/run/containerd/containerd.sock",
        "/run/crio/crio.sock",
        "/run/docker.sock",
        "/run/podman/podman.sock",
        "/var/run/docker.sock",
        "/var/run/podman/podman.sock",
    )
)


def _mount_exposes_container_runtime_socket(source: str) -> bool:
    """Reject direct, aliased, rootless, and parent-directory socket mounts."""
    candidate = Path(source).expanduser().resolve(strict=False)
    if candidate.name in _CONTAINER_RUNTIME_SOCKET_NAMES:
        return True
    for socket_path in _CONTAINER_RUNTIME_SOCKET_PATHS:
        canonical_socket = socket_path.resolve(strict=False)
        if canonical_socket == candidate or canonical_socket.is_relative_to(candidate):
            return True
    return False


def _network_policy_report(config: ProfileConfig) -> dict[str, Any]:
    policy = resolved_network_policy(config)
    if policy is None:
        return {
            "schema_version": "tasktrace-network-policy-legacy-docker.v1",
            "requested_mode": "legacy-docker-network",
            "effective_mode": "runtime-specific",
            "allowed_hosts": [],
            "legacy_docker_network": config.network,
            "enforcement": None,
        }
    payload = policy.to_report_payload()
    payload["comparability_sha256"] = policy.comparability_digest
    payload["enforcement"] = (
        dict(config.network_policy_enforcement) if config.network_policy_enforcement is not None else None
    )
    return payload


def _legacy_network_projection(config: ProfileConfig) -> str | None:
    """Return a stable v1-compatible network value for reports and hashes."""
    policy = resolved_network_policy(config)
    if policy is None:
        return config.network
    if policy.effective_mode is NetworkMode.PUBLIC:
        return None
    if policy.effective_mode is NetworkMode.NO_NETWORK:
        return "none"
    return "tasktrace-allowlist"


def _network_policy_comparability(config: ProfileConfig) -> dict[str, Any] | None:
    policy = resolved_network_policy(config)
    if policy is None:
        return None
    payload = policy.to_comparability_payload()
    enforcement = config.network_policy_enforcement
    if policy.effective_mode is NetworkMode.ALLOWLIST and isinstance(enforcement, Mapping):
        # The proxy implementation, allowed ports, immutable image, and egress
        # attachment affect behavior and latency. The enforcement layer owns a
        # canonical projection that excludes all ephemeral Docker identifiers.
        comparable_enforcement = enforcement.get("comparability")
        if isinstance(comparable_enforcement, Mapping):
            payload["enforcement"] = dict(comparable_enforcement)
    return payload


def _validate_current_runtime_extensions(config: ProfileConfig, policy: NetworkPolicy | None) -> None:
    """Reject profiler options the current-container backend cannot honor.

    ``measurement.current`` validates the original current-container contract
    (no image, mounts, CPU reconfiguration, scheduler tracing, and so on).
    Keep validation for newer coordinated, memory, and network controls here so
    rebasing those features cannot turn them into silently ignored requests.
    """
    if config.profile_sets:
        raise ProfileError("profile sets are not supported by current-container runtime")
    if config.coordinated_baseline:
        raise ProfileError("--coordinated-baseline is not supported by current-container runtime")
    if config.verification_command is not None:
        raise ProfileError("--verification-command is not supported by current-container runtime")

    scheduler_options_configured = (
        config.scheduler_attach_timeout_sec != 20.0
        or config.scheduler_cpu_tolerance_fraction != 0.02
        or config.scheduler_cpu_tolerance_sec != 0.005
        or config.scheduler_backend != "bcc"
        or config.scheduler_bcc_buffer_pages is not None
        or config.scheduler_perf_binary != "perf"
        or config.scheduler_perf_mmap_pages is not None
    )
    if scheduler_options_configured:
        raise ProfileError("scheduler profiling options are not supported by " "current-container runtime")
    if config.proc_stat_interval_sec != 0.5 or config.proc_stat_baseline_sec != 5.0:
        raise ProfileError("proc-stat sampling options are not supported by " "current-container runtime")

    memory_options_configured = (
        config.memory_profile
        or config.memory_sample_interval_sec != 0.1
        or config.memory_limit_bytes is not None
        or config.memory_swap_total_limit_bytes is not None
    )
    if memory_options_configured:
        raise ProfileError("memory profiling options are not supported by " "current-container runtime")

    if config.network_policy is not None:
        raise ProfileError("network policy controls are not supported by current-container runtime")
    if policy is None or policy.effective_mode is not NetworkMode.PUBLIC:
        raise ProfileError("non-public network policy is not supported by current-container runtime")
    if config.network_attachment is not None or config.network_policy_enforcement is not None:
        raise ProfileError("Docker network-policy enforcement state is not supported by " "current-container runtime")
    if (
        config.network_policy_proxy_image != DEFAULT_NETWORK_POLICY_PROXY_IMAGE
        or config.network_policy_external_network != "bridge"
    ):
        raise ProfileError("Docker network-policy proxy options are not supported by " "current-container runtime")

    hardware_options_configured = (
        config.hardware_perf_binary != "perf"
        or config.hardware_perf_events != DEFAULT_HARDWARE_PERF_EVENTS
        or config.hardware_perf_attach_timeout_sec != 20.0
        or config.hardware_perf_minimum_running_ratio != 0.95
    )
    if hardware_options_configured:
        raise ProfileError("hardware profiler options are not supported by current-container runtime")
    if config.docker != "docker" or config.enroot != "enroot":
        raise ProfileError("Docker and Enroot executable options are not supported by " "current-container runtime")


def validate_profile_config(config: ProfileConfig) -> None:
    """Validate side-effect-free profile constraints before creating outputs."""
    policy = resolved_network_policy(config)
    if not isinstance(config.coordinated_baseline, bool):
        raise ProfileError("coordinated_baseline must be boolean")
    if config.runtime not in PROFILE_RUNTIMES:
        raise ProfileError(f"unsupported container runtime: {config.runtime!r}")
    if any(not isinstance(profile_set, str) or not profile_set.strip() for profile_set in config.profile_sets):
        raise ProfileError("profile sets must be nonempty strings")
    if len(config.profile_sets) != len(set(config.profile_sets)):
        raise ProfileError("profile sets must not contain duplicates")
    unknown_profile_sets = sorted(set(config.profile_sets) - set(PROFILE_SETS))
    if unknown_profile_sets:
        raise ProfileError("unsupported profile sets: " + ", ".join(unknown_profile_sets))
    if config.runtime == "current":
        _validate_current_runtime_extensions(config, policy)
        # Keep the generic programmatic entry point fail-closed too.  In
        # particular, pipeline callers must learn that image, mount, CPU
        # enforcement, or process-scope requests are invalid before they
        # create any output directories.
        from .current import validate_current_profile_config

        validate_current_profile_config(config)
    if config.profile_sets and config.runtime != "docker":
        raise ProfileError("profile sets currently require the Docker runtime")
    if config.profile_sets and (config.image is None or config.container is not None):
        raise ProfileError("profile sets require disposable --image mode")
    if config.coordinated_baseline and config.profile_sets:
        raise ProfileError("--coordinated-baseline is mutually exclusive with --profile-set")
    if config.coordinated_baseline and config.runtime != "docker":
        raise ProfileError("--coordinated-baseline currently requires the Docker runtime")
    if config.coordinated_baseline and (config.image is None or config.container is not None):
        raise ProfileError("--coordinated-baseline requires disposable --image mode")
    if config.coordinated_baseline and (
        config.scheduler_concurrency or config.proc_stat_comparison or config.memory_profile
    ):
        raise ProfileError("--coordinated-baseline cannot enable scheduler, proc-stat, or memory profilers")
    if config.verification_command is not None:
        if not isinstance(config.verification_command, str) or not config.verification_command.strip():
            raise ProfileError("verification command must be a nonempty string")
        if config.runtime != "docker":
            raise ProfileError("--verification-command currently requires Docker")
        if config.image is None or config.container is not None:
            raise ProfileError("--verification-command requires disposable --image mode")
        if not (config.profile_sets or config.coordinated_baseline):
            raise ProfileError("--verification-command requires --profile-set or --coordinated-baseline")
    if not isinstance(config.render_unified_view, bool):
        raise ProfileError("render_unified_view must be boolean")
    if (
        not config.hardware_perf_events
        or any(not isinstance(event, str) or not event.strip() for event in config.hardware_perf_events)
        or len(config.hardware_perf_events) != len(set(config.hardware_perf_events))
    ):
        raise ProfileError("hardware perf events must be nonempty and unique")
    if config.hardware_perf_attach_timeout_sec <= 0:
        raise ProfileError("hardware perf attach timeout must be positive")
    if not 0 < config.hardware_perf_minimum_running_ratio <= 1:
        raise ProfileError("hardware perf minimum running ratio must be in (0, 1]")
    for name, identity in (
        ("workload identity", config.workload_identity),
        ("source identity", config.source_identity),
        ("expected outcome", config.expected_outcome),
    ):
        if identity is not None and (not isinstance(identity, Mapping) or not identity):
            raise ProfileError(f"{name} must be a nonempty mapping when supplied")
    if config.memory_sample_interval_sec <= 0:
        raise ProfileError("memory sample interval must be positive")
    if config.memory_limit_bytes is not None and config.memory_limit_bytes <= 0:
        raise ProfileError("memory limit must be positive")
    if config.memory_swap_total_limit_bytes is not None:
        if config.memory_swap_total_limit_bytes <= 0:
            raise ProfileError("memory swap total limit must be positive")
        if config.memory_limit_bytes is None:
            raise ProfileError("memory swap total limit requires a memory limit")
        if config.memory_swap_total_limit_bytes < config.memory_limit_bytes:
            raise ProfileError("memory swap total limit must be at least the memory limit")
    if config.runtime == "enroot" and config.memory_profile:
        raise ProfileError("--memory-profile is currently Docker-only")
    if config.runtime == "enroot" and config.memory_limit_bytes is not None:
        raise ProfileError("--memory-limit-bytes is currently Docker-only")
    if config.runtime == "enroot" and config.memory_swap_total_limit_bytes is not None:
        raise ProfileError("--memory-swap-total-limit-bytes is currently Docker-only")
    if config.container is not None and config.memory_profile:
        raise ProfileError("--memory-profile supports disposable --image mode only")
    if config.container is not None and config.memory_limit_bytes is not None:
        raise ProfileError("--memory-limit-bytes supports disposable --image mode only")
    if config.container is not None and config.memory_swap_total_limit_bytes is not None:
        raise ProfileError("--memory-swap-total-limit-bytes supports disposable --image mode only")
    if config.network_attachment is not None and config.runtime != "docker":
        raise ProfileError("runtime network attachments are currently Docker-only")
    if policy is not None and policy.effective_mode is not NetworkMode.PUBLIC:
        enroot_no_network = config.runtime == "enroot" and policy.effective_mode is NetworkMode.NO_NETWORK
        if config.runtime != "docker" and not enroot_no_network:
            raise ProfileError(f"network policy {policy.effective_mode.value!r} is currently Docker-only")
        if config.image is None or config.container is not None:
            raise ProfileError(f"network policy {policy.effective_mode.value!r} requires disposable --image mode")
        if enroot_no_network:
            for mount in config.mounts:
                source, _target, _readonly = _parse_docker_bind_mount(mount)
                if _mount_exposes_container_runtime_socket(source):
                    raise ProfileError("no-network policy rejects container-runtime socket mounts: " + source)
    if policy is not None and policy.effective_mode is NetworkMode.ALLOWLIST:
        if not isinstance(config.network_policy_proxy_image, str) or not config.network_policy_proxy_image.strip():
            raise ProfileError("allowlist policy requires a nonempty proxy image")
        if (
            not isinstance(config.network_policy_external_network, str)
            or not config.network_policy_external_network.strip()
        ):
            raise ProfileError("allowlist policy requires a nonempty external proxy network")
        for mount in config.mounts:
            source, _target, _readonly = _parse_docker_bind_mount(mount)
            if _mount_exposes_container_runtime_socket(source):
                raise ProfileError("allowlist policy rejects container-runtime socket mounts: " + source)


def parse_cpuset_cpu_ids(cpuset: str) -> tuple[int, ...]:
    """Return distinct logical CPU IDs in Docker cpuset syntax."""
    cpu_ids: set[int] = set()
    for raw_part in cpuset.split(","):
        part = raw_part.strip()
        if not part:
            raise ProfileError(f"invalid empty CPU set component in {cpuset!r}")
        if "-" in part:
            start_raw, end_raw = part.split("-", 1)
            try:
                start, end = int(start_raw), int(end_raw)
            except ValueError as exc:
                raise ProfileError(f"invalid CPU set range: {part!r}") from exc
            if start < 0 or end < start:
                raise ProfileError(f"invalid CPU set range: {part!r}")
            cpu_ids.update(range(start, end + 1))
        else:
            try:
                cpu_id = int(part)
            except ValueError as exc:
                raise ProfileError(f"invalid CPU ID: {part!r}") from exc
            if cpu_id < 0:
                raise ProfileError(f"invalid CPU ID: {part!r}")
            cpu_ids.add(cpu_id)
    if not cpu_ids:
        raise ProfileError("CPU set must not be empty")
    return tuple(sorted(cpu_ids))


def parse_cpuset_cpu_count(cpuset: str) -> int:
    """Return the number of distinct logical CPU IDs in Docker cpuset syntax."""
    return len(parse_cpuset_cpu_ids(cpuset))


def _skip_cpu_stats(config: ProfileConfig) -> bool:
    """Whether to skip cgroup ``cpu.stat`` accounting for this run.

    Rootless enroot does not expose a per-container ``/sys/fs/cgroup/cpu.stat``,
    so task-scoped CPU cannot be reported. The container's monotonic
    ``start_ns``/``end_ns`` window still provides wall time.
    """
    return config.runtime == "enroot"


def _read_cpu_stat(path: Path) -> dict[str, int]:
    stats: dict[str, int] = {}
    for line in path.read_text().splitlines():
        fields = line.split()
        if len(fields) == 2:
            stats[fields[0]] = int(fields[1])
    return stats


def _read_enroot_task_clock_reference(output_dir: Path) -> dict[str, Any]:
    source = (output_dir / "clock_source").read_text().strip()
    clock_name = (output_dir / "clock_name").read_text().strip()
    if source != "linux-proc-uptime" or clock_name != "CLOCK_BOOTTIME":
        raise ProfileError(
            "unsupported Enroot wall clock source or clock name: " f"source={source!r}, clock_name={clock_name!r}"
        )
    try:
        resolution_ns = int((output_dir / "clock_resolution_ns").read_text().strip())
    except ValueError as exc:
        raise ProfileError("invalid Enroot task clock resolution") from exc
    if resolution_ns <= 0:
        raise ProfileError("invalid Enroot task clock resolution")
    boot_id = (output_dir / "clock_boot_id").read_text().strip() or None
    time_namespace = (output_dir / "clock_time_namespace").read_text().strip() or None
    raw_offset = (output_dir / "clock_time_namespace_boottime_offset").read_text().strip()
    boottime_offset_ns: int | None = None
    if raw_offset:
        fields = raw_offset.split()
        if len(fields) != 2:
            raise ProfileError("invalid Enroot time-namespace boot-time offset")
        try:
            seconds, nanoseconds = (int(field) for field in fields)
        except ValueError as exc:
            raise ProfileError("invalid Enroot time-namespace boot-time offset") from exc
        if not 0 <= nanoseconds < 1_000_000_000:
            raise ProfileError("invalid Enroot time-namespace boot-time offset")
        boottime_offset_ns = seconds * 1_000_000_000 + nanoseconds
    return {
        "clock_id": "enroot-task-boottime",
        "clock_name": clock_name,
        "source": source,
        "timestamp_unit": "nanoseconds",
        "resolution_ns": resolution_ns,
        "storage_precision_ns": 1,
        "boot_id": boot_id,
        "time_namespace": time_namespace,
        "time_namespace_boottime_offset_ns": boottime_offset_ns,
        "accuracy_claim": None,
    }


def _load_enroot_lifecycle(output_dir: Path) -> dict[str, Any]:
    path = output_dir / ENROOT_LIFECYCLE_FILENAME
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ProfileError("invalid Enroot lifecycle evidence") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != "tasktrace-enroot-lifecycle.v1":
        raise ProfileError("invalid Enroot lifecycle evidence")
    clock = payload.get("clock_reference")
    phases = payload.get("phases")
    if (
        not isinstance(clock, dict)
        or clock.get("clock_name") != "CLOCK_BOOTTIME"
        or payload.get("process_containment") != ENROOT_PROCESS_CONTAINMENT
        or not isinstance(phases, list)
        or not phases
    ):
        raise ProfileError("invalid Enroot lifecycle evidence")
    for phase in phases:
        if not isinstance(phase, dict):
            raise ProfileError("invalid Enroot lifecycle phase evidence")
        start_ns = phase.get("start_ns")
        end_ns = phase.get("end_ns")
        if (
            isinstance(start_ns, bool)
            or not isinstance(start_ns, int)
            or isinstance(end_ns, bool)
            or not isinstance(end_ns, int)
            or start_ns < 0
            or end_ns < start_ns
        ):
            raise ProfileError("invalid Enroot lifecycle phase evidence")
    return payload


def _profile_environment(config: ProfileConfig, container_output: str) -> list[str]:
    coordinated = bool(config.profile_sets) or config.coordinated_baseline
    environment = [
        "--env",
        f"TASKTRACE_PROFILE_DIR={container_output}",
        "--env",
        f"TASKTRACE_PROFILE_COMMAND={config.command}",
        "--env",
        f"TASKTRACE_STRACE_MODE={config.strace_mode}",
        "--env",
        f"TASKTRACE_STRACE_STRING_LIMIT={config.string_limit}",
        "--env",
        f"TASKTRACE_PROFILE_SHELL={config.shell}",
    ]
    if _skip_cpu_stats(config):
        # Tell the in-container entrypoint not to read /sys/fs/cgroup/cpu.stat,
        # which is absent under rootless enroot and would abort the run.
        environment.extend(["--env", "TASKTRACE_PROFILE_SKIP_CPU_STATS=1"])
    if config.prepare_command is not None:
        environment.extend(["--env", f"TASKTRACE_PROFILE_PREPARE_COMMAND={config.prepare_command}"])
    if config.verification_command is not None:
        environment.extend(
            [
                "--env",
                f"TASKTRACE_PROFILE_VERIFICATION_COMMAND={config.verification_command}",
            ]
        )
    if config.memory_profile or coordinated:
        environment.extend(["--env", "TASKTRACE_RESOURCE_GATE_DIR=/tasktrace-profile-output"])
    if config.scheduler_concurrency:
        environment.extend(["--env", "TASKTRACE_SCHEDULER_TRACING=1"])
    if config.scheduler_concurrency or config.proc_stat_comparison or config.memory_profile or coordinated:
        environment.extend(["--env", "TASKTRACE_SCHEDULER_GATE_DIR=/tasktrace-profile-output"])
    for assignment in config.env:
        environment.extend(["--env", assignment])
    return environment


def _escape_enroot_mount_path(path: str) -> str:
    """Escape spaces in an enroot fstab-style mount path."""
    return path.replace(" ", "\\040")


def _parse_docker_bind_mount(mount: str) -> tuple[str, str, bool]:
    fields: dict[str, str] = {}
    readonly = False
    for raw_part in mount.split(","):
        part = raw_part.strip()
        if not part:
            continue
        if part in {"readonly", "ro"}:
            readonly = True
            continue
        if "=" in part:
            key, value = part.split("=", 1)
            fields[key] = value
    if fields.get("type") not in {None, "bind"}:
        raise ProfileError(f"enroot mode only supports bind mounts: {mount!r}")
    source = fields.get("src") or fields.get("source")
    destination = fields.get("dst") or fields.get("target") or fields.get("destination")
    if not source or not destination:
        raise ProfileError(f"invalid bind mount specification: {mount!r}")
    return source, destination, readonly


def _format_enroot_mount(mount: str) -> str:
    """Translate a Docker bind spec to a destination-creating Enroot fstab."""
    if mount.startswith("type=") or "src=" in mount or "source=" in mount:
        source, destination, readonly = _parse_docker_bind_mount(mount)
        options = "x-create=auto,bind,ro" if readonly else "x-create=auto,bind,rw"
        return f"{_escape_enroot_mount_path(source)}:" f"{_escape_enroot_mount_path(destination)}:none:{options}"
    return mount


def _normalized_mount_provenance(mount: str) -> dict[str, Any]:
    """Describe mount behavior without run-specific host source paths."""
    if mount.startswith("type=") or "src=" in mount or "source=" in mount:
        _source, destination, readonly = _parse_docker_bind_mount(mount)
        return {
            "type": "bind",
            "destination": destination,
            "readonly": readonly,
        }
    fields = mount.split(":")
    destination = fields[1] if len(fields) >= 2 else None
    options = set(fields[2].split(",")) if len(fields) >= 3 else set()
    return {
        "type": "bind",
        "destination": destination,
        "readonly": bool({"ro", "readonly"} & options),
        "parsed": destination is not None,
    }


def _host_memory_platform_provenance() -> dict[str, Any]:
    """Return stable host details that affect memory comparability."""
    try:
        page_size_bytes = os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError):
        page_size_bytes = None
    return {
        "kernel_release": platform.release(),
        "machine": platform.machine(),
        "cgroup_version": "v2",
        "page_size_bytes": page_size_bytes,
    }


def controller_source_sha256(source_root: Path | str | None = None) -> str:
    """Hash execution-relevant packaged Python and shell controller sources.

    This intentionally is not cached. Repeated runs can share one controller
    process, and a source edit between executions must change the authoritative
    execution-plan identity rather than being hidden by a process-local cache.
    """
    if source_root is None:
        package_root = Path(__file__).resolve().parents[1]
    else:
        root = Path(source_root).resolve()
        package_root = root if root.name == "offline_replayer" else root / "offline_replayer"
        if not package_root.is_dir():
            raise ProfileError("controller source mount does not contain offline_replayer: " f"{root}")
    digest = hashlib.sha256()
    paths = sorted(
        path
        for path in package_root.rglob("*")
        if path.is_file()
        and path.suffix in {".py", ".sh"}
        and "tests" not in path.relative_to(package_root).parts
        and "__pycache__" not in path.relative_to(package_root).parts
    )
    if not paths:
        raise ProfileError("controller source tree contains no packaged Python or shell files: " f"{package_root}")
    for path in paths:
        relative = path.relative_to(package_root).as_posix().encode()
        digest.update(relative)
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def execution_comparability_sha256(provenance: Mapping[str, Any]) -> str:
    """Hash the canonical cross-phase execution-comparability projection."""
    comparable = {field: provenance[field] for field in EXECUTION_COMPARABILITY_FIELDS}
    # Additive v2 field: historical reports omit it and retain their original
    # digest, while every new semantic policy participates in comparability.
    if "network_policy" in provenance:
        comparable["network_policy"] = provenance["network_policy"]
    encoded = json.dumps(comparable, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _environment_provenance(assignments: Sequence[str]) -> list[dict[str, Any]]:
    """Return stable environment evidence without persisting secret values.

    Environment values affect reproducibility, so omitting them entirely would
    make two materially different runs appear comparable.  Persisting them
    verbatim is unsafe because profile reports are commonly archived and may
    contain API keys.  The key plus a domain-separated value digest preserves
    equality checks without exposing the value itself.
    """
    result: list[dict[str, Any]] = []
    for assignment in assignments:
        key, separator, value = assignment.partition("=")
        digest = hashlib.sha256(b"tasktrace-profile-environment-value-v1\0" + value.encode()).hexdigest()
        result.append(
            {
                "key": key,
                "value_sha256": digest,
                "assignment_had_equals": bool(separator),
            }
        )
    return sorted(
        result,
        key=lambda item: (
            item["key"],
            item["value_sha256"],
            item["assignment_had_equals"],
        ),
    )


def _memory_execution_provenance(config: ProfileConfig, image_identity: dict[str, Any] | None) -> dict[str, Any]:
    """Return execution-affecting settings plus a stable comparison key."""
    normalized_mounts = sorted(
        (_normalized_mount_provenance(mount) for mount in config.mounts),
        key=lambda item: json.dumps(item, sort_keys=True),
    )
    configured_swap_policy = "runtime-default"
    if config.memory_swap_total_limit_bytes is not None:
        configured_swap_policy = (
            "disabled" if config.memory_swap_total_limit_bytes == config.memory_limit_bytes else "finite-enabled"
        )
    comparable = {
        "runtime": config.runtime,
        "image": config.image,
        "image_identity": image_identity,
        "command": config.command,
        "prepare_command": config.prepare_command,
        "workdir": config.workdir,
        "shell": config.shell,
        "user": config.user,
        "cpuset_cpus": config.cpuset_cpus,
        "allocated_cpus": config.allocated_cpus,
        "cpu_limit": config.cpu_limit,
        "network": _legacy_network_projection(config),
        "environment": _environment_provenance(config.env),
        "normalized_mounts": normalized_mounts,
        "strace_mode": config.strace_mode,
        "scheduler_concurrency": config.scheduler_concurrency,
        "scheduler_backend": (config.scheduler_backend if config.scheduler_concurrency else None),
        "proc_stat_comparison": config.proc_stat_comparison,
        "terminate_descendants_at_finish": config.terminate_descendants_at_finish,
        "host_platform": _host_memory_platform_provenance(),
        "controller_source_sha256": controller_source_sha256(),
    }
    network_policy = _network_policy_comparability(config)
    if network_policy is not None:
        comparable["network_policy"] = network_policy
    return {
        **comparable,
        "memory_limit_bytes": config.memory_limit_bytes,
        "memory_swap_total_limit_bytes": config.memory_swap_total_limit_bytes,
        "configured_swap_policy": configured_swap_policy,
        "memory_sample_interval_sec": config.memory_sample_interval_sec,
        "raw_mounts": list(config.mounts),
        "comparability_sha256": execution_comparability_sha256(comparable),
        "comparability_excludes_raw_host_mount_sources": True,
        "comparability_excludes_memory_limits": True,
    }


def _validate_prepared_allowlist_binding(
    config: ProfileConfig,
    policy: NetworkPolicy,
    network_attachment: str | None,
) -> None:
    """Require the exact active backend evidence before granting proxy access."""
    enforcement = config.network_policy_enforcement
    if not isinstance(enforcement, Mapping):
        raise ProfileError("allowlist policy was not prepared by the Docker enforcement backend")
    validation = enforcement.get("validation")
    enforced_policy = enforcement.get("policy")
    proxy = enforcement.get("proxy")
    required_validation = (
        "external_network_inspected",
        "internal_network_inspected",
        "proxy_container_inspected",
        "proxy_ready",
        "proxy_bound_to_internal_address",
        "state_disjoint_from_workload_mounts",
    )
    external_identity = enforcement.get("external_network_identity")
    if (
        enforcement.get("state") != "active"
        or enforcement.get("backend") != "docker-forward-proxy-v1"
        or enforcement.get("docker_network_override") != network_attachment
        or not isinstance(validation, Mapping)
        or not all(validation.get(key) is True for key in required_validation)
        or not isinstance(enforced_policy, Mapping)
        or enforced_policy.get("effective_mode") != NetworkMode.ALLOWLIST.value
        or enforced_policy.get("allowed_hosts") != list(policy.allowed_hosts)
        or not isinstance(proxy, Mapping)
        or not isinstance(proxy.get("url"), str)
        or not isinstance(external_identity, Mapping)
        or not isinstance(external_identity.get("configuration_sha256"), str)
    ):
        raise ProfileError("allowlist policy was not prepared by the active validated Docker " "enforcement backend")
    proxy_url = proxy["url"]
    environment = {
        name: value
        for assignment in config.env
        for name, separator, value in (assignment.partition("="),)
        if name and separator
    }
    if any(environment.get(key) != proxy_url for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy")):
        raise ProfileError("allowlist policy was not prepared with the enforced proxy environment")


def build_image_command(config: ProfileConfig, *, container_name: str | None = None) -> list[str]:
    """Build a disposable `docker run` command for a configured image run."""
    if config.image is None:
        raise ProfileError("image mode requires --image")
    output_dir = str(config.output_dir.resolve())
    command = [
        config.docker,
        "run",
        "--rm",
        "--name",
        container_name or f"tasktrace-profile-{uuid.uuid4().hex[:12]}",
    ]
    policy = resolved_network_policy(config)
    if config.strace_mode != "none":
        # Docker's default security profile blocks ptrace. Request these only
        # when process tracing is enabled so CPU-only baselines work in more
        # restricted Docker environments.
        command.extend(["--cap-add", "SYS_PTRACE", "--security-opt", "seccomp=unconfined"])
    if config.cpuset_cpus:
        # A fixed cpuset creates a stable capacity denominator: CPU seconds /
        # wall seconds can then be reported as utilization of this allocation.
        command.extend(["--cpuset-cpus", config.cpuset_cpus])
    if config.cpu_limit is not None:
        # Docker CPU quota applies a CFS bandwidth limit while preserving
        # the container's visible CPU affinity and nproc behavior.
        command.extend(["--cpus", str(config.cpu_limit)])
    if config.memory_limit_bytes is not None:
        command.extend(["--memory", str(config.memory_limit_bytes)])
    if config.memory_swap_total_limit_bytes is not None:
        command.extend(["--memory-swap", str(config.memory_swap_total_limit_bytes)])
    if config.workdir:
        command.extend(["--workdir", config.workdir])
    if config.user:
        command.extend(["--user", config.user])
    network_attachment = config.network_attachment
    if network_attachment is None:
        if policy is not None and policy.effective_mode is NetworkMode.NO_NETWORK:
            network_attachment = "none"
        elif policy is not None and policy.effective_mode is NetworkMode.ALLOWLIST:
            raise ProfileError("allowlist policy was not prepared by the Docker enforcement backend")
        else:
            network_attachment = config.network
    if network_attachment:
        command.extend(["--network", network_attachment])
    if policy is not None and policy.effective_mode is NetworkMode.ALLOWLIST:
        _validate_prepared_allowlist_binding(config, policy, network_attachment)
        # Raw packet sockets are unnecessary for HTTP(S) proxying and would
        # broaden the bypass surface on the isolated layer-2 network.
        command.extend(["--cap-drop", "NET_RAW"])
    for mount in config.mounts:
        command.extend(["--mount", mount])
    command.extend(
        [
            "--mount",
            f"type=bind,src={ENTRYPOINT.resolve()},dst=/tasktrace-profile-entrypoint.sh,readonly",
            "--mount",
            f"type=bind,src={output_dir},dst=/tasktrace-profile-output",
        ]
    )
    command.extend(_profile_environment(config, "/tasktrace-profile-output"))
    # Override arbitrary benchmark-image entrypoints so every image runs the
    # same measurement harness and applies identical measurement boundaries.
    command.extend(["--entrypoint", "/bin/sh", config.image, "/tasktrace-profile-entrypoint.sh"])
    return command


def build_enroot_image_command(
    config: ProfileConfig,
    *,
    network_evidence_path: Path | None = None,
) -> list[str]:
    """Build a disposable enroot command for a configured image run."""
    if config.image is None:
        raise ProfileError("enroot image mode requires --image")
    if config.container is not None:
        raise ProfileError("enroot mode does not support --container")
    policy = resolved_network_policy(config)
    if config.network not in (None, "none") or (policy is not None and policy.mode is NetworkMode.ALLOWLIST):
        raise ProfileError("Enroot supports only public or no-network policy")
    if config.cpu_limit is not None:
        raise ProfileError("--cpu-limit is Docker-only; use taskset affinity for Enroot")
    if config.memory_limit_bytes is not None:
        raise ProfileError("--memory-limit-bytes is currently Docker-only")
    if config.memory_swap_total_limit_bytes is not None:
        raise ProfileError("--memory-swap-total-limit-bytes is currently Docker-only")
    if config.memory_profile:
        raise ProfileError("--memory-profile requires a task-scoped cgroup and is currently Docker-only")
    output_dir = str(config.output_dir.resolve())
    # Parity with the Rust runtime: disable enroot's home hook so a site's
    # ENROOT_MOUNT_HOME=yes never mounts the invoking user's $HOME over /root,
    # which would hide the image's own /root dotfiles and run the user's shell
    # setup inside the task. Set on the enroot process, not the container.
    command = ["env", "ENROOT_MOUNT_HOME=no", config.enroot, "start", "--rw"]
    if config.user == "root":
        command.append("--root")
    elif config.user is not None:
        raise ProfileError("enroot mode only supports --user root")
    for mount in config.mounts:
        command.extend(["--mount", _format_enroot_mount(mount)])
    command.extend(
        [
            "--mount",
            _format_enroot_mount(f"type=bind,src={ENTRYPOINT.resolve()},dst=/tasktrace-profile-entrypoint.sh,readonly"),
            "--mount",
            _format_enroot_mount(f"type=bind,src={output_dir},dst=/tasktrace-profile-output"),
        ]
    )
    command.extend(_profile_environment(config, "/tasktrace-profile-output"))

    # Run the bind-mounted entrypoint through /bin/sh rather than exec-ing it
    # directly: the read-only mount carries the host file mode (0644), so a
    # direct exec fails with "Permission denied". The Docker path likewise runs
    # `/bin/sh <script>`.
    inner_command = "exec /bin/sh /tasktrace-profile-entrypoint.sh"
    if config.workdir:
        inner_command = f"cd {shlex.quote(config.workdir)} && {inner_command}"
    inner = ["/bin/sh", "-c", inner_command]
    command.extend(["--", config.image, *inner])
    if config.cpuset_cpus:
        command = ["taskset", "-c", config.cpuset_cpus, *command]
    # Known divergence from the Rust implementation: the native runner
    # detects hosts where unprivileged util-linux unshare cannot create user
    # namespaces (Ubuntu 24.04 AppArmor restriction) and falls back to
    # bubblewrap. This retained differential-test oracle deliberately keeps
    # the historical util-linux invocation and therefore cannot execute on
    # such hosts.
    namespace_command = [
        "unshare",
        "--user",
        "--map-current-user",
        "--pid",
        "--fork",
        "--kill-child=SIGKILL",
        "--mount-proc",
    ]
    if policy is not None and policy.effective_mode is NetworkMode.NO_NETWORK:
        if network_evidence_path is None:
            raise ProfileError("Enroot no-network requires an evidence path")
        command = [
            *namespace_command,
            "--net",
            "--",
            sys.executable,
            str(Path(__file__).with_name("enroot_network.py").resolve()),
            "--parent-netns-inode",
            str(Path("/proc/self/ns/net").stat().st_ino),
            "--controller-euid",
            str(os.geteuid()),
            "--controller-egid",
            str(os.getegid()),
            "--evidence-path",
            str(network_evidence_path.resolve()),
            "--",
            *command,
        ]
    else:
        command = [*namespace_command, "--", *command]
    return command


def _run_enroot_image(
    config: ProfileConfig,
    *,
    network_evidence_path: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a disposable enroot container, creating it from a .sqsh first.

    ``enroot start`` cannot mount a ``.sqsh`` path directly without squashfuse
    (absent on many hosts), so when given an image file we first
    ``enroot create`` an extracted, uniquely named container, run it, then
    remove it. This also gives each concurrent sandbox its own writable
    rootfs. A non-file ``image`` is treated as an already-created container name.
    """
    try:
        lifecycle_clock = boottime_clock_reference()
    except RuntimeError as exc:
        raise ProfileError(str(exc)) from exc
    phases: list[dict[str, Any]] = []

    def run_phase(phase: str, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        clock_id = getattr(time, "CLOCK_BOOTTIME", None)
        if clock_id is None:
            raise ProfileError("CLOCK_BOOTTIME is unavailable on this platform")
        start_ns = time.clock_gettime_ns(clock_id)
        try:
            completed = _run(command)
        except BaseException:
            end_ns = time.clock_gettime_ns(clock_id)
            phases.append(
                {
                    "phase": phase,
                    "start_ns": start_ns,
                    "end_ns": end_ns,
                    "duration_sec": (end_ns - start_ns) / 1_000_000_000,
                    "returncode": None,
                }
            )
            raise
        end_ns = time.clock_gettime_ns(clock_id)
        if end_ns < start_ns:
            raise ProfileError(f"CLOCK_BOOTTIME moved backwards during Enroot {phase}")
        phases.append(
            {
                "phase": phase,
                "start_ns": start_ns,
                "end_ns": end_ns,
                "duration_sec": (end_ns - start_ns) / 1_000_000_000,
                "returncode": completed.returncode,
            }
        )
        return completed

    def retain_lifecycle(
        *,
        container_name: str | None,
        container_created: bool,
        cleanup: Mapping[str, Any] | None,
    ) -> None:
        payload = {
            "schema_version": "tasktrace-enroot-lifecycle.v1",
            "clock_reference": lifecycle_clock,
            "process_containment": ENROOT_PROCESS_CONTAINMENT,
            "container_name": container_name,
            "container_created": container_created,
            "window_start_ns": phases[0]["start_ns"] if phases else None,
            "window_end_ns": phases[-1]["end_ns"] if phases else None,
            "phases": phases,
            "cleanup": dict(cleanup) if cleanup is not None else None,
        }
        (config.output_dir / ENROOT_LIFECYCLE_FILENAME).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    def cleanup_named_container(
        container_name: str,
    ) -> tuple[dict[str, Any], list[str]]:
        cleanup_errors: list[str] = []
        removed: subprocess.CompletedProcess[str] | None = None
        listed: subprocess.CompletedProcess[str] | None = None
        container_absent: bool | None = None
        try:
            removed = run_phase("remove", [config.enroot, "remove", "-f", container_name])
            if removed.returncode != 0:
                cleanup_errors.append("enroot remove failed for " f"{container_name!r}: {removed.stderr.strip()}")
        except BaseException as exc:
            cleanup_errors.append(f"enroot remove failed for {container_name!r}: {exc}")
        try:
            listed = run_phase("list", [config.enroot, "list"])
            if listed.returncode != 0:
                cleanup_errors.append("enroot list failed while verifying cleanup: " f"{listed.stderr.strip()}")
            else:
                container_absent = container_name not in {line.strip() for line in listed.stdout.splitlines()}
                if not container_absent:
                    cleanup_errors.append(f"Enroot container {container_name!r} survived removal")
        except BaseException as exc:
            cleanup_errors.append(f"enroot list failed while verifying cleanup: {exc}")
        return (
            {
                "remove_returncode": (removed.returncode if removed is not None else None),
                "list_returncode": listed.returncode if listed is not None else None,
                "container_absent": container_absent,
                "valid": not cleanup_errors and container_absent is True,
                "errors": cleanup_errors,
            },
            cleanup_errors,
        )

    image_path = Path(config.image) if config.image else None
    if image_path is not None and image_path.is_file():
        container_name = f"tasktrace-profile-{uuid.uuid4().hex[:12]}"
        try:
            created = run_phase(
                "create",
                [config.enroot, "create", "--name", container_name, str(image_path)],
            )
        except BaseException as create_error:
            cleanup, cleanup_errors = cleanup_named_container(container_name)
            retain_lifecycle(
                container_name=container_name,
                container_created=False,
                cleanup=cleanup,
            )
            if cleanup_errors:
                raise ProfileError("; ".join(cleanup_errors)) from create_error
            raise
        if created.returncode != 0:
            cleanup, cleanup_errors = cleanup_named_container(container_name)
            retain_lifecycle(
                container_name=container_name,
                container_created=False,
                cleanup=cleanup,
            )
            raise ProfileError(
                f"enroot create failed for {config.image!r}: {created.stderr.strip()}"
                + ("; " + "; ".join(cleanup_errors) if cleanup_errors else "")
            )
        completed: subprocess.CompletedProcess[str] | None = None
        execution_error: BaseException | None = None
        cleanup_errors: list[str] = []
        try:
            completed = run_phase(
                "start",
                build_enroot_image_command(
                    replace(config, image=container_name),
                    network_evidence_path=network_evidence_path,
                ),
            )
        except BaseException as exc:
            execution_error = exc
        finally:
            cleanup, cleanup_errors = cleanup_named_container(container_name)
            retain_lifecycle(
                container_name=container_name,
                container_created=True,
                cleanup=cleanup,
            )
        if cleanup_errors:
            cleanup_error = ProfileError("; ".join(cleanup_errors))
            if execution_error is not None:
                raise cleanup_error from execution_error
            raise cleanup_error
        if execution_error is not None:
            raise execution_error
        assert completed is not None
        return completed
    completed = run_phase(
        "start",
        build_enroot_image_command(
            config,
            network_evidence_path=network_evidence_path,
        ),
    )
    retain_lifecycle(
        container_name=config.image,
        container_created=False,
        cleanup=None,
    )
    return completed


def build_existing_container_exec_command(config: ProfileConfig, remote_dir: str, remote_entrypoint: str) -> list[str]:
    """Build the `docker exec` command used for an already-running container.

    This mode cannot retrofit cpuset or security settings on the existing
    container, which is why it is diagnostic rather than comparison-grade.
    """
    if config.container is None:
        raise ProfileError("existing-container mode requires --container")
    command = [config.docker, "exec"]
    if config.workdir:
        command.extend(["--workdir", config.workdir])
    if config.user:
        command.extend(["--user", config.user])
    command.extend(_profile_environment(config, remote_dir))
    command.extend([config.container, "/bin/sh", remote_entrypoint])
    return command


def build_report(
    config: ProfileConfig,
    *,
    scope: str,
    docker_returncode: int,
    image_identity: dict[str, Any] | None = None,
    run_id: str | None = None,
    profile_started_at_utc: str | None = None,
    coordinated_cpu_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON report from artifacts produced inside the target container."""
    skip_cpu = _skip_cpu_stats(config)
    required = ("start_ns", "end_ns", "exit_code")
    if skip_cpu:
        required = (
            *required,
            "clock_source",
            "clock_name",
            "clock_resolution_ns",
            "clock_boot_id",
            "clock_time_namespace",
            "clock_time_namespace_boottime_offset",
            "initial_effective_cpu_affinity",
            ENROOT_LIFECYCLE_FILENAME,
        )
    if not skip_cpu and coordinated_cpu_report is None:
        required = ("cpu.stat.before", "cpu.stat.after", *required)
    missing = [name for name in required if not (config.output_dir / name).exists()]
    if missing:
        raise ProfileError("profile run did not produce required artifacts: " + ", ".join(missing))
    start_ns = int((config.output_dir / "start_ns").read_text().strip())
    end_ns = int((config.output_dir / "end_ns").read_text().strip())
    if start_ns < 0 or end_ns <= start_ns:
        raise ProfileError("profile wall-clock timestamps are not strictly increasing")
    wall_sec = (end_ns - start_ns) / 1_000_000_000
    wall_clock_source = (
        "host-monotonic"
        if coordinated_cpu_report is not None
        else ("linux-proc-uptime" if skip_cpu else "container-realtime-with-proc-uptime-fallback")
    )
    task_clock_reference = _read_enroot_task_clock_reference(config.output_dir) if skip_cpu else None
    initial_effective_cpu_affinity: str | None = None
    initial_effective_logical_cpu_ids: tuple[int, ...] | None = None
    if skip_cpu:
        initial_effective_cpu_affinity = (config.output_dir / "initial_effective_cpu_affinity").read_text().strip()
        initial_effective_logical_cpu_ids = parse_cpuset_cpu_ids(initial_effective_cpu_affinity)
        if config.cpuset_cpus is not None and initial_effective_logical_cpu_ids != (
            parse_cpuset_cpu_ids(config.cpuset_cpus)
        ):
            raise ProfileError("initial effective Enroot CPU affinity differs from the request")
    enroot_lifecycle = _load_enroot_lifecycle(config.output_dir) if skip_cpu else None
    capacity = config.allocated_cpus
    if capacity is None and config.cpuset_cpus:
        capacity = float(parse_cpuset_cpu_count(config.cpuset_cpus))
    allocated_cpu_source: str | None = None
    if capacity is not None:
        if config.runtime == "enroot":
            allocated_cpu_source = "taskset-requested-affinity" if config.cpuset_cpus else "user-supplied-non-enforced"
        else:
            allocated_cpu_source = "cpuset-cpus" if config.cpuset_cpus else "user-supplied"
    if coordinated_cpu_report is not None:
        try:
            wall_sec = float(coordinated_cpu_report["wall_sec"])
            usage_sec = float(coordinated_cpu_report["usage_sec"])
            user_sec = float(coordinated_cpu_report["user_sec"])
            system_sec = float(coordinated_cpu_report["system_sec"])
            avg_cores = float(coordinated_cpu_report["average_logical_cores"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProfileError("coordinated cgroup CPU report is missing required numeric fields") from exc
        normalized_utilization = avg_cores / capacity if capacity is not None and capacity > 0 else None
    elif skip_cpu:
        # Rootless Enroot has no task-scoped cgroup cpu.stat; report wall only.
        usage_sec = user_sec = system_sec = None
        avg_cores = None
        normalized_utilization = None
    else:
        before = _read_cpu_stat(config.output_dir / "cpu.stat.before")
        after = _read_cpu_stat(config.output_dir / "cpu.stat.after")
        cpu_stats = {key: after.get(key, 0) - before.get(key, 0) for key in set(before) | set(after)}
        usage_sec = cpu_stats.get("usage_usec", 0) / 1_000_000
        user_sec = cpu_stats.get("user_usec", 0) / 1_000_000
        system_sec = cpu_stats.get("system_usec", 0) / 1_000_000
        # ``usage_sec / wall_sec`` is average busy logical-core equivalents. It
        # is intentionally not described as peak parallelism or CPU residency.
        avg_cores = usage_sec / wall_sec if wall_sec > 0 else None
        normalized_utilization = avg_cores / capacity if avg_cores is not None and capacity else None
    trace_path = config.output_dir / "process.strace.log"
    process_exec_path = config.output_dir / "process_exec_report.json"
    process_exec_report: dict[str, Any] | None = None
    if config.strace_mode == "process" and trace_path.is_file():
        process_exec_report = load_process_exec_trace(
            trace_path,
            initial_cwd=config.workdir,
        )
        process_exec_report["artifacts"] = {
            "raw_trace_path": str(trace_path),
            "report_path": str(process_exec_path),
        }
        write_process_exec_report(process_exec_report, process_exec_path)
    cpu_summary: dict[str, Any] = {
        "method": ("wall-clock-only-no-cgroup" if skip_cpu else "container-cgroup-v2-cpu.stat"),
        "wall_clock_source": wall_clock_source,
        "wall_sec": round(wall_sec, 9),
        "usage_sec": round(usage_sec, 9) if usage_sec is not None else None,
        "user_sec": round(user_sec, 9) if user_sec is not None else None,
        "system_sec": round(system_sec, 9) if system_sec is not None else None,
        "average_logical_cores_used": (round(avg_cores, 9) if avg_cores is not None else None),
        "allocated_logical_cpus": capacity,
        "enforced_logical_cpu_limit": config.cpu_limit,
        "enforced_cpu_limit_source": (
            "docker --cpus" if config.runtime == "docker" and config.cpu_limit is not None else None
        ),
        "allocated_cpu_source": allocated_cpu_source,
        "utilization_fraction_of_allocation": (
            round(normalized_utilization, 9) if normalized_utilization is not None else None
        ),
        "utilization_percent_of_allocation": (
            round(100 * normalized_utilization, 6) if normalized_utilization is not None else None
        ),
    }
    if skip_cpu:
        cpu_summary.update(
            {
                "window_start_ns": start_ns,
                "window_end_ns": end_ns,
                "clock_reference": task_clock_reference,
                "requested_cpu_affinity": config.cpuset_cpus,
                "initial_effective_cpu_affinity": initial_effective_cpu_affinity,
                "initial_effective_logical_cpu_ids": list(initial_effective_logical_cpu_ids or ()),
                "cpu_affinity_enforced": False,
            }
        )
    if coordinated_cpu_report is not None:
        # Keep the stable public aliases above while publishing the full exact
        # controlled-finish-gate report, including leaf/ancestor validity.
        cpu_summary = {**dict(coordinated_cpu_report), **cpu_summary}

    report = {
        "schema_version": "container-command-profile.v1",
        "runtime": config.runtime,
        "scope": scope,
        "image": config.image,
        "image_identity": image_identity,
        "container": config.container,
        "command": config.command,
        "prepare_command": config.prepare_command,
        "network": _legacy_network_projection(config),
        "network_policy": _network_policy_report(config),
        "shell": config.shell,
        "strace": {
            "mode": config.strace_mode,
            "log_path": str(trace_path) if trace_path.exists() else None,
            "structured_report_path": (str(process_exec_path) if process_exec_report is not None else None),
            "string_limit": config.string_limit,
        },
        "process_exec": process_exec_report,
        "command_exit_code": int((config.output_dir / "exit_code").read_text().strip()),
        "docker_returncode": docker_returncode,
        "cpu": cpu_summary,
        "warning": (
            "Existing-container mode includes CPU from all other concurrent work "
            "in the container; use disposable image mode for comparable results."
            if scope == "existing-container"
            else (
                f"This run includes {config.scheduler_backend} scheduler-tracing overhead. "
                "Use simultaneous-core fields only when their cgroup validation passes."
                if config.scheduler_concurrency
                else (
                    "This run includes host-side cgroup CPU/memory sampling. Use the "
                    "fresh-cgroup memory peak through the controlled finish gate for "
                    "replay capacity and the sampled timeline only for over-time shape."
                    if config.memory_profile
                    else (
                        "This run includes sampled host /proc/stat context on the allocated "
                        "CPUs. Use cgroup fields for task average CPU; sampled active CPUs "
                        "are not task peak concurrency."
                        if config.proc_stat_comparison
                        else (
                            "This diagnostic run includes strace overhead. Compare provider "
                            "performance using identically configured instrumentation or use "
                            "--strace-mode none for an untraced CPU baseline."
                            if config.strace_mode != "none"
                            else "This is an untraced CPU baseline; no process launch log was collected."
                        )
                    )
                )
            )
        ),
    }
    if enroot_lifecycle is not None:
        report["enroot_lifecycle"] = enroot_lifecycle
    if run_id is not None:
        report["run_id"] = run_id
    if profile_started_at_utc is not None:
        report["profile_started_at_utc"] = profile_started_at_utc
    return report


def _run(command: Sequence[str], *, check: bool = False) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, capture_output=True, check=check)


def _inspect_image_identity(config: ProfileConfig) -> dict[str, Any]:
    """Resolve an image tag to immutable and platform-specific evidence.

    Docker image IDs and root-filesystem layer digests identify the exact local
    image that is executed.  Repository digests retain the registry identity
    needed to establish that different native-ISA images came from the same
    pinned multi-platform source.  The selected config fields explain
    execution defaults without archiving image environment variables or
    arbitrary labels, either of which may contain credentials.
    """
    assert config.image is not None
    if config.runtime == "enroot":
        image_path = Path(config.image)
        identity: dict[str, Any] = {"runtime": "enroot", "image": config.image}
        if image_path.is_file():
            stat = image_path.stat()
            identity.update(
                {
                    "path": str(image_path.resolve()),
                    "size_bytes": stat.st_size,
                    "mtime_ns": stat.st_mtime_ns,
                }
            )
        return identity
    inspected = _run([config.docker, "image", "inspect", config.image])
    if inspected.returncode != 0:
        raise ProfileError(f"unable to inspect Docker image {config.image!r}: {inspected.stderr.strip()}")
    try:
        inspected_payload = json.loads(inspected.stdout)
    except json.JSONDecodeError as exc:
        raise ProfileError(f"invalid Docker image inspection output for {config.image!r}") from exc
    if (
        not isinstance(inspected_payload, list)
        or len(inspected_payload) != 1
        or not isinstance(inspected_payload[0], dict)
    ):
        raise ProfileError(f"invalid Docker image inspection output for {config.image!r}")
    metadata = inspected_payload[0]

    rootfs = metadata.get("RootFS")
    if not isinstance(rootfs, dict):
        rootfs = {}
    image_config = metadata.get("Config")
    if not isinstance(image_config, dict):
        image_config = {}
    return {
        "id": metadata.get("Id"),
        "repo_digests": metadata.get("RepoDigests", []),
        "created": metadata.get("Created"),
        "architecture": metadata.get("Architecture"),
        "os": metadata.get("Os"),
        "variant": metadata.get("Variant"),
        "size_bytes": metadata.get("Size"),
        "rootfs": {
            "type": rootfs.get("Type"),
            "layers": rootfs.get("Layers", []),
        },
        "config": {
            "user": image_config.get("User"),
            "working_dir": image_config.get("WorkingDir"),
            "entrypoint": image_config.get("Entrypoint"),
            "cmd": image_config.get("Cmd"),
            "shell": image_config.get("Shell"),
        },
    }


def _immutable_execution_config(config: ProfileConfig, image_identity: Mapping[str, Any]) -> ProfileConfig:
    """Bind a Docker run to the exact image that was inspected.

    Recording an image ID while launching a mutable tag leaves a race in which
    the tag can change between inspection and ``docker run``.  Keep the
    user-supplied reference in report provenance, but execute the resolved ID.
    """
    if config.runtime != "docker":
        return config
    image_id = image_identity.get("id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ProfileError("Docker image inspection did not return an immutable sha256 image ID")
    return replace(config, image=image_id)


def _preflight_check(name: str, status: str, **details: Any) -> dict[str, Any]:
    return {"name": name, "status": status, **details}


def _preflight_output(path: Path) -> dict[str, Any]:
    """Validate the target without creating it or probing with a write."""
    if path.exists():
        if not path.is_dir():
            raise ProfileError(f"profile output path is not a directory: {path}")
        if any(path.iterdir()):
            raise ProfileError(f"profile output directory must be new or empty: {path}")
        state = "existing-empty"
    else:
        state = "absent"
    return _preflight_check("output_directory", "passed", state=state)


def _preflight_command(command: Sequence[str], description: str) -> str:
    completed = _run(command)
    if completed.returncode != 0:
        diagnostic = completed.stderr.strip()[:512]
        raise ProfileError(
            f"{description} failed with exit code {completed.returncode}" + (f": {diagnostic}" if diagnostic else "")
        )
    return completed.stdout.strip()


def _preflight_executable(command: str, description: str) -> str:
    resolved = shutil.which(command)
    if resolved is None:
        raise ProfileError(f"{description} executable is unavailable")
    return str(Path(resolved).resolve())


def _preflight_cgroup_v2(*, required: bool, memory: bool) -> dict[str, Any]:
    try:
        controllers = set(Path("/sys/fs/cgroup/cgroup.controllers").read_text().split())
        unified = any(line.startswith("0::") for line in Path("/proc/self/cgroup").read_text().splitlines())
    except OSError as exc:
        if required:
            raise ProfileError(f"cgroup v2 is unavailable: {exc}") from exc
        return _preflight_check("cgroup_v2", "unavailable", required=False)
    needed = {"cpu"}
    if memory:
        needed.add("memory")
    reason = None
    if not unified:
        reason = "unified membership is absent"
    elif missing := sorted(needed - controllers):
        reason = "required controllers are absent: " + ", ".join(missing)
    if reason is not None:
        if required:
            raise ProfileError(f"cgroup v2 preflight failed: {reason}")
        return _preflight_check("cgroup_v2", "unavailable", required=False, reason=reason)
    return _preflight_check("cgroup_v2", "passed", required=required, controllers=sorted(needed))


def _online_cpu_ids() -> tuple[int, ...]:
    try:
        return parse_cpuset_cpu_ids(Path("/sys/devices/system/cpu/online").read_text().strip())
    except OSError:
        count = os.cpu_count()
        if count is None or count <= 0:
            raise ProfileError("host online CPU set is unavailable") from None
        return tuple(range(count))


def _preflight_cpu(config: ProfileConfig) -> dict[str, Any]:
    if config.cpu_limit is not None and config.cpu_limit <= 0:
        raise ProfileError("CPU limit must be positive")
    if config.allocated_cpus is not None and config.allocated_cpus <= 0:
        raise ProfileError("allocated CPU count must be positive")
    if config.cpuset_cpus is None:
        return _preflight_check(
            "cpu_allocation",
            "passed" if config.allocated_cpus is not None else "not_requested",
            source="allocated_cpus" if config.allocated_cpus is not None else None,
        )
    requested = set(parse_cpuset_cpu_ids(config.cpuset_cpus))
    if config.runtime == "docker":
        available = set(_online_cpu_ids())
    else:
        try:
            available = set(os.sched_getaffinity(0))
        except (AttributeError, OSError):
            available = set(_online_cpu_ids())
    if missing := sorted(requested - available):
        raise ProfileError("requested CPU set contains unavailable logical CPUs: " + ", ".join(map(str, missing)))
    return _preflight_check("cpu_allocation", "passed", source="cpuset_cpus", logical_cpu_count=len(requested))


def _safe_image_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Exclude image environment, labels, and command strings from public JSON."""
    keys = (
        "runtime",
        "id",
        "repo_digests",
        "created",
        "architecture",
        "os",
        "variant",
        "size_bytes",
        "rootfs",
        "path",
        "mtime_ns",
        "sha256",
        "container_name",
        "immutable",
    )
    return {key: identity[key] for key in keys if key in identity}


def _docker_preflight_identity(config: ProfileConfig, image: str) -> dict[str, Any]:
    inspect_config = replace(config, image=image, container=None)
    try:
        identity = _inspect_image_identity(inspect_config)
    except ProfileError as exc:
        # The image reference may contain registry credentials. Keep the public
        # preflight failure useful without reflecting that reference to JSON.
        raise ProfileError("unable to inspect Docker image identity") from exc
    _immutable_execution_config(inspect_config, identity)
    safe = _safe_image_identity(identity)
    image_id = safe.get("id")
    digest = image_id.removeprefix("sha256:") if isinstance(image_id, str) else ""
    if (
        not isinstance(image_id, str)
        or not image_id.startswith("sha256:")
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ProfileError("Docker preflight image identity has no immutable ID")
    return safe


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _preflight_perf(binary: str, name: str) -> dict[str, Any]:
    executable = _preflight_executable(binary, "perf")
    version = _preflight_command([binary, "--version"], "perf version inspection")
    return _preflight_check(
        name,
        "passed",
        executable=executable,
        version=version.splitlines()[0][:256] if version else None,
    )


def _require_bcc_bindings() -> None:
    try:
        from bcc import BPF  # type: ignore[import-not-found]  # noqa: F401
    except (ImportError, OSError) as exc:
        raise ProfileError("BCC scheduler profiling requires usable Python BCC bindings") from exc


def _preflight_runtime(
    config: ProfileConfig, checks: list[dict[str, Any]], limitations: list[str]
) -> dict[str, Any] | None:
    if config.runtime == "current":
        from .current import resolve_current_cgroup

        resolve_current_cgroup()
        checks.extend(
            (
                _preflight_check("runtime", "passed", runtime="current"),
                _preflight_check("cgroup_v2", "passed", required=True),
            )
        )
        limitations.append("The current runtime has no independently inspectable immutable image identity.")
        return None

    runtime = config.docker if config.runtime == "docker" else config.enroot
    executable = _preflight_executable(runtime, config.runtime)
    if config.runtime == "docker":
        version = _preflight_command(
            [config.docker, "version", "--format", "{{.Server.Version}}"],
            "Docker runtime inspection",
        )
        cgroup_version = _preflight_command(
            [config.docker, "info", "--format", "{{.CgroupVersion}}"],
            "Docker cgroup-version inspection",
        )
        if cgroup_version.strip().lower() not in {"2", "v2"}:
            raise ProfileError("Docker runtime does not use cgroup v2")
        checks.extend(
            (
                _preflight_check(
                    "runtime",
                    "passed",
                    runtime="docker",
                    executable=executable,
                    version=version.splitlines()[0][:256] if version else None,
                ),
                _preflight_cgroup_v2(
                    required=True,
                    memory=(config.memory_profile or bool({"cpu-compare", "memory-detail"} & set(config.profile_sets))),
                ),
            )
        )
        if config.image is not None:
            identity = _docker_preflight_identity(config, config.image)
            kind = "docker-image"
        elif config.container is not None:
            fields = _preflight_command(
                [config.docker, "inspect", "--format", "{{.Image}} {{.State.Running}}", config.container],
                "Docker container inspection",
            ).split()
            if len(fields) != 2 or fields[1].lower() != "true":
                raise ProfileError("Docker profiler container is not running")
            identity = _docker_preflight_identity(config, fields[0])
            kind = "existing-container-image"
        else:
            raise ProfileError("Docker preflight requires an image or container")
        checks.append(_preflight_check("image", "passed", kind=kind))
        policy = resolved_network_policy(config)
        if policy is not None and policy.effective_mode is NetworkMode.ALLOWLIST:
            proxy = _docker_preflight_identity(config, config.network_policy_proxy_image)
            checks.append(_preflight_check("network_policy_proxy_image", "passed", image_identity=proxy))
            message = "Allowlist network creation and proxy readiness remain pending execution."
            checks.append(_preflight_check("network_policy_dynamic_enforcement", "pending", reason=message))
            limitations.append(message)
        return identity

    version = _preflight_command([config.enroot, "version"], "Enroot runtime inspection")
    checks.extend(
        (
            _preflight_check(
                "runtime",
                "passed",
                runtime="enroot",
                executable=executable,
                version=version.splitlines()[0][:256] if version else None,
            ),
            _preflight_cgroup_v2(required=False, memory=False),
        )
    )
    assert config.image is not None
    image = Path(config.image)
    if image.is_file():
        digest = _file_sha256(image)
        identity = {
            "runtime": "enroot",
            "path": str(image.resolve()),
            "size_bytes": image.stat().st_size,
            "mtime_ns": image.stat().st_mtime_ns,
            "sha256": digest,
            "immutable": True,
        }
        kind = "enroot-image-file"
    else:
        names = {
            line.split()[0]
            for line in _preflight_command([config.enroot, "list"], "Enroot container inspection").splitlines()
            if line.split()
        }
        if config.image not in names:
            raise ProfileError("Enroot image file or created container is unavailable")
        identity = {"runtime": "enroot", "container_name": config.image, "immutable": False}
        kind = "enroot-created-container"
        limitations.append("An Enroot created-container name has no immutable image digest.")
    checks.append(_preflight_check("image", "passed", kind=kind))
    limitations.append("The current Enroot backend is wall-time-only; task-scoped resource evidence is unavailable.")
    return identity


def preflight_profile(config: ProfileConfig) -> dict[str, Any]:
    """Validate static prerequisites without creating output or running work."""
    validate_profile_config(config)
    if config.runtime in CONTAINER_RUNTIMES:
        if (config.image is None) == (config.container is None):
            raise ProfileError("profile request requires exactly one image or container")
    if config.runtime == "enroot":
        if config.container is not None:
            raise ProfileError("enroot mode does not support --container")
        if config.user not in (None, "root"):
            raise ProfileError("enroot mode only supports --user root")
        if config.strace_mode != "none":
            raise ProfileError("strace tracing requires Docker; use --strace-mode none")
    if (
        config.runtime == "docker"
        and config.container is not None
        and (config.cpuset_cpus is not None or config.cpu_limit is not None or config.mounts)
    ):
        raise ProfileError("existing-container preflight cannot apply CPU or mount controls")
    if config.string_limit <= 0:
        raise ProfileError("strace string limit must be positive")
    if config.scheduler_attach_timeout_sec <= 0:
        raise ProfileError("scheduler attach timeout must be positive")
    if min(config.scheduler_cpu_tolerance_fraction, config.scheduler_cpu_tolerance_sec) < 0:
        raise ProfileError("scheduler CPU tolerances must not be negative")
    if min(config.proc_stat_interval_sec, config.proc_stat_baseline_sec) <= 0:
        raise ProfileError("proc-stat interval and baseline durations must be positive")
    if config.runtime != "current" and not ENTRYPOINT.is_file():
        raise ProfileError(f"container profiling entrypoint is missing: {ENTRYPOINT}")

    checks = [
        _preflight_check("profile_config", "passed"),
        _preflight_output(config.output_dir),
        _preflight_cpu(config),
    ]
    limitations: list[str] = []
    identity = _preflight_runtime(config, checks, limitations)

    scheduler = config.scheduler_concurrency or bool({"cpu-compare", "cpu-peak"} & set(config.profile_sets))
    if scheduler:
        if config.scheduler_backend == "bcc":
            if os.geteuid() != 0:
                raise ProfileError("BCC scheduler profiling requires root privileges")
            _require_bcc_bindings()
            checks.append(
                _preflight_check(
                    "scheduler_backend_static",
                    "passed",
                    backend="bcc",
                    root_privileges=True,
                    python_bcc_bindings=True,
                )
            )
        elif config.scheduler_backend == "perf":
            checks.append(
                _preflight_perf(config.scheduler_perf_binary, "scheduler_backend_static") | {"backend": "perf"}
            )
        message = "Scheduler attach, event loss, and cgroup CPU reconciliation remain pending execution."
        checks.append(_preflight_check("scheduler_dynamic_attach", "pending", reason=message))
        limitations.append(message)

    if "hardware-deep" in config.profile_sets:
        checks.append(_preflight_perf(config.hardware_perf_binary, "hardware_perf_static"))
        message = "Hardware perf attachment and counter quality remain pending execution."
        checks.append(_preflight_check("hardware_perf_dynamic_attach", "pending", reason=message))
        limitations.append(message)

    if config.memory_profile or bool({"cpu-compare", "memory-detail"} & set(config.profile_sets)):
        message = "Leaf/ancestor memory scope and memory.peak remain pending container creation."
        checks.append(_preflight_check("memory_cgroup_dynamic_scope", "pending", reason=message))
        limitations.append(message)

    return {
        "schema_version": PROFILE_PREFLIGHT_SCHEMA_VERSION,
        "ready": True,
        "runtime": config.runtime,
        "image_identity": identity,
        "profile_sets": list(config.profile_sets),
        "cpuset_cpus": config.cpuset_cpus,
        "allocated_cpus": config.allocated_cpus,
        "checks": checks,
        "limitations": limitations,
    }


def _container_init_pid(config: ProfileConfig, container_name: str) -> int:
    inspected = _run([config.docker, "inspect", "--format", "{{.State.Pid}}", container_name])
    if inspected.returncode != 0:
        raise ProfileError(f"unable to inspect running profiler container: {inspected.stderr.strip()}")
    try:
        pid = int(inspected.stdout.strip())
    except ValueError as exc:
        raise ProfileError(f"invalid container init PID reported for {container_name!r}") from exc
    if pid <= 0:
        raise ProfileError(f"profiler container is not running: {container_name}")
    return pid


def _host_cgroup_path_for_pid(pid: int) -> Path:
    for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            relative = line.split("::", 1)[1].lstrip("/")
            return Path("/sys/fs/cgroup") / relative
    raise ProfileError(f"cgroup v2 membership not found for container PID {pid}")


def _cgroup_subtree_pids(path: Path) -> list[int]:
    pids: set[int] = set()
    for procs_file in {path / "cgroup.procs", *path.rglob("cgroup.procs")}:
        try:
            pids.update(int(pid) for pid in procs_file.read_text().split() if pid.isdigit())
        except OSError:
            continue
    return sorted(pids)


def _process_snapshot(pid: int) -> dict[str, Any]:
    """Capture enough host /proc state to explain a finish-gate survivor."""
    snapshot: dict[str, Any] = {"pid": pid}
    proc_path = Path(f"/proc/{pid}")
    try:
        stat = (proc_path / "stat").read_text()
        stat_fields = stat.rsplit(")", 1)[-1].split()
        snapshot["state"] = stat_fields[0] if stat_fields else None
        snapshot["ppid"] = int(stat_fields[1]) if len(stat_fields) > 1 else None
        snapshot["start_time_ticks"] = int(stat_fields[19]) if len(stat_fields) > 19 else None
    except (OSError, ValueError):
        snapshot["state"] = None
        snapshot["ppid"] = None
        snapshot["start_time_ticks"] = None
    for name in ("comm", "wchan"):
        try:
            snapshot[name] = (proc_path / name).read_text().strip()
        except OSError:
            snapshot[name] = None
    try:
        snapshot["cmdline"] = (proc_path / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace").strip()
    except OSError:
        snapshot["cmdline"] = None
    try:
        snapshot["exe"] = str((proc_path / "exe").resolve(strict=True))
    except OSError:
        snapshot["exe"] = None
    return snapshot


def _signal_cgroup_descendants(
    path: Path,
    *,
    root_tid: int,
    sig: signal.Signals,
) -> list[int]:
    """Signal current descendants after cgroup and process-identity revalidation."""
    candidates = {
        pid: _process_snapshot(pid).get("start_time_ticks") for pid in _cgroup_subtree_pids(path) if pid != root_tid
    }
    signaled: list[int] = []
    for pid, start_time_ticks in candidates.items():
        if pid not in _cgroup_subtree_pids(path):
            continue
        current_start_time = _process_snapshot(pid).get("start_time_ticks")
        if start_time_ticks is None or current_start_time != start_time_ticks:
            continue
        try:
            os.kill(pid, sig)
            signaled.append(pid)
        except OSError:
            continue
    return signaled


def _wait_for_cgroup_descendants_to_drain(
    path: Path,
    *,
    root_tid: int,
    tracer: Any | None,
    proc_sampler: ProcStatComparisonSampler | None = None,
    resource_sampler: CgroupResourceSampler | None = None,
    timeout_sec: float = SCHEDULER_DESCENDANT_DRAIN_GRACE_SEC,
) -> tuple[list[int], float, list[dict[str, Any]], list[dict[str, Any]]]:
    """Give completed descendants a bounded grace period to leave the cgroup."""
    started = time.monotonic()
    initial_pids = [pid for pid in _cgroup_subtree_pids(path) if pid != root_tid]
    initial_processes = [_process_snapshot(pid) for pid in initial_pids]
    while True:
        remaining = [pid for pid in _cgroup_subtree_pids(path) if pid != root_tid]
        elapsed = time.monotonic() - started
        if not remaining or elapsed >= timeout_sec:
            return (
                remaining,
                elapsed,
                initial_processes,
                [_process_snapshot(pid) for pid in remaining],
            )
        remaining_sec = timeout_sec - elapsed
        poll_ms = max(1, min(25, int(remaining_sec * 1000)))
        if tracer is not None:
            tracer.poll(timeout_ms=poll_ms)
        else:
            time.sleep(poll_ms / 1000)
        if proc_sampler is not None:
            proc_sampler.poll()
        if resource_sampler is not None:
            resource_sampler.poll()


def _wait_for_artifact(
    path: Path,
    *,
    process: subprocess.Popen[str],
    timeout_sec: float | None,
    description: str,
    poll_callback: Callable[[], None] | None = None,
) -> None:
    deadline = time.monotonic() + timeout_sec if timeout_sec is not None else None
    while deadline is None or time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            _stdout, stderr = process.communicate()
            raise ProfileError(f"container exited before {description}: {stderr.strip()}")
        if poll_callback is not None:
            poll_callback()
        time.sleep(0.01)
    raise ProfileError(f"timed out waiting for {description}: {path}")


def _wait_for_blocked_init(
    pid: int,
    *,
    process: subprocess.Popen[str],
    timeout_sec: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            _stdout, stderr = process.communicate()
            raise ProfileError(f"container exited before {description}: {stderr.strip()}")
        try:
            status = Path(f"/proc/{pid}/stat").read_text()
        except OSError:
            time.sleep(0.01)
            continue
        fields = status.rsplit(")", 1)[-1].split()
        if fields and fields[0] in {"S", "D"}:
            return
        time.sleep(0.001)
    raise ProfileError(f"timed out waiting for container init to block during {description}")


def _release_gate(path: Path) -> None:
    with path.open("w") as gate:
        gate.write("continue\n")


def _build_scheduler_tracer(config: ProfileConfig) -> Any:
    if config.scheduler_backend == "bcc":
        return SchedulerConcurrencyTracer(buffer_pages=config.scheduler_bcc_buffer_pages)
    if config.scheduler_backend == "perf":
        return PerfSchedulerConcurrencyTracer(
            output_dir=config.output_dir,
            cpuset_cpus=config.cpuset_cpus,
            perf=config.scheduler_perf_binary,
            attach_timeout_sec=config.scheduler_attach_timeout_sec,
            mmap_pages=config.scheduler_perf_mmap_pages,
        )
    raise ProfileError(f"unsupported scheduler backend: {config.scheduler_backend!r}")


def _build_proc_stat_sampler(config: ProfileConfig) -> ProcStatComparisonSampler | None:
    if not config.proc_stat_comparison:
        return None
    if config.cpuset_cpus is None:
        raise ProfileError("--proc-stat-comparison requires --cpuset-cpus")
    return ProcStatComparisonSampler(
        cpu_ids=parse_cpuset_cpu_ids(config.cpuset_cpus),
        interval_sec=config.proc_stat_interval_sec,
        baseline_sec=config.proc_stat_baseline_sec,
    )


def _run_image_with_host_observers(
    config: ProfileConfig,
) -> tuple[
    subprocess.CompletedProcess[str],
    dict[str, Any] | None,
    dict[str, Any] | None,
    dict[str, Any] | None,
]:
    """Run one gated container with any requested host-side observers.

    One coordinator owns the gates so scheduler tracing, host ``/proc/stat``,
    and cgroup CPU/memory sampling share command boundaries and never launch
    duplicate containers. Memory profiles add an earlier gate before the
    prepare command, allowing a full setup-to-finish explanatory timeline.
    """
    container_name = f"tasktrace-profile-{uuid.uuid4().hex[:12]}"
    command = build_image_command(config, container_name=container_name)
    tracer = _build_scheduler_tracer(config) if config.scheduler_concurrency else None
    proc_sampler = _build_proc_stat_sampler(config)
    resource_sampler: CgroupResourceSampler | None = None
    process = subprocess.Popen(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    scheduler_report: dict[str, Any] | None = None
    proc_report: dict[str, Any] | None = None
    memory_report: dict[str, Any] | None = None
    root_tid: int | None = None
    cgroup_path: Path | None = None
    try:
        if config.memory_profile:
            _wait_for_artifact(
                config.output_dir / "sandbox_attach_ready",
                process=process,
                timeout_sec=config.scheduler_attach_timeout_sec,
                description="sandbox-lifecycle resource gate",
            )
            root_tid = _container_init_pid(config, container_name)
            _wait_for_blocked_init(
                root_tid,
                process=process,
                timeout_sec=config.scheduler_attach_timeout_sec,
                description="sandbox-lifecycle resource gate",
            )
            cgroup_path = _host_cgroup_path_for_pid(root_tid)
            resource_sampler = CgroupResourceSampler(
                cgroup_path=cgroup_path,
                interval_sec=config.memory_sample_interval_sec,
            )
            resource_sampler.start()
            _release_gate(config.output_dir / "sandbox.start.gate")

        _wait_for_artifact(
            config.output_dir / "scheduler_attach_ready",
            process=process,
            # The early memory gate makes preparation observable. Preserve the
            # normal no-timeout prepare semantics instead of misusing the
            # scheduler attach timeout as a setup timeout.
            timeout_sec=(None if config.memory_profile else config.scheduler_attach_timeout_sec),
            description="host-coordinated measurement gate",
            poll_callback=(resource_sampler.poll if resource_sampler is not None else None),
        )
        if root_tid is None:
            root_tid = _container_init_pid(config, container_name)
        _wait_for_blocked_init(
            root_tid,
            process=process,
            timeout_sec=config.scheduler_attach_timeout_sec,
            description="host-coordinated start gate",
        )
        if cgroup_path is None:
            cgroup_path = _host_cgroup_path_for_pid(root_tid)
        if proc_sampler is not None:
            proc_sampler.collect_baseline(
                poll_callback=(resource_sampler.poll if resource_sampler is not None else None),
                poll_interval_sec=(
                    min(config.memory_sample_interval_sec / 2, 0.025) if resource_sampler is not None else None
                ),
            )
        if tracer is not None:
            tracer.start(root_tid, cgroup_path=cgroup_path)
        window_start_ns: int | None = None
        if tracer is not None:
            (config.output_dir / "cpu.stat.before").write_text((cgroup_path / "cpu.stat").read_text())
            window_start_ns = tracer.timestamp_ns()
        if proc_sampler is not None:
            proc_sampler.start_measurement()
        if resource_sampler is not None:
            resource_sampler.mark("command_start")
        _release_gate(config.output_dir / "scheduler.start.gate")

        done = config.output_dir / "scheduler_measurement_done"
        deadline = time.monotonic() + config.scheduler_attach_timeout_sec
        while not done.exists():
            if tracer is not None:
                tracer.poll(timeout_ms=25)
            else:
                time.sleep(0.025)
            if proc_sampler is not None:
                proc_sampler.poll()
            if resource_sampler is not None:
                resource_sampler.poll()
            if process.poll() is not None:
                _stdout, stderr = process.communicate()
                raise ProfileError(
                    "container exited before host-coordinated measurement completed: " f"{stderr.strip()}"
                )
            if time.monotonic() >= deadline and not (config.output_dir / "start_ns").exists():
                raise ProfileError("timed out waiting for measured container command to start")

        _wait_for_blocked_init(
            root_tid,
            process=process,
            timeout_sec=config.scheduler_attach_timeout_sec,
            description="host-coordinated finish gate",
        )
        if tracer is not None:
            tracer.poll(timeout_ms=0)
        if resource_sampler is not None:
            resource_sampler.mark("command_end")
        if proc_sampler is not None and tracer is None:
            # The ordinary CPU contract ends when the foreground command
            # returns. Memory observation may continue while descendants drain,
            # but host-pressure context must remain on that command boundary.
            proc_report = proc_sampler.finish()

        signaled_term: list[int] = []
        signaled_kill: list[int] = []
        remaining_descendant_pids: list[int] = []
        initial_descendant_processes: list[dict[str, Any]] = []
        remaining_descendant_processes: list[dict[str, Any]] = []
        descendant_drain_wait_sec = 0.0
        if tracer is not None or resource_sampler is not None:
            if config.terminate_descendants_at_finish:
                initial_descendant_pids = [pid for pid in _cgroup_subtree_pids(cgroup_path) if pid != root_tid]
                initial_descendant_processes = [_process_snapshot(pid) for pid in initial_descendant_pids]
                signaled_term = _signal_cgroup_descendants(
                    cgroup_path,
                    root_tid=root_tid,
                    sig=signal.SIGTERM,
                )
                (
                    remaining_descendant_pids,
                    term_wait_sec,
                    _processes_before_term_wait,
                    remaining_descendant_processes,
                ) = _wait_for_cgroup_descendants_to_drain(
                    cgroup_path,
                    root_tid=root_tid,
                    tracer=tracer,
                    proc_sampler=(proc_sampler if tracer is not None else None),
                    resource_sampler=resource_sampler,
                )
                kill_wait_sec = 0.0
                if remaining_descendant_pids:
                    signaled_kill = _signal_cgroup_descendants(
                        cgroup_path,
                        root_tid=root_tid,
                        sig=signal.SIGKILL,
                    )
                    (
                        remaining_descendant_pids,
                        kill_wait_sec,
                        _processes_before_kill_wait,
                        remaining_descendant_processes,
                    ) = _wait_for_cgroup_descendants_to_drain(
                        cgroup_path,
                        root_tid=root_tid,
                        tracer=tracer,
                        proc_sampler=(proc_sampler if tracer is not None else None),
                        resource_sampler=resource_sampler,
                    )
                descendant_drain_wait_sec = term_wait_sec + kill_wait_sec
            else:
                (
                    remaining_descendant_pids,
                    descendant_drain_wait_sec,
                    initial_descendant_processes,
                    remaining_descendant_processes,
                ) = _wait_for_cgroup_descendants_to_drain(
                    cgroup_path,
                    root_tid=root_tid,
                    tracer=tracer,
                    proc_sampler=(proc_sampler if tracer is not None else None),
                    resource_sampler=resource_sampler,
                )

        if proc_sampler is not None and proc_report is None:
            proc_report = proc_sampler.finish()
        window_end_ns: int | None = None
        if tracer is not None:
            (config.output_dir / "cpu.stat.after").write_text((cgroup_path / "cpu.stat").read_text())
            window_end_ns = tracer.timestamp_ns()
            # Scheduler reports deliberately include the descendant drain, so
            # the host owns and overwrites both matching CPU/wall boundaries.
            assert window_start_ns is not None
            (config.output_dir / "start_ns").write_text(f"{window_start_ns}\n")
            (config.output_dir / "end_ns").write_text(f"{window_end_ns}\n")

        if tracer is not None:
            assert window_start_ns is not None and window_end_ns is not None
            scheduler_report = tracer.finish(
                window_start_ns=window_start_ns,
                window_end_ns=window_end_ns,
            )
            scheduler_report["descendant_finish_policy"] = (
                "terminate-at-trial-end" if config.terminate_descendants_at_finish else "require-natural-exit"
            )
            scheduler_report["descendants_signaled_at_finish"] = {
                "sigterm": signaled_term,
                "sigkill": signaled_kill,
            }
            scheduler_report["descendant_drain_grace_sec"] = SCHEDULER_DESCENDANT_DRAIN_GRACE_SEC
            scheduler_report["descendant_drain_wait_sec"] = round(descendant_drain_wait_sec, 9)
            scheduler_report["descendant_processes_at_finish_gate"] = initial_descendant_processes
            scheduler_report["remaining_descendant_processes_at_finish_gate"] = remaining_descendant_processes
            scheduler_report["remaining_descendant_pids_at_finish_gate"] = remaining_descendant_pids
            if remaining_descendant_pids:
                boundary = (
                    "after trial-end cleanup"
                    if config.terminate_descendants_at_finish
                    else "after the command returned"
                )
                scheduler_report["anomalies"].append(
                    f"descendant PIDs remained in the container cgroup {boundary}: "
                    + ", ".join(str(pid) for pid in remaining_descendant_pids)
                )
                scheduler_report["trace_complete"] = False
            tracer.write_events(config.output_dir / "scheduler_events.jsonl")
        if proc_sampler is not None:
            proc_sampler.write_samples(config.output_dir / "proc_stat_samples.json")
        if resource_sampler is not None:
            memory_report = resource_sampler.finish()
            samples_path = config.output_dir / "resource_timeseries.json"
            resource_sampler.write_timeseries(samples_path)
            memory_report["samples_path"] = str(samples_path.resolve())
            memory_report["timeline_scope"] = (
                "From the pre-prepare lifecycle gate through command completion "
                "and bounded descendant drain. Container startup before observer "
                "attachment is retained by the fresh-cgroup memory.peak."
            )
            memory_report["scope_coverage"] = {
                "kind": "disposable-container-cgroup-subtree",
                "includes": [
                    "container init and profiling entrypoint",
                    "prepare command",
                    "measured command and descendants",
                    "cgroup-charged anonymous, file-cache, and kernel memory",
                ],
                "excludes": [
                    "final finish-gate FIFO unblock and shell/container-init exit",
                    "host profiler process",
                    "Docker daemon and container runtime processes",
                    "sibling or sidecar containers outside this cgroup",
                    "host page cache not charged to this cgroup",
                ],
                "complete_for_profiled_container": False,
                "complete_through_controlled_finish_gate": True,
                "container_init_exit_observed": False,
                "full_production_sandbox_requires_all_components_in_scope": True,
            }
            memory_report["descendant_finish_policy"] = (
                "terminate-at-trial-end" if config.terminate_descendants_at_finish else "require-natural-exit"
            )
            memory_report["remaining_descendant_pids_at_finish_gate"] = remaining_descendant_pids
            if remaining_descendant_pids:
                reason = "descendant PIDs remained at the end of the bounded " "sandbox observation: " + ", ".join(
                    str(pid) for pid in remaining_descendant_pids
                )
                memory_report["observation_boundary_complete"] = False
                memory_report["natural_peak_trustworthy"] = False
                memory_report["natural_peak_invalid_reasons"].append(reason)

        _release_gate(config.output_dir / "scheduler.finish.gate")
        stdout, stderr = process.communicate()
        return (
            subprocess.CompletedProcess(command, process.returncode, stdout, stderr),
            scheduler_report,
            proc_report,
            memory_report,
        )
    except (
        CgroupResourceSamplingError,
        ProcStatSamplingError,
        SchedulerTraceError,
    ) as exc:
        raise ProfileError(str(exc)) from exc
    finally:
        if tracer is not None:
            tracer.close()
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
            _run([config.docker, "rm", "-f", container_name])


def _format_float(value: Any, digits: int) -> str:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return f"{value:.{digits}f}"
    return "n/a"


def cpu_summary_lines(cpu: Mapping[str, Any]) -> list[str]:
    """Human-readable CPU lines that also handle wall-only Enroot reports."""
    lines = [f"Wall time: {_format_float(cpu.get('wall_sec'), 9)} sec"]
    if cpu.get("usage_sec") is None:
        lines.append("CPU time: n/a (cgroup cpu.stat unavailable)")
        lines.append("Average logical cores used: n/a (cgroup cpu.stat unavailable)")
    else:
        lines.append(f"CPU time: {_format_float(cpu.get('usage_sec'), 9)} sec")
        lines.append(f"Average logical cores used: {_format_float(cpu.get('average_logical_cores_used'), 6)}")
    if cpu.get("utilization_percent_of_allocation") is not None:
        lines.append(
            "Allocated CPU utilization: "
            f"{_format_float(cpu.get('utilization_percent_of_allocation'), 3)}% "
            f"of {cpu['allocated_logical_cpus']:g} logical CPUs"
        )
    return lines


def memory_oom_kill_observed(memory: Mapping[str, Any] | None) -> bool:
    """Return whether the measured cgroup reported a killed task."""
    if not isinstance(memory, Mapping):
        return False
    events = memory.get("events_delta")
    if not isinstance(events, Mapping):
        return False
    return any(
        isinstance(events.get(key), int) and not isinstance(events.get(key), bool) and events[key] > 0
        for key in ("oom_kill", "oom_group_kill")
    )


def _apply_memory_workload_outcome(memory_report: dict[str, Any], *, command_exit_code: int) -> None:
    """Attach workload outcome semantics before any report or bundle is written."""
    if command_exit_code == 0 or memory_report.get("censoring") != "none":
        return
    memory_report["censoring"] = "functional_failure"
    memory_report["natural_peak_trustworthy"] = False
    reason = "measured command exited nonzero; natural successful-task demand is unknown"
    reasons = memory_report.setdefault("natural_peak_invalid_reasons", [])
    if isinstance(reasons, list) and reason not in reasons:
        reasons.append(reason)


_POLICY_PROXY_ENV_KEYS = frozenset(
    {
        "ALL_PROXY",
        "HTTP_PROXY",
        "HTTPS_PROXY",
        "NO_PROXY",
        "all_proxy",
        "http_proxy",
        "https_proxy",
        "no_proxy",
    }
)


def _apply_network_policy_binding(
    config: ProfileConfig,
    *,
    policy: NetworkPolicy,
    docker_network: str | None,
    proxy_environment: Mapping[str, str],
    enforcement: Mapping[str, Any],
) -> ProfileConfig:
    """Return an execution config with only runtime policy details added."""
    retained_environment = tuple(
        assignment for assignment in config.env if assignment.partition("=")[0] not in _POLICY_PROXY_ENV_KEYS
    )
    policy_environment = tuple(f"{key}={value}" for key, value in proxy_environment.items())
    return replace(
        config,
        env=retained_environment + policy_environment,
        network_policy=policy,
        network_attachment=docker_network,
        network_policy_enforcement=dict(enforcement),
    )


def _retain_enroot_network_evidence(
    config: ProfileConfig,
    private_path: Path,
) -> dict[str, Any]:
    """Validate private launch evidence and publish it after Enroot exits."""
    private = private_path.resolve()
    if private.is_relative_to(config.output_dir.resolve()):
        raise ProfileError("Enroot network evidence path is exposed by the output mount")
    for mount in config.mounts:
        source, _target, _readonly = _parse_docker_bind_mount(mount)
        if private.is_relative_to(Path(source).expanduser().resolve(strict=False)):
            raise ProfileError("Enroot network evidence path is exposed by a workload mount")
    try:
        payload = json.loads(private.read_text())
        if not isinstance(payload, Mapping):
            raise RuntimeError("evidence is not an object")
        parent_inode = Path("/proc/self/ns/net").stat().st_ino
        validated = validate_evidence(
            payload,
            parent_netns_inode=parent_inode,
            controller_euid=os.geteuid(),
            controller_egid=os.getegid(),
        )
        retained = {
            **validated,
            "publication": {
                "source": "controller-private-temporary-state",
                "published_after_enroot_exit": True,
            },
        }
        path = config.output_dir / EVIDENCE_FILENAME
        temporary = path.with_name(f".{path.name}.tmp")
        temporary.write_text(json.dumps(retained, indent=2, sort_keys=True) + "\n")
        temporary.replace(path)
        if json.loads(path.read_text()) != retained:
            raise RuntimeError("published evidence differs from validated evidence")
    except (OSError, json.JSONDecodeError, RuntimeError) as exc:
        raise ProfileError(f"invalid Enroot no-network evidence: {exc}") from exc
    retained["artifact_path"] = str(path.resolve())
    return retained


def run_profile(
    config: ProfileConfig,
    *,
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Execute one profile and write ``profile_report.json``.

    Cooperative cancellation is available only for coordinated Docker
    profiles, where it enters the normal fail-closed container cleanup path.
    """
    validate_profile_config(config)
    coordinated = bool(config.profile_sets) or config.coordinated_baseline
    if cancel_requested is not None and (config.runtime != "docker" or not coordinated):
        raise ProfileError("cooperative cancellation is supported only for coordinated Docker profiles")
    if config.runtime == "current":
        from .current import run_current_profile

        return run_current_profile(config)
    profile_run_id = str(uuid.uuid4())
    profile_started_at_utc = datetime.now(timezone.utc).isoformat()
    controller_digest = (
        controller_source_sha256()
        if config.memory_profile or bool({"cpu-compare", "memory-detail"} & set(config.profile_sets))
        else None
    )
    if not ENTRYPOINT.exists():
        raise ProfileError(f"container profiling entrypoint is missing: {ENTRYPOINT}")
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ProfileError(f"profile output directory must be new or empty: {config.output_dir}")
    config.output_dir.mkdir(parents=True, exist_ok=True)
    coordinated_result: Any | None = None
    report_config = config
    hardware_report: dict[str, Any] | None = None
    if config.image is not None:
        # Capture identity before the run so a later retag cannot make the
        # result appear to have used a different benchmark environment. Docker
        # executes the resolved image ID below, closing the inspect/run tag race.
        image_identity = _inspect_image_identity(config)
        execution_config = _immutable_execution_config(config, image_identity)
        scheduler_report: dict[str, Any] | None = None
        proc_stat_report: dict[str, Any] | None = None
        memory_report: dict[str, Any] | None = None
        network_session: Any | None = None
        execution_error: BaseException | None = None
        try:
            if config.runtime == "docker":
                policy = resolved_network_policy(execution_config)
                if policy is not None:
                    from .network_enforcement import DockerNetworkPolicySession

                    network_session = DockerNetworkPolicySession(
                        policy,
                        config.output_dir,
                        docker=config.docker,
                        proxy_image=config.network_policy_proxy_image,
                        external_network=config.network_policy_external_network,
                        workload_mount_sources=tuple(_parse_docker_bind_mount(mount)[0] for mount in config.mounts),
                    )
                    network_session.__enter__()
                    execution_config = _apply_network_policy_binding(
                        execution_config,
                        policy=policy,
                        docker_network=network_session.docker_network,
                        proxy_environment=network_session.proxy_environment,
                        enforcement=network_session.report_payload(),
                    )
                report_config = replace(execution_config, image=config.image)
                if config.profile_sets or config.coordinated_baseline:
                    # The coordinator owns exactly one Docker session and exposes
                    # its cgroup scope, gates, and canonical lifecycle to every
                    # selected profiler.  Import lazily to avoid a module cycle:
                    # the bridge deliberately reuses the stable container helper
                    # functions above rather than implementing another runner.
                    from .container_coordinator import run_coordinated_docker_profile

                    coordinated_result = run_coordinated_docker_profile(
                        execution_config,
                        image_identity=image_identity,
                        run_id=profile_run_id,
                        cancel_requested=cancel_requested,
                    )
                    completed = coordinated_result.completed
                    scheduler_report = coordinated_result.scheduler_report
                    memory_report = coordinated_result.memory_report
                    hardware_report = coordinated_result.hardware_report
                    report_config = replace(
                        coordinated_result.effective_config,
                        image=config.image,
                    )
                elif config.scheduler_concurrency or config.proc_stat_comparison or config.memory_profile:
                    (
                        completed,
                        scheduler_report,
                        proc_stat_report,
                        memory_report,
                    ) = _run_image_with_host_observers(execution_config)
                else:
                    completed = _run(build_image_command(execution_config))
            else:
                if config.scheduler_concurrency:
                    raise ProfileError("--scheduler-concurrency is currently Docker-only")
                if config.proc_stat_comparison:
                    raise ProfileError("--proc-stat-comparison is currently Docker-only")
                if config.memory_profile:
                    raise ProfileError("--memory-profile is currently Docker-only")
                policy = resolved_network_policy(execution_config)
                report_config = replace(execution_config, image=config.image)
                if policy is not None and policy.effective_mode is NetworkMode.NO_NETWORK:
                    with tempfile.TemporaryDirectory(
                        prefix="tasktrace-enroot-network-",
                        dir="/tmp",
                    ) as private_directory:
                        private_path = Path(private_directory) / "evidence.json"
                        completed = _run_enroot_image(
                            execution_config,
                            network_evidence_path=private_path,
                        )
                        enforcement = _retain_enroot_network_evidence(
                            execution_config,
                            private_path,
                        )
                    report_config = replace(
                        report_config,
                        network_policy_enforcement=enforcement,
                    )
                else:
                    completed = _run_enroot_image(execution_config)
        except BaseException as exc:
            execution_error = exc
            raise
        finally:
            if network_session is not None:
                cleanup_errors = network_session.close()
                report_config = replace(
                    report_config,
                    network_policy_enforcement=network_session.report_payload(),
                )
                if cleanup_errors:
                    cleanup_error = ProfileError("Docker network-policy cleanup failed: " + "; ".join(cleanup_errors))
                    if execution_error is not None:
                        raise cleanup_error from execution_error
                    raise cleanup_error
        scope = "disposable-image-container"
    else:
        if config.runtime != "docker":
            raise ProfileError("--container mode is currently Docker-only")
        if config.scheduler_concurrency:
            raise ProfileError("--scheduler-concurrency supports disposable --image mode only")
        if config.proc_stat_comparison:
            raise ProfileError("--proc-stat-comparison supports disposable --image mode only")
        if config.memory_profile:
            raise ProfileError("--memory-profile supports disposable --image mode only")
        if config.memory_limit_bytes is not None:
            raise ProfileError("--memory-limit-bytes supports disposable --image mode only")
        if config.memory_swap_total_limit_bytes is not None:
            raise ProfileError("--memory-swap-total-limit-bytes supports disposable --image mode only")
        scheduler_report = None
        proc_stat_report = None
        memory_report = None
        image_identity = None
        assert config.container is not None
        # Existing-container mode copies only the harness; it must not rebuild,
        # restart, or reconfigure a container the caller may still be using.
        remote_base = f"/tmp/tasktrace-profile-{uuid.uuid4().hex[:12]}"
        remote_entrypoint = f"{remote_base}/entrypoint.sh"
        try:
            prepared = _run([config.docker, "exec", config.container, "/bin/mkdir", "-p", remote_base])
            if prepared.returncode != 0:
                raise ProfileError("unable to prepare profiler directory in container: " f"{prepared.stderr.strip()}")
            copy_script = _run([config.docker, "cp", str(ENTRYPOINT), f"{config.container}:{remote_entrypoint}"])
            if copy_script.returncode != 0:
                raise ProfileError(f"unable to copy profiler into container: {copy_script.stderr.strip()}")
            completed = _run(build_existing_container_exec_command(config, remote_base, remote_entrypoint))
            copied = _run([config.docker, "cp", f"{config.container}:{remote_base}/.", str(config.output_dir)])
            if copied.returncode != 0:
                raise ProfileError(f"unable to retrieve profiler output from container: {copied.stderr.strip()}")
        finally:
            _run([config.docker, "exec", config.container, "/bin/rm", "-rf", remote_base])
        scope = "existing-container"
    try:
        report = build_report(
            report_config,
            scope=scope,
            docker_returncode=completed.returncode,
            image_identity=image_identity,
            run_id=profile_run_id,
            profile_started_at_utc=profile_started_at_utc,
            coordinated_cpu_report=(coordinated_result.cpu_report if coordinated_result is not None else None),
        )
    except ProfileError as exc:
        diagnostic = completed.stderr.strip() or completed.stdout.strip()
        if diagnostic:
            raise ProfileError(f"{exc}; container diagnostic: {diagnostic}") from exc
        raise
    if scheduler_report is not None:
        if coordinated_result is not None:
            # The wrapper validates the scheduler integral against the shared
            # cgroup CPU window when tracing is available.  Unsupported BCC or
            # perf environments remain explicit invalid evidence here.
            report["scheduler_concurrency"] = scheduler_report
        else:
            report["scheduler_concurrency"] = add_cgroup_validation(
                scheduler_report,
                cgroup_usage_sec=float(report["cpu"]["usage_sec"]),
                tolerance_fraction=config.scheduler_cpu_tolerance_fraction,
                tolerance_sec=config.scheduler_cpu_tolerance_sec,
                per_observed_event_tolerance_sec=(
                    BCC_SCHEDULER_EVENT_BOUNDARY_TOLERANCE_SEC if config.scheduler_backend == "bcc" else 0.0
                ),
            )
            report["scheduler_concurrency"]["events_path"] = str(config.output_dir / "scheduler_events.jsonl")
    if proc_stat_report is not None:
        scheduler = report.get("scheduler_concurrency")
        scheduler_validation = scheduler["cgroup_validation"] if scheduler else None
        report["proc_stat_comparison"] = add_reference_comparison(
            proc_stat_report,
            cgroup_average_logical_cores=report["cpu"]["average_logical_cores_used"],
            scheduler_peak_simultaneous_logical_cores=(
                scheduler["peak_simultaneous_logical_cores"] if scheduler else None
            ),
            scheduler_peak_valid=(
                scheduler_validation["valid_for_concurrency_reporting"] if scheduler_validation else False
            ),
        )
        report["proc_stat_comparison"]["samples_path"] = str(config.output_dir / "proc_stat_samples.json")
    if memory_report is not None:
        assert controller_digest is not None
        memory_report["profile_run_id"] = profile_run_id
        memory_report["profile_started_at_utc"] = profile_started_at_utc
        memory_report["controller_source_sha256"] = controller_digest
        memory_report["execution_provenance"] = _memory_execution_provenance(report_config, image_identity)
        if coordinated_result is None and report_config.strace_mode != "none":
            memory_report["natural_peak_trustworthy"] = False
            reason = (
                "strace ran inside the measured cgroup and changed the sandbox "
                "memory footprint; rerun with --strace-mode none to measure "
                "an unperturbed natural peak"
            )
            if reason not in memory_report["natural_peak_invalid_reasons"]:
                memory_report["natural_peak_invalid_reasons"].append(reason)
        _apply_memory_workload_outcome(memory_report, command_exit_code=report["command_exit_code"])
        report["memory"] = memory_report
    if hardware_report is not None:
        report["hardware_perf"] = hardware_report
    if coordinated_result is not None:
        coordination = coordinated_result.coordination
        bundle_metadata = coordination.evidence_bundle.to_dict()
        verification = bundle_metadata["extensions"].get("workload_verification")
        if isinstance(verification, dict):
            report["verification"] = verification
        report["canonical_clock_reference"] = coordinated_result.canonical_clock_reference
        report["canonical_lifecycle"] = coordinated_result.canonical_lifecycle
        report["evidence_bundle"] = {
            "path": str(coordinated_result.bundle_path),
            "sha256": coordinated_result.bundle_sha256,
            "run_id": coordination.run_id,
            "scope_id": coordination.scope.scope_id,
            "profile_sets": list(config.profile_sets),
            "mode": ("baseline" if config.coordinated_baseline else "profile"),
            "coordination_status": coordination.status,
            "execution_plan_digest": coordination.execution_plan_digest,
            "scope": bundle_metadata["scope"],
            "timeline": bundle_metadata["timeline"],
            "provenance": bundle_metadata["provenance"],
            "evidence_status": bundle_metadata["extensions"].get("evidence_status"),
            "all_selected_profilers_valid": bundle_metadata["extensions"].get("all_selected_profilers_valid"),
            "workload_verification_valid": bundle_metadata["extensions"].get("workload_verification_valid"),
            "profilers": bundle_metadata["profilers"],
            "resolved_profiler_ids": list(coordination.resolved_profiler_ids),
            "intrusive_profiler_ids": list(coordination.intrusive_profiler_ids),
            "profiler_manifests": [
                {
                    "profiler_id": manifest.profiler_id,
                    "profiler_version": manifest.profiler_version,
                    "status": manifest.status,
                    "stream_ids": list(manifest.stream_ids),
                    "reasons": list(manifest.reasons),
                    "intrusive": manifest.intrusive,
                }
                for manifest in coordination.manifests
            ],
        }
        invalid_manifests = [
            manifest for manifest in coordination.manifests if manifest.status in {"invalid", "missing", "dropped"}
        ]
        verification_failed = (
            isinstance(verification, dict)
            and verification.get("requested") is True
            and verification.get("status") != "passed"
        )
        report["profiling_valid"] = not invalid_manifests and not verification_failed
        report["profiling_invalid_reasons"] = [
            {
                "profiler_id": manifest.profiler_id,
                "status": manifest.status,
                "reasons": list(manifest.reasons),
            }
            for manifest in invalid_manifests
        ]
        if verification_failed:
            report["profiling_invalid_reasons"].append(
                {
                    "profiler_id": "workload-verification",
                    "status": "invalid",
                    "reasons": list(verification.get("reasons") or []),
                }
            )
        if config.render_unified_view:
            report["unified_profile_artifacts"] = {
                "evidence_bundle": str(coordinated_result.bundle_path.resolve()),
                "view_model": str((config.output_dir / "view_model.json").resolve()),
                "html": str((config.output_dir / "unified_profile.html").resolve()),
                "svg": str((config.output_dir / "unified_profile.svg").resolve()),
                "animated_svg": str((config.output_dir / "unified_profile.animated.svg").resolve()),
                "manifest": str((config.output_dir / "unified_profile_artifacts.json").resolve()),
                "workload_executions": 1,
                "generation_method": "typed-evidence-bundle-render",
            }
    (config.output_dir / "profile_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if coordinated_result is not None and config.render_unified_view:
        from ..reporting.unified_view import write_unified_view_from_bundle

        try:
            rendered = write_unified_view_from_bundle(coordinated_result.bundle_path, config.output_dir)
        except (OSError, ValueError) as exc:
            raise ProfileError("profile completed but unified view rendering failed: " f"{exc}") from exc
        artifact_manifest = {
            "schema_version": "tasktrace-unified-profile-artifacts.v1",
            "run_id": profile_run_id,
            "workload_executions": 1,
            "evidence_bundle": str(coordinated_result.bundle_path.resolve()),
            "evidence_bundle_sha256": coordinated_result.bundle_sha256,
            "profile_report_sha256": hashlib.sha256(
                (config.output_dir / "profile_report.json").read_bytes()
            ).hexdigest(),
            **rendered,
        }
        (config.output_dir / "unified_profile_artifacts.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
        )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Profile one containerized shell command using strace and cgroup CPU statistics."
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--image", help="Image to start as a disposable profiling container.")
    target.add_argument("--container", help="Already-running Docker container to execute inside.")
    parser.add_argument("--runtime", choices=CONTAINER_RUNTIMES, default="docker")
    parser.add_argument("--command", required=True, help="Shell command to execute in the container.")
    parser.add_argument(
        "--prepare-command",
        help=(
            "Optional shell command executed in the disposable/running container before "
            "the measured window, for example installing strace when the image lacks it."
        ),
    )
    parser.add_argument(
        "--verification-command",
        help=(
            "Command run in the same disposable sandbox after the measured "
            "window and profiler stop; nonzero status invalidates the run."
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Host directory for trace and report artifacts.")
    parser.add_argument(
        "--cpuset-cpus",
        help="Logical CPU set for --image mode, for example 0-7. Provides both isolation and utilization capacity.",
    )
    parser.add_argument(
        "--allocated-cpus",
        type=float,
        help="CPU capacity denominator when it cannot be inferred from --cpuset-cpus.",
    )
    parser.add_argument(
        "--cpu-limit",
        type=float,
        help=(
            "Docker CPU quota applied in --image mode, for example 2 to match "
            "a benchmark task declaring cpus = 2 without changing nproc affinity."
        ),
    )
    parser.add_argument("--workdir", help="Working directory for the executed container command.")
    parser.add_argument("--user", help="Docker user for the container command.")
    parser.add_argument(
        "--network",
        help=(
            "Deprecated raw Docker network mode. Use --network-policy; "
            "--network none remains a compatible alias for no-network."
        ),
    )
    parser.add_argument(
        "--network-policy",
        choices=tuple(mode.value for mode in NetworkMode),
        help="Portable replay policy: public, no-network, or allowlist.",
    )
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help=("Exact DNS hostname permitted by allowlist policy; repeatable. " "An empty allowlist denies all egress."),
    )
    parser.add_argument(
        "--network-policy-proxy-image",
        default=DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
        help=(
            "Pre-pulled Python image used by the TaskTrace allowlist gateway; "
            "the local image is resolved to an immutable ID and never auto-pulled."
        ),
    )
    parser.add_argument(
        "--network-policy-external-network",
        default="bridge",
        help="Docker network used only by the allowlist proxy for outbound access.",
    )
    parser.add_argument(
        "--shell",
        default="/bin/bash",
        help="Non-login shell used for the command inside the container (default: /bin/bash).",
    )
    parser.add_argument(
        "--env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Additional environment setting passed to the target container; repeatable.",
    )
    parser.add_argument(
        "--mount",
        action="append",
        default=[],
        help="Additional bind mount specification in --image mode; repeatable.",
    )
    parser.add_argument(
        "--strace-mode",
        choices=STRACE_MODES,
        default="execve",
        help="Trace only executable launches, all process syscalls, or disable tracing for a baseline.",
    )
    profiler_mode = parser.add_mutually_exclusive_group()
    profiler_mode.add_argument(
        "--coordinated-baseline",
        action="store_true",
        help=(
            "Run an unprofiled baseline through the same coordinator, sandbox "
            "gates, lifecycle, and execution plan used by --profile-set."
        ),
    )
    profiler_mode.add_argument(
        "--profile-set",
        action="append",
        choices=PROFILE_SETS,
        default=[],
        help="Coordinated profiler preset; repeat to combine independent profilers.",
    )
    parser.add_argument(
        "--hardware-perf-binary",
        default="perf",
        help="Perf executable used by the hardware-deep profile set.",
    )
    parser.add_argument(
        "--hardware-perf-event",
        action="append",
        help=(
            "Perf event requested by hardware-deep; repeatable. Defaults to the "
            "portable cycles/instructions/branch/cache set."
        ),
    )
    parser.add_argument(
        "--hardware-perf-attach-timeout-sec",
        type=float,
        default=20.0,
        help="Seconds to wait for perf control-pipe acknowledgements.",
    )
    parser.add_argument(
        "--hardware-perf-minimum-running-ratio",
        type=float,
        default=0.95,
        help="Minimum time_running/time_enabled accepted for each counter.",
    )
    parser.add_argument(
        "--strace-string-limit",
        type=int,
        default=4096,
        help="Maximum string bytes rendered per strace argument.",
    )
    parser.add_argument(
        "--scheduler-concurrency",
        action="store_true",
        help=("Use sched_switch tracing to report simultaneous logical-CPU " "occupancy for disposable image runs."),
    )
    parser.add_argument(
        "--terminate-descendants-at-finish",
        action="store_true",
        help=(
            "For a closed trial, terminate descendants still in the measured "
            "container cgroup while scheduler tracing remains active."
        ),
    )
    parser.add_argument(
        "--scheduler-backend",
        choices=SCHEDULER_BACKENDS,
        default="bcc",
        help=(
            "Scheduler event collection backend. bcc loads a scoped eBPF "
            "program; perf records native cgroup-scoped context switches."
        ),
    )
    parser.add_argument(
        "--scheduler-bcc-buffer-pages",
        type=int,
        help=(
            "Optional positive power-of-two page count for each BCC perf buffer. "
            "Increase this for scheduler-event-heavy workloads."
        ),
    )
    parser.add_argument(
        "--scheduler-perf-binary",
        default="perf",
        help="Perf executable used when --scheduler-backend perf is selected.",
    )
    parser.add_argument(
        "--scheduler-perf-mmap-pages",
        type=int,
        help="Optional perf ring-buffer pages used by the perf backend.",
    )
    parser.add_argument(
        "--proc-stat-comparison",
        action="store_true",
        help=(
            "Collect TaskTrace-style sampled /proc/stat host-pressure context on "
            "the allocated CPUs during the measured command window. This does "
            "not establish task peak concurrency."
        ),
    )
    parser.add_argument(
        "--proc-stat-interval-sec",
        type=float,
        default=0.5,
        help="Sampling interval for --proc-stat-comparison (default: 0.5 seconds).",
    )
    parser.add_argument(
        "--proc-stat-baseline-sec",
        type=float,
        default=5.0,
        help="Pre-command baseline duration for --proc-stat-comparison (default: 5 seconds).",
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help=(
            "Collect a task-cgroup memory timeline plus same-clock sampled CPU "
            "demand and the exact observed fresh-cgroup memory.peak."
        ),
    )
    parser.add_argument(
        "--memory-sample-interval-sec",
        type=float,
        default=0.1,
        help="Cgroup CPU/memory sampling interval (default: 0.1 seconds).",
    )
    parser.add_argument(
        "--memory-limit-bytes",
        type=int,
        help="Optional Docker memory limit in bytes for finite-limit profiling experiments.",
    )
    parser.add_argument(
        "--memory-swap-total-limit-bytes",
        type=int,
        help=(
            "Optional Docker --memory-swap total (memory plus swap) limit in " "bytes; requires --memory-limit-bytes."
        ),
    )
    parser.add_argument(
        "--scheduler-attach-timeout-sec",
        type=float,
        default=20.0,
        help="Seconds to wait for the gated container/tracer handshake.",
    )
    parser.add_argument(
        "--scheduler-cpu-tolerance-fraction",
        type=float,
        default=0.02,
        help="Allowed fractional scheduler-versus-cgroup CPU delta.",
    )
    parser.add_argument(
        "--scheduler-cpu-tolerance-sec",
        type=float,
        default=0.005,
        help="Allowed absolute scheduler-versus-cgroup CPU delta in seconds.",
    )
    parser.add_argument("--docker", default="docker", help="Docker CLI executable.")
    parser.add_argument("--enroot", default="enroot", help="Enroot CLI executable.")
    args = parser.parse_args(argv)
    try:
        network_policy = resolve_cli_network_policy(
            mode=args.network_policy,
            allowed_hosts=args.allowed_host,
            legacy_network=args.network,
        )
    except NetworkPolicyError as exc:
        parser.error(str(exc))
    if args.runtime == "enroot" and args.container:
        parser.error("--container mode is currently Docker-only")
    if args.container and args.cpuset_cpus:
        parser.error("--cpuset-cpus applies only to disposable --image mode")
    if args.container and args.cpu_limit is not None:
        parser.error("--cpu-limit applies only to disposable --image mode")
    if args.container and args.memory_limit_bytes is not None:
        parser.error("--memory-limit-bytes applies only to disposable --image mode")
    if args.container and args.memory_swap_total_limit_bytes is not None:
        parser.error("--memory-swap-total-limit-bytes applies only to disposable --image mode")
    if args.image and not args.cpuset_cpus and args.allocated_cpus is None:
        parser.error("--image requires --cpuset-cpus or --allocated-cpus for normalized utilization")
    if args.container and (
        args.network or (network_policy is not None and network_policy.effective_mode is not NetworkMode.PUBLIC)
    ):
        parser.error("non-public network policy applies only to disposable --image mode")
    if args.container and args.mount:
        parser.error("--mount applies only to disposable --image mode")
    if args.container and args.scheduler_concurrency:
        parser.error("--scheduler-concurrency applies only to disposable --image mode")
    if args.container and args.profile_set:
        parser.error("--profile-set applies only to disposable --image mode")
    if args.terminate_descendants_at_finish and not (
        args.scheduler_concurrency or args.memory_profile or args.profile_set
    ):
        parser.error("--terminate-descendants-at-finish requires --scheduler-concurrency " "or --memory-profile")
    if args.container and args.proc_stat_comparison:
        parser.error("--proc-stat-comparison applies only to disposable --image mode")
    if args.container and args.memory_profile:
        parser.error("--memory-profile applies only to disposable --image mode")
    if args.runtime == "enroot" and args.cpu_limit is not None:
        parser.error("--cpu-limit is Docker-only; use taskset affinity for Enroot")
    if args.runtime == "enroot" and (
        args.network not in (None, "none")
        or (network_policy is not None and network_policy.mode is NetworkMode.ALLOWLIST)
    ):
        parser.error("Enroot supports only public or no-network policy")
    if args.runtime == "enroot" and args.scheduler_concurrency:
        parser.error("--scheduler-concurrency is currently Docker-only")
    if args.runtime == "enroot" and args.proc_stat_comparison:
        parser.error("--proc-stat-comparison is currently Docker-only")
    if args.runtime == "enroot" and args.memory_profile:
        parser.error("--memory-profile is currently Docker-only")
    if args.runtime == "enroot" and args.memory_limit_bytes is not None:
        parser.error("--memory-limit-bytes is currently Docker-only")
    if args.runtime == "enroot" and args.memory_swap_total_limit_bytes is not None:
        parser.error("--memory-swap-total-limit-bytes is currently Docker-only")
    if args.runtime == "enroot" and args.profile_set:
        parser.error("--profile-set currently requires the Docker runtime")
    if (
        args.scheduler_backend != "bcc"
        and not args.scheduler_concurrency
        and not {"cpu-compare", "cpu-peak"} & set(args.profile_set)
    ):
        parser.error("--scheduler-backend requires --scheduler-concurrency")
    if args.proc_stat_comparison and not args.cpuset_cpus:
        parser.error("--proc-stat-comparison requires --cpuset-cpus")
    if args.allocated_cpus is not None and args.allocated_cpus <= 0:
        parser.error("--allocated-cpus must be positive")
    if args.cpu_limit is not None and args.cpu_limit <= 0:
        parser.error("--cpu-limit must be positive")
    if args.memory_limit_bytes is not None and args.memory_limit_bytes <= 0:
        parser.error("--memory-limit-bytes must be positive")
    if args.memory_swap_total_limit_bytes is not None and args.memory_swap_total_limit_bytes <= 0:
        parser.error("--memory-swap-total-limit-bytes must be positive")
    if args.memory_swap_total_limit_bytes is not None and args.memory_limit_bytes is None:
        parser.error("--memory-swap-total-limit-bytes requires --memory-limit-bytes")
    if (
        args.memory_swap_total_limit_bytes is not None
        and args.memory_limit_bytes is not None
        and args.memory_swap_total_limit_bytes < args.memory_limit_bytes
    ):
        parser.error("--memory-swap-total-limit-bytes must be at least --memory-limit-bytes")
    if args.strace_string_limit <= 0:
        parser.error("--strace-string-limit must be positive")
    if args.scheduler_attach_timeout_sec <= 0:
        parser.error("--scheduler-attach-timeout-sec must be positive")
    if args.scheduler_cpu_tolerance_fraction < 0 or args.scheduler_cpu_tolerance_sec < 0:
        parser.error("scheduler CPU tolerances must not be negative")
    if args.scheduler_perf_mmap_pages is not None and args.scheduler_perf_mmap_pages <= 0:
        parser.error("--scheduler-perf-mmap-pages must be positive")
    if args.scheduler_bcc_buffer_pages is not None and (
        args.scheduler_bcc_buffer_pages <= 0 or args.scheduler_bcc_buffer_pages & (args.scheduler_bcc_buffer_pages - 1)
    ):
        parser.error("--scheduler-bcc-buffer-pages must be a positive power of two")
    if args.proc_stat_interval_sec <= 0 or args.proc_stat_baseline_sec <= 0:
        parser.error("proc-stat interval and baseline durations must be positive")
    if args.memory_sample_interval_sec <= 0:
        parser.error("--memory-sample-interval-sec must be positive")
    if args.hardware_perf_attach_timeout_sec <= 0:
        parser.error("--hardware-perf-attach-timeout-sec must be positive")
    if not 0 < args.hardware_perf_minimum_running_ratio <= 1:
        parser.error("--hardware-perf-minimum-running-ratio must be in (0, 1]")
    try:
        config = ProfileConfig(
            command=args.command,
            prepare_command=args.prepare_command,
            verification_command=args.verification_command,
            output_dir=args.output_dir,
            image=args.image,
            container=args.container,
            cpuset_cpus=args.cpuset_cpus,
            cpu_limit=args.cpu_limit,
            allocated_cpus=args.allocated_cpus,
            workdir=args.workdir,
            user=args.user,
            env=tuple(args.env),
            mounts=tuple(args.mount),
            strace_mode=args.strace_mode,
            string_limit=args.strace_string_limit,
            shell=args.shell,
            scheduler_concurrency=args.scheduler_concurrency,
            scheduler_attach_timeout_sec=args.scheduler_attach_timeout_sec,
            scheduler_cpu_tolerance_fraction=args.scheduler_cpu_tolerance_fraction,
            scheduler_cpu_tolerance_sec=args.scheduler_cpu_tolerance_sec,
            scheduler_backend=args.scheduler_backend,
            scheduler_bcc_buffer_pages=args.scheduler_bcc_buffer_pages,
            scheduler_perf_binary=args.scheduler_perf_binary,
            scheduler_perf_mmap_pages=args.scheduler_perf_mmap_pages,
            proc_stat_comparison=args.proc_stat_comparison,
            proc_stat_interval_sec=args.proc_stat_interval_sec,
            proc_stat_baseline_sec=args.proc_stat_baseline_sec,
            memory_profile=args.memory_profile,
            memory_sample_interval_sec=args.memory_sample_interval_sec,
            memory_limit_bytes=args.memory_limit_bytes,
            memory_swap_total_limit_bytes=args.memory_swap_total_limit_bytes,
            terminate_descendants_at_finish=args.terminate_descendants_at_finish,
            network=args.network,
            network_policy=network_policy,
            network_policy_proxy_image=args.network_policy_proxy_image,
            network_policy_external_network=args.network_policy_external_network,
            runtime=args.runtime,
            docker=args.docker,
            enroot=args.enroot,
            profile_sets=tuple(args.profile_set),
            coordinated_baseline=args.coordinated_baseline,
            hardware_perf_binary=args.hardware_perf_binary,
            hardware_perf_events=tuple(args.hardware_perf_event or DEFAULT_HARDWARE_PERF_EVENTS),
            hardware_perf_attach_timeout_sec=(args.hardware_perf_attach_timeout_sec),
            hardware_perf_minimum_running_ratio=(args.hardware_perf_minimum_running_ratio),
        )
        report = run_profile(config)
    except (OSError, ProfileError) as exc:
        parser.error(str(exc))
    cpu = report["cpu"]
    print(f"Command exit code: {report['command_exit_code']}")
    for line in cpu_summary_lines(cpu):
        print(line)
    if report.get("scheduler_concurrency") is not None:
        scheduler = report["scheduler_concurrency"]
        validation = scheduler["cgroup_validation"]
        print("Peak simultaneous logical cores: " f"{scheduler['peak_simultaneous_logical_cores']}")
        print(
            "Scheduler/cgroup CPU validation: "
            f"{'valid' if validation['valid_for_concurrency_reporting'] else 'invalid'}"
        )
    if report.get("proc_stat_comparison") is not None:
        proc = report["proc_stat_comparison"]
        print("Sampled /proc active-CPU peak (comparison only): " f"{proc['measurement']['peak_active_logical_cpus']}")
    if report.get("memory") is not None:
        memory = report["memory"]
        lifetime_peak = memory["lifetime"]["peak_bytes"]
        command_peak = (memory.get("command_window") or {}).get("sampled_peak_bytes")
        print(
            "Sandbox controlled-finish memory peak: "
            + (f"{lifetime_peak / (1024 * 1024):.3f} MiB" if isinstance(lifetime_peak, int) else "n/a")
        )
        print(
            "Command-window sampled memory peak: "
            + (f"{command_peak / (1024 * 1024):.3f} MiB" if isinstance(command_peak, int) else "n/a")
        )
        print(f"CPU/memory timeline: {memory['samples_path']}")
        print("Natural peak trustworthy: " f"{'yes' if memory['natural_peak_trustworthy'] else 'no'}")
        margin = memory.get("observed_limit_margin", {})
        remaining = margin.get("remaining_to_limit_bytes")
        if isinstance(remaining, int):
            print("Observed leaf-limit margin (not a recommendation): " f"{remaining / (1024 * 1024):.3f} MiB")
    print(f"Report: {config.output_dir / 'profile_report.json'}")
    if report.get("evidence_bundle") is not None:
        print(f"Evidence bundle: {report['evidence_bundle']['path']}")
    invalid_scheduler = (
        report.get("scheduler_concurrency") is not None
        and not report["scheduler_concurrency"]["cgroup_validation"]["valid_for_concurrency_reporting"]
    )
    invalid_memory = report.get("memory") is not None and (
        not report["memory"].get("instrumentation_valid", False)
        or not report["memory"].get("peak_valid", False)
        or not report["memory"].get("timeline_valid_for_shape", True)
        or memory_oom_kill_observed(report["memory"])
    )
    invalid_coordinated_profiling = report.get("profiling_valid") is False
    return int(report["command_exit_code"] != 0 or invalid_scheduler or invalid_memory or invalid_coordinated_profiling)


if __name__ == "__main__":
    sys.exit(main())
