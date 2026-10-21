#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile a finalized TaskTrace action trace inside a disposable task image.

The finalized tape carries the shell work from a real live run, while the
disposable task image supplies a controlled CPU measurement boundary. Keeping
these concerns separate lets the replay preserve source pacing and quota
semantics without invoking the model again.
"""

from __future__ import annotations

import argparse
import hashlib
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..measurement.container import (
    CONTAINER_RUNTIMES,
    DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
    PROFILE_SETS,
    SCHEDULER_BACKENDS,
    STRACE_MODES,
    ProfileConfig,
    ProfileError,
    cpu_summary_lines,
    memory_oom_kill_observed,
)
from ..measurement.network_policy import (
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyError,
    resolve_cli_network_policy,
)
from ..measurement.pipeline import ContainerMeasurementOptions, run_profiled_replay
from .trace_replay import PACING_MODES, TaskTraceTrace, read_trace
from .transcript_replay import ReplayError

TOOL_SOURCE_MOUNT = "/tasktrace-source"
TRACE_MOUNT = "/tasktrace-input"
REPLAY_MOUNT = "/tasktrace"


@dataclass(frozen=True)
class TaskTraceTraceProfileConfig:
    trace: Path
    image: str
    output_dir: Path
    cpu_limit: float | None
    cpuset_cpus: str | None
    allocated_cpus: float | None
    pacing: str
    timeout_sec: float | None
    prepare_command: str | None
    strace_mode: str
    string_limit: int
    shell: str
    workdir: str
    scheduler_concurrency: bool = False
    scheduler_attach_timeout_sec: float = 20.0
    scheduler_cpu_tolerance_fraction: float = 0.02
    scheduler_cpu_tolerance_sec: float = 0.005
    scheduler_backend: str = "bcc"
    scheduler_perf_binary: str = "perf"
    scheduler_perf_mmap_pages: int | None = None
    proc_stat_comparison: bool = False
    proc_stat_interval_sec: float = 0.5
    proc_stat_baseline_sec: float = 5.0
    memory_profile: bool = False
    memory_sample_interval_sec: float = 0.1
    memory_limit_bytes: int | None = None
    memory_swap_total_limit_bytes: int | None = None
    network: str | None = None
    network_policy: NetworkPolicy | None = None
    network_policy_proxy_image: str = DEFAULT_NETWORK_POLICY_PROXY_IMAGE
    network_policy_external_network: str = "bridge"
    runtime: str = "docker"
    docker: str = "docker"
    enroot: str = "enroot"
    profile_sets: tuple[str, ...] = ()
    coordinated_baseline: bool = False
    verification_command: str | None = None


def _shell_join(arguments: list[str]) -> str:
    return " ".join(shlex.quote(argument) for argument in arguments)


def _replay_command(config: TaskTraceTraceProfileConfig) -> str:
    command = [
        "python3",
        "-m",
        "offline_replayer.adapters.trace_replay",
        f"{TRACE_MOUNT}/{config.trace.name}",
        "--output",
        f"{REPLAY_MOUNT}/replay_report.json",
        "--cwd",
        config.workdir,
        "--pacing",
        config.pacing,
    ]
    if config.timeout_sec is not None:
        command.extend(["--timeout-sec", str(config.timeout_sec)])
    return f"PYTHONPATH={TOOL_SOURCE_MOUNT} {_shell_join(command)}"


def build_container_profile_config(
    config: TaskTraceTraceProfileConfig,
) -> tuple[ProfileConfig, TaskTraceTrace]:
    """Validate the finalized trace and construct its container measurement."""
    trace = read_trace(config.trace)
    # The replay module is mounted from the repository root so its package
    # import path remains identical inside the measured container.
    tool_source = Path(__file__).resolve().parents[2]
    replay_dir = config.output_dir / "replay"
    mounts = (
        f"type=bind,src={tool_source},dst={TOOL_SOURCE_MOUNT},readonly",
        f"type=bind,src={config.trace.resolve().parent},dst={TRACE_MOUNT},readonly",
        f"type=bind,src={replay_dir.resolve()},dst={REPLAY_MOUNT}",
    )
    profile = ContainerMeasurementOptions(
        image=config.image,
        cpu_limit=config.cpu_limit,
        cpuset_cpus=config.cpuset_cpus,
        allocated_cpus=config.allocated_cpus,
        strace_mode=config.strace_mode,
        string_limit=config.string_limit,
        shell=config.shell,
        workdir=config.workdir,
        scheduler_concurrency=config.scheduler_concurrency,
        scheduler_attach_timeout_sec=config.scheduler_attach_timeout_sec,
        scheduler_cpu_tolerance_fraction=config.scheduler_cpu_tolerance_fraction,
        scheduler_cpu_tolerance_sec=config.scheduler_cpu_tolerance_sec,
        scheduler_backend=config.scheduler_backend,
        scheduler_perf_binary=config.scheduler_perf_binary,
        scheduler_perf_mmap_pages=config.scheduler_perf_mmap_pages,
        proc_stat_comparison=config.proc_stat_comparison,
        proc_stat_interval_sec=config.proc_stat_interval_sec,
        proc_stat_baseline_sec=config.proc_stat_baseline_sec,
        memory_profile=config.memory_profile,
        memory_sample_interval_sec=config.memory_sample_interval_sec,
        memory_limit_bytes=config.memory_limit_bytes,
        memory_swap_total_limit_bytes=config.memory_swap_total_limit_bytes,
        network=config.network,
        network_policy=config.network_policy,
        network_policy_proxy_image=config.network_policy_proxy_image,
        network_policy_external_network=config.network_policy_external_network,
        runtime=config.runtime,
        docker=config.docker,
        enroot=config.enroot,
        profile_sets=config.profile_sets,
        coordinated_baseline=config.coordinated_baseline,
        verification_command=config.verification_command,
    ).build_profile(
        command=_replay_command(config),
        prepare_command=config.prepare_command,
        output_dir=config.output_dir / "container_profile",
        mounts=mounts,
        workload_identity={
            "task_name": trace.header.get("task_name"),
            "source_trial": trace.header.get("source_trial"),
        },
        source_identity={
            "format": "trace",
            "schema_version": trace.header.get("schema_version"),
            "sha256": hashlib.sha256(config.trace.read_bytes()).hexdigest(),
            "action_count": len(trace.actions),
        },
        expected_outcome={
            "kind": "source-recorded-action-outcomes",
            "action_count": len(trace.actions),
        },
    )
    return profile, trace


def run_tasktrace_trace_profile(config: TaskTraceTraceProfileConfig) -> dict[str, Any]:
    """Run one paced tape replay and preserve both CPU and fidelity evidence.

    A replay may produce useful CPU data even when background build timing
    causes one source outcome to drift. The pipeline report therefore combines
    the measurement with the outcome comparison instead of hiding either.
    """
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ProfileError(f"pipeline output directory must be new or empty: {config.output_dir}")
    profile, trace = build_container_profile_config(config)
    return run_profiled_replay(
        profile=profile,
        output_dir=config.output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={
            "schema_version": "container-trace-profile.v1",
            "task_name": trace.header.get("task_name"),
            "source_trial": trace.header.get("source_trial"),
            "source_reward": trace.header.get("reward"),
            "source_trace": str(config.trace.resolve()),
            "source_trace_sha256": hashlib.sha256(config.trace.read_bytes()).hexdigest(),
            "source_trace_schema_version": trace.header.get("schema_version"),
            "source_limitations": trace.header.get("limitations", []),
            "image": config.image,
            "pacing": config.pacing,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile a finalized TaskTrace shell-action trace inside a disposable "
            "container image using cgroup CPU statistics."
        )
    )
    parser.add_argument("trace", type=Path, help="Finalized tasktrace_shell_commands.jsonl.")
    parser.add_argument("--image", required=True, help="Prepared image for the task.")
    parser.add_argument("--runtime", choices=CONTAINER_RUNTIMES, default="docker")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument(
        "--cpu-limit",
        type=float,
        help="Docker CPU quota, used to match the source benchmark task limit.",
    )
    parser.add_argument("--cpuset-cpus", help="Optional logical CPU affinity for the replay.")
    parser.add_argument(
        "--allocated-cpus",
        type=float,
        help="CPU denominator when no --cpuset-cpus is set.",
    )
    parser.add_argument("--pacing", choices=PACING_MODES, default="recorded")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        help="Override per-action timeout; by default use the value recorded in the trace.",
    )
    parser.add_argument(
        "--prepare-command",
        help="Container setup performed before the measured replay, for example installing python3.",
    )
    parser.add_argument(
        "--verification-command",
        help=(
            "Command run in the replay sandbox after measurement; a nonzero "
            "exit invalidates the profile without changing measured counters."
        ),
    )
    parser.add_argument("--strace-mode", choices=STRACE_MODES, default="none")
    parser.add_argument("--strace-string-limit", type=int, default=4096)
    profiler_mode = parser.add_mutually_exclusive_group()
    profiler_mode.add_argument(
        "--coordinated-baseline",
        action="store_true",
        help=(
            "Run an unprofiled replay through the same coordinator, sandbox "
            "gates, lifecycle, and execution plan used by --profile-set."
        ),
    )
    profiler_mode.add_argument(
        "--profile-set",
        action="append",
        choices=PROFILE_SETS,
        default=[],
        help="Profiler set to run in the same sandbox execution; repeatable.",
    )
    parser.add_argument("--scheduler-concurrency", action="store_true")
    parser.add_argument("--scheduler-backend", choices=SCHEDULER_BACKENDS, default="bcc")
    parser.add_argument("--scheduler-perf-binary", default="perf")
    parser.add_argument("--scheduler-perf-mmap-pages", type=int)
    parser.add_argument("--proc-stat-comparison", action="store_true")
    parser.add_argument("--proc-stat-interval-sec", type=float, default=0.5)
    parser.add_argument("--proc-stat-baseline-sec", type=float, default=5.0)
    parser.add_argument("--memory-profile", action="store_true")
    parser.add_argument("--memory-sample-interval-sec", type=float, default=0.1)
    parser.add_argument("--memory-limit-bytes", type=int)
    parser.add_argument("--memory-swap-total-limit-bytes", type=int)
    parser.add_argument("--scheduler-attach-timeout-sec", type=float, default=20.0)
    parser.add_argument("--scheduler-cpu-tolerance-fraction", type=float, default=0.02)
    parser.add_argument("--scheduler-cpu-tolerance-sec", type=float, default=0.005)
    parser.add_argument("--shell", default="/bin/bash")
    parser.add_argument("--workdir", default="/workdir")
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
    parser.add_argument("--allowed-host", action="append", default=[])
    parser.add_argument(
        "--network-policy-proxy-image",
        default=DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
    )
    parser.add_argument("--network-policy-external-network", default="bridge")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--enroot", default="enroot")
    args = parser.parse_args(argv)
    try:
        network_policy = resolve_cli_network_policy(
            mode=args.network_policy,
            allowed_hosts=args.allowed_host,
            legacy_network=args.network,
        )
    except NetworkPolicyError as exc:
        parser.error(str(exc))
    if args.cpuset_cpus is None and args.allocated_cpus is None:
        parser.error("provide --cpuset-cpus or --allocated-cpus for utilization reporting")
    if args.cpu_limit is not None and args.cpu_limit <= 0:
        parser.error("--cpu-limit must be positive")
    if args.timeout_sec is not None and args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    if args.strace_string_limit <= 0:
        parser.error("--strace-string-limit must be positive")
    if args.scheduler_attach_timeout_sec <= 0:
        parser.error("--scheduler-attach-timeout-sec must be positive")
    if args.scheduler_cpu_tolerance_fraction < 0 or args.scheduler_cpu_tolerance_sec < 0:
        parser.error("scheduler CPU tolerances must not be negative")
    if (
        args.scheduler_backend != "bcc"
        and not args.scheduler_concurrency
        and not {"cpu-compare", "cpu-peak"} & set(args.profile_set)
    ):
        parser.error("--scheduler-backend requires --scheduler-concurrency")
    if args.proc_stat_comparison and not args.cpuset_cpus:
        parser.error("--proc-stat-comparison requires --cpuset-cpus")
    if args.scheduler_perf_mmap_pages is not None and args.scheduler_perf_mmap_pages <= 0:
        parser.error("--scheduler-perf-mmap-pages must be positive")
    if args.proc_stat_interval_sec <= 0 or args.proc_stat_baseline_sec <= 0:
        parser.error("proc-stat interval and baseline durations must be positive")
    if args.memory_sample_interval_sec <= 0:
        parser.error("--memory-sample-interval-sec must be positive")
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
    if args.runtime == "enroot" and args.coordinated_baseline:
        parser.error("--coordinated-baseline currently requires the Docker runtime")
    config = TaskTraceTraceProfileConfig(
        trace=args.trace,
        image=args.image,
        output_dir=args.output_dir,
        cpu_limit=args.cpu_limit,
        cpuset_cpus=args.cpuset_cpus,
        allocated_cpus=args.allocated_cpus,
        pacing=args.pacing,
        timeout_sec=args.timeout_sec,
        prepare_command=args.prepare_command,
        verification_command=args.verification_command,
        strace_mode=args.strace_mode,
        string_limit=args.strace_string_limit,
        shell=args.shell,
        workdir=args.workdir,
        scheduler_concurrency=args.scheduler_concurrency,
        scheduler_attach_timeout_sec=args.scheduler_attach_timeout_sec,
        scheduler_cpu_tolerance_fraction=args.scheduler_cpu_tolerance_fraction,
        scheduler_cpu_tolerance_sec=args.scheduler_cpu_tolerance_sec,
        scheduler_backend=args.scheduler_backend,
        scheduler_perf_binary=args.scheduler_perf_binary,
        scheduler_perf_mmap_pages=args.scheduler_perf_mmap_pages,
        proc_stat_comparison=args.proc_stat_comparison,
        proc_stat_interval_sec=args.proc_stat_interval_sec,
        proc_stat_baseline_sec=args.proc_stat_baseline_sec,
        memory_profile=args.memory_profile,
        memory_sample_interval_sec=args.memory_sample_interval_sec,
        memory_limit_bytes=args.memory_limit_bytes,
        memory_swap_total_limit_bytes=args.memory_swap_total_limit_bytes,
        network=args.network,
        network_policy=network_policy,
        network_policy_proxy_image=args.network_policy_proxy_image,
        network_policy_external_network=args.network_policy_external_network,
        runtime=args.runtime,
        docker=args.docker,
        enroot=args.enroot,
        profile_sets=tuple(args.profile_set),
        coordinated_baseline=args.coordinated_baseline,
    )
    try:
        report = run_tasktrace_trace_profile(config)
    except (OSError, ProfileError, ReplayError) as exc:
        parser.error(str(exc))
    cpu = report["container_cpu"]
    replay = report["replay_summary"]
    print(f"Task: {report['task_name']}")
    print(f"Source trial: {report['source_trial']}")
    print(
        "Actions: "
        f"{replay['actions_executed']}/{replay['actions_recorded']} executed, "
        f"{replay['actions_timed_out']} timed out, "
        f"{replay['source_outcome_mismatch_count']} source-outcome mismatches"
    )
    for line in cpu_summary_lines(cpu):
        print(line)
    if report.get("container_memory") is not None:
        memory = report["container_memory"]
        peak = memory["lifetime"]["peak_bytes"]
        print(
            "Sandbox controlled-work memory peak: "
            + (f"{peak / (1024 * 1024):.3f} MiB" if isinstance(peak, int) else "n/a")
        )
        print(f"CPU/memory timeline: {memory['samples_path']}")
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
    print(f"Report: {config.output_dir / 'pipeline_report.json'}")
    invalid_scheduler = (
        report.get("scheduler_concurrency") is not None
        and not report["scheduler_concurrency"]["cgroup_validation"]["valid_for_concurrency_reporting"]
    )
    invalid_memory = report.get("container_memory") is not None and (
        not report["container_memory"].get("instrumentation_valid", False)
        or not report["container_memory"].get("peak_valid", False)
        or not report["container_memory"].get("timeline_valid_for_shape", True)
        or memory_oom_kill_observed(report["container_memory"])
    )
    invalid_coordinated_profiling = report.get("profiling_valid") is False
    return int(
        replay["source_outcome_mismatch_count"] > 0
        or invalid_scheduler
        or invalid_memory
        or invalid_coordinated_profiling
    )


if __name__ == "__main__":
    sys.exit(main())
