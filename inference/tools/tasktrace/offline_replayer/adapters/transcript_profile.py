#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Profile one recorded transcript trial inside a disposable Docker image.

This module is the end-to-end adapter: it joins a transcript that records
agent tool calls to a benchmark image that supplies the original execution
environment. Keeping orchestration here leaves ``replay transcript`` usable
for host-only debugging and ``profile container`` usable for arbitrary
one-command container measurements.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shlex
import shutil
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

from ..measurement.container import (
    DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
    PROFILE_SETS,
    SCHEDULER_BACKENDS,
    STRACE_MODES,
    ProfileConfig,
    ProfileError,
    memory_oom_kill_observed,
)
from ..measurement.network_policy import (
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyError,
    resolve_cli_network_policy,
)
from ..measurement.pipeline import ContainerMeasurementOptions, run_profiled_replay
from .transcript_replay import ReplayError, extract_actions

TOOL_SOURCE_MOUNT = "/tasktrace-source"
TRANSCRIPT_MOUNT = "/tasktrace-input"
REPLAY_MOUNT = "/tasktrace"
INITIAL_WORKSPACE_MOUNT = "/tasktrace-initial-workspace"


@dataclass(frozen=True)
class TranscriptTrialProfileConfig:
    transcript: Path
    task_id: str
    trial_name: str
    image: str
    output_dir: Path
    cpuset_cpus: str | None
    allocated_cpus: float | None
    source_workspace: Path | None
    initial_workspace: Path | None
    pinchbench_skill_dir: Path | None
    timeout_sec: float
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
    docker: str = "docker"
    allow_unsupported: bool = False
    profile_sets: tuple[str, ...] = ()
    coordinated_baseline: bool = False
    verification_command: str | None = None


def _shell_join(arguments: list[str]) -> str:
    return " ".join(shlex.quote(argument) for argument in arguments)


def _replay_command(config: TranscriptTrialProfileConfig) -> str:
    """Build the inner command so replay itself runs in the task container.

    Running the interpreter inside the profiled container is deliberate: shell
    commands in a captured trajectory must see that image's tools, libraries,
    and OS behavior rather than whatever is installed on the host.
    """
    transcript_in_container = f"{TRANSCRIPT_MOUNT}/{config.transcript.name}"
    command = [
        "python3",
        "-m",
        "offline_replayer.adapters.transcript_replay",
        transcript_in_container,
        "--task-id",
        config.task_id,
        "--trial-name",
        config.trial_name,
        "--workspace",
        f"{REPLAY_MOUNT}/workspace",
        "--timeout-sec",
        str(config.timeout_sec),
        "--output-dir",
        f"{REPLAY_MOUNT}/output",
    ]
    if config.source_workspace is not None:
        command.extend(["--source-workspace", str(config.source_workspace)])
    if config.initial_workspace is not None:
        command.extend(["--initial-workspace", INITIAL_WORKSPACE_MOUNT])
    return f"PYTHONPATH={TOOL_SOURCE_MOUNT} {_shell_join(command)}"


def _contained_path(root: Path, relative: str, *, description: str) -> Path:
    """Resolve a task-declared fixture path without permitting path traversal."""
    target = (root / relative).resolve()
    try:
        target.relative_to(root.resolve())
    except ValueError as exc:
        raise ReplayError(f"{description} escapes its root: {relative}") from exc
    return target


def _path_tree_sha256(path: Path) -> str:
    """Hash a replay input without following symlinks.

    Transcript actions can depend on a restored workspace that is not encoded
    in the transcript itself. Include relative names, entry kinds, modes, link
    targets, file sizes, and file bytes so comparison campaigns cannot silently
    mix different starting state.
    """
    root = path.resolve()
    if not root.exists():
        raise ReplayError(f"workspace provenance input does not exist: {path}")
    entries = [root]
    if root.is_dir():
        entries.extend(sorted(root.rglob("*"), key=lambda entry: entry.relative_to(root).as_posix()))
    digest = hashlib.sha256()
    for entry in entries:
        relative = "." if entry == root else entry.relative_to(root).as_posix()
        metadata = entry.lstat()
        if entry.is_symlink():
            kind = "symlink"
            payload = entry.readlink().as_posix().encode()
            payload_size = len(payload)
        elif entry.is_dir():
            kind = "directory"
            payload = b""
            payload_size = 0
        elif entry.is_file():
            kind = "file"
            payload = None
            payload_size = metadata.st_size
        else:
            raise ReplayError(f"unsupported workspace entry for provenance: {entry}")
        digest.update(f"{kind}\0{relative}\0{metadata.st_mode & 0o7777:o}\0{payload_size}\0".encode())
        if payload is None:
            with entry.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
        else:
            digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def stage_pinchbench_initial_workspace(skill_dir: Path, task_id: str, destination: Path) -> tuple[Path, int]:
    """Restore the initial task workspace declared by official PinchBench metadata.

    A transcript records what the agent did after the task began; it does not
    embed large starting inputs such as logs, PDFs, or source trees. Restoring
    ``workspace_files`` here recreates those inputs before the measured replay.
    """
    canonical_task_id = task_id.rsplit("__", 1)[-1]
    task_file = skill_dir / "tasks" / f"{canonical_task_id}.md"
    if not task_file.is_file():
        raise ReplayError(f"PinchBench task definition does not exist: {task_file}")
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", task_file.read_text(), re.DOTALL)
    if match is None:
        raise ReplayError(f"PinchBench task definition has no YAML frontmatter: {task_file}")
    try:
        metadata = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError as exc:
        raise ReplayError(f"invalid PinchBench task frontmatter in {task_file}: {exc}") from exc
    workspace_files = metadata.get("workspace_files", [])
    if not isinstance(workspace_files, list):
        raise ReplayError(f"invalid PinchBench workspace_files in {task_file}")

    destination.mkdir(parents=True, exist_ok=True)
    restored = 0
    assets_root = (skill_dir / "assets").resolve()
    for specification in workspace_files:
        if not isinstance(specification, dict):
            raise ReplayError(f"invalid PinchBench workspace file specification in {task_file}")
        if "content" in specification:
            # Some tasks define small starter projects inline in YAML rather
            # than storing separate asset files.
            relative = specification.get("path")
            if not isinstance(relative, str) or not isinstance(specification["content"], str):
                raise ReplayError(f"invalid inline PinchBench workspace file in {task_file}")
            target = _contained_path(destination, relative, description="workspace fixture path")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(specification["content"])
            restored += 1
            continue
        source_relative = specification.get("source")
        destination_relative = specification.get("dest")
        if not isinstance(source_relative, str) or not isinstance(destination_relative, str):
            raise ReplayError(f"invalid sourced PinchBench workspace file in {task_file}")
        source = _contained_path(assets_root, source_relative, description="fixture source")
        if not source.is_file():
            raise ReplayError(f"PinchBench fixture source does not exist: {source}")
        target = _contained_path(destination, destination_relative, description="workspace fixture path")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        restored += 1
    return task_file, restored


def build_container_profile_config(
    config: TranscriptTrialProfileConfig,
) -> tuple[ProfileConfig, int]:
    """Validate one trial and construct its disposable container profile.

    The wrapper refuses unsupported tools by default because a partial replay
    can appear successful while omitting the work whose CPU cost is under test.
    """
    extraction = extract_actions(
        config.transcript,
        task_id=config.task_id,
        trial_name=config.trial_name,
    )
    unsupported = [action["source_tool"] for action in extraction.actions if not action["supported"]]
    if unsupported and not config.allow_unsupported:
        raise ReplayError(
            "selected trial contains unsupported tools; use --allow-unsupported "
            f"for a partial replay: {', '.join(sorted(set(unsupported)))}"
        )

    # The replay module is mounted from the repository root so its package
    # import path remains identical inside the measured container.
    tool_source = Path(__file__).resolve().parents[2]
    replay_dir = config.output_dir / "replay"
    # Source and transcript are inputs, so mount them read-only. Only the
    # dedicated replay directory is writable and becomes an audit artifact.
    mounts = [
        f"type=bind,src={tool_source},dst={TOOL_SOURCE_MOUNT},readonly",
        f"type=bind,src={config.transcript.resolve().parent},dst={TRANSCRIPT_MOUNT},readonly",
        f"type=bind,src={replay_dir.resolve()},dst={REPLAY_MOUNT}",
    ]
    if config.initial_workspace is not None:
        mounts.append(
            "type=bind," f"src={config.initial_workspace.resolve()}," f"dst={INITIAL_WORKSPACE_MOUNT},readonly"
        )

    return (
        ContainerMeasurementOptions(
            image=config.image,
            cpu_limit=None,
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
            docker=config.docker,
            profile_sets=config.profile_sets,
            coordinated_baseline=config.coordinated_baseline,
            verification_command=config.verification_command,
        ).build_profile(
            command=_replay_command(config),
            prepare_command=config.prepare_command,
            output_dir=config.output_dir / "container_profile",
            mounts=tuple(mounts),
            workload_identity={
                "task_id": config.task_id,
                "trial_name": config.trial_name,
            },
            source_identity={
                "format": "transcript-jsonl",
                "sha256": hashlib.sha256(config.transcript.read_bytes()).hexdigest(),
                "action_count": len(extraction.actions),
                "initial_workspace_sha256": (
                    _path_tree_sha256(config.initial_workspace) if config.initial_workspace is not None else None
                ),
            },
            expected_outcome={
                "kind": "source-recorded-action-outcomes",
                "action_count": len(extraction.actions),
                "allow_unsupported": config.allow_unsupported,
            },
        ),
        len(extraction.actions),
    )


def run_transcript_trial_profile(config: TranscriptTrialProfileConfig) -> dict[str, Any]:
    """Run one recorded trial in Docker and write a combined pipeline report."""
    if config.output_dir.exists() and any(config.output_dir.iterdir()):
        raise ProfileError(f"pipeline output directory must be new or empty: {config.output_dir}")
    if not config.transcript.is_file():
        raise ReplayError(f"transcript does not exist: {config.transcript}")
    if config.initial_workspace is not None and not config.initial_workspace.is_dir():
        raise ReplayError(f"initial workspace does not exist: {config.initial_workspace}")
    if config.initial_workspace is not None and config.pinchbench_skill_dir is not None:
        raise ReplayError("provide either --initial-workspace or --pinchbench-skill-dir, not both")

    runtime_config = config
    task_definition: str | None = None
    fixtures_restored = 0
    if config.pinchbench_skill_dir is not None:
        # Fixture restoration is host-side setup, intentionally completed
        # before run_profile starts the container's measured command window.
        seed_dir = config.output_dir / "initial_workspace"
        task_file, fixtures_restored = stage_pinchbench_initial_workspace(
            config.pinchbench_skill_dir, config.task_id, seed_dir
        )
        task_definition = str(task_file.resolve())
        runtime_config = replace(config, initial_workspace=seed_dir)

    container_config, action_count = build_container_profile_config(runtime_config)
    initial_workspace = runtime_config.initial_workspace
    source_workspace = runtime_config.source_workspace
    source_workspace_exists = source_workspace is not None and source_workspace.exists()
    return run_profiled_replay(
        profile=container_config,
        output_dir=config.output_dir,
        replay_report_relative_path=Path("replay/output/replay_report.json"),
        report_fields={
            "schema_version": "container-transcript-profile.v1",
            "task_id": config.task_id,
            "trial_name": config.trial_name,
            "transcript": str(config.transcript.resolve()),
            "transcript_sha256": hashlib.sha256(config.transcript.read_bytes()).hexdigest(),
            "initial_workspace": (str(initial_workspace.resolve()) if initial_workspace is not None else None),
            "initial_workspace_sha256": (
                _path_tree_sha256(initial_workspace) if initial_workspace is not None else None
            ),
            "source_workspace": (str(source_workspace.resolve()) if source_workspace is not None else None),
            "source_workspace_sha256": (_path_tree_sha256(source_workspace) if source_workspace_exists else None),
            "source_workspace_provenance_state": (
                "none" if source_workspace is None else ("hashed" if source_workspace_exists else "unavailable")
            ),
            "image": config.image,
            "actions_validated": action_count,
            "task_definition": task_definition,
            "fixtures_restored": fixtures_restored,
            "workspace": str((config.output_dir / "replay" / "workspace").resolve()),
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Profile one extracted transcript trial inside a disposable Docker image "
            "using process tracing and container cgroup CPU statistics."
        )
    )
    parser.add_argument("transcript", type=Path, help="Transcript JSONL containing the trial.")
    parser.add_argument("--task-id", required=True, help="Task ID to extract.")
    parser.add_argument(
        "--trial-name",
        required=True,
        help="Exact single recorded trial name to extract and replay.",
    )
    parser.add_argument("--image", required=True, help="Prepared Docker image for the task environment.")
    parser.add_argument("--output-dir", type=Path, required=True, help="New output directory.")
    parser.add_argument("--cpuset-cpus", help="Logical CPU set for the disposable container.")
    parser.add_argument(
        "--allocated-cpus",
        type=float,
        help="CPU denominator when no --cpuset-cpus is set.",
    )
    parser.add_argument(
        "--source-workspace",
        type=Path,
        help="Captured workspace prefix remapped to the fresh replay workspace.",
    )
    parser.add_argument(
        "--initial-workspace",
        type=Path,
        help="Optional initial workspace contents mounted and copied before replay.",
    )
    parser.add_argument(
        "--pinchbench-skill-dir",
        type=Path,
        help="Official PinchBench skill checkout used to restore task-declared input fixtures.",
    )
    parser.add_argument("--timeout-sec", type=float, default=60.0, help="Per-exec timeout.")
    parser.add_argument(
        "--prepare-command",
        help="Container setup executed before the measured replay, such as installing strace.",
    )
    parser.add_argument(
        "--verification-command",
        help=(
            "Command run in the replay sandbox after measurement; a nonzero "
            "exit invalidates the profile without changing measured counters."
        ),
    )
    parser.add_argument("--strace-mode", choices=STRACE_MODES, default="execve")
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
    parser.add_argument(
        "--scheduler-concurrency",
        action="store_true",
        help=(
            "Use scheduler tracing to report simultaneous " "logical-CPU occupancy for this disposable container trial."
        ),
    )
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
    parser.add_argument("--workdir", default="/workspace")
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
    parser.add_argument(
        "--allow-unsupported",
        action="store_true",
        help="Allow a partial replay when the selected trial uses unsupported tool calls.",
    )
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
    if args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
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
    config = TranscriptTrialProfileConfig(
        transcript=args.transcript,
        task_id=args.task_id,
        trial_name=args.trial_name,
        image=args.image,
        output_dir=args.output_dir,
        cpuset_cpus=args.cpuset_cpus,
        allocated_cpus=args.allocated_cpus,
        source_workspace=args.source_workspace,
        initial_workspace=args.initial_workspace,
        pinchbench_skill_dir=args.pinchbench_skill_dir,
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
        docker=args.docker,
        allow_unsupported=args.allow_unsupported,
        profile_sets=tuple(args.profile_set),
        coordinated_baseline=args.coordinated_baseline,
    )
    try:
        report = run_transcript_trial_profile(config)
    except (OSError, ProfileError, ReplayError) as exc:
        parser.error(str(exc))

    cpu = report["container_cpu"]
    replay = report["replay_summary"]
    print(f"Task: {report['task_id']}")
    print(f"Trial: {report['trial_name']}")
    print(
        "Actions: "
        f"{replay['actions_replayed']}/{replay['actions_extracted']} replayed, "
        f"{replay['actions_ok']} ok, {replay['actions_failed']} failed, "
        f"{replay['actions_unsupported']} unsupported"
    )
    print(f"Wall time: {cpu['wall_sec']:.9f} sec")
    print(f"CPU time: {cpu['usage_sec']:.9f} sec")
    print(f"Average logical cores used: {cpu['average_logical_cores_used']:.6f}")
    print(
        "Allocated CPU utilization: "
        f"{cpu['utilization_percent_of_allocation']:.3f}% of "
        f"{cpu['allocated_logical_cpus']:g} logical CPUs"
    )
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
    return int(invalid_scheduler or invalid_memory or invalid_coordinated_profiling)


if __name__ == "__main__":
    sys.exit(main())
