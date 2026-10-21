# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Common measured-container execution boundary for replay adapters.

Replay adapters are responsible for interpreting their source format and
constructing the in-container replay command. This module owns the common
measurement invocation and the stable portion of ``pipeline_report.json``.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping

from .container import (
    DEFAULT_NETWORK_POLICY_PROXY_IMAGE,
    ProfileConfig,
    ProfileError,
    controller_source_sha256,
    run_profile,
    validate_profile_config,
)
from .network_policy import NetworkMode, NetworkPolicy


@dataclass(frozen=True)
class ContainerMeasurementOptions:
    """Shared resource and instrumentation settings for one replay profile."""

    image: str | None
    cpu_limit: float | None
    cpuset_cpus: str | None
    allocated_cpus: float | None
    strace_mode: str
    string_limit: int
    shell: str
    workdir: str
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
    network: str | None = None
    network_policy: NetworkPolicy | None = None
    network_policy_proxy_image: str = DEFAULT_NETWORK_POLICY_PROXY_IMAGE
    network_policy_external_network: str = "bridge"
    runtime: str = "docker"
    docker: str = "docker"
    enroot: str = "enroot"
    current_container_exclusive: bool = False
    env: tuple[str, ...] = ()
    unset_env: tuple[str, ...] = ()
    profile_sets: tuple[str, ...] = ()
    coordinated_baseline: bool = False

    def build_profile(
        self,
        *,
        command: str,
        prepare_command: str | None,
        output_dir: Path,
        mounts: tuple[str, ...],
        workload_identity: Mapping[str, Any] | None = None,
        source_identity: Mapping[str, Any] | None = None,
        expected_outcome: Mapping[str, Any] | None = None,
    ) -> ProfileConfig:
        """Build the generic container measurement request for an adapter."""
        if self.runtime == "enroot" and (
            self.network not in (None, "none")
            or (self.network_policy is not None and self.network_policy.mode is NetworkMode.ALLOWLIST)
        ):
            raise ProfileError("Enroot supports only public or no-network policy")
        return ProfileConfig(
            command=command,
            prepare_command=prepare_command,
            output_dir=output_dir,
            image=self.image,
            container=None,
            cpuset_cpus=self.cpuset_cpus,
            cpu_limit=self.cpu_limit,
            allocated_cpus=self.allocated_cpus,
            workdir=self.workdir,
            user=None,
            env=self.env,
            mounts=mounts,
            strace_mode=self.strace_mode,
            string_limit=self.string_limit,
            shell=self.shell,
            verification_command=self.verification_command,
            scheduler_concurrency=self.scheduler_concurrency,
            scheduler_attach_timeout_sec=self.scheduler_attach_timeout_sec,
            scheduler_cpu_tolerance_fraction=self.scheduler_cpu_tolerance_fraction,
            scheduler_cpu_tolerance_sec=self.scheduler_cpu_tolerance_sec,
            scheduler_backend=self.scheduler_backend,
            scheduler_bcc_buffer_pages=self.scheduler_bcc_buffer_pages,
            scheduler_perf_binary=self.scheduler_perf_binary,
            scheduler_perf_mmap_pages=self.scheduler_perf_mmap_pages,
            proc_stat_comparison=self.proc_stat_comparison,
            proc_stat_interval_sec=self.proc_stat_interval_sec,
            proc_stat_baseline_sec=self.proc_stat_baseline_sec,
            memory_profile=self.memory_profile,
            memory_sample_interval_sec=self.memory_sample_interval_sec,
            memory_limit_bytes=self.memory_limit_bytes,
            memory_swap_total_limit_bytes=self.memory_swap_total_limit_bytes,
            terminate_descendants_at_finish=self.terminate_descendants_at_finish,
            network=self.network,
            network_policy=self.network_policy,
            network_policy_proxy_image=self.network_policy_proxy_image,
            network_policy_external_network=self.network_policy_external_network,
            runtime=self.runtime,
            docker=self.docker,
            enroot=self.enroot,
            current_container_exclusive=self.current_container_exclusive,
            unset_env=self.unset_env,
            profile_sets=self.profile_sets,
            coordinated_baseline=self.coordinated_baseline,
            workload_identity=workload_identity,
            source_identity=source_identity,
            expected_outcome=expected_outcome,
        )


def _missing_replay_report_error(output_dir: Path, replay_report_path: Path) -> ProfileError:
    stderr_path = output_dir / "container_profile" / "command.stderr.txt"
    diagnostic = ""
    if stderr_path.is_file():
        diagnostic = stderr_path.read_text(errors="replace").strip()[:1200]
        if diagnostic:
            diagnostic = f"; command stderr: {diagnostic}"
    return ProfileError(f"container command completed without a replay report: {replay_report_path}{diagnostic}")


def _atif_summary_is_valid(summary: dict[str, Any]) -> bool:
    def count(field: str, default: Any = None) -> int | None:
        value = summary.get(field, default)
        return value if type(value) is int and value >= 0 else None

    recorded = count("tool_actions_recorded")
    executed = count("actions_executed")
    reproduced = count("tool_actions_reproduced_without_execution", 0)
    return (
        all(
            count(field) == 0
            for field in (
                "actions_timed_out",
                "actions_skipped_after_timeout",
                "actions_launch_failed",
                "actions_source_argument_parse_failed",
                "source_outcome_mismatch_count",
                "source_outcome_comparison_unavailable_count",
            )
        )
        and recorded is not None
        and recorded > 0
        and count("tool_actions_replayed") == recorded
        and executed is not None
        and reproduced is not None
        and executed + reproduced == recorded
    )


def run_profiled_replay(
    *,
    profile: ProfileConfig,
    output_dir: Path,
    replay_report_relative_path: Path,
    report_fields: Mapping[str, Any],
    cancel_requested: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one measured replay and write the common pipeline report envelope.

    ``report_fields`` holds source-format identity fields. Common measured
    output fields remain identical across adapters and are written here once.
    The replay report path is supplied by the adapter because changing that
    layout would change already-published pipeline report provenance.
    """
    validate_profile_config(profile)
    (output_dir / "replay").mkdir(parents=True, exist_ok=True)
    coordinated = bool(profile.profile_sets) or profile.coordinated_baseline
    # A pipeline publishes one final augmented view. Retain the coordinator's
    # direct bundle, but defer rendering until replay evidence has been added so
    # one request never performs duplicate render passes.
    effective_profile = replace(profile, render_unified_view=False)
    container_report = (
        run_profile(effective_profile)
        if cancel_requested is None
        else run_profile(effective_profile, cancel_requested=cancel_requested)
    )
    replay_report_path = output_dir / replay_report_relative_path
    report: dict[str, Any] = dict(report_fields)
    report.update(
        {
            "image_identity": container_report.get("image_identity"),
            "replay_report": str(replay_report_path.resolve()),
            "container_profile_report": str((output_dir / "container_profile" / "profile_report.json").resolve()),
            "replay_summary": None,
            "container_cpu": container_report["cpu"],
            "scheduler_concurrency": container_report.get("scheduler_concurrency"),
            "proc_stat_comparison": container_report.get("proc_stat_comparison"),
            "strace": container_report["strace"],
            "warning": container_report["warning"],
            "status": "valid",
            "profile_provenance": {
                "mode": "profile" if profile.profile_sets else "baseline",
                "profile_sets": list(profile.profile_sets),
            },
        }
    )
    if container_report.get("runtime") == "current":
        report.update(
            {
                "runtime": "current",
                "measurement_scope": container_report.get("scope"),
                "environment_identity": container_report.get("environment_identity"),
                "task_environment": container_report.get("task_environment"),
                "measurement_validity": container_report.get("measurement_validity"),
            }
        )
    if container_report.get("network_policy") is not None:
        report["network_policy"] = deepcopy(container_report["network_policy"])
    if "network" in container_report:
        # Preserve the v1 top-level compatibility field.  The typed policy is
        # authoritative; this projection lets old readers still distinguish
        # public, deny-all, and allowlist executions.
        report["network"] = container_report["network"]
    if container_report.get("run_id") is not None:
        report["run_id"] = container_report["run_id"]
    if container_report.get("profile_started_at_utc") is not None:
        report["profile_started_at_utc"] = container_report["profile_started_at_utc"]
    if container_report.get("memory") is not None:
        report["container_memory"] = deepcopy(container_report["memory"])
        report["container_memory"].setdefault("controller_source_sha256", controller_source_sha256())
    for field in (
        "canonical_clock_reference",
        "canonical_lifecycle",
        "enroot_lifecycle",
        "evidence_bundle",
        "hardware_perf",
        "process_exec",
        "verification",
        "profiling_valid",
        "profiling_invalid_reasons",
        "unified_profile_artifacts",
    ):
        if container_report.get(field) is not None:
            report[field] = deepcopy(container_report[field])
    if report.get("runtime") == "current":
        measurement_validity = report.get("measurement_validity")
        valid_for_average_cpu = (
            isinstance(measurement_validity, Mapping)
            and measurement_validity.get("valid_for_average_cpu_reporting") is True
        )
        if not valid_for_average_cpu:
            reason = {
                "profiler_id": "current-container-cgroup-cpu",
                "status": "invalid",
                "reasons": ["current-container measurement is not valid for average CPU reporting"],
            }
            invalid_reasons = report.get("profiling_invalid_reasons")
            if not isinstance(invalid_reasons, list):
                invalid_reasons = []
                report["profiling_invalid_reasons"] = invalid_reasons
            if reason not in invalid_reasons:
                invalid_reasons.append(reason)
            report["profiling_valid"] = False
    evidence_attachment = report.get("evidence_bundle")
    if isinstance(evidence_attachment, dict):
        # The container report's attachment identifies the authoritative
        # coordinator bundle. Preserve it under an unambiguous name before the
        # public evidence_bundle pointer is replaced with the final augmented
        # pipeline bundle below.
        report["coordinator_evidence_bundle"] = deepcopy(evidence_attachment)
        for source, target in (
            ("execution_plan_digest", "execution_plan_digest"),
            ("scope", "scope"),
        ):
            if evidence_attachment.get(source) is not None:
                report[target] = deepcopy(evidence_attachment[source])
        attachment_provenance = evidence_attachment.get("provenance")
        if isinstance(attachment_provenance, dict) and isinstance(attachment_provenance.get("execution_plan"), dict):
            report["execution_plan"] = deepcopy(attachment_provenance["execution_plan"])
    if report.get("profiling_valid") is False:
        report["status"] = "invalid"
    if not replay_report_path.is_file():
        error = _missing_replay_report_error(output_dir, replay_report_path)
        memory = report.get("container_memory")
        if isinstance(memory, dict):
            if memory.get("censoring") == "none":
                memory["censoring"] = "functional_failure"
            memory["natural_peak_trustworthy"] = False
            reasons = memory.setdefault("natural_peak_invalid_reasons", [])
            if isinstance(reasons, list):
                reasons.append("container command completed without the required replay report")
        report["pipeline_status"] = "failed_missing_replay_report"
        report["status"] = "invalid"
        report["pipeline_error"] = str(error)
        report["container_command_exit_code"] = container_report.get("command_exit_code")
        (output_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        raise error
    replay_report = json.loads(replay_report_path.read_text())
    report["replay_summary"] = replay_report["summary"]
    summary = report["replay_summary"]
    if not isinstance(summary, dict):
        report["status"] = "invalid"
    else:
        timeout_count = summary.get("actions_timed_out", 0)
        mismatch_count = summary.get("source_outcome_mismatch_count", 0)
        unavailable_count = summary.get("source_outcome_comparison_unavailable_count", 0)
        if (
            not isinstance(timeout_count, int)
            or isinstance(timeout_count, bool)
            or timeout_count != 0
            or not isinstance(mismatch_count, int)
            or isinstance(mismatch_count, bool)
            or mismatch_count != 0
            or not isinstance(unavailable_count, int)
            or isinstance(unavailable_count, bool)
            or unavailable_count != 0
        ):
            report["status"] = "invalid"
    if replay_report.get("schema_version") == "offline-atif-trajectory-replay.v1" and (
        not isinstance(summary, dict) or not _atif_summary_is_valid(summary)
    ):
        report["status"] = "invalid"
    if coordinated:
        direct_attachment = report.get("coordinator_evidence_bundle")
        direct_bundle_attached = isinstance(direct_attachment, dict)
        generation_method = (
            "verified-direct-coordinator-bundle-augmentation"
            if direct_bundle_attached
            else "read-only-retained-evidence-projection"
        )
        direct_status = direct_attachment if isinstance(direct_attachment, dict) else {}
        report["evidence_bundle"] = {
            "path": str((output_dir / "evidence_bundle.json").resolve()),
            "authoritative": True,
            "generation_method": generation_method,
            "coordinator_source": ("coordinator_evidence_bundle" if direct_bundle_attached else None),
            "coordination_status": direct_status.get("coordination_status"),
            "evidence_status": direct_status.get("evidence_status"),
            "all_selected_profilers_valid": direct_status.get("all_selected_profilers_valid"),
            "profile_sets": list(profile.profile_sets),
            "sha256_manifest": str((output_dir / "unified_profile_artifacts.json").resolve()),
        }
        # Declare paths before the immutable pipeline report is written. The
        # read-only evidence builder hashes that report as raw evidence, so
        # rewriting it afterward would invalidate its own retained digest.
        report["unified_profile_artifacts"] = {
            "evidence_bundle": str((output_dir / "evidence_bundle.json").resolve()),
            "view_model": str((output_dir / "view_model.json").resolve()),
            "html": str((output_dir / "unified_profile.html").resolve()),
            "svg": str((output_dir / "unified_profile.svg").resolve()),
            "animated_svg": str((output_dir / "unified_profile.animated.svg").resolve()),
            "manifest": str((output_dir / "unified_profile_artifacts.json").resolve()),
            "workload_executions": 1,
            "generation_method": generation_method,
        }
    (output_dir / "pipeline_report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    if coordinated:
        # Reporting consumes retained artifacts only. It cannot invoke the
        # adapter command, which is guarded independently by replay fixtures.
        from ..reporting.unified_view import write_unified_view_artifacts
        from .evidence import bundle_sha256

        bundle_path = output_dir / "evidence_bundle.json"
        try:
            if direct_bundle_attached:
                from .evidence_augmentation import augment_coordinator_bundle

                bundle = augment_coordinator_bundle(
                    output_dir / "pipeline_report.json",
                    output_path=bundle_path,
                )
            else:
                from .evidence_projection import build_evidence_bundle

                bundle = build_evidence_bundle(output_dir / "pipeline_report.json", bundle_path)
            rendered = write_unified_view_artifacts(bundle, output_dir)
        except (OSError, ValueError) as exc:
            raise ProfileError("profile completed but unified evidence rendering failed: " f"{exc}") from exc
        artifact_manifest = {
            "schema_version": "tasktrace-unified-profile-artifacts.v1",
            "run_id": report.get("run_id"),
            "workload_executions": 1,
            "evidence_bundle": str(bundle_path.resolve()),
            "evidence_bundle_sha256": bundle_sha256(bundle),
            "pipeline_report_sha256": hashlib.sha256((output_dir / "pipeline_report.json").read_bytes()).hexdigest(),
            "generation_method": generation_method,
            **rendered,
        }
        (output_dir / "unified_profile_artifacts.json").write_text(
            json.dumps(artifact_manifest, indent=2, sort_keys=True) + "\n"
        )
    return report
