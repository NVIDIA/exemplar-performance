# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from offline_replayer.measurement import pipeline
from offline_replayer.measurement.container import ProfileError
from offline_replayer.measurement.network_policy import NetworkPolicy
from offline_replayer.measurement.pipeline import (
    ContainerMeasurementOptions,
    run_profiled_replay,
)


def _options() -> ContainerMeasurementOptions:
    return ContainerMeasurementOptions(
        image="task:local",
        cpu_limit=None,
        cpuset_cpus="16-23",
        allocated_cpus=None,
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        workdir="/app",
        scheduler_concurrency=True,
        scheduler_backend="bcc",
    )


def test_options_construct_generic_container_profile(tmp_path: Path) -> None:
    profile = replace(
        _options(),
        memory_profile=True,
        memory_limit_bytes=1024,
        memory_swap_total_limit_bytes=1024,
        profile_sets=("cpu-compare",),
    ).build_profile(
        command="python3 -m replay",
        prepare_command=None,
        output_dir=tmp_path / "container_profile",
        mounts=("type=bind,src=/src,dst=/source,readonly",),
        workload_identity={"task": "task-42"},
        source_identity={"trace_sha256": "a" * 64},
        expected_outcome={"kind": "source-outcomes"},
    )

    assert profile.image == "task:local"
    assert profile.cpuset_cpus == "16-23"
    assert profile.command == "python3 -m replay"
    assert profile.scheduler_concurrency is True
    assert profile.terminate_descendants_at_finish is False
    assert profile.memory_profile is True
    assert profile.memory_limit_bytes == 1024
    assert profile.memory_swap_total_limit_bytes == 1024
    assert profile.profile_sets == ("cpu-compare",)
    assert profile.workload_identity == {"task": "task-42"}
    assert profile.source_identity == {"trace_sha256": "a" * 64}
    assert profile.output_dir == tmp_path / "container_profile"


def test_options_construct_explicit_coordinated_baseline(tmp_path: Path) -> None:
    profile = replace(
        _options(),
        scheduler_concurrency=False,
        coordinated_baseline=True,
    ).build_profile(
        command="python3 -m replay",
        prepare_command=None,
        output_dir=tmp_path / "container_profile",
        mounts=(),
    )

    assert profile.coordinated_baseline is True
    assert profile.profile_sets == ()
    assert profile.scheduler_concurrency is False


def test_options_propagate_typed_network_policy_configuration(
    tmp_path: Path,
) -> None:
    policy = NetworkPolicy("allowlist", ("pypi.org",))
    profile = replace(
        _options(),
        network_policy=policy,
        network_policy_proxy_image="proxy:test",
        network_policy_external_network="fixture-egress",
    ).build_profile(
        command="true",
        prepare_command=None,
        output_dir=tmp_path / "container_profile",
        mounts=(),
    )

    assert profile.network_policy == policy
    assert profile.network_policy_proxy_image == "proxy:test"
    assert profile.network_policy_external_network == "fixture-egress"
    assert profile.network is None


def test_options_allow_no_network_mode_for_enroot(tmp_path: Path) -> None:
    profile = replace(
        _options(),
        runtime="enroot",
        network="none",
    ).build_profile(
        command="true",
        prepare_command=None,
        output_dir=tmp_path / "container_profile",
        mounts=(),
    )

    assert profile.runtime == "enroot"
    assert profile.network == "none"


def test_run_profiled_replay_writes_stable_pipeline_envelope(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "result"
    replay_report = output_dir / "replay" / "replay_report.json"
    replay_report.parent.mkdir(parents=True)
    replay_report.write_text(json.dumps({"summary": {"actions_ok": 1}}))
    profile = _options().build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    container_report = {
        "run_id": "run-123",
        "profile_started_at_utc": "2026-06-30T12:00:00+00:00",
        "image_identity": {"id": "sha256:image"},
        "cpu": {"usage_sec": 1.0},
        "scheduler_concurrency": {"peak_simultaneous_logical_cores": 2},
        "proc_stat_comparison": None,
        "strace": {"mode": "none"},
        "warning": None,
        "canonical_clock_reference": {"clock_id": "CLOCK_MONOTONIC"},
        "canonical_lifecycle": {"clock_id": "host-monotonic"},
        "evidence_bundle": {
            "path": str(output_dir / "container_profile" / "evidence_bundle.json"),
            "execution_plan_digest": "b" * 64,
            "scope": {"scope_id": "scope-1"},
        },
        "hardware_perf": {"all_requested_metrics_valid": True},
        "enroot_lifecycle": {"schema_version": "tasktrace-enroot-lifecycle.v1"},
    }
    monkeypatch.setattr(pipeline, "run_profile", lambda _profile: container_report)

    report = run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1", "image": "task:local"},
    )

    assert report["schema_version"] == "format.v1"
    assert report["run_id"] == "run-123"
    assert report["profile_started_at_utc"] == "2026-06-30T12:00:00+00:00"
    assert report["image_identity"] == {"id": "sha256:image"}
    assert report["replay_summary"] == {"actions_ok": 1}
    assert report["container_cpu"] == {"usage_sec": 1.0}
    assert report["execution_plan_digest"] == "b" * 64
    assert report["scope"] == {"scope_id": "scope-1"}
    assert report["hardware_perf"]["all_requested_metrics_valid"] is True
    assert report["enroot_lifecycle"] == {"schema_version": "tasktrace-enroot-lifecycle.v1"}
    assert report["status"] == "valid"
    assert json.loads((output_dir / "pipeline_report.json").read_text()) == report


def test_run_profiled_replay_forwards_cooperative_cancellation_callback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "result"
    replay_report = output_dir / "replay" / "replay_report.json"
    replay_report.parent.mkdir(parents=True)
    replay_report.write_text(json.dumps({"summary": {"actions_ok": 1}}))
    profile = _options().build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    captured = []

    def cancel_requested():
        return False

    def fake_run_profile(received_profile, *, cancel_requested=None):
        captured.append((received_profile, cancel_requested))
        return {
            "image_identity": {"id": "sha256:image"},
            "cpu": {"usage_sec": 1.0},
            "scheduler_concurrency": None,
            "proc_stat_comparison": None,
            "strace": {"mode": "none"},
            "warning": None,
        }

    monkeypatch.setattr(pipeline, "run_profile", fake_run_profile)

    run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1"},
        cancel_requested=cancel_requested,
    )

    assert len(captured) == 1
    assert captured[0][0].render_unified_view is False
    assert captured[0][1] is cancel_requested


def test_current_container_invalid_measurement_invalidates_pipeline_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "result"
    replay_report = output_dir / "replay" / "replay_report.json"
    replay_report.parent.mkdir(parents=True)
    replay_report.write_text(
        json.dumps(
            {
                "summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                }
            }
        )
    )
    profile = replace(
        _options(),
        image=None,
        cpuset_cpus=None,
        scheduler_concurrency=False,
        runtime="current",
        current_container_exclusive=False,
        workdir=str(tmp_path),
    ).build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    measurement_validity = {
        "valid_for_average_cpu_reporting": False,
        "exclusive_single_task_container_asserted": False,
        "cgroup_path_stable": True,
        "allocation_stable": True,
    }
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "runtime": "current",
            "scope": "current-container",
            "environment_identity": {"hostname": "sandbox"},
            "task_environment": {"unset_variables": [], "overridden_variables": []},
            "measurement_validity": measurement_validity,
            "image_identity": None,
            "cpu": {"average_logical_cores_used": 1.0},
            "scheduler_concurrency": None,
            "proc_stat_comparison": None,
            "strace": {"mode": "none"},
            "warning": "exclusive sandbox required",
        },
    )

    report = run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1"},
    )

    assert report["measurement_validity"] == measurement_validity
    assert report["status"] == "invalid"
    assert report["profiling_valid"] is False
    assert report["profiling_invalid_reasons"] == [
        {
            "profiler_id": "current-container-cgroup-cpu",
            "status": "invalid",
            "reasons": ["current-container measurement is not valid for average CPU reporting"],
        }
    ]
    assert json.loads((output_dir / "pipeline_report.json").read_text()) == report


def test_pipeline_propagates_network_policy_report_without_losing_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "result"
    replay_report = output_dir / "replay" / "replay_report.json"
    replay_report.parent.mkdir(parents=True)
    replay_report.write_text(json.dumps({"summary": {"actions_ok": 1}}))
    policy = NetworkPolicy("allowlist", ("pypi.org",))
    profile = replace(
        _options(),
        network_policy=policy,
        network_policy_proxy_image="proxy:test",
        network_policy_external_network="fixture-egress",
    ).build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    network_report = {
        **policy.to_report_payload(),
        "comparability_sha256": policy.comparability_digest,
        "enforcement": {
            "state": "closed",
            "ephemeral_internal_network": "tasktrace-policy-runtime-only",
            "comparability": {
                "schema_version": ("tasktrace-docker-network-enforcement-comparability.v1"),
                "policy": policy.to_comparability_payload(),
                "backend": "docker-forward-proxy-v1",
                "allowed_ports": [80, 443],
                "external_network": "fixture-egress",
                "proxy_source_sha256": "a" * 64,
                "proxy_image_identity": {"id": "sha256:" + "b" * 64},
            },
        },
    }
    captured_profiles = []

    def fake_run_profile(received_profile):
        captured_profiles.append(received_profile)
        return {
            "image_identity": {"id": "sha256:image"},
            "cpu": {},
            "strace": {"mode": "none"},
            "warning": None,
            "network": "tasktrace-allowlist",
            "network_policy": network_report,
        }

    monkeypatch.setattr(pipeline, "run_profile", fake_run_profile)

    report = run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1"},
    )

    assert captured_profiles[0].network_policy == policy
    assert captured_profiles[0].network_policy_proxy_image == "proxy:test"
    assert captured_profiles[0].network_policy_external_network == "fixture-egress"
    assert report["network"] == "tasktrace-allowlist"
    assert report["network_policy"] == network_report
    assert report["network_policy"]["enforcement"]["comparability"]["allowed_ports"] == [80, 443]
    assert json.loads((output_dir / "pipeline_report.json").read_text())["network_policy"] == network_report


def test_run_profiled_replay_propagates_optional_memory_summary(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "result"
    replay_report = output_dir / "replay" / "replay_report.json"
    replay_report.parent.mkdir(parents=True)
    replay_report.write_text(json.dumps({"summary": {"actions_ok": 1}}))
    profile = replace(_options(), memory_profile=True).build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    memory = {
        "schema_version": "container-cgroup-memory.v2",
        "peak_valid": True,
        "natural_peak_trustworthy": True,
        "lifetime": {"peak_bytes": 1234},
    }
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "image_identity": None,
            "cpu": {},
            "memory": memory,
            "strace": {},
            "warning": None,
        },
    )

    report = run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1"},
    )

    assert profile.memory_profile is True
    assert report["container_memory"]["lifetime"] == memory["lifetime"]
    assert report["container_memory"]["controller_source_sha256"] == pipeline.controller_source_sha256()
    assert len(report["container_memory"]["controller_source_sha256"]) == 64
    assert "controller_source_sha256" not in report


def test_pipeline_propagates_failed_workload_verification_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output_dir = tmp_path / "result"
    replay_report = output_dir / "replay" / "replay_report.json"
    replay_report.parent.mkdir(parents=True)
    replay_report.write_text(
        json.dumps(
            {
                "summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                }
            }
        )
    )
    profile = _options().build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    verification = {
        "requested": True,
        "status": "failed",
        "valid": False,
        "exit_code": 7,
        "reasons": ["verification command exited with status 7"],
    }
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "image_identity": None,
            "cpu": {},
            "strace": {},
            "warning": None,
            "verification": verification,
            "profiling_valid": False,
            "profiling_invalid_reasons": [
                {
                    "profiler_id": "workload-verification",
                    "status": "invalid",
                    "reasons": verification["reasons"],
                }
            ],
        },
    )

    report = run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1"},
    )

    assert report["verification"] == verification
    assert report["profiling_valid"] is False
    assert report["status"] == "invalid"


def test_run_profiled_replay_surfaces_inner_replay_stderr(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "result"
    stderr_path = output_dir / "container_profile" / "command.stderr.txt"
    stderr_path.parent.mkdir(parents=True)
    stderr_path.write_text("python3: missing module")
    profile = _options().build_profile(
        command="false",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "cpu": {},
            "strace": {},
            "warning": None,
        },
    )

    with pytest.raises(ProfileError, match="python3: missing module"):
        run_profiled_replay(
            profile=profile,
            output_dir=output_dir,
            replay_report_relative_path=Path("replay/replay_report.json"),
            report_fields={"schema_version": "format.v1"},
        )

    failure_report = json.loads((output_dir / "pipeline_report.json").read_text())
    assert failure_report["pipeline_status"] == "failed_missing_replay_report"
    assert failure_report["replay_summary"] is None
    assert "python3: missing module" in failure_report["pipeline_error"]


def test_missing_replay_report_preserves_oom_memory_evidence(tmp_path: Path, monkeypatch) -> None:
    output_dir = tmp_path / "result"
    profile = _options().build_profile(
        command="false",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    memory = {
        "censoring": "oom",
        "peak_valid": True,
        "natural_peak_trustworthy": False,
        "natural_peak_invalid_reasons": ["memory demand is constrained by oom"],
        "lifetime": {"peak_bytes": 1234},
    }
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "command_exit_code": 137,
            "cpu": {},
            "memory": memory,
            "strace": {},
            "warning": None,
        },
    )

    with pytest.raises(ProfileError, match="without a replay report"):
        run_profiled_replay(
            profile=profile,
            output_dir=output_dir,
            replay_report_relative_path=Path("replay/replay_report.json"),
            report_fields={"schema_version": "format.v1", "task_name": "task"},
        )

    failure_report = json.loads((output_dir / "pipeline_report.json").read_text())
    assert failure_report["container_command_exit_code"] == 137
    assert failure_report["container_memory"]["censoring"] == "oom"
    assert failure_report["container_memory"]["natural_peak_trustworthy"] is False
    assert failure_report["container_memory"]["lifetime"]["peak_bytes"] == 1234


@pytest.mark.parametrize("unavailable", [1, None, True, "0", -1])
def test_pipeline_rejects_unavailable_source_comparison_without_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unavailable
) -> None:
    output_dir = tmp_path / "result"
    replay_path = output_dir / "replay" / "replay_report.json"
    replay_path.parent.mkdir(parents=True)
    replay_path.write_text(
        json.dumps(
            {
                "summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                    "source_outcome_comparison_unavailable_count": unavailable,
                }
            }
        )
    )
    profile = _options().build_profile(
        command="true",
        prepare_command=None,
        output_dir=output_dir / "container_profile",
        mounts=(),
    )
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "image_identity": None,
            "cpu": {},
            "strace": {},
            "warning": None,
            "profiling_valid": True,
            "command_exit_code": 0,
        },
    )
    report = run_profiled_replay(
        profile=profile,
        output_dir=output_dir,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "format.v1"},
    )
    assert report["status"] == "invalid"
    assert report["replay_summary"]["source_outcome_mismatch_count"] == 0


def test_atif_summary_requires_complete_comparable_tool_replay() -> None:
    valid = {
        "actions_timed_out": 0,
        "actions_skipped_after_timeout": 0,
        "actions_launch_failed": 0,
        "actions_source_argument_parse_failed": 0,
        "source_outcome_mismatch_count": 0,
        "source_outcome_comparison_unavailable_count": 0,
        "tool_actions_recorded": 2,
        "tool_actions_replayed": 2,
        "actions_executed": 2,
    }
    assert pipeline._atif_summary_is_valid(valid)
    for field in valid:
        invalid_values = [None, True, "0", -1]
        invalid_values += [1] if valid[field] == 0 else [0, 1]
        for value in invalid_values:
            assert not pipeline._atif_summary_is_valid({**valid, field: value}), (field, value)
        assert not pipeline._atif_summary_is_valid({key: value for key, value in valid.items() if key != field})
    assert pipeline._atif_summary_is_valid(
        {**valid, "actions_executed": 1, "tool_actions_reproduced_without_execution": 1}
    )
    assert not pipeline._atif_summary_is_valid({**valid, "tool_actions_reproduced_without_execution": None})
