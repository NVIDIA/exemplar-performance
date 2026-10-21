# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.measurement import pipeline
from offline_replayer.measurement.container import ProfileConfig
from offline_replayer.measurement.evidence import (
    CanonicalTimeline,
    ClockReference,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceStream,
    MeasuredScope,
    ProfilerManifest,
    RawArtifact,
    bundle_sha256,
    read_bundle,
    validate_raw_artifacts,
    write_bundle,
)
from offline_replayer.measurement.evidence_augmentation import (
    EvidenceAugmentationError,
    augment_coordinator_bundle,
)
from offline_replayer.measurement.pipeline import run_profiled_replay
from offline_replayer.reporting.unified_view import assemble_view_model

RUN_ID = "11111111-2222-4333-8444-555555555555"
BOOT_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"


def _json(path: Path, value: object) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    return path


def _jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return path


def _direct_bundle(tmp_path: Path) -> tuple[Path, EvidenceBundle]:
    profile_dir = tmp_path / "container_profile"
    raw = profile_dir / "resource_timeseries.json"
    _json(raw, {"samples": [{"timestamp_ns": 1_200_000_000}]})
    artifact = RawArtifact.from_path(
        artifact_id="cgroup-resource-timeseries",
        path=raw,
        base_dir=profile_dir,
        media_type="application/json",
    )
    scope = MeasuredScope(
        scope_id="docker-cgroup-exact",
        kind="disposable-container-cgroup-subtree",
        identity={
            "container_name": "one-container",
            "cgroup_inode": 1234,
            "image_identity": {"id": "sha256:" + "a" * 64},
        },
        includes_descendants=True,
    )
    timeline = CanonicalTimeline(
        clock_id="host-monotonic",
        origin_ns=1_100_000_000,
        start_ns=1_000_000_000,
        end_ns=3_000_000_000,
    )
    cpu_record = EvidenceRecord(
        record_id="cpu:value",
        run_id=RUN_ID,
        scope_id=scope.scope_id,
        profiler_id="cgroup-cpu",
        stream_id="cpu-usage",
        shape="scalar",
        status="valid",
        value=1.25,
        raw_artifact_ids=(artifact.artifact_id,),
    )
    bundle = EvidenceBundle(
        run_id=RUN_ID,
        scope=scope,
        timeline=timeline,
        clocks=(
            ClockReference(
                clock_id="host-monotonic",
                clock_name="CLOCK_MONOTONIC",
                resolution_ns=1,
                namespace_identity=f"time:[1]|boot:{BOOT_ID}",
                provenance={
                    "clock_id": "CLOCK_MONOTONIC",
                    "time_namespace": "time:[1]",
                    "time_namespace_monotonic_offset_ns": 0,
                    "boot_id": BOOT_ID,
                    "storage_precision_ns": 1,
                },
            ),
        ),
        clock_mappings=(),
        profilers=(
            ProfilerManifest(
                profiler_id="cgroup-cpu",
                profiler_version="test.v1",
                status="valid",
                stream_ids=("cpu-usage",),
            ),
            ProfilerManifest(
                profiler_id="scheduler-concurrency",
                profiler_version="test.v1",
                status="invalid",
                reasons=("scheduler requires root privileges",),
            ),
        ),
        streams=(
            EvidenceStream(
                stream_id="cpu-usage",
                profiler_id="cgroup-cpu",
                metric="cpu.usage_seconds",
                shape="scalar",
                unit="second",
                status="valid",
                records=(cpu_record,),
                raw_artifact_ids=(artifact.artifact_id,),
            ),
        ),
        artifacts=(artifact,),
        provenance={
            "execution_plan_sha256": "b" * 64,
            "execution_plan": {
                "workload_identity": {"task": "one"},
                "source_identity": {"sha256": "c" * 64},
            },
            "profile_request": {"profile_sets": ["cpu-compare"]},
        },
        extensions={
            "coordination_status": "valid",
            "evidence_status": "invalid",
            "all_selected_profilers_valid": False,
            "profiler_statuses": {
                "cgroup-cpu": "valid",
                "scheduler-concurrency": "invalid",
            },
        },
    )
    path = profile_dir / "evidence_bundle.json"
    write_bundle(bundle, path)
    return path, bundle


def _pipeline(tmp_path: Path) -> tuple[Path, Path, EvidenceBundle]:
    direct_path, direct = _direct_bundle(tmp_path)
    _json(
        tmp_path / "container_profile" / "profile_report.json",
        {"schema_version": "container-command-profile.v1", "run_id": RUN_ID},
    )
    session_log = tmp_path / "replay" / "session_logs" / "session-7.log"
    session_log.parent.mkdir(parents=True)
    session_log.write_text("build output\n")
    replay = _json(
        tmp_path / "replay" / "replay_report.json",
        {
            "schema_version": "offline-atif-trajectory-replay.v1",
            "clock_reference": {
                "clock_id": "CLOCK_MONOTONIC",
                "time_namespace": "time:[2]",
                "time_namespace_monotonic_offset_ns": 0,
                "boot_id": BOOT_ID,
                "storage_precision_ns": 1,
            },
            "actions": [
                {
                    "action_index": 1,
                    "step_id": 2,
                    "tool": "exec_command",
                    "source_tool": "exec_command",
                    "command": "make -j2",
                    "status": "ok",
                    "returncode": 0,
                    "outcome_matches_source": True,
                    "session_id": "7",
                    "session_log": "/tasktrace/session_logs/session-7.log",
                    "replay_start_monotonic_ns": 1_250_000_000,
                    "replay_end_monotonic_ns": 1_650_000_000,
                }
            ],
            "summary": {"actions_executed": 1},
        },
    )
    phase_log = _jsonl(
        tmp_path / "container_profile" / "phases.jsonl",
        [
            {
                "record_type": "fixture",
                "schema_version": "tasktrace-m1-labeled-phase.v1",
                "clock_reference": {
                    "clock_id": "CLOCK_MONOTONIC",
                    "time_namespace": "time:[3]",
                    "time_namespace_monotonic_offset_ns": 0,
                    "boot_id": BOOT_ID,
                    "storage_precision_ns": 1,
                },
                "timestamp_ns": 1_050_000_000,
            },
            {
                "record_type": "phase_start",
                "phase": "build",
                "timestamp_ns": 1_150_000_000,
                "attributes": {"workers": 2},
            },
            {
                "record_type": "phase_end",
                "phase": "build",
                "timestamp_ns": 1_950_000_000,
            },
        ],
    )
    sentinel = tmp_path / "workload-invocations.txt"
    sentinel.write_text("1\n")
    report = _json(
        tmp_path / "pipeline_report.json",
        {
            "schema_version": "container-atif-trajectory-profile.v1",
            "status": "invalid",
            "run_id": RUN_ID,
            "scope": direct.scope.__dict__,
            "canonical_clock_reference": dict(direct.clocks[0].provenance),
            "container_profile_report": str(tmp_path / "container_profile" / "profile_report.json"),
            "replay_report": str(replay),
            "phase_log_path": str(phase_log),
            "coordinator_evidence_bundle": {
                "path": str(direct_path),
                "sha256": bundle_sha256(direct),
                "evidence_status": "invalid",
            },
            "evidence_bundle": {
                "path": str(tmp_path / "evidence_bundle.json"),
                "authoritative": True,
                "generation_method": ("verified-direct-coordinator-bundle-augmentation"),
                "coordinator_source": "coordinator_evidence_bundle",
            },
        },
    )
    return report, sentinel, direct


def test_augments_direct_bundle_without_losing_authoritative_evidence(
    tmp_path: Path,
) -> None:
    report, sentinel, direct = _pipeline(tmp_path)
    output = tmp_path / "evidence_bundle.json"

    augmented = augment_coordinator_bundle(report, output_path=output)

    assert sentinel.read_text() == "1\n"
    assert read_bundle(output).to_dict() == augmented.to_dict()
    assert augmented.run_id == direct.run_id
    assert augmented.scope == direct.scope
    assert augmented.timeline == direct.timeline
    assert augmented.extensions["coordination_status"] == "valid"
    assert augmented.extensions["evidence_status"] == "invalid"
    assert augmented.extensions["coordinator_evidence_status"] == "invalid"
    assert augmented.extensions["all_bundle_evidence_valid"] is False
    assert augmented.extensions["all_selected_profilers_valid"] is False
    assert augmented.provenance["execution_plan"] == direct.provenance["execution_plan"]
    assert augmented.provenance["profile_request"] == direct.provenance["profile_request"]
    assert augmented.provenance["pipeline_augmentation"]["workload_executions_added"] == 0
    assert augmented.provenance["pipeline_augmentation"]["source_attachment_field"] == "coordinator_evidence_bundle"

    manifests = {item.profiler_id: item for item in augmented.profilers}
    assert manifests["scheduler-concurrency"].status == "invalid"
    assert manifests["scheduler-concurrency"].reasons == ("scheduler requires root privileges",)
    assert manifests["replay-actions"].status == "valid"
    actions = next(stream for stream in augmented.streams if stream.metric == "replay.action")
    assert len(actions.records) == 1
    assert actions.records[0].scope_id == direct.scope.scope_id
    assert actions.records[0].canonical_start_ns == 1_250_000_000
    assert actions.records[0].canonical_end_ns == 1_650_000_000
    assert {mapping.source_clock_id for mapping in augmented.clock_mappings} == {
        "phase-monotonic",
        "replay-monotonic",
    }
    assert manifests["labeled-phases"].status == "valid"
    phases = next(phase_stream for phase_stream in augmented.streams if phase_stream.metric == "workload.phase")
    assert len(phases.records) == 1
    assert phases.records[0].scope_id == direct.scope.scope_id
    assert phases.records[0].value == {
        "phase": "build",
        "attributes": {"workers": 2},
    }
    assert phases.records[0].canonical_start_ns == 1_150_000_000
    assert phases.records[0].canonical_end_ns == 1_950_000_000

    paths = {artifact.relative_path for artifact in augmented.artifacts}
    assert "container_profile/resource_timeseries.json" in paths
    assert "container_profile/evidence_bundle.json" in paths
    assert "container_profile/phases.jsonl" in paths
    assert "container_profile/profile_report.json" in paths
    assert "replay/replay_report.json" in paths
    assert "replay/session_logs/session-7.log" in paths
    assert "pipeline_report.json" in paths
    validate_raw_artifacts(augmented, tmp_path)

    view = assemble_view_model(augmented)
    view_profilers = {item["profiler_id"]: item for item in view["profilers"]}
    assert view_profilers["scheduler-concurrency"]["status"] == "invalid"
    assert view_profilers["scheduler-concurrency"]["reasons"] == ["scheduler requires root privileges"]
    assert any(track["metric"] == "replay.action" for track in view["tracks"])
    assert any(track["metric"] == "workload.phase" for track in view["tracks"])


def test_replay_projection_retains_timed_parse_failure_and_untimed_non_execution(
    tmp_path: Path,
) -> None:
    report, _sentinel, direct = _pipeline(tmp_path)
    pipeline_report = json.loads(report.read_text())
    replay_path = Path(pipeline_report["replay_report"])
    replay = json.loads(replay_path.read_text())
    replay["actions"].extend(
        [
            {
                "action_index": 2,
                "step_id": 3,
                "tool": "update_plan",
                "source_tool": "update_plan",
                "source_status": "ignored_metadata",
                "status": "ignored_metadata",
                "outcome_matches_source": True,
            },
            {
                "action_index": 3,
                "step_id": 4,
                "tool": "exec_command",
                "source_tool": "bash",
                "source_status": "skipped_not_executed",
                "status": "skipped_not_executed",
                "outcome_matches_source": True,
            },
            {
                "action_index": 4,
                "step_id": 5,
                "tool": "exec_command",
                "source_tool": "exec_command",
                "source_status": "failed",
                "source_returncode": None,
                "command": None,
                "status": "failed",
                "returncode": None,
                "outcome_matches_source": True,
                "command_executed": False,
                "source_argument_parse_failed": True,
                "replay_start_monotonic_ns": 1_700_000_000,
                "replay_end_monotonic_ns": 1_710_000_000,
            },
        ]
    )
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")

    augmented = augment_coordinator_bundle(report)

    manifests = {item.profiler_id: item for item in augmented.profilers}
    replay_manifest = manifests["replay-actions"]
    assert replay_manifest.status == "valid"
    assert replay_manifest.stream_ids == (
        "replay-action-facts",
        "replay-actions",
    )
    assert replay_manifest.provenance["raw_action_count"] == 4
    assert replay_manifest.provenance["timed_action_count"] == 2
    assert replay_manifest.provenance["untimed_action_count"] == 2

    streams = {stream.stream_id: stream for stream in augmented.streams}
    timed = streams["replay-actions"]
    assert [record.value["action_index"] for record in timed.records] == [1, 4]
    parse_failure = timed.records[1]
    assert parse_failure.status == "valid"
    assert parse_failure.value["command_executed"] is False
    assert parse_failure.value["source_argument_parse_failed"] is True
    assert parse_failure.value["outcome_matches_source"] is True
    assert parse_failure.canonical_start_ns == 1_700_000_000
    assert parse_failure.canonical_end_ns == 1_710_000_000

    untimed = streams["replay-action-facts"]
    assert untimed.shape == "scalar"
    assert untimed.status == "valid"
    assert [record.value["action_index"] for record in untimed.records] == [2, 3]
    assert [record.value["status"] for record in untimed.records] == [
        "ignored_metadata",
        "skipped_not_executed",
    ]
    assert all(record.source_clock_id is None for record in untimed.records)
    assert all(record.native_start_ns is None for record in untimed.records)
    assert all(record.canonical_start_ns is None for record in untimed.records)
    assert [record.provenance["timing_disposition"] for record in untimed.records] == [
        "metadata-not-executed",
        "source-declared-not-executed",
    ]
    assert all(record.value["outcome_matches_source"] is True for record in untimed.records)
    assert {record.scope_id for record in untimed.records} == {direct.scope.scope_id}

    view = assemble_view_model(augmented)
    assert any(track["stream_id"] == "replay-actions" for track in view["tracks"])
    assert any(summary["stream_id"] == "replay-action-facts" for summary in view["summaries"])
    validate_raw_artifacts(augmented, tmp_path)


def test_replay_projection_with_only_nonexecuting_rows_is_valid_but_untimed(
    tmp_path: Path,
) -> None:
    report, _sentinel, _direct = _pipeline(tmp_path)
    pipeline_report = json.loads(report.read_text())
    replay_path = Path(pipeline_report["replay_report"])
    replay = json.loads(replay_path.read_text())
    replay["actions"] = [
        {
            "action_index": 1,
            "tool": "update_plan",
            "status": "ignored_metadata",
            "outcome_matches_source": True,
        },
        {
            "action_index": 2,
            "tool": "exec_command",
            "status": "skipped_not_executed",
            "outcome_matches_source": True,
        },
    ]
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")

    augmented = augment_coordinator_bundle(report)

    manifest = next(item for item in augmented.profilers if item.profiler_id == "replay-actions")
    streams = {stream.stream_id: stream for stream in augmented.streams if stream.profiler_id == "replay-actions"}
    assert manifest.status == "valid"
    assert streams["replay-actions"].status == "missing"
    assert "remain untimed" in streams["replay-actions"].reasons[0]
    assert streams["replay-action-facts"].status == "valid"
    assert len(streams["replay-action-facts"].records) == 2


@pytest.mark.parametrize(
    ("action", "message"),
    [
        (
            {
                "action_index": 1,
                "status": "ok",
                "outcome_matches_source": True,
            },
            "no complete monotonic interval",
        ),
        (
            {
                "action_index": 1,
                "status": "ignored_metadata",
                "outcome_matches_source": True,
                "replay_start_monotonic_ns": 1_250_000_000,
            },
            "partial monotonic interval",
        ),
        (
            {
                "action_index": 1,
                "status": "ok",
                "outcome_matches_source": True,
                "command_executed": "false",
                "replay_start_monotonic_ns": 1_250_000_000,
                "replay_end_monotonic_ns": 1_650_000_000,
            },
            "command_executed must be boolean",
        ),
        (
            {
                "action_index": 1,
                "status": "failed",
                "outcome_matches_source": True,
                "source_argument_parse_failed": True,
                "replay_start_monotonic_ns": 1_250_000_000,
                "replay_end_monotonic_ns": 1_650_000_000,
            },
            "parse-failure state is inconsistent",
        ),
        (
            {
                "action_index": 1,
                "status": "skipped_not_executed",
                "outcome_matches_source": True,
                "command_executed": True,
            },
            "declares non-execution",
        ),
        (
            {
                "action_index": 1,
                "status": "ok",
                "outcome_matches_source": "yes",
                "replay_start_monotonic_ns": 1_250_000_000,
                "replay_end_monotonic_ns": 1_650_000_000,
            },
            "outcome_matches_source must be boolean or null",
        ),
    ],
)
def test_replay_projection_rejects_malformed_timing_and_outcome_state(
    tmp_path: Path,
    action: dict[str, object],
    message: str,
) -> None:
    report, _sentinel, _direct = _pipeline(tmp_path)
    pipeline_report = json.loads(report.read_text())
    replay_path = Path(pipeline_report["replay_report"])
    replay = json.loads(replay_path.read_text())
    replay["actions"] = [action]
    replay_path.write_text(json.dumps(replay, indent=2, sort_keys=True) + "\n")

    with pytest.raises(EvidenceAugmentationError, match=message):
        augment_coordinator_bundle(report)


def test_direct_bundle_hash_mismatch_fails_closed(tmp_path: Path) -> None:
    report, sentinel, _direct = _pipeline(tmp_path)
    payload = json.loads(report.read_text())
    payload["coordinator_evidence_bundle"]["sha256"] = "0" * 64
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    with pytest.raises(EvidenceAugmentationError, match="digest"):
        augment_coordinator_bundle(report)

    assert sentinel.read_text() == "1\n"


def test_legacy_direct_attachment_remains_supported(tmp_path: Path) -> None:
    report, _sentinel, _direct = _pipeline(tmp_path)
    payload = json.loads(report.read_text())
    payload["evidence_bundle"] = payload.pop("coordinator_evidence_bundle")
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    augmented = augment_coordinator_bundle(report)

    assert augmented.provenance["pipeline_augmentation"]["source_attachment_field"] == "evidence_bundle"


def test_absent_phase_path_adds_no_fabricated_phase_profiler(tmp_path: Path) -> None:
    report, _sentinel, _direct = _pipeline(tmp_path)
    payload = json.loads(report.read_text())
    payload.pop("phase_log_path")
    report.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")

    augmented = augment_coordinator_bundle(report)

    assert "labeled-phases" not in {profiler.profiler_id for profiler in augmented.profilers}
    assert not any(stream.metric == "workload.phase" for stream in augmented.streams)


def test_malformed_declared_phase_log_fails_closed(tmp_path: Path) -> None:
    report, sentinel, _direct = _pipeline(tmp_path)
    payload = json.loads(report.read_text())
    Path(payload["phase_log_path"]).write_text("{not-json\n")

    with pytest.raises(EvidenceAugmentationError, match="invalid JSON"):
        augment_coordinator_bundle(report)

    assert sentinel.read_text() == "1\n"


def test_pipeline_exposes_final_bundle_and_preserves_coordinator_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    report_path, sentinel, direct = _pipeline(tmp_path)
    fixture_report = json.loads(report_path.read_text())
    coordinator_attachment = fixture_report["coordinator_evidence_bundle"]
    profile = ProfileConfig(
        command="true",
        prepare_command=None,
        output_dir=tmp_path / "container_profile",
        image="task:local",
        container=None,
        cpuset_cpus="0",
        cpu_limit=None,
        allocated_cpus=None,
        workdir="/app",
        user=None,
        env=(),
        mounts=(),
        strace_mode="none",
        string_limit=4096,
        shell="/bin/bash",
        profile_sets=("cpu-compare",),
    )
    monkeypatch.setattr(
        pipeline,
        "run_profile",
        lambda _profile: {
            "run_id": RUN_ID,
            "image_identity": {"id": "sha256:" + "a" * 64},
            "cpu": {"usage_sec": 1.0},
            "strace": {"mode": "none"},
            "warning": None,
            "canonical_clock_reference": dict(direct.clocks[0].provenance),
            "canonical_lifecycle": {"clock_id": direct.timeline.clock_id},
            "evidence_bundle": coordinator_attachment,
            "profiling_valid": False,
            "profiling_invalid_reasons": ["scheduler requires root privileges"],
        },
    )

    report = run_profiled_replay(
        profile=profile,
        output_dir=tmp_path,
        replay_report_relative_path=Path("replay/replay_report.json"),
        report_fields={"schema_version": "container-atif-trajectory-profile.v1"},
    )

    persisted = json.loads(report_path.read_text())
    assert persisted == report
    assert persisted["coordinator_evidence_bundle"] == coordinator_attachment
    assert persisted["evidence_bundle"]["path"] == str((tmp_path / "evidence_bundle.json").resolve())
    assert persisted["evidence_bundle"]["authoritative"] is True
    assert persisted["evidence_bundle"]["coordinator_source"] == "coordinator_evidence_bundle"
    assert "sha256" not in persisted["evidence_bundle"]
    assert persisted["evidence_bundle"]["sha256_manifest"] == str(
        (tmp_path / "unified_profile_artifacts.json").resolve()
    )
    manifest = json.loads((tmp_path / "unified_profile_artifacts.json").read_text())
    final_bundle = read_bundle(tmp_path / "evidence_bundle.json")
    assert manifest["evidence_bundle_sha256"] == bundle_sha256(final_bundle)
    assert final_bundle.provenance["pipeline_augmentation"]["source_attachment_field"] == "coordinator_evidence_bundle"
    assert sentinel.read_text() == "1\n"
