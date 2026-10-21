# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Augment one authoritative coordinator bundle with retained replay evidence.

The coordinator bundle remains authoritative for scope, lifecycle, profiler
status, and execution provenance.  This module performs no workload work: it
only verifies retained files, rebases their artifact paths to the pipeline
directory, and adds replay actions on a proven clock mapping.  Compatibility
projection remains a fallback for reports that have no coordinator bundle.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from ..timebase import monotonic_mapping_from_references
from .evidence import (
    ClockMapping,
    ClockReference,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceStream,
    ProfilerManifest,
    RawArtifact,
    bundle_sha256,
    read_bundle,
    validate_raw_artifacts,
    write_bundle,
)

AUGMENTATION_SCHEMA_VERSION = "tasktrace-coordinator-replay-augmentation.v1"
REPLAY_PROFILER_ID = "replay-actions"
REPLAY_CLOCK_ID = "replay-monotonic"
REPLAY_UNTIMED_STREAM_ID = "replay-action-facts"
PHASE_PROFILER_ID = "labeled-phases"
PHASE_CLOCK_ID = "phase-monotonic"

_UNTIMED_REPLAY_STATUSES = frozenset({"ignored_metadata", "skipped_not_executed"})
_REPLAY_ACTION_VALUE_FIELDS = (
    "action_index",
    "step_id",
    "tool",
    "source_tool",
    "command",
    "source_status",
    "source_returncode",
    "status",
    "returncode",
    "outcome_matches_source",
    "command_executed",
    "source_argument_parse_failed",
    "launch_failed",
    "completion_timing_drift",
    "session_id",
)


class EvidenceAugmentationError(ValueError):
    """Retained inputs cannot justify one authoritative augmented bundle."""


def _read_object(path: Path, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvidenceAugmentationError(f"unable to read {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EvidenceAugmentationError(f"{description} must be a JSON object: {path}")
    return value


def _inside(root: Path, value: str | Path, description: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise EvidenceAugmentationError(f"{description} is outside the pipeline output tree: {path}") from exc
    return path


def _artifact(
    *,
    artifact_id: str,
    path: Path,
    root: Path,
    role: str,
    media_type: str = "application/json",
) -> RawArtifact:
    if not path.is_file():
        raise EvidenceAugmentationError(f"required retained artifact is missing: {path}")
    return RawArtifact.from_path(
        artifact_id=artifact_id,
        path=path,
        base_dir=root,
        media_type=media_type,
        provenance={
            "augmentation_role": role,
            "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
        },
    )


def _rebase_artifact(artifact: RawArtifact, *, source_dir: Path, root: Path) -> RawArtifact:
    source_prefix = source_dir.resolve().relative_to(root.resolve())
    relative = PurePosixPath(source_prefix.as_posix()) / artifact.relative_path
    return RawArtifact(
        artifact_id=artifact.artifact_id,
        relative_path=relative.as_posix(),
        sha256=artifact.sha256,
        size_bytes=artifact.size_bytes,
        media_type=artifact.media_type,
        provenance=dict(artifact.provenance),
    )


def _canonical_reference(bundle: EvidenceBundle, source: Mapping[str, Any]) -> dict[str, Any]:
    canonical = next(
        (clock for clock in bundle.clocks if clock.clock_id == bundle.timeline.clock_id),
        None,
    )
    if canonical is not None and canonical.provenance:
        return dict(canonical.provenance)
    value = source.get("canonical_clock_reference")
    if isinstance(value, dict):
        return value
    value = bundle.provenance.get("canonical_clock_reference")
    if isinstance(value, dict):
        return value
    raise EvidenceAugmentationError("coordinator bundle has no canonical monotonic clock reference")


def _replay_clock(
    replay: Mapping[str, Any],
    *,
    bundle: EvidenceBundle,
    source: Mapping[str, Any],
) -> tuple[ClockReference, ClockMapping]:
    reference = replay.get("clock_reference")
    if not isinstance(reference, dict):
        raise EvidenceAugmentationError("replay report has no structured monotonic clock reference")
    canonical = _canonical_reference(bundle, source)
    raw_mapping = monotonic_mapping_from_references(reference, canonical)
    namespace = reference.get("time_namespace")
    boot_id = reference.get("boot_id")
    namespace_identity = (
        f"{namespace}|boot:{boot_id}"
        if isinstance(namespace, str) and namespace and isinstance(boot_id, str) and boot_id
        else None
    )
    clock = ClockReference(
        clock_id=REPLAY_CLOCK_ID,
        clock_name="CLOCK_MONOTONIC",
        resolution_ns=int(reference.get("storage_precision_ns") or 1),
        namespace_identity=namespace_identity,
        provenance=dict(reference),
    )
    common = {
        "source_clock_id": REPLAY_CLOCK_ID,
        "canonical_clock_id": bundle.timeline.clock_id,
        "status": raw_mapping["status"],
        "method": raw_mapping["method"],
        "reasons": tuple(raw_mapping.get("reasons", [])),
        "provenance": dict(raw_mapping.get("provenance", {})),
    }
    if raw_mapping["status"] == "valid":
        mapping = ClockMapping(
            **common,
            scale_numerator=int(raw_mapping["scale_numerator"]),
            scale_denominator=int(raw_mapping["scale_denominator"]),
            offset_ns=int(raw_mapping["offset_ns"]),
            max_error_ns=int(raw_mapping["max_error_ns"]),
        )
    else:
        mapping = ClockMapping(**common)
    return clock, mapping


def _session_log_path(root: Path, value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    prefix = "/tasktrace/"
    if value.startswith(prefix):
        return _inside(root, root / "replay" / value[len(prefix) :], "replay session log")
    path = Path(value)
    if path.is_absolute():
        return None
    return _inside(root, root / "replay" / path, "replay session log")


def _phase_log_path(source: Mapping[str, Any], root: Path) -> Path | None:
    """Resolve an explicitly retained workload-phase log, if one was declared."""
    for name in ("phase_log_path", "labeled_phase_log_path", "phase_log"):
        value = source.get(name)
        if isinstance(value, str) and value:
            return _inside(root, value, "labeled phase log")

    profile_value = source.get("container_profile_report")
    if not isinstance(profile_value, str) or not profile_value:
        return None
    profile_path = _inside(root, profile_value, "container profile report")
    if not profile_path.is_file():
        return None
    profile = _read_object(profile_path, "container profile report")
    for name in ("phase_log_path", "labeled_phase_log_path", "phase_log"):
        value = profile.get(name)
        if isinstance(value, str) and value:
            return _inside(root, value, "labeled phase log")
    return None


def _read_json_lines(path: Path, description: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvidenceAugmentationError(f"declared {description} does not exist: {path}")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceAugmentationError(f"unable to read {description} {path}: {exc}") from exc
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceAugmentationError(
                f"invalid JSON in {description} {path} at line {line_number}: {exc}"
            ) from exc
        if not isinstance(row, dict):
            raise EvidenceAugmentationError(f"{description} {path} line {line_number} is not an object")
        rows.append(row)
    return rows


def _phase_clock(
    reference: Mapping[str, Any],
    *,
    bundle: EvidenceBundle,
    source: Mapping[str, Any],
) -> tuple[ClockReference, ClockMapping]:
    canonical = _canonical_reference(bundle, source)
    raw_mapping = monotonic_mapping_from_references(reference, canonical)
    namespace = reference.get("time_namespace")
    boot_id = reference.get("boot_id")
    namespace_identity = (
        f"{namespace}|boot:{boot_id}"
        if isinstance(namespace, str) and namespace and isinstance(boot_id, str) and boot_id
        else None
    )
    clock = ClockReference(
        clock_id=PHASE_CLOCK_ID,
        clock_name="CLOCK_MONOTONIC",
        resolution_ns=int(reference.get("storage_precision_ns") or 1),
        namespace_identity=namespace_identity,
        provenance=dict(reference),
    )
    common = {
        "source_clock_id": PHASE_CLOCK_ID,
        "canonical_clock_id": bundle.timeline.clock_id,
        "status": raw_mapping["status"],
        "method": raw_mapping["method"],
        "reasons": tuple(raw_mapping.get("reasons", [])),
        "provenance": dict(raw_mapping.get("provenance", {})),
    }
    if raw_mapping["status"] == "valid":
        mapping = ClockMapping(
            **common,
            scale_numerator=int(raw_mapping["scale_numerator"]),
            scale_denominator=int(raw_mapping["scale_denominator"]),
            offset_ns=int(raw_mapping["offset_ns"]),
            max_error_ns=int(raw_mapping["max_error_ns"]),
        )
    else:
        mapping = ClockMapping(**common)
    return clock, mapping


def _phase_evidence(
    *,
    path: Path,
    source: Mapping[str, Any],
    bundle: EvidenceBundle,
    root: Path,
) -> tuple[
    ProfilerManifest,
    EvidenceStream,
    ClockReference,
    ClockMapping,
    RawArtifact,
]:
    rows = _read_json_lines(path, "labeled phase log")
    artifact = _artifact(
        artifact_id="raw-labeled-phase-log",
        path=path,
        root=root,
        role="labeled-phase-log",
        media_type="application/x-ndjson",
    )
    header = next((row for row in rows if row.get("record_type") == "fixture"), None)
    reference = header.get("clock_reference") if isinstance(header, dict) else None
    if not isinstance(reference, dict):
        raise EvidenceAugmentationError("labeled phase log has no fixture clock_reference")
    clock, mapping = _phase_clock(reference, bundle=bundle, source=source)

    open_phases: dict[str, list[dict[str, Any]]] = {}
    intervals: list[tuple[str, dict[str, Any], int, int]] = []
    anomalies: list[str] = []
    for line_number, row in enumerate(rows, start=1):
        record_type = row.get("record_type")
        if record_type not in {"phase_start", "phase_end"}:
            continue
        phase = row.get("phase")
        timestamp_ns = row.get("timestamp_ns")
        if not isinstance(phase, str) or not phase:
            anomalies.append(f"phase row {line_number} has no phase name")
            continue
        if not isinstance(timestamp_ns, int) or isinstance(timestamp_ns, bool) or timestamp_ns < 0:
            anomalies.append(f"phase {phase!r} row {line_number} has an invalid timestamp_ns")
            continue
        if record_type == "phase_start":
            open_phases.setdefault(phase, []).append(row)
            continue
        starts = open_phases.get(phase, [])
        if not starts:
            anomalies.append(f"phase {phase!r} ended without a start")
            continue
        start_row = starts.pop(0)
        start_ns = start_row.get("timestamp_ns")
        if not isinstance(start_ns, int) or isinstance(start_ns, bool):
            anomalies.append(f"phase {phase!r} has an invalid start timestamp")
            continue
        if timestamp_ns <= start_ns:
            anomalies.append(f"phase {phase!r} has a non-positive interval")
            continue
        intervals.append((phase, start_row, start_ns, timestamp_ns))
    for phase, starts in open_phases.items():
        if starts:
            anomalies.append(f"phase {phase!r} never ended")

    if not intervals:
        reason = "labeled phase log contains no complete positive-duration intervals"
        if anomalies:
            reason += ": " + "; ".join(sorted(set(anomalies)))
        stream = EvidenceStream(
            stream_id="labeled-phases",
            profiler_id=PHASE_PROFILER_ID,
            metric="workload.phase",
            shape="interval",
            unit="phase",
            status="missing",
            reasons=(reason,),
            raw_artifact_ids=(artifact.artifact_id,),
            provenance={
                "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
                "source_schema_version": header.get("schema_version"),
            },
            display={"group": "Workload", "label": "Labeled phases"},
        )
        manifest = ProfilerManifest(
            profiler_id=PHASE_PROFILER_ID,
            profiler_version=AUGMENTATION_SCHEMA_VERSION,
            status="missing",
            stream_ids=(stream.stream_id,),
            reasons=stream.reasons,
            provenance={
                "method": "read-only-retained-phase-augmentation",
                "does_not_execute_workload": True,
            },
        )
        return manifest, stream, clock, mapping, artifact

    structural_reasons = tuple(sorted(set(anomalies)))
    if mapping.status == "valid":
        status = "invalid" if structural_reasons else "valid"
        reasons = structural_reasons
    else:
        status = "unalignable"
        reasons = tuple(sorted({*mapping.reasons, *structural_reasons}))
    records: list[EvidenceRecord] = []
    for index, (phase, start_row, start_ns, end_ns) in enumerate(intervals):
        common: dict[str, Any] = {
            "record_id": f"labeled-phase:{index:06d}",
            "run_id": bundle.run_id,
            "scope_id": bundle.scope.scope_id,
            "profiler_id": PHASE_PROFILER_ID,
            "stream_id": "labeled-phases",
            "shape": "interval",
            "status": status,
            "reasons": reasons,
            "value": {
                "phase": phase,
                "attributes": dict(start_row.get("attributes") or {}),
            },
            "source_clock_id": PHASE_CLOCK_ID,
            "native_start_ns": start_ns,
            "native_end_ns": end_ns,
            "raw_artifact_ids": (artifact.artifact_id,),
            "provenance": {"source_schema_version": header.get("schema_version")},
            "attributes": {"lane": "workload-phases"},
        }
        if mapping.status == "valid":
            canonical_start = mapping.to_canonical_ns(start_ns)
            canonical_end = mapping.to_canonical_ns(end_ns)
            if canonical_start < bundle.timeline.start_ns or canonical_end > bundle.timeline.end_ns:
                raise EvidenceAugmentationError(f"labeled phase {phase!r} falls outside the coordinator timeline")
            common.update(
                canonical_start_ns=canonical_start,
                canonical_end_ns=canonical_end,
                alignment_error_ns=mapping.max_error_ns,
            )
        records.append(EvidenceRecord(**common))

    stream = EvidenceStream(
        stream_id="labeled-phases",
        profiler_id=PHASE_PROFILER_ID,
        metric="workload.phase",
        shape="interval",
        unit="phase",
        status=status,
        reasons=reasons,
        records=tuple(records),
        raw_artifact_ids=(artifact.artifact_id,),
        provenance={
            "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
            "source_schema_version": header.get("schema_version"),
            "clock_identity": dict(reference),
        },
        display={"group": "Workload", "label": "Labeled phases"},
    )
    manifest = ProfilerManifest(
        profiler_id=PHASE_PROFILER_ID,
        profiler_version=AUGMENTATION_SCHEMA_VERSION,
        status=status,
        stream_ids=(stream.stream_id,),
        reasons=reasons,
        provenance={
            "method": "read-only-retained-phase-augmentation",
            "does_not_execute_workload": True,
        },
    )
    return manifest, stream, clock, mapping, artifact


def _replay_evidence(
    *,
    replay: Mapping[str, Any],
    replay_path: Path,
    source: Mapping[str, Any],
    bundle: EvidenceBundle,
    root: Path,
) -> tuple[
    ProfilerManifest,
    tuple[EvidenceStream, ...],
    ClockReference,
    ClockMapping,
    tuple[RawArtifact, ...],
]:
    replay_artifact = _artifact(
        artifact_id="raw-replay-report",
        path=replay_path,
        root=root,
        role="replay-report",
    )
    clock, mapping = _replay_clock(replay, bundle=bundle, source=source)
    raw_actions = replay.get("actions")
    if not isinstance(raw_actions, list):
        raise EvidenceAugmentationError("replay report actions must be a list")

    artifacts: dict[str, RawArtifact] = {replay_artifact.artifact_id: replay_artifact}
    timed_records: list[EvidenceRecord] = []
    untimed_records: list[EvidenceRecord] = []
    action_indexes: set[int] = set()
    for position, action in enumerate(raw_actions):
        if not isinstance(action, dict):
            raise EvidenceAugmentationError(f"replay action {position} is not an object")
        action_index = action.get("action_index")
        action_status = action.get("status")
        start_ns = action.get("replay_start_monotonic_ns")
        end_ns = action.get("replay_end_monotonic_ns")
        if (
            not isinstance(action_index, int)
            or isinstance(action_index, bool)
            or action_index <= 0
            or action_index in action_indexes
        ):
            raise EvidenceAugmentationError(f"replay action {position} has an invalid or duplicate action_index")
        if not isinstance(action_status, str) or not action_status.strip():
            raise EvidenceAugmentationError(f"replay action {action_index} has no valid status")
        for field in ("command_executed", "source_argument_parse_failed", "launch_failed"):
            if field in action and not isinstance(action[field], bool):
                raise EvidenceAugmentationError(f"replay action {action_index} {field} must be boolean")
        if "outcome_matches_source" not in action or (
            action["outcome_matches_source"] is not None and not isinstance(action["outcome_matches_source"], bool)
        ):
            raise EvidenceAugmentationError(
                f"replay action {action_index} outcome_matches_source must be boolean or null"
            )
        if action.get("source_argument_parse_failed") is True and (
            action.get("command_executed") is not False or action_status != "failed"
        ):
            raise EvidenceAugmentationError(f"replay action {action_index} parse-failure state is inconsistent")
        if action_status in _UNTIMED_REPLAY_STATUSES and action.get("command_executed") is True:
            raise EvidenceAugmentationError(
                f"replay action {action_index} declares non-execution but command_executed is true"
            )

        has_start = start_ns is not None
        has_end = end_ns is not None
        if has_start != has_end:
            raise EvidenceAugmentationError(f"replay action {action_index} has a partial monotonic interval")
        has_interval = has_start and has_end
        if has_interval and (
            not isinstance(start_ns, int)
            or isinstance(start_ns, bool)
            or not isinstance(end_ns, int)
            or isinstance(end_ns, bool)
            or start_ns < 0
            or end_ns <= start_ns
        ):
            raise EvidenceAugmentationError(f"replay action {action_index} has an invalid monotonic interval")
        if not has_interval and action_status not in _UNTIMED_REPLAY_STATUSES:
            raise EvidenceAugmentationError(f"replay action {action_index} has no complete monotonic interval")
        action_indexes.add(action_index)
        raw_ids = [replay_artifact.artifact_id]
        session_path = _session_log_path(root, action.get("session_log"))
        if session_path is not None:
            session_id = str(action.get("session_id") or session_path.stem)
            artifact_id = "raw-replay-session-" + "".join(
                character if character.isalnum() or character in "_.-" else "-" for character in session_id
            )
            if artifact_id not in artifacts:
                artifacts[artifact_id] = _artifact(
                    artifact_id=artifact_id,
                    path=session_path,
                    root=root,
                    role="replay-session-log",
                    media_type="text/plain",
                )
            raw_ids.append(artifact_id)

        value = {key: action[key] for key in _REPLAY_ACTION_VALUE_FIELDS if key in action}
        common_provenance = {"source_schema_version": replay.get("schema_version")}
        if not has_interval:
            timing_disposition = (
                "metadata-not-executed" if action_status == "ignored_metadata" else "source-declared-not-executed"
            )
            untimed_records.append(
                EvidenceRecord(
                    record_id=f"replay-action-fact:{position:06d}",
                    run_id=bundle.run_id,
                    scope_id=bundle.scope.scope_id,
                    profiler_id=REPLAY_PROFILER_ID,
                    stream_id=REPLAY_UNTIMED_STREAM_ID,
                    shape="scalar",
                    status="valid",
                    value=value,
                    raw_artifact_ids=tuple(raw_ids),
                    provenance={
                        **common_provenance,
                        "timing_disposition": timing_disposition,
                    },
                    attributes={"lane": "replay-action-facts"},
                )
            )
            continue

        assert isinstance(start_ns, int) and isinstance(end_ns, int)
        if mapping.status == "valid":
            canonical_start = mapping.to_canonical_ns(start_ns)
            canonical_end = mapping.to_canonical_ns(end_ns)
            if canonical_start < bundle.timeline.start_ns or canonical_end > bundle.timeline.end_ns:
                raise EvidenceAugmentationError(f"replay action {action_index} falls outside the coordinator timeline")
            record = EvidenceRecord(
                record_id=f"replay-action:{position:06d}",
                run_id=bundle.run_id,
                scope_id=bundle.scope.scope_id,
                profiler_id=REPLAY_PROFILER_ID,
                stream_id="replay-actions",
                shape="interval",
                status="valid",
                value=value,
                source_clock_id=REPLAY_CLOCK_ID,
                native_start_ns=start_ns,
                native_end_ns=end_ns,
                canonical_start_ns=canonical_start,
                canonical_end_ns=canonical_end,
                alignment_error_ns=mapping.max_error_ns,
                raw_artifact_ids=tuple(raw_ids),
                provenance=common_provenance,
                attributes={"lane": "replay-actions"},
            )
        else:
            record = EvidenceRecord(
                record_id=f"replay-action:{position:06d}",
                run_id=bundle.run_id,
                scope_id=bundle.scope.scope_id,
                profiler_id=REPLAY_PROFILER_ID,
                stream_id="replay-actions",
                shape="interval",
                status="unalignable",
                reasons=mapping.reasons,
                value=value,
                source_clock_id=REPLAY_CLOCK_ID,
                native_start_ns=start_ns,
                native_end_ns=end_ns,
                raw_artifact_ids=tuple(raw_ids),
                provenance=common_provenance,
                attributes={"lane": "replay-actions"},
            )
        timed_records.append(record)

    timed_status = "valid" if mapping.status == "valid" else "unalignable"
    timed_reasons = () if timed_status == "valid" else mapping.reasons
    streams: list[EvidenceStream] = []
    if timed_records:
        streams.append(
            EvidenceStream(
                stream_id="replay-actions",
                profiler_id=REPLAY_PROFILER_ID,
                metric="replay.action",
                shape="interval",
                unit="action",
                status=timed_status,
                reasons=timed_reasons,
                records=tuple(timed_records),
                raw_artifact_ids=tuple(sorted(artifacts)),
                provenance={
                    "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
                    "source_schema_version": replay.get("schema_version"),
                    "raw_action_count": len(raw_actions),
                    "timed_action_count": len(timed_records),
                    "untimed_action_count": len(untimed_records),
                },
                display={"group": "Replay", "label": "Replay actions"},
            )
        )
    else:
        streams.append(
            EvidenceStream(
                stream_id="replay-actions",
                profiler_id=REPLAY_PROFILER_ID,
                metric="replay.action",
                shape="interval",
                unit="action",
                status="missing",
                reasons=(
                    "replay report contains no actions with measured intervals; "
                    "non-executing actions remain untimed",
                ),
                raw_artifact_ids=(replay_artifact.artifact_id,),
                provenance={
                    "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
                    "source_schema_version": replay.get("schema_version"),
                    "raw_action_count": len(raw_actions),
                    "timed_action_count": 0,
                    "untimed_action_count": len(untimed_records),
                },
                display={"group": "Replay", "label": "Replay actions"},
            )
        )
    if untimed_records:
        streams.append(
            EvidenceStream(
                stream_id=REPLAY_UNTIMED_STREAM_ID,
                profiler_id=REPLAY_PROFILER_ID,
                metric="replay.action.untimed",
                shape="scalar",
                unit="action",
                status="valid",
                records=tuple(untimed_records),
                raw_artifact_ids=(replay_artifact.artifact_id,),
                provenance={
                    "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
                    "source_schema_version": replay.get("schema_version"),
                    "raw_action_count": len(raw_actions),
                    "timed_action_count": len(timed_records),
                    "untimed_action_count": len(untimed_records),
                    "timing_claim": "none",
                },
                display={
                    "group": "Replay",
                    "label": "Untimed replay action facts",
                },
            )
        )

    if not raw_actions:
        manifest_status = "missing"
        manifest_reasons = ("replay report contains no actions",)
    elif timed_records:
        manifest_status = timed_status
        manifest_reasons = timed_reasons
    else:
        # A replay made only of source-declared non-execution/metadata still has
        # complete outcome evidence; it simply has nothing honest to place on a
        # timeline.  The interval stream above exposes that absence explicitly.
        manifest_status = "valid"
        manifest_reasons = ()
    manifest = ProfilerManifest(
        profiler_id=REPLAY_PROFILER_ID,
        profiler_version=AUGMENTATION_SCHEMA_VERSION,
        status=manifest_status,
        stream_ids=tuple(stream.stream_id for stream in streams),
        reasons=manifest_reasons,
        provenance={
            "method": "read-only-retained-replay-augmentation",
            "does_not_execute_workload": True,
            "raw_action_count": len(raw_actions),
            "timed_action_count": len(timed_records),
            "untimed_action_count": len(untimed_records),
        },
    )
    return manifest, tuple(streams), clock, mapping, tuple(artifacts.values())


def augment_coordinator_bundle(
    report_path: Path,
    *,
    direct_bundle_path: Path | None = None,
    output_path: Path | None = None,
) -> EvidenceBundle:
    """Preserve a verified direct bundle and add retained replay evidence."""
    report_path = report_path.resolve()
    root = report_path.parent
    source = _read_object(report_path, "pipeline report")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise EvidenceAugmentationError("pipeline report has no run_id")

    attachment_field = "coordinator_evidence_bundle"
    attachment = source.get(attachment_field)
    if attachment is None:
        # Backward compatibility for pipeline reports written before the
        # public ``evidence_bundle`` pointer was reserved for the final,
        # augmented bundle.  New reports must keep the coordinator attachment
        # under its explicit name so input and output cannot be confused.
        attachment_field = "evidence_bundle"
        attachment = source.get(attachment_field)
    if not isinstance(attachment, dict):
        raise EvidenceAugmentationError("pipeline report has no direct coordinator evidence attachment")
    declared_path = direct_bundle_path or attachment.get("path")
    if not isinstance(declared_path, (str, Path)):
        raise EvidenceAugmentationError("direct coordinator bundle path is absent")
    direct_path = _inside(root, declared_path, "direct coordinator bundle")
    if not direct_path.is_file():
        raise EvidenceAugmentationError(f"declared direct coordinator bundle does not exist: {direct_path}")
    direct = read_bundle(direct_path)
    if direct.run_id != run_id:
        raise EvidenceAugmentationError("pipeline report and direct coordinator bundle have different run IDs")
    expected_digest = attachment.get("sha256")
    actual_digest = bundle_sha256(direct)
    file_digest = hashlib.sha256(direct_path.read_bytes()).hexdigest()
    if not isinstance(expected_digest, str) or expected_digest != actual_digest:
        raise EvidenceAugmentationError("direct coordinator bundle digest does not match its attachment")
    if file_digest != actual_digest:
        raise EvidenceAugmentationError("direct coordinator bundle file is not its canonical hashed serialization")
    validate_raw_artifacts(direct, direct_path.parent)

    replay_value = source.get("replay_report")
    if not isinstance(replay_value, str) or not replay_value:
        raise EvidenceAugmentationError("pipeline report has no replay report path")
    replay_path = _inside(root, replay_value, "replay report")
    replay = _read_object(replay_path, "replay report")
    manifest, replay_streams, replay_clock, replay_mapping, replay_artifacts = _replay_evidence(
        replay=replay,
        replay_path=replay_path,
        source=source,
        bundle=direct,
        root=root,
    )
    phase_path = _phase_log_path(source, root)
    phase_evidence = (
        _phase_evidence(
            path=phase_path,
            source=source,
            bundle=direct,
            root=root,
        )
        if phase_path is not None
        else None
    )

    artifacts = [_rebase_artifact(artifact, source_dir=direct_path.parent, root=root) for artifact in direct.artifacts]
    artifacts.extend(replay_artifacts)
    if phase_evidence is not None:
        artifacts.append(phase_evidence[4])
    artifacts.append(
        _artifact(
            artifact_id="raw-direct-evidence-bundle",
            path=direct_path,
            root=root,
            role="authoritative-coordinator-bundle",
        )
    )
    artifacts.append(
        _artifact(
            artifact_id="raw-source-report",
            path=report_path,
            root=root,
            role="pipeline-report",
        )
    )
    profile_value = source.get("container_profile_report")
    if isinstance(profile_value, str) and profile_value:
        profile_path = _inside(root, profile_value, "container profile report")
        artifacts.append(
            _artifact(
                artifact_id="raw-container-profile-report",
                path=profile_path,
                root=root,
                role="container-profile-report",
            )
        )

    artifact_ids = [artifact.artifact_id for artifact in artifacts]
    if len(artifact_ids) != len(set(artifact_ids)):
        raise EvidenceAugmentationError("augmented bundle has duplicate artifact IDs")
    clock_ids = {clock.clock_id for clock in direct.clocks}
    mapping_ids = {mapping.source_clock_id for mapping in direct.clock_mappings}
    if replay_clock.clock_id in clock_ids or replay_mapping.source_clock_id in mapping_ids:
        raise EvidenceAugmentationError("direct bundle already defines replay clock evidence")
    profiler_ids = {profiler.profiler_id for profiler in direct.profilers}
    stream_ids = {existing.stream_id for existing in direct.streams}
    replay_stream_ids = {stream.stream_id for stream in replay_streams}
    if manifest.profiler_id in profiler_ids or replay_stream_ids & stream_ids:
        raise EvidenceAugmentationError("direct bundle already defines replay action evidence")
    added_clocks = [replay_clock]
    added_mappings = [replay_mapping]
    added_profilers = [manifest]
    added_streams = list(replay_streams)
    augmentation_statuses = {REPLAY_PROFILER_ID: manifest.status}
    if phase_evidence is not None:
        (
            phase_manifest,
            phase_stream,
            phase_clock,
            phase_mapping,
            _phase_artifact,
        ) = phase_evidence
        if phase_clock.clock_id in clock_ids or phase_mapping.source_clock_id in mapping_ids:
            raise EvidenceAugmentationError("direct bundle already defines labeled phase clock evidence")
        if phase_manifest.profiler_id in profiler_ids or phase_stream.stream_id in stream_ids:
            raise EvidenceAugmentationError("direct bundle already defines labeled phase evidence")
        added_clocks.append(phase_clock)
        added_mappings.append(phase_mapping)
        added_profilers.append(phase_manifest)
        added_streams.append(phase_stream)
        augmentation_statuses[PHASE_PROFILER_ID] = phase_manifest.status

    coordinator_evidence_status = direct.extensions.get("evidence_status")
    added_statuses = set(augmentation_statuses.values())
    if coordinator_evidence_status == "invalid" or added_statuses & {
        "invalid",
        "dropped",
    }:
        combined_evidence_status = "invalid"
    elif coordinator_evidence_status == "partial" or added_statuses & {
        "missing",
        "unalignable",
    }:
        combined_evidence_status = "partial"
    else:
        combined_evidence_status = "valid"

    augmented = replace(
        direct,
        clocks=(*direct.clocks, *added_clocks),
        clock_mappings=(*direct.clock_mappings, *added_mappings),
        profilers=(*direct.profilers, *added_profilers),
        streams=(*direct.streams, *added_streams),
        artifacts=tuple(artifacts),
        provenance={
            **dict(direct.provenance),
            "pipeline_augmentation": {
                "schema_version": AUGMENTATION_SCHEMA_VERSION,
                "method": "verified-direct-bundle-plus-retained-replay",
                "does_not_execute_workload": True,
                "workload_executions_added": 0,
                "source_report": report_path.name,
                "source_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
                "source_attachment_field": attachment_field,
                "direct_bundle": direct_path.relative_to(root).as_posix(),
                "direct_bundle_sha256": actual_digest,
                "phase_log": (phase_path.relative_to(root).as_posix() if phase_path is not None else None),
            },
        },
        extensions={
            **dict(direct.extensions),
            "augmentation_schema_version": AUGMENTATION_SCHEMA_VERSION,
            "augmentation_profiler_statuses": augmentation_statuses,
            "coordinator_evidence_status": coordinator_evidence_status,
            "evidence_status": combined_evidence_status,
            "all_bundle_evidence_valid": combined_evidence_status == "valid",
        },
    )
    validate_raw_artifacts(augmented, root)
    if output_path is not None:
        output_path = output_path.resolve()
        if output_path.parent != root:
            raise EvidenceAugmentationError("augmented bundle must be written beside the pipeline report")
        if output_path == direct_path or output_path == report_path:
            raise EvidenceAugmentationError("augmented bundle must not overwrite a consumed source artifact")
        write_bundle(augmented, output_path)
    return augmented


__all__ = [
    "AUGMENTATION_SCHEMA_VERSION",
    "EvidenceAugmentationError",
    "augment_coordinator_bundle",
]
