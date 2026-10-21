# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read-only projection of retained legacy reports into typed evidence.

The projector is a compatibility boundary, not a profiler.  It reads only the
supplied report and paths explicitly referenced by retained reports, records a
content digest for every consumed file, and never invokes a workload.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..timebase import monotonic_mapping_from_references
from .evidence import (
    AcquisitionBounds,
    Cadence,
    CanonicalTimeline,
    ClockMapping,
    ClockReference,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceStream,
    Gap,
    LifecycleMarker,
    MeasuredScope,
    ProfilerManifest,
    RawArtifact,
    write_bundle,
)

PROJECTION_SCHEMA_VERSION = "tasktrace-retained-report-evidence-projection.v1"
CANONICAL_CLOCK_ID = "host-monotonic"


class EvidenceProjectionError(ValueError):
    """Raised when retained inputs cannot justify a coherent evidence bundle."""


def _object(value: Any) -> dict[str, Any] | None:
    return value if isinstance(value, dict) else None


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _number(value: Any) -> int | float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        return value
    return None


def _integer(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ns_from_seconds(value: Any) -> int | None:
    number = _number(value)
    if number is None or number < 0:
        return None
    return round(float(number) * 1_000_000_000)


def _safe_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9_.-]+", "-", value).strip("-")
    return token or "evidence"


def _unique_reasons(values: Sequence[Any]) -> tuple[str, ...]:
    return tuple(sorted({str(value) for value in values if str(value).strip()}))


@dataclass
class _ArtifactRegistry:
    root: Path

    def __post_init__(self) -> None:
        self.root = self.root.resolve()
        self._by_path: dict[Path, RawArtifact] = {}
        self._ids: set[str] = set()

    def resolve(self, value: Any, *, relative_to: Path | None = None) -> Path | None:
        if not isinstance(value, str) or not value:
            return None
        path = Path(value)
        if not path.is_absolute():
            path = (relative_to or self.root) / path
        path = path.resolve()
        try:
            path.relative_to(self.root)
        except ValueError as exc:
            raise EvidenceProjectionError(f"referenced artifact is outside report tree {self.root}: {path}") from exc
        return path

    def add(
        self,
        path: Path,
        *,
        role: str,
        media_type: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> RawArtifact:
        path = path.resolve()
        if path in self._by_path:
            return self._by_path[path]
        if not path.is_file():
            raise EvidenceProjectionError(f"referenced artifact does not exist: {path}")
        relative = path.relative_to(self.root).as_posix()
        base_id = f"raw-{_safe_token(role)}"
        artifact_id = base_id
        if artifact_id in self._ids:
            suffix = hashlib.sha256(relative.encode()).hexdigest()[:10]
            artifact_id = f"{base_id}-{suffix}"
        artifact = RawArtifact.from_path(
            artifact_id=artifact_id,
            path=path,
            base_dir=self.root,
            media_type=media_type,
            provenance={"projection_role": role, **dict(provenance or {})},
        )
        self._by_path[path] = artifact
        self._ids.add(artifact_id)
        return artifact

    @property
    def artifacts(self) -> tuple[RawArtifact, ...]:
        return tuple(sorted(self._by_path.values(), key=lambda item: item.artifact_id))

    def contains(self, path: Path) -> bool:
        return path.resolve() in self._by_path


def _read_json(path: Path, registry: _ArtifactRegistry, *, role: str) -> tuple[dict[str, Any], RawArtifact]:
    artifact = registry.add(path, role=role, media_type="application/json")
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceProjectionError(f"unable to read retained JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise EvidenceProjectionError(f"retained JSON must be an object: {path}")
    return payload, artifact


def _read_jsonl(path: Path, registry: _ArtifactRegistry, *, role: str) -> tuple[list[dict[str, Any]], RawArtifact]:
    artifact = registry.add(path, role=role, media_type="application/x-ndjson")
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(errors="strict").splitlines()
    except (OSError, UnicodeError) as exc:
        raise EvidenceProjectionError(f"unable to read retained JSONL {path}: {exc}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise EvidenceProjectionError(f"invalid JSONL row {line_number} in {path}: {exc}") from exc
        if not isinstance(row, dict):
            raise EvidenceProjectionError(f"JSONL row {line_number} in {path} is not an object")
        rows.append(row)
    return rows, artifact


def _find_marker(timeline: CanonicalTimeline, name: str) -> int | None:
    for marker in timeline.lifecycle_markers:
        if marker.name == name:
            return marker.canonical_timestamp_ns
    return None


def _clock_identity(reference: Mapping[str, Any] | None) -> tuple[str, str] | None:
    if reference is None:
        return None
    namespace = reference.get("time_namespace")
    boot_id = reference.get("boot_id")
    if isinstance(namespace, str) and namespace and isinstance(boot_id, str) and boot_id:
        return namespace, boot_id
    return None


def _clock_reference(*, clock_id: str, clock_name: str, reference: Mapping[str, Any] | None) -> ClockReference:
    resolution = 1
    if reference is not None:
        candidate = _integer(reference.get("storage_precision_ns"))
        if candidate is not None and candidate > 0:
            resolution = candidate
    identity = _clock_identity(reference)
    return ClockReference(
        clock_id=clock_id,
        clock_name=clock_name,
        resolution_ns=resolution,
        namespace_identity=(f"{identity[0]}|boot:{identity[1]}" if identity else None),
        provenance=dict(reference or {}),
    )


def _parse_canonical_lifecycle(
    source: Mapping[str, Any],
) -> tuple[CanonicalTimeline, Mapping[str, Any] | None] | None:
    lifecycle = _object(source.get("canonical_lifecycle"))
    if lifecycle is None:
        return None
    clock_name = lifecycle.get("clock_name", "CLOCK_MONOTONIC")
    if clock_name != "CLOCK_MONOTONIC":
        raise EvidenceProjectionError("canonical_lifecycle must use CLOCK_MONOTONIC")
    clock_id = lifecycle.get("clock_id", CANONICAL_CLOCK_ID)
    start_ns = _integer(lifecycle.get("start_ns"))
    end_ns = _integer(lifecycle.get("end_ns"))
    origin_ns = _integer(lifecycle.get("origin_ns"))
    if not isinstance(clock_id, str) or start_ns is None or end_ns is None:
        raise EvidenceProjectionError("canonical_lifecycle lacks clock/start/end identity")
    raw_markers = lifecycle.get("lifecycle_markers", lifecycle.get("markers", []))
    markers: list[LifecycleMarker] = []
    if not isinstance(raw_markers, list):
        raise EvidenceProjectionError("canonical_lifecycle markers must be a list")
    for row in raw_markers:
        if not isinstance(row, dict):
            continue
        name = row.get("name")
        timestamp = _integer(row.get("canonical_timestamp_ns", row.get("timestamp_ns")))
        error = _integer(row.get("alignment_error_ns")) or 0
        if isinstance(name, str) and name and timestamp is not None:
            markers.append(
                LifecycleMarker(
                    name=name,
                    canonical_timestamp_ns=timestamp,
                    alignment_error_ns=error,
                    provenance=dict(_object(row.get("provenance")) or {}),
                )
            )
    if origin_ns is None:
        origin_ns = next(
            (
                marker.canonical_timestamp_ns
                for marker in markers
                if marker.name in {"measured_work_start", "command_start"}
            ),
            start_ns,
        )
    timeline = CanonicalTimeline(
        clock_id=clock_id,
        origin_ns=origin_ns,
        start_ns=start_ns,
        end_ns=end_ns,
        lifecycle_markers=tuple(markers),
    )
    reference = _object(lifecycle.get("clock_reference")) or _object(source.get("canonical_clock_reference"))
    return timeline, reference


def _fallback_timeline(
    *,
    timeseries: Mapping[str, Any] | None,
    scheduler: Mapping[str, Any] | None,
    hardware: Mapping[str, Any] | None,
) -> CanonicalTimeline:
    if timeseries is not None:
        samples = [row for row in _list(timeseries.get("samples")) if isinstance(row, dict)]
        starts = [
            value
            for row in samples
            if (value := _integer(row.get("acquisition_start_ns", row.get("timestamp_ns")))) is not None
        ]
        ends = [
            value
            for row in samples
            if (value := _integer(row.get("acquisition_end_ns", row.get("timestamp_ns")))) is not None
        ]
        markers: list[LifecycleMarker] = []
        seen: set[str] = set()
        for row in _list(timeseries.get("markers")):
            if not isinstance(row, dict):
                continue
            name = row.get("name")
            timestamp = _integer(row.get("timestamp_ns"))
            if isinstance(name, str) and name and name not in seen and timestamp is not None:
                markers.append(LifecycleMarker(name, timestamp, 0, {"source": "resource_timeseries"}))
                seen.add(name)
        if starts and ends:
            start_ns, end_ns = min(starts), max(ends)
            if end_ns > start_ns:
                origin = next(
                    (
                        marker.canonical_timestamp_ns
                        for marker in markers
                        if marker.name in {"measured_work_start", "command_start"}
                    ),
                    start_ns,
                )
                return CanonicalTimeline(
                    clock_id=CANONICAL_CLOCK_ID,
                    origin_ns=origin,
                    start_ns=start_ns,
                    end_ns=end_ns,
                    lifecycle_markers=tuple(markers),
                )
    for report, source in ((scheduler, "scheduler"), (hardware, "hardware_perf")):
        if report is None:
            continue
        start_ns = _integer(report.get("window_start_ns"))
        end_ns = _integer(report.get("window_end_ns"))
        if start_ns is not None and end_ns is not None and end_ns > start_ns:
            return CanonicalTimeline(
                clock_id=CANONICAL_CLOCK_ID,
                origin_ns=start_ns,
                start_ns=start_ns,
                end_ns=end_ns,
                lifecycle_markers=(
                    LifecycleMarker("measured_work_start", start_ns, 0, {"source": source}),
                    LifecycleMarker("measured_work_end", end_ns, 0, {"source": source}),
                ),
            )
    raise EvidenceProjectionError("report has no canonical_lifecycle and no bounded resource/scheduler window")


def _scope_from_source(
    source: Mapping[str, Any],
    *,
    run_id: str,
    timeseries: Mapping[str, Any] | None,
) -> MeasuredScope:
    explicit = _object(source.get("measured_scope"))
    if explicit is not None:
        return MeasuredScope.from_dict(explicit)
    identity: dict[str, Any] = {
        "projection": "legacy-report-derived",
        "runtime": source.get("runtime", "unknown"),
        "report_scope": source.get("scope", "disposable-image-container"),
    }
    if source.get("container") is not None:
        identity["container"] = source["container"]
    if source.get("image_identity") is not None:
        identity["image_identity"] = source["image_identity"]
    if timeseries is not None and timeseries.get("scope") is not None:
        identity["cgroup_path"] = timeseries["scope"]
    digest = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":"), default=str).encode() + run_id.encode()
    ).hexdigest()[:20]
    return MeasuredScope(
        scope_id=f"compat-scope:{digest}",
        kind="compatibility-projected-disposable-container-cgroup-subtree",
        identity=identity,
        includes_descendants=True,
        description=(
            "Limited scope reconstructed from a retained disposable-container report; "
            "the legacy report did not carry the coordinator's measured-scope object."
        ),
        extensions={"scope_identity_limited": True},
    )


class _ClockRegistry:
    def __init__(
        self,
        *,
        timeline: CanonicalTimeline,
        canonical_reference: Mapping[str, Any] | None,
    ) -> None:
        self.timeline = timeline
        self.canonical_reference = canonical_reference
        self.clocks: dict[str, ClockReference] = {
            timeline.clock_id: _clock_reference(
                clock_id=timeline.clock_id,
                clock_name="CLOCK_MONOTONIC",
                reference=canonical_reference,
            )
        }
        self.mappings: dict[str, ClockMapping] = {}

    def source_clock(
        self,
        *,
        clock_id: str,
        clock_name: str,
        reference: Mapping[str, Any] | None,
        explicit_mapping: Mapping[str, Any] | None = None,
        default_reason: str,
    ) -> tuple[str, ClockMapping | None]:
        if explicit_mapping is not None:
            mapping_payload = dict(explicit_mapping)
            mapping_payload.setdefault("source_clock_id", clock_id)
            mapping_payload.setdefault("canonical_clock_id", self.timeline.clock_id)
            mapping = ClockMapping.from_dict(mapping_payload)
            if mapping.source_clock_id != clock_id:
                raise EvidenceProjectionError(
                    f"explicit mapping source {mapping.source_clock_id!r} does not match {clock_id!r}"
                )
            if mapping.canonical_clock_id != self.timeline.clock_id:
                raise EvidenceProjectionError("explicit mapping does not target the bundle canonical clock")
        elif clock_name == "CLOCK_MONOTONIC":
            mapping_payload = monotonic_mapping_from_references(
                dict(reference or {}), dict(self.canonical_reference or {})
            )
            # The timebase helper deliberately emits generic IDs. The projector
            # binds that exact transform and provenance to this retained stream.
            mapping_payload["source_clock_id"] = clock_id
            mapping_payload["canonical_clock_id"] = self.timeline.clock_id
            mapping = ClockMapping.from_dict(mapping_payload)
        else:
            mapping = ClockMapping(
                source_clock_id=clock_id,
                canonical_clock_id=self.timeline.clock_id,
                status="unalignable",
                method="legacy-report-no-validated-clock-correlation",
                reasons=(default_reason,),
            )
        self.clocks[clock_id] = _clock_reference(clock_id=clock_id, clock_name=clock_name, reference=reference)
        self.mappings[clock_id] = mapping
        return clock_id, mapping

    def project(self, native_ns: int, mapping: ClockMapping | None) -> int | None:
        if mapping is None:
            return native_ns
        if mapping.status != "valid":
            return None
        return mapping.to_canonical_ns(native_ns)


def _scalar_stream(
    *,
    run_id: str,
    scope_id: str,
    profiler_id: str,
    stream_id: str,
    metric: str,
    unit: str,
    value: Any,
    artifact_ids: tuple[str, ...],
    provenance: Mapping[str, Any],
    attributes: Mapping[str, Any] | None = None,
    invalid_reason: str | None = None,
    missing_reason: str | None = None,
) -> EvidenceStream:
    if missing_reason is not None or value is None:
        return EvidenceStream(
            stream_id=stream_id,
            profiler_id=profiler_id,
            metric=metric,
            shape="scalar",
            unit=unit,
            status="missing",
            reasons=(missing_reason or f"{metric} is absent from the retained report",),
            raw_artifact_ids=artifact_ids,
            provenance=dict(provenance),
        )
    status = "invalid" if invalid_reason else "valid"
    reasons = (invalid_reason,) if invalid_reason else ()
    record = EvidenceRecord(
        record_id=f"{stream_id}:value",
        run_id=run_id,
        scope_id=scope_id,
        profiler_id=profiler_id,
        stream_id=stream_id,
        shape="scalar",
        status=status,
        reasons=reasons,
        value=value,
        raw_artifact_ids=artifact_ids,
        provenance=dict(provenance),
        attributes=dict(attributes or {}),
    )
    return EvidenceStream(
        stream_id=stream_id,
        profiler_id=profiler_id,
        metric=metric,
        shape="scalar",
        unit=unit,
        status=status,
        reasons=reasons,
        records=(record,),
        raw_artifact_ids=artifact_ids,
        provenance=dict(provenance),
    )


def _missing_stream(
    *,
    profiler_id: str,
    stream_id: str,
    metric: str,
    shape: str,
    unit: str,
    reason: str,
    artifact_ids: tuple[str, ...] = (),
) -> EvidenceStream:
    return EvidenceStream(
        stream_id=stream_id,
        profiler_id=profiler_id,
        metric=metric,
        shape=shape,
        unit=unit,
        status="missing",
        reasons=(reason,),
        raw_artifact_ids=artifact_ids,
        provenance={"projection": PROJECTION_SCHEMA_VERSION},
    )


def _manifest(
    profiler_id: str,
    streams: Sequence[EvidenceStream],
    *,
    source_provenance: Mapping[str, Any] | None = None,
) -> ProfilerManifest:
    statuses = {stream.status for stream in streams}
    if statuses == {"missing"}:
        status = "missing"
        reasons = _unique_reasons(reason for stream in streams for reason in stream.reasons)
    elif "invalid" in statuses or "dropped" in statuses:
        status = "invalid"
        reasons = _unique_reasons(
            reason for stream in streams if stream.status in {"invalid", "dropped"} for reason in stream.reasons
        )
    elif statuses == {"unalignable"} or statuses <= {"missing", "unalignable"}:
        status = "unalignable"
        reasons = _unique_reasons(reason for stream in streams for reason in stream.reasons)
    else:
        status = "valid"
        reasons = ()
    return ProfilerManifest(
        profiler_id=profiler_id,
        profiler_version=PROJECTION_SCHEMA_VERSION,
        status=status,
        reasons=reasons,
        stream_ids=tuple(stream.stream_id for stream in streams),
        provenance={
            **dict(source_provenance or {}),
            "method": "read-only-legacy-report-projection",
            "does_not_execute_workload": True,
        },
    )


def _cpu_streams(
    *,
    cpu: Mapping[str, Any] | None,
    timeseries: Mapping[str, Any] | None,
    timeline: CanonicalTimeline,
    run_id: str,
    scope_id: str,
    report_artifact_id: str,
    timeseries_artifact_id: str | None,
    timeline_valid: bool,
    timeline_invalid_reasons: tuple[str, ...],
) -> list[EvidenceStream]:
    profiler_id = "compat-cgroup-cpu"
    report_refs = (report_artifact_id,)
    provenance = {
        "projection": PROJECTION_SCHEMA_VERSION,
        "method": cpu.get("method") if cpu else None,
    }
    fields = (
        ("wall", "cpu.wall_seconds", "second", "wall_sec"),
        ("usage", "cpu.usage_seconds", "second", "usage_sec"),
        ("user", "cpu.user_seconds", "second", "user_sec"),
        ("system", "cpu.system_seconds", "second", "system_sec"),
        (
            "average-logical-cores",
            "cpu.average_logical_cores",
            "logical_cpu",
            "average_logical_cores_used",
        ),
        (
            "utilization-fraction",
            "cpu.utilization_fraction_of_allocation",
            "ratio",
            "utilization_fraction_of_allocation",
        ),
    )
    streams: list[EvidenceStream] = []
    for suffix, metric, unit, field_name in fields:
        value = _number(cpu.get(field_name)) if cpu else None
        streams.append(
            _scalar_stream(
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id=f"cpu-{suffix}",
                metric=metric,
                unit=unit,
                value=value,
                artifact_ids=report_refs,
                provenance=provenance,
                missing_reason=(None if value is not None else f"retained report has no numeric {field_name}"),
            )
        )

    records: list[EvidenceRecord] = []
    raw_samples = [row for row in _list(timeseries.get("samples") if timeseries else None) if isinstance(row, dict)]
    previous: Mapping[str, Any] | None = None
    for row in raw_samples:
        timestamp = _integer(row.get("timestamp_ns"))
        value = _number(row.get("cpu_average_logical_cores"))
        if previous is not None and timestamp is not None and value is not None:
            start = _integer(previous.get("timestamp_ns"))
            if start is not None and timestamp > start:
                error = (
                    max(
                        _integer(previous.get("acquisition_end_ns")) or start,
                        start,
                    )
                    - start
                )
                error = max(
                    error,
                    (_integer(row.get("acquisition_end_ns")) or timestamp) - timestamp,
                )
                records.append(
                    EvidenceRecord(
                        record_id=f"cpu-interval-average:{len(records):06d}",
                        run_id=run_id,
                        scope_id=scope_id,
                        profiler_id=profiler_id,
                        stream_id="cpu-interval-average",
                        shape="window_aggregate",
                        status="valid" if timeline_valid else "invalid",
                        reasons=(timeline_invalid_reasons if not timeline_valid else ()),
                        value=value,
                        source_clock_id=timeline.clock_id,
                        native_start_ns=start,
                        native_end_ns=timestamp,
                        canonical_start_ns=start,
                        canonical_end_ns=timestamp,
                        alignment_error_ns=max(error, 0),
                        raw_artifact_ids=((timeseries_artifact_id,) if timeseries_artifact_id else ()),
                        provenance={
                            "method": "delta-cgroup-cpu.stat-over-sample-window",
                            "source_field": "cpu_average_logical_cores",
                        },
                    )
                )
        previous = row
    if records:
        streams.append(
            EvidenceStream(
                stream_id="cpu-interval-average",
                profiler_id=profiler_id,
                metric="cpu.interval_average_logical_cores",
                shape="window_aggregate",
                unit="logical_cpu",
                status="valid" if timeline_valid else "invalid",
                reasons=(timeline_invalid_reasons if not timeline_valid else ()),
                records=tuple(records),
                raw_artifact_ids=((timeseries_artifact_id,) if timeseries_artifact_id else ()),
                provenance={"projection": PROJECTION_SCHEMA_VERSION},
            )
        )
    else:
        streams.append(
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="cpu-interval-average",
                metric="cpu.interval_average_logical_cores",
                shape="window_aggregate",
                unit="logical_cpu",
                reason="resource time series has no interval-average CPU samples",
                artifact_ids=((timeseries_artifact_id,) if timeseries_artifact_id else report_refs),
            )
        )
    return streams


def _memory_streams(
    *,
    memory: Mapping[str, Any] | None,
    timeseries: Mapping[str, Any] | None,
    timeline: CanonicalTimeline,
    run_id: str,
    scope_id: str,
    report_artifact_id: str,
    timeseries_artifact_id: str | None,
) -> list[EvidenceStream]:
    profiler_id = "compat-cgroup-memory"
    refs = tuple(value for value in (report_artifact_id, timeseries_artifact_id) if value is not None)
    samples = [row for row in _list(timeseries.get("samples") if timeseries else None) if isinstance(row, dict)]
    timeline_valid = bool(memory and memory.get("timeline_valid_for_shape", True))
    invalid_reasons = _unique_reasons(
        _list(memory.get("timeline_invalid_reasons") if memory else None)
        or (["memory timeline was marked invalid"] if not timeline_valid else [])
    )
    cadence_ns = _ns_from_seconds(timeseries.get("sample_interval_sec") if timeseries else None)
    cadence = (
        Cadence(
            requested_interval_ns=cadence_ns,
            maximum_acceptable_gap_ns=2 * cadence_ns,
            method="legacy-cgroup-resource-sampler",
        )
        if cadence_ns is not None and cadence_ns > 0
        else None
    )
    gaps: list[Gap] = []
    if cadence is not None:
        for previous, current in zip(samples, samples[1:], strict=False):
            start = _integer(previous.get("timestamp_ns"))
            end = _integer(current.get("timestamp_ns"))
            if (
                start is not None
                and end is not None
                and end - start > cadence.maximum_acceptable_gap_ns
                and timeline.start_ns <= start < end <= timeline.end_ns
            ):
                gaps.append(
                    Gap(
                        clock_id=timeline.clock_id,
                        start_ns=start,
                        end_ns=end,
                        reason="sample gap exceeded twice the requested cadence",
                    )
                )

    field_specs = (
        ("memory-current", "memory.current", "memory_current_bytes", None),
        ("memory-anon", "memory.anon", "anon", "memory_stat"),
        ("memory-file", "memory.file", "file", "memory_stat"),
        ("memory-kernel", "memory.kernel", "kernel", "memory_stat"),
    )
    streams: list[EvidenceStream] = []
    for stream_id, metric, field_name, parent in field_specs:
        records: list[EvidenceRecord] = []
        for row in samples:
            source = _object(row.get(parent)) if parent else row
            value = _number(source.get(field_name)) if source else None
            timestamp = _integer(row.get("timestamp_ns"))
            acquisition_start = _integer(row.get("acquisition_start_ns", row.get("timestamp_ns")))
            acquisition_end = _integer(row.get("acquisition_end_ns", row.get("timestamp_ns")))
            if (
                value is None
                or timestamp is None
                or acquisition_start is None
                or acquisition_end is None
                or acquisition_end < acquisition_start
            ):
                continue
            status = "valid" if timeline_valid else "invalid"
            records.append(
                EvidenceRecord(
                    record_id=f"{stream_id}:{len(records):06d}",
                    run_id=run_id,
                    scope_id=scope_id,
                    profiler_id=profiler_id,
                    stream_id=stream_id,
                    shape="sampled_gauge",
                    status=status,
                    reasons=invalid_reasons if status == "invalid" else (),
                    value=value,
                    source_clock_id=timeline.clock_id,
                    native_timestamp_ns=timestamp,
                    acquisition_bounds=AcquisitionBounds(acquisition_start, acquisition_end),
                    alignment_error_ns=acquisition_end - acquisition_start,
                    raw_artifact_ids=((timeseries_artifact_id,) if timeseries_artifact_id else ()),
                    provenance={
                        "method": "cgroup-v2-file-read",
                        "source_field": (f"{parent}.{field_name}" if parent else field_name),
                    },
                )
            )
        if records:
            status = "valid" if timeline_valid else "invalid"
            streams.append(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id=profiler_id,
                    metric=metric,
                    shape="sampled_gauge",
                    unit="byte",
                    status=status,
                    reasons=invalid_reasons if status == "invalid" else (),
                    records=tuple(records),
                    cadence=cadence,
                    gaps=tuple(gaps),
                    raw_artifact_ids=((timeseries_artifact_id,) if timeseries_artifact_id else ()),
                    provenance={
                        "projection": PROJECTION_SCHEMA_VERSION,
                        "scope": "cgroup-v2-subtree-charges-not-process-rss",
                    },
                )
            )
        else:
            streams.append(
                _missing_stream(
                    profiler_id=profiler_id,
                    stream_id=stream_id,
                    metric=metric,
                    shape="sampled_gauge",
                    unit="byte",
                    reason=f"resource time series has no {metric} samples",
                    artifact_ids=refs,
                )
            )

    lifetime = _object(memory.get("lifetime")) if memory else None
    peak = _number(lifetime.get("peak_bytes")) if lifetime else None
    exact = bool(
        lifetime
        and (lifetime.get("exact_through_controlled_finish_gate") or lifetime.get("exact_for_fresh_cgroup_lifetime"))
    )
    peak_invalid = None
    if peak is not None and not exact:
        peak_invalid = "retained peak is not declared exact through the controlled finish gate"
    streams.append(
        _scalar_stream(
            run_id=run_id,
            scope_id=scope_id,
            profiler_id=profiler_id,
            stream_id="memory-peak",
            metric="memory.peak",
            unit="byte",
            value=peak,
            artifact_ids=(report_artifact_id,),
            provenance={
                "projection": PROJECTION_SCHEMA_VERSION,
                "peak_source": lifetime.get("peak_source") if lifetime else None,
            },
            attributes={
                "exact": exact,
                "timing": "untimed",
                "natural_peak_trustworthy": bool(memory and memory.get("natural_peak_trustworthy")),
                "censoring": memory.get("censoring") if memory else None,
            },
            invalid_reason=peak_invalid,
            missing_reason=("retained report has no memory.peak value" if peak is None else None),
        )
    )
    limits = _object(memory.get("limits")) if memory else None
    memory_max = _number(limits.get("memory_max_bytes")) if limits else None
    if limits and limits.get("memory_max_state") == "finite" and memory_max is not None:
        streams.append(
            _scalar_stream(
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id="memory-max",
                metric="memory.max",
                unit="byte",
                value=memory_max,
                artifact_ids=(report_artifact_id,),
                provenance={
                    "projection": PROJECTION_SCHEMA_VERSION,
                    "timing": "observer-start",
                },
                attributes={"exact": True},
            )
        )
    margin = _object(memory.get("observed_limit_margin")) if memory else None
    remaining = _number(margin.get("remaining_to_limit_bytes")) if margin else None
    if remaining is not None:
        streams.append(
            _scalar_stream(
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id="memory-observed-limit-margin",
                metric="memory.observed_limit_margin",
                unit="byte",
                value=remaining,
                artifact_ids=(report_artifact_id,),
                provenance={
                    "projection": PROJECTION_SCHEMA_VERSION,
                    "meaning": "observed leaf memory.max minus exact observed peak",
                },
                attributes={
                    "exact": peak_invalid is None,
                    "not_a_recommendation": True,
                    "peak_fraction_of_limit": margin.get("peak_fraction_of_limit"),
                },
                invalid_reason=peak_invalid,
            )
        )

    window_start = _find_marker(timeline, "sandbox_observation_start") or timeline.start_ns
    window_end = _find_marker(timeline, "sandbox_observation_end") or timeline.end_ns
    if window_end <= window_start:
        window_start, window_end = timeline.start_ns, timeline.end_ns
    events = _object(memory.get("events_delta")) if memory else None
    event_streams = 0
    if events:
        for key in sorted(events):
            value = _number(events[key])
            if value is None:
                continue
            stream_id = f"memory-event-{_safe_token(key)}"
            record = EvidenceRecord(
                record_id=f"{stream_id}:window",
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id=stream_id,
                shape="window_aggregate",
                status="valid",
                value=value,
                source_clock_id=timeline.clock_id,
                native_start_ns=window_start,
                native_end_ns=window_end,
                canonical_start_ns=window_start,
                canonical_end_ns=window_end,
                alignment_error_ns=0,
                raw_artifact_ids=(report_artifact_id,),
                provenance={
                    "source": "memory.events delta",
                    "timing": "aggregate-only-event-occurrence-time-unknown",
                },
            )
            streams.append(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id=profiler_id,
                    metric=f"memory.events.{key}",
                    shape="window_aggregate",
                    unit="event",
                    status="valid",
                    records=(record,),
                    raw_artifact_ids=(report_artifact_id,),
                    provenance={"projection": PROJECTION_SCHEMA_VERSION},
                )
            )
            event_streams += 1
    if not event_streams:
        streams.append(
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="memory-events",
                metric="memory.events",
                shape="window_aggregate",
                unit="event",
                reason="retained report has no memory.events deltas",
                artifact_ids=(report_artifact_id,),
            )
        )

    pressure = _object(memory.get("pressure_delta")) if memory else None
    pressure_streams = 0
    if pressure:
        for kind in sorted(pressure):
            row = _object(pressure[kind])
            value = _number(row.get("total_stall_usec")) if row else None
            if value is None:
                continue
            stream_id = f"memory-pressure-{_safe_token(kind)}"
            record = EvidenceRecord(
                record_id=f"{stream_id}:window",
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id=stream_id,
                shape="window_aggregate",
                status="valid",
                value=value,
                source_clock_id=timeline.clock_id,
                native_start_ns=window_start,
                native_end_ns=window_end,
                canonical_start_ns=window_start,
                canonical_end_ns=window_end,
                alignment_error_ns=0,
                raw_artifact_ids=(report_artifact_id,),
                provenance={
                    "source": f"memory.pressure {kind} total delta",
                    "timing": "aggregate-only-stall-occurrence-time-unknown",
                },
            )
            streams.append(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id=profiler_id,
                    metric=f"memory.pressure.{kind}.total_stall",
                    shape="window_aggregate",
                    unit="microsecond",
                    status="valid",
                    records=(record,),
                    raw_artifact_ids=(report_artifact_id,),
                    provenance={"projection": PROJECTION_SCHEMA_VERSION},
                )
            )
            pressure_streams += 1
    if not pressure_streams:
        streams.append(
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="memory-pressure",
                metric="memory.pressure",
                shape="window_aggregate",
                unit="microsecond",
                reason="retained report has no PSI memory pressure delta",
                artifact_ids=(report_artifact_id,),
            )
        )
    return streams


def _scheduler_window(scheduler: Mapping[str, Any] | None, timeline: CanonicalTimeline) -> tuple[int, int]:
    start = _integer(scheduler.get("window_start_ns")) if scheduler else None
    end = _integer(scheduler.get("window_end_ns")) if scheduler else None
    start = start or _find_marker(timeline, "command_start") or timeline.start_ns
    end = end or _find_marker(timeline, "command_end") or timeline.end_ns
    if end <= start:
        return timeline.start_ns, timeline.end_ns
    return start, end


def _scheduler_streams(
    *,
    scheduler: Mapping[str, Any] | None,
    events: Sequence[Mapping[str, Any]] | None,
    timeline: CanonicalTimeline,
    run_id: str,
    scope_id: str,
    report_artifact_id: str,
    events_artifact_id: str | None,
) -> list[EvidenceStream]:
    profiler_id = "compat-scheduler"
    refs = tuple(value for value in (report_artifact_id, events_artifact_id) if value is not None)
    if events is None:
        reason = "retained report has no referenced raw scheduler event stream"
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="scheduler-transitions",
                metric="scheduler.transition",
                shape="point_event",
                unit="event",
                reason=reason,
                artifact_ids=refs,
            ),
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="scheduler-concurrency",
                metric="scheduler.concurrent_logical_cpus",
                shape="interval",
                unit="logical_cpu",
                reason=reason,
                artifact_ids=refs,
            ),
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="scheduler-per-core",
                metric="scheduler.logical_cpu_activity",
                shape="interval",
                unit="scheduled_interval",
                reason=reason,
                artifact_ids=refs,
            ),
        ]
    parsed: list[dict[str, Any]] = []
    for row in events:
        timestamp = _integer(row.get("timestamp_ns"))
        cpu = _integer(row.get("cpu"))
        tid = _integer(row.get("tid"))
        entering = row.get("entering")
        if timestamp is not None and cpu is not None and tid is not None and isinstance(entering, bool):
            parsed.append({"timestamp_ns": timestamp, "cpu": cpu, "tid": tid, "entering": entering})
    parsed.sort(key=lambda row: (row["timestamp_ns"], not row["entering"], row["cpu"], row["tid"]))
    if not parsed:
        reason = "raw scheduler artifact has no parseable transition events"
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id=stream_id,
                metric=metric,
                shape=shape,
                unit=unit,
                reason=reason,
                artifact_ids=refs,
            )
            for stream_id, metric, shape, unit in (
                ("scheduler-transitions", "scheduler.transition", "point_event", "event"),
                ("scheduler-concurrency", "scheduler.concurrent_logical_cpus", "interval", "logical_cpu"),
                ("scheduler-per-core", "scheduler.logical_cpu_activity", "interval", "scheduled_interval"),
            )
        ]
    validation = _object(scheduler.get("cgroup_validation")) if scheduler else None
    trace_valid = bool(scheduler and scheduler.get("trace_complete"))
    if validation is not None:
        trace_valid = trace_valid and bool(validation.get("valid_for_concurrency_reporting"))
    reasons: list[str] = []
    if not trace_valid:
        reasons.append("scheduler trace or cgroup cross-check is invalid")
    lost = _integer(scheduler.get("lost_perf_events")) if scheduler else None
    if lost:
        reasons.append(f"scheduler trace reports {lost} lost events")
    reasons.extend(str(reason) for reason in _list(scheduler.get("anomalies") if scheduler else None))
    invalid_reasons = _unique_reasons(reasons)
    record_status = "valid" if trace_valid and not invalid_reasons else "invalid"
    if record_status == "invalid" and not invalid_reasons:
        invalid_reasons = ("scheduler trace validity was not established",)
    start_ns, end_ns = _scheduler_window(scheduler, timeline)

    point_records: list[EvidenceRecord] = []
    for row in parsed:
        timestamp = row["timestamp_ns"]
        if not start_ns <= timestamp <= end_ns:
            continue
        point_records.append(
            EvidenceRecord(
                record_id=f"scheduler-transition:{len(point_records):08d}",
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id="scheduler-transitions",
                shape="point_event",
                status=record_status,
                reasons=invalid_reasons if record_status == "invalid" else (),
                value={
                    "cpu": row["cpu"],
                    "tid": row["tid"],
                    "direction": "enter" if row["entering"] else "leave",
                },
                source_clock_id=timeline.clock_id,
                native_timestamp_ns=timestamp,
                canonical_timestamp_ns=timestamp,
                alignment_error_ns=0,
                raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                provenance={"method": scheduler.get("method") if scheduler else None},
            )
        )
    streams: list[EvidenceStream] = []
    if point_records:
        streams.append(
            EvidenceStream(
                stream_id="scheduler-transitions",
                profiler_id=profiler_id,
                metric="scheduler.transition",
                shape="point_event",
                unit="event",
                status=record_status,
                reasons=invalid_reasons if record_status == "invalid" else (),
                records=tuple(point_records),
                raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                provenance={"projection": PROJECTION_SCHEMA_VERSION},
            )
        )
    else:
        streams.append(
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="scheduler-transitions",
                metric="scheduler.transition",
                shape="point_event",
                unit="event",
                reason="no scheduler transitions fall inside the canonical command window",
                artifact_ids=refs,
            )
        )

    active: dict[int, tuple[int, int]] = {}
    for row in parsed:
        if row["timestamp_ns"] >= start_ns:
            break
        if row["entering"]:
            active[row["cpu"]] = (start_ns, row["tid"])
        else:
            active.pop(row["cpu"], None)
    concurrency_records: list[EvidenceRecord] = []
    per_core_records: list[EvidenceRecord] = []
    cursor = start_ns
    within = [row for row in parsed if start_ns <= row["timestamp_ns"] <= end_ns]
    index = 0
    while index < len(within):
        timestamp = within[index]["timestamp_ns"]
        if timestamp > cursor:
            concurrency_records.append(
                EvidenceRecord(
                    record_id=f"scheduler-concurrency:{len(concurrency_records):08d}",
                    run_id=run_id,
                    scope_id=scope_id,
                    profiler_id=profiler_id,
                    stream_id="scheduler-concurrency",
                    shape="interval",
                    status=record_status,
                    reasons=invalid_reasons if record_status == "invalid" else (),
                    value=len(active),
                    source_clock_id=timeline.clock_id,
                    native_start_ns=cursor,
                    native_end_ns=timestamp,
                    canonical_start_ns=cursor,
                    canonical_end_ns=timestamp,
                    alignment_error_ns=0,
                    raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                    provenance={"method": "integrated-scheduler-transitions"},
                )
            )
        group: list[dict[str, Any]] = []
        while index < len(within) and within[index]["timestamp_ns"] == timestamp:
            group.append(within[index])
            index += 1
        for row in (row for row in group if not row["entering"]):
            started = active.pop(row["cpu"], None)
            if started is not None and timestamp > started[0]:
                per_core_records.append(
                    EvidenceRecord(
                        record_id=f"scheduler-per-core:{len(per_core_records):08d}",
                        run_id=run_id,
                        scope_id=scope_id,
                        profiler_id=profiler_id,
                        stream_id="scheduler-per-core",
                        shape="interval",
                        status=record_status,
                        reasons=invalid_reasons if record_status == "invalid" else (),
                        value={"cpu": row["cpu"], "tid": started[1]},
                        source_clock_id=timeline.clock_id,
                        native_start_ns=started[0],
                        native_end_ns=timestamp,
                        canonical_start_ns=started[0],
                        canonical_end_ns=timestamp,
                        alignment_error_ns=0,
                        raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                        provenance={"method": "paired-scheduler-enter-leave"},
                        attributes={"lane": f"logical-cpu-{row['cpu']}"},
                    )
                )
        for row in (row for row in group if row["entering"]):
            active[row["cpu"]] = (timestamp, row["tid"])
        cursor = timestamp
    if cursor < end_ns:
        concurrency_records.append(
            EvidenceRecord(
                record_id=f"scheduler-concurrency:{len(concurrency_records):08d}",
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id="scheduler-concurrency",
                shape="interval",
                status=record_status,
                reasons=invalid_reasons if record_status == "invalid" else (),
                value=len(active),
                source_clock_id=timeline.clock_id,
                native_start_ns=cursor,
                native_end_ns=end_ns,
                canonical_start_ns=cursor,
                canonical_end_ns=end_ns,
                alignment_error_ns=0,
                raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                provenance={"method": "integrated-scheduler-transitions"},
            )
        )
    for cpu, (started, tid) in sorted(active.items()):
        if end_ns > started:
            per_core_records.append(
                EvidenceRecord(
                    record_id=f"scheduler-per-core:{len(per_core_records):08d}",
                    run_id=run_id,
                    scope_id=scope_id,
                    profiler_id=profiler_id,
                    stream_id="scheduler-per-core",
                    shape="interval",
                    status=record_status,
                    reasons=invalid_reasons if record_status == "invalid" else (),
                    value={"cpu": cpu, "tid": tid},
                    source_clock_id=timeline.clock_id,
                    native_start_ns=started,
                    native_end_ns=end_ns,
                    canonical_start_ns=started,
                    canonical_end_ns=end_ns,
                    alignment_error_ns=0,
                    raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                    provenance={"method": "scheduler-active-at-window-end"},
                    attributes={"lane": f"logical-cpu-{cpu}"},
                )
            )
    for stream_id, metric, unit, records in (
        ("scheduler-concurrency", "scheduler.concurrent_logical_cpus", "logical_cpu", concurrency_records),
        ("scheduler-per-core", "scheduler.logical_cpu_activity", "scheduled_interval", per_core_records),
    ):
        if records:
            streams.append(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id=profiler_id,
                    metric=metric,
                    shape="interval",
                    unit=unit,
                    status=record_status,
                    reasons=invalid_reasons if record_status == "invalid" else (),
                    records=tuple(records),
                    raw_artifact_ids=((events_artifact_id,) if events_artifact_id else ()),
                    provenance={"projection": PROJECTION_SCHEMA_VERSION},
                )
            )
        else:
            streams.append(
                _missing_stream(
                    profiler_id=profiler_id,
                    stream_id=stream_id,
                    metric=metric,
                    shape="interval",
                    unit=unit,
                    reason="raw scheduler transitions contain no positive-duration intervals",
                    artifact_ids=refs,
                )
            )
    return streams


def _timed_record(
    *,
    record_id: str,
    run_id: str,
    scope_id: str,
    profiler_id: str,
    stream_id: str,
    shape: str,
    value: Any,
    source_clock_id: str,
    mapping: ClockMapping | None,
    clocks: _ClockRegistry,
    native_timestamp_ns: int | None = None,
    native_start_ns: int | None = None,
    native_end_ns: int | None = None,
    artifact_ids: tuple[str, ...] = (),
    provenance: Mapping[str, Any] | None = None,
    attributes: Mapping[str, Any] | None = None,
    source_invalid_reasons: tuple[str, ...] = (),
) -> EvidenceRecord:
    aligned = mapping is None or mapping.status == "valid"
    status = "invalid" if aligned and source_invalid_reasons else "valid" if aligned else "unalignable"
    reasons = source_invalid_reasons if status == "invalid" else mapping.reasons if mapping else ()
    alignment_error = (mapping.max_error_ns if mapping is not None else 0) if aligned else None
    fields: dict[str, Any] = {}
    if native_timestamp_ns is not None:
        fields["native_timestamp_ns"] = native_timestamp_ns
        canonical = clocks.project(native_timestamp_ns, mapping)
        if canonical is not None:
            fields["canonical_timestamp_ns"] = canonical
    else:
        assert native_start_ns is not None and native_end_ns is not None
        fields["native_start_ns"] = native_start_ns
        fields["native_end_ns"] = native_end_ns
        canonical_start = clocks.project(native_start_ns, mapping)
        canonical_end = clocks.project(native_end_ns, mapping)
        if canonical_start is not None and canonical_end is not None:
            fields["canonical_start_ns"] = canonical_start
            fields["canonical_end_ns"] = canonical_end
    return EvidenceRecord(
        record_id=record_id,
        run_id=run_id,
        scope_id=scope_id,
        profiler_id=profiler_id,
        stream_id=stream_id,
        shape=shape,
        status=status,
        reasons=reasons,
        value=value,
        source_clock_id=source_clock_id,
        alignment_error_ns=alignment_error,
        raw_artifact_ids=artifact_ids,
        provenance=dict(provenance or {}),
        attributes=dict(attributes or {}),
        **fields,
    )


def _action_streams(
    *,
    replay: Mapping[str, Any] | None,
    clocks: _ClockRegistry,
    explicit_mapping: Mapping[str, Any] | None,
    timeline: CanonicalTimeline,
    run_id: str,
    scope_id: str,
    artifact_id: str | None,
) -> list[EvidenceStream]:
    profiler_id = "compat-replay-actions"
    refs = (artifact_id,) if artifact_id else ()
    if replay is None:
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="replay-actions",
                metric="replay.action",
                shape="interval",
                unit="action",
                reason="retained report has no referenced replay action report",
                artifact_ids=refs,
            )
        ]
    reference = _object(replay.get("clock_reference"))
    source_clock_id, mapping = clocks.source_clock(
        clock_id="replay-monotonic",
        clock_name="CLOCK_MONOTONIC",
        reference=reference,
        explicit_mapping=explicit_mapping,
        default_reason=(
            "replay CLOCK_MONOTONIC namespace/boot identity was not proven equal " "to the canonical host clock"
        ),
    )
    records: list[EvidenceRecord] = []
    for action in _list(replay.get("actions")):
        if not isinstance(action, dict):
            continue
        start = _integer(action.get("replay_start_monotonic_ns"))
        end = _integer(action.get("replay_end_monotonic_ns"))
        if start is None or end is None or end <= start:
            continue
        value = {
            key: action[key]
            for key in (
                "action_index",
                "step_id",
                "tool",
                "source_tool",
                "command",
                "status",
                "returncode",
                "outcome_matches_source",
            )
            if key in action
        }
        records.append(
            _timed_record(
                record_id=f"replay-action:{len(records):06d}",
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id="replay-actions",
                shape="interval",
                value=value,
                source_clock_id=source_clock_id,
                mapping=mapping,
                clocks=clocks,
                native_start_ns=start,
                native_end_ns=end,
                artifact_ids=refs,
                provenance={"source_schema_version": replay.get("schema_version")},
                attributes={"lane": "replay-actions"},
            )
        )
    if not records:
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="replay-actions",
                metric="replay.action",
                shape="interval",
                unit="action",
                reason="replay report has no actions with monotonic start/end timestamps",
                artifact_ids=refs,
            )
        ]
    status = records[0].status
    reasons = records[0].reasons
    return [
        EvidenceStream(
            stream_id="replay-actions",
            profiler_id=profiler_id,
            metric="replay.action",
            shape="interval",
            unit="action",
            status=status,
            reasons=reasons,
            records=tuple(records),
            raw_artifact_ids=refs,
            provenance={
                "projection": PROJECTION_SCHEMA_VERSION,
                "clock_identity": dict(reference or {}),
            },
        )
    ]


def _process_streams(
    *,
    process: Mapping[str, Any] | None,
    clocks: _ClockRegistry,
    explicit_mapping: Mapping[str, Any] | None,
    run_id: str,
    scope_id: str,
    artifact_ids: tuple[str, ...],
) -> list[EvidenceStream]:
    profiler_id = "compat-process-exec"
    if process is None:
        reason = "retained report has no structured process-exec evidence"
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="process-lifecycles",
                metric="process.lifecycle",
                shape="interval",
                unit="process",
                reason=reason,
                artifact_ids=artifact_ids,
            ),
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="process-exec-events",
                metric="process.exec",
                shape="point_event",
                unit="event",
                reason=reason,
                artifact_ids=artifact_ids,
            ),
        ]
    source_clock = str(process.get("source_clock") or "realtime").lower()
    clock_name = "CLOCK_REALTIME" if source_clock == "realtime" else source_clock.upper()
    source_clock_id, mapping = clocks.source_clock(
        clock_id="process-realtime",
        clock_name=clock_name,
        reference=_object(process.get("clock_reference")),
        explicit_mapping=explicit_mapping,
        default_reason=("strace CLOCK_REALTIME timestamps have no validated mapping to canonical " "CLOCK_MONOTONIC"),
    )
    process_invalid = not bool(process.get("valid", False))
    source_reasons = _unique_reasons(
        [
            str(row.get("message"))
            for row in _list(process.get("anomalies"))
            if isinstance(row, dict) and row.get("message")
        ]
        or (["structured process trace is marked invalid"] if process_invalid else [])
    )
    lifecycle_records: list[EvidenceRecord] = []
    exec_records: list[EvidenceRecord] = []
    for row in _list(process.get("processes")):
        if not isinstance(row, dict):
            continue
        pid = _integer(row.get("pid"))
        start = _integer(row.get("start_ns"))
        end = _integer(row.get("end_ns"))
        if pid is not None and start is not None and end is not None and end > start:
            lifecycle_records.append(
                _timed_record(
                    record_id=f"process-lifecycle:{pid}",
                    run_id=run_id,
                    scope_id=scope_id,
                    profiler_id=profiler_id,
                    stream_id="process-lifecycles",
                    shape="interval",
                    value={
                        key: row.get(key)
                        for key in (
                            "pid",
                            "parent_pid",
                            "binary",
                            "argv",
                            "cwd",
                            "exit_code",
                            "exit_signal",
                            "in_scope",
                            "process_escape",
                        )
                    },
                    source_clock_id=source_clock_id,
                    mapping=mapping,
                    clocks=clocks,
                    native_start_ns=start,
                    native_end_ns=end,
                    artifact_ids=artifact_ids,
                    provenance={"method": process.get("method")},
                    attributes={"lane": f"pid-{pid}", "parent_pid": row.get("parent_pid")},
                    source_invalid_reasons=(
                        source_reasons if process_invalid and mapping and mapping.status == "valid" else ()
                    ),
                )
            )
        for event in _list(row.get("execs")):
            if not isinstance(event, dict):
                continue
            timestamp = _integer(event.get("started_ns"))
            if timestamp is None:
                continue
            exec_records.append(
                _timed_record(
                    record_id=f"process-exec:{len(exec_records):06d}",
                    run_id=run_id,
                    scope_id=scope_id,
                    profiler_id=profiler_id,
                    stream_id="process-exec-events",
                    shape="point_event",
                    value={"pid": pid, **event},
                    source_clock_id=source_clock_id,
                    mapping=mapping,
                    clocks=clocks,
                    native_timestamp_ns=timestamp,
                    artifact_ids=artifact_ids,
                    provenance={"method": process.get("method")},
                    source_invalid_reasons=(
                        source_reasons if process_invalid and mapping and mapping.status == "valid" else ()
                    ),
                )
            )
    streams: list[EvidenceStream] = []
    for stream_id, metric, shape, unit, records in (
        ("process-lifecycles", "process.lifecycle", "interval", "process", lifecycle_records),
        ("process-exec-events", "process.exec", "point_event", "event", exec_records),
    ):
        if records:
            statuses = {record.status for record in records}
            status = "invalid" if "invalid" in statuses else "unalignable" if "unalignable" in statuses else "valid"
            reasons = _unique_reasons(reason for record in records for reason in record.reasons)
            if status == "unalignable" and process_invalid:
                reasons = _unique_reasons((*reasons, *source_reasons))
                records = [
                    EvidenceRecord(
                        **{
                            **record.__dict__,
                            "reasons": reasons,
                        }
                    )
                    for record in records
                ]
            streams.append(
                EvidenceStream(
                    stream_id=stream_id,
                    profiler_id=profiler_id,
                    metric=metric,
                    shape=shape,
                    unit=unit,
                    status=status,
                    reasons=reasons if status != "valid" else (),
                    records=tuple(records),
                    raw_artifact_ids=artifact_ids,
                    provenance={
                        "projection": PROJECTION_SCHEMA_VERSION,
                        "source_valid": not process_invalid,
                    },
                )
            )
        else:
            streams.append(
                _missing_stream(
                    profiler_id=profiler_id,
                    stream_id=stream_id,
                    metric=metric,
                    shape=shape,
                    unit=unit,
                    reason=f"structured process report has no {metric} records",
                    artifact_ids=artifact_ids,
                )
            )
    return streams


def _hardware_streams(
    *,
    hardware: Mapping[str, Any] | None,
    timeline: CanonicalTimeline,
    run_id: str,
    scope_id: str,
    artifact_ids: tuple[str, ...],
) -> list[EvidenceStream]:
    profiler_id = "compat-hardware-perf"
    if hardware is None:
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="hardware-counters",
                metric="hardware.counter",
                shape="window_aggregate",
                unit="event",
                reason="retained report has no hardware perf-counter report",
                artifact_ids=artifact_ids,
            )
        ]
    start = _integer(hardware.get("window_start_ns"))
    end = _integer(hardware.get("window_end_ns"))
    start = start or _find_marker(timeline, "command_start") or timeline.start_ns
    end = end or _find_marker(timeline, "command_end") or timeline.end_ns
    if end <= start:
        raise EvidenceProjectionError("hardware perf command window is not positive")
    raw_rows = hardware.get("counters")
    counter_items = _list(raw_rows)
    rows = [row for row in counter_items if isinstance(row, dict)]
    global_invalid_reasons: list[str] = []
    if raw_rows is not None and not isinstance(raw_rows, list):
        global_invalid_reasons.append("hardware counters field is malformed")
    elif any(not isinstance(row, dict) for row in counter_items):
        global_invalid_reasons.append("hardware counters contain a malformed entry")
    if any(not isinstance(row.get("event"), str) or not row.get("event") for row in rows):
        global_invalid_reasons.append("hardware counters contain an entry without an event name")
    if "instrumentation_valid" in hardware:
        instrumentation_valid = hardware.get("instrumentation_valid")
        if instrumentation_valid is not True:
            global_invalid_reasons.append("hardware perf instrumentation was not marked valid")
    anomalies = hardware.get("anomalies")
    if anomalies is not None:
        if not isinstance(anomalies, list):
            global_invalid_reasons.append("hardware perf anomalies field is malformed")
        else:
            for anomaly in anomalies:
                if isinstance(anomaly, str) and anomaly.strip():
                    global_invalid_reasons.append(f"hardware perf anomaly: {anomaly.strip()}")
                elif anomaly is not None:
                    global_invalid_reasons.append("hardware perf anomalies contain a malformed entry")
    row_has_explicit_invalidity = any(row.get("valid") is not True or _number(row.get("value")) is None for row in rows)
    if "all_requested_metrics_valid" in hardware:
        all_requested_valid = hardware.get("all_requested_metrics_valid")
        if not isinstance(all_requested_valid, bool):
            global_invalid_reasons.append("hardware all_requested_metrics_valid field is malformed")
        elif not all_requested_valid and not row_has_explicit_invalidity:
            global_invalid_reasons.append("hardware report says not all requested metrics are valid")

    start_error_raw = hardware.get("window_start_alignment_error_ns")
    end_error_raw = hardware.get("window_end_alignment_error_ns")
    if start_error_raw is None and end_error_raw is None:
        # Older retained reports predate acknowledged perf control bounds.
        # Keep them readable, but never reinterpret missing uncertainty as
        # exact alignment.
        global_invalid_reasons.append("hardware report lacks counter-window alignment uncertainty")
        alignment_error_ns = end - start
    else:
        start_error = _integer(start_error_raw)
        end_error = _integer(end_error_raw)
        if start_error is None or end_error is None or start_error < 0 or end_error < 0:
            global_invalid_reasons.append("hardware counter-window alignment uncertainty is malformed")
            alignment_error_ns = end - start
        else:
            alignment_error_ns = max(start_error, end_error)
    global_reasons = _unique_reasons(global_invalid_reasons)

    streams: list[EvidenceStream] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        event = row.get("event")
        if not isinstance(event, str) or not event:
            continue
        stream_id = f"hardware-{_safe_token(event)}"
        valid = bool(row.get("valid"))
        value = _number(row.get("value"))
        invalid_reason = str(row.get("invalid_reason") or f"perf counter status is {row.get('status', 'invalid')}")
        reasons = _unique_reasons(
            (
                *global_reasons,
                *(() if valid and value is not None else (invalid_reason,)),
            )
        )
        status = "valid" if not reasons else "invalid"
        record = EvidenceRecord(
            record_id=f"{stream_id}:command-window",
            run_id=run_id,
            scope_id=scope_id,
            profiler_id=profiler_id,
            stream_id=stream_id,
            shape="window_aggregate",
            status=status,
            reasons=reasons,
            value=value,
            source_clock_id=timeline.clock_id,
            native_start_ns=start,
            native_end_ns=end,
            canonical_start_ns=start,
            canonical_end_ns=end,
            alignment_error_ns=alignment_error_ns,
            raw_artifact_ids=artifact_ids,
            provenance={
                "method": hardware.get("method"),
                "namespace": row.get("namespace"),
                "time_enabled_ns": row.get("time_enabled_ns"),
                "time_running_ns": row.get("time_running_ns"),
                "time_running_fraction": row.get("time_running_fraction"),
                "source_status": row.get("status"),
                "window_start_alignment_error_ns": start_error_raw,
                "window_end_alignment_error_ns": end_error_raw,
            },
        )
        streams.append(
            EvidenceStream(
                stream_id=stream_id,
                profiler_id=profiler_id,
                metric=f"hardware.{event}",
                shape="window_aggregate",
                unit=(str(row.get("unit")) if isinstance(row.get("unit"), str) and row.get("unit") else "event"),
                status=status,
                reasons=reasons,
                records=(record,),
                raw_artifact_ids=artifact_ids,
                provenance={
                    "projection": PROJECTION_SCHEMA_VERSION,
                    "namespace": row.get("namespace"),
                },
            )
        )
    if not streams:
        if global_reasons:
            record = EvidenceRecord(
                record_id="hardware-counters:invalid-window",
                run_id=run_id,
                scope_id=scope_id,
                profiler_id=profiler_id,
                stream_id="hardware-counters",
                shape="window_aggregate",
                status="invalid",
                reasons=global_reasons,
                value=None,
                source_clock_id=timeline.clock_id,
                native_start_ns=start,
                native_end_ns=end,
                canonical_start_ns=start,
                canonical_end_ns=end,
                alignment_error_ns=alignment_error_ns,
                raw_artifact_ids=artifact_ids,
            )
            streams.append(
                EvidenceStream(
                    stream_id="hardware-counters",
                    profiler_id=profiler_id,
                    metric="hardware.counter",
                    shape="window_aggregate",
                    unit="event",
                    status="invalid",
                    reasons=global_reasons,
                    records=(record,),
                    raw_artifact_ids=artifact_ids,
                    provenance={"projection": PROJECTION_SCHEMA_VERSION},
                )
            )
        else:
            streams.append(
                _missing_stream(
                    profiler_id=profiler_id,
                    stream_id="hardware-counters",
                    metric="hardware.counter",
                    shape="window_aggregate",
                    unit="event",
                    reason="hardware perf report contains no requested counters",
                    artifact_ids=artifact_ids,
                )
            )
    return streams


def _phase_streams(
    *,
    rows: Sequence[Mapping[str, Any]] | None,
    clocks: _ClockRegistry,
    explicit_mapping: Mapping[str, Any] | None,
    run_id: str,
    scope_id: str,
    artifact_id: str | None,
) -> list[EvidenceStream]:
    profiler_id = "compat-labeled-phases"
    refs = (artifact_id,) if artifact_id else ()
    if rows is None:
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="labeled-phases",
                metric="workload.phase",
                shape="interval",
                unit="phase",
                reason="retained report references no labeled phase log",
                artifact_ids=refs,
            )
        ]
    header = next((row for row in rows if row.get("record_type") == "fixture"), {})
    reference = _object(header.get("clock_reference"))
    source_clock_id, mapping = clocks.source_clock(
        clock_id="phase-monotonic",
        clock_name="CLOCK_MONOTONIC",
        reference=reference,
        explicit_mapping=explicit_mapping,
        default_reason=(
            "phase-log CLOCK_MONOTONIC namespace/boot identity was not proven " "equal to the canonical host clock"
        ),
    )
    open_phases: dict[str, list[Mapping[str, Any]]] = {}
    records: list[EvidenceRecord] = []
    anomalies: list[str] = []
    for row in rows:
        phase = row.get("phase")
        if not isinstance(phase, str) or not phase:
            continue
        record_type = row.get("record_type")
        if record_type == "phase_start":
            open_phases.setdefault(phase, []).append(row)
        elif record_type == "phase_end":
            starts = open_phases.get(phase, [])
            if not starts:
                anomalies.append(f"phase {phase!r} ended without a start")
                continue
            start_row = starts.pop(0)
            start = _integer(start_row.get("timestamp_ns"))
            end = _integer(row.get("timestamp_ns"))
            if start is None or end is None or end <= start:
                anomalies.append(f"phase {phase!r} has an invalid interval")
                continue
            records.append(
                _timed_record(
                    record_id=f"labeled-phase:{len(records):04d}:{_safe_token(phase)}",
                    run_id=run_id,
                    scope_id=scope_id,
                    profiler_id=profiler_id,
                    stream_id="labeled-phases",
                    shape="interval",
                    value={
                        "phase": phase,
                        "attributes": dict(_object(start_row.get("attributes")) or {}),
                    },
                    source_clock_id=source_clock_id,
                    mapping=mapping,
                    clocks=clocks,
                    native_start_ns=start,
                    native_end_ns=end,
                    artifact_ids=refs,
                    provenance={
                        "schema_version": header.get("schema_version"),
                        "source": "labeled-phase-log",
                    },
                    attributes={"lane": "workload-phases"},
                )
            )
    for phase, starts in open_phases.items():
        if starts:
            anomalies.append(f"phase {phase!r} never ended")
    if not records:
        return [
            _missing_stream(
                profiler_id=profiler_id,
                stream_id="labeled-phases",
                metric="workload.phase",
                shape="interval",
                unit="phase",
                reason="phase log contains no complete positive-duration intervals",
                artifact_ids=refs,
            )
        ]
    status = records[0].status
    reasons = records[0].reasons
    if anomalies and status == "valid":
        invalid_reasons = _unique_reasons(anomalies)
        records = [
            EvidenceRecord(**{**record.__dict__, "status": "invalid", "reasons": invalid_reasons}) for record in records
        ]
        status, reasons = "invalid", invalid_reasons
    elif anomalies:
        reasons = _unique_reasons((*reasons, *anomalies))
        records = [EvidenceRecord(**{**record.__dict__, "reasons": reasons}) for record in records]
    return [
        EvidenceStream(
            stream_id="labeled-phases",
            profiler_id=profiler_id,
            metric="workload.phase",
            shape="interval",
            unit="phase",
            status=status,
            reasons=reasons if status != "valid" else (),
            records=tuple(records),
            raw_artifact_ids=refs,
            provenance={
                "projection": PROJECTION_SCHEMA_VERSION,
                "clock_identity": dict(reference or {}),
            },
        )
    ]


def _referenced_mapping(source: Mapping[str, Any], name: str) -> Mapping[str, Any] | None:
    direct = _object(source.get(f"{name}_clock_mapping"))
    if direct is not None:
        return direct
    mappings = _object(source.get("validated_clock_mappings"))
    return _object(mappings.get(name)) if mappings else None


def _field(source: Mapping[str, Any], profile: Mapping[str, Any] | None, *names: str) -> Any:
    for container in (source, profile or {}):
        for name in names:
            if name in container and container[name] is not None:
                return container[name]
    return None


def _referenced_path(
    registry: _ArtifactRegistry,
    source: Mapping[str, Any],
    profile: Mapping[str, Any] | None,
    *names: str,
) -> Path | None:
    value = _field(source, profile, *names)
    return registry.resolve(value)


def build_evidence_bundle(report_path: Path, output_path: Path | None = None) -> EvidenceBundle:
    """Project one retained pipeline/profile report without executing anything."""
    report_path = report_path.resolve()
    if not report_path.is_file():
        raise EvidenceProjectionError(f"retained report does not exist: {report_path}")
    root = report_path.parent
    if output_path is not None and output_path.resolve().parent != root:
        raise EvidenceProjectionError(
            "projected bundle must be written beside the source report so relative raw-artifact links remain safe"
        )
    if output_path is not None and output_path.resolve() == report_path:
        raise EvidenceProjectionError("output_path must not overwrite a consumed source artifact")
    registry = _ArtifactRegistry(root)
    source, source_artifact = _read_json(report_path, registry, role="source-report")
    run_id = source.get("run_id")
    if not isinstance(run_id, str) or not run_id:
        raise EvidenceProjectionError(
            "retained report must carry a top-level run_id; nested or inferred IDs are not accepted"
        )

    profile: dict[str, Any] | None = None
    profile_artifact: RawArtifact | None = None
    profile_path = registry.resolve(source.get("container_profile_report"))
    if profile_path is not None and profile_path != report_path:
        profile, profile_artifact = _read_json(profile_path, registry, role="container-profile-report")
        profile_run_id = profile.get("run_id")
        if profile_run_id is not None and profile_run_id != run_id:
            raise EvidenceProjectionError("pipeline and referenced container profile have different run_id values")
    elif "cpu" in source or "memory" in source:
        profile = source
        profile_artifact = source_artifact

    metadata: dict[str, Any] = {}
    if profile is not None:
        metadata.update(profile)
    metadata.update(source)

    memory = _object(_field(source, profile, "container_memory", "memory"))
    timeseries: dict[str, Any] | None = None
    timeseries_artifact: RawArtifact | None = None
    if memory is not None:
        samples_path = registry.resolve(memory.get("samples_path"))
        if samples_path is not None:
            timeseries, timeseries_artifact = _read_json(samples_path, registry, role="resource-timeseries")

    scheduler = _object(_field(source, profile, "scheduler_concurrency"))
    scheduler_events: list[dict[str, Any]] | None = None
    scheduler_artifact: RawArtifact | None = None
    if scheduler is not None:
        events_path = registry.resolve(scheduler.get("events_path"))
        if events_path is not None:
            scheduler_events, scheduler_artifact = _read_jsonl(events_path, registry, role="scheduler-events")

    hardware = _object(_field(source, profile, "hardware_perf", "hardware_profile", "hardware"))
    hardware_artifacts: list[RawArtifact] = []
    hardware_path = _referenced_path(
        registry,
        source,
        profile,
        "hardware_perf_report",
        "hardware_perf_report_path",
    )
    if hardware is None and hardware_path is not None:
        hardware, hardware_report_artifact = _read_json(hardware_path, registry, role="hardware-perf-report")
        hardware_artifacts.append(hardware_report_artifact)
    if hardware is not None:
        for field_name, role, media_type in (
            ("stats_path", "hardware-perf-stats", "text/csv"),
            ("stderr_path", "hardware-perf-stderr", "text/plain"),
        ):
            path = registry.resolve(hardware.get(field_name))
            if path is not None and path.is_file():
                hardware_artifacts.append(registry.add(path, role=role, media_type=media_type))

    parsed_lifecycle = _parse_canonical_lifecycle(metadata)
    if parsed_lifecycle is not None:
        timeline, canonical_reference = parsed_lifecycle
    else:
        timeline = _fallback_timeline(timeseries=timeseries, scheduler=scheduler, hardware=hardware)
        canonical_reference = _object(metadata.get("canonical_clock_reference"))
    scope = _scope_from_source(metadata, run_id=run_id, timeseries=timeseries)
    clocks = _ClockRegistry(timeline=timeline, canonical_reference=canonical_reference)

    replay: dict[str, Any] | None = _object(_field(source, profile, "replay"))
    replay_artifact: RawArtifact | None = None
    replay_path = _referenced_path(registry, source, profile, "replay_report")
    if replay_path is not None:
        replay, replay_artifact = _read_json(replay_path, registry, role="replay-report")

    process = _object(_field(source, profile, "process_exec"))
    process_artifacts: list[RawArtifact] = []
    strace = _object(_field(source, profile, "strace"))
    structured_path = registry.resolve(strace.get("structured_report_path") if strace else None)
    if process is None and structured_path is not None:
        process, structured_artifact = _read_json(structured_path, registry, role="process-exec-report")
        process_artifacts.append(structured_artifact)
    elif structured_path is not None and structured_path.is_file():
        process_artifacts.append(
            registry.add(
                structured_path,
                role="process-exec-report",
                media_type="application/json",
            )
        )
    if process is not None:
        process_artifact_fields = _object(process.get("artifacts")) or {}
        report_copy_path = registry.resolve(process_artifact_fields.get("report_path"))
        if report_copy_path is not None and report_copy_path.is_file():
            process_artifacts.append(
                registry.add(
                    report_copy_path,
                    role="process-exec-report",
                    media_type="application/json",
                )
            )
        for value, role, media_type in (
            (process_artifact_fields.get("raw_trace_path"), "process-strace", "text/plain"),
            (strace.get("log_path") if strace else None, "process-strace", "text/plain"),
        ):
            path = registry.resolve(value)
            if path is not None and path.is_file():
                process_artifacts.append(registry.add(path, role=role, media_type=media_type))
    if profile_artifact is not None and process is not None:
        process_artifacts.append(profile_artifact)

    phase_rows: list[dict[str, Any]] | None = None
    phase_artifact: RawArtifact | None = None
    phase_path = _referenced_path(
        registry,
        source,
        profile,
        "phase_log_path",
        "labeled_phase_log_path",
        "phase_log",
    )
    if phase_path is not None:
        phase_rows, phase_artifact = _read_jsonl(phase_path, registry, role="labeled-phase-log")

    cpu = _object(_field(source, profile, "container_cpu", "cpu"))
    grouped_streams: list[tuple[str, list[EvidenceStream]]] = []
    grouped_streams.append(
        (
            "compat-cgroup-cpu",
            _cpu_streams(
                cpu=cpu,
                timeseries=timeseries,
                timeline=timeline,
                run_id=run_id,
                scope_id=scope.scope_id,
                report_artifact_id=source_artifact.artifact_id,
                timeseries_artifact_id=(timeseries_artifact.artifact_id if timeseries_artifact else None),
                timeline_valid=bool(memory is None or memory.get("timeline_valid_for_shape", True)),
                timeline_invalid_reasons=_unique_reasons(
                    _list(memory.get("timeline_invalid_reasons") if memory else None)
                    or (
                        ["resource timeline was marked invalid"]
                        if memory is not None and not memory.get("timeline_valid_for_shape", True)
                        else []
                    )
                ),
            ),
        )
    )
    grouped_streams.append(
        (
            "compat-cgroup-memory",
            _memory_streams(
                memory=memory,
                timeseries=timeseries,
                timeline=timeline,
                run_id=run_id,
                scope_id=scope.scope_id,
                report_artifact_id=source_artifact.artifact_id,
                timeseries_artifact_id=(timeseries_artifact.artifact_id if timeseries_artifact else None),
            ),
        )
    )
    grouped_streams.append(
        (
            "compat-scheduler",
            _scheduler_streams(
                scheduler=scheduler,
                events=scheduler_events,
                timeline=timeline,
                run_id=run_id,
                scope_id=scope.scope_id,
                report_artifact_id=source_artifact.artifact_id,
                events_artifact_id=(scheduler_artifact.artifact_id if scheduler_artifact else None),
            ),
        )
    )
    grouped_streams.append(
        (
            "compat-replay-actions",
            _action_streams(
                replay=replay,
                clocks=clocks,
                explicit_mapping=_referenced_mapping(metadata, "replay"),
                timeline=timeline,
                run_id=run_id,
                scope_id=scope.scope_id,
                artifact_id=(replay_artifact.artifact_id if replay_artifact else None),
            ),
        )
    )
    grouped_streams.append(
        (
            "compat-process-exec",
            _process_streams(
                process=process,
                clocks=clocks,
                explicit_mapping=_referenced_mapping(metadata, "process"),
                run_id=run_id,
                scope_id=scope.scope_id,
                artifact_ids=tuple(sorted({artifact.artifact_id for artifact in process_artifacts})),
            ),
        )
    )
    grouped_streams.append(
        (
            "compat-hardware-perf",
            _hardware_streams(
                hardware=hardware,
                timeline=timeline,
                run_id=run_id,
                scope_id=scope.scope_id,
                artifact_ids=tuple(
                    sorted(
                        {
                            source_artifact.artifact_id,
                            *(artifact.artifact_id for artifact in hardware_artifacts),
                        }
                    )
                ),
            ),
        )
    )
    grouped_streams.append(
        (
            "compat-labeled-phases",
            _phase_streams(
                rows=phase_rows,
                clocks=clocks,
                explicit_mapping=_referenced_mapping(metadata, "phase"),
                run_id=run_id,
                scope_id=scope.scope_id,
                artifact_id=(phase_artifact.artifact_id if phase_artifact else None),
            ),
        )
    )

    streams = tuple(stream for _profiler_id, profiler_streams in grouped_streams for stream in profiler_streams)
    hardware_provenance = _object(hardware.get("provenance")) if hardware is not None else None
    manifests = tuple(
        _manifest(
            profiler_id,
            profiler_streams,
            source_provenance=(hardware_provenance if profiler_id == "compat-hardware-perf" else None),
        )
        for profiler_id, profiler_streams in grouped_streams
    )
    profiler_statuses = {manifest.profiler_id: manifest.status for manifest in manifests}
    source_evidence_status: str | None = None
    source_attachment = _object(metadata.get("evidence_bundle"))
    if source_attachment is not None and source_attachment.get("evidence_status") in {"valid", "partial", "invalid"}:
        source_evidence_status = str(source_attachment["evidence_status"])
    elif isinstance(metadata.get("profiling_valid"), bool):
        source_evidence_status = "valid" if metadata["profiling_valid"] else "invalid"
    if any(status in {"invalid", "dropped"} for status in profiler_statuses.values()):
        evidence_status = "invalid"
    elif source_evidence_status == "invalid":
        evidence_status = "invalid"
    elif any(status == "unalignable" for status in profiler_statuses.values()):
        evidence_status = "partial"
    elif source_evidence_status is not None:
        evidence_status = source_evidence_status
    elif any(status == "missing" for status in profiler_statuses.values()):
        evidence_status = "partial"
    else:
        evidence_status = "valid"
    bundle = EvidenceBundle(
        run_id=run_id,
        scope=scope,
        timeline=timeline,
        clocks=tuple(clocks.clocks.values()),
        clock_mappings=tuple(clocks.mappings.values()),
        profilers=manifests,
        streams=streams,
        artifacts=registry.artifacts,
        provenance={
            "projection_schema_version": PROJECTION_SCHEMA_VERSION,
            "source_report": source_artifact.relative_path,
            "source_report_sha256": source_artifact.sha256,
            "projection_is_read_only": True,
            "scope_identity_limited": bool(scope.extensions.get("scope_identity_limited")),
        },
        extensions={
            "source_schema_version": source.get("schema_version"),
            "referenced_profile_schema_version": (profile.get("schema_version") if profile is not None else None),
            "evidence_status": evidence_status,
            "all_selected_profilers_valid": evidence_status == "valid",
            "all_selected_profilers_usable": evidence_status != "invalid",
            "profiler_statuses": profiler_statuses,
        },
    )
    if output_path is not None:
        resolved_output = output_path.resolve()
        if registry.contains(resolved_output):
            raise EvidenceProjectionError("output_path must not overwrite a consumed source artifact")
        write_bundle(bundle, resolved_output)
    return bundle


__all__ = [
    "EvidenceProjectionError",
    "PROJECTION_SCHEMA_VERSION",
    "build_evidence_bundle",
]
