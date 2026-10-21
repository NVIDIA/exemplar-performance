# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, versioned evidence shared by profilers and report renderers.

The contract deliberately separates native timestamps from the run's canonical
``CLOCK_MONOTONIC`` timeline.  Untimed facts remain untimed, and evidence from a
clock that cannot be mapped retains its native timing without being placed on
the canonical axis.
"""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Mapping, Sequence

EVIDENCE_BUNDLE_SCHEMA_VERSION = "tasktrace-profile-evidence-bundle.v1"
CANONICAL_CLOCK_NAME = "CLOCK_MONOTONIC"

RECORD_SHAPES = frozenset(
    {
        "scalar",
        "point_event",
        "interval",
        "sampled_gauge",
        "cumulative_counter",
        "window_aggregate",
    }
)
EVIDENCE_STATUSES = frozenset({"valid", "invalid", "missing", "dropped", "unalignable"})
TEMPORAL_SHAPES = RECORD_SHAPES - {"scalar"}
POINT_SHAPES = frozenset({"point_event", "sampled_gauge", "cumulative_counter"})
RANGE_SHAPES = frozenset({"interval", "window_aggregate"})


class EvidenceValidationError(ValueError):
    """Raised when evidence would make an ambiguous or false claim."""


def _require_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EvidenceValidationError(f"{field_name} must be a non-empty string")
    return value


def _require_int(value: Any, field_name: str, *, minimum: int | None = None) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise EvidenceValidationError(f"{field_name} must be an integer")
    if minimum is not None and value < minimum:
        raise EvidenceValidationError(f"{field_name} must be at least {minimum}")
    return value


def _validate_json_value(value: Any, field_name: str) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise EvidenceValidationError(f"{field_name} contains a non-finite number")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json_value(item, f"{field_name}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise EvidenceValidationError(f"{field_name} contains a non-string key")
            _validate_json_value(item, f"{field_name}.{key}")
        return
    raise EvidenceValidationError(f"{field_name} must contain only JSON-compatible values")


def _validate_mapping(value: Mapping[str, Any], field_name: str) -> None:
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"{field_name} must be a JSON object")
    _validate_json_value(value, field_name)


def _normalize_reasons(status: str, reasons: Sequence[str], field_name: str) -> tuple[str, ...]:
    if status not in EVIDENCE_STATUSES:
        raise EvidenceValidationError(f"{field_name}.status must be one of: {', '.join(sorted(EVIDENCE_STATUSES))}")
    normalized = tuple(sorted({_require_text(reason, f"{field_name}.reasons") for reason in reasons}))
    if status == "valid" and normalized:
        raise EvidenceValidationError(f"{field_name} valid evidence cannot have failure reasons")
    if status != "valid" and not normalized:
        raise EvidenceValidationError(f"{field_name} {status} evidence requires a reason")
    return normalized


def _unique_texts(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    normalized = tuple(sorted(_require_text(value, field_name) for value in values))
    if len(set(normalized)) != len(normalized):
        raise EvidenceValidationError(f"{field_name} must not contain duplicates")
    return normalized


def _expect_object(payload: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise EvidenceValidationError(f"{field_name} must be a JSON object")
    return payload


def _only_keys(payload: Mapping[str, Any], *, allowed: set[str], required: set[str], field_name: str) -> None:
    missing = sorted(required - set(payload))
    unknown = sorted(set(payload) - allowed)
    if missing:
        raise EvidenceValidationError(f"{field_name} is missing: {', '.join(missing)}")
    if unknown:
        raise EvidenceValidationError(f"{field_name} has unknown fields: {', '.join(unknown)}")


@dataclass(frozen=True)
class MeasuredScope:
    """Stable identity of the sandbox subtree observed by every profiler."""

    scope_id: str
    kind: str
    identity: Mapping[str, Any]
    includes_descendants: bool = True
    description: str | None = None
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.scope_id, "scope.scope_id")
        _require_text(self.kind, "scope.kind")
        if not isinstance(self.includes_descendants, bool):
            raise EvidenceValidationError("scope.includes_descendants must be boolean")
        if not self.identity:
            raise EvidenceValidationError("scope.identity must not be empty")
        _validate_mapping(self.identity, "scope.identity")
        if self.description is not None:
            _require_text(self.description, "scope.description")
        _validate_mapping(self.extensions, "scope.extensions")

    @classmethod
    def from_dict(cls, payload: Any) -> "MeasuredScope":
        data = _expect_object(payload, "scope")
        _only_keys(
            data,
            allowed={"scope_id", "kind", "identity", "includes_descendants", "description", "extensions"},
            required={"scope_id", "kind", "identity"},
            field_name="scope",
        )
        return cls(
            scope_id=data["scope_id"],
            kind=data["kind"],
            identity=data["identity"],
            includes_descendants=data.get("includes_descendants", True),
            description=data.get("description"),
            extensions=data.get("extensions", {}),
        )


@dataclass(frozen=True)
class LifecycleMarker:
    name: str
    canonical_timestamp_ns: int
    alignment_error_ns: int = 0
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.name, "marker.name")
        _require_int(self.canonical_timestamp_ns, "marker.canonical_timestamp_ns", minimum=0)
        _require_int(self.alignment_error_ns, "marker.alignment_error_ns", minimum=0)
        _validate_mapping(self.provenance, "marker.provenance")

    @classmethod
    def from_dict(cls, payload: Any) -> "LifecycleMarker":
        data = _expect_object(payload, "marker")
        _only_keys(
            data,
            allowed={"name", "canonical_timestamp_ns", "alignment_error_ns", "provenance"},
            required={"name", "canonical_timestamp_ns"},
            field_name="marker",
        )
        return cls(
            name=data["name"],
            canonical_timestamp_ns=data["canonical_timestamp_ns"],
            alignment_error_ns=data.get("alignment_error_ns", 0),
            provenance=data.get("provenance", {}),
        )


@dataclass(frozen=True)
class CanonicalTimeline:
    """The run-relative axis, backed only by Linux ``CLOCK_MONOTONIC``."""

    clock_id: str
    origin_ns: int
    start_ns: int
    end_ns: int
    lifecycle_markers: tuple[LifecycleMarker, ...] = ()

    def __post_init__(self) -> None:
        _require_text(self.clock_id, "timeline.clock_id")
        _require_int(self.origin_ns, "timeline.origin_ns", minimum=0)
        _require_int(self.start_ns, "timeline.start_ns", minimum=0)
        _require_int(self.end_ns, "timeline.end_ns", minimum=0)
        if self.end_ns <= self.start_ns:
            raise EvidenceValidationError("timeline end_ns must be greater than start_ns")
        if not self.start_ns <= self.origin_ns <= self.end_ns:
            raise EvidenceValidationError("timeline origin_ns must fall within the run window")
        markers = tuple(sorted(self.lifecycle_markers, key=lambda marker: (marker.canonical_timestamp_ns, marker.name)))
        names = [marker.name for marker in markers]
        if len(set(names)) != len(names):
            raise EvidenceValidationError("timeline lifecycle marker names must be unique")
        for marker in markers:
            if not self.start_ns <= marker.canonical_timestamp_ns <= self.end_ns:
                raise EvidenceValidationError(f"lifecycle marker {marker.name!r} falls outside the run window")
        object.__setattr__(self, "lifecycle_markers", markers)

    @property
    def clock_name(self) -> str:
        return CANONICAL_CLOCK_NAME

    def to_dict(self) -> dict[str, Any]:
        return {
            "clock_id": self.clock_id,
            "clock_name": CANONICAL_CLOCK_NAME,
            "origin_ns": self.origin_ns,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "lifecycle_markers": [asdict(marker) for marker in self.lifecycle_markers],
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "CanonicalTimeline":
        data = _expect_object(payload, "timeline")
        _only_keys(
            data,
            allowed={"clock_id", "clock_name", "origin_ns", "start_ns", "end_ns", "lifecycle_markers"},
            required={"clock_id", "clock_name", "origin_ns", "start_ns", "end_ns"},
            field_name="timeline",
        )
        if data["clock_name"] != CANONICAL_CLOCK_NAME:
            raise EvidenceValidationError(f"timeline.clock_name must be {CANONICAL_CLOCK_NAME}")
        markers = data.get("lifecycle_markers", [])
        if not isinstance(markers, list):
            raise EvidenceValidationError("timeline.lifecycle_markers must be a list")
        return cls(
            clock_id=data["clock_id"],
            origin_ns=data["origin_ns"],
            start_ns=data["start_ns"],
            end_ns=data["end_ns"],
            lifecycle_markers=tuple(LifecycleMarker.from_dict(item) for item in markers),
        )


@dataclass(frozen=True)
class ClockReference:
    clock_id: str
    clock_name: str
    resolution_ns: int
    namespace_identity: str | None = None
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.clock_id, "clock.clock_id")
        _require_text(self.clock_name, "clock.clock_name")
        _require_int(self.resolution_ns, "clock.resolution_ns", minimum=1)
        if self.namespace_identity is not None:
            _require_text(self.namespace_identity, "clock.namespace_identity")
        _validate_mapping(self.provenance, "clock.provenance")

    @classmethod
    def from_dict(cls, payload: Any) -> "ClockReference":
        data = _expect_object(payload, "clock")
        _only_keys(
            data,
            allowed={"clock_id", "clock_name", "resolution_ns", "namespace_identity", "provenance"},
            required={"clock_id", "clock_name", "resolution_ns"},
            field_name="clock",
        )
        return cls(
            clock_id=data["clock_id"],
            clock_name=data["clock_name"],
            resolution_ns=data["resolution_ns"],
            namespace_identity=data.get("namespace_identity"),
            provenance=data.get("provenance", {}),
        )


@dataclass(frozen=True)
class ClockMapping:
    """Affine native-to-canonical mapping, or an explicit inability to align."""

    source_clock_id: str
    canonical_clock_id: str
    status: str
    method: str
    scale_numerator: int | None = None
    scale_denominator: int | None = None
    offset_ns: int | None = None
    max_error_ns: int | None = None
    reasons: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.source_clock_id, "clock_mapping.source_clock_id")
        _require_text(self.canonical_clock_id, "clock_mapping.canonical_clock_id")
        _require_text(self.method, "clock_mapping.method")
        if self.status not in {"valid", "unalignable"}:
            raise EvidenceValidationError("clock_mapping.status must be valid or unalignable")
        reasons = _normalize_reasons(self.status, self.reasons, "clock_mapping")
        object.__setattr__(self, "reasons", reasons)
        transform = (
            self.scale_numerator,
            self.scale_denominator,
            self.offset_ns,
            self.max_error_ns,
        )
        if self.status == "valid":
            if any(value is None for value in transform):
                raise EvidenceValidationError("valid clock mapping requires a complete affine transform")
            _require_int(self.scale_numerator, "clock_mapping.scale_numerator", minimum=1)
            _require_int(self.scale_denominator, "clock_mapping.scale_denominator", minimum=1)
            _require_int(self.offset_ns, "clock_mapping.offset_ns")
            _require_int(self.max_error_ns, "clock_mapping.max_error_ns", minimum=0)
        elif any(value is not None for value in transform):
            raise EvidenceValidationError("unalignable clock mapping must not contain a fabricated transform")
        _validate_mapping(self.provenance, "clock_mapping.provenance")

    def to_canonical_ns(self, native_ns: int) -> int:
        """Map one native timestamp with deterministic nearest-nanosecond rounding."""
        _require_int(native_ns, "native_ns", minimum=0)
        if self.status != "valid":
            raise EvidenceValidationError("unalignable clock has no canonical timestamp")
        assert self.scale_numerator is not None
        assert self.scale_denominator is not None
        assert self.offset_ns is not None
        scaled_numerator = native_ns * self.scale_numerator
        quotient, remainder = divmod(scaled_numerator, self.scale_denominator)
        if remainder * 2 >= self.scale_denominator:
            quotient += 1
        return quotient + self.offset_ns

    @classmethod
    def from_dict(cls, payload: Any) -> "ClockMapping":
        data = _expect_object(payload, "clock_mapping")
        allowed = {
            "source_clock_id",
            "canonical_clock_id",
            "status",
            "method",
            "scale_numerator",
            "scale_denominator",
            "offset_ns",
            "max_error_ns",
            "reasons",
            "provenance",
        }
        _only_keys(
            data,
            allowed=allowed,
            required={"source_clock_id", "canonical_clock_id", "status", "method"},
            field_name="clock_mapping",
        )
        return cls(
            source_clock_id=data["source_clock_id"],
            canonical_clock_id=data["canonical_clock_id"],
            status=data["status"],
            method=data["method"],
            scale_numerator=data.get("scale_numerator"),
            scale_denominator=data.get("scale_denominator"),
            offset_ns=data.get("offset_ns"),
            max_error_ns=data.get("max_error_ns"),
            reasons=tuple(data.get("reasons", [])),
            provenance=data.get("provenance", {}),
        )


@dataclass(frozen=True)
class AcquisitionBounds:
    canonical_start_ns: int
    canonical_end_ns: int

    def __post_init__(self) -> None:
        _require_int(self.canonical_start_ns, "acquisition_bounds.canonical_start_ns", minimum=0)
        _require_int(self.canonical_end_ns, "acquisition_bounds.canonical_end_ns", minimum=0)
        if self.canonical_end_ns < self.canonical_start_ns:
            raise EvidenceValidationError("acquisition bounds end before they start")

    @classmethod
    def from_dict(cls, payload: Any) -> "AcquisitionBounds":
        data = _expect_object(payload, "acquisition_bounds")
        _only_keys(
            data,
            allowed={"canonical_start_ns", "canonical_end_ns"},
            required={"canonical_start_ns", "canonical_end_ns"},
            field_name="acquisition_bounds",
        )
        return cls(data["canonical_start_ns"], data["canonical_end_ns"])


@dataclass(frozen=True)
class Cadence:
    requested_interval_ns: int
    maximum_acceptable_gap_ns: int
    method: str

    def __post_init__(self) -> None:
        _require_int(self.requested_interval_ns, "cadence.requested_interval_ns", minimum=1)
        _require_int(
            self.maximum_acceptable_gap_ns,
            "cadence.maximum_acceptable_gap_ns",
            minimum=self.requested_interval_ns,
        )
        _require_text(self.method, "cadence.method")

    @classmethod
    def from_dict(cls, payload: Any) -> "Cadence":
        data = _expect_object(payload, "cadence")
        _only_keys(
            data,
            allowed={"requested_interval_ns", "maximum_acceptable_gap_ns", "method"},
            required={"requested_interval_ns", "maximum_acceptable_gap_ns", "method"},
            field_name="cadence",
        )
        return cls(**data)


@dataclass(frozen=True)
class Gap:
    clock_id: str
    start_ns: int
    end_ns: int
    reason: str
    canonical: bool = True
    dropped_records: int | None = None

    def __post_init__(self) -> None:
        _require_text(self.clock_id, "gap.clock_id")
        _require_int(self.start_ns, "gap.start_ns", minimum=0)
        _require_int(self.end_ns, "gap.end_ns", minimum=0)
        if self.end_ns <= self.start_ns:
            raise EvidenceValidationError("gap end_ns must be greater than start_ns")
        _require_text(self.reason, "gap.reason")
        if not isinstance(self.canonical, bool):
            raise EvidenceValidationError("gap.canonical must be boolean")
        if self.dropped_records is not None:
            _require_int(self.dropped_records, "gap.dropped_records", minimum=1)

    @classmethod
    def from_dict(cls, payload: Any) -> "Gap":
        data = _expect_object(payload, "gap")
        _only_keys(
            data,
            allowed={"clock_id", "start_ns", "end_ns", "reason", "canonical", "dropped_records"},
            required={"clock_id", "start_ns", "end_ns", "reason"},
            field_name="gap",
        )
        return cls(
            clock_id=data["clock_id"],
            start_ns=data["start_ns"],
            end_ns=data["end_ns"],
            reason=data["reason"],
            canonical=data.get("canonical", True),
            dropped_records=data.get("dropped_records"),
        )


@dataclass(frozen=True)
class RawArtifact:
    artifact_id: str
    relative_path: str
    sha256: str
    size_bytes: int
    media_type: str
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.artifact_id, "artifact.artifact_id")
        path = PurePosixPath(_require_text(self.relative_path, "artifact.relative_path"))
        if path.is_absolute() or ".." in path.parts or str(path) in {".", ""}:
            raise EvidenceValidationError("artifact.relative_path must be a safe relative path")
        if (
            not isinstance(self.sha256, str)
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise EvidenceValidationError("artifact.sha256 must be a lowercase SHA-256 digest")
        _require_int(self.size_bytes, "artifact.size_bytes", minimum=0)
        _require_text(self.media_type, "artifact.media_type")
        _validate_mapping(self.provenance, "artifact.provenance")

    @classmethod
    def from_path(
        cls,
        *,
        artifact_id: str,
        path: Path,
        base_dir: Path,
        media_type: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> "RawArtifact":
        try:
            relative = path.resolve().relative_to(base_dir.resolve())
        except ValueError as exc:
            raise EvidenceValidationError("raw artifact must be inside the bundle directory") from exc
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        return cls(
            artifact_id=artifact_id,
            relative_path=relative.as_posix(),
            sha256=digest,
            size_bytes=path.stat().st_size,
            media_type=media_type,
            provenance=provenance or {},
        )

    @classmethod
    def from_dict(cls, payload: Any) -> "RawArtifact":
        data = _expect_object(payload, "artifact")
        _only_keys(
            data,
            allowed={"artifact_id", "relative_path", "sha256", "size_bytes", "media_type", "provenance"},
            required={"artifact_id", "relative_path", "sha256", "size_bytes", "media_type"},
            field_name="artifact",
        )
        return cls(
            artifact_id=data["artifact_id"],
            relative_path=data["relative_path"],
            sha256=data["sha256"],
            size_bytes=data["size_bytes"],
            media_type=data["media_type"],
            provenance=data.get("provenance", {}),
        )


@dataclass(frozen=True)
class EvidenceRecord:
    """One typed fact. Scalars are necessarily untimed."""

    record_id: str
    run_id: str
    scope_id: str
    profiler_id: str
    stream_id: str
    shape: str
    status: str
    value: Any
    reasons: tuple[str, ...] = ()
    source_clock_id: str | None = None
    native_timestamp_ns: int | None = None
    native_start_ns: int | None = None
    native_end_ns: int | None = None
    canonical_timestamp_ns: int | None = None
    canonical_start_ns: int | None = None
    canonical_end_ns: int | None = None
    acquisition_bounds: AcquisitionBounds | None = None
    alignment_error_ns: int | None = None
    raw_artifact_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("record.record_id", self.record_id),
            ("record.run_id", self.run_id),
            ("record.scope_id", self.scope_id),
            ("record.profiler_id", self.profiler_id),
            ("record.stream_id", self.stream_id),
        ):
            _require_text(value, name)
        if self.shape not in RECORD_SHAPES:
            raise EvidenceValidationError(f"record.shape must be one of: {', '.join(sorted(RECORD_SHAPES))}")
        if self.status == "missing":
            raise EvidenceValidationError("missing evidence belongs on a stream, not in a fabricated record")
        reasons = _normalize_reasons(self.status, self.reasons, "record")
        object.__setattr__(self, "reasons", reasons)
        if self.status in {"valid", "unalignable"} and self.value is None:
            raise EvidenceValidationError(f"{self.status} record must contain a value")
        if self.status == "dropped" and self.value is not None:
            raise EvidenceValidationError("dropped record must not fabricate a value")
        _validate_json_value(self.value, "record.value")
        object.__setattr__(self, "raw_artifact_ids", _unique_texts(self.raw_artifact_ids, "record.raw_artifact_ids"))
        _validate_mapping(self.provenance, "record.provenance")
        _validate_mapping(self.attributes, "record.attributes")
        self._validate_timing()

    def _validate_timing(self) -> None:
        timing_values = (
            self.source_clock_id,
            self.native_timestamp_ns,
            self.native_start_ns,
            self.native_end_ns,
            self.canonical_timestamp_ns,
            self.canonical_start_ns,
            self.canonical_end_ns,
            self.acquisition_bounds,
            self.alignment_error_ns,
        )
        if self.shape == "scalar":
            if any(value is not None for value in timing_values):
                raise EvidenceValidationError("scalar record must remain untimed")
            return

        if self.source_clock_id is None:
            raise EvidenceValidationError("temporal record requires source_clock_id")
        _require_text(self.source_clock_id, "record.source_clock_id")
        for name, value in (
            ("native_timestamp_ns", self.native_timestamp_ns),
            ("native_start_ns", self.native_start_ns),
            ("native_end_ns", self.native_end_ns),
            ("canonical_timestamp_ns", self.canonical_timestamp_ns),
            ("canonical_start_ns", self.canonical_start_ns),
            ("canonical_end_ns", self.canonical_end_ns),
        ):
            if value is not None:
                _require_int(value, f"record.{name}", minimum=0)

        native_point = self.native_timestamp_ns is not None
        native_range = self.native_start_ns is not None or self.native_end_ns is not None
        canonical_point = self.canonical_timestamp_ns is not None
        canonical_range = self.canonical_start_ns is not None or self.canonical_end_ns is not None

        if self.shape in POINT_SHAPES:
            if not native_point or native_range:
                raise EvidenceValidationError(f"{self.shape} requires one native timestamp")
            if canonical_range:
                raise EvidenceValidationError(f"{self.shape} cannot contain a canonical range")
        else:
            if native_point or self.native_start_ns is None or self.native_end_ns is None:
                raise EvidenceValidationError(f"{self.shape} requires a native start/end range")
            if self.native_end_ns <= self.native_start_ns:
                raise EvidenceValidationError("native interval end must be greater than start")
            if canonical_point or self.acquisition_bounds is not None:
                raise EvidenceValidationError(f"{self.shape} requires a canonical range, not a point/acquisition bound")

        if self.status == "unalignable":
            if canonical_point or canonical_range or self.acquisition_bounds is not None:
                raise EvidenceValidationError("unalignable record must retain native timing without canonical timing")
            if self.alignment_error_ns is not None:
                raise EvidenceValidationError("unalignable record cannot claim a canonical alignment error")
            return

        if self.alignment_error_ns is None:
            raise EvidenceValidationError("aligned temporal record requires alignment_error_ns")
        _require_int(self.alignment_error_ns, "record.alignment_error_ns", minimum=0)
        if self.shape in POINT_SHAPES:
            if canonical_point == (self.acquisition_bounds is not None):
                raise EvidenceValidationError(
                    f"{self.shape} requires exactly one canonical timestamp or acquisition bounds"
                )
        else:
            if self.canonical_start_ns is None or self.canonical_end_ns is None:
                raise EvidenceValidationError(f"{self.shape} requires a canonical start/end range")
            if self.canonical_end_ns <= self.canonical_start_ns:
                raise EvidenceValidationError("canonical interval end must be greater than start")

    def sort_key(self) -> tuple[int, int, str]:
        if self.canonical_timestamp_ns is not None:
            return (0, self.canonical_timestamp_ns, self.record_id)
        if self.acquisition_bounds is not None:
            return (0, self.acquisition_bounds.canonical_start_ns, self.record_id)
        if self.canonical_start_ns is not None:
            return (0, self.canonical_start_ns, self.record_id)
        if self.native_timestamp_ns is not None:
            return (1, self.native_timestamp_ns, self.record_id)
        if self.native_start_ns is not None:
            return (1, self.native_start_ns, self.record_id)
        return (2, 0, self.record_id)

    @classmethod
    def from_dict(cls, payload: Any) -> "EvidenceRecord":
        data = _expect_object(payload, "record")
        allowed = {
            "record_id",
            "run_id",
            "scope_id",
            "profiler_id",
            "stream_id",
            "shape",
            "status",
            "value",
            "reasons",
            "source_clock_id",
            "native_timestamp_ns",
            "native_start_ns",
            "native_end_ns",
            "canonical_timestamp_ns",
            "canonical_start_ns",
            "canonical_end_ns",
            "acquisition_bounds",
            "alignment_error_ns",
            "raw_artifact_ids",
            "provenance",
            "attributes",
        }
        _only_keys(
            data,
            allowed=allowed,
            required={
                "record_id",
                "run_id",
                "scope_id",
                "profiler_id",
                "stream_id",
                "shape",
                "status",
                "value",
            },
            field_name="record",
        )
        bounds = data.get("acquisition_bounds")
        return cls(
            record_id=data["record_id"],
            run_id=data["run_id"],
            scope_id=data["scope_id"],
            profiler_id=data["profiler_id"],
            stream_id=data["stream_id"],
            shape=data["shape"],
            status=data["status"],
            value=data["value"],
            reasons=tuple(data.get("reasons", [])),
            source_clock_id=data.get("source_clock_id"),
            native_timestamp_ns=data.get("native_timestamp_ns"),
            native_start_ns=data.get("native_start_ns"),
            native_end_ns=data.get("native_end_ns"),
            canonical_timestamp_ns=data.get("canonical_timestamp_ns"),
            canonical_start_ns=data.get("canonical_start_ns"),
            canonical_end_ns=data.get("canonical_end_ns"),
            acquisition_bounds=(AcquisitionBounds.from_dict(bounds) if bounds is not None else None),
            alignment_error_ns=data.get("alignment_error_ns"),
            raw_artifact_ids=tuple(data.get("raw_artifact_ids", [])),
            provenance=data.get("provenance", {}),
            attributes=data.get("attributes", {}),
        )


@dataclass(frozen=True)
class EvidenceStream:
    stream_id: str
    profiler_id: str
    metric: str
    shape: str
    unit: str
    status: str
    records: tuple[EvidenceRecord, ...] = ()
    reasons: tuple[str, ...] = ()
    cadence: Cadence | None = None
    gaps: tuple[Gap, ...] = ()
    raw_artifact_ids: tuple[str, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    display: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, value in (
            ("stream.stream_id", self.stream_id),
            ("stream.profiler_id", self.profiler_id),
            ("stream.metric", self.metric),
            ("stream.unit", self.unit),
        ):
            _require_text(value, name)
        if self.shape not in RECORD_SHAPES:
            raise EvidenceValidationError(f"stream.shape must be one of: {', '.join(sorted(RECORD_SHAPES))}")
        reasons = _normalize_reasons(self.status, self.reasons, "stream")
        object.__setattr__(self, "reasons", reasons)
        records = tuple(sorted(self.records, key=EvidenceRecord.sort_key))
        ids = [record.record_id for record in records]
        if len(set(ids)) != len(ids):
            raise EvidenceValidationError(f"stream {self.stream_id!r} has duplicate record IDs")
        if self.status == "missing" and records:
            raise EvidenceValidationError("missing stream must not contain fabricated records")
        if self.status == "valid" and (not records or any(record.status != "valid" for record in records)):
            raise EvidenceValidationError("valid stream requires only valid records")
        if self.status == "invalid" and not any(record.status == "invalid" for record in records):
            raise EvidenceValidationError("invalid stream requires an invalid record")
        if self.status == "dropped" and records and not any(record.status == "dropped" for record in records):
            raise EvidenceValidationError("dropped stream records must expose a dropped record")
        if self.status == "unalignable" and (not records or any(record.status != "unalignable" for record in records)):
            raise EvidenceValidationError("unalignable stream requires only unalignable records")
        for record in records:
            if record.stream_id != self.stream_id or record.profiler_id != self.profiler_id:
                raise EvidenceValidationError("record stream/profiler identity does not match its stream")
            if record.shape != self.shape:
                raise EvidenceValidationError("record shape does not match its stream")
        object.__setattr__(self, "records", records)
        if self.cadence is not None and self.shape not in {"sampled_gauge", "cumulative_counter"}:
            raise EvidenceValidationError("cadence is only valid for sampled gauge/counter streams")
        if self.shape == "scalar" and self.gaps:
            raise EvidenceValidationError("untimed scalar stream cannot contain timeline gaps")
        object.__setattr__(
            self,
            "gaps",
            tuple(sorted(self.gaps, key=lambda gap: (not gap.canonical, gap.start_ns, gap.end_ns, gap.reason))),
        )
        object.__setattr__(self, "raw_artifact_ids", _unique_texts(self.raw_artifact_ids, "stream.raw_artifact_ids"))
        _validate_mapping(self.provenance, "stream.provenance")
        _validate_mapping(self.display, "stream.display")
        _validate_mapping(self.extensions, "stream.extensions")

    @classmethod
    def from_dict(cls, payload: Any) -> "EvidenceStream":
        data = _expect_object(payload, "stream")
        allowed = {
            "stream_id",
            "profiler_id",
            "metric",
            "shape",
            "unit",
            "status",
            "records",
            "reasons",
            "cadence",
            "gaps",
            "raw_artifact_ids",
            "provenance",
            "display",
            "extensions",
        }
        _only_keys(
            data,
            allowed=allowed,
            required={"stream_id", "profiler_id", "metric", "shape", "unit", "status"},
            field_name="stream",
        )
        raw_records = data.get("records", [])
        raw_gaps = data.get("gaps", [])
        if not isinstance(raw_records, list) or not isinstance(raw_gaps, list):
            raise EvidenceValidationError("stream records and gaps must be lists")
        cadence = data.get("cadence")
        return cls(
            stream_id=data["stream_id"],
            profiler_id=data["profiler_id"],
            metric=data["metric"],
            shape=data["shape"],
            unit=data["unit"],
            status=data["status"],
            records=tuple(EvidenceRecord.from_dict(item) for item in raw_records),
            reasons=tuple(data.get("reasons", [])),
            cadence=Cadence.from_dict(cadence) if cadence is not None else None,
            gaps=tuple(Gap.from_dict(item) for item in raw_gaps),
            raw_artifact_ids=tuple(data.get("raw_artifact_ids", [])),
            provenance=data.get("provenance", {}),
            display=data.get("display", {}),
            extensions=data.get("extensions", {}),
        )


@dataclass(frozen=True)
class ProfilerManifest:
    profiler_id: str
    profiler_version: str
    status: str
    stream_ids: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    intrusive: bool = False
    provenance: Mapping[str, Any] = field(default_factory=dict)
    configuration: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require_text(self.profiler_id, "profiler.profiler_id")
        _require_text(self.profiler_version, "profiler.profiler_version")
        object.__setattr__(self, "reasons", _normalize_reasons(self.status, self.reasons, "profiler"))
        object.__setattr__(self, "stream_ids", _unique_texts(self.stream_ids, "profiler.stream_ids"))
        if not isinstance(self.intrusive, bool):
            raise EvidenceValidationError("profiler.intrusive must be boolean")
        _validate_mapping(self.provenance, "profiler.provenance")
        _validate_mapping(self.configuration, "profiler.configuration")
        _validate_mapping(self.extensions, "profiler.extensions")

    @classmethod
    def from_dict(cls, payload: Any) -> "ProfilerManifest":
        data = _expect_object(payload, "profiler")
        allowed = {
            "profiler_id",
            "profiler_version",
            "status",
            "stream_ids",
            "reasons",
            "intrusive",
            "provenance",
            "configuration",
            "extensions",
        }
        _only_keys(
            data,
            allowed=allowed,
            required={"profiler_id", "profiler_version", "status"},
            field_name="profiler",
        )
        return cls(
            profiler_id=data["profiler_id"],
            profiler_version=data["profiler_version"],
            status=data["status"],
            stream_ids=tuple(data.get("stream_ids", [])),
            reasons=tuple(data.get("reasons", [])),
            intrusive=data.get("intrusive", False),
            provenance=data.get("provenance", {}),
            configuration=data.get("configuration", {}),
            extensions=data.get("extensions", {}),
        )


@dataclass(frozen=True)
class EvidenceBundle:
    run_id: str
    scope: MeasuredScope
    timeline: CanonicalTimeline
    clocks: tuple[ClockReference, ...]
    clock_mappings: tuple[ClockMapping, ...]
    profilers: tuple[ProfilerManifest, ...]
    streams: tuple[EvidenceStream, ...]
    artifacts: tuple[RawArtifact, ...] = ()
    provenance: Mapping[str, Any] = field(default_factory=dict)
    extensions: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = EVIDENCE_BUNDLE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        _require_text(self.run_id, "bundle.run_id")
        try:
            uuid.UUID(self.run_id)
        except (ValueError, AttributeError) as exc:
            raise EvidenceValidationError("bundle.run_id must be a UUID") from exc
        if self.schema_version != EVIDENCE_BUNDLE_SCHEMA_VERSION:
            raise EvidenceValidationError(f"unsupported evidence bundle schema: {self.schema_version!r}")
        _validate_mapping(self.provenance, "bundle.provenance")
        _validate_mapping(self.extensions, "bundle.extensions")

        clocks = tuple(sorted(self.clocks, key=lambda clock: clock.clock_id))
        clock_by_id = self._unique_by(clocks, "clock_id", "clock")
        canonical_clock = clock_by_id.get(self.timeline.clock_id)
        if canonical_clock is None:
            raise EvidenceValidationError("canonical timeline clock is absent from clocks")
        if canonical_clock.clock_name != CANONICAL_CLOCK_NAME:
            raise EvidenceValidationError("canonical timeline clock must be CLOCK_MONOTONIC")
        object.__setattr__(self, "clocks", clocks)

        mappings = tuple(sorted(self.clock_mappings, key=lambda mapping: mapping.source_clock_id))
        mapping_by_source = self._unique_by(mappings, "source_clock_id", "clock mapping")
        for mapping in mappings:
            if mapping.source_clock_id not in clock_by_id:
                raise EvidenceValidationError("clock mapping references an unknown source clock")
            if mapping.canonical_clock_id != self.timeline.clock_id:
                raise EvidenceValidationError("clock mapping targets a non-canonical clock")
            if mapping.source_clock_id == self.timeline.clock_id:
                raise EvidenceValidationError("canonical clock must not map to itself")
        for clock_id in clock_by_id:
            if clock_id != self.timeline.clock_id and clock_id not in mapping_by_source:
                raise EvidenceValidationError(f"source clock {clock_id!r} has no clock mapping")
        object.__setattr__(self, "clock_mappings", mappings)

        artifacts = tuple(sorted(self.artifacts, key=lambda artifact: artifact.artifact_id))
        artifact_by_id = self._unique_by(artifacts, "artifact_id", "artifact")
        object.__setattr__(self, "artifacts", artifacts)

        streams = tuple(sorted(self.streams, key=lambda stream: stream.stream_id))
        stream_by_id = self._unique_by(streams, "stream_id", "stream")
        object.__setattr__(self, "streams", streams)
        profilers = tuple(sorted(self.profilers, key=lambda profiler: profiler.profiler_id))
        profiler_by_id = self._unique_by(profilers, "profiler_id", "profiler")
        object.__setattr__(self, "profilers", profilers)

        expected_stream_ids: set[str] = set()
        for profiler in profilers:
            expected_stream_ids.update(profiler.stream_ids)
            actual = tuple(stream.stream_id for stream in streams if stream.profiler_id == profiler.profiler_id)
            if actual != profiler.stream_ids:
                raise EvidenceValidationError(
                    f"profiler {profiler.profiler_id!r} manifest stream IDs do not match bundle streams"
                )
            if profiler.status == "missing" and any(
                stream.status != "missing" for stream in streams if stream.profiler_id == profiler.profiler_id
            ):
                raise EvidenceValidationError("missing profiler may contain only missing streams")
        if expected_stream_ids != set(stream_by_id):
            raise EvidenceValidationError("bundle contains orphaned or unlisted streams")

        for stream in streams:
            if stream.profiler_id not in profiler_by_id:
                raise EvidenceValidationError("stream references an unknown profiler")
            self._validate_artifact_refs(stream.raw_artifact_ids, artifact_by_id)
            for gap in stream.gaps:
                self._validate_gap(gap, clock_by_id)
            for record in stream.records:
                if record.run_id != self.run_id:
                    raise EvidenceValidationError("record belongs to a different run")
                if record.scope_id != self.scope.scope_id:
                    raise EvidenceValidationError("record belongs to a different measured scope")
                if record.source_clock_id is not None:
                    if record.source_clock_id not in clock_by_id:
                        raise EvidenceValidationError("record references an unknown source clock")
                    mapping = mapping_by_source.get(record.source_clock_id)
                    if record.source_clock_id == self.timeline.clock_id:
                        if record.status == "unalignable":
                            raise EvidenceValidationError("canonical-clock record cannot be unalignable")
                    elif mapping is None:
                        raise EvidenceValidationError("record source clock has no mapping")
                    elif mapping.status == "unalignable" and record.status != "unalignable":
                        raise EvidenceValidationError("record on an unalignable clock must be marked unalignable")
                    elif mapping.status == "valid" and record.status == "unalignable":
                        raise EvidenceValidationError("unalignable record contradicts its valid clock mapping")
                    if record.status != "unalignable":
                        self._validate_clock_projection(record, mapping)
                self._validate_record_window(record)
                self._validate_artifact_refs(record.raw_artifact_ids, artifact_by_id)

    @staticmethod
    def _unique_by(items: Sequence[Any], attribute: str, label: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for item in items:
            identity = getattr(item, attribute)
            if identity in result:
                raise EvidenceValidationError(f"duplicate {label} identity: {identity}")
            result[identity] = item
        return result

    @staticmethod
    def _validate_artifact_refs(refs: Sequence[str], artifacts: Mapping[str, RawArtifact]) -> None:
        missing = sorted(set(refs) - set(artifacts))
        if missing:
            raise EvidenceValidationError("evidence references unknown raw artifacts: " + ", ".join(missing))

    def _validate_gap(self, gap: Gap, clocks: Mapping[str, ClockReference]) -> None:
        if gap.clock_id not in clocks:
            raise EvidenceValidationError("gap references an unknown clock")
        if gap.canonical:
            if gap.clock_id != self.timeline.clock_id:
                raise EvidenceValidationError("canonical gap must use the canonical clock")
            if gap.start_ns < self.timeline.start_ns or gap.end_ns > self.timeline.end_ns:
                raise EvidenceValidationError("canonical gap falls outside the run window")

    def _validate_record_window(self, record: EvidenceRecord) -> None:
        values: list[int] = []
        if record.canonical_timestamp_ns is not None:
            values.append(record.canonical_timestamp_ns)
        if record.canonical_start_ns is not None:
            values.extend([record.canonical_start_ns, record.canonical_end_ns])  # type: ignore[list-item]
        if record.acquisition_bounds is not None:
            values.extend(
                [
                    record.acquisition_bounds.canonical_start_ns,
                    record.acquisition_bounds.canonical_end_ns,
                ]
            )
        if values and (min(values) < self.timeline.start_ns or max(values) > self.timeline.end_ns):
            raise EvidenceValidationError("record canonical timing falls outside the run window")

    def _validate_clock_projection(self, record: EvidenceRecord, mapping: ClockMapping | None) -> None:
        """Ensure canonical timing agrees with its declared native-clock mapping."""
        assert record.alignment_error_ns is not None
        mapping_error_ns = mapping.max_error_ns if mapping is not None else 0
        assert mapping_error_ns is not None
        if record.alignment_error_ns < mapping_error_ns:
            raise EvidenceValidationError("record alignment error understates its clock mapping error")

        def project(native_ns: int) -> int:
            return mapping.to_canonical_ns(native_ns) if mapping is not None else native_ns

        if record.native_timestamp_ns is not None:
            expected = project(record.native_timestamp_ns)
            if record.canonical_timestamp_ns is not None:
                if abs(record.canonical_timestamp_ns - expected) > record.alignment_error_ns:
                    raise EvidenceValidationError("record canonical timestamp contradicts its clock mapping")
            else:
                assert record.acquisition_bounds is not None
                lower = expected - record.alignment_error_ns
                upper = expected + record.alignment_error_ns
                if (
                    upper < record.acquisition_bounds.canonical_start_ns
                    or lower > record.acquisition_bounds.canonical_end_ns
                ):
                    raise EvidenceValidationError("record acquisition bounds contradict its clock mapping")
            return

        assert record.native_start_ns is not None and record.native_end_ns is not None
        assert record.canonical_start_ns is not None and record.canonical_end_ns is not None
        if (
            abs(record.canonical_start_ns - project(record.native_start_ns)) > record.alignment_error_ns
            or abs(record.canonical_end_ns - project(record.native_end_ns)) > record.alignment_error_ns
        ):
            raise EvidenceValidationError("record canonical range contradicts its clock mapping")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "scope": asdict(self.scope),
            "timeline": self.timeline.to_dict(),
            "clocks": [asdict(clock) for clock in self.clocks],
            "clock_mappings": [asdict(mapping) for mapping in self.clock_mappings],
            "profilers": [asdict(profiler) for profiler in self.profilers],
            "streams": [asdict(stream) for stream in self.streams],
            "artifacts": [asdict(artifact) for artifact in self.artifacts],
            "provenance": dict(self.provenance),
            "extensions": dict(self.extensions),
        }

    @classmethod
    def from_dict(cls, payload: Any) -> "EvidenceBundle":
        data = _expect_object(payload, "bundle")
        allowed = {
            "schema_version",
            "run_id",
            "scope",
            "timeline",
            "clocks",
            "clock_mappings",
            "profilers",
            "streams",
            "artifacts",
            "provenance",
            "extensions",
        }
        _only_keys(
            data,
            allowed=allowed,
            required={
                "schema_version",
                "run_id",
                "scope",
                "timeline",
                "clocks",
                "clock_mappings",
                "profilers",
                "streams",
            },
            field_name="bundle",
        )
        for key in ("clocks", "clock_mappings", "profilers", "streams", "artifacts"):
            if not isinstance(data.get(key, []), list):
                raise EvidenceValidationError(f"bundle.{key} must be a list")
        return cls(
            schema_version=data["schema_version"],
            run_id=data["run_id"],
            scope=MeasuredScope.from_dict(data["scope"]),
            timeline=CanonicalTimeline.from_dict(data["timeline"]),
            clocks=tuple(ClockReference.from_dict(item) for item in data["clocks"]),
            clock_mappings=tuple(ClockMapping.from_dict(item) for item in data["clock_mappings"]),
            profilers=tuple(ProfilerManifest.from_dict(item) for item in data["profilers"]),
            streams=tuple(EvidenceStream.from_dict(item) for item in data["streams"]),
            artifacts=tuple(RawArtifact.from_dict(item) for item in data.get("artifacts", [])),
            provenance=data.get("provenance", {}),
            extensions=data.get("extensions", {}),
        )


def bundle_bytes(bundle: EvidenceBundle) -> bytes:
    """Return the canonical, newline-terminated JSON representation."""
    return (
        json.dumps(
            bundle.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def bundle_sha256(bundle: EvidenceBundle) -> str:
    return hashlib.sha256(bundle_bytes(bundle)).hexdigest()


def dumps_bundle(bundle: EvidenceBundle) -> str:
    return bundle_bytes(bundle).decode("utf-8")


def loads_bundle(payload: str | bytes) -> EvidenceBundle:
    try:
        decoded = json.loads(payload)
    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
        raise EvidenceValidationError(f"unable to decode evidence bundle JSON: {exc}") from exc
    return EvidenceBundle.from_dict(decoded)


def write_bundle(bundle: EvidenceBundle, path: Path) -> None:
    path.write_bytes(bundle_bytes(bundle))


def read_bundle(path: Path) -> EvidenceBundle:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise EvidenceValidationError(f"unable to read evidence bundle {path}: {exc}") from exc
    return loads_bundle(payload)


def validate_raw_artifacts(bundle: EvidenceBundle, base_dir: Path) -> None:
    """Fail closed if a retained raw artifact is absent, changed, or resized."""
    root = base_dir.resolve()
    for artifact in bundle.artifacts:
        path = (root / artifact.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise EvidenceValidationError(f"raw artifact escapes bundle directory: {artifact.relative_path}") from exc
        try:
            content = path.read_bytes()
        except OSError as exc:
            raise EvidenceValidationError(f"unable to read raw artifact {artifact.relative_path}: {exc}") from exc
        if len(content) != artifact.size_bytes:
            raise EvidenceValidationError(f"raw artifact size mismatch for {artifact.relative_path}")
        if hashlib.sha256(content).hexdigest() != artifact.sha256:
            raise EvidenceValidationError(f"raw artifact SHA-256 mismatch for {artifact.relative_path}")


__all__ = [
    "AcquisitionBounds",
    "CANONICAL_CLOCK_NAME",
    "Cadence",
    "CanonicalTimeline",
    "ClockMapping",
    "ClockReference",
    "EVIDENCE_BUNDLE_SCHEMA_VERSION",
    "EVIDENCE_STATUSES",
    "EvidenceBundle",
    "EvidenceRecord",
    "EvidenceStream",
    "EvidenceValidationError",
    "Gap",
    "LifecycleMarker",
    "MeasuredScope",
    "ProfilerManifest",
    "RECORD_SHAPES",
    "RawArtifact",
    "bundle_bytes",
    "bundle_sha256",
    "dumps_bundle",
    "loads_bundle",
    "read_bundle",
    "validate_raw_artifacts",
    "write_bundle",
]
