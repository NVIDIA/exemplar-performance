# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import uuid
from dataclasses import replace
from pathlib import Path

import pytest
from offline_replayer.measurement.evidence import (
    RECORD_SHAPES,
    AcquisitionBounds,
    Cadence,
    CanonicalTimeline,
    ClockMapping,
    ClockReference,
    EvidenceBundle,
    EvidenceRecord,
    EvidenceStream,
    EvidenceValidationError,
    Gap,
    LifecycleMarker,
    MeasuredScope,
    ProfilerManifest,
    RawArtifact,
    bundle_bytes,
    bundle_sha256,
    dumps_bundle,
    loads_bundle,
    read_bundle,
    validate_raw_artifacts,
    write_bundle,
)

RUN_ID = "12345678-1234-5678-9234-567812345678"
SCOPE_ID = "docker-cgroup:fixture"
CLOCK_ID = "host-monotonic"
START_NS = 1_000_000_000
END_NS = 2_000_000_000


def _record(shape: str, record_id: str, **overrides: object) -> EvidenceRecord:
    values: dict[str, object] = {
        "record_id": record_id,
        "run_id": RUN_ID,
        "scope_id": SCOPE_ID,
        "profiler_id": "fixture",
        "stream_id": f"stream-{shape}",
        "shape": shape,
        "status": "valid",
        "value": 1,
    }
    if shape == "point_event":
        values.update(
            source_clock_id=CLOCK_ID,
            native_timestamp_ns=1_200_000_000,
            canonical_timestamp_ns=1_200_000_000,
            alignment_error_ns=0,
        )
    elif shape in {"sampled_gauge", "cumulative_counter"}:
        values.update(
            source_clock_id=CLOCK_ID,
            native_timestamp_ns=1_300_000_000,
            canonical_timestamp_ns=1_300_000_000,
            alignment_error_ns=0,
        )
    elif shape in {"interval", "window_aggregate"}:
        values.update(
            source_clock_id=CLOCK_ID,
            native_start_ns=1_400_000_000,
            native_end_ns=1_500_000_000,
            canonical_start_ns=1_400_000_000,
            canonical_end_ns=1_500_000_000,
            alignment_error_ns=0,
        )
    values.update(overrides)
    return EvidenceRecord(**values)  # type: ignore[arg-type]


def _all_shape_streams(*, artifact_id: str | None = None) -> tuple[EvidenceStream, ...]:
    streams: list[EvidenceStream] = []
    for shape in sorted(RECORD_SHAPES):
        record = _record(shape, f"record-{shape}")
        cadence = (
            Cadence(
                requested_interval_ns=100_000_000,
                maximum_acceptable_gap_ns=200_000_000,
                method="fixed-deadline-polling",
            )
            if shape in {"sampled_gauge", "cumulative_counter"}
            else None
        )
        gaps = (
            (
                Gap(
                    clock_id=CLOCK_ID,
                    start_ns=1_600_000_000,
                    end_ns=1_700_000_000,
                    reason="collector missed a deadline",
                    dropped_records=1,
                ),
            )
            if shape == "sampled_gauge"
            else ()
        )
        streams.append(
            EvidenceStream(
                stream_id=f"stream-{shape}",
                profiler_id="fixture",
                metric=f"fixture.{shape}",
                shape=shape,
                unit="1",
                status="valid",
                records=(record,),
                cadence=cadence,
                gaps=gaps,
                raw_artifact_ids=((artifact_id,) if artifact_id and shape == "scalar" else ()),
                display={"group": "Fixture", "label": shape.replace("_", " ")},
                extensions={"example.org/future-field": {"retained": True}},
            )
        )
    return tuple(streams)


def _bundle(*, artifact: RawArtifact | None = None) -> EvidenceBundle:
    streams = _all_shape_streams(artifact_id=artifact.artifact_id if artifact else None)
    return EvidenceBundle(
        run_id=RUN_ID,
        scope=MeasuredScope(
            scope_id=SCOPE_ID,
            kind="disposable-container-cgroup-subtree",
            identity={
                "runtime": "docker",
                "container_id": "fixture-container",
                "cgroup_inode": 9876,
            },
            description="fixture cgroup and all descendants",
        ),
        timeline=CanonicalTimeline(
            clock_id=CLOCK_ID,
            origin_ns=1_100_000_000,
            start_ns=START_NS,
            end_ns=END_NS,
            lifecycle_markers=(
                LifecycleMarker("measured_work_end", 1_900_000_000),
                LifecycleMarker("measured_work_start", 1_100_000_000),
            ),
        ),
        clocks=(
            ClockReference(
                clock_id="unmapped-wall-clock",
                clock_name="CLOCK_REALTIME",
                resolution_ns=1,
            ),
            ClockReference(
                clock_id=CLOCK_ID,
                clock_name="CLOCK_MONOTONIC",
                resolution_ns=1,
                namespace_identity="time:[4026531834]",
            ),
        ),
        clock_mappings=(
            ClockMapping(
                source_clock_id="unmapped-wall-clock",
                canonical_clock_id=CLOCK_ID,
                status="unalignable",
                method="no-correlation-observed",
                reasons=("wall clock may have stepped during the run",),
            ),
        ),
        profilers=(
            ProfilerManifest(
                profiler_id="fixture",
                profiler_version="1.0",
                status="valid",
                stream_ids=tuple(stream.stream_id for stream in reversed(streams)),
                provenance={"implementation": "test fixture"},
            ),
        ),
        streams=tuple(reversed(streams)),
        artifacts=((artifact,) if artifact else ()),
        provenance={"trace_sha256": "a" * 64},
    )


def test_bundle_round_trip_preserves_every_standard_shape_and_stable_order() -> None:
    bundle = _bundle()

    decoded = loads_bundle(dumps_bundle(bundle))

    assert {stream.shape for stream in decoded.streams} == RECORD_SHAPES
    assert [stream.stream_id for stream in decoded.streams] == sorted(stream.stream_id for stream in decoded.streams)
    assert [marker.name for marker in decoded.timeline.lifecycle_markers] == [
        "measured_work_start",
        "measured_work_end",
    ]
    assert decoded.to_dict() == bundle.to_dict()
    assert bundle_bytes(decoded) == bundle_bytes(bundle)


def test_sample_can_use_acquisition_bounds_without_inventing_a_point_timestamp() -> None:
    record = _record(
        "sampled_gauge",
        "bounded-read",
        canonical_timestamp_ns=None,
        acquisition_bounds=AcquisitionBounds(1_299_999_000, 1_300_001_000),
        alignment_error_ns=1_000,
    )

    assert record.canonical_timestamp_ns is None
    assert record.acquisition_bounds == AcquisitionBounds(1_299_999_000, 1_300_001_000)


def test_scalar_rejects_all_timing_fields_so_memory_peak_cannot_get_fake_time() -> None:
    with pytest.raises(EvidenceValidationError, match="scalar record must remain untimed"):
        _record(
            "scalar",
            "fake-timed-peak",
            source_clock_id=CLOCK_ID,
            native_timestamp_ns=1_200_000_000,
            canonical_timestamp_ns=1_200_000_000,
            alignment_error_ns=0,
        )


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"canonical_timestamp_ns": None}, "exactly one canonical timestamp"),
        (
            {"acquisition_bounds": AcquisitionBounds(1_200_000_000, 1_200_000_100)},
            "exactly one canonical timestamp",
        ),
        ({"alignment_error_ns": None}, "requires alignment_error_ns"),
    ],
)
def test_aligned_point_timing_is_explicit_and_unambiguous(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(EvidenceValidationError, match=message):
        _record("point_event", "bad-point", **overrides)


def test_unalignable_evidence_retains_native_time_without_canonical_claim() -> None:
    unalignable = EvidenceRecord(
        record_id="native-only-event",
        run_id=RUN_ID,
        scope_id=SCOPE_ID,
        profiler_id="fixture",
        stream_id="native-only",
        shape="point_event",
        status="unalignable",
        reasons=("source clock has no validated correlation",),
        value={"event": "execve"},
        source_clock_id="unmapped-wall-clock",
        native_timestamp_ns=1_700_000_000_000_000_000,
    )
    stream = EvidenceStream(
        stream_id="native-only",
        profiler_id="fixture",
        metric="process.exec",
        shape="point_event",
        unit="event",
        status="unalignable",
        reasons=("source clock has no validated correlation",),
        records=(unalignable,),
    )
    bundle = replace(
        _bundle(),
        profilers=(
            ProfilerManifest(
                profiler_id="fixture",
                profiler_version="1.0",
                status="valid",
                stream_ids=(stream.stream_id,),
            ),
        ),
        streams=(stream,),
    )

    decoded = loads_bundle(dumps_bundle(bundle)).streams[0].records[0]
    assert decoded.native_timestamp_ns == 1_700_000_000_000_000_000
    assert decoded.canonical_timestamp_ns is None
    assert decoded.acquisition_bounds is None
    assert decoded.alignment_error_ns is None


def test_valid_clock_mapping_bounds_canonical_projection_and_uncertainty() -> None:
    base = _bundle()
    mapped_clock = ClockReference("container-monotonic", "CLOCK_MONOTONIC", 1)
    mapping = ClockMapping(
        source_clock_id=mapped_clock.clock_id,
        canonical_clock_id=CLOCK_ID,
        status="valid",
        method="bracketed-clock-correlation",
        scale_numerator=1,
        scale_denominator=1,
        offset_ns=1_000_000_000,
        max_error_ns=50,
    )
    record = _record(
        "point_event",
        "mapped",
        source_clock_id=mapped_clock.clock_id,
        native_timestamp_ns=200_000_000,
        canonical_timestamp_ns=1_200_000_040,
        alignment_error_ns=50,
    )
    stream = EvidenceStream(
        stream_id="stream-point_event",
        profiler_id="fixture",
        metric="fixture.mapped-event",
        shape="point_event",
        unit="event",
        status="valid",
        records=(record,),
    )
    bundle = replace(
        base,
        clocks=base.clocks + (mapped_clock,),
        clock_mappings=base.clock_mappings + (mapping,),
        streams=(stream,),
        profilers=(replace(base.profilers[0], stream_ids=(stream.stream_id,)),),
    )
    assert bundle.streams[0].records[0].canonical_timestamp_ns == 1_200_000_040

    understated = replace(record, alignment_error_ns=49)
    with pytest.raises(EvidenceValidationError, match="understates"):
        replace(bundle, streams=(replace(stream, records=(understated,)),))

    contradictory = replace(record, canonical_timestamp_ns=1_200_000_051)
    with pytest.raises(EvidenceValidationError, match="contradicts"):
        replace(bundle, streams=(replace(stream, records=(contradictory,)),))


def test_missing_invalid_and_dropped_states_have_reasons_and_no_fake_zero() -> None:
    missing = EvidenceStream(
        stream_id="missing",
        profiler_id="states",
        metric="pmu.cache_misses",
        shape="cumulative_counter",
        unit="event",
        status="missing",
        reasons=("PMU event is unsupported",),
    )
    invalid_record = EvidenceRecord(
        **{
            **_record("scalar", "invalid-value").__dict__,
            "profiler_id": "states",
            "stream_id": "invalid",
            "status": "invalid",
            "value": 17,
            "reasons": ("counter multiplexing ratio was too low",),
        }
    )
    invalid = EvidenceStream(
        stream_id="invalid",
        profiler_id="states",
        metric="pmu.instructions",
        shape="scalar",
        unit="instruction",
        status="invalid",
        reasons=("counter multiplexing ratio was too low",),
        records=(invalid_record,),
    )
    dropped_record = EvidenceRecord(
        **{
            **_record("point_event", "drop-location").__dict__,
            "profiler_id": "states",
            "stream_id": "dropped",
            "status": "dropped",
            "value": None,
            "reasons": ("perf buffer reported lost events",),
        }
    )
    dropped = EvidenceStream(
        stream_id="dropped",
        profiler_id="states",
        metric="scheduler.transition",
        shape="point_event",
        unit="event",
        status="dropped",
        reasons=("perf buffer reported lost events",),
        records=(dropped_record,),
    )
    bundle = replace(
        _bundle(),
        profilers=(
            ProfilerManifest(
                profiler_id="states",
                profiler_version="1",
                status="invalid",
                reasons=("one or more requested outputs are unusable",),
                stream_ids=("missing", "invalid", "dropped"),
            ),
        ),
        streams=(missing, invalid, dropped),
    )

    by_id = {stream.stream_id: stream for stream in bundle.streams}
    assert by_id["missing"].records == ()
    assert by_id["invalid"].records[0].value == 17
    assert by_id["dropped"].records[0].value is None


@pytest.mark.parametrize("status", ["invalid", "missing", "dropped", "unalignable"])
def test_nonvalid_status_requires_a_reason(status: str) -> None:
    with pytest.raises(EvidenceValidationError, match="requires a reason"):
        ProfilerManifest("profiler", "1", status)


def test_rejects_record_from_another_run_or_scope() -> None:
    base = _bundle()
    scalar_stream = next(stream for stream in base.streams if stream.shape == "scalar")
    scalar = scalar_stream.records[0]

    wrong_run = replace(scalar, run_id=str(uuid.uuid4()))
    with pytest.raises(EvidenceValidationError, match="different run"):
        replace(
            base,
            streams=(replace(scalar_stream, records=(wrong_run,)),),
            profilers=(replace(base.profilers[0], stream_ids=(scalar_stream.stream_id,)),),
        )

    wrong_scope = replace(scalar, scope_id="another-scope")
    with pytest.raises(EvidenceValidationError, match="different measured scope"):
        replace(
            base,
            streams=(replace(scalar_stream, records=(wrong_scope,)),),
            profilers=(replace(base.profilers[0], stream_ids=(scalar_stream.stream_id,)),),
        )


def test_rejects_canonical_time_outside_run_window() -> None:
    base = _bundle()
    point_stream = next(stream for stream in base.streams if stream.shape == "point_event")
    outside = replace(
        point_stream.records[0],
        native_timestamp_ns=END_NS + 1,
        canonical_timestamp_ns=END_NS + 1,
    )

    with pytest.raises(EvidenceValidationError, match="outside the run window"):
        replace(
            base,
            streams=(replace(point_stream, records=(outside,)),),
            profilers=(replace(base.profilers[0], stream_ids=(point_stream.stream_id,)),),
        )


def test_serialization_is_byte_deterministic_and_hashes_canonical_bytes(tmp_path: Path) -> None:
    first = _bundle()
    second = _bundle()
    first_path = tmp_path / "first.json"
    second_path = tmp_path / "second.json"

    write_bundle(first, first_path)
    write_bundle(second, second_path)

    assert first_path.read_bytes() == second_path.read_bytes() == bundle_bytes(first)
    assert read_bundle(first_path).to_dict() == first.to_dict()
    assert bundle_sha256(first) == hashlib.sha256(first_path.read_bytes()).hexdigest()
    assert first_path.read_bytes().endswith(b"\n")


def test_raw_artifact_digest_is_verified_and_tampering_fails(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "scheduler-events.jsonl"
    raw.parent.mkdir()
    raw.write_text('{"cpu":0,"entering":true}\n')
    artifact = RawArtifact.from_path(
        artifact_id="scheduler-events",
        path=raw,
        base_dir=tmp_path,
        media_type="application/x-ndjson",
    )
    bundle = _bundle(artifact=artifact)

    validate_raw_artifacts(bundle, tmp_path)
    raw.write_text("tampered but not trusted\n")

    with pytest.raises(EvidenceValidationError, match="size mismatch|SHA-256 mismatch"):
        validate_raw_artifacts(bundle, tmp_path)


def test_unknown_raw_artifact_reference_is_rejected() -> None:
    base = _bundle()
    scalar_stream = next(stream for stream in base.streams if stream.shape == "scalar")

    with pytest.raises(EvidenceValidationError, match="unknown raw artifacts"):
        replace(
            base,
            streams=(replace(scalar_stream, raw_artifact_ids=("absent",)),),
            profilers=(replace(base.profilers[0], stream_ids=(scalar_stream.stream_id,)),),
        )


def test_non_json_and_nonfinite_extension_values_fail_closed() -> None:
    with pytest.raises(EvidenceValidationError, match="non-finite"):
        replace(_bundle(), extensions={"future": float("nan")})
    with pytest.raises(EvidenceValidationError, match="JSON-compatible"):
        replace(_bundle(), provenance={"not_json": Path("raw.json")})
