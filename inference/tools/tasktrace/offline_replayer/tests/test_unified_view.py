# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

from offline_replayer.measurement.evidence import (
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
    bundle_bytes,
    write_bundle,
)
from offline_replayer.reporting.unified_view import (
    SHAPE_REGISTRY,
    assemble_view_model,
    render_unified_animated_svg,
    render_unified_html,
    render_unified_svg,
    view_model_bytes,
    view_model_sha256,
    write_unified_view_artifacts,
    write_unified_view_from_bundle,
)

RUN_ID = "87654321-4321-6789-a321-678943216789"
SCOPE_ID = "scope:unified-fixture"
CLOCK_ID = "host-monotonic"
START_NS = 10_000_000_000
ORIGIN_NS = 10_100_000_000
END_NS = 11_000_000_000


def _record(
    record_id: str,
    stream_id: str,
    shape: str,
    value: object,
    **overrides: object,
) -> EvidenceRecord:
    fields: dict[str, object] = {
        "record_id": record_id,
        "run_id": RUN_ID,
        "scope_id": SCOPE_ID,
        "profiler_id": "fixture-profiler",
        "stream_id": stream_id,
        "shape": shape,
        "status": "valid",
        "value": value,
    }
    if shape in {"point_event", "sampled_gauge", "cumulative_counter"}:
        fields.update(
            source_clock_id=CLOCK_ID,
            native_timestamp_ns=10_300_000_000,
            canonical_timestamp_ns=10_300_000_000,
            alignment_error_ns=250,
        )
    elif shape in {"interval", "window_aggregate"}:
        fields.update(
            source_clock_id=CLOCK_ID,
            native_start_ns=10_400_000_000,
            native_end_ns=10_600_000_000,
            canonical_start_ns=10_400_000_000,
            canonical_end_ns=10_600_000_000,
            alignment_error_ns=500,
        )
    fields.update(overrides)
    return EvidenceRecord(**fields)  # type: ignore[arg-type]


def _stream(
    stream_id: str,
    shape: str,
    records: tuple[EvidenceRecord, ...],
    *,
    status: str = "valid",
    reasons: tuple[str, ...] = (),
    unit: str = "1",
    cadence: Cadence | None = None,
    gaps: tuple[Gap, ...] = (),
    raw_artifact_ids: tuple[str, ...] = (),
) -> EvidenceStream:
    return EvidenceStream(
        stream_id=stream_id,
        profiler_id="fixture-profiler",
        metric=f"fixture.{stream_id}",
        shape=shape,
        unit=unit,
        status=status,
        reasons=reasons,
        records=records,
        cadence=cadence,
        gaps=gaps,
        raw_artifact_ids=raw_artifact_ids,
        display={"group": "Fixture profiler", "label": stream_id, "order": len(stream_id)},
        provenance={"collector": "fixture", "method": shape},
    )


def _bundle(tmp_path: Path) -> EvidenceBundle:
    raw_path = tmp_path / "raw" / "events.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    raw_path.write_text('{"event":"fixture"}\n')
    artifact = RawArtifact.from_path(
        artifact_id="raw-events",
        path=raw_path,
        base_dir=tmp_path,
        media_type="application/x-ndjson",
    )

    scalar = _stream(
        "memory-peak",
        "scalar",
        (
            _record(
                "memory-peak-record",
                "memory-peak",
                "scalar",
                268_435_456,
                raw_artifact_ids=(artifact.artifact_id,),
                attributes={"exact": True, "timing": "untimed"},
            ),
        ),
        unit="byte",
    )
    event = _stream(
        "phase-events",
        "point_event",
        (
            _record(
                "phase-start",
                "phase-events",
                "point_event",
                {"phase": "parallel-cpu"},
                canonical_timestamp_ns=None,
                acquisition_bounds=AcquisitionBounds(10_299_999_000, 10_300_001_000),
                alignment_error_ns=1_000,
            ),
        ),
        unit="event",
    )
    interval = _stream(
        "process-activity",
        "interval",
        (_record("child-process", "process-activity", "interval", {"binary": "/bin/true"}),),
        unit="process",
    )
    gauge = _stream(
        "memory-current",
        "sampled_gauge",
        (
            _record(
                "memory-sample-1",
                "memory-current",
                "sampled_gauge",
                100,
                native_timestamp_ns=10_200_000_000,
                canonical_timestamp_ns=10_200_000_000,
            ),
            _record(
                "memory-sample-2",
                "memory-current",
                "sampled_gauge",
                300,
                native_timestamp_ns=10_700_000_000,
                canonical_timestamp_ns=10_700_000_000,
            ),
        ),
        unit="byte",
        cadence=Cadence(100_000_000, 200_000_000, "fixed-deadline-polling"),
        gaps=(
            Gap(
                clock_id=CLOCK_ID,
                start_ns=10_450_000_000,
                end_ns=10_550_000_000,
                reason="one memory sample deadline was missed",
                dropped_records=1,
            ),
        ),
        raw_artifact_ids=(artifact.artifact_id,),
    )
    counter = _stream(
        "cpu-usage",
        "cumulative_counter",
        (_record("cpu-counter", "cpu-usage", "cumulative_counter", 42_000),),
        unit="microsecond",
        cadence=Cadence(100_000_000, 200_000_000, "fixed-deadline-polling"),
    )
    aggregate = _stream(
        "memory-pressure",
        "window_aggregate",
        (_record("psi-window", "memory-pressure", "window_aggregate", {"stall_usec": 17}),),
        unit="microsecond",
    )

    missing = _stream(
        "missing-pmu",
        "cumulative_counter",
        (),
        status="missing",
        reasons=("portable PMU event is unavailable",),
        unit="event",
    )
    invalid_record = _record(
        "invalid-instructions",
        "invalid-pmu",
        "scalar",
        17,
        status="invalid",
        reasons=("time_running/time_enabled was below 95%",),
    )
    invalid = _stream(
        "invalid-pmu",
        "scalar",
        (invalid_record,),
        status="invalid",
        reasons=("time_running/time_enabled was below 95%",),
        unit="instruction",
    )
    dropped_record = _record(
        "lost-transition",
        "dropped-scheduler",
        "point_event",
        None,
        status="dropped",
        reasons=("perf buffer reported one lost event",),
    )
    dropped = _stream(
        "dropped-scheduler",
        "point_event",
        (dropped_record,),
        status="dropped",
        reasons=("perf buffer reported one lost event",),
        unit="event",
    )
    native_only_record = _record(
        "wall-event",
        "unalignable-exec",
        "point_event",
        {"binary": "/usr/bin/tool"},
        status="unalignable",
        reasons=("wall clock had no validated monotonic correlation",),
        source_clock_id="unknown-wall",
        native_timestamp_ns=1_700_000_000_000_000_000,
        canonical_timestamp_ns=None,
        alignment_error_ns=None,
    )
    unalignable = _stream(
        "unalignable-exec",
        "point_event",
        (native_only_record,),
        status="unalignable",
        reasons=("wall clock had no validated monotonic correlation",),
        unit="event",
    )

    streams = (
        scalar,
        event,
        interval,
        gauge,
        counter,
        aggregate,
        missing,
        invalid,
        dropped,
        unalignable,
    )
    return EvidenceBundle(
        run_id=RUN_ID,
        scope=MeasuredScope(
            scope_id=SCOPE_ID,
            kind="disposable-container-cgroup-subtree",
            identity={"runtime": "docker", "container_id": "fixture"},
        ),
        timeline=CanonicalTimeline(
            clock_id=CLOCK_ID,
            origin_ns=ORIGIN_NS,
            start_ns=START_NS,
            end_ns=END_NS,
            lifecycle_markers=(
                LifecycleMarker("measured_work_start", 10_100_000_000, 200),
                LifecycleMarker("memory_plateau_start", 10_250_000_000, 1_000),
                LifecycleMarker("measured_work_end", 10_900_000_000, 200),
            ),
        ),
        clocks=(
            ClockReference(CLOCK_ID, "CLOCK_MONOTONIC", 1, "time:[fixture]"),
            ClockReference("unknown-wall", "CLOCK_REALTIME", 1),
        ),
        clock_mappings=(
            ClockMapping(
                source_clock_id="unknown-wall",
                canonical_clock_id=CLOCK_ID,
                status="unalignable",
                method="no-correlation-observed",
                reasons=("wall clock had no validated monotonic correlation",),
            ),
        ),
        profilers=(
            ProfilerManifest(
                profiler_id="fixture-profiler",
                profiler_version="1",
                status="valid",
                stream_ids=tuple(stream.stream_id for stream in streams),
                provenance={"implementation": "test-only generic profiler"},
            ),
        ),
        streams=streams,
        artifacts=(artifact,),
        provenance={"fixture": "labeled phases"},
    )


def test_generic_fixture_profiler_exposes_all_six_shapes_without_name_dispatch(
    tmp_path: Path,
) -> None:
    model = assemble_view_model(_bundle(tmp_path))

    assert set(SHAPE_REGISTRY) == {
        "scalar",
        "point_event",
        "interval",
        "sampled_gauge",
        "cumulative_counter",
        "window_aggregate",
    }
    all_streams = model["summaries"] + model["tracks"]
    assert {stream["shape"] for stream in all_streams} == set(SHAPE_REGISTRY)
    assert {stream["render_kind"] for stream in all_streams} >= {
        "summary_fact",
        "point_ticks",
        "interval_bands",
        "sample_series",
        "counter_series",
        "window_bands",
    }
    assert all(stream["profiler_id"] == "fixture-profiler" for stream in all_streams)


def test_shared_axis_markers_uncertainty_and_untimed_scalar_are_honest(
    tmp_path: Path,
) -> None:
    model = assemble_view_model(_bundle(tmp_path))

    assert model["axis"]["clock_name"] == "CLOCK_MONOTONIC"
    assert model["axis"]["minimum_offset_ns"] == -100_000_000
    assert model["axis"]["maximum_offset_ns"] == 900_000_000
    assert [marker["name"] for marker in model["lifecycle_markers"]] == [
        "measured_work_start",
        "memory_plateau_start",
        "measured_work_end",
    ]
    peak = next(summary for summary in model["summaries"] if summary["stream_id"] == "memory-peak")
    peak_record = peak["records"][0]
    assert not any("timestamp" in key or "offset" in key for key in peak_record)
    bounded_event = next(track for track in model["tracks"] if track["stream_id"] == "phase-events")
    assert bounded_event["records"][0]["acquisition_start_offset_ns"] == 199_999_000
    assert bounded_event["maximum_alignment_error_ns"] == 1_000


def test_statuses_gaps_and_unalignable_native_evidence_never_disappear_or_become_zero(
    tmp_path: Path,
) -> None:
    model = assemble_view_model(_bundle(tmp_path))
    html_output = render_unified_html(model)
    svg_output = render_unified_svg(model)

    for stream_id in (
        "missing-pmu",
        "invalid-pmu",
        "dropped-scheduler",
        "unalignable-exec",
    ):
        assert stream_id in html_output
        assert stream_id in svg_output
    for reason in (
        "portable PMU event is unavailable",
        "time_running/time_enabled was below 95%",
        "perf buffer reported one lost event",
        "wall clock had no validated monotonic correlation",
        "one memory sample deadline was missed",
    ):
        assert reason in html_output
        assert reason in svg_output
    unaligned = next(track for track in model["tracks"] if track["stream_id"] == "unalignable-exec")
    record = unaligned["records"][0]
    assert record["native_timestamp_ns"] == 1_700_000_000_000_000_000
    assert not any(key.startswith("canonical_") or key.startswith("acquisition_") for key in record)
    assert 'class="evidence-gap"' in html_output
    assert "INVALID" in svg_output and ">17 instruction<" in svg_output


def test_streamless_invalid_requested_profiler_is_visible_in_every_export(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    unavailable = ProfilerManifest(
        profiler_id="scheduler-concurrency",
        profiler_version="1",
        status="invalid",
        reasons=("BCC unavailable: root privilege is required",),
        stream_ids=(),
        configuration={"backend": "bcc"},
    )
    bundle = replace(
        bundle,
        profilers=(*bundle.profilers, unavailable),
        extensions={
            **bundle.extensions,
            # A stale or narrower coordinator summary must never override a
            # non-valid manifest present in the final bundle.
            "evidence_status": "valid",
            "all_selected_profilers_valid": False,
        },
    )

    model = assemble_view_model(bundle)
    profiler = next(item for item in model["profilers"] if item["profiler_id"] == "scheduler-concurrency")
    assert profiler["status"] == "invalid"
    assert profiler["stream_ids"] == []
    assert model["evidence_status"] == {
        "status": "invalid",
        "reasons": ["BCC unavailable: root privilege is required"],
        "all_selected_profilers_valid": False,
    }

    for output in (
        render_unified_html(model),
        render_unified_svg(model),
        render_unified_animated_svg(model),
    ):
        assert "scheduler-concurrency" in output
        assert "BCC unavailable: root privilege is required" in output
        assert "INVALID" in output
        assert "0 stream(s)" in output


def test_html_is_self_contained_and_has_shared_cursor_zoom_and_collapsible_tracks(
    tmp_path: Path,
) -> None:
    output = render_unified_html(assemble_view_model(_bundle(tmp_path)))

    assert "<style>" in output and "<script>" in output
    assert 'id="zoom"' in output
    assert 'id="reset"' in output
    assert 'class="shared-cursor"' in output
    assert '<details class="track' in output
    assert "pointermove" in output
    assert "viewBox" in output
    assert "https://" not in output


def test_all_exports_are_deterministic_and_embed_the_identical_model_sha(
    tmp_path: Path,
) -> None:
    model = assemble_view_model(_bundle(tmp_path))
    digest = view_model_sha256(model)

    first = (
        render_unified_html(model),
        render_unified_svg(model),
        render_unified_animated_svg(model),
    )
    second = (
        render_unified_html(model),
        render_unified_svg(model),
        render_unified_animated_svg(model),
    )

    assert first == second
    assert hashlib.sha256(view_model_bytes(model)).hexdigest() == digest
    assert all(digest in artifact for artifact in first)
    assert '<animate attributeName="x1"' not in first[1]
    assert '<animate attributeName="x1"' in first[2]
    assert first[2].count(digest) >= 2


def test_raw_artifact_links_and_provenance_are_preserved_in_model_and_exports(
    tmp_path: Path,
) -> None:
    model = assemble_view_model(_bundle(tmp_path))
    artifact = model["artifacts"][0]
    output = render_unified_html(model)

    assert artifact["href"] == "raw/events.jsonl"
    assert artifact["sha256"] in output
    assert 'href="raw/events.jsonl"' in output
    memory = next(track for track in model["tracks"] if track["stream_id"] == "memory-current")
    assert memory["provenance"] == {
        "collector": "fixture",
        "method": "sampled_gauge",
    }
    assert memory["cadence"]["requested_interval_ns"] == 100_000_000
    assert memory["gaps"][0]["dropped_records"] == 1


def test_rendering_from_retained_bundle_does_not_execute_or_mutate_inputs(
    tmp_path: Path,
) -> None:
    bundle = _bundle(tmp_path)
    bundle_before = bundle_bytes(bundle)
    sentinel = tmp_path / "execution-count.txt"
    sentinel.write_text("1\n")
    bundle_path = tmp_path / "evidence_bundle.json"
    write_bundle(bundle, bundle_path)

    first_outputs = write_unified_view_artifacts(bundle, tmp_path / "render-one")
    second_outputs = write_unified_view_from_bundle(bundle_path, tmp_path / "render-two")

    assert sentinel.read_text() == "1\n"
    assert bundle_bytes(bundle) == bundle_before
    assert bundle_path.read_bytes() == bundle_before
    assert first_outputs["view_model_sha256"] == second_outputs["view_model_sha256"]
    for name in ("view_model", "html", "svg", "animated_svg"):
        assert Path(first_outputs[name]).read_bytes() == Path(second_outputs[name]).read_bytes()


def test_embedded_html_model_is_the_same_canonical_model(tmp_path: Path) -> None:
    model = assemble_view_model(_bundle(tmp_path))
    output = render_unified_html(model)
    match = re.search(
        r'<script id="view-model" type="application/json">(.*?)</script>',
        output,
        re.DOTALL,
    )
    assert match is not None
    embedded = json.loads(match.group(1))
    assert embedded == model
