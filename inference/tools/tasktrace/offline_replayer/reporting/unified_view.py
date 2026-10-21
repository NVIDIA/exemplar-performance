# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generic unified profile view assembled from a typed evidence bundle.

This module is intentionally downstream-only: it accepts recorded evidence and
never imports or invokes a replay/runtime implementation.  Rendering dispatches
only on the six standard evidence shapes, so adding a profiler does not require
changes here.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import math
import re
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import quote

from ..measurement.evidence import (
    EvidenceBundle,
    EvidenceRecord,
    EvidenceStream,
    RawArtifact,
    read_bundle,
)

VIEW_MODEL_SCHEMA_VERSION = "tasktrace-unified-profile-view-model.v1"

# This is the entire renderer registry.  It is keyed by standard evidence shape,
# never by profiler or metric name.
SHAPE_REGISTRY: dict[str, dict[str, str]] = {
    "scalar": {"placement": "summary", "render_kind": "summary_fact"},
    "point_event": {"placement": "timeline", "render_kind": "point_ticks"},
    "interval": {"placement": "timeline", "render_kind": "interval_bands"},
    "sampled_gauge": {"placement": "timeline", "render_kind": "sample_series"},
    "cumulative_counter": {
        "placement": "timeline",
        "render_kind": "counter_series",
    },
    "window_aggregate": {
        "placement": "timeline",
        "render_kind": "window_bands",
    },
}

STATUS_COLORS = {
    "valid": "#15803d",
    "partial": "#a16207",
    "invalid": "#b91c1c",
    "missing": "#6b7280",
    "dropped": "#c2410c",
    "unalignable": "#7e22ce",
}
TRACK_COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#7c3aed",
    "#d97706",
    "#0891b2",
    "#be185d",
    "#4f46e5",
)


class UnifiedViewError(ValueError):
    """Raised when a view cannot be assembled or rendered honestly."""


def _json_bytes(value: Mapping[str, Any]) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def view_model_bytes(view_model: Mapping[str, Any]) -> bytes:
    """Return the canonical serialization hashed and embedded in every export."""
    if view_model.get("schema_version") != VIEW_MODEL_SCHEMA_VERSION:
        raise UnifiedViewError("unsupported or missing unified view-model schema")
    return _json_bytes(view_model)


def view_model_sha256(view_model: Mapping[str, Any]) -> str:
    return hashlib.sha256(view_model_bytes(view_model)).hexdigest()


def _offset(timestamp_ns: int, origin_ns: int) -> int:
    return timestamp_ns - origin_ns


def _artifact_view(artifact: RawArtifact) -> dict[str, Any]:
    return {
        "artifact_id": artifact.artifact_id,
        "relative_path": artifact.relative_path,
        "href": quote(artifact.relative_path, safe="/"),
        "sha256": artifact.sha256,
        "size_bytes": artifact.size_bytes,
        "media_type": artifact.media_type,
        "provenance": dict(artifact.provenance),
    }


def _record_view(
    record: EvidenceRecord,
    *,
    origin_ns: int,
    artifact_by_id: Mapping[str, RawArtifact],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "record_id": record.record_id,
        "status": record.status,
        "reasons": list(record.reasons),
        "value": record.value,
        "provenance": dict(record.provenance),
        "attributes": dict(record.attributes),
        "raw_artifacts": [_artifact_view(artifact_by_id[artifact_id]) for artifact_id in record.raw_artifact_ids],
    }
    for key, value in (
        ("source_clock_id", record.source_clock_id),
        ("native_timestamp_ns", record.native_timestamp_ns),
        ("native_start_ns", record.native_start_ns),
        ("native_end_ns", record.native_end_ns),
        ("alignment_error_ns", record.alignment_error_ns),
    ):
        if value is not None:
            result[key] = value
    if record.canonical_timestamp_ns is not None:
        result["canonical_timestamp_offset_ns"] = _offset(record.canonical_timestamp_ns, origin_ns)
    if record.canonical_start_ns is not None:
        assert record.canonical_end_ns is not None
        result["canonical_start_offset_ns"] = _offset(record.canonical_start_ns, origin_ns)
        result["canonical_end_offset_ns"] = _offset(record.canonical_end_ns, origin_ns)
    if record.acquisition_bounds is not None:
        result["acquisition_start_offset_ns"] = _offset(record.acquisition_bounds.canonical_start_ns, origin_ns)
        result["acquisition_end_offset_ns"] = _offset(record.acquisition_bounds.canonical_end_ns, origin_ns)
    return result


def _stream_artifacts(stream: EvidenceStream, artifact_by_id: Mapping[str, RawArtifact]) -> list[dict[str, Any]]:
    artifact_ids = set(stream.raw_artifact_ids)
    for record in stream.records:
        artifact_ids.update(record.raw_artifact_ids)
    return [_artifact_view(artifact_by_id[artifact_id]) for artifact_id in sorted(artifact_ids)]


def _display_text(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return format(value, ".9g")
    if isinstance(value, (str, int)):
        return str(value)
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _numeric_domain(records: Sequence[dict[str, Any]]) -> dict[str, float] | None:
    values = [
        float(record["value"])
        for record in records
        if isinstance(record.get("value"), (int, float))
        and not isinstance(record.get("value"), bool)
        and math.isfinite(float(record["value"]))
    ]
    if not values:
        return None
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0)
        minimum -= padding
        maximum += padding
    return {"minimum": minimum, "maximum": maximum}


def _stream_view(
    stream: EvidenceStream,
    *,
    origin_ns: int,
    artifact_by_id: Mapping[str, RawArtifact],
    profiler: Mapping[str, Any],
) -> dict[str, Any]:
    shape = SHAPE_REGISTRY[stream.shape]
    records = [_record_view(record, origin_ns=origin_ns, artifact_by_id=artifact_by_id) for record in stream.records]
    status_counts = {
        status: sum(record["status"] == status for record in records)
        for status in ("valid", "invalid", "missing", "dropped", "unalignable")
    }
    status_counts = {key: value for key, value in status_counts.items() if value}
    display = dict(stream.display)
    label = display.get("label")
    if not isinstance(label, str) or not label:
        label = stream.metric
    group = display.get("group")
    if not isinstance(group, str) or not group:
        group = profiler["profiler_id"]
    order = display.get("order", 0)
    try:
        order_valid = isinstance(order, (int, float)) and not isinstance(order, bool) and math.isfinite(order)
    except OverflowError:
        order_valid = False
    order = float(order) if order_valid else 0.0
    gaps: list[dict[str, Any]] = []
    for gap in stream.gaps:
        gap_view = {
            "clock_id": gap.clock_id,
            "start_ns": gap.start_ns,
            "end_ns": gap.end_ns,
            "reason": gap.reason,
            "canonical": gap.canonical,
            "dropped_records": gap.dropped_records,
        }
        if gap.canonical:
            gap_view["start_offset_ns"] = _offset(gap.start_ns, origin_ns)
            gap_view["end_offset_ns"] = _offset(gap.end_ns, origin_ns)
        gaps.append(gap_view)
    maximum_alignment_error = max(
        (int(record["alignment_error_ns"]) for record in records if record.get("alignment_error_ns") is not None),
        default=None,
    )
    return {
        "stream_id": stream.stream_id,
        "profiler_id": stream.profiler_id,
        "profiler": dict(profiler),
        "metric": stream.metric,
        "label": label,
        "group": group,
        "order": order,
        "shape": stream.shape,
        "placement": shape["placement"],
        "render_kind": shape["render_kind"],
        "unit": stream.unit,
        "status": stream.status,
        "reasons": list(stream.reasons),
        "records": records,
        "record_status_counts": status_counts,
        "cadence": asdict(stream.cadence) if stream.cadence is not None else None,
        "gaps": gaps,
        "maximum_alignment_error_ns": maximum_alignment_error,
        "raw_artifacts": _stream_artifacts(stream, artifact_by_id),
        "provenance": dict(stream.provenance),
        "display": display,
        "extensions": dict(stream.extensions),
        "numeric_value_domain": _numeric_domain(records),
    }


def assemble_view_model(bundle: EvidenceBundle) -> dict[str, Any]:
    """Purely normalize every stream into a deterministic generic view model."""
    artifact_by_id = {artifact.artifact_id: artifact for artifact in bundle.artifacts}
    profiler_by_id = {
        profiler.profiler_id: {
            "profiler_id": profiler.profiler_id,
            "profiler_version": profiler.profiler_version,
            "status": profiler.status,
            "reasons": list(profiler.reasons),
            "stream_ids": list(profiler.stream_ids),
            "intrusive": profiler.intrusive,
            "provenance": dict(profiler.provenance),
            "configuration": dict(profiler.configuration),
            "extensions": dict(profiler.extensions),
        }
        for profiler in bundle.profilers
    }
    stream_views = [
        _stream_view(
            stream,
            origin_ns=bundle.timeline.origin_ns,
            artifact_by_id=artifact_by_id,
            profiler=profiler_by_id[stream.profiler_id],
        )
        for stream in bundle.streams
    ]
    stream_views.sort(
        key=lambda stream: (
            str(stream["group"]),
            float(stream["order"]),
            str(stream["label"]),
            str(stream["stream_id"]),
        )
    )
    summaries = [stream for stream in stream_views if stream["placement"] == "summary"]
    tracks = [stream for stream in stream_views if stream["placement"] == "timeline"]
    markers = [
        {
            "name": marker.name,
            "canonical_timestamp_ns": marker.canonical_timestamp_ns,
            "offset_ns": _offset(marker.canonical_timestamp_ns, bundle.timeline.origin_ns),
            "alignment_error_ns": marker.alignment_error_ns,
            "provenance": dict(marker.provenance),
        }
        for marker in bundle.timeline.lifecycle_markers
    ]
    declared_evidence_status = bundle.extensions.get("evidence_status")
    profiler_statuses = {profiler.status for profiler in bundle.profilers}
    if declared_evidence_status == "invalid" or "invalid" in profiler_statuses or "dropped" in profiler_statuses:
        declared_evidence_status = "invalid"
    elif declared_evidence_status == "partial" or profiler_statuses & {"missing", "unalignable"}:
        declared_evidence_status = "partial"
    else:
        declared_evidence_status = "valid"
    evidence_reasons = sorted(
        {reason for profiler in bundle.profilers if profiler.status != "valid" for reason in profiler.reasons}
        | {
            str(reason)
            for reason in bundle.extensions.get("evidence_invalid_reasons", [])
            if isinstance(reason, str) and reason
        }
    )
    return {
        "schema_version": VIEW_MODEL_SCHEMA_VERSION,
        "run_id": bundle.run_id,
        "scope": asdict(bundle.scope),
        "axis": {
            "clock_id": bundle.timeline.clock_id,
            "clock_name": bundle.timeline.clock_name,
            "origin_ns": bundle.timeline.origin_ns,
            "start_ns": bundle.timeline.start_ns,
            "end_ns": bundle.timeline.end_ns,
            "minimum_offset_ns": _offset(bundle.timeline.start_ns, bundle.timeline.origin_ns),
            "maximum_offset_ns": _offset(bundle.timeline.end_ns, bundle.timeline.origin_ns),
            "duration_ns": bundle.timeline.end_ns - bundle.timeline.start_ns,
            "meaning": "run-relative nanoseconds from the declared canonical origin",
        },
        "lifecycle_markers": markers,
        "evidence_status": {
            "status": declared_evidence_status,
            "reasons": evidence_reasons,
            "all_selected_profilers_valid": bundle.extensions.get(
                "all_selected_profilers_valid",
                bundle.extensions.get("all_selected_profilers_usable"),
            ),
        },
        "shape_registry": {key: dict(SHAPE_REGISTRY[key]) for key in sorted(SHAPE_REGISTRY)},
        "profilers": [profiler_by_id[key] for key in sorted(profiler_by_id)],
        "summaries": summaries,
        "tracks": tracks,
        "artifacts": [_artifact_view(artifact) for artifact in bundle.artifacts],
        "clock_mappings": [{**asdict(mapping), "reasons": list(mapping.reasons)} for mapping in bundle.clock_mappings],
        "provenance": dict(bundle.provenance),
        "extensions": dict(bundle.extensions),
    }


def _safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value).strip("-")
    return cleaned or "track"


def _status_badge(status: str) -> str:
    color = STATUS_COLORS.get(status, "#374151")
    return (
        f'<span class="status status-{html.escape(status)}" '
        f'style="--status-color:{color}">{html.escape(status.upper())}</span>'
    )


def _artifact_links_html(artifacts: Sequence[Mapping[str, Any]]) -> str:
    if not artifacts:
        return '<span class="muted">raw evidence: none declared</span>'
    links = []
    for artifact in artifacts:
        links.append(
            f'<a href="{html.escape(str(artifact["href"]), quote=True)}" '
            f'data-sha256="{html.escape(str(artifact["sha256"]), quote=True)}">'
            f'{html.escape(str(artifact["artifact_id"]))}</a>'
        )
    return "raw evidence: " + ", ".join(links)


def _reasons_html(reasons: Sequence[Any]) -> str:
    if not reasons:
        return ""
    return '<ul class="reasons">' + "".join(f"<li>{html.escape(str(reason))}</li>" for reason in reasons) + "</ul>"


def _stream_metadata_html(stream: Mapping[str, Any]) -> str:
    cadence = stream.get("cadence")
    cadence_text = "unsampled / event-driven"
    if isinstance(cadence, dict):
        cadence_text = (
            f'requested {int(cadence["requested_interval_ns"]):,} ns; maximum gap '
            f'{int(cadence["maximum_acceptable_gap_ns"]):,} ns; '
            f'{html.escape(str(cadence["method"]))}'
        )
    gap_text = "none"
    gaps = stream.get("gaps") or []
    if gaps:
        gap_text = "; ".join(
            f'{html.escape(str(gap["reason"]))} ' f'({int(gap["end_ns"]) - int(gap["start_ns"]):,} ns)' for gap in gaps
        )
    uncertainty = stream.get("maximum_alignment_error_ns")
    uncertainty_text = "not applicable / unaligned" if uncertainty is None else f"±{int(uncertainty):,} ns"
    provenance = html.escape(json.dumps(stream.get("provenance") or {}, sort_keys=True, ensure_ascii=False, indent=2))
    native_only = [
        record
        for record in stream.get("records") or []
        if "native_timestamp_ns" in record
        and "canonical_timestamp_offset_ns" not in record
        and "acquisition_start_offset_ns" not in record
    ]
    native_text = ""
    if native_only:
        native_text = (
            "<div><b>native-only timing</b>: "
            + ", ".join(
                f'{html.escape(str(record["record_id"]))} @ '
                f'{int(record["native_timestamp_ns"]):,} ns '
                f'({html.escape(str(record.get("source_clock_id") or "unknown clock"))})'
                for record in native_only
            )
            + "</div>"
        )
    return f"""
<div class="stream-meta">
  <div><b>cadence</b>: {cadence_text}</div>
  <div><b>gaps</b>: {gap_text}</div>
  <div><b>alignment uncertainty</b>: {uncertainty_text}</div>
  {native_text}
  <div>{_artifact_links_html(stream.get("raw_artifacts") or [])}</div>
  <details class="provenance"><summary>provenance</summary><pre>{provenance}</pre></details>
</div>"""


PLOT_WIDTH = 1000.0
PLOT_HEIGHT = 112.0
PLOT_LEFT = 24.0
PLOT_RIGHT = 12.0
PLOT_TOP = 14.0
PLOT_BOTTOM = 20.0


def _x_for_offset(offset_ns: int, axis: Mapping[str, Any]) -> float:
    minimum = int(axis["minimum_offset_ns"])
    maximum = int(axis["maximum_offset_ns"])
    fraction = (offset_ns - minimum) / max(maximum - minimum, 1)
    return PLOT_LEFT + (PLOT_WIDTH - PLOT_LEFT - PLOT_RIGHT) * fraction


def _y_for_value(value: float, domain: Mapping[str, Any]) -> float:
    minimum = float(domain["minimum"])
    maximum = float(domain["maximum"])
    fraction = (value - minimum) / max(maximum - minimum, 1e-30)
    return PLOT_HEIGHT - PLOT_BOTTOM - fraction * (PLOT_HEIGHT - PLOT_TOP - PLOT_BOTTOM)


def _marker_lines(model: Mapping[str, Any]) -> str:
    axis = model["axis"]
    parts: list[str] = []
    for marker in model.get("lifecycle_markers", []):
        x = _x_for_offset(int(marker["offset_ns"]), axis)
        parts.append(
            f'<line class="lifecycle-marker" data-marker="{html.escape(str(marker["name"]), quote=True)}" '
            f'x1="{x:.3f}" y1="0" x2="{x:.3f}" y2="{PLOT_HEIGHT:.3f}"/>'
        )
    return "".join(parts)


def _gap_bands(stream: Mapping[str, Any], axis: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for gap in stream.get("gaps") or []:
        if not gap.get("canonical"):
            continue
        x1 = _x_for_offset(int(gap["start_offset_ns"]), axis)
        x2 = _x_for_offset(int(gap["end_offset_ns"]), axis)
        parts.append(
            f'<rect class="evidence-gap" x="{x1:.3f}" y="0" width="{max(x2 - x1, 1):.3f}" '
            f'height="{PLOT_HEIGHT:.3f}"><title>{html.escape(str(gap["reason"]))}</title></rect>'
        )
    return "".join(parts)


def _record_x(record: Mapping[str, Any], axis: Mapping[str, Any]) -> float | None:
    if "canonical_timestamp_offset_ns" in record:
        return _x_for_offset(int(record["canonical_timestamp_offset_ns"]), axis)
    if "acquisition_start_offset_ns" in record:
        start = int(record["acquisition_start_offset_ns"])
        end = int(record["acquisition_end_offset_ns"])
        return _x_for_offset((start + end) // 2, axis)
    return None


def _uncertainty_glyph(record: Mapping[str, Any], axis: Mapping[str, Any], y: float) -> str:
    if "acquisition_start_offset_ns" in record:
        start = int(record["acquisition_start_offset_ns"])
        end = int(record["acquisition_end_offset_ns"])
    elif "canonical_timestamp_offset_ns" in record and record.get("alignment_error_ns") is not None:
        timestamp = int(record["canonical_timestamp_offset_ns"])
        error = int(record["alignment_error_ns"])
        start, end = timestamp - error, timestamp + error
    else:
        return ""
    x1 = _x_for_offset(start, axis)
    x2 = _x_for_offset(end, axis)
    return f'<line class="alignment-uncertainty" x1="{x1:.3f}" y1="{y:.3f}" ' f'x2="{x2:.3f}" y2="{y:.3f}"/>'


def _render_point_ticks(stream: Mapping[str, Any], axis: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for index, record in enumerate(stream.get("records") or []):
        x = _record_x(record, axis)
        if x is None:
            continue
        y1 = 26 + (index % 3) * 18
        y2 = y1 + 28
        status = str(record["status"])
        parts.append(_uncertainty_glyph(record, axis, (y1 + y2) / 2))
        if status == "dropped":
            parts.append(
                f'<path class="record record-dropped" d="M{x-5:.3f},{y1+9:.3f} L{x+5:.3f},{y1+19:.3f} '
                f'M{x+5:.3f},{y1+9:.3f} L{x-5:.3f},{y1+19:.3f}"/>'
            )
        else:
            parts.append(
                f'<line class="record record-{html.escape(status)}" x1="{x:.3f}" y1="{y1:.3f}" '
                f'x2="{x:.3f}" y2="{y2:.3f}"><title>{html.escape(_display_text(record.get("value")))}</title></line>'
            )
    return "".join(parts)


def _render_bands(stream: Mapping[str, Any], axis: Mapping[str, Any], *, window: bool) -> str:
    parts: list[str] = []
    for index, record in enumerate(stream.get("records") or []):
        if "canonical_start_offset_ns" not in record:
            continue
        x1 = _x_for_offset(int(record["canonical_start_offset_ns"]), axis)
        x2 = _x_for_offset(int(record["canonical_end_offset_ns"]), axis)
        y = 25 + (index % 3) * 23
        css_class = "window-band" if window else "interval-band"
        parts.append(
            f'<rect class="record {css_class} record-{html.escape(str(record["status"]))}" '
            f'x="{x1:.3f}" y="{y:.3f}" width="{max(x2-x1, 1):.3f}" height="16">'
            f'<title>{html.escape(_display_text(record.get("value")))}</title></rect>'
        )
    return "".join(parts)


def _render_series(stream: Mapping[str, Any], axis: Mapping[str, Any]) -> str:
    domain = stream.get("numeric_value_domain")
    points: list[tuple[float, float, Mapping[str, Any]]] = []
    if isinstance(domain, dict):
        for record in stream.get("records") or []:
            value = record.get("value")
            x = _record_x(record, axis)
            if (
                x is not None
                and isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                points.append((x, _y_for_value(float(value), domain), record))
    parts: list[str] = []
    valid_points = [(x, y) for x, y, record in points if record["status"] == "valid"]
    if len(valid_points) >= 2:
        pairs = " ".join(f"{x:.3f},{y:.3f}" for x, y in valid_points)
        parts.append(f'<polyline class="series-line" points="{pairs}"/>')
    for x, y, record in points:
        parts.append(_uncertainty_glyph(record, axis, y))
        parts.append(
            f'<circle class="record series-point record-{html.escape(str(record["status"]))}" '
            f'cx="{x:.3f}" cy="{y:.3f}" r="3.5"><title>'
            f'{html.escape(_display_text(record.get("value")))}</title></circle>'
        )
    return "".join(parts)


def _track_status_message(stream: Mapping[str, Any]) -> str:
    status = str(stream["status"])
    reasons = stream.get("reasons") or []
    aligned = any(
        "canonical_timestamp_offset_ns" in record
        or "canonical_start_offset_ns" in record
        or "acquisition_start_offset_ns" in record
        for record in stream.get("records") or []
    )
    if status == "valid" and aligned:
        return ""
    detail = "; ".join(str(reason) for reason in reasons) or "no aligned records"
    if status == "unalignable":
        native_count = len(stream.get("records") or [])
        detail = f"{detail}; {native_count} native-timed record(s) retained off-axis"
    return (
        f'<text class="track-status-message status-text-{html.escape(status)}" '
        f'x="{PLOT_LEFT + 8:.3f}" y="{PLOT_HEIGHT / 2:.3f}">{html.escape(status.upper() + ": " + detail)}</text>'
    )


def _track_plot_svg(stream: Mapping[str, Any], model: Mapping[str, Any]) -> str:
    axis = model["axis"]
    kind = stream["render_kind"]
    if kind == "point_ticks":
        records = _render_point_ticks(stream, axis)
    elif kind == "interval_bands":
        records = _render_bands(stream, axis, window=False)
    elif kind == "window_bands":
        records = _render_bands(stream, axis, window=True)
    elif kind in {"sample_series", "counter_series"}:
        records = _render_series(stream, axis)
    else:
        raise UnifiedViewError(f"unsupported timeline render kind: {kind!r}")
    minimum = int(axis["minimum_offset_ns"])
    maximum = int(axis["maximum_offset_ns"])
    ticks = []
    for index in range(5):
        offset = minimum + (maximum - minimum) * index // 4
        x = _x_for_offset(offset, axis)
        ticks.append(f'<line class="axis-grid" x1="{x:.3f}" y1="0" x2="{x:.3f}" y2="{PLOT_HEIGHT:.3f}"/>')
        ticks.append(
            f'<text class="axis-tick" x="{x:.3f}" y="{PLOT_HEIGHT - 4:.3f}" '
            f'text-anchor="middle">{offset / 1_000_000_000:.3f}s</text>'
        )
    return (
        "".join(ticks)
        + _gap_bands(stream, axis)
        + _marker_lines(model)
        + records
        + _track_status_message(stream)
        + f'<line class="shared-cursor" x1="{PLOT_LEFT:.3f}" y1="0" '
        f'x2="{PLOT_LEFT:.3f}" y2="{PLOT_HEIGHT:.3f}"/>'
    )


def _summary_card_html(summary: Mapping[str, Any]) -> str:
    records = summary.get("records") or []
    values = "<br>".join(
        f'<span class="summary-value record-{html.escape(str(record["status"]))}">'
        f'{html.escape(_display_text(record.get("value")))} '
        f'<small>{html.escape(str(summary["unit"]))}</small></span>'
        for record in records
    )
    if not values:
        values = '<span class="summary-value muted">no value</span>'
    exact = (
        bool(records)
        and str(summary["status"]) == "valid"
        and all(record.get("attributes", {}).get("exact") is True for record in records)
    )
    timing_label = "exact untimed fact" if exact else "untimed evidence"
    provenance = html.escape(json.dumps(summary.get("provenance") or {}, sort_keys=True, ensure_ascii=False, indent=2))
    return f"""
<article class="summary-card status-border-{html.escape(str(summary["status"]))}" data-stream-id="{html.escape(str(summary["stream_id"]), quote=True)}">
  <header><span>{html.escape(str(summary["label"]))}</span>{_status_badge(str(summary["status"]))}</header>
  {values}
  {_reasons_html(summary.get("reasons") or [])}
  <div class="summary-meta">{timing_label}; never placed on the timeline</div>
  <div>{_artifact_links_html(summary.get("raw_artifacts") or [])}</div>
  <details class="provenance"><summary>provenance</summary><pre>{provenance}</pre></details>
</article>"""


def _profiler_card_html(profiler: Mapping[str, Any]) -> str:
    status = str(profiler["status"])
    stream_count = len(profiler.get("stream_ids") or [])
    intrusive = " · intrusive" if profiler.get("intrusive") else ""
    return f"""
<article class="profiler-card status-border-{html.escape(status)}" data-profiler-id="{html.escape(str(profiler['profiler_id']), quote=True)}">
  <header><span>{html.escape(str(profiler['profiler_id']))}</span>{_status_badge(status)}</header>
  <div class="profiler-meta">{stream_count} stream(s){intrusive}</div>
  {_reasons_html(profiler.get("reasons") or [])}
</article>"""


def _embedded_model_json(view_model: Mapping[str, Any]) -> str:
    # Prevent user-provided provenance from closing the script element while
    # preserving the exact parsed model represented by the external hash.
    return (
        view_model_bytes(view_model)
        .decode("utf-8")
        .rstrip("\n")
        .replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
    )


def render_unified_html(view_model: Mapping[str, Any]) -> str:
    """Render a dependency-free interactive HTML view from one immutable model."""
    digest = view_model_sha256(view_model)
    summaries = "".join(_summary_card_html(summary) for summary in view_model["summaries"])
    if not summaries:
        summaries = '<p class="muted">No scalar summaries were emitted.</p>'
    profilers = "".join(_profiler_card_html(profiler) for profiler in view_model["profilers"])
    evidence_status = view_model["evidence_status"]
    evidence_reason_text = "; ".join(evidence_status.get("reasons") or [])
    tracks: list[str] = []
    for track in view_model["tracks"]:
        reasons = _reasons_html(track.get("reasons") or [])
        tracks.append(f"""
<details class="track status-border-{html.escape(str(track["status"]))}" open data-stream-id="{html.escape(str(track["stream_id"]), quote=True)}">
  <summary><span class="track-group">{html.escape(str(track["group"]))}</span> · {html.escape(str(track["label"]))} <small>({html.escape(str(track["unit"]))})</small> {_status_badge(str(track["status"]))}</summary>
  {reasons}
  <svg class="track-plot" viewBox="0 0 {PLOT_WIDTH:.0f} {PLOT_HEIGHT:.0f}" role="img" aria-label="{html.escape(str(track["label"]), quote=True)} timeline">{_track_plot_svg(track, view_model)}</svg>
  {_stream_metadata_html(track)}
</details>""")
    markers = (
        ", ".join(
            f'{html.escape(str(marker["name"]))} ({int(marker["offset_ns"]) / 1_000_000_000:.6f}s ±{int(marker["alignment_error_ns"]):,}ns)'
            for marker in view_model["lifecycle_markers"]
        )
        or "none"
    )
    embedded = _embedded_model_json(view_model)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="view-model-sha256" content="{digest}">
<title>Unified sandbox profile</title>
<style>
:root{{--bg:#f8fafc;--panel:#fff;--ink:#111827;--muted:#64748b;--line:#cbd5e1}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:24px}} h1{{margin:0 0 4px}} .subtitle,.muted,.summary-meta{{color:var(--muted)}}
.controls{{position:sticky;top:0;z-index:5;display:flex;gap:18px;align-items:center;padding:10px 12px;margin:18px 0;background:#fffffff2;border:1px solid var(--line);border-radius:8px}}
.controls label{{display:flex;gap:8px;align-items:center}} #cursor-readout{{font-variant-numeric:tabular-nums}}
.summaries,.profiler-statuses{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:14px 0 22px}}
.summary-card,.profiler-card,.track{{background:var(--panel);border:1px solid var(--line);border-left-width:5px;border-radius:8px;padding:11px}}
.summary-card header{{display:flex;justify-content:space-between;gap:8px}} .summary-value{{display:block;font-size:22px;font-weight:700;margin:10px 0}}
.profiler-card header{{display:flex;justify-content:space-between;gap:8px;font-weight:700}} .profiler-meta{{margin-top:7px;color:var(--muted)}}
.status{{display:inline-block;padding:2px 7px;border:1px solid var(--status-color);border-radius:999px;color:var(--status-color);font-size:10px;font-weight:800;vertical-align:middle}}
.status-border-valid{{border-left-color:{STATUS_COLORS['valid']}}}.status-border-partial{{border-left-color:{STATUS_COLORS['partial']}}}.status-border-invalid{{border-left-color:{STATUS_COLORS['invalid']}}}.status-border-missing{{border-left-color:{STATUS_COLORS['missing']}}}.status-border-dropped{{border-left-color:{STATUS_COLORS['dropped']}}}.status-border-unalignable{{border-left-color:{STATUS_COLORS['unalignable']}}}
.track{{margin:10px 0;padding:0}} .track>summary{{cursor:pointer;padding:10px 12px;font-weight:650}} .track-group{{color:#475569}} .track-plot{{display:block;width:100%;height:128px;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;background:#fff;touch-action:none}}
.stream-meta{{display:flex;flex-wrap:wrap;gap:8px 18px;padding:8px 12px;color:#475569;font-size:12px}} .provenance{{width:100%}} pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.reasons{{margin:4px 12px 8px;color:#991b1b}} a{{color:#1d4ed8}} .axis-grid{{stroke:#e5e7eb;stroke-width:1}} .axis-tick{{fill:#64748b;font-size:10px}}
.lifecycle-marker{{stroke:#0f766e;stroke-width:1.2;stroke-dasharray:5 4}} .evidence-gap{{fill:#fee2e2;opacity:.75}}
.record{{stroke-width:2.5}} .record-valid{{stroke:{STATUS_COLORS['valid']};fill:{STATUS_COLORS['valid']}}}.record-invalid{{stroke:{STATUS_COLORS['invalid']};fill:{STATUS_COLORS['invalid']};stroke-dasharray:4 2}}.record-dropped{{stroke:{STATUS_COLORS['dropped']};fill:none}}.record-unalignable{{stroke:{STATUS_COLORS['unalignable']};fill:{STATUS_COLORS['unalignable']}}}
.series-line{{fill:none;stroke:#2563eb;stroke-width:2}} .series-point{{stroke:#fff;stroke-width:1}} .interval-band{{fill:#2563eb;opacity:.7}} .window-band{{fill:#7c3aed;opacity:.45;stroke-dasharray:4 2}} .alignment-uncertainty{{stroke:#111827;stroke-width:1;opacity:.65}} .shared-cursor{{stroke:#e11d48;stroke-width:1.5;visibility:hidden;pointer-events:none}} .track-status-message{{font-size:12px;fill:#475569}} .status-text-invalid{{fill:{STATUS_COLORS['invalid']}}}.status-text-missing{{fill:{STATUS_COLORS['missing']}}}.status-text-dropped{{fill:{STATUS_COLORS['dropped']}}}.status-text-unalignable{{fill:{STATUS_COLORS['unalignable']}}}
</style>
</head>
<body data-view-model-sha256="{digest}"><main>
<h1>Unified sandbox profile</h1>
<div class="subtitle">run {html.escape(str(view_model['run_id']))} · scope {html.escape(str(view_model['scope']['scope_id']))} · canonical {html.escape(str(view_model['axis']['clock_name']))}</div>
<div class="subtitle">lifecycle markers: {markers}</div>
<div class="summary-card status-border-{html.escape(str(evidence_status['status']))}"><b>Evidence status: {html.escape(str(evidence_status['status']).upper())}</b>{(' · ' + html.escape(evidence_reason_text)) if evidence_reason_text else ''}</div>
<div class="controls"><label>Zoom <input id="zoom" type="range" min="1" max="16" value="1" step="1"></label><button id="reset" type="button">Reset view</button><output id="cursor-readout">Move over a track to inspect aligned time</output></div>
<h2>Profiler status</h2><section class="profiler-statuses">{profilers}</section>
<h2>Untimed summaries</h2><section class="summaries">{summaries}</section>
<h2>Timeline tracks</h2><p class="muted">All aligned tracks share the axis below; native-only evidence remains labeled off-axis.</p><section id="tracks">{''.join(tracks)}</section>
<script id="view-model" type="application/json">{embedded}</script>
<script>
(()=>{{
  const model=JSON.parse(document.getElementById('view-model').textContent);
  const plots=[...document.querySelectorAll('.track-plot')];
  const zoom=document.getElementById('zoom'); const readout=document.getElementById('cursor-readout');
  const baseWidth={PLOT_WIDTH}; let visibleWidth=baseWidth;
  function applyZoom(){{visibleWidth=baseWidth/Number(zoom.value); const x=(baseWidth-visibleWidth)/2; plots.forEach(p=>p.setAttribute('viewBox',`${{x}} 0 ${{visibleWidth}} {PLOT_HEIGHT}`));}}
  function move(event){{const source=event.currentTarget; const rect=source.getBoundingClientRect(); const vb=source.viewBox.baseVal; const x=vb.x+Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*vb.width; plots.forEach(p=>{{const c=p.querySelector('.shared-cursor');c.setAttribute('x1',x);c.setAttribute('x2',x);c.style.visibility='visible';}}); const fraction=(x-{PLOT_LEFT})/({PLOT_WIDTH-PLOT_LEFT-PLOT_RIGHT}); const min=model.axis.minimum_offset_ns; const max=model.axis.maximum_offset_ns; const ns=Math.round(min+fraction*(max-min)); readout.value=`cursor: ${{(ns/1e9).toFixed(9)}} s from origin`;}}
  plots.forEach(p=>p.addEventListener('pointermove',move)); zoom.addEventListener('input',applyZoom); document.getElementById('reset').addEventListener('click',()=>{{zoom.value='1';applyZoom();}}); applyZoom();
}})();
</script>
</main></body></html>
"""


def _svg_text(x: float, y: float, text: Any, **attributes: str) -> str:
    attrs = " ".join(
        f'{key.replace("_", "-")}="{html.escape(str(value), quote=True)}"' for key, value in attributes.items()
    )
    return f'<text x="{x:.3f}" y="{y:.3f}" {attrs}>{html.escape(str(text))}</text>'


def _svg_raw_links(artifacts: Sequence[Mapping[str, Any]], *, x: float, y: float) -> str:
    if not artifacts:
        return _svg_text(x, y, "raw: none declared", fill="#64748b", font_size="10")
    parts: list[str] = []
    cursor = x
    for index, artifact in enumerate(artifacts):
        label = (", " if index else "raw: ") + str(artifact["artifact_id"])
        parts.append(
            f'<a href="{html.escape(str(artifact["href"]), quote=True)}" '
            f'data-sha256="{html.escape(str(artifact["sha256"]), quote=True)}">'
            f'{_svg_text(cursor, y, label, fill="#1d4ed8", font_size="10")}</a>'
        )
        cursor += max(len(label) * 6.2, 20)
    return "".join(parts)


def _render_unified_svg(view_model: Mapping[str, Any], *, animated: bool) -> str:
    digest = view_model_sha256(view_model)
    profilers = list(view_model["profilers"])
    summaries = list(view_model["summaries"])
    tracks = list(view_model["tracks"])
    profiler_rows = max(1, math.ceil(len(profilers) / 3))
    profiler_height = profiler_rows * 76
    summary_rows = max(1, math.ceil(len(summaries) / 3))
    summary_height = summary_rows * 100
    profiler_start = 146
    summary_title_y = profiler_start + profiler_height + 18
    summary_start = summary_title_y + 12
    track_height = 154
    track_start = summary_start + summary_height + 32
    height = track_start + max(len(tracks), 1) * track_height + 48
    width = 1200
    chart_left = 170
    chart_width = 1000
    model_json = html.escape(view_model_bytes(view_model).decode("utf-8").rstrip("\n"), quote=False)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="{width}" height="{height}" viewBox="0 0 {width} {height}" data-view-model-sha256="{digest}">',
        f'<metadata id="view-model" data-sha256="{digest}">{model_json}</metadata>',
        '<rect width="100%" height="100%" fill="#f8fafc"/>',
        _svg_text(24, 34, "Unified sandbox profile", fill="#111827", font_size="22", font_weight="700"),
        _svg_text(
            24,
            58,
            f"run {view_model['run_id']} · scope {view_model['scope']['scope_id']}",
            fill="#475569",
            font_size="12",
        ),
        _svg_text(
            24,
            78,
            f"canonical {view_model['axis']['clock_name']} · view model sha256 {digest}",
            fill="#64748b",
            font_size="10",
        ),
        _svg_text(
            24,
            100,
            f"Evidence status: {str(view_model['evidence_status']['status']).upper()}",
            fill=STATUS_COLORS.get(str(view_model["evidence_status"]["status"]), "#374151"),
            font_size="12",
            font_weight="700",
        ),
        _svg_text(24, 132, "Profiler status", fill="#111827", font_size="15", font_weight="700"),
    ]
    for index, profiler in enumerate(profilers):
        column = index % 3
        row = index // 3
        x = 24 + column * 382
        y = profiler_start + row * 76
        status = str(profiler["status"])
        reason = str((profiler.get("reasons") or [""])[0])
        parts.extend(
            [
                f'<rect x="{x}" y="{y}" width="360" height="64" rx="6" fill="#fff" stroke="#cbd5e1"/>',
                f'<rect x="{x}" y="{y}" width="5" height="64" fill="{STATUS_COLORS.get(status, "#374151")}"/>',
                _svg_text(x + 14, y + 22, profiler["profiler_id"], fill="#111827", font_size="12", font_weight="700"),
                _svg_text(
                    x + 268,
                    y + 22,
                    status.upper(),
                    fill=STATUS_COLORS.get(status, "#374151"),
                    font_size="10",
                    font_weight="700",
                ),
                _svg_text(
                    x + 14,
                    y + 41,
                    f"{len(profiler.get('stream_ids') or [])} stream(s)"
                    + (" · intrusive" if profiler.get("intrusive") else ""),
                    fill="#64748b",
                    font_size="9",
                ),
            ]
        )
        if reason:
            parts.append(_svg_text(x + 14, y + 56, reason, fill=STATUS_COLORS.get(status, "#374151"), font_size="8"))
    if not profilers:
        parts.append(
            _svg_text(24, profiler_start + 24, "No profiler manifests emitted", fill="#64748b", font_size="12")
        )
    parts.append(_svg_text(24, summary_title_y, "Untimed summaries", fill="#111827", font_size="15", font_weight="700"))
    for index, summary in enumerate(summaries):
        column = index % 3
        row = index // 3
        x = 24 + column * 382
        y = summary_start + row * 100
        status = str(summary["status"])
        parts.append(
            f'<rect x="{x}" y="{y}" width="360" height="88" rx="6" fill="#fff" stroke="#cbd5e1" stroke-width="1"/>'
        )
        parts.append(f'<rect x="{x}" y="{y}" width="5" height="88" fill="{STATUS_COLORS.get(status, "#374151")}"/>')
        parts.append(_svg_text(x + 14, y + 21, summary["label"], fill="#111827", font_size="12", font_weight="700"))
        values = summary.get("records") or []
        value = _display_text(values[0].get("value")) if values else "no value"
        parts.append(
            _svg_text(x + 14, y + 47, f"{value} {summary['unit']}", fill="#111827", font_size="17", font_weight="700")
        )
        parts.append(
            _svg_text(
                x + 250,
                y + 21,
                status.upper(),
                fill=STATUS_COLORS.get(status, "#374151"),
                font_size="10",
                font_weight="700",
            )
        )
        exact = (
            bool(values)
            and status == "valid"
            and all(record.get("attributes", {}).get("exact") is True for record in values)
        )
        timing_label = "exact untimed fact" if exact else "untimed evidence"
        parts.append(_svg_text(x + 14, y + 64, f"{timing_label} · not plotted", fill="#64748b", font_size="9"))
        reasons = summary.get("reasons") or []
        reason = str(reasons[0]) if reasons else ""
        if reason:
            parts.append(_svg_text(x + 14, y + 80, reason, fill=STATUS_COLORS.get(status, "#374151"), font_size="8"))
        parts.append(_svg_raw_links(summary.get("raw_artifacts") or [], x=x + 220, y=y + 64))
    if not summaries:
        parts.append(_svg_text(24, summary_start + 24, "No scalar summaries emitted", fill="#64748b", font_size="12"))
    parts.append(
        _svg_text(
            24,
            track_start - 16,
            "Shared run-relative timeline · native-only evidence retained off-axis",
            fill="#111827",
            font_size="15",
            font_weight="700",
        )
    )
    for index, track in enumerate(tracks):
        y = track_start + index * track_height
        status = str(track["status"])
        parts.extend(
            [
                f'<g class="track" data-stream-id="{html.escape(str(track["stream_id"]), quote=True)}" transform="translate(0,{y})">',
                '<rect x="16" y="0" width="1168" height="142" rx="6" fill="#fff" stroke="#cbd5e1"/>',
                f'<rect x="16" y="0" width="5" height="142" fill="{STATUS_COLORS.get(status, "#374151")}"/>',
                _svg_text(30, 22, track["group"], fill="#64748b", font_size="10"),
                _svg_text(30, 42, track["label"], fill="#111827", font_size="12", font_weight="700"),
                _svg_text(30, 59, f"{track['shape']} · {track['unit']}", fill="#475569", font_size="9"),
                _svg_text(
                    30, 76, status.upper(), fill=STATUS_COLORS.get(status, "#374151"), font_size="10", font_weight="700"
                ),
                f'<svg class="track-plot" x="{chart_left}" y="8" width="{chart_width}" height="112" viewBox="0 0 {PLOT_WIDTH:.0f} {PLOT_HEIGHT:.0f}">{_track_plot_svg(track, view_model)}</svg>',
                _svg_raw_links(track.get("raw_artifacts") or [], x=30, y=126),
                "</g>",
            ]
        )
    if not tracks:
        parts.append(_svg_text(24, track_start + 30, "No temporal streams emitted", fill="#64748b", font_size="12"))
    if animated and tracks:
        animation_top = track_start + 8
        animation_bottom = track_start + (len(tracks) - 1) * track_height + 120
        x1 = chart_left + PLOT_LEFT
        x2 = chart_left + PLOT_WIDTH - PLOT_RIGHT
        parts.append(
            f'<line class="animated-shared-cursor" x1="{x1:.3f}" y1="{animation_top:.3f}" '
            f'x2="{x1:.3f}" y2="{animation_bottom:.3f}" stroke="#e11d48" stroke-width="2">'
            f'<animate attributeName="x1" from="{x1:.3f}" to="{x2:.3f}" dur="8s" repeatCount="indefinite"/>'
            f'<animate attributeName="x2" from="{x1:.3f}" to="{x2:.3f}" dur="8s" repeatCount="indefinite"/>'
            "</line>"
        )
    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def render_unified_svg(view_model: Mapping[str, Any]) -> str:
    return _render_unified_svg(view_model, animated=False)


def render_unified_animated_svg(view_model: Mapping[str, Any]) -> str:
    """Render the deterministic Milestone-1 animation export as animated SVG."""
    return _render_unified_svg(view_model, animated=True)


def write_unified_view_artifacts(bundle: EvidenceBundle, output_dir: Path) -> dict[str, str]:
    """Write all exports from one assembled model without invoking execution."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "view_model": output_dir / "view_model.json",
        "html": output_dir / "unified_profile.html",
        "svg": output_dir / "unified_profile.svg",
        "animated_svg": output_dir / "unified_profile.animated.svg",
    }
    existing = [str(path) for path in paths.values() if path.exists()]
    if existing:
        raise UnifiedViewError("refusing to overwrite unified view artifacts: " + ", ".join(existing))
    model = assemble_view_model(bundle)
    paths["view_model"].write_bytes(view_model_bytes(model))
    paths["html"].write_text(render_unified_html(model))
    paths["svg"].write_text(render_unified_svg(model))
    paths["animated_svg"].write_text(render_unified_animated_svg(model))
    result = {key: str(path) for key, path in paths.items()}
    result["view_model_sha256"] = view_model_sha256(model)
    return result


def write_unified_view_from_bundle(bundle_path: Path, output_dir: Path) -> dict[str, str]:
    """Read a retained evidence bundle and render it; never execute a workload."""
    return write_unified_view_artifacts(read_bundle(bundle_path), output_dir)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Render one retained typed evidence bundle as synchronized HTML, "
            "static SVG, and animated SVG without rerunning the workload."
        )
    )
    parser.add_argument("bundle", type=Path, help="Typed evidence_bundle.json path.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New or reusable-empty directory for deterministic view artifacts.",
    )
    args = parser.parse_args(argv)
    try:
        outputs = write_unified_view_from_bundle(args.bundle, args.output_dir)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    print(f"Unified profile HTML: {outputs['html']}")
    print(f"Unified profile SVG: {outputs['svg']}")
    print(f"Unified animated SVG: {outputs['animated_svg']}")
    print(f"View-model SHA-256: {outputs['view_model_sha256']}")
    return 0


__all__ = [
    "SHAPE_REGISTRY",
    "UnifiedViewError",
    "VIEW_MODEL_SCHEMA_VERSION",
    "assemble_view_model",
    "render_unified_animated_svg",
    "render_unified_html",
    "render_unified_svg",
    "view_model_bytes",
    "view_model_sha256",
    "write_unified_view_artifacts",
    "write_unified_view_from_bundle",
]


if __name__ == "__main__":
    sys.exit(main())
