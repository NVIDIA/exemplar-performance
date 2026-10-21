# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render a clock-aligned cgroup CPU and memory timeline as dependency-free SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
import sys
from pathlib import Path
from typing import Any

from ..measurement.cgroup_resources import TIMESERIES_SCHEMA_VERSION


class ResourceTimelineError(RuntimeError):
    """Raised when a profile lacks valid resource time-series evidence."""


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ResourceTimelineError(f"unable to read JSON report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ResourceTimelineError(f"expected a JSON object in {path}")
    return payload


def _nonnegative_finite_number(value: Any, *, field: str) -> float:
    try:
        valid = isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except OverflowError:
        valid = False
    if not valid:
        raise ResourceTimelineError(f"{field} must be a finite nonnegative number")
    return float(value)


def _validate_resource_timeline(memory: dict[str, Any], timeline: dict[str, Any]) -> None:
    if memory.get("timeline_valid_for_shape") is False:
        reasons = memory.get("timeline_invalid_reasons") or []
        detail = "; ".join(str(reason) for reason in reasons)
        raise ResourceTimelineError(
            "container memory timeline is invalid for shape reporting" + (f": {detail}" if detail else "")
        )
    interval = _nonnegative_finite_number(timeline.get("sample_interval_sec"), field="sample_interval_sec")
    if interval == 0:
        raise ResourceTimelineError("sample_interval_sec must be positive")
    samples = timeline.get("samples")
    if not isinstance(samples, list) or len(samples) < 2:
        raise ResourceTimelineError("resource time series requires at least two samples")

    first_elapsed: float | None = None
    previous_elapsed: float | None = None
    for index, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise ResourceTimelineError(f"samples[{index}] must be an object")
        elapsed = _nonnegative_finite_number(sample.get("elapsed_sec"), field=f"samples[{index}].elapsed_sec")
        if previous_elapsed is not None and elapsed <= previous_elapsed:
            raise ResourceTimelineError("sample elapsed_sec values must be strictly increasing")
        if previous_elapsed is not None and elapsed - previous_elapsed > 2 * interval:
            raise ResourceTimelineError(f"samples[{index}] exceeds twice the requested sample interval")
        if first_elapsed is None:
            first_elapsed = elapsed
        previous_elapsed = elapsed
        _nonnegative_finite_number(
            sample.get("memory_current_bytes"),
            field=f"samples[{index}].memory_current_bytes",
        )
        cpu = sample.get("cpu_average_logical_cores")
        if cpu is not None:
            _nonnegative_finite_number(cpu, field=f"samples[{index}].cpu_average_logical_cores")
        memory_stat = sample.get("memory_stat")
        if not isinstance(memory_stat, dict):
            raise ResourceTimelineError(f"samples[{index}].memory_stat must be an object")
        for key in ("anon", "file", "kernel"):
            if key in memory_stat:
                _nonnegative_finite_number(memory_stat[key], field=f"samples[{index}].memory_stat.{key}")

    assert first_elapsed is not None and previous_elapsed is not None
    markers = timeline.get("markers")
    if not isinstance(markers, list):
        raise ResourceTimelineError("resource time series markers must be a list")
    previous_marker_elapsed: float | None = None
    marker_times: dict[str, float] = {}
    for index, marker in enumerate(markers):
        if not isinstance(marker, dict):
            raise ResourceTimelineError(f"markers[{index}] must be an object")
        name = marker.get("name")
        if not isinstance(name, str) or not name:
            raise ResourceTimelineError(f"markers[{index}].name must be nonempty")
        elapsed = _nonnegative_finite_number(marker.get("elapsed_sec"), field=f"markers[{index}].elapsed_sec")
        if elapsed < first_elapsed or elapsed > previous_elapsed:
            raise ResourceTimelineError(f"markers[{index}].elapsed_sec falls outside the sample window")
        if previous_marker_elapsed is not None and elapsed < previous_marker_elapsed:
            raise ResourceTimelineError("marker elapsed_sec values must be monotonic")
        previous_marker_elapsed = elapsed
        marker_times[name] = elapsed
    if (
        "command_start" in marker_times
        and "command_end" in marker_times
        and marker_times["command_end"] < marker_times["command_start"]
    ):
        raise ResourceTimelineError("command_end marker precedes command_start")

    lifetime = memory.get("lifetime")
    if isinstance(lifetime, dict) and lifetime.get("peak_bytes") is not None:
        _nonnegative_finite_number(lifetime["peak_bytes"], field="memory.lifetime.peak_bytes")


def load_resource_timeline(report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a profile/pipeline memory summary and its external raw samples."""
    report = _load_json(report_path)
    memory = report.get("container_memory") or report.get("memory")
    if not isinstance(memory, dict):
        raise ResourceTimelineError(f"report has no container memory profile: {report_path}")
    raw_samples_path = memory.get("samples_path")
    if not isinstance(raw_samples_path, str) or not raw_samples_path:
        raise ResourceTimelineError("container memory profile has no samples_path")
    samples_path = Path(raw_samples_path)
    if not samples_path.is_absolute():
        samples_path = report_path.parent / samples_path
    timeline = _load_json(samples_path)
    if timeline.get("schema_version") != TIMESERIES_SCHEMA_VERSION:
        raise ResourceTimelineError(f"unsupported resource time-series schema: {timeline.get('schema_version')!r}")
    _validate_resource_timeline(memory, timeline)
    cpu = report.get("container_cpu") or report.get("cpu") or {}
    if not isinstance(cpu, dict):
        raise ResourceTimelineError("container CPU summary must be an object")
    if cpu.get("allocated_logical_cpus") is not None:
        _nonnegative_finite_number(cpu["allocated_logical_cpus"], field="allocated_logical_cpus")
    return report, timeline


def _svg_text(x: float, y: float, value: str, **attributes: str) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{html.escape(attribute)}"' for key, attribute in attributes.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{html.escape(value)}</text>'


def _polyline(
    samples: list[dict[str, Any]],
    *,
    field: str,
    x_for: Any,
    y_for: Any,
    color: str,
    width: float = 2.0,
) -> str:
    points = [
        f"{x_for(float(sample['elapsed_sec'])):.2f},{y_for(float(value)):.2f}"
        for sample in samples
        if isinstance((value := sample.get(field)), (int, float))
    ]
    if len(points) < 2:
        return ""
    return f'<polyline points="{" ".join(points)}" fill="none" ' f'stroke="{color}" stroke-width="{width:.1f}"/>'


def _interval_steps(
    samples: list[dict[str, Any]],
    *,
    field: str,
    x_for: Any,
    y_for: Any,
    color: str,
    width: float = 2.0,
) -> str:
    """Draw each interval average over the interval it actually summarizes."""
    lines: list[str] = []
    previous_value: float | None = None
    for previous, current in zip(samples, samples[1:], strict=False):
        raw_value = current.get(field)
        if not isinstance(raw_value, (int, float)) or isinstance(raw_value, bool):
            previous_value = None
            continue
        value = float(raw_value)
        x_start = x_for(float(previous["elapsed_sec"]))
        x_end = x_for(float(current["elapsed_sec"]))
        y = y_for(value)
        if previous_value is not None:
            previous_y = y_for(previous_value)
            lines.append(
                f'<line class="cpu-interval-transition" x1="{x_start:.2f}" '
                f'y1="{previous_y:.2f}" x2="{x_start:.2f}" y2="{y:.2f}" '
                f'stroke="{color}" stroke-width="{width:.1f}"/>'
            )
        lines.append(
            f'<line class="cpu-interval" x1="{x_start:.2f}" y1="{y:.2f}" '
            f'x2="{x_end:.2f}" y2="{y:.2f}" stroke="{color}" '
            f'stroke-width="{width:.1f}"/>'
        )
        previous_value = value
    return "\n".join(lines)


def render_resource_timeline_svg(report_path: Path, output_path: Path) -> None:
    """Render aligned interval-CPU and charged-memory panels."""
    report, timeline = load_resource_timeline(report_path)
    memory = report.get("container_memory") or report["memory"]
    samples: list[dict[str, Any]] = timeline["samples"]
    markers = timeline.get("markers", [])
    duration = max(float(sample["elapsed_sec"]) for sample in samples)
    duration = max(duration, 0.001)
    cpu_values = [
        float(sample["cpu_average_logical_cores"])
        for sample in samples
        if isinstance(sample.get("cpu_average_logical_cores"), (int, float))
    ]
    memory_values = [float(sample["memory_current_bytes"]) for sample in samples]
    cpu_capacity = (report.get("container_cpu") or report.get("cpu") or {}).get("allocated_logical_cpus")
    cpu_max = max(cpu_values + [float(cpu_capacity or 0), 1.0]) * 1.08
    memory_max_mib = max(memory_values) / (1024 * 1024)
    lifetime_peak = memory.get("lifetime", {}).get("peak_bytes")
    if isinstance(lifetime_peak, (int, float)) and not isinstance(lifetime_peak, bool):
        peak_status = "exact" if memory.get("peak_valid") is True else "invalid"
        lifetime_peak_text = (
            "Untimed controlled-finish-gate peak: "
            f"{float(lifetime_peak) / (1024 * 1024):.1f} MiB · {peak_status} "
            "(annotated only; not on the timed y-axis)"
        )
    else:
        lifetime_peak_text = "Untimed controlled-finish-gate peak: unavailable"
    margin = memory.get("observed_limit_margin", {})
    configured_limit = margin.get("configured_limit_bytes")
    remaining_to_limit = margin.get("remaining_to_limit_bytes")
    peak_fraction = margin.get("peak_fraction_of_limit")
    if (
        isinstance(configured_limit, int)
        and isinstance(remaining_to_limit, int)
        and margin.get("arithmetic_valid") is True
    ):
        fraction_text = (
            f" · peak/limit={float(peak_fraction) * 100:.1f}%"
            if isinstance(peak_fraction, (int, float)) and not isinstance(peak_fraction, bool)
            else ""
        )
        limit_margin_text = (
            f"Leaf memory.max: {configured_limit / (1024 * 1024):.1f} MiB · "
            f"observed margin: {remaining_to_limit / (1024 * 1024):.1f} MiB"
            f"{fraction_text} · not a recommendation"
        )
    elif isinstance(configured_limit, int):
        limit_margin_text = (
            f"Leaf memory.max: {configured_limit / (1024 * 1024):.1f} MiB · "
            "observed margin invalid because the peak is invalid"
        )
    elif margin.get("configured_limit_state") == "unlimited":
        limit_margin_text = "Leaf memory.max: unlimited · observed margin unavailable"
    else:
        limit_margin_text = "Leaf memory.max and observed margin: unavailable"
    # ``memory.peak`` has no timestamp. Scaling a time-series axis to it would
    # imply a timed point and can flatten the observed curve after a short spike.
    memory_max_mib = max(memory_max_mib * 1.08, 1.0)

    width, height = 1080, 720
    left, right = 86, 38
    chart_width = width - left - right
    cpu_top, panel_height = 132, 220
    memory_top = 420

    def x_for(elapsed: float) -> float:
        return left + chart_width * elapsed / duration

    def cpu_y(value: float) -> float:
        return cpu_top + panel_height - panel_height * value / cpu_max

    def memory_y_bytes(value: float) -> float:
        value_mib = value / (1024 * 1024)
        return memory_top + panel_height - panel_height * value_mib / memory_max_mib

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" data-timed-memory-max-mib="{memory_max_mib:.6f}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(
            28,
            34,
            "Sandbox CPU and memory over time",
            fill="#111827",
            font_size="20",
            font_weight="bold",
        ),
        _svg_text(
            28,
            58,
            f"{report_path.name} · cgroup subtree · {len(samples)} samples · "
            f"interval={timeline.get('sample_interval_sec')}s · "
            f"timeline={'valid' if memory.get('timeline_valid_for_shape', True) else 'invalid'} · "
            "natural-peak="
            f"{'trustworthy' if memory.get('natural_peak_trustworthy') else 'constrained'}",
            fill="#4b5563",
            font_size="12",
        ),
        _svg_text(28, 80, lifetime_peak_text, fill="#6b7280", font_size="11"),
        _svg_text(28, 100, limit_margin_text, fill="#6b7280", font_size="11"),
        _svg_text(
            left, cpu_top - 18, "Sampled interval CPU demand", fill="#111827", font_size="14", font_weight="bold"
        ),
        _svg_text(
            left, memory_top - 18, "Cgroup-charged memory (MiB)", fill="#111827", font_size="14", font_weight="bold"
        ),
    ]
    for top in (cpu_top, memory_top):
        svg.append(
            f'<rect x="{left}" y="{top}" width="{chart_width}" height="{panel_height}" fill="#f9fafb" stroke="#d1d5db"/>'
        )
    for index in range(6):
        elapsed = duration * index / 5
        x = x_for(elapsed)
        for top in (cpu_top, memory_top):
            svg.append(
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + panel_height}" stroke="#e5e7eb" stroke-width="1"/>'
            )
        svg.append(_svg_text(x - 10, memory_top + panel_height + 22, f"{elapsed:.2f}", fill="#4b5563", font_size="10"))
    for index in range(5):
        fraction = index / 4
        cpu_value = cpu_max * (1 - fraction)
        memory_value = memory_max_mib * (1 - fraction)
        cpu_line_y = cpu_top + panel_height * fraction
        memory_line_y = memory_top + panel_height * fraction
        svg.extend(
            [
                f'<line x1="{left}" y1="{cpu_line_y:.2f}" x2="{left + chart_width}" y2="{cpu_line_y:.2f}" stroke="#e5e7eb"/>',
                f'<line x1="{left}" y1="{memory_line_y:.2f}" x2="{left + chart_width}" y2="{memory_line_y:.2f}" stroke="#e5e7eb"/>',
                _svg_text(18, cpu_line_y + 4, f"{cpu_value:.2f}", fill="#4b5563", font_size="10"),
                _svg_text(18, memory_line_y + 4, f"{memory_value:.1f}", fill="#4b5563", font_size="10"),
            ]
        )
    svg.extend(
        [
            _interval_steps(
                samples,
                field="cpu_average_logical_cores",
                x_for=x_for,
                y_for=cpu_y,
                color="#2563eb",
                width=2.4,
            ),
            _polyline(
                samples, field="memory_current_bytes", x_for=x_for, y_for=memory_y_bytes, color="#dc2626", width=2.5
            ),
        ]
    )
    breakdown = (
        ("memory_anon", "#f59e0b", "anon"),
        ("memory_file", "#16a34a", "file cache"),
        ("memory_kernel", "#7c3aed", "kernel"),
    )
    for field, color, _label in breakdown:
        projected = [
            {**sample, field: (sample.get("memory_stat") or {}).get(field.split("_", 1)[1])} for sample in samples
        ]
        svg.append(
            _polyline(
                projected,
                field=field,
                x_for=x_for,
                y_for=memory_y_bytes,
                color=color,
                width=1.4,
            )
        )
    marker_colors = {"command_start": "#059669", "command_end": "#9333ea"}
    for marker in markers:
        name = marker.get("name")
        elapsed = marker.get("elapsed_sec")
        if name not in marker_colors or not isinstance(elapsed, (int, float)):
            continue
        x = x_for(float(elapsed))
        color = marker_colors[name]
        svg.append(
            f'<line x1="{x:.2f}" y1="{cpu_top}" x2="{x:.2f}" y2="{memory_top + panel_height}" stroke="{color}" stroke-width="1.5" stroke-dasharray="5 4"/>'
        )
        svg.append(_svg_text(x + 4, cpu_top + 14, name.replace("_", " "), fill=color, font_size="10"))
    svg.extend(
        [
            _svg_text(left, height - 24, "Elapsed seconds (shared monotonic clock)", fill="#374151", font_size="11"),
            _svg_text(left + 280, cpu_top - 18, "blue: interval-average logical cores", fill="#2563eb", font_size="11"),
            _svg_text(left + 250, memory_top - 18, "red: total", fill="#dc2626", font_size="11"),
            _svg_text(left + 325, memory_top - 18, "orange: anon", fill="#f59e0b", font_size="11"),
            _svg_text(left + 420, memory_top - 18, "green: file", fill="#16a34a", font_size="11"),
            _svg_text(left + 505, memory_top - 18, "purple: kernel", fill="#7c3aed", font_size="11"),
            "</svg>",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(part for part in svg if part) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Render one profile's aligned cgroup CPU and memory timeline.")
    parser.add_argument("report", type=Path, help="pipeline_report.json or profile_report.json")
    parser.add_argument("--output", type=Path, required=True, help="New SVG output path.")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output path already exists: {args.output}")
    try:
        render_resource_timeline_svg(args.report, args.output)
    except ResourceTimelineError as exc:
        parser.error(str(exc))
    print(f"Resource timeline: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
