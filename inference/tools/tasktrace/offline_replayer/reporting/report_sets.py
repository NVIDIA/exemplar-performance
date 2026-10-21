#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regenerate scheduler-concurrency report bundles from named manifests.

Validated reports and diagnostic reports are intentionally declared in
separate manifest maps: the former feed charts, while the latter are rendered
as clearly labeled observed-window evidence only.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .scheduler_batch import BatchReportError, build_summary, write_outputs

MANIFEST_SCHEMA_VERSION = "scheduler-report-sets.v1"


class ReportSetError(RuntimeError):
    """Raised when a report-set manifest cannot be rendered."""


@dataclass(frozen=True)
class ReportSetConfig:
    name: str
    output_dir: Path
    pipeline_reports: tuple[Path, ...]
    diagnostic_pipeline_reports: tuple[Path, ...] = ()


def _resolve_path(value: Any, *, base_dir: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ReportSetError(f"{field} must be a non-empty path string")
    path = Path(value)
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def load_report_sets(manifest_path: Path) -> list[ReportSetConfig]:
    """Load reusable report bundles while preserving validated/diagnostic separation."""
    try:
        payload = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ReportSetError(f"unable to read report-set manifest {manifest_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReportSetError("report-set manifest root must be a JSON object")
    if payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ReportSetError(f"unsupported report-set manifest schema: {payload.get('schema_version')!r}")
    raw_reports = payload.get("reports")
    raw_sets = payload.get("sets")
    raw_diagnostic_reports = payload.get("diagnostic_reports", {})
    if not isinstance(raw_reports, dict) or not raw_reports:
        raise ReportSetError("report-set manifest must define a non-empty reports object")
    if not isinstance(raw_sets, list) or not raw_sets:
        raise ReportSetError("report-set manifest must define a non-empty sets list")
    if not isinstance(raw_diagnostic_reports, dict):
        raise ReportSetError("diagnostic_reports must be an object when defined")

    base_dir = manifest_path.resolve().parent
    reports: dict[str, Path] = {}
    for report_id, path in raw_reports.items():
        if not isinstance(report_id, str) or not report_id:
            raise ReportSetError("report IDs must be non-empty strings")
        reports[report_id] = _resolve_path(path, base_dir=base_dir, field=f"reports[{report_id!r}]")
    diagnostic_reports: dict[str, Path] = {}
    for report_id, path in raw_diagnostic_reports.items():
        if not isinstance(report_id, str) or not report_id:
            raise ReportSetError("diagnostic report IDs must be non-empty strings")
        diagnostic_reports[report_id] = _resolve_path(
            path, base_dir=base_dir, field=f"diagnostic_reports[{report_id!r}]"
        )

    sets: list[ReportSetConfig] = []
    names: set[str] = set()
    output_dirs: set[Path] = set()
    for index, raw_set in enumerate(raw_sets):
        if not isinstance(raw_set, dict):
            raise ReportSetError(f"sets[{index}] must be an object")
        name = raw_set.get("name")
        if not isinstance(name, str) or not name:
            raise ReportSetError(f"sets[{index}].name must be a non-empty string")
        if name in names:
            raise ReportSetError(f"duplicate report-set name: {name}")
        output_dir = _resolve_path(raw_set.get("output_dir"), base_dir=base_dir, field=f"sets[{index}].output_dir")
        if output_dir in output_dirs:
            raise ReportSetError(f"duplicate report-set output directory: {output_dir}")
        report_ids = raw_set.get("report_ids")
        if not isinstance(report_ids, list) or not report_ids:
            raise ReportSetError(f"sets[{index}].report_ids must be a non-empty list")
        unknown = [report_id for report_id in report_ids if not isinstance(report_id, str) or report_id not in reports]
        if unknown:
            raise ReportSetError(f"report set {name!r} references unknown report IDs: {unknown}")
        if len(set(report_ids)) != len(report_ids):
            raise ReportSetError(f"report set {name!r} repeats a pipeline report")
        diagnostic_report_ids = raw_set.get("diagnostic_report_ids", [])
        if not isinstance(diagnostic_report_ids, list):
            raise ReportSetError(f"sets[{index}].diagnostic_report_ids must be a list")
        unknown_diagnostic = [
            report_id
            for report_id in diagnostic_report_ids
            if not isinstance(report_id, str) or report_id not in diagnostic_reports
        ]
        if unknown_diagnostic:
            raise ReportSetError(
                f"report set {name!r} references unknown diagnostic report IDs: " f"{unknown_diagnostic}"
            )
        if len(set(diagnostic_report_ids)) != len(diagnostic_report_ids):
            raise ReportSetError(f"report set {name!r} repeats a diagnostic pipeline report")
        sets.append(
            ReportSetConfig(
                name=name,
                output_dir=output_dir,
                pipeline_reports=tuple(reports[report_id] for report_id in report_ids),
                diagnostic_pipeline_reports=tuple(diagnostic_reports[report_id] for report_id in diagnostic_report_ids),
            )
        )
        names.add(name)
        output_dirs.add(output_dir)
    return sets


def write_diagnostic_outputs(paths: tuple[Path, ...], output_dir: Path) -> dict[str, str]:
    """Write explicitly non-comparable observed-window metrics for invalid runs."""
    rows: list[dict[str, Any]] = []
    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            raise ReportSetError(f"unable to read diagnostic pipeline report {path}: {exc}") from exc
        replay = payload.get("replay_summary", {})
        cpu = payload.get("container_cpu", {})
        scheduler = payload.get("scheduler_concurrency", {})
        validation = scheduler.get("cgroup_validation", {})
        lost_events = int(scheduler.get("lost_perf_events") or 0)
        anomalies = list(scheduler.get("anomalies") or [])
        scheduler_valid = bool(validation.get("valid_for_concurrency_reporting"))
        peak = scheduler.get("peak_simultaneous_logical_cores")
        if scheduler_valid:
            peak_interpretation = "exact_for_observed_non_comparable_window"
        elif peak is not None:
            peak_interpretation = "observed_lower_bound_only"
        else:
            peak_interpretation = "unavailable"
        rows.append(
            {
                "task_name": str(payload.get("task_name") or ""),
                "source_trial": str(payload.get("source_trial") or ""),
                "pipeline_report": str(path.resolve()),
                "classification": "diagnostic_non_comparable_replay",
                "source_outcome_mismatch_count": int(replay.get("source_outcome_mismatch_count") or 0),
                "source_outcome_comparison_unavailable_count": int(
                    replay.get("source_outcome_comparison_unavailable_count") or 0
                ),
                "observed_window_wall_sec": float(cpu.get("wall_sec") or 0),
                "observed_window_cpu_usage_sec": float(cpu.get("usage_sec") or 0),
                "observed_window_average_logical_cores_used": float(cpu.get("average_logical_cores_used") or 0),
                "observed_window_utilization_percent_of_allocation": float(
                    cpu.get("utilization_percent_of_allocation") or 0
                ),
                "scheduler_observed_peak_logical_cores": peak,
                "scheduler_peak_interpretation": peak_interpretation,
                "scheduler_lost_perf_events": lost_events,
                "scheduler_anomaly_count": len(anomalies),
                "scheduler_valid_for_concurrency_reporting": scheduler_valid,
                "full_replay_average_inference_allowed": False,
            }
        )
    rows.sort(key=lambda row: row["task_name"])
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "diagnostic_observations.json"
    csv_path = output_dir / "diagnostic_observations.csv"
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "scheduler-diagnostic-observations.v1",
                "interpretation": (
                    "Observed-window cgroup averages from non-comparable replays. "
                    "These rows are excluded from validated charts. Scheduler peaks "
                    "are lower bounds when the scheduler trace is invalid."
                ),
                "tasks": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    fields = (
        list(rows[0])
        if rows
        else [
            "task_name",
            "classification",
            "observed_window_average_logical_cores_used",
        ]
    )
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {
        "diagnostic_json": str(json_path.resolve()),
        "diagnostic_csv": str(csv_path.resolve()),
    }


def render_report_sets(
    manifest_path: Path, *, render_png: bool = False, selected_names: tuple[str, ...] = ()
) -> dict[str, dict[str, str]]:
    """Render every selected set from a manifest using the validated batch renderer."""
    sets = load_report_sets(manifest_path)
    available = {report_set.name for report_set in sets}
    missing = sorted(set(selected_names) - available)
    if missing:
        raise ReportSetError("unknown requested report sets: " + ", ".join(missing))
    selected = [report_set for report_set in sets if report_set.name in selected_names] if selected_names else sets
    outputs_by_set: dict[str, dict[str, str]] = {}
    for report_set in selected:
        outputs = write_outputs(
            build_summary(list(report_set.pipeline_reports)),
            report_set.output_dir,
            render_png=render_png,
        )
        if report_set.diagnostic_pipeline_reports:
            outputs.update(write_diagnostic_outputs(report_set.diagnostic_pipeline_reports, report_set.output_dir))
        outputs_by_set[report_set.name] = outputs
    return outputs_by_set


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Regenerate scheduler-concurrency report bundles declared in a JSON manifest."
    )
    parser.add_argument("manifest", type=Path, help="Report-set manifest JSON path.")
    parser.add_argument(
        "--set",
        dest="selected_names",
        action="append",
        default=[],
        help="Regenerate only the named set; may be supplied more than once.",
    )
    parser.add_argument("--png", action="store_true", help="Also render PNG charts using matplotlib.")
    args = parser.parse_args(argv)
    try:
        outputs_by_set = render_report_sets(
            args.manifest,
            render_png=args.png,
            selected_names=tuple(args.selected_names),
        )
    except (BatchReportError, OSError, ReportSetError) as exc:
        parser.error(str(exc))
    for name, outputs in outputs_by_set.items():
        print(f"Report set: {name}")
        print(f"  Summary JSON: {outputs['json']}")
        if args.png:
            print(f"  Core timeline PNG: {outputs['core_timeline_png']}")
            print(f"  Tool family distribution PNG: {outputs['tool_family_distribution_png']}")
        if "diagnostic_csv" in outputs:
            print(f"  Diagnostic observations CSV: {outputs['diagnostic_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
