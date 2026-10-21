#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare validated wide-allocation ceiling hitters with a smaller baseline.

Peak concurrency alone is a weak sizing signal.  This analysis records how
long each task occupies the allocation ceiling and whether the same task also
hit the smaller baseline ceiling, using command attribution for interpretation.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "scheduler-ceiling-hitters.v1"


class CeilingAnalysisError(RuntimeError):
    """Raised when ceiling-hitter analysis inputs are incomplete."""


@dataclass(frozen=True)
class DominantCommand:
    command_family: str
    command_preview: str
    lifecycle_wall_sec: float
    average_cores: float
    peak_cores: int
    time_at_or_above_high_core_sec: float


def _number(value: Any, *, field: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise CeilingAnalysisError(f"{path} lacks numeric field {field}")
    return float(value)


def _load_summary(path: Path) -> dict[str, dict[str, Any]]:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CeilingAnalysisError(f"unable to read summary {path}: {exc}") from exc
    tasks = payload.get("tasks")
    if not isinstance(tasks, list):
        raise CeilingAnalysisError(f"{path} has no tasks list")
    loaded: dict[str, dict[str, Any]] = {}
    for task in tasks:
        if not isinstance(task, dict) or not isinstance(task.get("task_name"), str):
            raise CeilingAnalysisError(f"{path} contains an invalid task summary")
        loaded[task["task_name"]] = task
    return loaded


def _load_commands(path: Path) -> dict[str, list[dict[str, str]]]:
    try:
        with path.open(newline="") as source:
            rows = list(csv.DictReader(source))
    except OSError as exc:
        raise CeilingAnalysisError(f"unable to read command CSV {path}: {exc}") from exc
    required = {
        "task_name",
        "command_family",
        "command_preview",
        "lifecycle_wall_sec",
        "average_simultaneous_logical_cores",
        "peak_simultaneous_logical_cores",
        "time_at_or_above_8_cores_sec",
    }
    if not rows or not required.issubset(rows[0]):
        raise CeilingAnalysisError(f"{path} lacks required command attribution columns")
    by_task: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_task.setdefault(row["task_name"], []).append(row)
    return by_task


def _distribution(task: dict[str, Any], *, path: Path) -> dict[int, float]:
    raw = task.get("time_sec_by_simultaneous_logical_cores")
    if not isinstance(raw, dict):
        raise CeilingAnalysisError(f"{path} task {task.get('task_name')} lacks distribution")
    return {int(cores): _number(seconds, field=f"distribution[{cores}]", path=path) for cores, seconds in raw.items()}


def _dominant_command(rows: list[dict[str, str]]) -> DominantCommand | None:
    if not rows:
        return None
    row = max(
        rows,
        key=lambda candidate: (
            float(candidate["time_at_or_above_8_cores_sec"]),
            float(candidate["average_simultaneous_logical_cores"]),
            float(candidate["lifecycle_wall_sec"]),
        ),
    )
    return DominantCommand(
        command_family=row["command_family"],
        command_preview=row["command_preview"],
        lifecycle_wall_sec=float(row["lifecycle_wall_sec"]),
        average_cores=float(row["average_simultaneous_logical_cores"]),
        peak_cores=int(row["peak_simultaneous_logical_cores"]),
        time_at_or_above_high_core_sec=float(row["time_at_or_above_8_cores_sec"]),
    )


def _duration_class(*, average_cores: float, wall_sec: float, at_ceiling_sec: float) -> str:
    if at_ceiling_sec < 0.01:
        return "sub-10ms ceiling touch"
    if average_cores >= 2:
        return "elevated average multi-core use"
    if at_ceiling_sec / wall_sec < 0.01:
        return "brief ceiling burst in mostly narrow run"
    return "moderate multi-core burst"


def analyze(
    wide_summary_path: Path,
    baseline_summary_path: Path,
    wide_commands_path: Path,
    baseline_commands_path: Path,
    *,
    wide_ceiling: int = 16,
    baseline_ceiling: int = 8,
) -> dict[str, Any]:
    """Compare only wide-run ceiling hitters against completed validated summaries."""
    wide_tasks = _load_summary(wide_summary_path)
    baseline_tasks = _load_summary(baseline_summary_path)
    wide_commands = _load_commands(wide_commands_path)
    baseline_commands = _load_commands(baseline_commands_path)
    rows: list[dict[str, Any]] = []
    for task_name, wide in sorted(wide_tasks.items()):
        wide_peak = int(_number(wide.get("peak_simultaneous_logical_cores"), field="peak", path=wide_summary_path))
        if wide_peak != wide_ceiling:
            continue
        wide_distribution = _distribution(wide, path=wide_summary_path)
        wide_wall = _number(wide.get("wall_sec"), field="wall_sec", path=wide_summary_path)
        wide_average = _number(
            wide.get("average_logical_cores_used"),
            field="average_logical_cores_used",
            path=wide_summary_path,
        )
        wide_at_ceiling = wide_distribution.get(wide_ceiling, 0.0)
        wide_at_or_above_baseline = sum(
            seconds for cores, seconds in wide_distribution.items() if cores >= baseline_ceiling
        )
        baseline = baseline_tasks.get(task_name)
        baseline_distribution = _distribution(baseline, path=baseline_summary_path) if baseline is not None else {}
        baseline_peak = (
            int(_number(baseline.get("peak_simultaneous_logical_cores"), field="peak", path=baseline_summary_path))
            if baseline is not None
            else None
        )
        wide_command = _dominant_command(wide_commands.get(task_name, []))
        baseline_command = _dominant_command(baseline_commands.get(task_name, []))
        rows.append(
            {
                "task_name": task_name,
                "wide_average_logical_cores_used": wide_average,
                "wide_wall_sec": wide_wall,
                "wide_peak_simultaneous_logical_cores": wide_peak,
                "wide_time_at_ceiling_sec": wide_at_ceiling,
                "wide_fraction_at_ceiling": wide_at_ceiling / wide_wall if wide_wall else 0.0,
                "wide_time_at_or_above_8_cores_sec": wide_at_or_above_baseline,
                "wide_peak_duration_class": _duration_class(
                    average_cores=wide_average,
                    wall_sec=wide_wall,
                    at_ceiling_sec=wide_at_ceiling,
                ),
                "baseline_report_available": baseline is not None,
                "baseline_average_logical_cores_used": (
                    _number(
                        baseline.get("average_logical_cores_used"),
                        field="average_logical_cores_used",
                        path=baseline_summary_path,
                    )
                    if baseline is not None
                    else None
                ),
                "baseline_peak_simultaneous_logical_cores": baseline_peak,
                "baseline_time_at_ceiling_sec": (
                    baseline_distribution.get(baseline_ceiling, 0.0) if baseline is not None else None
                ),
                "baseline_also_reached_ceiling": (
                    baseline_peak == baseline_ceiling if baseline_peak is not None else None
                ),
                "wide_dominant_high_core_command_family": (wide_command.command_family if wide_command else ""),
                "wide_dominant_high_core_command_preview": (wide_command.command_preview if wide_command else ""),
                "wide_dominant_high_core_command_time_at_or_above_8_sec": (
                    wide_command.time_at_or_above_high_core_sec if wide_command else None
                ),
                "wide_dominant_high_core_command_peak": (wide_command.peak_cores if wide_command else None),
                "baseline_dominant_high_core_command_family": (
                    baseline_command.command_family if baseline_command else ""
                ),
                "same_dominant_command_family_as_baseline": (
                    wide_command.command_family == baseline_command.command_family
                    if wide_command and baseline_command
                    else None
                ),
            }
        )
    with_baseline = [row for row in rows if row["baseline_report_available"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "wide_summary": str(wide_summary_path.resolve()),
        "baseline_summary": str(baseline_summary_path.resolve()),
        "wide_ceiling": wide_ceiling,
        "baseline_ceiling": baseline_ceiling,
        "summary": {
            "wide_ceiling_hitter_count": len(rows),
            "with_comparable_baseline_count": len(with_baseline),
            "also_hit_baseline_ceiling_count": sum(
                row["baseline_also_reached_ceiling"] is True for row in with_baseline
            ),
            "same_dominant_command_family_count": sum(
                row["same_dominant_command_family_as_baseline"] is True for row in with_baseline
            ),
        },
        "tasks": rows,
    }


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ceiling_hitters_comparison.json"
    csv_path = output_dir / "ceiling_hitters_comparison.csv"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    rows = analysis["tasks"]
    fields = list(rows[0]) if rows else ["task_name"]
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    return {"json": json_path, "csv": csv_path}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare wide-allocation CPU ceiling hitters.")
    parser.add_argument("--wide-summary", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--wide-commands", type=Path, required=True)
    parser.add_argument("--baseline-commands", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wide-ceiling", type=int, default=16)
    parser.add_argument("--baseline-ceiling", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        analysis = analyze(
            args.wide_summary,
            args.baseline_summary,
            args.wide_commands,
            args.baseline_commands,
            wide_ceiling=args.wide_ceiling,
            baseline_ceiling=args.baseline_ceiling,
        )
        outputs = write_outputs(analysis, args.output_dir)
    except (CeilingAnalysisError, OSError, ValueError) as exc:
        parser.error(str(exc))
    summary = analysis["summary"]
    print(f"Wide-allocation ceiling hitters: {summary['wide_ceiling_hitter_count']}")
    print(f"With comparable baseline: {summary['with_comparable_baseline_count']}")
    print(f"Also reached baseline ceiling: {summary['also_hit_baseline_ceiling_count']}")
    print("Same dominant high-core command family as baseline: " f"{summary['same_dominant_command_family_count']}")
    print(f"JSON: {outputs['json']}")
    print(f"CSV: {outputs['csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
