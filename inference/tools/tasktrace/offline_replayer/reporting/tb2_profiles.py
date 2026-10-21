#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Analyze Terminal-Bench profile coverage once per benchmark task.

The task inventory comes from source trajectories, not from whichever reports
happen to exist. This makes missing or incomplete profiles visible and keeps
reruns from inflating task counts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "terminal-bench-profile-analysis.v1"
DEFAULT_REPORT_GLOB = "**/pipeline_report.json"
DEFAULT_STARTED_GLOB = "**/container_profile"


class AnalysisError(RuntimeError):
    """Raised when Terminal-Bench profile artifacts cannot be analyzed."""


@dataclass(frozen=True)
class ProfileObservation:
    task_name: str
    report_path: Path
    wall_sec: float
    cpu_usage_sec: float
    average_cores: float
    allocated_cpus: float
    mismatches: int
    unavailable_comparisons: int
    replay_valid: bool
    timed_out_actions: int
    scheduler_valid: bool
    scheduler_peak: int | None
    lost_perf_events: int
    scheduler_anomalies: int
    distribution: dict[int, float]

    @property
    def passed(self) -> bool:
        return (
            self.mismatches == 0
            and self.unavailable_comparisons == 0
            and self.timed_out_actions == 0
            and self.replay_valid
            and self.scheduler_valid
        )

    @property
    def failure_reasons(self) -> list[str]:
        reasons: list[str] = []
        if self.mismatches:
            reasons.append("source_outcome_mismatch")
        if self.unavailable_comparisons:
            reasons.append("source_outcome_comparison_unavailable")
        if self.timed_out_actions:
            reasons.append("timed_out_action")
        if not self.replay_valid:
            reasons.append("incomplete_replay")
        if not self.scheduler_valid:
            reasons.append("scheduler_validation_failed")
        return reasons


def _number(value: Any, *, field: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise AnalysisError(f"{path} lacks numeric field {field}")
    return float(value)


def load_inventory(trajectory_root: Path) -> set[str]:
    tasks = {path.parent.parent.name.split("__", 1)[0] for path in trajectory_root.glob("*/agent/trajectory.json")}
    if not tasks:
        raise AnalysisError(f"no Terminal-Bench trajectories found under {trajectory_root}")
    return tasks


def load_observation(path: Path, inventory: set[str]) -> ProfileObservation | None:
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AnalysisError(f"unable to read pipeline report {path}: {exc}") from exc
    task_name = str(payload.get("task_name") or "")
    if task_name not in inventory:
        return None
    replay = payload.get("replay_summary")
    cpu = payload.get("container_cpu")
    scheduler = payload.get("scheduler_concurrency")
    if not isinstance(replay, dict) or not isinstance(cpu, dict) or not isinstance(scheduler, dict):
        raise AnalysisError(f"pipeline report lacks replay, CPU, or scheduler data: {path}")
    validation = scheduler.get("cgroup_validation") or {}
    raw_distribution = scheduler.get("time_sec_by_simultaneous_logical_cores") or {}
    distribution: dict[int, float] = {}
    if isinstance(raw_distribution, dict):
        for cores, seconds in raw_distribution.items():
            distribution[int(cores)] = _number(seconds, field=f"scheduler distribution[{cores}]", path=path)
    peak = scheduler.get("peak_simultaneous_logical_cores")
    unavailable = replay.get("source_outcome_comparison_unavailable_count", 0)
    if type(unavailable) is not int or unavailable < 0:
        raise AnalysisError(
            f"{path} has invalid unsigned integer field replay_summary.source_outcome_comparison_unavailable_count"
        )
    return ProfileObservation(
        task_name=task_name,
        report_path=path.resolve(),
        wall_sec=_number(cpu.get("wall_sec"), field="container_cpu.wall_sec", path=path),
        cpu_usage_sec=_number(cpu.get("usage_sec"), field="container_cpu.usage_sec", path=path),
        average_cores=_number(
            cpu.get("average_logical_cores_used"),
            field="container_cpu.average_logical_cores_used",
            path=path,
        ),
        allocated_cpus=_number(
            cpu.get("allocated_logical_cpus"),
            field="container_cpu.allocated_logical_cpus",
            path=path,
        ),
        mismatches=int(replay.get("source_outcome_mismatch_count") or 0),
        unavailable_comparisons=unavailable,
        replay_valid=(
            payload.get("status", "valid") == "valid"
            and all(
                field not in replay or (type(replay[field]) is int and replay[field] == 0)
                for field in (
                    "actions_launch_failed",
                    "actions_source_argument_parse_failed",
                    "actions_skipped_after_timeout",
                )
            )
        ),
        timed_out_actions=int(replay.get("actions_timed_out") or 0),
        scheduler_valid=bool(validation.get("valid_for_concurrency_reporting")),
        scheduler_peak=int(peak) if isinstance(peak, (int, float)) else None,
        lost_perf_events=int(scheduler.get("lost_perf_events") or 0),
        scheduler_anomalies=len(scheduler.get("anomalies") or []),
        distribution=distribution,
    )


def _task_from_started_dir(path: Path, inventory: set[str]) -> str | None:
    dirname = path.parent.name
    candidates = [task for task in inventory if dirname == task or dirname.startswith(f"{task}-")]
    return max(candidates, key=len) if candidates else None


def _choose_observation(observations: list[ProfileObservation]) -> ProfileObservation:
    """Prefer a faithful completed replay; otherwise retain the newest diagnostic."""
    passed = [observation for observation in observations if observation.passed]
    candidates = passed or observations
    return max(candidates, key=lambda observation: observation.report_path.stat().st_mtime_ns)


def analyze(
    trajectory_root: Path,
    reports_root: Path,
    *,
    include_failed: bool,
    report_glob: str = DEFAULT_REPORT_GLOB,
    started_glob: str = DEFAULT_STARTED_GLOB,
) -> dict[str, Any]:
    """Classify task coverage and compute explicitly gated aggregate metrics.

    Failed-validation windows may be requested for diagnostic cgroup-average
    analysis. Their scheduler distribution contributes only when its own
    validation is sound, and no failed replay is relabeled as comparable.
    """
    inventory = load_inventory(trajectory_root)
    observations_by_task: dict[str, list[ProfileObservation]] = defaultdict(list)
    for path in reports_root.glob(report_glob):
        observation = load_observation(path, inventory)
        if observation is not None:
            observations_by_task[observation.task_name].append(observation)
    started_tasks = {
        task
        for path in reports_root.glob(started_glob)
        if (task := _task_from_started_dir(path, inventory)) is not None
    }

    selected_by_task = {task: _choose_observation(observations) for task, observations in observations_by_task.items()}
    task_rows: list[dict[str, Any]] = []
    exact_distribution: dict[int, float] = defaultdict(float)
    included_observations: list[ProfileObservation] = []
    exact_distribution_tasks = 0
    for task in sorted(inventory):
        observation = selected_by_task.get(task)
        if observation is None:
            status = "incomplete" if task in started_tasks else "not_profiled"
            task_rows.append(
                {
                    "task_name": task,
                    "status": status,
                    "attempt_count": 0,
                    "included_in_metrics": False,
                    "included_in_exact_concurrency_distribution": False,
                }
            )
            continue
        status = "passed" if observation.passed else "failed_validation"
        included = observation.passed or include_failed
        include_distribution = included and observation.scheduler_valid
        if included:
            included_observations.append(observation)
        if include_distribution:
            exact_distribution_tasks += 1
            for cores, seconds in observation.distribution.items():
                exact_distribution[cores] += seconds
        if observation.scheduler_valid:
            distribution_quality = "validated_replay" if observation.passed else "exact_observed_window_non_comparable"
        else:
            distribution_quality = "observed_peak_lower_bound_only"
        task_rows.append(
            {
                "task_name": task,
                "status": status,
                "attempt_count": len(observations_by_task[task]),
                "pipeline_report": str(observation.report_path),
                "failure_reasons": observation.failure_reasons,
                "timed_out_actions": observation.timed_out_actions,
                "source_outcome_mismatch_count": observation.mismatches,
                "source_outcome_comparison_unavailable_count": observation.unavailable_comparisons,
                "wall_sec": observation.wall_sec,
                "cpu_usage_sec": observation.cpu_usage_sec,
                "average_logical_cores_used": observation.average_cores,
                "allocated_logical_cpus": observation.allocated_cpus,
                "scheduler_peak_simultaneous_logical_cores": observation.scheduler_peak,
                "scheduler_lost_perf_events": observation.lost_perf_events,
                "scheduler_anomaly_count": observation.scheduler_anomalies,
                "scheduler_valid_for_concurrency_reporting": observation.scheduler_valid,
                "distribution_quality": distribution_quality,
                "included_in_metrics": included,
                "included_in_exact_concurrency_distribution": include_distribution,
            }
        )

    counts = {
        status: sum(row["status"] == status for row in task_rows)
        for status in ("passed", "failed_validation", "incomplete", "not_profiled")
    }
    counts["total_terminal_bench_tasks"] = len(inventory)
    counts["profiled_with_completed_report"] = counts["passed"] + counts["failed_validation"]
    counts["failed_validation_with_timed_out_actions"] = sum(
        row.get("timed_out_actions", 0) > 0 and row["status"] == "failed_validation" for row in task_rows
    )
    total_wall = sum(observation.wall_sec for observation in included_observations)
    total_cpu = sum(observation.cpu_usage_sec for observation in included_observations)
    distribution_total = sum(exact_distribution.values())
    distribution_rows = [
        {
            "simultaneous_logical_cores": cores,
            "observed_time_sec": seconds,
            "fraction_of_exact_distribution_window": seconds / distribution_total if distribution_total else 0,
        }
        for cores, seconds in sorted(exact_distribution.items())
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectory_root": str(trajectory_root.resolve()),
        "reports_root": str(reports_root.resolve()),
        "report_glob": report_glob,
        "include_failed": include_failed,
        "selection_rule": (
            "One row per Terminal-Bench task; prefer any passed report, otherwise "
            "select the most recent completed failed-validation report."
        ),
        "counts": counts,
        "metrics": {
            "included_task_count": len(included_observations),
            "observed_window_total_wall_sec": total_wall,
            "observed_window_total_cpu_usage_sec": total_cpu,
            "observed_window_weighted_average_logical_cores_used": (total_cpu / total_wall if total_wall else 0),
            "exact_concurrency_distribution_task_count": exact_distribution_tasks,
            "exact_concurrency_distribution_excludes_invalid_scheduler_traces": True,
        },
        "concurrency_distribution": distribution_rows,
        "tasks": task_rows,
    }


def write_outputs(analysis: dict[str, Any], output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "tb2_profile_analysis.json"
    task_csv_path = output_dir / "tb2_task_status.csv"
    distribution_csv_path = output_dir / "tb2_concurrency_distribution.csv"
    json_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    fieldnames = sorted({key for row in analysis["tasks"] for key in row})
    with task_csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(analysis["tasks"])
    distribution_fields = [
        "simultaneous_logical_cores",
        "observed_time_sec",
        "fraction_of_exact_distribution_window",
    ]
    with distribution_csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=distribution_fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(analysis["concurrency_distribution"])
    return {
        "json": json_path,
        "task_csv": task_csv_path,
        "distribution_csv": distribution_csv_path,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Analyze CPU profile coverage once per Terminal-Bench task.")
    parser.add_argument("--trajectory-root", required=True, type=Path)
    parser.add_argument("--reports-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include completed failed-validation task windows in observed CPU metrics.",
    )
    parser.add_argument(
        "--report-glob",
        default=DEFAULT_REPORT_GLOB,
        help="Pipeline-report glob relative to --reports-root.",
    )
    parser.add_argument(
        "--started-glob",
        default=DEFAULT_STARTED_GLOB,
        help="Started-profile directory glob relative to --reports-root.",
    )
    args = parser.parse_args(argv)
    try:
        analysis = analyze(
            args.trajectory_root,
            args.reports_root,
            include_failed=args.include_failed,
            report_glob=args.report_glob,
            started_glob=args.started_glob,
        )
        outputs = write_outputs(analysis, args.output_dir)
    except (AnalysisError, OSError) as exc:
        parser.error(str(exc))
    counts = analysis["counts"]
    metrics = analysis["metrics"]
    print(f"Terminal-Bench tasks: {counts['total_terminal_bench_tasks']}")
    print(
        "Coverage: "
        f"{counts['passed']} passed, "
        f"{counts['failed_validation']} failed validation "
        f"({counts['failed_validation_with_timed_out_actions']} with timed-out actions), "
        f"{counts['incomplete']} incomplete, "
        f"{counts['not_profiled']} not profiled"
    )
    print(f"Metrics mode: {'passed plus failed-validation windows' if args.include_failed else 'passed only'}")
    print(
        "Observed-window weighted average logical cores: "
        f"{metrics['observed_window_weighted_average_logical_cores_used']:.6f} "
        f"across {metrics['included_task_count']} tasks"
    )
    print(
        "Exact concurrency distributions included for: " f"{metrics['exact_concurrency_distribution_task_count']} tasks"
    )
    print(f"JSON: {outputs['json']}")
    print(f"Task CSV: {outputs['task_csv']}")
    print(f"Concurrency CSV: {outputs['distribution_csv']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
