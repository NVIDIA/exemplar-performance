#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Calculate aggregate average CPU-core use for a SWE-bench Pro replay campaign."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "swebench-average-cpu-cores.v1"


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text())
    except FileNotFoundError as exc:
        raise ValueError(f"required file does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path}: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root is not an object: {path}")
    return payload


def _require_number(value: Any, *, field: str, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} is not numeric: {value!r}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} is not finite: {value!r}")
    if positive and result <= 0:
        raise ValueError(f"{field} must be positive: {value!r}")
    return result


def _assert_close(actual: float, expected: float, *, field: str, tolerance: float) -> None:
    if not math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance):
        raise ValueError(f"{field} is inconsistent: reported={actual:.12f}, " f"recomputed={expected:.12f}")


def _aggregate(rows: list[dict[str, Any]], logical_cores: int) -> dict[str, Any]:
    total_cpu = math.fsum(row["cpu_sec"] for row in rows)
    total_wall = math.fsum(row["wall_sec"] for row in rows)
    weighted_average = total_cpu / total_wall if total_wall else None
    task_averages = [row["average_cores"] for row in rows]
    return {
        "task_count": len(rows),
        "total_cpu_sec": round(total_cpu, 9),
        "total_wall_sec": round(total_wall, 9),
        "weighted_average_cores": (round(weighted_average, 9) if weighted_average is not None else None),
        "weighted_utilization_percent_of_allocation": (
            round(weighted_average / logical_cores * 100.0, 6) if weighted_average is not None else None
        ),
        "unweighted_mean_task_average_cores": (round(statistics.fmean(task_averages), 9) if rows else None),
        "p50_task_average_cores": (round(statistics.median(task_averages), 9) if rows else None),
    }


def _load_task(
    *,
    task: dict[str, Any],
    output_root: Path,
    allocation: dict[str, Any],
    repetition: int,
) -> dict[str, Any]:
    task_name = task.get("name")
    if not isinstance(task_name, str) or not task_name:
        raise ValueError(f"manifest task has no valid name: {task!r}")
    allocation_name = allocation.get("name")
    if not isinstance(allocation_name, str) or not allocation_name:
        raise ValueError("manifest allocation has no valid name")
    logical_cores = int(allocation["logical_cores"])
    run_dir = output_root / allocation_name / task_name / f"rep-{repetition:02d}"
    report_path = run_dir / "pipeline_report.json"
    status_path = run_dir / "sweep_run_status.json"
    report = _read_json(report_path)
    status = _read_json(status_path)

    if report.get("task_name") != task_name:
        raise ValueError(f"report task mismatch in {report_path}: {report.get('task_name')!r}")
    if status.get("task_name") != task_name:
        raise ValueError(f"status task mismatch in {status_path}: {status.get('task_name')!r}")
    if status.get("allocation_name") != allocation_name:
        raise ValueError(f"allocation mismatch in {status_path}")
    if int(status.get("logical_cores", -1)) != logical_cores:
        raise ValueError(f"logical-core mismatch in {status_path}")
    if int(status.get("repetition", -1)) != repetition:
        raise ValueError(f"repetition mismatch in {status_path}")
    declared_report = status.get("pipeline_report")
    if not isinstance(declared_report, str) or Path(declared_report).resolve() != report_path.resolve():
        raise ValueError(f"pipeline-report path mismatch in {status_path}")

    cpu = report.get("container_cpu")
    if not isinstance(cpu, dict):
        raise ValueError(f"missing container_cpu object in {report_path}")
    if cpu.get("method") != "container-cgroup-v2-cpu.stat":
        raise ValueError(f"unexpected CPU measurement method in {report_path}")
    allocated = _require_number(cpu.get("allocated_logical_cpus"), field=f"{report_path}: allocated_logical_cpus")
    _assert_close(
        allocated,
        float(logical_cores),
        field=f"{report_path}: allocated_logical_cpus",
        tolerance=1e-12,
    )
    cpu_sec = _require_number(cpu.get("usage_sec"), field=f"{report_path}: usage_sec", positive=True)
    wall_sec = _require_number(cpu.get("wall_sec"), field=f"{report_path}: wall_sec", positive=True)
    average_cores = cpu_sec / wall_sec
    reported_average = _require_number(
        cpu.get("average_logical_cores_used"),
        field=f"{report_path}: average_logical_cores_used",
    )
    _assert_close(
        reported_average,
        average_cores,
        field=f"{report_path}: average_logical_cores_used",
        tolerance=5e-9,
    )
    reported_utilization = _require_number(
        cpu.get("utilization_percent_of_allocation"),
        field=f"{report_path}: utilization_percent_of_allocation",
    )
    _assert_close(
        reported_utilization,
        average_cores / logical_cores * 100.0,
        field=f"{report_path}: utilization_percent_of_allocation",
        tolerance=5e-6,
    )
    user_sec = _require_number(cpu.get("user_sec"), field=f"{report_path}: user_sec")
    system_sec = _require_number(cpu.get("system_sec"), field=f"{report_path}: system_sec")
    _assert_close(
        user_sec + system_sec,
        cpu_sec,
        field=f"{report_path}: user_sec + system_sec",
        tolerance=5e-6,
    )

    replay = report.get("replay_summary")
    scheduler = report.get("scheduler_concurrency")
    image_identity = report.get("image_identity")
    if not isinstance(replay, dict) or not isinstance(scheduler, dict):
        raise ValueError(f"missing replay/scheduler validation data in {report_path}")
    if not isinstance(image_identity, dict):
        raise ValueError(f"missing image identity in {report_path}")
    cgroup_validation = scheduler.get("cgroup_validation")
    if not isinstance(cgroup_validation, dict):
        raise ValueError(f"missing scheduler cgroup validation in {report_path}")

    expected_image_id = task.get("expected_image_id")
    image_valid = isinstance(expected_image_id, str) and image_identity.get("id") == expected_image_id
    scheduler_valid = bool(cgroup_validation.get("valid_for_concurrency_reporting"))
    timeout = int(replay.get("actions_timed_out", 0)) > 0
    mismatch_count = int(replay.get("source_outcome_mismatch_count", 0))
    fidelity_valid = scheduler_valid and image_valid and not timeout and mismatch_count == 0
    sensitivity_valid = scheduler_valid and image_valid and not timeout

    run_status = status.get("status")
    expected_status = "passed" if fidelity_valid else "failed_validation"
    if run_status != expected_status:
        raise ValueError(
            f"status/cohort disagreement for {task_name}: " f"status={run_status!r}, recomputed={expected_status!r}"
        )

    return {
        "task_name": task_name,
        "report": str(report_path),
        "status": run_status,
        "cpu_sec": cpu_sec,
        "wall_sec": wall_sec,
        "average_cores": average_cores,
        "fidelity_valid": fidelity_valid,
        "complete_scheduler_valid_non_timeout": sensitivity_valid,
        "image_valid": image_valid,
        "scheduler_valid": scheduler_valid,
        "timeout": timeout,
        "source_outcome_mismatch_count": mismatch_count,
    }


def summarize_campaign(manifest_path: Path) -> dict[str, Any]:
    manifest = _read_json(manifest_path)
    population = manifest.get("population")
    if not isinstance(population, dict) or population.get("benchmark") != "SWE-bench Pro":
        raise ValueError("manifest is not identified as a SWE-bench Pro campaign")
    allocations = manifest.get("allocations")
    if not isinstance(allocations, list) or len(allocations) != 1:
        raise ValueError("calculator requires exactly one allocation")
    if manifest.get("repetitions") != 1:
        raise ValueError("calculator requires exactly one repetition")
    tasks = manifest.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("manifest has no tasks")
    names = [task.get("name") for task in tasks if isinstance(task, dict)]
    if len(names) != len(tasks) or len(set(names)) != len(names):
        raise ValueError("manifest task names are missing or duplicated")

    allocation = allocations[0]
    logical_cores = int(allocation["logical_cores"])
    if logical_cores <= 0:
        raise ValueError("logical_cores must be positive")
    output_root = Path(manifest["output_root"])
    rows = [
        _load_task(
            task=task,
            output_root=output_root,
            allocation=allocation,
            repetition=1,
        )
        for task in tasks
    ]
    fidelity_rows = [row for row in rows if row["fidelity_valid"]]
    sensitivity_rows = [row for row in rows if row["complete_scheduler_valid_non_timeout"]]

    return {
        "schema_version": SCHEMA_VERSION,
        "manifest": str(manifest_path.resolve()),
        "experiment_name": manifest.get("name"),
        "formula": "sum(container cgroup usage_sec) / sum(container measured wall_sec)",
        "measurement": {
            "benchmark": "SWE-bench Pro",
            "method": "container-cgroup-v2-cpu.stat",
            "logical_cores": logical_cores,
            "cpuset_cpus": allocation.get("cpuset_cpus"),
            "execution_order": "sequential",
        },
        "validation": {
            "planned_tasks": len(tasks),
            "reports_validated": len(rows),
            "fidelity_valid_tasks": len(fidelity_rows),
            "diagnostic_tasks": len(rows) - len(fidelity_rows),
            "scheduler_valid_non_timeout_tasks": len(sensitivity_rows),
            "all_task_measurement_formulas_recomputed": True,
            "all_statuses_recomputed": True,
        },
        "cohorts": {
            "all_reports_diagnostic": _aggregate(rows, logical_cores),
            "fidelity_valid": _aggregate(fidelity_rows, logical_cores),
            "complete_scheduler_valid_non_timeout_sensitivity": _aggregate(sensitivity_rows, logical_cores),
        },
        "tasks": rows,
    }


def _print_summary(summary: dict[str, Any]) -> None:
    print("SWE-bench Pro aggregate average CPU cores")
    print(f"Formula: {summary['formula']}")
    print(f"Manifest: {summary['manifest']}")
    print()
    for name, cohort in summary["cohorts"].items():
        print(name)
        print(f"  tasks: {cohort['task_count']}")
        print(f"  total CPU seconds: {cohort['total_cpu_sec']:.9f}")
        print(f"  total wall seconds: {cohort['total_wall_sec']:.9f}")
        print(f"  weighted average cores: {cohort['weighted_average_cores']:.9f}")
        print("  weighted allocation utilization: " f"{cohort['weighted_utilization_percent_of_allocation']:.6f}%")
        print(f"  unweighted task mean: {cohort['unweighted_mean_task_average_cores']:.9f}")
        print(f"  p50 task average cores: {cohort['p50_task_average_cores']:.9f}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="SWE-bench Pro core-sweep manifest")
    parser.add_argument("--json-output", type=Path, help="Write full task-level audit JSON")
    args = parser.parse_args()

    try:
        summary = summarize_campaign(args.manifest)
    except (KeyError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    _print_summary(summary)
    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
