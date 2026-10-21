# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from offline_replayer.reporting.task_catalog import build_catalog


def _pipeline_report(path: Path, trajectory: Path) -> Path:
    path.write_text(json.dumps({"source_trajectory": str(trajectory)}))
    return path


def _trajectory(path: Path, instruction: str) -> Path:
    path.write_text(
        json.dumps(
            {
                "steps": [
                    {"source": "user", "message": "<environment_context>ignored</environment_context>"},
                    {"source": "user", "message": instruction},
                ]
            }
        )
    )
    return path


def test_build_catalog_includes_prompts_and_separates_validated_metrics(tmp_path: Path) -> None:
    numeric_trajectory = _trajectory(
        tmp_path / "numeric_trajectory.json",
        "Find the dominant eigenvalue using numpy and validate the numeric answer.",
    )
    failed_trajectory = _trajectory(
        tmp_path / "failed_trajectory.json",
        "Compile a large C project and run its test suite.",
    )
    numeric_report = _pipeline_report(tmp_path / "numeric_report.json", numeric_trajectory)
    failed_report = _pipeline_report(tmp_path / "failed_report.json", failed_trajectory)
    analysis = tmp_path / "analysis.json"
    analysis.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "largest-eigenval",
                        "status": "passed",
                        "distribution_quality": "validated_replay",
                        "pipeline_report": str(numeric_report),
                        "allocated_logical_cpus": 16,
                        "wall_sec": 2.0,
                        "cpu_usage_sec": 3.0,
                        "average_logical_cores_used": 1.5,
                        "scheduler_peak_simultaneous_logical_cores": 16,
                        "included_in_exact_concurrency_distribution": True,
                        "failure_reasons": [],
                    },
                    {
                        "task_name": "compile-compcert",
                        "status": "failed_validation",
                        "distribution_quality": "observed_peak_lower_bound_only",
                        "pipeline_report": str(failed_report),
                        "allocated_logical_cpus": 16,
                        "wall_sec": 30.0,
                        "cpu_usage_sec": 10.0,
                        "average_logical_cores_used": 0.3,
                        "scheduler_peak_simultaneous_logical_cores": 8,
                        "included_in_exact_concurrency_distribution": False,
                        "failure_reasons": ["scheduler_validation_failed"],
                    },
                ]
            }
        )
    )
    validated = tmp_path / "validated.json"
    validated.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "task_name": "largest-eigenval",
                        "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "16": 1.0},
                    }
                ]
            }
        )
    )
    commands = tmp_path / "commands.csv"
    with commands.open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "task_name",
                "command_family",
                "command_preview",
                "lifecycle_wall_sec",
                "average_simultaneous_logical_cores",
                "peak_simultaneous_logical_cores",
                "time_at_or_above_4_cores_sec",
                "time_at_or_above_8_cores_sec",
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "task_name": "largest-eigenval",
                "command_family": "python eval.py",
                "command_preview": "python /app/eval.py",
                "lifecycle_wall_sec": 2.0,
                "average_simultaneous_logical_cores": 3.0,
                "peak_simultaneous_logical_cores": 16,
                "time_at_or_above_4_cores_sec": 1.0,
                "time_at_or_above_8_cores_sec": 0.5,
            }
        )

    catalog = build_catalog(analysis, validated, commands)

    assert catalog["summary"] == {
        "task_count": 2,
        "validated_task_count": 1,
        "diagnostic_only_task_count": 1,
        "tasks_with_command_attribution": 1,
        "notes": catalog["summary"]["notes"],
    }
    by_name = {task["task_name"]: task for task in catalog["tasks"]}
    numeric = by_name["largest-eigenval"]
    assert "dominant eigenvalue" in numeric["task"]["instruction_excerpt"]
    assert numeric["task"]["what_it_does"] == numeric["task"]["instruction_excerpt"]
    assert numeric["cpu"]["burst_class"] == "measured high-core burst"
    assert numeric["explanation"]["method"] == "reviewed_from_prompt_and_command_evidence"
    failed = by_name["compile-compcert"]
    assert failed["cpu"]["exact_concurrency_valid"] is False
    assert failed["cpu"]["burst_class"] == "diagnostic-only observation"
    assert "diagnostic only" in failed["explanation"]["why_it_uses_cores"]

    assert "did not spend material time" not in failed["explanation"]["why_it_uses_cores_mechanism_only"]
    assert "CPU work comes from" in numeric["explanation"]["why_it_uses_cores_mechanism_only"]
