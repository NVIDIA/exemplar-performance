# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.reporting.tb2_profiles import analyze, write_outputs


def _trajectory(root: Path, task: str) -> None:
    path = root / f"{task}__trial" / "agent" / "trajectory.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}")


def _report(
    root: Path,
    dirname: str,
    task: str,
    *,
    mismatches: int = 0,
    unavailable: int = 0,
    timed_out: int = 0,
    scheduler_valid: bool = True,
) -> Path:
    path = root / "campaign-run" / dirname / "pipeline_report.json"
    path.parent.mkdir(parents=True)
    path.write_text(
        json.dumps(
            {
                "task_name": task,
                "replay_summary": {
                    "source_outcome_mismatch_count": mismatches,
                    "source_outcome_comparison_unavailable_count": unavailable,
                    "actions_timed_out": timed_out,
                },
                "container_cpu": {
                    "wall_sec": 2.0,
                    "usage_sec": 4.0,
                    "average_logical_cores_used": 2.0,
                    "allocated_logical_cpus": 8,
                },
                "scheduler_concurrency": {
                    "peak_simultaneous_logical_cores": 4,
                    "lost_perf_events": 0 if scheduler_valid else 2,
                    "anomalies": [] if scheduler_valid else ["drop"],
                    "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "3": 1.0},
                    "cgroup_validation": {"valid_for_concurrency_reporting": scheduler_valid},
                },
            }
        )
    )
    return path


def test_analysis_counts_each_terminal_bench_task_once_and_prefers_pass(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "trajectories"
    reports = tmp_path / "reports"
    for task in ("rerun-task", "failed-task", "unseen-task", "incomplete-task"):
        _trajectory(trajectories, task)
    _report(reports, "rerun-task-old", "rerun-task", mismatches=1)
    _report(reports, "rerun-task-valid", "rerun-task")
    _report(reports, "failed-task", "failed-task", timed_out=1, scheduler_valid=False)
    (reports / "campaign-run" / "incomplete-task" / "container_profile").mkdir(parents=True)

    analysis = analyze(trajectories, reports, include_failed=False)

    assert analysis["counts"] == {
        "passed": 1,
        "failed_validation": 1,
        "incomplete": 1,
        "not_profiled": 1,
        "total_terminal_bench_tasks": 4,
        "profiled_with_completed_report": 2,
        "failed_validation_with_timed_out_actions": 1,
    }
    assert analysis["metrics"]["included_task_count"] == 1
    row = next(row for row in analysis["tasks"] if row["task_name"] == "rerun-task")
    assert row["status"] == "passed"
    assert row["attempt_count"] == 2


def test_include_failed_uses_cgroup_average_but_only_valid_scheduler_distribution(
    tmp_path: Path,
) -> None:
    trajectories = tmp_path / "trajectories"
    reports = tmp_path / "reports"
    for task in ("passed", "drifted-valid-scheduler", "drifted-invalid-scheduler"):
        _trajectory(trajectories, task)
    _report(reports, "passed", "passed")
    _report(reports, "drifted-valid", "drifted-valid-scheduler", mismatches=1)
    _report(
        reports,
        "drifted-invalid",
        "drifted-invalid-scheduler",
        mismatches=1,
        scheduler_valid=False,
    )

    analysis = analyze(trajectories, reports, include_failed=True)

    assert analysis["metrics"]["included_task_count"] == 3
    assert analysis["metrics"]["observed_window_weighted_average_logical_cores_used"] == pytest.approx(2.0)
    assert analysis["metrics"]["exact_concurrency_distribution_task_count"] == 2
    assert sum(row["observed_time_sec"] for row in analysis["concurrency_distribution"]) == pytest.approx(4.0)

    outputs = write_outputs(analysis, tmp_path / "out")
    assert outputs["json"].exists()
    assert outputs["task_csv"].exists()
    assert outputs["distribution_csv"].exists()
    assert b"\r" not in outputs["task_csv"].read_bytes()
    assert b"\r" not in outputs["distribution_csv"].read_bytes()


def test_analysis_keeps_unavailable_outcomes_distinct_from_mismatches(tmp_path: Path) -> None:
    trajectories = tmp_path / "trajectories"
    reports = tmp_path / "reports"
    _trajectory(trajectories, "unavailable")
    _report(reports, "unavailable", "unavailable", unavailable=1)

    row = analyze(trajectories, reports, include_failed=False)["tasks"][0]

    assert row["failure_reasons"] == ["source_outcome_comparison_unavailable"]
    assert row["source_outcome_comparison_unavailable_count"] == 1


@pytest.mark.parametrize(
    "status, field",
    [("invalid", None), ("valid", "actions_launch_failed"), ("valid", "actions_source_argument_parse_failed")],
)
def test_analysis_rejects_incomplete_replay(tmp_path: Path, status: str, field: str | None) -> None:
    trajectories = tmp_path / "trajectories"
    reports = tmp_path / "reports"
    _trajectory(trajectories, "incomplete")
    path = _report(reports, "incomplete", "incomplete")
    payload = json.loads(path.read_text())
    payload["status"] = status
    if field:
        payload["replay_summary"][field] = 1
    path.write_text(json.dumps(payload))
    row = analyze(trajectories, reports, include_failed=False)["tasks"][0]
    assert "incomplete_replay" in row["failure_reasons"]
