# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parents[2] / "scripts" / "calculate_swebench_average_cores.py"
SPEC = importlib.util.spec_from_file_location("calculate_swebench_average_cores", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
summarize_campaign = MODULE.summarize_campaign


def _write_run(
    root: Path,
    *,
    name: str,
    cpu_sec: float,
    wall_sec: float,
    mismatch_count: int = 0,
    timeout_count: int = 0,
    scheduler_valid: bool = True,
) -> dict[str, object]:
    run_dir = root / "eight-core" / name / "rep-01"
    run_dir.mkdir(parents=True)
    image_id = f"sha256:{name}"
    average = cpu_sec / wall_sec
    fidelity_valid = scheduler_valid and timeout_count == 0 and mismatch_count == 0
    report_path = run_dir / "pipeline_report.json"
    report_path.write_text(
        json.dumps(
            {
                "task_name": name,
                "container_cpu": {
                    "method": "container-cgroup-v2-cpu.stat",
                    "allocated_logical_cpus": 8,
                    "usage_sec": cpu_sec,
                    "wall_sec": wall_sec,
                    "average_logical_cores_used": average,
                    "utilization_percent_of_allocation": average / 8 * 100,
                    "user_sec": cpu_sec * 0.75,
                    "system_sec": cpu_sec * 0.25,
                },
                "replay_summary": {
                    "actions_timed_out": timeout_count,
                    "source_outcome_mismatch_count": mismatch_count,
                },
                "scheduler_concurrency": {"cgroup_validation": {"valid_for_concurrency_reporting": scheduler_valid}},
                "image_identity": {"id": image_id},
            }
        )
    )
    (run_dir / "sweep_run_status.json").write_text(
        json.dumps(
            {
                "allocation_name": "eight-core",
                "logical_cores": 8,
                "pipeline_report": str(report_path),
                "repetition": 1,
                "status": "passed" if fidelity_valid else "failed_validation",
                "task_name": name,
            }
        )
    )
    return {"name": name, "expected_image_id": image_id}


def _write_manifest(tmp_path: Path) -> Path:
    output_root = tmp_path / "output"
    tasks = [
        _write_run(output_root, name="valid", cpu_sec=4.0, wall_sec=2.0),
        _write_run(
            output_root,
            name="mismatch",
            cpu_sec=9.0,
            wall_sec=3.0,
            mismatch_count=1,
        ),
        _write_run(
            output_root,
            name="timeout",
            cpu_sec=1.0,
            wall_sec=4.0,
            timeout_count=1,
        ),
    ]
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "name": "synthetic-swebench",
                "output_root": str(output_root),
                "repetitions": 1,
                "allocations": [{"name": "eight-core", "logical_cores": 8, "cpuset_cpus": "0-7"}],
                "population": {"benchmark": "SWE-bench Pro"},
                "tasks": tasks,
            }
        )
    )
    return manifest_path


def test_summarize_campaign_uses_ratio_of_totals_and_separates_cohorts(
    tmp_path: Path,
) -> None:
    summary = summarize_campaign(_write_manifest(tmp_path))

    all_reports = summary["cohorts"]["all_reports_diagnostic"]
    valid = summary["cohorts"]["fidelity_valid"]
    sensitivity = summary["cohorts"]["complete_scheduler_valid_non_timeout_sensitivity"]

    assert all_reports["task_count"] == 3
    assert all_reports["total_cpu_sec"] == 14.0
    assert all_reports["total_wall_sec"] == 9.0
    assert all_reports["weighted_average_cores"] == pytest.approx(14.0 / 9.0)
    assert all_reports["unweighted_mean_task_average_cores"] == pytest.approx((2.0 + 3.0 + 0.25) / 3)
    assert valid["task_count"] == 1
    assert valid["weighted_average_cores"] == 2.0
    assert sensitivity["task_count"] == 2
    assert sensitivity["weighted_average_cores"] == pytest.approx(13.0 / 5.0)


def test_summarize_campaign_rejects_inconsistent_reported_average(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path)
    report_path = tmp_path / "output/eight-core/valid/rep-01/pipeline_report.json"
    report = json.loads(report_path.read_text())
    report["container_cpu"]["average_logical_cores_used"] = 7.0
    report_path.write_text(json.dumps(report))

    with pytest.raises(ValueError, match="average_logical_cores_used is inconsistent"):
        summarize_campaign(manifest_path)
