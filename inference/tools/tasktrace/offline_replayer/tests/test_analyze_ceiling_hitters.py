# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from offline_replayer.reporting.ceiling_hitters import analyze, write_outputs


def _write_summary(path: Path, tasks: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"tasks": tasks}))
    return path


def _write_commands(path: Path, rows: list[dict[str, object]]) -> Path:
    fields = [
        "task_name",
        "command_family",
        "command_preview",
        "lifecycle_wall_sec",
        "average_simultaneous_logical_cores",
        "peak_simultaneous_logical_cores",
        "time_at_or_above_8_cores_sec",
    ]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_analyze_ceiling_hitters_matches_baseline_and_missing_baseline(tmp_path: Path) -> None:
    wide = _write_summary(
        tmp_path / "wide.json",
        [
            {
                "task_name": "numeric",
                "wall_sec": 2.0,
                "average_logical_cores_used": 3.0,
                "peak_simultaneous_logical_cores": 16,
                "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "16": 1.0},
            },
            {
                "task_name": "new-task",
                "wall_sec": 10.0,
                "average_logical_cores_used": 1.0,
                "peak_simultaneous_logical_cores": 16,
                "time_sec_by_simultaneous_logical_cores": {"1": 9.999, "16": 0.001},
            },
            {
                "task_name": "narrow",
                "wall_sec": 1.0,
                "average_logical_cores_used": 1.0,
                "peak_simultaneous_logical_cores": 4,
                "time_sec_by_simultaneous_logical_cores": {"1": 1.0},
            },
        ],
    )
    baseline = _write_summary(
        tmp_path / "baseline.json",
        [
            {
                "task_name": "numeric",
                "wall_sec": 2.0,
                "average_logical_cores_used": 2.0,
                "peak_simultaneous_logical_cores": 8,
                "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "8": 1.0},
            }
        ],
    )
    command_rows = [
        {
            "task_name": "numeric",
            "command_family": "numpy",
            "command_preview": "python compute.py",
            "lifecycle_wall_sec": 2.0,
            "average_simultaneous_logical_cores": 3.0,
            "peak_simultaneous_logical_cores": 16,
            "time_at_or_above_8_cores_sec": 1.0,
        },
        {
            "task_name": "new-task",
            "command_family": "startup",
            "command_preview": "python smoke.py",
            "lifecycle_wall_sec": 10.0,
            "average_simultaneous_logical_cores": 1.0,
            "peak_simultaneous_logical_cores": 16,
            "time_at_or_above_8_cores_sec": 0.001,
        },
    ]
    wide_commands = _write_commands(tmp_path / "wide.csv", command_rows)
    baseline_commands = _write_commands(
        tmp_path / "baseline.csv",
        [
            {
                "task_name": "numeric",
                "command_family": "numpy",
                "command_preview": "python compute.py",
                "lifecycle_wall_sec": 2.0,
                "average_simultaneous_logical_cores": 2.0,
                "peak_simultaneous_logical_cores": 8,
                "time_at_or_above_8_cores_sec": 1.0,
            }
        ],
    )

    analysis = analyze(wide, baseline, wide_commands, baseline_commands)

    assert analysis["summary"] == {
        "wide_ceiling_hitter_count": 2,
        "with_comparable_baseline_count": 1,
        "also_hit_baseline_ceiling_count": 1,
        "same_dominant_command_family_count": 1,
    }
    rows = {row["task_name"]: row for row in analysis["tasks"]}
    assert rows["numeric"]["wide_peak_duration_class"] == "elevated average multi-core use"
    assert rows["numeric"]["baseline_also_reached_ceiling"] is True
    assert rows["new-task"]["wide_peak_duration_class"] == "sub-10ms ceiling touch"
    assert rows["new-task"]["baseline_report_available"] is False

    outputs = write_outputs(analysis, tmp_path / "out")
    assert outputs["json"].is_file()
    assert outputs["csv"].is_file()
    assert b"\r" not in outputs["csv"].read_bytes()
