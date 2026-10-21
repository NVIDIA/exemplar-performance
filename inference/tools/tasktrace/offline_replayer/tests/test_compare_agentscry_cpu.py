# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import csv
import json
from pathlib import Path

from offline_replayer.reporting.agentscry_comparison import build_comparison_report


def _write_pipeline(path: Path, usage_sec: float = 1.0, wall_sec: float = 1.0) -> Path:
    path.write_text(
        json.dumps(
            {
                "task_id": "task",
                "trial_name": "task.1",
                "image_identity": {"id": "sha256:image"},
                "container_cpu": {
                    "usage_sec": usage_sec,
                    "wall_sec": wall_sec,
                    "average_logical_cores_used": usage_sec / wall_sec,
                },
            }
        )
    )
    return path


def _write_csv(path: Path, rows: list[dict[str, str]]) -> Path:
    fields = ["EPOCH", "DURATION", "COMM", "PPID", "PID", "CPU", "USER", "SYS", "NTHD", "ARGS"]
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_comparison_sums_apportioned_multi_exec_rows_without_losing_cpu(
    tmp_path: Path,
) -> None:
    pipeline = _write_pipeline(tmp_path / "pipeline.json", usage_sec=1.0)
    filtered = _write_csv(
        tmp_path / "agentscry.csv",
        [
            {
                "PID": "10",
                "PPID": "1",
                "USER": "0.100",
                "SYS": "0.100",
                "ARGS": "/bin/sh /tasktrace-profile-entrypoint.sh",
            },
            {
                "PID": "11",
                "PPID": "10",
                "USER": "0.200",
                "SYS": "0.200",
                "ARGS": "/bin/bash -c python3 -m offline_replayer.adapters.transcript_replay",
            },
            {
                "PID": "11",
                "PPID": "10",
                "USER": "0.200",
                "SYS": "0.200",
                "ARGS": "/usr/bin/python3 -m offline_replayer.adapters.transcript_replay",
            },
        ],
    )

    report = build_comparison_report(
        pipeline,
        filtered,
        root_command_substring="/tasktrace-profile-entrypoint.sh",
        tolerance_fraction=0.01,
    )

    assert report["agentscry_cpu"]["usage_sec"] == 1.0
    assert report["agentscry_cpu"]["exit_accounted_pid_count"] == 2
    assert report["agentscry_cpu"]["cpu_accounted_row_count"] == 3
    assert report["comparison"]["cpu_within_tolerance"] is True
    assert report["comparison"]["equivalent_for_utilization"] is True


def test_comparison_rejects_missing_exit_accounting_even_if_sum_is_close(
    tmp_path: Path,
) -> None:
    pipeline = _write_pipeline(tmp_path / "pipeline.json", usage_sec=1.0)
    filtered = _write_csv(
        tmp_path / "agentscry.csv",
        [
            {
                "PID": "10",
                "PPID": "1",
                "USER": "0.500",
                "SYS": "0.500",
                "ARGS": "/bin/sh /tasktrace-profile-entrypoint.sh",
            },
            {
                "PID": "12",
                "PPID": "10",
                "USER": "",
                "SYS": "",
                "ARGS": "/usr/bin/grep input.log",
            },
        ],
    )

    report = build_comparison_report(
        pipeline,
        filtered,
        root_command_substring="/tasktrace-profile-entrypoint.sh",
        tolerance_fraction=0.01,
    )

    assert report["agentscry_cpu"]["missing_exit_accounting_pid_count"] == 1
    assert report["comparison"]["cpu_within_tolerance"] is True
    assert report["comparison"]["equivalent_for_utilization"] is False
    assert "without exit CPU accounting prevent strict equivalence" in report["comparison"]["reason"]
