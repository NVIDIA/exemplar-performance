# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import offline_replayer.reporting.report_sets as report_sets
import pytest
from offline_replayer.reporting.report_sets import (
    ReportSetError,
    load_report_sets,
    write_diagnostic_outputs,
)


def _manifest(path: Path) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema_version": "scheduler-report-sets.v1",
                "reports": {
                    "a": "raw/a.json",
                    "b": "/var/reports/b.json",
                },
                "sets": [
                    {
                        "name": "all",
                        "output_dir": "outputs/all",
                        "report_ids": ["a", "b"],
                    }
                ],
            }
        )
    )
    return path


def test_load_report_sets_resolves_manifest_relative_paths(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "sets.json")

    loaded = load_report_sets(manifest)

    assert loaded[0].name == "all"
    assert loaded[0].output_dir == (tmp_path / "outputs/all").resolve()
    assert loaded[0].pipeline_reports == (
        (tmp_path / "raw/a.json").resolve(),
        Path("/var/reports/b.json"),
    )


def test_load_report_sets_rejects_unknown_report_alias(tmp_path: Path) -> None:
    manifest = tmp_path / "sets.json"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": "scheduler-report-sets.v1",
                "reports": {"a": "a.json"},
                "sets": [{"name": "broken", "output_dir": "out", "report_ids": ["missing"]}],
            }
        )
    )

    with pytest.raises(ReportSetError, match="unknown report IDs"):
        load_report_sets(manifest)


def test_render_report_sets_dispatches_selected_named_set(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest = _manifest(tmp_path / "sets.json")
    received: list[object] = []

    def fake_build_summary(paths: list[Path]) -> list[str]:
        received.append(paths)
        return ["summary"]

    def fake_write_outputs(tasks: list[str], output_dir: Path, *, render_png: bool) -> dict[str, str]:
        received.append((tasks, output_dir, render_png))
        return {"json": str(output_dir / "batch_summary.json")}

    monkeypatch.setattr(report_sets, "build_summary", fake_build_summary)
    monkeypatch.setattr(report_sets, "write_outputs", fake_write_outputs)

    outputs = report_sets.render_report_sets(manifest, render_png=True, selected_names=("all",))

    assert outputs["all"]["json"].endswith("outputs/all/batch_summary.json")
    assert received[0] == [(tmp_path / "raw/a.json").resolve(), Path("/var/reports/b.json")]
    assert received[1] == (["summary"], (tmp_path / "outputs/all").resolve(), True)


def test_write_diagnostic_outputs_labels_invalid_scheduler_peaks_as_lower_bounds(
    tmp_path: Path,
) -> None:
    report = tmp_path / "invalid.json"
    report.write_text(
        json.dumps(
            {
                "task_name": "caffe-cifar-10",
                "source_trial": "caffe-cifar-10__trial",
                "replay_summary": {"source_outcome_mismatch_count": 7},
                "container_cpu": {
                    "wall_sec": 288.4,
                    "usage_sec": 1183.6,
                    "average_logical_cores_used": 4.104,
                    "utilization_percent_of_allocation": 51.3,
                },
                "scheduler_concurrency": {
                    "peak_simultaneous_logical_cores": 8,
                    "lost_perf_events": 3063,
                    "anomalies": ["lost ordering"],
                    "cgroup_validation": {"valid_for_concurrency_reporting": False},
                },
            }
        )
    )

    outputs = write_diagnostic_outputs((report,), tmp_path / "out")

    summary = json.loads(Path(outputs["diagnostic_json"]).read_text())["tasks"][0]
    assert summary["classification"] == "diagnostic_non_comparable_replay"
    assert summary["observed_window_average_logical_cores_used"] == pytest.approx(4.104)
    assert summary["scheduler_peak_interpretation"] == "observed_lower_bound_only"
    assert summary["full_replay_average_inference_allowed"] is False
    assert b"\r" not in Path(outputs["diagnostic_csv"]).read_bytes()
