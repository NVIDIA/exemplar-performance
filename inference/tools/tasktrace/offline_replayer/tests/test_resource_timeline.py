# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.reporting.resource_timeline import (
    ResourceTimelineError,
    load_resource_timeline,
    render_resource_timeline_svg,
)

MIB = 1024 * 1024


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    samples = tmp_path / "resource_timeseries.json"
    samples.write_text(
        json.dumps(
            {
                "schema_version": "container-cgroup-resource-timeseries.v1",
                "sample_interval_sec": 0.5,
                "markers": [
                    {"name": "command_start", "elapsed_sec": 0.1},
                    {"name": "command_end", "elapsed_sec": 0.9},
                ],
                "samples": [
                    {
                        "elapsed_sec": 0.0,
                        "cpu_average_logical_cores": None,
                        "memory_current_bytes": 10 * MIB,
                        "memory_stat": {
                            "anon": 6 * MIB,
                            "file": 2 * MIB,
                            "kernel": 2 * MIB,
                        },
                    },
                    {
                        "elapsed_sec": 0.5,
                        "cpu_average_logical_cores": 1.5,
                        "memory_current_bytes": 70 * MIB,
                        "memory_stat": {
                            "anon": 60 * MIB,
                            "file": 5 * MIB,
                            "kernel": 5 * MIB,
                        },
                    },
                    {
                        "elapsed_sec": 1.0,
                        "cpu_average_logical_cores": 0.2,
                        "memory_current_bytes": 20 * MIB,
                        "memory_stat": {
                            "anon": 10 * MIB,
                            "file": 5 * MIB,
                            "kernel": 5 * MIB,
                        },
                    },
                ],
            }
        )
    )
    report = tmp_path / "pipeline_report.json"
    report.write_text(
        json.dumps(
            {
                "container_cpu": {"allocated_logical_cpus": 2},
                "container_memory": {
                    "samples_path": str(samples),
                    "timeline_valid_for_shape": True,
                    "peak_valid": True,
                    "natural_peak_trustworthy": True,
                    "lifetime": {"peak_bytes": 800 * MIB},
                    "observed_limit_margin": {
                        "configured_limit_state": "finite",
                        "configured_limit_bytes": 1024 * MIB,
                        "remaining_to_limit_bytes": 224 * MIB,
                        "peak_fraction_of_limit": 800 / 1024,
                        "arithmetic_valid": True,
                    },
                },
            }
        )
    )
    return report, samples


def test_render_resource_timeline_has_aligned_step_cpu_and_memory_panels(
    tmp_path: Path,
) -> None:
    report, _samples = _write_fixture(tmp_path)
    output = tmp_path / "timeline.svg"

    render_resource_timeline_svg(report, output)

    svg = output.read_text()
    assert "Sandbox CPU and memory over time" in svg
    assert "Sampled interval CPU demand" in svg
    assert "Cgroup-charged memory (MiB)" in svg
    assert "natural-peak=trustworthy" in svg
    assert "Leaf memory.max: 1024.0 MiB" in svg
    assert "observed margin: 224.0 MiB" in svg
    assert "not a recommendation" in svg
    assert "command start" in svg
    assert svg.count('class="cpu-interval"') == 2
    assert 'class="cpu-interval-transition"' in svg
    assert "#2563eb" in svg
    assert "#dc2626" in svg


def test_untimed_lifetime_peak_is_annotated_but_does_not_scale_timed_axis(
    tmp_path: Path,
) -> None:
    report, _samples = _write_fixture(tmp_path)
    output = tmp_path / "timeline.svg"

    render_resource_timeline_svg(report, output)

    svg = output.read_text()
    assert "Untimed controlled-finish-gate peak: 800.0 MiB · exact" in svg
    assert "not on the timed y-axis" in svg
    # The timed maximum is 70 MiB plus 8%, not the untimed 800 MiB peak.
    assert 'data-timed-memory-max-mib="75.600000"' in svg


def test_load_resource_timeline_rejects_missing_memory_profile(tmp_path: Path) -> None:
    report = tmp_path / "pipeline_report.json"
    report.write_text("{}\n")

    with pytest.raises(ResourceTimelineError, match="no container memory profile"):
        load_resource_timeline(report)


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("nonfinite_elapsed", "elapsed_sec must be a finite nonnegative"),
        ("negative_memory", "memory_current_bytes must be a finite nonnegative"),
        ("nonmonotonic", "strictly increasing"),
        ("negative_cpu", "cpu_average_logical_cores must be a finite nonnegative"),
        ("negative_breakdown", "memory_stat.anon must be a finite nonnegative"),
    ],
)
def test_load_resource_timeline_rejects_invalid_samples(tmp_path: Path, case: str, message: str) -> None:
    report, samples_path = _write_fixture(tmp_path)
    payload = json.loads(samples_path.read_text())
    if case == "nonfinite_elapsed":
        payload["samples"][1]["elapsed_sec"] = float("nan")
    elif case == "negative_memory":
        payload["samples"][1]["memory_current_bytes"] = -1
    elif case == "nonmonotonic":
        payload["samples"][1]["elapsed_sec"] = 0.0
    elif case == "negative_cpu":
        payload["samples"][1]["cpu_average_logical_cores"] = -0.1
    else:
        payload["samples"][1]["memory_stat"]["anon"] = -1
    samples_path.write_text(json.dumps(payload))

    with pytest.raises(ResourceTimelineError, match=message):
        load_resource_timeline(report)


def test_load_resource_timeline_rejects_marker_outside_sample_window(
    tmp_path: Path,
) -> None:
    report, samples_path = _write_fixture(tmp_path)
    payload = json.loads(samples_path.read_text())
    payload["markers"][1]["elapsed_sec"] = 1.1
    samples_path.write_text(json.dumps(payload))

    with pytest.raises(ResourceTimelineError, match="outside the sample window"):
        load_resource_timeline(report)


def test_load_resource_timeline_rejects_summary_marked_invalid(tmp_path: Path) -> None:
    report, _samples_path = _write_fixture(tmp_path)
    payload = json.loads(report.read_text())
    payload["container_memory"]["timeline_valid_for_shape"] = False
    payload["container_memory"]["timeline_invalid_reasons"] = ["sample gap"]
    report.write_text(json.dumps(payload))

    with pytest.raises(ResourceTimelineError, match="sample gap"):
        load_resource_timeline(report)
