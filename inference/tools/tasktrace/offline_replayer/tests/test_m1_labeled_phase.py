# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.examples.m1_labeled_phase import (
    REPLAY_SCHEMA_VERSION,
    SCHEMA_VERSION,
    run_fixture,
)


def _records(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def test_labeled_phase_fixture_emits_ordered_real_phase_intervals_once(
    tmp_path: Path,
) -> None:
    phase_log = tmp_path / "phases.jsonl"
    sentinel = tmp_path / "executed.once"
    replay_report = tmp_path / "replay_report.json"

    run_fixture(
        phase_log=phase_log,
        sentinel=sentinel,
        serial_cpu_sec=0.02,
        memory_mib=2,
        memory_hold_sec=0.02,
        parallel_workers=2,
        parallel_cpu_sec=0.03,
        teardown_hold_sec=0.02,
        child_binary="/bin/echo",
        replay_report=replay_report,
    )

    records = _records(phase_log)
    assert records[0]["schema_version"] == SCHEMA_VERSION
    assert records[0]["clock_reference"]["clock_id"] == "CLOCK_MONOTONIC"  # type: ignore[index]
    phase_starts = [record for record in records if record["record_type"] == "phase_start"]
    phase_ends = [record for record in records if record["record_type"] == "phase_end"]
    assert [record["phase"] for record in phase_starts] == [
        "serial_cpu",
        "memory_plateau",
        "parallel_cpu",
        "child_exec",
        "teardown",
    ]
    assert [record["phase"] for record in phase_ends] == [record["phase"] for record in phase_starts]
    for start, end in zip(phase_starts, phase_ends, strict=False):
        assert end["timestamp_ns"] >= start["timestamp_ns"]
    assert sentinel.read_text() == "executed-once\n"
    assert records[-1]["status"] == "ok"
    replay = json.loads(replay_report.read_text())
    assert replay["schema_version"] == REPLAY_SCHEMA_VERSION
    assert replay["clock_reference"] == records[0]["clock_reference"]
    assert replay["summary"]["actions_executed"] == 1
    assert replay["summary"]["source_outcome_mismatch_count"] == 0
    assert replay["actions"][0]["replay_start_monotonic_ns"] == records[0]["timestamp_ns"]
    assert replay["actions"][0]["replay_end_monotonic_ns"] == records[-1]["timestamp_ns"]

    with pytest.raises(FileExistsError):
        run_fixture(
            phase_log=phase_log,
            sentinel=sentinel,
            serial_cpu_sec=0.01,
            memory_mib=1,
            memory_hold_sec=0.01,
            parallel_workers=1,
            parallel_cpu_sec=0.01,
            teardown_hold_sec=0.01,
            child_binary="/bin/echo",
        )
