# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.adapters.transcript_replay import (
    ReplayError,
    _read_cpu_stat,
    extract_actions,
    replay_actions,
)


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> Path:
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    return path


def test_extract_actions_filters_task_and_parses_typed_payloads(tmp_path: Path) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "ignored",
                "answer": '[Tool Call: exec]\n{"command": "false"}',
            },
            {
                "task_id": "calendar",
                "turn": 0,
                "answer": '[Tool Call: exec]\n{"command": "printf ok"}',
            },
            {
                "task_id": "calendar",
                "turn": 1,
                "answer": (
                    '[Tool Call: write]\n{"path": "event.ics", ' '"content": "BEGIN:VCALENDAR\\nEND:VCALENDAR"}'
                ),
            },
        ],
    )

    extraction = extract_actions(transcript, task_id="calendar")

    assert [action["tool"] for action in extraction.actions] == ["exec", "write"]
    assert extraction.actions[0]["payload"]["command"] == "printf ok"
    assert extraction.actions[1]["payload"]["path"] == "event.ics"


def test_extract_actions_can_select_one_trial_for_a_repeated_task(tmp_path: Path) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "calendar",
                "trial_name": "calendar.1-of-3.default",
                "answer": '[Tool Call: exec]\n{"command": "printf first"}',
            },
            {
                "task_id": "calendar",
                "trial_name": "calendar.2-of-3.default",
                "answer": '[Tool Call: exec]\n{"command": "printf second"}',
            },
        ],
    )

    extraction = extract_actions(
        transcript,
        task_id="calendar",
        trial_name="calendar.2-of-3.default",
    )

    assert extraction.trial_name == "calendar.2-of-3.default"
    assert len(extraction.actions) == 1
    assert extraction.actions[0]["payload"]["command"] == "printf second"


def test_replay_actions_times_exec_and_remaps_agent_workspace_write(tmp_path: Path) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "calendar",
                "answer": ('[Tool Call: exec]\n' '{"command": "printf generated > generated.txt"}'),
            },
            {
                "task_id": "calendar",
                "answer": (
                    '[Tool Call: write]\n'
                    '{"path": "/tmp/pinchbench/0001/agent_workspace/event.ics", '
                    '"content": "calendar"}'
                ),
            },
        ],
    )
    extraction = extract_actions(transcript, task_id="calendar")

    report = replay_actions(extraction, workspace=tmp_path / "workspace")

    assert report["summary"]["actions_ok"] == 2
    assert report["summary"]["actions_failed"] == 0
    assert report["summary"]["cpu_time_sec"] >= 0
    assert (tmp_path / "workspace" / "generated.txt").read_text() == "generated"
    assert (tmp_path / "workspace" / "event.ics").read_text() == "calendar"
    assert all(action["cpu_time_sec"] >= 0 for action in report["actions"])
    assert report["clock_reference"]["clock_id"] == "CLOCK_MONOTONIC"
    assert all(action["replay_end_monotonic_ns"] >= action["replay_start_monotonic_ns"] for action in report["actions"])


def test_replay_timeout_partial_bytes_are_json_serializable(tmp_path: Path) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "slow",
                "answer": ('[Tool Call: exec]\n' '{"command": "printf partial; sleep 1"}'),
            },
        ],
    )

    report = replay_actions(
        extract_actions(transcript, task_id="slow"),
        workspace=tmp_path / "workspace",
        timeout_sec=0.01,
    )

    assert report["actions"][0]["status"] == "timed_out"
    assert report["actions"][0]["stdout_preview"] == "partial"
    json.dumps(report)


def test_replay_exec_replaces_non_utf8_output_bytes(tmp_path: Path) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "binary",
                "answer": '[Tool Call: exec]\n{"command": "printf \'\\\\377\'"}',
            },
        ],
    )

    report = replay_actions(
        extract_actions(transcript, task_id="binary"),
        workspace=tmp_path / "workspace",
    )

    assert report["actions"][0]["status"] == "ok"
    assert report["actions"][0]["stdout_preview"] == "\ufffd"
    json.dumps(report)


def test_replay_can_seed_initial_workspace_and_records_unsupported_tool(
    tmp_path: Path,
) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "input.txt").write_text("payload")
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "pdf",
                "answer": '[Tool Call: read]\n{"path": "input.txt"}',
            },
            {
                "task_id": "pdf",
                "answer": '[Tool Call: browser]\n{"action": "open", "url": "file:///input.pdf"}',
            },
        ],
    )

    report = replay_actions(
        extract_actions(transcript, task_id="pdf"),
        workspace=tmp_path / "workspace",
        initial_workspace=initial,
    )

    assert report["actions"][0]["bytes_read"] == len("payload")
    assert report["actions"][1]["status"] == "unsupported"
    assert report["summary"]["actions_unsupported"] == 1


def test_replay_read_applies_recorded_line_offset_and_limit(tmp_path: Path) -> None:
    initial = tmp_path / "initial"
    initial.mkdir()
    (initial / "input.txt").write_text("first\nlonger-second\nthird\n")
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {
                "task_id": "window",
                "answer": ('[Tool Call: read]\n' '{"path": "input.txt", "offset": 2, "limit": 1}'),
            }
        ],
    )

    report = replay_actions(
        extract_actions(transcript, task_id="window"),
        workspace=tmp_path / "workspace",
        initial_workspace=initial,
    )

    assert report["actions"][0]["bytes_read"] == len("longer-second\n")
    assert report["actions"][0]["offset"] == 2
    assert report["actions"][0]["limit"] == 1


def test_multiple_tasks_requires_explicit_task_selection(tmp_path: Path) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [
            {"task_id": "a", "answer": '[Tool Call: exec]\n{"command": "true"}'},
            {"task_id": "b", "answer": '[Tool Call: exec]\n{"command": "true"}'},
        ],
    )

    with pytest.raises(ReplayError, match="provide --task-id"):
        extract_actions(transcript)


def test_read_cpu_stat_parses_cgroup_v2_values(tmp_path: Path) -> None:
    (tmp_path / "cpu.stat").write_text("usage_usec 5000000\nuser_usec 4000000\nsystem_usec 1000000\n")

    assert _read_cpu_stat(tmp_path) == {
        "usage_usec": 5_000_000,
        "user_usec": 4_000_000,
        "system_usec": 1_000_000,
    }


def test_cgroup_cpu_instrumentation_is_attached_to_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    transcript = _write_jsonl(
        tmp_path / "trace.jsonl",
        [{"task_id": "cpu", "answer": '[Tool Call: exec]\n{"command": "true"}'}],
    )

    class FakeCgroupCpuMeter:
        def __init__(self) -> None:
            self.path = tmp_path / "fake-cgroup"

        def start(self) -> None:
            pass

        def begin_workload(self) -> None:
            pass

        def finish(self, wall_time_sec: float) -> dict[str, object]:
            return {
                "method": "cgroup-v2-cpu.stat",
                "usage_sec": 2.5,
                "average_logical_cores_used": 2.0,
                "logical_core_percent": 200.0,
            }

    monkeypatch.setattr("offline_replayer.adapters.transcript_replay.CgroupCpuMeter", FakeCgroupCpuMeter)

    report = replay_actions(
        extract_actions(transcript, task_id="cpu"),
        workspace=tmp_path / "workspace",
        cgroup_cpu=True,
        cgroup_ready_file=tmp_path / "ready",
    )

    assert report["summary"]["cgroup_cpu"]["usage_sec"] == 2.5
    assert (tmp_path / "ready").read_text() == str(tmp_path / "fake-cgroup") + "\n"
