# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import pytest
from offline_replayer.adapters.trace_replay import read_trace, replay_trace
from offline_replayer.adapters.transcript_replay import ReplayError


def _write_trace(path: Path, actions: list[dict[str, object]]) -> Path:
    rows = [
        {
            "record_type": "trace",
            "schema_version": "tasktrace-cpu-trace.v1",
            "source_trial": "task__trial",
            "task_name": "task",
            "environment": {"action_timeout_sec": 0.1},
            "limitations": [],
        },
        *actions,
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _action(
    index: int,
    command: str,
    *,
    start: float,
    end: float,
    returncode: int = 0,
    exception_info: str | None = None,
) -> dict[str, object]:
    return {
        "record_type": "action",
        "action_index": index,
        "command": command,
        "started_at_epoch": start,
        "finished_at_epoch": end,
        "wall_time_sec": end - start,
        "returncode": returncode,
        "exception_info": exception_info,
    }


def test_replay_trace_executes_actions_and_skips_unexecuted_source_action(
    tmp_path: Path,
) -> None:
    trace = read_trace(
        _write_trace(
            tmp_path / "trace.jsonl",
            [
                _action(1, "printf done > output.txt", start=1.0, end=1.01),
                _action(
                    2,
                    "printf should-not-run > skipped.txt",
                    start=1.02,
                    end=1.03,
                    returncode=-1,
                    exception_info="action was not executed",
                ),
            ],
        )
    )

    report = replay_trace(trace, cwd=tmp_path, pacing="unpaced")

    assert (tmp_path / "output.txt").read_text() == "done"
    assert not (tmp_path / "skipped.txt").exists()
    assert report["summary"]["actions_executed"] == 1
    assert report["summary"]["actions_skipped_not_executed"] == 1
    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert report["clock_reference"]["clock_id"] == "CLOCK_MONOTONIC"
    executed = report["actions"][0]
    assert executed["replay_end_monotonic_ns"] >= executed["replay_start_monotonic_ns"]


def test_replay_trace_recorded_pacing_sleeps_to_action_offset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    trace = read_trace(
        _write_trace(
            tmp_path / "trace.jsonl",
            [
                _action(1, "true", start=10.0, end=10.0),
                _action(2, "true", start=13.0, end=13.0),
            ],
        )
    )
    now = 0.0
    slept: list[float] = []

    def monotonic() -> float:
        return now

    def sleep(seconds: float) -> None:
        nonlocal now
        slept.append(seconds)
        now += seconds

    monkeypatch.setattr("offline_replayer.adapters.trace_replay.time.monotonic", monotonic)
    monkeypatch.setattr("offline_replayer.adapters.trace_replay.time.sleep", sleep)

    replay_trace(trace, cwd=tmp_path, pacing="recorded")

    assert slept == [3.0]


def test_replay_trace_matches_recorded_timeout(tmp_path: Path) -> None:
    trace = read_trace(
        _write_trace(
            tmp_path / "trace.jsonl",
            [
                _action(
                    1,
                    "sleep 1",
                    start=1.0,
                    end=1.01,
                    returncode=-1,
                    exception_info="command timed out after 0.01 seconds",
                )
            ],
        )
    )

    report = replay_trace(trace, cwd=tmp_path, pacing="unpaced", timeout_sec=0.01)

    assert report["actions"][0]["status"] == "timed_out"
    assert report["summary"]["source_outcome_mismatch_count"] == 0


def test_read_trace_requires_finalized_tasktrace_schema(tmp_path: Path) -> None:
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps({"command_shell_quoted": "true"}) + "\n")

    with pytest.raises(ReplayError, match="finalize the TaskTrace trace"):
        read_trace(path)


def test_replay_trace_requires_at_least_one_executed_source_action(
    tmp_path: Path,
) -> None:
    trace = read_trace(
        _write_trace(
            tmp_path / "trace.jsonl",
            [
                _action(
                    1,
                    "true",
                    start=1.0,
                    end=1.1,
                    returncode=-1,
                    exception_info="action was not executed",
                )
            ],
        )
    )

    with pytest.raises(ReplayError, match="no source-executed actions"):
        replay_trace(trace, cwd=tmp_path)


@pytest.mark.parametrize(
    "schema_version",
    ["tasktrace-cpu-trace.v1", "riptide-cpu-trace.v1", "tasktrace-capture-cpu-trace.v1"],
)
def test_read_trace_accepts_every_finalized_schema_id(tmp_path: Path, schema_version: str) -> None:
    """Traces finalized before the rename (riptide-cpu-trace.v1) and by the capture package's own rename must keep
    replaying: the record layout never changed, only the id string."""
    path = tmp_path / "trace.jsonl"
    rows = [
        {
            "record_type": "trace",
            "schema_version": schema_version,
            "source_trial": "t",
            "task_name": "task",
            "environment": {"action_timeout_sec": 0.1},
            "limitations": [],
        },
        {
            "record_type": "action",
            "action_index": 1,
            "command": "true",
            "started_at_epoch": 1.0,
            "finished_at_epoch": 1.01,
            "wall_time_sec": 0.01,
            "returncode": 0,
            "exception_info": None,
        },
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    trace = read_trace(path)
    assert trace.header["schema_version"] == schema_version
    assert len(trace.actions) == 1


@pytest.mark.parametrize("schema_version", ["something-else.v9", None, 7, ["tasktrace-cpu-trace.v1"], {"id": "x"}])
def test_read_trace_rejects_an_unknown_or_malformed_schema_id(tmp_path: Path, schema_version: object) -> None:
    """Unknown ids and non-string values (an unhashable list/object would otherwise raise TypeError from the set
    lookup) both surface as the reader's own ReplayError."""
    path = tmp_path / "trace.jsonl"
    path.write_text(json.dumps({"record_type": "trace", "schema_version": schema_version}) + "\n")
    with pytest.raises(ReplayError, match="unsupported TaskTrace trace schema"):
        read_trace(path)


def _untimed(index: int, command: str) -> dict[str, object]:
    return {"record_type": "action", "action_index": index, "command": command, "returncode": 0, "exception_info": None}


def test_untimed_trace_replays_unpaced_and_refuses_recorded_pacing(tmp_path: Path) -> None:
    trace = read_trace(
        _write_trace(tmp_path / "trace.jsonl", [_untimed(1, "printf a > a"), _untimed(2, "printf b > b")])
    )
    report = replay_trace(trace, cwd=tmp_path, pacing="unpaced")
    assert report["summary"]["actions_executed"] == 2
    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert report["summary"]["source_interval_sec"] is None
    assert report["actions"][0]["source_start_offset_sec"] is None
    assert report["actions"][0]["source_wall_time_sec"] is None
    with pytest.raises(ReplayError, match="recorded pacing needs the trace's timestamps"):
        replay_trace(trace, cwd=tmp_path, pacing="recorded")


def test_trace_mixing_timed_and_untimed_actions_is_rejected(tmp_path: Path) -> None:
    path = _write_trace(tmp_path / "trace.jsonl", [_action(1, "true", start=1.0, end=1.1), _untimed(2, "true")])
    with pytest.raises(ReplayError, match="must all have them or none"):
        read_trace(path)


def _in_loop(index: int, after: int | None, command: str) -> dict[str, object]:
    return dict(_untimed(index, command), after_llm_index=after)


def _wait_for_counter(directory: Path, name: str, at_least: int) -> None:
    deadline = time.monotonic() + 10
    while True:
        try:
            if int((directory / name).read_text().strip()) >= at_least:
                return
        except (OSError, ValueError):
            pass
        assert time.monotonic() < deadline, f"{name} never reached {at_least}"
        time.sleep(0.002)


def test_loop_sync_keeps_the_recorded_agent_loop_order(tmp_path: Path) -> None:
    rows = [
        _in_loop(1, None, "printf a >> order"),
        _in_loop(2, 1, "printf b >> order"),
        _in_loop(3, 2, "printf c >> order"),
    ]
    trace = read_trace(_write_trace(tmp_path / "trace.jsonl", rows))
    sync, order = tmp_path / "sync", tmp_path / "order"

    def llm_side() -> None:
        for llm_index, after_action in ((1, 1), (2, 2)):
            _wait_for_counter(sync, "tool_done", after_action)
            with order.open("a") as handle:
                handle.write(f"L{llm_index}")
            (sync / "llm_done").write_text(f"{llm_index}\n")
        (sync / "llm_end").write_bytes(b"")

    thread = threading.Thread(target=llm_side)
    thread.start()
    report = replay_trace(trace, cwd=tmp_path, pacing="unpaced", loop_sync_dir=sync)
    thread.join()
    assert order.read_text() == "aL1bL2c"
    assert report["loop_sync_dir"] == str(sync)
    assert isinstance(report["actions"][1]["loop_wait_sec"], float)
    assert (sync / "tool_end").exists() and (sync / "tool_done").read_text().strip() == "3"


def test_loop_sync_stops_waiting_once_the_llm_side_has_ended(tmp_path: Path) -> None:
    trace = read_trace(_write_trace(tmp_path / "trace.jsonl", [_in_loop(1, 5, "true")]))
    sync = tmp_path / "sync"
    sync.mkdir()
    (sync / "llm_end").write_bytes(b"")
    report = replay_trace(trace, cwd=tmp_path, pacing="unpaced", loop_sync_dir=sync)
    assert report["summary"]["actions_executed"] == 1


def test_loop_sync_needs_unpaced_pacing_and_the_loop_order(tmp_path: Path) -> None:
    trace = read_trace(_write_trace(tmp_path / "a.jsonl", [_in_loop(1, None, "true")]))
    with pytest.raises(ReplayError, match="use unpaced"):
        replay_trace(trace, cwd=tmp_path, pacing="recorded", loop_sync_dir=tmp_path / "sync")
    trace = read_trace(_write_trace(tmp_path / "b.jsonl", [_untimed(1, "true")]))
    with pytest.raises(ReplayError, match="has no after_llm_index"):
        replay_trace(trace, cwd=tmp_path, pacing="unpaced", loop_sync_dir=tmp_path / "sync")
