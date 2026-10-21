#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay top-level actions from a finalized TaskTrace shell-command trace.

TaskTrace's finalized trace records the shell actions emitted by an agent and
their observed timing. For workloads that launch background compilation, the
gaps between actions are part of the execution schedule: descendants continue
running while the model is producing the next action. The default paced replay
therefore holds each action until its original offset from the first action.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..timebase import monotonic_clock_reference
from .pacing import UNPACED
from .transcript_replay import ReplayError

TRACE_SCHEMA_VERSION = "tasktrace-cpu-trace.v1"
# Every id the shell-command trace has been finalized under. Traces are long-lived artefacts: those produced
# before the TaskTrace rename carry "riptide-cpu-trace.v1", and the capture package's own rename emits
# "tasktrace-capture-cpu-trace.v1". The record layout is identical across all three.
ACCEPTED_TRACE_SCHEMA_VERSIONS = frozenset(
    {
        TRACE_SCHEMA_VERSION,
        "riptide-cpu-trace.v1",
        "tasktrace-capture-cpu-trace.v1",
    }
)
REPLAY_SCHEMA_VERSION = "offline-trace-replay.v1"
PREVIEW_LENGTH = 1000
PACING_MODES = ("recorded", UNPACED)


@dataclass(frozen=True)
class TaskTraceTrace:
    source_path: Path
    header: dict[str, Any]
    actions: list[dict[str, Any]]

    @property
    def action_timeout_sec(self) -> float | None:
        environment = self.header.get("environment")
        timeout = environment.get("action_timeout_sec") if isinstance(environment, dict) else None
        if timeout is None:
            return None
        if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
            raise ReplayError("trace environment action timeout must be positive")
        return float(timeout)


def read_trace(path: Path) -> TaskTraceTrace:
    """Load and validate a finalized shell-command action stream (any id in ``ACCEPTED_TRACE_SCHEMA_VERSIONS``)."""
    if not path.is_file():
        raise ReplayError(f"TaskTrace trace does not exist: {path}")
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ReplayError(f"trace line {line_number} is invalid JSON: {exc.msg}") from exc
        if not isinstance(record, dict):
            raise ReplayError(f"trace line {line_number} is not a JSON object")
        records.append(record)
    header = next((record for record in records if record.get("record_type") == "trace"), None)
    if header is None:
        raise ReplayError("trace has no record_type=trace header; finalize the TaskTrace trace first")
    schema_version = header.get("schema_version")
    if not isinstance(schema_version, str) or schema_version not in ACCEPTED_TRACE_SCHEMA_VERSIONS:
        raise ReplayError(f"unsupported TaskTrace trace schema: {schema_version!r}")
    actions = [record for record in records if record.get("record_type") == "action"]
    if not actions:
        raise ReplayError("trace has no replayable action records")
    last_index = 0
    timed = None
    for action in actions:
        index = action.get("action_index")
        command = action.get("command")
        started = action.get("started_at_epoch")
        finished = action.get("finished_at_epoch")
        if not isinstance(index, int) or isinstance(index, bool) or index <= last_index:
            raise ReplayError("trace action indexes must be increasing positive integers")
        if not isinstance(command, str):
            raise ReplayError(f"trace action {index} has no string command")
        has_timestamps = "started_at_epoch" in action or "finished_at_epoch" in action
        if timed is None:
            timed = has_timestamps
        if timed != has_timestamps:
            state = "has" if has_timestamps else "has no"
            raise ReplayError(f"trace action {index} {state} timestamps; a trace's actions must all have them or none")
        if has_timestamps:
            if not isinstance(started, (int, float)) or not isinstance(finished, (int, float)):
                raise ReplayError(f"trace action {index} has invalid timestamps")
            if finished < started:
                raise ReplayError(f"trace action {index} finishes before it starts")
        last_index = index
    return TaskTraceTrace(source_path=path, header=header, actions=actions)


def _preview(value: str | bytes) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[:PREVIEW_LENGTH]


def _source_status(action: dict[str, Any]) -> str:
    exception_info = str(action.get("exception_info") or "")
    if exception_info == "action was not executed":
        return "skipped_not_executed"
    if "timed out" in exception_info.lower():
        return "timed_out"
    return "ok" if action.get("returncode") == 0 else "failed"


def _execute_action(action: dict[str, Any], *, cwd: Path, timeout_sec: float | None) -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["/bin/bash", "-c", str(action["command"])],
            cwd=cwd,
            text=True,
            errors="replace",
            capture_output=True,
            timeout=timeout_sec,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "status": "timed_out",
            "stdout_preview": _preview(exc.stdout or ""),
            "stderr_preview": _preview(exc.stderr or ""),
        }
    return {
        "status": "ok" if completed.returncode == 0 else "failed",
        "returncode": completed.returncode,
        "stdout_preview": _preview(completed.stdout),
        "stderr_preview": _preview(completed.stderr),
    }


class _LoopSync:
    """The tool side's half of a loop-order replay (see trace_replay.rs LoopSync): wait in the sync directory
    for the LLM call an action followed, and write each action's index to `tool_done` after it, `tool_end`
    when done."""

    POLL_SEC = 0.005

    def __init__(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        self.directory = directory

    def _counter(self, name: str) -> int | None:
        try:
            return int((self.directory / name).read_text().strip())
        except (OSError, ValueError):
            return None

    def wait_for_llm(self, llm_index: int) -> float:
        started = time.monotonic()
        while not (self.directory / "llm_end").exists():
            done = self._counter("llm_done")
            if done is not None and done >= llm_index:
                break
            time.sleep(self.POLL_SEC)
        return time.monotonic() - started

    def tool_done(self, action_index: object) -> None:
        partial = self.directory / ".tool_done.partial"
        partial.write_text(f"{action_index}\n")
        partial.replace(self.directory / "tool_done")

    def close(self) -> None:
        (self.directory / "tool_end").write_bytes(b"")


def replay_trace(
    trace: TaskTraceTrace,
    *,
    cwd: Path,
    pacing: str = "recorded",
    timeout_sec: float | None = None,
    loop_sync_dir: Path | None = None,
) -> dict[str, Any]:
    """Replay executed top-level actions and return a source-outcome comparison.

    Outcome mismatches are retained in the report instead of stopping the
    replay immediately: timing-sensitive background work can finish earlier
    or later offline, and the measured CPU behavior is still evidence worth
    preserving so long as the drift is explicit.
    """
    if pacing not in PACING_MODES:
        raise ReplayError(f"unsupported pacing mode: {pacing!r}")
    if not cwd.is_dir():
        raise ReplayError(f"replay working directory does not exist: {cwd}")
    effective_timeout = timeout_sec if timeout_sec is not None else trace.action_timeout_sec
    if effective_timeout is not None and effective_timeout <= 0:
        raise ReplayError("action timeout must be positive")

    executable_actions = [action for action in trace.actions if _source_status(action) != "skipped_not_executed"]
    if not executable_actions:
        raise ReplayError("trace has no source-executed actions to replay")
    timed = "started_at_epoch" in trace.actions[0]
    if pacing == "recorded" and not timed:
        raise ReplayError("recorded pacing needs the trace's timestamps, and this trace has none; use unpaced")
    if loop_sync_dir is not None:
        if pacing != "unpaced":
            raise ReplayError("a loop-sync replay keeps the recorded order, not the schedule; use unpaced pacing")
        for action in executable_actions:
            if "after_llm_index" not in action:
                raise ReplayError(
                    f"trace action {action['action_index']} has no after_llm_index; a loop-sync replay needs a "
                    "trace written with tasktrace-capture --trajectory-timings omit"
                )
    first_start = float(executable_actions[0]["started_at_epoch"]) if timed else None
    source_end = float(executable_actions[-1]["finished_at_epoch"]) if timed else None
    loop_sync = _LoopSync(loop_sync_dir) if loop_sync_dir is not None else None
    try:
        results, wall_time_sec = _replay_actions(trace, cwd, pacing, effective_timeout, timed, first_start, loop_sync)
    finally:
        if loop_sync is not None:
            loop_sync.close()
    mismatches = [result["action_index"] for result in results if not result["outcome_matches_source"]]
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_trace": str(trace.source_path.resolve()),
        "source_schema_version": trace.header.get("schema_version"),
        "source_trial": trace.header.get("source_trial"),
        "task_name": trace.header.get("task_name"),
        "source_reward": trace.header.get("reward"),
        "source_limitations": trace.header.get("limitations", []),
        "pacing": pacing,
        "loop_sync_dir": str(loop_sync_dir) if loop_sync_dir is not None else None,
        "action_timeout_sec": effective_timeout,
        "clock_reference": monotonic_clock_reference(),
        "actions": results,
        "summary": {
            "actions_recorded": len(trace.actions),
            "actions_executed": len(executable_actions),
            "actions_skipped_not_executed": len(trace.actions) - len(executable_actions),
            "actions_ok": sum(result["status"] == "ok" for result in results),
            "actions_failed": sum(result["status"] == "failed" for result in results),
            "actions_timed_out": sum(result["status"] == "timed_out" for result in results),
            "source_outcome_mismatch_action_indexes": mismatches,
            "source_outcome_mismatch_count": len(mismatches),
            "source_interval_sec": round(source_end - first_start, 9) if timed else None,
            "wall_time_sec": round(wall_time_sec, 9),
        },
    }


def _replay_actions(
    trace: TaskTraceTrace,
    cwd: Path,
    pacing: str,
    effective_timeout: float | None,
    timed: bool,
    first_start: float | None,
    loop_sync: _LoopSync | None,
) -> tuple[list[dict[str, Any]], float]:
    """Run the trace's actions in order; returns the per-action results and the replay's wall time."""
    replay_started = time.monotonic()
    results: list[dict[str, Any]] = []
    for action in trace.actions:
        expected_status = _source_status(action)
        result: dict[str, Any] = {
            "action_index": action["action_index"],
            "command": action["command"],
            "source_status": expected_status,
            "source_wall_time_sec": action.get("wall_time_sec"),
        }
        if expected_status == "skipped_not_executed":
            result["status"] = "skipped_not_executed"
            result["outcome_matches_source"] = True
            results.append(result)
            if loop_sync is not None:
                loop_sync.tool_done(action["action_index"])
            continue
        source_offset_sec = float(action["started_at_epoch"]) - first_start if timed else None
        result["source_start_offset_sec"] = round(source_offset_sec, 9) if timed else None
        if pacing == "recorded" and timed:
            sleep_sec = source_offset_sec - (time.monotonic() - replay_started)
            if sleep_sec > 0:
                time.sleep(sleep_sec)
        if loop_sync is not None:
            after = action.get("after_llm_index")
            waited = loop_sync.wait_for_llm(after) if isinstance(after, int) else 0.0
            result["loop_wait_sec"] = round(waited, 9)
        action_started = time.monotonic()
        action_started_ns = time.monotonic_ns()
        outcome = _execute_action(action, cwd=cwd, timeout_sec=effective_timeout)
        result.update(outcome)
        result["wall_time_sec"] = round(time.monotonic() - action_started, 9)
        result["replay_start_monotonic_ns"] = action_started_ns
        result["replay_end_monotonic_ns"] = time.monotonic_ns()
        result["outcome_matches_source"] = result["status"] == expected_status
        results.append(result)
        if loop_sync is not None:
            loop_sync.tool_done(action["action_index"])
    return results, time.monotonic() - replay_started


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay action records from a finalized TaskTrace CPU trace.")
    parser.add_argument("trace", type=Path, help="Finalized tasktrace_shell_commands.jsonl.")
    parser.add_argument("--output", type=Path, required=True, help="Replay report JSON path.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Replay working directory.")
    parser.add_argument("--pacing", choices=PACING_MODES, default="recorded")
    parser.add_argument(
        "--timeout-sec",
        type=float,
        help="Override the per-action timeout recorded in the source trace.",
    )
    parser.add_argument(
        "--loop-sync-dir",
        type=Path,
        help="Keep the recorded agent loop's order with a concurrent LLM-side replay, through files in this "
        "directory (unpaced; the trace's actions need after_llm_index).",
    )
    args = parser.parse_args(argv)
    if args.timeout_sec is not None and args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    try:
        report = replay_trace(
            read_trace(args.trace),
            cwd=args.cwd,
            pacing=args.pacing,
            timeout_sec=args.timeout_sec,
            loop_sync_dir=args.loop_sync_dir,
        )
    except (OSError, ReplayError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(f"Task: {report['task_name']}")
    print(f"Source trial: {report['source_trial']}")
    print(
        "Actions: "
        f"{summary['actions_executed']}/{summary['actions_recorded']} executed, "
        f"{summary['actions_timed_out']} timed out, "
        f"{summary['source_outcome_mismatch_count']} source-outcome mismatches"
    )
    print(f"Wall time: {summary['wall_time_sec']:.9f} sec")
    print(f"Report: {args.output}")
    return int(summary["source_outcome_mismatch_count"] > 0)


if __name__ == "__main__":
    sys.exit(main())
