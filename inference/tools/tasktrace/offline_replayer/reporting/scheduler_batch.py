#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Render validated scheduler-concurrency pipeline reports as batch charts.

Only replay-faithful, cgroup-validated reports enter these outputs. Command
lifetimes are attributed within the measured scheduler event stream; they are
useful for identifying CPU drivers but are not additive when async commands
overlap.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import re
import shlex
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SUMMARY_SCHEMA_VERSION = "scheduler-batch-summary.v1"
CORE_COLORS = (
    "#e5e7eb",
    "#2563eb",
    "#14b8a6",
    "#84cc16",
    "#f59e0b",
    "#f97316",
    "#ef4444",
    "#a855f7",
    "#111827",
)


class BatchReportError(RuntimeError):
    """Raised when input reports cannot produce a trustworthy batch summary."""


@dataclass(frozen=True)
class TaskSummary:
    task_name: str
    source_trial: str
    report_path: Path
    wall_sec: float
    cpu_usage_sec: float
    average_logical_cores_used: float
    scheduler_average_logical_cores_used: float
    peak_simultaneous_logical_cores: int
    allocated_logical_cpus: float
    time_sec_by_cores: dict[int, float]
    source_outcome_mismatch_count: int
    lost_perf_events: int
    replay_report_path: Path | None
    scheduler_events_path: Path | None

    @property
    def utilization_percent_of_allocation(self) -> float:
        return self.average_logical_cores_used / self.allocated_logical_cpus * 100.0

    @property
    def time_above_one_core_sec(self) -> float:
        return sum(value for cores, value in self.time_sec_by_cores.items() if cores > 1)


@dataclass(frozen=True)
class ShellCommandSummary:
    task_name: str
    action_index: int
    session_id: str
    poll_count: int
    command: str
    command_preview: str
    command_family: str
    lifecycle_start_monotonic_ns: int
    lifecycle_end_monotonic_ns: int
    lifecycle_wall_sec: float
    status: str
    source_status: str
    peak_simultaneous_logical_cores: int
    average_simultaneous_logical_cores: float
    time_at_or_above_2_cores_sec: float
    time_at_or_above_4_cores_sec: float
    time_at_or_above_6_cores_sec: float
    time_at_or_above_8_cores_sec: float


@dataclass(frozen=True)
class CommandFrequencySummary:
    command: str
    command_preview: str
    invocation_count: int
    task_count: int
    task_names: tuple[str, ...]
    total_lifecycle_wall_sec: float
    total_time_at_or_above_2_cores_sec: float
    max_peak_simultaneous_logical_cores: int


@dataclass(frozen=True)
class ToolFamilySummary:
    command_family: str
    invocation_count: int
    invocation_percent_of_commands: float
    task_count: int
    task_names: tuple[str, ...]
    summed_invocation_lifecycle_wall_sec: float
    active_lifecycle_wall_sec: float
    overlap_collapsed_wall_sec: float
    active_lifecycle_percent_of_replay_wall: float
    active_time_at_or_above_2_cores_sec: float
    active_time_at_or_above_4_cores_sec: float
    active_time_at_or_above_8_cores_sec: float
    max_peak_simultaneous_logical_cores: int
    command_examples: tuple[str, ...]


def _number(value: Any, *, field: str, path: Path) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise BatchReportError(f"{path} has invalid numeric field {field!r}")
    return float(value)


def load_summary(path: Path) -> TaskSummary:
    """Load one task only after enforcing the comparable-report contract."""
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchReportError(f"unable to read pipeline report {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise BatchReportError(f"pipeline report is not an object: {path}")
    if "status" in payload and payload["status"] != "valid":
        raise BatchReportError(f"pipeline report is not valid for profiling: {path}")
    replay = payload.get("replay_summary")
    scheduler = payload.get("scheduler_concurrency")
    cpu = payload.get("container_cpu")
    if not isinstance(replay, dict) or not isinstance(scheduler, dict) or not isinstance(cpu, dict):
        raise BatchReportError(f"pipeline report lacks replay, CPU, or scheduler data: {path}")
    mismatch_count = replay.get("source_outcome_mismatch_count")
    if mismatch_count != 0:
        raise BatchReportError(f"pipeline report has {mismatch_count} replay outcome mismatches: {path}")
    unavailable_count = replay.get("source_outcome_comparison_unavailable_count", 0)
    if not isinstance(unavailable_count, int) or isinstance(unavailable_count, bool) or unavailable_count != 0:
        raise BatchReportError(f"pipeline report has replay source outcome comparisons unavailable: {path}")
    for field in (
        "actions_timed_out",
        "actions_launch_failed",
        "actions_source_argument_parse_failed",
        "actions_skipped_after_timeout",
    ):
        if field in replay and (type(replay[field]) is not int or replay[field] != 0):
            raise BatchReportError(f"pipeline replay {field} is nonzero or invalid: {path}")
    if "tool_actions_recorded" in replay:
        recorded = replay["tool_actions_recorded"]
        replayed = replay.get("tool_actions_replayed")
        if type(recorded) is not int or recorded <= 0 or type(replayed) is not int or replayed != recorded:
            raise BatchReportError(f"pipeline replay tool actions are incomplete or invalid: {path}")
    validation = scheduler.get("cgroup_validation")
    if not isinstance(validation, dict) or not validation.get("valid_for_concurrency_reporting"):
        raise BatchReportError(f"scheduler concurrency is not cgroup-validated: {path}")
    distribution = scheduler.get("time_sec_by_simultaneous_logical_cores")
    if not isinstance(distribution, dict):
        raise BatchReportError(f"pipeline report lacks scheduler distribution: {path}")
    time_sec_by_cores: dict[int, float] = {}
    for key, value in distribution.items():
        try:
            cores = int(key)
        except (TypeError, ValueError) as exc:
            raise BatchReportError(f"{path} has invalid concurrency bucket {key!r}") from exc
        time_sec_by_cores[cores] = _number(value, field=f"distribution[{key}]", path=path)
    return TaskSummary(
        task_name=str(payload.get("task_name") or ""),
        source_trial=str(payload.get("source_trial") or ""),
        report_path=path.resolve(),
        wall_sec=_number(cpu.get("wall_sec"), field="container_cpu.wall_sec", path=path),
        cpu_usage_sec=_number(cpu.get("usage_sec"), field="container_cpu.usage_sec", path=path),
        average_logical_cores_used=_number(
            cpu.get("average_logical_cores_used"),
            field="container_cpu.average_logical_cores_used",
            path=path,
        ),
        scheduler_average_logical_cores_used=_number(
            scheduler.get("average_logical_cores_used"),
            field="scheduler_concurrency.average_logical_cores_used",
            path=path,
        ),
        peak_simultaneous_logical_cores=int(
            _number(
                scheduler.get("peak_simultaneous_logical_cores"),
                field="scheduler_concurrency.peak_simultaneous_logical_cores",
                path=path,
            )
        ),
        allocated_logical_cpus=_number(
            cpu.get("allocated_logical_cpus"),
            field="container_cpu.allocated_logical_cpus",
            path=path,
        ),
        time_sec_by_cores=time_sec_by_cores,
        source_outcome_mismatch_count=int(mismatch_count),
        lost_perf_events=int(scheduler.get("lost_perf_events") or 0),
        replay_report_path=(Path(str(payload["replay_report"])).resolve() if payload.get("replay_report") else None),
        scheduler_events_path=(Path(str(scheduler["events_path"])).resolve() if scheduler.get("events_path") else None),
    )


def build_summary(paths: list[Path]) -> list[TaskSummary]:
    if not paths:
        raise BatchReportError("provide at least one pipeline report")
    tasks = [load_summary(path) for path in paths]
    names = [task.task_name for task in tasks]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise BatchReportError("duplicate task reports: " + ", ".join(duplicates))
    return sorted(tasks, key=lambda task: task.task_name)


def _load_scheduler_events(task: TaskSummary) -> list[dict[str, Any]]:
    if task.scheduler_events_path is None:
        raise BatchReportError(f"{task.task_name} lacks scheduler-events path")
    try:
        return sorted(
            (json.loads(line) for line in task.scheduler_events_path.read_text().splitlines() if line.strip()),
            key=lambda event: int(event["timestamp_ns"]),
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        raise BatchReportError(f"unable to read scheduler events for {task.task_name}: {exc}") from exc


def _load_replay_actions(task: TaskSummary) -> list[dict[str, Any]]:
    if task.replay_report_path is None:
        raise BatchReportError(f"{task.task_name} lacks replay-report path")
    try:
        replay = json.loads(task.replay_report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchReportError(f"unable to read replay report for {task.task_name}: {exc}") from exc
    actions = [
        action
        for action in replay.get("actions", [])
        if action.get("status") != "ignored_metadata"
        and "replay_start_monotonic_ns" in action
        and "replay_end_monotonic_ns" in action
    ]
    if not actions:
        raise BatchReportError(f"{task.task_name} replay report has no timed executable actions")
    return actions


def _integrate_event_window(
    events: list[dict[str, Any]], *, window_start_ns: int, window_end_ns: int
) -> dict[int, float]:
    """Integrate simultaneous scheduled CPUs over a command or task window."""
    if window_end_ns <= window_start_ns:
        return {0: 0.0}
    active_cpus: set[int] = set()
    for event in events:
        timestamp_ns = int(event["timestamp_ns"])
        if timestamp_ns >= window_start_ns:
            break
        cpu = int(event["cpu"])
        if event["entering"]:
            active_cpus.add(cpu)
        else:
            active_cpus.discard(cpu)
    duration_ns_by_cores: dict[int, int] = defaultdict(int)
    cursor_ns = window_start_ns
    for event in events:
        timestamp_ns = int(event["timestamp_ns"])
        if timestamp_ns < window_start_ns:
            continue
        if timestamp_ns > window_end_ns:
            break
        if timestamp_ns > cursor_ns:
            duration_ns_by_cores[len(active_cpus)] += timestamp_ns - cursor_ns
        cpu = int(event["cpu"])
        if event["entering"]:
            active_cpus.add(cpu)
        else:
            active_cpus.discard(cpu)
        cursor_ns = timestamp_ns
    if window_end_ns > cursor_ns:
        duration_ns_by_cores[len(active_cpus)] += window_end_ns - cursor_ns
    return {cores: duration_ns / 1_000_000_000 for cores, duration_ns in sorted(duration_ns_by_cores.items())}


def _command_preview(action: dict[str, Any]) -> tuple[str, str]:
    command = str(action.get("command") or "")
    if command:
        normalized = " ".join(command.split())
        return normalized, normalized[:110].rstrip()
    return "", ""


_HOUSEKEEPING_FAMILIES = frozenset({":", "cd", "chmod", "cp", "false", "mkdir", "printf", "rm", "set", "touch", "true"})


def _segment_family(segment: str) -> str:
    try:
        tokens = shlex.split(segment, posix=True)
    except ValueError:
        tokens = segment.split()
    if not tokens:
        return ""
    while tokens and re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", tokens[0]):
        tokens.pop(0)
    if not tokens:
        return "environment assignment"
    if tokens[0] in {"time", "/usr/bin/time"}:
        index = 1
        while index < len(tokens) and tokens[index].startswith("-"):
            index += 1
        tokens = tokens[index:]
        if not tokens:
            return "time"
    executable = Path(tokens[0]).name
    rest = tokens[1:]
    if executable == "set":
        return "set"
    if executable in {"python", "python3"} or re.fullmatch(r"python\d+(?:\.\d+)?", executable):
        if len(rest) >= 3 and rest[:2] == ["-m", "pip"] and rest[2] == "install":
            return "python -m pip install"
        if len(rest) >= 2 and rest[:2] == ["-m", "pytest"]:
            return "python -m pytest"
        if len(rest) >= 2 and rest[:2] == ["-m", "py_compile"]:
            return "python -m py_compile"
        if len(rest) >= 2 and Path(rest[0]).name == "setup.py" and rest[1] == "build_ext":
            return "python setup.py build_ext"
        if rest and rest[0] == "-":
            return "python inline script"
        if rest and rest[0].endswith(".py"):
            return f"python {Path(rest[0]).name}"
        return "python"
    if executable in {"pip", "pip3"} and rest and rest[0] == "install":
        return "pip install"
    if (
        executable == "go"
        and rest
        and rest[0]
        in {
            "build",
            "fmt",
            "generate",
            "install",
            "run",
            "test",
            "vet",
        }
    ):
        return f"go {rest[0]}"
    if executable == "cargo" and rest and rest[0] in {"bench", "build", "check", "run", "test"}:
        return f"cargo {rest[0]}"
    if executable == "npm" and rest:
        if rest[0] == "run" and len(rest) > 1:
            return f"npm run {rest[1]}"
        if rest[0] in {"ci", "install", "test"}:
            return f"npm {rest[0]}"
    if executable == "git" and rest:
        return f"git {rest[0]}"
    if executable == "node":
        if rest and rest[0] == "--check":
            return "node --check"
        if rest and rest[0].endswith(".js"):
            return f"node {Path(rest[0]).name}"
        return "node"
    if executable == "uv" and len(rest) >= 2 and rest[:2] == ["run", "python"]:
        if len(rest) >= 3 and rest[2].endswith(".py"):
            return f"uv run python {Path(rest[2]).name}"
        return "uv run python"
    if executable in {"g++", "gcc", "clang", "clang++"}:
        return f"{executable} compile"
    if executable == "vim":
        return "vim batch edit"
    if executable in {"make", "cmake", "valgrind", "sqlite3", "pytest"}:
        return executable
    if executable.startswith("/") and not rest:
        return "program run"
    return executable


def _shell_segments(command: str) -> tuple[list[str], list[str]]:
    heredoc = re.search(
        r"(?m)^.*<<-?\s*['\"]?[A-Za-z_][A-Za-z0-9_]*['\"]?.*$",
        command,
    )
    if heredoc is not None:
        command = command[: heredoc.end()]
    try:
        lexer = shlex.shlex(
            command.replace("\n", ";"),
            posix=True,
            punctuation_chars="|&;",
        )
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        return [command], []
    segments: list[str] = []
    operators: list[str] = []
    current: list[str] = []
    for token in tokens:
        if token and set(token) <= {"|", "&", ";"}:
            if current:
                segments.append(shlex.join(current))
                current = []
            operators.append(token)
        else:
            current.append(token)
    if current:
        segments.append(shlex.join(current))
    return segments, operators


def classify_command_family(command: str) -> str:
    """Classify recorded shell text without pretending compounds are one tool.

    This operates on recorded top-level shell actions. It is intentionally not
    a Bash parser or a subcommand timer; recursive shell tracing is required to
    split wall time inside compound and pipeline actions.
    """
    normalized = " ".join(command.split())
    if not normalized:
        return "<empty>"
    segments, operators = _shell_segments(command)
    families = [family for family in (_segment_family(segment) for segment in segments) if family]
    meaningful = [family for family in families if family not in _HOUSEKEEPING_FAMILIES]
    selected = meaningful or families
    unique = list(dict.fromkeys(selected))
    if not unique:
        return "<unclassified>"
    if len(unique) == 1:
        return unique[0]
    if operators and all(operator == "|" for operator in operators):
        return f"{unique[0]} pipeline"
    return "multi-tool: " + " + ".join(unique[:3]) + (" + ..." if len(unique) > 3 else "")


def build_command_summary(tasks: list[TaskSummary]) -> list[ShellCommandSummary]:
    """Attribute async poll windows back to the originating shell command.

    Command lifetimes may overlap when a trajectory starts a background
    command and inspects other state before polling completion. Rows are
    therefore suitable for per-command attribution and ranking, not additive
    task wall-time accounting.
    """
    rows: list[ShellCommandSummary] = []
    for task in tasks:
        events = _load_scheduler_events(task)
        actions = _load_replay_actions(task)
        for action in actions:
            if action.get("tool") != "exec_command":
                continue
            start_ns = int(action["replay_start_monotonic_ns"])
            end_ns = int(action["replay_end_monotonic_ns"])
            session_id = str(action.get("session_id") or "")
            polls = [
                followup
                for followup in actions
                if session_id
                and followup.get("tool") == "write_stdin"
                and str(followup.get("session_id") or "") == session_id
                and int(followup["action_index"]) > int(action["action_index"])
            ]
            if polls:
                end_ns = max(end_ns, *(int(followup["replay_end_monotonic_ns"]) for followup in polls))
            durations = _integrate_event_window(events, window_start_ns=start_ns, window_end_ns=end_ns)
            wall_sec = max(0.0, (end_ns - start_ns) / 1_000_000_000)
            scheduled_sec = sum(cores * seconds for cores, seconds in durations.items())
            command, preview = _command_preview(action)
            terminal_action = polls[-1] if polls else action
            rows.append(
                ShellCommandSummary(
                    task_name=task.task_name,
                    action_index=int(action["action_index"]),
                    session_id=session_id,
                    poll_count=len(polls),
                    command=command,
                    command_preview=preview,
                    command_family=classify_command_family(str(action.get("command") or "")),
                    lifecycle_start_monotonic_ns=start_ns,
                    lifecycle_end_monotonic_ns=end_ns,
                    lifecycle_wall_sec=wall_sec,
                    status=str(terminal_action.get("status") or ""),
                    source_status=str(terminal_action.get("source_status") or ""),
                    peak_simultaneous_logical_cores=max(durations, default=0),
                    average_simultaneous_logical_cores=(scheduled_sec / wall_sec if wall_sec else 0.0),
                    time_at_or_above_2_cores_sec=sum(seconds for cores, seconds in durations.items() if cores >= 2),
                    time_at_or_above_4_cores_sec=sum(seconds for cores, seconds in durations.items() if cores >= 4),
                    time_at_or_above_6_cores_sec=sum(seconds for cores, seconds in durations.items() if cores >= 6),
                    time_at_or_above_8_cores_sec=sum(seconds for cores, seconds in durations.items() if cores >= 8),
                )
            )
    return rows


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for start_ns, end_ns in sorted(intervals):
        if not merged or start_ns > merged[-1][1]:
            merged.append((start_ns, end_ns))
        else:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end_ns))
    return merged


def build_tool_family_summary(tasks: list[TaskSummary], commands: list[ShellCommandSummary]) -> list[ToolFamilySummary]:
    """Aggregate semantic command families without double-counting same-family overlap."""
    grouped: dict[str, list[ShellCommandSummary]] = defaultdict(list)
    for command in commands:
        grouped[command.command_family].append(command)
    task_by_name = {task.task_name: task for task in tasks}
    total_wall_sec = sum(task.wall_sec for task in tasks)
    summaries: list[ToolFamilySummary] = []
    for family, invocations in grouped.items():
        intervals_by_task: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for invocation in invocations:
            intervals_by_task[invocation.task_name].append(
                (
                    invocation.lifecycle_start_monotonic_ns,
                    invocation.lifecycle_end_monotonic_ns,
                )
            )
        active_sec = 0.0
        summed_invocation_sec = sum(invocation.lifecycle_wall_sec for invocation in invocations)
        burst_2_sec = 0.0
        burst_4_sec = 0.0
        burst_8_sec = 0.0
        peak = 0
        for task_name, intervals in intervals_by_task.items():
            events = _load_scheduler_events(task_by_name[task_name])
            for start_ns, end_ns in _merge_intervals(intervals):
                active_sec += (end_ns - start_ns) / 1_000_000_000
                durations = _integrate_event_window(events, window_start_ns=start_ns, window_end_ns=end_ns)
                peak = max(peak, max(durations, default=0))
                burst_2_sec += sum(seconds for cores, seconds in durations.items() if cores >= 2)
                burst_4_sec += sum(seconds for cores, seconds in durations.items() if cores >= 4)
                burst_8_sec += sum(seconds for cores, seconds in durations.items() if cores >= 8)
        examples = tuple(
            dict.fromkeys(
                invocation.command_preview
                for invocation in sorted(invocations, key=lambda item: item.lifecycle_wall_sec, reverse=True)
            )
        )[:3]
        summaries.append(
            ToolFamilySummary(
                command_family=family,
                invocation_count=len(invocations),
                invocation_percent_of_commands=(len(invocations) / len(commands) * 100.0 if commands else 0.0),
                task_count=len(intervals_by_task),
                task_names=tuple(sorted(intervals_by_task)),
                summed_invocation_lifecycle_wall_sec=summed_invocation_sec,
                active_lifecycle_wall_sec=active_sec,
                overlap_collapsed_wall_sec=max(0.0, summed_invocation_sec - active_sec),
                active_lifecycle_percent_of_replay_wall=(
                    active_sec / total_wall_sec * 100.0 if total_wall_sec else 0.0
                ),
                active_time_at_or_above_2_cores_sec=burst_2_sec,
                active_time_at_or_above_4_cores_sec=burst_4_sec,
                active_time_at_or_above_8_cores_sec=burst_8_sec,
                max_peak_simultaneous_logical_cores=peak,
                command_examples=examples,
            )
        )
    return sorted(
        summaries,
        key=lambda row: (-row.active_lifecycle_wall_sec, -row.invocation_count, row.command_family),
    )


def build_command_frequency(
    commands: list[ShellCommandSummary],
) -> list[CommandFrequencySummary]:
    grouped: dict[str, list[ShellCommandSummary]] = defaultdict(list)
    for command in commands:
        grouped[command.command].append(command)
    frequency = [
        CommandFrequencySummary(
            command=command,
            command_preview=command[:110].rstrip(),
            invocation_count=len(invocations),
            task_count=len({invocation.task_name for invocation in invocations}),
            task_names=tuple(sorted({invocation.task_name for invocation in invocations})),
            total_lifecycle_wall_sec=sum(invocation.lifecycle_wall_sec for invocation in invocations),
            total_time_at_or_above_2_cores_sec=sum(
                invocation.time_at_or_above_2_cores_sec for invocation in invocations
            ),
            max_peak_simultaneous_logical_cores=max(
                invocation.peak_simultaneous_logical_cores for invocation in invocations
            ),
        )
        for command, invocations in grouped.items()
    ]
    return sorted(
        frequency,
        key=lambda row: (-row.invocation_count, -row.total_lifecycle_wall_sec, row.command),
    )


def _svg_text(x: float, y: float, text: str, **attributes: str) -> str:
    attrs = " ".join(f'{key.replace("_", "-")}="{html.escape(value)}"' for key, value in attributes.items())
    return f'<text x="{x:.1f}" y="{y:.1f}" {attrs}>{html.escape(text)}</text>'


def render_average_vs_peak(tasks: list[TaskSummary], output: Path) -> None:
    allocation = max(task.allocated_logical_cpus for task in tasks)
    left, top, chart_width, row_height = 245, 70, 680, 42
    width = 1080
    height = top + row_height * len(tasks) + 74
    scale = chart_width / allocation
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(
            22,
            30,
            "Average CPU Use vs Peak Simultaneous Logical CPUs",
            fill="#111827",
            font_size="19",
            font_weight="bold",
        ),
        _svg_text(
            22,
            53,
            "Validated one-at-a-time BCC replays; blue bar = average, red marker = observed peak",
            fill="#4b5563",
            font_size="12",
        ),
    ]
    for tick in range(0, int(allocation) + 1):
        x = left + tick * scale
        svg.append(f'<line x1="{x:.1f}" y1="{top - 14}" x2="{x:.1f}" y2="{height - 50}" stroke="#e5e7eb"/>')
        svg.append(_svg_text(x - 3, height - 28, str(tick), fill="#374151", font_size="11"))
    for index, task in enumerate(tasks):
        center_y = top + index * row_height + 15
        svg.append(_svg_text(22, center_y + 4, task.task_name, fill="#111827", font_size="12"))
        svg.append(
            f'<rect x="{left}" y="{center_y - 10}" width="{task.average_logical_cores_used * scale:.1f}" '
            'height="19" rx="3" fill="#2563eb"/>'
        )
        peak_x = left + task.peak_simultaneous_logical_cores * scale
        svg.append(f'<circle cx="{peak_x:.1f}" cy="{center_y - 1}" r="6" fill="#dc2626"/>')
        svg.append(
            _svg_text(
                left + chart_width + 17,
                center_y + 4,
                f"avg {task.average_logical_cores_used:.2f}  peak {task.peak_simultaneous_logical_cores}",
                fill="#374151",
                font_size="11",
            )
        )
    svg.append(_svg_text(left + chart_width / 2 - 45, height - 8, "Logical CPUs", fill="#374151", font_size="12"))
    svg.append("</svg>\n")
    output.write_text("\n".join(svg))


def render_distribution(tasks: list[TaskSummary], output: Path) -> None:
    max_bucket = max(max(task.time_sec_by_cores) for task in tasks)
    left, top, chart_width, row_height = 245, 90, 700, 43
    width = 1110
    height = top + row_height * len(tasks) + 86
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        _svg_text(
            22, 30, "Per-Task Simultaneous Logical-CPU Distribution", fill="#111827", font_size="19", font_weight="bold"
        ),
        _svg_text(
            22,
            52,
            "Each bar is 100% of measured wall time; segments indicate concurrent scheduled logical CPUs",
            fill="#4b5563",
            font_size="12",
        ),
    ]
    legend_x = 22
    for cores in range(max_bucket + 1):
        color = CORE_COLORS[min(cores, len(CORE_COLORS) - 1)]
        svg.append(f'<rect x="{legend_x}" y="65" width="13" height="13" fill="{color}"/>')
        svg.append(_svg_text(legend_x + 18, 76, str(cores), fill="#374151", font_size="11"))
        legend_x += 42
    for index, task in enumerate(tasks):
        y = top + index * row_height
        svg.append(_svg_text(22, y + 19, task.task_name, fill="#111827", font_size="12"))
        total = sum(task.time_sec_by_cores.values())
        offset = 0.0
        for cores in range(max_bucket + 1):
            value = task.time_sec_by_cores.get(cores, 0.0)
            segment_width = 0 if total == 0 else value / total * chart_width
            if segment_width > 0:
                color = CORE_COLORS[min(cores, len(CORE_COLORS) - 1)]
                svg.append(
                    f'<rect x="{left + offset:.2f}" y="{y + 4}" width="{segment_width:.2f}" '
                    f'height="22" fill="{color}"/>'
                )
            offset += segment_width
        svg.append(
            _svg_text(
                left + chart_width + 15,
                y + 19,
                f">1: {task.time_above_one_core_sec * 1000:.1f} ms",
                fill="#374151",
                font_size="11",
            )
        )
    svg.append(_svg_text(left + chart_width / 2 - 40, height - 14, "Wall time", fill="#374151", font_size="12"))
    svg.append("</svg>\n")
    output.write_text("\n".join(svg))


def _matplotlib_pyplot() -> Any:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as pyplot
    except ImportError as exc:
        raise BatchReportError(
            "PNG chart rendering requires matplotlib; run with "
            "`uv run --with matplotlib python -m offline_replayer report render-batch --png ...`"
        ) from exc
    return pyplot


def render_average_vs_peak_png(tasks: list[TaskSummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    labels = [task.task_name for task in tasks]
    averages = [task.average_logical_cores_used for task in tasks]
    peaks = [task.peak_simultaneous_logical_cores for task in tasks]
    allocation = max(task.allocated_logical_cpus for task in tasks)
    y_positions = list(range(len(tasks)))
    figure, axis = pyplot.subplots(figsize=(12, 0.5 * len(tasks) + 2.2), dpi=200)
    axis.barh(y_positions, averages, color="#2563eb", label="Average logical CPUs used")
    axis.scatter(peaks, y_positions, color="#dc2626", s=32, zorder=3, label="Peak simultaneous CPUs")
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, allocation + 1.15)
    axis.set_xlabel("Logical CPUs")
    axis.set_title("Average CPU Use vs Peak Simultaneous Logical CPUs")
    axis.grid(axis="x", color="#e5e7eb")
    axis.set_axisbelow(True)
    for y_position, average, peak in zip(y_positions, averages, peaks, strict=False):
        axis.text(allocation + 0.08, y_position, f"avg {average:.2f}  peak {peak}", va="center", fontsize=8)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.09), ncols=2)
    figure.suptitle("Validated one-at-a-time BCC replays on the assigned cpuset", fontsize=9, y=0.985)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def render_distribution_png(tasks: list[TaskSummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    labels = [task.task_name for task in tasks]
    y_positions = list(range(len(tasks)))
    max_bucket = max(max(task.time_sec_by_cores) for task in tasks)
    figure, axis = pyplot.subplots(figsize=(12, 0.5 * len(tasks) + 2.5), dpi=200)
    left = [0.0] * len(tasks)
    for cores in range(max_bucket + 1):
        fractions = []
        for task in tasks:
            total = sum(task.time_sec_by_cores.values())
            value = task.time_sec_by_cores.get(cores, 0.0)
            fractions.append(value / total * 100.0 if total else 0.0)
        axis.barh(
            y_positions,
            fractions,
            left=left,
            color=CORE_COLORS[min(cores, len(CORE_COLORS) - 1)],
            label=str(cores),
        )
        left = [position + fraction for position, fraction in zip(left, fractions, strict=False)]
    axis.set_yticks(y_positions, labels)
    axis.invert_yaxis()
    axis.set_xlim(0, 112)
    axis.set_xlabel("Percent of measured wall time")
    axis.set_title("Per-Task Simultaneous Logical-CPU Distribution")
    axis.grid(axis="x", color="#e5e7eb")
    axis.set_axisbelow(True)
    for y_position, task in zip(y_positions, tasks, strict=False):
        axis.text(101.0, y_position, f">1: {task.time_above_one_core_sec * 1000:.1f} ms", va="center", fontsize=8)
    axis.legend(title="Logical CPUs active", ncols=max_bucket + 1, loc="lower center", bbox_to_anchor=(0.5, -0.29))
    figure.suptitle("Each stacked bar is 100% of its measured task window", fontsize=9, y=0.985)
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def _load_timeline(task: TaskSummary) -> tuple[list[float], list[int], float]:
    if task.replay_report_path is None or task.scheduler_events_path is None:
        raise BatchReportError(f"{task.task_name} lacks replay-report or scheduler-events paths for timeline rendering")
    try:
        replay = json.loads(task.replay_report_path.read_text())
        events = [json.loads(line) for line in task.scheduler_events_path.read_text().splitlines() if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise BatchReportError(f"unable to read timeline inputs for {task.task_name}: {exc}") from exc
    actions = [
        action
        for action in replay.get("actions", [])
        if "replay_start_monotonic_ns" in action and "replay_end_monotonic_ns" in action
    ]
    if not actions:
        raise BatchReportError(f"{task.task_name} replay report has no timed action windows")
    window_start_ns = min(int(action["replay_start_monotonic_ns"]) for action in actions)
    window_end_ns = max(int(action["replay_end_monotonic_ns"]) for action in actions)
    if window_end_ns <= window_start_ns:
        raise BatchReportError(f"{task.task_name} action timeline has no positive duration")
    sorted_events = sorted(events, key=lambda event: int(event["timestamp_ns"]))
    active_cpus: set[int] = set()
    for event in sorted_events:
        timestamp_ns = int(event["timestamp_ns"])
        if timestamp_ns >= window_start_ns:
            break
        cpu = int(event["cpu"])
        if event["entering"]:
            active_cpus.add(cpu)
        else:
            active_cpus.discard(cpu)
    seconds = [0.0]
    running = [len(active_cpus)]
    for event in sorted_events:
        timestamp_ns = int(event["timestamp_ns"])
        if timestamp_ns < window_start_ns or timestamp_ns > window_end_ns:
            continue
        elapsed_sec = (timestamp_ns - window_start_ns) / 1_000_000_000
        seconds.append(elapsed_sec)
        running.append(len(active_cpus))
        cpu = int(event["cpu"])
        if event["entering"]:
            active_cpus.add(cpu)
        else:
            active_cpus.discard(cpu)
        seconds.append(elapsed_sec)
        running.append(len(active_cpus))
    duration_sec = (window_end_ns - window_start_ns) / 1_000_000_000
    seconds.append(duration_sec)
    running.append(len(active_cpus))
    return seconds, running, duration_sec


def render_timeline_png(tasks: list[TaskSummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    allocation = int(max(task.allocated_logical_cpus for task in tasks))
    figure, axes = pyplot.subplots(
        len(tasks),
        1,
        figsize=(15, max(10, len(tasks) * 1.62)),
        dpi=200,
        constrained_layout=True,
    )
    if len(tasks) == 1:
        axes = [axes]
    for axis, task in zip(axes, tasks, strict=False):
        seconds, running, duration_sec = _load_timeline(task)
        axis.plot(seconds, running, color="#2563eb", linewidth=0.85)
        axis.fill_between(seconds, running, step="pre", color="#2563eb", alpha=0.15)
        axis.axhline(
            allocation,
            color="#dc2626",
            linestyle="--",
            linewidth=0.9,
            label=f"Sandbox ceiling: {allocation} logical CPUs",
        )
        axis.set_ylim(-0.25, allocation + 0.55)
        axis.set_yticks(range(0, allocation + 1))
        axis.set_ylabel(task.task_name, rotation=0, ha="right", va="center", fontsize=8, labelpad=10)
        axis.grid(axis="both", color="#e5e7eb", linewidth=0.55)
        axis.set_axisbelow(True)
        axis.text(
            0.995,
            0.73,
            f"{duration_sec:.2f}s  avg {task.average_logical_cores_used:.2f}  peak {task.peak_simultaneous_logical_cores}",
            transform=axis.transAxes,
            ha="right",
            va="center",
            fontsize=7.5,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75},
        )
    axes[0].legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Elapsed seconds during recorded ATIF tool actions")
    figure.suptitle(
        "Per-Task Scheduled Logical-CPU Timeline\n"
        "Validated one-at-a-time BCC replays; each task uses its own time scale",
        fontsize=14,
    )
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def render_command_duration_png(commands: list[ShellCommandSummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    ranked = sorted(commands, key=lambda command: command.lifecycle_wall_sec, reverse=True)[:25]
    labels = [f"{command.task_name}: {command.command_preview[:76]}" for command in reversed(ranked)]
    wall = [command.lifecycle_wall_sec for command in reversed(ranked)]
    burst = [command.time_at_or_above_2_cores_sec for command in reversed(ranked)]
    figure, axis = pyplot.subplots(figsize=(15, 9.5), dpi=200)
    y_positions = list(range(len(ranked)))
    axis.barh(y_positions, wall, color="#2563eb", label="Command lifetime")
    axis.barh(y_positions, burst, color="#f97316", label="Time at >=2 concurrent CPUs")
    axis.set_yticks(y_positions, labels, fontsize=7.5)
    axis.set_xlabel("Seconds")
    axis.set_title("Longest Bash Commands and Their Concurrent-Core Burst Time")
    axis.grid(axis="x", color="#e5e7eb")
    axis.set_axisbelow(True)
    axis.legend()
    figure.suptitle(
        "Async session polls are attributed to their originating exec_command",
        fontsize=9,
        y=0.985,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def render_burstiest_commands_png(commands: list[ShellCommandSummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    ranked = sorted(commands, key=lambda command: command.time_at_or_above_2_cores_sec, reverse=True)[:25]
    labels = [f"{command.task_name}: {command.command_preview[:76]}" for command in reversed(ranked)]
    burst_2 = [command.time_at_or_above_2_cores_sec for command in reversed(ranked)]
    burst_4 = [command.time_at_or_above_4_cores_sec for command in reversed(ranked)]
    burst_8 = [command.time_at_or_above_8_cores_sec for command in reversed(ranked)]
    figure, axis = pyplot.subplots(figsize=(15, 9.5), dpi=200)
    y_positions = list(range(len(ranked)))
    axis.barh(y_positions, burst_2, color="#f59e0b", label=">=2 CPUs")
    axis.barh(y_positions, burst_4, color="#f97316", label=">=4 CPUs")
    axis.barh(y_positions, burst_8, color="#dc2626", label=">=8 CPUs")
    axis.set_yticks(y_positions, labels, fontsize=7.5)
    axis.set_xlabel("Seconds")
    axis.set_title("Bash Commands With the Most Concurrent-Core Demand")
    axis.grid(axis="x", color="#e5e7eb")
    axis.set_axisbelow(True)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def render_command_frequency_png(frequency: list[CommandFrequencySummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    repeated = [row for row in frequency if row.invocation_count > 1][:25]
    figure, axis = pyplot.subplots(figsize=(15, 7.5), dpi=200)
    if not repeated:
        axis.text(
            0.5,
            0.5,
            "No exact bash command string was invoked more than once.",
            transform=axis.transAxes,
            horizontalalignment="center",
            verticalalignment="center",
        )
        axis.set_axis_off()
    else:
        ranked = list(reversed(repeated))
        y_positions = list(range(len(ranked)))
        labels = [row.command_preview[:92] for row in ranked]
        counts = [row.invocation_count for row in ranked]
        axis.barh(y_positions, counts, color="#2563eb")
        axis.set_yticks(y_positions, labels, fontsize=7.5)
        axis.set_xlabel("Number of invocations")
        axis.set_title("Repeated Exact Bash Commands")
        axis.grid(axis="x", color="#e5e7eb")
        axis.set_axisbelow(True)
        for y_position, count in zip(y_positions, counts, strict=False):
            axis.text(count + 0.04, y_position, str(count), va="center", fontsize=8)
    figure.suptitle(
        "Normalized exec_command shell text; asynchronous write_stdin polls are collapsed",
        fontsize=9,
        y=0.985,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def render_tool_family_distribution_png(families: list[ToolFamilySummary], output: Path) -> None:
    pyplot = _matplotlib_pyplot()
    ranked = list(reversed(families[:20]))
    labels = [family.command_family for family in ranked]
    active_percent = [family.active_lifecycle_percent_of_replay_wall for family in ranked]
    invocation_percent = [family.invocation_percent_of_commands for family in ranked]
    burst_sec = [family.active_time_at_or_above_2_cores_sec for family in ranked]
    y_positions = list(range(len(ranked)))
    figure, axes = pyplot.subplots(1, 3, figsize=(20, max(7.5, len(ranked) * 0.4 + 2.1)), dpi=200, sharey=True)
    axes[0].barh(y_positions, active_percent, color="#2563eb")
    axes[0].set_yticks(y_positions, labels, fontsize=8)
    axes[0].set_xlabel("% of measured replay wall")
    axes[0].set_title("Active lifecycle time")
    axes[1].barh(y_positions, invocation_percent, color="#14b8a6")
    axes[1].set_xlabel("% of actual bash commands")
    axes[1].set_title("Invocation mix")
    axes[2].barh(y_positions, burst_sec, color="#f97316")
    axes[2].set_xlabel("Seconds at >=2 CPUs")
    axes[2].set_title("Concurrent-core burst")
    for axis in axes:
        axis.grid(axis="x", color="#e5e7eb")
        axis.set_axisbelow(True)
    figure.suptitle(
        "Semantic Command Families Across Validated Replays\n"
        "Active-time percentages can overlap when background commands remain alive",
        fontsize=14,
    )
    figure.tight_layout()
    figure.savefig(output, bbox_inches="tight")
    pyplot.close(figure)


def write_command_outputs(tasks: list[TaskSummary], output_dir: Path, *, render_png: bool = False) -> dict[str, str]:
    commands = build_command_summary(tasks)
    frequency = build_command_frequency(commands)
    tool_families = build_tool_family_summary(tasks, commands)
    json_path = output_dir / "bash_commands.json"
    csv_path = output_dir / "bash_commands.csv"
    frequency_json_path = output_dir / "bash_command_frequency.json"
    frequency_csv_path = output_dir / "bash_command_frequency.csv"
    family_json_path = output_dir / "tool_family_distribution.json"
    family_csv_path = output_dir / "tool_family_distribution.csv"
    fields = list(ShellCommandSummary.__dataclass_fields__)
    rows = [{field: getattr(command, field) for field in fields} for command in commands]
    json_path.write_text(
        json.dumps(
            {
                "schema_version": "scheduler-bash-command-summary.v1",
                "note": (
                    "Async write_stdin poll intervals are collapsed into the originating "
                    "exec_command lifecycle; overlapping command lifetimes are not additive."
                ),
                "bash_commands": rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    frequency_fields = list(CommandFrequencySummary.__dataclass_fields__)
    frequency_rows = [{field: getattr(command, field) for field in frequency_fields} for command in frequency]
    frequency_json_path.write_text(
        json.dumps(
            {
                "schema_version": "scheduler-bash-command-frequency.v1",
                "note": (
                    "Rows group exact normalized exec_command shell strings; aggregate "
                    "lifecycle totals may overlap for concurrently active commands."
                ),
                "bash_command_frequency": frequency_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with frequency_csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=frequency_fields, lineterminator="\n")
        writer.writeheader()
        for row in frequency_rows:
            writer.writerow({**row, "task_names": ";".join(row["task_names"])})
    family_fields = list(ToolFamilySummary.__dataclass_fields__)
    family_rows = [{field: getattr(family, field) for field in family_fields} for family in tool_families]
    family_json_path.write_text(
        json.dumps(
            {
                "schema_version": "scheduler-tool-family-distribution.v1",
                "classification": (
                    "Semantic top-level shell-action family; categories retain "
                    "multi-tool compound actions when subcommand time is not observed."
                ),
                "timing_note": (
                    "Active lifecycle time is unioned within each family and task; "
                    "overlap_collapsed_wall_sec reports duplicate same-family interval "
                    "time. Different families can still overlap when asynchronous "
                    "commands are active."
                ),
                "denominator": {
                    "actual_bash_commands": len(commands),
                    "measured_replay_wall_sec": sum(task.wall_sec for task in tasks),
                },
                "tool_families": family_rows,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    with family_csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=family_fields, lineterminator="\n")
        writer.writeheader()
        for row in family_rows:
            writer.writerow(
                {
                    **row,
                    "task_names": ";".join(row["task_names"]),
                    "command_examples": " ; ".join(row["command_examples"]),
                }
            )
    outputs = {
        "bash_commands_json": str(json_path.resolve()),
        "bash_commands_csv": str(csv_path.resolve()),
        "bash_command_frequency_json": str(frequency_json_path.resolve()),
        "bash_command_frequency_csv": str(frequency_csv_path.resolve()),
        "tool_family_distribution_json": str(family_json_path.resolve()),
        "tool_family_distribution_csv": str(family_csv_path.resolve()),
    }
    if render_png:
        duration_path = output_dir / "longest_bash_commands.png"
        burst_path = output_dir / "burstiest_bash_commands.png"
        frequency_path = output_dir / "bash_command_frequency.png"
        family_path = output_dir / "tool_family_distribution.png"
        render_command_duration_png(commands, duration_path)
        render_burstiest_commands_png(commands, burst_path)
        render_command_frequency_png(frequency, frequency_path)
        render_tool_family_distribution_png(tool_families, family_path)
        outputs["longest_bash_commands_png"] = str(duration_path.resolve())
        outputs["burstiest_bash_commands_png"] = str(burst_path.resolve())
        outputs["bash_command_frequency_png"] = str(frequency_path.resolve())
        outputs["tool_family_distribution_png"] = str(family_path.resolve())
    return outputs


def write_outputs(tasks: list[TaskSummary], output_dir: Path, *, render_png: bool = False) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "batch_summary.json"
    csv_path = output_dir / "batch_summary.csv"
    average_path = output_dir / "average_vs_peak.svg"
    distribution_path = output_dir / "core_distribution.svg"
    rows = [
        {
            "task_name": task.task_name,
            "source_trial": task.source_trial,
            "pipeline_report": str(task.report_path),
            "wall_sec": task.wall_sec,
            "cpu_usage_sec": task.cpu_usage_sec,
            "average_logical_cores_used": task.average_logical_cores_used,
            "scheduler_average_logical_cores_used": task.scheduler_average_logical_cores_used,
            "allocated_logical_cpus": task.allocated_logical_cpus,
            "utilization_percent_of_allocation": task.utilization_percent_of_allocation,
            "peak_simultaneous_logical_cores": task.peak_simultaneous_logical_cores,
            "time_above_one_core_sec": task.time_above_one_core_sec,
            "time_sec_by_simultaneous_logical_cores": {
                str(cores): seconds for cores, seconds in sorted(task.time_sec_by_cores.items())
            },
            "lost_perf_events": task.lost_perf_events,
        }
        for task in tasks
    ]
    json_path.write_text(
        json.dumps({"schema_version": SUMMARY_SCHEMA_VERSION, "tasks": rows}, indent=2, sort_keys=True) + "\n"
    )
    csv_fields = [
        "task_name",
        "source_trial",
        "pipeline_report",
        "wall_sec",
        "cpu_usage_sec",
        "average_logical_cores_used",
        "scheduler_average_logical_cores_used",
        "allocated_logical_cpus",
        "utilization_percent_of_allocation",
        "peak_simultaneous_logical_cores",
        "time_above_one_core_sec",
        "lost_perf_events",
    ]
    with csv_path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=csv_fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in csv_fields})
    render_average_vs_peak(tasks, average_path)
    render_distribution(tasks, distribution_path)
    outputs = {
        "json": str(json_path.resolve()),
        "csv": str(csv_path.resolve()),
        "average_vs_peak_svg": str(average_path.resolve()),
        "core_distribution_svg": str(distribution_path.resolve()),
    }
    outputs.update(write_command_outputs(tasks, output_dir, render_png=render_png))
    if render_png:
        average_png_path = output_dir / "average_vs_peak.png"
        distribution_png_path = output_dir / "core_distribution.png"
        render_average_vs_peak_png(tasks, average_png_path)
        render_distribution_png(tasks, distribution_png_path)
        timeline_png_path = output_dir / "core_timeline.png"
        render_timeline_png(tasks, timeline_png_path)
        outputs["average_vs_peak_png"] = str(average_png_path.resolve())
        outputs["core_distribution_png"] = str(distribution_png_path.resolve())
        outputs["core_timeline_png"] = str(timeline_png_path.resolve())
    return outputs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render validated scheduler-concurrency pipeline reports into batch SVG charts."
    )
    parser.add_argument("pipeline_reports", nargs="+", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--png", action="store_true", help="Also render PNG charts using matplotlib.")
    args = parser.parse_args(argv)
    try:
        outputs = write_outputs(build_summary(args.pipeline_reports), args.output_dir, render_png=args.png)
    except (BatchReportError, OSError) as exc:
        parser.error(str(exc))
    print(f"Summary JSON: {outputs['json']}")
    print(f"Summary CSV: {outputs['csv']}")
    print(f"Average vs peak chart: {outputs['average_vs_peak_svg']}")
    print(f"Core distribution chart: {outputs['core_distribution_svg']}")
    print(f"Bash commands JSON: {outputs['bash_commands_json']}")
    print(f"Bash commands CSV: {outputs['bash_commands_csv']}")
    print(f"Bash command frequency JSON: {outputs['bash_command_frequency_json']}")
    print(f"Bash command frequency CSV: {outputs['bash_command_frequency_csv']}")
    print(f"Tool family distribution JSON: {outputs['tool_family_distribution_json']}")
    print(f"Tool family distribution CSV: {outputs['tool_family_distribution_csv']}")
    if args.png:
        print(f"Average vs peak PNG: {outputs['average_vs_peak_png']}")
        print(f"Core distribution PNG: {outputs['core_distribution_png']}")
        print(f"Core timeline PNG: {outputs['core_timeline_png']}")
        print(f"Longest bash commands PNG: {outputs['longest_bash_commands_png']}")
        print(f"Burstiest bash commands PNG: {outputs['burstiest_bash_commands_png']}")
        print(f"Bash command frequency PNG: {outputs['bash_command_frequency_png']}")
        print(f"Tool family distribution PNG: {outputs['tool_family_distribution_png']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
