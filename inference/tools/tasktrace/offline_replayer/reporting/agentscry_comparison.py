#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare offline container CPU accounting with an AgentScry sidecar trace.

The unprivileged profiler reports the disposable replay container's cgroup v2
``cpu.stat`` usage. AgentScry reports CPU at process-exit events captured by a
privileged host-side eBPF sidecar. These are not automatically equivalent:
AgentScry can miss CPU for descendants that terminate without the exit hook
it uses for USER/SYS values, for example pipeline producers killed by SIGPIPE.

This analyzer roots AgentScry attribution at the profiling container's entry
shell, follows captured descendants, sums ``asfilter --duration`` CPU portions
for each retained exec row, and reports any descendant without exit CPU
accounting. ``asfilter`` apportions one process-exit CPU total across multiple
exec rows for that PID, so dropping duplicate-PID rows would undercount CPU.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA_VERSION = "offline-agentscry-comparison.v1"


class ComparisonError(RuntimeError):
    """Raised when comparison artifacts cannot support an honest result."""


@dataclass(frozen=True)
class ProcessAggregation:
    root_pid: int
    descendant_pids: frozenset[int]
    exit_accounted_pids: frozenset[int]
    cpu_accounted_rows: tuple[dict[str, str], ...]
    missing_rows: list[dict[str, Any]]
    user_sec: float
    system_sec: float

    @property
    def usage_sec(self) -> float:
        return self.user_sec + self.system_sec


def _read_filtered_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise ComparisonError(f"AgentScry filtered CSV does not exist: {path}")
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    required = {"PID", "PPID", "USER", "SYS", "ARGS"}
    fields = set(rows[0]) if rows else set()
    missing = sorted(required - fields)
    if not rows or missing:
        raise ComparisonError("AgentScry CSV is empty or missing required columns: " + ", ".join(missing))
    return rows


def _parse_pid(value: str) -> int | None:
    return int(value) if value.strip().isdigit() else None


def _parse_cpu_seconds(value: str) -> float | None:
    try:
        return float(value) if value.strip() else None
    except ValueError:
        return None


def aggregate_process_tree(rows: list[dict[str, str]], *, root_command_substring: str) -> ProcessAggregation:
    """Aggregate one AgentScry-recorded process tree from filtered exec rows."""
    roots = [row for row in rows if root_command_substring in row.get("ARGS", "")]
    root_pids = {_parse_pid(row.get("PID", "")) for row in roots}
    root_pids.discard(None)
    if len(root_pids) != 1:
        raise ComparisonError(
            f"expected exactly one root PID matching {root_command_substring!r}; " f"found {sorted(root_pids)}"
        )
    root_pid = next(iter(root_pids))
    assert root_pid is not None

    children: dict[int, set[int]] = defaultdict(set)
    rows_by_pid: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        pid = _parse_pid(row.get("PID", ""))
        ppid = _parse_pid(row.get("PPID", ""))
        if pid is None:
            continue
        rows_by_pid[pid].append(row)
        if ppid is not None:
            children[ppid].add(pid)

    descendants = {root_pid}
    pending = deque([root_pid])
    while pending:
        for child in children.get(pending.popleft(), set()):
            if child not in descendants:
                descendants.add(child)
                pending.append(child)

    exit_accounted_pids: set[int] = set()
    cpu_accounted_rows: list[dict[str, str]] = []
    missing_rows: list[dict[str, Any]] = []
    for pid in sorted(descendants):
        exit_rows = [
            row
            for row in rows_by_pid[pid]
            if _parse_cpu_seconds(row.get("USER", "")) is not None
            and _parse_cpu_seconds(row.get("SYS", "")) is not None
        ]
        if exit_rows:
            # asfilter splits one exit_group USER/SYS value proportionally
            # across all retained exec rows for a PID. Summing these portions
            # reconstructs that PID's total; retaining only one row loses CPU.
            exit_accounted_pids.add(pid)
            cpu_accounted_rows.extend(exit_rows)
            continue
        last = rows_by_pid[pid][-1]
        missing_rows.append(
            {
                "pid": pid,
                "comm": last.get("COMM"),
                "args": last.get("ARGS"),
                "duration": last.get("DURATION"),
            }
        )
    user_sec = sum(float(row["USER"]) for row in cpu_accounted_rows)
    system_sec = sum(float(row["SYS"]) for row in cpu_accounted_rows)
    return ProcessAggregation(
        root_pid=root_pid,
        descendant_pids=frozenset(descendants),
        exit_accounted_pids=frozenset(exit_accounted_pids),
        cpu_accounted_rows=tuple(cpu_accounted_rows),
        missing_rows=missing_rows,
        user_sec=user_sec,
        system_sec=system_sec,
    )


def build_comparison_report(
    pipeline_report_path: Path,
    filtered_csv_path: Path,
    *,
    root_command_substring: str,
    tolerance_fraction: float,
) -> dict[str, Any]:
    try:
        pipeline = json.loads(pipeline_report_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ComparisonError(f"unable to read pipeline report: {pipeline_report_path}") from exc
    cpu = pipeline.get("container_cpu")
    if not isinstance(cpu, dict) or not isinstance(cpu.get("usage_sec"), (int, float)):
        raise ComparisonError("pipeline report has no container_cpu.usage_sec")
    usage_sec = float(cpu["usage_sec"])
    wall_sec = float(cpu["wall_sec"])
    aggregation = aggregate_process_tree(
        _read_filtered_csv(filtered_csv_path),
        root_command_substring=root_command_substring,
    )
    delta_sec = aggregation.usage_sec - usage_sec
    delta_fraction = delta_sec / usage_sec if usage_sec else None
    cpu_within_tolerance = delta_fraction is not None and abs(delta_fraction) <= tolerance_fraction
    complete_exit_accounting = not aggregation.missing_rows
    strict_equivalence = cpu_within_tolerance and complete_exit_accounting
    return {
        "schema_version": SCHEMA_VERSION,
        "pipeline_report": str(pipeline_report_path.resolve()),
        "agentscry_filtered_csv": str(filtered_csv_path.resolve()),
        "measurement_scope": {
            "container_cpu": ("Disposable replay container cgroup v2 cpu.stat command interval."),
            "agentscry_cpu": (
                "Privileged host-side AgentScry USER+SYS for captured descendant "
                "PIDs rooted at the replay container entrypoint; sums CPU portions "
                "apportioned across retained exec rows by asfilter --duration."
            ),
        },
        "task_id": pipeline.get("task_id"),
        "trial_name": pipeline.get("trial_name"),
        "image_identity": pipeline.get("image_identity"),
        "container_cpu": {
            "usage_sec": usage_sec,
            "wall_sec": wall_sec,
            "average_logical_cores_used": cpu.get("average_logical_cores_used"),
        },
        "agentscry_cpu": {
            "root_pid": aggregation.root_pid,
            "captured_descendant_pid_count": len(aggregation.descendant_pids),
            "exit_accounted_pid_count": len(aggregation.exit_accounted_pids),
            "cpu_accounted_row_count": len(aggregation.cpu_accounted_rows),
            "missing_exit_accounting_pid_count": len(aggregation.missing_rows),
            "missing_exit_accounting": aggregation.missing_rows,
            "user_sec": round(aggregation.user_sec, 9),
            "system_sec": round(aggregation.system_sec, 9),
            "usage_sec": round(aggregation.usage_sec, 9),
            "average_logical_cores_on_container_wall": round(aggregation.usage_sec / wall_sec, 9) if wall_sec else None,
        },
        "comparison": {
            "agentscry_minus_container_usage_sec": round(delta_sec, 9),
            "agentscry_minus_container_fraction": round(delta_fraction, 9) if delta_fraction is not None else None,
            "tolerance_fraction": tolerance_fraction,
            "cpu_within_tolerance": cpu_within_tolerance,
            "complete_exit_accounting": complete_exit_accounting,
            "equivalent_for_utilization": strict_equivalence,
            "reason": (
                "AgentScry CPU is within tolerance and every captured replay " "descendant has exit accounting."
                if strict_equivalence
                else (
                    "AgentScry aggregate CPU is within tolerance, but replay descendants "
                    "without exit CPU accounting prevent strict equivalence."
                    if cpu_within_tolerance and aggregation.missing_rows
                    else (
                        "AgentScry CPU differs from container cgroup CPU beyond tolerance "
                        "and replay descendants lack exit CPU accounting."
                        if aggregation.missing_rows
                        else "AgentScry CPU differs from container cgroup CPU beyond tolerance."
                    )
                )
            ),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare a profiled offline replay's container CPU report with a "
            "filtered AgentScry process trace from that same run."
        )
    )
    parser.add_argument("pipeline_report", type=Path)
    parser.add_argument("agentscry_filtered_csv", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="JSON comparison report.")
    parser.add_argument(
        "--root-command-substring",
        default="/tasktrace-profile-entrypoint.sh",
        help="Substring identifying the measured replay container entry shell.",
    )
    parser.add_argument(
        "--tolerance-fraction",
        type=float,
        default=0.05,
        help="Maximum fractional CPU delta accepted only when no descendant accounting is missing.",
    )
    args = parser.parse_args(argv)
    if args.tolerance_fraction < 0:
        parser.error("--tolerance-fraction must not be negative")
    try:
        report = build_comparison_report(
            args.pipeline_report,
            args.agentscry_filtered_csv,
            root_command_substring=args.root_command_substring,
            tolerance_fraction=args.tolerance_fraction,
        )
    except ComparisonError as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    agent = report["agentscry_cpu"]
    comparison = report["comparison"]
    print(f"Container CPU seconds: {report['container_cpu']['usage_sec']:.9f}")
    print(f"AgentScry CPU seconds: {agent['usage_sec']:.9f}")
    print(f"AgentScry missing exit-accounted PIDs: {agent['missing_exit_accounting_pid_count']}")
    print(f"CPU within tolerance: {comparison['cpu_within_tolerance']}")
    print(f"Strict equivalent for utilization: {comparison['equivalent_for_utilization']}")
    print(f"Reason: {comparison['reason']}")
    print(f"Report: {args.output}")
    return int(not comparison["equivalent_for_utilization"])


if __name__ == "__main__":
    sys.exit(main())
