# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unified public command surface for offline replay and resource measurement.

This dispatcher is the supported command interface. Implementation modules
live under domain packages and own their validated argument handling.
"""

from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

from . import __version__


@dataclass(frozen=True)
class Command:
    group: str
    name: str
    module: str
    summary: str


@dataclass(frozen=True)
class TopLevelCommand:
    name: str
    module: str
    summary: str


TOP_LEVEL_COMMANDS = (
    TopLevelCommand(
        "run",
        "runs.execute",
        "Run ATIF tasks from one strict YAML file.",
    ),
)


COMMANDS = (
    Command("profile", "container", "measurement.container", "Measure one container command."),
    Command("profile", "atif", "adapters.atif_profile", "Measure one ATIF trajectory replay."),
    Command("profile", "trace", "adapters.trace_profile", "Measure one TaskTrace trace replay."),
    Command("profile", "transcript", "adapters.transcript_profile", "Measure one transcript trial replay."),
    Command("replay", "atif", "adapters.atif_replay", "Replay an ATIF trajectory."),
    Command("replay", "trace", "adapters.trace_replay", "Replay a TaskTrace trace."),
    Command("replay", "transcript", "adapters.transcript_replay", "Replay a transcript trial."),
    Command("report", "render-batch", "reporting.scheduler_batch", "Render report summaries."),
    Command("report", "render-set", "reporting.report_sets", "Render declared report sets."),
    Command("report", "analyze-tb2", "reporting.tb2_profiles", "Analyze Terminal-Bench reports."),
    Command(
        "report",
        "resource-timeline",
        "reporting.resource_timeline",
        "Render aligned cgroup CPU and memory over time.",
    ),
    Command(
        "report",
        "unified-view",
        "reporting.unified_view",
        "Render every profiler stream on one synchronized view.",
    ),
    Command("report", "ceiling-hitters", "reporting.ceiling_hitters", "Compare ceiling hitters."),
    Command("report", "task-catalog", "reporting.task_catalog", "Export task catalog JSON."),
    Command(
        "report",
        "process-inventory-bpf",
        "reporting.process_bpf_inventory",
        "Build a loss-audited BPF process inventory proof.",
    ),
    Command(
        "report",
        "process-inventory-bpf-repeatability",
        "reporting.process_bpf_repeatability",
        "Compare content-bound BPF process inventory proofs.",
    ),
    Command(
        "report",
        "compare-agentscry",
        "reporting.agentscry_comparison",
        "Compare container CPU accounting with AgentScry.",
    ),
)


def _load_main(module: str) -> Callable[[list[str] | None], int]:
    loaded = importlib.import_module(f".{module}", __package__)
    return loaded.main


def _program_name() -> str:
    if Path(sys.argv[0]).name in {"tasktrace", "tasktrace.exe"}:
        return "tasktrace"
    return "python -m offline_replayer"


def _print_help(group: str | None = None) -> None:
    program = _program_name()
    if group is None:
        print("Replay recorded tool-side work and measure CPU and memory demand.")
        print()
        print(f"usage: {program} run FILE.yaml")
        print(f"       {program} <group> <command> [arguments]")
        print()
        print("commands:")
        for command in TOP_LEVEL_COMMANDS:
            print(f"  {command.name:<19} {command.summary}")
        print()
        print("groups:")
        for name in sorted({command.group for command in COMMANDS}):
            print(f"  {name}")
        print()
        print("examples:")
        print(f"  {program} replay transcript <transcript.jsonl> --workspace <dir>")
        print(f"  {program} profile atif <trajectory.json> --image <image> --output-dir <dir>")
        print()
        print(f"Run `{program} <group> --help` for commands.")
        return
    commands = [command for command in COMMANDS if command.group == group]
    print(f"usage: {program} {group} <command> [arguments]")
    print()
    print("commands:")
    name_width = max(19, *(len(command.name) for command in commands))
    for command in commands:
        print(f"  {command.name:<{name_width}} {command.summary}")


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] in {"-V", "--version"}:
        print(__version__)
        return 0
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help()
        return 0
    command_name = arguments.pop(0)
    top_level = next(
        (command for command in TOP_LEVEL_COMMANDS if command.name == command_name),
        None,
    )
    if top_level is not None:
        program = _program_name()
        original_program = sys.argv[0]
        sys.argv[0] = f"{program} {top_level.name}"
        try:
            return _load_main(top_level.module)(arguments)
        finally:
            sys.argv[0] = original_program
    group = command_name
    groups = {command.group for command in COMMANDS}
    if group not in groups:
        print(f"unknown command group: {group}", file=sys.stderr)
        _print_help()
        return 2
    if not arguments or arguments[0] in {"-h", "--help"}:
        _print_help(group)
        return 0
    name = arguments.pop(0)
    selected = next(
        (command for command in COMMANDS if command.group == group and command.name == name),
        None,
    )
    if selected is None:
        print(f"unknown {group} command: {name}", file=sys.stderr)
        _print_help(group)
        return 2
    original_program = sys.argv[0]
    # Delegate parsing while keeping help and parser errors expressed in the
    # supported product-level command.
    sys.argv[0] = f"{_program_name()} {group} {name}"
    try:
        return _load_main(selected.module)(arguments)
    finally:
        sys.argv[0] = original_program
