#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Deterministic phase workload for Milestone-1 profiling validation.

The fixture emits its own monotonic phase intervals while producing observable
serial CPU, anonymous-memory, parallel CPU, child-exec, and teardown behavior.
An exclusive sentinel makes a second execution a hard failure, allowing report
and renderer tests to prove they consume retained evidence only.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from offline_replayer.timebase import monotonic_clock_reference

SCHEMA_VERSION = "tasktrace-m1-labeled-phase.v1"
REPLAY_SCHEMA_VERSION = "tasktrace-m1-labeled-replay.v1"


def _append_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
        output.flush()
        os.fsync(output.fileno())


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        json.dump(payload, output, indent=2, sort_keys=True)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


@contextmanager
def _phase(path: Path, name: str, **attributes: Any) -> Iterator[None]:
    start_ns = time.monotonic_ns()
    _append_json(
        path,
        {
            "record_type": "phase_start",
            "phase": name,
            "timestamp_ns": start_ns,
            "attributes": attributes,
        },
    )
    try:
        yield
    finally:
        _append_json(
            path,
            {
                "record_type": "phase_end",
                "phase": name,
                "timestamp_ns": time.monotonic_ns(),
            },
        )


def _busy_for(duration_sec: float) -> str:
    deadline = time.monotonic() + duration_sec
    value = b"tasktrace-m1-phase-fixture"
    iterations = 0
    while time.monotonic() < deadline:
        value = hashlib.sha256(value).digest()
        iterations += 1
    return f"{iterations}:{value.hex()}"


def _parallel_worker(duration_sec: float, start_event: multiprocessing.synchronize.Event) -> None:  # type: ignore[name-defined]
    start_event.wait()
    _busy_for(duration_sec)


def _create_sentinel(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write("executed-once\n")
        output.flush()
        os.fsync(output.fileno())


def run_fixture(
    *,
    phase_log: Path,
    sentinel: Path,
    serial_cpu_sec: float,
    memory_mib: int,
    memory_hold_sec: float,
    parallel_workers: int,
    parallel_cpu_sec: float,
    teardown_hold_sec: float,
    child_binary: str,
    replay_report: Path | None = None,
) -> None:
    _create_sentinel(sentinel)
    fixture_start_ns = time.monotonic_ns()
    clock_reference = monotonic_clock_reference()
    phase_log.parent.mkdir(parents=True, exist_ok=True)
    phase_log.write_text("", encoding="utf-8")
    _append_json(
        phase_log,
        {
            "record_type": "fixture",
            "schema_version": SCHEMA_VERSION,
            "pid": os.getpid(),
            "clock_reference": clock_reference,
            "timestamp_ns": fixture_start_ns,
        },
    )

    with _phase(phase_log, "serial_cpu", requested_duration_sec=serial_cpu_sec):
        _busy_for(serial_cpu_sec)

    allocation: bytearray | None = None
    with _phase(
        phase_log,
        "memory_plateau",
        requested_bytes=memory_mib * 1024 * 1024,
        requested_duration_sec=memory_hold_sec,
    ):
        allocation = bytearray(memory_mib * 1024 * 1024)
        allocation[::4096] = b"x" * (len(allocation) // 4096)
        time.sleep(memory_hold_sec)

    with _phase(
        phase_log,
        "parallel_cpu",
        workers=parallel_workers,
        requested_duration_sec=parallel_cpu_sec,
    ):
        start_event = multiprocessing.Event()
        workers = [
            multiprocessing.Process(
                target=_parallel_worker,
                args=(parallel_cpu_sec, start_event),
            )
            for _ in range(parallel_workers)
        ]
        for worker in workers:
            worker.start()
        start_event.set()
        for worker in workers:
            worker.join()
            if worker.exitcode != 0:
                raise RuntimeError(f"parallel fixture worker {worker.pid} exited {worker.exitcode}")

    with _phase(phase_log, "child_exec", binary=child_binary):
        completed = subprocess.run(
            [child_binary, "tasktrace-m1-child"],
            check=False,
            text=True,
            capture_output=True,
        )
        if completed.returncode != 0 or completed.stdout.strip() != "tasktrace-m1-child":
            raise RuntimeError(
                f"fixture child failed: rc={completed.returncode}, "
                f"stdout={completed.stdout!r}, stderr={completed.stderr!r}"
            )

    with _phase(phase_log, "teardown", requested_duration_sec=teardown_hold_sec):
        allocation = None
        gc.collect()
        time.sleep(teardown_hold_sec)

    fixture_end_ns = time.monotonic_ns()
    _append_json(
        phase_log,
        {
            "record_type": "fixture_end",
            "timestamp_ns": fixture_end_ns,
            "status": "ok",
        },
    )
    if replay_report is not None:
        wall_time_sec = (fixture_end_ns - fixture_start_ns) / 1_000_000_000
        _write_json(
            replay_report,
            {
                "schema_version": REPLAY_SCHEMA_VERSION,
                "clock_reference": clock_reference,
                "task_name": "m1-labeled-phase",
                "actions": [
                    {
                        "action_index": 1,
                        "step_id": 1,
                        "tool": "fixture",
                        "source_tool": "fixture",
                        "command": "run deterministic labeled profiling fixture",
                        "status": "ok",
                        "returncode": 0,
                        "source_returncode": 0,
                        "outcome_matches_source": True,
                        "replay_start_monotonic_ns": fixture_start_ns,
                        "replay_end_monotonic_ns": fixture_end_ns,
                        "wall_time_sec": wall_time_sec,
                    }
                ],
                "summary": {
                    "actions_recorded": 1,
                    "actions_executed": 1,
                    "actions_ok": 1,
                    "actions_failed": 0,
                    "actions_timed_out": 0,
                    "actions_skipped_after_timeout": 0,
                    "actions_skipped_not_executed": 0,
                    "source_outcome_mismatch_action_indexes": [],
                    "source_outcome_mismatch_count": 0,
                    "wall_time_sec": wall_time_sec,
                },
            },
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase-log", type=Path, required=True)
    parser.add_argument("--sentinel", type=Path, required=True)
    parser.add_argument("--serial-cpu-sec", type=float, default=0.35)
    parser.add_argument("--memory-mib", type=int, default=64)
    parser.add_argument("--memory-hold-sec", type=float, default=0.45)
    parser.add_argument("--parallel-workers", type=int, default=4)
    parser.add_argument("--parallel-cpu-sec", type=float, default=0.45)
    parser.add_argument("--teardown-hold-sec", type=float, default=0.25)
    parser.add_argument("--child-binary", default="/bin/echo")
    parser.add_argument(
        "--replay-report",
        type=Path,
        help=(
            "Optional retained one-action replay report for exercising the full "
            "profile pipeline without another workload execution."
        ),
    )
    args = parser.parse_args(argv)
    for name in (
        "serial_cpu_sec",
        "memory_hold_sec",
        "parallel_cpu_sec",
        "teardown_hold_sec",
    ):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.memory_mib <= 0 or args.parallel_workers <= 0:
        parser.error("memory size and worker count must be positive")
    try:
        run_fixture(
            phase_log=args.phase_log,
            sentinel=args.sentinel,
            serial_cpu_sec=args.serial_cpu_sec,
            memory_mib=args.memory_mib,
            memory_hold_sec=args.memory_hold_sec,
            parallel_workers=args.parallel_workers,
            parallel_cpu_sec=args.parallel_cpu_sec,
            teardown_hold_sec=args.teardown_hold_sec,
            child_binary=args.child_binary,
            replay_report=args.replay_report,
        )
    except FileExistsError:
        parser.error(f"execution sentinel already exists: {args.sentinel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
