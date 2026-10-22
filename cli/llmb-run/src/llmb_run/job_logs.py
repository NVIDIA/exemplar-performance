# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Helpers for locating and reading llmb-run job log files."""

from __future__ import annotations

import pathlib
import re
import subprocess
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class JobLogFile:
    path: pathlib.Path
    retry: int | None = None


def find_job_logs(log_dir: str | pathlib.Path, job_id: int) -> list[JobLogFile]:
    """Find retry-numbered workload log files for one Slurm job."""
    directory = pathlib.Path(log_dir)
    if not directory.is_dir():
        raise FileNotFoundError(f"Log directory not found: {directory}")

    pattern = re.compile(rf"^log-.*_{re.escape(str(job_id))}_(\d+)\.out$")
    logs = [
        JobLogFile(path=path, retry=int(match.group(1)))
        for path in directory.iterdir()
        if path.is_file() and (match := pattern.match(path.name))
    ]
    return sorted(logs, key=lambda f: f.retry)


def find_configured_sbatch_logs(log_dir: str | pathlib.Path, job_id: int) -> list[JobLogFile]:
    """Find configured_sbatch logs, preferring workload logs over Slurm stdout."""
    logs = find_job_logs(log_dir, job_id)
    if logs:
        return logs

    slurm_log = pathlib.Path(log_dir) / f"slurm-{job_id}.out"
    if slurm_log.is_file():
        return [JobLogFile(path=slurm_log)]

    return []


def active_job_log(logs: list[JobLogFile]) -> JobLogFile | None:
    if not logs:
        return None
    return logs[-1]


def read_tail(path: str | pathlib.Path, line_count: int) -> str:
    """Read the last line_count lines from a text log file."""
    if line_count < 1:
        raise ValueError("--tail must be at least 1.")

    with pathlib.Path(path).open("r", errors="replace") as f:
        lines = deque(f, maxlen=line_count)

    return "".join(lines).rstrip("\n")


def follow_tail(path: str | pathlib.Path, line_count: int) -> int:
    """Follow a log using the platform tail command."""
    if line_count < 1:
        raise ValueError("--tail must be at least 1.")

    result = subprocess.run(["tail", "-n", str(line_count), "-f", str(path)], check=False)
    return result.returncode
