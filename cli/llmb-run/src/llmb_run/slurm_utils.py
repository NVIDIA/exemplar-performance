# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""SLURM utilities for job management."""

import logging
import re
import shlex
import subprocess
from dataclasses import dataclass

logger = logging.getLogger('llmb_run.slurm_utils')

SACCT_TIMEOUT_SECONDS = 30


@dataclass
class SlurmJob:
    job_id: int | None
    job_status: str | None = None
    job_workdir: str | None = None
    llmb_config_path: str | None = None


@dataclass(frozen=True)
class SlurmAccountingRecord:
    job_id: int
    state: str
    elapsed: str
    submit_time: str
    node_list: str
    exit_code: str


def parse_slurm_job_id(raw_job_id: object) -> int:
    """Parse a Slurm job id from sbatch/NeMo output."""
    match = re.match(r'\s*(\d+)', str(raw_job_id or ''))
    if not match:
        raise ValueError(f"Unable to parse Slurm job id from '{raw_job_id}'.")
    return int(match.group(1))


def get_slurm_job_statuses(
    job_ids: list[int], *, start_time: str | None = None
) -> dict[int, SlurmAccountingRecord] | None:
    """Get Slurm accounting records for multiple jobs with one sacct call.

    ``start_time`` bounds the accounting window passed to sacct.
    Returns a dict (possibly empty) on success. Job ids that sacct does not
    know about are simply absent from the dict — sacct does not error for
    unknown ids. Returns None when sacct itself could not be queried (timeout,
    missing binary, non-zero exit), so callers can distinguish "no records
    found" from "could not refresh".
    """
    if not job_ids:
        return {}

    unique_job_ids = sorted({int(job_id) for job_id in job_ids})
    cmd = [
        "sacct",
        "-X",
        "-P",
        "--noheader",
        f"--jobs={','.join(str(job_id) for job_id in unique_job_ids)}",
        "--format=JobIDRaw,State,Elapsed,Submit,NodeList,ExitCode",
    ]
    if start_time:
        cmd.append(f"--starttime={start_time}")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=SACCT_TIMEOUT_SECONDS)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        stderr = getattr(e, 'stderr', '') or str(e)
        logger.warning(f"Unable to refresh Slurm job status with sacct: {stderr}")
        return None

    records: dict[int, SlurmAccountingRecord] = {}
    for line in result.stdout.splitlines():
        if not line.strip():
            continue

        fields = line.rstrip('\n').split('|')
        if len(fields) < 6:
            logger.debug(f"Skipping unexpected sacct output line: {line}")
            continue

        raw_job_id, state, elapsed, submit_time, node_list, exit_code = fields[:6]
        try:
            job_id = parse_slurm_job_id(raw_job_id)
        except ValueError:
            logger.debug(f"Skipping sacct output with invalid job id: {line}")
            continue

        records[job_id] = SlurmAccountingRecord(
            job_id=job_id,
            state=state.strip(),
            elapsed=elapsed.strip(),
            submit_time=submit_time.strip(),
            node_list=node_list.strip(),
            exit_code=exit_code.strip(),
        )

    return records


def get_cluster_name():
    """Get the cluster name from SLURM configuration.

    Returns:
        str: Cluster name from SLURM config, or None if not found or error occurred
    """
    cmd = "scontrol show config"
    try:
        result = subprocess.run(shlex.split(cmd), capture_output=True, text=True, check=True)

        for line in result.stdout.splitlines():
            line = line.strip()
            if line.startswith('ClusterName'):
                # Extract the value after the '=' sign
                parts = line.split('=', 1)
                if len(parts) == 2:
                    cluster_name = parts[1].strip()
                    logger.debug(f"Found cluster name from SLURM config: {cluster_name}")
                    return cluster_name

        logger.debug("ClusterName not found in SLURM config output")
        return None

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running scontrol show config: {e.stderr}")
        return None
