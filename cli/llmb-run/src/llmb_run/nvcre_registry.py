# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Find the WorkloadRun that a given nvcre submission created.

The nvcre scheduler records every submission in ``<nemorun_home>/.nvcre_jobs.json``, keyed by
an app id, with the WorkloadRun name and the ``job_dir`` it ran from. That file is shared by
every submission under the same NeMo-Run home, so the newest entry is not necessarily the job
being asked about: callers must identify it by its own directory.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger('llmb_run.nvcre_registry')

REGISTRY_FILE = '.nvcre_jobs.json'


def workloadrun_name_for(local_dir: str | None) -> str | None:
    """The WorkloadRun name registered for the submission whose directory is *local_dir*.

    ``local_dir`` is the task directory ``<nemorun_home>/experiments/<title>/<exp_id>/<task>``
    (or a directory inside it), which is the ``job_dir`` the scheduler recorded. Returns None when the registry or a
    matching entry cannot be found; it never falls back to another submission's entry.
    """
    if not local_dir:
        return None
    task_dir = Path(local_dir)
    # The registry lives in the NeMo-Run home, an ancestor of every task directory.
    registry = next((p / REGISTRY_FILE for p in (task_dir, *task_dir.parents) if (p / REGISTRY_FILE).is_file()), None)
    if registry is None:
        logger.debug(f"no nvcre registry found above {local_dir}")
        return None
    try:
        with open(registry) as f:
            jobs = json.load(f)
    except (OSError, ValueError) as e:
        logger.debug(f"nvcre registry {registry} unreadable: {e}")
        return None

    # A hand-edited or damaged registry must not abort the post-run flow (log pull, history
    # update, WorkloadRun cancel): treat anything but an object of objects as "no entry".
    if not isinstance(jobs, dict):
        logger.debug(f"nvcre registry {registry} has an unexpected shape")
        return None

    match = None
    for entry in jobs.values():
        if not isinstance(entry, dict):
            continue
        job_dir = entry.get('job_dir')
        if not job_dir or not entry.get('workloadrun_name'):
            continue
        job_dir = Path(job_dir)
        # The task directory itself, or a directory inside it.
        if task_dir == job_dir or job_dir in task_dir.parents:
            match = entry['workloadrun_name']  # entries are in submission order; keep the latest
    if match is None:
        logger.debug(f"no nvcre registry entry for job directory {local_dir}")
    return match
