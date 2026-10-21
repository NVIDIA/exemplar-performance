# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Resolve the cluster-side objects of one nvcre WorkloadRun.

A WorkloadRun is not what runs the pods. nvcre expands it into a ``Workflow``, which creates one
nvcre ``Job`` per group (``group-0``, and overflow groups), each backed by a TrainJob and a JobSet
named ``<job>-workload``. The job names are generated (``...-9b707``), so they cannot be derived
from the WorkloadRun's name, and pods carry only ``jobset.sigs.k8s.io/jobset-name=<job>-workload``.
"""

import json
import logging
import subprocess

logger = logging.getLogger('llmb_run.nvcre_cluster')

_STATUS_FIELDS = ('jobName', 'nvcreJobName', 'workloadJobName', 'internalJobName')
_PRIMARY_GROUP = 'group-0'


def internal_job_names(namespace: str, workloadrun_name: str) -> list[str]:
    """nvcre job names of this WorkloadRun, primary group first. Empty if none can be resolved.

    Scoped to this WorkloadRun: the nvcre ``Job`` objects are found through the
    ``nvcre.nvidia.com/workflow=<WorkloadRun name>`` label (the Workflow carries the WorkloadRun's
    name). The namespace's other jobs are never considered, so in a shared namespace another
    submission's pods or logs cannot be picked up.
    """
    try:
        r = subprocess.run(
            ['kubectl', 'get', 'workloadrun', workloadrun_name, '-n', namespace, '-o', 'json'],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if r.returncode != 0:
            return []
        data = json.loads(r.stdout)
        # Older nvcre versions exposed the name on the WorkloadRun itself.
        for field in _STATUS_FIELDS:
            val = (data.get('status') or {}).get(field)
            if val and val != workloadrun_name:
                return [val]
        val = ((data.get('metadata') or {}).get('labels') or {}).get('nvcre.nvidia.com/job')
        if val and val != workloadrun_name:
            return [val]

        jobs = subprocess.run(
            [
                'kubectl',
                'get',
                'jobs.nvcre.nvidia.com',
                '-n',
                namespace,
                '-l',
                f'nvcre.nvidia.com/workflow={workloadrun_name}',
                '--sort-by=.metadata.creationTimestamp',
                '-o',
                r'jsonpath={range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.nvcre\.nvidia\.com/group}{"\n"}{end}',
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if jobs.returncode != 0:
            return []
        entries = [line.split('\t') for line in jobs.stdout.splitlines() if line.strip()]
        names = [(e[0].strip(), e[1].strip() if len(e) > 1 else '') for e in entries]
        # Creation order, with the primary group (the training ranks) ahead of overflow groups.
        names.sort(key=lambda item: item[1] != _PRIMARY_GROUP)
        return [name for name, _ in names]
    except subprocess.TimeoutExpired:
        logger.warning("nvcre: resolving the jobs of WorkloadRun '%s' timed out", workloadrun_name)
    except Exception as e:
        logger.debug("Could not resolve nvcre jobs for '%s': %s", workloadrun_name, e)
    return []
