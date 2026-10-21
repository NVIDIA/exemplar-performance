# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent llmb-run job history backed by SQLite."""

from __future__ import annotations

import contextlib
import datetime
import json
import logging
import pathlib
import sqlite3
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import yaml
from rich import box
from rich.console import Console
from rich.table import Table

from llmb_run.config_manager import ClusterConfig, ExecutionTarget
from llmb_run.nvcre_cluster import internal_job_names
from llmb_run.nvcre_registry import workloadrun_name_for
from llmb_run.pretrain_log_parser import (
    MAX_ITERATION,
    MIN_ITERATION,
    PretrainLogParseStatus,
    parse_latest_pretrain_job_log,
    parser_name_for_framework,
)
from llmb_run.rl_metrics_parser import parse_rl_metrics
from llmb_run.slurm_utils import SlurmAccountingRecord, SlurmJob, get_slurm_job_statuses, parse_slurm_job_id
from llmb_run.tasks import WorkloadTask

logger = logging.getLogger('llmb_run.job_history')

# MLPerf recipes average a late iteration window to skip warmup (see their READMEs).
MLPERF_ITERATION_WINDOW = (485, 494)
MLPERF_ITERATION_WINDOW_OVERRIDES = {"pretrain_llama3.1_mlperf": (985, 994)}

DB_SCHEMA_VERSION = 3
HISTORY_DIR_NAME = ".llmb"
HISTORY_DB_NAME = "jobs.sqlite3"
# sacct accounting can lag behind sbatch by several seconds; don't mark a job
# PURGED if it was created within this window — sacct just hasn't seen it yet.
PURGE_GRACE_SECONDS = 300
SACCT_JOB_CHUNK_SIZE = 150
SACCT_START_TIME_SAFETY_DAYS = 1
TERMINAL_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "COMPLETED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PURGED",
    "TIMEOUT",
}


@dataclass(frozen=True)
class JobRecord:
    job_id: int
    launcher_type: str
    workload_key: str | None = None
    model_name: str | None = None
    model_size: str | None = None
    dtype: str | None = None
    scale: int | None = None
    profile_enabled: bool = False
    proxy: bool = False
    log_dir: str | None = None
    llmb_config_path: str | None = None
    submit_time: str | None = None
    env_overrides_json: str | None = None
    model_overrides_json: str | None = None
    namespace: str | None = None
    workloadrun_name: str | None = None


@dataclass(frozen=True)
class RebuildStats:
    scanned: int
    imported: int
    skipped: int
    db_path: pathlib.Path
    refresh_error: str | None = None


def get_history_db_path(llmb_install: str | pathlib.Path) -> pathlib.Path:
    return pathlib.Path(llmb_install) / HISTORY_DIR_NAME / HISTORY_DB_NAME


def base_slurm_state(state: str | None) -> str:
    if not state:
        return ""
    return state.strip().split()[0].upper()


def is_terminal_state(state: str | None) -> bool:
    return base_slurm_state(state) in TERMINAL_STATES


@contextlib.contextmanager
def _open_history_db(config: ClusterConfig) -> Iterator[sqlite3.Connection]:
    """Open the history DB, creating the parent dir and ensuring the schema."""
    db_path = get_history_db_path(config.llmb_install)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        _initialize_schema(conn)
        yield conn


def record_job_submission(
    config: ClusterConfig, task: WorkloadTask, slurm_job: SlurmJob, workloads: dict[str, Any]
) -> None:
    """Best-effort record of a successfully submitted primary workload job.

    Uses wall-clock time for submit_time; sacct will overwrite it with the
    canonical Slurm-side value on the next refresh. Logs a warning on any
    failure; never raises.
    """
    if not slurm_job.job_id:
        return

    try:
        job_id = parse_slurm_job_id(slurm_job.job_id)

        llmb_config_path = slurm_job.llmb_config_path
        log_dir = slurm_job.job_workdir or (str(pathlib.Path(llmb_config_path).parent) if llmb_config_path else None)
        workload_info = workloads.get(task.workload_key, {})
        metadata = workload_info.get('metadata', {})
        launcher_type = metadata.get('run', {}).get('launcher_type')
        if not launcher_type:
            logger.warning(f"Unable to record job history: missing launcher type for workload {task.workload_key}")
            return

        model_name = metadata.get('general', {}).get('model', workload_info.get('workload', ''))
        if launcher_type == 'sbatch':
            log_dir = None

        namespace = None
        workloadrun_name = None
        # Every task of a Kubernetes cluster goes through nvcre, whatever its metadata launcher_type.
        if config.execution_target == ExecutionTarget.KUBERNETES:
            k8s = config.extras.get('kubernetes', {})
            namespace = k8s.get('namespace')
            # The WorkloadRun this submission registered, found by its own job directory
            # (the registry also holds every other submission's entry).
            workloadrun_name = workloadrun_name_for(log_dir)

        record = JobRecord(
            job_id=job_id,
            launcher_type=launcher_type,
            workload_key=task.workload_key,
            model_name=model_name,
            model_size=task.model_size,
            dtype=task.dtype,
            scale=task.scale,
            profile_enabled=task.profile,
            proxy=task.proxy,
            log_dir=log_dir,
            llmb_config_path=llmb_config_path,
            submit_time=slurm_job.submit_time or _now_iso(),
            env_overrides_json=_json_dumps(task.env_overrides),
            model_overrides_json=_json_dumps(task.model_overrides),
            namespace=namespace,
            workloadrun_name=workloadrun_name,
        )

        upsert_static_job(config, record)
    except Exception as e:
        logger.warning(f"Unable to update llmb-run job history: {e}")


def upsert_static_job(config: ClusterConfig, record: JobRecord) -> None:
    now = _now_iso()

    with _open_history_db(config) as conn:
        # submit_time is INSERT-only so a follow-up rebuild can't clobber the
        # canonical sacct value written by _update_slurm_record.
        conn.execute(
            """
            INSERT INTO jobs (
                job_id,
                launcher_type,
                workload_key,
                model_name,
                model_size,
                dtype,
                scale,
                profile_enabled,
                proxy,
                log_dir,
                llmb_config_path,
                submit_time,
                created_at,
                updated_at,
                env_overrides_json,
                model_overrides_json,
                namespace,
                workloadrun_name
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(job_id) DO UPDATE SET
                launcher_type = excluded.launcher_type,
                workload_key = excluded.workload_key,
                model_name = excluded.model_name,
                model_size = excluded.model_size,
                dtype = excluded.dtype,
                scale = excluded.scale,
                profile_enabled = excluded.profile_enabled,
                proxy = excluded.proxy,
                log_dir = excluded.log_dir,
                llmb_config_path = excluded.llmb_config_path,
                updated_at = excluded.updated_at,
                env_overrides_json = excluded.env_overrides_json,
                model_overrides_json = excluded.model_overrides_json,
                namespace = excluded.namespace,
                workloadrun_name = excluded.workloadrun_name
            """,
            (
                record.job_id,
                record.launcher_type,
                record.workload_key,
                record.model_name,
                record.model_size,
                record.dtype,
                record.scale,
                int(record.profile_enabled),
                int(record.proxy),
                record.log_dir,
                record.llmb_config_path,
                record.submit_time,
                now,
                now,
                record.env_overrides_json,
                record.model_overrides_json,
                record.namespace,
                record.workloadrun_name,
            ),
        )
        conn.commit()


def list_jobs(config: ClusterConfig) -> list[sqlite3.Row]:
    with _open_history_db(config) as conn:
        return list(conn.execute("""
                SELECT *
                FROM jobs
                ORDER BY
                    LOWER(COALESCE(NULLIF(workload_key, ''), NULLIF(model_name, ''), '')) ASC,
                    CAST(REPLACE(LOWER(COALESCE(NULLIF(model_size, ''), '0')), 'b', '') AS REAL) DESC,
                    LOWER(COALESCE(NULLIF(model_size, ''), '')) ASC,
                    LOWER(COALESCE(NULLIF(dtype, ''), '')) ASC,
                    scale ASC,
                    profile_enabled ASC,
                    COALESCE(NULLIF(submit_time, ''), created_at) DESC,
                    job_id DESC
                """))


def get_job(config: ClusterConfig, job_id: int) -> sqlite3.Row | None:
    with _open_history_db(config) as conn:
        return conn.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()


def refresh_non_terminal_jobs(config: ClusterConfig) -> tuple[int, str | None]:
    """Refresh status for all non-terminal jobs.

    nvcre jobs are refreshed via nvcrectl; Slurm jobs via sacct.
    Returns (refreshed_count, error_message).
    """
    with _open_history_db(config) as conn:
        rows = list(
            conn.execute(
                "SELECT job_id, slurm_state, launcher_type, workloadrun_name, namespace, log_dir, submit_time "
                "FROM jobs ORDER BY job_id"
            )
        )

    nvcre_rows = [r for r in rows if r["workloadrun_name"] and not is_terminal_state(r["slurm_state"])]

    # nvcre jobs carry synthetic numeric ids that sacct has never heard of; sending them
    # there would eventually mark live WorkloadRuns PURGED. A Kubernetes install records only
    # nvcre jobs, so it has nothing to ask Slurm about.
    nvcre_ids = {int(r["job_id"]) for r in nvcre_rows}
    slurm_ids = (
        []
        if config.execution_target == ExecutionTarget.KUBERNETES
        else [
            int(r["job_id"])
            for r in rows
            if not is_terminal_state(r["slurm_state"]) and int(r["job_id"]) not in nvcre_ids
        ]
    )

    xcal_refreshed = _refresh_nvcre_statuses(config, nvcre_rows)
    slurm_refreshed, error = _refresh_slurm_statuses(config, slurm_ids)
    if error is None:
        _update_terminal_job_results(config)
    return xcal_refreshed + slurm_refreshed, error


def _nvcre_post_run_on_completion(config: ClusterConfig, row: sqlite3.Row, slurm_state: str) -> None:
    """Pull all pod logs at terminal state; cancel the WorkloadRun on success
    and extract the failure configmap on failure.

    Called by _refresh_nvcre_statuses when a job first reaches COMPLETED or FAILED.
    Never raises — logs warnings on any failure so jobs list is never broken.
    """
    import subprocess as _sp

    log_dir = row["log_dir"]
    workloadrun_name = row["workloadrun_name"]
    if not log_dir or not workloadrun_name:
        return

    k8s = config.extras.get('kubernetes') or {}
    namespace = row["namespace"] or k8s.get('namespace', 'default')
    nvcrectl = k8s.get('nvcrectl_bin', 'nvcrectl')

    try:
        if slurm_state == 'COMPLETED':
            # Pull all pod logs first. Cancelling the WorkloadRun deletes its
            # pods — the only remaining copy of the logs — so cancel only
            # after a successful pull; a failed one keeps the pods alive.
            if _nvcre_pull_all_logs(namespace, workloadrun_name, log_dir):
                # Cancel the WorkloadRun to release cluster resources
                r = _sp.run(
                    [nvcrectl, 'workloadrun', 'cancel', workloadrun_name, '-n', namespace],
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                if r.returncode == 0:
                    logger.info("Cancelled WorkloadRun '%s'", workloadrun_name)
                else:
                    logger.warning(
                        "nvcre post-run: cancel failed for '%s': %s",
                        workloadrun_name,
                        r.stderr.strip(),
                    )
            else:
                logger.warning(
                    "nvcre post-run: log pull failed for '%s'; keeping the WorkloadRun "
                    "so its logs stay recoverable from the pods",
                    workloadrun_name,
                )
        elif slurm_state == 'FAILED':
            # Extract the failure configmap first — it is cheap, and a slow or
            # failed pod-log pull must not cost us nvcre's own failure record.
            _nvcre_extract_failure_configmap(namespace, workloadrun_name, log_dir)
            _nvcre_pull_all_logs(namespace, workloadrun_name, log_dir)
    except Exception as e:
        logger.warning("nvcre post-run cleanup error for job %s: %s", row["job_id"], e)


def nvcre_rank0_pod(namespace: str, workloadrun_name: str) -> str | None:
    """Name of the WorkloadRun's rank-0 pod, or None if it cannot be resolved.

    The pod is found through its JobSet label — the same scoping as the
    terminal log pull — so a namespace-wide lookup can never return
    another job's pod.
    """
    import subprocess as _sp

    nvcre_job = _nvcre_internal_job_name(namespace, workloadrun_name) or workloadrun_name
    label = f"jobset.sigs.k8s.io/jobset-name={nvcre_job}-workload,batch.kubernetes.io/job-completion-index=0"
    try:
        r = _sp.run(
            ['kubectl', 'get', 'pods', '-l', label, '-n', namespace, '-o', 'jsonpath={.items[0].metadata.name}'],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except _sp.TimeoutExpired:
        # A stalled cluster must not turn `jobs log` into a traceback.
        logger.warning("nvcre: finding the rank-0 pod of '%s' timed out", workloadrun_name)
        return None
    except Exception as e:
        logger.debug("nvcre: could not find the rank-0 pod of '%s': %s", workloadrun_name, e)
        return None
    return r.stdout.strip() or None


def nvcre_pod_log_candidates(log_dir: str | pathlib.Path) -> list[pathlib.Path]:
    """Local pod-log files pulled at terminal state, rank-0 pod first.

    Pods are named ``<jobset>-node-<job>-<pod>-...``, so ``*-0-*.log``
    matches every pod of node 0. The rank-0 pod (job-completion-index=0)
    is node 0's first pod, which sorts ahead of its siblings — so
    candidates[0] is the rank-0 log, but only by sorting, not by the glob.
    """
    pod_logs_dir = pathlib.Path(log_dir) / 'pod_logs'
    if not pod_logs_dir.is_dir():
        return []
    candidates = sorted(pod_logs_dir.glob('*-0-*.log'))
    if not candidates:
        candidates = sorted(pod_logs_dir.glob('*.log'))
    return candidates


def _nvcre_pull_all_logs(namespace: str, workloadrun_name: str, local_dir: str) -> bool:
    """Pull logs from all pods into local_dir/pod_logs/ and tar them.

    Returns True when a tar archive was written (at least one pod's log
    was fetched), False when nothing could be pulled."""
    import subprocess as _sp
    import tarfile as _tarfile
    from pathlib import Path as _Path

    # Pods are found through the JobSet label of each nvcre job of this WorkloadRun (the primary
    # group and any overflow group). If the jobs cannot be resolved, try the WorkloadRun's own name.
    pods = []
    for nvcre_job in internal_job_names(namespace, workloadrun_name) or [workloadrun_name]:
        try:
            r = _sp.run(
                [
                    'kubectl',
                    'get',
                    'pods',
                    '-l',
                    f'jobset.sigs.k8s.io/jobset-name={nvcre_job}-workload',
                    '-n',
                    namespace,
                    '-o',
                    'jsonpath={range .items[*]}{.metadata.name}{"\\n"}{end}',
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
        except _sp.TimeoutExpired:
            logger.warning("nvcre post-run: listing pods of '%s' timed out", nvcre_job)
            continue
        pods += [p.strip() for p in r.stdout.splitlines() if p.strip()]

    # No fallback to other pods in the namespace: they belong to other jobs.
    logs_dir = _Path(local_dir) / 'pod_logs'

    if not pods:
        logger.warning("nvcre post-run: no pods found for '%s'", workloadrun_name)
        return False

    logs_dir.mkdir(parents=True, exist_ok=True)
    fetched = 0
    for pod in pods:
        # Full logs (no --tail) can take a while; one slow pod must not
        # abort the pull for the others.
        try:
            r = _sp.run(
                ['kubectl', 'logs', pod, '-n', namespace],
                capture_output=True,
                text=True,
                timeout=120,
            )
        except _sp.TimeoutExpired:
            logger.warning("nvcre post-run: log pull for pod '%s' timed out; skipping that pod", pod)
            continue
        if r.returncode == 0:
            (logs_dir / f'{pod}.log').write_text(r.stdout)
            fetched += 1
        else:
            logger.warning(
                "nvcre post-run: failed to fetch logs for pod '%s': %s",
                pod,
                r.stderr.strip(),
            )

    if fetched == 0:
        logger.warning("nvcre post-run: no pod logs retrieved for '%s'", workloadrun_name)
        return False

    tar_path = _Path(local_dir) / 'pod_logs.tar.gz'
    with _tarfile.open(tar_path, 'w:gz') as tar:
        tar.add(logs_dir, arcname='pod_logs')
    logger.info("Pulled logs from %d/%d pods -> %s", fetched, len(pods), logs_dir)
    return True


def _nvcre_internal_job_name(namespace: str, workloadrun_name: str) -> str | None:
    """The nvcre job (primary group) behind a WorkloadRun, or None if it cannot be resolved."""
    names = internal_job_names(namespace, workloadrun_name)
    return names[0] if names else None


def _nvcre_extract_failure_configmap(namespace: str, workloadrun_name: str, local_dir: str | None) -> None:
    """Extract 'Failed Nodes Ref' configmap from a failed WorkloadRun."""
    import json as _json
    import subprocess as _sp
    from pathlib import Path as _Path

    # Reached synchronously from `jobs list` / `jobs show`, so a stalled cluster must not hang them.
    try:
        r = _sp.run(
            ['kubectl', 'describe', 'workloadrun', workloadrun_name, '-n', namespace],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except _sp.TimeoutExpired:
        logger.warning("nvcre: 'kubectl describe workloadrun %s' timed out", workloadrun_name)
        return
    if r.returncode != 0:
        logger.warning(
            "nvcre: could not describe workloadrun '%s': %s",
            workloadrun_name,
            r.stderr.strip(),
        )
        return

    cm_name = None
    lines = r.stdout.splitlines()
    for i, line in enumerate(lines):
        if 'Failed Nodes Ref' not in line:
            continue
        after_colon = line.split(':', 1)[-1].strip()
        if after_colon:
            cm_name = after_colon
            break
        for sub in lines[i + 1 : i + 6]:
            stripped = sub.strip()
            if stripped.lower().startswith('name:'):
                cm_name = stripped.split(':', 1)[-1].strip()
                break
        break

    if not cm_name:
        logger.debug("nvcre: no 'Failed Nodes Ref' configmap found for '%s'", workloadrun_name)
        return

    try:
        r = _sp.run(
            ['kubectl', 'get', 'configmap', cm_name, '-n', namespace, '-o', 'json'],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except _sp.TimeoutExpired:
        logger.warning("nvcre: 'kubectl get configmap %s' timed out", cm_name)
        return
    if r.returncode != 0 or not local_dir:
        return
    try:
        data = _json.loads(r.stdout).get('data', {})
        fail_log = _Path(local_dir) / 'fail.log'
        with open(fail_log, 'w') as f:
            for k, v in data.items():
                f.write(f"=== {k} ===\n{v}\n\n")
        logger.info("nvcre: failure configmap written to %s", fail_log)
    except Exception as e:
        logger.warning("nvcre: could not write failure configmap: %s", e)


def _refresh_nvcre_statuses(config: ClusterConfig, rows: list) -> int:
    """Refresh status for nvcre WorkloadRun jobs via nvcrectl."""
    import shutil
    import subprocess as _sp

    k8s = config.extras.get('kubernetes') or {}
    nvcrectl = k8s.get('nvcrectl_bin', 'nvcrectl')
    if shutil.which(nvcrectl) is None:
        return 0

    refreshed = 0
    for row in rows:
        name = row["workloadrun_name"]
        ns = row["namespace"] or k8s.get('namespace', 'default')
        try:
            r = _sp.run([nvcrectl, 'workloadrun', 'status', name, '-n', ns], capture_output=True, text=True, timeout=30)
        except _sp.TimeoutExpired:
            # An unreachable cluster must not hang `jobs list` / `jobs show`.
            logger.warning("nvcre: 'nvcrectl workloadrun status %s' timed out; keeping the last known state", name)
            continue
        if r.returncode != 0:
            continue
        phase = r.stdout.strip()  # e.g. "Succeeded", "Failed", "InProgress"
        # Map nvcre phases to Slurm-style state strings for the existing display
        phase_to_state = {
            'Succeeded': 'COMPLETED',
            'Failed': 'FAILED',
            'InProgress': 'RUNNING',
            'Pending': 'PENDING',
        }
        slurm_state = phase_to_state.get(phase, phase.upper())
        prev_state = row["slurm_state"]
        # nvcre/nvcrectl only reports phase, not duration. Approximate elapsed
        # as wall-clock time since submit_time (sacct's Elapsed equivalent isn't
        # available here). Only meaningful once the job has actually started;
        # PENDING jobs are left blank rather than showing queue wait time.
        elapsed = ""
        if slurm_state in ("RUNNING", "COMPLETED", "FAILED"):
            elapsed = _elapsed_since(row["submit_time"])
        with _open_history_db(config) as conn:
            conn.execute(
                "UPDATE jobs SET slurm_state = ?, elapsed = ?, last_status_refresh = ?, updated_at = ? "
                "WHERE job_id = ?",
                (slurm_state, elapsed, _now_iso(), _now_iso(), int(row["job_id"])),
            )
            conn.commit()
        # Trigger post-run cleanup only when first transitioning to a terminal state
        if slurm_state in ('COMPLETED', 'FAILED') and not is_terminal_state(prev_state):
            _nvcre_post_run_on_completion(config, row, slurm_state)
            _update_terminal_job_results(config, job_ids=[int(row["job_id"])])
        refreshed += 1
    return refreshed


def refresh_requested_jobs(config: ClusterConfig, job_ids: list[int]) -> tuple[int, str | None]:
    """Force-refresh the requested jobs and update terminal results."""
    if config.execution_target == ExecutionTarget.KUBERNETES:
        # Same reason as refresh_non_terminal_jobs: these ids are not Slurm jobs.
        wanted = set(job_ids)
        with _open_history_db(config) as conn:
            rows = [
                r
                for r in conn.execute(
                    "SELECT job_id, slurm_state, launcher_type, workloadrun_name, namespace, log_dir, submit_time "
                    "FROM jobs ORDER BY job_id"
                )
                if int(r["job_id"]) in wanted and r["workloadrun_name"]
            ]
        refreshed = _refresh_nvcre_statuses(config, rows)
        _update_terminal_job_results(config, job_ids=job_ids, reparse_existing=True)
        return refreshed, None
    refreshed, error = _refresh_slurm_statuses(config, job_ids)
    if error is None:
        _update_terminal_job_results(config, job_ids=job_ids, reparse_existing=True)
    return refreshed, error


def _refresh_slurm_statuses(config: ClusterConfig, job_ids: list[int]) -> tuple[int, str | None]:
    """Refresh sacct status for the given jobs.

    Jobs in ``job_ids`` that sacct succeeds on but does not return are marked
    ``slurm_state = "PURGED"`` (sacct accounting has dropped them). Returns
    (refreshed_count, error_message); error_message is None on success.
    """
    if not job_ids:
        return 0, None

    refreshed = 0
    for chunk in _chunks(sorted({int(job_id) for job_id in job_ids}), SACCT_JOB_CHUNK_SIZE):
        records = get_slurm_job_statuses(chunk, start_time=_sacct_start_time(config, chunk))
        if records is None:
            return refreshed, "sacct query failed"

        with _open_history_db(config) as conn:
            for job_id in chunk:
                record = records.get(job_id)
                if record is not None:
                    _update_slurm_record(conn, record)
                    refreshed += 1
                elif _mark_job_purged(conn, job_id):
                    refreshed += 1
                # else: still inside PURGE_GRACE_SECONDS — leave for the next refresh.
            conn.commit()

    return refreshed, None


def _sacct_start_time(config: ClusterConfig, job_ids: list[int]) -> str | None:
    """Return a safe lower bound for a sacct query, or None for legacy rows."""
    placeholders = ','.join('?' for _ in job_ids)
    with _open_history_db(config) as conn:
        rows = list(
            conn.execute(
                f"SELECT submit_time FROM jobs WHERE job_id IN ({placeholders})",
                job_ids,
            )
        )

    submit_times = [row["submit_time"] for row in rows]
    if len(submit_times) != len(job_ids) or any(not value for value in submit_times):
        return None

    try:
        submit_dates = []
        for value in submit_times:
            timestamp = datetime.datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            if timestamp.tzinfo is not None:
                timestamp = timestamp.astimezone()
            submit_dates.append(timestamp.date())
    except ValueError:
        logger.debug("Unable to derive sacct start time from job submit timestamps")
        return None

    # Starting at the previous midnight covers small clock/timezone differences
    # while avoiding sacct's --jobs default of searching from Unix epoch 0.
    return (min(submit_dates) - datetime.timedelta(days=SACCT_START_TIME_SAFETY_DAYS)).isoformat()


def rebuild_history(config: ClusterConfig, workloads: dict[str, Any]) -> RebuildStats:
    db_path = get_history_db_path(config.llmb_install)
    workloads_root = pathlib.Path(config.llmb_install) / "workloads"
    scanned = 0
    imported = 0
    skipped = 0
    imported_job_ids: list[int] = []

    if not workloads_root.exists():
        return RebuildStats(scanned=0, imported=0, skipped=0, db_path=db_path)

    for config_path in sorted(workloads_root.glob("**/llmb-config_*.yaml")):
        scanned += 1
        record = job_record_from_config(config.llmb_install, config_path, workloads)
        if record is None:
            skipped += 1
            continue

        upsert_static_job(config, record)
        imported += 1
        imported_job_ids.append(record.job_id)

    _, refresh_error = _refresh_slurm_statuses(config, imported_job_ids)
    if refresh_error is None:
        _update_terminal_job_results(config)
    return RebuildStats(
        scanned=scanned, imported=imported, skipped=skipped, db_path=db_path, refresh_error=refresh_error
    )


def job_record_from_config(
    llmb_install: str | pathlib.Path, config_path: pathlib.Path, workloads: dict[str, Any]
) -> JobRecord | None:
    config_data = _load_llmb_config(config_path)
    if not config_data:
        return None

    job_info = config_data.get('job_info') or {}
    model_info = config_data.get('model_info') or {}
    job_config = config_data.get('job_config') or {}

    raw_job_id = job_info.get('job_id') or _job_id_from_config_filename(config_path)
    try:
        job_id = parse_slurm_job_id(raw_job_id)
    except ValueError:
        logger.debug(f"Skipping llmb config without a parseable job id: {config_path}")
        return None

    workloads_root = pathlib.Path(llmb_install) / "workloads"
    workload_key = None
    try:
        relative = config_path.relative_to(workloads_root)
        if relative.parts:
            workload_key = relative.parts[0]
    except ValueError:
        pass

    launcher_type = _launcher_type_for_workload(workloads, workload_key)
    if not launcher_type:
        logger.debug(f"Skipping llmb config without a known launcher type: {config_path}")
        return None
    if launcher_type == 'sbatch':
        logger.debug(f"Skipping legacy sbatch llmb config during rebuild: {config_path}")
        return None

    # Older llmb-config files call submit_time launch_time, but it is recorded
    # immediately after Slurm submission so the values are interchangeable.
    submit_time = job_info.get('submit_time') or job_info.get('launch_time')

    return JobRecord(
        job_id=job_id,
        launcher_type=launcher_type,
        workload_key=workload_key,
        model_name=model_info.get('model_name'),
        model_size=model_info.get('model_size'),
        dtype=model_info.get('dtype'),
        scale=_as_int(model_info.get('scale')),
        profile_enabled=bool(job_config.get('profile_enabled')),
        proxy=bool(job_config.get('proxy')),
        log_dir=str(config_path.parent),
        llmb_config_path=str(config_path),
        submit_time=submit_time,
        env_overrides_json=_json_dumps(job_config.get('env_overrides') or {}),
        model_overrides_json=_json_dumps(job_config.get('model_overrides') or {}),
    )


# States omitted from this map render in the terminal's default color.
# PENDING intentionally stays unstyled so RUNNING is easier to spot.
_SLURM_STATE_STYLES = {
    "COMPLETED": "green",
    "RUNNING": "cyan",
    "REQUEUED": "yellow",
    "PREEMPTED": "yellow",
    "CANCELLED": "yellow",
    "PURGED": "dim",
    "FAILED": "red",
    "BOOT_FAIL": "red",
    "NODE_FAIL": "red",
    "OUT_OF_MEMORY": "red",
    "DEADLINE": "red",
    "TIMEOUT": "red",
}


def format_jobs_table(rows: list[sqlite3.Row], workloads: dict[str, Any]) -> str:
    if not rows:
        return "No jobs found. Run `llmb-run jobs rebuild` to scan existing llmb-config files."

    # Color when stdout is a terminal; plain text when piped or under tests.
    use_color = sys.stdout.isatty()

    # Show the Tokens/s/GPU column only when an RL workload is present. This
    # keeps the pretrain-only view unchanged while still labeling unparsed or
    # failed RL rows with the right throughput metric.
    show_tokens_col = any(_is_rl_workload(workloads, row["workload_key"]) for row in rows)

    show_k8s_cols = any(row["workloadrun_name"] for row in rows)

    table = Table(box=box.SIMPLE_HEAD, header_style="bold", show_edge=False, pad_edge=False)
    table.add_column("Workload", overflow="fold")
    table.add_column("DType")
    table.add_column("Scale", justify="right")
    # Synthetic nvcre ids are 16 digits and must never be squeezed to an
    # ellipsis by the other columns — a truncated id cannot be looked up.
    table.add_column("Job ID", justify="right", no_wrap=True)
    if show_k8s_cols:
        table.add_column("Name")
        table.add_column("Namespace")
    table.add_column("Profile")
    table.add_column("Submit Time")
    table.add_column("Status")
    table.add_column("Elapsed")
    table.add_column("s/iter", justify="right")
    table.add_column("TFLOPS/GPU", justify="right")
    if show_tokens_col:
        table.add_column("Tokens/s/GPU", justify="right")

    for row in rows:
        slurm_state = row["slurm_state"] or ""
        style = _SLURM_STATE_STYLES.get(base_slurm_state(slurm_state)) if use_color else None
        display_state = _display_slurm_state(slurm_state)
        styled_state = f"[{style}]{display_state}[/{style}]" if style and display_state else display_state

        cells = [
            _display_workload(row),
            row["dtype"] or "",
            str(row["scale"]) if row["scale"] is not None else "",
            str(row["job_id"]),
        ]
        if show_k8s_cols:
            cells.append(row["workloadrun_name"] or "")
            cells.append(row["namespace"] or "")
        cells += [
            _yes_no(row["profile_enabled"]),
            _format_timestamp(row["submit_time"]),
            styled_state,
            row["elapsed"] or "",
            _format_perf_metric(row, "train_step_time_seconds"),
            _format_perf_metric(row, "tflops_per_gpu"),
        ]
        if show_tokens_col:
            cells.append(_format_float(row["tokens_per_sec_per_gpu"]))
        table.add_row(*cells)

    console = Console(
        width=160,
        force_terminal=use_color,
        color_system="truecolor" if use_color else None,
    )
    with console.capture() as capture:
        console.print(table)
    return capture.get().rstrip()


def format_job_details(row: sqlite3.Row, workloads: dict[str, Any]) -> str:
    # Show the throughput metric that's meaningful for this job's type:
    #   - pretrain jobs: TFLOPS/GPU
    #   - RL jobs:       Tokens/s/GPU
    # The workload's framework decides, resolved from the already-loaded
    # workload metadata (no per-row disk reads at display time).
    throughput_details: list[tuple[str, str]]
    if _is_rl_workload(workloads, row["workload_key"]):
        throughput_details = [("Tokens/s/GPU", _format_float(row["tokens_per_sec_per_gpu"]))]
    else:
        throughput_details = [("TFLOPS/GPU", _format_float(row["tflops_per_gpu"]))]

    _cols = {c[1] for c in row.description} if hasattr(row, 'description') else set(row.keys())
    details = [
        ("Job ID", str(row["job_id"])),
        ("Launcher", row["launcher_type"]),
        ("Workload", row["workload_key"]),
        ("Model", _display_model(row)),
        ("DType", row["dtype"]),
        ("Scale", str(row["scale"]) if row["scale"] is not None else ""),
        ("Profile", _yes_no(row["profile_enabled"])),
        ("Proxy", _yes_no(row["proxy"])),
        *(
            [("WorkloadRun", row["workloadrun_name"]), ("Namespace", row["namespace"])]
            if "workloadrun_name" in _cols and row["workloadrun_name"]
            else []
        ),
        ("Status", row["slurm_state"]),
        ("Elapsed", row["elapsed"]),
        ("Perf Parse", _display_perf_parse_status(row["perf_parse_status"])),
        ("s/iter", _format_float(row["train_step_time_seconds"])),
        *throughput_details,
        ("Submit Time", _format_timestamp(row["submit_time"])),
        ("Node List", row["node_list"]),
        ("Exit Code", row["exit_code"]),
        ("Log Dir", row["log_dir"]),
    ]
    width = max(len(label) for label, _ in details)
    return "\n".join(f"{label.ljust(width)} : {value or ''}" for label, value in details)


def _connect(db_path: pathlib.Path) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout = 5000")
    return conn


def _initialize_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """)
    stored = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()
    if stored is not None:
        try:
            stored_version = int(stored[0])
        except (TypeError, ValueError) as e:
            raise RuntimeError(f"Job history DB has an unparseable schema_version {stored[0]!r}.") from e
        if stored_version > DB_SCHEMA_VERSION:
            raise RuntimeError(
                f"Job history DB schema_version {stored_version} is newer than this llmb-run "
                f"build supports ({DB_SCHEMA_VERSION}). Upgrade llmb-run."
            )
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY,
            launcher_type TEXT NOT NULL,
            workload_key TEXT,
            model_name TEXT,
            model_size TEXT,
            dtype TEXT,
            scale INTEGER,
            profile_enabled INTEGER NOT NULL DEFAULT 0,
            proxy INTEGER NOT NULL DEFAULT 0,
            log_dir TEXT,
            llmb_config_path TEXT,
            created_at TEXT,
            updated_at TEXT,
            slurm_state TEXT,
            elapsed TEXT,
            submit_time TEXT,
            node_list TEXT,
            exit_code TEXT,
            last_status_refresh TEXT,
            env_overrides_json TEXT,
            model_overrides_json TEXT,
            train_step_time_seconds REAL,
            tflops_per_gpu REAL,
            tokens_per_sec_per_gpu REAL,
            perf_parse_status TEXT
        )
        """)
    # Schema migrations — idempotent, ordered by version.
    # v1 → v2: add tokens_per_sec_per_gpu column (RL workload throughput metric).
    # v2 → v3: add namespace and workloadrun_name columns (nvcre/Kubernetes jobs).
    existing_columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    if "tokens_per_sec_per_gpu" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN tokens_per_sec_per_gpu REAL")
    if "namespace" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN namespace TEXT")
    if "workloadrun_name" not in existing_columns:
        conn.execute("ALTER TABLE jobs ADD COLUMN workloadrun_name TEXT")

    conn.execute(
        """
        INSERT INTO metadata (key, value)
        VALUES ('schema_version', ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        (str(DB_SCHEMA_VERSION),),
    )


def _update_slurm_record(conn: sqlite3.Connection, record: SlurmAccountingRecord) -> None:
    now = _now_iso()
    conn.execute(
        """
        UPDATE jobs
        SET
            slurm_state = ?,
            elapsed = ?,
            submit_time = ?,
            node_list = ?,
            exit_code = ?,
            last_status_refresh = ?,
            updated_at = ?
        WHERE job_id = ?
        """,
        (
            record.state,
            record.elapsed,
            record.submit_time,
            record.node_list,
            record.exit_code,
            now,
            now,
            record.job_id,
        ),
    )


def _mark_job_purged(conn: sqlite3.Connection, job_id: int) -> bool:
    """Flag a job as PURGED when sacct reports nothing for it.

    Skips jobs created within ``PURGE_GRACE_SECONDS`` so a freshly submitted
    job that hasn't propagated to sacct yet isn't mislabeled.
    Returns True when the row was updated; False when the grace period blocked it.
    """
    now_dt = datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0)
    cutoff = (now_dt - datetime.timedelta(seconds=PURGE_GRACE_SECONDS)).isoformat()
    now = now_dt.isoformat()
    cursor = conn.execute(
        """
        UPDATE jobs
        SET slurm_state = ?, last_status_refresh = ?, updated_at = ?
        WHERE job_id = ? AND (created_at IS NULL OR created_at < ?)
        """,
        ("PURGED", now, now, job_id, cutoff),
    )
    return cursor.rowcount > 0


def _update_terminal_job_results(
    config: ClusterConfig, *, job_ids: list[int] | None = None, reparse_existing: bool = False
) -> None:
    """Update result columns for terminal jobs, skipping previous attempts by default."""
    with _open_history_db(config) as conn:
        filters = [
            "launcher_type IN ('nemo', 'megatron_bridge', 'configured_sbatch')",
            "log_dir IS NOT NULL",
            "llmb_config_path IS NOT NULL",
        ]
        params: list[Any] = []
        if not reparse_existing:
            filters.append("perf_parse_status IS NULL")
        if job_ids:
            job_ids = sorted({int(job_id) for job_id in job_ids})
            filters.append(f"job_id IN ({','.join('?' for _ in job_ids)})")
            params.extend(job_ids)

        rows = list(
            conn.execute(
                f"""
                SELECT *
                FROM jobs
                WHERE {' AND '.join(filters)}
                ORDER BY job_id
                """,
                params,
            )
        )

        for row in rows:
            if not is_terminal_state(row["slurm_state"]):
                continue

            result = _parse_job_performance(row)
            if result is None:
                continue

            now = _now_iso()
            conn.execute(
                """
                UPDATE jobs
                SET
                    perf_parse_status = ?,
                    train_step_time_seconds = ?,
                    tflops_per_gpu = ?,
                    tokens_per_sec_per_gpu = ?,
                    updated_at = ?
                WHERE job_id = ?
                """,
                (
                    result.status,
                    result.train_step_time_seconds,
                    result.tflops_per_gpu,
                    result.tokens_per_sec_per_gpu,
                    now,
                    row["job_id"],
                ),
            )

        conn.commit()


@dataclass(frozen=True)
class _PerfResult:
    status: str  # status.value string, e.g. "success", "incomplete"
    train_step_time_seconds: float | None = None
    tflops_per_gpu: float | None = None
    tokens_per_sec_per_gpu: float | None = None


def _parse_job_performance(row: sqlite3.Row) -> _PerfResult | None:
    if _is_rl_job(row):
        return _parse_rl_job_performance(row)
    return _parse_pretrain_job_performance(row)


def _is_rl_framework(framework: str | None) -> bool:
    return framework is not None and str(framework).strip().lower() == "nemo-rl"


def _is_rl_job(row: sqlite3.Row) -> bool:
    return _is_rl_framework(_framework_from_llmb_config(row["llmb_config_path"]))


def _is_rl_workload(workloads: dict[str, Any], workload_key: str | None) -> bool:
    """RL check against the loaded workload metadata — a dict lookup, no disk I/O."""
    if not workload_key:
        return False
    workload_info = workloads.get(workload_key)
    if not isinstance(workload_info, dict):
        return False
    metadata = workload_info.get('metadata')
    if not isinstance(metadata, dict):
        return False
    return _is_rl_framework((metadata.get('general') or {}).get('framework'))


def _parse_pretrain_job_performance(row: sqlite3.Row) -> _PerfResult | None:
    framework = _framework_from_llmb_config(row["llmb_config_path"])
    if parser_name_for_framework(framework) is None:
        return None

    workload_key = row["workload_key"] or ""
    if "mlperf" in workload_key:
        min_iteration, max_iteration = MLPERF_ITERATION_WINDOW_OVERRIDES.get(workload_key, MLPERF_ITERATION_WINDOW)
    else:
        min_iteration, max_iteration = MIN_ITERATION, MAX_ITERATION
    try:
        result = parse_latest_pretrain_job_log(
            row["log_dir"],
            int(row["job_id"]),
            framework,
            min_iteration=min_iteration,
            max_iteration=max_iteration,
        )
    except OSError as e:
        logger.debug(f"Unable to parse pretrain perf log for job {row['job_id']}: {e}")
        return None

    # nvcre jobs write their logs to pod_logs/*.log; fall back to the local
    # rank-0 pod log if the standard log discovery found nothing.
    if result.status.value in ("no_log", "no_data") and row["log_dir"]:
        candidates = nvcre_pod_log_candidates(row["log_dir"])
        if candidates:
            try:
                from llmb_run.pretrain_log_parser import parse_pretrain_log

                # Same steady-state window as Slurm (iterations 35-44): a run
                # that never got there reports incomplete rather than a
                # warmup-inclusive number.
                result = parse_pretrain_log(candidates[0], framework)
            except OSError as e:
                logger.debug(f"Unable to parse nvcre pod log for job {row['job_id']}: {e}")

    metrics = result.metrics
    return _PerfResult(
        status=result.status.value,
        train_step_time_seconds=metrics.time_mean_seconds if metrics else None,
        tflops_per_gpu=metrics.tflops_per_gpu_mean if metrics else None,
    )


def _parse_rl_job_performance(row: sqlite3.Row) -> _PerfResult | None:
    if not row["log_dir"]:
        return None

    result = parse_rl_metrics(row["log_dir"])
    metrics = result.metrics
    return _PerfResult(
        status=result.status.value,
        train_step_time_seconds=metrics.step_time_mean_seconds if metrics else None,
        tokens_per_sec_per_gpu=metrics.tokens_per_sec_per_gpu_mean if metrics else None,
    )


def _framework_from_llmb_config(config_path: str | pathlib.Path | None) -> str | None:
    if not config_path:
        return None

    config_data = _load_llmb_config(pathlib.Path(config_path))
    framework = (config_data.get('workload_info') or {}).get('framework')
    return str(framework) if framework else None


def _display_model(row: sqlite3.Row) -> str:
    model_name = row["model_name"] or row["workload_key"] or ""
    model_size = row["model_size"] or ""
    if model_name and model_size:
        return f"{model_name}_{model_size}"
    return model_name or model_size


def _display_workload(row: sqlite3.Row) -> str:
    workload_key = row["workload_key"] or ""
    model_size = row["model_size"] or ""
    if workload_key and model_size:
        return f"{workload_key}_{model_size}"
    return workload_key or _display_model(row)


def _display_slurm_state(state: str | None) -> str:
    if base_slurm_state(state) == "CANCELLED":
        return "CANCELLED"
    return state or ""


def _format_timestamp(value: str | None) -> str:
    if not value:
        return ""
    return value.replace("T", " ")[:16]


def _format_float(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def _elapsed_since(submit_time: str | None) -> str:
    """Wall-clock elapsed time since submit_time, formatted like sacct's Elapsed field."""
    if not submit_time:
        return ""
    try:
        start = datetime.datetime.fromisoformat(submit_time)
    except ValueError:
        return ""
    if start.tzinfo is None:
        # nvcre submit_time (job_launcher.py) is stamped with naive
        # datetime.now() — local time, not UTC. Compare against local "now" so
        # this doesn't go negative (and get clamped to 0) in timezones ahead
        # of UTC.
        now = datetime.datetime.now()
    else:
        now = datetime.datetime.now(datetime.timezone.utc)

    seconds = max(int((now - start).total_seconds()), 0)
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    if days:
        return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_perf_metric(row: sqlite3.Row, field: str) -> str:
    if row["perf_parse_status"] == PretrainLogParseStatus.INVALID_GRAD_NORM.value:
        return "Invalid"
    return _format_float(row[field])


def _display_perf_parse_status(status: str | None) -> str:
    if status == PretrainLogParseStatus.INVALID_GRAD_NORM.value:
        return "invalid: grad_norm=nan"
    return status or ""


def _yes_no(value: object) -> str:
    return "Yes" if value else "-"


def _load_llmb_config(config_path: pathlib.Path) -> dict[str, Any]:
    try:
        with config_path.open('r') as f:
            data = yaml.safe_load(f)
    except (OSError, yaml.YAMLError) as e:
        logger.debug(f"Unable to read llmb config {config_path}: {e}")
        return {}

    return data if isinstance(data, dict) else {}


def _launcher_type_for_workload(workloads: dict[str, Any], workload_key: str | None) -> str | None:
    if not workload_key:
        return None
    workload_info = workloads.get(workload_key)
    if not isinstance(workload_info, dict):
        return None
    metadata = workload_info.get('metadata')
    if not isinstance(metadata, dict):
        return None
    launcher_type = (metadata.get('run') or {}).get('launcher_type')
    return str(launcher_type) if launcher_type else None


def _job_id_from_config_filename(config_path: pathlib.Path) -> str | None:
    name = config_path.name
    prefix = "llmb-config_"
    suffix = ".yaml"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return None


def _json_dumps(value: Any) -> str:
    return json.dumps(value or {}, sort_keys=True)


def _as_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _chunks(values: list[int], size: int) -> list[list[int]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def _now_iso() -> str:
    return datetime.datetime.now(datetime.timezone.utc).replace(microsecond=0).isoformat()
