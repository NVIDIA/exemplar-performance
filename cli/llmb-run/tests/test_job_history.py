# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import sqlite3
import statistics
import subprocess
from unittest.mock import MagicMock, call, patch

import pytest
import yaml

from llmb_run.job_history import (
    JobRecord,
    _parse_pretrain_job_performance,
    format_job_details,
    format_jobs_table,
    get_history_db_path,
    get_job,
    is_terminal_state,
    list_jobs,
    rebuild_history,
    record_job_submission,
    refresh_non_terminal_jobs,
    refresh_requested_jobs,
    upsert_static_job,
)
from llmb_run.job_launcher import run_tests
from llmb_run.metadata_utils import ExecutionTarget
from llmb_run.pretrain_log_parser import MAX_ITERATION, MIN_ITERATION
from llmb_run.slurm_utils import SlurmAccountingRecord, SlurmJob, parse_slurm_job_id
from llmb_run.tasks import WorkloadTask


def test_parse_slurm_job_id_accepts_sbatch_cluster_suffix():
    assert parse_slurm_job_id("12345;cluster-a\n") == 12345


def test_preempted_is_not_terminal_for_requeue_refresh():
    assert is_terminal_state("PREEMPTED") is False
    assert is_terminal_state("REQUEUED") is False
    assert is_terminal_state("COMPLETED") is True
    assert is_terminal_state("CANCELLED by 1234") is True
    assert is_terminal_state("Cancelled by 1234") is True
    assert is_terminal_state("PURGED") is True


def test_record_job_submission_creates_install_local_db(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run" / "task"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_12345.yaml"
    llmb_config.write_text(
        yaml.safe_dump(
            {
                "job_info": {"job_id": "12345", "launch_time": "2026-04-17T13:42:00"},
                "model_info": {
                    "model_name": "foo",
                    "model_size": "7b",
                    "dtype": "fp16",
                    "scale": 8,
                },
                "job_config": {
                    "profile_enabled": False,
                    "proxy": False,
                    "env_overrides": {"A": "B"},
                    "model_overrides": {"tp": 1},
                },
            }
        )
    )

    task = WorkloadTask(
        workload_key="training_foo",
        model_size="7b",
        dtype="fp16",
        scale=8,
        env_overrides={"A": "B"},
        model_overrides={"tp": 1},
    )
    workloads = {
        "training_foo": {
            "metadata": {"general": {"model": "foo"}, "run": {"launcher_type": "nemo"}},
            "workload": "foo",
        }
    }

    record_job_submission(
        config,
        task,
        SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=str(llmb_config)),
        workloads,
    )

    db_path = get_history_db_path(llmb_install)
    assert db_path == llmb_install / ".llmb" / "jobs.sqlite3"
    assert db_path.exists()

    rows = list_jobs(config)
    assert len(rows) == 1
    row = rows[0]
    assert row["job_id"] == 12345
    assert row["launcher_type"] == "nemo"
    assert row["workload_key"] == "training_foo"
    assert row["model_name"] == "foo"
    assert row["log_dir"] == str(log_dir)
    assert row["llmb_config_path"] == str(llmb_config)
    # submit_time is wall-clock at submission; sacct will overwrite it on refresh.
    assert row["submit_time"]
    assert row["created_at"]

    with sqlite3.connect(db_path) as conn:
        schema_version = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()[0]
    assert schema_version == "3"


def test_run_tests_records_successful_submission(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    llmb_config = llmb_install / "workloads" / "training_foo" / "experiments" / "run" / "llmb-config_456.yaml"
    task = WorkloadTask("training_foo", "7b", "fp16", 8)
    workloads = {
        "training_foo": {
            "metadata": {
                "general": {"model": "foo"},
                "run": {"launcher_type": "fake"},
            },
            "workload": "foo",
        }
    }

    class FakeLauncher:
        def launch(self, launched_task):
            assert launched_task is task
            return SlurmJob(job_id=456, job_workdir=None, llmb_config_path=str(llmb_config))

    with patch("llmb_run.job_launcher.create_launcher", return_value=FakeLauncher()):
        run_tests(config, [task], workloads)

    row = get_job(config, 456)
    assert row is not None
    assert row["workload_key"] == "training_foo"
    assert row["launcher_type"] == "fake"
    assert row["model_name"] == "foo"
    assert row["log_dir"] == str(llmb_config.parent)
    assert row["llmb_config_path"] == str(llmb_config)


def test_record_sbatch_submission_keeps_job_but_not_log_dir(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    llmb_config = llmb_install / "workloads" / "inference_foo" / "llmb-config_777.yaml"
    task = WorkloadTask("inference_foo", "7b", "fp16", 8)
    workloads = {
        "inference_foo": {
            "metadata": {
                "general": {"model": "foo"},
                "run": {"launcher_type": "sbatch"},
            },
            "workload": "foo",
        }
    }

    record_job_submission(
        config,
        task,
        SlurmJob(job_id=777, job_workdir=None, llmb_config_path=str(llmb_config)),
        workloads,
    )

    row = get_job(config, 777)
    assert row is not None
    assert row["launcher_type"] == "sbatch"
    assert row["workload_key"] == "inference_foo"
    assert row["log_dir"] is None
    assert row["llmb_config_path"] == str(llmb_config)


def test_jobs_list_sorting_and_default_columns(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    records = [
        JobRecord(
            job_id=3,
            launcher_type="megatron_bridge",
            workload_key="pretrain_qwen3",
            model_name="qwen3",
            model_size="30b",
            dtype="fp8",
            scale=64,
        ),
        JobRecord(
            job_id=2,
            launcher_type="megatron_bridge",
            workload_key="pretrain_llama3.1",
            model_name="llama3.1",
            model_size="8b",
            dtype="fp8",
            scale=16,
        ),
        JobRecord(
            job_id=1,
            launcher_type="megatron_bridge",
            workload_key="pretrain_llama3.1",
            model_name="llama3.1",
            model_size="8b",
            dtype="bf16",
            scale=8,
        ),
        JobRecord(
            job_id=4,
            launcher_type="megatron_bridge",
            workload_key="pretrain_qwen3",
            model_name="qwen3",
            model_size="30b",
            dtype="bf16",
            scale=16,
        ),
        JobRecord(
            job_id=5,
            launcher_type="megatron_bridge",
            workload_key="pretrain_llama3.1",
            model_name="llama3.1",
            model_size="405b",
            dtype="bf16",
            scale=1024,
        ),
        JobRecord(
            job_id=6,
            launcher_type="megatron_bridge",
            workload_key="pretrain_llama3.1",
            model_name="llama3.1",
            model_size="70b",
            dtype="bf16",
            scale=64,
        ),
    ]
    for record in records:
        upsert_static_job(config, record)

    db_path = get_history_db_path(llmb_install)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE jobs
            SET
                node_list = ?,
                slurm_state = ?,
                elapsed = ?,
                submit_time = ?,
                train_step_time_seconds = ?,
                tflops_per_gpu = ?
            WHERE job_id = ?
            """,
            (
                "pool0-[001-128]",
                "COMPLETED",
                "00:10:00",
                "2026-04-17T13:42:00",
                12.345,
                987.65,
                1,
            ),
        )

    rows = list_jobs(config)
    assert [row["job_id"] for row in rows] == [5, 6, 1, 2, 4, 3]

    table = format_jobs_table(rows, {})
    assert "Workload" in table
    assert "Status" in table
    assert "s/iter" in table
    assert "TFLOPS/GPU" in table
    assert "Submit Time" in table
    assert table.index("Submit Time") < table.index("s/iter") < table.index("TFLOPS/GPU")
    assert table.index("pretrain_llama3.1_405b") < table.index("pretrain_llama3.1_70b")
    assert table.index("pretrain_llama3.1_70b") < table.index("pretrain_llama3.1_8b")
    assert "Node List" not in table
    assert "pool0" not in table
    assert "2026-04-17 13:42" in table
    assert "12.35" in table
    assert "987.65" in table

    details = format_job_details(next(row for row in rows if row["job_id"] == 1), {})
    assert "Submit Time" in details
    assert "Launcher" in details
    assert "s/iter" in details
    assert "TFLOPS/GPU" in details
    assert "12.35" in details
    assert "987.65" in details
    assert "LLMB Config" not in details
    assert "Launch Time" not in details


def test_jobs_table_truncates_cancelled_state_but_details_keep_raw_state(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    upsert_static_job(
        config,
        JobRecord(
            job_id=123,
            launcher_type="megatron_bridge",
            workload_key="pretrain_llama3.1",
            model_name="llama3.1",
            model_size="70b",
            dtype="fp8",
            scale=64,
        ),
    )

    db_path = get_history_db_path(llmb_install)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE jobs SET slurm_state = ? WHERE job_id = ?", ("Cancelled by 140751", 123))

    row = list_jobs(config)[0]
    table = format_jobs_table([row], {})
    assert "CANCELLED" in table
    assert "140751" not in table

    details = format_job_details(row, {})
    assert "Cancelled by 140751" in details


def test_rebuild_history_imports_configs_and_refreshes_sacct(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = (
        llmb_install
        / "workloads"
        / "pretrain_grok1"
        / "experiments"
        / "pretrain_grok1_314b_bf16_gpus256"
        / "run_123"
        / "pretrain_grok1_314b_bf16_gpus256"
    )
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_143608.yaml"
    llmb_config.write_text(
        yaml.safe_dump(
            {
                "job_info": {"job_id": "143608", "launch_time": "2026-04-17T15:24:00"},
                "model_info": {
                    "model_name": "grok1",
                    "model_size": "314b",
                    "dtype": "bf16",
                    "scale": 256,
                },
                "job_config": {
                    "profile_enabled": True,
                    "proxy": False,
                    "env_overrides": {},
                    "model_overrides": {},
                },
            }
        )
    )

    with patch(
        "llmb_run.job_history.get_slurm_job_statuses",
        return_value={
            143608: SlurmAccountingRecord(
                job_id=143608,
                state="COMPLETED",
                elapsed="00:21:09",
                submit_time="2026-04-17T15:24:02",
                node_list="node[001-032]",
                exit_code="0:0",
            )
        },
    ) as mock_sacct:
        stats = rebuild_history(
            config,
            {"pretrain_grok1": {"metadata": {"run": {"launcher_type": "nemo"}}}},
        )

    assert stats.scanned == 1
    assert stats.imported == 1
    assert stats.skipped == 0
    mock_sacct.assert_called_once_with([143608], start_time="2026-04-16")

    row = get_job(config, 143608)
    assert row is not None
    assert row["launcher_type"] == "nemo"
    assert row["workload_key"] == "pretrain_grok1"
    assert row["log_dir"] == str(log_dir)
    assert row["slurm_state"] == "COMPLETED"
    assert row["elapsed"] == "00:21:09"


def test_rebuild_history_skips_legacy_sbatch_configs(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    config_dir = llmb_install / "workloads" / "inference_foo"
    config_dir.mkdir(parents=True)
    (config_dir / "llmb-config_777.yaml").write_text(
        yaml.safe_dump(
            {
                "job_info": {"job_id": "777", "launch_time": "2026-04-17T15:24:00"},
                "model_info": {
                    "model_name": "foo",
                    "model_size": "7b",
                    "dtype": "fp16",
                    "scale": 8,
                },
                "job_config": {"profile_enabled": False, "proxy": False},
            }
        )
    )

    with patch("llmb_run.job_history.get_slurm_job_statuses") as mock_sacct:
        stats = rebuild_history(
            config,
            {"inference_foo": {"metadata": {"run": {"launcher_type": "sbatch"}}}},
        )

    assert stats.scanned == 1
    assert stats.imported == 0
    assert stats.skipped == 1
    assert get_job(config, 777) is None
    mock_sacct.assert_not_called()


def test_rebuild_does_not_clobber_sacct_submit_time(tmp_path, cluster_config_factory):
    """A second rebuild after a successful refresh must keep the sacct submit_time."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=999,
            launcher_type="megatron_bridge",
            model_name="m",
            model_size="7b",
            submit_time="local-time",
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=999,
        state="COMPLETED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n[1-2]",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={999: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [999])
    assert error is None and refreshed == 1
    assert get_job(config, 999)["submit_time"] == "2026-04-17T13:42:00"

    # Simulating a follow-up rebuild that re-imports the same job from llmb-config
    # must not overwrite the canonical sacct submit_time.
    upsert_static_job(
        config,
        JobRecord(
            job_id=999,
            launcher_type="megatron_bridge",
            model_name="m",
            model_size="7b",
            submit_time="local-time-2",
        ),
    )
    assert get_job(config, 999)["submit_time"] == "2026-04-17T13:42:00"


def test_refresh_parses_perf_for_failed_terminal_nemo_job(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_123.yaml"
    _write_llmb_config(llmb_config, 123, framework="nemo2")
    _write_nemo_perf_log(log_dir / "log-training_foo_123_0.out", time_seconds=1.5, tflops=700.0)

    upsert_static_job(
        config,
        JobRecord(
            job_id=123,
            launcher_type="nemo",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=123,
        state="FAILED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="1:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={123: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [123])

    assert error is None and refreshed == 1
    row = get_job(config, 123)
    assert row["slurm_state"] == "FAILED"
    assert row["perf_parse_status"] == "success"
    assert row["train_step_time_seconds"] == 1.5
    assert row["tflops_per_gpu"] == 700.0


def test_refresh_rejects_failed_nemo_job_without_final_iteration(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_124.yaml"
    _write_llmb_config(llmb_config, 124, framework="nemo2")
    (log_dir / "log-training_foo_124_0.out").write_text(
        "\n".join(
            f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: 1.5 | TFLOPS_per_GPU: 700.0"
            for iteration in range(MIN_ITERATION, MAX_ITERATION + 1)
        )
    )

    upsert_static_job(
        config,
        JobRecord(
            job_id=124,
            launcher_type="nemo",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=124,
        state="FAILED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="1:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={124: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [124])

    assert error is None and refreshed == 1
    row = get_job(config, 124)
    assert row["slurm_state"] == "FAILED"
    assert row["perf_parse_status"] == "incomplete"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None


def test_refresh_parses_megatron_bridge_finetune_logs(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "finetune_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_234.yaml"
    _write_llmb_config(llmb_config, 234, framework="megatron_bridge", workload_type="finetune")
    _write_megatron_bridge_perf_log(log_dir / "log-finetune_foo_234_0.out", time_seconds=2.0, tflops=800.0)

    upsert_static_job(
        config,
        JobRecord(
            job_id=234,
            launcher_type="megatron_bridge",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=234,
        state="COMPLETED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={234: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [234])

    assert error is None and refreshed == 1
    row = get_job(config, 234)
    assert row["perf_parse_status"] == "success"
    assert row["train_step_time_seconds"] == 2.0
    assert row["tflops_per_gpu"] == 800.0


def _nvcre_job_with_pod_log(tmp_path, cluster_config_factory, job_id, total_steps):
    """A completed nvcre job whose rank-0 pod log has warmup then steady-state steps."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory(
        {
            "launcher": {"llmb_install": str(llmb_install)},
            "execution_target": "kubernetes",
            "kubernetes": {"namespace": "ns"},
        }
    )
    log_dir = llmb_install / "workloads" / "w" / "experiments" / "run"
    pod_logs = log_dir / "pod_logs"
    pod_logs.mkdir(parents=True)
    llmb_config = log_dir / f"llmb-config_{job_id}.yaml"
    _write_llmb_config(llmb_config, job_id, framework="megatron_bridge")
    lines = []
    for iteration in range(1, total_steps + 1):
        seconds = 12.0 if iteration < MIN_ITERATION else 2.0  # warmup is slow
        lines.append(f"Step Time : {seconds:.2f}s GPU utilization: 800.0MODEL_TFLOP/s/GPU")
    (pod_logs / "wr-workload-node-0-0-xyz.log").write_text("\n".join(lines))  # the rank-0 pod
    (pod_logs / "wr-workload-node-0-1-xyz.log").write_text("another pod's log\n")
    upsert_static_job(
        config,
        JobRecord(
            job_id=job_id,
            launcher_type="megatron_bridge",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
            namespace="ns",
            workloadrun_name="wr",
        ),
    )
    with sqlite3.connect(get_history_db_path(llmb_install)) as conn:
        conn.execute("UPDATE jobs SET slurm_state = 'COMPLETED'")
        conn.commit()
    return config


def test_nvcre_pod_log_perf_uses_the_steady_state_window(tmp_path, cluster_config_factory):
    from llmb_run.job_history import _update_terminal_job_results

    config = _nvcre_job_with_pod_log(tmp_path, cluster_config_factory, 300, total_steps=50)

    _update_terminal_job_results(config, job_ids=[300], reparse_existing=True)

    row = get_job(config, 300)
    assert row["perf_parse_status"] == "success"
    assert row["train_step_time_seconds"] == 2.0  # the steady window, not the 12s warmup steps
    assert row["tflops_per_gpu"] == 800.0


def test_short_nvcre_runs_report_no_data_like_a_short_slurm_run(tmp_path, cluster_config_factory):
    from llmb_run.job_history import _update_terminal_job_results

    config = _nvcre_job_with_pod_log(tmp_path, cluster_config_factory, 301, total_steps=10)

    _update_terminal_job_results(config, job_ids=[301], reparse_existing=True)

    row = get_job(config, 301)
    # Zero samples in the steady-state window — the same status a 10-step
    # Slurm run reports, with no warmup-inclusive numbers recorded.
    assert row["perf_parse_status"] == "no_data"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None


def test_refresh_suppresses_megatron_bridge_perf_with_nan_grad_norm(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "finetune_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_235.yaml"
    _write_llmb_config(llmb_config, 235, framework="megatron_bridge", workload_type="finetune")
    _write_megatron_bridge_perf_log(
        log_dir / "log-finetune_foo_235_0.out", time_seconds=2.0, tflops=800.0, nan_grad_norm_iteration=33
    )

    upsert_static_job(
        config,
        JobRecord(
            job_id=235,
            launcher_type="megatron_bridge",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=235,
        state="COMPLETED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={235: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [235])

    assert error is None and refreshed == 1
    row = get_job(config, 235)
    assert row["slurm_state"] == "COMPLETED"
    assert row["perf_parse_status"] == "invalid_grad_norm"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None
    table = format_jobs_table([row], {})
    assert table.count("Invalid") == 2
    assert "grad_norm=nan" not in table
    assert "invalid: grad_norm=nan" in format_job_details(row, {})


def test_refresh_records_no_data_without_reparsing_until_forced(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_345.yaml"
    log_path = log_dir / "log-training_foo_345_0.out"
    _write_llmb_config(llmb_config, 345, framework="nemo2")
    log_path.write_text("started training but no timing samples yet")

    upsert_static_job(
        config,
        JobRecord(
            job_id=345,
            launcher_type="nemo",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=345,
        state="COMPLETED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={345: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [345])

    assert error is None and refreshed == 1
    row = get_job(config, 345)
    assert row["perf_parse_status"] == "no_data"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None

    _write_nemo_perf_log(log_path, time_seconds=1.5, tflops=700.0)
    with patch("llmb_run.job_history.parse_latest_pretrain_job_log") as mock_parse:
        refresh_non_terminal_jobs(config)
        mock_parse.assert_not_called()

    row = get_job(config, 345)
    assert row["perf_parse_status"] == "no_data"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None

    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={345: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [345])

    assert error is None and refreshed == 1
    row = get_job(config, 345)
    assert row["perf_parse_status"] == "success"
    assert row["train_step_time_seconds"] == 1.5
    assert row["tflops_per_gpu"] == 700.0


def test_refresh_records_no_log_without_reparsing(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_456.yaml"
    log_path = log_dir / "log-training_foo_456_0.out"
    _write_llmb_config(llmb_config, 456, framework="nemo2")

    upsert_static_job(
        config,
        JobRecord(
            job_id=456,
            launcher_type="nemo",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=456,
        state="COMPLETED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={456: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [456])

    assert error is None and refreshed == 1
    row = get_job(config, 456)
    assert row["perf_parse_status"] == "no_log"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None

    _write_nemo_perf_log(log_path, time_seconds=1.5, tflops=700.0)
    with patch("llmb_run.job_history.parse_latest_pretrain_job_log") as mock_parse:
        refresh_non_terminal_jobs(config)
        mock_parse.assert_not_called()

    row = get_job(config, 456)
    assert row["perf_parse_status"] == "no_log"
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None


@pytest.mark.parametrize(
    ("workload_key", "expected_window"),
    [
        ("finetune_llama2_mlperf", (485, 494)),
        ("pretrain_llama3.1_mlperf", (985, 994)),
        ("llama2", (MIN_ITERATION, MAX_ITERATION)),
    ],
)
def test_parse_pretrain_job_uses_workload_iteration_window(tmp_path, workload_key, expected_window):
    llmb_config = tmp_path / "llmb-config_456.yaml"
    _write_llmb_config(llmb_config, 456, framework="megatron_bridge")
    row = {
        "job_id": 456,
        "workload_key": workload_key,
        "llmb_config_path": str(llmb_config),
        "log_dir": str(tmp_path),
    }

    with patch("llmb_run.job_history.parse_latest_pretrain_job_log") as mock_parse:
        _parse_pretrain_job_performance(row)

    mock_parse.assert_called_once_with(
        str(tmp_path),
        456,
        "megatron_bridge",
        min_iteration=expected_window[0],
        max_iteration=expected_window[1],
    )


def test_refresh_does_not_parse_non_terminal_jobs(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_567.yaml"
    _write_llmb_config(llmb_config, 567, framework="nemo2")
    _write_nemo_perf_log(log_dir / "log-training_foo_567_0.out", time_seconds=1.5, tflops=700.0)

    upsert_static_job(
        config,
        JobRecord(
            job_id=567,
            launcher_type="nemo",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=567,
        state="RUNNING",
        elapsed="00:01:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with (
        patch("llmb_run.job_history.get_slurm_job_statuses", return_value={567: sacct_record}),
        patch("llmb_run.job_history.parse_latest_pretrain_job_log") as mock_parse,
    ):
        refreshed, error = refresh_requested_jobs(config, [567])

    assert error is None and refreshed == 1
    mock_parse.assert_not_called()
    row = get_job(config, 567)
    assert row["slurm_state"] == "RUNNING"
    assert row["perf_parse_status"] is None


def test_refresh_skips_unsupported_framework(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_678.yaml"
    _write_llmb_config(llmb_config, 678, framework="torchtitan")
    _write_nemo_perf_log(log_dir / "log-training_foo_678_0.out", time_seconds=1.5, tflops=700.0)

    upsert_static_job(
        config,
        JobRecord(
            job_id=678,
            launcher_type="nemo",
            model_name="foo",
            model_size="7b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=678,
        state="COMPLETED",
        elapsed="00:05:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with (
        patch("llmb_run.job_history.get_slurm_job_statuses", return_value={678: sacct_record}),
        patch("llmb_run.job_history.parse_latest_pretrain_job_log") as mock_parse,
    ):
        refreshed, error = refresh_requested_jobs(config, [678])

    assert error is None and refreshed == 1
    mock_parse.assert_not_called()
    row = get_job(config, 678)
    assert row["perf_parse_status"] is None
    assert row["train_step_time_seconds"] is None
    assert row["tflops_per_gpu"] is None


def test_refresh_marks_missing_jobs_as_purged(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(config, JobRecord(job_id=111, launcher_type="configured_sbatch"))
    upsert_static_job(config, JobRecord(job_id=222, launcher_type="configured_sbatch"))

    # Push created_at past the purge grace window so the test exercises the
    # PURGE branch, not the grace-period skip.
    db_path = get_history_db_path(llmb_install)
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE jobs SET created_at = '2020-01-01T00:00:00+00:00'")

    surviving = SlurmAccountingRecord(
        job_id=111,
        state="RUNNING",
        elapsed="00:01:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n1",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={111: surviving}):
        refreshed, error = refresh_non_terminal_jobs(config)

    assert error is None
    assert refreshed == 2  # one updated, one purged
    assert get_job(config, 111)["slurm_state"] == "RUNNING"
    assert get_job(config, 222)["slurm_state"] == "PURGED"

    # Subsequent refresh skips the purged job (now considered terminal).
    with patch("llmb_run.job_history.get_slurm_job_statuses") as mock_sacct:
        mock_sacct.return_value = {111: surviving}
        refresh_non_terminal_jobs(config)
        mock_sacct.assert_called_once_with([111], start_time="2026-04-16")


def test_refresh_chunks_and_bounds_sacct_queries(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    job_ids = list(range(1, 152))

    for job_id in job_ids:
        submit_date = "2026-04-17" if job_id <= 150 else "2026-04-18"
        upsert_static_job(
            config,
            JobRecord(
                job_id=job_id,
                launcher_type="configured_sbatch",
                submit_time=f"{submit_date}T13:42:00",
            ),
        )

    with patch("llmb_run.job_history.get_slurm_job_statuses", side_effect=[{}, {}]) as mock_sacct:
        refreshed, error = refresh_non_terminal_jobs(config)

    assert error is None
    assert refreshed == 0
    assert mock_sacct.call_args_list == [
        call(job_ids[:150], start_time="2026-04-16"),
        call(job_ids[150:], start_time="2026-04-17"),
    ]


def test_refresh_does_not_purge_within_grace_period(tmp_path, cluster_config_factory):
    """A just-submitted job that hasn't reached sacct yet must not be PURGED."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(config, JobRecord(job_id=12345, launcher_type="configured_sbatch"))  # created_at = now

    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={}) as mock_sacct:
        refreshed, error = refresh_non_terminal_jobs(config)

    assert error is None
    assert refreshed == 0
    mock_sacct.assert_called_once_with([12345], start_time=None)
    # Job stays non-terminal so a subsequent refresh keeps polling it.
    assert get_job(config, 12345)["slurm_state"] is None


def test_refresh_returns_error_when_sacct_fails(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(config, JobRecord(job_id=42, launcher_type="configured_sbatch"))

    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value=None):
        refreshed, error = refresh_non_terminal_jobs(config)

    assert refreshed == 0
    assert error is not None
    # Job state must not be touched when sacct itself failed.
    assert get_job(config, 42)["slurm_state"] is None


def test_elapsed_since_formats_like_sacct():
    import datetime

    from llmb_run.job_history import _elapsed_since

    now = datetime.datetime.now(datetime.timezone.utc)
    submit_time = (now - datetime.timedelta(hours=2, minutes=5, seconds=9)).isoformat()
    assert _elapsed_since(submit_time) == "02:05:09"

    submit_time_days = (now - datetime.timedelta(days=1, hours=3, minutes=2, seconds=1)).isoformat()
    assert _elapsed_since(submit_time_days) == "1-03:02:01"

    assert _elapsed_since(None) == ""
    assert _elapsed_since("not-a-timestamp") == ""


def test_elapsed_since_handles_naive_local_submit_time():
    """job_launcher.py stamps nvcre submit_time with naive datetime.now() (local
    time, not UTC). Regression test for treating that as UTC, which made elapsed
    appear negative (clamped to 0) in timezones ahead of UTC."""
    import datetime

    from llmb_run.job_history import _elapsed_since

    local_now = datetime.datetime.now()
    submit_time = (local_now - datetime.timedelta(minutes=10)).isoformat(timespec="seconds")

    elapsed = _elapsed_since(submit_time)
    assert elapsed != "00:00:00"
    hours, minutes, seconds = (int(part) for part in elapsed.split(":"))
    assert hours == 0
    assert 9 <= minutes <= 10


def test_refresh_nvcre_job_records_elapsed_while_running(tmp_path, cluster_config_factory):
    import datetime

    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    submit_time = (datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=10)).isoformat()
    upsert_static_job(
        config,
        JobRecord(
            job_id=555,
            launcher_type="megatron_bridge",
            model_name="qwen3",
            model_size="30b",
            submit_time=submit_time,
            namespace="test-k8s-ns",
            workloadrun_name="pretrain-qwen3-30b-a3b",
        ),
    )

    with (
        patch("shutil.which", return_value="/usr/bin/nvcrectl"),
        patch("subprocess.run") as mock_run,
        patch("llmb_run.job_history.get_slurm_job_statuses", return_value={}),
    ):
        mock_run.return_value = type("Result", (), {"returncode": 0, "stdout": "InProgress\n", "stderr": ""})()
        refreshed, error = refresh_non_terminal_jobs(config)

    assert error is None
    assert refreshed == 1
    row = get_job(config, 555)
    assert row["slurm_state"] == "RUNNING"
    assert row["elapsed"] != ""
    # Roughly 10 minutes elapsed; allow slack for test execution time.
    hours, minutes, seconds = (int(part) for part in row["elapsed"].split(":"))
    assert hours == 0
    assert 9 <= minutes <= 11


def test_refresh_nvcre_job_leaves_elapsed_blank_while_pending(tmp_path, cluster_config_factory):
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=556,
            launcher_type="megatron_bridge",
            model_name="qwen3",
            model_size="30b",
            namespace="test-k8s-ns",
            workloadrun_name="pretrain-qwen3-30b-a3b",
        ),
    )

    with (
        patch("shutil.which", return_value="/usr/bin/nvcrectl"),
        patch("subprocess.run") as mock_run,
        patch("llmb_run.job_history.get_slurm_job_statuses", return_value={}),
    ):
        mock_run.return_value = type("Result", (), {"returncode": 0, "stdout": "Pending\n", "stderr": ""})()
        refreshed, error = refresh_non_terminal_jobs(config)

    assert error is None
    assert refreshed == 1
    row = get_job(config, 556)
    assert row["slurm_state"] == "PENDING"
    assert row["elapsed"] == ""


def _write_llmb_config(path, job_id, framework, workload_type="pretrain"):
    path.write_text(
        yaml.safe_dump(
            {
                "job_info": {"job_id": str(job_id), "launch_time": "2026-04-17T13:42:00"},
                "workload_info": {"framework": framework, "workload_type": workload_type},
                "model_info": {
                    "model_name": "foo",
                    "model_size": "7b",
                    "dtype": "fp16",
                    "scale": 8,
                },
                "job_config": {
                    "profile_enabled": False,
                    "proxy": False,
                    "env_overrides": {},
                    "model_overrides": {},
                },
            }
        )
    )


def _write_nemo_perf_log(path, time_seconds, tflops):
    lines = [
        f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: {time_seconds} | "
        f"TFLOPS_per_GPU: {tflops}"
        for iteration in range(MIN_ITERATION, MAX_ITERATION + 1)
    ]
    lines.append("Training epoch 0, iteration 49/49 | training complete")
    path.write_text("\n".join(lines))


def _write_megatron_bridge_perf_log(path, time_seconds, tflops, nan_grad_norm_iteration=None):
    elapsed_ms = time_seconds * 1000.0
    lines = []
    for iteration in range(1, 51):
        grad_norm = "nan" if iteration == nan_grad_norm_iteration else "1.000"
        lines.append(f"Step Time : {time_seconds:.2f}s GPU utilization: {tflops}MODEL_TFLOP/s/GPU")
        lines.append(
            f"iteration {iteration}/50 | elapsed time per iteration (ms): {elapsed_ms} | grad norm: {grad_norm} |"
        )
    path.write_text("\n".join(lines))


def _write_rl_metrics(log_dir, step_times, tokens_values):
    # metrics.json keys are 1-indexed step numbers (see check_metrics.py usage in launch.sh)
    data = {
        "timing/train/total_step_time": {str(i): v for i, v in enumerate(step_times, start=1)},
        "performance/tokens_per_sec_per_gpu": {str(i): v for i, v in enumerate(tokens_values, start=1)},
    }
    (log_dir / "metrics.json").write_text(json.dumps(data))


def _write_ray_driver_log(log_dir, max_steps):
    """Write a ray-driver.log with the resolved-config dump the trainer emits."""
    driver_dir = log_dir / "logs" / "grpo-run"
    driver_dir.mkdir(parents=True, exist_ok=True)
    (driver_dir / "ray-driver.log").write_text(
        f"Final config:\n{{'grpo': {{'max_num_steps': {max_steps}}}, 'policy': {{'train_micro_batch_size': 1}}}}\n"
    )


_RL_WORKLOADS = {"rl_deepseek-v3": {"metadata": {"general": {"framework": "nemo-rl"}}}}


# ---------------------------------------------------------------------------
# RL-specific tests
# ---------------------------------------------------------------------------


def test_schema_migration_adds_tokens_column_to_existing_v1_db(tmp_path, cluster_config_factory):
    """Opening a v1 DB (no tokens_per_sec_per_gpu column) must add the column and bump version."""
    llmb_install = tmp_path / "llmb"
    db_dir = llmb_install / ".llmb"
    db_dir.mkdir(parents=True)
    db_path = db_dir / "jobs.sqlite3"

    # Create a v1 DB manually — all original columns except tokens_per_sec_per_gpu.
    with sqlite3.connect(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)
        """)
        conn.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
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
                perf_parse_status TEXT
            )
        """)
        conn.execute("INSERT INTO jobs (job_id, launcher_type) VALUES (1, 'nemo')")
        conn.commit()

    # Opening the DB through llmb_run should run the migration.
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    rows = list_jobs(config)

    # Old row survives with NULL for the new column.
    assert len(rows) == 1
    assert rows[0]["tokens_per_sec_per_gpu"] is None

    with sqlite3.connect(db_path) as conn:
        version = conn.execute("SELECT value FROM metadata WHERE key = 'schema_version'").fetchone()[0]
        columns = {row[1] for row in conn.execute("PRAGMA table_info(jobs)")}
    assert version == "3"
    assert "tokens_per_sec_per_gpu" in columns


def test_refresh_parses_rl_metrics_for_terminal_job(tmp_path, cluster_config_factory):
    """A completed RL job should populate s/iter and tokens_per_sec_per_gpu."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "rl_deepseek-v3" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_9001.yaml"
    _write_llmb_config(llmb_config, 9001, framework="nemo-rl", workload_type="rl")

    step_times = [60.0 + i * 0.5 for i in range(10)]
    tokens = [25.0 + i * 0.1 for i in range(10)]
    _write_rl_metrics(log_dir, step_times, tokens)
    _write_ray_driver_log(log_dir, 10)

    upsert_static_job(
        config,
        JobRecord(
            job_id=9001,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=9001,
        state="COMPLETED",
        elapsed="01:02:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n[1-64]",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={9001: sacct_record}):
        refreshed, error = refresh_requested_jobs(config, [9001])

    assert error is None and refreshed == 1
    row = get_job(config, 9001)
    assert row["perf_parse_status"] == "success"

    expected_step_time = statistics.mean(step_times[2:7])
    expected_tokens = statistics.mean(tokens[2:7])
    assert row["train_step_time_seconds"] == pytest.approx(expected_step_time)
    assert row["tokens_per_sec_per_gpu"] == pytest.approx(expected_tokens)
    assert row["tflops_per_gpu"] is None  # RL has no TFLOPS metric


def test_refresh_rl_job_reads_max_steps_from_ray_driver_log(tmp_path, cluster_config_factory):
    """The run's ray-driver.log config dump records the step count actually used.

    A 10-step metrics file from a 20-step run must be INCOMPLETE even though
    nothing in the DB mentions MAX_STEPS (e.g. `MAX_STEPS=20 llmb-run submit`
    sets it as an ambient env var that never reaches env_overrides_json).
    """
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "rl_deepseek-v3" / "experiments" / "run3"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_9003.yaml"
    _write_llmb_config(llmb_config, 9003, framework="nemo-rl", workload_type="rl")

    _write_ray_driver_log(log_dir, 20)
    # Job died after 10 of its 20 steps.
    _write_rl_metrics(log_dir, [60.0] * 10, [25.0] * 10)

    upsert_static_job(
        config,
        JobRecord(
            job_id=9003,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=9003,
        state="COMPLETED",
        elapsed="01:02:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n[1-64]",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={9003: sacct_record}):
        refresh_requested_jobs(config, [9003])

    row = get_job(config, 9003)
    assert row["perf_parse_status"] == "incomplete"
    assert row["train_step_time_seconds"] is None


def test_refresh_rl_job_without_run_log_is_no_data(tmp_path, cluster_config_factory):
    """A completed RL job with metrics but no ray-driver.log can't be validated → no_data."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "rl_deepseek-v3" / "experiments" / "run2"
    log_dir.mkdir(parents=True)
    llmb_config = log_dir / "llmb-config_9002.yaml"
    _write_llmb_config(llmb_config, 9002, framework="nemo-rl", workload_type="rl")

    _write_rl_metrics(log_dir, [60.0] * 10, [25.0] * 10)

    upsert_static_job(
        config,
        JobRecord(
            job_id=9002,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
            log_dir=str(log_dir),
            llmb_config_path=str(llmb_config),
        ),
    )

    sacct_record = SlurmAccountingRecord(
        job_id=9002,
        state="COMPLETED",
        elapsed="01:02:00",
        submit_time="2026-04-17T13:42:00",
        node_list="n[1-64]",
        exit_code="0:0",
    )
    with patch("llmb_run.job_history.get_slurm_job_statuses", return_value={9002: sacct_record}):
        refresh_requested_jobs(config, [9002])

    row = get_job(config, 9002)
    assert row["perf_parse_status"] == "no_data"
    assert row["train_step_time_seconds"] is None


def test_format_jobs_table_hides_tokens_column_for_pretrain_only(tmp_path, cluster_config_factory):
    """Tokens/s/GPU column must be absent when no RL workload is in the displayed rows."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=1,
            launcher_type="nemo",
            workload_key="pretrain_foo",
            model_name="foo",
            model_size="7b",
        ),
    )

    rows = list_jobs(config)
    table = format_jobs_table(rows, _RL_WORKLOADS)
    assert "TFLOPS/GPU" in table
    assert "Tokens/s/GPU" not in table


def test_format_jobs_table_never_truncates_the_job_id(tmp_path, cluster_config_factory):
    """A populated row squeezes the table; a 16-digit nvcre id must stay intact."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    upsert_static_job(
        config,
        JobRecord(
            job_id=1791408214621705,
            launcher_type="megatron_bridge",
            workload_key="pretrain_nemotron3.5_30b",
            model_name="nemotron3.5_30b",
            model_size="30b",
            dtype="fp8",
            scale=8,
            namespace="k8s-platform-north",
            workloadrun_name="pretrain-nemotron-3-5-lightning-fp8-mx-gpus8-gb300-39aa80",
        ),
    )
    with sqlite3.connect(get_history_db_path(llmb_install)) as conn:
        conn.execute(
            "UPDATE jobs SET slurm_state = ?, tflops_per_gpu = ?, train_step_time_seconds = ? WHERE job_id = ?",
            ("COMPLETED", 958.57, 15.40, 1791408214621705),
        )

    table = format_jobs_table(list_jobs(config), {})

    assert "1791408214621705" in table  # the full id renders, never an ellipsis-mangled copy
    assert "17914082146217…" not in table


def test_format_jobs_table_shows_tokens_column_when_rl_present(tmp_path, cluster_config_factory):
    """Tokens/s/GPU column appears when at least one displayed row is an RL job."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    # Pretrain job — no tokens metric.
    upsert_static_job(
        config,
        JobRecord(job_id=1, launcher_type="nemo", workload_key="pretrain_foo", model_name="foo", model_size="7b"),
    )
    # RL job — tokens metric populated.
    upsert_static_job(
        config,
        JobRecord(
            job_id=2,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
        ),
    )

    db_path = get_history_db_path(llmb_install)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tflops_per_gpu = ?, train_step_time_seconds = ? WHERE job_id = ?",
            (456.7, 12.34, 1),
        )
        conn.execute(
            "UPDATE jobs SET tokens_per_sec_per_gpu = ?, train_step_time_seconds = ? WHERE job_id = ?",
            (26.94, 61.20, 2),
        )

    rows = list_jobs(config)
    table = format_jobs_table(rows, _RL_WORKLOADS)

    assert "TFLOPS/GPU" in table
    assert "Tokens/s/GPU" in table
    # Pretrain row shows TFLOPS value; RL row shows tokens value.
    assert "456.70" in table
    assert "26.94" in table
    # RL row TFLOPS cell is blank (no value); pretrain tokens cell is blank too.
    assert "61.20" in table
    assert "12.34" in table


def test_format_jobs_table_shows_tokens_column_for_unparsed_rl_job(tmp_path, cluster_config_factory):
    """Tokens/s/GPU column appears for RL rows even before metrics are populated."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=3,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
        ),
    )

    table = format_jobs_table(list_jobs(config), _RL_WORKLOADS)

    assert "Tokens/s/GPU" in table


def test_format_job_details_shows_tflops_for_pretrain(tmp_path, cluster_config_factory):
    """A pretrain job's details show TFLOPS/GPU, not Tokens/s/GPU."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(job_id=10, launcher_type="nemo", workload_key="pretrain_foo", model_name="foo", model_size="7b"),
    )
    db_path = get_history_db_path(llmb_install)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tflops_per_gpu = ?, train_step_time_seconds = ? WHERE job_id = ?", (800.0, 10.0, 10)
        )

    row = get_job(config, 10)
    details = format_job_details(row, _RL_WORKLOADS)
    assert "TFLOPS/GPU" in details
    assert "800.00" in details
    assert "Tokens/s/GPU" not in details


def test_format_job_details_shows_tokens_for_rl(tmp_path, cluster_config_factory):
    """An RL job's details show Tokens/s/GPU, not TFLOPS/GPU."""
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=20,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
        ),
    )
    db_path = get_history_db_path(llmb_install)
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "UPDATE jobs SET tokens_per_sec_per_gpu = ?, train_step_time_seconds = ? WHERE job_id = ?",
            (26.94, 61.20, 20),
        )

    row = get_job(config, 20)
    details = format_job_details(row, _RL_WORKLOADS)
    assert "Tokens/s/GPU" in details
    assert "26.94" in details
    assert "TFLOPS/GPU" not in details


def test_format_job_details_unparsed_rl_job_shows_tokens_label(tmp_path, cluster_config_factory):
    """An RL job with no parsed metrics yet should still show Tokens/s/GPU label (not TFLOPS/GPU).

    Detection keys off the workload's framework in the loaded metadata,
    not the metric column value.
    """
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=30,
            launcher_type="configured_sbatch",
            workload_key="rl_deepseek-v3",
            model_size="671b",
        ),
    )

    row = get_job(config, 30)
    assert row["tokens_per_sec_per_gpu"] is None  # not yet parsed
    assert row["tflops_per_gpu"] is None
    details = format_job_details(row, _RL_WORKLOADS)
    assert "Tokens/s/GPU" in details
    assert "TFLOPS/GPU" not in details


def test_format_job_details_non_rl_configured_sbatch_shows_tflops(tmp_path, cluster_config_factory):
    """A non-RL configured_sbatch job (e.g. inference) must show TFLOPS/GPU, not Tokens/s/GPU.

    launcher_type must not be used to detect RL — it is shared by many recipe types.
    """
    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})

    upsert_static_job(
        config,
        JobRecord(
            job_id=40,
            launcher_type="configured_sbatch",
            workload_key="inference_llama3",
            model_size="70b",
        ),
    )

    row = get_job(config, 40)
    details = format_job_details(row, {"inference_llama3": {"metadata": {"general": {"framework": "vllm"}}}})
    assert "TFLOPS/GPU" in details
    assert "Tokens/s/GPU" not in details


def _kubernetes_config(tmp_path, cluster_config_factory):
    return cluster_config_factory(
        {
            "launcher": {"llmb_install": str(tmp_path / "llmb")},
            "execution_target": "kubernetes",
            "kubernetes": {"namespace": "ns"},
        }
    )


def test_kubernetes_jobs_are_never_sent_to_sacct(tmp_path, cluster_config_factory):
    config = _kubernetes_config(tmp_path, cluster_config_factory)
    # nvcre ids are synthetic; one row has its WorkloadRun name recorded and one does not.
    upsert_static_job(
        config,
        JobRecord(job_id=1790000000001, launcher_type="megatron_bridge", namespace="ns", workloadrun_name="wr-a"),
    )
    upsert_static_job(config, JobRecord(job_id=1790000000002, launcher_type="megatron_bridge", namespace="ns"))
    db_path = get_history_db_path(tmp_path / "llmb")
    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE jobs SET slurm_state = 'RUNNING', created_at = '2020-01-01T00:00:00+00:00'")

    with (
        patch("llmb_run.job_history.get_slurm_job_statuses", return_value={}) as sacct,
        patch("llmb_run.job_history._refresh_nvcre_statuses", return_value=1) as nvcre,
    ):
        refreshed, error = refresh_non_terminal_jobs(config)

    sacct.assert_not_called()  # sacct succeeding without these ids is what used to mark them PURGED
    assert error is None
    assert refreshed == 1
    assert [r["workloadrun_name"] for r in nvcre.call_args.args[1]] == ["wr-a"]
    assert get_job(config, 1790000000001)["slurm_state"] == "RUNNING"
    assert get_job(config, 1790000000002)["slurm_state"] == "RUNNING"


def test_requested_refresh_on_kubernetes_uses_nvcre_not_sacct(tmp_path, cluster_config_factory):
    config = _kubernetes_config(tmp_path, cluster_config_factory)
    upsert_static_job(
        config,
        JobRecord(job_id=1790000000001, launcher_type="megatron_bridge", namespace="ns", workloadrun_name="wr-a"),
    )

    with (
        patch("llmb_run.job_history.get_slurm_job_statuses") as sacct,
        patch("llmb_run.job_history._refresh_nvcre_statuses", return_value=1) as nvcre,
    ):
        refreshed, error = refresh_requested_jobs(config, [1790000000001])

    sacct.assert_not_called()
    assert (refreshed, error) == (1, None)
    assert [r["job_id"] for r in nvcre.call_args.args[1]] == [1790000000001]


def test_kubernetes_cluster_refuses_workloads_that_only_support_slurm(tmp_path, cluster_config_factory):
    config = _kubernetes_config(tmp_path, cluster_config_factory)
    ready = WorkloadTask("ready", "7b", "bf16", 8)
    slurm_only = WorkloadTask("mlperf", "7b", "bf16", 8)
    workloads = {
        "ready": {
            "metadata": {"run": {"launcher_type": "megatron_bridge", "execution_targets": ["slurm", "kubernetes"]}}
        },
        "mlperf": {"metadata": {"run": {"launcher_type": "megatron_bridge"}}},  # its launch.sh rejects kubernetes
    }
    launched = []

    class FakeLauncher:
        def launch(self, task):
            launched.append(task.workload_key)
            return SlurmJob(job_id=789, job_workdir=None, llmb_config_path=None)

    with patch("llmb_run.job_launcher.create_launcher", return_value=FakeLauncher()) as create:
        with pytest.raises(RuntimeError, match="Failed to launch 1 out of 2"):
            run_tests(config, [ready, slurm_only], workloads)

    assert launched == ["ready"]  # the Slurm-only workload never reached a launcher
    assert [c.args[0] for c in create.call_args_list] == ["nvcre"]


def test_kubernetes_identifiers_are_recorded_whatever_the_metadata_launcher(tmp_path, cluster_config_factory):
    config = _kubernetes_config(tmp_path, cluster_config_factory)
    config.extras["kubernetes"] = {"namespace": "ns"}
    llmb_config = tmp_path / "llmb" / "workloads" / "w" / "experiments" / "run" / "llmb-config_9.yaml"
    task = WorkloadTask("w", "7b", "bf16", 8)
    workloads = {"w": {"metadata": {"run": {"launcher_type": "nemo"}}, "workload": "w"}}  # not megatron_bridge

    record_job_submission(
        config, task, SlurmJob(job_id=9, job_workdir=None, llmb_config_path=str(llmb_config)), workloads
    )

    assert get_job(config, 9)["namespace"] == "ns"  # without it the nvcre refresh cannot find the row


def test_nvcre_rows_are_refreshed_whatever_their_launcher_type(tmp_path, cluster_config_factory):
    config = _kubernetes_config(tmp_path, cluster_config_factory)
    upsert_static_job(config, JobRecord(job_id=5, launcher_type="nemo", namespace="ns", workloadrun_name="wr"))

    with patch("llmb_run.job_history._refresh_nvcre_statuses", return_value=1) as nvcre:
        refresh_non_terminal_jobs(config)

    assert [r["job_id"] for r in nvcre.call_args.args[1]] == [5]


def test_nvcre_status_query_is_bounded_and_a_timeout_keeps_the_last_state(tmp_path, cluster_config_factory):
    import subprocess

    from llmb_run.job_history import _refresh_nvcre_statuses

    config = _kubernetes_config(tmp_path, cluster_config_factory)
    upsert_static_job(
        config, JobRecord(job_id=5, launcher_type="megatron_bridge", namespace="ns", workloadrun_name="wr")
    )
    row = list_jobs(config)[0]

    with (
        patch("shutil.which", return_value="/usr/bin/nvcrectl"),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvcrectl", 30)) as run,
    ):
        refreshed = _refresh_nvcre_statuses(config, [row])

    assert refreshed == 0
    assert run.call_args.kwargs["timeout"] > 0
    assert get_job(config, 5)["slurm_state"] != "COMPLETED"


def test_failed_nvcre_jobs_pull_pod_logs_and_the_failure_configmap(tmp_path, cluster_config_factory):
    import subprocess

    from llmb_run.job_history import _nvcre_post_run_on_completion

    config = _kubernetes_config(tmp_path, cluster_config_factory)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    upsert_static_job(
        config,
        JobRecord(
            job_id=6,
            launcher_type="megatron_bridge",
            namespace="ns",
            workloadrun_name="wr",
            log_dir=str(run_dir),
        ),
    )
    row = list_jobs(config)[0]
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        if argv[:3] == ["kubectl", "get", "pods"]:
            return subprocess.CompletedProcess(argv, 0, stdout="wr-rank0-pod\n", stderr="")
        if argv[:2] == ["kubectl", "logs"]:
            return subprocess.CompletedProcess(argv, 0, stdout="boom\n", stderr="")
        if argv[:3] == ["kubectl", "describe", "workloadrun"]:
            return subprocess.CompletedProcess(argv, 0, stdout="Failed Nodes Ref:  fail-cm\n", stderr="")
        if argv[:3] == ["kubectl", "get", "configmap"]:
            return subprocess.CompletedProcess(argv, 0, stdout='{"data": {"nodes": "n1"}}', stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")

    with patch("subprocess.run", fake_run):
        _nvcre_post_run_on_completion(config, row, "FAILED")

    assert (run_dir / "pod_logs" / "wr-rank0-pod.log").read_text() == "boom\n"  # the failure's pod logs
    assert (run_dir / "fail.log").exists()  # and the failure configmap alongside them
    assert not any("workloadrun" in c and "cancel" in c for c in seen)  # failure keeps the WorkloadRun


def test_failed_nvcre_configmap_survives_a_broken_log_pull(tmp_path, cluster_config_factory):
    import subprocess

    from llmb_run.job_history import _nvcre_post_run_on_completion

    config = _kubernetes_config(tmp_path, cluster_config_factory)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    upsert_static_job(
        config,
        JobRecord(
            job_id=8,
            launcher_type="megatron_bridge",
            namespace="ns",
            workloadrun_name="wr",
            log_dir=str(run_dir),
        ),
    )
    row = list_jobs(config)[0]

    def fake_run(argv, **kwargs):
        if argv[:3] == ["kubectl", "get", "pods"]:
            return subprocess.CompletedProcess(argv, 0, stdout="wr-rank0-pod\n", stderr="")
        if argv[:2] == ["kubectl", "logs"]:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])  # the log pull hangs
        if argv[:3] == ["kubectl", "describe", "workloadrun"]:
            return subprocess.CompletedProcess(argv, 0, stdout="Failed Nodes Ref:  fail-cm\n", stderr="")
        if argv[:3] == ["kubectl", "get", "configmap"]:
            return subprocess.CompletedProcess(argv, 0, stdout='{"data": {"nodes": "n1"}}', stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")

    with patch("subprocess.run", fake_run):
        _nvcre_post_run_on_completion(config, row, "FAILED")

    assert (run_dir / "fail.log").exists()  # the cheap configmap ran before the log pull hung


def test_synthetic_job_ids_are_unique_even_within_one_clock_tick():
    from llmb_run.job_launcher import _synthetic_job_id

    with patch("llmb_run.job_launcher.time.time_ns", return_value=1_790_000_000_000_000_000):
        ids = [_synthetic_job_id() for _ in range(1000)]

    assert len(set(ids)) == 1000
    assert ids == sorted(ids)


def test_jobs_show_does_not_hang_on_a_stuck_nvcrectl_report():
    import subprocess
    from types import SimpleNamespace

    from llmb_run import main

    row = {"workloadrun_name": "wr", "namespace": "ns"}
    app_ctx = SimpleNamespace(
        cluster_config=SimpleNamespace(extras={"kubernetes": {}}, execution_target=ExecutionTarget.KUBERNETES)
    )
    with (
        patch.object(main, "_ctx_app_context", return_value=app_ctx),
        patch.object(main, "refresh_non_terminal_jobs", return_value=(0, None)),
        patch.object(main, "_get_job_or_exit", return_value=row),
        patch.object(main, "format_job_details", return_value="details"),
        patch.object(main, "_display_workloads", return_value={}),
        patch("shutil.which", return_value="/usr/bin/nvcrectl"),
        patch("subprocess.run", side_effect=subprocess.TimeoutExpired("nvcrectl", 30)) as run,
    ):
        main.jobs_show(MagicMock(), 5)  # returns instead of blocking

    assert run.call_args.kwargs["timeout"] > 0


class _FakeKubectl:
    """Scripted kubectl: records every call and answers per the first matching rule."""

    def __init__(self, workloadrun=None, pods_by_label=None):
        self.calls = []
        self.workloadrun = workloadrun if workloadrun is not None else {"status": {}, "metadata": {}}
        self.pods_by_label = pods_by_label or {}

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        if argv[:3] == ["kubectl", "get", "workloadrun"]:
            return subprocess.CompletedProcess(argv, 0, stdout=json.dumps(self.workloadrun), stderr="")
        if argv[:3] == ["kubectl", "get", "pods"]:
            label = argv[argv.index("-l") + 1]
            return subprocess.CompletedProcess(argv, 0, stdout=self.pods_by_label.get(label, ""), stderr="")
        if argv[:2] == ["kubectl", "logs"]:
            return subprocess.CompletedProcess(argv, 0, stdout="log text", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")

    def kinds(self):
        return [" ".join(c[1:3]) for c in self.calls]


def test_unresolvable_jobset_is_not_guessed_from_other_jobs_in_the_namespace(tmp_path):
    from llmb_run.job_history import _nvcre_internal_job_name

    fake = _FakeKubectl()  # the WorkloadRun exposes no internal job name
    with patch("subprocess.run", fake):
        assert _nvcre_internal_job_name("ns", "wr") is None

    assert "get jobsets" not in fake.kinds()  # the namespace-wide "newest JobSet" guess is gone


def test_log_pull_stays_scoped_to_the_jobs_own_jobset(tmp_path):
    from llmb_run.job_history import _nvcre_pull_all_logs

    other_jobs_pods = {"jobset.sigs.k8s.io/jobset-name": "someone-elses-pod\n"}
    fake = _FakeKubectl(pods_by_label=other_jobs_pods)
    local_dir = tmp_path / "run"
    local_dir.mkdir()

    with patch("subprocess.run", fake):
        _nvcre_pull_all_logs("ns", "wr", str(local_dir))

    assert not any(c[:2] == ["kubectl", "logs"] for c in fake.calls)  # nothing fetched from another job's pod
    assert not (local_dir / "pod_logs").exists()
    for argv in fake.calls:
        if argv[:3] == ["kubectl", "get", "pods"]:
            assert argv[argv.index("-l") + 1] == "jobset.sigs.k8s.io/jobset-name=wr-workload"


def test_failure_configmap_kubectl_calls_are_bounded(tmp_path):
    from llmb_run.job_history import _nvcre_extract_failure_configmap

    def describe_then_hang(argv, **kwargs):
        assert kwargs["timeout"] > 0  # both calls are bounded
        if argv[1] == "describe":
            return subprocess.CompletedProcess(argv, 0, stdout="Failed Nodes Ref:  fail-cm\n", stderr="")
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    with patch("subprocess.run", side_effect=describe_then_hang):
        _nvcre_extract_failure_configmap("ns", "wr", str(tmp_path))  # returns instead of hanging

    assert not (tmp_path / "fail.log").exists()


def test_failure_describe_timeout_is_handled(tmp_path):
    from llmb_run.job_history import _nvcre_extract_failure_configmap

    with patch("subprocess.run", side_effect=subprocess.TimeoutExpired("kubectl", 30)) as run:
        _nvcre_extract_failure_configmap("ns", "wr", str(tmp_path))

    assert run.call_args.kwargs["timeout"] > 0


def _nvcre_launch_env(tmp_path, cluster_config_factory, gpu_type, images, **task_kwargs):
    """Run NVCreLauncher.launch up to the launch.sh call and return the environment it passes."""
    from llmb_run.job_launcher import NVCreLauncher

    # get_venv_environment validates that the venv (and its python3) exists on disk.
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True, exist_ok=True)
    (bin_dir / "python3").touch()
    venv_path = str(tmp_path / "venv")
    config = cluster_config_factory(
        {
            "launcher": {"llmb_install": str(tmp_path / "llmb"), "gpu_type": gpu_type},
            "execution_target": "kubernetes",
            "kubernetes": {"namespace": "ns"},
            "workloads": {"config": {"w": {"venv_path": venv_path, "venv_type": "uv"}}},
        }
    )
    workloads = {
        "w": {
            "dir": str(tmp_path),
            "metadata": {
                "container": {"images": images},
                "run": {"launcher_type": "megatron_bridge", "launch_script": "launch.sh"},
            },
        }
    }
    captured = {}

    def fake_run(argv, **kwargs):
        captured["env"] = kwargs["env"]
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="stop here")

    with patch("subprocess.run", side_effect=fake_run):
        NVCreLauncher(config, workloads).launch(WorkloadTask("w", "7b", "bf16", 8, **task_kwargs))
    return captured["env"]


def test_nvcre_launch_resolves_by_gpu_container_images(tmp_path, cluster_config_factory):
    # The shape upstream metadata uses today; indexing it like a list raised KeyError(0), shown as "Submission error: 0".
    images = {
        "by_gpu": {
            "default": ["nvcr.io#nvidia/nemo:26.08.01"],
            "h100": ["nvcr.io#nvidia/nemo:26.06.01"],
        }
    }

    assert (
        _nvcre_launch_env(tmp_path, cluster_config_factory, "gb300", images)["RUN_CONF_IMAGE"]
        == "nvcr.io/nvidia/nemo:26.08.01"
    )  # the default entry
    assert (
        _nvcre_launch_env(tmp_path, cluster_config_factory, "h100", images)["RUN_CONF_IMAGE"]
        == "nvcr.io/nvidia/nemo:26.06.01"
    )  # a GPU-specific entry wins


def test_nvcre_launch_still_accepts_a_plain_image_list(tmp_path, cluster_config_factory):
    env = _nvcre_launch_env(tmp_path, cluster_config_factory, "gb300", ["nvcr.io#nvidia/nemo:26.08.00"])

    assert env["RUN_CONF_IMAGE"] == "nvcr.io/nvidia/nemo:26.08.00"


@pytest.mark.parametrize(
    "task_kwarg, env_var",
    [("proxy", "PROXY_WORKLOAD"), ("profile", "ENABLE_PROFILE")],
)
def test_nvcre_launch_passes_task_flags_to_launch_sh(tmp_path, cluster_config_factory, task_kwarg, env_var):
    """--proxy / --profile must reach launch.sh, or its reduced-size and profiling overrides are skipped."""
    images = ["nvcr.io#nvidia/nemo:26.08.00"]

    assert _nvcre_launch_env(tmp_path, cluster_config_factory, "gb300", images, **{task_kwarg: True})[env_var] == "true"
    assert env_var not in _nvcre_launch_env(tmp_path, cluster_config_factory, "gb300", images)


def test_nvcre_launch_activates_the_workload_venv(tmp_path, cluster_config_factory):
    """Like Nemo2Launcher, the nvcre launcher runs launch.sh inside the workload venv."""
    env = _nvcre_launch_env(tmp_path, cluster_config_factory, "gb300", ["nvcr.io#nvidia/nemo:26.08.00"])

    venv_path = str(tmp_path / "venv")
    # the venv's bin dir is first on PATH, so launch.sh's plain `python3` resolves inside it
    assert env["PATH"].split(os.pathsep)[0] == os.path.join(venv_path, "bin")
    assert env["VIRTUAL_ENV"] == venv_path
    assert "LLMB_PYTHON" not in env  # the POC-era interpreter pin is gone


def test_completed_nvcre_job_is_cancelled_after_a_successful_pull(tmp_path, cluster_config_factory):
    import subprocess

    from llmb_run.job_history import _nvcre_post_run_on_completion

    config = _kubernetes_config(tmp_path, cluster_config_factory)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    upsert_static_job(
        config,
        JobRecord(
            job_id=10,
            launcher_type="megatron_bridge",
            namespace="ns",
            workloadrun_name="wr",
            log_dir=str(run_dir),
        ),
    )
    row = list_jobs(config)[0]
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        if argv[:3] == ["kubectl", "get", "pods"]:
            return subprocess.CompletedProcess(argv, 0, stdout="wr-rank0-pod\n", stderr="")
        if argv[:2] == ["kubectl", "logs"]:
            return subprocess.CompletedProcess(argv, 0, stdout="log text\n", stderr="")
        if argv[1:3] == ["workloadrun", "cancel"]:
            return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")

    with patch("subprocess.run", fake_run):
        _nvcre_post_run_on_completion(config, row, "COMPLETED")

    assert any(c[1:3] == ["workloadrun", "cancel"] for c in seen)  # resources released
    assert (run_dir / "pod_logs" / "wr-rank0-pod.log").exists()


def test_completed_nvcre_job_is_not_cancelled_when_the_log_pull_fails(tmp_path, cluster_config_factory):
    """Cancelling deletes the pods — the only remaining copy of the logs — so a failed pull must keep them."""
    import subprocess

    from llmb_run.job_history import _nvcre_post_run_on_completion

    config = _kubernetes_config(tmp_path, cluster_config_factory)
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    upsert_static_job(
        config,
        JobRecord(
            job_id=11,
            launcher_type="megatron_bridge",
            namespace="ns",
            workloadrun_name="wr",
            log_dir=str(run_dir),
        ),
    )
    row = list_jobs(config)[0]
    seen = []

    def fake_run(argv, **kwargs):
        seen.append(argv)
        if argv[:3] == ["kubectl", "get", "pods"]:
            return subprocess.CompletedProcess(argv, 0, stdout="wr-rank0-pod\n", stderr="")
        if argv[:2] == ["kubectl", "logs"]:
            return subprocess.CompletedProcess(argv, 1, stdout="", stderr="boom")  # every fetch fails
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")

    with patch("subprocess.run", fake_run):
        _nvcre_post_run_on_completion(config, row, "COMPLETED")

    assert not any(c[1:3] == ["workloadrun", "cancel"] for c in seen)  # pods kept, logs still recoverable
    assert not (run_dir / "pod_logs.tar.gz").exists()
