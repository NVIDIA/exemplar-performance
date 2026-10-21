# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from unittest.mock import patch

from llmb_run.job_history import record_job_submission
from llmb_run.job_logs import active_job_log, find_configured_sbatch_logs, find_job_logs, find_srt_slurm_logs, read_tail
from llmb_run.slurm_utils import SlurmJob
from llmb_run.tasks import WorkloadTask


def test_find_job_logs_filters_by_job_id_and_sorts_by_retry(tmp_path):
    log_dir = tmp_path / "experiment"
    log_dir.mkdir()
    (log_dir / "log-general_12345_0.out").write_text("first")
    (log_dir / "log-general_12345_2.out").write_text("active")
    (log_dir / "log-general_99999_9.out").write_text("debug rerun")
    (log_dir / "log-general_12345_retry.out").write_text("not a retry count")
    (log_dir / "slurm-12345.out").write_text("not a workload log")

    logs = find_job_logs(log_dir, 12345)

    assert [log.retry for log in logs] == [0, 2]
    assert active_job_log(logs).path == log_dir / "log-general_12345_2.out"


def test_configured_sbatch_logs_fall_back_to_slurm_stdout(tmp_path):
    log_dir = tmp_path / "experiment"
    log_dir.mkdir()
    slurm_log = log_dir / "slurm-12345.out"
    slurm_log.write_text("slurm output")

    logs = find_configured_sbatch_logs(log_dir, 12345)

    assert len(logs) == 1
    assert logs[0].path == slurm_log
    assert logs[0].retry is None


def test_srt_slurm_logs_find_sweep_log(tmp_path):
    log_dir = tmp_path / "output"
    logs_dir = log_dir / "logs"
    logs_dir.mkdir(parents=True)
    sweep_log = logs_dir / "sweep_12345.log"
    sweep_log.write_text("orchestration")

    assert [log.path for log in find_srt_slurm_logs(log_dir, 12345)] == [sweep_log]


def test_srt_slurm_logs_support_mock_worker_output(tmp_path):
    log_dir = tmp_path / "output"
    log_dir.mkdir()
    mock_log = log_dir / "mock_worker.log"
    mock_log.write_text("mock")

    assert [log.path for log in find_srt_slurm_logs(log_dir, 12345)] == [mock_log]


def test_srt_slurm_logs_prefer_sweep_log_over_mock_worker(tmp_path):
    log_dir = tmp_path / "output"
    logs_dir = log_dir / "logs"
    logs_dir.mkdir(parents=True)
    sweep_log = logs_dir / "sweep_12345.log"
    sweep_log.write_text("orchestration")
    (log_dir / "mock_worker.log").write_text("mock")

    assert [log.path for log in find_srt_slurm_logs(log_dir, 12345)] == [sweep_log]


def test_read_tail_returns_requested_line_count(tmp_path):
    log_file = tmp_path / "log-general_12345_0.out"
    log_file.write_text("one\ntwo\nthree\n")

    assert read_tail(log_file, 2) == "two\nthree"


def test_job_log_path_selects_highest_retry_for_requested_job(runner, tmp_path, cluster_config_factory):
    from llmb_run.main import app

    llmb_install = tmp_path / "llmb"
    cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    active_log = log_dir / "log-general_12345_3.out"
    (log_dir / "log-general_12345_0.out").write_text("old")
    active_log.write_text("active")
    (log_dir / "log-general_77777_9.out").write_text("debug rerun")

    task = WorkloadTask("training_foo", "7b", "fp16", 8)
    workloads = {
        "training_foo": {
            "metadata": {"general": {"model": "foo"}, "run": {"launcher_type": "nemo"}},
            "workload": "foo",
        }
    }
    record_job_submission(
        cluster_config,
        task,
        SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=None),
        workloads,
    )

    with patch('llmb_run.main.get_cluster_config', return_value=cluster_config):
        result = runner.invoke(app, ["jobs", "log", "12345", "--path"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert result.stdout.strip() == str(active_log)


def test_job_log_tails_active_log(runner, tmp_path, cluster_config_factory):
    from llmb_run.main import app

    llmb_install = tmp_path / "llmb"
    cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "training_foo" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    (log_dir / "log-general_12345_0.out").write_text("one\ntwo\nthree\n")

    task = WorkloadTask("training_foo", "7b", "fp16", 8)
    workloads = {
        "training_foo": {
            "metadata": {"general": {"model": "foo"}, "run": {"launcher_type": "nemo"}},
            "workload": "foo",
        }
    }
    record_job_submission(
        cluster_config,
        task,
        SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=None),
        workloads,
    )

    with patch('llmb_run.main.get_cluster_config', return_value=cluster_config):
        result = runner.invoke(app, ["jobs", "log", "12345", "--tail", "2"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert result.stdout == "two\nthree\n"


def test_job_log_uses_configured_sbatch_slurm_stdout_fallback(runner, tmp_path, cluster_config_factory):
    from llmb_run.main import app

    llmb_install = tmp_path / "llmb"
    cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "microbenchmark_system_info" / "experiments" / "run"
    log_dir.mkdir(parents=True)
    (log_dir / "slurm-12345.out").write_text("system\ninfo\n")

    task = WorkloadTask("microbenchmark_system_info", "1b", "bf16", 8)
    workloads = {
        "microbenchmark_system_info": {
            "metadata": {"general": {"model": "system_info"}, "run": {"launcher_type": "configured_sbatch"}},
            "workload": "system_info",
        }
    }
    record_job_submission(
        cluster_config,
        task,
        SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=None),
        workloads,
    )

    with patch('llmb_run.main.get_cluster_config', return_value=cluster_config):
        result = runner.invoke(app, ["jobs", "log", "12345"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert result.stdout == "system\ninfo\n"


def test_job_log_uses_srt_slurm_sweep_log(runner, tmp_path, cluster_config_factory):
    from llmb_run.main import app

    llmb_install = tmp_path / "llmb"
    cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    log_dir = llmb_install / "workloads" / "inference_glm5_sglang_dynamo" / "outputs" / "12345"
    logs_dir = log_dir / "logs"
    logs_dir.mkdir(parents=True)
    sweep_log = logs_dir / "sweep_12345.log"
    sweep_log.write_text("server\nready\n")

    task = WorkloadTask("inference_glm5_sglang_dynamo", "744b", "fp8", 1)
    workloads = {
        "inference_glm5_sglang_dynamo": {
            "metadata": {"general": {"model": "glm5"}, "run": {"launcher_type": "srt_slurm"}},
            "workload": "glm5",
        }
    }
    record_job_submission(
        cluster_config,
        task,
        SlurmJob(job_id=12345, job_workdir=str(log_dir), llmb_config_path=None),
        workloads,
    )

    with patch("llmb_run.main.get_cluster_config", return_value=cluster_config):
        result = runner.invoke(app, ["jobs", "log", "12345"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert result.stdout == "server\nready\n"


def test_job_log_rejects_legacy_sbatch(runner, tmp_path, cluster_config_factory, caplog):
    from llmb_run.main import app

    llmb_install = tmp_path / "llmb"
    cluster_config = cluster_config_factory({"launcher": {"llmb_install": str(llmb_install)}})
    task = WorkloadTask("inference_foo", "7b", "fp16", 8)
    workloads = {
        "inference_foo": {
            "metadata": {"general": {"model": "foo"}, "run": {"launcher_type": "sbatch"}},
            "workload": "foo",
        }
    }
    record_job_submission(
        cluster_config,
        task,
        SlurmJob(job_id=12345, job_workdir=None, llmb_config_path=None),
        workloads,
    )

    with patch('llmb_run.main.get_cluster_config', return_value=cluster_config):
        result = runner.invoke(app, ["jobs", "log", "12345"])

    assert result.exit_code != 0
    assert "legacy sbatch logging" in caplog.text


def _kubernetes_job(tmp_path, cluster_config_factory, *, state, with_pod_logs=False):
    """A recorded nvcre job on a Kubernetes install, optionally with a local log snapshot."""
    import sqlite3

    from llmb_run.job_history import JobRecord, get_history_db_path, upsert_static_job

    llmb_install = tmp_path / "llmb"
    config = cluster_config_factory(
        {
            "launcher": {"llmb_install": str(llmb_install)},
            "execution_target": "kubernetes",
            "kubernetes": {"namespace": "ns"},
        }
    )
    run_dir = llmb_install / "workloads" / "w" / "experiments" / "run"
    run_dir.mkdir(parents=True)
    upsert_static_job(
        config,
        JobRecord(
            job_id=1790000000001,
            launcher_type="megatron_bridge",
            namespace="ns",
            workloadrun_name="wr",
            log_dir=str(run_dir),
        ),
    )
    with sqlite3.connect(get_history_db_path(llmb_install)) as conn:
        conn.execute("UPDATE jobs SET slurm_state = ?", (state,))
        conn.commit()

    if with_pod_logs:
        pod_logs = run_dir / "pod_logs"
        pod_logs.mkdir()
        (pod_logs / "wr-workload-node-0-0-xyz.log").write_text("step 1\nstep 2\nstep 3\n")
        (pod_logs / "wr-workload-node-0-1-xyz.log").write_text("other pod\n")

    return config, run_dir


def test_job_log_for_running_nvcre_job_streams_the_rank0_pod_from_the_cluster(runner, tmp_path, cluster_config_factory):
    import subprocess

    from llmb_run import main

    config, _ = _kubernetes_job(tmp_path, cluster_config_factory, state="RUNNING")
    calls = []

    def fake_run(argv, **kwargs):
        calls.append(argv)
        if argv[:3] == ["kubectl", "get", "pods"]:
            return subprocess.CompletedProcess(argv, 0, stdout="wr-rank0-pod\n", stderr="")
        if argv[:2] == ["kubectl", "logs"]:
            return subprocess.CompletedProcess(argv, 0, stdout="step 1\nstep 2\n", stderr="")
        return subprocess.CompletedProcess(argv, 1, stdout="", stderr="unexpected")

    with (
        patch('llmb_run.main.get_cluster_config', return_value=config),
        patch('llmb_run.main.refresh_requested_jobs', return_value=(0, None)) as refresh,
        patch("subprocess.run", fake_run),
    ):
        result = runner.invoke(main.app, ["jobs", "log", "1790000000001", "--tail", "50"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    # A status line prints immediately, then the fetched log content follows.
    assert result.stdout == (
        "Job 1790000000001 is still running; fetching live logs from the cluster...\nstep 1\nstep 2\n"
    )
    refresh.assert_called_once()  # the stale-looking row was refreshed first
    logs_call = next(c for c in calls if c[:2] == ["kubectl", "logs"])
    assert logs_call[2] == "wr-rank0-pod"  # the rank-0 pod, never another job's
    assert logs_call[logs_call.index("--tail") + 1] == "50"


def test_job_log_for_finished_nvcre_job_reads_the_local_snapshot(runner, tmp_path, cluster_config_factory):
    from llmb_run import main

    config, run_dir = _kubernetes_job(tmp_path, cluster_config_factory, state="COMPLETED", with_pod_logs=True)

    with (
        patch('llmb_run.main.get_cluster_config', return_value=config),
        patch("subprocess.run") as run,
    ):
        result = runner.invoke(main.app, ["jobs", "log", "1790000000001", "--tail", "2"])

    assert result.exit_code == 0, f"Exit code: {result.exit_code}, Output: {result.stdout}"
    assert result.stdout == "step 2\nstep 3\n"  # the rank-0 pod log, not the other pod's
    run.assert_not_called()  # terminal jobs never hit the cluster

    with patch('llmb_run.main.get_cluster_config', return_value=config):
        result = runner.invoke(main.app, ["jobs", "log", "1790000000001", "--path"])

    assert result.stdout.strip() == str(run_dir / "pod_logs" / "wr-workload-node-0-0-xyz.log")
