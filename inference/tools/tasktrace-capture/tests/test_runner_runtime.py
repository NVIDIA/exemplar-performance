# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the --runtime wiring in runner.run_experiment.

Verifies that the harbor subprocess invocation and environment setup
correctly diverge for enroot vs docker runtimes.
"""

import inspect
import json
import os
from pathlib import Path
from unittest import mock

import pytest
from tasktrace_capture import runner


@pytest.fixture
def mock_harbor_run(tmp_path):
    """Stub out subprocess + threads so run_experiment returns without real work.

    Yields a dict with the captured harbor `cmd` list and `env` dict so
    tests can assert on how tasktrace-capture invoked harbor.
    """
    captured = {}

    class FakeProc:
        pid = 12345
        returncode = 0

        def communicate(self, timeout=None):
            return ("", "")

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = list(cmd)
        captured["env"] = dict(kwargs.get("env", {}))
        return FakeProc()

    def fake_thread(target=None, daemon=None):
        t = mock.Mock()
        t.start = mock.Mock()
        t.join = mock.Mock()
        return t

    def fake_getpgid(pid):
        return pid

    with mock.patch("tasktrace_capture.runner.subprocess.Popen", side_effect=fake_popen), mock.patch(
        "tasktrace_capture.runner.threading.Thread", side_effect=fake_thread
    ), mock.patch("tasktrace_capture.runner.os.getpgid", side_effect=fake_getpgid), mock.patch(
        "tasktrace_capture.runner._cleanup_harbor_containers"
    ) as cleanup, mock.patch(
        "tasktrace_capture.runner.find_latest_harbor_job", return_value=None
    ), mock.patch(
        "tasktrace_capture.runner.snapshot_histograms", return_value={}
    ), mock.patch(
        "tasktrace_capture.runner.sample_inference", return_value=[]
    ), mock.patch(
        "tasktrace_capture.runner.metrics_url_from_api_base", return_value=None
    ), mock.patch(
        "tasktrace_capture.runner.time.sleep"
    ):
        captured["_cleanup_mock"] = cleanup
        yield captured


def _call_run_experiment(
    jobs_dir: Path,
    runtime: str | None = None,
    n_concurrent: int = 1,
    instrument_shell_commands: bool = False,
    instrument_shell_perfetto_depth: str = "1",
    source_image_bashrc: bool = False,
    trajectory_timings: str = "keep",
    agent: str = "terminus-2",
    profile_live_cpu: bool = False,
    profile_windows: list[str] | None = None,
    profile_scheduler: str = "off",
    profile_cpuset_cpus: str | None = None,
    override_cpus: int | None = None,
    swebench_image_prefix: str | None = None,
    swebench_sif_template: str | None = None,
):
    kwargs = {
        "dataset": "hello-world",
        "model": "test-model",
        "api_base": "http://example:8000/v1",
        "model_info": {},
        "n_concurrent": n_concurrent,
        "trials": 1,
        "jobs_dir": jobs_dir,
        "agent": agent,
        "instrument_shell_commands": instrument_shell_commands,
        "instrument_shell_perfetto_depth": instrument_shell_perfetto_depth,
        "source_image_bashrc": source_image_bashrc,
        "trajectory_timings": trajectory_timings,
        "profile_live_cpu": profile_live_cpu,
        "profile_windows": profile_windows,
        "profile_scheduler": profile_scheduler,
        "profile_cpuset_cpus": profile_cpuset_cpus,
        "override_cpus": override_cpus,
        "swebench_image_prefix": swebench_image_prefix,
        "swebench_sif_template": swebench_sif_template,
    }
    if runtime is not None:
        kwargs["runtime"] = runtime
    return runner.run_experiment(**kwargs)


# -- signature ----------------------------------------------------------------


def test_run_experiment_accepts_runtime_kwarg():
    sig = inspect.signature(runner.run_experiment)
    assert "runtime" in sig.parameters
    # Default must stay enroot for backcompat.
    assert sig.parameters["runtime"].default == "enroot"


# -- harbor --environment-import-path routing --------------------------------


def test_enroot_uses_enroot_plugin_path(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="enroot")
    cmd = mock_harbor_run["cmd"]
    assert "--environment-import-path" in cmd
    idx = cmd.index("--environment-import-path")
    assert cmd[idx + 1] == "tasktrace_capture.enroot_env:EnrootEnvironment"


def test_docker_uses_docker_plugin_path(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="docker")
    cmd = mock_harbor_run["cmd"]
    assert "--environment-import-path" in cmd
    idx = cmd.index("--environment-import-path")
    assert cmd[idx + 1] == "tasktrace_capture.docker_env:TasktraceCaptureDockerEnv"


def test_apptainer_uses_apptainer_plugin_path(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="apptainer")
    cmd = mock_harbor_run["cmd"]
    idx = cmd.index("--environment-import-path")
    assert cmd[idx + 1] == "tasktrace_capture.apptainer_env:ApptainerEnvironment"


def test_current_uses_current_plugin_path(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="current")
    cmd = mock_harbor_run["cmd"]
    idx = cmd.index("--environment-import-path")
    assert cmd[idx + 1] == "tasktrace_capture.current_env:CurrentEnvironment"
    assert "HARBOR_ENROOT_CACHE" not in mock_harbor_run["env"]
    assert "_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY" not in mock_harbor_run["env"]


def test_current_requires_a_single_concurrent_trial(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="one at a time"):
        _call_run_experiment(tmp_path, runtime="current", n_concurrent=2)
    assert "cmd" not in mock_harbor_run  # rejected before harbor is invoked


def test_current_without_trial_timestamps_reports_an_error_instead_of_setup_cpu(mock_harbor_run, tmp_path):
    samples = [{"t": 0.5, "cpu_total": 90.0, "cores_active": 8, "n_procs": 5, "n_containers": 1}]

    class SyncThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            pass

    with mock.patch("tasktrace_capture.runner.threading.Thread", SyncThread), mock.patch(
        "tasktrace_capture.runner.sample_cpu", return_value=samples
    ):
        result, _ = _call_run_experiment(tmp_path, runtime="current")
    assert "trial window" in result["error"]


def test_current_whose_agent_ran_for_nearly_the_whole_run_reports_cpu(mock_harbor_run, tmp_path):
    # The agent's window covers over 90% of the samples, so the trial-time filter keeps them all.
    trial = tmp_path / "2026-01-01__00-00-00" / "task__trial"
    trial.mkdir(parents=True)
    (trial / "result.json").write_text(
        json.dumps({"agent_execution": {"started_at": "2026-01-01T00:00:00Z", "finished_at": "2026-01-01T00:00:10Z"}})
    )
    samples = [
        {"t": i * 0.5, "cpu_total": 40.0, "cores_active": 2, "n_procs": 3, "n_containers": 1, "n_cores": 8}
        for i in range(22)
    ]

    class SyncThread:
        def __init__(self, target=None, daemon=None):
            self.target = target

        def start(self):
            self.target()

        def join(self, timeout=None):
            pass

    with mock.patch("tasktrace_capture.runner.threading.Thread", SyncThread), mock.patch(
        "tasktrace_capture.runner.sample_cpu", return_value=samples
    ):
        result, _ = _call_run_experiment(tmp_path, runtime="current")
    assert "error" not in result
    assert result["avg_cpu"] == 40.0


def test_default_runtime_is_enroot(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path)
    cmd = mock_harbor_run["cmd"]
    idx = cmd.index("--environment-import-path")
    assert cmd[idx + 1] == "tasktrace_capture.enroot_env:EnrootEnvironment"


def test_enroot_best_effort_network_policy_reaches_plugin(mock_harbor_run, tmp_path):
    runner.run_experiment(
        task="terminal-bench/filter-js-from-html",
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=1,
        trials=1,
        jobs_dir=tmp_path,
        runtime="enroot",
        enroot_network_policy="best-effort",
    )
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_ENROOT_NETWORK_POLICY"] == "best-effort"


def test_best_effort_network_policy_rejects_non_enroot_runtime(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="best-effort requires --runtime enroot"):
        runner.run_experiment(
            task="terminal-bench/filter-js-from-html",
            model="test-model",
            api_base="http://example:8000/v1",
            model_info={},
            n_concurrent=1,
            trials=1,
            jobs_dir=tmp_path,
            runtime="docker",
            enroot_network_policy="best-effort",
        )


def test_harbor_subprocess_prefers_active_python_bin(
    mock_harbor_run,
    tmp_path,
):
    _call_run_experiment(tmp_path)
    path = mock_harbor_run["env"]["PATH"].split(os.pathsep)
    assert path[0] == str(Path(runner.sys.executable).parent)


def test_dataset_parallelism_preserves_attempts_and_forwards_limit(mock_harbor_run, tmp_path):
    runner.run_experiment(
        dataset="compilebench",
        n_tasks=4,
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=4,
        trials=1,
        jobs_dir=tmp_path,
        runtime="docker",
    )
    cmd = mock_harbor_run["cmd"]
    assert cmd[cmd.index("-l") + 1] == "4"
    assert cmd[cmd.index("-k") + 1] == "1"
    assert cmd[cmd.index("--n-concurrent") + 1] == "4"


def test_legacy_registry_task_uses_canonical_dataset_filter(mock_harbor_run, tmp_path):
    runner.run_experiment(
        task="terminal-bench/filter-js-from-html",
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=4,
        jobs_dir=tmp_path,
        runtime="docker",
    )
    cmd = mock_harbor_run["cmd"]
    assert cmd[cmd.index("-d") + 1] == "terminal-bench"
    assert cmd[cmd.index("-i") + 1] == "filter-js-from-html"
    assert "-p" not in cmd
    assert cmd[cmd.index("-k") + 1] == "4"
    assert cmd[cmd.index("--n-concurrent") + 1] == "4"


def test_local_task_path_stays_on_direct_task_path(mock_harbor_run, tmp_path):
    task_path = tmp_path / "local-task"
    task_path.mkdir()
    runner.run_experiment(
        task=str(task_path),
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=1,
        jobs_dir=tmp_path / "jobs",
        runtime="docker",
    )
    cmd = mock_harbor_run["cmd"]
    assert cmd[cmd.index("-p") + 1] == str(task_path)
    assert "-i" not in cmd


def test_suite_parallelism_does_not_multiply_attempts(mock_harbor_run, tmp_path):
    runner.run_experiment(
        include_tasks={"dataset": "compilebench", "tasks": ["jq", "cowsay"]},
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=2,
        trials=1,
        jobs_dir=tmp_path,
        runtime="docker",
    )
    cmd = mock_harbor_run["cmd"]
    assert cmd[cmd.index("-k") + 1] == "1"
    assert cmd[cmd.index("--n-concurrent") + 1] == "2"


def test_package_suite_preserves_dataset_and_task_names(mock_harbor_run, tmp_path):
    runner.run_experiment(
        include_tasks={
            "dataset": "datacurve/deep-swe",
            "tasks": ["datacurve/task-a", "other-org/task-b"],
        },
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=2,
        trials=1,
        jobs_dir=tmp_path,
        runtime="docker",
    )
    cmd = mock_harbor_run["cmd"]
    assert cmd[cmd.index("-d") + 1] == "datacurve/deep-swe"
    assert [cmd[index + 1] for index, value in enumerate(cmd) if value == "-i"] == [
        "datacurve/task-a",
        "other-org/task-b",
    ]


# -- HARBOR_ENROOT_CACHE gating ----------------------------------------------


def test_harbor_enroot_cache_set_for_enroot(mock_harbor_run, tmp_path, monkeypatch):
    monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
    _call_run_experiment(tmp_path, runtime="enroot")
    env = mock_harbor_run["env"]
    assert "HARBOR_ENROOT_CACHE" in env
    assert ".enroot" in env["HARBOR_ENROOT_CACHE"]


def test_harbor_enroot_cache_not_set_for_docker(mock_harbor_run, tmp_path, monkeypatch):
    monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
    _call_run_experiment(tmp_path, runtime="docker")
    env = mock_harbor_run["env"]
    assert "HARBOR_ENROOT_CACHE" not in env


def test_harbor_enroot_cache_not_set_for_apptainer(mock_harbor_run, tmp_path, monkeypatch):
    monkeypatch.delenv("HARBOR_ENROOT_CACHE", raising=False)
    _call_run_experiment(tmp_path, runtime="apptainer")
    assert "HARBOR_ENROOT_CACHE" not in mock_harbor_run["env"]


# -- cleanup gating -----------------------------------------------------------


def test_enroot_cleanup_called_for_enroot(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="enroot")
    mock_harbor_run["_cleanup_mock"].assert_called_once()


def test_enroot_cleanup_skipped_for_docker(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="docker")
    mock_harbor_run["_cleanup_mock"].assert_not_called()


# -- barrier + agent env vars (both runtimes) -------------------------------


def test_barrier_env_var_set_for_both_runtimes(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="enroot")
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_BARRIER_COUNT"] == "1"

    _call_run_experiment(tmp_path, runtime="docker")
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_BARRIER_COUNT"] == "1"


def test_barrier_can_be_disabled(mock_harbor_run, tmp_path):
    runner.run_experiment(
        dataset="compilebench",
        model="test-model",
        api_base="http://example:8000/v1",
        model_info={},
        n_concurrent=4,
        trials=1,
        jobs_dir=tmp_path,
        runtime="docker",
        sync_start=False,
    )
    assert "_TASKTRACE_CAPTURE_BARRIER_COUNT" not in mock_harbor_run["env"]


def test_agent_hint_env_var_set_for_both_runtimes(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="enroot")
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_AGENT"] == "terminus-2"

    _call_run_experiment(tmp_path, runtime="docker")
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_AGENT"] == "terminus-2"


def test_swebench_sif_template_forwarded_for_apptainer(mock_harbor_run, tmp_path):
    _call_run_experiment(
        tmp_path,
        runtime="apptainer",
        swebench_sif_template="/sifs/{instance_id}.sif",
    )
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_SWEBENCH_SIF_TEMPLATE"] == ("/sifs/{instance_id}.sif")


def test_shell_capture_flag_is_runtime_neutral(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="enroot", instrument_shell_commands=True)
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS"] == "1"

    _call_run_experiment(tmp_path, runtime="docker", instrument_shell_commands=True)
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_TRACE_SHELL_COMMANDS"] == "1"


def test_shell_capture_rejects_unhandled_agent_before_launch(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="currently supports"):
        _call_run_experiment(tmp_path, instrument_shell_commands=True, agent="codex")


def test_shell_capture_finalizes_only_new_job(mock_harbor_run, tmp_path):
    job = tmp_path / "new-job"
    job.mkdir()
    with mock.patch(
        "tasktrace_capture.runner.find_latest_harbor_job",
        side_effect=[None, job],
    ), mock.patch(
        "tasktrace_capture.runner.finalize_trajectories",
        return_value=[],
    ) as finalize:
        _call_run_experiment(tmp_path, instrument_shell_commands=True)
    finalize.assert_called_once_with(job, perfetto_depth="1", bash_env=None, trajectory_timings="keep")


def test_shell_capture_passes_perfetto_depth_to_finalizer(mock_harbor_run, tmp_path):
    job = tmp_path / "new-job"
    job.mkdir()
    with mock.patch(
        "tasktrace_capture.runner.find_latest_harbor_job",
        side_effect=[None, job],
    ), mock.patch(
        "tasktrace_capture.runner.finalize_trajectories",
        return_value=[],
    ) as finalize:
        _call_run_experiment(tmp_path, instrument_shell_commands=True, instrument_shell_perfetto_depth="max")
    finalize.assert_called_once_with(job, perfetto_depth="max", bash_env=None, trajectory_timings="keep")


def test_source_image_bashrc_reaches_the_plugin_and_the_trace_header(mock_harbor_run, tmp_path):
    job = tmp_path / "new-job"
    job.mkdir()
    with mock.patch(
        "tasktrace_capture.runner.find_latest_harbor_job",
        side_effect=[None, job],
    ), mock.patch(
        "tasktrace_capture.runner.finalize_trajectories",
        return_value=[],
    ) as finalize:
        _call_run_experiment(tmp_path, instrument_shell_commands=True, source_image_bashrc=True)
    finalize.assert_called_once_with(job, perfetto_depth="1", bash_env="/root/.bashrc", trajectory_timings="keep")
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_BASH_ENV"] == "/root/.bashrc"


def test_trajectory_timings_reaches_the_finalizer(mock_harbor_run, tmp_path):
    job = tmp_path / "new-job"
    job.mkdir()
    with mock.patch(
        "tasktrace_capture.runner.find_latest_harbor_job",
        side_effect=[None, job],
    ), mock.patch(
        "tasktrace_capture.runner.finalize_trajectories",
        return_value=[],
    ) as finalize:
        _call_run_experiment(tmp_path, instrument_shell_commands=True, trajectory_timings="omit")
    finalize.assert_called_once_with(job, perfetto_depth="1", bash_env=None, trajectory_timings="omit")
    with pytest.raises(ValueError, match="--trajectory-timings must be one of: keep, omit"):
        _call_run_experiment(tmp_path, instrument_shell_commands=True, trajectory_timings="sometimes")


def test_live_cpu_profile_wires_docker_plugin_environment(mock_harbor_run, tmp_path):
    _call_run_experiment(
        tmp_path,
        runtime="docker",
        profile_live_cpu=True,
        profile_windows=["agent_execution", "container_lifecycle_to_verified"],
        profile_scheduler="required",
        profile_cpuset_cpus="8-15",
    )
    env = mock_harbor_run["env"]
    assert env["_TASKTRACE_CAPTURE_PROFILE_LIVE_CPU"] == "1"
    assert env["_TASKTRACE_CAPTURE_PROFILE_WINDOWS"] == ("agent_execution,container_lifecycle_to_verified")
    assert env["_TASKTRACE_CAPTURE_PROFILE_SCHEDULER"] == "required"
    assert env["_TASKTRACE_CAPTURE_PROFILE_CPUSET_CPUS"] == "8-15"


def test_live_cpu_profile_rejects_enroot(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="runtime docker"):
        _call_run_experiment(tmp_path, runtime="enroot", profile_live_cpu=True)


def test_live_cpu_profile_rejects_invalid_cpuset(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="invalid CPU set range"):
        _call_run_experiment(
            tmp_path,
            runtime="docker",
            profile_live_cpu=True,
            profile_cpuset_cpus="15-8",
        )


def test_live_scheduler_profile_requires_named_cpuset(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="needs --profile-cpuset-cpus"):
        _call_run_experiment(
            tmp_path,
            runtime="docker",
            profile_live_cpu=True,
            profile_scheduler="required",
        )


def test_live_scheduler_profile_accepts_explicit_cpuset_and_width(mock_harbor_run, tmp_path):
    _call_run_experiment(
        tmp_path,
        runtime="docker",
        profile_live_cpu=True,
        profile_scheduler="required",
        override_cpus=8,
        profile_cpuset_cpus="8-15",
    )
    assert mock_harbor_run["env"]["_TASKTRACE_CAPTURE_PROFILE_CPUSET_CPUS"] == "8-15"


def test_live_scheduler_profile_rejects_parallel_trials(mock_harbor_run, tmp_path):
    with pytest.raises(ValueError, match="one live trial at a time"):
        _call_run_experiment(
            tmp_path,
            runtime="docker",
            n_concurrent=2,
            profile_live_cpu=True,
            profile_scheduler="required",
            profile_cpuset_cpus="8-15",
        )


def test_override_cpus_is_forwarded_to_harbor(mock_harbor_run, tmp_path):
    _call_run_experiment(tmp_path, runtime="docker", override_cpus=8)
    cmd = mock_harbor_run["cmd"]
    index = cmd.index("--override-cpus")
    assert cmd[index + 1] == "8"
