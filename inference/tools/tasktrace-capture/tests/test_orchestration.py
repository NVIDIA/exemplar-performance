# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Orchestration tests extracted from test_bench.py.

Covers: TestOrchestrationRepeat, TestOrchestrationTrials,
        TestOrchestrationAgentTimeout, TestOrchestrationAgentKwargs,
        TestOrchestrationAgent, TestOrchestrationSuiteMode,
        TestOrchestrationOutputJSON, TestOrchestrationOutputDir.

Run:  pytest tests/test_orchestration.py -v
"""

import json
import os

import pytest

# ===========================================================================
#  Helpers
# ===========================================================================


def _fake_experiment_result(n_concurrent):
    """Minimal result dict matching what run_experiment returns."""
    return {
        "n_concurrent": n_concurrent,
        "n_cores": 64,
        "duration_sec": 100.0,
        "n_samples": 10,
        "n_active_samples": 8,
        "avg_cpu": 2.0 * n_concurrent,
        "peak_cpu": 5.0 * n_concurrent,
        "avg_cores_active": 3.0 * n_concurrent,
        "peak_cores_active": 8 * n_concurrent,
        "p95_cores_active": 7 * n_concurrent,
        "avg_procs": 2.0 * n_concurrent,
        "peak_procs": 4 * n_concurrent,
        "pct_cpu_active": 2.0 * n_concurrent,
        "pct_waiting_llm": 98.0,
        "avg_proc_categories": {"python": float(n_concurrent)},
        "avg_disk_read_mb_s": 0.0,
        "avg_disk_write_mb_s": 10.0,
        "peak_disk_read_mb_s": 1.0,
        "peak_disk_write_mb_s": 50.0,
        "samples": [
            {
                "t": 0.5,
                "cpu_total": 1.0,
                "cores_active": 2,
                "n_cores": 64,
                "n_containers": n_concurrent,
                "n_procs": n_concurrent,
                "proc_cats": {},
                "disk_read_mb_s": 0,
                "disk_write_mb_s": 0,
            },
        ],
    }


def _fake_time_split(n_concurrent, rep=1):
    """Minimal TimeSplit object matching parse_agent_time_split output."""
    from tasktrace_capture.models import TimeSplit

    tool = 100.0 + rep * 10 + n_concurrent * 5
    wall = 900.0
    llm = wall - tool
    return TimeSplit(
        pct_cpu=round(100 * tool / wall, 1),
        pct_llm=round(100 * llm / wall, 1),
        tool_time_sec=tool,
        wall_time_sec=wall,
        llm_time_sec=llm,
        n_llm_calls=20,
        n_full=n_concurrent,
        n_partial=0,
        n_invalid=0,
        survival_rate=f"{n_concurrent}/{n_concurrent}",
        trials=[],
    )


def _run_main_with_mocks(tmp_path, cli_args, mock_splits=True):
    """Run cli.main() with mocked runner functions.

    Returns (calls_log, output_json, out_dir) where calls_log is a list of
    dicts from each run_experiment call, and output_json is the final scaling_results.
    """
    from unittest.mock import patch

    from tasktrace_capture import cli as benchmark_sweep

    calls_log = []
    job_counter = [0]
    out_dir = tmp_path / "sweep_out"

    def mock_run_experiment(**kwargs):
        n = kwargs["n_concurrent"]
        calls_log.append(
            {
                "n_concurrent": n,
                "trials": kwargs.get("trials"),
                "agent_timeout": kwargs.get("agent_timeout"),
                "dataset": kwargs.get("dataset"),
                "n_tasks": kwargs.get("n_tasks"),
                "task": kwargs.get("task"),
                "include_tasks": kwargs.get("include_tasks"),
                "extra_agent_kwargs": kwargs.get("extra_agent_kwargs"),
                "agent": kwargs.get("agent"),
                "sync_start": kwargs.get("sync_start"),
            }
        )
        return _fake_experiment_result(n), "fake stdout"

    def mock_find_latest(jobs_dir):
        job_counter[0] += 1
        job_dir = tmp_path / f"fake_job_{job_counter[0]}"
        job_dir.mkdir(exist_ok=True)
        return job_dir

    split_counter = [0]

    def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
        split_counter[0] += 1
        n = max(1, split_counter[0])
        return _fake_time_split(n, rep=split_counter[0])

    def mock_generate_plots(data, od):
        pass

    def mock_generate_timing_charts(d):
        return False

    def mock_sample_baseline(duration_sec=5):
        return {
            "duration_sec": duration_sec,
            "avg_cpu_pct": 0.5,
            "avg_cores_active": 2.0,
            "peak_cores_active": 5,
            "n_cores": 64,
        }

    # Default mock: return a broad set of task names so suite mode doesn't
    # emit spurious "missing tasks" warnings in unrelated tests.
    _common_tasks = {
        "qemu-startup",
        "sqlite-with-gcov",
        "headless-terminal",
        "task-a",
        "task-b",
        "task-c",
    }

    def mock_get_found_suite_tasks(job_dir):
        return _common_tasks

    patches = [
        patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment),
        patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
        patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
        patch.object(benchmark_sweep, "get_found_suite_tasks", side_effect=mock_get_found_suite_tasks),
        patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
        patch.object(benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}),
        patch("tasktrace_capture.monitor.sample_baseline", mock_sample_baseline),
        patch("tasktrace_capture.plot_scaling.generate_plots", mock_generate_plots),
        patch("tasktrace_capture.plot_timing.generate_timing_charts", mock_generate_timing_charts),
        patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
        # keep orchestration tests hermetic: no real sampling thread,
        # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
        patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
        patch("os._exit", side_effect=SystemExit(0)),
    ]

    full_args = cli_args + [
        "--output-dir",
        str(out_dir),
        "--api-base",
        "http://fake:8000/v1",
        "--skip-preflight",
    ]
    with patch("sys.argv", ["tasktrace-capture"] + full_args):
        for p in patches:
            p.start()
        try:
            benchmark_sweep.main()
        except SystemExit as e:
            if e.code != 0:
                raise
        finally:
            for p in patches:
                p.stop()

    json_path = out_dir / "scaling_results.json"
    output_json = None
    if json_path.exists():
        with open(json_path) as f:
            output_json = json.load(f)

    return calls_log, output_json, out_dir


# ===========================================================================
#  TestOrchestrationRepeat
# ===========================================================================


class TestOrchestrationRepeat:
    """Verify --repeat controls the number of run_experiment calls per concurrency level."""

    def test_repeat_1_single_concurrency(self, tmp_path):
        calls, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
            ],
        )
        assert len(calls) == 1
        assert calls[0]["n_concurrent"] == 4

    def test_repeat_3_single_concurrency(self, tmp_path):
        calls, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
                "--repeat",
                "3",
            ],
        )
        assert len(calls) == 3
        assert all(c["n_concurrent"] == 4 for c in calls)

    def test_repeat_2_multi_concurrency(self, tmp_path):
        calls, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,2,4",
                "--repeat",
                "2",
            ],
        )
        assert len(calls) == 6  # 3 levels x 2 repeats
        expected = [1, 1, 2, 2, 4, 4]
        assert [c["n_concurrent"] for c in calls] == expected

    def test_repeat_1_is_default(self, tmp_path):
        calls, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,2,4",
            ],
        )
        assert len(calls) == 3
        assert [c["n_concurrent"] for c in calls] == [1, 2, 4]


class TestOrchestrationTrials:
    """Verify --trials is passed through correctly and bumped when < concurrency."""

    def test_trials_none_by_default(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
            ],
        )
        assert calls[0]["trials"] is None

    def test_explicit_trials(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
                "-k",
                "8",
            ],
        )
        assert calls[0]["trials"] == 8

    def test_trials_passed_unchanged_even_when_less_than_concurrent(self, tmp_path):
        """benchmark_sweep passes raw trials; _compute_trials bumps it inside runner."""
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
                "-k",
                "2",
            ],
        )
        assert calls[0]["trials"] == 2
        assert calls[0]["n_concurrent"] == 4

    def test_trials_with_repeat(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "2",
                "-k",
                "6",
                "--repeat",
                "3",
            ],
        )
        assert len(calls) == 3
        assert all(c["trials"] == 6 for c in calls)
        assert all(c["n_concurrent"] == 2 for c in calls)


class TestOrchestrationAgentTimeout:
    """Verify --agent-timeout is forwarded to every run_experiment call."""

    def test_default_timeout(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
            ],
        )
        assert calls[0]["agent_timeout"] == 3600

    def test_custom_timeout(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,4",
                "--agent-timeout",
                "900",
            ],
        )
        assert all(c["agent_timeout"] == 900 for c in calls)

    def test_timeout_with_repeat(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "2",
                "--agent-timeout",
                "600",
                "--repeat",
                "3",
            ],
        )
        assert len(calls) == 3
        assert all(c["agent_timeout"] == 600 for c in calls)


class TestOrchestrationAgentKwargs:
    """Verify --agent-kwarg values are forwarded."""

    def test_single_kwarg(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
                "--agent-kwarg",
                "temperature=0.0",
            ],
        )
        assert calls[0]["extra_agent_kwargs"] == ["temperature=0.0"]

    def test_multiple_kwargs(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
                "--agent-kwarg",
                "temperature=0.0",
                "--agent-kwarg",
                "top_p=0.9",
            ],
        )
        assert calls[0]["extra_agent_kwargs"] == ["temperature=0.0", "top_p=0.9"]


class TestOrchestrationAgent:
    """Verify --agent flag is forwarded correctly."""

    def test_default_agent(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
            ],
        )
        assert calls[0]["agent"] == "terminus-2"

    def test_custom_agent(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
                "--agent",
                "claude-code",
                "--agent-env",
                "ANTHROPIC_BASE_URL=https://api.example.com",
                "--agent-env",
                "ANTHROPIC_API_KEY=test-key",
            ],
        )
        assert calls[0]["agent"] == "claude-code"


class TestOrchestrationAgentEnv:
    """Verify --agent-env vars are injected into subprocess env, not as CLI flags.

    Regression: Harbor's --agent-env flag broke on Python 3.13 (typer annotation bug).
    We now set them as process env vars instead of CLI flags.
    """

    def test_agent_env_not_in_harbor_cmd(self, tmp_path):
        """--agent-env values must NOT appear as harbor CLI flags."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        harbor_calls = []

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0
                if isinstance(cmd, list) and cmd and cmd[0] == "harbor":
                    harbor_calls.append({"cmd": cmd, "env": kwargs.get("env")})

            def communicate(self, timeout=None):
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test-model",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                extra_agent_env=["ANTHROPIC_API_KEY=sk-test123", "ANTHROPIC_BASE_URL=https://example.com"],
                jobs_dir=tmp_path,
                agent="claude-code",
            )

        assert len(harbor_calls) == 1
        cmd = harbor_calls[0]["cmd"]
        env = harbor_calls[0]["env"]
        # Env vars must NOT be in the command line
        assert "--agent-env" not in cmd
        assert "ANTHROPIC_API_KEY=sk-test123" not in cmd
        # Env vars MUST be in the subprocess environment
        assert env is not None
        assert env["ANTHROPIC_API_KEY"] == "sk-test123"
        assert env["ANTHROPIC_BASE_URL"] == "https://example.com"
        # Manifest var tells enroot_env.py exactly which vars to forward
        assert env["_TASKTRACE_CAPTURE_FORWARD_ENVS"] == "ANTHROPIC_API_KEY,ANTHROPIC_BASE_URL"

    def test_agent_env_inherits_existing_env(self, tmp_path):
        """Agent env vars are merged with existing environment, not replacing it."""
        import subprocess as real_subprocess
        from unittest.mock import patch

        from tasktrace_capture.runner import run_experiment

        harbor_envs = []

        class MockPopen:
            def __init__(self, cmd, **kwargs):
                self.pid = os.getpid()
                self.returncode = 0
                if isinstance(cmd, list) and cmd and cmd[0] == "harbor":
                    harbor_envs.append(kwargs.get("env"))

            def communicate(self, timeout=None):
                return "", ""

        with patch.object(real_subprocess, "Popen", MockPopen):
            run_experiment(
                dataset="hello-world",
                model="test-model",
                api_base="http://fake:8000/v1",
                model_info={},
                n_concurrent=1,
                extra_agent_env=["MY_VAR=hello"],
                jobs_dir=tmp_path,
                agent="claude-code",
            )

        env = harbor_envs[0]
        # Should have both the new var AND existing PATH
        assert env["MY_VAR"] == "hello"
        assert "PATH" in env
        # Manifest lists exactly the user-specified vars
        assert env["_TASKTRACE_CAPTURE_FORWARD_ENVS"] == "MY_VAR"


class TestOrchestrationSuiteMode:
    """Verify suite mode computes n_concurrent and trials correctly."""

    def test_suite_concurrent_multiplied(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup,sqlite-with-gcov,headless-terminal",
                "--concurrency",
                "2",
            ],
        )
        assert len(calls) == 1
        # 2 copies x 3 tasks = 6 concurrent
        assert calls[0]["n_concurrent"] == 6
        assert calls[0]["include_tasks"]["dataset"] == "terminal-bench"
        assert len(calls[0]["include_tasks"]["tasks"]) == 3
        assert calls[0]["trials"] == 2

    def test_suite_multi_concurrency(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/task-a,task-b",
                "--concurrency",
                "1,2,4",
            ],
        )
        assert len(calls) == 3
        assert [c["n_concurrent"] for c in calls] == [2, 4, 8]  # Nx2 tasks

    def test_scheduler_profile_suite_runs_selected_tasks_sequentially(self, tmp_path, capsys):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--runtime",
                "docker",
                "--task",
                "terminal-bench/task-a,task-b",
                "--concurrency",
                "1",
                "--override-cpus",
                "8",
                "--profile-live-cpu",
                "--profile-scheduler",
                "required",
                "--profile-cpuset-cpus",
                "8-15",
            ],
        )
        assert len(calls) == 1
        assert calls[0]["n_concurrent"] == 1
        assert calls[0]["include_tasks"]["tasks"] == ["task-a", "task-b"]
        output = capsys.readouterr().out
        assert "logical CPUs 8-15" in output
        assert "Sequential scheduler profile: 2 tasks, one at a time on CPUs 8-15" in output

    def test_scheduler_profile_rejects_parallel_plan_before_launch(self, tmp_path, capsys):
        with pytest.raises(SystemExit):
            _run_main_with_mocks(
                tmp_path,
                [
                    "--runtime",
                    "docker",
                    "--task",
                    "terminal-bench/task-a,task-b",
                    "--concurrency",
                    "2",
                    "--override-cpus",
                    "8",
                    "--profile-live-cpu",
                    "--profile-scheduler",
                    "required",
                    "--profile-cpuset-cpus",
                    "8-15",
                ],
            )
        assert "--profile-scheduler required runs sequentially" in capsys.readouterr().err

    def test_dataset_task_limit_passed_through(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--dataset",
                "compilebench",
                "--n-tasks",
                "4",
                "--concurrency",
                "4",
                "--trials",
                "1",
            ],
        )
        assert calls[0]["dataset"] == "compilebench"
        assert calls[0]["n_tasks"] == 4
        assert calls[0]["n_concurrent"] == 4
        assert calls[0]["trials"] == 1

    def test_start_barrier_can_be_disabled(self, tmp_path):
        calls, _, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--dataset",
                "compilebench",
                "--n-tasks",
                "4",
                "--concurrency",
                "4",
                "--trials",
                "1",
                "--no-start-barrier",
            ],
        )
        assert calls[0]["sync_start"] is False

    def test_suite_cpu_metrics_present_in_report(self, tmp_path):
        """Regression: suite mode must have CPU metrics at every concurrency level.

        Previously, suite mode stored results with n_concurrent=actual (e.g. 2 for 1x2 tasks)
        but time_splits keyed by multiplier (1). The results_by_n lookup failed, producing
        null CPU metrics in report.json.
        """
        from tasktrace_capture.cli import _write_report_json

        _, data, out_dir = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/task-a,task-b",
                "--concurrency",
                "1,2",
            ],
        )
        assert data is not None
        # time_splits keys are multipliers: "1", "2"
        assert "1" in data["time_splits"]
        assert "2" in data["time_splits"]
        # results should map to multiplier keys, not actual concurrent counts
        _write_report_json(data, out_dir)
        with open(out_dir / "report.json") as f:
            report = json.load(f)
        for n_key in ("1", "2"):
            cpu = report["concurrency"][n_key]["cpu"]
            assert cpu["avg_pct"] is not None, f"avg_pct is None for multiplier {n_key}"
            assert cpu["peak_cores"] is not None, f"peak_cores is None for multiplier {n_key}"


class TestOrchestrationOutputJSON:
    """Verify scaling_results.json has the right structure."""

    def test_single_run_result_is_dict(self, tmp_path):
        """With repeat=1, each entry in results is a single dict (not a list)."""
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,2",
            ],
        )
        assert data is not None
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert isinstance(r, dict)
            assert "n_concurrent" in r

    def test_repeat_result_is_list(self, tmp_path):
        """With repeat>1, each entry in results is a list of dicts."""
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,2",
                "--repeat",
                "3",
            ],
        )
        assert data is not None
        assert len(data["results"]) == 2
        for r in data["results"]:
            assert isinstance(r, list)
            assert len(r) == 3
            for sub in r:
                assert isinstance(sub, dict)
                assert "n_concurrent" in sub

    def test_json_has_n_repeats(self, tmp_path):
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
                "--repeat",
                "5",
            ],
        )
        assert data["n_repeats"] == 5

    def test_json_has_agent_timeout(self, tmp_path):
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
                "--agent-timeout",
                "900",
            ],
        )
        assert data["agent_timeout"] == 900

    def test_json_time_splits_raw_when_no_repeat(self, tmp_path):
        """With repeat=1, time_splits stores raw split dict (has pct_cpu as number)."""
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
            ],
        )
        split = data["time_splits"]["4"]
        assert isinstance(split["pct_cpu"], (int, float))

    def test_json_time_splits_aggregated_when_repeat(self, tmp_path):
        """With repeat>1, time_splits stores aggregated dict (has pct_cpu as dict with median)."""
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "4",
                "--repeat",
                "3",
            ],
        )
        split = data["time_splits"]["4"]
        assert isinstance(split["pct_cpu"], dict)
        assert "median" in split["pct_cpu"]
        assert "min" in split["pct_cpu"]
        assert "max" in split["pct_cpu"]
        assert split["n_repeats"] == 3

    def test_json_suite_mode_fields(self, tmp_path):
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/task-a,task-b",
                "--concurrency",
                "1",
            ],
        )
        assert data["suite_mode"] is True
        assert data["suite_dataset"] == "terminal-bench"
        assert data["suite_tasks"] == ["task-a", "task-b"]

    def test_json_non_suite_mode(self, tmp_path):
        _, data, _ = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1",
            ],
        )
        assert data["suite_mode"] is False
        assert data.get("suite_tasks") is None


class TestOrchestrationMissingTimeSplit:
    """Verify concurrency levels appear in report even when parse_agent_time_split returns None."""

    def test_none_split_still_in_time_splits(self, tmp_path):
        """When parse_agent_time_split returns None (e.g. harbor timed out),
        the concurrency level should still appear in time_splits as a placeholder."""
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        calls_log = []
        job_counter = [0]
        out_dir = tmp_path / "sweep_out"

        def mock_run_experiment(**kwargs):
            n = kwargs["n_concurrent"]
            calls_log.append(kwargs)
            return _fake_experiment_result(n), "fake stdout"

        def mock_find_latest(jobs_dir):
            job_counter[0] += 1
            job_dir = tmp_path / f"fake_job_{job_counter[0]}"
            job_dir.mkdir(exist_ok=True)
            return job_dir

        split_counter = [0]

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            split_counter[0] += 1
            # Return None for first call (c=1), valid split for second (c=2)
            if split_counter[0] == 1:
                return None
            return _fake_time_split(2)

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
            # keep orchestration tests hermetic: no real sampling thread,
            # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/qemu-startup",
            "--concurrency",
            "1,2",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                benchmark_sweep.main()
            except SystemExit as e:
                if e.code != 0:
                    raise
            finally:
                for p in patches:
                    p.stop()

        json_path = out_dir / "scaling_results.json"
        with open(json_path) as f:
            data = json.load(f)

        # Both concurrency levels must be in time_splits
        assert "1" in data["time_splits"], "c=1 missing from time_splits"
        assert "2" in data["time_splits"], "c=2 missing from time_splits"
        # c=1 placeholder should have survival_rate 0/0
        assert data["time_splits"]["1"]["survival_rate"] == "0/0"
        # c=2 should have real data
        assert data["time_splits"]["2"]["survival_rate"] != "0/0"

    def test_none_split_appears_in_report_txt(self, tmp_path):
        """Report.txt must include a row for concurrency levels with None split."""
        from tasktrace_capture.cli import _write_report

        data = {
            "dataset": "test",
            "model": "m",
            "api_base": "http://x",
            "timestamp": "2026-01-01",
            "hostname": "h",
            "cpu_info": {"model": "x", "cores_logical": 128},
            "agent_timeout": 3600,
            "n_repeats": 1,
            "suite_mode": False,
            "time_splits": {
                "1": {
                    "pct_cpu": None,
                    "pct_llm": None,
                    "survival_rate": "0/0",
                    "n_full": 0,
                    "n_partial": 0,
                    "n_invalid": 0,
                    "trials": [],
                },
                "2": {
                    "pct_cpu": 50.0,
                    "pct_llm": 50.0,
                    "tool_time_sec": 30.0,
                    "wall_time_sec": 60.0,
                    "llm_time_sec": 30.0,
                    "n_llm_calls": 5.0,
                    "avg_reward": 1.0,
                    "n_full": 2,
                    "n_partial": 0,
                    "n_invalid": 0,
                    "survival_rate": "2/2",
                    "trials": [],
                },
            },
            "results": [
                {
                    "n_concurrent": 1,
                    "n_cores": 128,
                    "avg_cpu": 0.5,
                    "peak_cpu": 5.0,
                    "avg_cores_active": 2,
                    "peak_cores_active": 10,
                    "samples": [],
                },
                {
                    "n_concurrent": 2,
                    "n_cores": 128,
                    "avg_cpu": 2.0,
                    "peak_cpu": 10.0,
                    "avg_cores_active": 8,
                    "peak_cores_active": 30,
                    "samples": [],
                },
            ],
            "baseline_cpu": {},
        }
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        # c=1 should appear as n/a row in Time Split section
        assert "n/a" in text
        # Both concurrency levels present in Rewards section
        assert "0/0" in text  # c=1 placeholder survival
        assert "2/2" in text  # c=2 real survival


class TestOrchestrationOutputDir:
    """Verify output directory structure."""

    def test_output_dir_has_scaling_results(self, tmp_path):
        _, _, out_dir = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,4",
            ],
        )
        assert (out_dir / "scaling_results.json").exists()

    def test_output_dir_has_runs_dir(self, tmp_path):
        _, _, out_dir = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "1,4",
            ],
        )
        assert (out_dir / "runs").is_dir()

    def test_output_dir_has_report_json_and_txt(self, tmp_path):
        _, _, out_dir = _run_main_with_mocks(
            tmp_path,
            [
                "--task",
                "terminal-bench/qemu-startup",
                "--concurrency",
                "2",
            ],
        )
        assert (out_dir / "report.json").exists()
        assert (out_dir / "report.txt").exists()


# ===========================================================================
#  Harbor failure exits non-zero, no DONE message
# ===========================================================================


class TestOrchestrationHarborFailure:
    """When harbor fails (all levels error), tasktrace-capture should exit non-zero."""

    def test_all_levels_error_exits_nonzero(self, tmp_path):
        """If run_experiment returns error dicts for ALL levels, main() should exit 1."""
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        out_dir = tmp_path / "sweep_out"

        def mock_run_experiment_fail(**kwargs):
            n = kwargs["n_concurrent"]
            return {
                "n_concurrent": n,
                "error": "harbor failed (rc=1)",
                "harbor_stderr": "some error",
            }, "fake stdout"

        def mock_find_latest(jobs_dir):
            return None

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            return None

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment_fail),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
            # keep orchestration tests hermetic: no real sampling thread,
            # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/qemu-startup",
            "--concurrency",
            "1,2",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                with pytest.raises(SystemExit) as exc_info:
                    benchmark_sweep.main()
                assert exc_info.value.code == 1
            finally:
                for p in patches:
                    p.stop()


# ===========================================================================
#  Container cleanup between concurrency levels
# ===========================================================================


class TestOrchestrationContainerCleanup:
    """Verify _cleanup_containers is called between concurrency levels."""

    def test_cleanup_called_between_levels(self, tmp_path):
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        cleanup_calls = []

        def mock_cleanup():
            cleanup_calls.append(1)

        out_dir = tmp_path / "sweep_out"
        job_counter = [0]

        def mock_run_experiment(**kwargs):
            n = kwargs["n_concurrent"]
            return _fake_experiment_result(n), "fake stdout"

        def mock_find_latest(jobs_dir):
            job_counter[0] += 1
            job_dir = tmp_path / f"fake_job_{job_counter[0]}"
            job_dir.mkdir(exist_ok=True)
            return job_dir

        split_counter = [0]

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            split_counter[0] += 1
            return _fake_time_split(max(1, split_counter[0]))

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", side_effect=mock_cleanup),
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/qemu-startup",
            "--concurrency",
            "1,2,4",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                benchmark_sweep.main()
            except SystemExit as e:
                if e.code != 0:
                    raise
            finally:
                for p in patches:
                    p.stop()

        # _cleanup_containers should be called once per concurrency level
        assert len(cleanup_calls) == 3, f"Expected 3 cleanup calls (one per level), got {len(cleanup_calls)}"


# ===========================================================================
#  Suite mode warns when requested tasks are missing from results
# ===========================================================================


class TestOrchestrationSuiteMissingTasks:
    """Verify suite mode emits a warning when harbor silently drops tasks."""

    def test_no_warning_when_all_tasks_present(self, tmp_path, capsys):
        """When all requested suite tasks are found, no warning is printed."""
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        out_dir = tmp_path / "sweep_out"
        job_counter = [0]

        def mock_run_experiment(**kwargs):
            n = kwargs["n_concurrent"]
            return _fake_experiment_result(n), "fake stdout"

        def mock_find_latest(jobs_dir):
            job_counter[0] += 1
            job_dir = tmp_path / f"fake_job_{job_counter[0]}"
            job_dir.mkdir(exist_ok=True)
            return job_dir

        split_counter = [0]

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            split_counter[0] += 1
            return _fake_time_split(max(1, split_counter[0]))

        def mock_get_found_suite_tasks(job_dir):
            # Return all requested tasks
            return {"task-a", "task-b"}

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_found_suite_tasks", side_effect=mock_get_found_suite_tasks),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
            # keep orchestration tests hermetic: no real sampling thread,
            # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/task-a,task-b",
            "--concurrency",
            "1",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                benchmark_sweep.main()
            except SystemExit as e:
                if e.code != 0:
                    raise
            finally:
                for p in patches:
                    p.stop()

        captured = capsys.readouterr()
        assert "missing from results" not in captured.out

    def test_warning_when_tasks_missing(self, tmp_path, capsys):
        """When requested suite tasks are not found in results, a warning is printed."""
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        out_dir = tmp_path / "sweep_out"
        job_counter = [0]

        def mock_run_experiment(**kwargs):
            n = kwargs["n_concurrent"]
            return _fake_experiment_result(n), "fake stdout"

        def mock_find_latest(jobs_dir):
            job_counter[0] += 1
            job_dir = tmp_path / f"fake_job_{job_counter[0]}"
            job_dir.mkdir(exist_ok=True)
            return job_dir

        split_counter = [0]

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            split_counter[0] += 1
            return _fake_time_split(max(1, split_counter[0]))

        def mock_get_found_suite_tasks(job_dir):
            # Only "password-recovery" was found; "hello-world" was silently dropped
            return {"password-recovery"}

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_found_suite_tasks", side_effect=mock_get_found_suite_tasks),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
            # keep orchestration tests hermetic: no real sampling thread,
            # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/hello-world,password-recovery",
            "--concurrency",
            "1",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                benchmark_sweep.main()
            except SystemExit as e:
                if e.code != 0:
                    raise
            finally:
                for p in patches:
                    p.stop()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "missing from results" in captured.out
        assert "hello-world" in captured.out
        assert "terminal-bench" in captured.out  # dataset name in warning

    def test_warning_with_multiple_missing_tasks(self, tmp_path, capsys):
        """Warning lists all missing tasks when multiple are absent."""
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        out_dir = tmp_path / "sweep_out"
        job_counter = [0]

        def mock_run_experiment(**kwargs):
            n = kwargs["n_concurrent"]
            return _fake_experiment_result(n), "fake stdout"

        def mock_find_latest(jobs_dir):
            job_counter[0] += 1
            job_dir = tmp_path / f"fake_job_{job_counter[0]}"
            job_dir.mkdir(exist_ok=True)
            return job_dir

        split_counter = [0]

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            split_counter[0] += 1
            return _fake_time_split(max(1, split_counter[0]))

        def mock_get_found_suite_tasks(job_dir):
            # Only "real-task" found; "bad-task-1" and "bad-task-2" missing
            return {"real-task"}

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_found_suite_tasks", side_effect=mock_get_found_suite_tasks),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
            # keep orchestration tests hermetic: no real sampling thread,
            # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/real-task,bad-task-1,bad-task-2",
            "--concurrency",
            "1",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                benchmark_sweep.main()
            except SystemExit as e:
                if e.code != 0:
                    raise
            finally:
                for p in patches:
                    p.stop()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "2 requested task(s) missing" in captured.out
        assert "bad-task-1" in captured.out
        assert "bad-task-2" in captured.out

    def test_warning_when_harbor_fails_no_job_dir(self, tmp_path, capsys):
        """When harbor fails entirely (no job dir), suite warning still fires."""
        from unittest.mock import patch

        from tasktrace_capture import cli as benchmark_sweep

        out_dir = tmp_path / "sweep_out"

        def mock_run_experiment_fail(**kwargs):
            n = kwargs["n_concurrent"]
            return {
                "n_concurrent": n,
                "error": "harbor failed (rc=1)",
                "harbor_stderr": "some error",
            }, "fake stdout"

        def mock_find_latest(jobs_dir):
            # Harbor failed — no new job directory was created
            return None

        def mock_parse_split(job_dir, agent_timeout=None, agent="terminus-2"):
            return None

        def mock_get_found_suite_tasks(job_dir):
            # Should NOT be called when harbor_job_dir is None
            raise AssertionError("get_found_suite_tasks should not be called when harbor_job_dir is None")

        patches = [
            patch.object(benchmark_sweep, "run_experiment", side_effect=mock_run_experiment_fail),
            patch.object(benchmark_sweep, "find_latest_harbor_job", side_effect=mock_find_latest),
            patch.object(benchmark_sweep, "parse_agent_time_split", side_effect=mock_parse_split),
            patch.object(benchmark_sweep, "get_found_suite_tasks", side_effect=mock_get_found_suite_tasks),
            patch.object(benchmark_sweep, "get_cpu_info", return_value={"model": "Test CPU", "cores_logical": 64}),
            patch.object(
                benchmark_sweep, "collect_system_info", return_value={"cpu": {"model": "Test CPU"}, "gpu": None}
            ),
            patch(
                "tasktrace_capture.monitor.sample_baseline",
                lambda duration_sec=5: {
                    "duration_sec": duration_sec,
                    "avg_cpu_pct": 0.5,
                    "avg_cores_active": 2.0,
                    "peak_cores_active": 5,
                    "n_cores": 64,
                },
            ),
            patch("tasktrace_capture.plot_scaling.generate_plots", lambda data, od: None),
            patch("tasktrace_capture.plot_timing.generate_timing_charts", lambda d: False),
            patch.object(benchmark_sweep, "_cleanup_containers", lambda: None),
            # keep orchestration tests hermetic: no real sampling thread,
            # df/lsblk subprocesses, or telemetry.raw in the tmp sweep dir
            patch("tasktrace_capture.node_telemetry.NodeTelemetryRecorder"),
            patch("os._exit", side_effect=SystemExit(0)),
        ]

        full_args = [
            "--task",
            "terminal-bench/task-a,task-b,task-c",
            "--concurrency",
            "1",
            "--output-dir",
            str(out_dir),
            "--api-base",
            "http://fake:8000/v1",
            "--skip-preflight",
        ]
        with patch("sys.argv", ["tasktrace-capture"] + full_args):
            for p in patches:
                p.start()
            try:
                with pytest.raises(SystemExit) as exc_info:
                    benchmark_sweep.main()
                assert exc_info.value.code == 1
            finally:
                for p in patches:
                    p.stop()

        captured = capsys.readouterr()
        assert "WARNING" in captured.out
        assert "Harbor produced no job directory" in captured.out
        assert "3 suite task(s) missing" in captured.out
        assert "task-a" in captured.out
        assert "task-b" in captured.out
        assert "task-c" in captured.out
