# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for report/profile writing and summary table output.

Extracted from test_bench.py: TestSummaryTable, TestWriteReportJson,
TestWriteReport, TestAllInvalidTrials, plus helper functions they depend on.

Run:  pytest tests/test_report.py -v
"""

import json
import random
from datetime import datetime, timedelta

from tasktrace_capture.runner import parse_agent_time_split

# ===========================================================================
#  Helpers -- build fake Harbor job directories
# ===========================================================================


def _make_result_json(
    tmp_path,
    trial_name,
    *,
    started_at,
    finished_at,
    api_times_msec=None,
    omit_agent_execution=False,
    omit_api_times=False,
    reward=None,
    n_episodes=None,
):
    """Write a minimal result.json under tmp_path/<trial_name>/result.json."""
    trial_dir = tmp_path / trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)

    agent_exec = {} if omit_agent_execution else {"started_at": started_at, "finished_at": finished_at}

    meta = {}
    if api_times_msec is not None and not omit_api_times:
        meta["api_request_times_msec"] = api_times_msec
    if n_episodes is not None:
        meta["n_episodes"] = n_episodes

    result = {
        "agent_execution": agent_exec,
        "agent_result": {"metadata": meta},
    }
    if reward is not None:
        result["verifier_result"] = {"rewards": {"reward": reward}}

    with open(trial_dir / "result.json", "w") as f:
        json.dump(result, f)
    return trial_dir


def _make_cpu_samples(n_concurrent, duration_sec=120, interval=0.5):
    """Generate realistic-looking CPU sample data for n_concurrent agents."""
    samples = []
    n_points = int(duration_sec / interval)
    for i in range(n_points):
        t = round(i * interval, 2)
        base_cpu = 0.3 * n_concurrent + random.uniform(0, 1.5)
        cores_active = max(2, int(n_concurrent * 2.5 + random.randint(-2, 4)))
        in_container = 0.1 < t / duration_sec < 0.9
        samples.append(
            {
                "t": t,
                "cpu_total": round(base_cpu, 1),
                "cores_active": cores_active,
                "n_cores": 128,
                "n_containers": n_concurrent if in_container else 0,
                "n_procs": n_concurrent * 3 if in_container else 0,
                "proc_cats": {"python": n_concurrent, "bash/sh": n_concurrent * 2} if in_container else {},
                "disk_read_mb_s": round(random.uniform(0, 2), 2),
                "disk_write_mb_s": round(random.uniform(5, 50) * n_concurrent, 2),
            }
        )
    return samples


def _make_scaling_results(concurrency_levels=(1, 2, 4), with_time_splits=True):
    """Build a complete scaling_results.json data dict."""
    results = []
    time_splits = {}
    for n in concurrency_levels:
        samples = _make_cpu_samples(n)
        active_samples = [s for s in samples if s["n_containers"] > 0]
        r = {
            "n_concurrent": n,
            "n_cores": 128,
            "duration_sec": round(samples[-1]["t"], 1),
            "n_samples": len(samples),
            "n_active_samples": len(active_samples),
            "avg_cpu": round(sum(s["cpu_total"] for s in active_samples) / max(len(active_samples), 1), 1),
            "peak_cpu": round(max(s["cpu_total"] for s in active_samples), 1) if active_samples else 0,
            "avg_cores_active": round(sum(s["cores_active"] for s in active_samples) / max(len(active_samples), 1), 1),
            "peak_cores_active": max(s["cores_active"] for s in active_samples) if active_samples else 0,
            "p95_cores_active": (
                sorted(s["cores_active"] for s in active_samples)[int(0.95 * len(active_samples))]
                if active_samples
                else 0
            ),
            "avg_procs": round(sum(s["n_procs"] for s in active_samples) / max(len(active_samples), 1), 1),
            "peak_procs": max(s["n_procs"] for s in active_samples) if active_samples else 0,
            "pct_cpu_active": round(sum(s["cpu_total"] for s in active_samples) / max(len(active_samples), 1), 1),
            "pct_waiting_llm": 95.0,
            "avg_proc_categories": {"python": float(n), "bash/sh": float(n * 2)},
            "avg_disk_read_mb_s": 0.5,
            "avg_disk_write_mb_s": round(20.0 * n, 1),
            "peak_disk_read_mb_s": 2.0,
            "peak_disk_write_mb_s": round(100.0 * n, 1),
            "samples": samples,
        }
        results.append(r)

        if with_time_splits:
            pct_cpu = round(40 + n * 5, 1)
            time_splits[str(n)] = {
                "pct_cpu": pct_cpu,
                "pct_llm": round(100 - pct_cpu, 1),
            }

    return {
        "cpu_info": {"model": "Test CPU Model X", "cores_logical": 128},
        "dataset": "test-dataset",
        "model": "hosted_vllm/model",
        "api_base": "http://test:8000/v1",
        "timestamp": "2026-04-01__00-00-00",
        "hostname": "test-node",
        "time_splits": time_splits,
        "suite_mode": False,
        "results": results,
    }


def _make_harbor_trial_result(trial_name, started_at_offset_s, total_s, llm_calls=15, base_time="2026-04-02T00:00:00Z"):
    """Build a realistic Harbor result.json dict with all phases."""
    base = datetime.fromisoformat(base_time.replace("Z", "+00:00"))
    started = base + timedelta(seconds=started_at_offset_s)
    finished = started + timedelta(seconds=total_s)

    env_end = started + timedelta(seconds=10)
    setup_end = env_end + timedelta(seconds=3)
    exec_start = setup_end
    exec_end = finished - timedelta(seconds=0.5)

    api_times = [random.uniform(500, 3000) for _ in range(llm_calls)]

    return {
        "trial_name": trial_name,
        "task_name": "test-task",
        "started_at": started.isoformat(),
        "finished_at": finished.isoformat(),
        "environment_setup": {
            "started_at": started.isoformat(),
            "finished_at": env_end.isoformat(),
        },
        "agent_setup": {
            "started_at": env_end.isoformat(),
            "finished_at": setup_end.isoformat(),
        },
        "agent_execution": {
            "started_at": exec_start.isoformat(),
            "finished_at": exec_end.isoformat(),
        },
        "verifier": None,
        "agent_result": {
            "n_input_tokens": random.randint(50000, 100000),
            "n_output_tokens": random.randint(2000, 8000),
            "metadata": {
                "api_request_times_msec": api_times,
                "n_episodes": llm_calls + 1,
            },
        },
    }


def _build_scaling_results_json(tmp_path, concurrency_levels):
    """Write a minimal scaling_results.json matching the given concurrency levels."""
    results = [
        {
            "n_concurrent": n,
            "n_cores": 128,
            "avg_cpu": 1.0,
            "peak_cpu": 5.0,
            "avg_cores_active": 8.0,
            "peak_cores_active": 30,
            "p95_cores_active": 28,
            "avg_procs": 0,
            "peak_procs": 0,
            "duration_sec": 60,
            "n_samples": 10,
            "n_active_samples": 10,
            "pct_cpu_active": 1.0,
            "pct_waiting_llm": 99.0,
            "avg_disk_read_mb_s": 0.0,
            "avg_disk_write_mb_s": 5.0,
            "peak_disk_read_mb_s": 0.0,
            "peak_disk_write_mb_s": 50.0,
            "avg_proc_categories": {},
            "samples": [],
        }
        for n in concurrency_levels
    ]
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
            str(n): {
                "pct_cpu": 50.0,
                "pct_llm": 50.0,
                "tool_time_sec": 30.0,
                "wall_time_sec": 60.0,
                "llm_time_sec": 30.0,
                "n_llm_calls": 5.0,
                "avg_reward": 1.0,
                "n_full": n,
                "n_partial": 0,
                "n_invalid": 0,
                "survival_rate": f"{n}/{n}",
                "trials": [],
            }
            for n in concurrency_levels
        },
        "results": results,
        "baseline_cpu": {},
    }
    path = tmp_path / "scaling_results.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return data


# ===========================================================================
#  Summary table
# ===========================================================================


class TestSummaryTable:
    """Tests for the CLI summary table printed after each run."""

    def _make_data(self, trials, model="test-model", dataset="test-task"):
        from tasktrace_capture.cli import _fmt_time, _print_summary_table

        return (
            _print_summary_table,
            _fmt_time,
            {
                "dataset": dataset,
                "model": model,
                "time_splits": {
                    "1": {
                        "pct_cpu": 72.1,
                        "pct_llm": 27.9,
                        "avg_reward": (
                            sum(t["reward"] for t in trials if t.get("reward") is not None)
                            / max(1, sum(1 for t in trials if t.get("reward") is not None))
                            if any(t.get("reward") is not None for t in trials)
                            else None
                        ),
                        "n_full": sum(1 for t in trials if t.get("status") == "full"),
                        "n_partial": sum(1 for t in trials if t.get("status") == "partial"),
                        "n_invalid": sum(1 for t in trials if t.get("status") == "invalid"),
                        "survival_rate": f"{sum(1 for t in trials if t.get('status') == 'full')}/{len(trials)}",
                        "trials": trials,
                    }
                },
            },
        )

    def test_fmt_time_basic(self):
        from tasktrace_capture.cli import _fmt_time

        assert _fmt_time(0) == "0m00s"
        assert _fmt_time(60) == "1m00s"
        assert _fmt_time(855) == "14m15s"
        assert _fmt_time(3600) == "60m00s"

    def test_no_crash_on_pass(self, capsys):
        fn, _, data = self._make_data(
            [
                {
                    "status": "full",
                    "wall_time_sec": 855.73,
                    "llm_time_sec": 238.68,
                    "tool_time_sec": 617.06,
                    "n_llm_calls": 121,
                    "pct_cpu": 72.1,
                    "pct_llm": 27.9,
                    "reward": 1.0,
                    "n_episodes": 121,
                }
            ]
        )
        fn(data)  # must not raise

    def test_no_crash_on_fail(self, capsys):
        fn, _, data = self._make_data(
            [
                {
                    "status": "full",
                    "wall_time_sec": 300.0,
                    "llm_time_sec": 100.0,
                    "tool_time_sec": 200.0,
                    "n_llm_calls": 50,
                    "pct_cpu": 66.7,
                    "pct_llm": 33.3,
                    "reward": 0.0,
                    "n_episodes": 50,
                }
            ]
        )
        fn(data)

    def test_no_crash_missing_reward(self, capsys):
        fn, _, data = self._make_data(
            [
                {
                    "status": "full",
                    "wall_time_sec": 300.0,
                    "llm_time_sec": 100.0,
                    "tool_time_sec": 200.0,
                    "n_llm_calls": 50,
                    "pct_cpu": 66.7,
                    "pct_llm": 33.3,
                    "reward": None,
                    "n_episodes": 50,
                }
            ]
        )
        fn(data)

    def test_empty_time_splits_no_crash(self, capsys):
        from tasktrace_capture.cli import _print_summary_table

        _print_summary_table({"dataset": "x", "model": "y", "time_splits": {}})

    def test_no_time_splits_key_no_crash(self, capsys):
        from tasktrace_capture.cli import _print_summary_table

        _print_summary_table({"dataset": "x", "model": "y"})


# ===========================================================================
#  _write_report_json
# ===========================================================================


class TestWriteReportJson:
    """_write_report_json produces valid report.json with expected structure."""

    def _make_data(self, tmp_path, concurrency_levels=(1, 2)):
        data = _build_scaling_results_json(tmp_path, concurrency_levels)
        # Add raw samples so p95 can be computed
        for r in data["results"]:
            r["samples"] = [{"cores_active": i, "n_containers": 1, "n_procs": 1} for i in range(1, 21)]
        data["baseline_cpu"] = {"avg_cpu_pct": 0.5, "avg_cores_active": 3.0, "peak_cores_active": 8, "n_cores": 128}
        return data

    def test_report_json_written(self, tmp_path):
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path)
        _write_report_json(data, tmp_path)
        assert (tmp_path / "report.json").exists()

    def test_report_json_has_top_level_keys(self, tmp_path):
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path)
        _write_report_json(data, tmp_path)
        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        for key in ("dataset", "model", "baseline", "concurrency", "cpu", "hostname", "system_info"):
            assert key in p, f"Missing top-level key: {key}"

    def test_report_json_concurrency_keys(self, tmp_path):
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path, concurrency_levels=(1, 2, 4))
        _write_report_json(data, tmp_path)
        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        assert set(p["concurrency"].keys()) == {"1", "2", "4"}

    def test_report_json_level_has_sections(self, tmp_path):
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path)
        _write_report_json(data, tmp_path)
        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        level = p["concurrency"]["1"]
        for section in ("timing", "cpu", "disk", "reward"):
            assert section in level, f"Missing section: {section}"

    def test_report_json_no_samples_array(self, tmp_path):
        """report.json must not contain the raw samples array."""
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path)
        _write_report_json(data, tmp_path)
        content = (tmp_path / "report.json").read_text()
        assert '"samples"' not in content

    def test_report_json_p95_computed(self, tmp_path):
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        _write_report_json(data, tmp_path)
        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        assert p["concurrency"]["1"]["cpu"]["p95_cores"] is not None

    def test_report_json_p95_never_exceeds_peak(self, tmp_path):
        """Regression: p95_cores must never exceed peak_cores.

        Previously p95 was computed from container-filtered samples while peak
        was computed from active_samples (trial-timestamp window). Different
        populations allowed p95 > peak. Now both come from active_samples.
        """
        from tasktrace_capture.cli import _write_report_json

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        # Set p95 and peak from the same active_samples population
        # (the runner now provides both)
        data["results"][0]["p95_cores_active"] = 25
        data["results"][0]["peak_cores_active"] = 30
        _write_report_json(data, tmp_path)
        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        cpu = p["concurrency"]["1"]["cpu"]
        assert (
            cpu["p95_cores"] <= cpu["peak_cores"]
        ), f"p95_cores ({cpu['p95_cores']}) > peak_cores ({cpu['peak_cores']})"


# ===========================================================================
#  _write_report
# ===========================================================================


class TestWriteReport:
    """_write_report produces a valid report.txt with expected content."""

    def _make_data(self, tmp_path, concurrency_levels=(1, 2)):
        data = _build_scaling_results_json(tmp_path, concurrency_levels)
        for r in data["results"]:
            r["samples"] = [{"cores_active": i, "n_containers": 1, "n_procs": 1} for i in range(1, 21)]
        data["baseline_cpu"] = {
            "avg_cpu_pct": 0.7,
            "avg_cores_active": 3.0,
            "peak_cores_active": 8,
            "n_cores": 128,
        }
        return data

    def test_report_txt_written(self, tmp_path):
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path)
        _write_report(data, tmp_path)
        assert (tmp_path / "report.txt").exists()

    def test_report_has_three_sections(self, tmp_path):
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path)
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "Time Split" in text
        assert "CPU Usage" in text
        assert "Rewards" in text

    def test_report_has_header_metadata(self, tmp_path):
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path)
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "Dataset" in text
        assert "Model" in text
        assert "Baseline" in text

    def test_report_has_all_concurrency_rows(self, tmp_path):
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1, 2, 4))
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        # Each concurrency level should appear in at least two sections
        for n in ("1", "2", "4"):
            assert text.count(f"{n:>12}") >= 2, f"Concurrency {n} missing from sections"

    def test_report_no_raw_samples(self, tmp_path):
        """report.txt must not contain the raw samples array."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path)
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert '"samples"' not in text
        assert "cores_active" not in text

    def test_report_empty_when_no_time_splits(self, tmp_path):
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path)
        data["time_splits"] = {}
        _write_report(data, tmp_path)
        assert not (tmp_path / "report.txt").exists()

    def test_report_reward_na_when_none(self, tmp_path):
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        data["time_splits"]["1"]["avg_reward"] = None
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "n/a" in text

    def test_report_reward_shows_verifier_errors(self, tmp_path):
        """When avg_reward is None due to verifier errors, report shows error details."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        data["time_splits"]["1"]["avg_reward"] = None
        data["time_splits"]["1"]["n_verifier_errors"] = 2
        data["time_splits"]["1"]["verifier_errors"] = ["AddTestsDirError"]
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "verifier err" in text
        assert "AddTestsDirError" in text

    def test_report_reward_no_verifier_error_when_reward_present(self, tmp_path):
        """When avg_reward is present, verifier error details are not shown."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        data["time_splits"]["1"]["avg_reward"] = 0.75
        data["time_splits"]["1"]["n_verifier_errors"] = 0
        data["time_splits"]["1"]["verifier_errors"] = []
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "0.750" in text
        assert "verifier err" not in text

    def test_report_survives_missing_samples(self, tmp_path):
        """Concurrency levels with no samples list still produce rows."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1, 2))
        for r in data["results"]:
            r.pop("samples", None)  # remove samples entirely
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "Time Split" in text
        assert "1" in text

    def test_report_gpu_new_dict_format(self, tmp_path):
        """GPU system info with new {summary: [...], nvidia_smi_q: ...} format."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        data["system_info"] = {
            "gpu": {
                "summary": [
                    {
                        "name": "NVIDIA H100 80GB HBM3",
                        "vbios": "96.00.BC.00.01",
                        "driver": "535.216.03",
                        "memory_mb": 81559,
                    }
                ],
                "nvidia_smi_q": "==============NVSMI LOG==============\nProduct Name: NVIDIA H100\n",
            }
        }
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "H100" in text
        assert "535.216.03" in text

    def test_report_gpu_legacy_list_format(self, tmp_path):
        """GPU system info with old list-of-dicts format still works."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        data["system_info"] = {
            "gpu": [{"name": "NVIDIA A100", "vbios": "94.00", "driver": "525.0", "memory_mb": 40960}],
        }
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "A100" in text
        assert "525.0" in text

    def test_report_gpu_none(self, tmp_path):
        """GPU = None produces no GPU line but doesn't crash."""
        from tasktrace_capture.cli import _write_report

        data = self._make_data(tmp_path, concurrency_levels=(1,))
        data["system_info"] = {"gpu": None}
        _write_report(data, tmp_path)
        text = (tmp_path / "report.txt").read_text()
        assert "GPU" not in text


# ===========================================================================
#  All-invalid trials: no KeyError anywhere
# ===========================================================================


class TestAllInvalidTrials:
    """When parse_agent_time_split returns a dict without pct_cpu (all trials
    invalid), nothing in the pipeline should crash."""

    def _make_all_invalid_split(self):
        """Mimics parse_agent_time_split return when every trial is invalid."""
        return {
            "n_full": 0,
            "n_partial": 0,
            "n_invalid": 3,
            "survival_rate": "0/3",
            "trials": [{"status": "invalid"}] * 3,
        }

    def test_parse_all_invalid_has_no_pct_cpu(self, tmp_path):
        """Truly invalid trials (zero wall, missing timestamps) have pct_cpu=None."""
        _make_result_json(
            tmp_path,
            "bad1",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:00Z",
            api_times_msec=[100],
        )
        _make_result_json(
            tmp_path,
            "bad2",
            started_at="2026-04-02T00:00:00Z",
            finished_at="2026-04-02T00:00:00Z",
            omit_api_times=True,
        )
        r = parse_agent_time_split(tmp_path)
        assert r is not None
        assert r.pct_cpu is None
        assert r.n_invalid == 2
        assert r.survival_rate == "0/2"

    def test_write_report_json_with_all_invalid_split(self, tmp_path):
        """report.json writes even when time_splits have no timing data."""
        from tasktrace_capture.cli import _write_report_json

        data = _build_scaling_results_json(tmp_path, (1,))
        data["time_splits"]["1"] = self._make_all_invalid_split()
        data["baseline_cpu"] = {}
        _write_report_json(data, tmp_path)
        assert (tmp_path / "report.json").exists()
        with open(tmp_path / "report.json") as f:
            p = json.load(f)
        level = p["concurrency"]["1"]
        assert level["timing"]["pct_tool"] is None
        assert level["reward"]["survival_rate"] == "0/3"

    def test_write_report_with_all_invalid_split(self, tmp_path):
        """report.txt writes rows showing 0.0 for missing timing fields."""
        from tasktrace_capture.cli import _write_report

        data = _build_scaling_results_json(tmp_path, (1, 2))
        data["time_splits"]["1"] = self._make_all_invalid_split()
        data["baseline_cpu"] = {}
        _write_report(data, tmp_path)
        assert (tmp_path / "report.txt").exists()
        text = (tmp_path / "report.txt").read_text()
        assert "0/3" in text  # survival rate appears
        assert "Time Split" in text

    def test_write_report_all_levels_invalid(self, tmp_path):
        """report.txt still writes when every concurrency level is all-invalid."""
        from tasktrace_capture.cli import _write_report

        data = _build_scaling_results_json(tmp_path, (1, 2, 4))
        for k in data["time_splits"]:
            data["time_splits"][k] = self._make_all_invalid_split()
        data["baseline_cpu"] = {}
        _write_report(data, tmp_path)
        assert (tmp_path / "report.txt").exists()

    def test_aggregate_splits_with_all_invalid(self):
        """_aggregate_splits handles splits that have no timing keys."""
        from tasktrace_capture.cli import _aggregate_splits

        splits = [
            {"n_full": 0, "n_partial": 0, "n_invalid": 2, "survival_rate": "0/2", "trials": []},
            {"n_full": 0, "n_partial": 0, "n_invalid": 3, "survival_rate": "0/3", "trials": []},
        ]
        agg = _aggregate_splits(splits)
        assert agg["pct_cpu"] is None
        assert agg["tool_time_sec"] is None
        assert agg["survival_rate"] == "0/5"
        assert agg["n_invalid"] == 5

    def test_aggregate_splits_mixed_valid_invalid(self):
        """_aggregate_splits with one valid + one all-invalid split."""
        from tasktrace_capture.cli import _aggregate_splits

        splits = [
            {
                "pct_cpu": 45.0,
                "pct_llm": 55.0,
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
            {"n_full": 0, "n_partial": 0, "n_invalid": 2, "survival_rate": "0/2", "trials": []},
        ]
        agg = _aggregate_splits(splits)
        assert agg["pct_cpu"] is not None
        assert agg["pct_cpu"]["median"] == 45.0
        assert agg["n_invalid"] == 2
        assert agg["n_full"] == 2
