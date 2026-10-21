# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Plot-related tests extracted from test_bench.py.

Covers: generate_plots (scaling charts), generate_timing_charts (per-trial timing),
        scaling_results.json round-trip, load_trials with new runs/ layout,
        and real-data integration tests.

Run:  pytest tests/test_plots.py -v
"""

import json
import random
from datetime import datetime, timedelta
from pathlib import Path

import pytest

# ===========================================================================
#  Helpers -- build synthetic data for plot tests
# ===========================================================================


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


def _build_timing_sweep_dir(tmp_path, concurrency_levels=(1, 2, 4)):
    """Build a fake sweep directory with harbor_job__N_agents symlinks and trial result.json files."""
    sweep_dir = tmp_path / "sweep_output"
    sweep_dir.mkdir()

    for n in concurrency_levels:
        job_dir = tmp_path / f"job_{n}"
        job_dir.mkdir()

        for i in range(n):
            trial_name = f"test-task__trial{i}"
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            result = _make_harbor_trial_result(
                trial_name,
                started_at_offset_s=i * 0.5,
                total_s=random.uniform(80, 150),
            )
            with open(trial_dir / "result.json", "w") as f:
                json.dump(result, f)

        link_name = sweep_dir / f"harbor_job__{n}_agents"
        link_name.symlink_to(job_dir.resolve())

    return sweep_dir


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


def _build_runs_layout(tmp_path, concurrency_levels=(1, 2)):
    """Build a sweep dir using the new runs/ layout (no symlinks)."""
    sweep_dir = tmp_path / "sweep"
    sweep_dir.mkdir()
    runs_dir = sweep_dir / "runs"
    runs_dir.mkdir()

    for n in concurrency_levels:
        job_dir = runs_dir / f"2026-01-01__00-0{n}-00"
        job_dir.mkdir()
        for i in range(n):
            trial_name = f"test-task__trial{i}"
            trial_dir = job_dir / trial_name
            trial_dir.mkdir()
            result = _make_harbor_trial_result(
                trial_name,
                started_at_offset_s=i * 0.5,
                total_s=random.uniform(80, 150),
            )
            with open(trial_dir / "result.json", "w") as f:
                json.dump(result, f)

    _build_scaling_results_json(sweep_dir, concurrency_levels)
    return sweep_dir


# ===========================================================================
#  plot.py tests -- generate_plots with synthetic data
# ===========================================================================


class TestGeneratePlots:

    EXPECTED_FILES = [
        "fig1_cores.png",
        "fig1b_proctypes.png",
        "fig3_bursts.png",
        "fig2_cores_timeline.png",
        "summary_table.png",
        "fig4_time_split.png",
        "fig6_disk.png",
    ]

    def test_all_charts_created(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results(concurrency_levels=(1, 2, 4))
        generate_plots(data, tmp_path)
        for fname in self.EXPECTED_FILES:
            fpath = tmp_path / fname
            assert fpath.exists(), f"Missing expected chart: {fname}"
            assert fpath.stat().st_size > 1000, f"Chart too small (corrupt?): {fname}"

    def test_charts_with_single_concurrency(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results(concurrency_levels=(1,))
        generate_plots(data, tmp_path)
        assert (tmp_path / "fig1_cores.png").exists()
        assert (tmp_path / "fig3_bursts.png").exists()
        assert (tmp_path / "summary_table.png").exists()

    def test_charts_without_time_splits(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results(with_time_splits=False)
        generate_plots(data, tmp_path)
        assert (tmp_path / "fig1_cores.png").exists()
        assert not (tmp_path / "fig4_time_split.png").exists()

    def test_charts_with_no_disk_io(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results()
        for r in data["results"]:
            r["avg_disk_read_mb_s"] = 0.0
            r["avg_disk_write_mb_s"] = 0.0
        generate_plots(data, tmp_path)
        assert (tmp_path / "fig1_cores.png").exists()
        assert not (tmp_path / "fig6_disk.png").exists()

    def test_charts_with_many_concurrency_levels(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results(concurrency_levels=(1, 2, 4, 8, 16))
        generate_plots(data, tmp_path)
        for fname in ["fig1_cores.png", "fig3_bursts.png", "summary_table.png"]:
            assert (tmp_path / fname).exists()

    def test_png_files_are_valid(self, tmp_path):
        """Check that PNGs start with the correct magic bytes."""
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results()
        generate_plots(data, tmp_path)
        png_magic = b"\x89PNG"
        for fname in self.EXPECTED_FILES:
            fpath = tmp_path / fname
            if fpath.exists():
                with open(fpath, "rb") as f:
                    header = f.read(4)
                assert header == png_magic, f"{fname} is not a valid PNG"

    def test_charts_with_empty_proc_categories(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results()
        for r in data["results"]:
            r["avg_proc_categories"] = {}
        generate_plots(data, tmp_path)
        assert (tmp_path / "fig1_cores.png").exists()
        assert not (tmp_path / "fig1b_proctypes.png").exists()


# ===========================================================================
#  plot_timing.py tests -- generate_timing_charts with synthetic job dirs
# ===========================================================================


class TestGenerateTimingCharts:

    EXPECTED_TIMING_FILES = [
        "timing_wall_time.png",
        "timing_phase_breakdown.png",
        "timing_llm_vs_tool.png",
        "timing_agent_timeline.png",
        "timing_llm_latency.png",
        "timing_summary.png",
    ]

    def test_all_timing_charts_created(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        sweep_dir = _build_timing_sweep_dir(tmp_path, concurrency_levels=(1, 2, 4))
        result = generate_timing_charts(str(sweep_dir))
        assert result is True
        charts_dir = sweep_dir / "timing_charts"
        for fname in self.EXPECTED_TIMING_FILES:
            fpath = charts_dir / fname
            assert fpath.exists(), f"Missing timing chart: {fname}"
            assert fpath.stat().st_size > 1000, f"Timing chart too small: {fname}"

    def test_timing_charts_single_concurrency(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        sweep_dir = _build_timing_sweep_dir(tmp_path, concurrency_levels=(1,))
        result = generate_timing_charts(str(sweep_dir))
        assert result is True
        charts_dir = sweep_dir / "timing_charts"
        assert (charts_dir / "timing_wall_time.png").exists()

    def test_timing_charts_empty_dir(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        empty = tmp_path / "empty_sweep"
        empty.mkdir()
        result = generate_timing_charts(str(empty))
        assert result is False

    def test_timing_charts_valid_pngs(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        sweep_dir = _build_timing_sweep_dir(tmp_path)
        generate_timing_charts(str(sweep_dir))
        charts_dir = sweep_dir / "timing_charts"
        png_magic = b"\x89PNG"
        for fname in self.EXPECTED_TIMING_FILES:
            fpath = charts_dir / fname
            if fpath.exists():
                with open(fpath, "rb") as f:
                    header = f.read(4)
                assert header == png_magic, f"{fname} is not a valid PNG"

    def test_timing_charts_many_agents(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        sweep_dir = _build_timing_sweep_dir(tmp_path, concurrency_levels=(1, 4, 8))
        result = generate_timing_charts(str(sweep_dir))
        assert result is True
        charts_dir = sweep_dir / "timing_charts"
        assert (charts_dir / "timing_agent_timeline.png").exists()


# ===========================================================================
#  New runs/ layout: load_trials + generate_timing_charts
# ===========================================================================


class TestLoadTrialsNewLayout:
    """load_trials with the new runs/ folder structure."""

    def test_loads_correct_concurrency_levels(self, tmp_path):
        from tasktrace_capture.plot_timing import load_trials

        sweep_dir = _build_runs_layout(tmp_path, concurrency_levels=(1, 2))
        data = load_trials(str(sweep_dir))
        assert set(data.keys()) == {1, 2}

    def test_trial_count_matches_concurrency(self, tmp_path):
        from tasktrace_capture.plot_timing import load_trials

        sweep_dir = _build_runs_layout(tmp_path, concurrency_levels=(1, 2, 4))
        data = load_trials(str(sweep_dir))
        assert len(data[1]) == 1
        assert len(data[2]) == 2
        assert len(data[4]) == 4

    def test_trial_has_expected_fields(self, tmp_path):
        from tasktrace_capture.plot_timing import load_trials

        sweep_dir = _build_runs_layout(tmp_path, concurrency_levels=(1,))
        data = load_trials(str(sweep_dir))
        trial = data[1][0]
        for field in ("trial_name", "total_s", "llm_total_s", "tool_time_s", "n_llm_calls", "llm_times_ms"):
            assert field in trial, f"Missing field: {field}"

    def test_timing_charts_generated_from_runs_layout(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        sweep_dir = _build_runs_layout(tmp_path, concurrency_levels=(1, 2))
        result = generate_timing_charts(str(sweep_dir))
        assert result is True
        charts_dir = sweep_dir / "timing_charts"
        assert (charts_dir / "timing_wall_time.png").exists()
        assert (charts_dir / "timing_llm_vs_tool.png").exists()

    def test_legacy_layout_still_works(self, tmp_path):
        """The old harbor_job__N_agents symlink layout must still function."""
        from tasktrace_capture.plot_timing import load_trials

        sweep_dir = _build_timing_sweep_dir(tmp_path, concurrency_levels=(1, 2))
        data = load_trials(str(sweep_dir))
        assert set(data.keys()) == {1, 2}


# ===========================================================================
#  Integration: scaling_results.json round-trip (write + plot)
# ===========================================================================


class TestScalingResultsRoundTrip:
    """Write scaling_results.json, then verify generate_plots reads it correctly."""

    def test_write_and_plot_from_json(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        data = _make_scaling_results(concurrency_levels=(1, 2, 4))
        json_path = tmp_path / "scaling_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f)

        with open(json_path) as f:
            loaded = json.load(f)

        generate_plots(loaded, tmp_path)

        assert (tmp_path / "fig1_cores.png").exists()
        assert (tmp_path / "fig4_time_split.png").exists()
        assert (tmp_path / "summary_table.png").exists()

    def test_json_structure_has_required_keys(self, tmp_path):
        data = _make_scaling_results()
        required_keys = {"cpu_info", "dataset", "results", "time_splits"}
        assert required_keys.issubset(set(data.keys()))

        for r in data["results"]:
            result_keys = {
                "n_concurrent",
                "n_cores",
                "avg_cpu",
                "peak_cpu",
                "avg_cores_active",
                "peak_cores_active",
                "avg_procs",
                "peak_procs",
                "samples",
            }
            assert result_keys.issubset(set(r.keys())), f"Missing keys in result: {result_keys - set(r.keys())}"


# ===========================================================================
#  Integration: plot with real data (if available)
# ===========================================================================

REAL_SWEEP_DIR = Path("/path/to/harbor/jobs/2026-04-02__09-53-11__sweep__qemu-startup")


@pytest.mark.skipif(not REAL_SWEEP_DIR.exists(), reason="Real sweep data not available")
class TestPlotsWithRealData:

    def test_generate_plots_from_real_data(self, tmp_path):
        from tasktrace_capture.plot_scaling import generate_plots

        with open(REAL_SWEEP_DIR / "scaling_results.json") as f:
            data = json.load(f)
        generate_plots(data, tmp_path)
        assert (tmp_path / "fig1_cores.png").exists()
        assert (tmp_path / "fig3_bursts.png").exists()
        assert (tmp_path / "summary_table.png").exists()

    def test_generate_timing_from_real_data(self, tmp_path):
        from tasktrace_capture.plot_timing import generate_timing_charts

        result = generate_timing_charts(str(REAL_SWEEP_DIR))
        # May or may not have symlinks that resolve; just verify no crash
        assert result in (True, False)
