# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import statistics

import pytest

from llmb_run.rl_metrics_parser import (
    _STEP_TIME_KEY,
    _TOKENS_PER_SEC_KEY,
    _WINDOW_END,
    _WINDOW_START,
    RLMetricsParseStatus,
    expected_steps_from_run_log,
    parse_rl_metrics,
)


def _make_series(values: list[float]) -> dict[str, float]:
    """Turn a list into the step-keyed dict that metrics.json uses (1-indexed)."""
    return {str(i): v for i, v in enumerate(values, start=1)}


def _write_metrics(path, step_time_values, tokens_values):
    data = {
        _STEP_TIME_KEY: _make_series(step_time_values),
        _TOKENS_PER_SEC_KEY: _make_series(tokens_values),
    }
    (path / "metrics.json").write_text(json.dumps(data))


def _write_ray_driver_log(path, max_steps):
    """Write a ray-driver.log with the resolved-config dump the trainer emits."""
    driver_dir = path / "logs" / "grpo-run" / "driver"
    driver_dir.mkdir(parents=True, exist_ok=True)
    (driver_dir / "ray-driver.log").write_text(
        "Some startup noise\n"
        "Final config:\n"
        f"{{'grpo': {{'max_num_steps': {max_steps},\n"
        "          'num_prompts_per_step': 64},\n"
        " 'policy': {'train_micro_batch_size': 1}}\n"
        "training begins\n"
    )


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_success_10_step_run_with_steps_from_run_log(tmp_path):
    """Standard 10-step run, step count read from ray-driver.log: mean/stdev over the [2:7] window."""
    step_times = [60.0 + i * 0.5 for i in range(10)]
    tokens = [25.0 + i * 0.1 for i in range(10)]
    _write_metrics(tmp_path, step_times, tokens)
    _write_ray_driver_log(tmp_path, 10)

    result = parse_rl_metrics(tmp_path)

    assert result.succeeded
    assert result.status == RLMetricsParseStatus.SUCCESS
    assert result.metrics is not None

    window_st = step_times[_WINDOW_START:_WINDOW_END]
    window_tk = tokens[_WINDOW_START:_WINDOW_END]

    assert result.metrics.step_time_mean_seconds == pytest.approx(statistics.mean(window_st))
    assert result.metrics.step_time_std_seconds == pytest.approx(statistics.stdev(window_st))
    assert result.metrics.tokens_per_sec_per_gpu_mean == pytest.approx(statistics.mean(window_tk))
    assert result.metrics.tokens_per_sec_per_gpu_std == pytest.approx(statistics.stdev(window_tk))
    assert result.metrics.sample_count == len(window_st)


def test_success_longer_run_uses_same_window(tmp_path):
    """A 20-step run still averages over iterations 3–7 (the same fixed window)."""
    step_times = [50.0 + i for i in range(20)]
    tokens = [30.0 + i * 0.2 for i in range(20)]
    _write_metrics(tmp_path, step_times, tokens)

    result = parse_rl_metrics(tmp_path, expected_steps=20)

    assert result.succeeded
    window_st = step_times[_WINDOW_START:_WINDOW_END]
    assert result.metrics.step_time_mean_seconds == pytest.approx(statistics.mean(window_st))
    assert result.metrics.sample_count == len(window_st)


def test_window_values_are_byte_identical_to_parse_results_py(tmp_path):
    """Numbers must exactly match what parse_results.py would produce.

    parse_results.py computes:
        window = list(raw.values())[2:7]
        mean = statistics.mean(window)
        std  = statistics.stdev(window)
    where raw is an ordered dict {str(step): value} for 10 steps.
    """
    step_times = [61.1956, 62.0, 60.5, 61.5, 62.2, 60.8, 61.9, 60.1, 62.5, 61.0]
    tokens = [26.94, 27.1, 26.5, 27.0, 26.8, 26.9, 27.2, 26.6, 27.3, 26.7]
    _write_metrics(tmp_path, step_times, tokens)

    result = parse_rl_metrics(tmp_path, expected_steps=10)

    window_st = step_times[2:7]
    window_tk = tokens[2:7]
    assert result.metrics.step_time_mean_seconds == pytest.approx(statistics.mean(window_st))
    assert result.metrics.step_time_std_seconds == pytest.approx(statistics.stdev(window_st))
    assert result.metrics.tokens_per_sec_per_gpu_mean == pytest.approx(statistics.mean(window_tk))


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_no_metrics_file(tmp_path):
    result = parse_rl_metrics(tmp_path)
    assert result.status == RLMetricsParseStatus.NO_METRICS_FILE
    assert not result.succeeded
    assert result.metrics is None


def test_invalid_json(tmp_path):
    (tmp_path / "metrics.json").write_text("not valid json {{{")
    result = parse_rl_metrics(tmp_path)
    assert result.status == RLMetricsParseStatus.NO_METRICS_FILE
    assert result.metrics is None


def test_missing_step_time_key(tmp_path):
    data = {_TOKENS_PER_SEC_KEY: _make_series([1.0] * 10)}
    (tmp_path / "metrics.json").write_text(json.dumps(data))
    result = parse_rl_metrics(tmp_path)
    assert result.status == RLMetricsParseStatus.NO_DATA
    assert result.metrics is None


def test_missing_tokens_key(tmp_path):
    data = {_STEP_TIME_KEY: _make_series([60.0] * 10)}
    (tmp_path / "metrics.json").write_text(json.dumps(data))
    result = parse_rl_metrics(tmp_path)
    assert result.status == RLMetricsParseStatus.NO_DATA
    assert result.metrics is None


def test_incomplete_step_time_series(tmp_path):
    """Series too short (crashed before expected_steps recorded) → INCOMPLETE."""
    step_times = [60.0] * 7  # only 7 steps, expected 10
    tokens = [26.0] * 7
    _write_metrics(tmp_path, step_times, tokens)
    result = parse_rl_metrics(tmp_path, expected_steps=10)
    assert result.status == RLMetricsParseStatus.INCOMPLETE
    assert result.metrics is None


def test_incomplete_tokens_series(tmp_path):
    """Step times complete but tokens series short → INCOMPLETE."""
    step_times = [60.0] * 10
    tokens = [26.0] * 8  # only 8 steps
    _write_metrics(tmp_path, step_times, tokens)
    result = parse_rl_metrics(tmp_path, expected_steps=10)
    assert result.status == RLMetricsParseStatus.INCOMPLETE
    assert result.metrics is None


def test_extra_steps_is_incomplete(tmp_path):
    """More steps than expected means the recorded MAX_STEPS is wrong — flag it.

    The benchmark contract is exact: a run launched with MAX_STEPS=10 must
    produce exactly 10 steps, matching parse_results.py's `len != MAX_STEPS`.
    """
    step_times = [60.0] * 12
    tokens = [26.0] * 12
    _write_metrics(tmp_path, step_times, tokens)
    result = parse_rl_metrics(tmp_path, expected_steps=10)
    assert result.status == RLMetricsParseStatus.INCOMPLETE
    assert result.metrics is None


def test_run_shorter_than_window_returns_no_data(tmp_path):
    """A complete run with MAX_STEPS <= 2 has an empty averaging window.

    Must return NO_DATA, not raise StatisticsError (regression test:
    statistics.mean([]) used to propagate and crash `llmb-run jobs list`).
    """
    _write_metrics(tmp_path, [60.0, 61.0], [25.0, 26.0])
    result = parse_rl_metrics(tmp_path, expected_steps=2)
    assert result.status == RLMetricsParseStatus.NO_DATA
    assert result.metrics is None


def test_custom_expected_steps(tmp_path):
    """expected_steps=5: a 5-step run is complete; a 4-step run is not."""
    step_times_5 = [60.0] * 5
    tokens_5 = [26.0] * 5
    _write_metrics(tmp_path, step_times_5, tokens_5)

    result = parse_rl_metrics(tmp_path, expected_steps=5)
    assert result.status == RLMetricsParseStatus.SUCCESS

    # Now truncate one step — should be INCOMPLETE.
    _write_metrics(tmp_path, step_times_5[:4], tokens_5[:4])
    result2 = parse_rl_metrics(tmp_path, expected_steps=5)
    assert result2.status == RLMetricsParseStatus.INCOMPLETE


def test_single_sample_window_std_is_zero(tmp_path):
    """When the window has only one sample, stdev should be 0.0 (not an error)."""
    # Make a 3-step run with expected_steps=3; window [2:7] = [step2] only
    step_times = [60.0, 61.0, 62.0]
    tokens = [26.0, 27.0, 28.0]
    _write_metrics(tmp_path, step_times, tokens)

    result = parse_rl_metrics(tmp_path, expected_steps=3)
    assert result.succeeded
    assert result.metrics.step_time_std_seconds == pytest.approx(0.0)
    assert result.metrics.tokens_per_sec_per_gpu_std == pytest.approx(0.0)
    assert result.metrics.sample_count == 1


# ---------------------------------------------------------------------------
# expected_steps_from_run_log
# ---------------------------------------------------------------------------


def test_expected_steps_from_nested_ray_driver_log(tmp_path):
    """max_num_steps is read from the config dump in a nested ray-driver.log."""
    _write_ray_driver_log(tmp_path, 20)
    assert expected_steps_from_run_log(tmp_path) == 20


def test_expected_steps_missing_ray_driver_log(tmp_path):
    assert expected_steps_from_run_log(tmp_path) is None


def test_expected_steps_log_without_config_dump(tmp_path):
    (tmp_path / "ray-driver.log").write_text("no config dump here\n")
    assert expected_steps_from_run_log(tmp_path) is None


def test_no_run_log_means_no_data(tmp_path):
    """Complete metrics but no ray-driver.log: step count is unknowable → NO_DATA."""
    _write_metrics(tmp_path, [60.0] * 10, [26.0] * 10)
    result = parse_rl_metrics(tmp_path)
    assert result.status == RLMetricsParseStatus.NO_DATA
    assert result.metrics is None


def test_step_count_mismatch_with_run_log(tmp_path):
    """ray-driver.log says 20 steps but only 10 recorded → INCOMPLETE."""
    _write_metrics(tmp_path, [60.0] * 10, [26.0] * 10)
    _write_ray_driver_log(tmp_path, 20)
    result = parse_rl_metrics(tmp_path)
    assert result.status == RLMetricsParseStatus.INCOMPLETE
    assert result.metrics is None
