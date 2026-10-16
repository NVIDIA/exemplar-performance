import json

import pytest

from llmb_run.rl_metrics_parser import (
    RLMetricsParseStatus,
    expected_steps_from_run_log,
    parse_rl_metrics,
)

S = RLMetricsParseStatus
STEP = "timing/train/total_step_time"
TOK = "performance/tokens_per_sec_per_gpu"


def write_metrics(d, n, step=None, tok=None):
    data = {
        STEP: {str(i): (step(i) if step else float(i)) for i in range(1, n + 1)},
        TOK: {str(i): (tok(i) if tok else 100.0 * i) for i in range(1, n + 1)},
    }
    (d / "metrics.json").write_text(json.dumps(data))


def test_missing_and_corrupt_metrics_file(tmp_path):
    assert parse_rl_metrics(tmp_path, expected_steps=5).status == S.NO_METRICS_FILE
    (tmp_path / "metrics.json").write_text("{not json")
    assert parse_rl_metrics(tmp_path, expected_steps=5).status == S.NO_METRICS_FILE


def test_success_averages_fixed_window_positions_3_to_7(tmp_path):
    write_metrics(tmp_path, 10)
    result = parse_rl_metrics(tmp_path, expected_steps=10)
    assert result.succeeded
    m = result.metrics
    assert m.sample_count == 5
    assert m.step_time_mean_seconds == pytest.approx(5.0)  # steps 3..7
    assert m.tokens_per_sec_per_gpu_mean == pytest.approx(500.0)
    assert m.step_time_std_seconds == pytest.approx(1.5811, rel=1e-3)


def test_step_count_mismatch_is_incomplete(tmp_path):
    write_metrics(tmp_path, 8)
    assert parse_rl_metrics(tmp_path, expected_steps=10).status == S.INCOMPLETE
    assert parse_rl_metrics(tmp_path, expected_steps=7).status == S.INCOMPLETE


def test_missing_or_empty_keys_is_no_data(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({STEP: {"1": 1.0}}))
    assert parse_rl_metrics(tmp_path, expected_steps=1).status == S.NO_DATA
    (tmp_path / "metrics.json").write_text(json.dumps({STEP: {}, TOK: {}}))
    assert parse_rl_metrics(tmp_path, expected_steps=1).status == S.NO_DATA


def test_run_shorter_than_window_is_no_data(tmp_path):
    write_metrics(tmp_path, 2)
    assert parse_rl_metrics(tmp_path, expected_steps=2).status == S.NO_DATA


def test_expected_steps_read_from_ray_driver_log(tmp_path):
    write_metrics(tmp_path, 6)
    log = tmp_path / "sub" / "ray-driver.log"
    log.parent.mkdir()
    log.write_text("Final config:\n{'grpo': {'max_num_steps': 6, 'x': 1}}\n")
    assert expected_steps_from_run_log(tmp_path) == 6
    assert parse_rl_metrics(tmp_path).succeeded


def test_no_steps_source_is_no_data(tmp_path):
    write_metrics(tmp_path, 6)
    assert expected_steps_from_run_log(tmp_path) is None
    assert parse_rl_metrics(tmp_path).status == S.NO_DATA


def test_expected_steps_accepts_double_quoted_json_style(tmp_path):
    (tmp_path / "ray-driver.log").write_text('"max_num_steps": 12\n')
    assert expected_steps_from_run_log(tmp_path) == 12
