import pytest

from llmb_run.pretrain_log_parser import (
    MAX_ITERATION,
    MIN_ITERATION,
    PretrainLogParseStatus,
    parse_latest_pretrain_job_log,
    parse_pretrain_log,
    parser_name_for_framework,
)

Status = PretrainLogParseStatus
FINAL = 50


def mbridge_lines(iterations, ms=1000.0, tflops=500.0, extra=''):
    lines = []
    for it in iterations:
        lines.append(f" iteration {it}/ {FINAL} | elapsed time per iteration (ms): {ms + it} | {extra}")
        lines.append(f" iteration {it}/ {FINAL} | throughput {tflops + it} TFLOP/s/GPU")
    return lines


def nemo_lines(iterations, secs=2.0, tflops=400.0, with_tflops=True):
    lines = []
    for it in iterations:
        tail = f" TFLOPS_per_GPU: {tflops + it}" if with_tflops else ''
        lines.append(f"Training epoch 0, iteration {it}/{FINAL} | train_step_timing in s: {secs}{tail}")
    return lines


def write(path, lines):
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.mark.parametrize(
    "framework, expected",
    [
        ("  NeMo2 ", "nemo"),
        ("megatron_bridge", "megatron_bridge"),
        ("nemo", None),
        (None, None),
    ],
)
def test_parser_name_for_framework(framework, expected):
    assert parser_name_for_framework(framework) == expected


def test_megatron_bridge_success_converts_ms_to_seconds(tmp_path):
    log = write(tmp_path / "a.out", mbridge_lines(range(1, FINAL + 1), ms=1000.0, tflops=500.0))
    result = parse_pretrain_log(log, "megatron_bridge")

    assert result.status == Status.SUCCESS and result.succeeded
    assert result.parser == "megatron_bridge"
    m = result.metrics
    assert m.time_sample_count == 10
    # iterations 35..44 -> (1000+it)/1000 s, mean it = 39.5
    assert m.time_mean_seconds == pytest.approx(1.0395)
    assert m.time_std_seconds > 0
    assert m.tflops_sample_count == 10
    assert m.tflops_per_gpu_mean == pytest.approx(539.5)
    assert result.max_iteration_seen == 44
    assert result.final_iteration_seen is True


def test_megatron_bridge_accepts_model_tflops_spelling(tmp_path):
    lines = [
        f"iteration {it}/{FINAL} | elapsed time per iteration (ms): 2000.0 | 300.0 MODEL_TFLOP/s/GPU"
        for it in range(1, FINAL + 1)
    ]
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge")
    assert result.succeeded
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(300.0)
    assert result.metrics.time_mean_seconds == pytest.approx(2.0)


def test_megatron_bridge_without_tflops_reports_none(tmp_path):
    lines = [f"iteration {it}/{FINAL} | elapsed time per iteration (ms): 1000.0" for it in range(1, FINAL + 1)]
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge")
    assert result.succeeded
    assert result.metrics.tflops_per_gpu_mean is None
    assert result.metrics.tflops_per_gpu_std is None
    assert result.metrics.tflops_sample_count == 0


def test_megatron_bridge_nan_grad_norm_is_invalid_even_if_otherwise_complete(tmp_path):
    lines = mbridge_lines(range(1, FINAL + 1))
    lines[20] = " iteration 11/ 50 | elapsed time per iteration (ms): 1000.0 | grad norm: NaN |"
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge")
    assert result.status == Status.INVALID_GRAD_NORM
    assert result.invalid_grad_norm_iteration == 11
    assert result.metrics is None
    assert not result.succeeded


def test_megatron_bridge_reports_first_nan_grad_norm(tmp_path):
    lines = [
        "iteration 3/50 | grad_norm: nan",
        "iteration 7/50 | grad norm : nan",
    ]
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge")
    assert result.invalid_grad_norm_iteration == 3


def test_megatron_bridge_nan_grad_norm_does_not_match_finite_value(tmp_path):
    lines = mbridge_lines(range(1, FINAL + 1), extra='grad norm: 1.5 |')
    assert parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge").succeeded


def test_megatron_bridge_missing_final_iteration_is_incomplete(tmp_path):
    log = write(tmp_path / "a.out", mbridge_lines(range(1, FINAL)))
    result = parse_pretrain_log(log, "megatron_bridge")
    assert result.status == Status.INCOMPLETE
    assert result.final_iteration_seen is False
    assert result.max_iteration_seen == 44
    assert result.metrics is None


def test_megatron_bridge_missing_window_iteration_is_incomplete(tmp_path):
    its = [i for i in range(1, FINAL + 1) if i != 40]
    result = parse_pretrain_log(write(tmp_path / "a.out", mbridge_lines(its)), "megatron_bridge")
    assert result.status == Status.INCOMPLETE
    assert result.final_iteration_seen is True


def test_megatron_bridge_timing_outside_window_is_no_data(tmp_path):
    lines = [f"iteration {it}/{FINAL} | elapsed time per iteration (ms): 1000.0" for it in (1, 2, 3, 50)]
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge")
    assert result.status == Status.NO_DATA
    assert result.max_iteration_seen is None
    assert result.final_iteration_seen is True


def test_megatron_bridge_iterations_outside_window_are_excluded_from_mean(tmp_path):
    lines = []
    for it in range(1, FINAL + 1):
        ms = 1000.0 if MIN_ITERATION <= it <= MAX_ITERATION else 99999.0
        lines.append(f"iteration {it}/{FINAL} | elapsed time per iteration (ms): {ms}")
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "megatron_bridge")
    assert result.metrics.time_mean_seconds == pytest.approx(1.0)
    assert result.metrics.time_std_seconds == 0.0


def test_megatron_bridge_custom_window(tmp_path):
    log = write(tmp_path / "a.out", mbridge_lines(range(1, FINAL + 1)))
    result = parse_pretrain_log(log, "megatron_bridge", min_iteration=10, max_iteration=12)
    assert result.succeeded
    assert result.metrics.time_sample_count == 3
    assert (result.min_iteration, result.max_iteration) == (10, 12)


def test_nemo_success(tmp_path):
    log = write(tmp_path / "a.out", nemo_lines(range(1, FINAL + 1), secs=2.5, tflops=400.0))
    result = parse_pretrain_log(log, "nemo2")
    assert result.succeeded and result.parser == "nemo"
    assert result.metrics.time_mean_seconds == pytest.approx(2.5)  # seconds, not converted
    assert result.metrics.time_sample_count == 10
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(439.5)
    assert result.metrics.tflops_sample_count == 10


def test_nemo_without_tflops(tmp_path):
    log = write(tmp_path / "a.out", nemo_lines(range(1, FINAL + 1), with_tflops=False))
    result = parse_pretrain_log(log, "nemo2")
    assert result.succeeded
    assert result.metrics.tflops_per_gpu_mean is None


def test_nemo_scientific_notation_time(tmp_path):
    lines = [f"iteration {it}/{FINAL} | train_step_timing in s: 2.5e-1" for it in range(1, FINAL + 1)]
    result = parse_pretrain_log(write(tmp_path / "a.out", lines), "nemo2")
    assert result.metrics.time_mean_seconds == pytest.approx(0.25)


def test_nemo_does_not_flag_nan_grad_norm(tmp_path):
    # Only the megatron_bridge parser checks grad norm.
    lines = nemo_lines(range(1, FINAL + 1))
    lines[0] += " grad_norm: nan"
    assert parse_pretrain_log(write(tmp_path / "a.out", lines), "nemo2").succeeded


def test_nemo_incomplete_without_final_iteration(tmp_path):
    result = parse_pretrain_log(write(tmp_path / "a.out", nemo_lines(range(1, 45))), "nemo2")
    assert result.status == Status.INCOMPLETE


def test_nemo_timing_lines_without_iteration_marker_are_ignored(tmp_path):
    lines = ["train_step_timing in s: 3.0"] * 20
    assert parse_pretrain_log(write(tmp_path / "a.out", lines), "nemo2").status == Status.NO_DATA


def test_empty_log_is_no_data(tmp_path):
    log = tmp_path / "a.out"
    log.write_text("")
    for framework in ("nemo2", "megatron_bridge"):
        assert parse_pretrain_log(log, framework).status == Status.NO_DATA


def test_invalid_utf8_bytes_do_not_crash_parser(tmp_path):
    log = tmp_path / "a.out"
    log.write_bytes(b"\xff\xfe garbage\n" + "\n".join(nemo_lines(range(1, FINAL + 1))).encode())
    assert parse_pretrain_log(log, "nemo2").succeeded


def test_unsupported_framework_via_parse_pretrain_log(tmp_path):
    result = parse_pretrain_log(tmp_path / "missing.out", "jax")
    assert result.status == Status.UNSUPPORTED_FRAMEWORK
    assert result.parser is None
    assert result.log_path == tmp_path / "missing.out"


def test_latest_job_log_picks_highest_retry(tmp_path):
    write(tmp_path / "log-wl_77_1.out", mbridge_lines(range(1, FINAL + 1), ms=9000.0))
    write(tmp_path / "log-wl_77_2.out", mbridge_lines(range(1, FINAL + 1), ms=1000.0))
    write(tmp_path / "log-wl_78_3.out", mbridge_lines(range(1, FINAL + 1), ms=5000.0))  # other job
    result = parse_latest_pretrain_job_log(tmp_path, 77, "megatron_bridge")
    assert result.succeeded
    assert result.log_path == tmp_path / "log-wl_77_2.out"
    assert result.metrics.time_mean_seconds == pytest.approx(1.0395)


def test_latest_job_log_retry_ordering_is_numeric(tmp_path):
    write(tmp_path / "log-wl_5_2.out", nemo_lines(range(1, FINAL + 1)))
    write(tmp_path / "log-wl_5_10.out", nemo_lines(range(1, FINAL + 1)))
    result = parse_latest_pretrain_job_log(tmp_path, 5, "nemo2")
    assert result.log_path.name == "log-wl_5_10.out"


def test_latest_job_log_job_id_is_not_substring_matched(tmp_path):
    write(tmp_path / "log-wl_123_0.out", nemo_lines(range(1, FINAL + 1)))
    assert parse_latest_pretrain_job_log(tmp_path, 23, "nemo2").status == Status.NO_LOG


def test_latest_job_log_no_log_when_directory_empty(tmp_path):
    result = parse_latest_pretrain_job_log(tmp_path, 1, "nemo2")
    assert result.status == Status.NO_LOG
    assert result.parser == "nemo"


def test_latest_job_log_no_log_when_directory_missing(tmp_path):
    assert parse_latest_pretrain_job_log(tmp_path / "nope", 1, "nemo2").status == Status.NO_LOG


def test_latest_job_log_ignores_plain_slurm_stdout(tmp_path):
    # parse_latest_pretrain_job_log only considers retry-numbered workload logs.
    write(tmp_path / "slurm-9.out", nemo_lines(range(1, FINAL + 1)))
    assert parse_latest_pretrain_job_log(tmp_path, 9, "nemo2").status == Status.NO_LOG


def test_latest_job_log_unsupported_framework_checked_before_log_lookup(tmp_path):
    result = parse_latest_pretrain_job_log(tmp_path / "nope", 1, "jax")
    assert result.status == Status.UNSUPPORTED_FRAMEWORK
    assert result.log_path is None
