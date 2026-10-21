# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pathlib
import statistics

import pytest

from llmb_run.pretrain_log_parser import (
    MAX_ITERATION,
    MIN_ITERATION,
    PretrainLogParseStatus,
    parse_latest_pretrain_job_log,
    parse_pretrain_log,
    parser_name_for_framework,
)

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures" / "pretrain_logs"
REAL_LOG_DIR = FIXTURE_DIR / "internal"

# Skip these tests when internal real-log fixtures are absent in the external release.
requires_real_logs = pytest.mark.skipif(
    not REAL_LOG_DIR.is_dir(),
    reason="real training-log fixtures are intentionally excluded from the public "
    "release; coverage is not missing — do not synthesize a fixture to unskip this test",
)


def test_parser_name_for_framework():
    assert parser_name_for_framework("nemo2") == "nemo"
    assert parser_name_for_framework("megatron_bridge") == "megatron_bridge"
    assert parser_name_for_framework("nemo") is None
    assert parser_name_for_framework("mbridge") is None
    assert parser_name_for_framework("megatron-bridge") is None
    assert parser_name_for_framework("trt-llm") is None
    assert parser_name_for_framework(None) is None


def test_parse_nemo_log_normalizes_to_seconds_and_uses_sample_stddev(tmp_path):
    log_path = tmp_path / "log-nemo_123_0.out"
    times = [1.0 + (i - MIN_ITERATION) * 0.1 for i in range(MIN_ITERATION, MAX_ITERATION + 1)]
    tflops = [800.0 + i for i in range(len(times))]
    lines = [
        f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: {time} | TFLOPS_per_GPU: {tflop}"
        for iteration, time, tflop in zip(range(MIN_ITERATION, MAX_ITERATION + 1), times, tflops, strict=True)
    ]
    lines.append("Training epoch 0, iteration 49/49 | training complete")
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "nemo2")

    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.parser == "nemo"
    assert result.max_iteration_seen == MAX_ITERATION
    assert result.metrics is not None
    assert result.metrics.time_mean_seconds == pytest.approx(statistics.mean(times))
    assert result.metrics.time_std_seconds == pytest.approx(statistics.stdev(times))
    assert result.metrics.time_sample_count == len(times)
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(tflops))
    assert result.metrics.tflops_sample_count == len(tflops)


@requires_real_logs
def test_parse_real_nemo_log_fixture():
    log_path = REAL_LOG_DIR / "nemo_nemotron4_15b_50_steps.txt"

    result = parse_pretrain_log(log_path, "nemo2")

    times = [1.753, 1.759, 1.75, 1.756, 1.755, 1.753, 1.752, 1.752, 1.751, 1.753]
    tflops = [833.3, 830.7, 834.8, 831.8, 832.5, 833.5, 833.9, 833.7, 834.4, 833.4]
    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.time_sample_count == 10
    assert result.metrics.tflops_sample_count == 10
    assert result.metrics.time_mean_seconds == pytest.approx(statistics.mean(times))
    assert result.metrics.time_std_seconds == pytest.approx(statistics.stdev(times))
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(tflops))


def test_parse_megatron_bridge_log_converts_ms_and_pairs_latest_tflops(tmp_path):
    log_path = tmp_path / "log-mbridge_123_0.out"
    elapsed_ms = [9000.0 + i * 10.0 for i in range(MAX_ITERATION - MIN_ITERATION + 1)]
    tflops = [700.0 + i for i in range(len(elapsed_ms))]
    lines = []
    for iteration, elapsed, tflop in zip(range(MIN_ITERATION, MAX_ITERATION + 1), elapsed_ms, tflops, strict=True):
        lines.append(f"Step Time : {elapsed / 1000.0:.2f}s GPU utilization: {tflop}MODEL_TFLOP/s/GPU")
        lines.append(
            f"[2026-04-27 15:26:57] iteration {iteration:8d}/      50 | "
            f"elapsed time per iteration (ms): {elapsed} | lm loss: 1.0 |"
        )
    lines.append("[2026-04-27 15:30:00] iteration       50/      50 | training complete |")
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "megatron_bridge")

    times_seconds = [value / 1000.0 for value in elapsed_ms]
    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.parser == "megatron_bridge"
    assert result.metrics is not None
    assert result.metrics.time_mean_seconds == pytest.approx(statistics.mean(times_seconds))
    assert result.metrics.time_std_seconds == pytest.approx(statistics.stdev(times_seconds))
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(tflops))


@requires_real_logs
def test_parse_real_megatron_bridge_log_fixture():
    log_path = REAL_LOG_DIR / "mbridge_llama3_8b_50_steps.txt"

    result = parse_pretrain_log(log_path, "megatron_bridge")

    elapsed_ms = [9341.6, 9287.9, 9285.1, 9218.2, 9233.6, 9313.3, 9166.7, 9144.1, 9190.0, 9245.8]
    times_seconds = [value / 1000.0 for value in elapsed_ms]
    tflops = [722.2, 726.4, 726.6, 731.8, 730.6, 724.4, 736.0, 737.8, 734.1, 729.7]
    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.time_sample_count == 10
    assert result.metrics.tflops_sample_count == 10
    assert result.metrics.time_mean_seconds == pytest.approx(statistics.mean(times_seconds))
    assert result.metrics.time_std_seconds == pytest.approx(statistics.stdev(times_seconds))
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(tflops))


def test_parse_real_megatron_bridge_slurm_log_not_treated_as_nvcre():
    # Slurm Megatron-Bridge logs emit the same "Step Time : Xs GPU utilization:
    # YMODEL_TFLOP/s/GPU" line format as nvcre pod logs, but also emit a real
    # "iteration X/Y ... elapsed time per iteration" line alongside it. Treating
    # every Step Time line as an nvcre sample (keyed off a synthetic sequential
    # counter instead of the real iteration number) double-counts timing samples
    # and desyncs the positional TFLOPS pairing, silently dropping TFLOPS/GPU for
    # later iterations even though the log has clean, correctly paired data.
    log_path = FIXTURE_DIR / "mbridge_qwen3_30b_slurm_50_steps.txt"

    result = parse_pretrain_log(log_path, "megatron_bridge")

    elapsed_ms = [8717.0, 8733.6, 8717.6, 8727.3, 8720.4, 8741.7, 8716.8, 8744.7, 8721.3, 8745.6]
    times_seconds = [value / 1000.0 for value in elapsed_ms]
    tflops = [691.9, 690.5, 691.8, 691.0, 691.6, 689.9, 691.9, 689.7, 691.5, 689.6]
    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.time_sample_count == 10
    assert result.metrics.tflops_sample_count == 10
    assert result.metrics.time_mean_seconds == pytest.approx(statistics.mean(times_seconds))
    assert result.metrics.time_std_seconds == pytest.approx(statistics.stdev(times_seconds))
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(tflops))


@requires_real_logs
def test_parse_real_megatron_bridge_missing_steps_fixture():
    log_path = REAL_LOG_DIR / "mbridge_qwen3_30b_missing_steps.txt"

    result = parse_pretrain_log(log_path, "megatron_bridge")

    assert result.status == PretrainLogParseStatus.INCOMPLETE
    assert result.max_iteration_seen == MAX_ITERATION
    assert result.final_iteration_seen is False
    assert result.metrics is None


@requires_real_logs
def test_parse_real_megatron_bridge_all_nan_fixture():
    log_path = REAL_LOG_DIR / "mbridge_llama3_70b_all-nan.txt"

    result = parse_pretrain_log(log_path, "megatron_bridge")

    assert result.status == PretrainLogParseStatus.INVALID_GRAD_NORM
    assert result.metrics is None
    assert result.invalid_grad_norm_iteration == 1


@requires_real_logs
def test_parse_real_megatron_bridge_partial_nan_fixture():
    log_path = REAL_LOG_DIR / "mbridge_llama3_70b_partial-nan.txt"

    result = parse_pretrain_log(log_path, "megatron_bridge")

    assert result.status == PretrainLogParseStatus.INVALID_GRAD_NORM
    assert result.metrics is None
    assert result.invalid_grad_norm_iteration == 33


def test_parse_megatron_bridge_log_allows_missing_tflops(tmp_path):
    log_path = tmp_path / "log-mbridge_123_0.out"
    lines = [
        "Step Time : 1.00s GPU utilization: 100.0MODEL_TFLOP/s/GPU",
        f"iteration {MIN_ITERATION}/50 | elapsed time per iteration (ms): 1000.0 |",
    ]
    for iteration in range(MIN_ITERATION + 1, MAX_ITERATION + 1):
        lines.append(f"iteration {iteration}/50 | elapsed time per iteration (ms): 2000.0 |")
    lines.append("iteration 50/50 | training complete |")
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "megatron_bridge")

    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.time_sample_count == MAX_ITERATION - MIN_ITERATION + 1
    assert result.metrics.tflops_sample_count == 1
    assert result.metrics.tflops_per_gpu_mean == 100.0
    assert result.metrics.tflops_per_gpu_std == 0.0


def test_parse_megatron_bridge_log_uses_tflops_stream_order(tmp_path):
    log_path = tmp_path / "log-mbridge_123_0.out"
    lines = [
        "Step Time : 1.00s GPU utilization: 999.0MODEL_TFLOP/s/GPU",
        f"iteration {MIN_ITERATION - 1}/50 | elapsed time per iteration (ms): 1000.0 |",
        f"iteration {MIN_ITERATION}/50 | elapsed time per iteration (ms): 2000.0 |",
        "Step Time : 2.00s GPU utilization: 100.0MODEL_TFLOP/s/GPU",
    ]
    delayed_tflops = [100.0]
    for iteration in range(MIN_ITERATION + 1, MAX_ITERATION + 1):
        tflop = 100.0 + (iteration - MIN_ITERATION)
        delayed_tflops.append(tflop)
        lines.append(f"Step Time : 2.00s GPU utilization: {tflop}MODEL_TFLOP/s/GPU")
        lines.append(f"iteration {iteration}/50 | elapsed time per iteration (ms): 2000.0 |")
    lines.append("iteration 50/50 | training complete |")
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "megatron_bridge")

    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.tflops_sample_count == MAX_ITERATION - MIN_ITERATION + 1
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(delayed_tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(delayed_tflops))


@pytest.mark.parametrize("bad_iteration", [1, 33, 50])
def test_parse_megatron_bridge_log_rejects_nan_grad_norm_any_iteration(tmp_path, bad_iteration):
    log_path = tmp_path / "log-mbridge_123_0.out"
    lines = []
    for iteration in range(1, 51):
        grad_norm = "nan" if iteration == bad_iteration else "1.000"
        lines.append("Step Time : 2.00s GPU utilization: 800.0MODEL_TFLOP/s/GPU")
        lines.append(
            f"[2026-05-02 02:06:10] iteration {iteration:8d}/      50 | "
            f"elapsed time per iteration (ms): 2000.0 | loss scale: 1.0 | grad norm: {grad_norm} |"
        )
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "megatron_bridge")

    assert result.status == PretrainLogParseStatus.INVALID_GRAD_NORM
    assert result.metrics is None
    assert result.invalid_grad_norm_iteration == bad_iteration


def test_parse_incomplete_log_returns_max_iteration_without_metrics(tmp_path):
    log_path = tmp_path / "log-nemo_123_0.out"
    log_path.write_text(
        "\n".join(
            f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: 1.0 | TFLOPS_per_GPU: 2.0"
            for iteration in range(MIN_ITERATION, MAX_ITERATION)
        )
    )

    result = parse_pretrain_log(log_path, "nemo2")

    assert result.status == PretrainLogParseStatus.INCOMPLETE
    assert result.max_iteration_seen == MAX_ITERATION - 1
    assert result.metrics is None


def test_parse_requires_final_iteration_from_denominator(tmp_path):
    log_path = tmp_path / "log-nemo_123_0.out"
    log_path.write_text(
        "\n".join(
            f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: 1.0 | TFLOPS_per_GPU: 2.0"
            for iteration in range(MIN_ITERATION, MAX_ITERATION + 1)
        )
    )

    result = parse_pretrain_log(log_path, "nemo2")

    assert result.status == PretrainLogParseStatus.INCOMPLETE
    assert result.final_iteration_seen is False
    assert result.metrics is None


def test_parse_accepts_non_default_final_iteration_denominator(tmp_path):
    log_path = tmp_path / "log-nemo_123_0.out"
    lines = [
        f"Training epoch 0, iteration {iteration}/200 | train_step_timing in s: 1.0 | TFLOPS_per_GPU: 2.0"
        for iteration in range(MIN_ITERATION, MAX_ITERATION + 1)
    ]
    lines.append("Training epoch 0, iteration 200/200 | training complete")
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "nemo2")

    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.final_iteration_seen is True
    assert result.metrics is not None


def test_parse_requires_all_perf_window_iterations(tmp_path):
    log_path = tmp_path / "log-nemo_123_0.out"
    lines = [
        f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: 1.0 | TFLOPS_per_GPU: 2.0"
        for iteration in range(MIN_ITERATION, MAX_ITERATION + 1)
        if iteration != MIN_ITERATION + 1
    ]
    lines.append("Training epoch 0, iteration 49/49 | training complete")
    log_path.write_text("\n".join(lines))

    result = parse_pretrain_log(log_path, "nemo2")

    assert result.status == PretrainLogParseStatus.INCOMPLETE
    assert result.final_iteration_seen is True
    assert result.max_iteration_seen == MAX_ITERATION
    assert result.metrics is None


def test_parse_no_data_log_returns_no_data(tmp_path):
    log_path = tmp_path / "log-nemo_123_0.out"
    log_path.write_text("started training but no timing samples yet")

    result = parse_pretrain_log(log_path, "nemo2")

    assert result.status == PretrainLogParseStatus.NO_DATA
    assert result.max_iteration_seen is None
    assert result.metrics is None


def test_parse_latest_pretrain_job_log_uses_highest_retry(tmp_path):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    _write_nemo_retry_log(log_dir / "log-account.workload_123_0.out", 1.0)
    _write_nemo_retry_log(log_dir / "log-account.workload_123_2.out", 3.0)

    result = parse_latest_pretrain_job_log(log_dir, 123, "nemo2")

    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.log_path == log_dir / "log-account.workload_123_2.out"
    assert result.metrics is not None
    assert result.metrics.time_mean_seconds == 3.0


def test_parse_latest_pretrain_job_log_handles_missing_or_unsupported_log(tmp_path):
    assert parse_latest_pretrain_job_log(tmp_path / "missing", 123, "nemo2").status == PretrainLogParseStatus.NO_LOG

    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    result = parse_latest_pretrain_job_log(log_dir, 123, "trt-llm")
    assert result.status == PretrainLogParseStatus.UNSUPPORTED_FRAMEWORK
    assert result.parser is None


def _write_nemo_retry_log(path: pathlib.Path, time_seconds: float) -> None:
    lines = [
        f"Training epoch 0, iteration {iteration}/49 | train_step_timing in s: {time_seconds} | TFLOPS_per_GPU: 2.0"
        for iteration in range(MIN_ITERATION, MAX_ITERATION + 1)
    ]
    lines.append("Training epoch 0, iteration 49/49 | training complete")
    path.write_text("\n".join(lines))


def test_parse_real_nvcre_pod_log_with_steady_state_window():
    # A genuine rank-0 pod log pulled from a Kubernetes run: "Step Time" lines
    # only (no "iteration X/Y" markers), NCCL INFO noise interleaved on stdout,
    # and a 166s first warmup step. The steady-state window (iterations
    # MIN_ITERATION..MAX_ITERATION) must measure the converged steps, never
    # the warmup.
    log_path = FIXTURE_DIR / "mbridge_nvcre_nemotron3.5_30b_50_steps.txt"

    result = parse_pretrain_log(log_path, "megatron_bridge")

    times_seconds = [15.44, 15.45, 15.39, 15.38, 15.37, 15.39, 15.35, 15.46, 15.38, 15.35]
    tflops = [956.0, 955.4, 959.1, 959.3, 960.0, 958.6, 961.5, 954.8, 959.7, 961.3]
    assert result.status == PretrainLogParseStatus.SUCCESS
    assert result.metrics is not None
    assert result.metrics.time_sample_count == 10
    assert result.metrics.tflops_sample_count == 10
    assert result.metrics.time_mean_seconds == pytest.approx(statistics.mean(times_seconds))
    assert result.metrics.time_std_seconds == pytest.approx(statistics.stdev(times_seconds))
    assert result.metrics.tflops_per_gpu_mean == pytest.approx(statistics.mean(tflops))
    assert result.metrics.tflops_per_gpu_std == pytest.approx(statistics.stdev(tflops))
