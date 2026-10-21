# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Internal pretraining log parsers for llmb-run."""

from __future__ import annotations

import pathlib
import re
import statistics
from dataclasses import dataclass
from enum import Enum

from llmb_run.job_logs import active_job_log, find_job_logs

MIN_ITERATION = 35
MAX_ITERATION = 44

_NUMBER = r"([0-9]+(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?)"
_ITERATION_RE = re.compile(r"iteration\s+(\d+)\s*/\s*(\d+)")
_NEMO_TIME_RE = re.compile(rf"train_step_timing in s:\s*{_NUMBER}")
_NEMO_TFLOPS_RE = re.compile(rf"TFLOPS_per_GPU:\s*{_NUMBER}")
_MBRIDGE_TIME_RE = re.compile(rf"elapsed time per iteration \(ms\):\s*{_NUMBER}")
_MBRIDGE_TFLOPS_RE = re.compile(rf"{_NUMBER}\s*(?:MODEL_TFLOP/s/GPU|TFLOP/s/GPU)")
_MBRIDGE_NAN_GRAD_NORM_RE = re.compile(r"\bgrad[ _]norm\s*:\s*nan\b", re.IGNORECASE)
# nvcre/Kubernetes pod logs emit a combined step-time+TFLOPS line and a
# separate iteration-count line in a different format than Slurm Megatron-Bridge.
_NVCRE_STEP_RE = re.compile(rf"Step Time\s*:\s*{_NUMBER}s\s+GPU utilization:\s*{_NUMBER}MODEL_TFLOP/s/GPU")
_NVCRE_ITER_RE = re.compile(r"\(after\s+(\d+)\s+iterations?\)")


class PretrainLogParseStatus(str, Enum):
    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    INVALID_GRAD_NORM = "invalid_grad_norm"
    NO_DATA = "no_data"
    NO_LOG = "no_log"
    UNSUPPORTED_FRAMEWORK = "unsupported_framework"


@dataclass(frozen=True)
class PretrainLogMetrics:
    """Averaged pretraining log metrics normalized for downstream use."""

    time_mean_seconds: float
    time_std_seconds: float
    time_sample_count: int
    tflops_per_gpu_mean: float | None
    tflops_per_gpu_std: float | None
    tflops_sample_count: int


@dataclass(frozen=True)
class PretrainLogParseResult:
    """Structured output from one pretraining log parse."""

    status: PretrainLogParseStatus
    parser: str | None
    framework: str
    log_path: pathlib.Path | None = None
    metrics: PretrainLogMetrics | None = None
    min_iteration: int = MIN_ITERATION
    max_iteration: int = MAX_ITERATION
    max_iteration_seen: int | None = None
    invalid_grad_norm_iteration: int | None = None
    final_iteration_seen: bool = False

    @property
    def succeeded(self) -> bool:
        return self.status == PretrainLogParseStatus.SUCCESS


def parse_latest_pretrain_job_log(
    log_dir: str | pathlib.Path,
    job_id: int,
    framework: str,
    min_iteration: int = MIN_ITERATION,
    max_iteration: int = MAX_ITERATION,
) -> PretrainLogParseResult:
    """Parse the most recent retry log for one tracked job."""

    parser = parser_name_for_framework(framework)
    if parser is None:
        return PretrainLogParseResult(
            status=PretrainLogParseStatus.UNSUPPORTED_FRAMEWORK,
            parser=None,
            framework=framework,
            min_iteration=min_iteration,
            max_iteration=max_iteration,
        )

    try:
        logs = find_job_logs(log_dir, job_id)
    except FileNotFoundError:
        logs = []

    active_log = active_job_log(logs)
    if active_log is None:
        return PretrainLogParseResult(
            status=PretrainLogParseStatus.NO_LOG,
            parser=parser,
            framework=framework,
            min_iteration=min_iteration,
            max_iteration=max_iteration,
        )

    return parse_pretrain_log(active_log.path, framework, min_iteration, max_iteration)


def parse_pretrain_log(
    log_path: str | pathlib.Path,
    framework: str,
    min_iteration: int = MIN_ITERATION,
    max_iteration: int = MAX_ITERATION,
) -> PretrainLogParseResult:
    """Parse one pretraining log using the parser selected by framework name."""

    parser = parser_name_for_framework(framework)
    path = pathlib.Path(log_path)

    if parser == "nemo":
        return _parse_nemo_log(path, framework, min_iteration, max_iteration)
    if parser == "megatron_bridge":
        return _parse_megatron_bridge_log(path, framework, min_iteration, max_iteration)

    return PretrainLogParseResult(
        status=PretrainLogParseStatus.UNSUPPORTED_FRAMEWORK,
        parser=None,
        framework=framework,
        log_path=path,
        min_iteration=min_iteration,
        max_iteration=max_iteration,
    )


def parser_name_for_framework(framework: str | None) -> str | None:
    """Return the parser family for a workload framework string."""

    if not framework:
        return None

    normalized = framework.strip().lower()
    if normalized == "nemo2":
        return "nemo"
    if normalized == "megatron_bridge":
        return "megatron_bridge"

    return None


def _parse_nemo_log(
    log_path: pathlib.Path, framework: str, min_iteration: int, max_iteration: int
) -> PretrainLogParseResult:
    times_seconds: list[float] = []
    tflops: list[float] = []
    perf_iterations_seen: set[int] = set()
    max_iteration_seen: int | None = None
    final_iteration_seen = False

    with log_path.open("r", errors="replace") as f:
        for line in f:
            iteration_marker = _iteration_marker_from_line(line)
            if iteration_marker is None:
                continue

            iteration, final_iteration = iteration_marker
            if iteration == final_iteration:
                final_iteration_seen = True

            if not min_iteration <= iteration <= max_iteration:
                continue

            time_match = _NEMO_TIME_RE.search(line)
            if not time_match:
                continue

            times_seconds.append(float(time_match.group(1)))
            perf_iterations_seen.add(iteration)
            max_iteration_seen = _max_optional(max_iteration_seen, iteration)

            tflops_match = _NEMO_TFLOPS_RE.search(line)
            if tflops_match:
                tflops.append(float(tflops_match.group(1)))

    return _build_result(
        parser="nemo",
        framework=framework,
        log_path=log_path,
        times_seconds=times_seconds,
        tflops=tflops,
        perf_iterations_seen=perf_iterations_seen,
        min_iteration=min_iteration,
        max_iteration=max_iteration,
        max_iteration_seen=max_iteration_seen,
        final_iteration_seen=final_iteration_seen,
    )


def _is_nvcre_pod_log(log_path: pathlib.Path) -> bool:
    """Detect genuine nvcre/Kubernetes pod logs vs. Slurm Megatron-Bridge logs.

    Both emit the same "Step Time : Xs GPU utilization: YMODEL_TFLOP/s/GPU" line,
    but Slurm logs also emit a standard "iteration X/Y ... elapsed time per
    iteration (ms): Z" line alongside it, while nvcre pod logs never do. Treat
    a log as nvcre only when no such standard iteration/timing line appears
    anywhere in the file, so the Step Time line isn't double-counted against the
    real iteration marker.
    """
    with log_path.open("r", errors="replace") as f:
        for line in f:
            if _iteration_marker_from_line(line) is not None and _MBRIDGE_TIME_RE.search(line):
                return False
    return True


def _parse_megatron_bridge_log(
    log_path: pathlib.Path, framework: str, min_iteration: int, max_iteration: int
) -> PretrainLogParseResult:
    # Timing and TFLOPS lines are emitted independently and the AWK in
    # common/parse_train_timing_mbridge.sh paired each iteration with the most
    # recent TFLOPS line, dropping any TFLOPS samples that arrived back-to-back.
    # Collect them as parallel lists and pair positionally so every sample counts.
    timing_samples: list[tuple[int, float]] = []
    tflops_samples: list[float] = []
    invalid_grad_norm_iteration: int | None = None
    final_iteration_seen = False
    is_nvcre = _is_nvcre_pod_log(log_path)
    # Counter for nvcre pod-log format: each "Step Time" line is one iteration.
    # Memory reporting "(after N iterations)" only appears for a few early steps
    # so we cannot rely on it for pairing; use a sequential counter instead.
    _nvcre_iteration: int = 0

    with log_path.open("r", errors="replace") as f:
        for line in f:
            # nvcre format: "Step Time : 8.39s GPU utilization: 718.8MODEL_TFLOP/s/GPU"
            nvcre_step = _NVCRE_STEP_RE.search(line) if is_nvcre else None
            if nvcre_step:
                _nvcre_iteration += 1
                time_s = float(nvcre_step.group(1))
                tflops_val = float(nvcre_step.group(2))
                timing_samples.append((_nvcre_iteration, time_s))
                tflops_samples.append(tflops_val)
                # nvcre logs have no "iteration X/X" completion marker;
                # any step data is sufficient to treat training as complete.
                final_iteration_seen = True
                continue

            tflops_match = _MBRIDGE_TFLOPS_RE.search(line)
            if tflops_match:
                tflops_samples.append(float(tflops_match.group(1)))

            iteration_marker = _iteration_marker_from_line(line)
            if iteration_marker is None:
                continue

            iteration, final_iteration = iteration_marker
            if iteration == final_iteration:
                final_iteration_seen = True

            if invalid_grad_norm_iteration is None and _MBRIDGE_NAN_GRAD_NORM_RE.search(line):
                invalid_grad_norm_iteration = iteration

            time_match = _MBRIDGE_TIME_RE.search(line)
            if not time_match:
                continue

            timing_samples.append((iteration, float(time_match.group(1)) / 1000.0))

    if invalid_grad_norm_iteration is not None:
        return PretrainLogParseResult(
            status=PretrainLogParseStatus.INVALID_GRAD_NORM,
            parser="megatron_bridge",
            framework=framework,
            log_path=log_path,
            min_iteration=min_iteration,
            max_iteration=max_iteration,
            invalid_grad_norm_iteration=invalid_grad_norm_iteration,
        )

    times_seconds: list[float] = []
    tflops: list[float] = []
    perf_iterations_seen: set[int] = set()
    max_iteration_seen: int | None = None
    for index, (iteration, time_seconds) in enumerate(timing_samples):
        if not min_iteration <= iteration <= max_iteration:
            continue

        times_seconds.append(time_seconds)
        perf_iterations_seen.add(iteration)
        max_iteration_seen = _max_optional(max_iteration_seen, iteration)
        if index < len(tflops_samples):
            tflops.append(tflops_samples[index])

    return _build_result(
        parser="megatron_bridge",
        framework=framework,
        log_path=log_path,
        times_seconds=times_seconds,
        tflops=tflops,
        perf_iterations_seen=perf_iterations_seen,
        min_iteration=min_iteration,
        max_iteration=max_iteration,
        max_iteration_seen=max_iteration_seen,
        final_iteration_seen=final_iteration_seen,
    )


def _build_result(
    parser: str,
    framework: str,
    log_path: pathlib.Path,
    times_seconds: list[float],
    tflops: list[float],
    perf_iterations_seen: set[int],
    min_iteration: int,
    max_iteration: int,
    max_iteration_seen: int | None,
    final_iteration_seen: bool,
) -> PretrainLogParseResult:
    if not times_seconds:
        return PretrainLogParseResult(
            status=PretrainLogParseStatus.NO_DATA,
            parser=parser,
            framework=framework,
            log_path=log_path,
            min_iteration=min_iteration,
            max_iteration=max_iteration,
            max_iteration_seen=max_iteration_seen,
            final_iteration_seen=final_iteration_seen,
        )

    expected_perf_iterations = set(range(min_iteration, max_iteration + 1))
    if not final_iteration_seen or not expected_perf_iterations.issubset(perf_iterations_seen):
        return PretrainLogParseResult(
            status=PretrainLogParseStatus.INCOMPLETE,
            parser=parser,
            framework=framework,
            log_path=log_path,
            min_iteration=min_iteration,
            max_iteration=max_iteration,
            max_iteration_seen=max_iteration_seen,
            final_iteration_seen=final_iteration_seen,
        )

    return PretrainLogParseResult(
        status=PretrainLogParseStatus.SUCCESS,
        parser=parser,
        framework=framework,
        log_path=log_path,
        metrics=PretrainLogMetrics(
            time_mean_seconds=statistics.mean(times_seconds),
            time_std_seconds=_stdev_or_zero(times_seconds),
            time_sample_count=len(times_seconds),
            tflops_per_gpu_mean=statistics.mean(tflops) if tflops else None,
            tflops_per_gpu_std=_stdev_or_zero(tflops) if tflops else None,
            tflops_sample_count=len(tflops),
        ),
        min_iteration=min_iteration,
        max_iteration=max_iteration,
        max_iteration_seen=max_iteration_seen,
        final_iteration_seen=final_iteration_seen,
    )


def _iteration_marker_from_line(line: str) -> tuple[int, int] | None:
    match = _ITERATION_RE.search(line)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0


def _max_optional(current: int | None, value: int) -> int:
    return value if current is None else max(current, value)
