# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RL (NeMo-RL / GRPO) metrics parser for llmb-run job history."""

from __future__ import annotations

import json
import logging
import pathlib
import re
import statistics
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

_STEP_TIME_KEY = "timing/train/total_step_time"
_TOKENS_PER_SEC_KEY = "performance/tokens_per_sec_per_gpu"

# The trainer prints its resolved config ("Final config:" block) into
# ray-driver.log; grpo.max_num_steps appears there as a dict entry.
_RAY_DRIVER_LOG_NAME = "ray-driver.log"
_MAX_NUM_STEPS_RE = re.compile(r"['\"]max_num_steps['\"]\s*:\s*(\d+)")

# Fixed averaging window: positions 2–6 inclusive = README "iterations 3–7"
# (1-indexed). Kept fixed regardless of MAX_STEPS so runs compare apples-to-apples.
_WINDOW_START = 2
_WINDOW_END = 7  # exclusive (slice)


class RLMetricsParseStatus(str, Enum):
    SUCCESS = "success"
    INCOMPLETE = "incomplete"
    NO_DATA = "no_data"
    NO_METRICS_FILE = "no_metrics_file"


@dataclass(frozen=True)
class RLMetrics:
    step_time_mean_seconds: float
    step_time_std_seconds: float
    tokens_per_sec_per_gpu_mean: float
    tokens_per_sec_per_gpu_std: float
    sample_count: int


@dataclass(frozen=True)
class RLMetricsParseResult:
    status: RLMetricsParseStatus
    metrics_path: pathlib.Path | None = None
    metrics: RLMetrics | None = None

    @property
    def succeeded(self) -> bool:
        return self.status == RLMetricsParseStatus.SUCCESS


def parse_rl_metrics(
    log_dir: str | pathlib.Path,
    expected_steps: int | None = None,
) -> RLMetricsParseResult:
    """Parse metrics.json from an RL experiment directory. Never raises.

    The expected step count is read from the run's own logs (the resolved
    config in ray-driver.log) unless supplied by the caller.
    """
    metrics_path = pathlib.Path(log_dir) / "metrics.json"

    try:
        with metrics_path.open("r") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(f"RL metrics file not readable at {metrics_path}: {e}")
        return RLMetricsParseResult(status=RLMetricsParseStatus.NO_METRICS_FILE, metrics_path=metrics_path)

    for key in (_STEP_TIME_KEY, _TOKENS_PER_SEC_KEY):
        if not data.get(key):
            logger.debug(f"Required RL metric key '{key}' missing or empty in {metrics_path}")
            return RLMetricsParseResult(status=RLMetricsParseStatus.NO_DATA, metrics_path=metrics_path)

    if expected_steps is None:
        expected_steps = expected_steps_from_run_log(log_dir)
    if expected_steps is None:
        logger.debug(f"No max_num_steps found in run logs under {log_dir}")
        return RLMetricsParseResult(status=RLMetricsParseStatus.NO_DATA, metrics_path=metrics_path)

    for key in (_STEP_TIME_KEY, _TOKENS_PER_SEC_KEY):
        if len(data[key]) != expected_steps:
            logger.debug(
                f"RL metric '{key}' has {len(data[key])} steps, expected exactly {expected_steps} in {metrics_path}"
            )
            return RLMetricsParseResult(status=RLMetricsParseStatus.INCOMPLETE, metrics_path=metrics_path)

    step_time_window = list(data[_STEP_TIME_KEY].values())[_WINDOW_START:_WINDOW_END]
    tokens_window = list(data[_TOKENS_PER_SEC_KEY].values())[_WINDOW_START:_WINDOW_END]

    if not step_time_window:
        # Runs shorter than the averaging window (MAX_STEPS <= 2) have no
        # benchmark-comparable iterations to average.
        logger.debug(
            f"RL run has only {expected_steps} steps — averaging window "
            f"(iterations {_WINDOW_START + 1}-{_WINDOW_END}) is empty for {metrics_path}"
        )
        return RLMetricsParseResult(status=RLMetricsParseStatus.NO_DATA, metrics_path=metrics_path)

    return RLMetricsParseResult(
        status=RLMetricsParseStatus.SUCCESS,
        metrics_path=metrics_path,
        metrics=RLMetrics(
            step_time_mean_seconds=statistics.mean(step_time_window),
            step_time_std_seconds=_stdev_or_zero(step_time_window),
            tokens_per_sec_per_gpu_mean=statistics.mean(tokens_window),
            tokens_per_sec_per_gpu_std=_stdev_or_zero(tokens_window),
            sample_count=len(step_time_window),
        ),
    )


def expected_steps_from_run_log(log_dir: str | pathlib.Path) -> int | None:
    """Read grpo.max_num_steps from the run's ray-driver.log config dump.

    This is the step count the job actually ran with, regardless of how
    MAX_STEPS was set (ambient environment variable, --env override, or the
    launch script default).
    Returns None when no ray-driver.log with a config dump is found.
    """
    try:
        candidates = sorted(pathlib.Path(log_dir).rglob(_RAY_DRIVER_LOG_NAME))
    except OSError as e:
        logger.debug(f"Unable to scan {log_dir} for {_RAY_DRIVER_LOG_NAME}: {e}")
        return None

    for path in candidates:
        try:
            with path.open("r", errors="replace") as f:
                for line in f:
                    match = _MAX_NUM_STEPS_RE.search(line)
                    if match:
                        return int(match.group(1))
        except OSError as e:
            logger.debug(f"Unable to read {path}: {e}")
    return None


def _stdev_or_zero(values: list[float]) -> float:
    return statistics.stdev(values) if len(values) > 1 else 0.0
