# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""Build the `aiperf profile` arguments for one concurrency rung.

The builder applies the recipe workload source, concurrency, warmup, request limits,
and timeout settings without invoking AIPerf.
"""

from __future__ import annotations


def _pos_int(v) -> int:
    """Parse to int; non-numeric / empty → 0 (mirrors the shell's `[ "$X" -gt 0 ] 2>/dev/null` guards)."""
    try:
        return int(str(v).strip())
    except (ValueError, AttributeError):
        return 0


# Environment variable to argument-builder field.
_ENV_MAP = {
    "SERVER_URL": "url",
    "AIPERF_MODEL_ID": "model",
    "AIPERF_TOKENIZER_ID": "tokenizer",
    "AIPERF_ENDPOINT_TYPE": "endpoint_type",
    "DATASET_TYPE": "dataset_type",
    "DATASET_PATH": "dataset_path",
    "NUM_PROFILE_RUNS": "num_profile_runs",
    "OUT": "out",
    "CACHE_BUST": "cache_bust",
    "AIPERF_SCENARIO": "scenario",
    "NUM_DATASET_ENTRIES": "num_dataset_entries",
    # Allow request-count-driven concurrency sweeps for timestamped traces.
    "NO_FIXED_SCHEDULE": "no_fixed_schedule",
    "WARMUP_REQUEST_MULTIPLIER": "warmup_request_multiplier",
    "REQUEST_COUNT_MULTIPLIER": "request_count_multiplier",
    "BENCH_DURATION": "bench_duration",
    "CONCURRENCY_RAMP_DURATION": "concurrency_ramp_duration",
    "BENCHMARK_GRACE_PERIOD": "benchmark_grace_period",
    "REQUEST_TIMEOUT_SECONDS": "request_timeout_seconds",
    "AIPERF_EXTRA_INPUTS": "aiperf_extra_inputs",
    # Synthetic mode generates prompts without an input file.
    "SYN_ISL": "syn_isl",
    "SYN_ISL_STDDEV": "syn_isl_stddev",
    "SYN_OSL": "syn_osl",
    "SYN_OSL_STDDEV": "syn_osl_stddev",
    "SYN_SEED": "syn_seed",
}


def cfg_from_env(env: dict) -> dict:
    """Map the bench-job env vars to a cfg dict (pure; pass os.environ)."""
    return {key: env[var] for var, key in _ENV_MAP.items() if env.get(var) not in (None, "")}


def build_aiperf_args(cfg: dict, n: int) -> list[str]:
    """Build `aiperf profile` arguments for one concurrency.

    Trace workloads use an input file; synthetic workloads use ISL/OSL parameters.
    """
    g = cfg.get
    synthetic = bool(g("syn_isl")) and bool(g("syn_osl")) and not g("dataset_path")
    args = [
        "profile",
        "--url",
        str(g("url", "")),
        "--model",
        str(g("model", "")),
        "--tokenizer",
        str(g("tokenizer", "")),
        "--use-server-token-count",
        "--endpoint-type",
        str(g("endpoint_type", "")),
        "--streaming",
    ]
    if synthetic:
        args += ["--isl", str(g("syn_isl")), "--osl", str(g("syn_osl"))]
        if _pos_int(g("syn_isl_stddev")) > 0:
            args += ["--isl-stddev", str(g("syn_isl_stddev"))]
        if _pos_int(g("syn_osl_stddev")) > 0:
            args += ["--osl-stddev", str(g("syn_osl_stddev"))]
        if str(g("syn_seed", "")).strip() != "":
            args += ["--random-seed", str(g("syn_seed"))]
    else:
        args += [
            "--custom-dataset-type",
            str(g("dataset_type", "")),
            "--input-file",
            str(g("dataset_path", "")),
        ]
        # Disable trace scheduling when the recipe requests a concurrency-driven sweep.
        if g("no_fixed_schedule"):
            args += ["--no-fixed-schedule"]
    args += [
        "--concurrency",
        str(n),
        "--num-profile-runs",
        str(g("num_profile_runs", 1)),
        "--unsafe-override",
        "--ui",
        "simple",
        "--artifact-dir",
        f"{g('out', '')}/logs/aiperf",
    ]
    if g("cache_bust"):
        args += ["--cache-bust", str(g("cache_bust"))]
    if g("scenario"):
        args += ["--scenario", str(g("scenario"))]
    if g("num_dataset_entries"):
        args += ["--num-dataset-entries", str(g("num_dataset_entries"))]
    warmup = _pos_int(g("warmup_request_multiplier"))
    if warmup > 0:
        args += ["--warmup-request-count", str(warmup * n)]
    # Workload cap priority: request count, duration, then full-trace replay.
    req_mult = _pos_int(g("request_count_multiplier"))
    duration = _pos_int(g("bench_duration"))
    if req_mult > 0:
        args += ["--request-count", str(req_mult * n)]
    elif duration > 0:
        args += ["--benchmark-duration", str(duration)]
        if g("concurrency_ramp_duration"):
            args += ["--concurrency-ramp-duration", str(g("concurrency_ramp_duration"))]
        if g("benchmark_grace_period"):
            args += ["--benchmark-grace-period", str(g("benchmark_grace_period"))]
    # Full-trace replay needs no cap argument.
    if g("request_timeout_seconds"):
        args += ["--request-timeout-seconds", str(g("request_timeout_seconds"))]
    if g("aiperf_extra_inputs"):
        args += ["--extra-inputs", str(g("aiperf_extra_inputs"))]
    return args
