#!/usr/bin/env python3
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

"""
Aggregate per-step artifacts from one run into a single metrics_summary.csv.

Reads ./results/<run-id>/concurrency_*/ and emits ./results/<run-id>/metrics_summary.csv,
one row per concurrency, joining together:

  - AIPerf's clean JSON aggregate (logs/aiperf/profile_export_aiperf.json)
  - The server's final /metrics snapshot (sglang_metrics.prom)
  - The periodic /metrics scrapes during the step (server_metrics_timeseries/*.prom)
  - The nvidia-smi rows captured by the bench Job (gpu_stats.csv)

GPU and time-series columns may be empty when telemetry was not collected.
The report generator accepts those runs.

The output schema is versioned via the leading `schema_version` column;
the file also starts with a `# schema_version=N` comment line so a
downstream reader can sniff the version without parsing the body.

Schema version bumps follow a semver convention:

    PATCH (e.g. 2 -> 2.1)  Reserved; we generally don't ship patch
                           bumps because the schema version is an integer.
    MINOR / "add column"   Bump to the next integer. New columns are
                           always appended; existing readers ignore
                           unknown trailing columns and continue to
                           work against a higher minor.
    MAJOR / "remove or     Bump to the next integer AND make sure
    rename a column, or    changelog + scripts/README.md +
    reshape the row model" RECIPE.yaml + README.md all advertise the
                           new column count and shape. Old consumers
                           need to be updated.

(The single-integer version stream we ship today doesn't distinguish
the three operations above visually, but the policy above documents
which kind of change motivated each bump.)

Pandas readers: the leading `# schema_version=…` line is NOT skipped
by default — `pandas.read_csv()`'s `comment` parameter defaults to
`None`. Pass `comment="#"`:

    import pandas as pd
    df = pd.read_csv("metrics_summary.csv", comment="#")

Usage:
    aggregate_metrics.py <run-dir> [--gpu-count N] [--out PATH]

Stdlib only (no pandas/plotly dependency); intentional so report.sh stays
useful on workstations without a full Python environment.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path
from typing import Iterable

SCHEMA_VERSION = "6"

# ---------------------------------------------------------------------------
# Schema. Declared up front so the CSV column order is stable and the dashboard
# can rely on it. Tuple = (column, default-when-missing).
# ---------------------------------------------------------------------------
SCHEMA: list[tuple[str, str]] = [
    ("schema_version", SCHEMA_VERSION),
    ("concurrency", ""),
    ("dirty", "0"),
    ("wall_seconds", ""),
    ("server_runtime", ""),
    ("cache_metrics_family", ""),
    # User-centric — sourced from profile_export_aiperf.json. We keep the
    # canonical p50/p90/p99 headline trio plus p10/p25/p75 so dashboard
    # box plots can render a proper quartile distribution per concurrency.
    # `request_count` = SUCCESSFUL requests reported by AIPerf.
    # `error_count` = errored requests (separate, non-overlapping tally).
    # `successful_request_count` = same as `request_count` (materialised
    # as its own column in schema v3 for downstream convenience).
    ("request_count", ""),
    ("error_count", ""),
    ("successful_request_count", ""),
    ("error_rate_pct", ""),
    ("request_latency_p10_ms", ""),
    ("request_latency_p25_ms", ""),
    ("request_latency_p50_ms", ""),
    ("request_latency_p75_ms", ""),
    ("request_latency_p90_ms", ""),
    ("request_latency_p99_ms", ""),
    ("ttft_p10_ms", ""),
    ("ttft_p25_ms", ""),
    ("ttft_p50_ms", ""),
    ("ttft_p75_ms", ""),
    ("ttft_p90_ms", ""),
    ("ttft_p99_ms", ""),
    ("itl_p10_ms", ""),
    ("itl_p25_ms", ""),
    ("itl_p50_ms", ""),
    ("itl_p75_ms", ""),
    ("itl_p90_ms", ""),
    ("itl_p99_ms", ""),
    ("output_token_throughput_tok_per_s", ""),
    # schema v6: total (prefill+decode) throughput kept as a secondary column.
    # TPS/GPU is now decode-only (see extract_aiperf); this preserves the old
    # total figure for anyone who wants prefill+decode work rate.
    ("effective_total_throughput_tok_per_s", ""),
    ("output_token_throughput_per_user_p50", ""),
    ("output_token_throughput_per_user_p90", ""),
    # Derived headline figures — what the dashboard's main scatter plots
    ("throughput_per_gpu_tok_per_s", ""),
    ("tokens_per_s_per_user_from_itl", ""),
    # SGLang KV-cache — sourced from the final sglang_metrics.prom snapshot.
    # These stay empty on vLLM/Nemotron runs; vLLM-specific cache columns below
    # carry the replacement prefix-cache/KV-usage surface.
    ("prompt_tokens_total", ""),
    ("cached_tokens_device", ""),
    ("cached_tokens_host", ""),
    ("cached_tokens_storage", ""),
    ("hit_rate_device_pct", ""),
    ("hit_rate_host_pct", ""),
    ("hit_rate_storage_pct", ""),
    ("hit_rate_overall_pct", ""),
    ("hicache_host_used_tokens", ""),
    ("hicache_host_total_tokens", ""),
    ("hicache_host_fill_pct", ""),
    # vLLM prefix-cache / KV-usage — sourced from the same final /metrics
    # snapshot when the server runtime is vLLM.
    ("vllm_prompt_tokens_total", ""),
    ("vllm_prompt_tokens_cached", ""),
    ("vllm_prefix_cache_queries", ""),
    ("vllm_prefix_cache_hits", ""),
    ("vllm_prefix_cache_hit_rate_pct", ""),
    ("vllm_kv_cache_usage_pct", ""),
    ("vllm_num_requests_running", ""),
    ("vllm_num_requests_waiting", ""),
    # KV-pressure — capacity ceilings + observed/estimated active KV.
    # `max_total_num_tokens` is the per-TP-rank GPU KV capacity (logically
    # same across ranks, so we take one rank's value). `est_active_kv_tokens`
    # is a back-of-the-envelope from AIPerf's avg ISL — useful when no
    # time-series is captured. `peak_kv_used_tokens` and
    # `peak_hicache_host_used_tokens` come from the per-step time-series and
    # stay empty for older runs.
    ("max_total_num_tokens", ""),
    ("input_sequence_length_avg", ""),
    ("isl_min", ""),
    ("isl_p25", ""),
    ("isl_p50", ""),
    ("isl_p75", ""),
    ("isl_p99", ""),
    ("isl_max", ""),
    ("est_active_kv_tokens", ""),
    ("peak_kv_used_tokens", ""),
    ("peak_hicache_host_used_tokens", ""),
    ("peak_vllm_kv_cache_usage_pct", ""),
    ("peak_vllm_num_requests_running", ""),
    ("peak_vllm_num_requests_waiting", ""),
    # GPU averages over the step — sourced from gpu_stats.csv (NEW capture)
    ("gpu_count_detected", ""),
    ("gpu_util_avg_pct", ""),
    ("gpu_util_max_pct", ""),
    ("gpu_mem_used_avg_gib", ""),
    ("gpu_mem_used_max_gib", ""),
    ("gpu_power_avg_w", ""),
    ("gpu_clock_sm_avg_mhz", ""),
    # Server pod CPU — derived from server_metrics_timeseries snapshots.
    # SGLang's /metrics surface namespaces every counter under `sglang:` and
    # does NOT include the stdlib prometheus_client `process_resident_memory_bytes`
    # exposer, so we cannot derive RSS from a /metrics scrape. The only
    # process-level counter exposed is `sglang:process_cpu_seconds_total`
    # (one series per `component` label: detokenizer + tokenizer; the
    # scheduler/main runs in a separate process and does not register a
    # counter on the /metrics endpoint). If we ever need RSS we'd have to
    # capture it via a kubectl-exec `ps`/`/proc/<pid>/status` tick the same
    # way the bench Job already captures nvidia-smi.
    ("server_cpu_avg_cores", ""),
    # Run-level lifecycle bookend — sourced from $RUN/run_meta.json
    # written by the bench Job at sweep start/end. These three columns
    # are repeated (identical) on every row of the same run; that's
    # intentional — the CSV is one-row-per-concurrency-step and the
    # row-level shape is much easier to consume than a side-channel
    # JSON file, so we denormalize. `run_wall_seconds` is the true
    # Job-level wallclock (sweep start → sweep complete, inclusive of
    # the orchestration between concurrency steps); compare with
    # `sum(wall_seconds)` to see how much of the Job lifetime was
    # NOT inside an aiperf invocation. Empty on runs from older bench
    # Jobs that predate the run_meta.json bookend.
    ("run_started_at", ""),
    ("run_completed_at", ""),
    ("run_wall_seconds", ""),
]
COLUMNS = [c for c, _ in SCHEMA]

# De-dup for the "declared vs detected GPU count" note: under the whole-node
# policy every rung would otherwise repeat the same line. Emit it once per run.
_GPU_BASIS_NOTES: set[str] = set()


# ---------------------------------------------------------------------------
# Lightweight Prometheus text-format parser. We only need a handful of named
# metrics; full label parsing for the one we care about (cached_tokens_total,
# which has cache_source={device,host,storage}).
# ---------------------------------------------------------------------------
PROM_LINE = re.compile(
    r"""^
    (?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)        # metric name
    (?:\{(?P<labels>[^}]*)\})?                 # optional labels
    \s+
    (?P<value>[+-]?[\d.eE+\-]+|NaN|\+Inf|-Inf) # value
    (?:\s+\d+)?                                 # optional timestamp (ignored)
    \s*$""",
    re.VERBOSE,
)
LABEL_PAIR = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom(path: Path) -> dict:
    """Return {metric_name: {labels_tuple: value, ...}} for the prom file.

    `labels_tuple` is a frozenset of (k, v) pairs so we can index by any
    subset. For unlabelled metrics, an empty frozenset is the key.
    """
    out: dict[str, dict[frozenset, float]] = {}
    try:
        with path.open() as f:
            for line in f:
                if not line or line.startswith("#"):
                    continue
                m = PROM_LINE.match(line.strip())
                if not m:
                    continue
                name = m.group("name")
                value_s = m.group("value")
                try:
                    value = float(value_s)
                except ValueError:
                    continue
                labels = m.group("labels") or ""
                pairs = frozenset(LABEL_PAIR.findall(labels))
                out.setdefault(name, {})[pairs] = value
    except OSError:
        pass
    return out


def prom_get(parsed: dict, name: str, **want_labels) -> float | None:
    """Lookup a single value by name, requiring all want_labels to match."""
    series = parsed.get(name)
    if not series:
        return None
    for labels_set, value in series.items():
        labels = dict(labels_set)
        if all(labels.get(k) == v for k, v in want_labels.items()):
            return value
    return None


def prom_sum(parsed: dict, name: str) -> float | None:
    """Sum over all label combinations for a metric."""
    series = parsed.get(name)
    if not series:
        return None
    return sum(series.values())


def prom_any(parsed: dict, name: str) -> float | None:
    """Return the first value for a metric, ignoring labels."""
    series = parsed.get(name)
    if not series:
        return None
    return next(iter(series.values()))


def prom_avg(parsed: dict, name: str) -> float | None:
    """Average over all label combinations for a metric."""
    series = parsed.get(name)
    if not series:
        return None
    values = list(series.values())
    return sum(values) / len(values) if values else None


def first_metric(parsed: dict, names: Iterable[str], reducer) -> float | None:
    """Return the first present metric from a compatibility-name list."""
    for name in names:
        value = reducer(parsed, name)
        if value is not None:
            return value
    return None


def metric_ratio_to_pct(value: float | None) -> float | None:
    """Normalize vLLM ratio gauges: docs expose 1.0 as 100 percent."""
    if value is None:
        return None
    if 0.0 <= value <= 1.0:
        return value * 100.0
    return value


# ---------------------------------------------------------------------------
# Per-step extractors.
# ---------------------------------------------------------------------------


def extract_aiperf(step_dir: Path) -> dict:
    """Pull what we want out of profile_export_aiperf.json."""
    p = step_dir / "logs" / "aiperf" / "profile_export_aiperf.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}

    def g(key, *path):
        node = data.get(key)
        if not isinstance(node, dict):
            return None
        for p_key in path:
            node = node.get(p_key) if isinstance(node, dict) else None
        return node

    out: dict = {}
    # Aggregate counts.
    # `request_count` is AIPerf's SUCCESSFUL request count over the step.
    # `error_count` is the errored request count (a separate, non-
    # overlapping tally). The successful count equals `request_count`
    # directly — do NOT subtract `error_count` from it.
    out["request_count"] = g("request_count", "avg")
    out["error_count"] = g("error_request_count", "avg")
    out["error_rate_pct"] = g("request_error_rate", "avg")

    # Latency-side percentiles. The p10/p25/p75 set lets the dashboard
    # render full-quartile box plots without re-parsing the per-request
    # JSONL on every render.
    for metric, prefix in (
        ("request_latency", "request_latency"),
        ("time_to_first_token", "ttft"),
        ("inter_token_latency", "itl"),
    ):
        for pct in ("p10", "p25", "p50", "p75", "p90", "p99"):
            out[f"{prefix}_{pct}_ms"] = g(metric, pct)

    # Throughputs. schema v6: TPS/GPU is now DECODE-ONLY
    # (`output_token_throughput`), per the operator's definition. Rationale:
    # total (prefill+decode) throughput = ISL×req/s, and because the fixed-
    # duration replay processes a lower-ISL request mix as concurrency rises
    # (and huge-context prefills don't complete within the window), the total
    # figure spuriously *falls* past saturation. Decode throughput rises to a
    # clean plateau — the correct capacity metric. We keep the total figure in
    # `effective_total_throughput_tok_per_s` as a secondary column.
    out["output_token_throughput_tok_per_s"] = g("output_token_throughput", "avg")
    if out["output_token_throughput_tok_per_s"] is None:
        print(
            f"WARN: `output_token_throughput.avg` absent in {p.parent.name}; "
            f"TPS/GPU (decode) will be empty for this rung.",
            file=sys.stderr,
        )
    for key in ("effective_total_throughput", "total_token_throughput"):
        v = g(key, "avg")
        if v is not None:
            out["effective_total_throughput_tok_per_s"] = v
            break
    out["output_token_throughput_per_user_p50"] = g("output_token_throughput_per_user", "p50")
    out["output_token_throughput_per_user_p90"] = g("output_token_throughput_per_user", "p90")

    # Wall seconds
    out["wall_seconds"] = g("benchmark_duration", "avg")

    # Average input sequence length, used by the KV-pressure chart as a
    # back-of-the-envelope for active KV footprint when no time-series is
    # captured (concurrency × avg_ISL ≈ peak live tokens).
    out["input_sequence_length_avg"] = g("input_sequence_length", "avg")
    for pct in ("p25", "p50", "p75", "p99"):
        out[f"isl_{pct}"] = g("input_sequence_length", pct)
    return out


def percentile(values: list[float], pct: float) -> float | None:
    """Linear-interpolated percentile over sorted per-request values."""
    if not values:
        return None
    xs = sorted(values)
    if len(xs) == 1:
        return xs[0]
    pos = (len(xs) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    frac = pos - lo
    return xs[lo] * (1.0 - frac) + xs[hi] * frac


def extract_isl_distribution(step_dir: Path) -> dict:
    """Compute per-request input-sequence-length min/max from JSONL.

    AIPerf's aggregate JSON is preferred for percentiles when it exposes
    `input_sequence_length.p25/p50/p75/p99`. The JSONL is still valuable for
    min/max, and remains a fallback for percentiles on older AIPerf builds.
    Keep the key lookup permissive so nearby builds that name the field
    slightly differently still produce the distribution.
    """
    p = step_dir / "logs" / "aiperf" / "profile_export.jsonl"
    if not p.exists():
        return {}

    values: list[float] = []

    def coerce(v) -> float | None:
        try:
            out = float(v)
        except (TypeError, ValueError):
            return None
        return out if out >= 0 else None

    def metric_value(metrics: dict, name: str):
        node = metrics.get(name)
        if isinstance(node, dict):
            return node.get("value")
        return node

    with p.open() as f:
        for line in f:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            metrics = rec.get("metrics") or {}
            metadata = rec.get("metadata") or {}
            if not isinstance(metrics, dict):
                metrics = {}
            if not isinstance(metadata, dict):
                metadata = {}
            candidates = (
                metric_value(metrics, "input_sequence_length"),
                metric_value(metrics, "input_length"),
                metric_value(metrics, "input_tokens"),
                rec.get("input_sequence_length"),
                metadata.get("input_sequence_length"),
                metadata.get("input_length"),
            )
            for candidate in candidates:
                value = coerce(candidate)
                if value is not None:
                    values.append(value)
                    break

    if not values:
        print(
            f"WARN: no per-request ISL values found in {p}; " "isl_min/isl_max will be empty.",
            file=sys.stderr,
        )
        return {}
    return {
        "isl_min": min(values),
        "isl_p25": percentile(values, 25),
        "isl_p50": percentile(values, 50),
        "isl_p75": percentile(values, 75),
        "isl_p99": percentile(values, 99),
        "isl_max": max(values),
    }


def extract_sglang_final(step_dir: Path) -> dict:
    """Pull KV-cache totals out of the final /metrics snapshot."""
    p = step_dir / "sglang_metrics.prom"
    parsed = parse_prom(p)
    if not parsed or not any(name.startswith("sglang:") for name in parsed):
        return {}

    prompt_total = prom_sum(parsed, "sglang:prompt_tokens_total")
    cached_device = prom_get(parsed, "sglang:cached_tokens_total", cache_source="device")
    cached_host = prom_get(parsed, "sglang:cached_tokens_total", cache_source="host")
    cached_storage = prom_get(parsed, "sglang:cached_tokens_total", cache_source="storage")
    hic_used = prom_get(parsed, "sglang:hicache_host_used_tokens")
    hic_total = prom_get(parsed, "sglang:hicache_host_total_tokens")
    # hicache_host_* are scoped to tp_rank="0" only by SGLang — fall back to any-label lookup.
    if hic_used is None:
        hic_used = next(iter(parsed.get("sglang:hicache_host_used_tokens", {}).values()), None)
    if hic_total is None:
        hic_total = next(iter(parsed.get("sglang:hicache_host_total_tokens", {}).values()), None)

    def rate(num):
        if num is None or not prompt_total:
            return None
        return 100.0 * num / prompt_total

    out: dict = {
        "cache_metrics_family": "sglang",
        "prompt_tokens_total": prompt_total,
        "cached_tokens_device": cached_device,
        "cached_tokens_host": cached_host,
        "cached_tokens_storage": cached_storage,
        "hit_rate_device_pct": rate(cached_device),
        "hit_rate_host_pct": rate(cached_host),
        "hit_rate_storage_pct": rate(cached_storage),
        "hicache_host_used_tokens": hic_used,
        "hicache_host_total_tokens": hic_total,
    }
    overall = sum(x or 0 for x in (cached_device, cached_host, cached_storage))
    out["hit_rate_overall_pct"] = 100.0 * overall / prompt_total if prompt_total else None
    if hic_used is not None and hic_total:
        out["hicache_host_fill_pct"] = 100.0 * hic_used / hic_total

    # GPU KV capacity ceiling. SGLang exposes `max_total_num_tokens` once per
    # TP rank with identical values; logically the GPU radix cache holds this
    # many tokens (shared across ranks since TP shards each KV block). Pick
    # any rank.
    series = parsed.get("sglang:max_total_num_tokens")
    if series:
        out["max_total_num_tokens"] = next(iter(series.values()))
    return out


def extract_vllm_final(step_dir: Path) -> dict:
    """Pull vLLM prefix-cache and scheduler gauges from the final /metrics snapshot."""
    p = step_dir / "sglang_metrics.prom"
    parsed = parse_prom(p)
    if not parsed or not any(name.startswith("vllm:") for name in parsed):
        return {}

    prompt_total = first_metric(parsed, ("vllm:prompt_tokens", "vllm:prompt_tokens_total"), prom_sum)
    prompt_cached = first_metric(parsed, ("vllm:prompt_tokens_cached",), prom_sum)
    prefix_queries = first_metric(
        parsed,
        (
            "vllm:prefix_cache_queries",
            "vllm:prefix_cache_queries_total",
            "vllm:gpu_prefix_cache_queries_total",
        ),
        prom_sum,
    )
    prefix_hits = first_metric(
        parsed,
        (
            "vllm:prefix_cache_hits",
            "vllm:prefix_cache_hits_total",
            "vllm:gpu_prefix_cache_hits_total",
        ),
        prom_sum,
    )
    hit_rate = first_metric(
        parsed,
        ("vllm:gpu_prefix_cache_hit_rate", "vllm:prefix_cache_hit_rate"),
        prom_avg,
    )
    if hit_rate is not None:
        hit_rate_pct = metric_ratio_to_pct(hit_rate)
    elif prefix_hits is not None and prefix_queries:
        hit_rate_pct = 100.0 * prefix_hits / prefix_queries
    else:
        hit_rate_pct = None

    kv_usage = first_metric(
        parsed,
        ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
        prom_avg,
    )
    running = first_metric(parsed, ("vllm:num_requests_running",), prom_avg)
    waiting = first_metric(parsed, ("vllm:num_requests_waiting",), prom_avg)

    return {
        "cache_metrics_family": "vllm",
        "vllm_prompt_tokens_total": prompt_total,
        "vllm_prompt_tokens_cached": prompt_cached,
        "vllm_prefix_cache_queries": prefix_queries,
        "vllm_prefix_cache_hits": prefix_hits,
        "vllm_prefix_cache_hit_rate_pct": hit_rate_pct,
        "vllm_kv_cache_usage_pct": metric_ratio_to_pct(kv_usage),
        "vllm_num_requests_running": running,
        "vllm_num_requests_waiting": waiting,
    }


def extract_server_timeseries(step_dir: Path) -> dict:
    """Average server-side metrics over the step's periodic snapshots."""
    ts_dir = step_dir / "server_metrics_timeseries"
    if not ts_dir.is_dir():
        return {}
    # CPU counter is exposed per-component (detokenizer + tokenizer); sum
    # across components per tick to get a single "server process group"
    # cumulative value, then rate over the step's wall window.
    cpu_seconds_samples: list[tuple[float, float]] = []
    # KV-pressure peaks. SGLang exposes kv_used_tokens (GPU active KV) and
    # hicache_host_used_tokens (host offload) per snapshot; peak across the
    # step is the most useful aggregate for the pressure chart.
    #
    # None vs 0 distinction. Both peaks start as `None`; we only assign on
    # first-observed tick. This means downstream consumers can distinguish:
    #   * peak == None   → metric never appeared in any /metrics scrape
    #                      (server build missing the gauge, or scrape file
    #                      empty) → rendered blank in the CSV.
    #   * peak == 0.0    → metric appeared, was always zero → rendered as
    #                      0.0 in the CSV (genuine cache miss / cold radix).
    peak_gpu_kv: float | None = None
    peak_host_kv: float | None = None
    peak_vllm_kv_usage_pct: float | None = None
    peak_vllm_running: float | None = None
    peak_vllm_waiting: float | None = None
    for entry in sorted(ts_dir.glob("*.prom")):
        parsed = parse_prom(entry)
        cpu_series = parsed.get("sglang:process_cpu_seconds_total")
        if cpu_series:
            try:
                ts = float(entry.stem)  # filename is unix seconds
            except ValueError:
                continue
            cpu_seconds_samples.append((ts, sum(cpu_series.values())))
        # GPU KV usage and host KV usage — both are gauges. Take any-label
        # match since SGLang only exposes them on tp_rank="0".
        gpu_kv_v = next(iter(parsed.get("sglang:kv_used_tokens", {}).values()), None)
        if gpu_kv_v is not None:
            peak_gpu_kv = gpu_kv_v if peak_gpu_kv is None else max(peak_gpu_kv, gpu_kv_v)
        host_kv_v = next(iter(parsed.get("sglang:hicache_host_used_tokens", {}).values()), None)
        if host_kv_v is not None:
            peak_host_kv = host_kv_v if peak_host_kv is None else max(peak_host_kv, host_kv_v)
        vllm_kv_v = first_metric(
            parsed,
            ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
            prom_avg,
        )
        vllm_kv_pct = metric_ratio_to_pct(vllm_kv_v)
        if vllm_kv_pct is not None:
            peak_vllm_kv_usage_pct = (
                vllm_kv_pct if peak_vllm_kv_usage_pct is None else max(peak_vllm_kv_usage_pct, vllm_kv_pct)
            )
        vllm_running = first_metric(parsed, ("vllm:num_requests_running",), prom_avg)
        if vllm_running is not None:
            peak_vllm_running = vllm_running if peak_vllm_running is None else max(peak_vllm_running, vllm_running)
        vllm_waiting = first_metric(parsed, ("vllm:num_requests_waiting",), prom_avg)
        if vllm_waiting is not None:
            peak_vllm_waiting = vllm_waiting if peak_vllm_waiting is None else max(peak_vllm_waiting, vllm_waiting)

    out: dict = {}
    if len(cpu_seconds_samples) >= 2:
        # cumulative CPU-seconds; mean rate = delta(cpu) / delta(time)
        cpu_seconds_samples.sort()
        first_t, first_cs = cpu_seconds_samples[0]
        last_t, last_cs = cpu_seconds_samples[-1]
        dt = last_t - first_t
        if dt > 0:
            out["server_cpu_avg_cores"] = (last_cs - first_cs) / dt
    if peak_gpu_kv is not None:
        out["peak_kv_used_tokens"] = peak_gpu_kv
    if peak_host_kv is not None:
        out["peak_hicache_host_used_tokens"] = peak_host_kv
    if peak_vllm_kv_usage_pct is not None:
        out["peak_vllm_kv_cache_usage_pct"] = peak_vllm_kv_usage_pct
    if peak_vllm_running is not None:
        out["peak_vllm_num_requests_running"] = peak_vllm_running
    if peak_vllm_waiting is not None:
        out["peak_vllm_num_requests_waiting"] = peak_vllm_waiting
    return out


def extract_gpu_stats(step_dir: Path) -> dict:
    """Average GPU stats over the step's gpu_stats.csv."""
    p = step_dir / "gpu_stats.csv"
    if not p.exists():
        return {}
    util: list[float] = []
    mem_used: list[float] = []  # GiB
    power: list[float] = []
    clock: list[float] = []
    gpu_indices: set[int] = set()
    try:
        with p.open() as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    gpu_indices.add(int(row["gpu_index"]))
                except (KeyError, ValueError):
                    pass
                for column, target in (
                    ("utilization_gpu_pct", util),
                    ("memory_used_mib", None),  # special handling
                    ("power_draw_w", power),
                    ("clocks_sm_mhz", clock),
                ):
                    # csv.DictReader populates short rows with None (not
                    # ""). `(row.get(...) or "").strip()` covers both.
                    v = (row.get(column) or "").strip()
                    if not v or v == "N/A":
                        continue
                    try:
                        f_v = float(v)
                    except ValueError:
                        continue
                    if column == "memory_used_mib":
                        mem_used.append(f_v / 1024)  # MiB -> GiB
                    else:
                        target.append(f_v)
    except OSError:
        return {}

    def avg(xs):
        return sum(xs) / len(xs) if xs else None

    def mx(xs):
        return max(xs) if xs else None

    return {
        "gpu_count_detected": len(gpu_indices) or None,
        "gpu_util_avg_pct": avg(util),
        "gpu_util_max_pct": mx(util),
        "gpu_mem_used_avg_gib": avg(mem_used),
        "gpu_mem_used_max_gib": mx(mem_used),
        "gpu_power_avg_w": avg(power),
        "gpu_clock_sm_avg_mhz": avg(clock),
    }


def extract_run_meta(run_dir: Path) -> dict:
    """Pull the Job-lifecycle bookend out of $RUN/run_meta.json.

    Written by the bench Job at sweep start (with `started_at_utc` +
    config) and re-templated at sweep end (with `completed_at_utc` +
    `wall_seconds_total`). Older runs that predate the bookend won't
    have the file at all; we return empty in that case so the row
    columns stay blank.
    """
    p = run_dir / "run_meta.json"
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    out: dict = {}
    if "started_at_utc" in data:
        out["run_started_at"] = data["started_at_utc"]
    if "completed_at_utc" in data:
        out["run_completed_at"] = data["completed_at_utc"]
    if "wall_seconds_total" in data:
        out["run_wall_seconds"] = data["wall_seconds_total"]
    if "server_runtime" in data:
        out["server_runtime"] = data["server_runtime"]
    if data.get("gpu_count"):
        out["gpu_count"] = data["gpu_count"]  # recipe's true GPU count → TPS/GPU denominator when DCGM is absent
    return out


# ---------------------------------------------------------------------------
# Row assembly.
# ---------------------------------------------------------------------------

CONCURRENCY_DIR_RE = re.compile(r"^concurrency_(\d+)$")


def concurrency_dirs(run_dir: Path) -> list[Path]:
    """Sorted (numerically) list of concurrency_<N> dirs under the run root.

    Strictly validates the directory name against `concurrency_<digits>`.
    A glob match like `concurrency_*` would also pick up
    `concurrency_TBD`, `concurrency_old-c4`, etc.; ignoring them
    silently would produce a misleadingly short metrics_summary.csv.
    """
    out: list[Path] = []
    for d in sorted(run_dir.glob("concurrency_*")):
        if not d.is_dir():
            continue
        m = CONCURRENCY_DIR_RE.match(d.name)
        if not m:
            print(
                f"WARN: skipping non-conforming concurrency dir: {d.name} " f"(expected `concurrency_<digits>`)",
                file=sys.stderr,
            )
            continue
        out.append(d)
    return sorted(out, key=lambda d: int(CONCURRENCY_DIR_RE.match(d.name).group(1)))


def resolve_gpu_basis(detected, gpu_count_override, run_meta_gpu_count, context: str = "") -> tuple:
    """PURE. Pick the TPS/GPU denominator + the one-line message to surface (or None).

    Returns (gpu_count, message). The denominator is the DECLARED count (override →
    run_meta's requires.gpu.count); `detected` is only a FALLBACK and otherwise a
    diagnostic. See the rationale block in build_row: under `whole_node: true` the pod
    is handed the whole node, so detected counts RESERVED GPUs, not WORKING ones —
    dividing by it deflates TPS/GPU by node_gpus/model_gpus. detected != declared is
    NORMAL there and must never fail; only a wholly-missing declared count warns.
    """
    gpu_count = gpu_count_override or run_meta_gpu_count
    if not gpu_count:
        where = f" for {context}" if context else ""
        return (detected or 8), (
            f"WARN: no --gpu-count override and no gpu_count in run_meta{where}; falling back to "
            f"{('detected ' + str(detected)) if detected else 'the 8-GPU default'} as the TPS/GPU "
            f"denominator — pass --gpu-count or re-run so the bench Job stamps "
            f"requires.gpu.count into run_meta."
        )
    if detected and str(detected) != str(gpu_count):
        return gpu_count, (
            f"note: model uses {gpu_count} GPU(s); {detected} reserved on this node "
            f"(whole_node) — TPS/GPU is per USED GPU; the reserved count is kept as a "
            f"diagnostic (gpu_count_detected), not the denominator."
        )
    return gpu_count, None


def build_row(step_dir: Path, gpu_count_override: int | None, run_meta: dict) -> dict:
    concurrency = step_dir.name.split("_", 1)[1] if "_" in step_dir.name else step_dir.name
    row: dict[str, object] = {c: default for c, default in SCHEMA}
    row["concurrency"] = concurrency
    row["dirty"] = "1" if (step_dir / "dirty.flag").exists() else "0"

    isl_dist = extract_isl_distribution(step_dir)
    aiperf = extract_aiperf(step_dir)
    sglang = extract_sglang_final(step_dir)
    vllm = extract_vllm_final(step_dir)
    server_ts = extract_server_timeseries(step_dir)
    gpu = extract_gpu_stats(step_dir)

    for source in (isl_dist, aiperf, sglang, vllm, server_ts, gpu, run_meta):
        for k, v in source.items():
            if v is not None:
                row[k] = v

    if not row.get("cache_metrics_family"):
        runtime = str(row.get("server_runtime") or "").lower()
        if runtime in ("vllm", "sglang"):
            row["cache_metrics_family"] = runtime
    if not row.get("server_runtime") and row.get("cache_metrics_family"):
        row["server_runtime"] = row["cache_metrics_family"]

    # AIPerf reports `request_count` and `error_request_count` as
    # SEPARATE, non-overlapping tallies (successful vs errored).
    # `successful_request_count` is therefore `request_count` itself,
    # NOT `request_count - error_count` (subtracting would double-
    # count the errors — e.g. 39 ok + 1 err would read as 38).
    req = row.get("request_count")
    if isinstance(req, (int, float)):
        row["successful_request_count"] = req

    # Derived headline figures. tokens/s/GPU (decode) is a PERFORMANCE metric:
    # throughput per accelerator DOING WORK. So the denominator is the DECLARED
    # count — the caller override, else the recipe's requires.gpu.count stamped
    # into run_meta — NOT the count DETECTED on the node.
    #
    # Why declared wins (whole-node policy): with `requires.gpu.whole_node: true`
    # the pod is handed the ENTIRE node, so DCGM enumerates every GPU on it (e.g.
    # 4 on a GB300) even when the model is tp=1 and uses ONE. Preferring detected
    # there silently deflated TPS/GPU by node_gpus/model_gpus (a tp=1 cell on a
    # 4-GPU node read 4x low -> pareto_geomean halved, sqrt(4)=2.0) and would
    # break comparability with every existing cell, the GB300<->B200 cross-
    # hardware pair, and external 1-GPU references. Node exclusivity is an
    # ISOLATION policy, not a redefinition of the metric; reserved-but-idle GPUs
    # are an OCCUPANCY COST belonging in gpu_hours accounting, never folded into
    # the headline. detected != declared is therefore NORMAL under whole_node and
    # must NOT fail the run — it is kept as a diagnostic + noted once below.
    #
    # Fail loudly ONLY when no declared count exists at all: without it TPS/GPU
    # is silently mis-scaled (the last-resort default of 8 is WRONG on any node
    # that isn't 8-GPU — DCGM-less clusters hit exactly that).
    gpu_count, _msg = resolve_gpu_basis(
        row.get("gpu_count_detected"),
        gpu_count_override,
        run_meta.get("gpu_count"),
        context=str(row.get("concurrency")),
    )
    if _msg and _msg not in _GPU_BASIS_NOTES:
        _GPU_BASIS_NOTES.add(_msg)
        print(_msg, file=sys.stderr)
    if isinstance(gpu_count, str) and gpu_count.isdigit():
        gpu_count = int(gpu_count)
    throughput = row.get("output_token_throughput_tok_per_s")
    if isinstance(throughput, (int, float)) and isinstance(gpu_count, (int, float)) and gpu_count > 0:
        row["throughput_per_gpu_tok_per_s"] = throughput / gpu_count

    itl_p50 = row.get("itl_p50_ms")
    if isinstance(itl_p50, (int, float)) and itl_p50 > 0:
        row["tokens_per_s_per_user_from_itl"] = 1000.0 / itl_p50

    # Estimated active KV footprint at peak. Concurrency × avg input
    # sequence length is a back-of-the-envelope for how many tokens the
    # KV cache had to hold concurrently. Used by the KV-pressure chart
    # when no time-series is available (older runs); the chart prefers
    # the observed peak_kv_used_tokens + peak_hicache_host_used_tokens
    # when both are present.
    try:
        conc_num = int(concurrency)
    except ValueError:
        conc_num = None
    isl = row.get("input_sequence_length_avg")
    if isinstance(isl, (int, float)) and conc_num:
        row["est_active_kv_tokens"] = conc_num * isl

    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir", help="Path to results/<run-id>/")
    ap.add_argument(
        "--gpu-count",
        type=int,
        default=None,
        help="GPU count to use when computing tokens/s/GPU if gpu_stats.csv is absent. Default: 4.",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Output CSV path. Default: <run-dir>/metrics_summary.csv",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    if not run_dir.is_dir():
        print(f"ERROR: run dir not found: {run_dir}", file=sys.stderr)
        return 1

    dirs = concurrency_dirs(run_dir)
    if not dirs:
        print(f"ERROR: no concurrency_* subdirectories under {run_dir}", file=sys.stderr)
        return 1

    out_path = Path(args.out) if args.out else run_dir / "metrics_summary.csv"
    run_meta = extract_run_meta(run_dir)
    rows = [build_row(d, args.gpu_count, run_meta) for d in dirs]

    with out_path.open("w", newline="") as f:
        # Schema-version comment line at the top so downstream consumers can
        # cheaply detect format drift. Most CSV readers skip lines starting
        # with '#'; the dashboard does that explicitly. NOTE: pandas does
        # NOT skip '#' lines by default — `comment` defaults to None — so
        # pandas readers must pass `comment="#"` (see module docstring).
        f.write(f"# schema_version={SCHEMA_VERSION}\n")
        # extrasaction="ignore": run_meta carries non-column helpers (e.g. gpu_count, used only for the
        # TPS/GPU denominator) that must NOT be written as columns.
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            # Floats with a sensible default precision; keep ints intact.
            formatted = {}
            for k, v in r.items():
                if isinstance(v, float):
                    formatted[k] = f"{v:.6f}".rstrip("0").rstrip(".")
                else:
                    formatted[k] = v
            w.writerow(formatted)

    print(f"Wrote {out_path} ({len(rows)} rows, {len(COLUMNS)} cols, schema v{SCHEMA_VERSION})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
