# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""vLLM inference server metrics collection via Prometheus /metrics endpoint."""

import time
import urllib.error
import urllib.request

# Map vLLM Prometheus metric names to short keys.
# Only these metrics are extracted — everything else is ignored.
_METRIC_MAP = {
    # Gauges (sampled per interval)
    "vllm:kv_cache_usage_perc": "kv_cache_pct",
    "vllm:num_requests_running": "requests_running",
    "vllm:num_requests_waiting": "requests_waiting",
    # Counters (raw cumulative — rates computed from deltas)
    "vllm:prompt_tokens_total": "prompt_tokens_total",
    "vllm:generation_tokens_total": "generation_tokens_total",
    "vllm:prefix_cache_hits_total": "prefix_cache_hits_total",
    "vllm:prefix_cache_queries_total": "prefix_cache_queries_total",
    "vllm:num_preemptions_total": "preemptions_total",
    # Histogram _sum/_count (for mean latency from before/after deltas)
    "vllm:time_to_first_token_seconds_sum": "ttft_sum",
    "vllm:time_to_first_token_seconds_count": "ttft_count",
    "vllm:e2e_request_latency_seconds_sum": "e2e_sum",
    "vllm:e2e_request_latency_seconds_count": "e2e_count",
    "vllm:request_time_per_output_token_seconds_sum": "tpot_sum",
    "vllm:request_time_per_output_token_seconds_count": "tpot_count",
    "vllm:request_prefill_time_seconds_sum": "prefill_sum",
    "vllm:request_prefill_time_seconds_count": "prefill_count",
    "vllm:request_decode_time_seconds_sum": "decode_sum",
    "vllm:request_decode_time_seconds_count": "decode_count",
}


def metrics_url_from_api_base(api_base: str) -> str:
    """Derive vLLM metrics URL from api_base.

    'http://host:8000/v1' -> 'http://host:8000/metrics'
    """
    base = api_base.rstrip("/")
    if base.endswith("/v1"):
        base = base[:-3]
    return base + "/metrics"


def scrape_vllm_metrics(metrics_url: str, timeout: float = 2.0) -> dict | None:
    """Fetch and parse vLLM Prometheus metrics. Returns None on error."""
    try:
        resp = urllib.request.urlopen(metrics_url, timeout=timeout)
        text = resp.read().decode("utf-8")
    except Exception:
        return None

    result = {}
    for line in text.split("\n"):
        if not line or line.startswith("#"):
            continue

        # Extract metric name: everything before '{' or first space
        brace = line.find("{")
        if brace >= 0:
            name = line[:brace]
        else:
            space = line.find(" ")
            if space < 0:
                continue
            name = line[:space]

        if name not in _METRIC_MAP:
            continue

        # Extract value: last token on the line
        try:
            val_str = line.rstrip().rsplit(" ", 1)[-1]
            val = float(val_str)
        except (ValueError, IndexError):
            continue

        result[_METRIC_MAP[name]] = val

    return result if result else None


def sample_inference(stop_event, metrics_url, interval=0.5):
    """Sample vLLM gauges and counters until stop_event is set.

    Returns list of sample dicts. If the endpoint is unreachable on
    the first attempt, returns [] immediately (graceful degradation).
    """
    # Connectivity test
    first = scrape_vllm_metrics(metrics_url)
    if first is None:
        return []

    samples = []
    start = time.time()
    samples.append(
        {
            "t": 0.0,
            **{
                k: first.get(k, 0)
                for k in [
                    "kv_cache_pct",
                    "requests_running",
                    "requests_waiting",
                    "prompt_tokens_total",
                    "generation_tokens_total",
                    "prefix_cache_hits_total",
                    "prefix_cache_queries_total",
                    "preemptions_total",
                ]
            },
        }
    )

    while not stop_event.is_set():
        time.sleep(interval)
        m = scrape_vllm_metrics(metrics_url)
        if m is None:
            continue
        samples.append(
            {
                "t": round(time.time() - start, 2),
                **{
                    k: m.get(k, 0)
                    for k in [
                        "kv_cache_pct",
                        "requests_running",
                        "requests_waiting",
                        "prompt_tokens_total",
                        "generation_tokens_total",
                        "prefix_cache_hits_total",
                        "prefix_cache_queries_total",
                        "preemptions_total",
                    ]
                },
            }
        )

    return samples


def snapshot_histograms(metrics_url: str) -> dict | None:
    """One-shot scrape of histogram _sum/_count fields."""
    m = scrape_vllm_metrics(metrics_url)
    if m is None:
        return None
    keys = [
        "ttft_sum",
        "ttft_count",
        "e2e_sum",
        "e2e_count",
        "tpot_sum",
        "tpot_count",
        "prefill_sum",
        "prefill_count",
        "decode_sum",
        "decode_count",
    ]
    return {k: m.get(k, 0) for k in keys}


def _safe_delta(first, last, key):
    """Compute delta between two samples, returning None on counter reset."""
    d = last.get(key, 0) - first.get(key, 0)
    return d if d >= 0 else None


def compute_inference_stats(samples, hist_before, hist_after):
    """Compute per-concurrency inference metrics from samples + histogram deltas.

    Returns a dict for inclusion in report.json, or None if no valid data.
    """
    if not samples or len(samples) < 2:
        return None

    first, last = samples[0], samples[-1]
    dt = last["t"] - first["t"]
    if dt <= 0:
        return None

    # Gauges: avg/peak from all samples
    kv_vals = [s["kv_cache_pct"] for s in samples]
    run_vals = [s["requests_running"] for s in samples]
    wait_vals = [s["requests_waiting"] for s in samples]

    # Counter deltas
    d_prompt = _safe_delta(first, last, "prompt_tokens_total")
    d_gen = _safe_delta(first, last, "generation_tokens_total")
    d_cache_hits = _safe_delta(first, last, "prefix_cache_hits_total")
    d_cache_queries = _safe_delta(first, last, "prefix_cache_queries_total")
    d_preemptions = _safe_delta(first, last, "preemptions_total")

    # Cache hit rate
    cache_hit_pct = None
    if d_cache_hits is not None and d_cache_queries and d_cache_queries > 0:
        cache_hit_pct = round(100 * d_cache_hits / d_cache_queries, 1)

    # Histogram deltas -> mean latency
    latency = {}
    if hist_before and hist_after:
        for short, sum_key, count_key in [
            ("avg_ttft_sec", "ttft_sum", "ttft_count"),
            ("avg_e2e_sec", "e2e_sum", "e2e_count"),
            ("avg_tpot_sec", "tpot_sum", "tpot_count"),
            ("avg_prefill_sec", "prefill_sum", "prefill_count"),
            ("avg_decode_sec", "decode_sum", "decode_count"),
        ]:
            d_sum = hist_after.get(sum_key, 0) - hist_before.get(sum_key, 0)
            d_count = hist_after.get(count_key, 0) - hist_before.get(count_key, 0)
            if d_count > 0 and d_sum >= 0:
                latency[short] = round(d_sum / d_count, 4)

    return {
        "kv_cache": {
            "avg_pct": round(sum(kv_vals) / len(kv_vals) * 100, 1),
            "peak_pct": round(max(kv_vals) * 100, 1),
        },
        "queue": {
            "avg_running": round(sum(run_vals) / len(run_vals), 1),
            "peak_running": int(max(run_vals)),
            "avg_waiting": round(sum(wait_vals) / len(wait_vals), 1),
            "peak_waiting": int(max(wait_vals)),
        },
        "throughput": {
            "prompt_tok_per_sec": round(d_prompt / dt, 1) if d_prompt is not None else None,
            "gen_tok_per_sec": round(d_gen / dt, 1) if d_gen is not None else None,
        },
        "prefix_cache_hit_pct": cache_hit_pct,
        "preemptions": int(d_preemptions) if d_preemptions is not None else 0,
        "latency": latency or None,
    }
