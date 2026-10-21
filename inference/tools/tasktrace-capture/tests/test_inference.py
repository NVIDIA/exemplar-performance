# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Tests for tasktrace_capture.inference_metrics module.

Covers: metrics_url_from_api_base, scrape_vllm_metrics, compute_inference_stats.
"""

from tasktrace_capture.inference_metrics import (
    compute_inference_stats,
    metrics_url_from_api_base,
    scrape_vllm_metrics,
)

_SAMPLE_PROM_TEXT = """\
# HELP vllm:kv_cache_usage_perc KV-cache usage.
# TYPE vllm:kv_cache_usage_perc gauge
vllm:kv_cache_usage_perc{engine="0",model_name="model"} 0.42
# HELP vllm:num_requests_running Number of requests running.
# TYPE vllm:num_requests_running gauge
vllm:num_requests_running{engine="0",model_name="model"} 3.0
# HELP vllm:num_requests_waiting Number of requests waiting.
# TYPE vllm:num_requests_waiting gauge
vllm:num_requests_waiting{engine="0",model_name="model"} 1.0
# HELP vllm:prompt_tokens_total Total prompt tokens.
# TYPE vllm:prompt_tokens_total counter
vllm:prompt_tokens_total{engine="0",model_name="model"} 50000.0
# HELP vllm:generation_tokens_total Total gen tokens.
# TYPE vllm:generation_tokens_total counter
vllm:generation_tokens_total{engine="0",model_name="model"} 5000.0
# HELP vllm:prefix_cache_hits_total Cache hits.
# TYPE vllm:prefix_cache_hits_total counter
vllm:prefix_cache_hits_total{engine="0",model_name="model"} 40000.0
# HELP vllm:prefix_cache_queries_total Cache queries.
# TYPE vllm:prefix_cache_queries_total counter
vllm:prefix_cache_queries_total{engine="0",model_name="model"} 50000.0
# HELP vllm:num_preemptions_total Preemptions.
# TYPE vllm:num_preemptions_total counter
vllm:num_preemptions_total{engine="0",model_name="model"} 2.0
# HELP vllm:time_to_first_token_seconds TTFT.
# TYPE vllm:time_to_first_token_seconds histogram
vllm:time_to_first_token_seconds_bucket{engine="0",le="0.01"} 5.0
vllm:time_to_first_token_seconds_bucket{engine="0",le="+Inf"} 100.0
vllm:time_to_first_token_seconds_sum{engine="0",model_name="model"} 10.0
vllm:time_to_first_token_seconds_count{engine="0",model_name="model"} 100.0
# HELP vllm:e2e_request_latency_seconds E2E latency.
# TYPE vllm:e2e_request_latency_seconds histogram
vllm:e2e_request_latency_seconds_sum{engine="0",model_name="model"} 200.0
vllm:e2e_request_latency_seconds_count{engine="0",model_name="model"} 100.0
# HELP vllm:request_time_per_output_token_seconds TPOT.
# TYPE vllm:request_time_per_output_token_seconds histogram
vllm:request_time_per_output_token_seconds_sum{engine="0",model_name="model"} 5.0
vllm:request_time_per_output_token_seconds_count{engine="0",model_name="model"} 100.0
"""


class TestMetricsUrlFromApiBase:
    def test_with_v1(self):
        assert metrics_url_from_api_base("http://host:8000/v1") == "http://host:8000/metrics"

    def test_with_trailing_slash(self):
        assert metrics_url_from_api_base("http://host:8000/v1/") == "http://host:8000/metrics"

    def test_without_v1(self):
        assert metrics_url_from_api_base("http://host:8000") == "http://host:8000/metrics"


class TestScrapeVllmMetrics:
    def test_parse_gauges(self):
        from unittest.mock import MagicMock, patch

        resp = MagicMock()
        resp.read.return_value = _SAMPLE_PROM_TEXT.encode()
        with patch("urllib.request.urlopen", return_value=resp):
            m = scrape_vllm_metrics("http://fake/metrics")
        assert m is not None
        assert m["kv_cache_pct"] == 0.42
        assert m["requests_running"] == 3.0
        assert m["requests_waiting"] == 1.0

    def test_parse_counters(self):
        from unittest.mock import MagicMock, patch

        resp = MagicMock()
        resp.read.return_value = _SAMPLE_PROM_TEXT.encode()
        with patch("urllib.request.urlopen", return_value=resp):
            m = scrape_vllm_metrics("http://fake/metrics")
        assert m["prompt_tokens_total"] == 50000.0
        assert m["generation_tokens_total"] == 5000.0
        assert m["prefix_cache_hits_total"] == 40000.0
        assert m["preemptions_total"] == 2.0

    def test_parse_histogram_sum_count(self):
        from unittest.mock import MagicMock, patch

        resp = MagicMock()
        resp.read.return_value = _SAMPLE_PROM_TEXT.encode()
        with patch("urllib.request.urlopen", return_value=resp):
            m = scrape_vllm_metrics("http://fake/metrics")
        assert m["ttft_sum"] == 10.0
        assert m["ttft_count"] == 100.0
        assert m["e2e_sum"] == 200.0
        assert m["tpot_sum"] == 5.0

    def test_returns_none_on_error(self):
        from unittest.mock import patch

        with patch("urllib.request.urlopen", side_effect=Exception("nope")):
            assert scrape_vllm_metrics("http://fake/metrics") is None


class TestComputeInferenceStats:
    def _make_samples(self, n=10, dt=1.0):
        samples = []
        for i in range(n):
            samples.append(
                {
                    "t": i * dt,
                    "kv_cache_pct": 0.1 + i * 0.02,
                    "requests_running": 2 + (i % 3),
                    "requests_waiting": i % 2,
                    "prompt_tokens_total": 1000 + i * 100,
                    "generation_tokens_total": 500 + i * 50,
                    "prefix_cache_hits_total": 800 + i * 80,
                    "prefix_cache_queries_total": 1000 + i * 100,
                    "preemptions_total": 0,
                }
            )
        return samples

    def test_basic_stats(self):
        samples = self._make_samples()
        hist_before = {
            "ttft_sum": 1.0,
            "ttft_count": 10,
            "e2e_sum": 20.0,
            "e2e_count": 10,
            "tpot_sum": 0.5,
            "tpot_count": 10,
            "prefill_sum": 0,
            "prefill_count": 0,
            "decode_sum": 0,
            "decode_count": 0,
        }
        hist_after = {
            "ttft_sum": 2.0,
            "ttft_count": 20,
            "e2e_sum": 40.0,
            "e2e_count": 20,
            "tpot_sum": 1.0,
            "tpot_count": 20,
            "prefill_sum": 0,
            "prefill_count": 0,
            "decode_sum": 0,
            "decode_count": 0,
        }
        stats = compute_inference_stats(samples, hist_before, hist_after)
        assert stats is not None
        assert stats["kv_cache"]["avg_pct"] > 0
        assert stats["kv_cache"]["peak_pct"] > stats["kv_cache"]["avg_pct"]
        assert stats["throughput"]["prompt_tok_per_sec"] > 0
        assert stats["throughput"]["gen_tok_per_sec"] > 0
        assert stats["prefix_cache_hit_pct"] == 80.0
        assert stats["latency"]["avg_ttft_sec"] == 0.1  # delta 1.0 / delta 10

    def test_empty_samples(self):
        assert compute_inference_stats([], None, None) is None

    def test_single_sample(self):
        assert compute_inference_stats([{"t": 0}], None, None) is None

    def test_no_histograms(self):
        samples = self._make_samples()
        stats = compute_inference_stats(samples, None, None)
        assert stats is not None
        assert stats["latency"] is None

    def test_zero_cache_queries(self):
        samples = self._make_samples()
        # Make cache queries stay constant (no new queries)
        for s in samples:
            s["prefix_cache_queries_total"] = 1000
            s["prefix_cache_hits_total"] = 800
        stats = compute_inference_stats(samples, None, None)
        assert stats["prefix_cache_hit_pct"] is None
