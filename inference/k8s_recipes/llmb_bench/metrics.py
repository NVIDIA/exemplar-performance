# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""llmb_bench.metrics — parse a vLLM Prometheus /metrics blob into a summary dict. Pure.

Carved from the inline python the bench runner uses to read the server's /metrics surface. Counters
(``*_total``) are summed across label sets; gauges take the last value. Missing metrics → 0.
"""

from __future__ import annotations

import re


def _sum_counter(text: str, name: str) -> float:
    """Sum a Prometheus counter across all label sets (bare or labelled lines)."""
    total = 0.0
    pat = re.compile(rf"^{re.escape(name)}(?:_total)?[ {{]")
    for ln in text.splitlines():
        if pat.match(ln):
            try:
                total += float(ln.split()[-1])
            except (ValueError, IndexError):
                pass
    return total


def _gauge(text: str, name: str) -> float:
    """Last value of a Prometheus gauge (bare or labelled)."""
    val = 0.0
    pat = re.compile(rf"^{re.escape(name)}[ {{]")
    for ln in text.splitlines():
        if pat.match(ln):
            try:
                val = float(ln.split()[-1])
            except (ValueError, IndexError):
                pass
    return val


def parse_metrics(text: str) -> dict:
    """vLLM /metrics blob → {generation_tokens, prompt_tokens, num_requests_running, num_requests_waiting}."""
    return {
        "generation_tokens": int(_sum_counter(text, "vllm:generation_tokens")),
        "prompt_tokens": int(_sum_counter(text, "vllm:prompt_tokens")),
        "num_requests_running": _gauge(text, "vllm:num_requests_running"),
        "num_requests_waiting": _gauge(text, "vllm:num_requests_waiting"),
    }
