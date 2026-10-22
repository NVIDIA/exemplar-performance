# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa

"""llmb_bench.gate — the per-rung SLA decision. Pure.

Carved from bench-job.yaml.j2's inline python that reads `profile_export_aiperf.json` after each concurrency
rung and decides PASS / breach / stop. This is THE gate the sweep hangs on: a rung passes iff aiperf exited 0
AND both TTFT and TPOT (at the configured stop-stat, default p50) are within their limits. `gating_ratio`
(max of the two ratios) is what the adaptive search steers on. Faithful to the shell, so the decision — and
therefore the published max_concurrency_at_sla — is unchanged by the extraction.
"""

from __future__ import annotations


def _metric(export: dict, key: str, stat: str):
    """export[key][stat] as float, or None (missing key / non-dict / non-numeric) — mirrors the shell's `metric`."""
    node = export.get(key)
    if not isinstance(node, dict):
        return None
    try:
        return float(node.get(stat))
    except (TypeError, ValueError):
        return None


def evaluate_rung(
    export: dict,
    *,
    stat: str = "p50",
    ttft_limit_ms: float,
    tpot_limit_ms: float,
    aiperf_rc: int = 0,
    concurrency: int = 0,
    phase: str = "",
) -> dict:
    """Decide one rung from its aiperf export. Returns the same dict shape the shell emitted:
    {concurrency, phase, aiperf_exit, stop_stat, ttft/tpot limits, parse_ok, passed, stop_reason, breaches,
     [ttft_ms, tpot_ms, gating_ratio]}. passed = (rc==0 AND ttft≤limit AND tpot≤limit at `stat`)."""
    result = {
        "phase": phase,
        "concurrency": concurrency,
        "aiperf_exit": aiperf_rc,
        "stop_stat": stat,
        "ttft_limit_ms": ttft_limit_ms,
        "tpot_limit_ms": tpot_limit_ms,
        "parse_ok": False,
        "passed": False,
        "stop_reason": "unparseable_metrics",
        "breaches": [],
    }
    try:
        ttft = _metric(export, "time_to_first_token", stat)
        tpot = _metric(export, "inter_token_latency", stat)
        if ttft is None or tpot is None:
            raise ValueError(f"missing {stat} metrics")
        result.update(
            {
                "parse_ok": True,
                "ttft_ms": ttft,
                "tpot_ms": tpot,
                "gating_ratio": max(ttft / ttft_limit_ms, tpot / tpot_limit_ms),
            }
        )
        breaches = []
        if ttft > ttft_limit_ms:
            breaches.append("ttft")
        if tpot > tpot_limit_ms:
            breaches.append("tpot")
        result["breaches"] = breaches
        if aiperf_rc != 0:
            result["stop_reason"] = "unparseable_metrics"  # a non-zero aiperf run is not a clean pass
        elif breaches:
            result["stop_reason"] = "threshold_exceeded"
        else:
            result["passed"] = True
            result["stop_reason"] = ""
    except Exception as exc:
        result["parse_error"] = str(exc)
    return result
