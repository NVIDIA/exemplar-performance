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

"""selftest_goal_handlers.py — offline guards for the GOAL-AWARE spine (one handler per (scenario, goal)).

No cluster, no network. Covers:
  A. REGISTRY + STARTUP ASSERTION — every schema goal (envelope.yaml enum) + every KNOWN_METRICS metric
     resolves to a handler; a declared-but-unhandled goal fails LOUDLY (the old leak-surface, now an assert).
  B. RESOLUTION — (scenario, goal) → the right handler.
  C. PER-HANDLER metric/headline/table/chart correctness for both combos A–B.
  D. compare.py — min_cost_at_target_score is lower-better; report() refuses to mix metrics.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "scripts"))
import goal_handlers as gh  # noqa: E402
import charts  # noqa: E402
import compare as cmp  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── A. registry + the startup assertion ─────────────────────────────────────────────────────────────────
check(
    "startup assertion: every schema goal + KNOWN_METRICS metric has a handler",
    gh.assert_every_goal_has_a_handler() is True,
)

# a declared goal with NO handler must FAIL LOUDLY (this is the leak-surface turned into an assertion)
raised = False
try:
    gh.assert_every_goal_has_a_handler(
        schema_goals=["max-concurrency-sla", "pareto", "agentic-eval-optimization", "ghost-future-goal"]
    )
except AssertionError as e:
    raised = "ghost-future-goal" in str(e)
check("startup assertion: a declared goal with no handler raises AssertionError (loud, not silent)", raised)

# a KNOWN metric with no handler also fails loudly
raised2 = False
try:
    gh.assert_every_goal_has_a_handler(known_metrics={"llm-perf": {"some_unhandled_metric"}})
except AssertionError as e:
    raised2 = "some_unhandled_metric" in str(e)
check("startup assertion: a KNOWN_METRICS metric with no handler raises", raised2)

check(
    "registry covers exactly the 2 (scenario, goal) combos",
    set(gh.REGISTRY.keys())
    == {
        ("llm-perf", "max-concurrency-sla"),
        ("llm-perf", "pareto"),
    },
)

# ── B. resolution (incl. the scenario-default lane) ─────────────────────────────────────────────────────
check(
    "resolve llm-perf·max-concurrency-sla → LlmSlaHandler",
    type(gh.resolve("llm-perf", "max-concurrency-sla")).__name__ == "LlmSlaHandler",
)
check(
    "resolve llm-perf·pareto → LlmParetoHandler", type(gh.resolve("llm-perf", "pareto")).__name__ == "LlmParetoHandler"
)
raised3 = False
try:
    gh.resolve("llm-perf", "no-such-goal")
except KeyError:
    raised3 = True
check("resolve of an unregistered (scenario, goal) raises KeyError (never a silent default)", raised3)

# ── C. per-handler correctness ──────────────────────────────────────────────────────────────────────────
# A — llm-perf · max-concurrency-sla
A = gh.resolve("llm-perf", "max-concurrency-sla")
A_recipe = {
    "envelope": {
        "exemplar": {
            "metric": "max_concurrency_at_sla",
            "comparison": "highest sweep_concurrency rung with TTFT<=limit",
        }
    },
    "bench": {"sla": {"ttft_ms": 10000, "tpot_ms": 100, "stop_stat": "p50"}},
}
A_rows = [
    {
        "concurrency": "256",
        "ttft_p50_ms": "910",
        "itl_p50_ms": "94",
        "throughput_per_gpu_tok_per_s": "145",
        "tokens_per_s_per_user_from_itl": "10",
        "error_rate_pct": "",
    },
    {
        "concurrency": "384",
        "ttft_p50_ms": "1487",
        "itl_p50_ms": "115",
        "throughput_per_gpu_tok_per_s": "148",
        "tokens_per_s_per_user_from_itl": "8",
        "error_rate_pct": "",
    },
]
mA = A.compute_metric(A_rows, A_recipe)
check(
    "A compute_metric → (max_concurrency_at_sla, highest passing rung 256, highest_sla_passing_rung)",
    mA == ("max_concurrency_at_sla", 256, "highest_sla_passing_rung"),
    str(mA),
)
rfA = A.result_fields(A_rows, A_recipe)
check(
    "A result_fields carry sla + crossing (SLA semantics belong to A only)",
    "sla" in rfA and "crossing" in rfA and rfA["value"] == 256,
)
hA = A.headline(
    {
        "value": 256,
        "unit": "concurrency",
        "passing_concurrencies": [256],
        "sla": {"ttft_ms": 10000, "tpot_ms": 100, "stop_stat": "p50"},
    }
)
check(
    "A headline: metric + passing rungs + SLA gate",
    hA == "`max_concurrency_at_sla` = **256** (unit: concurrency) · passing rungs: [256] · "
    "SLA TTFT≤10000ms / TPOT≤100ms @ p50",
    hA,
)
check("A table has the SLA column", "SLA |" in A.table(A_rows, A_recipe))
check("A chart_key = sla", A.chart_key() == "sla")

# B — llm-perf · pareto (NO SLA semantics; emits ONLY result.value)
B = gh.resolve("llm-perf", "pareto")
mB = B.compute_metric(A_rows, {})
check("B compute_metric → pareto_geomean over the frontier", mB[0] == "pareto_geomean" and mB[1] is not None, str(mB))
_rfB = B.result_fields(A_rows, {})
check(
    "B result_fields emit value + value_source + per_point + axes (point-by-point exemplar basis)",
    set(_rfB.keys()) == {"value", "value_source", "per_point", "axes"}
    and isinstance(_rfB["per_point"], list)
    and _rfB["axes"] == ["tps_per_gpu", "tps_per_user"],
    str(sorted(_rfB.keys())),
)
check("B per_point carries a g per rung", all("concurrency" in p and "g" in p for p in _rfB["per_point"]))
hB = B.headline(
    {
        "value": 397.6,
        "unit": "geomean(tps/gpu·tps/user)",
        "swept": [256, 384],
        "per_point": [{"concurrency": 256, "g": 1}, {"concurrency": 384, "g": 2}],
    }
)
check(
    "B headline: pareto_geomean + N points + axes + swept (not SLA rungs)",
    hB
    == (
        "`pareto_geomean` = **397.6** (geomean of 2 points; "
        "axes Output-TPS/GPU × Output-TPS/user) · swept: [256, 384]"
    ),
    hB,
)
_bt = B.table(A_rows, {})
check("B table has NO SLA column", "SLA |" not in _bt)
check(
    "B table leads with the two pareto metrics + a geomean column",
    "Out TPS/GPU" in _bt and "Out TPS/user" in _bt and "geomean" in _bt.lower() and "TPOT" not in _bt,
    _bt.splitlines()[0],
)
check("B chart_key = pareto", B.chart_key() == "pareto")

# ── D. compare.py: lower-better + refuse mixed metrics ──────────────────────────────────────────────────
check(
    "compare.HIGHER_BETTER: min_cost_at_target_score is LOWER-better",
    cmp.HIGHER_BETTER.get("min_cost_at_target_score") is False,
)
check("compare.HIGHER_BETTER: pareto_geomean is higher-better", cmp.HIGHER_BETTER.get("pareto_geomean") is True)
mixed = [
    {
        "cell": "a",
        "path": "a",
        "gpu": "GB300",
        "metric": "pareto_geomean",
        "value": 1.2,
        "unit": "u",
        "tol": 5,
        "status": "runs",
    },
    {
        "cell": "b",
        "path": "b",
        "gpu": "GB300",
        "metric": "min_cost_at_target_score",
        "value": 6.0,
        "unit": "u",
        "tol": 5,
        "status": "runs",
    },
]
lines, obj = cmp.report(mixed)
check(
    "compare.report REFUSES to compare mixed metrics (would invert lower-better vs higher-better)",
    obj is None and any("mix metrics" in l for l in (lines or [])),
    str(lines),
)
same = [
    {
        "cell": "a",
        "path": "a",
        "gpu": "GB300",
        "metric": "min_cost_at_target_score",
        "value": 6.0,
        "unit": "u",
        "tol": 5,
        "status": "runs",
    },
    {
        "cell": "b",
        "path": "b",
        "gpu": "GB300",
        "metric": "min_cost_at_target_score",
        "value": 9.0,
        "unit": "u",
        "tol": 5,
        "status": "runs",
    },
]
lines2, obj2 = cmp.report(same)
check(
    "compare.report on min_cost ranks LOWER-better (baseline = the cheaper cell 'a')",
    obj2
    and obj2["higher_better"] is False
    and lines2[1].startswith("_mode: **REPRODUCIBILITY**")
    and "baseline **a**" in lines2[1],
    str(lines2[:2]),
)

# ── goal=pareto WITHOUT bench.sla is a SUPPORTED, tested shape ────────────────────────────────────────
# SLA gates NOTHING in the pareto path: goal_handlers has no sla reference, sweep_mode=fixed means no
# SLA-based early stop (that belongs to max-concurrency-sla's adaptive search), and charts uses SLA limits
# only for the max-concurrency-sla 2D TTFT×TPOT plane. `sla` is schema-OPTIONAL, so a pareto cell may omit
# it — these guard that omitting it stays legal and never trips the invariant gate.
_bench_no_sla = {"sweep_mode": "fixed"}  # NO 'sla' key at all
_stat = (_bench_no_sla.get("sla") or {}).get("stop_stat")
check("check_invariants: absent bench.sla degrades to stop_stat=None (no crash)", _stat is None)
check("check_invariants: stop_stat=None passes the pinned-p50 protocol check", _stat in (None, "p50"))
# end-to-end on the real tree: every committed goal=pareto cell now omits sla.
import yaml as _yaml  # noqa: E402

_pareto = [
    p
    for p in (ROOT / "recipes").glob("**/recipe.yaml")
    if ((_yaml.safe_load(p.read_text()) or {}).get("envelope") or {}).get("goal") == "pareto"
]
check(
    "every committed goal=pareto cell omits bench.sla (dead config removed; shape real + tested)",
    bool(_pareto) and all("sla" not in ((_yaml.safe_load(p.read_text()) or {}).get("bench") or {}) for p in _pareto),
    f"{len(_pareto)} pareto cell(s)",
)

print(("\nFAIL: " + ", ".join(fails)) if fails else "\nselftest_goal_handlers: all checks passed")
sys.exit(1 if fails else 0)
