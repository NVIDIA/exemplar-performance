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

"""selftest_pooled_stats.py — guards for the pooled-headline rule (pooled_stats.py + check_pooled_publication.py).

No cluster, no network. Covers:
  A. KERNEL      — Student-t (not z), exact mean/sd/CI, n=1 is honest (no fabricated band), n=0 is None.
  B. POWER       — required_n / single_run_interval_pct reproduce the numbers the policy rests on.
  C. CI GATE     — ci_within_tolerance returns PASS / FAIL / INCONCLUSIVE, and a single run is
                   structurally INCONCLUSIVE rather than silently "passing" a point comparison.
  D. ANTI-BLEND  — pooling_key / assert_poolable refuse to merge across rung, context length,
                   attempts-per-task or scoring policy.
  E. PUBLICATION — check_pooled_publication refuses a single-run headline when the MEASURED sigma
                   exceeds the tolerance, allows it when the metric is genuinely quiet, ignores
                   ledger runs at a different recipe_hash, and honours the grandfather list.
  F. REGRESSION  — the real measured corpus reproduces the published pooled figure exactly.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pooled_stats as ps  # noqa: E402
import check_pooled_publication as cpp  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ---------------------------------------------------------------- A. kernel
print("A. KERNEL")
check("n=0 -> None (never a fabricated statistic)", ps.pooled([]) is None)

one = ps.pooled([0.42])
check(
    "n=1 -> mean set, sd/ci None, single_run flagged",
    one["n"] == 1 and one["mean"] == 0.42 and one["sd"] is None and one["ci_lo"] is None and one["single_run"] is True,
    str(one),
)
check("n=1 rendering says SINGLE RUN out loud", "SINGLE RUN" in ps.format_pooled(one), ps.format_pooled(one))

p = ps.pooled([1.0, 2.0, 3.0, 4.0])
check("mean exact", abs(p["mean"] - 2.5) < 1e-12)
check("sd is the SAMPLE sd (n-1)", abs(p["sd"] - 1.2909944487358056) < 1e-12, str(p["sd"]))
check("uses Student-t(3)=3.182, not z=1.96", abs(p["t"] - 3.182) < 1e-9, str(p["t"]))
check("CI half-width = t*se", abs((p["ci_hi"] - p["mean"]) - p["t"] * p["se"]) < 1e-12)

check("t table monotone decreasing toward z", ps.t_quantile_975(1) > ps.t_quantile_975(20) > ps.t_quantile_975(1000))
check("t(df>100) falls back to the normal quantile", abs(ps.t_quantile_975(5000) - 1.95996398454) < 1e-9)
check(
    "t interpolates between tabulated df (35 sits between 30 and 40)",
    ps.t_quantile_975(40) < ps.t_quantile_975(35) < ps.t_quantile_975(30),
)

check("pooled ignores None entries rather than crashing", ps.pooled([1.0, None, 3.0])["n"] == 2)

# ---------------------------------------------------------------- B. power
print("\nB. POWER ARITHMETIC")
check(
    "sigma 12.6% -> a single run carries ~+/-24.7%",
    abs(ps.single_run_interval_pct(12.6) - 24.7) < 0.1,
    f"{ps.single_run_interval_pct(12.6):.2f}",
)
check("+/-5% at sigma 12.6% needs n=27", ps.required_n(12.6, 5.0) == 27, str(ps.required_n(12.6, 5.0)))
# Student-t is unforgiving at tiny n: t(1)=12.706, so n=2 can never reach +/-5% unless sigma
# is under ~0.6%. A quiet 2% metric needs n=3. Using z=1.96 here would wrongly say n=2.
check(
    "a quiet metric (sigma 2%) needs n=3 for +/-5% (t, not z)",
    ps.required_n(2.0, 5.0) == 3,
    str(ps.required_n(2.0, 5.0)),
)
check(
    "required_n is monotone in sigma", ps.required_n(5.0, 5.0) <= ps.required_n(10.0, 5.0) <= ps.required_n(20.0, 5.0)
)
check("sigma=0 -> n=1 suffices", ps.required_n(0.0, 5.0) == 1)

# ---------------------------------------------------------------- C. CI gate
print("\nC. CI-BASED TOLERANCE GATE (replaces the point comparison)")
tight = ps.pooled([1.00, 1.01, 0.99, 1.00, 1.01, 0.99, 1.00, 1.00])
check("CI entirely inside the band -> PASS", ps.ci_within_tolerance(tight, 1.0, 5.0)["verdict"] == "PASS")

low = ps.pooled([0.50, 0.51, 0.49, 0.50, 0.51, 0.49, 0.50, 0.50])
check(
    "CI entirely below the band -> FAIL (a proven regression)",
    ps.ci_within_tolerance(low, 1.0, 5.0)["verdict"] == "FAIL",
)

noisy = ps.pooled([0.80, 1.20, 0.90, 1.10, 0.85, 1.15])
check(
    "CI straddling the band edge -> INCONCLUSIVE, not a silent pass",
    ps.ci_within_tolerance(noisy, 1.0, 5.0)["verdict"] == "INCONCLUSIVE",
)

single = ps.pooled([1.0])
v = ps.ci_within_tolerance(single, 1.0, 5.0)
check(
    "a SINGLE run is structurally INCONCLUSIVE even when it sits dead on the reference",
    v["verdict"] == "INCONCLUSIVE_SINGLE_RUN",
    v["verdict"],
)

# The case that motivated all of this: 15 real replicates vs the published c16 exemplar.
C16 = [
    0.31353,
    0.33975,
    0.36452,
    0.37179,
    0.39629,
    0.39713,
    0.40258,
    0.41663,
    0.42665,
    0.43267,
    0.43623,
    0.46296,
    0.46455,
    0.48631,
    0.49219,
]
real = ps.ci_within_tolerance(ps.pooled(C16), 0.46296, 5.0)
check(
    "15 real replicates vs the 0.46296 exemplar -> INCONCLUSIVE (the honest answer)",
    real["verdict"] == "INCONCLUSIVE",
    real["verdict"],
)

lower_better = ps.ci_within_tolerance(ps.pooled([100.0, 101.0, 99.0, 100.0]), 100.0, 5.0, higher_is_better=False)
check("lower-is-better direction is honoured", lower_better["verdict"] == "PASS")
check(
    "no reference -> NO_REFERENCE, never a fabricated verdict",
    ps.ci_within_tolerance(tight, None, 5.0)["verdict"] == "NO_REFERENCE",
)
check("no data -> NO_DATA", ps.ci_within_tolerance(None, 1.0, 5.0)["verdict"] == "NO_DATA")

# ---------------------------------------------------------------- D. anti-blend
print("\nD. ANTI-BLENDING GUARD")
base = {
    "cell": "x",
    "recipe_hash": "h",
    "metric": "net_behavior_score",
    "rung": "c16",
    "context_len": "256k",
    "attempts_per_task": 3,
    "scoring_policy": "persist",
}
check("identical keys are poolable", ps.assert_poolable([dict(base), dict(base)]) == [])
for field, other in (
    ("rung", "c32"),
    ("context_len", "1M"),
    ("attempts_per_task", 5),
    ("scoring_policy", "rich"),
    ("recipe_hash", "h2"),
):
    mixed = [dict(base), dict(base, **{field: other})]
    reasons = ps.assert_poolable(mixed)
    check(f"refuses to pool across differing {field}", any(field in r for r in reasons), str(reasons))
check(
    "every pooling-key field is actually consulted",
    len(ps.POOLING_KEY_FIELDS) == len(set(ps.POOLING_KEY_FIELDS))
    and "scoring_policy" in ps.POOLING_KEY_FIELDS
    and "rung" in ps.POOLING_KEY_FIELDS,
)

# ---------------------------------------------------------------- E. publication gate
print("\nE. PUBLICATION GATE")

REG = {
    "families": {
        "llm-perf/pareto/net_behavior_score": {"sigma_pct": 12.6, "n_replicates": 15},
        "llm-perf/sla/max_concurrency_at_sla": {"sigma_pct": 2.0, "n_replicates": 5},
    },
    "fallbacks": {"min_replicates_for_own_sigma": 3, "scenario_sigma_floor_pct": {"llm-perf": 12.6}},
}


def make_cell(
    td: str, name: str, scenario: str, goal: str, metric: str, values: list, rhash: str = "H", extra_hash_runs: int = 0
) -> Path:
    cell = Path(td) / name
    cell.mkdir(parents=True)
    (cell / "record.json").write_text(
        json.dumps(
            {
                "identity": {"scenario": scenario, "goal": goal, "status": "runs"},
                "fingerprint": {"recipe_hash": rhash},
                "result": {"metric": metric, "value": values[-1] if values else None, "tolerance_pct": 5.0},
            }
        )
    )
    (cell / "runs").mkdir()
    lines = [
        json.dumps({"run_id": f"r{i}", "metric": metric, "value": v, "recipe_hash": rhash})
        for i, v in enumerate(values)
    ]
    lines += [
        json.dumps({"run_id": f"x{i}", "metric": metric, "value": 9.9, "recipe_hash": "OTHER"})
        for i in range(extra_hash_runs)
    ]
    (cell / "runs" / "index.jsonl").write_text("\n".join(lines) + ("\n" if lines else ""))
    return cell


with tempfile.TemporaryDirectory() as td:
    c = make_cell(td, "single", "llm-perf", "pareto", "net_behavior_score", [0.46296])
    r = cpp.check_cell(c, REG)
    check(
        "single-run headline on a NOISY metric -> SINGLE_RUN (refused)",
        r["verdict"] == "SINGLE_RUN",
        f"{r['verdict']} / {r['detail']}",
    )
    check("  ... and it quotes the required n", r["required_n"] == 27, str(r["required_n"]))
    check(
        "  ... and the single-run interval it actually carries",
        abs(r["single_run_interval_pct"] - 24.7) < 0.2,
        str(r.get("single_run_interval_pct")),
    )

    c = make_cell(td, "few", "llm-perf", "pareto", "net_behavior_score", [0.40, 0.42, 0.44, 0.41])
    r = cpp.check_cell(c, REG)
    check(
        "n=4 against sigma 12.6% / tol 5% -> UNDERPOWERED (refused)",
        r["verdict"] == "UNDERPOWERED",
        f"{r['verdict']} / {r['detail']}",
    )

    c = make_cell(td, "quiet", "llm-perf", "sla", "max_concurrency_at_sla", [128.0])
    r = cpp.check_cell(c, REG)
    check(
        "single run on a genuinely QUIET metric (sigma 2% <= tol 5%) is allowed",
        r["verdict"] == "QUIET",
        f"{r['verdict']} / {r['detail']}",
    )

    c = make_cell(td, "enough", "llm-perf", "pareto", "net_behavior_score", [0.41 + 0.0005 * i for i in range(30)])
    r = cpp.check_cell(c, REG)
    check(
        "n=30 with a tight spread -> OK, and reports its pooled CI",
        r["verdict"] == "OK" and r["pooled"]["n"] == 30,
        f"{r['verdict']} n={r['n']}",
    )

    c = make_cell(td, "lucky", "llm-perf", "pareto", "net_behavior_score", [0.4260, 0.4259, 0.4258])
    r = cpp.check_cell(c, REG)
    check(
        "a lucky n=3 cannot certify itself quiet — family sigma wins",
        r["sigma_pct"] >= 12.6 and r["verdict"] != "OK",
        f"sigma={r['sigma_pct']} verdict={r['verdict']} src={r['sigma_source']}",
    )

    c = make_cell(td, "drift", "llm-perf", "pareto", "net_behavior_score", [0.46296], extra_hash_runs=40)
    r = cpp.check_cell(c, REG)
    check(
        "ledger runs at a DIFFERENT recipe_hash do not count toward n",
        r["n"] == 1 and r["dropped_hash_mismatch"] == 40,
        f"n={r['n']} dropped={r['dropped_hash_mismatch']}",
    )

    c = make_cell(td, "unmeasured", "some-scenario", "some-goal", "some_metric", [1.0])
    r = cpp.check_cell(c, REG)
    check(
        "no MEASURED sigma and n<3 -> UNKNOWN_SIGMA (advisory, not a fabricated pass)",
        r["verdict"] == "UNKNOWN_SIGMA",
        r["verdict"],
    )

check(
    "grandfather list is a set of repo-relative cell paths",
    isinstance(cpp.GRANDFATHERED, set) and all(q.startswith("recipes/") for q in cpp.GRANDFATHERED),
)

check(
    "the shipped sigma registry parses and carries provenance for every family",
    all(
        {"sigma_pct", "n_replicates", "source", "measured_on"} <= set(v)
        for v in cpp.load_registry()["families"].values()
    ),
)

# ---------------------------------------------------------------- F. regression on real data
print("\nF. REGRESSION AGAINST THE MEASURED CORPUS")
pc = ps.pooled(C16)
check(
    "15 GB300 c16 @3att/256k/persist replicates -> mean 0.41359", abs(pc["mean"] - 0.41359) < 5e-5, f"{pc['mean']:.5f}"
)
check("  ... sd 12.6% of the mean", abs(pc["cv_pct"] - 12.6) < 0.15, f"{pc['cv_pct']:.2f}")
check(
    "  ... 95% CI [0.3848, 0.4424] (+/-7.0%)",
    abs(pc["ci_lo"] - 0.38478) < 5e-4 and abs(pc["ci_hi"] - 0.44239) < 5e-4 and abs(pc["ci_half_pct"] - 7.0) < 0.1,
    f"[{pc['ci_lo']:.5f},{pc['ci_hi']:.5f}] +/-{pc['ci_half_pct']:.1f}%",
)
check(
    "the registry's published sigma matches the corpus it claims to come from",
    abs(cpp.load_registry()["families"]["llm-perf/pareto/net_behavior_score"]["sigma_pct"] - round(pc["cv_pct"], 1))
    < 0.05,
)

print(("\nFAIL: " + ", ".join(fails)) if fails else "\nselftest_pooled_stats: all checks passed")
sys.exit(1 if fails else 0)
