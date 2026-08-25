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

"""check_pooled_publication.py — refuse to publish a single-run headline for a NOISY metric.

THE RULE
--------
A published headline must be a mean of N with a confidence interval. A cell may publish a
single-run headline only when that is defensible: i.e. when the metric's MEASURED run-to-run
sigma is no larger than the cell's own exemplar tolerance. When sigma exceeds the tolerance,
one run cannot decide anything the gate claims to decide, and publication is refused.

    tolerance_pct = 5   and   measured sigma = 12.6%
    -> a single run carries a +/-24.7% 95% interval, five times the bar it is checked against
    -> n=27 runs are needed before the pooled CI is actually +/-5%

WHAT IT CHECKS, per cell carrying a record.json
----------------------------------------------
  1. n      — replicates behind the headline, counted from runs/index.jsonl (fallback runs.jsonl),
              restricted to entries whose recipe_hash matches the published record. A run at a
              different fingerprint is a different configuration and does not count.
  2. sigma  — the cell's own sigma when it has >= min_replicates_for_own_sigma runs, else the
              MEASURED family sigma from analysis/measured_sigma.json. A scenario sigma FLOOR
              stops a lucky n=2/n=3 certifying itself as quiet.
  3. verdict:
       OK              n >= required_n(sigma, tolerance)  — headline is a pooled mean with a CI
       SINGLE_RUN      n == 1 and sigma > tolerance       — REFUSED (this is the systemic defect)
       UNDERPOWERED    1 < n < required_n                 — REFUSED
       QUIET           sigma <= tolerance                 — a single run is defensible; allowed
       UNKNOWN_SIGMA   no measured sigma and n < 3        — ADVISORY only. The rule keys on
                                                MEASURED sigma; refusing on an unmeasured one
                                                would block every new cell and would be a
                                                different (unasked-for) policy. It is printed
                                                loudly so the gap is visible and closable by
                                                adding an entry to analysis/measured_sigma.json.

GRANDFATHERING
--------------
Cells already published before this gate existed are listed in GRANDFATHERED below. They are
reported loudly but do not fail CI, because fixing them means RE-PUBLISHING a number and that is
a human decision, not a script's. `--strict` fails on them too — run it to see the real backlog.
Nothing NEW can be added to that list without editing this file in a reviewed commit.

Read-only: never writes, never re-scores, never re-stamps.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import pooled_stats as ps  # noqa: E402

SIGMA_REGISTRY = ROOT / "analysis" / "measured_sigma.json"

# Cells published before the pooled-headline rule existed. Reported, not failed.
# To remove an entry: re-run the cell to the required n, re-publish, delete the line.
GRANDFATHERED = {
    "recipes/llm-perf/1m/nemotron-ultra-3-gb300-vllm-agg",
    "recipes/llm-perf/256k/nemotron-ultra-3-gb300-vllm-agg",
    "recipes/llm-perf/256k/nemotron-ultra-3-gb300-vllm-agg-pareto",
    "recipes/llm-perf/256k/nemotron-ultra-3-gb200-vllm-agg",
    "recipes/llm-perf/256k/nemotron-ultra-3-gb200-vllm-agg-pareto",
    "recipes/llm-perf/256k/nemotron-ultra-3-b200-vllm-agg",
    "recipes/llm-perf/256k/nemotron-ultra-3-b200-vllm-agg-pareto",
    "recipes/llm-perf/glm5-16k512/glm5-fp8-b200-sglang-16k512-hightpt-c240-1p1d",
    "recipes/llm-perf/glm5-9600/glm5-fp8-b200-sglang-1p1d-pareto",
}


def load_registry() -> dict:
    if not SIGMA_REGISTRY.exists():
        return {"families": {}, "fallbacks": {}}
    return json.loads(SIGMA_REGISTRY.read_text())


def read_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    out = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def ledger_values(cell: Path, metric: str, recipe_hash: str | None) -> tuple[list[float], int]:
    """Replicate values behind a headline. Returns (values, n_dropped_for_hash_mismatch)."""
    entries = read_jsonl(cell / "runs" / "index.jsonl") or read_jsonl(cell / "runs.jsonl")
    vals, dropped, seen = [], 0, set()
    for e in entries:
        if e.get("metric") != metric or e.get("value") is None:
            continue
        rid = e.get("run_id")
        if rid in seen:  # index.jsonl is keyed on run_id; runs.jsonl is not
            continue
        seen.add(rid)
        if recipe_hash and e.get("recipe_hash") and e["recipe_hash"] != recipe_hash:
            dropped += 1
            continue
        try:
            vals.append(float(e["value"]))
        except (TypeError, ValueError):
            continue
    return vals, dropped


def family_key(ident: dict, metric: str) -> str:
    return f"{ident.get('scenario')}/{ident.get('goal')}/{metric}"


def resolve_sigma(cell_rel: str, ident: dict, metric: str, vals: list[float], reg: dict) -> tuple[float | None, str]:
    """(sigma_pct, provenance). Cell's own sigma only when it has enough replicates."""
    fb = reg.get("fallbacks", {}) or {}
    min_own = int(fb.get("min_replicates_for_own_sigma", 3))
    floors = fb.get("scenario_sigma_floor_pct", {}) or {}
    floor = floors.get(ident.get("scenario"))

    own = None
    if len(vals) >= min_own:
        p = ps.pooled(vals)
        if p and p.get("cv_pct") is not None:
            own = p["cv_pct"]

    fam = (reg.get("families", {}) or {}).get(family_key(ident, metric))
    fam_sigma = fam.get("sigma_pct") if fam else None

    if own is not None and fam_sigma is not None:
        s = max(own, fam_sigma)
        return s, f"max(own n={len(vals)} {own:.1f}%, measured family {fam_sigma:.1f}%)"
    if own is not None:
        if floor is not None and own < floor:
            return floor, f"scenario floor {floor:.1f}% (own n={len(vals)} claimed {own:.1f}% — too quiet to believe)"
        return own, f"own replicates n={len(vals)}"
    if fam_sigma is not None:
        return fam_sigma, f"measured family sigma ({fam.get('n_replicates')} replicates)"
    if floor is not None:
        return floor, f"scenario floor {floor:.1f}% (no family entry)"
    return None, "UNMEASURED"


def check_cell(cell: Path, reg: dict) -> dict:
    rec = json.loads((cell / "record.json").read_text())
    ident = rec.get("identity", {}) or {}
    res = rec.get("result", {}) or {}
    metric = res.get("metric")
    tol = res.get("tolerance_pct")
    if tol in (None, 0):
        tol = 5.0
    rhash = (rec.get("fingerprint", {}) or {}).get("recipe_hash")
    try:
        rel = str(cell.relative_to(ROOT))
    except ValueError:  # cell outside the repo (selftest fixture)
        rel = str(cell)

    vals, dropped = ledger_values(cell, metric, rhash)
    pool = ps.pooled(vals)
    n = pool["n"] if pool else 0
    sigma, prov = resolve_sigma(rel, ident, metric, vals, reg)

    row = {
        "cell": rel,
        "metric": metric,
        "status": ident.get("status"),
        "value": res.get("value"),
        "n": n,
        "dropped_hash_mismatch": dropped,
        "tolerance_pct": tol,
        "sigma_pct": sigma,
        "sigma_source": prov,
        "pooled": pool,
        "grandfathered": rel in GRANDFATHERED,
    }

    if sigma is None:
        row["verdict"] = "OK" if n >= 3 else "UNKNOWN_SIGMA"
        row["required_n"] = None
        row["detail"] = "no measured sigma for this family and too few own replicates to derive one"
        return row

    need = ps.required_n(sigma, tol)
    row["required_n"] = need
    row["single_run_interval_pct"] = ps.single_run_interval_pct(sigma)

    if sigma <= tol:
        row["verdict"] = "QUIET"
        row["detail"] = f"sigma {sigma:.1f}% <= tolerance {tol:g}% — a single run is defensible"
    elif n == 0:
        row["verdict"] = "SINGLE_RUN"
        row["detail"] = "headline has NO replicate in the ledger at this recipe_hash"
    elif n == 1:
        row["verdict"] = "SINGLE_RUN"
        row["detail"] = (
            f"single-run headline against sigma {sigma:.1f}%: that one run carries a "
            f"+/-{row['single_run_interval_pct']:.1f}% 95% interval vs a +/-{tol:g}% gate; "
            f"need n={need}"
        )
    elif n < need:
        row["verdict"] = "UNDERPOWERED"
        row["detail"] = f"n={n} < required n={need} at sigma {sigma:.1f}% / tolerance {tol:g}%" + (
            f"; pooled CI is +/-{pool['ci_half_pct']:.1f}%" if pool and pool.get("ci_half_pct") else ""
        )
    else:
        row["verdict"] = "OK"
        row["detail"] = f"n={n} >= required n={need}; pooled CI +/-{pool['ci_half_pct']:.1f}%"
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="fail on grandfathered cells too — shows the real backlog")
    ap.add_argument("--json", action="store_true", help="emit machine-readable rows")
    args = ap.parse_args()

    reg = load_registry()
    rows = [check_cell(c.parent, reg) for c in sorted((ROOT / "recipes").rglob("record.json"))]

    if args.json:
        print(json.dumps(rows, indent=1, default=str))
    else:
        print("POOLED-PUBLICATION GATE — a headline must be a mean of N with a CI, never one run")
        print("=" * 118)
        print(f"  {'verdict':14s} {'n':>3s} {'need':>4s} {'sigma%':>7s} {'tol%':>5s}  cell")
        print("-" * 118)
        for r in sorted(rows, key=lambda r: (r["verdict"] == "OK", r["cell"])):
            mark = "  " if r["verdict"] in ("OK", "QUIET") else ("GF" if r["grandfathered"] else "!!")
            sig = f"{r['sigma_pct']:.1f}" if r["sigma_pct"] is not None else "  ?  "
            need = str(r["required_n"]) if r["required_n"] else "-"
            print(f"{mark} {r['verdict']:14s} {r['n']:3d} {need:>4s} {sig:>7s} {r['tolerance_pct']:5g}  {r['cell']}")
            if r["verdict"] not in ("OK", "QUIET"):
                print(f"     {r['detail']}")
            if r["dropped_hash_mismatch"]:
                print(f"     note: {r['dropped_hash_mismatch']} ledger run(s) ignored — different recipe_hash")

    ADVISORY = ("UNKNOWN_SIGMA",)  # no MEASURED sigma -> cannot refuse on measured noise
    bad = [r for r in rows if r["verdict"] not in ("OK", "QUIET")]
    advisory = [r for r in bad if r["verdict"] in ADVISORY]
    blocking = [r for r in bad if r["verdict"] not in ADVISORY and (args.strict or not r["grandfathered"])]
    for r in advisory:
        print(
            f"ADVISORY: {r['cell']} has no MEASURED run-to-run sigma "
            f"(family {family_key(json.loads((ROOT / r['cell'] / 'record.json').read_text()).get('identity', {}), r['metric'])}). "
            f"Its headline cannot be certified either way — add the family to analysis/measured_sigma.json."
        )
    print()
    if blocking:
        print(f"FAIL: {len(blocking)} cell(s) publish a headline their measured noise cannot support.")
        for r in blocking:
            print(f"  - {r['cell']}: {r['verdict']} ({r['detail']})")
        if not args.strict:
            print("  (add a cell to GRANDFATHERED in this file only with a reviewed reason)")
        return 1
    if bad:
        print(
            f"PASS (with {len(bad)} cell(s) below the bar: "
            f"{len(advisory)} advisory / {len(bad) - len(advisory)} grandfathered — run --strict for the backlog)"
        )
    else:
        print(f"PASS: all {len(rows)} published headline(s) are supported by their replicate count")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
