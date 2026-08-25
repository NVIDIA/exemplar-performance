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

"""compare.py <cell...> | --facet <scenario> <goal> <distribution> | --all | --runs <cell> | --repro <cell> [options] — comparison report.

  --runs <cell>   ONE cell run on ≥2 clusters (from its runs.jsonl ledger): same-cell cross-cluster
                  reproducibility — each cluster's value, the spread, and a within-tolerance verdict. This is
                  the same-hardware, different-cluster check (e.g. GB200 cluster-A vs cluster-B) that the
                  per-cell record.json can't show (publish overwrites it). Keeps the LATEST run per cluster.

  --repro <cell>  ONE cell run N times on the SAME cluster (from runs.jsonl): within-cluster reproducibility —
                  every repeat's value, the mean, spread%, and a within-tolerance verdict per cluster. This is
                  the 3×-repeat check the fleet agents produce; --runs can't see it (it collapses to 1/cluster).

  --all   the digest: every (scenario·goal·distribution) facet with ≥2 comparable cells, one section each.

The "is Cluster A within X% of B / what's the ranking" table for ONE metric across cells. Reads each cell's
committed exemplar (envelope.exemplar.{metric,reference,tolerance_pct}) + gpu_type; if the reference isn't set,
uses the latest clustered value from runs.jsonl. Legacy cells with no run ledger fall back to record.json's
result.value. Two modes, chosen automatically:

  - REPRODUCIBILITY (all cells the SAME gpu_type): compare each against a baseline bar; PASS if within
    tolerance. This is the real same-hardware reproducibility check (cluster A versus cluster B).
  - HARDWARE (mixed gpu_types): rank by value + % of the leader. A hardware comparison, NOT pass/fail —
    B200 vs GB300 is different silicon (exemplar_check --actual-gpu enforces that distinction elsewhere).

  compare.py --facet llm-perf max-concurrency-sla example-long-context-trace
  compare.py recipes/.../a-pareto recipes/.../b-pareto recipes/.../c-pareto
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
try:
    import yaml
except ImportError:
    sys.exit("compare: requires pyyaml")

# higher_better per metric (both scenarios). min_cost_at_target_score is the only lower-is-better one.
HIGHER_BETTER = {
    "max_concurrency_at_sla": True,
    "tps_per_gpu_at_sla": True,
    "pareto_geomean": True,
    "max_goodput": True,
    "min_cost_at_target_score": False,
    "net_behavior_score": True,
}


def _ec():
    """Load exemplar_check (dashed path → not importable by name) for the point-by-point pareto compare."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("exemplar_check", ROOT / "analysis/llm-perf/exemplar_check.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _per_point(cell: Path) -> list:
    """The committed per-rung geomeans (record.json result.per_point) for a pareto cell, or []."""
    rec = cell / "record.json"
    if not rec.is_file():
        return []
    return ((json.loads(rec.read_text()).get("result") or {}).get("per_point")) or []


def opt(flag, argv, default=None):
    return argv[argv.index(flag) + 1] if flag in argv and argv.index(flag) + 1 < len(argv) else default


def scope_to_run_ids(runs: list[dict], spec: str | None) -> list[dict]:
    """Keep only the runs whose run_id is in `spec` (space/comma separated). This scopes a reproducibility
    band to ONE sweep's own legs instead of banding the cell's WHOLE same-cluster history (D6): a variance
    sweep of N repeats must report the spread of exactly those N legs, not of unrelated earlier runs that
    happen to share the cluster. Empty/None spec → unchanged (band over everything, the legacy behaviour).
    """
    if not spec:
        return runs
    want = {x for x in spec.replace(",", " ").split() if x}
    return [r for r in runs if r.get("run_id") in want]


def latest_ledger_value(cell: Path) -> tuple[bool, object | None]:
    """Return (ledger_exists, latest clustered value).

    A present runs.jsonl is the append-only source of truth for fleet analysis. If it exists but has no valued
    clustered rows, the cell has no comparable value yet; do not fall back to a possibly stale record.json.
    """
    jl = cell / "runs.jsonl"
    if not jl.is_file():
        return False, None
    latest = None
    for raw in jl.read_text().splitlines():
        if not raw.strip():
            continue
        try:
            row = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if row.get("value") is not None and (row.get("cluster") or "").strip():
            latest = row.get("value")
    return True, latest


def valued_clustered_runs(runs: list[dict]) -> list[dict]:
    """Rows usable for compare math: metric value present and cluster label present."""
    return [r for r in runs if r.get("value") is not None and (r.get("cluster") or "").strip()]


def load_cell(cell: Path) -> dict | None:
    rp = cell / "recipe.yaml"
    if not rp.is_file():
        return None
    r = yaml.safe_load(rp.read_text()) or {}
    env = r.get("envelope") or {}
    ex = env.get("exemplar") or {}
    value = ex.get("reference")
    if value is None:
        has_ledger, ledger_value = latest_ledger_value(cell)
        if has_ledger:
            value = ledger_value
        else:
            rec = cell / "record.json"  # legacy fallback for cells published before run ledgers
            if rec.is_file():
                value = (json.loads(rec.read_text()).get("result") or {}).get("value")
    try:
        path = str(cell.relative_to(ROOT))
    except ValueError:
        path = str(cell)  # cell outside the repo (e.g. a test tmpdir)
    return {
        "cell": env.get("name") or cell.name,
        "path": path,
        "gpu": env.get("gpu_type"),
        "metric": ex.get("metric"),
        "value": value,
        "unit": ex.get("unit"),
        "tol": ex.get("tolerance_pct", 5),
        "status": env.get("status"),
        "per_point": _per_point(cell),
    }  # pareto_geomean: the per-rung geomeans for the point-by-point compare


def facet_cells(scenario, goal, distribution) -> list[Path]:
    out = []
    for rp in sorted((ROOT / "recipes").glob("**/recipe.yaml")):
        env = (yaml.safe_load(rp.read_text()) or {}).get("envelope") or {}
        if (
            env.get("scenario") == scenario
            and (env.get("goal") or "") == goal
            and env.get("distribution") == distribution
        ):
            out.append(rp.parent)
    return out


def _pareto_point_report(have, baseline, metric, tol, gpus, rows):
    """Point-by-point pareto comparison: each cell's per-rung geomeans vs the baseline's, gated at ±tol per
    shared rung (rung sets must MATCH). Reports a per-point diff table + worst% + verdict per cell, plus the
    scalar pareto_geomean (display only). Returns (lines, obj)."""
    ec = _ec()
    reproducibility = len(gpus) == 1
    mode = (
        "REPRODUCIBILITY (all " + next(iter(gpus)) + ")"
        if reproducibility
        else "HARDWARE (mixed GPUs: " + ", ".join(sorted(str(g) for g in gpus)) + ")"
    )
    lines = [
        f"### metric `{metric}` (point-by-point; exemplar iff every shared rung ≤ {tol}%)  ·  unit `{have[0]['unit']}`",
        f"_mode: **{mode}** · baseline **{baseline['cell']}** (scalar {baseline['value']}) · "
        f"gate: rung sets MATCH AND max per-point |Δ| ≤ {tol}%_",
        "",
    ]
    cmp_objs = []
    for r in have:
        if r is baseline:
            lines += [
                f"**{r['cell']}** — baseline · scalar `{metric}` = {r['value']}",
                "",
            ]
            continue
        cmp = ec.pareto_point_compare(r.get("per_point") or [], baseline.get("per_point") or [], tol)
        icon = {
            "exemplar": "✅ exemplar",
            "outside": "❌ outside",
            "not-comparable": "⚠ not-comparable",
        }[cmp["verdict"]]
        note = "" if cmp["matched"] else " · ⚠ rung sets DIFFER (not comparable — never compared on a subset)"
        lines += [
            f"**{r['cell']}** vs baseline — {icon} (worst {cmp['worst_pct']}%){note} · scalar `{metric}` = {r['value']}",
            "",
            "| concurrency | this g | baseline g | Δ% |",
            "|--:|--:|--:|--:|",
        ]
        for p in cmp["per_point"]:
            dp = "—" if p["diff_pct"] is None else f"{p['diff_pct']:+.2f}%"
            lines.append(f"| {p['concurrency']:g} | {p['a']} | {p['b']} | {dp} |")
        lines.append("")
        cmp_objs.append(
            {
                "cell": r["cell"],
                "verdict": cmp["verdict"],
                "worst_pct": cmp["worst_pct"],
                "matched": cmp["matched"],
                "per_point": cmp["per_point"],
            }
        )
    missing = [r["cell"] for r in rows if r["value"] is None or not r.get("per_point")]
    if missing:
        lines += [f"_no per-point data (not compared): {', '.join(missing)}_"]
    obj = {
        "metric": metric,
        "mode": "reproducibility" if reproducibility else "hardware",
        "comparison": "point-by-point",
        "tolerance_pct": tol,
        "baseline": baseline["cell"],
        "comparisons": cmp_objs,
    }
    return lines, obj


def report(rows, tol_override=None, base_name=None):
    """Build the comparison (lines, json_obj) for a set of loaded cells. Returns (None, None) if <2 have values."""
    have = [r for r in rows if r["value"] is not None]
    if len(have) < 2:
        return None, None
    # Guard: a comparison is only meaningful for ONE metric. Explicitly-listed cells can mix metrics (e.g. a
    # pareto cell + an eval-opt cell), and since min_cost_at_target_score is LOWER-better while the rest are
    # HIGHER-better, silently ranking them together would invert the loser and the winner. Refuse loudly.
    metrics = {r.get("metric") for r in have if r.get("metric")}
    if len(metrics) > 1:
        return (
            [
                f"### ⚠ refusing to compare — the selected cells mix metrics {sorted(metrics)}.",
                "_A single comparison must hold ONE metric (they have different units AND directions — "
                "min_cost_at_target_score is lower-better, the rest higher-better). Select cells that share "
                "a metric, or use `--facet <scenario> <goal> <distribution>` which groups by goal._",
            ],
            None,
        )
    metric = have[0]["metric"]
    hib = HIGHER_BETTER.get(metric, True)
    tol = float(tol_override) if tol_override is not None else float(have[0]["tol"])
    gpus = {r["gpu"] for r in have}
    reproducibility = len(gpus) == 1
    best = (max if hib else min)(have, key=lambda r: r["value"])
    baseline = (
        next((r for r in have if base_name and base_name in (r["cell"], r["path"])), None)
        or next((r for r in have if r["status"] == "exemplar"), None)
        or best
    )

    # pareto_geomean: the exemplar verdict is POINT-BY-POINT (not the scalar ±tol). For each cell, compare its
    # per-rung geomeans against the baseline's via exemplar_check.pareto_point_compare — rung sets must MATCH and
    # every shared rung must be within tol; report the per-point diff table + worst% + verdict.
    if metric == "pareto_geomean" and any(r.get("per_point") for r in have):
        return _pareto_point_report(have, baseline, metric, tol, gpus, rows)

    order = sorted(have, key=lambda r: r["value"], reverse=hib)

    lines = [f"### metric `{metric}` ({'higher' if hib else 'lower'} = better)  ·  unit `{have[0]['unit']}`"]
    if reproducibility:
        b = baseline["value"]
        lines += [
            f"_mode: **REPRODUCIBILITY** (all {next(iter(gpus))}) · baseline **{baseline['cell']}** = {b} · tolerance ±{tol}%_",
            "",
            "| cell | value | Δ% vs baseline | verdict |",
            "|---|--:|--:|:--|",
        ]
        for r in order:
            d = (r["value"] - b) / b * 100 if b else 0.0
            verdict = "◦ baseline" if r is baseline else ("✅ within" if abs(d) <= tol else "❌ outside")
            lines.append(f"| {r['cell']} | {r['value']} | {d:+.1f}% | {verdict} |")
    else:
        lead = best["value"]
        lines += [
            f"_mode: **HARDWARE** (mixed GPUs: {', '.join(sorted(gpus))}) · leader **{best['gpu']}** = {lead} · ranking, not pass/fail_",
            "",
            "| rank | cell | gpu | value | % of leader |",
            "|--:|---|---|--:|--:|",
        ]
        for i, r in enumerate(order, 1):
            pct = r["value"] / lead * 100 if lead else 0.0
            lines.append(f"| {i} | {r['cell']} | {r['gpu']} | {r['value']} | {pct:.1f}% |")
    missing = [r["cell"] for r in rows if r["value"] is None]
    if missing:
        lines += ["", f"_not yet run (no value): {', '.join(missing)}_"]
    obj = {
        "metric": metric,
        "higher_better": hib,
        "mode": "reproducibility" if reproducibility else "hardware",
        "tolerance_pct": tol,
        "cells": rows,
    }
    return lines, obj


def runs_cross_cluster(runs, metric, tol, higher_better=True):
    """Same-cell CROSS-CLUSTER comparison from the runs.jsonl ledger (each entry: cluster, value, recipe_hash,
    run_id, date). One cell run on ≥2 clusters — the reproducibility question record.json can't answer (publish
    overwrites it, so the per-run history lives in the ledger). Latest run per cluster; reports each value, the
    spread, and whether the clusters agree within ±tol. Flags mixed recipe_hash (different setups → NOT a pure
    reproducibility check). Pure. Returns (lines, obj) or (None, None) if <2 clusters carry a value.
    """
    by_cluster: dict = {}
    for r in runs:
        if r.get("value") is not None and (r.get("cluster") or "").strip():
            by_cluster[r["cluster"]] = r  # append order = chronological → last wins = latest per cluster
    picked = sorted(by_cluster.values(), key=lambda r: r["value"], reverse=higher_better)
    if len(picked) < 2:
        return None, None
    values = [r["value"] for r in picked]
    mean = sum(values) / len(values)
    spread = (max(values) - min(values)) / mean * 100 if mean else 0.0
    same_setup = len({r.get("recipe_hash") for r in picked if r.get("recipe_hash")}) <= 1
    reproducible = spread <= tol
    verdict = "✅ reproducible across clusters" if reproducible else "❌ clusters DISAGREE"
    mixed = "" if same_setup else " · ⚠ MIXED recipe_hash — setups differ, NOT a pure reproducibility check"
    lines = [
        f"### same-cell cross-cluster · metric `{metric}` ({'higher' if higher_better else 'lower'} = better)",
        f"_{len(picked)} clusters · spread **{spread:.1f}%** (tolerance ±{tol}%) → {verdict}{mixed}_",
        "",
        "| cluster | value | Δ% vs mean | run_id | recipe_hash |",
        "|---|--:|--:|---|---|",
    ]
    for r in picked:
        d = (r["value"] - mean) / mean * 100 if mean else 0.0
        lines.append(
            f"| {r['cluster']} | {r['value']} | {d:+.1f}% | {r.get('run_id', '?')} | "
            f"`{str(r.get('recipe_hash', ''))[:12]}` |"
        )
    obj = {
        "metric": metric,
        "mode": "cross-cluster",
        "clusters": len(picked),
        "spread_pct": round(spread, 2),
        "tolerance_pct": tol,
        "reproducible": reproducible,
        "same_setup": same_setup,
        "runs": picked,
    }
    return lines, obj


def runs_within_cluster(runs, metric, tol, higher_better=True):
    """Same-cell WITHIN-CLUSTER reproducibility from the runs.jsonl ledger — the N-repeats question
    (an agent runs a cell 3× on ONE cluster; is the metric stable?). `runs_cross_cluster` answers the
    across-clusters question and keeps only the latest run per cluster, so it CANNOT see 3 same-cluster
    repeats. This groups valued runs by cluster; for each cluster with ≥2 runs it reports every run's
    value, the mean, spread% = (max-min)/mean, and a within-tol verdict. Flags mixed recipe_hash within a
    cluster (the repeats weren't the same setup). Pure. Returns (lines, obj) or (None, None) if no cluster
    has ≥2 valued runs."""
    by_cluster: dict = {}
    for r in runs:
        if r.get("value") is not None and (r.get("cluster") or "").strip():
            by_cluster.setdefault(r["cluster"], []).append(r)
    groups = {c: rs for c, rs in by_cluster.items() if len(rs) >= 2}
    if not groups:
        return None, None
    lines = [
        f"### same-cell within-cluster reproducibility · metric `{metric}` "
        f"({'higher' if higher_better else 'lower'} = better)"
    ]
    out_groups = []
    for cluster in sorted(groups):
        rs = groups[cluster]
        values = [r["value"] for r in rs]
        mean = sum(values) / len(values)
        spread = (max(values) - min(values)) / mean * 100 if mean else 0.0
        same_setup = len({r.get("recipe_hash") for r in rs if r.get("recipe_hash")}) <= 1
        reproducible = spread <= tol
        verdict = "✅ reproducible" if reproducible else "❌ NOT reproducible"
        mixed = "" if same_setup else " · ⚠ MIXED recipe_hash within cluster — repeats differ in setup"
        lines += [
            "",
            f"_{cluster}: {len(rs)} runs · mean {mean:.4g} · spread **{spread:.1f}%** "
            f"(tolerance ±{tol}%) → {verdict}{mixed}_",
            "| run_id | value | Δ% vs mean | date | recipe_hash |",
            "|---|--:|--:|---|---|",
        ]
        for r in rs:
            d = (r["value"] - mean) / mean * 100 if mean else 0.0
            lines.append(
                f"| {r.get('run_id', '?')} | {r['value']} | {d:+.1f}% | {r.get('date', '?')} | "
                f"`{str(r.get('recipe_hash', ''))[:12]}` |"
            )
        out_groups.append(
            {
                "cluster": cluster,
                "n": len(rs),
                "mean": mean,
                "spread_pct": round(spread, 2),
                "reproducible": reproducible,
                "same_setup": same_setup,
                "values": values,
            }
        )
    obj = {
        "metric": metric,
        "mode": "within-cluster",
        "tolerance_pct": tol,
        "clusters": out_groups,
    }
    return lines, obj


def all_facets() -> dict:
    """Every (scenario, goal, distribution) → its cells, across the whole collection."""
    seen: dict = {}
    for rp in sorted((ROOT / "recipes").glob("**/recipe.yaml")):
        env = (yaml.safe_load(rp.read_text()) or {}).get("envelope") or {}
        seen.setdefault((env.get("scenario"), env.get("goal") or "", env.get("distribution")), []).append(rp.parent)
    return seen


def main() -> int:
    argv = sys.argv[1:]
    tol_override = opt("--tolerance-pct", argv)
    base_name = opt("--baseline", argv)
    run_ids_spec = opt("--run-ids", argv)  # D6: scope a --repro/--runs band to a sweep's own legs

    if "--all" in argv:  # every facet with ≥2 comparable cells — the digest
        blocks = []
        for (sc, goal, dist), cells in sorted(all_facets().items(), key=lambda kv: tuple(str(x) for x in kv[0])):
            lines, _ = report([c for c in (load_cell(p) for p in cells) if c], tol_override, base_name)
            if lines:
                blocks.append("\n".join([f"## {sc} · {goal or '—'} · {dist}"] + lines))
        if not blocks:
            print("compare --all: no facet has ≥2 cells with values yet")
            return 0
        print(f"# Cross-cluster comparison — {len(blocks)} facet(s) with comparable results\n")
        print("\n\n".join(blocks))
        return 0

    if "--runs" in argv:  # same cell, ≥2 clusters — from its runs.jsonl ledger
        i = argv.index("--runs")
        cell = Path(argv[i + 1]).resolve() if i + 1 < len(argv) else None
        if not cell or not (cell / "recipe.yaml").is_file():
            sys.exit("usage: compare --runs <cell-dir>   # same-cell cross-cluster reproducibility from runs.jsonl")
        env = (yaml.safe_load((cell / "recipe.yaml").read_text()) or {}).get("envelope") or {}
        ex = env.get("exemplar") or {}
        metric = ex.get("metric", "?")
        tol = float(tol_override) if tol_override is not None else float(ex.get("tolerance_pct", 5))
        jl = cell / "runs.jsonl"
        runs = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()] if jl.is_file() else []
        runs = scope_to_run_ids(runs, run_ids_spec)  # D6: optional scoping to a specific set of run_ids
        lines, obj = runs_cross_cluster(runs, metric, tol, HIGHER_BETTER.get(metric, True))
        if lines is None:
            usable = valued_clustered_runs(runs)
            print(
                f"compare --runs: need ≥2 clusters with a value in {cell.name}/runs.jsonl "
                f"(found {len(usable)} valued clustered run(s)) — run the cell on a second cluster + publish, then re-check"
            )
            return 0
        print(json.dumps(obj, indent=2) if "--json" in argv else "\n".join(lines))
        return 0

    if "--repro" in argv:  # same cell, N runs on ONE cluster — within-cluster spread
        i = argv.index("--repro")
        cell = Path(argv[i + 1]).resolve() if i + 1 < len(argv) else None
        if not cell or not (cell / "recipe.yaml").is_file():
            sys.exit("usage: compare --repro <cell-dir>   # within-cluster N-run reproducibility from runs.jsonl")
        env = (yaml.safe_load((cell / "recipe.yaml").read_text()) or {}).get("envelope") or {}
        ex = env.get("exemplar") or {}
        metric = ex.get("metric", "?")
        tol = float(tol_override) if tol_override is not None else float(ex.get("tolerance_pct", 5))
        jl = cell / "runs.jsonl"
        runs = [json.loads(x) for x in jl.read_text().splitlines() if x.strip()] if jl.is_file() else []
        runs = scope_to_run_ids(runs, run_ids_spec)  # D6: band ONLY the sweep's N legs, not the whole history
        lines, obj = runs_within_cluster(runs, metric, tol, HIGHER_BETTER.get(metric, True))
        if lines is None:
            usable = valued_clustered_runs(runs)
            print(
                f"compare --repro: need ≥2 runs with a value on ONE cluster in {cell.name}/runs.jsonl "
                f"(found {len(usable)} valued clustered run(s)) — publish more repeats, then re-check"
            )
            return 0
        print(json.dumps(obj, indent=2) if "--json" in argv else "\n".join(lines))
        return 0

    if "--facet" in argv:
        i = argv.index("--facet")
        cells = facet_cells(*argv[i + 1 : i + 4])
    else:
        cells = [Path(a).resolve() for a in argv if not a.startswith("--")]
    if not cells:
        sys.exit(__doc__)

    lines, obj = report([c for c in (load_cell(p) for p in cells) if c], tol_override, base_name)
    if lines is None:
        print("compare: need ≥2 selected cells with a value (exemplar.reference unset + no record.json)")
        return 0
    print(json.dumps(obj, indent=2) if "--json" in argv else "### Cross-cluster comparison\n" + "\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
