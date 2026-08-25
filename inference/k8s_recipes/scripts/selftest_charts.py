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

"""selftest_charts.py — offline guards for the deterministic goal-keyed chart renderer + its data path.

No cluster, no network. Covers the four deliverables:
  A. RENDERER DISPATCH + DETERMINISM — each (scenario, goal) dispatches to the right ASCII renderer;
     identical committed data yields byte-identical ASCII; PNG is byte-stable across re-runs OR cleanly
     skipped when matplotlib is unavailable (ASCII always emits).
  B. PER-POINT VARIANCE HONESTY — a rung with repeat stats (n≥2) draws an error band; a rung with n=1
     draws a plain point with NO fabricated band.
  C. DATA-MODEL EXTENSION — merge_rung_repeats / consolidate_rungs aggregate per-rung across legs
     (mean/min/max/spread/n), additive + back-compat (n<2 → no `repeats` block).
  D. LINK-GATING — the build_catalog RESULTS link is emitted only for cells with data (status ≥ runs),
     hiding empty/WIP cells; and the committed README matrix reflects that.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import build_catalog  # noqa: E402
import charts  # noqa: E402
import export_record  # noqa: E402
import repro_consolidate  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── fixtures: synthetic records, one per lane (no filesystem/committed-data dependency) ───────────────
SLA_REC = {
    "identity": {"scenario": "llm-perf", "goal": "max-concurrency-sla"},
    "result": {
        "metric": "max_concurrency_at_sla",
        "value": 256,
        "unit": "concurrency",
        "value_source": "highest_sla_passing_rung",
        "sla": {"ttft_ms": 10000, "tpot_ms": 100, "stat": "p50"},
        "crossing": {
            "bracket": [256, 384],
            "ratio": 1.5,
            "binding": "TPOT",
            "status": "ok",
            "note": "bracket [256,384] (1.50×), binding=TPOT",
        },
    },
    "detail": {
        "rungs": [
            # 256: TTFT 910 ≤ 10000 AND TPOT 94.4 ≤ 100 → GREEN (both-pass, the max passing rung)
            {
                "concurrency": 256,
                "tpot_p50_ms": 94.4,
                "ttft_p50_ms": 910.0,
                "tps_per_gpu": 145.8,
                "tps_per_user": 10.6,
            },
            # 384/512: TTFT still ≤ 10000 but TPOT > 100 → RED (one-fails: TPOT)
            {
                "concurrency": 384,
                "tpot_p50_ms": 115.2,
                "ttft_p50_ms": 1488.0,
                "tps_per_gpu": 149.0,
                "tps_per_user": 9.0,
            },
            {
                "concurrency": 512,
                "tpot_p50_ms": 125.2,
                "ttft_p50_ms": 1886.0,
                "tps_per_gpu": 144.0,
                "tps_per_user": 7.0,
            },
            # 1024: TTFT 12000 > 10000 AND TPOT 210 > 100 → DARK-RED (both-fail)
            {
                "concurrency": 1024,
                "tpot_p50_ms": 210.0,
                "ttft_p50_ms": 12000.0,
                "tps_per_gpu": 120.0,
                "tps_per_user": 4.0,
            },
        ]
    },
}
PARETO_REC = {
    "identity": {"scenario": "llm-perf", "goal": "pareto"},
    "result": {
        "metric": "pareto_geomean",
        "value": 53.7,
        "unit": "geomean(tps/gpu·tps/user)",
        "per_point": [
            {"concurrency": 32, "g": 81.7469},
            {"concurrency": 256, "g": 42.1663},
        ],
    },
    "detail": {
        "rungs": [
            {"concurrency": 32, "tps_per_gpu": 97.2, "tps_per_user": 68.8},
            {"concurrency": 256, "tps_per_gpu": 153.2, "tps_per_user": 11.6},
        ]
    },
}
# ── A. dispatch ───────────────────────────────────────────────────────────────────────────────────────
check(
    "dispatch llm-perf·max-concurrency-sla → SLA renderer",
    charts._dispatch("llm-perf", "max-concurrency-sla") is charts._ascii_sla,
)
check(
    "dispatch llm-perf·pareto → pareto renderer",
    charts._dispatch("llm-perf", "pareto") is charts._ascii_pareto,
)
check(
    "dispatch unknown (scenario,goal) → no renderer (graceful)",
    charts._dispatch("llm-perf", "no-such-goal") is None,
)
check(
    "render_ascii of unknown lane returns a note, not a crash",
    charts.render_ascii({"identity": {"scenario": "x", "goal": "y"}, "detail": {"rungs": [{}]}}, []).startswith("_("),
)
check(
    "render_ascii of empty rungs returns a note",
    "no rungs"
    in charts.render_ascii(
        {
            "identity": {"scenario": "llm-perf", "goal": "pareto"},
            "detail": {"rungs": []},
        },
        [],
    ),
)

# ── A. determinism (ASCII byte-identical across re-renders) ────────────────────────────────────────────
for name, rec in (("sla", SLA_REC), ("pareto", PARETO_REC)):
    a1 = charts.render_ascii(rec, [])
    a2 = charts.render_ascii(rec, [])
    check(f"ASCII deterministic ({name}) — identical bytes on re-render", a1 == a2)
    check(
        f"ASCII ({name}) is a fenced code block",
        a1.startswith("```text") and a1.rstrip().endswith("```"),
    )

# lane-specific content sanity — the 2D SLA zone view
sla_ascii = charts.render_ascii(SLA_REC, [])
check(
    "SLA chart states BOTH limits in the header (TTFT + TPOT)",
    "TTFT p50 ≤ 10000 ms" in sla_ascii and "TPOT p50 ≤ 100 ms" in sla_ascii,
)
check(
    "SLA chart shows per-rung TTFT and TPOT",
    "TTFT" in sla_ascii and "TPOT" in sla_ascii and "ms" in sla_ascii,
)
check("SLA zone: a both-pass rung reads ✅ both-pass", "✅ both-pass" in sla_ascii)
check("SLA zone: a one-fail rung names the failing axis", "⚠ one-fails: TPOT" in sla_ascii)
check("SLA zone: a both-fail rung reads ❌ both-fail", "❌ both-fail" in sla_ascii)
check("SLA chart highlights the max passing concurrency", "⟵ max passing" in sla_ascii)

# _sla_zone classifier: the shared implementation for both ASCII + PNG zones
zpass, fpass = charts._sla_zone(900.0, 90.0, 10000, 100)
check("_sla_zone both-pass → ('pass', [])", zpass == "pass" and fpass == [])
zone1, f1 = charts._sla_zone(900.0, 120.0, 10000, 100)
check("_sla_zone one-fail (TPOT) → ('one', ['TPOT'])", zone1 == "one" and f1 == ["TPOT"])
zone1b, f1b = charts._sla_zone(12000.0, 90.0, 10000, 100)
check("_sla_zone one-fail (TTFT) → ('one', ['TTFT'])", zone1b == "one" and f1b == ["TTFT"])
zboth, fboth = charts._sla_zone(12000.0, 120.0, 10000, 100)
check(
    "_sla_zone both-fail → ('both', ['TTFT','TPOT'])",
    zboth == "both" and fboth == ["TTFT", "TPOT"],
)
zna, fna = charts._sla_zone(None, 90.0, 10000, 100)
check(
    "_sla_zone missing value → ('unknown', []) (never guessed)",
    zna == "unknown" and fna == [],
)

# ── A. interpolated SLA crossing — re-projected from result.crossing (NOT recomputed with a new method) ──
cp = charts._crossing_point(SLA_REC)
check(
    "_crossing_point echoes committed status + binding",
    cp.get("status") == "ok" and cp.get("binding") == "TPOT",
)
check(
    "_crossing_point interpolates conc between the bracket rungs (~290)",
    cp.get("conc") is not None and 289.0 < cp["conc"] < 292.0,
    str(cp.get("conc")),
)
check(
    "_crossing_point marker sits ON the binding (TPOT) limit line (y == tpot_limit)",
    cp.get("y") == 100 and cp.get("x") is not None,
)
# the reprojection uses the IDENTICAL linear formula as exemplar_check._interp (same method, not a new one)
_spec = importlib.util.spec_from_file_location("exemplar_check", ROOT / "analysis/llm-perf/exemplar_check.py")
_exch = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_exch)
check(
    "_crossing_point conc == exemplar_check._interp (identical linear interpolation)",
    abs(cp["conc"] - _exch._interp(256, 94.4, 384, 115.2, 100)) < 1e-9,
)

sla_ascii = charts.render_ascii(SLA_REC, [])
check(
    "SLA header distinguishes verified last-pass floor vs interpolated crossing",
    "verified last-pass rung" in sla_ascii and "interp crossing" in sla_ascii,
)
check(
    "SLA header shows the crossing bracket + binding + committed note (honest caveat)",
    "SLA crossing ≈ c290" in sla_ascii
    and "bracket [256–384]" in sla_ascii
    and "binding TPOT" in sla_ascii
    and "note:" in sla_ascii,
)

# binding = TTFT → the crossing marker sits on the VERTICAL TTFT limit line (x == ttft_limit)
TTFT_REC = json.loads(json.dumps(SLA_REC))
TTFT_REC["result"]["sla"] = {"ttft_ms": 1000, "tpot_ms": 100, "stat": "p50"}
TTFT_REC["result"]["crossing"] = {
    "bracket": [256, 384],
    "ratio": 1.5,
    "binding": "TTFT",
    "status": "ok",
    "note": "bracket [256,384] (1.50×), binding=TTFT",
}
cpt = charts._crossing_point(TTFT_REC)
check(
    "_crossing_point binding=TTFT → marker ON the vertical TTFT limit (x == ttft_limit)",
    cpt.get("x") == 1000 and cpt.get("y") is not None,
)

# no clean bracket (status != 'ok') → NO bogus marker; ASCII says so; PNG still renders (graceful)
NOBRK_REC = json.loads(json.dumps(SLA_REC))
NOBRK_REC["result"]["crossing"] = {
    "status": "no_bracket",
    "note": "no clean pass→fail bracket (non-monotonic curve?)",
}
cpn = charts._crossing_point(NOBRK_REC)
check(
    "_crossing_point no-clean-bracket → geometry omitted (no bogus point drawn)",
    cpn.get("status") == "no_bracket" and "conc" not in cpn and "x" not in cpn,
)
nobrk_ascii = charts.render_ascii(NOBRK_REC, [])
check(
    "SLA ASCII with no clean bracket says so (no marker implied)",
    "no clean bracket" in nobrk_ascii and "no crossing marker drawn" in nobrk_ascii,
)

# _sla_png_caveat: coarse bracket (>1.5×) adds the ⚠ caveat; a fine bracket (≤1.5×) does not
check(
    "_sla_png_caveat coarse (>1.5×) → ⚠ approximate caveat",
    "⚠ coarse bracket"
    in charts._sla_png_caveat(
        {
            "status": "ok",
            "conc": 216,
            "bracket": [128, 256],
            "ratio": 2.0,
            "binding": "TPOT",
        }
    ),
)
check(
    "_sla_png_caveat fine (≤1.5×) → NO ⚠ caveat",
    "⚠"
    not in charts._sla_png_caveat(
        {
            "status": "ok",
            "conc": 290,
            "bracket": [256, 384],
            "ratio": 1.5,
            "binding": "TPOT",
        }
    ),
)
if charts._mpl() is not None:
    _p = Path(tempfile.mkdtemp()) / "nb.png"
    check(
        "PNG renders the crossing marker without crash (both-pass→fail bracket)",
        charts.render_png(SLA_REC, Path(tempfile.mkdtemp()) / "cx.png", []) is True,
    )
    check(
        "PNG still renders with a no-clean-bracket crossing (no marker, no crash)",
        charts.render_png(NOBRK_REC, _p, []) is True,
    )

check(
    "pareto chart annotates the pareto_geomean",
    "pareto_geomean = 53.7" in charts.render_ascii(PARETO_REC, []),
)

# ── A. PNG byte-stability OR clean skip ────────────────────────────────────────────────────────────────
have_mpl = charts._mpl() is not None
if have_mpl:
    hs = []
    for i in range(2):
        p = Path(tempfile.mkdtemp()) / "c.png"
        ok = charts.render_png(SLA_REC, p, [])
        check(f"PNG render #{i} succeeds with matplotlib", ok)
        hs.append(hashlib.sha256(p.read_bytes()).hexdigest())
    check(
        "PNG byte-stable across re-runs (same data → same bytes)",
        hs[0] == hs[1],
        f"{hs[0][:12]} vs {hs[1][:12]}",
    )
    # non-chartable lane → False (no PNG), ASCII still fine
    p = Path(tempfile.mkdtemp()) / "x.png"
    check(
        "PNG render returns False for a lane with no renderer",
        charts.render_png(
            {"identity": {"scenario": "x", "goal": "y"}, "detail": {"rungs": [{}]}},
            p,
            [],
        )
        is False,
    )
else:
    print("  (matplotlib unavailable — exercising the best-effort SKIP path)")
    p = Path(tempfile.mkdtemp()) / "c.png"
    check(
        "PNG cleanly skipped (returns False) without matplotlib",
        charts.render_png(SLA_REC, p, []) is False,
    )
    check(
        "ASCII still emits without matplotlib",
        charts.render_ascii(SLA_REC, []).startswith("```text"),
    )

# ── B. per-point variance honesty (n≥2 band vs n=1 plain) ─────────────────────────────────────────────
BANDED = json.loads(json.dumps(SLA_REC))
BANDED["detail"]["rungs"][0]["repeats"] = {"tpot_p50_ms": {"mean": 94.4, "min": 90.1, "max": 98.9, "n": 3}}
banded_ascii = charts.render_ascii(BANDED, [])
check(
    "n≥2 rung draws an error band (n=3 annotation)",
    "n=3" in banded_ascii and "90.1" in banded_ascii,
)
check("n=1 rung draws NO fabricated band", "n=1" not in banded_ascii)
# _rung_stat honesty
v, lo, hi, n = charts._rung_stat({"tpot_p50_ms": 94.4}, "tpot_p50_ms")
check("_rung_stat n=1 → no lo/hi band", n == 1 and lo is None and hi is None)
v, lo, hi, n = charts._rung_stat(
    {
        "tpot_p50_ms": 94.4,
        "repeats": {"tpot_p50_ms": {"mean": 94.4, "min": 90.1, "max": 98.9, "n": 3}},
    },
    "tpot_p50_ms",
)
check("_rung_stat n≥2 → lo/hi band from repeats", n == 3 and lo == 90.1 and hi == 98.9)

# ── C. data-model extension: merge_rung_repeats / consolidate_rungs ────────────────────────────────────
leg_a = [
    {"concurrency": 256, "tpot_p50_ms": 94.0},
    {"concurrency": 384, "tpot_p50_ms": 115.0},
]
leg_b = [
    {"concurrency": 256, "tpot_p50_ms": 96.0},
    {"concurrency": 384, "tpot_p50_ms": 117.0},
]
leg_c = [{"concurrency": 256, "tpot_p50_ms": 92.0}]  # only rung 256 present in this leg
merged = export_record.merge_rung_repeats(leg_a, [leg_a, leg_b, leg_c], ["tpot_p50_ms"])
r256 = next(r for r in merged if r["concurrency"] == 256)
r384 = next(r for r in merged if r["concurrency"] == 384)
check(
    "merge: rung with 3 legs gets repeats n=3",
    r256.get("repeats", {}).get("tpot_p50_ms", {}).get("n") == 3,
)
check(
    "merge: repeats mean/min/max correct",
    r256["repeats"]["tpot_p50_ms"]["min"] == 92.0
    and r256["repeats"]["tpot_p50_ms"]["max"] == 96.0
    and abs(r256["repeats"]["tpot_p50_ms"]["mean"] - 94.0) < 1e-9,
)
check(
    "merge: rung with only 2 legs still gets a band (n=2)",
    r384.get("repeats", {}).get("tpot_p50_ms", {}).get("n") == 2,
)
# n=1 → NO repeats block (back-compat)
solo = export_record.merge_rung_repeats(leg_a, [leg_a], ["tpot_p50_ms"])
check(
    "merge: single leg → NO repeats block (honest n=1)",
    all("repeats" not in r for r in solo),
)
check("merge: inputs not mutated", "repeats" not in leg_a[0])

# consolidate_rungs from committed-style record dicts
orig = json.loads(json.dumps(SLA_REC))
copy1 = json.loads(json.dumps(SLA_REC))
copy1["detail"]["rungs"][0]["tpot_p50_ms"] = 96.0
copy2 = json.loads(json.dumps(SLA_REC))
copy2["detail"]["rungs"][0]["tpot_p50_ms"] = 92.0
cons = repro_consolidate.consolidate_rungs(orig, [copy1, copy2])
check(
    "consolidate_rungs overlays repeats onto original",
    cons is not None and cons["detail"]["rungs"][0].get("repeats", {}).get("tpot_p50_ms", {}).get("n") == 3,
)
check("consolidate_rungs is non-mutating", "repeats" not in orig["detail"]["rungs"][0])
check(
    "consolidate_rungs with no copies → None (leaves record untouched)",
    repro_consolidate.consolidate_rungs(orig, []) is None,
)

# ── D. link-gating (data-gated RESULTS link) ──────────────────────────────────────────────────────────
IDX = build_catalog.STATUS_ORDER.index("runs")


def has_link(status):
    return build_catalog.rank(status) >= IDX


check("gate: 'planned' → no RESULTS link", not has_link("planned"))
check("gate: 'wip' → no RESULTS link", not has_link("wip"))
check("gate: alias 'deployable'(→wip) → no RESULTS link", not has_link("deployable"))
check("gate: 'runs' → RESULTS link", has_link("runs"))
check(
    "gate: 'performant'/'exemplar' → RESULTS link",
    has_link("performant") and has_link("exemplar"),
)

# The generated matrix links by recipe PATH, not envelope name or leaf basename. GLM intentionally has
# envelope names that differ from the directory, and multiple workload folders can share the same leaf
# basename; either shortcut creates false positives. Assert the gate against each unique relative path.
readme = (ROOT / "README.md").read_text()
import yaml as _yaml  # noqa: E402

_mstart, _mend = readme.find("<!-- MATRIX:START -->"), readme.find("<!-- MATRIX:END -->")
_matrix = readme[_mstart:_mend] if (_mstart >= 0 and _mend > _mstart) else readme
_viol = []
for _rp in sorted((ROOT / "recipes").glob("**/recipe.yaml")):
    _env = (_yaml.safe_load(_rp.read_text()) or {}).get("envelope") or {}
    _rel = _rp.parent.relative_to(ROOT).as_posix()
    _linked = f"{_rel}/RESULTS.md)" in _matrix
    _pub = has_link(_env.get("status"))
    if _linked != _pub:
        _viol.append(f"{_rel}: status={_env.get('status')} linked={_linked}")
check(
    "README matrix: a RESULTS link appears iff the cell is published (gate invariant, all cells)",
    not _viol,
    "; ".join(_viol),
)

print(("\nFAIL: " + ", ".join(fails)) if fails else "\nselftest_charts: all checks passed")
sys.exit(1 if fails else 0)
