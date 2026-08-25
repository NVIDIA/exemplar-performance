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

"""selftest.py — regression tests for the deterministic tools (no cluster needed). `make test` runs this.

Asserts the invariants that matter for trustworthy numbers: SLA gating (exemplar), publishable-only goodput,
hash determinism/sensitivity, and that the committed manifests lint. Keeps the tooling maintainable —
a change that breaks one of these fails here instead of silently corrupting a result.
"""

import atexit
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
fails = []

# The 256k max-concurrency-sla cell family was pruned in the strict consolidation (only the 2 north-star
# recipes survive: KVBM/GLM-5 — all goal=pareto or agentic-behavior). The exemplar_check / publish /
# export_record tests below assert SLA-goal semantics (metric max_concurrency_at_sla), so we synthesize a
# minimal, schema-complete goal=max-concurrency-sla cell in a tempdir instead of pointing at a real cell.
# NB: created UNDER the repo root (not /tmp) so export_record's `cell.relative_to(ROOT)` resolves; the dotdir
# is outside recipes/ so validate/contract/matrix never scan it, and atexit removes it at process exit.
_SLA_TMP = tempfile.mkdtemp(dir=ROOT, prefix=".selftest-sla-")
atexit.register(lambda: shutil.rmtree(_SLA_TMP, ignore_errors=True))
SLA_CELL = Path(_SLA_TMP) / "synth-256k-sla"
SLA_CELL.mkdir()
(SLA_CELL / "recipe.yaml").write_text(
    "envelope:\n"
    "  name: synth-256k-sla\n"
    "  model: nemotron-ultra-3\n"
    "  gpu_type: B200\n"
    "  arch: amd64\n"
    "  engine: vllm\n"
    "  serving_mode: aggregated\n"
    "  framework: none\n"
    "  scenario: llm-perf\n"
    "  distribution: d-256k\n"
    "  mode: mooncake-trace\n"
    "  launcher: aiperf\n"
    "  goal: max-concurrency-sla\n"
    "  status: runs\n"
    "  results_link: ./RESULTS.md\n"
    "  requires:\n"
    "    gpu: { count: 8 }\n"
    "  provenance:\n"
    '    image_digest: "sha256:' + "a" * 64 + '"\n'
    "  exemplar:\n"
    "    metric: max_concurrency_at_sla\n"
    "    unit: concurrency\n"
    "    reference: 256\n"
    "    tolerance_pct: 5\n"
    # 'highest'+'rung' (and no 'interpol') → measured_rung_policy True: publish the highest measured
    # SLA-passing rung, keeping the crossing as advisory metadata (mirrors the pruned 256k SLA cells).
    "    comparison: published value = the highest measured SLA-passing rung; crossing is advisory only\n"
    "serving:\n"
    "  tp: 8\n"
    "  max_model_len: 262144\n"
    "  gpu_mem_util: 0.85\n"
    "bench:\n"
    "  dataset: { sha256: ds123 }\n"
    "  sweep_concurrency: [128, 256]\n"
    "  sla: { ttft_ms: 10000, tpot_ms: 100, stop_stat: p50 }\n"
)


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def run(*args):
    return subprocess.run([sys.executable, *[str(a) for a in args]], capture_output=True, text=True)


# Migrated off the pruned 256k SLA family to the surviving Qwen3-KVBM gb300 cell (status: runs; carries an
# sla block ttft_ms=10000/tpot_ms=100 so the exemplar_check SLA gate below still resolves 128 pass / 256 fail).
B200 = ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"
# A currently-PUBLISHED (status: runs) cell, for the checks that need a real published record
# (provenance impact MATCH/DRIFT, observe.status_rows). Kept SEPARATE from B200 because both KVBM cells
# went `runs → wip` under the whole-node policy (their shared-node numbers were invalidated), so B200 is
# no longer published. These checks are scenario-agnostic, so any published cell works.
PUBLISHED = ROOT / "scripts/fixtures/sample_cells/nemotron-ultra-3-gb300-vllm-agg-pareto-c16"

# --- exemplar_check: the SLA gate must EXCLUDE failing rungs ---
with tempfile.TemporaryDirectory() as td:
    csv = Path(td) / "metrics_summary.csv"
    csv.write_text(
        "# schema_version=6\nconcurrency,ttft_p50_ms,itl_p50_ms,error_rate_pct,throughput_per_gpu_tok_per_s\n"
        "128,8000,95,1,1100\n256,12000,120,2,1300\n"
    )  # 128 passes; 256 fails (ttft>10s, itl>100ms)
    r = run(ROOT / "analysis/llm-perf/exemplar_check.py", SLA_CELL, csv, "--json")
    out = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    # 128 passes (itl 95); 256 fails (itl 120 > 100). The recipe comparison contract says "highest rung",
    # so the published value is the measured rung; the interpolated crossing remains advisory metadata.
    check(
        "exemplar_check: max_concurrency_at_sla uses the highest measured SLA-passing rung",
        out.get("value") == 128 and out.get("value_source") == "highest_sla_passing_rung",
        r.stdout[:150],
    )
    check(
        "exemplar_check: advisory crossing keeps binding=TPOT + wide-bracket (2.0×) guardrail",
        (out.get("crossing") or {}).get("binding") == "TPOT" and (out.get("crossing") or {}).get("ratio") == 2.0,
        str(out.get("crossing")),
    )

# --- recipe_hash: deterministic ---
h1 = run(ROOT / "scripts/recipe_hash.py", B200).stdout
h2 = run(ROOT / "scripts/recipe_hash.py", B200).stdout
check("recipe_hash: deterministic + non-empty", h1 == h2 and "recipe_hash:" in h1)

# --- benchmark_id: the STABLE benchmark identity — invariant across image rolls + extra_args toggles, but moves
#     when WHAT is measured changes. recipe_hash (the full fingerprint) moves on the same tuning changes. ---
sys.path.insert(0, str(ROOT / "scripts"))
import benchmark_id as _bid  # noqa: E402
import recipe_hash as _rh2  # noqa: E402


def _dc(r):
    return json.loads(json.dumps(r))  # deep copy via round-trip


def _wl(r):
    return json.dumps(_bid.benchmark_identity(r), sort_keys=True)  # canonical benchmark identity


def _fp(r):  # recipe_hash's fingerprint input (no rendered/)
    with tempfile.TemporaryDirectory() as _td:
        return json.dumps(_rh2.fingerprint_input(r, Path(_td)), sort_keys=True)


_base = {
    "envelope": {
        "model": "m",
        "gpu_type": "B200",
        "arch": "amd64",
        "engine": "vllm",
        "serving_mode": "aggregated",
        "framework": "none",
        "scenario": "llm-perf",
        "distribution": "d-256k",
        "mode": "mooncake-trace",
        "launcher": "aiperf",
        "goal": "pareto",
        "requires": {"gpu": {"count": 8}},
        "provenance": {"image_digest": "sha256:AAAA"},
    },
    "serving": {
        "tp": 8,
        "max_model_len": 262144,
        "gpu_mem_util": 0.85,
        "extra_args": ["--enable-prefix-caching", "--max-num-seqs 1024"],
    },
    "bench": {
        "dataset": {"sha256": "ds123"},
        "sweep_concurrency": [32, 64],
        "sla": {"tpot_ms": 100},
    },
}
_w0 = _wl(_base)
# image roll / extra_args toggle / gpu_mem_util tuning → benchmark_id STABLE, recipe_hash MOVES
_r_img = _dc(_base)
_r_img["envelope"]["provenance"]["image_digest"] = "sha256:BBBB"
check("benchmark_id: STABLE across image-digest change", _wl(_r_img) == _w0)
check(
    "recipe_hash: MOVES on image-digest change (reproducibility guard intact)",
    _fp(_r_img) != _fp(_base),
)
_r_flag = _dc(_base)
_r_flag["serving"]["extra_args"].append("--enable-expert-parallel")
check("benchmark_id: STABLE across extra_args toggle", _wl(_r_flag) == _w0)
check("recipe_hash: MOVES on extra_args toggle", _fp(_r_flag) != _fp(_base))
_r_mem = _dc(_base)
_r_mem["serving"]["gpu_mem_util"] = 0.9
check("benchmark_id: STABLE across gpu_mem_util tuning", _wl(_r_mem) == _w0)
# WHAT is measured changes → benchmark_id MOVES
_r_ds = _dc(_base)
_r_ds["bench"]["dataset"]["sha256"] = "ds999"
check("benchmark_id: MOVES when dataset sha256 changes", _wl(_r_ds) != _w0)
_r_sweep = _dc(_base)
_r_sweep["bench"]["sweep_concurrency"] = [32, 64, 128]
check("benchmark_id: MOVES when sweep changes", _wl(_r_sweep) != _w0)
_r_sla = _dc(_base)
_r_sla["bench"]["sla"] = {"tpot_ms": 50}
check("benchmark_id: MOVES when SLA changes", _wl(_r_sla) != _w0)
_r_tp = _dc(_base)
_r_tp["serving"]["tp"] = 4
check("benchmark_id: MOVES when tp (deployment shape) changes", _wl(_r_tp) != _w0)
_r_model = _dc(_base)
_r_model["envelope"]["model"] = "m2"
check("benchmark_id: MOVES when model changes", _wl(_r_model) != _w0)
# on a real cell the two identities are DIFFERENT values (distinct concepts, not an alias)
_bid_real = run(ROOT / "scripts/benchmark_id.py", B200).stdout.split("benchmark_id:")[-1].strip()[:64]
_rh_real = run(ROOT / "scripts/recipe_hash.py", B200).stdout.split("recipe_hash:")[-1].strip()[:64]
check(
    "benchmark_id != recipe_hash on a real cell (distinct, not aliased)",
    len(_bid_real) == 64 and len(_rh_real) == 64 and _bid_real != _rh_real,
)

# --- disagg_role.replicas: per-role worker-count support (1P8D … 2P1D) — additive + hash-safe ---
import check_invariants as _ci  # noqa: E402

# Based on whichever GLM-5 1P1D disagg cell the branch carries. gpu.count is DERIVED from that cell's own
# tp below, never hardcoded: a literal (it was 36, for a tp=4 cell) silently encodes one cell's topology and
# breaks the moment the base cell changes — which is exactly what happened when the tp=4 GB300 cell moved off
# this branch and a tp=8 B200 cell took its place.
_P1D1 = ROOT / (
    "recipes/llm-perf/Glm5/B200_k8s/Disagg/sglang_dynamo/1k_1k/glm5-fp8-b200-sglang-dynamo14-1k1k-hightpt-c2576-1p1d"
)
with tempfile.TemporaryDirectory() as td:
    cell = Path(td) / "glm5-fp8-gb300-sglang-1p8d-pareto"
    shutil.copytree(_P1D1, cell)
    import yaml as _yaml  # noqa: E402

    rec = _yaml.safe_load((cell / "recipe.yaml").read_text())
    # 1P8D: 1 prefill worker, 8 decode workers. gpu.count = (prefill.replicas + 8) × tp, from THIS recipe.
    _dis = rec["serving"]["disagg"]
    _tp = _dis["prefill"].get("tp") or rec["serving"]["tp"]
    rec["envelope"]["name"] = "glm5-fp8-gb300-sglang-1p8d-pareto"
    _dis["decode"]["replicas"] = 8
    rec["envelope"]["requires"]["gpu"]["count"] = (_dis["prefill"].get("replicas", 1) * _tp) + (
        8 * (_dis["decode"].get("tp") or _tp)
    )
    (cell / "recipe.yaml").write_text(_yaml.safe_dump(rec, sort_keys=False))
    shutil.rmtree(cell / "rendered", ignore_errors=True)
    rr = subprocess.run([str(ROOT / "scripts/render.sh"), str(cell)], capture_output=True, text=True)
    import re as _re2  # noqa: E402

    import yaml as _yaml2

    wtext = (cell / "rendered" / "workers.yaml").read_text() if (cell / "rendered" / "workers.yaml").exists() else ""
    wtext = _re2.sub(r"\$\{[^}]+\}", "ENVSUBST", wtext)  # neutralize ${...} launch-time vars for yaml parse
    deploys = [d for d in _yaml2.safe_load_all(wtext) if isinstance(d, dict) and d.get("kind") == "Deployment"]
    by_role = {d["metadata"]["labels"]["llmb.nvidia.com/role"]: d["spec"].get("replicas") for d in deploys}
    check(
        "disagg replicas: 1P8D renders exactly 2 role Deployments (1 prefill + 1 decode)",
        len(deploys) == 2 and set(by_role) == {"prefill", "decode"},
        f"rc={rr.returncode} roles={list(by_role)} {rr.stderr[:120]}",
    )
    check(
        "disagg replicas: prefill Deployment replicas=1 (default, unset), decode replicas=8",
        by_role.get("prefill") == 1 and by_role.get("decode") == 8,
        str(by_role),
    )
    # gpu.count derivation: (prefill.replicas + decode.replicas) × tp must validate; a wrong count FAILS.
    _prob, _ = _ci.check_cell(cell)
    check(
        "disagg replicas: gpu.count=36 validates for 1P8D (no invariant problem)",
        not any("gpu.count" in p for p in _prob),
        str(_prob)[:160],
    )
    rec["envelope"]["requires"]["gpu"]["count"] = 8  # the old 1P1D count is now WRONG for 1P8D
    (cell / "recipe.yaml").write_text(_yaml.safe_dump(rec, sort_keys=False))
    _prob2, _ = _ci.check_cell(cell)
    check(
        "disagg replicas: a stale gpu.count=16 is CAUGHT for 1P8D",
        any("gpu.count" in p for p in _prob2),
        str(_prob2)[:160],
    )

# hash sensitivity/safety on the identity subsets (pure): replicas>1 MOVES both fingerprints (real topology
# change); an explicit replicas==1 is stripped so it fingerprints identically to omitting it (no churn).
_disagg_base = {
    "envelope": {
        "model": "glm5-fp8",
        "gpu_type": "B200",
        "arch": "amd64",
        "engine": "sglang",
        "serving_mode": "disaggregated",
        "framework": "dynamo",
        "scenario": "llm-perf",
        "distribution": "d",
        "mode": "mooncake-trace",
        "launcher": "aiperf",
        "goal": "pareto",
        "requires": {"gpu": {"count": 16}},
        "provenance": {"image_digest": "sha256:AAAA"},
    },
    "serving": {
        "tp": 8,
        "stack": "sglang-disagg",
        "disagg": {
            "transfer_backend": "nixl",
            "prefill": {"tp": 8, "dp": 8, "ep": 1},
            "decode": {"tp": 8, "dp": 8, "ep": 1},
        },
    },
    "bench": {
        "dataset": {"sha256": "ds"},
        "sweep_concurrency": [8],
        "sla": {"tpot_ms": 100},
    },
}
_bid_1p1d = _wl(_disagg_base)
_fp_1p1d = _fp(_disagg_base)
_r_1p1e = _dc(_disagg_base)
_r_1p1e["serving"]["disagg"]["decode"]["replicas"] = 1  # explicit default
check(
    "disagg replicas: explicit replicas==1 keeps benchmark_id byte-identical (conditional hashing)",
    _wl(_r_1p1e) == _bid_1p1d,
)
check(
    "disagg replicas: explicit replicas==1 keeps recipe_hash byte-identical (default stripped)",
    _fp(_r_1p1e) == _fp_1p1d,
)
_r_1p8d = _dc(_disagg_base)
_r_1p8d["serving"]["disagg"]["decode"]["replicas"] = 8
_r_1p8d["envelope"]["requires"]["gpu"]["count"] = 72
check(
    "disagg replicas: replicas=8 (1P8D) MOVES benchmark_id (different deployment under test)",
    _wl(_r_1p8d) != _bid_1p1d,
)
check("disagg replicas: replicas=8 (1P8D) MOVES recipe_hash", _fp(_r_1p8d) != _fp_1p1d)

# --- pareto_geomean (goal=pareto): per-rung g=√(TPS/GPU·TPS/user), geomean over rungs ---
sys.path.insert(0, str(ROOT / "analysis/llm-perf"))
import exemplar_check as _ec  # noqa: E402

_pts = [
    {
        "concurrency": 256,
        "tokens_per_s_per_user_from_itl": 100,
        "throughput_per_gpu_tok_per_s": 10,
    },
    {
        "concurrency": 384,
        "tokens_per_s_per_user_from_itl": 50,
        "throughput_per_gpu_tok_per_s": 30,
    },
    {
        "concurrency": 512,
        "tokens_per_s_per_user_from_itl": 20,
        "throughput_per_gpu_tok_per_s": 50,
    },
]
# per-rung g: √(10·100)=31.6228 · √(30·50)=38.7298 · √(50·20)=31.6228 → geomean 33.8; per_point sorted by concurrency
_v_pg, _pp_pg = _ec.pareto_geomean(_pts)
check(
    "pareto_geomean: geomean over rungs of the per-rung √(TPS/GPU·TPS/user)",
    _v_pg == 33.8,
    str(_v_pg),
)
check(
    "pareto_geomean: per_point carries a g per rung, sorted by concurrency",
    [p["concurrency"] for p in _pp_pg] == [256, 384, 512] and _pp_pg[0]["g"] == 31.6228,
    str(_pp_pg),
)
# a SINGLE point is valid (geomean of one value); NO usable point → None
check(
    "pareto_geomean: single point is valid (geomean of one rung)",
    _ec.pareto_geomean([_pts[0]])[0] == 31.6 and len(_ec.pareto_geomean([_pts[0]])[1]) == 1,
)
check(
    "pareto_geomean: no usable frontier point → (None, [])",
    _ec.pareto_geomean([]) == (None, []),
)
# pareto_point_compare: the WORKED example — −15% at 2 of 5 rungs → worst 15% → OUTSIDE (rung sets match)
_ref5 = [{"concurrency": c, "g": g} for c, g in [(32, 81.8), (64, 71.0), (128, 60.0), (256, 45.6), (512, 33.5)]]
_reg5 = [
    {"concurrency": c, "g": (g * 0.85 if c in (256, 512) else g)}
    for c, g in [(32, 81.8), (64, 71.0), (128, 60.0), (256, 45.6), (512, 33.5)]
]
_cmp5 = _ec.pareto_point_compare(_reg5, _ref5, 5.0)
check(
    "pareto_point_compare: −15% at 2 of 5 rungs → worst 15% → verdict 'outside' (matched rung sets)",
    _cmp5["verdict"] == "outside" and _cmp5["worst_pct"] == 15.0 and _cmp5["matched"] is True,
    str(_cmp5),
)
check(
    "pareto_point_compare: identical curves → 'exemplar' (worst 0%)",
    _ec.pareto_point_compare(_ref5, _ref5, 5.0)["verdict"] == "exemplar",
)
check(
    "pareto_point_compare: disjoint rung sets → 'not-comparable' (never compared on a subset)",
    _ec.pareto_point_compare([{"concurrency": 1, "g": 5.0}], [{"concurrency": 2, "g": 5.0}])["verdict"]
    == "not-comparable",
)

# --- 2nd code-review batch: NaN handling, planner edge states, sh() rc propagation, all-non-publishable ---
check(
    "exemplar_check.as_float: NaN → None (not silently counted as a failing rung)",
    _ec.as_float(float("nan")) is None and _ec.as_float("nan") is None and _ec.as_float("12.5") == 12.5,
)
sys.path.insert(0, str(ROOT / "scripts"))
import sweep_planner as _planner  # noqa: E402

_pcfg = {
    "start": 128,
    "min": 16,
    "max": 512,
    "ratio": 1.3,
    "bracket_tolerance": 1.3,
    "max_runs": 6,
    "ttft_limit": 10000,
    "tpot_limit": 100,
}
check(
    "sweep_planner: all-fail-at-min → below_range",
    _planner.plan(_pcfg, [{"concurrency": 16, "ttft": 1000, "tpot": 200}]).get("reason") == "below_range",
)
check(
    "sweep_planner: a pass ABOVE a fail → non_monotonic",
    _planner.plan(
        _pcfg,
        [
            {"concurrency": 64, "ttft": 1000, "tpot": 50},
            {"concurrency": 32, "ttft": 1000, "tpot": 200},
        ],
    ).get("reason")
    == "non_monotonic",
)
import adaptive_sweep as _asw  # noqa: E402

_sh_raised = False
try:
    _asw.sh(["false"])  # exits 1 — must abort, not be ignored
except SystemExit:
    _sh_raised = True
check("adaptive_sweep.sh: aborts (SystemExit) when a step exits non-zero", _sh_raised)


# --- sweep_planner: deterministic geometric-grid search converges to a tight bracket in few runs ---
def _drive_planner(true_cross):
    cfg = [
        "--start",
        "128",
        "--min",
        "16",
        "--max",
        "512",
        "--ratio",
        "1.3",
        "--ttft-limit",
        "10000",
        "--tpot-limit",
        "100",
        "--json",
    ]
    meas, out = [], {}
    for _ in range(8):
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/sweep_planner.py"),
                *cfg,
                "--measurements",
                json.dumps(meas),
            ],
            capture_output=True,
            text=True,
        )
        out = json.loads(p.stdout)
        if out.get("action") != "run":
            break
        c = out["concurrency"]
        meas.append(
            {
                "concurrency": c,
                "ttft": 1200,
                "tpot": round(100 * (c / true_cross) ** 1.5, 1),
            }
        )
    return out, len(meas)


_o, _n = _drive_planner(145.0)
_lo, _hi = _o.get("bracket") or [0, 0]
check(
    "sweep_planner: converges to a tight bracket around the true crossing in ≤6 runs (deterministic)",
    _o.get("action") == "done" and _lo <= 145 <= _hi and _hi / max(_lo, 1) <= 1.35 and _n <= 6,
    f"{_o} runs={_n}",
)


# --- sweep_planner: a ceiling right at `max` must be BRACKETED, not reported above_range (B200 GLM finding) ---
def _drive_at_max(true_cross, start, mx):
    cfg = [
        "--start",
        str(start),
        "--min",
        "16",
        "--max",
        str(mx),
        "--ratio",
        "1.3",
        "--ttft-limit",
        "10000",
        "--tpot-limit",
        "100",
        "--json",
    ]
    meas, out = [], {}
    for _ in range(8):
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/sweep_planner.py"),
                *cfg,
                "--measurements",
                json.dumps(meas),
            ],
            capture_output=True,
            text=True,
        )
        out = json.loads(p.stdout)
        if out.get("action") != "run":
            break
        c = out["concurrency"]
        meas.append(
            {
                "concurrency": c,
                "ttft": 1200,
                "tpot": round(100 * (c / true_cross) ** 1.5, 1),
            }
        )
    return out, meas


# GLM 1P1D: crossing ~126.8, start 96, max 128 — old planner stopped `above_range` at c125; now it probes 128.
_om, _mm = _drive_at_max(126.8, 96, 128)
check(
    "sweep_planner: tests the max boundary (128) and brackets the ceiling instead of false above_range",
    _om.get("action") == "done" and 128 in {m["concurrency"] for m in _mm} and (_om.get("bracket") or [0, 0])[1] <= 128,
    f"{_om} rungs={sorted(m['concurrency'] for m in _mm)}",
)

# --- adaptive_sweep orchestrator: dry-run drives planner→crossing in few runs (also guards build_grid termination) ---
r = subprocess.run(
    [
        sys.executable,
        str(ROOT / "scripts/adaptive_sweep.py"),
        str(ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"),
        "dummy",
        "--dry-run",
        "--curve-crossing",
        "145",
        "--start",
        "128",
        "--grid-min",
        "32",
    ],
    capture_output=True,
    text=True,
    timeout=30,
)
check(
    "adaptive_sweep: dry-run converges to the crossing (~144) in ≤4 rungs (deterministic)",
    "interpolated max_concurrency_at_sla ≈ 144" in r.stdout
    and "bracket_tight" in r.stdout
    and any(f"({n}," in r.stdout for n in (3, 4)),
    (r.stdout or r.stderr)[-160:],
)

# --- kv_budget: parses vLLM's KV-cache lines → reports the full-context floor ---
r = subprocess.run(
    [sys.executable, str(ROOT / "scripts/kv_budget.py"), str(B200), "-"],
    input="GPU KV cache size: 7,077,454 tokens\nMaximum concurrency for 262144 tokens per request: 27.00x\n",
    capture_output=True,
    text=True,
)
check(
    "kv_budget: parses vLLM KV lines + reports the floor",
    "27" in r.stdout and "FULL-CONTEXT FLOOR" in r.stdout,
    r.stdout[:120],
)

# --- lint_manifests: the committed rendered manifests are valid k8s ---
r = run(ROOT / "scripts/lint_manifests.py")
check(
    "lint_manifests: committed rendered manifests valid",
    r.returncode == 0,
    (r.stdout or "").splitlines()[-1:] and r.stdout.splitlines()[-1][:120],
)

# --- publish: --dry-run chains aggregate→exemplar_check and auto-bumps wip→runs (no writes) ---
with tempfile.TemporaryDirectory() as td:
    rd = Path(td)
    cell = rd / "cell"
    shutil.copytree(SLA_CELL, cell)
    recipe = cell / "recipe.yaml"
    txt = recipe.read_text()
    txt = txt.replace("  status: runs", "  status: wip", 1)
    txt = txt.replace("    reference: 256", "    reference: null", 1)
    recipe.write_text(txt)
    (rd / "metrics_summary.csv").write_text(
        "# schema_version=6\nconcurrency,ttft_p50_ms,itl_p50_ms,error_rate_pct,throughput_per_gpu_tok_per_s\n"
        "128,8000,95,1,1100\n256,9000,130,1,1250\n"
    )  # 256 fails on TPOT → a real interpolated crossing exists
    (rd / "run_meta.json").write_text(json.dumps({"run_id": "selftest", "gpu_count": 8}))
    r = run(ROOT / "scripts/publish.py", cell, rd, "--dry-run")
    out = r.stdout
    check(
        "publish: --dry-run computes the metric + would bump wip→runs",
        "max_concurrency_at_sla" in out and "wip → runs" in out,
        out.strip()[:140],
    )

# --- publish: identity_mismatch, fetch_incomplete, refresh_narrative ---
import publish as _pub  # noqa: E402

check(
    "publish.identity_mismatch: matching model+gpu → belongs here (empty)",
    _pub.identity_mismatch(
        {"envelope": {"model": "nemotron-ultra-3"}, "serving": {"tp": 8}},
        {"model_name": "nemotron-ultra-3", "gpu_count": 8},
    )
    == [],
)
check(
    "publish.identity_mismatch: foreign result (glm5/16gpu into a nemotron/8 cell) → flagged",
    len(
        _pub.identity_mismatch(
            {"envelope": {"model": "nemotron-ultra-3"}, "serving": {"tp": 8}},
            {"model_name": "glm5-fp8", "gpu_count": 16},
        )
    )
    == 2,
)
# fetch_incomplete (R2-4): a dirty-fetch receipt makes publish refuse; absent receipt → allow.
# A complete receipt must include evidence that the expected files and bytes landed.
# Version 1 receipts lack that evidence and remain unverified. Full contract:
# scripts/fetch_receipt.py + scripts/selftest_fetch_receipt.py.
_V2_OK = {
    "receipt_version": 2,
    "complete": True,
    "failed": [],
    "files_written": 120,
    "bytes_written": 9_000_000,
    "remote_files": 120,
    "remote_bytes": 9_000_000,
    "reconciled": True,
}
check(
    "publish.fetch_incomplete: absent receipt → allow (old runs pre-date receipts)",
    _pub.fetch_incomplete(None) is None,
)
check(
    "publish.fetch_incomplete: verified v2 receipt → allow",
    _pub.fetch_incomplete(_V2_OK) is None,
)
check(
    "publish.fetch_incomplete: v1 'complete=true' receipt → REFUSE (no evidence, not grandfathered)",
    (lambda w: w is not None and "evidence" in w.lower())(_pub.fetch_incomplete({"complete": True, "failed": []})),
)
check(
    "publish.fetch_incomplete: zero files landed → REFUSE even with complete=true",
    _pub.fetch_incomplete({**_V2_OK, "files_written": 0, "bytes_written": 0}) is not None,
)
check(
    "publish.fetch_incomplete: incomplete → reason names the missing entries",
    (lambda w: w is not None and "concurrency_256" in w and "resume" in w.lower())(
        _pub.fetch_incomplete({**_V2_OK, "complete": False, "failed": ["concurrency_256"]})
    ),
)
# refresh_narrative (B2-5): strip ONLY the new-cell placeholder; warn (never rewrite) on other stale prose
_clean, _stale = _pub.refresh_narrative("# Results\n\n_TODO: fill in once the run completes._\n")
check(
    "publish.refresh_narrative: strips the exact new-cell placeholder line",
    "_TODO: fill in once the run completes._" not in _clean and "# Results" in _clean and _stale == [],
)
_clean2, _stale2 = _pub.refresh_narrative(
    "# Results\n\nThe 768/1024 top-out sweep in progress; numbers land when it completes.\n"
)
check(
    "publish.refresh_narrative: keeps human prose but flags stale phrases",
    "top-out" in _clean2 and set(_stale2) >= {"top-out", "sweep in progress", "numbers land when"},
)
_clean3, _stale3 = _pub.refresh_narrative("# Results\n\nFinal: 225 TPS/GPU plateau at 1M context.\n")
check(
    "publish.refresh_narrative: clean human narrative → unchanged, no warnings",
    "225 TPS/GPU" in _clean3 and _stale3 == [],
)

# --- port_recipe (R2-5): pure hardware-retarget helpers for the /port-recipe CLI surface ---
import port_recipe as _port  # noqa: E402

check(
    "port_recipe.arch_for: B200→amd64, GB200/GB300 (Grace)→arm64",
    _port.arch_for("B200") == "amd64" and _port.arch_for("GB200") == "arm64" and _port.arch_for("GB300") == "arm64",
)
check(
    "port_recipe.retarget_leaf: swaps ONLY the hardware token (case-insensitive)",
    _port.retarget_leaf("nemotron-ultra-3-gb200-vllm-agg", "GB200", "GB300") == "nemotron-ultra-3-gb300-vllm-agg",
)
check(
    "port_recipe.retarget_leaf: b200→gb300 keeps the rest of the slug",
    _port.retarget_leaf("nemotron-ultra-3-b200-vllm-agg-pareto", "B200", "GB300")
    == "nemotron-ultra-3-gb300-vllm-agg-pareto",
)
check(
    "port_recipe.gb_flag_hint: Grace target advises --disable-custom-all-reduce; B200 warns against it",
    "disable-custom-all-reduce" in _port.gb_flag_hint("GB300") and "do NOT" in _port.gb_flag_hint("B200"),
)
# --- export_record: the canonical DB record carries identity+fingerprint+result+per-rung detail ---
with tempfile.TemporaryDirectory() as td:
    rd = Path(td)
    (rd / "metrics_summary.csv").write_text(
        "# schema_version=6\nconcurrency,ttft_p50_ms,itl_p50_ms,error_rate_pct,throughput_per_gpu_tok_per_s,tokens_per_s_per_user_from_itl\n"
        "128,8000,95,1,1100,60\n256,9000,130,1,1250,40\n"
    )  # 256 fails TPOT → real interpolated crossing (~146)
    (rd / "run_meta.json").write_text(json.dumps({"run_id": "rectest", "gpu_count": 8, "wall_seconds_total": 3600}))
    r = run(ROOT / "scripts/export_record.py", SLA_CELL, rd)
    rec = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    check(
        "export_record: one record with identity(goal)+64-hex fingerprint+metric value+per-rung detail",
        rec.get("identity", {}).get("goal") == "max-concurrency-sla"
        and len(rec.get("fingerprint", {}).get("recipe_hash", "")) == 64
        and rec.get("result", {}).get("value") is not None
        and len(rec.get("detail", {}).get("rungs", [])) == 2,
        (r.stdout or r.stderr)[:160],
    )
    check(
        "export_record: a real run_meta.run_id still wins over the path (no existing record re-stamped)",
        rec.get("provenance", {}).get("run_id") == "rectest",
    )

# --- export_record: run identity is never null when the canonical path knows it. Some lanes emit
#     NO run_meta.json at all, so a run_meta-only read left provenance.run_id null —
#     a knowable id rendering as "this run has no name". ---
sys.path.insert(0, str(ROOT / "scripts"))
import export_record as _er  # noqa: E402

check(
    "export_record._run_id_from_path: runs/<run_id>/curated → <run_id>",
    _er._run_id_from_path(Path("/x/recipes/cell/runs/ttj2o4g/curated")) == "ttj2o4g",
)
check(
    "export_record._run_id_from_path: runs/<run_id> (uncurated) → <run_id>",
    _er._run_id_from_path(Path("/x/recipes/cell/runs/ttj2o4g")) == "ttj2o4g",
)
check(
    "export_record._run_id_from_path: a scratch dir with no runs/ ancestor stays None (never a guess)",
    _er._run_id_from_path(Path("/tmp/scratch-dir")) is None,
)
with tempfile.TemporaryDirectory() as td:
    cur = Path(td) / "recipes" / "c" / "runs" / "ttj2o4g" / "curated"  # lane shape with no run_meta
    cur.mkdir(parents=True)
    (cur / "metrics_summary.csv").write_text(
        "# schema_version=6\nconcurrency,ttft_p50_ms,itl_p50_ms,error_rate_pct,"
        "throughput_per_gpu_tok_per_s,tokens_per_s_per_user_from_itl\n128,8000,95,1,1100,60\n"
    )
    r = run(ROOT / "scripts/export_record.py", SLA_CELL, cur)
    rec = json.loads(r.stdout) if r.stdout.strip().startswith("{") else {}
    check(
        "export_record: run_id falls back to the canonical path when the lane wrote no run_meta",
        rec.get("provenance", {}).get("run_id") == "ttj2o4g",
        (r.stdout or r.stderr)[:160],
    )

# --- run_id: label-safe instance id; --fit keeps the whole Job name <cell>-<kind>-<id> ≤63 ---
sys.path.insert(0, str(ROOT / "scripts"))
import run_id as _rid  # noqa: E402

_H = "c0b1b4f1" + "0" * 56
_r = _rid.mint("nemotron-ultra-3-b200-vllm-agg-256k", _H, at="20260716t013045")
check(
    "run_id: plain id is compact-UTC + salt (YYMMDDtHHMM-xxxx), sortable, DNS-1123-safe, ≤63",
    _r.startswith("260716t0130-") and len(_r) == 16 and bool(_rid.LABEL_RE.match(_r)) and len(_r) <= 63,
    _r,
)
check(
    "run_id: time-sortable — a later stamp sorts lexically after an earlier one",
    _rid.mint("c", _H, at="20260716t013045") < _rid.mint("c", _H, at="20260716t014500"),
)
check(
    "run_id: deterministic given a fixed stamp",
    _rid.mint("x-cell", _H, at="20260716t013045") == _rid.mint("x-cell", _H, at="20260716t013045"),
)
_long = "nemotron-ultra-3-gb300-agentic-stable16-eval-opt"  # 48 chars — the longest real cell name
_fr = _rid.mint(_long, _H, at="20260716t013045", fit_kind="bench")
_jn = _rid.job_name(_long, "bench", _fr)
check(
    "run_id: --fit shrinks the id so <cell>-bench-<id> fits ≤63 (the >63-char-label fix)",
    bool(_rid.LABEL_RE.match(_fr)) and len(_jn) <= 63 and bool(_rid.LABEL_RE.match(_jn)),
    f"id={_fr} job={_jn}({len(_jn)})",
)
_jn2 = _rid.job_name("a" * 80, "bench", "c0b1b4f1-20260716t013045")
check(
    "run_id: job_name is DNS-1123 ≤63 for ANY cell-name length",
    len(_jn2) <= 63 and bool(_rid.LABEL_RE.match(_jn2)),
    f"{_jn2}({len(_jn2)})",
)

# --- compare: cross-cluster report (hardware ranking + same-GPU reproducibility verdict) ---
with tempfile.TemporaryDirectory() as td:

    def _cell(name, gpu, ref, metric="max_concurrency_at_sla", tol=5):
        d = Path(td) / name
        d.mkdir()
        (d / "recipe.yaml").write_text(
            f"envelope:\n  name: {name}\n  gpu_type: {gpu}\n  scenario: llm-perf\n  status: runs\n"
            f"  exemplar: {{metric: {metric}, unit: concurrency, reference: {ref}, tolerance_pct: {tol}}}\n"
        )
        return str(d)

    r = run(
        ROOT / "analysis/compare.py",
        _cell("a", "B200", 294.7),
        _cell("b", "GB300", 240.7),
        _cell("c", "GB200", 141.9),
    )
    check(
        "compare: mixed-GPU → HARDWARE ranking with % of leader",
        "HARDWARE" in r.stdout and "100.0%" in r.stdout and "81.7%" in r.stdout,
        r.stdout[:120],
    )
    r2 = run(
        ROOT / "analysis/compare.py",
        _cell("d", "B200", 100.0),
        _cell("e", "B200", 97.0),
        _cell("f", "B200", 80.0),
        "--baseline",
        "d",
    )
    check(
        "compare: same-GPU → REPRODUCIBILITY verdict (97 within 5%, 80 outside)",
        "REPRODUCIBILITY" in r2.stdout and "✅ within" in r2.stdout and "❌ outside" in r2.stdout,
        r2.stdout[:160],
    )
# --all digest over the real repo.
# (The old 256k SLA HARDWARE facet was pruned with the 256k family.)
r3 = run(ROOT / "analysis/compare.py", "--all")
# A digest needs >=2 cells carrying VALUES. Which facets exist depends on which cells have published
# numbers TODAY, so asserting on one named facet pins the repo's DATA rather than compare's BEHAVIOUR —
# and then a perfectly good publish turns CI red. (It did: publishing the KVBM cell produced a
# `pareto_geomean` facet, which is neither "the expected facet" nor "nothing to compare", so both
# branches missed and the check failed on a correct result.)
# The invariant that actually matters, and holds at every data state: compare --all EXITS CLEANLY and
# either emits at least one well-formed facet or says plainly that nothing is comparable. It must never
# crash and never emit a facet with no comparison in it.
_have_facet = "facet(s) with comparable results" in r3.stdout or (
    "net_behavior_score" in r3.stdout and "REPRODUCIBILITY" in r3.stdout
)
_said_nothing_to_compare = "no facet has" in r3.stdout
check(
    "compare --all: exits cleanly and either emits a facet or says nothing is comparable",
    r3.returncode == 0 and (_have_facet or _said_nothing_to_compare),
    r3.stdout[:200],
)
# A facet, when present, must carry an actual comparison — not an empty header.
check(
    "compare --all: an emitted facet contains a baseline + at least one compared cell",
    (not _have_facet) or ("baseline" in r3.stdout and "metric `" in r3.stdout),
    r3.stdout[:200],
)
# compare.runs_cross_cluster: same cell on ≥2 clusters (from runs.jsonl)
sys.path.insert(0, str(ROOT / "analysis"))
import compare as _cmp  # noqa: E402

with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td)
    _stale = _p / "stale-ledger"
    _stale.mkdir()
    (_stale / "recipe.yaml").write_text(
        "envelope:\n  name: stale-ledger\n  gpu_type: GB300\n  scenario: llm-perf\n  status: runs\n"
        "  exemplar: {metric: pareto_geomean, unit: geomean, reference: null, tolerance_pct: 5}\n"
    )
    (_stale / "record.json").write_text(json.dumps({"result": {"value": 13185.2}}))
    (_stale / "runs.jsonl").write_text('{"run_id":"old","cluster":"","wall_seconds":1}\n')
    check(
        "compare.load_cell: present value-less ledger suppresses stale record fallback",
        _cmp.load_cell(_stale)["value"] is None,
    )
    _valued = _p / "valued-ledger"
    shutil.copytree(_stale, _valued)
    (_valued / "runs.jsonl").write_text('{"run_id":"r1","cluster":"gb300","value":128}\n')
    check(
        "compare.load_cell: latest clustered ledger value wins when exemplar.reference is null",
        _cmp.load_cell(_valued)["value"] == 128,
    )
    _legacy = _p / "legacy-record"
    shutil.copytree(_stale, _legacy)
    (_legacy / "runs.jsonl").unlink()
    check(
        "compare.load_cell: legacy cell with no run ledger can still fall back to record",
        _cmp.load_cell(_legacy)["value"] == 13185.2,
    )
_rl, _ro = _cmp.runs_cross_cluster(
    [
        {
            "cluster": "cluster-a",
            "value": 13185.2,
            "run_id": "rA",
            "recipe_hash": "abc123",
        },
        {
            "cluster": "gb200-b",
            "value": 12980.5,
            "run_id": "rB",
            "recipe_hash": "abc123",
        },
    ],
    "pareto_geomean",
    5.0,
    True,
)
check(
    "compare.runs_cross_cluster: 2 clusters, same setup, within tol → reproducible",
    _ro["clusters"] == 2 and _ro["reproducible"] and _ro["same_setup"] and _ro["spread_pct"] < 2,
)
_, _ro2 = _cmp.runs_cross_cluster(
    [
        {"cluster": "x", "value": 100, "recipe_hash": "a"},
        {"cluster": "y", "value": 130, "recipe_hash": "b"},
    ],
    "m",
    5.0,
    True,
)
check(
    "compare.runs_cross_cluster: mixed recipe_hash + big spread → not reproducible, not same_setup",
    (not _ro2["reproducible"]) and (not _ro2["same_setup"]),
)
check(
    "compare.runs_cross_cluster: <2 clusters with a value → (None, None)",
    _cmp.runs_cross_cluster([{"cluster": "x", "value": 1, "recipe_hash": "a"}], "m", 5.0, True) == (None, None),
)

# compare.runs_within_cluster: same cell run N× on ONE cluster (the fleet's 3×-repeat reproducibility case)
_wl, _wo = _cmp.runs_within_cluster(
    [
        {"cluster": "gb300", "value": 240.0, "run_id": "r1", "recipe_hash": "h"},
        {"cluster": "gb300", "value": 244.0, "run_id": "r2", "recipe_hash": "h"},
        {"cluster": "gb300", "value": 242.0, "run_id": "r3", "recipe_hash": "h"},
    ],
    "max_concurrency_at_sla",
    5.0,
    True,
)
check(
    "compare.runs_within_cluster: 3 same-cluster repeats within tol → 1 group, reproducible, spread<2%",
    len(_wo["clusters"]) == 1
    and _wo["clusters"][0]["n"] == 3
    and _wo["clusters"][0]["reproducible"]
    and _wo["clusters"][0]["spread_pct"] < 2,
)
# an outlier repeat blows the spread past tolerance → NOT reproducible
_, _wo2 = _cmp.runs_within_cluster(
    [
        {"cluster": "gb300", "value": 240.0, "run_id": "r1", "recipe_hash": "h"},
        {"cluster": "gb300", "value": 300.0, "run_id": "r2", "recipe_hash": "h"},
    ],
    "m",
    5.0,
    True,
)
check(
    "compare.runs_within_cluster: outlier repeat → not reproducible",
    not _wo2["clusters"][0]["reproducible"],
)
# mixed recipe_hash across the repeats → flagged not same_setup
_, _wo3 = _cmp.runs_within_cluster(
    [
        {"cluster": "gb300", "value": 240.0, "run_id": "r1", "recipe_hash": "h1"},
        {"cluster": "gb300", "value": 241.0, "run_id": "r2", "recipe_hash": "h2"},
    ],
    "m",
    5.0,
    True,
)
check(
    "compare.runs_within_cluster: mixed recipe_hash within cluster → not same_setup",
    not _wo3["clusters"][0]["same_setup"],
)
# a lone run on a cluster (only 1 valued) → no group → (None, None)
check(
    "compare.runs_within_cluster: <2 runs on any single cluster → (None, None)",
    _cmp.runs_within_cluster(
        [
            {"cluster": "gb300", "value": 1, "recipe_hash": "h"},
            {"cluster": "b200", "value": 2, "recipe_hash": "h"},
        ],
        "m",
        5.0,
        True,
    )
    == (None, None),
)
check(
    "compare.valued_clustered_runs: excludes stubs and blank-cluster values from no-data hints",
    _cmp.valued_clustered_runs(
        [
            {"run_id": "stub", "cluster": "gb300"},
            {"run_id": "blank", "cluster": "", "value": 1},
            {"run_id": "good", "cluster": "gb300", "value": 2},
        ]
    )
    == [{"run_id": "good", "cluster": "gb300", "value": 2}],
)

# --- compare.scope_to_run_ids: band ONLY a sweep's own legs, not the whole same-cluster history (D6) ---
_hist = [
    {"run_id": "old1", "cluster": "gb300", "value": 100.0, "recipe_hash": "h"},
    {
        "run_id": "old2",
        "cluster": "gb300",
        "value": 400.0,
        "recipe_hash": "h",
    },  # unrelated outlier
    {"run_id": "leg1", "cluster": "gb300", "value": 240.0, "recipe_hash": "h"},
    {"run_id": "leg2", "cluster": "gb300", "value": 244.0, "recipe_hash": "h"},
]
check(
    "compare.scope_to_run_ids: keeps only the given run_ids (drops unrelated history)",
    [r["run_id"] for r in _cmp.scope_to_run_ids(_hist, "leg1 leg2")] == ["leg1", "leg2"],
)
check(
    "compare.scope_to_run_ids: comma-separated spec also parses",
    [r["run_id"] for r in _cmp.scope_to_run_ids(_hist, "leg1,leg2")] == ["leg1", "leg2"],
)
check(
    "compare.scope_to_run_ids: empty/None spec → unchanged (legacy band-over-all)",
    _cmp.scope_to_run_ids(_hist, None) == _hist,
)
# The band over ONLY the 2 sweep legs is tight (~1.7% → reproducible); over the whole history it blows up
# (old2=400 → ~122% spread → NOT reproducible). This is the D6 bug: the band must reflect the N repeats only.
_, _scoped = _cmp.runs_within_cluster(_cmp.scope_to_run_ids(_hist, "leg1 leg2"), "max_concurrency_at_sla", 5)
_, _allhist = _cmp.runs_within_cluster(_hist, "max_concurrency_at_sla", 5)
check(
    "compare D6: scoped band = only the 2 legs (reproducible); full history is NOT",
    _scoped["clusters"][0]["reproducible"] is True and _allhist["clusters"][0]["reproducible"] is False,
)

# --- repro_consolidate.consolidate: merge parallel-repro copy values into the original, re-stamped ---
import repro_consolidate as _rc  # noqa: E402

with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td)
    for _nm, _val in [("repro1", 240.0), ("repro2", 244.0)]:
        (_p / _nm).mkdir()
        (_p / _nm / "runs.jsonl").write_text(
            json.dumps(
                {
                    "run_id": _nm,
                    "cluster": "gb300",
                    "value": _val,
                    "metric": "max_concurrency_at_sla",
                    "recipe_hash": "copyhash",
                }
            )
            + "\n"
        )
    (_p / "empty").mkdir()  # a copy that never produced a value → skipped
    _entries = _rc.consolidate([_p / "repro1", _p / "repro2", _p / "empty"], "ORIGHASH")
    check(
        "repro_consolidate.consolidate: 2 valued copies merged, empty skipped",
        [e["value"] for e in _entries] == [240.0, 244.0],
    )
    check(
        "repro_consolidate.consolidate: re-stamped with ORIGINAL recipe_hash (clean same-setup spread)",
        all(e["recipe_hash"] == "ORIGHASH" for e in _entries),
    )
    check(
        "repro_consolidate.consolidate: tags each sample with its source copy",
        [e["repro_source"] for e in _entries] == ["repro1", "repro2"],
    )

# --- repro_consolidate D4 (idempotent re-collect) + D7 (setup-equality gate) ---
with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td)
    # ORIGINAL ledger already holds leg r1 (a prior consolidation).
    (_p / "orig").mkdir()
    (_p / "orig" / "runs.jsonl").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "cluster": "gb300",
                "value": 240.0,
                "metric": "max_concurrency_at_sla",
                "benchmark_id": "BID",
            }
        )
        + "\n"
    )
    for _nm, _rid, _val, _bid in [
        ("leg1", "r1", 240.0, "BID"),  # already in ledger  -> D4 skip
        ("leg2", "r2", 244.0, "BID"),  # NEW same-setup     -> merge
        ("leg3", "r3", 999.0, "OTHER"),
    ]:  # different setup   -> D7 refuse
        (_p / _nm).mkdir()
        (_p / _nm / "runs.jsonl").write_text(
            json.dumps(
                {
                    "run_id": _rid,
                    "cluster": "gb300",
                    "value": _val,
                    "metric": "max_concurrency_at_sla",
                    "benchmark_id": _bid,
                }
            )
            + "\n"
        )
    _legs = [_p / "leg1", _p / "leg2", _p / "leg3"]
    _e = _rc.consolidate(
        _legs,
        "RH",
        orig_benchmark_id="BID",
        existing_ids=_rc.existing_run_ids(_p / "orig"),
    )
    check(
        "repro_consolidate D4: leg already in the ledger (run_id r1) is skipped",
        all(x["run_id"] != "r1" for x in _e),
    )
    check(
        "repro_consolidate D7: mismatched benchmark_id (r3) REFUSED — not laundered as a repeat",
        all(x["run_id"] != "r3" for x in _e),
    )
    check(
        "repro_consolidate: only the NEW same-setup leg r2 merges (re-stamped recipe_hash=RH)",
        [x["run_id"] for x in _e] == ["r2"] and _e[0]["recipe_hash"] == "RH",
    )
    # file-level idempotency: run the real append→dedup cycle TWICE; the ledger is byte-identical after the 2nd.
    _jl = _p / "orig" / "runs.jsonl"

    def _pass():
        with _jl.open("a") as _fh:
            for _x in _rc.consolidate(
                _legs,
                "RH",
                orig_benchmark_id="BID",
                existing_ids=_rc.existing_run_ids(_p / "orig"),
            ):
                _fh.write(json.dumps(_x) + "\n")

    _pass()
    _after_first = _jl.read_text()
    _pass()
    check(
        "repro_consolidate D4: second consolidation pass leaves runs.jsonl BYTE-IDENTICAL (true no-op)",
        _jl.read_text() == _after_first,
        "re-collect double-counted",
    )

# --- fleet_status.roll_up: the coordinator dashboard's roll-up over per-cell rows ---
import fleet_status as _fs  # noqa: E402

_fs_rows = [
    {"n_valued": 3, "reproducible": True},  # published + reproducible
    {"n_valued": 2, "reproducible": False},  # published, not reproducible (outlier)
    {"n_valued": 0, "reproducible": False},  # no value yet
]
_ru = _fs.roll_up(_fs_rows)
check(
    "fleet_status.roll_up: 3 cells → 2 with data, 1 reproducible, 1 awaiting value",
    _ru == {"cells": 3, "with_data": 2, "reproducible": 1, "no_value": 1},
    _ru,
)
check(
    "fleet_status.cell_label: disambiguates same leaf recipe names by context path",
    _fs.cell_label(ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto")
    == "llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto",
)

# --- preflight pure checks (cluster-side wrappers untested offline): profile-completeness + secret content ---
sys.path.insert(0, str(ROOT / "scripts"))
import base64 as _b64  # noqa: E402

import preflight as _pf  # noqa: E402

for _script in ("sweep.sh", "dryrun.sh"):
    _txt = (ROOT / "scripts" / _script).read_text()
    _assign = _txt.find(': "${BENCH_NODE_SELECTOR:=}" "${BENCH_CPU_REQUEST:=16}"')
    _export = _txt.find("export ", _assign)
    _envsubst = _txt.find("envsubst", _assign)
    check(
        f"{_script}: exports BENCH_CPU_REQUEST fallback before envsubst",
        _assign >= 0
        and _export > _assign
        and _envsubst > _export
        and "BENCH_CPU_REQUEST" in _txt[_export : _txt.find("\n", _export)],
    )
with tempfile.TemporaryDirectory() as td:
    _c = Path(td)
    (_c / "rendered").mkdir()
    (_c / "rendered" / "s.yaml").write_text("schedulerName: ${SCHEDULER_NAME}\nx: ${GPU_PRODUCT}\ny: ${STOP_STAT}\n")
    check(
        "preflight.missing_profile_vars: flags empty critical vars, ignores runtime ${STOP_STAT}",
        _pf.missing_profile_vars(str(_c), {"GPU_PRODUCT": "B200", "SCHEDULER_NAME": ""}) == ["SCHEDULER_NAME"],
    )
check(
    "preflight.secret_content_issue: valid token OK; empty value flagged",
    _pf.secret_content_issue({"type": "Opaque", "data": {"token": _b64.b64encode(b"t").decode()}}) is None
    and "empty value"
    in (_pf.secret_content_issue({"type": "Opaque", "data": {"token": _b64.b64encode(b"").decode()}}) or ""),
)
# 0.6 resource summary: aggregate total/in-use/free + biggest-free node (pure)
_nodes = [
    {"metadata": {"name": "a"}, "status": {"allocatable": {"nvidia.com/gpu": "8"}}},
    {"metadata": {"name": "b"}, "status": {"allocatable": {"nvidia.com/gpu": "8"}}},
]
_gs = _pf.gpu_resource_summary(_nodes, {"a": 8}, "NVIDIA-B200")
check(
    "preflight.gpu_resource_summary: total/in-use/free + biggest-free node",
    "16 total" in _gs and "8 in use" in _gs and "8 free" in _gs and "biggest free node: 8" in _gs,
    _gs,
)

# --- provenance: the stamp carries the CURRENT recipe_hash (so RESULTS is traceable) ---
cur = run(ROOT / "scripts/recipe_hash.py", B200).stdout.split("recipe_hash:")[-1].strip()
stamp = run(ROOT / "scripts/provenance.py", B200, "--stamp").stdout
check("provenance: --stamp embeds the current recipe_hash", cur and cur in stamp, cur[:12])
# --check must pass on the committed tree (published cells cite matching hashes; pre-run cells skip)
r = run(ROOT / "scripts/provenance.py", "--check")
check(
    "provenance: --check passes on committed tree",
    r.returncode == 0,
    (r.stdout or "").strip().splitlines()[-1][:120],
)

# --- idle_guard: --sum-tokens sums vLLM generation counters (the flatline signal), excludes prompt tokens ---
_metrics = (
    'vllm:generation_tokens_total{model_name="m"} 1000.0\n'
    'vllm:generation_tokens_total{model_name="m2"} 234\n'
    'vllm:prompt_tokens_total{model_name="m"} 99999\n'
    "# HELP vllm:num_requests_running gauge\n"
    "vllm:num_requests_running 3\n"
)
_ig = subprocess.run(
    ["bash", str(ROOT / "scripts/idle_guard.sh"), "--sum-tokens"],
    input=_metrics,
    capture_output=True,
    text=True,
)
check(
    "idle_guard --sum-tokens: sums generation_tokens (1234), excludes prompt/gauge lines",
    _ig.stdout.strip() == "1234",
    _ig.stdout.strip() + " " + _ig.stderr[:80],
)


# --- idle_guard --pod-crashlooping (server-health): detect a dead server pod from `get pod -o json` ---
def _crashloop(js):
    return subprocess.run(
        ["bash", str(ROOT / "scripts/idle_guard.sh"), "--pod-crashlooping"],
        input=js,
        capture_output=True,
        text=True,
    ).stdout.strip()


check(
    "idle_guard --pod-crashlooping: CrashLoopBackOff → 1",
    _crashloop('{"status":{"containerStatuses":[{"state":{"waiting":{"reason":"CrashLoopBackOff"}}}]}}') == "1",
)
check(
    "idle_guard --pod-crashlooping: many restarts → 1",
    _crashloop('{"status":{"containerStatuses":[{"restartCount":5,"state":{"running":{}}}]}}') == "1",
)
check(
    "idle_guard --pod-crashlooping: healthy running pod → 0",
    _crashloop('{"status":{"containerStatuses":[{"restartCount":0,"state":{"running":{}}}]}}') == "0",
)


# --- idle_guard --job-verdict (B3-1): k8s omits absent numeric status fields, so an ACTIVE job renders as
#     "||1"; the old space-split read it as "1 succeeded" and tore down a live run's server. `|` keeps columns. ---
def _verdict(s):
    return subprocess.run(
        ["bash", str(ROOT / "scripts/idle_guard.sh"), "--job-verdict"],
        input=s,
        capture_output=True,
        text=True,
    ).stdout.strip()


check(
    "idle_guard --job-verdict: active job '||1' → active (NOT complete — the B3-1 false teardown)",
    _verdict("||1") == "active",
)
check("idle_guard --job-verdict: '1||' → complete", _verdict("1||") == "complete")
check("idle_guard --job-verdict: '|1|' → failed", _verdict("|1|") == "failed")
check(
    "idle_guard --job-verdict: '||' (all omitted) → pending",
    _verdict("||") == "pending",
)
check("idle_guard --job-verdict: empty → pending", _verdict("") == "pending")
check(
    "idle_guard --job-verdict: succeeded precedence over active ('1||1') → complete",
    _verdict("1||1") == "complete",
)
_idle_guard_src = (ROOT / "scripts/idle_guard.sh").read_text()
check(
    "idle_guard: a deleted Job reaches cleanup under set -e",
    'if JOB="$(find_job)"; then' in _idle_guard_src,
)
check(
    "idle_guard: API lookup errors exit without teardown",
    "Kubernetes API remained unavailable — guard exiting non-zero without teardown" in _idle_guard_src
    and 'if [ "$LOOKUP_RC" -eq 2 ]' in _idle_guard_src,
)
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    cell = tmp / "cell"
    cell.mkdir()
    (cell / "recipe.yaml").write_text("envelope:\n  name: auth-safe-cell\n")
    fake_bin = tmp / "bin"
    fake_bin.mkdir()
    kubectl_log = tmp / "kubectl.log"
    fake_kubectl = fake_bin / "kubectl"
    fake_kubectl.write_text(
        "#!/bin/sh\n"
        'printf \'%s\\n\' "$*" >> "$KLOG"\n'
        "echo 'Unable to connect to the server: expired credentials' >&2\n"
        "exit 1\n"
    )
    fake_kubectl.chmod(0o755)
    profile_name = ".selftest-idle-lookup"
    profile_path = ROOT / "cluster-profiles" / f"{profile_name}.env"
    profile_path.write_text("NAMESPACE=selftest\n")
    try:
        env = os.environ.copy()
        env["PATH"] = f"{fake_bin}:{env['PATH']}"
        env["KLOG"] = str(kubectl_log)
        auth_failure = subprocess.run(
            [
                "bash",
                str(ROOT / "scripts/idle_guard.sh"),
                str(cell),
                profile_name,
                "r1",
                "--poll",
                "0",
                "--grace",
                "1",
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=15,
        )
    finally:
        profile_path.unlink(missing_ok=True)
    kubectl_calls = kubectl_log.read_text() if kubectl_log.exists() else ""
    check(
        "idle_guard: transient API failures never scale a live server",
        auth_failure.returncode == 1
        and "API remained unavailable" in auth_failure.stdout
        and " scale " not in f" {kubectl_calls} ",
        f"rc={auth_failure.returncode} calls={kubectl_calls!r}",
    )


# --- idle_guard --pod-pending (GB300): a Job pod stuck before Running (ContainerCreating / image error) squats
#     the GPU doing no work — the live c8 sat in ContainerCreating ~2h. Detect it so the loop can fail fast. ---
def _pending(js):
    return subprocess.run(
        ["bash", str(ROOT / "scripts/idle_guard.sh"), "--pod-pending"],
        input=js,
        capture_output=True,
        text=True,
    ).stdout.strip()


check(
    "idle_guard --pod-pending: ContainerCreating (phase Pending) → 1",
    _pending(
        '{"status":{"phase":"Pending","containerStatuses":[{"state":{"waiting":{"reason":"ContainerCreating"}}}]}}'
    )
    == "1",
)
check(
    "idle_guard --pod-pending: ImagePullBackOff → 1",
    _pending('{"status":{"phase":"Pending","containerStatuses":[{"state":{"waiting":{"reason":"ImagePullBackOff"}}}]}}')
    == "1",
)
check(
    "idle_guard --pod-pending: running pod → 0",
    _pending('{"status":{"phase":"Running","containerStatuses":[{"state":{"running":{}}}]}}') == "0",
)
check(
    "idle_guard --pod-pending: succeeded pod → 0",
    _pending('{"status":{"phase":"Succeeded","containerStatuses":[{"state":{"terminated":{"exitCode":0}}}]}}') == "0",
)

# --- capacity.gpu_holders: who holds the GPUs (busiest first), llmb-owned vs external, terminal/CPU excluded ---
import capacity as _cap  # noqa: E402

_pods = [
    {
        "metadata": {"namespace": "ray", "name": "ray-head", "labels": {}},
        "spec": {
            "nodeName": "n1",
            "containers": [{"resources": {"requests": {"nvidia.com/gpu": "8"}}}],
        },
        "status": {"phase": "Running"},
    },
    {
        "metadata": {
            "namespace": "team-b",
            "name": "nemo",
            "labels": {"app.kubernetes.io/managed-by": "llmb-recipe"},
        },
        "spec": {
            "nodeName": "n2",
            "containers": [{"resources": {"requests": {"nvidia.com/gpu": "4"}}}],
        },
        "status": {"phase": "Running"},
    },
    {
        "metadata": {
            "namespace": "x",
            "name": "done",
            "labels": {},
        },  # terminal → excluded
        "spec": {
            "nodeName": "n1",
            "containers": [{"resources": {"requests": {"nvidia.com/gpu": "2"}}}],
        },
        "status": {"phase": "Succeeded"},
    },
    {
        "metadata": {
            "namespace": "x",
            "name": "offnode",
            "labels": {},
        },  # not on our nodes → excluded
        "spec": {
            "nodeName": "other",
            "containers": [{"resources": {"requests": {"nvidia.com/gpu": "1"}}}],
        },
        "status": {"phase": "Running"},
    },
]
_hold = _cap.gpu_holders(_pods, ["n1", "n2"])
check(
    "capacity.gpu_holders: busiest-first, terminal + off-node excluded",
    [h["gpus"] for h in _hold] == [8, 4],
)
check(
    "capacity.gpu_holders: llmb ownership flagged (managed-by=llmb-recipe)",
    _hold[0]["llmb"] is False and _hold[1]["llmb"] is True,
)

# --- capacity.evaluate_requirement: the machine-checkable launch/pre-deploy gate (exit 0 met / 3 short) ---
# fleet needs ≥6 nodes with ≥4 free each. 6 such nodes present (+2 half-full) → met.
_ok, _r = _cap.evaluate_requirement([4, 4, 4, 4, 4, 4, 2, 2], need_nodes=6, node_gpus=4)
check(
    "capacity.evaluate_requirement: 6×≥4-free satisfies --require-nodes 6 --node-gpus 4",
    _ok is True,
    _r,
)
# only 5 nodes have ≥4 free → SHORT.
_ok2, _r2 = _cap.evaluate_requirement([4, 4, 4, 4, 4, 3, 2], need_nodes=6, node_gpus=4)
check("capacity.evaluate_requirement: 5×≥4-free is SHORT of 6", _ok2 is False, _r2)
# total-free requirement, independent of per-node shape.
_ok3, _r3 = _cap.evaluate_requirement([2, 2, 2, 1], need_free_total=8)
check("capacity.evaluate_requirement: total free 7 is SHORT of 8", _ok3 is False, _r3)
_ok4, _r4 = _cap.evaluate_requirement([2, 2, 2, 2], need_free_total=8)
check("capacity.evaluate_requirement: total free 8 meets ≥8", _ok4 is True, _r4)
# both requirements combine — nodes met but total short → overall SHORT.
_ok5, _r5 = _cap.evaluate_requirement([4, 4], need_nodes=2, node_gpus=4, need_free_total=100)
check(
    "capacity.evaluate_requirement: combined reqs — all must hold (nodes ok, total short → SHORT)",
    _ok5 is False and len(_r5) == 2,
    _r5,
)


# --- dryrun --classify: a leftover ${VAR} is a REAL error only if the resolver owns it; the bench runner's
# own placeholders (${STOP_STAT}, ${KARCH}, ...) must be PRESERVED, not flagged (the GB200 false-positive
# that failed clean dry-runs and blocked run.sh's preflight). One shared classifier, exercised both ways. ---
def _classify(owned, *toks):
    r = subprocess.run(
        ["bash", str(ROOT / "scripts/dryrun.sh"), "--classify", owned, *toks],
        capture_output=True,
        text=True,
    )
    out = dict(line.split(":", 1) for line in r.stdout.strip().splitlines())
    return out.get("BAD", ""), out.get("KEPT", "")


_bad, _kept = _classify("NAMESPACE RUN_ID OWNER", "${STOP_STAT}", "${KARCH}", "${NAMESPACE}")
check(
    "dryrun --classify: bench runner placeholders preserved, owned cluster var flagged",
    "STOP_STAT" in _kept and "KARCH" in _kept and "NAMESPACE" in _bad and "STOP_STAT" not in _bad,
    f"BAD={_bad} KEPT={_kept}",
)
_bad2, _kept2 = _classify("__ALL__", "${NO_INTERNET_DNS_IP}")
check(
    "dryrun --classify: __ALL__ (server) flags every leftover as unresolved",
    "NO_INTERNET_DNS_IP" in _bad2 and "${" not in _kept2,
    f"BAD={_bad2} KEPT={_kept2}",
)

# --- profile_resolver: profile resolution is pure given an injected reachability probe ---
sys.path.insert(0, str(ROOT / "scripts"))
import profile_resolver as _pr  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    _pd = Path(td)
    (_pd / "alpha.env").write_text('NAMESPACE="a"\nKUBE_CONTEXT="ctx-alpha"\n')
    (_pd / "beta.env").write_text('NAMESPACE="b"\n')  # no context pinned
    (_pd / "beta.env.example").write_text('NAMESPACE="x"\n')  # must be excluded
    (_pd / "_template.env.example").write_text('NAMESPACE="t"\n')  # must be excluded
    check(
        "profile_resolver.list_profiles: real profiles only (no .example / _template)",
        _pr.list_profiles(_pd) == ["alpha", "beta"],
        str(_pr.list_profiles(_pd)),
    )

    def _up(ctx):
        return True  # reachable

    def _down(ctx):
        return False

    check(
        "resolve: missing profile → NOT_FOUND + lists existing + init hint",
        (_r := _pr.resolve("ghost", profiles_dir=_pd, probe=_up)).status == _pr.NOT_FOUND
        and "alpha" in _r.message
        and "profile init" in _r.message
        and _r.exit_code == 1,
    )
    check(
        "resolve: pinned context + reachable → OK",
        (_r := _pr.resolve("alpha", profiles_dir=_pd, probe=_up)).status == _pr.OK
        and _r.context == "ctx-alpha"
        and _r.ok
        and _r.exit_code == 0,
    )
    check(
        "resolve: pinned context + unreachable → UNREACHABLE (with reconnect hint)",
        (_r := _pr.resolve("alpha", profiles_dir=_pd, probe=_down)).status == _pr.UNREACHABLE
        and "unreachable" in _r.message
        and _r.exit_code == 1,
    )
    # Gap-2 detect-and-guide: the UNREACHABLE message prints an EXACT copy/paste login command. With no
    # CONNECT_CMD on the profile it derives `tsh kube login <ctx>`; with one it echoes it verbatim.
    check(
        "resolve: UNREACHABLE prints a derived `tsh kube login <ctx>` when no CONNECT_CMD",
        "tsh kube login ctx-alpha" in _r.message,
        _r.message,
    )
    (_pd / "gamma.env").write_text(
        'NAMESPACE="g"\nKUBE_CONTEXT="ctx-gamma"\n'
        'CONNECT_CMD="tsh login --proxy=tp.example.com:443 && tsh kube login ctx-gamma"\n'
    )
    _rg = _pr.resolve("gamma", profiles_dir=_pd, probe=_down)
    check(
        "resolve: UNREACHABLE echoes the profile's CONNECT_CMD verbatim (the exact SSO login)",
        "tsh login --proxy=tp.example.com:443 && tsh kube login ctx-gamma" in _rg.message,
        _rg.message,
    )
    # connect_hint is pure: CONNECT_CMD > derived tsh > generic.
    check(
        "connect_hint: CONNECT_CMD wins verbatim",
        _pr.connect_hint({"CONNECT_CMD": "my login", "KUBE_CONTEXT": "c"}) == "my login",
    )
    check(
        "connect_hint: derives `tsh kube login <ctx>` from context",
        _pr.connect_hint({"KUBE_CONTEXT": "kctx"}).startswith("tsh kube login kctx"),
    )
    check(
        "connect_hint: no context → generic instruction (still names CONNECT_CMD)",
        "CONNECT_CMD" in _pr.connect_hint({}),
    )
    _probed = []
    check(
        "resolve: no pinned context → OK without ever probing (backward-compatible)",
        (_r := _pr.resolve("beta", profiles_dir=_pd, probe=lambda c: _probed.append(c) or True)).status == _pr.OK
        and _r.context is None
        and _probed == [],
    )
    check(
        "resolve: no --cluster, current context matches exactly one → auto-select",
        (_r := _pr.resolve(None, profiles_dir=_pd, current_context="ctx-alpha", probe=_up)).status == _pr.OK
        and _r.name == "alpha",
    )
    check(
        "resolve: no --cluster, no unambiguous match → AMBIGUOUS (asks for --cluster)",
        (_r := _pr.resolve(None, profiles_dir=_pd, current_context="ctx-unknown", probe=_up)).status == _pr.AMBIGUOUS
        and _r.exit_code == 2,
    )
    # profile validate (0.4): completeness + reachability, pure given the probe.
    (_pd / "full.env").write_text(
        'NAMESPACE="n"\nGPU_PRODUCT="NVIDIA-B200"\nIMAGE_PULL_SECRET="s"\n'
        'MODEL_CACHE_PVC="p"\nKUBE_CONTEXT="ctx-full"\n'
    )
    _ok, _ln = _pr.validate_profile("full", profiles_dir=_pd, probe=_up)
    check(
        "validate: complete + reachable profile → ok",
        _ok and any("reachable" in line for line in _ln),
        str(_ln),
    )
    _ok2, _ln2 = _pr.validate_profile("full", profiles_dir=_pd, probe=_down)
    check(
        "validate: complete but unreachable pinned context → not ok",
        not _ok2 and any("unreachable" in line for line in _ln2),
    )
    _ok3, _ln3 = _pr.validate_profile("beta", profiles_dir=_pd, probe=_up)  # beta has only NAMESPACE
    check(
        "validate: missing required vars → not ok + names them",
        not _ok3 and any("missing required vars" in line and "GPU_PRODUCT" in line for line in _ln3),
        str(_ln3),
    )
    check(
        "validate: unknown profile → not ok + init hint",
        _pr.validate_profile("ghost", profiles_dir=_pd, probe=_up)[0] is False,
    )
    # GB200 round-2: a COMPLETE but UNPINNED profile is ok (legacy single-cluster runs on ambient), but must
    # carry the no-KUBE_CONTEXT advisory so the CLI renders the "UNPINNED" caveat instead of a clean bill.
    (_pd / "unpinned.env").write_text(
        'NAMESPACE="n"\nGPU_PRODUCT="NVIDIA-B200"\nIMAGE_PULL_SECRET="s"\n' 'MODEL_CACHE_PVC="p"\n'
    )  # complete required, NO KUBE_CONTEXT
    _oku, _lnu = _pr.validate_profile("unpinned", profiles_dir=_pd, probe=_up)
    check(
        "validate: complete but unpinned → ok, and flags 'no KUBE_CONTEXT pinned' (drives UNPINNED caveat)",
        _oku and any("no KUBE_CONTEXT pinned" in line for line in _lnu),
        str(_lnu),
    )

# --- target compatibility gate: refuse a recipe on the wrong hardware, statically ---
check(
    "compat: matching GPU (B200 recipe on NVIDIA-B200 cluster) → no issues",
    _pr.check_target_compat({"gpu_type": "B200", "arch": "amd64"}, {"GPU_PRODUCT": "NVIDIA-B200"}) == [],
)
check(
    "compat: GB200 recipe on a GB300 cluster → BLOCKED (both arm64 — the live arch guard misses this)",
    any(
        "GPU target mismatch" in i
        for i in _pr.check_target_compat({"gpu_type": "GB200", "arch": "arm64"}, {"GPU_PRODUCT": "NVIDIA-GB300"})
    ),
)
check(
    "compat: B200 recipe on a GB300 cluster → BLOCKED",
    _pr.check_target_compat({"gpu_type": "B200"}, {"GPU_PRODUCT": "NVIDIA-GB300"}) != [],
)
check(
    "compat: GPU_TYPE override lets a non-standard product label match",
    _pr.check_target_compat({"gpu_type": "GB200"}, {"GPU_PRODUCT": "custom-label", "GPU_TYPE": "GB200"}) == [],
)
check(
    "compat: arch mismatch flagged when the profile declares ARCH",
    any(
        "arch mismatch" in i
        for i in _pr.check_target_compat(
            {"gpu_type": "GB200", "arch": "arm64"},
            {"GPU_PRODUCT": "NVIDIA-GB200", "ARCH": "amd64"},
        )
    ),
)
check(
    "compat: colocated agent on wrong-arch target → BLOCKED",
    any(
        "agent arch mismatch" in i
        for i in _pr.check_target_compat(
            {
                "gpu_type": "B200",
                "arch": "amd64",
                "agent": {"placement": "colocated", "arch": "arm64"},
            },
            {"GPU_PRODUCT": "NVIDIA-B200"},
        )
    ),
)
check(
    "compat: unknown fields don't false-positive (no gpu_type or GPU_PRODUCT → allow)",
    _pr.check_target_compat({}, {}) == [],
)

# --- lane routing: shared implementation for (scenario, mode) → stage/bench scripts ---
import lane as _lane  # noqa: E402

check(
    "lane: llm-perf → stage-dataset + sweep (mode ignored), kind=bench",
    _lane.resolve_lane("llm-perf", "mooncake-trace")
    == {"stage": "stage-dataset.sh", "bench": "sweep.sh", "needs": "", "kind": "bench"},
)

# --- recipe-change impact guard: DRIFT when the published record no longer matches ---
import re as _re  # noqa: E402

import provenance as _prov  # noqa: E402

with tempfile.TemporaryDirectory() as td:
    _dst = Path(td) / "cell"
    # SYNTHESIZE a published cell rather than depending on a live one: every committed cell is currently
    # `wip` (the KVBM numbers were withdrawn for re-baseline), and a fixture that needs a PUBLISHED
    # cell should not silently erode the moment the published set changes. Copy a real cell for its
    # recipe.yaml + rendered/ (so recipe_hash is real), then mark it published and cite that exact hash.
    shutil.copytree(PUBLISHED, _dst)
    _rc = _dst / "recipe.yaml"
    _rc.write_text(_re.sub(r"^  status: \w+$", "  status: runs", _rc.read_text(), count=1, flags=_re.M))
    (_dst / "RESULTS.md").write_text(
        f"# fixture\n\n| recipe_hash | `{__import__('recipe_hash').recipe_hash(_dst)}` |\n"
    )
    check(
        "impact: unmodified published cell → MATCH",
        _prov.impact_one(_dst)[0] == "MATCH",
        _prov.impact_one(_dst)[0],
    )
    _res = _dst / "RESULTS.md"
    _cited = _prov.cited_hash(_dst)  # the recipe_hash the record cites (== current)
    _res.write_text(_res.read_text().replace(_cited, "f" * 64))  # make the cited recipe_hash stale
    _st, _msg = _prov.impact_one(_dst)
    check(
        "impact: published record drifted from recipe → DRIFT + remediation",
        _st == "DRIFT" and "status: wip" in _msg and "build_catalog" in _msg,
        _st,
    )
    # flip status out of PUBLISHED → advisory says there's nothing to invalidate
    _rc = _dst / "recipe.yaml"
    _rc.write_text(_rc.read_text().replace("status: runs", "status: wip", 1))
    check(
        "impact: non-published cell → UNPUBLISHED (no record to invalidate)",
        _prov.impact_one(_dst)[0] == "UNPUBLISHED",
        _prov.impact_one(_dst)[0],
    )

# --- observability helpers (status/jobs/logs): pure parts unit-tested, no cluster ---
import observe as _obs  # noqa: E402

check(
    "observe.pick_job: prefers <name>-bench-<run-id> over newest",
    _obs.pick_job(
        [
            ("c-bench-r1", "2026-01-01T00:00:00Z"),
            ("c-bench-r2", "2026-02-01T00:00:00Z"),
        ],
        "r1",
        "c",
    )
    == "c-bench-r1",
)
check(
    "observe.pick_job: resolves alternative job-kind prefixes",
    _obs.pick_job([("c-eval-r1", "2026-01-01T00:00:00Z")], "r1", "c") == "c-eval-r1",
)
check(
    "observe.pick_job: no run-id → newest by timestamp",
    _obs.pick_job(
        [("c-bench-r1", "2026-01-01T00:00:00Z"), ("c-eval-r2", "2026-02-01T00:00:00Z")],
        None,
        "c",
    )
    == "c-eval-r2",
)
check("observe.pick_job: empty → None", _obs.pick_job([], "r1", "c") is None)
_ji = [
    {
        "metadata": {
            "name": "c-bench-p1",
            "labels": {"llmb.nvidia.com/run-id": "p1"},
            "creationTimestamp": "2026-07-16T00:00:00Z",
        },
        "status": {"active": 1},
    }
]
check(
    "observe.format_jobs: renders state + run-id; empty selector → note",
    "active" in _obs.format_jobs(_ji) and "p1" in _obs.format_jobs(_ji) and "no jobs" in _obs.format_jobs([]),
)
check(
    "observe.active_job_names: filters .status.active in code (field-selector is API-rejected)",
    _obs.active_job_names(
        [
            {"metadata": {"name": "a"}, "status": {"active": 1}},
            {"metadata": {"name": "b"}, "status": {"succeeded": 1}},
            {"metadata": {"name": "c"}, "status": {}},
        ]
    )
    == ["a"],
)
with tempfile.TemporaryDirectory() as td:
    _c = Path(td) / "cell"
    (_c).mkdir()
    (_c / "runs.jsonl").write_text('{"run_id":"a","cluster":"x"}\n{"run_id":"b","cluster":"y","date":"2026-07-16"}\n')
    check("observe.last_run: last ledger entry", _obs.last_run(_c)["run_id"] == "b")
_rows = dict(_obs.status_rows(PUBLISHED, {}, {}))  # offline (no live facts): recipe/rendered/published/last-run
# The panel's SHAPE is invariant; the publish-state STRING depends on whether the cell is published.
# Every cell is currently `wip` (numbers withdrawn for re-baseline), so assert the shape unconditionally
# and the wording per state — this keeps working in both worlds instead of pinning one.
check(
    "observe.status_rows: offline panel includes publish state + rendered + last-run",
    "published" in _rows and "rendered" in _rows and "last run" in _rows,
    str(_rows),
)
check(
    "observe.status_rows: publish state reflects the cell's actual status",
    ("matches current recipe" in _rows["published"]) or ("not published" in _rows["published"]),
    str(_rows["published"]),
)
# observe.artifacts_summary (B2-4): post-TTL run record from persisted results/ (no cluster)
with tempfile.TemporaryDirectory() as td:
    _rd = Path(td) / "run42"
    _rd.mkdir()
    (_rd / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "run42",
                "model_name": "nemotron-ultra-3",
                "server_runtime": "vllm",
                "wall_seconds_total": 3600,
                "executed_sequence": [32, 64],
                "sweep_stop_reason": "fixed_complete",
                "sweep_steps": [
                    {"concurrency": 32, "aiperf_exit": 0, "breaches": []},
                    {"concurrency": 64, "aiperf_exit": 0, "breaches": ["tpot"]},
                ],
            }
        )
    )
    (_rd / "metrics_summary.csv").write_text("x")
    (_rd / "_fetch_status.json").write_text(
        json.dumps(
            {
                "receipt_version": 2,
                "complete": True,
                "failed": [],
                "files_written": 9,
                "bytes_written": 900_000,
                "remote_files": 9,
                "remote_bytes": 900_000,
                "reconciled": True,
            }
        )
    )
    _sum = _obs.artifacts_summary(_rd)
    check(
        "observe.artifacts_summary: renders per-rung + wall + publishable/fetch signals",
        "c32: ✓ ok" in _sum
        and "c64:" in _sum
        and "tpot" in _sum
        and "wall 1.0h" in _sum
        and "publishable" in _sum
        and "fetch complete" in _sum,
        _sum[:120],
    )
    check(
        "observe.artifacts_summary: missing dir → fetch hint",
        "fetch it first" in _obs.artifacts_summary(Path(td) / "nope"),
    )

# --- failure recovery: classify a failure + emit the resume command ---
import recovery as _rec  # noqa: E402

check("recovery.classify: exit 137 → oom", _rec.classify(exit_code=137) == "oom")
check(
    "recovery.classify: explicit reason wins",
    _rec.classify(reason="hang", exit_code=1) == "hang",
)
check(
    "recovery.classify: unknown fallback",
    _rec.classify(reason="weird", exit_code=1) == "unknown",
)
check(
    "recovery.remaining_rungs: preserves order, drops done",
    _rec.remaining_rungs(["32", "64"], ["32", "64", "128", "256"]) == "128 256",
)
check(
    "recovery.resume_cmd: uses named flags + --skip-server",
    "--recipe c --cluster p" in (_c := _rec.resume_cmd("c", "p", "128 256"))
    and '--rungs "128 256"' in _c
    and "--skip-server" in _c,
)
_rep = _rec.report(
    "c",
    "p",
    reason="benchmark",
    exit_code=1,
    run_id="r7",
    rungs_done=["32"],
    rungs_all=["32", "64", "128"],
)
check(
    "recovery.report: names cause + partial-fetch + resume with remaining rungs",
    "did not complete" in _rep and "fetch_results.sh --partial r7" in _rep and '--rungs "64 128"' in _rep,
    _rep,
)

_fetch_source = (ROOT / "scripts" / "fetch_results.sh").read_text()
check(
    "fetch_results: bare run-id path remains compatible with macOS Bash 3.2",
    "mapfile" not in _fetch_source and "readarray" not in _fetch_source and "declare -A" not in _fetch_source,
)
check(
    "fetch_results: bare run-id reconnect uses durable namespace as well as exact PVC",
    '_RUN_NAMESPACE="$(_run_field namespace)"' in _fetch_source and 'NAMESPACE="$_RUN_NAMESPACE"' in _fetch_source,
)

print(f"\nselftest: {'ALL PASS ✓' if not fails else f'{len(fails)} FAILED: ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
