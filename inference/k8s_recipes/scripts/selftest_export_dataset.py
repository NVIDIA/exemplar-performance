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

"""selftest_export_dataset.py — the upload-ready dataset export (scripts/export_dataset.py).

Guards the contract the DB loader depends on: ONE tailored schema per (scenario, goal) (shared CORE + a
per-lane extension — no column a lane can't populate), a tidy (one row per run x rung) grain, singleton group
defaults, quality flags, and byte-deterministic re-render. No cluster needed."""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_dataset as ed  # noqa: E402

ROOT = ed.ROOT
_fail = 0


def check(label, cond):
    global _fail
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        _fail += 1


def _rows(text):
    return list(csv.DictReader(io.StringIO(text)))


cells = ed.on_pipeline_cells(ROOT)

# SYNTHESIZED PUBLISHED CELL (fallback). The checks below assert that a PUBLISHED cell exports correctly —
# image_digest, served identity, ep from --enable-expert-parallel, kv_cache_dtype, a resolved
# launch_command. Those are real guarantees. But the committed corpus can legitimately have ZERO published
# cells for a stretch (right now every cell is `wip` while the KVBM numbers are re-baselined), which
# would leave only two bad options: delete the checks, or weaken them into vacuous truth.
# Instead, give them a published cell: copy the LIVE cell dir (for its REAL recipe.yaml + rendered/, which
# is where launch_command and the served identity are derived from) and add the committed record.json +
# runs/index.jsonl fixture to make it published. The assertions run UNCHANGED, and because the fixture is
# paired with the live recipe it cannot drift into fiction. A real published cell always takes precedence.
_FIXTURE = ROOT / "scripts" / "fixtures" / "published_cell"
_SYNTH_SRC = ROOT / "scripts/fixtures/sample_cells/nemotron-ultra-3-gb300-vllm-agg-pareto-c16"


def _build_synth_cell():
    """Materialize the fixture as a real on-pipeline cell dir; returns its path (or None)."""
    if not (_FIXTURE.is_dir() and _SYNTH_SRC.is_dir()):
        return None
    import shutil as _sh, tempfile as _tf

    _t = Path(_tf.mkdtemp())
    _d = _t / "recipes" / "synth-published-cell"
    _sh.copytree(_SYNTH_SRC, _d)
    for _st in ("record.json", "runs"):
        _q = _d / _st
        if _q.is_dir():
            _sh.rmtree(_q)
        elif _q.exists():
            _q.unlink()
    _sh.copy(_FIXTURE / "record.json", _d / "record.json")
    (_d / "runs").mkdir(parents=True, exist_ok=True)
    _sh.copy(_FIXTURE / "index.jsonl", _d / "runs" / "index.jsonl")
    return _t


_synth_tmp = None
if not cells and _FIXTURE.is_dir() and _SYNTH_SRC.is_dir():
    import shutil as _shutil, tempfile as _tempfile

    _synth_tmp = Path(_tempfile.mkdtemp())
    _dst = _synth_tmp / "recipes" / "synth-published-cell"
    _shutil.copytree(_SYNTH_SRC, _dst)
    for _stale in ("record.json", "runs"):
        _p = _dst / _stale
        if _p.is_dir():
            _shutil.rmtree(_p)
        elif _p.exists():
            _p.unlink()
    _shutil.copy(_FIXTURE / "record.json", _dst / "record.json")
    (_dst / "runs").mkdir(parents=True, exist_ok=True)
    _shutil.copy(_FIXTURE / "index.jsonl", _dst / "runs" / "index.jsonl")
    cells = ed.on_pipeline_cells(_synth_tmp)
    print(f"  (no published cell committed — synthesized one from {_SYNTH_SRC.name} + the fixture record)")

all_rows = [r for c in cells for r in ed._cell_rows(c)]

# FEATURE corpus: some checks below assert properties that only a cell WITH that feature can exhibit —
# EP from --enable-expert-parallel, kv_cache_dtype, a resolved launch_command. Those live on the nemotron
# cells. Whether such a cell happens to be PUBLISHED right now is unrelated to whether the extraction
# works, so pin the assertions to a corpus that always contains one: the real published cells PLUS the
# synthesized nemotron fixture. Without this the checks silently evaporate whenever the published set
# happens to hold only (say) a KVBM cell — which is exactly how they broke.
_feat_tmp = _build_synth_cell() if _synth_tmp is None else None
_feat_cells = list(cells) + (ed.on_pipeline_cells(_feat_tmp) if _feat_tmp else [])
feat_rows = _rows(ed._render_csv([r for c in _feat_cells for r in ed._cell_rows(c)], ed.SUPERSET))

# ── determinism (rendered over the superset) ────────────────────────────────────────────────────────────
a = ed._render_csv(all_rows, ed.SUPERSET)
b = ed._render_csv([r for c in cells for r in ed._cell_rows(c)], ed.SUPERSET)
check("export re-renders byte-identically (deterministic)", a == b)
rows = _rows(a)
check("export carries at least one row", len(rows) > 0)
check(
    "core id columns never empty (group_id, benchmark_id, recipe_hash, scenario)",
    all(r["group_id"] and r["benchmark_id"] and r["recipe_hash"] and r["scenario"] for r in rows),
)
# LIVE schema contract (commit 03597f7e + job_mode): n_runs is DROPPED, run_value trails CORE, and every row
# carries job_mode='inference' (the exported workload class for these serving cells; owner-agreed).
check(
    "schema is the live contract: n_runs dropped, run_value trails CORE, job_mode present",
    "n_runs" not in ed.CORE
    and "run_value" not in ed.CORE[:-1]
    and ed.CORE[-1] == "run_value"
    and "job_mode" in ed.CORE,
)
check(
    "every row emits job_mode='inference'",
    all(r["job_mode"] == "inference" for r in rows),
)

# ── ONE distinct tailored schema per (scenario, goal) ──────────────────────────────────────────────────
par, sla = (
    ed.lane_columns("llm-perf", "pareto"),
    ed.lane_columns("llm-perf", "max-concurrency-sla"),
)
check(
    "every lane shares the CORE prefix",
    all(cols[: len(ed.CORE)] == ed.CORE for cols in (par, sla)),
)
check(
    "sla_* live ONLY in max-concurrency-sla; crossing_concurrency dropped everywhere",
    "sla_pass" in sla
    and "sla_pass" not in par
    and all(
        "crossing_concurrency" not in ed.lane_columns(s, g)
        for (s, g) in [("llm-perf", "pareto"), ("llm-perf", "max-concurrency-sla")]
    ),
)
check(
    "kv_cache_usage_perc in llm-perf pareto lane",
    "kv_cache_usage_perc" in par,
)
check(
    "no lane carries a column outside its schema (CORE+ext+blobs)",
    all(
        set(ed.lane_columns(s, g)) <= set(ed.SUPERSET)
        for (s, g) in [("llm-perf", "pareto"), ("llm-perf", "max-concurrency-sla")]
    ),
)

# ── singleton group defaults ───────────────────────────────────────────────────────────────────────────
check("every row is a singleton: group_size==1", all(r["group_size"] == "1" for r in rows))
check(
    "singleton group_id == benchmark_id",
    all(r["group_id"] == r["benchmark_id"] for r in rows),
)

# ── quality + provenance ───────────────────────────────────────────────────────────────────────────────
check(
    "data_provenance is a known value on every row",
    all(
        r["data_provenance"]
        in {
            "archived",
            "reconstructed_from_record",
            "reconstructed_from_prose",
            "rerun",
        }
        for r in rows
    ),
)
# MEASURED rows = real run data (archived first-publish OR rerun re-publish OR live). Post-consolidation the
# keep-set's published cells are all `rerun` (the 256k `archived`-first-publish family was pruned), so key the
# image_digest / served-identity assertions off MEASURED rather than `archived` specifically.
measured = [r for r in rows if r["data_provenance"] in ("archived", "rerun", "live")]
check(
    "measured rows carry image_digest (from provenance, not the null fingerprint copy)",
    bool(measured) and all(r["image_digest"].startswith("sha256:") for r in measured),
)

# ── per-lane metric population + derived columns ───────────────────────────────────────────────────────
# WIP-TOLERANT: there may be NO on-pipeline llm-perf cell right now — both KVBM cells were withdrawn to
# `wip` under the whole-node policy (their numbers were measured while sharing a node), and GLM-5 is parked.
# So assert the lane SCHEMA unconditionally (columns can't silently disappear) plus the CONTRACT for every
# llm-perf row that does exist. These re-arm automatically on the next publish.
llm = [r for r in rows if r["scenario"] == "llm-perf" and r["goal"] == "pareto"]
check(
    "llm-perf pareto lane declares per-rung tps + denormalized headline columns",
    all(c in ed.lane_columns("llm-perf", "pareto") for c in ("tps_per_gpu", "cell_metric", "cell_value")),
)
check(
    "every llm-perf pareto row that exists carries per-rung tps + denormalized headline",
    all(r["tps_per_gpu"] and r["cell_metric"] and r["cell_value"] for r in llm),
)
measured_llm = [
    r
    for r in rows
    if r["scenario"] == "llm-perf" and r["ttft_p50_ms"] and r["data_provenance"] in ("archived", "rerun", "live")
]
check(
    "measured llm-perf rungs never leave error_rate_pct blank (zero errors → 0, not null)",
    all(r["error_rate_pct"] != "" for r in measured_llm),
)
# WIP-TOLERANT (post-consolidation): the max-concurrency-sla cell family was pruned to the 3 north-star
# recipes (all pareto / agentic-behavior), so there may be no max-sla rows. Assert the PLUMBING: sla_pass is a
# max-sla-lane column, any max-sla row that DOES exist carries a verdict, and pareto rows leave it blank.
sla_rows = [r for r in rows if r["goal"] == "max-concurrency-sla" and r["ttft_p50_ms"]]
check(
    "sla_pass is a max-sla-lane column + derived on any max-sla row; pareto rows leave it blank",
    "sla_pass" in ed.lane_columns("llm-perf", "max-concurrency-sla")
    and all(r["sla_pass"] in ("true", "false") for r in sla_rows)
    and all(r["sla_pass"] == "" for r in llm),
)

# ── load generator + JSON blobs ────────────────────────────────────────────────────────────────────────
check(
    "load_generator names the tool per scenario (aiperf)",
    all(r["load_generator"] == "aiperf" for r in rows),
)


def _valid(col, need):
    def ok(r):
        try:
            d = json.loads(r[col])
            return isinstance(d, dict) and all(k in d for k in need)
        except (ValueError, KeyError):
            return False

    return all(ok(r) for r in rows)


check(
    "details JSON carries the config long-tail (serving+bench)",
    _valid("details", ("serving", "bench")),
)
check(
    "cluster_details JSON present (cluster fingerprint)",
    _valid("cluster_details", ("cluster",)),
)
check(
    "launcher_command JSON names the generator",
    _valid("launcher_command", ("generator",)),
)

# ── runner attribution (run_by) — WHO ran it, fallback chain never blanks the export ─────────────────────
check("run_by is a CORE column (person-level attribution)", "run_by" in ed.CORE)
check(
    "every row carries a non-empty run_by (fallback chain never blocks the export)",
    all(str(r.get("run_by") or "").strip() for r in rows),
)

# ── model/serving reproducibility columns (parallelism + quant + weights identity) ───────────────────────
_REPRO = (
    "served_model_id",
    "model_repo",
    "model_revision",
    "ep",
    "dp",
    "pp",
    "quantization",
    "kv_cache_dtype",
    "max_model_len",
)
check("every reproducibility column is in CORE", all(c in ed.CORE for c in _REPRO))
check(
    "served_model_id + model_repo populated on measured rows (the served identity + pinned weights)",
    bool(measured) and all(r["served_model_id"] and r["model_repo"] for r in measured),
)
# TP/EP extraction: the nemotron-ultra cells run --enable-expert-parallel + --kv-cache-dtype fp8, so ep/kv
# must surface from extra_args; and a NVFP4/FP8 repo name must be sniffed into quantization.
_ep = [r for r in feat_rows if r["ep"]]
check(
    "EP surfaces from extra_args (--enable-expert-parallel → ep=true) on the cells that enable it",
    bool(_ep) and all(r["ep"] in ("true",) or str(r["ep"]).isdigit() for r in _ep),
)
check(
    "kv_cache_dtype + quantization + max_model_len populated where the recipe sets them",
    any(r["kv_cache_dtype"] and r["quantization"] and r["max_model_len"] for r in feat_rows),
)

# ── FULL launch command — the complete resolved server argv (distinct from the load-gen command) ─────────
check(
    "launch_command is a CORE_JSON blob (trails the load-gen launcher_command)",
    "launch_command" in ed.CORE_JSON and ed.CORE_JSON[-1] == "launch_command",
)


def _launch_ok(r):
    lc = r.get("launch_command") or ""
    if not lc:
        return True  # a cell with no committed rendered/ legitimately has none (never a hard fail)
    try:
        d = json.loads(lc)
    except ValueError:
        return False
    # every captured command is the RESOLVED server entrypoint (vLLM or sglang/dynamo), fully baked (no {{ }}).
    cmds = [c for cmds in d.values() for c in cmds]
    return bool(cmds) and all(
        ("python3 -m vllm" in c or "python3 -m dynamo" in c or "python3 -m sglang" in c) and "{{" not in c for c in cmds
    )


_have_lc = [r for r in feat_rows if r.get("launch_command")]
check(
    "launch_command captures the fully-resolved server argv (vLLM/sglang entrypoint, no template tokens)",
    bool(_have_lc) and all(_launch_ok(r) for r in feat_rows),
)
check(
    "launch_command is the SERVER, distinct from launcher_command (the load-gen)",
    all(r["launch_command"] != r["launcher_command"] for r in _have_lc),
)

# ── unit: extra_args parsing + exec extraction (pure helpers) ────────────────────────────────────────────
_xa = [
    "--kv-cache-dtype fp8",
    "--enable-expert-parallel",
    "--max-num-seqs 256",
    "--pipeline-parallel-size 2",
]
check(
    "_arg_value reads --flag value",
    ed._arg_value(_xa, "--kv-cache-dtype") == "fp8" and ed._arg_value(_xa, "--pipeline-parallel-size") == "2",
)
check(
    "_arg_present detects a bare boolean flag",
    ed._arg_present(_xa, "--enable-expert-parallel") and not ed._arg_present(_xa, "--enable-prefix-caching"),
)
check(
    "_quantization sniffs NVFP4 from a repo name, prefers explicit --quantization",
    ed._quantization("nvidia/Model-NVFP4", []) == "NVFP4"
    and ed._quantization("nvidia/Model-NVFP4", ["--quantization awq"]) == "awq",
)
_multi = (
    "x:\n  args:\n    - |\n      exec python3 -m vllm.foo \\\n        --model X \\\n        --tp 4\n"
    "  other:\n      exec python3 -m dynamo.frontend --http-port=8000\n"
)
_ex = ed._extract_execs(_multi)
check(
    "_extract_execs collapses a backslash-continued exec + finds every server exec",
    _ex
    == [
        "python3 -m vllm.foo --model X --tp 4",
        "python3 -m dynamo.frontend --http-port=8000",
    ],
)

# ── group-awareness (synthetic) ────────────────────────────────────────────────────────────────────────
_pc = [c for c in cells if "pareto" in c.name and "llm-perf" in str(c)][:2]
if len(_pc) == 2:
    _orig = ed._group_block
    _tag = {str(c): {"id": "st-frontier", "kind": "frontier", "varies": ["gpu_type"]} for c in _pc}
    ed._group_block = lambda cell: _tag.get(str(cell))
    try:
        gm = ed.group_map(_pc)
        gids = {v[0] for v in gm.values()}
        check(
            "grouped members share ONE real group_id (not a benchmark_id), group_size set",
            len(gm) == 2
            and len(gids) == 1
            and next(iter(gids)) not in {c.name for c in _pc}
            and all(v[1] == 2 for v in gm.values()),
        )
        check(
            "an UNGROUPED cell stays a singleton (group_id == benchmark_id, group_size 1)",
            all(r["group_id"] == r["benchmark_id"] and r["group_size"] == 1 for r in ed._cell_rows(_pc[0], {})),
        )
    finally:
        ed._group_block = _orig

print(f"\nselftest_export_dataset: {'all checks passed' if _fail == 0 else f'{_fail} FAILED'}")
raise SystemExit(1 if _fail else 0)
