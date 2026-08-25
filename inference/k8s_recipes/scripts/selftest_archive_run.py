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

"""selftest_archive_run.py — offline guards for the storage tier (archive_run.py). No cluster, no network.

Covers:
  A. FROM-RECORD (Bucket B) — reconstruct curated/rungs.csv from record.detail.rungs; the rungs.csv →
     native → detail.rungs round-trips byte-identical; the record's scalar metric recomputes; run_meta pins
     git; index.jsonl carries data_provenance=reconstructed_from_record. Idempotent (re-archive → 1 line).
  B. RESULTS SPLIT (Bucket A) — a synthetic results/ dir splits into curated/ (summary + rungs.csv +
     run_meta) and raw/ (the heavy *.prom / trial_/ report.json rest); raw is the gitignore-able tier.
  C. INDEX SCHEMA — every index line carries the evolved fields incl. data_provenance ∈ the allowed set.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import archive_run as ar  # noqa: E402
import goal_handlers as gh  # noqa: E402
import launch_attestation as la  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


RECIPE = """\
envelope:
  name: st-archive-pareto
  model: m
  gpu_type: B200
  arch: amd64
  engine: vllm
  serving_mode: aggregated
  framework: none
  scenario: llm-perf
  distribution: d
  mode: mooncake-trace
  launcher: aiperf
  goal: pareto
  status: runs
  provenance:
    image_digest: sha256:abc
  exemplar:
    metric: pareto_geomean
    unit: geomean(tps/gpu·tps/user)
    reference: 100.0
    tolerance_pct: 5
  requires:
    gpu: { count: 8 }
serving:
  tp: 8
bench:
  dataset: {id: fixture, sha256: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}
  sla: { ttft_ms: 10000, tpot_ms: 100, stop_stat: p50 }
  sweep_mode: fixed
  sweep_concurrency: [32, 64]
"""

RUNGS = [
    {
        "concurrency": 32,
        "ttft_p50_ms": 379.382611,
        "tpot_p50_ms": 14.545346,
        "tps_per_gpu": 97.200064,
        "tps_per_user": 68.750512,
        "error_rate_pct": None,
        "kv_cache_usage_perc": 7.702826,  # a populated kv value round-trips through the float repr
        "rung_wall_seconds": 78.271282,
    },  # a populated per-rung duration round-trips through the float repr
    {
        "concurrency": 64,
        "ttft_p50_ms": 464.417394,
        "tpot_p50_ms": 24.727543,
        "tps_per_gpu": 124.770989,
        "tps_per_user": 40.440735,
        "error_rate_pct": None,
        "kv_cache_usage_perc": None,
    },  # a null kv (reconstructed / raw-GC'd run) round-trips too
    # NOTE: rung 64 deliberately OMITS rung_wall_seconds — a run archived before that column existed. It
    # must round-trip back to key-ABSENT (not key-with-null), which is what keeps historical record.json
    # files regenerating byte-for-byte. Rung 32 above covers the populated case.
]


def _mk_cell(td: Path) -> Path:
    cell = td / "cell"
    cell.mkdir()
    (cell / "recipe.yaml").write_text(RECIPE)
    return cell


def _mk_record(cell: Path, value=76.2):
    rec = {
        "identity": {
            "cell": "st-archive-pareto",
            "scenario": "llm-perf",
            "goal": "pareto",
        },
        "fingerprint": {
            "recipe_hash": ar._rh.recipe_hash(cell),
            "git_commit": "deadbeef",
            "git_ref": "st-branch",
        },
        "provenance": {
            "run_id": "r1",
            "cluster": "example-gpu-cluster",
            "wall_seconds": 100,
            "gpu_count": 8,
            "completed_at_utc": "2026-07-16T00:00:00Z",
            "run_meta": {
                "run_id": "r1",
                "cluster": "example-gpu-cluster",
                "wall_seconds_total": 100,
                "gpu_count": 8,
                "completed_at_utc": "2026-07-16T00:00:00Z",
            },
        },
        "result": {"metric": "pareto_geomean", "value": value},
        "detail": {"rungs": RUNGS},
    }
    (cell / "record.json").write_text(json.dumps(rec, indent=2) + "\n")


# ── A. from-record (Bucket B) ────────────────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as _td:
    td = Path(_td)
    cell = _mk_cell(td)
    _mk_record(cell)
    res = ar.archive_from_record(cell, None, "reconstructed_from_record")
    curated = cell / "runs" / "r1" / "curated"
    check("from-record writes curated/rungs.csv", (curated / "rungs.csv").is_file())
    check(
        "from-record writes curated/run_meta.json",
        (curated / "run_meta.json").is_file(),
    )
    # rungs.csv → native → detail.rungs round-trips byte-identical
    h = gh.resolve("llm-perf", "pareto")
    rows = gh._read_csv(curated / "rungs.csv")
    rebuilt = [h.rung_from_native(h.native_from_rung(r)) for r in rows]
    check(
        "rungs.csv round-trips to the exact detail.rungs (determinism)",
        rebuilt == RUNGS,
        str(rebuilt[:1]),
    )
    _, v, _ = h.compute_metric([h.native_from_rung(r) for r in rows])
    check("scalar metric recomputes from curated rungs.csv", v == 76.2, str(v))
    rm = json.loads((curated / "run_meta.json").read_text())
    check(
        "run_meta pins git_commit/git_ref (deterministic regeneration)",
        rm.get("git_commit") == "deadbeef" and rm.get("git_ref") == "st-branch",
    )
    idx = [json.loads(l) for l in (cell / "runs" / "index.jsonl").read_text().splitlines() if l.strip()]
    check("index has exactly one line", len(idx) == 1)
    e = idx[0]
    check(
        "index carries data_provenance=reconstructed_from_record",
        e.get("data_provenance") == "reconstructed_from_record",
    )
    check(
        "index carries metric+value+recipe_hash+benchmark_id+curated",
        e.get("metric") == "pareto_geomean"
        and e.get("value") == 76.2
        and e.get("recipe_hash")
        and e.get("benchmark_id")
        and e.get("curated") == "runs/r1/curated",
    )
    csv_before = (curated / "rungs.csv").read_text()
    ar.archive_from_record(cell, None, "reconstructed_from_record")  # idempotent re-archive
    idx2 = [l for l in (cell / "runs" / "index.jsonl").read_text().splitlines() if l.strip()]
    check("re-archive is idempotent (still one index line, no duplicate)", len(idx2) == 1)
    check(
        "re-archive rewrites curated deterministically (byte-identical rungs.csv)",
        (curated / "rungs.csv").read_text() == csv_before,
    )


# ── B. results split (Bucket A) ──────────────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as _td:
    td = Path(_td)
    cell = _mk_cell(td)
    results = td / "results" / "run-xyz"
    results.mkdir(parents=True)
    # native summary the aggregator would have produced (only the columns the handler reads)
    (results / "metrics_summary.csv").write_text(
        "concurrency,ttft_p50_ms,itl_p50_ms,throughput_per_gpu_tok_per_s,tokens_per_s_per_user_from_itl,error_rate_pct\n"
        "32,379.382611,14.545346,97.200064,68.750512,\n"
        "64,464.417394,24.727543,124.770989,40.440735,\n"
    )
    (results / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "run-xyz",
                "cluster": "example-gpu-cluster",
                "wall_seconds_total": 200,
                "gpu_count": 8,
                "completed_at_utc": "2026-07-16T00:00:00Z",
            }
        )
        + "\n"
    )
    launch = la.capture(
        cell,
        "run-xyz",
        results / "launch_attestation.json",
        captured_at_utc="2026-07-16T00:00:00Z",
    )
    # heavy raw artifacts
    (results / "server.prom").write_text("# HELP x\n")
    (results / "report.json").write_text("{}")
    step = results / "concurrency_32"
    step.mkdir()
    (step / "trial_rows.jsonl").write_text('{"x":1}\n')
    out = ar.archive_results(cell, results, None, "archived")
    curated, raw = (
        cell / "runs" / "run-xyz" / "curated",
        cell / "runs" / "run-xyz" / "raw",
    )
    check(
        "Bucket A: curated has the native summary + rungs.csv + run_meta",
        (curated / "metrics_summary.csv").is_file()
        and (curated / "rungs.csv").is_file()
        and (curated / "run_meta.json").is_file(),
    )
    check(
        "Bucket A: heavy artifacts land in raw/ (the gitignored tier)",
        (raw / "server.prom").is_file()
        and (raw / "report.json").is_file()
        and (raw / "concurrency_32" / "trial_rows.jsonl").is_file(),
    )
    check(
        "Bucket A: raw/ does NOT duplicate the curated summary/run_meta",
        not (raw / "metrics_summary.csv").exists() and not (raw / "run_meta.json").exists(),
    )
    check(
        "Bucket A: index provenance=archived",
        out["entry"]["data_provenance"] == "archived",
    )
    check(
        "Bucket A: index recipe_hash is the pre-launch receipt, not an archive-time recomputation",
        out["entry"].get("recipe_hash") == launch["recipe_hash"]
        and out["entry"].get("recipe_hash_at_launch") == launch["recipe_hash"]
        and out["entry"].get("recipe_hash_source") == "launch_attestation",
    )
    # curated rungs.csv recomputes the same pareto_geomean as from-record
    h = gh.resolve("llm-perf", "pareto")
    rows = gh._read_csv(curated / "rungs.csv")
    _, v, _ = h.compute_metric([h.native_from_rung(r) for r in rows])
    check(
        "Bucket A: pareto_geomean computes from the archived curated rungs",
        v == 76.2,
        str(v),
    )

    # ── B2. IDEMPOTENCY GUARD (G18 data-loss regression) — re-archiving an ALREADY-ARCHIVED run must NOT
    # destroy it. Point archive_results at cell/runs/<run_id> itself (a re-score/re-publish, e.g. a goal
    # change): before the guard this rmtree'd its own source curated/ (deleting metrics_summary.csv, emptying
    # rungs.csv, publishing metric None) and nested raw/ into raw/raw + raw/curated.
    already = cell / "runs" / "run-xyz"
    summ_before = (curated / "metrics_summary.csv").read_text()
    rungs_before = (curated / "rungs.csv").read_text()
    out2 = ar.archive_results(cell, already, None, "rerun")
    check(
        "G18: re-archive of an already-archived run PRESERVES curated/metrics_summary.csv",
        (curated / "metrics_summary.csv").is_file() and (curated / "metrics_summary.csv").read_text() == summ_before,
    )
    check(
        "G18: re-archive keeps rungs.csv populated (not emptied to a bare header)",
        (curated / "rungs.csv").read_text() == rungs_before and len(rungs_before.splitlines()) == 3,
    )
    check(
        "G18: re-archive does NOT nest raw/ (no raw/raw or raw/curated)",
        not (raw / "raw").exists() and not (raw / "curated").exists(),
    )
    check(
        "G18: re-archive still recomputes the metric from the surviving curated",
        out2["entry"].get("value") == 76.2,
        str(out2["entry"].get("value")),
    )
    idxg = [l for l in (cell / "runs" / "index.jsonl").read_text().splitlines() if l.strip()]
    check(
        "G18: re-archive stays idempotent (one index line for the run)",
        len(idxg) == 1,
        str(len(idxg)),
    )

    # This is the motivating re-archive attack: alter the recipe after the run, then archive the same old
    # artifacts again. The index must keep the launch hash, so provenance-check will reject it as stale.
    recipe = (cell / "recipe.yaml").read_text().replace("tp: 8", "tp: 2")
    (cell / "recipe.yaml").write_text(recipe)
    out3 = ar.archive_results(cell, already, None, "rerun")
    check(
        "launch receipt defeats re-archive after a recipe edit",
        out3["entry"].get("recipe_hash") == launch["recipe_hash"]
        and out3["entry"].get("recipe_hash") != ar._rh.recipe_hash(cell),
        f"launch={launch['recipe_hash'][:12]} current={ar._rh.recipe_hash(cell)[:12]}",
    )


# ── C. index schema / provenance validation ──────────────────────────────────────────────────────────────
check(
    "archive_run declares the four data_provenance levels",
    ar.PROVENANCES == {"archived", "reconstructed_from_record", "reconstructed_from_prose", "rerun"},
)

# ── D. launcher_argv: lift aiperf's literal cli_command (input_config.cli_command) per rung ──────────────
with tempfile.TemporaryDirectory() as td:
    rd = Path(td)
    for c in (8, 16):
        pj = rd / f"concurrency_{c}" / "logs" / "aiperf"
        pj.mkdir(parents=True)
        (pj / "profile_export_aiperf.json").write_text(
            json.dumps({"input_config": {"cli_command": f"aiperf profile --concurrency {c} --model m"}})
        )
    argv = ar.launcher_argv(rd)
    check(
        "launcher_argv lifts cli_command per rung, sorted",
        argv
        == [
            {"concurrency": 16, "command": "aiperf profile --concurrency 16 --model m"},
            {"concurrency": 8, "command": "aiperf profile --concurrency 8 --model m"},
        ],
        str(argv),
    )
    check(
        "launcher_argv → None when no aiperf profile_export present (reconstructed/non-aiperf)",
        ar.launcher_argv(Path(td) / "empty") is None,
    )

print(("\nFAIL: " + ", ".join(fails)) if fails else "\nselftest_archive_run: all checks passed")
sys.exit(1 if fails else 0)
