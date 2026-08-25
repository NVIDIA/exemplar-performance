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

"""export_record.py <cell> <run-dir> [--out FILE] — the canonical FINAL record for a published/exemplar run.

ONE self-describing JSON per run — the artifact pushed to the results database. It consolidates everything
scattered across the cell + run into a single object: identity (scenario · goal · distribution + hardware +
serving), the recipe_hash fingerprint, provenance (image · dataset · run · cluster · timing), the exemplar
metric + reference + verdict, and the full per-rung detail. Everything needed to reproduce, compare, and
analyze — in one machine-readable record.
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis/llm-perf"))
sys.path.insert(0, str(ROOT / "analysis"))
import recipe_hash as _rh
import benchmark_id as _bid  # stable benchmark identity: same across image/flag rolls (recipe_hash is not)
import goal_handlers as _gh  # the GOAL-AWARE spine: one handler per (scenario, goal)
import reproduce as _repro  # the SINGLE generator of the three-step reproduce block (README + record + RESULTS)

try:
    import yaml
except ImportError:
    sys.exit("export_record: requires pyyaml")


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# ── per-rung-across-repeats variance (the data-model extension) ───────────────────────────────────────
# A --repeat sweep runs N legs of the SAME benchmark_id. Today only the SCALAR goal-metric spread survives
# (one value/run in runs.jsonl; record.json keeps the latest curve), so per-POINT error bars are impossible.
# These helpers aggregate each rung ACROSS the N legs and persist per-rung {mean,min,max,spread_pct,n} so a
# chart can draw honest error bars. Additive + back-compat: a cell with n=1 gets NO `repeats` block (never a
# fabricated band). The per-rung stat fields hashed by scenario (all higher-is-noisier latency/throughput):
_REPEAT_KEYS = {
    "llm-perf": ["ttft_p50_ms", "tpot_p50_ms", "tps_per_gpu", "tps_per_user"],
}


def _agg_stats(values):
    """PURE: {mean,min,max,spread_pct,n} over the non-None values, or None when <2 (no band for n<2)."""
    vals = [v for v in (f(x) for x in values) if v is not None]
    if len(vals) < 2:
        return None
    mean = sum(vals) / len(vals)
    lo, hi = min(vals), max(vals)
    spread = (hi - lo) / mean * 100 if mean else 0.0
    return {
        "mean": round(mean, 6),
        "min": round(lo, 6),
        "max": round(hi, 6),
        "spread_pct": round(spread, 4),
        "n": len(vals),
    }


def merge_rung_repeats(base_rungs, legs_rungs, keys, conc_key="concurrency"):
    """PURE + deterministic: overlay a per-rung `repeats` block onto base_rungs from the per-leg rung curves.
    `legs_rungs` is a list of rung-lists (one per --repeat leg, INCLUDING the primary). Each rung is matched
    across legs by concurrency; for every stat key with ≥2 leg values a {mean,min,max,spread_pct,n} block is
    attached under rung['repeats'][key]. Additive: rungs with <2 legs get no `repeats` (honest n=1). Never
    mutates the inputs."""
    by_conc: dict = {}
    for leg in legs_rungs:
        for r in leg or []:
            try:
                c = int(float(r.get(conc_key)))
            except (TypeError, ValueError):
                continue
            slot = by_conc.setdefault(c, {k: [] for k in keys})
            for k in keys:
                v = f(r.get(k))
                if v is not None:
                    slot[k].append(v)
    out = []
    for r in base_rungs:
        r = dict(r)
        try:
            c = int(float(r.get(conc_key)))
        except (TypeError, ValueError):
            c = None
        if c is not None and c in by_conc:
            reps = {k: st for k in keys if (st := _agg_stats(by_conc[c][k])) is not None}
            if reps:
                r["repeats"] = reps
        out.append(r)
    return out


def read_leg_rungs(run_dir: Path, scenario: str, stat: str = "p50"):
    """A leg run-dir → the same per-rung shape stored in record.detail.rungs (so merge_rung_repeats can match
    it against the primary). llm-perf reads metrics_summary.csv."""
    rows = read_csv(run_dir / "metrics_summary.csv")
    return [
        {
            "concurrency": x.get("concurrency"),
            "ttft_p50_ms": f(x.get("ttft_p50_ms")),
            "tpot_p50_ms": f(x.get("itl_p50_ms")),
            "tps_per_gpu": f(x.get("throughput_per_gpu_tok_per_s")),
            "tps_per_user": f(x.get("tokens_per_s_per_user_from_itl")),
        }
        for x in rows
    ]


def read_csv(p: Path):
    if not p.is_file():
        return []
    return list(csv.DictReader([l for l in p.read_text().splitlines() if not l.startswith("#")]))


def read_first_jsonl(p: Path):
    if not p.is_file():
        return {}
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            return json.loads(line)
        except ValueError:
            return {}
    return {}


def read_json(p: Path):
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def first_trial_value(run_dir: Path, key: str):
    p = run_dir / "trial_rows.jsonl"
    if not p.is_file():
        return None
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except ValueError:
            continue
        val = row.get(key)
        if val:
            return val
    return None


def _run_id_from_path(run_dir: Path):
    """PURE: the run-id implied by the canonical layout `recipes/<cell>/runs/<run_id>/curated` (or a raw
    `results/<run_id>` dir). Used ONLY as a fallback when run_meta carries no run_id — a scratch dir with no
    `runs/` ancestor and no curated/ leaf yields None rather than a guessed name."""
    if run_dir.name == "curated":
        return run_dir.parent.name or None
    if run_dir.parent.name == "runs":
        return run_dir.name or None
    return None


def git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def max_num_seqs(sv):
    """Resolved server concurrency cap (--max-num-seqs), across agg + disagg roles — part of 'what the run was'."""
    args = list(sv.get("extra_args") or [])
    dis = sv.get("disagg") or {}
    for role in ("prefill", "decode"):
        args += list((dis.get(role) or {}).get("extra_args") or [])
    mns = None
    for a in args:
        m = re.search(r"--max-num-seqs[= ]+(\d+)", str(a))
        if m:
            mns = int(m.group(1))
    return mns


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 2:
        sys.exit(__doc__)
    # The invoked cluster profile, threaded from publish.py as --cluster=<profile> (the same value that
    # stamps runs.jsonl 'cluster', fix c713ca56). FALLBACK ONLY for provenance.cluster when run_meta.json
    # lacks cluster/profile — never overwrites a real value. Metadata: does not feed recipe_hash/benchmark_id.
    _cluster_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cluster=")), "")
    cell, run_dir = Path(args[0]).resolve(), Path(args[1]).resolve()
    r = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    env, serving, bench = r.get("envelope") or {}, r.get("serving") or {}, r.get("bench") or {}
    prov = env.get("provenance") or {}
    ex = env.get("exemplar") or {}
    scenario = env.get("scenario")
    run_meta = {}
    rm = run_dir / "run_meta.json"
    if rm.is_file():
        run_meta = json.loads(rm.read_text())
    # RUN IDENTITY, never blank when it is knowable. run_meta.run_id is stamped only by the aiperf bench Job.
    # Derive it from the path layout as a LAST resort; a real run_meta.run_id always wins, so no existing
    # record changes.
    run_id = run_meta.get("run_id") or _run_id_from_path(run_dir)
    run_manifest = read_json(run_dir / "run_manifest.json")
    run_db = read_first_jsonl(run_dir / "runs.jsonl")

    rel = cell.relative_to(ROOT)
    # DETERMINISM: prefer the git identity PINNED in run_meta (stamped by archive_run into curated/) over live
    # git, so regenerating this record from committed curated data yields byte-identical bytes regardless of the
    # working-tree HEAD. Live runs (run_meta lacks it) still fall back to the repo — unchanged behavior.
    gc = run_meta.get("git_commit") or git("rev-parse", "--short", "HEAD")
    # A fresh archive carries the hash captured before its Job was submitted. Never replace that receipt with
    # recipe_hash(cell) during a later record regeneration: a re-archive after an edit must remain visibly stale.
    launch_receipt = read_json(run_dir / "launch_attestation.json")
    launch_hash = launch_receipt.get("recipe_hash")
    if (
        launch_receipt.get("kind") == "recipe_hash_at_launch"
        and isinstance(launch_hash, str)
        and len(launch_hash) == 64
    ):
        rh = launch_hash
        hash_source = "launch_attestation"
    else:
        rh = _rh.recipe_hash(cell)  # legacy curated data predates a launch receipt
        hash_source = "legacy_archive_derived"
    bid = _bid.benchmark_id(cell)  # stable benchmark identity: unchanged across image rolls + extra_args toggles
    ds = bench.get("dataset") or {}
    # launcher.ref is the CONTROLLER image (aiperf for llm-perf) — the thing that orchestrates the benchmark.
    launcher_ref = run_meta.get("aiperf_ref") or run_meta.get("launcher_ref") or serving.get("launcher_image")
    gpu_count = run_meta.get("gpu_count") or ((env.get("requires") or {}).get("gpu") or {}).get("count")
    sec = run_meta.get("wall_seconds_total")
    gpu_hours = round(sec / 3600 * int(gpu_count), 2) if sec and gpu_count else None

    # The copy/paste-able Reproduce block comes from the SINGLE generator (scripts/reproduce.py), the same one
    # that owns every recipe README's three steps — so the record, the README and RESULTS.md can never drift
    # into three different "how to reproduce" stories.
    repro = "\n".join(_repro.three_steps(cell)[1:-1]).strip()

    record = {
        "record_schema_version": 1,
        "identity": {
            "cell": env.get("name"),
            "scenario": scenario,
            "goal": env.get("goal"),
            "distribution": env.get("distribution"),
            "model": env.get("model"),
            "gpu_type": env.get("gpu_type"),
            "gpu_count": gpu_count,
            "arch": env.get("arch"),
            "engine": env.get("engine"),
            "serving_mode": env.get("serving_mode"),
            "framework": env.get("framework"),
            "launcher": env.get("launcher"),
            "agent_placement": (env.get("agent") or {}).get("placement"),
            "agent_arch": (env.get("agent") or {}).get("arch"),
            "agent_cpu": (env.get("agent") or {}).get("cpu"),
            "agent_cpu_cores": (env.get("agent") or {}).get("cpu_cores"),
            "mode": env.get("mode"),
            "status": env.get("status"),
        },
        # experiment_id == recipe_hash: our internal name is recipe_hash; the DB-facing field is experiment_id,
        # matching training's llmb experiment_id (sha256(config + fw_version)) so k8s + slurm results join on the
        # same full-definition identity. benchmark_id is the COARSER, stable identity to compare results ACROSS
        # image rolls / extra_args toggles (it excludes both) — group by benchmark_id to compare, use recipe_hash
        # to prove byte-identical setup.
        "fingerprint": {
            "recipe_hash": rh,
            "experiment_id": rh,
            "benchmark_id": bid,
            "git_commit": gc,
            "git_ref": run_meta.get("git_ref") or git("rev-parse", "--abbrev-ref", "HEAD"),
        },
        # everything needed to know exactly what ran, where, when, on which code + data — self-contained.
        "provenance": {
            "run_id": run_id,
            "cluster": run_meta.get("cluster") or run_meta.get("profile") or _cluster_arg or None,
            "started_at_utc": run_meta.get("started_at_utc") or None,
            "completed_at_utc": run_meta.get("completed_at_utc") or None,
            "date_utc": (run_meta.get("completed_at_utc") or run_meta.get("started_at_utc") or None),
            "wall_seconds": sec,
            "gpu_count": gpu_count,
            "gpu_hours": gpu_hours,
            "image_digest": prov.get("image_digest"),
            "launcher": {
                "name": env.get("launcher"),
                "ref": launcher_ref,
                "tokenizer_id": run_meta.get("aiperf_tokenizer_id"),
                "model_repo": run_meta.get("model_repo"),
            },
            "dataset": (
                {
                    "id": ds.get("id"),
                    "sha256": ds.get("sha256"),
                    "type": ds.get("type") or run_meta.get("dataset_type"),
                    "config_path": run_meta.get("dataset_config_path"),
                    "seed": run_meta.get("dataset_seed"),
                    "num_sessions": run_meta.get("dataset_num_sessions"),
                    "max_isl": run_meta.get("dataset_max_isl"),
                }
                if (ds or run_meta.get("dataset_type"))
                else None
            ),
            "reproduce_cmd": repro,
            "run_meta": run_meta or None,  # the sweep's own metadata, verbatim — nothing dropped
        },
        # the resolved benchmark definition: the exact server + sweep that produced these numbers.
        "config": {
            "serving": {
                "engine": env.get("engine"),
                "serving_mode": env.get("serving_mode"),
                "tp": serving.get("tp"),
                "dp": serving.get("dp"),
                "max_num_seqs": max_num_seqs(serving),
                "extra_args": serving.get("extra_args"),
                "disagg": serving.get("disagg"),
            },
            "bench": {
                "sla": bench.get("sla"),
                "sweep_mode": bench.get("sweep_mode", "fixed"),
                "sweep_concurrency": bench.get("sweep_concurrency"),
                "adaptive_sweep": bench.get("adaptive_sweep"),
            },
        },
        "result": {
            "metric": ex.get("metric"),
            "unit": ex.get("unit"),
            "value": None,
            "reference": ex.get("reference"),
            "tolerance_pct": ex.get("tolerance_pct"),
        },
    }
    if hash_source == "launch_attestation":
        record["fingerprint"]["recipe_hash_source"] = hash_source

    # --- scenario-specific: the metric value + full per-rung detail, dispatched by the GOAL handler ---
    # The handler for (scenario, goal) owns result.* (value + value_source + goal-specific fields).
    # ONE loop for every (scenario, goal): the handler owns the aggregator + native⇄normalized rung mapping,
    # so there is no `if scenario ==` fork here. `run_dir` may be a raw run-dir OR a committed curated/ dir —
    # read_native_rows prefers the native summary CSV, falls back to the normalized rungs.csv (curated, raw
    # GC'd), else aggregates a raw run-dir. Pointed at curated/, this is the KEYSTONE: the record is computed
    # from COMMITTED data, never a scratch dir, so publish regenerates it byte-identically.
    handler = _gh.resolve(scenario, env.get("goal"))
    rows = handler.read_native_rows(run_dir, env.get("goal"))
    for k, v in (handler.result_fields(rows, r) if rows else {"value": None}).items():
        record["result"][k] = v
    record["detail"] = {"rungs": [handler.rung_from_native(x) for x in rows]}

    # per-rung-across-repeats variance: if this was a --repeat sweep, overlay {mean,min,max,spread,n} per rung
    # from the sibling legs so charts can draw honest error bars. Legs come from explicit --repeat-leg <run-dir>
    # (repeatable); absent → n=1, no `repeats` block (unchanged, back-compat). The primary run is always leg 0.
    leg_dirs = [Path(sys.argv[i + 1]).resolve() for i, a in enumerate(sys.argv) if a == "--repeat-leg"]
    if leg_dirs:
        stat = (bench.get("sla") or {}).get("stop_stat", "p50")
        legs_rungs = [record["detail"]["rungs"]] + [read_leg_rungs(d, scenario, stat) for d in leg_dirs]
        keys = _REPEAT_KEYS.get(scenario, [])
        record["detail"]["rungs"] = merge_rung_repeats(record["detail"]["rungs"], legs_rungs, keys)
        record["detail"]["repeat_legs"] = len(leg_dirs) + 1  # provenance: N legs contributed the variance

    out = json.dumps(record, indent=2)
    if "--out" in sys.argv:
        Path(sys.argv[sys.argv.index("--out") + 1]).write_text(out + "\n")
    else:
        print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
