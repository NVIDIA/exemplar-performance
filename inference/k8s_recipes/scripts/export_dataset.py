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

"""Export normalized per-run benchmark data.

ONE tailored schema PER (scenario, goal). Different lanes genuinely need different metrics/params — SLA
crossing only means something for max-concurrency-sla — so a single wide union with lane-irrelevant nulls is
the wrong shape. Instead: a shared CORE (identity/provenance/hardware/cost/headline + the JSON blobs) + a
per-lane EXTENSION.

Emitted  : per-RUN `results/<run_id>/llmb_inference_export.csv` (EXPORT_FILENAME) — the DB-upload unit,
           co-located with that run's other (GITIGNORED) artifacts. EPHEMERAL: `publish <run_id>` (soon =
           push-to-DB) reads it → the DB stores it; the repo keeps only the cell's rolled-up record, never a
           committed export. No repo-side aggregate: the DB owns the union across runs.
Grain    : one row per (run × rung). Each run — a manual re-run OR an automated `--repeat N` variance leg —
           has its OWN unique run_id, folder, and rows; repeats are NEVER averaged into one (the variance band
           lives only in aggregate.json). The DB does any cross-run/cross-cell combination. Pure function of
           committed bytes (record.json + runs/index.jsonl + each run's curated/rungs.csv + run_meta.json) —
           byte-reproducible and CI-guardable (`make export-check`): regenerate + byte-diff, a hand-edit fails.
Grouping : group_id + group_size model DECLARED grouping (an operator-defined combination, e.g. a cross-hardware
           frontier). group_size is meaningless without group_id — they are a pair. A cell with no
           `envelope.group` block is a SINGLETON: group_id = benchmark_id, group_size 1. The DB may honour these
           or derive its own combinations from the dimensional columns.
Cost     : PRICE-FREE — ship gpu_count/gpu_type/gpu_hours + goodput_per_hour; the DB applies its own
           $/gpu-hr[gpu_type] table. No $-denominated column anywhere.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import results_dir as _rd  # noqa: E402 — shared implementation for results/<scenario>/<goal>/<name>/<run_id>
import runner_identity as _ri  # noqa: E402 — WHO ran it (run_by): stored → git email → $USER → "unknown"

try:
    import yaml
except ImportError:  # group tagging is optional; degrade to singleton-only if no yaml
    yaml = None

ROOT = Path(__file__).resolve().parent.parent
EXPORT_SCHEMA_VERSION = 1

# ── schema: shared CORE + per-lane EXTENSION + trailing JSON blobs ───────────────────────────────────────
CORE = [
    "export_schema_version",
    "group_id",
    "group_size",
    "benchmark_id",
    "recipe_hash",
    "run_id",
    "cluster",
    "scenario",
    "goal",
    # ── MODEL: what was served (model = family; served_model_id = the --served-model-name clients request;
    #    model_repo@model_revision = the exact pinned weights) ────────────────────────────────────────────
    "model",
    "served_model_id",
    "model_repo",
    "model_revision",
    "gpu_type",
    "gpu_count",
    "arch",
    "engine",
    "serving_mode",
    # ── SERVING/FRAMEWORK: how it was served — the parallelism + quant + context knobs that make a row
    #    reproducible (tp already existed; ep/dp/pp/quantization/kv_cache_dtype/max_model_len are new) ─────
    "tp",
    "ep",
    "dp",
    "pp",
    "quantization",
    "kv_cache_dtype",
    "max_model_len",
    "distribution",
    "mode",
    "job_mode",
    "load_generator",
    "date_utc",
    "git_commit",
    "image_digest",
    "data_provenance",
    "run_by",  # run_by: WHO ran it (person-level attribution)
    "wall_seconds",
    "gpu_hours",  # price-FREE cost basis (DB applies $/gpu-hr[gpu_type])
    "concurrency",
    "cell_metric",
    "cell_unit",
    "cell_value",
    "run_value",
]
# long-tail blobs, always last. launch_command = the COMPLETE resolved server launch argv (the ultimate
# reproducibility artifact) — distinct from launcher_command (the LOAD-GEN invocation).
CORE_JSON = ["details", "cluster_details", "launcher_command", "launch_command"]

# Per-run upload filename. Renamed from the bare `export.csv` to a self-describing name so the artifact is
# unambiguous inside an exported archive and on disk (the run-id is already in the parent path). The
# `inference` token mirrors job_mode; a future training lane would get its own name.
EXPORT_FILENAME = "llmb_inference_export.csv"

_LLM_METRICS = [
    "ttft_p50_ms",
    "tpot_p50_ms",
    "tps_per_gpu",
    "tps_per_user",
    "error_rate_pct",
    "kv_cache_usage_perc",
    # HOW LONG THIS ROW TOOK. Distinct from the run-level `wall_seconds`, which is the whole
    # sweep and is therefore identical on every rung row — useless for "how long did c=64 run".
    "rung_wall_seconds",
]
_SLA_ONLY = [
    "sla_ttft_ms",
    "sla_tpot_ms",
    "sla_pass",
]  # max-concurrency-sla ONLY (the ceiling itself = cell_value)

# one distinct extension per (scenario, goal).
EXT = {
    ("llm-perf", "pareto"): _LLM_METRICS,
    ("llm-perf", "max-concurrency-sla"): _LLM_METRICS + _SLA_ONLY,
}
# the superset of every column any lane can emit (rows are built full, then projected onto a lane's schema).
SUPERSET = list(dict.fromkeys(CORE + _LLM_METRICS + _SLA_ONLY + CORE_JSON))


def lane_columns(scenario, goal) -> list:
    """The tailored column list (a distinct schema) for one (scenario, goal): CORE + its extension + blobs."""
    return CORE + EXT.get((scenario, goal or None), []) + CORE_JSON


def lane_key(scenario, goal) -> str:
    """The per-lane file stem, e.g. llm-perf.pareto."""
    g = goal or "default"
    return f"{scenario}.{g}"


# rungs.csv columns → export columns that come FROM the per-rung table (renamed where the native name differs).
_RUNG_PASSTHROUGH = {
    "concurrency",
    *_LLM_METRICS,
}


def _read_jsonl(p: Path) -> list:
    if not p.is_file():
        return []
    return [json.loads(ln) for ln in p.read_text().splitlines() if ln.strip()]


def _read_csv(p: Path) -> list:
    if not p.is_file():
        return []
    return list(csv.DictReader(io.StringIO(p.read_text())))


def _read_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text()) or {}
    except ValueError:
        return {}


def _blank_row() -> dict:
    return {c: "" for c in SUPERSET}


def _sla_pass(row: dict) -> str:
    """Per-rung SLA verdict for max-concurrency-sla rows: pass iff ttft_p50 <= limit AND tpot_p50 <= limit.
    Empty when the limits or the measured latencies are absent (pareto rows, reconstructed-without-latency).
    """
    try:
        ttft, tpot = float(row["ttft_p50_ms"]), float(row["tpot_p50_ms"])
        lim_t, lim_p = float(row["sla_ttft_ms"]), float(row["sla_tpot_ms"])
    except (KeyError, TypeError, ValueError):
        return ""
    return "true" if (ttft <= lim_t and tpot <= lim_p) else "false"


def _cluster_details(run_meta: dict, cluster: str, ident: dict) -> str:
    """Canonical-JSON cluster fingerprint: run_meta.cluster_info (provider/region/GPU product/instance_type/
    gpus_per_node/k8s_version/node_count) when captured, always + the committed hardware facts.
    """
    out = {
        "cluster": cluster or None,
        "gpu_type": ident.get("gpu_type"),
        "gpu_count": ident.get("gpu_count"),
        "arch": ident.get("arch"),
    }
    out.update((run_meta or {}).get("cluster_info") or {})
    return json.dumps(out, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _launcher_command(run_meta: dict, launcher: dict, bench: dict) -> str:
    """The reproducible LOAD-GEN invocation as JSON: generator + version pin + args + sweep + dataset + the
    LITERAL argv (aiperf cli_command per rung / resolved run_manifest) captured at archive time.
    """
    gen = (launcher or {}).get("name") or ""
    args = {
        k[len(gen) + 1 :]: v
        for k, v in (run_meta or {}).items()
        if gen and k.startswith(gen + "_") and v not in (None, "")
    }
    cmd = {
        "generator": gen or None,
        "ref": (launcher or {}).get("ref"),
        "model_repo": (launcher or {}).get("model_repo"),
        "tokenizer": (launcher or {}).get("tokenizer_id"),
        "concurrencies": (bench or {}).get("sweep_concurrency"),
        "dataset": ((bench or {}).get("dataset") or {}).get("subpath"),
        "argv": (run_meta or {}).get("launcher_argv"),
        "task_source_sha256": (run_meta or {}).get("task_source_sha256"),  # unique id of the exact task bundle
        "args": args or None,
    }
    return json.dumps(
        {k: v for k, v in cmd.items() if v is not None},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )


# ── serving/model reproducibility fields (parallelism + quant + the full resolved server launch command) ──
_QUANT_TOKENS = ("nvfp4", "fp4", "fp8", "int8", "int4", "awq", "gptq", "bf16", "fp16")


def _recipe_serving(cell: Path) -> dict:
    """recipe.yaml's TOP-LEVEL `serving:` block (committed) — source for the model/serving fields that
    record.json's config.serving DROPS (served_model, model_repo, model_revision, max_model_len, gpu_mem_util).
    {} when no yaml lib or no recipe.yaml (mock stubs may omit it) — the export degrades to record-only.
    """
    rp = cell / "recipe.yaml"
    if yaml is None or not rp.is_file():
        return {}
    try:
        return (yaml.safe_load(rp.read_text()) or {}).get("serving") or {}
    except Exception:  # noqa: BLE001 — a malformed recipe never breaks the export
        return {}


def _arg_value(extra_args, *flags):
    """Value of a `--flag value` / `--flag=value` CLI arg among extra_args (each token is a whole authored
    `--flag value` string), or None. A bare boolean flag returns "true"."""
    for a in extra_args or []:
        toks = str(a).replace("=", " ").split()
        for i, t in enumerate(toks):
            if t in flags:
                return toks[i + 1] if i + 1 < len(toks) else "true"
    return None


def _arg_present(extra_args, *flags) -> bool:
    """True iff any bare boolean flag (e.g. --enable-expert-parallel) appears anywhere in extra_args."""
    toks = " ".join(str(a) for a in (extra_args or [])).replace("=", " ").split()
    return any(f in toks for f in flags)


def _quantization(model_repo, extra_args) -> str:
    """Weight quant/dtype: an explicit --quantization/--dtype wins; else sniff a known token (NVFP4/FP8/…)
    out of the model_repo name. "" when nothing is discoverable."""
    v = _arg_value(extra_args, "--quantization", "--dtype")
    if v:
        return v
    low = (model_repo or "").lower()
    for tok in _QUANT_TOKENS:
        if tok in low:
            return tok.upper()
    return ""


def _serving_fields(cell: Path, serving: dict, launcher: dict, launch_argv: list | None = None) -> dict:
    """The typed model/serving reproducibility columns for a cell — merges record.json config.serving
    (what ran: tp/dp/extra_args/disagg) with recipe.yaml serving (served_model/model_repo/revision/
    max_model_len) and PARSES extra_args for the parallelism + kv/quant knobs baked into the flags. All
    values stringify to "" when absent (never None), matching the blank-row contract.

    launch_argv = the RESOLVED server command tokens (from the committed rendered/ manifests). It is
    appended to the flag corpus so these columns reflect what the server ACTUALLY ran with, not only what
    the recipe declared in extra_args — a flag baked into the template (e.g. --kv-cache-dtype) is invisible
    to extra_args and used to render as a blank column."""
    rs = _recipe_serving(cell)
    _resolved = bool(launch_argv)  # did we actually read the server's resolved argv?
    extra = list(serving.get("extra_args") or []) + list(launch_argv or [])
    dis = serving.get("disagg") or {}
    for role in ("prefill", "decode"):  # disagg lanes carry per-role flags too
        extra += list((dis.get(role) or {}).get("extra_args") or [])
    model_repo = (launcher or {}).get("model_repo") or rs.get("model_repo") or ""
    dp = serving.get("dp")
    if dp in (None, ""):
        dp = _arg_value(extra, "--data-parallel-size", "--dp-size")
    pp = _arg_value(extra, "--pipeline-parallel-size", "--pp-size")
    ep = _arg_value(extra, "--expert-parallel-size", "--ep-size")
    if ep is None and _arg_present(extra, "--enable-expert-parallel"):
        ep = "true"
    mml = rs.get("max_model_len")
    if mml in (None, ""):
        mml = _arg_value(extra, "--max-model-len")
    return {
        "served_model_id": rs.get("served_model") or "",
        "model_repo": model_repo,
        "model_revision": rs.get("model_revision") or "",
        "ep": ep if ep is not None else "",
        "dp": dp if dp not in (None, "") else "",
        "pp": pp if pp is not None else "",
        # "" vs "engine-default" is a REAL distinction, not cosmetics. Blank used to mean both "nobody set
        # this" and "we never looked", which is exactly how a KVBM export could ship with no quant/kv info
        # and read as normal. Now: a value ⇒ explicitly set; "engine-default" ⇒ we parsed the resolved
        # server argv and it set nothing (so the engine default applies); "" ⇒ we could not resolve the
        # launch command at all, i.e. genuinely UNKNOWN.
        "quantization": _quantization(model_repo, extra) or (_resolved and "engine-default") or "",
        "kv_cache_dtype": (_arg_value(extra, "--kv-cache-dtype") or (_resolved and "engine-default") or ""),
        "max_model_len": mml if mml not in (None, "") else "",
    }


# A server launch line is EITHER an explicit `exec …` OR a bare invocation of a known model-server
# entrypoint. The `exec `-only rule silently missed every Dynamo/KVBM cell, whose server.yaml launches
# `python3 -m dynamo.vllm \` with no exec prefix — so launch_command came out "" on the one recipe whose
# purpose is the KV connector flags. Keep this list additive: an unknown launcher must surface as
# UNRESOLVED (see _launch_command), never as a blank column.
_SERVER_ENTRYPOINTS = (
    "-m dynamo.",  # dynamo.vllm / dynamo.frontend / dynamo.sglang
    "-m vllm",  # python3 -m vllm.entrypoints…
    "vllm serve",
    "-m sglang",  # python3 -m sglang.launch_server
    "sglang.launch_server",
    "trtllm-serve",
    "-m tensorrt_llm",
)


def _is_launch_line(s: str) -> bool:
    """True for a resolved server-launch line: an explicit `exec …`, or a bare known-entrypoint invocation.
    Comments never count — a rendered manifest documents the real command in prose right above it.
    """
    s = s.strip()
    if s.startswith("#"):
        return False
    if s.startswith("exec "):
        return True
    return any(tok in s for tok in _SERVER_ENTRYPOINTS)


def _extract_execs(text: str) -> list:
    """Every resolved server-launch command in a rendered manifest, each collapsed from its backslash-
    continued multi-line form into ONE normalized string. yaml-free (pure text) so it works without the
    optional yaml lib and regardless of indentation. A trailing `&` (backgrounded sidecar process, e.g.
    dynamo.frontend) is stripped — it is shell plumbing, not part of the argv."""
    lines = text.splitlines()
    out: list = []
    i = 0
    while i < len(lines):
        if _is_launch_line(lines[i]):
            collected = [lines[i].strip()]
            while collected[-1].endswith("\\") and i + 1 < len(lines):
                i += 1
                collected.append(lines[i].strip())
            joined = " ".join(c.rstrip("\\").strip() for c in collected).strip()
            if joined.startswith("exec "):
                joined = joined[len("exec ") :].strip()
            out.append(joined.rstrip("&").strip())
        i += 1
    return out


def _launch_command(cell: Path) -> str:
    """The COMPLETE, fully-RESOLVED server launch command(s) as JSON — extracted from the cell's committed
    rendered/ manifests (Jinja already baked; only ${cluster} vars remain literal). This is the true
    container argv the model server ran with (vLLM `python3 -m vllm…`, sglang/dynamo `python3 -m dynamo…`),
    NOT a template and NOT the load-gen (that's launcher_command). Every server manifest's launch line is
    captured; disagg lanes yield one per role (prefill/decode/frontend).

    ABSENCE IS NEVER SILENT. Three distinct outcomes, three distinct values:
      ""                          — no rendered/ committed: nothing was ever claimed, honestly empty.
      {"_unresolved": {...}}      — rendered/ EXISTS but no launch line was recognised. A blank here used
                                    to be indistinguishable from "no manifests", which is how the KVBM
                                    cells shipped exports with no launch command and nobody noticed.
      {"<file>.yaml": [argv, …]}  — resolved.
    """
    rd = cell / "rendered"
    if not rd.is_dir():
        return ""
    commands: dict = {}
    scanned: list = []
    for f in sorted(rd.glob("*.yaml")):
        if "bench-job" in f.name:  # the LOAD-GEN manifest — captured separately as launcher_command
            continue
        scanned.append(f.name)
        try:
            cmds = _extract_execs(f.read_text())
        except OSError:
            continue
        if cmds:
            commands[f.name] = cmds
    if not commands:
        return json.dumps(
            {
                "_unresolved": {
                    "reason": "rendered/ present but no recognised server launch line "
                    "(neither `exec …` nor a known entrypoint)",
                    "scanned": scanned,
                    "known_entrypoints": list(_SERVER_ENTRYPOINTS),
                }
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
    return json.dumps(commands, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _launch_argv_tokens(launch_command: str) -> list:
    """The resolved server argv flattened to a token list, for deriving the typed serving columns
    (quantization / kv_cache_dtype / …) from what ACTUALLY ran rather than only from recipe extra_args.
    Returns [] for "" and for the _unresolved marker — an unknown launcher must not fabricate knowledge.
    """
    if not launch_command:
        return []
    try:
        obj = json.loads(launch_command)
    except (ValueError, TypeError):
        return []
    if not isinstance(obj, dict) or "_unresolved" in obj:
        return []
    toks: list = []
    for cmds in obj.values():
        for c in cmds if isinstance(cmds, list) else [cmds]:
            toks.append(str(c))
    return toks


def _group_block(cell: Path) -> dict | None:
    """The optional `envelope.group` block a cell declares to join a group, or None (⇒ singleton)."""
    rp = cell / "recipe.yaml"
    if yaml is None or not rp.is_file():
        return None
    env = (yaml.safe_load(rp.read_text()) or {}).get("envelope") or {}
    g = env.get("group")
    return g if isinstance(g, dict) and g.get("id") else None


def group_map(cells: list) -> dict:
    """cell → (group_id, group_size). Members sharing an `envelope.group.id` collapse to ONE real
    group_id = sha256({scenario, goal, kind, sorted(varies), sorted(member (benchmark_id, cluster))})[:16]
    (kind still salts the id so distinct group TYPES over the same members don't collide, even though it is no
    longer an emitted column). A cell with no group block is a SINGLETON: group_id = benchmark_id, group_size 1.
    """
    groups: dict = {}
    for cell in cells:
        gb = _group_block(cell)
        if not gb:
            continue
        rec = json.loads((cell / "record.json").read_text()) if (cell / "record.json").is_file() else {}
        bid = (rec.get("fingerprint") or {}).get("benchmark_id") or ""
        ident = rec.get("identity") or {}
        clusters = sorted({(r.get("cluster") or "") for r in _read_jsonl(cell / "runs" / "index.jsonl")})
        g = groups.setdefault(
            gb["id"],
            {
                "kind": gb.get("kind") or "",
                "varies": gb.get("varies") or [],
                "scenario": ident.get("scenario"),
                "goal": ident.get("goal"),
                "members": [],
            },
        )
        g["members"].append((cell, bid, clusters or [""]))
    out: dict = {}
    for g in groups.values():
        member_tuples = sorted([bid, c] for (_, bid, clusters) in g["members"] for c in clusters)
        canon = json.dumps(
            {
                "scenario": g["scenario"],
                "goal": g["goal"],
                "kind": g["kind"],
                "varies": sorted(g["varies"]),
                "members": member_tuples,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        gid = hashlib.sha256(canon.encode("utf-8")).hexdigest()[:16]
        for cell, _bid, _cl in g["members"]:
            out[str(cell)] = (gid, len(g["members"]))
    return out


def _cell_rows(cell: Path, groups: dict | None = None, run_by: str | None = None) -> list:
    """Every (run × rung) row for one cell (full SUPERSET dicts; projected onto the lane schema at write).
    run_by (WHO ran it) is resolved ONCE by the caller and threaded in; None ⇒ resolve here (stored → git
    email → $USER → "unknown"), so a direct call still emits a runner and never blocks on a missing one.
    """
    if run_by is None:
        run_by = _ri.resolve_runner()
    index = _read_jsonl(cell / "runs" / "index.jsonl")
    if not index:
        return []
    rec = json.loads((cell / "record.json").read_text()) if (cell / "record.json").is_file() else {}
    ident = rec.get("identity") or {}
    fp = rec.get("fingerprint") or {}
    prov = rec.get("provenance") or {}
    res = rec.get("result") or {}
    serving = (rec.get("config") or {}).get("serving") or {}
    bench = (rec.get("config") or {}).get("bench") or {}
    launcher = prov.get("launcher") or {}
    sla = res.get("sla") or {}
    crossing = res.get("crossing") or {}
    benchmark_id = fp.get("benchmark_id") or ""
    gid, group_size = (groups or {}).get(str(cell), (benchmark_id, 1))
    # Resolve the server launch ONCE: it is both its own column and the flag corpus the typed serving
    # columns are derived from (quantization / kv_cache_dtype / parallelism baked into the template).
    _launch_cmd_json = _launch_command(cell)
    _launch_argv = _launch_argv_tokens(_launch_cmd_json)
    details = {
        "serving": serving,
        "bench": bench,
        "exemplar": {
            "reference": res.get("reference"),
            "tolerance_pct": res.get("tolerance_pct"),
        },
        "run": {
            "wall_seconds": prov.get("wall_seconds"),
            "gpu_hours": prov.get("gpu_hours"),
            "launcher": launcher,
        },
        "crossing": crossing,
        "experiment_id": fp.get("experiment_id"),
        "git_ref": fp.get("git_ref"),
    }
    base = _blank_row()
    base.update(
        {
            "export_schema_version": EXPORT_SCHEMA_VERSION,
            "group_id": gid,
            "group_size": group_size,
            "benchmark_id": benchmark_id,
            "recipe_hash": fp.get("recipe_hash") or "",
            "scenario": ident.get("scenario") or "",
            "goal": ident.get("goal") or "",
            "model": ident.get("model") or "",
            "gpu_type": ident.get("gpu_type") or "",
            "gpu_count": (ident.get("gpu_count") if ident.get("gpu_count") is not None else ""),
            "arch": ident.get("arch") or "",
            "engine": ident.get("engine") or "",
            "serving_mode": ident.get("serving_mode") or "",
            "tp": serving.get("tp") if serving.get("tp") is not None else "",
            # WHO ran it (person-level; NEVER blocks the export — falls back to "unknown").
            "run_by": run_by,
            # full resolved server launch argv (from committed rendered/*.yaml) — the ultimate reproducibility artifact.
            "launch_command": _launch_cmd_json,
            "distribution": ident.get("distribution") or "",
            "mode": ident.get("mode") or "",
            # job_mode: the exported workload class — these are all serving/INFERENCE cells (vs training). Constant
            # 'inference' for the current collection (owner-agreed); a training lane would emit its own value.
            "job_mode": "inference",
            "load_generator": launcher.get("name") or "",
            "image_digest": prov.get("image_digest") or fp.get("image_digest") or "",
            "cell_metric": res.get("metric") or "",
            "cell_unit": res.get("unit") or "",
            "cell_value": res.get("value") if res.get("value") is not None else "",
            "sla_ttft_ms": sla.get("ttft_ms") if sla.get("ttft_ms") is not None else "",
            "sla_tpot_ms": sla.get("tpot_ms") if sla.get("tpot_ms") is not None else "",
            "benchmark_valid": (rec.get("benchmark_valid", "") if rec.get("benchmark_valid") is not None else ""),
            "details": json.dumps(details, sort_keys=True, separators=(",", ":"), ensure_ascii=False),
        }
    )
    # typed model/serving reproducibility columns (served_model_id/model_repo/revision, ep/dp/pp,
    # quantization/kv_cache_dtype/max_model_len) — merged from record.json serving + recipe.yaml + extra_args.
    base.update(_serving_fields(cell, serving, launcher, _launch_argv))
    rows = []
    for run in index:
        curated = cell / (run.get("curated") or f"runs/{run.get('run_id')}/curated")
        rungs = _read_csv(curated / "rungs.csv")
        run_meta = _read_json(curated / "run_meta.json")
        cpu = (run_meta or {}).get("cpu_info") or {}
        ws = run.get("wall_seconds")
        gc = run.get("gpu_count") if run.get("gpu_count") is not None else ident.get("gpu_count")
        gpu_hours = round(gc * ws / 3600, 3) if isinstance(ws, (int, float)) and isinstance(gc, (int, float)) else ""
        run_base = dict(base)
        run_base.update(
            {
                "run_id": run.get("run_id") or "",
                "cluster": run.get("cluster") or "",
                "date_utc": run.get("date") or "",
                "git_commit": run.get("git_commit") or "",
                "data_provenance": run.get("data_provenance") or "",
                # THIS run's own headline metric (from its index entry) — distinct from cell_value (the cell's
                # published/band number, denormalized). Lets the DB compute headline variance = stddev(run_value)
                # over run_id, GROUP BY (benchmark_id, concurrency).
                "run_value": run.get("value") if run.get("value") is not None else "",
                "wall_seconds": ws if ws is not None else "",
                "gpu_hours": gpu_hours,
                "cpu_model": cpu.get("cpu_model") or "",
                "cpu_cores_logical": cpu.get("cpu_cores_logical") or "",
                "memory_total_gb": cpu.get("memory_total_gb") or "",
                "cluster_details": _cluster_details(run_meta, run.get("cluster") or "", ident),
                "launcher_command": _launcher_command(run_meta, launcher, bench),
            }
        )
        if not rungs:
            rows.append(run_base)
            continue
        measured = run.get("data_provenance") in ("archived", "rerun", "live")
        for rung in rungs:
            row = dict(run_base)
            for k, v in rung.items():
                col = k
                if col in _RUNG_PASSTHROUGH:
                    row[col] = v if v is not None else ""
            # aiperf omits request_error_rate when zero, so a blank on a REAL measured rung (has latency)
            # means zero errors observed → emit 0, not null. Null stays reserved for "never captured"
            # (reconstructed cells with no raw), so a blank truly signals missing, not clean.
            if measured and row.get("error_rate_pct") in ("", None) and row.get("ttft_p50_ms") not in ("", None):
                row["error_rate_pct"] = "0"
            row["sla_pass"] = _sla_pass(row)
            rows.append(row)
    return rows


def _sort_key(r: dict):
    def num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return float("inf")

    return (r["group_id"], r["run_id"], num(r["concurrency"]))


def _render_csv(rows: list, columns: list) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
    w.writeheader()
    for r in sorted(rows, key=_sort_key):
        w.writerow(r)
    return buf.getvalue()


def _write_csv(rows: list, out: Path, columns: list) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(_render_csv(rows, columns))


def on_pipeline_cells(root: Path) -> list:
    return sorted({p.parent.parent for p in root.glob("recipes/**/runs/index.jsonl")})


def _lane_of(rows: list) -> tuple:
    """(scenario, goal-or-None) for a cell's rows — a cell is exactly one lane."""
    return (rows[0]["scenario"], rows[0]["goal"] or None)


def _rows_by_run(rows: list) -> dict:
    """Group a cell's rows by run_id (order-preserving) — one per-run llmb_inference_export.csv per key."""
    by_run: dict = {}
    for r in rows:
        by_run.setdefault(r["run_id"], []).append(r)
    return by_run


def build(root: Path) -> tuple[int, int]:
    """Write per-RUN `results/<run_id>/llmb_inference_export.csv` (that run's rows, its cell's lane schema) — the DB upload
    unit, co-located with the run's other (gitignored) artifacts. EPHEMERAL: publish reads it → DB stores it;
    the repo keeps only the cell's rolled-up record. No repo-side aggregate; the DB owns the union across runs.
    """
    cells = on_pipeline_cells(root)
    groups = group_map(cells)
    run_by = _ri.resolve_runner()  # resolve WHO ran it ONCE per export (avoids per-cell git calls)
    n_cells = n_runs = 0
    for cell in cells:
        rows = _cell_rows(cell, groups, run_by)
        if not rows:
            continue
        n_cells += 1
        cols = lane_columns(*_lane_of(rows))
        for run_id, run_rows in _rows_by_run(rows).items():
            _write_csv(run_rows, root / _rd.results_dir(cell, run_id) / EXPORT_FILENAME, cols)
            n_runs += 1
    return n_cells, n_runs


def check(root: Path) -> int:
    """Confirm every run renders a valid `results/<run_id>/llmb_inference_export.csv` (EXPORT_FILENAME) — the
    export is a gitignored, on-demand DB-upload artifact, so there is no committed copy to byte-diff (its
    determinism follows record.json's). Also flags leftovers from the retired committed-export model: a per-run
    export.csv/llmb_inference_export.csv under recipes/, per-cell upload.csv, the aggregate export/*.csv.
    Read-only (renders in memory)."""
    problems: list = []
    cells = on_pipeline_cells(root)
    groups = group_map(cells)
    run_by = _ri.resolve_runner()
    n = 0
    for cell in cells:
        rows = _cell_rows(cell, groups, run_by)
        if not rows:
            continue
        cols = lane_columns(*_lane_of(rows))
        for run_id, run_rows in _rows_by_run(rows).items():
            try:
                _render_csv(run_rows, cols)  # must render without error
                n += 1
            except Exception as exc:  # noqa: BLE001 — report any render failure
                problems.append(f"{_rd.results_dir(cell, run_id)}/{EXPORT_FILENAME}: render failed: {exc}")
    for f in sorted(
        list(root.glob("recipes/**/runs/*/export.csv")) + list(root.glob(f"recipes/**/runs/*/{EXPORT_FILENAME}"))
    ):
        problems.append(
            f"{f.relative_to(root)}: STALE — the per-run export is a gitignored results/<run_id>/ "
            f"artifact, not committed under recipes/ (delete it)"
        )
    for f in sorted(root.glob("recipes/**/upload.csv")):
        problems.append(f"{f.relative_to(root)}: STALE — retired per-cell upload.csv (delete it)")
    exp = root / "export"
    for f in sorted(exp.glob("*.csv")) if exp.is_dir() else []:
        problems.append(f"export/{f.name}: STALE — retired aggregate export (delete it)")
    for p in problems:
        print(f"DRIFT  {p}")
    if problems:
        print(
            f"export-check: {len(problems)} issue(s) — 'make export' regenerates into results/; delete any stale committed export"
        )
        return 1
    print(f"export-check: {n} run(s) render a valid results/<run_id>/{EXPORT_FILENAME}; no stale committed artifacts")
    return 0


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    root = Path(args[0]).resolve() if args else ROOT
    if "--check" in argv:
        return check(root)
    n_cells, n_runs = build(root)
    print(
        f"[export_dataset] {n_cells} cell(s) → {n_runs} per-run {EXPORT_FILENAME} "
        f"(results/<scenario>/<goal>/<name>/<run_id>/{EXPORT_FILENAME})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
