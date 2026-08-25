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

"""check_invariants.py — enforce the cross-cutting invariants the schema + contract-check can't express.

The schema validates field shapes and contract-check validates required files. This module checks invariants
that span fields/files and would otherwise be honor-system:
  - content-pinning (inv 7/10): a cell that has RUN must carry a REAL image digest + dataset hash (not a
    placeholder). planned/wip may hold placeholders.
  - cross-field consistency: requires.gpu.count matches the tensor-parallel size (agg: serving.tp; disagg:
    prefill.tp + decode.tp).
  - exemplar metric validity: envelope.exemplar.metric is one the scenario's analysis actually computes.
  - cluster-specifics live ONLY in profiles (inv 2): rendered manifests parameterize the namespace with
    ${NAMESPACE} rather than a hardcoded value, and RDMA fabric identity (UCX transports / HCA rail names /
    addr type / rail count) comes from ${RDMA_UCX_*} — no recipe `serving.disagg.ucx`, no baked mlx5_*.
Run per cell; fails the build on any hard violation. WARNs are advisory (printed, don't fail).
"""

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("check_invariants: requires pyyaml")

ROOT = Path(__file__).resolve().parent.parent
RAN = {"runs", "performant", "exemplar"}  # a run happened -> pins must be real
KNOWN_METRICS = {
    "llm-perf": {"max_concurrency_at_sla", "tps_per_gpu_at_sla", "pareto_geomean"},
}
PLACEHOLDER_HASH = re.compile(r"^(sha256:)?0{64}$|pending|placeholder|TODO", re.I)
PLACEHOLDER_IMAGE = re.compile(r"PENDING|placeholder|TODO|<your", re.I)

# vLLM's engine default for --max-num-seqs when it is left unset (v0 + v1 schedulers). Encoded explicitly so
# the sweep-≤-cap gate below has a concrete ceiling even when a vLLM recipe never passes --max-num-seqs; a
# sweep rung above the served cap queues INTERNALLY and its "throughput" is a scheduler artifact, not GPU
# capacity → an INVALID pareto/max-sla point.
VLLM_DEFAULT_MAX_NUM_SEQS = 256


def _engine_family(serving: dict) -> str:
    """PURE. vLLM vs SGLang from serving.stack. Only vLLM has the well-known 256 default; SGLang's cap is
    --max-running-requests and memory-derived, so we DON'T assume a default there (→ 'unknown').
    """
    stack = str(serving.get("stack", "")).lower()
    if "vllm" in stack:
        return "vllm"
    if "sglang" in stack:
        return "sglang"
    return "unknown"


def resolve_max_num_seqs(serving: dict):
    """PURE. The served concurrency cap → (cap, source). Precedence:
      1. an explicit field: serving.max_num_seqs or serving.vllm_max_num_seqs
      2. --max-num-seqs N parsed from serving.extra_args (or a disagg role's extra_args)
      3. the vLLM engine default (256) — ONLY when the engine is vLLM
    Returns (None, 'unknown') when no cap is determinable (e.g. SGLang without --max-num-seqs: its
    --max-running-requests ceiling is memory-derived and not a static value we assert here).
    """
    for fld in ("max_num_seqs", "vllm_max_num_seqs"):
        v = serving.get(fld)
        if isinstance(v, int) and v > 0:
            return v, f"serving.{fld}"
    args = list(serving.get("extra_args") or [])
    dis = serving.get("disagg") or {}
    for role in ("prefill", "decode"):
        args += list((dis.get(role) or {}).get("extra_args") or [])
    mns = None
    for a in args:
        m = re.search(r"--max-num-seqs[= ]+(\d+)", str(a))
        if m:
            mns = int(m.group(1))
    if mns is not None:
        return mns, "--max-num-seqs"
    if _engine_family(serving) == "vllm":
        return (
            VLLM_DEFAULT_MAX_NUM_SEQS,
            f"vLLM engine default {VLLM_DEFAULT_MAX_NUM_SEQS}",
        )
    return None, "unknown"


def check_cell(cell: Path):
    problems, warns = [], []
    r = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    env, serving, bench = (
        r.get("envelope") or {},
        r.get("serving") or {},
        r.get("bench") or {},
    )
    replay = r.get("replay") or {}
    status = env.get("status", "planned")
    prov = env.get("provenance") or {}
    ran = status in RAN

    # 1) content-pinning: real image digest + dataset hash once it has RUN
    digest = prov.get("image_digest", "")
    if not digest or PLACEHOLDER_HASH.search(digest):
        (problems if ran else warns).append(
            f"image_digest is missing/placeholder ({digest or '—'}) — must be a real sha256 before `runs`"
        )
    # mode=synthetic has no trace file (aiperf generates ISL/OSL) → nothing to content-pin here.
    if env.get("scenario") == "llm-perf" and env.get("mode") != "synthetic":
        dsha = ((bench.get("dataset") or {}).get("sha256")) or ""
        if not dsha or PLACEHOLDER_HASH.search(dsha):
            (problems if ran else warns).append(
                f"bench.dataset.sha256 is placeholder ({dsha or '—'}) — pin the canonical trace before `runs`"
            )

    # 2) requires.gpu.count matches the TP size
    want = ((env.get("requires") or {}).get("gpu") or {}).get("count")
    if want is not None:
        disagg = serving.get("disagg")
        if disagg:
            # Each role runs `replicas` (default 1) independent TP-sized workers, so its GPU draw is
            # replicas × tp. Total = prefill + decode (1P1D reduces to prefill.tp + decode.tp).
            def _role_gpus(r):
                cfg = disagg.get(r) or {}
                return cfg.get("replicas", 1) * cfg.get("tp", 0)

            tp = _role_gpus("prefill") + _role_gpus("decode")
            if want != tp:
                problems.append(
                    f"requires.gpu.count={want} != (prefill.replicas×tp)+(decode.replicas×tp)={tp} (disagg needs every worker's GPUs)"
                )
        elif serving.get("tp") is not None and want != serving["tp"]:
            problems.append(
                f"requires.gpu.count={want} != serving.tp={serving['tp']} (a TP-{serving['tp']} server needs {serving['tp']} GPUs)"
            )

    # 2b) the served concurrency cap must cover the WHOLE sweep, or the top rungs measure QUEUE DEPTH, not
    #     GPU capacity: requests above the served max-num-seqs queue INTERNALLY and their "throughput" is a
    #     scheduler artifact → an INVALID pareto/max-sla point. Cap resolves from an explicit field, else
    #     --max-num-seqs, else the vLLM engine default (256). Applies wherever the recipe SWEEPS concurrency.
    if env.get("scenario") == "llm-perf":
        sweep_mode = bench.get("sweep_mode", "fixed")
        if sweep_mode == "adaptive":
            adaptive = bench.get("adaptive_sweep") or {}
            smax = adaptive.get("max")
            if smax is None:
                warns.append(
                    "sweep_mode=adaptive but adaptive_sweep.max is not set — cannot verify the served max-num-seqs covers the ceiling"
                )
        else:
            sweep = bench.get("sweep_concurrency") or []
            smax = max(sweep) if sweep else None
        if smax is not None:
            cap, src = resolve_max_num_seqs(serving)
            if cap is None:
                warns.append(
                    f"cannot determine the served concurrency cap (engine={_engine_family(serving)}) but the sweep reaches {smax} — confirm the cap ≥ {smax}, else the top rungs just queue"
                )
            elif smax > cap:
                problems.append(
                    f"sweep reaches {smax} but the served max-num-seqs is {cap} ({src}) — concurrency above {cap} only queues internally (a scheduler artifact, not GPU capacity); cap the sweep at {cap} or raise the server cap"
                )

    # 2c) serving reproducibility + render-safety
    if env.get("scenario") == "llm-perf":
        # model_revision pins the exact weights; without it the server fetches HF 'latest' → non-reproducible.
        # WARN not FAIL: several existing cells predate this and their actual revision is unknown (can't assert
        # one retroactively); new cells + any re-run should pin it. image_digest + dataset.sha256 stay hard-required.
        if not serving.get("model_revision"):
            warns.append(
                "serving.model_revision is unset — server fetches HF 'latest' (non-reproducible); pin the exact revision"
            )
        # disaggregated serving MUST carry serving.disagg or the workers manifest renders with empty prefill/decode.
        if env.get("serving_mode") == "disaggregated" and not serving.get("disagg"):
            problems.append(
                "serving_mode=disaggregated requires serving.disagg (prefill+decode) — else the workers manifest renders empty"
            )

    # 3) exemplar metric is one the scenario's analysis computes
    metric = (env.get("exemplar") or {}).get("metric")
    valid = KNOWN_METRICS.get(env.get("scenario"), set())
    if metric and metric not in valid:
        problems.append(f"exemplar.metric='{metric}' is not computed by {env.get('scenario')} analysis {sorted(valid)}")

    # 3a) exemplar publish gate: an exemplar is the committed cross-cluster bar pushed downstream, so it must
    #     carry (i) a declared goal — the bar IS that goal's measurement (llm-perf); and (ii) record.json — the
    #     canonical machine-readable record (identity+fingerprint+provenance+metric+detail) that goes to the DB.
    if status == "exemplar":
        if env.get("scenario") == "llm-perf" and not env.get("goal"):
            problems.append("status=exemplar requires envelope.goal — the committed exemplar bar is that goal's metric")
        if not (cell / "record.json").is_file():
            problems.append(
                "status=exemplar requires record.json (the canonical DB record) — run scripts/publish.py to emit it"
            )

    # 3b) goal contract: the methodology bundle's required fields must be present + consistent. The metric a
    #     goal implies is SCENARIO-DEPENDENT (a goal's axes differ by scenario), so branch on (goal, scenario).
    goal = env.get("goal")
    scenario = env.get("scenario")
    if goal == "max-concurrency-sla":
        if not (bench.get("dataset") or bench.get("synthetic")):
            problems.append(
                "goal=max-concurrency-sla requires bench.dataset (a trace) or bench.synthetic (mode=synthetic)"
            )
        if metric != "max_concurrency_at_sla":
            problems.append(
                f"goal=max-concurrency-sla requires exemplar.metric=max_concurrency_at_sla (got {metric!r})"
            )
    elif goal == "pareto":
        # the sweep can live in bench.sweep_concurrency (llm-perf) OR replay.rungs (replay).
        _pareto_sweep = bench.get("sweep_concurrency") or replay.get("rungs")
        if not _pareto_sweep:
            problems.append("goal=pareto requires a full concurrency sweep (bench.sweep_concurrency, or replay.rungs)")
        # Single-point guard: pareto_geomean is VALID for one rung (geomean of one value), so this is no longer
        # a degeneracy — just not a real frontier. WARN (do NOT fail): an intentionally single-point pareto
        # (e.g. glm5-16k512 c240) is valid but should be evaluated with the other points in its recipe family.
        elif len(_pareto_sweep) < 2:
            warns.append(
                f"goal=pareto '{env.get('name')}' is a single operating point ({_pareto_sweep}), not a "
                "real frontier → pareto_geomean is valid (geomean of one rung); evaluate it with the "
                "other operating points in the same recipe family."
            )
        # pareto MUST be a FIXED sweep: it needs the whole frontier. adaptive stops once it brackets the SLA
        # crossing — that's the max-concurrency-sla strategy, and it can't trace a curve. (Per-goal default:
        # max-concurrency-sla → adaptive search; pareto → fixed full sweep.)
        if bench.get("sweep_mode") == "adaptive":
            problems.append(
                "goal=pareto requires sweep_mode=fixed — adaptive stops at the crossing and can't trace the frontier"
            )
        want = "pareto_geomean"
        if metric != want:
            problems.append(f"goal=pareto on {scenario} requires exemplar.metric={want} (got {metric!r})")
        if scenario == "llm-perf" and not (bench.get("dataset") or bench.get("synthetic")):
            problems.append(
                "goal=pareto (llm-perf) requires bench.dataset (a trace) or bench.synthetic (mode=synthetic)"
            )

    # 3c) distribution must NAME the actual dataset file (or 'synthetic') — this is what would have caught the
    #     256k-label-on-a-1M-file drift: the label can't diverge from the workload it actually replays.
    if env.get("scenario") == "llm-perf":
        ds_id = (bench.get("dataset") or {}).get("id")
        dist = env.get("distribution")
        if ds_id and dist != ds_id:
            problems.append(
                f"distribution='{dist}' must equal bench.dataset.id='{ds_id}' (distribution names the actual dataset file)"
            )
        elif not ds_id and dist != "synthetic":
            problems.append(
                f"distribution='{dist}' but there is no bench.dataset — a file-less recipe must set distribution: synthetic"
            )

    # 3e) Release policy: reproducibility tolerance = 5 % and SLA stat = p50.
    #     Pinned repo-wide so cross-cluster comparisons stay apples-to-apples — change the
    #     enforced constants here, not a cell.
    tol_pct = (env.get("exemplar") or {}).get("tolerance_pct")
    if tol_pct not in (None, 5):
        problems.append(f"exemplar.tolerance_pct={tol_pct} — release policy requires 5% tolerance")
    if env.get("scenario") == "llm-perf":
        stat = (bench.get("sla") or {}).get("stop_stat")
        if stat not in (None, "p50"):
            problems.append(f"bench.sla.stop_stat={stat!r} — release policy requires p50")

    # 4) cluster-specifics only in profiles: rendered manifests parameterize the namespace
    rdir = cell / "rendered"
    if rdir.is_dir():
        for f in rdir.glob("*.yaml"):
            txt = f.read_text()
            if "namespace:" in txt and "${NAMESPACE}" not in txt:
                problems.append(f"rendered/{f.name} hardcodes a namespace (must be ${{NAMESPACE}} from the profile)")
                break

    # 5) cluster-specifics only in profiles, part 2: RDMA FABRIC IDENTITY.
    # 27 disagg cells shipped serving.disagg.ucx.{tls,net_devices} holding one cluster's 8-rail mlx5_*
    # list and an InfiniBand-only transport set. Because the recipe value WON over the profile, those
    # cells could not be corrected by any profile and were dead on a RoCE cluster (GLM-5 GB300:
    # NIXL_ERR_BACKEND / "no active messages transport", scheduler exit -3). The schema rejects the
    # recipe key; this is the belt-and-braces half — the RENDERED bytes must stay device-name-free, which
    # also catches a template that reintroduces a literal without any recipe key being involved.
    if (serving.get("disagg") or {}).get("ucx") is not None:
        problems.append(
            "serving.disagg.ucx is NOT supported — UCX fabric config (tls / net_devices / ib_addr_type / "
            "max_rndv_rails) is CLUSTER truth, not recipe truth, and a recipe value overrides the profile "
            "so nothing can correct it. Remove the block and set RDMA_UCX_TLS / RDMA_UCX_NET_DEVICES / "
            "RDMA_UCX_IB_ADDR_TYPE / RDMA_UCX_MAX_RNDV_RAILS in cluster-profiles/<cluster>.env instead "
            "(`llmb-k8s profile probe-fabric --cluster <name> --write` discovers most of them)."
        )
    if rdir.is_dir():
        for f in sorted(rdir.glob("*.yaml")):
            for ln in f.read_text().splitlines():
                if not re.search(r"\bname:\s*UCX_(TLS|NET_DEVICES|IB_ADDR_TYPE|MAX_RNDV_RAILS)\b", ln):
                    continue
                if not re.search(r"\$\{RDMA_UCX_(TLS|NET_DEVICES|IB_ADDR_TYPE|MAX_RNDV_RAILS)\}", ln):
                    problems.append(
                        f"rendered/{f.name} bakes a literal UCX fabric value ({ln.strip()}) — it must be "
                        "the ${RDMA_UCX_*} token so the cluster profile supplies it"
                    )
    return problems, warns


def main() -> int:
    cells = sorted({p.parent for p in (ROOT / "recipes").glob("**/recipe.yaml")})
    fails = 0
    for c in cells:
        rel = c.relative_to(ROOT)
        problems, warns = check_cell(c)
        for w in warns:
            print(f"WARN {rel}: {w}")
        if problems:
            fails += 1
            for p in problems:
                print(f"FAIL {rel}: {p}")
        else:
            print(f"OK   {rel}")

    # ${VAR} template-coverage guard (pairs with the runtime var-reconciliation): every ${VAR} any committed
    # cell's rendered manifests reference must be documented in _template.env.example (a profile var) OR a
    # known runtime placeholder. Otherwise a cell can ship needing a new ${VAR} that profile validate/init
    # + preflight don't know to verify — exactly how NO_INTERNET_DNS_IP slipped in. This makes onboarding's
    # what-to-check auto-extend with the recipes.
    try:
        import manifest_vars as _mv

        cov = _mv.template_coverage_gaps(cells)
        if cov:
            fails += 1
            for var, names in cov:
                print(
                    f"FAIL template-coverage: ${{{var}}} referenced by {', '.join(names)} is not documented "
                    "in cluster-profiles/_template.env.example nor a known runtime placeholder "
                    "(add it so init/preflight verify it)"
                )
        else:
            print(
                f"OK   template-coverage: every referenced ${{VAR}} is a documented profile var or runtime placeholder"
            )
    except Exception as _e:
        print(f"WARN template-coverage: skipped ({_e})")

    print(f"check-invariants: {len(cells) - fails}/{len(cells)} cells satisfy the cross-cutting invariants")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
