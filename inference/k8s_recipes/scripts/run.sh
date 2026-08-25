#!/usr/bin/env bash
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

# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# run.sh <cell-dir> <cluster-profile> [run-id] [flags]
#
# End-to-end lifecycle orchestrator. Drives the full sequence idempotently:
#   1. preflight   — dryrun.sh: catches misconfig before touching the cluster
#   2. namespace   — verify exists; --managed-ns creates it (labels it llmb-owned)
#   3. stage       — stage-dataset.sh stages the benchmark dataset
#   4. server      — deploy.sh kubectl apply (idempotent; --skip-server reuses running instance)
#   5. wait-ready  — poll Deployment/server pods until ready (wait_server_ready.sh); FAILS FAST on a crash-loop
#                    (restartCount>=K or CrashLoopBackOff>=M min) so a never-ready server frees its GPU in minutes
#                    instead of blocking the full serving.startup_timeout_s (up to 4h for 550B cells)
#   6. benchmark   — sweep.sh runs the configured concurrency rungs
#
# The supported llm-perf lane is routed through scripts/lane.py, shared with llmb-k8s.
#   7. fetch       — pull artifacts off the RWO PVC into results/<run-id>/
#   8. teardown    — kubectl scale ... --replicas=0; --teardown only (frees GPU; instant resume)
#   → prints publish command + elapsed wall time
#
# Two modes:
#   Default:       namespace must already exist (admin pre-provisioned it).
#   --managed-ns:  create namespace if missing.
#
# ATTACHED (default) vs DETACHED (--detach):
#   Default is ATTACHED — the full lifecycle above, babysat on this laptop (setup → wait-ready → benchmark →
#   fetch → publish-hint → teardown). Fine for SHORT runs; a multi-hour run must NOT stay attached, because a
#   laptop/session boundary reaps the orchestrator (the in-cluster run survives, but the auto fetch/publish is
#   lost). --attach / --wait are explicit synonyms for this default.
#   --detach does ONLY the ~2-minute setup — preflight → run-owner → stage → deploy → APPLY the bench Job —
#   then writes a durable run-id handle and EXITS 0. It does NOT wait-ready, follow, fetch, publish, or teardown.
#   The run is self-sufficient in-cluster: the in-cluster bench Job waits for the server itself, the run-owner
#   frees the GPU on the Job's terminal state, and results persist server-side (the artifacts PVC + the
#   netscore-<id> ConfigMap both survive the GC cascade — neither carries the run-owner ownerReference).
#   Harvest later, from anywhere:  llmb-k8s collect <run-id> [--cluster <profile>].
#
# Re-run friendly: every step is idempotent. Stage skips if sha256 matches; server apply is a no-op
# if unchanged; each sweep gets a fresh RUN_ID so jobs never collide.
#
# Flags:
#   --detach         setup-then-EXIT (no wait/fetch/publish/teardown); harvest later via `llmb-k8s collect`
#   --attach|--wait  explicit form of the default attached lifecycle
#   --managed-ns     create namespace if missing (default: verify only)
#   --skip-stage     skip dataset staging (assume already on PVC)
#   --skip-server    skip server deploy (reuse running instance)
#   --smoke-only     run one concurrency rung, WALL-CAPPED (SMOKE_WALL, default 300s) — a quick path check,
#                    not a full sweep (llm-perf: c=1)
#   --teardown       scale server to 0 after sweep; --managed-ns --teardown also deletes namespace
#   --no-fetch       skip the local artifact fetch (`collect` later)
#   --fetch          explicitly opt IN to the local git-provenance fetch (overrides a prior --no-fetch). The
#                    local artifact fetch; detached runs are harvested later with collect.
#                    Attached short runs fetch by default; --detach never fetches locally.

#   --rungs "N ..."  pass a concurrency override to sweep.sh
#   --crashloop-max-restarts K  wait-ready fail-fast: abort if a server pod restartCount reaches K (default 5;
#                    env CRASHLOOP_MAX_RESTARTS). Restart-based, so a slow legitimate cold load never trips it.
#   --crashloop-window-min M    wait-ready fail-fast: abort if the server stays CrashLoopBackOff (0 ready) for
#                    M minutes (default 5; env CRASHLOOP_WINDOW_MIN).
#   --env-set K=V    VARIANT run (repeatable): upsert K=V on the serving containers at APPLY time
#   --env-unset K    VARIANT run (repeatable): remove K from the serving containers at APPLY time
#
# VARIANT RUNS (the RECIPE-BUILDER path) — --env-set/--env-unset let a builder iterate on a cell (strip a
# flag, try a value, run an A/B control) through THIS path instead of hand-applying a manifest, which loses
# the run-owner GC (leaked GPUs), fleet attribution, the artifacts PVC, preflight and wait-ready. The override
# is applied in deploy.sh's apply stream (scripts/merge_env_override.py), so:
#   • recipe_hash is UNCHANGED — rendered/*.yaml is never touched; an override is a runtime choice, and a
#     variant must not masquerade as a new recipe; and therefore
#   • the run is MARKED instead: every k8s object gets llmb.nvidia.com/variant=true + an annotation carrying
#     the exact overrides, results/<run-id>/_variant.json records them locally, run_meta.json carries them
#     into the run's provenance, and scripts/publish.py REFUSES the run-dir.
# Make a change PERMANENT (and publishable) by editing the recipe: see the /change-recipe skill.
set -euo pipefail

# ──── argument parsing ────────────────────────────────────────────────────────
MANAGED_NS=0
SKIP_STAGE=0
SKIP_SERVER=0
SMOKE_ONLY=0
TEARDOWN=0
NO_FETCH=0
FETCH_OPT=0 # --fetch: explicitly opt IN to a local artifact fetch (git-provenance). The DISCONNECT-PROOF
#   local artifact fetch is optional
#   and can be deferred to collect for detached runs. See below and the
#   attached short runs still fetch by default; long/detached runs do not.
DETACH=0 # --detach: apply the Job then EXIT (no wait/fetch/publish/teardown); harvest via `collect`
RUNGS_OVERRIDE=""
IDLE_GUARD=0
IDLE_MIN=30
SMOKE_WALL="${SMOKE_WALL:-300}"
# VARIANT run overrides (builder path) — accumulated newline-separated and consumed by deploy.sh's
# merge_env_override.py stage. Pre-seeded from the environment so a wrapper (campaign, parallel_repro) can
# set them without re-plumbing flags.
LLMB_ENV_SET="${LLMB_ENV_SET:-}"
LLMB_ENV_UNSET="${LLMB_ENV_UNSET:-}"
# wait-ready crash-loop fail-fast (see scripts/wait_server_ready.sh): abort early — instead of blocking the full
# startup_timeout_s (up to 14400s = 4h for 550B cells) — when the server is clearly not going to become ready.
# K/M are restart/CrashLoopBackOff based (never wall-clock) so a slow-but-legitimate cold load never trips them.
CRASHLOOP_MAX_RESTARTS="${CRASHLOOP_MAX_RESTARTS:-5}" # K: restartCount at/above which a server pod is crash-looping
CRASHLOOP_WINDOW_MIN="${CRASHLOOP_WINDOW_MIN:-5}"     # M: minutes of sustained CrashLoopBackOff (0 ready) before abort
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --managed-ns)
            MANAGED_NS=1
            shift
            ;;
        --skip-stage)
            SKIP_STAGE=1
            shift
            ;;
        --skip-server)
            SKIP_SERVER=1
            shift
            ;;
        --no-load-gate)
            # opt out of model-load admission control; loads may then contend on a shared model cache
            LOAD_GATE=0
            shift
            ;;
        --smoke-only)
            SMOKE_ONLY=1
            shift
            ;;
        --teardown)
            TEARDOWN=1
            shift
            ;;
        --no-fetch)
            NO_FETCH=1
            shift
            ;;
        --fetch)
            # Explicitly opt IN to the local git-provenance fetch (last-flag-wins over a prior --no-fetch).
            FETCH_OPT=1
            NO_FETCH=0
            shift
            ;;
        --detach)
            DETACH=1
            shift
            ;;
        --attach | --wait)
            # Explicit form of the default attached lifecycle. A later --detach still wins,
            # and vice-versa — last flag on the line decides, so a wrapper can append an override.
            DETACH=0
            shift
            ;;
        --idle-guard)
            IDLE_GUARD=1
            shift
            ;;
        --idle-guard=*)
            IDLE_GUARD=1
            IDLE_MIN="${1#*=}"
            shift
            ;;
        --crashloop-max-restarts)
            CRASHLOOP_MAX_RESTARTS="${2:?--crashloop-max-restarts needs a value}"
            shift 2
            ;;
        --crashloop-max-restarts=*)
            CRASHLOOP_MAX_RESTARTS="${1#*=}"
            shift
            ;;
        --crashloop-window-min)
            CRASHLOOP_WINDOW_MIN="${2:?--crashloop-window-min needs a value}"
            shift 2
            ;;
        --crashloop-window-min=*)
            CRASHLOOP_WINDOW_MIN="${1#*=}"
            shift
            ;;
        --rungs)
            RUNGS_OVERRIDE="${2:?--rungs needs a value}"
            shift 2
            ;;
        --rungs=*)
            RUNGS_OVERRIDE="${1#*=}"
            shift
            ;;
        --env-set)
            LLMB_ENV_SET="${LLMB_ENV_SET}${LLMB_ENV_SET:+$'\n'}${2:?--env-set needs KEY=VALUE}"
            shift 2
            ;;
        --env-set=*)
            LLMB_ENV_SET="${LLMB_ENV_SET}${LLMB_ENV_SET:+$'\n'}${1#*=}"
            shift
            ;;
        --env-unset)
            LLMB_ENV_UNSET="${LLMB_ENV_UNSET}${LLMB_ENV_UNSET:+$'\n'}${2:?--env-unset needs KEY}"
            shift 2
            ;;
        --env-unset=*)
            LLMB_ENV_UNSET="${LLMB_ENV_UNSET}${LLMB_ENV_UNSET:+$'\n'}${1#*=}"
            shift
            ;;
        # Reject unknown flags; repeat orchestration belongs to submit.sh --repeat.
        -*)
            echo "run: unknown flag '$1'" >&2
            echo "  valid flags: --managed-ns --skip-stage --skip-server --no-load-gate --smoke-only" >&2
            echo "               --teardown --fetch --no-fetch --detach --idle-guard --rungs" >&2
            echo "               --crashloop-max-restarts --crashloop-window-min --env-set --env-unset" >&2
            echo "  for N repeat legs (a variance sweep) use: scripts/submit.sh <cell> <profile> --repeat N" >&2
            exit 2
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

CELL="${1:?usage: run.sh <cell-dir> <cluster-profile> [run-id] [flags]}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"

[ -f "$ENVF" ] || {
    echo "run: no profile at $ENVF" >&2
    exit 1
}
[ -f "$CELL/recipe.yaml" ] || {
    echo "run: no recipe.yaml in $CELL" >&2
    exit 1
}
[ -d "$CELL/rendered" ] || {
    echo "run: no rendered/ in $CELL — run scripts/render.sh $CELL first" >&2
    exit 1
}

set -a
. "$ENVF"
set +a
# Every kubectl in this script pins the profile's KUBE_CONTEXT (when set), so the namespace / server / teardown
# steps target the SAME cluster preflight validated — not whatever the ambient kubectl context happens to be.
# (B200 test-lap finding: preflight honored KUBE_CONTEXT but run.sh's raw kubectl queried the ambient cluster
# and falsely reported the namespace missing.)
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
: "${OWNER:=$(whoami)}"

# ──── variant run (--env-set / --env-unset) ───────────────────────────────────────────────────────────
# Validate the override spec HERE, before the cluster is touched: a typo'd KEY=VALUE must abort, never
# degrade into a silent pinned-config run that gets read as a variant. VARIANT_ID is a stable 8-hex digest of
# the override set — the handle that appears in the k8s labels, the local marker, and run_meta.json.
export LLMB_ENV_SET LLMB_ENV_UNSET
VARIANT_ID=""
VARIANT_DESC=""
VARIANT_JSON=""
if [ -n "$LLMB_ENV_SET" ] || [ -n "$LLMB_ENV_UNSET" ]; then
    VARIANT_JSON="$(python3 "$ROOT/scripts/merge_env_override.py" --json)" || {
        echo "run: invalid --env-set/--env-unset spec (see above) — nothing was launched" >&2
        exit 1
    }
    VARIANT_ID="$(python3 "$ROOT/scripts/merge_env_override.py" --variant-id)"
    VARIANT_DESC="$(python3 "$ROOT/scripts/merge_env_override.py" --describe)"
fi

NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
SCENARIO="$(sed -n 's/^  scenario: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
MODE="$(sed -n 's/^  mode: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
[ -n "$NAME" ] || {
    echo "run: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}
[ -n "$SCENARIO" ] || {
    echo "run: could not read envelope.scenario from $CELL/recipe.yaml" >&2
    exit 1
}

# Default run-id: sized so the bench Job name <cell>-<kind>-<id> stays ≤63 (k8s DNS-1123 label). run_id.py
# --fit shrinks it for long cell names. kind comes from lane.py so run.sh + the bench scripts agree.
KIND="$(python3 "$ROOT/scripts/lane.py" "$CELL" kind 2> /dev/null || echo bench)"
# A hand-passed run-id ($3) is routed through --label so it too gets the compact UTC stamp prefix
# (rp3 → 260728t0312-rp3), flows into the results path + Job name + labels, and is shrunk to fit ≤63
# (already-dated ids pass through unchanged). Both paths fall back if the helper is unavailable.
if [ -n "${3:-}" ]; then
    export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit "$KIND" --label "$3" 2> /dev/null || echo "$3")"
else
    export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit "$KIND" 2> /dev/null || date -u +%Y%m%d-%H%M%S)"
fi

# Reserve the run identity BEFORE preflight, run-owner creation, model loading, or any GPU deployment. A
# very long cell can have only a few run-id characters left in Kubernetes' 63-char name budget; collisions
# are therefore possible even with a strong generator. The immutable launch receipt is the authority. Its
# reservation is atomic, preserves same-cell resume behavior, and mints a same-length counter variant when
# another recipe already owns the candidate. sweep.sh captures the same receipt again immediately before
# Job apply; that later call is intentionally idempotent, not the first time we discover a conflict.
_RUN_ID_CANDIDATE="$RUN_ID"
RUN_ID="$(python3 "$ROOT/scripts/launch_attestation.py" "$CELL" "$RUN_ID" \
    --reserve-root "$ROOT/results" --print-run-id)"
export RUN_ID
# Child benchmark lanes must use this already-reserved identity verbatim, not stamp it as a new user label.
export LLMB_RUN_ID_FINAL=1
if [ "$RUN_ID" != "$_RUN_ID_CANDIDATE" ]; then
    echo "run: run-id collision avoided before deploy: $_RUN_ID_CANDIDATE → $RUN_ID"
fi

# Failure recovery: one EXIT trap. On any non-zero exit AFTER the server started coming up
# (STEP past "startup"), print the likely cause + resume command (recovery.py) and free the GPU (scale the
# server to 0), so a crash / hang-kill / Ctrl-C ends actionably instead of leaving a held node + bare error.
# Also reaps the idle-guard. STEP is advanced at the risky phases below.
STEP="startup"
GUARD_PID=""
# Treat an external SIGTERM separately from a benchmark failure. The benchmark Job remains in the cluster,
# so the exit handler leaves its server running for the run-owner to clean up when the Job finishes.
trap 'exit 143' TERM
_on_exit() {
    local rc=$?
    [ -n "${GUARD_PID:-}" ] && kill "$GUARD_PID" 2> /dev/null || true
    if [ "$rc" -eq 143 ]; then
        echo "" >&2
        echo "  ⚠ externally terminated (SIGTERM — e.g. a session/context boundary), NOT a benchmark failure." >&2
        echo "    ${NAME}-server left UP so the in-flight bench Job (job/${WATCHED_JOB:-${NAME}-bench-${RUN_ID}}) is not orphaned;" >&2
        echo "    it keeps running in-cluster. The in-cluster run-owner (${RUN_OWNER_NAME:-<none>}) keeps watching it and" >&2
        echo "    will GC-cascade the GPU-holding server the instant the Job reaches a terminal state — no laptop needed." >&2
        echo "    Reattach: scripts/sweep.sh (follows the same Job). Free the GPU NOW: scripts/run_owner.sh teardown $NAMESPACE ${RUN_OWNER_NAME:-<owner>}" >&2
        return 0
    fi
    # Any abort that is NOT an external SIGTERM (Ctrl-C 130 / crash-loop fast-fail / deploy or benchmark
    # failure): HALT the run by deleting the run-owner → native GC cascade-deletes the server + bench
    # immediately (delete, NOT scale-to-0, so no idle-server 0/0 shell accumulates). This runs regardless of
    # STEP so a failure during deploy/wait-ready also frees whatever came up. If the run-owner was never
    # established, fall back to the legacy scale-to-0 of the server so the GPU is still freed.
    if [ "$rc" -ne 0 ] && [ "$STEP" != "done" ]; then
        if [ "$STEP" != "startup" ]; then
            echo "" >&2
            python3 "$ROOT/scripts/recovery.py" "$CELL" "$PROFILE" --exit "$rc" --reason "$STEP" --run-id "$RUN_ID" >&2 || true
            bash "$ROOT/scripts/fetch_results.sh" --partial "$CELL" "$PROFILE" "$RUN_ID" > /dev/null 2>&1 || true
            mkdir -p "$ROOT/results/$RUN_ID" 2> /dev/null || true
            python3 "$ROOT/scripts/recovery.py" --meta "$CELL" "$PROFILE" --exit "$rc" --reason "$STEP" \
                --run-id "$RUN_ID" > "$ROOT/results/$RUN_ID/run_meta.json" 2> /dev/null || true
        fi
        # never strand the model-load slot on an abnormal exit (release is idempotent)
        if [ "${LOAD_GATE_HELD:-0}" = "1" ]; then
            bash "$ROOT/scripts/model_load_gate.sh" release "$CELL" "$PROFILE" "$RUN_ID" \
                --run-owner "${RUN_OWNER_NAME:-}" > /dev/null 2>&1 || true
            LOAD_GATE_HELD=0
        fi
        if [ -n "${RUN_OWNER_NAME:-}" ]; then
            bash "$ROOT/scripts/run_owner.sh" teardown "$NAMESPACE" "$RUN_OWNER_NAME" > /dev/null 2>&1 || true
        else
            kc -n "$NAMESPACE" scale deploy "${NAME}-server" --replicas=0 > /dev/null 2>&1 || true
        fi
    fi
    if [ "$rc" -ne 0 ] && [ "$STEP" != "startup" ] && [ "$STEP" != "done" ]; then
        # Print partial timeline even on failure so the operator knows which phase took time.
        # Write _FAILED_ (not _END_) so watch_runs.py can distinguish failed from successful runs.
        printf '%s\t%s\n' "$(date -u +%s)" "_FAILED_" >> "${_PHASES_LOG:-}" 2> /dev/null || true
        python3 "$ROOT/scripts/run_summary.py" \
            --phases "${_PHASES_LOG:-}" --t0 "${T0:-0}" \
            --run-id "${RUN_ID:-}" --cell "${NAME:-}" --status failed 2> /dev/null || true
    fi
    # Do NOT delete phases.log — it lives in results/${RUN_ID}/ and is read by `llmb-k8s watch`
    # to show the completed timeline. It persists alongside the other run artifacts.
    return 0
}
trap _on_exit EXIT

# Lane routing — shared implementation (scripts/lane.py) shared with llmb-k8s, so the
# stage/bench scripts a cell uses never diverge between the two entry points.
STAGE_SCRIPT="$(python3 "$ROOT/scripts/lane.py" "$CELL" stage)" || exit 1
BENCH_SCRIPT="$(python3 "$ROOT/scripts/lane.py" "$CELL" bench)" || exit 1

T0=$(date -u +%s)
# Phase log at a stable, RUN_ID-named path so `llmb-k8s watch` can find it
# during parallel runs without knowing the /tmp tempfile name.
mkdir -p "$ROOT/results/$RUN_ID"
_PHASES_LOG="$ROOT/results/$RUN_ID/phases.log"
printf '%s\t%s\n' "$T0" "_START_" > "$_PHASES_LOG" # seed with T0
# VARIANT MARKER — written BEFORE anything is deployed, so a run launched with overrides is marked from the
# very first second even if it later crashes, is killed, or is collected by a different code path.
# scripts/publish.py refuses any run-dir carrying this file; it is the durable local half of the k8s
# labels/annotations that merge_env_override.py stamps in-cluster.
if [ -n "$VARIANT_ID" ]; then
    python3 - "$ROOT/results/$RUN_ID/_variant.json" "$VARIANT_JSON" \
        "$RUN_ID" "$CELL" "$PROFILE" "$NAME" << 'PY'
import json, sys, datetime
path, spec, run_id, cell, profile, recipe = sys.argv[1:7]
d = json.loads(spec)
d.update({"run_id": run_id, "cell": cell, "profile": profile, "recipe": recipe,
          "publishable": False,
          "reason": "launched with runtime env overrides (--env-set/--env-unset); the served configuration "
                    "differs from the committed recipe, so recipe_hash does NOT describe this run",
          "marked_utc": datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")})
open(path, "w").write(json.dumps(d, indent=2, sort_keys=True) + "\n")
PY
fi
phase() {
    echo ""
    echo "════ [$(date -u +%H:%M:%S)] $* ════"
    printf '%s\t%s\n' "$(date -u +%s)" "$*" >> "$_PHASES_LOG"
}
ok() { echo "  ✓ $*"; }
skip() { echo "  – $* (skipped)"; }
STARTUP_TIMEOUT_S="$(
    python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
print(int((r.get("serving") or {}).get("startup_timeout_s") or 900))
PY
)"
SERVER_DEPLOY="${NAME}-server"

# ──── detached-mode durable handles ───────────────────────────────────────────────────────────────────
# In --detach the laptop applies the Job and exits, so `llmb-k8s collect <run-id>` must be able to resolve the
# run-id → cell/profile/namespace WITHOUT this process. We write the SAME two durable handles scripts/submit.sh
# writes for its detached bench path, so collect (scripts/resilient_status.py) resolves a run.sh-detached run
# identically to a submit'd one:
#   • a LOCAL record  results/.submits/<run-id>.json  (same-laptop reconnect), and
#   • a cluster-side INDEX ConfigMap llmb-submit-<run-id> (no ownerRef/TTL → survives Job GC; cold cross-machine
#     `collect --cluster <profile>`).  collect deletes the index CM after a successful publish.
_detach_write_submit_record() { # writes results/.submits/<run-id>.json (schema per scripts/resilient_status.py)
    local dir="$ROOT/results/.submits"
    mkdir -p "$dir"
    cat > "$dir/${RUN_ID}.json" << JSON
{
  "run_id": "$RUN_ID",
  "cell": "$CELL",
  "profile": "$PROFILE",
  "namespace": "${NAMESPACE:-}",
  "recipe": "$NAME",
  "job_name": "${WATCHED_JOB:-${NAME}-${KIND}-${RUN_ID}}",
  "artifacts_pvc": "${NAME}-artifacts",
  "lane": "$KIND",
  "detached_via": "run.sh",
  "variant_id": "${VARIANT_ID:-}",
  "submitted_utc": "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
}
JSON
}
# Merge the override spec into the run's run_meta.json — the file publish/archive/export all read, so the
# variant travels with the run's PROVENANCE, not just with a side-car marker. Called after the fetch (the
# bench Job writes run_meta.json in-cluster, so it only exists locally once fetched). Best-effort by design:
# the authoritative refusal is _variant.json, which exists from before the deploy.
_stamp_variant_meta() {
    [ -n "${VARIANT_ID:-}" ] || return 0
    local meta="$ROOT/results/$RUN_ID/run_meta.json"
    [ -f "$meta" ] || return 0
    python3 - "$meta" "$VARIANT_JSON" << 'PY' 2> /dev/null || true
import json, sys
p, spec = sys.argv[1], sys.argv[2]
d = json.load(open(p))
d["overrides"] = json.loads(spec)          # {"set": {...}, "unset": [...], "variant_id": "..."}
d["variant_id"] = d["overrides"].get("variant_id")
d["publishable"] = False
json.dump(d, open(p, "w"), indent=2)
PY
}
_detach_apply_index_cm() { # persistent run-id → cell index (no ownerRef/TTL → survives the run-owner GC cascade)
    # The lane captured this immutable receipt immediately before kubectl apply. Persist the two scalar facts in
    # the detached-run index so a later `collect` on another machine can restore the local sidecar before
    # archive/publish. Do not recompute recipe_hash here: that was the archive-time provenance hole.
    local _launch_receipt="$ROOT/results/$RUN_ID/launch_attestation.json"
    local _launch_hash="" _launch_captured=""
    if [ -f "$_launch_receipt" ]; then
        _launch_hash="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("recipe_hash") or "")' "$_launch_receipt" 2> /dev/null || true)"
        _launch_captured="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("captured_at_utc") or "")' "$_launch_receipt" 2> /dev/null || true)"
    fi
    kc apply -f - << EOF 2> /dev/null || echo "run: WARN — index ConfigMap apply failed (local record still written)" >&2
apiVersion: v1
kind: ConfigMap
metadata:
  name: llmb-submit-${RUN_ID}
  namespace: ${NAMESPACE}
  labels:
    app.kubernetes.io/managed-by: llmb-recipe
    llmb.nvidia.com/run-id: ${RUN_ID}
    llmb.nvidia.com/lifecycle: detached
    llmb.nvidia.com/variant: "$([ -n "${VARIANT_ID:-}" ] && echo true || echo false)"
data:
  variant_id: "${VARIANT_ID:-}"
  env_overrides: '${VARIANT_JSON:-}'
  run_id: "${RUN_ID}"
  cell: "${CELL}"
  profile: "${PROFILE}"
  namespace: "${NAMESPACE}"
  recipe: "${NAME}"
  recipe_hash_at_launch: "${_launch_hash}"
  recipe_hash_captured_at_utc: "${_launch_captured}"
  artifacts_pvc: "${NAME}-artifacts"
  job_name: "${WATCHED_JOB:-${NAME}-${KIND}-${RUN_ID}}"
  submitted_utc: "$(date -u +%Y-%m-%dT%H:%M:%SZ)"
EOF
}

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  llmb-k8s run  │  cell=$NAME  │  profile=$PROFILE"
echo "║  scenario=$SCENARIO  │  run-id=$RUN_ID"
if [ -n "$VARIANT_ID" ]; then
    echo "║  ⚠ VARIANT $VARIANT_ID  │  $VARIANT_DESC"
    echo "║  runtime override — recipe_hash UNCHANGED; results are NOT publishable"
fi
echo "╚══════════════════════════════════════════════════════════════╝"

# ──── 1. preflight ────────────────────────────────────────────────────────────
phase "1/8  preflight"
# Two sub-steps: offline then live.
#
# dryrun.sh (OFFLINE) — resolves profile vars, runs kubeconform, flags unresolved ${VAR}s.
# No cluster access; catches misconfig that would silently break scheduling or mounts.
# dryrun.sh owns its own validation and exits non-zero on unresolved variables/schema errors. Do not parse
# its human-readable output; rows like "recipe name:" and warnings are useful context, not failures.
_dry_rc=0
set +e
preflight_out="$(bash "$ROOT/scripts/dryrun.sh" "$CELL" "$PROFILE" 2>&1)"
_dry_rc=$?
set -e
if [ "$_dry_rc" -ne 0 ]; then
    echo "run: preflight (offline) found errors — fix them before continuing:" >&2
    printf '%s\n' "$preflight_out" >&2
    exit 1
fi
ok "offline check passed (dryrun.sh)"

# Live preflight validates the namespace, GPU availability, PVCs, secrets, and artifact storage class.
# Tee to a temp file so the ✅/❌ detail lines stream live for a human AND survive to re-emit on stderr on
# failure — otherwise (audit #5) the "resolve the issues above" banner lands on stderr with the detail on
# stdout, so a split-stream CI log shows the banner with no context.
_pf_out="$(mktemp)"
_pf_rc=0
set +e
python3 "$ROOT/scripts/preflight.py" "$CELL" "$PROFILE" 2>&1 | tee "$_pf_out"
_pf_rc=${PIPESTATUS[0]}
set -e
if [ "$_pf_rc" -eq 2 ]; then
    {
        echo "run: cluster unreachable — check VPN/Teleport, then retry"
        cat "$_pf_out"
    } >&2
    rm -f "$_pf_out"
    exit 1
elif [ "$_pf_rc" -ne 0 ]; then
    {
        echo ""
        echo "run: live preflight failed — resolve the issues above:"
        cat "$_pf_out"
        echo "run:   or provision missing prereqs with: llmb-k8s install --cluster $PROFILE"
    } >&2
    rm -f "$_pf_out"
    exit 1
fi
rm -f "$_pf_out"
ok "live preflight passed (preflight.py)"

# Auto-probe fabric: for disagg cells whose rendered manifests reference ${RDMA_UCX_NET_DEVICES}
# from the cluster profile, run probe-fabric once per profile if RDMA_FABRIC_PROBED is not set.
# Writes discovered values + sentinel to the profile, then re-sources it so deploy.sh picks them up.
# Non-fatal: on failure RDMA_UCX_NET_DEVICES stays "all" (UCX auto-select); never blocks deploy.
# The RDMA_FABRIC_PROBED sentinel prevents re-probing on every run once the profile is populated —
# even on clusters with no rdma/* capacity resources where UCX stays "all" after probing.
if grep -qlE 'RDMA_UCX_(TLS|NET_DEVICES|IB_ADDR_TYPE|MAX_RNDV_RAILS)' "$CELL/rendered/"*.yaml 2> /dev/null; then
    # Export safe defaults for all RDMA profile vars so deploy.sh envsubst never expands to "".
    # Profiles pre-dating this feature lack these lines; the defaults keep the pod config valid.
    export RDMA_UCX_NET_DEVICES="${RDMA_UCX_NET_DEVICES:-all}"
    export RDMA_UCX_IB_ADDR_TYPE="${RDMA_UCX_IB_ADDR_TYPE:-}"
    export RDMA_UCX_MAX_RNDV_RAILS="${RDMA_UCX_MAX_RNDV_RAILS:-4}"
    # UCX_TLS moved out of the 27 recipes into the profile. The default is the exact InfiniBand list
    # those recipes baked, so a profile that predates RDMA_UCX_TLS deploys byte-identically to before.
    # It is IB-oriented on purpose (that is what it replaces) — a RoCE cluster MUST set RDMA_UCX_TLS in
    # its profile without dc/dc_x, and preflight WARNs when the profile leaves it to this default.
    export RDMA_UCX_TLS="${RDMA_UCX_TLS:-rc,rc_x,dc,dc_x,cuda_copy}"
    if [ -z "${RDMA_FABRIC_PROBED:-}" ]; then
        phase "1.5/8  auto-probe fabric"
        echo "  RDMA_FABRIC_PROBED not set — discovering IB devices from live nodes (runs once per profile)..."
        if python3 "$ROOT/scripts/probe_fabric.py" "$PROFILE" --write 2>&1 | sed 's/^/  /'; then
            set -a
            . "$ENVF"
            set +a # re-source: picks up RDMA_UCX_NET_DEVICES + RDMA_FABRIC_PROBED
            ok "fabric probed → RDMA_UCX_NET_DEVICES=${RDMA_UCX_NET_DEVICES:-all}"
        else
            echo "  ⚠  probe-fabric failed (non-fatal) — UCX_NET_DEVICES='all' (UCX auto-select)"
        fi
    fi
fi

# 0.10 advisory (non-fatal): is this cell's committed record still tied to the current recipe? Surface DRIFT
# so a recipe edit that silently invalidated the published numbers is seen here, not later in CI.
_impact="$(python3 "$ROOT/scripts/provenance.py" "$CELL" --impact 2> /dev/null || true)"
case "$_impact" in DRIFT*) echo "  ⚠ ${_impact%%$'\n'*} (see: scripts/provenance.py $CELL --impact)" ;; esac

# ──── 2. namespace ────────────────────────────────────────────────────────────
phase "2/8  namespace  (ns=$NAMESPACE)"
if kc get namespace "$NAMESPACE" > /dev/null 2>&1; then
    ok "namespace $NAMESPACE exists"
elif [ "$MANAGED_NS" = 1 ]; then
    echo "  creating namespace "
    kc create namespace ""
    ok "namespace $NAMESPACE created"
else
    echo "run: namespace $NAMESPACE does not exist." >&2
    echo "run:   Ask your admin to create it, or pass --managed-ns to create it here." >&2
    exit 1
fi

# ──── 2.5 run-owner  (the PRIMARY, intrinsic GPU-lifecycle guarantee) ─────────────────────────────────
# Create the per-run run-owner Job FIRST — BEFORE the server — so the GPU-holding server Deployment is
# created OWNED FROM BIRTH (deploy.sh's merge_run_owner.py stamps its ownerReference at apply time, reading
# RUN_OWNER_NAME/UID exported here). The run-owner is a tiny watcher Job: it watches the bench Job and
# exits the instant it reaches ANY terminal state (Complete OR Failed OR gone), which Completes the owner; its
# short ttlSecondsAfterFinished then deletes it and native GC cascade-frees the server+bench — promptly, on
# fail as on finish, even if this orchestrator's laptop is gone (no client, no polling governor). On a hang
# the owner's activeDeadlineSeconds is the ceiling. Best-effort: if it can't be created (empty uid), the flow
# degrades to the legacy adopt_server.sh + governor backstop. See scripts/run_owner.sh for the full rationale.
WATCHED_JOB="${NAME}-${KIND}-${RUN_ID}"
LOAD_GATE="${LOAD_GATE:-1}" # --no-load-gate disables model-load admission control
LOAD_GATE_HELD=0
RUN_OWNER_NAME=""
RUN_OWNER_UID=""
phase "2.5/8  run-owner  (watches job/$WATCHED_JOB → GC cascade-frees the GPU on any terminal state)"
eval "$(bash "$ROOT/scripts/run_owner.sh" ensure "$NAMESPACE" "$NAME" "$RUN_ID" "$WATCHED_JOB" 2> /tmp/runowner.$$.err || true)"
sed 's/^/  /' /tmp/runowner.$$.err 2> /dev/null || true
rm -f /tmp/runowner.$$.err 2> /dev/null || true
export RUN_OWNER_NAME RUN_OWNER_UID
if [ -n "$RUN_OWNER_UID" ]; then
    ok "run-owner up: $RUN_OWNER_NAME (server+bench become its GC children — disconnect-proof GPU freeing)"
else
    echo "  ⚠ run-owner not established — falling back to adopt_server.sh + governor backstop (GPU still freed, less promptly)"
fi

# ──── 3. stage ───────────────────────────────────────────────────────────────
phase "3/8  stage  (scenario=$SCENARIO${MODE:+/$MODE} → $STAGE_SCRIPT)"
# Record the outcome into the per-cluster install stamp (cluster-profiles/.state/<cluster>.install.jsonl) so a
# cell staged HERE by `run` — not only by `llmb-k8s install` — appears in fleet's INSTALLED inventory ("what
# can be run here"), and a STAGE FAILURE is visible there too rather than silent. We reached this point only
# after the live preflight (phase 1) PASSED, so preflight=pass. Best-effort: `|| true` — a stamp write must
# never break a run. Reuses install.write_install_stamp so the schema matches what fleet reads.
_STEP_KEY="${STAGE_SCRIPT%.sh}"
_stamp_stage() { # <staged: ok|failed|skipped>
    python3 "$ROOT/scripts/install.py" --record-stage "$CELL" "$PROFILE" \
        --step "$_STEP_KEY" --staged "$1" --preflight pass --job-mode "$KIND" 2> /dev/null || true
}
if [ "$SKIP_STAGE" = 1 ]; then
    skip "staging (--skip-stage)"
    # already-staged by a prior run/install → record it as run-here (skipped this time, preflight passed).
    _stamp_stage skipped
else
    set +e
    bash "$ROOT/scripts/$STAGE_SCRIPT" "$CELL" "$PROFILE"
    _stage_rc=$?
    set -e
    if [ "$_stage_rc" -ne 0 ]; then
        _stamp_stage failed # record the FAILED stage → visible in fleet INSTALLED, not silent
        echo "run: staging failed (rc=$_stage_rc) — see $STAGE_SCRIPT output above" >&2
        exit "$_stage_rc"
    fi
    ok "staged ($STAGE_SCRIPT)"
    _stamp_stage ok
fi

# ──── 4. server ───────────────────────────────────────────────────────────────
phase "4/8  server deploy"
if [ "$SKIP_SERVER" = 1 ]; then
    skip "server deploy (--skip-server)"
    # --skip-server reuses a server that is ALREADY up, so the override never reaches a container. Say so
    # loudly — a builder who thinks they ran a variant and actually re-measured the pinned config is exactly
    # the failure this feature exists to prevent. The run stays MARKED as a variant either way (conservative:
    # we would rather refuse to publish a clean run than publish a variant one).
    if [ -n "$VARIANT_ID" ]; then
        echo "  ⚠ --env-set/--env-unset were NOT applied: --skip-server reuses the running server as-is." >&2
        echo "    Drop --skip-server to redeploy with the override. This run stays marked VARIANT (unpublishable)." >&2
    fi
    # The server already exists (a prior run brought it up), so it did NOT get this run's ownerReference at
    # apply time. Re-point the existing server Deployment(s) at THIS run's run-owner so the per-run lifetime
    # invariant still holds (and a stale prior owner can't cascade a server this run is using). Best-effort.
    if [ -n "$RUN_OWNER_UID" ]; then
        bash "$ROOT/scripts/run_owner.sh" adopt-deploy "$NAMESPACE" "$NAME" "$RUN_OWNER_NAME" "$RUN_OWNER_UID" 2>&1 | sed 's/^/  /' || true
    fi
else
    # Serialize only the model-load phase for runs sharing a cache PVC. Different PVCs remain independent,
    # and the lease is released as soon as the server becomes Ready.
    if [ "${LOAD_GATE:-1}" = "1" ]; then
        bash "$ROOT/scripts/model_load_gate.sh" acquire "$CELL" "$PROFILE" "$RUN_ID" \
            --run-owner "${RUN_OWNER_NAME:-}" 2>&1 | sed 's/^/  /' || true
        LOAD_GATE_HELD=1
    fi
    bash "$ROOT/scripts/deploy.sh" "$CELL" "$PROFILE"
    # Verify the exact server Deployments carry the run-owner even if an apply-time owner patch was dropped.
    if [ -n "$RUN_OWNER_UID" ]; then
        bash "$ROOT/scripts/run_owner.sh" adopt-deploy "$NAMESPACE" "$NAME" "$RUN_OWNER_NAME" "$RUN_OWNER_UID" 2>&1 | sed 's/^/  /' || true
    fi
    if [ -n "$RUN_OWNER_UID" ]; then
        ok "server manifests applied (owned from birth by run-owner $RUN_OWNER_NAME)"
    else
        ok "server manifests applied"
    fi
fi

# Detached mode applies the benchmark Job and returns without waiting, fetching, publishing, or tearing down.
# The run-owner releases serving resources when the Job reaches a terminal state, while result artifacts remain
# available for a later `llmb-k8s collect <run-id>` command.
if [ "$DETACH" = 1 ]; then
    STEP="benchmark"

    # Rung override for the Job apply (mirrors the attached step-6 mapping): --smoke-only → first rung
    # (llm-perf sweep → "1"); --rungs → verbatim; replay ignores rungs.
    _DETACH_RUNGS=()
    if [ "$SMOKE_ONLY" = 1 ] && [ "$BENCH_SCRIPT" = "sweep.sh" ]; then
        _DETACH_RUNGS=(--rungs "1")
    elif [ -n "$RUNGS_OVERRIDE" ]; then
        _DETACH_RUNGS=(--rungs "$RUNGS_OVERRIDE")
    fi

    phase "5/5  apply benchmark Job (DETACHED → $BENCH_SCRIPT; then EXIT)"
    # The lane's bench script applies its ConfigMaps + the Job + adopts the run-owner, THEN blocks following
    # logs ("Ctrl-C detaches; the Job keeps running"). We want ONLY the apply: run it backgrounded, wait until
    # stop following after the Job is applied and adopted by the run-owner. Reusing the bench script
    # (rather than re-inlining its ConfigMap+Job apply) keeps the detached apply from drifting from the attached
    # one. RUN_OWNER_NAME/UID are exported, so the bench script's adopt-job binds the Job to the run-owner.
    WATCHED_JOB="${NAME}-${KIND}-${RUN_ID}"
    (bash "$ROOT/scripts/$BENCH_SCRIPT" "$CELL" "$PROFILE" "$RUN_ID" "${_DETACH_RUNGS[@]+"${_DETACH_RUNGS[@]}"}" \
        > "$ROOT/results/$RUN_ID/apply.log" 2>&1) &
    APPLY_PID=$!

    # ── WHICH Job did we actually apply? ────────────────────────────────────────────────────────────────
    # "${NAME}-${KIND}-${RUN_ID}" is run.sh's GUESS at the Job name — and a guess is all it is. A lane script
    # may mint its OWN run-id, so the real Job can be named something else entirely.
    #
    # So ask the AUTHORITATIVE question instead — "which Job did MY run-owner adopt?" — keyed on the run-owner
    # UID. Every lane stamps it (sweep.sh and others all call `run_owner.sh adopt-job`), it is the
    # very link this poll is trying to verify, and unlike a name it cannot drift. The predicted name is only
    # used as a tiebreaker. Degraded (no owner uid → adopt-job is a no-op): fall back to the cell+lane name
    # PREFIX restricted to a non-terminal Job, never to a bare "some job of this cell exists".
    #
    # FAIL-SAFE, NOT FAIL-CLOSED. Tearing a healthy multi-hour run down to reclaim GPUs is far worse than
    # holding GPUs a while. We tear down on exactly ONE reading — a SUCCESSFUL cluster read showing NO
    # candidate Job at all, after the apply process has already exited ("genuinely gone"). A failed read, an
    # ambiguous match, or a timeout while the apply is still in flight leaves EVERYTHING running and says so.
    _DETACH_JOBS_JSONPATH='{range .items[*]}{.metadata.name}{"|"}{.metadata.ownerReferences[*].uid}'
    _DETACH_JOBS_JSONPATH="${_DETACH_JOBS_JSONPATH}"'{"|"}{.status.succeeded}{"|"}{.status.failed}{"\n"}{end}'
    _detach_scan_job() { # STDOUT (one word/line): "err" | "none" | "unadopted" | "ambiguous" | "job <name>"
        local raw n uids s f cands="" count cellish=0
        raw="$(kc -n "$NAMESPACE" get jobs -o jsonpath="$_DETACH_JOBS_JSONPATH" 2> /dev/null)" || {
            echo err # the READ failed (apiserver/cert/context) — never confuse that with "the Job is gone"
            return 0
        }
        while IFS='|' read -r n uids s f; do
            [ -n "$n" ] || continue
            if [ -n "${RUN_OWNER_NAME:-}" ] && [ "$n" = "$RUN_OWNER_NAME" ]; then continue; fi # not its own child

            # A live Job of THIS cell, whoever owns it: enough to forbid a "genuinely gone" verdict below.
            case "$n" in "${NAME}-"*) case "${s:-0}${f:-0}" in 00) cellish=1 ;; esac ;; esac
            if [ -n "${RUN_OWNER_UID:-}" ]; then
                case " $uids " in *" $RUN_OWNER_UID "*) ;; *) continue ;; esac
            else
                case "$n" in "${NAME}-${KIND}-"*) ;; *) continue ;; esac
                case "${s:-0}${f:-0}" in 00) ;; *) continue ;; esac # skip a prior run's TTL-lingering Job
            fi
            cands="${cands}${n}"$'\n'
        done << EOF
$raw
EOF
        count="$(printf '%s' "$cands" | grep -c . || true)"
        # Nothing adopted, but a live Job of this cell exists (adoption still in flight, or it failed): that is
        # NOT "gone" — never tear down on it.
        if [ "${count:-0}" -eq 0 ] && [ "$cellish" = 1 ]; then
            echo unadopted
            return 0
        fi
        if [ "${count:-0}" -eq 0 ]; then
            echo none
            return 0
        fi
        if [ "${count:-0}" -eq 1 ]; then
            echo "job $(printf '%s' "$cands" | grep . | head -1)"
            return 0
        fi
        # >1 adopted Job: if one of them IS the predicted name, that is unambiguous enough; else refuse to guess.
        if printf '%s' "$cands" | grep -qxF -- "$WATCHED_JOB"; then
            echo "job $WATCHED_JOB"
            return 0
        fi
        echo ambiguous
    }

    _applied=0
    _found=""
    _scan="none"
    _apply_alive=1
    for _i in $( # up to ~240s: covers ConfigMap+Job apply + adopt-job under any apiserver lag
        seq 1 120
    ); do
        _scan="$(_detach_scan_job)"
        if [ "${_scan#job }" != "$_scan" ]; then
            _found="${_scan#job }"
            if [ -z "${RUN_OWNER_UID:-}" ]; then
                # No run-owner (degraded): the bench script's backstop adopt_server (server->Job ownerRef)
                # runs just after the Job apply. Grace a few seconds so it completes before we stop the script,
                # otherwise the server could be left unowned. The governor backstop still covers a miss.
                sleep 5
            fi
            _applied=1
            break
        fi
        if ! kill -0 "$APPLY_PID" 2> /dev/null; then
            # The bench script exited. Look ONCE more (the Job can land in the final instant) before deciding —
            # this is the only place a "genuinely gone" verdict can be reached.
            _apply_alive=0
            sleep 3
            _scan="$(_detach_scan_job)"
            if [ "${_scan#job }" != "$_scan" ]; then
                _found="${_scan#job }"
                _applied=1
            fi
            break
        fi
        sleep 2
    done
    # Stop the follow-loop; the Job + server are durable in-cluster (run-owner owns them). The Job keeps running.
    kill "$APPLY_PID" 2> /dev/null || true
    wait "$APPLY_PID" 2> /dev/null || true
    if [ "$_applied" = 0 ]; then
        if [ "$_scan" = "none" ] && [ "$_apply_alive" = 0 ]; then
            STEP="benchmark"
            echo "run: DETACH — the bench Job did not register in-cluster; see results/$RUN_ID/apply.log" >&2
            exit 1
        fi
        # UNIDENTIFIABLE, NOT ABSENT: the read errored, >1 Job matched, or the apply was still in flight at the
        # timeout. Do NOT tear down — a healthy run may be underneath this. STEP=done keeps the EXIT trap's
        # hands off the cluster; the run-owner's own activeDeadlineSeconds + TTL still reclaim the GPU if this
        # really did fail, so nothing is stranded forever.
        STEP="done"
        {
            echo ""
            echo "  ⚠ DETACH — could not identify this run's bench Job (scan: $_scan)."
            echo "    NOTHING WAS TORN DOWN: the server + any applied Job are LEFT RUNNING, deliberately — an"
            echo "    unreadable/ambiguous cluster answer must never delete a possibly-healthy run."
            echo "    Check it:  kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE get jobs" \
                "-l llmb.nvidia.com/cell=$NAME"
            echo "    Apply log: results/$RUN_ID/apply.log"
            echo "    If it really failed, free the GPU now (run_owner.sh only inherits KUBE_CONTEXT from the"
            echo "    environment, so pass it or you will target the wrong cluster):"
            echo "      ${KUBE_CONTEXT:+KUBE_CONTEXT=$KUBE_CONTEXT }scripts/run_owner.sh teardown" \
                "$NAMESPACE ${RUN_OWNER_NAME:-<owner>}"
            echo "    Otherwise the run-owner reclaims it on the Job's terminal state (or its deadline)."
        } >&2
        exit 1
    fi
    if [ "$_found" != "$WATCHED_JOB" ]; then
        echo "  note: the lane minted its own run-id — the applied Job is job/$_found, not the predicted" >&2
        echo "        job/$WATCHED_JOB. Tracking the real one (matched via run-owner ownerReference)." >&2
        echo "        (The in-cluster run-owner matches by ownership too, so it still frees the GPU promptly" >&2
        echo "        on this Job's terminal state — not on its activeDeadlineSeconds.)" >&2
        WATCHED_JOB="$_found"
    fi
    ok "applied job/$WATCHED_JOB (detached; run-owner ${RUN_OWNER_NAME:-<none>} owns the GPU lifecycle)"

    # Durable run-id handles so `collect` can resolve this run without the laptop (see helper defs above).
    _detach_write_submit_record
    _detach_apply_index_cm
    STEP="done" # past setup: the EXIT trap must NOT treat our clean exit as a failure/teardown

    T1=$(date -u +%s)
    printf '%s\t%s\n' "$T1" "_DETACHED_" >> "$_PHASES_LOG"
    cat << EOF

╔══════════════════════════════════════════════════════════════╗
║  DETACHED — setup done in $((T1 - T0))s. You may disconnect (laptop + tsh) now.
╚══════════════════════════════════════════════════════════════╝
  run-id:  $RUN_ID
  cell:    $CELL   (ns=$NAMESPACE, profile=$PROFILE)
  The bench Job runs in-cluster; the run-owner (${RUN_OWNER_NAME:-<none>}) frees the GPU on its terminal
  state. Results persist server-side (artifacts PVC + netscore-$RUN_ID ConfigMap survive the GC cascade).
  ARTIFACTS:        persist on the run PVC; harvest with collect when convenient.
  GIT-PROVENANCE:   OPT-IN — RESULTS.md/record.json need local artifacts; harvest with collect when convenient.

  status:  llmb-k8s status  $RUN_ID --cluster $PROFILE
  collect: llmb-k8s collect $RUN_ID --cluster $PROFILE     # opt-in local fetch + git-provenance publish
  watch:   llmb-k8s fleet --watch
EOF
    # Printed AFTER the banner (not inside it) so the normal, non-variant detach output is byte-unchanged.
    # The collect hint matters here specifically: _variant.json lives on THIS laptop, so a cross-machine
    # collect must carry the marker over from the in-cluster index ConfigMap or it would look clean.
    if [ -n "$VARIANT_ID" ]; then
        echo ""
        echo "  ⚠ VARIANT $VARIANT_ID — $VARIANT_DESC. NOT publishable: publish.py refuses results/$RUN_ID"
        echo "    (_variant.json), and every k8s object carries"
        echo "    llmb.nvidia.com/variant=true. Collecting on a DIFFERENT machine? Carry the marker across:"
        echo "      kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE get cm llmb-submit-$RUN_ID \\"
        echo "        -o jsonpath='{.data.env_overrides}' > results/$RUN_ID/_variant.json"
    fi
    exit 0
fi

# ──── 5. wait-ready ───────────────────────────────────────────────────────────
phase "5/8  wait-ready  (timeout: ${STARTUP_TIMEOUT_S}s; crash-loop fail-fast: restarts>=${CRASHLOOP_MAX_RESTARTS} or CrashLoopBackOff>=${CRASHLOOP_WINDOW_MIN}m)"
STEP="wait-ready"
# Wait for aggregate or disaggregated serving pods and fail fast on sustained crash loops.
# On abort, the EXIT trap releases GPU resources and prints recovery guidance.
bash "$ROOT/scripts/wait_server_ready.sh" "$CELL" "$PROFILE" \
    --timeout "$STARTUP_TIMEOUT_S" \
    --max-restarts "$CRASHLOOP_MAX_RESTARTS" \
    --window-min "$CRASHLOOP_WINDOW_MIN"
ok "server ready ($SERVER_DEPLOY)"
# Weights are resident → free the model-load slot so a queued run can start loading while WE benchmark.
if [ "${LOAD_GATE_HELD:-0}" = "1" ]; then
    bash "$ROOT/scripts/model_load_gate.sh" release "$CELL" "$PROFILE" "$RUN_ID" \
        --run-owner "${RUN_OWNER_NAME:-}" 2>&1 | sed 's/^/  /' || true
    LOAD_GATE_HELD=0
fi

# ──── idle-guard (optional) ───────────────────────────────────────────────────
# Background watchdog: kills the run's Job if the server generates no new tokens for IDLE_MIN minutes (a genuine
# hang would otherwise block sweep.sh forever). --keep-server: run.sh owns teardown below, so the guard only
# unblocks a hang; it does not scale the server itself. Cleaned up on exit.
GUARD_PID=""
if [ "$IDLE_GUARD" = 1 ] || [ "$SMOKE_ONLY" = 1 ]; then
    _GUARD_ARGS=(--keep-server)
    _GUARD_MSG=""
    [ "$IDLE_GUARD" = 1 ] && {
        _GUARD_ARGS+=(--kill-hung --idle-timeout "$IDLE_MIN")
        _GUARD_MSG="hang window ${IDLE_MIN}m"
    }
    # --smoke-only maps to a single low rung, but a c=1 pass over a long trace is still an overnight job — cap the
    # wall so a smoke actually stays a smoke (proves the path, then stops). SMOKE_WALL overridable (default 300s).
    [ "$SMOKE_ONLY" = 1 ] && {
        _GUARD_ARGS+=(--max-wall "$SMOKE_WALL")
        _GUARD_MSG="${_GUARD_MSG:+$_GUARD_MSG; }smoke wall ${SMOKE_WALL}s"
    }
    phase "idle-guard  ($_GUARD_MSG)"
    bash "$ROOT/scripts/idle_guard.sh" "$CELL" "$PROFILE" "$RUN_ID" "${_GUARD_ARGS[@]}" &
    GUARD_PID=$! # reaped by the _on_exit trap set above
    ok "idle-guard watching (pid $GUARD_PID): $_GUARD_MSG"
fi

# ──── 6. benchmark (sweep / replay) ───────────────────────────────────────────
phase "6/8  benchmark  (→ $BENCH_SCRIPT)"
STEP="benchmark"
[ "$BENCH_SCRIPT" = "sweep.sh" ] || die "unsupported benchmark script in external package: $BENCH_SCRIPT"
SWEEP_EXTRA=()
if [ "$SMOKE_ONLY" = 1 ]; then
    SWEEP_EXTRA+=(--rungs "1")
    echo "  smoke-only mode: concurrency=1, wall-capped at ${SMOKE_WALL}s (proves the path; not a publishable result)"
elif [ -n "$RUNGS_OVERRIDE" ]; then
    SWEEP_EXTRA+=(--rungs "$RUNGS_OVERRIDE")
    echo "  rungs override: $RUNGS_OVERRIDE"
fi
bash "$ROOT/scripts/sweep.sh" "$CELL" "$PROFILE" "$RUN_ID" "${SWEEP_EXTRA[@]+"${SWEEP_EXTRA[@]}"}"
ok "benchmark complete ($BENCH_SCRIPT)"
STEP="done" # past here, a non-zero exit (fetch/teardown) isn't a run failure — trap stays quiet

# ──── 7. fetch (OPT-IN local artifacts for git-provenance) ─────────────────────────────────────────────────
# Attached runs fetch artifacts by default; --no-fetch leaves them on the PVC for a later collect.
phase "7/8  fetch artifacts"
if [ "$NO_FETCH" = 1 ]; then
    skip "artifact fetch (--no-fetch) — run 'llmb-k8s collect $RUN_ID' for git-provenance"
else
    # Use the shared fetch path for retries, authentication recovery, and per-rung preservation.
    bash "$ROOT/scripts/fetch_results.sh" "$CELL" "$PROFILE" "$RUN_ID"
    # the run's own provenance file now exists locally — record the overrides IN it (see _stamp_variant_meta)
    _stamp_variant_meta
    ok "artifacts → $ROOT/results/$RUN_ID"
fi

# ──── 8. teardown ─────────────────────────────────────────────────────────────
phase "8/8  teardown"
if [ "$TEARDOWN" = 1 ]; then
    if [ -n "$RUN_OWNER_UID" ]; then
        # Primary path: delete the run-owner → native GC cascade-DELETES the owned server + bench (delete,
        # NOT scale-to-0, so no idle-server 0/0 shell accumulates).
        bash "$ROOT/scripts/run_owner.sh" teardown "$NAMESPACE" "$RUN_OWNER_NAME" 2>&1 | sed 's/^/  /' || true
        ok "run-owner $RUN_OWNER_NAME deleted → GC cascade-freeing the GPU (server + bench deleted, no 0/0 shell)"
    elif kc -n "$NAMESPACE" get deployment "$SERVER_DEPLOY" > /dev/null 2>&1; then
        # Backstop path (no run-owner established): legacy scale-to-0 so the GPU is still freed.
        kc -n "$NAMESPACE" scale deployment "$SERVER_DEPLOY" --replicas=0
        ok "scaled $SERVER_DEPLOY to 0 (GPU slot freed; resume: scripts/deploy.sh $CELL $PROFILE)"
    else
        echo "  no single-deployment server to scale; skipping (disagg stacks: scale workers manually)"
    fi
    if [ "$MANAGED_NS" = 1 ]; then
        echo "  --managed-ns + --teardown: deleting namespace $NAMESPACE"
        echo "  (waiting 10s — ctrl-C to abort)"
        sleep 10
        kc delete namespace "$NAMESPACE" --ignore-not-found
        ok "namespace $NAMESPACE deleted"
    fi
else
    skip "teardown (pass --teardown to scale server to 0 after sweep)"
fi

# ──── done ────────────────────────────────────────────────────────────────────
T1=$(date -u +%s)
ELAPSED=$((T1 - T0))

# Timeline summary — print before the Next: block so the operator sees durations first.
printf '%s\t%s\n' "$T1" "_END_" >> "$_PHASES_LOG"
python3 "$ROOT/scripts/run_summary.py" \
    --phases "$_PHASES_LOG" --t0 "$T0" \
    --run-id "$RUN_ID" --cell "$NAME" --status ok 2> /dev/null || true
# ──── the RESULT ─────────────────────────────────────────────────────────────
# The verdict comes BEFORE "Next:", because "Next: publish" is not an answer to "was my run any good?".
# Everything needed is already local and verified by the fetch receipt; nothing rendered it, so a user
# finishing a 35-minute benchmark had to run two more scripts by hand to learn what it measured. This also
# WRITES metrics_summary.csv into the run directory — it did not exist until someone aggregated manually,
# leaving the directory missing the one file a human opens first.
# Never fails the run: a reporting problem is reported, not escalated (run_result.py always exits 0).
# Distinguish an intentional --no-fetch from a fetch that produced no local artifacts.
if [ "$NO_FETCH" = 1 ]; then
    echo ""
    echo "──── Result ─────────────────────────────────────────────────────────────"
    echo "  ⚠  NOT EVALUATED — results were not fetched to this machine (--no-fetch), so there is nothing"
    echo "     local to score. This is not a statement about the run."
    echo "     fetch, then score:  scripts/fetch_results.sh $RUN_ID"
elif [ -d "$ROOT/results/$RUN_ID" ]; then
    python3 "$ROOT/scripts/run_result.py" "$CELL" "$ROOT/results/$RUN_ID" || true
else
    echo ""
    echo "──── Result ─────────────────────────────────────────────────────────────"
    echo "  ⚠  NOT EVALUATED — the fetch ran, but $ROOT/results/$RUN_ID is not on this machine."
    echo "     Nothing local to score, and this is NOT a statement about the run: the benchmark may have"
    echo "     completed fine and the transfer failed."
    echo "     retry the fetch:  scripts/fetch_results.sh $RUN_ID"
fi

echo ""
echo "Next:"
if [ -n "$VARIANT_ID" ]; then
    # A variant is a DIAGNOSTIC, not a result: never print a publish command for it. publish.py refuses this
    # run-dir (results/$RUN_ID/_variant.json), so pointing at it here would only invite a --force attempt.
    echo "  ⚠ VARIANT $VARIANT_ID ($VARIANT_DESC) — NOT publishable (publish.py refuses results/$RUN_ID)"
    echo "    inspect:   results/$RUN_ID   (overrides recorded in _variant.json + run_meta.json)"
    echo "    make it real: edit the recipe (see the /change-recipe skill), re-render, re-run WITHOUT overrides"
elif [ "$NO_FETCH" = 0 ]; then
    echo "  scripts/publish.py $CELL results/$RUN_ID"
else
    echo "  scripts/fetch_results.sh $RUN_ID   # then: scripts/publish.py $CELL results/$RUN_ID"
fi
echo "  watch all runs: llmb-k8s fleet --watch"
if [ "$TEARDOWN" = 0 ]; then
    echo ""
    if [ -n "$RUN_OWNER_UID" ]; then
        echo "Server is owned by the run-owner ($RUN_OWNER_NAME). The bench Job has completed, so the run-owner"
        echo "watcher is now Complete → its ~${RUNOWNER_TTL_S:-120}s ttlSecondsAfterFinished then fires and native GC"
        echo "cascade-DELETES the server + bench (no idle-server 0/0 shell squats the GPU). To act now:"
        echo "  scripts/run_owner.sh teardown $NAMESPACE $RUN_OWNER_NAME   # free the GPU immediately (cascade)"
        echo "  scripts/deploy.sh $CELL $PROFILE                           # (re)deploy for another sweep"
    else
        echo "Server is still running (no run-owner was established — the governor backstop will reclaim it). To act now:"
        echo "  kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE scale deployment/$SERVER_DEPLOY --replicas=0  # free GPU slot immediately"
        echo "  scripts/deploy.sh $CELL $PROFILE                                    # (re)deploy for another sweep"
    fi
fi
