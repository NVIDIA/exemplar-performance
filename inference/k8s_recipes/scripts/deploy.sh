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

# deploy.sh <cell-dir> <cluster-profile> [--render-only] [--env-set K=V]... [--env-unset K]...
# Resolve profile variables in committed manifests and apply them. Variant overrides are
# runtime-only, explicitly labelled, and excluded from publishable results.
set -euo pipefail

CELL=""
PROFILE=""
RENDER_ONLY=0
LLMB_ENV_SET="${LLMB_ENV_SET:-}"     # newline-separated KEY=VALUE, read by merge_env_override.py
LLMB_ENV_UNSET="${LLMB_ENV_UNSET:-}" # newline-separated KEY
while [ $# -gt 0 ]; do
    case "$1" in
        --render-only)
            RENDER_ONLY=1
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
        *)
            if [ -z "$CELL" ]; then CELL="$1"; elif [ -z "$PROFILE" ]; then PROFILE="$1"; fi
            shift
            ;;
    esac
done
export LLMB_ENV_SET LLMB_ENV_UNSET
[ -n "$CELL" ] || {
    echo "usage: deploy.sh <cell-dir> <cluster-profile> [--render-only] [--env-set K=V] [--env-unset K]" >&2
    exit 1
}
[ -n "$PROFILE" ] || {
    echo "deploy.sh: need a cluster-profile name (cluster-profiles/<name>.env)" >&2
    exit 1
}

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "deploy.sh: no profile at $ENVF" >&2
    exit 1
}
[ -d "$CELL/rendered" ] || {
    echo "deploy.sh: no rendered/ in $CELL — run scripts/render.sh first" >&2
    exit 1
}
command -v envsubst > /dev/null || {
    echo "deploy.sh: envsubst not found (gettext)" >&2
    exit 1
}

set -a
. "$ENVF"
set +a # export the profile's cluster vars
# Resolve the same per-model cache claim used by installation; fail before rendering if unavailable.
. "$ROOT/scripts/_model_cache.sh"
llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

# Validate and announce runtime variant overrides before applying resources.
VARIANT_SPEC=""
if [ -n "${LLMB_ENV_SET:-}" ] || [ -n "${LLMB_ENV_UNSET:-}" ]; then
    VARIANT_SPEC="$(python3 "$ROOT/scripts/merge_env_override.py" --json)" || {
        echo "deploy.sh: invalid --env-set/--env-unset spec (see above) — nothing applied" >&2
        exit 1
    }
    echo "deploy.sh: ⚠ VARIANT deploy — $(python3 "$ROOT/scripts/merge_env_override.py" --describe)"
    echo "deploy.sh:   runtime override only: recipe_hash is UNCHANGED, rendered/*.yaml untouched."
    echo "deploy.sh:   objects are marked llmb.nvidia.com/variant=true; results are NOT publishable."
fi

# A direct `llmb-k8s deploy` is a supported live entry point, not merely an internal helper for run.sh.
# Gate it after local argument validation but before the first apply, just like run/submit: a disaggregated
# server needs one model-cache claim mountable by independently scheduled prefill/decode workers.
# A RWO claim can look healthy and complete yet deadlock at attach time on multiple nodes. Keep render-only
# fully offline, and keep malformed builder overrides fail-fast before any cluster read.
if [ "$RENDER_ONLY" = 0 ]; then
    _pf_out="$(mktemp)"
    set +e
    python3 "$ROOT/scripts/preflight.py" "$CELL" "$PROFILE" --stage-only 2>&1 | tee "$_pf_out"
    _pf_rc=${PIPESTATUS[0]}
    set -e
    if [ "$_pf_rc" -ne 0 ]; then
        rm -f "$_pf_out"
        echo "deploy.sh: live compatibility preflight failed - nothing applied" >&2
        exit "$_pf_rc"
    fi
    rm -f "$_pf_out"
fi

for f in "$CELL"/rendered/*.yaml; do
    [ -e "$f" ] || continue
    # The bench Job is launched separately (scripts/sweep.sh) — it needs a per-run RUN_ID + a strict
    # envsubst whitelist so the runner script's own ${BASH} refs survive. deploy.sh = serving stack only.
    case "$(basename "$f")" in bench-job.yaml) continue ;; esac
    # whitelist ONLY the ${VARS} this file uses, so $SNAPSHOT and $(python3 ...) inside pod args survive.
    wl=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*\}' "$f" | tr -d '${}' | sort -u | sed 's/^/$/' | paste -sd' ' -)
    # Two symmetric, hash-neutral APPLY-time IMEX patchers, keyed on the profile's NVLINK_MULTICAST_IMEX:
    #   • merge_imex_claim.py — when =provisioned, injects the ComputeDomain channel claim (Tier-2 fusion
    #     recovery) so FLASHINFER stays ON.
    #   • merge_imex_strip.py — when NOT provisioned, strips forced VLLM_ATTENTION_BACKEND=FLASHINFER so a
    #     FLASHINFER-forcing cell on an IMEX-less cluster runs correct-but-slower instead of CrashLooping on
    #     cuMulticastCreate code=800 (the C1 graceful degrade). Exactly one acts; both no-op otherwise.
    # Neither is baked into rendered/, so recipe_hash is unchanged (same discipline as merge_rdma_selector).
    # merge_run_owner.py is the LAST apply-stream stage: when run.sh has created a per-run run-owner Job and
    # exported RUN_OWNER_NAME/RUN_OWNER_UID, it stamps the ownerReference onto the GPU-holding server
    # Deployment(s) HERE — at apply time — so the server is created OWNED FROM BIRTH (zero unowned window;
    # native GC cascade-frees the GPU the instant the run-owner terminates). No run-owner in play (standalone
    # deploy) → it passes the stream through unchanged, and adopt_server.sh + the governor stay the backstop.
    # Hash-neutral (apply-stream patch, never rendered/*.yaml), exactly like the merge_imex_* stages.
    # merge_env_override.py is the BUILDER stage: with --env-set/--env-unset in play it patches the serving
    # containers' env AND marks every object llmb.nvidia.com/variant=true + annotates the exact overrides, so a
    # variant run is visible in fleet/kubectl and refused by publish. No override → byte-identical passthrough
    # (zero regression for the normal path). It runs BEFORE merge_run_owner so the ownerReference stamp — the
    # feature whose absence leaks GPUs — is applied to the patched objects exactly as on a normal deploy.
    if [ "$RENDER_ONLY" = 1 ]; then
        echo "# ---------- $(basename "$f") ----------"
        envsubst "$wl" < "$f" | python3 "$ROOT/scripts/merge_rdma_selector.py" | python3 "$ROOT/scripts/merge_imex_claim.py" | python3 "$ROOT/scripts/merge_imex_strip.py" | python3 "$ROOT/scripts/merge_env_override.py" | python3 "$ROOT/scripts/merge_run_owner.py"
    else
        echo "apply: $(basename "$f")  (ns=${NAMESPACE:-?})"
        envsubst "$wl" < "$f" | python3 "$ROOT/scripts/merge_rdma_selector.py" | python3 "$ROOT/scripts/merge_imex_claim.py" | python3 "$ROOT/scripts/merge_imex_strip.py" | python3 "$ROOT/scripts/merge_env_override.py" | python3 "$ROOT/scripts/merge_run_owner.py" | kc apply -f -
    fi
done

# Stamp lifetime-binding labels on the GPU-holding server Deployment(s) as a RUNTIME patch — hash-neutral. They
# are deliberately NOT in the server templates: rendered/server.yaml (and the sglang-disagg worker manifests)
# are recipe_hash inputs, so baking these in would drift every published cell's hash. As live `kubectl label`s
# they are runtime metadata (never fingerprinted). Three labels, all keyed to the EXACT server-object names
# (never an open prefix), so a sibling cell's server is never labelled:
#   • llmb.nvidia.com/cell           — POSITIVE cell→server link for the governor's orphan-sweep owner fallback
#     (governor_reconcile.sh sweep_orphans) so a genuine orphan resolves by cell; its ABSENCE is never a reap.
#   • llmb.nvidia.com/created-at      — deploy-time wall clock as EPOCH SECONDS (k8s label values forbid ISO's
#     ':'), the stable start for the governor's max-lifetime HARD CEILING (survives a governor restart / re-list).
#   • llmb.nvidia.com/max-lifetime-s  — the per-server lifetime ceiling (seconds). This is the belt to the
#     ownerRef/orphan-sweep suspenders: a TRUE k8s ownerRef→Job is impossible at deploy time (the server is
#     created BEFORE the bench Job exists; adopt_server stamps the ownerRef only once the Job is up), so a
#     runtime lifetime label is the simplest correct binding. The governor reaps a server past this ceiling only
#     when no live bench Job for the cell exists. Override the 24-hour default with SERVER_MAX_LIFETIME_S.
if [ "$RENDER_ONLY" = 0 ]; then
    NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
    if [ -n "$NAME" ]; then
        _created_at="$(date -u +%s)"
        _max_life="${SERVER_MAX_LIFETIME_S:-86400}" # 24h default; profile may override
        for _obj in "${NAME}-server" "${NAME}-prefill" "${NAME}-decode"; do
            kc -n "${NAMESPACE:-}" label deploy "$_obj" \
                "llmb.nvidia.com/cell=${NAME}" \
                "llmb.nvidia.com/created-at=${_created_at}" \
                "llmb.nvidia.com/max-lifetime-s=${_max_life}" \
                --overwrite > /dev/null 2>&1 || true
        done
    fi
fi
