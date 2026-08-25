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

# dryrun.sh <cell-dir> <cluster-profile>
#
# Validates a recipe cell without touching the cluster.
#
#   scripts/dryrun.sh recipes/llm-perf/256k/nemotron-ultra-3-b200-vllm-agg example-gpu-cluster
#
# What it does:
#   1. Sources the cluster profile and resolves all ${VARS} via envsubst (same as deploy.sh +
#      sweep.sh, but writes to a temp dir instead of calling kubectl apply).
#   2. Prints a summary table of all key resolved values (image, model, GPUs, PVCs, node selector).
#   3. Validates the resolved manifests with kubeconform (if available on PATH).
#   4. Warns about common misconfigurations (empty BENCH_NODE_SELECTOR, missing DCGM URL, etc.).
#
# Exit code: 0 = all checks pass, 1 = validation errors found.
# Does NOT require cluster access. Safe to run anywhere.
set -euo pipefail

# Return whether a variable name appears in a space-delimited allowlist.
in_list() { case " $2 " in *" $1 "*) return 0 ;; *) return 1 ;; esac }

# classify_vars <owned> <tok...> — split ${VAR} tokens into "unresolved" (owned) vs "preserved" (runner
# placeholders) and print two lines:  BAD:[ ... ]  then  KEPT:[ ... ]. <owned> is a space list of bare var
# names the resolver fills, or the literal __ALL__ (every token is owned → any leftover is a real gap).
# The variable-resolution report and the --classify test sub-mode share this ONE definition.
classify_vars() {
    local owned="$1"
    shift
    local bad="" kept="" tok var
    for tok in "$@"; do
        var="${tok#\$\{}"
        var="${var%\}}"
        if [ "$owned" = "__ALL__" ] || in_list "$var" "$owned"; then bad="$bad $tok"; else kept="$kept $tok"; fi
    done
    echo "BAD:[$bad ]"
    echo "KEPT:[$kept ]"
}

# Hidden test sub-mode (no cluster/profile needed):  dryrun.sh --classify "<owned>" ${TOK} ${TOK} ...
# Exercised by scripts/selftest.py so the placeholder-vs-unresolved logic can never silently regress.
if [ "${1:-}" = "--classify" ]; then
    shift
    classify_vars "$@"
    exit 0
fi

CELL="${1:?usage: dryrun.sh <cell-dir> <cluster-profile>}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"

[ -f "$ENVF" ] || {
    echo "dryrun: no profile at $ENVF" >&2
    exit 1
}
[ -d "$CELL/rendered" ] || {
    echo "dryrun: no rendered/ in $CELL — run scripts/render.sh $CELL first" >&2
    exit 1
}
command -v envsubst > /dev/null || {
    echo "dryrun: envsubst not found (install gettext)" >&2
    exit 1
}

# Load cluster profile (same as deploy.sh / sweep.sh).
set -a
. "$ENVF"
set +a

# Resolve the model-cache claim with the same shared helper used by install and deploy.
. "$ROOT/scripts/_model_cache.sh"
llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1

export RUN_ID="dryrun-$(date -u +%Y%m%d-%H%M%S)"
: "${OWNER:=$(whoami)}" "${CACHE_BUST:=}" "${DCGM_EXPORTER_URL:=}"
: "${BENCH_NODE_SELECTOR:=}" "${BENCH_CPU_REQUEST:=16}"
: "${NO_INTERNET_DNS_IP:=}" "${NO_INTERNET_KUBE_API_IP:=}"                                    # live no-internet policy
: "${RDMA_UCX_NET_DEVICES:=all}" "${RDMA_UCX_IB_ADDR_TYPE:=}" "${RDMA_UCX_MAX_RNDV_RAILS:=4}" # disagg fabric
# Default to InfiniBand transports; RoCE profiles must override this value.
: "${RDMA_UCX_TLS:=rc,rc_x,dc,dc_x,cuda_copy}"
export OWNER CACHE_BUST DCGM_EXPORTER_URL BENCH_NODE_SELECTOR BENCH_CPU_REQUEST NO_INTERNET_DNS_IP NO_INTERNET_KUBE_API_IP
export RDMA_UCX_NET_DEVICES RDMA_UCX_IB_ADDR_TYPE RDMA_UCX_MAX_RNDV_RAILS RDMA_UCX_TLS

NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
[ -n "$NAME" ] || {
    echo "dryrun: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}

TMPD=$(mktemp -d)
trap 'rm -rf "$TMPD"' EXIT
ERRORS=0

# ──── Summary table ───────────────────────────────────────────────────────────
echo "=== DRY RUN: $(basename "$CELL") → profile=$PROFILE ==="
echo ""
echo "──── Resolved cluster values ────────────────────────────────────────────"
printf "  %-30s %s\n" "recipe name:" "$NAME"
printf "  %-30s %s\n" "cluster profile:" "$PROFILE"
printf "  %-30s %s\n" "namespace:" "${NAMESPACE:-<UNSET — fix in profile>}"
printf "  %-30s %s\n" "owner:" "${OWNER:-<UNSET>}"
printf "  %-30s %s\n" "gpu_product:" "${GPU_PRODUCT:-<UNSET>}"
printf "  %-30s %s\n" "image_pull_secret:" "${IMAGE_PULL_SECRET:-<UNSET>}"
printf "  %-30s %s\n" "model_cache_pvc:" "${MODEL_CACHE_PVC:-<UNSET>}"
printf "  %-30s %s\n" "artifacts_class:" "${ARTIFACTS_STORAGE_CLASS:-<UNSET>}"
printf "  %-30s %s\n" "artifacts_size:" "${ARTIFACTS_SIZE:-20Gi}"
if [ -n "${BENCH_NODE_SELECTOR:-}" ]; then
    printf "  %-30s %s\n" "bench_node_selector:" "$BENCH_NODE_SELECTOR"
else
    printf "  %-30s %s\n" "bench_node_selector:" "(empty — bench pod may land on GPU nodes)"
fi
printf "  %-30s %s\n" "bench_cpu_request:" "${BENCH_CPU_REQUEST:-16}"
printf "  %-30s %s\n" "dcgm_exporter_url:" "${DCGM_EXPORTER_URL:-(unset, exec telemetry)}"
echo ""

# ──── Resolve each manifest ───────────────────────────────────────────────────
render_auto() {
    # Auto-whitelist: extract every ${VAR} token from the file so only those are substituted.
    local f="$1" out="$2"
    local wl
    wl=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*\}' "$f" | tr -d '${}' | sort -u \
        | sed 's/^/$/' | paste -sd' ' - || true)
    envsubst "$wl" < "$f" > "$out"
}

# server.yaml (deploy.sh path)
if [ -f "$CELL/rendered/server.yaml" ]; then
    render_auto "$CELL/rendered/server.yaml" "$TMPD/server.yaml"
fi

# bench-job.yaml (sweep.sh path — strict whitelist to protect embedded shell vars)
if [ -f "$CELL/rendered/bench-job.yaml" ]; then
    WL_BENCH='$NAMESPACE $RUN_ID $OWNER $IMAGE_PULL_SECRET $MODEL_CACHE_PVC $DCGM_EXPORTER_URL $CACHE_BUST $BENCH_NODE_SELECTOR $BENCH_CPU_REQUEST'
    envsubst "$WL_BENCH" < "$CELL/rendered/bench-job.yaml" > "$TMPD/bench-job.yaml"
fi

# Additional driver-job.yaml paths (if any overlay driver exists). Whitelist covers every bracketed var
# in the rendered driver — all cluster vars (no embedded-runner placeholders), so any leftover is a real gap.

# ──── kubeconform validation ──────────────────────────────────────────────────
if command -v kubeconform > /dev/null 2>&1; then
    echo "──── kubeconform validation ─────────────────────────────────────────────"
    for f in "$TMPD"/*.yaml; do
        [ -f "$f" ] || continue
        if output=$(kubeconform -strict -ignore-missing-schemas "$f" 2>&1); then
            printf "  OK   %s\n" "$(basename "$f")"
        else
            printf "  FAIL %s\n" "$(basename "$f")"
            printf '%s\n' "$output" | sed 's/^/       /'
            ERRORS=$((ERRORS + 1))
        fi
    done
    echo ""
else
    echo "──── kubeconform (not on PATH — skipping manifest schema validation) ────"
    echo "  Install: https://github.com/yannh/kubeconform/releases"
    echo "  Also available via: make ci  (downloads kubeconform automatically)"
    echo ""
fi

# ──── Variable resolution check ──────────────────────────────────────────────
# Classify remaining variables as unresolved profile values or intentionally preserved in-pod placeholders.
echo "──── Variable resolution ────────────────────────────────────────────────"
VARS_OK=1
for f in "$TMPD"/*.yaml; do
    [ -f "$f" ] || continue
    base="$(basename "$f")"
    found=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*\}' "$f" | sort -u || true)
    [ -n "$found" ] || continue
    case "$base" in
        # Only bench-job.yaml embeds a runner script whose ${UPPER} refs are meant to survive. Everywhere else
        # (server.yaml auto-whitelist, driver-job.yaml all-cluster-vars) every leftover token is a real gap.
        bench-job.yaml) owned="$(printf '%s' "${WL_BENCH:-}" | tr -d '$')" ;; # cluster vars the resolver fills
        *) owned="__ALL__" ;;
    esac
    bad="$(classify_vars "$owned" $found | sed -n 's/^BAD:\[\(.*\) \]$/\1/p')"
    kept="$(classify_vars "$owned" $found | sed -n 's/^KEPT:\[\(.*\) \]$/\1/p')"
    if [ -n "$bad" ]; then
        echo "  $base: UNRESOLVED cluster vars — set them in the profile or add to the template whitelist:"
        for t in $bad; do echo "    $t"; done
        ERRORS=$((ERRORS + 1))
        VARS_OK=0
    fi
    if [ -n "$kept" ]; then
        echo "  $base: preserved runner placeholders (expanded in-pod at sweep time, not here):"
        echo "    $(echo $kept)"
    fi
done
[ "$VARS_OK" = 1 ] && echo "  OK — every cluster var resolved (runner placeholders intentionally preserved)"
echo ""

# ──── Warnings ───────────────────────────────────────────────────────────────
echo "──── Warnings ───────────────────────────────────────────────────────────"
WARNED=0
if [ -z "${BENCH_NODE_SELECTOR:-}" ]; then
    # Explain why an unconstrained bench pod may be unschedulable for whole-node cells.
    echo "  WARN: BENCH_NODE_SELECTOR is empty — the bench pod may schedule onto a GPU node"
    if grep -q 'WHOLE_NODE_CPU' "$CELL"/rendered/*.yaml 2> /dev/null; then
        echo "        This cell reserves a WHOLE NODE for its server, so a bench pod landing there has to"
        echo "        fit in what the reservation leaves (WHOLE_NODE_CPU + BENCH_CPU_REQUEST <= allocatable)."
        echo "        If it does not, the server comes up healthy and the bench pod sits FailedScheduling."
    fi
    echo "        Add to $ENVF: BENCH_NODE_SELECTOR=\"<key>: <value>\""
    WARNED=$((WARNED + 1))
fi
if [ -z "${DCGM_EXPORTER_URL:-}" ]; then
    echo "  NOTE: DCGM_EXPORTER_URL is unset — GPU telemetry will use kubectl exec (needs pods/exec RBAC)"
    echo "        Set DCGM_EXPORTER_URL in $ENVF to use HTTP scrape instead (no RBAC needed)"
    WARNED=$((WARNED + 1))
fi
if [ -z "${MODEL_CACHE_PVC:-}" ]; then
    echo "  WARN: MODEL_CACHE_PVC is unset — set it in $ENVF to the PVC holding model weights"
    ERRORS=$((ERRORS + 1))
fi
if [ -z "${NAMESPACE:-}" ]; then
    echo "  WARN: NAMESPACE is unset — set it in $ENVF"
    ERRORS=$((ERRORS + 1))
fi
[ "$WARNED" -gt 0 ] || [ "$ERRORS" -gt 0 ] || echo "  none"
echo ""

# ──── Result ─────────────────────────────────────────────────────────────────
if [ "$ERRORS" -gt 0 ]; then
    echo "DRY RUN FAILED: $ERRORS error(s) — fix above before deploying"
    exit 1
fi
echo "DRY RUN OK"
echo "  deploy:  scripts/deploy.sh $CELL $PROFILE"
echo "  sweep:   scripts/sweep.sh  $CELL $PROFILE [run-id]"
