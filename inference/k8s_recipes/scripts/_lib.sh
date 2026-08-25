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

# Shared helpers for the llmb-benchmark k8s recipe scripts. Sourced, never run.
set -eu

# Resolve recipe root (parent of scripts/) regardless of where the caller is.
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export RECIPE_ROOT="$(cd "$_LIB_DIR/.." && pwd)"

# Serving backend the shared harness drives. The default backend is the
# aggregated vLLM stack; its chart/, manifests/ and contexts/ overlays live
# under serving/vllm-aggregated/. (The experimental dynamo-disagg backend is
# applied directly with kubectl — see serving/dynamo-disagg/README.md.)
export SERVING_ROOT="${SERVING_ROOT:-$RECIPE_ROOT/serving/vllm-aggregated}"

# ----- logging --------------------------------------------------------------
log() { printf '[%s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }
warn() { printf '[%s] WARN: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
err() { printf '[%s] ERROR: %s\n' "$(date -u +%H:%M:%S)" "$*" >&2; }
die() {
    err "$*"
    exit 1
}

# ----- exit codes -----------------------------------------------------------
# Reserved exit code emitted by llmb::ensure_kube_context when kube creds went
# stale during a long-running orchestrator invocation (e.g. mid `kubectl wait`)
# AND the orchestrator had already authenticated successfully earlier in this
# same invocation. Distinct from 1 (real failure) so callers / CI can tell
# "Job completed but post-run fetch needs manual rerun" apart from "something
# actually went wrong". Recovery: re-auth (`tsh kube login <cluster>`) then
# run scripts/fetch_results.sh <run-id> && scripts/report.sh <run-id>.
LLMB_EXIT_AUTH_STALE=2

# Wait for a Kubernetes Job to reach Complete. Wrapper around `kubectl wait`
# that distinguishes three failure modes:
#
#   0  Job completed cleanly (kubectl wait returned 0).
#   1  Job genuinely did NOT complete in time (wait returned non-zero AND
#      the Job's Complete condition is not "True").
#   $LLMB_EXIT_AUTH_STALE  Wait returned non-zero, BUT a short-timeout
#      `kubectl get job` confirms the Job IS Complete. That means the
#      Job finished and the wait failure was an auth/network glitch on
#      the orchestrator side — distinguishable from a real failure.
#
# A separate `can-i` probe with a short timeout is used to detect "no auth
# at all" (e.g. teleport session evicted) so the caller can print a
# different hint pointing at re-auth rather than just "rerun fetch".
#
# Arguments:
#   $1 = job name (within $NAMESPACE)
#   $2 = timeout (e.g. "18000s"); passed to the inner `kubectl wait`.
llmb::wait_job_complete() {
    local job="$1" timeout="$2"
    local rc=0
    llmb::kc wait "job/$job" --for=condition=Complete --timeout="$timeout" 2> /dev/null || rc=$?
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    # Wait failed. Decide: real failure vs auth/network blip vs creds-fully-gone.
    local complete
    complete="$(llmb::kc get "job/$job" \
        -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' \
        --request-timeout=10s 2> /dev/null || true)"
    if [ "$complete" = "True" ]; then
        warn "kubectl wait failed (rc=$rc) but job/$job is Complete — treating as auth-stale."
        return "${LLMB_EXIT_AUTH_STALE:-2}"
    fi
    # Probe whether we still have any kube auth at all.
    if ! kubectl auth can-i get pods -n "$NAMESPACE" \
        --request-timeout=10s > /dev/null 2>&1; then
        err "Lost kube auth during wait for job/$job (can-i probe also failed)."
        err "Re-auth then rerun fetch/report manually:"
        err "    tsh kube login ${KUBE_PROXY:+--proxy=$KUBE_PROXY }${KUBE_CLUSTER:-$CLUSTER}"
        err "    scripts/fetch_results.sh ${RUN_ID:-<run-id>}"
        err "    scripts/report.sh ${RUN_ID:-<run-id>}"
        return "${LLMB_EXIT_AUTH_STALE:-2}"
    fi
    # Auth is alive AND the Job is not Complete → genuine failure.
    return "$rc"
}

# ----- env loading ----------------------------------------------------------
# Loads recipe.env then the chosen cluster profile. Cluster profile keys win
# on overlap. Generates a RUN_ID if none was provided.
#
# CLUSTER must be set to the basename (no extension) of a profile in
# cluster-profiles/. If unset, we try to auto-detect a single non-template
# profile; if there's ambiguity we error out and point the user at the
# porting guide rather than silently defaulting.
llmb::load_env() {
    if [ ! -f "$RECIPE_ROOT/recipe.env" ]; then
        die "recipe.env not found at $RECIPE_ROOT/recipe.env"
    fi

    local cluster="${CLUSTER:-}"
    if [ -z "$cluster" ]; then
        # Auto-detect: exactly one real *.env (the gitignored user-owned files;
        # the tracked *.env.example templates don't match `*.env`).
        local matches
        matches=$(ls -1 "$RECIPE_ROOT/cluster-profiles"/*.env 2> /dev/null \
            | awk -F/ '{print $NF}' \
            | sed 's/\.env$//' || true)
        local count
        count=$(printf '%s\n' "$matches" | grep -c . || true)
        if [ "$count" -eq 1 ]; then
            cluster="$matches"
            log "CLUSTER not set; auto-detected single cluster profile: $cluster"
        elif [ "$count" -eq 0 ]; then
            err "No cluster profile found. Set CLUSTER=<your-cluster> or create one:"
            err "  cp $RECIPE_ROOT/cluster-profiles/_template.env.example \\"
            err "     $RECIPE_ROOT/cluster-profiles/<your-cluster>.env"
            err "See $RECIPE_ROOT/cluster-profiles/README.md for the full porting guide."
            exit 1
        else
            err "Multiple cluster profiles present; CLUSTER must be set explicitly:"
            printf '%s\n' "$matches" | sed 's/^/  CLUSTER=/' >&2
            exit 1
        fi
    fi

    local profile="$RECIPE_ROOT/cluster-profiles/${cluster}.env"
    if [ ! -f "$profile" ]; then
        err "Cluster profile not found: $profile"
        if [ -f "$RECIPE_ROOT/cluster-profiles/${cluster}.env.example" ]; then
            err "Copy the matching example and edit it:"
            err "  cp $RECIPE_ROOT/cluster-profiles/${cluster}.env.example $profile"
        else
            err "Start from the generic template:"
            err "  cp $RECIPE_ROOT/cluster-profiles/_template.env.example $profile"
            err "See $RECIPE_ROOT/cluster-profiles/README.md."
        fi
        exit 1
    fi

    # Auto-export every assignment so envsubst (a subprocess) sees them.
    set -a
    . "$RECIPE_ROOT/recipe.env"
    . "$profile"

    # Optional context overlay (256k | 1m), sourced LAST so it overrides recipe.env
    # + the cluster profile. Set by scripts/benchmark.sh. Only touches the
    # context-length + dataset knobs; see serving/vllm-aggregated/contexts/<name>.env.
    if [ -n "${CONTEXT:-}" ]; then
        local ctx_file="$SERVING_ROOT/contexts/${CONTEXT}.env"
        if [ ! -f "$ctx_file" ]; then
            set +a
            die "CONTEXT='$CONTEXT' but $ctx_file not found (expected 256k or 1m)"
        fi
        . "$ctx_file"
        log "Applied context overlay: $CONTEXT (CONTEXT_LENGTH=$CONTEXT_LENGTH, dataset=$DATASET_SUBPATH)"
    fi
    # Optional per-invocation concurrency override (e.g. a parallel matrix track
    # runs a subset of the sweep). Survives because it is never set in the sourced
    # files. Empty leaves the recipe/profile CONCURRENCIES intact.
    if [ -n "${CONCURRENCIES_OVERRIDE:-}" ]; then
        CONCURRENCIES="$CONCURRENCIES_OVERRIDE"
        log "Applied CONCURRENCIES override: $CONCURRENCIES"
    fi
    # Optional per-invocation model-cache PVC override. Lets a launcher spread
    # concurrent 550B cold loads across REPLICA FSx filesystems: each PVC is its
    # own filesystem with its own provisioned throughput, so N replicas = N×
    # aggregate read bandwidth and simultaneous loads stop starving each other on
    # one shared filesystem. Cluster-scoped (not part of recipe_hash), so a run on
    # a replica PVC is the same fingerprint / a legitimate reproducibility sample.
    # Survives because it is never set in the sourced files; empty leaves the
    # profile's MODEL_CACHE_PVC intact.
    if [ -n "${MODEL_CACHE_PVC_OVERRIDE:-}" ]; then
        MODEL_CACHE_PVC="$MODEL_CACHE_PVC_OVERRIDE"
        log "Applied MODEL_CACHE_PVC override: $MODEL_CACHE_PVC"
    fi
    set +a

    # Cluster profile may override the SGLang image (e.g. mirror.gcr.io).
    if [ -n "${SGLANG_IMAGE_OVERRIDE:-}" ]; then
        SGLANG_IMAGE="$SGLANG_IMAGE_OVERRIDE"
    fi

    # Owner defaults to the current user.
    OWNER="${OWNER:-$(whoami)}"
    # Teleport login target defaults to the profile selector. Profiles whose
    # filename differs from the real Teleport cluster can override this.
    KUBE_CLUSTER="${KUBE_CLUSTER:-$CLUSTER}"

    # Run-id: collision-free per-namespace identifier.
    RUN_ID="${RUN_ID:-$(date -u +%Y%m%d-%H%M%S)-${OWNER}-${RANDOM}}"

    export CLUSTER KUBE_CLUSTER RUN_ID OWNER SGLANG_IMAGE
}

# Sanity-check a required env var is set (after llmb::load_env).
llmb::require_var() {
    local v="$1"
    if [ -z "${!v:-}" ]; then
        die "Required variable $v is not set (check recipe.env or cluster-profiles/${CLUSTER}.env)"
    fi
}

# Sanity-check a required command is on PATH.
llmb::require_cmd() {
    command -v "$1" > /dev/null 2>&1 || die "Required command '$1' not found on PATH"
}

# ----- envsubst rendering ---------------------------------------------------
# Renders serving/vllm-aggregated/manifests/*.yaml into that dir's .rendered/
# with envsubst, then strips conditional blocks based on IMAGE_PULL_SECRET /
# USE_DYNAMO_TOLERATIONS.
llmb::render_manifests() {
    local src="$SERVING_ROOT/manifests"
    local dst="$SERVING_ROOT/manifests/.rendered"
    llmb::require_cmd envsubst

    rm -rf "$dst"
    mkdir -p "$dst"

    # Whitelist of variables substituted into manifests. Anything outside this
    # list (e.g. legitimate shell vars inside container args) is left alone.
    local vars
    vars='$NAMESPACE $OWNER $CLUSTER $RUN_ID $RECIPE_SHORTNAME'
    vars="$vars "'$MODEL_NAME $SERVED_MODEL_NAME $MODEL_REVISION $MODEL_HF_DIR'
    vars="$vars "'$MODEL_CACHE_SUBPATH $MODEL_SOURCE_CACHE_SUBPATH $MODEL_SERVER_CACHE_MOUNT'
    vars="$vars "'$DATASET_REPO $DATASET_REVISION $DATASET_TYPE $DATASET_SUBPATH'
    vars="$vars "'$DATASET_CONFIG_PATH $DATASET_NUM_SESSIONS $DATASET_SEED $DATASET_MAX_ISL'
    vars="$vars "'$SGLANG_IMAGE $DOWNLOAD_IMAGE $BENCH_IMAGE $UTIL_IMAGE $KUBECTL_IMAGE'
    vars="$vars "'$SERVER_RUNTIME $TP_SIZE $CONTEXT_LENGTH $MEM_FRACTION_STATIC'
    vars="$vars "'$HICACHE_RATIO $HICACHE_WRITE_POLICY $HICACHE_IO_BACKEND'
    vars="$vars "'$VLLM_KV_OFFLOADING_SIZE $VLLM_KV_OFFLOADING_BACKEND $VLLM_USE_SIMPLE_KV_OFFLOAD $VLLM_CPU_OFFLOAD_GB'
    vars="$vars "'$VLLM_ALLOW_LONG_MAX_MODEL_LEN $PYTORCH_CUDA_ALLOC_CONF $VLLM_GPU_MEMORY_UTILIZATION'
    vars="$vars "'$VLLM_MAX_NUM_SEQS $VLLM_MAX_NUM_BATCHED_TOKENS'
    vars="$vars "'$ROPE_SCALING_JSON'
    vars="$vars "'$SERVER_CPU_REQUEST $SERVER_CPU_LIMIT $SERVER_MEM_REQUEST $SERVER_MEM_LIMIT'
    vars="$vars "'$AIPERF_REF $AIPERF_SCENARIO $AIPERF_EXTRA_INPUTS $CONNECTION_REUSE_STRATEGY'
    vars="$vars "'$CONCURRENCIES $CONCURRENCY_START $CONCURRENCY_MULTIPLIER $CONCURRENCY_MAX'
    vars="$vars "'$TTFT_LIMIT_MS $TPOT_LIMIT_MS $STOP_STAT'
    vars="$vars "'$BINARY_SEARCH $BINARY_SEARCH_MAX_STEPS $BINARY_SEARCH_MIN_GAP'
    vars="$vars "'$REFINE_LOWER_FRAC $REFINE_UPPER_FRAC'
    vars="$vars "'$BENCH_DURATION $CONCURRENCY_RAMP_DURATION $BENCHMARK_GRACE_PERIOD $REQUEST_TIMEOUT_SECONDS'
    vars="$vars "'$REQUEST_COUNT_MULTIPLIER $WARMUP_REQUEST_MULTIPLIER'
    vars="$vars "'$PRE_SWEEP_SMOKE_PROMPT_TOKENS $PRE_SWEEP_SMOKE_TIMEOUT_SECONDS'
    vars="$vars "'$NUM_DATASET_ENTRIES $NUM_PROFILE_RUNS $CACHE_BUST'
    vars="$vars "'$GPU_TELEMETRY $SERVER_CONTAINER $DCGM_EXPORTER_URL'
    vars="$vars "'$CLEAN_ARTIFACTS_ROOT'
    vars="$vars "'$BENCH_CPU_REQUEST $BENCH_CPU_LIMIT $BENCH_MEM_REQUEST $BENCH_MEM_LIMIT'
    vars="$vars "'$MODEL_CACHE_PVC $MODEL_CACHE_MOUNT'
    vars="$vars "'$ARTIFACTS_STORAGE_CLASS $ARTIFACTS_SIZE'
    vars="$vars "'$IMAGE_PULL_SECRET $GPU_COUNT'
    vars="$vars "'$SERVER_EXTRA_TOLERATION_KEY $SERVER_EXTRA_TOLERATION_VALUE $SERVER_EXTRA_TOLERATION_EFFECT'
    vars="$vars "'$BENCH_EXTRA_TOLERATION_KEY $BENCH_EXTRA_TOLERATION_VALUE $BENCH_EXTRA_TOLERATION_EFFECT'
    vars="$vars "'$BENCH_EXTRA_TOLERATION2_KEY $BENCH_EXTRA_TOLERATION2_VALUE $BENCH_EXTRA_TOLERATION2_EFFECT'

    local f
    for f in "$src"/*.yaml; do
        local base
        base=$(basename "$f")
        # Skip example/secret stubs — they're documentation, not applied.
        case "$base" in
            *.example.yaml) continue ;;
        esac
        envsubst "$vars" < "$f" > "$dst/$base"

        # Strip the imagePullSecrets block when no pull secret is configured.
        if [ -z "${IMAGE_PULL_SECRET:-}" ]; then
            sed -i.bak '/# >>> IMAGE_PULL_SECRET_BLOCK >>>/,/# <<< IMAGE_PULL_SECRET_BLOCK <<</d' "$dst/$base"
        fi
        # Strip the tolerations block when USE_DYNAMO_TOLERATIONS is not true.
        if [ "${USE_DYNAMO_TOLERATIONS:-true}" != "true" ]; then
            sed -i.bak '/# >>> TOLERATIONS_BLOCK >>>/,/# <<< TOLERATIONS_BLOCK <<</d' "$dst/$base"
        fi
        # Strip optional server-only cluster taint toleration unless all fields are set.
        if [ -z "${SERVER_EXTRA_TOLERATION_KEY:-}" ] || [ -z "${SERVER_EXTRA_TOLERATION_VALUE:-}" ] || [ -z "${SERVER_EXTRA_TOLERATION_EFFECT:-}" ]; then
            sed -i.bak '/# >>> SERVER_EXTRA_TOLERATION_BLOCK >>>/,/# <<< SERVER_EXTRA_TOLERATION_BLOCK <<</d' "$dst/$base"
        fi
        # Strip optional bench-only taint tolerations unless all fields are set.
        if [ -z "${BENCH_EXTRA_TOLERATION_KEY:-}" ] || [ -z "${BENCH_EXTRA_TOLERATION_VALUE:-}" ] || [ -z "${BENCH_EXTRA_TOLERATION_EFFECT:-}" ]; then
            sed -i.bak '/# >>> BENCH_EXTRA_TOLERATION_BLOCK >>>/,/# <<< BENCH_EXTRA_TOLERATION_BLOCK <<</d' "$dst/$base"
        fi
        if [ -z "${BENCH_EXTRA_TOLERATION2_KEY:-}" ] || [ -z "${BENCH_EXTRA_TOLERATION2_VALUE:-}" ] || [ -z "${BENCH_EXTRA_TOLERATION2_EFFECT:-}" ]; then
            sed -i.bak '/# >>> BENCH_EXTRA_TOLERATION2_BLOCK >>>/,/# <<< BENCH_EXTRA_TOLERATION2_BLOCK <<</d' "$dst/$base"
        fi
        rm -f "$dst/$base.bak"
    done

    log "Rendered manifests into $dst"
}

# Run kubectl scoped to the configured namespace.
llmb::kc() {
    # Pin to the profile's KUBE_CONTEXT when set (the same field preflight/observe/reclaim use), so
    # fetch_results targets the right cluster instead of the ambient context. Empty → current behavior.
    kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} -n "$NAMESPACE" "$@"
}

# ----- idempotent Teleport login --------------------------------------------
# Ensures the active kubectl context targets $CLUSTER. Respects an existing
# $KUBECONFIG (per-agent isolation pattern — important when multiple agents
# share a host: each agent sets KUBECONFIG to their own file, so tsh kube
# login here never clobbers a peer's session).
#
# - Skips the login entirely if we're already authenticated to the cluster
#   (current context matches $KUBECTX_PATTERN AND we can list pods).
# - Falls through to `tsh kube login` only when needed AND we haven't already
#   succeeded once in this orchestrator's lifetime (see breadcrumb below).
# - When kube creds went stale mid-invocation (the post-`kubectl wait` case
#   that recurs at the fetch_results.sh boundary after multi-hour benches),
#   we refuse to launch an interactive `tsh kube login` browser flow that
#   would just time out on the now-unattended terminal. Instead we exit
#   $LLMB_EXIT_AUTH_STALE with a clear recovery hint pointing at the Job
#   that almost certainly already completed.
# - Errors clearly if neither path works.
#
# Breadcrumb: on first successful auth in this orchestrator's lifetime we
# export LLMB_KUBE_CTX_OK="$KUBE_CLUSTER". Children invoked from the orchestrator
# (fetch_results.sh, report.sh) inherit that env var; if their own
# can-i probe fails AND the var matches, we're in the stale-mid-run case.
#
# Call after llmb::load_env so $CLUSTER, $KUBECTX_PATTERN, $NAMESPACE are set.
llmb::ensure_kube_context() {
    : "${KUBECONFIG:=$HOME/.kube/config}"
    export KUBECONFIG

    # Prefer a profile-pinned KUBE_CONTEXT. Verify it before falling back to a cluster login.
    if [ -n "${KUBE_CONTEXT:-}" ]; then
        if kubectl --context "$KUBE_CONTEXT" auth can-i list pods --namespace "$NAMESPACE" > /dev/null 2>&1; then
            log "using pinned KUBE_CONTEXT '$KUBE_CONTEXT' (KUBECONFIG: $KUBECONFIG)"
            export LLMB_KUBE_CTX_OK="$KUBE_CONTEXT"
            return 0
        fi
        err "profile pins KUBE_CONTEXT='$KUBE_CONTEXT' but it is not usable (auth can-i failed)."
        err "  Fix: kubectl config get-contexts   (or re-auth: tsh kube login <cluster>), then retry."
        return 1
    fi

    local login_cluster="${KUBE_CLUSTER:-$CLUSTER}"
    local want_pattern="${KUBECTX_PATTERN:-$login_cluster}"
    local current_ctx
    current_ctx="$(kubectl config current-context 2> /dev/null || true)"

    if [ -n "$current_ctx" ] && echo "$current_ctx" | grep -q "$want_pattern"; then
        if kubectl auth can-i list pods --namespace "$NAMESPACE" > /dev/null 2>&1; then
            log "kubectl already authenticated to '$want_pattern' (context: $current_ctx, KUBECONFIG: $KUBECONFIG)"
            export LLMB_KUBE_CTX_OK="$login_cluster"
            return 0
        fi
    fi

    # Mid-run staleness path: the orchestrator that spawned us already
    # auth'd successfully against this same cluster. Don't pop an interactive
    # browser SSO flow that nobody is watching; bail with a distinct code and
    # tell the operator exactly how to recover.
    if [ "${LLMB_KUBE_CTX_OK:-}" = "$login_cluster" ]; then
        err "Kubernetes creds went stale mid-run for cluster '$login_cluster' (profile '$CLUSTER', KUBECONFIG=$KUBECONFIG)."
        err "The bench Job most likely already completed — artifacts are safe on the PVC."
        err "Refusing to launch interactive 'tsh kube login' (no human present after the long kubectl wait)."
        err "To recover, re-auth then re-run fetch + report manually:"
        err "    tsh kube login ${KUBE_PROXY:+--proxy=$KUBE_PROXY }$login_cluster"
        err "    scripts/fetch_results.sh ${RUN_ID:-<run-id>}"
        err "    scripts/report.sh ${RUN_ID:-<run-id>}"
        exit "${LLMB_EXIT_AUTH_STALE:-2}"
    fi

    if command -v tsh > /dev/null 2>&1; then
        log "Logging into '$login_cluster' via Teleport for profile '$CLUSTER' (KUBECONFIG=$KUBECONFIG) ..."
        if [ -n "${KUBE_PROXY:-}" ]; then
            tsh kube login --proxy="$KUBE_PROXY" "$login_cluster"
        else
            tsh kube login "$login_cluster"
        fi
        # Don't export LLMB_KUBE_CTX_OK on a happy-path exit code alone:
        # `tsh kube login` can return 0 even when the kubectx ends up
        # pointing at a cluster the caller can't actually list pods in
        # (wrong role, expired SSO under the hood, etc.). Re-probe with
        # the same can-i check we use up top, and only declare success if
        # it passes. Otherwise downstream scripts inherit the breadcrumb
        # and bail with LLMB_EXIT_AUTH_STALE instead of attempting a
        # second interactive login they can't complete unattended.
        if kubectl auth can-i list pods --namespace "$NAMESPACE" > /dev/null 2>&1; then
            export LLMB_KUBE_CTX_OK="$login_cluster"
        else
            err "tsh kube login returned 0 but kubectl can-i still fails for namespace '$NAMESPACE'."
            err "Not exporting LLMB_KUBE_CTX_OK; investigate role / namespace membership."
            return 1
        fi
    else
        die "kubectl context for '$login_cluster' not active and 'tsh' not on PATH. \
Either run 'tsh kube login ${KUBE_PROXY:+--proxy=$KUBE_PROXY }$login_cluster' manually or set KUBECONFIG to a file with the right context."
    fi
}
