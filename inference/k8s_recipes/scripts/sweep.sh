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

# sweep.sh <cell-dir> <cluster-profile> [run-id] [--rungs "<c1 c2 ...>"]
#
# Launches the aiperf concurrency sweep for a cell on the NEW declarative path: ensures the per-cell
# artifacts PVC, resolves ONLY the cluster/runtime ${VARS} in the committed rendered/bench-job.yaml
# (a STRICT whitelist — the runner script's own ${BASH} refs must survive envsubst), `kubectl apply`s
# it, and follows the logs.
#
# --rungs OVERRIDES the concurrency list for this launch only (the committed recipe is untouched). Use it
# for a fast single-rung FUNCTIONAL check that the whole path works before committing a full overnight sweep:
#     scripts/sweep.sh <cell> <profile> smoke1 --rungs "128"
# A set CONCURRENCIES runs as a fixed no-refine list, so one value = exactly one rung. Then run the full
# recipe sweep (no --rungs) overnight. A --rungs run is a smoke, not a publishable result.
#
# The server must already be up and Ready first:  scripts/deploy.sh <cell> <profile>
# (This is the Phase-B counterpart to deploy.sh; the legacy scripts/run.sh drives the old manifest web.)
set -euo pipefail

RUNGS=""
ARGS=()
while [ $# -gt 0 ]; do
    case "$1" in
        --rungs)
            RUNGS="${2:?--rungs needs a value, e.g. --rungs \"128\"}"
            shift 2
            ;;
        --rungs=*)
            RUNGS="${1#*=}"
            shift
            ;;
        *)
            ARGS+=("$1")
            shift
            ;;
    esac
done
set -- "${ARGS[@]}"

CELL="${1:?usage: sweep.sh <cell-dir> <cluster-profile> [run-id] [--rungs \"<c1 c2 ...>\"]}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
BENCH="$CELL/rendered/bench-job.yaml"
[ -f "$ENVF" ] || {
    echo "sweep.sh: no profile at $ENVF" >&2
    exit 1
}
[ -f "$BENCH" ] || {
    echo "sweep.sh: no $BENCH — run scripts/render.sh $CELL first" >&2
    exit 1
}
command -v envsubst > /dev/null || {
    echo "sweep.sh: envsubst not found (install gettext)" >&2
    exit 1
}

set -a
. "$ENVF"
set +a # export the profile's cluster vars
# Resolve THE model-cache claim for THIS cell (scripts/_model_cache.sh -> install.py
# --resolve-cache -> install.resolve_cache_claim). The rendered manifests below mount
# ${MODEL_CACHE_PVC}, so this MUST be the same string install downloaded into; it is, because it
# is the same function. Fail-closed: an unresolvable claim aborts here, not at model-load time.
. "$ROOT/scripts/_model_cache.sh"
llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1
# KUBECTL override → offline-testable shim (mirrors run_owner.sh / wait_server_ready.sh). The resilient helper
# below reuses this same binding for its cert/auth-expiry-tolerant reads.
kc() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
# Use resilient Kubernetes reads so transient authentication or network errors do not become terminal states.
. "$ROOT/scripts/_kubectl_resilient.sh"
# run-id: label-safe + sized so the Job name <cell>-bench-<id> fits ≤63 (scripts/run_id.py). A hand-passed
# run-id ($3) is routed through --label so it too gets the compact UTC stamp prefix (rp3 → 260728t0312-rp3),
# flows into the results path + Job name + labels, and is shrunk to fit ≤63 (already-dated ids pass through).
# Both paths fall back to a plain UTC stamp / the raw arg if the helper is unavailable.
if [ -n "${3:-}" ]; then
    if [ "${LLMB_RUN_ID_FINAL:-0}" = 1 ]; then export RUN_ID="$3"; else
        export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit bench --label "$3" 2> /dev/null || echo "$3")"
    fi
else
    export RUN_ID="$(python3 "$ROOT/scripts/run_id.py" "$CELL" --fit bench 2> /dev/null || date -u +%Y%m%d-%H%M%S)"
fi
: "${OWNER:=}" "${CACHE_BUST:=}" "${DCGM_EXPORTER_URL:=}" # optional -> empty, not unset
: "${BENCH_NODE_SELECTOR:=}" "${BENCH_CPU_REQUEST:=16}"   # bench pod placement/size (optional profile overrides)
export OWNER CACHE_BUST DCGM_EXPORTER_URL BENCH_NODE_SELECTOR BENCH_CPU_REQUEST
if [ -z "$BENCH_NODE_SELECTOR" ]; then
    echo "sweep: WARN: BENCH_NODE_SELECTOR is empty — bench pod may schedule on GPU nodes and consume GPU resources it doesn't use"
    echo "sweep:       Set BENCH_NODE_SELECTOR in cluster-profiles/${PROFILE}.env, e.g.: BENCH_NODE_SELECTOR=\"node-role.kubernetes.io/worker: true\""
fi
NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)" # envelope.name
[ -n "$NAME" ] || {
    echo "sweep.sh: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}

# Ensure the per-cell artifacts PVC (idempotent). Storage class/size come from the profile if set.
if ! kc -n "$NAMESPACE" get pvc "${NAME}-artifacts" > /dev/null 2>&1; then
    echo "sweep: creating artifacts PVC ${NAME}-artifacts"
    {
        echo "apiVersion: v1"
        echo "kind: PersistentVolumeClaim"
        echo "metadata: { name: ${NAME}-artifacts, namespace: ${NAMESPACE} }"
        echo "spec:"
        echo "  accessModes: [ReadWriteOnce]"
        [ -n "${ARTIFACTS_STORAGE_CLASS:-}" ] && echo "  storageClassName: ${ARTIFACTS_STORAGE_CLASS}"
        echo "  resources: { requests: { storage: ${ARTIFACTS_SIZE:-20Gi} } }"
    } | kc apply -f -
fi

# Collision guard: refuse to launch if another bench Job for this cell is already active.
# Two concurrent aiperf loads against the same server invalidate both results (shared KV cache,
# split throughput, inflated latency). The check is self-cleaning — no lock objects to manage.
# Kubernetes does not support `status.active=1` as a Job field selector, so filter active Jobs with awk.
# EXCLUDE the run-owner Job: run.sh creates a per-run run-owner (`<cell>-runowner/-ro-<hash>`) that ALSO
# carries llmb.nvidia.com/cell=<cell> and stays active for the whole run — so a bare cell-label match would
# flag it as a "concurrent bench" and block this run's OWN sweep. The run-owner is the only cell-labelled
# Job with app.kubernetes.io/component=run-owner; `component!=run-owner` also matches bench Jobs (which have
# no component label), so it drops ONLY the run-owner while still catching a genuine second bench Job.
_active=$(kc -n "$NAMESPACE" get jobs \
    -l "llmb.nvidia.com/cell=${NAME},app.kubernetes.io/component!=run-owner" \
    -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.status.active}{"\n"}{end}' 2> /dev/null \
    | awk -F'\t' '($2+0)>0 {printf "%s ", $1}' || true)
if [ -n "$_active" ]; then
    echo "sweep: BLOCKED — an active bench job for '$NAME' is already running: $_active" >&2
    echo "sweep:   Two concurrent sweeps against the same server corrupt both results." >&2
    echo "sweep:   Wait for it to finish, or follow its logs:" >&2
    echo "sweep:     kubectl -n $NAMESPACE logs -f job/$_active" >&2
    exit 1
fi

# Capture BEFORE the Job is submitted. archive_run.py consumes this immutable receipt instead of recomputing
# recipe_hash while archiving; a later recipe edit (or a re-archive) cannot rewrite run provenance.
python3 "$ROOT/scripts/launch_attestation.py" "$CELL" "$RUN_ID" \
    --out "$ROOT/results/$RUN_ID/launch_attestation.json"

# STRICT whitelist: ONLY cluster/runtime vars. Everything else (${ARTIFACTS_ROOT}, ${SERVER_URL},
# ${STOP_STAT}, ${host}, ...) is a runtime bash ref inside the runner and MUST survive envsubst.
WL='$NAMESPACE $RUN_ID $OWNER $IMAGE_PULL_SECRET $MODEL_CACHE_PVC $DCGM_EXPORTER_URL $CACHE_BUST $BENCH_NODE_SELECTOR $BENCH_CPU_REQUEST'
echo "sweep: apply bench Job  (cell=$NAME  run-id=$RUN_ID  ns=$NAMESPACE)"
if [ -n "$RUNGS" ]; then
    # Rungs MUST be space-separated integers. This also guards the sed below: a stray sed-special char
    # (esp. '&', which sed expands to the whole match) would silently corrupt the CONCURRENCIES field.
    case "$RUNGS" in
        *[!0-9\ ]*)
            echo "sweep.sh: --rungs must be space-separated integers (got: '$RUNGS')" >&2
            exit 1
            ;;
    esac
    echo "sweep: ⚡ RUNGS OVERRIDE -> \"$RUNGS\" (functional check; committed recipe sweep unchanged, NOT a publishable result)"
    echo "sweep:    → to make this rung set PERMANENT: edit bench.sweep_concurrency in the recipe + scripts/render.sh (this override is per-launch only)"
    envsubst "$WL" < "$BENCH" \
        | sed "s/\(name: CONCURRENCIES, value: \)\"[^\"]*\"/\1\"$RUNGS\"/" \
        | kc apply -f -
else
    envsubst "$WL" < "$BENCH" | kc apply -f -
fi

# ── ORPHAN-PREVENTION: bind the GPU lifetime to an in-cluster owner (pure-k8s GC) ──────────────────
JOB_NAME="${NAME}-bench-${RUN_ID}"
if [ -n "${RUN_OWNER_UID:-}" ]; then
    # PRIMARY path (run.sh created a per-run run-owner Job): the server is ALREADY owned by the run-owner
    # from birth (deploy.sh's merge_run_owner.py). Make THIS bench Job an ownerReference child of the SAME
    # run-owner too — so when the run-owner terminates (its watcher exits on this Job reaching Complete/Failed,
    # or its activeDeadlineSeconds fires) native GC cascade-deletes server + bench together, promptly and
    # disconnect-proof. Do NOT call adopt_server here: that stamps server->benchJob and would clobber the
    # server's run-owner ownerReference (patch --type merge replaces the array). Runtime patch → hash-neutral.
    bash "$ROOT/scripts/run_owner.sh" adopt-job "$NAMESPACE" "$JOB_NAME" "$RUN_OWNER_NAME" "$RUN_OWNER_UID" || true
else
    # BACKSTOP path (sweep.sh invoked standalone, no run-owner): fall back to the legacy server->benchJob
    # binding so a hard-killed orchestrator still frees the GPU when this deadline-bounded Job terminates.
    JOB_UID=""
    for _try in 1 2 3; do
        JOB_UID="$(kc -n "$NAMESPACE" get job "$JOB_NAME" -o jsonpath='{.metadata.uid}' 2> /dev/null || true)"
        [ -n "$JOB_UID" ] && break
        sleep "$_try"
    done
    bash "$ROOT/scripts/adopt_server.sh" "$NAMESPACE" "$NAME" "$JOB_NAME" "$JOB_UID" || true
fi

echo "sweep: following job/${JOB_NAME}  (Ctrl-C detaches; the Job keeps running)"
POD_READY_TIMEOUT="${BENCH_POD_READY_TIMEOUT:-900s}"
if ! kc -n "$NAMESPACE" wait --for=condition=ready pod -l "llmb.nvidia.com/run-id=${RUN_ID}" --timeout="$POD_READY_TIMEOUT"; then
    echo "sweep: BLOCKED - bench pod for job/${JOB_NAME} did not become Ready within ${POD_READY_TIMEOUT}" >&2
    echo "sweep: pod status/events:" >&2
    kc -n "$NAMESPACE" describe pod -l "llmb.nvidia.com/run-id=${RUN_ID}" >&2 || true
    exit 1
fi

set +e
# Pipe through rung_progress.py — passes all log lines through AND injects compact ✓/✗ summary
# lines at rung boundaries so concurrency sweep progress is visible without wading through aiperf output.
# --annotate-job publishes the completed-rung count onto the bench Job after EACH rung (llmb.nvidia.com/
# completed-rungs + /total-rungs) so `llmb-k8s fleet` advances the SWEEP dot-bar live; it's a plain kubectl
# annotate with the operator's own kubeconfig (no serving-template RBAC change). Errors in rung_progress.py
# (or a detached follower) are silently ignored so nothing ever breaks a running benchmark.
kc -n "$NAMESPACE" logs -f "job/${JOB_NAME}" \
    | KUBE_CONTEXT="${KUBE_CONTEXT:-}" KUBECTL="${KUBECTL:-kubectl}" \
        python3 "$ROOT/scripts/rung_progress.py" --annotate-job "$JOB_NAME" --namespace "$NAMESPACE" 2> /dev/null || true
LOG_RC=${PIPESTATUS[0]:-$?}
set -e

# Wait for a genuine Job terminal state while tolerating transient authentication and network errors.
llmb::follow_job_to_terminal "$JOB_NAME" "$NAMESPACE" "${SWEEP_JOB_POLL_S:-10}"
_jt=$?
case "$_jt" in
    0) exit 0 ;;
    1)
        echo "sweep: job/${JOB_NAME} failed (logs exit=${LOG_RC})" >&2
        exit 1
        ;;
    2)
        echo "sweep: job/${JOB_NAME} vanished after running (deleted/TTL-GC'd) — results are on the artifacts PVC; harvest with llmb-k8s collect" >&2
        exit 1
        ;;
esac
