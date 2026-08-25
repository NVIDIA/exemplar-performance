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

# wait_server_ready.sh <cell-dir> <cluster-profile> [flags] — poll a cell's server to READY, FAIL FAST on crash-loop.
#
# THE GAP THIS CLOSES (GPU-lifecycle resilience):
#   run.sh's wait-ready step used a blind `kubectl rollout status --timeout=${startup_timeout_s}s`. A server that
#   never becomes ready — e.g. a CrashLoopBackOff (NCCL init error, bad weights) restarting every ~30s — is NOT an
#   orphan the coord Job's activeDeadlineSeconds/ownerRef GC can reap (the coord never starts; the server was
#   never adopted), and the in-cluster governor's fail-safe orphan-sweep HOLDS an unowned server that has no
#   terminal/active bench Job for its cell. So the blind wait would block the FULL startup budget — 14400s = 4h
#   for the 550B cells — holding a whole GPU node doing zero work, until a human `kubectl delete`d it.
#
# This replaces the blind wait with a POLL that also watches the server pods and aborts EARLY when the server is
# clearly not going to become ready, so run.sh's EXIT trap frees the GPU (scale server -> 0) in minutes, not hours:
#   * restartCount >= K (default 5) on any server pod                          -> crash-loop, abort now
#   * CrashLoopBackOff sustained >= M minutes (default 5) with 0 ready replicas -> abort now
#   * neither trips, and startup_timeout_s elapses                             -> timeout, abort (unchanged ceiling)
#
# HAPPY-PATH SAFE (why this can't trip a slow-but-legitimate 550B cold load): both fast-fail predicates are
# RESTART/CrashLoopBackOff based, never wall-clock. A server still loading a 550B checkpoint from network storage
# sits Running with restartCount=0 while its long startupProbe holds — it has not crashed, so K and the
# CrashLoopBackOff window are both untriggered. Only a container that actually keeps DYING trips the fail-fast.
#
# Lane-agnostic: run.sh calls this for every lane at the shared wait-ready step. Exit 0 =
# ready; non-zero = abort with a one-line reason on stderr (run.sh's trap turns that into recovery + teardown).
#
#   scripts/wait_server_ready.sh <cell> <profile> [--timeout S] [--max-restarts K] [--window-min M] [--poll S]
#
# KUBECTL override (offline-testable with a fake shim, mirrors adopt_server.sh / the governor); KUBE_CONTEXT is
# read from the profile so the poll targets the SAME cluster the launch scripts act on.
set -euo pipefail

CELL=""
PROFILE=""
TIMEOUT_S=0                                 # 0 => take the cell's serving.startup_timeout_s
MAX_RESTARTS="${CRASHLOOP_MAX_RESTARTS:-5}" # K: restartCount at/above which a server pod is deemed crash-looping
WINDOW_MIN="${CRASHLOOP_WINDOW_MIN:-5}"     # M: minutes of sustained CrashLoopBackOff (0 ready) before aborting
POLL_S="${WAIT_POLL_S:-15}"
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --timeout)
            TIMEOUT_S="${2:?}"
            shift 2
            ;;
        --max-restarts)
            MAX_RESTARTS="${2:?}"
            shift 2
            ;;
        --window-min)
            WINDOW_MIN="${2:?}"
            shift 2
            ;;
        --poll)
            POLL_S="${2:?}"
            shift 2
            ;;
        -h | --help)
            sed -n '2,30p' "$0"
            exit 0
            ;;
        *)
            POSITIONAL+=("$1")
            shift
            ;;
    esac
done
set -- "${POSITIONAL[@]:-}"
CELL="${1:?usage: wait_server_ready.sh <cell-dir> <cluster-profile> [flags]}"
PROFILE="${2:?need a cluster-profile name}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "wait_server_ready: no profile at $ENVF" >&2
    exit 1
}
[ -f "$CELL/recipe.yaml" ] || {
    echo "wait_server_ready: no recipe.yaml in $CELL" >&2
    exit 1
}
set -a
. "$ENVF"
set +a
kc() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
# Cert/auth-expiry resilience: on a long 550B cold load the kube cert can expire mid-wait; without a heal the
# readiness probe would then fail forever and the loop would FALSE-timeout on a server that is actually fine.
# llmb::heal_auth re-auths (via the profile hook) when a short auth probe shows expired creds. Non-fatal.
. "$ROOT/scripts/_kubectl_resilient.sh"

NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
[ -n "$NAME" ] || {
    echo "wait_server_ready: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}
SERVER_DEPLOY="${NAME}-server"

if [ "$TIMEOUT_S" -le 0 ] 2> /dev/null; then
    TIMEOUT_S="$(
        python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
print(int((r.get("serving") or {}).get("startup_timeout_s") or 900))
PY
    )"
fi

# The classifier sub-modes live in idle_guard.sh (single home for pod-json parsing); shell into them as filters.
IG="$ROOT/scripts/idle_guard.sh"
max_restarts() { bash "$IG" --pod-max-restarts; }
in_backoff() { bash "$IG" --pod-in-backoff; }

# Ready check + server-pod json for whichever server shape the cell uses:
#   * agg / common  : Deployment <name>-server exists -> ready when readyReplicas>=1; pods -l app=<name>-server.
#   * sglang-disagg : no single deployment -> ready when a server-role pod is Ready; pods -l recipe=<name>.
if kc -n "$NAMESPACE" get deployment "$SERVER_DEPLOY" > /dev/null 2>&1; then
    POD_SELECTOR="app=${SERVER_DEPLOY}"
    PROBE_POD_SELECTOR="$POD_SELECTOR"
    MODEL_PROBE_URL="http://localhost:8000/v1/models"
    is_ready() {
        local rr
        rr="$(kc -n "$NAMESPACE" get deployment "$SERVER_DEPLOY" -o jsonpath='{.status.readyReplicas}' 2> /dev/null)"
        [ "${rr:-0}" -ge 1 ] 2> /dev/null
    }
else
    POD_SELECTOR="llmb.nvidia.com/recipe=${NAME}"
    # The recipe label also matches prefill/decode workers. Select the frontend
    # explicitly for the model probe so pod ordering cannot pair a worker pod
    # with the frontend container name.
    PROBE_POD_SELECTOR="app=${NAME}-frontend"
    PROBE_CONTAINER="frontend"
    # Disaggregated SGLang workers listen on 18081; the OpenAI endpoint is the
    # recipe frontend Service on port 8000. Probe the same Service used by the
    # benchmark from the explicitly selected frontend pod.
    MODEL_PROBE_URL="http://${NAME}-server:8000/v1/models"
    is_ready() {
        # any server-role pod reporting Ready=True
        kc -n "$NAMESPACE" get pods -l "$POD_SELECTOR" \
            -o jsonpath='{.items[*].status.conditions[?(@.type=="Ready")].status}' 2> /dev/null | grep -q True
    }
fi

# ── MODEL-REGISTRATION GATE (DEFECT #10) ─────────────────────────────────────────────────────────────
# k8s readiness is NOT sufficient. The readiness probe is `/health`, which an OpenAI-compatible server
# answers 200 as soon as its HTTP listener is up — BEFORE the engine has finished loading weights and
# REGISTERED the model. A run hit this live: wait-ready "succeeded" in 19s, then the bench fataled with
#   Listing models at .../v1/models -> {"object":"list","data":[]}
#   FATAL: AIPerf model 'qwen3-0-6b' is not served; available=[]
# An immediate retry passed, so it is a startup RACE, not a broken deployment — which is worse, because it
# fails intermittently and burns a whole deploy+bench cycle (and GPU time) each time. This is the same class
# as serving/dynamo-disagg/TROUBLESHOOTING.md §6 ("pod Ready but /v1/* 404 — premature readiness"), which the
# disagg lane documented but no lane actually guarded.
# So: after k8s-ready, additionally poll GET /v1/models until serving.served_model appears in data[].
# Probed from INSIDE a server pod (the endpoint is cluster-internal); curl if present, else python3.
SERVED_MODEL="$(
    python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
print(((r.get("serving") or {}).get("served_model") or "").strip())
PY
)"
# Which container to exec into. SAME convention AIPerf already uses for telemetry scraping:
# serving.telemetry_container when the serving container isn't named after the engine (Dynamo+vLLM names it
# 'dynamo-vllm', not 'vllm'), else envelope.engine. Avoids exec'ing into a sidecar (etcd/nats).
SERVER_CONTAINER="$(
    python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
sv, env = (r.get("serving") or {}), (r.get("envelope") or {})
print((sv.get("telemetry_container") or env.get("engine") or "").strip())
PY
)"
[ -n "${PROBE_CONTAINER:-}" ] && SERVER_CONTAINER="$PROBE_CONTAINER"

# Probe the server's /v1/models from INSIDE a server pod. Echoes the registered model ids (one per line).
# The EXIT CODE distinguishes three outcomes that must never be conflated:
#   0 = probe SUCCEEDED (stdout is the — possibly empty — id list)
#   2 = exec is FORBIDDEN (RBAC): we cannot verify at all, on any poll. Caller degrades to k8s-readiness.
#   3 = UNKNOWN/transient (no Running pod yet, exec blipped, no curl+python3, malformed JSON): retry.
# Conflating 2 or 3 with "model absent" would turn a permissions problem into a silent 900s hang; conflating
# 3 with 2 would permanently disable the gate because of one transient blip.
probe_models() {
    local pod out rc errf
    pod="$(kc -n "$NAMESPACE" get pods -l "$PROBE_POD_SELECTOR" \
        -o jsonpath='{.items[?(@.status.phase=="Running")].metadata.name}' 2> /dev/null | tr ' ' '\n' | head -1)"
    [ -n "$pod" ] || return 3 # nothing Running yet — unknown, not "absent"
    errf="$(mktemp)"
    rc=0
    out="$(kc -n "$NAMESPACE" exec "$pod" ${SERVER_CONTAINER:+-c "$SERVER_CONTAINER"} -- env LLMB_MODEL_PROBE_URL="$MODEL_PROBE_URL" sh -c '
    if command -v curl >/dev/null 2>&1; then curl -s --max-time 10 "$LLMB_MODEL_PROBE_URL"
    elif command -v python3 >/dev/null 2>&1; then python3 -c "import os,urllib.request;print(urllib.request.urlopen(os.environ[\"LLMB_MODEL_PROBE_URL\"],timeout=10).read().decode())"
    else exit 97
    fi' 2> "$errf")" || rc=$?
    if [ "$rc" -ne 0 ]; then
        # RBAC denial is a PERMISSION fact, not a probe failure — match the API server's own wording.
        if grep -qiE 'forbidden|cannot exec|not allowed|unauthorized|RBAC' "$errf" 2> /dev/null; then
            rm -f "$errf"
            return 2
        fi
        rm -f "$errf"
        return 3 # transient exec failure / no probe tool in the image
    fi
    rm -f "$errf"
    # A malformed/empty body is UNKNOWN (3), never "no models" — python exits 3 on a parse failure.
    printf '%s' "$out" | python3 -c 'import json,sys
try: d=json.load(sys.stdin)
except Exception: sys.exit(3)
for m in (d.get("data") or []):
    i=m.get("id") or ""
    if i: print(i)' 2> /dev/null
}

# Readiness verdict for the model-registration gate. Exit codes mirror probe_models:
#   0 = the cell's served_model IS registered      -> ready
#   1 = probe worked, model NOT there              -> the real race; KEEP WAITING
#   2 = exec forbidden                             -> cannot verify; caller warns once + degrades
#   3 = unknown/transient                          -> retry quietly
# With no served_model declared we cannot name a target, so any registered model counts (still strictly
# better than /health alone).
model_registered() {
    local ids rc
    rc=0
    ids="$(probe_models)" || rc=$?
    [ "$rc" -eq 0 ] || return "$rc"
    [ -n "$ids" ] || return 1
    if [ -n "$SERVED_MODEL" ]; then
        printf '%s\n' "$ids" | grep -qxF "$SERVED_MODEL" && return 0 || return 1
    fi
    return 0
}

log() { echo "[wait-ready $(date -u +%H:%M:%S)] $*"; }

log "watching $SERVER_DEPLOY (ns=$NAMESPACE) — ready-timeout=${TIMEOUT_S}s  fail-fast: restarts>=${MAX_RESTARTS} OR CrashLoopBackOff>=${WINDOW_MIN}m"
[ -n "$SERVED_MODEL" ] && log "readiness = pod Ready AND '$SERVED_MODEL' registered in /v1/models (not /health alone)"

START="$(date -u +%s)"
BACKOFF_SINCE=0
SAW_READY=0                                 # have we observed pod-Ready-but-model-not-registered? (drives the timeout message)
EXEC_WARNED=0                               # sticky: the pods/exec-forbidden warning is emitted at most ONCE
FORBIDDEN_STREAK=0                          # consecutive 'exec forbidden' probes (reset by any other outcome)
FORBIDDEN_CONFIRM="${FORBIDDEN_CONFIRM:-2}" # how many CONSECUTIVE forbidden probes before we degrade the gate
while true; do
    # Reactive credential heal (cheap): if the kube cert expired mid-cold-load, refresh it via the profile hook
    # so the readiness probe below can succeed again instead of silently failing until the ready-timeout.
    llmb::heal_auth
    if is_ready; then
        # k8s says Ready — now require the model to be REGISTERED before declaring readiness (see the gate above).
        MRC=0
        model_registered || MRC=$?
        case "$MRC" in
            0)
                log "$SERVER_DEPLOY is ready${SERVED_MODEL:+ (model '$SERVED_MODEL' registered)}"
                exit 0
                ;;
            2)
                # RBAC forbids pods/exec, so the gate cannot run here. Degrading is right (waiting out the full
                # timeout on a permissions problem is a mysterious hang), but it costs the verification gate for the
                # WHOLE run — so require TWO CONSECUTIVE forbidden responses first. The asymmetry is lopsided: one
                # extra poll costs seconds, while wrongly degrading on a single spurious authorization blip silently
                # disables the gate and lets the run reach the exact bench-time failure the gate exists to prevent.
                # Same principle applied elsewhere today: a SINGLE failed probe is never a definitive condition.
                FORBIDDEN_STREAK=$((FORBIDDEN_STREAK + 1))
                if [ "$FORBIDDEN_STREAK" -lt "$FORBIDDEN_CONFIRM" ]; then
                    log "pods/exec appears forbidden — re-probing once to confirm before degrading the readiness gate"
                else
                    # Confirmed. Warn ONCE, loudly and specifically, then accept k8s-readiness.
                    if [ "$EXEC_WARNED" -eq 0 ]; then
                        EXEC_WARNED=1
                        echo "wait_server_ready: ⚠ cannot verify model registration: pods/exec is forbidden in namespace '$NAMESPACE' (confirmed on ${FORBIDDEN_STREAK} consecutive probes)." >&2
                        echo "wait_server_ready:   Falling back to k8s-readiness ONLY. A premature-readiness failure will now surface" >&2
                        echo "wait_server_ready:   later, at bench time, as \"model ${SERVED_MODEL:+'$SERVED_MODEL' }is not served; available=[]\"." >&2
                        echo "wait_server_ready:   To restore the gate, grant pods/exec in '$NAMESPACE' (verify: kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE auth can-i create pods/exec)." >&2
                    fi
                    log "$SERVER_DEPLOY is ready (k8s-readiness only — model registration UNVERIFIED, pods/exec forbidden)"
                    exit 0
                fi
                ;;
            1)
                FORBIDDEN_STREAK=0 # a successful probe breaks the streak — "consecutive" must mean consecutive
                if [ "$SAW_READY" -eq 0 ]; then
                    SAW_READY=1
                    log "$SERVER_DEPLOY pod is Ready but ${SERVED_MODEL:-the model} is NOT registered yet — still loading; waiting for /v1/models"
                fi
                ;;
            *)
                FORBIDDEN_STREAK=0 # 3 = unknown/transient — retry quietly, and it also breaks the streak
                :
                ;;
        esac
    fi

    # Snapshot the server pods once per poll and run the two fail-fast predicates off it.
    PODS_JSON="$(kc -n "$NAMESPACE" get pods -l "$POD_SELECTOR" -o json 2> /dev/null || true)"
    if [ -n "$PODS_JSON" ]; then
        MAXR="$(printf '%s' "$PODS_JSON" | max_restarts)"
        if [ "${MAXR:-0}" -ge "$MAX_RESTARTS" ] 2> /dev/null; then
            echo "wait_server_ready: ABORT — $SERVER_DEPLOY pod restarted ${MAXR}× (>= ${MAX_RESTARTS}); crash-looping, will not become ready. Recent pod events:" >&2
            kc -n "$NAMESPACE" describe pods -l "$POD_SELECTOR" 2> /dev/null | sed -n '/Events:/,$p' | tail -12 | sed 's/^/    /' >&2 || true
            exit 1
        fi
        if [ "$(printf '%s' "$PODS_JSON" | in_backoff)" = 1 ]; then
            NOW="$(date -u +%s)"
            [ "$BACKOFF_SINCE" -eq 0 ] && BACKOFF_SINCE="$NOW"
            FOR=$((NOW - BACKOFF_SINCE))
            if [ "$FOR" -ge $((WINDOW_MIN * 60)) ]; then
                echo "wait_server_ready: ABORT — $SERVER_DEPLOY CrashLoopBackOff sustained $((FOR / 60))m (>= ${WINDOW_MIN}m) with 0 ready replicas. Recent pod events:" >&2
                kc -n "$NAMESPACE" describe pods -l "$POD_SELECTOR" 2> /dev/null | sed -n '/Events:/,$p' | tail -12 | sed 's/^/    /' >&2 || true
                exit 1
            fi
        else
            BACKOFF_SINCE=0 # recovered out of backoff (or transiently unqueryable) — reset the sustained-window timer
        fi
    fi

    ELAPSED=$(($(date -u +%s) - START))
    if [ "$ELAPSED" -ge "$TIMEOUT_S" ]; then
        if [ "$SAW_READY" -eq 1 ]; then
            echo "wait_server_ready: server $SERVER_DEPLOY was healthy but never registered ${SERVED_MODEL:-a model} within ${TIMEOUT_S}s." >&2
            echo "wait_server_ready:   The pod passed its /health probe, so the HTTP listener is up, but /v1/models never listed it —" >&2
            echo "wait_server_ready:   the engine is still loading weights or failed during load. Check the WORKER logs:" >&2
            echo "wait_server_ready:     kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE logs -l $POD_SELECTOR --tail=100" >&2
            echo "wait_server_ready:   and what it does serve: kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE exec deploy/$SERVER_DEPLOY -- curl -s localhost:8000/v1/models" >&2
        else
            echo "wait_server_ready: server $SERVER_DEPLOY did not become ready within ${TIMEOUT_S}s" >&2
            echo "wait_server_ready:   Check: kubectl ${KUBE_CONTEXT:+--context $KUBE_CONTEXT }-n $NAMESPACE get pods -l $POD_SELECTOR" >&2
        fi
        exit 1
    fi
    sleep "$POLL_S"
done
