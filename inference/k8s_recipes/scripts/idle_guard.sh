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

# idle_guard.sh <cell-dir> <cluster-profile> [run-id] [flags]
# Monitor benchmark progress and optionally stop a hung or unschedulable Job. When work reaches a
# terminal state, scale this cell.s server to zero unless --keep-server is set.
# Flags: --idle-timeout MIN, --poll SEC, --kill-hung, --keep-server, --grace SEC,
#        --max-wall SEC, --stuck-timeout MIN
set -euo pipefail

# ── testable sub-mode: sum vLLM generation-token counters from a /metrics blob on stdin (no cluster needed) ──
# Prometheus lines look like:  vllm:generation_tokens_total{model_name="..."} 12345.0   (or bare, no labels).
# Counters only increase, so the sum flatlining across polls == no generation progress.
if [ "${1:-}" = "--sum-tokens" ]; then
    awk '/^vllm:generation_tokens(_total)?[ {]/ { s += $NF + 0 } END { printf "%.0f\n", s }'
    exit 0
fi

# ── testable sub-mode: is a server pod CrashLoopBackOff? reads `kubectl get pod -o json` on stdin (no cluster) ──
# Prints 1 if any container is waiting in CrashLoopBackOff (or has restarted >= 3 times), else 0. The health
# signal the main loop acts on. Pure grep so it needs no jq/python.
if [ "${1:-}" = "--pod-crashlooping" ]; then
    _blob="$(cat)"
    if printf '%s' "$_blob" | grep -qE '"reason"[[:space:]]*:[[:space:]]*"CrashLoopBackOff"' \
        || printf '%s' "$_blob" | grep -qE '"restartCount"[[:space:]]*:[[:space:]]*([3-9]|[1-9][0-9]+)'; then
        echo 1
    else
        echo 0
    fi
    exit 0
fi

# ── testable sub-mode: max container restartCount across a `kubectl get pod(s) -o json` blob on stdin ──────
# Prints the LARGEST restartCount over every container in the blob (0 if none). Used by wait_server_ready.sh's
# fail-fast: a server pod whose restartCount climbs past K is crash-looping (a genuinely slow FIRST cold load
# never restarts — the container stays up while the startupProbe holds), so this is safe against slow loads.
# Pure awk (handles multiple matches on one line — kubectl -o json is often single-line).
if [ "${1:-}" = "--pod-max-restarts" ]; then
    awk '{ while (match($0, /"restartCount"[ \t]*:[ \t]*[0-9]+/)) {
           s = substr($0, RSTART, RLENGTH); gsub(/[^0-9]/, "", s);
           if (s + 0 > m) m = s + 0; $0 = substr($0, RSTART + RLENGTH) } }
       END { printf "%d\n", m + 0 }'
    exit 0
fi

# ── testable sub-mode: is any container CURRENTLY waiting in CrashLoopBackOff? reads pod json on stdin ───────
# Prints 1 iff some container's waiting.reason is CrashLoopBackOff, else 0. Unlike --pod-crashlooping (which
# ALSO trips on restartCount>=3), this is the PURE backoff-state signal so wait_server_ready.sh can time-bound a
# sustained CrashLoopBackOff window (M minutes) independently of the restartCount threshold (K).
if [ "${1:-}" = "--pod-in-backoff" ]; then
    if grep -qE '"reason"[[:space:]]*:[[:space:]]*"CrashLoopBackOff"'; then echo 1; else echo 0; fi
    exit 0
fi

# ── testable sub-mode: classify a Job's terminal state from a `succeeded|failed|active` string on stdin ──
# Kubernetes omits absent numeric fields, so delimiters preserve the succeeded, failed, and active columns.
# Prints: complete | failed | active | pending.
if [ "${1:-}" = "--job-verdict" ]; then
    IFS='|' read -r _s _f _a << EOF
$(cat)
EOF
    _s=${_s:-0}
    _f=${_f:-0}
    _a=${_a:-0}
    if [ "$_s" -ge 1 ] 2> /dev/null; then
        echo complete
    elif [ "$_f" -ge 1 ] 2> /dev/null; then
        echo failed
    elif [ "$_a" -ge 1 ] 2> /dev/null; then
        echo active
    else echo pending; fi
    exit 0
fi

# ── testable sub-mode: is a pod STUCK before Running? reads `kubectl get pod -o json` on stdin (no cluster) ──
# Prints 1 if the pod is Pending (ContainerCreating / Unschedulable) or a container is waiting on an image error
# (ImagePullBackOff / ErrImagePull / …), else 0. A pod that never leaves this state squats the GPU while doing
# no work — the GB300 live c8 sat in ContainerCreating for ~2h. The main loop time-bounds this before acting.
if [ "${1:-}" = "--pod-pending" ]; then
    _blob="$(cat)"
    if printf '%s' "$_blob" | grep -qE '"phase"[[:space:]]*:[[:space:]]*"Pending"' \
        || printf '%s' "$_blob" | grep -qE '"reason"[[:space:]]*:[[:space:]]*"(ImagePullBackOff|ErrImagePull|InvalidImageName|CreateContainerError|CreateContainerConfigError)"'; then
        echo 1
    else
        echo 0
    fi
    exit 0
fi

CELL=""
PROFILE=""
RUN_ID=""
IDLE_MIN=30
POLL=60
KILL_HUNG=0
KEEP_SERVER=0
GRACE=300
MAX_WALL=0
STUCK_MIN=15
POSITIONAL=()
while [ $# -gt 0 ]; do
    case "$1" in
        --idle-timeout)
            IDLE_MIN="${2:?}"
            shift 2
            ;;
        --poll)
            POLL="${2:?}"
            shift 2
            ;;
        --grace)
            GRACE="${2:?}"
            shift 2
            ;;
        --max-wall)
            MAX_WALL="${2:?}"
            shift 2
            ;; # hard wall cap (sec): cancel the Job after this long, active or not
        --stuck-timeout)
            STUCK_MIN="${2:?}"
            shift 2
            ;; # (--kill-hung) fail fast if the Job pod stays Pending this long
        --kill-hung)
            KILL_HUNG=1
            shift
            ;;
        --keep-server)
            KEEP_SERVER=1
            shift
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
CELL="${1:?usage: idle_guard.sh <cell-dir> <cluster-profile> [run-id] [flags]}"
PROFILE="${2:?need a cluster-profile name}"
RUN_ID="${3:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "idle_guard: no profile at $ENVF" >&2
    exit 1
}
[ -f "$CELL/recipe.yaml" ] || {
    echo "idle_guard: no recipe.yaml in $CELL" >&2
    exit 1
}
set -a
. "$ENVF"
set +a
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
NAME="$(sed -n 's/^  name: \(.*\)/\1/p' "$CELL/recipe.yaml" | head -1)"
[ -n "$NAME" ] || {
    echo "idle_guard: could not read envelope.name from $CELL/recipe.yaml" >&2
    exit 1
}
SERVER_DEPLOY="${NAME}-server"

log() { echo "[idle_guard $(date -u +%H:%M:%S)] $*"; }

# Resolve the run's Job: prefer an exact <name>-{bench}-<run-id>; else the newest Job for this recipe.
find_job() {
    if [ -n "$RUN_ID" ]; then
        local j="${NAME}-bench-${RUN_ID}" lookup rc
        if lookup="$(kc -n "$NAMESPACE" get job "$j" -o name 2>&1)"; then
            echo "$j"
            return 0
        else
            rc=$?
        fi
        # Return 1 only for a confirmed absence. Authentication, transport, and API-server failures are
        # operational errors (2), never evidence that a live Job is terminal and its server can be scaled down.
        if ! printf "%s" "$lookup" | grep -qiE "Error from server [(]NotFound[)]:"; then
            printf "idle_guard: kubectl lookup failed for job/%s (rc=%s): %s\n" \
                "$j" "$rc" "${lookup:0:240}" >&2
            return 2
        fi
        # A caller that passed a run-id wants that exact run. Do not fall back to the newest
        # older Job while run.sh is still between starting idle_guard and applying the new Job.
        return 1
    fi
    local jobs candidates
    if ! jobs="$(kc -n "$NAMESPACE" get jobs -l "llmb.nvidia.com/recipe=${NAME}" \
        --sort-by=.metadata.creationTimestamp -o name 2>&1)"; then
        printf "idle_guard: kubectl job-list lookup failed: %s\n" "${jobs:0:240}" >&2
        return 2
    fi
    candidates="$(printf "%s\n" "$jobs" | sed -n -e "s#^job[.]batch/##p" -e "s#^job/##p")"
    [ -n "$candidates" ] || return 1
    printf "%s\n" "$candidates" | tail -1
}

# The server Deployment's pod (name prefix <name>-server-), for reading /metrics off localhost:8000.
server_pod() { kc -n "$NAMESPACE" get pods -o name 2> /dev/null | grep "/${NAME}-server-" | head -1; }

# Sum of vLLM generation-token counters, or empty if the server/metrics are unreachable (→ hang check skips).
read_tokens() {
    local pod
    pod="$(server_pod)"
    [ -n "$pod" ] || return 1
    kc -n "$NAMESPACE" exec "$pod" -- sh -c \
        'curl -s --max-time 8 localhost:8000/metrics 2>/dev/null || python3 -c "import urllib.request,sys;sys.stdout.write(urllib.request.urlopen('\''http://localhost:8000/metrics'\'',timeout=8).read().decode())" 2>/dev/null' \
        2> /dev/null | "$0" --sum-tokens
}

# Is the served model's pod CrashLoopBackOff right now? (server-health). Returns 0 = unhealthy.
# Can't-tell (no pod / unreachable) → return 1 (treat healthy; don't kill a run on a transient query miss).
server_unhealthy() {
    local pod
    pod="$(server_pod)"
    [ -n "$pod" ] || return 1
    [ "$(kc -n "$NAMESPACE" get "$pod" -o json 2> /dev/null | "$0" --pod-crashlooping)" = "1" ]
}

# Is the run's Job pod stuck before Running (ContainerCreating / Pending / image error)? Can't-tell (no pod) →
# return 1 (not stuck) so we never fast-fail on a transient query miss.
job_pod_stuck() {
    local j="$1" js
    js="$(kc -n "$NAMESPACE" get pod -l "job-name=$j" -o json 2> /dev/null)"
    [ -n "$js" ] || return 1
    [ "$(printf '%s' "$js" | "$0" --pod-pending)" = "1" ]
}

scale_server_to_zero() {
    if [ "$KEEP_SERVER" = 1 ]; then
        log "terminal — leaving server up (--keep-server; caller owns teardown)"
        return
    fi
    if kc -n "$NAMESPACE" get deployment "$SERVER_DEPLOY" > /dev/null 2>&1; then
        kc -n "$NAMESPACE" scale deployment "$SERVER_DEPLOY" --replicas=0 > /dev/null
        log "scaled $SERVER_DEPLOY → 0 (GPU node freed; resume: scripts/deploy.sh $CELL $PROFILE)"
    else
        log "no single-deployment server ($SERVER_DEPLOY) to scale — disagg stack? scale workers manually"
    fi
}

log "watching cell=$NAME ns=$NAMESPACE  run-id=${RUN_ID:-<newest>}  idle-timeout=${IDLE_MIN}m  kill-hung=$KILL_HUNG"
WAITED=0
LOOKUP_FAILURES=0
while true; do
    if JOB="$(find_job)"; then
        [ -n "$JOB" ] && break
        LOOKUP_RC=1
    else
        LOOKUP_RC=$?
    fi
    if [ "$LOOKUP_RC" -eq 2 ]; then
        LOOKUP_FAILURES=$((LOOKUP_FAILURES + 1))
        log "Kubernetes lookup failed (${LOOKUP_FAILURES}/3) — leaving server untouched"
        if [ "$LOOKUP_FAILURES" -ge 3 ]; then
            log "Kubernetes API remained unavailable — guard exiting non-zero without teardown"
            exit 1
        fi
        sleep "$POLL"
        continue
    fi
    LOOKUP_FAILURES=0
    WAITED=$((WAITED + POLL))
    if [ "$WAITED" -ge "$GRACE" ]; then
        log "no Job for $NAME appeared within ${GRACE}s — nothing to guard"
        exit 0
    fi
    sleep "$POLL"
done
log "guarding job/$JOB${MAX_WALL:+  (max-wall ${MAX_WALL}s)}"

GUARD_START=$(date -u +%s)
LAST_TOK=-1
LAST_PROGRESS=$GUARD_START
STUCK_SINCE=0
while true; do
    if JOB="$(find_job)"; then
        LOOKUP_RC=0
    else
        LOOKUP_RC=$?
    fi
    if [ "$LOOKUP_RC" -eq 2 ]; then
        LOOKUP_FAILURES=$((LOOKUP_FAILURES + 1))
        log "Kubernetes lookup failed (${LOOKUP_FAILURES}/3) — leaving server untouched and retrying"
        if [ "$LOOKUP_FAILURES" -ge 3 ]; then
            log "Kubernetes API remained unavailable — guard exiting non-zero without teardown"
            exit 1
        fi
        sleep "$POLL"
        continue
    fi
    LOOKUP_FAILURES=0
    if [ "$LOOKUP_RC" -eq 1 ] || [ -z "$JOB" ]; then
        log "job confirmed absent — assuming terminal"
        scale_server_to_zero
        exit 0
    fi
    # Classify via the pure --job-verdict sub-mode (single source; k8s omits absent numeric fields → "||1" for an
    # active Job, which naive splitting would misread as "1 succeeded" and tear down a live run — B3-1).
    VERDICT="$(kc -n "$NAMESPACE" get job "$JOB" \
        -o jsonpath='{.status.succeeded}{"|"}{.status.failed}{"|"}{.status.active}' 2> /dev/null | "$0" --job-verdict)"
    if [ "$VERDICT" = complete ]; then
        log "job/$JOB Complete"
        scale_server_to_zero
        exit 0
    fi
    if [ "$VERDICT" = failed ]; then
        log "job/$JOB Failed"
        scale_server_to_zero
        exit 0
    fi

    # Hard wall cap (--max-wall): cancel the Job once it has run longer than the cap, regardless of progress.
    # Used by run.sh --smoke-only so a "smoke" against a long trace can't run for hours (a c=1 pass over a
    # 500-session trace is an overnight job, not a smoke) — it proves the path works, then stops.
    if [ "$MAX_WALL" -gt 0 ]; then
        ELAPSED=$(($(date -u +%s) - GUARD_START))
        if [ "$ELAPSED" -ge "$MAX_WALL" ]; then
            # Don't claim "smoke path proven" if the pod never actually ran: a short --smoke-only wall can fire before
            # the stuck-schedule guard (GB300), so check the pod state here too and give the honest verdict + Events.
            if [ "$KILL_HUNG" = 1 ] && job_pod_stuck "$JOB"; then
                log "MAX WALL: job/$JOB hit the ${MAX_WALL}s cap but its pod NEVER left Pending/ContainerCreating — the path did NOT run. Events:"
                kc -n "$NAMESPACE" describe pod -l "job-name=$JOB" 2> /dev/null | sed -n '/Events:/,$p' | tail -12 | sed 's/^/    /' >&2 || true
            else
                log "MAX WALL: job/$JOB ran ${ELAPSED}s (>= ${MAX_WALL}s cap) — cancelling (smoke path proven)"
            fi
            kc -n "$NAMESPACE" delete job "$JOB" --wait=false > /dev/null 2>&1 || true
            sleep "$POLL"
            continue
        fi
    fi

    if [ "$KILL_HUNG" = 1 ] && server_unhealthy; then
        log "SERVER UNHEALTHY: ${NAME}-server pod is CrashLoopBackOff while job active — deleting job/$JOB for fast recovery"
        kc -n "$NAMESPACE" delete job "$JOB" --wait=false > /dev/null 2>&1 || true
        sleep "$POLL"
        continue # next loop sees the Job gone → scale_server_to_zero; run.sh's trap prints recovery
    fi

    # Stuck-scheduling watchdog: a Job pod that never leaves Pending/ContainerCreating (bad image, unschedulable,
    # a volume that won't mount) does no work while the server squats the GPU — the GB300 live c8 sat in
    # ContainerCreating ~2h before a harness timeout. Fail fast once it has been stuck >= --stuck-timeout min,
    # surfacing the pod Events so the cause (image pull? PVC? task-source mount?) is in the log, not a mystery.
    if [ "$KILL_HUNG" = 1 ] && job_pod_stuck "$JOB"; then
        NOW=$(date -u +%s)
        [ "$STUCK_SINCE" -eq 0 ] && STUCK_SINCE=$NOW
        STUCK_FOR=$((NOW - STUCK_SINCE))
        if [ "$STUCK_FOR" -ge $((STUCK_MIN * 60)) ]; then
            log "STUCK: job/$JOB pod Pending/ContainerCreating for $((STUCK_FOR / 60))m (>= ${STUCK_MIN}m) — not progressing. Events:"
            kc -n "$NAMESPACE" describe pod -l "job-name=$JOB" 2> /dev/null | sed -n '/Events:/,$p' | tail -12 | sed 's/^/    /' >&2 || true
            kc -n "$NAMESPACE" delete job "$JOB" --wait=false > /dev/null 2>&1 || true
            sleep "$POLL"
            continue
        fi
    else
        STUCK_SINCE=0 # pod is running (or transiently unqueryable) — reset the stuck timer
    fi

    if [ "$KILL_HUNG" = 1 ]; then
        TOK="$(read_tokens || true)"
        if [ -n "${TOK:-}" ] && [ "$TOK" -ge 0 ] 2> /dev/null; then
            NOW=$(date -u +%s)
            if [ "$TOK" -gt "$LAST_TOK" ]; then
                LAST_TOK="$TOK"
                LAST_PROGRESS="$NOW"
            else
                IDLE=$((NOW - LAST_PROGRESS))
                if [ "$IDLE" -ge $((IDLE_MIN * 60)) ]; then
                    log "HUNG: no new generation tokens for $((IDLE / 60))m (>= ${IDLE_MIN}m) while job active — deleting job/$JOB"
                    kc -n "$NAMESPACE" delete job "$JOB" --wait=false > /dev/null 2>&1 || true
                    # next loop sees the Job gone → scale_server_to_zero
                fi
            fi
        fi
    fi
    sleep "$POLL"
done
