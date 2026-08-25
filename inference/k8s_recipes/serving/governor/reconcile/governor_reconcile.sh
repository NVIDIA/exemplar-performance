#!/bin/sh
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

# governor_reconcile.sh — the one lane-agnostic in-cluster resource governor (Phase-2).
#
# Implements the in-cluster governor contract (RWX-revised, §0/§2/§5/§8). Pure POSIX sh + kubectl + jq,
# mirroring scripts/idle_guard.sh so its classifier sub-modes unit-test offline with NO cluster. The
# governor mounts the shared per-namespace RWX CONTROL PVC (`llmb-control` at /control), scans every
# <run-id>/ subdir in ONE pass, and absorbs four responsibilities:
#
#   1. Orphan-sweep      — scale->0 / delete an `llmb`-labeled server Deployment whose owner bench Job is
#                          terminal-or-gone (the crash-safe NET behind Phase-1 ownerReference GC).      §2.1
#   2. Stall-halt        — delete an Active bench Job that is BOTH making no token progress AND still has
#                          work outstanding at the server (see the stall check below), plus the
#                          server-CrashLoop / stuck-schedule fast-fails ported from idle_guard.sh
#                          -> reason=stalled/server-crashed.                                            §2.2
#   3. Precise 2x-timeout— delete an Active bench Job whose age exceeds EXPECTED_RUNTIME_SECONDS x 2
#                          (clamped) -> reason=timed-out; complements native activeDeadlineSeconds.       §2.3
#   4. Tenant-safety     — label gate + never-kill-progressing + owner-terminal ownership gate +
#                          namespace-scoped only + DRY_RUN + conservative-on-absence.                     §5
#
# Every terminal action writes an OBVIOUS reason to <run-id>/governor.json on the control PVC (governor-owned
# file; single-writer-per-file => race-free with the wrapper's status.json, §1.3). `llmb-k8s status` reads it.
#
# Stall detection requires both a lack of token progress and outstanding server requests.
# A normal pause with no active requests does not trigger cleanup.
# In a healthy gap between rungs nothing is in flight, so the second half is false and nothing accrues.
# Both halves come from the wrapper's status.json (scripts/resilient_inject.py):
#   progress_utc  — when the MONOTONIC HIGH-WATER token counter last advanced (a counter reset cannot
#                   regress it, so a restart never reads as a stall)
#   idle_utc      — when work was last observed ABSENT **or unmeasurable**; NOW-idle_utc is therefore the
#                   length of the continuous window over which work has been outstanding
#   inflight_requests — instantaneous in-flight count; -1 (or absent, e.g. an older wrapper) == UNKNOWN
# Decision, evaluated ONLY while phase=generating:
#   both ages >= STALL_THRESHOLD                     -> WARN   (governor.json action=warn; nothing deleted)
#   both ages >= STALL_THRESHOLD * STALL_HALT_MULT   -> HALT   (reason=stalled, with the evidence in detail)
#   anything unknown / no work outstanding           -> NO ACTION, and the reason is LOGGED, never silent.
#
# Offline-testable knobs (default to the in-cluster values so the container needs no overrides):
#   KUBECTL       kubectl binary (a fake shim serves canned JSON in the selftest)   default: kubectl
#   CONTROL_ROOT  control-PVC mount root (a fixture dir in the selftest)             default: /control
#   NAMESPACE     namespace to reconcile (Role is namespace-scoped)                  required in-cluster
#   DRY_RUN       1 = log intended actions, mutate NOTHING                           default: 0
#   STALL_THRESHOLD / HEARTBEAT_DEAD / STUCK_THRESHOLD / TIMEOUT_MULT / MIN_KILL_SECONDS  (see §4 manifest)
#   STALL_HALT_MULT  warn at STALL_THRESHOLD, halt at this multiple of it            default: 2

set -eu

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# Testable stdin sub-modes — ported VERBATIM from idle_guard.sh:35-87 so `make test` exercises the classifier
# logic with no cluster. The main loop below shells back into `"$0" --<mode>` exactly as idle_guard.sh does.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────

# --sum-tokens: sum vLLM generation-token counters from a /metrics blob on stdin.
# NOT the live progress signal — the wrapper's _parse_tokens (scripts/resilient_inject.py) is, and it is
# lane-agnostic (Dynamo/SGLang too) and takes the MAX across counter families rather than a sum. This
# sub-mode is a vLLM-only convenience kept for offline parity checks; do not extend the governor onto it.
if [ "${1:-}" = "--sum-tokens" ]; then
    awk '/^vllm:generation_tokens(_total)?[ {]/ { s += $NF + 0 } END { printf "%.0f\n", s }'
    exit 0
fi

# --pod-crashlooping: reads `kubectl get pod -o json` on stdin. Prints 1 if any container is waiting in
# CrashLoopBackOff (or has restarted >= 3 times), else 0. Server-health fast-fail signal (§2.2).
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

# --pod-pending: reads `kubectl get pod -o json` on stdin. Prints 1 if the pod is stuck before Running
# (Pending / ContainerCreating / image error), else 0. Stuck-schedule fast-fail signal (§2.2).
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

# --job-verdict: classify a Job from a `succeeded|failed|active` string on stdin. The `|` delimiters keep the
# columns positional (k8s OMITS absent numeric fields -> an active Job renders "||1"; naive space-splitting
# misreads it as "1 succeeded" and tears down a live run — idle_guard.sh:57-72's B3-1 fix). Prints exactly
# one of: complete | failed | active | pending.
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

# ─────────────────────────────────────────────────────────────────────────────────────────────────────────
# Main reconcile pass.
# ─────────────────────────────────────────────────────────────────────────────────────────────────────────

: "${NAMESPACE:?governor_reconcile: NAMESPACE is required (namespace-scoped Role)}"
KUBECTL="${KUBECTL:-kubectl}"
CONTROL_ROOT="${CONTROL_ROOT:-/control}"
DRY_RUN="${DRY_RUN:-0}"
STALL_THRESHOLD="${STALL_THRESHOLD:-1800}" # 30m — WARN threshold; matches idle_guard --idle-timeout
# STALL_HALT_MULT — the HALT threshold is STALL_THRESHOLD x this (default 60m). Deliberately 2x the warn so
# an operator sees a stall warning for a full STALL_THRESHOLD before anything is ever deleted.
STALL_HALT_MULT="${STALL_HALT_MULT:-2}"
HEARTBEAT_DEAD="${HEARTBEAT_DEAD:-300}"   # 5m  — pod wedged if no heartbeat this long
STUCK_THRESHOLD="${STUCK_THRESHOLD:-900}" # 15m — matches idle_guard --stuck-timeout default (reserved)
TIMEOUT_MULT="${TIMEOUT_MULT:-2}"
MIN_KILL_SECONDS="${MIN_KILL_SECONDS:-3600}"
# ORPHAN_GRACE — the orphan-sweep FAIL-SAFE window. A server whose owner cannot be positively resolved is
# NEVER reaped until it is at least this old, because a just-brought-up server (deploy.sh runs the server
# BEFORE the bench Job exists / before adopt_server stamps the ownerRef) legitimately has no resolvable owner
# for a short window. Reaping on that transient absence would delete a healthy live server (the destructive
# bug this guard fixes). Defaults to MIN_KILL_SECONDS.
ORPHAN_GRACE="${ORPHAN_GRACE:-$MIN_KILL_SECONDS}"
# GOVERNOR_MODE — the AIRTIGHT top-level safety gate. `observe` (report-only) is the FAIL-SAFE DEFAULT: the
# governor computes what it WOULD reap/halt, writes it to a report ConfigMap + its logs, and mutates NOTHING.
# `enforce` is the ONLY value that ever permits a mutation, and even then DRY_RUN=1 still previews. A run must
# OPT IN to enforce; a missing/typo'd value degrades to observe. (the in-cluster governor contract §5 + deploy.)
GOVERNOR_MODE="${GOVERNOR_MODE:-observe}"
OBSERVE_REPORT_CM="${OBSERVE_REPORT_CM:-governor-observe-report}"
SELF="$0"
NOW="$(date -u +%s)"
SEL='app.kubernetes.io/managed-by=llmb-recipe' # master label gate (§5.1) — every enumeration carries it.
MANAGED_BY_KEY='app.kubernetes.io/managed-by'  # our managed-by label KEY (value must equal llmb-recipe).
MANAGED_BY_VAL='llmb-recipe'

# ── The single, airtight mutation gate ─────────────────────────────────────────────────────────────────────
# MUTATE=1 ONLY when explicitly enforcing AND not dry-running. observe mode can NEVER mutate — it FORCES
# DRY_RUN=1 and MUTATE=0 no matter what else is set. Every scale/delete path below is gated on $MUTATE, so a
# single guard (not one per call site) makes observe mode provably reap-nothing.
if [ "$GOVERNOR_MODE" = enforce ] && [ "$DRY_RUN" != 1 ]; then MUTATE=1; else MUTATE=0; fi
[ "$GOVERNOR_MODE" = observe ] && DRY_RUN=1

# Observe-mode report accumulator (one line per finding) — published to the ConfigMap + echoed to the logs.
OBSERVE_REPORT="${OBSERVE_REPORT:-$(mktemp 2> /dev/null || echo "/tmp/governor-observe-report.$$")}"

log() { echo "[governor $(date -u +%H:%M:%S)] $*"; }

# finding() — record an observe-mode line to BOTH the report accumulator and the pod logs (§ observe deploy).
finding() {
    printf '%s\n' "$*" >> "$OBSERVE_REPORT" 2> /dev/null || true
    log "$*"
}

# namespace-scoped kubectl (the Role cannot see or act outside $NAMESPACE — §5.3).
kj() { "$KUBECTL" -n "$NAMESPACE" "$@"; }

# ISO8601-UTC -> epoch seconds, portable across GNU/busybox (`date -d`) and BSD (`date -j -f`). Can't-tell ->
# echo $NOW so the caller sees "age 0 / fresh" and takes NO action (conservative-on-absence, §5.6).
epoch() {
    _d="$1"
    [ -n "$_d" ] && [ "$_d" != null ] || {
        echo "$NOW"
        return
    }
    _e="$(date -u -d "$_d" +%s 2> /dev/null)" && {
        echo "$_e"
        return
    }
    _e="$(date -u -j -f '%Y-%m-%dT%H:%M:%SZ' "$_d" +%s 2> /dev/null)" && {
        echo "$_e"
        return
    }
    echo "$NOW"
}

# Escape a string for a JSON double-quoted scalar (backslash first, then quote; control chars stripped).
# A detail string carrying a `"` would otherwise emit invalid governor.json and make the halt UNREADABLE —
# exactly the failure mode that lets a kill go unexplained.
jesc() { printf '%s' "${1:-}" | sed -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/[[:cntrl:]]//g' | tr -d '\n'; }

# GOVERNOR-owned governor.json — atomic .tmp->mv (rename is atomic on POSIX/NFS/Lustre; §1.3). The governor
# NEVER touches the wrapper-owned status.json, so there is no lost-update race in a shared <run-id>/ dir.
wr_reason() { # <run-id> <action> <reason> <detail>
    _d="$CONTROL_ROOT/$1"
    [ -d "$_d" ] || return 0
    printf '{"action":"%s","reason":"%s","utc":"%s","detail":"%s"}\n' \
        "$(jesc "$2")" "$(jesc "$3")" "$(date -u +%FT%TZ)" "$(jesc "$4")" > "$_d/governor.json.tmp" \
        && mv "$_d/governor.json.tmp" "$_d/governor.json"
}

# HALT = record the reason on the control PVC, then delete the Job (idempotent, --wait=false). ownerReference
# GC + responsibility 1 tear the server down afterward. In observe mode NOTHING is written or deleted — the
# intent is reported only; in enforce+DRY_RUN the reason is previewed on the PVC but the Job is not touched.
halt() { # <job> <run-id> <reason> <detail>
    if [ "$GOVERNOR_MODE" = observe ]; then
        finding "WOULD-HALT job/$1 reason=$3 -- $4" # report-only: no governor.json, no delete
        return 0
    fi
    wr_reason "$2" halt "$3" "$4"
    if [ "$MUTATE" != 1 ]; then
        log "[dry-run] would delete job/$1 reason=$3 ($4)"
        return 0
    fi
    kj delete job "$1" --wait=false 2> /dev/null || true
    log "deleted job/$1 reason=$3 ($4)"
}

# WARN = record an early, NON-destructive signal on the control PVC (`llmb-k8s status` prints it) and log it.
# Nothing is scaled, deleted or otherwise touched. This is what an operator sees for a full STALL_THRESHOLD
# before the halt threshold is reached, so a kill is never the first thing anyone hears about a stall.
warn_run() { # <job> <run-id> <reason> <detail>
    if [ "$GOVERNOR_MODE" = observe ]; then
        finding "WOULD-WARN job/$1 reason=$3 -- $4"
        return 0
    fi
    wr_reason "$2" warn "$3" "$4"
    log "WARN job/$1 reason=$3 ($4) -- NOT halted; halt threshold not reached"
}

# ── Responsibility 1: completion cleanup / orphan-sweep — drive off live k8s state (§2.1) ─────────────────
# A server Deployment is scaled/deleted ONLY when its owner bench Job is terminal-or-gone (ownership gate,
# §5.4). A running-owner server is left alone. Dedicated-per-cell (§9) makes "owner Job terminal-or-gone" the
# sole teardown predicate — no shared-server / last-user bookkeeping.
sweep_orphans() {
    kj get deploy -l "$SEL" -o json 2> /dev/null | jq -r '.items[] | [.metadata.name,
        ((.metadata.ownerReferences[]?|select(.kind=="Job")|.uid)//"none"),
        (.metadata.labels["llmb.nvidia.com/cell"]//""),
        (.metadata.creationTimestamp//""),
        (.status.availableReplicas//0),
        (.metadata.labels["llmb.nvidia.com/created-at"]//""),
        (.metadata.labels["llmb.nvidia.com/max-lifetime-s"]//"")] | @tsv' 2> /dev/null \
        | while IFS="$(printf '\t')" read -r dep owner cell created available life_created max_life; do
            [ -n "$dep" ] || continue
            available="${available:-0}"
            jobs="$(kj get jobs -l "$SEL" -o json 2> /dev/null)"
            # (1) Resolve the owner by its ownerReference UID — the PRIMARY, unambiguous link. A matched Job is
            #     terminal (reap-eligible), or present-but-not-terminal (active/pending -> KEEP). No match -> absent.
            owner_state="$(printf '%s' "$jobs" | jq -r --arg u "$owner" '
          ( .items[] | select($u!="none" and .metadata.uid==$u) )
          | if (.status.succeeded//0)>=1 or (.status.failed//0)>=1 then "terminal" else "present" end' \
                2> /dev/null | head -1)"
            [ -n "$owner_state" ] || owner_state="absent"
            # (2) Cell fallback signals — ONLY meaningful when the server carries llmb.nvidia.com/cell (deploy.sh
            #     stamps it as a runtime label; it is NOT in the hashed template). Count live vs finished bench Jobs
            #     for this cell. An empty cell label yields 0/0 (the fallback simply cannot fire — fail-safe).
            active_for_cell="$(printf '%s' "$jobs" | jq -r --arg c "$cell" \
                '[.items[]|select($c!="" and .metadata.labels["llmb.nvidia.com/cell"]==$c and (.status.active//0)>=1)]|length' 2> /dev/null)"
            terminal_for_cell="$(printf '%s' "$jobs" | jq -r --arg c "$cell" \
                '[.items[]|select($c!="" and .metadata.labels["llmb.nvidia.com/cell"]==$c and ((.status.succeeded//0)>=1 or (.status.failed//0)>=1))]|length' 2> /dev/null)"
            active_for_cell="${active_for_cell:-0}"
            terminal_for_cell="${terminal_for_cell:-0}"
            age=$((NOW - $(epoch "$created")))
            # Lifetime age is measured from the deploy-time llmb.nvidia.com/created-at stamp (deploy.sh), NOT the k8s
            # creationTimestamp, so a governor-restart / API re-list can't reset it. The stamp is EPOCH SECONDS (k8s
            # label values forbid the ':' in ISO8601), read as an integer directly. Absent/non-numeric -> unset (0).
            life_age=0
            case "$life_created" in
                '' | *[!0-9]*) : ;; # no label, or not a clean integer -> lifetime ceiling unset
                *) life_age=$((NOW - life_created)) ;;
            esac

            # (3) FAIL-SAFE decision. Reap ONLY on POSITIVE evidence of orphan-hood; NEVER on absence/ambiguity.
            #   keep  = leave the server ALONE (live owner, a live bench Job for the cell, too-recent, or no evidence)
            #   scale = scale->0 only (owner Job terminal-by-UID; ownerReference GC then deletes it)
            #   reap  = scale->0 + delete (owner UID gone AND a run demonstrably existed AND no live bench Job for cell)
            action=keep
            _why=""
            if [ "$owner_state" = present ] || [ "$active_for_cell" -ge 1 ] 2> /dev/null; then
                # NEVER reap while a live bench Job for the cell exists (CRITICAL SAFETY: cannot kill a running
                # benchmark) — this keep gate is evaluated FIRST, so even the max-lifetime ceiling below never fires on
                # a server with an active run. Benchmark runs stamp llmb.nvidia.com/cell on their bench Job.
                action=keep
                _why="owner Job active"
            elif [ -n "$max_life" ] && [ "$max_life" -gt 0 ] 2> /dev/null && [ "$life_age" -ge "$max_life" ] 2> /dev/null; then
                # S2 HARD-CEILING (belt to the orphan-sweep suspenders): a server whose deploy-time max-lifetime is
                # exceeded is reaped REGARDLESS of ownerRef / availability / cell-fallback state — the sole backstop for
                # a server that is stuck never-Available (perma-cold-load / silent CrashLoop) and so is invisible to the
                # availability-gated orphan branch below. Still gated on active_for_cell==0 (the keep branch above), so
                # it can never kill a running benchmark. Set generously (deploy.sh default) to never clip a real load.
                action=reap
                _why="max-lifetime ceiling exceeded (lifetime ${life_age}s >= ${max_life}s); no live bench Job for cell"
            elif [ "$owner_state" = terminal ]; then
                action=scale
                _why="owner Job terminal"
            elif [ "$age" -lt "$ORPHAN_GRACE" ] 2> /dev/null; then
                # owner UID unresolved but the server is RECENT -> the owner may still be materializing (deploy.sh
                # brings the server up before the bench Job / adopt_server). Do NOT reap. THE destructive-bug fix.
                action=keep
                _why="unresolved owner but recent (age ${age}s < ${ORPHAN_GRACE}s) -- fail-safe hold"
            elif [ -n "$cell" ] && { [ "$owner" != none ] || [ "$terminal_for_cell" -ge 1 ]; } && [ "$active_for_cell" -eq 0 ] 2> /dev/null; then
                # POSITIVE orphan: a run existed (ownerRef was stamped, or a finished bench Job for the cell is present)
                # AND there is definitively NO live bench Job for the cell. Safe to reclaim the GPU-holding server.
                action=reap
                _why="owner Job gone; no live bench Job for cell"
            elif [ "$owner" = none ] && [ -n "$cell" ] && [ "$active_for_cell" -eq 0 ] 2> /dev/null && [ "$available" -ge 1 ] 2> /dev/null; then
                # POSITIVE orphan #2 (never-adopted): NO ownerRef was EVER stamped + past the cold-load grace hold above
                # (age >= ORPHAN_GRACE) + NO live bench Job for the cell + the server is AVAILABLE (finished coming up,
                # not mid cold-load) -> the launching run.sh died BEFORE the coord Job was created (e.g. SIGKILL from a
                # laptop sleeping), leaving a bare GPU-holding server that neither ownerRef-GC nor a Job activeDeadline
                # can ever reap. This is the 11h-idle-orphan gap.
                #   The `available >= 1` gate is the CRITICAL cold-load protection: a giant 1M-ctx server takes 30-60min+
                #   to load weights and reports availableReplicas=0 (readiness probe failing) the whole time, with NO
                #   bench Job yet (the run launches only once the server is Ready). Without this gate the 20-min
                #   ORPHAN_GRACE would reap such a server mid-load. A still-loading (available=0) server is HELD here and
                #   bounded only by the far-larger max-lifetime ceiling above. Verified against the live GB300 1M study.
                action=reap
                _why="never-adopted server past grace (age ${age}s >= ${ORPHAN_GRACE}s), server Available; no ownerRef + no live bench Job for cell"
            elif [ "$owner" = none ] && [ -n "$cell" ] && [ "$active_for_cell" -eq 0 ] 2> /dev/null && [ "$available" -lt 1 ] 2> /dev/null; then
                # never-adopted + past grace but NOT yet Available -> still coming up (cold-load) OR wedged. HOLD; the
                # max-lifetime ceiling is the only thing that ever reaps a perma-unavailable server (fail-safe).
                action=keep
                _why="never-adopted past grace but server not Available (availableReplicas=0) -- cold-load/coming-up hold"
            else
                # No positive evidence (ownerless with no cell label, or nothing conclusive) -> conservative KEEP.
                action=keep
                _why="no positive orphan evidence (owner=$owner cell='${cell}') -- fail-safe hold"
            fi

            case "$action" in
                keep)
                    if [ "$GOVERNOR_MODE" = observe ]; then
                        if [ "$_why" = "owner Job active" ]; then
                            finding "HEALTHY-ACTIVE deploy/$dep (owner Job active) -- not a reap candidate"
                        else
                            finding "HELD deploy/$dep -- $_why (not reaped)"
                        fi
                    fi
                    ;;
                scale)
                    if [ "$MUTATE" != 1 ]; then
                        if [ "$GOVERNOR_MODE" = observe ]; then
                            finding "WOULD-REAP completion-cleanup deploy/$dep -> scale->0 (owner Job terminal)"
                        else
                            log "[dry-run] would scale->0 deploy/$dep (owner Job terminal)"
                        fi
                        continue
                    fi
                    kj scale deploy "$dep" --replicas=0 2> /dev/null || true
                    log "scaled deploy/$dep -> 0 (owner Job terminal)"
                    ;;
                reap)
                    if [ "$MUTATE" != 1 ]; then
                        if [ "$GOVERNOR_MODE" = observe ]; then
                            finding "WOULD-REAP orphan-sweep deploy/$dep -> scale->0 + delete ($_why)"
                        else
                            log "[dry-run] would scale->0 + delete deploy/$dep ($_why)"
                        fi
                        continue
                    fi
                    kj scale deploy "$dep" --replicas=0 2> /dev/null || true
                    log "scaled deploy/$dep -> 0 ($_why)"
                    kj delete deploy "$dep" --wait=false 2> /dev/null || true
                    log "deleted deploy/$dep ($_why)"
                    ;;
            esac
        done
}

# ── Responsibilities 2 & 3: stall-halt + precise timeout — drive off the control PVC (§2.2/§2.3) ──────────
scan_runs() {
    for d in "$CONTROL_ROOT"/*/; do
        [ -f "$d/status.json" ] || continue # run just started / no status yet -> skip, retry next pass
        rid="$(basename "$d")"
        s="$(cat "$d/status.json")"
        # A status.json that does not PARSE makes every `jq -r` below return empty, which silently drops the
        # run from ALL supervision (no stall, no timeout, no crash fast-fail). Absence of supervision must be
        # LOUD, never inferred from a blank field. (The wrapper now JSON-escapes every field it writes.)
        if ! printf '%s' "$s" | jq -e . > /dev/null 2>&1; then
            log "UNSUPERVISED run=$rid: status.json is not valid JSON -> no stall/timeout supervision this pass"
            if [ "$GOVERNOR_MODE" = observe ]; then
                finding "UNSUPERVISED run/$rid -- status.json is not valid JSON (no stall/timeout supervision)"
            fi
            continue
        fi
        state="$(printf '%s' "$s" | jq -r '.state // ""')"
        [ "$state" = running ] || continue # terminal/queued -> resp.1 (server) handles teardown

        # RECONCILE NOTE: the Job name is <recipe>-bench-<run-id> where <recipe>=envelope.name
        # (RECIPE_SHORTNAME). status.json's `.cell` is the cell DIRECTORY PATH (LLMB_CELL), NOT the short name —
        # so we build the Job name from `.recipe`, not `.cell` (the design §8 pseudocode said `.cell`; the
        # hardening's final wrapper writes the dir path there). See report / bench-job.yaml.j2:55.
        recipe="$(printf '%s' "$s" | jq -r '.recipe // ""')"
        [ -n "$recipe" ] || recipe="$(printf '%s' "$s" | jq -r '.cell // ""')" # legacy fallback
        job="${recipe}-bench-${rid}"

        jj="$(kj get job "$job" -o json 2> /dev/null)" || continue
        printf '%s' "$jj" | jq -e '(.status.active//0)>=1' > /dev/null 2>&1 || continue # only Active (safety)

        phase="$(printf '%s' "$s" | jq -r '.phase // ""')"
        hb="$(printf '%s' "$s" | jq -r '.heartbeat_utc // ""')"
        pu="$(printf '%s' "$s" | jq -r '.progress_utc // ""')"
        exp="$(printf '%s' "$s" | jq -r '.expected_runtime_seconds // 0')"
        start="$(printf '%s' "$jj" | jq -r '.status.startTime // .metadata.creationTimestamp // ""')"
        age=$((NOW - $(epoch "$start")))

        # (2) server-health fast-fail — server pod CrashLoopBackOff while the Job is Active (§2.2).
        pod="$(kj get pods -l "job-name=$job" -o json 2> /dev/null || echo '{}')"
        if [ "$(printf '%s' "$pod" | "$SELF" --pod-crashlooping)" = 1 ]; then
            halt "$job" "$rid" server-crashed "server CrashLoopBackOff while job active"
            continue
        fi
        # (2) stuck-schedule fast-fail — bench pod wedged Pending/ImagePull. status.json's own age is the
        # time-bound (a run stuck before Running never advances heartbeat past STUCK_THRESHOLD).
        if [ "$(printf '%s' "$pod" | "$SELF" --pod-pending)" = 1 ] && [ "$age" -ge "$STUCK_THRESHOLD" ]; then
            halt "$job" "$rid" unschedulable "bench pod stuck Pending/ImagePull ${age}s"
            continue
        fi

        # (2) stall — ONLY while generating (never-kill during bootstrap/collect; §5.2 phase gate).
        if [ "$phase" = generating ]; then
            hb_age=$((NOW - $(epoch "$hb")))
            if [ "$hb_age" -ge "$HEARTBEAT_DEAD" ]; then
                halt "$job" "$rid" stalled "heartbeat lost ${hb_age}s (pod/wrapper wedged)"
                continue
            fi

            # Stall only when token output is flat and requests remain active.
            pu_age=$((NOW - $(epoch "$pu")))
            # idle_utc = last time work was observed absent OR unmeasurable -> work_age is the length of the
            # CONTINUOUS outstanding-work window. Absent field (older wrapper) -> "" -> epoch() -> NOW -> 0.
            idle="$(printf '%s' "$s" | jq -r '.idle_utc // ""')"
            work_age=$((NOW - $(epoch "$idle")))
            # -1 / absent / null == UNKNOWN. UNKNOWN is NEVER treated as "work outstanding".
            inflight="$(printf '%s' "$s" | jq -r '(.inflight_requests // -1) | tostring')"
            queued="$(printf '%s' "$s" | jq -r '(.queued_requests // -1) | tostring')"
            pnote="$(printf '%s' "$s" | jq -r '.progress_note // "unknown"')"
            pcount="$(printf '%s' "$s" | jq -r '(.progress_counter // 0) | tostring')"
            halt_at=$((STALL_THRESHOLD * STALL_HALT_MULT))
            _work=0
            [ "$inflight" -gt 0 ] 2> /dev/null && [ "$work_age" -ge "$STALL_THRESHOLD" ] && _work=1
            if [ "$pu_age" -ge "$STALL_THRESHOLD" ] && [ "$_work" = 1 ]; then
                _ev="no output tokens for ${pu_age}s while ${inflight} request(s) stayed outstanding for ${work_age}s (queued=${queued}, tokens=${pcount}, metrics=${pnote})"
                if [ "$pu_age" -ge "$halt_at" ] && [ "$work_age" -ge "$halt_at" ]; then
                    halt "$job" "$rid" stalled "$_ev; over halt threshold ${halt_at}s (=${STALL_HALT_MULT}x STALL_THRESHOLD ${STALL_THRESHOLD}s)"
                    continue
                fi
                warn_run "$job" "$rid" stall-warning "$_ev; warn at ${STALL_THRESHOLD}s, halt at ${halt_at}s"
            elif [ "$pu_age" -ge "$STALL_THRESHOLD" ]; then
                # Stale tokens but NO outstanding work (or work unmeasurable) -> a legitimate pause between
                # rungs / during finalisation, or an unreadable server. Say so; never halt on staleness alone.
                log "no-halt run=$rid: token progress stale ${pu_age}s BUT no sustained outstanding work (inflight=${inflight} work_age=${work_age}s queued=${queued} metrics=${pnote}) -- staleness alone is not a stall"
            fi
        fi

        # (3) precise 2x-expected timeout — clamp to >= MIN_KILL_SECONDS (never below the native ADS ceiling in
        # practice). exp<=0 (cold/legacy): do NOTHING here — native activeDeadlineSeconds is the coarse backstop.
        if [ "$exp" -gt 0 ] 2> /dev/null; then
            kill_at=$((exp * TIMEOUT_MULT))
            [ "$kill_at" -lt "$MIN_KILL_SECONDS" ] && kill_at="$MIN_KILL_SECONDS"
            if [ "$age" -ge "$kill_at" ]; then
                halt "$job" "$rid" timed-out "age ${age}s >= ${kill_at}s (2x expected ${exp}s)"
                continue
            fi
        fi
    done
}

# ── Observe-only: enumerate OUR bench Jobs and classify (the live c240-style run must read HEALTHY-ACTIVE) ──
# Drives off the label gate — only jobs carrying managed-by=llmb-recipe are considered OURS. This makes the
# report say, in plain words, that a healthy running benchmark is NOT a reap candidate.
report_ours_jobs() {
    kj get jobs -l "$SEL" -o json 2> /dev/null | jq -r '.items[] | [.metadata.name,
        (if   (.status.succeeded//0)>=1 then "complete"
         elif (.status.failed//0)>=1   then "failed"
         elif (.status.active//0)>=1   then "active"
         else "pending" end)] | @tsv' 2> /dev/null \
        | while IFS="$(printf '\t')" read -r jname jstate; do
            [ -n "$jname" ] || continue
            case "$jstate" in
                active) finding "HEALTHY-ACTIVE job/$jname (active) -- not a reap candidate" ;;
                complete | failed) finding "OURS job/$jname ($jstate) -- teardown handled by orphan-sweep" ;;
                *) finding "OURS job/$jname ($jstate)" ;;
            esac
        done
}

# ── Observe-only: enumerate FOREIGN (unmanaged) workloads and REPORT them — NEVER act, even in enforce mode ─
# Foreign = any Deployment/Job WITHOUT app.kubernetes.io/managed-by=llmb-recipe. On shared clusters (other
# teams' lmsysdyn / dynamo-platform workloads) this is the "we do not kill others' GPUs" gate, surfaced.
report_foreign() {
    for _kind in deploy jobs; do
        kj get "$_kind" -o json 2> /dev/null | jq -r \
            --arg key "$MANAGED_BY_KEY" --arg val "$MANAGED_BY_VAL" --arg kind "$_kind" '
        .items[] | select((.metadata.labels[$key] // "") != $val)
        | "FOREIGN (unmanaged) " + $kind + "/" + .metadata.name
          + " managed-by=" + (.metadata.labels[$key] // "<none>")' 2> /dev/null \
            | while IFS= read -r _line; do
                [ -n "$_line" ] || continue
                finding "$_line -- not managed by llmb; NEVER acted on"
            done
    done
}

# ── Observe-only: publish the report to a single namespace ConfigMap using ONLY get/create/update verbs ─────
# (get -> create if absent, else replace(=PUT/update). NO apply/patch — the observe RBAC grants no patch.)
publish_report() {
    _body="$(mktemp 2> /dev/null || echo "/tmp/gov-obs-body.$$")"
    {
        echo "llmb Phase-2 governor — OBSERVE / REPORT-ONLY"
        echo "namespace:      $NAMESPACE"
        echo "generated_utc:  $(date -u +%FT%TZ)"
        echo "mode:           observe (mutations_performed=0)"
        echo "managed_by_gate: ${MANAGED_BY_KEY}=${MANAGED_BY_VAL}"
        echo "---"
        if [ -s "$OBSERVE_REPORT" ]; then
            cat "$OBSERVE_REPORT"
        else echo "no findings: no reap/halt candidates, no foreign workloads"; fi
        echo "---"
        echo "mutations_performed: 0"
    } > "$_body"
    if kj get configmap "$OBSERVE_REPORT_CM" > /dev/null 2>&1; then
        kj create configmap "$OBSERVE_REPORT_CM" --from-file=report.txt="$_body" --dry-run=client -o yaml 2> /dev/null \
            | kj replace -f - > /dev/null 2>&1 \
            && log "observe report -> configmap/$OBSERVE_REPORT_CM (updated)" \
            || log "observe report: configmap/$OBSERVE_REPORT_CM update skipped (RBAC?)"
    else
        kj create configmap "$OBSERVE_REPORT_CM" --from-file=report.txt="$_body" > /dev/null 2>&1 \
            && log "observe report -> configmap/$OBSERVE_REPORT_CM (created)" \
            || log "observe report: configmap/$OBSERVE_REPORT_CM create skipped (RBAC?)"
    fi
    rm -f "$_body" 2> /dev/null || true
}

log "reconcile ns=$NAMESPACE mode=$GOVERNOR_MODE control=$CONTROL_ROOT mutate=$MUTATE dry_run=$DRY_RUN stall=${STALL_THRESHOLD}s hb_dead=${HEARTBEAT_DEAD}s"
sweep_orphans
scan_runs
if [ "$GOVERNOR_MODE" = observe ]; then
    report_ours_jobs
    report_foreign
    publish_report
fi
log "reconcile done"
