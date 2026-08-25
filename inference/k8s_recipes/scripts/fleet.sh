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

# fleet.sh — read-only, client-side, MULTI-CLUSTER fleet status for the k8s benchmark system.
#
# One non-interactive command that shows every run / job / GPU / CPU / age across ALL configured clusters
# in a single JOB-primary pane — a k9s-like view, but multi-cluster:  llmb-k8s fleet --watch
#
# READ-ONLY BY CONSTRUCTION: only `kubectl get` (pods/deploy/jobs/nodes). Nothing is applied, scaled,
# deleted, or installed cluster-side. ORPHAN servers are *detected*, never acted on.
#
# Design / performance:
#   • Clusters are discovered from cluster-profiles/*.env and de-duplicated by PHYSICAL-CLUSTER IDENTITY
#     (KUBE_CLUSTER, else the KUBE_CONTEXT cluster token) — NOT by namespace: several profiles pinning the
#     same cluster with different namespaces collapse to ONE section, with every namespace's runs surfaced
#     under it via cross-namespace discovery. EVERY configured cluster is always shown — a disconnected one
#     appears as auth✗/unreachable, it never silently vanishes.
#   • FEWEST kubectl CALLS (the dominant cost over teleport is each call's TLS+auth handshake, not parsing):
#     the 3 namespaced reads are COLLAPSED into ONE `get pods,deployments,jobs -o json` per cluster (3→1),
#     which doubles as the connectivity probe. `FLEET_TIMING=1` prints per-cluster/per-phase ms to stderr.
#   • PER-CLUSTER READS RUN IN PARALLEL: each cluster probes in a background subshell; we `wait` then render
#     once. Total wall-time drops from Σ(clusters) → max(single cluster).
#   • TTL caches persist across watch frames for cluster and namespace queries; the namespaced
#     OURS read a short ~4s (FLEET_NS_TTL) so rapid refreshes reuse it while a live monitor stays fresh.
#     Steady-state a warm 15s frame is ~1 call/cluster (was ~5).
#   • Fast-fail: the connectivity probe uses a short --request-timeout so a hung/unreachable cluster returns
#     auth✗/unreachable quickly instead of stalling the frame.
#   • --watch is PIPELINED + double-buffered: the NEXT frame is pre-gathered in the background DURING the
#     refresh interval, so when the interval fires the data is warm and the repaint is instant. A per-frame
#     deadline shows a still-pending cluster from its last good frame (`…refreshing · updated Ns ago`).
#   • Stale cached frames are age-labelled and dropped after FLEET_LAST_GOOD_MAX.
#   • Each pane includes the build revision that produced it.
#   • All parsing/classification/rendering is in scripts/fleet_render.py (python3) — this script discovers
#     clusters and runs kubectl only.
#
# ACTIVE-RUNS-FIRST: the view answers, in the first two lines, "what's benchmarking · how many GPUs are mine
# · how long left". A run is ACTIVE only when a live bench/coord Job is DRIVING it — a bare server at 1/1 with
# no driving Job, and control-plane pods (etcd/nats/frontend), are NOT "running": they collapse to counts
# (`· N idle · M parked · K infra`, orphans shown with their held GPU). Only active runs get individual rows
# (name · cluster[TYPE] · GPUs · elapsed/expected). Below the runs, one compact line per cluster (capacity ·
# ours · active · collapsed rest). `--detail` expands each cluster's full job/server tree.
#
# Flags:
#   --cluster <name>   only this cluster profile
#   --stages           SIGNAL-OVER-NOISE hierarchy pane (the DEFAULT under --watch): CLUSTER → CAPACITY →
#                      NAMESPACE → the INSTALLED/RUN·SERVER stage
#                      sub-sections. A cluster is shown IN FULL only when active in a stage (blocked init /
#                      blocked install / an active·failed·orphaned run); idle-connected clusters collapse to
#                      ONE compact line, idle+unreachable to a `+N` tail. INIT/INSTALL come from LOCAL .state
#                      stamps (the wizard's <profile>.readiness.json + install's <cluster>.install.jsonl), so
#                      they render even for an UNREACHABLE cluster; only RUN needs the cluster. `--journey` alias.
#   --flat             the older flat active-runs-first pane. Only needed under --watch (which now defaults to
#                      the hierarchy); it is already the one-shot default. `--no-stages` alias. NOTE: --wide /
#                      --idle / --all / --failed are implemented ONLY by this pane, so passing any of them
#                      under --watch keeps you here rather than silently dropping the flag.
#   --detail           expand each cluster's full job/server tree under its summary line
#   --wide             extra columns in --detail (node, image tag, restarts)
#   --gpu-only         hide 0-GPU active runs (CPU-only bench clients)
#   --history [N|dur]  deeper terminal-run history (default: runs ended in the last hour). N = last-N runs,
#                      dur = a window like 6h/90m/2d, bare/`all` = every terminal run of ours in-namespace.
#   --failed           show ONLY recently-FAILED runs (window widened to a day)
#   --idle             expand the collapsed idle-server (scaled 0/0) list
#   --all              show everything: full job/server tree + all terminal history + idle servers
#   --fast | --mine    SKIP the cluster-scoped capacity reads (nodes / pods -A) — show only OUR namespaced
#                      workloads. The quick "what am I running" check; fast on every cluster.
#   --watch [secs]     live view — DEFAULTS TO THE HIERARCHY PANE (--stages); pass --flat for the old one.
#                      Built-in DOUBLE-BUFFERED refresh (default 15s): gathers the whole next frame before
#                      repainting, so the screen never blanks. A per-frame deadline shows a slow cluster from
#                      its last good frame (`…refreshing · updated Ns ago`) so one laggard never blocks.
#                      NEEDS A TTY: piped (not a terminal) it degrades to ONE full render — a watch stream
#                      into a capture buffer can only ever be truncated mid-table. On a TTY the frame is
#                      fitted to the live terminal size, collapsing idle/inventory/capacity sections first
#                      and NEVER the live RUN·SERVER rows, with an explicit "N of M lines hidden" footer.
#   --no-color/--color force color off/on (default: auto — off when piped / NO_COLOR / non-tty)
#   --once             single-shot (default; accepted for clarity next to --watch). Stays on the FLAT pane by
#                      default — it is the forensic/scripting surface (history, --wide/--idle/--all/--failed);
#                      add --stages for the hierarchy.
#
# Env overrides: FLEET_PROFILES_DIR, KUBECTL, FLEET_KUBECTL_TIMEOUT (25s), FLEET_PROBE_TIMEOUT (8s),
#   FLEET_NODES_TTL (300), FLEET_ALL_TTL (60), FLEET_LAST_GOOD_MAX (120 — hard ceiling on the --watch
#   laggard's last-good frame; older than this it renders as …refreshing rather than as stale-but-plausible
#   data), FLEET_MAX_JOBS (8), FLEET_FRAME_DEADLINE (watch; default
#   =interval), FLEET_SEQUENTIAL (=1 disables parallelism), FLEET_NOW (fixed clock), FLEET_WATCH_ITERATIONS
#   (bounded frame count — also forces the watch path when stdout is not a TTY), FLEET_FORCE_WATCH (=1 keeps
#   the live repaint loop even when piped; for tests — a piped watch stream can only be truncated).
set -uo pipefail # deliberately NOT -e: one bad cluster must never abort the whole command

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROFILES_DIR="${FLEET_PROFILES_DIR:-$ROOT/cluster-profiles}"
KUBECTL="${KUBECTL:-kubectl}"
TIMEOUT="${FLEET_KUBECTL_TIMEOUT:-25s}"
PROBE_TIMEOUT="${FLEET_PROBE_TIMEOUT:-8s}" # fast-fail on the connectivity probe
NS_TTL="${FLEET_NS_TTL:-4}"                # OURS (namespaced) reads: short TTL so rapid --watch reuses
NODES_TTL="${FLEET_NODES_TTL:-300}"        # node capacity changes slowly → cache 5 min
ALL_TTL="${FLEET_ALL_TTL:-60}"             # occupancy changes faster → cache 60s
MAX_JOBS="${FLEET_MAX_JOBS:-8}"            # concurrency cap on parallel per-cluster probes
TIMING="${FLEET_TIMING:-0}"                # =1 → print per-cluster/per-phase timing to stderr
# Maximum age for displaying a labelled last-good frame.
LAST_GOOD_MAX="${FLEET_LAST_GOOD_MAX:-120}" # seconds; 0 = never fall back to a stale frame

# Include a build stamp so captured output is attributable to a revision.
fleet_build() {
    local sha br
    command -v git > /dev/null 2>&1 || return 0
    sha="$(git -C "$ROOT" rev-parse --short=8 HEAD 2> /dev/null)" || return 0
    [ -n "$sha" ] || return 0
    git -C "$ROOT" diff --quiet HEAD -- "$SCRIPT_DIR" 2> /dev/null || sha="$sha-dirty"
    br="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2> /dev/null)"
    printf '%s · %s · %s' "$sha" "${br:-detached}" "$ROOT"
}
BUILD="$(fleet_build)"

_now_ms() { python3 -c 'import time;print(int(time.time()*1000))'; }
# Timing lines go to a FILE (not stderr): background per-cluster probes have their fds redirected away, but
# an explicit file append still lands. The foreground dumps the file to stderr at the end of each frame.
_tlog() {
    [ "$TIMING" = 1 ] && printf 'FLEET_TIMING %s\n' "$*" >> "${TIMING_LOG:-/dev/null}"
    return 0
}

FILTER=""
WIDE=0
GPU_ONLY=0
IDLE=0
DETAIL=0
FAST=0
COLOR="auto"
WATCH=0
INTERVAL=15
ALL=0
FAILED=0
HISTORY=""
STAGES=0
# VIEW is the EXPLICIT view choice ('' = the user didn't say). FLAT_ONLY_FLAG records that they passed a flag
# only the flat pane implements (--wide/--idle/--all/--failed) — that IS an implicit choice of the flat pane,
# so `--watch --wide` must not silently swallow the flag by switching panes.
VIEW=""
FLAT_ONLY_FLAG=0
while [ $# -gt 0 ]; do
    case "$1" in
        --cluster)
            FILTER="${2:-}"
            shift 2
            ;;
        --cluster=*)
            FILTER="${1#*=}"
            shift
            ;;
        --stages | --journey)
            VIEW="stages"
            shift
            ;;
        --flat | --no-stages)
            VIEW="flat"
            shift
            ;;
        --wide)
            WIDE=1
            FLAT_ONLY_FLAG=1
            shift
            ;;
        --gpu-only)
            GPU_ONLY=1
            shift
            ;;
        --idle)
            IDLE=1
            FLAT_ONLY_FLAG=1
            shift
            ;;
        --all)
            ALL=1
            FLAT_ONLY_FLAG=1
            shift
            ;;
        --detail)
            DETAIL=1
            shift
            ;;
        --failed)
            FAILED=1
            FLAT_ONLY_FLAG=1
            shift
            ;;
        --history)
            HISTORY="${2:-all}"
            case "${2:-}" in '' | -*) HISTORY="all" ;; *) shift ;; esac
            shift
            ;;
        --history=*)
            HISTORY="${1#*=}"
            shift
            ;;
        --fast | --mine)
            FAST=1
            shift
            ;;
        --no-color)
            COLOR="off"
            shift
            ;;
        --color)
            COLOR="on"
            shift
            ;;
        --once)
            WATCH=0
            shift
            ;;
        --watch)
            WATCH=1
            shift
            case "${1:-}" in '' | -*) : ;; *)
                INTERVAL="$1"
                shift
                ;;
            esac
            ;;
        --watch=*)
            WATCH=1
            INTERVAL="${1#*=}"
            shift
            ;;
        # --help prints the WHOLE leading comment block — derived, never a hardcoded line range. (It was
        # `sed -n '2,73p'`, which had already drifted past the header's end and silently swallowed the
        # `Env overrides:` paragraph; a help text that truncates as the file grows is the same class of bug
        # as a frame that truncates as the fleet grows.)
        -h | --help)
            awk 'NR<2{next} !/^#/{exit} {sub(/^# ?/,""); print}' "${BASH_SOURCE[0]}"
            exit 0
            ;;
        *)
            echo "fleet.sh: unknown flag '$1' (try --help)" >&2
            exit 2
            ;;
    esac
done
case "$INTERVAL" in '' | *[!0-9]*)
    echo "fleet.sh: --watch interval must be an integer (seconds)" >&2
    exit 2
    ;;
esac

# ── VIEW RESOLUTION ──────────────────────────────────────────────────────────────────────────────────
# `--watch` DEFAULTS to the hierarchy pane (CLUSTER → CAPACITY → NAMESPACE → INSTALLED/RUN·SERVER): the live
# view is exactly where you want structure, and having the better format be opt-in twice (`--watch --stages`)
# was silly. Escape hatch: `--flat` (alias `--no-stages`) keeps the flat active-runs-first pane under --watch.
# A flat-ONLY flag (--wide/--idle/--all/--failed) counts as choosing the flat pane, so `--watch --wide` still
# gets the pane that actually implements --wide instead of silently dropping it.
# The ONE-SHOT default stays FLAT: that pane is the forensic/scripting surface (terminal history with
# metric=value, --wide/--idle/--all/--failed, no idle-cluster collapse), and changing a one-shot command's
# stdout would break anything parsing it. `llmb-k8s fleet --stages` is one flag away.
if [ -z "$VIEW" ]; then
    if [ "$WATCH" = 1 ] && [ "$FLAT_ONLY_FLAG" = 0 ]; then VIEW="stages"; else VIEW="flat"; fi
fi
[ "$VIEW" = "stages" ] && STAGES=1 || STAGES=0

# ── --watch REQUIRES A TERMINAL ──────────────────────────────────────────────────────────────────────
# Watch is a REDRAW-IN-PLACE view: alternate screen buffer, cursor-home repaints, an UNBOUNDED frame loop.
# Piped to a file / a capture buffer / `head`, that is actively harmful — the consumer gets raw ANSI plus an
# ever-growing stack of near-identical frames, and whatever bounds the capture (a paste buffer, a tool's
# output cap, a log rotation) cuts the stream MID-TABLE. That truncation is what the "fleet --watch cuts off"
# report actually was. So: no TTY → exactly ONE clean full render, and SAY so on stderr. The pane choice
# (--watch defaults to --stages) is already resolved above, so you still get the view you asked for.
# Escape hatches for tests: FLEET_WATCH_ITERATIONS=N (an explicitly BOUNDED frame count) or FLEET_FORCE_WATCH=1.
if [ "$WATCH" = 1 ] && [ ! -t 1 ] && [ -z "${FLEET_WATCH_ITERATIONS:-}" ] && [ "${FLEET_FORCE_WATCH:-0}" != "1" ]; then
    echo "fleet.sh: --watch needs a terminal (stdout is not a TTY) — rendering ONE full frame instead." >&2
    WATCH=0
fi

# ── VIEWPORT (watch only) ────────────────────────────────────────────────────────────────────────────
# The alternate screen buffer has NO SCROLLBACK: a frame taller than the terminal scrolls its own top off
# into nothing — whole clusters disappear with no trace that they ever rendered. We therefore tell the
# renderer the live terminal geometry, and it COLLAPSES BY PRIORITY (idle clusters / build stamp / installed
# inventory / capacity first; live RUN·SERVER rows LAST) with an explicit "N lines hidden" accounting.
VIEWPORT_ROWS=0
VIEWPORT_COLS=0
LIVE_FOOTER_W=90 # width of the "  ↻ live · … · Ctrl-C to quit" line repaint() appends under the frame
if [ "$WATCH" = 1 ] && [ -t 1 ]; then
    exec 9<&1 # a stable dup of the REAL stdout: inside $(...) fd 1 is a pipe, so `stty size` must not use it
fi
_set_viewport() { # re-read EVERY frame so a terminal RESIZE is picked up (at most one frame of lag)
    local sz="" r="" c=""
    [ "$WATCH" = 1 ] && [ -t 1 ] || {
        VIEWPORT_ROWS=0
        VIEWPORT_COLS=0
        return 0
    }
    sz="$(stty size 2> /dev/null <&9)" || sz=""
    [ -z "$sz" ] && [ -r /dev/tty ] && { sz="$(stty size 2> /dev/null < /dev/tty)" || sz=""; }
    r="${sz%% *}"
    c="${sz##* }"
    case "$r" in '' | *[!0-9]*) r=0 ;; esac
    case "$c" in '' | *[!0-9]*) c=0 ;; esac
    # Reserve rows for the wrapping footer to avoid scrolling the alternate screen.
    local fw=1
    [ "$c" -gt 0 ] && fw=$(((LIVE_FOOTER_W + c - 1) / c))
    if [ "$r" -gt $((fw + 3)) ]; then VIEWPORT_ROWS=$((r - fw - 1)); else VIEWPORT_ROWS=0; fi
    VIEWPORT_COLS="$c"
    return 0
}

# color: auto → on only when stdout is a TTY and NO_COLOR is unset.
if [ "$COLOR" = "auto" ]; then
    if [ -t 1 ] && [ -z "${NO_COLOR:-}" ]; then COLOR="on"; else COLOR="off"; fi
fi
COLOR_FLAG="--no-color"
[ "$COLOR" = "on" ] && COLOR_FLAG="--color"

# Persistent (whole-process) scratch: TTL cache for cluster-scoped reads + last-good frames for laggards.
# Survives across --watch iterations; cleaned on exit.
PERSIST="$(mktemp -d "${TMPDIR:-/tmp}/fleetp.XXXXXX")"
CACHE_DIR="$PERSIST/cache"
STATE_DIR="$PERSIST/state"
mkdir -p "$CACHE_DIR" "$STATE_DIR"
TIMING_LOG="$PERSIST/timing.log"
GPID=""
_CLEANED=0
cleanup() {                       # idempotent; kills the background prefetch BEFORE removing $PERSIST so a
    [ "$_CLEANED" = 1 ] && return # still-running gather can't mktemp into (or write under) a deleted dir.
    _CLEANED=1
    [ -n "$GPID" ] && kill "$GPID" 2> /dev/null
    wait 2> /dev/null
    # leave the alternate screen buffer + restore the cursor so the terminal is exactly as we found it
    # (the pre-watch scrollback is intact — the live frames never leaked into it).
    [ "$WATCH" = 1 ] && printf '\033[?25h\033[?1049l'
    [ -n "${PERSIST:-}" ] && rm -rf "$PERSIST"
}
on_signal() {
    cleanup
    exit 130
} # signal path exits; EXIT-trap cleanup preserves the scripting exit code
trap cleanup EXIT
trap on_signal INT TERM

shopt -s nullglob
US=$'\x1f' # internal field separator for records/reg — NON-whitespace so `read` preserves EMPTY fields
# (an empty NAMESPACE/CONTEXT), which a whitespace IFS like $'\t' would silently collapse.

# ── read one KEY from a profile .env (quoted or bare-with-inline-comment; mirrors preflight.parse_env) ──
getval() { # <KEY> <file>
    awk -F= -v k="$1" '$1==k{sub(/^[^=]*=/,""); print; exit}' "$2" \
        | sed -e 's/^[[:space:]]*//' \
            -e 's/^"\([^"]*\)".*/\1/' -e "s/^'\([^']*\)'.*/\1/" \
            -e 's/[[:space:]]*#.*$//' -e 's/[[:space:]]*$//'
}

kubectl_get() { # <outfile> <errfile> <get-args...>   → returncode ; RTO caps the request timeout
    local out="$1" err="$2"
    shift 2
    "$KUBECTL" "$@" --request-timeout="${RTO:-$TIMEOUT}" -o json > "$out" 2> "$err"
}

# TTL-cached cluster-scoped read, keyed by context. Reuses the cache within its TTL (persisted across watch
# frames); on a fresh fetch failure it keeps any stale cache (better than n/a) but still marks the attempt so
# a Forbidden endpoint isn't hammered every frame. WORK/name/ctxargs come from the caller (dynamic scope).
ttl_cached_read() { # <name> <ckey> <kind nodes|all> <ttl> <get-args...>
    local nm="$1" ckey="$2" kind="$3" ttl="$4"
    shift 4
    local cache="$CACHE_DIR/ctx-$ckey.$kind.json" ts="$CACHE_DIR/ctx-$ckey.$kind.ts"
    local now tsv=0 age
    now="$(date +%s)"
    [ -f "$ts" ] && tsv="$(cat "$ts" 2> /dev/null || echo 0)"
    age=$((now - tsv))
    if [ "$age" -lt "$ttl" ]; then
        [ -f "$cache" ] && cp "$cache" "$WORK/$nm.$kind.json" # within TTL → reuse (or known-n/a: no file)
        return 0
    fi
    local tmp
    tmp="$(mktemp "${cache}.XXXXXX" 2> /dev/null)"
    [ -n "$tmp" ] || return 0 # $PERSIST reaped under a killed prefetch → skip rather than write to "/"
    local RTO="$TIMEOUT"
    if kubectl_get "$tmp" "$WORK/$nm.$kind.err" "$@"; then
        mv -f "$tmp" "$cache"
        echo "$now" > "$ts"
        cp "$cache" "$WORK/$nm.$kind.json"
    else
        rm -f "$tmp"
        echo "$now" > "$ts" # mark attempt; keep any prior stale cache
        [ -f "$cache" ] && cp "$cache" "$WORK/$nm.$kind.json"
    fi
}

# Probe ONE cluster (run in a background subshell): namespaced reads + optional cluster-scoped reads → write
# per-cluster json + a one-line meta, then touch <name>.done. Auth-robust: a failed probe writes an
# AUTH/UNREACH meta and returns — it never aborts the frame.
# Resolve a profile to a canonical PHYSICAL-cluster key so two profiles that target the same cluster (e.g.
# one pinning KUBE_CONTEXT="<proxy>.teleport.sh-<cluster>", another setting KUBE_CLUSTER="<cluster>") dedup
# to one row instead of showing as duplicates. KUBE_CLUSTER wins; else the cluster token of KUBE_CONTEXT
# (the part after the teleport proxy prefix `...teleport.sh-`); else the raw context.
cluster_key() { # <kube_cluster> <kube_context>
    local kc="$1" ctx="$2"
    if [ -n "$kc" ]; then
        printf '%s' "$kc"
        return
    fi
    case "$ctx" in
        *.teleport.sh-*) printf '%s' "${ctx##*.teleport.sh-}" ;;
        *) printf '%s' "$ctx" ;;
    esac
}

# <profile_files> is EVERY profile FILE (basename, comma-separated) that maps to this physical cluster — not
# just the alphabetical winner whose namespace we probe. The renderer resolves each against --profiles-dir and
# takes the UNION of their MODEL_CACHE_PVC resolutions: several profiles collapse onto one cluster row, and
# passing only the first hides the second's claim (the hazard discover_model_caches' own docstring warns of).
probe_cluster() { # <name> <ctx> <ns> <connect> <gpu_product> <profile_count> <profile_files>
    local name="$1" ctx="$2" ns="$3" connect="$4" gpu="$5" pcount="$6" pfiles="$7"
    local ctxargs=()
    [ -n "$ctx" ] && ctxargs=(--context "$ctx")
    local nsargs=()
    [ -n "$ns" ] && nsargs=(-n "$ns")
    local err="$WORK/$name.err" meta="$WORK/$name.meta"
    local ckey
    ckey="$(printf '%s' "${ctx:-_ambient}" | tr -c 'A-Za-z0-9' '_')"
    local t0 t1

    # ── OURS: ONE namespaced call for pods+deployments+jobs (was 3 separate calls → 1; each call over
    #    teleport pays a full TLS+auth handshake, so collapsing 3→1 is the biggest per-cluster win). Also the
    #    connectivity probe (short timeout). TTL-cached per (ctx,ns) so rapid --watch refreshes reuse it. ──
    local nsk="${ckey}__$(printf '%s' "${ns:-_all}" | tr -c 'A-Za-z0-9' '_')"
    local nscache="$CACHE_DIR/ns-$nsk.json" nsts="$CACHE_DIR/ns-$nsk.ts"
    local now tsv=0 age
    now="$(date +%s)"
    [ -f "$nsts" ] && tsv="$(cat "$nsts" 2> /dev/null || echo 0)"
    age=$((now - tsv))
    if [ -f "$nscache" ] && [ "$age" -lt "$NS_TTL" ]; then
        cp "$nscache" "$WORK/$name.nsread.json" # warm: reuse recent OURS read (no kubectl)
        _tlog "$name ns=cache(${age}s)"
    else
        [ "$TIMING" = 1 ] && t0="$(_now_ms)"
        local RTO="$PROBE_TIMEOUT" tmp
        tmp="$(mktemp "${nscache}.XXXXXX" 2> /dev/null)"
        [ -n "$tmp" ] || return 0 # $PERSIST reaped under a killed prefetch → skip rather than write to "/"
        if kubectl_get "$tmp" "$err" ${ctxargs[@]+"${ctxargs[@]}"} ${nsargs[@]+"${nsargs[@]}"} get pods,deployments,jobs; then
            mv -f "$tmp" "$nscache"
            echo "$now" > "$nsts"
            cp "$nscache" "$WORK/$name.nsread.json"
            [ "$TIMING" = 1 ] && _tlog "$name ns=live $(($(_now_ms) - t0))ms"
        else
            rm -f "$tmp"
            local low state="UNREACH"
            low="$(tr '[:upper:]' '[:lower:]' < "$err" 2> /dev/null)"
            case "$low" in
                *unauthorized* | *"exec plugin"* | *credential* | *"logged in"* | *expired* | *tsh* | *x509* | *certificate* | *"does not exist"* | *"no configuration has been provided"*)
                    state="AUTH"
                    ;;
                *"no such host"* | *"connection refused"* | *timeout* | *"timed out"* | *"unable to connect"* | *"i/o"* | *eof*)
                    state="UNREACH"
                    ;;
            esac
            _tlog "$name ns=FAIL($state)"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t\t%s\t%s\t%s\n' "$name" "$ctx" "$ns" "$state" "$err" "$connect" "$gpu" "$pcount" "$pfiles" > "$meta"
            : > "$WORK/$name.done"
            return 0
        fi
    fi

    if [ "$FAST" != 1 ]; then # --fast/--mine skips the slow capacity reads
        [ "$TIMING" = 1 ] && t1="$(_now_ms)"
        ttl_cached_read "$name" "$ckey" nodes "$NODES_TTL" ${ctxargs[@]+"${ctxargs[@]}"} get nodes
        ttl_cached_read "$name" "$ckey" all "$ALL_TTL" ${ctxargs[@]+"${ctxargs[@]}"} get pods -A
        # OUR workloads CLUSTER-WIDE (all namespaces), label-scoped to just ours → cheap + RBAC-friendly. This is
        # what lets the stages view attribute a disagg SERVER pod (its llmb labels live on the Deployment, not the
        # pod) to us in a per-worktree namespace, so a run LOADING in another ns is discovered, not invisible.
        ttl_cached_read "$name" "$ckey" allllmb "$ALL_TTL" ${ctxargs[@]+"${ctxargs[@]}"} \
            get deployments,jobs -A -l app.kubernetes.io/managed-by=llmb-recipe
        # MODEL-LOAD LOCK: the coordination Leases that serialize checkpoint loads against a shared model-cache
        # PVC (three concurrent ~500GB loads off one FSx measured 7.2x slower per shard than one alone). Label
        # -scoped to ours + cluster-wide, so it is one cheap call; the WAITERS come free from the allllmb Jobs
        # read above (their run-owner annotations), so this is the only extra read. RBAC-Forbidden → no file →
        # the renderer omits the MODEL LOAD line entirely rather than implying "nothing is loading".
        ttl_cached_read "$name" "$ckey" leases "$NS_TTL" ${ctxargs[@]+"${ctxargs[@]}"} \
            get leases -A -l llmb.nvidia.com/managed=true
        # MODEL CACHES: the PVCs that hold downloaded weights. Read from the CLUSTER, not from local .state stamps
        # or a profile's MODEL_CACHE_PVC — a model downloaded from another worktree, by a colleague, or before a
        # fresh clone must still be visible ("is it downloaded?" has to be answerable without kubectl). Reading
        # PVCs also gives the UNION of caches actually present in a namespace, which is what several profiles
        # mapping to one cluster+ns hides. Cluster-wide + TTL-cached; RBAC-Forbidden → no file → rows omitted.
        # WHY -A AND NOT `-n $ns`: the namespaces we manage are not all known here — beyond the profile's
        # configured ns they are DISCOVERED from the allllmb read above (per-worktree namespaces), so a per-ns
        # read would either miss them or need a second serialized round of calls (the dominant cost is the
        # per-call handshake, not the payload). One cached -A call stays cheapest. The renderer then SCOPES the
        # result to OUR namespaces (fleet_render.discover_model_caches) so a colleague's PVC is never rendered —
        # a shared cluster returns 469 PVCs across 229 namespaces and only ours are ours to report on.
        ttl_cached_read "$name" "$ckey" pvcs "$ALL_TTL" ${ctxargs[@]+"${ctxargs[@]}"} get pvc -A
        [ "$TIMING" = 1 ] && _tlog "$name cluster-scoped(nodes+all+allllmb+leases+pvcs) $(($(_now_ms) - t1))ms"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t\t%s\t%s\t%s\n' "$name" "$ctx" "$ns" "OK" "" "$connect" "$gpu" "$pcount" "$pfiles" > "$meta"
    : > "$WORK/$name.done"
}

save_last_good() { # <name> <ts> — snapshot a fresh frame for laggard fallback next frame
    local n="$1" ts="$2" k
    for k in nsread nodes all allllmb leases pvcs; do
        [ -f "$WORK/$n.$k.json" ] && cp "$WORK/$n.$k.json" "$STATE_DIR/$n.$k.json"
    done
    cp "$WORK/$n.meta" "$STATE_DIR/$n.meta" 2> /dev/null
    echo "$ts" > "$STATE_DIR/$n.goodts"
}

restore_last_good() { # <name>
    local n="$1" k
    for k in nsread nodes all allllmb leases pvcs; do
        [ -f "$STATE_DIR/$n.$k.json" ] && cp "$STATE_DIR/$n.$k.json" "$WORK/$n.$k.json"
    done
}

wait_for_probes() { # <deadline-secs|""> — empty = wait for all; else poll until all done or deadline
    local deadline="$1"
    if [ -z "$deadline" ]; then
        wait
        return
    fi
    local start done_all nm
    start="$(date +%s)"
    while :; do
        done_all=1
        while IFS= read -r nm; do
            [ -z "$nm" ] && continue
            [ -f "$WORK/$nm.done" ] || {
                done_all=0
                break
            }
        done << EOF
$LAUNCHED
EOF
        [ "$done_all" = 1 ] && break
        [ $(($(date +%s) - start)) -ge "$deadline" ] && break
        sleep 0.2
    done
}

# ── gather_and_render: compute ONE complete frame (all clusters, IN PARALLEL) to stdout, then clean up. ──
gather_and_render() { # [frame-deadline-secs]
    local deadline="${1:-}"
    # $PERSIST can vanish under a background prefetch if Ctrl-C fired and the EXIT trap reaped it mid-gather.
    # 2>/dev/null + the guard below mean we ABORT cleanly (return 1) instead of writing every path under "/".
    local WORK
    WORK="$(mktemp -d "$PERSIST/frame.XXXXXX" 2> /dev/null)"
    [ -n "$WORK" ] && [ -d "$WORK" ] || return 1
    local META="$WORK/meta.tsv"
    : > "$META"
    local _fstart
    [ "$TIMING" = 1 ] && {
        _fstart="$(_now_ms)"
        : > "$TIMING_LOG"
    }

    local profiles=("$PROFILES_DIR"/*.env)
    [ ${#profiles[@]} -eq 0 ] && profiles=("$PROFILES_DIR"/*.env.example)
    if [ ${#profiles[@]} -eq 0 ]; then
        echo "fleet: no cluster profiles in $PROFILES_DIR — create one: llmb-k8s profile init --cluster <name>" >&2
        rm -rf "$WORK"
        return 1
    fi
    # profile-NAME order (so `alpha` beats `alpha-dup`). No `declare -A` (bash 3.2).
    local namelist="" entry b n
    for f in "${profiles[@]}"; do
        b="$(basename "$f")"
        n="${b%.env}"
        n="${n%.env.example}"
        namelist="${namelist}${n}	${f}"$'\n'
    done
    local sorted
    IFS=$'\n' sorted=($(printf '%s' "$namelist" | sort))
    unset IFS

    # ── PASS 1: resolve every profile to its PHYSICAL-CLUSTER key and record it, so ALL profiles that target
    #    the same physical cluster collapse to ONE row — even when they pin DIFFERENT namespaces (a per-worktree
    #    per-worktree namespace split). The key is the cluster identity ALONE (not cluster|ns): cross-namespace run
    #    discovery (pods -A) already surfaces every namespace's runs under the one section, so a second
    #    same-cluster profile would only render a DUPLICATE section listing the same cluster-wide runs again.
    #    A record is `key \t name \t ns \t ctx \t connect \t gpu_product \t profile_file`, in profile-name
    #    order. The profile FILE rides along so PASS 2 can hand the renderer EVERY profile that maps to the
    #    cluster (the model ledger resolves each one's MODEL_CACHE_PVC and matches on the union). ───────────
    local records="" name f ctx ns connect gpu kc keyns
    for entry in "${sorted[@]}"; do
        name="${entry%%	*}"
        f="${entry#*	}"
        [ -n "$FILTER" ] && [ "$name" != "$FILTER" ] && continue
        ctx="$(getval KUBE_CONTEXT "$f")"
        ns="$(getval NAMESPACE "$f")"
        connect="$(getval CONNECT_CMD "$f")"
        gpu="$(getval GPU_PRODUCT "$f")"
        kc="$(getval KUBE_CLUSTER "$f")"
        keyns="$(cluster_key "$kc" "$ctx")"
        # An ambient-context profile (no KUBE_CLUSTER + no KUBE_CONTEXT) has no cluster identity to dedup on —
        # key it by its own profile name so it stays a distinct row (and is never skipped by the blank-key guard).
        [ -z "$keyns" ] && keyns="$name"
        # `${f##*/}`, NOT `$(basename "$f")`: this loop runs once per PROFILE on EVERY frame, and a fork per
        # profile is measurable against the 4s FLEET_NS_TTL that lets two rapid --watch frames reuse one read.
        records="${records}${keyns}${US}${name}${US}${ns}${US}${ctx}${US}${connect}${US}${gpu}${US}${f##*/}"$'\n'
    done

    # ── PASS 2: launch ONE probe per unique PHYSICAL CLUSTER — the first profile alphabetically wins (its
    #    namespace is the section's single-ns read; the others' namespaces come in via cross-ns discovery) —
    #    passing the count of profiles that map to it. IN PARALLEL (serialized if FLEET_SEQUENTIAL=1). Each bg
    #    probe's stdout/stderr is redirected away from the frame's command-substitution pipe, else a slow
    #    probe holding it open would block `$(gather_and_render)` past the deadline. ───────────────────────
    local seen_keys=$'\n' LAUNCHED="" PIDS="" running=0 pcount pfiles pinfo
    while IFS="$US" read -r keyns name ns ctx connect gpu pfile; do
        [ -z "$keyns" ] && continue
        case "$seen_keys" in *$'\n'"$keyns"$'\n'*) continue ;; esac
        seen_keys="${seen_keys}${keyns}"$'\n'
        # ONE awk pass yields BOTH the profile COUNT and the comma-joined profile FILES for this cluster key —
        # they are two views of the same set and must never disagree. Split in the shell rather than with a
        # second `$( )` or a heredoc: this runs per cluster on EVERY frame, and the 4s FLEET_NS_TTL that lets
        # two rapid --watch frames reuse one namespaced read leaves little room for avoidable forks.
        pinfo="$(printf '%s' "$records" | awk -F"$US" -v k="$keyns" -v us="$US" \
            '$1==k{ c++; if (s!="") s=s","; s=s $7 } END{ printf "%d%s%s", c+0, us, s }')"
        pcount="${pinfo%%"$US"*}"
        pfiles="${pinfo#*"$US"}"
        printf '%s%s%s%s%s%s%s%s%s%s%s%s%s\n' "$name" "$US" "$ctx" "$US" "$ns" "$US" "$connect" "$US" "$gpu" "$US" "$pcount" "$US" "$pfiles" > "$WORK/$name.reg"
        LAUNCHED="${LAUNCHED}${name}"$'\n'
        if [ "${FLEET_SEQUENTIAL:-0}" = 1 ]; then
            probe_cluster "$name" "$ctx" "$ns" "$connect" "$gpu" "$pcount" "$pfiles" > /dev/null 2>&1
        else
            probe_cluster "$name" "$ctx" "$ns" "$connect" "$gpu" "$pcount" "$pfiles" > /dev/null 2>&1 &
            PIDS="$PIDS $!"
            running=$((running + 1))
            # Concurrency cap: a blocking batch `wait` would defeat the --watch per-frame deadline (it waits for a
            # laggard), so only enforce the cap when there is NO deadline (single-shot waits for all anyway).
            [ -z "$deadline" ] && [ "$running" -ge "$MAX_JOBS" ] && {
                wait
                running=0
            }
        fi
    done << EOF
$records
EOF

    if [ -z "$LAUNCHED" ]; then
        echo "fleet: no clusters matched${FILTER:+ --cluster $FILTER}" >&2
        rm -rf "$WORK"
        return 1
    fi
    if [ "${FLEET_SEQUENTIAL:-0}" != 1 ]; then
        wait_for_probes "$deadline"
        # past a frame deadline, stop still-running laggards so they don't pile up across watch frames
        if [ -n "$deadline" ]; then
            local p
            for p in $PIDS; do kill "$p" 2> /dev/null; done
        fi
    fi
    [ "$TIMING" = 1 ] && _tlog "GATHER total $(($(_now_ms) - _fstart))ms (parallel over all clusters)"

    # ── assemble META in stable order: fresh clusters as-is; laggards from last-good (marked stale); a
    #    never-yet-seen laggard as a PENDING placeholder. ────────────────────────────────────────────────
    local now goodts age rn rctx rns rconn rgpu rcount rpfiles
    now="$(date +%s)"
    while IFS= read -r name; do
        [ -z "$name" ] && continue
        if [ -f "$WORK/$name.done" ]; then
            cat "$WORK/$name.meta" >> "$META"
            save_last_good "$name" "$now"
        elif [ -f "$STATE_DIR/$name.meta" ] \
            && {
                goodts="$(cat "$STATE_DIR/$name.goodts" 2> /dev/null || echo "$now")"
                age=$((now - goodts))
                [ "$LAST_GOOD_MAX" -gt 0 ] && [ "$age" -le "$LAST_GOOD_MAX" ]
            }; then
            # within the ceiling → show the last good frame, with its AGE stamped into field 7 so the renderer
            # marks the whole cluster ⚠ STALE. Never presented as live.
            restore_last_good "$name"
            awk -F'\t' -v age="$age" 'BEGIN{OFS="\t"} {$7=age; print}' "$STATE_DIR/$name.meta" >> "$META"
        else
            IFS="$US" read -r rn rctx rns rconn rgpu rcount rpfiles < "$WORK/$name.reg"
            printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$name" "$rctx" "$rns" "PENDING" "" "$rconn" "pending" "$rgpu" "$rcount" "$rpfiles" >> "$META"
        fi
    done << EOF
$LAUNCHED
EOF

    local extra=()
    [ "$WIDE" = 1 ] && extra+=(--wide)
    [ "$GPU_ONLY" = 1 ] && extra+=(--gpu-only)
    [ "$IDLE" = 1 ] && extra+=(--idle)
    [ "$FAST" = 1 ] && extra+=(--fast)
    [ "$DETAIL" = 1 ] && extra+=(--detail)
    [ "$ALL" = 1 ] && extra+=(--all)
    [ "$FAILED" = 1 ] && extra+=(--failed)
    # --profiles-dir is passed for EVERY pane, not just --stages: the model ledger resolves each cluster's
    # MODEL_CACHE_PVC out of the profile FILES named in meta field 10, and without the dir it cannot tell
    # "this claim is empty" from "I do not know which claim these weights belong in".
    extra+=(--profiles-dir "$PROFILES_DIR")
    [ "$STAGES" = 1 ] && extra+=(--stages)
    [ -n "$HISTORY" ] && extra+=(--history "$HISTORY")
    [ -n "${FLEET_NOW:-}" ] && extra+=(--now "$FLEET_NOW")
    [ -n "$BUILD" ] && extra+=(--build "$BUILD")
    [ "${VIEWPORT_ROWS:-0}" -gt 0 ] && extra+=(--viewport-rows "$VIEWPORT_ROWS")
    [ "${VIEWPORT_COLS:-0}" -gt 0 ] && extra+=(--viewport-cols "$VIEWPORT_COLS")
    local _rstart
    [ "$TIMING" = 1 ] && _rstart="$(_now_ms)"
    python3 "$SCRIPT_DIR/fleet_render.py" --meta "$META" --workdir "$WORK" "$COLOR_FLAG" \
        ${extra[@]+"${extra[@]}"}
    local rc=$?
    if [ "$TIMING" = 1 ]; then
        _tlog "RENDER (python parse+format) $(($(_now_ms) - _rstart))ms"
        _tlog "FRAME total $(($(_now_ms) - _fstart))ms"
        cat "$TIMING_LOG" >&2
    fi
    rm -rf "$WORK"
    return $rc
}

# ── single-shot (default) vs --watch (double-buffered) ─────────────────────────────────────────────────
if [ "$WATCH" != 1 ]; then
    gather_and_render # print one complete frame and exit (scripting-friendly)
    exit $?
fi

# --watch: double-buffered AND PIPELINED. The first frame is gathered synchronously (cold). Every later
# frame is PRE-GATHERED in the background DURING the previous refresh interval, so when the interval fires
# the data is already warm → the repaint is instant (the gather latency is hidden behind the sleep).
# REDRAW-IN-PLACE like top(1)/watch(1): we run on the ALTERNATE SCREEN BUFFER (\033[?1049h) and cursor-home +
# clear-to-end (\033[H\033[J) each frame, so frames REFRESH in place instead of scrolling/appending, and the
# pre-watch terminal (scrollback) is restored untouched on exit. A per-frame deadline bounds a laggard.
FRAME_DEADLINE="${FLEET_FRAME_DEADLINE:-$INTERVAL}"
printf '\033[?1049h\033[?25l\033[H\033[J' # enter alt-screen · hide cursor · home+clear (cleanup restores)
PF="$PERSIST/prefetch.frame"
repaint() { # <frame> <note>
    local stamp
    stamp="$(date -u +%H:%M:%SZ)"
    printf '\033[H\033[J%s\n  ↻ live · %s · refresh %ss · updated %s · Ctrl-C to quit\n' "$1" "$2" "$INTERVAL" "$stamp"
}
iter=0
next_frame=""
gnote="gathered now"
# A RESIZE makes the prefetched frame (fitted to the OLD geometry) wrong — drop it and re-gather.
_WINCH=0
trap '_WINCH=1' WINCH
while :; do
    [ "$_WINCH" = 1 ] && {
        _WINCH=0
        next_frame=""
    }
    _set_viewport
    if [ -z "$next_frame" ]; then
        t0=$(date +%s)
        frame="$(gather_and_render "$FRAME_DEADLINE")"
        gnote="gathered in $(($(date +%s) - t0))s"
    else
        frame="$next_frame"
        next_frame=""
        gnote="prefetched (instant)"
    fi
    repaint "$frame" "$gnote"
    iter=$((iter + 1))
    if [ -n "${FLEET_WATCH_ITERATIONS:-}" ] && [ "$iter" -ge "$FLEET_WATCH_ITERATIONS" ]; then
        break
    fi
    # PIPELINE: gather the NEXT frame in the background while we sleep out the interval, then collect it.
    # GPID is global so the signal/EXIT trap can kill this prefetch before removing $PERSIST (no /-writes).
    _set_viewport # geometry for the PREFETCHED frame, re-read at prefetch time
    (gather_and_render "$FRAME_DEADLINE" > "$PF" 2> /dev/null) &
    GPID=$!
    sleep "$INTERVAL"
    wait "$GPID" 2> /dev/null
    GPID=""
    next_frame="$(cat "$PF" 2> /dev/null)"
    rm -f "$PF"
done
