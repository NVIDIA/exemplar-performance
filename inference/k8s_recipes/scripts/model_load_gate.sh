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

# Serialize the model-load phase for runs that share a model-cache PVC. The lease is keyed by PVC, released
# after the server becomes Ready, and expires if its holder stops renewing it. Runs using different PVCs are
# independent.
#
#   acquire <cell> <profile> <run-id> [--run-owner <job>] [--timeout S] [--lease-seconds S]
#   release <cell> <profile> <run-id> [--run-owner <job>]
#   cache-name <cell> <profile>          # resolve + print the cache PVC this cell will use
#
# LOCK OBJECT (contract — fleet reads these):
#   coordination.k8s.io/v1 Lease  llmb-modelload-<sanitised-pvc>   in the run namespace
#     spec.holderIdentity        = run-id currently loading
#     spec.leaseDurationSeconds  = TTL; renewed while loading  → crash-safe (a dead holder expires)
#     labels: llmb.nvidia.com/managed=true, llmb.nvidia.com/model-cache=<pvc>
#
# WAITER/HOLDER VISIBILITY (contract — annotations on the run-owner Job):
#   llmb.nvidia.com/model-load-wait   = <pvc>      (queued on this cache)
#   llmb.nvidia.com/model-load-since  = <RFC3339>  (when the wait started)
#   llmb.nvidia.com/model-load-holder = <pvc>      (holds the slot)
#   Both wait/since are CLEARED on acquire and on release, so a stale annotation can never make fleet
#   report a phantom queue.
#
# If the coordination API is unavailable or access is denied, warn and proceed without the gate. Set
# LLMB_LOAD_GATE_TIMEOUT_S=0 to make the gate advisory.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LEASE_SECONDS="${LLMB_LOAD_GATE_LEASE_S:-300}"   # TTL; renewed at 1/3 of this while holding
WAIT_TIMEOUT="${LLMB_LOAD_GATE_TIMEOUT_S:-5400}" # max wait for the slot before proceeding anyway (0 = never block)
POLL_S="${LLMB_LOAD_GATE_POLL_S:-10}"

ANN_WAIT="llmb.nvidia.com/model-load-wait"
ANN_SINCE="llmb.nvidia.com/model-load-since"
ANN_HOLDER="llmb.nvidia.com/model-load-holder"
LABEL_MANAGED="llmb.nvidia.com/managed"
LABEL_CACHE="llmb.nvidia.com/model-cache"

log() { echo "[load-gate $(date -u +%H:%M:%S)] $*"; }
warn() { echo "load-gate: $*" >&2; }

# ── cache-name resolution ────────────────────────────────────────────────────────────────────────────
# The contended resource is the cell's ACTUAL model-cache PVC. With per-recipe caches that is derived
# (install.derive_recipe_cache), not simply MODEL_CACHE_PVC — so ask install.py rather than guessing.
resolve_cache_pvc() {
    local cell="$1" profile="$2" out
    # Explicit override: lets an operator point the gate at the real contended claim when the derivation
    # cannot see it (and gives the tests a seam that does not require install.py + a full profile).
    if [ -n "${LLMB_MODEL_CACHE_OVERRIDE:-}" ]; then
        printf '%s' "$LLMB_MODEL_CACHE_OVERRIDE"
        return 0
    fi
    # Use the shared cache resolver; an empty result disables the gate without inventing a claim.
    out="$(python3 "$ROOT/scripts/model_cache.py" resolve "$cell" "$ROOT/cluster-profiles/${profile}.env" 2> /dev/null)" || return 0
    case "$out" in
        *[!a-z0-9-]* | "") return 0 ;;
    esac
    printf '%s' "$out"
}

# RFC-1123 object name from a PVC name; hash-suffix fallback when it cannot be made valid.
lease_name_for() {
    local pvc="$1"
    local s
    s="$(printf '%s' "$pvc" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-' | sed 's/^-*//; s/-*$//')"
    if [ -z "$s" ] || [ ${#s} -gt 200 ]; then
        local h
        h="$(printf '%s' "$pvc" | shasum -a 256 2> /dev/null | cut -c1-10)"
        [ -z "$h" ] && h="$(printf '%s' "$pvc" | sha256sum | cut -c1-10)"
        warn "cache name '$pvc' is not a valid object-name component — using hash suffix $h"
        s="h${h}"
    fi
    printf 'llmb-modelload-%s' "$s"
}

# ── annotation helpers (best-effort; never fatal) ────────────────────────────────────────────────────
annotate_owner() { # annotate_owner <ns> <job> <k=v>...
    local ns="$1" job="$2"
    shift 2
    [ -n "$job" ] || return 0
    kc -n "$ns" annotate job "$job" --overwrite "$@" > /dev/null 2>&1 || true
}

main() {
    local verb="${1:?usage: model_load_gate.sh <acquire|release|cache-name> <cell> <profile> [run-id] [flags]}"
    local cell="${2:?cell}" profile="${3:?profile}"
    shift 3
    local run_id="${1:-}"
    [ $# -gt 0 ] && shift || true
    local run_owner=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --run-owner)
                run_owner="${2:-}"
                shift 2
                ;;
            --timeout)
                WAIT_TIMEOUT="${2:-}"
                shift 2
                ;;
            --lease-seconds)
                LEASE_SECONDS="${2:-}"
                shift 2
                ;;
            *) shift ;;
        esac
    done

    local envf="$ROOT/cluster-profiles/${profile}.env"
    [ -f "$envf" ] || {
        warn "no profile at $envf — proceeding without the load gate"
        return 0
    }
    set -a
    . "$envf"
    set +a
    kc() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

    local pvc
    pvc="$(resolve_cache_pvc "$cell" "$profile")"
    if [ -z "$pvc" ]; then
        [ "$verb" = "cache-name" ] && {
            echo ""
            return 0
        }
        warn "could not resolve a model-cache PVC for this cell — proceeding without the load gate"
        return 0
    fi
    [ "$verb" = "cache-name" ] && {
        echo "$pvc"
        return 0
    }

    local lease
    lease="$(lease_name_for "$pvc")"
    local ns="${NAMESPACE:?NAMESPACE missing from profile}"

    case "$verb" in
        acquire) do_acquire "$ns" "$lease" "$pvc" "$run_id" "$run_owner" ;;
        release) do_release "$ns" "$lease" "$pvc" "$run_id" "$run_owner" ;;
        *)
            warn "unknown verb '$verb'"
            return 0
            ;;
    esac
}

# Can we use Leases at all? Probe once; RBAC denial or an unreachable API ⇒ degrade forward.
lease_backend_ok() {
    local ns="$1"
    local out rc
    out="$(kc -n "$ns" get lease -o name 2>&1)"
    rc=$?
    if [ $rc -ne 0 ]; then
        if printf '%s' "$out" | grep -qiE 'forbidden|cannot (get|list)|unauthorized|RBAC'; then
            warn "⚠ cannot coordinate model loads: leases are FORBIDDEN in namespace '$ns'."
            warn "   Proceeding WITHOUT the load gate — concurrent runs sharing a model cache may contend for"
            warn "   read bandwidth and load slowly. Grant get/create/update on coordination.k8s.io/leases to restore it."
        else
            warn "⚠ cannot reach the Lease API in '$ns' — proceeding WITHOUT the load gate."
        fi
        return 1
    fi
    return 0
}

# Is the lease currently held by someone ELSE and NOT expired? echoes the holder when it is.
lease_active_holder() {
    local ns="$1" lease="$2" me="$3"
    local blob holder renew dur
    blob="$(kc -n "$ns" get lease "$lease" -o json 2> /dev/null)" || return 1
    [ -n "$blob" ] || return 1
    holder="$(printf '%s' "$blob" | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("spec") or {}).get("holderIdentity") or "")' 2> /dev/null)"
    renew="$(printf '%s' "$blob" | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("spec") or {}).get("renewTime") or (d.get("spec") or {}).get("acquireTime") or "")' 2> /dev/null)"
    dur="$(printf '%s' "$blob" | python3 -c 'import json,sys;d=json.load(sys.stdin);print((d.get("spec") or {}).get("leaseDurationSeconds") or 0)' 2> /dev/null)"
    [ -n "$holder" ] || return 1
    [ "$holder" = "$me" ] && return 1 # our own lease is not a blocker
    # expired? (crash-safe: a dead holder frees the slot)
    python3 - "$renew" "$dur" << 'PY' || return 1
import sys, datetime
renew, dur = sys.argv[1], sys.argv[2]
try:
    t = datetime.datetime.strptime(renew.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S.%f%z")
except Exception:
    try: t = datetime.datetime.strptime(renew.replace("Z", "+0000"), "%Y-%m-%dT%H:%M:%S%z")
    except Exception: sys.exit(0)      # unparsable → treat as ACTIVE (conservative)
age = (datetime.datetime.now(datetime.timezone.utc) - t).total_seconds()
sys.exit(0 if age <= float(dur or 0) else 1)          # 0 = still active
PY
    printf '%s' "$holder"
    return 0
}

write_lease() { # write_lease <ns> <lease> <pvc> <holder>
    local ns="$1" lease="$2" pvc="$3" me="$4" now
    now="$(date -u +%Y-%m-%dT%H:%M:%S.000000Z)"
    printf '%s\n' "apiVersion: coordination.k8s.io/v1
kind: Lease
metadata:
  name: ${lease}
  namespace: ${ns}
  labels:
    ${LABEL_MANAGED}: \"true\"
    ${LABEL_CACHE}: \"${pvc}\"
spec:
  holderIdentity: ${me}
  leaseDurationSeconds: ${LEASE_SECONDS}
  acquireTime: \"${now}\"
  renewTime: \"${now}\"" | kc -n "$ns" apply -f - > /dev/null 2>&1
}

do_acquire() {
    local ns="$1" lease="$2" pvc="$3" me="$4" owner="$5"
    lease_backend_ok "$ns" || {
        annotate_owner "$ns" "$owner" "${ANN_WAIT}-" "${ANN_SINCE}-"
        return 0
    }

    local start
    start="$(date -u +%s)"
    local announced=0
    while :; do
        local holder
        if ! holder="$(lease_active_holder "$ns" "$lease" "$me")"; then
            if write_lease "$ns" "$lease" "$pvc" "$me"; then
                annotate_owner "$ns" "$owner" "${ANN_WAIT}-" "${ANN_SINCE}-" "${ANN_HOLDER}=${pvc}"
                log "model-load slot ACQUIRED on cache '$pvc' (lease $lease)"
                start_renewer "$ns" "$lease" "$pvc" "$me" "$run_owner"
                return 0
            fi
            warn "⚠ could not write the model-load lease — proceeding WITHOUT the load gate."
            annotate_owner "$ns" "$owner" "${ANN_WAIT}-" "${ANN_SINCE}-"
            return 0
        fi
        if [ "$announced" -eq 0 ]; then
            announced=1
            annotate_owner "$ns" "$owner" "${ANN_WAIT}=${pvc}" "${ANN_SINCE}=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
            log "waiting for model-load slot on cache '$pvc' — another run is loading (holder=${holder})"
        fi
        if [ "${WAIT_TIMEOUT:-0}" != "0" ] && [ $(($(date -u +%s) - start)) -ge "${WAIT_TIMEOUT}" ]; then
            warn "⚠ waited ${WAIT_TIMEOUT}s for the model-load slot on '$pvc' (holder=${holder}) — proceeding ANYWAY."
            warn "   The load may contend for cache bandwidth. Raise LLMB_LOAD_GATE_TIMEOUT_S to wait longer."
            annotate_owner "$ns" "$owner" "${ANN_WAIT}-" "${ANN_SINCE}-"
            return 0
        fi
        sleep "$POLL_S"
    done
}

# Renew in the background while we hold the slot.
#
# Stop renewal when any of these conditions is met:
#   1. sentinel removed            → normal `release`
#   2. run-owner Job GONE/terminal → the run is over; release the slot immediately (an UNAMBIGUOUS read;
#                                    matches the run-owner watcher's ownership-not-name principle)
#   3. MAX_HOLD elapsed            → backstop for the no-owner / unreadable-API case
#
# On an AMBIGUOUS owner read (API error, RBAC) it keeps renewing — briefly holding a slot is better than
# releasing while a real load is in flight — but the MAX_HOLD backstop bounds even that, because "hold
# forever on a read error" is precisely the failure this comment exists to prevent.
MAX_HOLD_S="${LLMB_LOAD_GATE_MAX_HOLD_S:-7200}" # 2h: a model load longer than this is pathological
start_renewer() {
    local ns="$1" lease="$2" pvc="$3" me="$4" owner="${5:-}"
    local sent="${TMPDIR:-/tmp}/.llmb-loadgate-${lease}-${me}"
    : > "$sent"
    (
        started=$(date +%s)
        # The subshell's stdout/stderr go to /dev/null (it outlives its caller's terminal), so every decision
        # this renewer makes is appended HERE. A slot that gets released must never be released silently.
        rlog="${sent}.log"
        say() { echo "[load-gate-renewer $(date -u +%FT%TZ)] $*" >> "$rlog" 2> /dev/null || true; }
        say "renewing lease '$lease' for run '$me' on cache '$pvc' (owner='${owner:-<none>}', max_hold=${MAX_HOLD_S}s)"
        while [ -f "$sent" ]; do
            sleep $((LEASE_SECONDS / 3 > 0 ? LEASE_SECONDS / 3 : 10))
            [ -f "$sent" ] || break

            # (3) absolute backstop
            if [ $(($(date +%s) - started)) -ge "$MAX_HOLD_S" ]; then
                say "MAX_HOLD (${MAX_HOLD_S}s) reached — RELEASING the slot on '$pvc' so it cannot be held forever"
                do_release "$ns" "$lease" "$pvc" "$me" "$owner" || true
                break
            fi

            # (2) ownership: release as soon as the owning run is unambiguously over
            if [ -n "$owner" ]; then
                owner_json="$(kc -n "$ns" get job "$owner" -o json 2> /dev/null)"
                owner_rc=$?
                if [ "$owner_rc" -eq 0 ] && [ -n "$owner_json" ]; then
                    if printf '%s' "$owner_json" | python3 -c 'import json,sys
s=(json.load(sys.stdin).get("status") or {})
sys.exit(0 if (s.get("succeeded") or 0) or (s.get("failed") or 0) else 1)' 2> /dev/null; then
                        say "run-owner '$owner' is TERMINAL — RELEASING the model-load slot on '$pvc'"
                        do_release "$ns" "$lease" "$pvc" "$me" "$owner" || true
                        break
                    fi
                elif kc -n "$ns" get job "$owner" > /dev/null 2>&1; then
                    : # readable but json failed — ambiguous, keep holding
                else
                    # distinguish "definitively absent" from "cannot talk to the API at all"
                    if kc -n "$ns" get jobs > /dev/null 2>&1; then
                        say "run-owner '$owner' is GONE — RELEASING the model-load slot on '$pvc'"
                        do_release "$ns" "$lease" "$pvc" "$me" "$owner" || true
                        break
                    fi
                fi
            fi

            write_lease "$ns" "$lease" "$pvc" "$me" || true
        done
    ) > /dev/null 2>&1 &
    echo $! > "${sent}.pid" 2> /dev/null || true
}

do_release() { # idempotent: safe to call when we never held it
    local ns="$1" lease="$2" pvc="$3" me="$4" owner="$5"
    local sent="${TMPDIR:-/tmp}/.llmb-loadgate-${lease}-${me}"
    rm -f "$sent" 2> /dev/null || true
    if [ -f "${sent}.pid" ]; then
        kill "$(cat "${sent}.pid" 2> /dev/null)" 2> /dev/null || true
        rm -f "${sent}.pid" 2> /dev/null || true
    fi
    # only delete a lease we actually hold — never steal another run's slot
    local holder
    holder="$(kc -n "$ns" get lease "$lease" -o jsonpath='{.spec.holderIdentity}' 2> /dev/null)" || holder=""
    if [ -n "$holder" ] && [ "$holder" = "$me" ]; then
        kc -n "$ns" delete lease "$lease" > /dev/null 2>&1 || true
        log "model-load slot RELEASED on cache '$pvc'"
    fi
    annotate_owner "$ns" "$owner" "${ANN_WAIT}-" "${ANN_SINCE}-" "${ANN_HOLDER}-"
    return 0
}

main "$@"
