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

# shellcheck shell=bash
# _kubectl_resilient.sh provides bounded retries for transient kubectl and authentication failures.
# Re-authentication uses LLMB_REAUTH_HOOK, CONNECT_CMD, or a supported client discovered on PATH.
# Persistent failures return LLMB_RC_TRANSIENT so callers do not mistake an unknown state for completion.
# This file is sourced by client-side wait and fetch commands.

# Guard against double-source (callers may source both _lib.sh and this).
[ -n "${_LLMB_KUBECTL_RESILIENT_SOURCED:-}" ] && return 0 2> /dev/null || true
_LLMB_KUBECTL_RESILIENT_SOURCED=1

# Return code for an unresolved transient auth or network failure.
LLMB_RC_TRANSIENT=97

# Last captured stderr from llmb::kc_resilient (so callers can classify a genuine error, e.g. NotFound).
LLMB_KC_STDERR=""

# Underlying kubectl, pinned to the profile context. KUBECTL override → offline shim (mirrors run_owner.sh).
llmb::kc_raw() { "${KUBECTL:-kubectl}" ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }

# Classify kubectl stderr as auth, transient connectivity, or another error.
llmb::classify_kube_err() {
    local low
    low="$(printf '%s' "${1:-}" | tr '[:upper:]' '[:lower:]')"
    case "$low" in
        *"you must be logged in"* | *unauthorized* | *forbidden* | *"error: forbidden"* | \
            *"certificate has expired"* | *"certificate is expired"* | *"certificate has expired or is not yet valid"* | \
            *"x509"* | *"credentials"* | *"invalid bearer token"* | *"the token is expired"* | *"token has expired"* | \
            *"please enter"* | *"login again"* | *"tsh: relogin"* | *"401"* | *"403"*)
            echo auth
            return 0
            ;;
    esac
    case "$low" in
        *"unable to connect to the server"* | *"connection refused"* | *"i/o timeout"* | *"tls handshake timeout"* | \
            *"error dialing backend"* | *"context deadline exceeded"* | *"deadline exceeded"* | *"eof"* | \
            *"service unavailable"* | *"503"* | *"the server is currently unable"* | *"connection reset"* | \
            *"no route to host"* | *"temporary failure in name resolution"* | *"timed out"* | *"timeout"*)
            echo transient
            return 0
            ;;
    esac
    echo other
}

# Print each recovery notice once.
_llmb_reauth_notice() {
    local msg="$1"
    case "${_LLMB_REAUTH_NOTICES:-}" in *"[$msg]"*) return 0 ;; esac
    _LLMB_REAUTH_NOTICES="${_LLMB_REAUTH_NOTICES:-}[$msg]"
    printf '  ⚠ %s\n' "$msg" >&2
}

# Resolve the profile's re-auth command (portable). Prints it; empty if none is configured.
llmb::reauth_cmd() {
    if [ -n "${LLMB_REAUTH_HOOK:-}" ]; then
        printf '%s' "$LLMB_REAUTH_HOOK"
        return 0
    fi
    if [ -n "${CONNECT_CMD:-}" ]; then
        printf '%s' "$CONNECT_CMD"
        return 0
    fi
    if command -v tsh > /dev/null 2>&1; then
        local tgt="${KUBE_CLUSTER:-${CLUSTER:-${KUBE_CONTEXT:-}}}"
        if [ -n "$tgt" ]; then
            printf 'tsh kube login %s%s' "${KUBE_PROXY:+--proxy=$KUBE_PROXY }" "$tgt"
            return 0
        fi
    fi
    printf ''
}

# Run the configured re-authentication command without blocking for interactive input.
# Failure is reported once and callers retain their retry semantics.
llmb::reauth() {
    local cmd
    cmd="$(llmb::reauth_cmd)"
    if [ -z "$cmd" ]; then
        _llmb_reauth_notice 'credentials expired and no re-auth hook is configured — set CONNECT_CMD in the cluster profile (e.g. CONNECT_CMD="tsh kube login <ctx>"), then the wait/fetch resumes on its own'
        return 1
    fi
    printf '  … credentials expired — refreshing via profile hook: %s\n' "$cmd" >&2
    if (eval "$cmd") > /dev/null 2>&1; then
        return 0
    fi
    _llmb_reauth_notice "automatic re-auth failed (the SSO session likely also expired) — run this ONCE, then the wait/fetch resumes on its own:  $cmd"
    return 1
}

# Run kubectl with bounded re-authentication and transient-error retries.
# Other errors return immediately; retry exhaustion returns LLMB_RC_TRANSIENT.
# Usage:  out="$(llmb::kc_resilient -n "$NS" get job "$j" -o jsonpath='...')"; rc=$?
llmb::kc_resilient() {
    local tries="${LLMB_KC_TRIES:-6}" max_reauth="${LLMB_MAX_REAUTH:-3}" base="${LLMB_KC_BACKOFF:-3}"
    local t=1 reauthed=0 rc=0 out kind errf
    # Longer backoff on Teleport-proxied contexts (slower proxy round-trips) — mirrors _stream_retry.sh.
    case "${KUBE_CONTEXT:-}${CONNECT_CMD:-}" in *teleport* | *tsh*) base=$((base * 2)) ;; esac
    while :; do
        errf="$(mktemp -t llmb-kcres.XXXXXX)"
        # Capture the command status before running any other shell construct.
        out="$(llmb::kc_raw "$@" 2> "$errf")"
        rc=$?
        LLMB_KC_STDERR="$(cat "$errf")"
        rm -f "$errf"
        if [ "$rc" -eq 0 ]; then
            printf '%s' "$out"
            return 0
        fi
        kind="$(llmb::classify_kube_err "$LLMB_KC_STDERR")"
        if [ "$kind" = other ]; then
            # Genuine kubectl error (NotFound / bad args / real server error) — propagate immediately.
            return "$rc"
        fi
        if [ "$kind" = auth ] && [ "$reauthed" -lt "$max_reauth" ]; then
            llmb::reauth || true # best-effort; even if it fails we still retry (creds may heal externally)
            reauthed=$((reauthed + 1))
        fi
        if [ "$t" -ge "$tries" ]; then
            return "$LLMB_RC_TRANSIENT"
        fi
        sleep "$((t * base))"
        t=$((t + 1))
    done
}

# Best-effort authentication refresh for long polling loops.
llmb::heal_auth() {
    local errf kind
    errf="$(mktemp -t llmb-heal.XXXXXX)"
    if llmb::kc_raw ${NAMESPACE:+-n "$NAMESPACE"} auth can-i get pods \
        --request-timeout="${LLMB_AUTH_PROBE_TIMEOUT:-10s}" > /dev/null 2> "$errf"; then
        rm -f "$errf"
        return 0
    fi
    kind="$(llmb::classify_kube_err "$(cat "$errf")")"
    rm -f "$errf"
    [ "$kind" = auth ] && llmb::reauth || true
    return 0
}

# Resolve a Job state without converting transient API failures into terminal states.
# Prints: succeeded | failed | running | notfound | unknown.
llmb::job_state() {
    local job="$1" ns="${2:-$NAMESPACE}" out rc=0 s f a outf
    # Run in the current shell so LLMB_KC_STDERR remains available for classification.
    outf="$(mktemp -t llmb-jobstate.XXXXXX)"
    llmb::kc_resilient -n "$ns" get job "$job" \
        -o jsonpath='{.status.succeeded}|{.status.failed}|{.status.active}' > "$outf" || rc=$?
    out="$(cat "$outf")"
    rm -f "$outf"
    if [ "${rc:-0}" -eq "$LLMB_RC_TRANSIENT" ]; then
        echo unknown
        return 0
    fi
    if [ "${rc:-0}" -ne 0 ]; then
        case "$LLMB_KC_STDERR" in
            *NotFound* | *"not found"*)
                echo notfound
                return 0
                ;;
            *)
                echo unknown
                return 0
                ;; # a non-transient, non-NotFound error → SAFE default is keep-waiting
        esac
    fi
    IFS='|' read -r s f a << EOF
$out
EOF
    if [ "${s:-0}" -ge 1 ] 2> /dev/null; then
        echo succeeded
        return 0
    fi
    if [ "${f:-0}" -ge 1 ] 2> /dev/null; then
        echo failed
        return 0
    fi
    echo running
}

# Follow a Job until success, failure, or confirmed deletion after it was observed.
# Transient errors and temporary inactive states continue polling.
llmb::follow_job_to_terminal() {
    local job="$1" ns="${2:-$NAMESPACE}" poll="${3:-${LLMB_JOB_POLL_S:-15}}" seen=0 st
    while :; do
        st="$(llmb::job_state "$job" "$ns")"
        case "$st" in
            succeeded) return 0 ;;
            failed) return 1 ;;
            running) seen=1 ;;
            notfound) [ "$seen" = 1 ] && return 2 || true ;; # gone-after-seen → terminal; gone-before-seen → wait
            unknown) : ;;                                    # transient — job_state already re-authed; keep waiting
        esac
        sleep "$poll"
    done
}

# Return success only from an observed completion, failure only from an authoritative failed state,
# and wait for active or unknown states.
_verdict_from_state() {
    local s="${1:-0}" f="${2:-0}" _a="${3:-0}" ok="${4:-0}"
    case "$s" in '' | *[!0-9]*) s=0 ;; esac
    case "$f" in '' | *[!0-9]*) f=0 ;; esac
    if [ "$s" -ge 1 ]; then
        echo success
        return 0
    fi
    if [ "$ok" = 1 ] && [ "$f" -ge 1 ]; then
        echo failure
        return 0
    fi
    echo wait
    return 0
}
