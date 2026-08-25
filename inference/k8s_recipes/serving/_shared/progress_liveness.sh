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

# Restart the container only when requests are active and the generated-token count has stopped moving.
# A reachable frontend is not enough because its serving engine may still be stalled. Missing or invalid
# metrics leave the pod healthy; Kubelet controls the retry window through the probe configuration.
set -u

STATE="${PROGRESS_PROBE_STATE:-/tmp/.llmb_progress_probe}"
URL="${SERVER_URL:-http://127.0.0.1:8000}/metrics"

# Keep probe-local variables unbraced because the manifest renderer substitutes braced uppercase names.
UNSUP="$STATE.unsupported"

# Treat a failed metrics request as unknown and leave the pod healthy.
BODY="$(curl -fsS --connect-timeout 3 --max-time 8 "$URL" 2> /dev/null)" || {
    rm -f "$STATE"
    exit 0
}
[ -n "$BODY" ] || {
    rm -f "$STATE"
    exit 0
}

# Sum labels within each metric family, then use the largest matching family.
_family_max() { # $1 = extended regex of accepted metric names
    printf '%s\n' "$BODY" | awk -v pat="$1" '
    $0 ~ /^#/ { next }
    {
      name = $1; sub(/\{.*/, "", name)
      if (name !~ pat) next
      if (name ~ /_(bucket|sum|count|created)$/) next
      v = $NF + 0
      acc[name] += v
    }
    END { m = ""; for (n in acc) if (m == "" || acc[n] > m) m = acc[n]; print (m == "" ? "" : m) }'
}

# Accept the token and in-flight metric names used by supported runtimes.
TOK="$(_family_max '(generation|output)_tokens(_total)?$')"
INFLIGHT="$(_family_max '(num_requests_running|inflight_requests)$')"

# If the runtime does not expose both metrics, warn once and leave the pod healthy.
if [ -z "$TOK" ] && [ -z "$INFLIGHT" ]; then
    if [ ! -f "$UNSUP" ]; then
        : > "$UNSUP" 2> /dev/null || :
        echo "llmb-liveness: WARNING — /metrics answered but exposes NEITHER a generation/output-token" >&2
        echo "  counter NOR a running/inflight-request gauge. Stall detection is unavailable for this runtime:" >&2
        echo "  the container will remain healthy until both metrics are configured for this runtime." >&2
    fi
    rm -f "$STATE"
    exit 0
fi
# If either metric is missing, clear prior state and leave the pod healthy.
[ -n "$TOK" ] && [ -n "$INFLIGHT" ] || {
    rm -f "$STATE"
    exit 0
}

PREV_TOK=""
PREV_STRIKES=0
if [ -r "$STATE" ]; then
    read -r PREV_TOK PREV_STRIKES < "$STATE" 2> /dev/null || {
        PREV_TOK=""
        PREV_STRIKES=0
    }
fi
case "${PREV_STRIKES:-}" in '' | *[!0-9]*) PREV_STRIKES=0 ;; esac

STRIKES=0
if [ -n "$PREV_TOK" ]; then
    # Use awk because Prometheus counters may be floating-point values.
    ADVANCED="$(awk -v a="$TOK" -v b="$PREV_TOK" 'BEGIN{print (a+0 > b+0) ? 1 : 0}')"
    REGRESSED="$(awk -v a="$TOK" -v b="$PREV_TOK" 'BEGIN{print (a+0 < b+0) ? 1 : 0}')"
    BUSY="$(awk -v n="$INFLIGHT" 'BEGIN{print (n+0 > 0) ? 1 : 0}')"
    if [ "$REGRESSED" = 1 ]; then
        STRIKES=0 # The engine restarted and reset the counter.
    elif [ "$ADVANCED" = 0 ] && [ "$BUSY" = 1 ]; then
        STRIKES=$((PREV_STRIKES + 1)) # Requests are active but output is not advancing.
    else
        STRIKES=0 # Output advanced or the engine is idle.
    fi
fi

printf '%s %s\n' "$TOK" "$STRIKES" > "$STATE" 2> /dev/null || :

# Kubelet's failureThreshold controls how many failed probes trigger a restart.
if [ "$STRIKES" -ge "${PROGRESS_PROBE_MIN_STRIKES:-1}" ]; then
    echo "stalled: generated tokens unchanged at $TOK with $INFLIGHT request(s) active (strike $STRIKES)" >&2
    exit 1
fi
exit 0
