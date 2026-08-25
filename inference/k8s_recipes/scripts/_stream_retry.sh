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
# Retry idempotent streaming kubectl operations with bounded linear backoff.
# Usage: retry_stream "description" 3 -- <command...>

retry_stream() {
    local _desc="$1" _max="$2"
    shift 2
    [ "${1:-}" = "--" ] && shift
    # Proxied contexts use a longer backoff.
    local _base=3
    case "${KUBE_CONTEXT:-}" in *.teleport.sh-* | *teleport*) _base=6 ;; esac
    local _t _rc=1
    _t=1
    while [ "$_t" -le "$_max" ]; do
        # Capture the command status directly in the else branch.
        if "$@"; then
            return 0
        else
            _rc=$?
        fi
        if [ "$_t" -lt "$_max" ]; then
            printf 'retry_stream: %s — attempt %s/%s failed (rc=%s; Teleport/API stream stall?); retrying in %ss…\n' \
                "$_desc" "$_t" "$_max" "$_rc" "$((_t * _base))" >&2
            sleep "$((_t * _base))"
        fi
        _t=$((_t + 1))
    done
    printf 'retry_stream: %s — FAILED after %s attempts (last rc=%s; context deadline / stream error).\n' \
        "$_desc" "$_max" "$_rc" >&2
    return "$_rc"
}
