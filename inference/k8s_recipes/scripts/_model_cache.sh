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

# _model_cache.sh resolves the model-cache PVC through the same fail-closed rule used by install and serving.
# Sourced by shell entrypoints; never executed directly.
#
# Usage (after the profile has been sourced, so MODEL_CACHE_PVC_OVERRIDE is honoured):
#     . "$ROOT/scripts/_model_cache.sh"
#     llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1

# Validate PVC names as RFC-1123 subdomains before inserting them into manifests.
llmb::_is_pvc_name() {
    case "$1" in
        "") return 1 ;;
        *[!abcdefghijklmnopqrstuvwxyz0123456789.-]*) return 1 ;;
        [!abcdefghijklmnopqrstuvwxyz0123456789]*) return 1 ;;
        *[!abcdefghijklmnopqrstuvwxyz0123456789]) return 1 ;;
    esac
    return 0
}

# Resolve and EXPORT MODEL_CACHE_PVC for <cell-dir> under <profile-env-file>.
llmb::resolve_model_cache_pvc() {
    local cell="$1" envf="$2" out rc root
    root="${LLMB_K8S_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"

    # A per-invocation override outranks the profile but still must be a valid PVC name.
    if [ -n "${MODEL_CACHE_PVC_OVERRIDE:-}" ]; then
        if ! llmb::_is_pvc_name "$MODEL_CACHE_PVC_OVERRIDE"; then
            echo "model-cache: MODEL_CACHE_PVC_OVERRIDE is not a PVC name: '$MODEL_CACHE_PVC_OVERRIDE'" >&2
            echo "             (expected one RFC-1123 subdomain, e.g. glm5-fp8-model-cache). Refusing." >&2
            return 1
        fi
        export MODEL_CACHE_PVC="$MODEL_CACHE_PVC_OVERRIDE"
        echo "model-cache: MODEL_CACHE_PVC_OVERRIDE -> $MODEL_CACHE_PVC"
        return 0
    fi

    if [ ! -f "$cell/recipe.yaml" ]; then
        echo "model-cache: no recipe.yaml in '$cell' — cannot resolve the model-cache PVC" >&2
        return 1
    fi

    # Capture only stdout as the claim name; keep diagnostics on stderr.
    local err
    err="$(mktemp)"
    out="$(python3 "$root/scripts/model_cache.py" resolve "$cell" "$envf" 2> "$err")"
    rc=$?
    if [ $rc -ne 0 ] || [ -z "$out" ]; then
        echo "model-cache: could not resolve the model-cache PVC for $cell" >&2
        cat "$err" >&2
        rm -f "$err"
        echo "model-cache: refusing to continue — an empty claimName mounts nothing and the server would" >&2
        echo "             fail on model-not-found AFTER allocating GPUs." >&2
        return 1
    fi
    cat "$err" >&2 # surface warnings; they must never become part of the name
    rm -f "$err"

    # Apply the same validation to resolved and overridden claim names.
    if ! llmb::_is_pvc_name "$out"; then
        echo "model-cache: resolver returned something that is not a PVC name: '$out'" >&2
        echo "             (expected one RFC-1123 subdomain, e.g. glm5-fp8-model-cache). Refusing to continue." >&2
        return 1
    fi
    export MODEL_CACHE_PVC="$out"
    return 0
}

# Canonicalize MODEL_CACHE_NODE_SELECTOR with the shared parser.
# Invalid non-empty selectors fail closed rather than widening pod placement.
llmb::model_cache_node_selector_yaml() {
    local root
    root="${LLMB_K8S_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
    MODEL_CACHE_NODE_SELECTOR_YAML="$(python3 "$root/scripts/model_cache.py" node-selector \
        "${MODEL_CACHE_NODE_SELECTOR:-}")" || return 1
    export MODEL_CACHE_NODE_SELECTOR_YAML
    return 0
}
