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

# render.sh <cell-dir> [--to <out-dir>]
#
# Renders a cell's serving-stack Jinja templates from its recipe.yaml into
# <cell-dir>/rendered/ (the committed, digest-pinned, cluster-agnostic manifests).
# Launch then applies them with: envsubst < rendered/*.yaml | kubectl apply -f -
set -euo pipefail

CELL="${1:?usage: render.sh <cell-dir> [--to <out-dir>]}"
OUT="$CELL/rendered"
if [ "${2:-}" = "--to" ]; then OUT="${3:?--to needs a dir}"; fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" # k8s collection root
RECIPE="$CELL/recipe.yaml"
[ -f "$RECIPE" ] || {
    echo "render.sh: no recipe.yaml at $RECIPE" >&2
    exit 1
}

STACK="$(python3 -c 'import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))["serving"]["stack"])' "$RECIPE")"
LAUNCHER="$(python3 -c 'import yaml,sys; print((yaml.safe_load(open(sys.argv[1]))["envelope"]).get("launcher",""))' "$RECIPE")"
MODE="$(python3 -c 'import yaml,sys; print((yaml.safe_load(open(sys.argv[1]))["envelope"]).get("mode",""))' "$RECIPE")"
TPL="$ROOT/serving/$STACK/templates"
[ -d "$TPL" ] || {
    echo "render.sh: no templates for stack '$STACK' at $TPL" >&2
    exit 1
}

# 1) the shared serving stack (e.g. the vllm-agg server).
python3 "$ROOT/scripts/_render.py" "$RECIPE" "$TPL" "$OUT"

# 2) launcher OVERLAY: scenario-specific manifests on top of the shared server. Rendered into the same
#    rendered/ dir. Skipped when there is no overlay dir (llm-perf's launcher is aiperf, whose bench-Job
#    lives under vllm-agg).
OVL="$ROOT/serving/$LAUNCHER/templates"
if [ -n "$LAUNCHER" ] && [ "$OVL" != "$TPL" ] && compgen -G "$OVL/*.j2" > /dev/null 2>&1; then
    python3 "$ROOT/scripts/_render.py" "$RECIPE" "$OVL" "$OUT"
fi
