#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
