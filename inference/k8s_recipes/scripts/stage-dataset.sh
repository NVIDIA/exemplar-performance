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

# stage-dataset.sh <cell-dir> <cluster-profile>
#
# Distributes the ONE canonical trace committed in the repo to a cluster's model-cache PVC, so every
# cluster benchmarks byte-identical bytes (cross-cluster comparability). It does NOT regenerate: it copies
# the committed artifact and VERIFIES the on-PVC sha256 matches bench.dataset.sha256, aborting on mismatch.
# Idempotent: if the trace is already present on the PVC with the right hash, it skips.
#
# Run once per cluster before scripts/sweep.sh. Requires: kubectl, python3, gunzip.
set -euo pipefail

CELL="${1:?usage: stage-dataset.sh <cell-dir> <cluster-profile>}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
[ -f "$ENVF" ] || {
    echo "stage-dataset: no profile at $ENVF" >&2
    exit 1
}
[ -f "$CELL/recipe.yaml" ] || {
    echo "stage-dataset: no $CELL/recipe.yaml" >&2
    exit 1
}

set -a
. "$ENVF"
set +a
# Resolve THE model-cache claim for THIS cell (scripts/_model_cache.sh -> install.py
# --resolve-cache -> install.resolve_cache_claim). The rendered manifests below mount
# ${MODEL_CACHE_PVC}, so this MUST be the same string install downloaded into; it is, because it
# is the same function. Fail-closed: an unresolvable claim aborts here, not at model-load time.
. "$ROOT/scripts/_model_cache.sh"
llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1
# Canonicalise the cache node-selector through the SAME parser the Python mounters use
# (model_cache.parse_node_selector). Fail-closed on a malformed spec — see _model_cache.sh.
llmb::model_cache_node_selector_yaml || exit 1
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
. "$ROOT/scripts/_stream_retry.sh" # retry_stream: retry streaming kubectl cp over a Teleport/API proxy stall

# Synthetic AIPerf cells generate prompts in the bench pod and intentionally have no dataset artifact.
# Treat staging as a successful no-op so the shared llm-perf lifecycle does not demand bench.dataset.sha256.
MODE="$(python3 -c 'import sys,yaml; r=yaml.safe_load(open(sys.argv[1])) or {}; print((r.get("envelope") or {}).get("mode", ""))' "$CELL/recipe.yaml")"
if [ "$MODE" = "synthetic" ]; then
    echo "stage-dataset: synthetic mode — no dataset artifact to stage (successful no-op)."
    exit 0
fi

# --- read the dataset contract from recipe.yaml ---
read -r SUBPATH CONFIG_PATH SHA256 ARTIFACT NAME < <(
    python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
d = (r.get("bench") or {}).get("dataset") or {}
e = r.get("envelope") or {}
print(d.get("subpath",""), d.get("config_path",""), d.get("sha256",""),
      d.get("artifact",""), e.get("name",""))
PY
)
[ -n "$SUBPATH" ] || {
    echo "stage-dataset: bench.dataset.subpath missing" >&2
    exit 1
}
# THE GATE: refuse to stage without a pinned hash — that's what guarantees comparability.
[ -n "$SHA256" ] || {
    echo "stage-dataset: bench.dataset.sha256 is REQUIRED (comparability gate) — the canonical trace must be content-pinned before staging" >&2
    exit 2
}
[ -n "$ARTIFACT" ] && [ -f "$ROOT/$ARTIFACT" ] || {
    echo "stage-dataset: canonical artifact '$ARTIFACT' not found in repo — commit it first" >&2
    exit 2
}

# --- materialize the canonical trace locally (decompress if .gz) + verify it IS the pinned bytes ---
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
TRACE="$WORK/dataset.jsonl"
case "$ARTIFACT" in
    *.gz) gunzip -c "$ROOT/$ARTIFACT" > "$TRACE" ;;
    *) cp "$ROOT/$ARTIFACT" "$TRACE" ;;
esac
LOCAL_HASH="$(shasum -a 256 "$TRACE" | awk '{print $1}')"
[ "$LOCAL_HASH" = "$SHA256" ] || {
    echo "stage-dataset: committed artifact hash $LOCAL_HASH != recipe sha256 $SHA256 — the repo artifact and the recipe disagree" >&2
    exit 2
}

POD="${NAME}-dsstage"
DEST="/model-cache/${SUBPATH}"
echo "stage-dataset: cell=$NAME  ns=$NAMESPACE  dest=$DEST  sha256=$SHA256"

cleanup_pod() { kc -n "$NAMESPACE" delete pod "$POD" --ignore-not-found --wait=false > /dev/null 2>&1 || true; }
trap 'cleanup_pod; rm -rf "$WORK"' EXIT

# --- helper pod: mounts the model-cache PVC read-WRITE (server mounts it read-only, so we need our own) ---
kc -n "$NAMESPACE" delete pod "$POD" --ignore-not-found > /dev/null 2>&1 || true
{
    echo "apiVersion: v1"
    echo "kind: Pod"
    echo "metadata: { name: ${POD}, namespace: ${NAMESPACE}, labels: { app.kubernetes.io/managed-by: llmb-recipe } }"
    echo "spec:"
    echo "  restartPolicy: Never"
    [ -n "${IMAGE_PULL_SECRET:-}" ] && echo "  imagePullSecrets: [{ name: ${IMAGE_PULL_SECRET} }]"
    echo "  tolerations: [{ operator: Exists, effect: NoSchedule }, { operator: Exists, effect: PreferNoSchedule }]"
    # Scheduling effects only — NOT a blanket Exists. A bare `operator: Exists` also tolerates
    # NoExecute, which SUPPRESSES the not-ready/unreachable evictions kubernetes defaults onto every
    # pod, so a stager on a node that goes NotReady hangs until the wait budget expires. Mirrors
    # model_cache._CACHE_TOLERATIONS, which places every Python-side cache-mounting pod.
    # PIN to a node that can mount the claim. Some storage classes (NFS-backed RWX especially) mount on only
    # a subset of nodes; unpinned, this helper picks an unmountable one and the stage hangs on the mount.
    # From the profile's MODEL_CACHE_NODE_SELECTOR; omitted when unset.
    [ -n "${MODEL_CACHE_NODE_SELECTOR_YAML:-}" ] && echo "  nodeSelector: { ${MODEL_CACHE_NODE_SELECTOR_YAML} }"
    echo "  containers:"
    echo "  - name: stage"
    echo "    image: alpine:3"
    echo "    command: [sleep, '600']"
    echo "    volumeMounts: [{ name: cache, mountPath: /model-cache }]"
    echo "  volumes:"
    echo "  - name: cache"
    echo "    persistentVolumeClaim: { claimName: ${MODEL_CACHE_PVC} }"
} | kc apply -f -
kc -n "$NAMESPACE" wait --for=condition=ready "pod/$POD" --timeout=180s

# --- idempotent: already staged with the right hash? ---
EXIST="$(kc -n "$NAMESPACE" exec "$POD" -- sh -c "sha256sum '$DEST' 2>/dev/null | cut -d' ' -f1" || true)"
if [ "$EXIST" = "$SHA256" ]; then
    echo "stage-dataset: already present with matching hash — skipping."
    exit 0
fi
[ -n "$EXIST" ] && echo "stage-dataset: on-PVC hash $EXIST != canonical — REPLACING with the canonical bytes."

# --- copy canonical trace (+ config) into the PVC, then VERIFY the on-PVC hash ---
kc -n "$NAMESPACE" exec "$POD" -- mkdir -p "$(dirname "$DEST")"
retry_stream "dataset cp" 3 -- kc -n "$NAMESPACE" cp "$TRACE" "$POD:$DEST" \
    || {
        echo "stage-dataset: FAILED to copy the trace (context deadline / stream error)." >&2
        exit 1
    }
if [ -n "$CONFIG_PATH" ] && [ -f "$ROOT/datasets/$(basename "$CONFIG_PATH")" ]; then
    kc -n "$NAMESPACE" exec "$POD" -- mkdir -p "/model-cache/$(dirname "$CONFIG_PATH")"
    retry_stream "dataset config cp" 3 -- kc -n "$NAMESPACE" cp "$ROOT/datasets/$(basename "$CONFIG_PATH")" "$POD:/model-cache/$CONFIG_PATH" \
        || {
            echo "stage-dataset: FAILED to copy the dataset config (context deadline / stream error)." >&2
            exit 1
        }
fi
PVC_HASH="$(kc -n "$NAMESPACE" exec "$POD" -- sh -c "sha256sum '$DEST' | cut -d' ' -f1")"
[ "$PVC_HASH" = "$SHA256" ] || {
    echo "stage-dataset: POST-COPY VERIFY FAILED — on-PVC $PVC_HASH != canonical $SHA256" >&2
    exit 3
}
echo "stage-dataset: ✅ canonical trace staged + verified on PVC ($DEST = $SHA256)"
