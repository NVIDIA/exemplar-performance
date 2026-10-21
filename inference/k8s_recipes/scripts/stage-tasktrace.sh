#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# stage-tasktrace.sh — the agentic lane's install step: stages what a cell's TaskTrace Job reads under
# ${MODEL_CACHE_PVC}:/tasktrace/ (trajectory, recipe module, tasktrace binary, the coordinator's kubectl, and the
# capture venv for capture cells), skipping what is already there.
#
# Usage: scripts/stage-tasktrace.sh <cell-dir> <cluster-profile>
set -euo pipefail

CELL="${1:?usage: stage-tasktrace.sh <cell-dir> <cluster-profile>}"
PROFILE="${2:?need a cluster-profile name (cluster-profiles/<name>.env)}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$ROOT/../.." && pwd)"
ENVF="$ROOT/cluster-profiles/${PROFILE}.env"
COMMON="$REPO/inference/slurm_recipes/agentic/common"
[ -f "$ENVF" ] || {
    echo "stage-tasktrace: no profile at $ENVF" >&2
    exit 1
}
[ -f "$CELL/recipe.yaml" ] || {
    echo "stage-tasktrace: no $CELL/recipe.yaml" >&2
    exit 1
}
[ -f "$COMMON/tasktrace_recipe.py" ] || {
    echo "stage-tasktrace: the shared agentic module is missing at $COMMON (the Slurm recipes' common/ ships with the repo)" >&2
    exit 1
}

set -a
. "$ENVF"
set +a
. "$ROOT/scripts/_model_cache.sh"
llmb::resolve_model_cache_pvc "$CELL" "$ENVF" || exit 1
llmb::model_cache_node_selector_yaml || exit 1
kc() { kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} "$@"; }
. "$ROOT/scripts/_stream_retry.sh"

# --- the recipe's contract -----------------------------------------------------------------------------
read -r NAME MODE SELECTOR TRACE_PATH TRACE_SHA IMAGE_ARCH < <(
    python3 - "$CELL/recipe.yaml" << 'PY'
import sys, yaml
r = yaml.safe_load(open(sys.argv[1])) or {}
e, tt = r.get("envelope") or {}, r.get("tasktrace") or {}
t, img = tt.get("trace") or {}, tt.get("image") or {}
print(e.get("name", ""), e.get("mode", ""), t.get("selector", ""), t.get("path", ""), t.get("sha256", ""), img.get("arch", ""))
PY
)
for v in NAME MODE SELECTOR TRACE_PATH TRACE_SHA IMAGE_ARCH; do
    [ -n "${!v}" ] || {
        echo "stage-tasktrace: recipe.yaml is missing the field behind $v" >&2
        exit 1
    }
done
TRACE_DIR="$REPO/$TRACE_PATH"
TRACE_FILE="$TRACE_DIR/trajectory.jsonl"
[ -f "$TRACE_FILE" ] && [ -f "$TRACE_DIR/trace.json" ] || {
    echo "stage-tasktrace: $TRACE_PATH must hold trajectory.jsonl + trace.json (the shipped trajectory)" >&2
    exit 2
}
LOCAL_SHA="$(shasum -a 256 "$TRACE_FILE" | awk '{print $1}')"
[ "$LOCAL_SHA" = "$TRACE_SHA" ] || {
    echo "stage-tasktrace: trajectory.jsonl sha256 $LOCAL_SHA != recipe tasktrace.trace.sha256 $TRACE_SHA — the repo and the recipe disagree" >&2
    exit 2
}
grep -q "\"selector\": \"$SELECTOR\"" "$TRACE_DIR/trace.json" || {
    echo "stage-tasktrace: $TRACE_DIR/trace.json does not declare selector $SELECTOR" >&2
    exit 2
}
TT_ROOT=/model-cache/tasktrace
REL_TRACE="${TRACE_PATH#inference/trajectories/}"
echo "stage-tasktrace: cell=$NAME  mode=$MODE  ns=$NAMESPACE  pvc=$MODEL_CACHE_PVC  trace=$REL_TRACE  arch=$IMAGE_ARCH"

# --- 1. the static tasktrace binary, built here by the Slurm install script ---
BUILD_ROOT="${TASKTRACE_BUILD_ROOT:-$ROOT/.tasktrace-build}"
export LLMB_INSTALL="$BUILD_ROOT" LLMB_WORKLOAD="$BUILD_ROOT/workloads/k8s_tasktrace" LLMB_NODE_ARCHITECTURE="$IMAGE_ARCH"
mkdir -p "$LLMB_WORKLOAD"
echo "stage-tasktrace: building/refreshing the tasktrace binary for $IMAGE_ARCH under $BUILD_ROOT (install_tasktrace_replay.sh)"
bash "$COMMON/install_tasktrace_replay.sh" 2>&1 | sed 's/^/  /'
BIN="$LLMB_WORKLOAD/bin/tasktrace"
BIN_VERSION="$LLMB_WORKLOAD/bin/tasktrace.version"
[ -x "$BIN" ] && [ -f "$BIN_VERSION" ] || {
    echo "stage-tasktrace: the install script left no binary at $BIN" >&2
    exit 1
}
BIN_SOURCE_ID="$(sed -n 's/^source=//p' "$BIN_VERSION" | head -1)"

# --- 1b. kubectl for the coordinator, sha256-verified ---
KUBECTL_VERSION="${TASKTRACE_KUBECTL_VERSION:-v1.33.4}"
case "$IMAGE_ARCH" in
    aarch64) KUBECTL_ARCH=arm64 ;;
    x86_64) KUBECTL_ARCH=amd64 ;;
    *)
        echo "stage-tasktrace: no kubectl build for arch $IMAGE_ARCH" >&2
        exit 1
        ;;
esac
KUBECTL_DIR="$BUILD_ROOT/kubectl/$KUBECTL_VERSION/$KUBECTL_ARCH"
KUBECTL_BIN="$KUBECTL_DIR/kubectl"
if [ ! -x "$KUBECTL_BIN" ]; then
    mkdir -p "$KUBECTL_DIR"
    KUBECTL_URL="https://dl.k8s.io/release/$KUBECTL_VERSION/bin/linux/$KUBECTL_ARCH/kubectl"
    echo "stage-tasktrace: downloading kubectl $KUBECTL_VERSION ($KUBECTL_ARCH) for the coordinator"
    curl -fsSL --retry 3 -o "$KUBECTL_BIN.tmp" "$KUBECTL_URL"
    KUBECTL_WANT="$(curl -fsSL --retry 3 "$KUBECTL_URL.sha256" | awk '{print $1}')"
    KUBECTL_GOT="$(shasum -a 256 "$KUBECTL_BIN.tmp" | awk '{print $1}')"
    [ -n "$KUBECTL_WANT" ] && [ "$KUBECTL_GOT" = "$KUBECTL_WANT" ] || {
        rm -f "$KUBECTL_BIN.tmp"
        echo "stage-tasktrace: kubectl checksum mismatch (got $KUBECTL_GOT, published $KUBECTL_WANT)" >&2
        exit 1
    }
    chmod 0755 "$KUBECTL_BIN.tmp"
    mv -f "$KUBECTL_BIN.tmp" "$KUBECTL_BIN"
fi
KUBECTL_SHA="$(shasum -a 256 "$KUBECTL_BIN" | awk '{print $1}')"

# --- 2. a helper pod with the cache PVC mounted ---
POD="${NAME}-ttstage"
cleanup_pod() { kc -n "$NAMESPACE" delete pod "$POD" --ignore-not-found --wait=false > /dev/null 2>&1 || true; }
trap 'cleanup_pod' EXIT
kc -n "$NAMESPACE" delete pod "$POD" --ignore-not-found > /dev/null 2>&1 || true
{
    echo "apiVersion: v1"
    echo "kind: Pod"
    echo "metadata: { name: ${POD}, namespace: ${NAMESPACE}, labels: { app.kubernetes.io/managed-by: llmb-recipe } }"
    echo "spec:"
    echo "  restartPolicy: Never"
    [ -n "${IMAGE_PULL_SECRET:-}" ] && echo "  imagePullSecrets: [{ name: ${IMAGE_PULL_SECRET} }]"
    echo "  tolerations: [{ operator: Exists, effect: NoSchedule }, { operator: Exists, effect: PreferNoSchedule }]"
    [ -n "${MODEL_CACHE_NODE_SELECTOR_YAML:-}" ] && echo "  nodeSelector: { ${MODEL_CACHE_NODE_SELECTOR_YAML} }"
    echo "  containers:"
    echo "  - name: stage"
    echo "    image: docker.io/library/alpine:3"
    echo "    command: [sleep, '1800']"
    echo "    volumeMounts: [{ name: cache, mountPath: /model-cache }]"
    echo "  volumes:"
    echo "  - name: cache"
    echo "    persistentVolumeClaim: { claimName: ${MODEL_CACHE_PVC} }"
} | kc apply -f -
kc -n "$NAMESPACE" wait --for=condition=ready "pod/$POD" --timeout=300s
pexec() { kc -n "$NAMESPACE" exec "$POD" -- sh -c "$1"; }
pcp() { retry_stream "$3" 3 -- kc -n "$NAMESPACE" cp "$1" "$POD:$2"; }

# 2a. the trajectory
DEST_TRACE="$TT_ROOT/trajectories/$REL_TRACE"
EXIST="$(pexec "sha256sum '$DEST_TRACE/trajectory.jsonl' 2>/dev/null | cut -d' ' -f1" || true)"
if [ "$EXIST" = "$TRACE_SHA" ]; then
    echo "stage-tasktrace: trajectory already staged with matching hash — skipping."
else
    [ -n "$EXIST" ] && echo "stage-tasktrace: on-PVC trajectory hash $EXIST != pinned — REPLACING."
    pexec "mkdir -p '$DEST_TRACE'"
    for f in trajectory.jsonl trace.json trajectory.jsonl.sha256; do
        [ -f "$TRACE_DIR/$f" ] && pcp "$TRACE_DIR/$f" "$DEST_TRACE/$f" "trajectory $f cp"
    done
    PVC_SHA="$(pexec "sha256sum '$DEST_TRACE/trajectory.jsonl' | cut -d' ' -f1")"
    [ "$PVC_SHA" = "$TRACE_SHA" ] || {
        echo "stage-tasktrace: POST-COPY VERIFY FAILED — on-PVC $PVC_SHA != pinned $TRACE_SHA" >&2
        exit 3
    }
    echo "stage-tasktrace: trajectory staged + verified ($DEST_TRACE = $TRACE_SHA)"
fi

# 2b. the shared recipe module, install lib and venv-build script
pexec "mkdir -p '$TT_ROOT/common'"
for f in tasktrace_recipe.py tasktrace_install_lib.sh tasktrace_capture_venv_build.sh; do
    LOCAL_H="$(shasum -a 256 "$COMMON/$f" | awk '{print $1}')"
    REMOTE_H="$(pexec "sha256sum '$TT_ROOT/common/$f' 2>/dev/null | cut -d' ' -f1" || true)"
    if [ "$LOCAL_H" != "$REMOTE_H" ]; then
        pcp "$COMMON/$f" "$TT_ROOT/common/$f" "common $f cp"
    fi
done
echo "stage-tasktrace: recipe module staged ($TT_ROOT/common)"

# 2c. the binary, keyed by its source id
DEST_BIN_DIR="$TT_ROOT/bin/$IMAGE_ARCH"
REMOTE_SOURCE_ID="$(pexec "sed -n 's/^source=//p' '$DEST_BIN_DIR/tasktrace.version' 2>/dev/null | head -1" || true)"
if [ -n "$BIN_SOURCE_ID" ] && [ "$REMOTE_SOURCE_ID" = "$BIN_SOURCE_ID" ] && pexec "test -x '$DEST_BIN_DIR/tasktrace'"; then
    echo "stage-tasktrace: tasktrace binary already staged ($BIN_SOURCE_ID) — skipping."
else
    pexec "mkdir -p '$DEST_BIN_DIR'"
    pcp "$BIN" "$DEST_BIN_DIR/tasktrace.tmp" "tasktrace binary cp"
    pcp "$BIN_VERSION" "$DEST_BIN_DIR/tasktrace.version" "tasktrace version cp"
    pexec "chmod 0755 '$DEST_BIN_DIR/tasktrace.tmp' && mv -f '$DEST_BIN_DIR/tasktrace.tmp' '$DEST_BIN_DIR/tasktrace'"
    echo "stage-tasktrace: tasktrace binary staged ($DEST_BIN_DIR, $BIN_SOURCE_ID)"
fi

# 2d. the coordinator's kubectl, keyed by its sha256
REMOTE_KUBECTL_SHA="$(pexec "sha256sum '$DEST_BIN_DIR/kubectl' 2>/dev/null | cut -d' ' -f1" || true)"
if [ "$REMOTE_KUBECTL_SHA" = "$KUBECTL_SHA" ]; then
    echo "stage-tasktrace: kubectl $KUBECTL_VERSION already staged — skipping."
else
    pexec "mkdir -p '$DEST_BIN_DIR'"
    pcp "$KUBECTL_BIN" "$DEST_BIN_DIR/kubectl.tmp" "kubectl cp"
    pexec "chmod 0755 '$DEST_BIN_DIR/kubectl.tmp' && mv -f '$DEST_BIN_DIR/kubectl.tmp' '$DEST_BIN_DIR/kubectl'"
    echo "stage-tasktrace: kubectl $KUBECTL_VERSION staged ($DEST_BIN_DIR/kubectl)"
fi

# --- 3. capture cells: the tasktrace-capture venv, built in-cluster on a node of the image's arch ---
if [ "$MODE" = "tasktrace-capture-replay" ]; then
    SRC="$REPO/inference/tools/tasktrace-capture"
    # shellcheck source=/dev/null
    . "$COMMON/tasktrace_install_lib.sh"
    SOURCE_ID="$(tt_source_id "$SRC" .venv)"
    SOURCE_COMMIT="$(tt_source_commit "$SRC")"
    VENV_DIR="$TT_ROOT/venv/$IMAGE_ARCH"
    PY_VER="${TASKTRACE_CAPTURE_PYTHON:-3.12}"
    UV_VER="${TASKTRACE_CAPTURE_UV_VERSION:-0.9.28}"
    REMOTE_VENV_SOURCE="$(pexec "sed -n 's/^source=//p' '$VENV_DIR.version' 2>/dev/null | head -1" || true)"
    if [ "$REMOTE_VENV_SOURCE" = "$SOURCE_ID" ] && pexec "test -x '$VENV_DIR/bin/tasktrace-capture'"; then
        echo "stage-tasktrace: tasktrace-capture venv already built for this source ($SOURCE_ID) — skipping."
    else
        # The venv and its uv-managed CPython sit on the PVC at the paths the task pods see, so it works in their image.
        SRC_DEST="$TT_ROOT/src/tasktrace-capture/${SOURCE_ID//:/-}"
        WORK="$(mktemp -d)"
        tar -C "$SRC" --exclude .venv --exclude '__pycache__' --exclude .pytest_cache -czf "$WORK/src.tgz" .
        pexec "mkdir -p '$SRC_DEST'"
        pcp "$WORK/src.tgz" "$SRC_DEST.tgz" "tasktrace-capture source cp"
        pexec "rm -rf '$SRC_DEST' && mkdir -p '$SRC_DEST' && tar -xzf '$SRC_DEST.tgz' -C '$SRC_DEST' && rm -f '$SRC_DEST.tgz'"
        rm -rf "$WORK"
        # The build Job mounts the cache as well, placed by architecture rather than with the helper: release the
        # helper's mount first so a ReadWriteOnce cache is never attached to two nodes at once.
        kc -n "$NAMESPACE" delete pod "$POD" --ignore-not-found --wait=true > /dev/null
        NODE_ARCH=amd64
        [ "$IMAGE_ARCH" = aarch64 ] && NODE_ARCH=arm64
        BUILD_JOB="${NAME}-ttvenv"
        kc -n "$NAMESPACE" delete job "$BUILD_JOB" --ignore-not-found --wait=true > /dev/null 2>&1 || true
        echo "stage-tasktrace: building the tasktrace-capture venv in-cluster (job/$BUILD_JOB on a $NODE_ARCH node; python $PY_VER, uv $UV_VER)"
        {
            echo "apiVersion: batch/v1"
            echo "kind: Job"
            echo "metadata: { name: ${BUILD_JOB}, namespace: ${NAMESPACE}, labels: { app.kubernetes.io/managed-by: llmb-recipe, llmb.nvidia.com/cell: ${NAME} } }"
            echo "spec:"
            echo "  backoffLimit: 0"
            echo "  activeDeadlineSeconds: 3600"
            echo "  ttlSecondsAfterFinished: 86400"
            echo "  template:"
            echo "    spec:"
            echo "      restartPolicy: Never"
            [ -n "${IMAGE_PULL_SECRET:-}" ] && echo "      imagePullSecrets: [{ name: ${IMAGE_PULL_SECRET} }]"
            echo "      nodeSelector: { kubernetes.io/arch: ${NODE_ARCH}${TASKTRACE_NODE_SELECTOR:+, ${TASKTRACE_NODE_SELECTOR}}${MODEL_CACHE_NODE_SELECTOR_YAML:+, ${MODEL_CACHE_NODE_SELECTOR_YAML}} }"
            echo "      tolerations: [{ operator: Exists, effect: NoSchedule }, { operator: Exists, effect: PreferNoSchedule }]" # like the staging pod: TASKTRACE_NODE_SELECTOR may pick tainted GPU nodes
            echo "      containers:"
            echo "      - name: venv-build"
            echo "        image: docker.io/library/python:${PY_VER}-bookworm" # has bash, curl, git; the script installs its own uv + CPython
            echo "        command: [bash, -lc]"
            echo "        args:"
            echo "        - |"
            echo "          set -euo pipefail"
            echo "          export TT_LIB='$TT_ROOT/common/tasktrace_install_lib.sh' TT_SRC_DIR='$SRC_DEST' TT_VENV_DIR='$VENV_DIR'"
            echo "          export TT_UV_DIR='$TT_ROOT/uv/$IMAGE_ARCH' TT_PYTHON='$PY_VER' TT_UV_VERSION='$UV_VER' TT_EXPECT_ARCH='$IMAGE_ARCH'"
            echo "          export TT_SOURCE_ID='$SOURCE_ID' TT_SOURCE_COMMIT='$SOURCE_COMMIT' HOME=/tmp/home"
            echo '          mkdir -p "$HOME"'
            echo "          bash '$TT_ROOT/common/tasktrace_capture_venv_build.sh'"
            echo "          test -x '$VENV_DIR/bin/tasktrace-capture' || { echo 'venv build finished without $VENV_DIR/bin/tasktrace-capture' >&2; exit 1; }"
            echo "        resources: { requests: { cpu: '4', memory: 8Gi }, limits: { cpu: '8', memory: 16Gi } }"
            echo "        volumeMounts: [{ name: cache, mountPath: /model-cache }]"
            echo "      volumes:"
            echo "      - name: cache"
            echo "        persistentVolumeClaim: { claimName: ${MODEL_CACHE_PVC} }"
        } | kc apply -f -
        if ! kc -n "$NAMESPACE" wait --for=condition=complete "job/$BUILD_JOB" --timeout=3600s; then
            echo "stage-tasktrace: venv build job FAILED — log follows:" >&2
            kc -n "$NAMESPACE" logs "job/$BUILD_JOB" --tail=80 >&2 || true
            exit 1
        fi
        kc -n "$NAMESPACE" logs "job/$BUILD_JOB" --tail=5 | sed 's/^/  /' || true
        echo "stage-tasktrace: tasktrace-capture venv built ($VENV_DIR, $SOURCE_ID)"
    fi
fi
echo "stage-tasktrace: ✅ $NAME staged on $MODEL_CACHE_PVC:$TT_ROOT"
