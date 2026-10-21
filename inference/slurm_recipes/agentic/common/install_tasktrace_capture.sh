#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Install `tasktrace-capture` (riptide) into a Python virtual environment under
# $LLMB_WORKLOAD/venv/, built on a node with the compute nodes' CPU architecture.
#
# Shared by the agentic recipes that run a live agent capture: each recipe's
# metadata.yaml lists this script as a setup task
# (`cmd: '../../common/install_tasktrace_capture.sh'`, job_type: srun,
# requires_gpus: false). llmb-install runs setup tasks with the recipe directory
# as the working directory and a per-workload $LLMB_WORKLOAD.
#
# Source of truth is the repository checkout: the capture tool lives in this
# tree at inference/tools/tasktrace-capture (the vendored riptide, override
# TASKTRACE_CAPTURE_SOURCE_DIR) and is installed from there. Nothing is cloned.
#
# Where it builds: tasktrace-capture runs on the compute node's host (not
# inside a container) with a uv-managed CPython and native wheels, so the venv
# must be created on the compute nodes' CPU architecture. When the node running
# llmb-install has that architecture ($LLMB_NODE_ARCHITECTURE), the venv is
# built right here with no allocation. Otherwise the same build script runs on
# a compute node via srun. The result is cached by (source hash, architecture,
# requested Python, uv version) so re-running llmb-install is a no-op unless
# one of those changed.
#
# Framework contract (llmb-install, job_type: srun, requires_gpus: false):
#   $LLMB_WORKLOAD           workload install dir
#   $LLMB_NODE_ARCHITECTURE  compute nodes' CPU architecture from the cluster config
#   $SLURM_ACCOUNT           account for the fallback build allocation
#   $SLURM_PARTITION         CPU partition for the fallback build allocation
#
# Egress needed on the first install, from whichever node builds: astral.sh
# and releases.astral.sh (uv installer; the first redirects to the second),
# github.com (uv-managed CPython, Harbor git dependency), pypi.org.
#
# Overrides:
#   TASKTRACE_CAPTURE_SOURCE_DIR  in-tree source directory (contains pyproject.toml)
#   TASKTRACE_CAPTURE_BUILD       auto | local | srun (default auto: local when
#                                 this node's CPU architecture matches $LLMB_NODE_ARCHITECTURE)
#   TASKTRACE_CAPTURE_PYTHON      CPython version for the venv (default 3.12)
#   TASKTRACE_CAPTURE_UV_VERSION  uv release to bootstrap under $LLMB_WORKLOAD/uv
#                                 (default 0.9.28)

set -eu -o pipefail

export TT_TAG="[install_tasktrace_capture]" # consumed by the sourced library
# shellcheck source=tasktrace_install_lib.sh
source "$(dirname "${BASH_SOURCE[0]}")/tasktrace_install_lib.sh"
tt_require_bash
export LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD not set (framework should provide).}
tt_locate_repo # COMMON_DIR, REPO_ROOT (setup tasks run inline, so BASH_SOURCE is this file)
TASKTRACE_CAPTURE_SOURCE_DIR="${TASKTRACE_CAPTURE_SOURCE_DIR:-$REPO_ROOT/inference/tools/tasktrace-capture}"
TASKTRACE_CAPTURE_BUILD="${TASKTRACE_CAPTURE_BUILD:-auto}"
TASKTRACE_CAPTURE_PYTHON="${TASKTRACE_CAPTURE_PYTHON:-3.12}"
TASKTRACE_CAPTURE_UV_VERSION="${TASKTRACE_CAPTURE_UV_VERSION:-0.9.28}"

VENV_DIR="$LLMB_WORKLOAD/venv"
UV_DIR="$LLMB_WORKLOAD/uv"
VERSION_FILE="$VENV_DIR.version"
BUILD_SCRIPT="$COMMON_DIR/tasktrace_capture_venv_build.sh"
mkdir -p "$LLMB_WORKLOAD"

# --- Where to build ----------------------------------------------------------------
HOST_ARCH="$(uname -m)"
NODE_ARCH="${LLMB_NODE_ARCHITECTURE:-}"
case "$TASKTRACE_CAPTURE_BUILD" in
    local) BUILD_WHERE=local ;;
    srun) BUILD_WHERE=srun ;;
    auto)
        if [[ -z $NODE_ARCH ]]; then
            BUILD_WHERE=srun
            echo "[install_tasktrace_capture] LLMB_NODE_ARCHITECTURE is not set; building on a compute node to be safe."
        elif [[ $HOST_ARCH == "$NODE_ARCH" ]]; then
            BUILD_WHERE=local
        else
            BUILD_WHERE=srun
        fi
        ;;
    *)
        echo "[install_tasktrace_capture] ERROR: TASKTRACE_CAPTURE_BUILD must be auto, local or srun (got '$TASKTRACE_CAPTURE_BUILD')." >&2
        exit 1
        ;;
esac
EXPECT_ARCH="${NODE_ARCH:-$HOST_ARCH}"

# --- Locate the source ---------------------------------------------------------
SRC_DIR="$TASKTRACE_CAPTURE_SOURCE_DIR"
[[ -f "$SRC_DIR/pyproject.toml" ]] || {
    echo "[install_tasktrace_capture] ERROR: no tasktrace-capture source at $SRC_DIR (expected the vendored inference/tools/tasktrace-capture; set TASKTRACE_CAPTURE_SOURCE_DIR to override)." >&2
    exit 1
}
SOURCE_ID="$(tt_source_id "$SRC_DIR" .venv)"
SOURCE_COMMIT="$(tt_source_commit "$SRC_DIR")"
echo "[install_tasktrace_capture] installing from in-tree source $SRC_DIR"
[[ -f "$SRC_DIR/uv.lock" ]] || {
    echo "[install_tasktrace_capture] ERROR: $SRC_DIR has no uv.lock; the install requires a locked dependency set." >&2
    exit 1
}
echo "[install_tasktrace_capture] source id: $SOURCE_ID"

# --- Cache check (source id + architecture), before requesting any allocation ----
if [[ -x $VENV_DIR/bin/tasktrace-capture ]] && tt_version_matches "$VERSION_FILE" "source=$SOURCE_ID" "arch=$EXPECT_ARCH" \
    "python_requested=$TASKTRACE_CAPTURE_PYTHON" "uv=$TASKTRACE_CAPTURE_UV_VERSION"; then
    echo "[install_tasktrace_capture] up to date: $(tr '\n' ' ' < "$VERSION_FILE")"
    exit 0
fi

# --- The build itself: common/tasktrace_capture_venv_build.sh, here or on a compute node ---
export TT_VENV_DIR="$VENV_DIR" TT_UV_DIR="$UV_DIR" TT_SRC_DIR="$SRC_DIR" TT_SOURCE_ID="$SOURCE_ID" \
    TT_SOURCE_COMMIT="$SOURCE_COMMIT" TT_PYTHON="$TASKTRACE_CAPTURE_PYTHON" \
    TT_UV_VERSION="$TASKTRACE_CAPTURE_UV_VERSION" TT_EXPECT_ARCH="$EXPECT_ARCH" \
    TT_LIB="$COMMON_DIR/tasktrace_install_lib.sh"

if [[ $BUILD_WHERE == local ]]; then
    echo "[install_tasktrace_capture] building here on $(hostname) ($HOST_ARCH matches the compute nodes)"
    bash "$BUILD_SCRIPT"
else
    : "${SLURM_ACCOUNT:?SLURM_ACCOUNT not set; needed to build on a compute node.}"
    : "${SLURM_PARTITION:?SLURM_PARTITION not set; needed to build on a compute node.}"
    echo "[install_tasktrace_capture] building on a compute node ($SLURM_ACCOUNT, $SLURM_PARTITION): this node is $HOST_ARCH, the compute nodes are $EXPECT_ARCH"
    # Intentionally NOT passing --gpus/--gpus-per-node (same as the serving
    # recipes' srun installs): on clusters without GRES configured that flag
    # fails with "Invalid generic resource (gres) specification". This task is
    # requires_gpus: false, so llmb-install injects the CPU partition here and,
    # only if the cluster config declares one for it, that partition's GRES via
    # SLURM_GPUS_PER_NODE; srun honours the variable. The build only needs a
    # node with the compute nodes' CPU architecture. srun exports the
    # environment, including the TT_* inputs, by default.
    srun --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" \
        --time=00:30:00 --nodes=1 --ntasks=1 --cpus-per-task=8 \
        bash "$BUILD_SCRIPT"
fi

echo "[install_tasktrace_capture] done: $VENV_DIR ($(tr '\n' ' ' < "$VERSION_FILE"))"
