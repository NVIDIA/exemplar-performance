#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

export WORKLOAD_TYPE=microbenchmark
export MODEL_NAME=nccl
export FW_VERSION=26.08.01

# All three variables are set by the llmb-install script.
LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
LLMB_WORKLOAD=${LLMB_WORKLOAD:?Please set LLMB_WORKLOAD to the path of the workload directory}
GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

# Default FW_VERSION from metadata container tag unless explicitly overridden.
FW_VERSION=${FW_VERSION:-}
METADATA_FILE="$LLMB_INSTALL/llmb_repo/microbenchmarks/nccl/metadata.yaml"
if [ -z "$FW_VERSION" ] && [ -f "$METADATA_FILE" ]; then
    FW_VERSION=$(grep -Eo "nvcr\\.io[#/][^/[:space:]]+/nemo:[^'\"[:space:]]+" "$METADATA_FILE" | head -n1 | sed -E 's#.*:##' || true)
fi
if [ -z "$FW_VERSION" ]; then
    echo "Unable to determine FW_VERSION for NCCL launch generation. Set FW_VERSION or update $METADATA_FILE." >&2
    exit 1
fi
export FW_VERSION

# Repo downloaded by installer.
export NVCOMM_DIR=$LLMB_WORKLOAD/Nvidia-Comms-Perf-Suite

cp -r config "$NVCOMM_DIR/"
# Populate the user TOML template with the resolved framework image version.
sed -i -e "s|__FW_VERSION__|$FW_VERSION|g" "$NVCOMM_DIR/config/nemofw.user.toml"

pushd "$NVCOMM_DIR"

# Require uv in PATH (do not perform unpinned runtime installs here)
if ! command -v uv &> /dev/null; then
    echo "uv not found in PATH. Run install.sh first before setup_nccl_launcher.sh." >&2
    exit 1
fi

uv build
uv pip install dist/nvcomms_perf*.whl

nvcomms-perf generate-job-script --run-type=container --system $GPU_TYPE \
    --systems-toml config/systems.toml \
    --user-toml config/nemofw.user.toml \
    --testset-toml config/testset.toml > "$LLMB_INSTALL/llmb_repo/microbenchmarks/nccl/launch.sh"

# shellcheck disable=SC2016
sed -i -e 's|NVCOMMS_PERF_TOOLS_WORKSPACE:-.|NVCOMMS_PERF_TOOLS_WORKSPACE:-$LLMB_INSTALL/workloads/microbenchmark_nccl/experiments/|g' "$LLMB_INSTALL/llmb_repo/microbenchmarks/nccl/launch.sh"

mkdir -p "$LLMB_INSTALL/workloads/microbenchmark_nccl/experiments/"

popd
