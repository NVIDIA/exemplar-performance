#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

#SBATCH --job-name="llmb_collector:microbenchmark"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:15:00

set -euo pipefail

export WORKLOAD_TYPE=microbenchmark
export WORKLOAD=llmb_collector

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the LLMB installation directory}
export LLMB_REPO=${LLMB_REPO:-$LLMB_INSTALL/llmb_repo}
export LLMB_EXPERIMENT_DIR=${LLMB_EXPERIMENT_DIR:?LLMB_EXPERIMENT_DIR must be set by llmb-run}

# llmb-install stages uv/uvx for both the login-node and selected compute-node
# architectures. Resolve it here, after Slurm has placed us on the compute node.
ARCH=$(uname -m)
LLMB_BIN=${LLMB_BIN:-$LLMB_INSTALL/bin/$ARCH}
UVX=$LLMB_BIN/uvx

if [[ ! -x $UVX ]]; then
    echo "llmb-collector: uvx is missing for compute-node architecture '$ARCH': $UVX" >&2
    echo "Re-run llmb-install with the correct compute-node architecture." >&2
    exit 1
fi

# uvx stores its ephemeral environment and downloaded packages in UV_CACHE_DIR.
# Keep both that cache and any uv-managed Python installation out of $HOME.
export UV_CACHE_DIR=${UV_CACHE_DIR:-$LLMB_INSTALL/.cache/uv}
export UV_PYTHON_INSTALL_DIR=${UV_PYTHON_INSTALL_DIR:-$LLMB_INSTALL/.cache/uv/python}
mkdir -p "$UV_CACHE_DIR" "$UV_PYTHON_INSTALL_DIR" "$LLMB_EXPERIMENT_DIR"

SNAPSHOT_ROOT=$LLMB_REPO/microbenchmarks/llmb_collector/source_snapshot
COLLECTOR_SOURCE=$SNAPSHOT_ROOT/llmb_collector
CAPABILITIES_SOURCE=$SNAPSHOT_ROOT/llmb_capabilities
for source_dir in "$COLLECTOR_SOURCE" "$CAPABILITIES_SOURCE"; do
    if [[ ! -f $source_dir/pyproject.toml ]]; then
        echo "llmb-collector: source snapshot is missing: $source_dir" >&2
        exit 1
    fi
done

OUTPUT=$LLMB_EXPERIMENT_DIR/llmb-collector.json

echo "Running llmb-collector source snapshot on $(hostname) ($ARCH)"
echo "Collector source: $COLLECTOR_SOURCE"
echo "Capabilities source: $CAPABILITIES_SOURCE"
echo "uv cache: $UV_CACHE_DIR"

"$UVX" --from "$COLLECTOR_SOURCE" --with "$CAPABILITIES_SOURCE" llmb-collector collect \
    --host \
    --network \
    --env \
    --no-compact \
    --output "$OUTPUT"

echo "llmb-collector report: $OUTPUT"
