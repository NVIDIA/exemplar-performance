#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Setup task 3/4 (job_type:local).
#
# `make setup` downloads NATS/ETCD/uv binaries that srt-slurm needs at
# benchmark time. srtslurm.yaml is already in place (task 2 wrote it), so
# Makefile:setup's interactive read-prompt block is short-circuited by its
# existing-file check.
#
# Grace SKUs are aarch64; metadata.yaml.gpu_configs already restricts these
# recipes to gb200/gb300, so no runtime arch check is needed here.

set -eu -o pipefail

export LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD not set (framework should provide).}

SRT_SLURM_DIR="$LLMB_WORKLOAD/srt-slurm"
[[ -f "$SRT_SLURM_DIR/srtslurm.yaml" ]] || {
    echo "[setup_srtslurm] ERROR: srtslurm.yaml missing — generate_srtslurm_yaml must run first." >&2
    exit 1
}

cd "$SRT_SLURM_DIR"
echo "[setup_srtslurm] make setup ARCH=aarch64 (NATS/ETCD/uv download)"
make setup ARCH=aarch64
echo "[setup_srtslurm] done."
