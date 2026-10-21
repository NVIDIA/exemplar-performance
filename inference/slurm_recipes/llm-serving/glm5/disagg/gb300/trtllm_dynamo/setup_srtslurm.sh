#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu -o pipefail

: "${LLMB_WORKLOAD:?LLMB_WORKLOAD is required}"
: "${LLMB_NODE_ARCHITECTURE:?LLMB_NODE_ARCHITECTURE is required}"

srt_root="$LLMB_WORKLOAD/srt-slurm"
test -d "$srt_root/.git" || {
    echo "[setup_srtslurm] ERROR: pinned checkout not found: $srt_root" >&2
    exit 1
}

echo "[setup_srtslurm] make setup ARCH=$LLMB_NODE_ARCHITECTURE"
make -C "$srt_root" setup "ARCH=$LLMB_NODE_ARCHITECTURE"
