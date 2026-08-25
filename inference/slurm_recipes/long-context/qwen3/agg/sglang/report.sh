#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Grab concurrency, request latency, and effective throughput from each run in
# the sweep and show them as a table. Convenience helper — run by hand after a
# `llmb-run submit` sweep finishes. NOT invoked by llmb (workload_type:inference
# is exempt from the post-processing pipeline).
#
# Usage:
#   export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/inference_qwen3_long
#   ./report.sh
# (LLMB_WORKLOAD is derived from LLMB_INSTALL if only the latter is exported.)

set -eu -o pipefail

SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"

if [[ -z ${LLMB_WORKLOAD:-} ]]; then
    if [[ -n ${LLMB_INSTALL:-} ]]; then
        LLMB_WORKLOAD="$LLMB_INSTALL/workloads/inference_qwen3_long"
    else
        echo "[report] ERROR: set LLMB_WORKLOAD or LLMB_INSTALL first" >&2
        exit 1
    fi
fi

OUTPUTS="$LLMB_WORKLOAD/srt-slurm/outputs"
[[ -d $OUTPUTS ]] || {
    echo "[report] ERROR: no outputs dir at $OUTPUTS. Has a benchmark run finished yet?" >&2
    exit 1
}

cd "$LLMB_WORKLOAD/srt-slurm"
"$SCRIPT_DIR/extract_metrics.sh" -d ./outputs/
