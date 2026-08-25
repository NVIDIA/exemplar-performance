#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Setup task 4/4 (job_type:local).
#
# `srtctl install kimi26-trtllm ...` submits a SLURM job that fetches:
#   - Kimi-K2.6 NVFP4 weights from HF
#   - tensorrtllm-runtime container .sqsh
# srtctl install RETURNS AFTER SUBMISSION (srtctl/cli/install.py),
# so this script then polls squeue + sacct until the install job exits.
#
# The .sqsh stays where srtctl writes it
# ($LLMB_WORKLOAD/srt-slurm/install/containers/), which is under $LLMB_INSTALL.
# We don't move it into $LLMB_IMAGE_FOLDER — that variable is reserved for
# static, install-agnostic shared images, not for install-specific artifacts
# like the .sqsh srtctl produces here.
#
# job_type MUST be `local`: srtctl install explicitly refuses to run
# inside a SLURM job (srtctl/cli/install.py).

set -eu -o pipefail

export LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD not set (framework should provide).}
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set. Export it before running llmb-install, or set it in the environment block of cluster_config.yaml.}

SRT_SLURM_DIR="$LLMB_WORKLOAD/srt-slurm"

# Recipe-specific install parameters. Keep these grouped near the top so
# future model bumps are one-place edits.
INSTALL_NAME="kimi26-trtllm"
HF_REPO_ID="nvidia/Kimi-K2.6-NVFP4"
MODEL_ALIAS="nvidia/Kimi-K2.6-NVFP4"
CONTAINER_IMAGE="nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.1.0-dev.2"

# srtctl install defaults --venv to <srtctl_root>/.venv (the compute-side
# aarch64 venv install_srtslurm built). Pass it explicitly to harden
# against future srtctl default changes and document the dual-venv layout.
COMPUTE_VENV="$SRT_SLURM_DIR/.venv"
[[ -x "$COMPUTE_VENV/bin/srtctl" ]] || {
    echo "[install_model] ERROR: compute-side venv missing or broken at $COMPUTE_VENV. install_srtslurm must run first." >&2
    exit 1
}
if ! command -v srtctl > /dev/null 2>&1; then
    echo "[install_model] ERROR: login-side srtctl not on PATH. install_srtslurm should have uv-pip-installed it into the LLMB workload venv." >&2
    exit 1
fi

cd "$SRT_SLURM_DIR"

echo "[install_model] srtctl install $INSTALL_NAME (HF: $HF_REPO_ID, container: $CONTAINER_IMAGE)"

INSTALL_OUT=$(srtctl install "$INSTALL_NAME" \
    --hf-repo-id "$HF_REPO_ID" \
    --model-alias "$MODEL_ALIAS" \
    --container-image "$CONTAINER_IMAGE" \
    --venv "$COMPUTE_VENV" | tee /dev/stderr)

# shellcheck disable=SC2016  # srtctl prints literal "squeue -u $USER -j NNN" — match it verbatim
JOB_ID=$(printf "%s\n" "$INSTALL_OUT" | grep -oE 'squeue -u \$USER -j [0-9]+' | tail -1 | awk '{print $NF}' || true)
if [[ -z ${JOB_ID:-} ]]; then
    echo "[install_model] ERROR: could not parse install job id from srtctl output." >&2
    exit 1
fi

# Persist the submitted JOB_ID so install_srtslurm.sh (task 1) can refuse to
# rm -rf .venv on a retry while this install is still active. The file is
# removed once the job reaches a terminal state (COMPLETED/FAILED/CANCELLED);
# it is intentionally kept if the local poll is interrupted or hits the
# timeout below, so a re-run is blocked until the user clears the active job.
JOB_ID_FILE="$SRT_SLURM_DIR/.llmb-install-job-id"
printf "%s\n" "$JOB_ID" > "$JOB_ID_FILE"

# HF fetch + enroot import runs 5-40 min in observed runs. 5-min poll is
# coarse enough to be polite to the controller; 1.5 h ceiling is ~2x the
# longest observed install, so a hang fails fast rather than blocking forever.
POLL_SEC=300
MAX_WAIT_SEC=$((90 * 60)) # 1.5 h
echo "[install_model] waiting for install job $JOB_ID (poll every ${POLL_SEC}s, max ${MAX_WAIT_SEC}s)..."
ELAPSED=0
while squeue -h -j "$JOB_ID" 2> /dev/null | grep -q .; do
    if ((ELAPSED >= MAX_WAIT_SEC)); then
        echo "[install_model] ERROR: install job $JOB_ID still running after ${MAX_WAIT_SEC}s — aborting wait. Inspect $SRT_SLURM_DIR/install/install_${INSTALL_NAME}_${JOB_ID}.log" >&2
        echo "[install_model]        The job is still active — install_srtslurm will refuse to rebuild .venv on retry until it terminates." >&2
        exit 1
    fi
    sleep "$POLL_SEC"
    ELAPSED=$((ELAPSED + POLL_SEC))
done

# squeue may purge the finished job before we read it; sacct is authoritative.
STATE=$(sacct -nXP -j "$JOB_ID" -o State 2> /dev/null | head -1 | awk '{print $1}')
echo "[install_model] install job $JOB_ID final state: ${STATE:-UNKNOWN}"
if [[ ${STATE:-} != "COMPLETED" ]]; then
    # Terminal state (FAILED/CANCELLED/etc.) — job no longer holds resources,
    # so we clear the retry-block guard.
    rm -f "$JOB_ID_FILE"
    echo "[install_model] ERROR: install job did not complete cleanly. Check $SRT_SLURM_DIR/install/install_${INSTALL_NAME}_${JOB_ID}.log" >&2
    exit 1
fi

# Success — clear the retry-block guard.
rm -f "$JOB_ID_FILE"

echo "[install_model] done."
