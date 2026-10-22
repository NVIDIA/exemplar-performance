#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# This script is intended to be run by the LLMB installer as a *setup task*
# It imports the base checkpoint and downloads the training dataset for
# Llama 2 70B finetuning (LoRA).

set -eu -o pipefail

export WORKLOAD_TYPE=finetune
export MODEL_NAME=llama2
export FW_VERSION=26.06.01

# Skip checkpoint download
SKIP_CHECKPOINT_DOWNLOAD=${SKIP_CHECKPOINT_DOWNLOAD:-false}
if [[ $SKIP_CHECKPOINT_DOWNLOAD == true ]]; then
    echo "Skipping checkpoint download"
    exit 0
fi

# --- Required environment variables (provided by the installer) ---
: "${HF_TOKEN:?Required variable Hugging Face token}"
: "${GPU_TYPE:?Required variable GPU_TYPE}"
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"
: "${LLMB_WORKLOAD:?Provided by installer}"
export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# Directory for cached objects
mkdir -p "$LLMB_WORKLOAD/checkpoint"
export HF_HOME=${HF_HOME:-$LLMB_WORKLOAD/checkpoint}
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD/checkpoint}
export NEMORUN_HOME=$LLMB_WORKLOAD

SCRIPT_NAME="$LLMB_WORKLOAD/Megatron-Bridge/examples/conversion/convert_checkpoints.py"

export IMAGE="${IMAGE:-${LLMB_INSTALL}/images/nvidia+nemo+${FW_VERSION}.sqsh}"

TIME_LIMIT=${TIME_LIMIT:-"00:55:00"}
GPU_TYPE=${GPU_TYPE,,}

# Change to Megatron-Bridge directory
pushd "$LLMB_WORKLOAD/Megatron-Bridge" > /dev/null

# Run conversion
srun \
    --time="$TIME_LIMIT" \
    --container-image="$IMAGE" \
    --container-mounts="$LLMB_WORKLOAD:$LLMB_WORKLOAD" \
    --container-writable \
    --no-container-mount-home \
    python3 $SCRIPT_NAME import \
    --hf-model meta-llama/Llama-2-70b-hf \
    --megatron-path $LLMB_WORKLOAD/checkpoint/llama2_70b \
    --torch-dtype bfloat16

popd > /dev/null

echo "Checkpoint conversion completed!"
