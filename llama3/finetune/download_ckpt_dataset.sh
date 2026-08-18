#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# This script is intended to be run by the LLMB installer as a *setup task*
# (job_type: nemo2) after dependencies have already been installed.
# It imports the base checkpoint and downloads the training dataset for
# Llama-3 70B finetuning (LoRa).

set -eu -o pipefail

export WORKLOAD_TYPE=finetune
export MODEL_NAME=llama3
export FW_VERSION=26.06.01

# --- Required environment variables (provided by the installer) ---
: "${GPU_TYPE:?Required variable GPU_TYPE}"
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"
: "${LLMB_WORKLOAD:?Provided by installer}"
export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# Directory for cached objects
DATASET_ROOT=${DATASET_ROOT:-$LLMB_WORKLOAD/checkpoint_and_dataset/datasets}
SQUAD_REPO=${SQUAD_REPO:-$LLMB_WORKLOAD/squad}
mkdir -p "$LLMB_WORKLOAD/checkpoint_and_dataset" "$DATASET_ROOT"
export HF_HOME=${HF_HOME:-$LLMB_INSTALL/.cache/huggingface}
export HF_HUB_CACHE=${HF_HUB_CACHE:-$HF_HOME/hub}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$DATASET_ROOT}
mkdir -p "$HF_HOME" "$HF_HUB_CACHE" "$HF_DATASETS_CACHE"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD/checkpoint_and_dataset}
export NEMORUN_HOME=$LLMB_WORKLOAD

SCRIPT_NAME="$LLMB_WORKLOAD/Megatron-Bridge/examples/conversion/convert_checkpoints.py"
HF_MODEL_PATH=${HF_MODEL_PATH:-$LLMB_WORKLOAD/Meta-Llama-3-70B}

if [[ ! -d $HF_MODEL_PATH ]]; then
    echo "ERROR: Expected local Hugging Face model repo at $HF_MODEL_PATH." >&2
    echo "Run llmb-install with the llama3 finetune metadata downloads before running this setup task." >&2
    exit 1
fi

if [[ ! -d $SQUAD_REPO ]]; then
    echo "ERROR: Expected local SQuAD dataset repo at $SQUAD_REPO." >&2
    echo "Run llmb-install with the llama3 finetune metadata downloads before running this setup task." >&2
    exit 1
fi

export IMAGE="${IMAGE:-${LLMB_INSTALL}/images/nvidia+nemo+${FW_VERSION}.sqsh}"

TIME_LIMIT=${TIME_LIMIT:-"00:55:00"}
GPU_TYPE=${GPU_TYPE,,}
CONTAINER_MOUNTS="$LLMB_WORKLOAD:$LLMB_WORKLOAD,$HF_HOME:$HF_HOME"

echo "Preparing SQuAD dataset cache..."
srun \
    --job-name="infra_rd_gsw-llama3.squad-cache" \
    --time="$TIME_LIMIT" \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-writable \
    --no-container-mount-home \
    python3 -c 'import sys; from datasets import load_dataset; load_dataset(sys.argv[1], cache_dir=sys.argv[2])' \
    "$SQUAD_REPO" "$DATASET_ROOT"

# Loading a local checkout keys the prepared cache by its directory name,
# while offline loading by Hub ID looks for "rajpurkar___squad".
# Expose the same prepared cache under the canonical key used by Megatron-Bridge.
SQUAD_CACHE_KEY=$(basename "$SQUAD_REPO")
CANONICAL_SQUAD_CACHE="$DATASET_ROOT/rajpurkar___squad"
if [[ ! -e $CANONICAL_SQUAD_CACHE || -L $CANONICAL_SQUAD_CACHE ]]; then
    ln -sfn "$SQUAD_CACHE_KEY" "$CANONICAL_SQUAD_CACHE"
fi

# Change to Megatron-Bridge directory
pushd "$LLMB_WORKLOAD/Megatron-Bridge" > /dev/null

# Run conversion
srun \
    --job-name="infra_rd_gsw-llama3.checkpoint-convert" \
    --time="$TIME_LIMIT" \
    --container-image="$IMAGE" \
    --container-mounts="$CONTAINER_MOUNTS" \
    --container-writable \
    --no-container-mount-home \
    python3 "$SCRIPT_NAME" import \
    --hf-model "$HF_MODEL_PATH" \
    --megatron-path "$LLMB_WORKLOAD/checkpoint_and_dataset/llama3_70b" \
    --torch-dtype bfloat16

popd > /dev/null

echo "Checkpoint conversion completed!"
