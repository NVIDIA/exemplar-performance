#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=00:30:00

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# LLMB environment variables
export WORKLOAD_TYPE=rl
export MODEL_NAME=deepseek-v3
export GPU_TYPE=${GPU_TYPE:-gb200}
export GPU_TYPE=${GPU_TYPE,,}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-256}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

# RL experiment environment variables
FW_VERSION=v0.6.0
export CONTAINER=$LLMB_INSTALL/images/nvidia+nemo-rl+${FW_VERSION}.sqsh
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set in environment variables}
MODEL_PATH=${MODEL_PATH:-$LLMB_WORKLOAD/DeepSeek-V3-BF16}
export MOUNTS=${LLMB_INSTALL}:${LLMB_INSTALL},${LLMB_REPO}:${LLMB_REPO}
MAX_STEPS=${MAX_STEPS:-10}
ASYNC_MODE=${ASYNC_MODE:-true}

if [[ $GPU_TYPE == "gb200" ]]; then
    export GPUS_PER_NODE=4
fi

ASYNC_MODE_ARG=""
if [[ $ASYNC_MODE == "true" ]]; then
    ASYNC_MODE_ARG="-async-1off"
fi

NUM_NODES=$((JOB_TOTAL_GPUS / GPUS_PER_NODE))
JOB_NAME=${JOB_NAME:-grpo-deepseek-v3-${NUM_NODES}n${GPUS_PER_NODE}g${ASYNC_MODE_ARG}}

WANDB_ARGS=""
if [[ -n ${WANDB_API_KEY:-} ]]; then
    export WANDB_API_KEY
    WANDB_ARGS+=" logger.wandb_enabled=True"
    WANDB_ARGS+=" logger.wandb.project=nemo-rl"
    WANDB_ARGS+=" logger.wandb.name=$JOB_NAME"
else
    WANDB_ARGS+=" logger.wandb_enabled=False"
fi

export HF_HOME=$LLMB_INSTALL/.cache/huggingface
export UV_CACHE_DIR=$LLMB_INSTALL/.cache/uv
export BASE_LOG_DIR=$LLMB_EXPERIMENT_DIR

mkdir -p "$HF_HOME" "$UV_CACHE_DIR"

LOG_DIR=$LLMB_EXPERIMENT_DIR/logs
CKPT_DIR=$LLMB_EXPERIMENT_DIR/ckpts
JSON_METRICS=$LLMB_EXPERIMENT_DIR/metrics.json
RUN_LOG=$LLMB_EXPERIMENT_DIR/run.log
CONFIG_PATH=$LLMB_WORKLOAD/NeMo-RL/examples/configs/recipes/llm/performance/${JOB_NAME}.yaml

pushd $LLMB_WORKLOAD/NeMo-RL

TRAIN_CMD="
    uv run examples/run_grpo.py \
    --config $CONFIG_PATH \
    grpo.max_num_steps=$MAX_STEPS \
    policy.model_name=$MODEL_PATH \
    policy.tokenizer.name=$MODEL_PATH \
    logger.log_dir=$LOG_DIR \
    logger.monitor_gpus=True \
    logger.tensorboard_enabled=True \
    checkpointing.enabled=True \
    checkpointing.checkpoint_dir=$CKPT_DIR \
    $WANDB_ARGS \
    $* \
    2>&1 | tee $RUN_LOG"

METRICS_CMD="uv run tests/json_dump_tb_logs.py $LOG_DIR --output_path $JSON_METRICS"

CHECK_CMD="uv run tests/check_metrics.py $JSON_METRICS \
    'median(data[\"train/token_mult_prob_error\"]) < 1.1' \
    'data[\"train/token_mult_prob_error\"][\"10\"] < 1.1' 2>&1 | tee $LLMB_EXPERIMENT_DIR/check_metrics.log"

export COMMAND="
    cd $LLMB_WORKLOAD/NeMo-RL
    $TRAIN_CMD
    $METRICS_CMD
    $CHECK_CMD"

bash ray.sub

rm -rf $CKPT_DIR
