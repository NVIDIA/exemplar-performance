#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=nemotron_3

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

export FW_VERSION=26.08.01

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

DTYPE=${DTYPE:-bf16}
DTYPE=${DTYPE,,}
FP8_RECIPE=${FP8_RECIPE:-mx}
FP8_RECIPE=${FP8_RECIPE,,}
JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}
MODEL_SIZE=${MODEL_SIZE:-120b}
MODEL_SIZE=${MODEL_SIZE,,}

if [[ $MODEL_SIZE == "120b" ]]; then
    MODEL_RECIPE_NAME="nemotron_3_super"
elif [[ $MODEL_SIZE == "550b" ]]; then
    MODEL_RECIPE_NAME="nemotron_3_ultra"
else
    echo "Error: Invalid MODEL_SIZE '$MODEL_SIZE'. Supported values: 120b, 550b" >&2
    exit 1
fi

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
PROFILE_START_STEP=${PROFILE_START_STEP:-45}
PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING,,}
TIME_LIMIT=${TIME_LIMIT:-"00:45:00"}
MAX_STEPS=${MAX_STEPS:-50}

if [[ $DTYPE == "fp8" ]]; then
    COMPUTE_TYPE=${DTYPE}_${FP8_RECIPE}
else
    COMPUTE_TYPE=${DTYPE}
fi

CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

export HF_HOME="$LLMB_INSTALL/.cache/huggingface"
CONTAINER_MOUNTS="$HF_HOME"

if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

CONFIG_OVERRIDES=""
if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts $CONTAINER_MOUNTS"
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    PROFILE_RANKS=${PROFILE_RANKS:-$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)) | sed 's/,$//')}
    CONFIG_OVERRIDES+=" --profiling_ranks=$PROFILE_RANKS"
    CONFIG_OVERRIDES+=" --nsys_trace=cuda "
    CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL,--discard-environment=true  "
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" --profiling_gpu_metrics "
    fi
fi

if [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --pytorch_profiler true "
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

CONFIG_OVERRIDES+=" --enable_pct_binding $ENABLE_PCT_BINDING "

if [[ $GPU_TYPE == "gb300" ]] || [[ $GPU_TYPE == "gb200" ]]; then
    GPUS_PER_NODE=4
elif [[ $GPU_TYPE == "b300" ]] || [[ $GPU_TYPE == "b200" ]]; then
    GPUS_PER_NODE=8
else
    echo "Error: Invalid GPU_TYPE '$GPU_TYPE'. Supported values: gb300, gb200, b300, b200" >&2
    exit 1
fi

EXECUTOR_ARGS=()
if [[ $CLUSTER_TYPE == "kubernetes" ]]; then
    EXECUTOR_ARGS=(--nvcre_namespace "${NVCRE_NAMESPACE:?NVCRE_NAMESPACE required for kubernetes}")
    [[ -n ${NVCRE_IMAGE_PULL_SECRET:-} ]] && EXECUTOR_ARGS+=(--nvcre_image_pull_secret "$NVCRE_IMAGE_PULL_SECRET")
    [[ -n ${NVCRE_WORKDIR_PVC:-} ]] && EXECUTOR_ARGS+=(--nvcre_workdir_pvc "$NVCRE_WORKDIR_PVC")
    [[ -n ${NVCRE_NODE_SELECTOR_JSON:-} ]] && EXECUTOR_ARGS+=(--nvcre_node_selector_json "$NVCRE_NODE_SELECTOR_JSON")
    EXECUTOR_ARGS+=(--nvcre_timeout_per_job "${NVCRE_TIMEOUT_PER_JOB:-12h}")
    [[ -n ${NVCRE_GANG_SCHEDULER_NAME:-} ]] && EXECUTOR_ARGS+=(--nvcre_gang_scheduler_name "$NVCRE_GANG_SCHEDULER_NAME")
    # Run this checkout's Megatron-Bridge commit in the pods instead of the image's copy.
    EXECUTOR_ARGS+=(--nvcre_use_mbridge_commit)
    [[ -n ${NVCRE_HF_TOKEN_SECRET_NAME:-} ]] && EXECUTOR_ARGS+=(--nvcre_hf_token_secret_name "$NVCRE_HF_TOKEN_SECRET_NAME" --nvcre_hf_token_secret_key "${NVCRE_HF_TOKEN_SECRET_KEY:-token}")
elif [[ $CLUSTER_TYPE == "slurm" ]]; then
    read -ra SLURM_ARG_WORDS <<< "$SLURM_ARGS"
    EXECUTOR_ARGS=(--account "$SBATCH_ACCOUNT" --partition "$SBATCH_PARTITION" --time_limit "$TIME_LIMIT" ${SLURM_ARG_WORDS[@]+"${SLURM_ARG_WORDS[@]}"})
else
    echo "Unsupported CLUSTER_TYPE=${CLUSTER_TYPE}. Aborting"
    exit 1
fi

# run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python3 scripts/performance/setup_experiment.py \
    --container_image $IMAGE \
    --compute_dtype $COMPUTE_TYPE \
    --gpu $GPU_TYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --offline \
    --model_family_name nemotronh \
    --model_recipe_name ${MODEL_RECIPE_NAME} \
    ${CONFIG_OVERRIDES} \
    --log_dir $NEMORUN_HOME \
    --max_steps $MAX_STEPS \
    --packager none \
    "${EXECUTOR_ARGS[@]}" \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}

popd
