#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_FAMILY=gpt_oss
export MODEL_RECIPE=gpt_oss_120b

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

if [[ $GPU_TYPE == "h100" ]]; then
    export FW_VERSION=26.06.01
else
    export FW_VERSION=26.08.01
fi

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_FAMILY}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
DTYPE=${DTYPE:-bf16}
DTYPE=${DTYPE,,}
if [[ -z ${FP8_RECIPE:-} ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        FP8_RECIPE="cs"
    else
        FP8_RECIPE="mx"
    fi
fi
FP8_RECIPE=${FP8_RECIPE,,}
CONFIG_VARIANT=${CONFIG_VARIANT:-v2}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
ENABLED_GPU_METRICS=${ENABLE_GPU_METRICS:-false}
ENABLED_GPU_METRICS=${ENABLED_GPU_METRICS,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING,,}
PROFILE_START_STEP=${PROFILE_START_STEP:-45}
PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

CONTAINER_MOUNTS=""
export HF_HOME="$LLMB_INSTALL/.cache/huggingface"
CONTAINER_MOUNTS="$HF_HOME"
if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

CONFIG_OVERRIDES=""

TIME_LIMIT=${TIME_LIMIT:-"00:30:00"}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-$MAX_STEPS} # Default: save checkpoint at end of training

if [[ -n ${TP-} ]]; then
    CONFIG_OVERRIDES+="-tp $TP "
fi
if [[ -n ${PP-} ]]; then
    CONFIG_OVERRIDES+="-pp $PP "
fi
if [[ -n ${CP-} ]]; then
    CONFIG_OVERRIDES+="-cp $CP "
fi
if [[ -n ${VP-} ]]; then
    CONFIG_OVERRIDES+="-vp $VP "
fi
if [[ -z ${EP-} ]] && { [[ $GPU_TYPE == "b300" || $GPU_TYPE == "b200" ]] || ((JOB_TOTAL_GPUS % 72 == 0)); }; then
    EP=8
fi
if [[ -n ${EP-} ]]; then
    CONFIG_OVERRIDES+="-ep $EP "
fi
if [[ -n ${ET-} ]]; then
    CONFIG_OVERRIDES+="-et $ET "
fi
if [[ -z ${MBS-} ]] && ((JOB_TOTAL_GPUS % 72 == 0)); then
    MBS=5
fi
if [[ -n ${MBS-} ]]; then
    CONFIG_OVERRIDES+="-mb $MBS "
fi
if [[ -n ${GBS-} ]]; then
    CONFIG_OVERRIDES+="-gb $GBS "
fi

# Checkpoint configuration
if [[ $ENABLE_CHECKPOINT == true ]]; then
    # Set checkpoint directory if specified (defaults to experiment dir)
    if [[ -n ${CHECKPOINT_DIR-} ]]; then
        CONFIG_OVERRIDES+=" --save_dir=$CHECKPOINT_DIR "
    fi

    # Set checkpoint interval (defaults to MAX_STEPS if not specified)
    CONFIG_OVERRIDES+=" --save_interval=$CHECKPOINT_INTERVAL "
fi

if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
    MAX_STEPS=1
    CONFIG_OVERRIDES+=" --load_dir=$LOAD_CHECKPOINT_PATH "
fi

# Pass MAX_STEPS to training configuration (must be after checkpoint section which may set MAX_STEPS=1)
CONFIG_OVERRIDES+=" --max_steps=$MAX_STEPS "

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts=$CONTAINER_MOUNTS"
fi

if [[ $PROFILE_ENABLED == "true" ]] && [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    echo "Error: ENABLE_PROFILE and ENABLE_PYTORCH_PROFILE are mutually exclusive." >&2
    exit 1
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    PROFILE_RANKS=$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)) | sed 's/,$//')
    CONFIG_OVERRIDES+=" --profiling_ranks=$PROFILE_RANKS"
    CONFIG_OVERRIDES+=" --nsys_trace=cuda,nvtx "
    if [[ $GPU_TYPE == "h100" || $GPU_TYPE == "b300" ]]; then
        CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL "
    else
        CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL,--discard-environment=true "
    fi
    if [[ $ENABLED_GPU_METRICS == true ]]; then
        CONFIG_OVERRIDES+=" --profiling_gpu_metrics "
    fi
fi

if [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --pytorch_profiler true "
fi

if [[ $DTYPE == "fp8" ]]; then
    case "$FP8_RECIPE" in
        cs | mx)
            CONFIG_OVERRIDES+=" --compute_dtype fp8_$FP8_RECIPE "
            ;;
        *)
            echo "Error: Other FP8 types are not allowed"
            exit 1
            ;;
    esac

else
    CONFIG_OVERRIDES+=" --compute_dtype $DTYPE "
fi

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

CONFIG_OVERRIDES+=" --enable_pct_binding $ENABLE_PCT_BINDING "

if [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "gb300" ]]; then
    GPUS_PER_NODE=4
else
    GPUS_PER_NODE=8
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
    --model_family_name $MODEL_FAMILY \
    --model_recipe_name $MODEL_RECIPE \
    --task $WORKLOAD_TYPE \
    --gpu $GPU_TYPE \
    --container_image $IMAGE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --config_variant $CONFIG_VARIANT \
    --enable_vboost $ENABLE_VBOOST \
    --offline \
    $CONFIG_OVERRIDES \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    --packager none \
    "${EXECUTOR_ARGS[@]}" \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}
popd
