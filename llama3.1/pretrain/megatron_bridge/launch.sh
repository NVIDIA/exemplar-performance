#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=llama3.1

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

if [[ $GPU_TYPE == "h100" ]]; then
    export FW_VERSION=26.06.01
else
    export FW_VERSION=26.08.01
fi

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
CONFIG_VARIANT=${CONFIG_VARIANT:-v2}
CONFIG_VARIANT=${CONFIG_VARIANT,,}
FP8_RECIPE=${FP8_RECIPE:-cs}
FP8_RECIPE=${FP8_RECIPE,,}
MODEL_SIZE=${MODEL_SIZE:-405b}
MODEL_SIZE=${MODEL_SIZE,,}
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

# Time limit: 30 min for 70B, 2 hr for 405B (override with TIME_LIMIT env)
if [[ -z ${TIME_LIMIT:-} ]]; then
    if [[ $MODEL_SIZE == 405b ]]; then
        TIME_LIMIT="02:00:00"
    else
        TIME_LIMIT="00:30:00"
    fi
fi
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT:-false}
ENABLE_CHECKPOINT=${ENABLE_CHECKPOINT,,}
CHECKPOINT_INTERVAL=${CHECKPOINT_INTERVAL:-$MAX_STEPS} # Default: save checkpoint at end of training

if { [[ $GPU_TYPE == "b300" ]] || [[ $GPU_TYPE == "b200" ]]; } && [[ $MODEL_SIZE == "405b" ]]; then
    GBS=$((JOB_TOTAL_GPUS * 6))
fi

if [[ $GPU_TYPE == "b300" ]] && { [[ $MODEL_SIZE == "70b" ]] || [[ $MODEL_SIZE == "405b" ]]; } && [[ $JOB_TOTAL_GPUS -ge 512 ]]; then
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
fi

if { [[ $GPU_TYPE == "b300" ]] || [[ $GPU_TYPE == "b200" ]]; } \
    && [[ $MODEL_SIZE == "70b" ]] && [[ $DTYPE == "fp8" ]] \
    && [[ $JOB_TOTAL_GPUS -ge 128 ]]; then
    FP8_RECIPE=mx
fi

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
if [[ -n ${MBS-} ]]; then
    CONFIG_OVERRIDES+="-mb $MBS "
fi
if [[ -n ${GBS-} ]]; then
    CONFIG_OVERRIDES+="-gb $GBS "
fi

# Set model family and recipe names based on model size
MODEL_FAMILY_NAME="llama"
if [[ $MODEL_SIZE == 405b ]]; then
    MODEL_RECIPE_NAME="llama31_405b"
elif [[ $MODEL_SIZE == 70b ]]; then
    MODEL_RECIPE_NAME="llama3_70b"
else
    echo "Error: Unsupported MODEL_SIZE: $MODEL_SIZE"
    exit 1
fi

# H100 runs the older nemo:26.06.01 stack, where checkpointing has not been validated.
if { [[ $ENABLE_CHECKPOINT == true ]] || [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; } \
    && [[ $GPU_TYPE != gb300 && $GPU_TYPE != gb200 && $GPU_TYPE != b300 && $GPU_TYPE != b200 ]]; then
    echo "Error: Checkpoint save and load are supported only on GB300, GB200, B300 and B200 (GPU_TYPE=$GPU_TYPE)." >&2
    exit 1
fi

# On GB200, the 70B FP8 fsdp_dtensor checkpoint save runs out of host memory below 128 GPUs.
if { [[ $ENABLE_CHECKPOINT == true ]] || [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; } \
    && [[ $GPU_TYPE == gb200 && $MODEL_SIZE == 70b && $DTYPE == fp8 ]] && [[ $JOB_TOTAL_GPUS -lt 128 ]]; then
    echo "Error: Checkpoint save/load for Llama 3.1 70B FP8 on GB200 requires at least 128 GPUs (the fsdp_dtensor save runs out of host memory at 64 GPUs)." >&2
    exit 1
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

# Checkpoint load: loads model weights from LOAD_CHECKPOINT_PATH and trains for one step
if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
    if [[ $LOAD_CHECKPOINT_PATH != /* ]] || [[ ! -d $LOAD_CHECKPOINT_PATH ]]; then
        echo "Error: LOAD_CHECKPOINT_PATH must be an absolute path to an existing directory: $LOAD_CHECKPOINT_PATH" >&2
        exit 1
    fi
    MAX_STEPS=1
    # finetune resets the iteration and consumed samples to 0 so max_steps=1 trains one step.
    # It loads model weights only, not optimizer, LR scheduler or RNG state.
    CONFIG_OVERRIDES+=" --load_dir=$LOAD_CHECKPOINT_PATH checkpoint.finetune=true "
    # The job only mounts its own experiment directory, so the checkpoint needs its own mount.
    if [[ ",${CONTAINER_MOUNTS}," != *",${LOAD_CHECKPOINT_PATH},"* ]]; then
        CONTAINER_MOUNTS+=",${LOAD_CHECKPOINT_PATH}"
    fi
fi

# Megatron-FSDP recipes only accept the fsdp_dtensor checkpoint format for save/load.
# Override with CKPT_FORMAT if the recipe set changes.
if [[ $ENABLE_CHECKPOINT == true ]] || [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
    if [[ -z ${CKPT_FORMAT:-} ]] && [[ $MODEL_SIZE == 70b ]] && [[ $DTYPE == fp8 && $FP8_RECIPE == cs ]]; then
        CKPT_FORMAT=fsdp_dtensor
    fi
    if [[ -n ${CKPT_FORMAT:-} ]]; then
        CONFIG_OVERRIDES+=" checkpoint.ckpt_format=$CKPT_FORMAT "
    fi
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
    PROFILE_RANKS=${PROFILE_RANKS:-$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)) | sed 's/,$//')}
    CONFIG_OVERRIDES+=" --enable_nsys --profiling_start_step=$PROFILE_START_STEP --profiling_stop_step=$PROFILE_STOP_STEP --profiling_ranks $PROFILE_RANKS "
    CONFIG_OVERRIDES+=" --nsys_trace=cuda,nvtx "
    if [[ $GPU_TYPE == "h100" ]]; then
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
    # H100 supports only FP8 CS (mx not allowed)
    if [[ $GPU_TYPE == "h100" ]] && [[ $FP8_RECIPE == "mx" ]]; then
        echo "Error: H100 supports only FP8 CS; FP8 MX is not allowed for this GPU type."
        exit 1
    fi
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

if [[ $CONFIG_VARIANT == "v1" ]]; then
    CONFIG_OVERRIDES+=" --config_variant v1 "
else
    CONFIG_OVERRIDES+=" --config_variant v2 "
fi

if [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "gb300" ]]; then
    GPUS_PER_NODE=4
else
    GPUS_PER_NODE=8
fi

# Build executor-specific arguments: nvcre flags for kubernetes, slurm flags otherwise.
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

#run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python3 scripts/performance/setup_experiment.py \
    --model_family_name $MODEL_FAMILY_NAME \
    --model_recipe_name $MODEL_RECIPE_NAME \
    --gpu $GPU_TYPE \
    --container_image $IMAGE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --offline \
    $CONFIG_OVERRIDES \
    --log_dir $NEMORUN_HOME \
    --packager none \
    "${EXECUTOR_ARGS[@]}" \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}

popd
