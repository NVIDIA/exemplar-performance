#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=llama3.1
export FW_VERSION=26.06.01

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

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
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

# Time limit: 30 min for 8B/70B, 2 hr for 405B (override with TIME_LIMIT env)
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

if [[ $CLUSTER_TYPE != "slurm" ]]; then
    echo "Only SLURM is supported for this workload"
    exit 1
fi

# Set model family and recipe names based on model size
MODEL_FAMILY_NAME="llama"
if [[ $MODEL_SIZE == 405b ]]; then
    MODEL_RECIPE_NAME="llama31_405b"
elif [[ $MODEL_SIZE == 70b ]]; then
    MODEL_RECIPE_NAME="llama3_70b"
elif [[ $MODEL_SIZE == 8b ]]; then
    MODEL_RECIPE_NAME="llama3_8b"
else
    echo "Error: Unsupported MODEL_SIZE: $MODEL_SIZE"
    exit 1
fi

# Checkpoint configuration
if [[ $ENABLE_CHECKPOINT == true ]]; then
    # Disallow checkpointing for 70B and 405B (known NCCL error during checkpoint save).
    if [[ $MODEL_SIZE == 70b ]] || [[ $MODEL_SIZE == 405b ]]; then
        echo "Error: Checkpointing is not supported for 70B or 405B due to a known NCCL error during checkpoint save." >&2
        exit 1
    fi
    # Set checkpoint directory if specified (defaults to experiment dir)
    if [[ -n ${CHECKPOINT_DIR-} ]]; then
        CONFIG_OVERRIDES+=" --save_dir=$CHECKPOINT_DIR "
    fi

    # Set checkpoint interval (defaults to MAX_STEPS if not specified)
    CONFIG_OVERRIDES+=" --save_interval=$CHECKPOINT_INTERVAL "
fi

# Checkpoint load: only supported for 8B (same restriction as checkpoint save)
if [[ -n ${LOAD_CHECKPOINT_PATH-} ]]; then
    if [[ $MODEL_SIZE != 8b ]]; then
        echo "Error: Checkpoint load is only supported for 8B. Current MODEL_SIZE=$MODEL_SIZE." >&2
        exit 1
    fi
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
    PROFILE_RANKS=${PROFILE_RANKS:-$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)))}
    CONFIG_OVERRIDES+=" --enable_nsys --profiling_start_step=$PROFILE_START_STEP --profiling_stop_step=$PROFILE_STOP_STEP --profiling_ranks $PROFILE_RANKS "
    CONFIG_OVERRIDES+=" --nsys_trace=cuda,nvtx "
    CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL "
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

#run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python scripts/performance/setup_experiment.py \
    --model_family_name $MODEL_FAMILY_NAME \
    --model_recipe_name $MODEL_RECIPE_NAME \
    --gpu $GPU_TYPE \
    --container_image $IMAGE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --offline \
    $CONFIG_OVERRIDES \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    --packager none \
    $SLURM_ARGS \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}

popd
