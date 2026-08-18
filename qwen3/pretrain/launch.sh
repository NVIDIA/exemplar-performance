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

if [ ${BASH_VERSION:0:1} -lt 4 ] || [ ${BASH_VERSION:0:1} -eq 4 ] && [ ${BASH_VERSION:2:1} -lt 2 ]; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

set -eu -o pipefail

#Required environment variables
: "${LLMB_INSTALL:?Required variable LLMB_INSTALL}"

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=qwen3
export MODEL_SIZE=${MODEL_SIZE:-235b}
export MODEL_SIZE=${MODEL_SIZE,,}

# Map short model size to full model size for --model_size argument
if [ $MODEL_SIZE = "235b" ]; then
    MODEL_SIZE_FULL="235b_a22b"
elif [ $MODEL_SIZE = "30b" ]; then
    MODEL_SIZE_FULL="30b_a3b"
else
    echo "Unsupported MODEL_SIZE: $MODEL_SIZE"
    exit 1
fi

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD

export NEMORUN_HOME=$LLMB_WORKLOAD
export NEMO_HOME=${NEMO_HOME:-$LLMB_WORKLOAD}

export DTYPE=${DTYPE:-bf16}
export DTYPE=${DTYPE,,}
export GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
export GPU_TYPE=${GPU_TYPE,,}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

FW_VERSION=26.06.01

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}
if [ "$MODEL_SIZE" = "235b" ]; then
    export TIME_LIMIT=${TIME_LIMIT:-"02:00:00"}
else
    export TIME_LIMIT=${TIME_LIMIT:-"01:00:00"}
fi
export MAX_STEPS=${MAX_STEPS:-50}
export PROFILE_START_STEP=${PROFILE_START_STEP:-45}
export PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING,,}

# Handle additional SLURM parameters from environment variable
ADDITIONAL_SLURM_PARAMS=${ADDITIONAL_SLURM_PARAMS:-""}

# Add additional SLURM parameters if provided
SLURM_ARGS=""
if [ -n "$ADDITIONAL_SLURM_PARAMS" ]; then
    SLURM_ARGS="--additional_slurm_params ${ADDITIONAL_SLURM_PARAMS}"
fi

# Mount Hugging Face cache for tokenizers
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

# Optional overrides: only set when user provides values.
if [[ -n ${TP:-} ]]; then
    CONFIG_OVERRIDES+=" -tp $TP "
fi
if [[ -n ${PP:-} ]]; then
    CONFIG_OVERRIDES+=" -pp $PP "
fi
if [[ -n ${CP:-} ]]; then
    CONFIG_OVERRIDES+=" -cp $CP "
fi
if [[ -n ${EP:-} ]]; then
    CONFIG_OVERRIDES+=" -ep $EP "
fi
if [[ -n ${ETP:-} ]]; then
    CONFIG_OVERRIDES+=" -et $ETP "
fi
if [[ -n ${GBS:-} ]]; then
    CONFIG_OVERRIDES+=" -gb $GBS "
fi
if [[ -n ${MBS:-} ]]; then
    CONFIG_OVERRIDES+=" -mb $MBS "
fi
if [[ -n ${VP:-} ]]; then
    CONFIG_OVERRIDES+=" -vp $VP "
fi
if [[ -n ${CUDA_GRAPH_SCOPE:-} ]]; then
    CUDA_GRAPH_IMPL="transformer_engine"
    if [[ $CUDA_GRAPH_SCOPE == "full_iteration" ]]; then
        CUDA_GRAPH_IMPL="local"
    elif [[ $CUDA_GRAPH_SCOPE == "none" ]]; then
        CUDA_GRAPH_IMPL="none"
    fi
    CONFIG_OVERRIDES+=" --cuda_graph_impl=$CUDA_GRAPH_IMPL "
    if [[ $CUDA_GRAPH_IMPL != "none" ]]; then
        CONFIG_OVERRIDES+=" --cuda_graph_scope=$CUDA_GRAPH_SCOPE "
    fi
fi

if [[ $PROFILE_ENABLED == "true" ]] && [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    echo "Error: ENABLE_PROFILE and ENABLE_PYTORCH_PROFILE are mutually exclusive." >&2
    exit 1
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    CONFIG_OVERRIDES+=" --nsys_trace=cuda,nvtx "
    CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL "
    CONFIG_OVERRIDES+=" --profiling_ranks=$(seq -s, 0 $((JOB_TOTAL_GPUS - 1))) "
    if [[ $GPU_METRICS_ENABLED == true ]]; then
        CONFIG_OVERRIDES+=" --profiling_gpu_metrics "
    fi
    MAX_STEPS=$PROFILE_STOP_STEP
fi

if [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --pytorch_profiler true "
fi

if [[ $DTYPE == "fp8" ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        export FP8_RECIPE=${FP8_RECIPE:-fp8_cs}
    else
        export FP8_RECIPE=${FP8_RECIPE:-fp8_mx}
    fi
    export FP8_RECIPE=${FP8_RECIPE,,}
    COMPUTE_DTYPE=$FP8_RECIPE
else
    COMPUTE_DTYPE=$DTYPE
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

if [[ $DTYPE == "bf16" ]] && [[ $MODEL_SIZE == "235b" ]] && [[ $GPU_TYPE != "h100" ]]; then
    export NCCL_IB_QPS_PER_CONNECTION=${NCCL_IB_QPS_PER_CONNECTION:-4}
fi

#run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python3 scripts/performance/setup_experiment.py \
    --container_image $IMAGE \
    --offline \
    --compute_dtype $COMPUTE_DTYPE \
    --model_family_name qwen \
    --model_recipe_name qwen3_${MODEL_SIZE_FULL} \
    --gpu $GPU_TYPE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    ${CONFIG_OVERRIDES} \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    --max_steps $MAX_STEPS \
    --packager none \
    $SLURM_ARGS \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}

popd
