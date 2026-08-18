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

set -eu -o pipefail

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export WORKLOAD_TYPE=pretrain
export WORKLOAD=llama3.1_mlperf
export MODEL_FAMILY=llama
export MODEL_RECIPE=llama31_8b
export FW_VERSION=26.06.01
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${WORKLOAD}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
DTYPE=${DTYPE:-nvfp4}
DTYPE=${DTYPE,,}
GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
ENABLED_GPU_METRICS=${ENABLE_GPU_METRICS:-false}
ENABLED_GPU_METRICS=${ENABLED_GPU_METRICS,,}
ENABLE_VBOOST=${ENABLE_VBOOST:-false}
ENABLE_VBOOST=${ENABLE_VBOOST,,}
PROFILE_START_STEP=${PROFILE_START_STEP:-45}
PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}
MLPERF_DATA=${MLPERF_DATA:-mock}
MLPERF_DATA=${MLPERF_DATA,,}
CONFIG_OVERRIDES=""

JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

if [[ $JOB_TOTAL_GPUS == 8 ]]; then
    DEFAULT_CONFIG_VARIANT="v1"
elif [[ $JOB_TOTAL_GPUS == 72 ]]; then
    DEFAULT_CONFIG_VARIANT="v2"
else
    echo "Error: Invalid JOB_TOTAL_GPUS '$JOB_TOTAL_GPUS'. Supported values: 8, 72" >&2
    exit 1
fi

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
CONTAINER_MOUNTS+=",$LLMB_INSTALL/llmb_repo/llama3.1/pretrain/mlperf/custom_communicator_cta.yaml:/workspace/custom_communicator_cta.yaml"

# Mount Mbridge llama recipe not in container yet
CONTAINER_MOUNTS+=",$LLMB_WORKLOAD/Megatron-Bridge/src/megatron/bridge/recipes/llama:/opt/Megatron-Bridge/src/megatron/bridge/recipes/llama"

# Mount Mbridge train_utils.py not in container yet
CONTAINER_MOUNTS+=",$LLMB_WORKLOAD/Megatron-Bridge/src/megatron/bridge/training/utils/train_utils.py:/opt/Megatron-Bridge/src/megatron/bridge/training/utils/train_utils.py"

if [[ -n ${RUN_CONF_MOUNTS:-""} ]]; then
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="${RUN_CONF_MOUNTS}"
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts=$CONTAINER_MOUNTS "
fi

TIME_LIMIT=${TIME_LIMIT:-"00:30:00"}
MAX_STEPS=${MAX_STEPS:-50}
CPU_PER_TASK_PINNING=${CPU_PER_TASK_PINNING:-0}

DTYPE_EXTRA_ENVS=""
# Value may contain commas; pass via -E (to_dict splits on commas).
PYTORCH_ALLOC_CONF="expandable_segments:True"

if [[ -n ${TP-} ]]; then
    CONFIG_OVERRIDES+=" -tp $TP "
fi
if [[ -n ${PP-} ]]; then
    CONFIG_OVERRIDES+=" -pp $PP "
fi
if [[ -n ${CP-} ]]; then
    CONFIG_OVERRIDES+=" -cp $CP "
fi
if [[ -n ${VP-} ]]; then
    CONFIG_OVERRIDES+=" -vp $VP "
fi
if [[ -n ${EP-} ]]; then
    CONFIG_OVERRIDES+=" -ep $EP "
fi
if [[ -n ${ET-} ]]; then
    CONFIG_OVERRIDES+=" -et $ET "
fi
if [[ -n ${MBS-} ]]; then
    CONFIG_OVERRIDES+=" -mb $MBS "
fi
if [[ -n ${GBS-} ]]; then
    CONFIG_OVERRIDES+=" -gb $GBS "
fi

if [[ $CLUSTER_TYPE != "slurm" ]]; then
    echo "Only SLURM is supported for this workload"
    exit 1
fi

CONFIG_OVERRIDES+=" --max_steps=$MAX_STEPS "

if [[ $PROFILE_ENABLED == "true" ]] && [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    echo "Error: ENABLE_PROFILE and ENABLE_PYTORCH_PROFILE are mutually exclusive." >&2
    exit 1
fi

if [[ $PROFILE_ENABLED == "true" ]]; then
    CONFIG_OVERRIDES+=" --enable_nsys "
    CONFIG_OVERRIDES+=" --profiling_start_step=$PROFILE_START_STEP "
    CONFIG_OVERRIDES+=" --profiling_stop_step=$PROFILE_STOP_STEP "
    PROFILE_RANKS=$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)))
    CONFIG_OVERRIDES+=" --profiling_ranks=$PROFILE_RANKS"
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
    DTYPE="fp8_cs"
    DTYPE_EXTRA_ENVS+="NCCL_NET_PLUGIN=spcx,"
    DTYPE_EXTRA_ENVS+="NCCL_NVLS_ENABLE=0,"
    DTYPE_EXTRA_ENVS+="USE_TE_OPS=False,"
    DTYPE_EXTRA_ENVS+="UB_SKIPMC=1,"
elif [[ $DTYPE == "nvfp4" ]]; then
    DTYPE_EXTRA_ENVS+="NVTE_DPA_FP8CS_O_in_F16=1,"
    DTYPE_EXTRA_ENVS+="NVTE_DPA_FP8DS_AMAX_ALGO=most_recent,"
    DTYPE_EXTRA_ENVS+="NVTE_DPA_FP8DS_AMAX_HISTLEN=1,"
    DTYPE_EXTRA_ENVS+="NVTE_DPA_FP8_FORMAT=HYBRID,"
    DTYPE_EXTRA_ENVS+="NVTE_DPA_FP8_RECIPE=Float8CurrentScaling,"
    DTYPE_EXTRA_ENVS+="NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0,"
    DTYPE_EXTRA_ENVS+="NVTE_NVFP4_DISABLE_RHT=0,"
    DTYPE_EXTRA_ENVS+="NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0,"
    DTYPE_EXTRA_ENVS+="USE_TE_OPS=True,"
fi

CONFIG_VARIANT=${CONFIG_VARIANT:-$DEFAULT_CONFIG_VARIANT}

CONFIG_OVERRIDES+=" --compute_dtype $DTYPE "

CONFIG_OVERRIDES+=" --custom_env_vars ${DTYPE_EXTRA_ENVS}\
NCCL_CFG_PATH=/workspace/custom_communicator_cta.yaml,\
NCCL_MAX_CTAS=32,\
NCCL_MIN_CTAS=16,\
NCCL_MIN_NCHANNELS=4,\
NCCL_P2P_NET_CHUNKSIZE=2097152,\
NCCL_SHARP_GROUP_SIZE_THRESH=2,\
NCCL_WORK_FIFO_DEPTH=1048576,\
NVTE_BWD_LAYERNORM_SM_MARGIN=16,\
NVTE_DPA_FP8DS_REDUCE_AMAX=0,\
NVTE_FWD_LAYERNORM_SM_MARGIN=16,\
NVTE_NORM_BWD_USE_CUDNN=0,\
NVTE_NORM_FWD_USE_CUDNN=0,\
TE_UB_ATOMIC_GEMM_RS=0,\
CUDA_DEVICE_MAX_CONNECTIONS=1,\
TORCH_NCCL_AVOID_RECORD_STREAMS=1,\
TORCH_NCCL_HIGH_PRIORITY=1,\
MC_TP_OVERLAP_AG=True,\
MC_TP_OVERLAP_RS=True "

# Values containing commas cannot go in --custom_env_vars (to_dict splits on commas).
CONFIG_OVERRIDES+=" -E PYTORCH_CUDA_ALLOC_CONF=$PYTORCH_ALLOC_CONF "

if [[ $ENABLE_VBOOST == true ]]; then
    CONFIG_OVERRIDES+=" --enable_vboost true "
fi

if [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "gb300" ]]; then
    GPUS_PER_NODE=4
else
    GPUS_PER_NODE=8
fi

# run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python scripts/performance/setup_experiment.py \
    --model_family_name $MODEL_FAMILY \
    --model_recipe_name $MODEL_RECIPE \
    --task "pretrain" \
    --gpu $GPU_TYPE \
    --container_image $IMAGE \
    --num_gpus $JOB_TOTAL_GPUS \
    --gpus_per_node $GPUS_PER_NODE \
    --config_variant $CONFIG_VARIANT \
    --enable_vboost $ENABLE_VBOOST \
    --offline \
    $CONFIG_OVERRIDES \
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    $SLURM_ARGS \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}
popd
