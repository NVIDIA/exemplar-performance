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
export WORKLOAD=gpt_oss_mlperf
export MODEL_FAMILY=gpt_oss
export MODEL_RECIPE=gpt_oss_20b
export FW_VERSION=26.06.00
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

JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

if [[ $JOB_TOTAL_GPUS == 8 ]]; then
    DEFAULT_CONFIG_VARIANT="v1"
elif [[ $JOB_TOTAL_GPUS == 64 ]] || [[ $JOB_TOTAL_GPUS == 72 ]]; then
    DEFAULT_CONFIG_VARIANT="v2"
elif [[ $JOB_TOTAL_GPUS == 512 ]]; then
    DEFAULT_CONFIG_VARIANT="v3"
else
    echo "Error: Invalid JOB_TOTAL_GPUS '$JOB_TOTAL_GPUS'. Supported values: 8, 64, 72, 512" >&2
    exit 1
fi
CONFIG_VARIANT=${CONFIG_VARIANT:-$DEFAULT_CONFIG_VARIANT}

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
CONTAINER_MOUNTS+=",$LLMB_INSTALL/llmb_repo/gpt-oss/pretrain/mlperf/custom_communicator_cta.yaml:/workspace/custom_communicator_cta.yaml"
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

# Custom environment variables
NCCL_UB_DP=True
NVTE_CUTEDSL_FUSED_GROUPED_MLP=0
PYTORCH_ALLOC_CONF="expandable_segments:True"
USE_MNNVL=1
NVTE_NVFP4_DISABLE_RHT=0

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
if [[ -n ${EP-} ]]; then
    CONFIG_OVERRIDES+="-ep $EP "
fi
if [[ -n ${ET-} ]]; then
    CONFIG_OVERRIDES+="-et $ET "
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
    DTYPE="fp8_mx"
    NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
    PYTORCH_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
    if [[ ($GPU_TYPE == "gb300" || $GPU_TYPE == "gb200") && $JOB_TOTAL_GPUS -ge 72 ]]; then
        NCCL_UB_DP=False
    fi
elif [[ $DTYPE == "nvfp4" ]]; then
    NVTE_NVFP4_DISABLE_RHT=1
fi

if [[ $GPU_TYPE == "b300" || $GPU_TYPE == "b200" ]]; then
    USE_MNNVL=0
fi

CONFIG_OVERRIDES+=" --compute_dtype $DTYPE "

CONFIG_OVERRIDES+=" --custom_env_vars NCCL_UB_DP=$NCCL_UB_DP,\
NVTE_CUTEDSL_FUSED_GROUPED_MLP=$NVTE_CUTEDSL_FUSED_GROUPED_MLP,\
USE_MNNVL=$USE_MNNVL,\
NCCL_MIN_NCHANNELS=4,\
NCCL_P2P_NET_CHUNKSIZE=2097152,\
NCCL_CFG_PATH=/workspace/custom_communicator_cta.yaml,\
NCCL_SHARP_GROUP_SIZE_THRESH=2,\
NCCL_GRAPH_REGISTER=0,\
NCCL_MIN_CTAS=16,\
NCCL_MAX_CTAS=32,\
NVTE_NORM_FWD_USE_CUDNN=1,\
NVTE_NORM_BWD_USE_CUDNN=1,\
NVTE_NVFP4_DISABLE_RHT=$NVTE_NVFP4_DISABLE_RHT,\
NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0,\
NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0,\
NVTE_DPA_FP8_RECIPE=,\
NVTE_DPA_FP8DS_REDUCE_AMAX=0,\
TE_UB_ATOMIC_GEMM_RS=0,\
MC_TP_OVERLAP_AG=True,\
MC_TP_OVERLAP_RS=True,\
TORCH_CPP_LOG_LEVEL=ERROR,\
TORCH_NCCL_AVOID_RECORD_STREAMS=1,\
TORCH_NCCL_HIGH_PRIORITY=1 "
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

# MLPERF_DATA=c4: feed MLPerf preprocessed C4 blend via rp2 mode. See README.md
# for download instructions. C4_DATA_PREFIX must be set (path without .bin/.idx).
if [[ $MLPERF_DATA == "c4" ]]; then
    : "${C4_DATA_PREFIX:?C4_DATA_PREFIX must be set when MLPERF_DATA=c4 (see README.md 'Prepare Dataset')}"
    C4_INDEX_MAPPING_DIR=${C4_INDEX_MAPPING_DIR:-$LLMB_INSTALL/.cache/c4_index_mapping}
    mkdir -p $C4_INDEX_MAPPING_DIR
    CONFIG_OVERRIDES+=" --data rp2 --dataset_paths $C4_DATA_PREFIX --index_mapping_dir $C4_INDEX_MAPPING_DIR "
    if [[ -n ${CONTAINER_MOUNTS} ]]; then
        CONTAINER_MOUNTS+=","
    fi
    CONTAINER_MOUNTS+="$(dirname "$(dirname "$C4_DATA_PREFIX")"),$C4_INDEX_MAPPING_DIR"
fi

if [[ -n ${CONTAINER_MOUNTS} ]]; then
    CONFIG_OVERRIDES+=" --custom_mounts=$CONTAINER_MOUNTS"
fi

# run command
pushd $LLMB_WORKLOAD/Megatron-Bridge

python scripts/performance/setup_experiment.py \
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
    --account $SBATCH_ACCOUNT \
    --partition $SBATCH_PARTITION \
    --log_dir $NEMORUN_HOME \
    --time_limit $TIME_LIMIT \
    $SLURM_ARGS \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}
popd
