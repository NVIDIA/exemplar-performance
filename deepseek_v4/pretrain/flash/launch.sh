#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=deepseek-v4
export FW_VERSION=26.08.01
export OPENBLAS_NUM_THREADS=1

: "${LLMB_INSTALL:?LLMB_INSTALL is a required variable.}"

export LLMB_WORKLOAD=${LLMB_WORKLOAD:-$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}}
export NEMORUN_HOME=${NEMORUN_HOME:-$LLMB_WORKLOAD}

MBRIDGE=${MBRIDGE:-$LLMB_WORKLOAD/Megatron-Bridge}
MCORE=${MCORE:-$LLMB_WORKLOAD/Megatron-LM}

if [[ $# -ne 0 ]]; then
    echo "Usage: $0" >&2
    exit 1
fi

: "${GPU_TYPE:?GPU_TYPE is a required variable.}"
: "${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}"
CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}
if [[ $CLUSTER_TYPE == "slurm" ]]; then
    : "${SBATCH_ACCOUNT:?SBATCH_ACCOUNT is a required variable.}"
    : "${SBATCH_PARTITION:?SBATCH_PARTITION is a required variable.}"
fi

GPU_TYPE=${GPU_TYPE,,}
DTYPE=${DTYPE:-fp8}
DTYPE=${DTYPE,,}
FP8_RECIPE=${FP8_RECIPE:-mx}
FP8_RECIPE=${FP8_RECIPE,,}

if [[ $GPU_TYPE != "gb300" && $GPU_TYPE != "gb200" && $GPU_TYPE != "b300" && $GPU_TYPE != "b200" ]]; then
    echo "ERROR: DeepSeek-V4-Flash supports GB300, GB200, B300, and B200 only." >&2
    exit 1
fi
if [[ $DTYPE != "fp8" || $FP8_RECIPE != "mx" ]]; then
    echo "ERROR: DeepSeek-V4-Flash requires DTYPE=fp8 and FP8_RECIPE=mx." >&2
    exit 1
fi
if [[ $JOB_TOTAL_GPUS -ne 128 && $JOB_TOTAL_GPUS -ne 256 && $JOB_TOTAL_GPUS -ne 512 ]]; then
    echo "ERROR: DeepSeek-V4-Flash is validated at 128, 256, or 512 GPUs." >&2
    exit 1
fi

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}
export HF_HOME=${HF_HOME:-$LLMB_INSTALL/.cache/huggingface}
export HF_DATASETS_CACHE=${HF_DATASETS_CACHE:-$HF_HOME/datasets}
mkdir -p "$HF_HOME" "$HF_DATASETS_CACHE" "$NEMORUN_HOME"

if [[ ! -d $MBRIDGE/src/megatron ]]; then
    echo "ERROR: Expected Megatron-Bridge checkout at $MBRIDGE." >&2
    exit 1
fi
if [[ ! -d $MCORE/megatron/core ]]; then
    echo "ERROR: Expected Megatron-LM checkout at $MCORE." >&2
    exit 1
fi
if [[ $GPU_TYPE == "gb300" || $GPU_TYPE == "gb200" ]]; then
    GPUS_PER_NODE=4
else
    GPUS_PER_NODE=8
fi
MAX_STEPS=${MAX_STEPS:-50}
TIME_LIMIT=${TIME_LIMIT:-"01:10:00"}
COMPUTE_TYPE=fp8_mx

PROFILE_ENABLED=${ENABLE_PROFILE:-false}
PROFILE_ENABLED=${PROFILE_ENABLED,,}
PYTORCH_PROFILE_ENABLED=${ENABLE_PYTORCH_PROFILE:-false}
PYTORCH_PROFILE_ENABLED=${PYTORCH_PROFILE_ENABLED,,}
PROFILE_START_STEP=${PROFILE_START_STEP:-45}
PROFILE_STOP_STEP=${PROFILE_STOP_STEP:-50}
GPU_METRICS_ENABLED=${ENABLE_GPU_METRICS:-false}
GPU_METRICS_ENABLED=${GPU_METRICS_ENABLED,,}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING:-false}
ENABLE_PCT_BINDING=${ENABLE_PCT_BINDING,,}

if [[ $PROFILE_ENABLED == "true" ]] && [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    echo "ERROR: ENABLE_PROFILE and ENABLE_PYTORCH_PROFILE are mutually exclusive." >&2
    exit 1
fi

PROFILE_ARGS=()
if [[ $PROFILE_ENABLED == "true" ]]; then
    PROFILE_RANKS=${PROFILE_RANKS:-$(seq -s, 0 $((JOB_TOTAL_GPUS - 1)) | sed 's/,$//')}
    PROFILE_ARGS=(
        --enable_nsys
        --profiling_start_step "$PROFILE_START_STEP"
        --profiling_stop_step "$PROFILE_STOP_STEP"
        --profiling_ranks "$PROFILE_RANKS"
        --nsys_trace cuda
        --nsys_extra_args=--nvtx-domain-include=NCCL
    )
    if [[ $GPU_METRICS_ENABLED == "true" ]]; then
        PROFILE_ARGS+=(--profiling_gpu_metrics)
    fi
fi

if [[ $PYTORCH_PROFILE_ENABLED == "true" ]]; then
    PROFILE_ARGS+=(--pytorch_profiler true)
fi

CONTAINER_MOUNTS="$HF_HOME,$HF_DATASETS_CACHE,$MCORE:/opt/Megatron-Bridge/3rdparty/Megatron-LM"
if [[ -n ${RUN_CONF_MOUNTS:-} ]]; then
    CONTAINER_MOUNTS+=",${RUN_CONF_MOUNTS}"
fi

SLURM_ARGS=()
if [[ -n ${ADDITIONAL_SLURM_PARAMS:-} ]]; then
    SLURM_ARGS=(--additional_slurm_params "$ADDITIONAL_SLURM_PARAMS")
fi

EXECUTOR_ARGS=()
# On kubernetes the pod has its own HF access (token secret); the launcher's cache paths mean nothing there.
HF_ENV_ARGS=()
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
    # The pinned Megatron-Bridge also needs the pinned Megatron-LM. A bind mount
    # (the Slurm path) is not available in the pod, so ship the checkout through the workdir PVC; it
    # is merged into the directory the job runs from and put first on PYTHONPATH below.
    : "${NVCRE_WORKDIR_PVC:?NVCRE_WORKDIR_PVC is required for DeepSeek-V4-Flash on kubernetes (it ships the pinned Megatron-LM)}"
    EXECUTOR_ARGS+=(--nvcre_workdir_local_path "$MCORE")
    POD_PYTHONPATH=".:/opt/Megatron-Bridge/src"
elif [[ $CLUSTER_TYPE == "slurm" ]]; then
    POD_PYTHONPATH="/opt/Megatron-Bridge/3rdparty/Megatron-LM:/opt/Megatron-Bridge/src"
    # Slurm reads the HF cache that was downloaded at install time; offline.
    HF_ENV_ARGS=(
        -E HF_HOME="$HF_HOME"
        -E HF_DATASETS_CACHE="$HF_DATASETS_CACHE"
        -E TRANSFORMERS_OFFLINE=1
        -E HF_HUB_OFFLINE=1
    )
    EXECUTOR_ARGS=(--account "$SBATCH_ACCOUNT" --partition "$SBATCH_PARTITION" --time_limit "$TIME_LIMIT" "${SLURM_ARGS[@]}")
else
    echo "Unsupported CLUSTER_TYPE=${CLUSTER_TYPE}. Aborting" >&2
    exit 1
fi

pushd "$MBRIDGE" > /dev/null

PYTHONPATH="$MCORE:$MBRIDGE/src:$MBRIDGE/scripts/performance:${PYTHONPATH:-}" \
    python3 -m scripts.performance.setup_experiment \
    --container_image "$IMAGE" \
    --compute_dtype "$COMPUTE_TYPE" \
    --gpu "$GPU_TYPE" \
    --num_gpus "$JOB_TOTAL_GPUS" \
    --gpus_per_node "$GPUS_PER_NODE" \
    --offline \
    --model_family_name deepseek \
    --model_recipe_name deepseek_v4_flash \
    --task pretrain \
    --custom_mounts "$CONTAINER_MOUNTS" \
    --log_dir "$NEMORUN_HOME" \
    --max_steps "$MAX_STEPS" \
    --packager none \
    --enable_pct_binding "$ENABLE_PCT_BINDING" \
    "${PROFILE_ARGS[@]}" \
    -E PYTHONPATH="$POD_PYTHONPATH" \
    "${HF_ENV_ARGS[@]}" \
    "${EXECUTOR_ARGS[@]}" \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}

popd > /dev/null
