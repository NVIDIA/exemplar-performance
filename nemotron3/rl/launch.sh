#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=01:00:00

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

# ---- LLMB environment variables ----
export WORKLOAD_TYPE=rl
export MODEL_NAME=nemotron_3
export GPU_TYPE=${GPU_TYPE:-b200}
export GPU_TYPE=${GPU_TYPE,,}
export JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:-72}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export LLMB_REPO=$PWD
HF_TOKEN=${HF_TOKEN:?HF_TOKEN must be set in environment variables}

if [[ $GPU_TYPE == "gb200" ]]; then
    export GPUS_PER_NODE=4
elif [[ $GPU_TYPE == "b200" ]]; then
    export GPUS_PER_NODE=8
else
    echo "Error: Invalid GPU_TYPE '$GPU_TYPE'. Supported values: b200, gb200" >&2
    exit 1
fi
NUM_NODES=$((JOB_TOTAL_GPUS / GPUS_PER_NODE))

# ---- RL experiment environment variables ----
EXP_NAME=stage2.1-swe1
MAX_STEPS=${MAX_STEPS:-50}
suffix="${EXP_NAME##*-}"
CONTAINER=${LLMB_INSTALL}/images/nvidia+nemo-rl+v0.5.0.nemotron_3_super.sqsh
export CONTAINER
TRAIN_PATH=${LLMB_WORKLOAD}/Nemotron-RL-Super-Training-Blends/data/${suffix}/train-split.jsonl
VAL_PATH=${LLMB_WORKLOAD}/Nemotron-RL-Super-Training-Blends/data/${suffix}/val-split.jsonl
MODEL_PATH=${LLMB_WORKLOAD}/NVIDIA-Nemotron-3-Super-120B-A12B-BF16

if [[ -n ${WANDB_API_KEY:-} ]]; then
    export WANDB_API_KEY
    WANDB_ARGS+=" logger.wandb_enabled=True"
    WANDB_ARGS+=" logger.wandb.project=nemo-rl-nemotron3"
    WANDB_ARGS+=" logger.wandb.name=${EXP_NAME}"
else
    WANDB_ARGS+=" logger.wandb_enabled=False"
fi

export HF_HOME=$LLMB_INSTALL/.cache/huggingface
PERSISTENT_CACHE=$LLMB_INSTALL/persistent_cache
VLLM_CACHE_DIR=$PERSISTENT_CACHE/vllm_compile_cache
FLASHINFER_CUBIN_CACHE=$PERSISTENT_CACHE/flashinfer_cubins
FLASHINFER_WS_BASE=$PERSISTENT_CACHE/flashinfer_workspace
PREBAKED_VENVS=/opt/gym_venvs

mkdir -p "$HF_HOME" "$PERSISTENT_CACHE" "$VLLM_CACHE_DIR" "$FLASHINFER_CUBIN_CACHE" "$FLASHINFER_WS_BASE"

CODE_DIR=${LLMB_WORKLOAD}/NeMo-RL
LOG_DIR=${LLMB_EXPERIMENT_DIR}/logs/${EXP_NAME}
SNAPSHOT_DIR=${LLMB_EXPERIMENT_DIR}/code_snapshots/${EXP_NAME}
export BASE_LOG_DIR=${SNAPSHOT_DIR}

mkdir -p "$SNAPSHOT_DIR"
rsync -a --files-from=<(
    cd "${CODE_DIR}" && git ls-files --recurse-submodules --cached --full-name
) "${CODE_DIR}/" "$SNAPSHOT_DIR/"

SNAPSHOT_DIR=$(realpath "$SNAPSHOT_DIR")

pushd "${SNAPSHOT_DIR}"

CONFIG_PATH=${SNAPSHOT_DIR}/examples/configs/super/stage2_${suffix}.yaml

export VLLM_PRECOMPILED_WHEEL_LOCATION="https://github.com/vllm-project/vllm/releases/download/v0.13.0/vllm-0.13.0-cp38-abi3-manylinux_2_31_x86_64.whl"
export RAY_DEDUP_LOGS=1

# ---- Config overrides ----
if [[ $JOB_TOTAL_GPUS -eq 72 ]]; then
    NUM_PROMPTS_PER_STEP=4
    NUM_GENERATIONS_PER_PROMPT=4
    TRAIN_GLOBAL_BATCH_SIZE=$((NUM_PROMPTS_PER_STEP * NUM_GENERATIONS_PER_PROMPT))
    GENERATION_GPUS=8
    GENERATION_NODES=$((GENERATION_GPUS / GPUS_PER_NODE))
else
    echo "Error: Unsupported number of GPUs: ${JOB_TOTAL_GPUS}" >&2
    exit 1
fi

export OMP_NUM_THREADS=16

# ---- Container mounts ----
MOUNTS=${LLMB_INSTALL}:${LLMB_INSTALL},${LLMB_REPO}:${LLMB_REPO}
MOUNTS+=,${CODE_DIR}/3rdparty/Megatron-LM-workspace/Megatron-LM:/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM
MOUNTS+=,${LLMB_INSTALL}/workloads/rl_nemotron_3/NeMo-RL/nemo_rl/algorithms/grpo.py:/opt/nemo-rl/nemo_rl/algorithms/grpo.py:ro
MOUNTS+=,${LLMB_INSTALL}/workloads/rl_nemotron_3/NeMo-RL/nemo_rl/models/generation/vllm/vllm_worker_async.py:/opt/nemo-rl/nemo_rl/models/generation/vllm/vllm_worker_async.py:ro
export MOUNTS

# ---- Build the run command ----
export COMMAND="date ; \
    NRL_WG_USE_RAY_REF=1 \
    HF_HOME=${HF_HOME} \
    VLLM_CACHE_ROOT=${VLLM_CACHE_DIR} \
    DG_JIT_CACHE_DIR=${VLLM_CACHE_DIR}/deep_gemm \
    VLLM_DEEP_GEMM_WARMUP=skip \
    FLASHINFER_CUBIN_DIR=${FLASHINFER_CUBIN_CACHE} \
    FLASHINFER_WORKSPACE_BASE=${FLASHINFER_WS_BASE} \
    NEMO_GYM_VENV_DIR=${PREBAKED_VENVS} \
    NRL_VLLM_USE_V1=1 \
    NRL_IGNORE_VERSION_MISMATCH=1 \
    VLLM_ATTENTION_BACKEND=FLASH_ATTN \
    NRL_FORCE_REBUILD_VENVS=false \
    RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 \
    UV_HTTP_TIMEOUT=10 \
    VLLM_USE_PRECOMPILED=1 \
    VLLM_PRECOMPILED_WHEEL_LOCATION=${VLLM_PRECOMPILED_WHEEL_LOCATION} \
    python ${CODE_DIR}/examples/nemo_gym/run_grpo_nemo_gym.py \
    --config ${CONFIG_PATH} \
    policy.model_name=${MODEL_PATH} \
    checkpointing.enabled=false \
    logger.log_dir=${LOG_DIR} \
    logger.gpu_monitoring.collection_interval=100 \
    logger.gpu_monitoring.flush_interval=100 \
    data.train.data_path=${TRAIN_PATH} \
    data.validation.data_path=${VAL_PATH} \
    grpo.num_prompts_per_step=${NUM_PROMPTS_PER_STEP} \
    grpo.num_generations_per_prompt=${NUM_GENERATIONS_PER_PROMPT} \
    grpo.max_num_steps=${MAX_STEPS} \
    grpo.val_period=100 \
    policy.train_global_batch_size=${TRAIN_GLOBAL_BATCH_SIZE} \
    policy.generation.colocated.resources.gpus_per_node=${GPUS_PER_NODE} \
    policy.generation.colocated.resources.num_nodes=${GENERATION_NODES} \
    policy.generation.vllm_cfg.http_server_serving_chat_kwargs.reasoning_parser_plugin=${CODE_DIR}/nemo_rl/utils/nano_v3_reasoning_parser.py \
    cluster.gpus_per_node=${GPUS_PER_NODE} \
    cluster.num_nodes=${NUM_NODES} \
    $WANDB_ARGS \
    $* \
    2>&1 | tee ${BASE_LOG_DIR}/run.log"

bash ray.sub
