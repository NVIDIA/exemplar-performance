#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

set -eu -o pipefail

export WORKLOAD_TYPE=pretrain
export MODEL_NAME=deepseek-v3
export OPENBLAS_NUM_THREADS=1 # Required for login nodes with tight memory restrictions. Do not remove.

export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${MODEL_NAME}
export NEMORUN_HOME=$LLMB_WORKLOAD
export LLMB_REPO=$PWD

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}
DTYPE=${DTYPE:-bf16}
DTYPE=${DTYPE,,}

if [[ $GPU_TYPE == "h100" ]]; then
    FW_VERSION=26.06.01
else
    FW_VERSION=26.08.01
fi

if [[ $DTYPE == "fp8" ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        FP8_RECIPE="sc"
    else
        FP8_RECIPE="mx"
    fi
    COMPUTE_TYPE=${DTYPE}_${FP8_RECIPE}
else
    COMPUTE_TYPE=${DTYPE}
fi

export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/nvidia+nemo+$FW_VERSION.sqsh}

JOB_TOTAL_GPUS=${JOB_TOTAL_GPUS:?JOB_TOTAL_GPUS is a required variable.}

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
IS_PROXY_WORKLOAD=${PROXY_WORKLOAD:-false}
IS_PROXY_WORKLOAD=${IS_PROXY_WORKLOAD,,}
CPU_PERF_PROXY=${CPU_PERF_PROXY:-false}
CPU_PERF_PROXY=${CPU_PERF_PROXY,,}
DISABLE_CG=${DISABLE_CG:-false}
DISABLE_CG=${DISABLE_CG,,}
MAX_STEPS=${MAX_STEPS:-50}
if [[ $IS_PROXY_WORKLOAD == true ]]; then
    if [[ $GPU_TYPE == "h100" ]]; then
        TIME_LIMIT=${TIME_LIMIT:-"01:45:00"}
    else
        TIME_LIMIT=${TIME_LIMIT:-"01:15:00"}
    fi
elif [[ $GPU_TYPE == "h100" ]]; then
    TIME_LIMIT=${TIME_LIMIT:-"01:30:00"}
else
    TIME_LIMIT=${TIME_LIMIT:-"00:45:00"}
fi

# Handle additional SLURM parameters from environment variable
CLUSTER_TYPE=${CLUSTER_TYPE:-slurm}

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
    CONFIG_OVERRIDES+=" --nsys_trace=cuda "
    if [[ $GPU_TYPE == "h100" || $GPU_TYPE == "b300" ]]; then
        CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL "
    else
        CONFIG_OVERRIDES+=" --nsys_extra_args=--nvtx-domain-include=NCCL,--discard-environment=true "
    fi
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
elif [[ $GPU_TYPE == "b300" ]] || [[ $GPU_TYPE == "b200" ]] || [[ $GPU_TYPE == "h100" ]]; then
    GPUS_PER_NODE=8
else
    echo "Error: Unsupported GPU_TYPE '$GPU_TYPE'. Expected one of: gb300, gb200, b300, b200, h100." >&2
    exit 1
fi

if [[ $GPU_TYPE == "gb300" ]] && [[ $JOB_TOTAL_GPUS -eq 128 ]]; then
    CONFIG_OVERRIDES+=" -pp 4 "
    CONFIG_OVERRIDES+=" -vp 4 "
    CONFIG_OVERRIDES+=" -ep 32 "
    CONFIG_OVERRIDES+=" --recompute_modules=mla_up_proj "
fi

if [[ $GPU_TYPE == "gb200" ]] && [[ $JOB_TOTAL_GPUS -eq 128 ]] && [[ $COMPUTE_TYPE == "fp8_mx" ]]; then
    CONFIG_OVERRIDES+=" -tp 2 "
    CONFIG_OVERRIDES+=" -pp 2 "
    CONFIG_OVERRIDES+=" -vp 8 "
    CONFIG_OVERRIDES+=" -ep 32 "
    CONFIG_OVERRIDES+=" --recompute_modules=mla_up_proj "
fi

if [[ $GPU_TYPE == "b300" ]] && [[ $JOB_TOTAL_GPUS -eq 128 ]]; then
    if [[ $COMPUTE_TYPE == "bf16" ]]; then
        CONFIG_OVERRIDES+=" --cuda_graph_impl none "
    elif [[ $COMPUTE_TYPE == "fp8_mx" ]]; then
        CONFIG_OVERRIDES+=" --recompute_modules=mla_up_proj,core_attn "
    fi
fi

if [[ $GPU_TYPE == "b200" ]] && [[ $JOB_TOTAL_GPUS -eq 128 ]] && [[ $COMPUTE_TYPE == "fp8_mx" ]]; then
    CONFIG_OVERRIDES+=" --cuda_graph_impl none "
    CONFIG_OVERRIDES+=" --recompute_modules=mla_up_proj,core_attn,moe "
fi

if [[ $IS_PROXY_WORKLOAD == true ]]; then
    if [[ $CPU_PERF_PROXY == false ]]; then # Lower resource proxy workloads
        tp=1
        pp=1
        vp=None
        mb=1
        num_layers=31
        if [[ $GPU_TYPE == "gb300" || $GPU_TYPE == "gb200" ]]; then
            if ((JOB_TOTAL_GPUS % 72 == 0)); then
                ep=72
                num_moe_experts=144
            else
                ep=$JOB_TOTAL_GPUS
                num_moe_experts=$((JOB_TOTAL_GPUS * 2))
                if [[ $JOB_TOTAL_GPUS -eq 4 ]]; then
                    # Top-8 routing over 4 groups needs >= 2 experts per group, so 8 experts
                    # only fit 4 router groups, not the default 8.
                    CONFIG_OVERRIDES+=" model.moe_router_num_groups=4 "
                fi
            fi
        elif [[ $GPU_TYPE == "b300" || $GPU_TYPE == "b200" ]]; then
            ep=8
            num_moe_experts=$((JOB_TOTAL_GPUS * 2))
            if [[ $JOB_TOTAL_GPUS -eq 64 ]]; then
                pp=2
                vp=4
                pipeline_model_parallel_layout="Et*4\|\(t*4\|\)*6t*3mL"
                CONFIG_OVERRIDES+=" --pipeline_model_parallel_layout=$pipeline_model_parallel_layout "
            fi
        elif [[ $GPU_TYPE == "h100" ]]; then
            ep=8
            num_layers=8
            if [[ $JOB_TOTAL_GPUS -ge 64 ]]; then
                num_moe_experts=128
            else
                num_moe_experts=$((JOB_TOTAL_GPUS * 2))
            fi
        fi
        if [[ $GPU_TYPE == "gb300" || $GPU_TYPE == "b300" ]]; then
            mb=2
        elif [[ $GPU_TYPE == "h100" ]] && [[ $JOB_TOTAL_GPUS -lt 64 ]]; then
            mb=2
        fi
        CONFIG_OVERRIDES+=" -tp $tp "
        CONFIG_OVERRIDES+=" -pp $pp "
        CONFIG_OVERRIDES+=" -vp $vp "
        CONFIG_OVERRIDES+=" -ep $ep "
        CONFIG_OVERRIDES+=" -mb $mb "
        CONFIG_OVERRIDES+=" --num_layers $num_layers "
        CONFIG_OVERRIDES+=" --num_moe_experts $num_moe_experts "
    elif [[ $CPU_PERF_PROXY == true ]] && [[ $GPU_TYPE != "h100" ]]; then # CPU proxy workloads
        if [[ $JOB_TOTAL_GPUS -eq $GPUS_PER_NODE ]]; then
            CONFIG_OVERRIDES+=" -ep $GPUS_PER_NODE "
        elif [[ $JOB_TOTAL_GPUS -eq 64 ]] && { [[ $GPU_TYPE == "gb200" ]] || [[ $GPU_TYPE == "gb300" ]]; }; then
            CONFIG_OVERRIDES+=" -ep 64 "
        else
            echo "Error: CPU proxy workloads are supported only for a single node (gb300/gb200/b300/b200) or 64 GPUs (gb200/gb300)."
            exit 1
        fi
        CONFIG_OVERRIDES+=" -tp 1 "
        CONFIG_OVERRIDES+=" -pp 1 "
        CONFIG_OVERRIDES+=" -vp None "
        CONFIG_OVERRIDES+=" -mb 1 "
        CONFIG_OVERRIDES+=" --num_layers 31 "
        CONFIG_OVERRIDES+=" --hidden_size 512 "
    fi
fi

if [[ $DISABLE_CG == true ]]; then
    CONFIG_OVERRIDES+=" --cuda_graph_impl none "
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
    --model_family_name deepseek \
    --model_recipe_name deepseek_v3 \
    ${CONFIG_OVERRIDES} \
    --log_dir $NEMORUN_HOME \
    --max_steps $MAX_STEPS \
    --packager none \
    "${EXECUTOR_ARGS[@]}" \
    ${LLMB_MBRIDGE_EXTRA_ARGS:-}

popd
