#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

# Parameters
#SBATCH --exclusive
#SBATCH --job-name="cpu_overhead:microbenchmark"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

if [[ $SLURM_JOB_NUM_NODES -ne 1 ]]; then
    echo "This benchmark only supports a single node -- ${SLURM_JOB_NUM_NODES} nodes requested."
    exit 1
fi

export WORKLOAD_TYPE=microbenchmark
export WORKLOAD=cpu_overhead
export FW_VERSION=1.1.0rc5

export LLMB_INSTALL=${LLMB_INSTALL:?Please set LLMB_INSTALL to the path of the installation directory for all workloads}
export LLMB_WORKLOAD=$LLMB_INSTALL/workloads/${WORKLOAD_TYPE}_${WORKLOAD}
export LLMB_REPO=${LLMB_REPO:-$LLMB_INSTALL/llmb_repo}
export LLMB_EXPERIMENT_DIR=${LLMB_EXPERIMENT_DIR:?LLMB_EXPERIMENT_DIR must be set by the configured_sbatch launcher}
export IMAGE=${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/tensorrt-llm+release+${FW_VERSION}.sqsh}

export MODEL_PATH=$LLMB_WORKLOAD/gpt-oss-120b
export MOUNT_DIR=$LLMB_WORKLOAD,$LLMB_REPO:$LLMB_REPO:ro
export TRT_DIR=$LLMB_WORKLOAD/TensorRT-LLM
export RESULT_DIR=$LLMB_EXPERIMENT_DIR
export CPU_OVERHEAD_SCRIPT=${CPU_OVERHEAD_SCRIPT:-$LLMB_REPO/microbenchmarks/cpu_overhead/pytorch_kernel_launch_latency.py}
export USE_CASES=${USE_CASES:-"kernel_launch tokenization"}
export NUM_PROMPTS=${NUM_PROMPTS:-100000}

if [ ! -f "$CPU_OVERHEAD_SCRIPT" ]; then
    echo "CPU overhead benchmark script not found: $CPU_OVERHEAD_SCRIPT"
    exit 1
fi

GPU_TYPE=${GPU_TYPE:?GPU_TYPE is a required variable.}
GPU_TYPE=${GPU_TYPE,,}

# Decide binding option based on GPU_TYPE
if [ "$GPU_TYPE" == "b300" ]; then
    CPU_BIND="--physcpubind=0"
else
    CPU_BIND="--cpunodebind=0"
fi

# Loop over each use case
for value in $USE_CASES; do
    LOG_NAME=${value}_overhead

    if [ ${value} == "kernel_launch" ]; then
        CMD="cd \"$RESULT_DIR\" && numactl ${CPU_BIND} --membind=0 python \"$CPU_OVERHEAD_SCRIPT\" \
	    --start_size 4 --end_size 512 --iters 1000000"
    elif [ ${value} == "tokenization" ]; then
        export DATASET_FILE=$RESULT_DIR/dataset_1000_1000_${NUM_PROMPTS}.txt
        CMD="START_TIME=\$(date +%s); \
            python $TRT_DIR/benchmarks/cpp/prepare_dataset.py \
            --stdout --tokenizer $MODEL_PATH \
            token-norm-dist \
            --input-mean 1000 --output-mean 1000 \
            --input-stdev 0 --output-stdev 0 \
            --num-requests $NUM_PROMPTS > $DATASET_FILE; \
            END_TIME=\$(date +%s); \
            ELAPSED=\$((END_TIME - START_TIME)); \
            echo \"Tokenization time: \${ELAPSED}s\""
    fi

    export SLURM_MPI_TYPE="pmix"
    export SRUN_OUTPUT=${RESULT_DIR}/${LOG_NAME}_%N_%j.out
    export SRUN_ERROR=${RESULT_DIR}/${LOG_NAME}_%N_%j.err

    # Run one task with all CPUs in the node allocation.
    srun --ntasks=1 --cpus-per-task="$SLURM_CPUS_ON_NODE" --container-image "$IMAGE" \
        --container-mounts "$MOUNT_DIR" \
        --container-writable \
        --no-container-mount-home bash -c "$CMD"

    echo "Results of Benchmark: $SRUN_OUTPUT"
    echo "Error Log of Benchmark: $SRUN_ERROR"

done
