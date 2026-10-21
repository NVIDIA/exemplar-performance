#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=01:00:00
# One gb300 node: llmb-run turns --scale 4 into -N 1 --ntasks-per-node=4 plus the partition's
# GPUs-per-node request. 32 CPUs per task (128 of the node's 144) land in the batch step's cpuset:
# the replay pins RUN_CONF_TASKTRACE_CPUS of them (default 4), the vLLM server takes the rest.
#SBATCH --cpus-per-task=32

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL not set (framework should provide).}
export LLMB_EXPERIMENT_DIR=${LLMB_EXPERIMENT_DIR:?LLMB_EXPERIMENT_DIR not set (framework should provide).}

export WORKLOAD_KEY="inference_tasktrace_replay_gpu"
export LLMB_WORKLOAD="$LLMB_INSTALL/workloads/$WORKLOAD_KEY"

RECIPE_DIR=""
for candidate in "${SLURM_SUBMIT_DIR:-}" "$PWD" "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; do
    if [[ -n $candidate && -f $candidate/metadata.yaml && -f $candidate/launch.sh ]]; then
        RECIPE_DIR="$candidate"
        break
    fi
done
[[ -n $RECIPE_DIR ]] || {
    echo "[launch] ERROR: cannot locate the recipe directory (tried SLURM_SUBMIT_DIR, PWD, script path)." >&2
    exit 1
}

COMMON_DIR="$(cd "$RECIPE_DIR/../../common" && pwd)"
RECIPE_PY="$COMMON_DIR/tasktrace_recipe.py"

if [[ -n ${VIRTUAL_ENV:-} ]]; then
    PATH=$(echo ":${PATH}:" | sed "s|:${VIRTUAL_ENV}/bin:|:|g; s|^:||; s|:$||")
    unset VIRTUAL_ENV
fi
PYTHON="${RUN_CONF_PYTHON:-/usr/bin/python3}"
"$PYTHON" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)' 2> /dev/null || {
    echo "[launch] ERROR: $PYTHON is not a working Python >= 3.8 on $(hostname) ($(uname -m)). Set RUN_CONF_PYTHON to the compute node's interpreter." >&2
    exit 1
}
[[ -f $RECIPE_PY ]] || {
    echo "[launch] ERROR: missing $RECIPE_PY (the agentic common/ directory must ship with the recipe)." >&2
    exit 1
}
MODEL_SIZE="${MODEL_SIZE:-nemotron-3-ultra-vllm_miniswe_django-11490}"
TRAJECTORY_ROOT="$(cd "$RECIPE_DIR/../../../.." && pwd)/trajectories"

TRACE_VARS="$("$PYTHON" "$RECIPE_PY" resolve-trace --root "$TRAJECTORY_ROOT" --selector "$MODEL_SIZE")" || exit 1
eval "$TRACE_VARS"

TASKTRACE_BIN="$LLMB_WORKLOAD/bin/tasktrace"
IMAGE="${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/$IMAGE_FILE}"
if [[ $(uname -m) != "$IMAGE_ARCH" ]]; then
    echo "[launch] ERROR: task image is $IMAGE_ARCH but this node is $(uname -m)." >&2
    exit 1
fi
[[ -x $TASKTRACE_BIN ]] || {
    echo "[launch] ERROR: tasktrace binary missing at $TASKTRACE_BIN. Re-run llmb-install for $WORKLOAD_KEY." >&2
    exit 1
}
[[ -s $TRACE ]] || {
    echo "[launch] ERROR: no trace for '$TRACE_ID' at $TRACE." >&2
    exit 1
}
[[ -s $IMAGE ]] || {
    echo "[launch] ERROR: task image missing at $IMAGE. Re-run llmb-install for $WORKLOAD_KEY (it imports container.images)." >&2
    exit 1
}

# shellcheck source=../../common/tasktrace_gpu_lib.sh
source "$COMMON_DIR/tasktrace_gpu_lib.sh"
gpu_recipe_knobs "${TRACE_TIMED:-1}"
llm_server_load_config "${SERVING_CONFIG:-$TRACE_MODEL}"
llm_server_preflight
gpu_recipe_cpusets

OUT_DIR="$LLMB_EXPERIMENT_DIR/profile"
LLM_DIR="$LLMB_EXPERIMENT_DIR/llm-replay"
SERVER_DIR="$LLMB_EXPERIMENT_DIR/server"
mkdir -p "$OUT_DIR" "$LLM_DIR" "$SERVER_DIR"
export TMPDIR="${TMPDIR:-/tmp}/tasktrace-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'llm_server_stop; rm -rf "$TMPDIR"' EXIT

echo "[launch] node: $(hostname)  arch: $(uname -m)  gpus: $LLM_SERVER_GPU_COUNT x ${LLM_SERVER_GPU_NAME:-GPU}"
echo "[launch] tasktrace: $TASKTRACE_BIN ($(tr '\n' ' ' < "$LLMB_WORKLOAD/bin/tasktrace.version"))"
echo "[launch] selection: $MODEL_SIZE -> $TRACE_MODEL / $TRACE_HARNESS / $TRACE_DATASET / $TRACE_ID"
echo "[launch] trace: $TRACE"
echo "[launch] image: $IMAGE"
echo "[launch] server: $SERVING_MODEL_REPO@${SERVING_MODEL_REVISION:-default} as '$SERVING_SERVED_MODEL_NAME'  TP$SERVING_TP  (config $SERVING_NAME)"
echo "[launch] output: $LLMB_EXPERIMENT_DIR"

llm_server_bring_up "$SERVER_DIR" "$TRACE"
gpu_replay "$TRACE" "$IMAGE" "$OUT_DIR" "$LLM_DIR"
llm_server_stop

SUMMARY="$LLMB_EXPERIMENT_DIR/summary.json"
set +e
"$PYTHON" "$RECIPE_PY" summarize-replay \
    --out-dir "$OUT_DIR" \
    --summary "$SUMMARY" \
    --trace-id "$TRACE_ID" \
    --pacing "$PACING" \
    --cpus "$CPUS" \
    --cpuset "$CPUSET" \
    --replay-exit "$REPLAY_EXIT" \
    --workload tasktrace_replay_gpu \
    --llm-report "$LLM_REPORT" \
    --llm-exit "$LLM_EXIT" \
    --server-info "$SERVER_DIR/server.json"
STATUS=$?
set -e
echo "[launch] summary: $SUMMARY"
exit "$STATUS"
