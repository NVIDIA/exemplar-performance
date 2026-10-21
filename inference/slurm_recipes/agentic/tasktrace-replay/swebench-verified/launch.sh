#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=00:30:00
# The workload uses CPUs only. On a CPU partition a task gets one CPU unless asked, and the
# replay pins RUN_CONF_TASKTRACE_CPUS (default 4) of them; 8 leaves headroom for the capture
# tool. Override per run with `llmb-run submit --slurm-arg cpus-per-task=N`; that lands on
# the sbatch command line, which beats this directive (sbatch has no SBATCH_CPUS_PER_TASK variable).
#SBATCH --cpus-per-task=8

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL not set (framework should provide).}
export LLMB_EXPERIMENT_DIR=${LLMB_EXPERIMENT_DIR:?LLMB_EXPERIMENT_DIR not set (framework should provide).}

export WORKLOAD_KEY="inference_tasktrace_replay"
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

RECIPE_PY="$RECIPE_DIR/../../common/tasktrace_recipe.py"

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
MODEL_SIZE="${MODEL_SIZE:-nemotron-3-ultra_miniswe_django-11490}"
TRAJECTORY_ROOT="$(cd "$RECIPE_DIR/../../../.." && pwd)/trajectories"

TRACE_VARS="$("$PYTHON" "$RECIPE_PY" resolve-trace --root "$TRAJECTORY_ROOT" --selector "$MODEL_SIZE")" || exit 1
eval "$TRACE_VARS"
PACING="${RUN_CONF_TASKTRACE_PACING:-unpaced}"
CPUS="${RUN_CONF_TASKTRACE_CPUS:-4}"

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
    available=()
    shopt -s nullglob
    for candidate in "$TRACE_DIR"/*/trajectory.jsonl; do
        available+=("$(basename "$(dirname "$candidate")")")
    done
    shopt -u nullglob
    echo "[launch] Available: ${available[*]:-none}" >&2
    exit 1
}
[[ -s $IMAGE ]] || {
    echo "[launch] ERROR: task image missing at $IMAGE. Re-run llmb-install for $WORKLOAD_KEY (it imports container.images)." >&2
    exit 1
}
if [[ $PACING != "recorded" && $PACING != "unpaced" ]]; then
    echo "[launch] ERROR: RUN_CONF_TASKTRACE_PACING must be 'recorded' or 'unpaced' (got '$PACING')." >&2
    exit 1
fi
if [[ ${TRACE_TIMED:-1} != 1 && $PACING == "recorded" ]]; then
    echo "[launch] ERROR: the trajectory carries no recorded timings, so it can only replay unpaced (RUN_CONF_TASKTRACE_PACING is 'recorded')." >&2
    exit 1
fi
case "$CPUS" in *[!0-9]* | "" | 0)
    echo "[launch] ERROR: RUN_CONF_TASKTRACE_CPUS must be a positive integer (got '$CPUS')." >&2
    exit 1
    ;;
esac

CPUSET=$(taskset -cp $$ | awk -F': ' '{print $2}' | "$PYTHON" "$RECIPE_PY" cpuset --count "$CPUS")

OUT_DIR="$LLMB_EXPERIMENT_DIR/profile"
mkdir -p "$OUT_DIR"
export TMPDIR="${TMPDIR:-/tmp}/tasktrace-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR"
trap 'rm -rf "$TMPDIR"' EXIT

echo "[launch] node: $(hostname)  arch: $(uname -m)  gpu_type: ${GPU_TYPE:-unknown}"
echo "[launch] tasktrace: $TASKTRACE_BIN ($(tr '\n' ' ' < "$LLMB_WORKLOAD/bin/tasktrace.version"))"
echo "[launch] selection: $MODEL_SIZE -> $TRACE_MODEL / $TRACE_HARNESS / $TRACE_DATASET / $TRACE_ID"
echo "[launch] trace: $TRACE"
echo "[launch] image: $IMAGE"
echo "[launch] pacing: $PACING  cpus: $CPUS  cpuset: $CPUSET"
echo "[launch] output: $OUT_DIR"

TIMEOUT_ARGS=()
if [[ -n ${RUN_CONF_TASKTRACE_TIMEOUT:-} ]]; then
    TIMEOUT_ARGS=(--timeout-sec "$RUN_CONF_TASKTRACE_TIMEOUT")
fi

set +e
"$TASKTRACE_BIN" profile trace "$TRACE" \
    --runtime enroot \
    --image "$IMAGE" \
    --workdir /testbed \
    --allocated-cpus "$CPUS" \
    --cpuset-cpus "$CPUSET" \
    --pacing "$PACING" \
    --strace-mode none \
    "${TIMEOUT_ARGS[@]}" \
    --output-dir "$OUT_DIR"
REPLAY_EXIT=$?
set -e
echo "[launch] tasktrace exit status: $REPLAY_EXIT"

SUMMARY="$LLMB_EXPERIMENT_DIR/summary.json"
"$PYTHON" "$RECIPE_PY" summarize-replay \
    --out-dir "$OUT_DIR" --summary "$SUMMARY" --trace-id "$TRACE_ID" --pacing "$PACING" --cpus "$CPUS" \
    --cpuset "$CPUSET" --replay-exit "$REPLAY_EXIT"
STATUS=$?
echo "[launch] summary: $SUMMARY"
exit "$STATUS"
