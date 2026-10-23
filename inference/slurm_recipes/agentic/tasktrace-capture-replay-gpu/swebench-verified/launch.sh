#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=02:00:00
# One gb300 node: llmb-run turns --scale 4 into -N 1 --ntasks-per-node=4 plus the partition's
# GPUs-per-node request. 32 CPUs per task (128 of the node's 144) land in the batch step's cpuset:
# the capture and the replay pin RUN_CONF_TASKTRACE_CPUS of them (default 4), the vLLM server
# takes the rest.
#SBATCH --cpus-per-task=32

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL not set (framework should provide).}
export LLMB_EXPERIMENT_DIR=${LLMB_EXPERIMENT_DIR:?LLMB_EXPERIMENT_DIR not set (framework should provide).}

export WORKLOAD_KEY="inference_tasktrace_capture_replay_gpu"
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

# shellcheck source=../../common/tasktrace_gpu_lib.sh
source "$COMMON_DIR/tasktrace_gpu_lib.sh"
TRAJECTORY_TIMINGS="${RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS:-keep}"
if [[ $TRAJECTORY_TIMINGS != "keep" && $TRAJECTORY_TIMINGS != "omit" ]]; then
    echo "[launch] ERROR: RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS must be 'keep' or 'omit' (got '$TRAJECTORY_TIMINGS')." >&2
    exit 1
fi
# The knobs are checked against the requested timings now and again, after the capture, against what it wrote.
gpu_recipe_knobs "$([[ $TRAJECTORY_TIMINGS == "omit" ]] && echo 0 || echo 1)"
llm_server_load_config "${SERVING_CONFIG:-$TRACE_MODEL}"
LIVE_MODEL="openai/$SERVING_SERVED_MODEL_NAME"
API_BASE="$LLM_SERVER_API_BASE"
LOCAL_API_KEY="tasktrace-local-server" # vLLM runs without --api-key; the OpenAI client still wants one
AGENT_TIMEOUT="${RUN_CONF_AGENT_TIMEOUT:-2400}"

TASKTRACE_BIN="$LLMB_WORKLOAD/bin/tasktrace"
RIPTIDE_VENV="$LLMB_WORKLOAD/venv"
CAPTURE_BIN="$RIPTIDE_VENV/bin/tasktrace-capture"
IMAGE="${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/$IMAGE_FILE}"

# --- Preflight -----------------------------------------------------------------------------
if [[ $(uname -m) != "$IMAGE_ARCH" ]]; then
    echo "[launch] ERROR: task image is $IMAGE_ARCH but this node is $(uname -m)." >&2
    exit 1
fi
[[ -x $TASKTRACE_BIN ]] || {
    echo "[launch] ERROR: tasktrace binary missing at $TASKTRACE_BIN. Re-run llmb-install for $WORKLOAD_KEY." >&2
    exit 1
}
[[ -x $CAPTURE_BIN ]] || {
    echo "[launch] ERROR: tasktrace-capture missing at $CAPTURE_BIN. Re-run llmb-install for $WORKLOAD_KEY." >&2
    exit 1
}
[[ -s $IMAGE ]] || {
    echo "[launch] ERROR: task image missing at $IMAGE. Re-run llmb-install for $WORKLOAD_KEY (it imports container.images)." >&2
    exit 1
}
case "$AGENT_TIMEOUT" in *[!0-9]* | "" | 0)
    echo "[launch] ERROR: RUN_CONF_AGENT_TIMEOUT must be a positive integer (got '$AGENT_TIMEOUT')." >&2
    exit 1
    ;;
esac
case "$LLMB_EXPERIMENT_DIR" in *[[:space:]]*)
    echo "[launch] ERROR: LLMB_EXPERIMENT_DIR contains whitespace, which tasktrace-capture does not accept: '$LLMB_EXPERIMENT_DIR'" >&2
    exit 1
    ;;
esac
llm_server_preflight
gpu_recipe_cpusets

CAPTURE_DIR="$LLMB_EXPERIMENT_DIR/capture"
STAGED_DIR="$LLMB_EXPERIMENT_DIR/trajectories/$TRACE_MODEL/$TRACE_HARNESS/$TRACE_DATASET/$TRACE_ID"
PROFILE_DIR="$LLMB_EXPERIMENT_DIR/profile"
LLM_DIR="$LLMB_EXPERIMENT_DIR/llm-replay"
SERVER_DIR="$LLMB_EXPERIMENT_DIR/server"
SUMMARY="$LLMB_EXPERIMENT_DIR/summary.json"
PROVENANCE="$STAGED_DIR/provenance.json"
export TMPDIR="${TMPDIR:-/tmp}/tasktrace-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$STAGED_DIR" "$LLM_DIR" "$SERVER_DIR"
trap 'llm_server_stop; rm -rf "$TMPDIR"' EXIT
export MPLCONFIGDIR="$TMPDIR/mpl"

export PYTHONNOUSERSITE=1
export PATH="$RIPTIDE_VENV/bin:$PATH"

# tasktrace-capture pulls the task image by name into its cache; point that cache entry at the
# installed .sqsh so the live run uses the installer's image and the node pulls nothing.
export HARBOR_ENROOT_CACHE="$LLMB_WORKLOAD/enroot-cache"
mkdir -p "$HARBOR_ENROOT_CACHE"
CACHE_KEY=$("$PYTHON" "$RECIPE_PY" image-cache-key --image-ref "${IMAGE_REF%%:sha256:*}:latest")
CACHE_PATH="$HARBOR_ENROOT_CACHE/$CACHE_KEY"
if [[ -e $CACHE_PATH && ! -L $CACHE_PATH ]]; then
    rm -f "$CACHE_PATH"
fi
ln -sfn "$IMAGE" "$CACHE_PATH"
REPLAY_IMAGE="$(readlink -f "$CACHE_PATH")"
[[ -s $REPLAY_IMAGE ]] || {
    echo "[launch] ERROR: image cache entry $CACHE_PATH does not resolve to a file." >&2
    exit 1
}
IMAGE_SHA256=$(sha256sum "$REPLAY_IMAGE" | awk '{print $1}')

echo "[launch] node: $(hostname)  arch: $(uname -m)  gpus: $LLM_SERVER_GPU_COUNT x ${LLM_SERVER_GPU_NAME:-GPU}"
echo "[launch] tasktrace: $TASKTRACE_BIN ($(tr '\n' ' ' < "$LLMB_WORKLOAD/bin/tasktrace.version"))"
echo "[launch] tasktrace-capture: $CAPTURE_BIN ($(tr '\n' ' ' < "$RIPTIDE_VENV.version"))"
echo "[launch] selection: $MODEL_SIZE -> $TRACE_MODEL / $TRACE_HARNESS / $TRACE_DATASET / $TRACE_ID"
echo "[launch] server: $SERVING_MODEL_REPO@${SERVING_MODEL_REVISION:-default} as '$SERVING_SERVED_MODEL_NAME'  TP$SERVING_TP  (config $SERVING_NAME)"
echo "[launch] agent: $AGENT  model: $LIVE_MODEL  api base: $API_BASE  timeout: ${AGENT_TIMEOUT}s  cpuset: $CPUSET"
echo "[launch] image: $REPLAY_IMAGE  sha256: $IMAGE_SHA256"
echo "[launch] output: $LLMB_EXPERIMENT_DIR"

llm_server_bring_up "$SERVER_DIR" "$TRACE"

echo "[launch] endpoint check: POST $API_BASE/chat/completions model=$SERVING_SERVED_MODEL_NAME (1 token)"
TT_API_KEY="" "$PYTHON" "$RECIPE_PY" endpoint-check --base "$API_BASE" --model "$SERVING_SERVED_MODEL_NAME" || {
    echo "[launch] ERROR: the server is up but did not answer a chat completion; aborting before the capture (see $SERVER_DIR/vllm.log)." >&2
    exit 1
}

# --- Phase 1: capture --------------------------------------------------------------------
# --api-base tells tasktrace-capture where the model is (it scrapes the server's /metrics during
# the run); the agent itself learns the endpoint from the OPENAI_* variables.
CAPTURE_ARGS=(
    --runtime enroot
    --task "$TRACE_DATASET/$TRACE_ID"
    --agent "$AGENT"
    --model "$LIVE_MODEL"
    --api-base "$API_BASE"
    --allow-private-api
    --n-concurrent 1
    -k 1
    --agent-timeout "$AGENT_TIMEOUT"
    --instrument-shell-commands
    --instrument-shell-perfetto-depth 1
    --output-dir "$CAPTURE_DIR"
    --agent-env UV_EXCLUDE_NEWER=2026-10-12T00:00:00Z
)
if [[ ${RUN_CONF_SOURCE_IMAGE_BASHRC:-1} == 1 ]]; then
    CAPTURE_ARGS+=(--source-image-bashrc)
fi
if [[ $TRAJECTORY_TIMINGS == "omit" ]]; then
    CAPTURE_ARGS+=(--trajectory-timings omit)
fi
if [[ -n ${RUN_CONF_CAPTURE_ARGS:-} ]]; then
    read -r -a EXTRA_CAPTURE_ARGS <<< "$RUN_CONF_CAPTURE_ARGS"
    CAPTURE_ARGS+=(${EXTRA_CAPTURE_ARGS[@]+"${EXTRA_CAPTURE_ARGS[@]}"})
fi

echo "[launch] capture: taskset -c $CPUSET tasktrace-capture ${CAPTURE_ARGS[*]}"
CAPTURE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cd "$LLMB_EXPERIMENT_DIR"
set +e
taskset -c "$CPUSET" "$CAPTURE_BIN" "${CAPTURE_ARGS[@]}" \
    --agent-env "OPENAI_API_BASE=$API_BASE" \
    --agent-env "OPENAI_BASE_URL=$API_BASE" \
    --agent-env "OPENAI_API_KEY=$LOCAL_API_KEY"
CAPTURE_EXIT=$?
set -e
CAPTURE_FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[launch] tasktrace-capture exit status: $CAPTURE_EXIT"

# --- Harvest and stage the trajectory --------------------------------------------------------
TRACE=""
if [[ -d $CAPTURE_DIR/runs ]]; then
    TRACE=$(find "$CAPTURE_DIR/runs" -type f -path '*/agent/tasktrace_capture_replay_trajectory.jsonl' -printf '%T@ %p\n' 2> /dev/null \
        | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
if [[ -z $TRACE ]]; then
    echo "[launch] ERROR: capture produced no finalized trajectory under $CAPTURE_DIR/runs." >&2
    for log in "$CAPTURE_DIR"/runs/*/*/trial.log; do
        [[ -f $log ]] || continue
        echo "[launch] tail of $log:" >&2
        tail -n 20 "$log" | sed 's/^/[launch]   /' >&2
    done
fi

set +e
TT_SELECTOR="$MODEL_SIZE" \
    TT_TRACE_MODEL="$TRACE_MODEL" \
    TT_TRACE_HARNESS="$TRACE_HARNESS" \
    TT_TRACE_DATASET="$TRACE_DATASET" \
    TT_TRACE_ID="$TRACE_ID" \
    TT_AGENT="$AGENT" \
    TT_LIVE_MODEL="$LIVE_MODEL" \
    TT_LIVE_MODEL_DEFAULT="$LIVE_MODEL_DEFAULT" \
    TT_ENDPOINT_MODEL_ID="$SERVING_SERVED_MODEL_NAME" \
    TT_API_BASE="$API_BASE" \
    TT_API_KEY_VAR="" \
    TT_LLM_MODE="live-server" \
    TT_SERVER_INFO="$SERVER_DIR/server.json" \
    TT_SERVER_CONFIG="$SERVING_NAME" \
    TT_AGENT_TIMEOUT="$AGENT_TIMEOUT" \
    TT_CAPTURE_EXIT="$CAPTURE_EXIT" \
    TT_CAPTURE_STARTED_AT="$CAPTURE_STARTED_AT" \
    TT_CAPTURE_FINISHED_AT="$CAPTURE_FINISHED_AT" \
    TT_CAPTURE_ARGS="${CAPTURE_ARGS[*]}" \
    TT_CAPTURE_DIR="$CAPTURE_DIR" \
    TT_IMAGE_REF="$IMAGE_REF" \
    TT_IMAGE_PATH="$REPLAY_IMAGE" \
    TT_IMAGE_SHA256="$IMAGE_SHA256" \
    TT_IMAGE_ARCH="$IMAGE_ARCH" \
    TT_VENV_VERSION_FILE="$RIPTIDE_VENV.version" \
    TT_TASKTRACE_VERSION_FILE="$LLMB_WORKLOAD/bin/tasktrace.version" \
    TT_NODE="$(hostname)" \
    TT_JOB_ID="${SLURM_JOB_ID:-}" \
    TT_GPU_TYPE="${GPU_TYPE:-}" \
    TT_CPUSET="$CPUSET" \
    TT_CPUS="$CPUS" \
    "$PYTHON" "$RECIPE_PY" stage \
    --trace "$TRACE" \
    --staged-dir "$STAGED_DIR" \
    --provenance "$PROVENANCE"
STAGE_EXIT=$?
set -e

# --- Phase 2: replay both halves against the same server ------------------------------------
REPLAY_EXIT=""
LLM_EXIT=""
LLM_REPORT="$LLM_DIR/llm_replay_report.json"
if [[ $STAGE_EXIT -eq 0 ]]; then
    # Replay what the capture wrote, whichever setting omitted the timings (RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS,
    # or --trajectory-timings in RUN_CONF_CAPTURE_ARGS): an untimed recording replays in loop order.
    eval "$("$PYTHON" "$RECIPE_PY" trace-timed --trace "$STAGED_DIR/trajectory.jsonl")"
    if gpu_recipe_knobs "$TRACE_TIMED"; then
        gpu_replay "$STAGED_DIR/trajectory.jsonl" "$REPLAY_IMAGE" "$PROFILE_DIR" "$LLM_DIR"
    else
        echo "[launch] skipping replay: the pacing settings do not fit the recording the capture wrote." >&2
    fi
else
    echo "[launch] skipping replay: the capture did not yield a replayable trajectory." >&2
fi

llm_server_stop

# --- Secret scan and summary ---------------------------------------------------------------
# No key is configured (TT_SECRET empty); the scan still redacts anything shaped like one.
SCRUB_REPORT="$LLMB_EXPERIMENT_DIR/secret_scan.json"
TT_SECRET="" "$PYTHON" "$RECIPE_PY" secret-scan --root "$LLMB_EXPERIMENT_DIR" --report "$SCRUB_REPORT"

set +e
"$PYTHON" "$RECIPE_PY" summarize-capture \
    --provenance "$PROVENANCE" \
    --profile-dir "$PROFILE_DIR" \
    --summary "$SUMMARY" \
    --scrub-report "$SCRUB_REPORT" \
    --pacing "$PACING" \
    --cpus "$CPUS" \
    --cpuset "$CPUSET" \
    --stage-exit "$STAGE_EXIT" \
    --replay-exit "${REPLAY_EXIT:-}" \
    --capture-dir "$CAPTURE_DIR" \
    --workload tasktrace_capture_replay_gpu \
    --llm-report "$LLM_REPORT" \
    --llm-exit "${LLM_EXIT:-}" \
    --server-info "$SERVER_DIR/server.json"
STATUS=$?
set -e
echo "[launch] summary: $SUMMARY"
exit "$STATUS"
