#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --time=01:30:00
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

export WORKLOAD_KEY="inference_tasktrace_capture_replay"
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

LIVE_MODEL="${RUN_CONF_MODEL:-$LIVE_MODEL_DEFAULT}"
API_BASE="${RUN_CONF_API_BASE:-https://integrate.api.nvidia.com/v1}"
API_KEY_VAR="${RUN_CONF_API_KEY_VAR:-PUBLIC_NVIDIA_API_KEY}"
AGENT_TIMEOUT="${RUN_CONF_AGENT_TIMEOUT:-2400}"
PACING="${RUN_CONF_TASKTRACE_PACING:-unpaced}"
TRAJECTORY_TIMINGS="${RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS:-keep}"
CPUS="${RUN_CONF_TASKTRACE_CPUS:-4}"

TASKTRACE_BIN="$LLMB_WORKLOAD/bin/tasktrace"
RIPTIDE_VENV="$LLMB_WORKLOAD/venv"
CAPTURE_BIN="$RIPTIDE_VENV/bin/tasktrace-capture"
IMAGE="${RUN_CONF_IMAGE:-$LLMB_INSTALL/images/$IMAGE_FILE}"

# --- Preflight -----------------------------------------------------------------
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
command -v enroot > /dev/null 2>&1 || {
    echo "[launch] ERROR: enroot not found on PATH on $(hostname); both phases need it." >&2
    exit 1
}
if [[ $PACING != "recorded" && $PACING != "unpaced" ]]; then
    echo "[launch] ERROR: RUN_CONF_TASKTRACE_PACING must be 'recorded' or 'unpaced' (got '$PACING')." >&2
    exit 1
fi
if [[ $TRAJECTORY_TIMINGS != "keep" && $TRAJECTORY_TIMINGS != "omit" ]]; then
    echo "[launch] ERROR: RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS must be 'keep' or 'omit' (got '$TRAJECTORY_TIMINGS')." >&2
    exit 1
fi
if [[ $TRAJECTORY_TIMINGS == "omit" && $PACING == "recorded" ]]; then
    echo "[launch] ERROR: a trajectory captured without timings can only replay unpaced (RUN_CONF_TASKTRACE_PACING is 'recorded')." >&2
    exit 1
fi
case "$CPUS" in *[!0-9]* | "" | 0)
    echo "[launch] ERROR: RUN_CONF_TASKTRACE_CPUS must be a positive integer (got '$CPUS')." >&2
    exit 1
    ;;
esac
case "$AGENT_TIMEOUT" in *[!0-9]* | "" | 0)
    echo "[launch] ERROR: RUN_CONF_AGENT_TIMEOUT must be a positive integer (got '$AGENT_TIMEOUT')." >&2
    exit 1
    ;;
esac
if [[ ! $API_KEY_VAR =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
    echo "[launch] ERROR: RUN_CONF_API_KEY_VAR must be an environment variable name (got '$API_KEY_VAR')." >&2
    exit 1
fi
API_KEY="${!API_KEY_VAR:-}"
[[ -n $API_KEY ]] || {
    echo "[launch] ERROR: \$$API_KEY_VAR is empty. llmb-install collects it into the 'environment' section of cluster_config.yaml (re-run llmb-install, or add it there), or pass --env $API_KEY_VAR=... to 'llmb-run submit'." >&2
    exit 1
}
if [[ $API_BASE != */v1 ]]; then
    echo "[launch] WARNING: RUN_CONF_API_BASE does not end in /v1 ('$API_BASE'); mini-swe-agent's OpenAI client appends the route to the base." >&2
fi

if [[ $LIVE_MODEL != */* ]]; then
    echo "[launch] ERROR: RUN_CONF_MODEL must be <provider>/<model id> for mini-swe-agent; use openai/$LIVE_MODEL for an OpenAI-compatible endpoint." >&2
    exit 1
fi

ENDPOINT_MODEL_ID="${LIVE_MODEL#*/}"

case "$LLMB_EXPERIMENT_DIR" in *[[:space:]]*)
    echo "[launch] ERROR: LLMB_EXPERIMENT_DIR contains whitespace, which tasktrace-capture does not accept: '$LLMB_EXPERIMENT_DIR'" >&2
    exit 1
    ;;
esac

if [[ ${RUN_CONF_SKIP_ENDPOINT_CHECK:-0} != "1" ]]; then
    echo "[launch] endpoint check: POST $API_BASE/chat/completions model=$ENDPOINT_MODEL_ID (1 token)"
    set +e
    TT_API_KEY="$API_KEY" "$PYTHON" "$RECIPE_PY" endpoint-check --base "$API_BASE" --model "$ENDPOINT_MODEL_ID"
    ENDPOINT_CHECK_EXIT=$?
    set -e
    if [[ $ENDPOINT_CHECK_EXIT -ne 0 ]]; then
        echo "[launch] ERROR: aborting before the capture; set RUN_CONF_SKIP_ENDPOINT_CHECK=1 to bypass this check." >&2
        exit 1
    fi
fi

CPUSET=$(taskset -cp $$ | awk -F': ' '{print $2}' | "$PYTHON" "$RECIPE_PY" cpuset --count "$CPUS")

CAPTURE_DIR="$LLMB_EXPERIMENT_DIR/capture"
STAGED_DIR="$LLMB_EXPERIMENT_DIR/trajectories/$TRACE_MODEL/$TRACE_HARNESS/$TRACE_DATASET/$TRACE_ID"
PROFILE_DIR="$LLMB_EXPERIMENT_DIR/profile"
SUMMARY="$LLMB_EXPERIMENT_DIR/summary.json"
PROVENANCE="$STAGED_DIR/provenance.json"
export TMPDIR="${TMPDIR:-/tmp}/tasktrace-${SLURM_JOB_ID:-$$}"
mkdir -p "$TMPDIR" "$STAGED_DIR"
trap 'rm -rf "$TMPDIR"' EXIT
export MPLCONFIGDIR="$TMPDIR/mpl"

export PYTHONNOUSERSITE=1
export PATH="$RIPTIDE_VENV/bin:$PATH"

export HARBOR_ENROOT_CACHE="$LLMB_WORKLOAD/enroot-cache"
mkdir -p "$HARBOR_ENROOT_CACHE"
# tasktrace-capture resolves the Epoch image as <repo>:latest whatever the pin, so seed that name.
CACHE_KEY=$("$PYTHON" "$RECIPE_PY" image-cache-key --image-ref "${IMAGE_REF%%:sha256:*}:latest")
CACHE_PATH="$HARBOR_ENROOT_CACHE/$CACHE_KEY"
if [[ -e $CACHE_PATH && ! -L $CACHE_PATH ]]; then
    echo "[launch] image cache: replacing regular file $CACHE_PATH with a link to the installed image"
    rm -f "$CACHE_PATH"
fi
ln -sfn "$IMAGE" "$CACHE_PATH"
echo "[launch] image cache: $CACHE_PATH -> $IMAGE"
REPLAY_IMAGE="$(readlink -f "$CACHE_PATH")"
[[ -s $REPLAY_IMAGE ]] || {
    echo "[launch] ERROR: image cache entry $CACHE_PATH does not resolve to a file." >&2
    exit 1
}
echo "[launch] hashing task image $REPLAY_IMAGE"
IMAGE_SHA256=$(sha256sum "$REPLAY_IMAGE" | awk '{print $1}')

echo "[launch] node: $(hostname)  arch: $(uname -m)  gpu_type: ${GPU_TYPE:-unknown}"
echo "[launch] tasktrace: $TASKTRACE_BIN ($(tr '\n' ' ' < "$LLMB_WORKLOAD/bin/tasktrace.version"))"
echo "[launch] tasktrace-capture: $CAPTURE_BIN ($(tr '\n' ' ' < "$RIPTIDE_VENV.version"))"
echo "[launch] selection: $MODEL_SIZE -> $TRACE_MODEL / $TRACE_HARNESS / $TRACE_DATASET / $TRACE_ID"
echo "[launch] live model: $LIVE_MODEL (endpoint id: $ENDPOINT_MODEL_ID)  api base: $API_BASE  key: \$$API_KEY_VAR (set)"
echo "[launch] agent: $AGENT  timeout: ${AGENT_TIMEOUT}s  concurrency: 1  trials: 1  shell tracing: on"
echo "[launch] image: $REPLAY_IMAGE  sha256: $IMAGE_SHA256"
echo "[launch] cpu pin (both phases): cpus: $CPUS  cpuset: $CPUSET  replay pacing: $PACING"
echo "[launch] output: $LLMB_EXPERIMENT_DIR"

# --- Phase 1: capture ------------------------------------------------------------
CAPTURE_ARGS=(
    --runtime enroot
    --task "$TRACE_DATASET/$TRACE_ID"
    --agent "$AGENT"
    --model "$LIVE_MODEL"
    --n-concurrent 1
    -k 1
    --agent-timeout "$AGENT_TIMEOUT"
    --instrument-shell-commands
    --instrument-shell-perfetto-depth 1
    --output-dir "$CAPTURE_DIR"
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

echo "[launch] capture: taskset -c $CPUSET tasktrace-capture ${CAPTURE_ARGS[*]} --agent-env OPENAI_API_BASE=$API_BASE --agent-env OPENAI_BASE_URL=$API_BASE --agent-env OPENAI_API_KEY=<\$$API_KEY_VAR>"
CAPTURE_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
cd "$LLMB_EXPERIMENT_DIR"
set +e
taskset -c "$CPUSET" "$CAPTURE_BIN" "${CAPTURE_ARGS[@]}" \
    --agent-env "OPENAI_API_BASE=$API_BASE" \
    --agent-env "OPENAI_BASE_URL=$API_BASE" \
    --agent-env "OPENAI_API_KEY=$API_KEY"
CAPTURE_EXIT=$?
set -e
CAPTURE_FINISHED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
echo "[launch] tasktrace-capture exit status: $CAPTURE_EXIT"

# --- Harvest the trajectory ------------------------------------------------------
TRACE=""
if [[ -d $CAPTURE_DIR/runs ]]; then

    TRACE=$(find "$CAPTURE_DIR/runs" -type f -path '*/agent/tasktrace_capture_replay_trajectory.jsonl' -printf '%T@ %p\n' 2> /dev/null \
        | sort -n | tail -n 1 | cut -d' ' -f2-)
fi
if [[ -z $TRACE ]]; then
    echo "[launch] ERROR: capture produced no finalized trajectory under $CAPTURE_DIR/runs." >&2
    echo "[launch] Trial directories:" >&2
    find "$CAPTURE_DIR/runs" -mindepth 2 -maxdepth 2 -type d 2> /dev/null | sed 's/^/[launch]   /' >&2 || true
    for log in "$CAPTURE_DIR"/runs/*/*/trial.log; do
        [[ -f $log ]] || continue
        echo "[launch] tail of $log:" >&2
        tail -n 20 "$log" | sed 's/^/[launch]   /' >&2
    done
fi

set +e
TT_SELECTOR="$MODEL_SIZE" TT_TRACE_MODEL="$TRACE_MODEL" TT_TRACE_HARNESS="$TRACE_HARNESS" \
    TT_TRACE_DATASET="$TRACE_DATASET" TT_TRACE_ID="$TRACE_ID" TT_AGENT="$AGENT" \
    TT_LIVE_MODEL="$LIVE_MODEL" TT_ENDPOINT_MODEL_ID="$ENDPOINT_MODEL_ID" TT_API_BASE="$API_BASE" TT_API_KEY_VAR="$API_KEY_VAR" \
    TT_AGENT_TIMEOUT="$AGENT_TIMEOUT" TT_CAPTURE_EXIT="$CAPTURE_EXIT" \
    TT_CAPTURE_STARTED_AT="$CAPTURE_STARTED_AT" TT_CAPTURE_FINISHED_AT="$CAPTURE_FINISHED_AT" \
    TT_CAPTURE_ARGS="${CAPTURE_ARGS[*]}" TT_CAPTURE_DIR="$CAPTURE_DIR" \
    TT_IMAGE_REF="$IMAGE_REF" TT_IMAGE_PATH="$REPLAY_IMAGE" TT_IMAGE_SHA256="$IMAGE_SHA256" TT_IMAGE_ARCH="$IMAGE_ARCH" \
    TT_VENV_VERSION_FILE="$RIPTIDE_VENV.version" TT_TASKTRACE_VERSION_FILE="$LLMB_WORKLOAD/bin/tasktrace.version" \
    TT_NODE="$(hostname)" TT_JOB_ID="${SLURM_JOB_ID:-}" TT_GPU_TYPE="${GPU_TYPE:-}" TT_CPUSET="$CPUSET" TT_CPUS="$CPUS" \
    "$PYTHON" "$RECIPE_PY" stage --trace "$TRACE" --staged-dir "$STAGED_DIR" --provenance "$PROVENANCE"
STAGE_EXIT=$?
set -e

# --- Phase 2: replay -------------------------------------------------------------
REPLAY_EXIT=""
STAGED_TRACE="$STAGED_DIR/trajectory.jsonl"
TRACE_TIMED=1
if [[ $STAGE_EXIT -eq 0 ]]; then
    # Replay what the capture wrote, whichever setting omitted the timings (RUN_CONF_CAPTURE_TRAJECTORY_TIMINGS,
    # or --trajectory-timings in RUN_CONF_CAPTURE_ARGS).
    eval "$("$PYTHON" "$RECIPE_PY" trace-timed --trace "$STAGED_TRACE")"
fi
if [[ $STAGE_EXIT -eq 0 && $TRACE_TIMED != 1 && $PACING == "recorded" ]]; then
    echo "[launch] ERROR: the capture wrote no timings, so its trajectory can only replay unpaced (RUN_CONF_TASKTRACE_PACING is 'recorded'); skipping the replay." >&2
elif [[ $STAGE_EXIT -eq 0 ]]; then
    TIMEOUT_ARGS=()
    if [[ -n ${RUN_CONF_TASKTRACE_TIMEOUT:-} ]]; then
        TIMEOUT_ARGS=(--timeout-sec "$RUN_CONF_TASKTRACE_TIMEOUT")
    fi
    mkdir -p "$PROFILE_DIR"
    echo "[launch] replay: tasktrace profile trace $STAGED_TRACE --runtime enroot --image $REPLAY_IMAGE --pacing $PACING"
    set +e
    "$TASKTRACE_BIN" profile trace "$STAGED_TRACE" \
        --runtime enroot \
        --image "$REPLAY_IMAGE" \
        --workdir /testbed \
        --allocated-cpus "$CPUS" \
        --cpuset-cpus "$CPUSET" \
        --pacing "$PACING" \
        --strace-mode none \
        ${TIMEOUT_ARGS[@]+"${TIMEOUT_ARGS[@]}"} \
        --output-dir "$PROFILE_DIR"
    REPLAY_EXIT=$?
    set -e
    echo "[launch] tasktrace exit status: $REPLAY_EXIT"
else
    echo "[launch] skipping replay: the capture did not yield a replayable trajectory." >&2
fi

# --- Secret scan -------------------------------------------------------------------

SCRUB_REPORT="$LLMB_EXPERIMENT_DIR/secret_scan.json"
TT_SECRET="$API_KEY" "$PYTHON" "$RECIPE_PY" secret-scan --root "$LLMB_EXPERIMENT_DIR" --report "$SCRUB_REPORT"

# --- Summary -----------------------------------------------------------------------

set +e
"$PYTHON" "$RECIPE_PY" summarize-capture \
    --provenance "$PROVENANCE" --profile-dir "$PROFILE_DIR" --summary "$SUMMARY" --scrub-report "$SCRUB_REPORT" \
    --pacing "$PACING" --cpus "$CPUS" --cpuset "$CPUSET" --stage-exit "$STAGE_EXIT" --replay-exit "${REPLAY_EXIT:-}" \
    --capture-dir "$CAPTURE_DIR"
STATUS=$?
set -e
echo "[launch] summary: $SUMMARY"
exit "$STATUS"
