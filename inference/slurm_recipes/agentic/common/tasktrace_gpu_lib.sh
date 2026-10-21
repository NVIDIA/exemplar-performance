# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# shellcheck shell=bash

# Steps shared by the *-gpu launch scripts (sourced; functions only). The
# caller provides PYTHON, RECIPE_PY, COMMON_DIR, TASKTRACE_BIN, LLMB_INSTALL,
# LLMB_WORKLOAD and WORKLOAD_KEY, then calls in order:
#   gpu_recipe_knobs, llm_server_load_config, llm_server_preflight,
#   gpu_recipe_cpusets, llm_server_bring_up, gpu_replay (gpu_replay_both_sides / gpu_replay_loop_order),
#   llm_server_stop (also from the EXIT trap; it is idempotent).
#
# The vLLM server runs in an enroot container on the node, like the recipes'
# task containers, pinned to the CPUs the replay does not use, and binds
# 127.0.0.1 (enroot shares the host network, so the agent inside the task
# container reaches it). Its shape comes from common/serving/<name>.env; the
# knobs below only move things around:
#   RUN_CONF_SERVER_STARTUP_TIMEOUT  seconds to wait for /health (default: the serving config's)
#   RUN_CONF_SERVER_PORT             port on 127.0.0.1 (default 8000)
#   RUN_CONF_SERVER_MODEL_DIR        model snapshot directory (default: $LLMB_INSTALL/datasets/<name>)
#   RUN_CONF_SERVER_IMAGE            serving image .sqsh (default: the imported container.images entry)

LLM_SERVER_PID=""
LLM_SERVER_EXIT=""

_llm_server_err() {
    echo "[launch] ERROR: server $*" >&2
}

# llm_server_load_config <name>: source common/serving/<name>.env and apply overrides.
llm_server_load_config() {
    local name=$1 file="$COMMON_DIR/serving/$1.env" var available
    if [[ ! -f $file ]]; then
        available=$(find "$COMMON_DIR/serving" -maxdepth 1 -name '*.env' -printf '%f\n' 2> /dev/null | sed 's/\.env$//' | tr '\n' ' ')
        _llm_server_err "config '$name' not found at $file (available: ${available:-none}); the trajectory's trace.json names it (serving_config), else its model token."
        return 1
    fi
    # shellcheck source=/dev/null
    source "$file"
    for var in SERVING_NAME SERVING_ENGINE SERVING_MODEL_REPO SERVING_MODEL_DIR_NAME SERVING_SERVED_MODEL_NAME \
        SERVING_IMAGE_REF SERVING_TP SERVING_MAX_MODEL_LEN SERVING_GPU_MEM_UTIL SERVING_STARTUP_TIMEOUT_SEC; do
        if [[ -z ${!var:-} ]]; then
            _llm_server_err "config $file does not set $var."
            return 1
        fi
    done
    if [[ $SERVING_ENGINE != "vllm" ]]; then
        _llm_server_err "config $file names engine '$SERVING_ENGINE'; only vllm is supported."
        return 1
    fi
    SERVING_STARTUP_TIMEOUT_SEC="${RUN_CONF_SERVER_STARTUP_TIMEOUT:-$SERVING_STARTUP_TIMEOUT_SEC}"
    LLM_SERVER_PORT="${RUN_CONF_SERVER_PORT:-8000}"
    for var in SERVING_TP SERVING_MAX_MODEL_LEN SERVING_STARTUP_TIMEOUT_SEC LLM_SERVER_PORT; do
        case "${!var}" in *[!0-9]* | "" | 0)
            _llm_server_err "$var must be a positive integer (got '${!var}')."
            return 1
            ;;
        esac
    done
    LLM_SERVER_MODEL_DIR="${RUN_CONF_SERVER_MODEL_DIR:-$LLMB_INSTALL/datasets/$SERVING_MODEL_DIR_NAME}"
    LLM_SERVER_IMAGE="${RUN_CONF_SERVER_IMAGE:-$LLMB_INSTALL/images/$("$PYTHON" "$RECIPE_PY" image-file-name --image-ref "$SERVING_IMAGE_REF")}"
    LLM_SERVER_API_BASE="http://127.0.0.1:$LLM_SERVER_PORT/v1"
    LLM_SERVER_NAME="tasktrace-llm-${SLURM_JOB_ID:-$$}"
    # Home for the serving container, node-local and per job: the JIT kernel caches (Triton,
    # FlashInfer, torch inductor) live here and must not be shared across runs or filesystems.
    # Every run starts cold; the warm-up requests absorb the compile time.
    LLM_SERVER_HOME="${TMPDIR:-/tmp}/tasktrace-llm-home-${SLURM_JOB_ID:-$$}"
    export LLM_SERVER_PORT LLM_SERVER_MODEL_DIR LLM_SERVER_IMAGE LLM_SERVER_API_BASE LLM_SERVER_NAME LLM_SERVER_HOME
}

llm_server_preflight() {
    local gpus=()
    command -v enroot > /dev/null 2>&1 || {
        _llm_server_err "enroot not found on PATH on $(hostname); the server runs in an enroot container."
        return 1
    }
    command -v curl > /dev/null 2>&1 || {
        _llm_server_err "curl not found on PATH on $(hostname); the server's readiness checks use it."
        return 1
    }
    command -v nvidia-smi > /dev/null 2>&1 || {
        _llm_server_err "nvidia-smi not found on $(hostname); this node exposes no GPUs. The gpu recipes need the GPU partition (llmb-run submits them there when run.requires_gpus is true)."
        return 1
    }
    mapfile -t gpus < <(nvidia-smi --query-gpu=name --format=csv,noheader 2> /dev/null | sed 's/^ *//; s/ *$//')
    LLM_SERVER_GPU_COUNT=${#gpus[@]}
    LLM_SERVER_GPU_NAME="${gpus[0]:-}"
    if ((LLM_SERVER_GPU_COUNT < SERVING_TP)); then
        _llm_server_err "allocation exposes $LLM_SERVER_GPU_COUNT GPUs, need $SERVING_TP for tensor parallelism $SERVING_TP. Submit with --scale $SERVING_TP (one node)."
        return 1
    fi
    [[ -s $LLM_SERVER_IMAGE ]] || {
        _llm_server_err "serving image missing at $LLM_SERVER_IMAGE. Re-run llmb-install for $WORKLOAD_KEY (it imports container.images), or point RUN_CONF_SERVER_IMAGE at a .sqsh."
        return 1
    }
    [[ -f $LLM_SERVER_MODEL_DIR/config.json ]] || {
        _llm_server_err "model snapshot missing at $LLM_SERVER_MODEL_DIR (no config.json). Re-run llmb-install for $WORKLOAD_KEY (it downloads $SERVING_MODEL_REPO under downloads.huggingface.repos), or point RUN_CONF_SERVER_MODEL_DIR at a snapshot."
        return 1
    }
    if (exec 3<> "/dev/tcp/127.0.0.1/$LLM_SERVER_PORT") 2> /dev/null; then
        _llm_server_err "port $LLM_SERVER_PORT on 127.0.0.1 is already in use on $(hostname); set RUN_CONF_SERVER_PORT."
        return 1
    fi
    export LLM_SERVER_GPU_COUNT LLM_SERVER_GPU_NAME
    echo "[server] preflight: $LLM_SERVER_GPU_COUNT x ${LLM_SERVER_GPU_NAME:-GPU}, image $LLM_SERVER_IMAGE, model $LLM_SERVER_MODEL_DIR"
}

# llm_server_start <cpuset> <server-dir>: create the container and start vLLM in the background,
# pinned to <cpuset>, logging to <server-dir>/vllm.log.
llm_server_start() {
    local cpuset=$1 dir=$2 cmd=() start=() var
    LLM_SERVER_CPUSET="$cpuset"
    LLM_SERVER_DIR="$dir"
    LLM_SERVER_LOG="$dir/vllm.log"
    mkdir -p "$dir" "$LLM_SERVER_HOME"
    enroot remove -f "$LLM_SERVER_NAME" > /dev/null 2>&1 || true
    echo "[server] enroot create $LLM_SERVER_NAME <- $LLM_SERVER_IMAGE"
    if ! enroot create --name "$LLM_SERVER_NAME" "$LLM_SERVER_IMAGE" > "$dir/enroot-create.log" 2>&1; then
        cat "$dir/enroot-create.log" >&2
        _llm_server_err "enroot create failed for $LLM_SERVER_IMAGE (log above)."
        return 1
    fi
    cmd=(
        python3 -m vllm.entrypoints.openai.api_server
        --model /model
        --served-model-name "$SERVING_SERVED_MODEL_NAME"
        --tensor-parallel-size "$SERVING_TP"
        --max-model-len "$SERVING_MAX_MODEL_LEN"
        --gpu-memory-utilization "$SERVING_GPU_MEM_UTIL"
        --host 127.0.0.1
        --port "$LLM_SERVER_PORT"
        "${SERVING_EXTRA_ARGS[@]}"
        --trust-remote-code
    )
    start=(
        enroot start
        --rw
        --mount "$LLM_SERVER_MODEL_DIR:/model"
        --mount "$LLM_SERVER_HOME:/llm-home"
        --env HOME=/llm-home
        --env XDG_CACHE_HOME=/llm-home/.cache
        --env NVIDIA_VISIBLE_DEVICES=all
        --env "NVIDIA_DRIVER_CAPABILITIES=compute,utility"
        --env PYTHONUNBUFFERED=1
    )
    for var in "${SERVING_ENV[@]}"; do
        start+=(--env "$var")
    done
    echo "[server] command: ${cmd[*]}"
    echo "[server] cpuset: $cpuset  log: $LLM_SERVER_LOG"
    # Its own session (setsid): the server and its TP workers form a process group the stop
    # function can signal without touching the launch script's own group.
    (exec setsid taskset -c "$cpuset" "${start[@]}" "$LLM_SERVER_NAME" "${cmd[@]}") > "$LLM_SERVER_LOG" 2>&1 &
    LLM_SERVER_PID=$!
    LLM_SERVER_STARTED_EPOCH=$(date +%s)
    LLM_SERVER_STARTED_AT=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    export LLM_SERVER_CPUSET LLM_SERVER_DIR LLM_SERVER_LOG LLM_SERVER_STARTED_AT
    echo "[server] started pid $LLM_SERVER_PID at $LLM_SERVER_STARTED_AT"
}

_llm_server_log_tail() {
    echo "[server] last lines of $LLM_SERVER_LOG:" >&2
    tail -n 40 "$LLM_SERVER_LOG" 2> /dev/null | sed 's/^/[server]   /' >&2 || true
}

# llm_server_wait_ready: poll /health until the model is loaded (vLLM binds the port only then),
# then confirm /v1/models serves the configured id. Fails fast when the server process dies.
llm_server_wait_ready() {
    local health="${LLM_SERVER_API_BASE%/v1}/health" deadline now
    deadline=$((LLM_SERVER_STARTED_EPOCH + SERVING_STARTUP_TIMEOUT_SEC))
    echo "[server] waiting for $health (up to ${SERVING_STARTUP_TIMEOUT_SEC}s for the model load)"
    while :; do
        if ! kill -0 "$LLM_SERVER_PID" 2> /dev/null; then
            _llm_server_log_tail
            _llm_server_err "exited before it became ready (pid $LLM_SERVER_PID); see $LLM_SERVER_LOG."
            return 1
        fi
        if curl -sf --max-time 10 -o /dev/null "$health"; then
            break
        fi
        now=$(date +%s)
        if ((now >= deadline)); then
            _llm_server_log_tail
            _llm_server_err "did not become ready within ${SERVING_STARTUP_TIMEOUT_SEC}s; see $LLM_SERVER_LOG (raise RUN_CONF_SERVER_STARTUP_TIMEOUT for a slower filesystem)."
            return 1
        fi
        sleep 15
    done
    LLM_SERVER_STARTUP_SEC=$(($(date +%s) - LLM_SERVER_STARTED_EPOCH))
    if ! curl -sf --max-time 30 "$LLM_SERVER_API_BASE/models" | grep -q "\"id\": *\"$SERVING_SERVED_MODEL_NAME\""; then
        _llm_server_err "is up but GET $LLM_SERVER_API_BASE/models does not list '$SERVING_SERVED_MODEL_NAME'."
        return 1
    fi
    LLM_SERVER_VERSION=$(curl -sf --max-time 10 "${LLM_SERVER_API_BASE%/v1}/version" | sed -n 's/.*"version": *"\([^"]*\)".*/\1/p' || true)
    export LLM_SERVER_STARTUP_SEC LLM_SERVER_VERSION
    echo "[server] ready after ${LLM_SERVER_STARTUP_SEC}s: $LLM_SERVER_API_BASE (vllm ${LLM_SERVER_VERSION:-unknown}, model $SERVING_SERVED_MODEL_NAME)"
}

# llm_server_export_info: the TT_SERVER_* variables `server-info` and `stage` read.
llm_server_export_info() {
    export \
        TT_SERVER_CONFIG="$SERVING_NAME" \
        TT_SERVER_ENGINE="$SERVING_ENGINE" \
        TT_SERVER_VERSION="${LLM_SERVER_VERSION:-}" \
        TT_SERVER_IMAGE_REF="$SERVING_IMAGE_REF" \
        TT_SERVER_IMAGE_PATH="$LLM_SERVER_IMAGE" \
        TT_SERVER_MODEL_REPO="$SERVING_MODEL_REPO" \
        TT_SERVER_MODEL_REVISION="${SERVING_MODEL_REVISION:-}" \
        TT_SERVER_MODEL_DIR="$LLM_SERVER_MODEL_DIR" \
        TT_SERVER_SERVED_MODEL="$SERVING_SERVED_MODEL_NAME" \
        TT_SERVER_TP="$SERVING_TP" \
        TT_SERVER_MAX_MODEL_LEN="$SERVING_MAX_MODEL_LEN" \
        TT_SERVER_GPU_MEM_UTIL="$SERVING_GPU_MEM_UTIL" \
        TT_SERVER_API_BASE="$LLM_SERVER_API_BASE" \
        TT_SERVER_PORT="$LLM_SERVER_PORT" \
        TT_SERVER_GPU_COUNT="${LLM_SERVER_GPU_COUNT:-}" \
        TT_SERVER_GPU_NAME="${LLM_SERVER_GPU_NAME:-}" \
        TT_SERVER_CPUSET="${LLM_SERVER_CPUSET:-}" \
        TT_SERVER_STARTUP_SEC="${LLM_SERVER_STARTUP_SEC:-}" \
        TT_SERVER_STARTED_AT="${LLM_SERVER_STARTED_AT:-}" \
        TT_SERVER_LOG="${LLM_SERVER_LOG:-}" \
        TT_SERVER_HOME="$LLM_SERVER_HOME" \
        TT_SERVER_NODE="$(hostname)"
    TT_SERVER_EXTRA_ARGS=$(printf '%s\n' "${SERVING_EXTRA_ARGS[@]}")
    TT_SERVER_ENV=$(printf '%s\n' "${SERVING_ENV[@]}")
    export TT_SERVER_EXTRA_ARGS TT_SERVER_ENV
}

llm_server_write_info() {
    llm_server_export_info
    "$PYTHON" "$RECIPE_PY" server-info --out "$1"
}

llm_server_stop() {
    local waited=0
    [[ -n $LLM_SERVER_PID ]] || return 0
    if kill -0 "$LLM_SERVER_PID" 2> /dev/null; then
        echo "[server] stopping pid $LLM_SERVER_PID"
        kill -TERM -- "-$LLM_SERVER_PID" 2> /dev/null || kill -TERM "$LLM_SERVER_PID" 2> /dev/null || true
        while kill -0 "$LLM_SERVER_PID" 2> /dev/null && ((waited < 60)); do
            sleep 2
            waited=$((waited + 2))
        done
        if kill -0 "$LLM_SERVER_PID" 2> /dev/null; then
            echo "[server] still running after ${waited}s; killing"
            kill -KILL -- "-$LLM_SERVER_PID" 2> /dev/null || kill -KILL "$LLM_SERVER_PID" 2> /dev/null || true
        fi
    fi
    # `wait` returns the server's exit status; under `set -e` a non-zero one must not abort the cleanup
    if wait "$LLM_SERVER_PID" 2> /dev/null; then
        LLM_SERVER_EXIT=0
    else
        LLM_SERVER_EXIT=$?
    fi
    LLM_SERVER_PID=""
    enroot remove -f "$LLM_SERVER_NAME" > /dev/null 2>&1 || true
    rm -rf "${LLM_SERVER_HOME:?}"
    echo "[server] stopped (exit status $LLM_SERVER_EXIT); log: ${LLM_SERVER_LOG:-none}"
}

# --- Steps shared by the gpu launch scripts ---------------------------------------------------

# gpu_recipe_knobs [timed]: the replay knobs both gpu recipes read, validated. timed is 1 (default) when the
# trajectory to replay carries its recorded timings; 0 when it doesn't, which allows only unpaced: the two
# sides then replay in the recorded loop order instead of on the recorded schedule.
#   RUN_CONF_TASKTRACE_PACING  tool side: recorded | unpaced (default recorded, unpaced without timings)
#   RUN_CONF_LLM_PACING        LLM side, same values (default: the tool pacing)
#   RUN_CONF_TASKTRACE_CPUS    CPUs pinned for the replay; the server gets the rest (default 4)
#   RUN_CONF_LLM_TIMEOUT       per-call client timeout of the LLM replay, seconds (default 1800)
gpu_recipe_knobs() {
    local var timed=${1:-1} default=recorded
    [[ $timed == 1 ]] || default=unpaced
    TRACE_TIMED=$timed
    PACING="${RUN_CONF_TASKTRACE_PACING:-$default}"
    LLM_PACING="${RUN_CONF_LLM_PACING:-$PACING}"
    CPUS="${RUN_CONF_TASKTRACE_CPUS:-4}"
    LLM_TIMEOUT="${RUN_CONF_LLM_TIMEOUT:-1800}"
    for var in PACING LLM_PACING; do
        if [[ ${!var} != "recorded" && ${!var} != "unpaced" ]]; then
            echo "[launch] ERROR: RUN_CONF_TASKTRACE_PACING and RUN_CONF_LLM_PACING must be 'recorded' or 'unpaced' (got '${!var}')." >&2
            return 1
        fi
        if [[ $timed != 1 && ${!var} == "recorded" ]]; then
            echo "[launch] ERROR: the trajectory carries no recorded timings, so it can only replay unpaced (RUN_CONF_TASKTRACE_PACING / RUN_CONF_LLM_PACING is 'recorded')." >&2
            return 1
        fi
    done
    for var in CPUS LLM_TIMEOUT; do
        case "${!var}" in *[!0-9]* | "" | 0)
            echo "[launch] ERROR: RUN_CONF_TASKTRACE_CPUS and RUN_CONF_LLM_TIMEOUT must be positive integers (got '${!var}')." >&2
            return 1
            ;;
        esac
    done
}

# gpu_recipe_cpusets: the replay (and the capture) pin the first CPUS of the allocation, the
# server takes everything after them.
gpu_recipe_cpusets() {
    local allowed
    allowed=$(taskset -cp $$ | awk -F': ' '{print $2}')
    CPUSET=$(echo "$allowed" | "$PYTHON" "$RECIPE_PY" cpuset --count "$CPUS")
    SERVER_CPUSET=$(echo "$allowed" | "$PYTHON" "$RECIPE_PY" cpuset \
        --skip "$CPUS" \
        --all-remaining \
        --min 8)
}

# llm_server_bring_up <server-dir> <trace>: start the server, wait for it, write server.json, warm it up.
llm_server_bring_up() {
    llm_server_start "$SERVER_CPUSET" "$1"
    llm_server_wait_ready
    llm_server_write_info "$1/server.json"
    echo "[launch] warming the server up"
    "$PYTHON" "$RECIPE_PY" llm-replay \
        --trace "$2" \
        --api-base "$LLM_SERVER_API_BASE" \
        --model "$SERVING_SERVED_MODEL_NAME" \
        --timeout-sec "$LLM_TIMEOUT" \
        --warmup-only
}

# gpu_replay_both_sides <trace> <image> <profile-dir> <llm-dir>: the LLM side against the server
# and the tool side in the task container, together, each paced to the recording. The tool side
# needs seconds to create its container, so the LLM side times itself from the tool side's first
# action (the container writes container_profile/start_ns then), offset as recorded.
# Sets REPLAY_EXIT, LLM_EXIT and LLM_REPORT.
gpu_replay_both_sides() {
    local trace=$1 image=$2 profile_dir=$3 llm_dir=$4 timeout_args=() tool_pid lead
    LLM_REPORT="$llm_dir/llm_replay_report.json"
    mkdir -p "$profile_dir" "$llm_dir"
    if [[ -n ${RUN_CONF_TASKTRACE_TIMEOUT:-} ]]; then
        timeout_args=(--timeout-sec "$RUN_CONF_TASKTRACE_TIMEOUT")
    fi
    lead=$("$PYTHON" "$RECIPE_PY" llm-replay --trace "$trace" --lead-only)
    LLM_START_OFFSET=${lead#LLM_START_OFFSET=}
    echo "[launch] llm replay: $LLM_SERVER_API_BASE model $SERVING_SERVED_MODEL_NAME  pacing $LLM_PACING  log $llm_dir/llm-replay.log"
    echo "[launch] tool replay: tasktrace profile trace $trace --image $image --pacing $PACING  cpuset $CPUSET"
    echo "[launch] first llm call ${LLM_START_OFFSET}s after the first tool action, as recorded"
    set +e
    "$TASKTRACE_BIN" profile trace "$trace" \
        --runtime enroot \
        --image "$image" \
        --workdir /testbed \
        --allocated-cpus "$CPUS" \
        --cpuset-cpus "$CPUSET" \
        --pacing "$PACING" \
        --strace-mode none \
        ${timeout_args[@]+"${timeout_args[@]}"} \
        --output-dir "$profile_dir" &
    tool_pid=$!
    "$PYTHON" "$RECIPE_PY" llm-replay \
        --trace "$trace" \
        --api-base "$LLM_SERVER_API_BASE" \
        --model "$SERVING_SERVED_MODEL_NAME" \
        --pacing "$LLM_PACING" \
        --timeout-sec "$LLM_TIMEOUT" \
        --start-delay "$LLM_START_OFFSET" \
        --align-file "$profile_dir/container_profile/start_ns" \
        --align-pid "$tool_pid" \
        --output "$LLM_REPORT" > "$llm_dir/llm-replay.log" 2>&1
    LLM_EXIT=$?
    wait "$tool_pid"
    REPLAY_EXIT=$?
    set -e
    echo "[launch] tasktrace exit status: $REPLAY_EXIT"
    echo "[launch] llm-replay exit status: $LLM_EXIT"
    tail -n 1 "$llm_dir/llm-replay.log" 2> /dev/null | sed 's/^/[launch] /' || true
}

# gpu_replay_loop_order <trace> <image> <profile-dir> <llm-dir>: for a trajectory without recorded timings,
# which has no schedule to pace by. Both sides run together, unpaced, in the order the agent loop recorded:
# each tool action waits for the LLM call it followed, each LLM call for the actions before it. They hand
# off through files in the tool replay's container_profile/loop_order directory (the tool side creates it).
# Sets REPLAY_EXIT, LLM_EXIT and LLM_REPORT.
gpu_replay_loop_order() {
    local trace=$1 image=$2 profile_dir=$3 llm_dir=$4 timeout_args=() tool_pid sync_dir
    LLM_REPORT="$llm_dir/llm_replay_report.json"
    sync_dir="$profile_dir/container_profile/loop_order"
    mkdir -p "$profile_dir" "$llm_dir"
    if [[ -n ${RUN_CONF_TASKTRACE_TIMEOUT:-} ]]; then
        timeout_args=(--timeout-sec "$RUN_CONF_TASKTRACE_TIMEOUT")
    fi
    echo "[launch] the trajectory carries no recorded timings: both sides replay unpaced, in the recorded loop order"
    echo "[launch] llm replay: $LLM_SERVER_API_BASE model $SERVING_SERVED_MODEL_NAME  log $llm_dir/llm-replay.log"
    echo "[launch] tool replay: tasktrace profile trace $trace --image $image --pacing unpaced --loop-order  cpuset $CPUSET"
    set +e
    "$TASKTRACE_BIN" profile trace "$trace" \
        --runtime enroot \
        --image "$image" \
        --workdir /testbed \
        --allocated-cpus "$CPUS" \
        --cpuset-cpus "$CPUSET" \
        --pacing unpaced \
        --loop-order \
        --strace-mode none \
        ${timeout_args[@]+"${timeout_args[@]}"} \
        --output-dir "$profile_dir" &
    tool_pid=$!
    "$PYTHON" "$RECIPE_PY" llm-replay \
        --trace "$trace" \
        --api-base "$LLM_SERVER_API_BASE" \
        --model "$SERVING_SERVED_MODEL_NAME" \
        --pacing unpaced \
        --timeout-sec "$LLM_TIMEOUT" \
        --loop-sync-dir "$sync_dir" \
        --align-pid "$tool_pid" \
        --output "$LLM_REPORT" > "$llm_dir/llm-replay.log" 2>&1
    LLM_EXIT=$?
    # whatever ended the LLM side, the tool side must not wait for it any longer
    [[ -d $sync_dir ]] && : > "$sync_dir/llm_end"
    wait "$tool_pid"
    REPLAY_EXIT=$?
    set -e
    echo "[launch] tasktrace exit status: $REPLAY_EXIT"
    echo "[launch] llm-replay exit status: $LLM_EXIT"
    tail -n 1 "$llm_dir/llm-replay.log" 2> /dev/null | sed 's/^/[launch] /' || true
}

# gpu_replay <trace> <image> <profile-dir> <llm-dir>: both sides of the replay, on the recorded schedule when
# the trajectory carries its timings (TRACE_TIMED=1, set by gpu_recipe_knobs), else in the recorded loop order.
gpu_replay() {
    if [[ ${TRACE_TIMED:-1} == 1 ]]; then
        gpu_replay_both_sides "$@"
    else
        gpu_replay_loop_order "$@"
    fi
}
