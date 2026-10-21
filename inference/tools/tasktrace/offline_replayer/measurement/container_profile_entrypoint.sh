#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Run inside a dedicated profiling container. The host-side native controller
# supplies configuration through TASKTRACE_PROFILE_* environment variables.
# Keep the timing boundary here, inside the target cgroup: host-side Docker
# startup, image setup, and artifact transfer must not inflate task CPU usage.
set -eu

: "${TASKTRACE_PROFILE_DIR:?TASKTRACE_PROFILE_DIR is required}"
: "${TASKTRACE_PROFILE_COMMAND:?TASKTRACE_PROFILE_COMMAND is required}"

mkdir -p "$TASKTRACE_PROFILE_DIR"

mode="${TASKTRACE_STRACE_MODE:-execve}"
string_limit="${TASKTRACE_STRACE_STRING_LIMIT:-4096}"
command_shell="${TASKTRACE_PROFILE_SHELL:-/bin/bash}"
prepare_command="${TASKTRACE_PROFILE_PREPARE_COMMAND:-}"
verification_command="${TASKTRACE_PROFILE_VERIFICATION_COMMAND:-}"
scheduler_gate_dir="${TASKTRACE_SCHEDULER_GATE_DIR:-}"
resource_gate_dir="${TASKTRACE_RESOURCE_GATE_DIR:-}"
scheduler_tracing="${TASKTRACE_SCHEDULER_TRACING:-}"
skip_cpu_stats="${TASKTRACE_PROFILE_SKIP_CPU_STATS:-}"
run_wave_start_gate="${TASKTRACE_PROFILE_RUN_WAVE_START_GATE:-}"

timestamp_ns() {
    timestamp="$(date +%s%N 2> /dev/null || true)"
    case "$timestamp" in
        "" | *[!0-9]*) timestamp="" ;;
    esac
    if [ "${#timestamp}" -ge 18 ]; then
        printf '%s\n' "$timestamp"
    elif [ -r /proc/uptime ]; then
        awk '{ split($1, parts, "."); fraction = parts[2] "000000000"; print parts[1] substr(fraction, 1, 9); exit }' /proc/uptime
    else
        printf '%s000000000\n' "$(date +%s)"
    fi
}

monotonic_timestamp_ns() {
    if [ ! -r /proc/uptime ]; then
        printf 'monotonic clock source /proc/uptime is unavailable\n' >&2
        return 1
    fi
    IFS=' ' read -r uptime _ < /proc/uptime
    seconds="${uptime%%.*}"
    fraction="${uptime#*.}"
    if [ "$fraction" = "$uptime" ]; then
        fraction=""
    fi
    case "$seconds$fraction" in
        "" | *[!0-9]*)
            printf 'invalid monotonic clock value in /proc/uptime\n' >&2
            return 1
            ;;
    esac
    while [ "${#fraction}" -lt 9 ]; do
        fraction="${fraction}0"
    done
    printf '%s%.9s\n' "$seconds" "$fraction"
}

task_timestamp_ns() {
    if [ -n "$skip_cpu_stats" ]; then
        monotonic_timestamp_ns
    else
        timestamp_ns
    fi
}

write_boottime_clock_evidence() {
    IFS=' ' read -r uptime _ < /proc/uptime
    fraction="${uptime#*.}"
    if [ "$fraction" = "$uptime" ]; then
        fraction=""
    fi
    case "${#fraction}" in
        0) resolution_ns=1000000000 ;;
        1) resolution_ns=100000000 ;;
        2) resolution_ns=10000000 ;;
        3) resolution_ns=1000000 ;;
        4) resolution_ns=100000 ;;
        5) resolution_ns=10000 ;;
        6) resolution_ns=1000 ;;
        7) resolution_ns=100 ;;
        8) resolution_ns=10 ;;
        *) resolution_ns=1 ;;
    esac
    printf 'CLOCK_BOOTTIME\n' > "$TASKTRACE_PROFILE_DIR/clock_name"
    printf '%s\n' "$resolution_ns" > "$TASKTRACE_PROFILE_DIR/clock_resolution_ns"
    if [ -r /proc/sys/kernel/random/boot_id ]; then
        IFS= read -r boot_id < /proc/sys/kernel/random/boot_id
        printf '%s\n' "$boot_id" > "$TASKTRACE_PROFILE_DIR/clock_boot_id"
    else
        : > "$TASKTRACE_PROFILE_DIR/clock_boot_id"
    fi
    time_namespace="$(readlink /proc/self/ns/time 2> /dev/null || true)"
    printf '%s\n' "$time_namespace" > "$TASKTRACE_PROFILE_DIR/clock_time_namespace"
    : > "$TASKTRACE_PROFILE_DIR/clock_time_namespace_boottime_offset"
    if [ -r /proc/self/timens_offsets ]; then
        while IFS=' ' read -r clock_name offset_seconds offset_nanoseconds; do
            if [ "$clock_name" = "boottime" ]; then
                printf '%s %s\n' "$offset_seconds" "$offset_nanoseconds" \
                    > "$TASKTRACE_PROFILE_DIR/clock_time_namespace_boottime_offset"
                break
            fi
        done < /proc/self/timens_offsets
    fi
}

if [ -n "$skip_cpu_stats" ]; then
    printf 'linux-proc-uptime\n' > "$TASKTRACE_PROFILE_DIR/clock_source"
    write_boottime_clock_evidence
    awk '
        /^Cpus_allowed_list:/ { print $2; found = 1; exit }
        END { if (!found) exit 1 }
    ' /proc/self/status > "$TASKTRACE_PROFILE_DIR/initial_effective_cpu_affinity"
fi

case "$mode" in
    execve)
        strace_filter="execve"
        ;;
    process)
        # Keep -f with one shared -o file below: strace then prefixes every
        # record with the numeric traced task ID consumed by process_exec.py.
        # CWD syscalls are required to resolve relative executable paths.
        strace_filter="%process,chdir,fchdir"
        ;;
    none)
        strace_filter=""
        ;;
    *)
        printf 'unsupported TASKTRACE_STRACE_MODE: %s\n' "$mode" >&2
        exit 2
        ;;
esac

# Memory-capacity profiling needs an earlier lifecycle boundary than the
# command-only CPU gate below. Block before preparation so the host observer
# can follow setup memory and CPU on the same monotonic timeline. The fresh
# cgroup's memory.peak also retains any startup peak before this handshake.
if [ -n "$resource_gate_dir" ]; then
    sandbox_start_gate="$resource_gate_dir/sandbox.start.gate"
    rm -f "$sandbox_start_gate"
    mkfifo "$sandbox_start_gate"
    chmod a+rw "$sandbox_start_gate"
    printf 'ready\n' > "$TASKTRACE_PROFILE_DIR/sandbox_attach_ready"
    IFS= read -r _ < "$sandbox_start_gate"
fi

setup_status=0
setup_error=""
if [ ! -x "$command_shell" ]; then
    setup_status=127
    setup_error="command shell is not executable in target container: $command_shell"
    printf '%s\n' "$setup_error" >&2
fi
if [ -n "$prepare_command" ] && [ "$setup_status" -eq 0 ]; then
    # Packages such as strace may not belong in the original benchmark image.
    # Install or prepare them before snapshots so instrumentation setup is not
    # charged to the replayed task.
    set +e
    "$command_shell" -c "$prepare_command" \
        > "$TASKTRACE_PROFILE_DIR/prepare.stdout.txt" \
        2> "$TASKTRACE_PROFILE_DIR/prepare.stderr.txt"
    prepare_status=$?
    set -e
    printf '%s\n' "$prepare_status" > "$TASKTRACE_PROFILE_DIR/prepare_exit_code"
    if [ "$prepare_status" -ne 0 ]; then
        setup_status=$prepare_status
        setup_error="prepare command failed with exit code $prepare_status"
    fi
fi
if [ "$setup_status" -eq 0 ] && [ "$mode" != "none" ] && ! command -v strace > /dev/null 2>&1; then
    setup_status=127
    setup_error="strace is not installed in the target container"
    printf '%s\n' "$setup_error" >&2
fi
if [ "$setup_status" -ne 0 ] && [ -z "$resource_gate_dir" ]; then
    # Preserve the historical fast-fail behavior when no early resource
    # observer is active. Memory profiles continue through the terminal gates
    # so OOM/setup evidence and invalidity metadata can be finalized safely.
    exit "$setup_status"
fi

# In host-coordinated mode, wait blocked until the host has attached an
# optional scheduler tracer and/or collected a /proc/stat baseline. The host
# reads cpu.stat while the container is blocked so its cgroup and observer
# windows share the same command boundary. Normal mode keeps snapshots local.
if [ -n "$scheduler_gate_dir" ]; then
    start_gate="$scheduler_gate_dir/scheduler.start.gate"
    finish_gate="$scheduler_gate_dir/scheduler.finish.gate"
    rm -f "$start_gate" "$finish_gate"
    mkfifo "$start_gate" "$finish_gate"
    chmod a+rw "$start_gate" "$finish_gate"
    printf 'ready\n' > "$TASKTRACE_PROFILE_DIR/scheduler_attach_ready"
    IFS= read -r _ < "$start_gate"
elif [ -z "$skip_cpu_stats" ]; then
    # Snapshot the disposable container around the invoked command. This
    # captures end-to-end command CPU, including small boundary costs. Skipped
    # under rootless enroot where /sys/fs/cgroup/cpu.stat is not exposed.
    cat /sys/fs/cgroup/cpu.stat > "$TASKTRACE_PROFILE_DIR/cpu.stat.before"
fi

# Ordered Enroot waves use a controller-created release marker to exclude
# cold image extraction and preparation from launch alignment. Publish
# readiness atomically only after setup is complete, then observe release on
# the same task CLOCK_BOOTTIME source used by start_ns below.
if [ -n "$run_wave_start_gate" ]; then
    if [ "$run_wave_start_gate" != "1" ]; then
        printf 'invalid TASKTRACE_PROFILE_RUN_WAVE_START_GATE: %s\n' \
            "$run_wave_start_gate" >&2
        exit 2
    fi
    wave_ready="$TASKTRACE_PROFILE_DIR/run_wave_start_ready"
    wave_ready_tmp="$TASKTRACE_PROFILE_DIR/.run_wave_start_ready.$$"
    wave_release="$TASKTRACE_PROFILE_DIR/run_wave_start_release"
    if [ -e "$wave_ready" ] || [ -L "$wave_ready" ]; then
        printf 'run-wave start ready marker already exists\n' >&2
        exit 2
    fi
    rm -f "$wave_ready_tmp"
    if ! (
        umask 077
        set -C
        printf 'ready\n' > "$wave_ready_tmp"
    ); then
        printf 'could not create private run-wave start ready marker\n' >&2
        exit 2
    fi
    chmod 600 "$wave_ready_tmp"
    mv "$wave_ready_tmp" "$wave_ready"
    while :; do
        if [ -L "$wave_release" ]; then
            printf 'run-wave start release marker must not be a symlink\n' >&2
            exit 2
        fi
        if [ -f "$wave_release" ]; then
            release_size="$(wc -c < "$wave_release" | tr -d '[:space:]')"
            if [ "$release_size" = "8" ]; then
                release_value=""
                IFS= read -r release_value < "$wave_release" || true
                if [ "$release_value" = "release" ]; then
                    break
                fi
                printf 'invalid run-wave start release marker\n' >&2
                exit 2
            fi
            case "$release_size" in
                0 | 1 | 2 | 3 | 4 | 5 | 6 | 7) ;;
                *)
                    printf 'invalid run-wave start release marker length\n' >&2
                    exit 2
                    ;;
            esac
        fi
        sleep 0.01
    done
    task_timestamp_ns > "$TASKTRACE_PROFILE_DIR/run_wave_start_released_ns"
fi
if [ -n "$scheduler_gate_dir" ] && [ -n "$scheduler_tracing" ]; then
    # The host overwrites these placeholders with monotonic boundaries after
    # the finish gate. Avoid spawning timestamp helpers in traced windows.
    printf '0\n' > "$TASKTRACE_PROFILE_DIR/start_ns"
    chmod a+rw "$TASKTRACE_PROFILE_DIR/start_ns"
else
    if [ -n "$scheduler_gate_dir" ] && [ -z "$skip_cpu_stats" ]; then
        cat /sys/fs/cgroup/cpu.stat > "$TASKTRACE_PROFILE_DIR/cpu.stat.before"
    fi
    task_timestamp_ns > "$TASKTRACE_PROFILE_DIR/start_ns"
fi

set +e
# Use a non-login shell to avoid profile scripts introducing unrelated
# subprocesses into both the CPU accounting and process-launch trace.
if [ "$setup_status" -ne 0 ]; then
    : > "$TASKTRACE_PROFILE_DIR/command.stdout.txt"
    printf '%s\n' "$setup_error" > "$TASKTRACE_PROFILE_DIR/command.stderr.txt"
    status=$setup_status
elif [ "$mode" = "none" ]; then
    "$command_shell" -c "$TASKTRACE_PROFILE_COMMAND" \
        > "$TASKTRACE_PROFILE_DIR/command.stdout.txt" \
        2> "$TASKTRACE_PROFILE_DIR/command.stderr.txt"
    status=$?
else
    strace -f -ttt -s "$string_limit" \
        -e "trace=$strace_filter" \
        -o "$TASKTRACE_PROFILE_DIR/process.strace.log" \
        "$command_shell" -c "$TASKTRACE_PROFILE_COMMAND" \
        > "$TASKTRACE_PROFILE_DIR/command.stdout.txt" \
        2> "$TASKTRACE_PROFILE_DIR/command.stderr.txt"
    status=$?
fi
set -e

if [ -n "$scheduler_gate_dir" ]; then
    if [ -n "$scheduler_tracing" ]; then
        printf '0\n' > "$TASKTRACE_PROFILE_DIR/end_ns"
        chmod a+rw "$TASKTRACE_PROFILE_DIR/end_ns"
    else
        task_timestamp_ns > "$TASKTRACE_PROFILE_DIR/end_ns"
        if [ -z "$skip_cpu_stats" ]; then
            cat /sys/fs/cgroup/cpu.stat > "$TASKTRACE_PROFILE_DIR/cpu.stat.after"
        fi
    fi
    # Emit all report artifacts before the controlled finish gate so the final
    # cgroup peak includes their in-container work. Only init unblocking and
    # exit remain outside the bounded observation.
    printf '%s\n' "$status" > "$TASKTRACE_PROFILE_DIR/exit_code"
    printf '%s\n' "$mode" > "$TASKTRACE_PROFILE_DIR/strace_mode"
    printf 'done\n' > "$TASKTRACE_PROFILE_DIR/scheduler_measurement_done"
    IFS= read -r _ < "$finish_gate"
else
    task_timestamp_ns > "$TASKTRACE_PROFILE_DIR/end_ns"
    if [ -z "$skip_cpu_stats" ]; then
        cat /sys/fs/cgroup/cpu.stat > "$TASKTRACE_PROFILE_DIR/cpu.stat.after"
    fi
    printf '%s\n' "$status" > "$TASKTRACE_PROFILE_DIR/exit_code"
    printf '%s\n' "$mode" > "$TASKTRACE_PROFILE_DIR/strace_mode"
fi

# Verification deliberately begins only after the host releases the controlled
# finish gate. By then every profiler has stopped and captured its terminal
# counters, so verifier CPU, memory, processes, and PMU activity cannot inflate
# the measured workload window. The verifier still runs in this same sandbox
# and can inspect filesystem side effects before Docker removes the container.
if [ -n "$verification_command" ]; then
    set +e
    "$command_shell" -c "$verification_command" \
        > "$TASKTRACE_PROFILE_DIR/verification.stdout.txt" \
        2> "$TASKTRACE_PROFILE_DIR/verification.stderr.txt"
    verification_exit_code=$?
    set -e
    printf '%s\n' "$verification_exit_code" \
        > "$TASKTRACE_PROFILE_DIR/verification_exit_code"
    if [ "$verification_exit_code" -eq 0 ]; then
        printf 'passed\n' > "$TASKTRACE_PROFILE_DIR/verification_status"
    else
        printf 'failed\n' > "$TASKTRACE_PROFILE_DIR/verification_status"
    fi
fi

exit "$status"
