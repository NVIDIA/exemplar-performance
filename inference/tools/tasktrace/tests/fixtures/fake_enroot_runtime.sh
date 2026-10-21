#!/bin/sh

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu

# This fixture is invoked through a per-test symlink. Keep the state beside
# that symlink so parallel tests remain isolated while the executable inode is
# stable and never rewritten during the test process.
root=${0%/*}
control="$root/fake-enroot-control"
{
    printf 'CALL'
    for argument in "$@"; do
        printf '\t%s' "$argument"
    done
    printf '\n'
} >> "$control/calls"

case "${1-}" in
    version)
        printf 'enroot version 3.5.0\n'
        ;;
    list)
        if [ -e "$control/created-name" ]; then
            cat "$control/created-name"
        fi
        printf 'named-container RUNNING\n'
        ;;
    create)
        if [ -e "$control/fail-create" ]; then
            printf 'synthetic create failure\n' >&2
            exit 31
        fi
        if [ -e "$control/create-sleep" ]; then
            sleep 30 &
            child="$!"
            printf '%s\n' "$child" > "$control/create-child-pid"
            : > "$control/create-entered"
            wait "$child"
            exit "$?"
        fi
        printf '%s\n' "$3" > "$control/created-name"
        ;;
    start)
        if [ -e "$control/start-background-child" ]; then
            sleep 30 &
            child="$!"
            printf '%s\n' "$child" > "$control/start-child-pid"
            : > "$control/start-entered"
            exit 0
        fi
        if [ -e "$control/start-stubborn-background-child" ]; then
            /bin/sh -c 'trap "" HUP TERM; while :; do sleep 1; done' &
            child="$!"
            printf '%s\n' "$child" > "$control/start-child-pid"
            : > "$control/start-entered"
            exit 0
        fi
        if [ -e "$control/start-stubborn-child" ]; then
            trap 'exit 143' TERM
            /bin/sh -c 'trap "" TERM; while :; do sleep 1; done' &
            child="$!"
            printf '%s\n' "$child" > "$control/start-child-pid"
            : > "$control/start-entered"
            wait "$child"
            exit "$?"
        fi
        if [ -e "$control/start-sleep" ]; then
            sleep 30 &
            child="$!"
            printf '%s\n' "$child" > "$control/start-child-pid"
            : > "$control/start-entered"
            wait "$child"
            exit "$?"
        fi
        if [ -e "$control/start-exit" ]; then
            exit "$(cat "$control/start-exit")"
        fi
        ;;
    remove)
        if [ -e "$control/fail-remove" ]; then
            printf 'synthetic remove failure\n' >&2
            exit 41
        fi
        rm -f "$control/created-name"
        ;;
    *)
        printf 'unexpected fake enroot operation: %s\n' "${1-}" >&2
        exit 90
        ;;
esac
