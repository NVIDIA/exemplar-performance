#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Helpers shared by the agentic install scripts (source this file; it defines
# functions only). TT_TAG prefixes messages, e.g. "[install_tasktrace_replay]".
# bash-regression:skip-guard -- sourced library; sourcing scripts own the guard.

TT_TAG="${TT_TAG:-[tasktrace-install]}"

tt_require_bash() {
    if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
        printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
        exit 1
    fi
}

# Sets COMMON_DIR (this directory) and REPO_ROOT (four levels up: the scripts
# live at inference/slurm_recipes/agentic/common/).
tt_locate_repo() {
    COMMON_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    # shellcheck disable=SC2034 # set for the sourcing script
    REPO_ROOT="$(cd "$COMMON_DIR/../../../.." && pwd)"
}

# tt_source_id <dir> <build-subdir-to-skip>: "tree:<sha>" of the tracked files
# plus local modifications when <dir> is inside a git checkout, else
# "dir:<sha>" of the files themselves (an llmb_repo copy has no .git).
tt_source_id() {
    local dir=$1 skip=$2
    if git -C "$dir" rev-parse --is-inside-work-tree > /dev/null 2>&1; then
        echo "tree:$( (git -C "$dir" ls-files -s . && git -C "$dir" diff HEAD -- .) | sha256sum | awk '{print $1}')"
    else
        echo "dir:$(find "$dir" -type f -not -path "*/$skip/*" -print0 | sort -z | xargs -0 sha256sum | sha256sum | awk '{print $1}')"
    fi
}

tt_source_commit() {
    git -C "$1" rev-parse HEAD 2> /dev/null || echo unknown
}

# tt_version_matches <version-file> key=value...: true when every pair is recorded.
tt_version_matches() {
    local file=$1 pair
    [[ -f $file ]] || return 1
    shift
    for pair in "$@"; do
        grep -qx "$pair" "$file" || return 1
    done
}

# tt_fetch_script <url> <dest> <label>: download an installer and refuse
# anything that is not a shell script (a redirect page, an error body).
tt_fetch_script() {
    local url=$1 dest=$2 label=$3
    # -L: some hosts (astral.sh) answer with a 301, and -f does not treat 3xx
    # as failure; without -L the redirect page is saved and 'sh' fails on it.
    curl -LsSf "$url" -o "$dest"
    head -c 2 "$dest" | grep -q '#!' || {
        echo "$TT_TAG $label installer download is not a shell script:" >&2
        head -c 300 "$dest" >&2
        echo >&2
        exit 1
    }
}

# tt_run_installer <label> <log> <script> [args...]: run it quietly, show the
# log only on failure.
tt_run_installer() {
    local label=$1 log=$2 script=$3
    shift 3
    sh "$script" "$@" > "$log" 2>&1 || {
        echo "$TT_TAG $label installer failed; log follows ($log):" >&2
        cat "$log" >&2
        exit 1
    }
}
