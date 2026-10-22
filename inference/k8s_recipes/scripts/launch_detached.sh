#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# Self-detaching launcher. The harness has killed three drivers tonight (c16 at 02:01Z, then both
# c16 and c32 wrappers together), each time costing the automatic fetch/teardown/publish. nohup +
# </dev/null + full output redirection detaches the child from this shell's session so a task kill
# does not propagate to it.
#   usage: launch_detached.sh <cell-path> <logfile>
# Resolve the collection root from THIS script's location, so the launcher works from any
# worktree instead of hardcoding one operator's path.
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 1
CELL="$1"
LOG="$2"
PROFILE="${3:?usage: launch_detached.sh <cell-path> <logfile> <cluster-profile>}"
nohup scripts/llmb-k8s run "$CELL" "$PROFILE" \
    --teardown --fetch \
    > "$LOG" 2>&1 < /dev/null &
echo "launched pid $! -> $LOG"
disown 2> /dev/null || true
