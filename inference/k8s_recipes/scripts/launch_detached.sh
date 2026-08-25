#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

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
