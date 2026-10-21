#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Smoke-test both runtimes end-to-end. Requires:
# - ANTHROPIC_API_KEY env var set (or provided on the command line)
# - For enroot: enroot on PATH
# - For docker: docker daemon reachable
#
# Usage:
#   export ANTHROPIC_API_KEY=sk-...
#   scripts/smoke_test_runtimes.sh [enroot|docker|both]
#
# Default: both. Writes output under ./jobs/.
set -euo pipefail

RUNTIME="${1:-both}"

if [[ -z ${ANTHROPIC_API_KEY:-} ]]; then
    echo "ERROR: set ANTHROPIC_API_KEY" >&2
    exit 1
fi

common=(
    --agent claude-code
    -d hello-world
    --concurrency 1
    -m azure/anthropic/claude-sonnet-4-5
    --agent-env "ANTHROPIC_BASE_URL=https://<your-inference-endpoint>"
    --agent-env "ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}"
    --agent-env "ANTHROPIC_DEFAULT_SONNET_MODEL=azure/anthropic/claude-sonnet-4-5"
)

run_smoke() {
    local runtime="$1"
    echo "================================================"
    echo "  Running --runtime ${runtime} c=1 smoke test"
    echo "================================================"
    uv run tasktrace-capture --runtime "${runtime}" "${common[@]}"
    local rc=$?
    if [[ $rc -ne 0 ]]; then
        echo "FAIL: --runtime ${runtime} exited with ${rc}" >&2
        return $rc
    fi
    echo "OK: --runtime ${runtime} smoke passed"
}

if [[ ${RUNTIME} == "both" || ${RUNTIME} == "enroot" ]]; then
    run_smoke enroot
fi
if [[ ${RUNTIME} == "both" || ${RUNTIME} == "docker" ]]; then
    run_smoke docker
    # Verify no container leak after docker run.
    leaked=$(docker ps -aq --filter "name=hb__" 2> /dev/null | wc -l)
    if [[ $leaked != "0" ]]; then
        echo "WARNING: ${leaked} leaked hb__* containers after docker smoke" >&2
        docker ps -a --filter "name=hb__"
        exit 2
    fi
    echo "OK: no container leak after docker smoke"
fi
