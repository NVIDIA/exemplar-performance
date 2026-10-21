#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# test_github_connectivity.sh — Diagnose github.com connectivity from HPC nodes
#
# Usage:
#   bash test_github_connectivity.sh              # run on current node
#   srun --partition=<part> bash test_github_connectivity.sh  # run on compute node
#
# Background: Terminal-Bench verifiers download uv/pytest from github.com at
# runtime. Nodes with restricted outbound access fail silently, producing
# reward=0 that looks like an agent failure but is actually infrastructure.

set -u

TARGETS="github.com objects.githubusercontent.com astral.sh pypi.org deb.debian.org"
N_ATTEMPTS=10
TIMEOUT_SEC=5

echo "=========================================="
echo "  GitHub/PyPI Connectivity Test"
echo "=========================================="
echo "  Hostname:  $(hostname)"
echo "  Date:      $(date -u)"
echo "  Node:      ${SLURM_JOB_NODELIST:-not in SLURM}"
echo "  Job ID:    ${SLURM_JOB_ID:-N/A}"
echo ""

for target in $TARGETS; do
    echo "--- $target ---"
    pass=0
    fail=0
    total_ms=0
    for i in $(seq 1 $N_ATTEMPTS); do
        start=$(date +%s%N)
        code=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout $TIMEOUT_SEC --max-time $((TIMEOUT_SEC * 2)) "https://$target" 2> /dev/null)
        end=$(date +%s%N)
        ms=$(((end - start) / 1000000))
        if [ "$code" = "000" ]; then
            echo "  [$i/$N_ATTEMPTS] TIMEOUT (${ms}ms)"
            fail=$((fail + 1))
        else
            echo "  [$i/$N_ATTEMPTS] HTTP $code (${ms}ms)"
            pass=$((pass + 1))
            total_ms=$((total_ms + ms))
        fi
    done
    avg_ms=0
    [ $pass -gt 0 ] && avg_ms=$((total_ms / pass))
    pct=$((fail * 100 / N_ATTEMPTS))
    echo "  Result: $pass/$N_ATTEMPTS passed, ${pct}% failure rate, avg latency ${avg_ms}ms"
    echo ""
done

# DNS resolution test
echo "--- DNS Resolution ---"
for target in github.com objects.githubusercontent.com; do
    ips=$(dig +short "$target" 2> /dev/null | head -3 | tr '\n' ' ')
    echo "  $target -> ${ips:-FAILED}"
done
echo ""

# Traceroute (first 10 hops)
echo "--- Traceroute to github.com (first 10 hops) ---"
traceroute -m 10 -w 2 github.com 2> /dev/null || echo "  traceroute not available"
echo ""

# Check proxy settings
echo "--- Proxy Configuration ---"
echo "  http_proxy:  ${http_proxy:-not set}"
echo "  https_proxy: ${https_proxy:-not set}"
echo "  no_proxy:    ${no_proxy:-not set}"
echo ""

# Summary
echo "=========================================="
echo "  If github.com shows >10% failure rate, this node has restricted"
echo "  outbound internet. Terminal-Bench verifiers will fail intermittently."
echo "  Contact cluster admins to allow outbound HTTPS to:"
echo "    - github.com / objects.githubusercontent.com (uv binary download)"
echo "    - astral.sh (uv installer redirect)"
echo "    - pypi.org (pytest and dependencies)"
echo "=========================================="
