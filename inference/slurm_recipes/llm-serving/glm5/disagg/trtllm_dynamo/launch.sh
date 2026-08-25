#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# GLM5 inference dispatcher — thin pass-through to `srtctl apply -f`.
#
# Framework contract — ConfiguredSbatchLauncher injects these:
#   $LLMB_INSTALL          installation dir for all workloads
#   $LLMB_WORKLOAD         $LLMB_INSTALL/workloads/inference_glm5
#   $LLMB_EXPERIMENT_DIR   per-run output dir, framework pre-creates
#   $GPU_TYPE              host SKU
#
# Dispatcher only kicks the benchmark off — actual sweep is sbatched by
# srtctl apply across multiple GPU nodes. The dispatcher exits as soon
# as srtctl returns (fire-and-forget).
#
# Overrides:
#   RUN_CONF_GLM5_RECIPE   path relative to
#                          CrossCluster_Recipes/GLM5/Disagg/trtllm_dynamo/
#                          (default: 8k_1k/ctx4dep2_gen3tep8_batch64_eplb0_mtp0_conc_64_128_256.yaml)

#SBATCH --time=00:30:00
# NOTE: --job-name and --output are set by ConfiguredSbatchLauncher.
# Matching #SBATCH headers here are silently overridden — kept minimal
# on purpose.

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
    exit 1
fi

export LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL not set (framework should provide).}

# ConfiguredSbatchLauncher only injects LLMB_INSTALL (not LLMB_WORKLOAD).
# Derive LLMB_WORKLOAD locally — matches the convention used by
# legacy LLMB inference recipes.
export WORKLOAD_KEY="inference_glm5"
export LLMB_WORKLOAD="$LLMB_INSTALL/workloads/$WORKLOAD_KEY"

SRT_SLURM_DIR="$LLMB_WORKLOAD/srt-slurm"
RECIPE_REL="${RUN_CONF_GLM5_RECIPE:-8k_1k/ctx4dep2_gen3tep8_batch64_eplb0_mtp0_conc_64_128_256.yaml}"
RECIPE_PATH="$SRT_SLURM_DIR/CrossCluster_Recipes/GLM5/Disagg/trtllm_dynamo/$RECIPE_REL"

[[ -f $RECIPE_PATH ]] || {
    echo "[launch] ERROR: recipe not found at $RECIPE_PATH." >&2
    echo "[launch] Set RUN_CONF_GLM5_RECIPE to a path under CrossCluster_Recipes/GLM5/Disagg/trtllm_dynamo/." >&2
    exit 1
}

COMPUTE_VENV="$SRT_SLURM_DIR/.venv"
[[ -x "$COMPUTE_VENV/bin/srtctl" ]] || {
    echo "[launch] ERROR: compute venv missing/broken at $COMPUTE_VENV. Re-run llmb-install for inference_glm5." >&2
    exit 1
}

# Scrub login-node env pollution before calling srtctl. The dispatcher was
# sbatched by llmb-run, which inherited $PATH + $VIRTUAL_ENV from the LLMB
# bootstrap venv (x86_64 login node). If we carry that env into srtctl apply,
# the benchmark sbatch it generates propagates the same x86 paths to compute
# nodes — and the sweep orchestrator's `uv` finds the x86 python3 first,
# yielding "Exec format error". Surgically remove just $VIRTUAL_ENV/bin from
# PATH so /sbin, /opt/*, and module-loaded dirs survive. We call srtctl via
# absolute path; its shebang resolves the aarch64 python on its own.
if [[ -n ${VIRTUAL_ENV:-} ]]; then
    PATH=$(echo ":${PATH}:" | sed "s|:${VIRTUAL_ENV}/bin:|:|g; s|^:||; s|:$||")
fi
unset VIRTUAL_ENV

# srtctl apply must run from the srt-slurm root: srtslurm.yaml is found by
# walking cwd ↑ ≤2 dirs, and recipe YAMLs reference paths relative to root.
cd "$SRT_SLURM_DIR"

echo "[launch] srtctl apply -f $RECIPE_PATH"
echo "[launch] benchmark outputs will land under: $SRT_SLURM_DIR/outputs/"
"$COMPUTE_VENV/bin/srtctl" apply -f "$RECIPE_PATH"

echo "[launch] dispatcher done. Monitor real benchmark jobs via: squeue --me"
