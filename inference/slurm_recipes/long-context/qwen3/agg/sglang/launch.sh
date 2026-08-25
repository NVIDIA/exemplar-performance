#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Qwen3 long-context inference dispatcher — renders this folder's
# config_template.yaml once per session concurrency and `srtctl apply`s each.
#
# Framework contract — ConfiguredSbatchLauncher injects these:
#   $LLMB_INSTALL          installation dir for all workloads
#   $LLMB_WORKLOAD         $LLMB_INSTALL/workloads/inference_qwen3_long
#   $LLMB_EXPERIMENT_DIR   per-run output dir, framework pre-creates
#   $GPU_TYPE              host SKU
#
# The dispatcher renders + submits each concurrency and exits as soon as the
# last `srtctl apply` returns — actual sweep jobs run on multiple GPU nodes
# (fire-and-forget).
#
# Overrides:
#   RUN_CONF_AGENTIC_SESSION_CONCURRENCIES  space-separated concurrency list
#                                       (default: "4 8 16 24 32 48 64")

#SBATCH --time=01:00:00
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
# Derive LLMB_WORKLOAD locally — matches the convention used by the other
# inference recipes.
export WORKLOAD_KEY="inference_qwen3_long"
export LLMB_WORKLOAD="$LLMB_INSTALL/workloads/$WORKLOAD_KEY"

SRT_SLURM_DIR="$LLMB_WORKLOAD/srt-slurm"

TEMPLATE="$LLMB_WORKLOAD/config_template.yaml"
[[ -f $TEMPLATE ]] || {
    echo "[launch] ERROR: config_template.yaml not found at $TEMPLATE." >&2
    exit 1
}

# Resolves the ${DATASET_DIR} mount in config_template.yaml. Populated by the
# download_dataset setup task during llmb-install.
export DATASET_DIR="$LLMB_WORKLOAD/dataset/agentic_coding"
[[ -d $DATASET_DIR ]] || {
    echo "[launch] ERROR: dataset missing at $DATASET_DIR. Re-run llmb-install for $WORKLOAD_KEY (download_dataset task)." >&2
    exit 1
}

COMPUTE_VENV="$SRT_SLURM_DIR/.venv"
[[ -x "$COMPUTE_VENV/bin/srtctl" ]] || {
    echo "[launch] ERROR: compute venv missing/broken at $COMPUTE_VENV. Re-run llmb-install for $WORKLOAD_KEY." >&2
    exit 1
}

command -v envsubst > /dev/null 2>&1 || {
    echo "[launch] ERROR: envsubst not found on PATH (provided by gettext). Install it or load the module on this node." >&2
    exit 1
}

CONCURRENCIES="${RUN_CONF_AGENTIC_SESSION_CONCURRENCIES:-4 8 16 24 32 48 64}"

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
# walking cwd ↑ ≤2 dirs, and rendered configs live here.
cd "$SRT_SLURM_DIR"

echo "[launch] sweeping concurrencies: $CONCURRENCIES"
echo "[launch] benchmark outputs will land under: $SRT_SLURM_DIR/outputs/"
for SESSION_CONCURRENCY in $CONCURRENCIES; do
    export SESSION_CONCURRENCY
    CONF="$SRT_SLURM_DIR/config_${SESSION_CONCURRENCY}.yaml"
    # we want to be specific with envsubst about which variables
    # shellcheck disable=SC2016
    envsubst '$SESSION_CONCURRENCY $DATASET_DIR' < "$TEMPLATE" > "$CONF"
    echo "[launch] srtctl apply -f $CONF (concurrency=$SESSION_CONCURRENCY)"
    "$COMPUTE_VENV/bin/srtctl" apply -f "$CONF"
done

echo "[launch] dispatcher done. Monitor real benchmark jobs via: squeue --me"
echo "[launch] view results once jobs finish with: $(dirname "$(readlink -f "$0")")/report.sh"
