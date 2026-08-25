#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT

# Setup task 5/5 (job_type:local).
#
# Fetch the agentic_coding dataset from HuggingFace into
#   $LLMB_WORKLOAD/dataset/agentic_coding
# which config_template.yaml mounts to /dataset inside the sglang container at
# run time (AIPerf reads it via --input-file /dataset). srtctl install only
# handles model + container, so the dataset is a separate step here.
#
# The dataset is gated: request access at
#   https://huggingface.co/datasets/nv-camilom/agentic_coding
# and export HF_TOKEN (with that access) before running llmb-install.
#
# job_type:local — no SLURM resources needed; this is just a download + unzip
# on the login node (run inside tmux for slow links; see the recipe README).

set -eu -o pipefail

export LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD not set (framework should provide).}
export HF_TOKEN=${HF_TOKEN:?HF_TOKEN not set. Export it (with nv-camilom/agentic_coding access) before running llmb-install, or set it in the environment block of cluster_config.yaml.}

# First copy other required files to the workload dir
cp -f config_template.yaml report.sh extract_metrics.sh $LLMB_WORKLOAD/.

DATASET_REPO="nv-camilom/agentic_coding"
DATASET_ROOT="$LLMB_WORKLOAD/dataset"
DATASET_DIR="$DATASET_ROOT/agentic_coding"

# Idempotent: a populated agentic_coding/ means a previous run already
# downloaded + unzipped. Re-running llmb-install then skips this cheaply.
if [[ -d $DATASET_DIR ]] && [[ -n "$(ls -A "$DATASET_DIR" 2> /dev/null)" ]]; then
    echo "[download_dataset] $DATASET_DIR already populated — skipping download."
    exit 0
fi

mkdir -p "$DATASET_ROOT"

# Resolve a HuggingFace CLI. Match the source benchmark's `uvx hf` approach so
# we don't depend on huggingface_hub being installed in the workload venv;
# prefer a bare `hf` if one is already on PATH.
if command -v hf > /dev/null 2>&1; then
    HF=(hf)
elif command -v uvx > /dev/null 2>&1; then
    HF=(uvx hf)
else
    echo "[download_dataset] ERROR: neither 'hf' nor 'uvx' found on PATH. Install the HuggingFace CLI (or uv) and re-run." >&2
    exit 1
fi

echo "[download_dataset] downloading $DATASET_REPO -> $DATASET_ROOT (via: ${HF[*]})"
"${HF[@]}" download "$DATASET_REPO" --repo-type dataset --local-dir "$DATASET_ROOT/"

# The repo ships a zip; unpack it into $DATASET_ROOT, producing agentic_coding/.
ZIP="$DATASET_ROOT/agentic_coding.zip"
if [[ -f $ZIP ]]; then
    command -v unzip > /dev/null 2>&1 || {
        echo "[download_dataset] ERROR: 'unzip' not found on PATH; cannot unpack $ZIP." >&2
        exit 1
    }
    echo "[download_dataset] unzip $ZIP"
    unzip -o -q "$ZIP" -d "$DATASET_ROOT"
fi

if ! [[ -d $DATASET_DIR ]] || [[ -z "$(ls -A "$DATASET_DIR" 2> /dev/null)" ]]; then
    echo "[download_dataset] ERROR: expected dataset at $DATASET_DIR after download/unzip, but it is missing or empty." >&2
    echo "[download_dataset]        Inspect $DATASET_ROOT and verify the repo layout / your HF access." >&2
    exit 1
fi

echo "[download_dataset] done. dataset ready at $DATASET_DIR"
