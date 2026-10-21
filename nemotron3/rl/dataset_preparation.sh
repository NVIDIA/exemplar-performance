#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --exclusive
#SBATCH --job-name="nemotron3:rl-datasets"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:25:00

set -eu -o pipefail

if ((BASH_VERSINFO[0] < 4 || (BASH_VERSINFO[0] == 4 && BASH_VERSINFO[1] < 2))); then
    printf 'Requires Bash 4.2 or greater (running: %s).\n' "${BASH_VERSION:-not Bash}" >&2
    exit 1
fi

LLMB_INSTALL=${LLMB_INSTALL:?LLMB_INSTALL must be set}
LLMB_WORKLOAD=${LLMB_WORKLOAD:?LLMB_WORKLOAD must be set}

DATASET_DIR="$LLMB_WORKLOAD/Nemotron-RL-Super-Training-Blends"
LLMB_BIN="${LLMB_BIN:-$LLMB_INSTALL/bin/$(uname -m)}"

if [ -d "$DATASET_DIR/data" ] \
    && compgen -G "$DATASET_DIR/data/*/train-split.jsonl" > /dev/null \
    && compgen -G "$DATASET_DIR/data/*/val-split.jsonl" > /dev/null; then
    echo "Nemotron-RL-Super-Training-Blends dataset already prepared at $DATASET_DIR/data, skipping preprocessing."
    exit 0
fi

cd "$DATASET_DIR"

"$LLMB_BIN/uv" run --no-project \
    --with datasets \
    python fill_placeholders.py \
    --input-dir $DATASET_DIR \
    --output-dir data_filled

for f in data_filled/*.jsonl; do
    name=$(basename "$f" .jsonl)
    mkdir -p "data/$name"
    head -n -100 "$f" > "data/$name/train-split.jsonl"
    tail -n -100 "$f" > "data/$name/val-split.jsonl"
done
