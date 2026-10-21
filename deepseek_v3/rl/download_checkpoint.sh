#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

#SBATCH --exclusive
#SBATCH --job-name="deepseek_v3:rl-checkpoint-conversion"
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

FP8_DIR="$LLMB_WORKLOAD/DeepSeek-V3-FP8"
BF16_DIR="$LLMB_WORKLOAD/DeepSeek-V3-BF16"
TMP_BF16_DIR="$BF16_DIR.tmp"
DEEPSEEK_REPO="$LLMB_WORKLOAD/DeepSeek-V3"
CONVERSION_SCRIPT="$DEEPSEEK_REPO/inference/fp8_cast_bf16.py"
LLMB_BIN="${LLMB_BIN:-$LLMB_INSTALL/bin/$(uname -m)}"

if [ -d "$BF16_DIR" ] && [ "$(ls -A "$BF16_DIR" 2> /dev/null)" ]; then
    echo "DeepSeek-V3 BF16 checkpoint already exists at $BF16_DIR, skipping conversion."
    exit 0
fi

rm -rf "$BF16_DIR" "$TMP_BF16_DIR"
sed -i 's/save_file(new_state_dict, new_safetensor_file)/save_file(new_state_dict, new_safetensor_file, metadata={"format": "pt"})/' "$CONVERSION_SCRIPT"

pushd "$DEEPSEEK_REPO/inference" > /dev/null

"$LLMB_BIN/uv" run --no-project \
    --with torch \
    --with safetensors \
    --with numpy \
    --with tqdm \
    python fp8_cast_bf16.py \
    --input-fp8-hf-path "$FP8_DIR" \
    --output-bf16-hf-path "$TMP_BF16_DIR"

popd > /dev/null

cp "$FP8_DIR"/tokenizer_config.json "$FP8_DIR"/tokenizer.json "$FP8_DIR"/modeling_deepseek.py "$FP8_DIR"/configuration_deepseek.py "$TMP_BF16_DIR"/

# Can't assume jq availability
"$LLMB_BIN/uv" run --no-project python -c '
import json
import sys

with open(sys.argv[1], "r") as f:
    config = json.load(f)
config.pop("quantization_config", None)
config["num_nextn_predict_layers"] = 0
with open(sys.argv[2], "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")
' "$FP8_DIR/config.json" "$TMP_BF16_DIR/config.json"

mv "$TMP_BF16_DIR" "$BF16_DIR"

echo "DeepSeek-V3 BF16 checkpoint conversion complete at $BF16_DIR"
