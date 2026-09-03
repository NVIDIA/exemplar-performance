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

#SBATCH --exclusive
#SBATCH --job-name="deepseek_v3:rl-checkpoint-conversion"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=00:25:00

set -eu -o pipefail

if [ "${BASH_VERSINFO[0]}" -lt 4 ] || { [ "${BASH_VERSINFO[0]}" -eq 4 ] && [ "${BASH_VERSINFO[1]}" -lt 2 ]; }; then
    printf "Unsupported %s version: %s\n" "${BASH}" "${BASH_VERSION}" >&2
    echo "Requires Bash 4.2 or greater." >&2
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

# Honour the installer's LLMB_DISABLE_MANAGED_PYTHON opt-out. Passing
# --no-managed-python is required to actually get system python; omitting the
# flag leaves uv's default preference, which still favours managed python.
UV_MANAGED_PYTHON_FLAG=--managed-python
case "${LLMB_DISABLE_MANAGED_PYTHON:-}" in
    1 | [Tt][Rr][Uu][Ee] | [Yy][Ee][Ss]) UV_MANAGED_PYTHON_FLAG=--no-managed-python ;;
esac

if [ -d "$BF16_DIR" ] && [ "$(ls -A "$BF16_DIR" 2> /dev/null)" ]; then
    echo "DeepSeek-V3 BF16 checkpoint already exists at $BF16_DIR, skipping conversion."
    exit 0
fi

rm -rf "$BF16_DIR" "$TMP_BF16_DIR"
sed -i 's/save_file(new_state_dict, new_safetensor_file)/save_file(new_state_dict, new_safetensor_file, metadata={"format": "pt"})/' "$CONVERSION_SCRIPT"

pushd "$DEEPSEEK_REPO/inference" > /dev/null

"$LLMB_BIN/uv" run --no-project \
    "$UV_MANAGED_PYTHON_FLAG" \
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
"$LLMB_BIN/uv" run --no-project "$UV_MANAGED_PYTHON_FLAG" python -c '
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
