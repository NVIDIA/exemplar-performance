#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -eu -o pipefail

# Validate required environment variables
if [ -z "${LLMB_WORKLOAD:-}" ]; then
    echo "Error: LLMB_WORKLOAD environment variable is not set" >&2
    exit 1
fi

cd "$LLMB_WORKLOAD/torchtitan" || exit 1

# Check if download script exists
DOWNLOAD_SCRIPT="scripts/download_hf_assets.py"
if [ ! -f "$DOWNLOAD_SCRIPT" ]; then
    echo "Error: Download script not found: $DOWNLOAD_SCRIPT" >&2
    exit 1
fi

# Pass HF_TOKEN directly to the download script without calling login()
# The login() function makes additional API calls that can trigger rate limits
if [ -n "$HF_TOKEN" ]; then
    echo "Using HuggingFace token for download..."
    python "$DOWNLOAD_SCRIPT" --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer --hf_token "$HF_TOKEN"
else
    echo "WARNING: No HF_TOKEN provided"
    python "$DOWNLOAD_SCRIPT" --repo_id deepseek-ai/DeepSeek-V3.1-Base --assets tokenizer
fi
