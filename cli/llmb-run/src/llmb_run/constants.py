# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants used throughout the llmb-run application."""

# SLURM configuration
SLURM_OUTPUT_PATTERN = 'slurm-%j.out'

# File patterns
METADATA_FILE_PATTERN = '**/metadata.yaml'

# Fallback NVLink domain sizes (GPUs) used when llmb-config.yaml predates the
# nvl_domain field. GPU types absent from this map default to one node.
DEFAULT_NVL_DOMAIN_BY_GPU = {"vr": 72, "gb300": 72, "gb200": 72}

# Megatron-Bridge uses chip names instead of LLMB GPU types. Only GPU types
# whose Megatron-Bridge name differs are listed; everything else passes
# GPU_TYPE through unchanged.
MBRIDGE_GPU_TYPE_MAP = {"vr": "vr200"}

# Mapping of GPU types to number of GPUs per node
GPU_TYPE_TO_NUM_GPUS = {
    "vr": 4,
    "gb300": 4,
    "gb200": 4,
    "b300": 8,
    "b200": 8,
    "h100": 8,
    "a100": 8,
}

# Workload exclusions (currently empty but keeping for future use)
EXCLUDE_WORKLOADS = []

# Env vars carrying the pipeline-triggering/human user, checked in order.
# GITLAB_USER_LOGIN/GITHUB_ACTOR are CI-provided (the OS/container user in a
# runner is generic, e.g. root); SUDO_USER covers the interactive sudo case.
USER_ENV_VARS = ('GITLAB_USER_LOGIN', 'GITHUB_ACTOR', 'SUDO_USER')
