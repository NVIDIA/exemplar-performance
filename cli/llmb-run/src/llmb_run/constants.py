# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Constants used throughout the llmb-run application."""

# SLURM configuration
SLURM_OUTPUT_PATTERN = 'slurm-%j.out'

# File patterns
METADATA_FILE_PATTERN = '**/metadata.yaml'

# NVLink switch domain size for GB200/GB300 (GB200 NVL72 = 18 nodes x 4 GPUs)
NVLINK_DOMAIN_SIZE = 18

# Mapping of GPU types to number of GPUs per node
GPU_TYPE_TO_NUM_GPUS = {
    "a100": 8,
    "h100": 8,
    "b200": 8,
    "b300": 8,
    "gb200": 4,
    "gb300": 4,
}

# Workload exclusions (currently empty but keeping for future use)
EXCLUDE_WORKLOADS = []

# Env vars carrying the pipeline-triggering/human user, checked in order.
# GITLAB_USER_LOGIN/GITHUB_ACTOR are CI-provided (the OS/container user in a
# runner is generic, e.g. root); SUDO_USER covers the interactive sudo case.
USER_ENV_VARS = ('GITLAB_USER_LOGIN', 'GITHUB_ACTOR', 'SUDO_USER')
