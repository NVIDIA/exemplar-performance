# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Central constants for LLMB installer - single source of truth."""

from dataclasses import dataclass
from typing import Dict

# Process exit codes
EXIT_SUCCESS = 0
EXIT_ERROR = 1
# Used for user-initiated early termination (Ctrl-C or explicit cancel/exit
# choices) so wrapper scripts can suppress post-install summaries.
EXIT_CANCELLED = 130

# Python version requirements for virtual environments
MIN_PYTHON_VERSION = "3.12"
MAX_PYTHON_VERSION = "3.13"
MIN_PYTHON_VERSION_TUPLE = tuple(map(int, MIN_PYTHON_VERSION.split('.')))
MAX_PYTHON_VERSION_TUPLE = tuple(map(int, MAX_PYTHON_VERSION.split('.')))

# GPU types supported
SUPPORTED_GPU_TYPES = {'vr', 'gb300', 'gb200', 'b300', 'b200', 'h100'}

# Productized NVLink domains (GPUs) per GPU type. Domains larger than a node are auto-detected from Slurm block topology.
NVL_DOMAINS_BY_GPU = {
    'vr': [72],
    'gb300': [72],
    'gb200': [72],
    'b300': [8],
    'b200': [8],
    'h100': [8],
}
GPU_TYPE_TO_NUM_GPUS = {
    'vr': 4,
    'gb300': 4,
    'gb200': 4,
    'b300': 8,
    'b200': 8,
    'h100': 8,
}

# Display order for the GPU type picker: newest chips first.
# Every entry in SUPPORTED_GPU_TYPES must appear in this list.
GPU_DISPLAY_ORDER = ['vr', 'gb300', 'gb200', 'b300', 'b200', 'h100']

# Architecture types
ARCHITECTURES = {'x86_64': 'x86_64', 'aarch64': 'aarch64'}

# Default architecture by GPU type
DEFAULT_ARCHITECTURE_BY_GPU = {
    'vr': {'arch': 'aarch64', 'fixed_arch': True},
    'gb300': {'arch': 'aarch64', 'fixed_arch': True},
    'gb200': {'arch': 'aarch64', 'fixed_arch': True},
    'b300': {'arch': 'x86_64', 'fixed_arch': False},
    'b200': {'arch': 'x86_64', 'fixed_arch': False},
    'h100': {'arch': 'x86_64', 'fixed_arch': False},
}

### TOOLS CONSTANTS ###
# Tool download patterns (tool-specific)
# NSys download URL and filename patterns
# Note: Versions include full version string with build number (e.g., '2025.5.1.121-3638078')
NSYS_BASE_URL = 'https://developer.download.nvidia.com/devtools/nsight-systems'
NSYS_FILENAME_PATTERNS = {
    'x86_64': 'NsightSystems-linux-public-{version}.run',
    'aarch64': 'NsightSystems-linux-sbsa-public-{version}.run',
}


@dataclass(frozen=True)
class NsysSource:
    """A download source for NSys builds (see NSYS_SOURCES)."""

    version_pattern: str
    base_url: str
    url_template: str
    cache_name_template: str
    filenames: Dict[str, str]
    verify_tls: bool


# NSys download sources: _nsys_download_spec uses the first entry whose
# 'version_pattern' matches the pinned version. verify_tls=False is only for
# servers whose CA is not in the default trust store.
NSYS_SOURCES = [
    NsysSource(
        version_pattern=r'.*',
        base_url=NSYS_BASE_URL,
        url_template='{base_url}/{filename}',
        cache_name_template='{filename}',
        filenames=NSYS_FILENAME_PATTERNS,
        verify_tls=True,
    ),
]

# CUDA CUPTI download URL and filename patterns
# Note: Versions are in format like '13.0.85'
CUDA_CUPTI_BASE_URL = 'https://developer.download.nvidia.com/compute/cuda/redist/cuda_cupti'
CUDA_CUPTI_FILENAME_PATTERNS = {
    'x86_64': 'cuda_cupti-linux-x86_64-{version}-archive.tar.xz',
    'aarch64': 'cuda_cupti-linux-sbsa-{version}-archive.tar.xz',
}

# Supported tools (explicit allow-list)
SUPPORTED_TOOLS = {'nsys', 'cuda_cupti_lib'}

# uv standalone binary download patterns for compute-node use.
UV_BASE_URL = 'https://github.com/astral-sh/uv/releases/download'
UV_GNU_TARGETS = {
    'x86_64': 'x86_64-unknown-linux-gnu',
    'aarch64': 'aarch64-unknown-linux-gnu',
}
