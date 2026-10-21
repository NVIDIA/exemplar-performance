# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""NVLink topology detection utilities."""

import re
import subprocess
from dataclasses import dataclass
from typing import Literal, Optional

from llmb_install.constants import GPU_TYPE_TO_NUM_GPUS, NVL_DOMAINS_BY_GPU


@dataclass(frozen=True)
class NvlDetection:
    block_sizes: Optional[list[int]]
    nvl_domain: Optional[int]
    reason: Literal['detected', 'no_blocks', 'error']
    gpu_type: str
    error: Optional[str] = None


def describe_failure(detection: NvlDetection) -> str:
    if detection.reason == 'detected':
        expected = "/".join(f"NVL{domain}" for domain in NVL_DOMAINS_BY_GPU[detection.gpu_type])
        return (
            f"Detected NVL{detection.nvl_domain} from the Slurm topology; "
            f"{detection.gpu_type.upper()} systems are expected to be {expected}."
        )
    if detection.reason == 'no_blocks':
        return "Could not detect the NVLink domain from the Slurm topology."
    return f"Could not query the Slurm topology: {detection.error}"


def uses_block_topology(gpu_type: str) -> bool:
    """NVLink spans nodes, so the domain comes from Slurm block topology."""
    return max(NVL_DOMAINS_BY_GPU[gpu_type]) > GPU_TYPE_TO_NUM_GPUS[gpu_type]


def parse_topology_block_sizes(output: str) -> list[int]:
    """Extract distinct Slurm NVLink block sizes."""
    sizes: set[int] = set()
    for line in output.splitlines():
        if not (line.startswith('BlockName=') or line.startswith('AggregatedBlock=')):
            continue
        size_match = re.search(r'\bBlockSize=(\d+)', line)
        if size_match:
            sizes.add(int(size_match.group(1)))
    return sorted(sizes)


def detect_nvl_topology(gpu_type: str) -> NvlDetection:
    """Detect NVLink domain size from Slurm block topology."""
    gpus_per_node = GPU_TYPE_TO_NUM_GPUS[gpu_type]
    try:
        topology_result = subprocess.run(['scontrol', 'show', 'topology'], capture_output=True, text=True, check=True)
        block_sizes = parse_topology_block_sizes(topology_result.stdout)
        if not block_sizes:
            return NvlDetection(block_sizes=None, nvl_domain=None, reason='no_blocks', gpu_type=gpu_type)
        # Block sizes are node counts; the largest (aggregated) block is the NVLink domain.
        return NvlDetection(
            block_sizes=block_sizes,
            nvl_domain=max(block_sizes) * gpus_per_node,
            reason='detected',
            gpu_type=gpu_type,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return NvlDetection(block_sizes=None, nvl_domain=None, reason='error', gpu_type=gpu_type, error=str(exc))
