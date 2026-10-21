# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# For each dataset a user elects to use, the user is responsible for
# checking if the dataset license is fit for the intended purpose.

"""Utilities for working with workload metadata structures.

Centralizes normalization for dtype/scales configuration so all call sites
behave consistently.

Scale Validation Modes:
-----------------------
The 'exact_scales' field controls how scale validation works:

- exact_scales=False (default): Flexible scale validation
  * Allows scales explicitly listed in metadata
  * Allows scales ABOVE the maximum tested scale (with warning)
  * Use case: Workloads that can scale beyond tested configurations
  * Example: If metadata has scales=[128, 256], user can run scale=512

- exact_scales=True: Strict scale validation
  * ONLY allows scales explicitly listed in metadata
  * Rejects any scale not in the list
  * Use case: Workloads with specific hardware requirements or tested configs only
  * Example: If metadata has scales=[128, 256], scale=512 is rejected

This distinction is important for:
1. Preventing users from running untested configurations on strict workloads
2. Allowing exploration of larger scales on flexible workloads
3. Providing clear validation errors when scales are not supported
"""

from __future__ import annotations

import logging
import re
from typing import Annotated, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from llmb_run.config_manager import ExecutionTarget

logger = logging.getLogger("llmb_run.metadata_utils")

# Known dtype keys supported by our tooling. Extend as needed.
_KNOWN_DTYPES = {"fp8", "bf16", "nvfp4", "mxfp4"}
_MODEL_SIZE_SUFFIX_RE = re.compile(r'^(?P<value>\d+(?:\.\d+)?)(?P<unit>[bt])$', re.IGNORECASE)


ExecutionTargetList = Annotated[List[ExecutionTarget], Field(min_length=1)]


class ExecutionTargetsByGpu(BaseModel):
    by_gpu: Dict[str, ExecutionTargetList]


class RunConfig(BaseModel):
    """The subset of a workload's ``run`` block that llmb-run interprets.

    ``extra='ignore'`` keeps this a partial view: the authoritative metadata contract
    is the yamale schema in .gitlab/ci/metadata_schema.yaml. This model validates only
    the fields the CLIs read at runtime, and must stay in sync with the schema. The
    same model is duplicated in llmb-install, which is a separate package.
    """

    model_config = ConfigDict(extra='ignore')

    execution_targets: ExecutionTargetList | ExecutionTargetsByGpu = Field(
        default_factory=lambda: [ExecutionTarget.SLURM]
    )

    def targets_for(self, gpu_type: Optional[str] = None) -> List[ExecutionTarget]:
        if isinstance(self.execution_targets, list):
            return self.execution_targets
        by_gpu = self.execution_targets.by_gpu
        return by_gpu.get(gpu_type, by_gpu.get('default', []))

    def supports(self, execution_target: str, gpu_type: Optional[str] = None) -> bool:
        """Whether this workload's launch script can run on ``execution_target``."""
        return execution_target in self.targets_for(gpu_type)


def normalize_model_dtype_config(model_config: dict) -> Dict[str, Dict[str, object]]:
    """Normalize a model_config's dtype/scales definition.

    Returns a mapping of dtype -> { 'scales': list[int], 'exact_scales': bool, 'proxy_scales': list[int] }

    Accepted input forms on the model_config:
      1) Legacy form
         dtypes: ['fp8','bf16'] | 'fp8'
         scales: [128,256]
         exact_scales: bool (optional)
         proxy_scales: [8,16] (optional)

      2) Mapping form (per-dtype config)
         dtypes:
           fp8: [128, 256]                 # short form = scales only
           bf16: { scales: [256, 512], exact_scales: true, proxy_scales: [16, 32] }

    Notes:
      - If mapping form is used but contains non-dtype keys (e.g., mistakenly
        nested 'scales' or 'exact_scales' under 'dtypes'), those keys are
        ignored with a debug log.
      - If mapping form is used, any top-level scales are ignored.
      - proxy_scales are always treated as exact (no power-of-2 expansion).
    """
    normalized: Dict[str, Dict[str, object]] = {}

    dtypes_value = model_config.get("dtypes")
    model_scales = model_config.get("scales", [])
    model_exact = bool(model_config.get("exact_scales", False))
    model_proxy_scales = model_config.get("proxy_scales", [])

    # Mapping form
    if isinstance(dtypes_value, dict):
        for key, val in dtypes_value.items():
            if key not in _KNOWN_DTYPES:
                # Ignore non-dtype keys under the mapping and continue
                logger.debug("Ignoring non-dtype key under dtypes mapping: %s", key)
                continue

            if isinstance(val, list):
                normalized[key] = {
                    "scales": [int(s) for s in val],
                    "exact_scales": model_exact,
                    "proxy_scales": [int(s) for s in model_proxy_scales],
                }
            elif isinstance(val, dict):
                dtype_scales = [int(s) for s in val.get("scales", [])]
                dtype_exact = bool(val.get("exact_scales", model_exact))
                dtype_proxy_scales = [int(s) for s in val.get("proxy_scales", model_proxy_scales)]
                normalized[key] = {
                    "scales": dtype_scales,
                    "exact_scales": dtype_exact,
                    "proxy_scales": dtype_proxy_scales,
                }
            else:
                logger.debug(
                    "Unsupported dtype mapping value for key %s: %r (ignored)",
                    key,
                    val,
                )
        return normalized

    # Legacy forms
    if isinstance(dtypes_value, str):
        dtype_list: List[str] = [dtypes_value]
    elif isinstance(dtypes_value, list):
        dtype_list = list(dtypes_value)
    else:
        dtype_list = []

    for dt in dtype_list:
        normalized[dt] = {
            "scales": [int(s) for s in model_scales],
            "exact_scales": model_exact,
            "proxy_scales": [int(s) for s in model_proxy_scales],
        }

    return normalized


def parse_workload_name(workload_name: str) -> tuple[str, str | None]:
    """Parse workload name into base workload_key and optional model_size suffix.

    The model_size suffix pattern is: _<digits>[.<digits>](b|t) at end of string

    Args:
        workload_name: Full workload name (e.g., 'pretrain_llama3.1_70b')

    Returns:
        Tuple of (workload_key, model_size) or (workload_name, None) if no valid suffix

    Examples:
        >>> parse_workload_name('pretrain_foo_7b')
        ('pretrain_foo', '7b')
        >>> parse_workload_name('pretrain_bar_340b')
        ('pretrain_bar', '340b')
        >>> parse_workload_name('pretrain_kimi-k2_1t')
        ('pretrain_kimi-k2', '1t')
        >>> parse_workload_name('pretrain_baz')
        ('pretrain_baz', None)
        >>> parse_workload_name('pretrain_invalid_7x')
        ('pretrain_invalid_7x', None)
    """
    if '_' not in workload_name:
        return workload_name, None

    parts = workload_name.rsplit('_', 1)
    potential_size = parts[1]

    # Check if last segment matches model size pattern
    if _MODEL_SIZE_SUFFIX_RE.match(potential_size):
        return parts[0], potential_size.lower()

    return workload_name, None


def model_size_to_billions(model_size: str) -> float:
    """Convert a model size suffix to billions for numeric comparisons.

    Examples:
        "70b" -> 70.0
        "1t" -> 1000.0
        "1.5t" -> 1500.0
    """
    match = _MODEL_SIZE_SUFFIX_RE.match(model_size)
    if not match:
        return 0.0

    value = float(match.group('value'))
    unit = match.group('unit').lower()
    return value * 1000 if unit == 't' else value
