# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Configuration data models for LLMB Install."""

from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import AfterValidator, BaseModel, BeforeValidator, model_validator

from llmb_install.constants import SUPPORTED_GPU_TYPES


def _check_gpu_type(v: str) -> str:
    if v not in SUPPORTED_GPU_TYPES:
        raise ValueError(f"Unsupported gpu_type '{v}'. Must be one of: {sorted(SUPPORTED_GPU_TYPES)}")
    return v


def _coerce_env_vars(v: Any) -> Any:
    if v is None:
        return {}
    if not isinstance(v, dict):
        return v  # let pydantic's type check handle it
    null_keys = [k for k, val in v.items() if val is None]
    if null_keys:
        raise ValueError(
            f"environment_vars contains null values for: {null_keys}. "
            "Use empty strings ('') instead of null/blank values, or remove the keys entirely."
        )
    return {k: str(val) for k, val in v.items()}


def _check_non_blank(v: str) -> str:
    if not v.strip():
        raise ValueError("Value must not be blank or whitespace-only")
    return v


GpuType = Annotated[str, AfterValidator(_check_gpu_type)]
NonBlankStr = Annotated[str, AfterValidator(_check_non_blank)]
EnvironmentVars = Annotated[Dict[str, str], BeforeValidator(_coerce_env_vars)]


class SlurmConfig(BaseModel):
    """SLURM cluster configuration."""

    account: NonBlankStr
    gpu_partition: NonBlankStr
    cpu_partition: NonBlankStr
    gpu_partition_gres: Optional[int] = None
    cpu_partition_gres: Optional[int] = None


class ClusterSettings(BaseModel):
    """Base cluster configuration shared across all config models.

    Contains the stable settings that describe a cluster environment:
    GPU type, architecture, venv strategy, slurm configuration, etc.
    """

    venv_type: Literal['uv', 'venv', 'conda']
    gpu_type: GpuType
    node_architecture: Literal['x86_64', 'aarch64']
    install_method: Literal['local', 'slurm'] = 'slurm'
    slurm: Optional[SlurmConfig] = None
    workload_selection_mode: Literal['custom', 'exemplar'] = 'custom'
    environment_vars: EnvironmentVars = {}
    image_folder: Optional[str] = None


class PlayfileConfig(ClusterSettings):
    """Schema for headless playfile configuration.

    Defines the fields that are valid in a playfile YAML, along with
    playfile-specific validation rules (non-empty workloads, deprecated
    key rejection). Slurm configuration is always required.
    """

    install_path: NonBlankStr
    install_method: Literal['local', 'slurm']  # required in playfiles (no default)
    selected_workloads: List[NonBlankStr]
    slurm: SlurmConfig  # required in playfiles (no default)

    @model_validator(mode='before')
    @classmethod
    def reject_deprecated_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # TODO: Remove this deprecated-key check after next public release.
            deprecated_keys = [key for key in ('slurm_info', 'env_vars') if key in data]
            if deprecated_keys:
                raise ValueError(
                    "Playfiles must use top-level slurm and environment_vars; "
                    "slurm_info/env_vars are no longer supported."
                )
        return data

    @model_validator(mode='after')
    def validate_playfile_rules(self) -> 'PlayfileConfig':
        if not self.selected_workloads:
            raise ValueError("selected_workloads cannot be empty")
        return self


class InstallConfig(ClusterSettings):
    """Central configuration object for LLMB installation."""

    # Required fields (no defaults)
    install_path: NonBlankStr

    # Optional fields (with defaults)
    selected_workloads: List[NonBlankStr] = []
    ui_mode: Literal['simple', 'rich', 'express'] = 'simple'
    cache_dirs_configured: bool = False
    dev_mode: bool = False
    llmb_repo: Optional[str] = None
    is_incremental_install: bool = False

    def to_play_dict(self) -> Dict[str, Any]:
        """Convert config to playfile-compatible dictionary."""
        data = PlayfileConfig.model_validate(self.model_dump()).model_dump()
        if data.get('image_folder') is None:
            data.pop('image_folder', None)
        return data

    def get_remaining_workloads(self, completed: List[str]) -> List[str]:
        """Get workloads that still need to be installed."""
        return [w for w in self.selected_workloads if w not in completed]

    @property
    def locked_fields_for_resume(self) -> List[str]:
        """Get list of fields that cannot be changed during resume edit."""
        return ['install_path', 'gpu_type', 'node_architecture', 'venv_type', 'llmb_repo', 'dev_mode', 'image_folder']

    @property
    def editable_fields_for_resume(self) -> List[str]:
        """Get list of fields that can be changed during resume edit."""
        return ['slurm', 'install_method', 'selected_workloads']
