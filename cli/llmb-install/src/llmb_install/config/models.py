# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


"""Configuration data models for LLMB Install."""

import re
from enum import StrEnum
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import (
    AfterValidator,
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    PositiveInt,
    field_validator,
    model_validator,
)

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
_ENVIRONMENT_VARIABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class EnvironmentPrompt(BaseModel):
    """Declaration of an environment variable the installer prompts for."""

    name: NonBlankStr
    description: str = ""
    required: bool = True
    workloads: List[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _coerce_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        return data

    @field_validator("name")
    @classmethod
    def _validate_identifier(cls, v: str) -> str:
        if not _ENVIRONMENT_VARIABLE_NAME.fullmatch(v):
            raise ValueError(f"'{v}' is not a valid environment variable name")
        return v


class ExecutionTarget(StrEnum):
    """Where a workload can be executed (mirrors cluster_config.yaml's ``execution_target``)."""

    SLURM = 'slurm'
    KUBERNETES = 'kubernetes'


ExecutionTargetList = Annotated[List[ExecutionTarget], Field(min_length=1)]


class ExecutionTargetsByGpu(BaseModel):
    by_gpu: Dict[str, ExecutionTargetList]


class RunConfig(BaseModel):
    """The subset of a workload's ``run`` block that the installer interprets.

    ``extra='ignore'`` keeps this a partial view: the authoritative metadata contract
    is the yamale schema in .gitlab/ci/metadata_schema.yaml. This model validates only
    the fields the CLIs read at runtime, and must stay in sync with the schema. The
    same model is duplicated in llmb-run, which is a separate package.
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


class SlurmConfig(BaseModel):
    """SLURM cluster configuration."""

    account: NonBlankStr
    gpu_partition: NonBlankStr
    cpu_partition: NonBlankStr
    gpu_partition_gres: Optional[int] = None
    cpu_partition_gres: Optional[int] = None


class KubernetesConfig(BaseModel):
    """Kubernetes cluster configuration for the 'kubernetes' install method.

    Used when workloads run as nvcre WorkloadRun custom resources rather than
    SLURM jobs. Written verbatim to cluster_config.yaml's `kubernetes` block and
    consumed by llmb-run's NVCreLauncher.
    """

    namespace: NonBlankStr = 'default'
    kubeconfig: Optional[str] = None  # None -> standard resolution ($KUBECONFIG, ~/.kube/config)
    image_pull_secrets: List[NonBlankStr] = []
    # Optional PVC for syncing job code/configs before WorkloadRun submission.
    workdir_pvc: Optional[str] = None
    # JSON-encoded node selector dict, e.g. '{"nvidia.com/gpu.product":"NVIDIA-GB200-NVL72"}'.
    node_selector_json: Optional[str] = None
    # Per-job timeout forwarded to nvcrectl (e.g. "12h", "30m").
    timeout_per_job: str = '12h'
    # Gang scheduler to request via spec.gangScheduler.schedulerName (e.g. "kai-scheduler").
    gang_scheduler_name: Optional[str] = None
    # Install-time only (collected by the prompt, never written to cluster_config.yaml).
    # The NGC key is a secret: excluded from model_dump() so it is not saved to system config
    # or install state either; read it from the object when creating image pull secrets.
    ngc_api_key: Optional[str] = Field(default=None, exclude=True, repr=False)
    run_nvcrectl_init: bool = False
    nvcrectl_chart_version: Optional[str] = None


class ClusterSettings(BaseModel):
    """Base cluster configuration shared across all config models.

    Contains the stable settings that describe a cluster environment:
    GPU type, architecture, venv strategy, slurm configuration, etc.
    """

    venv_type: Literal['uv', 'venv', 'conda']
    gpu_type: GpuType
    node_architecture: Literal['x86_64', 'aarch64']
    image_download_method: Literal['local', 'slurm'] = 'slurm'
    slurm: Optional[SlurmConfig] = None
    kubernetes: Optional[KubernetesConfig] = None
    workload_selection_mode: Literal['custom', 'exemplar'] = 'custom'
    environment_vars: EnvironmentVars = {}
    image_folder: Optional[str] = None
    nvl_domain: Optional[PositiveInt] = None
    # Raw Slurm BlockSize values (nodes), as reported, even when nvl_domain was chosen manually.
    nvl_block_sizes: Optional[List[PositiveInt]] = None

    @property
    def execution_target(self) -> ExecutionTarget:
        """Orchestration target: 'kubernetes' when a kubernetes block is configured, else 'slurm'."""
        return ExecutionTarget.KUBERNETES if self.kubernetes is not None else ExecutionTarget.SLURM


def migrate_legacy_install_method(data: Any) -> Any:
    """Return playfile *data* with the pre-rename ``install_method`` key as ``image_download_method``.

    ``kubernetes`` was never a download method (it is selected by the ``kubernetes``
    section), so that value is dropped.
    """
    if isinstance(data, dict) and 'install_method' in data:
        data = dict(data)
        legacy = data.pop('install_method')
        if legacy in ('local', 'slurm'):
            data.setdefault('image_download_method', legacy)
    return data


class PlayfileConfig(ClusterSettings):
    """Schema for headless playfile configuration.

    Defines the fields that are valid in a playfile YAML, along with
    playfile-specific validation rules (non-empty workloads, deprecated
    key rejection). SLURM configuration is required for the 'slurm' and 'local'
    methods; the 'kubernetes' method requires no SLURM section.
    """

    install_path: NonBlankStr
    selected_workloads: List[NonBlankStr]
    # Required for slurm/local, absent for kubernetes; enforced per-target below.
    slurm: Optional[SlurmConfig] = None

    @model_validator(mode='before')
    @classmethod
    def migrate_install_method(cls, data: Any) -> Any:
        return migrate_legacy_install_method(data)

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
        if self.kubernetes is not None:
            if self.slurm is not None:
                raise ValueError("slurm must not be set when kubernetes is configured.")
        elif self.slurm is None:
            raise ValueError(
                f"slurm configuration is required for image_download_method '{self.image_download_method}'."
            )
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
        if data.get('nvl_domain') is None:
            data.pop('nvl_domain', None)
        if data.get('nvl_block_sizes') is None:
            data.pop('nvl_block_sizes', None)

        # Drop the unused target block so playfiles stay clean: kubernetes installs
        # carry no slurm, slurm/local installs carry no kubernetes.
        for key in ('slurm', 'kubernetes'):
            if data.get(key) is None:
                data.pop(key, None)

        return data

    def get_remaining_workloads(self, completed: List[str]) -> List[str]:
        """Get workloads that still need to be installed."""
        return [w for w in self.selected_workloads if w not in completed]

    @property
    def locked_fields_for_resume(self) -> List[str]:
        """Get list of fields that cannot be changed during resume edit."""
        return [
            'install_path',
            'gpu_type',
            'node_architecture',
            'venv_type',
            'llmb_repo',
            'dev_mode',
            'image_folder',
            'nvl_domain',
            'nvl_block_sizes',
        ]

    @property
    def editable_fields_for_resume(self) -> List[str]:
        """Get list of fields that can be changed during resume edit."""
        return ['slurm', 'kubernetes', 'image_download_method', 'selected_workloads']
