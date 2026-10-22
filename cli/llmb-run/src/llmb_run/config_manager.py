# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration management for llmb-run."""

import logging
import os
import pathlib
from dataclasses import dataclass, field
from typing import Any, Dict

import yaml

logger = logging.getLogger('llmb_run.config_manager')

_LEGACY_SLURM_FIELDS = {'gpu_partition', 'cpu_partition', 'gpu_gres', 'cpu_gres'}
_TARGET_KEYS = {'gpu', 'cpu'}
_SCHEMA_VERSION_KEY = 'schema_version'
_LEGACY_SCHEMA_VERSION = 1
_CURRENT_SCHEMA_VERSION = 2
_FLAT_CLUSTER_KEYS = {'gpu_type', 'llmb_install', 'llmb_repo', 'cluster_name'}
_REQUIRED_CLUSTER_KEYS = ('gpu_type', 'llmb_install', 'llmb_repo')


@dataclass(frozen=True)
class SlurmTargetConfig:
    """Resolved Slurm environment variables for a single target (gpu/cpu)."""

    env_vars: Dict[str, str]

    def env(self) -> Dict[str, str]:
        return dict(self.env_vars)

    @property
    def partition(self) -> str:
        return self.env_vars.get('SBATCH_PARTITION') or self.env_vars.get('SLURM_PARTITION') or ''

    @property
    def gpus_per_node(self) -> str:
        return self.env_vars.get('SBATCH_GPUS_PER_NODE') or self.env_vars.get('SLURM_GPUS_PER_NODE') or ''


@dataclass(frozen=True)
class SlurmConfig:
    """Typed access to resolved Slurm target configuration."""

    gpu: SlurmTargetConfig
    cpu: SlurmTargetConfig

    def _select_target(self, target: str) -> SlurmTargetConfig:
        if target not in _TARGET_KEYS:
            raise ValueError(f"Invalid Slurm target '{target}'. Expected one of: {', '.join(sorted(_TARGET_KEYS))}")
        return self.gpu if target == 'gpu' else self.cpu

    def env(self, target: str = 'gpu') -> Dict[str, str]:
        return self._select_target(target).env()

    def partition(self, target: str = 'gpu') -> str:
        return self._select_target(target).partition

    def gpus_per_node(self, target: str = 'gpu') -> str:
        return self._select_target(target).gpus_per_node


@dataclass(frozen=True)
class WorkloadsConfig:
    """Typed workloads section from cluster config."""

    installed: list[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class InstallConfig:
    """Typed install metadata from cluster config."""

    node_architecture: str = 'x86_64'
    extras: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ClusterConfig:
    """Resolved cluster configuration for llmb-run."""

    schema_version: int
    gpu_type: str
    llmb_install: str
    llmb_repo: str
    cluster_name: str | None
    install: InstallConfig
    slurm: SlurmConfig
    workloads: WorkloadsConfig
    environment: Dict[str, Any]
    cwd: pathlib.Path
    extras: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw_config: Dict[str, Any]) -> 'ClusterConfig':
        if not isinstance(raw_config, dict):
            raise ValueError("Invalid cluster configuration: expected a dictionary.")

        normalized_config = _normalize_cluster_config(raw_config)

        resolved_targets = resolve_slurm_targets(normalized_config)
        slurm = SlurmConfig(
            gpu=SlurmTargetConfig(resolved_targets['gpu']),
            cpu=SlurmTargetConfig(resolved_targets['cpu']),
        )

        workloads = _parse_workloads_config(normalized_config)
        install = _parse_install_config(normalized_config)

        environment = normalized_config.get('environment')
        if environment is None:
            environment = {}
        if not isinstance(environment, dict):
            raise ValueError("Invalid cluster configuration: 'environment' section must be a dictionary when provided.")

        cwd = normalized_config.get('cwd')
        if cwd is None:
            cwd = pathlib.Path.cwd()
        elif not isinstance(cwd, pathlib.Path):
            cwd = pathlib.Path(cwd)

        known_top_level_keys = {
            _SCHEMA_VERSION_KEY,
            'gpu_type',
            'llmb_install',
            'llmb_repo',
            'cluster_name',
            'install',
            'slurm',
            'workloads',
            'environment',
            'cwd',
        }
        extras = {key: value for key, value in normalized_config.items() if key not in known_top_level_keys}

        return cls(
            schema_version=normalized_config[_SCHEMA_VERSION_KEY],
            gpu_type=normalized_config['gpu_type'],
            llmb_install=normalized_config['llmb_install'],
            llmb_repo=normalized_config['llmb_repo'],
            cluster_name=normalized_config.get('cluster_name'),
            install=install,
            slurm=slurm,
            workloads=workloads,
            environment=dict(environment),
            cwd=cwd,
            extras=extras,
        )

    def workload_config(self, workload_key: str) -> Dict[str, Any]:
        return self.workloads.config.get(workload_key, {})


def _parse_schema_version(raw_config: Dict[str, Any]) -> int:
    schema_version = raw_config.get(_SCHEMA_VERSION_KEY, _LEGACY_SCHEMA_VERSION)
    if not isinstance(schema_version, int) or isinstance(schema_version, bool):
        raise ValueError("Invalid cluster configuration: 'schema_version' must be an integer when provided.")
    return schema_version


def _validate_required_string(value: Any, field_name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing required '{field_name}' field in cluster configuration.")


def _normalize_legacy_cluster_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    flattened_keys_present = [key for key in sorted(_FLAT_CLUSTER_KEYS) if key in raw_config]

    if 'launcher' not in raw_config:
        if flattened_keys_present:
            raise ValueError("Invalid cluster configuration: flattened top-level keys require 'schema_version: 2'.")
        raise ValueError("Missing required 'launcher' section in cluster configuration.")

    if flattened_keys_present:
        raise ValueError(
            "Invalid cluster configuration: legacy 'launcher' section cannot be used with flattened top-level keys."
        )

    launcher_config = raw_config['launcher']
    if not isinstance(launcher_config, dict):
        raise ValueError("Invalid cluster configuration: 'launcher' section must be a dictionary.")

    if 'gpu_type' not in launcher_config:
        raise ValueError("Missing required 'launcher.gpu_type' field in cluster configuration.")
    if 'llmb_install' not in launcher_config:
        raise ValueError("Missing required 'launcher.llmb_install' field in cluster configuration.")
    if 'llmb_repo' not in launcher_config:
        raise ValueError("Missing required 'launcher.llmb_repo' field in cluster configuration.")
    _validate_required_string(launcher_config['llmb_repo'], 'launcher.llmb_repo')

    normalized = dict(raw_config)
    normalized[_SCHEMA_VERSION_KEY] = _LEGACY_SCHEMA_VERSION
    normalized['gpu_type'] = launcher_config['gpu_type']
    normalized['llmb_install'] = launcher_config['llmb_install']
    normalized['llmb_repo'] = launcher_config['llmb_repo']
    if 'cluster_name' in launcher_config:
        normalized['cluster_name'] = launcher_config['cluster_name']
    normalized.pop('launcher', None)

    return normalized


def _normalize_v2_cluster_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    if 'launcher' in raw_config:
        raise ValueError(
            "Invalid cluster configuration: 'launcher' section is not supported for schema_version: 2. "
            "Use flattened top-level keys instead."
        )

    for key in _REQUIRED_CLUSTER_KEYS:
        if key not in raw_config:
            raise ValueError(f"Missing required '{key}' field in cluster configuration.")
    _validate_required_string(raw_config['llmb_repo'], 'llmb_repo')

    normalized = dict(raw_config)
    normalized[_SCHEMA_VERSION_KEY] = _CURRENT_SCHEMA_VERSION
    return normalized


def _normalize_cluster_config(raw_config: Dict[str, Any]) -> Dict[str, Any]:
    schema_version = _parse_schema_version(raw_config)
    if schema_version == _LEGACY_SCHEMA_VERSION:
        return _normalize_legacy_cluster_config(raw_config)
    if schema_version == _CURRENT_SCHEMA_VERSION:
        return _normalize_v2_cluster_config(raw_config)
    raise ValueError(
        f"Unsupported cluster configuration schema_version '{schema_version}'. "
        f"Supported versions are: {_LEGACY_SCHEMA_VERSION}, {_CURRENT_SCHEMA_VERSION}."
    )


def _parse_workloads_config(raw_config: Dict[str, Any]) -> WorkloadsConfig:
    workloads_config = raw_config.get('workloads')
    if workloads_config is None:
        workloads_config = {}
    if not isinstance(workloads_config, dict):
        raise ValueError("Invalid cluster configuration: 'workloads' section must be a dictionary when provided.")

    installed = workloads_config.get('installed')
    if installed is None:
        installed = []
    if not isinstance(installed, list):
        raise ValueError("Invalid cluster configuration: 'workloads.installed' must be a list when provided.")

    workload_settings = workloads_config.get('config')
    if workload_settings is None:
        workload_settings = {}
    if not isinstance(workload_settings, dict):
        raise ValueError("Invalid cluster configuration: 'workloads.config' must be a dictionary when provided.")

    extras = {key: value for key, value in workloads_config.items() if key not in {'installed', 'config'}}

    return WorkloadsConfig(installed=list(installed), config=dict(workload_settings), extras=extras)


def _parse_install_config(raw_config: Dict[str, Any]) -> InstallConfig:
    install_config = raw_config.get('install')
    if install_config is None:
        return InstallConfig()
    if not isinstance(install_config, dict):
        raise ValueError("Invalid cluster configuration: 'install' section must be a dictionary when provided.")

    node_architecture = install_config.get('node_architecture', 'x86_64')
    if not isinstance(node_architecture, str) or not node_architecture.strip():
        raise ValueError(
            "Invalid cluster configuration: 'install.node_architecture' must be a non-empty string when provided."
        )

    extras = {key: value for key, value in install_config.items() if key not in {'node_architecture'}}

    return InstallConfig(node_architecture=node_architecture, extras=extras)


def _normalize_slurm_key(key: str) -> str:
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"Invalid Slurm key '{key}'. Keys must be non-empty strings.")

    key_norm = key.strip().replace('-', '_')
    key_lower = key_norm.lower()
    if key_lower == 'gres':
        # Backward-compatible behavior: config "gres: <n>" means GPUs-per-node request.
        return 'SBATCH_GPUS_PER_NODE'

    if key_lower.startswith('sbatch_'):
        return 'SBATCH_' + key_norm[7:].upper()
    if key_lower.startswith('slurm_'):
        return 'SLURM_' + key_norm[6:].upper()

    return 'SBATCH_' + key_norm.upper()


def _normalize_slurm_block(
    raw_block: Dict[str, Any], *, block_name: str, allow_null_unset: bool = True
) -> Dict[str, str | None]:
    if raw_block is None:
        return {}
    if not isinstance(raw_block, dict):
        raise ValueError(f"Invalid Slurm '{block_name}' block: expected a dictionary.")

    normalized: Dict[str, str | None] = {}
    for raw_key, raw_value in raw_block.items():
        key = _normalize_slurm_key(raw_key)
        if raw_value is None:
            if allow_null_unset:
                normalized[key] = None
            continue
        normalized[key] = str(raw_value)
    return normalized


def _apply_target_overrides(base: Dict[str, str], overrides: Dict[str, str | None]) -> Dict[str, str]:
    merged = dict(base)
    for key, value in overrides.items():
        if value is None:
            merged.pop(key, None)
        else:
            merged[key] = value
    return merged


def _get_legacy_target_blocks(slurm_config: Dict[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    common = {}
    for key, value in slurm_config.items():
        if key in _LEGACY_SLURM_FIELDS:
            continue
        if isinstance(value, dict):
            raise ValueError(
                f"Invalid nested Slurm block '{key}'. Nested blocks are only supported for 'gpu' and 'cpu'."
            )
        common[key] = value

    gpu = {}
    if slurm_config.get('gpu_partition') is not None:
        gpu['partition'] = slurm_config['gpu_partition']
    if slurm_config.get('gpu_gres') is not None:
        gpu['gpus_per_node'] = slurm_config['gpu_gres']

    cpu = {}
    if slurm_config.get('cpu_partition') is not None:
        cpu['partition'] = slurm_config['cpu_partition']
    if slurm_config.get('cpu_gres') is not None:
        cpu['gpus_per_node'] = slurm_config['cpu_gres']

    return common, gpu, cpu


def _get_new_target_blocks(
    slurm_config: Dict[str, Any],
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any] | None]:
    if 'gpu' not in slurm_config:
        raise ValueError("Invalid Slurm configuration: 'slurm.gpu' section is required in the new schema.")

    common = {}
    for key, value in slurm_config.items():
        if key in _TARGET_KEYS:
            continue
        if isinstance(value, dict):
            raise ValueError(f"Invalid nested Slurm block '{key}'. Only 'slurm.gpu' and 'slurm.cpu' are supported.")
        common[key] = value

    gpu = slurm_config.get('gpu')
    cpu = slurm_config.get('cpu')

    if not isinstance(gpu, dict):
        raise ValueError("Invalid Slurm configuration: 'slurm.gpu' must be a dictionary.")
    if cpu is not None and not isinstance(cpu, dict):
        raise ValueError("Invalid Slurm configuration: 'slurm.cpu' must be a dictionary when provided.")

    return common, gpu, cpu


def resolve_slurm_targets(config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    slurm_config = config.get('slurm', {})
    if not slurm_config:
        raise ValueError(
            "Missing required 'slurm' section in cluster configuration. llmb-run currently requires Slurm settings."
        )
    if not isinstance(slurm_config, dict):
        raise ValueError("Invalid Slurm configuration: 'slurm' section must be a dictionary.")

    has_legacy = any(field in slurm_config for field in _LEGACY_SLURM_FIELDS)
    has_target_blocks = any(field in slurm_config for field in _TARGET_KEYS)

    if has_legacy and has_target_blocks:
        raise ValueError(
            "Invalid Slurm configuration: legacy keys (gpu_partition/cpu_partition/gpu_gres/cpu_gres) "
            "cannot be used with the new slurm.gpu/slurm.cpu schema."
        )

    if has_target_blocks:
        raw_common, raw_gpu, raw_cpu = _get_new_target_blocks(slurm_config)
    elif has_legacy:
        raw_common, raw_gpu, raw_cpu = _get_legacy_target_blocks(slurm_config)
    else:
        # slurm exists but does not use legacy keys or target blocks; this is invalid in the new schema.
        raise ValueError("Invalid Slurm configuration: 'slurm.gpu' section is required.")

    common = _normalize_slurm_block(raw_common, block_name='common', allow_null_unset=False)
    gpu_overrides = _normalize_slurm_block(raw_gpu, block_name='gpu', allow_null_unset=True)
    cpu_overrides = _normalize_slurm_block(raw_cpu, block_name='cpu', allow_null_unset=True) if raw_cpu else None

    gpu_env = _apply_target_overrides(common, gpu_overrides)
    cpu_env = _apply_target_overrides(common, cpu_overrides) if cpu_overrides is not None else dict(gpu_env)

    if 'SBATCH_PARTITION' not in gpu_env and 'SLURM_PARTITION' not in gpu_env:
        raise ValueError(
            "Invalid Slurm configuration: resolved GPU target is missing partition. "
            "Set 'slurm.gpu.partition' (or equivalent SBATCH_PARTITION/SLURM_PARTITION key)."
        )

    return {'gpu': gpu_env, 'cpu': cpu_env}


def get_cluster_config() -> ClusterConfig:
    """
    Load and validate cluster configuration from cluster_config.yaml.

    It will check for the cluster_config.yaml file in the current directory and the path specified by the LLMB_INSTALL environment variable.
    Priority is given to the current directory.
    """
    config_file_name = 'cluster_config.yaml'
    config_path_in_cwd = pathlib.Path.cwd() / config_file_name
    config_path = None

    llmb_install_path = os.environ.get('LLMB_INSTALL')

    # Configuration file in the current directory takes precedence.
    if config_path_in_cwd.exists():
        config_path = config_path_in_cwd
        if llmb_install_path:
            logger.debug(f"Found '{config_file_name}' in current directory, which takes precedence over LLMB_INSTALL.")
    # If not in CWD, check the path specified by the LLMB_INSTALL environment variable.
    elif llmb_install_path:
        llmb_config_path = pathlib.Path(llmb_install_path) / config_file_name
        if llmb_config_path.exists():
            config_path = llmb_config_path

    if config_path:
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing cluster configuration file '{config_path}': {e}") from e
    else:
        raise FileNotFoundError(
            f"Cluster configuration file '{config_file_name}' not found. Looked in the current directory and under the path specified by the LLMB_INSTALL environment variable."
        )

    # Add current working directory for output logs
    config['cwd'] = pathlib.Path.cwd()
    # Resolve and validate Slurm targets once during load, then expose typed config object.
    return ClusterConfig.from_dict(config)
