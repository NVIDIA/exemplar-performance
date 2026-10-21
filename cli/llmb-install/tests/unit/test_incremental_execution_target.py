# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Incremental installs detect and reconstruct the cluster's execution target."""

from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import yaml

from llmb_install.core.installer import Installer

KUBERNETES_CLUSTER_CONFIG = {
    'schema_version': 2,
    'execution_target': 'kubernetes',
    'gpu_type': 'h100',
    'llmb_repo': '/test/repo',
    'install': {'node_architecture': 'x86_64', 'venv_type': 'uv', 'image_download_method': 'local'},
    'workloads': {'installed': [], 'config': {}},
    'kubernetes': {
        'namespace': 'ns',
        'image_pull_secrets': [],
        'hf_token_secret_name': 'hf-token-secret',
        'hf_token_secret_key': 'hf-token',
    },
    'environment': {},
}

SLURM_CLUSTER_CONFIG = {
    'schema_version': 2,
    'execution_target': 'slurm',
    'gpu_type': 'h100',
    'llmb_repo': '/test/repo',
    'install': {'node_architecture': 'x86_64', 'venv_type': 'uv', 'image_download_method': 'local'},
    'workloads': {'installed': [], 'config': {}},
    'slurm': {'account': 'acct', 'gpu': {'partition': 'gpu'}, 'cpu': {'partition': 'cpu'}},
    'environment': {},
}

WORKLOADS = {'pretrain_x': {'run': {'execution_targets': ['slurm', 'kubernetes'], 'gpu_configs': {'h100': {}}}}}


def _write_config(tmpdir, config):
    with open(Path(tmpdir) / "cluster_config.yaml", 'w') as f:
        yaml.dump(config, f)


def _collect(installer, cluster_config):
    with (
        patch('llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads) as filt,
        patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}),
        patch('llmb_install.core.installer.resolve_gpu_overrides', return_value=installer.workloads),
        patch('llmb_install.core.installer.prompt_workload_selection', return_value=(['pretrain_x'], 'custom')),
    ):
        config, _ = installer._collect_incremental_configuration('simple', '/tmp/install', cluster_config, args=None)
    return config, filt


def test_kubernetes_installation_is_detected(tmpdir):
    installer = Installer()
    _write_config(tmpdir, KUBERNETES_CLUSTER_CONFIG)

    result = installer._detect_incremental_install(str(tmpdir))

    assert result is not None
    assert result['execution_target'] == 'kubernetes'


def test_kubernetes_target_without_kubernetes_block_is_rejected(tmpdir):
    installer = Installer()
    config = deepcopy(KUBERNETES_CLUSTER_CONFIG)
    del config['kubernetes']
    _write_config(tmpdir, config)

    assert installer._detect_incremental_install(str(tmpdir)) is None


def test_collect_incremental_configuration_builds_kubernetes_config():
    installer = Installer()
    installer.workloads = WORKLOADS

    config, filt = _collect(installer, KUBERNETES_CLUSTER_CONFIG)

    assert config is not None
    assert config.execution_target == 'kubernetes'
    assert config.slurm is None
    assert config.kubernetes is not None
    assert config.kubernetes.namespace == 'ns'
    assert filt.call_args.kwargs['execution_target'] == 'kubernetes'


def test_collect_incremental_configuration_builds_slurm_config():
    installer = Installer()
    installer.workloads = WORKLOADS

    config, filt = _collect(installer, SLURM_CLUSTER_CONFIG)

    assert config is not None
    assert config.execution_target == 'slurm'
    assert config.kubernetes is None
    assert config.slurm is not None
    assert config.slurm.account == 'acct'
    assert filt.call_args.kwargs['execution_target'] == 'slurm'


@pytest.mark.parametrize('selected_cluster_type', ['slurm', 'kubernetes'])
def test_path_selection_uses_existing_kubernetes_target(tmp_path, selected_cluster_type):
    installer = Installer()
    installer.root_dir = str(tmp_path)
    installer.workloads = WORKLOADS

    existing = deepcopy(KUBERNETES_CLUSTER_CONFIG)
    existing['llmb_repo'] = str(tmp_path)
    _write_config(tmp_path, existing)

    ui = Mock()
    ui.prompt_select.return_value = 'incremental'
    with (
        patch.object(installer, 'create_ui', return_value=ui),
        patch('llmb_install.core.installer.load_system_config', return_value=None),
        patch('llmb_install.core.installer.prompt_environment_type', return_value='uv'),
        patch('llmb_install.core.installer.prompt_cluster_type', return_value=selected_cluster_type),
        patch('llmb_install.core.installer.prompt_install_location', return_value=str(tmp_path)),
        patch('llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads),
        patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}),
        patch('llmb_install.core.installer.resolve_gpu_overrides', return_value=installer.workloads),
        patch('llmb_install.core.installer.prompt_workload_selection', return_value=(['pretrain_x'], 'custom')),
    ):
        result = installer._collect_configuration('simple')

    assert result.install_path == str(tmp_path)
    assert result.execution_target == 'kubernetes'
    assert result.kubernetes is not None
    assert result.kubernetes.namespace == 'ns'
    choices = ui.prompt_select.call_args.kwargs['choices']
    assert choices[0]['name'] == 'Add workloads to existing Kubernetes installation'
