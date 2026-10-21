# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for incremental flow exit-code behavior."""

from unittest.mock import patch

import pytest

from llmb_install.constants import EXIT_CANCELLED, EXIT_ERROR
from llmb_install.core.installer import Installer


def _base_cluster_config() -> dict:
    return {
        'workloads': {'installed': [], 'config': {}},
        'gpu_type': 'h100',
        'llmb_repo': '/test/repo',
        'install': {'method': 'local', 'node_architecture': 'x86_64', 'venv_type': 'uv'},
        'slurm': {},
        'environment': {},
    }


def test_collect_incremental_configuration_keyboard_interrupt_exits_cancelled():
    installer = Installer()
    installer.workloads = {'pretrain_nemotron-h': {'name': 'Nemotron-H'}}

    with patch('llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads):
        with patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}):
            with patch('llmb_install.core.installer.resolve_gpu_overrides', return_value=installer.workloads):
                with patch('llmb_install.core.installer.prompt_workload_selection', side_effect=KeyboardInterrupt):
                    with pytest.raises(SystemExit) as exc_info:
                        installer._collect_incremental_configuration(
                            'simple', '/tmp/install', _base_cluster_config(), args=None
                        )

    assert exc_info.value.code == EXIT_CANCELLED


def test_collect_incremental_configuration_empty_selection_exits_cancelled():
    installer = Installer()
    installer.workloads = {'pretrain_nemotron-h': {'name': 'Nemotron-H'}}

    with patch('llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads):
        with patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}):
            with patch('llmb_install.core.installer.resolve_gpu_overrides', return_value=installer.workloads):
                with patch('llmb_install.core.installer.prompt_workload_selection', return_value=([], 'custom')):
                    with pytest.raises(SystemExit) as exc_info:
                        installer._collect_incremental_configuration(
                            'simple', '/tmp/install', _base_cluster_config(), args=None
                        )

    assert exc_info.value.code == EXIT_CANCELLED


def test_collect_incremental_configuration_missing_node_arch_exits_error():
    installer = Installer()
    installer.workloads = {'pretrain_nemotron-h': {'name': 'Nemotron-H'}}

    cluster_config = _base_cluster_config()
    cluster_config['install'].pop('node_architecture')

    with pytest.raises(SystemExit) as exc_info:
        installer._collect_incremental_configuration('simple', '/tmp/install', cluster_config, args=None)

    assert exc_info.value.code == EXIT_ERROR


def test_collect_incremental_configuration_no_available_workloads_exits_cancelled():
    installer = Installer()
    installer.workloads = {'pretrain_nemotron-h': {'name': 'Nemotron-H'}}

    cluster_config = _base_cluster_config()
    cluster_config['workloads']['installed'] = ['pretrain_nemotron-h']

    with patch('llmb_install.core.installer.filter_workloads_by_gpu_type', return_value=installer.workloads):
        with patch('llmb_install.core.installer.filter_tools_from_workload_list', return_value={}):
            with patch('llmb_install.core.installer.resolve_gpu_overrides', return_value=installer.workloads):
                with pytest.raises(SystemExit) as exc_info:
                    installer._collect_incremental_configuration('simple', '/tmp/install', cluster_config, args=None)

    assert exc_info.value.code == EXIT_CANCELLED
