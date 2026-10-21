# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for incremental install detection logic."""

import os
from copy import deepcopy
from pathlib import Path

import yaml

from llmb_install.config.cluster import normalize_cluster_config
from llmb_install.core.installer import Installer

# Shared test data structures
VALID_CLUSTER_CONFIG = {
    'schema_version': 2,
    'gpu_type': 'h100',
    'llmb_repo': '/path/to/repo',
    'install': {
        'node_architecture': 'x86_64',
    },
    'workloads': {
        'installed': ['workload1', 'workload2'],
        'config': {
            'workload1': {'venv_path': '/path/venv1', 'venv_type': 'uv'},
            'workload2': {'venv_path': '/path/venv2', 'venv_type': 'uv'},
        },
    },
    'slurm': {
        'account': 'test-account',
        'gpu': {'partition': 'gpu'},
        'cpu': {'partition': 'cpu'},
    },
    'environment': {'VAR1': 'value1'},
}


class TestIncrementalDetection:
    """Tests for detecting incremental install candidates."""

    def test_detect_from_cwd_with_config(self, tmpdir):
        """Should detect install directory when cluster_config.yaml exists in cwd."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"
        config_file.touch()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = installer._check_if_running_from_install_dir()
            assert result == str(tmpdir)
        finally:
            os.chdir(original_cwd)

    def test_detect_from_cwd_without_config(self, tmpdir):
        """Should return None when no cluster_config.yaml in cwd."""
        installer = Installer()

        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)
            result = installer._check_if_running_from_install_dir()
            assert result is None
        finally:
            os.chdir(original_cwd)

    def test_valid_installation_detected(self, tmpdir):
        """Should detect valid installation with all required fields."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        with open(config_file, 'w') as f:
            yaml.dump(VALID_CLUSTER_CONFIG, f)

        result = installer._detect_incremental_install(str(tmpdir))

        assert result is not None
        assert result['gpu_type'] == 'h100'
        assert result['workloads']['installed'] == ['workload1', 'workload2']

    def test_missing_required_field_gpu_type(self, tmpdir):
        """Should reject config missing 'gpu_type' field."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        invalid_config = deepcopy(VALID_CLUSTER_CONFIG)
        del invalid_config['gpu_type']

        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        result = installer._detect_incremental_install(str(tmpdir))
        assert result is None

    def test_missing_required_field_workloads(self, tmpdir):
        """Should reject config missing 'workloads' field."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        invalid_config = deepcopy(VALID_CLUSTER_CONFIG)
        del invalid_config['workloads']

        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        result = installer._detect_incremental_install(str(tmpdir))
        assert result is None

    def test_missing_required_field_install(self, tmpdir):
        """Should reject config missing 'install' field (wrong arch would cause silent bugs)."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        invalid_config = deepcopy(VALID_CLUSTER_CONFIG)
        del invalid_config['install']

        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        result = installer._detect_incremental_install(str(tmpdir))
        assert result is None

    def test_missing_required_field_slurm(self, tmpdir):
        """Should reject config missing 'slurm' field."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        invalid_config = deepcopy(VALID_CLUSTER_CONFIG)
        del invalid_config['slurm']

        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        result = installer._detect_incremental_install(str(tmpdir))
        assert result is None

    def test_workloads_not_list(self, tmpdir):
        """Should reject config where workloads.installed is not a list."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        invalid_config = deepcopy(VALID_CLUSTER_CONFIG)
        invalid_config['workloads']['installed'] = 'not-a-list'

        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)

        result = installer._detect_incremental_install(str(tmpdir))
        assert result is None

    def test_nonexistent_path(self):
        """Should return None for nonexistent path."""
        installer = Installer()
        result = installer._detect_incremental_install('/nonexistent/path')
        assert result is None

    def test_corrupted_yaml(self, tmpdir):
        """Should return None for corrupted YAML file."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        with open(config_file, 'w') as f:
            f.write("invalid: yaml: content: [\n")

        result = installer._detect_incremental_install(str(tmpdir))
        assert result is None


class TestLlmbRepoValidation:
    """Tests for llmb_repo field extraction (validation happens separately)."""

    def test_llmb_repo_present_in_config(self, tmpdir):
        """Should extract llmb_repo from config when present."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"
        llmb_repo = Path(tmpdir) / "llmb_repo"
        llmb_repo.mkdir()

        config = deepcopy(VALID_CLUSTER_CONFIG)
        config['llmb_repo'] = str(llmb_repo)

        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        cluster_config = installer._detect_incremental_install(str(tmpdir))
        assert cluster_config is not None
        assert cluster_config['llmb_repo'] == str(llmb_repo)

    def test_llmb_repo_optional_for_detection(self, tmpdir):
        """Detection succeeds even without llmb_repo (validation happens later)."""
        installer = Installer()
        config_file = Path(tmpdir) / "cluster_config.yaml"

        config = deepcopy(VALID_CLUSTER_CONFIG)
        del config['llmb_repo']

        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        cluster_config = installer._detect_incremental_install(str(tmpdir))
        # Detection passes - actual validation happens in run() method
        assert cluster_config is not None
        assert 'llmb_repo' not in cluster_config


class TestNormalizeClusterConfig:
    """Tests for normalize_cluster_config() v1 → v2 conversion."""

    def test_v1_to_v2_conversion(self):
        """Should convert a complete v1 config to v2 format."""
        v1_config = {
            'launcher': {
                'llmb_repo': '/path/to/repo',
                'llmb_install': '/path/to/install',
                'gpu_type': 'h100',
                'cluster_name': 'my-cluster',
                'node_architecture': 'x86_64',
            },
            'workloads': {
                'installed': ['workload1'],
                'config': {},
            },
            'slurm': {
                'account': 'my-account',
                'gpu_partition': 'gpu-part',
                'gpu_partition_gres': 8,
                'cpu_partition': 'cpu-part',
                'cpu_partition_gres': None,
            },
            'environment': {'KEY': 'val'},
        }

        result = normalize_cluster_config(v1_config)

        assert 'launcher' not in result
        assert result['schema_version'] == 2
        assert result['gpu_type'] == 'h100'
        assert result['llmb_repo'] == '/path/to/repo'
        assert result['llmb_install'] == '/path/to/install'
        assert result['cluster_name'] == 'my-cluster'
        assert result['install']['node_architecture'] == 'x86_64'
        assert result['slurm']['account'] == 'my-account'
        assert result['slurm']['gpu'] == {'partition': 'gpu-part', 'gres': 8}
        assert result['slurm']['cpu'] == {'partition': 'cpu-part', 'gres': None}

    def test_v2_passthrough(self):
        """Should return v2 config unchanged."""
        v2_config = deepcopy(VALID_CLUSTER_CONFIG)
        result = normalize_cluster_config(v2_config)
        assert result is v2_config

    def test_v1_missing_optionals(self):
        """Should handle v1 config with missing optional fields gracefully."""
        v1_config = {
            'launcher': {
                'gpu_type': 'a100',
                'node_architecture': 'arm64',
            },
            'workloads': {'installed': [], 'config': {}},
            'slurm': {
                'gpu_partition': 'gpu',
                'cpu_partition': 'cpu',
            },
        }

        result = normalize_cluster_config(v1_config)

        assert 'launcher' not in result
        assert result['schema_version'] == 2
        assert result['gpu_type'] == 'a100'
        assert 'cluster_name' not in result
        assert result['install']['node_architecture'] == 'arm64'
        assert result['slurm']['gpu']['gres'] is None
        assert result['slurm']['cpu']['gres'] is None

    def test_v1_node_arch_not_overwritten_if_in_install(self):
        """Should not overwrite install.node_architecture if already present."""
        v1_config = {
            'launcher': {
                'gpu_type': 'h100',
                'node_architecture': 'x86_64',
            },
            'install': {
                'node_architecture': 'arm64',
            },
            'workloads': {'installed': [], 'config': {}},
            'slurm': {},
        }

        result = normalize_cluster_config(v1_config)

        assert result['install']['node_architecture'] == 'arm64'
