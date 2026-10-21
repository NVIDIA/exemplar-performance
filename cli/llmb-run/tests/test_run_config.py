# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import tempfile
from unittest.mock import patch

import pytest
import yaml

from llmb_run.run_config import (
    config_to_bytes,
    create_llmb_config,
    find_experiment_config_file,
    generate_experiment_id,
    remove_nsys_callback_from_config,
    resolve_container_images,
)
from llmb_run.tasks import WorkloadTask

# ============================================================================
# Fixtures for code reuse
# ============================================================================


@pytest.fixture
def temp_repo():
    """Create temporary directory with release.yaml."""
    with tempfile.TemporaryDirectory() as temp_dir:
        release_yaml_path = os.path.join(temp_dir, 'release.yaml')
        with open(release_yaml_path, 'w') as f:
            yaml.dump({'llmb_version': '25.10'}, f)
        yield temp_dir


@pytest.fixture
def mock_cluster_config(temp_repo, cluster_config_factory):
    """Create standard mock cluster config."""
    return cluster_config_factory(
        {
            'launcher': {
                'llmb_install': '/test/llmb_install',
                'cluster_name': 'test_cluster',
                'gpu_type': 'h100',
                'llmb_repo': temp_repo,
            },
            'slurm': {
                'account': 'test_account',
                'gpu_partition': 'test_partition',
                'gpu_gres': '8',
            },
        }
    )


@pytest.fixture
def mock_workloads():
    """Create standard mock workloads structure."""
    return {
        'test_workload': {
            'metadata': {
                'general': {'framework': 'nemo2', 'gsw_version': '25.07'},
                'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
            },
            'workload': 'test_model',
            'workload_type': 'train',
            'synthetic_dataset': True,
        }
    }


def create_experiment_config(temp_dir):
    """Helper to create experiment config directory and file."""
    configs_dir = os.path.join(temp_dir, 'configs')
    os.makedirs(configs_dir)

    config_data = {
        'trainer': {'callbacks': [{'_target_': 'nemo.utils.exp_manager.TimingCallback'}]},
        'model': {'hidden_size': 1024},
    }

    config_file_path = os.path.join(configs_dir, 'test_config.yaml')
    with open(config_file_path, 'w') as f:
        yaml.dump(config_data, f)

    return configs_dir


def test_find_experiment_config_file_success():
    """Test successful finding of experiment config file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configs directory and config file
        configs_dir = os.path.join(temp_dir, 'configs')
        os.makedirs(configs_dir)

        config_file_path = os.path.join(configs_dir, 'test_experiment_config.yaml')
        with open(config_file_path, 'w') as f:
            f.write("test: config")

        result = find_experiment_config_file(temp_dir)
        assert result == config_file_path


def test_find_experiment_config_file_no_configs_dir():
    """Test when configs directory doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        result = find_experiment_config_file(temp_dir)
        assert result is None


def test_find_experiment_config_file_no_config_file():
    """Test when configs directory exists but no *_config.yaml file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        configs_dir = os.path.join(temp_dir, 'configs')
        os.makedirs(configs_dir)

        # Create non-matching file
        with open(os.path.join(configs_dir, 'other.yaml'), 'w') as f:
            f.write("other: file")

        result = find_experiment_config_file(temp_dir)
        assert result is None


def test_remove_nsys_callback_from_config():
    """Test removal of NsysCallback from trainer callbacks."""
    config_data = {
        'trainer': {
            'callbacks': [
                {'_target_': 'nemo.utils.exp_manager.TimingCallback'},
                {
                    '_target_': 'nemo.lightning.pytorch.callbacks.nsys.NsysCallback',
                    'start_step': 45,
                    'end_step': 50,
                    'ranks': [0, 1, 2],
                },
                {
                    '_target_': 'nemo.lightning.pytorch.callbacks.garbage_collection.GarbageCollectionCallback',
                    'gc_interval_train': 100,
                },
            ]
        }
    }

    # Function modifies in place and returns None
    result = remove_nsys_callback_from_config(config_data)
    assert result is None

    # Should have 2 callbacks remaining in the original dict (removed NsysCallback)
    assert len(config_data['trainer']['callbacks']) == 2

    # Verify NsysCallback was removed
    callback_targets = [cb['_target_'] for cb in config_data['trainer']['callbacks']]
    assert 'nemo.lightning.pytorch.callbacks.nsys.NsysCallback' not in callback_targets
    assert 'nemo.utils.exp_manager.TimingCallback' in callback_targets
    assert 'nemo.lightning.pytorch.callbacks.garbage_collection.GarbageCollectionCallback' in callback_targets


def test_remove_nsys_callback_no_trainer():
    """Test when config has no trainer section."""
    config_data = {'other': 'data'}
    original_data = config_data.copy()
    result = remove_nsys_callback_from_config(config_data)
    assert result is None
    assert config_data == original_data


def test_remove_nsys_callback_no_callbacks():
    """Test when trainer has no callbacks."""
    config_data = {'trainer': {'other': 'setting'}}
    original_data = config_data.copy()
    result = remove_nsys_callback_from_config(config_data)
    assert result is None
    assert config_data == original_data


def test_config_to_bytes():
    """Test config conversion to normalized bytes."""
    config1 = {'b': 2, 'a': 1, 'c': {'z': 3, 'y': 4}}
    config2 = {'a': 1, 'c': {'y': 4, 'z': 3}, 'b': 2}

    # Both should produce the same normalized bytes
    bytes1 = config_to_bytes(config1)
    bytes2 = config_to_bytes(config2)

    assert bytes1 == bytes2
    assert bytes1 == b'{"a":1,"b":2,"c":{"y":4,"z":3}}'


def test_generate_experiment_id_success():
    """Test successful experiment ID generation with required fw_version."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configs directory and config file
        configs_dir = os.path.join(temp_dir, 'configs')
        os.makedirs(configs_dir)

        config_data = {
            'trainer': {
                'callbacks': [
                    {'_target_': 'nemo.utils.exp_manager.TimingCallback'},
                    {
                        '_target_': 'nemo.lightning.pytorch.callbacks.nsys.NsysCallback',
                        'start_step': 45,
                        'end_step': 50,
                    },
                ]
            },
            'model': {'hidden_size': 1024},
        }

        config_file_path = os.path.join(configs_dir, 'test_config.yaml')
        with open(config_file_path, 'w') as f:
            yaml.dump(config_data, f)

        experiment_id = generate_experiment_id(temp_dir, fw_version='25.04.01')

        # Should return a valid SHA256 hash
        assert experiment_id is not None
        assert len(experiment_id) == 64  # SHA256 hex length
        assert all(c in '0123456789abcdef' for c in experiment_id)


def test_generate_experiment_id_byte_concatenation():
    """Test that the byte concatenation approach works correctly."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create configs directory and config file
        configs_dir = os.path.join(temp_dir, 'configs')
        os.makedirs(configs_dir)

        config_data = {
            'trainer': {'callbacks': [{'_target_': 'nemo.utils.exp_manager.TimingCallback'}]},
            'model': {'hidden_size': 1024},
        }

        config_file_path = os.path.join(configs_dir, 'test_config.yaml')
        with open(config_file_path, 'w') as f:
            yaml.dump(config_data, f)

        # Test that the byte concatenation produces different results for different fw_versions
        experiment_id_1 = generate_experiment_id(temp_dir, fw_version='25.04.01')
        experiment_id_2 = generate_experiment_id(temp_dir, fw_version='25.05.01')

        # Should return valid SHA256 hashes that are different
        assert experiment_id_1 is not None
        assert experiment_id_2 is not None
        assert len(experiment_id_1) == 64  # SHA256 hex length
        assert len(experiment_id_2) == 64  # SHA256 hex length
        assert all(c in '0123456789abcdef' for c in experiment_id_1)
        assert all(c in '0123456789abcdef' for c in experiment_id_2)
        assert experiment_id_1 != experiment_id_2


def test_generate_experiment_id_no_config():
    """Test experiment ID generation when config file doesn't exist."""
    with tempfile.TemporaryDirectory() as temp_dir:
        experiment_id = generate_experiment_id(temp_dir, fw_version='25.04.01')
        assert experiment_id is None


def test_generate_experiment_id_consistency():
    """Test that same config produces same experiment ID."""
    config_data = {
        'trainer': {
            'callbacks': [
                {'_target_': 'nemo.utils.exp_manager.TimingCallback'},
                {'_target_': 'nemo.lightning.pytorch.callbacks.nsys.NsysCallback', 'start_step': 45},
            ]
        },
        'model': {'size': 'large'},
    }

    experiment_ids = []

    for _ in range(2):
        with tempfile.TemporaryDirectory() as temp_dir:
            configs_dir = os.path.join(temp_dir, 'configs')
            os.makedirs(configs_dir)

            config_file_path = os.path.join(configs_dir, 'test_config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(config_data, f)

            experiment_id = generate_experiment_id(temp_dir, fw_version='25.04.01')
            experiment_ids.append(experiment_id)

    # Both should be the same
    assert experiment_ids[0] == experiment_ids[1]


def test_generate_experiment_id_fw_version_affects_hash():
    """Test that different fw_versions produce different experiment IDs."""
    config_data = {
        'trainer': {'callbacks': [{'_target_': 'nemo.utils.exp_manager.TimingCallback'}]},
        'model': {'size': 'large'},
    }

    experiment_ids = []

    for fw_version in ['25.04.01', '25.05.01']:
        with tempfile.TemporaryDirectory() as temp_dir:
            configs_dir = os.path.join(temp_dir, 'configs')
            os.makedirs(configs_dir)

            config_file_path = os.path.join(configs_dir, 'test_config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(config_data, f)

            experiment_id = generate_experiment_id(temp_dir, fw_version=fw_version)
            experiment_ids.append(experiment_id)

    # Different fw_versions should produce different hashes
    assert experiment_ids[0] != experiment_ids[1]


def test_generate_experiment_id_nsys_removal():
    """Test that NsysCallback removal affects the hash as expected."""
    base_config = {
        'trainer': {'callbacks': [{'_target_': 'nemo.utils.exp_manager.TimingCallback'}]},
        'model': {'size': 'large'},
    }

    config_with_nsys = {
        'trainer': {
            'callbacks': [
                {'_target_': 'nemo.utils.exp_manager.TimingCallback'},
                {'_target_': 'nemo.lightning.pytorch.callbacks.nsys.NsysCallback', 'start_step': 45, 'end_step': 50},
            ]
        },
        'model': {'size': 'large'},
    }

    experiment_ids = []

    for config_data in [base_config, config_with_nsys]:
        with tempfile.TemporaryDirectory() as temp_dir:
            configs_dir = os.path.join(temp_dir, 'configs')
            os.makedirs(configs_dir)

            config_file_path = os.path.join(configs_dir, 'test_config.yaml')
            with open(config_file_path, 'w') as f:
                yaml.dump(config_data, f)

            experiment_id = generate_experiment_id(temp_dir, fw_version='25.04.01')
            experiment_ids.append(experiment_id)

    # Both should produce the same hash since NsysCallback is removed
    assert experiment_ids[0] == experiment_ids[1]


def test_resolve_container_images_simple_list():
    """Test resolving images from a simple list of strings."""
    images_field = ['nvcr.io#nvidia/nemo:25.04.01', 'nvcr.io#nvidia/pytorch:24.12-py3']
    result = resolve_container_images(images_field, 'h100')
    assert result == ['nvcr.io#nvidia/nemo:25.04.01', 'nvcr.io#nvidia/pytorch:24.12-py3']


def test_resolve_container_images_dict_with_url():
    """Test resolving images from a list of dicts with 'url' key."""
    images_field = [
        {'url': 'nvcr.io#nvidia/nemo:25.04.01', 'name': 'my-custom-name.sqsh'},
        {'url': 'nvcr.io#nvidia/pytorch:24.12-py3'},
    ]
    result = resolve_container_images(images_field, 'h100')
    assert result == ['nvcr.io#nvidia/nemo:25.04.01', 'nvcr.io#nvidia/pytorch:24.12-py3']


def test_resolve_container_images_by_gpu_specific_match():
    """Test resolving images with by_gpu structure when GPU type matches."""
    images_field = {
        'by_gpu': {
            'h100': ['nvcr.io#nvidia/nemo:25.01'],
            'gb200': ['nvcr.io#nvidia/nemo:25.05'],
            'default': ['nvcr.io#nvidia/nemo:25.07.01'],
        }
    }
    result = resolve_container_images(images_field, 'h100')
    assert result == ['nvcr.io#nvidia/nemo:25.01']

    result = resolve_container_images(images_field, 'gb200')
    assert result == ['nvcr.io#nvidia/nemo:25.05']


def test_resolve_container_images_by_gpu_default_fallback():
    """Test resolving images with by_gpu structure falling back to default."""
    images_field = {
        'by_gpu': {
            'h100': ['nvcr.io#nvidia/nemo:25.01'],
            'gb200': ['nvcr.io#nvidia/nemo:25.05'],
            'default': ['nvcr.io#nvidia/nemo:25.07.01'],
        }
    }
    # b200 not in the dict, should fall back to default
    result = resolve_container_images(images_field, 'b200')
    assert result == ['nvcr.io#nvidia/nemo:25.07.01']


def test_resolve_container_images_by_gpu_no_default():
    """Test resolving images with by_gpu structure when no match and no default."""
    images_field = {'by_gpu': {'h100': ['nvcr.io#nvidia/nemo:25.01'], 'gb200': ['nvcr.io#nvidia/nemo:25.05']}}
    # b200 not in the dict and no default key
    result = resolve_container_images(images_field, 'b200')
    assert result == []


def test_resolve_container_images_by_gpu_with_dict_items():
    """Test resolving images with by_gpu structure containing dict items."""
    images_field = {
        'by_gpu': {
            'h100': [{'url': 'nvcr.io#nvidia/nemo:25.01', 'name': 'nemo-h100.sqsh'}],
            'default': [{'url': 'nvcr.io#nvidia/nemo:25.07.01'}],
        }
    }
    result = resolve_container_images(images_field, 'h100')
    assert result == ['nvcr.io#nvidia/nemo:25.01']

    result = resolve_container_images(images_field, 'b200')
    assert result == ['nvcr.io#nvidia/nemo:25.07.01']


def test_resolve_container_images_empty_input():
    """Test resolving images with empty/None input."""
    assert resolve_container_images(None, 'h100') == []
    assert resolve_container_images([], 'h100') == []
    assert resolve_container_images({}, 'h100') == []


def test_resolve_container_images_mixed_list():
    """Test resolving images from a list with mixed string and dict items."""
    images_field = [
        'nvcr.io#nvidia/nemo:25.04.01',
        {'url': 'nvcr.io#nvidia/pytorch:24.12-py3', 'name': 'custom.sqsh'},
    ]
    result = resolve_container_images(images_field, 'h100')
    assert result == ['nvcr.io#nvidia/nemo:25.04.01', 'nvcr.io#nvidia/pytorch:24.12-py3']


def test_create_llmb_config_includes_experiment_id(temp_repo, mock_cluster_config, mock_workloads):
    """Test that create_llmb_config includes experiment_id in job_info."""
    create_experiment_config(temp_repo)

    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)

    assert config_file_path is not None
    assert os.path.exists(config_file_path)

    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify experiment_id is present and is a valid SHA256 hash
    experiment_id = llmb_config_data['job_info']['experiment_id']
    assert experiment_id is not None
    assert len(experiment_id) == 64
    assert all(c in '0123456789abcdef' for c in experiment_id)


def test_create_llmb_config_no_experiment_config(temp_repo, mock_cluster_config, mock_workloads):
    """Test create_llmb_config when experiment config is not found."""
    # No experiment config created - should result in experiment_id being None
    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)

    assert config_file_path is not None
    assert os.path.exists(config_file_path)

    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # experiment_id should be None when config is not found
    assert llmb_config_data['job_info']['experiment_id'] is None


def test_create_llmb_config_with_model_attribute(temp_repo, mock_cluster_config):
    """Test that general.model attribute is used for model_name when present."""
    # Create workloads with model attribute different from workload
    workloads = {
        'test_workload': {
            'metadata': {
                'general': {
                    'framework': 'nemo2',
                    'gsw_version': '25.07',
                    'model': 'llama3',  # Override model name
                },
                'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
            },
            'workload': 'llama3.3',  # Different from model
            'workload_type': 'pretrain',
            'synthetic_dataset': True,
        }
    }

    task = WorkloadTask(workload_key="test_workload", model_size="70b", dtype="fp8", scale=64, profile=False)
    config_file_path = create_llmb_config(task, "55555", temp_repo, mock_cluster_config, workloads)

    assert config_file_path is not None
    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify model_name uses the 'model' attribute instead of 'workload'
    assert llmb_config_data['model_info']['model_name'] == 'llama3'


def test_create_llmb_config_without_model_attribute(temp_repo, mock_cluster_config):
    """Test that workload is used for model_name when general.model is not present."""
    # Create workloads WITHOUT model attribute - should fall back to workload
    workloads = {
        'test_workload': {
            'metadata': {
                'general': {
                    'framework': 'nemo2',
                    'gsw_version': '25.07',
                    # No 'model' attribute here
                },
                'container': {'images': ['nvcr.io#nvidia/nemo:25.04.01']},
            },
            'workload': 'llama3.1',
            'workload_type': 'pretrain',
            'synthetic_dataset': True,
        }
    }

    task = WorkloadTask(workload_key="test_workload", model_size="405b", dtype="bf16", scale=512, profile=False)
    config_file_path = create_llmb_config(task, "66666", temp_repo, mock_cluster_config, workloads)

    assert config_file_path is not None
    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify model_name falls back to 'workload'
    assert llmb_config_data['model_info']['model_name'] == 'llama3.1'


def test_create_llmb_config_with_by_gpu_images(temp_repo, cluster_config_factory):
    """Test create_llmb_config with by_gpu image structure."""
    create_experiment_config(temp_repo)

    # Override only gpu_type for this scenario.
    config = cluster_config_factory({'launcher': {'gpu_type': 'gb200', 'cluster_name': 'test_cluster'}})

    # Create workloads with by_gpu image structure
    workloads = {
        'test_workload': {
            'metadata': {
                'general': {'framework': 'nemo2', 'gsw_version': '25.10'},
                'container': {
                    'images': {
                        'by_gpu': {
                            'default': ['nvcr.io#nvidia/nemo:25.09'],
                            'gb200': ['nvcr.io#nvidia/nemo:25.07.01'],
                            'b200': ['nvcr.io#nvidia/nemo:25.07.01'],
                        }
                    }
                },
            },
            'workload': 'nemotron4',
            'workload_type': 'pretrain',
            'synthetic_dataset': True,
        }
    }

    task = WorkloadTask(workload_key="test_workload", model_size="340b", dtype="bf16", scale=256, profile=False)
    config_file_path = create_llmb_config(task, "12345", temp_repo, config, workloads)

    assert config_file_path is not None
    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify resolved images for gb200
    assert llmb_config_data['container_info']['images'] == ['nvcr.io#nvidia/nemo:25.07.01']
    assert llmb_config_data['workload_info']['fw_version'] == '25.07.01'
    assert llmb_config_data['job_info']['experiment_id'] is not None


def test_create_llmb_config_with_by_gpu_images_default_fallback(temp_repo, mock_cluster_config):
    """Test create_llmb_config with by_gpu falling back to default."""
    # Create workloads with by_gpu structure (h100 not in list, should use default)
    workloads = {
        'test_workload': {
            'metadata': {
                'general': {'framework': 'nemo2', 'gsw_version': '25.10'},
                'container': {
                    'images': {
                        'by_gpu': {
                            'default': ['nvcr.io#nvidia/nemo:25.09'],
                            'gb200': ['nvcr.io#nvidia/nemo:25.07.01'],
                            'b200': ['nvcr.io#nvidia/nemo:25.07.01'],
                        }
                    }
                },
            },
            'workload': 'nemotron4',
            'workload_type': 'pretrain',
            'synthetic_dataset': True,
        }
    }

    task = WorkloadTask(workload_key="test_workload", model_size="340b", dtype="bf16", scale=256, profile=False)
    config_file_path = create_llmb_config(task, "12346", temp_repo, mock_cluster_config, workloads)

    assert config_file_path is not None
    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify it used the default image for h100
    assert llmb_config_data['container_info']['images'] == ['nvcr.io#nvidia/nemo:25.09']
    assert llmb_config_data['workload_info']['fw_version'] == '25.09'


def test_create_llmb_config_with_proxy_true(temp_repo, mock_cluster_config, mock_workloads):
    """Test that create_llmb_config includes proxy: true in job_config."""
    create_experiment_config(temp_repo)

    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False, proxy=True)
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)

    assert config_file_path is not None
    assert os.path.exists(config_file_path)

    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify job_config.proxy is true
    assert llmb_config_data['job_config']['proxy'] is True


def test_create_llmb_config_with_proxy_false(temp_repo, mock_cluster_config, mock_workloads):
    """Test that create_llmb_config includes proxy: false in job_config."""
    create_experiment_config(temp_repo)

    task = WorkloadTask(
        workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False, proxy=False
    )
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)

    assert config_file_path is not None
    assert os.path.exists(config_file_path)

    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    # Verify job_config.proxy is false
    assert llmb_config_data['job_config']['proxy'] is False


def test_create_llmb_config_includes_mbridge_args(temp_repo, mock_cluster_config, mock_workloads):
    """Persist raw MBridge arguments for debugging and reproducibility."""
    create_experiment_config(temp_repo)

    task = WorkloadTask(
        workload_key="test_workload",
        model_size="1b",
        dtype="bf16",
        scale=8,
        mbridge_args=("--micro_batch_size=2", "model.hidden_size=4096"),
    )
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)

    assert config_file_path is not None
    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    assert llmb_config_data['job_config']['mbridge_args'] == [
        '--micro_batch_size=2',
        'model.hidden_size=4096',
    ]


def test_create_llmb_config_cluster_info_slurm_fields(temp_repo, mock_cluster_config, mock_workloads):
    """Test that cluster_info contains correct slurm_account, partitions, and gpus_per_node."""
    create_experiment_config(temp_repo)

    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)

    assert config_file_path is not None

    with open(config_file_path, 'r') as f:
        llmb_config_data = yaml.safe_load(f)

    cluster_info = llmb_config_data['cluster_info']
    assert cluster_info['slurm_account'] == 'test_account'
    assert cluster_info['slurm_gpu_partition'] == 'test_partition'
    assert cluster_info['gpus_per_node'] == '8'
    assert cluster_info['llmb_install'] == '/test/llmb_install'
    assert cluster_info['llmb_repo'] == temp_repo


def test_create_llmb_config_cluster_info_nvlink_fields(temp_repo, cluster_config_factory, mock_workloads):
    config = cluster_config_factory(
        {
            "launcher": {
                "gpu_type": "gb200",
                "llmb_install": "/test/llmb_install",
                "llmb_repo": temp_repo,
                "cluster_name": "test_cluster",
            },
            "nvl_domain": 72,
            "nvl_block_sizes": [18],
        }
    )
    create_experiment_config(temp_repo)
    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)
    config_file_path = create_llmb_config(task, "12345", temp_repo, config, mock_workloads)
    with open(config_file_path, 'r') as f:
        cluster_info = yaml.safe_load(f)['cluster_info']
    assert cluster_info['nvl_domain'] == 72
    assert cluster_info['nvl_block_sizes'] == [18]


def test_create_llmb_config_emits_null_for_unset_nvlink_fields(temp_repo, mock_cluster_config, mock_workloads):
    create_experiment_config(temp_repo)
    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)
    config_file_path = create_llmb_config(task, "12345", temp_repo, mock_cluster_config, mock_workloads)
    with open(config_file_path, 'r') as f:
        cluster_info = yaml.safe_load(f)['cluster_info']
    assert cluster_info['nvl_domain'] is None
    assert cluster_info['nvl_block_sizes'] is None


def test_create_llmb_config_requires_cluster_name_on_slurm(temp_repo, cluster_config_factory, mock_workloads):
    """On Slurm a missing cluster name must refuse the run, not silently skip submission."""
    config = cluster_config_factory(
        {
            'launcher': {
                'llmb_install': '/test/llmb_install',
                'cluster_name': None,
                'gpu_type': 'h100',
                'llmb_repo': temp_repo,
            }
        }
    )
    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)

    with (
        patch('llmb_run.run_config._internal_modules_available', True),
        patch('llmb_run.run_config.get_cluster_name', return_value=None),
    ):
        assert create_llmb_config(task, "12345", temp_repo, config, mock_workloads) is None


def test_create_llmb_config_allows_missing_cluster_name_on_kubernetes(
    temp_repo, cluster_config_factory, mock_workloads
):
    """A Kubernetes install has no Slurm cluster name; the run must proceed."""
    config = cluster_config_factory(
        {
            'launcher': {
                'llmb_install': '/test/llmb_install',
                'cluster_name': None,
                'gpu_type': 'gb300',
                'llmb_repo': temp_repo,
            },
            'execution_target': 'kubernetes',
            'kubernetes': {'namespace': 'ns'},
        }
    )
    task = WorkloadTask(workload_key="test_workload", model_size="1b", dtype="bf16", scale=8, profile=False)

    with (
        patch('llmb_run.run_config._internal_modules_available', True),
        patch('llmb_run.run_config.get_cluster_name', return_value=None),
    ):
        config_file_path = create_llmb_config(task, "12345", temp_repo, config, mock_workloads)

    assert config_file_path is not None
