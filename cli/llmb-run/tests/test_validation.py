# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from llmb_run.workload_validator import (
    ValidationErrorType,
    validate_workload_with_details,
)


@pytest.fixture()
def _base_workloads():
    """Return a minimal synthetic workloads dictionary."""
    return {
        "training_foo": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "7b",
                                    "dtypes": ["fp8", "bf16"],
                                    "scales": [16, 32, 64, 128],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/tmp",
            "workload": "foo",
            "workload_type": "training",
        },
        "training_bar": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "gb200": {
                            "model_configs": [
                                {
                                    "model_size": "405b",
                                    "dtypes": ["fp8", "bf16"],
                                    "scales": [128, 256, 512],
                                }
                            ]
                        }
                    }
                }
            },
            "dir": "/tmp",
            "workload": "bar",
            "workload_type": "training",
        },
    }


@pytest.fixture()
def _cluster_config(cluster_config_factory):
    return cluster_config_factory({"workloads": {"installed": ["training_foo"]}})


def test_workload_not_found(_base_workloads, _cluster_config):
    """Unknown workload keys should be flagged appropriately."""
    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "unknown_workload",
        model_size="7b",
        dtype="fp8",
        scale=16,
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.WORKLOAD_NOT_FOUND
    # Suggestions should include the installed workload key
    assert "training_foo" in suggestions


def test_model_size_not_supported(_base_workloads, _cluster_config):
    """Requesting an unsupported model size should fail with the correct error type."""
    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="13b",  # Unsupported
        dtype="fp8",
        scale=16,
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.MODEL_SIZE_NOT_SUPPORTED
    # The valid model size should be suggested
    assert "7b" in suggestions


def test_dtype_not_supported(_base_workloads, _cluster_config):
    """Unsupported dtypes should be caught."""
    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="7b",
        dtype="fp32",  # Unsupported
        scale=16,
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.DTYPE_NOT_SUPPORTED
    assert "fp8" in suggestions and "bf16" in suggestions


def test_scale_not_supported(_base_workloads, _cluster_config):
    """Unsupported scales should be caught."""
    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="7b",
        dtype="fp8",
        scale=8,  # Unsupported
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.SCALE_NOT_SUPPORTED
    assert 16 in suggestions and 32 in suggestions and 64 in suggestions and 128 in suggestions


def test_per_dtype_scale_validation(cluster_config_factory):
    """Validation should use per-dtype scales when provided via mapping."""
    workloads = {
        "pretrain_custom": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "15b",
                                    "dtypes": {
                                        "fp8": [128, 256],
                                        "bf16": {"scales": [256, 512], "exact_scales": True},
                                    },
                                }
                            ]
                        }
                    }
                }
            },
            "workload_type": "pretrain",
        }
    }

    cluster_config = cluster_config_factory({"workloads": {"installed": ["pretrain_custom"]}})

    # fp8 scale 128 should be valid
    valid, _, _, _ = validate_workload_with_details(
        workloads,
        "pretrain_custom",
        model_size="15b",
        dtype="fp8",
        scale=128,
        cluster_gpu_type="h100",
        cluster_config=cluster_config,
    )
    assert valid is True

    # fp8 scale 1024 should be allowed (non-exact, warn-but-allow behavior)
    valid, _, _, _ = validate_workload_with_details(
        workloads,
        "pretrain_custom",
        model_size="15b",
        dtype="fp8",
        scale=1024,
        cluster_gpu_type="h100",
        cluster_config=cluster_config,
    )
    assert valid is True

    # bf16 exact scales: only 256, 512 allowed
    valid, err_type, _, suggestions = validate_workload_with_details(
        workloads,
        "pretrain_custom",
        model_size="15b",
        dtype="bf16",
        scale=1024,
        cluster_gpu_type="h100",
        cluster_config=cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.SCALE_NOT_SUPPORTED
    assert suggestions == [256, 512]


def test_validation_ignores_non_dtype_keys_in_mapping(cluster_config_factory):
    """Ensure accidental non-dtype keys under dtypes mapping do not break validation."""
    workloads = {
        "pretrain_custom": {
            "metadata": {
                "run": {
                    "gpu_configs": {
                        "h100": {
                            "model_configs": [
                                {
                                    "model_size": "15b",
                                    "dtypes": {
                                        "scales": [128, 256],  # Incorrectly placed, should be ignored
                                        "exact_scales": True,  # Incorrectly placed, should be ignored
                                        "fp8": [128, 256],
                                        "bf16": {"scales": [256, 512]},
                                    },
                                }
                            ]
                        }
                    }
                }
            },
            "workload_type": "pretrain",
        }
    }

    cluster_config = cluster_config_factory({"workloads": {"installed": ["pretrain_custom"]}})

    # fp8 scale 128 valid
    valid, _, _, _ = validate_workload_with_details(
        workloads,
        "pretrain_custom",
        model_size="15b",
        dtype="fp8",
        scale=128,
        cluster_gpu_type="h100",
        cluster_config=cluster_config,
    )
    assert valid is True

    # Unknown dtype should be rejected
    valid, err_type, _, suggestions = validate_workload_with_details(
        workloads,
        "pretrain_custom",
        model_size="15b",
        dtype="fp16",
        scale=128,
        cluster_gpu_type="h100",
        cluster_config=cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.DTYPE_NOT_SUPPORTED
    assert set(suggestions) == {"fp8", "bf16"}


def test_gpu_type_not_supported(_base_workloads, _cluster_config):
    """Unsupported gpu types should be caught."""
    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="7b",
        dtype="fp8",
        scale=16,
        cluster_gpu_type="v100",  # Unsupported
        cluster_config=_cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.GPU_TYPE_NOT_SUPPORTED
    assert "h100" in suggestions


def test_workload_not_installed(_base_workloads, _cluster_config):
    """Workloads that are not installed should be flagged appropriately."""
    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_bar",
        model_size="405b",
        dtype="fp8",
        scale=128,
        cluster_gpu_type="gb200",
        cluster_config=_cluster_config,
    )
    assert valid is False
    assert err_type == ValidationErrorType.WORKLOAD_NOT_INSTALLED
    assert "training_foo" in suggestions


def test_proxy_scale_validation_valid(_base_workloads, _cluster_config):
    """Test that proxy mode with valid proxy_scale passes validation."""
    # Add proxy_scales to the workload metadata
    _base_workloads["training_foo"]["metadata"]["run"]["gpu_configs"]["h100"]["model_configs"][0]["proxy_scales"] = [
        8,
        16,
    ]

    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="7b",
        dtype="fp8",
        scale=8,
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
        proxy=True,
    )
    assert valid is True


def test_proxy_scale_validation_invalid(_base_workloads, _cluster_config):
    """Test that proxy mode with non-proxy scale fails validation."""
    # Add proxy_scales to the workload metadata
    _base_workloads["training_foo"]["metadata"]["run"]["gpu_configs"]["h100"]["model_configs"][0]["proxy_scales"] = [
        8,
        16,
    ]

    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="7b",
        dtype="fp8",
        scale=32,  # Not in proxy_scales
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
        proxy=True,
    )
    assert valid is False
    assert err_type == ValidationErrorType.SCALE_NOT_SUPPORTED
    assert 8 in suggestions and 16 in suggestions


def test_proxy_mode_without_proxy_scales(_base_workloads, _cluster_config):
    """Test that proxy mode without proxy_scales defined fails validation."""
    # Ensure no proxy_scales are defined
    assert (
        "proxy_scales"
        not in _base_workloads["training_foo"]["metadata"]["run"]["gpu_configs"]["h100"]["model_configs"][0]
    )

    valid, err_type, err_msg, suggestions = validate_workload_with_details(
        _base_workloads,
        "training_foo",
        model_size="7b",
        dtype="fp8",
        scale=16,
        cluster_gpu_type="h100",
        cluster_config=_cluster_config,
        proxy=True,
    )
    assert valid is False
    assert err_type == ValidationErrorType.SCALE_NOT_SUPPORTED
