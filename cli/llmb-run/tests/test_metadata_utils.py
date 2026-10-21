# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for metadata_utils module."""

import pytest

from llmb_run.metadata_utils import RunConfig, model_size_to_billions, normalize_model_dtype_config, parse_workload_name


def test_execution_target_override_is_specific_to_gpu():
    run = RunConfig.model_validate(
        {
            'execution_targets': {'by_gpu': {'default': ['slurm', 'kubernetes'], 'h100': ['slurm']}},
        }
    )
    assert run.supports('slurm', 'h100')
    assert not run.supports('kubernetes', 'h100')
    assert run.supports('kubernetes', 'gb300')


class TestParseWorkloadName:
    """Tests for parse_workload_name function."""

    def test_parse_simple_model_size(self):
        """Parse workload with simple model size suffix."""
        workload_key, model_size = parse_workload_name('pretrain_foo_7b')
        assert workload_key == 'pretrain_foo'
        assert model_size == '7b'

    def test_parse_large_model_size(self):
        """Parse workload with large model size suffix."""
        workload_key, model_size = parse_workload_name('pretrain_bar_340b')
        assert workload_key == 'pretrain_bar'
        assert model_size == '340b'

    def test_parse_decimal_model_size(self):
        """Parse workload with decimal model size suffix."""
        workload_key, model_size = parse_workload_name('pretrain_baz_7.1b')
        assert workload_key == 'pretrain_baz'
        assert model_size == '7.1b'

    def test_parse_hyphenated_workload(self):
        """Parse workload with hyphens in name."""
        workload_key, model_size = parse_workload_name('pretrain_nemotron4-340b_340b')
        assert workload_key == 'pretrain_nemotron4-340b'
        assert model_size == '340b'

    def test_parse_complex_workload(self):
        """Parse workload with complex naming like llama3.1."""
        workload_key, model_size = parse_workload_name('pretrain_llama3.1_70b')
        assert workload_key == 'pretrain_llama3.1'
        assert model_size == '70b'

    def test_parse_without_suffix_returns_none(self):
        """Workload without model size suffix returns None for model_size."""
        workload_key, model_size = parse_workload_name('pretrain_foo')
        assert workload_key == 'pretrain_foo'
        assert model_size is None

    def test_parse_invalid_suffix_returns_none(self):
        """Invalid suffix (not matching pattern) returns None for model_size."""
        workload_key, model_size = parse_workload_name('pretrain_foo_7x')
        assert workload_key == 'pretrain_foo_7x'
        assert model_size is None

    def test_parse_no_underscore_returns_none(self):
        """Workload without underscore returns original with None."""
        workload_key, model_size = parse_workload_name('pretrainfoo')
        assert workload_key == 'pretrainfoo'
        assert model_size is None

    def test_parse_edge_case_numbers_in_base(self):
        """Workload with numbers elsewhere should still parse correctly."""
        workload_key, model_size = parse_workload_name('pretrain_silly_75b_7.1b')
        assert workload_key == 'pretrain_silly_75b'
        assert model_size == '7.1b'

    def test_parse_single_digit_model_size(self):
        """Parse single digit model sizes."""
        workload_key, model_size = parse_workload_name('pretrain_tiny_1b')
        assert workload_key == 'pretrain_tiny'
        assert model_size == '1b'

    def test_parse_three_digit_model_size(self):
        """Parse three digit model sizes."""
        workload_key, model_size = parse_workload_name('pretrain_huge_671b')
        assert workload_key == 'pretrain_huge'
        assert model_size == '671b'

    def test_parse_trillion_model_size(self):
        """Parse trillion-scale model sizes."""
        workload_key, model_size = parse_workload_name('pretrain_kimi-k2_1t')
        assert workload_key == 'pretrain_kimi-k2'
        assert model_size == '1t'

    def test_parse_uppercase_model_size_normalizes_suffix(self):
        """Parse uppercase suffixes and normalize them for metadata matching."""
        workload_key, model_size = parse_workload_name('pretrain_kimi-k2_1T')
        assert workload_key == 'pretrain_kimi-k2'
        assert model_size == '1t'


class TestModelSizeToBillions:
    """Tests for model size unit conversion."""

    def test_billion_size(self):
        assert model_size_to_billions('405b') == 405.0

    def test_trillion_size(self):
        assert model_size_to_billions('1t') == 1000.0

    def test_decimal_trillion_size(self):
        assert model_size_to_billions('1.5t') == 1500.0


class TestNormalizeModelDtypeConfig:
    """Tests for normalize_model_dtype_config function."""

    def test_legacy_list_dtypes(self):
        """Legacy form: dtypes as a list, shared scales."""
        config = {"dtypes": ["fp8", "bf16"], "scales": [128, 256]}
        result = normalize_model_dtype_config(config)
        assert set(result.keys()) == {"fp8", "bf16"}
        assert result["fp8"]["scales"] == [128, 256]
        assert result["bf16"]["scales"] == [128, 256]
        assert result["fp8"]["exact_scales"] is False

    def test_legacy_string_dtype(self):
        """Legacy form: single dtype as a string."""
        config = {"dtypes": "fp8", "scales": [64]}
        result = normalize_model_dtype_config(config)
        assert list(result.keys()) == ["fp8"]
        assert result["fp8"]["scales"] == [64]

    def test_legacy_exact_scales(self):
        """Legacy form: exact_scales propagates to all dtypes."""
        config = {"dtypes": ["fp8", "bf16"], "scales": [128], "exact_scales": True}
        result = normalize_model_dtype_config(config)
        assert result["fp8"]["exact_scales"] is True
        assert result["bf16"]["exact_scales"] is True

    def test_legacy_proxy_scales(self):
        """Legacy form: proxy_scales propagates to all dtypes."""
        config = {"dtypes": ["fp8"], "scales": [128, 256], "proxy_scales": [8, 16]}
        result = normalize_model_dtype_config(config)
        assert result["fp8"]["proxy_scales"] == [8, 16]

    def test_mapping_form_list_shorthand(self):
        """Mapping form: dtype key maps to a list (shorthand for scales only)."""
        config = {"dtypes": {"fp8": [128, 256], "bf16": [256, 512]}}
        result = normalize_model_dtype_config(config)
        assert result["fp8"]["scales"] == [128, 256]
        assert result["bf16"]["scales"] == [256, 512]

    def test_mapping_form_dict_with_per_dtype_config(self):
        """Mapping form: dtype key maps to a dict with per-dtype settings."""
        config = {
            "dtypes": {
                "fp8": {"scales": [128], "exact_scales": True, "proxy_scales": [8]},
                "bf16": {"scales": [256]},
            }
        }
        result = normalize_model_dtype_config(config)
        assert result["fp8"]["scales"] == [128]
        assert result["fp8"]["exact_scales"] is True
        assert result["fp8"]["proxy_scales"] == [8]
        # bf16 inherits top-level defaults (exact_scales=False, proxy_scales=[])
        assert result["bf16"]["exact_scales"] is False
        assert result["bf16"]["proxy_scales"] == []

    def test_mapping_form_ignores_non_dtype_keys(self):
        """Mapping form: non-dtype keys under dtypes dict are ignored."""
        config = {"dtypes": {"fp8": [128], "scales": [999], "exact_scales": True}}
        result = normalize_model_dtype_config(config)
        assert list(result.keys()) == ["fp8"]
        assert result["fp8"]["scales"] == [128]

    def test_empty_dtypes_returns_empty(self):
        """No dtypes defined returns empty dict."""
        result = normalize_model_dtype_config({})
        assert result == {}

    def test_none_dtypes_returns_empty(self):
        """dtypes=None returns empty dict."""
        result = normalize_model_dtype_config({"dtypes": None, "scales": [128]})
        assert result == {}

    @pytest.mark.parametrize("dtype", ["fp8", "bf16", "nvfp4", "mxfp4"])
    def test_all_known_dtypes_recognized(self, dtype):
        """All known dtypes are recognized in mapping form."""
        config = {"dtypes": {dtype: [128]}}
        result = normalize_model_dtype_config(config)
        assert dtype in result
