import pytest

from llmb_run.metadata_utils import model_size_to_billions, normalize_model_dtype_config, parse_workload_name


@pytest.mark.parametrize(
    "name, expected",
    [
        ("pretrain_foo_7b", ("pretrain_foo", "7b")),
        ("pretrain_llama3.1_70b", ("pretrain_llama3.1", "70b")),
        ("pretrain_kimi-k2_1t", ("pretrain_kimi-k2", "1t")),
        ("pretrain_foo_3.5B", ("pretrain_foo", "3.5b")),
        ("pretrain_baz", ("pretrain_baz", None)),
        ("pretrain_invalid_7x", ("pretrain_invalid_7x", None)),
        ("pretrain_foo_b", ("pretrain_foo_b", None)),
        ("nounderscore", ("nounderscore", None)),
    ],
)
def test_parse_workload_name(name, expected):
    assert parse_workload_name(name) == expected


@pytest.mark.parametrize(
    "size, expected",
    [("70b", 70.0), ("1t", 1000.0), ("3.5B", 3.5), ("bogus", 0.0)],
)
def test_model_size_to_billions(size, expected):
    assert model_size_to_billions(size) == pytest.approx(expected)


def test_normalize_legacy_list_form():
    cfg = {"dtypes": ["fp8", "bf16"], "scales": [128, "256"], "exact_scales": True, "proxy_scales": [8]}
    assert normalize_model_dtype_config(cfg) == {
        "fp8": {"scales": [128, 256], "exact_scales": True, "proxy_scales": [8]},
        "bf16": {"scales": [128, 256], "exact_scales": True, "proxy_scales": [8]},
    }


def test_normalize_legacy_string_form_and_defaults():
    assert normalize_model_dtype_config({"dtypes": "fp8", "scales": [64]}) == {
        "fp8": {"scales": [64], "exact_scales": False, "proxy_scales": []}
    }


def test_normalize_no_dtypes_is_empty():
    assert normalize_model_dtype_config({"scales": [1]}) == {}
    assert normalize_model_dtype_config({"dtypes": None}) == {}


def test_normalize_mapping_short_form_inherits_model_level_flags():
    cfg = {"dtypes": {"fp8": [128, 256]}, "exact_scales": True, "proxy_scales": [16], "scales": [999]}
    assert normalize_model_dtype_config(cfg) == {
        "fp8": {"scales": [128, 256], "exact_scales": True, "proxy_scales": [16]}
    }


def test_normalize_mapping_long_form_overrides_per_dtype():
    cfg = {
        "dtypes": {"bf16": {"scales": [256], "exact_scales": True, "proxy_scales": [4]}, "fp8": {"scales": [8]}},
        "exact_scales": False,
        "proxy_scales": [16],
    }
    result = normalize_model_dtype_config(cfg)
    assert result["bf16"] == {"scales": [256], "exact_scales": True, "proxy_scales": [4]}
    assert result["fp8"] == {"scales": [8], "exact_scales": False, "proxy_scales": [16]}


def test_normalize_mapping_ignores_non_dtype_keys_and_bad_values():
    cfg = {"dtypes": {"scales": [1], "exact_scales": True, "fp8": [8], "bf16": "oops"}}
    assert list(normalize_model_dtype_config(cfg)) == ["fp8"]
