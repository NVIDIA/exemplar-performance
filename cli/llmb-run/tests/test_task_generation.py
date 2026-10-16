import pytest

from llmb_run.task_generation import (
    TaskGenerationRequest,
    ValidationError,
    _generate_scales_up_to_max,
    generate_submit_all_tasks,
    generate_tasks,
    parse_comma_list,
)


@pytest.mark.parametrize(
    "scales, max_scale, exact, expected",
    [
        ([], 512, False, []),
        ([8, 16, 32], None, False, [8, 16, 32]),
        ([8, 16, 32], 16, False, [8, 16]),
        ([32, 8, 16], 512, True, [8, 16, 32]),
        ([8, 16, 32], 512, False, [8, 16, 32, 64, 128, 256, 512]),
        ([8, 16, 32], 100, False, [8, 16, 32, 64]),
        ([8, 24], 100, False, [8, 24, 32, 64]),
        ([8, 16], 4, False, []),
        (["8", "16"], 16, False, [8, 16]),
    ],
)
def test_generate_scales_up_to_max(scales, max_scale, exact, expected):
    assert _generate_scales_up_to_max(scales, max_scale, exact) == expected


def test_parse_comma_list():
    assert parse_comma_list(None) == []
    assert parse_comma_list("") == []
    assert parse_comma_list(" a, b ,,c ") == ["a", "b", "c"]


def request(make_cluster_config, **kwargs):
    return TaskGenerationRequest(workloads={}, cluster_config=make_cluster_config(), **kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({}, "Must specify scale"),
        ({"file_path": "t.yaml", "workload": "w", "scale": "8"}, "Cannot mix --file"),
        ({"model_size": "7b", "scale": "8"}, "requires --workload"),
        ({"workload": "a,b", "model_size": "7b", "scale": "8"}, "multiple workloads"),
        ({"workload": "pretrain_x_7b", "model_size": "70b", "scale": "8"}, "implies size '7b'"),
        ({"workload": "w", "scale": "8", "max_scale": 16}, "Cannot use --scale"),
        ({"workload": "w", "scale": "8,16", "dtype": "fp8", "model_size": "7b", "force": True}, "single values"),
        ({"workload": "w", "max_scale": 8, "dtype": "fp8", "model_size": "7b", "force": True}, "only supported"),
        ({"workload": "w", "scale": "8", "dtype": "fp8", "force": True}, "only supported"),
        ({"workload": "w", "scale": "8", "model_size": "7b", "force": True}, "only supported"),
        ({"file_path": "t.yaml", "force": True}, "only supported"),
    ],
)
def test_validate_rejects(make_cluster_config, kwargs, match):
    with pytest.raises(ValidationError, match=match):
        request(make_cluster_config, **kwargs).validate()


def test_validate_normalizes_model_size_and_strips_redundant_suffix(make_cluster_config):
    req = request(make_cluster_config, workload="pretrain_llama_7b", model_size=" 7B ", scale="8")
    req.validate()
    assert (req.workload, req.model_size) == ("pretrain_llama", "7b")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"scale": "8"},
        {"file_path": "t.yaml"},
        {"workload": "pretrain_x_7b", "dtype": "fp8", "scale": "8", "force": True},
        {"workload": "pretrain_x", "model_size": "7b", "dtype": "fp8", "scale": "8", "force": True},
    ],
)
def test_validate_accepts(make_cluster_config, kwargs):
    request(make_cluster_config, **kwargs).validate()


def test_generate_tasks_reraises_validation_error_as_value_error(make_cluster_config):
    with pytest.raises(ValueError, match="Must specify scale"):
        generate_tasks(request(make_cluster_config))


WORKLOADS = {
    "pretrain_a": {
        "workload_type": "pretrain",
        "metadata": {
            "run": {
                "gpu_configs": {
                    "gb200": {
                        "model_configs": [
                            {
                                "model_size": "7b",
                                "dtypes": {"fp8": [8, 16], "bf16": {"scales": [32], "exact_scales": True}},
                            },
                            {"model_size": "70b", "dtypes": ["fp8"], "scales": [64], "proxy_scales": [8]},
                        ]
                    }
                }
            }
        },
    },
    "inference_b": {
        "workload_type": "inference",
        "metadata": {
            "run": {
                "gpu_configs": {"gb200": {"model_configs": [{"model_size": "1b", "dtypes": ["fp8"], "scales": [8]}]}}
            }
        },
    },
    "pretrain_notinstalled": {
        "workload_type": "pretrain",
        "metadata": {
            "run": {
                "gpu_configs": {"gb200": {"model_configs": [{"model_size": "1b", "dtypes": ["fp8"], "scales": [8]}]}}
            }
        },
    },
}


def discover(make_cluster_config, **kwargs):
    cfg = make_cluster_config(installed=["pretrain_a", "inference_b"])
    kwargs.setdefault("max_scale", None)
    return generate_submit_all_tasks(WORKLOADS, cfg, **kwargs)


def triples(tasks):
    return [(t.workload_key, t.model_size, t.dtype, t.scale) for t in tasks]


def test_discovery_defaults_to_installed_pretrain_and_finetune_only(make_cluster_config):
    tasks = discover(make_cluster_config, max_scale=None)
    assert {t.workload_key for t in tasks} == {"pretrain_a"}


def test_discovery_expands_power_of_two_unless_exact(make_cluster_config):
    tasks = discover(make_cluster_config, max_scale=64)
    assert triples(tasks) == [
        ("pretrain_a", "7b", "fp8", 8),
        ("pretrain_a", "7b", "fp8", 16),
        ("pretrain_a", "7b", "fp8", 32),
        ("pretrain_a", "7b", "fp8", 64),
        ("pretrain_a", "7b", "bf16", 32),  # metadata exact_scales wins
        ("pretrain_a", "70b", "fp8", 64),
    ]


def test_discovery_runtime_exact_scales_disables_expansion(make_cluster_config):
    tasks = discover(make_cluster_config, max_scale=64, exact_scales=True, dtype_filter=["fp8"])
    assert triples(tasks) == [
        ("pretrain_a", "7b", "fp8", 8),
        ("pretrain_a", "7b", "fp8", 16),
        ("pretrain_a", "70b", "fp8", 64),
    ]


def test_discovery_min_scale_picks_smallest_per_dtype(make_cluster_config):
    tasks = discover(make_cluster_config, min_scale=True)
    assert triples(tasks) == [
        ("pretrain_a", "7b", "fp8", 8),
        ("pretrain_a", "7b", "bf16", 32),
        ("pretrain_a", "70b", "fp8", 64),
    ]


def test_discovery_min_scale_with_max_scale_below_minimum_yields_nothing(make_cluster_config):
    assert discover(make_cluster_config, min_scale=True, max_scale=4) == []


def test_discovery_proxy_uses_proxy_scales_only(make_cluster_config):
    tasks = discover(make_cluster_config, proxy=True, max_scale=512)
    assert triples(tasks) == [("pretrain_a", "70b", "fp8", 8)]
    assert tasks[0].proxy is True


def test_discovery_specific_scales_filtered_by_support(make_cluster_config):
    tasks = discover(
        make_cluster_config, specific_scales=[4, 16, 128], dtype_filter=["fp8"], workload_filter=["pretrain_a_7b"]
    )
    # 4 is below minimum; 16 is listed; 128 is above the listed scales but kept by power-of-two extension.
    assert triples(tasks) == [("pretrain_a", "7b", "fp8", 16), ("pretrain_a", "7b", "fp8", 128)]


def test_discovery_specific_scales_respect_exact_metadata(make_cluster_config):
    tasks = discover(
        make_cluster_config, specific_scales=[32, 64], workload_filter=["pretrain_a_7b"], dtype_filter=["bf16"]
    )
    assert triples(tasks) == [("pretrain_a", "7b", "bf16", 32)]


def test_workload_filter_can_select_non_pretrain_type(make_cluster_config):
    tasks = discover(make_cluster_config, workload_filter=["inference_b"])
    assert triples(tasks) == [("inference_b", "1b", "fp8", 8)]


def test_repeats_and_profile_propagate(make_cluster_config):
    tasks = discover(make_cluster_config, repeats=3, profile=True, workload_filter=["pretrain_a_70b"])
    assert len(tasks) == 3
    assert all(t.profile for t in tasks)


def test_uninstalled_and_wrong_gpu_yield_nothing(make_cluster_config):
    cfg = make_cluster_config(gpu_type="h100", installed=["pretrain_a"])
    assert generate_submit_all_tasks(WORKLOADS, cfg, None) == []
    cfg = make_cluster_config(installed=[])
    assert generate_submit_all_tasks(WORKLOADS, cfg, None) == []


def test_generate_tasks_applies_env_and_mbridge_modifiers(make_cluster_config):
    cfg = make_cluster_config(installed=["pretrain_a"])
    req = TaskGenerationRequest(
        workloads=WORKLOADS,
        cluster_config=cfg,
        workload="pretrain_a_70b",
        scale="64",
        explicit_env_overrides={"A": "1"},
        mbridge_args=("--x",),
        extra_workload_args=("k=v",),
    )
    (task,) = generate_tasks(req)
    assert task.explicit_env_overrides == {"A": "1"}
    assert task.env_overrides == {"A": "1"}
    assert task.mbridge_args == ("--x",)
    assert task.extra_workload_args == ("k=v",)
