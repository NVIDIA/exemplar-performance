import pytest
import yaml

from llmb_run.exemplar import (
    compute_and_validate_eligible_configs,
    generate_exemplar_tasks,
    get_exemplar_configs_from_yaml,
    parse_exemplar_workload_name,
    validate_exemplar_yaml_schema,
    validate_strict_installs,
    validate_yaml_config_against_metadata,
)
from llmb_run.task_generation import ValidationError


def wl(gpu="gb200", sizes=None):
    sizes = sizes or {"7b": {"fp8": [128, 512]}}
    return {
        "metadata": {
            "run": {
                "gpu_configs": {
                    gpu: {"model_configs": [{"model_size": s, "dtypes": d} for s, d in sizes.items()]},
                }
            }
        }
    }


WORKLOADS = {
    "pretrain_b": wl(sizes={"70b": {"fp8": [256, 512], "bf16": [512]}, "7b": {"fp8": [512]}, "1t": {"fp8": [512]}}),
    "pretrain_a": wl(sizes={"8b": {"fp8": [512]}}),
    "pretrain_noscale": wl(sizes={"8b": {"fp8": [128, 256]}}),
    "pretrain_h100only": wl(gpu="h100"),
}


def write_yaml(repo, config, workloads, gpu="gb200"):
    repo.mkdir(parents=True, exist_ok=True)
    (repo / "exemplar.yaml").write_text(yaml.safe_dump({"config": config, "workloads": {gpu: workloads}}))


def setup(make_cluster_config, tmp_path, entries, config=None, installed=None):
    repo = tmp_path / "repo"
    write_yaml(repo, {} if config is None else config, entries)
    if installed is None:
        installed = sorted({list(e)[0].rsplit("_", 1)[0] for e in entries})
    return make_cluster_config(llmb_repo=repo, installed=installed)


ENTRIES = [
    {"pretrain_b_70b": {"dtypes": ["fp8", "bf16"]}},
    {"pretrain_a_8b": {"dtypes": ["fp8"]}},
    {"pretrain_b_7b": {"dtypes": ["fp8"]}},
    {"pretrain_b_1t": {"dtypes": ["fp8"]}},
]


@pytest.mark.parametrize(
    "name, expected",
    [("pretrain_llama3.1_70b", ("pretrain_llama3.1", "70b"))],
)
def test_parse_exemplar_workload_name(name, expected):
    assert parse_exemplar_workload_name(name) == expected


@pytest.mark.parametrize("name", ["pretrain_nosize", "pretrain_7x"])
def test_parse_exemplar_workload_name_rejects(name):
    with pytest.raises(ValidationError, match="Invalid workload name"):
        parse_exemplar_workload_name(name)


def valid(entries=None):
    return {"config": {}, "workloads": {"gb200": entries or [{"a_7b": {"dtypes": ["fp8"]}}]}}


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda d: d.pop("config"), "'config'"),
        (lambda d: d.pop("workloads"), "'workloads'"),
        (lambda d: d.update(workloads=[]), "must be a mapping"),
        (lambda d: d["workloads"].pop("gb200"), "does not contain workloads for gpu_type"),
        (lambda d: d["workloads"].update(gb200=[]), "is empty"),
        (lambda d: d["workloads"].update(gb200={}), "must be a list"),
        (lambda d: d["workloads"].update(gb200=["x"]), "must be a mapping"),
        (
            lambda d: d["workloads"].update(gb200=[{"a_7b": {"dtypes": ["fp8"]}, "b_7b": {"dtypes": ["fp8"]}}]),
            "exactly one key",
        ),
        (lambda d: d["workloads"].update(gb200=[{"a_7b": {}}]), "missing required key 'dtypes'"),
        (lambda d: d["workloads"].update(gb200=[{"a_7b": {"dtypes": "fp8"}}]), "must be a list"),
        (lambda d: d["workloads"].update(gb200=[{"a_7b": {"dtypes": []}}]), "is empty"),
        (lambda d: d["workloads"].update(gb200=[{"a_7b": {"dtypes": ["fp8"]}}] * 2), "duplicate entry"),
    ],
)
def test_validate_schema_rejects(mutate, match):
    data = valid()
    mutate(data)
    with pytest.raises(ValidationError, match=match):
        validate_exemplar_yaml_schema(data, "gb200")


def test_validate_schema_accepts_valid():
    validate_exemplar_yaml_schema(valid(), "gb200")


def test_get_configs_expands_dtypes_and_dedups_dtype_repeats():
    data = valid([{"a_7b": {"dtypes": ["fp8", "bf16", "fp8"]}}])
    assert get_exemplar_configs_from_yaml(data, "gb200") == [("a", "7b", "fp8"), ("a", "7b", "bf16")]


@pytest.mark.parametrize(
    "args, match",
    [
        (("missing", "7b", "fp8", 512, "gb200"), "does not exist"),
        (("pretrain_h100only", "7b", "fp8", 512, "gb200"), "does not support gpu_type"),
        (("pretrain_a", "9b", "fp8", 512, "gb200"), "does not have model_size"),
        (("pretrain_a", "8b", "bf16", 512, "gb200"), "does not support dtype"),
        (("pretrain_noscale", "8b", "fp8", 512, "gb200"), "does not explicitly support scale 512"),
    ],
)
def test_validate_against_metadata_rejects(args, match):
    with pytest.raises(ValidationError, match=match):
        validate_yaml_config_against_metadata(*args[:5], WORKLOADS)


def test_validate_strict_installs_empty_and_missing(make_cluster_config):
    cfg = make_cluster_config(installed=["pretrain_a"])
    with pytest.raises(ValidationError, match="No eligible workloads"):
        validate_strict_installs([], cfg)
    with pytest.raises(ValidationError, match=r"1 eligible workload\(s\) are not installed[\s\S]*pretrain_b"):
        validate_strict_installs([("pretrain_a", "8b", "fp8"), ("pretrain_b", "7b", "fp8")], cfg)
    with pytest.raises(ValidationError, match="No workloads are installed"):
        validate_strict_installs([("pretrain_a", "8b", "fp8")], make_cluster_config(installed=[]))
    validate_strict_installs([("pretrain_a", "8b", "fp8")], cfg)


def test_compute_eligible_defaults(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES)
    configs, scale, repeats, profile = compute_and_validate_eligible_configs(WORKLOADS, cfg)
    assert scale == 512 and repeats == 1 and profile is False
    assert len(configs) == 5


def test_compute_does_not_filter_on_workload_type(make_cluster_config, tmp_path):
    # The llmb-run README says exemplar eligibility requires workload type
    # `pretrain`, but the code never checks it: any workload listed in
    # exemplar.yaml with a matching GPU/dtype/scale is eligible.
    workloads = dict(WORKLOADS)
    workloads["finetune_x"] = wl(sizes={"70b": {"fp8": [512]}})
    workloads["finetune_x"]["metadata"]["general"] = {"workload_type": "finetune"}
    cfg = setup(make_cluster_config, tmp_path, [{"finetune_x_70b": {"dtypes": ["fp8"]}}])
    configs, _, _, _ = compute_and_validate_eligible_configs(workloads, cfg)
    assert configs == [("finetune_x", "70b", "fp8")]


def test_compute_reads_config_values(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES[1:2], config={"scale": 512, "repeats": 4, "profile": True})
    _, scale, repeats, profile = compute_and_validate_eligible_configs(WORKLOADS, cfg)
    assert (scale, repeats, profile) == (512, 4, True)


def test_compute_explicit_scale_must_be_listed(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, [{"pretrain_noscale_8b": {"dtypes": ["fp8"]}}], config={"scale": 512})
    with pytest.raises(ValidationError, match="does not explicitly support scale 512"):
        compute_and_validate_eligible_configs(WORKLOADS, cfg)


def test_compute_uses_listed_non_default_scale(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, [{"pretrain_noscale_8b": {"dtypes": ["fp8"]}}], config={"scale": 256})
    _, scale, _, _ = compute_and_validate_eligible_configs(WORKLOADS, cfg)
    assert scale == 256


def test_compute_raises_when_workload_not_installed(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES, installed=["pretrain_a"])
    with pytest.raises(ValidationError, match="not installed"):
        compute_and_validate_eligible_configs(WORKLOADS, cfg)


def test_compute_raises_when_gpu_type_unset(make_cluster_config, tmp_path):
    with pytest.raises(ValidationError, match="No GPU type"):
        compute_and_validate_eligible_configs(WORKLOADS, make_cluster_config(gpu_type=""))


def test_compute_raises_when_exemplar_yaml_missing(make_cluster_config, tmp_path):
    with pytest.raises(ValidationError, match="exemplar.yaml not found"):
        compute_and_validate_eligible_configs(WORKLOADS, make_cluster_config(llmb_repo=tmp_path / "nowhere"))


def test_compute_raises_on_malformed_yaml(make_cluster_config, tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "exemplar.yaml").write_text("config: [unclosed\n")
    with pytest.raises(ValidationError, match="Failed to parse"):
        compute_and_validate_eligible_configs(WORKLOADS, make_cluster_config(llmb_repo=repo))
    (repo / "exemplar.yaml").write_text("- a\n- b\n")
    with pytest.raises(ValidationError, match="YAML mapping"):
        compute_and_validate_eligible_configs(WORKLOADS, make_cluster_config(llmb_repo=repo))


def ident(task):
    return (task.workload_key, task.model_size, task.dtype, task.profile)


def test_generate_orders_by_workload_then_numeric_size_then_dtype(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES)
    tasks = generate_exemplar_tasks(WORKLOADS, cfg)
    assert [(t.workload_key, t.model_size, t.dtype) for t in tasks] == [
        ("pretrain_a", "8b", "fp8"),
        ("pretrain_b", "7b", "fp8"),
        ("pretrain_b", "70b", "bf16"),
        ("pretrain_b", "70b", "fp8"),
        ("pretrain_b", "1t", "fp8"),
    ]
    assert {t.scale for t in tasks} == {512}


def test_generate_repeats_are_contiguous_and_profile_is_last_only(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES[:1], config={"repeats": 3, "profile": True})
    tasks = generate_exemplar_tasks(WORKLOADS, cfg)
    assert [ident(t) for t in tasks] == [
        ("pretrain_b", "70b", "bf16", False),
        ("pretrain_b", "70b", "bf16", False),
        ("pretrain_b", "70b", "bf16", True),
        ("pretrain_b", "70b", "fp8", False),
        ("pretrain_b", "70b", "fp8", False),
        ("pretrain_b", "70b", "fp8", True),
    ]


def test_generate_without_profile_never_profiles(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES[1:2], config={"repeats": 2})
    assert [t.profile for t in generate_exemplar_tasks(WORKLOADS, cfg)] == [False, False]


def test_generate_single_repeat_with_profile_is_profiled(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES[1:2], config={"profile": True})
    assert [t.profile for t in generate_exemplar_tasks(WORKLOADS, cfg)] == [True]


def test_generate_cli_repeats_override_yaml(make_cluster_config, tmp_path):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES[1:2], config={"repeats": 5})
    assert len(generate_exemplar_tasks(WORKLOADS, cfg, repeats=2)) == 2


@pytest.mark.parametrize("bad", [0, True, "3", 1.5])
def test_generate_rejects_invalid_repeats(make_cluster_config, tmp_path, bad):
    cfg = setup(make_cluster_config, tmp_path, ENTRIES[1:2], config={"repeats": bad})
    with pytest.raises(ValidationError, match="Invalid exemplar repeats"):
        generate_exemplar_tasks(WORKLOADS, cfg)
