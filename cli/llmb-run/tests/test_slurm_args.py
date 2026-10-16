import pytest

from llmb_run.slurm_args import build_cli_slurm_args, validate_no_additional_slurm_params_conflict


@pytest.fixture(autouse=True)
def _clean_process_env(monkeypatch):
    monkeypatch.delenv("ADDITIONAL_SLURM_PARAMS", raising=False)


def test_no_inputs_returns_none():
    assert build_cli_slurm_args() is None
    assert build_cli_slurm_args(slurm_args=[]) is None


def test_named_flags_render_in_canonical_order_regardless_of_call_order():
    args = build_cli_slurm_args(nice=5, segment=2, reservation="r", exclude="n1", nodelist="n[2-3]")
    assert args.to_sbatch_args() == ["--nodelist=n[2-3]", "--exclude=n1", "--reservation=r", "--segment=2", "--nice=5"]
    assert args.to_additional_slurm_params() == "nodelist=n[2-3];exclude=n1;reservation=r;segment=2;nice=5"


def test_zero_valued_ints_are_kept():
    args = build_cli_slurm_args(nice=0, segment=0)
    assert args.to_sbatch_args() == ["--segment=0", "--nice=0"]


def test_passthrough_follows_named_and_keeps_order_and_bare_flags():
    args = build_cli_slurm_args(nodelist="n1", slurm_args=["constraint=gpu", "exclusive", " time = 01:00 "])
    assert args.to_sbatch_args() == ["--nodelist=n1", "--constraint=gpu", "--exclusive", "--time=01:00"]
    assert args.to_additional_slurm_params() == "nodelist=n1;constraint=gpu;exclusive;time=01:00"


def test_get_named_param_and_is_empty():
    args = build_cli_slurm_args(nodelist="n1")
    assert args.get_named_param("nodelist") == "n1"
    assert args.get_named_param("exclude") is None
    assert not args.is_empty()


@pytest.mark.parametrize(
    "raw, match",
    [
        ("", "cannot be empty"),
        ("--exclusive", "leading '--'"),
        ("=gpu", "non-empty key and value"),
        ("constraint=", "non-empty key and value"),
        ("nodelist=n1", "dedicated flag"),
    ],
)
def test_bad_slurm_arg_rejected(raw, match):
    with pytest.raises(ValueError, match=match):
        build_cli_slurm_args(slurm_args=[raw])


def test_duplicate_passthrough_rejected():
    with pytest.raises(ValueError, match="Duplicate.*'constraint'"):
        build_cli_slurm_args(slurm_args=["constraint=a", "constraint=b"])


def test_value_containing_equals_is_preserved():
    args = build_cli_slurm_args(slurm_args=["comment=a=b"])
    assert args.to_sbatch_args() == ["--comment=a=b"]


def test_conflict_noop_without_cli_args():
    validate_no_additional_slurm_params_conflict(cli_args=None, cluster_environment={"ADDITIONAL_SLURM_PARAMS": "x=1"})


@pytest.mark.parametrize(
    "kwarg, label",
    [
        ("cluster_environment", "cluster config environment"),
        ("workload_environment", "workload config environment"),
        ("task_environment", "task environment overrides"),
    ],
)
def test_conflict_raises_naming_source(kwarg, label):
    cli = build_cli_slurm_args(nodelist="n1")
    with pytest.raises(ValueError, match=label):
        validate_no_additional_slurm_params_conflict(cli_args=cli, **{kwarg: {"ADDITIONAL_SLURM_PARAMS": "x=1"}})


def test_conflict_raises_for_process_environment(monkeypatch):
    monkeypatch.setenv("ADDITIONAL_SLURM_PARAMS", "x=1")
    with pytest.raises(ValueError, match="process environment"):
        validate_no_additional_slurm_params_conflict(cli_args=build_cli_slurm_args(nice=1))


def test_conflict_lists_all_sources():
    cli = build_cli_slurm_args(nice=1)
    env = {"ADDITIONAL_SLURM_PARAMS": "x=1"}
    with pytest.raises(ValueError, match="cluster config environment, workload config environment"):
        validate_no_additional_slurm_params_conflict(cli_args=cli, cluster_environment=env, workload_environment=env)


@pytest.mark.parametrize("value", ["   ", None])
def test_blank_env_values_do_not_conflict(value):
    validate_no_additional_slurm_params_conflict(
        cli_args=build_cli_slurm_args(nice=1), cluster_environment={"ADDITIONAL_SLURM_PARAMS": value}
    )
