import shlex

import pytest

from llmb_run import env_args
from llmb_run.env_args import (
    apply_mbridge_extra_args_contract,
    apply_nemo_explicit_env_contract,
    apply_nemo_workload_args_contract,
    apply_sbatch_explicit_env_contract,
    build_mbridge_extra_args,
    build_nemo_env_override_flags,
    parse_cli_env_args,
    parse_cli_mbridge_args,
    validate_env_key,
    validate_shell_safe_env_value,
)


@pytest.mark.parametrize("key", ["_x", "NCCL_DEBUG"])
def test_validate_env_key_accepts_shell_names(key):
    assert validate_env_key(key) == key


def test_validate_env_key_strips_whitespace():
    assert validate_env_key("  FOO ") == "FOO"


@pytest.mark.parametrize("key", ["1ABC", "A B", "é"])
def test_validate_env_key_rejects_bad_names(key):
    with pytest.raises(ValueError, match="invalid"):
        validate_env_key(key)


@pytest.mark.parametrize("key", ["", "   "])
def test_validate_env_key_rejects_empty(key):
    with pytest.raises(ValueError, match="non-empty"):
        validate_env_key(key)


@pytest.mark.parametrize("key", [None, 5])
def test_validate_env_key_rejects_non_string(key):
    with pytest.raises(ValueError, match="invalid"):
        validate_env_key(key)


def test_validate_env_key_error_names_source():
    with pytest.raises(ValueError, match="^workload env variable name"):
        validate_env_key("1x", source="workload env")


@pytest.mark.parametrize("value", ["", "a_b@c%d+e=f:g,h./i-j"])
def test_shell_safe_value_accepts(value):
    validate_shell_safe_env_value("K", value)


@pytest.mark.parametrize("value", ["a b", "a'b", "$HOME", "a\nb"])
def test_shell_safe_value_rejects(value):
    with pytest.raises(ValueError, match="'K'.*shell-special"):
        validate_shell_safe_env_value("K", value)


def test_parse_cli_env_args_none_and_empty():
    assert parse_cli_env_args(None) == {}
    assert parse_cli_env_args([]) == {}


def test_parse_cli_env_args_preserves_order_and_splits_on_first_equals():
    parsed = parse_cli_env_args(["B=2", "A=x=y", "C="])
    assert list(parsed.items()) == [("B", "2"), ("A", "x=y"), ("C", "")]


def test_parse_cli_env_args_requires_equals():
    with pytest.raises(ValueError, match="KEY=value"):
        parse_cli_env_args(["NOEQUALS"])


def test_parse_cli_env_args_rejects_duplicates():
    with pytest.raises(ValueError, match="Duplicate.*'A'"):
        parse_cli_env_args(["A=1", "A=2"])


def test_parse_cli_env_args_rejects_bad_key_and_unsafe_value():
    with pytest.raises(ValueError, match="`--env`"):
        parse_cli_env_args(["1A=x"])
    with pytest.raises(ValueError, match="shell-special"):
        parse_cli_env_args(["A=has space"])


def test_parse_cli_mbridge_args_preserves_order():
    assert parse_cli_mbridge_args(["--a", "1", "x=y"]) == ("--a", "1", "x=y")
    assert parse_cli_mbridge_args(None) == ()


@pytest.mark.parametrize("bad", ["", "a b"])
def test_parse_cli_mbridge_args_rejects_empty_and_whitespace(bad):
    with pytest.raises(ValueError, match="--mbridge-arg"):
        parse_cli_mbridge_args(["ok", bad])


def test_build_nemo_env_override_flags_plain():
    assert build_nemo_env_override_flags({"A": "1", "B": "x"}) == "-E A=1 -E B=x"
    assert build_nemo_env_override_flags({}) == ""


def test_build_nemo_env_override_flags_quotes_and_round_trips():
    flags = build_nemo_env_override_flags({"A": "has space", "B": "q'uote", "C": "$X"})
    assert shlex.split(flags) == ["-E", "A=has space", "-E", "B=q'uote", "-E", "C=$X"]


def test_sbatch_contract_creates_var():
    env = {}
    apply_sbatch_explicit_env_contract(env, {"A": "1", "B": "2"})
    assert env[env_args.LLMB_CONTAINER_ENV] == "A,B"


def test_sbatch_contract_appends_and_dedups_preserving_order():
    env = {"LLMB_CONTAINER_ENV": "X,A,,"}
    apply_sbatch_explicit_env_contract(env, {"A": "1", "B": "2"})
    assert env["LLMB_CONTAINER_ENV"] == "X,A,B"


def test_sbatch_contract_noop_without_overrides():
    env = {}
    apply_sbatch_explicit_env_contract(env, {})
    assert env == {}


def test_nemo_env_contract_creates_then_appends():
    env = {}
    apply_nemo_explicit_env_contract(env, {"A": "1"})
    assert env["CONFIG_OVERRIDES"] == "-E A=1"
    apply_nemo_explicit_env_contract(env, {"B": "2"})
    assert env["CONFIG_OVERRIDES"] == "-E A=1 -E B=2"


def test_nemo_env_contract_appends_after_existing_and_strips():
    env = {"CONFIG_OVERRIDES": "  model.x=1  "}
    apply_nemo_explicit_env_contract(env, {"A": "1"})
    assert env["CONFIG_OVERRIDES"] == "model.x=1 -E A=1"


def test_nemo_env_contract_noop_without_overrides():
    env = {"CONFIG_OVERRIDES": " keep "}
    apply_nemo_explicit_env_contract(env, {})
    assert env == {"CONFIG_OVERRIDES": " keep "}


def test_nemo_workload_args_contract():
    env = {}
    apply_nemo_workload_args_contract(env, ["a=1", "b=2"])
    assert env["CONFIG_OVERRIDES"] == "a=1 b=2"
    apply_nemo_workload_args_contract(env, ["c=3"])
    assert env["CONFIG_OVERRIDES"] == "a=1 b=2 c=3"


def test_nemo_workload_args_contract_noop_when_empty():
    env = {}
    apply_nemo_workload_args_contract(env, [])
    assert env == {}


def test_build_mbridge_extra_args_ordering_compat_then_env_then_raw():
    rendered = build_mbridge_extra_args(["--raw", "1"], compatibility_args=["legacy=1"], env_overrides={"A": "1"})
    assert rendered == "legacy=1 -E A=1 --raw 1"


def test_build_mbridge_extra_args_skips_empty_segments():
    assert build_mbridge_extra_args(["--raw"]) == "--raw"
    assert build_mbridge_extra_args([], env_overrides={"A": "1"}) == "-E A=1"
    assert build_mbridge_extra_args([]) == ""


def test_mbridge_contract_creates_appends_and_noops():
    env = {}
    apply_mbridge_extra_args_contract(env, [])
    assert env == {}
    apply_mbridge_extra_args_contract(env, ["--a"], env_overrides={"K": "v"})
    assert env["LLMB_MBRIDGE_EXTRA_ARGS"] == "-E K=v --a"
    apply_mbridge_extra_args_contract(env, ["--b"])
    assert env["LLMB_MBRIDGE_EXTRA_ARGS"] == "-E K=v --a --b"
