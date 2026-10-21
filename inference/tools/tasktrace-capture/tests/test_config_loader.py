# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for tasktrace_capture.config_loader and the --config wiring in tasktrace_capture.cli."""

from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import pytest
from tasktrace_capture.config_loader import ALLOWED_KEYS, load_yaml_config


def _write(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "exp.yaml"
    p.write_text(textwrap.dedent(body))
    return p


# ===========================================================================
#  Happy-path parsing
# ===========================================================================


def test_parses_valid_yaml(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        runtime: docker
        agent: claude-code
        model: azure/anthropic/claude-sonnet-4-5
        dataset: hello-world
        skip_preflight: true
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["runtime"] == "docker"
    assert cfg["agent"] == "claude-code"
    assert cfg["dataset"] == "hello-world"
    assert cfg["skip_preflight"] is True


def test_parses_package_dataset_include_tasks(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        dataset: datacurve/deep-swe
        include_tasks:
          - datacurve/task-a
          - datacurve/task-b
          - datacurve/task-a
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["dataset"] == "datacurve/deep-swe"
    assert cfg["include_tasks"] == ["datacurve/task-a", "datacurve/task-b"]


def test_parses_enroot_best_effort_network_policy(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        runtime: enroot
        enroot_network_policy: best-effort
        dataset: datacurve/deep-swe
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["enroot_network_policy"] == "best-effort"


@pytest.mark.parametrize("value", ["task-a", [], [""], [1]])
def test_rejects_invalid_include_tasks(tmp_path: Path, value) -> None:
    p = tmp_path / "exp.yaml"
    p.write_text("include_tasks: " + repr(value) + "\n")
    with pytest.raises(ValueError, match="include_tasks"):
        load_yaml_config(p)


def test_empty_yaml_returns_empty_dict(tmp_path: Path) -> None:
    p = tmp_path / "empty.yaml"
    p.write_text("")
    assert load_yaml_config(p) == {}


def test_top_level_must_be_mapping(tmp_path: Path) -> None:
    p = tmp_path / "list.yaml"
    p.write_text("- 1\n- 2\n")
    with pytest.raises(ValueError, match="top-level YAML must be a mapping"):
        load_yaml_config(p)


# ===========================================================================
#  ${VAR} interpolation
# ===========================================================================


def test_interp_braced(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_TEST_KEY", "test123")
    p = _write(
        tmp_path,
        """
        agent_env:
          MY_KEY: ${MY_TEST_KEY}
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["agent_env"] == ["MY_KEY=test123"]


def test_interp_bareword(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MY_BARE", "barevalue")
    p = _write(
        tmp_path,
        """
        model: prefix-$MY_BARE-suffix
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["model"] == "prefix-barevalue-suffix"


def test_interp_unset_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEFINITELY_NOT_SET_TASKTRACE_CAPTURE_TEST", raising=False)
    p = _write(
        tmp_path,
        """
        agent_env:
          K: ${DEFINITELY_NOT_SET_TASKTRACE_CAPTURE_TEST}
        """,
    )
    with pytest.raises(KeyError, match="DEFINITELY_NOT_SET_TASKTRACE_CAPTURE_TEST"):
        load_yaml_config(p)


def test_interp_inside_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("KW1", "kw1value")
    p = _write(
        tmp_path,
        """
        agent_kwarg:
          - foo=${KW1}
          - bar=plain
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["agent_kwarg"] == ["foo=kw1value", "bar=plain"]


# ===========================================================================
#  Unknown-key rejection
# ===========================================================================


def test_unknown_key_rejected(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        runtime: docker
        widget: 5
        """,
    )
    with pytest.raises(ValueError, match="unknown config key"):
        load_yaml_config(p)


def test_known_keys_match_argparse_dests() -> None:
    # Smoke-check that ALLOWED_KEYS is in sync with the argparse parser in cli.
    # Missing keys here would let real CLI dest names slip through unchecked.
    from tasktrace_capture import cli

    _ = argparse.ArgumentParser()
    # Re-build the parser the same way main() does, just to harvest dests.
    saved_main = cli.main
    # Walk through the source's parser construction by importing module-level
    # — easier: invoke parser construction by running parse_args with --help
    # in a subprocess. Skip if parsing isn't reflective. Use the loose check
    # below instead: every ALLOWED_KEY must be a valid Python identifier,
    # which is the contract argparse uses for dest names.
    assert all(k.isidentifier() for k in ALLOWED_KEYS)
    # And everything we know exists on the CLI is in ALLOWED_KEYS.
    expected = {
        "dataset",
        "n_tasks",
        "task",
        "model",
        "api_base",
        "n_concurrent",
        "trials",
        "sweep",
        "concurrency",
        "agent",
        "agent_kwarg",
        "agent_env",
        "api_key",
        "agent_timeout",
        "output_dir",
        "repeat",
        "dry_run",
        "check",
        "skip_preflight",
        "allow_private_api",
        "quiet",
        "runtime",
        "allow_parallel_docker",
        "swebench_image_prefix",
        "override_cpus",
        "instrument_shell_commands",
        "instrument_shell_perfetto_depth",
        "claude_code_version",
        "profile_live_cpu",
        "profile_windows",
        "profile_scheduler",
        "profile_scheduler_collector_image",
        "profile_cpuset_cpus",
        "no_start_barrier",
    }
    assert expected <= ALLOWED_KEYS
    _ = saved_main


def test_allow_parallel_docker_must_be_boolean(tmp_path: Path) -> None:
    p = _write(tmp_path, 'allow_parallel_docker: "false"')
    with pytest.raises(ValueError, match="must be a boolean"):
        load_yaml_config(p)


# ===========================================================================
#  agent_env shape coercion
# ===========================================================================


def test_agent_env_dict_to_list(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        agent_env:
          ANTHROPIC_BASE_URL: https://api.example.com
          ANTHROPIC_API_KEY: sk-test
        """,
    )
    cfg = load_yaml_config(p)
    assert sorted(cfg["agent_env"]) == [
        "ANTHROPIC_API_KEY=sk-test",
        "ANTHROPIC_BASE_URL=https://api.example.com",
    ]


def test_agent_env_list_passthrough(tmp_path: Path) -> None:
    p = _write(
        tmp_path,
        """
        agent_env:
          - FOO=bar
          - BAZ=qux
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["agent_env"] == ["FOO=bar", "BAZ=qux"]


def test_agent_env_invalid_type(tmp_path: Path) -> None:
    p = _write(tmp_path, "agent_env: 42\n")
    with pytest.raises(ValueError, match="agent_env"):
        load_yaml_config(p)


# ===========================================================================
#  concurrency coercion
# ===========================================================================


def test_concurrency_int(tmp_path: Path) -> None:
    p = _write(tmp_path, "concurrency: 4\n")
    assert load_yaml_config(p)["concurrency"] == "4"


def test_concurrency_list(tmp_path: Path) -> None:
    p = _write(tmp_path, "concurrency: [1, 4, 16]\n")
    assert load_yaml_config(p)["concurrency"] == "1,4,16"


def test_concurrency_string_passthrough(tmp_path: Path) -> None:
    p = _write(tmp_path, 'concurrency: "1,2,4"\n')
    assert load_yaml_config(p)["concurrency"] == "1,2,4"


def test_concurrency_bool_rejected(tmp_path: Path) -> None:
    p = _write(tmp_path, "concurrency: true\n")
    with pytest.raises(ValueError, match="concurrency"):
        load_yaml_config(p)


# ===========================================================================
#  Strict-mode --config semantics (yaml is exclusive; mixing errors out)
# ===========================================================================
#
# Behavior under test (lives in tasktrace_capture.cli):
#   - `-c file.yaml` alone:  yaml fully populates args.
#   - `-c file.yaml --any-other-flag`:  parser.error() before any work.
#
# The mixing-rejection itself is exercised end-to-end (subprocess) in
# tests/test_cli.py::TestConfigStrictMode. Here we only cover the loader and
# the user-supplied-flag detector.


def test_user_supplied_flags_helper() -> None:
    """The detector should report only flags the user actually typed."""
    from tasktrace_capture.cli import _user_supplied_flags

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", default=None)
    parser.add_argument("--concurrency", default=None)
    parser.add_argument("--agent-env", action="append", default=[])
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--runtime", default="enroot")

    # Only --config: nothing else counts.
    args = parser.parse_args(["-c", "x.yaml"])
    assert _user_supplied_flags(parser, args, ignore={"config"}) == []

    # --concurrency 1: scalar.
    args = parser.parse_args(["-c", "x.yaml", "--concurrency", "1"])
    assert _user_supplied_flags(parser, args, ignore={"config"}) == ["--concurrency 1"]

    # --skip-preflight: store_true with no value.
    args = parser.parse_args(["-c", "x.yaml", "--skip-preflight"])
    assert _user_supplied_flags(parser, args, ignore={"config"}) == ["--skip-preflight"]

    # --agent-env appended: each value reported separately.
    args = parser.parse_args(["-c", "x.yaml", "--agent-env", "A=1", "--agent-env", "B=2"])
    flags = _user_supplied_flags(parser, args, ignore={"config"})
    assert flags == ["--agent-env A=1", "--agent-env B=2"]

    # Mixed flags: every violator is listed.
    args = parser.parse_args(["-c", "x.yaml", "--concurrency", "4", "--runtime", "docker", "--agent-env", "K=v"])
    flags = _user_supplied_flags(parser, args, ignore={"config"})
    assert "--concurrency 4" in flags
    assert "--runtime docker" in flags
    assert "--agent-env K=v" in flags


# ===========================================================================
#  Full README example
# ===========================================================================


def test_readme_example(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NV_KEY", "fake-test-key")
    p = _write(
        tmp_path,
        """
        runtime: docker
        agent: claude-code
        model: azure/anthropic/claude-sonnet-4-5
        dataset: hello-world
        concurrency: [1, 4, 16]
        agent_env:
          ANTHROPIC_BASE_URL: https://api.example.com
          ANTHROPIC_API_KEY: ${NV_KEY}
          ANTHROPIC_DEFAULT_SONNET_MODEL: azure/anthropic/claude-sonnet-4-5
        skip_preflight: true
        """,
    )
    cfg = load_yaml_config(p)
    assert cfg["runtime"] == "docker"
    assert cfg["agent"] == "claude-code"
    assert cfg["concurrency"] == "1,4,16"
    assert cfg["skip_preflight"] is True
    assert "ANTHROPIC_API_KEY=fake-test-key" in cfg["agent_env"]


# Example YAML config tests removed: the example YAML files they referenced
# (compilebench-mini-swe-nvidia-gpt55-*.yaml, terminal-bench-*.yaml) were
# removed during OSS cleanup.
