# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for llmb_collector."""

import json
import os
from unittest.mock import patch

import pytest
from llmb_collector.collect import (
    HOST_COMMANDS,
    HOST_DOMAINS,
    NETWORK_COMMANDS,
    REDACTED_VALUE,
    CollectConfig,
    _expand_names,
    _format_text_report,
    _resolve_container_commands,
    _resolve_host_commands,
    _resolve_network_commands,
    _run_cmd,
    _serialize,
    _write_to_dir,
    collect,
)


def test_run_cmd_returns_dict():
    result = _run_cmd(["echo", "hello"])
    assert result["returncode"] == 0
    assert "hello" in result["stdout"]
    assert result["cmdline"] == "echo hello"
    assert "exception" in result


def test_run_cmd_nonexistent():
    result = _run_cmd(["/nonexistent/binary"])
    assert result["returncode"] == -1
    assert result["exception"] is not None


def test_expand_names_domain():
    got = _expand_names(["cpu", "gpu"], HOST_DOMAINS, list(HOST_COMMANDS))
    assert got == ["lscpu", "nvidia_smi"]


def test_expand_names_individual():
    got = _expand_names(["lscpu", "free"], HOST_DOMAINS, list(HOST_COMMANDS))
    assert got == ["lscpu", "free"]


def test_expand_names_mixed():
    got = _expand_names(["cpu", "free"], HOST_DOMAINS, list(HOST_COMMANDS))
    assert "lscpu" in got
    assert "free" in got


def test_resolve_host_commands_all_when_no_lists():
    got = _resolve_host_commands(None, None)
    assert set(got) == set(HOST_COMMANDS)


def test_resolve_host_commands_whitelist():
    got = _resolve_host_commands(["cpu", "gpu"], None)
    assert set(got) == {"lscpu", "nvidia_smi"}


def test_resolve_host_commands_blacklist():
    got = _resolve_host_commands(None, ["gpu"])
    assert "nvidia_smi" not in got
    assert "lscpu" in got


def test_resolve_network_commands_all_when_no_lists():
    got = _resolve_network_commands(None, None)
    assert set(got) == set(NETWORK_COMMANDS)


def test_resolve_network_commands_whitelist():
    got = _resolve_network_commands(["interfaces"], None)
    assert set(got) == {"ip", "ethtool"}


def test_resolve_network_commands_blacklist():
    got = _resolve_network_commands(None, ["network"])
    assert got == []


def test_resolve_container_commands_whitelist():
    got = _resolve_container_commands(["gpu"], None)
    assert got == ["container_nvidia_smi"]


@patch.dict(os.environ, {"LLMB_COLLECT_FOO": "1", "PATH": "/usr/bin"}, clear=False)
def test_collect_env():
    config = CollectConfig(
        host=False,
        container=False,
        env=True,
        env_pattern=r"^LLMB_",
    )
    data = collect(config)
    assert data["env"] is not None
    assert "LLMB_COLLECT_FOO" in data["env"]
    assert data["env"]["LLMB_COLLECT_FOO"] == "1"


@patch.dict(
    os.environ,
    {
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITLAB_CI": "true",
        "BUILDKITE_BUILD_ID": "123",
        "JENKINS_URL": "https://jenkins.example",
        "UNRELATED_VAR": "skip",
    },
    clear=True,
)
def test_collect_env_default_pattern_includes_known_pipeline_runners():
    config = CollectConfig(host=False, network=False, container=False, env=True)
    data = collect(config)

    assert data["env"] == {
        "BUILDKITE_BUILD_ID": "123",
        "CI": "true",
        "GITHUB_ACTIONS": "true",
        "GITLAB_CI": "true",
        "JENKINS_URL": "https://jenkins.example",
    }


@patch.dict(
    os.environ,
    {
        "GITHUB_ACTIONS": "true",
        "GITHUB_TOKEN": "ghs_secret",
        "CI_JOB_TOKEN": "gitlab_secret",
        "SYSTEM_ACCESSTOKEN": "azure_secret",
        "BUILDKITE_AGENT_TOKEN": "buildkite_secret",
        "PYTHON_VERSION": "3.12",
    },
    clear=True,
)
def test_collect_env_redacts_sensitive_values_by_default():
    config = CollectConfig(host=False, network=False, container=False, env=True)
    data = collect(config)

    assert data["env"] == {
        "BUILDKITE_AGENT_TOKEN": REDACTED_VALUE,
        "CI_JOB_TOKEN": REDACTED_VALUE,
        "GITHUB_ACTIONS": "true",
        "GITHUB_TOKEN": REDACTED_VALUE,
        "PYTHON_VERSION": "3.12",
        "SYSTEM_ACCESSTOKEN": REDACTED_VALUE,
    }


@patch.dict(os.environ, {"GITHUB_TOKEN": "ghs_secret"}, clear=True)
def test_collect_env_redaction_can_be_disabled():
    config = CollectConfig(
        host=False,
        network=False,
        container=False,
        env=True,
        redact_env=False,
    )
    data = collect(config)

    assert data["env"] == {"GITHUB_TOKEN": "ghs_secret"}


@patch.dict(os.environ, {"CI_RUN_ID": "value-scanner-secret"}, clear=True)
@patch("llmb_collector.collect._detect_secrets_flags_value", return_value=True)
def test_collect_env_uses_optional_value_secret_detection(_mock_detect):
    config = CollectConfig(host=False, network=False, container=False, env=True)
    data = collect(config)

    assert data["env"] == {"CI_RUN_ID": REDACTED_VALUE}


@patch("llmb_collector.collect._run_cmd")
def test_collect_host_respects_config(mock_run):
    mock_run.return_value = {
        "cmdline": "test",
        "returncode": 0,
        "stdout": "out",
        "stderr": "",
        "exception": None,
    }
    config = CollectConfig(
        host=True,
        network=False,
        container=False,
        env=False,
        host_collect=["lscpu"],
    )
    data = collect(config)
    assert "system" in data
    assert list(data["system"]) == ["lscpu"]
    assert data["network"] is None
    assert data["container"] is None
    assert data["env"] is None


@patch("llmb_collector.collect._run_cmd")
def test_collect_network_is_own_group(mock_run):
    mock_run.return_value = {
        "cmdline": "test",
        "returncode": 0,
        "stdout": "out",
        "stderr": "",
        "exception": None,
    }
    config = CollectConfig(
        host=True,
        container=False,
        env=False,
        host_collect=["network"],
    )
    data = collect(config)
    assert data["system"] == {}
    assert set(data["network"]) == set(NETWORK_COMMANDS)


def test_collect_has_version_and_timestamp():
    with patch("llmb_collector.collect._run_cmd") as mock_run:
        mock_run.return_value = {
            "cmdline": "",
            "returncode": 0,
            "stdout": "",
            "stderr": "",
            "exception": None,
        }
        config = CollectConfig(host=True, container=False, env=False, host_collect=["lscpu"])
        data = collect(config)
    assert data["version"] == "1"
    assert "collected_at" in data


def test_serialize_json():
    data = {"version": "1", "x": 2}
    s = _serialize(data, "json", compact=True)
    assert " " not in s or "\n" not in s
    loaded = json.loads(s)
    assert loaded == data


def test_serialize_json_no_compact():
    data = {"version": "1"}
    s = _serialize(data, "json", compact=False)
    assert "\n" in s
    json.loads(s)


def test_serialize_json_omits_disabled_collectors():
    data = {"version": "1", "system": None, "container": {"x": 1}, "env": None}
    s = _serialize(data, "json", compact=True)
    loaded = json.loads(s)
    assert "system" not in loaded
    assert "env" not in loaded
    assert loaded["container"] == {"x": 1}


def test_serialize_json_omits_none_exception():
    data = {"system": {"lscpu": {"returncode": 0, "exception": None}}}
    s = _serialize(data, "json", compact=True)
    loaded = json.loads(s)
    assert "exception" not in loaded["system"]["lscpu"]


def test_serialize_yaml():
    data = {"version": "1", "a": [1, 2]}
    s = _serialize(data, "yaml", compact=True)
    assert "version" in s and "1" in s
    assert "{version:" not in s
    assert "\na:\n- 1\n- 2\n" in s


def test_serialize_yaml_multiline_strings_use_literal_blocks():
    data = {"stdout": "line1\nline2\n", "stderr": ""}
    s = _serialize(data, "yaml", compact=True)
    assert "stdout: |" in s
    assert "  line1\n  line2\n" in s
    assert "\\n" not in s


def test_serialize_yaml_tabbed_multiline_strings_use_literal_blocks():
    data = {"stdout": "x\n\ty\t \n", "stderr": "a\n\tb\t \n"}
    s = _serialize(data, "yaml", compact=True)
    assert "stdout: |" in s
    assert "stderr: |" in s
    assert "\\n" not in s


def test_serialize_yaml_control_chars_use_literal_blocks():
    data = {"stdout": "line1\n\x1b[31mline2\x1b[0m\n"}
    s = _serialize(data, "yaml", compact=True)
    assert "stdout: |" in s
    assert "\\x1b" not in s
    assert "\\n" not in s


def test_serialize_yaml_parses_json_stdout_as_yaml():
    data = {"system": {"lscpu": {"stdout": "{\"a\": 1, \"b\": [2, 3]}"}}}
    s = _serialize(data, "yaml", compact=True)
    assert "stdout:\n" in s
    assert "  a: 1\n" in s
    assert "  b:\n" in s
    assert "{\\\"a\\\"" not in s


def test_serialize_yaml_omits_disabled_collectors():
    data = {"version": "1", "network": None, "container": {"x": 1}, "env": None}
    s = _serialize(data, "yaml", compact=True)
    assert "\nnetwork:" not in s
    assert "\nenv:" not in s
    assert "\ncontainer:\n" in s


def test_serialize_yaml_omits_none_exception():
    data = {"container": {"container_nvidia_smi": {"returncode": 0, "exception": None}}}
    s = _serialize(data, "yaml", compact=True)
    assert "\nexception:" not in s


def test_serialize_text():
    data = {
        "collected_at": "2025-01-01T00:00:00Z",
        "system": {
            "lscpu": {
                "stdout": "cpu info",
                "stderr": "",
                "cmdline": "",
                "returncode": 0,
                "exception": None,
            }
        },
        "container": None,
        "env": None,
    }
    s = _format_text_report(data)
    assert "=== System" in s or "lscpu" in s
    assert "cpu info" in s


def test_write_to_dir(tmp_path):
    data = {
        "system": {
            "lscpu": {
                "cmdline": "lscpu --json",
                "returncode": 0,
                "stdout": "{}",
                "stderr": "",
                "exception": None,
            },
        },
        "container": None,
        "env": {"FOO": "bar"},
    }
    _write_to_dir(data, tmp_path)
    cloudperf = tmp_path / "_cloudperf"
    assert cloudperf.is_dir()
    env_content = (cloudperf / "job_envs.json").read_text()
    assert json.loads(env_content) == {"FOO": "bar"}
    assert (cloudperf / "lscpu" / "cmdline").read_text() == "lscpu --json"
    assert (cloudperf / "lscpu" / "stdout").read_text() == "{}"
    assert (cloudperf / "lscpu" / "returncode").read_text() == "0"


def test_cli_format_json(capsys):
    with patch("llmb_collector.cli.collect") as mock_collect:
        mock_collect.return_value = {"version": "1", "test": True}
        with patch("sys.argv", ["llmb-collect", "collect", "--format", "json", "--host"]):
            from llmb_collector.cli import main

            main()
    out = capsys.readouterr().out
    data = json.loads(out)
    assert data["version"] == "1"
    assert data["test"] is True


def test_cli_format_yaml(capsys):
    with patch("llmb_collector.cli.collect") as mock_collect:
        mock_collect.return_value = {"version": "1"}
        with patch("sys.argv", ["llmb-collect", "collect", "--format", "yaml", "--host"]):
            from llmb_collector.cli import main

            main()
    out = capsys.readouterr().out
    assert "version" in out and "1" in out


def test_cli_output_to_file(tmp_path):
    out_file = tmp_path / "report.json"
    with patch(
        "sys.argv",
        ["llmb-collect", "collect", "--host-collect", "lscpu", "--output", str(out_file)],
    ):
        with patch("llmb_collector.cli.collect") as mock_collect:
            mock_collect.return_value = {"version": "1"}
            from llmb_collector.cli import main

            main()
    assert out_file.exists()
    data = json.loads(out_file.read_text())
    assert data["version"] == "1"


def test_cli_output_to_directory(tmp_path):
    out_dir = tmp_path / "results"
    out_dir.mkdir()
    with patch(
        "sys.argv",
        ["llmb-collect", "collect", "--output", str(out_dir), "--network-collect", "network"],
    ):
        with patch("llmb_collector.cli.collect") as mock_collect:
            mock_collect.return_value = {
                "system": None,
                "container": None,
                "env": {"X": "1"},
            }
            from llmb_collector.cli import main

            main()
    cloudperf = out_dir / "_cloudperf"
    assert cloudperf.is_dir()
    assert (cloudperf / "job_envs.json").exists()


def test_cli_network_flag_sets_config(capsys):
    with patch("llmb_collector.cli.collect") as mock_collect:
        mock_collect.return_value = {"version": "1"}
        with patch("sys.argv", ["llmb-collect", "collect", "--host", "--format", "json"]):
            from llmb_collector.cli import main

            main()

    _ = capsys.readouterr().out
    config = mock_collect.call_args.args[0]
    assert config.network is False


def test_cli_host_and_host_collect_are_mutually_exclusive():
    with patch("sys.argv", ["llmb-collect", "collect", "--host", "--host-collect", "cpu"]):
        from llmb_collector.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2


def test_cli_network_and_network_collect_are_mutually_exclusive():
    with patch("sys.argv", ["llmb-collect", "collect", "--network", "--network-collect", "network"]):
        from llmb_collector.cli import main

        with pytest.raises(SystemExit) as exc:
            main()
        assert exc.value.code == 2


def test_cli_list_prints_and_exits_without_collect(capsys):
    with patch("llmb_collector.cli.collect") as mock_collect:
        with patch("sys.argv", ["llmb-collect", "list"]):
            from llmb_collector.cli import main

            main()

    captured = capsys.readouterr()
    assert "Available domains and sub-collectors:" in captured.out
    assert "Host domains:" in captured.out
    assert "Network domains:" in captured.out
    assert "Container domains:" in captured.out
    assert "cpu: lscpu" in captured.out
    assert "interfaces: ip, ethtool" in captured.out
    assert "infiniband: ibstat, ibstat_list" in captured.out
    assert "runtime: container_python3_version" in captured.out
    assert "gpu: container_nvidia_smi" in captured.out
    mock_collect.assert_not_called()


def test_cli_list_network_prints_network_only(capsys):
    with patch("llmb_collector.cli.collect") as mock_collect:
        with patch("sys.argv", ["llmb-collect", "list", "network"]):
            from llmb_collector.cli import main

            main()

    captured = capsys.readouterr()
    assert "Available network domains and sub-collectors:" in captured.out
    assert "interfaces: ip, ethtool" in captured.out
    assert "infiniband: ibstat, ibstat_list" in captured.out
    assert "Host domains:" not in captured.out
    assert "Container domains:" not in captured.out
    mock_collect.assert_not_called()


def test_cli_requires_at_least_one_collection_target(capsys):
    with patch("llmb_collector.cli.collect") as mock_collect:
        with patch("sys.argv", ["llmb-collect", "collect"]):
            from llmb_collector.cli import main

            main()

    captured = capsys.readouterr()
    assert "At least one collection target must be enabled" in captured.err
    assert "Host/system (enable with --host)" in captured.err
    assert "Network (enable with --network)" in captured.err
    assert "Container (enable with --container)" in captured.err
    assert "Environment (enable with --env)" in captured.err
    assert "nvidia-smi -q" in captured.err
    assert "lscpu --json" in captured.err
    mock_collect.assert_not_called()


def test_cli_validate_config_prints_findings(capsys):
    with patch("sys.argv", ["llmb-collect", "validate-config"]):
        from llmb_collector.cli import main

        main()

    captured = capsys.readouterr()
    assert "Command config findings:" in captured.out
    assert "duplicate_argv" in captured.out
    assert "nvidia_smi" in captured.out
    assert "container_nvidia_smi" in captured.out
