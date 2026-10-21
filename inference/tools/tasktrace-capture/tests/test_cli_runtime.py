# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the --runtime flag wiring in cli.py."""

import argparse
from unittest import mock

import pytest
from tasktrace_capture import cli


def _build_parser():
    """Extract the argparse parser from cli.main by parsing help.

    The parser lives inside main(), so we parse known args via cli.main's
    entry path. Simpler: construct a fresh parser mirroring the relevant flags.
    For these tests we just validate behaviour via cli.main with heavy mocking.
    """
    # Minimal parser built to exercise just the --runtime behaviour.
    # This mirrors the real parser's --runtime definition.
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runtime",
        choices=["enroot", "docker", "apptainer", "opensandbox", "current"],
        default="enroot",
    )
    return p


# -- argparse basics ----------------------------------------------------------


def test_runtime_flag_defaults_to_enroot():
    p = _build_parser()
    args = p.parse_args([])
    assert args.runtime == "enroot"


def test_runtime_flag_accepts_current():
    p = _build_parser()
    args = p.parse_args(["--runtime", "current"])
    assert args.runtime == "current"


def test_runtime_flag_accepts_docker():
    p = _build_parser()
    args = p.parse_args(["--runtime", "docker"])
    assert args.runtime == "docker"


def test_runtime_flag_accepts_apptainer():
    p = _build_parser()
    args = p.parse_args(["--runtime", "apptainer"])
    assert args.runtime == "apptainer"


def test_runtime_flag_accepts_opensandbox():
    p = _build_parser()
    args = p.parse_args(["--runtime", "opensandbox"])
    assert args.runtime == "opensandbox"


def test_runtime_flag_rejects_invalid():
    p = _build_parser()
    with pytest.raises(SystemExit):
        p.parse_args(["--runtime", "gvisor"])


# -- CLI smoke ----------------------------------------------------------------


def test_runtime_flag_visible_in_help(capsys):
    """--runtime must appear in tasktrace-capture --help output."""
    with pytest.raises(SystemExit):
        with mock.patch("sys.argv", ["tasktrace-capture", "--help"]):
            cli.main()
    out = capsys.readouterr().out
    assert "--runtime" in out
    assert "--instrument-shell-commands" in out
    assert "--instrument-shell-perfetto-depth" in out
    assert "--profile-live-cpu" in out
    assert "--profile-window" in out
    assert "--profile-scheduler" in out
    assert "--profile-cpuset-cpus" in out
    assert "--override-cpus" in out
    assert "enroot" in out
    assert "docker" in out
    assert "apptainer" in out
    assert "opensandbox" in out


# -- _preflight runtime delegation -------------------------------------------


def test_preflight_enroot_calls_runtime_preflight_checks():
    fake_checks = [("enroot on PATH", True, "")]
    with mock.patch(
        "tasktrace_capture.runtimes.runtime_preflight_checks",
        return_value=fake_checks,
    ) as helper, mock.patch("tasktrace_capture.monitor._iter_container_pids", return_value=[]), mock.patch(
        "urllib.request.urlopen"
    ):
        cli._preflight(
            api_base="http://example:8000/v1",
            quiet=True,
            agent="terminus-2",
            allow_private_api=True,
            runtime="enroot",
        )
    helper.assert_called_once_with("enroot")


def test_preflight_docker_calls_runtime_preflight_checks():
    fake_checks = [
        ("docker on PATH", True, ""),
        ("docker daemon reachable", True, ""),
    ]
    with mock.patch(
        "tasktrace_capture.runtimes.runtime_preflight_checks",
        return_value=fake_checks,
    ) as helper, mock.patch("tasktrace_capture.monitor._iter_container_pids", return_value=[]), mock.patch(
        "urllib.request.urlopen"
    ):
        cli._preflight(
            api_base="http://example:8000/v1",
            quiet=True,
            agent="terminus-2",
            allow_private_api=True,
            runtime="docker",
        )
    helper.assert_called_once_with("docker")


def test_preflight_parallel_docker_skips_process_check():
    with mock.patch(
        "tasktrace_capture.runtimes.runtime_preflight_checks",
        return_value=[],
    ), mock.patch(
        "tasktrace_capture.monitor._iter_container_pids",
        return_value=[],
    ), mock.patch("urllib.request.urlopen"), mock.patch("subprocess.run") as run:
        cli._preflight(
            api_base="http://example:8000/v1",
            quiet=True,
            allow_private_api=True,
            runtime="docker",
            allow_parallel_docker=True,
        )
    run.assert_not_called()


def test_preflight_apptainer_calls_runtime_preflight_checks():
    fake_checks = [
        ("apptainer on PATH", True, ""),
        ("unprivileged user namespaces", True, ""),
    ]
    with mock.patch(
        "tasktrace_capture.runtimes.runtime_preflight_checks",
        return_value=fake_checks,
    ) as helper, mock.patch("tasktrace_capture.monitor._iter_container_pids", return_value=[]), mock.patch(
        "urllib.request.urlopen"
    ):
        cli._preflight(
            api_base="http://example:8000/v1",
            quiet=True,
            agent="oracle",
            allow_private_api=True,
            runtime="apptainer",
        )
    helper.assert_called_once_with("apptainer")


def test_preflight_opensandbox_calls_runtime_preflight_checks():
    fake_checks = [
        ("OpenSandbox endpoint configured", True, ""),
        ("OpenSandbox API key configured", True, ""),
    ]
    with mock.patch(
        "tasktrace_capture.runtimes.runtime_preflight_checks",
        return_value=fake_checks,
    ) as helper, mock.patch("tasktrace_capture.monitor._iter_container_pids", return_value=[]), mock.patch(
        "urllib.request.urlopen"
    ):
        cli._preflight(
            api_base="",
            quiet=True,
            agent="oracle",
            allow_private_api=True,
            runtime="opensandbox",
        )
    helper.assert_called_once_with("opensandbox")


def test_preflight_opensandbox_allows_concurrent_coordinators():
    fake_checks = [
        ("OpenSandbox endpoint configured", True, ""),
        ("OpenSandbox API key configured", True, ""),
    ]
    with mock.patch(
        "tasktrace_capture.runtimes.runtime_preflight_checks",
        return_value=fake_checks,
    ), mock.patch(
        "tasktrace_capture.monitor._iter_container_pids", return_value=[]
    ) as container_scan, mock.patch("subprocess.run") as process_scan, mock.patch("urllib.request.urlopen"):
        cli._preflight(
            api_base="",
            quiet=True,
            agent="oracle",
            allow_private_api=True,
            runtime="opensandbox",
        )

    process_scan.assert_not_called()
    container_scan.assert_not_called()


# -- report header ------------------------------------------------------------


def test_report_header_enroot_backcompat_string_exact():
    """Exact string for enroot must match the original for report.json parsers."""
    from tasktrace_capture.cli import _environment_label_for_runtime

    assert _environment_label_for_runtime("enroot") == "EnrootEnvironment (enroot_env.py)"


def test_report_header_docker_string():
    from tasktrace_capture.cli import _environment_label_for_runtime

    assert _environment_label_for_runtime("docker") == "TasktraceCaptureDockerEnv (docker_env.py)"


def test_report_header_apptainer_string():
    from tasktrace_capture.cli import _environment_label_for_runtime

    assert _environment_label_for_runtime("apptainer") == ("ApptainerEnvironment (apptainer_env.py)")


def test_report_header_opensandbox_string():
    from tasktrace_capture.cli import _environment_label_for_runtime

    assert _environment_label_for_runtime("opensandbox") == ("OpenSandboxEnvironment (opensandbox_env.py)")


def test_system_versions_report_opensandbox_sdk():
    from tasktrace_capture.monitor import _collect_versions

    with mock.patch("importlib.metadata.version", return_value="0.1.15"), mock.patch(
        "tasktrace_capture.monitor._run", return_value="harbor 0.1.0"
    ) as run_mock:
        versions = _collect_versions("opensandbox")

    assert versions == {
        "opensandbox": "0.1.15",
        "harbor": "harbor 0.1.0",
    }
    run_mock.assert_called_once_with(["harbor", "version"])


def test_monitor_does_not_query_local_runtime_for_opensandbox():
    from tasktrace_capture.monitor import get_n_containers

    with mock.patch("tasktrace_capture.monitor.subprocess.run") as run_mock:
        assert get_n_containers("opensandbox") == 0
    run_mock.assert_not_called()


def test_system_versions_use_configured_apptainer_binary(monkeypatch):
    from tasktrace_capture.monitor import _collect_versions

    monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_BIN", "/opt/apptainer/bin/apptainer")
    outputs = {
        ("/opt/apptainer/bin/apptainer", "version"): ("apptainer version 1.5.0"),
        ("harbor", "version"): "harbor 0.1.0",
    }
    with mock.patch(
        "tasktrace_capture.monitor._run",
        side_effect=lambda command, **kwargs: outputs.get(tuple(command)),
    ) as run_mock:
        versions = _collect_versions("apptainer")

    assert versions == {
        "apptainer": "apptainer version 1.5.0",
        "harbor": "harbor 0.1.0",
    }
    assert [call.args[0] for call in run_mock.call_args_list] == [
        ["/opt/apptainer/bin/apptainer", "version"],
        ["harbor", "version"],
    ]


def test_system_versions_never_fall_back_to_path_apptainer(monkeypatch):
    from tasktrace_capture.monitor import _collect_versions

    monkeypatch.delenv("TASKTRACE_CAPTURE_APPTAINER_BIN", raising=False)
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value=None,
    ), mock.patch(
        "tasktrace_capture.monitor._run",
        return_value="harbor 0.1.0",
    ) as run_mock:
        versions = _collect_versions("apptainer")

    assert versions == {"harbor": "harbor 0.1.0"}
    run_mock.assert_called_once_with(["harbor", "version"])


def test_monitor_counts_only_tasktrace_capture_apptainer_instances():
    from tasktrace_capture.monitor import get_n_containers

    listed = mock.Mock(
        returncode=0,
        stdout=(
            '{"instances": ['
            '{"instance": "tasktrace-capture-hb-task-one"},'
            '{"instance": "unrelated-service"},'
            '{"name": "tasktrace-capture-hb-task-two"}'
            "]}"
        ),
    )
    with mock.patch(
        "tasktrace_capture.apptainer_env.apptainer_binary",
        return_value="/opt/apptainer/bin/apptainer",
    ), mock.patch(
        "tasktrace_capture.monitor.subprocess.run",
        return_value=listed,
    ) as run_mock:
        assert get_n_containers("apptainer") == 2

    assert run_mock.call_args.args[0] == [
        "/opt/apptainer/bin/apptainer",
        "instance",
        "list",
        "--json",
    ]


def test_monitor_counts_harbor_docker_containers():
    from tasktrace_capture.monitor import get_n_containers

    listed = mock.Mock(returncode=0, stdout="abc123\ndef456\n")
    with mock.patch(
        "tasktrace_capture.monitor.subprocess.run",
        return_value=listed,
    ) as run_mock:
        assert get_n_containers("docker") == 2

    args = run_mock.call_args.args[0]
    assert "label=com.docker.compose.service=main" in args
    assert "name=__" in args


# -- output-dir whitespace check ---------------------------------------------


def test_output_dir_whitespace_rejected_for_enroot():
    from tasktrace_capture.cli import _validate_output_dir_whitespace

    # For enroot, whitespace in path is rejected.
    assert _validate_output_dir_whitespace("/foo bar/baz", runtime="enroot") is False
    assert _validate_output_dir_whitespace("/foobar/baz", runtime="enroot") is True


def test_output_dir_whitespace_allowed_for_docker():
    from tasktrace_capture.cli import _validate_output_dir_whitespace

    assert _validate_output_dir_whitespace("/foo bar/baz", runtime="docker") is True


# -- SIGINT / per-level cleanup -----------------------------------------------


def test_shutdown_handler_calls_runtime_cleanup():
    """The SIGINT handler must route cleanup through runtime_cleanup."""
    with mock.patch("tasktrace_capture.runtimes.runtime_cleanup") as rc, mock.patch(
        "tasktrace_capture.cli.get_active_harbor_pgid", return_value=0
    ), mock.patch("tasktrace_capture.cli.sys.exit"):
        cli._run_shutdown("docker", signum=15, docker_owner="rt123")
    rc.assert_called_once_with("docker", "rt123")


# -- report.json contract ----------------------------------------------------


def test_report_json_contains_runtime_field(tmp_path):
    from tasktrace_capture.cli import _write_report_json

    _write_report_json(
        {
            "dataset": "hello-world",
            "model": "m",
            "api_base": "",
            "timestamp": "x",
            "hostname": "h",
            "runtime": "docker",
            "cpu_info": {},
            "baseline_cpu": {},
            "system_info": {},
            "time_splits": {},
            "results": [],
        },
        tmp_path,
    )
    import json

    data = json.loads((tmp_path / "report.json").read_text())
    assert data["runtime"] == "docker"


def test_report_json_runtime_default_enroot_when_missing(tmp_path):
    """Legacy reports without runtime field must still load; default = None."""
    from tasktrace_capture.cli import _write_report_json

    _write_report_json(
        {
            "dataset": "hello-world",
            "model": "m",
            "api_base": "",
            "timestamp": "x",
            "hostname": "h",
            "cpu_info": {},
            "baseline_cpu": {},
            "system_info": {},
            "time_splits": {},
            "results": [],
        },
        tmp_path,
    )
    import json

    data = json.loads((tmp_path / "report.json").read_text())
    # Missing runtime in input → None (or absent) in output. Consumers must
    # handle this for legacy reports.
    assert data.get("runtime") in (None, "enroot")
