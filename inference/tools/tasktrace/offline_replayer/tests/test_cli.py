# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
from offline_replayer import cli


def _run_module(*arguments: str) -> subprocess.CompletedProcess[str]:
    repo_root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(repo_root)
    return subprocess.run(
        [sys.executable, "-m", "offline_replayer", *arguments],
        cwd=repo_root,
        env=environment,
        text=True,
        capture_output=True,
        check=False,
    )


def test_top_level_help_exposes_architecture_groups(capsys) -> None:
    assert cli.main(["--help"]) == 0

    output = capsys.readouterr().out
    assert "python -m offline_replayer <group> <command>" in output
    assert "python -m offline_replayer run FILE.yaml" in output
    assert "Replay recorded tool-side work" in output
    assert "profile atif" in output
    assert "profile" in output
    assert "replay" in output
    assert "experiment" not in output
    assert "campaign" not in output
    assert "report" in output


def test_group_help_lists_supported_commands(capsys) -> None:
    assert cli.main(["report", "--help"]) == 0
    output = capsys.readouterr().out
    assert "compare-agentscry" in output
    assert "resource-timeline" in output
    assert "summarize-experiment" not in output
    assert "unified-view" in output
    assert "process-inventory-bpf" in output
    assert "process-inventory-bpf-repeatability" in output


@pytest.mark.parametrize(
    "command, expected_option",
    [
        ("process-inventory-bpf", "expected_manifest"),
        ("process-inventory-bpf-repeatability", "proof"),
    ],
)
def test_bpf_process_report_leaf_help_uses_public_command(command: str, expected_option: str, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["report", command, "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert f"python -m offline_replayer report {command}" in output
    assert expected_option in output


def test_dispatch_forwards_arguments_to_packaged_implementation(monkeypatch) -> None:
    observed: dict[str, object] = {}

    def fake_main(arguments: list[str] | None) -> int:
        observed["arguments"] = arguments
        return 17

    def fake_load_main(module: str):
        observed["module"] = module
        return fake_main

    monkeypatch.setattr(cli, "_load_main", fake_load_main)

    assert cli.main(["profile", "atif", "trajectory.json", "--image", "image"]) == 17
    assert observed == {
        "module": "adapters.atif_profile",
        "arguments": ["trajectory.json", "--image", "image"],
    }


def test_unknown_command_reports_available_group(capsys) -> None:
    assert cli.main(["report", "missing"]) == 2

    captured = capsys.readouterr()
    assert "unknown report command: missing" in captured.err
    assert "render-set" in captured.out


def test_removed_campaign_group_has_no_compatibility_alias(capsys) -> None:
    assert cli.main(["campaign", "--help"]) == 2

    captured = capsys.readouterr()
    assert "unknown command group: campaign" in captured.err
    assert "run FILE.yaml" in captured.out


def test_run_dispatch_and_leaf_help_use_public_command(monkeypatch, capsys) -> None:
    observed: dict[str, object] = {}

    def fake_main(arguments: list[str] | None) -> int:
        observed["arguments"] = arguments
        return 23

    def fake_load_main(module: str):
        observed["module"] = module
        return fake_main

    monkeypatch.setattr(cli, "_load_main", fake_load_main)

    assert cli.main(["run", "tasks.yaml"]) == 23
    assert observed == {
        "module": "runs.execute",
        "arguments": ["tasks.yaml"],
    }

    monkeypatch.undo()
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["run", "--help"])

    assert exit_info.value.code == 0
    assert "python -m offline_replayer run" in capsys.readouterr().out


def test_removed_experiment_group_has_no_compatibility_alias(capsys) -> None:
    assert cli.main(["experiment", "run", "old.yaml"]) == 2

    captured = capsys.readouterr()
    assert "unknown command group: experiment" in captured.err


@pytest.mark.parametrize("command", ("container", "atif", "trace", "transcript"))
def test_every_public_profile_command_exposes_coordinated_baseline(command: str, capsys) -> None:
    with pytest.raises(SystemExit) as exit_info:
        cli.main(["profile", command, "--help"])

    assert exit_info.value.code == 0
    output = capsys.readouterr().out
    assert "--coordinated-baseline" in output
    assert "--profile-set" in output
    assert "--memory-profile" in output
    assert "--memory-limit-bytes" in output


def test_module_entrypoint_and_leaf_help_are_usable_from_shell() -> None:
    top_level = _run_module("--help")
    leaf = _run_module("profile", "atif", "--help")

    assert top_level.returncode == 0
    assert "python -m offline_replayer <group> <command>" in top_level.stdout
    assert "python -m offline_replayer run FILE.yaml" in top_level.stdout
    assert leaf.returncode == 0
    assert "python -m offline_replayer profile atif" in leaf.stdout


def test_version_and_installed_command_label(monkeypatch, capsys) -> None:
    assert cli.main(["--version"]) == 0
    assert capsys.readouterr().out.strip() == "0.1.0"

    monkeypatch.setattr(sys, "argv", ["/tmp/venv/bin/tasktrace"])
    assert cli.main(["--help"]) == 0
    output = capsys.readouterr().out
    assert "usage: tasktrace run FILE.yaml" in output
    assert "tasktrace <group> <command>" in output
    assert "tasktrace run FILE.yaml" in output

    with pytest.raises(SystemExit) as exit_info:
        cli.main(["profile", "atif", "--help"])
    assert exit_info.value.code == 0
    assert "usage: tasktrace profile atif" in capsys.readouterr().out


def test_module_entrypoint_reports_unknown_command_from_shell() -> None:
    result = _run_module("report", "missing")

    assert result.returncode == 2
    assert "unknown report command: missing" in result.stderr
    assert "render-set" in result.stdout
