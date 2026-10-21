# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

import pytest
from offline_replayer.adapters.atif_replay import read_trajectory, replay_trajectory
from offline_replayer.adapters.transcript_replay import ReplayError

FIXTURE_ROOT = Path(__file__).with_name("fixtures") / "representative_atif"


def _trajectory(name: str):
    return read_trajectory(FIXTURE_ROOT / name / "agent" / "trajectory.json")


def test_representative_fixture_manifest_is_complete() -> None:
    manifest = json.loads((FIXTURE_ROOT / "manifest.json").read_text())

    assert manifest["schema_version"] == "representative-atif-fixtures.v1"
    assert {item["dataset"] for item in manifest["fixtures"]} == {
        "DeepSWE",
        "SWE-bench Pro",
        "Terminal-Bench 2.0",
        "negative contract",
    }
    for item in manifest["fixtures"]:
        assert (FIXTURE_ROOT / item["path"]).is_file()
        source_sha256 = item["source_sha256"]
        assert source_sha256 is None or (len(source_sha256) == 64 and set(source_sha256) <= set("0123456789abcdef"))
        assert item["traits"]
        assert item["curation"]


def test_deepswe_mini_swe_fixture_replays_outcomes_and_combined_output(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "action_outputs"
    trajectory = _trajectory("deepswe-mini-swe__fixture")

    report = replay_trajectory(
        trajectory,
        cwd=tmp_path,
        pacing="unpaced",
        timeout_sec=2,
        action_output_dir=output_dir,
    )

    assert trajectory.source_agent_original_format == "mini-swe-agent-1.1"
    assert [action.source_tool for action in trajectory.actions] == ["bash"] * 4
    assert [action.tool for action in trajectory.actions] == ["exec_command"] * 4
    assert report["summary"]["actions_executed"] == 3
    assert report["summary"]["actions_failed"] == 2
    assert report["summary"]["actions_skipped_not_executed"] == 1
    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert report["actions"][1]["returncode"] == 2
    assert report["actions"][2]["returncode"] == 7
    assert not (tmp_path / "skipped.txt").exists()
    assert (output_dir / "action-0001.output.txt").read_text() == ("deep-stdout\ndeep-stderr\n")
    assert (output_dir / "action-0003.output.txt").read_text() == "source-failure\n"


def test_swebenchpro_codex_fixture_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        replay_trajectory(_trajectory("swebenchpro-codex__fixture"), cwd=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_terminalbench_codex_exec_fixture_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        replay_trajectory(_trajectory("terminalbench-codex-exec__fixture"), cwd=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_swebenchpro_malformed_call_fixture_is_rejected_before_execution(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        replay_trajectory(_trajectory("swebenchpro-malformed-call__fixture"), cwd=tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_generic_bare_bash_fixture_remains_rejected(tmp_path: Path) -> None:
    with pytest.raises(ReplayError, match="unsupported executable tool calls"):
        replay_trajectory(
            _trajectory("generic-bare-bash__fixture"),
            cwd=tmp_path,
            pacing="unpaced",
        )
