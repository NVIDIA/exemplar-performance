# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from offline_replayer.adapters.atif_profile import (
    AtifTrajectoryProfileConfig,
    build_container_profile_config,
)
from offline_replayer.adapters.atif_replay import (
    ReplayError,
    read_trajectory,
    replay_trajectory,
)

CALLS = [
    ("exec_command", {"cmd": "touch actual"}),
    ("write_stdin", {"session_id": 44, "chars": "touch actual\n"}),
    ("apply_patch", {"input": "*** Begin Patch\n*** Add File: actual\n+changed\n*** End Patch"}),
    ("exec", {"input": "await tools.exec_command({cmd:'touch actual'});"}),
    ("exec", {"input": "await Promise.all([tools.exec_command({cmd:'touch actual'})]);"}),
    ("update_plan", {"plan": []}),
]


def document(tool: str, arguments: dict, nested: bool) -> dict:
    codex = {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "codex",
        "agent": {"name": "codex", "extra": {"originator": "codex_exec"}},
        "steps": [
            {"timestamp": 0, "tool_calls": [{"tool_call_id": "a", "function_name": tool, "arguments": arguments}]}
        ],
    }
    if not nested:
        return codex
    return {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": "root",
        "agent": {"name": "generic"},
        "steps": [
            {"timestamp": 0, "tool_calls": [{"function_name": "exec_command", "arguments": {"cmd": "touch actual"}}]}
        ],
        "subagent_trajectories": [codex],
    }


@pytest.mark.parametrize("tool,arguments", CALLS)
@pytest.mark.parametrize("nested", [False, True])
def test_codex_exec_rejection_prevents_all_replay_side_effects(
    tmp_path: Path, tool: str, arguments: dict, nested: bool
) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(document(tool, arguments, nested)))
    trajectory = read_trajectory(path)
    with pytest.raises(ReplayError, match="Codex exec.*not supported in release 26.10"):
        replay_trajectory(
            trajectory, cwd=tmp_path, artifacts_dir=tmp_path / "sessions", action_output_dir=tmp_path / "outputs"
        )
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trajectory.json"]


@pytest.mark.parametrize("runtime", ["docker", "enroot", "current"])
@pytest.mark.parametrize("nested", [False, True])
def test_codex_exec_profile_rejection_precedes_runtime_setup(tmp_path: Path, runtime: str, nested: bool) -> None:
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(document(*CALLS[0], nested)))
    config = AtifTrajectoryProfileConfig(
        trajectory=path,
        image="unused:latest",
        output_dir=tmp_path / "profile",
        cpu_limit=1,
        cpuset_cpus=None,
        allocated_cpus=1,
        pacing="unpaced",
        timeout_sec=2,
        prepare_command=None,
        strace_mode="none",
        string_limit=1000,
        shell="/bin/bash",
        workdir="/testbed",
        runtime=runtime,
    )
    with pytest.raises(ReplayError, match="Codex exec.*not supported in release 26.10"):
        build_container_profile_config(config)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trajectory.json"]


def test_empty_nested_codex_exec_node_is_still_rejected(tmp_path: Path) -> None:
    payload = document(*CALLS[0], True)
    payload["subagent_trajectories"][0]["steps"] = []
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ReplayError, match="Codex exec.*not supported in release 26.10"):
        replay_trajectory(read_trajectory(path), cwd=tmp_path)
    assert not (tmp_path / "actual").exists()
