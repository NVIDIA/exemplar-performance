# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest
from offline_replayer.adapters.atif_replay import (
    ReplayError,
    _trajectory_nodes,
    read_trajectory,
    replay_trajectory,
)


def node(identity: str, timestamp: int, children: list | None = None, command: str | None = None) -> dict:
    children = children or []
    return {
        "schema_version": "ATIF-v1.7",
        "trajectory_id": identity,
        "agent": {"name": "codex"},
        "steps": [
            {
                "timestamp": timestamp,
                "tool_calls": [
                    {
                        "tool_call_id": identity,
                        "function_name": "exec_command",
                        "arguments": {
                            "cmd": command or f"printf {identity} >> order.txt",
                        },
                    }
                ],
                "observation": {
                    "results": [
                        {
                            "source_call_id": identity,
                            "extra": {"tool_result_is_error": False},
                            "subagent_trajectory_ref": [
                                {"trajectory_id": child["trajectory_id"]} for child in children
                            ],
                        }
                    ]
                },
            }
        ],
        "subagent_trajectories": children,
    }


def load(tmp_path: Path, document: dict):
    path = tmp_path / "trajectory.json"
    path.write_text(json.dumps(document))
    return read_trajectory(path)


def test_recursive_children_replay_in_timestamp_order_with_provenance(tmp_path: Path) -> None:
    trajectory = load(tmp_path, node("root", 3, [node("child", 2, [node("grandchild", 1)])]))
    assert [a.source_trajectory_id for a in trajectory.actions] == ["grandchild", "child", "root"]
    assert [a.index for a in trajectory.actions] == [1, 2, 3]
    report = replay_trajectory(trajectory, cwd=tmp_path, pacing="unpaced", artifacts_dir=tmp_path / "sessions")
    assert (tmp_path / "order.txt").read_text() == "grandchildchildroot"
    assert report["subagent_trajectories_flattened"] == 2
    assert report["summary"]["tool_actions_recorded"] == report["summary"]["tool_actions_replayed"] == 3
    assert report["summary"]["source_outcome_mismatch_count"] == 0
    assert [a["source_trajectory_id"] for a in report["actions"]] == ["grandchild", "child", "root"]


def test_equal_timestamps_preserve_root_then_child_order(tmp_path: Path) -> None:
    trajectory = load(tmp_path, node("root", 1, [node("child", 1)]))
    assert [a.source_trajectory_id for a in trajectory.actions] == ["root", "child"]


@pytest.mark.parametrize(
    "malformation, message",
    [
        ("dangling", "unembedded subagent"),
        ("duplicate", "duplicated"),
        ("missing_id", "has no trajectory_id"),
        ("bad_child", "not an object"),
        ("bad_children", "not an array"),
        ("bad_ref", "reference has no trajectory_id"),
        ("self_ref", "unembedded subagent"),
    ],
)
def test_incomplete_child_provenance_is_rejected(tmp_path: Path, malformation: str, message: str) -> None:
    document = node("root", 0, [node("child", 1)])
    refs = document["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"]
    if malformation == "dangling":
        document["subagent_trajectories"] = []
    elif malformation == "duplicate":
        document["subagent_trajectories"].append(node("child", 2))
    elif malformation == "missing_id":
        document["subagent_trajectories"][0].pop("trajectory_id")
    elif malformation == "bad_child":
        document["subagent_trajectories"] = [42]
    elif malformation == "bad_children":
        document["subagent_trajectories"] = {}
    elif malformation == "bad_ref":
        refs[0] = {}
    elif malformation == "self_ref":
        refs[0]["trajectory_id"] = "root"
    with pytest.raises(ReplayError, match=message):
        load(tmp_path, document)


def test_duplicate_identity_bounds_in_memory_cycle() -> None:
    document = node("root", 0)
    document["subagent_trajectories"] = [document]
    with pytest.raises(ReplayError, match="duplicated"):
        _trajectory_nodes(document, None)


def test_preflattened_references_are_allowed_only_at_root() -> None:
    document = node("root", 0)
    document["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "flat-child"}]
    assert len(_trajectory_nodes(document, {"root", "flat-child"})) == 1
    document["subagent_trajectories"] = [node("child", 1)]
    child = document["subagent_trajectories"][0]
    child["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"] = [{"trajectory_id": "flat-child"}]
    with pytest.raises(ReplayError, match="unembedded subagent"):
        _trajectory_nodes(document, {"root", "flat-child"})
