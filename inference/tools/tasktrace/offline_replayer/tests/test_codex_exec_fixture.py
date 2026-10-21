# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest
from offline_replayer.adapters.atif_replay import (
    ReplayError,
    read_trajectory,
    replay_trajectory,
    validate_supported_actions,
)

FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "representative_atif"
    / "terminalbench-codex-exec__fixture"
    / "agent"
    / "trajectory.json"
)

PROMISE_ALL_FIXTURE = (
    Path(__file__).with_name("fixtures")
    / "representative_atif"
    / "terminalbench-codex-exec-promise-all__fixture"
    / "agent"
    / "trajectory.json"
)


@pytest.mark.parametrize("fixture", [FIXTURE, PROMISE_ALL_FIXTURE])
def test_codex_exec_fixtures_are_rejected_before_execution(tmp_path: Path, fixture: Path) -> None:
    trajectory = read_trajectory(fixture)
    artifacts = tmp_path / "sessions"
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        validate_supported_actions(trajectory)
    with pytest.raises(ReplayError, match="not supported in release 26.10"):
        replay_trajectory(trajectory, cwd=tmp_path, artifacts_dir=artifacts)
    assert list(tmp_path.iterdir()) == []
