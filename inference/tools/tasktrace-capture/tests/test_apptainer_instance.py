# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import logging
from pathlib import Path
from unittest import mock

import pytest
from harbor.environments.base import ExecResult
from tasktrace_capture.apptainer.config import ApptainerSettings, RuntimeLayout
from tasktrace_capture.apptainer.instance import ApptainerInstance
from tasktrace_capture.apptainer.mounts import BindMount
from tasktrace_capture.apptainer.process import CommandTimeoutError


def _settings(tmp_path: Path) -> ApptainerSettings:
    return ApptainerSettings(
        binary="/opt/apptainer/bin/apptainer",
        layout=RuntimeLayout(tmp_path / "runtime", "tasktrace-capture-hb-test"),
    )


def test_instance_owns_start_stop_and_force_fallback(tmp_path: Path):
    ok = ExecResult(return_code=0)
    graceful_failed = ExecResult(return_code=1, stderr="still running")
    runner = mock.Mock()
    runner.run = mock.AsyncMock(side_effect=[ok, ok, graceful_failed, ok])
    settings = _settings(tmp_path)
    instance = ApptainerInstance(
        settings=settings,
        runner=runner,
        logger=logging.getLogger("test.apptainer.instance"),
        gpus=0,
    )

    mount = BindMount(tmp_path, "/workspace", read_only=True)

    async def scenario():
        await instance.start(tmp_path / "base.sif", [mount], timeout_sec=30)
        assert instance.started is True
        await instance.stop()

    asyncio.run(scenario())

    start_command = runner.run.await_args_list[1].args[0]
    assert start_command[:3] == [settings.binary, "instance", "start"]
    assert "--overlay" in start_command
    assert mount.apptainer_spec() in start_command
    assert instance.started is False
    force_command = runner.run.await_args_list[3].args[0]
    assert force_command[-3:] == ["stop", "--force", instance.name]


def test_exec_timeout_stops_the_owned_instance(tmp_path: Path):
    ok = ExecResult(return_code=0)
    runner = mock.Mock()
    runner.run = mock.AsyncMock(
        side_effect=[
            ok,
            ok,
            CommandTimeoutError("Command timed out after 0.1s"),
            ok,
        ]
    )
    instance = ApptainerInstance(
        settings=_settings(tmp_path),
        runner=runner,
        logger=logging.getLogger("test.apptainer.instance"),
        gpus=0,
    )

    async def scenario():
        await instance.start(tmp_path / "base.sif", [], timeout_sec=30)
        with pytest.raises(RuntimeError, match="timed out"):
            await instance.exec(["sleep", "60"], timeout_sec=0.1)

    asyncio.run(scenario())
    assert instance.started is False
    force_command = runner.run.await_args_list[-1].args[0]
    assert force_command[-3:] == ["stop", "--force", instance.name]


def test_instance_detects_root_mapped_only_fakeroot(tmp_path: Path):
    ok = ExecResult(return_code=0)
    root_mapped = ExecResult(
        return_code=0,
        stderr=(
            "INFO: User not listed in /etc/subuid\n"
            "INFO: fakeroot command not found, using only root-mapped namespace\n"
        ),
    )
    runner = mock.Mock()
    runner.run = mock.AsyncMock(side_effect=[ok, root_mapped])
    instance = ApptainerInstance(
        settings=_settings(tmp_path),
        runner=runner,
        logger=logging.getLogger("test.apptainer.instance"),
        gpus=0,
    )

    asyncio.run(instance.start(tmp_path / "base.sif", [], timeout_sec=30))

    assert instance.root_mapped_only is True
