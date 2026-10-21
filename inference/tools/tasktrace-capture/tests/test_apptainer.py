# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from pathlib import Path
from unittest import mock

import pytest
from harbor.environments.capabilities import EnvironmentResourceCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from tasktrace_capture.apptainer.images import swebench_sif_path
from tasktrace_capture.apptainer_env import (
    APPTAINER_INSTANCE_PREFIX,
    ApptainerEnvironment,
    _mount_spec,
    _sanitize_instance_name,
)


def _make_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    docker_image: str,
) -> ApptainerEnvironment:
    configured_binary = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_BIN")
    if configured_binary is None and not os.environ.get("TASKTRACE_CAPTURE_APPTAINER_E2E_SIF"):
        fake_binary = tmp_path / "fake-apptainer"
        fake_binary.write_text("#!/bin/sh\n")
        fake_binary.chmod(0o755)
        configured_binary = str(fake_binary)
    if configured_binary is not None:
        monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_BIN", configured_binary)
    monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_TMPDIR", str(tmp_path / "apptainer-runtime"))

    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)

    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()
    mounts = [
        {
            "type": "bind",
            "source": str(trial_paths.agent_dir),
            "target": "/logs/agent",
        },
        {
            "type": "bind",
            "source": str(trial_paths.verifier_dir),
            "target": "/logs/verifier",
        },
        {
            "type": "bind",
            "source": str(trial_paths.artifacts_dir),
            "target": "/logs/artifacts",
        },
    ]
    return ApptainerEnvironment(
        environment_dir=environment_dir,
        environment_name="test-task",
        session_id="Test.Task__123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=docker_image,
            build_timeout_sec=180,
        ),
        logger=logging.getLogger("test.apptainer"),
        mounts=mounts,
    )


def test_instance_name_is_scoped_and_stable():
    first = _sanitize_instance_name("Task.Name__abc")
    second = _sanitize_instance_name("Task.Name__abc")
    assert first == second
    assert first.startswith(APPTAINER_INSTANCE_PREFIX)
    assert len(first) <= 96


def test_swebench_sif_template(tmp_path, monkeypatch):
    environment = tmp_path / "psf__requests-1921" / "environment"
    environment.mkdir(parents=True)
    monkeypatch.setenv(
        "TASKTRACE_CAPTURE_SWEBENCH_SIF_TEMPLATE",
        "/sifs/swebench_sweb.eval.arm64.{instance_id}.sif",
    )
    assert swebench_sif_path(environment) == ("/sifs/swebench_sweb.eval.arm64.psf__requests-1921.sif")


def test_swebench_sif_template_does_not_match_pro_task(tmp_path, monkeypatch):
    environment = tmp_path / "instance_repo__repo-deadbeef-vabc" / "environment"
    environment.mkdir(parents=True)
    monkeypatch.setenv("TASKTRACE_CAPTURE_SWEBENCH_SIF_TEMPLATE", "/sifs/{instance_id}.sif")
    assert swebench_sif_path(environment) is None


def test_mount_spec_supports_spaces_and_rejects_commas(tmp_path):
    source = tmp_path / "path with spaces"
    assert _mount_spec(source, "/target path") == (f"type=bind,src={source},dst=/target path")
    with pytest.raises(ValueError, match="cannot contain commas"):
        _mount_spec(tmp_path / "bad,path", "/target")


def test_resource_capabilities_do_not_claim_cgroup_enforcement():
    assert ApptainerEnvironment.resource_capabilities() == EnvironmentResourceCapabilities()


def test_subprocess_env_strips_host_container_env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("APPTAINERENV_LEAK", "host")
    monkeypatch.setenv("APPTAINER_MOUNT", "type=bind,src=/,dst=/host")
    monkeypatch.setenv("APPTAINER_OVERLAY", "/tmp/untrusted.img")
    monkeypatch.setenv("APPTAINER_DOCKER_PASSWORD", "secret")
    monkeypatch.setenv("DOCKER_PASSWORD", "secret-alias")
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/nonexistent.sif",
    )
    runtime_env = environment._subprocess_env({"VISIBLE": "yes"})
    assert "APPTAINERENV_LEAK" not in runtime_env
    assert "APPTAINER_MOUNT" not in runtime_env
    assert "APPTAINER_OVERLAY" not in runtime_env
    assert "APPTAINER_DOCKER_PASSWORD" not in runtime_env
    assert "DOCKER_PASSWORD" not in runtime_env
    assert runtime_env["APPTAINERENV_VISIBLE"] == "yes"


def test_environment_rejects_oci_images(tmp_path, monkeypatch: pytest.MonkeyPatch):
    with pytest.raises(ValueError, match="existing .sif files only"):
        _make_environment(
            tmp_path,
            monkeypatch,
            docker_image="registry.example/team/image:tag",
        )


def test_resolved_mounts_include_host_dns(tmp_path, monkeypatch: pytest.MonkeyPatch):
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/nonexistent.sif",
    )
    mounts = environment._resolved_mounts()
    by_target = {mount.target: (mount.source, mount.read_only) for mount in mounts}
    assert by_target["/etc/resolv.conf"] == (
        Path("/etc/resolv.conf").resolve(),
        True,
    )
    assert by_target["/etc/hosts"] == (Path("/etc/hosts").resolve(), True)


def test_matching_architecture_is_accepted(tmp_path, monkeypatch: pytest.MonkeyPatch):
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/base.sif",
    )
    environment._sif_architecture = mock.AsyncMock(return_value="arm64")
    monkeypatch.setattr(
        "tasktrace_capture.apptainer.config.platform.machine",
        lambda: "aarch64",
    )

    asyncio.run(environment._validate_image_arch(Path("/tmp/base.sif")))


def test_foreign_architecture_is_rejected(tmp_path, monkeypatch: pytest.MonkeyPatch):
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/base.sif",
    )
    environment._sif_architecture = mock.AsyncMock(return_value="amd64")
    monkeypatch.setattr(
        "tasktrace_capture.apptainer.config.platform.machine",
        lambda: "aarch64",
    )

    with pytest.raises(RuntimeError, match="does not match host architecture"):
        asyncio.run(environment._validate_image_arch(Path("/tmp/base.sif")))


def test_start_failure_force_stops_partial_instance_and_removes_state(tmp_path, monkeypatch: pytest.MonkeyPatch):
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/base.sif",
    )
    sif_path = tmp_path / "base.sif"
    sif_path.touch()
    environment._pull_image = mock.AsyncMock(return_value=sif_path)
    environment._validate_image_arch = mock.AsyncMock()
    instance = mock.Mock(started=False)
    instance.start = mock.AsyncMock(side_effect=RuntimeError("start failed"))
    instance.force_stop = mock.AsyncMock()
    environment._create_instance = mock.Mock(return_value=instance)

    with pytest.raises(RuntimeError, match="start failed"):
        asyncio.run(environment.start(force_build=False))

    instance.force_stop.assert_awaited_once()
    assert environment._started is False
    assert not environment._runtime_dir.exists()


def test_stop_failure_force_stops_and_removes_state(tmp_path, monkeypatch: pytest.MonkeyPatch):
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/base.sif",
    )
    environment._prepare_runtime_dirs()
    instance = mock.Mock(started=True)
    instance.stop = mock.AsyncMock(side_effect=RuntimeError("stop timed out"))
    environment._instance = instance
    environment._started = True

    with pytest.raises(RuntimeError, match="stop timed out"):
        asyncio.run(environment.stop(delete=True))

    instance.stop.assert_awaited_once()
    assert environment._started is False
    assert not environment._runtime_dir.exists()


def test_stop_cancellation_still_removes_state(tmp_path, monkeypatch: pytest.MonkeyPatch):
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image="/tmp/base.sif",
    )
    environment._prepare_runtime_dirs()
    instance = mock.Mock(started=True)
    instance.stop = mock.AsyncMock(side_effect=asyncio.CancelledError)
    environment._instance = instance
    environment._started = True

    with pytest.raises(asyncio.CancelledError):
        asyncio.run(environment.stop(delete=True))

    instance.stop.assert_awaited_once()
    assert environment._started is False
    assert not environment._runtime_dir.exists()


@pytest.mark.skipif(
    not os.environ.get("TASKTRACE_CAPTURE_APPTAINER_E2E_SIF"),
    reason="set TASKTRACE_CAPTURE_APPTAINER_E2E_SIF to run real Apptainer integration",
)
def test_real_apptainer_native_sif_lifecycle(tmp_path, monkeypatch: pytest.MonkeyPatch):
    sif_path = Path(os.environ["TASKTRACE_CAPTURE_APPTAINER_E2E_SIF"]).resolve()
    assert sif_path.is_file()
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image=str(sif_path),
    )
    environment.task_env_config.workdir = "/opt/tasktrace-capture-task"
    (environment.environment_dir / "environment-marker.txt").write_text("uploaded")
    monkeypatch.setenv("TASKTRACE_CAPTURE_HOST_ONLY_SENTINEL", "must-not-leak")

    async def scenario():
        await environment.start(force_build=False)
        try:
            first = await environment.exec(
                "mkdir -p /opt/tasktrace-capture-e2e && "
                "printf persisted > /opt/tasktrace-capture-e2e/state && "
                "id -u && "
                'test "$(cat environment-marker.txt)" = uploaded && '
                'test -z "${TASKTRACE_CAPTURE_HOST_ONLY_SENTINEL:-}" && '
                'printf "%s\\n" "$TASKTRACE_CAPTURE_EXEC_ENV"',
                env={"TASKTRACE_CAPTURE_EXEC_ENV": "visible"},
            )
            assert first.return_code == 0
            assert first.stdout is not None
            assert first.stdout.splitlines() == ["0", "visible"]

            second = await environment.exec(
                'test "$(cat /opt/tasktrace-capture-e2e/state)" = persisted && '
                "printf mounted > /logs/agent/apptainer-e2e.txt"
            )
            assert second.return_code == 0

            upload = tmp_path / "upload.txt"
            upload.write_text("round-trip")
            await environment.upload_file(upload, "/opt/tasktrace-capture-e2e/upload.txt")
            downloaded = tmp_path / "download.txt"
            await environment.download_file("/opt/tasktrace-capture-e2e/upload.txt", downloaded)
            assert downloaded.read_text() == "round-trip"
        finally:
            await environment.stop(delete=True)

    asyncio.run(scenario())
    assert (environment.trial_paths.agent_dir / "apptainer-e2e.txt").read_text() == "mounted"
    assert not environment._runtime_dir.exists()


@pytest.mark.skipif(
    not os.environ.get("TASKTRACE_CAPTURE_APPTAINER_E2E_SIF"),
    reason="set TASKTRACE_CAPTURE_APPTAINER_E2E_SIF to run real Apptainer integration",
)
def test_real_apptainer_timeout_stops_instance_and_removes_state(tmp_path, monkeypatch: pytest.MonkeyPatch):
    sif_path = Path(os.environ["TASKTRACE_CAPTURE_APPTAINER_E2E_SIF"]).resolve()
    assert sif_path.is_file()
    environment = _make_environment(
        tmp_path,
        monkeypatch,
        docker_image=str(sif_path),
    )

    async def scenario():
        await environment.start(force_build=False)
        try:
            with pytest.raises(RuntimeError, match="timed out"):
                await environment.exec("sleep 60", timeout_sec=0.25)
            assert environment._started is False
        finally:
            await environment.stop(delete=True)

    asyncio.run(scenario())

    listed = subprocess.run(
        [environment._apptainer, "instance", "list", "--json"],
        check=True,
        capture_output=True,
        text=True,
        cwd="/",
        env=environment._subprocess_env(),
        timeout=10,
    )
    active_names = {
        instance.get("instance") or instance.get("name")
        for instance in json.loads(listed.stdout or "{}").get("instances", [])
    }
    assert environment._instance_name not in active_names
    assert not environment._runtime_dir.exists()
