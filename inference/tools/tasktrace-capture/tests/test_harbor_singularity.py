# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import logging
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest
from harbor.environments.singularity import SingularityEnvironment
from harbor.models.environment_type import EnvironmentType
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths


def _make_environment(
    tmp_path: Path,
    *,
    docker_image: str,
) -> SingularityEnvironment:
    environment_dir = tmp_path / "task" / "environment"
    environment_dir.mkdir(parents=True)
    # Harbor passes --pwd before bootstrap can create its default /app.
    # Use an existing image directory so the rest of the lifecycle can run.
    (environment_dir / "Dockerfile").write_text("WORKDIR /tmp\n")

    trial_paths = TrialPaths(tmp_path / "trial")
    trial_paths.mkdir()

    python_wrapper = tmp_path / "python3-wrapper"
    python_wrapper.write_text(
        "#!/bin/sh\n" 'exec "${TASKTRACE_CAPTURE_HARBOR_IMAGE_PYTHON:-/usr/local/bin/python3}" "$@"\n'
    )
    python_wrapper.chmod(0o755)

    resolver_candidates = (
        Path("/run/systemd/resolve/resolv.conf"),
        Path("/etc/resolv.conf").resolve(),
    )
    resolver = next(path for path in resolver_candidates if path.is_file())

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
        {
            "type": "bind",
            "source": str(resolver),
            "target": "/etc/resolv.conf",
        },
        {
            "type": "bind",
            "source": "/etc/hosts",
            "target": "/etc/hosts",
        },
        {
            "type": "bind",
            "source": str(python_wrapper),
            "target": "/usr/bin/python3",
        },
    ]
    return SingularityEnvironment(
        environment_dir=environment_dir,
        environment_name="harbor-singularity-test",
        session_id="harbor-singularity-test__123",
        trial_paths=trial_paths,
        task_env_config=EnvironmentConfig(
            docker_image=docker_image,
            build_timeout_sec=300,
        ),
        logger=logging.getLogger("test.harbor-singularity"),
        mounts=mounts,
        singularity_image_cache_dir=tmp_path / "singularity-cache",
    )


def test_harbor_registers_singularity_environment():
    assert SingularityEnvironment.type() is EnvironmentType.SINGULARITY


def _server_pids(port: int) -> list[int]:
    result = subprocess.run(
        ["pgrep", "-f", f"_hbexec.py --port {port}"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode == 1:
        return []
    result.check_returncode()
    return [int(line) for line in result.stdout.splitlines()]


def _wait_for_server_exit(port: int, timeout_sec: float) -> bool:
    deadline = time.monotonic() + timeout_sec
    while _server_pids(port):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.1)
    return True


def _kill_server_pids(port: int) -> None:
    for pid in _server_pids(port):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@pytest.mark.skipif(
    not os.environ.get("TASKTRACE_CAPTURE_HARBOR_SINGULARITY_E2E_SIF"),
    reason=("set TASKTRACE_CAPTURE_HARBOR_SINGULARITY_E2E_SIF to run Harbor's real " "Singularity integration"),
)
def test_real_harbor_singularity_lifecycle(
    tmp_path,
    monkeypatch: pytest.MonkeyPatch,
):
    sif_path = Path(os.environ["TASKTRACE_CAPTURE_HARBOR_SINGULARITY_E2E_SIF"]).resolve()
    assert sif_path.is_file()
    monkeypatch.setenv(
        "TASKTRACE_CAPTURE_HARBOR_HOST_ENV_SENTINEL",
        "host-environment-leaked",
    )
    environment = _make_environment(tmp_path, docker_image=str(sif_path))

    async def scenario():
        await environment.start(force_build=False)
        try:
            first = await environment.exec(
                "set -eu; "
                "test \"$(id -u)\" = 0; "
                "mkdir -p /opt/harbor-singularity; "
                "printf persisted > /opt/harbor-singularity/state; "
                "printf mounted > /logs/agent/harbor-singularity-e2e.txt; "
                "printf stdout-value; "
                "printf stderr-value >&2",
                env={"HARBOR_SINGULARITY_EXEC_ENV": "visible"},
            )
            assert first.return_code == 0
            assert "stdout-value" in (first.stdout or "")
            assert "stderr-value" in (first.stdout or "")
            assert first.stderr is None

            second = await environment.exec(
                "test \"$(cat /opt/harbor-singularity/state)\" = persisted && "
                "test \"$HARBOR_SINGULARITY_EXEC_ENV\" = visible",
                env={"HARBOR_SINGULARITY_EXEC_ENV": "visible"},
            )
            assert second.return_code == 0

            leaked_env = await environment.exec(
                'if test -n "${TASKTRACE_CAPTURE_HARBOR_HOST_ENV_SENTINEL+x}"; then '
                "printf leaked; "
                "else "
                "printf isolated; "
                "fi"
            )
            assert leaked_env.return_code == 0
            assert leaked_env.stdout == "isolated"

            upload = tmp_path / "upload.txt"
            upload.write_text("round-trip")
            await environment.upload_file(
                upload,
                "/opt/harbor-singularity/upload.txt",
            )
            downloaded = tmp_path / "download.txt"
            await environment.download_file(
                "/opt/harbor-singularity/upload.txt",
                downloaded,
            )
            assert downloaded.read_text() == "round-trip"

            timeout = await environment.exec(
                "sleep 60 & " "child=$!; " "printf %s \"$child\" > /tmp/harbor-timeout-child.pid; " "wait \"$child\"",
                timeout_sec=1,
            )
            assert timeout.return_code == 124

            await asyncio.sleep(0.2)
            child_state = await environment.exec(
                "pid=$(cat /tmp/harbor-timeout-child.pid); "
                "if kill -0 \"$pid\" 2>/dev/null; then "
                "printf alive; "
                "else "
                "printf gone; "
                "fi"
            )
            assert child_state.return_code == 0
            assert child_state.stdout == "gone"
        finally:
            if environment._http_client and environment._server_port:
                try:
                    await environment._http_client.post(f"http://localhost:{environment._server_port}/shutdown")
                except Exception:
                    pass
            await environment.stop(delete=True)

    asyncio.run(scenario())

    assert (environment.trial_paths.agent_dir / "harbor-singularity-e2e.txt").read_text() == "mounted"
    assert environment._staging_dir is not None
    assert not environment._staging_dir.exists()
    assert environment._server_process is not None
    assert environment._server_process.returncode is not None

    assert environment._server_port is not None
    assert _wait_for_server_exit(environment._server_port, timeout_sec=5)


@pytest.mark.skipif(
    not os.environ.get("TASKTRACE_CAPTURE_HARBOR_SINGULARITY_E2E_SIF"),
    reason=("set TASKTRACE_CAPTURE_HARBOR_SINGULARITY_E2E_SIF to run Harbor's real " "Singularity integration"),
)
@pytest.mark.xfail(
    reason=(
        "Harbor stop() terminates the outer singularity process but leaves " "the in-container FastAPI server running"
    ),
    strict=True,
)
def test_real_harbor_singularity_stop_reaps_server(tmp_path):
    sif_path = Path(os.environ["TASKTRACE_CAPTURE_HARBOR_SINGULARITY_E2E_SIF"]).resolve()
    environment = _make_environment(tmp_path, docker_image=str(sif_path))

    async def scenario():
        await environment.start(force_build=False)
        result = await environment.exec("printf ready")
        assert result.return_code == 0
        assert result.stdout == "ready"
        await environment.stop(delete=True)

    asyncio.run(scenario())

    assert environment._server_port is not None
    try:
        assert _wait_for_server_exit(environment._server_port, timeout_sec=2)
    finally:
        _kill_server_pids(environment._server_port)
