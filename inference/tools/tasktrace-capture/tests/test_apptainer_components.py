# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import shutil
from pathlib import Path
from unittest import mock

import pytest
from harbor.environments.base import ExecResult
from tasktrace_capture.apptainer.config import ApptainerSettings, RuntimeLayout
from tasktrace_capture.apptainer.images import SifImageStore


def _settings(tmp_path: Path) -> ApptainerSettings:
    return ApptainerSettings(
        binary="/opt/apptainer/bin/apptainer",
        layout=RuntimeLayout(tmp_path / "runtime", "tasktrace-capture-hb-test"),
    )


def test_sif_store_resolves_local_image_and_parses_metadata(tmp_path: Path):
    image = tmp_path / "base.sif"
    image.touch()
    runner = mock.Mock()
    runner.run = mock.AsyncMock(
        side_effect=[
            ExecResult(return_code=0, stdout="1 | root | FS (Squashfs/*System)\n"),
            ExecResult(
                return_code=0,
                stdout="Architecture: aarch64\nOffset: 32768\n",
            ),
        ]
    )
    store = SifImageStore(
        settings=_settings(tmp_path),
        runner=runner,
        environment_dir=tmp_path,
    )

    async def scenario():
        resolved = await store.resolve(str(image))
        metadata = await store.inspect(image)
        return resolved, metadata

    resolved, metadata = asyncio.run(scenario())
    assert resolved == image
    assert metadata.path == image
    assert metadata.architecture == "arm64"


def test_sif_store_rejects_oci_and_force_build(tmp_path: Path):
    store = SifImageStore(
        settings=_settings(tmp_path),
        runner=mock.Mock(),
        environment_dir=tmp_path,
    )

    with pytest.raises(ValueError, match="does not convert"):
        asyncio.run(store.resolve("docker://ubuntu:24.04"))
    with pytest.raises(ValueError, match="does not support --force-build"):
        asyncio.run(store.resolve(str(tmp_path / "base.sif"), force=True))


def test_runtime_layout_remove_handles_zero_mode_overlay_work_dir(
    tmp_path: Path,
):
    layout = RuntimeLayout(tmp_path / "runtime", "tasktrace-capture-hb-test")
    layout.prepare()
    blocked = layout.overlay / "work" / "work"
    blocked.mkdir()
    blocked.chmod(0)

    layout.remove()

    assert not layout.trial.exists()


def test_runtime_layout_remove_retries_transient_overlay_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    layout = RuntimeLayout(tmp_path / "runtime", "tasktrace-capture-hb-test")
    layout.prepare()
    original_rmtree = shutil.rmtree
    calls = 0

    def transient_rmtree(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("overlay still busy")
        return original_rmtree(path, *args, **kwargs)

    monkeypatch.setattr("tasktrace_capture.apptainer.config.shutil.rmtree", transient_rmtree)

    layout.remove()

    assert calls == 2
    assert not layout.trial.exists()


def test_runtime_layout_remove_handles_post_delete_overlay_recreation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    layout = RuntimeLayout(tmp_path / "runtime", "tasktrace-capture-hb-test")
    layout.prepare()
    original_rmtree = shutil.rmtree
    calls = 0

    def recreating_rmtree(path, *args, **kwargs):
        nonlocal calls
        calls += 1
        original_rmtree(path, *args, **kwargs)
        if calls == 1:
            blocked = Path(path) / "overlay" / "work" / "work"
            blocked.mkdir(parents=True)
            blocked.chmod(0)

    monkeypatch.setattr("tasktrace_capture.apptainer.config.shutil.rmtree", recreating_rmtree)

    layout.remove()

    assert calls == 2
    assert not layout.trial.exists()
