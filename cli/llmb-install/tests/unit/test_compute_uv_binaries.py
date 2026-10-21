# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for per-architecture uv binary installation."""

import shutil
import tarfile
from pathlib import Path
from unittest.mock import patch

from llmb_install.downloads.compute_uv import ActiveUv, setup_compute_uv_binaries


def _write_binary(path: Path, content: str) -> None:
    path.write_text(content)
    path.chmod(0o755)


def _prepare_install_dirs(tmp_path: Path) -> None:
    """Create the install ``bin/`` directory the installer sets up before calling setup."""
    (tmp_path / "install" / "bin").mkdir(parents=True)


def _make_uv_archive(path: Path, target: str = "uv-x86_64-unknown-linux-gnu") -> Path:
    source_dir = path.parent / target
    source_dir.mkdir()
    _write_binary(source_dir / "uv", "downloaded uv")
    _write_binary(source_dir / "uvx", "downloaded uvx")

    with tarfile.open(path, "w:gz") as tar:
        tar.add(source_dir / "uv", arcname=f"{target}/uv")
        tar.add(source_dir / "uvx", arcname=f"{target}/uvx")

    return path


def test_copies_active_host_uv_when_version_matches(tmp_path, monkeypatch):
    """Host arch uses active uv/uvx without a redundant download."""
    monkeypatch.delenv("LLMB_UV_VERSION", raising=False)
    _prepare_install_dirs(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    uv = source_dir / "uv"
    uvx = source_dir / "uvx"
    _write_binary(uv, "host uv")
    _write_binary(uvx, "host uvx")

    active_uv = ActiveUv(version="0.9.28", uv_path=str(uv), uvx_path=str(uvx))

    with (
        patch("llmb_install.downloads.compute_uv.platform.machine", return_value="x86_64"),
        patch("llmb_install.downloads.compute_uv._detect_active_uv", return_value=active_uv),
        patch("llmb_install.downloads.compute_uv.download_file") as mock_download,
    ):
        setup_compute_uv_binaries(str(tmp_path / "install"), "x86_64")

    assert (tmp_path / "install" / "bin" / "x86_64" / "uv").read_text() == "host uv"
    assert (tmp_path / "install" / "bin" / "x86_64" / "uvx").read_text() == "host uvx"
    mock_download.assert_not_called()


def test_downloads_compute_arch_when_arch_differs(tmp_path, monkeypatch):
    """Compute arch mismatch downloads the GNU tarball for the selected compute arch."""
    monkeypatch.delenv("LLMB_UV_VERSION", raising=False)
    _prepare_install_dirs(tmp_path)
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    uv = source_dir / "uv"
    uvx = source_dir / "uvx"
    _write_binary(uv, "host uv")
    _write_binary(uvx, "host uvx")
    archive = _make_uv_archive(tmp_path / "uv-aarch64-unknown-linux-gnu.tar.gz", "uv-aarch64-unknown-linux-gnu")
    active_uv = ActiveUv(version="0.9.28", uv_path=str(uv), uvx_path=str(uvx))
    urls = []

    def fake_download(url, dest_path):
        urls.append(url)
        shutil.copy2(archive, dest_path)

    with (
        patch("llmb_install.downloads.compute_uv.platform.machine", return_value="x86_64"),
        patch("llmb_install.downloads.compute_uv._detect_active_uv", return_value=active_uv),
        patch("llmb_install.downloads.compute_uv.download_file", side_effect=fake_download),
    ):
        setup_compute_uv_binaries(str(tmp_path / "install"), "aarch64")

    assert urls == ["https://github.com/astral-sh/uv/releases/download/0.9.28/uv-aarch64-unknown-linux-gnu.tar.gz"]
    assert (tmp_path / "install" / "bin" / "x86_64" / "uv").read_text() == "host uv"
    assert (tmp_path / "install" / "bin" / "aarch64" / "uv").read_text() == "downloaded uv"
    assert (tmp_path / "install" / "bin" / "aarch64" / "uvx").read_text() == "downloaded uvx"


def test_skips_existing_arch_dirs(tmp_path, monkeypatch):
    """Existing uv+uvx files are treated as an installed point-in-time snapshot."""
    monkeypatch.delenv("LLMB_UV_VERSION", raising=False)
    install_path = tmp_path / "install"
    for arch in ("x86_64", "aarch64"):
        arch_dir = install_path / "bin" / arch
        arch_dir.mkdir(parents=True)
        _write_binary(arch_dir / "uv", f"{arch} uv")
        _write_binary(arch_dir / "uvx", f"{arch} uvx")

    active_uv = ActiveUv(version="0.9.28", uv_path="/unused/uv", uvx_path="/unused/uvx")

    with (
        patch("llmb_install.downloads.compute_uv.platform.machine", return_value="x86_64"),
        patch("llmb_install.downloads.compute_uv._detect_active_uv", return_value=active_uv),
        patch("llmb_install.downloads.compute_uv.download_file") as mock_download,
    ):
        setup_compute_uv_binaries(str(install_path), "aarch64")

    assert (install_path / "bin" / "aarch64" / "uv").read_text() == "aarch64 uv"
    mock_download.assert_not_called()


def test_downloads_host_arch_when_uvx_is_missing(tmp_path, monkeypatch):
    """A missing active uvx falls back to the release tarball for host arch."""
    monkeypatch.delenv("LLMB_UV_VERSION", raising=False)
    _prepare_install_dirs(tmp_path)
    archive = _make_uv_archive(tmp_path / "uv-x86_64-unknown-linux-gnu.tar.gz")
    active_uv = ActiveUv(version="0.9.28", uv_path="/unused/uv", uvx_path=None)
    urls = []

    def fake_download(url, dest_path):
        urls.append(url)
        shutil.copy2(archive, dest_path)

    with (
        patch("llmb_install.downloads.compute_uv.platform.machine", return_value="x86_64"),
        patch("llmb_install.downloads.compute_uv._detect_active_uv", return_value=active_uv),
        patch("llmb_install.downloads.compute_uv.download_file", side_effect=fake_download),
    ):
        setup_compute_uv_binaries(str(tmp_path / "install"), "x86_64")

    assert urls == ["https://github.com/astral-sh/uv/releases/download/0.9.28/uv-x86_64-unknown-linux-gnu.tar.gz"]
    assert (tmp_path / "install" / "bin" / "x86_64" / "uvx").read_text() == "downloaded uvx"


def test_env_version_override_downloads_requested_version(tmp_path, monkeypatch):
    """LLMB_UV_VERSION overrides active uv version and uses the release tarball."""
    monkeypatch.setenv("LLMB_UV_VERSION", "0.10.0")
    _prepare_install_dirs(tmp_path)
    archive = _make_uv_archive(tmp_path / "uv-x86_64-unknown-linux-gnu.tar.gz")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    uv = source_dir / "uv"
    uvx = source_dir / "uvx"
    _write_binary(uv, "host uv")
    _write_binary(uvx, "host uvx")
    active_uv = ActiveUv(version="0.9.28", uv_path=str(uv), uvx_path=str(uvx))
    urls = []

    def fake_download(url, dest_path):
        urls.append(url)
        shutil.copy2(archive, dest_path)

    with (
        patch("llmb_install.downloads.compute_uv.platform.machine", return_value="x86_64"),
        patch("llmb_install.downloads.compute_uv._detect_active_uv", return_value=active_uv),
        patch("llmb_install.downloads.compute_uv.download_file", side_effect=fake_download),
    ):
        setup_compute_uv_binaries(str(tmp_path / "install"), "x86_64")

    assert urls == ["https://github.com/astral-sh/uv/releases/download/0.10.0/uv-x86_64-unknown-linux-gnu.tar.gz"]
    assert (tmp_path / "install" / "bin" / "x86_64" / "uv").read_text() == "downloaded uv"
