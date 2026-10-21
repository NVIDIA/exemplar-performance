# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Install per-architecture uv binaries for compute-node setup tasks."""

import os
import platform
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from llmb_install.constants import UV_BASE_URL, UV_GNU_TARGETS
from llmb_install.environment.detector import normalize_architecture
from llmb_install.utils.download import download_file
from llmb_install.utils.logging import get_logger

logger = get_logger(__name__)

_UV_BINARIES = ("uv", "uvx")


@dataclass(frozen=True)
class ActiveUv:
    """The uv binaries resolved in the installer environment."""

    version: str | None
    uv_path: str | None
    uvx_path: str | None


def setup_compute_uv_binaries(install_path: str, compute_arch: str) -> None:
    """Populate $LLMB_INSTALL/bin/<arch>/ with uv and uvx.

    The host architecture is always populated. If the selected compute-node
    architecture differs from the host architecture, the compute architecture is
    populated from the matching GNU uv release tarball.

    Args:
        install_path: Base LLMB installation directory.
        compute_arch: Selected compute-node architecture ('x86_64' or 'aarch64').
    """
    host_arch = normalize_architecture(platform.machine())
    compute_arch = normalize_architecture(compute_arch)
    active_uv = _detect_active_uv()
    version = _resolve_requested_uv_version(active_uv.version)

    arch_list = [host_arch]
    if compute_arch != host_arch:
        arch_list.append(compute_arch)

    bin_root = Path(install_path) / "bin"

    print("\nConfiguring per-architecture uv binaries.")
    print("-----------------------------------------")
    print(f"uv version: {version}")

    for arch in arch_list:
        _ensure_uv_for_arch(install_path, bin_root, arch, host_arch, version, active_uv)


def _detect_active_uv() -> ActiveUv:
    """Find the active uv/uvx binaries and parse the active uv version."""
    uv_path = shutil.which("uv")
    uvx_path = shutil.which("uvx")
    version = None

    if uv_path:
        try:
            result = subprocess.run([uv_path, "--version"], check=True, capture_output=True, text=True)
            version = _parse_uv_version(result.stdout.strip())
        except subprocess.CalledProcessError as e:
            logger.warning(f"Could not determine active uv version: {e}")

    return ActiveUv(version=version, uv_path=uv_path, uvx_path=uvx_path)


def _parse_uv_version(output: str) -> str | None:
    """Parse `uv --version` output such as `uv 0.9.28`."""
    parts = output.split()
    if len(parts) >= 2 and parts[0] == "uv":
        return parts[1]
    return None


def _resolve_requested_uv_version(active_version: str | None) -> str:
    """Resolve the uv version requested for LLMB_BIN."""
    env_version = os.environ.get("LLMB_UV_VERSION", "").strip()
    if env_version:
        return env_version

    if active_version:
        return active_version

    raise RuntimeError("Could not determine uv version. Set LLMB_UV_VERSION or ensure uv is available in PATH.")


def _ensure_uv_for_arch(
    install_path: str,
    bin_root: Path,
    arch: str,
    host_arch: str,
    version: str,
    active_uv: ActiveUv,
) -> None:
    """Install uv/uvx for one architecture unless both binaries already exist."""
    arch_dir = bin_root / arch
    if _has_uv_binaries(arch_dir):
        print(f"Skipping uv binaries for {arch} -- already installed at {arch_dir}")
        return

    if arch == host_arch:
        copyable = _copyable_active_paths(version, active_uv)
        if copyable is not None:
            print(f"Copying active uv binaries for {arch}...")
            _copy_active_uv(copyable, bin_root, arch_dir)
            return

    print(f"Downloading uv {version} for {arch}...")
    archive_path = _download_uv_archive(install_path, arch, version)
    _install_uv_from_archive(archive_path, bin_root, arch_dir)


def _has_uv_binaries(arch_dir: Path) -> bool:
    """Return True when an architecture directory already has uv and uvx."""
    return all((arch_dir / binary).is_file() for binary in _UV_BINARIES)


def _copyable_active_paths(version: str, active_uv: ActiveUv) -> tuple[str, str] | None:
    """Return the active (uv, uvx) paths when they match the requested version, else None."""
    if (
        active_uv.version == version
        and active_uv.uv_path is not None
        and active_uv.uvx_path is not None
        and Path(active_uv.uv_path).is_file()
        and Path(active_uv.uvx_path).is_file()
    ):
        return active_uv.uv_path, active_uv.uvx_path
    return None


def _copy_active_uv(paths: tuple[str, str], bin_root: Path, arch_dir: Path) -> None:
    """Copy the given active uv and uvx into a staged arch directory."""
    uv_path, uvx_path = paths
    with _staged_arch_dir(bin_root, arch_dir) as stage_dir:
        shutil.copy2(uv_path, stage_dir / "uv")
        shutil.copy2(uvx_path, stage_dir / "uvx")


def _download_uv_archive(install_path: str, arch: str, version: str) -> Path:
    """Download a uv GNU release tarball into the install cache."""
    target = UV_GNU_TARGETS.get(arch)
    if not target:
        raise ValueError(f"Unsupported architecture for uv download: {arch}")

    filename = f"uv-{target}.tar.gz"
    download_url = f"{UV_BASE_URL}/{version}/{filename}"
    cache_dir = Path(install_path) / ".cache" / "tools" / "uv" / version
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached_archive = cache_dir / filename

    if cached_archive.exists():
        print("  Using cached uv archive")
        return cached_archive

    print(f"  Downloading from: {download_url}")
    temp_download = cached_archive.with_suffix(cached_archive.suffix + ".tmp")
    try:
        download_file(download_url, temp_download)
        temp_download.replace(cached_archive)
        return cached_archive
    finally:
        if temp_download.exists():
            temp_download.unlink()


def _install_uv_from_archive(archive_path: Path, bin_root: Path, arch_dir: Path) -> None:
    """Extract uv and uvx from a release archive into an arch directory."""
    with _staged_arch_dir(bin_root, arch_dir) as stage_dir, tarfile.open(archive_path, "r:gz") as tar:
        for member in tar.getmembers():
            binary_name = Path(member.name).name
            if binary_name not in _UV_BINARIES or not member.isfile():
                continue

            source = tar.extractfile(member)
            if source is None:
                continue

            with source, open(stage_dir / binary_name, "wb") as dest:
                shutil.copyfileobj(source, dest)


@contextmanager
def _staged_arch_dir(bin_root: Path, arch_dir: Path) -> Iterator[Path]:
    """Yield a temp staging dir, then verify it and replace ``arch_dir``.

    The caller writes uv/uvx into the yielded directory. On clean exit the staged
    files are verified and the directory is renamed into place; on any error the
    partial staging directory is removed and the original ``arch_dir`` left untouched.
    """
    stage_dir = Path(tempfile.mkdtemp(prefix=f".{arch_dir.name}.", dir=bin_root))
    try:
        yield stage_dir
        _verify_and_chmod(stage_dir)
        _replace_arch_dir(stage_dir, arch_dir)
        print(f"✓ Installed uv binaries for {arch_dir.name} at {arch_dir}")
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)


def _verify_and_chmod(stage_dir: Path) -> None:
    """Verify staged uv binaries and mark them executable."""
    missing = [binary for binary in _UV_BINARIES if not (stage_dir / binary).is_file()]
    if missing:
        raise RuntimeError(f"uv archive did not contain required binaries: {', '.join(missing)}")

    for binary in _UV_BINARIES:
        os.chmod(stage_dir / binary, 0o755)


def _replace_arch_dir(stage_dir: Path, arch_dir: Path) -> None:
    """Replace the final architecture directory with the staged files."""
    if arch_dir.exists():
        shutil.rmtree(arch_dir)
    stage_dir.rename(arch_dir)
