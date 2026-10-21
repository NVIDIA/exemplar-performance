# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Selection and staging of TaskTrace Capture's userspace Apptainer runtime."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import pwd
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

MANAGED_APPTAINER_VERSION = "1.5.2"
MANAGED_APPTAINER_RELEASE = f"{MANAGED_APPTAINER_VERSION}-upstream.1"
UPSTREAM_INSTALLER_URL = (
    "https://raw.githubusercontent.com/apptainer/apptainer/"
    f"v{MANAGED_APPTAINER_VERSION}/tools/install-unprivileged.sh"
)
UPSTREAM_INSTALLER_SHA256 = "4d84a839a0d97177f2852b565ccbb60d8364a1ba7a6adaf00e654bfc229bba7e"
BOOTSTRAP_IMAGE = "docker://ubuntu:24.04"

_ARCHITECTURES = {
    "aarch64": ("arm64", "aarch64"),
    "arm64": ("arm64", "aarch64"),
    "x86_64": ("amd64", "x86_64"),
    "amd64": ("amd64", "x86_64"),
}
_ARCHIVE_ROOT = "apptainer"
_INSTALL_TIMEOUT_SEC = 15 * 60


@dataclass(frozen=True)
class ApptainerInstallation:
    """The selected runtime and where it came from."""

    binary: Path | None
    source: str
    detail: str = ""
    external_binary: Path | None = None
    nss_wrapper: Path | None = None

    @property
    def available(self) -> bool:
        return self.binary is not None

    @property
    def managed(self) -> bool:
        return self.source == "managed"


@dataclass(frozen=True)
class BundleArtifact:
    """A verified, relocatable managed-runtime archive."""

    path: Path
    sha256: str


def _architecture(value: str | None = None) -> tuple[str, str] | None:
    machine = platform.machine() if value is None else value
    return _ARCHITECTURES.get(machine.strip().lower())


def managed_install_root(architecture: str | None = None) -> Path:
    """Return the node-local path used by TaskTrace Capture's managed runtime."""
    configured = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_MANAGED_ROOT")
    if configured:
        return Path(configured).expanduser()

    selected = _architecture(architecture)
    suffix = selected[0] if selected is not None else "unsupported"
    node_tmp = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
    return (
        Path(node_tmp)
        / f"tasktrace-capture-apptainer-{os.getuid()}"
        / f"apptainer-{MANAGED_APPTAINER_RELEASE}-{suffix}"
    )


def managed_cache_root() -> Path:
    """Return the persistent cache used to avoid rebuilding each allocation."""
    configured = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg_cache).expanduser() if xdg_cache else Path.home() / ".cache"
    return base / "tasktrace-capture" / "apptainer" / MANAGED_APPTAINER_RELEASE


def _marker_path(install_root: Path) -> Path:
    return install_root / ".tasktrace-capture-install.json"


def _nss_wrapper_path(install_root: Path) -> Path:
    return install_root / "tasktrace-capture" / "lib" / "libnss_wrapper.so"


def _managed_files(
    install_root: Path,
    architecture: str,
) -> tuple[Path, Path] | None:
    binary = install_root / "bin" / "apptainer"
    nss_wrapper = _nss_wrapper_path(install_root)
    marker = _marker_path(install_root)
    if not all(
        (
            binary.is_file(),
            os.access(binary, os.X_OK),
            nss_wrapper.is_file(),
            marker.is_file(),
        )
    ):
        return None
    try:
        payload = json.loads(marker.read_text())
    except (OSError, ValueError):
        return None
    if (
        payload.get("release") != MANAGED_APPTAINER_RELEASE
        or payload.get("architecture") != architecture
        or payload.get("installer_sha256") != UPSTREAM_INSTALLER_SHA256
    ):
        return None
    return binary.resolve(), nss_wrapper.resolve()


def _external_apptainer() -> Path | None:
    resolved = shutil.which("apptainer")
    return Path(resolved).resolve() if resolved else None


def inspect_apptainer_installation() -> ApptainerInstallation:
    """Inspect selection state without downloading or changing files."""
    configured = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_BIN")
    external = _external_apptainer()
    if configured:
        binary = Path(configured).expanduser()
        if binary.is_file() and os.access(binary, os.X_OK):
            return ApptainerInstallation(
                binary=binary.resolve(),
                source="explicit",
                detail="selected by TASKTRACE_CAPTURE_APPTAINER_BIN",
                external_binary=external,
            )
        return ApptainerInstallation(
            binary=None,
            source="explicit",
            detail=("TASKTRACE_CAPTURE_APPTAINER_BIN does not name an executable file: " f"{binary}"),
            external_binary=external,
        )

    selected = _architecture()
    if selected is None:
        return ApptainerInstallation(
            binary=None,
            source="managed",
            detail=(
                "TaskTrace Capture's managed Apptainer supports arm64 and amd64, not "
                f"{platform.machine()!r}; set TASKTRACE_CAPTURE_APPTAINER_BIN to an "
                "explicit installation"
            ),
            external_binary=external,
        )

    architecture, _ = selected
    install_root = managed_install_root()
    files = _managed_files(install_root, architecture)
    if files is not None:
        binary, nss_wrapper = files
        return ApptainerInstallation(
            binary=binary,
            source="managed",
            detail=("TaskTrace Capture-managed, unprivileged Apptainer " f"{MANAGED_APPTAINER_VERSION}"),
            external_binary=external,
            nss_wrapper=nss_wrapper,
        )
    if install_root.exists():
        detail = (
            f"incomplete managed Apptainer installation at {install_root}; "
            "remove that TaskTrace Capture-owned directory and retry"
        )
    else:
        detail = f"managed Apptainer is not staged at {install_root}"
    return ApptainerInstallation(
        binary=None,
        source="managed",
        detail=detail,
        external_binary=external,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _download(
    url: str,
    destination: Path,
    *,
    expected_sha256: str | None = None,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "tasktrace-capture"})
        with urllib.request.urlopen(request, timeout=60) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output, length=1024 * 1024)
                output.flush()
                os.fsync(output.fileno())
        temporary.chmod(0o700)
        if expected_sha256 is not None and _sha256(temporary) != expected_sha256:
            raise RuntimeError(f"downloaded file failed SHA-256 verification: {url}")
        os.replace(temporary, destination)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"download failed ({exc.code}) for {url}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"download failed for {url}: {exc.reason}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _verified_installer(cache_root: Path) -> Path:
    configured = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_INSTALLER")
    if configured:
        source = Path(configured).expanduser()
        if not source.is_file():
            raise RuntimeError(f"Apptainer installer does not exist: {source}")
        if _sha256(source) != UPSTREAM_INSTALLER_SHA256:
            raise RuntimeError(f"Apptainer installer SHA-256 mismatch for {source}")
        return source.resolve()

    installer = cache_root / f"install-unprivileged-{MANAGED_APPTAINER_VERSION}.sh"
    if installer.is_file() and _sha256(installer) == UPSTREAM_INSTALLER_SHA256:
        return installer
    installer.unlink(missing_ok=True)
    url = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_INSTALLER_URL", UPSTREAM_INSTALLER_URL)
    _download(url, installer, expected_sha256=UPSTREAM_INSTALLER_SHA256)
    return installer


def _clean_bootstrap_environment() -> dict[str, str]:
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("APPTAINER_", "SINGULARITY_")):
            env.pop(key, None)
    env["LC_ALL"] = "C"
    return env


def _run_checked(
    command: list[str],
    *,
    env: Mapping[str, str],
    timeout_sec: int,
    label: str,
) -> None:
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=dict(env),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"{label} failed: {exc}") from exc
    if result.returncode == 0:
        return
    output = "\n".join(value.strip() for value in (result.stdout, result.stderr) if value.strip())
    if len(output) > 4000:
        output = output[-4000:]
    raise RuntimeError(f"{label} failed with status {result.returncode}" + (f":\n{output}" if output else ""))


def _find_host_nss_wrapper() -> Path | None:
    roots = (Path("/usr/lib"), Path("/usr/lib64"), Path("/lib"), Path("/lib64"))
    for root in roots:
        candidate = root / "libnss_wrapper.so"
        if candidate.is_file():
            return candidate
        try:
            candidates = tuple(root.glob("*-linux-gnu/libnss_wrapper.so"))
        except OSError:
            candidates = ()
        if candidates:
            return candidates[0]
    return None


def _install_with_host_tools(
    installer: Path,
    destination: Path,
    upstream_architecture: str,
    nss_wrapper: Path,
) -> None:
    command = [
        "bash",
        str(installer),
        "-e",
        "-a",
        upstream_architecture,
        "-v",
        MANAGED_APPTAINER_VERSION,
        str(destination),
    ]
    _run_checked(
        command,
        env=_clean_bootstrap_environment(),
        timeout_sec=_INSTALL_TIMEOUT_SEC,
        label="Apptainer's official unprivileged installer",
    )
    target = _nss_wrapper_path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(nss_wrapper, target)
    target.chmod(0o755)


def _bootstrap_image(cache_root: Path, workspace: Path) -> Path:
    image = cache_root / f"bootstrap-ubuntu-24.04-{platform.machine()}.sqsh"
    if image.is_file() and image.stat().st_size > 0:
        return image

    temporary = image.with_name(f".{image.name}.{uuid.uuid4().hex}.tmp")
    enroot_env = _clean_bootstrap_environment()
    enroot_env.update(
        {
            "ENROOT_CACHE_PATH": str(workspace / "import-cache"),
            "ENROOT_TEMP_PATH": str(workspace / "import-tmp"),
        }
    )
    Path(enroot_env["ENROOT_CACHE_PATH"]).mkdir(parents=True)
    Path(enroot_env["ENROOT_TEMP_PATH"]).mkdir(parents=True)
    uri = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_BOOTSTRAP_IMAGE", BOOTSTRAP_IMAGE)
    try:
        _run_checked(
            ["enroot", "import", "--output", str(temporary), uri],
            env=enroot_env,
            timeout_sec=_INSTALL_TIMEOUT_SEC,
            label=f"Enroot import of the Apptainer installer image ({uri})",
        )
        os.replace(temporary, image)
    finally:
        temporary.unlink(missing_ok=True)
    return image


def _install_with_enroot(
    installer: Path,
    destination: Path,
    upstream_architecture: str,
    cache_root: Path,
    workspace: Path,
) -> None:
    image = _bootstrap_image(cache_root, workspace)
    mount_dir = workspace / "bootstrap-mount"
    mount_dir.mkdir()
    staged_installer = mount_dir / "install-unprivileged.sh"
    shutil.copy2(installer, staged_installer)
    staged_installer.chmod(0o755)

    data_path = workspace / "enroot-data"
    runtime_path = workspace / "enroot-runtime"
    data_path.mkdir()
    runtime_path.mkdir()
    name = f"tasktrace-capture-apptainer-bootstrap-{uuid.uuid4().hex[:12]}"
    enroot_env = _clean_bootstrap_environment()
    enroot_env.update(
        {
            "ENROOT_DATA_PATH": str(data_path),
            "ENROOT_RUNTIME_PATH": str(runtime_path),
        }
    )
    _run_checked(
        ["enroot", "create", "--name", name, str(image)],
        env=enroot_env,
        timeout_sec=120,
        label="Enroot creation of the Apptainer installer container",
    )

    version = shlex.quote(MANAGED_APPTAINER_VERSION)
    architecture = shlex.quote(upstream_architecture)
    bootstrap = f"""
set -euo pipefail
apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates cpio curl libnss-wrapper rpm2cpio
/mnt/install-unprivileged.sh -e -a {architecture} -v {version} /mnt/apptainer
nss_wrapper="$(dpkg -L libnss-wrapper | grep '/libnss_wrapper\\.so$' | head -n 1)"
test -n "$nss_wrapper"
install -D -m 0755 "$nss_wrapper" \
    /mnt/apptainer/tasktrace-capture/lib/libnss_wrapper.so
""".strip()
    _run_checked(
        [
            "enroot",
            "start",
            "--root",
            "--rw",
            "--mount",
            f"{mount_dir}:/mnt",
            name,
            "/bin/bash",
            "-lc",
            bootstrap,
        ],
        env=enroot_env,
        timeout_sec=_INSTALL_TIMEOUT_SEC,
        label="Apptainer userspace installation inside Enroot",
    )
    os.replace(mount_dir / "apptainer", destination)


def _identity_name() -> str:
    candidate = os.environ.get("USER") or f"tasktrace-capture-{os.getuid()}"
    candidate = re.sub(r"[^A-Za-z0-9_.-]", "_", candidate)
    return candidate or f"tasktrace-capture-{os.getuid()}"


def _write_identity_files(install_root: Path) -> None:
    identity_dir = install_root / "tasktrace-capture" / "etc"
    identity_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    username = _identity_name()
    uid = os.getuid()
    gid = os.getgid()
    home = str(Path.home()).replace(":", "_").replace("\n", "_")
    passwd = identity_dir / "passwd"
    passwd.write_text(f"{username}:x:{uid}:{gid}:TaskTrace Capture user:{home}:/bin/sh\n")

    gids = sorted({gid, *os.getgroups()})
    groups = []
    for current_gid in gids:
        name = username if current_gid == gid else f"tasktrace-capture-{current_gid}"
        groups.append(f"{name}:x:{current_gid}:{username}")
    group = identity_dir / "group"
    group.write_text("\n".join(groups) + "\n")
    passwd.chmod(0o600)
    group.chmod(0o600)


def _host_identity_available() -> bool:
    try:
        pwd.getpwuid(os.getuid())
    except KeyError:
        return False
    return True


def managed_compatibility_environment(
    environment: Mapping[str, str],
    installation: ApptainerInstallation | None = None,
) -> dict[str, str]:
    """Add a private NSS database when the allocation exposes only a numeric UID."""
    env = dict(environment)
    selected = installation or inspect_apptainer_installation()
    if not selected.managed or _host_identity_available():
        return env
    if selected.binary is None or selected.nss_wrapper is None:
        return env

    install_root = selected.binary.parent.parent
    passwd = install_root / "tasktrace-capture" / "etc" / "passwd"
    group = install_root / "tasktrace-capture" / "etc" / "group"
    if not passwd.is_file() or not group.is_file():
        return env
    env["LD_PRELOAD"] = str(selected.nss_wrapper)
    env["NSS_WRAPPER_PASSWD"] = str(passwd)
    env["NSS_WRAPPER_GROUP"] = str(group)
    return env


def _validate_install(install_root: Path, architecture: str) -> None:
    binary = install_root / "bin" / "apptainer"
    nss_wrapper = _nss_wrapper_path(install_root)
    if not binary.is_file() or not os.access(binary, os.X_OK):
        raise RuntimeError("official installer did not produce bin/apptainer")
    if not nss_wrapper.is_file():
        raise RuntimeError("managed install is missing libnss_wrapper.so")

    _write_identity_files(install_root)
    installation = ApptainerInstallation(
        binary=binary,
        source="managed",
        nss_wrapper=nss_wrapper,
    )
    env = managed_compatibility_environment(_clean_bootstrap_environment(), installation)
    try:
        result = subprocess.run(
            [str(binary), "version"],
            capture_output=True,
            text=True,
            timeout=30,
            env=env,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"managed Apptainer validation failed: {exc}") from exc
    output = (result.stdout or result.stderr or "").strip()
    if result.returncode != 0 or MANAGED_APPTAINER_VERSION not in output:
        raise RuntimeError("managed Apptainer validation failed" + (f": {output}" if output else ""))

    expected_architecture = _architecture()
    if expected_architecture is None or expected_architecture[0] != architecture:
        raise RuntimeError("host architecture changed during Apptainer installation")


def _create_archive(source: Path, destination: Path) -> BundleArtifact:
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    try:
        with tarfile.open(temporary, "w:gz") as archive:
            archive.add(source, arcname=_ARCHIVE_ROOT, recursive=True)
        temporary.chmod(0o600)
        digest = _sha256(temporary)
        os.replace(temporary, destination)
        return BundleArtifact(path=destination, sha256=digest)
    finally:
        temporary.unlink(missing_ok=True)


def _bundle_metadata_path(archive: Path) -> Path:
    return archive.with_suffix(archive.suffix + ".json")


def _load_cached_bundle(archive: Path, architecture: str) -> BundleArtifact | None:
    metadata_path = _bundle_metadata_path(archive)
    if not archive.is_file() or not metadata_path.is_file():
        return None
    try:
        metadata = json.loads(metadata_path.read_text())
    except (OSError, ValueError):
        return None
    if (
        metadata.get("release") != MANAGED_APPTAINER_RELEASE
        or metadata.get("architecture") != architecture
        or metadata.get("installer_sha256") != UPSTREAM_INSTALLER_SHA256
    ):
        return None
    expected = metadata.get("archive_sha256")
    if not isinstance(expected, str) or _sha256(archive) != expected:
        return None
    return BundleArtifact(path=archive, sha256=expected)


def _publish_bundle_metadata(artifact: BundleArtifact, architecture: str) -> None:
    metadata_path = _bundle_metadata_path(artifact.path)
    temporary = metadata_path.with_name(f".{metadata_path.name}.{uuid.uuid4().hex}.tmp")
    payload = {
        "release": MANAGED_APPTAINER_RELEASE,
        "version": MANAGED_APPTAINER_VERSION,
        "architecture": architecture,
        "installer_sha256": UPSTREAM_INSTALLER_SHA256,
        "archive_sha256": artifact.sha256,
    }
    try:
        temporary.write_text(json.dumps(payload, sort_keys=True) + "\n")
        temporary.chmod(0o600)
        os.replace(temporary, metadata_path)
    finally:
        temporary.unlink(missing_ok=True)


def _build_cached_bundle(
    cache_root: Path,
    architecture: str,
    upstream_architecture: str,
) -> BundleArtifact:
    cache_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    archive = cache_root / (f"apptainer-{MANAGED_APPTAINER_RELEASE}-linux-{architecture}.tar.gz")
    lock_path = cache_root / f".{architecture}.bundle.lock"
    with lock_path.open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        cached = _load_cached_bundle(archive, architecture)
        if cached is not None:
            return cached

        archive.unlink(missing_ok=True)
        _bundle_metadata_path(archive).unlink(missing_ok=True)
        node_tmp = os.environ.get("SLURM_TMPDIR") or tempfile.gettempdir()
        workspace = Path(tempfile.mkdtemp(prefix="tasktrace-capture-apptainer-build-", dir=node_tmp))
        try:
            installer = _verified_installer(cache_root)
            destination = workspace / _ARCHIVE_ROOT
            host_nss_wrapper = _find_host_nss_wrapper()
            host_tools = all(shutil.which(command) is not None for command in ("bash", "cpio", "curl", "rpm2cpio"))
            if host_tools and host_nss_wrapper is not None:
                _install_with_host_tools(
                    installer,
                    destination,
                    upstream_architecture,
                    host_nss_wrapper,
                )
            elif shutil.which("enroot") is not None:
                _install_with_enroot(
                    installer,
                    destination,
                    upstream_architecture,
                    cache_root,
                    workspace,
                )
            else:
                raise RuntimeError(
                    "building TaskTrace Capture's userspace Apptainer requires either "
                    "Enroot, or host curl/rpm2cpio/cpio plus libnss_wrapper"
                )
            _validate_install(destination, architecture)
            artifact = _create_archive(destination, archive)
            _publish_bundle_metadata(artifact, architecture)
            return artifact
        finally:
            shutil.rmtree(workspace, ignore_errors=True)


def _source_archive(
    architecture: str,
    upstream_architecture: str,
) -> BundleArtifact:
    configured = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_BUNDLE")
    if configured:
        archive = Path(configured).expanduser()
        if not archive.is_file():
            raise RuntimeError(f"Apptainer bundle does not exist: {archive}")
        digest = _sha256(archive)
        expected = os.environ.get("TASKTRACE_CAPTURE_APPTAINER_BUNDLE_SHA256")
        if expected and digest != expected:
            raise RuntimeError(
                f"Apptainer bundle SHA-256 mismatch for {archive}: expected " f"{expected}, got {digest}"
            )
        return BundleArtifact(path=archive.resolve(), sha256=digest)
    return _build_cached_bundle(managed_cache_root(), architecture, upstream_architecture)


def _extract_bundle(
    artifact: BundleArtifact,
    architecture: str,
    install_root: Path,
) -> None:
    actual = _sha256(artifact.path)
    if actual != artifact.sha256:
        raise RuntimeError(
            f"Apptainer bundle SHA-256 mismatch for {artifact.path}: expected " f"{artifact.sha256}, got {actual}"
        )

    parent = install_root.parent
    parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    staging = Path(tempfile.mkdtemp(prefix=".apptainer-install-", dir=parent))
    try:
        with tarfile.open(artifact.path, "r:gz") as archive:
            roots = {Path(member.name).parts[0] for member in archive.getmembers() if Path(member.name).parts}
            if roots != {_ARCHIVE_ROOT}:
                raise RuntimeError(f"invalid Apptainer bundle layout in {artifact.path}: " f"{sorted(roots)}")
            archive.extractall(staging, filter="data")

        extracted = staging / _ARCHIVE_ROOT
        binary = extracted / "bin" / "apptainer"
        nss_wrapper = _nss_wrapper_path(extracted)
        if not binary.is_file() or not os.access(binary, os.X_OK):
            raise RuntimeError(f"Apptainer executable is missing from {artifact.path}")
        if not nss_wrapper.is_file():
            raise RuntimeError(f"NSS compatibility library is missing from {artifact.path}")
        _write_identity_files(extracted)
        marker = {
            "release": MANAGED_APPTAINER_RELEASE,
            "version": MANAGED_APPTAINER_VERSION,
            "architecture": architecture,
            "installer_sha256": UPSTREAM_INSTALLER_SHA256,
            "archive_sha256": artifact.sha256,
        }
        _marker_path(extracted).write_text(json.dumps(marker, sort_keys=True) + "\n")
        os.replace(extracted, install_root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def ensure_apptainer_installation() -> ApptainerInstallation:
    """Select an override or atomically stage TaskTrace Capture's managed runtime."""
    current = inspect_apptainer_installation()
    if current.available or current.source == "explicit":
        return current

    selected = _architecture()
    if selected is None:
        return current
    architecture, upstream_architecture = selected
    install_root = managed_install_root()
    if install_root.exists():
        return current

    lock_path = install_root.parent / f".{MANAGED_APPTAINER_RELEASE}.install.lock"
    install_root.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        with lock_path.open("w") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            current = inspect_apptainer_installation()
            if current.available:
                return current
            artifact = _source_archive(architecture, upstream_architecture)
            _extract_bundle(artifact, architecture, install_root)
    except (OSError, RuntimeError, tarfile.TarError) as exc:
        return ApptainerInstallation(
            binary=None,
            source="managed",
            detail=str(exc),
            external_binary=current.external_binary,
        )
    return inspect_apptainer_installation()
