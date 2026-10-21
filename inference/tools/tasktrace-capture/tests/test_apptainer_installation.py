# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

from tasktrace_capture.apptainer import installation
from tasktrace_capture.apptainer.installation import (
    MANAGED_APPTAINER_RELEASE,
    BundleArtifact,
    ensure_apptainer_installation,
    inspect_apptainer_installation,
    managed_compatibility_environment,
)


def _fake_install_tree(parent: Path) -> Path:
    source = parent / "apptainer"
    binary = source / "bin" / "apptainer"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho 1.5.2\n")
    binary.chmod(0o755)
    nss_wrapper = source / "tasktrace-capture" / "lib" / "libnss_wrapper.so"
    nss_wrapper.parent.mkdir(parents=True)
    nss_wrapper.write_text("test library\n")
    nss_wrapper.chmod(0o755)
    return source


def _fake_bundle(tmp_path: Path) -> BundleArtifact:
    source = _fake_install_tree(tmp_path / "source")
    archive = tmp_path / "managed.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        tar.add(source, arcname="apptainer")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    return BundleArtifact(path=archive, sha256=digest)


def _configure_managed_install(
    tmp_path: Path,
    monkeypatch,
    artifact: BundleArtifact,
) -> Path:
    install_root = tmp_path / "runtime" / "apptainer"
    monkeypatch.delenv("TASKTRACE_CAPTURE_APPTAINER_BIN", raising=False)
    monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_MANAGED_ROOT", str(install_root))
    monkeypatch.setattr(installation.platform, "machine", lambda: "aarch64")
    monkeypatch.setattr(
        installation,
        "_source_archive",
        lambda architecture, upstream_architecture: artifact,
    )
    monkeypatch.setattr(installation, "_external_apptainer", lambda: None)
    return install_root


def test_explicit_binary_wins_without_managed_install(
    tmp_path: Path,
    monkeypatch,
):
    explicit = tmp_path / "conda" / "bin" / "apptainer"
    explicit.parent.mkdir(parents=True)
    explicit.write_text("user installation\n")
    explicit.chmod(0o755)
    monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_BIN", str(explicit))
    monkeypatch.setenv(
        "TASKTRACE_CAPTURE_APPTAINER_MANAGED_ROOT",
        str(tmp_path / "must-not-be-created"),
    )

    selected = ensure_apptainer_installation()

    assert selected.source == "explicit"
    assert selected.binary == explicit.resolve()
    assert explicit.read_text() == "user installation\n"
    assert not (tmp_path / "must-not-be-created").exists()


def test_invalid_explicit_binary_does_not_silently_fall_back(
    tmp_path: Path,
    monkeypatch,
):
    missing = tmp_path / "missing-apptainer"
    monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_BIN", str(missing))

    selected = ensure_apptainer_installation()

    assert selected.source == "explicit"
    assert selected.binary is None
    assert str(missing) in selected.detail


def test_managed_bundle_is_verified_and_staged_atomically(
    tmp_path: Path,
    monkeypatch,
):
    artifact = _fake_bundle(tmp_path)
    install_root = _configure_managed_install(tmp_path, monkeypatch, artifact)

    selected = ensure_apptainer_installation()

    assert selected.source == "managed"
    assert selected.binary == (install_root / "bin" / "apptainer").resolve()
    assert selected.nss_wrapper == (install_root / "tasktrace-capture" / "lib" / "libnss_wrapper.so").resolve()
    marker = installation._marker_path(install_root)
    assert MANAGED_APPTAINER_RELEASE in marker.read_text()
    assert not list(install_root.parent.glob(".apptainer-install-*"))


def test_external_binary_is_ignored_and_left_untouched(
    tmp_path: Path,
    monkeypatch,
):
    artifact = _fake_bundle(tmp_path)
    external = tmp_path / "conda" / "bin" / "apptainer"
    external.parent.mkdir(parents=True)
    external.write_text("external\n")
    external.chmod(0o755)
    install_root = _configure_managed_install(tmp_path, monkeypatch, artifact)
    monkeypatch.setattr(installation, "_external_apptainer", lambda: external.resolve())

    selected = ensure_apptainer_installation()

    assert selected.managed
    assert selected.binary == (install_root / "bin" / "apptainer").resolve()
    assert selected.binary != external.resolve()
    assert selected.external_binary == external.resolve()
    assert external.read_text() == "external\n"


def test_bundle_hash_mismatch_never_publishes_install(
    tmp_path: Path,
    monkeypatch,
):
    artifact = _fake_bundle(tmp_path)
    bad_artifact = BundleArtifact(path=artifact.path, sha256="0" * 64)
    install_root = _configure_managed_install(tmp_path, monkeypatch, bad_artifact)

    selected = ensure_apptainer_installation()

    assert selected.binary is None
    assert "SHA-256 mismatch" in selected.detail
    assert not install_root.exists()


def test_unsafe_archive_member_is_rejected(tmp_path: Path, monkeypatch):
    archive = tmp_path / "unsafe.tar.gz"
    payload = b"escape"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("apptainer/../../escape")
        member.size = len(payload)
        tar.addfile(member, io.BytesIO(payload))
    artifact = BundleArtifact(
        path=archive,
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
    )
    install_root = _configure_managed_install(tmp_path, monkeypatch, artifact)

    selected = ensure_apptainer_installation()

    assert selected.binary is None
    assert not (tmp_path / "escape").exists()
    assert not install_root.exists()


def test_inspection_never_installs(tmp_path: Path, monkeypatch):
    artifact = _fake_bundle(tmp_path)
    install_root = _configure_managed_install(tmp_path, monkeypatch, artifact)

    selected = inspect_apptainer_installation()

    assert selected.binary is None
    assert not install_root.exists()


def test_managed_nss_compatibility_is_scoped_to_apptainer(
    tmp_path: Path,
    monkeypatch,
):
    artifact = _fake_bundle(tmp_path)
    install_root = _configure_managed_install(tmp_path, monkeypatch, artifact)
    selected = ensure_apptainer_installation()
    monkeypatch.setattr(installation, "_host_identity_available", lambda: False)

    original = {"PATH": "/usr/bin", "LD_PRELOAD": "/user/library.so"}
    environment = managed_compatibility_environment(original, selected)

    assert original["LD_PRELOAD"] == "/user/library.so"
    assert environment["LD_PRELOAD"] == str(selected.nss_wrapper)
    assert environment["NSS_WRAPPER_PASSWD"] == str(install_root / "tasktrace-capture" / "etc" / "passwd")
    assert environment["NSS_WRAPPER_GROUP"] == str(install_root / "tasktrace-capture" / "etc" / "group")


def test_explicit_install_never_receives_managed_nss_environment(
    tmp_path: Path,
    monkeypatch,
):
    explicit = tmp_path / "apptainer"
    explicit.write_text("#!/bin/sh\n")
    explicit.chmod(0o755)
    monkeypatch.setenv("TASKTRACE_CAPTURE_APPTAINER_BIN", str(explicit))
    monkeypatch.setattr(installation, "_host_identity_available", lambda: False)
    selected = inspect_apptainer_installation()

    environment = managed_compatibility_environment({"LD_PRELOAD": "/user/library.so"}, selected)

    assert environment == {"LD_PRELOAD": "/user/library.so"}


def test_cached_bundle_is_reused_without_rebuilding(
    tmp_path: Path,
    monkeypatch,
):
    installer = tmp_path / "installer.sh"
    installer.write_text("installer\n")
    calls = []

    monkeypatch.setattr(installation, "_verified_installer", lambda cache_root: installer)
    monkeypatch.setattr(installation, "_find_host_nss_wrapper", lambda: None)
    monkeypatch.setattr(
        installation.shutil,
        "which",
        lambda command: "/usr/bin/enroot" if command == "enroot" else None,
    )

    def fake_install(
        installer_path,
        destination,
        upstream_architecture,
        cache_root,
        workspace,
    ):
        calls.append(destination)
        source = _fake_install_tree(workspace / "built")
        source.rename(destination)

    monkeypatch.setattr(installation, "_install_with_enroot", fake_install)
    monkeypatch.setattr(installation, "_validate_install", lambda *args: None)

    first = installation._build_cached_bundle(tmp_path / "cache", "arm64", "aarch64")
    second = installation._build_cached_bundle(tmp_path / "cache", "arm64", "aarch64")

    assert first == second
    assert len(calls) == 1


def test_unsupported_architecture_requires_explicit_install(monkeypatch):
    monkeypatch.delenv("TASKTRACE_CAPTURE_APPTAINER_BIN", raising=False)
    monkeypatch.setattr(installation.platform, "machine", lambda: "riscv64")
    monkeypatch.setattr(installation, "_external_apptainer", lambda: None)

    selected = ensure_apptainer_installation()

    assert selected.binary is None
    assert "arm64 and amd64" in selected.detail
