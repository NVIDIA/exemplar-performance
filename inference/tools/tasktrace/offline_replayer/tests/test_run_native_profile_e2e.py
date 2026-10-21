# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "run_native_profile_e2e.py"
SPEC = importlib.util.spec_from_file_location("run_native_profile_e2e", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
native_e2e = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(native_e2e)

DIGEST = "sha256:" + "a" * 64
IMAGE = "registry.example/tasktrace/profile-fixture@" + DIGEST


@pytest.mark.parametrize(
    "image",
    [
        "python:3.12-slim",
        "registry.example/fixture@sha256:abc",
        "registry.example/fixture@sha256:" + "A" * 64,
    ],
)
def test_immutable_image_reference_rejects_tags_and_malformed_digests(
    image: str,
) -> None:
    with pytest.raises(native_e2e.PreflightError, match="must be immutable"):
        native_e2e.validate_image_reference(image)


@pytest.mark.parametrize("image", [DIGEST, IMAGE])
def test_immutable_image_reference_accepts_digest_or_digest_reference(
    image: str,
) -> None:
    native_e2e.validate_image_reference(image)


def test_preflight_rejects_non_native_host_before_docker(monkeypatch) -> None:
    monkeypatch.setattr(native_e2e.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(
        native_e2e,
        "_run_checked",
        lambda _command: pytest.fail("Docker must not run after an ISA mismatch"),
    )

    with pytest.raises(native_e2e.PreflightError, match="must not use.*emulation"):
        native_e2e.preflight(
            expected_machine="aarch64",
            image=IMAGE,
            cpuset="20-23",
        )


def test_preflight_checks_docker_image_and_native_probe(tmp_path: Path, monkeypatch) -> None:
    controllers = tmp_path / "cgroup.controllers"
    controllers.write_text("cpu memory\n")
    commands: list[list[str]] = []

    def fake_run(command):
        command = list(command)
        commands.append(command)
        if command[1:3] == ["image", "inspect"]:
            return subprocess.CompletedProcess(command, 0, f"linux\tamd64\t{DIGEST}\n", "")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(native_e2e.platform, "machine", lambda: "x86_64")
    monkeypatch.setattr(native_e2e, "_run_checked", fake_run)

    assert (
        native_e2e.preflight(
            expected_machine="x86_64",
            image=IMAGE,
            cpuset="20-23",
            cgroup_controllers=controllers,
        )
        == DIGEST
    )
    assert commands[0][:2] == ["docker", "version"]
    assert commands[1][:3] == ["docker", "image", "inspect"]
    probe = commands[2]
    assert probe[:3] == ["docker", "run", "--rm"]
    assert probe[probe.index("--network") + 1] == "none"
    assert probe[probe.index("--cpuset-cpus") + 1] == "20-23"
    assert probe[-1] == "x86_64"


def test_inspected_image_must_match_native_architecture(monkeypatch) -> None:
    monkeypatch.setattr(
        native_e2e,
        "_run_checked",
        lambda command: subprocess.CompletedProcess(command, 0, f"linux\tarm64\t{DIGEST}\n", ""),
    )

    with pytest.raises(native_e2e.PreflightError, match="expected linux/amd64"):
        native_e2e._inspect_image("docker", IMAGE, "x86_64")
