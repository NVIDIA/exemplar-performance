# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "tools" / "build_tb2_image.py"
SPEC = importlib.util.spec_from_file_location("build_tb2_image", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
build_tb2_image = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(build_tb2_image)


def test_read_dockerfile_instructions_joins_continuations(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "# comment\n"
        "FROM --platform=linux/amd64 python:3.12-slim AS base\n"
        "RUN printf one \\\n"
        "    && printf two\n"
        "WORKDIR /app\n"
    )

    instructions = build_tb2_image._read_dockerfile_instructions(dockerfile)

    assert instructions == [
        ("FROM", "--platform=linux/amd64 python:3.12-slim AS base"),
        ("RUN", "printf one && printf two"),
        ("WORKDIR", "/app"),
    ]
    assert build_tb2_image._from_base_ref(instructions) == "python:3.12-slim"


def test_enroot_uri_handles_default_and_explicit_registries() -> None:
    assert build_tb2_image._enroot_uri("python:3.12-slim") == "docker://python:3.12-slim"
    assert build_tb2_image._enroot_uri("ghcr.io/org/image:tag") == "docker://ghcr.io#org/image:tag"


def test_build_script_inherits_workdir_and_translates_copy_env(tmp_path: Path) -> None:
    instructions = [
        ("FROM", "python:3.12-slim"),
        ("ARG", "BUILD_MODE=release"),
        ("ENV", "APP_MODE production"),
        ("COPY", "src/ ."),
        ("RUN", "printf ready > marker.txt"),
        ("WORKDIR", "nested"),
        ("COPY", "config.json settings.json"),
    ]

    script = build_tb2_image.build_script(instructions, base_workdir="/base")

    assert "mkdir -p /base && cd /base" in script
    assert "export BUILD_MODE=release" in script
    assert "export APP_MODE=production" in script
    assert "cp -a /tb2-build-context/src/. /base/./" in script
    assert "cd /base\nprintf ready > marker.txt" in script
    assert "mkdir -p /base/nested && cd /base/nested" in script
    assert "cp -a /tb2-build-context/config.json /base/nested/settings.json" in script


def test_build_script_persists_modern_multi_assignment_env_individually() -> None:
    script = build_tb2_image.build_script(
        [
            ("FROM", "python:3.13-slim"),
            (
                "ENV",
                "OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1",
            ),
        ]
    )

    assert "export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1" in script
    for assignment in [
        "OMP_NUM_THREADS=1",
        "MKL_NUM_THREADS=1",
        "OPENBLAS_NUM_THREADS=1",
    ]:
        assert f"printf '%s\\n' {assignment} >> /etc/environment" in script
    assert (
        "printf '%s\\n' 'OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 "
        "OPENBLAS_NUM_THREADS=1' >> /etc/environment" not in script
    )


def test_builder_rejects_multistage_before_invoking_enroot(tmp_path: Path) -> None:
    task = tmp_path / "task"
    task.mkdir()
    (task / "Dockerfile").write_text(
        "FROM python:3.12-slim AS build\n" "RUN true\n" "FROM python:3.12-slim\n" "COPY --from=build /tmp/x /tmp/x\n"
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--task-dir",
            str(task),
            "--output",
            str(tmp_path / "out.sqsh"),
            "--enroot",
            str(tmp_path / "must-not-run"),
        ],
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 2
    assert "multi-stage Dockerfile" in completed.stderr
