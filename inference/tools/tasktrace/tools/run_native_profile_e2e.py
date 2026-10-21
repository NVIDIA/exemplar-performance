#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Run the focused profiler E2E suite on one native Linux ISA."""

from __future__ import annotations

import argparse
import os
import platform
import re
import subprocess
import sys
from pathlib import Path
from typing import Sequence

ROOT = Path(__file__).resolve().parents[1]
CGROUP_CONTROLLERS = Path("/sys/fs/cgroup/cgroup.controllers")
DOCKER_ARCHITECTURE = {"x86_64": "amd64", "aarch64": "arm64"}
IMMUTABLE_IMAGE = re.compile(r"(?:[^@\s]+@)?sha256:[0-9a-f]{64}", re.ASCII)
E2E_TESTS = (
    "offline_replayer/tests/test_memory_profile_docker.py",
    (
        "offline_replayer/tests/test_m1_labeled_phase_docker.py::"
        "test_real_coordinated_cpu_compare_writes_aligned_bundle_and_views_once"
    ),
    (
        "offline_replayer/tests/test_m1_labeled_phase_docker.py::"
        "test_real_pipeline_unifies_all_profile_families_and_phases_once"
    ),
    "offline_replayer/tests/test_coordinated_baseline_docker.py",
)


class PreflightError(RuntimeError):
    """The host cannot make the requested native-E2E claim."""


def _run_checked(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise PreflightError(f"could not execute {command[0]!r}: {exc}") from exc
    if completed.returncode:
        detail = (completed.stderr or completed.stdout).strip()
        suffix = f": {detail}" if detail else ""
        raise PreflightError(f"command failed ({completed.returncode}): {' '.join(command)}{suffix}")
    return completed


def validate_image_reference(image: str) -> None:
    if IMMUTABLE_IMAGE.fullmatch(image) is None:
        raise PreflightError(
            "--image must be immutable: use repo@sha256:<64 lowercase hex> "
            "or a local sha256:<64 lowercase hex> image ID, not a tag"
        )


def _inspect_image(docker: str, image: str, expected_machine: str) -> str:
    completed = _run_checked(
        [
            docker,
            "image",
            "inspect",
            "--format",
            "{{.Os}}\t{{.Architecture}}\t{{.Id}}",
            image,
        ]
    )
    fields = completed.stdout.strip().split("\t")
    if len(fields) != 3:
        raise PreflightError("docker image inspect returned an unexpected OS/architecture/ID record")
    image_os, image_architecture, image_id = fields
    expected_architecture = DOCKER_ARCHITECTURE[expected_machine]
    if image_os != "linux" or image_architecture != expected_architecture:
        raise PreflightError(
            f"image is {image_os}/{image_architecture}, expected "
            f"linux/{expected_architecture} for native {expected_machine} E2E"
        )
    if IMMUTABLE_IMAGE.fullmatch(image_id) is None:
        raise PreflightError(f"docker returned a malformed immutable image ID: {image_id!r}")
    return image_id


def _probe_image(docker: str, image: str, cpuset: str, expected_machine: str) -> None:
    probe = """
import os, platform, shutil, sys
expected = sys.argv[1]
failures = []
actual = platform.machine()
if actual != expected:
    failures.append(f"container machine is {actual!r}, expected {expected!r}")
cpu_count = len(os.sched_getaffinity(0))
if cpu_count < 4:
    failures.append(f"cpuset exposes {cpu_count} CPUs; at least 4 are required")
for binary in ("bash", "strace"):
    if shutil.which(binary) is None:
        failures.append(f"fixture image is missing {binary}")
if failures:
    raise SystemExit("; ".join(failures))
""".strip()
    _run_checked(
        [
            docker,
            "run",
            "--rm",
            "--network",
            "none",
            "--cpuset-cpus",
            cpuset,
            image,
            "python",
            "-c",
            probe,
            expected_machine,
        ]
    )


def preflight(
    *,
    expected_machine: str,
    image: str,
    cpuset: str,
    docker: str = "docker",
    cgroup_controllers: Path = CGROUP_CONTROLLERS,
) -> str:
    validate_image_reference(image)
    actual_machine = platform.machine()
    if actual_machine != expected_machine:
        raise PreflightError(
            f"host machine is {actual_machine!r}, expected {expected_machine!r}; "
            "native CI must not use cross-ISA emulation"
        )
    if not cgroup_controllers.is_file():
        raise PreflightError(f"cgroup v2 is required but {cgroup_controllers} is unavailable")
    _run_checked([docker, "version", "--format", "{{.Server.Version}}"])
    image_id = _inspect_image(docker, image, expected_machine)
    _probe_image(docker, image, cpuset, expected_machine)
    return image_id


def run_tests(*, image: str, cpuset: str) -> int:
    environment = os.environ.copy()
    environment.update(
        {
            "TASKTRACE_DOCKER_TEST_IMAGE": image,
            "TASKTRACE_M1_DOCKER_TEST_IMAGE": image,
            "TASKTRACE_M1_CPUSET": cpuset,
        }
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", *E2E_TESTS],
            cwd=ROOT,
            env=environment,
            check=False,
        )
    except OSError as exc:
        print(f"unable to start pytest: {exc}", file=sys.stderr)
        return 2
    return completed.returncode


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight a native Docker/cgroup-v2 runner, then run the focused "
            "CPU, memory, evidence-view, and baseline profiler E2Es."
        )
    )
    parser.add_argument("--expected-machine", required=True, choices=tuple(DOCKER_ARCHITECTURE))
    parser.add_argument("--image", required=True, help="immutable fixture image")
    parser.add_argument("--cpuset", required=True, help="four or more runner-owned logical CPUs")
    parser.add_argument("--docker", default="docker", help=argparse.SUPPRESS)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        image_id = preflight(
            expected_machine=args.expected_machine,
            image=args.image,
            cpuset=args.cpuset,
            docker=args.docker,
        )
    except PreflightError as exc:
        print(f"native profile E2E preflight failed: {exc}", file=sys.stderr)
        return 2
    print(
        f"native profile E2E: host={args.expected_machine} " f"image_id={image_id} cpuset={args.cpuset}",
        flush=True,
    )
    return run_tests(image=args.image, cpuset=args.cpuset)


if __name__ == "__main__":
    raise SystemExit(main())
