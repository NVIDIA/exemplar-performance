# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

import pytest


def test_native_wheel_contains_no_python_runtime() -> None:
    configured = os.environ.get("TASKTRACE_NATIVE_WHEEL")
    if not configured:
        pytest.skip("set TASKTRACE_NATIVE_WHEEL to validate a built native wheel")

    wheel = Path(configured)
    assert wheel.is_file(), wheel

    with ZipFile(wheel) as archive:
        members = archive.namelist()
        scripts = [name for name in members if name.endswith(".data/scripts/tasktrace")]
        metadata_members = [name for name in members if name.endswith(".dist-info/METADATA")]

        assert len(scripts) == 1, members
        assert len(metadata_members) == 1, members
        assert not any(name.endswith((".py", ".pyc")) for name in members), members
        assert not any("offline_replayer" in name for name in members), members

        executable = archive.read(scripts[0])
        assert executable.startswith(b"\x7fELF"), "Linux wheel script is not a native ELF binary"
        for forbidden in (
            b"importlib.import_module",
            b"offline_replayer.adapters",
            b"offline_replayer.measurement",
            b"TASKTRACE_RUST_DELEGATED",
            b"libpython",
        ):
            assert forbidden not in executable, forbidden

        metadata = Parser().parsestr(archive.read(metadata_members[0]).decode("utf-8"))
        assert metadata["Name"] == "tasktrace"
        requirements = metadata.get_all("Requires-Dist", [])
        assert len(requirements) == 2, requirements
        assert {requirement.split(";", 1)[0].strip().lower() for requirement in requirements} == {
            "pytest>=7",
            "pyyaml>=6",
        }
        for requirement in requirements:
            assert "extra == 'test'" in requirement, requirement
