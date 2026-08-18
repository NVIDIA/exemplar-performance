# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for command registry loading."""

from pathlib import Path
from unittest.mock import patch

import pytest
import yaml
from llmb_collector.command_loader import (
    REALM_CONTAINER,
    REALM_NETWORK,
    REALM_SYSTEM,
    discover_entrypoint_config_roots,
    get_packaged_commands_config,
    load_command_registry,
    load_registry_from_directory,
)
from llmb_collector.provider import get_provider


def _write_domain(root: Path, realm: str, domain: str, commands: dict) -> None:
    realm_dir = root / realm
    realm_dir.mkdir(parents=True, exist_ok=True)
    (realm_dir / f"{domain}.yaml").write_text(
        yaml.safe_dump({"commands": commands}, sort_keys=False),
        encoding="utf-8",
    )


def _write_minimal_registry(root: Path) -> None:
    _write_domain(
        root,
        REALM_SYSTEM,
        "cpu",
        {"lscpu": {"description": "CPU info", "argv": ["lscpu", "--json"]}},
    )
    _write_domain(
        root,
        REALM_NETWORK,
        "interfaces",
        {"ip": {"description": "Links", "argv": ["ip", "link"]}},
    )
    _write_domain(
        root,
        REALM_CONTAINER,
        "runtime",
        {"container_python3_version": {"description": "Python", "argv": ["python3", "--version"]}},
    )


def test_load_registry_from_directory_discovers_realms_and_domains(tmp_path: Path):
    _write_minimal_registry(tmp_path)

    registry = load_registry_from_directory(tmp_path)

    assert registry.commands["lscpu"].realm == REALM_SYSTEM
    assert registry.commands["lscpu"].domain == "cpu"
    assert registry.domains_for_realm(REALM_SYSTEM) == {"cpu": ["lscpu"]}
    assert registry.commands_for_realm(REALM_NETWORK) == {"ip": ["ip", "link"]}


def test_duplicate_command_id_raises(tmp_path: Path):
    _write_domain(
        tmp_path,
        REALM_SYSTEM,
        "cpu",
        {"same": {"description": "A", "argv": ["echo", "a"]}},
    )
    _write_domain(
        tmp_path,
        REALM_NETWORK,
        "interfaces",
        {"same": {"description": "B", "argv": ["echo", "b"]}},
    )
    _write_domain(
        tmp_path,
        REALM_CONTAINER,
        "runtime",
        {"container_python3_version": {"description": "Python", "argv": ["python3", "--version"]}},
    )

    with pytest.raises(ValueError, match="Duplicate command id"):
        load_registry_from_directory(tmp_path)


def test_exact_duplicate_argv_is_reported(tmp_path: Path):
    _write_minimal_registry(tmp_path)
    _write_domain(
        tmp_path,
        REALM_CONTAINER,
        "gpu",
        {"container_lscpu": {"description": "Same argv", "argv": ["lscpu", "--json"]}},
    )

    registry = load_registry_from_directory(tmp_path)

    assert any(finding.kind == "duplicate_argv" for finding in registry.findings)


def test_similar_argv_is_reported_for_reordered_args(tmp_path: Path):
    _write_minimal_registry(tmp_path)
    _write_domain(
        tmp_path,
        REALM_NETWORK,
        "storage_like",
        {
            "lsblk_a": {"description": "A", "argv": ["lsblk", "-O", "--json"]},
            "lsblk_b": {"description": "B", "argv": ["lsblk", "--json", "-O"]},
        },
    )

    registry = load_registry_from_directory(tmp_path)

    assert any(finding.kind == "similar_argv" for finding in registry.findings)


def test_env_override_loads_registry(tmp_path: Path):
    _write_minimal_registry(tmp_path)

    with patch.dict("os.environ", {"LLMB_COMMANDS_DIR": str(tmp_path)}):
        registry = load_command_registry()

    assert registry.commands["lscpu"].argv == ["lscpu", "--json"]


def test_packaged_commands_config_entrypoint_root_loads():
    root = get_packaged_commands_config()

    registry = load_registry_from_directory(root)

    assert "lscpu" in registry.commands
    assert "interfaces" in registry.domains_for_realm(REALM_NETWORK)


def test_discovers_llmb_entrypoint_config_roots(tmp_path: Path):
    _write_minimal_registry(tmp_path)

    class FakeEntryPoint:
        def load(self):
            return lambda: tmp_path

    with patch(
        "llmb_collector.command_loader.metadata.entry_points",
        return_value=[FakeEntryPoint()],
    ):
        roots = discover_entrypoint_config_roots()

    assert roots == [tmp_path]
    registry = load_registry_from_directory(roots[0])
    assert registry.commands["lscpu"].argv == ["lscpu", "--json"]


def test_discovers_config_root_from_runnable_provider():
    provider = get_provider()

    class FakeEntryPoint:
        def load(self):
            return lambda: provider

    with patch(
        "llmb_collector.command_loader.metadata.entry_points",
        return_value=[FakeEntryPoint()],
    ):
        roots = discover_entrypoint_config_roots()

    assert roots == [provider.commands_config()]


def test_runnable_provider_can_collect_from_another_app():
    provider = get_provider()

    with patch("llmb_collector.collect._run_cmd") as mock_run:
        mock_run.return_value = {
            "cmdline": "test",
            "returncode": 0,
            "stdout": "out",
            "stderr": "",
            "exception": None,
        }
        data = provider.collect(
            host=True,
            network=False,
            container=False,
            env=False,
            host_collect=["lscpu"],
        )

    assert list(data["system"]) == ["lscpu"]
    assert data["network"] is None
    assert data["container"] is None
