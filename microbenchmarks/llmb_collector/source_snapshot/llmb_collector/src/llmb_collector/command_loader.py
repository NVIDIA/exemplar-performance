# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load collector command specs from realm/domain YAML files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib import metadata, resources
from importlib.resources.abc import Traversable
from pathlib import Path
from typing import Any, Literal

import yaml

REALM_SYSTEM = "system"
REALM_NETWORK = "network"
REALM_CONTAINER = "container"
REALM_IDS = (REALM_SYSTEM, REALM_NETWORK, REALM_CONTAINER)
ENTRY_POINT_GROUP = "llmb"


@dataclass(frozen=True)
class CommandSpec:
    id: str
    realm: str
    domain: str
    description: str
    argv: list[str]
    source: str


@dataclass(frozen=True)
class CommandFinding:
    severity: Literal["warning", "error"]
    kind: Literal["duplicate_argv", "similar_argv"]
    command_ids: tuple[str, str]
    sources: tuple[str, str]
    argv: tuple[tuple[str, ...], tuple[str, ...]]
    message: str


@dataclass(frozen=True)
class CommandRegistry:
    commands: dict[str, CommandSpec]
    by_realm: dict[str, list[str]]
    by_domain: dict[tuple[str, str], list[str]]
    findings: tuple[CommandFinding, ...]

    def domains_for_realm(self, realm: str) -> dict[str, list[str]]:
        """Return domain -> command ids for a realm."""
        return {domain: names for (domain_realm, domain), names in self.by_domain.items() if domain_realm == realm}

    def commands_for_realm(self, realm: str) -> dict[str, list[str]]:
        """Return command id -> argv for a realm."""
        return {name: self.commands[name].argv for name in self.by_realm.get(realm, [])}


def _as_str(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label}: expected string")
    return value


def _as_argv(value: Any, label: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label}: expected non-empty list")
    out: list[str] = []
    for item in value:
        if item is None or isinstance(item, bool):
            raise ValueError(f"{label}: invalid argv entry {item!r}")
        out.append(str(item))
    return out


def _parse_domain_document(
    raw: Any,
    source: str,
    realm: str,
    domain: str,
) -> list[CommandSpec]:
    if not isinstance(raw, dict):
        raise ValueError(f"{source}: root must be a mapping")
    commands_raw = raw.get("commands")
    if not isinstance(commands_raw, dict):
        raise ValueError(f"{source}: requires 'commands' mapping")

    commands: list[CommandSpec] = []
    for command_id, spec_raw in commands_raw.items():
        command_id = _as_str(command_id, f"{source}: command id")
        if not isinstance(spec_raw, dict):
            raise ValueError(f"{source} commands[{command_id}]: expected mapping")
        description = spec_raw.get("description", "")
        commands.append(
            CommandSpec(
                id=command_id,
                realm=realm,
                domain=domain,
                description=_as_str(description, f"{source} commands[{command_id}].description"),
                argv=_as_argv(spec_raw.get("argv"), f"{source} commands[{command_id}].argv"),
                source=source,
            )
        )
    return commands


def _load_domain_file(path: Path, realm: str, domain: str) -> list[CommandSpec]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse_domain_document(raw, str(path), realm, domain)


def _load_domain_resource(
    resource: Traversable,
    realm: str,
    domain: str,
    source: str,
) -> list[CommandSpec]:
    raw = yaml.safe_load(resource.read_text(encoding="utf-8"))
    return _parse_domain_document(raw, source, realm, domain)


def _normalized_argv(argv: list[str]) -> tuple[str, ...]:
    return tuple(argv)


def _argv_similarity_key(argv: list[str]) -> tuple[str, tuple[str, ...]]:
    if not argv:
        return ("", ())
    return (argv[0], tuple(sorted(argv[1:])))


def _is_similar_argv(left: list[str], right: list[str]) -> bool:
    if not left or not right or left[0] != right[0] or left == right:
        return False
    if _argv_similarity_key(left) == _argv_similarity_key(right):
        return True
    left_args = set(left[1:])
    right_args = set(right[1:])
    if left_args and right_args and (left_args <= right_args or right_args <= left_args):
        return True
    overlap = len(left_args & right_args)
    shortest = min(len(left_args), len(right_args))
    return shortest > 0 and overlap / shortest >= 0.75


def find_command_findings(commands: dict[str, CommandSpec]) -> tuple[CommandFinding, ...]:
    findings: list[CommandFinding] = []
    command_items = list(commands.items())

    seen_argv: dict[tuple[str, ...], CommandSpec] = {}
    for _, spec in command_items:
        key = _normalized_argv(spec.argv)
        previous = seen_argv.get(key)
        if previous is not None:
            findings.append(
                CommandFinding(
                    severity="warning",
                    kind="duplicate_argv",
                    command_ids=(previous.id, spec.id),
                    sources=(previous.source, spec.source),
                    argv=(tuple(previous.argv), tuple(spec.argv)),
                    message=f"{previous.id!r} and {spec.id!r} use identical argv",
                )
            )
        else:
            seen_argv[key] = spec

    for index, (_, left) in enumerate(command_items):
        for _, right in command_items[index + 1 :]:
            if tuple(left.argv) == tuple(right.argv):
                continue
            if _is_similar_argv(left.argv, right.argv):
                findings.append(
                    CommandFinding(
                        severity="warning",
                        kind="similar_argv",
                        command_ids=(left.id, right.id),
                        sources=(left.source, right.source),
                        argv=(tuple(left.argv), tuple(right.argv)),
                        message=f"{left.id!r} and {right.id!r} use similar argv",
                    )
                )
    return tuple(findings)


def _empty_registry_parts() -> tuple[
    dict[str, CommandSpec],
    dict[str, list[str]],
    dict[tuple[str, str], list[str]],
]:
    return {}, {realm: [] for realm in REALM_IDS}, {}


def _add_specs(
    specs: list[CommandSpec],
    commands: dict[str, CommandSpec],
    by_realm: dict[str, list[str]],
    by_domain: dict[tuple[str, str], list[str]],
) -> None:
    for spec in specs:
        if spec.id in commands:
            previous = commands[spec.id]
            raise ValueError(f"Duplicate command id {spec.id!r}: " f"{previous.source} and {spec.source}")
        commands[spec.id] = spec
        by_realm[spec.realm].append(spec.id)
        by_domain.setdefault((spec.realm, spec.domain), []).append(spec.id)


def _build_registry(
    commands: dict[str, CommandSpec],
    by_realm: dict[str, list[str]],
    by_domain: dict[tuple[str, str], list[str]],
) -> CommandRegistry:
    return CommandRegistry(
        commands=commands,
        by_realm=by_realm,
        by_domain=by_domain,
        findings=find_command_findings(commands),
    )


def _load_registry_from_resource_root(root: Traversable, source_prefix: str) -> CommandRegistry:
    commands, by_realm, by_domain = _empty_registry_parts()
    for realm in REALM_IDS:
        realm_dir = root.joinpath(realm)
        if not realm_dir.is_dir():
            raise ValueError(f"{source_prefix}: missing realm directory {realm!r}")
        for resource in sorted(realm_dir.iterdir(), key=lambda item: item.name):
            if not resource.name.endswith(".yaml") or not resource.is_file():
                continue
            domain = resource.name[:-5]
            source = f"{source_prefix}/{realm}/{resource.name}"
            specs = _load_domain_resource(resource, realm, domain, source)
            _add_specs(specs, commands, by_realm, by_domain)
    return _build_registry(commands, by_realm, by_domain)


def load_registry_from_directory(directory: Path | str) -> CommandRegistry:
    root = Path(directory)
    commands, by_realm, by_domain = _empty_registry_parts()

    for realm in REALM_IDS:
        realm_dir = root / realm
        if not realm_dir.is_dir():
            raise ValueError(f"{root}: missing realm directory {realm!r}")
        for path in sorted(realm_dir.glob("*.yaml")):
            domain = path.stem
            _add_specs(_load_domain_file(path, realm, domain), commands, by_realm, by_domain)

    return _build_registry(commands, by_realm, by_domain)


def _try_directory(path: Path) -> CommandRegistry | None:
    if not path.is_dir():
        return None
    try:
        return load_registry_from_directory(path)
    except (OSError, ValueError, yaml.YAMLError):
        return None


def _try_package_registry() -> CommandRegistry | None:
    try:
        root = resources.files("llmb_collector").joinpath("commands_config")
    except (ModuleNotFoundError, FileNotFoundError):
        return None
    try:
        return _load_registry_from_resource_root(root, "commands_config")
    except (OSError, ValueError, yaml.YAMLError):
        return None


def get_packaged_commands_config() -> Traversable:
    """Entry point target returning this package's bundled command config root."""
    return resources.files("llmb_collector").joinpath("commands_config")


def _as_config_root(value: Any) -> Path | Traversable:
    if hasattr(value, "commands_config"):
        return _as_config_root(value.commands_config())
    if isinstance(value, (str, Path)):
        return Path(value)
    if all(hasattr(value, attr) for attr in ("joinpath", "is_dir", "iterdir")):
        return value
    raise TypeError(f"Unsupported command config entry point value: {value!r}")


def discover_entrypoint_config_roots() -> list[Path | Traversable]:
    """Discover command config roots advertised through the ``llmb`` entry point group."""
    roots: list[Path | Traversable] = []
    for entry_point in metadata.entry_points(group=ENTRY_POINT_GROUP):
        loaded = entry_point.load()
        value = loaded() if callable(loaded) else loaded
        roots.append(_as_config_root(value))
    return roots


def _try_config_root(root: Path | Traversable, source_prefix: str) -> CommandRegistry | None:
    try:
        if isinstance(root, Path):
            return load_registry_from_directory(root)
        return _load_registry_from_resource_root(root, source_prefix)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        return None


def resolve_commands_config_dirs() -> list[Path]:
    out: list[Path] = []
    seen: set[Path] = set()
    env = os.environ.get("LLMB_COMMANDS_DIR", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        seen.add(path)
        out.append(path)
    repo_config = (Path(__file__).resolve().parents[2] / "configs" / "commands").resolve()
    if repo_config not in seen:
        out.append(repo_config)
    return out


def load_command_registry() -> CommandRegistry:
    checked_dirs = resolve_commands_config_dirs()
    if os.environ.get("LLMB_COMMANDS_DIR", "").strip():
        return load_registry_from_directory(checked_dirs[0])
    for directory in checked_dirs:
        loaded = _try_directory(directory)
        if loaded is not None:
            return loaded
    for index, root in enumerate(discover_entrypoint_config_roots()):
        loaded = _try_config_root(root, f"{ENTRY_POINT_GROUP} entry point #{index + 1}")
        if loaded is not None:
            return loaded
    packaged = _try_package_registry()
    if packaged is not None:
        return packaged
    dirs_hint = ", ".join(str(path) for path in checked_dirs) if checked_dirs else "(none)"
    raise RuntimeError(
        "Collector command registry must come from realm/domain YAML files. "
        "Provide system/, network/, and container/ directories under LLMB_COMMANDS_DIR, "
        "use a repo checkout with configs/commands/, or install bundled package data. "
        f"Directories checked: {dirs_hint}"
    )
