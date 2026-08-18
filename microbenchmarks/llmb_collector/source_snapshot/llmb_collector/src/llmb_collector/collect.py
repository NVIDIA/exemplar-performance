# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Collect cluster and hardware info; output JSON, YAML, text, or directory layout."""

import importlib
import json
import os
import re
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yaml

from llmb_collector.command_loader import (
    REALM_CONTAINER,
    REALM_NETWORK,
    REALM_SYSTEM,
    load_command_registry,
)

_COMMAND_REGISTRY = load_command_registry()

SYSTEM_DOMAINS = _COMMAND_REGISTRY.domains_for_realm(REALM_SYSTEM)
SYSTEM_COMMANDS = _COMMAND_REGISTRY.commands_for_realm(REALM_SYSTEM)
NETWORK_DOMAINS = _COMMAND_REGISTRY.domains_for_realm(REALM_NETWORK)
NETWORK_COMMANDS = _COMMAND_REGISTRY.commands_for_realm(REALM_NETWORK)

# Backward compatibility aliases.
HOST_DOMAINS = SYSTEM_DOMAINS
HOST_COMMANDS = SYSTEM_COMMANDS

CONTAINER_DOMAINS = _COMMAND_REGISTRY.domains_for_realm(REALM_CONTAINER)
CONTAINER_COMMANDS = _COMMAND_REGISTRY.commands_for_realm(REALM_CONTAINER)

DEFAULT_ENV_PATTERN = (
    r"^(SLURM_|CLOUDPERF_|RUN_CONF_|CUDA_|TORCH_|NCCL_|TRANSFORMERS_|"
    r"TF_|XLA_|JAX_|ENABLE_|NVTE_|HOOPER_|PYTHON|CI_|CI$|GITHUB_|"
    r"GITLAB_|CIRCLECI|CIRCLE_|BUILDKITE_|JENKINS_|TRAVIS_|APPVEYOR_|"
    r"AZURE_|TF_BUILD|SYSTEM_|AGENT_|TEAMCITY_|BITBUCKET_|BAMBOO_|"
    r"DRONE_|WOODPECKER_|CODEBUILD_|CODEPIPELINE_|GITEA_|FORGEJO_|"
    r"HARNESS_|SEMAPHORE_|VERCEL_|NETLIFY_).*"
)
DEFAULT_ENV_REDACT_PATTERN = (
    r"(TOKEN|SECRET|PASSWORD|PASSWD|PASS|KEY|CREDENTIAL|CREDENTIALS|AUTH|"
    r"BEARER|COOKIE|SESSION|PRIVATE|CERT|SIGNATURE)"
)
REDACTED_VALUE = "<redacted>"
MIN_SECRET_VALUE_LENGTH = 16

NO_OUTPUT = "(no output)"


class _BlockStyleSafeDumper(yaml.SafeDumper):
    """Safe dumper that prefers literal block scalars for multiline strings."""


def _represent_block_str(dumper: yaml.SafeDumper, value: str) -> yaml.ScalarNode:
    if "\n" in value:
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|")
    return dumper.represent_scalar("tag:yaml.org,2002:str", value)


_BlockStyleSafeDumper.add_representer(str, _represent_block_str)


def _expand_names(
    names: list[str],
    domains: dict[str, list[str]],
    all_command_names: list[str],
    realm_aliases: Optional[set[str]] = None,
) -> list[str]:
    """Expand domain names to command names; pass through unknown names as-is."""
    out: list[str] = []
    aliases = realm_aliases or set()
    for name in names:
        name = name.strip()
        if not name:
            continue
        if name in aliases:
            out.extend(all_command_names)
        elif name in domains:
            out.extend(domains[name])
        elif name in all_command_names:
            out.append(name)
    return list(dict.fromkeys(out))  # preserve order, dedupe


def _resolve_host_commands(collect_list: Optional[list[str]], exclude_list: Optional[list[str]]) -> list[str]:
    all_names = list(SYSTEM_COMMANDS)
    if collect_list:
        return _expand_names(collect_list, SYSTEM_DOMAINS, all_names, {"host", "system"})
    base = all_names
    if exclude_list:
        exclude_set = set(_expand_names(exclude_list, SYSTEM_DOMAINS, all_names, {"host", "system"}))
        return [n for n in base if n not in exclude_set]
    return base


def _resolve_network_commands(collect_list: Optional[list[str]], exclude_list: Optional[list[str]]) -> list[str]:
    all_names = list(NETWORK_COMMANDS)
    if collect_list:
        return _expand_names(collect_list, NETWORK_DOMAINS, all_names, {"network"})
    base = all_names
    if exclude_list:
        exclude_set = set(_expand_names(exclude_list, NETWORK_DOMAINS, all_names, {"network"}))
        return [n for n in base if n not in exclude_set]
    return base


def _resolve_container_commands(collect_list: Optional[list[str]], exclude_list: Optional[list[str]]) -> list[str]:
    all_names = list(CONTAINER_COMMANDS)
    if collect_list:
        return _expand_names(collect_list, CONTAINER_DOMAINS, all_names, {"container"})
    base = all_names
    if exclude_list:
        exclude_set = set(_expand_names(exclude_list, CONTAINER_DOMAINS, all_names, {"container"}))
        return [n for n in base if n not in exclude_set]
    return base


@dataclass
class CollectConfig:
    result_dir: str = "."
    output_path: Optional[str] = None
    format: Literal["json", "yaml", "text"] = "json"
    compact: bool = True
    host: bool = True
    network: bool = True
    container: bool = False
    env: bool = True
    env_pattern: str = field(default_factory=lambda: DEFAULT_ENV_PATTERN)
    env_redact_pattern: str = field(default_factory=lambda: DEFAULT_ENV_REDACT_PATTERN)
    redact_env: bool = True
    host_collect: Optional[list[str]] = None
    host_exclude: Optional[list[str]] = None
    network_collect: Optional[list[str]] = None
    network_exclude: Optional[list[str]] = None
    container_collect: Optional[list[str]] = None
    container_exclude: Optional[list[str]] = None


def _run_cmd(cmd: list[str]) -> dict:
    result = {
        "cmdline": " ".join(cmd),
        "returncode": -1,
        "stdout": "",
        "stderr": "",
        "exception": None,
    }
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        result["returncode"] = r.returncode
        result["stdout"] = r.stdout or ""
        result["stderr"] = r.stderr or ""
    except Exception as e:
        result["exception"] = str(e)
    return result


def _collect_system_info(data: dict, collect_list: list[str], exclude_list: list[str]) -> None:
    names = _resolve_host_commands(collect_list or None, exclude_list or None)
    data["system"] = {}
    for name in names:
        argv = SYSTEM_COMMANDS.get(name)
        if argv:
            data["system"][name] = _run_cmd(argv)


def _collect_network_info(data: dict, collect_list: list[str], exclude_list: list[str]) -> None:
    names = _resolve_network_commands(collect_list or None, exclude_list or None)
    data["network"] = {}
    for name in names:
        argv = NETWORK_COMMANDS.get(name)
        if argv:
            data["network"][name] = _run_cmd(argv)


def _collect_container_info(data: dict, collect_list: list[str], exclude_list: list[str]) -> None:
    names = _resolve_container_commands(collect_list or None, exclude_list or None)
    data["container"] = {}
    for name in names:
        argv = CONTAINER_COMMANDS.get(name)
        if argv:
            data["container"][name] = _run_cmd(argv)


def _detect_secrets_flags_value(value: str) -> bool:
    """Return whether optional detect-secrets flags an env value."""
    normalized = value.strip()
    if len(normalized) < MIN_SECRET_VALUE_LENGTH:
        return False
    if normalized.lower() in {"true", "false", "yes", "no", "none", "null"}:
        return False
    if normalized.startswith(("http://", "https://")) and not re.search(
        r"(token|secret|password|key|credential|auth)",
        normalized,
        re.IGNORECASE,
    ):
        return False
    try:
        scan_module = importlib.import_module("detect_secrets.core.scan")
        settings_module = importlib.import_module("detect_secrets.settings")
        scan_line = scan_module.scan_line
        default_settings = settings_module.default_settings
    except ImportError:
        return False

    try:
        with default_settings():
            return any(scan_line(f"value = {normalized}"))
    except Exception:
        return False


def _redact_env_value(
    key: str,
    value: str,
    redact_regex: re.Pattern[str] | None,
) -> str:
    if redact_regex and redact_regex.search(key):
        return REDACTED_VALUE
    if redact_regex and _detect_secrets_flags_value(value):
        return REDACTED_VALUE
    return value


def _collect_env(
    data: dict,
    pattern: str,
    redact_pattern: str,
    redact: bool,
) -> None:
    regex = re.compile(pattern)
    redact_regex = re.compile(redact_pattern, re.IGNORECASE) if redact else None
    data["env"] = {k: _redact_env_value(k, v, redact_regex) for k, v in sorted(os.environ.items()) if regex.search(k)}


def collect(config: CollectConfig) -> dict:
    data: dict = {
        "version": "1",
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "system": None,
        "network": None,
        "container": None,
        "env": None,
    }
    if config.host:
        _collect_system_info(
            data,
            config.host_collect or [],
            config.host_exclude or [],
        )
    else:
        data["system"] = None

    if config.network:
        _collect_network_info(
            data,
            config.network_collect or [],
            config.network_exclude or [],
        )
    else:
        data["network"] = None
    if config.container:
        _collect_container_info(
            data,
            config.container_collect or [],
            config.container_exclude or [],
        )
    else:
        data["container"] = None
    if config.env:
        _collect_env(
            data,
            config.env_pattern,
            config.env_redact_pattern,
            config.redact_env,
        )
    else:
        data["env"] = None
    return data


def _write_to_dir(data: dict, root: Path) -> None:
    out = root / "_cloudperf"
    out.mkdir(parents=True, exist_ok=True)
    # Env
    if data.get("env") is not None:
        (out / "job_envs.json").write_text(json.dumps(data["env"], indent=2))
    # System
    if data.get("system"):
        for name, run_result in data["system"].items():
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "cmdline").write_text(run_result.get("cmdline", ""))
            (d / "returncode").write_text(str(run_result.get("returncode", -1)))
            (d / "stdout").write_text(run_result.get("stdout", ""))
            (d / "stderr").write_text(run_result.get("stderr", ""))
            (d / "exception").write_text(str(run_result.get("exception") or ""))
    # Network
    if data.get("network"):
        for name, run_result in data["network"].items():
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "cmdline").write_text(run_result.get("cmdline", ""))
            (d / "returncode").write_text(str(run_result.get("returncode", -1)))
            (d / "stdout").write_text(run_result.get("stdout", ""))
            (d / "stderr").write_text(run_result.get("stderr", ""))
            (d / "exception").write_text(str(run_result.get("exception") or ""))
    # Container
    if data.get("container"):
        for name, run_result in data["container"].items():
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            (d / "cmdline").write_text(run_result.get("cmdline", ""))
            (d / "returncode").write_text(str(run_result.get("returncode", -1)))
            (d / "stdout").write_text(run_result.get("stdout", ""))
            (d / "stderr").write_text(run_result.get("stderr", ""))
            (d / "exception").write_text(str(run_result.get("exception") or ""))


def _append_command_section(lines: list[str], title: str, section: dict, include_stderr: bool) -> None:
    if not section:
        return
    lines.append(title)
    for name, run_result in section.items():
        lines.append(f"--- {name} ---")
        lines.append(run_result.get("stdout", "") or NO_OUTPUT)
        if include_stderr and run_result.get("stderr"):
            lines.append(f"stderr: {run_result['stderr']}")
        lines.append("")


def _append_env_section(lines: list[str], env: Optional[dict]) -> None:
    if env is None:
        return
    lines.append("=== Env ===")
    for k, v in env.items():
        lines.append(f"{k}={v}")


def _format_text_report(data: dict) -> str:
    lines: list[str] = [f"Collected at: {data.get('collected_at', '')}", ""]
    _append_command_section(lines, "=== System (host) ===", data.get("system"), include_stderr=True)
    _append_command_section(lines, "=== Container ===", data.get("container"), include_stderr=False)
    _append_command_section(lines, "=== Network ===", data.get("network"), include_stderr=True)
    _append_env_section(lines, data.get("env"))
    return "\n".join(lines)


def _serialize(data: dict, fmt: str, compact: bool) -> str:
    serializable_data = _drop_none_exceptions(_drop_none_collectors(data))
    if fmt == "text":
        return _format_text_report(serializable_data)
    if fmt == "json":
        return json.dumps(serializable_data, indent=None if compact else 2)
    if fmt == "yaml":
        yaml_data = _prepare_for_yaml(serializable_data)
        # Emit block-style YAML mappings and multiline strings as literal blocks.
        return yaml.dump(
            yaml_data,
            Dumper=_BlockStyleSafeDumper,
            default_flow_style=False,
            allow_unicode=True,
            sort_keys=False,
        )
    raise ValueError(f"Unknown format: {fmt}")


def _looks_like_json_payload(text: str) -> bool:
    stripped = text.strip()
    return stripped.startswith("{") or stripped.startswith("[")


def _decode_stdout_for_yaml(key: str, value):
    if key not in ("stdout", "stderr") or not isinstance(value, str):
        return _prepare_for_yaml(value)

    normalized = _normalize_multiline_output(value)

    if key == "stderr":
        return normalized

    if not _looks_like_json_payload(normalized):
        return normalized
    try:
        return json.loads(normalized)
    except json.JSONDecodeError:
        return normalized


def _normalize_multiline_output(value: str) -> str:
    # Normalize line endings and replace control characters that force quoted YAML style.
    value = value.replace("\r\n", "\n").replace("\r", "\n")
    value = "".join(ch if (ch in ("\n", "\t") or ord(ch) >= 32) else " " for ch in value)

    if "\n" not in value:
        return value
    # Remove trailing spaces and tabs, and expand tabs so YAML can emit literal block style.
    lines = [line.expandtabs(8).rstrip() for line in value.split("\n")]
    normalized = "\n".join(lines)
    if value.endswith("\n"):
        normalized += "\n"
    return normalized


def _drop_none_collectors(data: dict) -> dict:
    collector_keys = {"system", "network", "container", "env"}
    return {k: v for k, v in data.items() if not (k in collector_keys and v is None)}


def _drop_none_exceptions(value):
    if isinstance(value, dict):
        return {k: _drop_none_exceptions(v) for k, v in value.items() if not (k == "exception" and v is None)}
    if isinstance(value, list):
        return [_drop_none_exceptions(v) for v in value]
    return value


def _prepare_for_yaml(value):
    """Convert JSON-looking command stdout into native objects for YAML output."""
    if isinstance(value, dict):
        return {k: _decode_stdout_for_yaml(k, v) for k, v in value.items()}
    if isinstance(value, list):
        return [_prepare_for_yaml(v) for v in value]
    return value
