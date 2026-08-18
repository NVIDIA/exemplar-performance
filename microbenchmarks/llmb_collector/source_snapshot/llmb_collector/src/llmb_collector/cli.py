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

"""CLI entrypoint for llmb-collector."""

import argparse
import os
import sys
from pathlib import Path
from typing import Literal

from llmb_collector.collect import (
    CONTAINER_COMMANDS,
    CONTAINER_DOMAINS,
    DEFAULT_ENV_PATTERN,
    NETWORK_COMMANDS,
    NETWORK_DOMAINS,
    SYSTEM_COMMANDS,
    SYSTEM_DOMAINS,
    CollectConfig,
    _serialize,
    _write_to_dir,
    collect,
)
from llmb_collector.command_loader import (
    CommandFinding,
    load_command_registry,
)


def _str_to_bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    v = value.lower().strip()
    if v in ("true", "t", "yes", "y", "1"):
        return True
    if v in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {value!r}")


def _infer_format_from_path(path: str) -> Literal["json", "yaml", "text"]:
    p = path.lower()
    if p.endswith(".yaml") or p.endswith(".yml"):
        return "yaml"
    if p.endswith(".txt"):
        return "text"
    return "json"


def _no_collectors_selected_message() -> str:
    def _command_lines(names: list[str], command_map: dict[str, list[str]]) -> str:
        return "\n".join(f"    - {' '.join(command_map[name])}" for name in names if name in command_map)

    host_names = list(SYSTEM_COMMANDS)
    network_names = list(NETWORK_COMMANDS)
    container_names = list(CONTAINER_COMMANDS)

    host_domains = ", ".join(SYSTEM_DOMAINS.keys())
    network_domains = ", ".join(NETWORK_DOMAINS.keys())
    container_domains = ", ".join(CONTAINER_DOMAINS.keys())

    return (
        "At least one collection target must be enabled.\n"
        "\n"
        "Available collectors by subject:\n"
        "  Host/system (enable with --host)\n"
        f"    Domains: {host_domains or '(none)'}\n"
        f"{_command_lines(host_names, SYSTEM_COMMANDS)}\n"
        "  Network (enable with --network)\n"
        f"    Domains: {network_domains or '(none)'}\n"
        f"{_command_lines(network_names, NETWORK_COMMANDS)}\n"
        "  Container (enable with --container)\n"
        f"    Domains: {container_domains}\n"
        f"{_command_lines(container_names, CONTAINER_COMMANDS)}\n"
        "  Environment (enable with --env)\n"
        "    Uses --env-pattern regex to filter environment variables\n"
        "\n"
        "Examples:\n"
        "  llmb-collect collect --host-collect cpu,gpu --format yaml\n"
        "  llmb-collect collect --network-collect network --format yaml\n"
        "  llmb-collect collect --container-collect gpu --format yaml\n"
    )


def _domains_listing_message(section: str = "all") -> str:
    def _domain_lines(domains: dict[str, list[str]]) -> str:
        return "\n".join(f"  - {domain}: {', '.join(names)}" for domain, names in domains.items())

    if section == "host":
        return "Available host domains and sub-collectors:\n" f"{_domain_lines(SYSTEM_DOMAINS)}\n"
    if section == "network":
        return "Available network domains and sub-collectors:\n" f"{_domain_lines(NETWORK_DOMAINS)}\n"
    if section == "container":
        return "Available container domains and sub-collectors:\n" f"{_domain_lines(CONTAINER_DOMAINS)}\n"
    return (
        "Available domains and sub-collectors:\n"
        "\n"
        "Host domains:\n"
        f"{_domain_lines(SYSTEM_DOMAINS)}\n"
        "\n"
        "Network domains:\n"
        f"{_domain_lines(NETWORK_DOMAINS)}\n"
        "\n"
        "Container domains:\n"
        f"{_domain_lines(CONTAINER_DOMAINS)}\n"
    )


def _add_collect_arguments(parser: argparse.ArgumentParser) -> None:
    host_family = parser.add_argument_group("host")
    host_mode_group = host_family.add_mutually_exclusive_group()
    host_mode_group.add_argument(
        "--host",
        action="store_true",
        default=_str_to_bool(os.environ.get("LLMB_COLLECT_HOST", "false")),
        help="Collect all host/system info.",
    )
    host_mode_group.add_argument(
        "--host-collect",
        type=str,
        default=os.environ.get("LLMB_HOST_COLLECT", ""),
        metavar="LIST",
        help="Collect only specified host commands/domains.",
    )
    host_family.add_argument(
        "--host-exclude",
        type=str,
        default=os.environ.get("LLMB_HOST_EXCLUDE", ""),
        metavar="LIST",
        help="Comma-separated host commands or domains to skip (blacklist).",
    )

    network_family = parser.add_argument_group("network")
    network_mode_group = network_family.add_mutually_exclusive_group()
    network_mode_group.add_argument(
        "--network",
        action="store_true",
        default=_str_to_bool(os.environ.get("LLMB_COLLECT_NETWORK", "false")),
        help="Collect all network info.",
    )
    network_mode_group.add_argument(
        "--network-collect",
        type=str,
        default=os.environ.get("LLMB_NETWORK_COLLECT", ""),
        metavar="LIST",
        help="Collect only specified network commands/domains.",
    )
    network_family.add_argument(
        "--network-exclude",
        type=str,
        default=os.environ.get("LLMB_NETWORK_EXCLUDE", ""),
        metavar="LIST",
        help="Comma-separated network commands or domains to skip (blacklist).",
    )

    container_family = parser.add_argument_group("container")
    container_mode_group = container_family.add_mutually_exclusive_group()
    container_mode_group.add_argument(
        "--container",
        action="store_true",
        default=_str_to_bool(os.environ.get("LLMB_COLLECT_CONTAINER", "false")),
        help="Collect all container info.",
    )
    container_mode_group.add_argument(
        "--container-collect",
        type=str,
        default=os.environ.get("LLMB_CONTAINER_COLLECT", ""),
        metavar="LIST",
        help="Collect only specified container commands/domains.",
    )
    container_family.add_argument(
        "--container-exclude",
        type=str,
        default=os.environ.get("LLMB_CONTAINER_EXCLUDE", ""),
        metavar="LIST",
        help="Comma-separated container commands or domains to skip (blacklist).",
    )

    environment_family = parser.add_argument_group("environment")
    environment_family.add_argument(
        "--env",
        action="store_true",
        default=_str_to_bool(os.environ.get("LLMB_COLLECT_ENV", "false")),
        help="Collect filtered env vars.",
    )
    environment_family.add_argument(
        "--env-pattern",
        type=str,
        default=DEFAULT_ENV_PATTERN,
        metavar="REGEX",
        help="Regex for env vars to include.",
    )

    output_family = parser.add_argument_group("output")
    output_family.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        metavar="PATH",
        help="Output file, directory, or '-' for stdout. If omitted, print to screen.",
    )
    output_family.add_argument(
        "--format",
        "-f",
        choices=["json", "yaml", "text"],
        default="json",
        help="Output format for file or stdout (default: json).",
    )
    output_family.add_argument(
        "--compact",
        action="store_true",
        default=True,
        help="JSON: no indent (default).",
    )
    output_family.add_argument(
        "--no-compact",
        action="store_false",
        dest="compact",
        help="JSON: indent for readability.",
    )
    output_family.add_argument(
        "--result-dir",
        type=str,
        default=os.environ.get("CLOUDPERF_RESULT_DIR", "."),
        metavar="DIR",
        help="Result directory when writing to a directory (default: .).",
    )


def _parse_csv(value: str) -> list[str] | None:
    items = [x.strip() for x in value.split(",") if x.strip()]
    return items or None


def _write_output(data: dict, out_path: str | None, fmt: str, compact: bool) -> None:
    if out_path is None or out_path == "-":
        sys.stdout.write(_serialize(data, fmt, compact))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return

    path = Path(out_path)
    if path.is_dir():
        _write_to_dir(data, path)
        return

    if path.suffix.lower() in (".json", ".yaml", ".yml", ".txt"):
        file_fmt = _infer_format_from_path(out_path)
        text = _serialize(data, file_fmt, compact)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
        return

    # No extension or unknown: treat as directory.
    _write_to_dir(data, path)


def _format_validation_finding(finding: CommandFinding) -> str:
    left_id, right_id = finding.command_ids
    left_source, right_source = finding.sources
    left_argv, right_argv = finding.argv
    return (
        f"- [{finding.severity}] {finding.kind}: {left_id}, {right_id}\n"
        f"  {left_id}: {' '.join(left_argv)} ({left_source})\n"
        f"  {right_id}: {' '.join(right_argv)} ({right_source})\n"
        f"  {finding.message}"
    )


def _validation_report() -> str:
    registry = load_command_registry()
    if not registry.findings:
        return "Command config OK: no duplicate or near-duplicate commands found.\n"
    body = "\n".join(_format_validation_finding(finding) for finding in registry.findings)
    return f"Command config findings:\n{body}\n"


def _run_collect(args: argparse.Namespace) -> None:
    host_collect_raw = args.host_collect.strip()
    network_collect_raw = args.network_collect.strip()
    container_collect_raw = args.container_collect.strip()
    host_enabled = args.host or bool(host_collect_raw)
    network_enabled = args.network or bool(network_collect_raw)
    container_enabled = args.container or bool(container_collect_raw)

    if not any([host_enabled, network_enabled, container_enabled, args.env]):
        sys.stderr.write(_no_collectors_selected_message())
        return

    host_collect = _parse_csv(host_collect_raw)
    network_collect = _parse_csv(network_collect_raw)
    host_exclude = _parse_csv(args.host_exclude)
    network_exclude = _parse_csv(args.network_exclude)
    container_collect = _parse_csv(container_collect_raw)
    container_exclude = _parse_csv(args.container_exclude)

    config = CollectConfig(
        result_dir=args.result_dir,
        output_path=args.output if args.output and args.output != "-" else None,
        format=args.format,
        compact=args.compact,
        host=host_enabled,
        network=network_enabled,
        container=container_enabled,
        env=args.env,
        env_pattern=args.env_pattern,
        host_collect=host_collect,
        host_exclude=host_exclude,
        network_collect=network_collect,
        network_exclude=network_exclude,
        container_collect=container_collect,
        container_exclude=container_exclude,
    )
    data = collect(config)

    _write_output(data, args.output, config.format, config.compact)


def _run_list(_: argparse.Namespace) -> None:
    sys.stdout.write(_domains_listing_message())


def _run_list_host(_: argparse.Namespace) -> None:
    sys.stdout.write(_domains_listing_message("host"))


def _run_list_network(_: argparse.Namespace) -> None:
    sys.stdout.write(_domains_listing_message("network"))


def _run_list_container(_: argparse.Namespace) -> None:
    sys.stdout.write(_domains_listing_message("container"))


def _run_validate_config(_: argparse.Namespace) -> None:
    sys.stdout.write(_validation_report())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape cluster and hardware info; output JSON, YAML, text, or directory layout.",
    )
    subparsers = parser.add_subparsers(dest="command")

    collect_parser = subparsers.add_parser(
        "collect",
        help="Collect cluster and hardware information.",
    )
    _add_collect_arguments(collect_parser)
    collect_parser.set_defaults(handler=_run_collect)

    list_parser = subparsers.add_parser("list", help="List available collector metadata.")
    list_parser.set_defaults(handler=_run_list)
    list_subparsers = list_parser.add_subparsers(dest="list_command")
    list_host_parser = list_subparsers.add_parser(
        "host",
        help="List host domains and sub-collectors.",
    )
    list_host_parser.set_defaults(handler=_run_list_host)
    list_network_parser = list_subparsers.add_parser(
        "network",
        help="List network domains and sub-collectors.",
    )
    list_network_parser.set_defaults(handler=_run_list_network)
    list_container_parser = list_subparsers.add_parser(
        "container",
        help="List container domains and sub-collectors.",
    )
    list_container_parser.set_defaults(handler=_run_list_container)

    validate_config_parser = subparsers.add_parser(
        "validate-config",
        help="Validate command config and report duplicate or near-duplicate commands.",
    )
    validate_config_parser.set_defaults(handler=_run_validate_config)

    args = parser.parse_args()
    if not hasattr(args, "handler"):
        parser.print_help()
        return
    args.handler(args)
