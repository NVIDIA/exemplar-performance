#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate a fresh no-network namespace, then exec Enroot inside it."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

EVIDENCE_FILENAME = "enroot_network_enforcement.json"
SCHEMA_VERSION = "tasktrace-enroot-network-enforcement.v1"
BACKEND = "linux-user-network-namespace-v1"


_LOOPBACK_IPV6_ADDRESS = "00000000000000000000000000000001"


def _loopback_only_ipv6(addresses: list[str]) -> bool:
    """Only the loopback address on lo is permitted (bubblewrap ups lo)."""
    for line in addresses:
        fields = line.split()
        if not fields or fields[0] != _LOOPBACK_IPV6_ADDRESS or fields[-1] != "lo":
            return False
    return True


def _netns_inode() -> int:
    return Path("/proc/self/ns/net").stat().st_ino


def _interfaces() -> list[str]:
    lines = Path("/proc/net/dev").read_text().splitlines()[2:]
    return sorted(line.split(":", 1)[0].strip() for line in lines if ":" in line)


def validate(
    parent_netns_inode: int,
    controller_euid: int,
    controller_egid: int,
) -> dict[str, object]:
    """Return proof that this process has no usable IP network."""
    netns_inode = _netns_inode()
    interfaces = _interfaces()
    routes = [line.strip() for line in Path("/proc/net/route").read_text().splitlines()[1:] if line.strip()]
    ipv6 = [line.strip() for line in Path("/proc/net/if_inet6").read_text().splitlines() if line.strip()]
    if netns_inode == parent_netns_inode:
        raise RuntimeError("network namespace did not change")
    if os.geteuid() != controller_euid or os.getegid() != controller_egid:
        raise RuntimeError("user namespace did not preserve controller UID/GID")
    if interfaces != ["lo"] or routes or not _loopback_only_ipv6(ipv6):
        raise RuntimeError(f"namespace is not isolated: interfaces={interfaces}, routes={routes}, ipv6={ipv6}")
    return {
        "schema_version": SCHEMA_VERSION,
        "state": "enforced",
        "backend": BACKEND,
        "policy_effective_mode": "no-network",
        "valid": True,
        "parent_netns_inode": parent_netns_inode,
        "netns_inode": netns_inode,
        "netns_differs": True,
        "controller_euid": controller_euid,
        "controller_egid": controller_egid,
        "namespace_euid": os.geteuid(),
        "namespace_egid": os.getegid(),
        "interfaces": interfaces,
        "ipv4_routes": routes,
        "ipv6_addresses": ipv6,
    }


def validate_evidence(
    payload: Mapping[str, Any],
    *,
    parent_netns_inode: int,
    controller_euid: int,
    controller_egid: int,
) -> dict[str, Any]:
    """Fail closed when retained evidence does not match this controller."""
    child_inode = payload.get("netns_inode")
    required = {
        "schema_version": SCHEMA_VERSION,
        "state": "enforced",
        "backend": BACKEND,
        "policy_effective_mode": "no-network",
        "valid": True,
        "parent_netns_inode": parent_netns_inode,
        "netns_differs": True,
        "controller_euid": controller_euid,
        "controller_egid": controller_egid,
        "namespace_euid": controller_euid,
        "namespace_egid": controller_egid,
        "interfaces": ["lo"],
        "ipv4_routes": [],
    }
    if any(payload.get(key) != value for key, value in required.items()):
        raise RuntimeError("Enroot no-network evidence is invalid")
    if not _loopback_only_ipv6(payload.get("ipv6_addresses", [])):
        raise RuntimeError("Enroot no-network evidence is invalid")
    if (
        not isinstance(child_inode, int)
        or isinstance(child_inode, bool)
        or child_inode <= 0
        or child_inode == parent_netns_inode
    ):
        raise RuntimeError("Enroot no-network namespace identity is invalid")
    return dict(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parent-netns-inode", type=int, required=True)
    parser.add_argument("--controller-euid", type=int, required=True)
    parser.add_argument("--controller-egid", type=int, required=True)
    parser.add_argument("--evidence-path", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    command = list(args.command)
    if command and command[0] == "--":
        command.pop(0)
    try:
        if not command:
            raise RuntimeError("missing Enroot command")
        if args.evidence_path.exists():
            raise RuntimeError(f"evidence path already exists: {args.evidence_path}")
        evidence = validate(
            args.parent_netns_inode,
            args.controller_euid,
            args.controller_egid,
        )
        args.evidence_path.parent.mkdir(parents=True, exist_ok=True)
        args.evidence_path.write_text(json.dumps(evidence, sort_keys=True) + "\n")
        os.execvp(command[0], command)
    except (OSError, RuntimeError) as exc:
        print(f"Enroot no-network failed: {exc}", file=sys.stderr)
        return 125
    return 125


if __name__ == "__main__":
    raise SystemExit(main())
