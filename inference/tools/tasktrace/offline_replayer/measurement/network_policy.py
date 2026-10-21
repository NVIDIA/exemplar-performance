# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Typed, canonical network-policy configuration for offline replay.

Version 1 allowlist entries are exact, multi-label ASCII DNS hostnames.
Matching is case-insensitive after canonical lower-casing, and an entry does
not include its subdomains.  URL syntax, wildcards, IP literals, trailing root
dots, ``host:port`` entries, single-label names, and local/localhost namespaces
are rejected deliberately.  Port/protocol and resolved-address restrictions
belong to the enforcement-backend contract rather than being guessed by this
portable policy model.

This module models requested policy only; it does not claim that a runtime has
enforced it.  An enforcement implementation must report its own validation
alongside :meth:`NetworkPolicy.to_report_payload`.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable

NETWORK_POLICY_SCHEMA_VERSION = "tasktrace-network-policy.v1"

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_LEGACY_NETWORK_UNSET = object()
# Public so callers migrating an option object can distinguish an omitted
# legacy flag from the legacy value ``None`` (which means public networking).
LEGACY_NETWORK_UNSET: Final[object] = _LEGACY_NETWORK_UNSET


class NetworkPolicyError(ValueError):
    """Raised when a network policy is ambiguous or cannot be enforced safely."""


class NetworkMode(str, Enum):
    """Portable offline-replay network modes."""

    PUBLIC = "public"
    NO_NETWORK = "no-network"
    ALLOWLIST = "allowlist"


def normalize_allowed_host(value: str) -> str:
    """Return one canonical exact DNS hostname or raise ``NetworkPolicyError``."""
    if not isinstance(value, str):
        raise NetworkPolicyError("allowed hosts must be strings")
    if not value:
        raise NetworkPolicyError("allowed hosts must not be empty")
    if value != value.strip():
        raise NetworkPolicyError(f"allowed host {value!r} must not contain surrounding whitespace")
    if not value.isascii():
        raise NetworkPolicyError(f"allowed host {value!r} must be an ASCII DNS hostname")

    host = value.lower()
    if host.endswith("."):
        raise NetworkPolicyError(f"allowed host {value!r} must not use a trailing DNS root dot")
    if len(host) > 253:
        raise NetworkPolicyError(f"allowed host {value!r} exceeds the 253-byte DNS hostname limit")

    # Check IP literals explicitly so policy authors cannot accidentally rely
    # on DNS-only enforcement for an address that bypasses hostname matching.
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise NetworkPolicyError(f"allowed host {value!r} must be a DNS hostname, not an IP literal")

    labels = host.split(".")
    if any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise NetworkPolicyError(
            f"allowed host {value!r} is not an exact ASCII DNS hostname; "
            "URLs, wildcards, ports, underscores, and empty labels are unsupported"
        )
    if host == "localhost" or host.endswith(".localhost"):
        raise NetworkPolicyError(f"allowed host {value!r} uses the prohibited localhost namespace")
    if host == "local" or host.endswith(".local"):
        raise NetworkPolicyError(f"allowed host {value!r} uses the prohibited local namespace")
    if len(labels) < 2:
        raise NetworkPolicyError(f"allowed host {value!r} must be a multi-label DNS hostname")
    return host


@dataclass(frozen=True)
class NetworkPolicy:
    """Normalized network intent, independent of a particular container runtime.

    ``mode`` retains the requested mode for provenance.  An empty allowlist is
    valid and has an effective mode of :attr:`NetworkMode.NO_NETWORK`.
    """

    mode: NetworkMode | str = NetworkMode.PUBLIC
    allowed_hosts: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            mode = NetworkMode(self.mode)
        except (TypeError, ValueError) as exc:
            supported = ", ".join(item.value for item in NetworkMode)
            raise NetworkPolicyError(f"network mode must be one of: {supported}") from exc

        try:
            hosts = tuple(sorted({normalize_allowed_host(host) for host in self.allowed_hosts}))
        except TypeError as exc:
            raise NetworkPolicyError("allowed_hosts must be an iterable of strings") from exc

        if mode is not NetworkMode.ALLOWLIST and hosts:
            raise NetworkPolicyError(
                f"allowed hosts require mode={NetworkMode.ALLOWLIST.value!r}; "
                f"mode={mode.value!r} does not accept them"
            )
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "allowed_hosts", hosts)

    @property
    def effective_mode(self) -> NetworkMode:
        """Return the behavior a runtime must enforce."""
        if self.mode is NetworkMode.ALLOWLIST and not self.allowed_hosts:
            return NetworkMode.NO_NETWORK
        return self.mode

    def to_report_payload(self) -> dict[str, Any]:
        """Return canonical policy provenance for an execution report."""
        return {
            "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
            "requested_mode": self.mode.value,
            "effective_mode": self.effective_mode.value,
            "allowed_hosts": list(self.allowed_hosts),
            "host_matching": "exact-ascii-dns-case-insensitive",
            "includes_subdomains": False,
            "host_ports_supported": False,
        }

    def to_comparability_payload(self) -> dict[str, Any]:
        """Return only behavior that must match across comparable runs.

        Empty ``allowlist`` and explicit ``no-network`` requests intentionally
        produce identical payloads because their required runtime behavior is
        identical.
        """
        return {
            "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
            "effective_mode": self.effective_mode.value,
            "allowed_hosts": (list(self.allowed_hosts) if self.effective_mode is NetworkMode.ALLOWLIST else []),
            "host_matching": "exact-ascii-dns-case-insensitive",
            "includes_subdomains": False,
            "host_ports_supported": False,
        }

    @property
    def comparability_digest(self) -> str:
        """Return a stable digest suitable for reproducibility checks."""
        encoded = json.dumps(
            self.to_comparability_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(b"tasktrace-network-policy-comparability-v1\0")
        digest.update(encoded)
        return digest.hexdigest()

    @classmethod
    def from_legacy_network(cls, network: str | None) -> NetworkPolicy:
        """Map the former Docker ``network`` field without broadening access.

        The legacy values in the supported public API are ``None`` (default
        public networking) and ``"none"``.  Named Docker networks are not
        portable policy and therefore require an explicit migration decision.
        """
        if network is None:
            return cls(NetworkMode.PUBLIC)
        if network == "none":
            return cls(NetworkMode.NO_NETWORK)
        raise NetworkPolicyError(
            "legacy network must be None or 'none'; named Docker networks do not "
            "have a portable network-policy mapping"
        )


def resolve_network_policy(
    *,
    policy: NetworkPolicy | None = None,
    mode: NetworkMode | str | None = None,
    allowed_hosts: Iterable[str] = (),
    legacy_network: str | None | object = LEGACY_NETWORK_UNSET,
) -> NetworkPolicy:
    """Resolve new and legacy inputs, rejecting behavior-changing conflicts.

    ``LEGACY_NETWORK_UNSET`` means the old option was not supplied.  Passing
    ``None`` explicitly means the legacy public-network value.  Equivalent
    dual specifications are accepted to make staged migrations possible.
    """
    supplied_hosts = tuple(allowed_hosts)
    if policy is not None:
        if not isinstance(policy, NetworkPolicy):
            raise NetworkPolicyError("policy must be a NetworkPolicy")
        if mode is not None or supplied_hosts:
            raise NetworkPolicyError("policy cannot be combined with separate mode or allowed_hosts inputs")
        requested = policy
        has_new_input = True
    elif mode is not None:
        requested = NetworkPolicy(mode=mode, allowed_hosts=supplied_hosts)
        has_new_input = True
    elif supplied_hosts:
        raise NetworkPolicyError("allowed_hosts requires an explicit allowlist mode")
    else:
        requested = NetworkPolicy(NetworkMode.PUBLIC)
        has_new_input = False

    has_legacy_input = legacy_network is not LEGACY_NETWORK_UNSET
    if not has_legacy_input:
        return requested

    legacy = NetworkPolicy.from_legacy_network(legacy_network)  # type: ignore[arg-type]
    if not has_new_input:
        return legacy
    if requested.to_comparability_payload() != legacy.to_comparability_payload():
        raise NetworkPolicyError(
            "conflicting network configuration: "
            f"new policy is {requested.effective_mode.value!r}, but legacy "
            f"network maps to {legacy.effective_mode.value!r}"
        )
    return requested


def resolve_cli_network_policy(
    *,
    mode: NetworkMode | str | None,
    allowed_hosts: Iterable[str],
    legacy_network: str | None,
) -> NetworkPolicy | None:
    """Resolve command-line inputs while retaining named legacy networks.

    A named ``--network`` value remains an explicit Docker-only escape hatch
    for backward compatibility. It cannot be combined with semantic policy
    flags and returns ``None`` so callers preserve the raw runtime setting.
    """
    hosts = tuple(allowed_hosts)
    if legacy_network not in (None, "none"):
        if mode is not None or hosts:
            raise NetworkPolicyError(
                "named legacy --network cannot be combined with " "--network-policy or --allowed-host"
            )
        return None
    return resolve_network_policy(
        mode=mode,
        allowed_hosts=hosts,
        legacy_network=(legacy_network if legacy_network is not None else LEGACY_NETWORK_UNSET),
    )
