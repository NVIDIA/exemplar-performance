# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
from offline_replayer.measurement.network_policy import (
    LEGACY_NETWORK_UNSET,
    NETWORK_POLICY_SCHEMA_VERSION,
    NetworkMode,
    NetworkPolicy,
    NetworkPolicyError,
    normalize_allowed_host,
    resolve_cli_network_policy,
    resolve_network_policy,
)


def test_modes_and_empty_allowlist_effective_behavior() -> None:
    assert NetworkPolicy().effective_mode is NetworkMode.PUBLIC
    assert NetworkPolicy("no-network").effective_mode is NetworkMode.NO_NETWORK

    empty = NetworkPolicy("allowlist")
    assert empty.mode is NetworkMode.ALLOWLIST
    assert empty.effective_mode is NetworkMode.NO_NETWORK
    assert empty.allowed_hosts == ()
    assert empty.to_report_payload()["requested_mode"] == "allowlist"
    assert empty.to_report_payload()["effective_mode"] == "no-network"


def test_allowlist_is_lowercase_sorted_and_deduplicated() -> None:
    policy = NetworkPolicy(
        NetworkMode.ALLOWLIST,
        ("PyPI.org", "files.PYTHONHOSTED.org", "pypi.ORG", "allowed.test"),
    )

    assert policy.allowed_hosts == (
        "allowed.test",
        "files.pythonhosted.org",
        "pypi.org",
    )


@pytest.mark.parametrize(
    "host",
    [
        "",
        " pypi.org",
        "pypi.org ",
        "https://pypi.org",
        "*.pypi.org",
        ".pypi.org",
        "pypi.org.",
        "pypi..org",
        "bad_label.example",
        "-bad.example",
        "bad-.example",
        "pypi.org:443",
        "127.0.0.1",
        "2001:db8::1",
        "münchen.example",
        "localhost",
        "service.localhost",
        "printer.local",
        "intranet",
    ],
)
def test_invalid_or_ambiguous_allowed_hosts_are_rejected(host: str) -> None:
    with pytest.raises(NetworkPolicyError):
        normalize_allowed_host(host)


def test_dns_length_limits_are_enforced() -> None:
    assert normalize_allowed_host("a" * 63 + ".example") == "a" * 63 + ".example"
    with pytest.raises(NetworkPolicyError, match="DNS hostname limit"):
        normalize_allowed_host("a" * 63 + "." + "b" * 63 + "." + "c" * 63 + "." + "d" * 62)
    with pytest.raises(NetworkPolicyError, match="exact ASCII DNS"):
        normalize_allowed_host("a" * 64 + ".example")


def test_ordinary_two_label_special_use_test_name_is_valid() -> None:
    assert normalize_allowed_host("Allowed.TEST") == "allowed.test"


@pytest.mark.parametrize("mode", [NetworkMode.PUBLIC, NetworkMode.NO_NETWORK])
def test_allowed_hosts_are_rejected_outside_allowlist(mode: NetworkMode) -> None:
    with pytest.raises(NetworkPolicyError, match="allowed hosts require"):
        NetworkPolicy(mode, ("example.com",))


def test_unknown_mode_and_non_string_hosts_are_rejected() -> None:
    with pytest.raises(NetworkPolicyError, match="network mode"):
        NetworkPolicy("private")
    with pytest.raises(NetworkPolicyError, match="allowed hosts must be strings"):
        NetworkPolicy("allowlist", ("example.com", 443))  # type: ignore[arg-type]


def test_report_and_comparability_payloads_are_canonical() -> None:
    first = NetworkPolicy("allowlist", ("B.example", "a.example", "A.EXAMPLE"))
    second = NetworkPolicy("allowlist", ("a.example", "b.example"))

    assert first.to_report_payload() == {
        "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
        "requested_mode": "allowlist",
        "effective_mode": "allowlist",
        "allowed_hosts": ["a.example", "b.example"],
        "host_matching": "exact-ascii-dns-case-insensitive",
        "includes_subdomains": False,
        "host_ports_supported": False,
    }
    assert first.to_comparability_payload() == second.to_comparability_payload()
    assert first.comparability_digest == second.comparability_digest

    empty_allowlist = NetworkPolicy("allowlist")
    no_network = NetworkPolicy("no-network")
    assert empty_allowlist.to_comparability_payload() == no_network.to_comparability_payload()
    assert empty_allowlist.comparability_digest == no_network.comparability_digest


@pytest.mark.parametrize(
    ("legacy", "expected"),
    [(None, NetworkMode.PUBLIC), ("none", NetworkMode.NO_NETWORK)],
)
def test_legacy_network_mapping(legacy: str | None, expected: NetworkMode) -> None:
    assert NetworkPolicy.from_legacy_network(legacy).effective_mode is expected
    assert resolve_network_policy(legacy_network=legacy).effective_mode is expected


def test_named_legacy_docker_network_has_no_implicit_mapping() -> None:
    with pytest.raises(NetworkPolicyError, match="named Docker networks"):
        NetworkPolicy.from_legacy_network("bridge")


def test_resolver_distinguishes_omitted_legacy_value_from_none() -> None:
    assert LEGACY_NETWORK_UNSET is not None
    allowlist = resolve_network_policy(mode="allowlist", allowed_hosts=("example.com",))
    assert allowlist.effective_mode is NetworkMode.ALLOWLIST

    with pytest.raises(NetworkPolicyError, match="conflicting network configuration"):
        resolve_network_policy(
            mode="allowlist",
            allowed_hosts=("example.com",),
            legacy_network=None,
        )


def test_resolver_allows_equivalent_dual_inputs_and_rejects_conflicts() -> None:
    explicit = NetworkPolicy("no-network")
    assert resolve_network_policy(policy=explicit, legacy_network="none") is explicit

    empty_allowlist = resolve_network_policy(mode="allowlist", legacy_network="none")
    assert empty_allowlist.mode is NetworkMode.ALLOWLIST
    assert empty_allowlist.effective_mode is NetworkMode.NO_NETWORK

    with pytest.raises(NetworkPolicyError, match="conflicting network configuration"):
        resolve_network_policy(mode="no-network", legacy_network=None)


def test_resolver_rejects_ambiguous_new_inputs() -> None:
    with pytest.raises(NetworkPolicyError, match="separate mode"):
        resolve_network_policy(
            policy=NetworkPolicy("allowlist", ("example.com",)),
            mode="allowlist",
        )
    with pytest.raises(NetworkPolicyError, match="explicit allowlist mode"):
        resolve_network_policy(allowed_hosts=("example.com",))


def test_cli_resolver_preserves_named_legacy_network_escape_hatch() -> None:
    assert (
        resolve_cli_network_policy(
            mode=None,
            allowed_hosts=(),
            legacy_network="existing-test-network",
        )
        is None
    )

    with pytest.raises(NetworkPolicyError, match="named legacy --network"):
        resolve_cli_network_policy(
            mode="allowlist",
            allowed_hosts=("example.com",),
            legacy_network="existing-test-network",
        )


def test_cli_resolver_maps_legacy_and_typed_deny_all_forms() -> None:
    legacy = resolve_cli_network_policy(
        mode=None,
        allowed_hosts=(),
        legacy_network="none",
    )
    typed = resolve_cli_network_policy(
        mode="no-network",
        allowed_hosts=(),
        legacy_network=None,
    )

    assert legacy is not None and typed is not None
    assert legacy.to_comparability_payload() == typed.to_comparability_payload()
