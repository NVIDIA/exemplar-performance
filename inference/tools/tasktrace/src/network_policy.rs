// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Typed, canonical network-policy configuration for offline replay.
//!
//! Version 1 allowlists contain exact, multi-label ASCII DNS hostnames. This
//! module records requested behavior only; a runtime enforcement backend must
//! retain its own validation evidence separately.

use serde_json::{json, Value};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use std::net::IpAddr;
use std::str::FromStr;

pub const NETWORK_POLICY_SCHEMA_VERSION: &str = "tasktrace-network-policy.v1";
const COMPARABILITY_DOMAIN: &[u8] = b"tasktrace-network-policy-comparability-v1\0";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkPolicyError(String);

impl NetworkPolicyError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for NetworkPolicyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for NetworkPolicyError {}

#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum NetworkMode {
    Public,
    NoNetwork,
    Allowlist,
}

impl NetworkMode {
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Public => "public",
            Self::NoNetwork => "no-network",
            Self::Allowlist => "allowlist",
        }
    }
}

impl fmt::Display for NetworkMode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl FromStr for NetworkMode {
    type Err = NetworkPolicyError;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value {
            "public" => Ok(Self::Public),
            "no-network" => Ok(Self::NoNetwork),
            "allowlist" => Ok(Self::Allowlist),
            _ => Err(NetworkPolicyError::new(
                "network mode must be one of: public, no-network, allowlist",
            )),
        }
    }
}

fn valid_dns_label(label: &str) -> bool {
    let bytes = label.as_bytes();
    if bytes.is_empty() || bytes.len() > 63 {
        return false;
    }
    let alphanumeric = |byte: u8| byte.is_ascii_lowercase() || byte.is_ascii_digit();
    if !alphanumeric(bytes[0]) || !alphanumeric(bytes[bytes.len() - 1]) {
        return false;
    }
    bytes
        .iter()
        .all(|byte| alphanumeric(*byte) || *byte == b'-')
}

/// Return one canonical exact DNS hostname.
pub fn normalize_allowed_host(value: &str) -> Result<String, NetworkPolicyError> {
    if value.is_empty() {
        return Err(NetworkPolicyError::new("allowed hosts must not be empty"));
    }
    if value != value.trim() {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} must not contain surrounding whitespace"
        )));
    }
    if !value.is_ascii() {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} must be an ASCII DNS hostname"
        )));
    }
    let host = value.to_ascii_lowercase();
    if host.ends_with('.') {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} must not use a trailing DNS root dot"
        )));
    }
    if host.len() > 253 {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} exceeds the 253-byte DNS hostname limit"
        )));
    }
    if host.parse::<IpAddr>().is_ok() {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} must be a DNS hostname, not an IP literal"
        )));
    }
    let labels: Vec<&str> = host.split('.').collect();
    if labels.iter().any(|label| !valid_dns_label(label)) {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} is not an exact ASCII DNS hostname; \
             URLs, wildcards, ports, underscores, and empty labels are unsupported"
        )));
    }
    if host == "localhost" || host.ends_with(".localhost") {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} uses the prohibited localhost namespace"
        )));
    }
    if host == "local" || host.ends_with(".local") {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} uses the prohibited local namespace"
        )));
    }
    if labels.len() < 2 {
        return Err(NetworkPolicyError::new(format!(
            "allowed host {value:?} must be a multi-label DNS hostname"
        )));
    }
    Ok(host)
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkPolicy {
    mode: NetworkMode,
    allowed_hosts: Vec<String>,
}

impl Default for NetworkPolicy {
    fn default() -> Self {
        Self {
            mode: NetworkMode::Public,
            allowed_hosts: Vec::new(),
        }
    }
}

impl NetworkPolicy {
    pub fn new<I, S>(mode: NetworkMode, allowed_hosts: I) -> Result<Self, NetworkPolicyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let hosts: BTreeSet<String> = allowed_hosts
            .into_iter()
            .map(|host| normalize_allowed_host(host.as_ref()))
            .collect::<Result<_, _>>()?;
        if mode != NetworkMode::Allowlist && !hosts.is_empty() {
            return Err(NetworkPolicyError::new(format!(
                "allowed hosts require mode={:?}; mode={:?} does not accept them",
                NetworkMode::Allowlist.as_str(),
                mode.as_str()
            )));
        }
        Ok(Self {
            mode,
            allowed_hosts: hosts.into_iter().collect(),
        })
    }

    pub fn parse<I, S>(mode: &str, allowed_hosts: I) -> Result<Self, NetworkPolicyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        Self::new(NetworkMode::from_str(mode)?, allowed_hosts)
    }

    pub fn mode(&self) -> NetworkMode {
        self.mode
    }

    pub fn allowed_hosts(&self) -> &[String] {
        &self.allowed_hosts
    }

    pub fn effective_mode(&self) -> NetworkMode {
        if self.mode == NetworkMode::Allowlist && self.allowed_hosts.is_empty() {
            NetworkMode::NoNetwork
        } else {
            self.mode
        }
    }

    pub fn to_report_payload(&self) -> Value {
        json!({
            "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
            "requested_mode": self.mode.as_str(),
            "effective_mode": self.effective_mode().as_str(),
            "allowed_hosts": self.allowed_hosts,
            "host_matching": "exact-ascii-dns-case-insensitive",
            "includes_subdomains": false,
            "host_ports_supported": false,
        })
    }

    pub fn to_comparability_payload(&self) -> Value {
        let effective_mode = self.effective_mode();
        json!({
            "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
            "effective_mode": effective_mode.as_str(),
            "allowed_hosts": if effective_mode == NetworkMode::Allowlist {
                self.allowed_hosts.clone()
            } else {
                Vec::<String>::new()
            },
            "host_matching": "exact-ascii-dns-case-insensitive",
            "includes_subdomains": false,
            "host_ports_supported": false,
        })
    }

    pub fn comparability_digest(&self) -> String {
        let encoded = canonical_json(&self.to_comparability_payload());
        let mut input = Vec::with_capacity(COMPARABILITY_DOMAIN.len() + encoded.len());
        input.extend_from_slice(COMPARABILITY_DOMAIN);
        input.extend_from_slice(encoded.as_bytes());
        hex_digest(&sha256(&input))
    }

    pub fn from_legacy_network(network: Option<&str>) -> Result<Self, NetworkPolicyError> {
        match network {
            None => Ok(Self::default()),
            Some("none") => Self::new(NetworkMode::NoNetwork, std::iter::empty::<&str>()),
            Some(_) => Err(NetworkPolicyError::new(
                "legacy network must be None or 'none'; named Docker networks do not \
                 have a portable network-policy mapping",
            )),
        }
    }
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.to_string(),
        Value::String(value) => {
            serde_json::to_string(value).expect("JSON string is always serializable")
        }
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut fields: Vec<_> = values.iter().collect();
            fields.sort_by(|(left, _), (right, _)| left.cmp(right));
            format!(
                "{{{}}}",
                fields
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}:{}",
                        serde_json::to_string(key).expect("JSON object key is always serializable"),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

/// Distinguishes an omitted legacy field from its explicit public (`None`) value.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum LegacyNetwork {
    Unset,
    Explicit(Option<String>),
}

impl Default for LegacyNetwork {
    fn default() -> Self {
        Self::Unset
    }
}

/// Resolve semantic and legacy policy inputs without broadening access.
pub fn resolve_network_policy<I, S>(
    policy: Option<NetworkPolicy>,
    mode: Option<&str>,
    allowed_hosts: I,
    legacy_network: LegacyNetwork,
) -> Result<NetworkPolicy, NetworkPolicyError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let supplied_hosts: Vec<String> = allowed_hosts
        .into_iter()
        .map(|host| host.as_ref().to_owned())
        .collect();
    let (requested, has_new_input) = if let Some(policy) = policy {
        if mode.is_some() || !supplied_hosts.is_empty() {
            return Err(NetworkPolicyError::new(
                "policy cannot be combined with separate mode or allowed_hosts inputs",
            ));
        }
        (policy, true)
    } else if let Some(mode) = mode {
        (NetworkPolicy::parse(mode, &supplied_hosts)?, true)
    } else if !supplied_hosts.is_empty() {
        return Err(NetworkPolicyError::new(
            "allowed_hosts requires an explicit allowlist mode",
        ));
    } else {
        (NetworkPolicy::default(), false)
    };

    let LegacyNetwork::Explicit(legacy_network) = legacy_network else {
        return Ok(requested);
    };
    let legacy = NetworkPolicy::from_legacy_network(legacy_network.as_deref())?;
    if !has_new_input {
        return Ok(legacy);
    }
    if requested.to_comparability_payload() != legacy.to_comparability_payload() {
        return Err(NetworkPolicyError::new(format!(
            "conflicting network configuration: new policy is {:?}, but legacy \
             network maps to {:?}",
            requested.effective_mode().as_str(),
            legacy.effective_mode().as_str(),
        )));
    }
    Ok(requested)
}

/// CLI resolver retaining named legacy Docker networks as an explicit escape hatch.
pub fn resolve_cli_network_policy<I, S>(
    mode: Option<&str>,
    allowed_hosts: I,
    legacy_network: Option<&str>,
) -> Result<Option<NetworkPolicy>, NetworkPolicyError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let hosts: Vec<String> = allowed_hosts
        .into_iter()
        .map(|host| host.as_ref().to_owned())
        .collect();
    if !matches!(legacy_network, None | Some("none")) {
        if mode.is_some() || !hosts.is_empty() {
            return Err(NetworkPolicyError::new(
                "named legacy --network cannot be combined with \
                 --network-policy or --allowed-host",
            ));
        }
        return Ok(None);
    }
    let legacy = match legacy_network {
        Some(network) => LegacyNetwork::Explicit(Some(network.to_owned())),
        None => LegacyNetwork::Unset,
    };
    resolve_network_policy(None, mode, &hosts, legacy).map(Some)
}

fn hex_digest(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(HEX[(byte >> 4) as usize] as char);
        output.push(HEX[(byte & 0x0f) as usize] as char);
    }
    output
}

// Small dependency-free SHA-256 implementation. Keeping the digest here avoids
// adding a second cryptography crate solely for a stable policy identity.
fn sha256(input: &[u8]) -> [u8; 32] {
    const INITIAL: [u32; 8] = [
        0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab,
        0x5be0cd19,
    ];
    const ROUND: [u32; 64] = [
        0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
        0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
        0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
        0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
        0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
        0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
        0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
        0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
        0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
        0xc67178f2,
    ];

    let bit_length = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_length.to_be_bytes());

    let mut state = INITIAL;
    for chunk in padded.chunks_exact(64) {
        let mut words = [0u32; 64];
        for (index, bytes) in chunk.chunks_exact(4).enumerate() {
            words[index] = u32::from_be_bytes(bytes.try_into().expect("four-byte word"));
        }
        for index in 16..64 {
            let s0 = words[index - 15].rotate_right(7)
                ^ words[index - 15].rotate_right(18)
                ^ (words[index - 15] >> 3);
            let s1 = words[index - 2].rotate_right(17)
                ^ words[index - 2].rotate_right(19)
                ^ (words[index - 2] >> 10);
            words[index] = words[index - 16]
                .wrapping_add(s0)
                .wrapping_add(words[index - 7])
                .wrapping_add(s1);
        }
        let [mut a, mut b, mut c, mut d, mut e, mut f, mut g, mut h] = state;
        for index in 0..64 {
            let sum1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let choose = (e & f) ^ ((!e) & g);
            let temporary1 = h
                .wrapping_add(sum1)
                .wrapping_add(choose)
                .wrapping_add(ROUND[index])
                .wrapping_add(words[index]);
            let sum0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let majority = (a & b) ^ (a & c) ^ (b & c);
            let temporary2 = sum0.wrapping_add(majority);
            h = g;
            g = f;
            f = e;
            e = d.wrapping_add(temporary1);
            d = c;
            c = b;
            b = a;
            a = temporary1.wrapping_add(temporary2);
        }
        for (slot, value) in state.iter_mut().zip([a, b, c, d, e, f, g, h]) {
            *slot = slot.wrapping_add(value);
        }
    }
    let mut output = [0u8; 32];
    for (chunk, value) in output.chunks_exact_mut(4).zip(state) {
        chunk.copy_from_slice(&value.to_be_bytes());
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hosts(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn modes_and_empty_allowlist_effective_behavior() {
        assert_eq!(
            NetworkPolicy::default().effective_mode(),
            NetworkMode::Public
        );
        assert_eq!(
            NetworkPolicy::parse("no-network", Vec::<String>::new())
                .unwrap()
                .effective_mode(),
            NetworkMode::NoNetwork
        );
        let empty = NetworkPolicy::parse("allowlist", Vec::<String>::new()).unwrap();
        assert_eq!(empty.mode, NetworkMode::Allowlist);
        assert_eq!(empty.effective_mode(), NetworkMode::NoNetwork);
        assert!(empty.allowed_hosts.is_empty());
        assert_eq!(empty.to_report_payload()["requested_mode"], "allowlist");
        assert_eq!(empty.to_report_payload()["effective_mode"], "no-network");
    }

    #[test]
    fn allowlist_is_lowercase_sorted_and_deduplicated() {
        let policy = NetworkPolicy::new(
            NetworkMode::Allowlist,
            hosts(&[
                "PyPI.org",
                "files.PYTHONHOSTED.org",
                "pypi.ORG",
                "allowed.test",
            ]),
        )
        .unwrap();
        assert_eq!(
            policy.allowed_hosts,
            hosts(&["allowed.test", "files.pythonhosted.org", "pypi.org"])
        );
    }

    #[test]
    fn invalid_or_ambiguous_allowed_hosts_are_rejected() {
        for host in [
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
        ] {
            assert!(normalize_allowed_host(host).is_err(), "accepted {host:?}");
        }
    }

    #[test]
    fn dns_length_limits_are_enforced() {
        let valid = format!("{}.example", "a".repeat(63));
        assert_eq!(normalize_allowed_host(&valid).unwrap(), valid);
        let too_long = format!(
            "{}.{}.{}.{}",
            "a".repeat(63),
            "b".repeat(63),
            "c".repeat(63),
            "d".repeat(62)
        );
        assert!(normalize_allowed_host(&too_long)
            .unwrap_err()
            .to_string()
            .contains("DNS hostname limit"));
        assert!(
            normalize_allowed_host(&format!("{}.example", "a".repeat(64)))
                .unwrap_err()
                .to_string()
                .contains("exact ASCII DNS")
        );
    }

    #[test]
    fn ordinary_two_label_test_name_is_valid() {
        assert_eq!(
            normalize_allowed_host("Allowed.TEST").unwrap(),
            "allowed.test"
        );
    }

    #[test]
    fn allowed_hosts_are_rejected_outside_allowlist() {
        for mode in [NetworkMode::Public, NetworkMode::NoNetwork] {
            assert!(NetworkPolicy::new(mode, hosts(&["example.com"]))
                .unwrap_err()
                .to_string()
                .contains("allowed hosts require"));
        }
        assert!(NetworkMode::from_str("private").is_err());
    }

    #[test]
    fn report_comparability_and_digests_are_canonical() {
        let first =
            NetworkPolicy::parse("allowlist", hosts(&["B.example", "a.example", "A.EXAMPLE"]))
                .unwrap();
        let second = NetworkPolicy::parse("allowlist", hosts(&["a.example", "b.example"])).unwrap();
        assert_eq!(
            first.to_report_payload(),
            json!({
                "schema_version": NETWORK_POLICY_SCHEMA_VERSION,
                "requested_mode": "allowlist",
                "effective_mode": "allowlist",
                "allowed_hosts": ["a.example", "b.example"],
                "host_matching": "exact-ascii-dns-case-insensitive",
                "includes_subdomains": false,
                "host_ports_supported": false,
            })
        );
        assert_eq!(
            first.to_comparability_payload(),
            second.to_comparability_payload()
        );
        assert_eq!(first.comparability_digest(), second.comparability_digest());
        assert_eq!(
            first.comparability_digest(),
            "fcd89295a0e9d96184765d1ee0f0600157e5afc7374c5f1c8183ba8549163ba9"
        );

        let empty = NetworkPolicy::parse("allowlist", Vec::<String>::new()).unwrap();
        let none = NetworkPolicy::parse("no-network", Vec::<String>::new()).unwrap();
        assert_eq!(
            empty.to_comparability_payload(),
            none.to_comparability_payload()
        );
        assert_eq!(empty.comparability_digest(), none.comparability_digest());
        assert_eq!(
            none.comparability_digest(),
            "7871707a495b1e6ebe18e414c11fe495cb2fce19cf9b9422ee0f29a9d03c2809"
        );
    }

    #[test]
    fn legacy_mapping_and_named_network_rejection() {
        assert_eq!(
            NetworkPolicy::from_legacy_network(None)
                .unwrap()
                .effective_mode(),
            NetworkMode::Public
        );
        assert_eq!(
            NetworkPolicy::from_legacy_network(Some("none"))
                .unwrap()
                .effective_mode(),
            NetworkMode::NoNetwork
        );
        assert!(NetworkPolicy::from_legacy_network(Some("bridge"))
            .unwrap_err()
            .to_string()
            .contains("named Docker networks"));
    }

    #[test]
    fn resolver_distinguishes_omitted_legacy_from_public() {
        let allowlist = resolve_network_policy(
            None,
            Some("allowlist"),
            hosts(&["example.com"]),
            LegacyNetwork::Unset,
        )
        .unwrap();
        assert_eq!(allowlist.effective_mode(), NetworkMode::Allowlist);
        assert!(resolve_network_policy(
            None,
            Some("allowlist"),
            hosts(&["example.com"]),
            LegacyNetwork::Explicit(None),
        )
        .unwrap_err()
        .to_string()
        .contains("conflicting network configuration"));
    }

    #[test]
    fn resolver_allows_equivalent_inputs_and_rejects_conflicts() {
        let explicit = NetworkPolicy::parse("no-network", Vec::<String>::new()).unwrap();
        assert_eq!(
            resolve_network_policy(
                Some(explicit.clone()),
                None,
                Vec::<String>::new(),
                LegacyNetwork::Explicit(Some("none".to_owned())),
            )
            .unwrap(),
            explicit
        );
        let empty = resolve_network_policy(
            None,
            Some("allowlist"),
            Vec::<String>::new(),
            LegacyNetwork::Explicit(Some("none".to_owned())),
        )
        .unwrap();
        assert_eq!(empty.mode, NetworkMode::Allowlist);
        assert_eq!(empty.effective_mode(), NetworkMode::NoNetwork);
        assert!(resolve_network_policy(
            None,
            Some("no-network"),
            Vec::<String>::new(),
            LegacyNetwork::Explicit(None),
        )
        .is_err());
    }

    #[test]
    fn resolver_rejects_ambiguous_new_inputs() {
        let policy = NetworkPolicy::parse("allowlist", hosts(&["example.com"])).unwrap();
        assert!(resolve_network_policy(
            Some(policy),
            Some("allowlist"),
            Vec::<String>::new(),
            LegacyNetwork::Unset,
        )
        .unwrap_err()
        .to_string()
        .contains("separate mode"));
        assert!(
            resolve_network_policy(None, None, hosts(&["example.com"]), LegacyNetwork::Unset,)
                .unwrap_err()
                .to_string()
                .contains("explicit allowlist mode")
        );
    }

    #[test]
    fn cli_resolver_preserves_named_escape_hatch_and_deny_all_forms() {
        assert_eq!(
            resolve_cli_network_policy(None, Vec::<String>::new(), Some("existing-test-network"))
                .unwrap(),
            None
        );
        assert!(resolve_cli_network_policy(
            Some("allowlist"),
            hosts(&["example.com"]),
            Some("existing-test-network"),
        )
        .is_err());
        let legacy = resolve_cli_network_policy(None, Vec::<String>::new(), Some("none"))
            .unwrap()
            .unwrap();
        let typed = resolve_cli_network_policy(Some("no-network"), Vec::<String>::new(), None)
            .unwrap()
            .unwrap();
        assert_eq!(
            legacy.to_comparability_payload(),
            typed.to_comparability_payload()
        );
    }

    #[test]
    fn sha256_matches_standard_vector() {
        assert_eq!(
            hex_digest(&sha256(b"abc")),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
