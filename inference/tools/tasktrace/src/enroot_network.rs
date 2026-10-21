// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native fail-closed network-namespace validation for Enroot launches.
//!
//! The controller starts this helper as the first process inside a fresh
//! `unshare --user --map-current-user --pid --fork --mount-proc --net`
//! namespace.  The helper proves that the network namespace changed while the
//! controller UID/GID mapping was preserved, retains one private evidence file,
//! and only then replaces itself with the requested taskset/Enroot command.

use serde::{Deserialize, Serialize};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::Command;

pub const ENROOT_NETWORK_EVIDENCE_SCHEMA_VERSION: &str = "tasktrace-enroot-network-enforcement.v1";
pub const ENROOT_NETWORK_BACKEND: &str = "linux-user-network-namespace-v1";
pub const INTERNAL_COMMAND: &str = "__enroot-network";
const MAX_EVIDENCE_BYTES: u64 = 64 * 1024;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnrootNetworkError(String);

impl EnrootNetworkError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for EnrootNetworkError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for EnrootNetworkError {}

impl From<io::Error> for EnrootNetworkError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for EnrootNetworkError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EnrootNetworkExpectation {
    pub parent_netns_inode: u64,
    pub controller_euid: u32,
    pub controller_egid: u32,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct EnrootNetworkEvidence {
    pub schema_version: String,
    pub state: String,
    pub backend: String,
    pub policy_effective_mode: String,
    pub valid: bool,
    pub parent_netns_inode: u64,
    pub netns_inode: u64,
    pub netns_differs: bool,
    pub controller_euid: u32,
    pub controller_egid: u32,
    pub namespace_euid: u32,
    pub namespace_egid: u32,
    pub interfaces: Vec<String>,
    pub ipv4_routes: Vec<String>,
    pub ipv6_addresses: Vec<String>,
}

fn netns_inode(path: &Path) -> Result<u64, EnrootNetworkError> {
    path.metadata()
        .map(|metadata| metadata.ino())
        .map_err(|error| {
            EnrootNetworkError::new(format!(
                "unable to inspect network namespace {}: {error}",
                path.display()
            ))
        })
}

pub fn controller_expectation() -> Result<EnrootNetworkExpectation, EnrootNetworkError> {
    Ok(EnrootNetworkExpectation {
        parent_netns_inode: netns_inode(Path::new("/proc/self/ns/net"))?,
        controller_euid: unsafe { libc::geteuid() },
        controller_egid: unsafe { libc::getegid() },
    })
}

fn interfaces(path: &Path) -> Result<Vec<String>, EnrootNetworkError> {
    let contents = fs::read_to_string(path).map_err(|error| {
        EnrootNetworkError::new(format!(
            "unable to inspect network interfaces {}: {error}",
            path.display()
        ))
    })?;
    let mut values = contents
        .lines()
        .skip(2)
        .filter_map(|line| line.split_once(':').map(|(name, _)| name.trim()))
        .filter(|name| !name.is_empty())
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    values.sort();
    values.dedup();
    Ok(values)
}

fn nonempty_data_lines(path: &Path, skip: usize) -> Result<Vec<String>, EnrootNetworkError> {
    let contents = fs::read_to_string(path).map_err(|error| {
        EnrootNetworkError::new(format!(
            "unable to inspect namespace network state {}: {error}",
            path.display()
        ))
    })?;
    Ok(contents
        .lines()
        .skip(skip)
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .map(ToOwned::to_owned)
        .collect())
}

/// `/proc/net/if_inet6` loopback address record on the `lo` interface.
///
/// A fresh network namespace has no addresses when the loopback interface is
/// left down (util-linux `unshare --net`), but bubblewrap's `--unshare-net`
/// brings `lo` up, which assigns `::1`. Both states are equally isolated:
/// only the loopback address on the loopback interface is permitted here.
const LOOPBACK_IPV6_ADDRESS: &str = "00000000000000000000000000000001";

pub fn loopback_only_ipv6(addresses: &[String]) -> bool {
    addresses.iter().all(|line| {
        let mut fields = line.split_whitespace();
        let address = fields.next().unwrap_or("");
        let interface = fields.last().unwrap_or("");
        address == LOOPBACK_IPV6_ADDRESS && interface == "lo"
    })
}

fn evidence_from_snapshot(
    expectation: EnrootNetworkExpectation,
    child_netns_inode: u64,
    namespace_euid: u32,
    namespace_egid: u32,
    interfaces: Vec<String>,
    ipv4_routes: Vec<String>,
    ipv6_addresses: Vec<String>,
) -> Result<EnrootNetworkEvidence, EnrootNetworkError> {
    if expectation.parent_netns_inode == 0
        || child_netns_inode == 0
        || child_netns_inode == expectation.parent_netns_inode
    {
        return Err(EnrootNetworkError::new(
            "Enroot no-network namespace did not change",
        ));
    }
    if namespace_euid != expectation.controller_euid
        || namespace_egid != expectation.controller_egid
    {
        return Err(EnrootNetworkError::new(format!(
            "Enroot no-network user namespace changed controller identity: expected {}:{}, got {}:{}",
            expectation.controller_euid,
            expectation.controller_egid,
            namespace_euid,
            namespace_egid
        )));
    }
    if interfaces != ["lo"] || !ipv4_routes.is_empty() || !loopback_only_ipv6(&ipv6_addresses) {
        return Err(EnrootNetworkError::new(format!(
            "Enroot no-network namespace is not isolated: interfaces={interfaces:?}, ipv4_routes={ipv4_routes:?}, ipv6_addresses={ipv6_addresses:?}"
        )));
    }
    Ok(EnrootNetworkEvidence {
        schema_version: ENROOT_NETWORK_EVIDENCE_SCHEMA_VERSION.to_owned(),
        state: "enforced".to_owned(),
        backend: ENROOT_NETWORK_BACKEND.to_owned(),
        policy_effective_mode: "no-network".to_owned(),
        valid: true,
        parent_netns_inode: expectation.parent_netns_inode,
        netns_inode: child_netns_inode,
        netns_differs: true,
        controller_euid: expectation.controller_euid,
        controller_egid: expectation.controller_egid,
        namespace_euid,
        namespace_egid,
        interfaces,
        ipv4_routes,
        ipv6_addresses,
    })
}

pub fn inspect_current_namespace(
    expectation: EnrootNetworkExpectation,
) -> Result<EnrootNetworkEvidence, EnrootNetworkError> {
    evidence_from_snapshot(
        expectation,
        netns_inode(Path::new("/proc/self/ns/net"))?,
        unsafe { libc::geteuid() },
        unsafe { libc::getegid() },
        interfaces(Path::new("/proc/net/dev"))?,
        nonempty_data_lines(Path::new("/proc/net/route"), 1)?,
        nonempty_data_lines(Path::new("/proc/net/if_inet6"), 0)?,
    )
}

pub fn validate_evidence(
    evidence: &EnrootNetworkEvidence,
    expectation: EnrootNetworkExpectation,
) -> Result<(), EnrootNetworkError> {
    let expected = evidence_from_snapshot(
        expectation,
        evidence.netns_inode,
        evidence.namespace_euid,
        evidence.namespace_egid,
        evidence.interfaces.clone(),
        evidence.ipv4_routes.clone(),
        evidence.ipv6_addresses.clone(),
    )?;
    if evidence != &expected {
        return Err(EnrootNetworkError::new(
            "retained Enroot no-network evidence is invalid",
        ));
    }
    Ok(())
}

fn validate_private_parent(path: &Path, expected_uid: u32) -> Result<&Path, EnrootNetworkError> {
    if !path.is_absolute() {
        return Err(EnrootNetworkError::new(
            "Enroot network evidence path must be absolute",
        ));
    }
    let parent = path.parent().ok_or_else(|| {
        EnrootNetworkError::new("Enroot network evidence path has no parent directory")
    })?;
    let metadata = fs::symlink_metadata(parent).map_err(|error| {
        EnrootNetworkError::new(format!(
            "unable to inspect Enroot network evidence parent {}: {error}",
            parent.display()
        ))
    })?;
    if !metadata.is_dir() || metadata.file_type().is_symlink() {
        return Err(EnrootNetworkError::new(
            "Enroot network evidence parent must be a real directory",
        ));
    }
    if metadata.uid() != expected_uid || metadata.mode() & 0o077 != 0 {
        return Err(EnrootNetworkError::new(format!(
            "Enroot network evidence parent must be controller-owned and private: {}",
            parent.display()
        )));
    }
    Ok(parent)
}

pub fn validate_evidence_destination(
    path: &Path,
    expected_uid: u32,
) -> Result<(), EnrootNetworkError> {
    validate_private_parent(path, expected_uid)?;
    match fs::symlink_metadata(path) {
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(EnrootNetworkError::new(format!(
            "unable to inspect Enroot network evidence destination {}: {error}",
            path.display()
        ))),
        Ok(_) => Err(EnrootNetworkError::new(format!(
            "Enroot network evidence destination already exists: {}",
            path.display()
        ))),
    }
}

fn write_evidence(path: &Path, evidence: &EnrootNetworkEvidence) -> Result<(), EnrootNetworkError> {
    validate_evidence_destination(path, evidence.controller_euid)?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|error| {
            EnrootNetworkError::new(format!(
                "unable to create Enroot network evidence {}: {error}",
                path.display()
            ))
        })?;
    serde_json::to_writer(&mut file, evidence)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    Ok(())
}

pub fn read_and_validate_evidence(
    path: &Path,
    expectation: EnrootNetworkExpectation,
) -> Result<EnrootNetworkEvidence, EnrootNetworkError> {
    validate_private_parent(path, expectation.controller_euid)?;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|error| {
            EnrootNetworkError::new(format!(
                "unable to read Enroot network evidence {}: {error}",
                path.display()
            ))
        })?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.nlink() != 1
        || metadata.uid() != expectation.controller_euid
        || metadata.mode() & 0o077 != 0
        || metadata.len() > MAX_EVIDENCE_BYTES
    {
        return Err(EnrootNetworkError::new(
            "Enroot network evidence file metadata is invalid",
        ));
    }
    let mut payload = Vec::with_capacity(metadata.len() as usize);
    file.take(MAX_EVIDENCE_BYTES + 1)
        .read_to_end(&mut payload)?;
    if payload.len() as u64 > MAX_EVIDENCE_BYTES {
        return Err(EnrootNetworkError::new(
            "Enroot network evidence exceeds the size limit",
        ));
    }
    let evidence: EnrootNetworkEvidence = serde_json::from_slice(&payload)?;
    validate_evidence(&evidence, expectation)?;
    Ok(evidence)
}

#[derive(Debug)]
struct InternalArguments {
    expectation: EnrootNetworkExpectation,
    evidence_path: PathBuf,
    command: Vec<OsString>,
}

fn required_value(
    arguments: &[OsString],
    index: &mut usize,
    option: &str,
) -> Result<OsString, EnrootNetworkError> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| EnrootNetworkError::new(format!("{option} requires a value")))
}

fn parse_u64(value: &OsStr, option: &str) -> Result<u64, EnrootNetworkError> {
    value
        .to_str()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| EnrootNetworkError::new(format!("{option} requires an unsigned integer")))
}

fn parse_u32(value: &OsStr, option: &str) -> Result<u32, EnrootNetworkError> {
    value
        .to_str()
        .and_then(|value| value.parse().ok())
        .ok_or_else(|| EnrootNetworkError::new(format!("{option} requires an unsigned integer")))
}

fn parse_internal_arguments(
    arguments: &[OsString],
) -> Result<InternalArguments, EnrootNetworkError> {
    let mut parent_netns_inode = None;
    let mut controller_euid = None;
    let mut controller_egid = None;
    let mut evidence_path = None;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if argument == "--" {
            let command = arguments[index + 1..].to_vec();
            if command.is_empty() {
                return Err(EnrootNetworkError::new(
                    "Enroot network helper requires a command after --",
                ));
            }
            return Ok(InternalArguments {
                expectation: EnrootNetworkExpectation {
                    parent_netns_inode: parent_netns_inode.ok_or_else(|| {
                        EnrootNetworkError::new("--parent-netns-inode is required")
                    })?,
                    controller_euid: controller_euid
                        .ok_or_else(|| EnrootNetworkError::new("--controller-euid is required"))?,
                    controller_egid: controller_egid
                        .ok_or_else(|| EnrootNetworkError::new("--controller-egid is required"))?,
                },
                evidence_path: evidence_path
                    .ok_or_else(|| EnrootNetworkError::new("--evidence-path is required"))?,
                command,
            });
        }
        match argument.to_str() {
            Some("--parent-netns-inode") => {
                let value = required_value(arguments, &mut index, "--parent-netns-inode")?;
                parent_netns_inode = Some(parse_u64(&value, "--parent-netns-inode")?);
            }
            Some("--controller-euid") => {
                let value = required_value(arguments, &mut index, "--controller-euid")?;
                controller_euid = Some(parse_u32(&value, "--controller-euid")?);
            }
            Some("--controller-egid") => {
                let value = required_value(arguments, &mut index, "--controller-egid")?;
                controller_egid = Some(parse_u32(&value, "--controller-egid")?);
            }
            Some("--evidence-path") => {
                evidence_path = Some(PathBuf::from(required_value(
                    arguments,
                    &mut index,
                    "--evidence-path",
                )?));
            }
            _ => {
                return Err(EnrootNetworkError::new(format!(
                    "unknown Enroot network helper option: {argument:?}"
                )))
            }
        }
        index += 1;
    }
    Err(EnrootNetworkError::new(
        "Enroot network helper requires -- before its command",
    ))
}

fn execute_internal(arguments: &[OsString]) -> Result<(), EnrootNetworkError> {
    let parsed = parse_internal_arguments(arguments)?;
    let evidence = inspect_current_namespace(parsed.expectation)?;
    write_evidence(&parsed.evidence_path, &evidence)?;
    let (program, arguments) = parsed
        .command
        .split_first()
        .expect("internal command was validated as nonempty");
    let error = Command::new(program).args(arguments).exec();
    Err(EnrootNetworkError::new(format!(
        "unable to exec Enroot network command {program:?}: {error}"
    )))
}

/// Hidden native dispatch used only as the first process inside `unshare`.
pub fn internal_main(arguments: &[OsString]) -> i32 {
    match execute_internal(arguments) {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("Enroot no-network failed: {error}");
            125
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn expectation() -> EnrootNetworkExpectation {
        EnrootNetworkExpectation {
            parent_netns_inode: 101,
            controller_euid: unsafe { libc::geteuid() },
            controller_egid: unsafe { libc::getegid() },
        }
    }

    #[test]
    fn isolated_snapshot_is_exact_and_routes_fail_closed() {
        let expected = expectation();
        let evidence = evidence_from_snapshot(
            expected,
            202,
            expected.controller_euid,
            expected.controller_egid,
            vec!["lo".to_owned()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        validate_evidence(&evidence, expected).unwrap();

        let error = evidence_from_snapshot(
            expected,
            202,
            expected.controller_euid,
            expected.controller_egid,
            vec!["lo".to_owned()],
            vec!["00000000 route".to_owned()],
            Vec::new(),
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("not isolated"), "{error}");
    }

    #[test]
    fn loopback_only_ipv6_is_isolated_and_other_addresses_fail_closed() {
        let expected = expectation();
        // bubblewrap --unshare-net brings lo up, assigning exactly ::1.
        evidence_from_snapshot(
            expected,
            202,
            expected.controller_euid,
            expected.controller_egid,
            vec!["lo".to_owned()],
            Vec::new(),
            vec!["00000000000000000000000000000001 01 80 10 80       lo".to_owned()],
        )
        .unwrap();

        for line in [
            // Global address on lo.
            "20010db8000000000000000000000001 01 40 00 80       lo",
            // Loopback address on a non-loopback interface name.
            "00000000000000000000000000000001 02 80 10 80     eth0",
        ] {
            let error = evidence_from_snapshot(
                expected,
                202,
                expected.controller_euid,
                expected.controller_egid,
                vec!["lo".to_owned()],
                Vec::new(),
                vec![line.to_owned()],
            )
            .unwrap_err()
            .to_string();
            assert!(error.contains("not isolated"), "{error}");
        }
    }

    #[test]
    fn private_evidence_round_trip_rejects_mutation() {
        let temporary = tempfile::tempdir().unwrap();
        fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let path = temporary.path().join("network.json");
        let expected = expectation();
        let evidence = evidence_from_snapshot(
            expected,
            202,
            expected.controller_euid,
            expected.controller_egid,
            vec!["lo".to_owned()],
            Vec::new(),
            Vec::new(),
        )
        .unwrap();
        write_evidence(&path, &evidence).unwrap();
        assert_eq!(
            read_and_validate_evidence(&path, expected).unwrap(),
            evidence
        );

        let mut payload: EnrootNetworkEvidence =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        payload.valid = false;
        fs::write(&path, serde_json::to_vec(&payload).unwrap()).unwrap();
        assert!(read_and_validate_evidence(&path, expected)
            .unwrap_err()
            .to_string()
            .contains("invalid"));
    }

    #[test]
    fn hidden_arguments_preserve_the_exec_vector() {
        let parsed = parse_internal_arguments(&[
            "--parent-netns-inode".into(),
            "101".into(),
            "--controller-euid".into(),
            "1000".into(),
            "--controller-egid".into(),
            "1001".into(),
            "--evidence-path".into(),
            "/private/evidence.json".into(),
            "--".into(),
            "taskset".into(),
            "-c".into(),
            "4".into(),
            "enroot".into(),
            "start".into(),
        ])
        .unwrap();
        assert_eq!(parsed.expectation.parent_netns_inode, 101);
        assert_eq!(parsed.evidence_path, Path::new("/private/evidence.json"));
        assert_eq!(
            parsed.command,
            ["taskset", "-c", "4", "enroot", "start"]
                .into_iter()
                .map(OsString::from)
                .collect::<Vec<_>>()
        );
    }
}
