// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Content-bound Enroot runtime provenance for ordered native runs.
//!
//! Capture is intentionally contextual: `enroot info` runs through `taskset`
//! on the reserved controller CPUs and on every distinct launch affinity. The
//! paths that select Enroot's implementation, configuration, and mutable state
//! must not change with affinity. Callers capture once before preflight and use
//! [`recapture_and_validate`] after all waves to fail closed on runtime drift.

use serde::Serialize;
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{CString, OsStr};
use std::fmt;
use std::fs::{self, OpenOptions};
use std::io::{self, Read};
use std::mem;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::io::AsRawFd;
use std::os::unix::process::CommandExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const ENROOT_PROVENANCE_SCHEMA_VERSION: &str = "tasktrace-enroot-runtime.v1";
const PACKAGED_ENROOT_ROOT: &str = "/usr/share/enroot";

const INSPECT_OUTPUT_LIMIT_BYTES: usize = 1024 * 1024;
const INSPECT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const INSPECT_CLEANUP_TIMEOUT: Duration = Duration::from_secs(1);
const INSPECT_PIPE_DRAIN_TIMEOUT: Duration = Duration::from_millis(250);

const REQUIRED_STABLE_SETTINGS: &[&str] = &[
    "ENROOT_LIBRARY_PATH",
    "ENROOT_SYSCONF_PATH",
    "ENROOT_CONFIG_PATH",
    "ENROOT_CACHE_PATH",
    "ENROOT_DATA_PATH",
    "ENROOT_RUNTIME_PATH",
    "ENROOT_TEMP_PATH",
];

const STATE_PATH_SETTINGS: &[&str] = &[
    "ENROOT_CACHE_PATH",
    "ENROOT_DATA_PATH",
    "ENROOT_RUNTIME_PATH",
    "ENROOT_TEMP_PATH",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnrootProvenanceError(String);

impl EnrootProvenanceError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for EnrootProvenanceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for EnrootProvenanceError {}

impl From<io::Error> for EnrootProvenanceError {
    fn from(error: io::Error) -> Self {
        Self::new(error.to_string())
    }
}

/// Executables and CPU contexts used to capture one Enroot runtime identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnrootProvenanceConfig {
    pub controller_cpus: String,
    pub launch_cpu_sets: Vec<String>,
    pub enroot: String,
    pub taskset: String,
    pub unshare: String,
    pub unsquashfs: String,
    pub inspect_timeout: Duration,
    pub mountinfo_path: PathBuf,
}

impl EnrootProvenanceConfig {
    pub fn new(controller_cpus: impl Into<String>, launch_cpu_sets: Vec<String>) -> Self {
        Self {
            controller_cpus: controller_cpus.into(),
            launch_cpu_sets,
            enroot: "enroot".to_owned(),
            taskset: "taskset".to_owned(),
            unshare: "unshare".to_owned(),
            unsquashfs: "unsquashfs".to_owned(),
            inspect_timeout: Duration::from_secs(10),
            mountinfo_path: PathBuf::from("/proc/self/mountinfo"),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EffectiveConfigIdentity {
    pub cpus: String,
    pub sha256: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EffectiveConfigIdentities {
    pub controller: EffectiveConfigIdentity,
    pub launches: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct StatePathIdentity {
    pub path: String,
    pub filesystem_type: Option<String>,
    pub device_major: u64,
    pub device_minor: u64,
    pub filesystem_id: String,
}

/// Secret-safe, deterministic identity used in run comparison and validation.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnrootRuntimeProvenance {
    pub schema_version: String,
    pub version: String,
    pub executable: String,
    /// Resolved path of the namespace-containment executable that launches
    /// actually use (util-linux unshare, or bubblewrap on hosts where
    /// unprivileged unshare cannot create user namespaces).
    pub containment_executable: String,
    pub runtime_content_sha256: String,
    pub effective_config_sha256: EffectiveConfigIdentities,
    pub system_config_sha256: Option<String>,
    pub user_config_sha256: Option<String>,
    pub state_paths: BTreeMap<String, StatePathIdentity>,
    pub comparability_sha256: String,
}

#[derive(Debug)]
struct CommandOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

fn set_nonblocking(pipe: &impl AsRawFd) -> io::Result<()> {
    let descriptor = pipe.as_raw_fd();
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFL) };
    if flags < 0 {
        return Err(io::Error::last_os_error());
    }
    if unsafe { libc::fcntl(descriptor, libc::F_SETFL, flags | libc::O_NONBLOCK) } < 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(())
}

fn drain_pipe<R: Read>(
    pipe: &mut Option<R>,
    bytes: &mut Vec<u8>,
    limit: usize,
) -> io::Result<bool> {
    let Some(reader) = pipe.as_mut() else {
        return Ok(false);
    };
    let mut block = [0_u8; 16 * 1024];
    loop {
        match reader.read(&mut block) {
            Ok(0) => {
                *pipe = None;
                return Ok(false);
            }
            Ok(count) => {
                let remaining = limit.saturating_sub(bytes.len());
                bytes.extend_from_slice(&block[..count.min(remaining)]);
                if count > remaining {
                    return Ok(true);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => return Ok(false),
            Err(error) => return Err(error),
        }
    }
}

fn signal_process_group(process_group: u32, signal: libc::c_int) -> io::Result<bool> {
    let process_group = i32::try_from(process_group)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid process group"))?;
    if unsafe { libc::kill(-process_group, signal) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(false)
    } else {
        Err(error)
    }
}

fn process_group_exists(process_group: u32) -> io::Result<bool> {
    let process_group = i32::try_from(process_group)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "invalid process group"))?;
    if unsafe { libc::kill(-process_group, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(error),
    }
}

fn kill_reap_and_verify_process_group(
    child: &mut Child,
    process_group: u32,
    mut leader_reaped: bool,
) -> Result<(), EnrootProvenanceError> {
    let mut cleanup_errors = Vec::new();
    if let Err(error) = signal_process_group(process_group, libc::SIGKILL) {
        cleanup_errors.push(format!(
            "unable to kill provenance inspection process group: {error}"
        ));
    }
    if !leader_reaped {
        match child.try_wait() {
            Ok(Some(_)) => leader_reaped = true,
            Ok(None) => {
                if let Err(error) = child.kill() {
                    cleanup_errors.push(format!(
                        "unable to kill provenance inspection leader: {error}"
                    ));
                }
            }
            Err(error) => cleanup_errors.push(format!(
                "unable to inspect provenance inspection leader before cleanup: {error}"
            )),
        }
    }
    let deadline = Instant::now() + INSPECT_CLEANUP_TIMEOUT;
    loop {
        if !leader_reaped {
            match child.try_wait() {
                Ok(Some(_)) => leader_reaped = true,
                Ok(None) => {}
                Err(error) => {
                    return Err(EnrootProvenanceError::new(format!(
                        "unable to reap provenance inspection leader: {error}"
                    )))
                }
            }
        }
        let group_gone = !process_group_exists(process_group).map_err(|error| {
            EnrootProvenanceError::new(format!(
                "unable to verify provenance inspection process group cleanup: {error}; {}",
                cleanup_errors.join("; ")
            ))
        })?;
        if leader_reaped && group_gone {
            return if cleanup_errors.is_empty() {
                Ok(())
            } else {
                Err(EnrootProvenanceError::new(cleanup_errors.join("; ")))
            };
        }
        if Instant::now() >= deadline {
            cleanup_errors.push(format!(
                "bounded SIGKILL cleanup ended with leader_reaped={leader_reaped}, process_group_gone={group_gone}"
            ));
            return Err(EnrootProvenanceError::new(cleanup_errors.join("; ")));
        }
        thread::sleep(INSPECT_POLL_INTERVAL);
    }
}

fn finish_failed_command(
    child: &mut Child,
    process_group: u32,
    leader_reaped: bool,
    reason: String,
) -> EnrootProvenanceError {
    match kill_reap_and_verify_process_group(child, process_group, leader_reaped) {
        Ok(()) => EnrootProvenanceError::new(reason),
        Err(cleanup_error) => {
            EnrootProvenanceError::new(format!("{reason}; cleanup failed: {cleanup_error}"))
        }
    }
}

fn drain_pipes_after_exit<R: Read, E: Read>(
    stdout: &mut Option<R>,
    stderr: &mut Option<E>,
    stdout_bytes: &mut Vec<u8>,
    stderr_bytes: &mut Vec<u8>,
) -> Result<(), EnrootProvenanceError> {
    let deadline = Instant::now() + INSPECT_PIPE_DRAIN_TIMEOUT;
    loop {
        if drain_pipe(stdout, stdout_bytes, INSPECT_OUTPUT_LIMIT_BYTES)? {
            return Err(EnrootProvenanceError::new(
                "provenance inspection stdout exceeded the capture limit",
            ));
        }
        if drain_pipe(stderr, stderr_bytes, INSPECT_OUTPUT_LIMIT_BYTES)? {
            return Err(EnrootProvenanceError::new(
                "provenance inspection stderr exceeded the capture limit",
            ));
        }
        if stdout.is_none() && stderr.is_none() {
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err(EnrootProvenanceError::new(
                "provenance inspection pipes remained open after process-group cleanup",
            ));
        }
        thread::sleep(INSPECT_POLL_INTERVAL);
    }
}

fn command_output_with_timeout(
    command: &mut Command,
    timeout: Duration,
    label: &str,
) -> Result<CommandOutput, EnrootProvenanceError> {
    command.stdin(Stdio::null());
    command.process_group(0);
    let mut child = command
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| EnrootProvenanceError::new(format!("unable to start {label}: {error}")))?;
    let process_group = child.id();
    let mut stdout =
        Some(child.stdout.take().ok_or_else(|| {
            EnrootProvenanceError::new(format!("unable to capture {label} stdout"))
        })?);
    let mut stderr =
        Some(child.stderr.take().ok_or_else(|| {
            EnrootProvenanceError::new(format!("unable to capture {label} stderr"))
        })?);
    if let Err(error) = set_nonblocking(stdout.as_ref().expect("stdout is present"))
        .and_then(|()| set_nonblocking(stderr.as_ref().expect("stderr is present")))
    {
        return Err(finish_failed_command(
            &mut child,
            process_group,
            false,
            format!("unable to configure bounded {label} output capture: {error}"),
        ));
    }
    let mut stdout_bytes = Vec::new();
    let mut stderr_bytes = Vec::new();
    let started = Instant::now();
    let status = loop {
        if drain_pipe(&mut stdout, &mut stdout_bytes, INSPECT_OUTPUT_LIMIT_BYTES).map_err(
            |error| {
                finish_failed_command(
                    &mut child,
                    process_group,
                    false,
                    format!("unable to capture {label} stdout: {error}"),
                )
            },
        )? {
            return Err(finish_failed_command(
                &mut child,
                process_group,
                false,
                format!(
                    "{label} stdout exceeded the {} byte capture limit",
                    INSPECT_OUTPUT_LIMIT_BYTES
                ),
            ));
        }
        if drain_pipe(&mut stderr, &mut stderr_bytes, INSPECT_OUTPUT_LIMIT_BYTES).map_err(
            |error| {
                finish_failed_command(
                    &mut child,
                    process_group,
                    false,
                    format!("unable to capture {label} stderr: {error}"),
                )
            },
        )? {
            return Err(finish_failed_command(
                &mut child,
                process_group,
                false,
                format!(
                    "{label} stderr exceeded the {} byte capture limit",
                    INSPECT_OUTPUT_LIMIT_BYTES
                ),
            ));
        }
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if started.elapsed() < timeout => {
                thread::sleep(INSPECT_POLL_INTERVAL);
            }
            Ok(None) => {
                return Err(finish_failed_command(
                    &mut child,
                    process_group,
                    false,
                    format!(
                        "{label} timed out after {:.3} seconds",
                        timeout.as_secs_f64()
                    ),
                ));
            }
            Err(error) => {
                return Err(finish_failed_command(
                    &mut child,
                    process_group,
                    false,
                    format!("unable to wait for {label}: {error}"),
                ));
            }
        }
    };
    match process_group_exists(process_group) {
        Ok(true) => {
            return Err(finish_failed_command(
                &mut child,
                process_group,
                true,
                format!("{label} left background descendants after its leader exited"),
            ))
        }
        Ok(false) => {}
        Err(error) => {
            return Err(finish_failed_command(
                &mut child,
                process_group,
                true,
                format!("unable to verify {label} process group after leader exit: {error}"),
            ))
        }
    }
    drain_pipes_after_exit(
        &mut stdout,
        &mut stderr,
        &mut stdout_bytes,
        &mut stderr_bytes,
    )?;
    Ok(CommandOutput {
        status,
        stdout: stdout_bytes,
        stderr: stderr_bytes,
    })
}

fn resolve_executable(name: &str) -> Result<PathBuf, EnrootProvenanceError> {
    if name.is_empty() {
        return Err(EnrootProvenanceError::new(
            "runtime executable name must not be empty",
        ));
    }
    let candidates = if name.contains('/') {
        vec![PathBuf::from(name)]
    } else {
        env::var_os("PATH")
            .map(|path| {
                env::split_paths(&path)
                    .map(|root| root.join(name))
                    .collect()
            })
            .unwrap_or_else(Vec::new)
    };
    for candidate in candidates {
        let Ok(metadata) = fs::metadata(&candidate) else {
            continue;
        };
        if metadata.is_file() && metadata.permissions().mode() & 0o111 != 0 {
            return fs::canonicalize(&candidate).map_err(|error| {
                EnrootProvenanceError::new(format!(
                    "unable to resolve runtime executable {}: {error}",
                    candidate.display()
                ))
            });
        }
    }
    Err(EnrootProvenanceError::new(format!(
        "runtime executable not found or not executable: {name}"
    )))
}

fn validate_cpu_set(value: &str, field: &str) -> Result<(), EnrootProvenanceError> {
    if value.is_empty() || value.trim() != value {
        return Err(EnrootProvenanceError::new(format!(
            "{field} must be a non-empty trimmed CPU set"
        )));
    }
    Ok(())
}

fn inspect(
    taskset: &Path,
    enroot: &Path,
    cpus: &str,
    subcommand: &str,
    timeout: Duration,
) -> Result<String, EnrootProvenanceError> {
    let label = format!("Enroot {subcommand} inspection on CPUs {cpus}");
    let output = command_output_with_timeout(
        Command::new(taskset)
            .arg("-c")
            .arg(cpus)
            .arg(enroot)
            .arg(subcommand)
            .env("LC_ALL", "C"),
        timeout,
        &label,
    )?;
    if !output.status.success() {
        let stderr_summary = if output.stderr.is_empty() {
            "no stderr was captured".to_owned()
        } else {
            format!(
                "stderr content omitted ({} bounded bytes captured)",
                output.stderr.len()
            )
        };
        return Err(EnrootProvenanceError::new(format!(
            "{label} failed with status {}; {stderr_summary}",
            output
                .status
                .code()
                .map_or_else(|| "signal".to_owned(), |code| code.to_string())
        )));
    }
    String::from_utf8(output.stdout)
        .map_err(|error| EnrootProvenanceError::new(format!("{label} was not UTF-8: {error}")))
}

fn parse_settings(output: &str) -> Result<BTreeMap<String, String>, EnrootProvenanceError> {
    let mut settings = BTreeMap::new();
    for line in output.lines() {
        let entry = line.trim();
        if !entry.starts_with("ENROOT_") {
            continue;
        }
        let Some((key, value)) = entry.split_once('=') else {
            return Err(EnrootProvenanceError::new(
                "Enroot info contained a malformed setting",
            ));
        };
        if !key
            .bytes()
            .all(|byte| byte == b'_' || byte.is_ascii_uppercase() || byte.is_ascii_digit())
        {
            return Err(EnrootProvenanceError::new(
                "Enroot info contained an invalid setting name",
            ));
        }
        if settings.insert(key.to_owned(), value.to_owned()).is_some() {
            return Err(EnrootProvenanceError::new(format!(
                "Enroot info repeated setting {key}"
            )));
        }
    }
    Ok(settings)
}

fn update_length_prefixed(digest: &mut Sha256, bytes: &[u8]) {
    digest.update((bytes.len() as u64).to_be_bytes());
    digest.update(bytes);
}

fn settings_sha256(settings: &BTreeMap<String, String>) -> String {
    let mut digest = Sha256::new();
    update_length_prefixed(&mut digest, b"tasktrace-enroot-settings.v1");
    digest.update((settings.len() as u64).to_be_bytes());
    for (key, value) in settings {
        update_length_prefixed(&mut digest, key.as_bytes());
        update_length_prefixed(&mut digest, value.as_bytes());
    }
    format!("{:x}", digest.finalize())
}

fn path_string(path: &Path, field: &str) -> Result<String, EnrootProvenanceError> {
    path.to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| EnrootProvenanceError::new(format!("{field} path is not valid UTF-8")))
}

fn lexical_absolute(path: &Path) -> Result<PathBuf, EnrootProvenanceError> {
    let source = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::from("/");
    for component in source.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Normal(value) => normalized.push(value),
            Component::Prefix(_) => {
                return Err(EnrootProvenanceError::new(format!(
                    "unsupported path prefix: {}",
                    source.display()
                )))
            }
        }
    }
    Ok(normalized)
}

fn resolve_allow_missing(path: &Path) -> Result<PathBuf, EnrootProvenanceError> {
    let absolute = lexical_absolute(path)?;
    match fs::canonicalize(&absolute) {
        Ok(resolved) => Ok(resolved),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(absolute),
        Err(error) => Err(EnrootProvenanceError::new(format!(
            "unable to resolve configuration path: {error}"
        ))),
    }
}

fn excluded_credential_path(relative: &Path) -> bool {
    relative
        .components()
        .any(|component| component.as_os_str() == OsStr::new(".credentials"))
}

#[derive(Clone, Copy, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ContentMetadataIdentity {
    device: u64,
    inode: u64,
    mode: u32,
    uid: u32,
    gid: u32,
    length: u64,
    modified_seconds: i64,
    modified_nanoseconds: i64,
    changed_seconds: i64,
    changed_nanoseconds: i64,
}

fn content_metadata_identity(metadata: &fs::Metadata) -> ContentMetadataIdentity {
    ContentMetadataIdentity {
        device: metadata.dev(),
        inode: metadata.ino(),
        mode: metadata.mode(),
        uid: metadata.uid(),
        gid: metadata.gid(),
        length: metadata.len(),
        modified_seconds: metadata.mtime(),
        modified_nanoseconds: metadata.mtime_nsec(),
        changed_seconds: metadata.ctime(),
        changed_nanoseconds: metadata.ctime_nsec(),
    }
}

fn require_stable_content_identity(
    before: ContentMetadataIdentity,
    after: &fs::Metadata,
    kind: &str,
) -> Result<(), EnrootProvenanceError> {
    if before == content_metadata_identity(after) {
        Ok(())
    } else {
        Err(EnrootProvenanceError::new(format!(
            "runtime content {kind} changed while its identity was captured"
        )))
    }
}

fn finalized_sha256(digest: Sha256) -> [u8; 32] {
    let value = digest.finalize();
    let mut bytes = [0_u8; 32];
    bytes.copy_from_slice(&value);
    bytes
}

fn update_behavior_metadata(digest: &mut Sha256, identity: ContentMetadataIdentity) {
    digest.update(identity.mode.to_be_bytes());
    digest.update(identity.uid.to_be_bytes());
    digest.update(identity.gid.to_be_bytes());
}

fn stable_file_content_sha256(
    path: &Path,
    expected: ContentMetadataIdentity,
) -> Result<[u8; 32], EnrootProvenanceError> {
    let mut file = OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW)
        .open(path)
        .map_err(|error| {
            EnrootProvenanceError::new(format!(
                "unable to open runtime content file while capturing identity: {error}"
            ))
        })?;
    let opened = file.metadata()?;
    if !opened.is_file() || content_metadata_identity(&opened) != expected {
        return Err(EnrootProvenanceError::new(
            "runtime content file was replaced while its identity was captured",
        ));
    }
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 1024 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    require_stable_content_identity(expected, &file.metadata()?, "file")?;
    let final_path_metadata = fs::symlink_metadata(path)?;
    if final_path_metadata.file_type().is_symlink() || !final_path_metadata.is_file() {
        return Err(EnrootProvenanceError::new(
            "runtime content file became a symlink or non-file while its identity was captured",
        ));
    }
    require_stable_content_identity(expected, &final_path_metadata, "file path")?;
    Ok(finalized_sha256(digest))
}

fn hash_regular_file_node(
    path: &Path,
    relative: &Path,
    expected: ContentMetadataIdentity,
) -> Result<[u8; 32], EnrootProvenanceError> {
    let content_digest = stable_file_content_sha256(path, expected)?;
    let mut digest = content_node_header(relative, b"file");
    update_behavior_metadata(&mut digest, expected);
    digest.update(expected.length.to_be_bytes());
    update_length_prefixed(&mut digest, &content_digest);
    Ok(finalized_sha256(digest))
}

fn content_node_header(relative: &Path, kind: &[u8]) -> Sha256 {
    let mut digest = Sha256::new();
    update_length_prefixed(&mut digest, b"tasktrace-content-node.v2");
    update_length_prefixed(&mut digest, relative.as_os_str().as_bytes());
    update_length_prefixed(&mut digest, kind);
    digest
}

fn hash_content_node(
    path: &Path,
    relative: &Path,
    boundary: &Path,
    exclude_credentials: bool,
    allowed_external_file_roots: &[PathBuf],
    active_directories: &mut BTreeSet<(u64, u64)>,
) -> Result<Option<[u8; 32]>, EnrootProvenanceError> {
    if exclude_credentials && excluded_credential_path(relative) {
        return Ok(None);
    }
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        EnrootProvenanceError::new(format!(
            "unable to inspect runtime content while capturing identity: {error}"
        ))
    })?;
    let expected = content_metadata_identity(&metadata);
    if metadata.is_file() {
        return Ok(Some(hash_regular_file_node(path, relative, expected)?));
    }
    if metadata.is_dir() {
        let directory_identity = (metadata.dev(), metadata.ino());
        if !active_directories.insert(directory_identity) {
            return Err(EnrootProvenanceError::new(
                "runtime content contains a directory symlink cycle",
            ));
        }
        let captured = (|| {
            let directory = OpenOptions::new()
                .read(true)
                .custom_flags(libc::O_CLOEXEC | libc::O_DIRECTORY | libc::O_NOFOLLOW)
                .open(path)
                .map_err(|error| {
                    EnrootProvenanceError::new(format!(
                        "unable to open runtime content directory while capturing identity: {error}"
                    ))
                })?;
            let opened_metadata = directory.metadata()?;
            if !opened_metadata.is_dir() || content_metadata_identity(&opened_metadata) != expected
            {
                return Err(EnrootProvenanceError::new(
                    "runtime content directory was replaced while its identity was captured",
                ));
            }
            let descriptor_path =
                PathBuf::from("/proc/self/fd").join(directory.as_raw_fd().to_string());
            let mut children = fs::read_dir(&descriptor_path)
                .map_err(|error| {
                    EnrootProvenanceError::new(format!(
                        "unable to read runtime content directory while capturing identity: {error}"
                    ))
                })?
                .map(|entry| entry.map(|entry| entry.path()))
                .collect::<Result<Vec<_>, _>>()?;
            children.sort_by(|left, right| {
                left.file_name()
                    .map(OsStr::as_bytes)
                    .cmp(&right.file_name().map(OsStr::as_bytes))
            });
            let mut child_identities = Vec::new();
            for child in children {
                let name = child.file_name().ok_or_else(|| {
                    EnrootProvenanceError::new("runtime content directory entry has no filename")
                })?;
                let child_relative = relative.join(name);
                if let Some(identity) = hash_content_node(
                    &child,
                    &child_relative,
                    boundary,
                    exclude_credentials,
                    allowed_external_file_roots,
                    active_directories,
                )? {
                    child_identities.push((name.as_bytes().to_vec(), identity));
                }
            }
            require_stable_content_identity(expected, &directory.metadata()?, "directory handle")?;
            let final_metadata = fs::symlink_metadata(path)?;
            if !final_metadata.is_dir() || final_metadata.file_type().is_symlink() {
                return Err(EnrootProvenanceError::new(
                    "runtime content directory was replaced while its identity was captured",
                ));
            }
            require_stable_content_identity(expected, &final_metadata, "directory")?;
            let mut digest = content_node_header(relative, b"directory");
            update_behavior_metadata(&mut digest, expected);
            digest.update((child_identities.len() as u64).to_be_bytes());
            for (name, identity) in child_identities {
                update_length_prefixed(&mut digest, &name);
                update_length_prefixed(&mut digest, &identity);
            }
            Ok(Some(finalized_sha256(digest)))
        })();
        active_directories.remove(&directory_identity);
        return captured;
    }
    if metadata.file_type().is_symlink() {
        let link_target = fs::read_link(path)?;
        let resolved_target = fs::canonicalize(path).map_err(|error| {
            EnrootProvenanceError::new(format!(
                "unable to resolve runtime content symlink target: {error}"
            ))
        })?;
        let target_within_boundary = resolved_target.starts_with(boundary);
        let external_target_allowed = allowed_external_file_roots
            .iter()
            .any(|root| resolved_target.starts_with(root));
        if !target_within_boundary && !external_target_allowed {
            return Err(EnrootProvenanceError::new(format!(
                "runtime content symlink escaped its allowed root at {}",
                relative.display()
            )));
        }
        let target_relative = if target_within_boundary {
            resolved_target
                .strip_prefix(boundary)
                .map_err(|_| {
                    EnrootProvenanceError::new(
                        "runtime content symlink target escaped its allowed root",
                    )
                })?
                .to_path_buf()
        } else {
            resolved_target.clone()
        };
        let external_target_identity = if target_within_boundary {
            None
        } else {
            let target_metadata = fs::symlink_metadata(&resolved_target)?;
            if target_metadata.file_type().is_symlink() || !target_metadata.is_file() {
                return Err(EnrootProvenanceError::new(
                    "allowed external runtime content target is not a regular file",
                ));
            }
            Some(content_metadata_identity(&target_metadata))
        };
        let target_is_credential =
            exclude_credentials && excluded_credential_path(&target_relative);
        let effective_identity = if target_is_credential {
            None
        } else if let Some(target_identity) = external_target_identity {
            Some(hash_regular_file_node(
                &resolved_target,
                &relative.join("@effective-target"),
                target_identity,
            )?)
        } else {
            hash_content_node(
                &resolved_target,
                &relative.join("@effective-target"),
                boundary,
                exclude_credentials,
                allowed_external_file_roots,
                active_directories,
            )?
        };
        let final_link_target = fs::read_link(path)?;
        let final_resolved_target = fs::canonicalize(path)?;
        let final_metadata = fs::symlink_metadata(path)?;
        if !final_metadata.file_type().is_symlink()
            || link_target != final_link_target
            || resolved_target != final_resolved_target
        {
            return Err(EnrootProvenanceError::new(
                "runtime content symlink changed while its identity was captured",
            ));
        }
        require_stable_content_identity(expected, &final_metadata, "symlink")?;
        if let Some(target_identity) = external_target_identity {
            let final_target_metadata = fs::symlink_metadata(&resolved_target)?;
            if final_target_metadata.file_type().is_symlink() || !final_target_metadata.is_file() {
                return Err(EnrootProvenanceError::new(
                    "allowed external runtime content target changed file type during capture",
                ));
            }
            require_stable_content_identity(
                target_identity,
                &final_target_metadata,
                "external file target",
            )?;
        }
        let mut digest = content_node_header(relative, b"symlink");
        update_behavior_metadata(&mut digest, expected);
        update_length_prefixed(&mut digest, link_target.as_os_str().as_bytes());
        update_length_prefixed(&mut digest, target_relative.as_os_str().as_bytes());
        if target_is_credential {
            update_length_prefixed(&mut digest, b"credential-target-excluded");
        } else {
            let identity = effective_identity.ok_or_else(|| {
                EnrootProvenanceError::new(
                    "runtime content symlink target was unexpectedly excluded",
                )
            })?;
            update_length_prefixed(&mut digest, &identity);
        }
        return Ok(Some(finalized_sha256(digest)));
    }
    Err(EnrootProvenanceError::new(
        "runtime content contains an unsupported non-file entry",
    ))
}

fn content_tree_sha256_with_policy(
    root: &Path,
    exclude_credentials: bool,
    allowed_external_file_roots: &[PathBuf],
) -> Result<Option<String>, EnrootProvenanceError> {
    for allowed_root in allowed_external_file_roots {
        let metadata = fs::symlink_metadata(allowed_root)?;
        if metadata.file_type().is_symlink()
            || !metadata.is_dir()
            || fs::canonicalize(allowed_root)? != *allowed_root
        {
            return Err(EnrootProvenanceError::new(
                "external runtime content allowlist root is not a canonical directory",
            ));
        }
    }
    let metadata = match fs::symlink_metadata(root) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let resolved_root = fs::canonicalize(root).map_err(|error| {
        EnrootProvenanceError::new(format!(
            "unable to resolve runtime content root while capturing identity: {error}"
        ))
    })?;
    let resolved_metadata = fs::metadata(&resolved_root)?;
    let boundary = if resolved_metadata.is_dir() {
        resolved_root.clone()
    } else {
        resolved_root
            .parent()
            .ok_or_else(|| EnrootProvenanceError::new("runtime content root has no parent"))?
            .to_path_buf()
    };
    let logical_root =
        if metadata.is_dir() {
            PathBuf::new()
        } else {
            PathBuf::from(root.file_name().ok_or_else(|| {
                EnrootProvenanceError::new("runtime content root has no filename")
            })?)
        };
    let path_to_hash = if metadata.file_type().is_symlink() {
        root
    } else {
        &resolved_root
    };
    let identity = hash_content_node(
        path_to_hash,
        &logical_root,
        &boundary,
        exclude_credentials,
        allowed_external_file_roots,
        &mut BTreeSet::new(),
    )?
    .unwrap_or_else(|| finalized_sha256(Sha256::new()));
    Ok(Some(format!("{:x}", Sha256::digest(identity))))
}

fn content_tree_sha256(
    root: &Path,
    exclude_credentials: bool,
) -> Result<Option<String>, EnrootProvenanceError> {
    content_tree_sha256_with_policy(root, exclude_credentials, &[])
}

fn packaged_enroot_external_file_roots() -> Result<Vec<PathBuf>, EnrootProvenanceError> {
    match fs::canonicalize(PACKAGED_ENROOT_ROOT) {
        Ok(root) => {
            if !fs::metadata(&root)?.is_dir() {
                return Err(EnrootProvenanceError::new(
                    "packaged Enroot root is not a directory",
                ));
            }
            Ok(vec![root])
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(Vec::new()),
        Err(error) => Err(EnrootProvenanceError::new(format!(
            "unable to resolve packaged Enroot root: {error}"
        ))),
    }
}

fn enroot_sibling_paths(executable: &Path) -> Result<Vec<PathBuf>, EnrootProvenanceError> {
    let parent = executable.parent().ok_or_else(|| {
        EnrootProvenanceError::new(format!(
            "Enroot executable has no parent directory: {}",
            executable.display()
        ))
    })?;
    let mut paths = Vec::new();
    for entry in fs::read_dir(parent).map_err(|error| {
        EnrootProvenanceError::new(format!(
            "unable to inspect Enroot executable directory {}: {error}",
            parent.display()
        ))
    })? {
        let entry = entry?;
        if entry.file_name().as_bytes().starts_with(b"enroot") {
            paths.push(entry.path());
        }
    }
    paths.sort();
    Ok(paths)
}

fn runtime_content_sha256(
    enroot: &Path,
    taskset: &Path,
    unshare: &Path,
    unsquashfs: &Path,
    library_path: &Path,
) -> Result<String, EnrootProvenanceError> {
    let mut paths = BTreeSet::new();
    paths.insert(enroot.to_path_buf());
    paths.insert(taskset.to_path_buf());
    paths.insert(unshare.to_path_buf());
    paths.insert(unsquashfs.to_path_buf());
    paths.insert(library_path.to_path_buf());
    paths.extend(enroot_sibling_paths(enroot)?);
    let mut digest = Sha256::new();
    update_length_prefixed(&mut digest, b"tasktrace-enroot-runtime-content.v2");
    digest.update((paths.len() as u64).to_be_bytes());
    for path in paths {
        let content_digest = content_tree_sha256(&path, false)?.ok_or_else(|| {
            EnrootProvenanceError::new(
                "required Enroot runtime content path disappeared during capture",
            )
        })?;
        update_length_prefixed(
            &mut digest,
            path_string(&path, "runtime content")?.as_bytes(),
        );
        update_length_prefixed(&mut digest, content_digest.as_bytes());
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn decode_mountinfo_path(value: &str) -> PathBuf {
    PathBuf::from(
        value
            .replace("\\040", " ")
            .replace("\\011", "\t")
            .replace("\\012", "\n")
            .replace("\\134", "\\"),
    )
}

fn mount_filesystem_type(
    path: &Path,
    mountinfo_path: &Path,
) -> Result<String, EnrootProvenanceError> {
    let text = fs::read_to_string(mountinfo_path).map_err(|error| {
        EnrootProvenanceError::new(format!(
            "unable to read mountinfo for state identity: {error}"
        ))
    })?;
    let mut best: Option<(usize, String)> = None;
    for (index, line) in text.lines().enumerate() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        let Some(separator) = fields.iter().position(|field| *field == "-") else {
            return Err(EnrootProvenanceError::new(format!(
                "mountinfo line {} is malformed",
                index + 1
            )));
        };
        if separator < 6 || separator + 3 >= fields.len() {
            return Err(EnrootProvenanceError::new(format!(
                "mountinfo line {} is malformed",
                index + 1
            )));
        }
        let mount_point = decode_mountinfo_path(fields[4]);
        if !mount_point.is_absolute() || fields[separator + 1].is_empty() {
            return Err(EnrootProvenanceError::new(format!(
                "mountinfo line {} has an invalid mount point or filesystem type",
                index + 1
            )));
        }
        if !path.starts_with(&mount_point) {
            continue;
        }
        let depth = mount_point.components().count();
        if best.as_ref().map_or(true, |(current, _)| depth > *current) {
            best = Some((depth, fields[separator + 1].to_owned()));
        }
    }
    best.map(|(_, filesystem)| filesystem).ok_or_else(|| {
        EnrootProvenanceError::new("mountinfo did not identify the state path filesystem")
    })
}

fn statvfs_filesystem_id(path: &Path) -> Result<String, EnrootProvenanceError> {
    let encoded = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| EnrootProvenanceError::new("state path contains a NUL byte"))?;
    let mut state = unsafe { mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(encoded.as_ptr(), &mut state) } != 0 {
        return Err(EnrootProvenanceError::new(format!(
            "unable to inspect state filesystem: {}",
            io::Error::last_os_error()
        )));
    }
    Ok(state.f_fsid.to_string())
}

fn state_path_identity(
    raw: &str,
    mountinfo_path: &Path,
) -> Result<StatePathIdentity, EnrootProvenanceError> {
    let path = fs::canonicalize(raw).map_err(|error| {
        EnrootProvenanceError::new(format!("unable to resolve required state path: {error}"))
    })?;
    let metadata = fs::metadata(&path)?;
    if !metadata.is_dir() {
        return Err(EnrootProvenanceError::new(
            "required Enroot state path is not a directory",
        ));
    }
    let device = metadata.dev() as libc::dev_t;
    Ok(StatePathIdentity {
        path: path_string(&path, "state")?,
        filesystem_type: Some(mount_filesystem_type(&path, mountinfo_path)?),
        device_major: unsafe { libc::major(device) } as u64,
        device_minor: unsafe { libc::minor(device) } as u64,
        filesystem_id: statvfs_filesystem_id(&path)?,
    })
}

fn comparable_digest(
    provenance: &EnrootRuntimeProvenance,
) -> Result<String, EnrootProvenanceError> {
    let mut value = serde_json::to_value(provenance).map_err(|error| {
        EnrootProvenanceError::new(format!("unable to encode Enroot provenance: {error}"))
    })?;
    let object = value
        .as_object_mut()
        .ok_or_else(|| EnrootProvenanceError::new("encoded Enroot provenance is not an object"))?;
    object.remove("comparability_sha256");
    let encoded = serde_json::to_vec(&canonical_json(value)).map_err(|error| {
        EnrootProvenanceError::new(format!("unable to encode Enroot provenance: {error}"))
    })?;
    Ok(format!("{:x}", Sha256::digest(encoded)))
}

fn canonical_json(value: serde_json::Value) -> serde_json::Value {
    match value {
        serde_json::Value::Array(values) => {
            serde_json::Value::Array(values.into_iter().map(canonical_json).collect())
        }
        serde_json::Value::Object(values) => {
            let sorted = values
                .into_iter()
                .map(|(key, value)| (key, canonical_json(value)))
                .collect::<BTreeMap<_, _>>();
            serde_json::Value::Object(sorted.into_iter().collect())
        }
        scalar => scalar,
    }
}

/// Capture Enroot runtime and configuration identity in every CPU context.
pub fn capture_runtime_provenance(
    config: &EnrootProvenanceConfig,
) -> Result<EnrootRuntimeProvenance, EnrootProvenanceError> {
    validate_cpu_set(&config.controller_cpus, "controller_cpus")?;
    if config.inspect_timeout.is_zero() {
        return Err(EnrootProvenanceError::new(
            "inspect_timeout must be positive",
        ));
    }
    for (index, cpus) in config.launch_cpu_sets.iter().enumerate() {
        validate_cpu_set(cpus, &format!("launch_cpu_sets[{index}]"))?;
    }

    let enroot = resolve_executable(&config.enroot)?;
    let taskset = resolve_executable(&config.taskset)?;
    let unshare = resolve_executable(&config.unshare)?;
    let unsquashfs = resolve_executable(&config.unsquashfs)?;
    let version = inspect(
        &taskset,
        &enroot,
        &config.controller_cpus,
        "version",
        config.inspect_timeout,
    )?
    .trim()
    .to_owned();
    if version.is_empty() {
        return Err(EnrootProvenanceError::new(
            "Enroot version inspection returned empty output",
        ));
    }

    let mut cpu_sets = BTreeSet::new();
    cpu_sets.insert(config.controller_cpus.clone());
    cpu_sets.extend(config.launch_cpu_sets.iter().cloned());
    let mut settings_by_cpus = BTreeMap::new();
    for cpus in cpu_sets {
        let settings = parse_settings(&inspect(
            &taskset,
            &enroot,
            &cpus,
            "info",
            config.inspect_timeout,
        )?)?;
        let missing = REQUIRED_STABLE_SETTINGS
            .iter()
            .filter(|key| !settings.contains_key(**key))
            .copied()
            .collect::<Vec<_>>();
        if !missing.is_empty() {
            return Err(EnrootProvenanceError::new(format!(
                "Enroot info on CPUs {cpus} omitted required settings: {}",
                missing.join(", ")
            )));
        }
        settings_by_cpus.insert(cpus, settings);
    }
    let controller_settings = settings_by_cpus
        .get(&config.controller_cpus)
        .expect("controller settings were captured");
    for (cpus, settings) in &settings_by_cpus {
        for key in REQUIRED_STABLE_SETTINGS {
            if settings.get(*key) != controller_settings.get(*key) {
                return Err(EnrootProvenanceError::new(format!(
                    "Enroot runtime path {key} changed with CPU affinity context {cpus}"
                )));
            }
        }
    }

    let setting = |key: &str| {
        controller_settings
            .get(key)
            .map(String::as_str)
            .ok_or_else(|| EnrootProvenanceError::new(format!("missing Enroot setting {key}")))
    };
    let library_path = fs::canonicalize(setting("ENROOT_LIBRARY_PATH")?).map_err(|error| {
        EnrootProvenanceError::new(format!("unable to resolve ENROOT_LIBRARY_PATH: {error}"))
    })?;
    let system_config_path = resolve_allow_missing(Path::new(setting("ENROOT_SYSCONF_PATH")?))?;
    let user_config_path = resolve_allow_missing(Path::new(setting("ENROOT_CONFIG_PATH")?))?;

    let mut launches = BTreeMap::new();
    for cpus in config
        .launch_cpu_sets
        .iter()
        .cloned()
        .collect::<BTreeSet<_>>()
    {
        launches.insert(
            cpus.clone(),
            settings_sha256(
                settings_by_cpus
                    .get(&cpus)
                    .expect("launch settings were captured"),
            ),
        );
    }
    let mut state_paths = BTreeMap::new();
    for key in STATE_PATH_SETTINGS {
        state_paths.insert(
            (*key).to_owned(),
            state_path_identity(setting(key)?, &config.mountinfo_path)?,
        );
    }
    let config_external_file_roots = packaged_enroot_external_file_roots()?;

    let mut provenance = EnrootRuntimeProvenance {
        schema_version: ENROOT_PROVENANCE_SCHEMA_VERSION.to_owned(),
        version,
        executable: path_string(&enroot, "Enroot executable")?,
        containment_executable: path_string(&unshare, "containment executable")?,
        runtime_content_sha256: runtime_content_sha256(
            &enroot,
            &taskset,
            &unshare,
            &unsquashfs,
            &library_path,
        )?,
        effective_config_sha256: EffectiveConfigIdentities {
            controller: EffectiveConfigIdentity {
                cpus: config.controller_cpus.clone(),
                sha256: settings_sha256(controller_settings),
            },
            launches,
        },
        system_config_sha256: content_tree_sha256_with_policy(
            &system_config_path,
            true,
            &config_external_file_roots,
        )?,
        user_config_sha256: content_tree_sha256_with_policy(
            &user_config_path,
            true,
            &config_external_file_roots,
        )?,
        state_paths,
        comparability_sha256: String::new(),
    };
    provenance.comparability_sha256 = comparable_digest(&provenance)?;
    Ok(provenance)
}

/// Reject any runtime, configuration, or state-filesystem drift after a run.
pub fn validate_runtime_provenance_unchanged(
    before: &EnrootRuntimeProvenance,
    after: &EnrootRuntimeProvenance,
) -> Result<(), EnrootProvenanceError> {
    if before != after {
        return Err(EnrootProvenanceError::new(format!(
            "Enroot runtime provenance changed during the run: before={}, after={}",
            before.comparability_sha256, after.comparability_sha256
        )));
    }
    Ok(())
}

/// Capture the post-run identity and compare it with the pre-run snapshot.
pub fn recapture_and_validate(
    config: &EnrootProvenanceConfig,
    before: &EnrootRuntimeProvenance,
) -> Result<EnrootRuntimeProvenance, EnrootProvenanceError> {
    let after = capture_runtime_provenance(config)?;
    validate_runtime_provenance_unchanged(before, &after)?;
    Ok(after)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn write(root: &Path, relative: &str, bytes: &[u8]) {
        let path = root.join(relative);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn content_hash_is_deterministic_across_creation_order() {
        let first = tempfile::tempdir().unwrap();
        let second = tempfile::tempdir().unwrap();
        write(first.path(), "z/last", b"last");
        write(first.path(), "a/first", b"first");
        write(second.path(), "a/first", b"first");
        write(second.path(), "z/last", b"last");

        assert_eq!(
            content_tree_sha256(first.path(), false).unwrap(),
            content_tree_sha256(second.path(), false).unwrap()
        );
    }

    #[test]
    fn content_hash_distinguishes_the_old_delimiter_collision() {
        let embedded_record = tempfile::tempdir().unwrap();
        let separate_records = tempfile::tempdir().unwrap();
        write(embedded_record.path(), "a", b"X\0b\0file\0Y");
        write(separate_records.path(), "a", b"X");
        write(separate_records.path(), "b", b"Y");

        assert_ne!(
            content_tree_sha256(embedded_record.path(), false).unwrap(),
            content_tree_sha256(separate_records.path(), false).unwrap()
        );
    }

    #[test]
    fn system_and_user_config_hashes_exclude_credentials() {
        for config in [tempfile::tempdir().unwrap(), tempfile::tempdir().unwrap()] {
            write(config.path(), "enroot.conf", b"ENROOT_REMAP_ROOT=y\n");
            write(config.path(), ".credentials", b"registry secret one\n");
            write(
                config.path(),
                "nested/.credentials/token",
                b"nested secret one\n",
            );
            let initial = content_tree_sha256(config.path(), true).unwrap();

            write(config.path(), ".credentials", b"registry secret two\n");
            write(
                config.path(),
                "nested/.credentials/token",
                b"nested secret two\n",
            );
            assert_eq!(initial, content_tree_sha256(config.path(), true).unwrap());

            write(config.path(), "enroot.conf", b"ENROOT_REMAP_ROOT=n\n");
            assert_ne!(initial, content_tree_sha256(config.path(), true).unwrap());
        }
    }

    #[test]
    fn content_hash_binds_effective_symlink_target_contents() {
        let root = tempfile::tempdir().unwrap();
        write(root.path(), "target", b"first target contents");
        symlink("target", root.path().join("link")).unwrap();
        let initial = content_tree_sha256(root.path(), false).unwrap();

        write(root.path(), "target", b"second target content");
        assert_ne!(initial, content_tree_sha256(root.path(), false).unwrap());
    }

    #[test]
    fn content_hash_rejects_symlink_targets_outside_the_root() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        fs::create_dir(&root).unwrap();
        write(parent.path(), "outside", b"outside contents");
        symlink("../outside", root.join("link")).unwrap();

        let error = content_tree_sha256(&root, false).unwrap_err();
        assert!(error.to_string().contains("escaped its allowed root"));
    }

    #[test]
    fn config_hash_binds_allowed_external_file_contents() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        let packaged = parent.path().join("packaged-enroot");
        fs::create_dir(&root).unwrap();
        write(&packaged, "hooks.d/hook", b"first hook contents");
        let packaged = fs::canonicalize(packaged).unwrap();
        symlink(packaged.join("hooks.d/hook"), root.join("hook")).unwrap();

        let initial = content_tree_sha256_with_policy(&root, true, &[packaged.clone()]).unwrap();
        write(&packaged, "hooks.d/hook", b"second hook content");
        assert_ne!(
            initial,
            content_tree_sha256_with_policy(&root, true, &[packaged]).unwrap()
        );
    }

    #[test]
    fn config_hash_rejects_allowed_external_directories() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        let packaged = parent.path().join("packaged-enroot");
        fs::create_dir(&root).unwrap();
        fs::create_dir_all(packaged.join("hooks.d")).unwrap();
        let packaged = fs::canonicalize(packaged).unwrap();
        symlink(packaged.join("hooks.d"), root.join("hooks.d")).unwrap();

        let error = content_tree_sha256_with_policy(&root, true, &[packaged]).unwrap_err();
        assert!(error.to_string().contains("is not a regular file"));
    }

    #[test]
    fn config_hash_binds_external_hook_executable_mode() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        let packaged = parent.path().join("packaged-enroot");
        fs::create_dir(&root).unwrap();
        write(&packaged, "hooks.d/hook", b"#!/bin/sh\nexit 0\n");
        let packaged = fs::canonicalize(packaged).unwrap();
        let hook = packaged.join("hooks.d/hook");
        fs::set_permissions(&hook, fs::Permissions::from_mode(0o755)).unwrap();
        symlink(&hook, root.join("hook")).unwrap();
        let executable = content_tree_sha256_with_policy(&root, true, &[packaged.clone()]).unwrap();

        fs::set_permissions(&hook, fs::Permissions::from_mode(0o644)).unwrap();
        assert_ne!(
            executable,
            content_tree_sha256_with_policy(&root, true, &[packaged]).unwrap()
        );
    }

    #[test]
    fn config_hash_excludes_allowed_external_credentials() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        let packaged = parent.path().join("packaged-enroot");
        fs::create_dir(&root).unwrap();
        write(&packaged, ".credentials/token", b"first secret");
        let packaged = fs::canonicalize(packaged).unwrap();
        symlink(
            packaged.join(".credentials/token"),
            root.join("credential-alias"),
        )
        .unwrap();
        let initial = content_tree_sha256_with_policy(&root, true, &[packaged.clone()]).unwrap();

        write(&packaged, ".credentials/token", b"second secret");
        assert_eq!(
            initial,
            content_tree_sha256_with_policy(&root, true, &[packaged]).unwrap()
        );
    }

    #[test]
    fn config_hash_binds_external_retarget_and_rejects_non_allowlisted_target() {
        let parent = tempfile::tempdir().unwrap();
        let root = parent.path().join("root");
        let packaged = parent.path().join("packaged-enroot");
        fs::create_dir(&root).unwrap();
        write(&packaged, "hooks.d/first", b"same contents");
        write(&packaged, "hooks.d/second", b"same contents");
        write(parent.path(), "outside", b"same contents");
        let packaged = fs::canonicalize(packaged).unwrap();
        let link = root.join("hook");
        symlink(packaged.join("hooks.d/first"), &link).unwrap();
        let first = content_tree_sha256_with_policy(&root, true, &[packaged.clone()]).unwrap();

        fs::remove_file(&link).unwrap();
        symlink(packaged.join("hooks.d/second"), &link).unwrap();
        assert_ne!(
            first,
            content_tree_sha256_with_policy(&root, true, &[packaged.clone()]).unwrap()
        );

        fs::remove_file(&link).unwrap();
        symlink(parent.path().join("outside"), &link).unwrap();
        let error = content_tree_sha256_with_policy(&root, true, &[packaged]).unwrap_err();
        assert!(error.to_string().contains("escaped its allowed root"));
    }

    #[test]
    fn inspection_output_limit_kills_the_owned_process_group() {
        let mut command = Command::new("/bin/sh");
        command.args([
            "-c",
            "while :; do printf '0123456789abcdef0123456789abcdef'; printf 'fedcba9876543210fedcba9876543210' >&2; done",
        ]);
        let started = Instant::now();
        let error = command_output_with_timeout(
            &mut command,
            Duration::from_secs(5),
            "bounded-output test",
        )
        .unwrap_err();

        assert!(error.to_string().contains("capture limit"));
        assert!(started.elapsed() < Duration::from_secs(3));
    }

    #[test]
    fn inspection_timeout_kills_descendants_without_waiting_on_pipes() {
        let root = tempfile::tempdir().unwrap();
        let pid_file = root.path().join("inspection-pids");
        let mut command = Command::new("/bin/sh");
        command.env("PID_FILE", &pid_file);
        command.args([
            "-c",
            "printf '%s\\n' \"$$\" >\"$PID_FILE\"; trap '' TERM; (trap '' TERM; while :; do sleep 1; done) & printf '%s\\n' \"$!\" >>\"$PID_FILE\"; while :; do sleep 1; done",
        ]);
        let started = Instant::now();
        let error = command_output_with_timeout(
            &mut command,
            Duration::from_millis(250),
            "descendant-timeout test",
        )
        .unwrap_err();

        assert!(error.to_string().contains("timed out"));
        assert!(!error.to_string().contains("cleanup failed"));
        assert!(started.elapsed() < Duration::from_secs(3));
        let pids = fs::read_to_string(pid_file).unwrap();
        for pid in pids.lines().map(|line| line.parse::<i32>().unwrap()) {
            assert_eq!(unsafe { libc::kill(pid, 0) }, -1);
            assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
        }
    }

    #[test]
    fn inspection_errors_do_not_echo_stderr_or_setting_values() {
        let root = tempfile::tempdir().unwrap();
        let taskset = root.path().join("taskset");
        fs::write(
            &taskset,
            b"#!/bin/sh\nprintf 'ENROOT_PASSWORD=do-not-disclose\\n' >&2\nexit 7\n",
        )
        .unwrap();
        fs::set_permissions(&taskset, fs::Permissions::from_mode(0o700)).unwrap();
        let error = inspect(
            &taskset,
            Path::new("/bin/true"),
            "0",
            "info",
            Duration::from_secs(1),
        )
        .unwrap_err();
        assert!(!error.to_string().contains("do-not-disclose"));
        assert!(!error.to_string().contains("ENROOT_PASSWORD"));

        let duplicate =
            parse_settings("ENROOT_PASSWORD=first-secret\nENROOT_PASSWORD=second-secret\n")
                .unwrap_err();
        assert!(!duplicate.to_string().contains("first-secret"));
        assert!(!duplicate.to_string().contains("second-secret"));
    }

    #[test]
    fn malformed_mountinfo_fails_state_identity() {
        let state = tempfile::tempdir().unwrap();
        let mountinfo = state.path().join("mountinfo");
        fs::write(&mountinfo, b"this is not mountinfo\n").unwrap();

        let error = state_path_identity(state.path().to_str().unwrap(), &mountinfo).unwrap_err();
        assert!(error.to_string().contains("mountinfo line 1 is malformed"));
    }

    #[test]
    fn state_identity_requires_a_directory() {
        let root = tempfile::tempdir().unwrap();
        let state = root.path().join("state-file");
        fs::write(&state, b"not a directory").unwrap();
        let mountinfo = root.path().join("mountinfo");
        fs::write(
            &mountinfo,
            b"36 25 0:32 / / rw,relatime - ext4 /dev/root rw\n",
        )
        .unwrap();

        let error = state_path_identity(state.to_str().unwrap(), &mountinfo).unwrap_err();
        assert!(error.to_string().contains("is not a directory"));
    }

    #[test]
    fn comparability_digest_excludes_only_its_own_value() {
        let mut provenance = EnrootRuntimeProvenance {
            schema_version: ENROOT_PROVENANCE_SCHEMA_VERSION.to_owned(),
            version: "4.0.1".to_owned(),
            executable: "/usr/bin/enroot".to_owned(),
            containment_executable: "/usr/bin/unshare".to_owned(),
            runtime_content_sha256: "a".repeat(64),
            effective_config_sha256: EffectiveConfigIdentities {
                controller: EffectiveConfigIdentity {
                    cpus: "0-3".to_owned(),
                    sha256: "b".repeat(64),
                },
                launches: BTreeMap::from([("4".to_owned(), "c".repeat(64))]),
            },
            system_config_sha256: Some("d".repeat(64)),
            user_config_sha256: Some("e".repeat(64)),
            state_paths: BTreeMap::new(),
            comparability_sha256: String::new(),
        };
        let first = comparable_digest(&provenance).unwrap();
        provenance.comparability_sha256 = "untrusted prior value".to_owned();
        assert_eq!(first, comparable_digest(&provenance).unwrap());
        provenance.version = "4.0.2".to_owned();
        assert_ne!(first, comparable_digest(&provenance).unwrap());
    }
}
