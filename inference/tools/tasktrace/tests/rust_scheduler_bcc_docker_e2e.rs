// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in privileged Docker gate for the native BCC scheduler backend.
//!
//! The controller runs in a disposable, root-capable wrapper because the test
//! runner itself need not be root. The immutable wrapper image only needs a
//! compatible `/usr/sbin/chroot` and `/bin/sh`; the wrapper uses its read-only rootfs as an
//! empty `/host` skeleton, selectively binds only the host paths required by
//! the controller recursively read-only, overlays the exact artifact root and
//! host tracefs read-write, and chroots there. The writable Docker socket and
//! privileged host namespaces confer root-equivalent host control authority.
//! In particular, host-PID `/proc/<pid>/root` can expose paths outside the
//! selective binds, so this is only appropriate on a dedicated test host. The
//! task image must be preloaded and contain Bash and Python 3. Worker CPUs and
//! controller CPUs must be reserved and disjoint. The artifact directory must
//! be a new direct child of `/tmp`.
//!
//! ```text
//! TASKTRACE_RUST_BCC_WRAPPER_IMAGE=sha256:<64-lowercase-hex> \
//! TASKTRACE_RUST_BCC_CONTROLLER_CPUSET=0-3 \
//! TASKTRACE_RUST_DOCKER_TEST_IMAGE=sha256:<64-lowercase-hex> \
//! TASKTRACE_RUST_DOCKER_CPUSET=16-19 \
//! TASKTRACE_RUST_BCC_ARTIFACT_DIR=/tmp/tasktrace-rust-bcc-e2e-<unique> \
//! cargo test --offline --test rust_scheduler_bcc_docker_e2e -- \
//!   --ignored --nocapture --test-threads=1
//! ```

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::io::Read;
use std::os::unix::fs::{FileTypeExt, MetadataExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const WRAPPER_CHROOT: &str = "/host";
const DOCKER_SOCKET: &str = "/var/run/docker.sock";
const CHROOT_DOCKER_SOCKET: &str = "/run/docker.sock";
const WRAPPER_DOCKER_SOCKET: &str = "/host/run/docker.sock";
const HOST_TRACEFS: &str = "/sys/kernel/debug/tracing";
const WRAPPER_TRACEFS: &str = "/host/sys/kernel/debug/tracing";
const READ_ONLY_HOST_PATHS: [&str; 10] = [
    "/usr", "/bin", "/sbin", "/lib", "/lib64", "/etc", "/proc", "/sys", "/dev", "/boot",
];
const WRAPPER_WAIT: Duration = Duration::from_secs(90);
const DOCKER_WAIT: Duration = Duration::from_secs(30);
const DOCKER_CLEANUP_WAIT: Duration = Duration::from_secs(15);
const DOCKER_REAP_WAIT: Duration = Duration::from_secs(2);
const DOCKER_CAPTURE_WAIT: Duration = Duration::from_secs(2);
const CLEANUP_SCAN_GAP: Duration = Duration::from_millis(250);
const INDETERMINATE_SETTLE_WINDOW: Duration = Duration::from_secs(10);

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
}

#[derive(Debug)]
enum DockerCliError {
    Start {
        operation: &'static str,
        error: String,
    },
    Wait {
        operation: &'static str,
        error: String,
        kill_error: Option<String>,
        reap_error: Option<String>,
    },
    Timeout {
        operation: &'static str,
        timeout: Duration,
        kill_error: Option<String>,
        reap_error: Option<String>,
    },
    Capture {
        operation: &'static str,
        stream: &'static str,
        error: String,
    },
    CaptureTimeout {
        operation: &'static str,
        stream: &'static str,
        timeout: Duration,
    },
}

impl DockerCliError {
    fn kind(&self) -> &'static str {
        match self {
            Self::Start { .. } => "start-error",
            Self::Wait { .. } => "wait-error",
            Self::Timeout { .. } => "timeout",
            Self::Capture { .. } => "capture-error",
            Self::CaptureTimeout { .. } => "capture-timeout",
        }
    }
}

impl std::fmt::Display for DockerCliError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Start { operation, error } => {
                write!(formatter, "Docker {operation} start error: {error}")
            }
            Self::Wait {
                operation,
                error,
                kill_error,
                reap_error,
            } => write!(
                formatter,
                "Docker {operation} wait error: {error}; kill_error={kill_error:?}; reap_error={reap_error:?}"
            ),
            Self::Timeout {
                operation,
                timeout,
                kill_error,
                reap_error,
            } => write!(
                formatter,
                "Docker {operation} timed out after {timeout:?}; kill_error={kill_error:?}; reap_error={reap_error:?}"
            ),
            Self::Capture {
                operation,
                stream,
                error,
            } => write!(
                formatter,
                "Docker {operation} {stream} capture error: {error}"
            ),
            Self::CaptureTimeout {
                operation,
                stream,
                timeout,
            } => write!(
                formatter,
                "Docker {operation} {stream} capture timed out after {timeout:?}"
            ),
        }
    }
}

fn terminate_and_reap(child: &mut std::process::Child) -> (Option<String>, Option<String>) {
    let kill_error = child.kill().err().map(|error| error.to_string());
    let deadline = Instant::now() + DOCKER_REAP_WAIT;
    let reap_error = loop {
        match child.try_wait() {
            Ok(Some(_)) => break None,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(10)),
            Ok(None) => {
                break Some(format!(
                    "child was not reaped within {:?} after kill",
                    DOCKER_REAP_WAIT
                ))
            }
            Err(error) => break Some(error.to_string()),
        }
    };
    (kill_error, reap_error)
}

fn receive_capture(
    receiver: mpsc::Receiver<Result<Vec<u8>, String>>,
    operation: &'static str,
    stream: &'static str,
) -> Result<Vec<u8>, DockerCliError> {
    match receiver.recv_timeout(DOCKER_CAPTURE_WAIT) {
        Ok(Ok(bytes)) => Ok(bytes),
        Ok(Err(error)) => Err(DockerCliError::Capture {
            operation,
            stream,
            error,
        }),
        Err(mpsc::RecvTimeoutError::Timeout) => Err(DockerCliError::CaptureTimeout {
            operation,
            stream,
            timeout: DOCKER_CAPTURE_WAIT,
        }),
        Err(mpsc::RecvTimeoutError::Disconnected) => Err(DockerCliError::Capture {
            operation,
            stream,
            error: "capture worker disconnected".to_owned(),
        }),
    }
}

fn docker_bounded<I, S>(
    operation: &'static str,
    timeout: Duration,
    arguments: I,
) -> Result<Output, DockerCliError>
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    let mut child = Command::new("docker")
        .args(arguments)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|error| DockerCliError::Start {
            operation,
            error: error.to_string(),
        })?;
    let stdout = match child.stdout.take() {
        Some(stdout) => stdout,
        None => {
            let _ = terminate_and_reap(&mut child);
            return Err(DockerCliError::Capture {
                operation,
                stream: "stdout",
                error: "piped stdout was unavailable".to_owned(),
            });
        }
    };
    let stderr = match child.stderr.take() {
        Some(stderr) => stderr,
        None => {
            let _ = terminate_and_reap(&mut child);
            return Err(DockerCliError::Capture {
                operation,
                stream: "stderr",
                error: "piped stderr was unavailable".to_owned(),
            });
        }
    };
    let (stdout_tx, stdout_rx) = mpsc::channel();
    let (stderr_tx, stderr_rx) = mpsc::channel();
    if let Err(error) = thread::Builder::new()
        .name("tasktrace-docker-stdout".to_owned())
        .spawn(move || {
            let mut reader = stdout;
            let mut bytes = Vec::new();
            let result = reader
                .read_to_end(&mut bytes)
                .map(|_| bytes)
                .map_err(|error| error.to_string());
            let _ = stdout_tx.send(result);
        })
    {
        let _ = terminate_and_reap(&mut child);
        return Err(DockerCliError::Capture {
            operation,
            stream: "stdout",
            error: format!("capture worker start error: {error}"),
        });
    }
    if let Err(error) = thread::Builder::new()
        .name("tasktrace-docker-stderr".to_owned())
        .spawn(move || {
            let mut reader = stderr;
            let mut bytes = Vec::new();
            let result = reader
                .read_to_end(&mut bytes)
                .map(|_| bytes)
                .map_err(|error| error.to_string());
            let _ = stderr_tx.send(result);
        })
    {
        let _ = terminate_and_reap(&mut child);
        return Err(DockerCliError::Capture {
            operation,
            stream: "stderr",
            error: format!("capture worker start error: {error}"),
        });
    }

    let deadline = Instant::now() + timeout;
    let status: ExitStatus = loop {
        match child.try_wait() {
            Ok(Some(status)) => break status,
            Ok(None) if Instant::now() < deadline => thread::sleep(Duration::from_millis(20)),
            Ok(None) => {
                let (kill_error, reap_error) = terminate_and_reap(&mut child);
                return Err(DockerCliError::Timeout {
                    operation,
                    timeout,
                    kill_error,
                    reap_error,
                });
            }
            Err(error) => {
                let (kill_error, reap_error) = terminate_and_reap(&mut child);
                return Err(DockerCliError::Wait {
                    operation,
                    error: error.to_string(),
                    kill_error,
                    reap_error,
                });
            }
        }
    };
    let stdout = receive_capture(stdout_rx, operation, "stdout")?;
    let stderr = receive_capture(stderr_rx, operation, "stderr")?;
    Ok(Output {
        status,
        stdout,
        stderr,
    })
}

fn assert_success(output: &Output, description: &str) {
    assert!(
        output.status.success(),
        "{description} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn immutable_digest(value: &str) -> bool {
    let digest = value.rsplit_once('@').map_or(value, |(_, digest)| digest);
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn image_id(image: &str) -> String {
    let output = docker_bounded(
        "image inspect",
        DOCKER_WAIT,
        ["image", "inspect", "--format", "{{.Id}}", image],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_success(&output, "inspect preloaded immutable image");
    String::from_utf8(output.stdout)
        .expect("Docker image ID is UTF-8")
        .trim()
        .to_owned()
}

#[derive(Debug, Clone)]
struct CanonicalReadOnlyMount {
    requested_source: &'static str,
    source: PathBuf,
    destination: String,
}

#[derive(Debug, Clone)]
struct SelectedOverlapSource {
    label: String,
    bind_source: PathBuf,
    overlap_path: PathBuf,
    destination: String,
    resolution_method: &'static str,
    filesystem_traversal: bool,
    resolution_evidence: Value,
}

fn canonical_bind_source(path: &Path, description: &str) -> PathBuf {
    assert!(
        path.is_absolute(),
        "{description} must be absolute: {path:?}"
    );
    let canonical = path
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize {description} {path:?}: {error}"));
    let text = canonical
        .to_str()
        .unwrap_or_else(|| panic!("{description} is not UTF-8: {canonical:?}"));
    assert!(
        !text
            .chars()
            .any(|character| character == ',' || character.is_control()),
        "{description} cannot be represented safely in Docker --mount syntax: {canonical:?}"
    );
    canonical
}

fn canonical_read_only_mounts() -> Vec<CanonicalReadOnlyMount> {
    READ_ONLY_HOST_PATHS
        .iter()
        .map(|requested_source| CanonicalReadOnlyMount {
            requested_source,
            source: canonical_bind_source(
                Path::new(requested_source),
                "selective read-only host source",
            ),
            destination: format!("{WRAPPER_CHROOT}{requested_source}"),
        })
        .collect()
}

fn filesystem_canonicalized_overlap_source(
    label: impl Into<String>,
    bind_source: PathBuf,
    destination: impl Into<String>,
) -> SelectedOverlapSource {
    let label = label.into();
    let overlap_path = canonical_bind_source(
        &bind_source,
        &format!("filesystem-canonicalized overlap source {label}"),
    );
    SelectedOverlapSource {
        label,
        bind_source: bind_source.clone(),
        overlap_path: overlap_path.clone(),
        destination: destination.into(),
        resolution_method: "std::fs::canonicalize",
        filesystem_traversal: true,
        resolution_evidence: json!({
            "input_path": bind_source,
            "canonical_path": overlap_path,
        }),
    }
}

fn tracefs_mountinfo_overlap_source(
    tracefs_contract: &Value,
    destination: impl Into<String>,
) -> SelectedOverlapSource {
    let mountpoint = tracefs_contract["mountpoint"]
        .as_str()
        .expect("tracefs mount contract retains its exact mountpoint");
    assert_eq!(
        mountpoint, HOST_TRACEFS,
        "tracefs overlap path is the exact preflighted mountpoint"
    );
    let overlap_path = PathBuf::from(mountpoint);
    assert!(
        overlap_path.is_absolute()
            && !overlap_path.components().any(|component| {
                matches!(
                    component,
                    std::path::Component::CurDir
                        | std::path::Component::ParentDir
                        | std::path::Component::Prefix(_)
                )
            }),
        "tracefs mountinfo path must be absolute and lexically normalized: {overlap_path:?}"
    );
    SelectedOverlapSource {
        label: "tracefs-read-write".to_owned(),
        bind_source: PathBuf::from(HOST_TRACEFS),
        overlap_path,
        destination: destination.into(),
        resolution_method: "exact /proc/self/mountinfo mountpoint",
        filesystem_traversal: false,
        resolution_evidence: tracefs_contract.clone(),
    }
}

fn docker_root_dir_contract(
    selected_sources: &[SelectedOverlapSource],
    retained_path: &Path,
) -> Value {
    let output = docker_bounded(
        "query DockerRootDir",
        DOCKER_WAIT,
        ["info", "--format", "{{json .DockerRootDir}}"],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_success(&output, "query DockerRootDir");
    let reported = serde_json::from_slice::<String>(&output.stdout)
        .expect("DockerRootDir query returns one JSON string");
    let reported_path = PathBuf::from(&reported);
    assert!(
        reported_path.is_absolute(),
        "DockerRootDir must be absolute: {reported:?}"
    );
    let canonical_root = reported_path
        .canonicalize()
        .unwrap_or_else(|error| panic!("canonicalize DockerRootDir {reported_path:?}: {error}"));
    assert!(
        canonical_root.is_dir(),
        "canonical DockerRootDir is not a directory: {canonical_root:?}"
    );

    let root_bytes = canonical_root.as_os_str().as_encoded_bytes();
    let mut checked_sources = Vec::with_capacity(selected_sources.len());
    let mut overlap_diagnostics = Vec::new();
    for source in selected_sources {
        assert!(
            source.overlap_path.is_absolute(),
            "pre-resolved overlap source must be absolute: {:?}",
            source.overlap_path
        );
        let component_overlap = source.overlap_path.starts_with(&canonical_root)
            || canonical_root.starts_with(&source.overlap_path);
        let source_bytes = source.overlap_path.as_os_str().as_encoded_bytes();
        let lexical_prefix_overlap =
            source_bytes.starts_with(root_bytes) || root_bytes.starts_with(source_bytes);
        if component_overlap || lexical_prefix_overlap {
            overlap_diagnostics.push(format!(
                "selected source {:?} ({:?}) overlaps DockerRootDir {canonical_root:?}; component_overlap={component_overlap}; lexical_prefix_overlap={lexical_prefix_overlap}",
                source.label, source.overlap_path,
            ));
        }
        checked_sources.push(json!({
            "label": source.label,
            "bind_source": source.bind_source,
            "overlap_path": source.overlap_path,
            "destination": source.destination,
            "resolution_provenance": {
                "method": source.resolution_method,
                "filesystem_traversal": source.filesystem_traversal,
                "evidence": source.resolution_evidence,
            },
            "component_overlap": component_overlap,
            "lexical_prefix_overlap": lexical_prefix_overlap,
        }));
    }

    let contract = json!({
        "query": "docker info --format {{json .DockerRootDir}}",
        "reported": reported,
        "canonical": canonical_root,
        "overlap_policy": "reject either component-wise overlap or raw lexical-prefix overlap in either direction after each source's retained resolution procedure",
        "selected_sources_checked_before_wrapper_create": checked_sources,
        "overlap_diagnostics": overlap_diagnostics,
        "validation_status": if overlap_diagnostics.is_empty() { "passed" } else { "failed" },
    });
    write_json(retained_path, &contract);
    assert!(
        overlap_diagnostics.is_empty(),
        "refusing privileged wrapper before create because a selected bind source overlaps DockerRootDir; retained diagnostics: {}",
        retained_path.display()
    );
    contract
}

fn parse_cpuset(value: &str) -> BTreeSet<u32> {
    let mut cpus = BTreeSet::new();
    for component in value.split(',') {
        assert!(
            !component.is_empty(),
            "empty component in CPU set {value:?}"
        );
        if let Some((start, end)) = component.split_once('-') {
            let start = start
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("invalid CPU range start in {value:?}"));
            let end = end
                .parse::<u32>()
                .unwrap_or_else(|_| panic!("invalid CPU range end in {value:?}"));
            assert!(start <= end, "descending CPU range in {value:?}");
            cpus.extend(start..=end);
        } else {
            cpus.insert(
                component
                    .parse::<u32>()
                    .unwrap_or_else(|_| panic!("invalid CPU in {value:?}")),
            );
        }
    }
    assert!(!cpus.is_empty(), "CPU set must not be empty");
    cpus
}

fn tracefs_mount_contract(path: &Path) -> Value {
    let mountinfo = fs::read_to_string("/proc/self/mountinfo")
        .expect("read host mountinfo for explicit tracefs preflight");
    let path = path
        .to_str()
        .expect("tracefs path is UTF-8 for mountinfo comparison");
    let fields = mountinfo
        .lines()
        .find_map(|line| {
            let fields = line.split_whitespace().collect::<Vec<_>>();
            (fields.get(4) == Some(&path)).then_some(fields)
        })
        .unwrap_or_else(|| panic!("{path} is not an explicit host mountpoint"));
    let separator = fields
        .iter()
        .position(|field| *field == "-")
        .expect("mountinfo tracefs separator");
    let filesystem = fields
        .get(separator + 1)
        .expect("mountinfo tracefs filesystem");
    let mount_options = fields.get(5).expect("mountinfo tracefs mount options");
    let super_options = fields
        .get(separator + 3)
        .expect("mountinfo tracefs super options");
    assert_eq!(*filesystem, "tracefs", "required host mount is tracefs");
    assert!(
        mount_options.split(',').any(|option| option == "rw"),
        "host tracefs mount must be writable for BCC attach/detach"
    );
    assert!(
        super_options.split(',').any(|option| option == "rw"),
        "host tracefs superblock must be writable for BCC attach/detach"
    );
    json!({
        "source": path,
        "mountpoint": path,
        "filesystem": filesystem,
        "mount_options": mount_options,
        "super_options": super_options,
        "writable": true,
        "path_resolution": {
            "method": "exact /proc/self/mountinfo mountpoint field",
            "filesystem_traversal": false,
            "filesystem_canonicalization": false,
        },
    })
}

fn proc_status_field<'a>(status: &'a str, name: &str) -> &'a str {
    status
        .lines()
        .find_map(|line| {
            let (field, value) = line.split_once(':')?;
            (field == name).then_some(value.trim())
        })
        .unwrap_or_else(|| panic!("process status has no {name} field"))
}

fn wrapper_effective_placement(root_pid: u64, expected_cpus: &BTreeSet<u32>) -> Value {
    let proc_root = PathBuf::from(format!("/proc/{root_pid}"));
    let status = fs::read_to_string(proc_root.join("status"))
        .expect("read running wrapper process status for effective affinity");
    let affinity = proc_status_field(&status, "Cpus_allowed_list");
    assert_eq!(
        parse_cpuset(affinity),
        *expected_cpus,
        "wrapper root process affinity must equal reserved controller CPUs"
    );
    let cgroup = fs::read_to_string(proc_root.join("cgroup"))
        .expect("read running wrapper cgroup membership");
    let cgroup_path = cgroup
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .expect("wrapper must have cgroup v2 membership");
    let cgroup_dir = Path::new("/sys/fs/cgroup").join(cgroup_path.trim_start_matches('/'));
    let effective_path = cgroup_dir.join("cpuset.cpus.effective");
    let effective = fs::read_to_string(effective_path)
        .expect("read wrapper effective cgroup cpuset")
        .trim()
        .to_owned();
    assert_eq!(
        parse_cpuset(&effective),
        *expected_cpus,
        "wrapper effective cgroup cpuset must equal reserved controller CPUs"
    );
    json!({
        "root_pid": root_pid,
        "proc_status_cpus_allowed_list": affinity,
        "cgroup_v2_path": cgroup_path,
        "cgroup_cpuset_cpus_effective": effective,
        "validation_status": "passed",
    })
}

fn unescape_mountinfo_path(value: &str) -> String {
    let bytes = value.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\\'
            && index + 3 < bytes.len()
            && bytes[index + 1..=index + 3].iter().all(u8::is_ascii_digit)
        {
            let octal = &bytes[index + 1..=index + 3];
            if octal.iter().all(|byte| (b'0'..=b'7').contains(byte)) {
                decoded.push((octal[0] - b'0') * 64 + (octal[1] - b'0') * 8 + octal[2] - b'0');
                index += 4;
                continue;
            }
        }
        decoded.push(bytes[index]);
        index += 1;
    }
    String::from_utf8(decoded).expect("wrapper mountinfo path is UTF-8")
}

fn wrapper_mount_namespace_contract(
    root_pid: u64,
    read_only_mounts: &[CanonicalReadOnlyMount],
    artifact_destination: &str,
    retained_mountinfo_path: &Path,
    rootfs_write_probe: &Value,
) -> Value {
    let mountinfo = fs::read_to_string(format!("/proc/{root_pid}/mountinfo"))
        .expect("read running wrapper mount namespace contract");
    fs::write(retained_mountinfo_path, mountinfo.as_bytes())
        .expect("retain raw wrapper mountinfo before validation");
    let mut effective_mounts = BTreeMap::<String, (u64, String, Vec<String>)>::new();
    for line in mountinfo.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() < 7 {
            continue;
        }
        let mount_id = fields[0].parse::<u64>().expect("numeric wrapper mount ID");
        let mountpoint = unescape_mountinfo_path(fields[4]);
        let options = fields[5].to_owned();
        let separator = fields
            .iter()
            .position(|field| *field == "-")
            .expect("wrapper mountinfo separator");
        let optional_fields = fields[6..separator]
            .iter()
            .map(|field| (*field).to_owned())
            .collect::<Vec<_>>();
        if effective_mounts
            .get(&mountpoint)
            .map_or(true, |(current_id, _, _)| mount_id > *current_id)
        {
            effective_mounts.insert(mountpoint, (mount_id, options, optional_fields));
        }
    }
    let chroot_paths_are_visible = effective_mounts
        .keys()
        .any(|mountpoint| Path::new(mountpoint).starts_with(WRAPPER_CHROOT));
    let runtime_destination = |destination: &str| {
        if chroot_paths_are_visible {
            destination.to_owned()
        } else {
            destination
                .strip_prefix(WRAPPER_CHROOT)
                .unwrap_or_else(|| panic!("wrapper destination is outside chroot: {destination}"))
                .to_owned()
        }
    };
    let runtime_artifact = runtime_destination(artifact_destination);
    let runtime_tracefs = runtime_destination(WRAPPER_TRACEFS);
    let runtime_docker_socket = runtime_destination(WRAPPER_DOCKER_SOCKET);
    let runtime_read_only = read_only_mounts
        .iter()
        .map(|mount| runtime_destination(&mount.destination))
        .collect::<Vec<_>>();
    let runtime_write = [&runtime_artifact, &runtime_tracefs, &runtime_docker_socket];

    if let Some((_, root_options, _)) = effective_mounts.get("/") {
        assert!(
            root_options.split(',').any(|option| option == "ro")
                && !root_options.split(',').any(|option| option == "rw"),
            "visible wrapper root mount must be read-only: {root_options}"
        );
    }
    if chroot_paths_are_visible {
        assert!(
            !effective_mounts.contains_key(WRAPPER_CHROOT),
            "wrapper /host must remain a rootfs skeleton, not a host-root bind"
        );
    }

    let mut exact_read_only_mounts = BTreeSet::new();
    let mut exact_write_mounts = BTreeSet::new();
    let mut recursively_read_only_mounts = 0_u64;
    let mut declared_write_mounts = Vec::new();
    let mut validated_mounts = Vec::new();
    for (mountpoint, (_, options, optional_fields)) in &effective_mounts {
        if mountpoint == "/" {
            continue;
        }
        if chroot_paths_are_visible && !Path::new(mountpoint).starts_with(WRAPPER_CHROOT) {
            continue;
        }
        assert!(
            !optional_fields.iter().any(|field| {
                field == "unbindable"
                    || field.starts_with("shared:")
                    || field.starts_with("master:")
                    || field.starts_with("propagate_from:")
            }),
            "selective chroot mount is not private: {mountpoint} ({optional_fields:?})"
        );
        let options = options.as_str();
        let read_only = options.split(',').any(|option| option == "ro");
        let read_write = options.split(',').any(|option| option == "rw");
        let path = Path::new(mountpoint);
        let declared_write_surface = runtime_write
            .iter()
            .find(|destination| path.starts_with(destination));
        if let Some(destination) = declared_write_surface {
            if read_write {
                declared_write_mounts.push(json!({
                    "destination": mountpoint,
                    "mount_options": options,
                }));
            }
            if mountpoint == *destination {
                assert!(
                    read_write && !read_only,
                    "declared writable mount is not read-write: {mountpoint} ({options})"
                );
                exact_write_mounts.insert(mountpoint.clone());
            }
            validated_mounts.push(json!({
                "destination": mountpoint,
                "mount_options": options,
                "propagation": "private",
                "classification": "declared-write-surface",
            }));
        } else {
            let selected_root = runtime_read_only
                .iter()
                .find(|destination| path.starts_with(destination))
                .unwrap_or_else(|| {
                    panic!("mount outside selective chroot whitelist: {mountpoint} ({options})")
                });
            assert!(
                read_only && !read_write,
                "selective host mount is not recursively read-only: {mountpoint} ({options})"
            );
            if mountpoint == selected_root {
                exact_read_only_mounts.insert(mountpoint.clone());
            }
            recursively_read_only_mounts += 1;
            validated_mounts.push(json!({
                "destination": mountpoint,
                "mount_options": options,
                "propagation": "private",
                "classification": "selective-recursive-read-only",
            }));
        }
    }

    assert_eq!(
        exact_read_only_mounts,
        runtime_read_only.iter().cloned().collect(),
        "every selective read-only mount must be present exactly once"
    );
    assert_eq!(
        exact_write_mounts,
        [
            runtime_artifact.clone(),
            runtime_tracefs.clone(),
            runtime_docker_socket.clone(),
        ]
        .into_iter()
        .collect(),
        "only the exact artifact root, tracefs, and Docker socket mounts are read-write"
    );
    json!({
        "wrapper_rootfs_read_only": true,
        "wrapper_rootfs_live_write_probe": rootfs_write_probe,
        "host_root_bind_present": false,
        "mountinfo_path_view": if chroot_paths_are_visible { "wrapper-root-relative" } else { "chroot-relative" },
        "selective_read_only_mounts": runtime_read_only,
        "recursively_read_only_mount_count": recursively_read_only_mounts,
        "declared_write_mounts": declared_write_mounts,
        "validated_mount_whitelist": validated_mounts,
        "writable_mount_destinations": [runtime_artifact, runtime_tracefs, runtime_docker_socket],
        "host_control_authority": "root-equivalent",
        "authority_reasons": [
            "writable Docker socket",
            "privileged host PID namespace with /proc PID-root traversal",
            "privileged host cgroup namespace",
            "writable tracefs and transient kernel BPF/kprobe state",
        ],
        "validation_status": "passed",
    })
}

fn sha256_file(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(fs::read(path).expect("read retained artifact for SHA-256"))
    )
}

fn write_json(path: &Path, value: &Value) {
    let mut bytes = serde_json::to_vec_pretty(value).expect("serialize retained JSON proof");
    bytes.push(b'\n');
    fs::write(path, bytes).expect("write retained JSON proof");
}

#[derive(Debug)]
enum InspectError {
    Docker(DockerCliError),
    CommandFailed { status: Option<i32> },
    Parse { error: String },
    UnexpectedShape,
}

impl std::fmt::Display for InspectError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Docker(error) => write!(formatter, "{error}"),
            Self::CommandFailed { status } => {
                write!(formatter, "Docker container inspect failed with {status:?}")
            }
            Self::Parse { error } => {
                write!(formatter, "Docker container inspect JSON error: {error}")
            }
            Self::UnexpectedShape => write!(
                formatter,
                "Docker container inspect returned an unexpected result shape"
            ),
        }
    }
}

fn exact_container_not_found(output: &Output, reference: &str) -> bool {
    if output.status.success() {
        return false;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !matches!(stdout.trim(), "" | "[]") {
        return false;
    }
    let Ok(stderr) = std::str::from_utf8(&output.stderr) else {
        return false;
    };
    let stderr = stderr.trim_end_matches(['\r', '\n']);
    stderr == format!("Error: No such container: {reference}")
        || stderr == format!("Error response from daemon: No such container: {reference}")
        || stderr == format!("Error: No such object: {reference}")
}

fn inspect_container(reference: &str, timeout: Duration) -> Result<Option<Value>, InspectError> {
    let output = docker_bounded(
        "container inspect",
        timeout,
        ["container", "inspect", reference],
    )
    .map_err(InspectError::Docker)?;
    if !output.status.success() {
        if exact_container_not_found(&output, reference) {
            return Ok(None);
        }
        return Err(InspectError::CommandFailed {
            status: output.status.code(),
        });
    }
    let value =
        serde_json::from_slice::<Value>(&output.stdout).map_err(|error| InspectError::Parse {
            error: error.to_string(),
        })?;
    let containers = value.as_array().ok_or(InspectError::UnexpectedShape)?;
    if containers.len() != 1 {
        return Err(InspectError::UnexpectedShape);
    }
    Ok(containers.first().cloned())
}

fn safe_wrapper_projection(inspect: &Value) -> Value {
    let mounts = inspect["Mounts"]
        .as_array()
        .expect("wrapper mount inspection")
        .iter()
        .map(|mount| {
            json!({
                "type": mount["Type"],
                "source": mount["Source"],
                "destination": mount["Destination"],
                "mode": mount["Mode"],
                "rw": mount["RW"],
                "propagation": mount["Propagation"],
            })
        })
        .collect::<Vec<_>>();
    json!({
        "id": inspect["Id"],
        "name": inspect["Name"],
        "image_id": inspect["Image"],
        "config": {
            "image": inspect["Config"]["Image"],
            "entrypoint": inspect["Config"]["Entrypoint"],
            "command": inspect["Config"]["Cmd"],
            "user": inspect["Config"]["User"],
            "stop_signal": inspect["Config"]["StopSignal"],
        },
        "host_config": {
            "privileged": inspect["HostConfig"]["Privileged"],
            "pid_mode": inspect["HostConfig"]["PidMode"],
            "cgroup_namespace_mode": inspect["HostConfig"]["CgroupnsMode"],
            "network_mode": inspect["HostConfig"]["NetworkMode"],
            "cpuset_cpus": inspect["HostConfig"]["CpusetCpus"],
            "readonly_rootfs": inspect["HostConfig"]["ReadonlyRootfs"],
        },
        "state": {
            "status": inspect["State"]["Status"],
            "running": inspect["State"]["Running"],
            "exit_code": inspect["State"]["ExitCode"],
            "oom_killed": inspect["State"]["OOMKilled"],
            "error": inspect["State"]["Error"],
        },
        "mounts": mounts,
    })
}

fn mount<'a>(inspect: &'a Value, destination: &str) -> &'a Value {
    inspect["Mounts"]
        .as_array()
        .expect("container mounts")
        .iter()
        .find(|mount| mount["Destination"] == destination)
        .unwrap_or_else(|| panic!("missing wrapper mount at {destination}"))
}

fn wait_for_wrapper_exit(name: &str) -> Result<Value, String> {
    let deadline = Instant::now() + WRAPPER_WAIT;
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Err(format!(
                "wrapper did not exit within {} seconds",
                WRAPPER_WAIT.as_secs()
            ));
        }
        let inspect_timeout = deadline.saturating_duration_since(now).min(DOCKER_WAIT);
        let Some(inspect) = inspect_container(name, inspect_timeout)
            .map_err(|error| format!("inspect wrapper while waiting: {error}"))?
        else {
            return Err("wrapper disappeared before its state could be retained".to_owned());
        };
        if inspect["State"]["Running"] == false {
            return Ok(inspect);
        }
        thread::sleep(Duration::from_millis(200));
    }
}

#[derive(Clone, Debug)]
struct CleanupReport {
    trigger: String,
    wrapper_absent: bool,
    removed_workload_containers: Vec<String>,
    surviving_workload_containers: Vec<String>,
    surviving_probe_containers: Vec<String>,
    issues: Vec<String>,
    scope_verification_complete: bool,
    indeterminate_mutations: Vec<String>,
    indeterminate_settle_required: bool,
    settle_elapsed_ms: u64,
    consecutive_clean_scans: u64,
    attempts: Vec<Value>,
    tracefs_cleanup_required: bool,
    tracefs_cleanup_verified: bool,
    tracefs_snapshot_matches_baseline: bool,
    tracefs_residue_detected: Vec<String>,
}

impl CleanupReport {
    fn verified(&self) -> bool {
        self.wrapper_absent
            && self.surviving_workload_containers.is_empty()
            && self.surviving_probe_containers.is_empty()
            && self.scope_verification_complete
            && self.tracefs_cleanup_verified
    }
}

fn elapsed_millis(start: Instant) -> u64 {
    u64::try_from(start.elapsed().as_millis()).unwrap_or(u64::MAX)
}

fn list_container_ids_for_scope_scan() -> Result<Vec<String>, String> {
    let listed = docker_bounded(
        "list containers for exact output scope scan",
        DOCKER_CLEANUP_WAIT,
        [
            "container",
            "ls",
            "--all",
            "--no-trunc",
            "--format",
            "{{.ID}}",
        ],
    )
    .map_err(|error| error.to_string())?;
    if !listed.status.success() {
        return Err(format!(
            "Docker list profile containers failed with {:?}",
            listed.status.code()
        ));
    }
    let stdout = std::str::from_utf8(&listed.stdout)
        .map_err(|error| format!("Docker list profile containers UTF-8 error: {error}"))?;
    Ok(stdout
        .lines()
        .filter(|line| !line.is_empty())
        .map(str::to_owned)
        .collect())
}

fn container_has_exact_output_mount(
    inspect: &Value,
    canonical_output_dir: &Path,
) -> Result<bool, String> {
    let mounts = inspect["Mounts"]
        .as_array()
        .ok_or_else(|| "container inspection has no Mounts array".to_owned())?;
    for mount in mounts {
        if mount["Destination"] != "/tasktrace-profile-output" {
            continue;
        }
        let source = mount["Source"]
            .as_str()
            .ok_or_else(|| "profile output mount has no string Source".to_owned())?;
        let source = Path::new(source).canonicalize().map_err(|error| {
            format!("cannot canonicalize candidate profile output mount source: {error}")
        })?;
        return Ok(source == canonical_output_dir);
    }
    Ok(false)
}

fn find_scoped_profile_containers(canonical_output_dir: &Path) -> (Vec<String>, Vec<String>) {
    let mut scoped = Vec::new();
    let mut issues = Vec::new();
    let ids = match list_container_ids_for_scope_scan() {
        Ok(ids) => ids,
        Err(error) => return (scoped, vec![error]),
    };
    for id in ids {
        let inspect = match inspect_container(&id, DOCKER_CLEANUP_WAIT) {
            Ok(Some(inspect)) => inspect,
            Ok(None) => continue,
            Err(error) => {
                issues.push(format!("inspect cleanup candidate {id}: {error}"));
                continue;
            }
        };
        match container_has_exact_output_mount(&inspect, canonical_output_dir) {
            Ok(true) => scoped.push(id),
            Ok(false) => {}
            Err(error) => issues.push(format!("classify cleanup candidate {id}: {error}")),
        }
    }
    (scoped, issues)
}

#[derive(Debug)]
struct ScopeObservation {
    wrapper_absent: bool,
    workload_containers: Vec<String>,
    probe_containers: Vec<String>,
    issues: Vec<String>,
}

impl ScopeObservation {
    fn clean(&self) -> bool {
        self.wrapper_absent
            && self.workload_containers.is_empty()
            && self.probe_containers.is_empty()
            && self.issues.is_empty()
    }
}

fn scheduler_kprobe_events(snapshot: &str) -> Result<BTreeSet<String>, String> {
    let mut events = BTreeSet::new();
    for line in snapshot.lines().filter(|line| !line.trim().is_empty()) {
        let descriptor = line
            .split_whitespace()
            .next()
            .ok_or_else(|| "kprobe_events line has no descriptor".to_owned())?;
        let Some(event_name) = descriptor.strip_prefix("p:kprobes/") else {
            continue;
        };
        if event_name.starts_with("p_finish_task_switch") {
            if !event_name
                .bytes()
                .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-'))
            {
                return Err("scheduler kprobe event name contains unsafe bytes".to_owned());
            }
            events.insert(format!("kprobes/{event_name}"));
        }
    }
    Ok(events)
}

struct OwnedDockerScope {
    wrapper_name: String,
    canonical_output_dir: PathBuf,
    artifact_root: PathBuf,
    diagnostic_path: PathBuf,
    verified_report: Option<CleanupReport>,
    indeterminate_mutations: Vec<String>,
    probe_image: String,
    probe_name_prefix: String,
    next_probe_index: u64,
    tracked_probe_names: BTreeSet<String>,
    tracefs_baseline: Option<String>,
    tracefs_cleanup_required: bool,
}

#[derive(Debug)]
struct TracefsCleanupOutcome {
    verified: bool,
    snapshot_matches_baseline: bool,
    residue_detected: Vec<String>,
}

impl OwnedDockerScope {
    fn new(
        wrapper_name: String,
        canonical_output_dir: PathBuf,
        artifact_root: &Path,
        probe_image: String,
        probe_name_prefix: String,
    ) -> Self {
        Self {
            wrapper_name,
            canonical_output_dir,
            artifact_root: artifact_root.to_path_buf(),
            diagnostic_path: artifact_root.join("cleanup_diagnostic.json"),
            verified_report: None,
            indeterminate_mutations: Vec::new(),
            probe_image,
            probe_name_prefix,
            next_probe_index: 0,
            tracked_probe_names: BTreeSet::new(),
            tracefs_baseline: None,
            tracefs_cleanup_required: false,
        }
    }

    fn record_indeterminate_mutation(&mut self, operation: &str, outcome: &str) {
        let entry = format!("{operation}:{outcome}");
        if !self.indeterminate_mutations.contains(&entry) {
            self.indeterminate_mutations.push(entry);
        }
    }

    fn require_tracefs_cleanup(&mut self) {
        self.tracefs_cleanup_required = true;
        self.verified_report = None;
    }

    fn run_tracefs_probe(
        &mut self,
        writable: bool,
        remediation_events: &[String],
    ) -> Result<String, String> {
        self.next_probe_index += 1;
        let probe_name = format!("{}-{}", self.probe_name_prefix, self.next_probe_index);
        match inspect_container(&probe_name, DOCKER_CLEANUP_WAIT) {
            Ok(None) => {}
            Ok(Some(_)) => return Err(format!("tracefs probe name already exists: {probe_name}")),
            Err(error) => return Err(format!("establish tracefs probe name absence: {error}")),
        }
        self.tracked_probe_names.insert(probe_name.clone());
        self.verified_report = None;
        let mount = if writable {
            format!("type=bind,src={HOST_TRACEFS},dst=/tracefs")
        } else {
            format!("type=bind,src={HOST_TRACEFS},dst=/tracefs,readonly")
        };
        let script = if writable {
            "set -eu; for event in \"$@\"; do case \"$event\" in kprobes/p_finish_task_switch*) ;; *) exit 64;; esac; printf -- '-:%s\\n' \"$event\" >> /tracefs/kprobe_events; done"
        } else {
            "set -eu; cat /tracefs/kprobe_events"
        };
        let mut arguments = vec![
            OsString::from("container"),
            OsString::from("create"),
            OsString::from("--name"),
            OsString::from(&probe_name),
            OsString::from("--label"),
            OsString::from("io.tasktrace.test=native-bcc-tracefs-probe"),
            OsString::from("--privileged"),
            OsString::from("--network"),
            OsString::from("none"),
            OsString::from("--read-only"),
            OsString::from("--user"),
            OsString::from("0:0"),
            OsString::from("--mount"),
            OsString::from(mount),
            OsString::from("--entrypoint"),
            OsString::from("/bin/bash"),
            OsString::from(&self.probe_image),
            OsString::from("-c"),
            OsString::from(script),
        ];
        if writable {
            arguments.push(OsString::from("tracefs-remediation"));
            arguments.extend(remediation_events.iter().map(OsString::from));
        }
        let created = docker_bounded("create tracefs probe", DOCKER_CLEANUP_WAIT, &arguments)
            .map_err(|error| {
                self.record_indeterminate_mutation("create tracefs probe", error.kind());
                error.to_string()
            })?;
        if !created.status.success() {
            let outcome = format!("nonzero-status-{:?}", created.status.code());
            self.record_indeterminate_mutation("create tracefs probe", &outcome);
            return Err(format!(
                "create tracefs probe failed with {:?}",
                created.status.code()
            ));
        }
        let inspect = inspect_container(&probe_name, DOCKER_CLEANUP_WAIT)
            .map_err(|error| format!("inspect created tracefs probe: {error}"))?
            .ok_or_else(|| "created tracefs probe became exactly absent".to_owned())?;
        let mounts = inspect["Mounts"]
            .as_array()
            .ok_or_else(|| "tracefs probe inspection has no Mounts array".to_owned())?;
        if mounts.len() != 1
            || mounts[0]["Source"] != HOST_TRACEFS
            || mounts[0]["Destination"] != "/tracefs"
            || mounts[0]["RW"] != writable
            || inspect["HostConfig"]["Privileged"] != true
            || inspect["HostConfig"]["NetworkMode"] != "none"
            || inspect["HostConfig"]["ReadonlyRootfs"] != true
            || inspect["Config"]["User"] != "0:0"
        {
            return Err("tracefs probe isolation inspection failed".to_owned());
        }
        let started = docker_bounded(
            "start attached tracefs probe",
            DOCKER_CLEANUP_WAIT,
            ["container", "start", "--attach", &probe_name],
        )
        .map_err(|error| {
            self.record_indeterminate_mutation("start attached tracefs probe", error.kind());
            error.to_string()
        })?;
        if !started.status.success() {
            let outcome = format!("nonzero-status-{:?}", started.status.code());
            self.record_indeterminate_mutation("start attached tracefs probe", &outcome);
            return Err(format!(
                "tracefs probe exited with {:?}",
                started.status.code()
            ));
        }
        let removed = docker_bounded(
            "remove completed tracefs probe",
            DOCKER_CLEANUP_WAIT,
            ["container", "rm", "--force", &probe_name],
        )
        .map_err(|error| {
            self.record_indeterminate_mutation("remove completed tracefs probe", error.kind());
            error.to_string()
        })?;
        if !removed.status.success() && !exact_container_not_found(&removed, &probe_name) {
            let outcome = format!("nonzero-status-{:?}", removed.status.code());
            self.record_indeterminate_mutation("remove completed tracefs probe", &outcome);
            return Err(format!(
                "remove completed tracefs probe failed with {:?}",
                removed.status.code()
            ));
        }
        match inspect_container(&probe_name, DOCKER_CLEANUP_WAIT) {
            Ok(None) => {
                self.tracked_probe_names.remove(&probe_name);
            }
            Ok(Some(_)) => return Err("completed tracefs probe survived removal".to_owned()),
            Err(error) => {
                return Err(format!("verify completed tracefs probe absence: {error}"));
            }
        }
        String::from_utf8(started.stdout)
            .map_err(|error| format!("tracefs probe output is not UTF-8: {error}"))
    }

    fn capture_tracefs_baseline(&mut self) -> Result<String, String> {
        let snapshot = self.run_tracefs_probe(false, &[])?;
        let scheduler_events = scheduler_kprobe_events(&snapshot)?;
        if !scheduler_events.is_empty() {
            return Err(format!(
                "scheduler kprobe baseline is not clean: {scheduler_events:?}"
            ));
        }
        self.tracefs_baseline = Some(snapshot.clone());
        Ok(snapshot)
    }

    fn verify_and_remediate_tracefs(&mut self, issues: &mut Vec<String>) -> TracefsCleanupOutcome {
        if !self.tracefs_cleanup_required {
            return TracefsCleanupOutcome {
                verified: true,
                snapshot_matches_baseline: true,
                residue_detected: Vec::new(),
            };
        }
        let Some(baseline) = self.tracefs_baseline.clone() else {
            issues.push("tracefs cleanup was required without a retained baseline".to_owned());
            return TracefsCleanupOutcome {
                verified: false,
                snapshot_matches_baseline: false,
                residue_detected: Vec::new(),
            };
        };
        let snapshot = match self.run_tracefs_probe(false, &[]) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                issues.push(format!("read post-wrapper tracefs state: {error}"));
                return TracefsCleanupOutcome {
                    verified: false,
                    snapshot_matches_baseline: false,
                    residue_detected: Vec::new(),
                };
            }
        };
        let post_snapshot_retained = if let Err(error) = fs::write(
            self.artifact_root.join("tracefs_kprobe_events.after.txt"),
            snapshot.as_bytes(),
        ) {
            issues.push(format!("retain post-wrapper tracefs snapshot: {error}"));
            false
        } else {
            true
        };
        let residue = match scheduler_kprobe_events(&snapshot) {
            Ok(events) => events,
            Err(error) => {
                issues.push(format!("parse post-wrapper tracefs state: {error}"));
                return TracefsCleanupOutcome {
                    verified: false,
                    snapshot_matches_baseline: false,
                    residue_detected: Vec::new(),
                };
            }
        };
        let residue_detected = residue.iter().cloned().collect::<Vec<_>>();
        if residue.is_empty() {
            let matches = snapshot == baseline;
            if !matches {
                issues.push("post-wrapper tracefs snapshot differs from baseline".to_owned());
            }
            return TracefsCleanupOutcome {
                verified: matches && post_snapshot_retained,
                snapshot_matches_baseline: matches,
                residue_detected,
            };
        }
        if let Err(error) = self.run_tracefs_probe(true, &residue_detected) {
            issues.push(format!(
                "remove deterministic scheduler kprobe residue: {error}"
            ));
            return TracefsCleanupOutcome {
                verified: false,
                snapshot_matches_baseline: false,
                residue_detected,
            };
        }
        let remediated = match self.run_tracefs_probe(false, &[]) {
            Ok(snapshot) => snapshot,
            Err(error) => {
                issues.push(format!("verify remediated tracefs state: {error}"));
                return TracefsCleanupOutcome {
                    verified: false,
                    snapshot_matches_baseline: false,
                    residue_detected,
                };
            }
        };
        let remediated_snapshot_retained = if let Err(error) = fs::write(
            self.artifact_root
                .join("tracefs_kprobe_events.after_remediation.txt"),
            remediated.as_bytes(),
        ) {
            issues.push(format!("retain remediated tracefs snapshot: {error}"));
            false
        } else {
            true
        };
        let final_events = match scheduler_kprobe_events(&remediated) {
            Ok(events) => events,
            Err(error) => {
                issues.push(format!("parse remediated tracefs state: {error}"));
                return TracefsCleanupOutcome {
                    verified: false,
                    snapshot_matches_baseline: false,
                    residue_detected,
                };
            }
        };
        let matches = final_events.is_empty() && remediated == baseline;
        if !matches {
            issues.push("tracefs state did not return to its exact baseline".to_owned());
        }
        TracefsCleanupOutcome {
            verified: matches && post_snapshot_retained && remediated_snapshot_retained,
            snapshot_matches_baseline: matches,
            residue_detected,
        }
    }

    fn observe(&self) -> ScopeObservation {
        let mut issues = Vec::new();
        let wrapper_absent = match inspect_container(&self.wrapper_name, DOCKER_CLEANUP_WAIT) {
            Ok(None) => true,
            Ok(Some(_)) => false,
            Err(error) => {
                issues.push(format!("verify owned wrapper absence: {error}"));
                false
            }
        };
        let (workload_containers, scan_issues) =
            find_scoped_profile_containers(&self.canonical_output_dir);
        issues.extend(scan_issues);
        let mut probe_containers = Vec::new();
        for probe_name in &self.tracked_probe_names {
            match inspect_container(probe_name, DOCKER_CLEANUP_WAIT) {
                Ok(Some(_)) => probe_containers.push(probe_name.clone()),
                Ok(None) => {}
                Err(error) => issues.push(format!(
                    "verify tracefs probe {probe_name} absence: {error}"
                )),
            }
        }
        ScopeObservation {
            wrapper_absent,
            workload_containers,
            probe_containers,
            issues,
        }
    }

    fn remediate_dirty_scope(
        &mut self,
        workload_containers: &[String],
        probe_containers: &[String],
        removed_workload_containers: &mut BTreeSet<String>,
        issues: &mut Vec<String>,
    ) -> Vec<Value> {
        let mut actions = Vec::new();
        match docker_bounded(
            "stop owned wrapper gracefully",
            DOCKER_CLEANUP_WAIT,
            ["container", "stop", "--time", "5", &self.wrapper_name],
        ) {
            Ok(output) if output.status.success() => actions.push(json!({
                "action": "stop-wrapper-sigterm",
                "outcome": "success",
            })),
            Ok(output) if exact_container_not_found(&output, &self.wrapper_name) => {
                actions.push(json!({
                    "action": "stop-wrapper-sigterm",
                    "outcome": "exact-not-found",
                }));
            }
            Ok(output) => {
                let outcome = format!("nonzero-status-{:?}", output.status.code());
                self.record_indeterminate_mutation("stop owned wrapper gracefully", &outcome);
                issues.push(format!(
                    "graceful stop of owned wrapper returned {:?}",
                    output.status.code()
                ));
                actions.push(json!({
                    "action": "stop-wrapper-sigterm",
                    "outcome": outcome,
                }));
            }
            Err(error) => {
                self.record_indeterminate_mutation("stop owned wrapper gracefully", error.kind());
                issues.push(error.to_string());
                actions.push(json!({
                    "action": "stop-wrapper-sigterm",
                    "outcome": error.kind(),
                }));
            }
        }
        match docker_bounded(
            "remove owned wrapper",
            DOCKER_CLEANUP_WAIT,
            ["container", "rm", "--force", &self.wrapper_name],
        ) {
            Ok(output) if output.status.success() => actions.push(json!({
                "action": "remove-wrapper-force",
                "outcome": "success",
            })),
            Ok(output) if exact_container_not_found(&output, &self.wrapper_name) => {
                actions.push(json!({
                    "action": "remove-wrapper-force",
                    "outcome": "exact-not-found",
                }));
            }
            Ok(output) => {
                let outcome = format!("nonzero-status-{:?}", output.status.code());
                self.record_indeterminate_mutation("remove owned wrapper", &outcome);
                issues.push(format!(
                    "remove owned wrapper returned {:?}",
                    output.status.code()
                ));
                actions.push(json!({
                    "action": "remove-wrapper-force",
                    "outcome": outcome,
                }));
            }
            Err(error) => {
                self.record_indeterminate_mutation("remove owned wrapper", error.kind());
                issues.push(error.to_string());
                actions.push(json!({
                    "action": "remove-wrapper-force",
                    "outcome": error.kind(),
                }));
            }
        }
        for id in workload_containers {
            match docker_bounded(
                "remove output-scoped workload container",
                DOCKER_CLEANUP_WAIT,
                ["container", "rm", "--force", id],
            ) {
                Ok(output) if output.status.success() => {
                    removed_workload_containers.insert(id.clone());
                    actions.push(json!({
                        "action": "remove-output-scoped-workload",
                        "container_id": id,
                        "outcome": "success",
                    }));
                }
                Ok(output) if exact_container_not_found(&output, id) => {
                    removed_workload_containers.insert(id.clone());
                    actions.push(json!({
                        "action": "remove-output-scoped-workload",
                        "container_id": id,
                        "outcome": "exact-not-found",
                    }));
                }
                Ok(output) => {
                    let outcome = format!("nonzero-status-{:?}", output.status.code());
                    self.record_indeterminate_mutation(
                        "remove output-scoped workload container",
                        &outcome,
                    );
                    issues.push(format!(
                        "remove output-scoped workload container {id} returned {:?}",
                        output.status.code()
                    ));
                    actions.push(json!({
                        "action": "remove-output-scoped-workload",
                        "container_id": id,
                        "outcome": outcome,
                    }));
                }
                Err(error) => {
                    self.record_indeterminate_mutation(
                        "remove output-scoped workload container",
                        error.kind(),
                    );
                    issues.push(format!(
                        "remove output-scoped workload container {id}: {error}"
                    ));
                    actions.push(json!({
                        "action": "remove-output-scoped-workload",
                        "container_id": id,
                        "outcome": error.kind(),
                    }));
                }
            }
        }
        for probe_name in probe_containers {
            match docker_bounded(
                "remove owned tracefs probe container",
                DOCKER_CLEANUP_WAIT,
                ["container", "rm", "--force", probe_name],
            ) {
                Ok(output) if output.status.success() => actions.push(json!({
                    "action": "remove-owned-tracefs-probe",
                    "probe_name": probe_name,
                    "outcome": "success",
                })),
                Ok(output) if exact_container_not_found(&output, probe_name) => {
                    actions.push(json!({
                        "action": "remove-owned-tracefs-probe",
                        "probe_name": probe_name,
                        "outcome": "exact-not-found",
                    }));
                }
                Ok(output) => {
                    let outcome = format!("nonzero-status-{:?}", output.status.code());
                    self.record_indeterminate_mutation(
                        "remove owned tracefs probe container",
                        &outcome,
                    );
                    issues.push(format!(
                        "remove owned tracefs probe {probe_name} returned {:?}",
                        output.status.code()
                    ));
                    actions.push(json!({
                        "action": "remove-owned-tracefs-probe",
                        "probe_name": probe_name,
                        "outcome": outcome,
                    }));
                }
                Err(error) => {
                    self.record_indeterminate_mutation(
                        "remove owned tracefs probe container",
                        error.kind(),
                    );
                    issues.push(format!("remove owned tracefs probe {probe_name}: {error}"));
                    actions.push(json!({
                        "action": "remove-owned-tracefs-probe",
                        "probe_name": probe_name,
                        "outcome": error.kind(),
                    }));
                }
            }
        }
        actions
    }

    fn retain_report(&self, report: &CleanupReport) {
        let value = json!({
            "schema_version": "tasktrace-native-bcc-docker-cleanup-diagnostic.v1",
            "trigger": report.trigger,
            "wrapper_name": self.wrapper_name,
            "canonical_output_dir": self.canonical_output_dir,
            "wrapper_absent": report.wrapper_absent,
            "removed_output_scoped_workload_containers": report.removed_workload_containers,
            "surviving_output_scoped_workload_containers": report.surviving_workload_containers,
            "surviving_owned_tracefs_probe_containers": report.surviving_probe_containers,
            "scope_verification_complete": report.scope_verification_complete,
            "verified": report.verified(),
            "issues": report.issues,
            "indeterminate_mutations": report.indeterminate_mutations,
            "indeterminate_settle_required": report.indeterminate_settle_required,
            "settle_elapsed_ms": report.settle_elapsed_ms,
            "consecutive_clean_scans": report.consecutive_clean_scans,
            "attempts": report.attempts,
            "tracefs_cleanup_required": report.tracefs_cleanup_required,
            "tracefs_cleanup_verified": report.tracefs_cleanup_verified,
            "tracefs_snapshot_matches_baseline": report.tracefs_snapshot_matches_baseline,
            "tracefs_residue_detected": report.tracefs_residue_detected,
            "note": "No Docker stdout, stderr, environment, command, or mount inventory is retained here.",
        });
        let retained = serde_json::to_vec_pretty(&value)
            .map_err(|error| error.to_string())
            .and_then(|mut bytes| {
                bytes.push(b'\n');
                fs::write(&self.diagnostic_path, bytes).map_err(|error| error.to_string())
            });
        if let Err(error) = retained {
            eprintln!(
                "unable to retain BCC Docker cleanup diagnostic at {}: {}",
                self.diagnostic_path.display(),
                error
            );
        }
    }

    fn cleanup(&mut self, trigger: &str) -> CleanupReport {
        if let Some(report) = &self.verified_report {
            return report.clone();
        }
        let started = Instant::now();
        let mut issues = Vec::new();
        let mut attempts = Vec::new();
        let mut removed_workload_containers = BTreeSet::new();
        let mut consecutive_clean_scans = 0_u64;
        let mut last_clean_scan = None;
        let mut settle_not_before = (!self.indeterminate_mutations.is_empty())
            .then(|| started + INDETERMINATE_SETTLE_WINDOW);
        let default_deadline = started + INDETERMINATE_SETTLE_WINDOW;
        let final_observation;
        loop {
            let observation = self.observe();
            let observed_at = Instant::now();
            let clean = observation.clean();
            if clean {
                if last_clean_scan.map_or(true, |previous: Instant| {
                    observed_at.saturating_duration_since(previous) >= CLEANUP_SCAN_GAP
                }) {
                    consecutive_clean_scans += 1;
                    last_clean_scan = Some(observed_at);
                }
            } else {
                consecutive_clean_scans = 0;
                last_clean_scan = None;
            }
            let settle_satisfied =
                settle_not_before.map_or(true, |not_before| observed_at >= not_before);
            let verified = clean && consecutive_clean_scans >= 2 && settle_satisfied;
            let mut actions = Vec::new();
            if !clean {
                issues.extend(observation.issues.iter().cloned());
                actions = self.remediate_dirty_scope(
                    &observation.workload_containers,
                    &observation.probe_containers,
                    &mut removed_workload_containers,
                    &mut issues,
                );
                if settle_not_before.is_none() && !self.indeterminate_mutations.is_empty() {
                    settle_not_before = Some(Instant::now() + INDETERMINATE_SETTLE_WINDOW);
                }
            }
            attempts.push(json!({
                "attempt": attempts.len() + 1,
                "elapsed_ms": elapsed_millis(started),
                "wrapper_absent": observation.wrapper_absent,
                "output_scoped_workload_containers": &observation.workload_containers,
                "owned_tracefs_probe_containers": &observation.probe_containers,
                "scan_issues": &observation.issues,
                "clean": clean,
                "consecutive_clean_scans": consecutive_clean_scans,
                "actions": actions,
            }));
            if verified {
                final_observation = observation;
                break;
            }
            let deadline = settle_not_before.unwrap_or(default_deadline);
            if Instant::now() >= deadline {
                final_observation = observation;
                break;
            }
            let remaining = deadline.saturating_duration_since(Instant::now());
            thread::sleep(CLEANUP_SCAN_GAP.min(remaining));
        }
        let mut final_observation = final_observation;
        let tracefs_outcome = if final_observation.clean() {
            self.verify_and_remediate_tracefs(&mut issues)
        } else {
            TracefsCleanupOutcome {
                verified: !self.tracefs_cleanup_required,
                snapshot_matches_baseline: !self.tracefs_cleanup_required,
                residue_detected: Vec::new(),
            }
        };
        let post_trace_observation = self.observe();
        if !post_trace_observation.clean() {
            issues.extend(post_trace_observation.issues.iter().cloned());
            let actions = self.remediate_dirty_scope(
                &post_trace_observation.workload_containers,
                &post_trace_observation.probe_containers,
                &mut removed_workload_containers,
                &mut issues,
            );
            attempts.push(json!({
                "attempt": attempts.len() + 1,
                "elapsed_ms": elapsed_millis(started),
                "phase": "post-tracefs-probe-cleanup",
                "wrapper_absent": post_trace_observation.wrapper_absent,
                "output_scoped_workload_containers": &post_trace_observation.workload_containers,
                "owned_tracefs_probe_containers": &post_trace_observation.probe_containers,
                "scan_issues": &post_trace_observation.issues,
                "clean": false,
                "consecutive_clean_scans": 0,
                "actions": actions,
            }));
            final_observation = self.observe();
            consecutive_clean_scans = 0;
        } else {
            final_observation = post_trace_observation;
        }
        let scope_verification_complete = final_observation.issues.is_empty()
            && final_observation.clean()
            && consecutive_clean_scans >= 2
            && settle_not_before.map_or(true, |not_before| Instant::now() >= not_before);
        let report = CleanupReport {
            trigger: trigger.to_owned(),
            wrapper_absent: final_observation.wrapper_absent,
            removed_workload_containers: removed_workload_containers.into_iter().collect(),
            surviving_workload_containers: final_observation.workload_containers,
            surviving_probe_containers: final_observation.probe_containers,
            issues,
            scope_verification_complete,
            indeterminate_mutations: self.indeterminate_mutations.clone(),
            indeterminate_settle_required: !self.indeterminate_mutations.is_empty(),
            settle_elapsed_ms: elapsed_millis(started),
            consecutive_clean_scans,
            attempts,
            tracefs_cleanup_required: self.tracefs_cleanup_required,
            tracefs_cleanup_verified: tracefs_outcome.verified,
            tracefs_snapshot_matches_baseline: tracefs_outcome.snapshot_matches_baseline,
            tracefs_residue_detected: tracefs_outcome.residue_detected,
        };
        self.retain_report(&report);
        if report.verified() {
            self.verified_report = Some(report.clone());
        }
        report
    }
}

impl Drop for OwnedDockerScope {
    fn drop(&mut self) {
        let _ = self.cleanup("drop");
    }
}

fn panic_after_cleanup(scope: &mut OwnedDockerScope, message: impl std::fmt::Display) -> ! {
    let report = scope.cleanup("explicit-error");
    panic!(
        "{message}; cleanup_verified={}; cleanup_diagnostic={}",
        report.verified(),
        scope.diagnostic_path.display()
    )
}

fn docker_in_scope<I, S>(
    scope: &mut OwnedDockerScope,
    operation: &'static str,
    timeout: Duration,
    arguments: I,
) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    docker_bounded(operation, timeout, arguments)
        .unwrap_or_else(|error| panic_after_cleanup(scope, error))
}

fn docker_mutation_in_scope<I, S>(
    scope: &mut OwnedDockerScope,
    operation: &'static str,
    timeout: Duration,
    arguments: I,
) -> Output
where
    I: IntoIterator<Item = S>,
    S: AsRef<std::ffi::OsStr>,
{
    match docker_bounded(operation, timeout, arguments) {
        Ok(output) if output.status.success() => output,
        Ok(output) => {
            let outcome = format!("nonzero-status-{:?}", output.status.code());
            scope.record_indeterminate_mutation(operation, &outcome);
            let stdout_path = scope
                .artifact_root
                .join("failed_docker_mutation.stdout.txt");
            let stderr_path = scope
                .artifact_root
                .join("failed_docker_mutation.stderr.txt");
            let stdout_retained = fs::write(&stdout_path, &output.stdout).is_ok();
            let stderr_retained = fs::write(&stderr_path, &output.stderr).is_ok();
            panic_after_cleanup(
                scope,
                format!(
                    "Docker {operation} failed with {:?}; stdout_retained={stdout_retained} at {}; stderr_retained={stderr_retained} at {}",
                    output.status.code(),
                    stdout_path.display(),
                    stderr_path.display()
                ),
            )
        }
        Err(error) => {
            scope.record_indeterminate_mutation(operation, error.kind());
            panic_after_cleanup(scope, error)
        }
    }
}

fn assert_success_in_scope(scope: &mut OwnedDockerScope, output: &Output, description: &str) {
    if !output.status.success() {
        panic_after_cleanup(
            scope,
            format!("{description} failed with {:?}", output.status.code()),
        );
    }
}

fn wrapper_rootfs_read_only_probe(
    scope: &mut OwnedDockerScope,
    wrapper_name: &str,
    artifact_root: &Path,
) -> Value {
    let script = r#"set -u
probe=/host/.tasktrace-rootfs-write-probe
if [ ! -d /host ]; then
    echo "wrapper chroot skeleton is absent" >&2
    exit 90
fi
if [ -e "$probe" ] || [ -L "$probe" ]; then
    echo "wrapper rootfs probe path already exists" >&2
    exit 91
fi
if (umask 077; : > "$probe"); then
    rm -f "$probe"
    echo "wrapper rootfs unexpectedly accepted a write" >&2
    exit 92
fi
if [ -e "$probe" ] || [ -L "$probe" ]; then
    echo "failed rootfs write left probe residue" >&2
    exit 93
fi
"#;
    let output = docker_in_scope(
        scope,
        "probe wrapper rootfs read-only state",
        DOCKER_WAIT,
        [
            "container",
            "exec",
            "--user",
            "0:0",
            "--env",
            "LC_ALL=C",
            wrapper_name,
            "/bin/sh",
            "-c",
            script,
        ],
    );
    let stdout_path = artifact_root.join("wrapper_rootfs_probe.stdout.txt");
    let stderr_path = artifact_root.join("wrapper_rootfs_probe.stderr.txt");
    fs::write(&stdout_path, &output.stdout).unwrap_or_else(|error| {
        panic_after_cleanup(scope, format!("retain rootfs probe stdout: {error}"))
    });
    fs::write(&stderr_path, &output.stderr).unwrap_or_else(|error| {
        panic_after_cleanup(scope, format!("retain rootfs probe stderr: {error}"))
    });
    assert_success_in_scope(scope, &output, "privileged wrapper rootfs write probe");
    let stderr = std::str::from_utf8(&output.stderr).unwrap_or_else(|error| {
        panic_after_cleanup(scope, format!("rootfs probe stderr is not UTF-8: {error}"))
    });
    if !stderr.contains("Read-only file system") {
        panic_after_cleanup(
            scope,
            format!(
                "rootfs write probe did not report EROFS; retained stderr: {}",
                stderr_path.display()
            ),
        );
    }
    json!({
        "method": "privileged in-wrapper root write attempt",
        "exec_user": "0:0",
        "probe_path": "/host/.tasktrace-rootfs-write-probe",
        "expected_error": "Read-only file system (EROFS)",
        "exit_code": output.status.code(),
        "stdout_path": stdout_path,
        "stdout_sha256": sha256_file(&stdout_path),
        "stderr_path": stderr_path,
        "stderr_sha256": sha256_file(&stderr_path),
        "effective_write_denied": true,
        "probe_residue_absent": true,
        "wrapper_shell_prerequisite_proven": "/bin/sh executed the probe",
    })
}

fn validate_readback_tree(scope: &mut OwnedDockerScope, readback_dir: &Path, proof_path: &Path) {
    let root_metadata = fs::symlink_metadata(readback_dir).unwrap_or_else(|error| {
        panic_after_cleanup(scope, format!("lstat profile readback root: {error}"))
    });
    if !root_metadata.file_type().is_dir() {
        panic_after_cleanup(scope, "profile readback root is not a real directory");
    }
    let canonical_root = readback_dir.canonicalize().unwrap_or_else(|error| {
        panic_after_cleanup(
            scope,
            format!("canonicalize profile readback root: {error}"),
        )
    });
    if canonical_root != readback_dir {
        panic_after_cleanup(
            scope,
            format!(
                "profile readback root changed path identity: expected {readback_dir:?}, observed {canonical_root:?}"
            ),
        );
    }

    let mut directories = vec![readback_dir.to_path_buf()];
    let mut regular_files = Vec::<(String, u64, u64)>::new();
    while let Some(directory) = directories.pop() {
        let entries = fs::read_dir(&directory).unwrap_or_else(|error| {
            panic_after_cleanup(
                scope,
                format!("read profile readback directory {directory:?}: {error}"),
            )
        });
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic_after_cleanup(scope, format!("read profile readback entry: {error}"))
            });
            let path = entry.path();
            let metadata = fs::symlink_metadata(&path).unwrap_or_else(|error| {
                panic_after_cleanup(
                    scope,
                    format!("lstat profile readback path {path:?}: {error}"),
                )
            });
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                panic_after_cleanup(
                    scope,
                    format!("profile readback contains a symlink: {path:?}"),
                );
            }
            if file_type.is_dir() {
                directories.push(path);
                continue;
            }
            if !file_type.is_file() {
                panic_after_cleanup(
                    scope,
                    format!("profile readback contains a special file: {path:?}"),
                );
            }
            if metadata.nlink() != 1 {
                panic_after_cleanup(
                    scope,
                    format!(
                        "profile readback regular file has {} links instead of one: {path:?}",
                        metadata.nlink()
                    ),
                );
            }
            let canonical = path.canonicalize().unwrap_or_else(|error| {
                panic_after_cleanup(
                    scope,
                    format!("canonicalize profile readback file {path:?}: {error}"),
                )
            });
            if !canonical.starts_with(&canonical_root) {
                panic_after_cleanup(
                    scope,
                    format!("profile readback file escaped its root: {canonical:?}"),
                );
            }
            let relative = canonical
                .strip_prefix(&canonical_root)
                .expect("canonical readback file has validated root prefix")
                .to_string_lossy()
                .into_owned();
            regular_files.push((relative, metadata.len(), metadata.nlink()));
        }
    }
    regular_files.sort();
    write_json(
        proof_path,
        &json!({
            "schema_version": "tasktrace-native-bcc-readback-tree.v1",
            "canonical_root": canonical_root,
            "symlinks_allowed": false,
            "special_files_allowed": false,
            "regular_file_link_count": 1,
            "regular_files": regular_files.iter().map(|(relative_path, size_bytes, link_count)| json!({
                "relative_path": relative_path,
                "size_bytes": size_bytes,
                "link_count": link_count,
            })).collect::<Vec<_>>(),
            "validation_status": "passed",
        }),
    );
}

fn enumerate_profile_readback_sources(
    scope: &mut OwnedDockerScope,
    output_dir: &Path,
    proof_path: &Path,
) -> (Vec<PathBuf>, Vec<PathBuf>) {
    let allowed_fifos = BTreeSet::from([
        "sandbox.start.gate",
        "scheduler.start.gate",
        "scheduler.finish.gate",
    ]);
    let mut pending = vec![output_dir.to_path_buf()];
    let mut directories = Vec::new();
    let mut regular_files = Vec::new();
    let mut skipped_fifos = Vec::new();
    while let Some(directory) = pending.pop() {
        let entries = fs::read_dir(&directory).unwrap_or_else(|error| {
            panic_after_cleanup(
                scope,
                format!("read authoritative profile directory {directory:?}: {error}"),
            )
        });
        for entry in entries {
            let entry = entry.unwrap_or_else(|error| {
                panic_after_cleanup(scope, format!("read authoritative profile entry: {error}"))
            });
            let path = entry.path();
            let relative = path
                .strip_prefix(output_dir)
                .expect("authoritative profile entry is beneath enumerated root")
                .to_path_buf();
            let relative_text = relative.to_str().unwrap_or_else(|| {
                panic_after_cleanup(
                    scope,
                    format!("authoritative profile path is not UTF-8: {relative:?}"),
                )
            });
            if relative_text
                .chars()
                .any(|character| character.is_control())
            {
                panic_after_cleanup(
                    scope,
                    format!("authoritative profile path contains control bytes: {relative:?}"),
                );
            }
            let metadata = fs::symlink_metadata(&path).unwrap_or_else(|error| {
                panic_after_cleanup(
                    scope,
                    format!("lstat authoritative profile path {path:?}: {error}"),
                )
            });
            let file_type = metadata.file_type();
            if file_type.is_symlink() {
                panic_after_cleanup(
                    scope,
                    format!("authoritative profile contains a symlink: {path:?}"),
                );
            }
            if file_type.is_dir() {
                directories.push(relative);
                pending.push(path);
                continue;
            }
            if file_type.is_file() {
                if metadata.nlink() != 1 {
                    panic_after_cleanup(
                        scope,
                        format!(
                            "authoritative profile regular file has {} links instead of one: {path:?}",
                            metadata.nlink()
                        ),
                    );
                }
                regular_files.push(relative);
                continue;
            }
            if file_type.is_fifo() && allowed_fifos.contains(relative_text) {
                skipped_fifos.push(relative_text.to_owned());
                continue;
            }
            panic_after_cleanup(
                scope,
                format!("authoritative profile contains an unapproved special file: {path:?}"),
            );
        }
    }
    directories.sort();
    regular_files.sort();
    skipped_fifos.sort();
    write_json(
        proof_path,
        &json!({
            "schema_version": "tasktrace-native-bcc-readback-source.v1",
            "authoritative_root": output_dir,
            "regular_files_copied_individually": regular_files,
            "directories_recreated_by_harness": directories,
            "approved_fifos_skipped": skipped_fifos,
            "allowed_fifo_paths": allowed_fifos,
            "symlinks_allowed": false,
            "other_special_files_allowed": false,
            "regular_file_link_count": 1,
            "validation_status": "passed",
        }),
    );
    (directories, regular_files)
}

fn copy_profile_for_unprivileged_readback(
    scope: &mut OwnedDockerScope,
    wrapper_name: &str,
    output_dir: &Path,
    artifact_root: &Path,
) -> PathBuf {
    let expected_output = artifact_root.join("profile");
    if output_dir != expected_output {
        panic_after_cleanup(
            scope,
            format!(
                "profile copy source is not the exact artifact output: expected {expected_output:?}, observed {output_dir:?}"
            ),
        );
    }
    let source_metadata = fs::symlink_metadata(output_dir).unwrap_or_else(|error| {
        panic_after_cleanup(
            scope,
            format!("lstat authoritative profile output: {error}"),
        )
    });
    if !source_metadata.file_type().is_dir() {
        panic_after_cleanup(
            scope,
            "authoritative profile output is not a real directory",
        );
    }
    let canonical_output = output_dir.canonicalize().unwrap_or_else(|error| {
        panic_after_cleanup(
            scope,
            format!("canonicalize authoritative profile output: {error}"),
        )
    });
    if canonical_output != expected_output {
        panic_after_cleanup(
            scope,
            format!(
                "authoritative profile output changed path identity: expected {expected_output:?}, observed {canonical_output:?}"
            ),
        );
    }
    let (source_directories, source_regular_files) = enumerate_profile_readback_sources(
        scope,
        output_dir,
        &artifact_root.join("profile_readback_source.json"),
    );
    let readback_dir = artifact_root.join("profile-readback");
    fs::create_dir(&readback_dir).unwrap_or_else(|error| {
        panic_after_cleanup(scope, format!("create profile readback directory: {error}"))
    });
    let stdout_path = artifact_root.join("profile_readback_copy.stdout.txt");
    let stderr_path = artifact_root.join("profile_readback_copy.stderr.txt");
    let mut combined_stdout = Vec::new();
    let mut combined_stderr = Vec::new();
    for relative in source_directories {
        let destination = readback_dir.join(&relative);
        fs::create_dir_all(&destination).unwrap_or_else(|error| {
            panic_after_cleanup(
                scope,
                format!("create profile readback directory {destination:?}: {error}"),
            )
        });
    }
    for relative in source_regular_files {
        let destination = readback_dir.join(&relative);
        match fs::symlink_metadata(&destination) {
            Ok(metadata) => panic_after_cleanup(
                scope,
                format!(
                    "individual profile readback destination unexpectedly exists ({:?}): {destination:?}",
                    metadata.file_type()
                ),
            ),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => panic_after_cleanup(
                scope,
                format!("lstat individual profile readback destination: {error}"),
            ),
        }
        let source_path = format!(
            "{wrapper_name}:{WRAPPER_CHROOT}{}",
            output_dir.join(&relative).display()
        );
        let arguments = [
            OsString::from("container"),
            OsString::from("cp"),
            OsString::from(source_path),
            destination.as_os_str().to_owned(),
        ];
        let copied = docker_in_scope(
            scope,
            "copy one regular profile file for unprivileged readback",
            DOCKER_WAIT,
            &arguments,
        );
        combined_stdout.extend_from_slice(relative.to_string_lossy().as_bytes());
        combined_stdout.push(b'\n');
        combined_stdout.extend_from_slice(&copied.stdout);
        combined_stderr.extend_from_slice(relative.to_string_lossy().as_bytes());
        combined_stderr.push(b'\n');
        combined_stderr.extend_from_slice(&copied.stderr);
        fs::write(&stdout_path, &combined_stdout).unwrap_or_else(|error| {
            panic_after_cleanup(scope, format!("retain profile readback stdout: {error}"))
        });
        fs::write(&stderr_path, &combined_stderr).unwrap_or_else(|error| {
            panic_after_cleanup(scope, format!("retain profile readback stderr: {error}"))
        });
        assert_success_in_scope(
            scope,
            &copied,
            "copy one regular profile file for unprivileged readback",
        );
    }
    validate_readback_tree(
        scope,
        &readback_dir,
        &artifact_root.join("profile_readback_tree.json"),
    );
    fs::read(readback_dir.join("profile_report.json")).unwrap_or_else(|error| {
        panic_after_cleanup(
            scope,
            format!("read copied profile report as the unprivileged harness: {error}"),
        )
    });
    readback_dir
}

fn inspect_required_in_scope(scope: &mut OwnedDockerScope, reference: &str) -> Value {
    match inspect_container(reference, DOCKER_WAIT) {
        Ok(Some(inspect)) => inspect,
        Ok(None) => panic_after_cleanup(
            scope,
            format!("owned container {reference} was exactly absent during inspection"),
        ),
        Err(error) => panic_after_cleanup(scope, error),
    }
}

fn scoped_profile_containers(output_dir: &Path) -> Result<Vec<String>, Vec<String>> {
    let expected_source = output_dir
        .canonicalize()
        .map_err(|error| vec![format!("canonicalize retained BCC profile output: {error}")])?;
    let (scoped, issues) = find_scoped_profile_containers(&expected_source);
    if issues.is_empty() {
        Ok(scoped)
    } else {
        Err(issues)
    }
}

fn parse_cpu_stat(path: &Path) -> std::collections::BTreeMap<String, u64> {
    fs::read_to_string(path)
        .expect("read retained cgroup cpu.stat")
        .lines()
        .map(|line| {
            let mut fields = line.split_whitespace();
            let key = fields.next().expect("cpu.stat field name").to_owned();
            let value = fields
                .next()
                .expect("cpu.stat field value")
                .parse::<u64>()
                .expect("numeric cpu.stat value");
            assert!(fields.next().is_none(), "malformed cpu.stat line: {line}");
            (key, value)
        })
        .collect()
}

fn assert_close(actual: f64, expected: f64, tolerance: f64, description: &str) {
    assert!(
        (actual - expected).abs() <= tolerance,
        "{description}: observed {actual}, expected {expected}, tolerance {tolerance}",
    );
}

fn weighted_width_integral(report: &Value) -> f64 {
    report["time_sec_by_simultaneous_logical_cores"]
        .as_object()
        .expect("scheduler width distribution")
        .iter()
        .map(|(width, seconds)| {
            width.parse::<u32>().expect("numeric scheduler width") as f64
                * seconds.as_f64().expect("numeric time at scheduler width")
        })
        .sum()
}

fn sum_object_values(value: &Value, description: &str) -> f64 {
    value
        .as_object()
        .unwrap_or_else(|| panic!("{description} is an object"))
        .values()
        .map(|value| {
            value
                .as_f64()
                .unwrap_or_else(|| panic!("{description} contains numeric values"))
        })
        .sum()
}

fn artifact_readback_path(
    authoritative_output: &Path,
    readable_output: &Path,
    artifact: &Value,
) -> PathBuf {
    let reported_candidate = artifact["path"]
        .as_str()
        .map(PathBuf::from)
        .unwrap_or_else(|| {
            authoritative_output.join(
                artifact["relative_path"]
                    .as_str()
                    .expect("artifact has a path or relative_path"),
            )
        });
    let canonical_authoritative = authoritative_output
        .canonicalize()
        .expect("canonicalize authoritative profile output directory");
    let canonical_reported = reported_candidate
        .canonicalize()
        .expect("canonicalize authoritative artifact path");
    assert!(
        canonical_reported.starts_with(&canonical_authoritative),
        "reported artifact path escaped canonical authoritative output: {}",
        canonical_reported.display()
    );
    let relative = canonical_reported
        .strip_prefix(&canonical_authoritative)
        .expect("reported artifact has validated authoritative prefix");
    let canonical_readable = readable_output
        .canonicalize()
        .expect("canonicalize profile readback directory");
    let readable_candidate = readable_output
        .join(relative)
        .canonicalize()
        .expect("canonicalize copied artifact path");
    assert!(
        readable_candidate.starts_with(canonical_readable),
        "copied artifact path escaped canonical readback output: {}",
        readable_candidate.display()
    );
    readable_candidate
}

fn assert_reported_artifact_paths_confined(value: &Value, output_dir: &Path) {
    let canonical_output = output_dir
        .canonicalize()
        .expect("canonicalize profile output for artifact confinement");
    fn assert_confined(path: &str, canonical_output: &Path) {
        let path = Path::new(path);
        let candidate = if path.is_absolute() {
            path.to_path_buf()
        } else {
            canonical_output.join(path)
        };
        let canonical = candidate
            .canonicalize()
            .expect("canonicalize reported artifact path");
        assert!(
            canonical.starts_with(canonical_output),
            "reported artifact path escaped output: {}",
            canonical.display()
        );
    }
    fn walk(value: &Value, canonical_output: &Path) {
        match value {
            Value::Array(values) => {
                for value in values {
                    walk(value, canonical_output);
                }
            }
            Value::Object(object) => {
                let artifact_descriptor = object.contains_key("artifact_id")
                    || (object.contains_key("sha256") && object.contains_key("size_bytes"));
                if artifact_descriptor {
                    if let Some(path) = object
                        .get("path")
                        .or_else(|| object.get("relative_path"))
                        .and_then(Value::as_str)
                    {
                        assert_confined(path, canonical_output);
                    }
                }
                for (key, value) in object {
                    if matches!(
                        key.as_str(),
                        "events_path" | "raw_trace_path" | "report_path"
                    ) {
                        if let Some(path) = value.as_str() {
                            assert_confined(path, canonical_output);
                        }
                    }
                    walk(value, canonical_output);
                }
            }
            _ => {}
        }
    }
    walk(value, &canonical_output);
}

#[test]
#[ignore = "requires a privileged BCC/libbcc wrapper, host PID and cgroup namespaces, Docker socket access, immutable preloaded images, and disjoint reserved controller/worker CPUs"]
fn native_bcc_scheduler_is_scope_exact_loss_aware_reconciled_and_clean() {
    let wrapper_image = env::var("TASKTRACE_RUST_BCC_WRAPPER_IMAGE")
        .expect("set TASKTRACE_RUST_BCC_WRAPPER_IMAGE to a preloaded immutable BCC image");
    let controller_cpuset = env::var("TASKTRACE_RUST_BCC_CONTROLLER_CPUSET")
        .expect("set TASKTRACE_RUST_BCC_CONTROLLER_CPUSET to reserved controller CPUs");
    let task_image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable task image");
    let worker_cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to reserved worker CPUs");
    let configured_artifact_root = PathBuf::from(
        env::var_os("TASKTRACE_RUST_BCC_ARTIFACT_DIR")
            .expect("set TASKTRACE_RUST_BCC_ARTIFACT_DIR to a new direct child of /tmp"),
    );

    assert!(
        immutable_digest(&wrapper_image),
        "BCC wrapper image must be an immutable image ID or repo digest"
    );
    assert!(
        immutable_digest(&task_image),
        "task image must be an immutable image ID or repo digest"
    );
    let controller_cpus = parse_cpuset(&controller_cpuset);
    let worker_cpus = parse_cpuset(&worker_cpuset);
    assert!(
        worker_cpus.len() >= 2,
        "BCC peak gate requires at least two worker CPUs"
    );
    assert!(
        controller_cpus.is_disjoint(&worker_cpus),
        "controller and worker CPU sets must be disjoint"
    );
    assert_eq!(
        configured_artifact_root.parent(),
        Some(Path::new("/tmp")),
        "artifact root must be a new direct child of /tmp"
    );
    let artifact_basename = configured_artifact_root
        .file_name()
        .and_then(|name| name.to_str())
        .expect("artifact root basename must be nonempty UTF-8");
    assert!(
        !artifact_basename
            .chars()
            .any(|character| character == ',' || character.is_control()),
        "artifact root basename must not contain commas or control characters"
    );
    fs::create_dir(&configured_artifact_root)
        .expect("atomically create caller-owned BCC artifact root");
    let artifact_root = configured_artifact_root
        .canonicalize()
        .expect("canonicalize BCC artifact root");
    assert!(artifact_root.starts_with("/tmp"));
    let output_dir = artifact_root.join("profile");
    let bcc_home = artifact_root.join("bcc-home");
    let docker_config = artifact_root.join("empty-docker-config");
    let controller_dir = artifact_root.join("controller");
    fs::create_dir(&bcc_home).expect("create isolated BCC HOME/TMPDIR");
    fs::create_dir(&docker_config).expect("create empty isolated Docker config directory");
    fs::create_dir(&controller_dir).expect("create artifact-owned controller directory");
    let tracefs_contract = tracefs_mount_contract(Path::new(HOST_TRACEFS));
    let read_only_mounts = canonical_read_only_mounts();
    let artifact_destination = format!("{WRAPPER_CHROOT}{}", artifact_root.display());
    assert_eq!(
        artifact_destination,
        format!("{WRAPPER_CHROOT}/tmp/{artifact_basename}"),
        "exact artifact mount preserves its absolute path inside the chroot"
    );
    let mut selected_sources = read_only_mounts
        .iter()
        .map(|mount| {
            filesystem_canonicalized_overlap_source(
                format!("selective-read-only:{}", mount.requested_source),
                mount.source.clone(),
                mount.destination.clone(),
            )
        })
        .collect::<Vec<_>>();
    selected_sources.extend([
        filesystem_canonicalized_overlap_source(
            "exact-artifact-root-read-write".to_owned(),
            artifact_root.clone(),
            artifact_destination.clone(),
        ),
        tracefs_mountinfo_overlap_source(&tracefs_contract, WRAPPER_TRACEFS),
        filesystem_canonicalized_overlap_source(
            "docker-socket-read-write".to_owned(),
            PathBuf::from(DOCKER_SOCKET),
            WRAPPER_DOCKER_SOCKET.to_owned(),
        ),
    ]);
    let docker_root_contract = docker_root_dir_contract(
        &selected_sources,
        &artifact_root.join("docker_root_dir_preflight.json"),
    );

    let source_controller = binary()
        .canonicalize()
        .expect("canonicalize Cargo-built public tasktrace binary");
    let source_controller_sha256 = sha256_file(&source_controller);
    let controller = controller_dir.join("tasktrace");
    fs::copy(&source_controller, &controller)
        .expect("copy Cargo-built controller to same-path host/chroot location");
    fs::set_permissions(&controller, fs::Permissions::from_mode(0o555))
        .expect("make copied controller read-only and executable");
    let controller = controller
        .canonicalize()
        .expect("canonicalize copied controller binary");
    let controller_sha256 = sha256_file(&controller);
    assert_eq!(
        controller_sha256, source_controller_sha256,
        "copied controller digest must match Cargo-built binary"
    );
    assert_eq!(
        controller.metadata().unwrap().permissions().mode() & 0o777,
        0o555,
        "copied controller mode"
    );
    let wrapper_image_id = image_id(&wrapper_image);
    let task_image_id = image_id(&task_image);
    let worker_count = worker_cpus.len().min(8);
    let workload = format!(
        r#"python3 - <<'PY'
import os
import time

deadline = time.monotonic() + 3.0
pids = []
for worker in range({worker_count}):
    pid = os.fork()
    if pid == 0:
        value = worker + 1
        while time.monotonic() < deadline:
            value = (value * 1664525 + 1013904223) & 0xffffffff
        os._exit(0)
    pids.append(pid)
for pid in pids:
    _, status = os.waitpid(pid, 0)
    if status != 0:
        raise SystemExit(1)
print("tasktrace-native-bcc-parallel-ok", len(pids))
PY"#
    );

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock after Unix epoch")
        .as_nanos();
    let wrapper_name = format!("tasktrace-bcc-e2e-{}-{nonce}", std::process::id());
    let probe_name_prefix = format!("tasktrace-bcc-probe-{}-{nonce}", std::process::id());
    let scope_hash = format!(
        "{:x}",
        Sha256::digest(artifact_root.as_os_str().as_encoded_bytes())
    );
    let scope_label = format!("io.tasktrace.test.scope={}", &scope_hash[..20]);

    write_json(
        &artifact_root.join("test_contract.json"),
        &json!({
            "schema_version": "tasktrace-native-bcc-docker-e2e.v2",
            "docker_daemon_root": docker_root_contract,
            "wrapper": {
                "image_reference": wrapper_image,
                "image_id": wrapper_image_id,
                "controller_cpuset": controller_cpuset,
                "privileged": true,
                "pid_namespace": "host",
                "cgroup_namespace": "host",
                "network": "none",
                "read_only_rootfs": true,
                "entrypoint": "/usr/sbin/chroot",
                "required_wrapper_executables": ["/usr/sbin/chroot", "/bin/sh"],
                "wrapper_executable_validation": "chroot is exercised by container start; /bin/sh is exercised by the live rootfs write probe",
                "chroot": WRAPPER_CHROOT,
                "host_root_mount_present": false,
                "chroot_skeleton": "wrapper image rootfs, read-only",
                "selective_host_mounts": read_only_mounts.iter().map(|mount| json!({
                    "requested_source": mount.requested_source,
                    "canonical_source": mount.source,
                    "destination": mount.destination,
                    "read_only": true,
                    "recursive_read_only": true,
                    "propagation": "rprivate",
                })).collect::<Vec<_>>(),
                "host_tracefs_mount": {
                    "source": HOST_TRACEFS,
                    "destination": WRAPPER_TRACEFS,
                    "read_only": false,
                    "preflight": tracefs_contract,
                    "reason": "narrow writable host tracefs surface required for BCC attach/detach",
                },
                "exact_artifact_mount": {
                    "source": artifact_root,
                    "destination": artifact_destination,
                    "read_only": false,
                    "propagation": "rprivate",
                    "full_host_tmp_mounted": false,
                },
                "docker_socket_mount": {
                    "source": DOCKER_SOCKET,
                    "destination_before_chroot": WRAPPER_DOCKER_SOCKET,
                    "destination_inside_chroot": CHROOT_DOCKER_SOCKET,
                    "docker_host": format!("unix://{CHROOT_DOCKER_SOCKET}"),
                    "read_only": false,
                    "propagation": "rprivate",
                },
                "security_boundary": {
                    "root_equivalent_host_root_visibility": true,
                    "direct_host_root_bind_present": false,
                    "selective_mounts_are_not_a_security_boundary": true,
                    "host_control_authority": "root-equivalent",
                    "authority_reasons": [
                        "the writable Docker socket can create privileged host bind mounts",
                        "host-PID /proc exposes /proc/<pid>/root traversal into host process roots",
                        "privileged host PID/cgroup namespaces expose host process and cgroup control",
                        "writable tracefs and transient kernel BPF/kprobe state affect the host kernel",
                    ],
                    "proc_pid_root_disclosure": "the read-only /proc bind plus host PID namespace can expose paths outside the selective chroot through /proc/<pid>/root; read-only bind flags are not an authorization boundary",
                    "docker_socket_disclosure": "write access to the Docker socket is root-equivalent host authority regardless of mount visibility",
                    "declared_direct_write_surfaces": [artifact_root, DOCKER_SOCKET, HOST_TRACEFS, "transient kernel BPF/kprobe state"],
                    "intentionally_not_mounted": ["/", "/var tree (the Docker socket file is the sole /var source)", "/home", "/workspace", "/tmp (except the exact artifact root)"],
                    "suitable_only_for_a_dedicated_test_host": true,
                },
            },
            "workload": {
                "image_reference": task_image,
                "image_id": task_image_id,
                "worker_cpuset": worker_cpuset,
                "parallel_workers": worker_count,
                "network_policy": "no-network",
                "command": workload,
            },
            "controller": {
                "cargo_binary_path": source_controller,
                "cargo_binary_sha256": source_controller_sha256,
                "host_and_chroot_path": controller,
                "wrapper_pre_chroot_path": format!("{WRAPPER_CHROOT}{}", controller.display()),
                "sha256": controller_sha256,
                "mode": "0555",
                "same_absolute_path_for_controller_and_host_docker_daemon": true,
                "legacy_python": "/tasktrace-test/python-fallback-must-not-run",
            },
            "artifact_root": artifact_root,
            "profile_output": output_dir,
            "wrapper_name": wrapper_name,
            "tracefs_probe": {
                "name_prefix": probe_name_prefix,
                "image_reference": task_image,
                "privileged": true,
                "network": "none",
                "read_only_rootfs": true,
                "only_bind_mount": HOST_TRACEFS,
                "baseline_read_only": true,
                "remediation_read_write_only_when_residue_is_detected": true,
            },
            "scope_label": scope_label,
        }),
    );

    let controller_text = controller.to_string_lossy().into_owned();
    let output_text = output_dir.to_string_lossy().into_owned();
    let bcc_home_text = bcc_home.to_string_lossy().into_owned();
    let docker_config_text = docker_config.to_string_lossy().into_owned();
    let mut create_arguments = vec![
        OsString::from("container"),
        OsString::from("create"),
        OsString::from("--name"),
        OsString::from(&wrapper_name),
        OsString::from("--stop-signal"),
        OsString::from("SIGTERM"),
        OsString::from("--label"),
        OsString::from("io.tasktrace.test=native-bcc-e2e"),
        OsString::from("--label"),
        OsString::from(&scope_label),
        OsString::from("--privileged"),
        OsString::from("--pid"),
        OsString::from("host"),
        OsString::from("--cgroupns"),
        OsString::from("host"),
        OsString::from("--network"),
        OsString::from("none"),
        OsString::from("--cpuset-cpus"),
        OsString::from(&controller_cpuset),
        OsString::from("--user"),
        OsString::from("0:0"),
        OsString::from("--read-only"),
        OsString::from("--env"),
        OsString::from(format!("DOCKER_HOST=unix://{CHROOT_DOCKER_SOCKET}")),
        OsString::from("--env"),
        OsString::from("TASKTRACE_LEGACY_PYTHON=/tasktrace-test/python-fallback-must-not-run"),
        OsString::from("--env"),
        OsString::from(format!("HOME={bcc_home_text}")),
        OsString::from("--env"),
        OsString::from(format!("TMPDIR={bcc_home_text}")),
        OsString::from("--env"),
        OsString::from(format!("DOCKER_CONFIG={docker_config_text}")),
        OsString::from("--env"),
        OsString::from("PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"),
    ];
    for mount in &read_only_mounts {
        create_arguments.extend([
            OsString::from("--mount"),
            OsString::from(format!(
                "type=bind,src={},dst={},readonly,bind-propagation=rprivate,bind-recursive=readonly",
                mount.source.display(),
                mount.destination,
            )),
        ]);
    }
    create_arguments.extend([
        OsString::from("--mount"),
        OsString::from(format!(
            "type=bind,src={HOST_TRACEFS},dst={WRAPPER_TRACEFS},bind-propagation=rprivate"
        )),
        OsString::from("--mount"),
        OsString::from(format!(
            "type=bind,src={},dst={artifact_destination},bind-propagation=rprivate",
            artifact_root.display()
        )),
        OsString::from("--mount"),
        OsString::from(format!(
            "type=bind,src={DOCKER_SOCKET},dst={WRAPPER_DOCKER_SOCKET},bind-propagation=rprivate"
        )),
    ]);
    create_arguments.extend([
        OsString::from("--entrypoint"),
        OsString::from("/usr/sbin/chroot"),
        OsString::from(&wrapper_image),
        OsString::from(WRAPPER_CHROOT),
        OsString::from(&controller_text),
        OsString::from("profile"),
        OsString::from("container"),
        OsString::from("--image"),
        OsString::from(&task_image),
        OsString::from("--cpuset-cpus"),
        OsString::from(&worker_cpuset),
        OsString::from("--workdir"),
        OsString::from("/tmp"),
        OsString::from("--network-policy"),
        OsString::from("no-network"),
        OsString::from("--strace-mode"),
        OsString::from("none"),
        OsString::from("--profile-set"),
        OsString::from("cpu-peak"),
        OsString::from("--scheduler-concurrency"),
        OsString::from("--scheduler-backend"),
        OsString::from("bcc"),
        OsString::from("--scheduler-bcc-buffer-pages"),
        OsString::from("64"),
        OsString::from("--scheduler-attach-timeout-sec"),
        OsString::from("30"),
        OsString::from("--command"),
        OsString::from(&workload),
        OsString::from("--output-dir"),
        OsString::from(&output_text),
    ]);

    match inspect_container(&wrapper_name, DOCKER_WAIT) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("unique wrapper name unexpectedly exists before create"),
        Err(error) => panic!("cannot establish wrapper-name absence before create: {error}"),
    }
    // Arm the one scope guard before create so even an indeterminate CLI
    // timeout is cleaned. The name was just proven absent; a successful create
    // therefore returns with the owned wrapper already guarded, before writes.
    let mut wrapper = OwnedDockerScope::new(
        wrapper_name.clone(),
        output_dir.clone(),
        &artifact_root,
        task_image.clone(),
        probe_name_prefix,
    );
    let tracefs_baseline = wrapper
        .capture_tracefs_baseline()
        .unwrap_or_else(|error| panic_after_cleanup(&mut wrapper, error));
    fs::write(
        artifact_root.join("tracefs_kprobe_events.before.txt"),
        tracefs_baseline.as_bytes(),
    )
    .expect("retain clean tracefs kprobe baseline");
    let created = docker_mutation_in_scope(
        &mut wrapper,
        "create native BCC wrapper",
        DOCKER_WAIT,
        &create_arguments,
    );
    fs::write(
        artifact_root.join("wrapper_create.stdout.txt"),
        &created.stdout,
    )
    .expect("retain wrapper create stdout");
    fs::write(
        artifact_root.join("wrapper_create.stderr.txt"),
        &created.stderr,
    )
    .expect("retain wrapper create stderr");

    let created_inspect = inspect_required_in_scope(&mut wrapper, &wrapper_name);
    write_json(
        &artifact_root.join("wrapper_inspect.created.json"),
        &safe_wrapper_projection(&created_inspect),
    );
    assert_eq!(created_inspect["Image"], wrapper_image_id);
    assert_eq!(created_inspect["Config"]["User"], "0:0");
    assert_eq!(created_inspect["Config"]["StopSignal"], "SIGTERM");
    assert_eq!(created_inspect["HostConfig"]["Privileged"], true);
    assert_eq!(created_inspect["HostConfig"]["PidMode"], "host");
    assert_eq!(created_inspect["HostConfig"]["CgroupnsMode"], "host");
    assert_eq!(created_inspect["HostConfig"]["NetworkMode"], "none");
    assert_eq!(created_inspect["HostConfig"]["ReadonlyRootfs"], true);
    assert_eq!(
        created_inspect["HostConfig"]["CpusetCpus"],
        controller_cpuset
    );
    assert_eq!(
        created_inspect["Config"]["Entrypoint"],
        json!(["/usr/sbin/chroot"])
    );
    assert_eq!(created_inspect["Config"]["Cmd"][0], WRAPPER_CHROOT);
    assert_eq!(created_inspect["Config"]["Cmd"][1], controller_text);
    assert!(
        created_inspect["Config"]["Env"]
            .as_array()
            .expect("wrapper environment inspection")
            .iter()
            .any(|value| value
                == &Value::String(format!("DOCKER_HOST=unix://{CHROOT_DOCKER_SOCKET}"))),
        "wrapper Docker client uses the socket path visible inside the chroot"
    );
    let inspected_mounts = created_inspect["Mounts"]
        .as_array()
        .expect("wrapper mount inspection");
    let actual_mount_destinations = inspected_mounts
        .iter()
        .map(|value| {
            value["Destination"]
                .as_str()
                .expect("wrapper mount destination is a string")
                .to_owned()
        })
        .collect::<BTreeSet<_>>();
    let expected_mount_destinations = read_only_mounts
        .iter()
        .map(|mount| mount.destination.clone())
        .chain([
            WRAPPER_TRACEFS.to_owned(),
            artifact_destination.clone(),
            WRAPPER_DOCKER_SOCKET.to_owned(),
        ])
        .collect::<BTreeSet<_>>();
    assert_eq!(
        inspected_mounts.len(),
        expected_mount_destinations.len(),
        "wrapper inspect must not contain duplicate or additional mounts"
    );
    assert_eq!(
        actual_mount_destinations, expected_mount_destinations,
        "wrapper exposes exactly the selective mount whitelist"
    );
    assert!(
        !actual_mount_destinations.contains(WRAPPER_CHROOT),
        "host root must not be bound at /host"
    );
    assert!(
        !actual_mount_destinations.contains("/host/tmp"),
        "full host /tmp must not be mounted"
    );
    for expected in &read_only_mounts {
        let inspected = mount(&created_inspect, &expected.destination);
        assert_eq!(
            inspected["Source"],
            expected.source.to_string_lossy().as_ref()
        );
        assert_eq!(inspected["RW"], false);
        assert_eq!(inspected["Propagation"], "rprivate");
    }
    let tracefs_mount = mount(&created_inspect, WRAPPER_TRACEFS);
    assert_eq!(tracefs_mount["Source"], HOST_TRACEFS);
    assert_eq!(tracefs_mount["RW"], true);
    assert_eq!(tracefs_mount["Propagation"], "rprivate");
    let artifact_mount = mount(&created_inspect, &artifact_destination);
    assert_eq!(
        artifact_mount["Source"],
        artifact_root.to_string_lossy().as_ref()
    );
    assert_eq!(artifact_mount["RW"], true);
    assert_eq!(artifact_mount["Propagation"], "rprivate");
    let socket_mount = mount(&created_inspect, WRAPPER_DOCKER_SOCKET);
    assert_eq!(socket_mount["Source"], DOCKER_SOCKET);
    assert_eq!(socket_mount["RW"], true);
    assert_eq!(socket_mount["Propagation"], "rprivate");

    wrapper.require_tracefs_cleanup();
    let started = docker_mutation_in_scope(
        &mut wrapper,
        "start native BCC wrapper",
        DOCKER_WAIT,
        ["container", "start", &wrapper_name],
    );
    fs::write(
        artifact_root.join("wrapper_start.stdout.txt"),
        &started.stdout,
    )
    .expect("retain wrapper start stdout");
    fs::write(
        artifact_root.join("wrapper_start.stderr.txt"),
        &started.stderr,
    )
    .expect("retain wrapper start stderr");
    let running_inspect = inspect_required_in_scope(&mut wrapper, &wrapper_name);
    assert_eq!(
        running_inspect["State"]["Running"], true,
        "wrapper must remain running long enough to validate effective placement"
    );
    let wrapper_root_pid = running_inspect["State"]["Pid"]
        .as_u64()
        .expect("running wrapper has a numeric host PID");
    assert!(wrapper_root_pid > 0, "running wrapper host PID is positive");
    let wrapper_placement = wrapper_effective_placement(wrapper_root_pid, &controller_cpus);
    let wrapper_rootfs_probe =
        wrapper_rootfs_read_only_probe(&mut wrapper, &wrapper_name, &artifact_root);
    let wrapper_mount_contract = wrapper_mount_namespace_contract(
        wrapper_root_pid,
        &read_only_mounts,
        &artifact_destination,
        &artifact_root.join("wrapper.mountinfo.txt"),
        &wrapper_rootfs_probe,
    );
    write_json(
        &artifact_root.join("wrapper_effective_runtime_contract.json"),
        &json!({
            "placement": wrapper_placement,
            "mount_namespace": wrapper_mount_contract,
        }),
    );

    let finished_inspect = match wait_for_wrapper_exit(&wrapper_name) {
        Ok(inspect) => inspect,
        Err(error) => {
            match docker_bounded(
                "read wrapper logs after wait failure",
                DOCKER_CLEANUP_WAIT,
                ["container", "logs", &wrapper_name],
            ) {
                Ok(logs) => {
                    let _ = fs::write(artifact_root.join("wrapper.stdout.txt"), &logs.stdout);
                    let _ = fs::write(artifact_root.join("wrapper.stderr.txt"), &logs.stderr);
                }
                Err(log_error) => {
                    let _ = fs::write(
                        artifact_root.join("wrapper_log_capture_error.txt"),
                        format!("{log_error}\n"),
                    );
                }
            }
            panic_after_cleanup(&mut wrapper, error)
        }
    };
    let logs = docker_in_scope(
        &mut wrapper,
        "read native BCC wrapper logs",
        DOCKER_WAIT,
        ["container", "logs", &wrapper_name],
    );
    fs::write(artifact_root.join("wrapper.stdout.txt"), &logs.stdout)
        .expect("retain wrapper stdout");
    fs::write(artifact_root.join("wrapper.stderr.txt"), &logs.stderr)
        .expect("retain wrapper stderr");
    assert_success_in_scope(&mut wrapper, &logs, "read native BCC wrapper logs");
    write_json(
        &artifact_root.join("wrapper_inspect.finished.json"),
        &safe_wrapper_projection(&finished_inspect),
    );
    let wrapper_exit = finished_inspect["State"]["ExitCode"].as_i64();
    let wrapper_oom = finished_inspect["State"]["OOMKilled"]
        .as_bool()
        .expect("finished wrapper inspection has a boolean OOMKilled field");
    let wrapper_running = finished_inspect["State"]["Running"]
        .as_bool()
        .expect("finished wrapper inspection has a boolean Running field");
    if wrapper_running || wrapper_exit != Some(0) || wrapper_oom {
        panic_after_cleanup(
            &mut wrapper,
            format!(
                "native BCC wrapper failed before evidence readback: running={wrapper_running}, exit={wrapper_exit:?}, oom={wrapper_oom}\nstdout:\n{}\nstderr:\n{}",
                String::from_utf8_lossy(&logs.stdout),
                String::from_utf8_lossy(&logs.stderr),
            ),
        );
    }
    let pre_readback_survivors = scoped_profile_containers(&output_dir).unwrap_or_else(|errors| {
        panic_after_cleanup(
            &mut wrapper,
            format!("cannot establish quiescent profile output before readback: {errors:?}"),
        )
    });
    if !pre_readback_survivors.is_empty() {
        panic_after_cleanup(
            &mut wrapper,
            format!(
                "output-scoped workload containers remained before readback: {pre_readback_survivors:?}"
            ),
        );
    }
    write_json(
        &artifact_root.join("profile_readback_quiescence.json"),
        &json!({
            "schema_version": "tasktrace-native-bcc-readback-quiescence.v1",
            "wrapper_running": wrapper_running,
            "wrapper_exit_code": wrapper_exit,
            "wrapper_oom_killed": wrapper_oom,
            "output_scoped_workload_containers": pre_readback_survivors,
            "validation_status": "passed",
        }),
    );
    let readable_output_dir = copy_profile_for_unprivileged_readback(
        &mut wrapper,
        &wrapper_name,
        &output_dir,
        &artifact_root,
    );
    let cleanup_report = wrapper.cleanup("normal");
    assert!(
        cleanup_report.verified(),
        "owned native BCC scope cleanup was not verified: {cleanup_report:?}"
    );
    assert!(
        cleanup_report.tracefs_residue_detected.is_empty(),
        "BCC left deterministic tracefs kprobe residue that required remediation: {:?}",
        cleanup_report.tracefs_residue_detected
    );
    match inspect_container(&wrapper_name, DOCKER_CLEANUP_WAIT) {
        Ok(None) => {}
        Ok(Some(_)) => panic!("owned native BCC wrapper survived explicit cleanup"),
        Err(error) => panic!("could not verify owned native BCC wrapper cleanup: {error}"),
    }
    let labeled = docker_bounded(
        "list scope-labeled wrappers after cleanup",
        DOCKER_CLEANUP_WAIT,
        [
            "container",
            "ls",
            "--all",
            "--filter",
            &format!("label={scope_label}"),
            "--format",
            "{{.ID}}",
        ],
    )
    .unwrap_or_else(|error| panic!("{error}"));
    assert_success(&labeled, "list output-scoped BCC wrappers after cleanup");
    assert!(
        String::from_utf8_lossy(&labeled.stdout).trim().is_empty(),
        "output-scoped BCC wrapper remained after cleanup: {}",
        String::from_utf8_lossy(&labeled.stdout)
    );

    let retained_report = fs::read_to_string(readable_output_dir.join("profile_report.json")).ok();
    assert!(!wrapper_oom, "native BCC wrapper was OOM-killed");
    assert_eq!(
        wrapper_exit,
        Some(0),
        "native BCC wrapper failed\nstdout:\n{}\nstderr:\n{}\nretained report:\n{}",
        String::from_utf8_lossy(&logs.stdout),
        String::from_utf8_lossy(&logs.stderr),
        retained_report.as_deref().unwrap_or("<none>"),
    );
    assert!(
        String::from_utf8_lossy(&logs.stdout).contains("Report:"),
        "public native binary did not print its report path"
    );

    let report: Value = serde_json::from_str(retained_report.as_deref().expect("retained report"))
        .expect("parse native BCC profile report");
    let evidence_path = readable_output_dir.join("evidence_bundle.json");
    let evidence: Value = serde_json::from_slice(
        &fs::read(evidence_path).expect("read retained native BCC evidence bundle"),
    )
    .expect("parse native BCC evidence bundle");
    assert_reported_artifact_paths_confined(&report, &output_dir);
    assert_reported_artifact_paths_confined(&evidence, &output_dir);

    assert_eq!(report["schema_version"], "container-command-profile.v1");
    assert_eq!(report["runtime"], "docker");
    assert_eq!(report["scope"], "disposable-image-container");
    assert_eq!(report["image_identity"]["id"], task_image_id);
    assert_eq!(report["profiling_valid"], true);
    assert_eq!(report["command_exit_code"], 0);
    assert_eq!(report["docker_returncode"], 0);
    assert_eq!(report["cpu"]["valid_for_cpu_performance"], true);
    assert!(report["cpu"]["usage_sec"].as_f64().unwrap() > 0.0);
    assert!(
        report["cpu"]["average_logical_cores_used"]
            .as_f64()
            .unwrap()
            > 0.0
    );

    let before = parse_cpu_stat(&readable_output_dir.join("cgroup_cpu.stat.observer_start.txt"));
    let after =
        parse_cpu_stat(&readable_output_dir.join("cgroup_cpu.stat.controlled_finish_gate.txt"));
    let usage_delta = after["usage_usec"] - before["usage_usec"];
    assert!(usage_delta > 0, "task cgroup reported zero CPU usage");
    assert_eq!(report["cpu"]["cpu_stat_delta"]["usage_usec"], usage_delta);
    let authoritative_usage_sec = usage_delta as f64 / 1_000_000.0;
    assert_close(
        report["cpu"]["usage_sec"].as_f64().unwrap(),
        authoritative_usage_sec,
        0.000_000_001,
        "cgroup CPU usage derived from retained cpu.stat",
    );

    let scheduler = &report["scheduler_concurrency"];
    assert_eq!(scheduler["backend"], "bcc");
    assert_eq!(
        scheduler["method"],
        "ebpf-bcc-finish-task-switch-cgroup-subtree"
    );
    assert_eq!(
        scheduler["scope"],
        "all tasks scheduled from the measured container cgroup or descendant cgroups"
    );
    assert_eq!(scheduler["trace_complete"], true);
    assert_eq!(scheduler["lost_perf_events"], 0);
    assert!(scheduler["events_observed"].as_u64().unwrap() > 0);
    let peak = scheduler["peak_simultaneous_logical_cores"]
        .as_u64()
        .unwrap();
    assert!(
        peak >= 2,
        "parallel workload did not achieve a BCC peak >= 2"
    );
    assert!(peak <= worker_cpus.len() as u64);
    let scheduled_runtime = scheduler["scheduled_runtime_sec"].as_f64().unwrap();
    assert!(scheduled_runtime > 0.0);
    let scheduler_minus_cgroup = scheduled_runtime - authoritative_usage_sec;
    let allowed_reconciliation_delta = 0.005_f64.max(0.02 * authoritative_usage_sec);
    let validation = &scheduler["cgroup_validation"];
    assert_close(
        validation["scheduler_minus_cgroup_usage_sec"]
            .as_f64()
            .expect("reported scheduler/cgroup delta"),
        scheduler_minus_cgroup,
        0.000_000_001,
        "reported scheduler/cgroup delta is independently recomputed",
    );
    assert_close(
        validation["allowed_delta_sec"]
            .as_f64()
            .expect("reported scheduler/cgroup allowance"),
        allowed_reconciliation_delta,
        0.000_000_001,
        "reported scheduler/cgroup allowance is independently recomputed",
    );
    assert!(
        scheduler_minus_cgroup.abs() <= allowed_reconciliation_delta,
        "scheduler/cgroup delta {scheduler_minus_cgroup} exceeded independent allowance {allowed_reconciliation_delta}"
    );
    assert_eq!(validation["event_scaled_tolerance_applied"], false);
    assert_close(
        weighted_width_integral(scheduler),
        scheduled_runtime,
        0.000_001,
        "time-at-width integral",
    );
    assert_close(
        sum_object_values(
            &scheduler["runtime_sec_by_logical_cpu"],
            "per-logical-CPU runtime",
        ),
        scheduled_runtime,
        0.000_001,
        "per-logical-CPU runtime integral",
    );
    assert_close(
        sum_object_values(
            &scheduler["time_sec_by_simultaneous_logical_cores"],
            "time-at-width distribution",
        ),
        scheduler["window_wall_sec"].as_f64().unwrap(),
        0.000_001,
        "time-at-width wall integral",
    );
    assert_eq!(scheduler["cgroup_validation"]["within_tolerance"], true);
    assert_eq!(
        scheduler["cgroup_validation"]["valid_for_concurrency_reporting"],
        true
    );
    assert_eq!(
        scheduler["cgroup_validation"]["per_observed_event_tolerance_sec"],
        0.000_002
    );
    assert_close(
        scheduler["cgroup_validation"]["authoritative_cgroup_usage_sec"]
            .as_f64()
            .unwrap(),
        report["cpu"]["usage_sec"].as_f64().unwrap(),
        0.000_000_001,
        "BCC reconciliation uses authoritative task-cgroup CPU",
    );

    let events = fs::read_to_string(readable_output_dir.join("scheduler_events.jsonl"))
        .expect("read retained native BCC events");
    let mut raw_event_count = 0_u64;
    for line in events.lines().filter(|line| !line.is_empty()) {
        let event: Value = serde_json::from_str(line).expect("parse retained scheduler event row");
        let cpu = u32::try_from(event["cpu"].as_u64().expect("scheduler event numeric CPU"))
            .expect("scheduler event CPU fits u32");
        assert!(
            worker_cpus.contains(&cpu),
            "scheduler event CPU {cpu} escaped enforced worker cpuset {worker_cpus:?}"
        );
        raw_event_count += 1;
    }
    assert_eq!(
        raw_event_count,
        scheduler["events_observed"].as_u64().unwrap()
            + scheduler["events_ignored_before_window"].as_u64().unwrap()
            + scheduler["events_ignored_after_window"].as_u64().unwrap(),
        "raw callback rows equal selected plus pre/post-window events"
    );
    for cpu in scheduler["runtime_sec_by_logical_cpu"]
        .as_object()
        .expect("per-logical-CPU runtime object")
        .keys()
    {
        let cpu = cpu.parse::<u32>().expect("numeric per-CPU runtime key");
        assert!(
            worker_cpus.contains(&cpu),
            "reported runtime CPU {cpu} escaped enforced worker cpuset {worker_cpus:?}"
        );
    }
    let scheduler_artifacts = scheduler["artifacts"]
        .as_array()
        .expect("scheduler artifact descriptors");
    assert_eq!(scheduler_artifacts.len(), 1);
    assert_eq!(scheduler_artifacts[0]["artifact_id"], "scheduler-events");
    for artifact in scheduler_artifacts {
        let path = artifact_readback_path(&output_dir, &readable_output_dir, artifact);
        assert_eq!(artifact["sha256"], sha256_file(&path));
        assert_eq!(artifact["size_bytes"], path.metadata().unwrap().len());
    }

    assert_eq!(
        evidence["schema_version"],
        "tasktrace-profile-evidence-bundle.v1"
    );
    assert_eq!(evidence["extensions"]["coordination_status"], "valid");
    assert_eq!(evidence["extensions"]["evidence_status"], "valid");
    assert_eq!(evidence["extensions"]["all_selected_profilers_valid"], true);
    let scheduler_profiler = evidence["profilers"]
        .as_array()
        .unwrap()
        .iter()
        .find(|profiler| profiler["profiler_id"] == "scheduler-concurrency")
        .expect("typed scheduler-concurrency profiler evidence");
    assert_eq!(scheduler_profiler["status"], "valid");
    assert_eq!(scheduler_profiler["configuration"]["backend"], "bcc");
    assert_eq!(scheduler_profiler["configuration"]["bcc_buffer_pages"], 64);
    assert!(scheduler_profiler["provenance"]["requirements"]
        .as_array()
        .unwrap()
        .iter()
        .any(|requirement| requirement == "runtime-loadable-libbcc"));
    for (stream_id, shape) in [
        ("cpu-peak-simultaneous", "scalar"),
        ("cpu-scheduled-runtime", "scalar"),
        ("scheduler-concurrency", "interval"),
        ("scheduler-per-core", "interval"),
    ] {
        assert!(evidence["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|stream| {
                stream["stream_id"] == stream_id
                    && stream["shape"] == shape
                    && stream["status"] == "valid"
            }));
    }

    let scope = &evidence["scope"];
    assert_eq!(scope["kind"], "disposable-container-cgroup-subtree");
    assert_eq!(scope["includes_descendants"], true);
    assert_eq!(scope["scope_id"], report["cpu"]["scope_id"]);
    assert_eq!(scope["scope_id"], scheduler["scope_id"]);
    assert_eq!(
        scope["identity"]["cgroup_inode"],
        scheduler["target_cgroup_id"]
    );
    assert_eq!(scope["identity"]["root_pid"], scheduler["root_tid"]);
    assert_eq!(scope["identity"]["image_identity"]["id"], task_image_id);
    let placement = &scope["identity"]["cpu_placement"];
    assert_eq!(placement["requested_cpuset_cpus"], worker_cpuset);
    assert_eq!(placement["docker_host_config_cpuset_cpus"], worker_cpuset);
    assert_eq!(placement["validation_status"], "passed");
    assert_eq!(
        parse_cpuset(
            placement["cgroup_cpuset_cpus_effective"]
                .as_str()
                .expect("effective cgroup CPU placement")
        ),
        worker_cpus
    );
    assert_eq!(
        parse_cpuset(
            placement["container_init_cpus_allowed_list"]
                .as_str()
                .expect("container init affinity")
        ),
        worker_cpus
    );

    let execution_plan = &evidence["provenance"]["execution_plan"];
    assert_eq!(execution_plan["runtime"], "docker");
    assert_eq!(execution_plan["image_identity"]["id"], task_image_id);
    assert_eq!(execution_plan["extensions"]["cpuset_cpus"], worker_cpuset);
    assert_eq!(
        execution_plan["extensions"]["cpuset_contract"],
        "docker-cpuset-enforced-and-validated"
    );
    assert_eq!(execution_plan["extensions"]["scheduler_concurrency"], true);
    assert_eq!(execution_plan["extensions"]["scheduler_backend"], "bcc");
    assert_eq!(execution_plan["network"], "none");
    assert_eq!(
        execution_plan["source_identity"]["rust_controller_binary_sha256"],
        controller_sha256
    );

    let runtime_report = &evidence["extensions"]["runtime_report"];
    let workload_container = runtime_report["container_name"]
        .as_str()
        .expect("retained workload container name");
    assert_eq!(runtime_report["runtime"], "docker");
    assert_eq!(runtime_report["returncode"], 0);
    assert_eq!(runtime_report["root_pid"], scope["identity"]["root_pid"]);
    assert_eq!(
        runtime_report["cgroup_path"],
        scope["identity"]["cgroup_path"]
    );
    match inspect_container(workload_container, DOCKER_CLEANUP_WAIT) {
        Ok(None) => {}
        Ok(Some(_)) => {
            panic!("native profiler workload container survived cleanup: {workload_container}")
        }
        Err(error) => panic!(
            "could not verify native profiler workload container cleanup for {workload_container}: {error}"
        ),
    }
    let scoped_survivors = scoped_profile_containers(&output_dir)
        .unwrap_or_else(|errors| panic!("could not verify output-scoped cleanup: {errors:?}"));
    assert!(
        scoped_survivors.is_empty(),
        "output-scoped workload containers survived cleanup: {scoped_survivors:?}"
    );
    assert!(
        !output_dir.join(".tasktrace-profile-claim").exists(),
        "profile output ownership claim survived controller cleanup"
    );
    let post_execution_controller_sha256 = sha256_file(&controller);
    assert_eq!(
        post_execution_controller_sha256, controller_sha256,
        "writable artifact mount must not alter the copied controller"
    );
    assert_eq!(
        controller.metadata().unwrap().permissions().mode() & 0o777,
        0o555,
        "copied controller mode remains read-only after execution"
    );
    write_json(
        &artifact_root.join("cleanup_proof.json"),
        &json!({
            "schema_version": "tasktrace-native-bcc-docker-cleanup.v1",
            "wrapper_container_absent": true,
            "wrapper_label_scope_absent": true,
            "workload_container": workload_container,
            "workload_container_absent": true,
            "output_scoped_workload_containers": scoped_survivors,
            "output_claim_absent": true,
            "authoritative_profile_output": output_dir,
            "unprivileged_readback_copy": readable_output_dir,
            "readback_method": "docker container cp before owned wrapper removal; source modes and ownership remain unchanged",
            "wrapper_exit_zero_proved_checked_scheduler_close": wrapper_exit == Some(0),
            "owned_scope_cleanup_verified": cleanup_report.verified(),
            "tracefs_cleanup_verified": cleanup_report.tracefs_cleanup_verified,
            "tracefs_snapshot_matches_baseline": cleanup_report.tracefs_snapshot_matches_baseline,
            "tracefs_residue_detected": cleanup_report.tracefs_residue_detected,
            "tracefs_baseline_sha256": sha256_file(&artifact_root.join("tracefs_kprobe_events.before.txt")),
            "tracefs_post_cleanup_sha256": sha256_file(&artifact_root.join("tracefs_kprobe_events.after.txt")),
            "same_path_controller": controller,
            "same_path_controller_sha256": controller_sha256,
            "post_execution_controller_sha256": post_execution_controller_sha256,
            "controller_digest_unchanged": true,
            "cleanup_diagnostic": wrapper.diagnostic_path,
        }),
    );

    eprintln!(
        "retained native BCC Docker E2E proof: {}",
        artifact_root.display()
    );
}
