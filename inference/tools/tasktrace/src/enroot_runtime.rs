// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native Enroot lifecycle support for wall-time-only container profiles.
//!
//! Enroot does not expose a task-owned cgroup to this controller.  This module
//! therefore never manufactures CPU, memory, scheduler, or process evidence.
//! It owns only static validation, immutable `.sqsh` identity, bind-mount
//! translation, the create/start/remove lifecycle, and suspend-aware host
//! `CLOCK_BOOTTIME` timing.

#[path = "enroot_network.rs"]
#[allow(dead_code)] // Hidden helper dispatch is absent in standalone runtime integration tests.
pub mod enroot_network;

use self::enroot_network::{
    controller_expectation, read_and_validate_evidence, validate_evidence_destination,
    EnrootNetworkEvidence,
};
use crate::network_policy::NetworkMode;
use serde::Serialize;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::OnceLock;
use std::thread;
use std::time::{Duration, Instant};

pub const ENROOT_RUNTIME_SCHEMA_VERSION: &str = "tasktrace-enroot-runtime.v1";
/// enroot reads this from its own environment (overriding enroot.conf) to decide whether
/// hooks.d/10-home.sh mounts the host home into the container. See [`build_start_command`].
pub const ENROOT_HOME_HOOK_VARIABLE: &str = "ENROOT_MOUNT_HOME";
const IMAGE_ENV_EXEC_PREAMBLE: &str = "if [ -r /etc/environment ]; then \
     while IFS= read -r line || [ -n \"$line\" ]; do \
     case $line in \"#\"*) ;; *=*) set -- \"$line\" \"$@\" ;; esac; \
     done < /etc/environment; fi; ";
const DOCKER_DEFAULT_PATH: &str = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin";
pub const ENROOT_WALL_TIME_METHOD: &str = "wall-clock-only-no-cgroup";
const WALL_TIME_LIMITATION: &str =
    "The current Enroot backend is wall-time-only; task-scoped resource evidence is unavailable.";
const CANCEL_POLL: Duration = Duration::from_millis(10);
const CANCEL_TERM_GRACE: Duration = Duration::from_secs(1);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EnrootError(String);

impl EnrootError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for EnrootError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for EnrootError {}

impl From<io::Error> for EnrootError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

/// Which executable provides the per-launch user/PID namespace containment.
///
/// `UtilLinuxUnshare` is the historical default. Hosts that restrict
/// unprivileged user-namespace creation to AppArmor-permitted binaries
/// (Ubuntu 23.10+ with `kernel.apparmor_restrict_unprivileged_userns=1`)
/// deny util-linux `unshare --user`; on such hosts bubblewrap carries the
/// required AppArmor grant and creates equivalent namespaces.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContainmentFlavor {
    UtilLinuxUnshare,
    Bubblewrap,
}

/// Probe which containment executable can create user namespaces on this
/// host. Returns the executable name and its flavor; the result is decided
/// once per process. Falls back to util-linux `unshare` when neither probe
/// succeeds so the true launch-time error stays visible.
pub fn detect_containment() -> (String, ContainmentFlavor) {
    static DETECTED: OnceLock<(String, ContainmentFlavor)> = OnceLock::new();
    DETECTED
        .get_or_init(|| {
            let probe = |program: &str, arguments: &[&str]| {
                Command::new(program)
                    .args(arguments)
                    .stdin(Stdio::null())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .map(|status| status.success())
                    .unwrap_or(false)
            };
            if probe("unshare", &["--user", "--map-current-user", "/bin/true"]) {
                return ("unshare".to_owned(), ContainmentFlavor::UtilLinuxUnshare);
            }
            let euid = unsafe { libc::geteuid() }.to_string();
            let egid = unsafe { libc::getegid() }.to_string();
            if probe(
                "bwrap",
                &[
                    "--dev-bind", "/", "/",
                    "--unshare-user", "--uid", &euid, "--gid", &egid,
                    "--unshare-pid", "--proc", "/proc",
                    "/bin/true",
                ],
            ) {
                return ("bwrap".to_owned(), ContainmentFlavor::Bubblewrap);
            }
            ("unshare".to_owned(), ContainmentFlavor::UtilLinuxUnshare)
        })
        .clone()
}

/// Runtime-only configuration.  The adapter that owns the public CLI maps its
/// richer profile configuration into this type after generic parsing.
#[derive(Clone, Debug)]
pub struct EnrootRuntimeConfig {
    pub enroot: String,
    pub taskset: String,
    pub unshare: String,
    /// How `unshare` (the containment executable above) is invoked.
    pub containment_flavor: ContainmentFlavor,
    pub image: String,
    /// Docker-style existing-container mode remains unsupported.  A created
    /// Enroot container is selected by putting its name in `image`.
    pub container: Option<String>,
    pub command: String,
    /// Internal run-controller handshake. When enabled, the task entrypoint
    /// publishes readiness and waits for the wave release marker before its
    /// authoritative measured-replay timestamp.
    pub run_wave_start_gate: bool,
    pub output_dir: PathBuf,
    pub entrypoint: PathBuf,
    pub prepare_command: Option<String>,
    pub cpuset_cpus: Option<String>,
    pub allocated_cpus: Option<f64>,
    pub cpu_limit: Option<f64>,
    pub workdir: Option<String>,
    pub user: Option<String>,
    pub environment: Vec<String>,
    pub mounts: Vec<String>,
    pub shell: String,
    pub string_limit: u64,
    pub strace_mode: String,
    pub network: Option<String>,
    pub network_mode: NetworkMode,
    /// Controller-private, initially absent evidence path. Required exactly
    /// when `network_mode` is `NoNetwork`.
    pub network_evidence_path: Option<PathBuf>,
    pub scheduler_concurrency: bool,
    pub proc_stat_comparison: bool,
    pub memory_profile: bool,
    pub memory_limit_bytes: Option<u64>,
    pub memory_swap_total_limit_bytes: Option<u64>,
    pub profile_sets: Vec<String>,
    pub coordinated_baseline: bool,
    pub verification_command: Option<String>,
    pub terminate_descendants_at_finish: bool,
}

impl EnrootRuntimeConfig {
    pub fn new(
        image: impl Into<String>,
        command: impl Into<String>,
        output_dir: impl Into<PathBuf>,
        entrypoint: impl Into<PathBuf>,
    ) -> Self {
        Self {
            enroot: "enroot".to_owned(),
            taskset: "taskset".to_owned(),
            unshare: "unshare".to_owned(),
            containment_flavor: ContainmentFlavor::UtilLinuxUnshare,
            image: image.into(),
            container: None,
            command: command.into(),
            run_wave_start_gate: false,
            output_dir: output_dir.into(),
            entrypoint: entrypoint.into(),
            prepare_command: None,
            cpuset_cpus: None,
            allocated_cpus: None,
            cpu_limit: None,
            workdir: None,
            user: None,
            environment: Vec::new(),
            mounts: Vec::new(),
            shell: "/bin/bash".to_owned(),
            string_limit: 4096,
            strace_mode: "none".to_owned(),
            network: None,
            network_mode: NetworkMode::Public,
            network_evidence_path: None,
            scheduler_concurrency: false,
            proc_stat_comparison: false,
            memory_profile: false,
            memory_limit_bytes: None,
            memory_swap_total_limit_bytes: None,
            profile_sets: Vec::new(),
            coordinated_baseline: false,
            verification_command: None,
            terminate_descendants_at_finish: false,
        }
    }

    pub fn validate(&self) -> Result<(), EnrootError> {
        if self.enroot.trim().is_empty() {
            return Err(EnrootError::new("--enroot must be nonempty"));
        }
        if self.unshare.trim().is_empty() {
            return Err(EnrootError::new(
                "unshare executable must be nonempty for Enroot process containment",
            ));
        }
        if self.image.trim().is_empty() {
            return Err(EnrootError::new("enroot image mode requires --image"));
        }
        if self.container.is_some() {
            return Err(EnrootError::new("enroot mode does not support --container"));
        }
        if self.command.trim().is_empty() {
            return Err(EnrootError::new("--command must be nonempty"));
        }
        if self.output_dir.as_os_str().is_empty() {
            return Err(EnrootError::new("--output-dir must be nonempty"));
        }
        if self.entrypoint.as_os_str().is_empty() {
            return Err(EnrootError::new(
                "Enroot profiling entrypoint must be nonempty",
            ));
        }
        if self.cpuset_cpus.is_none() && self.allocated_cpus.is_none() {
            return Err(EnrootError::new(
                "--image requires --cpuset-cpus or --allocated-cpus for normalized utilization",
            ));
        }
        if let Some(cpuset) = &self.cpuset_cpus {
            parse_cpuset_cpu_ids(cpuset)?;
            if self.taskset.trim().is_empty() {
                return Err(EnrootError::new(
                    "taskset executable must be nonempty when --cpuset-cpus is requested",
                ));
            }
        }
        if self
            .allocated_cpus
            .is_some_and(|value| !value.is_finite() || value <= 0.0)
        {
            return Err(EnrootError::new("--allocated-cpus must be positive"));
        }
        if self.cpu_limit.is_some() {
            return Err(EnrootError::new(
                "--cpu-limit is Docker-only; use Slurm/taskset CPU allocation for enroot",
            ));
        }
        if self.network.is_some() {
            return Err(EnrootError::new(
                "legacy --network is unsupported by native Enroot; use network policy",
            ));
        }
        match (&self.network_mode, &self.network_evidence_path) {
            (NetworkMode::Public, None) => {}
            (NetworkMode::Public, Some(_)) => {
                return Err(EnrootError::new(
                    "Enroot network evidence path is only valid for no-network mode",
                ))
            }
            (NetworkMode::NoNetwork, Some(path)) if !path.as_os_str().is_empty() => {}
            (NetworkMode::NoNetwork, _) => {
                return Err(EnrootError::new(
                    "Enroot no-network mode requires a controller-private network evidence path",
                ))
            }
            (NetworkMode::Allowlist, _) => {
                return Err(EnrootError::new(
                    "Enroot does not support allowlist network policy",
                ))
            }
        }
        if !matches!(self.user.as_deref(), None | Some("root")) {
            return Err(EnrootError::new("enroot mode only supports --user root"));
        }
        if self.strace_mode != "none" {
            return Err(EnrootError::new(
                "strace tracing requires Docker; use --strace-mode none",
            ));
        }
        if self.scheduler_concurrency {
            return Err(EnrootError::new(
                "--scheduler-concurrency is currently Docker-only",
            ));
        }
        if self.proc_stat_comparison {
            return Err(EnrootError::new(
                "--proc-stat-comparison is currently Docker-only",
            ));
        }
        if self.memory_profile {
            return Err(EnrootError::new(
                "--memory-profile is currently Docker-only",
            ));
        }
        if self.memory_limit_bytes.is_some() {
            return Err(EnrootError::new(
                "--memory-limit-bytes is currently Docker-only",
            ));
        }
        if self.memory_swap_total_limit_bytes.is_some() {
            return Err(EnrootError::new(
                "--memory-swap-total-limit-bytes is currently Docker-only",
            ));
        }
        if !self.profile_sets.is_empty() {
            return Err(EnrootError::new(
                "--profile-set currently requires the Docker runtime",
            ));
        }
        if self.coordinated_baseline {
            return Err(EnrootError::new(
                "--coordinated-baseline currently requires the Docker runtime",
            ));
        }
        if self.verification_command.is_some() {
            return Err(EnrootError::new(
                "--verification-command currently requires Docker",
            ));
        }
        if self.terminate_descendants_at_finish {
            return Err(EnrootError::new(
                "--terminate-descendants-at-finish requires a Docker host-side profiler",
            ));
        }
        if self.shell.trim().is_empty() {
            return Err(EnrootError::new("--shell must be nonempty"));
        }
        if self.string_limit == 0 {
            return Err(EnrootError::new("strace string limit must be positive"));
        }
        validate_environment(&self.environment)?;
        for mount in &self.mounts {
            let provenance = normalized_mount_provenance(mount)?;
            if matches!(
                provenance.destination.as_deref(),
                Some("/tasktrace-profile-output" | "/tasktrace-profile-entrypoint.sh")
            ) {
                return Err(EnrootError::new(format!(
                    "mount destination is reserved by the profiler: {}",
                    provenance.destination.expect("matched destination")
                )));
            }
            format_enroot_mount(mount)?;
        }
        Ok(())
    }

    pub fn requested_capacity(&self) -> Result<Option<f64>, EnrootError> {
        if let Some(value) = self.allocated_cpus {
            return Ok(Some(value));
        }
        self.cpuset_cpus
            .as_deref()
            .map(parse_cpuset_cpu_ids)
            .transpose()
            .map(|cpus| cpus.map(|values| values.len() as f64))
    }
}

fn validate_environment(assignments: &[String]) -> Result<(), EnrootError> {
    for assignment in assignments {
        let (name, _) = assignment.split_once('=').unwrap_or((assignment, ""));
        if name.is_empty() {
            return Err(EnrootError::new("--env requires a nonempty variable name"));
        }
        if name.starts_with("TASKTRACE_PROFILE_")
            || matches!(
                name,
                "TASKTRACE_STRACE_MODE"
                    | "TASKTRACE_STRACE_STRING_LIMIT"
                    | "TASKTRACE_RESOURCE_GATE_DIR"
                    | "TASKTRACE_SCHEDULER_GATE_DIR"
                    | "TASKTRACE_SCHEDULER_TRACING"
            )
        {
            return Err(EnrootError::new(format!(
                "--env cannot override profiler control variable {name:?}"
            )));
        }
    }
    Ok(())
}

pub fn parse_cpuset_cpu_ids(value: &str) -> Result<Vec<u32>, EnrootError> {
    let mut cpus = BTreeSet::new();
    for raw in value.split(',') {
        let component = raw.trim();
        if component.is_empty() {
            return Err(EnrootError::new("CPU set contains an empty component"));
        }
        if let Some((start, end)) = component.split_once('-') {
            let start = start
                .parse::<u32>()
                .map_err(|_| EnrootError::new(format!("invalid CPU range: {component:?}")))?;
            let end = end
                .parse::<u32>()
                .map_err(|_| EnrootError::new(format!("invalid CPU range: {component:?}")))?;
            if end < start {
                return Err(EnrootError::new(format!(
                    "invalid descending CPU range: {component:?}"
                )));
            }
            if u64::from(end) - u64::from(start) > 1_000_000 {
                return Err(EnrootError::new(format!(
                    "CPU range is unreasonably large: {component:?}"
                )));
            }
            cpus.extend(start..=end);
        } else {
            cpus.insert(
                component
                    .parse::<u32>()
                    .map_err(|_| EnrootError::new(format!("invalid CPU ID: {component:?}")))?,
            );
        }
    }
    if cpus.is_empty() {
        return Err(EnrootError::new("CPU set must not be empty"));
    }
    Ok(cpus.into_iter().collect())
}

fn parse_docker_bind_mount(mount: &str) -> Result<(String, String, bool), EnrootError> {
    let mut fields = BTreeMap::new();
    let mut readonly = false;
    for raw in mount.split(',') {
        let part = raw.trim();
        if part.is_empty() {
            continue;
        }
        if matches!(part, "readonly" | "ro") {
            readonly = true;
        } else if let Some((key, value)) = part.split_once('=') {
            fields.insert(key.to_owned(), value.to_owned());
        }
    }
    if fields
        .get("type")
        .is_some_and(|kind| kind.as_str() != "bind")
    {
        return Err(EnrootError::new(format!(
            "enroot mode only supports bind mounts: {mount:?}"
        )));
    }
    let source = fields
        .get("src")
        .or_else(|| fields.get("source"))
        .filter(|value| !value.is_empty())
        .cloned();
    let destination = fields
        .get("dst")
        .or_else(|| fields.get("target"))
        .or_else(|| fields.get("destination"))
        .filter(|value| !value.is_empty())
        .cloned();
    match (source, destination) {
        (Some(source), Some(destination)) => Ok((source, destination, readonly)),
        _ => Err(EnrootError::new(format!(
            "invalid bind mount specification: {mount:?}"
        ))),
    }
}

fn is_docker_bind_spec(mount: &str) -> bool {
    mount.starts_with("type=") || mount.contains("src=") || mount.contains("source=")
}

fn escape_enroot_mount_path(path: &str) -> String {
    path.replace(' ', "\\040")
}

fn enroot_bind_mount(source: &str, destination: &str, readonly: bool) -> String {
    let access = if readonly { "ro" } else { "rw" };
    format!(
        "{}:{}:none:x-create=auto,bind,{access}",
        escape_enroot_mount_path(source),
        escape_enroot_mount_path(destination),
    )
}

/// Translate Docker's bind syntax into Enroot's fstab-style mount syntax.
pub fn format_enroot_mount(mount: &str) -> Result<String, EnrootError> {
    if mount.is_empty() {
        return Err(EnrootError::new("mount specification must not be empty"));
    }
    if !is_docker_bind_spec(mount) {
        return Ok(mount.to_owned());
    }
    let (source, destination, readonly) = parse_docker_bind_mount(mount)?;
    Ok(enroot_bind_mount(&source, &destination, readonly))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnrootMountProvenance {
    #[serde(rename = "type")]
    pub mount_type: String,
    pub destination: Option<String>,
    pub readonly: bool,
    pub parsed: bool,
}

/// Describe the mount without persisting a run-specific host source path.
pub fn normalized_mount_provenance(mount: &str) -> Result<EnrootMountProvenance, EnrootError> {
    if is_docker_bind_spec(mount) {
        let (_, destination, readonly) = parse_docker_bind_mount(mount)?;
        return Ok(EnrootMountProvenance {
            mount_type: "bind".to_owned(),
            destination: Some(destination),
            readonly,
            parsed: true,
        });
    }
    if mount.is_empty() {
        return Err(EnrootError::new("mount specification must not be empty"));
    }
    let fields: Vec<&str> = mount.split(':').collect();
    let destination = fields.get(1).map(|value| (*value).to_owned());
    let options: BTreeSet<&str> = fields
        .get(if fields.len() >= 4 { 3 } else { 2 })
        .map(|value| value.split(',').collect())
        .unwrap_or_default();
    Ok(EnrootMountProvenance {
        mount_type: "bind".to_owned(),
        readonly: options.contains("ro") || options.contains("readonly"),
        parsed: destination.is_some(),
        destination,
    })
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "kebab-case")]
pub enum EnrootImageKind {
    ImageFile,
    CreatedContainer,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnrootImageIdentity {
    pub runtime: String,
    pub kind: EnrootImageKind,
    pub image: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub path: Option<PathBuf>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub container_name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub size_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mtime_ns: Option<i64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sha256: Option<String>,
    pub immutable: bool,
}

fn sha256_file(path: &Path) -> Result<String, EnrootError> {
    let mut file = File::open(path).map_err(|error| {
        EnrootError::new(format!(
            "unable to read Enroot image {}: {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn mtime_ns(metadata: &fs::Metadata) -> Result<i64, EnrootError> {
    metadata
        .mtime()
        .checked_mul(1_000_000_000)
        .and_then(|value| value.checked_add(metadata.mtime_nsec()))
        .ok_or_else(|| EnrootError::new("Enroot image mtime does not fit nanoseconds"))
}

fn image_file_identity(path: &Path, configured: &str) -> Result<EnrootImageIdentity, EnrootError> {
    let canonical = path.canonicalize().map_err(|error| {
        EnrootError::new(format!(
            "unable to resolve Enroot image {}: {error}",
            path.display()
        ))
    })?;
    let metadata = canonical.metadata()?;
    if !metadata.is_file() {
        return Err(EnrootError::new(format!(
            "Enroot image is not a regular file: {}",
            canonical.display()
        )));
    }
    Ok(EnrootImageIdentity {
        runtime: "enroot".to_owned(),
        kind: EnrootImageKind::ImageFile,
        image: configured.to_owned(),
        path: Some(canonical.clone()),
        container_name: None,
        size_bytes: Some(metadata.len()),
        mtime_ns: Some(mtime_ns(&metadata)?),
        sha256: Some(sha256_file(&canonical)?),
        immutable: true,
    })
}

pub fn parse_enroot_list(output: &str) -> BTreeSet<String> {
    output
        .lines()
        .filter_map(|line| line.split_whitespace().next())
        .map(ToOwned::to_owned)
        .collect()
}

fn executable_file(path: &Path) -> bool {
    path.metadata()
        .map(|metadata| metadata.is_file() && metadata.permissions().mode() & 0o111 != 0)
        .unwrap_or(false)
}

fn resolve_executable(program: &str) -> Result<PathBuf, EnrootError> {
    let configured = Path::new(program);
    if configured.components().count() > 1 || configured.is_absolute() {
        if !executable_file(configured) {
            return Err(EnrootError::new(format!(
                "Enroot executable is unavailable or not executable: {program}"
            )));
        }
        return configured.canonicalize().map_err(EnrootError::from);
    }
    for directory in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
        let candidate = directory.join(configured);
        if executable_file(&candidate) {
            return candidate.canonicalize().map_err(EnrootError::from);
        }
    }
    Err(EnrootError::new(format!(
        "Enroot executable is unavailable or not executable: {program}"
    )))
}

fn run_output(arguments: &[String], label: &str) -> Result<Output, EnrootError> {
    let (program, arguments) = arguments
        .split_first()
        .ok_or_else(|| EnrootError::new(format!("{label} command is empty")))?;
    Command::new(program)
        .args(arguments)
        .output()
        .map_err(|error| {
            EnrootError::new(format!(
                "unable to run {label} executable {program:?}: {error}"
            ))
        })
}

fn output_stderr(output: &Output) -> String {
    String::from_utf8_lossy(&output.stderr).trim().to_owned()
}

fn successful_output(arguments: &[String], label: &str) -> Result<Output, EnrootError> {
    let output = run_output(arguments, label)?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(EnrootError::new(format!(
            "{label} failed with status {}: {}",
            output.status.code().unwrap_or(-1),
            output_stderr(&output)
        )))
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct EnrootProcessContainmentEvidence {
    pub backend: String,
    pub namespace_init_exit_kills_remaining_processes: bool,
    pub outer_exit_kills_namespace_init: bool,
}

fn process_containment_evidence() -> EnrootProcessContainmentEvidence {
    EnrootProcessContainmentEvidence {
        backend: "linux-user-pid-namespace-v1".to_owned(),
        namespace_init_exit_kills_remaining_processes: true,
        outer_exit_kills_namespace_init: true,
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnrootPreflight {
    pub schema_version: String,
    pub ready: bool,
    pub runtime: String,
    pub executable: PathBuf,
    pub taskset_executable: Option<PathBuf>,
    pub unshare_executable: PathBuf,
    pub version: String,
    pub image_identity: EnrootImageIdentity,
    pub cpuset_cpus: Option<String>,
    pub allocated_cpus: Option<f64>,
    pub affinity_contract: String,
    pub process_containment: EnrootProcessContainmentEvidence,
    #[serde(skip_serializing)]
    pub network_evidence_path: Option<PathBuf>,
    pub mounts: Vec<EnrootMountProvenance>,
    pub limitations: Vec<String>,
}

pub fn preflight(config: &EnrootRuntimeConfig) -> Result<EnrootPreflight, EnrootError> {
    config.validate()?;
    if !config.entrypoint.is_file() {
        return Err(EnrootError::new(format!(
            "container profiling entrypoint is missing: {}",
            config.entrypoint.display()
        )));
    }
    let executable = resolve_executable(&config.enroot)?;
    let unshare_executable = resolve_executable(&config.unshare)?;
    let taskset_executable = config
        .cpuset_cpus
        .as_ref()
        .map(|_| resolve_executable(&config.taskset))
        .transpose()?;
    if let Some(path) = &config.network_evidence_path {
        let expectation =
            controller_expectation().map_err(|error| EnrootError::new(error.to_string()))?;
        validate_evidence_destination(path, expectation.controller_euid)
            .map_err(|error| EnrootError::new(error.to_string()))?;
        let helper = env::current_exe().map_err(EnrootError::from)?;
        if !executable_file(&helper) {
            return Err(EnrootError::new(format!(
                "native Enroot network helper is unavailable or not executable: {}",
                helper.display()
            )));
        }
    }
    let version_output = successful_output(
        &[config.enroot.clone(), "version".to_owned()],
        "Enroot runtime inspection",
    )?;
    let version = String::from_utf8_lossy(&version_output.stdout)
        .lines()
        .next()
        .unwrap_or_default()
        .trim()
        .to_owned();
    if version.is_empty() {
        return Err(EnrootError::new(
            "Enroot runtime inspection returned an empty version",
        ));
    }
    let image_path = Path::new(&config.image);
    let mut limitations = vec![WALL_TIME_LIMITATION.to_owned()];
    if config.containment_flavor == ContainmentFlavor::Bubblewrap {
        limitations.push(
            "User/PID namespace containment uses bubblewrap: util-linux unshare cannot \
             create user namespaces on this host (AppArmor unprivileged-userns restriction)."
                .to_owned(),
        );
    }
    if config.network_mode == NetworkMode::NoNetwork {
        limitations.push(
            "Network access is disabled by a validated fresh Linux user/network namespace."
                .to_owned(),
        );
    }
    let image_identity = if image_path.is_file() {
        image_file_identity(image_path, &config.image)?
    } else {
        if config.image.split_whitespace().count() != 1 {
            return Err(EnrootError::new(
                "Enroot created-container name must not contain whitespace",
            ));
        }
        let listed = successful_output(
            &[config.enroot.clone(), "list".to_owned()],
            "Enroot container inspection",
        )?;
        let names = parse_enroot_list(&String::from_utf8_lossy(&listed.stdout));
        if !names.contains(&config.image) {
            return Err(EnrootError::new(
                "Enroot image file or created container is unavailable",
            ));
        }
        limitations
            .push("An Enroot created-container name has no immutable image digest.".to_owned());
        EnrootImageIdentity {
            runtime: "enroot".to_owned(),
            kind: EnrootImageKind::CreatedContainer,
            image: config.image.clone(),
            path: None,
            container_name: Some(config.image.clone()),
            size_bytes: None,
            mtime_ns: None,
            sha256: None,
            immutable: false,
        }
    };
    let mounts = config
        .mounts
        .iter()
        .map(|mount| normalized_mount_provenance(mount))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(EnrootPreflight {
        schema_version: ENROOT_RUNTIME_SCHEMA_VERSION.to_owned(),
        ready: true,
        runtime: "enroot".to_owned(),
        executable,
        taskset_executable,
        unshare_executable,
        version,
        image_identity,
        cpuset_cpus: config.cpuset_cpus.clone(),
        allocated_cpus: config.allocated_cpus,
        affinity_contract: "requested-taskset-launch-affinity-not-cpu-containment".to_owned(),
        process_containment: process_containment_evidence(),
        network_evidence_path: config.network_evidence_path.clone(),
        mounts,
        limitations,
    })
}

fn lexical_absolute(path: &Path) -> Result<PathBuf, EnrootError> {
    let source = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut result = PathBuf::new();
    for component in source.components() {
        match component {
            Component::RootDir => result.push("/"),
            Component::CurDir => {}
            Component::Normal(value) => result.push(value),
            Component::ParentDir => {
                if !result.pop() {
                    return Err(EnrootError::new(format!(
                        "path escapes filesystem root: {}",
                        path.display()
                    )));
                }
            }
            Component::Prefix(_) => {
                return Err(EnrootError::new("Windows path prefixes are unsupported"))
            }
        }
    }
    Ok(result)
}

fn utf8_absolute(path: &Path, label: &str) -> Result<String, EnrootError> {
    lexical_absolute(path)?
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| EnrootError::new(format!("{label} must be valid UTF-8")))
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&byte))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\"'\"'"))
    }
}

/// Build the exact `enroot start` command.  `image_reference` is either the
/// configured created-container name or the unique name extracted from a
/// `.sqsh` file by [`run`].
pub fn build_start_command(
    config: &EnrootRuntimeConfig,
    image_reference: &str,
) -> Result<Vec<String>, EnrootError> {
    config.validate()?;
    if image_reference.is_empty() {
        return Err(EnrootError::new("Enroot image reference must be nonempty"));
    }
    // Run enroot with its home hook disabled. With `ENROOT_MOUNT_HOME=yes` in a site's
    // enroot.conf, hooks.d/10-home.sh bind-mounts the invoking user's host $HOME over the
    // container's /root (for `--root` starts), which hides the image's own /root dotfiles
    // (SWE-bench images activate their conda env from /root/.bashrc), runs the user's
    // personal ~/.bashrc inside the task container, and exposes their home to the replayed
    // commands. A replay must see the image as built, on every site, so the variable is
    // pinned on the enroot process itself; it does not affect the container's environment.
    let mut command = vec![
        "env".to_owned(),
        format!("{ENROOT_HOME_HOOK_VARIABLE}=no"),
        config.enroot.clone(),
        "start".to_owned(),
        "--rw".to_owned(),
    ];
    if config.user.as_deref() == Some("root") {
        command.push("--root".to_owned());
    }
    for mount in &config.mounts {
        command.push("--mount".to_owned());
        command.push(format_enroot_mount(mount)?);
    }
    let entrypoint = utf8_absolute(&config.entrypoint, "Enroot entrypoint path")?;
    let output_dir = utf8_absolute(&config.output_dir, "Enroot output path")?;
    command.push("--mount".to_owned());
    command.push(enroot_bind_mount(
        &entrypoint,
        "/tasktrace-profile-entrypoint.sh",
        true,
    ));
    command.push("--mount".to_owned());
    command.push(enroot_bind_mount(
        &output_dir,
        "/tasktrace-profile-output",
        false,
    ));

    let controls = [
        // Replay-environment hygiene: some sites' Enroot configurations
        // inject host Python paths into containers (environ.d PYTHONPATH,
        // mounted $HOME exposing ~/.local site-packages), which lets host
        // packages satisfy imports that failed in the recording and flips
        // recorded outcomes. The recording environments had neither set, so
        // pin them inert on every site. PYTHONPATH must NOT be empty: CPython
        // treats an empty entry as the current directory, which would alter
        // sys.path relative to the recording; /dev/null is never importable.
        ("PYTHONPATH", "/dev/null".to_owned()),
        ("PYTHONNOUSERSITE", "1".to_owned()),
        ("TASKTRACE_PROFILE_DIR", "/tasktrace-profile-output".to_owned()),
        ("TASKTRACE_PROFILE_COMMAND", config.command.clone()),
        ("TASKTRACE_STRACE_MODE", "none".to_owned()),
        (
            "TASKTRACE_STRACE_STRING_LIMIT",
            config.string_limit.to_string(),
        ),
        ("TASKTRACE_PROFILE_SHELL", config.shell.clone()),
        ("TASKTRACE_PROFILE_SKIP_CPU_STATS", "1".to_owned()),
    ];
    let mut assignments: Vec<String> = controls
        .iter()
        .map(|(name, value)| format!("{name}={value}"))
        .collect();
    if config.run_wave_start_gate {
        assignments.push("TASKTRACE_PROFILE_RUN_WAVE_START_GATE=1".to_owned());
    }
    if let Some(prepare) = &config.prepare_command {
        assignments.push(format!("TASKTRACE_PROFILE_PREPARE_COMMAND={prepare}"));
    }
    // Bare NAME entries are unsets; enroot's `--env NAME` would instead
    // import the variable from the host, so unsets ride only in the
    // env(1) exec prefix below.
    assignments.extend(
        config
            .environment
            .iter()
            .filter(|entry| entry.contains('='))
            .cloned(),
    );
    for assignment in &assignments {
        command.push("--env".to_owned());
        command.push(assignment.clone());
    }

    let mut inner = format!(
        "set -- {}; {IMAGE_ENV_EXEC_PREAMBLE}exec /usr/bin/env -i HOME=\"$HOME\" PATH={DOCKER_DEFAULT_PATH} \"$@\"",
        assignments
            .iter()
            .map(|assignment| shell_quote(assignment))
            .collect::<Vec<_>>()
            .join(" ")
    );
    let unsets: Vec<String> = config
        .environment
        .iter()
        .filter(|entry| !entry.contains('='))
        .flat_map(|name| ["-u".to_owned(), shell_quote(name)])
        .collect();
    if !unsets.is_empty() {
        inner = format!("{inner} /usr/bin/env {}", unsets.join(" "));
    }
    inner = format!("{inner} /bin/sh /tasktrace-profile-entrypoint.sh");
    if let Some(workdir) = &config.workdir {
        inner = format!("cd {} && {{ {inner}; }}", shell_quote(workdir));
    }
    command.extend([
        "--".to_owned(),
        image_reference.to_owned(),
        "/bin/sh".to_owned(),
        "-c".to_owned(),
        inner,
    ]);
    if let Some(cpuset) = &config.cpuset_cpus {
        let mut affinity = vec![config.taskset.clone(), "-c".to_owned(), cpuset.clone()];
        affinity.extend(command);
        command = affinity;
    }

    let mut isolated = match config.containment_flavor {
        ContainmentFlavor::UtilLinuxUnshare => vec![
            config.unshare.clone(),
            "--user".to_owned(),
            "--map-current-user".to_owned(),
            "--pid".to_owned(),
            "--fork".to_owned(),
            "--kill-child=SIGKILL".to_owned(),
            "--mount-proc".to_owned(),
        ],
        // Failure-evidence caveat: for signal-killed sandboxes the two
        // flavors report differently. util-linux unshare re-raises the
        // child's signal (recorded as exit_code: null, signal: N; util-linux
        // >= 2.39 may surface exit 1 when the SIGKILL re-raise fails), while
        // bubblewrap propagates 128+N (exit_code: 137, signal: null).
        // Validity handling collapses both; cross-host comparison of raw
        // failure records must not assume one shape.
        ContainmentFlavor::Bubblewrap => vec![
            config.unshare.clone(),
            "--dev-bind".to_owned(),
            "/".to_owned(),
            "/".to_owned(),
            "--unshare-user".to_owned(),
            "--uid".to_owned(),
            unsafe { libc::geteuid() }.to_string(),
            "--gid".to_owned(),
            unsafe { libc::getegid() }.to_string(),
            "--unshare-pid".to_owned(),
            "--proc".to_owned(),
            "/proc".to_owned(),
            "--die-with-parent".to_owned(),
        ],
    };
    if config.network_mode == NetworkMode::NoNetwork {
        let expectation =
            controller_expectation().map_err(|error| EnrootError::new(error.to_string()))?;
        let evidence_path = config
            .network_evidence_path
            .as_ref()
            .expect("no-network evidence path was validated");
        validate_evidence_destination(evidence_path, expectation.controller_euid)
            .map_err(|error| EnrootError::new(error.to_string()))?;
        let helper = env::current_exe().map_err(EnrootError::from)?;
        let helper = utf8_absolute(&helper, "native Enroot network helper path")?;
        let evidence_path = utf8_absolute(evidence_path, "Enroot network evidence path")?;
        isolated.push(match config.containment_flavor {
            ContainmentFlavor::UtilLinuxUnshare => "--net".to_owned(),
            ContainmentFlavor::Bubblewrap => "--unshare-net".to_owned(),
        });
        isolated.extend([
            "--".to_owned(),
            helper,
            enroot_network::INTERNAL_COMMAND.to_owned(),
            "--parent-netns-inode".to_owned(),
            expectation.parent_netns_inode.to_string(),
            "--controller-euid".to_owned(),
            expectation.controller_euid.to_string(),
            "--controller-egid".to_owned(),
            expectation.controller_egid.to_string(),
            "--evidence-path".to_owned(),
            evidence_path,
            "--".to_owned(),
        ]);
    } else {
        isolated.push("--".to_owned());
    }
    isolated.extend(command);
    Ok(isolated)
}

fn boottime_ns() -> u64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_gettime(libc::CLOCK_BOOTTIME, &mut value) };
    assert_eq!(result, 0, "clock_gettime(CLOCK_BOOTTIME) failed");
    assert!(
        value.tv_sec >= 0 && value.tv_nsec >= 0,
        "CLOCK_BOOTTIME reported a negative timestamp"
    );
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
}

fn boottime_resolution_ns() -> u64 {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    let result = unsafe { libc::clock_getres(libc::CLOCK_BOOTTIME, &mut value) };
    assert_eq!(result, 0, "clock_getres(CLOCK_BOOTTIME) failed");
    assert!(
        value.tv_sec >= 0 && value.tv_nsec >= 0,
        "CLOCK_BOOTTIME reported a negative resolution"
    );
    (value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
        .max(1)
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnrootPhaseTiming {
    pub clock_id: String,
    pub clock_source: String,
    pub clock_resolution_ns: u64,
    pub start_monotonic_ns: u64,
    pub end_monotonic_ns: u64,
    pub duration_sec: f64,
    pub boundary: String,
}

impl EnrootPhaseTiming {
    fn new(start: u64, end: u64, boundary: &str) -> Result<Self, EnrootError> {
        if end <= start {
            return Err(EnrootError::new(format!(
                "monotonic clock did not advance during {boundary}"
            )));
        }
        Ok(Self {
            clock_id: "host-boottime".to_owned(),
            clock_source: "CLOCK_BOOTTIME".to_owned(),
            clock_resolution_ns: boottime_resolution_ns(),
            start_monotonic_ns: start,
            end_monotonic_ns: end,
            duration_sec: (end - start) as f64 / 1_000_000_000.0,
            boundary: boundary.to_owned(),
        })
    }
}

fn timed_output(
    arguments: &[String],
    label: &str,
    boundary: &str,
) -> Result<(Output, EnrootPhaseTiming), EnrootError> {
    let start = boottime_ns();
    let output = run_output(arguments, label)?;
    let end = boottime_ns();
    Ok((output, EnrootPhaseTiming::new(start, end, boundary)?))
}

#[derive(Debug)]
struct CancelledCommand {
    output: Output,
    timing: EnrootPhaseTiming,
    process_group_id: u32,
    process_group_verified_gone: bool,
    sigterm_sent: bool,
    sigkill_sent: bool,
    signal_error: Option<String>,
}

enum InterruptibleCommandError {
    Failed(EnrootError),
    Cancelled(CancelledCommand),
}

impl From<EnrootError> for InterruptibleCommandError {
    fn from(error: EnrootError) -> Self {
        Self::Failed(error)
    }
}

fn read_captured_output(
    status: std::process::ExitStatus,
    stdout_path: &Path,
    stderr_path: &Path,
) -> Result<Output, EnrootError> {
    Ok(Output {
        status,
        stdout: fs::read(stdout_path)?,
        stderr: fs::read(stderr_path)?,
    })
}

fn signal_process_group(process_group_id: u32, signal: libc::c_int) -> Result<bool, EnrootError> {
    let process_group_id = i32::try_from(process_group_id)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| EnrootError::new("Enroot process-group ID is unrepresentable"))?;
    let result = unsafe { libc::kill(-process_group_id, signal) };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(false)
    } else {
        Err(EnrootError::new(format!(
            "unable to signal Enroot process group {process_group_id}: {error}"
        )))
    }
}

fn process_group_exists(process_group_id: u32) -> Result<bool, EnrootError> {
    let process_group_id = i32::try_from(process_group_id)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| EnrootError::new("Enroot process-group ID is unrepresentable"))?;
    let result = unsafe { libc::kill(-process_group_id, 0) };
    if result == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(EnrootError::new(format!(
            "unable to inspect Enroot process group {process_group_id}: {error}"
        ))),
    }
}

fn append_signal_error(target: &mut Option<String>, error: impl fmt::Display) {
    *target = Some(match target.take() {
        Some(previous) => format!("{previous}; {error}"),
        None => error.to_string(),
    });
}

fn wait_for_process_group_exit_with<E>(
    process_group_id: u32,
    deadline: Instant,
    process_group_exists: &mut E,
) -> Result<bool, EnrootError>
where
    E: FnMut(u32) -> Result<bool, EnrootError>,
{
    loop {
        if !process_group_exists(process_group_id)? {
            return Ok(true);
        }
        if Instant::now() >= deadline {
            return Ok(false);
        }
        thread::sleep(CANCEL_POLL);
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnrootProcessGroupCleanupEvidence {
    pub process_group_id: u32,
    pub leader_reaped: bool,
    pub sigterm_sent: bool,
    pub sigkill_sent: bool,
    pub verified_gone: bool,
}

fn process_group_cleanup_evidence(
    process_group_id: u32,
    sigterm_sent: bool,
    sigkill_sent: bool,
) -> EnrootProcessGroupCleanupEvidence {
    EnrootProcessGroupCleanupEvidence {
        process_group_id,
        leader_reaped: true,
        sigterm_sent,
        sigkill_sent,
        verified_gone: true,
    }
}

/// A successfully reaped command leader is not proof that the command scope
/// is gone: shells and Enroot hooks can leave background members in the fresh
/// process group created below.  Always close that owned scope before treating
/// the leader's status as a completed command.
pub(crate) fn cleanup_process_group_after_leader_exit_with<E, S>(
    process_group_id: u32,
    label: &str,
    term_grace: Duration,
    mut process_group_exists: E,
    mut signal_process_group: S,
) -> Result<EnrootProcessGroupCleanupEvidence, EnrootError>
where
    E: FnMut(u32) -> Result<bool, EnrootError>,
    S: FnMut(u32, libc::c_int) -> Result<bool, EnrootError>,
{
    let mut cleanup_errors = Vec::new();
    match process_group_exists(process_group_id) {
        Ok(false) => {
            return Ok(process_group_cleanup_evidence(
                process_group_id,
                false,
                false,
            ))
        }
        Ok(true) => {}
        Err(error) => cleanup_errors.push(format!(
            "initial process-group inspection after leader exit failed: {error}"
        )),
    }

    let term_signalled = match signal_process_group(process_group_id, libc::SIGTERM) {
        Ok(signalled) => signalled,
        Err(error) => {
            cleanup_errors.push(format!("SIGTERM failed: {error}"));
            false
        }
    };
    if term_signalled {
        match wait_for_process_group_exit_with(
            process_group_id,
            Instant::now() + term_grace,
            &mut process_group_exists,
        ) {
            Ok(true) => {
                return Ok(process_group_cleanup_evidence(
                    process_group_id,
                    true,
                    false,
                ))
            }
            Ok(false) => {}
            Err(error) => cleanup_errors.push(format!("post-SIGTERM inspection failed: {error}")),
        }
    }

    let sigkill_sent = match signal_process_group(process_group_id, libc::SIGKILL) {
        Ok(signalled) => signalled,
        Err(error) => {
            cleanup_errors.push(format!("SIGKILL failed: {error}"));
            false
        }
    };
    match wait_for_process_group_exit_with(
        process_group_id,
        Instant::now() + term_grace,
        &mut process_group_exists,
    ) {
        Ok(true) => Ok(process_group_cleanup_evidence(
            process_group_id,
            term_signalled,
            sigkill_sent,
        )),
        Ok(false) => {
            cleanup_errors.push(
                "process group remained present after bounded SIGTERM/SIGKILL cleanup".to_owned(),
            );
            Err(EnrootError::new(format!(
                "{label} process group {process_group_id} cleanup failed; process scope is uncertain: {}",
                cleanup_errors.join("; ")
            )))
        }
        Err(error) => {
            cleanup_errors.push(format!("post-SIGKILL inspection failed: {error}"));
            Err(EnrootError::new(format!(
                "{label} process group {process_group_id} cleanup failed; process scope is uncertain: {}",
                cleanup_errors.join("; ")
            )))
        }
    }
}

fn cleanup_process_group_after_leader_exit(
    process_group_id: u32,
    label: &str,
) -> Result<EnrootProcessGroupCleanupEvidence, EnrootError> {
    cleanup_process_group_after_leader_exit_with(
        process_group_id,
        label,
        CANCEL_TERM_GRACE,
        process_group_exists,
        signal_process_group,
    )
}

// Cancellation deliberately carries the complete captured output and timing
// evidence so callers can retain a fail-closed diagnostic report.
#[allow(clippy::result_large_err)]
fn timed_output_interruptible(
    arguments: &[String],
    label: &str,
    boundary: &str,
    stdout_path: &Path,
    stderr_path: &Path,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<(Output, EnrootPhaseTiming, EnrootProcessGroupCleanupEvidence), InterruptibleCommandError>
{
    let (program, arguments) = arguments
        .split_first()
        .ok_or_else(|| EnrootError::new(format!("{label} command is empty")))?;
    let stdout = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(stdout_path)
        .map_err(EnrootError::from)?;
    let stderr = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(stderr_path)
        .map_err(EnrootError::from)?;
    let mut command = Command::new(program);
    command
        .args(arguments)
        .stdin(Stdio::null())
        .stdout(Stdio::from(stdout))
        .stderr(Stdio::from(stderr));
    unsafe {
        command.pre_exec(|| {
            if libc::setpgid(0, 0) == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        });
    }
    let start = boottime_ns();
    let mut child = command.spawn().map_err(|error| {
        EnrootError::new(format!(
            "unable to run {label} executable {program:?}: {error}"
        ))
    })?;
    let process_group_id = child.id();
    loop {
        if let Some(status) = child
            .try_wait()
            .map_err(|error| EnrootError::new(format!("unable to poll {label}: {error}")))?
        {
            let process_group_cleanup =
                cleanup_process_group_after_leader_exit(process_group_id, label)?;
            let end = boottime_ns();
            let output = read_captured_output(status, stdout_path, stderr_path)?;
            return Ok((
                output,
                EnrootPhaseTiming::new(start, end, boundary)?,
                process_group_cleanup,
            ));
        }
        if cancel_requested.is_some_and(|callback| callback()) {
            let mut signal_error = None;
            let sigterm_sent = match signal_process_group(process_group_id, libc::SIGTERM) {
                Ok(sent) => sent,
                Err(error) => {
                    signal_error = Some(error.to_string());
                    let _ = child.kill();
                    false
                }
            };
            let deadline = Instant::now() + CANCEL_TERM_GRACE;
            let mut status = None;
            let mut group_alive = true;
            while Instant::now() < deadline {
                if status.is_none() {
                    status = child.try_wait().map_err(|error| {
                        EnrootError::new(format!("unable to poll cancelled {label}: {error}"))
                    })?;
                }
                match process_group_exists(process_group_id) {
                    Ok(false) => {
                        group_alive = false;
                        break;
                    }
                    Ok(true) => {}
                    Err(error) => {
                        append_signal_error(&mut signal_error, error);
                        break;
                    }
                }
                thread::sleep(CANCEL_POLL);
            }
            let mut sigkill_sent = false;
            if group_alive {
                match signal_process_group(process_group_id, libc::SIGKILL) {
                    Ok(sent) => sigkill_sent = sent,
                    Err(error) => {
                        append_signal_error(&mut signal_error, error);
                        let _ = child.kill();
                    }
                }
            }
            if status.is_none() {
                status = Some(child.wait().map_err(|error| {
                    EnrootError::new(format!("unable to reap cancelled {label}: {error}"))
                })?);
            }
            let verification_deadline = Instant::now() + CANCEL_TERM_GRACE;
            let process_group_verified_gone = loop {
                match process_group_exists(process_group_id) {
                    Ok(false) => break true,
                    Ok(true) if Instant::now() < verification_deadline => {
                        thread::sleep(CANCEL_POLL);
                    }
                    Ok(true) => {
                        append_signal_error(
                            &mut signal_error,
                            "Enroot process group remained present after cancellation",
                        );
                        break false;
                    }
                    Err(error) => {
                        append_signal_error(&mut signal_error, error);
                        break false;
                    }
                }
            };
            let status = status.expect("cancelled Enroot command was reaped");
            let end = boottime_ns();
            let output = read_captured_output(status, stdout_path, stderr_path)?;
            return Err(InterruptibleCommandError::Cancelled(CancelledCommand {
                output,
                timing: EnrootPhaseTiming::new(start, end, boundary)?,
                process_group_id,
                process_group_verified_gone,
                sigterm_sent,
                sigkill_sent,
                signal_error,
            }));
        }
        thread::sleep(CANCEL_POLL);
    }
}

static CONTAINER_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn unique_container_name() -> String {
    let sequence = CONTAINER_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    format!(
        "tasktrace-profile-{:x}-{:x}-{sequence:x}",
        std::process::id(),
        boottime_ns()
    )
}

struct OwnedContainer {
    enroot: String,
    name: String,
    active: bool,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnrootCleanupEvidence {
    pub remove_timing: Option<EnrootPhaseTiming>,
    pub verification_timing: Option<EnrootPhaseTiming>,
    pub remove_status: Option<i32>,
    pub verification_status: Option<i32>,
    pub verified_absent: bool,
    pub succeeded: bool,
    pub reasons: Vec<String>,
}

impl EnrootCleanupEvidence {
    fn error(&self, name: &str) -> EnrootError {
        EnrootError::new(format!(
            "Enroot container cleanup failed for {name:?}: {}",
            self.reasons.join("; ")
        ))
    }
}

impl OwnedContainer {
    fn new(enroot: &str, name: String) -> Self {
        Self {
            enroot: enroot.to_owned(),
            name,
            active: true,
        }
    }

    fn close(&mut self) -> EnrootCleanupEvidence {
        if !self.active {
            return EnrootCleanupEvidence {
                remove_timing: None,
                verification_timing: None,
                remove_status: None,
                verification_status: None,
                verified_absent: false,
                succeeded: false,
                reasons: vec!["Enroot container cleanup was already attempted".to_owned()],
            };
        }
        self.active = false;
        let remove_arguments = vec![
            self.enroot.clone(),
            "remove".to_owned(),
            "-f".to_owned(),
            self.name.clone(),
        ];
        let remove = timed_output(
            &remove_arguments,
            "Enroot container cleanup",
            "enroot-remove-owned-container",
        );
        let verify_arguments = vec![self.enroot.clone(), "list".to_owned()];
        let verification = timed_output(
            &verify_arguments,
            "Enroot cleanup verification",
            "enroot-list-after-remove",
        );
        let mut reasons = Vec::new();
        let (remove_timing, remove_status) = match remove {
            Ok((output, timing)) => {
                if !output.status.success() {
                    reasons.push(format!(
                        "remove returned status {}: {}",
                        output.status.code().unwrap_or(-1),
                        output_stderr(&output)
                    ));
                }
                (Some(timing), output.status.code())
            }
            Err(error) => {
                reasons.push(format!("remove command failed: {error}"));
                (None, None)
            }
        };
        let (verification_timing, verification_status, verified_absent) = match verification {
            Ok((output, timing)) => {
                let listed = parse_enroot_list(&String::from_utf8_lossy(&output.stdout));
                let absent = output.status.success() && !listed.contains(&self.name);
                if !output.status.success() {
                    reasons.push(format!(
                        "post-remove list returned status {}: {}",
                        output.status.code().unwrap_or(-1),
                        output_stderr(&output)
                    ));
                } else if !absent {
                    reasons.push(format!(
                        "owned container {:?} remains present after remove",
                        self.name
                    ));
                }
                (Some(timing), output.status.code(), absent)
            }
            Err(error) => {
                reasons.push(format!("post-remove list verification failed: {error}"));
                (None, None, false)
            }
        };
        let succeeded = reasons.is_empty() && verified_absent;
        EnrootCleanupEvidence {
            remove_timing,
            verification_timing,
            remove_status,
            verification_status,
            verified_absent,
            succeeded,
            reasons,
        }
    }
}

impl Drop for OwnedContainer {
    fn drop(&mut self) {
        if self.active {
            self.active = false;
            let _ = Command::new(&self.enroot)
                .args(["remove", "-f", &self.name])
                .output();
        }
    }
}

fn combine_cleanup(
    primary: EnrootError,
    cleanup: &EnrootCleanupEvidence,
    name: &str,
) -> EnrootError {
    if cleanup.succeeded {
        primary
    } else {
        EnrootError::new(format!("{primary}; {}", cleanup.error(name)))
    }
}

fn cancellation_error(
    config: &EnrootRuntimeConfig,
    phase: &str,
    cancelled: &CancelledCommand,
    owned_container_name: Option<&str>,
    cleanup: Option<&EnrootCleanupEvidence>,
) -> EnrootError {
    let mut reasons = vec![format!("Enroot {phase} was cancelled")];
    if let Some(error) = &cancelled.signal_error {
        reasons.push(error.clone());
    }
    if let Some(cleanup) = cleanup.filter(|cleanup| !cleanup.succeeded) {
        reasons.extend(cleanup.reasons.clone());
    }
    let payload = json!({
        "schema_version": "tasktrace-enroot-runtime-failure.v1",
        "runtime": "enroot",
        "status": "cancelled",
        "profiling_valid": false,
        "phase": phase,
        "reasons": reasons,
        "command_timing": cancelled.timing,
        "host_process_group": {
            "id": cancelled.process_group_id,
            "owned_by_controller": true,
            "verified_gone": cancelled.process_group_verified_gone,
            "sigterm_sent": cancelled.sigterm_sent,
            "sigkill_sent": cancelled.sigkill_sent,
            "signal_error": cancelled.signal_error,
            "returncode": cancelled.output.status.code(),
            "signal": cancelled.output.status.signal(),
        },
        "owned_container_name": owned_container_name,
        "process_containment": process_containment_evidence(),
        "cleanup_required": owned_container_name.is_some(),
        "cleanup": cleanup,
        "cleanup_succeeded": cleanup.map_or(true, |evidence| evidence.succeeded),
        "caller_owned_named_container_removed": false,
        "timing_valid_for_performance": false,
    });
    let path = config.output_dir.join("enroot_runtime_failure.json");
    let retained = serde_json::to_vec_pretty(&payload)
        .map_err(|error| {
            EnrootError::new(format!("unable to encode cancellation evidence: {error}"))
        })
        .and_then(|mut bytes| {
            bytes.push(b'\n');
            fs::write(&path, bytes).map_err(|error| {
                EnrootError::new(format!(
                    "unable to retain Enroot cancellation evidence {}: {error}",
                    path.display()
                ))
            })
        });
    let primary = EnrootError::new(format!("Enroot {phase} cancelled"));
    match retained {
        Ok(()) => primary,
        Err(error) => EnrootError::new(format!("{primary}; {error}")),
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct EnrootRunResult {
    pub schema_version: String,
    pub runtime: String,
    pub runtime_version: String,
    pub image_identity: EnrootImageIdentity,
    pub owned_container_name: Option<String>,
    pub create_timing: Option<EnrootPhaseTiming>,
    pub create_process_group_cleanup: Option<EnrootProcessGroupCleanupEvidence>,
    pub start_timing: EnrootPhaseTiming,
    pub start_process_group_cleanup: EnrootProcessGroupCleanupEvidence,
    pub process_containment: EnrootProcessContainmentEvidence,
    #[serde(skip_serializing)]
    pub network_evidence_path: Option<PathBuf>,
    pub network_evidence: Option<EnrootNetworkEvidence>,
    pub cleanup_timing: Option<EnrootPhaseTiming>,
    pub cleanup_verification_timing: Option<EnrootPhaseTiming>,
    pub cleanup_verified_absent: Option<bool>,
    pub cleanup: Option<EnrootCleanupEvidence>,
    pub cleanup_required: bool,
    pub cleanup_succeeded: bool,
    pub exit_code: Option<i32>,
    pub signal: Option<i32>,
    pub stdout: String,
    pub stderr: String,
    pub requested_cpuset_cpus: Option<String>,
    pub requested_allocated_cpus: Option<f64>,
    pub cpu_containment_enforced: bool,
}

impl EnrootRunResult {
    /// A report fragment that cannot be mistaken for cgroup CPU evidence.
    pub fn wall_time_cpu_report(&self, config: &EnrootRuntimeConfig) -> Result<Value, EnrootError> {
        Ok(json!({
            "method": ENROOT_WALL_TIME_METHOD,
            "wall_sec": self.start_timing.duration_sec,
            "usage_sec": Value::Null,
            "average_logical_cores_used": Value::Null,
            "utilization_percent_of_allocation": Value::Null,
            "allocated_logical_cpus": config.requested_capacity()?,
            "requested_cpuset_cpus": config.cpuset_cpus,
            "cpu_containment_enforced": false,
            "affinity_contract": "requested-taskset-launch-affinity-not-cpu-containment",
            "clock_id": self.start_timing.clock_id,
            "clock_source": self.start_timing.clock_source,
            "clock_resolution_ns": self.start_timing.clock_resolution_ns,
            "start_monotonic_ns": self.start_timing.start_monotonic_ns,
            "end_monotonic_ns": self.start_timing.end_monotonic_ns,
            "timing_boundary": self.start_timing.boundary,
        }))
    }
}

/// Execute one Enroot image without an external cancellation callback.
pub fn run(config: &EnrootRuntimeConfig) -> Result<EnrootRunResult, EnrootError> {
    run_with_cancel(config, None)
}

/// Execute one Enroot image with cooperative cancellation.
///
/// Create and start each run in a fresh host process group.  Normal completion
/// verifies that the group is empty and terminates any surviving members;
/// cancellation signals only that owned group before following the verified
/// container-cleanup path.  A caller-owned named Enroot container is never
/// removed.
pub fn run_with_cancel(
    config: &EnrootRuntimeConfig,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<EnrootRunResult, EnrootError> {
    config.validate()?;
    if !config.output_dir.is_dir() {
        return Err(EnrootError::new(format!(
            "Enroot profile output directory does not exist: {}",
            config.output_dir.display()
        )));
    }
    let preflight = preflight(config)?;
    let network_expectation = if config.network_mode == NetworkMode::NoNetwork {
        Some(controller_expectation().map_err(|error| EnrootError::new(error.to_string()))?)
    } else {
        None
    };
    let mut owned_name = None;
    let mut create_timing = None;
    let mut create_process_group_cleanup = None;
    let mut owner = None;
    let image_reference = match preflight.image_identity.kind {
        EnrootImageKind::CreatedContainer => config.image.clone(),
        EnrootImageKind::ImageFile => {
            let name = unique_container_name();
            let start_command = build_start_command(config, &name)?;
            drop(start_command);
            let image_path = preflight
                .image_identity
                .path
                .as_ref()
                .and_then(|path| path.to_str())
                .ok_or_else(|| EnrootError::new("Enroot image path must be valid UTF-8"))?
                .to_owned();
            let mut created = OwnedContainer::new(&config.enroot, name.clone());
            let arguments = vec![
                config.enroot.clone(),
                "create".to_owned(),
                "--name".to_owned(),
                name.clone(),
                image_path,
            ];
            let create = timed_output_interruptible(
                &arguments,
                "Enroot image extraction",
                "enroot-create-image-file",
                &config.output_dir.join("enroot_create.stdout.txt"),
                &config.output_dir.join("enroot_create.stderr.txt"),
                cancel_requested,
            );
            let (output, timing, process_group_cleanup) = match create {
                Ok(result) => result,
                Err(InterruptibleCommandError::Failed(error)) => {
                    let cleanup = created.close();
                    return Err(combine_cleanup(error, &cleanup, &name));
                }
                Err(InterruptibleCommandError::Cancelled(cancelled)) => {
                    let cleanup = created.close();
                    return Err(cancellation_error(
                        config,
                        "create",
                        &cancelled,
                        Some(&name),
                        Some(&cleanup),
                    ));
                }
            };
            if !output.status.success() {
                let error = EnrootError::new(format!(
                    "enroot create failed for {:?} with status {}: {}",
                    config.image,
                    output.status.code().unwrap_or(-1),
                    output_stderr(&output)
                ));
                let cleanup = created.close();
                return Err(combine_cleanup(error, &cleanup, &name));
            }
            let current_identity =
                match image_file_identity(Path::new(&config.image), &config.image) {
                    Ok(identity) => identity,
                    Err(error) => {
                        let cleanup = created.close();
                        return Err(combine_cleanup(error, &cleanup, &name));
                    }
                };
            if current_identity != preflight.image_identity {
                let cleanup = created.close();
                return Err(combine_cleanup(
                    EnrootError::new(
                        "Enroot image file changed while its container was being created",
                    ),
                    &cleanup,
                    &name,
                ));
            }
            create_timing = Some(timing);
            create_process_group_cleanup = Some(process_group_cleanup);
            owned_name = Some(name.clone());
            owner = Some(created);
            name
        }
    };

    let start_command = match build_start_command(config, &image_reference) {
        Ok(command) => command,
        Err(error) => {
            if let Some(owner) = owner.as_mut() {
                let cleanup = owner.close();
                return Err(combine_cleanup(error, &cleanup, &owner.name));
            }
            return Err(error);
        }
    };
    let started = timed_output_interruptible(
        &start_command,
        "Enroot container start",
        "host-enroot-start-lifecycle-inclusive",
        &config.output_dir.join("enroot_start.stdout.txt"),
        &config.output_dir.join("enroot_start.stderr.txt"),
        cancel_requested,
    );
    let cleanup = owner.as_mut().map(OwnedContainer::close);
    let cleanup_required = owned_name.is_some();
    let cleanup_succeeded = cleanup.as_ref().map_or(true, |value| value.succeeded);
    let (output, start_timing, start_process_group_cleanup) = match started {
        Ok(result) => {
            if let (Some(name), Some(cleanup)) = (owned_name.as_deref(), cleanup.as_ref()) {
                if !cleanup.succeeded {
                    return Err(cleanup.error(name));
                }
            }
            result
        }
        Err(InterruptibleCommandError::Failed(error)) => {
            return Err(match (owned_name.as_deref(), cleanup.as_ref()) {
                (Some(name), Some(cleanup)) => combine_cleanup(error, cleanup, name),
                _ => error,
            });
        }
        Err(InterruptibleCommandError::Cancelled(cancelled)) => {
            return Err(cancellation_error(
                config,
                "start",
                &cancelled,
                owned_name.as_deref(),
                cleanup.as_ref(),
            ));
        }
    };
    let cleanup_timing = cleanup
        .as_ref()
        .and_then(|value| value.remove_timing.clone());
    let cleanup_verification_timing = cleanup
        .as_ref()
        .and_then(|value| value.verification_timing.clone());
    let cleanup_verified_absent = cleanup.as_ref().map(|value| value.verified_absent);
    let network_evidence = match (config.network_evidence_path.as_deref(), network_expectation) {
        (Some(path), Some(expectation)) => Some(
            read_and_validate_evidence(path, expectation)
                .map_err(|error| EnrootError::new(error.to_string()))?,
        ),
        (None, None) => None,
        _ => {
            return Err(EnrootError::new(
                "Enroot network evidence configuration changed during execution",
            ))
        }
    };
    Ok(EnrootRunResult {
        schema_version: ENROOT_RUNTIME_SCHEMA_VERSION.to_owned(),
        runtime: "enroot".to_owned(),
        runtime_version: preflight.version,
        image_identity: preflight.image_identity,
        owned_container_name: owned_name,
        create_timing,
        create_process_group_cleanup,
        start_timing,
        start_process_group_cleanup,
        process_containment: process_containment_evidence(),
        network_evidence_path: config.network_evidence_path.clone(),
        network_evidence,
        cleanup_timing,
        cleanup_verification_timing,
        cleanup_verified_absent,
        cleanup,
        cleanup_required,
        cleanup_succeeded,
        exit_code: output.status.code(),
        signal: output.status.signal(),
        stdout: String::from_utf8_lossy(&output.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&output.stderr).into_owned(),
        requested_cpuset_cpus: config.cpuset_cpus.clone(),
        requested_allocated_cpus: config.allocated_cpus,
        cpu_containment_enforced: false,
    })
}

#[cfg(test)]
mod namespace_tests {
    use super::*;
    use std::os::unix::fs::PermissionsExt;

    fn config(root: &Path) -> EnrootRuntimeConfig {
        let mut config = EnrootRuntimeConfig::new(
            root.join("task.sqsh").to_string_lossy(),
            "true",
            root.join("output"),
            root.join("entrypoint.sh"),
        );
        config.cpuset_cpus = Some("4".to_owned());
        config
    }

    #[test]
    fn every_start_uses_fresh_user_and_pid_namespaces() {
        let temporary = tempfile::tempdir().unwrap();
        let command = build_start_command(&config(temporary.path()), "task").unwrap();
        assert_eq!(
            &command[..8],
            [
                "unshare",
                "--user",
                "--map-current-user",
                "--pid",
                "--fork",
                "--kill-child=SIGKILL",
                "--mount-proc",
                "--",
            ]
        );
        let taskset = command.iter().position(|value| value == "taskset").unwrap();
        let enroot = command.iter().position(|value| value == "enroot").unwrap();
        assert!(taskset < enroot);
        assert_eq!(&command[taskset..taskset + 3], ["taskset", "-c", "4"]);
    }

    #[test]
    fn every_start_disables_the_enroot_home_hook() {
        let temporary = tempfile::tempdir().unwrap();
        for flavor in [
            ContainmentFlavor::UtilLinuxUnshare,
            ContainmentFlavor::Bubblewrap,
        ] {
            let mut config = config(temporary.path());
            if flavor == ContainmentFlavor::Bubblewrap {
                config.unshare = "bwrap".to_owned();
            }
            config.containment_flavor = flavor;
            let command = build_start_command(&config, "task").unwrap();
            let enroot = command.iter().position(|value| value == "enroot").unwrap();
            assert_eq!(
                &command[enroot - 2..enroot + 2],
                ["env", "ENROOT_MOUNT_HOME=no", "enroot", "start"],
                "the host home hook must be disabled on the enroot process ({flavor:?})"
            );
            // the container's own environment is untouched: never passed via --env
            let via_container_env = command
                .windows(2)
                .any(|pair| pair[0] == "--env" && pair[1].starts_with("ENROOT_MOUNT_HOME"));
            assert!(!via_container_env);
        }
    }

    #[test]
    fn bubblewrap_flavor_builds_equivalent_namespace_prefix() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = config(temporary.path());
        config.unshare = "bwrap".to_owned();
        config.containment_flavor = ContainmentFlavor::Bubblewrap;
        let command = build_start_command(&config, "task").unwrap();
        assert_eq!(&command[..4], ["bwrap", "--dev-bind", "/", "/"]);
        let euid = unsafe { libc::geteuid() }.to_string();
        let egid = unsafe { libc::getegid() }.to_string();
        assert_eq!(
            &command[4..14],
            [
                "--unshare-user",
                "--uid",
                euid.as_str(),
                "--gid",
                egid.as_str(),
                "--unshare-pid",
                "--proc",
                "/proc",
                "--die-with-parent",
                "--",
            ]
        );
        let taskset = command.iter().position(|value| value == "taskset").unwrap();
        let enroot = command.iter().position(|value| value == "enroot").unwrap();
        assert!(taskset < enroot);
        assert!(!command.contains(&"--map-current-user".to_owned()));
    }

    #[test]
    fn wave_start_gate_control_is_typed_and_not_user_overridable() {
        let temporary = tempfile::tempdir().unwrap();
        let mut gated = config(temporary.path());
        gated.run_wave_start_gate = true;
        let command = build_start_command(&gated, "task").unwrap();
        assert_eq!(
            command
                .iter()
                .filter(|value| value.as_str() == "TASKTRACE_PROFILE_RUN_WAVE_START_GATE=1")
                .count(),
            1
        );

        let mut ungated = config(temporary.path());
        let command = build_start_command(&ungated, "task").unwrap();
        assert!(!command
            .iter()
            .any(|value| value.starts_with("TASKTRACE_PROFILE_RUN_WAVE_START_GATE=")));

        ungated
            .environment
            .push("TASKTRACE_PROFILE_RUN_WAVE_START_GATE=1".to_owned());
        assert!(ungated
            .validate()
            .unwrap_err()
            .to_string()
            .contains("profiler control variable"));
    }

    #[test]
    fn no_network_inserts_validating_helper_before_enroot() {
        let temporary = tempfile::tempdir().unwrap();
        fs::set_permissions(temporary.path(), fs::Permissions::from_mode(0o700)).unwrap();
        let mut config = config(temporary.path());
        config.network_mode = NetworkMode::NoNetwork;
        config.network_evidence_path = Some(temporary.path().join("network.json"));
        let command = build_start_command(&config, "task").unwrap();

        assert_eq!(command[7], "--net");
        assert_eq!(command[8], "--");
        let helper = command
            .iter()
            .position(|value| value == enroot_network::INTERNAL_COMMAND)
            .unwrap();
        let taskset = command.iter().position(|value| value == "taskset").unwrap();
        assert!(helper < taskset);
        assert_eq!(command[helper + 1], "--parent-netns-inode");
        assert!(command[helper..taskset].contains(&"--evidence-path".to_owned()));
    }

    #[test]
    fn network_evidence_configuration_is_exact() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = config(temporary.path());
        config.network_mode = NetworkMode::NoNetwork;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("requires a controller-private network evidence path"));

        config.network_mode = NetworkMode::Allowlist;
        assert!(config
            .validate()
            .unwrap_err()
            .to_string()
            .contains("does not support allowlist"));
    }

    #[test]
    fn lifecycle_uses_suspend_aware_boottime() {
        let timing = EnrootPhaseTiming::new(1, 2, "test").unwrap();
        assert_eq!(timing.clock_id, "host-boottime");
        assert_eq!(timing.clock_source, "CLOCK_BOOTTIME");
        assert!(timing.clock_resolution_ns >= 1);
    }
}
