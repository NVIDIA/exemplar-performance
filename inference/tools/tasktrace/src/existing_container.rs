// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native profiling for a Docker container that is already running.
//!
//! Existing-container measurements are diagnostic: the measured cgroup also
//! contains any unrelated work in the caller-owned container.  This module
//! consequently keeps lifecycle ownership with the caller, binds every
//! operation to the exact inspected container ID, and fails closed when the
//! container, cgroup, allocation, inherited network, or process population is
//! not stable.  It never updates, restarts, stops, removes, or reconnects the
//! container.

use crate::network_policy::NetworkPolicy;
use crate::process_exec;
use crate::timebase::{clock_reference, monotonic_ns, monotonic_resolution_ns};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Output, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::Duration;

pub const PROFILE_SCHEMA_VERSION: &str = "container-command-profile.v1";
pub const EXISTING_CONTAINER_SCHEMA_VERSION: &str = "tasktrace-existing-container-profile.v1";
const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const DEFAULT_SURVIVOR_GRACE: Duration = Duration::from_secs(1);
const DEFAULT_POLL_INTERVAL: Duration = Duration::from_millis(10);
const INTERNAL_ENV_PREFIX: &str = "TASKTRACE_EXISTING_";
const UNTRACED_SCRIPT: &str = "exec \"$TASKTRACE_EXISTING_SHELL\" -c \"$TASKTRACE_EXISTING_COMMAND\"";
const TRACED_SCRIPT: &str = "exec strace -f -ttt -s \"$TASKTRACE_EXISTING_STRACE_LIMIT\" -e \"trace=$TASKTRACE_EXISTING_STRACE_FILTER\" -o \"$TASKTRACE_EXISTING_STRACE_OUTPUT\" \"$TASKTRACE_EXISTING_SHELL\" -c \"$TASKTRACE_EXISTING_COMMAND\"";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ExistingContainerError(String);

impl ExistingContainerError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ExistingContainerError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ExistingContainerError {}

impl From<io::Error> for ExistingContainerError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ExistingContainerError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExistingStraceMode {
    Execve,
    Process,
    None,
}

impl ExistingStraceMode {
    pub fn parse(value: &str) -> Result<Self, ExistingContainerError> {
        match value {
            "execve" => Ok(Self::Execve),
            "process" => Ok(Self::Process),
            "none" => Ok(Self::None),
            _ => Err(ExistingContainerError::new(format!(
                "invalid choice for --strace-mode: {value:?} (choose from 'execve', 'process', 'none')"
            ))),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Execve => "execve",
            Self::Process => "process",
            Self::None => "none",
        }
    }

    fn filter(self) -> Option<&'static str> {
        match self {
            Self::Execve => Some("execve"),
            Self::Process => Some("%process,chdir,fchdir"),
            Self::None => None,
        }
    }
}

/// Options that would create or reconfigure a disposable sandbox.
///
/// The shared CLI parser populates this structure before it hands an
/// existing-container request to this module.  Keeping the rejection here as
/// well makes programmatic callers fail before any Docker or output mutation.
#[derive(Clone, Debug, Default)]
pub struct ExistingContainerUnsupportedOptions {
    pub image: Option<String>,
    pub cpuset_cpus: Option<String>,
    pub cpu_limit: Option<f64>,
    pub memory_limit_bytes: Option<u64>,
    pub memory_swap_total_limit_bytes: Option<u64>,
    pub mounts: Vec<String>,
    pub docker_network: Option<String>,
    pub effective_network_policy: Option<String>,
    pub allowed_hosts: Vec<String>,
    pub proxy_image_explicit: bool,
    pub external_proxy_network_explicit: bool,
    pub profile_sets: Vec<String>,
    pub coordinated_baseline: bool,
    pub verification_command: Option<String>,
    pub scheduler_concurrency: bool,
    pub proc_stat_comparison: bool,
    pub memory_profile: bool,
    pub terminate_descendants_at_finish: bool,
    pub other: Vec<String>,
}

impl ExistingContainerUnsupportedOptions {
    fn requested(&self) -> Vec<String> {
        let mut options = Vec::new();
        if self.image.is_some() {
            options.push("--image".to_owned());
        }
        if self.cpuset_cpus.is_some() {
            options.push("--cpuset-cpus".to_owned());
        }
        if self.cpu_limit.is_some() {
            options.push("--cpu-limit".to_owned());
        }
        if self.memory_limit_bytes.is_some() {
            options.push("--memory-limit-bytes".to_owned());
        }
        if self.memory_swap_total_limit_bytes.is_some() {
            options.push("--memory-swap-total-limit-bytes".to_owned());
        }
        if !self.mounts.is_empty() {
            options.push("--mount".to_owned());
        }
        if self.docker_network.is_some() {
            options.push("--network".to_owned());
        }
        if self
            .effective_network_policy
            .as_deref()
            .is_some_and(|mode| mode != "public")
        {
            options.push("--network-policy".to_owned());
        }
        if !self.allowed_hosts.is_empty() {
            options.push("--allowed-host".to_owned());
        }
        if self.proxy_image_explicit {
            options.push("--network-policy-proxy-image".to_owned());
        }
        if self.external_proxy_network_explicit {
            options.push("--network-policy-external-network".to_owned());
        }
        if !self.profile_sets.is_empty() {
            options.push("--profile-set".to_owned());
        }
        if self.coordinated_baseline {
            options.push("--coordinated-baseline".to_owned());
        }
        if self.verification_command.is_some() {
            options.push("--verification-command".to_owned());
        }
        if self.scheduler_concurrency {
            options.push("--scheduler-concurrency".to_owned());
        }
        if self.proc_stat_comparison {
            options.push("--proc-stat-comparison".to_owned());
        }
        if self.memory_profile {
            options.push("--memory-profile".to_owned());
        }
        if self.terminate_descendants_at_finish {
            options.push("--terminate-descendants-at-finish".to_owned());
        }
        options.extend(self.other.iter().cloned());
        options.sort();
        options.dedup();
        options
    }
}

#[derive(Clone, Debug)]
pub struct ExistingContainerConfig {
    pub container: String,
    pub command: String,
    pub output_dir: PathBuf,
    pub prepare_command: Option<String>,
    pub workdir: Option<String>,
    pub user: Option<String>,
    pub environment: Vec<String>,
    pub shell: String,
    pub strace_mode: ExistingStraceMode,
    pub string_limit: u64,
    pub allocated_cpus: Option<f64>,
    pub docker: String,
    pub unsupported: ExistingContainerUnsupportedOptions,
    pub survivor_grace: Duration,
    pub poll_interval: Duration,
    pub docker_exec_client_terminate_grace: Duration,
}

impl ExistingContainerConfig {
    pub fn new(
        container: impl Into<String>,
        command: impl Into<String>,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            container: container.into(),
            command: command.into(),
            output_dir: output_dir.into(),
            prepare_command: None,
            workdir: None,
            user: None,
            environment: Vec::new(),
            shell: "/bin/bash".to_owned(),
            strace_mode: ExistingStraceMode::Execve,
            string_limit: 4096,
            allocated_cpus: None,
            docker: "docker".to_owned(),
            unsupported: ExistingContainerUnsupportedOptions::default(),
            survivor_grace: DEFAULT_SURVIVOR_GRACE,
            poll_interval: DEFAULT_POLL_INTERVAL,
            docker_exec_client_terminate_grace: Duration::from_secs(1),
        }
    }
}

/// Return whether this public profile-container argv selects a caller-owned
/// running container.  This is deliberately syntax-only so the shared router
/// can enter the native parser without consulting Python.
pub fn is_existing_container_request(arguments: &[OsString]) -> bool {
    arguments.iter().any(|argument| {
        argument
            .to_str()
            .is_some_and(|value| value == "--container" || value.starts_with("--container="))
    })
}

/// Parse the stable existing-container CLI surface without touching Docker or
/// the output path.  Image-only options are parsed and then rejected by the
/// same programmatic validation used by `run_profile`.
pub fn parse_arguments(
    arguments: &[OsString],
) -> Result<Option<ExistingContainerConfig>, ExistingContainerError> {
    if arguments
        .iter()
        .any(|argument| matches!(argument.to_str(), Some("-h" | "--help")))
    {
        return Ok(None);
    }
    let mut container = None;
    let mut command = None;
    let mut output_dir = None;
    let mut runtime = "docker".to_owned();
    let mut prepare_command = None;
    let mut workdir = None;
    let mut user = None;
    let mut environment = Vec::new();
    let mut shell = "/bin/bash".to_owned();
    let mut strace_mode = ExistingStraceMode::Execve;
    let mut string_limit = 4096_u64;
    let mut allocated_cpus = None;
    let mut docker = "docker".to_owned();
    let mut unsupported = ExistingContainerUnsupportedOptions::default();
    let mut index = 0;
    while index < arguments.len() {
        let raw = arguments[index].to_str().ok_or_else(|| {
            ExistingContainerError::new("profile container arguments must be valid UTF-8")
        })?;
        let (option, attached) = raw
            .split_once('=')
            .map_or((raw, None), |(option, value)| (option, Some(value)));
        let mut take = || -> Result<String, ExistingContainerError> {
            if let Some(value) = attached {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .and_then(|value| value.to_str())
                .map(ToOwned::to_owned)
                .ok_or_else(|| {
                    ExistingContainerError::new(format!(
                        "argument {option}: expected one UTF-8 argument"
                    ))
                })
        };
        let flag = || -> Result<(), ExistingContainerError> {
            if attached.is_some() {
                Err(ExistingContainerError::new(format!(
                    "{option} does not take a value"
                )))
            } else {
                Ok(())
            }
        };
        match option {
            "--container" => container = Some(take()?),
            "--image" => unsupported.image = Some(take()?),
            "--runtime" => runtime = take()?,
            "--command" => command = Some(take()?),
            "--prepare-command" => prepare_command = Some(take()?),
            "--verification-command" => unsupported.verification_command = Some(take()?),
            "--output-dir" => output_dir = Some(PathBuf::from(take()?)),
            "--cpuset-cpus" => unsupported.cpuset_cpus = Some(take()?),
            "--allocated-cpus" => {
                let value = take()?;
                allocated_cpus = Some(value.parse::<f64>().map_err(|_| {
                    ExistingContainerError::new(format!(
                        "invalid value for --allocated-cpus: {value:?}"
                    ))
                })?);
            }
            "--cpu-limit" => {
                let value = take()?;
                unsupported.cpu_limit = Some(value.parse::<f64>().map_err(|_| {
                    ExistingContainerError::new(format!("invalid value for --cpu-limit: {value:?}"))
                })?);
            }
            "--workdir" => workdir = Some(take()?),
            "--user" => user = Some(take()?),
            "--network" => unsupported.docker_network = Some(take()?),
            "--network-policy" => {
                let value = take()?;
                if !matches!(value.as_str(), "public" | "no-network" | "allowlist") {
                    return Err(ExistingContainerError::new(format!(
                        "invalid choice for --network-policy: {value:?}"
                    )));
                }
                unsupported.effective_network_policy = Some(value);
            }
            "--allowed-host" => unsupported.allowed_hosts.push(take()?),
            "--network-policy-proxy-image" => {
                let _ = take()?;
                unsupported.proxy_image_explicit = true;
            }
            "--network-policy-external-network" => {
                let _ = take()?;
                unsupported.external_proxy_network_explicit = true;
            }
            "--shell" => shell = take()?,
            "--env" => environment.push(take()?),
            "--mount" => unsupported.mounts.push(take()?),
            "--strace-mode" => strace_mode = ExistingStraceMode::parse(&take()?)?,
            "--strace-string-limit" => {
                let value = take()?;
                string_limit = value.parse::<u64>().map_err(|_| {
                    ExistingContainerError::new(format!(
                        "invalid value for --strace-string-limit: {value:?}"
                    ))
                })?;
            }
            "--profile-set" => unsupported.profile_sets.push(take()?),
            "--coordinated-baseline" => {
                flag()?;
                unsupported.coordinated_baseline = true;
            }
            "--scheduler-concurrency" => {
                flag()?;
                unsupported.scheduler_concurrency = true;
            }
            "--terminate-descendants-at-finish" => {
                flag()?;
                unsupported.terminate_descendants_at_finish = true;
            }
            "--proc-stat-comparison" => {
                flag()?;
                unsupported.proc_stat_comparison = true;
            }
            "--memory-profile" => {
                flag()?;
                unsupported.memory_profile = true;
            }
            "--memory-limit-bytes" => {
                let value = take()?;
                unsupported.memory_limit_bytes = Some(value.parse::<u64>().map_err(|_| {
                    ExistingContainerError::new(format!(
                        "invalid value for --memory-limit-bytes: {value:?}"
                    ))
                })?);
            }
            "--memory-swap-total-limit-bytes" => {
                let value = take()?;
                unsupported.memory_swap_total_limit_bytes =
                    Some(value.parse::<u64>().map_err(|_| {
                        ExistingContainerError::new(format!(
                            "invalid value for --memory-swap-total-limit-bytes: {value:?}"
                        ))
                    })?);
            }
            "--hardware-perf-binary"
            | "--hardware-perf-event"
            | "--hardware-perf-attach-timeout-sec"
            | "--hardware-perf-minimum-running-ratio"
            | "--scheduler-backend"
            | "--scheduler-bcc-buffer-pages"
            | "--scheduler-perf-binary"
            | "--scheduler-perf-mmap-pages"
            | "--proc-stat-interval-sec"
            | "--proc-stat-baseline-sec"
            | "--memory-sample-interval-sec"
            | "--scheduler-attach-timeout-sec"
            | "--scheduler-cpu-tolerance-fraction"
            | "--scheduler-cpu-tolerance-sec" => {
                let _ = take()?;
                unsupported.other.push(option.to_owned());
            }
            "--docker" => docker = take()?,
            "--enroot" => {
                let _ = take()?;
            }
            "--" => {
                return Err(ExistingContainerError::new(
                    "unrecognized arguments after --",
                ))
            }
            _ => {
                return Err(ExistingContainerError::new(format!(
                    "unrecognized arguments: {raw}"
                )))
            }
        }
        index += 1;
    }
    if runtime != "docker" {
        return Err(ExistingContainerError::new(
            "--container mode is currently Docker-only",
        ));
    }
    let container = container
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ExistingContainerError::new("--container is required"))?;
    let command = command
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ExistingContainerError::new("--command is required"))?;
    let output_dir =
        output_dir.ok_or_else(|| ExistingContainerError::new("--output-dir is required"))?;
    let mut config = ExistingContainerConfig::new(container, command, output_dir);
    config.prepare_command = prepare_command;
    config.workdir = workdir;
    config.user = user;
    config.environment = environment;
    config.shell = shell;
    config.strace_mode = strace_mode;
    config.string_limit = string_limit;
    config.allocated_cpus = allocated_cpus;
    config.docker = docker;
    config.unsupported = unsupported;
    validate_config(&config)?;
    Ok(Some(config))
}

#[derive(Clone, Debug)]
pub struct ExistingContainerRun {
    pub report: Value,
    pub report_path: PathBuf,
    pub command_exit_code: i32,
}

trait DockerRunner: Send {
    fn run(&mut self, program: &OsStr, arguments: &[OsString]) -> io::Result<Output>;
    fn spawn(
        &mut self,
        program: &OsStr,
        arguments: &[OsString],
    ) -> io::Result<Box<dyn DockerProcess>>;
}

trait DockerProcess: Send {
    fn id(&self) -> u32;
    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>>;
    fn signal(&mut self, signal: libc::c_int) -> io::Result<()>;
    fn finish(self: Box<Self>, status: ExitStatus) -> io::Result<Output>;
}

struct SystemDockerRunner;

impl DockerRunner for SystemDockerRunner {
    fn run(&mut self, program: &OsStr, arguments: &[OsString]) -> io::Result<Output> {
        Command::new(program)
            .args(arguments)
            .stdin(Stdio::null())
            .output()
    }

    fn spawn(
        &mut self,
        program: &OsStr,
        arguments: &[OsString],
    ) -> io::Result<Box<dyn DockerProcess>> {
        let mut child = Command::new(program)
            .args(arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "Docker exec stdout pipe is unavailable",
            )
        })?;
        let stderr = child.stderr.take().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                "Docker exec stderr pipe is unavailable",
            )
        })?;
        Ok(Box::new(SystemDockerProcess {
            child: Some(child),
            stdout: Some(spawn_reader(stdout)),
            stderr: Some(spawn_reader(stderr)),
        }))
    }
}

struct SystemDockerProcess {
    child: Option<Child>,
    stdout: Option<thread::JoinHandle<io::Result<Vec<u8>>>>,
    stderr: Option<thread::JoinHandle<io::Result<Vec<u8>>>>,
}

impl DockerProcess for SystemDockerProcess {
    fn id(&self) -> u32 {
        self.child.as_ref().map(Child::id).unwrap_or(0)
    }

    fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
        self.child
            .as_mut()
            .ok_or_else(|| io::Error::new(io::ErrorKind::Other, "Docker exec client is closed"))?
            .try_wait()
    }

    fn signal(&mut self, signal: libc::c_int) -> io::Result<()> {
        let pid = self.id();
        if pid == 0 {
            return Ok(());
        }
        let result = unsafe { libc::kill(pid as libc::pid_t, signal) };
        if result == 0 {
            return Ok(());
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }

    fn finish(mut self: Box<Self>, status: ExitStatus) -> io::Result<Output> {
        // `try_wait` already reaped the child. Dropping the handle here cannot
        // signal a container process or alter container lifecycle.
        self.child.take();
        Ok(Output {
            status,
            stdout: join_reader(self.stdout.take(), "stdout")?,
            stderr: join_reader(self.stderr.take(), "stderr")?,
        })
    }
}

fn spawn_reader<R>(mut reader: R) -> thread::JoinHandle<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    thread::spawn(move || {
        let mut bytes = Vec::new();
        reader.read_to_end(&mut bytes)?;
        Ok(bytes)
    })
}

fn join_reader(
    reader: Option<thread::JoinHandle<io::Result<Vec<u8>>>>,
    name: &str,
) -> io::Result<Vec<u8>> {
    reader
        .ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Docker exec {name} reader is unavailable"),
            )
        })?
        .join()
        .map_err(|_| {
            io::Error::new(
                io::ErrorKind::Other,
                format!("Docker exec {name} reader panicked"),
            )
        })?
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct DockerAllocation {
    cpuset_cpus: String,
    nano_cpus: i64,
    cpu_quota: i64,
    cpu_period: i64,
    cpu_shares: i64,
    cgroup_parent: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NetworkAttachment {
    network_id: String,
    endpoint_id: String,
    ip_address: String,
    global_ipv6_address: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ContainerObservation {
    requested_reference: String,
    id: String,
    name: String,
    image_id: String,
    config_image: String,
    created: String,
    running: bool,
    paused: bool,
    restarting: bool,
    pid: u32,
    started_at: String,
    restart_count: u64,
    allocation: DockerAllocation,
    network_mode: String,
    networks: BTreeMap<String, NetworkAttachment>,
}

impl ContainerObservation {
    fn stable_identity(&self) -> Value {
        json!({
            "container_id": self.id,
            "container_name": self.name,
            "image_id": self.image_id,
            "created": self.created,
            "init_pid": self.pid,
            "started_at": self.started_at,
            "restart_count": self.restart_count,
        })
    }

    fn allocation_report(&self) -> Value {
        json!({
            "docker_host_config_cpuset_cpus": self.allocation.cpuset_cpus,
            "docker_host_config_nano_cpus": self.allocation.nano_cpus,
            "docker_host_config_cpu_quota": self.allocation.cpu_quota,
            "docker_host_config_cpu_period": self.allocation.cpu_period,
            "docker_host_config_cpu_shares": self.allocation.cpu_shares,
            "docker_host_config_cgroup_parent": self.allocation.cgroup_parent,
        })
    }

    fn network_report(&self) -> Value {
        let networks = self
            .networks
            .iter()
            .map(|(name, attachment)| {
                (
                    name.clone(),
                    json!({
                        "network_id": attachment.network_id,
                        "endpoint_id": attachment.endpoint_id,
                        "ip_address": attachment.ip_address,
                        "global_ipv6_address": attachment.global_ipv6_address,
                    }),
                )
            })
            .collect::<Map<_, _>>();
        json!({
            "mode": "inherited-existing-container",
            "docker_host_config_network_mode": self.network_mode,
            "attachments": networks,
            "topology_mutated": false,
        })
    }
}

#[derive(Clone, Debug, Eq, Ord, PartialEq, PartialOrd)]
struct ProcessIdentity {
    pid: u32,
    start_time_ticks: u64,
    comm: String,
}

impl ProcessIdentity {
    fn report(&self) -> Value {
        json!({
            "pid": self.pid,
            "start_time_ticks": self.start_time_ticks,
            "comm": self.comm,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScopeIdentity {
    path: PathBuf,
    inode: u64,
    device: u64,
    cgroup_type: String,
    init_pid: u32,
    init_start_time_ticks: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct KernelAllocation {
    cpu_max: String,
    cpuset_cpus_effective: String,
    cpuset_mems_effective: String,
    init_cpus_allowed_list: String,
    ancestor_chain: Vec<CgroupPolicyNode>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CgroupPolicyNode {
    path: PathBuf,
    inode: u64,
    device: u64,
    cpu_max: Option<String>,
    cpuset_cpus_effective: String,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CpuStat {
    raw: String,
    fields: BTreeMap<String, u64>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct ScopeObservation {
    identity: ScopeIdentity,
    allocation: KernelAllocation,
    processes: BTreeSet<ProcessIdentity>,
    cpu: CpuStat,
}

trait HostProbe: Send {
    fn observe(&mut self, init_pid: u32) -> Result<ScopeObservation, ExistingContainerError>;
    fn monotonic_ns(&mut self) -> u64;
    fn sleep(&mut self, duration: Duration);
}

struct SystemHostProbe;

impl HostProbe for SystemHostProbe {
    fn observe(&mut self, init_pid: u32) -> Result<ScopeObservation, ExistingContainerError> {
        observe_scope(init_pid)
    }

    fn monotonic_ns(&mut self) -> u64 {
        monotonic_ns()
    }

    fn sleep(&mut self, duration: Duration) {
        thread::sleep(duration);
    }
}

pub fn run_profile(
    config: ExistingContainerConfig,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<ExistingContainerRun, ExistingContainerError> {
    run_profile_with(
        config,
        Box::new(SystemDockerRunner),
        Box::new(SystemHostProbe),
        cancel_requested,
    )
}

fn run_profile_with(
    config: ExistingContainerConfig,
    mut docker: Box<dyn DockerRunner>,
    mut host: Box<dyn HostProbe>,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<ExistingContainerRun, ExistingContainerError> {
    validate_config(&config)?;
    preflight_output(&config.output_dir)?;

    // Resolve a mutable name/tag exactly once.  Every later operation uses
    // the full immutable container ID returned by this inspection.
    let initial_container = inspect_container(
        docker.as_mut(),
        &config.docker,
        &config.container,
        &config.container,
    )?;
    let image_identity =
        inspect_image(docker.as_mut(), &config.docker, &initial_container.image_id)?;
    let preflight_scope = host.observe(initial_container.pid)?;
    validate_container_scope(&initial_container, &preflight_scope)?;

    let (output_dir, _claim) = create_output(&config.output_dir)?;

    if let Some(prepare_command) = &config.prepare_command {
        let before_prepare = host.observe(initial_container.pid)?;
        let prepared = docker_exec(
            docker.as_mut(),
            host.as_mut(),
            &config,
            &initial_container.id,
            prepare_command,
            None,
            cancel_requested,
        )?;
        atomic_write(
            &output_dir.join("prepare.stdout.txt"),
            &prepared.output.stdout,
        )?;
        atomic_write(
            &output_dir.join("prepare.stderr.txt"),
            &prepared.output.stderr,
        )?;
        atomic_write(
            &output_dir.join("prepare_exit_code"),
            format!("{}\n", exit_code(prepared.output.status)).as_bytes(),
        )?;
        let after_prepare_container = inspect_container(
            docker.as_mut(),
            &config.docker,
            &initial_container.id,
            &config.container,
        )?;
        let after_prepare = host.observe(initial_container.pid)?;
        let prepare_stability = compare_stability(
            &initial_container,
            &after_prepare_container,
            &before_prepare,
            &after_prepare,
        );
        if prepared.client.cancellation_requested {
            let (finish, drain) = drain_to_baseline(
                host.as_mut(),
                initial_container.pid,
                &before_prepare,
                config.survivor_grace,
                config.poll_interval,
            )?;
            let finish_container = inspect_container(
                docker.as_mut(),
                &config.docker,
                &initial_container.id,
                &config.container,
            )?;
            let cleanup_stability = compare_stability(
                &initial_container,
                &finish_container,
                &before_prepare,
                &finish,
            );
            let evidence = json!({
                "schema_version": EXISTING_CONTAINER_SCHEMA_VERSION,
                "phase": "prepare",
                "valid": false,
                "reason": "cancellation requested while Docker exec preparation was active",
                "docker_exec_client_cleanup": prepared.client.report(),
                "scope_stability": cleanup_stability.report(),
                "survivor_handling": {
                    "policy": "bounded-natural-exit-no-signals",
                    "remaining_processes": process_reports(&drain.remaining),
                    "remaining_pids": drain.remaining.iter().map(|process| process.pid).collect::<Vec<_>>(),
                    "signals_sent": [],
                },
            });
            atomic_write_json(
                &output_dir.join("existing_container_cleanup.json"),
                &evidence,
            )?;
            return Err(ExistingContainerError::new(format!(
                "existing-container preparation was cancelled; invalid cleanup evidence retained at {}",
                output_dir.join("existing_container_cleanup.json").display()
            )));
        }
        if !prepare_stability.all_stable() {
            return Err(ExistingContainerError::new(
                "prepare command changed the existing container identity, cgroup, allocation, network, or process population",
            ));
        }
        if !prepared.output.status.success() {
            return Err(ExistingContainerError::new(format!(
                "prepare command failed with exit code {}{}",
                exit_code(prepared.output.status),
                output_suffix(&prepared.output)
            )));
        }
    }

    // This is the process population to which the controlled finish gate must
    // return naturally.  Existing-container mode must not signal processes it
    // does not own.
    let baseline_container = inspect_container(
        docker.as_mut(),
        &config.docker,
        &initial_container.id,
        &config.container,
    )?;
    let baseline = host.observe(initial_container.pid)?;
    validate_container_scope(&baseline_container, &baseline)?;
    let pre_measure_stability = compare_stability(
        &initial_container,
        &baseline_container,
        &preflight_scope,
        &baseline,
    );
    if !pre_measure_stability.identity_stable
        || !pre_measure_stability.scope_stable
        || !pre_measure_stability.allocation_stable
        || !pre_measure_stability.network_stable
    {
        return Err(ExistingContainerError::new(
            "existing container changed between preflight and measurement attachment",
        ));
    }

    let trace = match config.strace_mode.filter() {
        Some(filter) => Some(TraceCapture::new(filter)?),
        None => None,
    };
    let start_ns = host.monotonic_ns();
    let completed = docker_exec(
        docker.as_mut(),
        host.as_mut(),
        &config,
        &baseline_container.id,
        &config.command,
        trace.as_ref(),
        cancel_requested,
    )?;
    let command_return_ns = host.monotonic_ns();

    let (finish, drain) = drain_to_baseline(
        host.as_mut(),
        baseline_container.pid,
        &baseline,
        config.survivor_grace,
        config.poll_interval,
    )?;
    let end_ns = host.monotonic_ns();
    if end_ns <= start_ns || command_return_ns < start_ns || command_return_ns > end_ns {
        return Err(ExistingContainerError::new(
            "host CLOCK_MONOTONIC returned a non-increasing measurement window",
        ));
    }

    let final_container = inspect_container(
        docker.as_mut(),
        &config.docker,
        &baseline_container.id,
        &config.container,
    )?;
    let stability = compare_stability(&baseline_container, &final_container, &baseline, &finish);
    let cpu = cpu_report(
        &baseline.cpu,
        &finish.cpu,
        start_ns,
        end_ns,
        command_return_ns,
        config.allocated_cpus,
    )?;

    atomic_write(
        &output_dir.join("command.stdout.txt"),
        &completed.output.stdout,
    )?;
    atomic_write(
        &output_dir.join("command.stderr.txt"),
        &completed.output.stderr,
    )?;
    atomic_write(
        &output_dir.join("cpu.stat.before"),
        baseline.cpu.raw.as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("cpu.stat.after"),
        finish.cpu.raw.as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("start_ns"),
        format!("{start_ns}\n").as_bytes(),
    )?;
    atomic_write(&output_dir.join("end_ns"), format!("{end_ns}\n").as_bytes())?;
    let command_exit_code = exit_code(completed.output.status);
    atomic_write(
        &output_dir.join("exit_code"),
        format!("{command_exit_code}\n").as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("strace_mode"),
        format!("{}\n", config.strace_mode.as_str()).as_bytes(),
    )?;

    let mut trace_error = None;
    if let Some(trace) = &trace {
        if let Err(error) = retrieve_trace(
            docker.as_mut(),
            &config,
            &baseline_container.id,
            trace,
            &output_dir,
        ) {
            trace_error = Some(error.to_string());
        }
    }
    let process_exec_report =
        if config.strace_mode == ExistingStraceMode::Process && trace_error.is_none() {
            let report = process_exec::load_process_exec_trace(
                &output_dir.join("process.strace.log"),
                config.workdir.as_deref(),
                None,
            );
            let encoded = process_exec::serialize_process_exec_report(&report)
                .map_err(|error| ExistingContainerError::new(error.to_string()))?;
            atomic_write(
                &output_dir.join("process_exec_report.json"),
                encoded.as_bytes(),
            )?;
            Some(report)
        } else {
            None
        };

    // Trace retrieval and its narrowly scoped remote-file cleanup happen
    // outside the measured window.  Revalidate once more so even that cleanup
    // cannot hide a lifecycle or allocation transition.
    let post_artifact_container = inspect_container(
        docker.as_mut(),
        &config.docker,
        &baseline_container.id,
        &config.container,
    )?;
    let post_artifact_scope = host.observe(baseline_container.pid)?;
    let artifact_stability = compare_stability(
        &final_container,
        &post_artifact_container,
        &finish,
        &post_artifact_scope,
    );

    let stability_valid =
        stability.all_stable() && artifact_stability.all_stable() && drain.remaining.is_empty();
    let profiling_valid = stability_valid
        && trace_error.is_none()
        && !completed.client.cancellation_requested
        && completed.client.client_reaped
        && completed.client.cleanup_errors.is_empty();
    let environment = environment_provenance(&config.environment);
    let provenance = json!({
        "runtime": "docker",
        "scope": "existing-container",
        "container_id": baseline_container.id,
        "image_id": baseline_container.image_id,
        "cgroup_path": baseline.identity.path,
        "cgroup_inode": baseline.identity.inode,
        "cgroup_device": baseline.identity.device,
        "command": config.command,
        "prepare_command": config.prepare_command,
        "workdir": config.workdir,
        "user": config.user,
        "shell": config.shell,
        "environment": environment,
        "allocated_cpus": config.allocated_cpus,
        "strace_mode": config.strace_mode.as_str(),
        "network_inheritance": baseline_container.network_report(),
    });
    let comparability_sha256 = digest_json(&provenance);
    let scope_id = digest_json(&json!({
        "container_id": baseline_container.id,
        "image_id": baseline_container.image_id,
        "cgroup_path": baseline.identity.path,
        "cgroup_inode": baseline.identity.inode,
        "cgroup_device": baseline.identity.device,
        "init_start_time_ticks": baseline.identity.init_start_time_ticks,
    }));
    let mut network_policy = NetworkPolicy::default().to_report_payload();
    if let Some(policy) = network_policy.as_object_mut() {
        policy.insert(
            "comparability_sha256".to_owned(),
            Value::String(NetworkPolicy::default().comparability_digest()),
        );
        policy.insert("enforcement".to_owned(), Value::Null);
        policy.insert(
            "existing_container_inheritance".to_owned(),
            baseline_container.network_report(),
        );
    }

    let report = json!({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "runtime": "docker",
        "scope": "existing-container",
        "image": Value::Null,
        "image_identity": image_identity,
        "container": config.container,
        "command": config.command,
        "prepare_command": config.prepare_command,
        "network": Value::Null,
        "network_policy": network_policy,
        "shell": config.shell,
        "strace": {
            "mode": config.strace_mode.as_str(),
            "log_path": if trace.is_some() && trace_error.is_none() {
                json!(output_dir.join("process.strace.log"))
            } else {
                Value::Null
            },
            "structured_report_path": if process_exec_report.is_some() {
                json!(output_dir.join("process_exec_report.json"))
            } else {
                Value::Null
            },
            "string_limit": config.string_limit,
            "retrieval_error": trace_error,
        },
        "process_exec": process_exec_report,
        "command_exit_code": command_exit_code,
        "docker_returncode": command_exit_code,
        "measurement_validity": {
            "valid_for_average_cpu_reporting": profiling_valid,
            "cancelled": completed.client.cancellation_requested,
            "container_identity_stable": stability.identity_stable,
            "cgroup_scope_stable": stability.scope_stable,
            "allocation_stable": stability.allocation_stable,
            "network_inheritance_stable": stability.network_stable,
            "process_population_stable": stability.process_population_stable,
            "survivors_absent": drain.remaining.is_empty(),
            "diagnostic_existing_container_scope": true,
        },
        "cancellation": {
            "requested": completed.client.cancellation_requested,
            "sent_sigterm_to_host_docker_exec_client": completed.client.sigterm_sent,
            "sent_sigkill_to_host_docker_exec_client": completed.client.sigkill_sent,
            "host_docker_exec_client_reaped": completed.client.client_reaped,
            "container_processes_signaled": false,
            "container_lifecycle_mutated": false,
        },
        "cpu": cpu,
        "profiling_valid": profiling_valid,
        "warning": "Existing-container mode includes CPU from all other concurrent work in the container; use disposable image mode for comparable results.",
        "existing_container": {
            "schema_version": EXISTING_CONTAINER_SCHEMA_VERSION,
            "diagnostic_only": true,
            "requested_reference": config.container,
            "resolved_container": baseline_container.stable_identity(),
            "config_image_reference": baseline_container.config_image,
            "measured_scope": {
                "scope_id": format!("docker-existing-cgroup-{}", &scope_id[..24]),
                "kind": "existing-container-cgroup-subtree",
                "description": "Exact existing Docker container cgroup-v2 leaf and all descendant cgroups/processes during the controlled host-monotonic window.",
                "includes_descendants": true,
                "cgroup_path": baseline.identity.path,
                "cgroup_inode": baseline.identity.inode,
                "cgroup_device": baseline.identity.device,
                "cgroup_type": baseline.identity.cgroup_type,
                "init_pid": baseline.identity.init_pid,
                "init_start_time_ticks": baseline.identity.init_start_time_ticks,
            },
            "observed_allocation": {
                "docker": baseline_container.allocation_report(),
                "kernel": kernel_allocation_report(&baseline.allocation),
                "allocated_cpus_normalization_override": config.allocated_cpus,
                "normalization_override_enforced": false,
            },
            "network_inheritance": baseline_container.network_report(),
            "controlled_finish_observation": {
                "container": final_container.stable_identity(),
                "scope": {
                    "cgroup_path": finish.identity.path,
                    "cgroup_inode": finish.identity.inode,
                    "cgroup_device": finish.identity.device,
                    "init_start_time_ticks": finish.identity.init_start_time_ticks,
                },
                "allocation": {
                    "docker": final_container.allocation_report(),
                    "kernel": kernel_allocation_report(&finish.allocation),
                },
                "network_inheritance": final_container.network_report(),
            },
            "stability": stability.report(),
            "post_artifact_stability": artifact_stability.report(),
            "docker_exec_client_cleanup": completed.client.report(),
            "survivor_handling": {
                "policy": "bounded-natural-exit-no-signals",
                "grace_ns": duration_ns(config.survivor_grace),
                "waited_ns": drain.waited_ns,
                "processes_at_command_return": process_reports(&drain.initial),
                "remaining_processes_at_finish_gate": process_reports(&drain.remaining),
                "remaining_pids_at_finish_gate": drain.remaining.iter().map(|process| process.pid).collect::<Vec<_>>(),
                "signals_sent": [],
                "valid": drain.remaining.is_empty(),
            },
            "lifecycle_ownership": {
                "owner": "caller",
                "container_started": false,
                "container_updated": false,
                "container_restarted": false,
                "container_stopped": false,
                "container_removed": false,
                "network_topology_mutated": false,
            },
            "canonical_lifecycle": {
                "clock_id": "host-monotonic",
                "clock": "CLOCK_MONOTONIC",
                "resolution_ns": monotonic_resolution_ns(),
                "start_ns": start_ns,
                "docker_exec_return_ns": command_return_ns,
                "controlled_finish_ns": end_ns,
            },
            "execution_provenance": provenance,
            "comparability_sha256": comparability_sha256,
            "environment_values_redacted": true,
        },
        "clock_reference": clock_reference(),
    });
    let report_path = output_dir.join("profile_report.json");
    atomic_write_json(&report_path, &report)?;
    Ok(ExistingContainerRun {
        report,
        report_path,
        command_exit_code,
    })
}

fn validate_config(config: &ExistingContainerConfig) -> Result<(), ExistingContainerError> {
    if config.container.trim().is_empty() {
        return Err(ExistingContainerError::new("--container must not be empty"));
    }
    if config.command.trim().is_empty() {
        return Err(ExistingContainerError::new("--command must not be empty"));
    }
    if config.shell.trim().is_empty() || !config.shell.starts_with('/') {
        return Err(ExistingContainerError::new(
            "existing-container command shell must be an absolute nonempty path",
        ));
    }
    if config.string_limit == 0 {
        return Err(ExistingContainerError::new(
            "strace string limit must be positive",
        ));
    }
    if config.poll_interval.is_zero()
        || config.survivor_grace.is_zero()
        || config.docker_exec_client_terminate_grace.is_zero()
    {
        return Err(ExistingContainerError::new(
            "existing-container survivor, Docker-client termination, and poll intervals must be positive",
        ));
    }
    if config
        .allocated_cpus
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(ExistingContainerError::new(
            "allocated CPU count must be positive",
        ));
    }
    if let Some(command) = &config.prepare_command {
        if command.trim().is_empty() {
            return Err(ExistingContainerError::new(
                "prepare command must be nonempty when supplied",
            ));
        }
    }
    validate_environment(&config.environment)?;
    let unsupported = config.unsupported.requested();
    if !unsupported.is_empty() {
        return Err(ExistingContainerError::new(format!(
            "existing-container mode cannot create or reconfigure the caller-owned container; unsupported options: {}",
            unsupported.join(", ")
        )));
    }
    Ok(())
}

fn validate_environment(values: &[String]) -> Result<(), ExistingContainerError> {
    let mut names = BTreeSet::new();
    for assignment in values {
        let (name, _value) = assignment.split_once('=').ok_or_else(|| {
            ExistingContainerError::new(format!(
                "environment assignment must use NAME=VALUE syntax: {assignment:?}"
            ))
        })?;
        if name.is_empty()
            || !name.bytes().enumerate().all(|(index, byte)| {
                byte == b'_' || byte.is_ascii_alphabetic() || (index > 0 && byte.is_ascii_digit())
            })
        {
            return Err(ExistingContainerError::new(format!(
                "invalid environment variable name: {name:?}"
            )));
        }
        if name.starts_with(INTERNAL_ENV_PREFIX) {
            return Err(ExistingContainerError::new(format!(
                "environment variable {name:?} conflicts with native existing-container control state"
            )));
        }
        if !names.insert(name) {
            return Err(ExistingContainerError::new(format!(
                "environment variable {name:?} is assigned more than once"
            )));
        }
    }
    Ok(())
}

fn preflight_output(path: &Path) -> Result<(), ExistingContainerError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(ExistingContainerError::new(format!(
                    "profile output path must be a real directory: {}",
                    path.display()
                )));
            }
            if fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(ExistingContainerError::new(format!(
                    "profile output directory must be new or empty: {}",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut ancestor = path.parent().unwrap_or_else(|| Path::new("."));
            while !ancestor.exists() {
                ancestor = ancestor.parent().ok_or_else(|| {
                    ExistingContainerError::new(format!(
                        "profile output has no existing ancestor: {}",
                        path.display()
                    ))
                })?;
            }
            if !fs::metadata(ancestor)?.is_dir() {
                return Err(ExistingContainerError::new(format!(
                    "profile output ancestor is not a directory: {}",
                    ancestor.display()
                )));
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

struct OutputClaim {
    path: PathBuf,
}

impl Drop for OutputClaim {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn create_output(path: &Path) -> Result<(PathBuf, OutputClaim), ExistingContainerError> {
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(path)?;
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ExistingContainerError::new(format!(
            "profile output path changed after preflight: {}",
            path.display()
        )));
    }
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Err(ExistingContainerError::new(format!(
            "profile output directory changed after preflight: {}",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(path)?;
    let claim_path = canonical.join(".tasktrace-profile-claim");
    let mut claim = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&claim_path)?;
    writeln!(claim, "pid={}", std::process::id())?;
    claim.sync_all()?;
    Ok((canonical, OutputClaim { path: claim_path }))
}

fn inspect_container(
    docker: &mut dyn DockerRunner,
    executable: &str,
    reference: &str,
    requested_reference: &str,
) -> Result<ContainerObservation, ExistingContainerError> {
    let output = run_docker(docker, executable, &["container", "inspect", reference])?;
    if !output.status.success() {
        return Err(ExistingContainerError::new(format!(
            "unable to inspect running Docker container {reference:?}{}",
            output_suffix(&output)
        )));
    }
    let row = one_object(&output.stdout, "Docker container inspection")?;
    let id = required_string(&row, "Id", "container inspection")?.to_owned();
    if id.len() != 64 || !id.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ExistingContainerError::new(
            "Docker container inspection omitted a full immutable container ID",
        ));
    }
    let image_id = required_string(&row, "Image", "container inspection")?.to_owned();
    validate_image_id(&image_id)?;
    let state = required_object(&row, "State", "container inspection")?;
    let running = required_bool(state, "Running", "container state")?;
    let paused = state
        .get("Paused")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let restarting = state
        .get("Restarting")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    if !running || paused || restarting {
        return Err(ExistingContainerError::new(
            "Docker profiler container must be running, unpaused, and not restarting",
        ));
    }
    let pid = required_u64(state, "Pid", "container state").and_then(|value| {
        u32::try_from(value)
            .ok()
            .filter(|pid| *pid > 0)
            .ok_or_else(|| ExistingContainerError::new("container init PID is invalid"))
    })?;
    let host = required_object(&row, "HostConfig", "container inspection")?;
    let config = required_object(&row, "Config", "container inspection")?;
    let network_settings = required_object(&row, "NetworkSettings", "container inspection")?;
    let network_rows = required_object(network_settings, "Networks", "container networks")?;
    let mut networks = BTreeMap::new();
    for (name, value) in network_rows {
        let attachment = value.as_object().ok_or_else(|| {
            ExistingContainerError::new("Docker network attachment was not an object")
        })?;
        networks.insert(
            name.clone(),
            NetworkAttachment {
                network_id: optional_string(attachment, "NetworkID")?,
                endpoint_id: optional_string(attachment, "EndpointID")?,
                ip_address: optional_string(attachment, "IPAddress")?,
                global_ipv6_address: optional_string(attachment, "GlobalIPv6Address")?,
            },
        );
    }
    Ok(ContainerObservation {
        requested_reference: requested_reference.to_owned(),
        id,
        name: optional_string(&row, "Name")?
            .trim_start_matches('/')
            .to_owned(),
        image_id,
        config_image: optional_string(config, "Image")?,
        created: required_string(&row, "Created", "container inspection")?.to_owned(),
        running,
        paused,
        restarting,
        pid,
        started_at: required_string(state, "StartedAt", "container state")?.to_owned(),
        restart_count: row.get("RestartCount").and_then(Value::as_u64).unwrap_or(0),
        allocation: DockerAllocation {
            cpuset_cpus: optional_string(host, "CpusetCpus")?,
            nano_cpus: optional_i64(host, "NanoCpus")?,
            cpu_quota: optional_i64(host, "CpuQuota")?,
            cpu_period: optional_i64(host, "CpuPeriod")?,
            cpu_shares: optional_i64(host, "CpuShares")?,
            cgroup_parent: optional_string(host, "CgroupParent")?,
        },
        network_mode: optional_string(host, "NetworkMode")?,
        networks,
    })
}

fn inspect_image(
    docker: &mut dyn DockerRunner,
    executable: &str,
    image_id: &str,
) -> Result<Value, ExistingContainerError> {
    let output = run_docker(docker, executable, &["image", "inspect", image_id])?;
    if !output.status.success() {
        return Err(ExistingContainerError::new(format!(
            "unable to inspect immutable image {image_id:?}{}",
            output_suffix(&output)
        )));
    }
    let row = one_object(&output.stdout, "Docker image inspection")?;
    let observed = required_string(&row, "Id", "image inspection")?;
    validate_image_id(observed)?;
    if observed != image_id {
        return Err(ExistingContainerError::new(format!(
            "Docker image identity changed during inspection: container={image_id:?}, image={observed:?}"
        )));
    }
    let rootfs = row.get("RootFS").and_then(Value::as_object);
    let config = row.get("Config").and_then(Value::as_object);
    Ok(json!({
        "runtime": "docker",
        "id": observed,
        "repo_digests": row.get("RepoDigests").cloned().unwrap_or_else(|| json!([])),
        "created": row.get("Created").cloned().unwrap_or(Value::Null),
        "architecture": row.get("Architecture").cloned().unwrap_or(Value::Null),
        "os": row.get("Os").cloned().unwrap_or(Value::Null),
        "variant": row.get("Variant").cloned().unwrap_or(Value::Null),
        "size_bytes": row.get("Size").cloned().unwrap_or(Value::Null),
        "rootfs": {
            "type": rootfs.and_then(|value| value.get("Type")).cloned().unwrap_or(Value::Null),
            "layers": rootfs.and_then(|value| value.get("Layers")).cloned().unwrap_or_else(|| json!([])),
        },
        "config": {
            "user": config.and_then(|value| value.get("User")).cloned().unwrap_or(Value::Null),
            "working_dir": config.and_then(|value| value.get("WorkingDir")).cloned().unwrap_or(Value::Null),
            "entrypoint": config.and_then(|value| value.get("Entrypoint")).cloned().unwrap_or(Value::Null),
            "cmd": config.and_then(|value| value.get("Cmd")).cloned().unwrap_or(Value::Null),
            "shell": config.and_then(|value| value.get("Shell")).cloned().unwrap_or(Value::Null),
        },
        "immutable": true,
    }))
}

struct TraceCapture {
    remote_path: String,
    filter: String,
}

impl TraceCapture {
    fn new(filter: &str) -> Result<Self, ExistingContainerError> {
        let mut random = [0_u8; 8];
        File::open("/dev/urandom")?.read_exact(&mut random)?;
        let token = hex_bytes(&random);
        Ok(Self {
            remote_path: format!("/tmp/tasktrace-existing-{token}.strace"),
            filter: filter.to_owned(),
        })
    }
}

#[derive(Clone, Debug)]
struct DockerExecClientEvidence {
    pid: u32,
    cancellation_requested: bool,
    sigterm_sent: bool,
    sigkill_sent: bool,
    client_reaped: bool,
    cleanup_errors: Vec<String>,
}

impl DockerExecClientEvidence {
    fn report(&self) -> Value {
        json!({
            "host_docker_exec_client_pid": self.pid,
            "cancellation_requested": self.cancellation_requested,
            "sigterm_sent_to_host_client": self.sigterm_sent,
            "sigkill_sent_to_host_client": self.sigkill_sent,
            "host_client_reaped": self.client_reaped,
            "cleanup_errors": self.cleanup_errors,
            "container_processes_signaled": false,
            "container_lifecycle_mutated": false,
        })
    }
}

struct DockerExecOutcome {
    output: Output,
    client: DockerExecClientEvidence,
}

fn docker_exec(
    docker: &mut dyn DockerRunner,
    host: &mut dyn HostProbe,
    config: &ExistingContainerConfig,
    container_id: &str,
    command: &str,
    trace: Option<&TraceCapture>,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<DockerExecOutcome, ExistingContainerError> {
    if cancel_requested.is_some_and(|requested| requested()) {
        return Err(ExistingContainerError::new(
            "cancellation requested before Docker exec; no container work was started",
        ));
    }
    let mut arguments = vec![OsString::from("exec")];
    if let Some(workdir) = &config.workdir {
        arguments.extend([OsString::from("--workdir"), OsString::from(workdir)]);
    }
    if let Some(user) = &config.user {
        arguments.extend([OsString::from("--user"), OsString::from(user)]);
    }
    for assignment in &config.environment {
        arguments.extend([OsString::from("--env"), OsString::from(assignment)]);
    }
    for assignment in [
        format!("TASKTRACE_EXISTING_COMMAND={command}"),
        format!("TASKTRACE_EXISTING_SHELL={}", config.shell),
    ] {
        arguments.extend([OsString::from("--env"), OsString::from(assignment)]);
    }
    let script = if let Some(trace) = trace {
        for assignment in [
            format!("TASKTRACE_EXISTING_STRACE_LIMIT={}", config.string_limit),
            format!("TASKTRACE_EXISTING_STRACE_FILTER={}", trace.filter),
            format!("TASKTRACE_EXISTING_STRACE_OUTPUT={}", trace.remote_path),
        ] {
            arguments.extend([OsString::from("--env"), OsString::from(assignment)]);
        }
        TRACED_SCRIPT
    } else {
        UNTRACED_SCRIPT
    };
    arguments.extend([
        OsString::from(container_id),
        OsString::from("/bin/sh"),
        OsString::from("-c"),
        OsString::from(script),
    ]);
    let mut process = docker
        .spawn(OsStr::new(&config.docker), &arguments)
        .map_err(|error| {
            ExistingContainerError::new(format!("unable to start Docker exec: {error}"))
        })?;
    let mut evidence = DockerExecClientEvidence {
        pid: process.id(),
        cancellation_requested: false,
        sigterm_sent: false,
        sigkill_sent: false,
        client_reaped: false,
        cleanup_errors: Vec::new(),
    };
    let status = loop {
        if let Some(observed) = process.try_wait().map_err(|error| {
            ExistingContainerError::new(format!("unable to poll host Docker exec client: {error}"))
        })? {
            break Some(observed);
        }
        if cancel_requested.is_some_and(|requested| requested()) {
            evidence.cancellation_requested = true;
            match process.signal(libc::SIGTERM) {
                Ok(()) => evidence.sigterm_sent = true,
                Err(error) => evidence.cleanup_errors.push(format!(
                    "unable to SIGTERM host Docker exec client: {error}"
                )),
            }
            let mut cancellation_status = poll_client_until(
                process.as_mut(),
                host,
                config.docker_exec_client_terminate_grace,
                config.poll_interval,
            )?;
            if cancellation_status.is_none() {
                match process.signal(libc::SIGKILL) {
                    Ok(()) => evidence.sigkill_sent = true,
                    Err(error) => evidence.cleanup_errors.push(format!(
                        "unable to SIGKILL host Docker exec client: {error}"
                    )),
                }
                cancellation_status = poll_client_until(
                    process.as_mut(),
                    host,
                    config.docker_exec_client_terminate_grace,
                    config.poll_interval,
                )?;
            }
            break cancellation_status;
        }
        host.sleep(config.poll_interval);
    };
    if let Some(status) = status {
        evidence.client_reaped = true;
        let output = process.finish(status).map_err(|error| {
            ExistingContainerError::new(format!(
                "unable to collect host Docker exec client output: {error}"
            ))
        })?;
        return Ok(DockerExecOutcome {
            output,
            client: evidence,
        });
    }

    evidence
        .cleanup_errors
        .push("host Docker exec client remained after bounded SIGTERM/SIGKILL cleanup".to_owned());
    // The client was signaled by PID only.  Do not block indefinitely or
    // infer that its in-container exec process also stopped.
    drop(process);
    Ok(DockerExecOutcome {
        output: Output {
            status: ExitStatus::from_raw(libc::SIGKILL),
            stdout: Vec::new(),
            stderr: b"host Docker exec client did not exit after bounded cancellation cleanup\n"
                .to_vec(),
        },
        client: evidence,
    })
}

fn poll_client_until(
    process: &mut dyn DockerProcess,
    host: &mut dyn HostProbe,
    grace: Duration,
    poll: Duration,
) -> Result<Option<ExitStatus>, ExistingContainerError> {
    let deadline = host.monotonic_ns().saturating_add(duration_ns(grace));
    loop {
        if let Some(status) = process.try_wait().map_err(|error| {
            ExistingContainerError::new(format!(
                "unable to poll host Docker exec client during cancellation: {error}"
            ))
        })? {
            return Ok(Some(status));
        }
        if host.monotonic_ns() >= deadline {
            return Ok(None);
        }
        host.sleep(poll);
    }
}

fn retrieve_trace(
    docker: &mut dyn DockerRunner,
    config: &ExistingContainerConfig,
    container_id: &str,
    trace: &TraceCapture,
    output_dir: &Path,
) -> Result<(), ExistingContainerError> {
    let destination = output_dir.join("process.strace.log");
    let staging_suffix = digest_bytes(
        b"tasktrace-existing-container-trace-stage-v1\0",
        trace.remote_path.as_bytes(),
    );
    let staging = output_dir.join(format!(
        ".process.strace.log.docker-copy-{}",
        &staging_suffix[..16]
    ));
    if destination.exists() || staging.exists() {
        return Err(ExistingContainerError::new(
            "existing-container trace destination changed after output claim",
        ));
    }
    let source = format!("{container_id}:{}", trace.remote_path);
    let copied = docker.run(
        OsStr::new(&config.docker),
        &[
            OsString::from("cp"),
            OsString::from(source),
            staging.as_os_str().to_owned(),
        ],
    );
    let mut remove_arguments = vec![OsString::from("exec")];
    if let Some(user) = &config.user {
        remove_arguments.extend([OsString::from("--user"), OsString::from(user)]);
    }
    remove_arguments.extend([
        OsString::from(container_id),
        OsString::from("/bin/rm"),
        OsString::from("-f"),
        OsString::from("--"),
        OsString::from(&trace.remote_path),
    ]);
    // Attempt remote-file cleanup even when the copy CLI could not start.
    // This is the only resource created in the caller-owned container.
    let removed = docker.run(OsStr::new(&config.docker), &remove_arguments);
    let copied = copied.map_err(|error| {
        let _ = fs::remove_file(&staging);
        let cleanup = match &removed {
            Ok(output) if output.status.success() => "remote cleanup succeeded".to_owned(),
            Ok(output) => format!("remote cleanup also failed{}", output_suffix(output)),
            Err(cleanup_error) => format!("remote cleanup could not start: {cleanup_error}"),
        };
        ExistingContainerError::new(format!(
            "unable to start Docker trace copy: {error}; {cleanup}"
        ))
    })?;
    let removed = removed.map_err(|error| {
        let _ = fs::remove_file(&staging);
        ExistingContainerError::new(format!(
            "unable to start temporary existing-container trace cleanup: {error}"
        ))
    })?;
    if !copied.status.success() {
        let _ = fs::remove_file(&staging);
        return Err(ExistingContainerError::new(format!(
            "unable to retrieve existing-container strace artifact{}",
            output_suffix(&copied)
        )));
    }
    let metadata = fs::symlink_metadata(&staging).map_err(|error| {
        ExistingContainerError::new(format!(
            "Docker reported a successful trace copy but the staged artifact is unavailable: {error}"
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        let _ = fs::remove_file(&staging);
        return Err(ExistingContainerError::new(
            "Docker trace copy did not produce a regular file",
        ));
    }
    fs::set_permissions(&staging, fs::Permissions::from_mode(0o600))?;
    if destination.exists() {
        let _ = fs::remove_file(&staging);
        return Err(ExistingContainerError::new(
            "existing-container trace destination changed before publication",
        ));
    }
    fs::rename(&staging, &destination)?;
    if !removed.status.success() {
        return Err(ExistingContainerError::new(format!(
            "unable to remove temporary existing-container strace artifact{}",
            output_suffix(&removed)
        )));
    }
    Ok(())
}

fn observe_scope(init_pid: u32) -> Result<ScopeObservation, ExistingContainerError> {
    let root = fs::canonicalize(CGROUP_ROOT)?;
    let path = cgroup_path_for_pid(init_pid, &root)?;
    if path == root || !path.starts_with(&root) {
        return Err(ExistingContainerError::new(format!(
            "container init PID did not resolve to a dedicated cgroup-v2 scope: {}",
            path.display()
        )));
    }
    let mut last_race = None;
    for _ in 0..3 {
        let captured = (|| {
            let metadata = fs::metadata(&path)?;
            if !metadata.is_dir() {
                return Err(ExistingContainerError::new(
                    "measured cgroup scope is not a directory",
                ));
            }
            let init_process = read_process(init_pid)?;
            let processes = read_process_population(&path)?;
            if !processes.contains(&init_process) {
                return Err(ExistingContainerError::new(
                    "container init PID was absent from its cgroup subtree",
                ));
            }
            let identity = ScopeIdentity {
                path: path.clone(),
                inode: metadata.ino(),
                device: metadata.dev(),
                cgroup_type: read_trimmed(&path.join("cgroup.type"))?,
                init_pid,
                init_start_time_ticks: init_process.start_time_ticks,
            };
            let allocation = KernelAllocation {
                cpu_max: read_trimmed(&path.join("cpu.max"))?,
                cpuset_cpus_effective: read_trimmed(&path.join("cpuset.cpus.effective"))?,
                cpuset_mems_effective: read_trimmed(&path.join("cpuset.mems.effective"))?,
                init_cpus_allowed_list: process_cpuset(init_pid)?,
                ancestor_chain: read_cpu_policy_chain(&path, &root)?,
            };
            let cpu = read_cpu_stat(&path.join("cpu.stat"))?;

            // Close cgroup deletion/recreation, PID reuse, and membership
            // races around the multi-file snapshot.
            let final_metadata = fs::metadata(&path)?;
            let final_init = read_process(init_pid)?;
            let final_path = cgroup_path_for_pid(init_pid, &root)?;
            if final_metadata.ino() != metadata.ino()
                || final_metadata.dev() != metadata.dev()
                || final_init != init_process
                || final_path != path
            {
                return Err(ExistingContainerError::new(
                    "container cgroup or init-process identity changed during snapshot",
                ));
            }
            Ok(ScopeObservation {
                identity,
                allocation,
                processes,
                cpu,
            })
        })();
        match captured {
            Ok(observation) => return Ok(observation),
            Err(error) => last_race = Some(error.to_string()),
        }
        thread::yield_now();
    }
    Err(ExistingContainerError::new(format!(
        "unable to capture a stable descendant process scope: {}",
        last_race.unwrap_or_else(|| "unknown process-enumeration race".to_owned())
    )))
}

fn cgroup_path_for_pid(init_pid: u32, root: &Path) -> Result<PathBuf, ExistingContainerError> {
    let membership = fs::read_to_string(format!("/proc/{init_pid}/cgroup"))?;
    let relative = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| {
            ExistingContainerError::new(format!(
                "cgroup v2 membership not found for container init PID {init_pid}"
            ))
        })?;
    let relative = Path::new(relative.trim_start_matches('/'));
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ExistingContainerError::new(
            "container cgroup membership contains unsafe path components",
        ));
    }
    let path = fs::canonicalize(root.join(relative))?;
    if !path.starts_with(root) {
        return Err(ExistingContainerError::new(format!(
            "container init PID resolved outside the cgroup-v2 root: {}",
            path.display()
        )));
    }
    Ok(path)
}

fn read_cpu_policy_chain(
    leaf: &Path,
    root: &Path,
) -> Result<Vec<CgroupPolicyNode>, ExistingContainerError> {
    let mut chain = Vec::new();
    let mut current = leaf.to_path_buf();
    loop {
        let metadata = fs::metadata(&current)?;
        chain.push(CgroupPolicyNode {
            path: current
                .strip_prefix(root)
                .unwrap_or(Path::new(""))
                .to_path_buf(),
            inode: metadata.ino(),
            device: metadata.dev(),
            cpu_max: if current == root && !current.join("cpu.max").exists() {
                None
            } else {
                Some(read_trimmed(&current.join("cpu.max"))?)
            },
            cpuset_cpus_effective: read_trimmed(&current.join("cpuset.cpus.effective"))?,
        });
        if current == root {
            break;
        }
        current = current
            .parent()
            .ok_or_else(|| {
                ExistingContainerError::new("measured cgroup ancestor chain did not reach its root")
            })?
            .to_path_buf();
        if !current.starts_with(root) {
            return Err(ExistingContainerError::new(
                "measured cgroup ancestor chain escaped its root",
            ));
        }
    }
    Ok(chain)
}

fn read_process_population(
    cgroup: &Path,
) -> Result<BTreeSet<ProcessIdentity>, ExistingContainerError> {
    let mut pids = BTreeSet::new();
    read_cgroup_pids(cgroup, &mut pids)?;
    pids.into_iter().map(read_process).collect()
}

fn read_cgroup_pids(cgroup: &Path, pids: &mut BTreeSet<u32>) -> Result<(), ExistingContainerError> {
    let text = fs::read_to_string(cgroup.join("cgroup.procs")).map_err(|error| {
        ExistingContainerError::new(format!(
            "unable to read descendant process scope {}: {error}",
            cgroup.display()
        ))
    })?;
    for value in text.split_whitespace() {
        pids.insert(value.parse::<u32>().map_err(|_| {
            ExistingContainerError::new(format!("invalid PID in cgroup process scope: {value:?}"))
        })?);
    }
    for entry in fs::read_dir(cgroup)? {
        let entry = entry?;
        if entry.file_type()?.is_dir() {
            read_cgroup_pids(&entry.path(), pids)?;
        }
    }
    Ok(())
}

fn read_process(pid: u32) -> Result<ProcessIdentity, ExistingContainerError> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat")).map_err(|error| {
        ExistingContainerError::new(format!(
            "unable to inspect process identity for PID {pid}: {error}"
        ))
    })?;
    let (left, after) = text
        .rsplit_once(')')
        .ok_or_else(|| ExistingContainerError::new(format!("malformed /proc/{pid}/stat")))?;
    let comm = left
        .split_once('(')
        .map(|(_, comm)| comm)
        .filter(|comm| !comm.is_empty())
        .ok_or_else(|| ExistingContainerError::new(format!("malformed /proc/{pid}/stat")))?;
    let start_time_ticks = after
        .split_whitespace()
        .nth(19)
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| ExistingContainerError::new(format!("malformed /proc/{pid}/stat")))?;
    Ok(ProcessIdentity {
        pid,
        start_time_ticks,
        comm: comm.to_owned(),
    })
}

fn process_cpuset(pid: u32) -> Result<String, ExistingContainerError> {
    fs::read_to_string(format!("/proc/{pid}/status"))?
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:").map(str::trim))
        .filter(|value| !value.is_empty())
        .map(ToOwned::to_owned)
        .ok_or_else(|| {
            ExistingContainerError::new("container init process status omitted Cpus_allowed_list")
        })
}

fn read_trimmed(path: &Path) -> Result<String, ExistingContainerError> {
    let value = fs::read_to_string(path).map_err(|error| {
        ExistingContainerError::new(format!("unable to read {}: {error}", path.display()))
    })?;
    let value = value.trim();
    if value.is_empty() {
        return Err(ExistingContainerError::new(format!(
            "{} was empty",
            path.display()
        )));
    }
    Ok(value.to_owned())
}

fn read_cpu_stat(path: &Path) -> Result<CpuStat, ExistingContainerError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        ExistingContainerError::new(format!("unable to read {}: {error}", path.display()))
    })?;
    let mut fields = BTreeMap::new();
    for line in raw.lines() {
        let parts = line.split_whitespace().collect::<Vec<_>>();
        if parts.len() != 2 {
            return Err(ExistingContainerError::new(format!(
                "malformed cgroup cpu.stat record: {line:?}"
            )));
        }
        let value = parts[1].parse::<u64>().map_err(|_| {
            ExistingContainerError::new(format!("malformed cgroup cpu.stat value: {line:?}"))
        })?;
        if fields.insert(parts[0].to_owned(), value).is_some() {
            return Err(ExistingContainerError::new(format!(
                "duplicate cgroup cpu.stat field: {:?}",
                parts[0]
            )));
        }
    }
    for required in ["usage_usec", "user_usec", "system_usec"] {
        if !fields.contains_key(required) {
            return Err(ExistingContainerError::new(format!(
                "cgroup cpu.stat omitted required field {required:?}"
            )));
        }
    }
    Ok(CpuStat { raw, fields })
}

fn validate_container_scope(
    container: &ContainerObservation,
    scope: &ScopeObservation,
) -> Result<(), ExistingContainerError> {
    if container.pid != scope.identity.init_pid {
        return Err(ExistingContainerError::new(
            "Docker init PID does not match the measured cgroup scope",
        ));
    }
    if !scope.processes.iter().any(|process| {
        process.pid == container.pid
            && process.start_time_ticks == scope.identity.init_start_time_ticks
    }) {
        return Err(ExistingContainerError::new(
            "container init process identity is absent from the measured cgroup subtree",
        ));
    }
    parse_cpu_max(&scope.allocation.cpu_max)?;
    parse_cpuset(&scope.allocation.cpuset_cpus_effective)?;
    parse_cpuset(&scope.allocation.init_cpus_allowed_list)?;
    if scope.allocation.ancestor_chain.is_empty() {
        return Err(ExistingContainerError::new(
            "measured cgroup CPU-policy ancestor chain is empty",
        ));
    }
    for node in &scope.allocation.ancestor_chain {
        if let Some(cpu_max) = &node.cpu_max {
            parse_cpu_max(cpu_max)?;
        }
        parse_cpuset(&node.cpuset_cpus_effective)?;
    }
    if !container.allocation.cpuset_cpus.is_empty() {
        let docker = parse_cpuset(&container.allocation.cpuset_cpus)?;
        let kernel = parse_cpuset(&scope.allocation.cpuset_cpus_effective)?;
        let process = parse_cpuset(&scope.allocation.init_cpus_allowed_list)?;
        if docker != kernel || docker != process {
            return Err(ExistingContainerError::new(format!(
                "Docker, cgroup, and init-process CPU placement disagree: Docker={:?}, cgroup={:?}, process={:?}",
                container.allocation.cpuset_cpus,
                scope.allocation.cpuset_cpus_effective,
                scope.allocation.init_cpus_allowed_list,
            )));
        }
    }
    Ok(())
}

fn parse_cpu_max(value: &str) -> Result<(), ExistingContainerError> {
    let fields = value.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 2
        || (fields[0] != "max"
            && fields[0]
                .parse::<u64>()
                .ok()
                .filter(|value| *value > 0)
                .is_none())
        || fields[1]
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .is_none()
    {
        return Err(ExistingContainerError::new(format!(
            "invalid cgroup cpu.max value: {value:?}"
        )));
    }
    Ok(())
}

fn parse_cpuset(value: &str) -> Result<BTreeSet<u32>, ExistingContainerError> {
    let mut result = BTreeSet::new();
    for part in value.split(',') {
        if part.is_empty() {
            return Err(ExistingContainerError::new(format!(
                "invalid empty CPU-set component in {value:?}"
            )));
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start.parse::<u32>().map_err(|_| {
                ExistingContainerError::new(format!("invalid CPU-set range: {part:?}"))
            })?;
            let end = end.parse::<u32>().map_err(|_| {
                ExistingContainerError::new(format!("invalid CPU-set range: {part:?}"))
            })?;
            if end < start {
                return Err(ExistingContainerError::new(format!(
                    "invalid CPU-set range: {part:?}"
                )));
            }
            result.extend(start..=end);
        } else {
            result.insert(
                part.parse::<u32>().map_err(|_| {
                    ExistingContainerError::new(format!("invalid CPU ID: {part:?}"))
                })?,
            );
        }
    }
    if result.is_empty() {
        return Err(ExistingContainerError::new("CPU set must not be empty"));
    }
    Ok(result)
}

#[derive(Clone, Debug)]
struct Stability {
    identity_stable: bool,
    scope_stable: bool,
    allocation_stable: bool,
    network_stable: bool,
    process_population_stable: bool,
    added_processes: Vec<ProcessIdentity>,
    missing_processes: Vec<ProcessIdentity>,
}

impl Stability {
    fn all_stable(&self) -> bool {
        self.identity_stable
            && self.scope_stable
            && self.allocation_stable
            && self.network_stable
            && self.process_population_stable
    }

    fn report(&self) -> Value {
        json!({
            "container_identity_stable": self.identity_stable,
            "cgroup_scope_stable": self.scope_stable,
            "allocation_stable": self.allocation_stable,
            "network_inheritance_stable": self.network_stable,
            "process_population_stable": self.process_population_stable,
            "added_processes": process_reports(&self.added_processes),
            "missing_or_replaced_processes": process_reports(&self.missing_processes),
            "valid": self.all_stable(),
        })
    }
}

fn compare_stability(
    before_container: &ContainerObservation,
    after_container: &ContainerObservation,
    before: &ScopeObservation,
    after: &ScopeObservation,
) -> Stability {
    let added_processes = after
        .processes
        .difference(&before.processes)
        .cloned()
        .collect::<Vec<_>>();
    let missing_processes = before
        .processes
        .difference(&after.processes)
        .cloned()
        .collect::<Vec<_>>();
    Stability {
        identity_stable: before_container.id == after_container.id
            && before_container.name == after_container.name
            && before_container.image_id == after_container.image_id
            && before_container.config_image == after_container.config_image
            && before_container.created == after_container.created
            && before_container.pid == after_container.pid
            && before_container.started_at == after_container.started_at
            && before_container.restart_count == after_container.restart_count
            && after_container.running
            && !after_container.paused
            && !after_container.restarting,
        scope_stable: before.identity == after.identity,
        allocation_stable: before_container.allocation == after_container.allocation
            && before.allocation == after.allocation,
        network_stable: before_container.network_mode == after_container.network_mode
            && before_container.networks == after_container.networks,
        process_population_stable: added_processes.is_empty() && missing_processes.is_empty(),
        added_processes,
        missing_processes,
    }
}

struct DrainResult {
    initial: Vec<ProcessIdentity>,
    remaining: Vec<ProcessIdentity>,
    waited_ns: u64,
}

fn drain_to_baseline(
    host: &mut dyn HostProbe,
    init_pid: u32,
    baseline: &ScopeObservation,
    grace: Duration,
    poll: Duration,
) -> Result<(ScopeObservation, DrainResult), ExistingContainerError> {
    let started = host.monotonic_ns();
    let deadline = started.saturating_add(duration_ns(grace));
    let mut observation = host.observe(init_pid)?;
    let initial = observation
        .processes
        .difference(&baseline.processes)
        .cloned()
        .collect::<Vec<_>>();
    loop {
        let remaining = observation
            .processes
            .difference(&baseline.processes)
            .cloned()
            .collect::<Vec<_>>();
        let missing = baseline
            .processes
            .difference(&observation.processes)
            .next()
            .is_some();
        if remaining.is_empty() || host.monotonic_ns() >= deadline {
            return Ok((
                observation,
                DrainResult {
                    initial,
                    remaining,
                    waited_ns: host.monotonic_ns().saturating_sub(started),
                },
            ));
        }
        // Missing baseline work is a stability failure, but it cannot be
        // repaired by waiting and must never be replaced or signaled by us.
        if missing {
            return Ok((
                observation,
                DrainResult {
                    initial,
                    remaining,
                    waited_ns: host.monotonic_ns().saturating_sub(started),
                },
            ));
        }
        host.sleep(poll);
        observation = host.observe(init_pid)?;
    }
}

fn cpu_report(
    before: &CpuStat,
    after: &CpuStat,
    start_ns: u64,
    end_ns: u64,
    command_return_ns: u64,
    allocated_cpus: Option<f64>,
) -> Result<Value, ExistingContainerError> {
    let delta = |field: &str| -> Result<u64, ExistingContainerError> {
        let before = before.fields.get(field).copied().ok_or_else(|| {
            ExistingContainerError::new(format!("initial cpu.stat omitted {field}"))
        })?;
        let after = after.fields.get(field).copied().ok_or_else(|| {
            ExistingContainerError::new(format!("final cpu.stat omitted {field}"))
        })?;
        after.checked_sub(before).ok_or_else(|| {
            ExistingContainerError::new(format!(
                "cgroup cpu.stat field {field} regressed during measurement"
            ))
        })
    };
    let wall_sec = (end_ns - start_ns) as f64 / 1_000_000_000.0;
    let command_wall_sec = (command_return_ns - start_ns) as f64 / 1_000_000_000.0;
    let usage_sec = delta("usage_usec")? as f64 / 1_000_000.0;
    let user_sec = delta("user_usec")? as f64 / 1_000_000.0;
    let system_sec = delta("system_usec")? as f64 / 1_000_000.0;
    let average = usage_sec / wall_sec;
    let utilization = allocated_cpus.map(|capacity| average / capacity);
    Ok(json!({
        "method": "container-cgroup-v2-cpu.stat",
        "wall_sec": round9(wall_sec),
        "command_wall_sec": round9(command_wall_sec),
        "controlled_finish_drain_sec": round9(wall_sec - command_wall_sec),
        "usage_sec": round9(usage_sec),
        "user_sec": round9(user_sec),
        "system_sec": round9(system_sec),
        "average_logical_cores_used": round9(average),
        "allocated_logical_cpus": allocated_cpus,
        "enforced_logical_cpu_limit": Value::Null,
        "enforced_cpu_limit_source": Value::Null,
        "allocated_cpu_source": if allocated_cpus.is_some() {
            json!("user-supplied")
        } else {
            Value::Null
        },
        "utilization_fraction_of_allocation": utilization.map(round9),
        "utilization_percent_of_allocation": utilization.map(|value| round6(100.0 * value)),
        "scope_includes_descendants": true,
        "scope_includes_unrelated_existing_container_work": true,
    }))
}

fn kernel_allocation_report(allocation: &KernelAllocation) -> Value {
    json!({
        "cgroup_cpu_max": allocation.cpu_max,
        "cgroup_cpuset_cpus_effective": allocation.cpuset_cpus_effective,
        "cgroup_cpuset_mems_effective": allocation.cpuset_mems_effective,
        "container_init_cpus_allowed_list": allocation.init_cpus_allowed_list,
        "ancestor_cpu_policy_chain": allocation.ancestor_chain.iter().map(|node| json!({
            "path_relative_to_cgroup_root": node.path,
            "inode": node.inode,
            "device": node.device,
            "cpu_max": node.cpu_max,
            "cpu_max_applicable": node.cpu_max.is_some(),
            "cpuset_cpus_effective": node.cpuset_cpus_effective,
        })).collect::<Vec<_>>(),
    })
}

fn process_reports(processes: &[ProcessIdentity]) -> Vec<Value> {
    processes.iter().map(ProcessIdentity::report).collect()
}

fn environment_provenance(assignments: &[String]) -> Vec<Value> {
    let mut values = assignments
        .iter()
        .filter_map(|assignment| assignment.split_once('='))
        .map(|(key, value)| {
            json!({
                "key": key,
                "value_sha256": digest_bytes(b"tasktrace-profile-environment-value-v1\0", value.as_bytes()),
                "assignment_had_equals": true,
            })
        })
        .collect::<Vec<_>>();
    values.sort_by_key(canonical_json);
    values
}

fn atomic_write_json(path: &Path, value: &Value) -> Result<(), ExistingContainerError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    atomic_write(path, &bytes)
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ExistingContainerError> {
    static SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let name = path.file_name().and_then(OsStr::to_str).ok_or_else(|| {
        ExistingContainerError::new(format!("invalid output filename: {}", path.display()))
    })?;
    let temporary = path.with_file_name(format!(
        ".{name}.tmp.{}.{}",
        std::process::id(),
        SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path).map_err(|error| {
        ExistingContainerError::new(format!(
            "unable to publish profile artifact {}: {error}",
            path.display()
        ))
    })
}

fn run_docker(
    docker: &mut dyn DockerRunner,
    executable: &str,
    arguments: &[&str],
) -> Result<Output, ExistingContainerError> {
    docker
        .run(
            OsStr::new(executable),
            &arguments.iter().map(OsString::from).collect::<Vec<_>>(),
        )
        .map_err(|error| ExistingContainerError::new(format!("unable to run Docker CLI: {error}")))
}

fn one_object(
    bytes: &[u8],
    description: &str,
) -> Result<Map<String, Value>, ExistingContainerError> {
    let payload: Value = serde_json::from_slice(bytes).map_err(|_| {
        ExistingContainerError::new(format!("{description} returned malformed JSON"))
    })?;
    let rows = payload
        .as_array()
        .filter(|rows| rows.len() == 1)
        .ok_or_else(|| {
            ExistingContainerError::new(format!("{description} must return exactly one object"))
        })?;
    rows[0].as_object().cloned().ok_or_else(|| {
        ExistingContainerError::new(format!("{description} must return exactly one object"))
    })
}

fn required_object<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    description: &str,
) -> Result<&'a Map<String, Value>, ExistingContainerError> {
    value.get(field).and_then(Value::as_object).ok_or_else(|| {
        ExistingContainerError::new(format!("{description} omitted object field {field}"))
    })
}

fn required_string<'a>(
    value: &'a Map<String, Value>,
    field: &str,
    description: &str,
) -> Result<&'a str, ExistingContainerError> {
    value
        .get(field)
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            ExistingContainerError::new(format!(
                "{description} omitted nonempty string field {field}"
            ))
        })
}

fn required_bool(
    value: &Map<String, Value>,
    field: &str,
    description: &str,
) -> Result<bool, ExistingContainerError> {
    value.get(field).and_then(Value::as_bool).ok_or_else(|| {
        ExistingContainerError::new(format!("{description} omitted boolean field {field}"))
    })
}

fn required_u64(
    value: &Map<String, Value>,
    field: &str,
    description: &str,
) -> Result<u64, ExistingContainerError> {
    value.get(field).and_then(Value::as_u64).ok_or_else(|| {
        ExistingContainerError::new(format!("{description} omitted integer field {field}"))
    })
}

fn optional_string(
    value: &Map<String, Value>,
    field: &str,
) -> Result<String, ExistingContainerError> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(String::new()),
        Some(Value::String(value)) => Ok(value.clone()),
        Some(_) => Err(ExistingContainerError::new(format!(
            "Docker inspection field {field} was not a string"
        ))),
    }
}

fn optional_i64(value: &Map<String, Value>, field: &str) -> Result<i64, ExistingContainerError> {
    match value.get(field) {
        None | Some(Value::Null) => Ok(0),
        Some(value) => value.as_i64().ok_or_else(|| {
            ExistingContainerError::new(format!(
                "Docker inspection field {field} was not an integer"
            ))
        }),
    }
}

fn validate_image_id(value: &str) -> Result<(), ExistingContainerError> {
    let digest = value.strip_prefix("sha256:").unwrap_or_default();
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ExistingContainerError::new(
            "Docker inspection did not return an immutable sha256 image ID",
        ));
    }
    Ok(())
}

fn exit_code(status: ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

fn output_suffix(output: &Output) -> String {
    let bytes = if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    };
    let diagnostic = String::from_utf8_lossy(bytes);
    let diagnostic = diagnostic.trim();
    if diagnostic.is_empty() {
        format!(" (exit code {})", exit_code(output.status))
    } else {
        format!(
            " (exit code {}): {}",
            exit_code(output.status),
            diagnostic.chars().take(1000).collect::<String>()
        )
    }
}

fn duration_ns(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn round9(value: f64) -> f64 {
    (value * 1_000_000_000.0).round() / 1_000_000_000.0
}

fn round6(value: f64) -> f64 {
    (value * 1_000_000.0).round() / 1_000_000.0
}

fn digest_bytes(domain: &[u8], bytes: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(bytes);
    hex_bytes(&digest.finalize())
}

fn hex_bytes(bytes: &[u8]) -> String {
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn digest_json(value: &Value) -> String {
    digest_bytes(
        b"tasktrace-existing-container-json-v1\0",
        canonical_json(value).as_bytes(),
    )
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("serialize JSON string"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut fields = values.iter().collect::<Vec<_>>();
            fields.sort_by(|(left, _), (right, _)| left.cmp(right));
            format!(
                "{{{}}}",
                fields
                    .into_iter()
                    .map(|(name, value)| format!(
                        "{}:{}",
                        serde_json::to_string(name).expect("serialize JSON field"),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering as AtomicOrdering};
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    const CONTAINER_ID: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    const IMAGE_ID: &str =
        "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

    #[derive(Default)]
    struct DockerState {
        calls: Vec<Vec<String>>,
        allocation_changed: bool,
        network_changed: bool,
        command_status: i32,
        copy_io_error: bool,
        remove_status: i32,
        trace_destination: Option<PathBuf>,
        exec_blocks_until_kill: bool,
        client_never_reaps: bool,
        client_signals: Vec<i32>,
        exec_spawned: bool,
    }

    struct FakeDocker {
        state: Arc<Mutex<DockerState>>,
    }

    impl FakeDocker {
        fn success(bytes: impl AsRef<[u8]>) -> Output {
            Output {
                status: ExitStatus::from_raw(0),
                stdout: bytes.as_ref().to_vec(),
                stderr: Vec::new(),
            }
        }

        fn status(code: i32, stdout: &str, stderr: &str) -> Output {
            Output {
                status: ExitStatus::from_raw(code << 8),
                stdout: stdout.as_bytes().to_vec(),
                stderr: stderr.as_bytes().to_vec(),
            }
        }

        fn container(state: &DockerState) -> Value {
            let cpuset = if state.allocation_changed && state.exec_spawned {
                "0"
            } else {
                "0-1"
            };
            let network = if state.network_changed && state.exec_spawned {
                "changed"
            } else {
                "bridge"
            };
            json!({
                "Id": CONTAINER_ID,
                "Name": "/fixture-live",
                "Image": IMAGE_ID,
                "Created": "2026-01-01T00:00:00Z",
                "RestartCount": 0,
                "State": {
                    "Running": true,
                    "Paused": false,
                    "Restarting": false,
                    "Pid": 4242,
                    "StartedAt": "2026-01-01T00:00:01Z",
                },
                "Config": {"Image": "fixture:tag"},
                "HostConfig": {
                    "CpusetCpus": cpuset,
                    "NanoCpus": 0,
                    "CpuQuota": 0,
                    "CpuPeriod": 0,
                    "CpuShares": 0,
                    "CgroupParent": "",
                    "NetworkMode": network,
                },
                "NetworkSettings": {"Networks": {
                    network: {
                        "NetworkID": format!("{network}-id"),
                        "EndpointID": "endpoint-id",
                        "IPAddress": "172.17.0.2",
                        "GlobalIPv6Address": "",
                    }
                }},
            })
        }
    }

    impl DockerRunner for FakeDocker {
        fn run(&mut self, _program: &OsStr, raw: &[OsString]) -> io::Result<Output> {
            let arguments = raw
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let mut state = self.state.lock().unwrap();
            state.calls.push(arguments.clone());
            match arguments.first().map(String::as_str) {
                Some("container") if arguments.get(1).map(String::as_str) == Some("inspect") => Ok(
                    Self::success(serde_json::to_vec(&vec![Self::container(&state)])?),
                ),
                Some("image") if arguments.get(1).map(String::as_str) == Some("inspect") => {
                    Ok(Self::success(serde_json::to_vec(&vec![json!({
                        "Id": IMAGE_ID,
                        "RepoDigests": ["fixture@sha256:cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc"],
                        "Created": "2026-01-01T00:00:00Z",
                        "Architecture": "amd64",
                        "Os": "linux",
                        "Size": 123,
                        "RootFS": {"Type": "layers", "Layers": ["sha256:layer"]},
                    })])?))
                }
                Some("exec") if arguments.iter().any(|value| value == "/bin/rm") => {
                    Ok(Self::status(state.remove_status, "", "remove failed"))
                }
                Some("cp") => {
                    if state.copy_io_error {
                        return Err(io::Error::new(
                            io::ErrorKind::Other,
                            "fixture copy could not start",
                        ));
                    }
                    let destination = PathBuf::from(arguments.last().unwrap());
                    fs::write(&destination, b"4242 execve(\"/bin/true\", ...) = 0\n")?;
                    state.trace_destination = Some(destination);
                    Ok(Self::success(""))
                }
                _ => panic!("unexpected fake Docker command: {arguments:?}"),
            }
        }

        fn spawn(
            &mut self,
            _program: &OsStr,
            raw: &[OsString],
        ) -> io::Result<Box<dyn DockerProcess>> {
            let arguments = raw
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            assert_eq!(arguments.first().map(String::as_str), Some("exec"));
            assert!(arguments
                .iter()
                .any(|value| value == UNTRACED_SCRIPT || value == TRACED_SCRIPT));
            let status = self.state.lock().unwrap().command_status;
            {
                let mut state = self.state.lock().unwrap();
                state.calls.push(arguments);
                state.exec_spawned = true;
            }
            Ok(Box::new(FakeDockerProcess {
                state: Arc::clone(&self.state),
                status,
            }))
        }
    }

    struct FakeDockerProcess {
        state: Arc<Mutex<DockerState>>,
        status: i32,
    }

    impl DockerProcess for FakeDockerProcess {
        fn id(&self) -> u32 {
            9100
        }

        fn try_wait(&mut self) -> io::Result<Option<ExitStatus>> {
            let state = self.state.lock().unwrap();
            if state.client_never_reaps
                || (state.exec_blocks_until_kill
                    && !state
                        .client_signals
                        .iter()
                        .any(|signal| *signal == libc::SIGKILL))
            {
                Ok(None)
            } else if state
                .client_signals
                .iter()
                .any(|signal| *signal == libc::SIGKILL)
            {
                Ok(Some(ExitStatus::from_raw(libc::SIGKILL)))
            } else {
                Ok(Some(ExitStatus::from_raw(self.status << 8)))
            }
        }

        fn signal(&mut self, signal: libc::c_int) -> io::Result<()> {
            self.state.lock().unwrap().client_signals.push(signal);
            Ok(())
        }

        fn finish(self: Box<Self>, status: ExitStatus) -> io::Result<Output> {
            Ok(Output {
                status,
                stdout: b"workload stdout\n".to_vec(),
                stderr: b"workload stderr\n".to_vec(),
            })
        }
    }

    #[derive(Default)]
    struct HostState {
        calls: usize,
        now: u64,
        survivor: bool,
        allocation_changed: bool,
    }

    struct FakeHost {
        state: Arc<Mutex<HostState>>,
    }

    fn process(pid: u32, start: u64, comm: &str) -> ProcessIdentity {
        ProcessIdentity {
            pid,
            start_time_ticks: start,
            comm: comm.to_owned(),
        }
    }

    impl HostProbe for FakeHost {
        fn observe(&mut self, _init_pid: u32) -> Result<ScopeObservation, ExistingContainerError> {
            let mut state = self.state.lock().unwrap();
            state.calls += 1;
            let usage = if state.calls <= 2 {
                1_000_000
            } else {
                3_000_000
            };
            let mut processes =
                BTreeSet::from([process(4242, 10, "init"), process(4243, 11, "service")]);
            // The first two observations establish preflight and measurement
            // baselines.  A configured survivor appears only afterward.
            if state.survivor && state.calls > 2 {
                processes.insert(process(5000, 50, "leftover"));
            }
            Ok(ScopeObservation {
                identity: ScopeIdentity {
                    path: PathBuf::from("/sys/fs/cgroup/system.slice/docker-fixture.scope"),
                    inode: 100,
                    device: 1,
                    cgroup_type: "domain".to_owned(),
                    init_pid: 4242,
                    init_start_time_ticks: 10,
                },
                allocation: KernelAllocation {
                    cpu_max: "max 100000".to_owned(),
                    cpuset_cpus_effective: if state.allocation_changed && state.calls > 2 {
                        "0".to_owned()
                    } else {
                        "0-1".to_owned()
                    },
                    cpuset_mems_effective: "0".to_owned(),
                    init_cpus_allowed_list: if state.allocation_changed && state.calls > 2 {
                        "0".to_owned()
                    } else {
                        "0-1".to_owned()
                    },
                    ancestor_chain: vec![CgroupPolicyNode {
                        path: PathBuf::from("system.slice/docker-fixture.scope"),
                        inode: 100,
                        device: 1,
                        cpu_max: Some("max 100000".to_owned()),
                        cpuset_cpus_effective: if state.allocation_changed && state.calls > 2 {
                            "0".to_owned()
                        } else {
                            "0-1".to_owned()
                        },
                    }],
                },
                processes,
                cpu: CpuStat {
                    raw: format!(
                        "usage_usec {usage}\nuser_usec {}\nsystem_usec {}\n",
                        usage * 3 / 4,
                        usage / 4
                    ),
                    fields: BTreeMap::from([
                        ("usage_usec".to_owned(), usage),
                        ("user_usec".to_owned(), usage * 3 / 4),
                        ("system_usec".to_owned(), usage / 4),
                    ]),
                },
            })
        }

        fn monotonic_ns(&mut self) -> u64 {
            let mut state = self.state.lock().unwrap();
            state.now += 250_000_000;
            state.now
        }

        fn sleep(&mut self, duration: Duration) {
            let mut state = self.state.lock().unwrap();
            state.now = state.now.saturating_add(duration_ns(duration));
        }
    }

    fn fixture() -> (
        TempDir,
        ExistingContainerConfig,
        Arc<Mutex<DockerState>>,
        Arc<Mutex<HostState>>,
    ) {
        let temporary = TempDir::new().unwrap();
        let mut config = ExistingContainerConfig::new(
            "fixture-live",
            "printf work",
            temporary.path().join("output"),
        );
        config.strace_mode = ExistingStraceMode::None;
        config.allocated_cpus = Some(2.0);
        config.survivor_grace = Duration::from_millis(20);
        config.poll_interval = Duration::from_millis(5);
        let docker = Arc::new(Mutex::new(DockerState::default()));
        let host = Arc::new(Mutex::new(HostState::default()));
        (temporary, config, docker, host)
    }

    fn arguments(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    #[test]
    fn native_cli_parser_accepts_the_complete_existing_container_surface() {
        let parsed = parse_arguments(&arguments(&[
            "--container=live",
            "--runtime",
            "docker",
            "--command",
            "make -j2",
            "--prepare-command",
            "command -v make",
            "--output-dir",
            "/tmp/existing-output",
            "--allocated-cpus",
            "2",
            "--workdir",
            "/workdir",
            "--user",
            "1000:1000",
            "--env",
            "MODE=test",
            "--shell",
            "/bin/sh",
            "--strace-mode",
            "process",
            "--strace-string-limit",
            "8192",
            "--network-policy",
            "public",
            "--docker",
            "/usr/bin/docker",
        ]))
        .unwrap()
        .unwrap();

        assert_eq!(parsed.container, "live");
        assert_eq!(parsed.command, "make -j2");
        assert_eq!(parsed.prepare_command.as_deref(), Some("command -v make"));
        assert_eq!(parsed.allocated_cpus, Some(2.0));
        assert_eq!(parsed.workdir.as_deref(), Some("/workdir"));
        assert_eq!(parsed.user.as_deref(), Some("1000:1000"));
        assert_eq!(parsed.environment, vec!["MODE=test".to_owned()]);
        assert_eq!(parsed.shell, "/bin/sh");
        assert_eq!(parsed.strace_mode, ExistingStraceMode::Process);
        assert_eq!(parsed.string_limit, 8192);
        assert_eq!(parsed.docker, "/usr/bin/docker");
    }

    #[test]
    fn native_cli_parser_rejects_image_only_mutation_and_profiler_options() {
        let error = parse_arguments(&arguments(&[
            "--container",
            "live",
            "--image",
            "other",
            "--command",
            "true",
            "--output-dir",
            "/tmp/existing-output",
            "--cpuset-cpus",
            "0-1",
            "--mount",
            "/host:/target",
            "--network-policy",
            "allowlist",
            "--allowed-host",
            "example.com",
            "--profile-set",
            "cpu-average",
            "--scheduler-concurrency",
        ]))
        .unwrap_err()
        .to_string();
        for option in [
            "--image",
            "--cpuset-cpus",
            "--mount",
            "--network-policy",
            "--allowed-host",
            "--profile-set",
            "--scheduler-concurrency",
        ] {
            assert!(error.contains(option), "missing {option} in {error}");
        }
    }

    #[test]
    fn syntax_router_recognizes_attached_and_separate_container_options() {
        assert!(is_existing_container_request(&arguments(&[
            "--container",
            "live"
        ])));
        assert!(is_existing_container_request(&arguments(&[
            "--container=live"
        ])));
        assert!(!is_existing_container_request(&arguments(&[
            "--image", "fixture"
        ])));
    }

    fn run_fake(
        config: ExistingContainerConfig,
        docker: &Arc<Mutex<DockerState>>,
        host: &Arc<Mutex<HostState>>,
    ) -> Result<ExistingContainerRun, ExistingContainerError> {
        run_profile_with(
            config,
            Box::new(FakeDocker {
                state: Arc::clone(docker),
            }),
            Box::new(FakeHost {
                state: Arc::clone(host),
            }),
            None,
        )
    }

    fn run_fake_cancelled(
        config: ExistingContainerConfig,
        docker: &Arc<Mutex<DockerState>>,
        host: &Arc<Mutex<HostState>>,
    ) -> Result<ExistingContainerRun, ExistingContainerError> {
        let checks = AtomicUsize::new(0);
        let cancel = || checks.fetch_add(1, AtomicOrdering::SeqCst) > 0;
        run_profile_with(
            config,
            Box::new(FakeDocker {
                state: Arc::clone(docker),
            }),
            Box::new(FakeHost {
                state: Arc::clone(host),
            }),
            Some(&cancel),
        )
    }

    #[test]
    fn lifecycle_uses_exact_id_and_never_mutates_or_owns_container() {
        let (_temporary, config, docker, host) = fixture();
        let output = config.output_dir.clone();
        let result = run_fake(config, &docker, &host).unwrap();

        assert_eq!(result.command_exit_code, 0);
        assert_eq!(result.report["schema_version"], PROFILE_SCHEMA_VERSION);
        assert_eq!(result.report["scope"], "existing-container");
        assert_eq!(result.report["profiling_valid"], true);
        assert_eq!(result.report["cpu"]["usage_sec"], 2.0);
        assert_eq!(result.report["cpu"]["scope_includes_descendants"], true);
        assert_eq!(
            result.report["existing_container"]["lifecycle_ownership"]["container_removed"],
            false
        );
        assert_eq!(
            result.report["existing_container"]["network_inheritance"]["topology_mutated"],
            false
        );
        assert_eq!(
            fs::read_to_string(output.join("command.stdout.txt")).unwrap(),
            "workload stdout\n"
        );
        assert!(output.join("profile_report.json").is_file());

        let calls = &docker.lock().unwrap().calls;
        let measured = calls
            .iter()
            .find(|call| call.first().map(String::as_str) == Some("exec"))
            .unwrap();
        assert!(measured.iter().any(|value| value == CONTAINER_ID));
        for forbidden in [
            "run", "create", "start", "stop", "restart", "update", "rm", "network",
        ] {
            assert!(
                !calls
                    .iter()
                    .any(|call| call.first().map(String::as_str) == Some(forbidden)),
                "unexpected Docker lifecycle operation: {forbidden}"
            );
        }
        assert!(calls
            .iter()
            .filter(|call| call.first().map(String::as_str) == Some("container"))
            .skip(1)
            .all(|call| call.last().map(String::as_str) == Some(CONTAINER_ID)));
    }

    #[test]
    fn image_only_and_enforcement_options_fail_before_docker_or_output() {
        let (_temporary, mut config, docker, host) = fixture();
        config.unsupported.mounts.push("/host:/target".to_owned());
        config.unsupported.effective_network_policy = Some("allowlist".to_owned());
        let output = config.output_dir.clone();

        let error = run_fake(config, &docker, &host).unwrap_err();
        assert!(error.to_string().contains("--mount"));
        assert!(error.to_string().contains("--network-policy"));
        assert!(docker.lock().unwrap().calls.is_empty());
        assert!(!output.exists());
    }

    #[test]
    fn surviving_exec_descendant_invalidates_without_signals_or_container_cleanup() {
        let (_temporary, config, docker, host) = fixture();
        host.lock().unwrap().survivor = true;

        let result = run_fake(config, &docker, &host).unwrap();
        assert_eq!(result.report["profiling_valid"], false);
        assert_eq!(
            result.report["existing_container"]["survivor_handling"]["valid"],
            false
        );
        assert_eq!(
            result.report["existing_container"]["survivor_handling"]
                ["remaining_pids_at_finish_gate"],
            json!([5000])
        );
        assert_eq!(
            result.report["existing_container"]["survivor_handling"]["signals_sent"],
            json!([])
        );
        let calls = &docker.lock().unwrap().calls;
        assert!(!calls.iter().any(|call| matches!(
            call.first().map(String::as_str),
            Some("stop" | "rm" | "kill")
        )));
    }

    #[test]
    fn cancellation_kills_only_host_docker_client_and_retains_survivor_evidence() {
        let (_temporary, mut config, docker, host) = fixture();
        config.docker_exec_client_terminate_grace = Duration::from_millis(10);
        docker.lock().unwrap().exec_blocks_until_kill = true;
        host.lock().unwrap().survivor = true;

        let result = run_fake_cancelled(config, &docker, &host).unwrap();
        assert_eq!(result.command_exit_code, 128 + libc::SIGKILL);
        assert_eq!(result.report["profiling_valid"], false);
        let client = &result.report["existing_container"]["docker_exec_client_cleanup"];
        assert_eq!(client["cancellation_requested"], true);
        assert_eq!(client["sigterm_sent_to_host_client"], true);
        assert_eq!(client["sigkill_sent_to_host_client"], true);
        assert_eq!(client["host_client_reaped"], true);
        assert_eq!(client["container_processes_signaled"], false);
        assert_eq!(
            result.report["existing_container"]["survivor_handling"]
                ["remaining_pids_at_finish_gate"],
            json!([5000])
        );
        assert_eq!(
            docker.lock().unwrap().client_signals,
            vec![libc::SIGTERM, libc::SIGKILL]
        );
        assert!(!docker.lock().unwrap().calls.iter().any(|call| {
            matches!(
                call.first().map(String::as_str),
                Some("stop" | "kill" | "rm" | "restart" | "update")
            )
        }));
    }

    #[test]
    fn cancellation_cleanup_is_bounded_when_host_client_cannot_be_reaped() {
        let (_temporary, mut config, docker, host) = fixture();
        config.docker_exec_client_terminate_grace = Duration::from_millis(10);
        docker.lock().unwrap().exec_blocks_until_kill = true;
        docker.lock().unwrap().client_never_reaps = true;
        host.lock().unwrap().survivor = true;

        let result = run_fake_cancelled(config, &docker, &host).unwrap();
        assert_eq!(result.report["profiling_valid"], false);
        let client = &result.report["existing_container"]["docker_exec_client_cleanup"];
        assert_eq!(client["host_client_reaped"], false);
        assert!(client["cleanup_errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason.as_str().unwrap().contains("remained after bounded")));
        assert_eq!(client["container_processes_signaled"], false);
        assert_eq!(
            result.report["existing_container"]["survivor_handling"]
                ["remaining_pids_at_finish_gate"],
            json!([5000])
        );
    }

    #[test]
    fn preparation_cancellation_retains_cleanup_evidence_before_returning_error() {
        let (_temporary, mut config, docker, host) = fixture();
        config.prepare_command = Some("prepare something".to_owned());
        config.docker_exec_client_terminate_grace = Duration::from_millis(10);
        let output = config.output_dir.clone();
        docker.lock().unwrap().exec_blocks_until_kill = true;
        host.lock().unwrap().survivor = true;

        let error = run_fake_cancelled(config, &docker, &host).unwrap_err();
        assert!(error.to_string().contains("preparation was cancelled"));
        let evidence: Value = serde_json::from_slice(
            &fs::read(output.join("existing_container_cleanup.json")).unwrap(),
        )
        .unwrap();
        assert_eq!(evidence["valid"], false);
        assert_eq!(evidence["phase"], "prepare");
        assert_eq!(
            evidence["survivor_handling"]["remaining_pids"],
            json!([5000])
        );
        assert_eq!(
            evidence["docker_exec_client_cleanup"]["container_processes_signaled"],
            false
        );
    }

    #[test]
    fn allocation_change_is_retained_as_invalid_evidence() {
        let (_temporary, config, docker, host) = fixture();
        host.lock().unwrap().allocation_changed = true;

        let result = run_fake(config, &docker, &host).unwrap();
        assert_eq!(result.report["profiling_valid"], false);
        assert_eq!(
            result.report["existing_container"]["stability"]["allocation_stable"],
            false
        );
    }

    #[test]
    fn inherited_network_change_is_retained_without_topology_mutation() {
        let (_temporary, config, docker, host) = fixture();
        docker.lock().unwrap().network_changed = true;

        let result = run_fake(config, &docker, &host).unwrap();
        assert_eq!(result.report["profiling_valid"], false);
        assert_eq!(
            result.report["existing_container"]["stability"]["network_inheritance_stable"],
            false
        );
        assert_eq!(
            result.report["existing_container"]["lifecycle_ownership"]["network_topology_mutated"],
            false
        );
        assert!(!docker
            .lock()
            .unwrap()
            .calls
            .iter()
            .any(|call| { call.first().map(String::as_str) == Some("network") }));
    }

    #[test]
    fn command_failure_preserves_report_and_public_exit_code() {
        let (_temporary, config, docker, host) = fixture();
        docker.lock().unwrap().command_status = 7;

        let result = run_fake(config, &docker, &host).unwrap();
        assert_eq!(result.command_exit_code, 7);
        assert_eq!(result.report["command_exit_code"], 7);
        assert_eq!(result.report["profiling_valid"], true);
    }

    #[test]
    fn strace_copy_and_remote_file_cleanup_do_not_remove_container() {
        let (_temporary, mut config, docker, host) = fixture();
        config.strace_mode = ExistingStraceMode::Execve;
        let output = config.output_dir.clone();

        let result = run_fake(config, &docker, &host).unwrap();
        assert_eq!(result.report["profiling_valid"], true);
        assert!(output.join("process.strace.log").is_file());
        let calls = &docker.lock().unwrap().calls;
        assert!(calls
            .iter()
            .any(|call| call.first().map(String::as_str) == Some("cp")));
        assert!(calls
            .iter()
            .any(|call| call.first().map(String::as_str) == Some("exec")
                && call.iter().any(|value| value == "/bin/rm")));
        assert!(!calls
            .iter()
            .any(|call| call.first().map(String::as_str) == Some("rm")));
    }

    #[test]
    fn failed_trace_copy_still_attempts_remote_file_cleanup_and_invalidates() {
        let (_temporary, mut config, docker, host) = fixture();
        config.strace_mode = ExistingStraceMode::Execve;
        docker.lock().unwrap().copy_io_error = true;

        let result = run_fake(config, &docker, &host).unwrap();
        assert_eq!(result.report["profiling_valid"], false);
        assert!(result.report["strace"]["retrieval_error"]
            .as_str()
            .unwrap()
            .contains("copy could not start"));
        let calls = &docker.lock().unwrap().calls;
        assert!(calls.iter().any(|call| {
            call.first().map(String::as_str) == Some("exec")
                && call.iter().any(|value| value == "/bin/rm")
        }));
    }

    #[test]
    fn nonempty_output_fails_before_runtime_inspection() {
        let (_temporary, config, docker, host) = fixture();
        fs::create_dir(&config.output_dir).unwrap();
        fs::write(config.output_dir.join("owner.txt"), b"caller data").unwrap();

        let error = run_fake(config, &docker, &host).unwrap_err();
        assert!(error.to_string().contains("new or empty"));
        assert!(docker.lock().unwrap().calls.is_empty());
    }

    #[test]
    fn environment_report_hashes_values_and_exec_receives_them() {
        let (_temporary, mut config, docker, host) = fixture();
        config.environment = vec!["TOKEN=top-secret".to_owned()];

        let result = run_fake(config, &docker, &host).unwrap();
        let encoded = serde_json::to_string(&result.report).unwrap();
        assert!(!encoded.contains("top-secret"));
        assert!(encoded.contains("value_sha256"));
        let calls = &docker.lock().unwrap().calls;
        assert!(calls
            .iter()
            .any(|call| call.iter().any(|value| value == "TOKEN=top-secret")));
    }
}
