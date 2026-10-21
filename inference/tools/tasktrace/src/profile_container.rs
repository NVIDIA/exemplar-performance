// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native disposable-Docker command profiling.
//!
//! The native controller owns baseline, CPU, memory, scheduler, process, and
//! hardware-counter profile sets, optionally with sampled host `/proc/stat`
//! comparison context. They share one host-coordinated container gate,
//! cgroup-v2 accounting, placement validation, monotonic clock, descendant
//! drain, evidence, and checked cleanup path.

mod cgroup_memory;

use crate::network_enforcement::{
    load_network_enforcement_report, validate_workload_mount_sources, DockerNetworkBinding,
    DockerNetworkPolicyConfig, DockerNetworkPolicySession, DEFAULT_EXTERNAL_NETWORK,
    DEFAULT_PROXY_IMAGE,
};
use crate::network_policy::{resolve_cli_network_policy, NetworkMode, NetworkPolicy};
use crate::proc_stat::{add_reference_comparison, parse_proc_stat, ProcStatComparisonSampler};
use crate::process_exec;
use crate::timebase::{clock_reference, monotonic_ns, monotonic_resolution_ns};
use serde_json::{json, Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::ffi::{CString, OsStr, OsString};
use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::{DirBuilderExt, MetadataExt, OpenOptionsExt, PermissionsExt};
use std::os::unix::process::ExitStatusExt;
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

const ENTRYPOINT: &str =
    include_str!("../offline_replayer/measurement/container_profile_entrypoint.sh");
const PROFILE_SCHEMA_VERSION: &str = "container-command-profile.v1";
const CPU_SCHEMA_VERSION: &str = "cgroup-cpu-coordinated.v1";
const EVIDENCE_SCHEMA_VERSION: &str = "tasktrace-profile-evidence-bundle.v1";
const CPU_SOURCE_SCHEMA_VERSION: &str = "container-cgroup-cpu-source-snapshots.v1";
const PROFILER_VERSION: &str = "container-coordinated-profiler.v1";
const CGROUP_ROOT: &str = "/sys/fs/cgroup";
const ATTACH_POLL: Duration = Duration::from_millis(10);
const DESCENDANT_GRACE: Duration = Duration::from_secs(1);
const CLEANUP_GRACE: Duration = Duration::from_secs(5);
const MAX_TASK_LOG_BYTES: u64 = 1 << 30;
const MAX_TASK_CONTROL_BYTES: u64 = 4096;
const DEFAULT_HARDWARE_EVENTS: &[&str] = &[
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProfileError(String);

impl ProfileError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ProfileError {}

impl From<io::Error> for ProfileError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ProfileError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<process_exec::ProcessExecError> for ProfileError {
    fn from(error: process_exec::ProcessExecError) -> Self {
        Self(error.to_string())
    }
}

impl From<crate::scheduler_perf::SchedulerPerfError> for ProfileError {
    fn from(error: crate::scheduler_perf::SchedulerPerfError) -> Self {
        Self(error.to_string())
    }
}

impl From<crate::scheduler_bcc::BccSchedulerError> for ProfileError {
    fn from(error: crate::scheduler_bcc::BccSchedulerError) -> Self {
        Self(error.to_string())
    }
}

impl From<crate::hardware_perf::HardwarePerfError> for ProfileError {
    fn from(error: crate::hardware_perf::HardwarePerfError) -> Self {
        Self(error.to_string())
    }
}

impl From<crate::enroot_runtime::EnrootError> for ProfileError {
    fn from(error: crate::enroot_runtime::EnrootError) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum StraceMode {
    Execve,
    Process,
    None,
}

impl StraceMode {
    fn parse(value: &str) -> Result<Self, ProfileError> {
        match value {
            "execve" => Ok(Self::Execve),
            "process" => Ok(Self::Process),
            "none" => Ok(Self::None),
            _ => Err(ProfileError::new(format!(
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
}

#[derive(Clone, Debug)]
pub struct ProfileConfig {
    pub runtime: String,
    pub run_wave_start_gate: bool,
    pub image: String,
    pub command: String,
    pub output_dir: PathBuf,
    pub prepare_command: Option<String>,
    pub verification_command: Option<String>,
    pub cpuset_cpus: Option<String>,
    pub allocated_cpus: Option<f64>,
    pub cpu_limit: Option<f64>,
    pub memory_limit_bytes: Option<u64>,
    pub memory_swap_total_limit_bytes: Option<u64>,
    pub workdir: Option<String>,
    pub user: Option<String>,
    pub environment: Vec<String>,
    pub mounts: Vec<String>,
    pub shell: String,
    strace_mode: StraceMode,
    pub string_limit: u64,
    pub profile_sets: Vec<String>,
    pub coordinated_baseline: bool,
    pub terminate_descendants_at_finish: bool,
    pub hardware_perf_binary: String,
    pub hardware_perf_events: Vec<String>,
    pub hardware_perf_attach_timeout: Duration,
    pub hardware_perf_minimum_running_ratio: f64,
    pub scheduler_concurrency: bool,
    pub scheduler_backend: String,
    pub scheduler_bcc_buffer_pages: Option<u64>,
    pub scheduler_perf_binary: String,
    pub scheduler_perf_mmap_pages: Option<u64>,
    pub scheduler_cpu_tolerance_fraction: f64,
    pub scheduler_cpu_tolerance_sec: f64,
    pub proc_stat_comparison: bool,
    pub proc_stat_interval_sec: f64,
    pub proc_stat_baseline_sec: f64,
    pub memory_profile: bool,
    pub memory_sample_interval_sec: f64,
    pub network: Option<String>,
    pub network_policy: NetworkPolicy,
    pub network_policy_proxy_image: String,
    pub network_policy_external_network: String,
    pub docker: String,
    pub enroot: String,
    pub attach_timeout: Duration,
    pub workload_identity: Option<Value>,
    pub source_identity: Option<Value>,
    pub expected_outcome: Option<Value>,
}

impl ProfileConfig {
    pub fn new(
        image: impl Into<String>,
        command: impl Into<String>,
        output_dir: impl Into<PathBuf>,
    ) -> Self {
        Self {
            runtime: "docker".to_owned(),
            run_wave_start_gate: false,
            image: image.into(),
            command: command.into(),
            output_dir: output_dir.into(),
            prepare_command: None,
            verification_command: None,
            cpuset_cpus: None,
            allocated_cpus: None,
            cpu_limit: None,
            memory_limit_bytes: None,
            memory_swap_total_limit_bytes: None,
            workdir: None,
            user: None,
            environment: Vec::new(),
            mounts: Vec::new(),
            shell: "/bin/bash".to_owned(),
            strace_mode: StraceMode::None,
            string_limit: 4096,
            profile_sets: vec!["cpu-average".to_owned()],
            coordinated_baseline: false,
            terminate_descendants_at_finish: false,
            hardware_perf_binary: "perf".to_owned(),
            hardware_perf_events: DEFAULT_HARDWARE_EVENTS
                .iter()
                .map(|event| (*event).to_owned())
                .collect(),
            hardware_perf_attach_timeout: Duration::from_secs(20),
            hardware_perf_minimum_running_ratio: crate::hardware_perf::MINIMUM_RUNNING_RATIO,
            scheduler_concurrency: false,
            scheduler_backend: "bcc".to_owned(),
            scheduler_bcc_buffer_pages: None,
            scheduler_perf_binary: "perf".to_owned(),
            scheduler_perf_mmap_pages: None,
            scheduler_cpu_tolerance_fraction: 0.02,
            scheduler_cpu_tolerance_sec: 0.005,
            proc_stat_comparison: false,
            proc_stat_interval_sec: 0.5,
            proc_stat_baseline_sec: 5.0,
            memory_profile: false,
            memory_sample_interval_sec: 0.1,
            network: None,
            network_policy: NetworkPolicy::default(),
            network_policy_proxy_image: DEFAULT_PROXY_IMAGE.to_owned(),
            network_policy_external_network: DEFAULT_EXTERNAL_NETWORK.to_owned(),
            docker: "docker".to_owned(),
            enroot: "enroot".to_owned(),
            attach_timeout: Duration::from_secs(20),
            workload_identity: None,
            source_identity: None,
            expected_outcome: None,
        }
    }

    pub fn set_untraced_baseline(&mut self) {
        self.profile_sets.clear();
        self.coordinated_baseline = false;
        self.strace_mode = StraceMode::None;
    }

    pub fn set_coordinated_baseline(&mut self) {
        self.profile_sets.clear();
        self.coordinated_baseline = true;
        self.strace_mode = StraceMode::None;
    }

    pub fn set_strace_mode(&mut self, value: &str) -> Result<(), ProfileError> {
        self.strace_mode = StraceMode::parse(value)?;
        Ok(())
    }

    fn effective_strace_mode(&self) -> StraceMode {
        if self
            .profile_sets
            .iter()
            .any(|value| value == "software-deep")
        {
            StraceMode::Process
        } else if self.coordinated_baseline || !self.profile_sets.is_empty() {
            StraceMode::None
        } else {
            self.strace_mode
        }
    }

    fn is_coordinated(&self) -> bool {
        self.coordinated_baseline || !self.profile_sets.is_empty()
    }

    fn memory_enabled(&self) -> bool {
        self.memory_profile
            || self
                .profile_sets
                .iter()
                .any(|value| matches!(value.as_str(), "memory-detail" | "cpu-compare"))
    }

    fn scheduler_enabled(&self) -> bool {
        self.scheduler_concurrency
            || self
                .profile_sets
                .iter()
                .any(|value| matches!(value.as_str(), "cpu-peak" | "cpu-compare"))
    }

    fn hardware_enabled(&self) -> bool {
        self.profile_sets
            .iter()
            .any(|value| value == "hardware-deep")
    }

    fn cpu_evidence_selected(&self) -> bool {
        self.coordinated_baseline
            || self.proc_stat_comparison
            || self
                .profile_sets
                .iter()
                .any(|value| matches!(value.as_str(), "cpu-average" | "cpu-peak" | "cpu-compare"))
    }

    fn allocated_capacity(&self) -> Option<f64> {
        self.allocated_cpus.or_else(|| {
            self.cpuset_cpus
                .as_deref()
                .and_then(|value| parse_cpuset(value).ok())
                .map(|cpus| cpus.len() as f64)
        })
    }
}

fn enroot_runtime_config(
    config: &ProfileConfig,
    entrypoint: impl Into<PathBuf>,
    network_evidence_path: Option<PathBuf>,
) -> crate::enroot_runtime::EnrootRuntimeConfig {
    let mut runtime = crate::enroot_runtime::EnrootRuntimeConfig::new(
        config.image.clone(),
        config.command.clone(),
        config.output_dir.clone(),
        entrypoint,
    );
    runtime.enroot = config.enroot.clone();
    let (containment_executable, containment_flavor) =
        crate::enroot_runtime::detect_containment();
    runtime.unshare = containment_executable;
    runtime.containment_flavor = containment_flavor;
    runtime.prepare_command = config.prepare_command.clone();
    runtime.cpuset_cpus = config.cpuset_cpus.clone();
    runtime.allocated_cpus = config.allocated_cpus;
    runtime.cpu_limit = config.cpu_limit;
    runtime.workdir = config.workdir.clone();
    runtime.user = config.user.clone();
    runtime.environment = config.environment.clone();
    runtime.mounts = config.mounts.clone();
    runtime.shell = config.shell.clone();
    runtime.string_limit = config.string_limit;
    runtime.strace_mode = config.effective_strace_mode().as_str().to_owned();
    // `--network none` is the deprecated spelling of the typed no-network
    // policy, not an Enroot network attachment.  Keep it in the public legacy
    // projection while omitting it from the backend runtime configuration.
    runtime.network = config
        .network
        .as_ref()
        .filter(|network| network.as_str() != "none")
        .cloned();
    runtime.network_mode = config.network_policy.effective_mode();
    runtime.network_evidence_path = network_evidence_path;
    runtime.scheduler_concurrency = config.scheduler_concurrency;
    runtime.proc_stat_comparison = config.proc_stat_comparison;
    runtime.memory_profile = config.memory_profile;
    runtime.memory_limit_bytes = config.memory_limit_bytes;
    runtime.memory_swap_total_limit_bytes = config.memory_swap_total_limit_bytes;
    runtime.profile_sets = config.profile_sets.clone();
    runtime.coordinated_baseline = config.coordinated_baseline;
    runtime.verification_command = config.verification_command.clone();
    runtime.terminate_descendants_at_finish = config.terminate_descendants_at_finish;
    runtime.run_wave_start_gate = config.run_wave_start_gate;
    runtime
}

fn validate_profile_identities(config: &ProfileConfig) -> Result<(), ProfileError> {
    for (name, identity) in [
        ("workload identity", &config.workload_identity),
        ("source identity", &config.source_identity),
        ("expected outcome", &config.expected_outcome),
    ] {
        if identity
            .as_ref()
            .is_some_and(|value| !value.as_object().is_some_and(|object| !object.is_empty()))
        {
            return Err(ProfileError::new(format!(
                "{name} must be a nonempty JSON object when supplied"
            )));
        }
    }
    Ok(())
}

#[derive(Default)]
struct ParsedArguments {
    image: Option<String>,
    container: Option<String>,
    runtime: String,
    command: Option<String>,
    prepare_command: Option<String>,
    verification_command: Option<String>,
    output_dir: Option<PathBuf>,
    cpuset_cpus: Option<String>,
    allocated_cpus: Option<f64>,
    cpu_limit: Option<f64>,
    workdir: Option<String>,
    user: Option<String>,
    network: Option<String>,
    network_mode: Option<String>,
    allowed_hosts: Vec<String>,
    network_proxy_image: String,
    network_external: String,
    shell: String,
    environment: Vec<String>,
    mounts: Vec<String>,
    strace_mode: String,
    coordinated_baseline: bool,
    profile_sets: Vec<String>,
    hardware_perf_binary: String,
    hardware_perf_events: Vec<String>,
    hardware_attach_timeout_sec: f64,
    hardware_running_ratio: f64,
    string_limit: i64,
    scheduler_concurrency: bool,
    terminate_descendants_at_finish: bool,
    scheduler_backend: String,
    scheduler_bcc_buffer_pages: Option<i64>,
    scheduler_perf_binary: String,
    scheduler_perf_mmap_pages: Option<i64>,
    proc_stat_comparison: bool,
    proc_stat_interval_sec: f64,
    proc_stat_baseline_sec: f64,
    memory_profile: bool,
    memory_sample_interval_sec: f64,
    memory_limit_bytes: Option<i64>,
    memory_swap_total_limit_bytes: Option<i64>,
    scheduler_attach_timeout_sec: f64,
    scheduler_cpu_tolerance_fraction: f64,
    scheduler_cpu_tolerance_sec: f64,
    docker: String,
    enroot: String,
}

impl ParsedArguments {
    fn defaults() -> Self {
        Self {
            runtime: "docker".to_owned(),
            network_proxy_image: "python:3.12-slim".to_owned(),
            network_external: "bridge".to_owned(),
            shell: "/bin/bash".to_owned(),
            strace_mode: "execve".to_owned(),
            hardware_perf_binary: "perf".to_owned(),
            hardware_attach_timeout_sec: 20.0,
            hardware_running_ratio: 0.95,
            string_limit: 4096,
            scheduler_backend: "bcc".to_owned(),
            scheduler_perf_binary: "perf".to_owned(),
            proc_stat_interval_sec: 0.5,
            proc_stat_baseline_sec: 5.0,
            memory_sample_interval_sec: 0.1,
            scheduler_attach_timeout_sec: 20.0,
            scheduler_cpu_tolerance_fraction: 0.02,
            scheduler_cpu_tolerance_sec: 0.005,
            docker: "docker".to_owned(),
            enroot: "enroot".to_owned(),
            ..Self::default()
        }
    }
}

fn text_argument(value: &OsStr, option: &str) -> Result<String, ProfileError> {
    value
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| ProfileError::new(format!("{option} must be valid UTF-8")))
}

fn parse_number<T>(value: &str, option: &str) -> Result<T, ProfileError>
where
    T: std::str::FromStr,
{
    value
        .parse::<T>()
        .map_err(|_| ProfileError::new(format!("invalid value for {option}: {value:?}")))
}

fn split_long_option(value: &str) -> (&str, Option<&str>) {
    value
        .split_once('=')
        .map_or((value, None), |(option, argument)| (option, Some(argument)))
}

fn parse_arguments(arguments: &[OsString]) -> Result<Option<ProfileConfig>, ProfileError> {
    if arguments.len() == 1 && matches!(arguments[0].to_str(), Some("-h" | "--help")) {
        return Ok(None);
    }
    let mut parsed = ParsedArguments::defaults();
    let mut index = 0;
    while index < arguments.len() {
        let raw = text_argument(&arguments[index], "argument")?;
        if raw == "-h" || raw == "--help" {
            return Ok(None);
        }
        let (option, attached) = split_long_option(&raw);
        let flag = |attached: Option<&str>| -> Result<(), ProfileError> {
            if attached.is_some() {
                Err(ProfileError::new(format!("{option} does not take a value")))
            } else {
                Ok(())
            }
        };
        let mut take = || -> Result<String, ProfileError> {
            if let Some(value) = attached {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .ok_or_else(|| {
                    ProfileError::new(format!("argument {option}: expected one argument"))
                })
                .and_then(|value| text_argument(value, option))
        };
        match option {
            "--image" => parsed.image = Some(take()?),
            "--container" => parsed.container = Some(take()?),
            "--runtime" => parsed.runtime = take()?,
            "--command" => parsed.command = Some(take()?),
            "--prepare-command" => parsed.prepare_command = Some(take()?),
            "--verification-command" => parsed.verification_command = Some(take()?),
            "--output-dir" => parsed.output_dir = Some(PathBuf::from(take()?)),
            "--cpuset-cpus" => parsed.cpuset_cpus = Some(take()?),
            "--allocated-cpus" => {
                let value = take()?;
                parsed.allocated_cpus = Some(parse_number(&value, option)?);
            }
            "--cpu-limit" => {
                let value = take()?;
                parsed.cpu_limit = Some(parse_number(&value, option)?);
            }
            "--workdir" => parsed.workdir = Some(take()?),
            "--user" => parsed.user = Some(take()?),
            "--network" => parsed.network = Some(take()?),
            "--network-policy" => parsed.network_mode = Some(take()?),
            "--allowed-host" => parsed.allowed_hosts.push(take()?),
            "--network-policy-proxy-image" => parsed.network_proxy_image = take()?,
            "--network-policy-external-network" => parsed.network_external = take()?,
            "--shell" => parsed.shell = take()?,
            "--env" => parsed.environment.push(take()?),
            "--mount" => parsed.mounts.push(take()?),
            "--strace-mode" => parsed.strace_mode = take()?,
            "--coordinated-baseline" => {
                flag(attached)?;
                parsed.coordinated_baseline = true;
            }
            "--profile-set" => parsed.profile_sets.push(take()?),
            "--hardware-perf-binary" => parsed.hardware_perf_binary = take()?,
            "--hardware-perf-event" => parsed.hardware_perf_events.push(take()?),
            "--hardware-perf-attach-timeout-sec" => {
                let value = take()?;
                parsed.hardware_attach_timeout_sec = parse_number(&value, option)?;
            }
            "--hardware-perf-minimum-running-ratio" => {
                let value = take()?;
                parsed.hardware_running_ratio = parse_number(&value, option)?;
            }
            "--strace-string-limit" => {
                let value = take()?;
                parsed.string_limit = parse_number(&value, option)?;
            }
            "--scheduler-concurrency" => {
                flag(attached)?;
                parsed.scheduler_concurrency = true;
            }
            "--terminate-descendants-at-finish" => {
                flag(attached)?;
                parsed.terminate_descendants_at_finish = true;
            }
            "--scheduler-backend" => parsed.scheduler_backend = take()?,
            "--scheduler-bcc-buffer-pages" => {
                let value = take()?;
                parsed.scheduler_bcc_buffer_pages = Some(parse_number(&value, option)?);
            }
            "--scheduler-perf-binary" => parsed.scheduler_perf_binary = take()?,
            "--scheduler-perf-mmap-pages" => {
                let value = take()?;
                parsed.scheduler_perf_mmap_pages = Some(parse_number(&value, option)?);
            }
            "--proc-stat-comparison" => {
                flag(attached)?;
                parsed.proc_stat_comparison = true;
            }
            "--proc-stat-interval-sec" => {
                let value = take()?;
                parsed.proc_stat_interval_sec = parse_number(&value, option)?;
            }
            "--proc-stat-baseline-sec" => {
                let value = take()?;
                parsed.proc_stat_baseline_sec = parse_number(&value, option)?;
            }
            "--memory-profile" => {
                flag(attached)?;
                parsed.memory_profile = true;
            }
            "--memory-sample-interval-sec" => {
                let value = take()?;
                parsed.memory_sample_interval_sec = parse_number(&value, option)?;
            }
            "--memory-limit-bytes" => {
                let value = take()?;
                parsed.memory_limit_bytes = Some(parse_number(&value, option)?);
            }
            "--memory-swap-total-limit-bytes" => {
                let value = take()?;
                parsed.memory_swap_total_limit_bytes = Some(parse_number(&value, option)?);
            }
            "--scheduler-attach-timeout-sec" => {
                let value = take()?;
                parsed.scheduler_attach_timeout_sec = parse_number(&value, option)?;
            }
            "--scheduler-cpu-tolerance-fraction" => {
                let value = take()?;
                parsed.scheduler_cpu_tolerance_fraction = parse_number(&value, option)?;
            }
            "--scheduler-cpu-tolerance-sec" => {
                let value = take()?;
                parsed.scheduler_cpu_tolerance_sec = parse_number(&value, option)?;
            }
            "--docker" => parsed.docker = take()?,
            "--enroot" => parsed.enroot = take()?,
            "--" => {
                return Err(ProfileError::new("unrecognized arguments after --"));
            }
            _ => return Err(ProfileError::new(format!("unrecognized arguments: {raw}"))),
        }
        index += 1;
    }
    validate_parsed(parsed).map(Some)
}

fn positive_finite(value: f64, option: &str) -> Result<f64, ProfileError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(ProfileError::new(format!("{option} must be positive")))
    }
}

fn validate_parsed(parsed: ParsedArguments) -> Result<ProfileConfig, ProfileError> {
    let image = parsed
        .image
        .ok_or_else(|| ProfileError::new("one of the arguments --image --container is required"))?;
    if parsed.container.is_some() {
        return Err(ProfileError::new(
            "existing-container requests must use the native existing-container parser",
        ));
    }
    if !matches!(parsed.runtime.as_str(), "docker" | "enroot") {
        return Err(ProfileError::new(
            "invalid choice for --runtime (choose from 'docker', 'enroot')",
        ));
    }
    let command = parsed
        .command
        .filter(|value| !value.is_empty())
        .ok_or_else(|| ProfileError::new("the following arguments are required: --command"))?;
    let output_dir = parsed
        .output_dir
        .ok_or_else(|| ProfileError::new("the following arguments are required: --output-dir"))?;
    let allocated_cpus = parsed
        .allocated_cpus
        .map(|value| positive_finite(value, "--allocated-cpus"))
        .transpose()?;
    let cpu_limit = parsed
        .cpu_limit
        .map(|value| positive_finite(value, "--cpu-limit"))
        .transpose()?;
    if parsed.cpuset_cpus.is_none() && allocated_cpus.is_none() {
        return Err(ProfileError::new(
            "--image requires --cpuset-cpus or --allocated-cpus for normalized utilization",
        ));
    }
    if let Some(cpuset) = &parsed.cpuset_cpus {
        parse_cpuset(cpuset)?;
    }
    if parsed.coordinated_baseline && !parsed.profile_sets.is_empty() {
        return Err(ProfileError::new(
            "--coordinated-baseline is mutually exclusive with --profile-set",
        ));
    }
    let unique_sets: BTreeSet<_> = parsed.profile_sets.iter().collect();
    if unique_sets.len() != parsed.profile_sets.len() {
        return Err(ProfileError::new(
            "profile sets must not contain duplicates",
        ));
    }
    if parsed.profile_sets.iter().any(|value| {
        !matches!(
            value.as_str(),
            "cpu-average"
                | "cpu-peak"
                | "cpu-compare"
                | "memory-detail"
                | "software-deep"
                | "hardware-deep"
        )
    }) {
        return Err(ProfileError::new(
            "native profile container supports cpu-average, cpu-peak, cpu-compare, memory-detail, software-deep, and hardware-deep when privileged backends are available",
        ));
    }
    let scheduler_requested = parsed.scheduler_concurrency
        || parsed
            .profile_sets
            .iter()
            .any(|value| matches!(value.as_str(), "cpu-peak" | "cpu-compare"));
    if parsed.coordinated_baseline && parsed.memory_profile {
        return Err(ProfileError::new(
            "--coordinated-baseline cannot enable memory profiling",
        ));
    }
    if parsed.string_limit <= 0 {
        return Err(ProfileError::new("--strace-string-limit must be positive"));
    }
    positive_finite(
        parsed.scheduler_attach_timeout_sec,
        "--scheduler-attach-timeout-sec",
    )?;
    if !parsed.scheduler_cpu_tolerance_fraction.is_finite()
        || parsed.scheduler_cpu_tolerance_fraction < 0.0
        || !parsed.scheduler_cpu_tolerance_sec.is_finite()
        || parsed.scheduler_cpu_tolerance_sec < 0.0
    {
        return Err(ProfileError::new(
            "scheduler CPU tolerances must not be negative",
        ));
    }
    if parsed.scheduler_backend != "bcc" && parsed.scheduler_backend != "perf" {
        return Err(ProfileError::new(
            "invalid choice for --scheduler-backend (choose from 'bcc', 'perf')",
        ));
    }
    if !scheduler_requested && parsed.scheduler_backend != "bcc" {
        return Err(ProfileError::new(
            "--scheduler-backend requires --scheduler-concurrency",
        ));
    }
    if let Some(value) = parsed.scheduler_bcc_buffer_pages {
        if value <= 0 || (value & (value - 1)) != 0 {
            return Err(ProfileError::new(
                "--scheduler-bcc-buffer-pages must be a positive power of two",
            ));
        }
    }
    if !scheduler_requested && parsed.scheduler_bcc_buffer_pages.is_some() {
        return Err(ProfileError::new(
            "--scheduler-bcc-buffer-pages requires --scheduler-concurrency or a scheduler profile set",
        ));
    }
    if parsed
        .scheduler_perf_mmap_pages
        .is_some_and(|value| value <= 0)
    {
        return Err(ProfileError::new(
            "--scheduler-perf-mmap-pages must be positive",
        ));
    }
    if scheduler_requested
        && parsed.scheduler_backend == "perf"
        && parsed.scheduler_perf_binary.trim().is_empty()
    {
        return Err(ProfileError::new(
            "--scheduler-perf-binary must be nonempty",
        ));
    }
    positive_finite(parsed.proc_stat_interval_sec, "proc-stat interval")?;
    positive_finite(parsed.proc_stat_baseline_sec, "proc-stat baseline")?;
    if parsed.proc_stat_comparison && parsed.cpuset_cpus.is_none() {
        return Err(ProfileError::new(
            "--proc-stat-comparison requires --cpuset-cpus",
        ));
    }
    positive_finite(
        parsed.memory_sample_interval_sec,
        "--memory-sample-interval-sec",
    )?;
    positive_finite(
        parsed.hardware_attach_timeout_sec,
        "--hardware-perf-attach-timeout-sec",
    )?;
    if !parsed.hardware_running_ratio.is_finite()
        || parsed.hardware_running_ratio <= 0.0
        || parsed.hardware_running_ratio > 1.0
    {
        return Err(ProfileError::new(
            "--hardware-perf-minimum-running-ratio must be in (0, 1]",
        ));
    }
    if parsed.hardware_perf_binary.trim().is_empty() {
        return Err(ProfileError::new("--hardware-perf-binary must be nonempty"));
    }
    let hardware_perf_events = if parsed.hardware_perf_events.is_empty() {
        DEFAULT_HARDWARE_EVENTS
            .iter()
            .map(|value| (*value).to_owned())
            .collect::<Vec<_>>()
    } else {
        parsed.hardware_perf_events
    };
    if hardware_perf_events
        .iter()
        .any(|event| event.trim().is_empty())
        || hardware_perf_events.iter().collect::<BTreeSet<_>>().len() != hardware_perf_events.len()
    {
        return Err(ProfileError::new(
            "hardware perf events must be nonempty and unique",
        ));
    }
    let memory_limit = parsed
        .memory_limit_bytes
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| ProfileError::new("--memory-limit-bytes must be positive"))
        })
        .transpose()?;
    let memory_swap = parsed
        .memory_swap_total_limit_bytes
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    ProfileError::new("--memory-swap-total-limit-bytes must be positive")
                })
        })
        .transpose()?;
    if memory_swap.is_some() && memory_limit.is_none() {
        return Err(ProfileError::new(
            "--memory-swap-total-limit-bytes requires --memory-limit-bytes",
        ));
    }
    if memory_swap
        .zip(memory_limit)
        .is_some_and(|(swap, memory)| swap < memory)
    {
        return Err(ProfileError::new(
            "--memory-swap-total-limit-bytes must be at least --memory-limit-bytes",
        ));
    }
    if parsed
        .verification_command
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(ProfileError::new(
            "verification command must be a nonempty string",
        ));
    }
    if parsed.verification_command.is_some()
        && !parsed.coordinated_baseline
        && parsed.profile_sets.is_empty()
    {
        return Err(ProfileError::new(
            "--verification-command requires --profile-set or --coordinated-baseline",
        ));
    }
    if parsed.terminate_descendants_at_finish
        && parsed.profile_sets.is_empty()
        && !parsed.scheduler_concurrency
        && !parsed.proc_stat_comparison
        && !parsed.memory_profile
    {
        return Err(ProfileError::new(
            "--terminate-descendants-at-finish requires an active host-side profiler",
        ));
    }
    let resolved = resolve_cli_network_policy(
        parsed.network_mode.as_deref(),
        &parsed.allowed_hosts,
        parsed.network.as_deref(),
    )
    .map_err(|error| ProfileError::new(error.to_string()))?;
    let network_policy = resolved.unwrap_or_default();
    validate_mounts(&parsed.mounts)?;
    Ok(ProfileConfig {
        runtime: parsed.runtime,
        run_wave_start_gate: false,
        image,
        command,
        output_dir,
        prepare_command: parsed.prepare_command,
        verification_command: parsed.verification_command,
        cpuset_cpus: parsed.cpuset_cpus,
        allocated_cpus,
        cpu_limit,
        memory_limit_bytes: memory_limit,
        memory_swap_total_limit_bytes: memory_swap,
        workdir: parsed.workdir,
        user: parsed.user,
        environment: parsed.environment,
        mounts: parsed.mounts,
        shell: parsed.shell,
        strace_mode: StraceMode::parse(&parsed.strace_mode)?,
        string_limit: parsed.string_limit as u64,
        profile_sets: parsed.profile_sets,
        coordinated_baseline: parsed.coordinated_baseline,
        terminate_descendants_at_finish: parsed.terminate_descendants_at_finish,
        hardware_perf_binary: parsed.hardware_perf_binary,
        hardware_perf_events,
        hardware_perf_attach_timeout: Duration::from_secs_f64(parsed.hardware_attach_timeout_sec),
        hardware_perf_minimum_running_ratio: parsed.hardware_running_ratio,
        scheduler_concurrency: parsed.scheduler_concurrency,
        scheduler_backend: parsed.scheduler_backend,
        scheduler_bcc_buffer_pages: parsed.scheduler_bcc_buffer_pages.map(|value| value as u64),
        scheduler_perf_binary: parsed.scheduler_perf_binary,
        scheduler_perf_mmap_pages: parsed.scheduler_perf_mmap_pages.map(|value| value as u64),
        scheduler_cpu_tolerance_fraction: parsed.scheduler_cpu_tolerance_fraction,
        scheduler_cpu_tolerance_sec: parsed.scheduler_cpu_tolerance_sec,
        proc_stat_comparison: parsed.proc_stat_comparison,
        proc_stat_interval_sec: parsed.proc_stat_interval_sec,
        proc_stat_baseline_sec: parsed.proc_stat_baseline_sec,
        memory_profile: parsed.memory_profile,
        memory_sample_interval_sec: parsed.memory_sample_interval_sec,
        network: parsed.network,
        network_policy,
        network_policy_proxy_image: parsed.network_proxy_image,
        network_policy_external_network: parsed.network_external,
        docker: parsed.docker,
        enroot: parsed.enroot,
        attach_timeout: Duration::from_secs_f64(parsed.scheduler_attach_timeout_sec),
        workload_identity: None,
        source_identity: None,
        expected_outcome: None,
    })
}

fn parse_mount_destination(value: &str) -> Option<&str> {
    if value.contains("src=") || value.contains("source=") || value.starts_with("type=") {
        for part in value.split(',') {
            if let Some((key, argument)) = part.trim().split_once('=') {
                if matches!(key, "dst" | "target" | "destination") {
                    return Some(argument);
                }
            }
        }
        None
    } else {
        value.split(':').nth(1)
    }
}

fn parse_mount_source(value: &str) -> Option<&str> {
    if value.contains("src=") || value.contains("source=") || value.starts_with("type=") {
        for part in value.split(',') {
            if let Some((key, argument)) = part.trim().split_once('=') {
                if matches!(key, "src" | "source") {
                    return Some(argument);
                }
            }
        }
        None
    } else {
        value.split(':').next()
    }
}

fn validate_mounts(mounts: &[String]) -> Result<(), ProfileError> {
    for mount in mounts {
        let destination = parse_mount_destination(mount).ok_or_else(|| {
            ProfileError::new(format!("invalid bind mount specification: {mount:?}"))
        })?;
        if matches!(
            destination,
            "/tasktrace-profile-output"
                | "/tasktrace-profile-exchange"
                | "/tasktrace-profile-entrypoint.sh"
        ) {
            return Err(ProfileError::new(format!(
                "mount destination is reserved by the profiler: {destination}"
            )));
        }
    }
    Ok(())
}

fn workload_mount_sources(mounts: &[String]) -> Result<Vec<PathBuf>, ProfileError> {
    mounts
        .iter()
        .map(|mount| {
            parse_mount_source(mount).map(PathBuf::from).ok_or_else(|| {
                ProfileError::new(format!("invalid bind mount specification: {mount:?}"))
            })
        })
        .collect()
}

fn validate_environment(assignments: &[String]) -> Result<(), ProfileError> {
    for assignment in assignments {
        let (name, _value) = assignment.split_once('=').unwrap_or((assignment, ""));
        if name.is_empty() {
            return Err(ProfileError::new("--env requires a nonempty variable name"));
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
            return Err(ProfileError::new(format!(
                "--env cannot override profiler control variable {name:?}"
            )));
        }
    }
    Ok(())
}

pub fn parse_cpuset(value: &str) -> Result<BTreeSet<u32>, ProfileError> {
    let mut cpus = BTreeSet::new();
    for raw in value.split(',') {
        let component = raw.trim();
        if component.is_empty() {
            return Err(ProfileError::new(format!(
                "invalid empty CPU set component in {value:?}"
            )));
        }
        if let Some((start, end)) = component.split_once('-') {
            if end.contains('-') {
                return Err(ProfileError::new(format!(
                    "invalid CPU set range: {component:?}"
                )));
            }
            let start: u32 = parse_number(start, "CPU set")?;
            let end: u32 = parse_number(end, "CPU set")?;
            if end < start {
                return Err(ProfileError::new(format!(
                    "invalid CPU set range: {component:?}"
                )));
            }
            cpus.extend(start..=end);
        } else {
            cpus.insert(parse_number(component, "CPU set")?);
        }
    }
    if cpus.is_empty() {
        return Err(ProfileError::new("CPU set must not be empty"));
    }
    Ok(cpus)
}

fn command_output(
    command: &str,
    arguments: &[&OsStr],
) -> Result<std::process::Output, ProfileError> {
    Command::new(command)
        .args(arguments)
        .output()
        .map_err(|error| ProfileError::new(format!("unable to execute {command:?}: {error}")))
}

fn command_text(
    command: &str,
    arguments: &[&OsStr],
    description: &str,
) -> Result<String, ProfileError> {
    let output = command_output(command, arguments)?;
    if !output.status.success() {
        let diagnostic = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        return Err(ProfileError::new(format!(
            "{description} failed with exit code {}{}",
            output.status.code().unwrap_or(-1),
            if diagnostic.is_empty() {
                String::new()
            } else {
                format!(": {}", truncate(&diagnostic, 512))
            }
        )));
    }
    String::from_utf8(output.stdout)
        .map(|value| value.trim().to_owned())
        .map_err(|_| ProfileError::new(format!("{description} returned non-UTF-8 output")))
}

fn truncate(value: &str, limit: usize) -> String {
    value.chars().take(limit).collect()
}

fn preflight_output(path: &Path) -> Result<(), ProfileError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(ProfileError::new(format!(
                    "profile output path must not be a symbolic link: {}",
                    path.display()
                )));
            }
            if !metadata.is_dir() {
                return Err(ProfileError::new(format!(
                    "profile output path is not a directory: {}",
                    path.display()
                )));
            }
            if fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(ProfileError::new(format!(
                    "profile output directory must be new or empty: {}",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut ancestor = path.parent().unwrap_or_else(|| Path::new("."));
            while !ancestor.exists() {
                ancestor = ancestor.parent().ok_or_else(|| {
                    ProfileError::new(format!(
                        "profile output has no existing ancestor: {}",
                        path.display()
                    ))
                })?;
            }
            let metadata = fs::metadata(ancestor)?;
            if !metadata.is_dir() {
                return Err(ProfileError::new(format!(
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

fn create_output(path: &Path) -> Result<(PathBuf, OutputClaim), ProfileError> {
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::create_dir(path).map_err(|error| {
            ProfileError::new(format!(
                "unable to create profile output directory {}: {error}",
                path.display()
            ))
        })?;
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(ProfileError::new(format!(
            "profile output path changed after preflight: {}",
            path.display()
        )));
    }
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Err(ProfileError::new(format!(
            "profile output directory changed after preflight: {}",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(path)?;
    let claim_path = canonical.join(".tasktrace-profile-claim");
    let mut options = OpenOptions::new();
    options.create_new(true).write(true).mode(0o600);
    let mut claim = options.open(&claim_path).map_err(|error| {
        ProfileError::new(format!(
            "profile output directory could not be claimed atomically: {}: {error}",
            canonical.display()
        ))
    })?;
    claim.write_all(format!("pid={}\n", std::process::id()).as_bytes())?;
    claim.sync_all()?;
    Ok((canonical, OutputClaim { path: claim_path }))
}

fn validate_private_paths_isolated_from_workload_mounts(
    private_paths: &[(&str, &Path)],
    mounts: &[String],
) -> Result<(), ProfileError> {
    let private_paths = private_paths
        .iter()
        .map(|(description, path)| Ok((*description, fs::canonicalize(path)?)))
        .collect::<Result<Vec<_>, ProfileError>>()?;
    for source in workload_mount_sources(mounts)? {
        let absolute = if source.is_absolute() {
            source
        } else {
            env::current_dir()?.join(source)
        };
        let normalized = normalize_components(&absolute)?;
        let resolved = fs::canonicalize(&normalized).unwrap_or(normalized);
        for (description, private_path) in &private_paths {
            if private_path.starts_with(&resolved) || resolved.starts_with(private_path) {
                return Err(ProfileError::new(format!(
                    "workload bind mount overlaps the host-write-private {description}: {}",
                    resolved.display()
                )));
            }
        }
    }
    Ok(())
}

fn preflight_cgroup_v2(require_memory: bool) -> Result<(), ProfileError> {
    let controllers = fs::read_to_string(Path::new(CGROUP_ROOT).join("cgroup.controllers"))
        .map_err(|error| ProfileError::new(format!("cgroup v2 is unavailable: {error}")))?;
    let controllers: BTreeSet<_> = controllers.split_whitespace().collect();
    if !controllers.contains("cpu") {
        return Err(ProfileError::new(
            "cgroup v2 preflight failed: required cpu controller is absent",
        ));
    }
    if require_memory && !controllers.contains("memory") {
        return Err(ProfileError::new(
            "cgroup v2 preflight failed: required memory controller is absent",
        ));
    }
    let membership = fs::read_to_string("/proc/self/cgroup")?;
    if !membership.lines().any(|line| line.starts_with("0::")) {
        return Err(ProfileError::new(
            "cgroup v2 preflight failed: unified membership is absent",
        ));
    }
    Ok(())
}

fn preflight_runtime(config: &ProfileConfig) -> Result<Option<Value>, ProfileError> {
    let scheduler_preflight = if config.scheduler_enabled() {
        match config.scheduler_backend.as_str() {
            "bcc" => Some(crate::scheduler_bcc::BccSchedulerTracer::preflight(
                None,
                config.scheduler_bcc_buffer_pages,
            )?),
            "perf" => {
                let perf_version = command_text(
                    &config.scheduler_perf_binary,
                    &[OsStr::new("version")],
                    "scheduler perf inspection",
                )?;
                if perf_version.trim().is_empty() {
                    return Err(ProfileError::new(
                        "scheduler perf inspection returned an empty version",
                    ));
                }
                Some(json!({
                    "schema_version": "scheduler-perf-preflight.v1",
                    "ready": true,
                    "backend": "perf",
                    "executable": config.scheduler_perf_binary,
                    "version": perf_version.trim(),
                    "clockid": "mono",
                    "mmap_pages": config.scheduler_perf_mmap_pages,
                }))
            }
            _ => None,
        }
    } else {
        None
    };
    let version = command_text(
        &config.docker,
        &[
            OsStr::new("version"),
            OsStr::new("--format"),
            OsStr::new("{{.Server.Version}}"),
        ],
        "Docker runtime inspection",
    )?;
    if version.trim().is_empty() {
        return Err(ProfileError::new(
            "Docker runtime inspection returned an empty server version",
        ));
    }
    let cgroup = command_text(
        &config.docker,
        &[
            OsStr::new("info"),
            OsStr::new("--format"),
            OsStr::new("{{.CgroupVersion}}"),
        ],
        "Docker cgroup-version inspection",
    )?;
    if !matches!(cgroup.trim().to_ascii_lowercase().as_str(), "2" | "v2") {
        return Err(ProfileError::new("Docker runtime does not use cgroup v2"));
    }
    preflight_cgroup_v2(config.memory_enabled())?;
    if let Some(requested) = &config.cpuset_cpus {
        let requested = parse_cpuset(requested)?;
        let online = fs::read_to_string("/sys/devices/system/cpu/online").map_err(|error| {
            ProfileError::new(format!("host online CPU set is unavailable: {error}"))
        })?;
        let online = parse_cpuset(online.trim())?;
        let missing: Vec<_> = requested.difference(&online).copied().collect();
        if !missing.is_empty() {
            return Err(ProfileError::new(format!(
                "requested CPU set contains unavailable logical CPUs: {}",
                missing
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    Ok(scheduler_preflight)
}

fn validate_config(config: &ProfileConfig) -> Result<(), ProfileError> {
    if !matches!(config.runtime.as_str(), "docker" | "enroot") {
        return Err(ProfileError::new(format!(
            "unsupported container runtime: {:?}",
            config.runtime
        )));
    }
    if config.image.trim().is_empty() {
        return Err(ProfileError::new("image must be a nonempty string"));
    }
    if config.command.is_empty() {
        return Err(ProfileError::new("command must be a nonempty string"));
    }
    if config.shell.trim().is_empty() {
        return Err(ProfileError::new("shell must be a nonempty string"));
    }
    if config.string_limit == 0 {
        return Err(ProfileError::new("strace string limit must be positive"));
    }
    if config.attach_timeout.is_zero() {
        return Err(ProfileError::new(
            "scheduler attach timeout must be positive",
        ));
    }
    for (name, value) in [
        ("proc-stat interval", config.proc_stat_interval_sec),
        ("proc-stat baseline", config.proc_stat_baseline_sec),
        ("memory sample interval", config.memory_sample_interval_sec),
    ] {
        if !value.is_finite() || value <= 0.0 {
            return Err(ProfileError::new(format!(
                "{name} duration must be positive"
            )));
        }
    }
    if config.cpuset_cpus.is_none() && config.allocated_cpus.is_none() {
        return Err(ProfileError::new(
            "image profile requires cpuset_cpus or allocated_cpus for normalized utilization",
        ));
    }
    if let Some(cpuset) = &config.cpuset_cpus {
        parse_cpuset(cpuset)?;
    }
    if config.proc_stat_comparison && config.cpuset_cpus.is_none() {
        return Err(ProfileError::new(
            "proc-stat comparison requires a concrete cpuset_cpus allocation",
        ));
    }
    if config
        .allocated_cpus
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(ProfileError::new("allocated CPU count must be positive"));
    }
    validate_profile_identities(config)?;
    if config.runtime == "enroot" {
        let harness = HarnessFile::create()?;
        let network_evidence_path = (config.network_policy.effective_mode()
            == NetworkMode::NoNetwork)
            .then(|| harness.directory.join("enroot_network_enforcement.json"));
        return enroot_runtime_config(config, harness.path.clone(), network_evidence_path)
            .validate()
            .map_err(ProfileError::from);
    }
    if config
        .cpu_limit
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(ProfileError::new("CPU limit must be positive"));
    }
    if config
        .memory_swap_total_limit_bytes
        .zip(config.memory_limit_bytes)
        .is_some_and(|(swap, memory)| swap < memory)
    {
        return Err(ProfileError::new(
            "memory swap total limit must be at least the memory limit",
        ));
    }
    if config.memory_limit_bytes == Some(0) {
        return Err(ProfileError::new("memory limit must be positive"));
    }
    if config.memory_swap_total_limit_bytes == Some(0) {
        return Err(ProfileError::new(
            "memory swap total limit must be positive",
        ));
    }
    if config.memory_swap_total_limit_bytes.is_some() && config.memory_limit_bytes.is_none() {
        return Err(ProfileError::new(
            "memory swap total limit requires a memory limit",
        ));
    }
    if config.coordinated_baseline && !config.profile_sets.is_empty() {
        return Err(ProfileError::new(
            "coordinated baseline is mutually exclusive with profile sets",
        ));
    }
    let sets: BTreeSet<_> = config.profile_sets.iter().collect();
    if sets.len() != config.profile_sets.len() {
        return Err(ProfileError::new(
            "profile sets must not contain duplicates",
        ));
    }
    if config.profile_sets.iter().any(|value| {
        !matches!(
            value.as_str(),
            "cpu-average"
                | "cpu-peak"
                | "cpu-compare"
                | "memory-detail"
                | "software-deep"
                | "hardware-deep"
        )
    }) {
        return Err(ProfileError::new(
            "native container measurement supports only cpu-average, cpu-peak, cpu-compare, memory-detail, software-deep, and hardware-deep profile sets",
        ));
    }
    if config.hardware_perf_binary.trim().is_empty() {
        return Err(ProfileError::new(
            "hardware perf executable must be nonempty",
        ));
    }
    if config.hardware_perf_attach_timeout.is_zero() {
        return Err(ProfileError::new(
            "hardware perf attach timeout must be positive",
        ));
    }
    if !config.hardware_perf_minimum_running_ratio.is_finite()
        || config.hardware_perf_minimum_running_ratio <= 0.0
        || config.hardware_perf_minimum_running_ratio > 1.0
    {
        return Err(ProfileError::new(
            "hardware perf minimum running ratio must be in (0, 1]",
        ));
    }
    if config.hardware_perf_events.is_empty()
        || config
            .hardware_perf_events
            .iter()
            .any(|event| event.trim().is_empty())
        || config
            .hardware_perf_events
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != config.hardware_perf_events.len()
    {
        return Err(ProfileError::new(
            "hardware perf events must be nonempty and unique",
        ));
    }
    if config.scheduler_enabled() && !matches!(config.scheduler_backend.as_str(), "bcc" | "perf") {
        return Err(ProfileError::new(
            "native scheduler profiling supports only bcc or perf backends",
        ));
    }
    if config.scheduler_enabled()
        && config.scheduler_backend == "perf"
        && config.scheduler_perf_binary.trim().is_empty()
    {
        return Err(ProfileError::new(
            "scheduler perf executable must be nonempty",
        ));
    }
    if config
        .scheduler_bcc_buffer_pages
        .is_some_and(|pages| pages == 0 || !pages.is_power_of_two() || pages > i32::MAX as u64)
    {
        return Err(ProfileError::new(
            "scheduler BCC perf-buffer pages must be a positive power of two representable as i32",
        ));
    }
    if config.scheduler_perf_mmap_pages == Some(0) {
        return Err(ProfileError::new(
            "scheduler perf mmap pages must be positive",
        ));
    }
    if !config.scheduler_cpu_tolerance_fraction.is_finite()
        || config.scheduler_cpu_tolerance_fraction < 0.0
        || !config.scheduler_cpu_tolerance_sec.is_finite()
        || config.scheduler_cpu_tolerance_sec < 0.0
    {
        return Err(ProfileError::new(
            "scheduler CPU tolerances must be finite and nonnegative",
        ));
    }
    if config.coordinated_baseline && config.memory_enabled() {
        return Err(ProfileError::new(
            "coordinated baseline cannot enable memory profiling",
        ));
    }
    if config.network_policy.effective_mode() == NetworkMode::Allowlist {
        if config.network_policy_proxy_image.trim().is_empty() {
            return Err(ProfileError::new(
                "allowlist policy requires a nonempty proxy image",
            ));
        }
        if config.network_policy_external_network.trim().is_empty() {
            return Err(ProfileError::new(
                "allowlist policy requires a nonempty external proxy network",
            ));
        }
        validate_workload_mount_sources(&workload_mount_sources(&config.mounts)?)
            .map_err(|error| ProfileError::new(error.to_string()))?;
    }
    if let Some(command) = &config.verification_command {
        if command.trim().is_empty() {
            return Err(ProfileError::new(
                "verification command must be a nonempty string",
            ));
        }
        if !config.is_coordinated() {
            return Err(ProfileError::new(
                "verification command requires a profile set or coordinated baseline",
            ));
        }
    }
    validate_mounts(&config.mounts)?;
    validate_environment(&config.environment)?;
    Ok(())
}

fn preflight_enroot_profile(config: &ProfileConfig) -> Result<Value, ProfileError> {
    let harness = HarnessFile::create()?;
    let network_evidence_path = (config.network_policy.effective_mode() == NetworkMode::NoNetwork)
        .then(|| harness.directory.join("enroot_network_enforcement.json"));
    let runtime = enroot_runtime_config(config, harness.path.clone(), network_evidence_path);
    let preflight = crate::enroot_runtime::preflight(&runtime)?;
    let image_kind = match preflight.image_identity.kind {
        crate::enroot_runtime::EnrootImageKind::ImageFile => "enroot-image-file",
        crate::enroot_runtime::EnrootImageKind::CreatedContainer => "enroot-created-container",
    };
    let image_identity = serde_json::to_value(&preflight.image_identity)?;
    let mut checks = vec![
        json!({"name": "profile_config", "status": "passed"}),
        json!({"name": "output_directory", "status": "passed"}),
        json!({
            "name": "cpu_allocation",
            "status": "passed",
            "enforcement": false,
            "affinity_contract": preflight.affinity_contract,
        }),
        json!({
            "name": "runtime",
            "status": "passed",
            "runtime": "enroot",
            "executable": preflight.executable,
            "version": preflight.version,
            "taskset_executable": preflight.taskset_executable,
        }),
        json!({
            "name": "cgroup_v2",
            "status": "passed",
            "required": false,
            "controllers": Vec::<String>::new(),
        }),
        json!({"name": "image", "status": "passed", "kind": image_kind}),
    ];
    if config.cpuset_cpus.is_some() {
        checks.push(json!({
            "name": "requested_launch_affinity",
            "status": "passed",
            "cpuset_cpus": config.cpuset_cpus,
            "containment_enforced": false,
            "exclusivity_enforced": false,
        }));
    }
    Ok(json!({
        "schema_version": "tasktrace-profile-preflight.v1",
        "ready": true,
        "runtime": "enroot",
        "image_identity": image_identity,
        "profile_sets": config.profile_sets,
        "cpuset_cpus": config.cpuset_cpus,
        "allocated_cpus": config.allocated_cpus,
        "proc_stat_comparison": false,
        "scheduler_concurrency": false,
        "hardware_perf": Value::Null,
        "memory_profile": false,
        "mounts": preflight.mounts,
        "checks": checks,
        "limitations": preflight.limitations,
    }))
}

/// Validate static prerequisites without creating the output directory or
/// launching workload code.
pub fn preflight_profile(config: &ProfileConfig) -> Result<Value, ProfileError> {
    validate_config(config)?;
    preflight_output(&config.output_dir)?;
    if config.runtime == "enroot" {
        return preflight_enroot_profile(config);
    }
    let scheduler_preflight = preflight_runtime(config)?;
    if config.proc_stat_comparison {
        let counters = parse_proc_stat(&fs::read_to_string("/proc/stat")?).map_err(|error| {
            ProfileError::new(format!("host /proc/stat preflight failed: {error}"))
        })?;
        let requested = parse_cpuset(
            config
                .cpuset_cpus
                .as_deref()
                .expect("validated proc-stat cpuset"),
        )?;
        let missing: Vec<_> = requested
            .iter()
            .filter(|cpu| !counters.contains_key(cpu))
            .copied()
            .collect();
        if !missing.is_empty() {
            return Err(ProfileError::new(format!(
                "host /proc/stat omits requested logical CPUs: {}",
                missing
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    let identity = image_identity(config)?;
    let mut checks = vec![
        json!({"name": "profile_config", "status": "passed"}),
        json!({"name": "output_directory", "status": "passed"}),
        json!({"name": "cpu_allocation", "status": "passed"}),
        json!({"name": "runtime", "status": "passed", "runtime": "docker"}),
        json!({"name": "cgroup_v2", "status": "passed", "required": true, "controllers": if config.memory_enabled() { vec!["cpu", "memory"] } else { vec!["cpu"] }}),
        json!({"name": "image", "status": "passed", "kind": "docker-image"}),
    ];
    let mut limitations = vec![
        "Dynamic cgroup scope, descendant cleanup, and effective CPU placement remain pending container creation."
            .to_owned(),
    ];
    if config.network_policy.effective_mode() == NetworkMode::Allowlist {
        let proxy_identity = image_identity_for(config, &config.network_policy_proxy_image)?;
        checks.push(json!({
            "name": "network_policy_proxy_image",
            "status": "passed",
            "image_identity": proxy_identity,
        }));
        let reason = "Allowlist network creation, immutable topology validation, and proxy readiness remain pending execution.".to_owned();
        checks.push(json!({
            "name": "network_policy_dynamic_enforcement",
            "status": "pending",
            "reason": reason,
        }));
        limitations.push(reason);
    }
    if config.proc_stat_comparison {
        checks.push(json!({
            "name": "proc_stat_host_context",
            "status": "passed",
            "path": "/proc/stat",
            "cpuset_cpus": config.cpuset_cpus,
            "task_attributed": false,
        }));
    }
    if config.hardware_enabled() {
        checks.push(json!({
            "name": "hardware_perf_dynamic_attach",
            "status": "pending",
            "executable": config.hardware_perf_binary,
            "events": config.hardware_perf_events,
            "cpuset_cpus": config.cpuset_cpus,
            "reason": "Cgroup-scoped perf_event policy, exact cgroup attribution, acknowledged control readiness, and counter runtime ratios remain pending container creation.",
        }));
    }
    if config.scheduler_enabled() {
        let mut static_check = scheduler_preflight.unwrap_or_else(|| json!({}));
        let static_object = static_check
            .as_object_mut()
            .expect("scheduler preflight report is an object");
        static_object.insert(
            "name".to_owned(),
            Value::String(format!("scheduler_{}_static", config.scheduler_backend)),
        );
        static_object.insert("status".to_owned(), Value::String("passed".to_owned()));
        checks.push(static_check);
        checks.push(json!({
            "name": format!("scheduler_{}_dynamic_attach", config.scheduler_backend),
            "status": "pending",
            "reason": "Cgroup-scoped scheduler attachment, event loss, and cgroup CPU reconciliation remain pending execution.",
        }));
    }
    if config.memory_enabled() {
        checks.push(json!({
            "name": "memory_cgroup_dynamic_scope",
            "status": "pending",
            "reason": "Leaf/ancestor memory scope and memory.peak remain pending container creation.",
        }));
    }
    Ok(json!({
        "schema_version": "tasktrace-profile-preflight.v1",
        "ready": true,
        "runtime": "docker",
        "image_identity": identity,
        "profile_sets": config.profile_sets,
        "cpuset_cpus": config.cpuset_cpus,
        "allocated_cpus": config.allocated_cpus,
        "proc_stat_comparison": config.proc_stat_comparison,
        "proc_stat_interval_sec": config.proc_stat_interval_sec,
        "proc_stat_baseline_sec": config.proc_stat_baseline_sec,
        "scheduler_concurrency": config.scheduler_enabled(),
        "scheduler_backend": if config.scheduler_enabled() { json!(config.scheduler_backend) } else { Value::Null },
        "scheduler_bcc_buffer_pages": if config.scheduler_enabled() && config.scheduler_backend == "bcc" { json!(config.scheduler_bcc_buffer_pages) } else { Value::Null },
        "scheduler_perf_binary": if config.scheduler_enabled() && config.scheduler_backend == "perf" { json!(config.scheduler_perf_binary) } else { Value::Null },
        "scheduler_perf_mmap_pages": if config.scheduler_enabled() && config.scheduler_backend == "perf" { json!(config.scheduler_perf_mmap_pages) } else { Value::Null },
        "hardware_perf": if config.hardware_enabled() {
            json!({
                "binary": config.hardware_perf_binary,
                "events": config.hardware_perf_events,
                "attach_timeout_sec": config.hardware_perf_attach_timeout.as_secs_f64(),
                "minimum_running_ratio": config.hardware_perf_minimum_running_ratio,
                "cpuset_cpus": config.cpuset_cpus,
                "hybrid_pmu_counts_merged": true,
            })
        } else {
            Value::Null
        },
        "memory_profile": config.memory_enabled(),
        "memory_sample_interval_sec": config.memory_sample_interval_sec,
        "checks": checks,
        "limitations": limitations,
    }))
}

fn image_identity(config: &ProfileConfig) -> Result<Value, ProfileError> {
    image_identity_for(config, &config.image)
}

fn image_identity_for(config: &ProfileConfig, image: &str) -> Result<Value, ProfileError> {
    let output = command_output(
        &config.docker,
        &[
            OsStr::new("image"),
            OsStr::new("inspect"),
            OsStr::new(image),
        ],
    )?;
    if !output.status.success() {
        return Err(ProfileError::new("unable to inspect Docker image identity"));
    }
    let payload: Value = serde_json::from_slice(&output.stdout)
        .map_err(|_| ProfileError::new("invalid Docker image inspection output"))?;
    let rows = payload
        .as_array()
        .filter(|rows| rows.len() == 1)
        .ok_or_else(|| ProfileError::new("invalid Docker image inspection output"))?;
    let row = rows[0]
        .as_object()
        .ok_or_else(|| ProfileError::new("invalid Docker image inspection output"))?;
    let id = row
        .get("Id")
        .and_then(Value::as_str)
        .ok_or_else(|| ProfileError::new("Docker preflight image identity has no immutable ID"))?;
    let digest = id.strip_prefix("sha256:").unwrap_or_default();
    if digest.len() != 64 || !digest.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(ProfileError::new(
            "Docker preflight image identity has no immutable ID",
        ));
    }
    let rootfs = row.get("RootFS").and_then(Value::as_object);
    Ok(json!({
        "id": id,
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
    }))
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, ProfileError> {
    let mut file = File::open(path)?;
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

fn random_bytes<const N: usize>() -> Result<[u8; N], ProfileError> {
    let mut bytes = [0u8; N];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(bytes)
}

fn run_id() -> Result<String, ProfileError> {
    let mut bytes = random_bytes::<16>()?;
    bytes[6] = (bytes[6] & 0x0f) | 0x40;
    bytes[8] = (bytes[8] & 0x3f) | 0x80;
    Ok(format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        bytes[8], bytes[9], bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15]
    ))
}

fn utc_now() -> Result<String, ProfileError> {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| {
            ProfileError::new(format!("system clock is before Unix epoch: {error}"))
        })?;
    let seconds: libc::time_t = duration
        .as_secs()
        .try_into()
        .map_err(|_| ProfileError::new("system time does not fit time_t"))?;
    let mut result = std::mem::MaybeUninit::<libc::tm>::zeroed();
    let converted = unsafe { libc::gmtime_r(&seconds, result.as_mut_ptr()) };
    if converted.is_null() {
        return Err(ProfileError::new("unable to convert current UTC time"));
    }
    let result = unsafe { result.assume_init() };
    Ok(format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}+00:00",
        result.tm_year + 1900,
        result.tm_mon + 1,
        result.tm_mday,
        result.tm_hour,
        result.tm_min,
        result.tm_sec,
        duration.subsec_nanos()
    ))
}

static TEMP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// A workload-writable exchange directory that is never used as the final
/// artifact publication directory.
///
/// The directory descriptor is opened before Docker starts and every entry is
/// addressed by one validated basename with `openat(O_NOFOLLOW)`.  This keeps
/// a task-created symlink, directory, device, socket, or substituted pathname
/// from redirecting a privileged controller read or FIFO write.
#[derive(Debug)]
struct TaskExchange {
    path: PathBuf,
    directory: File,
    cleanup_complete: bool,
}

impl TaskExchange {
    fn create(parent: &Path) -> Result<Self, ProfileError> {
        let path = parent.join(".tasktrace-task-exchange");
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o777);
        builder.create(&path)?;
        // DirBuilder's mode is affected by umask. Workloads may run as an
        // arbitrary non-root UID, so make the dedicated exchange explicitly
        // writable. Only this child is mounted writable; its publication
        // parent is mounted separately read-only.
        fs::set_permissions(&path, fs::Permissions::from_mode(0o777))?;
        let directory = OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_CLOEXEC)
            .open(&path)?;
        let metadata = directory.metadata()?;
        if !metadata.is_dir() {
            return Err(ProfileError::new(format!(
                "task exchange is not a directory: {}",
                path.display()
            )));
        }
        Ok(Self {
            path,
            directory,
            cleanup_complete: false,
        })
    }

    fn path(&self) -> &Path {
        &self.path
    }

    fn entry_name(name: &str) -> Result<CString, ProfileError> {
        let mut components = Path::new(name).components();
        if !matches!(components.next(), Some(Component::Normal(_))) || components.next().is_some() {
            return Err(ProfileError::new(format!(
                "task exchange entry must be one basename: {name:?}"
            )));
        }
        CString::new(name.as_bytes())
            .map_err(|_| ProfileError::new(format!("task exchange entry contains NUL: {name:?}")))
    }

    fn open_path_entry(&self, name: &str) -> Result<Option<File>, ProfileError> {
        let name_c = Self::entry_name(name)?;
        let descriptor = unsafe {
            libc::openat(
                self.directory.as_raw_fd(),
                name_c.as_ptr(),
                libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC,
            )
        };
        if descriptor >= 0 {
            return Ok(Some(unsafe { File::from_raw_fd(descriptor) }));
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ENOENT) {
            return Ok(None);
        }
        Err(ProfileError::new(format!(
            "could not safely acquire task exchange entry {name:?}: {error}"
        )))
    }

    fn reopen_path(path_file: &File, write: bool, nonblocking: bool) -> io::Result<File> {
        let path = PathBuf::from(format!("/proc/self/fd/{}", path_file.as_raw_fd()));
        let mut options = OpenOptions::new();
        options
            .read(!write)
            .write(write)
            .custom_flags(libc::O_CLOEXEC | if nonblocking { libc::O_NONBLOCK } else { 0 });
        options.open(path)
    }

    fn same_inode(
        name: &str,
        expected: &fs::Metadata,
        observed: &fs::Metadata,
    ) -> Result<(), ProfileError> {
        if expected.dev() != observed.dev() || expected.ino() != observed.ino() {
            return Err(ProfileError::new(format!(
                "validated task exchange entry changed inode while opening: {name:?}"
            )));
        }
        Ok(())
    }

    fn open_regular(&self, name: &str) -> Result<Option<File>, ProfileError> {
        let Some(path_file) = self.open_path_entry(name)? else {
            return Ok(None);
        };
        let path_metadata = path_file.metadata()?;
        Self::validate_regular(name, &path_metadata)?;
        let file = Self::reopen_path(&path_file, false, false).map_err(|error| {
            ProfileError::new(format!(
                "could not reopen validated task exchange entry {name:?}: {error}"
            ))
        })?;
        let metadata = file.metadata()?;
        Self::same_inode(name, &path_metadata, &metadata)?;
        Self::validate_regular(name, &metadata)?;
        Ok(Some(file))
    }

    fn validate_regular(name: &str, metadata: &fs::Metadata) -> Result<(), ProfileError> {
        if !metadata.is_file() {
            return Err(ProfileError::new(format!(
                "task exchange entry is not a regular file: {name:?}"
            )));
        }
        if metadata.nlink() != 1 {
            return Err(ProfileError::new(format!(
                "task exchange regular file has an unsafe hard-link count: {name:?}: {}",
                metadata.nlink()
            )));
        }
        Ok(())
    }

    fn regular_ready(&self, name: &str) -> Result<bool, ProfileError> {
        let Some(file) = self.open_regular(name)? else {
            return Ok(false);
        };
        let metadata = file.metadata()?;
        Self::validate_regular(name, &metadata)?;
        Ok(metadata.len() > 0)
    }

    fn copy_regular_to(
        &self,
        name: &str,
        destination: &Path,
        limit: u64,
    ) -> Result<bool, ProfileError> {
        static COPY_SEQUENCE: AtomicU64 = AtomicU64::new(0);
        let Some(mut file) = self.open_regular(name)? else {
            return Ok(false);
        };
        let before = file.metadata()?;
        Self::validate_regular(name, &before)?;
        if before.len() > limit {
            return Err(ProfileError::new(format!(
                "task exchange artifact exceeds the {} byte safety limit: {name:?}",
                limit
            )));
        }
        let destination_name =
            destination
                .file_name()
                .and_then(OsStr::to_str)
                .ok_or_else(|| {
                    ProfileError::new(format!(
                        "invalid task artifact destination: {}",
                        destination.display()
                    ))
                })?;
        let temporary = destination.with_file_name(format!(
            ".{destination_name}.task-copy.{}.{}",
            std::process::id(),
            COPY_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        ));
        let copied = (|| -> Result<(), ProfileError> {
            let mut destination_file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&temporary)?;
            let copied = io::copy(&mut (&mut file).take(limit + 1), &mut destination_file)?;
            if copied > limit {
                return Err(ProfileError::new(format!(
                    "task exchange artifact grew past the {limit} byte safety limit while reading: {name:?}"
                )));
            }
            destination_file.sync_all()?;
            drop(destination_file);
            let after = file.metadata()?;
            Self::validate_regular(name, &after)?;
            let changed = before.dev() != after.dev()
                || before.ino() != after.ino()
                || before.len() != after.len()
                || before.len() != copied
                || before.mtime() != after.mtime()
                || before.mtime_nsec() != after.mtime_nsec()
                || before.ctime() != after.ctime()
                || before.ctime_nsec() != after.ctime_nsec();
            if changed {
                return Err(ProfileError::new(format!(
                    "task exchange artifact changed while it was being copied: {name:?}"
                )));
            }
            fs::rename(&temporary, destination).map_err(|error| {
                ProfileError::new(format!(
                    "unable to publish copied task artifact {}: {error}",
                    destination.display()
                ))
            })
        })();
        if copied.is_err() {
            let _ = fs::remove_file(&temporary);
        }
        copied.map(|()| true)
    }

    fn write_fifo(&self, name: &str, bytes: &[u8]) -> Result<bool, ProfileError> {
        let Some(path_file) = self.open_path_entry(name)? else {
            return Ok(false);
        };
        let path_metadata = path_file.metadata()?;
        if path_metadata.mode() & libc::S_IFMT != libc::S_IFIFO {
            return Err(ProfileError::new(format!(
                "task exchange coordinator gate is not a FIFO: {name:?}"
            )));
        }
        if path_metadata.nlink() != 1 {
            return Err(ProfileError::new(format!(
                "task exchange FIFO has an unsafe hard-link count: {name:?}: {}",
                path_metadata.nlink()
            )));
        }
        let mut file = match Self::reopen_path(&path_file, true, true) {
            Ok(file) => file,
            Err(error) => {
                // No FIFO reader yet. The caller polls until its bounded
                // deadline; every retry repeats O_PATH type validation.
                if matches!(
                    error.raw_os_error(),
                    Some(libc::EAGAIN | libc::EINTR | libc::ENXIO)
                ) {
                    return Ok(false);
                }
                return Err(ProfileError::new(format!(
                    "could not reopen validated task exchange FIFO {name:?}: {error}"
                )));
            }
        };
        let metadata = file.metadata()?;
        Self::same_inode(name, &path_metadata, &metadata)?;
        if metadata.mode() & libc::S_IFMT != libc::S_IFIFO {
            return Err(ProfileError::new(format!(
                "validated task exchange coordinator gate reopened as a non-FIFO: {name:?}"
            )));
        }
        if metadata.nlink() != 1 {
            return Err(ProfileError::new(format!(
                "validated task exchange FIFO changed hard-link count: {name:?}: {}",
                metadata.nlink()
            )));
        }
        if let Err(error) = file.write_all(bytes) {
            if matches!(
                error.raw_os_error(),
                Some(libc::EAGAIN | libc::EINTR | libc::EPIPE)
            ) {
                return Ok(false);
            }
            return Err(error.into());
        }
        Ok(true)
    }

    fn cleanup(&mut self) -> Result<(), ProfileError> {
        if self.cleanup_complete {
            return Ok(());
        }
        fs::remove_dir_all(&self.path).map_err(|error| {
            ProfileError::new(format!(
                "could not remove task exchange {}: {error}",
                self.path.display()
            ))
        })?;
        match fs::symlink_metadata(&self.path) {
            Err(error) if error.kind() == io::ErrorKind::NotFound => {
                self.cleanup_complete = true;
                Ok(())
            }
            Ok(_) => Err(ProfileError::new(format!(
                "task exchange survived cleanup: {}",
                self.path.display()
            ))),
            Err(error) => Err(ProfileError::new(format!(
                "could not verify task exchange cleanup {}: {error}",
                self.path.display()
            ))),
        }
    }
}

impl Drop for TaskExchange {
    fn drop(&mut self) {
        if !self.cleanup_complete {
            let _ = fs::remove_dir_all(&self.path);
        }
    }
}

struct HarnessFile {
    directory: PathBuf,
    path: PathBuf,
    docker_stdout: PathBuf,
    docker_stderr: PathBuf,
}

impl HarnessFile {
    fn create() -> Result<Self, ProfileError> {
        let random = random_bytes::<8>()?;
        let sequence = TEMP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let directory = env::temp_dir().join(format!(
            "tasktrace-profile-rust-{}-{:016x}-{sequence}",
            std::process::id(),
            u64::from_ne_bytes(random)
        ));
        let mut builder = fs::DirBuilder::new();
        builder.mode(0o700);
        builder.create(&directory)?;
        let path = directory.join("entrypoint.sh");
        let mut options = OpenOptions::new();
        options.write(true).create_new(true).mode(0o644);
        let mut file = options.open(&path)?;
        file.write_all(ENTRYPOINT.as_bytes())?;
        file.sync_all()?;
        let docker_stdout = directory.join("docker.stdout.txt");
        let docker_stderr = directory.join("docker.stderr.txt");
        Ok(Self {
            directory,
            path,
            docker_stdout,
            docker_stderr,
        })
    }
}

impl Drop for HarnessFile {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

fn normalize_components(path: &Path) -> Result<PathBuf, ProfileError> {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::RootDir => normalized.push("/"),
            Component::Normal(value) => normalized.push(value),
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() {
                    return Err(ProfileError::new(format!(
                        "path escapes its root: {}",
                        path.display()
                    )));
                }
            }
            Component::Prefix(_) => {
                return Err(ProfileError::new("Windows path prefixes are unsupported"));
            }
        }
    }
    Ok(normalized)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct CpuLimit {
    quota_usec: Option<u64>,
    period_usec: Option<u64>,
    state: &'static str,
}

#[derive(Clone, Debug)]
struct CpuSourceSnapshot {
    path: PathBuf,
    relative_path: String,
    acquisition_start_ns: u64,
    acquisition_end_ns: u64,
    cpu_stat_text: String,
    cpu_stat: BTreeMap<String, u64>,
    cpu_max_text: Option<String>,
    cpu_limit: CpuLimit,
    errors: Vec<String>,
}

fn parse_cpu_stat(text: &str, source: &str) -> Result<BTreeMap<String, u64>, ProfileError> {
    let mut counters = BTreeMap::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.is_empty() {
            continue;
        }
        if fields.len() != 2 {
            return Err(ProfileError::new(format!(
                "malformed {source} row: {line:?}"
            )));
        }
        let value = fields[1]
            .parse::<u64>()
            .map_err(|_| ProfileError::new(format!("invalid {source} counter in row: {line:?}")))?;
        if counters.insert(fields[0].to_owned(), value).is_some() {
            return Err(ProfileError::new(format!(
                "duplicate {source} counter: {:?}",
                fields[0]
            )));
        }
    }
    for required in [
        "usage_usec",
        "user_usec",
        "system_usec",
        "nr_periods",
        "nr_throttled",
        "throttled_usec",
    ] {
        if !counters.contains_key(required) {
            return Err(ProfileError::new(format!(
                "{source} lacks required counter {required:?}"
            )));
        }
    }
    Ok(counters)
}

fn parse_cpu_max(text: &str, root: bool, source: &str) -> Result<CpuLimit, ProfileError> {
    if root && text.trim().is_empty() {
        return Ok(CpuLimit {
            quota_usec: None,
            period_usec: None,
            state: "not-applicable-root",
        });
    }
    let fields: Vec<_> = text.split_whitespace().collect();
    if fields.len() != 2 {
        return Err(ProfileError::new(format!("malformed {source}: {text:?}")));
    }
    let period = fields[1]
        .parse::<u64>()
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| ProfileError::new(format!("invalid {source} period: {:?}", fields[1])))?;
    let quota = if fields[0] == "max" {
        None
    } else {
        Some(
            fields[0]
                .parse::<u64>()
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    ProfileError::new(format!("invalid {source} quota: {:?}", fields[0]))
                })?,
        )
    };
    Ok(CpuLimit {
        quota_usec: quota,
        period_usec: Some(period),
        state: if quota.is_some() {
            "finite"
        } else {
            "unlimited"
        },
    })
}

fn relative_cgroup_path(path: &Path) -> Result<String, ProfileError> {
    let relative = path.strip_prefix(CGROUP_ROOT).map_err(|_| {
        ProfileError::new(format!(
            "measured cgroup is outside the cgroup-v2 root: {}",
            path.display()
        ))
    })?;
    let text = relative.to_string_lossy();
    Ok(if text.is_empty() {
        ".".to_owned()
    } else {
        text.into_owned()
    })
}

fn capture_cpu_source(path: &Path) -> CpuSourceSnapshot {
    let acquisition_start_ns = monotonic_ns();
    let mut errors = Vec::new();
    let cpu_stat_text = match fs::read_to_string(path.join("cpu.stat")) {
        Ok(value) => value,
        Err(error) => {
            errors.push(format!(
                "required CPU accounting is unavailable: cpu.stat: {error}"
            ));
            String::new()
        }
    };
    let cpu_stat = match parse_cpu_stat(&cpu_stat_text, "cpu.stat") {
        Ok(value) => value,
        Err(error) => {
            errors.push(error.to_string());
            BTreeMap::new()
        }
    };
    let root = path == Path::new(CGROUP_ROOT);
    let cpu_max_text = if root {
        None
    } else {
        match fs::read_to_string(path.join("cpu.max")) {
            Ok(value) => Some(value),
            Err(error) => {
                errors.push(format!(
                    "required CPU limit is unavailable: cpu.max: {error}"
                ));
                None
            }
        }
    };
    let cpu_limit = match parse_cpu_max(cpu_max_text.as_deref().unwrap_or(""), root, "cpu.max") {
        Ok(value) => value,
        Err(error) => {
            errors.push(error.to_string());
            CpuLimit {
                quota_usec: None,
                period_usec: None,
                state: "invalid",
            }
        }
    };
    let relative_path = relative_cgroup_path(path).unwrap_or_else(|error| {
        errors.push(error.to_string());
        "<invalid>".to_owned()
    });
    CpuSourceSnapshot {
        path: path.to_path_buf(),
        relative_path,
        acquisition_start_ns,
        acquisition_end_ns: monotonic_ns(),
        cpu_stat_text,
        cpu_stat,
        cpu_max_text,
        cpu_limit,
        errors,
    }
}

fn ancestor_paths(leaf: &Path) -> Result<Vec<PathBuf>, ProfileError> {
    let root = fs::canonicalize(CGROUP_ROOT)?;
    let leaf = fs::canonicalize(leaf)?;
    if !leaf.starts_with(&root) || leaf == root {
        return Err(ProfileError::new(format!(
            "invalid disposable-container cgroup path: {}",
            leaf.display()
        )));
    }
    let mut paths = Vec::new();
    let mut current = leaf
        .parent()
        .ok_or_else(|| ProfileError::new("measured cgroup has no parent"))?;
    loop {
        paths.push(current.to_path_buf());
        if current == root {
            break;
        }
        current = current.parent().ok_or_else(|| {
            ProfileError::new("measured cgroup ancestor chain does not reach cgroup root")
        })?;
    }
    Ok(paths)
}

fn capture_chain(leaf: &Path) -> Result<(CpuSourceSnapshot, Vec<CpuSourceSnapshot>), ProfileError> {
    let leaf_snapshot = capture_cpu_source(leaf);
    let ancestors = ancestor_paths(leaf)?
        .into_iter()
        .map(|path| capture_cpu_source(&path))
        .collect();
    Ok((leaf_snapshot, ancestors))
}

fn snapshot_json(snapshot: &CpuSourceSnapshot) -> Value {
    json!({
        "acquisition_start_ns": snapshot.acquisition_start_ns,
        "acquisition_end_ns": snapshot.acquisition_end_ns,
        "cpu_max_quota_usec": snapshot.cpu_limit.quota_usec,
        "cpu_max_period_usec": snapshot.cpu_limit.period_usec,
        "cpu_max_state": snapshot.cpu_limit.state,
        "cpu_stat": snapshot.cpu_stat,
        "cpu_stat_available": snapshot.errors.iter().all(|value| !value.contains("cpu.stat")),
        "errors": snapshot.errors,
        "path": snapshot.relative_path,
    })
}

fn counter_delta(
    before: &BTreeMap<String, u64>,
    after: &BTreeMap<String, u64>,
    scope: &str,
) -> (BTreeMap<String, u64>, Vec<String>) {
    let mut keys: BTreeSet<&String> = before.keys().collect();
    keys.extend(after.keys());
    let mut delta = BTreeMap::new();
    let mut errors = Vec::new();
    for key in keys {
        match (before.get(key), after.get(key)) {
            (Some(start), Some(end)) if end >= start => {
                delta.insert(key.clone(), end - start);
            }
            (Some(start), Some(end)) => errors.push(format!(
                "{scope} cpu.stat counter {key:?} moved backwards: {start} -> {end}"
            )),
            _ => errors.push(format!(
                "{scope} cpu.stat counter {key:?} was absent at one boundary"
            )),
        }
    }
    (delta, errors)
}

fn source_pair_report(before: &CpuSourceSnapshot, after: &CpuSourceSnapshot, leaf: bool) -> Value {
    let scope = if leaf { "leaf" } else { "ancestor" };
    let (delta, regression_errors) = counter_delta(&before.cpu_stat, &after.cpu_stat, scope);
    let mut evidence_errors = before.errors.clone();
    evidence_errors.extend(after.errors.clone());
    evidence_errors.extend(regression_errors.clone());
    deduplicate(&mut evidence_errors);
    let limit_changed = before.cpu_limit != after.cpu_limit;
    if limit_changed {
        evidence_errors.push(format!(
            "{scope} cpu.max changed during observation: {:?} -> {:?}",
            before.cpu_limit, after.cpu_limit
        ));
    }
    let finite = before.cpu_limit.state == "finite" || after.cpu_limit.state == "finite";
    let throttle: BTreeMap<_, _> = ["nr_throttled", "throttled_usec", "nr_bursts", "burst_usec"]
        .into_iter()
        .filter_map(|key| {
            delta
                .get(key)
                .copied()
                .filter(|value| *value > 0)
                .map(|value| (key, value))
        })
        .collect();
    let mut invalid = evidence_errors.clone();
    if !throttle.is_empty() {
        invalid.push(format!(
            "{scope} cgroup recorded CPU throttling; CPU performance evidence fails closed"
        ));
    }
    if finite {
        invalid.push(format!(
            "{scope} cgroup has a finite cpu.max; unconstrained CPU performance is not proven"
        ));
    }
    deduplicate(&mut invalid);
    json!({
        "path": if leaf { Value::Null } else { Value::String(before.relative_path.clone()) },
        "start": snapshot_json(before),
        "finish": snapshot_json(after),
        "cpu_stat_delta": delta,
        "cpu_max_changed": limit_changed,
        "counter_regression_observed": !regression_errors.is_empty(),
        "finite_limit_present": finite,
        "throttle_deltas": throttle,
        "throttling_observed": !throttle.is_empty(),
        "evidence_complete": evidence_errors.is_empty(),
        "evidence_valid": evidence_errors.is_empty(),
        "limits_stable": !limit_changed,
        "policy_stable": !limit_changed,
        "valid_for_cpu_performance": invalid.is_empty(),
        "evidence_errors": evidence_errors,
        "invalid_reasons": invalid,
    })
}

fn deduplicate(values: &mut Vec<String>) {
    let mut observed = BTreeSet::new();
    values.retain(|value| observed.insert(value.clone()));
}

#[derive(Clone, Debug)]
struct CpuPlacement {
    requested: Option<String>,
    docker: Option<String>,
    process: Option<String>,
    cgroup: Option<String>,
    status: &'static str,
}

impl CpuPlacement {
    fn to_json(&self) -> Value {
        json!({
            "requested_cpuset_cpus": self.requested,
            "docker_host_config_cpuset_cpus": self.docker,
            "container_init_cpus_allowed_list": self.process,
            "cgroup_cpuset_cpus_effective": self.cgroup,
            "validation_status": self.status,
        })
    }
}

#[derive(Debug)]
struct RunningContainer {
    docker: String,
    name: String,
    child: Option<Child>,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
    cleanup_complete: bool,
}

impl RunningContainer {
    fn spawn(
        config: &ProfileConfig,
        image_id: &str,
        output_dir: &Path,
        task_exchange_dir: &Path,
        harness: &HarnessFile,
        network_binding: Option<&DockerNetworkBinding>,
    ) -> Result<Self, ProfileError> {
        let random = random_bytes::<6>()?;
        let random_hex = random.iter().fold(
            String::with_capacity(random.len() * 2),
            |mut output, byte| {
                write!(&mut output, "{byte:02x}").expect("writing to a String cannot fail");
                output
            },
        );
        let name = format!("tasktrace-profile-{random_hex}");
        let mut command = Command::new(&config.docker);
        command.args(["run", "--rm", "--name", &name]);
        if config.effective_strace_mode() != StraceMode::None {
            command.args(["--cap-add", "SYS_PTRACE"]);
            command.args(["--security-opt", "seccomp=unconfined"]);
        }
        if let Some(cpuset) = &config.cpuset_cpus {
            command.args(["--cpuset-cpus", cpuset]);
        }
        if let Some(limit) = config.cpu_limit {
            command.args(["--cpus", &limit.to_string()]);
        }
        if let Some(limit) = config.memory_limit_bytes {
            command.args(["--memory", &limit.to_string()]);
        }
        if let Some(limit) = config.memory_swap_total_limit_bytes {
            command.args(["--memory-swap", &limit.to_string()]);
        }
        if let Some(workdir) = &config.workdir {
            command.args(["--workdir", workdir]);
        }
        if let Some(user) = &config.user {
            command.args(["--user", user]);
        }
        let network = match config.network_policy.effective_mode() {
            NetworkMode::NoNetwork => Some("none"),
            NetworkMode::Public => config.network.as_deref(),
            NetworkMode::Allowlist => network_binding
                .and_then(|binding| binding.docker_network.as_deref())
                .ok_or_else(|| {
                    ProfileError::new(
                        "allowlist policy was not prepared by the native Docker enforcement backend",
                    )
                })
                .map(Some)?,
        };
        if let Some(network) = network {
            command.args(["--network", network]);
        }
        if config.network_policy.effective_mode() == NetworkMode::Allowlist {
            command.args(["--cap-drop", "NET_RAW"]);
        }
        for mount in &config.mounts {
            command.args(["--mount", mount]);
        }
        command.arg("--mount").arg(format!(
            "type=bind,src={},dst=/tasktrace-profile-entrypoint.sh,readonly",
            harness.path.display()
        ));
        command.arg("--mount").arg(format!(
            "type=bind,src={},dst=/tasktrace-profile-output,readonly",
            output_dir.display()
        ));
        command.arg("--mount").arg(format!(
            "type=bind,src={},dst=/tasktrace-profile-exchange",
            task_exchange_dir.display()
        ));
        let controls = [
            (
                "TASKTRACE_PROFILE_DIR",
                "/tasktrace-profile-exchange".to_owned(),
            ),
            ("TASKTRACE_PROFILE_COMMAND", config.command.clone()),
            (
                "TASKTRACE_STRACE_MODE",
                config.effective_strace_mode().as_str().to_owned(),
            ),
            (
                "TASKTRACE_STRACE_STRING_LIMIT",
                config.string_limit.to_string(),
            ),
            ("TASKTRACE_PROFILE_SHELL", config.shell.clone()),
            (
                "TASKTRACE_RESOURCE_GATE_DIR",
                "/tasktrace-profile-exchange".to_owned(),
            ),
            (
                "TASKTRACE_SCHEDULER_GATE_DIR",
                "/tasktrace-profile-exchange".to_owned(),
            ),
        ];
        for (name, value) in controls {
            command.args(["--env", &format!("{name}={value}")]);
        }
        if let Some(prepare) = &config.prepare_command {
            command.args([
                "--env",
                &format!("TASKTRACE_PROFILE_PREPARE_COMMAND={prepare}"),
            ]);
        }
        if let Some(verification) = &config.verification_command {
            command.args([
                "--env",
                &format!("TASKTRACE_PROFILE_VERIFICATION_COMMAND={verification}"),
            ]);
        }
        for assignment in &config.environment {
            command.args(["--env", assignment]);
        }
        command
            .args(["--entrypoint", "/bin/sh"])
            .arg(image_id)
            .arg("/tasktrace-profile-entrypoint.sh");

        let stdout = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&harness.docker_stdout)?;
        let stderr = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(&harness.docker_stderr)?;
        let child = command
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr))
            .spawn()
            .map_err(|error| ProfileError::new(format!("unable to launch Docker: {error}")))?;
        Ok(Self {
            docker: config.docker.clone(),
            name,
            child: Some(child),
            stdout_path: harness.docker_stdout.clone(),
            stderr_path: harness.docker_stderr.clone(),
            cleanup_complete: false,
        })
    }

    fn status(&mut self) -> Result<Option<ExitStatus>, ProfileError> {
        self.child
            .as_mut()
            .ok_or_else(|| ProfileError::new("Docker child was already reaped"))?
            .try_wait()
            .map_err(Into::into)
    }

    fn diagnostic(&self) -> String {
        let stderr = fs::read_to_string(&self.stderr_path).unwrap_or_default();
        let stdout = fs::read_to_string(&self.stdout_path).unwrap_or_default();
        truncate(
            if stderr.trim().is_empty() {
                stdout.trim()
            } else {
                stderr.trim()
            },
            1024,
        )
    }

    fn wait(&mut self) -> Result<ExitStatus, ProfileError> {
        self.child
            .take()
            .ok_or_else(|| ProfileError::new("Docker child was already reaped"))?
            .wait()
            .map_err(Into::into)
    }

    fn force_remove(&mut self) -> Result<(), ProfileError> {
        if self.cleanup_complete {
            return Ok(());
        }
        let output = command_output(
            &self.docker,
            &[OsStr::new("rm"), OsStr::new("-f"), OsStr::new(&self.name)],
        )?;
        let diagnostic = String::from_utf8_lossy(if output.stderr.is_empty() {
            &output.stdout
        } else {
            &output.stderr
        });
        if !output.status.success() && !diagnostic.contains("No such container") {
            return Err(ProfileError::new(format!(
                "could not remove profiler container {:?}: {}",
                self.name,
                diagnostic.trim()
            )));
        }
        if let Some(mut child) = self.child.take() {
            let deadline = monotonic_ns().saturating_add(CLEANUP_GRACE.as_nanos() as u64);
            loop {
                if child.try_wait()?.is_some() {
                    break;
                }
                if monotonic_ns() >= deadline {
                    child.kill()?;
                    let _ = child.wait();
                    break;
                }
                thread::sleep(ATTACH_POLL);
            }
        }
        let inspected = command_output(
            &self.docker,
            &[
                OsStr::new("container"),
                OsStr::new("inspect"),
                OsStr::new(&self.name),
            ],
        )?;
        if inspected.status.success() {
            return Err(ProfileError::new(format!(
                "profiler container survived cleanup: {}",
                self.name
            )));
        }
        if !missing_container_diagnostic(&inspected) {
            return Err(ProfileError::new(format!(
                "could not verify profiler container removal: {}",
                output_diagnostic(&inspected)
            )));
        }
        self.cleanup_complete = true;
        Ok(())
    }

    fn verify_removed(&mut self) -> Result<(), ProfileError> {
        let inspected = command_output(
            &self.docker,
            &[
                OsStr::new("container"),
                OsStr::new("inspect"),
                OsStr::new(&self.name),
            ],
        )?;
        if inspected.status.success() {
            self.force_remove()?;
            return Err(ProfileError::new(
                "Docker --rm did not remove the completed profiler container",
            ));
        }
        if !missing_container_diagnostic(&inspected) {
            return Err(ProfileError::new(format!(
                "could not verify Docker --rm cleanup: {}",
                output_diagnostic(&inspected)
            )));
        }
        self.cleanup_complete = true;
        Ok(())
    }
}

fn output_diagnostic(output: &std::process::Output) -> String {
    let bytes = if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    };
    truncate(String::from_utf8_lossy(bytes).trim(), 1024)
}

fn missing_container_diagnostic(output: &std::process::Output) -> bool {
    let diagnostic = output_diagnostic(output);
    diagnostic.contains("No such container") || diagnostic.contains("No such object")
}

impl Drop for RunningContainer {
    fn drop(&mut self) {
        if !self.cleanup_complete {
            let _ = self.force_remove();
        }
    }
}

fn wait_for_marker(
    exchange: &TaskExchange,
    name: &str,
    container: &mut RunningContainer,
    timeout: Option<Duration>,
    description: &str,
    cancel_requested: Option<&dyn Fn() -> bool>,
    mut observer: Option<&mut dyn FnMut() -> Result<(), ProfileError>>,
) -> Result<(), ProfileError> {
    let deadline =
        timeout.map(|duration| monotonic_ns().saturating_add(duration.as_nanos() as u64));
    loop {
        if let Some(observer) = observer.as_mut() {
            observer()?;
        }
        if exchange.regular_ready(name)? {
            return Ok(());
        }
        if cancellation_requested(cancel_requested) {
            return Err(ProfileError::new(format!(
                "interrupted while waiting for {description}"
            )));
        }
        if let Some(status) = container.status()? {
            return Err(ProfileError::new(format!(
                "container exited before {description} (status {}): {}",
                status.code().unwrap_or(-1),
                container.diagnostic()
            )));
        }
        if deadline.is_some_and(|deadline| monotonic_ns() >= deadline) {
            return Err(ProfileError::new(format!(
                "timed out waiting for {description}: {}",
                exchange.path().join(name).display()
            )));
        }
        thread::sleep(ATTACH_POLL);
    }
}

fn release_fifo(
    exchange: &TaskExchange,
    name: &str,
    container: &mut RunningContainer,
    timeout: Duration,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<(), ProfileError> {
    let deadline =
        monotonic_ns().saturating_add(timeout.min(Duration::from_secs(1)).as_nanos() as u64);
    loop {
        if cancellation_requested(cancel_requested) {
            return Err(ProfileError::new(
                "interrupted while releasing coordinator gate",
            ));
        }
        if let Some(status) = container.status()? {
            return Err(ProfileError::new(format!(
                "container exited before coordinator gate release (status {}): {}",
                status.code().unwrap_or(-1),
                container.diagnostic()
            )));
        }
        match exchange.write_fifo(name, b"continue\n") {
            Ok(true) => return Ok(()),
            Ok(false) => {}
            Err(error) => return Err(error),
        }
        if monotonic_ns() >= deadline {
            return Err(ProfileError::new(format!(
                "timed out waiting for coordinator FIFO reader: {}",
                exchange.path().join(name).display()
            )));
        }
        thread::sleep(ATTACH_POLL);
    }
}

fn inspect_container(
    config: &ProfileConfig,
    container: &RunningContainer,
) -> Result<(u32, String, Option<i64>), ProfileError> {
    let output = command_output(
        &config.docker,
        &[OsStr::new("inspect"), OsStr::new(&container.name)],
    )?;
    if !output.status.success() {
        return Err(ProfileError::new(format!(
            "unable to inspect running profiler container: {}",
            container.diagnostic()
        )));
    }
    let payload: Value = serde_json::from_slice(&output.stdout).map_err(|_| {
        ProfileError::new("Docker returned malformed container inspection evidence")
    })?;
    let row = payload
        .as_array()
        .filter(|rows| rows.len() == 1)
        .and_then(|rows| rows[0].as_object())
        .ok_or_else(|| {
            ProfileError::new("Docker returned malformed container inspection evidence")
        })?;
    let pid = row
        .get("State")
        .and_then(Value::as_object)
        .and_then(|state| state.get("Pid"))
        .and_then(Value::as_u64)
        .and_then(|value| u32::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| ProfileError::new("profiler container is not running"))?;
    let host = row
        .get("HostConfig")
        .and_then(Value::as_object)
        .ok_or_else(|| ProfileError::new("Docker inspection omitted HostConfig"))?;
    let cpuset = host
        .get("CpusetCpus")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    let nano_cpus = host.get("NanoCpus").and_then(Value::as_i64);
    Ok((pid, cpuset, nano_cpus))
}

fn read_cgroup_limit(
    path: &Path,
    filename: &str,
) -> Result<(Option<u64>, &'static str), ProfileError> {
    let text = fs::read_to_string(path.join(filename)).map_err(|error| {
        ProfileError::new(format!("unable to read measured {filename}: {error}"))
    })?;
    let raw = text.trim();
    if raw == "max" {
        return Ok((None, "unlimited"));
    }
    let value = raw
        .parse::<u64>()
        .map_err(|_| ProfileError::new(format!("invalid measured {filename} value: {raw:?}")))?;
    Ok((Some(value), "finite"))
}

fn inspect_memory_enforcement(
    config: &ProfileConfig,
    container: &RunningContainer,
    cgroup: &Path,
) -> Result<Value, ProfileError> {
    let output = command_output(
        &config.docker,
        &[OsStr::new("inspect"), OsStr::new(&container.name)],
    )?;
    if !output.status.success() {
        return Err(ProfileError::new(
            "unable to inspect Docker memory enforcement",
        ));
    }
    let payload: Value = serde_json::from_slice(&output.stdout)
        .map_err(|_| ProfileError::new("Docker returned malformed memory enforcement evidence"))?;
    let host = payload
        .as_array()
        .filter(|rows| rows.len() == 1)
        .and_then(|rows| rows[0].get("HostConfig"))
        .and_then(Value::as_object)
        .ok_or_else(|| ProfileError::new("Docker memory inspection omitted HostConfig"))?;
    let docker_memory = host
        .get("Memory")
        .and_then(Value::as_i64)
        .ok_or_else(|| ProfileError::new("Docker inspection omitted HostConfig.Memory"))?;
    let docker_swap = host
        .get("MemorySwap")
        .and_then(Value::as_i64)
        .ok_or_else(|| ProfileError::new("Docker inspection omitted HostConfig.MemorySwap"))?;
    let expected_memory = config.memory_limit_bytes.unwrap_or(0);
    if docker_memory < 0 || docker_memory as u64 != expected_memory {
        return Err(ProfileError::new(format!(
            "Docker did not retain requested memory limit: requested={expected_memory}, observed={docker_memory}"
        )));
    }
    if let Some(expected_swap) = config.memory_swap_total_limit_bytes {
        if docker_swap < 0 || docker_swap as u64 != expected_swap {
            return Err(ProfileError::new(format!(
                "Docker did not retain requested memory-swap total: requested={expected_swap}, observed={docker_swap}"
            )));
        }
    }
    let (memory_max, memory_state) = read_cgroup_limit(cgroup, "memory.max")?;
    let (swap_max, swap_state) = read_cgroup_limit(cgroup, "memory.swap.max")?;
    match config.memory_limit_bytes {
        Some(expected) if memory_max != Some(expected) => {
            return Err(ProfileError::new(format!(
                "measured cgroup memory.max differs from requested Docker limit: requested={expected}, observed={memory_max:?}"
            )));
        }
        None if memory_max.is_some() => {
            return Err(ProfileError::new(format!(
                "measured cgroup unexpectedly has a finite memory.max: {memory_max:?}"
            )));
        }
        _ => {}
    }
    let expected_swap_max = config
        .memory_swap_total_limit_bytes
        .zip(config.memory_limit_bytes)
        .map(|(total, memory)| total - memory);
    if let Some(expected) = expected_swap_max {
        if swap_max != Some(expected) {
            return Err(ProfileError::new(format!(
                "measured cgroup memory.swap.max differs from requested total-minus-memory limit: requested={expected}, observed={swap_max:?}"
            )));
        }
    }
    Ok(json!({
        "validation_status": "passed",
        "requested_memory_limit_bytes": config.memory_limit_bytes,
        "requested_memory_swap_total_limit_bytes": config.memory_swap_total_limit_bytes,
        "docker_host_config_memory_bytes": docker_memory,
        "docker_host_config_memory_swap_total_bytes": docker_swap,
        "cgroup_memory_max_bytes": memory_max,
        "cgroup_memory_max_state": memory_state,
        "cgroup_memory_swap_max_bytes": swap_max,
        "cgroup_memory_swap_max_state": swap_state,
        "configured_swap_policy": if config.memory_swap_total_limit_bytes == config.memory_limit_bytes && config.memory_limit_bytes.is_some() { "disabled" } else if config.memory_swap_total_limit_bytes.is_some() { "finite-enabled" } else { "runtime-default" },
    }))
}

fn cgroup_for_pid(pid: u32) -> Result<PathBuf, ProfileError> {
    let membership = fs::read_to_string(format!("/proc/{pid}/cgroup"))?;
    let relative = membership
        .lines()
        .find_map(|line| line.strip_prefix("0::"))
        .ok_or_else(|| {
            ProfileError::new(format!(
                "cgroup v2 membership not found for container PID {pid}"
            ))
        })?;
    let relative = Path::new(relative.trim_start_matches('/'));
    if relative
        .components()
        .any(|component| !matches!(component, Component::Normal(_)))
    {
        return Err(ProfileError::new(
            "container cgroup membership contains unsafe path components",
        ));
    }
    let joined = normalize_components(&Path::new(CGROUP_ROOT).join(relative))?;
    let canonical = fs::canonicalize(joined)?;
    let root = fs::canonicalize(CGROUP_ROOT)?;
    if canonical == root || !canonical.starts_with(&root) {
        return Err(ProfileError::new(format!(
            "container PID did not resolve to a disposable leaf cgroup: {}",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn status_cpuset(pid: u32) -> Result<String, ProfileError> {
    fs::read_to_string(format!("/proc/{pid}/status"))?
        .lines()
        .find_map(|line| line.strip_prefix("Cpus_allowed_list:").map(str::trim))
        .map(ToOwned::to_owned)
        .ok_or_else(|| ProfileError::new("container init status omitted Cpus_allowed_list"))
}

fn validate_cpu_placement(
    config: &ProfileConfig,
    pid: u32,
    cgroup: &Path,
    docker_cpuset: &str,
    nano_cpus: Option<i64>,
) -> Result<CpuPlacement, ProfileError> {
    if let Some(expected_limit) = config.cpu_limit {
        let observed = nano_cpus
            .ok_or_else(|| ProfileError::new("Docker inspection omitted HostConfig.NanoCpus"))?;
        let expected = expected_limit * 1_000_000_000.0;
        if observed <= 0 || (observed as f64 - expected).abs() > 0.5 {
            return Err(ProfileError::new(format!(
                "Docker did not retain the requested CPU quota: requested={expected_limit}, NanoCpus={observed}"
            )));
        }
    }
    let Some(requested_text) = &config.cpuset_cpus else {
        return Ok(CpuPlacement {
            requested: None,
            docker: None,
            process: None,
            cgroup: None,
            status: "not-requested",
        });
    };
    let requested = parse_cpuset(requested_text)?;
    if docker_cpuset.is_empty() {
        return Err(ProfileError::new(
            "Docker did not retain a non-empty HostConfig.CpusetCpus value",
        ));
    }
    let process = status_cpuset(pid)?;
    let effective = fs::read_to_string(cgroup.join("cpuset.cpus.effective"))?
        .trim()
        .to_owned();
    for (source, value) in [
        ("Docker HostConfig.CpusetCpus", docker_cpuset),
        ("container init Cpus_allowed_list", process.as_str()),
        ("measured cgroup cpuset.cpus.effective", effective.as_str()),
    ] {
        let observed = parse_cpuset(value).map_err(|error| {
            ProfileError::new(format!("{source} is invalid: {value:?}: {error}"))
        })?;
        if observed != requested {
            return Err(ProfileError::new(format!(
                "{source} differs from the requested cpuset: requested={requested_text:?}, observed={value:?}"
            )));
        }
    }
    Ok(CpuPlacement {
        requested: Some(requested_text.clone()),
        docker: Some(docker_cpuset.to_owned()),
        process: Some(process),
        cgroup: Some(effective),
        status: "passed",
    })
}

fn read_cgroup_pids(path: &Path, output: &mut BTreeSet<u32>) -> Result<(), ProfileError> {
    let text = fs::read_to_string(path.join("cgroup.procs")).map_err(|error| {
        ProfileError::new(format!(
            "unable to inspect measured cgroup process scope {}: {error}",
            path.display()
        ))
    })?;
    for value in text.split_whitespace() {
        let pid = value
            .parse::<u32>()
            .map_err(|_| ProfileError::new(format!("invalid PID in measured cgroup: {value:?}")))?;
        output.insert(pid);
    }
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let file_type = entry.file_type()?;
        if file_type.is_dir() {
            read_cgroup_pids(&entry.path(), output)?;
        }
    }
    Ok(())
}

fn cgroup_descendants(path: &Path, root_pid: u32) -> Result<Vec<u32>, ProfileError> {
    let mut pids = BTreeSet::new();
    read_cgroup_pids(path, &mut pids)?;
    pids.remove(&root_pid);
    Ok(pids.into_iter().collect())
}

fn process_start_ticks(pid: u32) -> Option<u64> {
    let text = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let after = text.rsplit_once(')')?.1;
    // The state is field 3; starttime is field 22, therefore index 19 in the
    // whitespace fields after the closing command parenthesis.
    after.split_whitespace().nth(19)?.parse().ok()
}

fn process_snapshot(pid: u32) -> Value {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok();
    let fields: Vec<_> = stat
        .as_deref()
        .and_then(|text| text.rsplit_once(')'))
        .map(|(_, after)| after.split_whitespace().collect())
        .unwrap_or_default();
    let read = |name: &str| {
        fs::read_to_string(format!("/proc/{pid}/{name}"))
            .ok()
            .map(|value| value.trim().to_owned())
    };
    let cmdline = fs::read(format!("/proc/{pid}/cmdline")).ok().map(|bytes| {
        String::from_utf8_lossy(&bytes)
            .replace('\0', " ")
            .trim()
            .to_owned()
    });
    json!({
        "pid": pid,
        "state": fields.first().copied(),
        "ppid": fields.get(1).and_then(|value| value.parse::<u32>().ok()),
        "start_time_ticks": process_start_ticks(pid),
        "comm": read("comm"),
        "wchan": read("wchan"),
        "cmdline": cmdline,
        "exe": fs::read_link(format!("/proc/{pid}/exe")).ok().map(|path| path.to_string_lossy().into_owned()),
    })
}

fn signal_scoped_descendants(
    cgroup: &Path,
    root_pid: u32,
    signal: libc::c_int,
) -> Result<Vec<u32>, ProfileError> {
    let candidates: BTreeMap<_, _> = cgroup_descendants(cgroup, root_pid)?
        .into_iter()
        .map(|pid| (pid, process_start_ticks(pid)))
        .collect();
    let mut signaled = Vec::new();
    for (pid, start_ticks) in candidates {
        if start_ticks.is_none()
            || process_start_ticks(pid) != start_ticks
            || !cgroup_descendants(cgroup, root_pid)?.contains(&pid)
        {
            continue;
        }
        let result = unsafe { libc::kill(pid as libc::pid_t, signal) };
        if result == 0 {
            signaled.push(pid);
        } else if !matches!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH)) {
            return Err(ProfileError::new(format!(
                "unable to signal measured descendant PID {pid}: {}",
                io::Error::last_os_error()
            )));
        }
    }
    Ok(signaled)
}

#[derive(Clone, Debug)]
struct DescendantDrain {
    initial: Vec<Value>,
    remaining: Vec<Value>,
    signaled_term: Vec<u32>,
    signaled_kill: Vec<u32>,
    waited_ns: u64,
}

fn wait_for_descendants(
    config: &ProfileConfig,
    cgroup: &Path,
    root_pid: u32,
    cancel_requested: Option<&dyn Fn() -> bool>,
    mut observer: Option<&mut dyn FnMut() -> Result<(), ProfileError>>,
) -> Result<DescendantDrain, ProfileError> {
    let started = monotonic_ns();
    let initial_pids = cgroup_descendants(cgroup, root_pid)?;
    let initial = initial_pids
        .iter()
        .map(|pid| process_snapshot(*pid))
        .collect();
    let mut signaled_term = Vec::new();
    let mut signaled_kill = Vec::new();
    if config.terminate_descendants_at_finish && !initial_pids.is_empty() {
        signaled_term = signal_scoped_descendants(cgroup, root_pid, libc::SIGTERM)?;
    }
    let first_deadline = started.saturating_add(DESCENDANT_GRACE.as_nanos() as u64);
    loop {
        if let Some(observer) = observer.as_mut() {
            observer()?;
        }
        let pids = cgroup_descendants(cgroup, root_pid)?;
        if pids.is_empty() {
            return Ok(DescendantDrain {
                initial,
                remaining: Vec::new(),
                signaled_term,
                signaled_kill,
                waited_ns: monotonic_ns().saturating_sub(started),
            });
        }
        if cancellation_requested(cancel_requested) {
            return Err(ProfileError::new("interrupted during descendant drain"));
        }
        if monotonic_ns() >= first_deadline {
            break;
        }
        thread::sleep(ATTACH_POLL);
    }
    if config.terminate_descendants_at_finish {
        signaled_kill = signal_scoped_descendants(cgroup, root_pid, libc::SIGKILL)?;
        let deadline = monotonic_ns().saturating_add(DESCENDANT_GRACE.as_nanos() as u64);
        loop {
            if let Some(observer) = observer.as_mut() {
                observer()?;
            }
            let pids = cgroup_descendants(cgroup, root_pid)?;
            if pids.is_empty() || monotonic_ns() >= deadline {
                break;
            }
            if cancellation_requested(cancel_requested) {
                return Err(ProfileError::new("interrupted during descendant cleanup"));
            }
            thread::sleep(ATTACH_POLL);
        }
    }
    let remaining_pids = cgroup_descendants(cgroup, root_pid)?;
    Ok(DescendantDrain {
        initial,
        remaining: remaining_pids
            .iter()
            .map(|pid| process_snapshot(*pid))
            .collect(),
        signaled_term,
        signaled_kill,
        waited_ns: monotonic_ns().saturating_sub(started),
    })
}

fn atomic_write(path: &Path, bytes: &[u8]) -> Result<(), ProfileError> {
    static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| ProfileError::new(format!("invalid output filename: {}", path.display())))?;
    let temporary = path.with_file_name(format!(
        ".{name}.tmp.{}.{}",
        std::process::id(),
        WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true).mode(0o600);
    let mut file = options.open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        ProfileError::new(format!("unable to publish {}: {error}", path.display()))
    })
}

fn sorted_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sorted_json).collect()),
        Value::Object(values) => {
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            let mut output = Map::new();
            for key in keys {
                output.insert(key.clone(), sorted_json(&values[key]));
            }
            Value::Object(output)
        }
        value => value.clone(),
    }
}

fn write_json(path: &Path, value: &Value) -> Result<(), ProfileError> {
    let mut encoded = serde_json::to_vec_pretty(&sorted_json(value))?;
    encoded.push(b'\n');
    atomic_write(path, &encoded)
}

fn round(value: f64, digits: i32) -> f64 {
    let scale = 10_f64.powi(digits);
    (value * scale).round() / scale
}

fn number(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn canonical_bytes(value: &Value) -> Result<Vec<u8>, ProfileError> {
    serde_json::to_vec(&sorted_json(value)).map_err(Into::into)
}

fn canonical_digest(value: &Value) -> Result<String, ProfileError> {
    Ok(sha256_bytes(&canonical_bytes(value)?))
}

fn raw_read(
    snapshot: &CpuSourceSnapshot,
    phase: &str,
    source_scope: &str,
    filename: &str,
    raw_text: Option<&str>,
    index: usize,
) -> Value {
    json!({
        "read_index": index,
        "phase": phase,
        "source_scope": source_scope,
        "source_path": snapshot.relative_path,
        "filename": filename,
        "acquisition_start_ns": snapshot.acquisition_start_ns,
        "acquisition_end_ns": snapshot.acquisition_end_ns,
        "available": raw_text.is_some(),
        "raw_text": raw_text,
        "error_type": if raw_text.is_some() { Value::Null } else { Value::String("ReadError".to_owned()) },
    })
}

fn write_cpu_source_artifacts(
    output_dir: &Path,
    leaf_before: &CpuSourceSnapshot,
    leaf_after: &CpuSourceSnapshot,
    ancestors_before: &[CpuSourceSnapshot],
    ancestors_after: &[CpuSourceSnapshot],
) -> Result<Value, ProfileError> {
    atomic_write(
        &output_dir.join("cgroup_cpu.stat.observer_start.txt"),
        leaf_before.cpu_stat_text.as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("cgroup_cpu.stat.controlled_finish_gate.txt"),
        leaf_after.cpu_stat_text.as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("cgroup_cpu.max.observer_start.txt"),
        leaf_before.cpu_max_text.as_deref().unwrap_or("").as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("cgroup_cpu.max.controlled_finish_gate.txt"),
        leaf_after.cpu_max_text.as_deref().unwrap_or("").as_bytes(),
    )?;
    // Preserve the historical direct aliases using the exact host-consumed
    // leaf sources and canonical monotonic boundaries.
    atomic_write(
        &output_dir.join("cpu.stat.before"),
        leaf_before.cpu_stat_text.as_bytes(),
    )?;
    atomic_write(
        &output_dir.join("cpu.stat.after"),
        leaf_after.cpu_stat_text.as_bytes(),
    )?;

    let leaf_reads = vec![
        raw_read(
            leaf_before,
            "observer_start",
            "leaf",
            "cpu.max",
            leaf_before.cpu_max_text.as_deref(),
            0,
        ),
        raw_read(
            leaf_before,
            "observer_start",
            "leaf",
            "cpu.stat",
            Some(&leaf_before.cpu_stat_text),
            1,
        ),
        raw_read(
            leaf_after,
            "controlled_finish_gate",
            "leaf",
            "cpu.max",
            leaf_after.cpu_max_text.as_deref(),
            2,
        ),
        raw_read(
            leaf_after,
            "controlled_finish_gate",
            "leaf",
            "cpu.stat",
            Some(&leaf_after.cpu_stat_text),
            3,
        ),
    ];
    let mut ancestor_reads = Vec::new();
    for (phase, snapshots) in [
        ("observer_start", ancestors_before),
        ("controlled_finish_gate", ancestors_after),
    ] {
        for snapshot in snapshots {
            ancestor_reads.push(raw_read(
                snapshot,
                phase,
                "ancestor",
                "cpu.max",
                snapshot.cpu_max_text.as_deref(),
                ancestor_reads.len(),
            ));
            ancestor_reads.push(raw_read(
                snapshot,
                phase,
                "ancestor",
                "cpu.stat",
                Some(&snapshot.cpu_stat_text),
                ancestor_reads.len(),
            ));
        }
    }
    let payload = json!({
        "schema_version": CPU_SOURCE_SCHEMA_VERSION,
        "scope": {
            "kind": "cgroup-v2-leaf-and-ancestor-chain",
            "path_representation": "relative-to-configured-cgroup-root",
        },
        "phases": ["observer_start", "controlled_finish_gate"],
        "leaf": {
            "reads": leaf_reads,
            "meaning": "Exact leaf cpu.max and cpu.stat text consumed by the CPU performance validity gate.",
        },
        "ancestors": {
            "reads": ancestor_reads,
            "meaning": "Exact ancestor cpu.max and cpu.stat text consumed by the CPU performance validity gate.",
        },
        "meaning": "Exact leaf and ancestor CPU controller source text retained for the shared baseline/profile CPU performance validity gate.",
    });
    write_json(
        &output_dir.join("cgroup_cpu_source_snapshots.json"),
        &payload,
    )?;
    Ok(payload)
}

// The arguments keep the two measurement boundaries and their ancestor
// snapshots explicit; grouping them would obscure the validation contract.
#[allow(clippy::too_many_arguments)]
fn cpu_report(
    config: &ProfileConfig,
    scope_id: &str,
    window_start_ns: u64,
    window_end_ns: u64,
    leaf_before: &CpuSourceSnapshot,
    leaf_after: &CpuSourceSnapshot,
    ancestors_before: &[CpuSourceSnapshot],
    ancestors_after: &[CpuSourceSnapshot],
    drain: &DescendantDrain,
) -> Result<Value, ProfileError> {
    if window_end_ns <= window_start_ns {
        return Err(ProfileError::new(
            "measured monotonic window is non-increasing",
        ));
    }
    if ancestors_before.len() != ancestors_after.len()
        || ancestors_before
            .iter()
            .zip(ancestors_after)
            .any(|(before, after)| before.path != after.path)
    {
        return Err(ProfileError::new(
            "measured cgroup ancestor chain changed during execution",
        ));
    }
    let leaf = source_pair_report(leaf_before, leaf_after, true);
    let ancestors: Vec<_> = ancestors_before
        .iter()
        .zip(ancestors_after)
        .map(|(before, after)| source_pair_report(before, after, false))
        .collect();
    let mut invalid: Vec<String> = leaf["invalid_reasons"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect();
    for ancestor in &ancestors {
        invalid.extend(
            ancestor["invalid_reasons"]
                .as_array()
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .map(ToOwned::to_owned),
        );
    }
    if !drain.remaining.is_empty() {
        invalid.push(format!(
            "descendant processes remained at the controlled finish gate: {}",
            drain
                .remaining
                .iter()
                .filter_map(|value| value["pid"].as_u64())
                .map(|pid| pid.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }
    deduplicate(&mut invalid);
    let delta = leaf["cpu_stat_delta"]
        .as_object()
        .ok_or_else(|| ProfileError::new("leaf CPU delta is unavailable"))?;
    let counter = |key: &str| -> Result<u64, ProfileError> {
        delta
            .get(key)
            .and_then(Value::as_u64)
            .ok_or_else(|| ProfileError::new(format!("leaf CPU delta omitted {key}")))
    };
    let usage_sec = counter("usage_usec")? as f64 / 1_000_000.0;
    let user_sec = counter("user_usec")? as f64 / 1_000_000.0;
    let system_sec = counter("system_usec")? as f64 / 1_000_000.0;
    let wall_sec = (window_end_ns - window_start_ns) as f64 / 1_000_000_000.0;
    let average = usage_sec / wall_sec;
    let capacity = config.allocated_capacity();
    let utilization = capacity.map(|capacity| average / capacity);
    let raw_read_count = 4 + 4 * ancestors.len();
    let ancestor_invalid: Vec<_> = ancestors
        .iter()
        .flat_map(|value| {
            value["invalid_reasons"]
                .as_array()
                .into_iter()
                .flatten()
                .cloned()
        })
        .collect();
    Ok(json!({
        "schema_version": CPU_SCHEMA_VERSION,
        "method": "container-cgroup-v2-cpu.stat",
        "scope_id": scope_id,
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "wall_sec": wall_sec,
        "usage_sec": usage_sec,
        "user_sec": user_sec,
        "system_sec": system_sec,
        "average_logical_cores": average,
        "average_logical_cores_used": round(average, 9),
        "allocated_logical_cpus": capacity,
        "allocated_cpu_source": if config.cpuset_cpus.is_some() { Value::String("cpuset-cpus".to_owned()) } else if capacity.is_some() { Value::String("user-supplied".to_owned()) } else { Value::Null },
        "enforced_logical_cpu_limit": config.cpu_limit,
        "enforced_cpu_limit_source": if config.cpu_limit.is_some() { Value::String("docker --cpus".to_owned()) } else { Value::Null },
        "utilization_fraction_of_allocation": utilization.map(|value| round(value, 9)),
        "utilization_percent_of_allocation": utilization.map(|value| round(100.0 * value, 6)),
        "cpu_stat_delta": leaf["cpu_stat_delta"].clone(),
        "leaf_cpu": leaf,
        "ancestor_cpu": {
            "ancestor_count": ancestors.len(),
            "ancestors": ancestors,
            "cgroup_root": CGROUP_ROOT,
            "cgroup_root_cpu_max_applicable": false,
            "chain_order": "leaf-parent-to-root",
            "leaf_path": leaf_before.relative_path,
            "evidence_complete": ancestor_invalid.is_empty(),
            "limits_stable": ancestor_invalid.iter().all(|value| !value.as_str().is_some_and(|reason| reason.contains("changed"))),
            "counter_regression_observed": ancestor_invalid.iter().any(|value| value.as_str().is_some_and(|reason| reason.contains("backwards"))),
            "finite_limits_present": ancestor_invalid.iter().any(|value| value.as_str().is_some_and(|reason| reason.contains("finite cpu.max"))),
            "throttling_observed": ancestor_invalid.iter().any(|value| value.as_str().is_some_and(|reason| reason.contains("throttling"))),
            "valid_for_cpu_performance": ancestor_invalid.is_empty(),
            "evidence_errors": [],
            "invalid_reasons": ancestor_invalid,
            "raw_source_read_count": raw_read_count.saturating_sub(4),
            "meaning": "Every ancestor cgroup is sampled at observer start and controlled finish. Missing or malformed sources, cpu.max changes, cpu.stat regressions, finite bandwidth limits, or positive throttling deltas fail closed for unconstrained CPU performance.",
        },
        "raw_source_snapshot_schema_version": CPU_SOURCE_SCHEMA_VERSION,
        "raw_source_snapshot_read_count": raw_read_count,
        "valid_for_cpu_performance": invalid.is_empty(),
        "invalid_reasons": invalid,
        "measurement_domain_contract": {
            "measurement_domains": ["cpu.capacity"],
            "perturbations": [],
            "status": "unperturbed",
        },
    }))
}

fn environment_provenance(assignments: &[String]) -> Value {
    let mut rows: Vec<_> = assignments
        .iter()
        .map(|assignment| {
            let (key, separator, value) = assignment
                .split_once('=')
                .map_or((assignment.as_str(), false, ""), |(key, value)| (key, true, value));
            json!({
                "key": key,
                "value_sha256": sha256_bytes([b"tasktrace-profile-environment-value-v1\0".as_slice(), value.as_bytes()].concat().as_slice()),
                "assignment_had_equals": separator,
            })
        })
        .collect();
    rows.sort_by(|left, right| {
        canonical_bytes(left)
            .unwrap()
            .cmp(&canonical_bytes(right).unwrap())
    });
    Value::Array(rows)
}

fn mount_provenance(mounts: &[String], controller_digest: &str) -> Result<Value, ProfileError> {
    let mut rows = Vec::new();
    for mount in mounts {
        let target = parse_mount_destination(mount).ok_or_else(|| {
            ProfileError::new(format!("invalid bind mount specification: {mount:?}"))
        })?;
        let source = parse_mount_source(mount).ok_or_else(|| {
            ProfileError::new(format!("invalid bind mount specification: {mount:?}"))
        })?;
        let source_identity = if target == "/tasktrace-controller/tasktrace" {
            let path = Path::new(source);
            if !path.is_file() {
                return Err(ProfileError::new(format!(
                    "Rust controller mount is not a file: {}",
                    path.display()
                )));
            }
            let actual = sha256_file(path)?;
            if actual != controller_digest {
                return Err(ProfileError::new(format!(
                    "Rust controller mount differs from the executing native controller: expected {controller_digest}, observed {actual}"
                )));
            }
            json!({
                "kind": "rust-controller-binary",
                "sha256": actual,
                "host_path_in_comparability": false,
            })
        } else {
            json!({
                "kind": "external-bind",
                "host_path_in_comparability": false,
            })
        };
        rows.push(json!({
            "kind": "bind",
            "target": target,
            "read_only": mount.split(',').any(|value| matches!(value.trim(), "readonly" | "ro")) || mount.split(':').nth(2).is_some_and(|value| value.split(',').any(|option| option == "ro")),
            "source_identity": source_identity,
            "options": {},
        }));
    }
    Ok(Value::Array(rows))
}

fn execution_plan(
    config: &ProfileConfig,
    image_identity: &Value,
    controller_digest: &str,
) -> Result<(Value, String), ProfileError> {
    let mut source = config
        .source_identity
        .clone()
        .unwrap_or_else(|| json!({"kind": "tasktrace-container-profile"}));
    let source_object = source.as_object_mut().expect("validated source identity");
    if let Some(expected) = source_object
        .get("rust_controller_binary_sha256")
        .and_then(Value::as_str)
    {
        if expected != controller_digest {
            return Err(ProfileError::new(format!(
                "adapter controller identity differs from the executing native controller: expected {expected}, observed {controller_digest}"
            )));
        }
    }
    source_object.insert(
        "rust_controller_binary_sha256".to_owned(),
        Value::String(controller_digest.to_owned()),
    );
    let workload = config.workload_identity.clone().unwrap_or_else(|| {
        json!({
            "kind": "container-command",
            "command_sha256": sha256_bytes(config.command.as_bytes()),
        })
    });
    let expected = config
        .expected_outcome
        .clone()
        .unwrap_or_else(|| json!({"kind": "observe-command-exit"}));
    let network = match config.network_policy.effective_mode() {
        NetworkMode::NoNetwork => Value::String("none".to_owned()),
        NetworkMode::Public => config
            .network
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
        NetworkMode::Allowlist => Value::String("tasktrace-allowlist".to_owned()),
    };
    let mut comparable_network_policy = config.network_policy.to_comparability_payload();
    if config.network_policy.effective_mode() == NetworkMode::Allowlist {
        let enforcement = load_network_enforcement_report(&config.output_dir)
            .map_err(|error| ProfileError::new(error.to_string()))?;
        let comparable = enforcement.get("comparability").cloned().ok_or_else(|| {
            ProfileError::new("active native allowlist enforcement omitted comparability evidence")
        })?;
        comparable_network_policy
            .as_object_mut()
            .expect("network policy comparability is an object")
            .insert("enforcement".to_owned(), comparable);
    }
    let image_reference = if config.runtime == "docker" {
        image_identity["id"].clone()
    } else {
        image_identity
            .get("sha256")
            .filter(|value| !value.is_null())
            .or_else(|| image_identity.get("container_name"))
            .cloned()
            .unwrap_or_else(|| Value::String(config.image.clone()))
    };
    let plan = json!({
        "schema_version": "tasktrace-execution-plan.v1",
        "workload_identity": workload,
        "source_identity": source,
        "command": [config.shell, "-c", config.command],
        "cwd": config.workdir.as_deref().unwrap_or("/"),
        "mounts": mount_provenance(&config.mounts, controller_digest)?,
        "image_reference": image_reference,
        "image_identity": image_identity,
        "runtime": config.runtime,
        "network": network,
        "environment": environment_provenance(&config.environment),
        "expected_outcome": expected,
        "extensions": {
            "network_policy": comparable_network_policy,
            "cpuset_cpus": config.cpuset_cpus,
            "cpuset_contract": if config.runtime == "enroot" { "requested-taskset-launch-affinity-not-cpu-containment" } else { "docker-cpuset-enforced-and-validated" },
            "enroot_executable": if config.runtime == "enroot" { json!(config.enroot) } else { Value::Null },
            "cpu_limit": config.cpu_limit,
            "memory_limit_bytes": config.memory_limit_bytes,
            "memory_swap_total_limit_bytes": config.memory_swap_total_limit_bytes,
            "prepare_command_sha256": config.prepare_command.as_ref().map(|value| sha256_bytes(value.as_bytes())),
            "verification_command_sha256": config.verification_command.as_ref().map(|value| sha256_bytes(value.as_bytes())),
            "user": config.user,
            "terminate_descendants_at_finish": config.terminate_descendants_at_finish,
            "scheduler_concurrency": config.scheduler_enabled(),
            "scheduler_backend": if config.scheduler_enabled() { json!(config.scheduler_backend) } else { Value::Null },
            "scheduler_bcc_buffer_pages": if config.scheduler_enabled() && config.scheduler_backend == "bcc" { json!(config.scheduler_bcc_buffer_pages) } else { Value::Null },
            "scheduler_perf_binary": if config.scheduler_enabled() && config.scheduler_backend == "perf" { json!(config.scheduler_perf_binary) } else { Value::Null },
            "scheduler_perf_mmap_pages": if config.scheduler_enabled() && config.scheduler_backend == "perf" { json!(config.scheduler_perf_mmap_pages) } else { Value::Null },
            "scheduler_cpu_tolerance_fraction": config.scheduler_cpu_tolerance_fraction,
            "scheduler_cpu_tolerance_sec": config.scheduler_cpu_tolerance_sec,
            "hardware_perf_binary": if config.hardware_enabled() { json!(config.hardware_perf_binary) } else { Value::Null },
            "hardware_perf_events": if config.hardware_enabled() { json!(config.hardware_perf_events) } else { Value::Null },
            "hardware_perf_attach_timeout_sec": if config.hardware_enabled() { json!(config.hardware_perf_attach_timeout.as_secs_f64()) } else { Value::Null },
            "hardware_perf_minimum_running_ratio": if config.hardware_enabled() { json!(config.hardware_perf_minimum_running_ratio) } else { Value::Null },
            "hardware_perf_hybrid_merge": config.hardware_enabled(),
            "hardware_perf_cpuset_cpus": if config.hardware_enabled() { json!(config.cpuset_cpus) } else { Value::Null },
            "proc_stat_comparison": config.proc_stat_comparison,
            "proc_stat_interval_sec": config.proc_stat_interval_sec,
            "proc_stat_baseline_sec": config.proc_stat_baseline_sec,
            "memory_profile": config.memory_enabled(),
            "memory_sample_interval_sec": config.memory_sample_interval_sec,
            "run_wave_start_gate": config.run_wave_start_gate,
        },
    });
    let digest = canonical_digest(&plan)?;
    Ok((plan, digest))
}

fn marker(name: &str, timestamp_ns: u64) -> Value {
    json!({
        "name": name,
        "canonical_timestamp_ns": timestamp_ns,
        "alignment_error_ns": 0,
        "provenance": {},
    })
}

fn artifact(
    output_dir: &Path,
    name: &str,
    id: &str,
    media_type: &str,
) -> Result<Value, ProfileError> {
    let path = output_dir.join(name);
    let metadata = path.metadata()?;
    Ok(json!({
        "artifact_id": id,
        "relative_path": name,
        "media_type": media_type,
        "sha256": sha256_file(&path)?,
        "size_bytes": metadata.len(),
        "provenance": {},
    }))
}

fn evidence_record(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    value: Value,
    status: &str,
    reasons: &[Value],
) -> Value {
    json!({
        "record_id": format!("{stream_id}:value"),
        "run_id": run_id,
        "scope_id": scope_id,
        "profiler_id": "cgroup-cpu",
        "stream_id": stream_id,
        "shape": "scalar",
        "status": status,
        "reasons": reasons,
        "value": value,
        "source_clock_id": Value::Null,
        "native_timestamp_ns": Value::Null,
        "native_start_ns": Value::Null,
        "native_end_ns": Value::Null,
        "canonical_timestamp_ns": Value::Null,
        "canonical_start_ns": Value::Null,
        "canonical_end_ns": Value::Null,
        "alignment_error_ns": Value::Null,
        "acquisition_bounds": Value::Null,
        "attributes": {},
        "raw_artifact_ids": ["cgroup-cpu-after", "cgroup-cpu-before", "cgroup-cpu-source-snapshots"],
        "provenance": {},
    })
}

// Evidence-schema fields are intentionally passed one-for-one at this leaf.
#[allow(clippy::too_many_arguments)]
fn scalar_stream(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    metric: &str,
    unit: &str,
    value: Value,
    status: &str,
    reasons: &[Value],
) -> Value {
    json!({
        "stream_id": stream_id,
        "profiler_id": "cgroup-cpu",
        "metric": metric,
        "shape": "scalar",
        "unit": unit,
        "status": status,
        "reasons": reasons,
        "records": [evidence_record(run_id, scope_id, stream_id, value, status, reasons)],
        "raw_artifact_ids": ["cgroup-cpu-after", "cgroup-cpu-before", "cgroup-cpu-source-snapshots"],
        "cadence": Value::Null,
        "gaps": [],
        "display": {},
        "provenance": {},
        "extensions": {"measurement_domains": ["cpu.capacity"], "measurement_perturbations": []},
    })
}

// Evidence-schema fields are intentionally passed one-for-one at this leaf.
#[allow(clippy::too_many_arguments)]
fn proc_stat_window_stream(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    metric: &str,
    unit: &str,
    value: Value,
    window: &Value,
    sample_interval_sec: f64,
) -> Result<Value, ProfileError> {
    let start_ns = window
        .get("start_ns")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ProfileError::new(format!(
                "proc-stat stream {stream_id} has no valid start_ns"
            ))
        })?;
    let end_ns = window
        .get("end_ns")
        .and_then(Value::as_u64)
        .ok_or_else(|| {
            ProfileError::new(format!("proc-stat stream {stream_id} has no valid end_ns"))
        })?;
    if end_ns <= start_ns {
        return Err(ProfileError::new(format!(
            "proc-stat stream {stream_id} has a non-positive window"
        )));
    }
    if !value.is_number() {
        return Err(ProfileError::new(format!(
            "proc-stat stream {stream_id} has no numeric aggregate"
        )));
    }
    Ok(json!({
        "stream_id": stream_id,
        "profiler_id": "proc-stat-host-context",
        "metric": metric,
        "shape": "window_aggregate",
        "unit": unit,
        "status": "valid",
        "reasons": [],
        "records": [{
            "record_id": format!("{stream_id}:window"),
            "run_id": run_id,
            "scope_id": scope_id,
            "profiler_id": "proc-stat-host-context",
            "stream_id": stream_id,
            "shape": "window_aggregate",
            "status": "valid",
            "reasons": [],
            "value": value,
            "source_clock_id": "host-monotonic",
            "native_timestamp_ns": Value::Null,
            "native_start_ns": start_ns,
            "native_end_ns": end_ns,
            "canonical_timestamp_ns": Value::Null,
            "canonical_start_ns": start_ns,
            "canonical_end_ns": end_ns,
            "alignment_error_ns": 0,
            "acquisition_bounds": Value::Null,
            "attributes": {
                "scope_semantics": "host-pressure-projected-onto-allocated-logical-cpus",
                "task_attributed": false,
                "scheduler_concurrency": false,
            },
            "raw_artifact_ids": ["proc-stat-samples"],
            "provenance": {
                "formula": "user+nice+system+irq+softirq divided by busy+idle+iowait",
                "not_task_demand": true,
            },
        }],
        "raw_artifact_ids": ["proc-stat-samples"],
        "cadence": Value::Null,
        "gaps": [],
        "display": {"group": "Host context", "label": stream_id},
        "provenance": {
            "scope": "host /proc/stat on allocated CPU IDs",
            "task_attributed": false,
            "peak_is_not_task_concurrency": true,
            "requested_sample_interval_sec": sample_interval_sec,
        },
        "extensions": {
            "measurement_domains": ["host.cpu.pressure"],
            "measurement_perturbations": [],
        },
    }))
}

fn report_reason_strings(report: &Value, field: &str) -> Vec<String> {
    report
        .get(field)
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(ToOwned::to_owned)
        .collect()
}

fn memory_invalid_reasons(memory: &Value) -> Vec<String> {
    let mut reasons = Vec::new();
    for field in [
        "instrumentation_invalid_reasons",
        "peak_invalid_reasons",
        "timeline_invalid_reasons",
        "policy_drift_reasons",
    ] {
        reasons.extend(report_reason_strings(memory, field));
    }
    if memory.get("observation_boundary_complete") != Some(&Value::Bool(true)) {
        reasons.push("memory observation boundary was incomplete".to_owned());
    }
    if memory.get("oom_kill_observed") == Some(&Value::Bool(true)) {
        reasons.push("an OOM kill was observed in the measured memory scope".to_owned());
    }
    if memory.get("policy_stable") == Some(&Value::Bool(false)) {
        reasons.push("memory controller policy changed during observation".to_owned());
    }
    let mut seen = BTreeSet::new();
    reasons.retain(|reason| seen.insert(reason.clone()));
    if memory.get("valid_for_observed_memory_reporting") != Some(&Value::Bool(true))
        && reasons.is_empty()
    {
        reasons.push("observed memory evidence is invalid for reporting".to_owned());
    }
    reasons
}

// Evidence-schema fields are intentionally passed one-for-one at this leaf.
#[allow(clippy::too_many_arguments)]
fn memory_scalar_stream(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    metric: &str,
    unit: &str,
    value: Value,
    reasons: &[String],
    attributes: Value,
) -> Value {
    let artifact_ids = json!([
        "cgroup-resource-timeseries",
        "cgroup-memory-source-snapshots"
    ]);
    if value.is_null() {
        return json!({
            "stream_id": stream_id,
            "profiler_id": "cgroup-memory",
            "metric": metric,
            "shape": "scalar",
            "unit": unit,
            "status": "missing",
            "reasons": if reasons.is_empty() { vec![format!("{metric} was not observed")] } else { reasons.to_vec() },
            "records": [],
            "raw_artifact_ids": artifact_ids,
            "cadence": Value::Null,
            "gaps": [],
            "display": {},
            "provenance": {},
            "extensions": {"measurement_domains": ["memory.capacity"], "measurement_perturbations": []},
        });
    }
    let status = if reasons.is_empty() {
        "valid"
    } else {
        "invalid"
    };
    json!({
        "stream_id": stream_id,
        "profiler_id": "cgroup-memory",
        "metric": metric,
        "shape": "scalar",
        "unit": unit,
        "status": status,
        "reasons": reasons,
        "records": [{
            "record_id": format!("{stream_id}:value"),
            "run_id": run_id,
            "scope_id": scope_id,
            "profiler_id": "cgroup-memory",
            "stream_id": stream_id,
            "shape": "scalar",
            "status": status,
            "reasons": reasons,
            "value": value,
            "source_clock_id": Value::Null,
            "native_timestamp_ns": Value::Null,
            "native_start_ns": Value::Null,
            "native_end_ns": Value::Null,
            "canonical_timestamp_ns": Value::Null,
            "canonical_start_ns": Value::Null,
            "canonical_end_ns": Value::Null,
            "alignment_error_ns": Value::Null,
            "acquisition_bounds": Value::Null,
            "attributes": attributes,
            "raw_artifact_ids": artifact_ids,
            "provenance": {},
        }],
        "raw_artifact_ids": artifact_ids,
        "cadence": Value::Null,
        "gaps": [],
        "display": {},
        "provenance": {},
        "extensions": {"measurement_domains": ["memory.capacity"], "measurement_perturbations": []},
    })
}

// Evidence-schema fields are intentionally passed one-for-one at this leaf.
#[allow(clippy::too_many_arguments)]
fn memory_sample_stream(
    run_id: &str,
    scope_id: &str,
    memory: &Value,
    timeseries: &Value,
    stream_id: &str,
    metric: &str,
    field: &str,
    parent: Option<&str>,
) -> Value {
    let reasons = report_reason_strings(memory, "timeline_invalid_reasons");
    let record_status = if reasons.is_empty() {
        "valid"
    } else {
        "invalid"
    };
    let mut records = Vec::new();
    for (index, row) in timeseries
        .get("samples")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
    {
        let source = parent.and_then(|name| row.get(name)).unwrap_or(row);
        let Some(value) = source.get(field).filter(|value| value.is_number()) else {
            continue;
        };
        let Some(timestamp) = row.get("timestamp_ns").and_then(Value::as_u64) else {
            continue;
        };
        let acquisition_start = row
            .get("acquisition_start_ns")
            .and_then(Value::as_u64)
            .unwrap_or(timestamp);
        let acquisition_end = row
            .get("acquisition_end_ns")
            .and_then(Value::as_u64)
            .unwrap_or(timestamp);
        if acquisition_end < acquisition_start {
            continue;
        }
        records.push(json!({
            "record_id": format!("{stream_id}:{index:06}"),
            "run_id": run_id,
            "scope_id": scope_id,
            "profiler_id": "cgroup-memory",
            "stream_id": stream_id,
            "shape": "sampled_gauge",
            "status": record_status,
            "reasons": reasons,
            "value": value,
            "source_clock_id": "host-monotonic",
            "native_timestamp_ns": timestamp,
            "native_start_ns": Value::Null,
            "native_end_ns": Value::Null,
            "canonical_timestamp_ns": Value::Null,
            "canonical_start_ns": Value::Null,
            "canonical_end_ns": Value::Null,
            "alignment_error_ns": acquisition_end - acquisition_start,
            "acquisition_bounds": {
                "canonical_start_ns": acquisition_start,
                "canonical_end_ns": acquisition_end,
            },
            "attributes": {"task_attributed": true, "scope_includes_descendants": true},
            "raw_artifact_ids": ["cgroup-resource-timeseries"],
            "provenance": {"method": "cgroup-v2-periodic-read"},
        }));
    }
    let (status, stream_reasons) = if records.is_empty() {
        (
            "missing",
            vec![format!("no {metric} samples were observed")],
        )
    } else {
        (record_status, reasons)
    };
    let interval_ns = (memory
        .get("sample_interval_sec")
        .and_then(Value::as_f64)
        .unwrap_or(0.1)
        * 1_000_000_000.0)
        .round() as u64;
    json!({
        "stream_id": stream_id,
        "profiler_id": "cgroup-memory",
        "metric": metric,
        "shape": "sampled_gauge",
        "unit": "byte",
        "status": status,
        "reasons": stream_reasons,
        "records": records,
        "raw_artifact_ids": ["cgroup-resource-timeseries"],
        "cadence": {
            "requested_interval_ns": interval_ns,
            "maximum_acceptable_gap_ns": interval_ns.saturating_mul(2),
            "method": "cgroup-v2-periodic-read",
        },
        "gaps": [],
        "display": {"group": "Memory", "label": metric},
        "provenance": {"scope": "container cgroup v2 subtree", "not_process_rss": true},
        "extensions": {"measurement_domains": ["memory.capacity"], "measurement_perturbations": []},
    })
}

fn memory_evidence_streams(
    run_id: &str,
    scope_id: &str,
    memory: &cgroup_memory::MemoryOutput,
) -> Vec<Value> {
    let report = &memory.report;
    let mut streams = vec![
        memory_sample_stream(
            run_id,
            scope_id,
            report,
            &memory.timeseries,
            "memory-current",
            "memory.current",
            "memory_current_bytes",
            None,
        ),
        memory_sample_stream(
            run_id,
            scope_id,
            report,
            &memory.timeseries,
            "memory-anon",
            "memory.anon",
            "anon",
            Some("memory_stat"),
        ),
        memory_sample_stream(
            run_id,
            scope_id,
            report,
            &memory.timeseries,
            "memory-file",
            "memory.file",
            "file",
            Some("memory_stat"),
        ),
        memory_sample_stream(
            run_id,
            scope_id,
            report,
            &memory.timeseries,
            "memory-kernel",
            "memory.kernel",
            "kernel",
            Some("memory_stat"),
        ),
    ];
    let mut peak_reasons = report_reason_strings(report, "peak_invalid_reasons");
    if report.get("peak_valid") != Some(&Value::Bool(true)) && peak_reasons.is_empty() {
        peak_reasons.push("memory.peak validity was not established".to_owned());
    }
    streams.push(memory_scalar_stream(
        run_id,
        scope_id,
        "memory-peak",
        "memory.peak",
        "byte",
        report.pointer("/lifetime/peak_bytes").cloned().unwrap_or(Value::Null),
        &peak_reasons,
        json!({
            "exact": report.get("peak_valid") == Some(&Value::Bool(true)),
            "natural_peak_trustworthy": report.get("natural_peak_trustworthy") == Some(&Value::Bool(true)),
            "censoring": report.get("censoring").cloned().unwrap_or(Value::Null),
        }),
    ));
    if report
        .pointer("/limits/memory_max_state")
        .and_then(Value::as_str)
        == Some("finite")
    {
        streams.push(memory_scalar_stream(
            run_id,
            scope_id,
            "memory-max",
            "memory.max",
            "byte",
            report
                .pointer("/limits/memory_max_bytes")
                .cloned()
                .unwrap_or(Value::Null),
            &[],
            json!({"exact": true, "timing": "observer-start"}),
        ));
    }
    if let Some(value) = report.pointer("/observed_limit_margin/remaining_to_limit_bytes") {
        if value.is_number() {
            streams.push(memory_scalar_stream(
                run_id,
                scope_id,
                "memory-observed-limit-margin",
                "memory.observed_limit_margin",
                "byte",
                value.clone(),
                &peak_reasons,
                json!({
                    "exact": peak_reasons.is_empty(),
                    "not_a_recommendation": true,
                    "peak_fraction_of_limit": report.pointer("/observed_limit_margin/peak_fraction_of_limit").cloned().unwrap_or(Value::Null),
                }),
            ));
        }
    }
    let instrumentation_reasons = if report.get("instrumentation_valid") == Some(&Value::Bool(true))
    {
        Vec::new()
    } else {
        let mut reasons = report_reason_strings(report, "instrumentation_invalid_reasons");
        if reasons.is_empty() {
            reasons.push("memory instrumentation validity was not established".to_owned());
        }
        reasons
    };
    let mut event_names = report
        .get("events_delta")
        .and_then(Value::as_object)
        .map(|events| events.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    event_names.sort();
    for name in event_names {
        if let Some(value) = report.pointer(&format!("/events_delta/{name}")) {
            if value.as_u64().is_some() {
                streams.push(memory_scalar_stream(
                    run_id,
                    scope_id,
                    &format!("memory-event-{name}"),
                    &format!("memory.events.{name}"),
                    "event",
                    value.clone(),
                    &instrumentation_reasons,
                    json!({}),
                ));
            }
        }
    }
    let markers: BTreeMap<_, _> = report
        .get("markers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|marker| {
            marker
                .get("name")
                .and_then(Value::as_str)
                .zip(marker.get("timestamp_ns").and_then(Value::as_u64))
                .map(|(name, timestamp)| (name.to_owned(), timestamp))
        })
        .collect();
    let pressure_window = markers
        .get("sandbox_observation_start")
        .copied()
        .zip(markers.get("sandbox_observation_end").copied())
        .filter(|(start, end)| end > start);
    let mut pressure_names = report
        .get("pressure_delta")
        .and_then(Value::as_object)
        .map(|values| values.keys().cloned().collect::<Vec<_>>())
        .unwrap_or_default();
    pressure_names.sort();
    if let Some((start_ns, end_ns)) = pressure_window {
        for name in pressure_names {
            let Some(value) = report
                .pointer(&format!("/pressure_delta/{name}/total_stall_usec"))
                .and_then(Value::as_u64)
            else {
                continue;
            };
            let status = if instrumentation_reasons.is_empty() {
                "valid"
            } else {
                "invalid"
            };
            let stream_id = format!("memory-pressure-{name}");
            streams.push(json!({
                "stream_id": stream_id,
                "profiler_id": "cgroup-memory",
                "metric": format!("memory.pressure.{name}.total_stall"),
                "shape": "window_aggregate",
                "unit": "microsecond",
                "status": status,
                "reasons": instrumentation_reasons,
                "records": [{
                    "record_id": format!("memory-pressure-{name}:window"),
                    "run_id": run_id,
                    "scope_id": scope_id,
                    "profiler_id": "cgroup-memory",
                    "stream_id": format!("memory-pressure-{name}"),
                    "shape": "window_aggregate",
                    "status": status,
                    "reasons": instrumentation_reasons,
                    "value": value,
                    "source_clock_id": "host-monotonic",
                    "native_timestamp_ns": Value::Null,
                    "native_start_ns": start_ns,
                    "native_end_ns": end_ns,
                    "canonical_timestamp_ns": Value::Null,
                    "canonical_start_ns": start_ns,
                    "canonical_end_ns": end_ns,
                    "alignment_error_ns": 0,
                    "acquisition_bounds": Value::Null,
                    "attributes": {},
                    "raw_artifact_ids": ["cgroup-resource-timeseries", "cgroup-memory-source-snapshots"],
                    "provenance": {"timing": "aggregate-only-stall-occurrence-time-unknown"},
                }],
                "raw_artifact_ids": ["cgroup-resource-timeseries", "cgroup-memory-source-snapshots"],
                "cadence": Value::Null,
                "gaps": [],
                "display": {"group": "Memory pressure", "label": name},
                "provenance": {},
                "extensions": {"measurement_domains": ["memory.capacity"], "measurement_perturbations": []},
            }));
        }
    }
    streams
}

fn verification_report(config: &ProfileConfig, output_dir: &Path) -> Result<Value, ProfileError> {
    let Some(command) = &config.verification_command else {
        return Ok(json!({
            "requested": false,
            "status": "not_requested",
            "valid": Value::Null,
            "reasons": [],
        }));
    };
    let files = [
        ("stdout", "verification.stdout.txt"),
        ("stderr", "verification.stderr.txt"),
        ("exit_code", "verification_exit_code"),
        ("status", "verification_status"),
    ];
    let mut descriptors = Map::new();
    for (name, filename) in files {
        let path = output_dir.join(filename);
        if !path.is_file() {
            return Err(ProfileError::new(format!(
                "workload verification did not retain required artifact: {filename}"
            )));
        }
        descriptors.insert(
            name.to_owned(),
            json!({
                "path": path,
                "sha256": sha256_file(&path)?,
                "size_bytes": path.metadata()?.len(),
            }),
        );
    }
    let exit_code = fs::read_to_string(output_dir.join("verification_exit_code"))?
        .trim()
        .parse::<i32>()
        .map_err(|_| ProfileError::new("workload verification exit code is malformed"))?;
    let status = fs::read_to_string(output_dir.join("verification_status"))?
        .trim()
        .to_owned();
    let expected = if exit_code == 0 { "passed" } else { "failed" };
    if status != expected {
        return Err(ProfileError::new(
            "workload verification status disagrees with its exit code",
        ));
    }
    Ok(json!({
        "requested": true,
        "status": status,
        "valid": exit_code == 0,
        "exit_code": exit_code,
        "stdout": fs::read_to_string(output_dir.join("verification.stdout.txt"))?,
        "stderr": fs::read_to_string(output_dir.join("verification.stderr.txt"))?,
        "reasons": if exit_code == 0 { Vec::<String>::new() } else { vec![format!("verification command exited with status {exit_code}")] },
        "command_sha256": sha256_bytes(command.as_bytes()),
        "artifacts": descriptors,
        "measurement_boundary": "after-profiler-stop",
        "measured_counters_include_verification": false,
    }))
}

fn collect_process_exec_report(
    config: &ProfileConfig,
    output_dir: &Path,
) -> Result<Option<Value>, ProfileError> {
    if config.effective_strace_mode() != StraceMode::Process {
        return Ok(None);
    }
    let trace_path = output_dir.join("process.strace.log");
    let report_path = output_dir.join("process_exec_report.json");
    let mut report =
        process_exec::load_process_exec_trace(&trace_path, config.workdir.as_deref(), None);
    report
        .as_object_mut()
        .expect("process exec report object")
        .insert(
            "artifacts".to_owned(),
            json!({
                "raw_trace_path": trace_path,
                "report_path": report_path,
            }),
        );
    process_exec::write_process_exec_report(&report, &report_path)?;
    Ok(Some(report))
}

fn process_exec_reasons(report: &Value) -> Vec<Value> {
    report
        .get("anomalies")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|anomaly| anomaly.get("message").and_then(Value::as_str))
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .map(Value::String)
        .collect()
}

#[allow(clippy::too_many_arguments)]
fn process_exec_scalar_stream(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    metric: &str,
    unit: &str,
    value: Value,
    status: &str,
    reasons: &[Value],
    artifact_ids: &[String],
) -> Value {
    json!({
        "stream_id": stream_id,
        "profiler_id": "process-exec",
        "metric": metric,
        "shape": "scalar",
        "unit": unit,
        "status": status,
        "reasons": reasons,
        "records": [{
            "record_id": format!("{stream_id}:value"),
            "run_id": run_id,
            "scope_id": scope_id,
            "profiler_id": "process-exec",
            "stream_id": stream_id,
            "shape": "scalar",
            "status": status,
            "reasons": reasons,
            "value": value,
            "source_clock_id": Value::Null,
            "native_timestamp_ns": Value::Null,
            "native_start_ns": Value::Null,
            "native_end_ns": Value::Null,
            "canonical_timestamp_ns": Value::Null,
            "canonical_start_ns": Value::Null,
            "canonical_end_ns": Value::Null,
            "alignment_error_ns": Value::Null,
            "acquisition_bounds": Value::Null,
            "attributes": {},
            "raw_artifact_ids": artifact_ids,
            "provenance": {},
        }],
        "raw_artifact_ids": artifact_ids,
        "cadence": Value::Null,
        "gaps": [],
        "display": {},
        "provenance": {},
        "extensions": {
            "measurement_domains": ["software.process"],
            "measurement_perturbations": [],
        },
    })
}

fn process_exec_event_stream(
    run_id: &str,
    scope_id: &str,
    report: &Value,
    artifact_ids: &[String],
) -> Option<Value> {
    let reason = Value::String(
        "strace CLOCK_REALTIME was not correlated to host CLOCK_MONOTONIC during this run"
            .to_owned(),
    );
    let mut records = Vec::new();
    for process in report
        .get("processes")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for event in process
            .get("execs")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let Some(started_ns) = event.get("started_ns").and_then(Value::as_u64) else {
                continue;
            };
            records.push(json!({
                "record_id": format!("process-exec-event:{:06}", records.len()),
                "run_id": run_id,
                "scope_id": scope_id,
                "profiler_id": "process-exec",
                "stream_id": "process-exec-events",
                "shape": "point_event",
                "status": "unalignable",
                "reasons": [reason.clone()],
                "value": {
                    "pid": process.get("pid").cloned().unwrap_or(Value::Null),
                    "binary": event.get("binary").cloned().unwrap_or(Value::Null),
                    "argv": event.get("argv").cloned().unwrap_or(Value::Null),
                    "cwd": event.get("cwd").cloned().unwrap_or(Value::Null),
                    "syscall": event.get("syscall").cloned().unwrap_or(Value::Null),
                },
                "source_clock_id": "strace-realtime",
                "native_timestamp_ns": started_ns,
                "native_start_ns": Value::Null,
                "native_end_ns": Value::Null,
                "canonical_timestamp_ns": Value::Null,
                "canonical_start_ns": Value::Null,
                "canonical_end_ns": Value::Null,
                "alignment_error_ns": Value::Null,
                "acquisition_bounds": Value::Null,
                "attributes": {},
                "raw_artifact_ids": artifact_ids,
                "provenance": {},
            }));
        }
    }
    if records.is_empty() {
        return None;
    }
    Some(json!({
        "stream_id": "process-exec-events",
        "profiler_id": "process-exec",
        "metric": "software.exec",
        "shape": "point_event",
        "unit": "event",
        "status": "unalignable",
        "reasons": [reason],
        "records": records,
        "raw_artifact_ids": artifact_ids,
        "cadence": Value::Null,
        "gaps": [],
        "display": {},
        "provenance": {},
        "extensions": {
            "measurement_domains": ["software.process"],
            "measurement_perturbations": [],
        },
    }))
}

fn scheduler_invalid_reasons(report: &Value) -> Vec<Value> {
    let mut reasons = report
        .get("anomalies")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect::<BTreeSet<_>>();
    if let Some(lost) = report.get("lost_perf_events").and_then(Value::as_u64) {
        if lost > 0 {
            reasons.insert(format!("scheduler trace reported {lost} lost perf events"));
        }
    }
    if report.get("trace_complete") != Some(&Value::Bool(true)) {
        reasons.insert("scheduler trace failed its completeness gate".to_owned());
    }
    if report.pointer("/cgroup_validation/within_tolerance") != Some(&Value::Bool(true)) {
        reasons.insert("scheduler trace failed cgroup-integral reconciliation".to_owned());
    }
    let authoritative_usage_sec = report
        .pointer("/cgroup_validation/authoritative_cgroup_usage_sec")
        .and_then(Value::as_f64)
        .unwrap_or(0.0);
    let raw_event_observed = [
        "events_observed",
        "events_ignored_before_window",
        "events_ignored_after_window",
    ]
    .iter()
    .any(|field| report.get(field).and_then(Value::as_u64).unwrap_or(0) > 0);
    if authoritative_usage_sec > 0.0 && !raw_event_observed {
        reasons.insert(
            "scheduler trace observed no raw events despite nonzero cgroup CPU usage".to_owned(),
        );
    }
    reasons.into_iter().map(Value::String).collect()
}

#[allow(clippy::too_many_arguments)]
fn scheduler_scalar_stream(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    metric: &str,
    unit: &str,
    value: Value,
    status: &str,
    reasons: &[Value],
    artifact_ids: &[String],
) -> Value {
    json!({
        "stream_id": stream_id,
        "profiler_id": "scheduler-concurrency",
        "metric": metric,
        "shape": "scalar",
        "unit": unit,
        "status": status,
        "reasons": reasons,
        "records": [{
            "record_id": format!("{stream_id}:value"),
            "run_id": run_id,
            "scope_id": scope_id,
            "profiler_id": "scheduler-concurrency",
            "stream_id": stream_id,
            "shape": "scalar",
            "status": status,
            "reasons": reasons,
            "value": value,
            "source_clock_id": Value::Null,
            "native_timestamp_ns": Value::Null,
            "native_start_ns": Value::Null,
            "native_end_ns": Value::Null,
            "canonical_timestamp_ns": Value::Null,
            "canonical_start_ns": Value::Null,
            "canonical_end_ns": Value::Null,
            "alignment_error_ns": Value::Null,
            "acquisition_bounds": Value::Null,
            "attributes": {},
            "raw_artifact_ids": artifact_ids,
            "provenance": {},
        }],
        "raw_artifact_ids": artifact_ids,
        "cadence": Value::Null,
        "gaps": [],
        "display": {},
        "provenance": {},
        "extensions": {
            "measurement_domains": ["cpu.capacity"],
            "measurement_perturbations": [],
        },
    })
}

fn read_scheduler_events(
    path: &Path,
) -> Result<Vec<crate::scheduler_perf::SchedulerEvent>, ProfileError> {
    let text = fs::read_to_string(path).map_err(|error| {
        ProfileError::new(format!(
            "unable to read scheduler event artifact safely: {error}"
        ))
    })?;
    let mut events = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let event = serde_json::from_str::<crate::scheduler_perf::SchedulerEvent>(line).map_err(
            |error| {
                ProfileError::new(format!(
                    "scheduler event artifact line {} is malformed: {error}",
                    index + 1
                ))
            },
        )?;
        events.push(event);
    }
    if events.is_empty() {
        return Err(ProfileError::new(
            "scheduler event artifact contains no transitions",
        ));
    }
    events.sort_by_key(|event| (event.timestamp_ns, !event.entering, event.cpu, event.tid));
    Ok(events)
}

fn scheduler_interval_streams(
    run_id: &str,
    scope_id: &str,
    report: &Value,
    events_path: &Path,
    status: &str,
    reasons: &[Value],
    artifact_ids: &[String],
) -> Result<Vec<Value>, ProfileError> {
    let events = read_scheduler_events(events_path)?;
    let start_ns = report
        .get("window_start_ns")
        .and_then(Value::as_u64)
        .ok_or_else(|| ProfileError::new("scheduler report omitted window_start_ns"))?;
    let end_ns = report
        .get("window_end_ns")
        .and_then(Value::as_u64)
        .filter(|end| *end > start_ns)
        .ok_or_else(|| ProfileError::new("scheduler report has an invalid measured window"))?;
    let mut active = BTreeMap::<u32, (u64, i64)>::new();
    for event in events.iter().filter(|event| event.timestamp_ns < start_ns) {
        if event.entering {
            active.insert(event.cpu, (start_ns, event.tid));
        } else {
            active.remove(&event.cpu);
        }
    }
    let within = events
        .iter()
        .filter(|event| start_ns <= event.timestamp_ns && event.timestamp_ns <= end_ns)
        .collect::<Vec<_>>();
    let mut concurrency = Vec::new();
    let mut per_core = Vec::new();
    let mut cursor = start_ns;
    let mut index = 0;
    while index < within.len() {
        let timestamp = within[index].timestamp_ns;
        if timestamp > cursor {
            concurrency.push(scheduler_interval_record(
                run_id,
                scope_id,
                "scheduler-concurrency",
                concurrency.len(),
                cursor,
                timestamp,
                json!(active.len()),
                status,
                reasons,
                artifact_ids,
                json!({}),
            ));
        }
        let mut end = index + 1;
        while end < within.len() && within[end].timestamp_ns == timestamp {
            end += 1;
        }
        for event in within[index..end].iter().filter(|event| !event.entering) {
            if let Some((started_ns, tid)) = active.remove(&event.cpu) {
                if timestamp > started_ns {
                    per_core.push(scheduler_interval_record(
                        run_id,
                        scope_id,
                        "scheduler-per-core",
                        per_core.len(),
                        started_ns,
                        timestamp,
                        json!({"cpu": event.cpu, "tid": tid}),
                        status,
                        reasons,
                        artifact_ids,
                        json!({"lane": format!("logical-cpu-{}", event.cpu)}),
                    ));
                }
            }
        }
        for event in within[index..end].iter().filter(|event| event.entering) {
            active.insert(event.cpu, (timestamp, event.tid));
        }
        cursor = timestamp;
        index = end;
    }
    if cursor < end_ns {
        concurrency.push(scheduler_interval_record(
            run_id,
            scope_id,
            "scheduler-concurrency",
            concurrency.len(),
            cursor,
            end_ns,
            json!(active.len()),
            status,
            reasons,
            artifact_ids,
            json!({}),
        ));
    }
    for (cpu, (started_ns, tid)) in active {
        if end_ns > started_ns {
            per_core.push(scheduler_interval_record(
                run_id,
                scope_id,
                "scheduler-per-core",
                per_core.len(),
                started_ns,
                end_ns,
                json!({"cpu": cpu, "tid": tid}),
                status,
                reasons,
                artifact_ids,
                json!({"lane": format!("logical-cpu-{cpu}")}),
            ));
        }
    }
    Ok(vec![
        scheduler_interval_stream(
            "scheduler-concurrency",
            "scheduler.concurrent_logical_cpus",
            "logical_cpu",
            concurrency,
            status,
            reasons,
            artifact_ids,
        ),
        scheduler_interval_stream(
            "scheduler-per-core",
            "scheduler.logical_cpu_activity",
            "scheduled_interval",
            per_core,
            status,
            reasons,
            artifact_ids,
        ),
    ])
}

#[allow(clippy::too_many_arguments)]
fn scheduler_interval_record(
    run_id: &str,
    scope_id: &str,
    stream_id: &str,
    index: usize,
    start_ns: u64,
    end_ns: u64,
    value: Value,
    status: &str,
    reasons: &[Value],
    artifact_ids: &[String],
    attributes: Value,
) -> Value {
    json!({
        "record_id": format!("{stream_id}:{index:08}"),
        "run_id": run_id,
        "scope_id": scope_id,
        "profiler_id": "scheduler-concurrency",
        "stream_id": stream_id,
        "shape": "interval",
        "status": status,
        "reasons": reasons,
        "value": value,
        "source_clock_id": "host-monotonic",
        "native_timestamp_ns": Value::Null,
        "native_start_ns": start_ns,
        "native_end_ns": end_ns,
        "canonical_timestamp_ns": Value::Null,
        "canonical_start_ns": start_ns,
        "canonical_end_ns": end_ns,
        "alignment_error_ns": 0,
        "acquisition_bounds": Value::Null,
        "attributes": attributes,
        "raw_artifact_ids": artifact_ids,
        "provenance": {"method": "integrated-scheduler-transitions"},
    })
}

fn scheduler_interval_stream(
    stream_id: &str,
    metric: &str,
    unit: &str,
    records: Vec<Value>,
    status: &str,
    reasons: &[Value],
    artifact_ids: &[String],
) -> Value {
    let (stream_status, stream_reasons) = if records.is_empty() {
        (
            "missing",
            vec![Value::String(
                "scheduler transitions contain no positive-duration intervals".to_owned(),
            )],
        )
    } else {
        (status, reasons.to_vec())
    };
    json!({
        "stream_id": stream_id,
        "profiler_id": "scheduler-concurrency",
        "metric": metric,
        "shape": "interval",
        "unit": unit,
        "status": stream_status,
        "reasons": stream_reasons,
        "records": records,
        "raw_artifact_ids": artifact_ids,
        "cadence": Value::Null,
        "gaps": [],
        "display": {},
        "provenance": {},
        "extensions": {
            "measurement_domains": ["cpu.capacity"],
            "measurement_perturbations": [],
        },
    })
}

type SchedulerArtifactSpec = (&'static str, &'static str, &'static str);

fn scheduler_artifact_specs(config: &ProfileConfig) -> &'static [SchedulerArtifactSpec] {
    const BCC: &[SchedulerArtifactSpec] = &[(
        "scheduler_events.jsonl",
        "scheduler-events",
        "application/x-ndjson",
    )];
    const PERF: &[SchedulerArtifactSpec] = &[
        (
            "scheduler_perf.data",
            "scheduler-perf-data",
            "application/octet-stream",
        ),
        (
            "scheduler_perf_switch_events.txt",
            "scheduler-perf-decoded",
            "text/plain",
        ),
        (
            "scheduler_perf.stderr.txt",
            "scheduler-perf-stderr",
            "text/plain",
        ),
        (
            "scheduler_events.jsonl",
            "scheduler-events",
            "application/x-ndjson",
        ),
    ];
    if config.scheduler_backend == "bcc" {
        BCC
    } else {
        PERF
    }
}

fn scheduler_configuration(config: &ProfileConfig) -> Value {
    if config.scheduler_backend == "bcc" {
        json!({
            "backend": "bcc",
            "method": "ebpf-bcc-finish-task-switch-cgroup-subtree",
            "bcc_buffer_pages": config.scheduler_bcc_buffer_pages,
            "cpu_tolerance_fraction": config.scheduler_cpu_tolerance_fraction,
            "cpu_tolerance_sec": config.scheduler_cpu_tolerance_sec,
            "per_observed_event_tolerance_sec": crate::scheduler_bcc::EVENT_BOUNDARY_TOLERANCE_SEC,
            "clockid": "mono",
        })
    } else {
        json!({
            "backend": "perf",
            "method": "perf-record-cgroup-native-switch-events",
            "perf_binary": config.scheduler_perf_binary,
            "perf_mmap_pages": config.scheduler_perf_mmap_pages,
            "cpu_tolerance_fraction": config.scheduler_cpu_tolerance_fraction,
            "cpu_tolerance_sec": config.scheduler_cpu_tolerance_sec,
            "clockid": "mono",
        })
    }
}

fn scheduler_provenance(config: &ProfileConfig) -> Value {
    let requirements = if config.scheduler_backend == "bcc" {
        json!([
            "cgroup-v2-descendant-scope",
            "root",
            "runtime-loadable-libbcc",
            "kernel-ebpf-kprobe",
            "bpf_get_current_ancestor_cgroup_id"
        ])
    } else {
        json!(["cgroup-v2-descendant-scope", "perf-event-cgroup"])
    };
    json!({
        "requirements": requirements,
        "scope_includes_descendants": true,
        "authoritative_cpu_integral": "cgroup-cpu",
        "event_clock": "CLOCK_MONOTONIC",
    })
}

#[allow(clippy::too_many_arguments)]
fn build_evidence_bundle(
    config: &ProfileConfig,
    run_id: &str,
    scope_id: &str,
    scope: &Value,
    lifecycle: &Value,
    clock: &Value,
    plan: &Value,
    plan_digest: &str,
    cpu: &Value,
    scheduler: Option<&Value>,
    hardware: Option<&Value>,
    proc_stat: Option<&Value>,
    memory: Option<&cgroup_memory::MemoryOutput>,
    process_exec: Option<&Value>,
    runtime_report: &Value,
    verification: &Value,
    output_dir: &Path,
) -> Result<Value, ProfileError> {
    let invalid_reasons: Vec<Value> = cpu["invalid_reasons"]
        .as_array()
        .cloned()
        .unwrap_or_default();
    let raw_cpu_valid = invalid_reasons.is_empty();
    let cpu_selected = config.cpu_evidence_selected();
    let cpu_valid = !cpu_selected || raw_cpu_valid;
    let cpu_status = if raw_cpu_valid { "valid" } else { "invalid" };
    let stream_ids = vec![
        "cpu-average-logical-cores",
        "cpu-interval-average",
        "cpu-system",
        "cpu-usage",
        "cpu-user",
        "cpu-wall",
    ];
    let mut profilers = Vec::new();
    let mut streams = Vec::new();
    let mut artifacts = Vec::new();
    if cpu_selected {
        profilers.push(json!({
            "profiler_id": "cgroup-cpu",
            "profiler_version": PROFILER_VERSION,
            "status": cpu_status,
            "reasons": invalid_reasons,
            "stream_ids": stream_ids,
            "intrusive": false,
            "configuration": {"method": "cgroup-v2-cpu.stat", "cpuset_cpus": config.cpuset_cpus},
            "provenance": {
                "requirements": ["cgroup-v2-descendant-scope"],
                "conflicts": [],
                "intrusive_reason": Value::Null,
                "scope_includes_descendants": true,
                "cpu_policy_gate_scope": "leaf-and-every-ancestor-to-cgroup-root",
                "cpu_policy_gate_shared_by_baseline_and-profile": true,
            },
            "extensions": {
                "measurement_domains": ["cpu.capacity"],
                "measurement_perturbations": [],
                "invalidates_measurement_domains": [],
                "invalidates_metrics": [],
            },
        }));
        streams.extend([
            scalar_stream(run_id, scope_id, "cpu-average-logical-cores", "cpu.average_logical_cores", "logical_cpu", cpu["average_logical_cores"].clone(), cpu_status, &invalid_reasons),
            json!({
            "stream_id": "cpu-interval-average",
            "profiler_id": "cgroup-cpu",
            "metric": "cpu.interval_average_logical_cores",
            "shape": "window_aggregate",
            "unit": "logical_cpu",
            "status": "missing",
            "reasons": ["resource sampler produced no CPU interval deltas"],
            "records": [],
            "raw_artifact_ids": ["cgroup-cpu-source-snapshots"],
            "cadence": Value::Null,
            "gaps": [],
            "display": {},
            "provenance": {},
            "extensions": {"measurement_domains": ["cpu.capacity"], "measurement_perturbations": []},
            }),
            scalar_stream(run_id, scope_id, "cpu-system", "cpu.system_seconds", "second", cpu["system_sec"].clone(), cpu_status, &invalid_reasons),
            scalar_stream(run_id, scope_id, "cpu-usage", "cpu.usage_seconds", "second", cpu["usage_sec"].clone(), cpu_status, &invalid_reasons),
            scalar_stream(run_id, scope_id, "cpu-user", "cpu.user_seconds", "second", cpu["user_sec"].clone(), cpu_status, &invalid_reasons),
            scalar_stream(run_id, scope_id, "cpu-wall", "cpu.wall_seconds", "second", cpu["wall_sec"].clone(), cpu_status, &invalid_reasons),
        ]);
        artifacts.extend([
            artifact(
                output_dir,
                "cgroup_cpu.stat.controlled_finish_gate.txt",
                "cgroup-cpu-after",
                "text/plain",
            )?,
            artifact(
                output_dir,
                "cgroup_cpu.stat.observer_start.txt",
                "cgroup-cpu-before",
                "text/plain",
            )?,
            artifact(
                output_dir,
                "cgroup_cpu_source_snapshots.json",
                "cgroup-cpu-source-snapshots",
                "application/json",
            )?,
        ]);
    }
    let scheduler_reasons = scheduler.map(scheduler_invalid_reasons).unwrap_or_default();
    let scheduler_valid = scheduler.map_or(true, |report| {
        report.pointer("/cgroup_validation/valid_for_concurrency_reporting")
            == Some(&Value::Bool(true))
    });
    if let Some(report) = scheduler {
        let scheduler_status = if scheduler_valid { "valid" } else { "invalid" };
        let scheduler_stream_ids = [
            "cpu-peak-simultaneous",
            "cpu-scheduled-runtime",
            "scheduler-concurrency",
            "scheduler-per-core",
        ];
        let artifact_ids = scheduler_artifact_specs(config)
            .iter()
            .map(|(_, artifact_id, _)| (*artifact_id).to_owned())
            .collect::<Vec<_>>();
        for (filename, artifact_id, media_type) in scheduler_artifact_specs(config) {
            artifacts.push(artifact(output_dir, filename, artifact_id, media_type)?);
        }
        profilers.push(json!({
            "profiler_id": "scheduler-concurrency",
            "profiler_version": PROFILER_VERSION,
            "status": scheduler_status,
            "reasons": scheduler_reasons,
            "stream_ids": scheduler_stream_ids,
            "intrusive": false,
            "configuration": scheduler_configuration(config),
            "provenance": scheduler_provenance(config),
            "extensions": {
                "measurement_domains": ["cpu.capacity"],
                "measurement_perturbations": [],
                "invalidates_measurement_domains": [],
                "invalidates_metrics": [],
            },
        }));
        streams.push(scheduler_scalar_stream(
            run_id,
            scope_id,
            "cpu-peak-simultaneous",
            "cpu.peak_simultaneous_logical_cores",
            "logical_cpu",
            report
                .get("peak_simultaneous_logical_cores")
                .cloned()
                .unwrap_or(Value::Null),
            scheduler_status,
            &scheduler_reasons,
            &artifact_ids,
        ));
        streams.push(scheduler_scalar_stream(
            run_id,
            scope_id,
            "cpu-scheduled-runtime",
            "cpu.scheduled_runtime_seconds",
            "second",
            report
                .get("scheduled_runtime_sec")
                .cloned()
                .unwrap_or(Value::Null),
            scheduler_status,
            &scheduler_reasons,
            &artifact_ids,
        ));
        streams.extend(scheduler_interval_streams(
            run_id,
            scope_id,
            report,
            &output_dir.join("scheduler_events.jsonl"),
            scheduler_status,
            &scheduler_reasons,
            &artifact_ids,
        )?);
    }
    let hardware_evidence = hardware
        .map(|report| {
            crate::hardware_perf::build_hardware_perf_evidence(run_id, scope_id, report, output_dir)
        })
        .transpose()?;
    let hardware_valid = hardware_evidence
        .as_ref()
        .map_or(true, |evidence| evidence.valid);
    let hardware_reasons = hardware_evidence
        .as_ref()
        .map(|evidence| evidence.reasons.clone())
        .unwrap_or_default();
    if let Some(evidence) = &hardware_evidence {
        profilers.push(evidence.profiler.clone());
        streams.extend(evidence.streams.iter().cloned());
        artifacts.extend(evidence.artifacts.iter().cloned());
    }
    if let Some(proc_stat) = proc_stat {
        let baseline_window = proc_stat
            .get("baseline_window")
            .ok_or_else(|| ProfileError::new("proc-stat report omitted baseline_window"))?;
        let measurement_window = proc_stat
            .get("measurement_window")
            .ok_or_else(|| ProfileError::new("proc-stat report omitted measurement_window"))?;
        let proc_stream_ids = vec![
            "proc-stat-baseline-average-busy",
            "proc-stat-measurement-average-busy",
            "proc-stat-baseline-subtracted-average-busy",
            "proc-stat-measurement-peak-active",
        ];
        profilers.push(json!({
            "profiler_id": "proc-stat-host-context",
            "profiler_version": "tasktrace-proc-stat-comparison.v1",
            "status": "valid",
            "reasons": [],
            "stream_ids": proc_stream_ids,
            "intrusive": false,
            "configuration": {
                "method": "tasktrace-style-proc-stat-sampling",
                "cpuset_cpus": config.cpuset_cpus,
                "sample_interval_sec": config.proc_stat_interval_sec,
                "baseline_sec": config.proc_stat_baseline_sec,
                "active_threshold_percent": proc_stat.get("active_threshold_percent").cloned().unwrap_or(Value::Null),
            },
            "provenance": {
                "requirements": ["host-proc-stat", "concrete-cpuset"],
                "scope": "host pressure projected onto allocated logical CPU IDs",
                "task_attributed": false,
                "authoritative_task_cpu_source": "cgroup-cpu",
                "peak_is_not_task_concurrency": true,
            },
            "extensions": {
                "measurement_domains": ["host.cpu.pressure"],
                "measurement_perturbations": [],
                "invalidates_measurement_domains": [],
                "invalidates_metrics": [],
            },
        }));
        streams.push(proc_stat_window_stream(
            run_id,
            scope_id,
            proc_stream_ids[0],
            "host.cpu.baseline_average_busy_logical_cores",
            "logical_cpu",
            proc_stat["baseline"]["average_busy_logical_cores"].clone(),
            baseline_window,
            config.proc_stat_interval_sec,
        )?);
        streams.push(proc_stat_window_stream(
            run_id,
            scope_id,
            proc_stream_ids[1],
            "host.cpu.measurement_average_busy_logical_cores",
            "logical_cpu",
            proc_stat["measurement"]["average_busy_logical_cores"].clone(),
            measurement_window,
            config.proc_stat_interval_sec,
        )?);
        streams.push(proc_stat_window_stream(
            run_id,
            scope_id,
            proc_stream_ids[2],
            "host.cpu.baseline_subtracted_average_busy_logical_cores",
            "logical_cpu",
            proc_stat["baseline_subtracted_estimate"]["average_busy_logical_cores"].clone(),
            measurement_window,
            config.proc_stat_interval_sec,
        )?);
        streams.push(proc_stat_window_stream(
            run_id,
            scope_id,
            proc_stream_ids[3],
            "host.cpu.sampled_peak_active_logical_cpus",
            "logical_cpu",
            proc_stat["measurement"]["peak_active_logical_cpus"].clone(),
            measurement_window,
            config.proc_stat_interval_sec,
        )?);
        artifacts.push(artifact(
            output_dir,
            "proc_stat_samples.json",
            "proc-stat-samples",
            "application/json",
        )?);
    }
    let memory_reasons = memory
        .map(|output| memory_invalid_reasons(&output.report))
        .unwrap_or_default();
    let memory_valid = memory.map_or(true, |output| {
        output.report.get("valid_for_observed_memory_reporting") == Some(&Value::Bool(true))
    });
    let memory_status = if memory_valid { "valid" } else { "invalid" };
    if let Some(memory) = memory {
        let memory_streams = memory_evidence_streams(run_id, scope_id, memory);
        let memory_stream_ids = memory_streams
            .iter()
            .filter_map(|stream| stream.get("stream_id").and_then(Value::as_str))
            .map(ToOwned::to_owned)
            .collect::<Vec<_>>();
        streams.extend(memory_streams);
        profilers.push(json!({
            "profiler_id": "cgroup-memory",
            "profiler_version": PROFILER_VERSION,
            "status": memory_status,
            "reasons": memory_reasons,
            "stream_ids": memory_stream_ids,
            "intrusive": false,
            "configuration": {
                "method": "cgroup-v2-memory-and-cpu-stat-sampling",
                "sample_interval_sec": config.memory_sample_interval_sec,
            },
            "provenance": {
                "requirements": ["cgroup-v2-descendant-scope", "memory-controller"],
                "scope_includes_descendants": true,
                "ancestor_chain_observed": true,
                "peak_source": "memory.peak",
                "natural_peak_is_separate_from_observed_peak": true,
            },
            "extensions": {
                "measurement_domains": ["memory.capacity"],
                "measurement_perturbations": [],
                "invalidates_measurement_domains": [],
                "invalidates_metrics": [],
            },
        }));
        artifacts.push(artifact(
            output_dir,
            "resource_timeseries.json",
            "cgroup-resource-timeseries",
            "application/json",
        )?);
        artifacts.push(artifact(
            output_dir,
            "cgroup_memory_source_snapshots.json",
            "cgroup-memory-source-snapshots",
            "application/json",
        )?);
    }
    let process_reasons = process_exec.map(process_exec_reasons).unwrap_or_default();
    let process_valid = process_exec.map_or(true, |report| {
        report.get("valid") == Some(&Value::Bool(true))
    });
    if let Some(report) = process_exec {
        let mut artifact_ids = Vec::new();
        let raw_trace = output_dir.join("process.strace.log");
        if raw_trace.is_file() {
            artifacts.push(artifact(
                output_dir,
                "process.strace.log",
                "process-strace-raw",
                "text/plain",
            )?);
            artifact_ids.push("process-strace-raw".to_owned());
        }
        artifacts.push(artifact(
            output_dir,
            "process_exec_report.json",
            "process-exec-report",
            "application/json",
        )?);
        artifact_ids.push("process-exec-report".to_owned());
        let scalar_status = if process_valid { "valid" } else { "invalid" };
        let scalar_reasons = if process_valid {
            Vec::new()
        } else {
            process_reasons.clone()
        };
        let mut process_stream_ids = vec!["process-count", "binary-count"];
        streams.push(process_exec_scalar_stream(
            run_id,
            scope_id,
            "process-count",
            "software.process_count",
            "process",
            report
                .pointer("/summary/process_count")
                .cloned()
                .unwrap_or(Value::Null),
            scalar_status,
            &scalar_reasons,
            &artifact_ids,
        ));
        streams.push(process_exec_scalar_stream(
            run_id,
            scope_id,
            "binary-count",
            "software.distinct_binary_count",
            "binary",
            json!(report
                .get("binary_inventory")
                .and_then(Value::as_array)
                .map_or(0, Vec::len)),
            scalar_status,
            &scalar_reasons,
            &artifact_ids,
        ));
        if let Some(events) = process_exec_event_stream(run_id, scope_id, report, &artifact_ids) {
            process_stream_ids.push("process-exec-events");
            streams.push(events);
        }
        let profiler_reasons = if process_valid {
            vec![Value::String(
                "strace event timestamps cannot be placed on the canonical monotonic axis"
                    .to_owned(),
            )]
        } else {
            vec![Value::String(
                process_reasons
                    .iter()
                    .filter_map(Value::as_str)
                    .collect::<Vec<_>>()
                    .join("; "),
            )]
        };
        profilers.push(json!({
            "profiler_id": "process-exec",
            "profiler_version": PROFILER_VERSION,
            "status": if process_valid { "unalignable" } else { "invalid" },
            "reasons": profiler_reasons,
            "stream_ids": process_stream_ids,
            "intrusive": true,
            "configuration": {
                "method": process_exec::PROCESS_EXEC_METHOD,
                "strace_mode": "process",
                "string_limit": config.string_limit,
            },
            "provenance": {
                "requirements": ["process-trace-artifact"],
                "conflicts": [],
                "intrusive_reason": "strace runs inside the measured container cgroup",
                "scope_includes_descendants": true,
                "timestamp_clock": "CLOCK_REALTIME",
                "timestamp_alignment": "unalignable",
            },
            "extensions": {
                "measurement_domains": ["software.process"],
                "measurement_perturbations": [],
                "invalidates_measurement_domains": ["cpu.capacity", "memory.capacity"],
                "invalidates_metrics": ["cpu.capacity", "memory.capacity"],
            },
        }));
    }
    let profile_mode = if config.coordinated_baseline {
        "baseline"
    } else {
        "profile"
    };
    let verification_failed = verification["requested"] == Value::Bool(true)
        && verification["valid"] != Value::Bool(true);
    let profiler_evidence_status =
        if !cpu_valid || !scheduler_valid || !hardware_valid || !memory_valid || !process_valid {
            "invalid"
        } else if process_exec.is_some() {
            "partial"
        } else {
            "valid"
        };
    let evidence_status = if verification_failed {
        "invalid"
    } else {
        profiler_evidence_status
    };
    let mut evidence_invalid_reasons = if cpu_selected {
        invalid_reasons.clone()
    } else {
        Vec::new()
    };
    evidence_invalid_reasons.extend(memory_reasons.iter().cloned().map(Value::String));
    if !scheduler_valid {
        evidence_invalid_reasons.extend(scheduler_reasons.iter().cloned());
    }
    if !hardware_valid {
        evidence_invalid_reasons.extend(hardware_reasons.iter().cloned());
    }
    if !process_valid {
        evidence_invalid_reasons.extend(process_reasons.iter().cloned());
    }
    if verification_failed {
        evidence_invalid_reasons.extend(
            verification["reasons"]
                .as_array()
                .cloned()
                .unwrap_or_default(),
        );
    }
    let mut requested_profilers = Vec::new();
    if config.proc_stat_comparison {
        requested_profilers.push("proc-stat-host-context");
    }
    if config.memory_profile {
        requested_profilers.push("cgroup-memory");
    }
    if scheduler.is_some() {
        requested_profilers.push("scheduler-concurrency");
    }
    if hardware.is_some() {
        requested_profilers.push("hardware-perf");
    }
    if process_exec.is_some() {
        requested_profilers.push("process-exec");
    }
    let mut clocks = vec![json!({
        "clock_id": "host-monotonic",
        "clock_name": "CLOCK_MONOTONIC",
        "resolution_ns": monotonic_resolution_ns(),
        "namespace_identity": Value::Null,
        "provenance": clock,
    })];
    let mut clock_mappings = Vec::new();
    if process_exec.is_some() {
        clocks.push(json!({
            "clock_id": "strace-realtime",
            "clock_name": "CLOCK_REALTIME",
            "resolution_ns": 1_000,
            "namespace_identity": scope.pointer("/identity/time_namespace").cloned().unwrap_or(Value::Null),
            "provenance": {
                "method": "strace--ttt",
                "timestamp_encoding": "decimal-epoch-seconds-converted-exactly-to-nanoseconds",
            },
        }));
        clock_mappings.push(json!({
            "source_clock_id": "strace-realtime",
            "canonical_clock_id": "host-monotonic",
            "status": "unalignable",
            "method": "no-runtime-clock-correlation",
            "scale_numerator": Value::Null,
            "scale_denominator": Value::Null,
            "offset_ns": Value::Null,
            "max_error_ns": Value::Null,
            "reasons": ["no bounded CLOCK_REALTIME to CLOCK_MONOTONIC correlation was captured"],
            "provenance": {},
        }));
    }
    Ok(json!({
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "run_id": run_id,
        "scope": scope,
        "timeline": lifecycle,
        "clocks": clocks,
        "clock_mappings": clock_mappings,
        "profilers": profilers,
        "streams": streams,
        "artifacts": artifacts,
        "provenance": {
            "execution_plan": plan,
            "execution_plan_sha256": plan_digest,
            "profile_request": {
                "schema_version": "tasktrace-profile-request.v1",
                "mode": profile_mode,
                "profile_sets": config.profile_sets,
                "profilers": requested_profilers,
                "configuration": {
                    "cgroup-cpu": if cpu_selected {
                        json!({"method": "cgroup-v2-cpu.stat", "cpuset_cpus": config.cpuset_cpus})
                    } else {
                        Value::Null
                    },
                    "cgroup-memory": if config.memory_enabled() {
                        json!({"sample_interval_sec": config.memory_sample_interval_sec})
                    } else {
                        Value::Null
                    },
                    "scheduler-concurrency": if scheduler.is_some() {
                        scheduler_configuration(config)
                    } else {
                        Value::Null
                    },
                    "hardware-perf": if hardware.is_some() {
                        json!({
                            "perf_binary": config.hardware_perf_binary,
                            "events": config.hardware_perf_events,
                            "attach_timeout_sec": config.hardware_perf_attach_timeout.as_secs_f64(),
                            "minimum_running_ratio": config.hardware_perf_minimum_running_ratio,
                            "cpuset_cpus": config.cpuset_cpus,
                            "hybrid_pmu_counts_merged": true,
                        })
                    } else {
                        Value::Null
                    },
                    "proc-stat-host-context": if config.proc_stat_comparison {
                        json!({"sample_interval_sec": config.proc_stat_interval_sec, "baseline_sec": config.proc_stat_baseline_sec, "cpuset_cpus": config.cpuset_cpus})
                    } else {
                        Value::Null
                    },
                    "process-exec": if process_exec.is_some() {
                        json!({"method": process_exec::PROCESS_EXEC_METHOD, "strace_mode": "process", "string_limit": config.string_limit})
                    } else {
                        Value::Null
                    },
                },
            },
            "canonical_clock_reference": clock,
            "canonical_lifecycle": lifecycle,
        },
        "extensions": {
            "coordination_status": "valid",
            "coordination_failures": [],
            "run_status": "valid",
            "raw_profiler_reports": {
                "cgroup-cpu": cpu,
                "scheduler-concurrency": scheduler.cloned().unwrap_or(Value::Null),
                "hardware-perf": hardware.cloned().unwrap_or(Value::Null),
                "proc-stat-host-context": proc_stat.cloned().unwrap_or(Value::Null),
                "cgroup-memory": memory.map(|output| output.report.clone()).unwrap_or(Value::Null),
                "process-exec": process_exec.cloned().unwrap_or(Value::Null),
            },
            "runtime_report": runtime_report,
            "workload_verification": verification,
            "evidence_status": evidence_status,
            "evidence_invalid_reasons": evidence_invalid_reasons,
            "all_selected_profilers_valid": profiler_evidence_status == "valid",
            "all_selected_profilers_usable": profiler_evidence_status != "invalid",
            "workload_verification_valid": if verification["requested"] == Value::Bool(true) { verification["valid"].clone() } else { Value::Null },
            "profiler_statuses": {
                "cgroup-cpu": if cpu_selected { cpu_status } else { "not-selected" },
                "scheduler-concurrency": if scheduler.is_some() {
                    if scheduler_valid { "valid" } else { "invalid" }
                } else {
                    "not-selected"
                },
                "hardware-perf": if hardware.is_some() {
                    if hardware_valid { "valid" } else { "invalid" }
                } else {
                    "not-selected"
                },
                "proc-stat-host-context": if config.proc_stat_comparison { "valid" } else { "not-selected" },
                "cgroup-memory": if memory.is_some() { memory_status } else { "not-selected" },
                "process-exec": if process_exec.is_some() {
                    if process_valid { "unalignable" } else { "invalid" }
                } else {
                    "not-selected"
                },
            },
        },
    }))
}

fn network_report_with_enforcement(
    config: &ProfileConfig,
    enroot_enforcement: Option<Value>,
) -> Result<Value, ProfileError> {
    let mut payload = config.network_policy.to_report_payload();
    if let Some(object) = payload.as_object_mut() {
        object.insert(
            "comparability_sha256".to_owned(),
            Value::String(config.network_policy.comparability_digest()),
        );
        let enforcement = if let Some(mut enforcement) = enroot_enforcement {
            if config.runtime != "enroot"
                || config.network_policy.effective_mode() != NetworkMode::NoNetwork
            {
                return Err(ProfileError::new(
                    "Enroot network evidence was supplied for an incompatible runtime policy",
                ));
            }
            let enforcement_object = enforcement.as_object_mut().ok_or_else(|| {
                ProfileError::new("Enroot no-network enforcement evidence is not an object")
            })?;
            enforcement_object.insert(
                "publication".to_owned(),
                json!({
                    "source": "controller-private-temporary-state",
                    "published_after_enroot_exit": true,
                }),
            );
            enforcement
        } else if config.network_policy.effective_mode() == NetworkMode::Allowlist {
            load_network_enforcement_report(&config.output_dir)
                .map_err(|error| ProfileError::new(error.to_string()))?
        } else {
            Value::Null
        };
        object.insert("enforcement".to_owned(), enforcement);
    }
    Ok(payload)
}

fn network_report(config: &ProfileConfig) -> Result<Value, ProfileError> {
    network_report_with_enforcement(config, None)
}

fn legacy_network(config: &ProfileConfig) -> Value {
    match config.network_policy.effective_mode() {
        NetworkMode::NoNetwork => Value::String("none".to_owned()),
        NetworkMode::Public => config
            .network
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
        NetworkMode::Allowlist => Value::String("tasktrace-allowlist".to_owned()),
    }
}

fn memory_report_stream_ids(memory: &Value) -> Vec<String> {
    let mut ids = vec![
        "memory-current".to_owned(),
        "memory-anon".to_owned(),
        "memory-file".to_owned(),
        "memory-kernel".to_owned(),
        "memory-peak".to_owned(),
    ];
    if memory
        .pointer("/limits/memory_max_state")
        .and_then(Value::as_str)
        == Some("finite")
    {
        ids.push("memory-max".to_owned());
    }
    if memory
        .pointer("/observed_limit_margin/remaining_to_limit_bytes")
        .is_some_and(Value::is_number)
    {
        ids.push("memory-observed-limit-margin".to_owned());
    }
    let mut event_ids = memory
        .get("events_delta")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|values| values.keys())
        .map(|name| format!("memory-event-{name}"))
        .collect::<Vec<_>>();
    event_ids.sort();
    ids.extend(event_ids);
    let mut pressure_ids = memory
        .get("pressure_delta")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|values| values.keys())
        .map(|name| format!("memory-pressure-{name}"))
        .collect::<Vec<_>>();
    pressure_ids.sort();
    ids.extend(pressure_ids);
    ids
}

fn scheduler_instrumentation_warning(backend: &str) -> &'static str {
    if backend == "bcc" {
        "This run includes BCC/eBPF scheduler-tracing overhead. Use simultaneous-core fields only when their trace-completeness and cgroup-integral validation passes."
    } else {
        "This run includes perf scheduler-tracing overhead. Use simultaneous-core fields only when their trace-completeness and cgroup-integral validation passes."
    }
}

fn retained_artifact(
    path: &Path,
    artifact_id: &str,
    media_type: &str,
) -> Result<Option<Value>, ProfileError> {
    if !path.is_file() {
        return Ok(None);
    }
    Ok(Some(json!({
        "artifact_id": artifact_id,
        "path": path,
        "media_type": media_type,
        "sha256": sha256_file(path)?,
        "size_bytes": path.metadata()?.len(),
    })))
}

#[allow(clippy::too_many_arguments)]
fn build_profile_report(
    config: &ProfileConfig,
    image_identity: &Value,
    run_id: &str,
    started_at: &str,
    command_exit_code: i32,
    docker_returncode: i32,
    cpu: &Value,
    scheduler: Option<&Value>,
    hardware: Option<&Value>,
    proc_stat: Option<&Value>,
    memory: Option<&Value>,
    process_exec: Option<&Value>,
    clock: &Value,
    lifecycle: &Value,
    evidence_path: Option<&Path>,
    evidence_sha256: Option<&str>,
    plan_digest: &str,
    scope_id: &str,
    verification: &Value,
    scope: &Value,
    plan: &Value,
) -> Result<Value, ProfileError> {
    let memory_valid = memory.map_or(true, |report| {
        report.get("valid_for_observed_memory_reporting") == Some(&Value::Bool(true))
    });
    let scheduler_valid = scheduler.map_or(true, |report| {
        report.pointer("/cgroup_validation/valid_for_concurrency_reporting")
            == Some(&Value::Bool(true))
    });
    let hardware_evidence = hardware
        .map(|report| {
            crate::hardware_perf::build_hardware_perf_evidence(
                run_id,
                scope_id,
                report,
                &config.output_dir,
            )
        })
        .transpose()?;
    let hardware_valid = hardware_evidence
        .as_ref()
        .map_or(true, |evidence| evidence.valid);
    let process_exec_valid = process_exec.map_or(true, |report| {
        report.get("valid") == Some(&Value::Bool(true))
    });
    let cpu_selected = config.cpu_evidence_selected();
    let profiling_valid = (!cpu_selected || cpu["valid_for_cpu_performance"] == Value::Bool(true))
        && scheduler_valid
        && hardware_valid
        && memory_valid
        && process_exec_valid
        && (verification["requested"] != Value::Bool(true)
            || verification["valid"] == Value::Bool(true));
    let trace_path = config.output_dir.join("process.strace.log");
    let process_exec_path = config.output_dir.join("process_exec_report.json");
    let strace_mode = config.effective_strace_mode();
    let raw_trace_artifact = retained_artifact(&trace_path, "process-strace-raw", "text/plain")?;
    let structured_trace_artifact = retained_artifact(
        &process_exec_path,
        "process-exec-report",
        "application/json",
    )?;
    let network_policy = network_report(config)?;
    let mut report = json!({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "runtime": "docker",
        "scope": "disposable-image-container",
        "image": config.image,
        "image_identity": image_identity,
        "container": Value::Null,
        "command": config.command,
        "prepare_command": config.prepare_command,
        "network": legacy_network(config),
        "network_policy": network_policy,
        "shell": config.shell,
        "strace": {
            "mode": strace_mode.as_str(),
            "log_path": if trace_path.is_file() { json!(trace_path) } else { Value::Null },
            "structured_report_path": if process_exec.is_some() { json!(process_exec_path) } else { Value::Null },
            "raw_artifact": raw_trace_artifact,
            "structured_artifact": structured_trace_artifact,
            "string_limit": config.string_limit,
        },
        "process_exec": process_exec.cloned().unwrap_or(Value::Null),
        "command_exit_code": command_exit_code,
        "docker_returncode": docker_returncode,
        "cpu": cpu,
        "scheduler_concurrency": scheduler.cloned().unwrap_or(Value::Null),
        "hardware_perf": hardware.cloned().unwrap_or(Value::Null),
        "memory": memory.cloned().unwrap_or(Value::Null),
        "warning": if scheduler.is_some() {
            scheduler_instrumentation_warning(&config.scheduler_backend)
        } else if hardware.is_some() {
            "This run includes cgroup-scoped perf hardware counters. Unsupported, missing, malformed, scope-mismatched, or multiplexed counters remain explicit invalid evidence and are never reported as zero."
        } else if proc_stat.is_some() && memory.is_some() {
            "This run includes exact cgroup charged-memory gates and sampled host /proc/stat context. The memory peak is observed charge, not process RSS; host active CPUs are not task peak concurrency."
        } else if memory.is_some() {
            "This run reports cgroup-v2 charged memory through the controlled finish gate. memory.peak is observed charge, not process RSS, and observed limit margin is not a sizing recommendation."
        } else if proc_stat.is_some() {
            "This run includes sampled host /proc/stat context on the allocated CPUs. Use cgroup fields for task average CPU; sampled active CPUs are not task peak concurrency."
        } else if strace_mode != StraceMode::None {
            "This diagnostic run includes strace overhead. Compare provider performance using identically configured instrumentation or use --strace-mode none for an untraced CPU baseline."
        } else {
            "This is an untraced CPU baseline; no process launch log was collected."
        },
        "run_id": run_id,
        "profile_started_at_utc": started_at,
    });
    if let Some(proc_stat) = proc_stat {
        report
            .as_object_mut()
            .expect("profile report object")
            .insert("proc_stat_comparison".to_owned(), proc_stat.clone());
    }
    if let (Some(path), Some(digest)) = (evidence_path, evidence_sha256) {
        let cpu_profiler_reasons = cpu["invalid_reasons"].clone();
        let manifest_status = if cpu["valid_for_cpu_performance"] == Value::Bool(true) {
            "valid"
        } else {
            "invalid"
        };
        let mode = if config.coordinated_baseline {
            "baseline"
        } else {
            "profile"
        };
        let profiler_evidence_status = if (cpu_selected
            && cpu["valid_for_cpu_performance"] != Value::Bool(true))
            || !scheduler_valid
            || !hardware_valid
            || !memory_valid
            || !process_exec_valid
        {
            "invalid"
        } else if process_exec.is_some() {
            "partial"
        } else {
            "valid"
        };
        let evidence_status = if verification["requested"] == Value::Bool(true)
            && verification["valid"] != Value::Bool(true)
        {
            "invalid"
        } else {
            profiler_evidence_status
        };
        let mut resolved_profiler_ids = Vec::new();
        let mut profiler_rows = Vec::new();
        let mut manifest_rows = Vec::new();
        if cpu_selected {
            resolved_profiler_ids.push(Value::String("cgroup-cpu".to_owned()));
            profiler_rows.push(json!({
                "profiler_id": "cgroup-cpu",
                "profiler_version": PROFILER_VERSION,
                "status": manifest_status,
                "reasons": cpu["invalid_reasons"],
                "stream_ids": ["cpu-average-logical-cores", "cpu-interval-average", "cpu-system", "cpu-usage", "cpu-user", "cpu-wall"],
                "intrusive": false,
                "configuration": {"method": "cgroup-v2-cpu.stat", "cpuset_cpus": config.cpuset_cpus},
                "provenance": {"scope_includes_descendants": true, "cpu_policy_gate_scope": "leaf-and-every-ancestor-to-cgroup-root"},
            }));
            manifest_rows.push(json!({
                "profiler_id": "cgroup-cpu",
                "profiler_version": PROFILER_VERSION,
                "status": manifest_status,
                "stream_ids": ["cpu-average-logical-cores", "cpu-interval-average", "cpu-system", "cpu-usage", "cpu-user", "cpu-wall"],
                "reasons": cpu["invalid_reasons"],
                "intrusive": false,
            }));
        }
        if let Some(scheduler) = scheduler {
            let status = if scheduler_valid { "valid" } else { "invalid" };
            let reasons = scheduler_invalid_reasons(scheduler);
            let stream_ids = [
                "cpu-peak-simultaneous",
                "cpu-scheduled-runtime",
                "scheduler-concurrency",
                "scheduler-per-core",
            ];
            resolved_profiler_ids.push(Value::String("scheduler-concurrency".to_owned()));
            profiler_rows.push(json!({
                "profiler_id": "scheduler-concurrency",
                "profiler_version": PROFILER_VERSION,
                "status": status,
                "reasons": reasons,
                "stream_ids": stream_ids,
                "intrusive": false,
                "configuration": scheduler_configuration(config),
                "provenance": scheduler_provenance(config),
            }));
            manifest_rows.push(json!({
                "profiler_id": "scheduler-concurrency",
                "profiler_version": PROFILER_VERSION,
                "status": status,
                "stream_ids": stream_ids,
                "reasons": reasons,
                "intrusive": false,
            }));
        }
        if let Some(evidence) = &hardware_evidence {
            resolved_profiler_ids.push(Value::String("hardware-perf".to_owned()));
            profiler_rows.push(evidence.profiler.clone());
            manifest_rows.push(json!({
                "profiler_id": "hardware-perf",
                "profiler_version": crate::hardware_perf::HARDWARE_PERF_SCHEMA_VERSION,
                "status": if evidence.valid { "valid" } else { "invalid" },
                "stream_ids": evidence.stream_ids,
                "reasons": evidence.reasons,
                "intrusive": false,
                "artifacts": evidence.artifacts,
            }));
        }
        if proc_stat.is_some() {
            resolved_profiler_ids.push(Value::String("proc-stat-host-context".to_owned()));
            profiler_rows.push(json!({
                "profiler_id": "proc-stat-host-context",
                "profiler_version": "tasktrace-proc-stat-comparison.v1",
                "status": "valid",
                "reasons": [],
                "stream_ids": [
                    "proc-stat-baseline-average-busy",
                    "proc-stat-measurement-average-busy",
                    "proc-stat-baseline-subtracted-average-busy",
                    "proc-stat-measurement-peak-active",
                ],
                "intrusive": false,
                "configuration": {
                    "sample_interval_sec": config.proc_stat_interval_sec,
                    "baseline_sec": config.proc_stat_baseline_sec,
                    "cpuset_cpus": config.cpuset_cpus,
                },
                "provenance": {
                    "scope": "host pressure projected onto allocated logical CPUs",
                    "task_attributed": false,
                    "peak_is_not_task_concurrency": true,
                },
            }));
            manifest_rows.push(json!({
                "profiler_id": "proc-stat-host-context",
                "profiler_version": "tasktrace-proc-stat-comparison.v1",
                "status": "valid",
                "stream_ids": [
                    "proc-stat-baseline-average-busy",
                    "proc-stat-measurement-average-busy",
                    "proc-stat-baseline-subtracted-average-busy",
                    "proc-stat-measurement-peak-active",
                ],
                "reasons": [],
                "intrusive": false,
            }));
        }
        if let Some(memory) = memory {
            let memory_reasons = memory_invalid_reasons(memory);
            let memory_status = if memory_valid { "valid" } else { "invalid" };
            let memory_stream_ids = memory_report_stream_ids(memory);
            resolved_profiler_ids.push(Value::String("cgroup-memory".to_owned()));
            profiler_rows.push(json!({
                "profiler_id": "cgroup-memory",
                "profiler_version": PROFILER_VERSION,
                "status": memory_status,
                "reasons": memory_reasons,
                "stream_ids": memory_stream_ids,
                "intrusive": false,
                "configuration": {
                    "method": "cgroup-v2-memory-and-cpu-stat-sampling",
                    "sample_interval_sec": config.memory_sample_interval_sec,
                },
                "provenance": {
                    "scope_includes_descendants": true,
                    "ancestor_chain_observed": true,
                    "peak_source": "memory.peak",
                    "observed_margin_is_not_a_recommendation": true,
                },
            }));
            manifest_rows.push(json!({
                "profiler_id": "cgroup-memory",
                "profiler_version": PROFILER_VERSION,
                "status": memory_status,
                "stream_ids": memory_stream_ids,
                "reasons": memory_reasons,
                "intrusive": false,
            }));
        }
        if let Some(process_exec) = process_exec {
            let reasons = process_exec_reasons(process_exec);
            let status = if process_exec_valid {
                "unalignable"
            } else {
                "invalid"
            };
            let mut stream_ids = vec!["process-count", "binary-count"];
            let has_events = process_exec
                .get("processes")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|process| {
                    process
                        .get("execs")
                        .and_then(Value::as_array)
                        .is_some_and(|events| !events.is_empty())
                });
            if has_events {
                stream_ids.push("process-exec-events");
            }
            let manifest_reasons = if process_exec_valid {
                vec![Value::String(
                    "strace event timestamps cannot be placed on the canonical monotonic axis"
                        .to_owned(),
                )]
            } else {
                reasons
            };
            resolved_profiler_ids.push(Value::String("process-exec".to_owned()));
            profiler_rows.push(json!({
                "profiler_id": "process-exec",
                "profiler_version": PROFILER_VERSION,
                "status": status,
                "reasons": manifest_reasons,
                "stream_ids": stream_ids,
                "intrusive": true,
                "configuration": {
                    "method": process_exec::PROCESS_EXEC_METHOD,
                    "strace_mode": "process",
                    "string_limit": config.string_limit,
                },
                "provenance": {
                    "requirements": ["process-trace-artifact"],
                    "intrusive_reason": "strace runs inside the measured container cgroup",
                    "timestamp_alignment": "unalignable",
                },
            }));
            manifest_rows.push(json!({
                "profiler_id": "process-exec",
                "profiler_version": PROFILER_VERSION,
                "status": status,
                "stream_ids": stream_ids,
                "reasons": manifest_reasons,
                "intrusive": true,
            }));
        }
        let mut profiling_invalid_reasons = Vec::new();
        if cpu_selected && cpu["valid_for_cpu_performance"] != Value::Bool(true) {
            profiling_invalid_reasons.push(json!({
                "profiler_id": "cgroup-cpu",
                "status": manifest_status,
                "reasons": cpu_profiler_reasons,
            }));
        }
        if let Some(scheduler) = scheduler {
            if !scheduler_valid {
                profiling_invalid_reasons.push(json!({
                    "profiler_id": "scheduler-concurrency",
                    "status": "invalid",
                    "reasons": scheduler_invalid_reasons(scheduler),
                }));
            }
        }
        if let Some(evidence) = &hardware_evidence {
            if !evidence.valid {
                profiling_invalid_reasons.push(json!({
                    "profiler_id": "hardware-perf",
                    "status": "invalid",
                    "reasons": evidence.reasons,
                }));
            }
        }
        if let Some(memory) = memory {
            if !memory_valid {
                profiling_invalid_reasons.push(json!({
                    "profiler_id": "cgroup-memory",
                    "status": "invalid",
                    "reasons": memory_invalid_reasons(memory),
                }));
            }
        }
        if let Some(process_exec) = process_exec {
            if !process_exec_valid {
                profiling_invalid_reasons.push(json!({
                    "profiler_id": "process-exec",
                    "status": "invalid",
                    "reasons": process_exec_reasons(process_exec),
                }));
            }
        }
        if verification["requested"] == Value::Bool(true)
            && verification["valid"] != Value::Bool(true)
        {
            profiling_invalid_reasons.push(json!({
                "profiler_id": "workload-verification",
                "status": "invalid",
                "reasons": verification["reasons"],
            }));
        }
        let profiler_order = |profiler_id: &str| match profiler_id {
            "scheduler-concurrency" => 100,
            "hardware-perf" => 150,
            "proc-stat-host-context" => 175,
            "cgroup-cpu" => 200,
            "cgroup-memory" => 300,
            "process-exec" => 400,
            _ => 1_000,
        };
        resolved_profiler_ids
            .sort_by_key(|value| profiler_order(value.as_str().unwrap_or_default()));
        profiler_rows.sort_by_key(|value| {
            profiler_order(
                value
                    .get("profiler_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
        });
        manifest_rows.sort_by_key(|value| {
            profiler_order(
                value
                    .get("profiler_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
            )
        });
        let object = report.as_object_mut().expect("report object");
        object.insert("canonical_clock_reference".to_owned(), clock.clone());
        object.insert("canonical_lifecycle".to_owned(), lifecycle.clone());
        object.insert("verification".to_owned(), verification.clone());
        object.insert("profiling_valid".to_owned(), Value::Bool(profiling_valid));
        object.insert(
            "profiling_invalid_reasons".to_owned(),
            Value::Array(profiling_invalid_reasons),
        );
        object.insert(
            "evidence_bundle".to_owned(),
            json!({
                "path": path,
                "sha256": digest,
                "run_id": run_id,
                "scope_id": scope_id,
                "profile_sets": config.profile_sets,
                "mode": mode,
                "coordination_status": "valid",
                "execution_plan_digest": plan_digest,
                "scope": scope,
                "timeline": lifecycle,
                "provenance": {
                    "execution_plan": plan,
                    "execution_plan_sha256": plan_digest,
                    "canonical_clock_reference": clock,
                    "canonical_lifecycle": lifecycle,
                },
                "evidence_status": evidence_status,
                "all_selected_profilers_valid": profiler_evidence_status == "valid",
                "workload_verification_valid": if verification["requested"] == Value::Bool(true) { verification["valid"].clone() } else { Value::Null },
                "resolved_profiler_ids": resolved_profiler_ids,
                "intrusive_profiler_ids": if process_exec.is_some() { vec!["process-exec"] } else { Vec::<&str>::new() },
                "profilers": profiler_rows,
                "profiler_manifests": manifest_rows,
            }),
        );
    }
    Ok(report)
}

#[derive(Clone, Debug)]
pub struct ProfileRun {
    pub report: Value,
    pub report_path: PathBuf,
    pub evidence_bundle_path: Option<PathBuf>,
    pub command_exit_code: i32,
}

fn read_enroot_control(path: &Path, label: &str) -> Result<String, ProfileError> {
    fs::read_to_string(path)
        .map(|value| value.trim().to_owned())
        .map_err(|error| {
            ProfileError::new(format!(
                "unable to read Enroot {label} evidence {}: {error}",
                path.display()
            ))
        })
}

fn read_exact_enroot_gate_control(
    output_dir: &Path,
    name: &str,
    expected: &[u8],
) -> Result<(), ProfileError> {
    let path = output_dir.join(name);
    let metadata = fs::symlink_metadata(&path).map_err(|error| {
        ProfileError::new(format!(
            "unable to read Enroot run-wave gate artifact {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() || metadata.nlink() != 1 {
        return Err(ProfileError::new(format!(
            "Enroot run-wave gate artifact must be a regular single-link file: {}",
            path.display()
        )));
    }
    if fs::read(&path)? != expected {
        return Err(ProfileError::new(format!(
            "Enroot run-wave gate artifact has invalid contents: {}",
            path.display()
        )));
    }
    Ok(())
}

fn enroot_run_wave_start_gate_report(
    runtime: &crate::enroot_runtime::EnrootRuntimeConfig,
    measured_start_ns: u64,
    clock: &Value,
) -> Result<Value, ProfileError> {
    if !runtime.run_wave_start_gate {
        return Ok(json!({
            "schema_version": "tasktrace-run-wave-start-gate.v1",
            "requested": false,
            "state": "not-requested",
            "ready": false,
            "released": false,
            "valid": Value::Null,
            "released_ns": Value::Null,
            "measured_start_ns": measured_start_ns,
            "release_to_start_ns": Value::Null,
            "clock_reference": clock,
        }));
    }

    read_exact_enroot_gate_control(&runtime.output_dir, "run_wave_start_ready", b"ready\n")?;
    read_exact_enroot_gate_control(&runtime.output_dir, "run_wave_start_release", b"release\n")?;
    let released_path = runtime.output_dir.join("run_wave_start_released_ns");
    let released_metadata = fs::symlink_metadata(&released_path)?;
    if released_metadata.file_type().is_symlink()
        || !released_metadata.is_file()
        || released_metadata.nlink() != 1
    {
        return Err(ProfileError::new(format!(
            "Enroot run-wave release timestamp must be a regular single-link file: {}",
            released_path.display()
        )));
    }
    let released_raw = fs::read(&released_path)?;
    let released_digits = released_raw.strip_suffix(b"\n").ok_or_else(|| {
        ProfileError::new("Enroot run-wave release timestamp must end in one newline")
    })?;
    if released_digits.is_empty() || !released_digits.iter().all(|byte| byte.is_ascii_digit()) {
        return Err(ProfileError::new(
            "Enroot run-wave release timestamp must contain decimal nanoseconds",
        ));
    }
    let released_ns = std::str::from_utf8(released_digits)
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .ok_or_else(|| ProfileError::new("invalid Enroot run-wave release timestamp"))?;
    if released_ns > measured_start_ns {
        return Err(ProfileError::new(format!(
            "Enroot run-wave release timestamp follows measured start: released={released_ns}, start={measured_start_ns}"
        )));
    }
    Ok(json!({
        "schema_version": "tasktrace-run-wave-start-gate.v1",
        "requested": true,
        "state": "released",
        "ready": true,
        "released": true,
        "valid": true,
        "released_ns": released_ns,
        "measured_start_ns": measured_start_ns,
        "release_to_start_ns": measured_start_ns - released_ns,
        "clock_reference": clock,
    }))
}

fn enroot_task_clock_reference(output_dir: &Path) -> Result<Value, ProfileError> {
    let source = read_enroot_control(&output_dir.join("clock_source"), "clock source")?;
    let clock_name = read_enroot_control(&output_dir.join("clock_name"), "clock name")?;
    if source != "linux-proc-uptime" || clock_name != "CLOCK_BOOTTIME" {
        return Err(ProfileError::new(format!(
            "unsupported Enroot wall clock: source={source:?}, clock_name={clock_name:?}"
        )));
    }
    let resolution_ns =
        read_enroot_control(&output_dir.join("clock_resolution_ns"), "clock resolution")?
            .parse::<u64>()
            .ok()
            .filter(|value| *value > 0)
            .ok_or_else(|| ProfileError::new("invalid Enroot task clock resolution"))?;
    let boot_id = read_enroot_control(&output_dir.join("clock_boot_id"), "boot ID")?;
    let time_namespace =
        read_enroot_control(&output_dir.join("clock_time_namespace"), "time namespace")?;
    let raw_offset = read_enroot_control(
        &output_dir.join("clock_time_namespace_boottime_offset"),
        "time-namespace boottime offset",
    )?;
    let boottime_offset_ns = if raw_offset.is_empty() {
        None
    } else {
        let fields: Vec<_> = raw_offset.split_whitespace().collect();
        if fields.len() != 2 {
            return Err(ProfileError::new(
                "invalid Enroot time-namespace boottime offset",
            ));
        }
        let seconds = fields[0]
            .parse::<i64>()
            .map_err(|_| ProfileError::new("invalid Enroot time-namespace boottime offset"))?;
        let nanoseconds = fields[1]
            .parse::<i64>()
            .map_err(|_| ProfileError::new("invalid Enroot time-namespace boottime offset"))?;
        if !(0..1_000_000_000).contains(&nanoseconds) {
            return Err(ProfileError::new(
                "invalid Enroot time-namespace boottime offset",
            ));
        }
        Some(
            seconds
                .checked_mul(1_000_000_000)
                .and_then(|value| value.checked_add(nanoseconds))
                .ok_or_else(|| {
                    ProfileError::new("Enroot time-namespace boottime offset overflow")
                })?,
        )
    };
    Ok(json!({
        "clock_id": "enroot-task-boottime",
        "clock_name": clock_name,
        "source": source,
        "timestamp_unit": "nanoseconds",
        "resolution_ns": resolution_ns,
        "storage_precision_ns": 1,
        "boot_id": if boot_id.is_empty() { Value::Null } else { Value::String(boot_id) },
        "time_namespace": if time_namespace.is_empty() { Value::Null } else { Value::String(time_namespace) },
        "time_namespace_boottime_offset_ns": boottime_offset_ns,
        "accuracy_claim": Value::Null,
    }))
}

fn enroot_cpu_report(
    result: &crate::enroot_runtime::EnrootRunResult,
    runtime: &crate::enroot_runtime::EnrootRuntimeConfig,
) -> Result<(Value, Value, Value), ProfileError> {
    let task_clock = enroot_task_clock_reference(&runtime.output_dir)?;
    let start_ns = read_enroot_control(&runtime.output_dir.join("start_ns"), "task start")?
        .parse::<u64>()
        .map_err(|_| ProfileError::new("invalid Enroot task start timestamp"))?;
    let end_ns = read_enroot_control(&runtime.output_dir.join("end_ns"), "task end")?
        .parse::<u64>()
        .map_err(|_| ProfileError::new("invalid Enroot task end timestamp"))?;
    if end_ns <= start_ns {
        return Err(ProfileError::new(
            "Enroot task clock did not advance across the replay command",
        ));
    }
    let initial_affinity = read_enroot_control(
        &runtime.output_dir.join("initial_effective_cpu_affinity"),
        "initial effective CPU affinity",
    )?;
    let initial_affinity_ids = crate::enroot_runtime::parse_cpuset_cpu_ids(&initial_affinity)?;
    let wall_sec = (end_ns - start_ns) as f64 / 1_000_000_000.0;
    let mut report = result.wall_time_cpu_report(runtime)?;
    let object = report
        .as_object_mut()
        .expect("Enroot wall-time report is an object");
    object.insert("wall_sec".to_owned(), json!(wall_sec));
    object.insert("window_start_ns".to_owned(), json!(start_ns));
    object.insert("window_end_ns".to_owned(), json!(end_ns));
    object.insert(
        "clock_id".to_owned(),
        Value::String("enroot-task-boottime".to_owned()),
    );
    object.insert(
        "clock_source".to_owned(),
        Value::String("linux-proc-uptime".to_owned()),
    );
    object.insert(
        "clock_resolution_ns".to_owned(),
        task_clock
            .get("resolution_ns")
            .cloned()
            .unwrap_or(Value::Null),
    );
    object.insert(
        "timing_boundary".to_owned(),
        Value::String("measured-replay-command".to_owned()),
    );
    object.insert(
        "wall_clock_source".to_owned(),
        Value::String("linux-proc-uptime".to_owned()),
    );
    object.insert("clock_reference".to_owned(), task_clock.clone());
    object.insert(
        "requested_cpu_affinity".to_owned(),
        runtime
            .cpuset_cpus
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    object.insert(
        "initial_effective_cpu_affinity".to_owned(),
        Value::String(initial_affinity),
    );
    object.insert(
        "initial_effective_logical_cpu_ids".to_owned(),
        json!(initial_affinity_ids),
    );
    object.insert("cpu_affinity_enforced".to_owned(), Value::Bool(false));
    object.insert(
        "schema_version".to_owned(),
        Value::String("wall-clock-only-no-cgroup.v1".to_owned()),
    );
    object.insert("user_sec".to_owned(), Value::Null);
    object.insert("system_sec".to_owned(), Value::Null);
    object.insert("average_logical_cores".to_owned(), Value::Null);
    object.insert(
        "allocated_cpu_source".to_owned(),
        if runtime.cpuset_cpus.is_some() {
            Value::String("taskset-requested-affinity".to_owned())
        } else if runtime.allocated_cpus.is_some() {
            Value::String("user-supplied-visible-capacity".to_owned())
        } else {
            Value::Null
        },
    );
    object.insert("enforced_logical_cpu_limit".to_owned(), Value::Null);
    object.insert("enforced_cpu_limit_source".to_owned(), Value::Null);
    object.insert(
        "valid_for_wall_time_reporting".to_owned(),
        Value::Bool(true),
    );
    object.insert("valid_for_cpu_performance".to_owned(), Value::Bool(false));
    object.insert(
        "invalid_reasons".to_owned(),
        json!(["rootless Enroot exposes no task-scoped cgroup CPU accounting"]),
    );
    object.insert(
        "measurement_domain_contract".to_owned(),
        json!({
            "measurement_domains": ["wall.time"],
            "unavailable_domains": ["cpu.capacity", "memory.capacity", "scheduler.concurrency"],
            "perturbations": [],
            "status": "wall-time-only",
        }),
    );
    let run_wave_start_gate = enroot_run_wave_start_gate_report(runtime, start_ns, &task_clock)?;
    Ok((report, task_clock, run_wave_start_gate))
}

#[allow(clippy::too_many_arguments)]
fn build_enroot_profile_report(
    config: &ProfileConfig,
    image_identity: &Value,
    run_id: &str,
    started_at: &str,
    command_exit_code: i32,
    runtime_returncode: i32,
    cpu: &Value,
    clock: &Value,
    lifecycle: &Value,
    plan: &Value,
    plan_digest: &str,
    scope: &Value,
    runtime_report: &Value,
    enroot_lifecycle: &Value,
    network_enforcement: Option<Value>,
    run_wave_start_gate: &Value,
    verification: &Value,
) -> Result<Value, ProfileError> {
    let report_scope = match image_identity.get("kind").and_then(Value::as_str) {
        Some("image-file") => "disposable-image-container",
        Some("created-container") => "caller-owned-enroot-container",
        Some(kind) => {
            return Err(ProfileError::new(format!(
                "unsupported Enroot image identity kind in profile report: {kind:?}"
            )))
        }
        None => {
            return Err(ProfileError::new(
                "Enroot image identity has no string kind",
            ))
        }
    };
    Ok(json!({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "runtime": "enroot",
        "scope": report_scope,
        "measurement_scope": scope,
        "image": config.image,
        "image_identity": image_identity,
        "container": Value::Null,
        "command": config.command,
        "prepare_command": config.prepare_command,
        "network": legacy_network(config),
        "network_policy": network_report_with_enforcement(config, network_enforcement)?,
        "shell": config.shell,
        "strace": {
            "mode": "none",
            "log_path": Value::Null,
            "structured_report_path": Value::Null,
            "raw_artifact": Value::Null,
            "structured_artifact": Value::Null,
            "string_limit": config.string_limit,
        },
        "process_exec": Value::Null,
        "command_exit_code": command_exit_code,
        "docker_returncode": runtime_returncode,
        "enroot_returncode": runtime_returncode,
        "cpu": cpu,
        "scheduler_concurrency": Value::Null,
        "hardware_perf": Value::Null,
        "memory": Value::Null,
        "warning": "The Enroot backend is wall-time-only. CPU, memory, scheduler-concurrency, and process evidence are unavailable; taskset is requested launch affinity, not CPU containment or exclusivity.",
        "run_id": run_id,
        "profile_started_at_utc": started_at,
        "canonical_clock_reference": clock,
        "canonical_lifecycle": lifecycle,
        "verification": verification,
        "profiling_valid": true,
        "profiling_invalid_reasons": [],
        "measurement_validity": {
            "valid_for_wall_time_reporting": true,
            "valid_for_average_cpu_reporting": false,
            "valid_for_peak_cpu_reporting": false,
            "valid_for_memory_reporting": false,
            "valid_for_process_reporting": false,
            "reasons": ["rootless Enroot exposes no task-scoped cgroup resource accounting"],
        },
        "execution_plan": plan,
        "execution_plan_sha256": plan_digest,
        "runtime_report": runtime_report,
        "enroot_lifecycle": enroot_lifecycle,
        "run_wave_start_gate": run_wave_start_gate,
    }))
}

fn enroot_lifecycle_report(
    result: &crate::enroot_runtime::EnrootRunResult,
    clock: &Value,
) -> Value {
    let mut phases = Vec::new();
    if let Some(timing) = &result.create_timing {
        phases.push(json!({
            "phase": "create",
            "start_ns": timing.start_monotonic_ns,
            "end_ns": timing.end_monotonic_ns,
            "duration_sec": timing.duration_sec,
            "returncode": 0,
        }));
    }
    phases.push(json!({
        "phase": "start",
        "start_ns": result.start_timing.start_monotonic_ns,
        "end_ns": result.start_timing.end_monotonic_ns,
        "duration_sec": result.start_timing.duration_sec,
        "returncode": result.exit_code,
    }));
    if let Some(timing) = &result.cleanup_timing {
        phases.push(json!({
            "phase": "remove",
            "start_ns": timing.start_monotonic_ns,
            "end_ns": timing.end_monotonic_ns,
            "duration_sec": timing.duration_sec,
            "returncode": result.cleanup.as_ref().and_then(|value| value.remove_status),
        }));
    }
    if let Some(timing) = &result.cleanup_verification_timing {
        phases.push(json!({
            "phase": "list",
            "start_ns": timing.start_monotonic_ns,
            "end_ns": timing.end_monotonic_ns,
            "duration_sec": timing.duration_sec,
            "returncode": result.cleanup.as_ref().and_then(|value| value.verification_status),
        }));
    }
    let window_start_ns = phases
        .first()
        .and_then(|phase| phase.get("start_ns"))
        .cloned()
        .unwrap_or(Value::Null);
    let window_end_ns = phases
        .last()
        .and_then(|phase| phase.get("end_ns"))
        .cloned()
        .unwrap_or(Value::Null);
    let cleanup = result.cleanup.as_ref().map(|value| {
        json!({
            "remove_returncode": value.remove_status,
            "list_returncode": value.verification_status,
            "container_absent": value.verified_absent,
            "valid": value.succeeded && value.verified_absent,
            "errors": value.reasons,
        })
    });
    json!({
        "schema_version": "tasktrace-enroot-lifecycle.v1",
        "clock_reference": clock,
        "process_containment": {
            "backend": "linux-user-pid-namespace-v1",
            "namespace_init_exit_kills_remaining_processes": true,
            "outer_exit_kills_namespace_init": true,
        },
        "container_name": result.owned_container_name,
        "container_created": result.owned_container_name.is_some(),
        "window_start_ns": window_start_ns,
        "window_end_ns": window_end_ns,
        "phases": phases,
        "cleanup": cleanup,
    })
}

fn run_enroot_profile(
    mut config: ProfileConfig,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<ProfileRun, ProfileError> {
    preflight_output(&config.output_dir)?;
    let harness = HarnessFile::create()?;
    let network_evidence_path = (config.network_policy.effective_mode() == NetworkMode::NoNetwork)
        .then(|| harness.directory.join("enroot_network_enforcement.json"));
    let static_runtime =
        enroot_runtime_config(&config, harness.path.clone(), network_evidence_path.clone());
    let static_preflight = crate::enroot_runtime::preflight(&static_runtime)?;
    if cancellation_requested(cancel_requested) {
        return Err(ProfileError::new(
            "profile cancelled before output creation",
        ));
    }
    let (output_dir, _output_claim) = create_output(&config.output_dir)?;
    config.output_dir = output_dir.clone();
    let runtime = enroot_runtime_config(&config, harness.path.clone(), network_evidence_path);
    let profile_run_id = run_id()?;
    let profile_started_at = utc_now()?;
    let executable = env::current_exe()?;
    let controller_digest = sha256_file(&executable)?;
    let image = serde_json::to_value(static_preflight.image_identity)?;
    let (plan, plan_digest) = execution_plan(&config, &image, &controller_digest)?;
    let cancellation = || cancellation_requested(cancel_requested);
    let result = crate::enroot_runtime::run_with_cancel(&runtime, Some(&cancellation))?;
    let observed_image = serde_json::to_value(&result.image_identity)?;
    if observed_image != image {
        return Err(ProfileError::new(
            "Enroot image identity changed between preflight and execution",
        ));
    }
    let runtime_returncode = result
        .exit_code
        .unwrap_or_else(|| 128 + result.signal.unwrap_or(0));
    let command_exit_code = parse_exit_artifact(&output_dir)?;
    let verification = verification_report(&config, &output_dir)?;
    let (cpu, task_clock, run_wave_start_gate) = enroot_cpu_report(&result, &runtime)?;
    let run_start_ns = result
        .create_timing
        .as_ref()
        .map_or(result.start_timing.start_monotonic_ns, |timing| {
            timing.start_monotonic_ns
        });
    let run_end_ns = result
        .cleanup_verification_timing
        .as_ref()
        .or(result.cleanup_timing.as_ref())
        .map_or(result.start_timing.end_monotonic_ns, |timing| {
            timing.end_monotonic_ns
        });
    if run_end_ns <= run_start_ns {
        return Err(ProfileError::new(
            "Enroot lifecycle is not a positive monotonic interval",
        ));
    }
    let mut markers = vec![marker("run_start", run_start_ns)];
    if let Some(timing) = &result.create_timing {
        markers.push(marker("enroot_create_start", timing.start_monotonic_ns));
        markers.push(marker("enroot_create_end", timing.end_monotonic_ns));
    }
    markers.push(marker(
        "enroot_start_begin",
        result.start_timing.start_monotonic_ns,
    ));
    markers.push(marker(
        "enroot_start_end",
        result.start_timing.end_monotonic_ns,
    ));
    if let Some(timing) = &result.cleanup_timing {
        markers.push(marker("enroot_remove_start", timing.start_monotonic_ns));
        markers.push(marker("enroot_remove_end", timing.end_monotonic_ns));
    }
    if let Some(timing) = &result.cleanup_verification_timing {
        markers.push(marker(
            "enroot_cleanup_verify_start",
            timing.start_monotonic_ns,
        ));
        markers.push(marker("enroot_cleanup_verify_end", timing.end_monotonic_ns));
    }
    markers.push(marker("run_end", run_end_ns));
    let clock = task_clock;
    let lifecycle = json!({
        "clock_id": "enroot-task-boottime",
        "clock_name": result.start_timing.clock_source,
        "clock_resolution_ns": result.start_timing.clock_resolution_ns,
        "origin_ns": result.start_timing.start_monotonic_ns,
        "start_ns": run_start_ns,
        "end_ns": run_end_ns,
        "measured_wall_start_ns": result.start_timing.start_monotonic_ns,
        "measured_wall_end_ns": result.start_timing.end_monotonic_ns,
        "measured_wall_boundary": result.start_timing.boundary,
        "lifecycle_markers": markers,
    });
    let scope_identity = json!({
        "runtime": "enroot",
        "run_id": profile_run_id,
        "image_identity": image,
        "owned_container_name": result.owned_container_name,
        "requested_cpuset_cpus": config.cpuset_cpus,
        "requested_allocated_cpus": config.allocated_cpus,
        "cpu_containment_enforced": false,
    });
    let scope_id = format!("enroot-wall-{}", &canonical_digest(&scope_identity)?[..24]);
    let scope = json!({
        "scope_id": scope_id,
        "kind": "enroot-start-process-wall-window",
        "identity": scope_identity,
        "includes_descendants": false,
        "description": "Suspend-aware host CLOCK_BOOTTIME lifetime of `enroot start`; no task-owned cgroup resource scope is available.",
        "extensions": {
            "canonical_clock": "CLOCK_BOOTTIME",
            "measurement_method": crate::enroot_runtime::ENROOT_WALL_TIME_METHOD,
            "requested_affinity_only": config.cpuset_cpus,
            "containment_enforced": false,
            "exclusivity_enforced": false,
        },
    });
    let runtime_report = serde_json::to_value(&result)?;
    let enroot_lifecycle = enroot_lifecycle_report(&result, &clock);
    let network_enforcement = result
        .network_evidence
        .as_ref()
        .map(serde_json::to_value)
        .transpose()?;
    let report = build_enroot_profile_report(
        &config,
        &image,
        &profile_run_id,
        &profile_started_at,
        command_exit_code,
        runtime_returncode,
        &cpu,
        &clock,
        &lifecycle,
        &plan,
        &plan_digest,
        &scope,
        &runtime_report,
        &enroot_lifecycle,
        network_enforcement,
        &run_wave_start_gate,
        &verification,
    )?;
    let report_path = output_dir.join("profile_report.json");
    write_json(&report_path, &report)?;
    Ok(ProfileRun {
        report,
        report_path,
        evidence_bundle_path: None,
        command_exit_code,
    })
}

fn exit_code(status: ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

fn publish_task_artifact(
    exchange: &TaskExchange,
    output_dir: &Path,
    name: &str,
    required: bool,
) -> Result<bool, ProfileError> {
    let limit = if matches!(
        name,
        "exit_code"
            | "strace_mode"
            | "prepare_exit_code"
            | "verification_exit_code"
            | "verification_status"
    ) {
        MAX_TASK_CONTROL_BYTES
    } else {
        MAX_TASK_LOG_BYTES
    };
    let published = exchange.copy_regular_to(name, &output_dir.join(name), limit)?;
    if !published {
        if required {
            return Err(ProfileError::new(format!(
                "task did not retain required regular artifact: {name}"
            )));
        }
        return Ok(false);
    }
    Ok(true)
}

fn publish_measurement_control_artifacts(
    exchange: &TaskExchange,
    output_dir: &Path,
) -> Result<(), ProfileError> {
    for name in ["exit_code", "strace_mode"] {
        publish_task_artifact(exchange, output_dir, name, true)?;
    }
    Ok(())
}

fn publish_measurement_payload_artifacts(
    exchange: &TaskExchange,
    output_dir: &Path,
) -> Result<(), ProfileError> {
    for name in ["command.stdout.txt", "command.stderr.txt"] {
        publish_task_artifact(exchange, output_dir, name, true)?;
    }
    for name in [
        "prepare.stdout.txt",
        "prepare.stderr.txt",
        "prepare_exit_code",
        "process.strace.log",
    ] {
        publish_task_artifact(exchange, output_dir, name, false)?;
    }
    Ok(())
}

fn publish_verification_artifacts(
    config: &ProfileConfig,
    exchange: &TaskExchange,
    output_dir: &Path,
) -> Result<(), ProfileError> {
    if config.verification_command.is_none() {
        return Ok(());
    }
    for name in [
        "verification.stdout.txt",
        "verification.stderr.txt",
        "verification_exit_code",
        "verification_status",
    ] {
        publish_task_artifact(exchange, output_dir, name, true)?;
    }
    Ok(())
}

fn parse_exit_artifact(output_dir: &Path) -> Result<i32, ProfileError> {
    fs::read_to_string(output_dir.join("exit_code"))?
        .trim()
        .parse::<i32>()
        .map_err(|_| ProfileError::new("profile command exit-code artifact is malformed"))
}

fn read_link_text(path: impl AsRef<Path>) -> Option<String> {
    fs::read_link(path)
        .ok()
        .map(|value| value.to_string_lossy().into_owned())
}

fn time_namespace_offset(pid: u32) -> Option<i64> {
    let text = fs::read_to_string(format!("/proc/{pid}/timens_offsets")).ok()?;
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() == 3 && fields[0] == "monotonic" {
            let seconds = fields[1].parse::<i64>().ok()?;
            let nanoseconds = fields[2].parse::<i64>().ok()?;
            if !(0..1_000_000_000).contains(&nanoseconds) {
                return None;
            }
            return seconds
                .checked_mul(1_000_000_000)
                .and_then(|value| value.checked_add(nanoseconds));
        }
    }
    None
}

fn wait_for_container_exit(
    container: &mut RunningContainer,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<ExitStatus, ProfileError> {
    loop {
        if container.status()?.is_some() {
            return container.wait();
        }
        if cancellation_requested(cancel_requested) {
            return Err(ProfileError::new(
                "interrupted while waiting for Docker cleanup",
            ));
        }
        thread::sleep(ATTACH_POLL);
    }
}

fn close_network_policy_session(
    session: &mut Option<DockerNetworkPolicySession>,
) -> Result<(), ProfileError> {
    let Some(session) = session.as_mut() else {
        return Ok(());
    };
    if session.is_closed() {
        return Ok(());
    }
    session.close().map_err(|error| {
        ProfileError::new(format!("Docker network-policy cleanup failed: {error}"))
    })
}

fn with_network_cleanup_error(
    primary: ProfileError,
    session: &mut Option<DockerNetworkPolicySession>,
) -> ProfileError {
    match close_network_policy_session(session) {
        Ok(()) => primary,
        Err(cleanup) => ProfileError::new(format!("{primary}; {cleanup}")),
    }
}

enum NativeSchedulerTracer {
    Bcc(crate::scheduler_bcc::BccSchedulerTracer),
    Perf(crate::scheduler_perf::PerfSchedulerTracer),
}

impl NativeSchedulerTracer {
    fn attach(
        config: &ProfileConfig,
        output_dir: &Path,
        root_tid: u32,
        cgroup_path: &Path,
    ) -> Result<Self, ProfileError> {
        match config.scheduler_backend.as_str() {
            "bcc" => {
                let mut tracer = crate::scheduler_bcc::BccSchedulerTracer::new(
                    config.scheduler_bcc_buffer_pages,
                )?;
                tracer.start(root_tid, cgroup_path)?;
                Ok(Self::Bcc(tracer))
            }
            "perf" => {
                let mut tracer = crate::scheduler_perf::PerfSchedulerTracer::new(
                    output_dir,
                    config.cpuset_cpus.as_deref(),
                    &config.scheduler_perf_binary,
                    config.attach_timeout,
                    config.scheduler_perf_mmap_pages,
                )?;
                tracer.start(root_tid, cgroup_path)?;
                Ok(Self::Perf(tracer))
            }
            backend => Err(ProfileError::new(format!(
                "unsupported scheduler backend after validation: {backend:?}"
            ))),
        }
    }

    fn poll(&mut self) -> Result<(), ProfileError> {
        match self {
            Self::Bcc(tracer) => tracer.poll(0).map_err(ProfileError::from),
            Self::Perf(tracer) => tracer.poll().map_err(ProfileError::from),
        }
    }

    fn finish(
        &mut self,
        window_start_ns: u64,
        window_end_ns: u64,
        output_dir: &Path,
    ) -> Result<Value, ProfileError> {
        match self {
            Self::Bcc(tracer) => {
                let report = tracer.finish(window_start_ns, window_end_ns)?;
                tracer.write_events(&output_dir.join("scheduler_events.jsonl"))?;
                Ok(report)
            }
            Self::Perf(tracer) => {
                let output = tracer.finish(window_start_ns, window_end_ns)?;
                crate::scheduler_perf::write_scheduler_events(
                    &output_dir.join("scheduler_events.jsonl"),
                    &output.events,
                )?;
                Ok(output.report)
            }
        }
    }

    fn close_checked(&mut self) -> Result<(), ProfileError> {
        match self {
            Self::Bcc(tracer) => tracer.close().map_err(|error| {
                ProfileError::new(format!("BCC scheduler cleanup failed: {error}"))
            }),
            Self::Perf(tracer) => tracer.close().map_err(|error| {
                ProfileError::new(format!("perf scheduler cleanup failed: {error}"))
            }),
        }
    }
}

fn with_scheduler_cleanup<T>(
    primary: Result<T, ProfileError>,
    cleanup: impl FnOnce() -> Result<(), ProfileError>,
) -> Result<T, ProfileError> {
    let cleanup = cleanup();
    match (primary, cleanup) {
        (Ok(value), Ok(())) => Ok(value),
        (Err(primary), Ok(())) => Err(primary),
        (Ok(_), Err(cleanup)) => Err(cleanup),
        (Err(primary), Err(cleanup)) => Err(ProfileError::new(format!("{primary}; {cleanup}"))),
    }
}

/// Run one native disposable-container profile.
///
/// The callback is polled at every host-side gate and during descendant drain;
/// returning true aborts through the same forced Docker cleanup path used for
/// runtime errors.
pub fn run_profile(
    mut config: ProfileConfig,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<ProfileRun, ProfileError> {
    validate_config(&config)?;
    if config.runtime == "enroot" {
        return run_enroot_profile(config, cancel_requested);
    }
    preflight_output(&config.output_dir)?;
    let _scheduler_preflight = preflight_runtime(&config)?;
    let image = image_identity(&config)?;
    let image_id = image["id"]
        .as_str()
        .ok_or_else(|| ProfileError::new("Docker image identity omitted immutable ID"))?
        .to_owned();
    if cancellation_requested(cancel_requested) {
        return Err(ProfileError::new(
            "profile cancelled before output creation",
        ));
    }
    let (output_dir, _output_claim) = create_output(&config.output_dir)?;
    let harness = HarnessFile::create()?;
    // Keep the exchange on the caller-selected output filesystem so large
    // traces do not spill into a smaller temporary filesystem. Only this
    // child is writable in the workload; the publication root is mounted
    // separately read-only for cleanup-scope discovery.
    let mut task_exchange = TaskExchange::create(&output_dir)?;
    validate_private_paths_isolated_from_workload_mounts(
        &[
            ("profile output directory", &output_dir),
            ("controller harness directory", &harness.directory),
        ],
        &config.mounts,
    )?;
    let profile_run_id = run_id()?;
    let profile_started_at = utc_now()?;
    let executable = env::current_exe()?;
    let controller_digest = sha256_file(&executable)?;
    config.output_dir = output_dir.clone();
    let mut network_session = None;
    let network_binding = if config.network_policy.effective_mode() == NetworkMode::Allowlist {
        let mut enforcement =
            DockerNetworkPolicyConfig::new(config.network_policy.clone(), &output_dir)
                .map_err(|error| ProfileError::new(error.to_string()))?;
        enforcement.docker = config.docker.clone();
        enforcement.proxy_image = config.network_policy_proxy_image.clone();
        enforcement.external_network = config.network_policy_external_network.clone();
        enforcement.workload_mount_sources = workload_mount_sources(&config.mounts)?;
        enforcement.proxy_executable = executable.canonicalize()?;
        let mut session = DockerNetworkPolicySession::new(enforcement)
            .map_err(|error| ProfileError::new(error.to_string()))?;
        let binding = session
            .prepare()
            .map_err(|error| ProfileError::new(error.to_string()))?;
        config.environment = binding.applied_environment(&config.environment);
        network_session = Some(session);
        Some(binding)
    } else {
        None
    };
    if cancellation_requested(cancel_requested) {
        return Err(with_network_cleanup_error(
            ProfileError::new("profile cancelled after network-policy preparation"),
            &mut network_session,
        ));
    }
    let (plan, plan_digest) = match execution_plan(&config, &image, &controller_digest) {
        Ok(plan) => plan,
        Err(error) => return Err(with_network_cleanup_error(error, &mut network_session)),
    };
    let run_start_ns = monotonic_ns();
    let mut container = match RunningContainer::spawn(
        &config,
        &image_id,
        &output_dir,
        task_exchange.path(),
        &harness,
        network_binding.as_ref(),
    ) {
        Ok(container) => container,
        Err(error) => return Err(with_network_cleanup_error(error, &mut network_session)),
    };

    let mut scheduler_tracer = None;
    let mut hardware_profiler = None;
    let execution = (|| -> Result<ProfileRun, ProfileError> {
        wait_for_marker(
            &task_exchange,
            "sandbox_attach_ready",
            &mut container,
            Some(config.attach_timeout),
            "coordinator sandbox attachment gate",
            cancel_requested,
            None,
        )?;
        let (root_pid, docker_cpuset, nano_cpus) = inspect_container(&config, &container)?;
        let cgroup = cgroup_for_pid(root_pid)?;
        let metadata = cgroup.metadata()?;
        let placement =
            validate_cpu_placement(&config, root_pid, &cgroup, &docker_cpuset, nano_cpus)?;
        let memory_enforcement = if config.memory_enabled() {
            Some(inspect_memory_enforcement(&config, &container, &cgroup)?)
        } else {
            None
        };
        let mut memory_sampler = if config.memory_enabled() {
            let mut sampler = cgroup_memory::MemorySampler::new(
                &cgroup,
                Path::new(CGROUP_ROOT),
                config.memory_sample_interval_sec,
            )
            .map_err(|error| {
                ProfileError::new(format!(
                    "unable to initialize cgroup memory sampler: {error}"
                ))
            })?;
            sampler.start().map_err(|error| {
                ProfileError::new(format!("unable to start cgroup memory sampling: {error}"))
            })?;
            Some(sampler)
        } else {
            None
        };
        let sandbox_ready_ns = monotonic_ns();
        if let Some(session) = network_session.as_mut() {
            session
                .validate_before_workload_release(&container.name)
                .map_err(|error| {
                    ProfileError::new(format!(
                        "allowlist enforcement validation failed before workload release: {error}"
                    ))
                })?;
        }
        let sandbox_start_ns = monotonic_ns();
        release_fifo(
            &task_exchange,
            "sandbox.start.gate",
            &mut container,
            config.attach_timeout,
            cancel_requested,
        )?;
        if let Some(sampler) = memory_sampler.as_mut() {
            let mut sample = || {
                sampler.poll().map_err(|error| {
                    ProfileError::new(format!("cgroup memory sampling failed: {error}"))
                })
            };
            wait_for_marker(
                &task_exchange,
                "scheduler_attach_ready",
                &mut container,
                Some(config.attach_timeout),
                "coordinator command attachment gate",
                cancel_requested,
                Some(&mut sample),
            )?;
        } else {
            wait_for_marker(
                &task_exchange,
                "scheduler_attach_ready",
                &mut container,
                Some(config.attach_timeout),
                "coordinator command attachment gate",
                cancel_requested,
                None,
            )?;
        }
        let command_ready_ns = monotonic_ns();
        hardware_profiler = if config.hardware_enabled() {
            Some(crate::hardware_perf::HardwarePerfRun::new(
                &output_dir,
                config.cpuset_cpus.clone(),
                config.hardware_perf_binary.clone(),
                config.hardware_perf_events.clone(),
                config.hardware_perf_attach_timeout,
                config.hardware_perf_minimum_running_ratio,
            )?)
        } else {
            None
        };
        if let Some(profiler) = hardware_profiler.as_mut() {
            // Attachment is deliberately non-fatal: unavailable PMU access is
            // retained as explicit invalid evidence while the independently
            // useful workload and correctness checks still run. Uncertain
            // child cleanup is fatal and aborts before workload release.
            profiler.attach(&cgroup)?;
        }
        let mut proc_sampler = if config.proc_stat_comparison {
            let cpu_ids = parse_cpuset(
                config
                    .cpuset_cpus
                    .as_deref()
                    .expect("validated proc-stat cpuset"),
            )?
            .into_iter()
            .collect::<Vec<_>>();
            Some(
                ProcStatComparisonSampler::with_path(
                    cpu_ids,
                    config.proc_stat_interval_sec,
                    config.proc_stat_baseline_sec,
                    3.0,
                    PathBuf::from("/proc/stat"),
                )
                .map_err(|error| {
                    ProfileError::new(format!(
                        "unable to initialize proc-stat comparison: {error}"
                    ))
                })?,
            )
        } else {
            None
        };
        let proc_baseline_window = if let Some(sampler) = proc_sampler.as_mut() {
            let baseline_start_ns = monotonic_ns();
            let mut baseline_poll = || {
                if cancellation_requested(cancel_requested) {
                    return Err(crate::proc_stat::ProcStatSamplingError::new(
                        "proc-stat baseline interrupted",
                    ));
                }
                if let Some(memory) = memory_sampler.as_mut() {
                    memory.poll().map_err(|error| {
                        crate::proc_stat::ProcStatSamplingError::new(format!(
                            "cgroup memory sampling failed during proc-stat baseline: {error}"
                        ))
                    })?;
                }
                if let Some(hardware) = hardware_profiler.as_mut() {
                    hardware.poll().map_err(|error| {
                        crate::proc_stat::ProcStatSamplingError::new(format!(
                            "hardware perf profiler cleanup failed during proc-stat baseline: {error}"
                        ))
                    })?;
                }
                if let Some(status) = container.status().map_err(|error| {
                    crate::proc_stat::ProcStatSamplingError::new(format!(
                        "unable to inspect container during proc-stat baseline: {error}"
                    ))
                })? {
                    return Err(crate::proc_stat::ProcStatSamplingError::new(format!(
                        "container exited during proc-stat baseline with status {}",
                        status.code().unwrap_or(-1)
                    )));
                }
                Ok(())
            };
            sampler
                .collect_baseline(Some(&mut baseline_poll), Some(ATTACH_POLL.as_secs_f64()))
                .map_err(|error| {
                    ProfileError::new(format!("proc-stat baseline failed: {error}"))
                })?;
            Some((baseline_start_ns, monotonic_ns()))
        } else {
            None
        };
        // Attach cgroup-scoped scheduler tracing only after any host-pressure
        // baseline. The workload is still blocked on scheduler.start.gate, so
        // capture is active before release. Any pre-window transitions are
        // retained in the raw artifact and clipped by the shared summary.
        scheduler_tracer = if config.scheduler_enabled() {
            Some(NativeSchedulerTracer::attach(
                &config,
                &output_dir,
                root_pid,
                &cgroup,
            )?)
        } else {
            None
        };
        if let Some(sampler) = memory_sampler.as_mut() {
            sampler.mark("command_start").map_err(|error| {
                ProfileError::new(format!("unable to mark memory command start: {error}"))
            })?;
        }
        let (leaf_before, ancestors_before) = capture_chain(&cgroup)?;
        // Match perf's CLOCK_MONOTONIC integration window to the end of the
        // authoritative leaf cpu.stat acquisition. This mirrors the Python
        // contract and keeps short-job snapshot skew inside the strict 5 ms
        // reconciliation floor instead of timing controller setup.
        let measured_start_ns = leaf_before.acquisition_end_ns;
        let mut proc_measurement_start_ns = None;
        if let Some(sampler) = proc_sampler.as_mut() {
            proc_measurement_start_ns = Some(monotonic_ns());
            sampler.start_measurement().map_err(|error| {
                ProfileError::new(format!("proc-stat measurement start failed: {error}"))
            })?;
        }
        // perf was attached with -D -1 and acknowledged an idempotent
        // disable.  Enable only now, immediately before releasing the
        // measured command gate; request/ACK uncertainty is retained by the
        // hardware report rather than pretending this marker is exact.
        if let Some(hardware) = hardware_profiler.as_mut() {
            hardware.enable()?;
        }
        release_fifo(
            &task_exchange,
            "scheduler.start.gate",
            &mut container,
            config.attach_timeout,
            cancel_requested,
        )?;
        {
            let mut sample = || {
                if let Some(sampler) = proc_sampler.as_mut() {
                    sampler.poll().map_err(|error| {
                        ProfileError::new(format!("proc-stat sampling failed: {error}"))
                    })?;
                }
                if let Some(sampler) = memory_sampler.as_mut() {
                    sampler.poll().map_err(|error| {
                        ProfileError::new(format!("cgroup memory sampling failed: {error}"))
                    })?;
                }
                if let Some(scheduler) = scheduler_tracer.as_mut() {
                    scheduler.poll().map_err(|error| {
                        ProfileError::new(format!("scheduler recorder failed: {error}"))
                    })?;
                }
                if let Some(hardware) = hardware_profiler.as_mut() {
                    hardware.poll()?;
                }
                Ok(())
            };
            wait_for_marker(
                &task_exchange,
                "scheduler_measurement_done",
                &mut container,
                None,
                "coordinated command finish gate",
                cancel_requested,
                Some(&mut sample),
            )?;
        }
        let command_end_ns = monotonic_ns();
        if let Some(sampler) = memory_sampler.as_mut() {
            sampler.mark("command_end").map_err(|error| {
                ProfileError::new(format!("unable to mark memory command end: {error}"))
            })?;
        }
        let drain = {
            let mut sample = || {
                if let Some(sampler) = proc_sampler.as_mut() {
                    sampler.poll().map_err(|error| {
                        ProfileError::new(format!("proc-stat sampling failed: {error}"))
                    })?;
                }
                if let Some(sampler) = memory_sampler.as_mut() {
                    sampler.poll().map_err(|error| {
                        ProfileError::new(format!("cgroup memory sampling failed: {error}"))
                    })?;
                }
                if let Some(scheduler) = scheduler_tracer.as_mut() {
                    scheduler.poll().map_err(|error| {
                        ProfileError::new(format!("scheduler recorder failed: {error}"))
                    })?;
                }
                if let Some(hardware) = hardware_profiler.as_mut() {
                    hardware.poll()?;
                }
                Ok(())
            };
            wait_for_descendants(
                &config,
                &cgroup,
                root_pid,
                cancel_requested,
                Some(&mut sample),
            )?
        };
        let (leaf_after, ancestors_after) = capture_chain(&cgroup)?;
        let measured_end_ns = leaf_after.acquisition_end_ns;
        let hardware_report = hardware_profiler
            .as_mut()
            .map(|profiler| profiler.finish(measured_start_ns, measured_end_ns))
            .transpose()?;
        let scheduler_output = if let Some(tracer) = scheduler_tracer.as_mut() {
            Some(tracer.finish(measured_start_ns, measured_end_ns, &output_dir)?)
        } else {
            None
        };
        let mut proc_measurement_window = None;
        let proc_stat_base_report = if let Some(sampler) = proc_sampler.as_mut() {
            let report = sampler.finish().map_err(|error| {
                ProfileError::new(format!("proc-stat measurement finish failed: {error}"))
            })?;
            proc_measurement_window = Some((
                proc_measurement_start_ns.expect("started proc-stat sampler"),
                monotonic_ns(),
            ));
            sampler
                .write_samples(&output_dir.join("proc_stat_samples.json"))
                .map_err(|error| {
                    ProfileError::new(format!("unable to retain proc-stat samples: {error}"))
                })?;
            Some(report)
        } else {
            None
        };
        // Import only the tiny exit/status controls needed to finalize the
        // memory profiler. Large task logs remain outside every measurement
        // boundary below.
        publish_measurement_control_artifacts(&task_exchange, &output_dir)?;
        let command_exit_code_at_gate = parse_exit_artifact(&output_dir)?;
        let remaining_descendant_pids: Vec<u64> = drain
            .remaining
            .iter()
            .filter_map(|value| value["pid"].as_u64())
            .collect();
        let mut memory_output = if let Some(sampler) = memory_sampler.as_mut() {
            let mut output = sampler
                .finish(command_exit_code_at_gate, &remaining_descendant_pids)
                .map_err(|error| {
                    ProfileError::new(format!("cgroup memory sampling finish failed: {error}"))
                })?;
            let timeseries_path = output_dir.join("resource_timeseries.json");
            let sources_path = output_dir.join("cgroup_memory_source_snapshots.json");
            write_json(&timeseries_path, &output.timeseries)?;
            write_json(&sources_path, &output.raw_sources)?;
            let report = output.report.as_object_mut().expect("memory report object");
            report.insert(
                "samples_path".to_owned(),
                Value::String(timeseries_path.to_string_lossy().into_owned()),
            );
            report.insert(
                "configured_enforcement".to_owned(),
                memory_enforcement.clone().unwrap_or(Value::Null),
            );
            report.insert(
                "timeline_scope".to_owned(),
                Value::String("From the pre-prepare lifecycle gate through command completion and bounded descendant drain. Container startup before observer attachment is retained by the fresh-cgroup memory.peak.".to_owned()),
            );
            Some(output)
        } else {
            None
        };
        // Every resource/software/hardware profiler is stopped now. Stream
        // task-owned payloads into the host-write-private publication directory only
        // after those boundaries, then parse the published process trace.
        publish_measurement_payload_artifacts(&task_exchange, &output_dir)?;
        let process_exec_report = collect_process_exec_report(&config, &output_dir)?;
        write_cpu_source_artifacts(
            &output_dir,
            &leaf_before,
            &leaf_after,
            &ancestors_before,
            &ancestors_after,
        )?;
        atomic_write(
            &output_dir.join("start_ns"),
            format!("{measured_start_ns}\n").as_bytes(),
        )?;
        atomic_write(
            &output_dir.join("end_ns"),
            format!("{measured_end_ns}\n").as_bytes(),
        )?;
        let identity = json!({
            "runtime": "docker",
            "container_name": container.name,
            "root_pid": root_pid,
            "cgroup_path": cgroup,
            "cgroup_inode": metadata.ino(),
            "cgroup_device": metadata.dev(),
            "image_identity": image,
            "time_namespace": read_link_text(format!("/proc/{root_pid}/ns/time")),
            "time_for_children_namespace": read_link_text(format!("/proc/{root_pid}/ns/time_for_children")),
            "time_namespace_monotonic_offset_ns": time_namespace_offset(root_pid),
            "boot_id": fs::read_to_string("/proc/sys/kernel/random/boot_id").ok().map(|value| value.trim().to_owned()),
            "cpu_placement": placement.to_json(),
        });
        let scope_id = format!("docker-cgroup-{}", &canonical_digest(&identity)?[..24]);
        if let Some(memory) = memory_output.as_mut() {
            let report = memory.report.as_object_mut().expect("memory report object");
            report.insert("scope_id".to_owned(), Value::String(scope_id.clone()));
            report.insert(
                "profile_run_id".to_owned(),
                Value::String(profile_run_id.clone()),
            );
            report.insert(
                "profile_started_at_utc".to_owned(),
                Value::String(profile_started_at.clone()),
            );
            report.insert(
                "controller_source_sha256".to_owned(),
                Value::String(controller_digest.clone()),
            );
            report.insert(
                "descendant_finish_policy".to_owned(),
                Value::String(
                    if config.terminate_descendants_at_finish {
                        "terminate-at-trial-end"
                    } else {
                        "require-natural-exit"
                    }
                    .to_owned(),
                ),
            );
            report.insert(
                "scope_coverage".to_owned(),
                json!({
                    "kind": "disposable-container-cgroup-subtree",
                    "includes": [
                        "container init and profiling entrypoint",
                        "prepare command",
                        "measured command and descendants",
                        "cgroup-charged anonymous, file-cache, and kernel memory",
                    ],
                    "excludes": [
                        "post-measurement workload verification, final finish-gate FIFO unblock, and shell/container-init exit",
                        "host profiler process",
                        "Docker daemon and container runtime processes",
                        "sibling or sidecar containers outside this cgroup",
                        "host page cache not charged to this cgroup",
                    ],
                    "complete_for_profiled_container": false,
                    "complete_through_controlled_finish_gate": true,
                    "container_init_exit_observed": false,
                    "full_production_sandbox_requires_all_components_in_scope": true,
                }),
            );
            report.insert(
                "execution_provenance".to_owned(),
                json!({
                    "runtime": "docker",
                    "image_identity": image,
                    "execution_plan_sha256": plan_digest,
                    "controller_source_sha256": controller_digest,
                    "cpuset_cpus": config.cpuset_cpus,
                    "allocated_cpus": config.allocated_cpus,
                    "memory_limit_bytes": config.memory_limit_bytes,
                    "memory_swap_total_limit_bytes": config.memory_swap_total_limit_bytes,
                    "memory_sample_interval_sec": config.memory_sample_interval_sec,
                    "network_policy": plan
                        .pointer("/extensions/network_policy")
                        .cloned()
                        .unwrap_or_else(|| config.network_policy.to_comparability_payload()),
                    "configured_enforcement": memory_enforcement,
                }),
            );
            if config.effective_strace_mode() != StraceMode::None {
                report.insert("natural_peak_trustworthy".to_owned(), Value::Bool(false));
                report.insert("valid_for_sizing".to_owned(), Value::Bool(false));
                let reason = Value::String(
                    "strace ran inside the measured cgroup and changed the sandbox memory footprint; rerun with --strace-mode none to measure an unperturbed natural peak"
                        .to_owned(),
                );
                let reasons = report
                    .entry("natural_peak_invalid_reasons".to_owned())
                    .or_insert_with(|| Value::Array(Vec::new()));
                if let Some(reasons) = reasons.as_array_mut() {
                    if !reasons.contains(&reason) {
                        reasons.push(reason);
                    }
                }
            }
        }
        let cpu = cpu_report(
            &config,
            &scope_id,
            measured_start_ns,
            measured_end_ns,
            &leaf_before,
            &leaf_after,
            &ancestors_before,
            &ancestors_after,
            &drain,
        )?;
        let scheduler_report = if let Some(output) = scheduler_output {
            let mut report = output;
            let object = report.as_object_mut().expect("scheduler report object");
            object.insert("window_start_ns".to_owned(), json!(measured_start_ns));
            object.insert("window_end_ns".to_owned(), json!(measured_end_ns));
            object.insert("scope_id".to_owned(), Value::String(scope_id.clone()));
            object.insert(
                "events_path".to_owned(),
                json!(output_dir.join("scheduler_events.jsonl")),
            );
            object.insert(
                "descendant_finish_policy".to_owned(),
                Value::String(
                    if config.terminate_descendants_at_finish {
                        "terminate-at-trial-end"
                    } else {
                        "require-natural-exit"
                    }
                    .to_owned(),
                ),
            );
            object.insert(
                "descendants_signaled_at_finish".to_owned(),
                json!({"sigterm": drain.signaled_term, "sigkill": drain.signaled_kill}),
            );
            object.insert(
                "descendant_drain_wait_sec".to_owned(),
                json!(round(drain.waited_ns as f64 / 1_000_000_000.0, 9)),
            );
            object.insert(
                "descendant_processes_at_finish_gate".to_owned(),
                json!(drain.initial),
            );
            object.insert(
                "remaining_descendant_processes_at_finish_gate".to_owned(),
                json!(drain.remaining),
            );
            object.insert(
                "remaining_descendant_pids_at_finish_gate".to_owned(),
                json!(remaining_descendant_pids),
            );
            if !remaining_descendant_pids.is_empty() {
                let anomaly = Value::String(format!(
                    "descendant PIDs remained in the container cgroup after the measured boundary: {}",
                    remaining_descendant_pids
                        .iter()
                        .map(u64::to_string)
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
                object
                    .entry("anomalies".to_owned())
                    .or_insert_with(|| Value::Array(Vec::new()))
                    .as_array_mut()
                    .expect("scheduler anomalies array")
                    .push(anomaly);
                object.insert("trace_complete".to_owned(), Value::Bool(false));
            }
            let cgroup_usage = cpu["usage_sec"].as_f64().ok_or_else(|| {
                ProfileError::new("cgroup CPU report omitted scheduler validation usage")
            })?;
            let mut validated = if config.scheduler_backend == "bcc" {
                crate::scheduler_perf::add_cgroup_validation_with_event_tolerance(
                    &report,
                    cgroup_usage,
                    config.scheduler_cpu_tolerance_fraction,
                    config.scheduler_cpu_tolerance_sec,
                    crate::scheduler_bcc::EVENT_BOUNDARY_TOLERANCE_SEC,
                )?
            } else {
                crate::scheduler_perf::add_cgroup_validation(
                    &report,
                    cgroup_usage,
                    config.scheduler_cpu_tolerance_fraction,
                    config.scheduler_cpu_tolerance_sec,
                )?
            };
            let mut descriptors = Vec::new();
            for (filename, artifact_id, media_type) in scheduler_artifact_specs(&config) {
                let path = output_dir.join(filename);
                let descriptor =
                    retained_artifact(&path, artifact_id, media_type)?.ok_or_else(|| {
                        ProfileError::new(format!(
                            "scheduler profiler did not retain required artifact: {filename}"
                        ))
                    })?;
                descriptors.push(descriptor);
            }
            validated
                .as_object_mut()
                .expect("validated scheduler report object")
                .insert("artifacts".to_owned(), Value::Array(descriptors));
            Some(validated)
        } else {
            None
        };
        let proc_stat_report = proc_stat_base_report
            .as_ref()
            .map(|report| {
                add_reference_comparison(
                    report,
                    cpu["average_logical_cores_used"].as_f64(),
                    scheduler_report.as_ref().and_then(|scheduler| {
                        scheduler["peak_simultaneous_logical_cores"].as_i64()
                    }),
                    scheduler_report.as_ref().is_some_and(|scheduler| {
                        scheduler.pointer("/cgroup_validation/valid_for_concurrency_reporting")
                            == Some(&Value::Bool(true))
                    }),
                )
                .map_err(|error| {
                    ProfileError::new(format!(
                        "unable to join proc-stat and cgroup references: {error}"
                    ))
                })
            })
            .transpose()?
            .map(|mut report| {
                let object = report
                    .as_object_mut()
                    .expect("proc-stat comparison report is an object");
                object.insert(
                    "samples_path".to_owned(),
                    Value::String(
                        output_dir
                            .join("proc_stat_samples.json")
                            .to_string_lossy()
                            .into_owned(),
                    ),
                );
                object.insert(
                    "baseline_window".to_owned(),
                    proc_baseline_window.map_or(Value::Null, |(start, end)| {
                        json!({
                            "clock_id": "host-monotonic",
                            "start_ns": start,
                            "end_ns": end,
                            "outside_measured_window": true,
                        })
                    }),
                );
                object.insert(
                    "measurement_window".to_owned(),
                    proc_measurement_window.map_or(Value::Null, |(start, end)| {
                        json!({
                            "clock_id": "host-monotonic",
                            "start_ns": start,
                            "end_ns": end,
                            "boundary_kind": "enclosing-sampler-endpoint-acquisitions",
                            "includes_command": true,
                            "includes_descendant_drain": true,
                            "contains_every_sample_interval": true,
                        })
                    }),
                );
                object.insert("instrumentation_valid".to_owned(), Value::Bool(true));
                report
            });
        release_fifo(
            &task_exchange,
            "scheduler.finish.gate",
            &mut container,
            config.attach_timeout,
            cancel_requested,
        )?;
        let status = wait_for_container_exit(&mut container, cancel_requested)?;
        let docker_returncode = exit_code(status);
        container.verify_removed()?;
        publish_verification_artifacts(&config, &task_exchange, &output_dir)?;
        task_exchange.cleanup()?;
        let run_end_ns = monotonic_ns();
        close_network_policy_session(&mut network_session)?;
        let command_exit_code = parse_exit_artifact(&output_dir)?;
        let verification = verification_report(&config, &output_dir)?;
        let clock = clock_reference();
        let mut markers = vec![
            marker("run_start", run_start_ns),
            marker("sandbox_ready", sandbox_ready_ns),
            marker("sandbox_start", sandbox_start_ns),
            marker("command_ready", command_ready_ns),
        ];
        if let Some((baseline_start_ns, baseline_end_ns)) = proc_baseline_window {
            markers.push(marker("proc_stat_baseline_start", baseline_start_ns));
            markers.push(marker("proc_stat_baseline_end", baseline_end_ns));
        }
        markers.extend([
            marker("measured_work_start", measured_start_ns),
            marker("command_end", command_end_ns),
            marker("measured_work_end", measured_end_ns),
            marker("run_end", run_end_ns),
        ]);
        let lifecycle = json!({
            "clock_id": "host-monotonic",
            "clock_name": "CLOCK_MONOTONIC",
            "origin_ns": measured_start_ns,
            "start_ns": run_start_ns,
            "end_ns": run_end_ns,
            "lifecycle_markers": markers,
        });
        let scope = json!({
            "scope_id": scope_id,
            "kind": "disposable-container-cgroup-subtree",
            "identity": identity,
            "includes_descendants": true,
            "description": "Disposable Docker container leaf cgroup and every descendant cgroup/process through the controlled finish gate.",
            "extensions": {
                "canonical_clock": "CLOCK_MONOTONIC",
                "measurement_method": "linux-cgroup-v2-descendant-subtree",
                "launch_marker_ns": run_start_ns,
                "run_id": profile_run_id,
            },
        });
        let runtime_report = json!({
            "runtime": "docker",
            "container_name": container.name,
            "returncode": docker_returncode,
            "root_pid": root_pid,
            "cgroup_path": cgroup,
            "descendant_finish_policy": if config.terminate_descendants_at_finish { "terminate-at-trial-end" } else { "require-natural-exit" },
            "descendant_processes_at_finish_gate": drain.initial,
            "remaining_descendant_processes_at_finish_gate": drain.remaining,
            "remaining_descendant_pids_at_finish_gate": drain.remaining.iter().filter_map(|value| value["pid"].as_u64()).collect::<Vec<_>>(),
            "descendants_signaled_at_finish": {"sigterm": drain.signaled_term, "sigkill": drain.signaled_kill},
            "descendant_drain_wait_sec": round(drain.waited_ns as f64 / 1_000_000_000.0, 9),
            "stdout": fs::read_to_string(&container.stdout_path).unwrap_or_default(),
            "stderr": fs::read_to_string(&container.stderr_path).unwrap_or_default(),
            "verification": verification,
        });
        let (evidence_path, evidence_digest) = if config.is_coordinated() {
            let bundle = build_evidence_bundle(
                &config,
                &profile_run_id,
                &scope_id,
                &scope,
                &lifecycle,
                &clock,
                &plan,
                &plan_digest,
                &cpu,
                scheduler_report.as_ref(),
                hardware_report.as_ref(),
                proc_stat_report.as_ref(),
                memory_output.as_ref(),
                process_exec_report.as_ref(),
                &runtime_report,
                &verification,
                &output_dir,
            )?;
            let path = output_dir.join("evidence_bundle.json");
            write_json(&path, &bundle)?;
            let digest = sha256_file(&path)?;
            (Some(path), Some(digest))
        } else {
            (None, None)
        };
        let report = build_profile_report(
            &config,
            &image,
            &profile_run_id,
            &profile_started_at,
            command_exit_code,
            docker_returncode,
            &cpu,
            scheduler_report.as_ref(),
            hardware_report.as_ref(),
            proc_stat_report.as_ref(),
            memory_output.as_ref().map(|output| &output.report),
            process_exec_report.as_ref(),
            &clock,
            &lifecycle,
            evidence_path.as_deref(),
            evidence_digest.as_deref(),
            &plan_digest,
            &scope_id,
            &verification,
            &scope,
            &plan,
        )?;
        let report_path = output_dir.join("profile_report.json");
        write_json(&report_path, &report)?;
        Ok(ProfileRun {
            report,
            report_path,
            evidence_bundle_path: evidence_path,
            command_exit_code,
        })
    })();

    let execution = with_scheduler_cleanup(execution, || {
        scheduler_tracer
            .as_mut()
            .map(NativeSchedulerTracer::close_checked)
            .transpose()
            .map(|_| ())
    });
    let execution = with_scheduler_cleanup(execution, || {
        hardware_profiler
            .as_mut()
            .map(crate::hardware_perf::HardwarePerfRun::close)
            .transpose()
            .map(|_| ())
            .map_err(|error| {
                ProfileError::new(format!("perf hardware profiler cleanup failed: {error}"))
            })
    });

    match execution {
        Ok(result) => Ok(result),
        Err(error) => {
            let mut failures = vec![error.to_string()];
            if let Err(cleanup) = container.force_remove() {
                failures.push(format!("Docker cleanup also failed: {cleanup}"));
            }
            if let Err(cleanup) = close_network_policy_session(&mut network_session) {
                failures.push(cleanup.to_string());
            }
            if let Err(cleanup) = task_exchange.cleanup() {
                failures.push(format!("task-exchange cleanup also failed: {cleanup}"));
            }
            Err(ProfileError::new(failures.join("; ")))
        }
    }
}

static SIGNAL_CANCELLED: AtomicBool = AtomicBool::new(false);

extern "C" fn signal_handler(_signal: libc::c_int) {
    SIGNAL_CANCELLED.store(true, Ordering::SeqCst);
}

pub fn process_signal_cancelled() -> bool {
    SIGNAL_CANCELLED.load(Ordering::SeqCst)
}

fn cancellation_requested(callback: Option<&dyn Fn() -> bool>) -> bool {
    process_signal_cancelled() || callback.is_some_and(|callback| callback())
}

pub struct ProfileSignalGuard {
    previous_interrupt: libc::sighandler_t,
    previous_terminate: libc::sighandler_t,
}

impl ProfileSignalGuard {
    pub fn install() -> Result<Self, ProfileError> {
        SIGNAL_CANCELLED.store(false, Ordering::SeqCst);
        let previous_interrupt =
            unsafe { libc::signal(libc::SIGINT, signal_handler as libc::sighandler_t) };
        if previous_interrupt == libc::SIG_ERR {
            return Err(ProfileError::new(format!(
                "unable to install interrupt handler: {}",
                io::Error::last_os_error()
            )));
        }
        let previous_terminate =
            unsafe { libc::signal(libc::SIGTERM, signal_handler as libc::sighandler_t) };
        if previous_terminate == libc::SIG_ERR {
            unsafe {
                libc::signal(libc::SIGINT, previous_interrupt);
            }
            return Err(ProfileError::new(format!(
                "unable to install termination handler: {}",
                io::Error::last_os_error()
            )));
        }
        Ok(Self {
            previous_interrupt,
            previous_terminate,
        })
    }
}

impl Drop for ProfileSignalGuard {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.previous_interrupt);
            libc::signal(libc::SIGTERM, self.previous_terminate);
        }
    }
}

fn print_help(program: &str) {
    println!("usage: {program} (--image IMAGE | --container CONTAINER) --command COMMAND --output-dir OUTPUT_DIR [options]");
    println!();
    println!("Profile one containerized shell command using task-scoped cgroup CPU statistics.");
    println!();
    println!("native measurement modes:");
    println!("  --profile-set cpu-average   coordinated cgroup-v2 average CPU profile");
    println!("  --profile-set cpu-peak      coordinated cgroup CPU plus scheduler peak profile");
    println!("  --profile-set cpu-compare   coordinated CPU, scheduler, and memory profile");
    println!("  --profile-set memory-detail coordinated cgroup-v2 charged-memory profile");
    println!("  --profile-set software-deep structured process/exec lifecycle profile");
    println!("  --profile-set hardware-deep cgroup-scoped perf stat hardware counters");
    println!("  --coordinated-baseline      same coordinator with no optional profiler");
    println!("  --strace-mode execve        raw execve trace for a diagnostic baseline");
    println!("  --strace-mode process       structured process/exec lifecycle evidence");
    println!("  --strace-mode none          untraced diagnostic baseline");
    println!("  --proc-stat-comparison      sampled host pressure on --cpuset-cpus");
    println!("  --scheduler-concurrency     diagnostic scheduler trace");
    println!(
        "  --scheduler-backend bcc     native cgroup-subtree eBPF tracing (default; root + libbcc)"
    );
    println!("  --scheduler-backend perf    native cgroup-wide perf scheduler tracing");
    println!("  --memory-profile            cgroup-v2 charged-memory detail");
    println!();
    println!("native existing-Docker-container diagnostic mode:");
    println!("  --container NAME            inherit the exact running container cgroup/network");
    println!("                               never update, restart, stop, or remove it");
    println!("                               concurrent container work is included in CPU totals");
    println!();
    println!("native Enroot wall-time mode:");
    println!("  --runtime enroot --strace-mode none");
    println!("                               .sqsh create/start/remove or an existing Enroot name");
    println!(
        "                               (taskset is requested affinity; no task cgroup evidence)"
    );
    println!();
}

fn cli_error(public_program: &str, error: impl fmt::Display) -> i32 {
    eprintln!("usage: {public_program} (--image IMAGE | --container CONTAINER) --command COMMAND --output-dir OUTPUT_DIR [options]");
    eprintln!("{public_program}: error: {error}");
    2
}

fn public_result_exit_code(result: &ProfileRun) -> i32 {
    i32::from(
        result.command_exit_code != 0
            || result.report.get("profiling_valid") == Some(&Value::Bool(false))
            || result
                .report
                .get("scheduler_concurrency")
                .filter(|value| !value.is_null())
                .is_some_and(|scheduler| {
                    scheduler.pointer("/cgroup_validation/valid_for_concurrency_reporting")
                        != Some(&Value::Bool(true))
                })
            || result
                .report
                .get("memory")
                .filter(|value| !value.is_null())
                .is_some_and(|memory| {
                    memory.get("valid_for_observed_memory_reporting") != Some(&Value::Bool(true))
                }),
    )
}

pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    if crate::existing_container::is_existing_container_request(arguments) {
        return existing_container_main(arguments, public_program);
    }
    let config = match parse_arguments(arguments) {
        Ok(Some(config)) => config,
        Ok(None) => {
            print_help(public_program);
            return 0;
        }
        Err(error) => return cli_error(public_program, error),
    };
    let guard = match ProfileSignalGuard::install() {
        Ok(guard) => guard,
        Err(error) => return cli_error(public_program, error),
    };
    let result = run_profile(config, None);
    drop(guard);
    match result {
        Ok(result) => {
            let cpu = &result.report["cpu"];
            println!("Command exit code: {}", result.command_exit_code);
            println!(
                "Wall time: {:.9} sec",
                cpu["wall_sec"].as_f64().unwrap_or(0.0)
            );
            if let Some(usage) = cpu["usage_sec"].as_f64() {
                println!("CPU time: {usage:.9} sec");
            } else {
                println!("CPU time: n/a (task-scoped cgroup cpu.stat unavailable)");
            }
            if let Some(average) = cpu["average_logical_cores_used"].as_f64() {
                println!("Average logical cores used: {average:.6}");
            } else {
                println!(
                    "Average logical cores used: n/a (task-scoped cgroup cpu.stat unavailable)"
                );
            }
            if let (Some(percent), Some(capacity)) = (
                cpu["utilization_percent_of_allocation"].as_f64(),
                cpu["allocated_logical_cpus"].as_f64(),
            ) {
                println!("Allocated CPU utilization: {percent:.3}% of {capacity} logical CPUs");
            }
            if let Some(peak) = result
                .report
                .pointer("/proc_stat_comparison/measurement/peak_active_logical_cpus")
                .and_then(Value::as_u64)
            {
                println!("Sampled /proc active-CPU peak (comparison only): {peak}");
            }
            if let Some(memory) = result.report.get("memory").filter(|value| !value.is_null()) {
                let lifetime_peak = memory
                    .pointer("/lifetime/peak_bytes")
                    .and_then(Value::as_u64);
                let command_peak = memory
                    .pointer("/command_window/sampled_peak_bytes")
                    .and_then(Value::as_u64);
                println!(
                    "Sandbox controlled-finish memory peak: {}",
                    lifetime_peak.map_or_else(
                        || "n/a".to_owned(),
                        |value| format!("{:.3} MiB", value as f64 / (1024.0 * 1024.0)),
                    )
                );
                println!(
                    "Command-window sampled memory peak: {}",
                    command_peak.map_or_else(
                        || "n/a".to_owned(),
                        |value| format!("{:.3} MiB", value as f64 / (1024.0 * 1024.0)),
                    )
                );
                if let Some(path) = memory.get("samples_path").and_then(Value::as_str) {
                    println!("CPU/memory timeline: {path}");
                }
                println!(
                    "Natural peak trustworthy: {}",
                    if memory.get("natural_peak_trustworthy") == Some(&Value::Bool(true)) {
                        "yes"
                    } else {
                        "no"
                    }
                );
                if let Some(remaining) = memory
                    .pointer("/observed_limit_margin/remaining_to_limit_bytes")
                    .and_then(Value::as_u64)
                {
                    println!(
                        "Observed leaf-limit margin (not a recommendation): {:.3} MiB",
                        remaining as f64 / (1024.0 * 1024.0)
                    );
                }
            }
            println!("Report: {}", result.report_path.display());
            if let Some(path) = &result.evidence_bundle_path {
                println!("Evidence bundle: {}", path.display());
            }
            public_result_exit_code(&result)
        }
        Err(error) if process_signal_cancelled() => {
            eprintln!("{public_program}: interrupted: {error}");
            130
        }
        Err(error) => cli_error(public_program, error),
    }
}

fn existing_container_main(arguments: &[OsString], public_program: &str) -> i32 {
    let config = match crate::existing_container::parse_arguments(arguments) {
        Ok(Some(config)) => config,
        Ok(None) => {
            print_help(public_program);
            return 0;
        }
        Err(error) => return cli_error(public_program, error),
    };
    let guard = match ProfileSignalGuard::install() {
        Ok(guard) => guard,
        Err(error) => return cli_error(public_program, error),
    };
    let result = crate::existing_container::run_profile(config, Some(&process_signal_cancelled));
    drop(guard);
    match result {
        Ok(result) => {
            let cpu = &result.report["cpu"];
            println!("Command exit code: {}", result.command_exit_code);
            println!(
                "Wall time: {:.9} sec",
                cpu["wall_sec"].as_f64().unwrap_or(0.0)
            );
            println!(
                "CPU time: {:.9} sec",
                cpu["usage_sec"].as_f64().unwrap_or(0.0)
            );
            println!(
                "Average logical cores used: {:.6}",
                cpu["average_logical_cores_used"].as_f64().unwrap_or(0.0)
            );
            if let (Some(percent), Some(capacity)) = (
                cpu["utilization_percent_of_allocation"].as_f64(),
                cpu["allocated_logical_cpus"].as_f64(),
            ) {
                println!("Allocated CPU utilization: {percent:.3}% of {capacity} logical CPUs");
            }
            println!("Report: {}", result.report_path.display());
            if result
                .report
                .pointer("/cancellation/requested")
                .and_then(Value::as_bool)
                == Some(true)
            {
                return 130;
            }
            i32::from(
                result.command_exit_code != 0
                    || result.report.get("profiling_valid") == Some(&Value::Bool(false)),
            )
        }
        Err(error) if process_signal_cancelled() => {
            eprintln!("{public_program}: interrupted: {error}");
            130
        }
        Err(error) => cli_error(public_program, error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_enroot_clock_artifacts(output_dir: &Path, offset: &str) {
        for (name, value) in [
            ("clock_source", "linux-proc-uptime\n"),
            ("clock_name", "CLOCK_BOOTTIME\n"),
            ("clock_resolution_ns", "1\n"),
            ("clock_boot_id", "boot-id\n"),
            ("clock_time_namespace", "time:[42]\n"),
            ("clock_time_namespace_boottime_offset", offset),
        ] {
            fs::write(output_dir.join(name), value).unwrap();
        }
    }

    #[test]
    fn enroot_task_clock_artifacts_preserve_boottime_identity_and_offset() {
        let root = tempfile::tempdir().unwrap();
        write_enroot_clock_artifacts(root.path(), "-2 500000000\n");

        let clock = enroot_task_clock_reference(root.path()).unwrap();
        assert_eq!(clock["clock_id"], "enroot-task-boottime");
        assert_eq!(clock["clock_name"], "CLOCK_BOOTTIME");
        assert_eq!(clock["source"], "linux-proc-uptime");
        assert_eq!(clock["resolution_ns"], 1);
        assert_eq!(clock["boot_id"], "boot-id");
        assert_eq!(clock["time_namespace"], "time:[42]");
        assert_eq!(
            clock["time_namespace_boottime_offset_ns"],
            -1_500_000_000_i64
        );

        fs::write(
            root.path().join("clock_time_namespace_boottime_offset"),
            "0 1000000000\n",
        )
        .unwrap();
        let error = enroot_task_clock_reference(root.path())
            .unwrap_err()
            .to_string();
        assert!(error.contains("time-namespace boottime offset"), "{error}");
    }

    #[test]
    fn enroot_wave_start_gate_report_requires_ordered_typed_evidence() {
        let root = tempfile::tempdir().unwrap();
        let mut runtime = crate::enroot_runtime::EnrootRuntimeConfig::new(
            "image.sqsh",
            "true",
            root.path(),
            root.path().join("entrypoint.sh"),
        );
        runtime.run_wave_start_gate = true;
        fs::write(root.path().join("run_wave_start_ready"), b"ready\n").unwrap();
        fs::write(root.path().join("run_wave_start_release"), b"release\n").unwrap();
        fs::write(root.path().join("run_wave_start_released_ns"), b"100\n").unwrap();
        let clock = json!({"clock_name": "CLOCK_BOOTTIME"});

        let report = enroot_run_wave_start_gate_report(&runtime, 125, &clock).unwrap();
        assert_eq!(report["schema_version"], "tasktrace-run-wave-start-gate.v1");
        assert_eq!(report["valid"], true);
        assert_eq!(report["released_ns"], 100);
        assert_eq!(report["measured_start_ns"], 125);
        assert_eq!(report["release_to_start_ns"], 25);
        assert_eq!(report["clock_reference"], clock);

        fs::write(root.path().join("run_wave_start_released_ns"), b"126\n").unwrap();
        let error = enroot_run_wave_start_gate_report(&runtime, 125, &clock)
            .unwrap_err()
            .to_string();
        assert!(error.contains("follows measured start"), "{error}");

        fs::write(root.path().join("run_wave_start_released_ns"), b"100\n").unwrap();
        fs::write(root.path().join("run_wave_start_release"), b"continue\n").unwrap();
        let error = enroot_run_wave_start_gate_report(&runtime, 125, &clock)
            .unwrap_err()
            .to_string();
        assert!(error.contains("invalid contents"), "{error}");
    }

    #[test]
    fn enroot_entrypoint_blocks_before_measured_start_until_wave_release() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("output");
        fs::create_dir(&output).unwrap();
        let entrypoint = root.path().join("entrypoint.sh");
        fs::write(&entrypoint, ENTRYPOINT).unwrap();
        let mut child = Command::new("/bin/sh")
            .arg(&entrypoint)
            .env("TASKTRACE_PROFILE_DIR", &output)
            .env("TASKTRACE_PROFILE_COMMAND", ":")
            .env("TASKTRACE_PROFILE_SHELL", "/bin/sh")
            .env("TASKTRACE_STRACE_MODE", "none")
            .env("TASKTRACE_PROFILE_SKIP_CPU_STATS", "1")
            .env("TASKTRACE_PROFILE_RUN_WAVE_START_GATE", "1")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();

        let deadline = std::time::Instant::now() + Duration::from_secs(5);
        while !output.join("run_wave_start_ready").is_file() {
            assert!(child.try_wait().unwrap().is_none());
            assert!(std::time::Instant::now() < deadline);
            thread::sleep(Duration::from_millis(5));
        }
        assert_eq!(
            fs::read(output.join("run_wave_start_ready")).unwrap(),
            b"ready\n"
        );
        assert!(!output.join("start_ns").exists());
        let mut release = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o600)
            .open(output.join("run_wave_start_release"))
            .unwrap();
        release.write_all(b"release\n").unwrap();
        release.sync_all().unwrap();
        drop(release);

        let status = child.wait().unwrap();
        assert!(status.success());
        let released = fs::read_to_string(output.join("run_wave_start_released_ns"))
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap();
        let start = fs::read_to_string(output.join("start_ns"))
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap();
        let end = fs::read_to_string(output.join("end_ns"))
            .unwrap()
            .trim()
            .parse::<u64>()
            .unwrap();
        assert!(released <= start);
        assert!(start <= end);
    }

    #[test]
    fn enroot_no_network_report_publishes_validated_enforcement_shape() {
        let root = tempfile::tempdir().unwrap();
        let mut config = ProfileConfig::new("image.sqsh", "true", root.path());
        config.runtime = "enroot".to_owned();
        config.cpuset_cpus = Some("0".to_owned());
        config.set_untraced_baseline();
        config.network = Some("none".to_owned());
        config.network_policy =
            NetworkPolicy::new(NetworkMode::NoNetwork, std::iter::empty::<&str>()).unwrap();

        let runtime = enroot_runtime_config(
            &config,
            root.path().join("entrypoint.sh"),
            Some(root.path().join("private-network-evidence.json")),
        );
        assert_eq!(runtime.network, None);
        assert_eq!(runtime.network_mode, NetworkMode::NoNetwork);
        runtime.validate().unwrap();

        let enforcement = json!({
            "schema_version": "tasktrace-enroot-network-enforcement.v1",
            "state": "enforced",
            "backend": "linux-user-network-namespace-v1",
            "policy_effective_mode": "no-network",
            "valid": true,
            "parent_netns_inode": 10,
            "netns_inode": 11,
            "netns_differs": true,
            "controller_euid": 1000,
            "controller_egid": 1000,
            "namespace_euid": 1000,
            "namespace_egid": 1000,
            "interfaces": ["lo"],
            "ipv4_routes": [],
            "ipv6_addresses": [],
        });
        let report = network_report_with_enforcement(&config, Some(enforcement.clone())).unwrap();
        assert_eq!(report["effective_mode"], "no-network");
        assert_eq!(
            report["enforcement"]["schema_version"],
            enforcement["schema_version"]
        );
        assert_eq!(report["enforcement"]["valid"], true);
        assert_eq!(
            report["enforcement"]["publication"],
            json!({
                "source": "controller-private-temporary-state",
                "published_after_enroot_exit": true,
            })
        );

        config.runtime = "docker".to_owned();
        let error = network_report_with_enforcement(&config, Some(enforcement))
            .unwrap_err()
            .to_string();
        assert!(error.contains("incompatible runtime policy"), "{error}");
    }

    #[test]
    fn task_exchange_never_follows_artifact_or_marker_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let exchange = TaskExchange::create(root.path()).unwrap();
        let victim = root.path().join("victim.txt");
        fs::write(&victim, b"outside-sentinel").unwrap();
        std::os::unix::fs::symlink(&victim, exchange.path().join("exit_code")).unwrap();

        let output = root.path().join("published");
        fs::create_dir(&output).unwrap();
        let error = publish_task_artifact(&exchange, &output, "exit_code", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");
        assert_eq!(fs::read(&victim).unwrap(), b"outside-sentinel");
        assert!(!output.join("exit_code").exists());

        std::os::unix::fs::symlink(&victim, exchange.path().join("ready-marker")).unwrap();
        let error = exchange
            .regular_ready("ready-marker")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");
    }

    #[test]
    fn task_exchange_fifo_release_rejects_symlinks_and_regular_files_before_write() {
        let root = tempfile::tempdir().unwrap();
        let exchange = TaskExchange::create(root.path()).unwrap();
        let victim = root.path().join("victim.txt");
        fs::write(&victim, b"outside-sentinel").unwrap();
        std::os::unix::fs::symlink(&victim, exchange.path().join("symlink.gate")).unwrap();
        let error = exchange
            .write_fifo("symlink.gate", b"continue\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a FIFO"), "{error}");
        assert_eq!(fs::read(&victim).unwrap(), b"outside-sentinel");

        fs::write(exchange.path().join("regular.gate"), b"task-data").unwrap();
        let error = exchange
            .write_fifo("regular.gate", b"continue\n")
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a FIFO"), "{error}");
        assert_eq!(
            fs::read(exchange.path().join("regular.gate")).unwrap(),
            b"task-data"
        );
    }

    #[test]
    fn task_exchange_rejects_nonregular_artifacts_before_access_reopen() {
        let root = tempfile::tempdir().unwrap();
        let exchange = TaskExchange::create(root.path()).unwrap();
        let output = root.path().join("published");
        fs::create_dir(&output).unwrap();

        fs::create_dir(exchange.path().join("directory-artifact")).unwrap();
        let error = publish_task_artifact(&exchange, &output, "directory-artifact", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");

        let fifo_path = exchange.path().join("fifo-artifact");
        let fifo_c = CString::new(fifo_path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
        let error = publish_task_artifact(&exchange, &output, "fifo-artifact", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("not a regular file"), "{error}");
        assert!(!output.join("fifo-artifact").exists());

        let device_path = exchange.path().join("device-artifact");
        let device_c = CString::new(device_path.to_string_lossy().as_bytes()).unwrap();
        let created = unsafe {
            libc::mknod(
                device_c.as_ptr(),
                libc::S_IFCHR | 0o600,
                libc::makedev(1, 3),
            )
        };
        if created == 0 {
            let error = publish_task_artifact(&exchange, &output, "device-artifact", true)
                .unwrap_err()
                .to_string();
            assert!(error.contains("not a regular file"), "{error}");
            assert!(!output.join("device-artifact").exists());
        } else {
            assert!(matches!(
                io::Error::last_os_error().raw_os_error(),
                Some(libc::EPERM | libc::EACCES)
            ));
        }
    }

    #[test]
    fn task_exchange_reopens_the_validated_fifo_inode_for_release() {
        let root = tempfile::tempdir().unwrap();
        let exchange = TaskExchange::create(root.path()).unwrap();
        let fifo_path = exchange.path().join("finish.gate");
        let fifo_c = CString::new(fifo_path.to_string_lossy().as_bytes()).unwrap();
        assert_eq!(unsafe { libc::mkfifo(fifo_c.as_ptr(), 0o600) }, 0);
        let reader = thread::spawn(move || {
            let mut file = File::open(fifo_path).unwrap();
            let mut value = String::new();
            file.read_to_string(&mut value).unwrap();
            value
        });
        let deadline = std::time::Instant::now() + Duration::from_secs(2);
        loop {
            if exchange.write_fifo("finish.gate", b"continue\n").unwrap() {
                break;
            }
            assert!(std::time::Instant::now() < deadline);
            thread::sleep(Duration::from_millis(1));
        }
        assert_eq!(reader.join().unwrap(), "continue\n");
    }

    #[test]
    fn task_exchange_dirfd_survives_path_substitution_without_reading_replacement() {
        let root = tempfile::tempdir().unwrap();
        let exchange = TaskExchange::create(root.path()).unwrap();
        fs::write(exchange.path().join("exit_code"), b"7\n").unwrap();
        let original_path = exchange.path().to_path_buf();
        let moved_path = root.path().join("moved-exchange");
        fs::rename(&original_path, moved_path).unwrap();
        fs::create_dir(&original_path).unwrap();
        let victim = root.path().join("victim.txt");
        fs::write(&victim, b"99\n").unwrap();
        std::os::unix::fs::symlink(&victim, original_path.join("exit_code")).unwrap();

        let output = root.path().join("published");
        fs::create_dir(&output).unwrap();
        assert!(exchange
            .copy_regular_to(
                "exit_code",
                &output.join("exit_code"),
                MAX_TASK_CONTROL_BYTES,
            )
            .unwrap());
        assert_eq!(fs::read(output.join("exit_code")).unwrap(), b"7\n");
        assert_eq!(fs::read(&victim).unwrap(), b"99\n");
    }

    #[test]
    fn task_exchange_rejects_hard_links_and_oversized_controls_without_publication() {
        let root = tempfile::tempdir().unwrap();
        let exchange = TaskExchange::create(root.path()).unwrap();
        let linked = exchange.path().join("exit_code");
        fs::write(&linked, b"0\n").unwrap();
        fs::hard_link(&linked, exchange.path().join("alias")).unwrap();
        let output = root.path().join("published");
        fs::create_dir(&output).unwrap();
        let error = publish_task_artifact(&exchange, &output, "exit_code", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("unsafe hard-link count"), "{error}");
        assert!(!output.join("exit_code").exists());

        fs::remove_file(exchange.path().join("alias")).unwrap();
        fs::remove_file(&linked).unwrap();
        let oversized = OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(&linked)
            .unwrap();
        oversized.set_len(MAX_TASK_CONTROL_BYTES + 1).unwrap();
        let error = publish_task_artifact(&exchange, &output, "exit_code", true)
            .unwrap_err()
            .to_string();
        assert!(error.contains("byte safety limit"), "{error}");
        assert!(!output.join("exit_code").exists());
    }

    #[test]
    fn broad_workload_mount_cannot_reexpose_private_output_or_exchange() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("output");
        fs::create_dir(&output).unwrap();
        let harness = HarnessFile::create().unwrap();
        let mount = format!("type=bind,src={},dst=/host", root.path().display());
        let error = validate_private_paths_isolated_from_workload_mounts(
            &[
                ("profile output directory", &output),
                ("controller harness directory", &harness.directory),
            ],
            &[mount],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("profile output directory"), "{error}");

        let harness_parent = harness.directory.parent().unwrap();
        let mount = format!("type=bind,src={},dst=/tmp-host", harness_parent.display());
        let error = validate_private_paths_isolated_from_workload_mounts(
            &[("controller harness directory", &harness.directory)],
            &[mount],
        )
        .unwrap_err()
        .to_string();
        assert!(error.contains("controller harness directory"), "{error}");
    }

    fn assert_artifact_descriptor_for_test(descriptor: &Value) {
        let path = PathBuf::from(descriptor["path"].as_str().unwrap());
        let bytes = fs::read(path).unwrap();
        assert_eq!(descriptor["size_bytes"], bytes.len() as u64);
        assert_eq!(
            descriptor["sha256"],
            format!("{:x}", Sha256::digest(&bytes))
        );
    }

    fn snapshot(path: &str, usage: u64, user: u64, system: u64, finite: bool) -> CpuSourceSnapshot {
        let cpu_stat = BTreeMap::from([
            ("usage_usec".to_owned(), usage),
            ("user_usec".to_owned(), user),
            ("system_usec".to_owned(), system),
            ("nr_periods".to_owned(), 0),
            ("nr_throttled".to_owned(), 0),
            ("throttled_usec".to_owned(), 0),
        ]);
        CpuSourceSnapshot {
            path: PathBuf::from(CGROUP_ROOT).join(path),
            relative_path: path.to_owned(),
            acquisition_start_ns: 1,
            acquisition_end_ns: 2,
            cpu_stat_text: cpu_stat
                .iter()
                .fold(String::new(), |mut output, (key, value)| {
                    writeln!(&mut output, "{key} {value}")
                        .expect("writing to a String cannot fail");
                    output
                }),
            cpu_stat,
            cpu_max_text: Some(if finite {
                "100000 100000\n".to_owned()
            } else {
                "max 100000\n".to_owned()
            }),
            cpu_limit: CpuLimit {
                quota_usec: finite.then_some(100_000),
                period_usec: Some(100_000),
                state: if finite { "finite" } else { "unlimited" },
            },
            errors: Vec::new(),
        }
    }

    #[test]
    fn cpuset_parser_is_strict_and_deduplicates() {
        assert_eq!(parse_cpuset("3,1-2,2").unwrap(), BTreeSet::from([1, 2, 3]));
        assert!(parse_cpuset("1,,2").is_err());
        assert!(parse_cpuset("3-1").is_err());
        assert!(parse_cpuset("-1").is_err());
    }

    #[test]
    fn post_attach_error_surfaces_bcc_cleanup_failure_once() {
        let mut close_attempts = 0;
        let result = with_scheduler_cleanup(
            Err::<(), _>(ProfileError::new(
                "injected post-attach measurement failure",
            )),
            || {
                close_attempts += 1;
                Err(ProfileError::new(
                    "BCC scheduler cleanup failed: injected detach failure",
                ))
            },
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("injected post-attach measurement failure"));
        assert!(error.contains("BCC scheduler cleanup failed: injected detach failure"));
        assert_eq!(close_attempts, 1);
    }

    #[test]
    fn post_attach_cancellation_surfaces_bcc_cleanup_failure_once() {
        let mut close_attempts = 0;
        let result = with_scheduler_cleanup(
            Err::<(), _>(ProfileError::new(
                "profile cancelled while scheduler trace was attached",
            )),
            || {
                close_attempts += 1;
                Err(ProfileError::new(
                    "BCC scheduler cleanup failed: injected close failure",
                ))
            },
        );
        let error = result.unwrap_err().to_string();
        assert!(error.contains("profile cancelled while scheduler trace was attached"));
        assert!(error.contains("BCC scheduler cleanup failed: injected close failure"));
        assert_eq!(close_attempts, 1);
    }

    #[test]
    fn scheduler_event_coverage_failure_is_an_explicit_invalid_reason() {
        let report = json!({
            "anomalies": [],
            "lost_perf_events": 0,
            "trace_complete": true,
            "events_observed": 0,
            "events_ignored_before_window": 0,
            "events_ignored_after_window": 0,
            "cgroup_validation": {
                "within_tolerance": true,
                "authoritative_cgroup_usage_sec": 0.004,
            },
        });
        assert_eq!(
            scheduler_invalid_reasons(&report),
            [Value::String(
                "scheduler trace observed no raw events despite nonzero cgroup CPU usage"
                    .to_owned()
            )]
        );
    }

    #[test]
    fn scheduler_warning_names_the_selected_backend() {
        let bcc = scheduler_instrumentation_warning("bcc");
        assert!(bcc.contains("BCC/eBPF scheduler-tracing overhead"));
        assert!(!bcc.contains("perf scheduler-tracing overhead"));
        assert!(
            scheduler_instrumentation_warning("perf").contains("perf scheduler-tracing overhead")
        );
    }

    #[test]
    fn output_preflight_creates_nothing() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("nested/profile");
        preflight_output(&output).unwrap();
        assert!(!root.path().join("nested").exists());
        assert!(!output.exists());
    }

    #[test]
    fn output_preflight_rejects_symlink() {
        use std::os::unix::fs::symlink;
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target");
        fs::create_dir(&target).unwrap();
        let link = root.path().join("link");
        symlink(&target, &link).unwrap();
        assert!(preflight_output(&link)
            .unwrap_err()
            .to_string()
            .contains("symbolic link"));
    }

    #[test]
    fn existing_empty_output_has_one_atomic_race_winner() {
        use std::sync::{mpsc, Arc, Barrier};
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("profile");
        fs::create_dir(&output).unwrap();
        let start = Arc::new(Barrier::new(3));
        let finish = Arc::new(Barrier::new(3));
        let (send, receive) = mpsc::channel();
        let mut threads = Vec::new();
        for _ in 0..2 {
            let output = output.clone();
            let start = Arc::clone(&start);
            let finish = Arc::clone(&finish);
            let send = send.clone();
            threads.push(thread::spawn(move || {
                start.wait();
                let claim = create_output(&output);
                send.send(claim.is_ok()).unwrap();
                finish.wait();
                drop(claim);
            }));
        }
        start.wait();
        let winners = (0..2)
            .map(|_| receive.recv().unwrap())
            .filter(|won| *won)
            .count();
        assert_eq!(winners, 1);
        finish.wait();
        for handle in threads {
            handle.join().unwrap();
        }
        assert!(!output.join(".tasktrace-profile-claim").exists());
    }

    #[test]
    fn perf_scheduler_cli_preserves_explicit_and_named_coordination_semantics() {
        let base = [
            "--image",
            "image",
            "--command",
            "true",
            "--output-dir",
            "/tmp/profile",
            "--cpuset-cpus",
            "2-3",
            "--scheduler-backend",
            "perf",
            "--scheduler-perf-binary",
            "/usr/bin/perf",
            "--scheduler-perf-mmap-pages",
            "128",
            "--scheduler-cpu-tolerance-fraction",
            "0.03",
            "--scheduler-cpu-tolerance-sec",
            "0.007",
        ];
        let explicit = base
            .iter()
            .copied()
            .chain(["--scheduler-concurrency"])
            .map(OsString::from)
            .collect::<Vec<_>>();
        let explicit = parse_arguments(&explicit).unwrap().unwrap();
        assert!(explicit.scheduler_enabled());
        assert!(!explicit.is_coordinated());
        assert!(!explicit.cpu_evidence_selected());
        assert_eq!(explicit.effective_strace_mode(), StraceMode::Execve);
        assert_eq!(explicit.scheduler_perf_binary, "/usr/bin/perf");
        assert_eq!(explicit.scheduler_perf_mmap_pages, Some(128));
        assert_eq!(explicit.scheduler_cpu_tolerance_fraction, 0.03);
        assert_eq!(explicit.scheduler_cpu_tolerance_sec, 0.007);

        let peak = base
            .iter()
            .copied()
            .chain(["--profile-set", "cpu-peak"])
            .map(OsString::from)
            .collect::<Vec<_>>();
        let peak = parse_arguments(&peak).unwrap().unwrap();
        assert!(peak.scheduler_enabled());
        assert!(peak.is_coordinated());
        assert!(peak.cpu_evidence_selected());
        assert!(!peak.memory_enabled());
        assert_eq!(peak.effective_strace_mode(), StraceMode::None);

        let compare = base
            .iter()
            .copied()
            .chain(["--profile-set", "cpu-compare"])
            .map(OsString::from)
            .collect::<Vec<_>>();
        let compare = parse_arguments(&compare).unwrap().unwrap();
        assert!(compare.scheduler_enabled());
        assert!(compare.is_coordinated());
        assert!(compare.cpu_evidence_selected());
        assert!(compare.memory_enabled());

        let invalid_mmap = base
            .iter()
            .copied()
            .chain([
                "--scheduler-concurrency",
                "--scheduler-perf-mmap-pages",
                "0",
            ])
            .map(OsString::from)
            .collect::<Vec<_>>();
        assert!(parse_arguments(&invalid_mmap)
            .unwrap_err()
            .to_string()
            .contains("--scheduler-perf-mmap-pages must be positive"));
    }

    #[test]
    fn bcc_scheduler_cli_preserves_buffer_pages_and_coordination_semantics() {
        let values = |items: &[&str]| items.iter().map(OsString::from).collect::<Vec<_>>();
        let explicit = values(&[
            "--image",
            "image",
            "--command",
            "true",
            "--output-dir",
            "/tmp/profile",
            "--cpuset-cpus",
            "2-3",
            "--scheduler-concurrency",
            "--scheduler-backend",
            "bcc",
            "--scheduler-bcc-buffer-pages",
            "64",
        ]);
        let explicit = parse_arguments(&explicit).unwrap().unwrap();
        assert!(explicit.scheduler_enabled());
        assert_eq!(explicit.scheduler_backend, "bcc");
        assert_eq!(explicit.scheduler_bcc_buffer_pages, Some(64));
        assert!(!explicit.is_coordinated());

        let named = values(&[
            "--image",
            "image",
            "--command",
            "true",
            "--output-dir",
            "/tmp/profile",
            "--cpuset-cpus",
            "2-3",
            "--profile-set",
            "cpu-peak",
            "--scheduler-bcc-buffer-pages=128",
        ]);
        let named = parse_arguments(&named).unwrap().unwrap();
        assert!(named.scheduler_enabled());
        assert_eq!(named.scheduler_backend, "bcc");
        assert_eq!(named.scheduler_bcc_buffer_pages, Some(128));
        assert!(named.is_coordinated());

        let invalid = values(&[
            "--image",
            "image",
            "--command",
            "true",
            "--output-dir",
            "/tmp/profile",
            "--cpuset-cpus",
            "2-3",
            "--scheduler-concurrency",
            "--scheduler-bcc-buffer-pages",
            "3",
        ]);
        assert!(parse_arguments(&invalid)
            .unwrap_err()
            .to_string()
            .contains("positive power of two"));
    }

    #[test]
    fn proc_stat_cli_is_native_and_requires_a_concrete_cpuset() {
        let arguments = [
            "--image",
            "image",
            "--command",
            "true",
            "--output-dir",
            "/tmp/profile",
            "--cpuset-cpus",
            "2-3",
            "--proc-stat-comparison",
            "--proc-stat-interval-sec",
            "0.125",
            "--proc-stat-baseline-sec",
            "0.25",
        ]
        .into_iter()
        .map(OsString::from)
        .collect::<Vec<_>>();
        let config = parse_arguments(&arguments).unwrap().unwrap();
        assert!(config.proc_stat_comparison);
        assert_eq!(config.proc_stat_interval_sec, 0.125);
        assert_eq!(config.proc_stat_baseline_sec, 0.25);
        assert_eq!(config.effective_strace_mode(), StraceMode::Execve);
        assert!(!config.is_coordinated());

        let without_cpuset = arguments
            .iter()
            .enumerate()
            .filter(|(index, _)| !matches!(*index, 6 | 7))
            .map(|(_, value)| value.clone())
            .collect::<Vec<_>>();
        assert!(parse_arguments(&without_cpuset)
            .unwrap_err()
            .to_string()
            .contains("requires --cpuset-cpus"));
    }

    #[test]
    fn memory_cli_preserves_legacy_flag_and_named_set_coordination_semantics() {
        let base = [
            "--image",
            "image",
            "--command",
            "true",
            "--output-dir",
            "/tmp/profile",
            "--cpuset-cpus",
            "2-3",
        ];
        let explicit = base
            .iter()
            .copied()
            .chain(["--memory-profile"])
            .map(OsString::from)
            .collect::<Vec<_>>();
        let explicit = parse_arguments(&explicit).unwrap().unwrap();
        assert!(explicit.memory_enabled());
        assert!(!explicit.is_coordinated());
        assert!(!explicit.cpu_evidence_selected());
        assert_eq!(explicit.effective_strace_mode(), StraceMode::Execve);

        let detail = base
            .iter()
            .copied()
            .chain(["--profile-set", "memory-detail"])
            .map(OsString::from)
            .collect::<Vec<_>>();
        let detail = parse_arguments(&detail).unwrap().unwrap();
        assert!(detail.memory_enabled());
        assert!(detail.is_coordinated());
        assert!(!detail.cpu_evidence_selected());
        assert_eq!(detail.effective_strace_mode(), StraceMode::None);

        let combined = base
            .iter()
            .copied()
            .chain([
                "--profile-set",
                "memory-detail",
                "--profile-set",
                "cpu-average",
            ])
            .map(OsString::from)
            .collect::<Vec<_>>();
        let combined = parse_arguments(&combined).unwrap().unwrap();
        assert!(combined.cpu_evidence_selected());
    }

    #[test]
    fn memory_detail_manifest_selects_memory_without_certifying_cpu() {
        let mut config = ProfileConfig::new("image", "true", "/tmp/out");
        config.cpuset_cpus = Some("0".to_owned());
        config.profile_sets = vec!["memory-detail".to_owned()];
        let cpu = json!({
            "valid_for_cpu_performance": false,
            "invalid_reasons": ["finite cpu.max constrained the diagnostic CPU report"],
        });
        let memory = json!({
            "valid_for_observed_memory_reporting": true,
            "valid_for_sizing": true,
            "instrumentation_valid": true,
            "peak_valid": true,
            "timeline_valid_for_shape": true,
            "policy_stable": true,
            "observation_boundary_complete": true,
            "oom_kill_observed": false,
            "limits": {"memory_max_state": "unlimited", "memory_max_bytes": Value::Null},
            "events_delta": {},
            "pressure_delta": {},
        });
        let report = build_profile_report(
            &config,
            &json!({"id": "sha256:image"}),
            "00000000-0000-4000-8000-000000000000",
            "2026-07-06T00:00:00+00:00",
            0,
            0,
            &cpu,
            None,
            None,
            None,
            Some(&memory),
            None,
            &json!({}),
            &json!({}),
            Some(Path::new("/tmp/out/evidence_bundle.json")),
            Some(&"a".repeat(64)),
            &"b".repeat(64),
            "scope",
            &json!({"requested": false, "valid": Value::Null, "reasons": []}),
            &json!({"scope_id": "scope"}),
            &json!({}),
        )
        .unwrap();
        assert_eq!(report["profiling_valid"], true);
        assert_eq!(
            report["evidence_bundle"]["resolved_profiler_ids"],
            json!(["cgroup-memory"])
        );
        assert_eq!(
            report["evidence_bundle"]["profiler_manifests"]
                .as_array()
                .unwrap()
                .len(),
            1
        );
        assert_eq!(
            report["evidence_bundle"]["profiler_manifests"][0]["profiler_id"],
            "cgroup-memory"
        );
        assert_eq!(report["cpu"]["valid_for_cpu_performance"], false);

        config.profile_sets.clear();
        config.memory_profile = true;
        let uncoordinated = build_profile_report(
            &config,
            &json!({"id": "sha256:image"}),
            "00000000-0000-4000-8000-000000000000",
            "2026-07-06T00:00:00+00:00",
            0,
            0,
            &cpu,
            None,
            None,
            None,
            Some(&memory),
            None,
            &json!({}),
            &json!({}),
            None,
            None,
            &"b".repeat(64),
            "scope",
            &json!({"requested": false, "valid": Value::Null, "reasons": []}),
            &json!({"scope_id": "scope"}),
            &json!({}),
        )
        .unwrap();
        assert!(uncoordinated.get("evidence_bundle").is_none());
        assert_eq!(uncoordinated["memory"]["valid_for_sizing"], true);
    }

    #[test]
    fn process_trace_collection_retains_raw_structured_and_unalignable_evidence() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("process.strace.log"),
            concat!(
                "41 1700000000.000000001 execve(\"/bin/sh\", [\"sh\", \"-c\", \"true\"], 0x1) = 0\n",
                "41 1700000000.010000001 +++ exited with 0 +++\n",
            ),
        )
        .unwrap();
        let mut config = ProfileConfig::new("image", "true", directory.path());
        config.cpuset_cpus = Some("0".to_owned());
        config.workdir = Some("/".to_owned());
        config.strace_mode = StraceMode::Process;
        config.profile_sets = vec!["software-deep".to_owned()];
        assert_eq!(config.effective_strace_mode(), StraceMode::Process);
        assert!(!config.cpu_evidence_selected());

        let report = collect_process_exec_report(&config, directory.path())
            .unwrap()
            .expect("process report selected");
        assert_eq!(report["valid"], true);
        assert_eq!(report["summary"]["process_count"], 1);
        assert_eq!(
            report["artifacts"]["raw_trace_path"],
            json!(directory.path().join("process.strace.log"))
        );
        assert!(directory.path().join("process_exec_report.json").is_file());

        let bundle = build_evidence_bundle(
            &config,
            "00000000-0000-4000-8000-000000000000",
            "scope",
            &json!({"scope_id": "scope", "identity": {"time_namespace": "time:[1]"}}),
            &json!({"clock_id": "host-monotonic", "start_ns": 1, "end_ns": 2}),
            &json!({"clock_id": "CLOCK_MONOTONIC"}),
            &json!({}),
            &"a".repeat(64),
            &json!({"invalid_reasons": []}),
            None,
            None,
            None,
            None,
            Some(&report),
            &json!({}),
            &json!({"requested": false, "valid": Value::Null, "reasons": []}),
            directory.path(),
        )
        .unwrap();
        assert_eq!(bundle["profilers"].as_array().unwrap().len(), 1);
        assert_eq!(bundle["profilers"][0]["profiler_id"], "process-exec");
        assert_eq!(bundle["profilers"][0]["status"], "unalignable");
        assert_eq!(bundle["clock_mappings"][0]["status"], "unalignable");
        assert_eq!(bundle["extensions"]["evidence_status"], "partial");
        assert_eq!(bundle["extensions"]["all_selected_profilers_valid"], false);
        assert_eq!(bundle["extensions"]["all_selected_profilers_usable"], true);
        assert_eq!(
            bundle["streams"]
                .as_array()
                .unwrap()
                .iter()
                .map(|stream| stream["metric"].as_str().unwrap())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from([
                "software.distinct_binary_count",
                "software.exec",
                "software.process_count",
            ])
        );
        assert_eq!(
            bundle["artifacts"]
                .as_array()
                .unwrap()
                .iter()
                .map(|artifact| artifact["artifact_id"].as_str().unwrap())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["process-exec-report", "process-strace-raw"])
        );

        let profile = build_profile_report(
            &config,
            &json!({"id": "sha256:image"}),
            "00000000-0000-4000-8000-000000000000",
            "2026-07-06T00:00:00+00:00",
            0,
            0,
            &json!({
                "valid_for_cpu_performance": false,
                "invalid_reasons": ["diagnostic CPU was not selected"],
            }),
            None,
            None,
            None,
            None,
            Some(&report),
            &json!({}),
            &json!({}),
            Some(&directory.path().join("evidence_bundle.json")),
            Some(&"b".repeat(64)),
            &"a".repeat(64),
            "scope",
            &json!({"requested": false, "valid": Value::Null, "reasons": []}),
            &json!({"scope_id": "scope"}),
            &json!({}),
        )
        .unwrap();
        assert_eq!(profile["profiling_valid"], true);
        assert_eq!(profile["strace"]["mode"], "process");
        assert_eq!(profile["evidence_bundle"]["evidence_status"], "partial");
        assert_eq!(
            profile["evidence_bundle"]["all_selected_profilers_valid"],
            false
        );
        assert_eq!(
            profile["evidence_bundle"]["resolved_profiler_ids"],
            json!(["process-exec"])
        );
        assert_artifact_descriptor_for_test(&profile["strace"]["raw_artifact"]);
        assert_artifact_descriptor_for_test(&profile["strace"]["structured_artifact"]);
    }

    #[test]
    fn hardware_deep_selects_only_hardware_evidence_and_retains_raw_artifacts() {
        let directory = tempfile::tempdir().unwrap();
        fs::write(
            directory.path().join("hardware_perf_stat.csv"),
            "123;;cycles;/system.slice/docker-example.scope;1000000;100.00;;\n",
        )
        .unwrap();
        fs::write(directory.path().join("hardware_perf.stderr.txt"), b"").unwrap();
        let hardware = json!({
            "schema_version": crate::hardware_perf::HARDWARE_PERF_SCHEMA_VERSION,
            "method": "perf-stat-cgroup-cpu-wide",
            "scope": "measured container cgroup on configured logical CPUs",
            "perf_cgroup": "/system.slice/docker-example.scope",
            "cpuset_cpus": "0",
            "window_start_ns": 100_u64,
            "window_end_ns": 200_u64,
            "window_start_alignment_error_ns": 3_u64,
            "window_end_alignment_error_ns": 4_u64,
            "definite_active_window_start_ns": 103_u64,
            "definite_active_window_end_ns": 196_u64,
            "control_bounds": {
                "attach_ready": {"action": "disable", "request_monotonic_ns": 1, "acknowledgement_monotonic_ns": 2, "uncertainty_ns": 1, "clock": "CLOCK_MONOTONIC"},
                "enable": {"action": "enable", "request_monotonic_ns": 100, "acknowledgement_monotonic_ns": 103, "uncertainty_ns": 3, "clock": "CLOCK_MONOTONIC"},
                "disable": {"action": "disable", "request_monotonic_ns": 196, "acknowledgement_monotonic_ns": 200, "uncertainty_ns": 4, "clock": "CLOCK_MONOTONIC"},
            },
            "minimum_running_ratio": 0.95,
            "hybrid_pmu_counts_merged": true,
            "provenance": {"perf_binary": "perf", "perf_version": "perf version test"},
            "counters": [{
                "event": "cycles",
                "namespace": "portable",
                "status": "valid",
                "value": 123,
                "unit": Value::Null,
                "time_enabled_ns": 1_000_000_u64,
                "time_running_ns": 1_000_000_u64,
                "time_running_fraction": 1.0,
                "valid": true,
                "invalid_reason": Value::Null,
                "observed_cgroup": "/system.slice/docker-example.scope",
            }],
            "anomalies": [],
            "instrumentation_valid": true,
            "all_requested_metrics_valid": true,
        });
        write_json(
            &directory.path().join("hardware_perf_report.json"),
            &hardware,
        )
        .unwrap();
        let mut config = ProfileConfig::new("image", "true", directory.path());
        config.cpuset_cpus = Some("0".to_owned());
        config.profile_sets = vec!["hardware-deep".to_owned()];
        let cpu = json!({
            "valid_for_cpu_performance": false,
            "invalid_reasons": ["diagnostic CPU gate was not selected"],
            "average_logical_cores": 0.0,
            "average_logical_cores_used": 0.0,
            "system_sec": 0.0,
            "usage_sec": 0.0,
            "user_sec": 0.0,
            "wall_sec": 1.0,
        });
        let verification = json!({"requested": false, "valid": Value::Null, "reasons": []});
        let bundle = build_evidence_bundle(
            &config,
            "00000000-0000-4000-8000-000000000000",
            "scope",
            &json!({"scope_id": "scope", "identity": {}}),
            &json!({"clock_id": "host-monotonic", "start_ns": 1, "end_ns": 300}),
            &json!({"clock_id": "CLOCK_MONOTONIC"}),
            &json!({}),
            &"a".repeat(64),
            &cpu,
            None,
            Some(&hardware),
            None,
            None,
            None,
            &json!({}),
            &verification,
            directory.path(),
        )
        .unwrap();
        assert_eq!(
            bundle["profilers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|profiler| profiler["profiler_id"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["hardware-perf"]
        );
        assert_eq!(bundle["streams"][0]["metric"], "hardware.cycles");
        assert_eq!(bundle["streams"][0]["records"][0]["value"], 123);
        assert_eq!(bundle["streams"][0]["records"][0]["alignment_error_ns"], 4);
        assert_eq!(bundle["artifacts"].as_array().unwrap().len(), 3);
        assert!(bundle["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .all(|artifact| {
                artifact["sha256"]
                    .as_str()
                    .is_some_and(|value| value.len() == 64)
                    && artifact["size_bytes"].as_u64().is_some()
            }));
        assert_eq!(
            bundle["extensions"]["profiler_statuses"]["cgroup-cpu"],
            "not-selected"
        );
        assert_eq!(
            bundle["extensions"]["raw_profiler_reports"]["cgroup-cpu"],
            cpu
        );

        let profile = build_profile_report(
            &config,
            &json!({"id": "sha256:image"}),
            "00000000-0000-4000-8000-000000000000",
            "2026-07-06T00:00:00+00:00",
            0,
            0,
            &cpu,
            None,
            Some(&hardware),
            None,
            None,
            None,
            &json!({}),
            &json!({}),
            Some(&directory.path().join("evidence_bundle.json")),
            Some(&"b".repeat(64)),
            &"a".repeat(64),
            "scope",
            &verification,
            &json!({"scope_id": "scope"}),
            &json!({}),
        )
        .unwrap();
        assert_eq!(profile["profiling_valid"], true);
        assert_eq!(profile["hardware_perf"], hardware);
        assert_eq!(
            profile["evidence_bundle"]["resolved_profiler_ids"],
            json!(["hardware-perf"])
        );
        assert_eq!(profile["cpu"]["valid_for_cpu_performance"], false);
    }

    #[test]
    fn cpu_peak_selects_valid_perf_scheduler_scalar_and_interval_evidence() {
        let directory = tempfile::tempdir().unwrap();
        for filename in [
            "cgroup_cpu.stat.controlled_finish_gate.txt",
            "cgroup_cpu.stat.observer_start.txt",
            "cgroup_cpu_source_snapshots.json",
            "scheduler_perf.data",
            "scheduler_perf_switch_events.txt",
            "scheduler_perf.stderr.txt",
        ] {
            fs::write(directory.path().join(filename), b"fixture\n").unwrap();
        }
        let events = vec![
            crate::scheduler_perf::SchedulerEvent {
                timestamp_ns: 100_000_000,
                cpu: 0,
                tid: 42,
                entering: true,
            },
            crate::scheduler_perf::SchedulerEvent {
                timestamp_ns: 900_000_000,
                cpu: 0,
                tid: 42,
                entering: false,
            },
        ];
        crate::scheduler_perf::write_scheduler_events(
            &directory.path().join("scheduler_events.jsonl"),
            &events,
        )
        .unwrap();
        let summary = crate::scheduler_perf::summarize_scheduler_events(
            &events,
            0,
            1_000_000_000,
            0,
            "perf-record-cgroup-native-switch-events",
            "fixture cgroup",
        )
        .unwrap();
        let mut scheduler =
            crate::scheduler_perf::add_cgroup_validation(&summary, 0.8, 0.02, 0.005).unwrap();
        let scheduler_object = scheduler.as_object_mut().unwrap();
        scheduler_object.insert("window_start_ns".to_owned(), json!(0));
        scheduler_object.insert("window_end_ns".to_owned(), json!(1_000_000_000_u64));

        let mut config = ProfileConfig::new("image", "true", directory.path());
        config.cpuset_cpus = Some("0".to_owned());
        config.profile_sets = vec!["cpu-peak".to_owned()];
        config.scheduler_backend = "perf".to_owned();
        let cpu = json!({
            "valid_for_cpu_performance": true,
            "invalid_reasons": [],
            "average_logical_cores": 0.8,
            "average_logical_cores_used": 0.8,
            "system_sec": 0.1,
            "usage_sec": 0.8,
            "user_sec": 0.7,
            "wall_sec": 1.0,
        });
        let verification = json!({"requested": false, "valid": Value::Null, "reasons": []});
        let bundle = build_evidence_bundle(
            &config,
            "00000000-0000-4000-8000-000000000000",
            "scope",
            &json!({"scope_id": "scope", "identity": {}}),
            &json!({"clock_id": "host-monotonic", "start_ns": 0, "end_ns": 1_000_000_000_u64}),
            &json!({"clock_id": "CLOCK_MONOTONIC"}),
            &json!({}),
            &"a".repeat(64),
            &cpu,
            Some(&scheduler),
            None,
            None,
            None,
            None,
            &json!({}),
            &verification,
            directory.path(),
        )
        .unwrap();
        assert_eq!(bundle["extensions"]["evidence_status"], "valid");
        assert_eq!(
            bundle["profilers"]
                .as_array()
                .unwrap()
                .iter()
                .map(|profiler| profiler["profiler_id"].as_str().unwrap())
                .collect::<BTreeSet<_>>(),
            BTreeSet::from(["cgroup-cpu", "scheduler-concurrency"])
        );
        let metrics = bundle["streams"]
            .as_array()
            .unwrap()
            .iter()
            .map(|stream| stream["metric"].as_str().unwrap())
            .collect::<BTreeSet<_>>();
        assert!(metrics.contains("cpu.peak_simultaneous_logical_cores"));
        assert!(metrics.contains("scheduler.concurrent_logical_cpus"));
        assert!(metrics.contains("scheduler.logical_cpu_activity"));
        assert_eq!(
            bundle["streams"]
                .as_array()
                .unwrap()
                .iter()
                .find(|stream| stream["metric"] == "scheduler.concurrent_logical_cpus")
                .unwrap()["records"]
                .as_array()
                .unwrap()
                .iter()
                .map(|record| record["value"].as_u64().unwrap())
                .max(),
            Some(1)
        );

        let profile = build_profile_report(
            &config,
            &json!({"id": "sha256:image"}),
            "00000000-0000-4000-8000-000000000000",
            "2026-07-06T00:00:00+00:00",
            0,
            0,
            &cpu,
            Some(&scheduler),
            None,
            None,
            None,
            None,
            &json!({}),
            &json!({}),
            Some(&directory.path().join("evidence_bundle.json")),
            Some(&"b".repeat(64)),
            &"a".repeat(64),
            "scope",
            &verification,
            &json!({"scope_id": "scope"}),
            &json!({}),
        )
        .unwrap();
        assert_eq!(profile["profiling_valid"], true);
        assert_eq!(
            profile["evidence_bundle"]["resolved_profiler_ids"],
            json!(["scheduler-concurrency", "cgroup-cpu"])
        );

        scheduler["cgroup_validation"]["valid_for_concurrency_reporting"] = json!(false);
        let invalid_run = ProfileRun {
            report: json!({"scheduler_concurrency": scheduler}),
            report_path: PathBuf::from("report.json"),
            evidence_bundle_path: None,
            command_exit_code: 0,
        };
        assert_eq!(public_result_exit_code(&invalid_run), 1);
    }

    #[test]
    fn cpu_peak_selects_typed_bcc_scheduler_evidence_and_only_its_raw_events() {
        let directory = tempfile::tempdir().unwrap();
        for filename in [
            "cgroup_cpu.stat.controlled_finish_gate.txt",
            "cgroup_cpu.stat.observer_start.txt",
            "cgroup_cpu_source_snapshots.json",
        ] {
            fs::write(directory.path().join(filename), b"fixture\n").unwrap();
        }
        let events = vec![
            crate::scheduler_perf::SchedulerEvent {
                timestamp_ns: 100_000_000,
                cpu: 0,
                tid: 42,
                entering: true,
            },
            crate::scheduler_perf::SchedulerEvent {
                timestamp_ns: 900_000_000,
                cpu: 0,
                tid: 42,
                entering: false,
            },
        ];
        crate::scheduler_perf::write_scheduler_events(
            &directory.path().join("scheduler_events.jsonl"),
            &events,
        )
        .unwrap();
        let mut scheduler =
            crate::scheduler_bcc::summarize_bcc_events(&events, 0, 1_000_000_000, 0).unwrap();
        scheduler["backend"] = json!("bcc");
        scheduler = crate::scheduler_perf::add_cgroup_validation_with_event_tolerance(
            &scheduler,
            0.8,
            0.02,
            0.005,
            crate::scheduler_bcc::EVENT_BOUNDARY_TOLERANCE_SEC,
        )
        .unwrap();
        scheduler["window_start_ns"] = json!(0);
        scheduler["window_end_ns"] = json!(1_000_000_000_u64);

        let mut config = ProfileConfig::new("image", "true", directory.path());
        config.cpuset_cpus = Some("0".to_owned());
        config.profile_sets = vec!["cpu-peak".to_owned()];
        config.scheduler_backend = "bcc".to_owned();
        config.scheduler_bcc_buffer_pages = Some(64);
        let cpu = json!({
            "valid_for_cpu_performance": true,
            "invalid_reasons": [],
            "average_logical_cores": 0.8,
            "average_logical_cores_used": 0.8,
            "system_sec": 0.1,
            "usage_sec": 0.8,
            "user_sec": 0.7,
            "wall_sec": 1.0,
        });
        let verification = json!({"requested": false, "valid": Value::Null, "reasons": []});
        let bundle = build_evidence_bundle(
            &config,
            "00000000-0000-4000-8000-000000000000",
            "scope",
            &json!({"scope_id": "scope", "identity": {}}),
            &json!({"clock_id": "host-monotonic", "start_ns": 0, "end_ns": 1_000_000_000_u64}),
            &json!({"clock_id": "CLOCK_MONOTONIC"}),
            &json!({}),
            &"a".repeat(64),
            &cpu,
            Some(&scheduler),
            None,
            None,
            None,
            None,
            &json!({}),
            &verification,
            directory.path(),
        )
        .unwrap();
        let scheduler_profiler = bundle["profilers"]
            .as_array()
            .unwrap()
            .iter()
            .find(|profiler| profiler["profiler_id"] == "scheduler-concurrency")
            .unwrap();
        assert_eq!(scheduler_profiler["configuration"]["backend"], "bcc");
        assert_eq!(scheduler_profiler["configuration"]["bcc_buffer_pages"], 64);
        assert_eq!(
            scheduler_profiler["provenance"]["requirements"],
            json!([
                "cgroup-v2-descendant-scope",
                "root",
                "runtime-loadable-libbcc",
                "kernel-ebpf-kprobe",
                "bpf_get_current_ancestor_cgroup_id"
            ])
        );
        assert_eq!(
            bundle["artifacts"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|artifact| artifact["artifact_id"] == "scheduler-events")
                .count(),
            1
        );
        assert!(!bundle["artifacts"]
            .as_array()
            .unwrap()
            .iter()
            .any(|artifact| artifact["artifact_id"] == "scheduler-perf-data"));
        assert_eq!(
            scheduler["cgroup_validation"]["per_observed_event_tolerance_sec"],
            crate::scheduler_bcc::EVENT_BOUNDARY_TOLERANCE_SEC
        );
        assert!(bundle["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|stream| stream["metric"] == "scheduler.concurrent_logical_cpus"));
    }

    #[test]
    fn proc_stat_evidence_is_host_context_not_task_concurrency() {
        let stream = proc_stat_window_stream(
            "run",
            "scope",
            "proc-stat-measurement-peak-active",
            "host.cpu.sampled_peak_active_logical_cpus",
            "logical_cpu",
            Value::from(2),
            &json!({"start_ns": 100, "end_ns": 200}),
            0.05,
        )
        .unwrap();
        assert_eq!(stream["profiler_id"], "proc-stat-host-context");
        assert_eq!(stream["records"][0]["canonical_start_ns"], 100);
        assert_eq!(stream["records"][0]["canonical_end_ns"], 200);
        assert_eq!(stream["records"][0]["attributes"]["task_attributed"], false);
        assert_eq!(stream["provenance"]["peak_is_not_task_concurrency"], true);
        assert_eq!(stream["raw_artifact_ids"][0], "proc-stat-samples");
    }

    #[test]
    fn cpu_report_preserves_metric_but_fails_closed_for_finite_limit() {
        let mut config = ProfileConfig::new("image", "true", "/tmp/out");
        config.cpuset_cpus = Some("0-1".to_owned());
        let before = snapshot("leaf", 10, 6, 4, true);
        let after = snapshot("leaf", 2_000_010, 1_500_006, 500_004, true);
        let drain = DescendantDrain {
            initial: Vec::new(),
            remaining: Vec::new(),
            signaled_term: Vec::new(),
            signaled_kill: Vec::new(),
            waited_ns: 0,
        };
        let report = cpu_report(
            &config,
            "scope",
            1_000_000_000,
            2_000_000_000,
            &before,
            &after,
            &[],
            &[],
            &drain,
        )
        .unwrap();
        assert_eq!(report["usage_sec"], 2.0);
        assert_eq!(report["average_logical_cores_used"], 2.0);
        assert_eq!(report["utilization_percent_of_allocation"], 100.0);
        assert_eq!(report["valid_for_cpu_performance"], false);
        assert!(report["invalid_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value.as_str().unwrap().contains("finite cpu.max")));
    }

    #[test]
    fn controller_mount_is_content_bound() {
        let executable = env::current_exe().unwrap();
        let digest = sha256_file(&executable).unwrap();
        let mount = format!(
            "type=bind,src={},dst=/tasktrace-controller/tasktrace,readonly",
            executable.display()
        );
        let value = mount_provenance(&[mount], &digest).unwrap();
        assert_eq!(
            value[0]["source_identity"]["kind"],
            "rust-controller-binary"
        );
        assert_eq!(value[0]["source_identity"]["sha256"], digest);
    }

    #[test]
    fn ephemeral_host_mount_paths_do_not_change_comparability_identity() {
        let controller_digest = "a".repeat(64);
        let left = mount_provenance(
            &["type=bind,src=/tmp/run-one/output,dst=/tasktrace".to_owned()],
            &controller_digest,
        )
        .unwrap();
        let right = mount_provenance(
            &["type=bind,src=/tmp/run-two/output,dst=/tasktrace".to_owned()],
            &controller_digest,
        )
        .unwrap();
        assert_eq!(left, right);
        assert_eq!(left[0]["source_identity"]["kind"], "external-bind");
        assert_eq!(
            left[0]["source_identity"]["host_path_in_comparability"],
            false
        );
        assert!(!left.to_string().contains("run-one"));
    }

    #[test]
    fn public_status_fails_for_command_or_profile_invalidity() {
        let make = |command_exit_code, profiling_valid: Option<bool>| ProfileRun {
            report: profiling_valid
                .map(|valid| json!({"profiling_valid": valid}))
                .unwrap_or_else(|| json!({})),
            report_path: PathBuf::from("report.json"),
            evidence_bundle_path: None,
            command_exit_code,
        };
        assert_eq!(public_result_exit_code(&make(0, Some(true))), 0);
        assert_eq!(public_result_exit_code(&make(1, Some(true))), 1);
        assert_eq!(public_result_exit_code(&make(0, Some(false))), 1);
        assert_eq!(public_result_exit_code(&make(0, None)), 0);
    }

    #[test]
    #[ignore = "requires a local cgroup-v2 Docker daemon and explicit image/cpuset"]
    fn real_docker_cpu_average_gate() {
        let image =
            env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE").expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE");
        let cpuset =
            env::var("TASKTRACE_RUST_DOCKER_CPUSET").expect("set TASKTRACE_RUST_DOCKER_CPUSET");
        let root = tempfile::tempdir().unwrap();
        let mut config = ProfileConfig::new(
            image,
            "python3 -c 'sum(i*i for i in range(2000000))'",
            root.path().join("profile"),
        );
        config.cpuset_cpus = Some(cpuset.clone());
        config.network_policy =
            NetworkPolicy::new(NetworkMode::NoNetwork, std::iter::empty::<&str>()).unwrap();
        let result = run_profile(config, None).unwrap();
        assert_eq!(result.command_exit_code, 0);
        assert!(result.report["cpu"]["usage_sec"].as_f64().unwrap() > 0.0);
        assert!(result.report["cpu"]["wall_sec"].as_f64().unwrap() > 0.0);
        assert_eq!(
            result.report["cpu"]["allocated_logical_cpus"]
                .as_f64()
                .unwrap(),
            parse_cpuset(&cpuset).unwrap().len() as f64
        );
        assert_eq!(result.report["profiling_valid"], true);
        assert_eq!(
            result.report["evidence_bundle"]["scope"]["identity"]["cpu_placement"]
                ["validation_status"],
            "passed"
        );
        let bundle = result.evidence_bundle_path.unwrap();
        assert!(bundle.is_file());
        let evidence: Value = serde_json::from_slice(&fs::read(bundle).unwrap()).unwrap();
        assert_eq!(evidence["extensions"]["evidence_status"], "valid");
        assert_eq!(
            evidence["scope"]["identity"]["cpu_placement"]["validation_status"],
            "passed"
        );
        let start = evidence["timeline"]["start_ns"].as_u64().unwrap();
        let measured_start = result.report["cpu"]["window_start_ns"].as_u64().unwrap();
        let measured_end = result.report["cpu"]["window_end_ns"].as_u64().unwrap();
        let end = evidence["timeline"]["end_ns"].as_u64().unwrap();
        assert!(start <= measured_start && measured_start < measured_end && measured_end <= end);
    }
}
