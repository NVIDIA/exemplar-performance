// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native profiling adapter for one transcript trial.
//!
//! Transcript interpretation stays in `transcript_replay`; disposable Docker
//! lifecycle, cgroup accounting, CPU placement, cancellation, and cleanup stay
//! in `profile_container`.  The current native executable is content-bound and
//! mounted read-only into the measured container, so the replay workload runs
//! once inside the measured cgroup without inheriting a host Python runtime.

#[cfg(test)]
use crate::network_policy::NetworkMode;
use crate::network_policy::{resolve_cli_network_policy, NetworkPolicy};
use crate::profile_container::{self, ProfileConfig};
use crate::transcript_replay::{self, Extraction};
use serde_json::{json, Value};
use serde_yaml::Value as YamlValue;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

const TRANSCRIPT_MOUNT: &str = "/tasktrace-input";
const REPLAY_MOUNT: &str = "/tasktrace";
const INITIAL_WORKSPACE_MOUNT: &str = "/tasktrace-initial-workspace";
const CONTROLLER_EXECUTABLE: &str = "/tasktrace-controller/tasktrace";
const PIPELINE_SCHEMA_VERSION: &str = "container-transcript-profile.v1";

#[derive(Debug)]
struct TranscriptProfileError(String);

impl TranscriptProfileError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TranscriptProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TranscriptProfileError {}

impl From<io::Error> for TranscriptProfileError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for TranscriptProfileError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_yaml::Error> for TranscriptProfileError {
    fn from(error: serde_yaml::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug)]
struct TranscriptProfileConfig {
    runtime: String,
    transcript: PathBuf,
    task_id: String,
    trial_name: String,
    image: String,
    output_dir: PathBuf,
    cpuset_cpus: Option<String>,
    allocated_cpus: Option<f64>,
    source_workspace: Option<PathBuf>,
    initial_workspace: Option<PathBuf>,
    pinchbench_skill_dir: Option<PathBuf>,
    timeout_sec: f64,
    prepare_command: Option<String>,
    verification_command: Option<String>,
    strace_mode: String,
    string_limit: u64,
    shell: String,
    workdir: String,
    memory_profile: bool,
    memory_sample_interval_sec: f64,
    memory_limit_bytes: Option<u64>,
    memory_swap_total_limit_bytes: Option<u64>,
    hardware_perf_binary: String,
    hardware_perf_events: Vec<String>,
    hardware_perf_attach_timeout: Duration,
    hardware_perf_minimum_running_ratio: f64,
    scheduler_concurrency: bool,
    scheduler_backend: String,
    scheduler_bcc_buffer_pages: Option<u64>,
    scheduler_perf_binary: String,
    scheduler_perf_mmap_pages: Option<u64>,
    scheduler_cpu_tolerance_fraction: f64,
    scheduler_cpu_tolerance_sec: f64,
    proc_stat_comparison: bool,
    proc_stat_interval_sec: f64,
    proc_stat_baseline_sec: f64,
    network: Option<String>,
    network_policy: NetworkPolicy,
    network_policy_proxy_image: String,
    network_policy_external_network: String,
    docker: String,
    enroot: String,
    allow_unsupported: bool,
    profile_sets: Vec<String>,
    coordinated_baseline: bool,
    attach_timeout: Duration,
}

#[derive(Default)]
struct ParsedArguments {
    runtime: String,
    transcript: Option<PathBuf>,
    task_id: Option<String>,
    trial_name: Option<String>,
    image: Option<String>,
    output_dir: Option<PathBuf>,
    cpuset_cpus: Option<String>,
    allocated_cpus: Option<f64>,
    source_workspace: Option<PathBuf>,
    initial_workspace: Option<PathBuf>,
    pinchbench_skill_dir: Option<PathBuf>,
    timeout_sec: f64,
    prepare_command: Option<String>,
    verification_command: Option<String>,
    strace_mode: String,
    string_limit: i64,
    shell: String,
    workdir: String,
    memory_profile: bool,
    memory_sample_interval_sec: f64,
    memory_limit_bytes: Option<i64>,
    memory_swap_total_limit_bytes: Option<i64>,
    hardware_perf_binary: String,
    hardware_perf_events: Vec<String>,
    hardware_perf_attach_timeout_sec: f64,
    hardware_perf_minimum_running_ratio: f64,
    scheduler_concurrency: bool,
    scheduler_backend: String,
    scheduler_bcc_buffer_pages: Option<i64>,
    scheduler_perf_binary: String,
    scheduler_perf_mmap_pages: Option<i64>,
    scheduler_cpu_tolerance_fraction: f64,
    scheduler_cpu_tolerance_sec: f64,
    proc_stat_comparison: bool,
    proc_stat_interval_sec: f64,
    proc_stat_baseline_sec: f64,
    network: Option<String>,
    network_mode: Option<String>,
    allowed_hosts: Vec<String>,
    network_proxy_image: String,
    network_external: String,
    docker: String,
    enroot: String,
    allow_unsupported: bool,
    profile_sets: Vec<String>,
    coordinated_baseline: bool,
    scheduler_attach_timeout_sec: f64,
}

impl ParsedArguments {
    fn defaults() -> Self {
        Self {
            runtime: "docker".to_owned(),
            timeout_sec: 60.0,
            strace_mode: "execve".to_owned(),
            string_limit: 4096,
            shell: "/bin/bash".to_owned(),
            workdir: "/workspace".to_owned(),
            proc_stat_interval_sec: 0.5,
            proc_stat_baseline_sec: 5.0,
            memory_sample_interval_sec: 0.1,
            hardware_perf_binary: "perf".to_owned(),
            hardware_perf_attach_timeout_sec: 20.0,
            hardware_perf_minimum_running_ratio: crate::hardware_perf::MINIMUM_RUNNING_RATIO,
            scheduler_backend: "bcc".to_owned(),
            scheduler_perf_binary: "perf".to_owned(),
            scheduler_cpu_tolerance_fraction: 0.02,
            scheduler_cpu_tolerance_sec: 0.005,
            network_proxy_image: "python:3.12-slim".to_owned(),
            network_external: "bridge".to_owned(),
            docker: "docker".to_owned(),
            enroot: "enroot".to_owned(),
            scheduler_attach_timeout_sec: 20.0,
            ..Self::default()
        }
    }
}

fn text_argument(value: &OsStr, option: &str) -> Result<String, TranscriptProfileError> {
    value
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| TranscriptProfileError::new(format!("{option} must be valid UTF-8")))
}

fn split_long_option(value: &str) -> (&str, Option<&str>) {
    value
        .split_once('=')
        .map_or((value, None), |(option, argument)| (option, Some(argument)))
}

fn parse_number<T>(value: &str, option: &str) -> Result<T, TranscriptProfileError>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| TranscriptProfileError::new(format!("invalid value for {option}: {value:?}")))
}

fn parse_arguments(
    arguments: &[OsString],
) -> Result<Option<TranscriptProfileConfig>, TranscriptProfileError> {
    if arguments.len() == 1 && matches!(arguments[0].to_str(), Some("-h" | "--help")) {
        return Ok(None);
    }
    let mut parsed = ParsedArguments::defaults();
    let mut index = 0;
    while index < arguments.len() {
        let raw = text_argument(&arguments[index], "argument")?;
        if matches!(raw.as_str(), "-h" | "--help") {
            return Ok(None);
        }
        let (option, attached) = split_long_option(&raw);
        let flag = |attached: Option<&str>| -> Result<(), TranscriptProfileError> {
            if attached.is_some() {
                Err(TranscriptProfileError::new(format!(
                    "{option} does not take a value"
                )))
            } else {
                Ok(())
            }
        };
        let mut take = || -> Result<String, TranscriptProfileError> {
            if let Some(value) = attached {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .ok_or_else(|| {
                    TranscriptProfileError::new(format!("argument {option}: expected one argument"))
                })
                .and_then(|value| text_argument(value, option))
        };
        match option {
            "--runtime" => parsed.runtime = take()?,
            "--task-id" => parsed.task_id = Some(take()?),
            "--trial-name" => parsed.trial_name = Some(take()?),
            "--image" => parsed.image = Some(take()?),
            "--output-dir" => parsed.output_dir = Some(PathBuf::from(take()?)),
            "--cpuset-cpus" => parsed.cpuset_cpus = Some(take()?),
            "--allocated-cpus" => {
                let value = take()?;
                parsed.allocated_cpus = Some(parse_number(&value, option)?);
            }
            "--source-workspace" => parsed.source_workspace = Some(PathBuf::from(take()?)),
            "--initial-workspace" => parsed.initial_workspace = Some(PathBuf::from(take()?)),
            "--pinchbench-skill-dir" => parsed.pinchbench_skill_dir = Some(PathBuf::from(take()?)),
            "--timeout-sec" => {
                let value = take()?;
                parsed.timeout_sec = parse_number(&value, option)?;
            }
            "--prepare-command" => parsed.prepare_command = Some(take()?),
            "--verification-command" => parsed.verification_command = Some(take()?),
            "--strace-mode" => parsed.strace_mode = take()?,
            "--strace-string-limit" => {
                let value = take()?;
                parsed.string_limit = parse_number(&value, option)?;
            }
            "--shell" => parsed.shell = take()?,
            "--workdir" => parsed.workdir = take()?,
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
            "--hardware-perf-binary" => parsed.hardware_perf_binary = take()?,
            "--hardware-perf-event" => parsed.hardware_perf_events.push(take()?),
            "--hardware-perf-attach-timeout-sec" => {
                let value = take()?;
                parsed.hardware_perf_attach_timeout_sec = parse_number(&value, option)?;
            }
            "--hardware-perf-minimum-running-ratio" => {
                let value = take()?;
                parsed.hardware_perf_minimum_running_ratio = parse_number(&value, option)?;
            }
            "--scheduler-concurrency" => {
                flag(attached)?;
                parsed.scheduler_concurrency = true;
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
            "--scheduler-cpu-tolerance-fraction" => {
                let value = take()?;
                parsed.scheduler_cpu_tolerance_fraction = parse_number(&value, option)?;
            }
            "--scheduler-cpu-tolerance-sec" => {
                let value = take()?;
                parsed.scheduler_cpu_tolerance_sec = parse_number(&value, option)?;
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
            "--network" => parsed.network = Some(take()?),
            "--network-policy" => parsed.network_mode = Some(take()?),
            "--allowed-host" => parsed.allowed_hosts.push(take()?),
            "--network-policy-proxy-image" => parsed.network_proxy_image = take()?,
            "--network-policy-external-network" => parsed.network_external = take()?,
            "--docker" => parsed.docker = take()?,
            "--enroot" => parsed.enroot = take()?,
            "--allow-unsupported" => {
                flag(attached)?;
                parsed.allow_unsupported = true;
            }
            "--profile-set" => parsed.profile_sets.push(take()?),
            "--coordinated-baseline" => {
                flag(attached)?;
                parsed.coordinated_baseline = true;
            }
            "--scheduler-attach-timeout-sec" => {
                let value = take()?;
                parsed.scheduler_attach_timeout_sec = parse_number(&value, option)?;
            }
            "--" => {
                index += 1;
                while index < arguments.len() {
                    if parsed.transcript.is_some() {
                        return Err(TranscriptProfileError::new(format!(
                            "unrecognized arguments: {}",
                            arguments[index].to_string_lossy()
                        )));
                    }
                    parsed.transcript = Some(PathBuf::from(&arguments[index]));
                    index += 1;
                }
                break;
            }
            value if value.starts_with('-') => {
                return Err(TranscriptProfileError::new(format!(
                    "unrecognized arguments: {raw}"
                )))
            }
            _ if parsed.transcript.is_none() => {
                parsed.transcript = Some(PathBuf::from(&arguments[index]))
            }
            _ => {
                return Err(TranscriptProfileError::new(format!(
                    "unrecognized arguments: {raw}"
                )))
            }
        }
        index += 1;
    }
    validate_arguments(parsed).map(Some)
}

fn positive_finite(value: f64, option: &str) -> Result<f64, TranscriptProfileError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(TranscriptProfileError::new(format!(
            "{option} must be positive"
        )))
    }
}

fn positive_bytes(value: i64, option: &str) -> Result<u64, TranscriptProfileError> {
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| TranscriptProfileError::new(format!("{option} must be positive")))
}

fn validate_arguments(
    parsed: ParsedArguments,
) -> Result<TranscriptProfileConfig, TranscriptProfileError> {
    let transcript = parsed.transcript.ok_or_else(|| {
        TranscriptProfileError::new("the following arguments are required: transcript")
    })?;
    let task_id = parsed
        .task_id
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            TranscriptProfileError::new("the following arguments are required: --task-id")
        })?;
    let trial_name = parsed
        .trial_name
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            TranscriptProfileError::new("the following arguments are required: --trial-name")
        })?;
    let image = parsed
        .image
        .filter(|value| !value.is_empty())
        .ok_or_else(|| {
            TranscriptProfileError::new("the following arguments are required: --image")
        })?;
    let output_dir = parsed.output_dir.ok_or_else(|| {
        TranscriptProfileError::new("the following arguments are required: --output-dir")
    })?;
    if !matches!(parsed.runtime.as_str(), "docker" | "enroot") {
        return Err(TranscriptProfileError::new(format!(
            "invalid choice for --runtime: {:?} (choose from 'docker', 'enroot')",
            parsed.runtime
        )));
    }
    let timeout_sec = positive_finite(parsed.timeout_sec, "--timeout-sec")?;
    let allocated_cpus = parsed
        .allocated_cpus
        .map(|value| positive_finite(value, "--allocated-cpus"))
        .transpose()?;
    if parsed.cpuset_cpus.is_none() && allocated_cpus.is_none() {
        return Err(TranscriptProfileError::new(
            "provide --cpuset-cpus or --allocated-cpus for utilization reporting",
        ));
    }
    if let Some(cpuset) = &parsed.cpuset_cpus {
        profile_container::parse_cpuset(cpuset)
            .map_err(|error| TranscriptProfileError::new(error.to_string()))?;
    }
    if parsed.initial_workspace.is_some() && parsed.pinchbench_skill_dir.is_some() {
        return Err(TranscriptProfileError::new(
            "provide either --initial-workspace or --pinchbench-skill-dir, not both",
        ));
    }
    if !matches!(parsed.strace_mode.as_str(), "execve" | "process" | "none") {
        return Err(TranscriptProfileError::new(format!(
            "invalid choice for --strace-mode: {:?} (choose from 'execve', 'process', 'none')",
            parsed.strace_mode
        )));
    }
    if parsed.string_limit <= 0 {
        return Err(TranscriptProfileError::new(
            "--strace-string-limit must be positive",
        ));
    }
    if parsed.coordinated_baseline && !parsed.profile_sets.is_empty() {
        return Err(TranscriptProfileError::new(
            "--coordinated-baseline is mutually exclusive with --profile-set",
        ));
    }
    let mut observed = BTreeSet::new();
    if parsed
        .profile_sets
        .iter()
        .any(|value| !observed.insert(value.clone()))
    {
        return Err(TranscriptProfileError::new(
            "profile sets must not contain duplicates",
        ));
    }
    if let Some(unsupported) = parsed.profile_sets.iter().find(|value| {
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
        return Err(TranscriptProfileError::new(format!(
            "native profile transcript does not yet support --profile-set {unsupported}; cpu-average, cpu-peak, cpu-compare, memory-detail, software-deep, and hardware-deep are available"
        )));
    }
    let scheduler_requested = parsed.scheduler_concurrency
        || parsed
            .profile_sets
            .iter()
            .any(|value| matches!(value.as_str(), "cpu-peak" | "cpu-compare"));
    if parsed
        .verification_command
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(TranscriptProfileError::new(
            "verification command must be a nonempty string",
        ));
    }
    if parsed.verification_command.is_some()
        && !parsed.coordinated_baseline
        && parsed.profile_sets.is_empty()
        && !parsed.proc_stat_comparison
    {
        return Err(TranscriptProfileError::new(
            "--verification-command requires --profile-set or --coordinated-baseline",
        ));
    }
    let memory_limit_bytes = parsed
        .memory_limit_bytes
        .map(|value| positive_bytes(value, "--memory-limit-bytes"))
        .transpose()?;
    let memory_swap_total_limit_bytes = parsed
        .memory_swap_total_limit_bytes
        .map(|value| positive_bytes(value, "--memory-swap-total-limit-bytes"))
        .transpose()?;
    if memory_swap_total_limit_bytes.is_some() && memory_limit_bytes.is_none() {
        return Err(TranscriptProfileError::new(
            "--memory-swap-total-limit-bytes requires --memory-limit-bytes",
        ));
    }
    if memory_swap_total_limit_bytes
        .zip(memory_limit_bytes)
        .is_some_and(|(swap, memory)| swap < memory)
    {
        return Err(TranscriptProfileError::new(
            "--memory-swap-total-limit-bytes must be at least --memory-limit-bytes",
        ));
    }
    positive_finite(parsed.proc_stat_interval_sec, "--proc-stat-interval-sec")?;
    positive_finite(parsed.proc_stat_baseline_sec, "--proc-stat-baseline-sec")?;
    positive_finite(
        parsed.memory_sample_interval_sec,
        "--memory-sample-interval-sec",
    )?;
    positive_finite(
        parsed.hardware_perf_attach_timeout_sec,
        "--hardware-perf-attach-timeout-sec",
    )?;
    if parsed.hardware_perf_binary.trim().is_empty() {
        return Err(TranscriptProfileError::new(
            "--hardware-perf-binary must be nonempty",
        ));
    }
    if !parsed.hardware_perf_minimum_running_ratio.is_finite()
        || parsed.hardware_perf_minimum_running_ratio <= 0.0
        || parsed.hardware_perf_minimum_running_ratio > 1.0
    {
        return Err(TranscriptProfileError::new(
            "--hardware-perf-minimum-running-ratio must be in (0, 1]",
        ));
    }
    if parsed
        .hardware_perf_events
        .iter()
        .any(|event| event.trim().is_empty())
        || parsed
            .hardware_perf_events
            .iter()
            .collect::<BTreeSet<_>>()
            .len()
            != parsed.hardware_perf_events.len()
    {
        return Err(TranscriptProfileError::new(
            "hardware perf events must be nonempty and unique",
        ));
    }
    if !matches!(parsed.scheduler_backend.as_str(), "bcc" | "perf") {
        return Err(TranscriptProfileError::new(
            "invalid choice for --scheduler-backend (choose from 'bcc', 'perf')",
        ));
    }
    if !scheduler_requested && parsed.scheduler_backend != "bcc" {
        return Err(TranscriptProfileError::new(
            "--scheduler-backend requires --scheduler-concurrency",
        ));
    }
    if let Some(value) = parsed.scheduler_bcc_buffer_pages {
        if value <= 0 || (value & (value - 1)) != 0 {
            return Err(TranscriptProfileError::new(
                "--scheduler-bcc-buffer-pages must be a positive power of two",
            ));
        }
    }
    if !scheduler_requested && parsed.scheduler_bcc_buffer_pages.is_some() {
        return Err(TranscriptProfileError::new(
            "--scheduler-bcc-buffer-pages requires --scheduler-concurrency or a scheduler profile set",
        ));
    }
    if parsed
        .scheduler_perf_mmap_pages
        .is_some_and(|value| value <= 0)
    {
        return Err(TranscriptProfileError::new(
            "--scheduler-perf-mmap-pages must be positive",
        ));
    }
    if scheduler_requested
        && parsed.scheduler_backend == "perf"
        && parsed.scheduler_perf_binary.trim().is_empty()
    {
        return Err(TranscriptProfileError::new(
            "--scheduler-perf-binary must be nonempty",
        ));
    }
    if !parsed.scheduler_cpu_tolerance_fraction.is_finite()
        || parsed.scheduler_cpu_tolerance_fraction < 0.0
        || !parsed.scheduler_cpu_tolerance_sec.is_finite()
        || parsed.scheduler_cpu_tolerance_sec < 0.0
    {
        return Err(TranscriptProfileError::new(
            "scheduler CPU tolerances must be finite and nonnegative",
        ));
    }
    if parsed.proc_stat_comparison && parsed.cpuset_cpus.is_none() {
        return Err(TranscriptProfileError::new(
            "--proc-stat-comparison requires --cpuset-cpus",
        ));
    }
    let attach_timeout = positive_finite(
        parsed.scheduler_attach_timeout_sec,
        "--scheduler-attach-timeout-sec",
    )?;
    let resolved_policy = resolve_cli_network_policy(
        parsed.network_mode.as_deref(),
        &parsed.allowed_hosts,
        parsed.network.as_deref(),
    )
    .map_err(|error| TranscriptProfileError::new(error.to_string()))?;
    let network_policy = resolved_policy.unwrap_or_default();
    if parsed.runtime == "enroot" {
        if parsed.network.is_some()
            || network_policy.effective_mode() != crate::network_policy::NetworkMode::Public
        {
            return Err(TranscriptProfileError::new(
                "non-public network policy is currently Docker-only",
            ));
        }
        if parsed.strace_mode != "none" {
            return Err(TranscriptProfileError::new(
                "strace tracing requires Docker ptrace capabilities; use --strace-mode none",
            ));
        }
        if parsed.scheduler_concurrency {
            return Err(TranscriptProfileError::new(
                "--scheduler-concurrency is currently Docker-only",
            ));
        }
        if parsed.proc_stat_comparison {
            return Err(TranscriptProfileError::new(
                "--proc-stat-comparison is currently Docker-only",
            ));
        }
        if parsed.memory_profile {
            return Err(TranscriptProfileError::new(
                "--memory-profile is currently Docker-only",
            ));
        }
        if memory_limit_bytes.is_some() {
            return Err(TranscriptProfileError::new(
                "--memory-limit-bytes is currently Docker-only",
            ));
        }
        if memory_swap_total_limit_bytes.is_some() {
            return Err(TranscriptProfileError::new(
                "--memory-swap-total-limit-bytes is currently Docker-only",
            ));
        }
        if !parsed.profile_sets.is_empty() {
            return Err(TranscriptProfileError::new(
                "--profile-set currently requires the Docker runtime",
            ));
        }
        if parsed.coordinated_baseline {
            return Err(TranscriptProfileError::new(
                "--coordinated-baseline currently requires the Docker runtime",
            ));
        }
        if parsed.verification_command.is_some() {
            return Err(TranscriptProfileError::new(
                "--verification-command currently requires Docker",
            ));
        }
    }
    Ok(TranscriptProfileConfig {
        runtime: parsed.runtime,
        transcript,
        task_id,
        trial_name,
        image,
        output_dir,
        cpuset_cpus: parsed.cpuset_cpus,
        allocated_cpus,
        source_workspace: parsed.source_workspace,
        initial_workspace: parsed.initial_workspace,
        pinchbench_skill_dir: parsed.pinchbench_skill_dir,
        timeout_sec,
        prepare_command: parsed.prepare_command,
        verification_command: parsed.verification_command,
        strace_mode: parsed.strace_mode,
        string_limit: parsed.string_limit as u64,
        shell: parsed.shell,
        workdir: parsed.workdir,
        memory_profile: parsed.memory_profile,
        memory_sample_interval_sec: parsed.memory_sample_interval_sec,
        memory_limit_bytes,
        memory_swap_total_limit_bytes,
        hardware_perf_binary: parsed.hardware_perf_binary,
        hardware_perf_events: if parsed.hardware_perf_events.is_empty() {
            crate::hardware_perf::PORTABLE_HARDWARE_EVENTS
                .iter()
                .map(|event| (*event).to_owned())
                .collect()
        } else {
            parsed.hardware_perf_events
        },
        hardware_perf_attach_timeout: Duration::from_secs_f64(
            parsed.hardware_perf_attach_timeout_sec,
        ),
        hardware_perf_minimum_running_ratio: parsed.hardware_perf_minimum_running_ratio,
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
        network: parsed.network,
        network_policy,
        network_policy_proxy_image: parsed.network_proxy_image,
        network_policy_external_network: parsed.network_external,
        docker: parsed.docker,
        enroot: parsed.enroot,
        allow_unsupported: parsed.allow_unsupported,
        profile_sets: parsed.profile_sets,
        coordinated_baseline: parsed.coordinated_baseline,
        attach_timeout: Duration::from_secs_f64(attach_timeout),
    })
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn sha256_file(path: &Path) -> Result<String, TranscriptProfileError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn absolute_normalized(path: &Path) -> Result<PathBuf, TranscriptProfileError> {
    if path.exists() {
        return path.canonicalize().map_err(Into::into);
    }
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut normalized = PathBuf::from("/");
    for component in absolute.components() {
        match component {
            Component::RootDir | Component::CurDir => {}
            Component::Normal(value) => normalized.push(value),
            Component::ParentDir => {
                normalized.pop();
            }
            Component::Prefix(_) => {
                return Err(TranscriptProfileError::new(
                    "Windows path prefixes are unsupported",
                ))
            }
        }
    }
    Ok(normalized)
}

fn collect_tree_entries(
    root: &Path,
    directory: &Path,
    entries: &mut Vec<PathBuf>,
) -> Result<(), TranscriptProfileError> {
    let mut children = fs::read_dir(directory)?.collect::<Result<Vec<_>, _>>()?;
    children.sort_by(|left, right| {
        left.file_name()
            .as_bytes()
            .cmp(right.file_name().as_bytes())
    });
    for child in children {
        let path = child.path();
        let metadata = fs::symlink_metadata(&path)?;
        entries.push(path.clone());
        if metadata.file_type().is_dir() {
            collect_tree_entries(root, &path, entries)?;
        }
    }
    entries.sort_by(|left, right| {
        left.strip_prefix(root)
            .unwrap_or(left)
            .as_os_str()
            .as_bytes()
            .cmp(
                right
                    .strip_prefix(root)
                    .unwrap_or(right)
                    .as_os_str()
                    .as_bytes(),
            )
    });
    Ok(())
}

/// Content identity for starting workspaces, matching the established adapter.
fn path_tree_sha256(path: &Path) -> Result<String, TranscriptProfileError> {
    if !path.exists() {
        return Err(TranscriptProfileError::new(format!(
            "workspace provenance input does not exist: {}",
            path.display()
        )));
    }
    let root = path.canonicalize()?;
    let root_metadata = fs::symlink_metadata(&root)?;
    let mut entries = vec![root.clone()];
    if root_metadata.file_type().is_dir() {
        collect_tree_entries(&root, &root, &mut entries)?;
    }
    let mut digest = Sha256::new();
    for entry in entries {
        let relative = if entry == root {
            b".".to_vec()
        } else {
            entry
                .strip_prefix(&root)
                .expect("collected entry stays under root")
                .as_os_str()
                .as_bytes()
                .to_vec()
        };
        let metadata = fs::symlink_metadata(&entry)?;
        let (kind, payload_size) = if metadata.file_type().is_symlink() {
            (
                "symlink",
                fs::read_link(&entry)?.as_os_str().as_bytes().len() as u64,
            )
        } else if metadata.file_type().is_dir() {
            ("directory", 0)
        } else if metadata.file_type().is_file() {
            ("file", metadata.len())
        } else {
            return Err(TranscriptProfileError::new(format!(
                "unsupported workspace entry for provenance: {}",
                entry.display()
            )));
        };
        digest.update(kind.as_bytes());
        digest.update(b"\0");
        digest.update(&relative);
        digest.update(b"\0");
        digest.update(format!("{:o}", metadata.mode() & 0o7777).as_bytes());
        digest.update(b"\0");
        digest.update(payload_size.to_string().as_bytes());
        digest.update(b"\0");
        if metadata.file_type().is_symlink() {
            digest.update(fs::read_link(&entry)?.as_os_str().as_bytes());
        } else if metadata.file_type().is_file() {
            let mut file = File::open(&entry)?;
            let mut buffer = [0_u8; 1024 * 1024];
            loop {
                let count = file.read(&mut buffer)?;
                if count == 0 {
                    break;
                }
                digest.update(&buffer[..count]);
            }
        }
        digest.update(b"\0");
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn safe_relative_path(value: &str, description: &str) -> Result<PathBuf, TranscriptProfileError> {
    let path = Path::new(value);
    if path.is_absolute() {
        return Err(TranscriptProfileError::new(format!(
            "{description} escapes its root: {value}"
        )));
    }
    let mut relative = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => relative.push(value),
            Component::CurDir => {}
            Component::ParentDir | Component::RootDir | Component::Prefix(_) => {
                return Err(TranscriptProfileError::new(format!(
                    "{description} escapes its root: {value}"
                )))
            }
        }
    }
    if relative.as_os_str().is_empty() {
        return Err(TranscriptProfileError::new(format!(
            "{description} must not be empty"
        )));
    }
    Ok(relative)
}

fn yaml_field<'a>(mapping: &'a serde_yaml::Mapping, key: &str) -> Option<&'a YamlValue> {
    mapping.get(YamlValue::String(key.to_owned()))
}

fn frontmatter(text: &str, task_file: &Path) -> Result<YamlValue, TranscriptProfileError> {
    let pattern =
        regex::Regex::new(r"(?s)^---\s*\n(.*?)\n---\s*\n").expect("frontmatter regex is valid");
    let body = pattern
        .captures(text)
        .and_then(|capture| capture.get(1))
        .ok_or_else(|| {
            TranscriptProfileError::new(format!(
                "PinchBench task definition has no YAML frontmatter: {}",
                task_file.display()
            ))
        })?;
    serde_yaml::from_str(body.as_str()).map_err(|error| {
        TranscriptProfileError::new(format!(
            "invalid PinchBench task frontmatter in {}: {error}",
            task_file.display()
        ))
    })
}

fn stage_pinchbench_initial_workspace(
    skill_dir: &Path,
    task_id: &str,
    destination: &Path,
) -> Result<(PathBuf, u64), TranscriptProfileError> {
    let canonical_task_id = task_id.rsplit("__").next().unwrap_or(task_id);
    let task_component = safe_relative_path(canonical_task_id, "PinchBench task ID")?;
    if task_component.components().count() != 1 {
        return Err(TranscriptProfileError::new(format!(
            "PinchBench task ID is not a filename component: {canonical_task_id}"
        )));
    }
    let task_file = skill_dir
        .join("tasks")
        .join(task_component)
        .with_extension("md");
    if !task_file.is_file() {
        return Err(TranscriptProfileError::new(format!(
            "PinchBench task definition does not exist: {}",
            task_file.display()
        )));
    }
    let metadata = frontmatter(&fs::read_to_string(&task_file)?, &task_file)?;
    let mapping = metadata.as_mapping().ok_or_else(|| {
        TranscriptProfileError::new(format!(
            "invalid PinchBench task frontmatter in {}",
            task_file.display()
        ))
    })?;
    let workspace_files = match yaml_field(mapping, "workspace_files") {
        None | Some(YamlValue::Null) => &[][..],
        Some(YamlValue::Sequence(values)) => values.as_slice(),
        Some(_) => {
            return Err(TranscriptProfileError::new(format!(
                "invalid PinchBench workspace_files in {}",
                task_file.display()
            )))
        }
    };
    fs::create_dir_all(destination)?;
    let assets_root = absolute_normalized(&skill_dir.join("assets"))?;
    let mut restored = 0_u64;
    for specification in workspace_files {
        let specification = specification.as_mapping().ok_or_else(|| {
            TranscriptProfileError::new(format!(
                "invalid PinchBench workspace file specification in {}",
                task_file.display()
            ))
        })?;
        if let Some(content) = yaml_field(specification, "content") {
            let relative = yaml_field(specification, "path")
                .and_then(YamlValue::as_str)
                .ok_or_else(|| {
                    TranscriptProfileError::new(format!(
                        "invalid inline PinchBench workspace file in {}",
                        task_file.display()
                    ))
                })?;
            let content = content.as_str().ok_or_else(|| {
                TranscriptProfileError::new(format!(
                    "invalid inline PinchBench workspace file in {}",
                    task_file.display()
                ))
            })?;
            let target = destination.join(safe_relative_path(relative, "workspace fixture path")?);
            if let Some(parent) = target.parent() {
                fs::create_dir_all(parent)?;
            }
            fs::write(target, content)?;
            restored += 1;
            continue;
        }
        let source_relative = yaml_field(specification, "source")
            .and_then(YamlValue::as_str)
            .ok_or_else(|| {
                TranscriptProfileError::new(format!(
                    "invalid sourced PinchBench workspace file in {}",
                    task_file.display()
                ))
            })?;
        let destination_relative = yaml_field(specification, "dest")
            .and_then(YamlValue::as_str)
            .ok_or_else(|| {
                TranscriptProfileError::new(format!(
                    "invalid sourced PinchBench workspace file in {}",
                    task_file.display()
                ))
            })?;
        let source = assets_root.join(safe_relative_path(source_relative, "fixture source")?);
        let canonical_source = source.canonicalize().map_err(|error| {
            TranscriptProfileError::new(format!(
                "PinchBench fixture source does not exist: {}: {error}",
                source.display()
            ))
        })?;
        if !canonical_source.starts_with(&assets_root) || !canonical_source.is_file() {
            return Err(TranscriptProfileError::new(format!(
                "PinchBench fixture source does not exist: {}",
                source.display()
            )));
        }
        let target = destination.join(safe_relative_path(
            destination_relative,
            "workspace fixture path",
        )?);
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(canonical_source, target)?;
        restored += 1;
    }
    Ok((task_file.canonicalize()?, restored))
}

static CLAIM_SEQUENCE: AtomicU64 = AtomicU64::new(0);

struct PipelineOutput {
    root: PathBuf,
    claim: PathBuf,
}

impl Drop for PipelineOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.claim);
    }
}

fn validate_output_root(path: &Path) -> Result<(), TranscriptProfileError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(TranscriptProfileError::new(format!(
                    "pipeline output path is not a directory: {}",
                    path.display()
                )));
            }
            if fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(TranscriptProfileError::new(format!(
                    "pipeline output directory must be new or empty: {}",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut ancestor = path.parent().unwrap_or_else(|| Path::new("."));
            while !ancestor.exists() {
                ancestor = ancestor.parent().ok_or_else(|| {
                    TranscriptProfileError::new(format!(
                        "pipeline output has no existing ancestor: {}",
                        path.display()
                    ))
                })?;
            }
            if !ancestor.is_dir() {
                return Err(TranscriptProfileError::new(format!(
                    "pipeline output ancestor is not a directory: {}",
                    ancestor.display()
                )));
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

fn prepare_pipeline_output(path: &Path) -> Result<PipelineOutput, TranscriptProfileError> {
    validate_output_root(path)?;
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent)?;
        }
        match fs::create_dir(path) {
            Ok(()) => {}
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {}
            Err(error) => return Err(error.into()),
        }
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(TranscriptProfileError::new(format!(
            "pipeline output path changed after preflight: {}",
            path.display()
        )));
    }
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Err(TranscriptProfileError::new(format!(
            "pipeline output directory changed after preflight: {}",
            path.display()
        )));
    }
    let root = path.canonicalize()?;
    let claim = root.join(".tasktrace-transcript-profile-claim");
    let mut options = OpenOptions::new();
    options.create_new(true).write(true).mode(0o600);
    let mut file = options.open(&claim).map_err(|error| {
        TranscriptProfileError::new(format!(
            "pipeline output directory could not be claimed atomically: {}: {error}",
            root.display()
        ))
    })?;
    file.write_all(
        format!(
            "pid={} sequence={}\n",
            std::process::id(),
            CLAIM_SEQUENCE.fetch_add(1, Ordering::Relaxed)
        )
        .as_bytes(),
    )?;
    file.sync_all()?;
    fs::create_dir(root.join("replay"))?;
    fs::create_dir(root.join("transcript_input"))?;
    Ok(PipelineOutput { root, claim })
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&byte))
    {
        value.to_owned()
    } else {
        format!("'{}'", value.replace('\'', "'\\''"))
    }
}

fn shell_join(arguments: &[String]) -> String {
    arguments
        .iter()
        .map(|argument| shell_quote(argument))
        .collect::<Vec<_>>()
        .join(" ")
}

fn python_float(value: f64) -> String {
    if value.fract() == 0.0 {
        format!("{value:.1}")
    } else {
        value.to_string()
    }
}

fn mount(source: &Path, target: &str, readonly: bool) -> Result<String, TranscriptProfileError> {
    let source = source.to_str().ok_or_else(|| {
        TranscriptProfileError::new(format!(
            "native Docker bind mount path must be valid UTF-8: {}",
            source.display()
        ))
    })?;
    if source.contains(',') {
        return Err(TranscriptProfileError::new(format!(
            "native Docker bind mount path must not contain a comma: {source}"
        )));
    }
    Ok(format!(
        "type=bind,src={source},dst={target}{}",
        if readonly { ",readonly" } else { "" }
    ))
}

fn replay_command(
    config: &TranscriptProfileConfig,
    transcript_name: &str,
    initial_workspace: bool,
) -> String {
    let mut arguments = vec![
        CONTROLLER_EXECUTABLE.to_owned(),
        "replay".to_owned(),
        "transcript".to_owned(),
        format!("{TRANSCRIPT_MOUNT}/{transcript_name}"),
        "--task-id".to_owned(),
        config.task_id.clone(),
        "--trial-name".to_owned(),
        config.trial_name.clone(),
        "--workspace".to_owned(),
        format!("{REPLAY_MOUNT}/workspace"),
        "--timeout-sec".to_owned(),
        python_float(config.timeout_sec),
        "--output-dir".to_owned(),
        format!("{REPLAY_MOUNT}/output"),
    ];
    if let Some(source_workspace) = &config.source_workspace {
        arguments.extend([
            "--source-workspace".to_owned(),
            source_workspace.to_string_lossy().into_owned(),
        ]);
    }
    if initial_workspace {
        arguments.extend([
            "--initial-workspace".to_owned(),
            INITIAL_WORKSPACE_MOUNT.to_owned(),
        ]);
    }
    shell_join(&arguments)
}

fn unsupported_tools(extraction: &Extraction) -> Vec<String> {
    let mut tools = BTreeSet::new();
    for action in &extraction.actions {
        if action.get("supported").and_then(Value::as_bool) != Some(true) {
            tools.insert(
                action
                    .get("source_tool")
                    .and_then(Value::as_str)
                    .unwrap_or("<unknown>")
                    .to_owned(),
            );
        }
    }
    tools.into_iter().collect()
}

fn check_cancelled(phase: &str) -> Result<(), TranscriptProfileError> {
    if profile_container::process_signal_cancelled() {
        Err(TranscriptProfileError::new(format!(
            "profile interrupted {phase}"
        )))
    } else {
        Ok(())
    }
}

struct PreparedProfile {
    profile: ProfileConfig,
    output: PipelineOutput,
    transcript_path: PathBuf,
    transcript_snapshot: PathBuf,
    transcript_sha256: String,
    controller_sha256: String,
    initial_workspace: Option<PathBuf>,
    initial_workspace_sha256: Option<String>,
    source_workspace_path: Option<PathBuf>,
    source_workspace_sha256: Option<String>,
    source_workspace_state: &'static str,
    task_definition: Option<PathBuf>,
    fixtures_restored: u64,
    action_count: usize,
    unsupported_action_count: usize,
}

fn prepare_profile(
    config: &TranscriptProfileConfig,
) -> Result<PreparedProfile, TranscriptProfileError> {
    validate_output_root(&config.output_dir)?;
    if !config.transcript.is_file() {
        return Err(TranscriptProfileError::new(format!(
            "transcript does not exist: {}",
            config.transcript.display()
        )));
    }
    if config
        .initial_workspace
        .as_ref()
        .is_some_and(|path| !path.is_dir())
    {
        return Err(TranscriptProfileError::new(format!(
            "initial workspace does not exist: {}",
            config
                .initial_workspace
                .as_ref()
                .expect("checked above")
                .display()
        )));
    }
    let transcript_path = config.transcript.canonicalize()?;
    let transcript_sha256 = sha256_file(&transcript_path)?;
    let extraction = transcript_replay::extract_actions(
        &transcript_path,
        Some(&config.task_id),
        Some(&config.trial_name),
    )
    .map_err(|error| TranscriptProfileError::new(error.to_string()))?;
    let unsupported = unsupported_tools(&extraction);
    if !unsupported.is_empty() && !config.allow_unsupported {
        return Err(TranscriptProfileError::new(format!(
            "selected trial contains unsupported tools; use --allow-unsupported for a partial replay: {}",
            unsupported.join(", ")
        )));
    }
    check_cancelled("during transcript validation")?;

    let source_workspace_path = config
        .source_workspace
        .as_deref()
        .map(absolute_normalized)
        .transpose()?;
    let (source_workspace_sha256, source_workspace_state) = match source_workspace_path.as_deref() {
        None => (None, "none"),
        Some(path) if path.exists() => (Some(path_tree_sha256(path)?), "hashed"),
        Some(_) => (None, "unavailable"),
    };
    let user_initial_sha256 = config
        .initial_workspace
        .as_deref()
        .map(path_tree_sha256)
        .transpose()?;
    check_cancelled("during workspace provenance collection")?;

    let controller = env::current_exe()?.canonicalize()?;
    if !controller.is_file() {
        return Err(TranscriptProfileError::new(format!(
            "native replay controller is not a regular file: {}",
            controller.display()
        )));
    }
    let controller_sha256 = sha256_file(&controller)?;
    let transcript_name = transcript_path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| TranscriptProfileError::new("transcript filename must be valid UTF-8"))?;

    let output = prepare_pipeline_output(&config.output_dir)?;
    let transcript_snapshot = output.root.join("transcript_input").join(transcript_name);
    let transcript_bytes = fs::read(&transcript_path)?;
    let observed_sha256 = sha256_bytes(&transcript_bytes);
    if observed_sha256 != transcript_sha256 {
        return Err(TranscriptProfileError::new(
            "transcript content changed during profile preparation",
        ));
    }
    let mut snapshot_options = OpenOptions::new();
    snapshot_options.create_new(true).write(true).mode(0o600);
    let mut snapshot_file = snapshot_options.open(&transcript_snapshot)?;
    snapshot_file.write_all(&transcript_bytes)?;
    snapshot_file.sync_all()?;
    if sha256_file(&transcript_snapshot)? != transcript_sha256 {
        return Err(TranscriptProfileError::new(
            "retained transcript snapshot failed its content identity check",
        ));
    }

    let mut task_definition = None;
    let mut fixtures_restored = 0_u64;
    let initial_workspace = if let Some(skill_dir) = &config.pinchbench_skill_dir {
        let destination = output.root.join("initial_workspace");
        let (definition, restored) =
            stage_pinchbench_initial_workspace(skill_dir, &config.task_id, &destination)?;
        task_definition = Some(definition);
        fixtures_restored = restored;
        Some(destination.canonicalize()?)
    } else {
        config
            .initial_workspace
            .as_deref()
            .map(Path::canonicalize)
            .transpose()?
    };
    let initial_workspace_sha256 = if config.pinchbench_skill_dir.is_some() {
        initial_workspace
            .as_deref()
            .map(path_tree_sha256)
            .transpose()?
    } else {
        user_initial_sha256
    };
    check_cancelled("during replay input staging")?;

    let replay_dir = output.root.join("replay");
    let mut profile = ProfileConfig::new(
        config.image.clone(),
        replay_command(config, transcript_name, initial_workspace.is_some()),
        output.root.join("container_profile"),
    );
    profile.prepare_command = config.prepare_command.clone();
    profile.verification_command = config.verification_command.clone();
    profile.cpuset_cpus = config.cpuset_cpus.clone();
    profile.allocated_cpus = config.allocated_cpus;
    profile.memory_limit_bytes = config.memory_limit_bytes;
    profile.memory_swap_total_limit_bytes = config.memory_swap_total_limit_bytes;
    profile.memory_profile = config.memory_profile;
    profile.memory_sample_interval_sec = config.memory_sample_interval_sec;
    profile.hardware_perf_binary = config.hardware_perf_binary.clone();
    profile.hardware_perf_events = config.hardware_perf_events.clone();
    profile.hardware_perf_attach_timeout = config.hardware_perf_attach_timeout;
    profile.hardware_perf_minimum_running_ratio = config.hardware_perf_minimum_running_ratio;
    profile.scheduler_concurrency = config.scheduler_concurrency;
    profile.scheduler_backend = config.scheduler_backend.clone();
    profile.scheduler_bcc_buffer_pages = config.scheduler_bcc_buffer_pages;
    profile.scheduler_perf_binary = config.scheduler_perf_binary.clone();
    profile.scheduler_perf_mmap_pages = config.scheduler_perf_mmap_pages;
    profile.scheduler_cpu_tolerance_fraction = config.scheduler_cpu_tolerance_fraction;
    profile.scheduler_cpu_tolerance_sec = config.scheduler_cpu_tolerance_sec;
    profile.proc_stat_comparison = config.proc_stat_comparison;
    profile.proc_stat_interval_sec = config.proc_stat_interval_sec;
    profile.proc_stat_baseline_sec = config.proc_stat_baseline_sec;
    profile.workdir = Some(config.workdir.clone());
    profile.shell = config.shell.clone();
    profile.string_limit = config.string_limit;
    profile.network = config.network.clone();
    profile.network_policy = config.network_policy.clone();
    profile.network_policy_proxy_image = config.network_policy_proxy_image.clone();
    profile.network_policy_external_network = config.network_policy_external_network.clone();
    profile.runtime = config.runtime.clone();
    profile.docker = config.docker.clone();
    profile.enroot = config.enroot.clone();
    profile.attach_timeout = config.attach_timeout;
    if config.coordinated_baseline {
        profile.set_coordinated_baseline();
    } else if config.profile_sets.is_empty() {
        profile.set_untraced_baseline();
        profile
            .set_strace_mode(&config.strace_mode)
            .map_err(|error| TranscriptProfileError::new(error.to_string()))?;
    } else {
        profile.profile_sets = config.profile_sets.clone();
    }
    profile.terminate_descendants_at_finish = config.runtime == "docker";
    profile.mounts = vec![
        mount(
            transcript_snapshot
                .parent()
                .expect("transcript snapshot has a parent"),
            TRANSCRIPT_MOUNT,
            true,
        )?,
        mount(&replay_dir, REPLAY_MOUNT, false)?,
        mount(&controller, CONTROLLER_EXECUTABLE, true)?,
    ];
    if let Some(initial_workspace) = &initial_workspace {
        profile
            .mounts
            .push(mount(initial_workspace, INITIAL_WORKSPACE_MOUNT, true)?);
    }
    profile.workload_identity = Some(json!({
        "task_id": config.task_id,
        "trial_name": config.trial_name,
    }));
    profile.source_identity = Some(json!({
        "format": "transcript-jsonl",
        "sha256": transcript_sha256,
        "action_count": extraction.actions.len(),
        "initial_workspace_sha256": initial_workspace_sha256,
        "source_workspace_sha256": source_workspace_sha256,
        "source_workspace_provenance_state": source_workspace_state,
        "rust_controller_binary_sha256": controller_sha256,
    }));
    profile.expected_outcome = Some(json!({
        "kind": "source-recorded-action-outcomes",
        "action_count": extraction.actions.len(),
        "allow_unsupported": config.allow_unsupported,
    }));
    Ok(PreparedProfile {
        profile,
        output,
        transcript_path,
        transcript_snapshot,
        transcript_sha256,
        controller_sha256,
        initial_workspace,
        initial_workspace_sha256,
        source_workspace_path,
        source_workspace_sha256,
        source_workspace_state,
        task_definition,
        fixtures_restored,
        action_count: extraction.actions.len(),
        unsupported_action_count: unsupported.len(),
    })
}

fn report_fields(config: &TranscriptProfileConfig, prepared: &PreparedProfile) -> Value {
    json!({
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "task_id": config.task_id,
        "trial_name": config.trial_name,
        "transcript": prepared.transcript_path.to_string_lossy(),
        "transcript_snapshot": prepared.transcript_snapshot.to_string_lossy(),
        "transcript_sha256": prepared.transcript_sha256,
        "initial_workspace": prepared.initial_workspace.as_ref().map(|path| path.to_string_lossy().into_owned()),
        "initial_workspace_sha256": prepared.initial_workspace_sha256,
        "source_workspace": prepared.source_workspace_path.as_ref().map(|path| path.to_string_lossy().into_owned()),
        "source_workspace_sha256": prepared.source_workspace_sha256,
        "source_workspace_provenance_state": prepared.source_workspace_state,
        "image": config.image,
        "runtime": config.runtime,
        "actions_validated": prepared.action_count,
        "allow_unsupported": config.allow_unsupported,
        "unsupported_actions_validated": prepared.unsupported_action_count,
        "replay_fidelity": {
            "status": if prepared.unsupported_action_count == 0 { "complete-for-implemented-tools" } else { "intentional-partial-replay" },
            "unsupported_action_count": prepared.unsupported_action_count,
            "allow_unsupported": config.allow_unsupported,
        },
        "task_definition": prepared.task_definition.as_ref().map(|path| path.to_string_lossy().into_owned()),
        "fixtures_restored": prepared.fixtures_restored,
        "workspace": prepared.output.root.join("replay/workspace").to_string_lossy(),
        "controller_runtime": "rust-host-mount",
        "controller_identity": {
            "kind": "rust-host-mounted-binary",
            "sha256": prepared.controller_sha256,
            "container_path": CONTROLLER_EXECUTABLE,
        },
    })
}

fn run_transcript_profile(
    config: &TranscriptProfileConfig,
) -> Result<Value, TranscriptProfileError> {
    let prepared = prepare_profile(config)?;
    let fields = report_fields(config, &prepared);
    let run = profile_container::run_profile(prepared.profile.clone(), None)
        .map_err(|error| TranscriptProfileError::new(error.to_string()))?;
    check_cancelled("before pipeline publication")?;
    let report = crate::profile_pipeline::finish_profiled_replay(
        &run.report,
        &prepared.output.root,
        Path::new("replay/output/replay_report.json"),
        fields,
        &config.profile_sets,
        config.coordinated_baseline,
    )
    .map_err(|error| TranscriptProfileError::new(error.to_string()))?;
    check_cancelled("during pipeline publication")?;
    Ok(report)
}

fn usage(program: &str, output: &mut dyn Write) {
    let _ = writeln!(
        output,
        "usage: {program} [-h] --task-id TASK_ID --trial-name TRIAL_NAME --image IMAGE --output-dir OUTPUT_DIR [options] transcript"
    );
}

fn help(program: &str) {
    usage(program, &mut io::stdout());
    println!();
    println!("Profile one extracted transcript trial in a disposable Docker or Enroot image.");
    println!();
    println!("The native replay controller is mounted read-only into the measured sandbox.");
    println!("Supported profiler modes: the default execve trace, explicit --strace-mode execve/process, untraced baseline, --memory-profile, --scheduler-concurrency with --scheduler-backend perf, --coordinated-baseline, --profile-set cpu-average, --profile-set cpu-peak, --profile-set cpu-compare, --profile-set memory-detail, --profile-set software-deep, --profile-set hardware-deep, or host-context --proc-stat-comparison.");
    println!("Supported network policies: public, no-network, and exact-host allowlist.");
    println!("Native scheduler tracing supports BCC (default) and perf backends.");
    println!("Native Enroot requires --runtime enroot --strace-mode none and is wall-time-only.");
}

fn cli_error(program: &str, message: impl fmt::Display) -> i32 {
    usage(program, &mut io::stderr());
    eprintln!("{program}: error: {message}");
    2
}

fn count(report: &Value, field: &str) -> u64 {
    report["replay_summary"][field].as_u64().unwrap_or(0)
}

fn print_cpu(cpu: &Value) {
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
        println!("Average logical cores used: n/a (task-scoped cgroup cpu.stat unavailable)");
    }
    if let (Some(percent), Some(capacity)) = (
        cpu["utilization_percent_of_allocation"].as_f64(),
        cpu["allocated_logical_cpus"].as_f64(),
    ) {
        println!("Allocated CPU utilization: {percent:.3}% of {capacity} logical CPUs");
    }
}

/// Native CLI entry point used by the public dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let config = match parse_arguments(arguments) {
        Ok(Some(config)) => config,
        Ok(None) => {
            help(public_program);
            return 0;
        }
        Err(error) => return cli_error(public_program, error),
    };
    let guard = match profile_container::ProfileSignalGuard::install() {
        Ok(guard) => guard,
        Err(error) => return cli_error(public_program, error),
    };
    let result = run_transcript_profile(&config);
    drop(guard);
    let report = match result {
        Ok(report) => report,
        Err(error) if profile_container::process_signal_cancelled() => {
            eprintln!("{public_program}: interrupted: {error}");
            return 130;
        }
        Err(error) => return cli_error(public_program, error),
    };
    println!("Task: {}", report["task_id"].as_str().unwrap_or(""));
    println!("Trial: {}", report["trial_name"].as_str().unwrap_or(""));
    println!(
        "Actions: {}/{} replayed, {} ok, {} failed, {} unsupported",
        count(&report, "actions_replayed"),
        count(&report, "actions_extracted"),
        count(&report, "actions_ok"),
        count(&report, "actions_failed"),
        count(&report, "actions_unsupported"),
    );
    print_cpu(&report["container_cpu"]);
    if let Some(peak) = report
        .pointer("/scheduler_concurrency/peak_simultaneous_logical_cores")
        .filter(|value| !value.is_null())
    {
        println!("Peak simultaneous logical CPUs: {peak}");
    }
    if let Some(peak) = report.pointer("/proc_stat_comparison/measurement/peak_active_logical_cpus")
    {
        println!("Sampled /proc active-CPU peak (comparison only): {}", peak);
    }
    println!(
        "Report: {}",
        config.output_dir.join("pipeline_report.json").display()
    );
    let invalid_memory = report
        .get("container_memory")
        .filter(|value| !value.is_null())
        .is_some_and(|memory| {
            memory.get("valid_for_observed_memory_reporting") != Some(&Value::Bool(true))
        });
    let invalid_scheduler = report
        .get("scheduler_concurrency")
        .filter(|value| !value.is_null())
        .is_some_and(|scheduler| {
            scheduler.pointer("/cgroup_validation/valid_for_concurrency_reporting")
                != Some(&Value::Bool(true))
        });
    (report.get("status").and_then(Value::as_str) != Some("valid")
        || invalid_memory
        || invalid_scheduler) as i32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arguments(values: &[&str]) -> Vec<OsString> {
        values.iter().map(OsString::from).collect()
    }

    fn base_config(root: &Path) -> TranscriptProfileConfig {
        TranscriptProfileConfig {
            runtime: "docker".to_owned(),
            transcript: root.join("trace.jsonl"),
            task_id: "task".to_owned(),
            trial_name: "task.1".to_owned(),
            image: "image".to_owned(),
            output_dir: root.join("output"),
            cpuset_cpus: Some("0".to_owned()),
            allocated_cpus: None,
            source_workspace: Some(PathBuf::from("/captured/agent_workspace")),
            initial_workspace: None,
            pinchbench_skill_dir: None,
            timeout_sec: 60.0,
            prepare_command: None,
            verification_command: None,
            strace_mode: "none".to_owned(),
            string_limit: 4096,
            shell: "/bin/bash".to_owned(),
            workdir: "/workspace".to_owned(),
            memory_profile: false,
            memory_sample_interval_sec: 0.1,
            memory_limit_bytes: None,
            memory_swap_total_limit_bytes: None,
            hardware_perf_binary: "perf".to_owned(),
            hardware_perf_events: crate::hardware_perf::PORTABLE_HARDWARE_EVENTS
                .iter()
                .map(|event| (*event).to_owned())
                .collect(),
            hardware_perf_attach_timeout: Duration::from_secs(20),
            hardware_perf_minimum_running_ratio: 0.95,
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
            network: None,
            network_policy: NetworkPolicy::default(),
            network_policy_proxy_image: "python:3.12-slim".to_owned(),
            network_policy_external_network: "bridge".to_owned(),
            docker: "docker".to_owned(),
            enroot: "enroot".to_owned(),
            allow_unsupported: false,
            profile_sets: vec!["cpu-average".to_owned()],
            coordinated_baseline: false,
            attach_timeout: Duration::from_secs(20),
        }
    }

    #[test]
    fn parser_accepts_the_native_cpu_average_slice() {
        let parsed = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--profile-set",
            "cpu-average",
            "--network-policy",
            "no-network",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.task_id, "task");
        assert_eq!(parsed.cpuset_cpus.as_deref(), Some("0-3"));
        assert_eq!(parsed.profile_sets, ["cpu-average"]);
        assert_eq!(
            parsed.network_policy.effective_mode(),
            NetworkMode::NoNetwork
        );
    }

    #[test]
    fn parser_accepts_native_proc_stat_context_with_custom_sampling() {
        let values = arguments(&[
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--proc-stat-comparison",
            "--proc-stat-interval-sec",
            "0.05",
            "--proc-stat-baseline-sec",
            "0.15",
        ]);
        let parsed = parse_arguments(&values).unwrap().unwrap();
        assert!(parsed.proc_stat_comparison);
        assert_eq!(parsed.proc_stat_interval_sec, 0.05);
        assert_eq!(parsed.proc_stat_baseline_sec, 0.15);
    }

    #[test]
    fn parser_and_routing_accept_memory_flag_and_memory_detail() {
        let values = arguments(&[
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--memory-profile",
            "--memory-sample-interval-sec",
            "0.02",
            "--memory-limit-bytes",
            "134217728",
            "--memory-swap-total-limit-bytes",
            "134217728",
        ]);
        let parsed = parse_arguments(&values).unwrap().unwrap();
        assert!(parsed.memory_profile);
        assert_eq!(parsed.strace_mode, "execve");
        assert!(parsed.profile_sets.is_empty());
        assert_eq!(parsed.memory_sample_interval_sec, 0.02);
        assert_eq!(parsed.memory_limit_bytes, Some(134_217_728));
        assert_eq!(parsed.memory_swap_total_limit_bytes, Some(134_217_728));

        let detail = arguments(&[
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--profile-set",
            "memory-detail",
        ]);
        let parsed = parse_arguments(&detail).unwrap().unwrap();
        assert_eq!(parsed.profile_sets, ["memory-detail"]);
    }

    #[test]
    fn parser_and_routing_accept_native_perf_and_bcc_scheduler_modes() {
        let base = [
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--scheduler-backend",
            "perf",
            "--scheduler-perf-binary",
            "/usr/bin/perf",
            "--scheduler-perf-mmap-pages",
            "64",
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
        assert!(explicit.scheduler_concurrency);
        assert!(explicit.profile_sets.is_empty());
        assert_eq!(explicit.strace_mode, "execve");
        assert_eq!(explicit.scheduler_backend, "perf");
        assert_eq!(explicit.scheduler_perf_binary, "/usr/bin/perf");
        assert_eq!(explicit.scheduler_perf_mmap_pages, Some(64));
        assert_eq!(explicit.scheduler_cpu_tolerance_fraction, 0.03);
        assert_eq!(explicit.scheduler_cpu_tolerance_sec, 0.007);

        for profile_set in ["cpu-peak", "cpu-compare"] {
            let named = base
                .iter()
                .copied()
                .chain(["--profile-set", profile_set])
                .map(OsString::from)
                .collect::<Vec<_>>();
            let named = parse_arguments(&named).unwrap().unwrap();
            assert!(!named.scheduler_concurrency);
            assert_eq!(named.profile_sets, [profile_set]);
        }

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

        let bcc = arguments(&[
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--profile-set",
            "cpu-peak",
            "--scheduler-backend",
            "bcc",
            "--scheduler-bcc-buffer-pages",
            "64",
        ]);
        let bcc = parse_arguments(&bcc).unwrap().unwrap();
        assert_eq!(bcc.scheduler_backend, "bcc");
        assert_eq!(bcc.scheduler_bcc_buffer_pages, Some(64));
    }

    #[test]
    fn proc_stat_requires_enforced_cpuset() {
        let error = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "2",
            "--proc-stat-comparison",
        ]))
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("--proc-stat-comparison requires --cpuset-cpus"));
    }

    #[test]
    fn replay_command_uses_native_controller_and_stable_layout() {
        let temporary = tempfile::tempdir().unwrap();
        let config = base_config(temporary.path());
        let command = replay_command(&config, "trace file.jsonl", true);
        assert!(command.starts_with("/tasktrace-controller/tasktrace replay transcript"));
        assert!(command.contains("'/tasktrace-input/trace file.jsonl'"));
        assert!(command.contains("--workspace /tasktrace/workspace"));
        assert!(command.contains("--output-dir /tasktrace/output"));
        assert!(command.contains("--initial-workspace /tasktrace-initial-workspace"));
        assert!(!command.contains("python"));
    }

    #[test]
    fn explicit_process_trace_is_uncoordinated_and_software_deep_is_coordinated() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = base_config(temporary.path());
        config.profile_sets.clear();
        config.strace_mode = "process".to_owned();
        fs::write(
            &config.transcript,
            "{\"task_id\":\"task\",\"trial_name\":\"task.1\",\"answer\":\"[Tool Call: exec] {\\\"command\\\":\\\"true\\\"}\"}\n",
        )
        .unwrap();

        let prepared = prepare_profile(&config).unwrap();
        assert!(prepared.profile.profile_sets.is_empty());
        assert!(!prepared.profile.coordinated_baseline);

        let software_root = temporary.path().join("software");
        fs::create_dir(&software_root).unwrap();
        let mut config = base_config(&software_root);
        config.profile_sets = vec!["software-deep".to_owned()];
        fs::write(
            &config.transcript,
            "{\"task_id\":\"task\",\"trial_name\":\"task.1\",\"answer\":\"[Tool Call: exec] {\\\"command\\\":\\\"true\\\"}\"}\n",
        )
        .unwrap();
        let prepared = prepare_profile(&config).unwrap();
        assert_eq!(prepared.profile.profile_sets, ["software-deep"]);
        assert!(!prepared.profile.coordinated_baseline);
    }

    #[test]
    fn workspace_provenance_changes_with_content_and_symlink() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::write(workspace.join("input.txt"), "first").unwrap();
        std::os::unix::fs::symlink("input.txt", workspace.join("alias")).unwrap();
        let before = path_tree_sha256(&workspace).unwrap();
        fs::write(workspace.join("input.txt"), "second").unwrap();
        assert_ne!(path_tree_sha256(&workspace).unwrap(), before);
    }

    #[test]
    fn unsupported_tools_fail_before_output_creation() {
        let temporary = tempfile::tempdir().unwrap();
        let mut config = base_config(temporary.path());
        fs::write(
            &config.transcript,
            "{\"task_id\":\"task\",\"trial_name\":\"task.1\",\"answer\":\"[Tool Call: browser] {\\\"url\\\":\\\"https://example.com\\\"}\"}\n",
        )
        .unwrap();
        let error = match prepare_profile(&config) {
            Ok(_) => panic!("unsupported transcript unexpectedly prepared"),
            Err(error) => error,
        };
        assert!(error.to_string().contains("unsupported tools"));
        assert!(!config.output_dir.exists());
        config.allow_unsupported = true;
        config.proc_stat_comparison = true;
        config.proc_stat_interval_sec = 0.05;
        config.proc_stat_baseline_sec = 0.15;
        config.memory_profile = true;
        config.memory_sample_interval_sec = 0.02;
        config.memory_limit_bytes = Some(134_217_728);
        config.memory_swap_total_limit_bytes = Some(134_217_728);
        config.profile_sets = vec!["cpu-compare".to_owned()];
        config.scheduler_backend = "perf".to_owned();
        config.scheduler_perf_binary = "/usr/bin/perf".to_owned();
        config.scheduler_perf_mmap_pages = Some(128);
        config.scheduler_cpu_tolerance_fraction = 0.03;
        config.scheduler_cpu_tolerance_sec = 0.007;
        let prepared = prepare_profile(&config).unwrap();
        assert_eq!(prepared.unsupported_action_count, 1);
        assert!(prepared.profile.proc_stat_comparison);
        assert_eq!(prepared.profile.proc_stat_interval_sec, 0.05);
        assert_eq!(prepared.profile.proc_stat_baseline_sec, 0.15);
        assert!(prepared.profile.memory_profile);
        assert_eq!(prepared.profile.memory_sample_interval_sec, 0.02);
        assert_eq!(prepared.profile.memory_limit_bytes, Some(134_217_728));
        assert_eq!(
            prepared.profile.memory_swap_total_limit_bytes,
            Some(134_217_728)
        );
        assert_eq!(prepared.profile.profile_sets, ["cpu-compare"]);
        assert_eq!(prepared.profile.scheduler_backend, "perf");
        assert_eq!(prepared.profile.scheduler_perf_binary, "/usr/bin/perf");
        assert_eq!(prepared.profile.scheduler_perf_mmap_pages, Some(128));
        assert_eq!(prepared.profile.scheduler_cpu_tolerance_fraction, 0.03);
        assert_eq!(prepared.profile.scheduler_cpu_tolerance_sec, 0.007);
    }

    #[test]
    fn pinchbench_staging_restores_sourced_and_inline_files() {
        let temporary = tempfile::tempdir().unwrap();
        let skill = temporary.path().join("skill");
        fs::create_dir_all(skill.join("tasks")).unwrap();
        fs::create_dir_all(skill.join("assets/logs")).unwrap();
        fs::write(skill.join("assets/logs/input.log"), "entry\n").unwrap();
        fs::write(
            skill.join("tasks/task_fixture.md"),
            "---\nid: task_fixture\nworkspace_files:\n  - source: logs/input.log\n    dest: input.log\n  - path: config/settings.json\n    content: '{}'\n---\nPrompt\n",
        )
        .unwrap();
        let destination = temporary.path().join("seed");
        let (task_file, count) =
            stage_pinchbench_initial_workspace(&skill, "category__task_fixture", &destination)
                .unwrap();
        assert_eq!(
            task_file,
            skill.join("tasks/task_fixture.md").canonicalize().unwrap()
        );
        assert_eq!(count, 2);
        assert_eq!(
            fs::read_to_string(destination.join("input.log")).unwrap(),
            "entry\n"
        );
        assert_eq!(
            fs::read_to_string(destination.join("config/settings.json")).unwrap(),
            "{}"
        );
    }
}
