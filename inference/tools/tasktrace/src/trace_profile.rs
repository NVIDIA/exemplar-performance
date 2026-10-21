// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native profiling adapter for finalized shell-command trace streams (`tasktrace-cpu-trace.v1` and its earlier ids).
//!
//! Format interpretation stays here; disposable Docker lifecycle and cgroup
//! accounting stay in `profile_container`.  The replay controller is the
//! current native executable mounted read-only into the measured sandbox.

use crate::network_policy::{resolve_cli_network_policy, NetworkPolicy};
use crate::profile_container::{self, ProfileConfig};
use crate::trace_replay::{self, TaskTraceTrace};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;

const TRACE_MOUNT: &str = "/tasktrace-input";
const REPLAY_MOUNT: &str = "/tasktrace";
const CONTROLLER_MOUNT: &str = "/tasktrace-controller";
const CONTROLLER_EXECUTABLE: &str = "/tasktrace-controller/tasktrace";
/// Where the enroot runtime mounts <output-dir>/container_profile (read-write) in the task container.
const LOOP_SYNC_PROFILE_DIR: &str = "/tasktrace-profile-output";
const PIPELINE_SCHEMA_VERSION: &str = "container-trace-profile.v1";

#[derive(Debug)]
struct TraceProfileError(String);

impl TraceProfileError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TraceProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for TraceProfileError {}

impl From<io::Error> for TraceProfileError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for TraceProfileError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug)]
struct TraceProfileConfig {
    runtime: String,
    trace: PathBuf,
    /// Disposable image for the docker/enroot runtimes; `None` for `--runtime current`.
    image: Option<String>,
    output_dir: PathBuf,
    cpu_limit: Option<f64>,
    cpuset_cpus: Option<String>,
    allocated_cpus: Option<f64>,
    pacing: String,
    timeout_sec: Option<f64>,
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
    profile_sets: Vec<String>,
    coordinated_baseline: bool,
    attach_timeout: Duration,
    current_container_exclusive: bool,
    /// Replay in the recorded agent loop's order alongside an LLM-side replay (enroot, unpaced).
    loop_order: bool,
}

#[derive(Default)]
struct ParsedArguments {
    trace: Option<PathBuf>,
    image: Option<String>,
    runtime: String,
    output_dir: Option<PathBuf>,
    cpu_limit: Option<f64>,
    cpuset_cpus: Option<String>,
    allocated_cpus: Option<f64>,
    pacing: String,
    timeout_sec: Option<f64>,
    prepare_command: Option<String>,
    verification_command: Option<String>,
    strace_mode: String,
    string_limit: i64,
    profile_sets: Vec<String>,
    coordinated_baseline: bool,
    scheduler_concurrency: bool,
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
    hardware_perf_binary: String,
    hardware_perf_events: Vec<String>,
    hardware_perf_attach_timeout_sec: f64,
    hardware_perf_minimum_running_ratio: f64,
    scheduler_attach_timeout_sec: f64,
    scheduler_cpu_tolerance_fraction: f64,
    scheduler_cpu_tolerance_sec: f64,
    shell: String,
    workdir: String,
    network: Option<String>,
    network_mode: Option<String>,
    allowed_hosts: Vec<String>,
    network_proxy_image: String,
    network_external: String,
    docker: String,
    enroot: String,
    current_container_exclusive: bool,
    loop_order: bool,
    // Presence, not value: `--runtime current --image ""` is still an explicit request to reject.
    image_was_explicit: bool,
    scheduler_backend_was_explicit: bool,
    docker_was_explicit: bool,
    enroot_was_explicit: bool,
}

impl ParsedArguments {
    fn defaults() -> Self {
        Self {
            runtime: "docker".to_owned(),
            pacing: "recorded".to_owned(),
            strace_mode: "none".to_owned(),
            string_limit: 4096,
            scheduler_backend: "bcc".to_owned(),
            scheduler_perf_binary: "perf".to_owned(),
            proc_stat_interval_sec: 0.5,
            proc_stat_baseline_sec: 5.0,
            memory_sample_interval_sec: 0.1,
            hardware_perf_binary: "perf".to_owned(),
            hardware_perf_attach_timeout_sec: 20.0,
            hardware_perf_minimum_running_ratio: crate::hardware_perf::MINIMUM_RUNNING_RATIO,
            scheduler_attach_timeout_sec: 20.0,
            scheduler_cpu_tolerance_fraction: 0.02,
            scheduler_cpu_tolerance_sec: 0.005,
            shell: "/bin/bash".to_owned(),
            workdir: "/workdir".to_owned(),
            network_proxy_image: "python:3.12-slim".to_owned(),
            network_external: "bridge".to_owned(),
            docker: "docker".to_owned(),
            enroot: "enroot".to_owned(),
            ..Self::default()
        }
    }
}

fn text_argument(value: &OsStr, option: &str) -> Result<String, TraceProfileError> {
    value
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| TraceProfileError::new(format!("{option} must be valid UTF-8")))
}

fn split_long_option(value: &str) -> (&str, Option<&str>) {
    value
        .split_once('=')
        .map_or((value, None), |(option, argument)| (option, Some(argument)))
}

fn parse_number<T>(value: &str, option: &str) -> Result<T, TraceProfileError>
where
    T: std::str::FromStr,
{
    value
        .parse()
        .map_err(|_| TraceProfileError::new(format!("invalid value for {option}: {value:?}")))
}

fn parse_arguments(
    arguments: &[OsString],
) -> Result<Option<TraceProfileConfig>, TraceProfileError> {
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
        let flag = |attached: Option<&str>| -> Result<(), TraceProfileError> {
            if attached.is_some() {
                Err(TraceProfileError::new(format!(
                    "{option} does not take a value"
                )))
            } else {
                Ok(())
            }
        };
        let mut take = || -> Result<String, TraceProfileError> {
            if let Some(value) = attached {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .ok_or_else(|| {
                    TraceProfileError::new(format!("argument {option}: expected one argument"))
                })
                .and_then(|value| text_argument(value, option))
        };
        match option {
            "--image" => {
                parsed.image = Some(take()?);
                parsed.image_was_explicit = true;
            }
            "--runtime" => parsed.runtime = take()?,
            "--output-dir" => parsed.output_dir = Some(PathBuf::from(take()?)),
            "--cpu-limit" => {
                let value = take()?;
                parsed.cpu_limit = Some(parse_number(&value, option)?);
            }
            "--cpuset-cpus" => parsed.cpuset_cpus = Some(take()?),
            "--allocated-cpus" => {
                let value = take()?;
                parsed.allocated_cpus = Some(parse_number(&value, option)?);
            }
            "--pacing" => parsed.pacing = take()?,
            "--timeout-sec" => {
                let value = take()?;
                parsed.timeout_sec = Some(parse_number(&value, option)?);
            }
            "--prepare-command" => parsed.prepare_command = Some(take()?),
            "--verification-command" => parsed.verification_command = Some(take()?),
            "--strace-mode" => parsed.strace_mode = take()?,
            "--strace-string-limit" => {
                let value = take()?;
                parsed.string_limit = parse_number(&value, option)?;
            }
            "--profile-set" => parsed.profile_sets.push(take()?),
            "--coordinated-baseline" => {
                flag(attached)?;
                parsed.coordinated_baseline = true;
            }
            "--current-container-exclusive" => {
                flag(attached)?;
                parsed.current_container_exclusive = true;
            }
            "--loop-order" => {
                flag(attached)?;
                parsed.loop_order = true;
            }
            "--scheduler-concurrency" => {
                flag(attached)?;
                parsed.scheduler_concurrency = true;
            }
            "--scheduler-backend" => {
                parsed.scheduler_backend = take()?;
                parsed.scheduler_backend_was_explicit = true;
            }
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
            "--shell" => parsed.shell = take()?,
            "--workdir" => parsed.workdir = take()?,
            "--network" => parsed.network = Some(take()?),
            "--network-policy" => parsed.network_mode = Some(take()?),
            "--allowed-host" => parsed.allowed_hosts.push(take()?),
            "--network-policy-proxy-image" => parsed.network_proxy_image = take()?,
            "--network-policy-external-network" => parsed.network_external = take()?,
            "--docker" => {
                parsed.docker = take()?;
                parsed.docker_was_explicit = true;
            }
            "--enroot" => {
                parsed.enroot = take()?;
                parsed.enroot_was_explicit = true;
            }
            "--" => {
                index += 1;
                while index < arguments.len() {
                    if parsed.trace.is_some() {
                        return Err(TraceProfileError::new(format!(
                            "unrecognized arguments: {}",
                            arguments[index].to_string_lossy()
                        )));
                    }
                    parsed.trace = Some(PathBuf::from(&arguments[index]));
                    index += 1;
                }
                break;
            }
            value if value.starts_with('-') => {
                return Err(TraceProfileError::new(format!(
                    "unrecognized arguments: {raw}"
                )))
            }
            _ if parsed.trace.is_none() => parsed.trace = Some(PathBuf::from(&arguments[index])),
            _ => {
                return Err(TraceProfileError::new(format!(
                    "unrecognized arguments: {raw}"
                )))
            }
        }
        index += 1;
    }
    validate_arguments(parsed).map(Some)
}

fn positive_finite(value: f64, option: &str) -> Result<f64, TraceProfileError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(TraceProfileError::new(format!("{option} must be positive")))
    }
}

fn positive_bytes(value: i64, option: &str) -> Result<u64, TraceProfileError> {
    u64::try_from(value)
        .ok()
        .filter(|value| *value > 0)
        .ok_or_else(|| TraceProfileError::new(format!("{option} must be positive")))
}

fn validate_arguments(
    mut parsed: ParsedArguments,
) -> Result<TraceProfileConfig, TraceProfileError> {
    let trace = parsed
        .trace
        .ok_or_else(|| TraceProfileError::new("the following arguments are required: trace"))?;
    let image = parsed.image.filter(|value| !value.is_empty());
    let output_dir = parsed.output_dir.ok_or_else(|| {
        TraceProfileError::new("the following arguments are required: --output-dir")
    })?;
    if !matches!(parsed.runtime.as_str(), "docker" | "enroot" | "current") {
        return Err(TraceProfileError::new(format!(
            "invalid choice for --runtime: {:?} (choose from 'docker', 'enroot', 'current')",
            parsed.runtime
        )));
    }
    if parsed.runtime != "current" && image.is_none() {
        return Err(TraceProfileError::new(
            "the following arguments are required: --image",
        ));
    }
    if parsed.runtime == "current" && (image.is_some() || parsed.image_was_explicit) {
        return Err(TraceProfileError::new(
            "--runtime current does not accept --image",
        ));
    }
    parsed.pacing = crate::pacing::canonical(&parsed.pacing).to_owned();
    if !matches!(parsed.pacing.as_str(), "recorded" | "unpaced") {
        return Err(TraceProfileError::new(format!(
            "invalid choice for --pacing: {:?} (choose from 'recorded', 'unpaced')",
            parsed.pacing
        )));
    }
    let timeout_sec = parsed
        .timeout_sec
        .map(|value| positive_finite(value, "--timeout-sec"))
        .transpose()?;
    if parsed.loop_order && !matches!(parsed.runtime.as_str(), "enroot" | "current") {
        return Err(TraceProfileError::new(
            "--loop-order needs --runtime enroot or current: the LLM-side replay reads its hand-off files under --output-dir",
        ));
    }
    if parsed.loop_order && parsed.pacing != "unpaced" {
        return Err(TraceProfileError::new(
            "--loop-order keeps the recorded order, not the schedule; use --pacing unpaced",
        ));
    }
    let cpu_limit = parsed
        .cpu_limit
        .map(|value| positive_finite(value, "--cpu-limit"))
        .transpose()?;
    let allocated_cpus = parsed
        .allocated_cpus
        .map(|value| positive_finite(value, "--allocated-cpus"))
        .transpose()?;
    if parsed.runtime != "current" && parsed.cpuset_cpus.is_none() && allocated_cpus.is_none() {
        return Err(TraceProfileError::new(
            "provide --cpuset-cpus or --allocated-cpus for utilization reporting",
        ));
    }
    if let Some(cpuset) = &parsed.cpuset_cpus {
        profile_container::parse_cpuset(cpuset)
            .map_err(|error| TraceProfileError::new(error.to_string()))?;
    }
    if parsed.string_limit <= 0 {
        return Err(TraceProfileError::new(
            "--strace-string-limit must be positive",
        ));
    }
    if !matches!(parsed.strace_mode.as_str(), "execve" | "process" | "none") {
        return Err(TraceProfileError::new(format!(
            "invalid choice for --strace-mode: {:?} (choose from 'execve', 'process', 'none')",
            parsed.strace_mode
        )));
    }
    if parsed.coordinated_baseline && !parsed.profile_sets.is_empty() {
        return Err(TraceProfileError::new(
            "--coordinated-baseline is mutually exclusive with --profile-set",
        ));
    }
    let mut observed = std::collections::BTreeSet::new();
    if parsed
        .profile_sets
        .iter()
        .any(|value| !observed.insert(value.clone()))
    {
        return Err(TraceProfileError::new(
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
        return Err(TraceProfileError::new(format!(
            "native profile trace does not yet support --profile-set {unsupported}; cpu-average, cpu-peak, cpu-compare, memory-detail, software-deep, and hardware-deep are available"
        )));
    }
    let scheduler_requested = parsed.scheduler_concurrency
        || parsed
            .profile_sets
            .iter()
            .any(|value| matches!(value.as_str(), "cpu-peak" | "cpu-compare"));
    if !matches!(parsed.scheduler_backend.as_str(), "bcc" | "perf") {
        return Err(TraceProfileError::new(
            "invalid choice for --scheduler-backend (choose from 'bcc', 'perf')",
        ));
    }
    if !scheduler_requested && parsed.scheduler_backend != "bcc" {
        return Err(TraceProfileError::new(
            "--scheduler-backend requires --scheduler-concurrency",
        ));
    }
    if let Some(value) = parsed.scheduler_bcc_buffer_pages {
        if value <= 0 || (value & (value - 1)) != 0 {
            return Err(TraceProfileError::new(
                "--scheduler-bcc-buffer-pages must be a positive power of two",
            ));
        }
    }
    if !scheduler_requested && parsed.scheduler_bcc_buffer_pages.is_some() {
        return Err(TraceProfileError::new(
            "--scheduler-bcc-buffer-pages requires --scheduler-concurrency or a scheduler profile set",
        ));
    }
    if scheduler_requested
        && parsed.scheduler_backend == "perf"
        && parsed.scheduler_perf_binary.trim().is_empty()
    {
        return Err(TraceProfileError::new(
            "--scheduler-perf-binary must be nonempty",
        ));
    }
    if parsed
        .scheduler_perf_mmap_pages
        .is_some_and(|value| value <= 0)
    {
        return Err(TraceProfileError::new(
            "--scheduler-perf-mmap-pages must be positive",
        ));
    }
    for (value, option) in [
        (parsed.proc_stat_interval_sec, "--proc-stat-interval-sec"),
        (parsed.proc_stat_baseline_sec, "--proc-stat-baseline-sec"),
        (
            parsed.memory_sample_interval_sec,
            "--memory-sample-interval-sec",
        ),
        (
            parsed.scheduler_attach_timeout_sec,
            "--scheduler-attach-timeout-sec",
        ),
        (
            parsed.hardware_perf_attach_timeout_sec,
            "--hardware-perf-attach-timeout-sec",
        ),
    ] {
        positive_finite(value, option)?;
    }
    if parsed.hardware_perf_binary.trim().is_empty() {
        return Err(TraceProfileError::new(
            "--hardware-perf-binary must be nonempty",
        ));
    }
    if !parsed.hardware_perf_minimum_running_ratio.is_finite()
        || parsed.hardware_perf_minimum_running_ratio <= 0.0
        || parsed.hardware_perf_minimum_running_ratio > 1.0
    {
        return Err(TraceProfileError::new(
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
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            != parsed.hardware_perf_events.len()
    {
        return Err(TraceProfileError::new(
            "hardware perf events must be nonempty and unique",
        ));
    }
    if parsed.proc_stat_comparison && parsed.cpuset_cpus.is_none() {
        return Err(TraceProfileError::new(
            "--proc-stat-comparison requires --cpuset-cpus",
        ));
    }
    if parsed.scheduler_cpu_tolerance_fraction < 0.0
        || !parsed.scheduler_cpu_tolerance_fraction.is_finite()
        || parsed.scheduler_cpu_tolerance_sec < 0.0
        || !parsed.scheduler_cpu_tolerance_sec.is_finite()
    {
        return Err(TraceProfileError::new(
            "scheduler CPU tolerances must not be negative",
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
        return Err(TraceProfileError::new(
            "--memory-swap-total-limit-bytes requires --memory-limit-bytes",
        ));
    }
    if memory_swap_total_limit_bytes
        .zip(memory_limit_bytes)
        .is_some_and(|(swap, memory)| swap < memory)
    {
        return Err(TraceProfileError::new(
            "--memory-swap-total-limit-bytes must be at least --memory-limit-bytes",
        ));
    }
    if parsed
        .verification_command
        .as_ref()
        .is_some_and(|value| value.trim().is_empty())
    {
        return Err(TraceProfileError::new(
            "verification command must be a nonempty string",
        ));
    }
    if parsed.verification_command.is_some()
        && !parsed.coordinated_baseline
        && parsed.profile_sets.is_empty()
        && !parsed.proc_stat_comparison
    {
        return Err(TraceProfileError::new(
            "--verification-command requires --profile-set or --coordinated-baseline",
        ));
    }
    let network_policy = resolve_cli_network_policy(
        parsed.network_mode.as_deref(),
        &parsed.allowed_hosts,
        parsed.network.as_deref(),
    )
    .map_err(|error| TraceProfileError::new(error.to_string()))?
    .unwrap_or_default();
    if parsed.runtime == "current" {
        // TaskTrace cannot create, limit, pin or isolate this container: reject those controls.
        if parsed.cpuset_cpus.is_some() {
            return Err(TraceProfileError::new(
                "--runtime current cannot enforce --cpuset-cpus; configure the sandbox allocation",
            ));
        }
        if cpu_limit.is_some() {
            return Err(TraceProfileError::new(
                "--runtime current cannot enforce --cpu-limit; configure the sandbox allocation",
            ));
        }
        let scheduler_options = parsed.scheduler_concurrency
            || parsed.scheduler_backend_was_explicit
            || parsed.scheduler_backend != "bcc"
            || parsed.scheduler_bcc_buffer_pages.is_some()
            || parsed.scheduler_perf_binary != "perf"
            || parsed.scheduler_perf_mmap_pages.is_some()
            || parsed.scheduler_attach_timeout_sec != 20.0
            || parsed.scheduler_cpu_tolerance_fraction != 0.02
            || parsed.scheduler_cpu_tolerance_sec != 0.005;
        if scheduler_options {
            return Err(TraceProfileError::new(
                "scheduler profiling options are not supported by current-container runtime",
            ));
        }
        if parsed.proc_stat_comparison
            || parsed.proc_stat_interval_sec != 0.5
            || parsed.proc_stat_baseline_sec != 5.0
        {
            return Err(TraceProfileError::new(
                "proc-stat sampling options are not supported by current-container runtime",
            ));
        }
        if parsed.memory_profile
            || parsed.memory_sample_interval_sec != 0.1
            || memory_limit_bytes.is_some()
            || memory_swap_total_limit_bytes.is_some()
        {
            return Err(TraceProfileError::new(
                "memory profiling options are not supported by current-container runtime",
            ));
        }
        if !parsed.profile_sets.is_empty() {
            return Err(TraceProfileError::new(
                "--profile-set is not supported by current-container runtime",
            ));
        }
        if parsed.coordinated_baseline {
            return Err(TraceProfileError::new(
                "--coordinated-baseline is not supported by current-container runtime",
            ));
        }
        if parsed.verification_command.is_some() {
            return Err(TraceProfileError::new(
                "--verification-command is not supported by current-container runtime",
            ));
        }
        if parsed.hardware_perf_binary != "perf"
            || !parsed.hardware_perf_events.is_empty()
            || parsed.hardware_perf_attach_timeout_sec != 20.0
            || parsed.hardware_perf_minimum_running_ratio
                != crate::hardware_perf::MINIMUM_RUNNING_RATIO
        {
            return Err(TraceProfileError::new(
                "hardware profiling options are not supported by current-container runtime",
            ));
        }
        if parsed.network_mode.is_some()
            || !parsed.allowed_hosts.is_empty()
            || parsed.network.is_some()
            || parsed.network_proxy_image != "python:3.12-slim"
            || parsed.network_external != "bridge"
        {
            return Err(TraceProfileError::new(
                "network policy controls are not supported by current-container runtime",
            ));
        }
        if parsed.docker_was_explicit
            || parsed.enroot_was_explicit
            || parsed.docker != "docker"
            || parsed.enroot != "enroot"
        {
            return Err(TraceProfileError::new(
                "runtime executable overrides are not supported by current-container runtime",
            ));
        }
    } else if parsed.current_container_exclusive {
        return Err(TraceProfileError::new(
            "--current-container-exclusive requires --runtime current",
        ));
    }
    if parsed.runtime == "enroot" {
        if parsed.strace_mode != "none" {
            return Err(TraceProfileError::new(
                "--strace-mode execve/process is currently Docker-only",
            ));
        }
        if cpu_limit.is_some() {
            return Err(TraceProfileError::new(
                "--cpu-limit is Docker-only; use Slurm/taskset CPU allocation for enroot",
            ));
        }
        if parsed.network.is_some()
            || network_policy.effective_mode() != crate::network_policy::NetworkMode::Public
        {
            return Err(TraceProfileError::new(
                "non-public network policy is currently Docker-only",
            ));
        }
        if parsed.scheduler_concurrency {
            return Err(TraceProfileError::new(
                "--scheduler-concurrency is currently Docker-only",
            ));
        }
        if parsed.proc_stat_comparison {
            return Err(TraceProfileError::new(
                "--proc-stat-comparison is currently Docker-only",
            ));
        }
        if parsed.memory_profile {
            return Err(TraceProfileError::new(
                "--memory-profile is currently Docker-only",
            ));
        }
        if memory_limit_bytes.is_some() {
            return Err(TraceProfileError::new(
                "--memory-limit-bytes is currently Docker-only",
            ));
        }
        if memory_swap_total_limit_bytes.is_some() {
            return Err(TraceProfileError::new(
                "--memory-swap-total-limit-bytes is currently Docker-only",
            ));
        }
        if !parsed.profile_sets.is_empty() {
            return Err(TraceProfileError::new(
                "--profile-set currently requires the Docker runtime",
            ));
        }
        if parsed.coordinated_baseline {
            return Err(TraceProfileError::new(
                "--coordinated-baseline currently requires the Docker runtime",
            ));
        }
        if parsed.verification_command.is_some() {
            return Err(TraceProfileError::new(
                "--verification-command currently requires Docker",
            ));
        }
    }
    Ok(TraceProfileConfig {
        runtime: parsed.runtime,
        trace,
        image,
        output_dir,
        cpu_limit,
        cpuset_cpus: parsed.cpuset_cpus,
        allocated_cpus,
        pacing: parsed.pacing,
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
        profile_sets: parsed.profile_sets,
        coordinated_baseline: parsed.coordinated_baseline,
        attach_timeout: Duration::from_secs_f64(parsed.scheduler_attach_timeout_sec),
        current_container_exclusive: parsed.current_container_exclusive,
        loop_order: parsed.loop_order,
    })
}

fn sha256_file(path: &Path) -> Result<String, TraceProfileError> {
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

fn mount(source: &Path, target: &str, readonly: bool) -> Result<String, TraceProfileError> {
    let source = source.to_str().ok_or_else(|| {
        TraceProfileError::new(format!(
            "native Docker bind mount path must be valid UTF-8: {}",
            source.display()
        ))
    })?;
    if source.contains(',') {
        return Err(TraceProfileError::new(format!(
            "native Docker bind mount path must not contain a comma: {source}"
        )));
    }
    Ok(format!(
        "type=bind,src={source},dst={target}{}",
        if readonly { ",readonly" } else { "" }
    ))
}

fn prepare_pipeline_output(path: &Path) -> Result<PathBuf, TraceProfileError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() {
                return Err(TraceProfileError::new(format!(
                    "pipeline output path must not be a symbolic link: {}",
                    path.display()
                )));
            }
            if !metadata.is_dir() {
                return Err(TraceProfileError::new(format!(
                    "pipeline output path is not a directory: {}",
                    path.display()
                )));
            }
            if fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(TraceProfileError::new(format!(
                    "pipeline output directory must be new or empty: {}",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            fs::create_dir_all(path).map_err(|error| {
                TraceProfileError::new(format!(
                    "unable to create pipeline output directory {}: {error}",
                    path.display()
                ))
            })?;
        }
        Err(error) => return Err(error.into()),
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(TraceProfileError::new(format!(
            "pipeline output path changed after preflight: {}",
            path.display()
        )));
    }
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Err(TraceProfileError::new(format!(
            "pipeline output directory changed after preflight: {}",
            path.display()
        )));
    }
    let canonical = fs::canonicalize(path)?;
    fs::create_dir(canonical.join("replay"))?;
    Ok(canonical)
}

fn trace_header<'a>(trace: &'a TaskTraceTrace, field: &str) -> &'a Value {
    trace.header().get(field).unwrap_or(&Value::Null)
}

fn replay_command(config: &TraceProfileConfig, trace_name: &str) -> String {
    let mut arguments = vec![
        CONTROLLER_EXECUTABLE.to_owned(),
        "replay".to_owned(),
        "trace".to_owned(),
        format!("{TRACE_MOUNT}/{trace_name}"),
        "--output".to_owned(),
        format!("{REPLAY_MOUNT}/replay_report.json"),
        "--cwd".to_owned(),
        config.workdir.clone(),
        "--pacing".to_owned(),
        config.pacing.clone(),
    ];
    if let Some(timeout) = config.timeout_sec {
        arguments.extend(["--timeout-sec".to_owned(), timeout.to_string()]);
    }
    if config.loop_order {
        // the profile output directory, which the host sees as <output-dir>/container_profile
        arguments.extend([
            "--loop-sync-dir".to_owned(),
            format!("{LOOP_SYNC_PROFILE_DIR}/loop_order"),
        ]);
    }
    arguments
        .iter()
        .map(|argument| shell_quote(argument))
        .collect::<Vec<_>>()
        .join(" ")
}

/// The finalized trace plus the native replay controller (this executable), resolved and hashed.
struct TraceSource {
    trace: TaskTraceTrace,
    trace_path: PathBuf,
    trace_sha256: String,
    controller: PathBuf,
    controller_sha256: String,
}

fn load_trace_source(config: &TraceProfileConfig) -> Result<TraceSource, TraceProfileError> {
    let trace = trace_replay::read_trace(&config.trace)
        .map_err(|error| TraceProfileError::new(error.to_string()))?;
    let trace_path = config.trace.canonicalize().map_err(|error| {
        TraceProfileError::new(format!(
            "unable to canonicalize TaskTrace trace {}: {error}",
            config.trace.display()
        ))
    })?;
    let trace_sha256 = sha256_file(&trace_path)?;
    let controller = env::current_exe()?.canonicalize()?;
    if !controller.is_file() {
        return Err(TraceProfileError::new(format!(
            "native replay controller is not a regular file: {}",
            controller.display()
        )));
    }
    let controller_sha256 = sha256_file(&controller)?;
    Ok(TraceSource {
        trace,
        trace_path,
        trace_sha256,
        controller,
        controller_sha256,
    })
}

struct PreparedProfile {
    profile: ProfileConfig,
    source: TraceSource,
    output_dir: PathBuf,
}

fn prepare_profile(config: &TraceProfileConfig) -> Result<PreparedProfile, TraceProfileError> {
    let image = config.image.clone().ok_or_else(|| {
        TraceProfileError::new("--image is required for the docker and enroot runtimes")
    })?;
    let source = load_trace_source(config)?;
    let trace_name = source
        .trace_path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| TraceProfileError::new("TaskTrace trace filename must be valid UTF-8"))?;
    let trace_parent = source
        .trace_path
        .parent()
        .ok_or_else(|| TraceProfileError::new("TaskTrace trace has no parent directory"))?;
    let output_dir = prepare_pipeline_output(&config.output_dir)?;
    let replay_dir = output_dir.join("replay");

    let mut profile = ProfileConfig::new(
        image,
        replay_command(config, trace_name),
        output_dir.join("container_profile"),
    );
    profile.prepare_command = config.prepare_command.clone();
    profile.verification_command = config.verification_command.clone();
    profile.cpuset_cpus = config.cpuset_cpus.clone();
    profile.allocated_cpus = config.allocated_cpus;
    profile.cpu_limit = config.cpu_limit;
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
            .map_err(|error| TraceProfileError::new(error.to_string()))?;
    } else {
        profile.profile_sets = config.profile_sets.clone();
    }
    profile.mounts = vec![
        mount(trace_parent, TRACE_MOUNT, true)?,
        mount(&replay_dir, REPLAY_MOUNT, false)?,
        mount(&source.controller, CONTROLLER_EXECUTABLE, true)?,
    ];
    profile.workload_identity = Some(json!({
        "task_name": trace_header(&source.trace, "task_name"),
        "source_trial": trace_header(&source.trace, "source_trial"),
    }));
    profile.source_identity = Some(json!({
        "format": "trace",
        "schema_version": trace_header(&source.trace, "schema_version"),
        "sha256": source.trace_sha256,
        "action_count": source.trace.action_count(),
        "rust_controller_binary_sha256": source.controller_sha256,
    }));
    profile.expected_outcome = Some(json!({
        "kind": "source-recorded-action-outcomes",
        "action_count": source.trace.action_count(),
    }));
    Ok(PreparedProfile {
        profile,
        source,
        output_dir,
    })
}

/// `--runtime current`: this executable replays in place, on the container's own filesystem.
fn current_replay_command(
    config: &TraceProfileConfig,
    controller: &Path,
    trace_path: &Path,
    output_dir: &Path,
) -> String {
    let mut arguments = vec![
        controller.to_string_lossy().into_owned(),
        "replay".to_owned(),
        "trace".to_owned(),
        trace_path.to_string_lossy().into_owned(),
        "--output".to_owned(),
        output_dir
            .join("replay/replay_report.json")
            .to_string_lossy()
            .into_owned(),
        "--cwd".to_owned(),
        config.workdir.clone(),
        "--pacing".to_owned(),
        config.pacing.clone(),
    ];
    if let Some(timeout) = config.timeout_sec {
        arguments.extend(["--timeout-sec".to_owned(), timeout.to_string()]);
    }
    if config.loop_order {
        // in place, the same directory the enroot runtime mounts: <output-dir>/container_profile/loop_order
        arguments.extend([
            "--loop-sync-dir".to_owned(),
            output_dir
                .join("container_profile/loop_order")
                .to_string_lossy()
                .into_owned(),
        ]);
    }
    arguments
        .iter()
        .map(|argument| shell_quote(argument))
        .collect::<Vec<_>>()
        .join(" ")
}

fn current_profile_config(
    config: &TraceProfileConfig,
    source: &TraceSource,
    output_dir: &Path,
) -> Result<crate::current_profile::CurrentProfileConfig, TraceProfileError> {
    let invocation_cwd = env::current_dir()?;
    let controller_source = crate::current_profile::discover_controller_source(&source.controller);
    let task_environment = crate::current_profile::sanitized_controller_environment(
        &source.controller,
        controller_source.as_deref(),
        &invocation_cwd,
    )
    .map_err(|error| TraceProfileError::new(error.to_string()))?;
    Ok(crate::current_profile::CurrentProfileConfig {
        command: current_replay_command(config, &source.controller, &source.trace_path, output_dir),
        prepare_command: config.prepare_command.clone(),
        output_dir: output_dir.join("container_profile"),
        workdir: PathBuf::from(&config.workdir),
        shell: config.shell.clone(),
        environment: task_environment.assignments,
        unset_environment: task_environment.unset_names,
        strace_mode: config.strace_mode.clone(),
        string_limit: config.string_limit,
        allocated_cpus: config.allocated_cpus,
        exclusive: config.current_container_exclusive,
        adopt_orphans: true,
    })
}

fn report_fields(config: &TraceProfileConfig, source: &TraceSource) -> Value {
    let current = config.runtime == "current";
    let controller_identity = if current {
        json!({
            "kind": "rust-current-container-binary",
            "sha256": source.controller_sha256,
            "path": source.controller,
        })
    } else {
        json!({
            "kind": "rust-host-mounted-binary",
            "sha256": source.controller_sha256,
            "container_path": CONTROLLER_EXECUTABLE,
        })
    };
    json!({
        "schema_version": PIPELINE_SCHEMA_VERSION,
        "task_name": trace_header(&source.trace, "task_name"),
        "source_trial": trace_header(&source.trace, "source_trial"),
        "source_reward": trace_header(&source.trace, "reward"),
        "source_trace": source.trace_path.to_string_lossy(),
        "source_trace_sha256": source.trace_sha256,
        "source_trace_schema_version": trace_header(&source.trace, "schema_version"),
        "source_limitations": source.trace.header().get("limitations").cloned().unwrap_or_else(|| json!([])),
        "image": config.image,
        "runtime": config.runtime,
        "pacing": config.pacing,
        "loop_order": config.loop_order,
        "controller_runtime": if current { "image" } else { "rust-host-mount" },
        "controller_identity": controller_identity,
        "current_container_exclusive": if current { json!(config.current_container_exclusive) } else { Value::Null },
    })
}

fn run_current_trace_profile(config: &TraceProfileConfig) -> Result<Value, TraceProfileError> {
    let cancelled = || profile_container::process_signal_cancelled();
    if cancelled() {
        return Err(TraceProfileError::new(
            "current-container profile cancelled before output creation",
        ));
    }
    let source = load_trace_source(config)?;
    let output_dir = prepare_pipeline_output(&config.output_dir)?;
    let request = current_profile_config(config, &source, &output_dir)?;
    let mut container_report =
        crate::current_profile::run_current_profile_with_cancel(&request, Some(&cancelled))
            .map_err(|error| TraceProfileError::new(error.to_string()))?;
    if cancelled() {
        if let Some(validity) = container_report
            .get_mut("measurement_validity")
            .and_then(Value::as_object_mut)
        {
            validity.insert(
                "valid_for_average_cpu_reporting".to_owned(),
                Value::Bool(false),
            );
            validity.insert("cancelled".to_owned(), Value::Bool(true));
        }
        if let Some(report) = container_report.as_object_mut() {
            let cancellation = report
                .entry("cancellation".to_owned())
                .or_insert_with(|| json!({}));
            if let Some(cancellation) = cancellation.as_object_mut() {
                cancellation.insert("requested".to_owned(), Value::Bool(true));
            }
        }
    }
    let report = crate::profile_pipeline::finish_profiled_replay(
        &container_report,
        &output_dir,
        Path::new("replay/replay_report.json"),
        report_fields(config, &source),
        &[],
        false,
    )
    .map_err(|error| TraceProfileError::new(error.to_string()))?;
    if cancelled() {
        return Err(TraceProfileError::new(
            "profile interrupted during pipeline publication",
        ));
    }
    Ok(report)
}

fn run_trace_profile(config: &TraceProfileConfig) -> Result<Value, TraceProfileError> {
    if config.runtime == "current" {
        return run_current_trace_profile(config);
    }
    let prepared = prepare_profile(config)?;
    let fields = report_fields(config, &prepared.source);
    let run = profile_container::run_profile(prepared.profile, None)
        .map_err(|error| TraceProfileError::new(error.to_string()))?;
    if profile_container::process_signal_cancelled() {
        return Err(TraceProfileError::new(
            "profile interrupted before pipeline publication",
        ));
    }
    let report = crate::profile_pipeline::finish_profiled_replay(
        &run.report,
        &prepared.output_dir,
        Path::new("replay/replay_report.json"),
        fields,
        &config.profile_sets,
        config.coordinated_baseline,
    )
    .map_err(|error| TraceProfileError::new(error.to_string()))?;
    if profile_container::process_signal_cancelled() {
        return Err(TraceProfileError::new(
            "profile interrupted during pipeline publication",
        ));
    }
    Ok(report)
}

fn usage(program: &str, output: &mut dyn Write) {
    let _ = writeln!(
        output,
        "usage: {program} [-h] [--image IMAGE] [--runtime {{docker,enroot,current}}] --output-dir OUTPUT_DIR [--cpu-limit CPU_LIMIT] [--cpuset-cpus CPUSET_CPUS] [--allocated-cpus ALLOCATED_CPUS] [--pacing {{recorded,unpaced}}] [--loop-order] [options] trace"
    );
}

fn help(program: &str) {
    usage(program, &mut io::stdout());
    println!();
    println!(
        "Profile a finalized TaskTrace shell-action trace in a disposable Docker or Enroot image."
    );
    println!();
    println!("The native replay controller is mounted read-only into the measured sandbox.");
    println!("Supported profiler modes: explicit --strace-mode execve/process, untraced baseline, --memory-profile, --coordinated-baseline, --profile-set cpu-average, --profile-set cpu-peak, --profile-set cpu-compare, --profile-set memory-detail, --profile-set software-deep, --profile-set hardware-deep, and host-context --proc-stat-comparison.");
    println!("Native scheduler tracing supports --scheduler-backend bcc (default) and perf.");
    println!("Native Enroot requests are wall-time-only and do not accept profiler sets.");
    println!("--runtime current replays inside the container that already runs TaskTrace (no --image, no nested container): CPU accounting comes from this container's cgroup; --current-container-exclusive asserts the sandbox is dedicated to the replay.");
    println!("--loop-order (enroot or current, --pacing unpaced) replays alongside an LLM-side replay in the recorded agent loop's order, handing off through <output-dir>/container_profile/loop_order; the trace's actions need after_llm_index (tasktrace-capture --trajectory-timings omit).");
}

fn cli_error(program: &str, message: impl fmt::Display) -> i32 {
    usage(program, &mut io::stderr());
    eprintln!("{program}: error: {message}");
    2
}

fn display(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Null) | None => "None".to_owned(),
        Some(value) => value.to_string(),
    }
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
    let result = run_trace_profile(&config);
    drop(guard);
    let report = match result {
        Ok(report) => report,
        Err(error) if profile_container::process_signal_cancelled() => {
            eprintln!("{public_program}: interrupted: {error}");
            return 130;
        }
        Err(error) => return cli_error(public_program, error),
    };
    let replay = &report["replay_summary"];
    println!("Task: {}", display(report.get("task_name")));
    println!("Source trial: {}", display(report.get("source_trial")));
    println!(
        "Actions: {}/{} executed, {} timed out, {} source-outcome mismatches",
        replay["actions_executed"],
        replay["actions_recorded"],
        replay["actions_timed_out"],
        replay["source_outcome_mismatch_count"],
    );
    print_cpu(&report["container_cpu"]);
    if let Some(scheduler) = report
        .get("scheduler_concurrency")
        .filter(|value| !value.is_null())
    {
        println!(
            "Peak simultaneous logical cores: {}",
            scheduler["peak_simultaneous_logical_cores"]
        );
        println!(
            "Scheduler/cgroup CPU validation: {}",
            if scheduler["cgroup_validation"]["valid_for_concurrency_reporting"]
                == Value::Bool(true)
            {
                "valid"
            } else {
                "invalid"
            }
        );
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

    #[test]
    fn accepts_native_cpu_average_slice() {
        let parsed = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--image",
            "sha256:image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--pacing",
            "unpaced",
            "--profile-set",
            "cpu-average",
            "--network-policy",
            "no-network",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.cpuset_cpus.as_deref(), Some("0-3"));
        assert_eq!(parsed.profile_sets, ["cpu-average"]);
        assert_eq!(
            parsed.network_policy.effective_mode(),
            crate::network_policy::NetworkMode::NoNetwork
        );
    }

    #[test]
    fn accepts_and_maps_native_proc_stat_context() {
        let values = arguments(&[
            "trace.jsonl",
            "--image",
            "sha256:image",
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
    fn accepts_native_memory_flag_and_memory_detail_set() {
        let values = arguments(&[
            "trace.jsonl",
            "--image",
            "sha256:image",
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
        assert!(parsed.profile_sets.is_empty());
        assert_eq!(parsed.memory_sample_interval_sec, 0.02);
        assert_eq!(parsed.memory_limit_bytes, Some(134_217_728));
        assert_eq!(parsed.memory_swap_total_limit_bytes, Some(134_217_728));

        let detail = arguments(&[
            "trace.jsonl",
            "--image",
            "sha256:image",
            "--output-dir",
            "out",
            "--cpuset-cpus",
            "0-3",
            "--profile-set",
            "memory-detail",
        ]);
        let parsed = parse_arguments(&detail).unwrap().unwrap();
        assert!(!parsed.memory_profile);
        assert_eq!(parsed.profile_sets, ["memory-detail"]);
    }

    #[test]
    fn proc_stat_requires_enforced_cpuset() {
        let error = parse_arguments(&arguments(&[
            "trace.jsonl",
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
    fn perf_scheduler_modes_stay_native_and_preserve_configuration() {
        for mode in [
            vec!["--scheduler-concurrency"],
            vec!["--profile-set", "cpu-peak"],
            vec!["--profile-set", "cpu-compare"],
        ] {
            let mut values = vec![
                "trace.jsonl",
                "--image",
                "image",
                "--output-dir",
                "out",
                "--cpuset-cpus",
                "0-3",
                "--scheduler-backend",
                "perf",
                "--scheduler-perf-binary",
                "/opt/test-perf",
                "--scheduler-perf-mmap-pages",
                "128",
                "--scheduler-attach-timeout-sec",
                "3.5",
                "--scheduler-cpu-tolerance-fraction",
                "0.03",
                "--scheduler-cpu-tolerance-sec",
                "0.007",
            ];
            values.extend(mode);
            let arguments = arguments(&values);
            let parsed = parse_arguments(&arguments).unwrap().unwrap();
            assert_eq!(parsed.scheduler_backend, "perf");
            assert_eq!(parsed.scheduler_perf_binary, "/opt/test-perf");
            assert_eq!(parsed.scheduler_perf_mmap_pages, Some(128));
            assert_eq!(parsed.attach_timeout, Duration::from_secs_f64(3.5));
            assert_eq!(parsed.scheduler_cpu_tolerance_fraction, 0.03);
            assert_eq!(parsed.scheduler_cpu_tolerance_sec, 0.007);
        }
    }

    #[test]
    fn bcc_scheduler_modes_stay_native_and_preserve_buffer_pages() {
        for mode in [
            vec!["--scheduler-concurrency"],
            vec!["--profile-set", "cpu-peak"],
            vec!["--profile-set", "cpu-compare"],
        ] {
            let mut values = vec![
                "trace.jsonl",
                "--image",
                "image",
                "--output-dir",
                "out",
                "--cpuset-cpus",
                "0-3",
                "--scheduler-backend",
                "bcc",
                "--scheduler-bcc-buffer-pages",
                "64",
            ];
            values.extend(mode);
            let arguments = arguments(&values);
            let parsed = parse_arguments(&arguments).unwrap().unwrap();
            assert_eq!(parsed.scheduler_backend, "bcc");
            assert_eq!(parsed.scheduler_bcc_buffer_pages, Some(64));
        }
    }

    #[test]
    fn process_modes_stay_native_and_preserve_coordination_selection() {
        for (mode, expected) in [("execve", "execve"), ("process", "process")] {
            let values = arguments(&[
                "trace.jsonl",
                "--image",
                "image",
                "--output-dir",
                "out",
                "--allocated-cpus",
                "2",
                "--strace-mode",
                mode,
            ]);
            let parsed = parse_arguments(&values).unwrap().unwrap();
            assert_eq!(parsed.strace_mode, expected);
            assert!(parsed.profile_sets.is_empty());
            assert!(!parsed.coordinated_baseline);
        }

        let software_deep = arguments(&[
            "trace.jsonl",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "2",
            "--profile-set",
            "software-deep",
        ]);
        let parsed = parse_arguments(&software_deep).unwrap().unwrap();
        assert_eq!(parsed.strace_mode, "none");
        assert_eq!(parsed.profile_sets, ["software-deep"]);
        assert!(!parsed.coordinated_baseline);
    }

    #[test]
    fn enroot_wall_only_requests_parse_and_profiler_modes_fail_locally() {
        let supported = arguments(&[
            "trace.jsonl",
            "--runtime",
            "enroot",
            "--enroot",
            "/opt/enroot",
            "--image",
            "task.sqsh",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "2",
            "--strace-mode",
            "none",
        ]);
        let parsed = parse_arguments(&supported).unwrap().unwrap();
        assert_eq!(parsed.runtime, "enroot");
        assert_eq!(parsed.enroot, "/opt/enroot");
        assert!(parsed.profile_sets.is_empty());

        for (mode, expected) in [
            (
                vec!["--profile-set", "cpu-average"],
                "requires the Docker runtime",
            ),
            (vec!["--scheduler-concurrency"], "currently Docker-only"),
            (vec!["--memory-profile"], "currently Docker-only"),
            (vec!["--cpu-limit", "1"], "Docker-only"),
        ] {
            let mut values = vec![
                "trace.jsonl",
                "--runtime",
                "enroot",
                "--image",
                "task.sqsh",
                "--output-dir",
                "out",
                "--allocated-cpus",
                "2",
                "--strace-mode",
                "none",
            ];
            values.extend(mode);
            let error = parse_arguments(&arguments(&values)).unwrap_err();
            assert!(error.to_string().contains(expected), "{values:?}: {error}");
        }
    }

    #[test]
    fn loop_order_needs_enroot_or_current_and_unpaced_and_hands_off_through_the_profile_dir() {
        let enroot = |extra: &[&'static str]| {
            let mut values = vec![
                "trace.jsonl",
                "--runtime",
                "enroot",
                "--image",
                "task.sqsh",
                "--output-dir",
                "out",
                "--allocated-cpus",
                "2",
                "--strace-mode",
                "none",
                "--loop-order",
            ];
            values.extend(extra);
            parse_arguments(&arguments(&values))
        };
        let config = enroot(&["--pacing", "unpaced"]).unwrap().unwrap();
        assert!(config.loop_order);
        assert!(replay_command(&config, "trace.jsonl")
            .ends_with("--pacing unpaced --loop-sync-dir /tasktrace-profile-output/loop_order"));

        let error = enroot(&["--pacing", "recorded"]).unwrap_err();
        assert!(
            error.to_string().contains("use --pacing unpaced"),
            "{error}"
        );
        let error = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--pacing",
            "unpaced",
            "--loop-order",
        ]))
        .unwrap_err();
        assert!(error.to_string().contains("enroot or current"), "{error}");
    }

    #[test]
    fn allowlist_and_backend_overrides_stay_native() {
        let parsed = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "2",
            "--network-policy",
            "allowlist",
            "--allowed-host",
            "Allowed.TEST",
            "--network-policy-proxy-image",
            "fixture-proxy",
            "--network-policy-external-network",
            "fixture-external",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(
            parsed.network_policy.effective_mode(),
            crate::network_policy::NetworkMode::Allowlist
        );
    }

    #[test]
    fn unsupported_profiler_options_fail_before_runtime() {
        let mut values = vec![
            "trace.jsonl",
            "--image",
            "image",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "2",
        ];
        values.extend(["--profile-set", "unknown-deep"]);
        let error = parse_arguments(&arguments(&values)).unwrap_err();
        assert!(error.to_string().contains("does not yet support"));
    }

    #[test]
    fn current_runtime_parses_without_image_and_rejects_docker_only_controls() {
        let parsed = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--runtime",
            "current",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "4",
            "--workdir",
            "/testbed",
            "--current-container-exclusive",
        ]))
        .unwrap()
        .unwrap();
        assert_eq!(parsed.runtime, "current");
        assert_eq!(parsed.image, None);
        assert!(parsed.current_container_exclusive);
        assert_eq!(parsed.allocated_cpus, Some(4.0));
        assert_eq!(parsed.workdir, "/testbed");

        let minimal = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--runtime",
            "current",
            "--output-dir",
            "out",
        ]))
        .unwrap()
        .unwrap();
        assert!(!minimal.current_container_exclusive);
        assert_eq!(minimal.allocated_cpus, None);

        for (extra, message) in [
            (
                vec!["--image", "sha256:image"],
                "--runtime current does not accept --image",
            ),
            (vec!["--cpuset-cpus", "0-3"], "cannot enforce --cpuset-cpus"),
            (vec!["--cpu-limit", "2"], "cannot enforce --cpu-limit"),
            (
                vec!["--profile-set", "cpu-average"],
                "--profile-set is not supported by current-container runtime",
            ),
            (
                vec!["--network-policy", "no-network"],
                "network policy controls are not supported",
            ),
            (
                vec!["--memory-profile"],
                "memory profiling options are not supported",
            ),
            (
                vec!["--scheduler-concurrency"],
                "scheduler profiling options are not supported",
            ),
            (vec!["--image", ""], "--runtime current does not accept --image"),
            (
                vec!["--scheduler-backend", "bcc"],
                "scheduler profiling options are not supported",
            ),
            (
                vec!["--docker", "docker"],
                "runtime executable overrides are not supported",
            ),
            (
                vec!["--enroot", "enroot"],
                "runtime executable overrides are not supported",
            ),
        ] {
            let mut values = vec!["trace.jsonl", "--runtime", "current", "--output-dir", "out"];
            values.extend(extra);
            let error = parse_arguments(&arguments(&values))
                .err()
                .expect("current-container runtime rejects Docker-only controls");
            assert!(error.to_string().contains(message), "{error}");
        }

        let error = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--image",
            "sha256:image",
            "--output-dir",
            "out",
            "--allocated-cpus",
            "2",
            "--current-container-exclusive",
        ]))
        .err()
        .expect("exclusivity assertion is current-only");
        assert!(
            error
                .to_string()
                .contains("--current-container-exclusive requires --runtime current"),
            "{error}"
        );
        let error = parse_arguments(&arguments(&["trace.jsonl", "--output-dir", "out"]))
            .err()
            .expect("docker still requires --image");
        assert!(
            error
                .to_string()
                .contains("the following arguments are required: --image"),
            "{error}"
        );
    }

    #[test]
    fn current_replay_command_targets_local_trace_and_output_without_mounts() {
        let config = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--runtime",
            "current",
            "--output-dir",
            "out",
            "--workdir",
            "/testbed",
            "--pacing",
            "unpaced",
            "--timeout-sec",
            "30",
        ]))
        .unwrap()
        .unwrap();
        let command = current_replay_command(
            &config,
            Path::new("/opt/tasktrace/bin/tasktrace"),
            Path::new("/traces/pytest-7236/trajectory.jsonl"),
            Path::new("/artifacts/run/profile"),
        );
        assert_eq!(
            command,
            "/opt/tasktrace/bin/tasktrace replay trace /traces/pytest-7236/trajectory.jsonl --output /artifacts/run/profile/replay/replay_report.json --cwd /testbed --pacing unpaced --timeout-sec 30"
        );
        assert!(!command.contains(TRACE_MOUNT));
        assert!(!command.contains(CONTROLLER_EXECUTABLE));
    }

    #[test]
    fn current_loop_order_hands_off_through_the_profile_dir_in_place() {
        let config = parse_arguments(&arguments(&[
            "trace.jsonl",
            "--runtime",
            "current",
            "--output-dir",
            "out",
            "--pacing",
            "unpaced",
            "--loop-order",
        ]))
        .unwrap()
        .unwrap();
        assert!(config.loop_order);
        let command = current_replay_command(
            &config,
            Path::new("/opt/tasktrace/bin/tasktrace"),
            Path::new("/traces/trajectory.jsonl"),
            Path::new("/artifacts/run/profile"),
        );
        assert!(command.ends_with(
            "--pacing unpaced --loop-sync-dir /artifacts/run/profile/container_profile/loop_order"
        ));
        assert!(!command.contains(LOOP_SYNC_PROFILE_DIR));
    }

    #[test]
    fn replay_command_uses_native_controller_and_quotes_paths() {
        let config = TraceProfileConfig {
            runtime: "docker".to_owned(),
            trace: PathBuf::from("trace.jsonl"),
            image: Some("image".to_owned()),
            output_dir: PathBuf::from("out"),
            cpu_limit: None,
            cpuset_cpus: Some("0".to_owned()),
            allocated_cpus: None,
            pacing: "recorded".to_owned(),
            timeout_sec: Some(2.5),
            prepare_command: None,
            verification_command: None,
            strace_mode: "none".to_owned(),
            string_limit: 4096,
            shell: "/bin/bash".to_owned(),
            workdir: "/work dir".to_owned(),
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
            profile_sets: Vec::new(),
            coordinated_baseline: false,
            attach_timeout: Duration::from_secs(20),
            current_container_exclusive: false,
            loop_order: false,
        };
        let command = replay_command(&config, "trace file.jsonl");
        assert!(command.starts_with("/tasktrace-controller/tasktrace replay trace"));
        assert!(command.contains("'/tasktrace-input/trace file.jsonl'"));
        assert!(command.contains("'/work dir'"));
        assert!(!command.contains("python"));
    }
}
