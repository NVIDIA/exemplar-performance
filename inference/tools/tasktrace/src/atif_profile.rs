// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native measured ATIF replay adapter.
//!
//! The adapter owns ATIF-specific identity, replay command construction, and
//! the stable pipeline envelope.  Disposable-container lifecycle and cgroup
//! accounting stay in `profile_container`; replay semantics stay in
//! `atif_replay`.

use crate::atif_replay::{
    read_trajectory, validate_supported_actions, AtifTrajectory, PACING_MODES,
};
use crate::network_policy::{resolve_cli_network_policy, NetworkPolicy};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::time::Duration;

pub const PROFILE_SCHEMA_VERSION: &str = "container-atif-trajectory-profile.v1";
pub const PREFLIGHT_SCHEMA_VERSION: &str = "tasktrace-atif-profile-preflight.v1";
pub const TOOL_SOURCE_MOUNT: &str = "/tasktrace-source";
pub const TRAJECTORY_MOUNT: &str = "/tasktrace-input";
pub const REPLAY_MOUNT: &str = "/tasktrace";
pub const RUST_CONTROLLER_MOUNT: &str = "/tasktrace-controller";
pub const RUST_CONTROLLER_EXECUTABLE: &str = "/tasktrace-controller/tasktrace";
pub const RUST_CONTROLLER_DIGEST_ENV: &str = "TASKTRACE_RUST_CONTROLLER_SHA256";
pub const RUN_CANCEL_FILE_ENV: &str = "TASKTRACE_RUN_CANCEL_FILE";
const DEFAULT_PROXY_IMAGE: &str = "python:3.12-slim";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtifProfileError(String);

impl AtifProfileError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for AtifProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for AtifProfileError {}

impl From<io::Error> for AtifProfileError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for AtifProfileError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

/// Full public ATIF-profile request.
///
/// Options unsupported by a selected runtime are retained and validated here
/// so the native controller fails closed instead of silently dropping them.
#[derive(Clone, Debug, PartialEq)]
pub struct AtifProfileConfig {
    pub trajectory: PathBuf,
    pub image: Option<String>,
    pub output_dir: PathBuf,
    pub task_name: Option<String>,
    pub cpu_limit: Option<f64>,
    pub cpuset_cpus: Option<String>,
    pub allocated_cpus: Option<f64>,
    pub pacing: String,
    pub timeout_sec: Option<f64>,
    pub capture_action_outputs: bool,
    pub prepare_command: Option<String>,
    pub verification_command: Option<String>,
    pub controller_runtime: String,
    pub hide_prepared_python3: bool,
    pub strace_mode: String,
    pub coordinated_baseline: bool,
    pub profile_sets: Vec<String>,
    pub string_limit: u64,
    pub scheduler_concurrency: bool,
    pub scheduler_backend: String,
    pub scheduler_bcc_buffer_pages: Option<u64>,
    pub scheduler_perf_binary: String,
    pub scheduler_perf_mmap_pages: Option<u64>,
    pub proc_stat_comparison: bool,
    pub proc_stat_interval_sec: f64,
    pub proc_stat_baseline_sec: f64,
    pub memory_profile: bool,
    pub memory_sample_interval_sec: f64,
    pub memory_limit_bytes: Option<u64>,
    pub memory_swap_total_limit_bytes: Option<u64>,
    pub hardware_perf_binary: String,
    pub hardware_perf_events: Vec<String>,
    pub hardware_perf_attach_timeout_sec: f64,
    pub hardware_perf_minimum_running_ratio: f64,
    pub scheduler_attach_timeout_sec: f64,
    pub scheduler_cpu_tolerance_fraction: f64,
    pub scheduler_cpu_tolerance_sec: f64,
    pub shell: String,
    pub workdir: String,
    /// NAME=VALUE assignments applied inside the measured container. Used to
    /// pin the recording environment so replays do not inherit site- or
    /// enroot-version-dependent image-ENV application.
    pub environment: Vec<String>,
    pub legacy_network: Option<String>,
    pub network_policy: NetworkPolicy,
    pub network_policy_was_explicit: bool,
    pub network_policy_proxy_image: String,
    pub network_policy_external_network: String,
    pub patch_roots: Vec<String>,
    pub docker: String,
    pub enroot: String,
    pub runtime: String,
    pub current_container_exclusive: bool,
    pub run_wave_start_gate: bool,
}

impl AtifProfileConfig {
    fn native_slice_error(&self) -> Option<String> {
        if self.runtime == "current" {
            let unsupported = self.controller_runtime != "image"
                || self.hide_prepared_python3
                || self.scheduler_concurrency
                || self.scheduler_backend != "bcc"
                || self.scheduler_bcc_buffer_pages.is_some()
                || self.scheduler_perf_binary != "perf"
                || self.scheduler_perf_mmap_pages.is_some()
                || self.scheduler_attach_timeout_sec != 20.0
                || self.scheduler_cpu_tolerance_fraction != 0.02
                || self.scheduler_cpu_tolerance_sec != 0.005
                || self.proc_stat_comparison
                || self.proc_stat_interval_sec != 0.5
                || self.proc_stat_baseline_sec != 5.0
                || self.memory_profile
                || self.memory_sample_interval_sec != 0.1
                || self.memory_limit_bytes.is_some()
                || self.memory_swap_total_limit_bytes.is_some()
                || self.hardware_perf_binary != "perf"
                || self.hardware_perf_events
                    != crate::hardware_perf::PORTABLE_HARDWARE_EVENTS
                        .iter()
                        .map(|event| (*event).to_owned())
                        .collect::<Vec<_>>()
                || self.hardware_perf_attach_timeout_sec != 20.0
                || self.hardware_perf_minimum_running_ratio != 0.95
                || self.coordinated_baseline
                || !self.profile_sets.is_empty()
                || self.verification_command.is_some()
                || self.network_policy_was_explicit
                || self.network_policy_proxy_image != DEFAULT_PROXY_IMAGE
                || self.network_policy_external_network != "bridge"
                || self.docker != "docker"
                || self.enroot != "enroot";
            return unsupported.then(|| {
                "current-container runtime supports CPU accounting and optional strace only; Docker-oriented profiler, verification, network, and runtime override options are rejected"
                    .to_owned()
            });
        }
        if self.runtime == "enroot" {
            if self.controller_runtime != "rust-host-mount" {
                return Some(format!(
                    "native Enroot ATIF profiling requires --controller-runtime rust-host-mount; {:?} is unsupported",
                    self.controller_runtime
                ));
            }
            let unsupported = self.cpu_limit.is_some()
                || self.strace_mode != "none"
                || self.coordinated_baseline
                || !self.profile_sets.is_empty()
                || self.scheduler_concurrency
                || self.scheduler_backend != "bcc"
                || self.scheduler_bcc_buffer_pages.is_some()
                || self.scheduler_perf_binary != "perf"
                || self.scheduler_perf_mmap_pages.is_some()
                || self.scheduler_attach_timeout_sec != 20.0
                || self.scheduler_cpu_tolerance_fraction != 0.02
                || self.scheduler_cpu_tolerance_sec != 0.005
                || self.proc_stat_comparison
                || self.proc_stat_interval_sec != 0.5
                || self.proc_stat_baseline_sec != 5.0
                || self.memory_profile
                || self.memory_sample_interval_sec != 0.1
                || self.memory_limit_bytes.is_some()
                || self.memory_swap_total_limit_bytes.is_some()
                || self.hardware_perf_binary != "perf"
                || self.hardware_perf_events
                    != crate::hardware_perf::PORTABLE_HARDWARE_EVENTS
                        .iter()
                        .map(|event| (*event).to_owned())
                        .collect::<Vec<_>>()
                || self.hardware_perf_attach_timeout_sec != 20.0
                || self.hardware_perf_minimum_running_ratio != 0.95
                || self.verification_command.is_some()
                || self
                    .legacy_network
                    .as_deref()
                    .is_some_and(|network| network != "none")
                || self.network_policy.effective_mode()
                    == crate::network_policy::NetworkMode::Allowlist
                || self.network_policy_proxy_image != DEFAULT_PROXY_IMAGE
                || self.network_policy_external_network != "bridge"
                || self.docker != "docker"
                || self.current_container_exclusive;
            return unsupported.then(|| {
                "native Enroot ATIF profiling is wall-time-only; Docker resource profilers, verification, allowlist networking, and unrelated runtime overrides are unsupported"
                    .to_owned()
            });
        }
        if self.runtime != "docker" {
            return Some(format!(
                "native profile atif requires --runtime docker, enroot, or current; {:?} is unsupported",
                self.runtime
            ));
        }
        if self.controller_runtime != "rust-host-mount" {
            return Some(format!(
                "native Docker ATIF profiling requires --controller-runtime rust-host-mount; {:?} is unsupported",
                self.controller_runtime
            ));
        }
        if !matches!(self.scheduler_backend.as_str(), "bcc" | "perf")
            || self.profile_sets.iter().any(|value| {
                !matches!(
                    value.as_str(),
                    "cpu-average"
                        | "cpu-peak"
                        | "cpu-compare"
                        | "memory-detail"
                        | "software-deep"
                        | "hardware-deep"
                )
            })
        {
            return Some(
                "native profile atif supports explicit process and BCC/perf scheduler tracing plus every declared profile set"
                    .to_owned(),
            );
        }
        None
    }
}

#[derive(Default)]
struct ParsedArguments {
    trajectory: Option<PathBuf>,
    image: Option<String>,
    output_dir: Option<PathBuf>,
    task_name: Option<String>,
    cpu_limit: Option<f64>,
    cpuset_cpus: Option<String>,
    allocated_cpus: Option<f64>,
    pacing: String,
    timeout_sec: Option<f64>,
    capture_action_outputs: bool,
    prepare_command: Option<String>,
    verification_command: Option<String>,
    controller_runtime: String,
    controller_runtime_was_explicit: bool,
    hide_prepared_python3: bool,
    strace_mode: String,
    coordinated_baseline: bool,
    profile_sets: Vec<String>,
    string_limit: i64,
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
    environment: Vec<String>,
    legacy_network: Option<String>,
    network_policy_mode: Option<String>,
    allowed_hosts: Vec<String>,
    network_policy_proxy_image: String,
    network_policy_external_network: String,
    patch_roots: Vec<String>,
    docker: String,
    enroot: String,
    runtime: String,
    current_container_exclusive: bool,
    run_wave_start_gate: bool,
}

impl ParsedArguments {
    fn defaults() -> Self {
        Self {
            pacing: "recorded".to_owned(),
            controller_runtime: "rust-host-mount".to_owned(),
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
            workdir: "/app".to_owned(),
            network_policy_proxy_image: DEFAULT_PROXY_IMAGE.to_owned(),
            network_policy_external_network: "bridge".to_owned(),
            docker: "docker".to_owned(),
            enroot: "enroot".to_owned(),
            runtime: "docker".to_owned(),
            ..Self::default()
        }
    }
}

fn text_argument(value: &OsStr, option: &str) -> Result<String, AtifProfileError> {
    value
        .to_str()
        .map(ToOwned::to_owned)
        .ok_or_else(|| AtifProfileError::new(format!("{option} must be valid UTF-8")))
}

fn parse_number<T>(value: &str, option: &str) -> Result<T, AtifProfileError>
where
    T: std::str::FromStr,
{
    value
        .parse::<T>()
        .map_err(|_| AtifProfileError::new(format!("invalid value for {option}: {value:?}")))
}

fn positive_finite(value: f64, option: &str) -> Result<f64, AtifProfileError> {
    if value.is_finite() && value > 0.0 {
        Ok(value)
    } else {
        Err(AtifProfileError::new(format!("{option} must be positive")))
    }
}

fn positive_u64(value: Option<i64>, option: &str) -> Result<Option<u64>, AtifProfileError> {
    value
        .map(|value| {
            u64::try_from(value)
                .ok()
                .filter(|value| *value > 0)
                .ok_or_else(|| AtifProfileError::new(format!("{option} must be positive")))
        })
        .transpose()
}

fn split_long_option(value: &str) -> (&str, Option<&str>) {
    value
        .split_once('=')
        .map_or((value, None), |(option, argument)| (option, Some(argument)))
}

pub fn parse_arguments(
    arguments: &[OsString],
) -> Result<Option<AtifProfileConfig>, AtifProfileError> {
    if arguments.len() == 1 && matches!(arguments[0].to_str(), Some("-h" | "--help")) {
        return Ok(None);
    }
    let mut parsed = ParsedArguments::defaults();
    let mut index = 0;
    let mut positional_only = false;
    while index < arguments.len() {
        let raw = text_argument(&arguments[index], "argument")?;
        if !positional_only && matches!(raw.as_str(), "-h" | "--help") {
            return Ok(None);
        }
        if !positional_only && raw == "--" {
            positional_only = true;
            index += 1;
            continue;
        }
        if positional_only || !raw.starts_with('-') {
            if parsed.trajectory.is_some() {
                return Err(AtifProfileError::new(format!(
                    "unrecognized arguments: {raw}"
                )));
            }
            parsed.trajectory = Some(PathBuf::from(&raw));
            index += 1;
            continue;
        }
        let (option, attached) = split_long_option(&raw);
        let flag = |attached: Option<&str>| -> Result<(), AtifProfileError> {
            if attached.is_some() {
                Err(AtifProfileError::new(format!(
                    "{option} does not take a value"
                )))
            } else {
                Ok(())
            }
        };
        let mut take = || -> Result<String, AtifProfileError> {
            if let Some(value) = attached {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .ok_or_else(|| {
                    AtifProfileError::new(format!("argument {option}: expected one argument"))
                })
                .and_then(|value| text_argument(value, option))
        };
        match option {
            "--image" => parsed.image = Some(take()?),
            "--runtime" => parsed.runtime = take()?,
            "--output-dir" => parsed.output_dir = Some(PathBuf::from(take()?)),
            "--task-name" => parsed.task_name = Some(take()?),
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
            "--capture-action-outputs" => {
                flag(attached)?;
                parsed.capture_action_outputs = true;
            }
            "--prepare-command" => parsed.prepare_command = Some(take()?),
            "--verification-command" => parsed.verification_command = Some(take()?),
            "--controller-runtime" => {
                parsed.controller_runtime = take()?;
                parsed.controller_runtime_was_explicit = true;
            }
            "--hide-prepared-python3" => {
                flag(attached)?;
                parsed.hide_prepared_python3 = true;
            }
            "--strace-mode" => parsed.strace_mode = take()?,
            "--coordinated-baseline" => {
                flag(attached)?;
                parsed.coordinated_baseline = true;
            }
            "--profile-set" => parsed.profile_sets.push(take()?),
            "--strace-string-limit" => {
                let value = take()?;
                parsed.string_limit = parse_number(&value, option)?;
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
            "--env" => parsed.environment.push(take()?),
            "--network" => parsed.legacy_network = Some(take()?),
            "--network-policy" => parsed.network_policy_mode = Some(take()?),
            "--allowed-host" => parsed.allowed_hosts.push(take()?),
            "--network-policy-proxy-image" => parsed.network_policy_proxy_image = take()?,
            "--network-policy-external-network" => {
                parsed.network_policy_external_network = take()?;
            }
            "--patch-root" => parsed.patch_roots.push(take()?),
            "--docker" => parsed.docker = take()?,
            "--enroot" => parsed.enroot = take()?,
            "--current-container-exclusive" => {
                flag(attached)?;
                parsed.current_container_exclusive = true;
            }
            "--run-wave-start-gate" => {
                flag(attached)?;
                parsed.run_wave_start_gate = true;
            }
            _ => {
                return Err(AtifProfileError::new(format!(
                    "unrecognized arguments: {raw}"
                )))
            }
        }
        index += 1;
    }
    validate_arguments(parsed).map(Some)
}

fn validate_arguments(mut parsed: ParsedArguments) -> Result<AtifProfileConfig, AtifProfileError> {
    let trajectory = parsed
        .trajectory
        .ok_or_else(|| AtifProfileError::new("the following arguments are required: trajectory"))?;
    let output_dir = parsed.output_dir.ok_or_else(|| {
        AtifProfileError::new("the following arguments are required: --output-dir")
    })?;
    if !matches!(parsed.runtime.as_str(), "docker" | "enroot" | "current") {
        return Err(AtifProfileError::new(format!(
            "invalid choice for --runtime: {:?} (choose from 'docker', 'enroot', 'current')",
            parsed.runtime
        )));
    }
    if parsed.runtime == "current" && !parsed.controller_runtime_was_explicit {
        parsed.controller_runtime = "image".to_owned();
    }
    if parsed.runtime != "current" && parsed.image.is_none() {
        return Err(AtifProfileError::new(
            "--image is required for Docker and Enroot runtimes",
        ));
    }
    if parsed.runtime == "current" && parsed.image.is_some() {
        return Err(AtifProfileError::new(
            "--runtime current does not accept --image",
        ));
    }
    if parsed.runtime != "current"
        && parsed.cpuset_cpus.is_none()
        && parsed.allocated_cpus.is_none()
    {
        return Err(AtifProfileError::new(
            "provide --cpuset-cpus or --allocated-cpus for utilization reporting",
        ));
    }
    if parsed.run_wave_start_gate && parsed.runtime != "enroot" {
        return Err(AtifProfileError::new(
            "--run-wave-start-gate requires --runtime enroot",
        ));
    }
    parsed.pacing = crate::pacing::canonical(&parsed.pacing).to_owned();
    if !PACING_MODES.contains(&parsed.pacing.as_str()) {
        return Err(AtifProfileError::new(format!(
            "invalid choice for --pacing: {:?}",
            parsed.pacing
        )));
    }
    if !matches!(
        parsed.controller_runtime.as_str(),
        "image" | "host-mount" | "rust-host-mount"
    ) {
        return Err(AtifProfileError::new(format!(
            "invalid choice for --controller-runtime: {:?}",
            parsed.controller_runtime
        )));
    }
    if parsed.runtime != "current" && parsed.controller_runtime != "rust-host-mount" {
        return Err(AtifProfileError::new(format!(
            "--controller-runtime {} is no longer supported for Docker or Enroot profiling; use rust-host-mount",
            parsed.controller_runtime
        )));
    }
    if !matches!(parsed.strace_mode.as_str(), "execve" | "process" | "none") {
        return Err(AtifProfileError::new(format!(
            "invalid choice for --strace-mode: {:?}",
            parsed.strace_mode
        )));
    }
    if parsed.coordinated_baseline && !parsed.profile_sets.is_empty() {
        return Err(AtifProfileError::new(
            "--coordinated-baseline is mutually exclusive with --profile-set",
        ));
    }
    let supported_sets = [
        "cpu-average",
        "cpu-peak",
        "memory-detail",
        "cpu-compare",
        "software-deep",
        "hardware-deep",
    ];
    if let Some(value) = parsed
        .profile_sets
        .iter()
        .find(|value| !supported_sets.contains(&value.as_str()))
    {
        return Err(AtifProfileError::new(format!(
            "invalid choice for --profile-set: {value:?}"
        )));
    }
    if parsed.profile_sets.iter().collect::<BTreeSet<_>>().len() != parsed.profile_sets.len() {
        return Err(AtifProfileError::new(
            "profile sets must not contain duplicates",
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
    let timeout_sec = parsed
        .timeout_sec
        .map(|value| positive_finite(value, "--timeout-sec"))
        .transpose()?;
    if parsed.string_limit <= 0 {
        return Err(AtifProfileError::new(
            "--strace-string-limit must be positive",
        ));
    }
    positive_finite(
        parsed.scheduler_attach_timeout_sec,
        "--scheduler-attach-timeout-sec",
    )?;
    if parsed.scheduler_cpu_tolerance_fraction < 0.0 || parsed.scheduler_cpu_tolerance_sec < 0.0 {
        return Err(AtifProfileError::new(
            "scheduler CPU tolerances must not be negative",
        ));
    }
    if !matches!(parsed.scheduler_backend.as_str(), "bcc" | "perf") {
        return Err(AtifProfileError::new(format!(
            "invalid choice for --scheduler-backend: {:?}",
            parsed.scheduler_backend
        )));
    }
    let scheduler_requested = parsed.scheduler_concurrency
        || parsed
            .profile_sets
            .iter()
            .any(|value| value == "cpu-compare" || value == "cpu-peak");
    if parsed.scheduler_backend != "bcc" && !scheduler_requested {
        return Err(AtifProfileError::new(
            "--scheduler-backend requires --scheduler-concurrency",
        ));
    }
    if let Some(value) = parsed.scheduler_bcc_buffer_pages {
        if value <= 0 || value & (value - 1) != 0 {
            return Err(AtifProfileError::new(
                "--scheduler-bcc-buffer-pages must be a positive power of two",
            ));
        }
    }
    if !scheduler_requested && parsed.scheduler_bcc_buffer_pages.is_some() {
        return Err(AtifProfileError::new(
            "--scheduler-bcc-buffer-pages requires --scheduler-concurrency or a scheduler profile set",
        ));
    }
    if scheduler_requested
        && parsed.scheduler_backend == "perf"
        && parsed.scheduler_perf_binary.trim().is_empty()
    {
        return Err(AtifProfileError::new(
            "--scheduler-perf-binary must be nonempty",
        ));
    }
    if parsed
        .scheduler_perf_mmap_pages
        .is_some_and(|value| value <= 0)
    {
        return Err(AtifProfileError::new(
            "--scheduler-perf-mmap-pages must be positive",
        ));
    }
    positive_finite(parsed.proc_stat_interval_sec, "proc-stat interval")?;
    positive_finite(parsed.proc_stat_baseline_sec, "proc-stat baseline")?;
    positive_finite(
        parsed.memory_sample_interval_sec,
        "--memory-sample-interval-sec",
    )?;
    positive_finite(
        parsed.hardware_perf_attach_timeout_sec,
        "--hardware-perf-attach-timeout-sec",
    )?;
    if !parsed.hardware_perf_minimum_running_ratio.is_finite()
        || parsed.hardware_perf_minimum_running_ratio <= 0.0
        || parsed.hardware_perf_minimum_running_ratio > 1.0
    {
        return Err(AtifProfileError::new(
            "--hardware-perf-minimum-running-ratio must be in (0, 1]",
        ));
    }
    if parsed.hardware_perf_binary.trim().is_empty() {
        return Err(AtifProfileError::new(
            "--hardware-perf-binary must be nonempty",
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
        return Err(AtifProfileError::new(
            "hardware perf events must be nonempty and unique",
        ));
    }
    if parsed.proc_stat_comparison && parsed.cpuset_cpus.is_none() {
        return Err(AtifProfileError::new(
            "--proc-stat-comparison requires --cpuset-cpus",
        ));
    }
    let memory_limit_bytes = positive_u64(parsed.memory_limit_bytes, "--memory-limit-bytes")?;
    let memory_swap_total_limit_bytes = positive_u64(
        parsed.memory_swap_total_limit_bytes,
        "--memory-swap-total-limit-bytes",
    )?;
    if memory_swap_total_limit_bytes.is_some() && memory_limit_bytes.is_none() {
        return Err(AtifProfileError::new(
            "--memory-swap-total-limit-bytes requires --memory-limit-bytes",
        ));
    }
    if memory_swap_total_limit_bytes
        .zip(memory_limit_bytes)
        .is_some_and(|(swap, memory)| swap < memory)
    {
        return Err(AtifProfileError::new(
            "--memory-swap-total-limit-bytes must be at least --memory-limit-bytes",
        ));
    }
    if parsed.runtime == "current" {
        if parsed.cpuset_cpus.is_some() {
            return Err(AtifProfileError::new(
                "--runtime current cannot enforce --cpuset-cpus; configure the sandbox allocation",
            ));
        }
        if cpu_limit.is_some() {
            return Err(AtifProfileError::new(
                "--runtime current cannot enforce --cpu-limit; configure the sandbox allocation",
            ));
        }
        if parsed.controller_runtime != "image" {
            return Err(AtifProfileError::new(format!(
                "--controller-runtime {} is not applicable to --runtime current",
                parsed.controller_runtime
            )));
        }
        if parsed.hide_prepared_python3 {
            return Err(AtifProfileError::new(
                "--hide-prepared-python3 is unsafe with --runtime current",
            ));
        }
        let scheduler_options = parsed.scheduler_concurrency
            || parsed.scheduler_backend != "bcc"
            || parsed.scheduler_bcc_buffer_pages.is_some()
            || parsed.scheduler_perf_binary != "perf"
            || parsed.scheduler_perf_mmap_pages.is_some()
            || parsed.scheduler_attach_timeout_sec != 20.0
            || parsed.scheduler_cpu_tolerance_fraction != 0.02
            || parsed.scheduler_cpu_tolerance_sec != 0.005;
        if scheduler_options {
            return Err(AtifProfileError::new(
                "scheduler profiling options are not supported by current-container runtime",
            ));
        }
        if parsed.proc_stat_comparison
            || parsed.proc_stat_interval_sec != 0.5
            || parsed.proc_stat_baseline_sec != 5.0
        {
            return Err(AtifProfileError::new(
                "proc-stat sampling options are not supported by current-container runtime",
            ));
        }
        if parsed.memory_profile
            || parsed.memory_sample_interval_sec != 0.1
            || memory_limit_bytes.is_some()
            || memory_swap_total_limit_bytes.is_some()
        {
            return Err(AtifProfileError::new(
                "memory profiling options are not supported by current-container runtime",
            ));
        }
        if !parsed.profile_sets.is_empty() {
            return Err(AtifProfileError::new(
                "--profile-set is not supported by current-container runtime",
            ));
        }
        if parsed.coordinated_baseline {
            return Err(AtifProfileError::new(
                "--coordinated-baseline is not supported by current-container runtime",
            ));
        }
        if parsed.verification_command.is_some() {
            return Err(AtifProfileError::new(
                "--verification-command is not supported by current-container runtime",
            ));
        }
        if parsed.hardware_perf_binary != "perf"
            || !parsed.hardware_perf_events.is_empty()
            || parsed.hardware_perf_attach_timeout_sec != 20.0
            || parsed.hardware_perf_minimum_running_ratio
                != crate::hardware_perf::MINIMUM_RUNNING_RATIO
        {
            return Err(AtifProfileError::new(
                "hardware profiling options are not supported by current-container runtime",
            ));
        }
        if parsed.network_policy_mode.is_some()
            || !parsed.allowed_hosts.is_empty()
            || parsed.legacy_network.is_some()
            || parsed.network_policy_proxy_image != DEFAULT_PROXY_IMAGE
            || parsed.network_policy_external_network != "bridge"
        {
            return Err(AtifProfileError::new(
                "network policy controls are not supported by current-container runtime",
            ));
        }
        if parsed.docker != "docker" || parsed.enroot != "enroot" {
            return Err(AtifProfileError::new(
                "runtime executable overrides are not supported by current-container runtime",
            ));
        }
        // The established backend permits a false exclusivity assertion but
        // reports the resulting measurement invalid, so retain the flag.
    } else {
        if parsed.current_container_exclusive {
            return Err(AtifProfileError::new(
                "--current-container-exclusive requires --runtime current",
            ));
        }
    }
    if parsed.runtime == "enroot" && cpu_limit.is_some() {
        return Err(AtifProfileError::new(
            "--cpu-limit is Docker-only; use Slurm/taskset CPU allocation for enroot",
        ));
    }
    let explicit_network = parsed.network_policy_mode.is_some()
        || !parsed.allowed_hosts.is_empty()
        || parsed.legacy_network.is_some();
    let network_policy = resolve_cli_network_policy(
        parsed.network_policy_mode.as_deref(),
        &parsed.allowed_hosts,
        parsed.legacy_network.as_deref(),
    )
    .map_err(|error| AtifProfileError::new(error.to_string()))?
    .unwrap_or_default();
    for assignment in &parsed.environment {
        let name = assignment.split('=').next().unwrap_or("");
        if name.is_empty() {
            return Err(AtifProfileError::new(format!(
                "--env requires NAME=VALUE (or bare NAME to unset): {assignment:?}"
            )));
        }
    }
    let patch_roots = if parsed.patch_roots.is_empty() {
        vec!["/tmp".to_owned()]
    } else {
        parsed.patch_roots
    };
    Ok(AtifProfileConfig {
        trajectory,
        image: parsed.image,
        output_dir,
        task_name: parsed.task_name,
        cpu_limit,
        cpuset_cpus: parsed.cpuset_cpus,
        allocated_cpus,
        pacing: parsed.pacing,
        timeout_sec,
        capture_action_outputs: parsed.capture_action_outputs,
        prepare_command: parsed.prepare_command,
        verification_command: parsed.verification_command,
        controller_runtime: parsed.controller_runtime,
        hide_prepared_python3: parsed.hide_prepared_python3,
        strace_mode: parsed.strace_mode,
        coordinated_baseline: parsed.coordinated_baseline,
        profile_sets: parsed.profile_sets,
        string_limit: parsed.string_limit as u64,
        scheduler_concurrency: parsed.scheduler_concurrency,
        scheduler_backend: parsed.scheduler_backend,
        scheduler_bcc_buffer_pages: parsed.scheduler_bcc_buffer_pages.map(|value| value as u64),
        scheduler_perf_binary: parsed.scheduler_perf_binary,
        scheduler_perf_mmap_pages: parsed.scheduler_perf_mmap_pages.map(|value| value as u64),
        proc_stat_comparison: parsed.proc_stat_comparison,
        proc_stat_interval_sec: parsed.proc_stat_interval_sec,
        proc_stat_baseline_sec: parsed.proc_stat_baseline_sec,
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
        hardware_perf_attach_timeout_sec: parsed.hardware_perf_attach_timeout_sec,
        hardware_perf_minimum_running_ratio: parsed.hardware_perf_minimum_running_ratio,
        scheduler_attach_timeout_sec: parsed.scheduler_attach_timeout_sec,
        scheduler_cpu_tolerance_fraction: parsed.scheduler_cpu_tolerance_fraction,
        scheduler_cpu_tolerance_sec: parsed.scheduler_cpu_tolerance_sec,
        shell: parsed.shell,
        workdir: parsed.workdir,
        environment: parsed.environment,
        legacy_network: parsed.legacy_network,
        network_policy,
        network_policy_was_explicit: explicit_network,
        network_policy_proxy_image: parsed.network_policy_proxy_image,
        network_policy_external_network: parsed.network_policy_external_network,
        patch_roots,
        docker: parsed.docker,
        enroot: parsed.enroot,
        runtime: parsed.runtime,
        current_container_exclusive: parsed.current_container_exclusive,
        run_wave_start_gate: parsed.run_wave_start_gate,
    })
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RustControllerRuntime {
    pub executable: PathBuf,
    pub sha256: String,
}

fn sha256_file(path: &Path) -> Result<String, AtifProfileError> {
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

pub fn rust_controller_runtime() -> Result<RustControllerRuntime, AtifProfileError> {
    let executable = env::var_os("TASKTRACE_RUST_REPLAY_EXECUTABLE")
        .map(PathBuf::from)
        .map(Ok)
        .unwrap_or_else(env::current_exe)?;
    let executable = fs::canonicalize(&executable).map_err(|error| {
        AtifProfileError::new(format!(
            "Rust replay controller is unavailable at {}: {error}",
            executable.display()
        ))
    })?;
    let metadata = fs::metadata(&executable)?;
    if !metadata.is_file() {
        return Err(AtifProfileError::new(format!(
            "Rust replay controller is not a file: {}",
            executable.display()
        )));
    }
    if metadata.mode() & 0o111 == 0 {
        return Err(AtifProfileError::new(format!(
            "Rust replay controller is not executable: {}",
            executable.display()
        )));
    }
    Ok(RustControllerRuntime {
        sha256: sha256_file(&executable)?,
        executable,
    })
}

fn shell_quote(value: &str) -> String {
    if !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"_@%+=:,./-".contains(&byte))
    {
        return value.to_owned();
    }
    format!("'{}'", value.replace('\'', "'\"'\"'"))
}

fn shell_join(values: &[String]) -> String {
    values
        .iter()
        .map(|value| shell_quote(value))
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

/// Adapter-owned request passed to the common container coordinator.
#[derive(Clone, Debug, PartialEq)]
pub struct AtifContainerPlan {
    pub image: String,
    pub command: String,
    pub prepare_command: Option<String>,
    pub output_dir: PathBuf,
    pub mounts: Vec<String>,
    pub workload_identity: Value,
    pub source_identity: Value,
    pub expected_outcome: Value,
    pub controller: RustControllerRuntime,
}

pub fn container_profile_config(
    config: &AtifProfileConfig,
    plan: &AtifContainerPlan,
) -> crate::profile_container::ProfileConfig {
    let mut profile = crate::profile_container::ProfileConfig::new(
        plan.image.clone(),
        plan.command.clone(),
        plan.output_dir.clone(),
    );
    profile.prepare_command = plan.prepare_command.clone();
    profile.verification_command = config.verification_command.clone();
    profile.cpuset_cpus = config.cpuset_cpus.clone();
    profile.allocated_cpus = config.allocated_cpus;
    profile.cpu_limit = config.cpu_limit;
    profile.memory_limit_bytes = config.memory_limit_bytes;
    profile.memory_swap_total_limit_bytes = config.memory_swap_total_limit_bytes;
    profile.workdir = Some(config.workdir.clone());
    profile.environment = config.environment.clone();
    profile.mounts = plan.mounts.clone();
    profile.shell = config.shell.clone();
    profile.string_limit = config.string_limit;
    profile.proc_stat_comparison = config.proc_stat_comparison;
    profile.proc_stat_interval_sec = config.proc_stat_interval_sec;
    profile.proc_stat_baseline_sec = config.proc_stat_baseline_sec;
    profile.memory_profile = config.memory_profile;
    profile.memory_sample_interval_sec = config.memory_sample_interval_sec;
    profile.hardware_perf_binary = config.hardware_perf_binary.clone();
    profile.hardware_perf_events = config.hardware_perf_events.clone();
    profile.hardware_perf_attach_timeout =
        Duration::from_secs_f64(config.hardware_perf_attach_timeout_sec);
    profile.hardware_perf_minimum_running_ratio = config.hardware_perf_minimum_running_ratio;
    profile.scheduler_concurrency = config.scheduler_concurrency;
    profile.scheduler_backend = config.scheduler_backend.clone();
    profile.scheduler_bcc_buffer_pages = config.scheduler_bcc_buffer_pages;
    profile.scheduler_perf_binary = config.scheduler_perf_binary.clone();
    profile.scheduler_perf_mmap_pages = config.scheduler_perf_mmap_pages;
    profile.scheduler_cpu_tolerance_fraction = config.scheduler_cpu_tolerance_fraction;
    profile.scheduler_cpu_tolerance_sec = config.scheduler_cpu_tolerance_sec;
    if config.coordinated_baseline {
        profile.set_coordinated_baseline();
    } else if config.profile_sets.is_empty() {
        profile.set_untraced_baseline();
        profile
            .set_strace_mode(&config.strace_mode)
            .expect("ATIF strace mode was validated before profile construction");
    } else {
        profile.profile_sets = config.profile_sets.clone();
    }
    // ATIF actions can leave background work behind.  The format adapter owns
    // this stronger finish gate even when the sole selected profiler is CPU.
    profile.terminate_descendants_at_finish = config.runtime == "docker";
    profile.network = config.legacy_network.clone();
    profile.network_policy = config.network_policy.clone();
    profile.network_policy_proxy_image = config.network_policy_proxy_image.clone();
    profile.network_policy_external_network = config.network_policy_external_network.clone();
    profile.runtime = config.runtime.clone();
    profile.run_wave_start_gate = config.run_wave_start_gate;
    profile.docker = config.docker.clone();
    profile.enroot = config.enroot.clone();
    profile.attach_timeout = Duration::from_secs_f64(config.scheduler_attach_timeout_sec);
    profile.workload_identity = Some(plan.workload_identity.clone());
    profile.source_identity = Some(plan.source_identity.clone());
    profile.expected_outcome = Some(plan.expected_outcome.clone());
    profile
}

pub fn build_container_plan(
    config: &AtifProfileConfig,
) -> Result<(AtifContainerPlan, AtifTrajectory), AtifProfileError> {
    let controller = rust_controller_runtime()?;
    build_container_plan_with_controller(config, controller)
}

fn current_replay_command(
    config: &AtifProfileConfig,
    controller: &RustControllerRuntime,
) -> Result<String, AtifProfileError> {
    let output_root = absolute(&config.output_dir)?;
    let mut command = vec![
        controller.executable.to_string_lossy().into_owned(),
        "replay".to_owned(),
        "atif".to_owned(),
        fs::canonicalize(&config.trajectory)?
            .to_string_lossy()
            .into_owned(),
        "--output".to_owned(),
        output_root
            .join("replay/replay_report.json")
            .to_string_lossy()
            .into_owned(),
        "--cwd".to_owned(),
        config.workdir.clone(),
        "--pacing".to_owned(),
        config.pacing.clone(),
    ];
    if config.capture_action_outputs {
        command.extend([
            "--action-output-dir".to_owned(),
            output_root
                .join("replay/action_outputs")
                .to_string_lossy()
                .into_owned(),
        ]);
    }
    if let Some(timeout) = config.timeout_sec {
        command.extend(["--timeout-sec".to_owned(), python_float(timeout)]);
    }
    for root in &config.patch_roots {
        command.extend(["--patch-root".to_owned(), root.clone()]);
    }
    Ok(shell_join(&command))
}

fn current_profile_config(
    config: &AtifProfileConfig,
    controller: &RustControllerRuntime,
) -> Result<crate::current_profile::CurrentProfileConfig, AtifProfileError> {
    let invocation_cwd = env::current_dir()?;
    let controller_source =
        crate::current_profile::discover_controller_source(&controller.executable);
    let task_environment = crate::current_profile::sanitized_controller_environment(
        &controller.executable,
        controller_source.as_deref(),
        &invocation_cwd,
    )
    .map_err(|error| AtifProfileError::new(error.to_string()))?;
    Ok(crate::current_profile::CurrentProfileConfig {
        command: current_replay_command(config, controller)?,
        prepare_command: config.prepare_command.clone(),
        output_dir: absolute(&config.output_dir)?.join("container_profile"),
        workdir: PathBuf::from(&config.workdir),
        shell: config.shell.clone(),
        environment: task_environment.assignments,
        unset_environment: task_environment.unset_names,
        strace_mode: config.strace_mode.clone(),
        string_limit: config.string_limit,
        allocated_cpus: config.allocated_cpus,
        exclusive: config.current_container_exclusive,
        adopt_orphans: false,
    })
}

fn build_container_plan_with_controller(
    config: &AtifProfileConfig,
    controller: RustControllerRuntime,
) -> Result<(AtifContainerPlan, AtifTrajectory), AtifProfileError> {
    if let Some(error) = config.native_slice_error() {
        return Err(AtifProfileError::new(error));
    }
    let trajectory = read_trajectory(&config.trajectory)
        .map_err(|error| AtifProfileError::new(error.to_string()))?;
    validate_supported_actions(&trajectory)
        .map_err(|error| AtifProfileError::new(error.to_string()))?;
    let trajectory_parent = config
        .trajectory
        .canonicalize()
        .map_err(|error| {
            AtifProfileError::new(format!(
                "unable to resolve ATIF trajectory {}: {error}",
                config.trajectory.display()
            ))
        })?
        .parent()
        .ok_or_else(|| AtifProfileError::new("ATIF trajectory has no parent directory"))?
        .to_path_buf();
    let source_sha256 = sha256_file(&config.trajectory)?;
    let trajectory_name = config
        .trajectory
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| AtifProfileError::new("ATIF trajectory name must be valid UTF-8"))?;
    let replay_report = format!("{REPLAY_MOUNT}/replay_report.json");
    let mut command = vec![
        RUST_CONTROLLER_EXECUTABLE.to_owned(),
        "replay".to_owned(),
        "atif".to_owned(),
        format!("{TRAJECTORY_MOUNT}/{trajectory_name}"),
        "--output".to_owned(),
        replay_report,
        "--cwd".to_owned(),
        config.workdir.clone(),
        "--pacing".to_owned(),
        config.pacing.clone(),
    ];
    if config.capture_action_outputs {
        command.extend([
            "--action-output-dir".to_owned(),
            format!("{REPLAY_MOUNT}/action_outputs"),
        ]);
    }
    if let Some(timeout) = config.timeout_sec {
        command.extend(["--timeout-sec".to_owned(), python_float(timeout)]);
    }
    for root in &config.patch_roots {
        command.extend(["--patch-root".to_owned(), root.clone()]);
    }
    let command = format!(
        "TASKTRACE_WORKLOAD_ENV_{digest_env}=\"${{{digest_env}-}}\" TASKTRACE_WORKLOAD_ENV_PRESENT_{digest_env}=\"${{{digest_env}+1}}\" {digest_env}={} TASKTRACE_CONTROLLER_ENV_VARS={digest_env} {}",
        controller.sha256,
        shell_join(&command),
        digest_env = RUST_CONTROLLER_DIGEST_ENV,
    );
    let mut prepare = Vec::new();
    if let Some(command) = &config.prepare_command {
        prepare.push(format!("({command})"));
    }
    if config.hide_prepared_python3 {
        prepare.push("rm -f /usr/bin/python3 /usr/local/bin/python3".to_owned());
    }
    let prepare_command = (!prepare.is_empty()).then(|| prepare.join(" && "));
    let task_name = config
        .task_name
        .clone()
        .unwrap_or_else(|| trajectory.task_name());
    let source_identity = json!({
        "format": "atif",
        "schema_version": trajectory.schema_version,
        "sha256": source_sha256,
        "action_count": trajectory.actions.len(),
        "rust_controller_binary_sha256": controller.sha256,
    });
    let replay_dir = config.output_dir.join("replay");
    let mounts = vec![
        format!(
            "type=bind,src={},dst={TRAJECTORY_MOUNT},readonly",
            trajectory_parent.display()
        ),
        format!(
            "type=bind,src={},dst={REPLAY_MOUNT}",
            absolute(&replay_dir)?.display()
        ),
        format!(
            "type=bind,src={},dst={RUST_CONTROLLER_EXECUTABLE},readonly",
            controller.executable.display()
        ),
    ];
    Ok((
        AtifContainerPlan {
            image: config
                .image
                .clone()
                .expect("native Docker slice requires image"),
            command,
            prepare_command,
            output_dir: config.output_dir.join("container_profile"),
            mounts,
            workload_identity: json!({
                "task_name": task_name,
                "trial_name": trajectory.trial_name(),
            }),
            source_identity,
            expected_outcome: json!({
                "kind": "source-recorded-action-outcomes",
                "action_count": trajectory.actions.len(),
            }),
            controller,
        },
        trajectory,
    ))
}

fn absolute(path: &Path) -> Result<PathBuf, AtifProfileError> {
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(env::current_dir()?.join(path))
    }
}

pub fn validate_output_root(path: &Path) -> Result<(), AtifProfileError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(AtifProfileError::new(format!(
                    "pipeline output path is not a directory: {}",
                    path.display()
                )));
            }
            if fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(AtifProfileError::new(format!(
                    "pipeline output directory must be new or empty: {}",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut ancestor = path.parent().unwrap_or_else(|| Path::new("."));
            while !ancestor.exists() {
                ancestor = ancestor.parent().ok_or_else(|| {
                    AtifProfileError::new(format!(
                        "pipeline output has no existing ancestor: {}",
                        path.display()
                    ))
                })?;
            }
            if !fs::metadata(ancestor)?.is_dir() {
                return Err(AtifProfileError::new(format!(
                    "pipeline output ancestor is not a directory: {}",
                    ancestor.display()
                )));
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

pub fn prepare_output_root(path: &Path) -> Result<PathBuf, AtifProfileError> {
    validate_output_root(path)?;
    if !path.exists() {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).map_err(|error| {
                AtifProfileError::new(format!(
                    "unable to create pipeline output parent {}: {error}",
                    parent.display()
                ))
            })?;
        }
        fs::create_dir(path).map_err(|error| {
            AtifProfileError::new(format!(
                "unable to create pipeline output directory {}: {error}",
                path.display()
            ))
        })?;
    }
    let metadata = fs::symlink_metadata(path)?;
    if metadata.file_type().is_symlink() || !metadata.is_dir() {
        return Err(AtifProfileError::new(format!(
            "pipeline output path changed after preflight: {}",
            path.display()
        )));
    }
    if fs::read_dir(path)?.next().transpose()?.is_some() {
        return Err(AtifProfileError::new(format!(
            "pipeline output directory changed after preflight: {}",
            path.display()
        )));
    }
    fs::create_dir(path.join("replay"))?;
    fs::canonicalize(path).map_err(Into::into)
}

fn clone_if_present(source: &Map<String, Value>, target: &mut Map<String, Value>, field: &str) {
    if let Some(value) = source.get(field) {
        if !value.is_null() {
            target.insert(field.to_owned(), value.clone());
        }
    }
}

/// Compose the stable format-specific envelope around one container report.
pub fn compose_pipeline_report(
    config: &AtifProfileConfig,
    trajectory: &AtifTrajectory,
    controller: &RustControllerRuntime,
    container_report: &Value,
) -> Result<Value, AtifProfileError> {
    let container = container_report
        .as_object()
        .ok_or_else(|| AtifProfileError::new("container profile report is not a JSON object"))?;
    let output_root = fs::canonicalize(&config.output_dir).map_err(|error| {
        AtifProfileError::new(format!(
            "pipeline output directory is unavailable {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let replay_path = output_root.join("replay/replay_report.json");
    let profile_path = output_root.join("container_profile/profile_report.json");
    let mut report = Map::new();
    report.insert("schema_version".to_owned(), json!(PROFILE_SCHEMA_VERSION));
    report.insert(
        "task_name".to_owned(),
        json!(config
            .task_name
            .clone()
            .unwrap_or_else(|| trajectory.task_name())),
    );
    report.insert("source_trial".to_owned(), json!(trajectory.trial_name()));
    report.insert(
        "source_trajectory".to_owned(),
        json!(fs::canonicalize(&config.trajectory)?.display().to_string()),
    );
    report.insert(
        "source_trajectory_sha256".to_owned(),
        json!(sha256_file(&config.trajectory)?),
    );
    report.insert(
        "source_trajectory_schema_version".to_owned(),
        json!(trajectory.schema_version),
    );
    report.insert(
        "source_agent_original_format".to_owned(),
        trajectory
            .source_agent_original_format
            .clone()
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    report.insert("agent".to_owned(), Value::Object(trajectory.agent.clone()));
    report.insert("image".to_owned(), json!(config.image));
    report.insert("runtime".to_owned(), json!(config.runtime));
    report.insert(
        "image_identity".to_owned(),
        container
            .get("image_identity")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert("pacing".to_owned(), json!(config.pacing));
    report.insert(
        "controller_runtime".to_owned(),
        json!(config.controller_runtime),
    );
    report.insert(
        "controller_identity".to_owned(),
        json!({
            "kind": "rust-host-mounted-binary",
            "sha256": controller.sha256,
            "container_path": RUST_CONTROLLER_EXECUTABLE,
        }),
    );
    report.insert(
        "hide_prepared_python3".to_owned(),
        json!(config.hide_prepared_python3),
    );
    report.insert("patch_roots".to_owned(), json!(config.patch_roots));
    report.insert("network".to_owned(), json!(config.legacy_network));
    report.insert(
        "action_output_capture".to_owned(),
        json!({
            "enabled": config.capture_action_outputs,
            "directory": if config.capture_action_outputs {
                Some(output_root.join("replay/action_outputs").display().to_string())
            } else {
                None
            },
        }),
    );
    report.insert(
        "replay_report".to_owned(),
        json!(replay_path.display().to_string()),
    );
    report.insert(
        "container_profile_report".to_owned(),
        json!(profile_path.display().to_string()),
    );
    report.insert("replay_summary".to_owned(), Value::Null);
    report.insert(
        "container_cpu".to_owned(),
        container.get("cpu").cloned().unwrap_or(Value::Null),
    );
    report.insert(
        "run_wave_start_gate".to_owned(),
        container
            .get("run_wave_start_gate")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "scheduler_concurrency".to_owned(),
        container
            .get("scheduler_concurrency")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "proc_stat_comparison".to_owned(),
        container
            .get("proc_stat_comparison")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "strace".to_owned(),
        container.get("strace").cloned().unwrap_or_else(|| {
            json!({"mode": config.strace_mode, "log_path": null, "structured_report_path": null, "string_limit": config.string_limit})
        }),
    );
    report.insert(
        "warning".to_owned(),
        container.get("warning").cloned().unwrap_or(Value::Null),
    );
    report.insert("status".to_owned(), json!("valid"));
    report.insert(
        "profile_provenance".to_owned(),
        json!({
            "mode": if config.profile_sets.is_empty() { "baseline" } else { "profile" },
            "profile_sets": config.profile_sets,
        }),
    );
    if let Some(value) = container.get("network_policy") {
        report.insert("network_policy".to_owned(), value.clone());
    }
    for field in [
        "run_id",
        "profile_started_at_utc",
        "canonical_clock_reference",
        "canonical_lifecycle",
        "hardware_perf",
        "process_exec",
        "verification",
        "profiling_valid",
        "profiling_invalid_reasons",
        "unified_profile_artifacts",
    ] {
        clone_if_present(container, &mut report, field);
    }
    if let Some(attachment) = container.get("evidence_bundle") {
        report.insert("coordinator_evidence_bundle".to_owned(), attachment.clone());
        report.insert("evidence_bundle".to_owned(), attachment.clone());
        if let Some(object) = attachment.as_object() {
            clone_if_present(object, &mut report, "execution_plan_digest");
            clone_if_present(object, &mut report, "scope");
            if let Some(plan) = object
                .get("provenance")
                .and_then(Value::as_object)
                .and_then(|value| value.get("execution_plan"))
            {
                report.insert("execution_plan".to_owned(), plan.clone());
            }
        }
    }
    if !replay_path.is_file() {
        report.insert(
            "pipeline_status".to_owned(),
            json!("failed_missing_replay_report"),
        );
        report.insert("status".to_owned(), json!("invalid"));
        let mut diagnostic = String::new();
        let stderr = config
            .output_dir
            .join("container_profile/command.stderr.txt");
        if stderr.is_file() {
            diagnostic = fs::read_to_string(stderr)
                .unwrap_or_default()
                .trim()
                .chars()
                .take(1200)
                .collect();
        }
        let message = format!(
            "container command completed without a replay report: {}{}",
            replay_path.display(),
            if diagnostic.is_empty() {
                String::new()
            } else {
                format!("; command stderr: {diagnostic}")
            }
        );
        report.insert("pipeline_error".to_owned(), json!(message));
        report.insert(
            "container_command_exit_code".to_owned(),
            container
                .get("command_exit_code")
                .cloned()
                .unwrap_or(Value::Null),
        );
        return Ok(Value::Object(report));
    }
    let replay: Value = serde_json::from_slice(&fs::read(&replay_path)?)?;
    let summary = replay
        .get("summary")
        .cloned()
        .ok_or_else(|| AtifProfileError::new("ATIF replay report has no summary"))?;
    let valid_summary = summary
        .as_object()
        .is_some_and(crate::profile_pipeline::atif_summary_is_valid);
    report.insert("replay_summary".to_owned(), summary);
    if !valid_summary || container.get("profiling_valid") == Some(&Value::Bool(false)) {
        report.insert("status".to_owned(), json!("invalid"));
    }
    Ok(Value::Object(report))
}

pub fn write_json(path: &Path, value: &Value) -> Result<(), AtifProfileError> {
    let parent = path
        .parent()
        .ok_or_else(|| AtifProfileError::new("JSON output path has no parent"))?;
    if !parent.is_dir() {
        return Err(AtifProfileError::new(format!(
            "JSON output parent is unavailable: {}",
            parent.display()
        )));
    }
    let temporary = parent.join(format!(
        ".{}.tmp-{}",
        path.file_name()
            .and_then(OsStr::to_str)
            .unwrap_or("pipeline-report"),
        std::process::id()
    ));
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&temporary)?;
    serde_json::to_writer_pretty(&mut file, value)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    fs::rename(&temporary, path)?;
    Ok(())
}

pub fn preflight_envelope(
    config: &AtifProfileConfig,
    trajectory: &AtifTrajectory,
    plan: &AtifContainerPlan,
    container_preflight: Value,
) -> Result<Value, AtifProfileError> {
    let mut report = container_preflight
        .as_object()
        .cloned()
        .ok_or_else(|| AtifProfileError::new("container preflight is not a JSON object"))?;
    let checks = report
        .remove("checks")
        .and_then(|value| value.as_array().cloned())
        .unwrap_or_default();
    let mut all_checks = vec![json!({
        "name": "atif_source",
        "status": "passed",
        "schema_version": trajectory.schema_version,
        "supported_action_count": trajectory.actions.len(),
    })];
    all_checks.extend(checks);
    report.insert("schema_version".to_owned(), json!(PREFLIGHT_SCHEMA_VERSION));
    report.insert("ready".to_owned(), Value::Bool(true));
    report.insert(
        "task_name".to_owned(),
        json!(config
            .task_name
            .clone()
            .unwrap_or_else(|| trajectory.task_name())),
    );
    report.insert("source_trial".to_owned(), json!(trajectory.trial_name()));
    report.insert(
        "source_sha256".to_owned(),
        plan.source_identity["sha256"].clone(),
    );
    report.insert(
        "source_schema_version".to_owned(),
        json!(trajectory.schema_version),
    );
    report.insert(
        "source_action_count".to_owned(),
        json!(trajectory.actions.len()),
    );
    report.insert("runtime".to_owned(), json!(config.runtime));
    report.insert("profile_sets".to_owned(), json!(config.profile_sets));
    report.insert("cpuset_cpus".to_owned(), json!(config.cpuset_cpus));
    report.insert("allocated_cpus".to_owned(), json!(config.allocated_cpus));
    report.insert("checks".to_owned(), Value::Array(all_checks));
    report.insert(
        "controller_identity".to_owned(),
        json!({
            "kind": "rust-host-mounted-binary",
            "sha256": plan.controller.sha256,
            "container_path": RUST_CONTROLLER_EXECUTABLE,
        }),
    );
    Ok(Value::Object(report))
}

pub fn preflight_atif_profile(config: &AtifProfileConfig) -> Result<Value, AtifProfileError> {
    validate_output_root(&config.output_dir)?;
    if config.runtime == "current" {
        if let Some(error) = config.native_slice_error() {
            return Err(AtifProfileError::new(error));
        }
        let trajectory = read_trajectory(&config.trajectory)
            .map_err(|error| AtifProfileError::new(error.to_string()))?;
        validate_supported_actions(&trajectory)
            .map_err(|error| AtifProfileError::new(error.to_string()))?;
        let controller = rust_controller_runtime()?;
        let request = current_profile_config(config, &controller)?;
        let mut preflight = crate::current_profile::preflight_current_profile(&request)
            .map_err(|error| AtifProfileError::new(error.to_string()))?
            .as_object()
            .cloned()
            .ok_or_else(|| AtifProfileError::new("current preflight is not an object"))?;
        let mut checks = vec![json!({
            "name": "atif_source",
            "status": "passed",
            "schema_version": trajectory.schema_version,
            "supported_action_count": trajectory.actions.len(),
        })];
        checks.extend(
            preflight
                .remove("checks")
                .and_then(|value| value.as_array().cloned())
                .unwrap_or_default(),
        );
        preflight.insert("schema_version".to_owned(), json!(PREFLIGHT_SCHEMA_VERSION));
        preflight.insert(
            "task_name".to_owned(),
            json!(config
                .task_name
                .clone()
                .unwrap_or_else(|| trajectory.task_name())),
        );
        preflight.insert("source_trial".to_owned(), json!(trajectory.trial_name()));
        preflight.insert(
            "source_sha256".to_owned(),
            json!(sha256_file(&config.trajectory)?),
        );
        preflight.insert(
            "source_schema_version".to_owned(),
            json!(trajectory.schema_version),
        );
        preflight.insert(
            "source_action_count".to_owned(),
            json!(trajectory.actions.len()),
        );
        preflight.insert("runtime".to_owned(), Value::String("current".to_owned()));
        preflight.insert("profile_sets".to_owned(), json!([]));
        preflight.insert("checks".to_owned(), Value::Array(checks));
        preflight.insert(
            "controller_identity".to_owned(),
            json!({
                "kind": "rust-current-container-binary",
                "sha256": controller.sha256,
                "path": controller.executable,
            }),
        );
        return Ok(Value::Object(preflight));
    }
    let (plan, trajectory) = build_container_plan(config)?;
    let profile = container_profile_config(config, &plan);
    let preflight = crate::profile_container::preflight_profile(&profile)
        .map_err(|error| AtifProfileError::new(error.to_string()))?;
    preflight_envelope(config, &trajectory, &plan, preflight)
}

pub fn run_atif_profile(
    config: &AtifProfileConfig,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<Value, AtifProfileError> {
    validate_output_root(&config.output_dir)?;
    if config.runtime == "current" {
        if let Some(error) = config.native_slice_error() {
            return Err(AtifProfileError::new(error));
        }
        let current_cancelled = || {
            crate::profile_container::process_signal_cancelled()
                || cancel_requested.is_some_and(|callback| callback())
        };
        if current_cancelled() {
            return Err(AtifProfileError::new(
                "current-container profile cancelled before output creation",
            ));
        }
        let trajectory = read_trajectory(&config.trajectory)
            .map_err(|error| AtifProfileError::new(error.to_string()))?;
        validate_supported_actions(&trajectory)
            .map_err(|error| AtifProfileError::new(error.to_string()))?;
        let controller = rust_controller_runtime()?;
        prepare_output_root(&config.output_dir)?;
        let request = current_profile_config(config, &controller)?;
        let mut container_report = crate::current_profile::run_current_profile_with_cancel(
            &request,
            Some(&current_cancelled),
        )
        .map_err(|error| AtifProfileError::new(error.to_string()))?;
        if current_cancelled() {
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
            let cancellation = container_report
                .as_object_mut()
                .expect("current-container report is an object")
                .entry("cancellation".to_owned())
                .or_insert_with(|| json!({}));
            if let Some(cancellation) = cancellation.as_object_mut() {
                cancellation.insert("requested".to_owned(), Value::Bool(true));
            }
        }
        return crate::profile_pipeline::finish_profiled_replay(
            &container_report,
            &config.output_dir,
            Path::new("replay/replay_report.json"),
            adapter_report_fields(config, &trajectory, &controller)?,
            &[],
            false,
        )
        .map_err(|error| AtifProfileError::new(error.to_string()));
    }
    let (plan, trajectory) = build_container_plan(config)?;
    prepare_output_root(&config.output_dir)?;
    let profile = container_profile_config(config, &plan);
    let profile_run = crate::profile_container::run_profile(profile, cancel_requested)
        .map_err(|error| AtifProfileError::new(error.to_string()))?;
    let mut report_fields = adapter_report_fields(config, &trajectory, &plan.controller)?;
    if let Some(gate) = profile_run.report.get("run_wave_start_gate") {
        report_fields
            .as_object_mut()
            .expect("adapter report fields are an object")
            .insert("run_wave_start_gate".to_owned(), gate.clone());
    }
    crate::profile_pipeline::finish_profiled_replay(
        &profile_run.report,
        &config.output_dir,
        Path::new("replay/replay_report.json"),
        report_fields,
        &config.profile_sets,
        config.coordinated_baseline,
    )
    .map_err(|error| AtifProfileError::new(error.to_string()))
}

fn adapter_report_fields(
    config: &AtifProfileConfig,
    trajectory: &AtifTrajectory,
    controller: &RustControllerRuntime,
) -> Result<Value, AtifProfileError> {
    let output_root = fs::canonicalize(&config.output_dir).map_err(|error| {
        AtifProfileError::new(format!(
            "pipeline output directory is unavailable {}: {error}",
            config.output_dir.display()
        ))
    })?;
    let controller_identity = if config.runtime == "current" {
        json!({
            "kind": "rust-current-container-binary",
            "sha256": controller.sha256,
            "path": controller.executable,
        })
    } else {
        json!({
            "kind": "rust-host-mounted-binary",
            "sha256": controller.sha256,
            "container_path": RUST_CONTROLLER_EXECUTABLE,
        })
    };
    Ok(json!({
        "schema_version": PROFILE_SCHEMA_VERSION,
        "task_name": config.task_name.clone().unwrap_or_else(|| trajectory.task_name()),
        "source_trial": trajectory.trial_name(),
        "source_trajectory": fs::canonicalize(&config.trajectory)?.to_string_lossy(),
        "source_trajectory_sha256": sha256_file(&config.trajectory)?,
        "source_trajectory_schema_version": trajectory.schema_version,
        "source_agent_original_format": trajectory.source_agent_original_format,
        "agent": trajectory.agent,
        "image": config.image,
        "runtime": config.runtime,
        "pacing": config.pacing,
        "controller_runtime": config.controller_runtime,
        "controller_identity": controller_identity,
        "hide_prepared_python3": config.hide_prepared_python3,
        "patch_roots": config.patch_roots,
        "network": config.legacy_network,
        "action_output_capture": {
            "enabled": config.capture_action_outputs,
            "directory": if config.capture_action_outputs {
                Some(output_root.join("replay/action_outputs").to_string_lossy().into_owned())
            } else {
                None
            },
        },
    }))
}

fn cli_error(public_program: &str, error: impl fmt::Display) -> i32 {
    eprintln!("usage: {public_program} [-h] [--image IMAGE] [--runtime {{docker,enroot,current}}] --output-dir OUTPUT_DIR [options] trajectory");
    eprintln!("{public_program}: error: {error}");
    2
}

fn number(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

fn print_cpu_summary(cpu: &Value) {
    println!(
        "Wall time: {} sec",
        number(cpu.get("wall_sec"))
            .map(|value| format!("{value:.9}"))
            .unwrap_or_else(|| "n/a".to_owned())
    );
    if let Some(usage) = number(cpu.get("usage_sec")) {
        println!("CPU time: {usage:.9} sec");
        println!(
            "Average logical cores used: {}",
            number(cpu.get("average_logical_cores_used"))
                .map(|value| format!("{value:.6}"))
                .unwrap_or_else(|| "n/a".to_owned())
        );
    } else {
        println!("CPU time: n/a (cgroup cpu.stat unavailable)");
        println!("Average logical cores used: n/a (cgroup cpu.stat unavailable)");
    }
    if let (Some(percent), Some(capacity)) = (
        number(cpu.get("utilization_percent_of_allocation")),
        number(cpu.get("allocated_logical_cpus")),
    ) {
        let capacity = if capacity.fract() == 0.0 {
            format!("{capacity:.0}")
        } else {
            capacity.to_string()
        };
        println!("Allocated CPU utilization: {percent:.3}% of {capacity} logical CPUs");
    }
}

fn report_exit_code(report: &Value) -> i32 {
    let mismatches = report["replay_summary"]["source_outcome_mismatch_count"]
        .as_u64()
        .unwrap_or(0);
    let invalid_memory = report
        .get("container_memory")
        .filter(|value| !value.is_null())
        .is_some_and(|memory| {
            memory.get("valid_for_observed_memory_reporting") != Some(&Value::Bool(true))
        });
    i32::from(
        mismatches > 0
            || report.get("status").and_then(Value::as_str) != Some("valid")
            || invalid_memory
            || report.get("profiling_valid") == Some(&Value::Bool(false)),
    )
}

pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    if arguments
        .iter()
        .any(|argument| matches!(argument.to_str(), Some("-h" | "--help")))
    {
        help(public_program);
        return 0;
    }
    let config = match parse_arguments(arguments) {
        Ok(Some(config)) => config,
        Ok(None) => {
            help(public_program);
            return 0;
        }
        Err(error) => return cli_error(public_program, error),
    };
    if let Some(error) = config.native_slice_error() {
        return cli_error(public_program, error);
    }
    let guard = match crate::profile_container::ProfileSignalGuard::install() {
        Ok(guard) => guard,
        Err(error) => return cli_error(public_program, error),
    };
    let cancel_path = env::var_os(RUN_CANCEL_FILE_ENV).map(PathBuf::from);
    let callback = || cancel_path.as_ref().is_some_and(|path| path.exists());
    let result = run_atif_profile(
        &config,
        cancel_path.as_ref().map(|_| &callback as &dyn Fn() -> bool),
    );
    drop(guard);
    match result {
        Ok(report) => {
            let replay = &report["replay_summary"];
            println!("Task: {}", report["task_name"].as_str().unwrap_or(""));
            println!(
                "Source trial: {}",
                report["source_trial"].as_str().unwrap_or("")
            );
            println!(
                "Actions: {}/{} executed, {} timed out, {} source-outcome mismatches",
                replay["actions_executed"].as_u64().unwrap_or(0),
                replay["actions_recorded"].as_u64().unwrap_or(0),
                replay["actions_timed_out"].as_u64().unwrap_or(0),
                replay["source_outcome_mismatch_count"]
                    .as_u64()
                    .unwrap_or(0),
            );
            print_cpu_summary(&report["container_cpu"]);
            if let Some(valid) = report
                .pointer("/measurement_validity/valid_for_average_cpu_reporting")
                .and_then(Value::as_bool)
            {
                println!(
                    "Current-container CPU validity: {}",
                    if valid { "valid" } else { "invalid" }
                );
            }
            if let Some(peak) =
                report.pointer("/proc_stat_comparison/measurement/peak_active_logical_cpus")
            {
                println!("Sampled /proc active-CPU peak (comparison only): {}", peak);
            }
            if let Some(warning) = report.get("warning").and_then(Value::as_str) {
                println!("Warning: {warning}");
            }
            println!(
                "Report: {}",
                config.output_dir.join("pipeline_report.json").display()
            );
            if crate::profile_container::process_signal_cancelled() {
                130
            } else {
                report_exit_code(&report)
            }
        }
        Err(error) if crate::profile_container::process_signal_cancelled() => {
            eprintln!("{public_program}: interrupted: {error}");
            130
        }
        Err(error) => cli_error(public_program, error),
    }
}

pub fn help(program: &str) {
    println!("usage: {program} [-h] [--image IMAGE] [--runtime {{docker,enroot,current}}]");
    println!("       {program} --output-dir OUTPUT_DIR [options] trajectory");
    println!();
    println!("Profile supported actions from an ATIF trajectory in a measured task environment.");
    println!("Native profiles include explicit execve/process tracing, BCC/perf scheduler tracing, and every named profile set.");
    println!("Native Enroot uses --controller-runtime rust-host-mount and is wall-time-only.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn trajectory(root: &Path) -> PathBuf {
        let trial = root.join("task__hash/trial");
        fs::create_dir_all(&trial).unwrap();
        let path = trial.join("trajectory.json");
        fs::write(
            &path,
            serde_json::to_vec(&json!({
                "schema_version": "ATIF-v1.5",
                "session_id": "session",
                "agent": {"name": "codex"},
                "steps": [{
                    "step_id": 1,
                    "timestamp": "2026-05-24T01:00:00+00:00",
                    "tool_calls": [{
                        "function_name": "exec_command",
                        "arguments": {"cmd": "true"},
                    }],
                    "observation": {"content": "Process exited with code 0"},
                }],
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    fn arguments(path: &Path, output: &Path) -> Vec<OsString> {
        vec![
            path.as_os_str().to_owned(),
            "--image".into(),
            "sha256:image".into(),
            "--output-dir".into(),
            output.as_os_str().to_owned(),
            "--cpuset-cpus".into(),
            "16-19".into(),
            "--controller-runtime".into(),
            "rust-host-mount".into(),
            "--strace-mode".into(),
            "none".into(),
            "--profile-set".into(),
            "cpu-average".into(),
            "--network-policy".into(),
            "no-network".into(),
        ]
    }

    fn enroot_arguments(path: &Path, output: &Path) -> Vec<OsString> {
        let mut values = arguments(path, output);
        let profile_set = values
            .iter()
            .position(|value| value == "--profile-set")
            .unwrap();
        values.drain(profile_set..=profile_set + 1);
        values.extend(["--runtime".into(), "enroot".into()]);
        values
    }

    #[test]
    fn enroot_native_slice_accepts_typed_and_legacy_no_network_only() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");

        let typed = parse_arguments(&enroot_arguments(&path, &output))
            .unwrap()
            .unwrap();
        assert_eq!(typed.runtime, "enroot");
        assert_eq!(
            typed.network_policy.effective_mode(),
            crate::network_policy::NetworkMode::NoNetwork
        );
        assert!(typed.native_slice_error().is_none());

        let mut legacy_values = enroot_arguments(&path, &output);
        let policy = legacy_values
            .iter()
            .position(|value| value == "--network-policy")
            .unwrap();
        legacy_values[policy] = "--network".into();
        legacy_values[policy + 1] = "none".into();
        let legacy = parse_arguments(&legacy_values).unwrap().unwrap();
        assert_eq!(legacy.legacy_network.as_deref(), Some("none"));
        assert_eq!(
            legacy.network_policy.effective_mode(),
            crate::network_policy::NetworkMode::NoNetwork
        );
        assert!(legacy.native_slice_error().is_none());

        let mut allowlist_values = enroot_arguments(&path, &output);
        let policy = allowlist_values
            .iter()
            .position(|value| value == "--network-policy")
            .unwrap();
        allowlist_values[policy + 1] = "allowlist".into();
        allowlist_values.extend(["--allowed-host".into(), "example.com".into()]);
        let allowlist = parse_arguments(&allowlist_values).unwrap().unwrap();
        assert!(allowlist.native_slice_error().is_some());
    }

    #[test]
    fn run_wave_start_gate_is_hidden_typed_and_enroot_only() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");

        let mut gated = enroot_arguments(&path, &output);
        gated.push("--run-wave-start-gate".into());
        let gated = parse_arguments(&gated).unwrap().unwrap();
        assert!(gated.run_wave_start_gate);
        assert!(gated.native_slice_error().is_none());

        let mut docker = arguments(&path, &output);
        docker.push("--run-wave-start-gate".into());
        assert!(parse_arguments(&docker)
            .unwrap_err()
            .to_string()
            .contains("requires --runtime enroot"));

        let mut attached = enroot_arguments(&path, &output);
        attached.push("--run-wave-start-gate=true".into());
        assert!(parse_arguments(&attached)
            .unwrap_err()
            .to_string()
            .contains("does not take a value"));
    }

    #[test]
    fn parser_retains_native_run_contract_and_rejects_dropped_options() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");
        let config = parse_arguments(&arguments(&path, &output))
            .unwrap()
            .unwrap();
        assert_eq!(config.cpuset_cpus.as_deref(), Some("16-19"));
        assert_eq!(config.profile_sets, ["cpu-average"]);
        assert_eq!(
            config.network_policy.effective_mode(),
            crate::network_policy::NetworkMode::NoNetwork
        );
        assert_eq!(config.patch_roots, ["/tmp"]);

        let mut proc_stat = arguments(&path, &output);
        proc_stat.extend([
            "--proc-stat-comparison".into(),
            "--proc-stat-interval-sec".into(),
            "0.05".into(),
            "--proc-stat-baseline-sec".into(),
            "0.15".into(),
        ]);
        let config = parse_arguments(&proc_stat).unwrap().unwrap();
        assert!(config.native_slice_error().is_none());
        let plan = AtifContainerPlan {
            image: "sha256:image".to_owned(),
            command: "true".to_owned(),
            prepare_command: None,
            output_dir: output.clone(),
            mounts: Vec::new(),
            workload_identity: json!({}),
            source_identity: json!({}),
            expected_outcome: json!({}),
            controller: RustControllerRuntime {
                executable: PathBuf::from("tasktrace"),
                sha256: "controller".to_owned(),
            },
        };
        let profile = container_profile_config(&config, &plan);
        assert!(profile.proc_stat_comparison);
        assert_eq!(profile.proc_stat_interval_sec, 0.05);
        assert_eq!(profile.proc_stat_baseline_sec, 0.15);

        let mut memory = arguments(&path, &output);
        memory.extend([
            "--memory-profile".into(),
            "--memory-sample-interval-sec".into(),
            "0.02".into(),
            "--memory-limit-bytes".into(),
            "134217728".into(),
            "--memory-swap-total-limit-bytes".into(),
            "134217728".into(),
        ]);
        let config = parse_arguments(&memory).unwrap().unwrap();
        assert!(config.native_slice_error().is_none());
        let profile = container_profile_config(&config, &plan);
        assert!(profile.memory_profile);
        assert_eq!(profile.memory_sample_interval_sec, 0.02);
        assert_eq!(profile.memory_limit_bytes, Some(134_217_728));
        assert_eq!(profile.memory_swap_total_limit_bytes, Some(134_217_728));
    }

    #[test]
    fn process_modes_stay_native_with_the_expected_coordination_selection() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");
        let plan = AtifContainerPlan {
            image: "sha256:image".to_owned(),
            command: "true".to_owned(),
            prepare_command: None,
            output_dir: output.clone(),
            mounts: Vec::new(),
            workload_identity: json!({}),
            source_identity: json!({}),
            expected_outcome: json!({}),
            controller: RustControllerRuntime {
                executable: PathBuf::from("tasktrace"),
                sha256: "controller".to_owned(),
            },
        };

        for mode in ["execve", "process"] {
            let mut values = arguments(&path, &output);
            let profile_set = values
                .iter()
                .position(|value| value == "--profile-set")
                .unwrap();
            values.drain(profile_set..=profile_set + 1);
            let strace_mode = values
                .iter()
                .position(|value| value == "--strace-mode")
                .unwrap();
            values[strace_mode + 1] = mode.into();

            let config = parse_arguments(&values).unwrap().unwrap();
            assert!(config.native_slice_error().is_none(), "{mode}");
            assert_eq!(config.strace_mode, mode);
            assert!(config.profile_sets.is_empty());
            let profile = container_profile_config(&config, &plan);
            assert!(profile.profile_sets.is_empty());
            assert!(!profile.coordinated_baseline);
        }

        let mut values = arguments(&path, &output);
        let profile_set = values
            .iter()
            .position(|value| value == "--profile-set")
            .unwrap();
        values[profile_set + 1] = "software-deep".into();
        let config = parse_arguments(&values).unwrap().unwrap();
        assert!(config.native_slice_error().is_none());
        assert_eq!(config.profile_sets, ["software-deep"]);
        let profile = container_profile_config(&config, &plan);
        assert_eq!(profile.profile_sets, ["software-deep"]);
        assert!(!profile.coordinated_baseline);

        let mut values = arguments(&path, &output);
        let profile_set = values
            .iter()
            .position(|value| value == "--profile-set")
            .unwrap();
        values[profile_set + 1] = "cpu-peak".into();
        values.extend([
            "--scheduler-backend".into(),
            "bcc".into(),
            "--scheduler-bcc-buffer-pages".into(),
            "64".into(),
        ]);
        let config = parse_arguments(&values).unwrap().unwrap();
        assert!(config.native_slice_error().is_none());
        assert_eq!(config.scheduler_backend, "bcc");
        assert_eq!(config.scheduler_bcc_buffer_pages, Some(64));
        let profile = container_profile_config(&config, &plan);
        assert_eq!(profile.profile_sets, ["cpu-peak"]);
        assert_eq!(profile.scheduler_backend, "bcc");
        assert_eq!(profile.scheduler_bcc_buffer_pages, Some(64));
    }

    #[test]
    fn current_runtime_is_native_and_rejects_docker_only_extensions() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");
        let current = vec![
            path.as_os_str().to_owned(),
            "--runtime".into(),
            "current".into(),
            "--output-dir".into(),
            output.as_os_str().to_owned(),
            "--workdir".into(),
            root.path().as_os_str().to_owned(),
            "--current-container-exclusive".into(),
        ];
        let config = parse_arguments(&current).unwrap().unwrap();
        assert_eq!(config.runtime, "current");
        assert_eq!(config.controller_runtime, "image");
        assert!(config.current_container_exclusive);
        assert!(config.native_slice_error().is_none());
        let controller = RustControllerRuntime {
            executable: env::current_exe().unwrap(),
            sha256: "a".repeat(64),
        };
        let request = current_profile_config(&config, &controller).unwrap();
        assert!(request.command.contains(" replay atif "));
        assert!(request.command.contains("replay/replay_report.json"));
        assert!(!request.command.contains("python"));
        assert_eq!(request.output_dir, output.join("container_profile"));
        assert!(request.exclusive);

        let mut unsupported = current.clone();
        unsupported.extend(["--profile-set".into(), "cpu-average".into()]);
        assert!(parse_arguments(&unsupported)
            .unwrap_err()
            .to_string()
            .contains("not supported by current-container"));
        let mut network = current;
        network.extend(["--network-policy".into(), "no-network".into()]);
        assert!(parse_arguments(&network)
            .unwrap_err()
            .to_string()
            .contains("network policy controls"));
    }

    #[test]
    fn plan_content_binds_controller_and_never_invokes_python() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");
        fs::create_dir(&output).unwrap();
        fs::create_dir(output.join("replay")).unwrap();
        let controller = root.path().join("tasktrace");
        fs::write(&controller, b"native-controller").unwrap();
        let mut permissions = fs::metadata(&controller).unwrap().permissions();
        use std::os::unix::fs::PermissionsExt;
        permissions.set_mode(0o755);
        fs::set_permissions(&controller, permissions).unwrap();
        let config = parse_arguments(&arguments(&path, &output))
            .unwrap()
            .unwrap();
        let expected_controller = RustControllerRuntime {
            sha256: sha256_file(&controller).unwrap(),
            executable: controller.clone(),
        };
        let (plan, parsed) =
            build_container_plan_with_controller(&config, expected_controller).unwrap();

        assert_eq!(parsed.actions.len(), 1);
        assert!(plan
            .command
            .contains("/tasktrace-controller/tasktrace replay atif"));
        assert!(!plan.command.contains("python"));
        assert_eq!(plan.controller.sha256, sha256_file(&controller).unwrap());
        assert_eq!(
            plan.source_identity["rust_controller_binary_sha256"],
            plan.controller.sha256
        );
        assert!(plan
            .mounts
            .iter()
            .any(|mount| { mount.contains("dst=/tasktrace-controller/tasktrace,readonly") }));
    }

    #[test]
    fn output_root_rejects_symlinks_and_nonempty_directories() {
        let root = TempDir::new().unwrap();
        let nonempty = root.path().join("nonempty");
        fs::create_dir(&nonempty).unwrap();
        fs::write(nonempty.join("existing"), b"x").unwrap();
        assert!(validate_output_root(&nonempty)
            .unwrap_err()
            .to_string()
            .contains("new or empty"));

        let target = root.path().join("target");
        fs::create_dir(&target).unwrap();
        let link = root.path().join("link");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        assert!(validate_output_root(&link)
            .unwrap_err()
            .to_string()
            .contains("not a directory"));
    }

    #[test]
    fn nested_output_preflight_is_read_only_and_prepare_claims_final_directory() {
        let root = TempDir::new().unwrap();
        let output = root.path().join("waves/0001/launch/profile");
        validate_output_root(&output).unwrap();
        assert!(!root.path().join("waves").exists());

        let prepared = prepare_output_root(&output).unwrap();
        assert_eq!(prepared, output.canonicalize().unwrap());
        assert!(output.join("replay").is_dir());
        assert!(validate_output_root(&output).is_err());
    }

    #[test]
    fn pipeline_composition_invalidates_replay_divergence() {
        let root = TempDir::new().unwrap();
        let path = trajectory(root.path());
        let output = root.path().join("out");
        fs::create_dir(&output).unwrap();
        fs::create_dir(output.join("replay")).unwrap();
        fs::write(
            output.join("replay/replay_report.json"),
            serde_json::to_vec(&json!({
                "summary": {
                    "actions_executed": 1,
                    "actions_recorded": 1,
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 1,
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let config = parse_arguments(&arguments(&path, &output))
            .unwrap()
            .unwrap();
        let parsed = read_trajectory(&path).unwrap();
        let controller = RustControllerRuntime {
            executable: root.path().join("controller"),
            sha256: "a".repeat(64),
        };
        let report = compose_pipeline_report(
            &config,
            &parsed,
            &controller,
            &json!({
                "image_identity": {"id": "sha256:image"},
                "cpu": {"usage_sec": 1.0},
                "strace": {"mode": "none"},
                "profiling_valid": true,
                "evidence_bundle": {
                    "execution_plan_digest": "digest",
                    "provenance": {"execution_plan": {"source_identity": {"format": "atif"}}},
                },
            }),
        )
        .unwrap();
        assert_eq!(report["status"], "invalid");
        assert_eq!(report["replay_summary"]["source_outcome_mismatch_count"], 1);
        assert_eq!(
            report["execution_plan"]["source_identity"]["format"],
            "atif"
        );
    }
    #[test]
    fn invalid_replay_pipeline_cannot_exit_success_with_zero_mismatches() {
        let mut report = json!({
            "status": "invalid", "profiling_valid": true,
            "replay_summary": {"source_outcome_mismatch_count": 0},
        });
        assert_eq!(report_exit_code(&report), 1);
        report["status"] = json!("valid");
        assert_eq!(report_exit_code(&report), 0);
    }
}
