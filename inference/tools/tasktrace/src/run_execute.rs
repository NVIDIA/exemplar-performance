// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Execute strict, sealed offline-replay run manifests.
//!
//! The executor deliberately treats replay correctness, measurement validity,
//! and batch/wave analysis validity as separate gates.  Rows are retained even
//! when one of those gates fails; only unambiguous startup failures prevent an
//! output directory from being claimed.

use crate::enroot_provenance::{
    capture_runtime_provenance, recapture_and_validate, EnrootProvenanceConfig,
};
use crate::network_policy::NetworkPolicy;
use crate::run_host::collect_host_cpu_identity;
use crate::run_spec::{
    load_run_spec, MetadataScalar, RunLaunchSpec, RunSpec, RunSpecV1, RunSpecV2, RunSpecV3,
    RunTaskSpec, RunWaveSpec, SCHEMA_VERSION_V3,
};
use crate::timebase::{clock_reference, monotonic_ns, monotonic_resolution_ns};
use serde_json::{json, Map, Number, Value};
use serde_yaml::{Mapping as YamlMapping, Value as YamlValue};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
#[cfg(unix)]
use std::os::unix::fs::{MetadataExt, OpenOptionsExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

pub const SUMMARY_SCHEMA_VERSION_V1: &str = "tasktrace-run-summary.v1";
pub const SUMMARY_SCHEMA_VERSION_V2: &str = "tasktrace-run-summary.v2";
pub const SUMMARY_SCHEMA_VERSION_V3: &str = "tasktrace-run-summary.v3";
#[allow(dead_code)] // Backward-compatible alias for callers that only support v1.
pub const SUMMARY_SCHEMA_VERSION: &str = SUMMARY_SCHEMA_VERSION_V1;

pub const RESULT_COLUMNS: &[&str] = &[
    "task_id",
    "repetition",
    "status",
    "duration_sec",
    "duration_valid",
    "reason",
    "profile_dir",
    "source_sha256",
    "image_id",
    "execution_plan_digest",
];

pub const RESULT_COLUMNS_V2: &[&str] = &[
    "attempt_index",
    "completion_index",
    "task_id",
    "repetition",
    "status",
    "duration_sec",
    "duration_valid",
    "reason",
    "profile_dir",
    "source_sha256",
    "image_id",
    "execution_plan_digest",
    "window_start_ns",
    "window_end_ns",
    "cpu_usage_sec",
    "average_logical_cores",
];

pub const RESULT_COLUMNS_V3: &[&str] = &[
    "attempt_index",
    "completion_index",
    "wave_index",
    "wave_id",
    "wave_scored",
    "launch_index",
    "cpus",
    "task_id",
    "repetition",
    "status",
    "duration_sec",
    "duration_valid",
    "reason",
    "profile_dir",
    "source_sha256",
    "image_id",
    "execution_plan_digest",
    "window_start_ns",
    "window_end_ns",
    "cpu_usage_sec",
    "average_logical_cores",
];

const ENROOT_RESULT_COLUMNS: &[&str] = &["cleanup_valid", "clock_reference"];

const CPU_CAPACITY_COUNTER_TOLERANCE_SEC_PER_ATTEMPT: f64 = 1e-6;
const CPU_CAPACITY_FLOAT_TOLERANCE_FRACTION: f64 = 1e-9;
const ENROOT_TASK_CLOCK_SOURCE: &str = "linux-proc-uptime";
const ENROOT_OUTER_WALL_TOLERANCE_SEC: f64 = 0.05;
const ENROOT_STATE_ENVIRONMENT: &[&str] = &[
    "ENROOT_CACHE_PATH",
    "ENROOT_DATA_PATH",
    "ENROOT_RUNTIME_PATH",
];
const ENROOT_FORBIDDEN_ENVIRONMENT: &[&str] = &["BASH_ENV", "PYTHONHOME"];
const RUN_CANCEL_FILE_ENV: &str = "TASKTRACE_RUN_CANCEL_FILE";
const RUN_WAVE_START_READY: &str = "run_wave_start_ready";
const RUN_WAVE_START_RELEASE: &str = "run_wave_start_release";
const RUN_WAVE_START_READY_BYTES: &[u8] = b"ready\n";
const RUN_WAVE_START_RELEASE_BYTES: &[u8] = b"release\n";
const RUN_WAVE_START_POLL: Duration = Duration::from_millis(20);
const RUN_WAVE_START_TIMEOUT: Duration = Duration::from_secs(900);
static CANCEL_FILE_SEQUENCE: AtomicUsize = AtomicUsize::new(0);

struct CancelMarker {
    directory: PathBuf,
    path: PathBuf,
    stdout_path: PathBuf,
    stderr_path: PathBuf,
}

impl CancelMarker {
    fn new() -> Result<Self, RunExecutionError> {
        let mut random = [0_u8; 16];
        File::open("/dev/urandom")
            .and_then(|mut source| source.read_exact(&mut random))
            .map_err(|error| {
                RunExecutionError::new(format!(
                    "unable to acquire cancellation-marker randomness: {error}"
                ))
            })?;
        let mut token = String::with_capacity(random.len() * 2);
        for byte in random {
            write!(&mut token, "{byte:02x}").expect("writing to a String cannot fail");
        }
        let directory = env::temp_dir().join(format!(
            "tasktrace-run-cancel-{}-{}-{token}",
            std::process::id(),
            CANCEL_FILE_SEQUENCE.fetch_add(1, Ordering::SeqCst)
        ));
        let mut builder = fs::DirBuilder::new();
        #[cfg(unix)]
        {
            use std::os::unix::fs::DirBuilderExt;
            builder.mode(0o700);
        }
        builder.create(&directory).map_err(|error| {
            RunExecutionError::new(format!(
                "unable to create private cancellation directory {}: {error}",
                directory.display()
            ))
        })?;
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let permissions = fs::metadata(&directory)
                .map(|metadata| metadata.permissions().mode())
                .map_err(|error| {
                    let _ = fs::remove_dir(&directory);
                    RunExecutionError::new(format!(
                        "unable to verify cancellation directory {}: {error}",
                        directory.display()
                    ))
                })?;
            if permissions & 0o077 != 0 {
                let _ = fs::remove_dir(&directory);
                return Err(RunExecutionError::new(format!(
                    "cancellation directory permissions are not private: {}",
                    directory.display()
                )));
            }
        }
        let path = directory.join("requested");
        let stdout_path = directory.join("profile.stdout");
        let stderr_path = directory.join("profile.stderr");
        Ok(Self {
            directory,
            path,
            stdout_path,
            stderr_path,
        })
    }

    fn publish(&self) -> Result<(), RunExecutionError> {
        match OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&self.path)
            .and_then(|mut file| file.write_all(b"cancel\n"))
        {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => Ok(()),
            Err(error) => Err(RunExecutionError::new(format!(
                "unable to publish cooperative profile cancellation {}: {error}",
                self.path.display()
            ))),
        }
    }
}

impl Drop for CancelMarker {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.directory);
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunExecutionError(String);

impl RunExecutionError {
    pub fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for RunExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RunExecutionError {}

impl From<io::Error> for RunExecutionError {
    fn from(error: io::Error) -> Self {
        Self::new(error.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct RunExecutionResult {
    pub exit_code: i32,
    pub rows: Vec<Map<String, Value>>,
    pub summary: Value,
}

/// Cooperative cancellation shared by dispatch and the active profile leaf.
#[derive(Clone, Debug)]
pub struct CancellationToken {
    requested: Arc<AtomicBool>,
    observe_process_signal: bool,
}

impl Default for CancellationToken {
    fn default() -> Self {
        Self::new()
    }
}

impl CancellationToken {
    pub fn new() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            observe_process_signal: false,
        }
    }

    fn observing_process_signal() -> Self {
        Self {
            requested: Arc::new(AtomicBool::new(false)),
            observe_process_signal: true,
        }
    }

    #[allow(dead_code)] // Used by injectable launchers and cancellation-focused tests.
    pub fn cancel(&self) {
        self.requested.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.requested.load(Ordering::SeqCst)
            || (self.observe_process_signal && SIGNAL_CANCELLED.load(Ordering::SeqCst))
    }
}

/// Complete public-profile configuration for one preflight or attempt.
#[derive(Clone, Debug, PartialEq)]
pub struct ProfileRequest {
    pub task_id: String,
    pub trajectory: PathBuf,
    pub image: String,
    pub output_dir: PathBuf,
    pub cpuset_cpus: String,
    pub pacing: String,
    pub timeout_sec: Option<f64>,
    pub runtime: String,
    pub enroot_executable: Option<PathBuf>,
    pub controller_runtime: String,
    pub profile_sets: Vec<String>,
    pub network_policy: Option<String>,
    pub allowed_hosts: Vec<String>,
    /// Optional in-container working directory; when absent the profile
    /// command's historical `/app` default applies.
    pub workdir: Option<String>,
    /// NAME=VALUE assignments pinned inside the measured container.
    pub environment: Vec<String>,
}

/// Injectable boundary around profile preflight and actual public CLI launch.
pub trait AttemptLauncher: Send + Sync {
    /// Whether this launcher implements the private Enroot wave-start gate
    /// carried by the public profile command.
    fn enroot_wave_start_gate_enabled(&self) -> bool {
        false
    }

    fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError>;

    fn preflight_with_cancellation(
        &self,
        request: &ProfileRequest,
        cancellation: &CancellationToken,
    ) -> Result<Value, RunExecutionError> {
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new(
                "run interrupted before profile preflight",
            ));
        }
        self.preflight(request)
    }

    fn run_profile(
        &self,
        request: &ProfileRequest,
        cancellation: &CancellationToken,
    ) -> Result<Value, RunExecutionError>;
}

/// Launcher used by the product CLI.
///
/// Attempts always traverse the current executable's public `profile atif`
/// command. Preflight calls the same native parser and planning API directly,
/// so run execution never depends on an installed Python package.
#[derive(Clone, Debug)]
pub struct PublicProfileLauncher {
    executable: PathBuf,
    cancellation_grace: Duration,
}

impl PublicProfileLauncher {
    pub fn new(executable: PathBuf) -> Self {
        Self {
            executable,
            cancellation_grace: Duration::from_secs(30),
        }
    }

    fn profile_arguments(request: &ProfileRequest) -> Result<Vec<OsString>, RunExecutionError> {
        let mut arguments = vec![
            OsString::from("profile"),
            OsString::from("atif"),
            request.trajectory.as_os_str().to_owned(),
            OsString::from("--image"),
            OsString::from(&request.image),
            OsString::from("--output-dir"),
            request.output_dir.as_os_str().to_owned(),
            OsString::from("--task-name"),
            OsString::from(&request.task_id),
            OsString::from("--cpuset-cpus"),
            OsString::from(&request.cpuset_cpus),
            OsString::from("--pacing"),
            OsString::from(&request.pacing),
            OsString::from("--runtime"),
            OsString::from(&request.runtime),
            OsString::from("--controller-runtime"),
            OsString::from(&request.controller_runtime),
            OsString::from("--strace-mode"),
            OsString::from("none"),
        ];
        if let Some(workdir) = &request.workdir {
            arguments.push(OsString::from("--workdir"));
            arguments.push(OsString::from(workdir));
        }
        for assignment in &request.environment {
            arguments.push(OsString::from("--env"));
            arguments.push(OsString::from(assignment));
        }
        if request.runtime == "enroot" {
            let executable = request.enroot_executable.as_ref().ok_or_else(|| {
                RunExecutionError::new(
                    "Enroot profile request is missing its captured runtime executable",
                )
            })?;
            if !executable.is_absolute() {
                return Err(RunExecutionError::new(
                    "captured Enroot runtime executable is not absolute",
                ));
            }
            arguments.push(OsString::from("--enroot"));
            arguments.push(executable.as_os_str().to_owned());
            arguments.push(OsString::from("--run-wave-start-gate"));
        }
        if let Some(timeout) = request.timeout_sec {
            arguments.push(OsString::from("--timeout-sec"));
            arguments.push(OsString::from(timeout.to_string()));
        }
        if let Some(policy) = &request.network_policy {
            arguments.push(OsString::from("--network-policy"));
            arguments.push(OsString::from(policy));
            for host in &request.allowed_hosts {
                arguments.push(OsString::from("--allowed-host"));
                arguments.push(OsString::from(host));
            }
        }
        for profile_set in &request.profile_sets {
            arguments.push(OsString::from("--profile-set"));
            arguments.push(OsString::from(profile_set));
        }
        Ok(arguments)
    }

    fn run_preflight(
        &self,
        request: &ProfileRequest,
        cancellation: &CancellationToken,
    ) -> Result<Value, RunExecutionError> {
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new(
                "run interrupted before native profile preflight",
            ));
        }
        let arguments = Self::profile_arguments(request)?;
        let config = crate::atif_profile::parse_arguments(&arguments[2..])
            .map_err(|error| {
                RunExecutionError::new(format!(
                    "native profile preflight arguments are invalid: {error}"
                ))
            })?
            .ok_or_else(|| {
                RunExecutionError::new("native profile preflight unexpectedly requested help")
            })?;
        let report = crate::atif_profile::preflight_atif_profile(&config).map_err(|error| {
            RunExecutionError::new(format!("native profile preflight failed: {error}"))
        })?;
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new(
                "run interrupted during native profile preflight",
            ));
        }
        Ok(report)
    }
}

impl Default for PublicProfileLauncher {
    fn default() -> Self {
        Self::new(env::current_exe().unwrap_or_else(|_| PathBuf::from("tasktrace")))
    }
}

impl AttemptLauncher for PublicProfileLauncher {
    fn enroot_wave_start_gate_enabled(&self) -> bool {
        true
    }

    fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
        self.run_preflight(request, &CancellationToken::new())
    }

    fn preflight_with_cancellation(
        &self,
        request: &ProfileRequest,
        cancellation: &CancellationToken,
    ) -> Result<Value, RunExecutionError> {
        self.run_preflight(request, cancellation)
    }

    fn run_profile(
        &self,
        request: &ProfileRequest,
        cancellation: &CancellationToken,
    ) -> Result<Value, RunExecutionError> {
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new(
                "run interrupted before public profile command spawn",
            ));
        }
        let cancel_marker = CancelMarker::new()?;
        let stdout_sink = File::create(&cancel_marker.stdout_path).map_err(|error| {
            RunExecutionError::new(format!("unable to create profile stdout sink: {error}"))
        })?;
        let stderr_sink = File::create(&cancel_marker.stderr_path).map_err(|error| {
            RunExecutionError::new(format!("unable to create profile stderr sink: {error}"))
        })?;
        let mut command = Command::new(&self.executable);
        command
            .args(Self::profile_arguments(request)?)
            .stdin(Stdio::null())
            .stdout(Stdio::from(stdout_sink))
            .stderr(Stdio::from(stderr_sink))
            .env(RUN_CANCEL_FILE_ENV, &cancel_marker.path);
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            command.process_group(0);
        }
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new(
                "run interrupted before public profile command spawn",
            ));
        }
        let mut child = command.spawn().map_err(|error| {
            RunExecutionError::new(format!("unable to start public profile command: {error}"))
        })?;
        let process_group = child.id() as i32;
        let status = match wait_for_profile(
            &mut child,
            process_group,
            cancellation,
            &cancel_marker,
            self.cancellation_grace,
        ) {
            Ok(status) => status,
            Err(error) => {
                let _ = send_process_group_signal(process_group, libc::SIGKILL);
                let _ = child.wait();
                let process_cleanup = cleanup_process_group(process_group);
                let runtime_cleanup = cleanup_profile_runtime(request);
                if process_cleanup.is_err() || runtime_cleanup.is_err() {
                    return Err(RunExecutionError::new(format!(
                        "{error}; profile descendant cleanup: {}; scoped runtime cleanup: {}",
                        process_cleanup
                            .err()
                            .unwrap_or_else(|| "succeeded".to_owned()),
                        runtime_cleanup
                            .err()
                            .map(|error| error.to_string())
                            .unwrap_or_else(|| "succeeded".to_owned())
                    )));
                }
                let stdout = fs::read(&cancel_marker.stdout_path).unwrap_or_default();
                let stderr = fs::read(&cancel_marker.stderr_path).unwrap_or_default();
                return Err(RunExecutionError::new(format!(
                    "{error}: {}",
                    diagnostic_output(&stdout, &stderr)
                )));
            }
        };
        // Clean the group before joining pipe readers. A leaked descendant can
        // otherwise retain a pipe descriptor and deadlock the runner forever.
        let process_cleanup = cleanup_process_group(process_group);
        let runtime_cleanup = cleanup_profile_runtime(request);
        if process_cleanup.is_err() || runtime_cleanup.is_err() {
            return Err(RunExecutionError::new(format!(
                "profile descendant cleanup: {}; scoped runtime cleanup: {}",
                process_cleanup
                    .err()
                    .unwrap_or_else(|| "succeeded".to_owned()),
                runtime_cleanup
                    .err()
                    .map(|error| error.to_string())
                    .unwrap_or_else(|| "succeeded".to_owned())
            )));
        }
        let stdout = fs::read(&cancel_marker.stdout_path).unwrap_or_default();
        let stderr = fs::read(&cancel_marker.stderr_path).unwrap_or_default();

        let report_path = request.output_dir.join("pipeline_report.json");
        let raw = match fs::read(&report_path) {
            Ok(raw) => raw,
            Err(error) => {
                return Err(RunExecutionError::new(format!(
                    "public profile command exited with {} without a readable report at {}: {error}: {}",
                    display_status(status),
                    report_path.display(),
                    diagnostic_output(&stdout, &stderr)
                )))
            }
        };
        let report = serde_json::from_slice(&raw).map_err(|error| {
            RunExecutionError::new(format!(
                "profile report {} is invalid JSON: {error}",
                report_path.display()
            ))
        })?;
        if !matches!(status.code(), Some(0 | 1)) {
            return Err(RunExecutionError::new(format!(
                "public profile command produced a report but exited with {}: {}",
                display_status(status),
                diagnostic_output(&stdout, &stderr)
            )));
        }
        Ok(report)
    }
}

fn diagnostic_output(stdout: &[u8], stderr: &[u8]) -> String {
    let mut parts = Vec::new();
    let out = String::from_utf8_lossy(stdout);
    let err = String::from_utf8_lossy(stderr);
    if !out.trim().is_empty() {
        parts.push(format!("stdout={:?}", truncate(out.trim(), 4000)));
    }
    if !err.trim().is_empty() {
        parts.push(format!("stderr={:?}", truncate(err.trim(), 4000)));
    }
    if parts.is_empty() {
        "no diagnostic output".to_owned()
    } else {
        parts.join(", ")
    }
}

fn truncate(value: &str, limit: usize) -> &str {
    if value.len() <= limit {
        value
    } else {
        let mut boundary = limit;
        while !value.is_char_boundary(boundary) {
            boundary -= 1;
        }
        &value[..boundary]
    }
}

fn display_status(status: ExitStatus) -> String {
    status
        .code()
        .map(|code| format!("status {code}"))
        .unwrap_or_else(|| "termination by signal".to_owned())
}

fn send_process_group_signal(process_group: i32, signal: i32) -> io::Result<()> {
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(-process_group, signal) };
        if result == 0 || io::Error::last_os_error().raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(io::Error::last_os_error())
        }
    }
    #[cfg(not(unix))]
    {
        let _ = (process_group, signal);
        Ok(())
    }
}

fn process_group_exists(process_group: i32) -> io::Result<bool> {
    #[cfg(unix)]
    {
        let result = unsafe { libc::kill(-process_group, 0) };
        if result == 0 {
            return Ok(true);
        }
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(false)
        } else if error.raw_os_error() == Some(libc::EPERM) {
            Ok(true)
        } else {
            Err(error)
        }
    }
    #[cfg(not(unix))]
    {
        let _ = process_group;
        Ok(false)
    }
}

fn wait_for_profile(
    child: &mut Child,
    _process_group: i32,
    cancellation: &CancellationToken,
    cancel_marker: &CancelMarker,
    cancellation_grace: Duration,
) -> Result<ExitStatus, RunExecutionError> {
    let mut cancellation_started = None;
    loop {
        if let Some(status) = child.try_wait().map_err(|error| {
            RunExecutionError::new(format!(
                "unable to wait for public profile command: {error}"
            ))
        })? {
            return Ok(status);
        }
        if cancellation.is_cancelled() {
            if cancellation_started.is_none() {
                cancel_marker.publish()?;
                cancellation_started = Some(Instant::now());
            }
            if cancellation_started.is_some_and(|started| started.elapsed() > cancellation_grace) {
                return Err(RunExecutionError::new(
                    format!(
                        "public profile command did not complete cleanup within {:.3} seconds of cancellation",
                        cancellation_grace.as_secs_f64()
                    ),
                ));
            }
        }
        thread::sleep(Duration::from_millis(20));
    }
}

fn cleanup_process_group(process_group: i32) -> Result<(), String> {
    if !process_group_exists(process_group).map_err(|error| error.to_string())? {
        return Ok(());
    }
    send_process_group_signal(process_group, libc::SIGTERM).map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_millis(500);
    while Instant::now() < deadline {
        if !process_group_exists(process_group).map_err(|error| error.to_string())? {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    send_process_group_signal(process_group, libc::SIGKILL).map_err(|error| error.to_string())?;
    let deadline = Instant::now() + Duration::from_millis(500);
    while Instant::now() < deadline {
        if !process_group_exists(process_group).map_err(|error| error.to_string())? {
            return Ok(());
        }
        thread::sleep(Duration::from_millis(10));
    }
    Err(format!(
        "process group {process_group} still has observable descendants after SIGKILL"
    ))
}

#[derive(Default)]
struct ScopedDockerResources {
    workload_containers: BTreeSet<String>,
    proxy_containers: BTreeSet<String>,
    networks: BTreeSet<String>,
    state_paths: BTreeSet<PathBuf>,
}

impl ScopedDockerResources {
    fn is_empty(&self) -> bool {
        self.workload_containers.is_empty()
            && self.proxy_containers.is_empty()
            && self.networks.is_empty()
            && self.state_paths.is_empty()
    }
}

fn docker_output(arguments: &[&str]) -> Result<std::process::Output, RunExecutionError> {
    Command::new("docker")
        .args(arguments)
        .output()
        .map_err(|error| {
            RunExecutionError::new(format!(
                "unable to execute Docker runtime cleanup command {:?}: {error}",
                arguments
            ))
        })
}

fn checked_docker_output(arguments: &[&str]) -> Result<std::process::Output, RunExecutionError> {
    let output = docker_output(arguments)?;
    if output.status.success() {
        Ok(output)
    } else {
        Err(RunExecutionError::new(format!(
            "Docker runtime cleanup command {:?} exited with {}: {}",
            arguments,
            display_status(output.status),
            diagnostic_output(&output.stdout, &output.stderr)
        )))
    }
}

fn docker_object_absent(kind: &str, output: &std::process::Output) -> bool {
    if output.status.success() {
        return false;
    }
    let stdout = String::from_utf8_lossy(&output.stdout);
    if !matches!(stdout.trim(), "" | "[]") {
        return false;
    }
    let diagnostic = String::from_utf8_lossy(&output.stderr)
        .trim()
        .to_ascii_lowercase();
    if diagnostic.bytes().any(|byte| matches!(byte, b'\r' | b'\n')) {
        return false;
    }
    let valid_reference = |reference: &str| {
        !reference.is_empty()
            && reference.bytes().all(|byte| {
                byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b':' | b'/')
            })
    };
    let has_reference = |prefixes: &[&str]| {
        prefixes
            .iter()
            .any(|prefix| diagnostic.strip_prefix(prefix).is_some_and(valid_reference))
    };
    match kind {
        "container" => has_reference(&[
            "error: no such container: ",
            "error: no such object: ",
            "error response from daemon: no such container: ",
            "error response from daemon: no such object: ",
        ]),
        "network" => {
            has_reference(&[
                "error: no such network: ",
                "error: no such object: ",
                "error response from daemon: no such network: ",
                "error response from daemon: no such object: ",
            ]) || diagnostic
                .strip_prefix("error response from daemon: network ")
                .and_then(|value| value.strip_suffix(" not found"))
                .is_some_and(valid_reference)
        }
        _ => false,
    }
}

fn add_policy_state_paths(resources: &mut ScopedDockerResources, network: &str) {
    let Some(token) = network.strip_prefix("tasktrace-policy-") else {
        return;
    };
    if !(8..=16).contains(&token.len())
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
    {
        return;
    }
    let temporary = env::temp_dir();
    resources
        .state_paths
        .insert(temporary.join(format!(".tasktrace-network-policy-state-{token}")));
    resources
        .state_paths
        .insert(temporary.join(format!(".tasktrace-network-decisions-{token}.tmp")));
}

fn add_network_enforcement_resources(output_dir: &Path, resources: &mut ScopedDockerResources) {
    let path = output_dir.join("network_enforcement.json");
    let Ok(raw) = fs::read(path) else {
        return;
    };
    let Ok(report) = serde_json::from_slice::<Value>(&raw) else {
        return;
    };
    let enforcement = report
        .get("enforcement")
        .and_then(Value::as_object)
        .or_else(|| {
            (report.get("schema_version").and_then(Value::as_str)
                == Some("tasktrace-docker-network-enforcement.v1"))
            .then(|| report.as_object())
            .flatten()
        });
    let Some(enforcement) = enforcement else {
        return;
    };
    if let Some(network) = enforcement
        .get("ephemeral_internal_network")
        .and_then(Value::as_str)
        .filter(|name| name.starts_with("tasktrace-policy-"))
    {
        resources.networks.insert(network.to_owned());
        add_policy_state_paths(resources, network);
    }
    if let Some(proxy) = enforcement
        .get("proxy")
        .and_then(Value::as_object)
        .and_then(|proxy| proxy.get("container_name"))
        .and_then(Value::as_str)
        .filter(|name| name.starts_with("tasktrace-policy-proxy-"))
    {
        resources.proxy_containers.insert(proxy.to_owned());
    }
}

fn discover_scoped_docker_resources(
    output_dir: &Path,
) -> Result<ScopedDockerResources, RunExecutionError> {
    let canonical_output = fs::canonicalize(output_dir).map_err(|error| {
        RunExecutionError::new(format!(
            "unable to resolve profile output scope {} during Docker cleanup: {error}",
            output_dir.display()
        ))
    })?;
    let mut resources = ScopedDockerResources::default();
    add_network_enforcement_resources(output_dir, &mut resources);
    let listed = checked_docker_output(&[
        "ps",
        "--all",
        "--filter",
        "name=tasktrace-profile-",
        "--format",
        "{{.ID}}",
    ])?;
    for container in String::from_utf8_lossy(&listed.stdout)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
    {
        let inspected = docker_output(&["inspect", container])?;
        if !inspected.status.success() {
            if docker_object_absent("container", &inspected) {
                // Another bounded-concurrency worker may have completed and
                // removed its unrelated candidate after the shared listing.
                continue;
            }
            return Err(RunExecutionError::new(format!(
                "Docker runtime cleanup could not inspect candidate container {container}: {}",
                diagnostic_output(&inspected.stdout, &inspected.stderr)
            )));
        }
        let value: Value = serde_json::from_slice(&inspected.stdout).map_err(|error| {
            RunExecutionError::new(format!(
                "Docker inspect returned invalid JSON for container {container}: {error}"
            ))
        })?;
        let Some(inspect) = value.as_array().and_then(|values| values.first()) else {
            return Err(RunExecutionError::new(format!(
                "Docker inspect returned no object for container {container}"
            )));
        };
        let scoped = inspect
            .get("Mounts")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|mount| {
                mount.get("Destination").and_then(Value::as_str)
                    == Some("/tasktrace-profile-output")
            })
            .filter_map(|mount| mount.get("Source").and_then(Value::as_str))
            .filter_map(|source| fs::canonicalize(source).ok())
            .any(|source| source == canonical_output || source.starts_with(&canonical_output));
        if !scoped {
            continue;
        }
        resources.workload_containers.insert(container.to_owned());
        if let Some(networks) = inspect
            .pointer("/NetworkSettings/Networks")
            .and_then(Value::as_object)
        {
            for network in networks
                .keys()
                .filter(|name| name.starts_with("tasktrace-policy-"))
            {
                resources.networks.insert(network.clone());
                add_policy_state_paths(&mut resources, network);
            }
        }
    }
    for network in resources.networks.clone() {
        let listed = checked_docker_output(&[
            "ps",
            "--all",
            "--filter",
            &format!("network={network}"),
            "--format",
            "{{.Names}}",
        ])?;
        for name in String::from_utf8_lossy(&listed.stdout)
            .lines()
            .map(str::trim)
            .filter(|name| name.starts_with("tasktrace-policy-proxy-"))
        {
            resources.proxy_containers.insert(name.to_owned());
        }
    }
    Ok(resources)
}

fn remove_docker_object(kind: &str, name: &str) -> Result<(), RunExecutionError> {
    let arguments: Vec<&str> = match kind {
        "container" => vec!["container", "rm", "--force", name],
        "network" => vec!["network", "rm", name],
        _ => unreachable!("fixed Docker cleanup object kind"),
    };
    let output = docker_output(&arguments)?;
    if output.status.success() || docker_object_absent(kind, &output) {
        Ok(())
    } else {
        Err(RunExecutionError::new(format!(
            "unable to remove scoped Docker {kind} {name:?}: {}: {}",
            display_status(output.status),
            diagnostic_output(&output.stdout, &output.stderr)
        )))
    }
}

/// Remove runtime objects that are cryptographically/namespacely tied to one
/// profile output. Returns whether any leaked object or host-state path was
/// observed. Discovery and post-removal checks fail closed.
fn cleanup_scoped_docker_resources(output_dir: &Path) -> Result<bool, RunExecutionError> {
    let resources = discover_scoped_docker_resources(output_dir)?;
    let observed = !resources.is_empty();
    for container in resources
        .workload_containers
        .iter()
        .chain(resources.proxy_containers.iter())
    {
        remove_docker_object("container", container)?;
    }
    for network in &resources.networks {
        remove_docker_object("network", network)?;
    }
    for path in &resources.state_paths {
        if path.is_dir() {
            fs::remove_dir_all(path).map_err(|error| {
                RunExecutionError::new(format!(
                    "unable to remove scoped network-policy state {}: {error}",
                    path.display()
                ))
            })?;
        } else if path.exists() {
            fs::remove_file(path).map_err(|error| {
                RunExecutionError::new(format!(
                    "unable to remove scoped network-policy state {}: {error}",
                    path.display()
                ))
            })?;
        }
    }
    let remaining = discover_scoped_docker_resources(output_dir)?;
    if !remaining.workload_containers.is_empty() {
        return Err(RunExecutionError::new(format!(
            "scoped workload containers survived cleanup: {:?}",
            remaining.workload_containers
        )));
    }
    for proxy in &resources.proxy_containers {
        let inspected = docker_output(&["container", "inspect", proxy])?;
        if inspected.status.success() {
            return Err(RunExecutionError::new(format!(
                "scoped policy proxy survived cleanup: {proxy}"
            )));
        }
        if !docker_object_absent("container", &inspected) {
            return Err(RunExecutionError::new(format!(
                "scoped policy proxy cleanup could not verify absence for {proxy}: {}",
                diagnostic_output(&inspected.stdout, &inspected.stderr)
            )));
        }
    }
    for network in &resources.networks {
        let inspected = docker_output(&["network", "inspect", network])?;
        if inspected.status.success() {
            return Err(RunExecutionError::new(format!(
                "scoped policy network survived cleanup: {network}"
            )));
        }
        if !docker_object_absent("network", &inspected) {
            return Err(RunExecutionError::new(format!(
                "scoped policy network cleanup could not verify absence for {network}: {}",
                diagnostic_output(&inspected.stdout, &inspected.stderr)
            )));
        }
    }
    for path in &resources.state_paths {
        if path.exists() {
            return Err(RunExecutionError::new(format!(
                "scoped network-policy host state survived cleanup: {}",
                path.display()
            )));
        }
    }
    Ok(observed)
}

fn cleanup_profile_runtime(request: &ProfileRequest) -> Result<bool, RunExecutionError> {
    match request.runtime.as_str() {
        "docker" => cleanup_scoped_docker_resources(&request.output_dir),
        // The native Enroot leaf owns create/start/remove and namespace teardown.
        // Calling Docker discovery here is both unrelated and invalid on the
        // Docker-less ARM hosts this runtime is intended to support.
        "enroot" => Ok(false),
        runtime => Err(RunExecutionError::new(format!(
            "unsupported profile runtime during cleanup: {runtime:?}"
        ))),
    }
}

static SIGNAL_CANCELLED: AtomicBool = AtomicBool::new(false);

extern "C" fn interrupt_handler(_signal: libc::c_int) {
    SIGNAL_CANCELLED.store(true, Ordering::SeqCst);
}

struct SignalGuard {
    previous_interrupt: libc::sighandler_t,
    previous_terminate: libc::sighandler_t,
}

impl SignalGuard {
    fn install() -> Result<Self, RunExecutionError> {
        SIGNAL_CANCELLED.store(false, Ordering::SeqCst);
        let previous =
            unsafe { libc::signal(libc::SIGINT, interrupt_handler as libc::sighandler_t) };
        if previous == libc::SIG_ERR {
            return Err(RunExecutionError::new(format!(
                "unable to install interrupt handler: {}",
                io::Error::last_os_error()
            )));
        }
        let previous_terminate =
            unsafe { libc::signal(libc::SIGTERM, interrupt_handler as libc::sighandler_t) };
        if previous_terminate == libc::SIG_ERR {
            unsafe {
                libc::signal(libc::SIGINT, previous);
            }
            return Err(RunExecutionError::new(format!(
                "unable to install termination handler: {}",
                io::Error::last_os_error()
            )));
        }
        Ok(Self {
            previous_interrupt: previous,
            previous_terminate,
        })
    }
}

impl Drop for SignalGuard {
    fn drop(&mut self) {
        unsafe {
            libc::signal(libc::SIGINT, self.previous_interrupt);
            libc::signal(libc::SIGTERM, self.previous_terminate);
        }
    }
}

#[derive(Clone, Debug)]
struct Preflight {
    image_id: String,
    image_identity: Value,
    controller_sha256: Option<String>,
    enroot_runtime_provenance_sha256: Option<String>,
    enroot_executable: Option<PathBuf>,
}

#[derive(Clone, Debug)]
struct Attempt {
    attempt_index: usize,
    repetition: usize,
    task: RunTaskSpec,
}

#[derive(Clone, Debug)]
struct WaveAttempt {
    attempt_index: usize,
    wave_index: usize,
    launch_index: usize,
    wave: RunWaveSpec,
    launch: RunLaunchSpec,
    task: RunTaskSpec,
}

#[derive(Clone, Debug)]
struct ProjectedDuration {
    duration_sec: Option<f64>,
    valid: bool,
    reason: String,
    image_id: String,
    execution_plan_digest: Option<String>,
    window_start_ns: Option<u64>,
    window_end_ns: Option<u64>,
    cpu_usage_sec: Option<f64>,
    average_logical_cores: Option<f64>,
    cleanup_valid: Option<bool>,
    clock_reference: Option<Value>,
}

fn file_sha256(path: &Path) -> Result<String, io::Error> {
    let mut source = File::open(path)?;
    let mut digest = Sha256::new();
    let mut block = [0_u8; 1024 * 1024];
    loop {
        let count = source.read(&mut block)?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn ensure_output_absent(output: &Path) -> Result<(), RunExecutionError> {
    if output.exists() {
        Err(RunExecutionError::new(format!(
            "output path must not already exist: {}",
            output.display()
        )))
    } else {
        Ok(())
    }
}

fn validate_ordered_runtime_environment(spec: &RunSpecV3) -> Result<(), RunExecutionError> {
    if spec.defaults.runtime != "enroot" {
        return Ok(());
    }
    let mut inherited = env::vars_os()
        .filter_map(|(name, _value)| name.into_string().ok())
        .filter(|name| {
            (name.starts_with("ENROOT_") && !ENROOT_STATE_ENVIRONMENT.contains(&name.as_str()))
                || ENROOT_FORBIDDEN_ENVIRONMENT.contains(&name.as_str())
        })
        .collect::<Vec<_>>();
    inherited.sort();
    inherited.dedup();
    if inherited.is_empty() {
        Ok(())
    } else {
        Err(RunExecutionError::new(format!(
            "unsealed ambient Enroot/runtime configuration is not supported: {}",
            inherited.join(", ")
        )))
    }
}

fn claim_output(output: &Path) -> Result<(), RunExecutionError> {
    if let Some(parent) = output.parent() {
        fs::create_dir_all(parent).map_err(|error| {
            RunExecutionError::new(format!(
                "unable to create output parent {}: {error}",
                parent.display()
            ))
        })?;
    }
    match fs::create_dir(output) {
        Ok(()) => Ok(()),
        Err(error) if error.kind() == io::ErrorKind::AlreadyExists => {
            Err(RunExecutionError::new(format!(
                "output path was claimed while preflight ran: {}",
                output.display()
            )))
        }
        Err(error) => Err(RunExecutionError::new(format!(
            "unable to create output directory {}: {error}",
            output.display()
        ))),
    }
}

fn v1_request(spec: &RunSpecV1, task: &RunTaskSpec, output_dir: PathBuf) -> ProfileRequest {
    ProfileRequest {
        task_id: task.task_id.clone(),
        trajectory: task.atif.clone(),
        image: task.image.clone(),
        output_dir,
        cpuset_cpus: spec.cpus.clone(),
        pacing: crate::pacing::UNPACED.to_owned(),
        timeout_sec: task.action_timeout_sec,
        runtime: "docker".to_owned(),
        enroot_executable: None,
        controller_runtime: "rust-host-mount".to_owned(),
        profile_sets: vec!["cpu-average".to_owned()],
        network_policy: None,
        allowed_hosts: Vec::new(),
        workdir: None,
        environment: Vec::new(),
    }
}

fn v2_request(spec: &RunSpecV2, task: &RunTaskSpec, output_dir: PathBuf) -> ProfileRequest {
    ProfileRequest {
        task_id: task.task_id.clone(),
        trajectory: task.atif.clone(),
        image: task.image.clone(),
        output_dir,
        cpuset_cpus: spec.cpus.clone(),
        pacing: crate::pacing::UNPACED.to_owned(),
        timeout_sec: task.action_timeout_sec,
        runtime: "docker".to_owned(),
        enroot_executable: None,
        controller_runtime: "rust-host-mount".to_owned(),
        profile_sets: vec!["cpu-average".to_owned()],
        network_policy: None,
        allowed_hosts: Vec::new(),
        workdir: None,
        environment: Vec::new(),
    }
}

fn v3_request(
    spec: &RunSpecV3,
    task: &RunTaskSpec,
    cpus: &str,
    output_dir: PathBuf,
    enroot_executable: Option<&Path>,
) -> ProfileRequest {
    ProfileRequest {
        task_id: task.task_id.clone(),
        trajectory: task.atif.clone(),
        image: task.image.clone(),
        output_dir,
        cpuset_cpus: cpus.to_owned(),
        pacing: spec.defaults.pacing.clone(),
        timeout_sec: Some(
            task.action_timeout_sec
                .unwrap_or(spec.defaults.action_timeout_sec),
        ),
        runtime: spec.defaults.runtime.clone(),
        enroot_executable: enroot_executable.map(Path::to_path_buf),
        controller_runtime: spec.defaults.controller_runtime.clone(),
        profile_sets: spec.defaults.profile_sets.clone(),
        network_policy: Some(spec.defaults.network_policy.clone()),
        allowed_hosts: spec.defaults.allowed_hosts.clone(),
        workdir: spec.defaults.workdir.clone(),
        environment: spec.defaults.environment.clone(),
    }
}

fn value_object(value: &Value) -> Option<&Map<String, Value>> {
    value.as_object()
}

fn image_id(report: &Map<String, Value>) -> Option<String> {
    let identity = report.get("image_identity")?.as_object()?;
    if let Some(value) = identity
        .get("id")
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    {
        return Some(value.to_owned());
    }
    if identity.get("runtime").and_then(Value::as_str) != Some("enroot") {
        return None;
    }
    sha256_digest(identity.get("sha256")).map(|digest| format!("sha256:{digest}"))
}

fn json_string_array(value: Option<&Value>) -> Option<Vec<String>> {
    value?
        .as_array()?
        .iter()
        .map(|item| item.as_str().map(str::to_owned))
        .collect()
}

fn check_preflight(
    report: Value,
    task: &RunTaskSpec,
    cpus: &str,
    profile_sets: &[String],
    runtime: &str,
    controller_runtime: &str,
    enroot_executable: Option<&Path>,
) -> Result<Preflight, Vec<String>> {
    let Some(report) = value_object(&report) else {
        return Err(vec!["profile preflight returned a non-object".to_owned()]);
    };
    let mut reasons = Vec::new();
    if report.get("ready") != Some(&Value::Bool(true)) {
        reasons.push("profile preflight did not report ready=true".to_owned());
    }
    if report.get("source_sha256").and_then(Value::as_str) != Some(task.source_sha256.as_str()) {
        reasons.push("preflight source digest differs from the sealed ATIF".to_owned());
    }
    if report.get("cpuset_cpus").and_then(Value::as_str) != Some(cpus) {
        reasons.push("preflight CPU set differs from the requested CPU set".to_owned());
    }
    if json_string_array(report.get("profile_sets")).as_deref() != Some(profile_sets) {
        reasons.push("preflight profile sets differ from the requested profile sets".to_owned());
    }
    if report.get("runtime").and_then(Value::as_str) != Some(runtime) {
        reasons.push("preflight runtime differs from the requested runtime".to_owned());
    }
    let image_identity = report.get("image_identity").cloned().unwrap_or(Value::Null);
    if runtime == "enroot" {
        let identity = image_identity.as_object();
        if identity
            .and_then(|value| value.get("runtime"))
            .and_then(Value::as_str)
            != Some("enroot")
            || identity
                .and_then(|value| value.get("kind"))
                .and_then(Value::as_str)
                != Some("image-file")
            || identity
                .and_then(|value| value.get("immutable"))
                .and_then(Value::as_bool)
                != Some(true)
        {
            reasons.push("preflight Enroot image identity is missing or invalid".to_owned());
        }
        let expected_path = Path::new(&task.image).canonicalize().ok();
        let observed_path = identity
            .and_then(|value| value.get("path"))
            .and_then(Value::as_str)
            .map(PathBuf::from);
        if expected_path.is_none() || observed_path != expected_path {
            reasons.push(
                "preflight Enroot image path differs from the declared task image".to_owned(),
            );
        }
        let expected_executable = enroot_executable.filter(|path| path.is_absolute());
        let runtime_check = report
            .get("checks")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .find(|check| check.get("name").and_then(Value::as_str) == Some("runtime"));
        let observed_executable = runtime_check
            .and_then(|check| check.get("executable"))
            .and_then(Value::as_str)
            .map(PathBuf::from);
        if expected_executable.is_none()
            || observed_executable.as_deref() != expected_executable
            || runtime_check
                .and_then(|check| check.get("status"))
                .and_then(Value::as_str)
                != Some("passed")
            || runtime_check
                .and_then(|check| check.get("runtime"))
                .and_then(Value::as_str)
                != Some("enroot")
        {
            reasons.push(
                "preflight Enroot executable differs from captured runtime provenance".to_owned(),
            );
        }
    }
    let observed = image_id(report);
    if observed.is_none() {
        reasons.push("preflight did not resolve an image ID".to_owned());
    } else if task.expected_image_id.is_some()
        && task.expected_image_id.as_deref() != observed.as_deref()
    {
        reasons.push(format!(
            "resolved image ID {:?} differs from expected {:?}",
            observed.as_deref().unwrap_or_default(),
            task.expected_image_id.as_deref().unwrap_or_default()
        ));
    }
    let controller_identity = report.get("controller_identity").and_then(Value::as_object);
    let controller_sha256 =
        controller_identity.and_then(|identity| sha256_digest(identity.get("sha256")));
    if controller_runtime == "rust-host-mount" {
        if controller_identity
            .and_then(|identity| identity.get("kind"))
            .and_then(Value::as_str)
            != Some("rust-host-mounted-binary")
        {
            reasons
                .push("preflight Rust controller identity kind is missing or invalid".to_owned());
        }
        if controller_sha256.is_none() {
            reasons.push("preflight did not content-identify the Rust controller".to_owned());
        }
    }
    if reasons.is_empty() {
        Ok(Preflight {
            image_id: observed.expect("image ID checked above"),
            image_identity,
            controller_sha256,
            enroot_runtime_provenance_sha256: None,
            enroot_executable: enroot_executable.map(Path::to_path_buf),
        })
    } else {
        Err(reasons)
    }
}

fn preflight_v1(
    spec: &RunSpecV1,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Result<HashMap<String, Preflight>, RunExecutionError> {
    let mut results = HashMap::new();
    let mut errors = Vec::new();
    let profile_sets = vec!["cpu-average".to_owned()];
    for task in &spec.tasks {
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new("profile preflight interrupted"));
        }
        let output = spec
            .output
            .join("tasks")
            .join(&task.task_id)
            .join("rep-0001")
            .join("profile");
        let request = v1_request(spec, task, output);
        match launcher
            .preflight_with_cancellation(&request, cancellation)
            .map_err(|error| vec![error.to_string()])
            .and_then(|report| {
                check_preflight(
                    report,
                    task,
                    &spec.cpus,
                    &profile_sets,
                    &request.runtime,
                    &request.controller_runtime,
                    request.enroot_executable.as_deref(),
                )
            }) {
            Ok(preflight) => {
                results.insert(task.task_id.clone(), preflight);
            }
            Err(reasons) => errors.push(format!("{}: {}", task.task_id, reasons.join("; "))),
        }
    }
    if errors.is_empty() {
        Ok(results)
    } else {
        Err(RunExecutionError::new(format!(
            "preflight failed: {}",
            errors.join(" | ")
        )))
    }
}

fn preflight_v2(
    spec: &RunSpecV2,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Result<HashMap<String, Preflight>, RunExecutionError> {
    let mut results = HashMap::new();
    let mut errors = Vec::new();
    let profile_sets = vec!["cpu-average".to_owned()];
    for task in &spec.tasks {
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new("profile preflight interrupted"));
        }
        let output = spec
            .output
            .join("tasks")
            .join(&task.task_id)
            .join("rep-0001")
            .join("profile");
        let request = v2_request(spec, task, output);
        match launcher
            .preflight_with_cancellation(&request, cancellation)
            .map_err(|error| vec![error.to_string()])
            .and_then(|report| {
                check_preflight(
                    report,
                    task,
                    &spec.cpus,
                    &profile_sets,
                    &request.runtime,
                    &request.controller_runtime,
                    request.enroot_executable.as_deref(),
                )
            }) {
            Ok(preflight) => {
                results.insert(task.task_id.clone(), preflight);
            }
            Err(reasons) => errors.push(format!("{}: {}", task.task_id, reasons.join("; "))),
        }
    }
    if errors.is_empty() {
        Ok(results)
    } else {
        Err(RunExecutionError::new(format!(
            "preflight failed: {}",
            errors.join(" | ")
        )))
    }
}

fn preflight_v3(
    spec: &RunSpecV3,
    attempts: &[WaveAttempt],
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
    enroot_runtime_binding: Option<(&str, &Path)>,
) -> Result<HashMap<(String, String), Preflight>, RunExecutionError> {
    if spec.defaults.runtime == "enroot"
        && !enroot_runtime_binding.is_some_and(|(digest, executable)| {
            executable.is_absolute()
                && digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        return Err(RunExecutionError::new(
            "Enroot runtime provenance digest or absolute executable is missing or invalid",
        ));
    }
    let mut results = HashMap::new();
    let mut image_ids_by_task: HashMap<String, String> = HashMap::new();
    let mut enroot_by_task: HashMap<String, Preflight> = HashMap::new();
    let mut controller_sha256: Option<String> = None;
    let mut errors = Vec::new();
    for attempt in attempts {
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new("profile preflight interrupted"));
        }
        let key = (attempt.task.task_id.clone(), attempt.launch.cpus.clone());
        if results.contains_key(&key) {
            continue;
        }
        if spec.defaults.runtime == "enroot" {
            if let Some(cached) = enroot_by_task.get(&attempt.task.task_id) {
                results.insert(key, cached.clone());
                continue;
            }
        }
        let output = wave_profile_path(spec, attempt);
        let request = v3_request(
            spec,
            &attempt.task,
            &attempt.launch.cpus,
            output,
            enroot_runtime_binding.map(|(_, executable)| executable),
        );
        let checked = launcher
            .preflight_with_cancellation(&request, cancellation)
            .map_err(|error| vec![error.to_string()])
            .and_then(|report| {
                check_preflight(
                    report,
                    &attempt.task,
                    &attempt.launch.cpus,
                    &spec.defaults.profile_sets,
                    &request.runtime,
                    &request.controller_runtime,
                    request.enroot_executable.as_deref(),
                )
            });
        match checked {
            Ok(mut preflight) => {
                if spec.defaults.runtime == "enroot" {
                    preflight.enroot_runtime_provenance_sha256 =
                        enroot_runtime_binding.map(|(digest, _)| digest.to_owned());
                    preflight.enroot_executable =
                        enroot_runtime_binding.map(|(_, executable)| executable.to_path_buf());
                }
                if let Some(previous) = image_ids_by_task.get(&attempt.task.task_id) {
                    if previous != &preflight.image_id {
                        errors.push(format!(
                            "{} on {}: resolved image ID changed across launch placements",
                            attempt.task.task_id, attempt.launch.cpus
                        ));
                        continue;
                    }
                }
                image_ids_by_task
                    .entry(attempt.task.task_id.clone())
                    .or_insert_with(|| preflight.image_id.clone());
                if let Some(observed) = preflight.controller_sha256.as_ref() {
                    if controller_sha256
                        .as_ref()
                        .is_some_and(|previous| previous != observed)
                    {
                        errors.push(format!(
                            "{} on {}: Rust controller content identity changed across preflight placements",
                            attempt.task.task_id, attempt.launch.cpus
                        ));
                        continue;
                    }
                    controller_sha256.get_or_insert_with(|| observed.clone());
                }
                if spec.defaults.runtime == "enroot" {
                    enroot_by_task.insert(attempt.task.task_id.clone(), preflight.clone());
                }
                results.insert(key, preflight);
            }
            Err(reasons) => errors.push(format!(
                "{} on {}: {}",
                attempt.task.task_id,
                attempt.launch.cpus,
                reasons.join("; ")
            )),
        }
    }
    if errors.is_empty() {
        Ok(results)
    } else {
        Err(RunExecutionError::new(format!(
            "preflight failed: {}",
            errors.join(" | ")
        )))
    }
}

fn native_controller_digest<'a>(
    preflights: impl Iterator<Item = &'a Preflight>,
) -> Result<String, RunExecutionError> {
    let digests: BTreeSet<_> = preflights
        .filter_map(|preflight| preflight.controller_sha256.clone())
        .collect();
    match digests.len() {
        1 => Ok(digests.into_iter().next().expect("one digest was observed")),
        0 => Err(RunExecutionError::new(
            "native run preflight did not retain a Rust controller content identity",
        )),
        _ => Err(RunExecutionError::new(
            "Rust controller content identity changed across preflight placements",
        )),
    }
}

fn positive_number(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_f64()?;
    (number.is_finite() && number > 0.0).then_some(number)
}

fn nonnegative_number(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_f64()?;
    (number.is_finite() && number >= 0.0).then_some(number)
}

fn nonnegative_integer(value: Option<&Value>) -> Option<u64> {
    value?.as_u64()
}

fn push_incomplete_replay_reason(replay: Option<&Map<String, Value>>, reasons: &mut Vec<String>) {
    let Some(replay) = replay else { return };
    for field in [
        "source_outcome_comparison_unavailable_count",
        "actions_launch_failed",
        "actions_source_argument_parse_failed",
        "actions_skipped_after_timeout",
    ] {
        if replay
            .get(field)
            .is_some_and(|value| value.as_u64() != Some(0))
        {
            push_reason(reasons, format!("replay {field} is nonzero or invalid"));
        }
    }
    if let Some(recorded) = replay.get("tool_actions_recorded") {
        if !recorded.as_u64().is_some_and(|count| {
            count > 0 && replay.get("tool_actions_replayed").and_then(Value::as_u64) == Some(count)
        }) {
            push_reason(reasons, "replay tool actions are incomplete or invalid");
        }
    }
}

fn sha256_digest(value: Option<&Value>) -> Option<String> {
    let text = value?.as_str()?;
    (text.len() == 64
        && text
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte)))
    .then(|| text.to_owned())
}

fn close(left: f64, right: f64) -> bool {
    (left - right).abs() <= (1e-9_f64).max(1e-9 * left.abs().max(right.abs()))
}

fn canonical_clock_reference_matches(report: &Map<String, Value>) -> bool {
    let Some(observed) = report
        .get("canonical_clock_reference")
        .and_then(Value::as_object)
    else {
        return false;
    };
    let expected = clock_reference();
    let expected = expected
        .as_object()
        .expect("local monotonic clock reference is an object");
    [
        "clock_id",
        "timestamp_unit",
        "time_namespace",
        "time_namespace_monotonic_offset_ns",
        "boot_id",
        "storage_precision_ns",
    ]
    .iter()
    .all(|field| observed.get(*field) == expected.get(*field))
}

fn push_reason(reasons: &mut Vec<String>, reason: impl Into<String>) {
    let reason = reason.into();
    if !reasons.contains(&reason) {
        reasons.push(reason);
    }
}

fn cpuset_ids(value: &str) -> Option<BTreeSet<u32>> {
    let mut ids = BTreeSet::new();
    for component in value.split(',') {
        let component = component.trim();
        if component.is_empty() {
            return None;
        }
        if let Some((first, last)) = component.split_once('-') {
            let first = first.trim().parse::<u32>().ok()?;
            let last = last.trim().parse::<u32>().ok()?;
            if last < first {
                return None;
            }
            ids.extend(first..=last);
        } else {
            ids.insert(component.parse::<u32>().ok()?);
        }
    }
    (!ids.is_empty()).then_some(ids)
}

fn json_u32_array(value: Option<&Value>) -> Option<Vec<u32>> {
    value?
        .as_array()?
        .iter()
        .map(|value| value.as_u64().and_then(|value| u32::try_from(value).ok()))
        .collect()
}

#[cfg(unix)]
fn metadata_mtime_ns(metadata: &fs::Metadata) -> Option<i64> {
    metadata
        .mtime()
        .checked_mul(1_000_000_000)?
        .checked_add(metadata.mtime_nsec())
}

#[cfg(not(unix))]
fn metadata_mtime_ns(_metadata: &fs::Metadata) -> Option<i64> {
    None
}

fn current_network_namespace_inode() -> Option<u64> {
    fs::metadata("/proc/self/ns/net")
        .ok()
        .map(|metadata| metadata.ino())
}

fn enroot_no_network_evidence_valid(policy: &Map<String, Value>) -> bool {
    if policy.get("effective_mode").and_then(Value::as_str) != Some("no-network") {
        return false;
    }
    let Some(enforcement) = policy.get("enforcement").and_then(Value::as_object) else {
        return false;
    };
    let parent_inode = current_network_namespace_inode();
    let child_inode = enforcement.get("netns_inode").and_then(Value::as_u64);
    let interfaces = json_string_array(enforcement.get("interfaces"));
    let routes = enforcement.get("ipv4_routes").and_then(Value::as_array);
    let publication = enforcement.get("publication").and_then(Value::as_object);
    enforcement.get("schema_version").and_then(Value::as_str)
        == Some("tasktrace-enroot-network-enforcement.v1")
        && enforcement.get("state").and_then(Value::as_str) == Some("enforced")
        && enforcement.get("backend").and_then(Value::as_str)
            == Some("linux-user-network-namespace-v1")
        && enforcement
            .get("policy_effective_mode")
            .and_then(Value::as_str)
            == Some("no-network")
        && enforcement.get("valid").and_then(Value::as_bool) == Some(true)
        && enforcement.get("netns_differs").and_then(Value::as_bool) == Some(true)
        && parent_inode.is_some()
        && enforcement
            .get("parent_netns_inode")
            .and_then(Value::as_u64)
            == parent_inode
        && child_inode.is_some_and(|child| Some(child) != parent_inode && child > 0)
        && enforcement.get("controller_euid").and_then(Value::as_u64)
            == Some(unsafe { libc::geteuid() } as u64)
        && enforcement.get("controller_egid").and_then(Value::as_u64)
            == Some(unsafe { libc::getegid() } as u64)
        && enforcement.get("namespace_euid").and_then(Value::as_u64)
            == Some(unsafe { libc::geteuid() } as u64)
        && enforcement.get("namespace_egid").and_then(Value::as_u64)
            == Some(unsafe { libc::getegid() } as u64)
        && interfaces.as_deref() == Some(&["lo".to_owned()][..])
        && routes.is_some_and(Vec::is_empty)
        && json_string_array(enforcement.get("ipv6_addresses"))
            .is_some_and(|lines| crate::enroot_runtime::enroot_network::loopback_only_ipv6(&lines))
        && publication
            .and_then(|value| value.get("source"))
            .and_then(Value::as_str)
            == Some("controller-private-temporary-state")
        && publication
            .and_then(|value| value.get("published_after_enroot_exit"))
            .and_then(Value::as_bool)
            == Some(true)
}

fn enroot_execution_plan_digest(
    task: &RunTaskSpec,
    spec: &RunSpecV3,
    launch: &RunLaunchSpec,
    preflight: &Preflight,
) -> String {
    let payload = json!({
        "runtime": "enroot",
        "task_id": task.task_id,
        "source_sha256": task.source_sha256,
        "image_id": preflight.image_id,
        "rust_controller_binary_sha256": preflight.controller_sha256,
        "enroot_runtime_provenance_sha256": preflight.enroot_runtime_provenance_sha256,
        "enroot_executable": preflight.enroot_executable,
        "cpus": launch.cpus,
        "pacing": spec.defaults.pacing,
        "network_policy": spec.defaults.network_policy,
        "allowed_hosts": spec.defaults.allowed_hosts,
        "controller_runtime": spec.defaults.controller_runtime,
        "controller_cpus": spec.defaults.controller_cpus,
        "wave_start_coordination": "all-ready-release-v1",
        "effective_action_timeout_sec": task
            .action_timeout_sec
            .unwrap_or(spec.defaults.action_timeout_sec),
    });
    format!(
        "{:x}",
        Sha256::digest(
            serde_json::to_vec(&payload).expect("validated Enroot execution plan serializes")
        )
    )
}

fn project_enroot_duration(
    report: &Map<String, Value>,
    task: &RunTaskSpec,
    preflight: &Preflight,
    spec: &RunSpecV3,
    launch: &RunLaunchSpec,
    require_wave_start_gate: bool,
) -> ProjectedDuration {
    let mut reasons = Vec::new();
    if report.get("runtime").and_then(Value::as_str) != Some("enroot") {
        push_reason(&mut reasons, "profile runtime is missing or is not enroot");
    }
    if report.get("controller_runtime").and_then(Value::as_str)
        != Some(spec.defaults.controller_runtime.as_str())
    {
        push_reason(
            &mut reasons,
            "profile controller runtime differs from defaults.controller_runtime",
        );
    }
    let controller = report.get("controller_identity").and_then(Value::as_object);
    if controller
        .and_then(|identity| identity.get("kind"))
        .and_then(Value::as_str)
        != Some("rust-host-mounted-binary")
    {
        push_reason(
            &mut reasons,
            "profile Rust controller identity kind is missing or invalid",
        );
    }
    let observed_controller = controller.and_then(|identity| sha256_digest(identity.get("sha256")));
    if observed_controller.as_deref() != preflight.controller_sha256.as_deref() {
        push_reason(
            &mut reasons,
            "Rust controller content identity differs from preflight",
        );
    }
    if !preflight
        .enroot_runtime_provenance_sha256
        .as_deref()
        .is_some_and(|digest| {
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
        })
    {
        push_reason(
            &mut reasons,
            "Enroot runtime provenance identity is missing or invalid",
        );
    }
    if !preflight
        .enroot_executable
        .as_deref()
        .is_some_and(Path::is_absolute)
    {
        push_reason(
            &mut reasons,
            "captured Enroot runtime executable identity is missing or invalid",
        );
    }

    let cpu = report.get("container_cpu").and_then(Value::as_object);
    if cpu.is_none() {
        push_reason(&mut reasons, "container_cpu evidence is missing");
    }
    let duration = cpu.and_then(|cpu| positive_number(cpu.get("wall_sec")));
    if duration.is_none() {
        push_reason(
            &mut reasons,
            "container_cpu.wall_sec is not a finite positive number",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("method"))
        .and_then(Value::as_str)
        != Some("wall-clock-only-no-cgroup")
    {
        push_reason(
            &mut reasons,
            "Enroot CPU method is not wall-clock-only-no-cgroup",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("wall_clock_source"))
        .and_then(Value::as_str)
        != Some(ENROOT_TASK_CLOCK_SOURCE)
    {
        push_reason(
            &mut reasons,
            "Enroot wall duration is not from one monotonic clock",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("valid_for_wall_time_reporting"))
        .and_then(Value::as_bool)
        != Some(true)
    {
        push_reason(
            &mut reasons,
            "Enroot wall-time evidence is not explicitly valid",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("valid_for_cpu_performance"))
        .and_then(Value::as_bool)
        != Some(false)
    {
        push_reason(
            &mut reasons,
            "Enroot report must not claim task-scoped CPU performance validity",
        );
    }
    if [
        "usage_sec",
        "user_sec",
        "system_sec",
        "average_logical_cores",
        "average_logical_cores_used",
        "utilization_fraction_of_allocation",
        "utilization_percent_of_allocation",
    ]
    .iter()
    .any(|field| {
        cpu.and_then(|cpu| cpu.get(*field))
            .is_some_and(|value| !value.is_null())
    }) {
        push_reason(
            &mut reasons,
            "Enroot report manufactured task-scoped CPU usage or average cores",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("cpu_containment_enforced"))
        .and_then(Value::as_bool)
        != Some(false)
        || cpu
            .and_then(|cpu| cpu.get("cpu_affinity_enforced"))
            .and_then(Value::as_bool)
            != Some(false)
    {
        push_reason(
            &mut reasons,
            "Enroot taskset placement must not claim CPU containment or exclusivity",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("allocated_cpu_source"))
        .and_then(Value::as_str)
        != Some("taskset-requested-affinity")
    {
        push_reason(
            &mut reasons,
            "Enroot requested task affinity evidence is missing",
        );
    }
    if cpu.and_then(|cpu| positive_number(cpu.get("allocated_logical_cpus")))
        != Some(launch.cpu_ids.len() as f64)
    {
        push_reason(
            &mut reasons,
            "Enroot allocation width differs from the launch CPU set",
        );
    }
    if cpu
        .and_then(|cpu| cpu.get("requested_cpu_affinity"))
        .and_then(Value::as_str)
        != Some(launch.cpus.as_str())
    {
        push_reason(
            &mut reasons,
            "Enroot requested task affinity differs from the launch CPU set",
        );
    }
    if cpu
        .and_then(|cpu| json_u32_array(cpu.get("initial_effective_logical_cpu_ids")))
        .as_deref()
        != Some(launch.cpu_ids.as_slice())
    {
        push_reason(
            &mut reasons,
            "Enroot initial effective task affinity differs from the launch CPU set",
        );
    }

    let window_start_ns = cpu.and_then(|cpu| nonnegative_integer(cpu.get("window_start_ns")));
    let window_end_ns = cpu.and_then(|cpu| nonnegative_integer(cpu.get("window_end_ns")));
    let clock = cpu
        .and_then(|cpu| cpu.get("clock_reference"))
        .and_then(Value::as_object);
    let resolution_ns = clock.and_then(|clock| nonnegative_integer(clock.get("resolution_ns")));
    if clock.is_none()
        || clock
            .and_then(|clock| clock.get("clock_name"))
            .and_then(Value::as_str)
            != Some("CLOCK_BOOTTIME")
        || clock
            .and_then(|clock| clock.get("source"))
            .and_then(Value::as_str)
            != Some(ENROOT_TASK_CLOCK_SOURCE)
        || !resolution_ns.is_some_and(|value| value > 0)
        || !clock
            .and_then(|clock| clock.get("boot_id"))
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
        || !clock
            .and_then(|clock| clock.get("time_namespace"))
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
    {
        push_reason(&mut reasons, "Enroot task clock reference is invalid");
    }
    if !matches!((window_start_ns, window_end_ns), (Some(start), Some(end)) if end > start) {
        push_reason(
            &mut reasons,
            "Enroot task window is missing or non-increasing",
        );
    } else if let (Some(duration), Some(start), Some(end), Some(resolution)) =
        (duration, window_start_ns, window_end_ns, resolution_ns)
    {
        let observed = (end - start) as f64 / 1_000_000_000.0;
        if (observed - duration).abs() > 2.0 * resolution as f64 / 1_000_000_000.0 {
            push_reason(
                &mut reasons,
                "Enroot wall duration disagrees with raw task timestamps",
            );
        }
    }

    if require_wave_start_gate {
        let gate = report.get("run_wave_start_gate").and_then(Value::as_object);
        let released_ns = gate.and_then(|gate| nonnegative_integer(gate.get("released_ns")));
        let measured_start_ns =
            gate.and_then(|gate| nonnegative_integer(gate.get("measured_start_ns")));
        let release_to_start_ns =
            gate.and_then(|gate| nonnegative_integer(gate.get("release_to_start_ns")));
        let valid = gate.is_some_and(|gate| {
            gate.get("schema_version").and_then(Value::as_str)
                == Some("tasktrace-run-wave-start-gate.v1")
                && gate.get("requested").and_then(Value::as_bool) == Some(true)
                && gate.get("state").and_then(Value::as_str) == Some("released")
                && gate.get("ready").and_then(Value::as_bool) == Some(true)
                && gate.get("released").and_then(Value::as_bool) == Some(true)
                && gate.get("valid").and_then(Value::as_bool) == Some(true)
                && gate.get("clock_reference").and_then(Value::as_object) == clock
        }) && matches!(
            (released_ns, measured_start_ns, window_start_ns, release_to_start_ns),
            (Some(released), Some(measured), Some(window), Some(delta))
                if released > 0
                    && released <= measured
                    && measured == window
                    && delta == measured - released
        );
        if !valid {
            push_reason(
                &mut reasons,
                "Enroot coordinated wave-start gate evidence is missing or invalid",
            );
        }
    }

    if report.get("image").and_then(Value::as_str) != Some(task.image.as_str()) {
        push_reason(
            &mut reasons,
            "profile image reference differs from the declared task image",
        );
    }
    let observed_identity = report.get("image_identity");
    if observed_identity != Some(&preflight.image_identity) {
        push_reason(
            &mut reasons,
            "profile Enroot image identity differs from preflight",
        );
    }
    let observed_image_id = image_id(report);
    let retained_image_id = observed_image_id
        .clone()
        .unwrap_or_else(|| preflight.image_id.clone());
    if observed_image_id.as_deref() != Some(preflight.image_id.as_str()) {
        push_reason(&mut reasons, "profile image ID differs from preflight");
    }
    match (
        Path::new(&task.image).canonicalize(),
        fs::metadata(&task.image),
    ) {
        (Ok(path), Ok(metadata)) => {
            let identity = observed_identity.and_then(Value::as_object);
            if identity
                .and_then(|identity| identity.get("path"))
                .and_then(Value::as_str)
                != path.to_str()
                || identity
                    .and_then(|identity| identity.get("size_bytes"))
                    .and_then(Value::as_u64)
                    != Some(metadata.len())
                || identity
                    .and_then(|identity| identity.get("mtime_ns"))
                    .and_then(Value::as_i64)
                    != metadata_mtime_ns(&metadata)
            {
                push_reason(
                    &mut reasons,
                    "profile image identity differs from the declared local sqsh",
                );
            }
        }
        (Err(error), _) | (_, Err(error)) => push_reason(
            &mut reasons,
            format!("unable to revalidate Enroot image after replay: {error}"),
        ),
    }

    if report.get("pacing").and_then(Value::as_str) != Some(spec.defaults.pacing.as_str()) {
        push_reason(&mut reasons, "profile pacing differs from defaults.pacing");
    }
    match report.get("profile_provenance").and_then(Value::as_object) {
        Some(provenance)
            if json_string_array(provenance.get("profile_sets")).as_deref()
                == Some(spec.defaults.profile_sets.as_slice()) => {}
        _ => push_reason(
            &mut reasons,
            "profile sets differ from defaults.profile_sets",
        ),
    }
    let expected_network =
        NetworkPolicy::parse(&spec.defaults.network_policy, &spec.defaults.allowed_hosts)
            .expect("strict run spec already validated its network policy")
            .comparability_digest();
    let network = report.get("network_policy").and_then(Value::as_object);
    if network
        .and_then(|policy| policy.get("comparability_sha256"))
        .and_then(Value::as_str)
        != Some(expected_network.as_str())
    {
        push_reason(
            &mut reasons,
            "network policy differs from defaults.network_policy",
        );
    } else if spec.defaults.network_policy == "no-network"
        && !network.is_some_and(enroot_no_network_evidence_valid)
    {
        push_reason(
            &mut reasons,
            "Enroot no-network enforcement evidence is invalid",
        );
    }

    let replay = report.get("replay_summary").and_then(Value::as_object);
    push_incomplete_replay_reason(replay, &mut reasons);
    match replay.and_then(|replay| nonnegative_integer(replay.get("actions_timed_out"))) {
        None => push_reason(&mut reasons, "replay timeout count is missing or invalid"),
        Some(count) if count > 0 => {
            push_reason(&mut reasons, format!("{count} replay action(s) timed out"))
        }
        _ => {}
    }
    match replay.and_then(|replay| nonnegative_integer(replay.get("source_outcome_mismatch_count")))
    {
        None => push_reason(
            &mut reasons,
            "replay outcome-mismatch count is missing or invalid",
        ),
        Some(count) if count > 0 => push_reason(
            &mut reasons,
            format!("{count} replay source-outcome mismatch(es)"),
        ),
        _ => {}
    }
    if report.get("profiling_valid").and_then(Value::as_bool) != Some(true) {
        push_reason(&mut reasons, "profiling evidence is not explicitly valid");
    }
    if report.get("status").and_then(Value::as_str) != Some("valid") {
        push_reason(&mut reasons, "pipeline status is not valid");
    }
    if report
        .get("source_trajectory_sha256")
        .and_then(Value::as_str)
        != Some(task.source_sha256.as_str())
    {
        push_reason(
            &mut reasons,
            "profile source digest differs from the sealed ATIF",
        );
    }

    let lifecycle = report
        .get("enroot_lifecycle")
        .or_else(|| report.get("canonical_lifecycle"))
        .and_then(Value::as_object)
        .filter(|lifecycle| {
            lifecycle.get("schema_version").and_then(Value::as_str)
                == Some("tasktrace-enroot-lifecycle.v1")
        });
    let mut cleanup_valid = false;
    if let Some(lifecycle) = lifecycle {
        let lifecycle_clock = lifecycle.get("clock_reference").and_then(Value::as_object);
        if lifecycle_clock
            .and_then(|clock| clock.get("clock_name"))
            .and_then(Value::as_str)
            != Some("CLOCK_BOOTTIME")
            || lifecycle_clock.and_then(|clock| clock.get("boot_id"))
                != clock.and_then(|clock| clock.get("boot_id"))
            || lifecycle_clock.and_then(|clock| clock.get("time_namespace"))
                != clock.and_then(|clock| clock.get("time_namespace"))
        {
            push_reason(
                &mut reasons,
                "Enroot task and lifecycle clocks lack an identity mapping",
            );
        }
        let containment = lifecycle
            .get("process_containment")
            .and_then(Value::as_object);
        if containment
            .and_then(|value| value.get("backend"))
            .and_then(Value::as_str)
            != Some("linux-user-pid-namespace-v1")
            || containment
                .and_then(|value| value.get("namespace_init_exit_kills_remaining_processes"))
                .and_then(Value::as_bool)
                != Some(true)
            || containment
                .and_then(|value| value.get("outer_exit_kills_namespace_init"))
                .and_then(Value::as_bool)
                != Some(true)
        {
            push_reason(
                &mut reasons,
                "Enroot PID-namespace containment evidence is invalid",
            );
        }
        let cleanup = lifecycle.get("cleanup").and_then(Value::as_object);
        cleanup_valid = cleanup
            .and_then(|cleanup| cleanup.get("valid"))
            .and_then(Value::as_bool)
            == Some(true)
            && cleanup
                .and_then(|cleanup| cleanup.get("container_absent"))
                .and_then(Value::as_bool)
                == Some(true);
        if !cleanup_valid {
            push_reason(&mut reasons, "Enroot cleanup evidence is invalid");
        }
        let starts = lifecycle
            .get("phases")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
            .filter(|phase| phase.get("phase").and_then(Value::as_str) == Some("start"))
            .collect::<Vec<_>>();
        if starts.len() != 1 {
            push_reason(&mut reasons, "Enroot start lifecycle boundary is missing");
        } else if let (Some(task_start), Some(task_end), Some(resolution)) =
            (window_start_ns, window_end_ns, resolution_ns)
        {
            let phase_start = starts[0].get("start_ns").and_then(Value::as_u64);
            let phase_end = starts[0].get("end_ns").and_then(Value::as_u64);
            if phase_start.is_none()
                || phase_end.is_none()
                || task_start.saturating_add(resolution) < phase_start.unwrap_or_default()
                || task_end > phase_end.unwrap_or_default().saturating_add(resolution)
            {
                push_reason(
                    &mut reasons,
                    "Enroot task window falls outside the start lifecycle phase",
                );
            }
        }
    } else {
        push_reason(&mut reasons, "Enroot lifecycle evidence is missing");
    }

    ProjectedDuration {
        duration_sec: duration,
        valid: duration.is_some() && reasons.is_empty(),
        reason: reasons.join("; "),
        image_id: retained_image_id,
        execution_plan_digest: Some(enroot_execution_plan_digest(task, spec, launch, preflight)),
        window_start_ns,
        window_end_ns,
        cpu_usage_sec: None,
        average_logical_cores: None,
        cleanup_valid: Some(cleanup_valid),
        clock_reference: clock.cloned().map(Value::Object),
    }
}

fn project_duration(
    report: &Map<String, Value>,
    task: &RunTaskSpec,
    preflight: &Preflight,
    require_common_window: bool,
    expected_cpuset: Option<(&str, usize)>,
    ordered: Option<(&RunSpecV3, &RunLaunchSpec)>,
    require_enroot_wave_start_gate: bool,
) -> ProjectedDuration {
    if let Some((spec, launch)) = ordered {
        if spec.defaults.runtime == "enroot" {
            return project_enroot_duration(
                report,
                task,
                preflight,
                spec,
                launch,
                require_enroot_wave_start_gate,
            );
        }
    }
    let mut reasons = Vec::new();
    if report.get("controller_runtime").and_then(Value::as_str) != Some("rust-host-mount") {
        push_reason(
            &mut reasons,
            "profile controller runtime is missing or is not rust-host-mount",
        );
    }
    if report
        .get("controller_identity")
        .and_then(Value::as_object)
        .and_then(|identity| identity.get("kind"))
        .and_then(Value::as_str)
        != Some("rust-host-mounted-binary")
    {
        push_reason(
            &mut reasons,
            "profile Rust controller identity kind is missing or invalid",
        );
    }
    let cpu = match report.get("container_cpu").and_then(Value::as_object) {
        Some(cpu) => cpu,
        None => {
            push_reason(&mut reasons, "container_cpu evidence is missing");
            static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
            EMPTY.get_or_init(Map::new)
        }
    };
    let duration = positive_number(cpu.get("wall_sec"));
    if duration.is_none() {
        push_reason(
            &mut reasons,
            "container_cpu.wall_sec is not a finite positive number",
        );
    }
    if cpu.get("valid_for_cpu_performance") != Some(&Value::Bool(true)) {
        push_reason(
            &mut reasons,
            "container CPU evidence is not valid for performance comparison",
        );
    }
    let window_start_ns = nonnegative_integer(cpu.get("window_start_ns"));
    let window_end_ns = nonnegative_integer(cpu.get("window_end_ns"));
    let cpu_usage_sec = nonnegative_number(cpu.get("usage_sec"));
    let average_logical_cores = nonnegative_number(cpu.get("average_logical_cores"));

    if let Some((expected_cpus, expected_cpu_count)) = expected_cpuset {
        if cpu.get("allocated_cpu_source").and_then(Value::as_str) != Some("cpuset-cpus") {
            push_reason(
                &mut reasons,
                "container CPU allocation is not sourced from cpuset-cpus",
            );
        }
        if positive_number(cpu.get("allocated_logical_cpus")) != Some(expected_cpu_count as f64) {
            push_reason(
                &mut reasons,
                "container CPU allocation width differs from the requested CPU set",
            );
        }
        let extension_cpus = report
            .get("execution_plan")
            .and_then(Value::as_object)
            .and_then(|plan| plan.get("extensions"))
            .and_then(Value::as_object);
        match extension_cpus {
            None => push_reason(&mut reasons, "execution plan extensions are missing"),
            Some(extensions)
                if extensions.get("cpuset_cpus").and_then(Value::as_str) != Some(expected_cpus) =>
            {
                push_reason(
                    &mut reasons,
                    "execution plan CPU set differs from the requested CPU set",
                )
            }
            _ => {}
        }
        let placement = report
            .get("scope")
            .and_then(Value::as_object)
            .and_then(|scope| scope.get("identity"))
            .and_then(Value::as_object)
            .and_then(|identity| identity.get("cpu_placement"))
            .and_then(Value::as_object);
        let expected_ids = cpuset_ids(expected_cpus);
        match placement {
            None => push_reason(
                &mut reasons,
                "observed runtime/kernel CPU placement evidence is missing",
            ),
            Some(placement) => {
                if placement.get("validation_status").and_then(Value::as_str) != Some("passed") {
                    push_reason(
                        &mut reasons,
                        "observed runtime/kernel CPU placement did not pass validation",
                    );
                }
                for (field, description) in [
                    ("requested_cpuset_cpus", "retained requested cpuset"),
                    ("docker_host_config_cpuset_cpus", "Docker HostConfig cpuset"),
                    (
                        "container_init_cpus_allowed_list",
                        "container init allowed CPU list",
                    ),
                    ("cgroup_cpuset_cpus_effective", "effective cgroup cpuset"),
                ] {
                    let observed = placement
                        .get(field)
                        .and_then(Value::as_str)
                        .and_then(cpuset_ids);
                    if observed.is_none() || observed != expected_ids {
                        push_reason(
                            &mut reasons,
                            format!("{description} differs from the requested CPU set"),
                        );
                    }
                }
            }
        }
    }

    if let Some((spec, _launch)) = ordered {
        if report.get("pacing").and_then(Value::as_str) != Some(spec.defaults.pacing.as_str()) {
            push_reason(&mut reasons, "profile pacing differs from defaults.pacing");
        }
        match report.get("profile_provenance").and_then(Value::as_object) {
            None => push_reason(&mut reasons, "profile provenance is missing"),
            Some(provenance)
                if json_string_array(provenance.get("profile_sets")).as_deref()
                    != Some(spec.defaults.profile_sets.as_slice()) =>
            {
                push_reason(
                    &mut reasons,
                    "profile sets differ from defaults.profile_sets",
                )
            }
            _ => {}
        }
        let expected_network =
            NetworkPolicy::parse(&spec.defaults.network_policy, &spec.defaults.allowed_hosts)
                .expect("strict run spec already validated its network policy")
                .comparability_digest();
        match report.get("network_policy").and_then(Value::as_object) {
            None => push_reason(&mut reasons, "network policy evidence is missing"),
            Some(policy)
                if policy.get("comparability_sha256").and_then(Value::as_str)
                    != Some(expected_network.as_str()) =>
            {
                push_reason(
                    &mut reasons,
                    "network policy differs from defaults.network_policy",
                )
            }
            _ => {}
        }
    }

    if require_common_window {
        if !canonical_clock_reference_matches(report) {
            push_reason(
                &mut reasons,
                "canonical clock reference is missing or differs from the runner CLOCK_MONOTONIC domain",
            );
        }
        if window_start_ns.is_none() {
            push_reason(
                &mut reasons,
                "container_cpu.window_start_ns is missing or invalid",
            );
        }
        if window_end_ns.is_none() {
            push_reason(
                &mut reasons,
                "container_cpu.window_end_ns is missing or invalid",
            );
        }
        if matches!((window_start_ns, window_end_ns), (Some(start), Some(end)) if end <= start) {
            push_reason(
                &mut reasons,
                "container CPU common-window timestamps are not increasing",
            );
        }
        if cpu_usage_sec.is_none() {
            push_reason(
                &mut reasons,
                "container_cpu.usage_sec is missing or invalid",
            );
        }
        if average_logical_cores.is_none() {
            push_reason(
                &mut reasons,
                "container_cpu.average_logical_cores is missing or invalid",
            );
        }
        if let (Some(duration), Some(start), Some(end)) = (duration, window_start_ns, window_end_ns)
        {
            if !close((end - start) as f64 / 1_000_000_000.0, duration) {
                push_reason(
                    &mut reasons,
                    "container CPU wall time disagrees with common-window timestamps",
                );
            }
        }
        if let (Some(duration), Some(usage), Some(average)) =
            (duration, cpu_usage_sec, average_logical_cores)
        {
            if !close(usage / duration, average) {
                push_reason(
                    &mut reasons,
                    "container CPU average cores disagrees with CPU/wall time",
                );
            }
        }
    }

    let replay = match report.get("replay_summary").and_then(Value::as_object) {
        Some(replay) => replay,
        None => {
            push_reason(&mut reasons, "replay summary is missing");
            static EMPTY: std::sync::OnceLock<Map<String, Value>> = std::sync::OnceLock::new();
            EMPTY.get_or_init(Map::new)
        }
    };
    push_incomplete_replay_reason(Some(replay), &mut reasons);
    match nonnegative_integer(replay.get("actions_timed_out")) {
        None => push_reason(&mut reasons, "replay timeout count is missing or invalid"),
        Some(count) if count > 0 => {
            push_reason(&mut reasons, format!("{count} replay action(s) timed out"))
        }
        _ => {}
    }
    match nonnegative_integer(replay.get("source_outcome_mismatch_count")) {
        None => push_reason(
            &mut reasons,
            "replay outcome-mismatch count is missing or invalid",
        ),
        Some(count) if count > 0 => push_reason(
            &mut reasons,
            format!("{count} replay source-outcome mismatch(es)"),
        ),
        _ => {}
    }
    if report.get("profiling_valid") != Some(&Value::Bool(true)) {
        push_reason(&mut reasons, "profiling evidence is not explicitly valid");
    }
    if report.get("status").and_then(Value::as_str) != Some("valid") {
        push_reason(&mut reasons, "pipeline status is not valid");
    }
    if report
        .get("source_trajectory_sha256")
        .and_then(Value::as_str)
        != Some(task.source_sha256.as_str())
    {
        push_reason(
            &mut reasons,
            "profile source digest differs from the sealed ATIF",
        );
    }
    let execution_plan_digest = sha256_digest(report.get("execution_plan_digest"));
    if execution_plan_digest.is_none() {
        push_reason(&mut reasons, "execution plan digest is missing or invalid");
    }
    let observed_image_id = image_id(report);
    let retained_image_id = observed_image_id
        .clone()
        .unwrap_or_else(|| preflight.image_id.clone());
    if observed_image_id.is_none() {
        push_reason(&mut reasons, "profile did not retain an observed image ID");
    } else if observed_image_id.as_deref() != Some(preflight.image_id.as_str()) {
        push_reason(&mut reasons, "profile image ID differs from preflight");
    }
    if task.expected_image_id.as_deref().is_some()
        && task.expected_image_id.as_deref() != Some(retained_image_id.as_str())
    {
        push_reason(
            &mut reasons,
            "profile image ID differs from expected image ID",
        );
    }
    if let Some(expected_controller) = preflight.controller_sha256.as_deref() {
        let observed_controller = report
            .get("controller_identity")
            .and_then(Value::as_object)
            .and_then(|identity| sha256_digest(identity.get("sha256")));
        if observed_controller.as_deref() != Some(expected_controller) {
            push_reason(
                &mut reasons,
                "Rust controller content identity differs from preflight",
            );
        }
        let planned_controller = report
            .get("execution_plan")
            .and_then(Value::as_object)
            .and_then(|plan| plan.get("source_identity"))
            .and_then(Value::as_object)
            .and_then(|identity| sha256_digest(identity.get("rust_controller_binary_sha256")));
        if planned_controller.as_deref() != Some(expected_controller) {
            push_reason(
                &mut reasons,
                "execution plan Rust controller identity differs from preflight",
            );
        }
    }
    ProjectedDuration {
        duration_sec: duration,
        valid: duration.is_some() && reasons.is_empty(),
        reason: reasons.join("; "),
        image_id: retained_image_id,
        execution_plan_digest,
        window_start_ns,
        window_end_ns,
        cpu_usage_sec,
        average_logical_cores,
        cleanup_valid: None,
        clock_reference: None,
    }
}

fn number_value(value: Option<f64>) -> Value {
    value
        .and_then(Number::from_f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn u64_value(value: Option<u64>) -> Value {
    value
        .map(|value| Value::Number(Number::from(value)))
        .unwrap_or(Value::Null)
}

fn failed_row(
    task: &RunTaskSpec,
    repetition: usize,
    profile_dir: String,
    image_id: &str,
    reason: impl Into<String>,
    attempt_index: Option<usize>,
    ordered_attempt: Option<&WaveAttempt>,
    enroot: bool,
) -> Map<String, Value> {
    let mut row = Map::new();
    row.insert("task_id".to_owned(), Value::String(task.task_id.clone()));
    row.insert(
        "repetition".to_owned(),
        Value::Number(Number::from(repetition as u64)),
    );
    row.insert("status".to_owned(), Value::String("failed".to_owned()));
    row.insert("duration_sec".to_owned(), Value::Null);
    row.insert("duration_valid".to_owned(), Value::Bool(false));
    row.insert("reason".to_owned(), Value::String(reason.into()));
    row.insert("profile_dir".to_owned(), Value::String(profile_dir));
    row.insert(
        "source_sha256".to_owned(),
        Value::String(task.source_sha256.clone()),
    );
    row.insert("image_id".to_owned(), Value::String(image_id.to_owned()));
    row.insert("execution_plan_digest".to_owned(), Value::Null);
    if let Some(attempt_index) = attempt_index {
        row.insert(
            "attempt_index".to_owned(),
            Value::Number(Number::from(attempt_index as u64)),
        );
        row.insert("completion_index".to_owned(), Value::Null);
        row.insert("window_start_ns".to_owned(), Value::Null);
        row.insert("window_end_ns".to_owned(), Value::Null);
        row.insert("cpu_usage_sec".to_owned(), Value::Null);
        row.insert("average_logical_cores".to_owned(), Value::Null);
    }
    if let Some(attempt) = ordered_attempt {
        row.insert(
            "wave_index".to_owned(),
            Value::Number(Number::from(attempt.wave_index as u64)),
        );
        row.insert(
            "wave_id".to_owned(),
            Value::String(attempt.wave.wave_id.clone()),
        );
        row.insert("wave_scored".to_owned(), Value::Bool(attempt.wave.scored));
        row.insert(
            "launch_index".to_owned(),
            Value::Number(Number::from(attempt.launch_index as u64)),
        );
        row.insert(
            "cpus".to_owned(),
            Value::String(attempt.launch.cpus.clone()),
        );
    }
    if enroot {
        row.insert("cleanup_valid".to_owned(), Value::Null);
        row.insert("clock_reference".to_owned(), Value::Null);
    }
    row
}

fn attempt_profile_path(output: &Path, attempt: &Attempt) -> PathBuf {
    output
        .join("tasks")
        .join(&attempt.task.task_id)
        .join(format!("rep-{:04}", attempt.repetition))
        .join("profile")
}

fn wave_profile_path(spec: &RunSpecV3, attempt: &WaveAttempt) -> PathBuf {
    spec.output
        .join("waves")
        .join(format!(
            "{:04}-{}",
            attempt.wave_index, attempt.wave.wave_id
        ))
        .join(format!(
            "launch-{:04}-{}",
            attempt.launch_index, attempt.task.task_id
        ))
        .join("profile")
}

fn relative_profile(output: &Path, profile: &Path) -> String {
    profile
        .strip_prefix(output)
        .expect("profile paths are constructed beneath output")
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn attempts_v1(spec: &RunSpecV1) -> Vec<Attempt> {
    let mut attempts = Vec::with_capacity(spec.tasks.len() * spec.repetitions);
    for repetition in 1..=spec.repetitions {
        for task in &spec.tasks {
            attempts.push(Attempt {
                attempt_index: attempts.len() + 1,
                repetition,
                task: task.clone(),
            });
        }
    }
    attempts
}

fn attempts_v2(spec: &RunSpecV2) -> Vec<Attempt> {
    let mut attempts = Vec::with_capacity(spec.tasks.len() * spec.repetitions);
    for repetition in 1..=spec.repetitions {
        for task in &spec.tasks {
            attempts.push(Attempt {
                attempt_index: attempts.len() + 1,
                repetition,
                task: task.clone(),
            });
        }
    }
    attempts
}

fn wave_attempts(spec: &RunSpecV3) -> Vec<WaveAttempt> {
    let task_by_id: HashMap<&str, &RunTaskSpec> = spec
        .tasks
        .iter()
        .map(|task| (task.task_id.as_str(), task))
        .collect();
    let mut attempts = Vec::new();
    for (wave_offset, wave) in spec.waves.iter().enumerate() {
        for (launch_offset, launch) in wave.launches.iter().enumerate() {
            attempts.push(WaveAttempt {
                attempt_index: attempts.len() + 1,
                wave_index: wave_offset + 1,
                launch_index: launch_offset + 1,
                wave: wave.clone(),
                launch: launch.clone(),
                task: (*task_by_id[launch.task_id.as_str()]).clone(),
            });
        }
    }
    attempts
}

fn run_attempt_v1(
    spec: &RunSpecV1,
    attempt: &Attempt,
    preflight: &Preflight,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Map<String, Value> {
    let profile = attempt_profile_path(&spec.output, attempt);
    run_attempt_common(
        &attempt.task,
        attempt.repetition,
        relative_profile(&spec.output, &profile),
        profile.clone(),
        preflight,
        None,
        None,
        None,
        false,
        launcher,
        cancellation,
        v1_request(spec, &attempt.task, profile),
    )
}

fn run_attempt_v2(
    spec: &RunSpecV2,
    attempt: &Attempt,
    preflight: &Preflight,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Map<String, Value> {
    let profile = attempt_profile_path(&spec.output, attempt);
    run_attempt_common(
        &attempt.task,
        attempt.repetition,
        relative_profile(&spec.output, &profile),
        profile.clone(),
        preflight,
        Some(attempt.attempt_index),
        None,
        Some(spec.cpu_ids.len()),
        false,
        launcher,
        cancellation,
        v2_request(spec, &attempt.task, profile),
    )
}

fn run_attempt_v3(
    spec: &RunSpecV3,
    attempt: &WaveAttempt,
    preflight: &Preflight,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Map<String, Value> {
    let profile = wave_profile_path(spec, attempt);
    run_attempt_common(
        &attempt.task,
        1,
        relative_profile(&spec.output, &profile),
        profile.clone(),
        preflight,
        Some(attempt.attempt_index),
        Some((spec, attempt)),
        Some(attempt.launch.cpu_ids.len()),
        launcher.enroot_wave_start_gate_enabled(),
        launcher,
        cancellation,
        v3_request(
            spec,
            &attempt.task,
            &attempt.launch.cpus,
            profile,
            preflight.enroot_executable.as_deref(),
        ),
    )
}

#[allow(clippy::too_many_arguments)]
fn run_attempt_common(
    task: &RunTaskSpec,
    repetition: usize,
    profile_relative: String,
    _profile: PathBuf,
    preflight: &Preflight,
    attempt_index: Option<usize>,
    ordered: Option<(&RunSpecV3, &WaveAttempt)>,
    expected_cpu_count: Option<usize>,
    require_enroot_wave_start_gate: bool,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
    request: ProfileRequest,
) -> Map<String, Value> {
    if cancellation.is_cancelled() {
        return failed_row(
            task,
            repetition,
            profile_relative,
            &preflight.image_id,
            "run interrupted before attempt start",
            attempt_index,
            ordered.map(|(_, attempt)| attempt),
            request.runtime == "enroot",
        );
    }
    let current_digest = match file_sha256(&task.atif) {
        Ok(digest) => digest,
        Err(error) => {
            return failed_row(
                task,
                repetition,
                profile_relative,
                &preflight.image_id,
                format!("unable to re-read ATIF source: {error}"),
                attempt_index,
                ordered.map(|(_, attempt)| attempt),
                request.runtime == "enroot",
            )
        }
    };
    if current_digest != task.source_sha256 {
        return failed_row(
            task,
            repetition,
            profile_relative,
            &preflight.image_id,
            format!(
                "ATIF source changed after the run file was loaded: expected {}, got {}",
                task.source_sha256, current_digest
            ),
            attempt_index,
            ordered.map(|(_, attempt)| attempt),
            request.runtime == "enroot",
        );
    }
    let report = match launcher.run_profile(&request, cancellation) {
        Ok(report) => report,
        Err(error) => {
            return failed_row(
                task,
                repetition,
                profile_relative,
                &preflight.image_id,
                format!("profile failed: {error}"),
                attempt_index,
                ordered.map(|(_, attempt)| attempt),
                request.runtime == "enroot",
            )
        }
    };
    let Some(report) = report.as_object() else {
        return failed_row(
            task,
            repetition,
            profile_relative,
            &preflight.image_id,
            "profile returned a non-object report",
            attempt_index,
            ordered.map(|(_, attempt)| attempt),
            request.runtime == "enroot",
        );
    };
    let projection = project_duration(
        report,
        task,
        preflight,
        attempt_index.is_some(),
        expected_cpu_count.map(|count| (request.cpuset_cpus.as_str(), count)),
        ordered.map(|(spec, attempt)| (spec, &attempt.launch)),
        require_enroot_wave_start_gate,
    );
    let mut row = Map::new();
    row.insert("task_id".to_owned(), Value::String(task.task_id.clone()));
    row.insert(
        "repetition".to_owned(),
        Value::Number(Number::from(repetition as u64)),
    );
    row.insert("status".to_owned(), Value::String("completed".to_owned()));
    row.insert(
        "duration_sec".to_owned(),
        number_value(projection.duration_sec),
    );
    row.insert("duration_valid".to_owned(), Value::Bool(projection.valid));
    row.insert("reason".to_owned(), Value::String(projection.reason));
    row.insert("profile_dir".to_owned(), Value::String(profile_relative));
    row.insert(
        "source_sha256".to_owned(),
        Value::String(task.source_sha256.clone()),
    );
    row.insert("image_id".to_owned(), Value::String(projection.image_id));
    row.insert(
        "execution_plan_digest".to_owned(),
        projection
            .execution_plan_digest
            .map(Value::String)
            .unwrap_or(Value::Null),
    );
    if let Some(attempt_index) = attempt_index {
        row.insert(
            "attempt_index".to_owned(),
            Value::Number(Number::from(attempt_index as u64)),
        );
        row.insert("completion_index".to_owned(), Value::Null);
        row.insert(
            "window_start_ns".to_owned(),
            u64_value(projection.window_start_ns),
        );
        row.insert(
            "window_end_ns".to_owned(),
            u64_value(projection.window_end_ns),
        );
        row.insert(
            "cpu_usage_sec".to_owned(),
            number_value(projection.cpu_usage_sec),
        );
        row.insert(
            "average_logical_cores".to_owned(),
            number_value(projection.average_logical_cores),
        );
    }
    if request.runtime == "enroot" {
        row.insert(
            "cleanup_valid".to_owned(),
            projection
                .cleanup_valid
                .map(Value::Bool)
                .unwrap_or(Value::Null),
        );
        row.insert(
            "clock_reference".to_owned(),
            projection.clock_reference.unwrap_or(Value::Null),
        );
    }
    if let Some((_, attempt)) = ordered {
        row.insert(
            "wave_index".to_owned(),
            Value::Number(Number::from(attempt.wave_index as u64)),
        );
        row.insert(
            "wave_id".to_owned(),
            Value::String(attempt.wave.wave_id.clone()),
        );
        row.insert("wave_scored".to_owned(), Value::Bool(attempt.wave.scored));
        row.insert(
            "launch_index".to_owned(),
            Value::Number(Number::from(attempt.launch_index as u64)),
        );
        row.insert(
            "cpus".to_owned(),
            Value::String(attempt.launch.cpus.clone()),
        );
    }
    row
}

#[cfg(target_os = "linux")]
fn current_affinity() -> Result<Vec<u32>, RunExecutionError> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let result = unsafe {
        libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set as *mut libc::cpu_set_t,
        )
    };
    if result != 0 {
        return Err(RunExecutionError::new(format!(
            "unable to inspect controller CPU affinity: {}",
            io::Error::last_os_error()
        )));
    }
    let mut cpus = Vec::new();
    for cpu in 0..libc::CPU_SETSIZE as usize {
        if unsafe { libc::CPU_ISSET(cpu, &set) } {
            cpus.push(cpu as u32);
        }
    }
    Ok(cpus)
}

#[cfg(not(target_os = "linux"))]
fn current_affinity() -> Result<Vec<u32>, RunExecutionError> {
    Err(RunExecutionError::new(
        "controller CPU affinity requires Linux",
    ))
}

#[cfg(target_os = "linux")]
fn mutate_current_thread_affinity(cpu_ids: &[u32]) -> Result<(), RunExecutionError> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    unsafe { libc::CPU_ZERO(&mut set) };
    for cpu in cpu_ids {
        if *cpu as usize >= libc::CPU_SETSIZE as usize {
            return Err(RunExecutionError::new(format!(
                "unable to enforce controller CPU affinity: logical CPU {cpu} exceeds CPU_SETSIZE"
            )));
        }
        unsafe { libc::CPU_SET(*cpu as usize, &mut set) };
    }
    let result = unsafe {
        libc::sched_setaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &set as *const libc::cpu_set_t,
        )
    };
    if result != 0 {
        return Err(RunExecutionError::new(format!(
            "unable to enforce controller CPU affinity: {}",
            io::Error::last_os_error()
        )));
    }
    Ok(())
}

#[cfg(not(target_os = "linux"))]
fn mutate_current_thread_affinity(_cpu_ids: &[u32]) -> Result<(), RunExecutionError> {
    Err(RunExecutionError::new(
        "controller CPU affinity requires Linux",
    ))
}

fn set_current_thread_affinity(cpu_ids: &[u32]) -> Result<Vec<u32>, RunExecutionError> {
    mutate_current_thread_affinity(cpu_ids)?;
    let observed = current_affinity()?;
    if observed != cpu_ids {
        return Err(RunExecutionError::new(format!(
            "controller CPU affinity differs after enforcement: requested={cpu_ids:?}, observed={observed:?}"
        )));
    }
    Ok(observed)
}

struct AffinityRestoreGuard {
    previous: Vec<u32>,
    restored: bool,
}

impl AffinityRestoreGuard {
    fn new(previous: Vec<u32>) -> Self {
        Self {
            previous,
            restored: false,
        }
    }

    fn restore(&mut self) -> Result<(), RunExecutionError> {
        let restored = set_current_thread_affinity(&self.previous)?;
        if restored != self.previous {
            return Err(RunExecutionError::new(format!(
                "caller CPU affinity differs after restoration: expected={:?}, observed={restored:?}",
                self.previous
            )));
        }
        self.restored = true;
        Ok(())
    }
}

impl Drop for AffinityRestoreGuard {
    fn drop(&mut self) {
        if !self.restored {
            let _ = set_current_thread_affinity(&self.previous);
        }
    }
}

fn with_controller_affinity<T>(
    spec: &RunSpecV3,
    operation: impl FnOnce(&mut Map<String, Value>) -> Result<T, RunExecutionError>,
) -> Result<(T, Value), RunExecutionError> {
    let previous = current_affinity()?;
    let previous_set: HashSet<u32> = previous.iter().copied().collect();
    let missing: Vec<u32> = spec
        .defaults
        .controller_cpu_ids
        .iter()
        .copied()
        .filter(|cpu| !previous_set.contains(cpu))
        .collect();
    if !missing.is_empty() {
        return Err(RunExecutionError::new(format!(
            "controller CPU set is outside the caller's available affinity: {}",
            missing
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    if spec.defaults.runtime == "enroot" {
        let unavailable_launch_cpus = spec
            .waves
            .iter()
            .flat_map(|wave| &wave.launches)
            .flat_map(|launch| launch.cpu_ids.iter().copied())
            .filter(|cpu| !previous_set.contains(cpu))
            .collect::<BTreeSet<_>>();
        if !unavailable_launch_cpus.is_empty() {
            return Err(RunExecutionError::new(format!(
                "requested Enroot launch CPU set is outside the caller's available affinity: {}",
                unavailable_launch_cpus
                    .iter()
                    .map(u32::to_string)
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
    }
    // Install the restoration guard immediately after the first successful
    // affinity mutation. Even inspection/verification failures below must not
    // strand the caller on the controller CPU set.
    mutate_current_thread_affinity(&spec.defaults.controller_cpu_ids)?;
    let mut restore_guard = AffinityRestoreGuard::new(previous.clone());
    let mut evidence = Map::new();
    evidence.insert(
        "requested_cpus".to_owned(),
        Value::String(spec.defaults.controller_cpus.clone()),
    );
    evidence.insert(
        "requested_logical_cpu_ids".to_owned(),
        json!(spec.defaults.controller_cpu_ids),
    );
    evidence.insert("caller_affinity_before".to_owned(), json!(previous));
    evidence.insert("caller_affinity_restored".to_owned(), Value::Bool(false));
    evidence.insert(
        "scope".to_owned(),
        Value::String(if spec.defaults.runtime == "enroot" {
            "host runner threads and inherited Enroot launch helpers; not the taskset-pinned in-container replay controller or kernel work".to_owned()
        } else {
            "host runner threads and inherited Docker-client helpers; not the in-container replay controller, Docker daemon, or kernel work".to_owned()
        }),
    );

    let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let observed = current_affinity()?;
        if observed != spec.defaults.controller_cpu_ids {
            return Err(RunExecutionError::new(format!(
                "controller CPU affinity differs after enforcement: requested={:?}, observed={observed:?}",
                spec.defaults.controller_cpu_ids
            )));
        }
        evidence.insert(
            "observed_main_thread_logical_cpu_ids".to_owned(),
            json!(observed),
        );
        operation(&mut evidence)
    }));
    let restore = restore_guard.restore();
    match outcome {
        Ok(Ok(value)) => {
            restore?;
            evidence.insert("caller_affinity_restored".to_owned(), Value::Bool(true));
            Ok((value, Value::Object(evidence)))
        }
        Ok(Err(operation_error)) => match restore {
            Ok(()) => Err(operation_error),
            Err(restore_error) => Err(RunExecutionError::new(format!(
                "{operation_error}; caller CPU affinity restoration failed: {restore_error}"
            ))),
        },
        Err(payload) => {
            if let Err(error) = restore {
                eprintln!("caller CPU affinity restoration failed while unwinding: {error}");
            }
            std::panic::resume_unwind(payload)
        }
    }
}

fn set_completion_index(row: &mut Map<String, Value>, completion_index: usize) {
    row.insert(
        "completion_index".to_owned(),
        Value::Number(Number::from(completion_index as u64)),
    );
}

fn execute_v1(
    spec: &RunSpecV1,
    attempts: &[Attempt],
    preflights: &HashMap<String, Preflight>,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> (Vec<Map<String, Value>>, bool) {
    let mut rows = Vec::with_capacity(attempts.len());
    for attempt in attempts {
        if cancellation.is_cancelled() {
            let profile = attempt_profile_path(&spec.output, attempt);
            rows.push(failed_row(
                &attempt.task,
                attempt.repetition,
                relative_profile(&spec.output, &profile),
                &preflights[&attempt.task.task_id].image_id,
                "run interrupted before attempt dispatch",
                None,
                None,
                false,
            ));
            continue;
        }
        rows.push(run_attempt_v1(
            spec,
            attempt,
            &preflights[&attempt.task.task_id],
            launcher,
            cancellation,
        ));
    }
    (rows, cancellation.is_cancelled())
}

fn execute_v2(
    spec: &RunSpecV2,
    attempts: &[Attempt],
    preflights: &HashMap<String, Preflight>,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> (Vec<Map<String, Value>>, bool) {
    let queue = Arc::new(Mutex::new(VecDeque::from(attempts.to_vec())));
    let started = Arc::new(Mutex::new(HashSet::<usize>::new()));
    let completion = Arc::new(AtomicUsize::new(0));
    let (sender, receiver) = mpsc::channel::<(usize, Map<String, Value>)>();
    thread::scope(|scope| {
        for _ in 0..spec.concurrency {
            let queue = Arc::clone(&queue);
            let started = Arc::clone(&started);
            let completion = Arc::clone(&completion);
            let sender = sender.clone();
            let cancellation = cancellation.clone();
            scope.spawn(move || loop {
                if cancellation.is_cancelled() {
                    break;
                }
                let attempt = {
                    let mut queue = queue.lock().expect("attempt queue lock poisoned");
                    if cancellation.is_cancelled() {
                        None
                    } else {
                        queue.pop_front()
                    }
                };
                let Some(attempt) = attempt else {
                    break;
                };
                started
                    .lock()
                    .expect("started-attempt lock poisoned")
                    .insert(attempt.attempt_index);
                let row = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    run_attempt_v2(
                        spec,
                        &attempt,
                        &preflights[&attempt.task.task_id],
                        launcher,
                        &cancellation,
                    )
                }))
                .unwrap_or_else(|_| {
                    let profile = attempt_profile_path(&spec.output, &attempt);
                    failed_row(
                        &attempt.task,
                        attempt.repetition,
                        relative_profile(&spec.output, &profile),
                        &preflights[&attempt.task.task_id].image_id,
                        if cancellation.is_cancelled() {
                            "profile interrupted: profile worker panicked"
                        } else {
                            "profile worker failed: profile worker panicked"
                        },
                        Some(attempt.attempt_index),
                        None,
                        false,
                    )
                });
                let index = completion.fetch_add(1, Ordering::SeqCst) + 1;
                let mut row = row;
                set_completion_index(&mut row, index);
                if sender.send((attempt.attempt_index, row)).is_err() {
                    break;
                }
            });
        }
        drop(sender);
    });

    let mut rows_by_attempt: BTreeMap<usize, Map<String, Value>> = receiver.into_iter().collect();
    let interrupted = cancellation.is_cancelled();
    let started = started.lock().expect("started-attempt lock poisoned");
    for attempt in attempts {
        if rows_by_attempt.contains_key(&attempt.attempt_index) {
            continue;
        }
        let profile = attempt_profile_path(&spec.output, attempt);
        let reason = if interrupted && started.contains(&attempt.attempt_index) {
            "run interrupted before attempt start"
        } else if interrupted {
            "run interrupted before attempt dispatch"
        } else {
            "profile worker failed without returning a row"
        };
        rows_by_attempt.insert(
            attempt.attempt_index,
            failed_row(
                &attempt.task,
                attempt.repetition,
                relative_profile(&spec.output, &profile),
                &preflights[&attempt.task.task_id].image_id,
                reason,
                Some(attempt.attempt_index),
                None,
                false,
            ),
        );
    }
    (rows_by_attempt.into_values().collect(), interrupted)
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EnrootWaveAttestationPhase {
    Before,
    After,
}

impl EnrootWaveAttestationPhase {
    fn label(self) -> &'static str {
        match self {
            Self::Before => "before",
            Self::After => "after",
        }
    }
}

type EnrootWaveReattestor<'a> = dyn FnMut(usize, EnrootWaveAttestationPhase) -> bool + 'a;
type V3Execution = (
    Vec<Map<String, Value>>,
    bool,
    HashMap<usize, HostWindow>,
    Option<String>,
);

fn wave_attestation_failure_reason(wave_index: usize, phase: EnrootWaveAttestationPhase) -> String {
    format!(
        "Enroot runtime provenance re-attestation failed {} wave {wave_index}",
        phase.label()
    )
}

fn wave_start_control_path(spec: &RunSpecV3, attempt: &WaveAttempt, name: &str) -> PathBuf {
    wave_profile_path(spec, attempt)
        .join("container_profile")
        .join(name)
}

fn regular_file_has_exact_content(path: &Path, expected: &[u8]) -> Result<bool, RunExecutionError> {
    let mut file = match OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
    {
        Ok(file) => file,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => {
            return Err(RunExecutionError::new(format!(
                "unable to safely open wave-start control {}: {error}",
                path.display()
            )))
        }
    };
    let before = file.metadata().map_err(|error| {
        RunExecutionError::new(format!(
            "unable to inspect wave-start control {}: {error}",
            path.display()
        ))
    })?;
    if !before.is_file() || before.nlink() != 1 {
        return Err(RunExecutionError::new(format!(
            "wave-start control is not a single-link regular file: {}",
            path.display()
        )));
    }
    if before.len() != expected.len() as u64 {
        return Err(RunExecutionError::new(format!(
            "wave-start control has invalid content length: {}",
            path.display()
        )));
    }
    let mut observed = Vec::with_capacity(expected.len() + 1);
    (&mut file)
        .take(expected.len() as u64 + 1)
        .read_to_end(&mut observed)
        .map_err(|error| {
            RunExecutionError::new(format!(
                "unable to read wave-start control {}: {error}",
                path.display()
            ))
        })?;
    let after = file.metadata().map_err(|error| {
        RunExecutionError::new(format!(
            "unable to re-inspect wave-start control {}: {error}",
            path.display()
        ))
    })?;
    if before.dev() != after.dev()
        || before.ino() != after.ino()
        || before.len() != after.len()
        || before.mtime() != after.mtime()
        || before.mtime_nsec() != after.mtime_nsec()
        || before.ctime() != after.ctime()
        || before.ctime_nsec() != after.ctime_nsec()
    {
        return Err(RunExecutionError::new(format!(
            "wave-start control changed while it was read: {}",
            path.display()
        )));
    }
    if observed != expected {
        return Err(RunExecutionError::new(format!(
            "wave-start control has invalid content: {}",
            path.display()
        )));
    }
    Ok(true)
}

fn publish_wave_start_release(path: &Path) -> Result<(), RunExecutionError> {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .custom_flags(libc::O_NOFOLLOW | libc::O_CLOEXEC)
        .open(path)
        .map_err(|error| {
            RunExecutionError::new(format!(
                "unable to publish wave-start release {}: {error}",
                path.display()
            ))
        })?;
    file.write_all(RUN_WAVE_START_RELEASE_BYTES)
        .map_err(|error| {
            RunExecutionError::new(format!(
                "unable to write wave-start release {}: {error}",
                path.display()
            ))
        })?;
    Ok(())
}

fn coordinate_enroot_wave_start(
    spec: &RunSpecV3,
    attempts: &[WaveAttempt],
    receiver: &mpsc::Receiver<(usize, Map<String, Value>)>,
    completed_before_release: &mut Vec<(usize, Map<String, Value>)>,
    cancellation: &CancellationToken,
    timeout: Duration,
) -> Result<(), RunExecutionError> {
    let deadline = Instant::now() + timeout;
    let mut ready = BTreeSet::new();
    loop {
        match receiver.try_recv() {
            Ok(row) => {
                let attempt_index = row.0;
                completed_before_release.push(row);
                return Err(RunExecutionError::new(format!(
                    "attempt {attempt_index} completed before the wave-start barrier released"
                )));
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                return Err(RunExecutionError::new(
                    "profile workers disconnected before the wave-start barrier released",
                ))
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
        if cancellation.is_cancelled() {
            return Err(RunExecutionError::new(
                "run interrupted while waiting for the Enroot wave-start barrier",
            ));
        }
        for attempt in attempts {
            if ready.contains(&attempt.attempt_index) {
                continue;
            }
            let release = wave_start_control_path(spec, attempt, RUN_WAVE_START_RELEASE);
            match fs::symlink_metadata(&release) {
                Ok(_) => {
                    return Err(RunExecutionError::new(format!(
                        "wave-start release existed before all attempts were ready: {}",
                        release.display()
                    )))
                }
                Err(error) if error.kind() == io::ErrorKind::NotFound => {}
                Err(error) => {
                    return Err(RunExecutionError::new(format!(
                        "unable to verify wave-start release absence {}: {error}",
                        release.display()
                    )))
                }
            }
            let marker = wave_start_control_path(spec, attempt, RUN_WAVE_START_READY);
            if regular_file_has_exact_content(&marker, RUN_WAVE_START_READY_BYTES)? {
                ready.insert(attempt.attempt_index);
            }
        }
        if ready.len() == attempts.len() {
            break;
        }
        if Instant::now() >= deadline {
            let pending = attempts
                .iter()
                .filter(|attempt| !ready.contains(&attempt.attempt_index))
                .map(|attempt| {
                    format!(
                        "attempt={} launch={} task={} path={}",
                        attempt.attempt_index,
                        attempt.launch_index,
                        attempt.task.task_id,
                        wave_start_control_path(spec, attempt, RUN_WAVE_START_READY).display()
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(RunExecutionError::new(format!(
                "timed out after {:.3} seconds waiting for Enroot wave {} readiness ({}/{} ready): {pending}",
                timeout.as_secs_f64(),
                attempts.first().map_or(0, |attempt| attempt.wave_index),
                ready.len(),
                attempts.len(),
            )));
        }
        thread::sleep(RUN_WAVE_START_POLL);
    }

    match receiver.try_recv() {
        Ok(row) => {
            let attempt_index = row.0;
            completed_before_release.push(row);
            return Err(RunExecutionError::new(format!(
                "attempt {attempt_index} completed after readiness but before wave-start release"
            )));
        }
        Err(mpsc::TryRecvError::Disconnected) => {
            return Err(RunExecutionError::new(
                "profile workers disconnected after readiness but before wave-start release",
            ))
        }
        Err(mpsc::TryRecvError::Empty) => {}
    }
    if cancellation.is_cancelled() {
        return Err(RunExecutionError::new(
            "run interrupted after Enroot wave readiness and before release",
        ));
    }

    // Once release publication starts, finish publishing every marker. This
    // avoids stranding a subset of attempts if cancellation races the small
    // sequential publication window; the shared cancellation path will then
    // terminate every owned Enroot process group.
    for attempt in attempts {
        publish_wave_start_release(&wave_start_control_path(
            spec,
            attempt,
            RUN_WAVE_START_RELEASE,
        ))?;
    }
    for attempt in attempts {
        let path = wave_start_control_path(spec, attempt, RUN_WAVE_START_RELEASE);
        if !regular_file_has_exact_content(&path, RUN_WAVE_START_RELEASE_BYTES)? {
            return Err(RunExecutionError::new(format!(
                "wave-start release disappeared after publication: {}",
                path.display()
            )));
        }
    }
    Ok(())
}

fn execute_v3(
    spec: &RunSpecV3,
    attempts: &[WaveAttempt],
    preflights: &HashMap<(String, String), Preflight>,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
    mut enroot_reattest: Option<&mut EnrootWaveReattestor<'_>>,
) -> Result<V3Execution, RunExecutionError> {
    let mut rows_by_attempt = BTreeMap::<usize, Map<String, Value>>::new();
    let completion = AtomicUsize::new(0);
    let mut wave_outer_windows = HashMap::new();
    let mut attestation_failure = None;

    for wave_index in 1..=spec.waves.len() {
        if cancellation.is_cancelled() {
            break;
        }
        if enroot_reattest
            .as_deref_mut()
            .is_some_and(|reattest| !reattest(wave_index, EnrootWaveAttestationPhase::Before))
        {
            attestation_failure = Some(wave_attestation_failure_reason(
                wave_index,
                EnrootWaveAttestationPhase::Before,
            ));
            break;
        }
        let wave_attempts: Vec<WaveAttempt> = attempts
            .iter()
            .filter(|attempt| attempt.wave_index == wave_index)
            .cloned()
            .collect();
        let coordinate_wave_start =
            spec.defaults.runtime == "enroot" && launcher.enroot_wave_start_gate_enabled();
        let coordination_attempts = wave_attempts.clone();
        let started_ns = ordered_clock_ns(&spec.defaults.runtime)?;
        let (sender, receiver) = mpsc::channel::<(usize, Map<String, Value>)>();
        let mut completed_before_release = Vec::new();
        let mut wave_start_failure = None;
        thread::scope(|scope| {
            for attempt in wave_attempts {
                let sender = sender.clone();
                let cancellation = cancellation.clone();
                let completion = &completion;
                scope.spawn(move || {
                    let key = (attempt.task.task_id.clone(), attempt.launch.cpus.clone());
                    let row = match set_current_thread_affinity(&spec.defaults.controller_cpu_ids) {
                        Ok(_) => std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            run_attempt_v3(
                                spec,
                                &attempt,
                                &preflights[&key],
                                launcher,
                                &cancellation,
                            )
                        }))
                        .unwrap_or_else(|_| {
                            let profile = wave_profile_path(spec, &attempt);
                            failed_row(
                                &attempt.task,
                                1,
                                relative_profile(&spec.output, &profile),
                                &preflights[&key].image_id,
                                if cancellation.is_cancelled() {
                                    "profile interrupted: profile worker panicked"
                                } else {
                                    "profile worker failed: profile worker panicked"
                                },
                                Some(attempt.attempt_index),
                                Some(&attempt),
                                spec.defaults.runtime == "enroot",
                            )
                        }),
                        Err(error) => {
                            let profile = wave_profile_path(spec, &attempt);
                            failed_row(
                                &attempt.task,
                                1,
                                relative_profile(&spec.output, &profile),
                                &preflights[&key].image_id,
                                format!("profile worker failed: {error}"),
                                Some(attempt.attempt_index),
                                Some(&attempt),
                                spec.defaults.runtime == "enroot",
                            )
                        }
                    };
                    let index = completion.fetch_add(1, Ordering::SeqCst) + 1;
                    let mut row = row;
                    set_completion_index(&mut row, index);
                    let _ = sender.send((attempt.attempt_index, row));
                });
            }
            drop(sender);
            if coordinate_wave_start {
                if let Err(error) = coordinate_enroot_wave_start(
                    spec,
                    &coordination_attempts,
                    &receiver,
                    &mut completed_before_release,
                    cancellation,
                    RUN_WAVE_START_TIMEOUT,
                ) {
                    wave_start_failure = Some(error.to_string());
                    cancellation.cancel();
                }
            }
        });
        let ended_ns = ordered_clock_ns(&spec.defaults.runtime);
        let post_attestation_failed = enroot_reattest
            .as_deref_mut()
            .is_some_and(|reattest| !reattest(wave_index, EnrootWaveAttestationPhase::After));
        let ended_ns = ended_ns?;
        wave_outer_windows.insert(
            wave_index,
            HostWindow::for_runtime(&spec.defaults.runtime, started_ns, ended_ns)?,
        );
        let mut wave_rows = completed_before_release;
        wave_rows.extend(receiver.into_iter());
        if let Some(reason) = wave_start_failure.as_deref() {
            let reason = format!("Enroot wave-start coordination failed: {reason}");
            for (_, row) in &mut wave_rows {
                row.insert("duration_valid".to_owned(), Value::Bool(false));
                append_reason_once(row, &reason);
            }
        }
        rows_by_attempt.extend(wave_rows);
        if post_attestation_failed {
            attestation_failure = Some(wave_attestation_failure_reason(
                wave_index,
                EnrootWaveAttestationPhase::After,
            ));
            break;
        }
        if cancellation.is_cancelled() {
            break;
        }
    }

    let interrupted = cancellation.is_cancelled();
    if let Some(reason) = attestation_failure.as_deref() {
        for row in rows_by_attempt.values_mut() {
            row.insert("duration_valid".to_owned(), Value::Bool(false));
            append_reason(row, reason);
        }
    }
    for attempt in attempts {
        if rows_by_attempt.contains_key(&attempt.attempt_index) {
            continue;
        }
        let key = (attempt.task.task_id.clone(), attempt.launch.cpus.clone());
        let profile = wave_profile_path(spec, attempt);
        rows_by_attempt.insert(
            attempt.attempt_index,
            failed_row(
                &attempt.task,
                1,
                relative_profile(&spec.output, &profile),
                &preflights[&key].image_id,
                if interrupted {
                    "run interrupted before attempt dispatch"
                } else if let Some(reason) = attestation_failure.as_deref() {
                    reason
                } else {
                    "profile worker failed without returning a row"
                },
                Some(attempt.attempt_index),
                Some(attempt),
                spec.defaults.runtime == "enroot",
            ),
        );
    }
    Ok((
        rows_by_attempt.into_values().collect(),
        interrupted,
        wave_outer_windows,
        attestation_failure,
    ))
}

fn append_reason(row: &mut Map<String, Value>, reason: &str) {
    let previous = row.get("reason").and_then(Value::as_str).unwrap_or("");
    let combined = if previous.is_empty() {
        reason.to_owned()
    } else {
        format!("{previous}; {reason}")
    };
    row.insert("reason".to_owned(), Value::String(combined));
}

fn append_reason_once(row: &mut Map<String, Value>, reason: &str) {
    let present = row
        .get("reason")
        .and_then(Value::as_str)
        .is_some_and(|previous| previous.split("; ").any(|value| value == reason));
    if !present {
        append_reason(row, reason);
    }
}

fn linux_clock_ns(clock_id: libc::clockid_t, name: &str) -> Result<u64, RunExecutionError> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_gettime(clock_id, &mut value) } != 0
        || value.tv_sec < 0
        || value.tv_nsec < 0
    {
        return Err(RunExecutionError::new(format!(
            "unable to read {name}: {}",
            io::Error::last_os_error()
        )));
    }
    Ok((value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64))
}

fn linux_clock_resolution_ns(
    clock_id: libc::clockid_t,
    name: &str,
) -> Result<u64, RunExecutionError> {
    let mut value = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    if unsafe { libc::clock_getres(clock_id, &mut value) } != 0
        || value.tv_sec < 0
        || value.tv_nsec < 0
    {
        return Err(RunExecutionError::new(format!(
            "unable to read {name} resolution: {}",
            io::Error::last_os_error()
        )));
    }
    Ok((value.tv_sec as u64)
        .saturating_mul(1_000_000_000)
        .saturating_add(value.tv_nsec as u64)
        .max(1))
}

fn ordered_clock_ns(runtime: &str) -> Result<u64, RunExecutionError> {
    if runtime == "enroot" {
        linux_clock_ns(libc::CLOCK_BOOTTIME, "CLOCK_BOOTTIME")
    } else {
        Ok(monotonic_ns())
    }
}

fn ordered_clock_resolution_ns(runtime: &str) -> Result<u64, RunExecutionError> {
    if runtime == "enroot" {
        linux_clock_resolution_ns(libc::CLOCK_BOOTTIME, "CLOCK_BOOTTIME")
    } else {
        Ok(monotonic_resolution_ns())
    }
}

#[derive(Clone, Copy, Debug)]
struct HostWindow {
    start_ns: u64,
    end_ns: u64,
    resolution_ns: u64,
}

impl HostWindow {
    fn new(start_ns: u64, end_ns: u64) -> Result<Self, RunExecutionError> {
        Self::with_resolution(
            start_ns,
            end_ns,
            monotonic_resolution_ns(),
            "CLOCK_MONOTONIC",
        )
    }

    fn for_runtime(runtime: &str, start_ns: u64, end_ns: u64) -> Result<Self, RunExecutionError> {
        Self::with_resolution(
            start_ns,
            end_ns,
            ordered_clock_resolution_ns(runtime)?,
            if runtime == "enroot" {
                "CLOCK_BOOTTIME"
            } else {
                "CLOCK_MONOTONIC"
            },
        )
    }

    fn with_resolution(
        start_ns: u64,
        end_ns: u64,
        resolution_ns: u64,
        clock_name: &str,
    ) -> Result<Self, RunExecutionError> {
        if end_ns <= start_ns {
            return Err(RunExecutionError::new(format!(
                "host {clock_name} execution window is not increasing"
            )));
        }
        Ok(Self {
            start_ns,
            end_ns,
            resolution_ns,
        })
    }

    fn wall_sec(self) -> f64 {
        (self.end_ns - self.start_ns) as f64 / 1_000_000_000.0
    }
}

fn invalidate_window_containment(
    rows: &mut [Map<String, Value>],
    window: HostWindow,
    wave_index: Option<usize>,
) {
    for row in rows.iter_mut().filter(|row| {
        wave_index.map_or(true, |expected| {
            row_u64(row, "wave_index") == Some(expected as u64)
        })
    }) {
        let (Some(start), Some(end)) = (
            row_u64(row, "window_start_ns"),
            row_u64(row, "window_end_ns"),
        ) else {
            continue;
        };
        let lower = window.start_ns.saturating_sub(window.resolution_ns);
        let upper = window.end_ns.saturating_add(window.resolution_ns);
        if start < lower || end > upper {
            row.insert("duration_valid".to_owned(), Value::Bool(false));
            append_reason(
                row,
                "measured command window lies outside its enclosing host CLOCK_MONOTONIC window",
            );
        }
    }
}

fn invalidate_enroot_duration_bounds(
    rows: &mut [Map<String, Value>],
    window: HostWindow,
    wave_index: usize,
) {
    let upper = window.wall_sec() + ENROOT_OUTER_WALL_TOLERANCE_SEC;
    for row in rows.iter_mut().filter(|row| {
        row_u64(row, "wave_index") == Some(wave_index as u64) && row_bool(row, "duration_valid")
    }) {
        if row_f64(row, "duration_sec")
            .is_some_and(|duration| duration.is_finite() && duration > 0.0 && duration <= upper)
        {
            continue;
        }
        row.insert("duration_valid".to_owned(), Value::Bool(false));
        append_reason(
            row,
            "task duration exceeds the lifecycle-inclusive outer wave bound",
        );
    }
}

fn invalidate_changed_enroot_images(spec: &RunSpecV3, rows: &mut [Map<String, Value>]) {
    if spec.defaults.runtime != "enroot" {
        return;
    }
    let mut observed = HashMap::<PathBuf, Result<String, String>>::new();
    for task in &spec.tasks {
        let path = PathBuf::from(&task.image);
        let current = observed.entry(path.clone()).or_insert_with(|| {
            file_sha256(&path)
                .map(|digest| format!("sha256:{digest}"))
                .map_err(|error| error.to_string())
        });
        for row in rows
            .iter_mut()
            .filter(|row| row.get("task_id").and_then(Value::as_str) == Some(task.task_id.as_str()))
        {
            let matches = current
                .as_ref()
                .ok()
                .is_some_and(|digest| row.get("image_id").and_then(Value::as_str) == Some(digest));
            if matches {
                continue;
            }
            row.insert("duration_valid".to_owned(), Value::Bool(false));
            append_reason(
                row,
                match current {
                    Ok(_) => "Enroot image content changed after preflight".to_owned(),
                    Err(error) => format!("unable to re-read Enroot image after replay: {error}"),
                }
                .as_str(),
            );
        }
    }
}

fn invalidate_plan_drift(rows: &mut [Map<String, Value>]) {
    let groups: BTreeSet<(String, Option<String>)> = rows
        .iter()
        .map(|row| {
            (
                row.get("task_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                row.contains_key("wave_id").then(|| {
                    row.get("cpus")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                }),
            )
        })
        .collect();
    for (task_id, cpus) in groups {
        let digests: BTreeSet<String> = rows
            .iter()
            .filter(|row| {
                row.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
                    && row.contains_key("wave_id").then(|| {
                        row.get("cpus")
                            .and_then(Value::as_str)
                            .unwrap_or_default()
                            .to_owned()
                    }) == cpus
            })
            .filter_map(|row| {
                row.get("execution_plan_digest")
                    .and_then(Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        if digests.len() <= 1 {
            continue;
        }
        for row in rows.iter_mut().filter(|row| {
            row.get("task_id").and_then(Value::as_str) == Some(task_id.as_str())
                && row.contains_key("wave_id").then(|| {
                    row.get("cpus")
                        .and_then(Value::as_str)
                        .unwrap_or_default()
                        .to_owned()
                }) == cpus
        }) {
            row.insert("duration_valid".to_owned(), Value::Bool(false));
            append_reason(row, "execution plan changed across repetitions");
        }
    }
}

fn row_bool(row: &Map<String, Value>, key: &str) -> bool {
    row.get(key).and_then(Value::as_bool) == Some(true)
}

fn row_u64(row: &Map<String, Value>, key: &str) -> Option<u64> {
    row.get(key).and_then(Value::as_u64)
}

fn row_f64(row: &Map<String, Value>, key: &str) -> Option<f64> {
    row.get(key)
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
}

fn status_count(rows: &[Map<String, Value>], status: &str) -> usize {
    rows.iter()
        .filter(|row| row.get("status").and_then(Value::as_str) == Some(status))
        .count()
}

fn precise_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut sum = 0.0;
    let mut correction = 0.0;
    for value in values {
        let adjusted = value - correction;
        let next = sum + adjusted;
        correction = (next - sum) - adjusted;
        sum = next;
    }
    sum
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.total_cmp(right));
    let middle = values.len() / 2;
    if values.len() % 2 == 0 {
        Some((values[middle - 1] + values[middle]) / 2.0)
    } else {
        Some(values[middle])
    }
}

#[derive(Clone, Debug)]
struct Overlap {
    window_start_ns: Option<u64>,
    window_end_ns: Option<u64>,
    window_sec: Option<f64>,
    peak: usize,
    seconds_by_active: BTreeMap<usize, f64>,
}

fn measured_attempt_overlap(intervals: &[(u64, u64)]) -> Result<Overlap, RunExecutionError> {
    if intervals.is_empty() {
        return Ok(Overlap {
            window_start_ns: None,
            window_end_ns: None,
            window_sec: None,
            peak: 0,
            seconds_by_active: BTreeMap::new(),
        });
    }
    let mut changes = BTreeMap::<u64, i64>::new();
    for (start, end) in intervals {
        *changes.entry(*start).or_default() += 1;
        *changes.entry(*end).or_default() -= 1;
    }
    let mut cursor = *changes.keys().next().expect("nonempty changes");
    let mut active = 0_i64;
    let mut peak = 0_i64;
    let mut duration_by_active = BTreeMap::<usize, u64>::new();
    for (timestamp, change) in &changes {
        if *timestamp > cursor {
            if active < 0 {
                return Err(RunExecutionError::new(
                    "measured-attempt overlap became negative",
                ));
            }
            *duration_by_active.entry(active as usize).or_default() += *timestamp - cursor;
        }
        active += change;
        peak = peak.max(active);
        cursor = *timestamp;
    }
    if active != 0 {
        return Err(RunExecutionError::new(
            "measured-attempt overlap intervals did not close",
        ));
    }
    let start = *changes.keys().next().expect("nonempty changes");
    let end = *changes.keys().next_back().expect("nonempty changes");
    Ok(Overlap {
        window_start_ns: Some(start),
        window_end_ns: Some(end),
        window_sec: Some((end - start) as f64 / 1_000_000_000.0),
        peak: peak as usize,
        seconds_by_active: duration_by_active
            .into_iter()
            .map(|(active, duration)| (active, duration as f64 / 1_000_000_000.0))
            .collect(),
    })
}

fn seconds_by_active_value(values: &BTreeMap<usize, f64>) -> Value {
    let mut object = Map::new();
    for (active, duration) in values {
        object.insert(active.to_string(), number_value(Some(*duration)));
    }
    Value::Object(object)
}

#[allow(clippy::too_many_arguments)]
fn concurrent_overall(
    rows: &[Map<String, Value>],
    interrupted: bool,
    outer_window: Option<HostWindow>,
    expected_attempt_count: usize,
    expected_attempt_indexes: &BTreeSet<usize>,
    pool_cpu_ids: &[u32],
    requested_concurrency: usize,
) -> Result<Value, RunExecutionError> {
    let outer_run_wall_sec = outer_window.map(HostWindow::wall_sec).unwrap_or(0.0);
    let valid_rows: Vec<&Map<String, Value>> = rows
        .iter()
        .filter(|row| row_bool(row, "duration_valid"))
        .collect();
    let valid_durations: Vec<f64> = valid_rows
        .iter()
        .filter_map(|row| row_f64(row, "duration_sec"))
        .collect();
    let measured_window_rows: Vec<&Map<String, Value>> = rows
        .iter()
        .filter(|row| {
            matches!(
                (row_u64(row, "window_start_ns"), row_u64(row, "window_end_ns")),
                (Some(start), Some(end)) if end > start
            )
        })
        .collect();
    let measured_intervals: Vec<(u64, u64)> = measured_window_rows
        .iter()
        .map(|row| {
            (
                row_u64(row, "window_start_ns").expect("filtered start"),
                row_u64(row, "window_end_ns").expect("filtered end"),
            )
        })
        .collect();
    let valid_intervals: Vec<(u64, u64)> = valid_rows
        .iter()
        .filter_map(|row| {
            let start = row_u64(row, "window_start_ns")?;
            let end = row_u64(row, "window_end_ns")?;
            (end > start).then_some((start, end))
        })
        .collect();
    let overlap = measured_attempt_overlap(&measured_intervals)?;
    let observed_indexes: Vec<usize> = rows
        .iter()
        .filter_map(|row| row_u64(row, "attempt_index").map(|value| value as usize))
        .collect();
    let cpu_usage_complete = valid_rows.iter().all(|row| {
        row_f64(row, "cpu_usage_sec")
            .map(|number| number >= 0.0)
            .unwrap_or(false)
    });
    let mut invalid_reasons = Vec::<String>::new();
    if interrupted {
        invalid_reasons.push("run was interrupted".to_owned());
    }
    if rows.len() != expected_attempt_count {
        invalid_reasons.push(format!(
            "retained {} of {} declared attempts",
            rows.len(),
            expected_attempt_count
        ));
    }
    let observed_set: BTreeSet<usize> = observed_indexes.iter().copied().collect();
    if observed_indexes.len() != rows.len()
        || &observed_set != expected_attempt_indexes
        || observed_set.len() != observed_indexes.len()
    {
        invalid_reasons.push("retained attempt indexes are incomplete or duplicated".to_owned());
    }
    if valid_rows.len() != expected_attempt_count {
        invalid_reasons.push(format!(
            "only {} of {} attempts have valid duration evidence",
            valid_rows.len(),
            expected_attempt_count
        ));
    }
    if valid_intervals.len() != expected_attempt_count {
        invalid_reasons.push(format!(
            "only {} of {} valid attempts have usable measured windows",
            valid_intervals.len(),
            expected_attempt_count
        ));
    }
    if !cpu_usage_complete || valid_rows.len() != expected_attempt_count {
        invalid_reasons.push("not every declared attempt has valid cgroup CPU usage".to_owned());
    }
    if overlap.peak > requested_concurrency {
        invalid_reasons.push(format!(
            "measured-attempt overlap exceeds requested concurrency: peak={}, requested={}",
            overlap.peak, requested_concurrency
        ));
    }

    let base_complete = invalid_reasons.is_empty();
    let candidate_batch_wall_sec = base_complete.then_some(overlap.window_sec).flatten();
    let candidate_total_cpu_sec = base_complete.then(|| {
        precise_sum(
            valid_rows
                .iter()
                .filter_map(|row| row_f64(row, "cpu_usage_sec")),
        )
    });
    let cpu_capacity_sec = candidate_batch_wall_sec.map(|wall| wall * pool_cpu_ids.len() as f64);
    let cpu_capacity_tolerance_sec = cpu_capacity_sec.map(|capacity| {
        (expected_attempt_count as f64 * CPU_CAPACITY_COUNTER_TOLERANCE_SEC_PER_ATTEMPT)
            .max(capacity * CPU_CAPACITY_FLOAT_TOLERANCE_FRACTION)
    });
    if let (Some(usage), Some(capacity), Some(tolerance)) = (
        candidate_total_cpu_sec,
        cpu_capacity_sec,
        cpu_capacity_tolerance_sec,
    ) {
        if usage > capacity + tolerance {
            invalid_reasons.push(format!(
                "aggregate cgroup CPU exceeds declared pool capacity: usage={usage:.9}s, capacity={capacity:.9}s, tolerance={tolerance:.9}s"
            ));
        }
    }
    let complete = invalid_reasons.is_empty();
    let batch_wall_sec = complete.then_some(candidate_batch_wall_sec).flatten();
    let total_cpu_sec = complete.then_some(candidate_total_cpu_sec).flatten();
    let aggregate_average_cores = match (total_cpu_sec, batch_wall_sec) {
        (Some(cpu), Some(wall)) if wall > 0.0 => Some(cpu / wall),
        _ => None,
    };
    let requested_concurrency_sec = overlap
        .seconds_by_active
        .get(&requested_concurrency)
        .copied()
        .unwrap_or(0.0);
    let throughput = batch_wall_sec
        .filter(|wall| *wall > 0.0)
        .map(|wall| expected_attempt_count as f64 / wall);
    let summed_service =
        (!valid_durations.is_empty()).then(|| precise_sum(valid_durations.iter().copied()));
    let pool_utilization =
        aggregate_average_cores.map(|average| 100.0 * average / pool_cpu_ids.len() as f64);

    Ok(json!({
        "attempt_count": rows.len(),
        "declared_attempt_count": expected_attempt_count,
        "completed": status_count(rows, "completed"),
        "failed": status_count(rows, "failed"),
        "valid_duration_count": valid_rows.len(),
        "summed_service_time_sec": summed_service,
        "interrupted": interrupted,
        "valid_for_throughput": complete,
        "throughput_invalid_reasons": invalid_reasons,
        "outer_run_wall_sec": outer_run_wall_sec,
        "outer_window_start_ns": outer_window.map(|window| window.start_ns),
        "outer_window_end_ns": outer_window.map(|window| window.end_ns),
        "clock_resolution_ns": outer_window.map(|window| window.resolution_ns),
        "window_containment_tolerance_ns": outer_window.map(|window| window.resolution_ns),
        "batch_window_start_ns": overlap.window_start_ns,
        "batch_window_end_ns": overlap.window_end_ns,
        "measured_window_count": measured_window_rows.len(),
        "measured_window_span_sec": overlap.window_sec,
        "batch_wall_sec": batch_wall_sec,
        "throughput_tasks_per_sec": throughput,
        "throughput_tasks_per_hour": throughput.map(|value| value * 3600.0),
        "total_cgroup_cpu_sec": total_cpu_sec,
        "cpu_pool_capacity_sec": cpu_capacity_sec,
        "cpu_pool_capacity_tolerance_sec": cpu_capacity_tolerance_sec,
        "aggregate_average_logical_cores": aggregate_average_cores,
        "pool_utilization_percent": pool_utilization,
        "requested_concurrency": requested_concurrency,
        "achieved_peak_measured_attempts": overlap.peak,
        "time_sec_by_measured_attempts": seconds_by_active_value(&overlap.seconds_by_active),
        "requested_concurrency_sec": requested_concurrency_sec,
        "requested_concurrency_fraction": batch_wall_sec
            .filter(|wall| *wall > 0.0)
            .map(|wall| requested_concurrency_sec / wall),
    }))
}

fn enroot_wave_overall(
    rows: &[Map<String, Value>],
    interrupted: bool,
    outer_window: Option<HostWindow>,
    expected_attempt_count: usize,
    expected_attempt_indexes: &BTreeSet<usize>,
) -> Result<Value, RunExecutionError> {
    let valid_rows = rows
        .iter()
        .filter(|row| row_bool(row, "duration_valid"))
        .collect::<Vec<_>>();
    let durations = valid_rows
        .iter()
        .filter_map(|row| row_f64(row, "duration_sec"))
        .collect::<Vec<_>>();
    let intervals = valid_rows
        .iter()
        .filter_map(|row| {
            let start = row_u64(row, "window_start_ns")?;
            let end = row_u64(row, "window_end_ns")?;
            (end > start).then_some((start, end))
        })
        .collect::<Vec<_>>();
    let overlap = measured_attempt_overlap(&intervals)?;
    let observed_indexes = rows
        .iter()
        .filter_map(|row| row_u64(row, "attempt_index").map(|value| value as usize))
        .collect::<Vec<_>>();
    let observed_index_set = observed_indexes.iter().copied().collect::<BTreeSet<_>>();
    let clock_identities = valid_rows
        .iter()
        .filter_map(|row| row.get("clock_reference").and_then(Value::as_object))
        .filter_map(|clock| {
            Some((
                clock.get("clock_name")?.as_str()?.to_owned(),
                clock.get("boot_id")?.as_str()?.to_owned(),
                clock.get("time_namespace")?.as_str()?.to_owned(),
            ))
        })
        .collect::<Vec<_>>();
    let unique_clock_identities = clock_identities.iter().cloned().collect::<BTreeSet<_>>();
    let mut reasons = Vec::<String>::new();
    if interrupted {
        reasons.push("run was interrupted".to_owned());
    }
    if rows.len() != expected_attempt_count {
        reasons.push(format!(
            "retained {} of {} declared attempts",
            rows.len(),
            expected_attempt_count
        ));
    }
    if observed_indexes.len() != rows.len()
        || &observed_index_set != expected_attempt_indexes
        || observed_index_set.len() != observed_indexes.len()
    {
        reasons.push("retained attempt indexes are incomplete or duplicated".to_owned());
    }
    if valid_rows.len() != expected_attempt_count {
        reasons.push(format!(
            "only {} of {} attempts have valid wall-only duration evidence",
            valid_rows.len(),
            expected_attempt_count
        ));
    }
    if intervals.len() != expected_attempt_count {
        reasons.push(format!(
            "only {} of {} attempts have usable absolute replay windows",
            intervals.len(),
            expected_attempt_count
        ));
    }
    if clock_identities.len() != expected_attempt_count || unique_clock_identities.len() != 1 {
        reasons.push("replay windows do not share one monotonic clock identity".to_owned());
    }
    if overlap.peak != expected_attempt_count {
        reasons.push(format!(
            "replay commands did not achieve all-launch overlap: peak={}, expected={expected_attempt_count}",
            overlap.peak
        ));
    }
    let outer_wall_sec = outer_window.map(HostWindow::wall_sec);
    if !outer_wall_sec.is_some_and(|wall| wall.is_finite() && wall > 0.0) {
        reasons
            .push("lifecycle-inclusive outer wave wall time is not finite and positive".to_owned());
    }
    let complete = reasons.is_empty();
    let replay_wall = complete.then_some(overlap.window_sec).flatten();
    let replay_throughput = replay_wall
        .filter(|wall| *wall > 0.0)
        .map(|wall| expected_attempt_count as f64 / wall);
    let observed_lifecycle_throughput = outer_wall_sec
        .filter(|wall| *wall > 0.0)
        .map(|wall| expected_attempt_count as f64 / wall);
    let lifecycle_throughput = complete.then_some(observed_lifecycle_throughput).flatten();
    let starts = intervals
        .iter()
        .map(|(start, _)| *start)
        .collect::<Vec<_>>();
    let start_spread_sec = (!starts.is_empty()).then(|| {
        (starts.iter().max().unwrap() - starts.iter().min().unwrap()) as f64 / 1_000_000_000.0
    });
    let all_launch_overlap_sec = overlap
        .seconds_by_active
        .get(&expected_attempt_count)
        .copied()
        .unwrap_or(0.0);
    Ok(json!({
        "attempt_count": rows.len(),
        "declared_attempt_count": expected_attempt_count,
        "completed": status_count(rows, "completed"),
        "failed": status_count(rows, "failed"),
        "valid_duration_count": valid_rows.len(),
        "summed_service_time_sec": (!durations.is_empty()).then(|| precise_sum(durations.iter().copied())),
        "interrupted": interrupted,
        "valid_for_throughput": complete,
        "valid_for_wall_only_throughput": complete,
        "throughput_invalid_reasons": reasons,
        "measurement_mode": "wall-only-replay-command",
        "outer_run_wall_sec": outer_wall_sec,
        "outer_window_start_ns": outer_window.map(|window| window.start_ns),
        "outer_window_end_ns": outer_window.map(|window| window.end_ns),
        "clock_resolution_ns": outer_window.map(|window| window.resolution_ns),
        "window_containment_tolerance_ns": Value::Null,
        "lifecycle_inclusive_wall_sec": outer_wall_sec,
        "observed_lifecycle_inclusive_throughput_tasks_per_sec": observed_lifecycle_throughput,
        "lifecycle_inclusive_throughput_tasks_per_sec": lifecycle_throughput,
        "lifecycle_inclusive_throughput_tasks_per_hour": lifecycle_throughput.map(|value| value * 3600.0),
        "batch_window_start_ns": overlap.window_start_ns,
        "batch_window_end_ns": overlap.window_end_ns,
        "measured_window_count": intervals.len(),
        "measured_window_span_sec": overlap.window_sec,
        "batch_wall_sec": replay_wall,
        "replay_command_batch_wall_sec": replay_wall,
        "throughput_tasks_per_sec": replay_throughput,
        "throughput_tasks_per_hour": replay_throughput.map(|value| value * 3600.0),
        "wall_only_throughput_tasks_per_sec": replay_throughput,
        "wall_only_throughput_tasks_per_hour": replay_throughput.map(|value| value * 3600.0),
        "total_cgroup_cpu_sec": Value::Null,
        "cpu_pool_capacity_sec": Value::Null,
        "cpu_pool_capacity_tolerance_sec": Value::Null,
        "aggregate_average_logical_cores": Value::Null,
        "pool_utilization_percent": Value::Null,
        "requested_concurrency": expected_attempt_count,
        "achieved_peak_measured_attempts": overlap.peak,
        "start_spread_sec": start_spread_sec,
        "all_launch_overlap_sec": all_launch_overlap_sec,
        "time_sec_by_measured_attempts": seconds_by_active_value(&overlap.seconds_by_active),
        "requested_concurrency_sec": all_launch_overlap_sec,
        "requested_concurrency_fraction": replay_wall
            .filter(|wall| *wall > 0.0)
            .map(|wall| all_launch_overlap_sec / wall),
    }))
}

fn unique_string_field(rows: &[&Map<String, Value>], field: &str) -> Vec<String> {
    rows.iter()
        .filter_map(|row| row.get(field).and_then(Value::as_str))
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect()
}

fn task_summaries(
    tasks: &[RunTaskSpec],
    rows: &[Map<String, Value>],
    repetitions: Option<usize>,
) -> Vec<Value> {
    tasks
        .iter()
        .map(|task| {
            let task_rows: Vec<&Map<String, Value>> = rows
                .iter()
                .filter(|row| {
                    row.get("task_id").and_then(Value::as_str)
                        == Some(task.task_id.as_str())
                })
                .collect();
            let valid_durations: Vec<f64> = task_rows
                .iter()
                .filter(|row| row_bool(row, "duration_valid"))
                .filter_map(|row| row_f64(row, "duration_sec"))
                .collect();
            let complete = repetitions
                .map(|count| task_rows.len() == count && valid_durations.len() == count)
                .unwrap_or(false);
            let mut value = json!({
                "task_id": task.task_id,
                "source_sha256": task.source_sha256,
                "image_ids": unique_string_field(&task_rows, "image_id"),
                "execution_plan_digests": unique_string_field(&task_rows, "execution_plan_digest"),
                "attempt_count": task_rows.len(),
                "completed": task_rows.iter().filter(|row| row.get("status").and_then(Value::as_str) == Some("completed")).count(),
                "failed": task_rows.iter().filter(|row| row.get("status").and_then(Value::as_str) == Some("failed")).count(),
                "valid_duration_count": valid_durations.len(),
            });
            if let Some(object) = value.as_object_mut() {
                if repetitions.is_some() {
                    object.insert(
                        "median_duration_sec".to_owned(),
                        number_value(complete.then(|| median(valid_durations)).flatten()),
                    );
                }
            }
            value
        })
        .collect()
}

fn summary_v1(
    spec: &RunSpecV1,
    rows: &[Map<String, Value>],
    host_cpu: Value,
    controller_digest: &str,
    interrupted: bool,
    outer_window: HostWindow,
) -> Value {
    let valid_durations: Vec<f64> = rows
        .iter()
        .filter(|row| row_bool(row, "duration_valid"))
        .filter_map(|row| row_f64(row, "duration_sec"))
        .collect();
    let total = precise_sum(valid_durations.iter().copied());
    let complete = !rows.is_empty() && valid_durations.len() == rows.len();
    json!({
        "schema_version": SUMMARY_SCHEMA_VERSION_V1,
        "manifest": spec.manifest_path,
        "manifest_sha256": spec.manifest_sha256,
        "cpus": spec.cpus,
        "host_cpu": host_cpu,
        "controller_runtime": "rust-host-mount",
        "controller_runtime_content_bound": true,
        "controller_runtime_limitation": Value::Null,
        "controller_binary_sha256": controller_digest,
        "duration_clock": "host-monotonic",
        "duration_boundary": "measured replay command",
        "canonical_clock_reference": clock_reference(),
        "outer_window_start_ns": outer_window.start_ns,
        "outer_window_end_ns": outer_window.end_ns,
        "clock_resolution_ns": outer_window.resolution_ns,
        "window_containment_tolerance_ns": outer_window.resolution_ns,
        "repetitions": spec.repetitions,
        "results_csv": "results.csv",
        "tasks": task_summaries(&spec.tasks, rows, Some(spec.repetitions)),
        "overall": {
            "attempt_count": rows.len(),
            "completed": status_count(rows, "completed"),
            "failed": status_count(rows, "failed"),
            "valid_duration_count": valid_durations.len(),
            "interrupted": interrupted,
            "valid_duration_total_sec": (!valid_durations.is_empty()).then_some(total),
            "replay_tasks_per_sec": (!interrupted && complete && total > 0.0).then_some(rows.len() as f64 / total),
        },
    })
}

fn summary_v2(
    spec: &RunSpecV2,
    rows: &[Map<String, Value>],
    host_cpu: Value,
    controller_digest: &str,
    interrupted: bool,
    outer_window: HostWindow,
) -> Result<Value, RunExecutionError> {
    let indexes: BTreeSet<usize> = (1..=spec.tasks.len() * spec.repetitions).collect();
    let overall = concurrent_overall(
        rows,
        interrupted,
        Some(outer_window),
        indexes.len(),
        &indexes,
        &spec.cpu_ids,
        spec.concurrency,
    )?;
    Ok(json!({
        "schema_version": SUMMARY_SCHEMA_VERSION_V2,
        "manifest": spec.manifest_path,
        "manifest_sha256": spec.manifest_sha256,
        "cpus": spec.cpus,
        "host_cpu": host_cpu,
        "controller_runtime": "rust-host-mount",
        "controller_runtime_content_bound": true,
        "controller_runtime_limitation": Value::Null,
        "controller_binary_sha256": controller_digest,
        "duration_clock": "host-monotonic",
        "duration_boundary": "measured replay command",
        "canonical_clock_reference": clock_reference(),
        "repetitions": spec.repetitions,
        "results_csv": "results.csv",
        "tasks": task_summaries(&spec.tasks, rows, Some(spec.repetitions)),
        "concurrency": spec.concurrency,
        "scheduling": {
            "mode": "bounded-fifo-shared-cpuset",
            "container_cpu_placement": "all attempts share cpus",
            "completion_index_semantics": "worker-return-order",
            "cpu_exclusivity_verified": false,
        },
        "overall": overall,
    }))
}

fn metadata_value(metadata: &[(String, MetadataScalar)]) -> Value {
    let mut object = Map::new();
    for (key, scalar) in metadata {
        let value = serde_json::to_value(scalar).expect("metadata scalars serialize to JSON");
        object.insert(key.clone(), value);
    }
    Value::Object(object)
}

#[allow(clippy::too_many_arguments)]
fn summary_v3(
    spec: &RunSpecV3,
    rows: &[Map<String, Value>],
    host_cpu: Value,
    interrupted: bool,
    outer_window: HostWindow,
    wave_outer_windows: &HashMap<usize, HostWindow>,
    controller_affinity: Value,
) -> Result<Value, RunExecutionError> {
    let mut wave_summaries = Vec::new();
    let mut first_attempt_index = 1_usize;
    for (offset, wave) in spec.waves.iter().enumerate() {
        let wave_index = offset + 1;
        let wave_rows: Vec<Map<String, Value>> = rows
            .iter()
            .filter(|row| row_u64(row, "wave_index") == Some(wave_index as u64))
            .cloned()
            .collect();
        let indexes: BTreeSet<usize> =
            (first_attempt_index..first_attempt_index + wave.launches.len()).collect();
        first_attempt_index += wave.launches.len();
        let pool_cpu_ids: Vec<u32> = wave
            .launches
            .iter()
            .flat_map(|launch| launch.cpu_ids.iter().copied())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut overall = if spec.defaults.runtime == "enroot" {
            enroot_wave_overall(
                &wave_rows,
                interrupted,
                wave_outer_windows.get(&wave_index).copied(),
                wave.launches.len(),
                &indexes,
            )?
        } else {
            concurrent_overall(
                &wave_rows,
                interrupted,
                wave_outer_windows.get(&wave_index).copied(),
                wave.launches.len(),
                &indexes,
                &pool_cpu_ids,
                wave.launches.len(),
            )?
        };
        let valid_durations: Vec<f64> = wave_rows
            .iter()
            .filter(|row| row_bool(row, "duration_valid"))
            .filter_map(|row| row_f64(row, "duration_sec"))
            .collect();
        let valid = overall.get("valid_for_throughput").and_then(Value::as_bool) == Some(true);
        let overall_object = overall
            .as_object_mut()
            .expect("concurrent overall is an object");
        if valid && !valid_durations.is_empty() {
            let mean = precise_sum(valid_durations.iter().copied()) / valid_durations.len() as f64;
            overall_object.insert(
                "mean_task_duration_sec".to_owned(),
                number_value(Some(mean)),
            );
            overall_object.insert(
                "median_task_duration_sec".to_owned(),
                number_value(median(valid_durations)),
            );
            if spec.defaults.runtime == "enroot" {
                overall_object.insert("mean_batch_relative_completion_sec".to_owned(), Value::Null);
            } else {
                let batch_start = overall_object
                    .get("batch_window_start_ns")
                    .and_then(Value::as_u64)
                    .expect("valid batch has start");
                let completions = wave_rows.iter().filter_map(|row| {
                    row_u64(row, "window_end_ns")
                        .map(|end| end.saturating_sub(batch_start) as f64 / 1_000_000_000.0)
                });
                let completions: Vec<f64> = completions.collect();
                overall_object.insert(
                    "mean_batch_relative_completion_sec".to_owned(),
                    number_value(Some(
                        precise_sum(completions.iter().copied()) / completions.len() as f64,
                    )),
                );
            }
        } else {
            overall_object.insert("mean_task_duration_sec".to_owned(), Value::Null);
            overall_object.insert("median_task_duration_sec".to_owned(), Value::Null);
            overall_object.insert("mean_batch_relative_completion_sec".to_owned(), Value::Null);
        }
        let cpus = pool_cpu_ids
            .iter()
            .map(u32::to_string)
            .collect::<Vec<_>>()
            .join(",");
        let mut wave_summary = json!({
            "wave_index": wave_index,
            "wave_id": wave.wave_id,
            "scored": wave.scored,
            "metadata": metadata_value(&wave.metadata),
            "cpus": cpus.clone(),
            "logical_cpu_count": pool_cpu_ids.len(),
            "launch_count": wave.launches.len(),
            "overall": overall,
        });
        if spec.defaults.runtime == "enroot" {
            let object = wave_summary
                .as_object_mut()
                .expect("wave summary is an object");
            object.insert("requested_cpu_lanes".to_owned(), Value::String(cpus));
            object.insert(
                "requested_logical_cpu_lane_count".to_owned(),
                Value::Number(Number::from(pool_cpu_ids.len() as u64)),
            );
        }
        wave_summaries.push(wave_summary);
    }
    let declared_attempt_count: usize = spec.waves.iter().map(|wave| wave.launches.len()).sum();
    let all_rows_valid = rows.len() == declared_attempt_count
        && rows.iter().all(|row| row_bool(row, "duration_valid"));
    let valid_wave_count = wave_summaries
        .iter()
        .filter(|wave| {
            wave.get("overall")
                .and_then(|value| value.get("valid_for_throughput"))
                .and_then(Value::as_bool)
                == Some(true)
        })
        .count();
    let all_waves_valid = valid_wave_count == wave_summaries.len();
    let enroot = spec.defaults.runtime == "enroot";
    let canonical_clock = if enroot {
        rows.iter()
            .find_map(|row| row.get("clock_reference").cloned())
            .unwrap_or(Value::Null)
    } else {
        clock_reference()
    };
    let valid_for_analysis = !interrupted && all_rows_valid && all_waves_valid;
    let mut summary = json!({
        "schema_version": SUMMARY_SCHEMA_VERSION_V3,
        "manifest": spec.manifest_path,
        "manifest_sha256": spec.manifest_sha256,
        "input_root": spec.input_root,
        "host_cpu": host_cpu,
        "controller_affinity": controller_affinity,
        "controller_runtime": spec.defaults.controller_runtime,
        "controller_runtime_content_bound": true,
        "controller_runtime_limitation": Value::Null,
        "duration_clock": if enroot { ENROOT_TASK_CLOCK_SOURCE } else { "host-monotonic" },
        "duration_boundary": "measured replay command",
        "canonical_clock_reference": canonical_clock,
        "pacing": spec.defaults.pacing,
        "network_policy": {
            "mode": spec.defaults.network_policy,
            "allowed_hosts": spec.defaults.allowed_hosts,
        },
        "profile_sets": spec.defaults.profile_sets,
        "results_csv": "results.csv",
        "scheduling": {
            "mode": "ordered-waves",
            "authoring_mode": if spec.defaults.placement.is_some() { "compact-worker-pool" } else { "explicit-launches" },
            "worker_cpus": spec.defaults.worker_cpus,
            "placement": spec.defaults.placement,
            "within_wave_dispatch": "concurrent-submit",
            "measured_start_alignment": if enroot { "runner-all-ready-release-gate" } else { "not-enforced" },
            "within_wave_start_barrier": if enroot { "all launches publish ready before any release is created" } else { "not-enforced" },
            "wave_barrier": "all launches return before the next wave is dispatched",
            "container_cpu_placement": if enroot { "taskset-requested-per-launch-affinity" } else { "per-launch cpuset" },
            "completion_index_semantics": "worker-return-order",
            "cpu_exclusivity_verified": false,
        },
        "tasks": task_summaries(&spec.tasks, rows, None),
        "waves": wave_summaries,
        "overall": {
            "attempt_count": rows.len(),
            "declared_attempt_count": declared_attempt_count,
            "completed": status_count(rows, "completed"),
            "failed": status_count(rows, "failed"),
            "valid_duration_count": rows.iter().filter(|row| row_bool(row, "duration_valid")).count(),
            "wave_count": spec.waves.len(),
            "valid_wave_count": valid_wave_count,
            "interrupted": interrupted,
            "outer_run_wall_sec": outer_window.wall_sec(),
            "outer_window_start_ns": outer_window.start_ns,
            "outer_window_end_ns": outer_window.end_ns,
            "clock_resolution_ns": outer_window.resolution_ns,
            "window_containment_tolerance_ns": if enroot { Value::Null } else { Value::from(outer_window.resolution_ns) },
            "valid_for_analysis": valid_for_analysis,
        },
    });
    if enroot {
        let object = summary.as_object_mut().expect("v3 summary is an object");
        object.insert("runtime".to_owned(), Value::String("enroot".to_owned()));
        let scheduling = object
            .get_mut("scheduling")
            .and_then(Value::as_object_mut)
            .expect("v3 scheduling summary is an object");
        scheduling.insert(
            "throughput_wall_clock".to_owned(),
            Value::String("CLOCK_BOOTTIME".to_owned()),
        );
        scheduling.insert(
            "replay_throughput_boundary".to_owned(),
            Value::String(
                "earliest measured replay-command start through latest measured replay-command end"
                    .to_owned(),
            ),
        );
        scheduling.insert(
            "lifecycle_throughput_boundary".to_owned(),
            Value::String(
                "host wave dispatch through Enroot create, preparation, replay, remove, and worker return"
                    .to_owned(),
            ),
        );
        scheduling.insert(
            "independent_performance_unit".to_owned(),
            Value::String("wave".to_owned()),
        );
        let overall = object
            .get_mut("overall")
            .and_then(Value::as_object_mut)
            .expect("v3 overall summary is an object");
        overall.insert(
            "analysis_mode".to_owned(),
            Value::String("wall-only".to_owned()),
        );
        overall.insert(
            "valid_for_wall_only_analysis".to_owned(),
            Value::Bool(valid_for_analysis),
        );
        overall.insert("valid_for_cpu_analysis".to_owned(), Value::Bool(false));
    }
    Ok(summary)
}

fn resolved_atif_path(task: &RunTaskSpec, input_root: &Path) -> String {
    task.atif
        .strip_prefix(input_root)
        .unwrap_or(&task.atif)
        .to_string_lossy()
        .replace(std::path::MAIN_SEPARATOR, "/")
}

fn yaml_string(value: impl Into<String>) -> YamlValue {
    YamlValue::String(value.into())
}

fn yaml_insert(mapping: &mut YamlMapping, key: &str, value: YamlValue) {
    mapping.insert(yaml_string(key), value);
}

fn metadata_yaml(value: &MetadataScalar) -> YamlValue {
    match value {
        MetadataScalar::String(value) => yaml_string(value),
        MetadataScalar::Boolean(value) => YamlValue::Bool(*value),
        MetadataScalar::Integer(value) => {
            serde_yaml::to_value(value).expect("validated metadata integer serializes to YAML")
        }
        MetadataScalar::Number(value) => {
            serde_yaml::to_value(value).expect("validated metadata number serializes to YAML")
        }
    }
}

fn resolved_ordered_payload(
    spec: &RunSpecV3,
    preflights: &HashMap<(String, String), Preflight>,
) -> Result<YamlValue, RunExecutionError> {
    let mut image_ids_by_task = HashMap::<String, String>::new();
    for ((task_id, _), preflight) in preflights {
        if let Some(previous) = image_ids_by_task.get(task_id) {
            if previous != &preflight.image_id {
                return Err(RunExecutionError::new(format!(
                    "resolved image ID changed across placements for task {task_id:?}"
                )));
            }
        } else {
            image_ids_by_task.insert(task_id.clone(), preflight.image_id.clone());
        }
    }

    let mut root = YamlMapping::new();
    yaml_insert(&mut root, "schema_version", yaml_string(SCHEMA_VERSION_V3));
    yaml_insert(
        &mut root,
        "output",
        yaml_string(spec.output.to_string_lossy()),
    );
    yaml_insert(
        &mut root,
        "input_root",
        yaml_string(spec.input_root.to_string_lossy()),
    );
    let mut defaults = YamlMapping::new();
    if spec.defaults.runtime == "enroot" {
        yaml_insert(
            &mut defaults,
            "runtime",
            yaml_string(&spec.defaults.runtime),
        );
    }
    yaml_insert(&mut defaults, "pacing", yaml_string(&spec.defaults.pacing));
    yaml_insert(
        &mut defaults,
        "network_policy",
        yaml_string(&spec.defaults.network_policy),
    );
    yaml_insert(
        &mut defaults,
        "allowed_hosts",
        YamlValue::Sequence(
            spec.defaults
                .allowed_hosts
                .iter()
                .map(yaml_string)
                .collect(),
        ),
    );
    yaml_insert(
        &mut defaults,
        "profile_sets",
        YamlValue::Sequence(spec.defaults.profile_sets.iter().map(yaml_string).collect()),
    );
    yaml_insert(
        &mut defaults,
        "controller_runtime",
        yaml_string(&spec.defaults.controller_runtime),
    );
    yaml_insert(
        &mut defaults,
        "controller_cpus",
        yaml_string(&spec.defaults.controller_cpus),
    );
    yaml_insert(
        &mut defaults,
        "action_timeout_sec",
        serde_yaml::to_value(spec.defaults.action_timeout_sec)
            .expect("finite timeout serializes to YAML"),
    );
    yaml_insert(&mut root, "defaults", YamlValue::Mapping(defaults));

    let mut tasks = Vec::new();
    for task in &spec.tasks {
        let image_id = image_ids_by_task.get(&task.task_id).ok_or_else(|| {
            RunExecutionError::new(format!(
                "resolved plan is missing preflight image ID for task {:?}",
                task.task_id
            ))
        })?;
        let mut item = YamlMapping::new();
        yaml_insert(&mut item, "id", yaml_string(&task.task_id));
        yaml_insert(
            &mut item,
            "atif",
            yaml_string(resolved_atif_path(task, &spec.input_root)),
        );
        yaml_insert(
            &mut item,
            "expected_source_sha256",
            yaml_string(&task.source_sha256),
        );
        yaml_insert(&mut item, "image", yaml_string(&task.image));
        yaml_insert(&mut item, "expected_image_id", yaml_string(image_id));
        if let Some(timeout) = task.action_timeout_sec {
            yaml_insert(
                &mut item,
                "action_timeout_sec",
                serde_yaml::to_value(timeout).expect("finite timeout serializes to YAML"),
            );
        }
        tasks.push(YamlValue::Mapping(item));
    }
    yaml_insert(&mut root, "tasks", YamlValue::Sequence(tasks));

    let mut waves = Vec::new();
    for wave in &spec.waves {
        let mut item = YamlMapping::new();
        yaml_insert(&mut item, "id", yaml_string(&wave.wave_id));
        yaml_insert(&mut item, "scored", YamlValue::Bool(wave.scored));
        let mut metadata = YamlMapping::new();
        for (key, value) in &wave.metadata {
            yaml_insert(&mut metadata, key, metadata_yaml(value));
        }
        yaml_insert(&mut item, "metadata", YamlValue::Mapping(metadata));
        let launches = wave
            .launches
            .iter()
            .map(|launch| {
                let mut item = YamlMapping::new();
                yaml_insert(&mut item, "task", yaml_string(&launch.task_id));
                yaml_insert(&mut item, "cpus", yaml_string(&launch.cpus));
                YamlValue::Mapping(item)
            })
            .collect();
        yaml_insert(&mut item, "launches", YamlValue::Sequence(launches));
        waves.push(YamlValue::Mapping(item));
    }
    yaml_insert(&mut root, "waves", YamlValue::Sequence(waves));
    Ok(YamlValue::Mapping(root))
}

fn write_resolved_ordered_manifest(
    spec: &RunSpecV3,
    preflights: &HashMap<(String, String), Preflight>,
) -> Result<(String, String), RunExecutionError> {
    let filename = "resolved-run.yaml".to_owned();
    let path = spec.output.join(&filename);
    let temporary = spec.output.join(".resolved-run.yaml.tmp");
    let source_manifest = serde_json::to_string(&spec.manifest_path.to_string_lossy())
        .expect("path display serializes to JSON");
    let header = format!(
        "# AUTO-GENERATED: fully resolved tasktrace-run.v3 execution plan.\n\
         # DO NOT EDIT: edit the source manifest and run it again.\n\
         # source_manifest: {source_manifest}\n\
         # source_manifest_sha256: {}\n",
        spec.manifest_sha256
    );
    let body =
        serde_yaml::to_string(&resolved_ordered_payload(spec, preflights)?).map_err(|error| {
            RunExecutionError::new(format!("unable to encode resolved run manifest: {error}"))
        })?;
    fs::write(&temporary, format!("{header}{body}")).map_err(|error| {
        RunExecutionError::new(format!(
            "unable to write resolved run manifest {}: {error}",
            path.display()
        ))
    })?;
    fs::rename(&temporary, &path).map_err(|error| {
        RunExecutionError::new(format!(
            "unable to publish resolved run manifest {}: {error}",
            path.display()
        ))
    })?;
    let digest = file_sha256(&path).map_err(|error| {
        RunExecutionError::new(format!(
            "unable to hash resolved run manifest {}: {error}",
            path.display()
        ))
    })?;
    Ok((filename, digest))
}

fn csv_field(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Bool(value)) => {
            if *value {
                "True".to_owned()
            } else {
                "False".to_owned()
            }
        }
        Some(Value::Number(value)) => value.to_string(),
        Some(value) => value.to_string(),
    }
}

fn write_csv(
    rows: &[Map<String, Value>],
    path: &Path,
    columns: &[&str],
) -> Result<(), RunExecutionError> {
    let filename = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("results.csv");
    let temporary = path.with_file_name(format!(".{filename}.tmp"));
    let mut output = File::create(&temporary).map_err(|error| {
        RunExecutionError::new(format!("unable to write {}: {error}", path.display()))
    })?;
    write_csv_record(
        &mut output,
        columns.iter().map(|column| (*column).to_owned()),
    )?;
    for row in rows {
        write_csv_record(
            &mut output,
            columns.iter().map(|column| csv_field(row.get(*column))),
        )?;
    }
    output.flush()?;
    drop(output);
    fs::rename(&temporary, path).map_err(|error| {
        RunExecutionError::new(format!("unable to publish {}: {error}", path.display()))
    })
}

fn write_csv_record(
    output: &mut File,
    fields: impl IntoIterator<Item = String>,
) -> Result<(), RunExecutionError> {
    let mut first = true;
    for field in fields {
        if !first {
            output.write_all(b",")?;
        }
        first = false;
        if field
            .chars()
            .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
        {
            output.write_all(b"\"")?;
            output.write_all(field.replace('"', "\"\"").as_bytes())?;
            output.write_all(b"\"")?;
        } else {
            output.write_all(field.as_bytes())?;
        }
    }
    output.write_all(b"\n")?;
    Ok(())
}

fn sorted_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sorted_json).collect()),
        Value::Object(values) => {
            let mut sorted = BTreeMap::new();
            for (key, value) in values {
                sorted.insert(key.clone(), sorted_json(value));
            }
            Value::Object(sorted.into_iter().collect())
        }
        value => value.clone(),
    }
}

fn write_json(value: &Value, path: &Path) -> Result<(), RunExecutionError> {
    let filename = path
        .file_name()
        .and_then(OsStr::to_str)
        .unwrap_or("summary.json");
    let temporary = path.with_file_name(format!(".{filename}.tmp"));
    let mut encoded = serde_json::to_string_pretty(&sorted_json(value)).map_err(|error| {
        RunExecutionError::new(format!("unable to encode {}: {error}", path.display()))
    })?;
    encoded.push('\n');
    fs::write(&temporary, encoded).map_err(|error| {
        RunExecutionError::new(format!("unable to write {}: {error}", path.display()))
    })?;
    fs::rename(&temporary, path).map_err(|error| {
        RunExecutionError::new(format!("unable to publish {}: {error}", path.display()))
    })
}

fn host_cpu(cpu_ids: &[u32]) -> Result<Value, RunExecutionError> {
    let cpu_ids: Vec<i64> = cpu_ids.iter().map(|cpu| i64::from(*cpu)).collect();
    collect_host_cpu_identity(&cpu_ids).map_err(|error| {
        RunExecutionError::new(format!("host CPU identity preflight failed: {error}"))
    })
}

fn execute_v1_run(
    spec: &RunSpecV1,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Result<RunExecutionResult, RunExecutionError> {
    ensure_output_absent(&spec.output)?;
    let host = host_cpu(&spec.cpu_ids)?;
    let preflights = preflight_v1(spec, launcher, cancellation)?;
    let controller_digest = native_controller_digest(preflights.values())?;
    claim_output(&spec.output)?;
    let started = monotonic_ns();
    let attempts = attempts_v1(spec);
    let (mut rows, mut interrupted) =
        execute_v1(spec, &attempts, &preflights, launcher, cancellation);
    let outer_window = HostWindow::new(started, monotonic_ns())?;
    invalidate_plan_drift(&mut rows);
    interrupted |= cancellation.is_cancelled();
    let mut summary = summary_v1(
        spec,
        &rows,
        host.clone(),
        &controller_digest,
        interrupted,
        outer_window,
    );
    write_csv(&rows, &spec.output.join("results.csv"), RESULT_COLUMNS)?;
    if cancellation.is_cancelled() && !interrupted {
        interrupted = true;
        summary = summary_v1(
            spec,
            &rows,
            host.clone(),
            &controller_digest,
            interrupted,
            outer_window,
        );
    }
    write_json(&summary, &spec.output.join("summary.json"))?;
    if cancellation.is_cancelled() && !interrupted {
        interrupted = true;
        summary = summary_v1(
            spec,
            &rows,
            host,
            &controller_digest,
            interrupted,
            outer_window,
        );
        write_json(&summary, &spec.output.join("summary.json"))?;
    }
    let all_valid = rows.iter().all(|row| row_bool(row, "duration_valid"));
    Ok(RunExecutionResult {
        exit_code: if interrupted {
            130
        } else if all_valid {
            0
        } else {
            1
        },
        rows,
        summary,
    })
}

fn execute_v2_run(
    spec: &RunSpecV2,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Result<RunExecutionResult, RunExecutionError> {
    ensure_output_absent(&spec.output)?;
    let host = host_cpu(&spec.cpu_ids)?;
    let preflights = preflight_v2(spec, launcher, cancellation)?;
    let controller_digest = native_controller_digest(preflights.values())?;
    claim_output(&spec.output)?;
    let started = monotonic_ns();
    let attempts = attempts_v2(spec);
    let (mut rows, mut interrupted) =
        execute_v2(spec, &attempts, &preflights, launcher, cancellation);
    let outer_window = HostWindow::new(started, monotonic_ns())?;
    invalidate_window_containment(&mut rows, outer_window, None);
    invalidate_plan_drift(&mut rows);
    interrupted |= cancellation.is_cancelled();
    let mut summary = summary_v2(
        spec,
        &rows,
        host.clone(),
        &controller_digest,
        interrupted,
        outer_window,
    )?;
    write_csv(&rows, &spec.output.join("results.csv"), RESULT_COLUMNS_V2)?;
    if cancellation.is_cancelled() && !interrupted {
        interrupted = true;
        summary = summary_v2(
            spec,
            &rows,
            host.clone(),
            &controller_digest,
            interrupted,
            outer_window,
        )?;
    }
    write_json(&summary, &spec.output.join("summary.json"))?;
    if cancellation.is_cancelled() && !interrupted {
        interrupted = true;
        summary = summary_v2(
            spec,
            &rows,
            host,
            &controller_digest,
            interrupted,
            outer_window,
        )?;
        write_json(&summary, &spec.output.join("summary.json"))?;
    }
    let all_valid = rows.iter().all(|row| row_bool(row, "duration_valid"));
    let throughput_valid = summary
        .get("overall")
        .and_then(|value| value.get("valid_for_throughput"))
        .and_then(Value::as_bool)
        == Some(true);
    Ok(RunExecutionResult {
        exit_code: if interrupted {
            130
        } else if all_valid && throughput_valid {
            0
        } else {
            1
        },
        rows,
        summary,
    })
}

fn execute_v3_run(
    spec: &RunSpecV3,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Result<RunExecutionResult, RunExecutionError> {
    validate_ordered_runtime_environment(spec)?;
    ensure_output_absent(&spec.output)?;
    let attempts = wave_attempts(spec);
    let selected_cpu_ids: Vec<u32> = spec
        .defaults
        .controller_cpu_ids
        .iter()
        .copied()
        .chain(
            spec.waves
                .iter()
                .flat_map(|wave| &wave.launches)
                .flat_map(|launch| launch.cpu_ids.iter().copied()),
        )
        .collect::<BTreeSet<_>>()
        .into_iter()
        .collect();
    let enroot_provenance_config = (spec.defaults.runtime == "enroot").then(|| {
        let launch_cpu_sets = spec
            .waves
            .iter()
            .flat_map(|wave| &wave.launches)
            .map(|launch| launch.cpus.clone())
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect();
        let mut provenance_config =
            EnrootProvenanceConfig::new(spec.defaults.controller_cpus.clone(), launch_cpu_sets);
        // Bind runtime identity to the containment executable that launches
        // actually use (bubblewrap on hosts where util-linux unshare cannot
        // create user namespaces), not the default name.
        let (containment_executable, _) = crate::enroot_runtime::detect_containment();
        provenance_config.unshare = containment_executable;
        provenance_config
    });
    let (
        (
            host,
            resolved_manifest,
            resolved_digest,
            controller_digest,
            mut rows,
            mut interrupted,
            wall,
            waves,
            enroot_provenance,
            enroot_provenance_validation_errors,
        ),
        controller_affinity,
    ) = with_controller_affinity(spec, |_evidence| {
        let host = host_cpu(&selected_cpu_ids)?;
        let enroot_provenance = enroot_provenance_config
            .as_ref()
            .map(|config| {
                capture_runtime_provenance(config).map_err(|error| {
                    RunExecutionError::new(format!(
                        "Enroot runtime identity preflight failed: {error}"
                    ))
                })
            })
            .transpose()?;
        let preflights = preflight_v3(
            spec,
            &attempts,
            launcher,
            cancellation,
            enroot_provenance.as_ref().map(|provenance| {
                (
                    provenance.comparability_sha256.as_str(),
                    Path::new(&provenance.executable),
                )
            }),
        )?;
        let controller_digest = native_controller_digest(preflights.values())?;
        claim_output(&spec.output)?;
        let (resolved_manifest, resolved_digest) =
            write_resolved_ordered_manifest(spec, &preflights)?;
        let started = ordered_clock_ns(&spec.defaults.runtime)?;
        let mut wave_reattest = |_wave_index: usize, _phase: EnrootWaveAttestationPhase| match (
            enroot_provenance_config.as_ref(),
            enroot_provenance.as_ref(),
        ) {
            (Some(config), Some(baseline)) => recapture_and_validate(config, baseline).is_ok(),
            (None, None) => true,
            _ => false,
        };
        let reattestor = (spec.defaults.runtime == "enroot")
            .then_some(&mut wave_reattest as &mut EnrootWaveReattestor<'_>);
        let (rows, interrupted, wave_windows, wave_attestation_failure) = execute_v3(
            spec,
            &attempts,
            &preflights,
            launcher,
            cancellation,
            reattestor,
        )?;
        let outer_window = HostWindow::for_runtime(
            &spec.defaults.runtime,
            started,
            ordered_clock_ns(&spec.defaults.runtime)?,
        )?;
        let final_attestation_failure = enroot_provenance_config
            .as_ref()
            .zip(enroot_provenance.as_ref())
            .and_then(|(config, before)| {
                recapture_and_validate(config, before)
                    .err()
                    .map(|_| "Enroot runtime provenance final re-attestation failed".to_owned())
            });
        let enroot_provenance_validation_errors = wave_attestation_failure
            .into_iter()
            .chain(final_attestation_failure)
            .collect::<Vec<_>>();
        Ok((
            host,
            resolved_manifest,
            resolved_digest,
            controller_digest,
            rows,
            interrupted,
            outer_window,
            wave_windows,
            enroot_provenance,
            enroot_provenance_validation_errors,
        ))
    })?;
    if spec.defaults.runtime == "enroot" {
        for reason in &enroot_provenance_validation_errors {
            for row in &mut rows {
                row.insert("duration_valid".to_owned(), Value::Bool(false));
                append_reason_once(row, reason);
            }
        }
        invalidate_changed_enroot_images(spec, &mut rows);
        for (wave_index, window) in &waves {
            invalidate_enroot_duration_bounds(&mut rows, *window, *wave_index);
        }
    } else {
        for (wave_index, window) in &waves {
            invalidate_window_containment(&mut rows, *window, Some(*wave_index));
        }
    }
    invalidate_plan_drift(&mut rows);
    let build_summary = |interrupted: bool| -> Result<Value, RunExecutionError> {
        let mut summary = summary_v3(
            spec,
            &rows,
            host.clone(),
            interrupted,
            wall,
            &waves,
            controller_affinity.clone(),
        )?;
        let summary_object = summary.as_object_mut().expect("v3 summary is object");
        summary_object.insert(
            "resolved_manifest".to_owned(),
            Value::String(resolved_manifest.clone()),
        );
        summary_object.insert(
            "resolved_manifest_sha256".to_owned(),
            Value::String(resolved_digest.clone()),
        );
        summary_object.insert(
            "controller_binary_sha256".to_owned(),
            Value::String(controller_digest.clone()),
        );
        if let Some(provenance) = enroot_provenance.as_ref() {
            summary_object.insert(
                "enroot_runtime_provenance".to_owned(),
                serde_json::to_value(provenance).map_err(|error| {
                    RunExecutionError::new(format!(
                        "unable to serialize Enroot runtime provenance: {error}"
                    ))
                })?,
            );
            summary_object.insert(
                "enroot_runtime_provenance_unchanged".to_owned(),
                Value::Bool(enroot_provenance_validation_errors.is_empty()),
            );
            if !enroot_provenance_validation_errors.is_empty() {
                summary_object.insert(
                    "enroot_runtime_provenance_validation_error".to_owned(),
                    Value::String(enroot_provenance_validation_errors.join("; ")),
                );
                summary_object.insert(
                    "enroot_runtime_provenance_validation_errors".to_owned(),
                    json!(enroot_provenance_validation_errors),
                );
            }
        }
        Ok(summary)
    };
    interrupted |= cancellation.is_cancelled();
    let mut summary = build_summary(interrupted)?;
    let mut result_columns = RESULT_COLUMNS_V3.to_vec();
    if spec.defaults.runtime == "enroot" {
        result_columns.extend_from_slice(ENROOT_RESULT_COLUMNS);
    }
    write_csv(&rows, &spec.output.join("results.csv"), &result_columns)?;
    if cancellation.is_cancelled() && !interrupted {
        interrupted = true;
        summary = build_summary(interrupted)?;
    }
    write_json(&summary, &spec.output.join("summary.json"))?;
    if cancellation.is_cancelled() && !interrupted {
        interrupted = true;
        summary = build_summary(interrupted)?;
        write_json(&summary, &spec.output.join("summary.json"))?;
    }
    let valid = summary
        .get("overall")
        .and_then(|value| value.get("valid_for_analysis"))
        .and_then(Value::as_bool)
        == Some(true);
    Ok(RunExecutionResult {
        exit_code: if interrupted {
            130
        } else if valid {
            0
        } else {
            1
        },
        rows,
        summary,
    })
}

/// Execute a loaded manifest with an explicit launcher and cancellation token.
pub fn execute_run_with_launcher_and_cancellation(
    spec: &RunSpec,
    launcher: &dyn AttemptLauncher,
    cancellation: &CancellationToken,
) -> Result<RunExecutionResult, RunExecutionError> {
    match spec {
        RunSpec::V1(spec) => execute_v1_run(spec, launcher, cancellation),
        RunSpec::V2(spec) => execute_v2_run(spec, launcher, cancellation),
        RunSpec::V3(spec) => execute_v3_run(spec, launcher, cancellation),
    }
}

/// Execute a loaded manifest with an injectable profile launcher.
#[allow(dead_code)] // Public injection boundary retained for embedding and contract tests.
pub fn execute_run_with_launcher(
    spec: &RunSpec,
    launcher: &dyn AttemptLauncher,
) -> Result<RunExecutionResult, RunExecutionError> {
    execute_run_with_launcher_and_cancellation(spec, launcher, &CancellationToken::new())
}

/// Execute a loaded manifest using the product's public profile leaf.
#[allow(dead_code)] // Convenience API; the CLI uses the cancellation-aware form.
pub fn execute_run(spec: &RunSpec) -> Result<RunExecutionResult, RunExecutionError> {
    let launcher = PublicProfileLauncher::default();
    execute_run_with_launcher(spec, &launcher)
}

fn print_help(public_program: &str) {
    println!("Run ATIF tasks from a strict YAML file in fresh runtime sandboxes.");
    println!();
    println!("usage: {public_program} [-h] FILE.yaml");
    println!();
    println!("v1 runs sequentially; v2 uses bounded concurrent execution; v3 uses ordered waves.");
    println!();
    println!("positional arguments:");
    println!("  FILE.yaml   Strict tasktrace-run YAML manifest.");
    println!();
    println!("options:");
    println!("  -h, --help  show this help message and exit");
}

fn cli_error(public_program: &str, message: impl fmt::Display) -> i32 {
    eprintln!("usage: {public_program} [-h] FILE.yaml");
    eprintln!("{public_program}: error: {message}");
    2
}

fn print_result(spec: &RunSpec, result: &RunExecutionResult) {
    let overall = &result.summary["overall"];
    println!(
        "Completed: {}; failed: {}; valid durations: {}",
        overall["completed"].as_u64().unwrap_or(0),
        overall["failed"].as_u64().unwrap_or(0),
        overall["valid_duration_count"].as_u64().unwrap_or(0),
    );
    println!("Results: {}", spec.output().join("results.csv").display());
    println!("Summary: {}", spec.output().join("summary.json").display());
    match spec {
        RunSpec::V2(_) => {
            let throughput = overall["throughput_tasks_per_sec"].as_f64();
            let utilization = overall["pool_utilization_percent"].as_f64();
            if let (Some(throughput), Some(utilization)) = (throughput, utilization) {
                println!(
                    "Throughput: {throughput:.6} tasks/sec; pool utilization: {utilization:.2}%"
                );
            } else {
                println!("Throughput: unavailable because the concurrent run was invalid");
            }
        }
        RunSpec::V3(_) => println!(
            "Waves valid: {}/{}; per-wave latency, throughput, and utilization are in summary.json",
            overall["valid_wave_count"].as_u64().unwrap_or(0),
            overall["wave_count"].as_u64().unwrap_or(0),
        ),
        RunSpec::V1(_) => {}
    }
}

/// Native `tasktrace run FILE.yaml` leaf.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    if arguments.len() == 1 && matches!(arguments[0].to_str(), Some("-h" | "--help")) {
        print_help(public_program);
        return 0;
    }
    if arguments.len() != 1
        || arguments[0]
            .to_str()
            .is_some_and(|value| value.starts_with('-'))
    {
        return cli_error(
            public_program,
            "the following arguments are required: FILE.yaml",
        );
    }
    let spec = match load_run_spec(Path::new(&arguments[0])) {
        Ok(spec) => spec,
        Err(error) => return cli_error(public_program, error),
    };
    let signal_guard = match SignalGuard::install() {
        Ok(guard) => guard,
        Err(error) => return cli_error(public_program, error),
    };
    let cancellation = CancellationToken::observing_process_signal();
    let launcher = PublicProfileLauncher::default();
    let result = execute_run_with_launcher_and_cancellation(&spec, &launcher, &cancellation);
    drop(signal_guard);
    match result {
        Ok(result) => {
            print_result(&spec, &result);
            result.exit_code
        }
        Err(error) if cancellation.is_cancelled() => {
            eprintln!("{public_program}: interrupted: {error}");
            130
        }
        Err(error) => cli_error(public_program, error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::Mutex;

    fn docker_output_with(
        status: i32,
        stdout: impl Into<Vec<u8>>,
        stderr: impl Into<Vec<u8>>,
    ) -> std::process::Output {
        std::process::Output {
            status: ExitStatus::from_raw(status << 8),
            stdout: stdout.into(),
            stderr: stderr.into(),
        }
    }

    #[test]
    fn docker_absence_diagnostics_accept_lowercase_docker_29_errors() {
        for (kind, stderr) in [
            ("container", "error: no such object: deadbeef"),
            ("container", "Error: No such container: deadbeef"),
            (
                "container",
                "error response from daemon: no such object: deadbeef",
            ),
            (
                "container",
                "error response from daemon: no such container: deadbeef",
            ),
            (
                "network",
                "error: no such object: tasktrace-policy-deadbeef",
            ),
            (
                "network",
                "error: no such network: tasktrace-policy-deadbeef",
            ),
            (
                "network",
                "error response from daemon: no such object: tasktrace-policy-deadbeef",
            ),
            (
                "network",
                "error response from daemon: no such network: tasktrace-policy-deadbeef",
            ),
            (
                "network",
                "error response from daemon: network tasktrace-policy-deadbeef not found",
            ),
        ] {
            let output = docker_output_with(1, b"[]\n", stderr.as_bytes());
            assert!(docker_object_absent(kind, &output), "{kind}: {stderr}");
        }
    }

    #[test]
    fn docker_absence_diagnostics_preserve_fail_closed_errors() {
        for stderr in [
            "permission denied while trying to connect to the Docker daemon socket",
            "cannot connect to the Docker daemon",
            "error response from daemon: context deadline exceeded",
            "permission denied while inspecting; no such object: deadbeef",
            "error: permission denied; no such container: deadbeef",
            "error: permission denied\nerror: no such object: deadbeef",
            "error: no such object: deadbeef; permission denied",
        ] {
            let output = docker_output_with(1, Vec::new(), stderr.as_bytes());
            assert!(!docker_object_absent("container", &output), "{stderr}");
            assert!(!docker_object_absent("network", &output), "{stderr}");
        }

        let unexpected_stdout = docker_output_with(
            1,
            b"partial inspect payload\n",
            b"error: no such object: deadbeef\n",
        );
        assert!(!docker_object_absent("container", &unexpected_stdout));

        let successful = docker_output_with(0, Vec::new(), b"error: no such object: deadbeef\n");
        assert!(!docker_object_absent("container", &successful));
    }

    fn native_preflight_request(root: &Path) -> ProfileRequest {
        ProfileRequest {
            task_id: "native-preflight".to_owned(),
            trajectory: root.join("missing-trajectory.json"),
            image: format!("sha256:{}", "a".repeat(64)),
            output_dir: root.join("profile"),
            cpuset_cpus: "0".to_owned(),
            pacing: crate::pacing::UNPACED.to_owned(),
            timeout_sec: Some(5.0),
            runtime: "docker".to_owned(),
            enroot_executable: None,
            controller_runtime: "rust-host-mount".to_owned(),
            profile_sets: vec!["cpu-average".to_owned()],
            network_policy: Some("no-network".to_owned()),
            allowed_hosts: Vec::new(),
            workdir: None,
            environment: Vec::new(),
        }
    }

    #[test]
    fn direct_native_network_enforcement_report_seeds_forced_cleanup() {
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("profile");
        fs::create_dir(&output).unwrap();
        fs::write(
            output.join("network_enforcement.json"),
            serde_json::to_vec(&json!({
                "schema_version": "tasktrace-docker-network-enforcement.v1",
                "ephemeral_internal_network": "tasktrace-policy-a1b2c3d4",
                "proxy": {"container_name": "tasktrace-policy-proxy-a1b2c3d4"},
            }))
            .unwrap(),
        )
        .unwrap();
        let mut resources = ScopedDockerResources::default();
        add_network_enforcement_resources(&output, &mut resources);
        assert!(resources.networks.contains("tasktrace-policy-a1b2c3d4"));
        assert!(resources
            .proxy_containers
            .contains("tasktrace-policy-proxy-a1b2c3d4"));
        assert!(resources
            .state_paths
            .contains(&env::temp_dir().join(".tasktrace-network-policy-state-a1b2c3d4")));
    }

    #[test]
    fn rust_host_mount_preflight_does_not_start_python() {
        let root = tempfile::tempdir().unwrap();
        let launcher = PublicProfileLauncher::new(env::current_exe().unwrap());
        let error = launcher
            .preflight(&native_preflight_request(root.path()))
            .unwrap_err()
            .to_string();
        assert!(error.contains("native profile preflight failed"), "{error}");
        assert!(error.contains("missing-trajectory.json"), "{error}");
        assert!(
            !error.contains("unable to start profile preflight"),
            "{error}"
        );
    }

    #[test]
    fn allowlist_preflight_with_hosts_does_not_start_python() {
        let root = tempfile::tempdir().unwrap();
        let launcher = PublicProfileLauncher::new(env::current_exe().unwrap());
        let mut request = native_preflight_request(root.path());
        request.network_policy = Some("allowlist".to_owned());
        request.allowed_hosts = vec!["api.example.com".to_owned(), "files.example.com".to_owned()];
        let error = launcher.preflight(&request).unwrap_err().to_string();
        assert!(error.contains("native profile preflight failed"), "{error}");
        assert!(error.contains("missing-trajectory.json"), "{error}");
        assert!(
            !error.contains("unable to start profile preflight"),
            "{error}"
        );
    }

    #[test]
    fn multi_set_preflight_does_not_start_python() {
        let root = tempfile::tempdir().unwrap();
        let launcher = PublicProfileLauncher::new(env::current_exe().unwrap());
        let mut request = native_preflight_request(root.path());
        request.profile_sets = vec![
            "cpu-average".to_owned(),
            "memory-detail".to_owned(),
            "hardware-deep".to_owned(),
        ];
        let error = launcher.preflight(&request).unwrap_err().to_string();
        assert!(error.contains("native profile preflight failed"), "{error}");
        assert!(error.contains("missing-trajectory.json"), "{error}");
        assert!(
            !error.contains("unable to start profile preflight"),
            "{error}"
        );
    }

    #[test]
    fn every_preflight_uses_the_native_parser() {
        let root = tempfile::tempdir().unwrap();
        let launcher = PublicProfileLauncher::new(env::current_exe().unwrap());
        let mut request = native_preflight_request(root.path());
        request.controller_runtime = "rust-host-mount".to_owned();
        let error = launcher.preflight(&request).unwrap_err().to_string();
        assert!(error.contains("native profile preflight failed"), "{error}");
        assert!(
            !error.contains("unable to start profile preflight"),
            "{error}"
        );
    }

    #[derive(Default)]
    struct MockLauncher {
        calls: Mutex<Vec<(String, String)>>,
    }

    #[derive(Default)]
    struct CancelAfterFirstLauncher {
        inner: MockLauncher,
        cancelled: AtomicBool,
    }

    #[derive(Default)]
    struct RecordingEnrootFailureLauncher {
        requests: Mutex<Vec<ProfileRequest>>,
    }

    impl AttemptLauncher for RecordingEnrootFailureLauncher {
        fn preflight(&self, _request: &ProfileRequest) -> Result<Value, RunExecutionError> {
            panic!("wave re-attestation tests use retained preflights")
        }

        fn run_profile(
            &self,
            request: &ProfileRequest,
            _cancellation: &CancellationToken,
        ) -> Result<Value, RunExecutionError> {
            self.requests.lock().unwrap().push(request.clone());
            thread::sleep(Duration::from_millis(1));
            Err(RunExecutionError::new("synthetic profile failure"))
        }
    }

    impl MockLauncher {
        fn image_id(request: &ProfileRequest) -> String {
            format!("sha256:{}", request.task_id)
        }

        fn cpu_count(cpus: &str) -> usize {
            cpus.split(',')
                .map(|item| {
                    item.split_once('-').map_or(1, |(first, last)| {
                        last.parse::<usize>().unwrap() - first.parse::<usize>().unwrap() + 1
                    })
                })
                .sum()
        }
    }

    impl AttemptLauncher for MockLauncher {
        fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
            let mut report = json!({
                "ready": true,
                "runtime": request.runtime,
                "source_sha256": file_sha256(&request.trajectory)?,
                "image_identity": {"id": Self::image_id(request)},
                "profile_sets": request.profile_sets,
                "cpuset_cpus": request.cpuset_cpus,
            });
            if request.controller_runtime == "rust-host-mount" {
                report["controller_identity"] = json!({
                    "kind": "rust-host-mounted-binary",
                    "sha256": "c".repeat(64),
                });
            }
            Ok(report)
        }

        fn run_profile(
            &self,
            request: &ProfileRequest,
            _cancellation: &CancellationToken,
        ) -> Result<Value, RunExecutionError> {
            self.calls
                .lock()
                .unwrap()
                .push((request.task_id.clone(), request.cpuset_cpus.clone()));
            let start = monotonic_ns();
            thread::sleep(Duration::from_millis(2));
            let end = monotonic_ns();
            let wall = (end - start) as f64 / 1_000_000_000.0;
            let plan = format!(
                "{:x}",
                Sha256::digest(format!("{}:{}", request.task_id, request.cpuset_cpus))
            );
            let mut report = json!({
                "source_trajectory_sha256": file_sha256(&request.trajectory)?,
                "image_identity": {"id": Self::image_id(request)},
                "container_cpu": {
                    "wall_sec": wall,
                    "valid_for_cpu_performance": true,
                    "window_start_ns": start,
                    "window_end_ns": end,
                    "usage_sec": 0.0,
                    "average_logical_cores": 0.0,
                },
                "replay_summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                },
                "profiling_valid": true,
                "status": "valid",
                "execution_plan_digest": plan,
                "canonical_clock_reference": clock_reference(),
                "scope": {"identity": {"cpu_placement": {
                    "requested_cpuset_cpus": request.cpuset_cpus,
                    "docker_host_config_cpuset_cpus": request.cpuset_cpus,
                    "container_init_cpus_allowed_list": request.cpuset_cpus,
                    "cgroup_cpuset_cpus_effective": request.cpuset_cpus,
                    "validation_status": "passed",
                }}},
                "execution_plan": {
                    "extensions": {"cpuset_cpus": request.cpuset_cpus},
                    "source_identity": {
                        "rust_controller_binary_sha256": if request.controller_runtime == "rust-host-mount" { Value::String("c".repeat(64)) } else { Value::Null }
                    }
                },
            });
            report["container_cpu"]["allocated_cpu_source"] =
                Value::String("cpuset-cpus".to_owned());
            report["container_cpu"]["allocated_logical_cpus"] =
                json!(Self::cpu_count(&request.cpuset_cpus) as f64);
            report["controller_runtime"] = Value::String(request.controller_runtime.clone());
            if request.controller_runtime == "rust-host-mount" {
                report["controller_identity"] = json!({
                    "kind": "rust-host-mounted-binary",
                    "sha256": "c".repeat(64),
                });
            }
            if let Some(policy) = &request.network_policy {
                let digest = NetworkPolicy::parse(policy, &request.allowed_hosts)
                    .unwrap()
                    .comparability_digest();
                report["pacing"] = Value::String(request.pacing.clone());
                report["profile_provenance"] = json!({"profile_sets": request.profile_sets});
                report["network_policy"] = json!({"comparability_sha256": digest});
            }
            Ok(report)
        }
    }

    impl AttemptLauncher for CancelAfterFirstLauncher {
        fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
            self.inner.preflight(request)
        }

        fn run_profile(
            &self,
            request: &ProfileRequest,
            cancellation: &CancellationToken,
        ) -> Result<Value, RunExecutionError> {
            let report = self.inner.run_profile(request, cancellation)?;
            if !self.cancelled.swap(true, Ordering::SeqCst) {
                cancellation.cancel();
            }
            Ok(report)
        }
    }

    fn available_cpus() -> Vec<u32> {
        let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
        let result =
            unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
        assert_eq!(result, 0);
        (0..libc::CPU_SETSIZE as usize)
            .filter(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
            .map(|cpu| cpu as u32)
            .collect()
    }

    fn write_sources(root: &Path) {
        fs::create_dir_all(root).unwrap();
        fs::write(root.join("one.json"), b"{\"task\":1}\n").unwrap();
        fs::write(root.join("two.json"), b"{\"task\":2}\n").unwrap();
    }

    fn write_v1_or_v2(root: &Path, cpus: u32, version: u8) -> PathBuf {
        write_sources(root);
        let concurrency = if version == 2 { "concurrency: 2\n" } else { "" };
        let manifest = root.join("run.yaml");
        fs::write(
            &manifest,
            format!(
                "schema_version: tasktrace-run.v{version}\noutput: results\ncpus: '{cpus}'\nrepetitions: 2\n{concurrency}tasks:\n  - id: one\n    atif: one.json\n    image: image-one\n  - id: two\n    atif: two.json\n    image: image-two\n"
            ),
        )
        .unwrap();
        manifest
    }

    fn write_v3(root: &Path, controller: u32, worker: u32) -> PathBuf {
        write_sources(root);
        let manifest = root.join("run.yaml");
        fs::write(
            &manifest,
            format!(
                "schema_version: tasktrace-run.v3\noutput: results\ndefaults:\n  pacing: llm-gaps\n  network_policy: public\n  profile_sets: [cpu-average]\n  controller_runtime: rust-host-mount\n  controller_cpus: '{controller}'\n  action_timeout_sec: 90\ntasks:\n  - id: one\n    atif: one.json\n    image: image-one@sha256:one\n    expected_image_id: sha256:one\n  - id: two\n    atif: two.json\n    image: image-two@sha256:two\n    expected_image_id: sha256:two\nwaves:\n  - id: wave-one\n    scored: true\n    metadata:\n      repeat: 1\n    launches:\n      - task: one\n        cpus: '{worker}'\n      - task: two\n        cpus: '{worker}'\n"
            ),
        )
        .unwrap();
        manifest
    }

    fn enroot_fixture(root: &Path) -> (RunTaskSpec, RunSpecV3, Preflight) {
        fs::create_dir_all(root).unwrap();
        let trajectory = root.join("task.json");
        let image = root.join("task.sqsh");
        fs::write(&trajectory, b"{\"task\":1}\n").unwrap();
        fs::write(&image, b"immutable-sqsh-fixture\n").unwrap();
        let image = image.canonicalize().unwrap();
        let metadata = fs::metadata(&image).unwrap();
        let image_digest = file_sha256(&image).unwrap();
        let image_identity = json!({
            "runtime": "enroot",
            "kind": "image-file",
            "image": image.to_string_lossy(),
            "path": image.to_string_lossy(),
            "size_bytes": metadata.len(),
            "mtime_ns": metadata_mtime_ns(&metadata),
            "sha256": image_digest,
            "immutable": true,
        });
        let task = RunTaskSpec {
            task_id: "task".to_owned(),
            atif: trajectory.clone(),
            source_sha256: file_sha256(&trajectory).unwrap(),
            image: image.to_string_lossy().into_owned(),
            expected_image_id: Some(format!("sha256:{image_digest}")),
            action_timeout_sec: None,
        };
        let launches = vec![
            RunLaunchSpec {
                task_id: task.task_id.clone(),
                cpus: "4".to_owned(),
                cpu_ids: vec![4],
            },
            RunLaunchSpec {
                task_id: task.task_id.clone(),
                cpus: "5".to_owned(),
                cpu_ids: vec![5],
            },
        ];
        let spec = RunSpecV3 {
            schema_version: SCHEMA_VERSION_V3.to_owned(),
            manifest_path: root.join("run.yaml"),
            manifest_sha256: "d".repeat(64),
            output: root.join("results"),
            input_root: root.to_path_buf(),
            defaults: crate::run_spec::RunDefaultsSpec {
                pacing: crate::pacing::UNPACED.to_owned(),
                network_policy: "no-network".to_owned(),
                allowed_hosts: Vec::new(),
                runtime: "enroot".to_owned(),
                profile_sets: Vec::new(),
                controller_runtime: "rust-host-mount".to_owned(),
                controller_cpus: "0".to_owned(),
                controller_cpu_ids: vec![0],
                action_timeout_sec: 30.0,
                worker_cpus: None,
                worker_cpu_ids: Vec::new(),
                placement: None,
                workdir: None,
                environment: Vec::new(),
            },
            tasks: vec![task.clone()],
            waves: vec![RunWaveSpec {
                wave_id: "wave".to_owned(),
                scored: true,
                metadata: Vec::new(),
                launches,
            }],
        };
        let preflight = Preflight {
            image_id: format!("sha256:{image_digest}"),
            image_identity,
            controller_sha256: Some("c".repeat(64)),
            enroot_runtime_provenance_sha256: Some("e".repeat(64)),
            enroot_executable: Some(PathBuf::from("/opt/enroot/bin/enroot")),
        };
        (task, spec, preflight)
    }

    fn enroot_wave_execution_fixture(
        root: &Path,
    ) -> (
        RunSpecV3,
        Vec<WaveAttempt>,
        HashMap<(String, String), Preflight>,
    ) {
        let cpus = available_cpus();
        assert!(cpus.len() >= 2);
        let (_task, mut spec, preflight) = enroot_fixture(root);
        spec.defaults.controller_cpus = cpus[0].to_string();
        spec.defaults.controller_cpu_ids = vec![cpus[0]];
        for launch in &mut spec.waves[0].launches {
            launch.cpus = cpus[1].to_string();
            launch.cpu_ids = vec![cpus[1]];
        }
        let mut second_wave = spec.waves[0].clone();
        second_wave.wave_id = "wave-two".to_owned();
        spec.waves.push(second_wave);
        let attempts = wave_attempts(&spec);
        let preflights = attempts
            .iter()
            .map(|attempt| {
                (
                    (attempt.task.task_id.clone(), attempt.launch.cpus.clone()),
                    preflight.clone(),
                )
            })
            .collect();
        (spec, attempts, preflights)
    }

    #[test]
    fn enroot_wave_start_coordinator_releases_only_after_every_ready_marker() {
        let root = tempfile::tempdir().unwrap();
        let (_task, spec, _preflight) = enroot_fixture(root.path());
        let attempts = wave_attempts(&spec);
        for attempt in &attempts {
            let ready = wave_start_control_path(&spec, attempt, RUN_WAVE_START_READY);
            fs::create_dir_all(ready.parent().unwrap()).unwrap();
            fs::write(ready, RUN_WAVE_START_READY_BYTES).unwrap();
        }
        let (_sender, receiver) = mpsc::channel();
        let mut completed = Vec::new();
        coordinate_enroot_wave_start(
            &spec,
            &attempts,
            &receiver,
            &mut completed,
            &CancellationToken::new(),
            Duration::from_millis(100),
        )
        .unwrap();
        assert!(completed.is_empty());
        for attempt in &attempts {
            assert_eq!(
                fs::read(wave_start_control_path(
                    &spec,
                    attempt,
                    RUN_WAVE_START_RELEASE,
                ))
                .unwrap(),
                RUN_WAVE_START_RELEASE_BYTES,
            );
        }

        let (sender, receiver) = mpsc::channel();
        sender
            .send((attempts[0].attempt_index, Map::new()))
            .unwrap();
        let mut completed = Vec::new();
        let error = coordinate_enroot_wave_start(
            &spec,
            &attempts,
            &receiver,
            &mut completed,
            &CancellationToken::new(),
            Duration::from_millis(100),
        )
        .unwrap_err();
        assert!(error.to_string().contains("completed before"));
        assert_eq!(completed.len(), 1);
    }

    fn enroot_report(
        task: &RunTaskSpec,
        spec: &RunSpecV3,
        launch: &RunLaunchSpec,
        preflight: &Preflight,
        start_ns: u64,
        end_ns: u64,
        boot_id: &str,
    ) -> Value {
        let resolution_ns = 10_000_000_u64;
        let clock = json!({
            "clock_name": "CLOCK_BOOTTIME",
            "source": ENROOT_TASK_CLOCK_SOURCE,
            "resolution_ns": resolution_ns,
            "boot_id": boot_id,
            "time_namespace": "time:[4026531834]",
        });
        let parent_netns_inode = current_network_namespace_inode().unwrap();
        let child_netns_inode = parent_netns_inode
            .checked_add(1)
            .unwrap_or(parent_netns_inode - 1);
        let network_digest =
            NetworkPolicy::parse(&spec.defaults.network_policy, &spec.defaults.allowed_hosts)
                .unwrap()
                .comparability_digest();
        let wall_sec = (end_ns - start_ns) as f64 / 1_000_000_000.0;
        json!({
            "runtime": "enroot",
            "controller_runtime": spec.defaults.controller_runtime,
            "controller_identity": {
                "kind": "rust-host-mounted-binary",
                "sha256": preflight.controller_sha256,
            },
            "container_cpu": {
                "method": "wall-clock-only-no-cgroup",
                "wall_clock_source": ENROOT_TASK_CLOCK_SOURCE,
                "wall_sec": wall_sec,
                "usage_sec": null,
                "user_sec": null,
                "system_sec": null,
                "average_logical_cores": null,
                "average_logical_cores_used": null,
                "utilization_fraction_of_allocation": null,
                "utilization_percent_of_allocation": null,
                "valid_for_wall_time_reporting": true,
                "valid_for_cpu_performance": false,
                "cpu_containment_enforced": false,
                "cpu_affinity_enforced": false,
                "allocated_cpu_source": "taskset-requested-affinity",
                "allocated_logical_cpus": launch.cpu_ids.len(),
                "requested_cpu_affinity": launch.cpus,
                "initial_effective_logical_cpu_ids": launch.cpu_ids,
                "window_start_ns": start_ns,
                "window_end_ns": end_ns,
                "clock_reference": clock,
            },
            "image": task.image,
            "image_identity": preflight.image_identity,
            "pacing": spec.defaults.pacing,
            "profile_provenance": {"profile_sets": spec.defaults.profile_sets},
            "network_policy": {
                "effective_mode": "no-network",
                "comparability_sha256": network_digest,
                "enforcement": {
                    "schema_version": "tasktrace-enroot-network-enforcement.v1",
                    "state": "enforced",
                    "backend": "linux-user-network-namespace-v1",
                    "policy_effective_mode": "no-network",
                    "valid": true,
                    "parent_netns_inode": parent_netns_inode,
                    "netns_inode": child_netns_inode,
                    "netns_differs": true,
                    "controller_euid": unsafe { libc::geteuid() },
                    "controller_egid": unsafe { libc::getegid() },
                    "namespace_euid": unsafe { libc::geteuid() },
                    "namespace_egid": unsafe { libc::getegid() },
                    "interfaces": ["lo"],
                    "ipv4_routes": [],
                    "ipv6_addresses": [],
                    "publication": {
                        "source": "controller-private-temporary-state",
                        "published_after_enroot_exit": true,
                    },
                },
            },
            "replay_summary": {
                "actions_timed_out": 0,
                "source_outcome_mismatch_count": 0,
            },
            "profiling_valid": true,
            "status": "valid",
            "source_trajectory_sha256": task.source_sha256,
            "enroot_lifecycle": {
                "schema_version": "tasktrace-enroot-lifecycle.v1",
                "clock_reference": clock,
                "process_containment": {
                    "backend": "linux-user-pid-namespace-v1",
                    "namespace_init_exit_kills_remaining_processes": true,
                    "outer_exit_kills_namespace_init": true,
                },
                "cleanup": {"valid": true, "container_absent": true},
                "phases": [{
                    "phase": "start",
                    "start_ns": start_ns.saturating_sub(resolution_ns),
                    "end_ns": end_ns.saturating_add(resolution_ns),
                }],
            },
        })
    }

    fn enroot_projection_row(
        attempt_index: usize,
        projection: &ProjectedDuration,
    ) -> Map<String, Value> {
        Map::from_iter([
            ("attempt_index".to_owned(), json!(attempt_index)),
            ("status".to_owned(), json!("completed")),
            (
                "duration_sec".to_owned(),
                number_value(projection.duration_sec),
            ),
            ("duration_valid".to_owned(), json!(projection.valid)),
            ("reason".to_owned(), json!(projection.reason)),
            (
                "window_start_ns".to_owned(),
                u64_value(projection.window_start_ns),
            ),
            (
                "window_end_ns".to_owned(),
                u64_value(projection.window_end_ns),
            ),
            (
                "cpu_usage_sec".to_owned(),
                number_value(projection.cpu_usage_sec),
            ),
            (
                "average_logical_cores".to_owned(),
                number_value(projection.average_logical_cores),
            ),
            (
                "cleanup_valid".to_owned(),
                projection
                    .cleanup_valid
                    .map(Value::Bool)
                    .unwrap_or(Value::Null),
            ),
            (
                "clock_reference".to_owned(),
                projection.clock_reference.clone().unwrap_or(Value::Null),
            ),
        ])
    }

    #[test]
    fn public_profile_arguments_bind_only_enroot_to_captured_absolute_executable() {
        let root = tempfile::tempdir().unwrap();
        let mut request = native_preflight_request(root.path());
        let docker_arguments = PublicProfileLauncher::profile_arguments(&request).unwrap();
        assert!(!docker_arguments.iter().any(|value| value == "--enroot"));
        request.runtime = "enroot".to_owned();
        request.enroot_executable = Some(PathBuf::from("/captured/bin/enroot"));
        request.profile_sets.clear();
        let arguments = PublicProfileLauncher::profile_arguments(&request)
            .unwrap()
            .iter()
            .map(|value| value.to_string_lossy().into_owned())
            .collect::<Vec<_>>();
        assert!(arguments
            .windows(2)
            .any(|values| values == ["--runtime", "enroot"]));
        assert!(arguments
            .windows(2)
            .any(|values| values == ["--enroot", "/captured/bin/enroot"]));
        assert!(arguments
            .iter()
            .any(|value| value == "--run-wave-start-gate"));

        request.enroot_executable = Some(PathBuf::from("relative/enroot"));
        assert!(PublicProfileLauncher::profile_arguments(&request)
            .unwrap_err()
            .to_string()
            .contains("not absolute"));
    }

    #[test]
    fn enroot_preflight_is_cached_per_task_across_launch_placements() {
        struct EnrootPreflightLauncher {
            calls: AtomicUsize,
            image_identity: Value,
        }

        impl AttemptLauncher for EnrootPreflightLauncher {
            fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
                self.calls.fetch_add(1, Ordering::SeqCst);
                Ok(json!({
                    "ready": true,
                    "runtime": request.runtime,
                    "source_sha256": file_sha256(&request.trajectory)?,
                    "image_identity": self.image_identity,
                    "profile_sets": request.profile_sets,
                    "cpuset_cpus": request.cpuset_cpus,
                    "controller_identity": {
                        "kind": "rust-host-mounted-binary",
                        "sha256": "c".repeat(64),
                    },
                    "checks": [{
                        "name": "runtime",
                        "status": "passed",
                        "runtime": "enroot",
                        "executable": request.enroot_executable,
                    }],
                }))
            }

            fn run_profile(
                &self,
                _request: &ProfileRequest,
                _cancellation: &CancellationToken,
            ) -> Result<Value, RunExecutionError> {
                panic!("preflight cache test must not dispatch a profile")
            }
        }

        let root = tempfile::tempdir().unwrap();
        let (task, spec, preflight) = enroot_fixture(root.path());
        let wave = spec.waves[0].clone();
        let attempts = wave
            .launches
            .iter()
            .enumerate()
            .map(|(offset, launch)| WaveAttempt {
                attempt_index: offset + 1,
                wave_index: 1,
                launch_index: offset + 1,
                wave: wave.clone(),
                launch: launch.clone(),
                task: task.clone(),
            })
            .collect::<Vec<_>>();
        let launcher = EnrootPreflightLauncher {
            calls: AtomicUsize::new(0),
            image_identity: preflight.image_identity,
        };
        let provenance_digest = "e".repeat(64);
        let enroot_executable = Path::new("/opt/enroot/bin/enroot");
        let observed = preflight_v3(
            &spec,
            &attempts,
            &launcher,
            &CancellationToken::new(),
            Some((&provenance_digest, enroot_executable)),
        )
        .unwrap();
        assert_eq!(launcher.calls.load(Ordering::SeqCst), 1);
        assert_eq!(observed.len(), 2);
        assert_eq!(
            observed[&("task".to_owned(), "4".to_owned())].image_id,
            preflight.image_id
        );
        assert_eq!(
            observed[&("task".to_owned(), "5".to_owned())].image_id,
            preflight.image_id
        );
        assert!(observed
            .values()
            .all(|value| { value.enroot_executable.as_deref() == Some(enroot_executable) }));

        let launch = &spec.waves[0].launches[0];
        let request = v3_request(
            &spec,
            &task,
            &launch.cpus,
            root.path().join("mismatched-preflight"),
            Some(enroot_executable),
        );
        let mut mismatched = launcher.preflight(&request).unwrap();
        mismatched["checks"][0]["executable"] = json!("/other/bin/enroot");
        let reasons = check_preflight(
            mismatched,
            &task,
            &launch.cpus,
            &request.profile_sets,
            &request.runtime,
            &request.controller_runtime,
            request.enroot_executable.as_deref(),
        )
        .unwrap_err();
        assert!(reasons
            .iter()
            .any(|reason| reason.contains("differs from captured runtime provenance")));
    }

    #[test]
    fn ordered_enroot_reattests_before_and_after_every_wave_and_binds_launches() {
        let root = tempfile::tempdir().unwrap();
        let (spec, attempts, preflights) = enroot_wave_execution_fixture(root.path());
        let launcher = RecordingEnrootFailureLauncher::default();
        let mut observations = Vec::new();
        let mut reattest = |wave_index, phase| {
            observations.push((wave_index, phase));
            true
        };
        let (rows, interrupted, windows, failure) = execute_v3(
            &spec,
            &attempts,
            &preflights,
            &launcher,
            &CancellationToken::new(),
            Some(&mut reattest),
        )
        .unwrap();
        drop(reattest);

        assert!(!interrupted);
        assert_eq!(failure, None);
        assert_eq!(rows.len(), 4);
        assert_eq!(windows.len(), 2);
        assert_eq!(
            observations,
            [
                (1, EnrootWaveAttestationPhase::Before),
                (1, EnrootWaveAttestationPhase::After),
                (2, EnrootWaveAttestationPhase::Before),
                (2, EnrootWaveAttestationPhase::After),
            ]
        );
        let requests = launcher.requests.lock().unwrap();
        assert_eq!(requests.len(), 4);
        assert!(requests.iter().all(|request| {
            request.runtime == "enroot"
                && request.enroot_executable.as_deref() == Some(Path::new("/opt/enroot/bin/enroot"))
        }));
    }

    #[test]
    fn ordered_enroot_pre_wave_attestation_failure_prevents_dispatch() {
        let root = tempfile::tempdir().unwrap();
        let (spec, attempts, preflights) = enroot_wave_execution_fixture(root.path());
        let launcher = RecordingEnrootFailureLauncher::default();
        let mut observations = Vec::new();
        let mut reattest = |wave_index, phase| {
            observations.push((wave_index, phase));
            false
        };
        let (rows, interrupted, windows, failure) = execute_v3(
            &spec,
            &attempts,
            &preflights,
            &launcher,
            &CancellationToken::new(),
            Some(&mut reattest),
        )
        .unwrap();
        drop(reattest);

        let reason = "Enroot runtime provenance re-attestation failed before wave 1";
        assert!(!interrupted);
        assert_eq!(failure.as_deref(), Some(reason));
        assert_eq!(observations, [(1, EnrootWaveAttestationPhase::Before)]);
        assert!(windows.is_empty());
        assert!(launcher.requests.lock().unwrap().is_empty());
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| {
            !row_bool(row, "duration_valid")
                && row.get("reason").and_then(Value::as_str) == Some(reason)
        }));
    }

    #[test]
    fn ordered_enroot_post_wave_attestation_failure_stops_later_waves() {
        let root = tempfile::tempdir().unwrap();
        let (spec, attempts, preflights) = enroot_wave_execution_fixture(root.path());
        let launcher = RecordingEnrootFailureLauncher::default();
        let mut observations = Vec::new();
        let mut reattest = |wave_index, phase| {
            observations.push((wave_index, phase));
            phase != EnrootWaveAttestationPhase::After
        };
        let (rows, interrupted, windows, failure) = execute_v3(
            &spec,
            &attempts,
            &preflights,
            &launcher,
            &CancellationToken::new(),
            Some(&mut reattest),
        )
        .unwrap();
        drop(reattest);

        let reason = "Enroot runtime provenance re-attestation failed after wave 1";
        assert!(!interrupted);
        assert_eq!(failure.as_deref(), Some(reason));
        assert_eq!(
            observations,
            [
                (1, EnrootWaveAttestationPhase::Before),
                (1, EnrootWaveAttestationPhase::After),
            ]
        );
        assert_eq!(windows.len(), 1);
        assert_eq!(launcher.requests.lock().unwrap().len(), 2);
        assert_eq!(rows.len(), 4);
        assert!(rows.iter().all(|row| {
            !row_bool(row, "duration_valid")
                && row
                    .get("reason")
                    .and_then(Value::as_str)
                    .is_some_and(|value| value.contains(reason))
        }));
    }

    #[test]
    fn enroot_projection_and_wave_summary_are_wall_only_and_require_full_overlap() {
        let root = tempfile::tempdir().unwrap();
        let (task, spec, preflight) = enroot_fixture(root.path());
        let first_report = enroot_report(
            &task,
            &spec,
            &spec.waves[0].launches[0],
            &preflight,
            1_000_000_000,
            3_000_000_000,
            "boot-a",
        );
        let second_report = enroot_report(
            &task,
            &spec,
            &spec.waves[0].launches[1],
            &preflight,
            1_500_000_000,
            2_500_000_000,
            "boot-a",
        );
        let first = project_enroot_duration(
            first_report.as_object().unwrap(),
            &task,
            &preflight,
            &spec,
            &spec.waves[0].launches[0],
            false,
        );
        let second = project_enroot_duration(
            second_report.as_object().unwrap(),
            &task,
            &preflight,
            &spec,
            &spec.waves[0].launches[1],
            false,
        );
        assert!(first.valid, "{}", first.reason);
        assert!(second.valid, "{}", second.reason);
        assert_eq!(first.cpu_usage_sec, None);
        assert_eq!(first.average_logical_cores, None);
        assert_eq!(first.cleanup_valid, Some(true));
        let missing_gate = project_enroot_duration(
            first_report.as_object().unwrap(),
            &task,
            &preflight,
            &spec,
            &spec.waves[0].launches[0],
            true,
        );
        assert!(!missing_gate.valid);
        assert!(missing_gate.reason.contains("wave-start gate evidence"));
        let mut gated_report = first_report.clone();
        let gate_clock = gated_report["container_cpu"]["clock_reference"].clone();
        gated_report["run_wave_start_gate"] = json!({
            "schema_version": "tasktrace-run-wave-start-gate.v1",
            "requested": true,
            "state": "released",
            "ready": true,
            "released": true,
            "valid": true,
            "released_ns": 999_999_900_u64,
            "measured_start_ns": 1_000_000_000_u64,
            "release_to_start_ns": 100_u64,
            "clock_reference": gate_clock,
        });
        let gated = project_enroot_duration(
            gated_report.as_object().unwrap(),
            &task,
            &preflight,
            &spec,
            &spec.waves[0].launches[0],
            true,
        );
        assert!(gated.valid, "{}", gated.reason);
        let mut changed_runtime = preflight.clone();
        changed_runtime.enroot_runtime_provenance_sha256 = Some("f".repeat(64));
        assert_ne!(
            enroot_execution_plan_digest(&task, &spec, &spec.waves[0].launches[0], &preflight,),
            enroot_execution_plan_digest(
                &task,
                &spec,
                &spec.waves[0].launches[0],
                &changed_runtime,
            )
        );
        let mut changed_executable = preflight.clone();
        changed_executable.enroot_executable = Some(PathBuf::from("/other/bin/enroot"));
        assert_ne!(
            enroot_execution_plan_digest(&task, &spec, &spec.waves[0].launches[0], &preflight,),
            enroot_execution_plan_digest(
                &task,
                &spec,
                &spec.waves[0].launches[0],
                &changed_executable,
            )
        );

        let rows = vec![
            enroot_projection_row(1, &first),
            enroot_projection_row(2, &second),
        ];
        let expected_indexes = BTreeSet::from([1_usize, 2_usize]);
        let outer = HostWindow {
            start_ns: 500_000_000,
            end_ns: 4_000_000_000,
            resolution_ns: 1,
        };
        let overall = enroot_wave_overall(&rows, false, Some(outer), 2, &expected_indexes).unwrap();
        assert_eq!(overall["valid_for_wall_only_throughput"], true);
        assert_eq!(overall["achieved_peak_measured_attempts"], 2);
        assert_eq!(overall["replay_command_batch_wall_sec"], 2.0);
        assert_eq!(overall["throughput_tasks_per_sec"], 1.0);
        assert_eq!(overall["lifecycle_inclusive_wall_sec"], 3.5);
        assert_eq!(overall["total_cgroup_cpu_sec"], Value::Null);
        assert_eq!(overall["aggregate_average_logical_cores"], Value::Null);

        let mut mismatched_clock = rows.clone();
        mismatched_clock[1].get_mut("clock_reference").unwrap()["boot_id"] = json!("boot-b");
        let invalid =
            enroot_wave_overall(&mismatched_clock, false, Some(outer), 2, &expected_indexes)
                .unwrap();
        assert_eq!(invalid["valid_for_throughput"], false);
        assert!(invalid["throughput_invalid_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason.as_str().unwrap().contains("clock identity")));

        let mut no_overlap = rows;
        no_overlap[1].insert("window_start_ns".to_owned(), json!(4_000_000_000_u64));
        no_overlap[1].insert("window_end_ns".to_owned(), json!(5_000_000_000_u64));
        let invalid =
            enroot_wave_overall(&no_overlap, false, Some(outer), 2, &expected_indexes).unwrap();
        assert_eq!(invalid["valid_for_throughput"], false);
        assert_eq!(invalid["achieved_peak_measured_attempts"], 1);
    }

    fn assert_native_controller_summary(summary: &Value) {
        assert_eq!(summary["controller_runtime"], "rust-host-mount");
        assert_eq!(summary["controller_runtime_content_bound"], true);
        assert!(summary["controller_runtime_limitation"].is_null());
        assert_eq!(
            summary["controller_binary_sha256"],
            Value::String("c".repeat(64))
        );
    }

    #[test]
    fn mock_v1_v2_v3_execute_and_publish_contract_artifacts() {
        let cpus = available_cpus();
        assert!(cpus.len() >= 2);
        let root = tempfile::tempdir().unwrap();
        let launcher = MockLauncher::default();

        let v1_root = root.path().join("v1");
        let v1 = load_run_spec(&write_v1_or_v2(&v1_root, cpus[1], 1)).unwrap();
        let v1_result = execute_run_with_launcher(&v1, &launcher).unwrap();
        assert_eq!(v1_result.exit_code, 0);
        assert_eq!(v1_result.rows.len(), 4);
        assert_eq!(v1_result.summary["duration_clock"], "host-monotonic");
        assert!(
            v1_result.summary["outer_window_end_ns"].as_u64().unwrap()
                > v1_result.summary["outer_window_start_ns"].as_u64().unwrap()
        );
        assert_native_controller_summary(&v1_result.summary);
        assert!(v1_root.join("results/results.csv").is_file());

        let v2_root = root.path().join("v2");
        let v2 = load_run_spec(&write_v1_or_v2(&v2_root, cpus[1], 2)).unwrap();
        let v2_result = execute_run_with_launcher(&v2, &launcher).unwrap();
        assert_eq!(v2_result.exit_code, 0, "{}", v2_result.summary);
        assert_eq!(v2_result.rows.len(), 4);
        assert_eq!(v2_result.summary["overall"]["valid_for_throughput"], true);
        assert_eq!(
            v2_result.summary["canonical_clock_reference"]["clock_id"],
            "CLOCK_MONOTONIC"
        );
        assert_native_controller_summary(&v2_result.summary);

        let v3_root = root.path().join("v3");
        let v3 = load_run_spec(&write_v3(&v3_root, cpus[0], cpus[1])).unwrap();
        let before: BTreeSet<_> = available_cpus().into_iter().collect();
        let v3_result = execute_run_with_launcher(&v3, &launcher).unwrap();
        let after: BTreeSet<_> = available_cpus().into_iter().collect();
        assert_eq!(before, after);
        assert_eq!(v3_result.exit_code, 0, "{}", v3_result.summary);
        assert_eq!(v3_result.summary["overall"]["valid_for_analysis"], true);
        assert_native_controller_summary(&v3_result.summary);
        assert!(v3_root.join("results/resolved-run.yaml").is_file());
        assert!(v3_root.join("results/summary.json").is_file());
        let resolved: YamlValue =
            serde_yaml::from_slice(&fs::read(v3_root.join("results/resolved-run.yaml")).unwrap())
                .unwrap();
        assert_eq!(
            resolved["defaults"]["controller_runtime"],
            YamlValue::String("rust-host-mount".to_owned())
        );
    }

    #[test]
    fn pre_cancelled_v2_stops_before_preflight_and_output_claim() {
        let cpus = available_cpus();
        let root = tempfile::tempdir().unwrap();
        let manifest = write_v1_or_v2(root.path(), cpus[0], 2);
        let spec = load_run_spec(&manifest).unwrap();
        let token = CancellationToken::new();
        token.cancel();
        let error =
            execute_run_with_launcher_and_cancellation(&spec, &MockLauncher::default(), &token)
                .unwrap_err();
        assert!(error.to_string().contains("preflight interrupted"));
        assert!(!root.path().join("results").exists());
    }

    #[test]
    fn v1_interruption_after_active_attempt_retains_remaining_rows() {
        let cpus = available_cpus();
        let root = tempfile::tempdir().unwrap();
        let manifest = write_v1_or_v2(root.path(), cpus[0], 1);
        let spec = load_run_spec(&manifest).unwrap();
        let token = CancellationToken::new();
        let result = execute_run_with_launcher_and_cancellation(
            &spec,
            &CancelAfterFirstLauncher::default(),
            &token,
        )
        .unwrap();
        assert_eq!(result.exit_code, 130);
        assert_eq!(result.rows.len(), 4);
        assert_eq!(status_count(&result.rows, "completed"), 1);
        assert_eq!(status_count(&result.rows, "failed"), 3);
        assert!(result.rows[1]["reason"]
            .as_str()
            .unwrap()
            .contains("before attempt dispatch"));
    }

    #[test]
    #[ignore = "requires Docker and a preloaded TASKTRACE_RUST_DOCKER_TEST_IMAGE"]
    fn forced_profile_timeout_removes_output_scoped_docker_and_policy_state() {
        use std::os::unix::fs::PermissionsExt;

        fn quote(value: &str) -> String {
            format!("'{}'", value.replace('\'', "'\\''"))
        }

        let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
            .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE");
        let root = tempfile::tempdir().unwrap();
        let output = root.path().join("profile");
        fs::create_dir(&output).unwrap();
        let token = format!("{:08x}{:08x}", std::process::id(), 17_u32);
        let network = format!("tasktrace-policy-{token}");
        let proxy = format!("tasktrace-policy-proxy-{token}");
        let workload = format!("tasktrace-profile-hard-{token}");
        let state = env::temp_dir().join(format!(".tasktrace-network-policy-state-{token}"));
        fs::create_dir(&state).unwrap();
        fs::write(
            output.join("network_enforcement.json"),
            serde_json::to_vec(&json!({
                "enforcement": {
                    "ephemeral_internal_network": network,
                    "proxy": {"container_name": proxy},
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let script = root.path().join("unresponsive-profile.sh");
        fs::write(
            &script,
            format!(
                "#!/bin/sh\nset -eu\ndocker network create {network}\ndocker run --detach --name {proxy} --network {network} {image} sh -c 'trap \"\" TERM INT; sleep 300'\ndocker run --detach --name {workload} --network {network} --mount {} {image} sh -c 'trap \"\" TERM INT; sleep 300'\ntrap '' TERM INT\nwhile :; do sleep 1; done\n",
                quote(&format!(
                    "type=bind,src={},dst=/tasktrace-profile-output",
                    output.display()
                )),
                image = quote(&image),
            ),
        )
        .unwrap();
        fs::set_permissions(&script, fs::Permissions::from_mode(0o700)).unwrap();
        let launcher = PublicProfileLauncher {
            executable: script,
            cancellation_grace: Duration::from_millis(200),
        };
        let cancellation = CancellationToken::new();
        let watcher_token = cancellation.clone();
        let watcher_name = workload.clone();
        let watcher = thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(20);
            while Instant::now() < deadline {
                if Command::new("docker")
                    .args(["container", "inspect", &watcher_name])
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .status()
                    .is_ok_and(|status| status.success())
                {
                    watcher_token.cancel();
                    return true;
                }
                thread::sleep(Duration::from_millis(20));
            }
            false
        });
        let request = ProfileRequest {
            task_id: "forced-cleanup".to_owned(),
            trajectory: root.path().join("unused.json"),
            image: image.clone(),
            output_dir: output.clone(),
            cpuset_cpus: "0".to_owned(),
            pacing: crate::pacing::UNPACED.to_owned(),
            timeout_sec: None,
            runtime: "docker".to_owned(),
            enroot_executable: None,
            controller_runtime: "rust-host-mount".to_owned(),
            profile_sets: vec!["cpu-average".to_owned()],
            network_policy: None,
            allowed_hosts: Vec::new(),
            workdir: None,
            environment: Vec::new(),
        };
        let error = launcher.run_profile(&request, &cancellation).unwrap_err();
        assert!(watcher.join().unwrap(), "workload container never started");
        assert!(
            error.to_string().contains("did not complete cleanup"),
            "{error}"
        );
        for arguments in [
            vec!["container", "inspect", workload.as_str()],
            vec!["container", "inspect", proxy.as_str()],
            vec!["network", "inspect", network.as_str()],
        ] {
            assert!(!Command::new("docker")
                .args(arguments)
                .output()
                .unwrap()
                .status
                .success());
        }
        assert!(!state.exists());
    }

    #[test]
    fn clock_domain_and_outer_containment_fail_closed() {
        let task = RunTaskSpec {
            task_id: "task".to_owned(),
            atif: PathBuf::from("task.json"),
            source_sha256: "a".repeat(64),
            image: "image".to_owned(),
            expected_image_id: None,
            action_timeout_sec: None,
        };
        let preflight = Preflight {
            image_id: "sha256:task".to_owned(),
            image_identity: json!({"id": "sha256:task"}),
            controller_sha256: None,
            enroot_runtime_provenance_sha256: None,
            enroot_executable: None,
        };
        let mut report = json!({
            "container_cpu": {
                "wall_sec": 1.0,
                "valid_for_cpu_performance": true,
                "window_start_ns": 10,
                "window_end_ns": 1_000_000_010_u64,
                "usage_sec": 0.0,
                "average_logical_cores": 0.0,
            },
            "replay_summary": {"actions_timed_out": 0, "source_outcome_mismatch_count": 0},
            "profiling_valid": true,
            "status": "valid",
            "source_trajectory_sha256": "a".repeat(64),
            "execution_plan_digest": "b".repeat(64),
            "image_identity": {"id": "sha256:task"},
        });
        let projection = project_duration(
            report.as_object().unwrap(),
            &task,
            &preflight,
            true,
            None,
            None,
            false,
        );
        assert!(!projection.valid);
        assert!(projection.reason.contains("canonical clock reference"));
        report["canonical_clock_reference"] = clock_reference();
        let mut row = Map::from_iter([
            ("duration_valid".to_owned(), Value::Bool(true)),
            ("reason".to_owned(), Value::String(String::new())),
            ("window_start_ns".to_owned(), json!(5_u64)),
            ("window_end_ns".to_owned(), json!(20_u64)),
        ]);
        invalidate_window_containment(
            std::slice::from_mut(&mut row),
            HostWindow {
                start_ns: 10,
                end_ns: 30,
                resolution_ns: 1,
            },
            None,
        );
        assert!(!row_bool(&row, "duration_valid"));
        assert!(row["reason"].as_str().unwrap().contains("outside"));
    }

    #[test]
    fn v2_projection_rejects_source_plan_image_and_effective_cpuset_drift() {
        let root = tempfile::tempdir().unwrap();
        let trajectory = root.path().join("task.json");
        fs::write(&trajectory, b"{}\n").unwrap();
        let task = RunTaskSpec {
            task_id: "task".to_owned(),
            atif: trajectory.clone(),
            source_sha256: file_sha256(&trajectory).unwrap(),
            image: "image".to_owned(),
            expected_image_id: Some("sha256:task".to_owned()),
            action_timeout_sec: None,
        };
        let request = ProfileRequest {
            task_id: task.task_id.clone(),
            trajectory,
            image: task.image.clone(),
            output_dir: root.path().join("profile"),
            cpuset_cpus: "2-3".to_owned(),
            pacing: crate::pacing::UNPACED.to_owned(),
            timeout_sec: None,
            runtime: "docker".to_owned(),
            enroot_executable: None,
            controller_runtime: "rust-host-mount".to_owned(),
            profile_sets: vec!["cpu-average".to_owned()],
            network_policy: None,
            allowed_hosts: Vec::new(),
            workdir: None,
            environment: Vec::new(),
        };
        let launcher = MockLauncher::default();
        let preflight = check_preflight(
            launcher.preflight(&request).unwrap(),
            &task,
            &request.cpuset_cpus,
            &request.profile_sets,
            &request.runtime,
            &request.controller_runtime,
            request.enroot_executable.as_deref(),
        )
        .unwrap();
        let report = launcher
            .run_profile(&request, &CancellationToken::new())
            .unwrap();
        let valid = project_duration(
            report.as_object().unwrap(),
            &task,
            &preflight,
            true,
            Some((&request.cpuset_cpus, 2)),
            None,
            false,
        );
        assert!(valid.valid, "{}", valid.reason);

        for (pointer, replacement, expected_reason) in [
            (
                "/source_trajectory_sha256",
                json!("f".repeat(64)),
                "source digest",
            ),
            ("/image_identity/id", json!("sha256:other"), "image ID"),
            (
                "/execution_plan/extensions/cpuset_cpus",
                json!("2"),
                "execution plan CPU set",
            ),
            (
                "/scope/identity/cpu_placement/cgroup_cpuset_cpus_effective",
                json!("2"),
                "effective cgroup cpuset",
            ),
            ("/controller_runtime", json!("image"), "controller runtime"),
            (
                "/controller_identity/kind",
                json!("legacy-controller"),
                "controller identity kind",
            ),
        ] {
            let mut drifted = report.clone();
            *drifted.pointer_mut(pointer).unwrap() = replacement;
            let projection = project_duration(
                drifted.as_object().unwrap(),
                &task,
                &preflight,
                true,
                Some((&request.cpuset_cpus, 2)),
                None,
                false,
            );
            assert!(!projection.valid, "drift at {pointer} remained valid");
            assert!(
                projection.reason.contains(expected_reason),
                "{}",
                projection.reason
            );
        }
        for (field, expected_reason) in [
            ("controller_runtime", "controller runtime"),
            ("controller_identity", "controller identity kind"),
        ] {
            let mut drifted = report.clone();
            drifted.as_object_mut().unwrap().remove(field);
            let projection = project_duration(
                drifted.as_object().unwrap(),
                &task,
                &preflight,
                true,
                Some((&request.cpuset_cpus, 2)),
                None,
                false,
            );
            assert!(!projection.valid, "missing {field} remained valid");
            assert!(
                projection.reason.contains(expected_reason),
                "{}",
                projection.reason
            );
        }
    }

    #[test]
    fn output_claim_race_fails_before_any_attempt_dispatch() {
        struct RacingLauncher {
            inner: MockLauncher,
        }

        impl AttemptLauncher for RacingLauncher {
            fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
                fs::create_dir_all(&request.output_dir).unwrap();
                self.inner.preflight(request)
            }

            fn run_profile(
                &self,
                _request: &ProfileRequest,
                _cancellation: &CancellationToken,
            ) -> Result<Value, RunExecutionError> {
                panic!("an output-claim race must fail before dispatch")
            }
        }

        let cpus = available_cpus();
        let root = tempfile::tempdir().unwrap();
        let manifest = write_v1_or_v2(root.path(), cpus[0], 1);
        let spec = load_run_spec(&manifest).unwrap();
        let error = execute_run_with_launcher(
            &spec,
            &RacingLauncher {
                inner: MockLauncher::default(),
            },
        )
        .unwrap_err();
        assert!(error.to_string().contains("claimed while preflight ran"));
        assert!(!root.path().join("results/results.csv").exists());
        assert!(!root.path().join("results/summary.json").exists());
    }

    #[test]
    fn controller_identity_drift_fails_before_output_claim_or_attempt_dispatch() {
        struct DriftingControllerLauncher {
            inner: MockLauncher,
        }

        impl AttemptLauncher for DriftingControllerLauncher {
            fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
                let mut report = self.inner.preflight(request)?;
                report["controller_identity"]["sha256"] =
                    Value::String(if request.task_id == "one" { "c" } else { "d" }.repeat(64));
                Ok(report)
            }

            fn run_profile(
                &self,
                _request: &ProfileRequest,
                _cancellation: &CancellationToken,
            ) -> Result<Value, RunExecutionError> {
                panic!("controller identity drift must fail before attempt dispatch")
            }
        }

        let cpus = available_cpus();
        let root = tempfile::tempdir().unwrap();
        let spec = load_run_spec(&write_v1_or_v2(root.path(), cpus[0], 1)).unwrap();
        let error = execute_run_with_launcher(
            &spec,
            &DriftingControllerLauncher {
                inner: MockLauncher::default(),
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("Rust controller content identity changed across preflight placements"),
            "{error}"
        );
        assert!(!root.path().join("results").exists());
    }

    #[test]
    fn invalid_controller_identity_kind_fails_before_output_claim_or_attempt_dispatch() {
        struct WrongKindLauncher {
            inner: MockLauncher,
        }

        impl AttemptLauncher for WrongKindLauncher {
            fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
                let mut report = self.inner.preflight(request)?;
                report["controller_identity"]["kind"] =
                    Value::String("legacy-controller".to_owned());
                Ok(report)
            }

            fn run_profile(
                &self,
                _request: &ProfileRequest,
                _cancellation: &CancellationToken,
            ) -> Result<Value, RunExecutionError> {
                panic!("invalid controller identity kind must fail before attempt dispatch")
            }
        }

        let cpus = available_cpus();
        let root = tempfile::tempdir().unwrap();
        let spec = load_run_spec(&write_v1_or_v2(root.path(), cpus[0], 1)).unwrap();
        let error = execute_run_with_launcher(
            &spec,
            &WrongKindLauncher {
                inner: MockLauncher::default(),
            },
        )
        .unwrap_err();
        assert!(
            error
                .to_string()
                .contains("preflight Rust controller identity kind is missing or invalid"),
            "{error}"
        );
        assert!(!root.path().join("results").exists());
    }

    #[test]
    fn v2_profile_failure_is_retained_while_other_attempts_continue() {
        struct FailOneLauncher {
            inner: MockLauncher,
        }

        impl AttemptLauncher for FailOneLauncher {
            fn preflight(&self, request: &ProfileRequest) -> Result<Value, RunExecutionError> {
                self.inner.preflight(request)
            }

            fn run_profile(
                &self,
                request: &ProfileRequest,
                cancellation: &CancellationToken,
            ) -> Result<Value, RunExecutionError> {
                if request.task_id == "one" {
                    Err(RunExecutionError::new("injected profile failure"))
                } else {
                    self.inner.run_profile(request, cancellation)
                }
            }
        }

        let cpus = available_cpus();
        let root = tempfile::tempdir().unwrap();
        let spec = load_run_spec(&write_v1_or_v2(root.path(), cpus[0], 2)).unwrap();
        let result = execute_run_with_launcher(
            &spec,
            &FailOneLauncher {
                inner: MockLauncher::default(),
            },
        )
        .unwrap();
        assert_eq!(result.exit_code, 1);
        assert_eq!(result.rows.len(), 4);
        assert_eq!(status_count(&result.rows, "failed"), 2);
        assert_eq!(status_count(&result.rows, "completed"), 2);
        assert_eq!(result.summary["overall"]["valid_for_throughput"], false);
        assert!(result
            .rows
            .iter()
            .filter(|row| { row.get("task_id").and_then(Value::as_str) == Some("one") })
            .all(|row| row["reason"]
                .as_str()
                .unwrap()
                .contains("injected profile failure")));
    }
    #[test]
    fn replay_projection_rejects_unavailable_failed_and_missing_tools() {
        let valid = json!({
            "tool_actions_recorded": 2, "tool_actions_replayed": 2,
            "source_outcome_comparison_unavailable_count": 0,
            "actions_launch_failed": 0, "actions_source_argument_parse_failed": 0,
            "actions_skipped_after_timeout": 0,
        });
        let mut reasons = Vec::new();
        push_incomplete_replay_reason(valid.as_object(), &mut reasons);
        assert!(reasons.is_empty());
        for field in [
            "source_outcome_comparison_unavailable_count",
            "actions_launch_failed",
            "actions_source_argument_parse_failed",
            "actions_skipped_after_timeout",
        ] {
            for value in [json!(1), Value::Null, json!(true), json!("0"), json!(-1)] {
                let mut invalid = valid.clone();
                invalid[field] = value;
                let mut reasons = Vec::new();
                push_incomplete_replay_reason(invalid.as_object(), &mut reasons);
                assert!(!reasons.is_empty(), "{invalid}");
            }
        }
        for (recorded, replayed) in [
            (json!(0), Value::Null),
            (json!(2), json!(1)),
            (Value::Null, Value::Null),
        ] {
            let invalid =
                json!({"tool_actions_recorded": recorded, "tool_actions_replayed": replayed});
            let mut reasons = Vec::new();
            push_incomplete_replay_reason(invalid.as_object(), &mut reasons);
            assert!(!reasons.is_empty());
        }
    }
}
