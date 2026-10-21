// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict decoding contracts for cgroup-scoped `perf stat` hardware counters.
//!
//! Counts that are missing, unsupported, multiplexed, malformed, duplicated,
//! or attributed to another cgroup remain explicit invalid measurements.  In
//! particular, this module never turns an unavailable PMU event into zero.

use crate::timebase::monotonic_ns;
use serde::Serialize;
use serde_json::{json, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, RawFd};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::Duration;

pub(crate) const HARDWARE_PERF_SCHEMA_VERSION: &str = "cgroup-perf-hardware.v1";
pub(crate) const MINIMUM_RUNNING_RATIO: f64 = 0.95;
const PERF_EVENT_SOURCE_ROOT: &str = "/sys/bus/event_source/devices";
const CONTROL_POLL_INTERVAL: Duration = Duration::from_millis(100);
const STOP_TIMEOUT: Duration = Duration::from_secs(10);
const CLEANUP_TERM_TIMEOUT: Duration = Duration::from_secs(5);
const CLEANUP_KILL_TIMEOUT: Duration = Duration::from_secs(5);
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);

pub(crate) const PORTABLE_HARDWARE_EVENTS: &[&str] = &[
    "cycles",
    "instructions",
    "branches",
    "branch-misses",
    "cache-references",
    "cache-misses",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HardwarePerfError {
    message: String,
    cleanup_failed: bool,
}

impl HardwarePerfError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            cleanup_failed: false,
        }
    }

    fn with_cleanup_failure(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            cleanup_failed: true,
        }
    }
}

impl fmt::Display for HardwarePerfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl Error for HardwarePerfError {}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct PerfCounterResult {
    pub(crate) event: String,
    pub(crate) namespace: String,
    pub(crate) status: String,
    pub(crate) value: Option<Value>,
    pub(crate) unit: Option<String>,
    pub(crate) time_enabled_ns: Option<u64>,
    pub(crate) time_running_ns: Option<u64>,
    pub(crate) time_running_fraction: Option<f64>,
    pub(crate) valid: bool,
    pub(crate) invalid_reason: Option<String>,
    pub(crate) observed_cgroup: Option<String>,
}

fn event_namespace(event: &str) -> &'static str {
    if PORTABLE_HARDWARE_EVENTS.contains(&event) {
        "portable"
    } else {
        "custom-unclassified"
    }
}

fn invalid_counter(
    event: &str,
    status: &str,
    reason: impl Into<String>,
    observed_cgroup: Option<String>,
) -> PerfCounterResult {
    PerfCounterResult {
        event: event.to_owned(),
        namespace: event_namespace(event).to_owned(),
        status: status.to_owned(),
        value: None,
        unit: None,
        time_enabled_ns: None,
        time_running_ns: None,
        time_running_fraction: None,
        valid: false,
        invalid_reason: Some(reason.into()),
        observed_cgroup,
    }
}

fn normalized_cgroup(value: &str) -> Result<String, HardwarePerfError> {
    let value = value.trim();
    if value.is_empty() {
        return Err(HardwarePerfError::new("empty cgroup"));
    }
    if value == "/" {
        Ok("/".to_owned())
    } else {
        Ok(format!("/{}", value.trim_matches('/')))
    }
}

fn csv_rows(text: &str) -> Result<Vec<Vec<String>>, HardwarePerfError> {
    let mut rows = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        let mut row = Vec::new();
        let mut field = String::new();
        let mut quoted = false;
        let mut characters = line.chars().peekable();
        while let Some(character) = characters.next() {
            match character {
                '"' if quoted && characters.peek() == Some(&'"') => {
                    characters.next();
                    field.push('"');
                }
                '"' => quoted = !quoted,
                ';' if !quoted => {
                    row.push(std::mem::take(&mut field));
                }
                other => field.push(other),
            }
        }
        if quoted {
            return Err(HardwarePerfError::new(format!(
                "perf stat row {} has an unterminated quoted field",
                line_index + 1
            )));
        }
        row.push(field);
        rows.push(row);
    }
    Ok(rows)
}

fn perf_number(raw: &str) -> Result<Value, HardwarePerfError> {
    if let Ok(value) = raw.parse::<u64>() {
        return Ok(Value::Number(Number::from(value)));
    }
    let value = raw
        .parse::<f64>()
        .map_err(|_| HardwarePerfError::new(format!("invalid perf count: {raw:?}")))?;
    if !value.is_finite() || value < 0.0 {
        return Err(HardwarePerfError::new(format!(
            "invalid perf count: {raw:?}"
        )));
    }
    if value.fract() == 0.0 && value <= u64::MAX as f64 {
        return Ok(Value::Number(Number::from(value as u64)));
    }
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| HardwarePerfError::new(format!("invalid perf count: {raw:?}")))
}

fn rounded(value: f64, digits: i32) -> f64 {
    let factor = 10_f64.powi(digits);
    (value * factor).round() / factor
}

pub(crate) fn parse_perf_stat_csv(
    text: &str,
    requested_events: &[String],
    minimum_running_ratio: f64,
    expected_cgroup: Option<&str>,
) -> Result<(Vec<PerfCounterResult>, Vec<String>), HardwarePerfError> {
    if !minimum_running_ratio.is_finite()
        || minimum_running_ratio <= 0.0
        || minimum_running_ratio > 1.0
    {
        return Err(HardwarePerfError::new(
            "minimum running ratio must be in (0, 1]",
        ));
    }
    if requested_events.is_empty()
        || requested_events.iter().any(|event| event.trim().is_empty())
        || requested_events.iter().collect::<BTreeSet<_>>().len() != requested_events.len()
    {
        return Err(HardwarePerfError::new(
            "requested perf events must be nonempty and unique",
        ));
    }
    let expected = expected_cgroup
        .map(normalized_cgroup)
        .transpose()
        .map_err(|_| HardwarePerfError::new("expected perf cgroup must be a nonempty path"))?;
    let requested = requested_events
        .iter()
        .map(String::as_str)
        .collect::<BTreeSet<_>>();
    let mut parsed = BTreeMap::<String, PerfCounterResult>::new();
    let mut anomalies = Vec::new();

    for (line_index, row) in csv_rows(text)?.into_iter().enumerate() {
        if row.iter().all(|field| field.trim().is_empty())
            || row
                .first()
                .is_some_and(|field| field.trim_start().starts_with('#'))
        {
            continue;
        }
        let required = if expected.is_some() { 6 } else { 5 };
        if row.len() < required {
            anomalies.push(format!(
                "perf stat row {} has fewer than {} fields",
                line_index + 1,
                required
            ));
            continue;
        }
        let raw_value = row[0].trim();
        let raw_unit = row[1].trim();
        let raw_event = row[2].trim();
        // Preserve an explicitly requested custom event containing `:`.  For
        // the generic aliases, perf may append privilege modifiers such as
        // `:u`; only strip those when the complete spelling was not requested.
        let event = if requested.contains(raw_event) {
            raw_event
        } else {
            raw_event.split(':').next().unwrap_or_default()
        };
        if !requested.contains(event) {
            continue;
        }
        let (observed_cgroup, raw_enabled, raw_percent, raw_cgroup) = if expected.is_some() {
            let raw_cgroup = row[3].trim();
            (
                normalized_cgroup(raw_cgroup)
                    .ok()
                    .or_else(|| (!raw_cgroup.is_empty()).then(|| raw_cgroup.to_owned())),
                row[4].trim(),
                row[5].trim(),
                Some(raw_cgroup),
            )
        } else {
            (None, row[3].trim(), row[4].trim(), None)
        };

        if let Some(existing) = parsed.get_mut(event) {
            let reason = format!("perf stat emitted duplicate event {event:?}");
            anomalies.push(reason.clone());
            existing.status = "duplicate".to_owned();
            existing.valid = false;
            existing.invalid_reason = Some(reason);
            continue;
        }
        if let Some(expected) = &expected {
            if observed_cgroup.as_ref() != Some(expected) {
                let reason = format!(
                    "perf attributed {} to cgroup {:?}, expected {:?}",
                    event,
                    raw_cgroup.unwrap_or_default(),
                    expected_cgroup.unwrap_or_default()
                );
                anomalies.push(reason.clone());
                parsed.insert(
                    event.to_owned(),
                    invalid_counter(event, "scope_mismatch", reason, observed_cgroup),
                );
                continue;
            }
        }
        if matches!(raw_value, "<not supported>" | "<not counted>") {
            let status = if raw_value == "<not supported>" {
                "unsupported"
            } else {
                "not_counted"
            };
            parsed.insert(
                event.to_owned(),
                invalid_counter(
                    event,
                    status,
                    format!("perf reported {raw_value} for {event}"),
                    observed_cgroup,
                ),
            );
            continue;
        }
        let value = perf_number(raw_value);
        let enabled_ns = raw_enabled.parse::<u64>();
        let running_fraction = raw_percent.parse::<f64>().map(|value| value / 100.0);
        let (Ok(value), Ok(enabled_ns), Ok(running_fraction)) =
            (value, enabled_ns, running_fraction)
        else {
            anomalies.push(format!("perf stat emitted malformed values for {event:?}"));
            parsed.insert(
                event.to_owned(),
                invalid_counter(
                    event,
                    "malformed",
                    "perf emitted a malformed count or runtime field",
                    observed_cgroup,
                ),
            );
            continue;
        };
        if enabled_ns == 0
            || !running_fraction.is_finite()
            || !(0.0..=1.0).contains(&running_fraction)
        {
            anomalies.push(format!(
                "perf stat emitted invalid runtime metadata for {event:?}"
            ));
            parsed.insert(
                event.to_owned(),
                invalid_counter(
                    event,
                    "malformed",
                    "perf emitted invalid enabled/running time metadata",
                    observed_cgroup,
                ),
            );
            continue;
        }
        let valid = running_fraction >= minimum_running_ratio;
        let invalid_reason = (!valid).then(|| {
            format!(
                "time_running/time_enabled {:.6} is below the required {:.6}",
                running_fraction, minimum_running_ratio
            )
        });
        parsed.insert(
            event.to_owned(),
            PerfCounterResult {
                event: event.to_owned(),
                namespace: event_namespace(event).to_owned(),
                status: if valid { "valid" } else { "multiplexed" }.to_owned(),
                value: Some(value),
                unit: (!raw_unit.is_empty()).then(|| raw_unit.to_owned()),
                time_enabled_ns: Some(enabled_ns),
                time_running_ns: Some((enabled_ns as f64 * running_fraction).round() as u64),
                time_running_fraction: Some(rounded(running_fraction, 9)),
                valid,
                invalid_reason,
                observed_cgroup,
            },
        );
    }

    let results = requested_events
        .iter()
        .map(|event| {
            parsed.remove(event).unwrap_or_else(|| {
                invalid_counter(
                    event,
                    "missing",
                    "perf stat output omitted the requested event",
                    None,
                )
            })
        })
        .collect();
    Ok((results, anomalies))
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct PerfControlBounds {
    pub(crate) action: String,
    pub(crate) request_monotonic_ns: u64,
    pub(crate) acknowledgement_monotonic_ns: u64,
    pub(crate) uncertainty_ns: u64,
    pub(crate) clock: &'static str,
}

impl PerfControlBounds {
    pub(crate) fn new(
        action: &str,
        request_monotonic_ns: u64,
        acknowledgement_monotonic_ns: u64,
    ) -> Result<Self, HardwarePerfError> {
        if !matches!(action, "enable" | "disable") {
            return Err(HardwarePerfError::new(format!(
                "unsupported perf control action: {action:?}"
            )));
        }
        let uncertainty_ns = acknowledgement_monotonic_ns
            .checked_sub(request_monotonic_ns)
            .ok_or_else(|| HardwarePerfError::new("perf control bounds are not monotonic"))?;
        Ok(Self {
            action: action.to_owned(),
            request_monotonic_ns,
            acknowledgement_monotonic_ns,
            uncertainty_ns,
            clock: "CLOCK_MONOTONIC",
        })
    }
}

fn io_error(context: &str, error: io::Error) -> HardwarePerfError {
    HardwarePerfError::new(format!("{context}: {error}"))
}

fn pipe() -> Result<(File, File), HardwarePerfError> {
    let mut descriptors = [-1_i32; 2];
    // CLOEXEC keeps both ends private by default.  The two ends needed by
    // perf have CLOEXEC cleared only across Command::spawn below.
    let result = unsafe { libc::pipe2(descriptors.as_mut_ptr(), libc::O_CLOEXEC) };
    if result != 0 {
        return Err(io_error(
            "unable to create perf control pipe",
            io::Error::last_os_error(),
        ));
    }
    // SAFETY: pipe2 returned two new, independently owned file descriptors.
    Ok(unsafe {
        (
            File::from_raw_fd(descriptors[0]),
            File::from_raw_fd(descriptors[1]),
        )
    })
}

fn set_close_on_exec(descriptor: RawFd, enabled: bool) -> Result<(), HardwarePerfError> {
    let flags = unsafe { libc::fcntl(descriptor, libc::F_GETFD) };
    if flags < 0 {
        return Err(io_error(
            "unable to inspect perf control descriptor",
            io::Error::last_os_error(),
        ));
    }
    let updated = if enabled {
        flags | libc::FD_CLOEXEC
    } else {
        flags & !libc::FD_CLOEXEC
    };
    if unsafe { libc::fcntl(descriptor, libc::F_SETFD, updated) } < 0 {
        return Err(io_error(
            "unable to configure perf control descriptor",
            io::Error::last_os_error(),
        ));
    }
    Ok(())
}

trait HardwarePerfChildOperations {
    type Status;

    fn poll(&mut self) -> Result<Option<Self::Status>, HardwarePerfError>;
    fn signal(&mut self, signal: libc::c_int) -> Result<(), HardwarePerfError>;
    fn wait_bounded(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Self::Status>, HardwarePerfError>;
}

struct RealHardwarePerfChild<'a> {
    child: &'a mut Child,
    process_group_id: libc::pid_t,
}

impl HardwarePerfChildOperations for RealHardwarePerfChild<'_> {
    type Status = std::process::ExitStatus;

    fn poll(&mut self) -> Result<Option<Self::Status>, HardwarePerfError> {
        self.child
            .try_wait()
            .map_err(|error| io_error("unable to wait for perf hardware profiler", error))
    }

    fn signal(&mut self, signal: libc::c_int) -> Result<(), HardwarePerfError> {
        signal_hardware_perf_process_group(self.process_group_id, signal)
    }

    fn wait_bounded(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Self::Status>, HardwarePerfError> {
        let deadline = monotonic_ns().saturating_add(timeout.as_nanos() as u64);
        loop {
            if let Some(status) = self.poll()? {
                return Ok(Some(status));
            }
            let now = monotonic_ns();
            if now >= deadline {
                return Ok(None);
            }
            let remaining = Duration::from_nanos(deadline - now);
            thread::sleep(CHILD_POLL_INTERVAL.min(remaining));
        }
    }
}

#[derive(Debug)]
struct HardwarePerfChildShutdown<Status> {
    status: Option<Status>,
    errors: Vec<String>,
}

fn shutdown_hardware_perf_child<Operations: HardwarePerfChildOperations>(
    operations: &mut Operations,
    graceful_signal: libc::c_int,
    graceful_timeout: Duration,
    kill_timeout: Duration,
) -> HardwarePerfChildShutdown<Operations::Status> {
    let mut errors = Vec::new();
    let mut status = match operations.poll() {
        Ok(status) => status,
        Err(error) => {
            errors.push(error.to_string());
            None
        }
    };
    if status.is_none() {
        if let Err(error) = operations.signal(graceful_signal) {
            errors.push(error.to_string());
        }
        match operations.wait_bounded(graceful_timeout) {
            Ok(Some(observed)) => status = Some(observed),
            Ok(None) => errors.push(format!(
                "timed out after {:.3}s waiting for perf hardware profiler to stop after signal {graceful_signal}",
                graceful_timeout.as_secs_f64()
            )),
            Err(error) => errors.push(error.to_string()),
        }
    }
    if status.is_none() {
        if let Err(error) = operations.signal(libc::SIGKILL) {
            errors.push(error.to_string());
        }
        match operations.wait_bounded(kill_timeout) {
            Ok(Some(observed)) => status = Some(observed),
            Ok(None) => errors.push(format!(
                "perf hardware profiler remains unreaped after SIGKILL and a {:.3}s bounded wait",
                kill_timeout.as_secs_f64()
            )),
            Err(error) => errors.push(error.to_string()),
        }
    }
    if status.is_none()
        && !errors
            .iter()
            .any(|error| error.contains("remains unreaped after SIGKILL"))
    {
        errors
            .push("perf hardware profiler absence could not be verified after cleanup".to_owned());
    }
    HardwarePerfChildShutdown { status, errors }
}

fn combine_cleanup_error(
    primary: HardwarePerfError,
    cleanup: Result<(), HardwarePerfError>,
) -> HardwarePerfError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup) => HardwarePerfError::with_cleanup_failure(format!(
            "{primary}; perf hardware profiler cleanup also failed: {cleanup}"
        )),
    }
}

#[derive(Debug)]
struct HardwarePerfProcessGroupShutdown {
    verified_absent: bool,
    errors: Vec<String>,
}

fn validate_hardware_perf_process_group(
    process_group_id: libc::pid_t,
) -> Result<(), HardwarePerfError> {
    if process_group_id <= 0 {
        return Err(HardwarePerfError::new(
            "perf hardware process-group ID must be positive",
        ));
    }
    let controller_process_group = unsafe { libc::getpgrp() };
    if process_group_id == controller_process_group {
        return Err(HardwarePerfError::new(format!(
            "refusing to signal controller process group {process_group_id} while cleaning perf hardware profiler"
        )));
    }
    Ok(())
}

fn signal_hardware_perf_process_group(
    process_group_id: libc::pid_t,
    signal: libc::c_int,
) -> Result<(), HardwarePerfError> {
    validate_hardware_perf_process_group(process_group_id)?;
    if unsafe { libc::kill(-process_group_id, signal) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(HardwarePerfError::new(format!(
            "unable to signal perf hardware process group {process_group_id} with signal {signal}: {error}"
        )))
    }
}

fn hardware_perf_process_group_exists(
    process_group_id: libc::pid_t,
) -> Result<bool, HardwarePerfError> {
    validate_hardware_perf_process_group(process_group_id)?;
    if unsafe { libc::kill(-process_group_id, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(HardwarePerfError::new(format!(
            "unable to inspect perf hardware process group {process_group_id}: {error}"
        ))),
    }
}

fn wait_for_hardware_perf_process_group_absence<Exists>(
    process_group_id: libc::pid_t,
    timeout: Duration,
    process_group_exists: &mut Exists,
) -> Result<bool, HardwarePerfError>
where
    Exists: FnMut(libc::pid_t) -> Result<bool, HardwarePerfError>,
{
    let deadline = monotonic_ns().saturating_add(timeout.as_nanos() as u64);
    loop {
        if !process_group_exists(process_group_id)? {
            return Ok(true);
        }
        let now = monotonic_ns();
        if now >= deadline {
            return Ok(false);
        }
        thread::sleep(CHILD_POLL_INTERVAL.min(Duration::from_nanos(deadline - now)));
    }
}

fn cleanup_hardware_perf_process_group_with<Exists, Signal>(
    process_group_id: libc::pid_t,
    term_timeout: Duration,
    kill_timeout: Duration,
    mut process_group_exists: Exists,
    mut signal_process_group: Signal,
) -> HardwarePerfProcessGroupShutdown
where
    Exists: FnMut(libc::pid_t) -> Result<bool, HardwarePerfError>,
    Signal: FnMut(libc::pid_t, libc::c_int) -> Result<(), HardwarePerfError>,
{
    let mut errors = Vec::new();
    match process_group_exists(process_group_id) {
        Ok(false) => {
            return HardwarePerfProcessGroupShutdown {
                verified_absent: true,
                errors,
            }
        }
        Ok(true) => {}
        Err(error) => errors.push(error.to_string()),
    }

    if let Err(error) = signal_process_group(process_group_id, libc::SIGTERM) {
        errors.push(error.to_string());
    }
    match wait_for_hardware_perf_process_group_absence(
        process_group_id,
        term_timeout,
        &mut process_group_exists,
    ) {
        Ok(true) => {
            return HardwarePerfProcessGroupShutdown {
                verified_absent: true,
                errors,
            }
        }
        Ok(false) => errors.push(format!(
            "timed out after {:.3}s waiting for perf hardware process group {process_group_id} to exit after SIGTERM",
            term_timeout.as_secs_f64()
        )),
        Err(error) => errors.push(error.to_string()),
    }

    if let Err(error) = signal_process_group(process_group_id, libc::SIGKILL) {
        errors.push(error.to_string());
    }
    let verified_absent = match wait_for_hardware_perf_process_group_absence(
        process_group_id,
        kill_timeout,
        &mut process_group_exists,
    ) {
        Ok(absent) => absent,
        Err(error) => {
            errors.push(error.to_string());
            false
        }
    };
    if !verified_absent {
        errors.push(format!(
            "perf hardware process group {process_group_id} remains observable after SIGKILL and a {:.3}s bounded wait",
            kill_timeout.as_secs_f64()
        ));
    }
    HardwarePerfProcessGroupShutdown {
        verified_absent,
        errors,
    }
}

fn cleanup_hardware_perf_process_group(
    process_group_id: libc::pid_t,
    term_timeout: Duration,
    kill_timeout: Duration,
) -> HardwarePerfProcessGroupShutdown {
    cleanup_hardware_perf_process_group_with(
        process_group_id,
        term_timeout,
        kill_timeout,
        hardware_perf_process_group_exists,
        signal_hardware_perf_process_group,
    )
}

pub(crate) fn collect_hardware_perf_provenance(perf: &str) -> Value {
    let mut errors = Vec::new();
    let perf_version = match Command::new(perf).arg("--version").output() {
        Ok(output) => {
            let selected = if output.stdout.is_empty() {
                &output.stderr
            } else {
                &output.stdout
            };
            let text = String::from_utf8_lossy(selected);
            let first = text
                .lines()
                .next()
                .map(str::trim)
                .filter(|line| !line.is_empty());
            if output.status.success() {
                first.map(|line| line.chars().take(512).collect::<String>())
            } else {
                errors.push(format!(
                    "perf --version exited {}: {}",
                    output.status,
                    first.unwrap_or("no diagnostic")
                ));
                None
            }
        }
        Err(error) => {
            errors.push(format!("unable to query perf version: {error}"));
            None
        }
    };
    let event_source_names = match fs::read_dir(PERF_EVENT_SOURCE_ROOT) {
        Ok(entries) => {
            let mut names = Vec::new();
            for entry in entries {
                match entry {
                    Ok(entry) if entry.path().is_dir() => {
                        names.push(entry.file_name().to_string_lossy().into_owned());
                    }
                    Ok(_) => {}
                    Err(error) => {
                        errors.push(format!("unable to inspect a perf event source: {error}"))
                    }
                }
            }
            names.sort();
            names
        }
        Err(error) => {
            errors.push(format!("unable to enumerate perf event sources: {error}"));
            Vec::new()
        }
    };
    let uname = fs::read_to_string("/proc/sys/kernel/osrelease")
        .ok()
        .map(|value| value.trim().to_owned())
        .filter(|value| !value.is_empty());
    json!({
        "perf_binary": perf,
        "perf_version": perf_version,
        "kernel_release": uname,
        "machine": std::env::consts::ARCH,
        "event_source_root": PERF_EVENT_SOURCE_ROOT,
        "event_source_names": event_source_names,
        "provenance_valid": errors.is_empty(),
        "provenance_errors": errors,
    })
}

fn perf_cgroup(cgroup_path: &Path) -> Result<String, HardwarePerfError> {
    let relative = cgroup_path.strip_prefix("/sys/fs/cgroup").map_err(|_| {
        HardwarePerfError::new(format!(
            "perf cgroup is outside /sys/fs/cgroup: {}",
            cgroup_path.display()
        ))
    })?;
    let relative = relative.to_string_lossy();
    normalized_cgroup(&relative)
}

fn perf_arguments(
    stats_path: &Path,
    events: &[String],
    perf_cgroup: &str,
    cpuset_cpus: Option<&str>,
    control_read_fd: RawFd,
    acknowledgement_write_fd: RawFd,
) -> Vec<String> {
    let mut arguments = vec![
        "stat".to_owned(),
        "-D".to_owned(),
        "-1".to_owned(),
        "--control".to_owned(),
        format!("fd:{control_read_fd},{acknowledgement_write_fd}"),
        "-x".to_owned(),
        ";".to_owned(),
        "--no-big-num".to_owned(),
        "--hybrid-merge".to_owned(),
        "-o".to_owned(),
        stats_path.display().to_string(),
    ];
    for event in events {
        arguments.push("-e".to_owned());
        arguments.push(event.clone());
    }
    arguments.extend(["-G".to_owned(), perf_cgroup.to_owned(), "-a".to_owned()]);
    if let Some(cpuset_cpus) = cpuset_cpus {
        arguments.extend(["-C".to_owned(), cpuset_cpus.to_owned()]);
    }
    arguments
}

/// One disabled, cgroup-scoped perf-stat process owned by the Docker
/// coordinator.  The workload is launched elsewhere; this collector only
/// attaches to its already-created cgroup and observes that scope.
pub(crate) struct PerfCgroupHardwareProfiler {
    output_dir: PathBuf,
    cpuset_cpus: Option<String>,
    perf: String,
    events: Vec<String>,
    attach_timeout: Duration,
    minimum_running_ratio: f64,
    stats_path: PathBuf,
    stderr_path: PathBuf,
    process: Option<Child>,
    process_group_id: Option<libc::pid_t>,
    control_write: Option<File>,
    acknowledgement_read: Option<File>,
    acknowledgement_buffer: Vec<u8>,
    perf_cgroup: Option<String>,
    attach_ready_bounds: Option<PerfControlBounds>,
    enable_bounds: Option<PerfControlBounds>,
    disable_bounds: Option<PerfControlBounds>,
    enabled: bool,
    provenance: Value,
}

impl PerfCgroupHardwareProfiler {
    pub(crate) fn new(
        output_dir: impl Into<PathBuf>,
        cpuset_cpus: Option<String>,
        perf: impl Into<String>,
        events: Vec<String>,
        attach_timeout: Duration,
        minimum_running_ratio: f64,
    ) -> Result<Self, HardwarePerfError> {
        let output_dir = output_dir.into();
        if attach_timeout.is_zero() {
            return Err(HardwarePerfError::new(
                "perf attach timeout must be positive",
            ));
        }
        if events.is_empty()
            || events.iter().any(|event| event.trim().is_empty())
            || events.iter().collect::<BTreeSet<_>>().len() != events.len()
        {
            return Err(HardwarePerfError::new(
                "perf event list must be nonempty and unique",
            ));
        }
        if !minimum_running_ratio.is_finite()
            || minimum_running_ratio <= 0.0
            || minimum_running_ratio > 1.0
        {
            return Err(HardwarePerfError::new(
                "minimum running ratio must be in (0, 1]",
            ));
        }
        Ok(Self {
            stats_path: output_dir.join("hardware_perf_stat.csv"),
            stderr_path: output_dir.join("hardware_perf.stderr.txt"),
            output_dir,
            cpuset_cpus,
            perf: perf.into(),
            events,
            attach_timeout,
            minimum_running_ratio,
            process: None,
            process_group_id: None,
            control_write: None,
            acknowledgement_read: None,
            acknowledgement_buffer: Vec::new(),
            perf_cgroup: None,
            attach_ready_bounds: None,
            enable_bounds: None,
            disable_bounds: None,
            enabled: false,
            provenance: Value::Null,
        })
    }

    pub(crate) fn stats_path(&self) -> &Path {
        &self.stats_path
    }

    pub(crate) fn stderr_path(&self) -> &Path {
        &self.stderr_path
    }

    pub(crate) fn provenance(&self) -> &Value {
        &self.provenance
    }

    pub(crate) fn stderr_text(&self) -> String {
        fs::read_to_string(&self.stderr_path)
            .unwrap_or_default()
            .trim()
            .to_owned()
    }

    pub(crate) fn is_running(&mut self) -> Result<bool, HardwarePerfError> {
        let Some(process) = self.process.as_mut() else {
            return Ok(false);
        };
        process
            .try_wait()
            .map(|status| status.is_none())
            .map_err(|error| io_error("unable to poll perf hardware profiler", error))
    }

    fn control(&mut self, action: &str) -> Result<PerfControlBounds, HardwarePerfError> {
        if !matches!(action, "enable" | "disable") {
            return Err(HardwarePerfError::new(format!(
                "unsupported perf control action: {action:?}"
            )));
        }
        if self.process.is_none()
            || self.control_write.is_none()
            || self.acknowledgement_read.is_none()
        {
            return Err(HardwarePerfError::new(
                "perf hardware profiler has not been started",
            ));
        }
        if !self.is_running()? {
            return Err(HardwarePerfError::new(format!(
                "perf hardware profiler exited before {action}: {}",
                self.stderr_text()
            )));
        }
        let request_ns = monotonic_ns();
        self.control_write
            .as_mut()
            .expect("control descriptor checked")
            .write_all(format!("{action}\n").as_bytes())
            .map_err(|error| io_error(&format!("unable to send perf {action}"), error))?;
        let deadline = monotonic_ns().saturating_add(self.attach_timeout.as_nanos() as u64);
        loop {
            if let Some(newline) = self
                .acknowledgement_buffer
                .iter()
                .position(|byte| *byte == b'\n')
            {
                let mut response = self
                    .acknowledgement_buffer
                    .drain(..=newline)
                    .collect::<Vec<_>>();
                let first = response
                    .iter()
                    .position(|byte| !byte.is_ascii_whitespace() && *byte != 0)
                    .unwrap_or(response.len());
                let end = response
                    .iter()
                    .rposition(|byte| !byte.is_ascii_whitespace() && *byte != 0)
                    .map_or(first, |index| index + 1);
                response = response[first..end].to_vec();
                if response != b"ack" {
                    return Err(HardwarePerfError::new(format!(
                        "unexpected perf control response: {:?}",
                        String::from_utf8_lossy(&response)
                    )));
                }
                return PerfControlBounds::new(action, request_ns, monotonic_ns());
            }
            if !self.is_running()? {
                return Err(HardwarePerfError::new(format!(
                    "perf hardware profiler exited before {action} acknowledgement: {}",
                    self.stderr_text()
                )));
            }
            let now = monotonic_ns();
            if now >= deadline {
                return Err(HardwarePerfError::new(format!(
                    "timed out waiting for perf {action} acknowledgement"
                )));
            }
            let remaining_ns = deadline - now;
            let timeout = Duration::from_nanos(remaining_ns).min(CONTROL_POLL_INTERVAL);
            let timeout_ms = timeout.as_millis().max(1).min(i32::MAX as u128) as i32;
            let descriptor = self
                .acknowledgement_read
                .as_ref()
                .expect("acknowledgement descriptor checked")
                .as_raw_fd();
            let mut descriptor_event = libc::pollfd {
                fd: descriptor,
                events: libc::POLLIN,
                revents: 0,
            };
            let result = unsafe { libc::poll(&mut descriptor_event, 1, timeout_ms) };
            if result < 0 {
                let error = io::Error::last_os_error();
                if error.kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                return Err(io_error("unable to poll perf acknowledgement", error));
            }
            if result == 0 {
                continue;
            }
            if descriptor_event.revents & (libc::POLLERR | libc::POLLHUP | libc::POLLNVAL) != 0
                && descriptor_event.revents & libc::POLLIN == 0
            {
                return Err(HardwarePerfError::new(format!(
                    "perf control pipe closed before {action} acknowledgement"
                )));
            }
            let mut chunk = [0_u8; 4096];
            let count = self
                .acknowledgement_read
                .as_mut()
                .expect("acknowledgement descriptor checked")
                .read(&mut chunk)
                .map_err(|error| io_error("unable to read perf acknowledgement", error))?;
            if count == 0 {
                return Err(HardwarePerfError::new(format!(
                    "perf control pipe closed before {action} acknowledgement"
                )));
            }
            self.acknowledgement_buffer
                .extend_from_slice(&chunk[..count]);
        }
    }

    pub(crate) fn attach(&mut self, cgroup_path: &Path) -> Result<(), HardwarePerfError> {
        if self.process.is_some() {
            return Err(HardwarePerfError::new(
                "perf hardware profiler is already attached",
            ));
        }
        fs::create_dir_all(&self.output_dir)
            .map_err(|error| io_error("unable to create hardware perf output directory", error))?;
        File::create(&self.stats_path)
            .map_err(|error| io_error("unable to create perf stat artifact", error))?;
        self.provenance = collect_hardware_perf_provenance(&self.perf);
        let perf_cgroup = perf_cgroup(cgroup_path)?;
        let (control_read, control_write) = pipe()?;
        let (acknowledgement_read, acknowledgement_write) = pipe()?;
        let control_read_fd = control_read.as_raw_fd();
        let acknowledgement_write_fd = acknowledgement_write.as_raw_fd();
        let arguments = perf_arguments(
            &self.stats_path,
            &self.events,
            &perf_cgroup,
            self.cpuset_cpus.as_deref(),
            control_read_fd,
            acknowledgement_write_fd,
        );
        let stderr = File::create(&self.stderr_path)
            .map_err(|error| io_error("unable to create perf stderr artifact", error))?;
        set_close_on_exec(control_read_fd, false)?;
        if let Err(error) = set_close_on_exec(acknowledgement_write_fd, false) {
            let _ = set_close_on_exec(control_read_fd, true);
            return Err(error);
        }
        let spawn_result = Command::new(&self.perf)
            .args(&arguments)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .process_group(0)
            .spawn();
        let restore_control = set_close_on_exec(control_read_fd, true);
        let restore_ack = set_close_on_exec(acknowledgement_write_fd, true);
        if let Err(error) = restore_control.and(restore_ack) {
            if let Ok(process) = spawn_result {
                drop(control_read);
                drop(acknowledgement_write);
                self.process_group_id = Some(
                    libc::pid_t::try_from(process.id())
                        .expect("Linux child process IDs are representable as pid_t"),
                );
                self.process = Some(process);
                let cleanup = self.close();
                return Err(combine_cleanup_error(error, cleanup));
            }
            return Err(error);
        }
        let process = spawn_result.map_err(|error| {
            io_error(
                &format!(
                    "unable to start perf hardware profiler with {:?}",
                    self.perf
                ),
                error,
            )
        })?;
        drop(control_read);
        drop(acknowledgement_write);
        self.process_group_id = Some(
            libc::pid_t::try_from(process.id())
                .expect("Linux child process IDs are representable as pid_t"),
        );
        self.process = Some(process);
        self.control_write = Some(control_write);
        self.acknowledgement_read = Some(acknowledgement_read);
        self.perf_cgroup = Some(perf_cgroup);
        self.attach_ready_bounds = None;
        self.enable_bounds = None;
        self.disable_bounds = None;
        self.enabled = false;
        self.acknowledgement_buffer.clear();
        match self.control("disable") {
            Ok(bounds) => {
                self.attach_ready_bounds = Some(bounds);
                Ok(())
            }
            Err(error) => {
                let cleanup = self.close();
                Err(combine_cleanup_error(error, cleanup))
            }
        }
    }

    pub(crate) fn enable(&mut self) -> Result<PerfControlBounds, HardwarePerfError> {
        if self.process.is_none() || self.perf_cgroup.is_none() {
            return Err(HardwarePerfError::new(
                "perf hardware profiler has not been attached",
            ));
        }
        if self.enabled || self.enable_bounds.is_some() {
            return Err(HardwarePerfError::new(
                "perf hardware profiler is already enabled",
            ));
        }
        let bounds = self.control("enable")?;
        self.enable_bounds = Some(bounds.clone());
        self.enabled = true;
        Ok(bounds)
    }

    pub(crate) fn poll(&mut self) -> Result<(), HardwarePerfError> {
        if self.process.is_none() {
            return Err(HardwarePerfError::new(
                "perf hardware profiler has not been started",
            ));
        }
        if self.is_running()? {
            Ok(())
        } else {
            Err(HardwarePerfError::new(format!(
                "perf hardware profiler exited before measurement completed: {}",
                self.stderr_text()
            )))
        }
    }

    fn shutdown_profiler_scope(
        &mut self,
        graceful_signal: libc::c_int,
        graceful_timeout: Duration,
    ) -> HardwarePerfChildShutdown<std::process::ExitStatus> {
        let mut outcome = if let Some(process) = self.process.as_mut() {
            let process_group_id = self
                .process_group_id
                .expect("started perf hardware child has an owned process group");
            shutdown_hardware_perf_child(
                &mut RealHardwarePerfChild {
                    child: process,
                    process_group_id,
                },
                graceful_signal,
                graceful_timeout,
                CLEANUP_KILL_TIMEOUT,
            )
        } else {
            HardwarePerfChildShutdown {
                status: None,
                errors: Vec::new(),
            }
        };
        if outcome.status.is_some() {
            self.process = None;
        }
        if self.process.is_none() {
            if let Some(process_group_id) = self.process_group_id {
                let group = cleanup_hardware_perf_process_group(
                    process_group_id,
                    CLEANUP_TERM_TIMEOUT,
                    CLEANUP_KILL_TIMEOUT,
                );
                if group.verified_absent {
                    self.process_group_id = None;
                }
                outcome.errors.extend(group.errors);
            }
        }
        outcome
    }

    pub(crate) fn finish(
        &mut self,
        requested_window_start_ns: u64,
        requested_window_end_ns: u64,
    ) -> Result<Value, HardwarePerfError> {
        let result = self.finish_inner(requested_window_start_ns, requested_window_end_ns);
        match result {
            Ok(report) => Ok(report),
            Err(primary) => {
                let cleanup = self.close();
                Err(combine_cleanup_error(primary, cleanup))
            }
        }
    }

    fn finish_inner(
        &mut self,
        requested_window_start_ns: u64,
        requested_window_end_ns: u64,
    ) -> Result<Value, HardwarePerfError> {
        if self.process.is_none() || self.perf_cgroup.is_none() {
            return Err(HardwarePerfError::new(
                "perf hardware profiler has not been started",
            ));
        }
        if !self.enabled || self.enable_bounds.is_none() {
            return Err(HardwarePerfError::new(
                "perf hardware profiler was never enabled",
            ));
        }
        if requested_window_end_ns <= requested_window_start_ns {
            return Err(HardwarePerfError::new(
                "hardware profiling window ends before it starts",
            ));
        }
        let disable_bounds = self.control("disable")?;
        self.disable_bounds = Some(disable_bounds.clone());
        self.enabled = false;
        let outcome = self.shutdown_profiler_scope(libc::SIGINT, STOP_TIMEOUT);
        let status = outcome.status;
        self.control_write = None;
        self.acknowledgement_read = None;
        if !outcome.errors.is_empty() {
            return Err(HardwarePerfError::with_cleanup_failure(
                outcome.errors.join("; "),
            ));
        }
        let status = status.ok_or_else(|| {
            HardwarePerfError::with_cleanup_failure(
                "perf hardware profiler cleanup returned without verified child absence",
            )
        })?;
        if !status.success() && status.signal() != Some(libc::SIGINT) {
            return Err(HardwarePerfError::new(format!(
                "perf hardware profiler failed while stopping: {}",
                self.stderr_text()
            )));
        }
        let stats = fs::read_to_string(&self.stats_path)
            .map_err(|error| io_error("perf hardware profiler produced no stat output", error))?;
        let perf_cgroup = self.perf_cgroup.as_deref().expect("cgroup checked");
        let (counters, mut anomalies) = parse_perf_stat_csv(
            &stats,
            &self.events,
            self.minimum_running_ratio,
            Some(perf_cgroup),
        )?;
        let attach_ready = self
            .attach_ready_bounds
            .as_ref()
            .ok_or_else(|| HardwarePerfError::new("perf readiness acknowledgement is missing"))?;
        let enable_bounds = self
            .enable_bounds
            .as_ref()
            .ok_or_else(|| HardwarePerfError::new("perf enable acknowledgement is missing"))?;
        let outer_start_ns = enable_bounds.request_monotonic_ns;
        let outer_end_ns = disable_bounds.acknowledgement_monotonic_ns;
        if outer_end_ns <= outer_start_ns {
            return Err(HardwarePerfError::new(
                "perf control window did not have positive duration",
            ));
        }
        let definite_start_ns = enable_bounds.acknowledgement_monotonic_ns;
        let definite_end_ns = disable_bounds.request_monotonic_ns;
        let (definite_start, definite_end) = if definite_end_ns < definite_start_ns {
            anomalies.push(
                "perf enable/disable acknowledgement bounds overlap; no definite active window"
                    .to_owned(),
            );
            (None, None)
        } else {
            (Some(definite_start_ns), Some(definite_end_ns))
        };
        let all_counters_valid = counters.iter().all(|counter| counter.valid);
        let instrumentation_valid = anomalies.is_empty();
        Ok(json!({
            "schema_version": HARDWARE_PERF_SCHEMA_VERSION,
            "method": "perf-stat-cgroup-cpu-wide",
            "scope": "measured container cgroup on configured logical CPUs",
            "perf_cgroup": perf_cgroup,
            "cpuset_cpus": self.cpuset_cpus,
            "window_start_ns": outer_start_ns,
            "window_end_ns": outer_end_ns,
            "window_duration_sec": rounded((outer_end_ns - outer_start_ns) as f64 / 1_000_000_000.0, 9),
            "window_start_alignment_error_ns": enable_bounds.uncertainty_ns,
            "window_end_alignment_error_ns": disable_bounds.uncertainty_ns,
            "requested_lifecycle_window_start_ns": requested_window_start_ns,
            "requested_lifecycle_window_end_ns": requested_window_end_ns,
            "definite_active_window_start_ns": definite_start,
            "definite_active_window_end_ns": definite_end,
            "control_bounds": {
                "attach_ready": attach_ready,
                "enable": enable_bounds,
                "disable": disable_bounds,
            },
            "minimum_running_ratio": self.minimum_running_ratio,
            "hybrid_pmu_counts_merged": true,
            "provenance": self.provenance,
            "counters": counters,
            "anomalies": anomalies,
            "instrumentation_valid": instrumentation_valid,
            "all_requested_metrics_valid": instrumentation_valid && all_counters_valid,
            "stats_path": self.stats_path,
            "stderr_path": self.stderr_path,
        }))
    }

    pub(crate) fn close(&mut self) -> Result<(), HardwarePerfError> {
        self.control_write = None;
        self.acknowledgement_read = None;
        self.enabled = false;
        let outcome = self.shutdown_profiler_scope(libc::SIGTERM, CLEANUP_TERM_TIMEOUT);
        if outcome.errors.is_empty() {
            Ok(())
        } else {
            Err(HardwarePerfError::with_cleanup_failure(
                outcome.errors.join("; "),
            ))
        }
    }
}

impl Drop for PerfCgroupHardwareProfiler {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

/// Fail-closed lifecycle wrapper used by the coordinator.  PMU availability
/// is a property of the host/kernel, not of the workload: an attach or control
/// failure therefore remains an invalid hardware-profiler report while the
/// workload and its independent correctness checks are still allowed to run.
pub(crate) struct HardwarePerfRun {
    collector: PerfCgroupHardwareProfiler,
    error: Option<String>,
    report: Option<Value>,
}

impl HardwarePerfRun {
    pub(crate) fn new(
        output_dir: impl Into<PathBuf>,
        cpuset_cpus: Option<String>,
        perf: impl Into<String>,
        events: Vec<String>,
        attach_timeout: Duration,
        minimum_running_ratio: f64,
    ) -> Result<Self, HardwarePerfError> {
        Ok(Self {
            collector: PerfCgroupHardwareProfiler::new(
                output_dir,
                cpuset_cpus,
                perf,
                events,
                attach_timeout,
                minimum_running_ratio,
            )?,
            error: None,
            report: None,
        })
    }

    fn record_error(&mut self, error: HardwarePerfError) -> Result<(), HardwarePerfError> {
        let prior_cleanup_failed = error.cleanup_failed;
        match self.collector.close() {
            Ok(()) => {
                if self.error.is_none() {
                    self.error = Some(format!("HardwarePerfError: {error}"));
                }
                if prior_cleanup_failed {
                    Err(error)
                } else {
                    Ok(())
                }
            }
            Err(cleanup) => {
                let combined = HardwarePerfError::with_cleanup_failure(format!(
                    "{error}; perf hardware profiler cleanup also failed: {cleanup}"
                ));
                if self.error.is_none() {
                    self.error = Some(format!("HardwarePerfError: {combined}"));
                }
                Err(combined)
            }
        }
    }

    pub(crate) fn attach(&mut self, cgroup_path: &Path) -> Result<(), HardwarePerfError> {
        if self.error.is_none() {
            if let Err(error) = self.collector.attach(cgroup_path) {
                self.record_error(error)?;
            }
        }
        Ok(())
    }

    pub(crate) fn enable(&mut self) -> Result<(), HardwarePerfError> {
        if self.error.is_none() {
            if let Err(error) = self.collector.enable() {
                self.record_error(error)?;
            }
        }
        Ok(())
    }

    pub(crate) fn poll(&mut self) -> Result<(), HardwarePerfError> {
        if self.error.is_none() {
            if let Err(error) = self.collector.poll() {
                self.record_error(error)?;
            }
        }
        Ok(())
    }

    pub(crate) fn close(&mut self) -> Result<(), HardwarePerfError> {
        self.collector.close()
    }

    fn unavailable_report(&self) -> Value {
        let reason = self
            .error
            .clone()
            .unwrap_or_else(|| "hardware perf profiler produced no report".to_owned());
        let counters = self
            .collector
            .events
            .iter()
            .map(|event| {
                json!({
                    "event": event,
                    "namespace": event_namespace(event),
                    "status": "unavailable",
                    "value": Value::Null,
                    "unit": Value::Null,
                    "time_enabled_ns": Value::Null,
                    "time_running_ns": Value::Null,
                    "time_running_fraction": Value::Null,
                    "valid": false,
                    "invalid_reason": reason,
                    "observed_cgroup": Value::Null,
                })
            })
            .collect::<Vec<_>>();
        json!({
            "schema_version": "cgroup-perf-hardware-unavailable.v1",
            "instrumentation_valid": false,
            "all_requested_metrics_valid": false,
            "invalid_reasons": [reason],
            "provenance": self.collector.provenance,
            "counters": counters,
            "stats_path": self.collector.stats_path,
            "stderr_path": self.collector.stderr_path,
        })
    }

    pub(crate) fn finish(
        &mut self,
        requested_window_start_ns: u64,
        requested_window_end_ns: u64,
    ) -> Result<Value, HardwarePerfError> {
        if self.error.is_none() {
            match self
                .collector
                .finish(requested_window_start_ns, requested_window_end_ns)
            {
                Ok(report) => self.report = Some(report),
                Err(error) => self.record_error(error)?,
            }
        }
        let report = self
            .report
            .clone()
            .unwrap_or_else(|| self.unavailable_report());
        let mut encoded = serde_json::to_vec_pretty(&report)
            .map_err(|error| HardwarePerfError::new(error.to_string()))?;
        encoded.push(b'\n');
        fs::write(
            self.collector.output_dir.join("hardware_perf_report.json"),
            encoded,
        )
        .map_err(|error| io_error("unable to retain hardware perf report", error))?;
        Ok(report)
    }
}

#[derive(Clone, Debug)]
pub(crate) struct HardwarePerfEvidence {
    pub(crate) raw_report: Value,
    pub(crate) valid: bool,
    pub(crate) reasons: Vec<Value>,
    pub(crate) stream_ids: Vec<String>,
    pub(crate) profiler: Value,
    pub(crate) streams: Vec<Value>,
    pub(crate) artifacts: Vec<Value>,
}

fn sha256_file(path: &Path) -> Result<String, HardwarePerfError> {
    let mut file = File::open(path)
        .map_err(|error| io_error("unable to open hardware perf artifact", error))?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file
            .read(&mut buffer)
            .map_err(|error| io_error("unable to read hardware perf artifact", error))?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn retained_artifact(
    output_dir: &Path,
    filename: &str,
    artifact_id: &str,
    media_type: &str,
) -> Result<Value, HardwarePerfError> {
    let path = output_dir.join(filename);
    let metadata = path.metadata().map_err(|error| {
        io_error(
            &format!("hardware perf did not retain required artifact {filename:?}"),
            error,
        )
    })?;
    if !metadata.is_file() {
        return Err(HardwarePerfError::new(format!(
            "hardware perf artifact is not a regular file: {}",
            path.display()
        )));
    }
    Ok(json!({
        "artifact_id": artifact_id,
        "relative_path": filename,
        "media_type": media_type,
        "sha256": sha256_file(&path)?,
        "size_bytes": metadata.len(),
    }))
}

fn hardware_stream_id(event: &str) -> String {
    format!("hardware-{}", event.replace('/', "-"))
}

fn report_reasons(report: &Value) -> Vec<Value> {
    let mut reasons = BTreeSet::new();
    for reason in report
        .get("invalid_reasons")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        reasons.insert(reason.to_owned());
    }
    for reason in report
        .get("anomalies")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
    {
        reasons.insert(reason.to_owned());
    }
    for reason in report
        .get("counters")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|counter| counter.get("valid") != Some(&Value::Bool(true)))
        .filter_map(|counter| counter.get("invalid_reason").and_then(Value::as_str))
    {
        reasons.insert(reason.to_owned());
    }
    if report.get("all_requested_metrics_valid") != Some(&Value::Bool(true)) && reasons.is_empty() {
        reasons.insert("hardware counters invalid".to_owned());
    }
    reasons.into_iter().map(Value::String).collect()
}

pub(crate) fn build_hardware_perf_evidence(
    run_id: &str,
    scope_id: &str,
    report: &Value,
    output_dir: &Path,
) -> Result<HardwarePerfEvidence, HardwarePerfError> {
    let artifacts = vec![
        retained_artifact(
            output_dir,
            "hardware_perf_report.json",
            "hardware-perf-report",
            "application/json",
        )?,
        retained_artifact(
            output_dir,
            "hardware_perf_stat.csv",
            "hardware-perf-stat",
            "text/csv",
        )?,
        retained_artifact(
            output_dir,
            "hardware_perf.stderr.txt",
            "hardware-perf-stderr",
            "text/plain",
        )?,
    ];
    let artifact_ids = vec![
        "hardware-perf-report",
        "hardware-perf-stat",
        "hardware-perf-stderr",
    ];
    let outer_start = report.get("window_start_ns").and_then(Value::as_u64);
    let outer_end = report.get("window_end_ns").and_then(Value::as_u64);
    let has_window = outer_start
        .zip(outer_end)
        .is_some_and(|(start, end)| end > start);
    let alignment_error_ns = report
        .get("window_start_alignment_error_ns")
        .and_then(Value::as_u64)
        .unwrap_or(0)
        .max(
            report
                .get("window_end_alignment_error_ns")
                .and_then(Value::as_u64)
                .unwrap_or(0),
        );
    let counters = report
        .get("counters")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    let mut stream_ids = Vec::new();
    let mut streams = Vec::new();
    let mut observed_ids = BTreeSet::new();
    for (index, counter) in counters.iter().enumerate() {
        let event = counter
            .get("event")
            .and_then(Value::as_str)
            .map(ToOwned::to_owned)
            .unwrap_or_else(|| format!("unknown-{index}"));
        let mut stream_id = hardware_stream_id(&event);
        if !observed_ids.insert(stream_id.clone()) {
            stream_id = format!("{stream_id}-{index}");
            observed_ids.insert(stream_id.clone());
        }
        let counter_valid = counter.get("valid") == Some(&Value::Bool(true));
        let reason = counter
            .get("invalid_reason")
            .and_then(Value::as_str)
            .unwrap_or("hardware counter did not produce a value")
            .to_owned();
        let status = if counter_valid { "valid" } else { "invalid" };
        let reasons = if counter_valid {
            Vec::<String>::new()
        } else {
            vec![reason]
        };
        let shape = if has_window {
            "window_aggregate"
        } else {
            "scalar"
        };
        let record = json!({
            "record_id": if has_window { format!("{stream_id}:window") } else { format!("{stream_id}:value") },
            "run_id": run_id,
            "scope_id": scope_id,
            "profiler_id": "hardware-perf",
            "stream_id": stream_id,
            "shape": shape,
            "status": status,
            "reasons": reasons,
            "value": counter.get("value").cloned().unwrap_or(Value::Null),
            "source_clock_id": if has_window { json!("host-monotonic") } else { Value::Null },
            "native_timestamp_ns": Value::Null,
            "native_start_ns": if has_window { outer_start.map_or(Value::Null, Value::from) } else { Value::Null },
            "native_end_ns": if has_window { outer_end.map_or(Value::Null, Value::from) } else { Value::Null },
            "canonical_timestamp_ns": Value::Null,
            "canonical_start_ns": if has_window { outer_start.map_or(Value::Null, Value::from) } else { Value::Null },
            "canonical_end_ns": if has_window { outer_end.map_or(Value::Null, Value::from) } else { Value::Null },
            "alignment_error_ns": if has_window { json!(alignment_error_ns) } else { Value::Null },
            "acquisition_bounds": Value::Null,
            "attributes": {
                "event_namespace": counter.get("namespace").cloned().unwrap_or(Value::Null),
                "counter_status": counter.get("status").cloned().unwrap_or(Value::Null),
                "time_enabled_ns": counter.get("time_enabled_ns").cloned().unwrap_or(Value::Null),
                "time_running_ns": counter.get("time_running_ns").cloned().unwrap_or(Value::Null),
                "time_running_fraction": counter.get("time_running_fraction").cloned().unwrap_or(Value::Null),
                "observed_cgroup": counter.get("observed_cgroup").cloned().unwrap_or(Value::Null),
                "definite_active_window_start_ns": report.get("definite_active_window_start_ns").cloned().unwrap_or(Value::Null),
                "definite_active_window_end_ns": report.get("definite_active_window_end_ns").cloned().unwrap_or(Value::Null),
                "control_bounds": report.get("control_bounds").cloned().unwrap_or(Value::Null),
            },
            "raw_artifact_ids": artifact_ids,
            "provenance": {
                "method": "perf-stat-cgroup-cpu-wide",
                "scope_attribution_checked": counter
                    .get("observed_cgroup")
                    .and_then(Value::as_str)
                    .is_some(),
            },
        });
        streams.push(json!({
            "stream_id": stream_id,
            "profiler_id": "hardware-perf",
            "metric": format!("hardware.{event}"),
            "shape": shape,
            "unit": counter.get("unit").and_then(Value::as_str).unwrap_or("count"),
            "status": status,
            "reasons": reasons,
            "records": [record],
            "raw_artifact_ids": artifact_ids,
            "cadence": Value::Null,
            "gaps": [],
            "display": {"group": "Hardware counters", "label": event},
            "provenance": {},
            "extensions": {
                "measurement_domains": ["hardware.pmu"],
                "measurement_perturbations": [],
            },
        }));
        stream_ids.push(stream_id);
    }
    let valid = report.get("all_requested_metrics_valid") == Some(&Value::Bool(true))
        && report.get("instrumentation_valid") == Some(&Value::Bool(true))
        && !stream_ids.is_empty();
    let reasons = report_reasons(report);
    let profiler = json!({
        "profiler_id": "hardware-perf",
        "profiler_version": HARDWARE_PERF_SCHEMA_VERSION,
        "status": if valid { "valid" } else { "invalid" },
        "reasons": reasons,
        "stream_ids": stream_ids,
        "intrusive": false,
        "configuration": {
            "method": report.get("method").cloned().unwrap_or_else(|| json!("perf-stat-cgroup-cpu-wide")),
            "perf_cgroup": report.get("perf_cgroup").cloned().unwrap_or(Value::Null),
            "cpuset_cpus": report.get("cpuset_cpus").cloned().unwrap_or(Value::Null),
            "minimum_running_ratio": report.get("minimum_running_ratio").cloned().unwrap_or(Value::Null),
            "hybrid_pmu_counts_merged": report.get("hybrid_pmu_counts_merged").cloned().unwrap_or(Value::Null),
        },
        "provenance": report.get("provenance").cloned().unwrap_or_else(|| json!({})),
        "extensions": {
            "measurement_domains": ["hardware.pmu"],
            "measurement_perturbations": [],
            "invalidates_measurement_domains": [],
            "invalidates_metrics": [],
        },
    });
    Ok(HardwarePerfEvidence {
        raw_report: report.clone(),
        valid,
        reasons,
        stream_ids,
        profiler,
        streams,
        artifacts,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::collections::VecDeque;
    use std::os::unix::fs::PermissionsExt;
    use tempfile::tempdir;

    struct FakeChildOperations {
        polls: VecDeque<Result<Option<i32>, HardwarePerfError>>,
        signals: VecDeque<Result<(), HardwarePerfError>>,
        waits: VecDeque<Result<Option<i32>, HardwarePerfError>>,
        observed_signals: Vec<libc::c_int>,
        observed_waits: Vec<Duration>,
    }

    impl HardwarePerfChildOperations for FakeChildOperations {
        type Status = i32;

        fn poll(&mut self) -> Result<Option<Self::Status>, HardwarePerfError> {
            self.polls.pop_front().unwrap_or(Ok(None))
        }

        fn signal(&mut self, signal: libc::c_int) -> Result<(), HardwarePerfError> {
            self.observed_signals.push(signal);
            self.signals.pop_front().unwrap_or(Ok(()))
        }

        fn wait_bounded(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<Self::Status>, HardwarePerfError> {
            self.observed_waits.push(timeout);
            self.waits.pop_front().unwrap_or(Ok(None))
        }
    }

    fn fake_child_operations() -> FakeChildOperations {
        FakeChildOperations {
            polls: VecDeque::from([Ok(None)]),
            signals: VecDeque::new(),
            waits: VecDeque::new(),
            observed_signals: Vec::new(),
            observed_waits: Vec::new(),
        }
    }

    fn events(values: &[&str]) -> Vec<String> {
        values.iter().map(|value| (*value).to_owned()).collect()
    }

    #[test]
    fn cleanup_retains_signal_error_even_when_child_is_reaped() {
        let mut operations = fake_child_operations();
        operations
            .signals
            .push_back(Err(HardwarePerfError::new("injected SIGTERM failure")));
        operations.waits.push_back(Ok(Some(0)));

        let outcome = shutdown_hardware_perf_child(
            &mut operations,
            libc::SIGTERM,
            Duration::from_millis(7),
            Duration::from_millis(11),
        );

        assert_eq!(outcome.status, Some(0));
        assert_eq!(outcome.errors, ["injected SIGTERM failure"]);
        assert_eq!(operations.observed_signals, [libc::SIGTERM]);
        assert_eq!(operations.observed_waits, [Duration::from_millis(7)]);
    }

    #[test]
    fn cleanup_escalates_after_wait_error_and_retains_it() {
        let mut operations = fake_child_operations();
        operations
            .waits
            .push_back(Err(HardwarePerfError::new("injected wait failure")));
        operations.waits.push_back(Ok(Some(9)));

        let outcome = shutdown_hardware_perf_child(
            &mut operations,
            libc::SIGTERM,
            Duration::from_millis(7),
            Duration::from_millis(11),
        );

        assert_eq!(outcome.status, Some(9));
        assert_eq!(outcome.errors, ["injected wait failure"]);
        assert_eq!(operations.observed_signals, [libc::SIGTERM, libc::SIGKILL]);
        assert_eq!(
            operations.observed_waits,
            [Duration::from_millis(7), Duration::from_millis(11)]
        );
    }

    #[test]
    fn cleanup_timeouts_are_bounded_and_unreaped_child_fails_closed() {
        let mut operations = fake_child_operations();
        operations.waits.extend([Ok(None), Ok(None)]);

        let outcome = shutdown_hardware_perf_child(
            &mut operations,
            libc::SIGTERM,
            Duration::from_millis(7),
            Duration::from_millis(11),
        );

        assert_eq!(outcome.status, None);
        assert!(outcome.errors[0].contains("timed out after 0.007s"));
        assert!(outcome.errors[1].contains("remains unreaped after SIGKILL"));
        assert_eq!(operations.observed_signals, [libc::SIGTERM, libc::SIGKILL]);
        assert_eq!(
            operations.observed_waits,
            [Duration::from_millis(7), Duration::from_millis(11)]
        );
    }

    #[test]
    fn cancellation_and_cleanup_failures_are_combined() {
        let error = combine_cleanup_error(
            HardwarePerfError::new("profile cancelled after PMU attachment"),
            Err(HardwarePerfError::new("injected child reap failure")),
        );
        assert!(error
            .to_string()
            .contains("profile cancelled after PMU attachment"));
        assert!(error.to_string().contains("injected child reap failure"));
        assert!(error.cleanup_failed);
    }

    #[test]
    fn reaped_leader_does_not_hide_lingering_process_group_member() {
        let mut observations = VecDeque::from([Ok(true), Ok(true), Ok(false)]);
        let mut signals = Vec::new();
        let outcome = cleanup_hardware_perf_process_group_with(
            4242,
            Duration::ZERO,
            Duration::ZERO,
            |_| observations.pop_front().unwrap_or(Ok(false)),
            |_, signal| {
                signals.push(signal);
                Ok(())
            },
        );

        assert!(outcome.verified_absent);
        assert_eq!(signals, [libc::SIGTERM, libc::SIGKILL]);
        assert!(outcome.errors[0].contains("timed out after 0.000s"));
    }

    #[test]
    fn process_group_inspection_error_is_retained_after_absence_is_proven() {
        let mut observations = VecDeque::from([
            Err(HardwarePerfError::new("injected group inspection failure")),
            Ok(false),
        ]);
        let outcome = cleanup_hardware_perf_process_group_with(
            4242,
            Duration::ZERO,
            Duration::ZERO,
            |_| observations.pop_front().unwrap_or(Ok(false)),
            |_, _| Ok(()),
        );

        assert!(outcome.verified_absent);
        assert_eq!(outcome.errors, ["injected group inspection failure"]);
    }

    #[test]
    fn wrapper_never_downgrades_structured_cleanup_failure() {
        let directory = tempdir().unwrap();
        let mut run = HardwarePerfRun::new(
            directory.path(),
            Some("0".to_owned()),
            "/missing/perf",
            events(&["cycles"]),
            Duration::from_millis(10),
            0.95,
        )
        .unwrap();
        let error = run
            .record_error(HardwarePerfError::with_cleanup_failure(
                "injected finish signal failure after verified leader reap",
            ))
            .unwrap_err();

        assert!(error.cleanup_failed);
        assert!(error.to_string().contains("injected finish signal failure"));
        assert!(run.error.as_deref().unwrap().contains("signal failure"));
    }

    #[test]
    fn preserves_counts_and_runtime_validity() {
        let (counters, anomalies) = parse_perf_stat_csv(
            "1000;;cycles;2000000;100.00;;\n500;;instructions;2000000;96.00;;\n25;;branches;2000000;90.00;;\n",
            &events(&["cycles", "instructions", "branches"]),
            0.95,
            None,
        )
        .unwrap();
        assert!(anomalies.is_empty());
        assert_eq!(
            serde_json::to_value(&counters[0]).unwrap(),
            json!({
                "event": "cycles", "namespace": "portable", "status": "valid",
                "value": 1000, "unit": null, "time_enabled_ns": 2_000_000,
                "time_running_ns": 2_000_000, "time_running_fraction": 1.0,
                "valid": true, "invalid_reason": null, "observed_cgroup": null,
            })
        );
        assert!(counters[1].valid);
        assert_eq!(counters[1].time_running_ns, Some(1_920_000));
        assert_eq!(counters[2].status, "multiplexed");
        assert!(!counters[2].valid);
    }

    #[test]
    fn unsupported_not_counted_and_missing_remain_invalid() {
        let (counters, anomalies) = parse_perf_stat_csv(
            "<not supported>;;cycles;0;100.00;;\n<not counted>;;instructions;0;100.00;;\n",
            &events(&["cycles", "instructions", "branches"]),
            0.95,
            None,
        )
        .unwrap();
        assert!(anomalies.is_empty());
        assert_eq!(
            counters
                .iter()
                .map(|counter| counter.status.as_str())
                .collect::<Vec<_>>(),
            ["unsupported", "not_counted", "missing"]
        );
        assert!(counters
            .iter()
            .all(|counter| counter.value.is_none() && !counter.valid));
    }

    #[test]
    fn malformed_duplicate_and_short_rows_fail_closed() {
        let (counters, anomalies) = parse_perf_stat_csv(
            "bad;;cycles;oops;100.00;;\n12;;instructions;1000;100.00;;\n13;;instructions;1000;100.00;;\ntoo;short\n",
            &events(&["cycles", "instructions"]),
            0.95,
            None,
        )
        .unwrap();
        assert_eq!(counters[0].status, "malformed");
        assert_eq!(counters[1].status, "duplicate");
        assert_eq!(counters[1].value, Some(json!(12)));
        assert!(anomalies
            .iter()
            .any(|value| value.contains("malformed values")));
        assert!(anomalies.iter().any(|value| value.contains("duplicate")));
        assert!(anomalies.iter().any(|value| value.contains("fewer than 5")));
    }

    #[test]
    fn verifies_cgroup_layout_without_weakening_attribution() {
        let cgroup = "/system.slice/docker-example.scope";
        let (counters, anomalies) = parse_perf_stat_csv(
            "1200;;cycles;/system.slice/docker-example.scope;2000000;99.50;;\n<not supported>;;instructions;/system.slice/docker-example.scope;0;100.00;;\n",
            &events(&["cycles", "instructions"]),
            0.95,
            Some(cgroup),
        )
        .unwrap();
        assert!(anomalies.is_empty());
        assert_eq!(counters[0].time_running_ns, Some(1_990_000));
        assert_eq!(counters[0].observed_cgroup.as_deref(), Some(cgroup));
        let (wrong, anomalies) = parse_perf_stat_csv(
            "1200;;cycles;/system.slice/wrong.scope;2000000;100.00;;\n",
            &events(&["cycles"]),
            0.95,
            Some(cgroup),
        )
        .unwrap();
        assert_eq!(wrong[0].status, "scope_mismatch");
        assert!(!wrong[0].valid);
        assert!(anomalies[0].contains("expected"));
    }

    #[test]
    fn control_bounds_reject_backwards_or_unknown_actions() {
        assert_eq!(
            PerfControlBounds::new("enable", 1_000, 1_075)
                .unwrap()
                .uncertainty_ns,
            75
        );
        assert!(PerfControlBounds::new("start", 1, 2).is_err());
        assert!(PerfControlBounds::new("disable", 2, 1).is_err());
    }

    #[test]
    fn integer_counts_do_not_lose_u64_precision() {
        let (counters, anomalies) = parse_perf_stat_csv(
            "18446744073709551615;;custom:event;1000;100.00;;\n",
            &events(&["custom:event"]),
            0.95,
            None,
        )
        .unwrap();
        assert!(anomalies.is_empty());
        assert_eq!(counters[0].value, Some(json!(u64::MAX)));
        assert_eq!(counters[0].namespace, "custom-unclassified");
    }

    #[test]
    fn perf_command_starts_disabled_and_keeps_exact_scope() {
        let arguments = perf_arguments(
            Path::new("/tmp/result/hardware_perf_stat.csv"),
            &events(&["cycles", "instructions"]),
            "/system.slice/docker-example.scope",
            Some("2-3"),
            17,
            18,
        );
        assert_eq!(
            arguments,
            [
                "stat",
                "-D",
                "-1",
                "--control",
                "fd:17,18",
                "-x",
                ";",
                "--no-big-num",
                "--hybrid-merge",
                "-o",
                "/tmp/result/hardware_perf_stat.csv",
                "-e",
                "cycles",
                "-e",
                "instructions",
                "-G",
                "/system.slice/docker-example.scope",
                "-a",
                "-C",
                "2-3",
            ]
        );
        assert_eq!(
            perf_cgroup(Path::new(
                "/sys/fs/cgroup/system.slice/docker-example.scope"
            ))
            .unwrap(),
            "/system.slice/docker-example.scope"
        );
        assert!(perf_cgroup(Path::new("/tmp/not-a-cgroup")).is_err());
    }

    #[test]
    fn collector_acknowledges_ready_enable_and_disable_bounds() {
        let directory = tempdir().unwrap();
        let fake_perf = directory.path().join("fake-perf");
        fs::write(
            &fake_perf,
            r#"#!/bin/bash
set -eu
if [[ ${1:-} == --version ]]; then
  printf 'perf version fake-test\n'
  exit 0
fi
control=
output=
cgroup=
events=()
while (($#)); do
  case "$1" in
    --control) control=$2; shift 2 ;;
    -o) output=$2; shift 2 ;;
    -G) cgroup=$2; shift 2 ;;
    -e) events+=("$2"); shift 2 ;;
    *) shift ;;
  esac
done
pair=${control#fd:}
control_fd=${pair%,*}
ack_fd=${pair#*,}
finish() {
  : >"$output"
  value=100
  for event in "${events[@]}"; do
    printf '%s;;%s;%s;1000000;100.00;;\n' "$value" "$event" "$cgroup" >>"$output"
    value=$((value + 1))
  done
  exit 0
}
trap finish INT TERM
while IFS= read -r action <&"$control_fd"; do
  case "$action" in
    enable|disable) printf '\0ack\n' >&"$ack_fd" ;;
    *) printf 'unexpected\n' >&"$ack_fd" ;;
  esac
done
"#,
        )
        .unwrap();
        let mut permissions = fs::metadata(&fake_perf).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&fake_perf, permissions).unwrap();

        let mut collector = PerfCgroupHardwareProfiler::new(
            directory.path(),
            Some("2-3".to_owned()),
            fake_perf.to_string_lossy(),
            events(&["cycles", "custom-test-event"]),
            Duration::from_secs(2),
            0.95,
        )
        .unwrap();
        collector
            .attach(Path::new(
                "/sys/fs/cgroup/system.slice/docker-example.scope",
            ))
            .unwrap();
        let child_pid = libc::pid_t::try_from(collector.process.as_ref().unwrap().id()).unwrap();
        assert_eq!(unsafe { libc::getpgid(child_pid) }, child_pid);
        assert_ne!(child_pid, unsafe { libc::getpgrp() });
        let lifecycle_start = monotonic_ns();
        let enabled = collector.enable().unwrap();
        assert_eq!(enabled.action, "enable");
        collector.poll().unwrap();
        let lifecycle_end = monotonic_ns().max(lifecycle_start + 1);
        let report = collector.finish(lifecycle_start, lifecycle_end).unwrap();

        assert_eq!(report["schema_version"], HARDWARE_PERF_SCHEMA_VERSION);
        assert_eq!(report["perf_cgroup"], "/system.slice/docker-example.scope");
        assert_eq!(report["cpuset_cpus"], "2-3");
        assert_eq!(report["hybrid_pmu_counts_merged"], true);
        assert_eq!(report["instrumentation_valid"], true);
        assert_eq!(report["all_requested_metrics_valid"], true);
        assert_eq!(report["counters"][0]["value"], 100);
        assert_eq!(report["counters"][1]["namespace"], "custom-unclassified");
        assert_eq!(
            report["control_bounds"]["attach_ready"]["action"],
            "disable"
        );
        assert_eq!(report["control_bounds"]["enable"]["action"], "enable");
        assert_eq!(report["control_bounds"]["disable"]["action"], "disable");
        assert!(collector.stats_path().is_file());
        assert!(collector.stderr_path().is_file());
        assert_eq!(
            collector.provenance()["perf_version"],
            "perf version fake-test"
        );
        assert!(collector.stderr_text().is_empty());
        let mut encoded = serde_json::to_vec_pretty(&report).unwrap();
        encoded.push(b'\n');
        fs::write(directory.path().join("hardware_perf_report.json"), encoded).unwrap();
        let evidence =
            build_hardware_perf_evidence("run-test", "scope-test", &report, directory.path())
                .unwrap();
        assert!(evidence.valid);
        assert_eq!(evidence.stream_ids.len(), 2);
        assert_eq!(evidence.streams[0]["shape"], "window_aggregate");
        assert_eq!(evidence.streams[0]["records"][0]["value"], 100);
        assert_eq!(evidence.artifacts.len(), 3);
        assert!(evidence.artifacts.iter().all(|artifact| artifact["sha256"]
            .as_str()
            .unwrap()
            .len()
            == 64));
    }

    #[test]
    fn unavailable_perf_is_retained_as_null_invalid_counters() {
        let directory = tempdir().unwrap();
        let missing = directory.path().join("perf-does-not-exist");
        let mut run = HardwarePerfRun::new(
            directory.path(),
            Some("0".to_owned()),
            missing.to_string_lossy(),
            events(&["cycles", "custom-test-event"]),
            Duration::from_millis(100),
            0.95,
        )
        .unwrap();
        run.attach(Path::new(
            "/sys/fs/cgroup/system.slice/docker-example.scope",
        ))
        .unwrap();
        run.enable().unwrap();
        run.poll().unwrap();
        let report = run.finish(10, 20).unwrap();

        assert_eq!(
            report["schema_version"],
            "cgroup-perf-hardware-unavailable.v1"
        );
        assert_eq!(report["instrumentation_valid"], false);
        assert_eq!(report["all_requested_metrics_valid"], false);
        assert_eq!(report["counters"][0]["status"], "unavailable");
        assert_eq!(report["counters"][0]["value"], Value::Null);
        assert_eq!(report["counters"][0]["valid"], false);
        assert_eq!(report["counters"][1]["namespace"], "custom-unclassified");
        let reason = report["invalid_reasons"][0].as_str().unwrap();
        assert!(reason.contains("unable to start perf hardware profiler"));
        assert!(reason.contains("perf-does-not-exist"));
        assert!(directory.path().join("hardware_perf_stat.csv").is_file());
        assert!(directory.path().join("hardware_perf.stderr.txt").is_file());
        assert!(directory.path().join("hardware_perf_report.json").is_file());
        let evidence =
            build_hardware_perf_evidence("run-test", "scope-test", &report, directory.path())
                .unwrap();
        assert!(!evidence.valid);
        assert_eq!(evidence.streams[0]["shape"], "scalar");
        assert_eq!(evidence.streams[0]["records"][0]["value"], Value::Null);
        assert_eq!(evidence.artifacts.len(), 3);
    }
}
