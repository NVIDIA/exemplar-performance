// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native parsing and validation for cgroup-scoped `perf record` switch events.
//!
//! The collector integration lives with the Docker coordinator.  This module
//! owns the loss-aware decoder and the scheduler-concurrency math so those
//! contracts can be tested without privileges or a running container.

use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::{json, Map, Number, Value};
use std::collections::{btree_map::Entry, BTreeMap};
use std::error::Error;
use std::fmt;
use std::fs::{self, File};
use std::io;
use std::os::fd::RawFd;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::path::PathBuf;
use std::process::{Child, Command, ExitStatus, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const RECORDER_STOP_TIMEOUT: Duration = Duration::from_secs(10);
const CLEANUP_TERM_TIMEOUT: Duration = Duration::from_secs(5);
const CLEANUP_KILL_TIMEOUT: Duration = Duration::from_secs(5);
const CHILD_POLL_INTERVAL: Duration = Duration::from_millis(10);

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerPerfError(String);

impl SchedulerPerfError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for SchedulerPerfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for SchedulerPerfError {}

impl From<std::io::Error> for SchedulerPerfError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct SchedulerEvent {
    pub(crate) timestamp_ns: u64,
    pub(crate) cpu: u32,
    pub(crate) tid: i64,
    pub(crate) entering: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct NativePerfRecord {
    timestamp_ns: u64,
    cpu: u32,
    tid: i64,
    event: String,
    entering: Option<bool>,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct ParsedPerfEvents {
    pub(crate) events: Vec<SchedulerEvent>,
    pub(crate) lost_events: u64,
    pub(crate) anomalies: Vec<String>,
}

#[derive(Clone, Debug)]
pub(crate) struct PerfSchedulerOutput {
    pub(crate) report: Value,
    pub(crate) events: Vec<SchedulerEvent>,
}

#[derive(Debug)]
pub(crate) struct PerfSchedulerTracer {
    perf: String,
    cpuset_cpus: Option<String>,
    mmap_pages: Option<u64>,
    attach_timeout: Duration,
    root_tid: Option<u32>,
    perf_cgroup: Option<String>,
    data_path: PathBuf,
    decoded_path: PathBuf,
    stderr_path: PathBuf,
    child: Option<Child>,
    process_group_id: Option<libc::pid_t>,
    control_write_fd: Option<RawFd>,
    ack_read_fd: Option<RawFd>,
    ack_buffer: Vec<u8>,
}

trait PerfChildOperations {
    type Status;

    fn poll(&mut self) -> Result<Option<Self::Status>, SchedulerPerfError>;
    fn signal(&mut self, signal: libc::c_int) -> Result<(), SchedulerPerfError>;
    fn wait_bounded(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Self::Status>, SchedulerPerfError>;
}

struct RealPerfChild<'a> {
    child: &'a mut Child,
    process_group_id: libc::pid_t,
}

impl PerfChildOperations for RealPerfChild<'_> {
    type Status = ExitStatus;

    fn poll(&mut self) -> Result<Option<Self::Status>, SchedulerPerfError> {
        self.child.try_wait().map_err(|error| {
            SchedulerPerfError::new(format!(
                "unable to wait for perf scheduler recorder: {error}"
            ))
        })
    }

    fn signal(&mut self, signal: libc::c_int) -> Result<(), SchedulerPerfError> {
        signal_perf_process_group(self.process_group_id, signal)
    }

    fn wait_bounded(
        &mut self,
        timeout: Duration,
    ) -> Result<Option<Self::Status>, SchedulerPerfError> {
        let deadline = Instant::now() + timeout;
        loop {
            if let Some(status) = self.poll()? {
                return Ok(Some(status));
            }
            let now = Instant::now();
            if now >= deadline {
                return Ok(None);
            }
            thread::sleep(CHILD_POLL_INTERVAL.min(deadline.saturating_duration_since(now)));
        }
    }
}

#[derive(Debug)]
struct PerfChildShutdown<Status> {
    status: Option<Status>,
    errors: Vec<String>,
}

fn shutdown_perf_child<Operations: PerfChildOperations>(
    operations: &mut Operations,
    graceful_signal: libc::c_int,
    graceful_timeout: Duration,
    kill_timeout: Duration,
) -> PerfChildShutdown<Operations::Status> {
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
                "timed out after {:.3}s waiting for perf scheduler recorder to stop after signal {graceful_signal}",
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
                "perf scheduler recorder remains unreaped after SIGKILL and a {:.3}s bounded wait",
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
            .push("perf scheduler recorder absence could not be verified after cleanup".to_owned());
    }
    PerfChildShutdown { status, errors }
}

fn combine_cleanup_error(
    primary: SchedulerPerfError,
    cleanup: Result<(), SchedulerPerfError>,
) -> SchedulerPerfError {
    match cleanup {
        Ok(()) => primary,
        Err(cleanup) => SchedulerPerfError::new(format!(
            "{primary}; perf scheduler cleanup also failed: {cleanup}"
        )),
    }
}

#[derive(Debug)]
struct PerfProcessGroupShutdown {
    verified_absent: bool,
    errors: Vec<String>,
}

fn validate_perf_process_group(process_group_id: libc::pid_t) -> Result<(), SchedulerPerfError> {
    if process_group_id <= 0 {
        return Err(SchedulerPerfError::new(
            "perf scheduler process-group ID must be positive",
        ));
    }
    let controller_process_group = unsafe { libc::getpgrp() };
    if process_group_id == controller_process_group {
        return Err(SchedulerPerfError::new(format!(
            "refusing to signal controller process group {process_group_id} while cleaning perf scheduler recorder"
        )));
    }
    Ok(())
}

fn signal_perf_process_group(
    process_group_id: libc::pid_t,
    signal: libc::c_int,
) -> Result<(), SchedulerPerfError> {
    validate_perf_process_group(process_group_id)?;
    if unsafe { libc::kill(-process_group_id, signal) } == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(SchedulerPerfError::new(format!(
            "unable to signal perf scheduler process group {process_group_id} with signal {signal}: {error}"
        )))
    }
}

fn perf_process_group_exists(process_group_id: libc::pid_t) -> Result<bool, SchedulerPerfError> {
    validate_perf_process_group(process_group_id)?;
    if unsafe { libc::kill(-process_group_id, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(SchedulerPerfError::new(format!(
            "unable to inspect perf scheduler process group {process_group_id}: {error}"
        ))),
    }
}

fn wait_for_perf_process_group_absence<Exists>(
    process_group_id: libc::pid_t,
    timeout: Duration,
    process_group_exists: &mut Exists,
) -> Result<bool, SchedulerPerfError>
where
    Exists: FnMut(libc::pid_t) -> Result<bool, SchedulerPerfError>,
{
    let deadline = Instant::now() + timeout;
    loop {
        if !process_group_exists(process_group_id)? {
            return Ok(true);
        }
        let now = Instant::now();
        if now >= deadline {
            return Ok(false);
        }
        thread::sleep(CHILD_POLL_INTERVAL.min(deadline.saturating_duration_since(now)));
    }
}

fn cleanup_perf_process_group_with<Exists, Signal>(
    process_group_id: libc::pid_t,
    term_timeout: Duration,
    kill_timeout: Duration,
    mut process_group_exists: Exists,
    mut signal_process_group: Signal,
) -> PerfProcessGroupShutdown
where
    Exists: FnMut(libc::pid_t) -> Result<bool, SchedulerPerfError>,
    Signal: FnMut(libc::pid_t, libc::c_int) -> Result<(), SchedulerPerfError>,
{
    let mut errors = Vec::new();
    match process_group_exists(process_group_id) {
        Ok(false) => {
            return PerfProcessGroupShutdown {
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
    match wait_for_perf_process_group_absence(
        process_group_id,
        term_timeout,
        &mut process_group_exists,
    ) {
        Ok(true) => {
            return PerfProcessGroupShutdown {
                verified_absent: true,
                errors,
            }
        }
        Ok(false) => errors.push(format!(
            "timed out after {:.3}s waiting for perf scheduler process group {process_group_id} to exit after SIGTERM",
            term_timeout.as_secs_f64()
        )),
        Err(error) => errors.push(error.to_string()),
    }

    if let Err(error) = signal_process_group(process_group_id, libc::SIGKILL) {
        errors.push(error.to_string());
    }
    let verified_absent = match wait_for_perf_process_group_absence(
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
            "perf scheduler process group {process_group_id} remains observable after SIGKILL and a {:.3}s bounded wait",
            kill_timeout.as_secs_f64()
        ));
    }
    PerfProcessGroupShutdown {
        verified_absent,
        errors,
    }
}

fn cleanup_perf_process_group(
    process_group_id: libc::pid_t,
    term_timeout: Duration,
    kill_timeout: Duration,
) -> PerfProcessGroupShutdown {
    cleanup_perf_process_group_with(
        process_group_id,
        term_timeout,
        kill_timeout,
        perf_process_group_exists,
        signal_perf_process_group,
    )
}

impl PerfSchedulerTracer {
    pub(crate) fn new(
        output_dir: &Path,
        cpuset_cpus: Option<&str>,
        perf: &str,
        attach_timeout: Duration,
        mmap_pages: Option<u64>,
    ) -> Result<Self, SchedulerPerfError> {
        if perf.trim().is_empty() {
            return Err(SchedulerPerfError::new(
                "perf scheduler executable must be nonempty",
            ));
        }
        if attach_timeout.is_zero() {
            return Err(SchedulerPerfError::new(
                "perf scheduler attach timeout must be positive",
            ));
        }
        if mmap_pages == Some(0) {
            return Err(SchedulerPerfError::new(
                "perf scheduler mmap page count must be positive",
            ));
        }
        Ok(Self {
            perf: perf.to_owned(),
            cpuset_cpus: cpuset_cpus.map(str::to_owned),
            mmap_pages,
            attach_timeout,
            root_tid: None,
            perf_cgroup: None,
            data_path: output_dir.join("scheduler_perf.data"),
            decoded_path: output_dir.join("scheduler_perf_switch_events.txt"),
            stderr_path: output_dir.join("scheduler_perf.stderr.txt"),
            child: None,
            process_group_id: None,
            control_write_fd: None,
            ack_read_fd: None,
            ack_buffer: Vec::new(),
        })
    }

    pub(crate) fn start(
        &mut self,
        root_tid: u32,
        cgroup_path: &Path,
    ) -> Result<(), SchedulerPerfError> {
        if self.child.is_some() {
            return Err(SchedulerPerfError::new(
                "perf scheduler tracer has already been started",
            ));
        }
        let relative = cgroup_path.strip_prefix("/sys/fs/cgroup").map_err(|_| {
            SchedulerPerfError::new(format!(
                "perf scheduler cgroup is outside /sys/fs/cgroup: {}",
                cgroup_path.display()
            ))
        })?;
        let perf_cgroup = if relative.as_os_str().is_empty() {
            "/".to_owned()
        } else {
            format!("/{}", relative.to_string_lossy())
        };
        let (control_read_fd, control_write_fd) = pipe_pair()?;
        let (ack_read_fd, ack_write_fd) = match pipe_pair() {
            Ok(pair) => pair,
            Err(error) => {
                close_fd(control_read_fd);
                close_fd(control_write_fd);
                return Err(error);
            }
        };
        if let Err(error) = set_close_on_exec(control_write_fd, true)
            .and_then(|()| set_close_on_exec(ack_read_fd, true))
        {
            for fd in [control_read_fd, control_write_fd, ack_read_fd, ack_write_fd] {
                close_fd(fd);
            }
            return Err(error);
        }
        let mut command = Command::new(&self.perf);
        command.args([
            "record",
            "--switch-events",
            "--sample-cpu",
            "--clockid",
            "mono",
            "-T",
            "-B",
            "-N",
            "-D",
            "-1",
            "--control",
            &format!("fd:{control_read_fd},{ack_write_fd}"),
            "--synth",
            "no",
        ]);
        if let Some(pages) = self.mmap_pages {
            command.args(["-m", &pages.to_string()]);
        }
        command.args([
            "-o",
            self.data_path.to_string_lossy().as_ref(),
            "-e",
            "dummy",
            "-G",
            &perf_cgroup,
            "-a",
        ]);
        if let Some(cpuset) = &self.cpuset_cpus {
            command.args(["-C", cpuset]);
        }
        let stderr = match File::create(&self.stderr_path) {
            Ok(file) => file,
            Err(error) => {
                for fd in [control_read_fd, control_write_fd, ack_read_fd, ack_write_fd] {
                    close_fd(fd);
                }
                return Err(error.into());
            }
        };
        let spawned = command
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::from(stderr))
            .process_group(0)
            .spawn();
        close_fd(control_read_fd);
        close_fd(ack_write_fd);
        let child = match spawned {
            Ok(child) => child,
            Err(error) => {
                close_fd(control_write_fd);
                close_fd(ack_read_fd);
                return Err(SchedulerPerfError::new(format!(
                    "unable to start perf scheduler recorder: {error}"
                )));
            }
        };
        self.root_tid = Some(root_tid);
        self.perf_cgroup = Some(perf_cgroup);
        self.process_group_id = Some(
            libc::pid_t::try_from(child.id())
                .expect("Linux child process IDs are representable as pid_t"),
        );
        self.child = Some(child);
        self.control_write_fd = Some(control_write_fd);
        self.ack_read_fd = Some(ack_read_fd);
        if let Err(error) = self.control("enable") {
            let cleanup = self.close();
            return Err(combine_cleanup_error(error, cleanup));
        }
        Ok(())
    }

    pub(crate) fn poll(&mut self) -> Result<(), SchedulerPerfError> {
        let child = self
            .child
            .as_mut()
            .ok_or_else(|| SchedulerPerfError::new("perf scheduler tracer has not been started"))?;
        if let Some(status) = child.try_wait()? {
            return Err(SchedulerPerfError::new(format!(
                "perf scheduler recorder exited before measurement completed with {status}: {}",
                self.stderr_text()
            )));
        }
        Ok(())
    }

    pub(crate) fn finish(
        &mut self,
        window_start_ns: u64,
        window_end_ns: u64,
    ) -> Result<PerfSchedulerOutput, SchedulerPerfError> {
        let result = self.finish_inner(window_start_ns, window_end_ns);
        match result {
            Ok(output) => Ok(output),
            Err(primary) => {
                let cleanup = self.close();
                Err(combine_cleanup_error(primary, cleanup))
            }
        }
    }

    fn finish_inner(
        &mut self,
        window_start_ns: u64,
        window_end_ns: u64,
    ) -> Result<PerfSchedulerOutput, SchedulerPerfError> {
        if self.root_tid.is_none() || self.child.is_none() {
            return Err(SchedulerPerfError::new(
                "perf scheduler tracer has not been started",
            ));
        }
        self.control("disable")?;
        self.stop_recorder()?;
        self.close_control_fds();
        if !self.data_path.is_file() {
            return Err(SchedulerPerfError::new(format!(
                "perf scheduler recorder produced no data file: {}",
                self.data_path.display()
            )));
        }
        let decoded_output = File::create(&self.decoded_path)?;
        let decoded = Command::new(&self.perf)
            .args([
                "script",
                "--ns",
                "-i",
                self.data_path.to_string_lossy().as_ref(),
                "--show-switch-events",
                "--show-task-events",
                "--show-lost-events",
                "-F",
                "comm,pid,tid,time,cpu,event",
            ])
            .stdin(Stdio::null())
            .stdout(Stdio::from(decoded_output))
            .stderr(Stdio::piped())
            .output()
            .map_err(|error| {
                SchedulerPerfError::new(format!("unable to start perf scheduler decoder: {error}"))
            })?;
        if !decoded.status.success() {
            return Err(SchedulerPerfError::new(format!(
                "unable to decode perf scheduler records: {}",
                String::from_utf8_lossy(&decoded.stderr).trim()
            )));
        }
        let decoded_text = String::from_utf8_lossy(&fs::read(&self.decoded_path)?).into_owned();
        let parsed = parse_perf_script_switch_events(&decoded_text)?;
        if parsed.events.is_empty() {
            return Err(SchedulerPerfError::new(
                "perf scheduler record produced no context-switch transitions; do not interpret this capture as zero concurrency",
            ));
        }
        let mut report = summarize_scheduler_events(
            &parsed.events,
            window_start_ns,
            window_end_ns,
            parsed.lost_events,
            "perf-record-cgroup-native-switch-events",
            "native perf CPU-wide context-switch records restricted to the measured container cgroup",
        )?;
        if !parsed.anomalies.is_empty() {
            let object = report.as_object_mut().expect("scheduler report object");
            let existing = object
                .get("anomalies")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default();
            object.insert(
                "anomalies".to_owned(),
                Value::Array(
                    parsed
                        .anomalies
                        .iter()
                        .cloned()
                        .map(Value::String)
                        .chain(existing)
                        .collect(),
                ),
            );
            object.insert("trace_complete".to_owned(), Value::Bool(false));
        }
        let object = report.as_object_mut().expect("scheduler report object");
        object.insert("root_tid".to_owned(), json!(self.root_tid));
        object.insert("backend".to_owned(), Value::String("perf".to_owned()));
        object.insert("perf_clockid".to_owned(), Value::String("mono".to_owned()));
        object.insert("perf_cgroup".to_owned(), json!(self.perf_cgroup));
        object.insert("perf_cpuset_cpus".to_owned(), json!(self.cpuset_cpus));
        object.insert("perf_mmap_pages".to_owned(), json!(self.mmap_pages));
        object.insert("perf_data_path".to_owned(), json!(self.data_path));
        object.insert(
            "perf_decoded_switch_events_path".to_owned(),
            json!(self.decoded_path),
        );
        object.insert("perf_stderr_path".to_owned(), json!(self.stderr_path));
        Ok(PerfSchedulerOutput {
            report,
            events: parsed.events,
        })
    }

    fn control(&mut self, action: &str) -> Result<(), SchedulerPerfError> {
        self.poll()?;
        let write_fd = self
            .control_write_fd
            .ok_or_else(|| SchedulerPerfError::new("perf scheduler tracer has no control pipe"))?;
        write_all_fd(write_fd, format!("{action}\n").as_bytes()).map_err(|error| {
            SchedulerPerfError::new(format!(
                "unable to send perf scheduler {action} command: {error}"
            ))
        })?;
        let deadline = Instant::now() + self.attach_timeout;
        loop {
            if let Some(newline) = self.ack_buffer.iter().position(|byte| *byte == b'\n') {
                let mut response = self.ack_buffer.drain(..=newline).collect::<Vec<_>>();
                response.pop();
                let response = response
                    .iter()
                    .copied()
                    .filter(|byte| !matches!(*byte, 0 | b' ' | b'\t' | b'\r'))
                    .collect::<Vec<_>>();
                if response != b"ack" {
                    return Err(SchedulerPerfError::new(format!(
                        "unexpected perf scheduler control response: {:?}",
                        String::from_utf8_lossy(&response)
                    )));
                }
                return Ok(());
            }
            self.poll().map_err(|error| {
                SchedulerPerfError::new(format!(
                    "perf scheduler recorder exited before {action} acknowledgement: {error}"
                ))
            })?;
            let now = Instant::now();
            if now >= deadline {
                return Err(SchedulerPerfError::new(format!(
                    "timed out waiting for perf scheduler {action} acknowledgement"
                )));
            }
            let remaining = deadline.saturating_duration_since(now);
            let timeout_ms = i32::try_from(remaining.min(Duration::from_millis(100)).as_millis())
                .unwrap_or(100)
                .max(1);
            let read_fd = self.ack_read_fd.ok_or_else(|| {
                SchedulerPerfError::new("perf scheduler tracer has no acknowledgement pipe")
            })?;
            let mut descriptor = libc::pollfd {
                fd: read_fd,
                events: libc::POLLIN,
                revents: 0,
            };
            let ready = loop {
                let result = unsafe { libc::poll(&mut descriptor, 1, timeout_ms) };
                if result < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted {
                    continue;
                }
                if result < 0 {
                    return Err(io::Error::last_os_error().into());
                }
                break result;
            };
            if ready > 0 && descriptor.revents & libc::POLLIN != 0 {
                let mut buffer = [0_u8; 4096];
                let count = loop {
                    let result =
                        unsafe { libc::read(read_fd, buffer.as_mut_ptr().cast(), buffer.len()) };
                    if result < 0 && io::Error::last_os_error().kind() == io::ErrorKind::Interrupted
                    {
                        continue;
                    }
                    if result < 0 {
                        return Err(io::Error::last_os_error().into());
                    }
                    break result as usize;
                };
                if count == 0 {
                    return Err(SchedulerPerfError::new(format!(
                        "perf scheduler control pipe closed before {action} acknowledgement"
                    )));
                }
                self.ack_buffer.extend_from_slice(&buffer[..count]);
            }
        }
    }

    fn shutdown_recorder_scope(
        &mut self,
        graceful_signal: libc::c_int,
        graceful_timeout: Duration,
    ) -> PerfChildShutdown<ExitStatus> {
        let mut outcome = if let Some(child) = self.child.as_mut() {
            let process_group_id = self
                .process_group_id
                .expect("started perf scheduler child has an owned process group");
            shutdown_perf_child(
                &mut RealPerfChild {
                    child,
                    process_group_id,
                },
                graceful_signal,
                graceful_timeout,
                CLEANUP_KILL_TIMEOUT,
            )
        } else {
            PerfChildShutdown {
                status: None,
                errors: Vec::new(),
            }
        };
        if outcome.status.is_some() {
            self.child = None;
        }
        if self.child.is_none() {
            if let Some(process_group_id) = self.process_group_id {
                let group = cleanup_perf_process_group(
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

    fn stop_recorder(&mut self) -> Result<(), SchedulerPerfError> {
        let outcome = self.shutdown_recorder_scope(libc::SIGINT, RECORDER_STOP_TIMEOUT);
        let status = outcome.status;
        if !outcome.errors.is_empty() {
            return Err(SchedulerPerfError::new(outcome.errors.join("; ")));
        }
        let status = status.ok_or_else(|| {
            SchedulerPerfError::new(
                "perf scheduler recorder cleanup returned without verified child absence",
            )
        })?;
        if !status.success() && status.signal() != Some(libc::SIGINT) {
            return Err(SchedulerPerfError::new(format!(
                "perf scheduler recorder failed while stopping with {status}: {}",
                self.stderr_text()
            )));
        }
        Ok(())
    }

    fn stderr_text(&self) -> String {
        fs::read(&self.stderr_path)
            .map(|bytes| String::from_utf8_lossy(&bytes).trim().to_owned())
            .unwrap_or_default()
    }

    fn close_control_fds(&mut self) {
        if let Some(fd) = self.control_write_fd.take() {
            close_fd(fd);
        }
        if let Some(fd) = self.ack_read_fd.take() {
            close_fd(fd);
        }
    }

    pub(crate) fn close(&mut self) -> Result<(), SchedulerPerfError> {
        self.close_control_fds();
        let outcome = self.shutdown_recorder_scope(libc::SIGTERM, CLEANUP_TERM_TIMEOUT);
        if outcome.errors.is_empty() {
            Ok(())
        } else {
            Err(SchedulerPerfError::new(outcome.errors.join("; ")))
        }
    }
}

impl Drop for PerfSchedulerTracer {
    fn drop(&mut self) {
        let _ = self.close();
    }
}

fn pipe_pair() -> Result<(RawFd, RawFd), SchedulerPerfError> {
    let mut fds = [-1; 2];
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok((fds[0], fds[1]))
}

fn set_close_on_exec(fd: RawFd, enabled: bool) -> Result<(), SchedulerPerfError> {
    let flags = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    if flags < 0 {
        return Err(io::Error::last_os_error().into());
    }
    let flags = if enabled {
        flags | libc::FD_CLOEXEC
    } else {
        flags & !libc::FD_CLOEXEC
    };
    if unsafe { libc::fcntl(fd, libc::F_SETFD, flags) } < 0 {
        return Err(io::Error::last_os_error().into());
    }
    Ok(())
}

fn write_all_fd(fd: RawFd, mut bytes: &[u8]) -> io::Result<()> {
    while !bytes.is_empty() {
        let written = unsafe { libc::write(fd, bytes.as_ptr().cast(), bytes.len()) };
        if written < 0 {
            let error = io::Error::last_os_error();
            if error.kind() == io::ErrorKind::Interrupted {
                continue;
            }
            return Err(error);
        }
        if written == 0 {
            return Err(io::Error::new(
                io::ErrorKind::WriteZero,
                "perf control pipe accepted zero bytes",
            ));
        }
        bytes = &bytes[written as usize..];
    }
    Ok(())
}

fn close_fd(fd: RawFd) {
    unsafe {
        libc::close(fd);
    }
}

fn timestamp_ns(value: &str) -> Result<u64, SchedulerPerfError> {
    let (seconds, fraction) = value.split_once('.').ok_or_else(|| {
        SchedulerPerfError::new(format!("invalid perf script timestamp: {value:?}"))
    })?;
    if seconds.is_empty()
        || fraction.is_empty()
        || !seconds.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(SchedulerPerfError::new(format!(
            "invalid perf script timestamp: {value:?}"
        )));
    }
    let seconds = seconds.parse::<u64>().map_err(|_| {
        SchedulerPerfError::new(format!("invalid perf script timestamp: {value:?}"))
    })?;
    let mut nanos = 0_u64;
    for byte in fraction.bytes().take(9) {
        nanos = nanos
            .checked_mul(10)
            .and_then(|current| current.checked_add(u64::from(byte - b'0')))
            .ok_or_else(|| SchedulerPerfError::new("perf timestamp overflow"))?;
    }
    for _ in fraction.len().min(9)..9 {
        nanos = nanos
            .checked_mul(10)
            .ok_or_else(|| SchedulerPerfError::new("perf timestamp overflow"))?;
    }
    seconds
        .checked_mul(1_000_000_000)
        .and_then(|current| current.checked_add(nanos))
        .ok_or_else(|| SchedulerPerfError::new("perf timestamp overflow"))
}

fn parse_native_records(text: &str) -> Result<(Vec<NativePerfRecord>, u64), SchedulerPerfError> {
    let record_pattern = Regex::new(
        r"^\s*.*?\s+(?P<pid>-?\d+)/(?P<tid>-?\d+)\s+\[(?P<cpu>\d+)\]\s+(?P<timestamp>[0-9]+\.[0-9]+):\s+PERF_RECORD_(?P<event>SWITCH(?:_CPU_WIDE)?|EXIT)\b(?P<details>.*)$",
    )
    .expect("static perf record regex");
    let direction_pattern =
        Regex::new(r"^\s+(?P<direction>IN|OUT)\b").expect("static perf switch-direction regex");
    let lost_pattern = Regex::new(r"(?i)\blost(?:\s+events?)?\s*[:=]?\s*(?P<count>\d+)")
        .expect("static perf lost-event regex");

    let mut records = Vec::new();
    let mut lost_events = 0_u64;
    for line in text.lines() {
        if line.contains("PERF_RECORD_LOST") {
            let count = lost_pattern
                .captures(line)
                .and_then(|captures| captures.name("count"))
                .map(|value| value.as_str().parse::<u64>())
                .transpose()
                .map_err(|_| SchedulerPerfError::new("perf lost-event count overflow"))?
                .unwrap_or(1);
            lost_events = lost_events
                .checked_add(count)
                .ok_or_else(|| SchedulerPerfError::new("perf lost-event count overflow"))?;
        }
        let Some(captures) = record_pattern.captures(line) else {
            continue;
        };
        let event = captures["event"].to_owned();
        let entering = if event.starts_with("SWITCH") {
            let direction = direction_pattern
                .captures(&captures["details"])
                .ok_or_else(|| {
                    SchedulerPerfError::new(format!(
                        "unable to decode perf switch direction: {line:?}"
                    ))
                })?;
            Some(&direction["direction"] == "IN")
        } else {
            None
        };
        records.push(NativePerfRecord {
            timestamp_ns: timestamp_ns(&captures["timestamp"])?,
            cpu: captures["cpu"]
                .parse::<u32>()
                .map_err(|_| SchedulerPerfError::new("perf logical CPU ID overflow"))?,
            tid: captures["tid"]
                .parse::<i64>()
                .map_err(|_| SchedulerPerfError::new("perf TID overflow"))?,
            event,
            entering,
        });
    }
    Ok((records, lost_events))
}

pub(crate) fn parse_perf_script_switch_events(
    text: &str,
) -> Result<ParsedPerfEvents, SchedulerPerfError> {
    let (mut records, lost_events) = parse_native_records(text)?;
    records.sort_by_key(|record| (record.timestamp_ns, record.cpu));
    let mut active_tid_by_cpu = BTreeMap::<u32, i64>::new();
    let mut active_cpu_by_tid = BTreeMap::<i64, u32>::new();
    let mut events = Vec::new();
    let mut anomalies = Vec::new();

    for record in records {
        if record.event.starts_with("SWITCH") && record.entering == Some(true) {
            let active_tid = active_tid_by_cpu.get(&record.cpu).copied();
            if let Some(active_tid) = active_tid {
                anomalies.push(format!(
                    "logical CPU {} switched into TID {} while recorded TID {} was active",
                    record.cpu, record.tid, active_tid
                ));
            }
            let previous_cpu = active_cpu_by_tid.get(&record.tid).copied();
            if let Some(previous_cpu) = previous_cpu {
                anomalies.push(format!(
                    "TID {} switched into logical CPU {} while active on logical CPU {}",
                    record.tid, record.cpu, previous_cpu
                ));
            }
            if active_tid.is_none() && previous_cpu.is_none() {
                active_tid_by_cpu.insert(record.cpu, record.tid);
                active_cpu_by_tid.insert(record.tid, record.cpu);
            }
            events.push(SchedulerEvent {
                timestamp_ns: record.timestamp_ns,
                cpu: record.cpu,
                tid: record.tid,
                entering: true,
            });
            continue;
        }
        if record.event.starts_with("SWITCH") {
            let active_tid = active_tid_by_cpu.get(&record.cpu).copied();
            let leaving_tid = match (record.tid, active_tid) {
                (-1, Some(active_tid)) => active_tid,
                (record_tid, active_tid) => {
                    if active_tid != Some(record_tid) {
                        anomalies.push(format!(
                            "logical CPU {} switched out TID {} while recorded active TID was {}",
                            record.cpu,
                            record_tid,
                            active_tid.map_or_else(|| "None".to_owned(), |value| value.to_string())
                        ));
                    }
                    record_tid
                }
            };
            if active_tid == Some(leaving_tid) {
                active_tid_by_cpu.remove(&record.cpu);
                active_cpu_by_tid.remove(&leaving_tid);
            }
            events.push(SchedulerEvent {
                timestamp_ns: record.timestamp_ns,
                cpu: record.cpu,
                tid: leaving_tid,
                entering: false,
            });
        }
        // PERF_RECORD_EXIT is useful retained diagnostic text, but the
        // CPU-wide SWITCH OUT is the authoritative occupancy transition.
    }

    Ok(ParsedPerfEvents {
        events,
        lost_events,
        anomalies,
    })
}

fn rounded(value: f64, digits: i32) -> f64 {
    let factor = 10_f64.powi(digits);
    (value * factor).round() / factor
}

fn seconds(nanoseconds: u64) -> f64 {
    rounded(nanoseconds as f64 / 1_000_000_000.0, 9)
}

fn number(value: f64) -> Value {
    Value::Number(Number::from_f64(value).expect("finite scheduler value"))
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn summarize_scheduler_events(
    events: &[SchedulerEvent],
    window_start_ns: u64,
    window_end_ns: u64,
    lost_events: u64,
    method: &str,
    scope: &str,
) -> Result<Value, SchedulerPerfError> {
    if window_end_ns <= window_start_ns {
        return Err(SchedulerPerfError::new(
            "scheduler measurement window must have positive duration",
        ));
    }
    let mut ordered = events.to_vec();
    ordered.sort_by_key(|event| (event.timestamp_ns, !event.entering, event.cpu, event.tid));
    let selected = ordered
        .iter()
        .filter(|event| {
            window_start_ns <= event.timestamp_ns && event.timestamp_ns <= window_end_ns
        })
        .cloned()
        .collect::<Vec<_>>();
    let ignored_before = ordered
        .iter()
        .filter(|event| event.timestamp_ns < window_start_ns)
        .count() as u64;
    let ignored_after = ordered
        .iter()
        .filter(|event| event.timestamp_ns > window_end_ns)
        .count() as u64;

    let mut active_cpu_started = BTreeMap::<u32, u64>::new();
    for event in ordered
        .iter()
        .filter(|event| event.timestamp_ns < window_start_ns)
    {
        if event.entering {
            active_cpu_started.insert(event.cpu, window_start_ns);
        } else {
            active_cpu_started.remove(&event.cpu);
        }
    }
    let active_at_window_start = active_cpu_started.keys().copied().collect::<Vec<_>>();
    let mut runtime_ns_by_cpu = BTreeMap::<u32, u64>::new();
    let mut duration_ns_by_concurrency = BTreeMap::<usize, u64>::new();
    let mut anomalies = Vec::new();
    let mut peak = 0_usize;
    let mut cursor_ns = window_start_ns;
    let mut index = 0_usize;
    while index < selected.len() {
        let timestamp = selected[index].timestamp_ns;
        let mut end = index + 1;
        while end < selected.len() && selected[end].timestamp_ns == timestamp {
            end += 1;
        }
        let interval_ns = timestamp.saturating_sub(cursor_ns);
        if interval_ns > 0 {
            let running = active_cpu_started.len();
            *duration_ns_by_concurrency.entry(running).or_default() += interval_ns;
            peak = peak.max(running);
        }
        for event in selected[index..end].iter().filter(|event| !event.entering) {
            if let Some(started_ns) = active_cpu_started.remove(&event.cpu) {
                *runtime_ns_by_cpu.entry(event.cpu).or_default() +=
                    event.timestamp_ns.saturating_sub(started_ns);
            } else {
                anomalies.push(format!(
                    "logical CPU {} left the measured scope without an active interval",
                    event.cpu
                ));
            }
        }
        for event in selected[index..end].iter().filter(|event| event.entering) {
            match active_cpu_started.entry(event.cpu) {
                Entry::Occupied(_) => anomalies.push(format!(
                    "logical CPU {} entered the measured scope while already active",
                    event.cpu
                )),
                Entry::Vacant(entry) => {
                    entry.insert(event.timestamp_ns);
                }
            }
        }
        cursor_ns = timestamp;
        index = end;
    }
    let tail_ns = window_end_ns.saturating_sub(cursor_ns);
    if tail_ns > 0 {
        let running = active_cpu_started.len();
        *duration_ns_by_concurrency.entry(running).or_default() += tail_ns;
        peak = peak.max(running);
    }
    if !active_cpu_started.is_empty() {
        for (cpu, started_ns) in &active_cpu_started {
            *runtime_ns_by_cpu.entry(*cpu).or_default() +=
                window_end_ns.saturating_sub(*started_ns);
        }
        anomalies.push(format!(
            "measurement window ended while target tasks were still scheduled on {}",
            active_cpu_started
                .keys()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        ));
    }

    let window_ns = window_end_ns - window_start_ns;
    let scheduled_runtime_ns =
        duration_ns_by_concurrency
            .iter()
            .try_fold(0_u64, |total, (concurrency, duration)| {
                let concurrency = u64::try_from(*concurrency)
                    .map_err(|_| SchedulerPerfError::new("scheduler concurrency overflow"))?;
                total
                    .checked_add(duration.checked_mul(concurrency).ok_or_else(|| {
                        SchedulerPerfError::new("scheduler runtime integral overflow")
                    })?)
                    .ok_or_else(|| SchedulerPerfError::new("scheduler runtime integral overflow"))
            })?;
    let per_cpu_runtime_ns = runtime_ns_by_cpu.values().try_fold(0_u64, |total, value| {
        total
            .checked_add(*value)
            .ok_or_else(|| SchedulerPerfError::new("per-CPU scheduler runtime overflow"))
    })?;
    if scheduled_runtime_ns != per_cpu_runtime_ns {
        anomalies.push(format!(
            "scheduler concurrency integral differs from per-CPU runtime total ({} ns versus {} ns)",
            scheduled_runtime_ns, per_cpu_runtime_ns
        ));
    }

    let duration_sec = duration_ns_by_concurrency
        .iter()
        .map(|(concurrency, duration)| (concurrency.to_string(), number(seconds(*duration))))
        .collect::<Map<_, _>>();
    let duration_fraction = duration_ns_by_concurrency
        .iter()
        .map(|(concurrency, duration)| {
            (
                concurrency.to_string(),
                number(rounded(*duration as f64 / window_ns as f64, 9)),
            )
        })
        .collect::<Map<_, _>>();
    let runtime_sec_by_cpu = runtime_ns_by_cpu
        .iter()
        .map(|(cpu, duration)| (cpu.to_string(), number(seconds(*duration))))
        .collect::<Map<_, _>>();
    let trace_complete = lost_events == 0 && anomalies.is_empty();

    Ok(json!({
        "method": method,
        "scope": scope,
        "window_wall_sec": seconds(window_ns),
        "events_observed": selected.len(),
        "events_ignored_before_window": ignored_before,
        "events_ignored_after_window": ignored_after,
        "logical_cpus_active_at_window_start": active_at_window_start,
        "lost_perf_events": lost_events,
        "scheduled_runtime_sec": seconds(scheduled_runtime_ns),
        "average_logical_cores_used": rounded(scheduled_runtime_ns as f64 / window_ns as f64, 9),
        "peak_simultaneous_logical_cores": peak,
        "time_sec_by_simultaneous_logical_cores": duration_sec,
        "fraction_of_window_by_simultaneous_logical_cores": duration_fraction,
        "runtime_sec_by_logical_cpu": runtime_sec_by_cpu,
        "anomalies": anomalies,
        "trace_complete": trace_complete,
        "meaning": "Time-integrated simultaneous scheduled execution of the profiled measured container scope on logical CPUs; this is not physical-core use.",
    }))
}

pub(crate) fn add_cgroup_validation(
    report: &Value,
    cgroup_usage_sec: f64,
    tolerance_fraction: f64,
    tolerance_sec: f64,
) -> Result<Value, SchedulerPerfError> {
    add_cgroup_validation_with_event_tolerance(
        report,
        cgroup_usage_sec,
        tolerance_fraction,
        tolerance_sec,
        0.0,
    )
}

pub(crate) fn add_cgroup_validation_with_event_tolerance(
    report: &Value,
    cgroup_usage_sec: f64,
    tolerance_fraction: f64,
    tolerance_sec: f64,
    per_observed_event_tolerance_sec: f64,
) -> Result<Value, SchedulerPerfError> {
    if !cgroup_usage_sec.is_finite()
        || cgroup_usage_sec < 0.0
        || !tolerance_fraction.is_finite()
        || tolerance_fraction < 0.0
        || !tolerance_sec.is_finite()
        || tolerance_sec < 0.0
        || !per_observed_event_tolerance_sec.is_finite()
        || per_observed_event_tolerance_sec < 0.0
    {
        return Err(SchedulerPerfError::new(
            "scheduler cgroup validation inputs must be finite and nonnegative",
        ));
    }
    let scheduler_usage_sec = report
        .get("scheduled_runtime_sec")
        .and_then(Value::as_f64)
        .ok_or_else(|| SchedulerPerfError::new("scheduler report has no runtime integral"))?;
    let trace_complete = report
        .get("trace_complete")
        .and_then(Value::as_bool)
        .ok_or_else(|| SchedulerPerfError::new("scheduler report has no completeness gate"))?;
    let delta_sec = scheduler_usage_sec - cgroup_usage_sec;
    let allowed = tolerance_sec.max(cgroup_usage_sec.abs() * tolerance_fraction);
    let event_count = report
        .get("events_observed")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ignored_before = report
        .get("events_ignored_before_window")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let ignored_after = report
        .get("events_ignored_after_window")
        .and_then(Value::as_u64)
        .unwrap_or(0);
    let raw_event_count = event_count
        .checked_add(ignored_before)
        .and_then(|count| count.checked_add(ignored_after))
        .ok_or_else(|| SchedulerPerfError::new("scheduler raw event count overflow"))?;
    let event_coverage_valid = cgroup_usage_sec == 0.0 || raw_event_count > 0;
    let event_scaled_tolerance_sec = event_count as f64 * per_observed_event_tolerance_sec;
    let within_tolerance = -allowed <= delta_sec && delta_sec <= allowed;
    let mut result = report
        .as_object()
        .cloned()
        .ok_or_else(|| SchedulerPerfError::new("scheduler report must be an object"))?;
    result.insert(
        "cgroup_validation".to_owned(),
        json!({
            "authoritative_cgroup_usage_sec": cgroup_usage_sec,
            "scheduler_minus_cgroup_usage_sec": rounded(delta_sec, 9),
            "allowed_delta_sec": rounded(allowed, 9),
            "allowed_negative_delta_sec": rounded(allowed, 9),
            "allowed_positive_delta_sec": rounded(allowed, 9),
            "fractional_or_absolute_tolerance_sec": rounded(allowed, 9),
            "per_observed_event_tolerance_sec": per_observed_event_tolerance_sec,
            "event_scaled_tolerance_sec": rounded(event_scaled_tolerance_sec, 9),
            "event_scaled_tolerance_applied": false,
            "tolerance_fraction": tolerance_fraction,
            "tolerance_sec": tolerance_sec,
            "within_tolerance": within_tolerance,
            "valid_for_concurrency_reporting": trace_complete && within_tolerance && event_coverage_valid,
        }),
    );
    Ok(Value::Object(result))
}

pub(crate) fn write_scheduler_events(
    path: &Path,
    events: &[SchedulerEvent],
) -> Result<(), SchedulerPerfError> {
    let mut ordered = events.to_vec();
    ordered.sort_by_key(|event| event.timestamp_ns);
    let mut content = Vec::new();
    for event in ordered {
        serde_json::to_writer(&mut content, &event)
            .map_err(|error| SchedulerPerfError::new(error.to_string()))?;
        content.push(b'\n');
    }
    fs::write(path, content)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use tempfile::tempdir;

    struct FakeChildOperations {
        polls: VecDeque<Result<Option<i32>, SchedulerPerfError>>,
        signals: VecDeque<Result<(), SchedulerPerfError>>,
        waits: VecDeque<Result<Option<i32>, SchedulerPerfError>>,
        observed_signals: Vec<libc::c_int>,
        observed_waits: Vec<Duration>,
    }

    impl PerfChildOperations for FakeChildOperations {
        type Status = i32;

        fn poll(&mut self) -> Result<Option<Self::Status>, SchedulerPerfError> {
            self.polls.pop_front().unwrap_or(Ok(None))
        }

        fn signal(&mut self, signal: libc::c_int) -> Result<(), SchedulerPerfError> {
            self.observed_signals.push(signal);
            self.signals.pop_front().unwrap_or(Ok(()))
        }

        fn wait_bounded(
            &mut self,
            timeout: Duration,
        ) -> Result<Option<Self::Status>, SchedulerPerfError> {
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

    fn fake_perf() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/fake-perf-scheduler")
    }

    #[test]
    fn cleanup_retains_signal_error_even_when_child_is_reaped() {
        let mut operations = fake_child_operations();
        operations
            .signals
            .push_back(Err(SchedulerPerfError::new("injected SIGTERM failure")));
        operations.waits.push_back(Ok(Some(0)));

        let outcome = shutdown_perf_child(
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
            .push_back(Err(SchedulerPerfError::new("injected wait failure")));
        operations.waits.push_back(Ok(Some(9)));

        let outcome = shutdown_perf_child(
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

        let outcome = shutdown_perf_child(
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
            SchedulerPerfError::new("profile cancelled after perf attachment"),
            Err(SchedulerPerfError::new("injected child reap failure")),
        );
        assert!(error
            .to_string()
            .contains("profile cancelled after perf attachment"));
        assert!(error.to_string().contains("injected child reap failure"));
    }

    #[test]
    fn reaped_leader_does_not_hide_lingering_process_group_member() {
        let mut observations = VecDeque::from([Ok(true), Ok(true), Ok(false)]);
        let mut signals = Vec::new();
        let outcome = cleanup_perf_process_group_with(
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
            Err(SchedulerPerfError::new("injected group inspection failure")),
            Ok(false),
        ]);
        let outcome = cleanup_perf_process_group_with(
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
    fn collector_controls_fake_perf_and_retains_decoded_capture() {
        let directory = tempdir().unwrap();
        let fake_perf = fake_perf();

        let mut tracer = PerfSchedulerTracer::new(
            directory.path(),
            Some("0"),
            fake_perf.to_str().unwrap(),
            Duration::from_secs(2),
            Some(8),
        )
        .unwrap();
        tracer.start(42, Path::new("/sys/fs/cgroup")).unwrap();
        let child_pid = libc::pid_t::try_from(tracer.child.as_ref().unwrap().id()).unwrap();
        assert_eq!(unsafe { libc::getpgid(child_pid) }, child_pid);
        assert_ne!(child_pid, unsafe { libc::getpgrp() });
        tracer.poll().unwrap();
        let output = tracer.finish(1_000_000_000, 2_000_000_000).unwrap();
        assert_eq!(output.report["trace_complete"], true);
        assert_eq!(output.report["events_observed"], 2);
        assert_eq!(output.report["peak_simultaneous_logical_cores"], 1);
        assert_eq!(output.report["scheduled_runtime_sec"], 0.2);
        assert_eq!(output.report["perf_cgroup"], "/");
        assert_eq!(output.events.len(), 2);
        assert!(directory.path().join("scheduler_perf.data").is_file());
        assert!(
            fs::read_to_string(directory.path().join("scheduler_perf_switch_events.txt"))
                .unwrap()
                .contains("PERF_RECORD_SWITCH_CPU_WIDE")
        );
        assert!(directory.path().join("scheduler_perf.stderr.txt").is_file());
    }

    #[test]
    fn native_perf_records_measure_concurrency() {
        let parsed = parse_perf_script_switch_events(
            "\n root 42/42 [000] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0\nchild 43/43 [001] 1.150000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0\n   :-1 -1/-1 [001] 1.250000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0\n root 42/42 [000] 1.300000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0\n",
        )
        .unwrap();
        assert_eq!(parsed.lost_events, 0);
        assert!(parsed.anomalies.is_empty());
        let report = summarize_scheduler_events(
            &parsed.events,
            1_000_000_000,
            1_400_000_000,
            0,
            "perf-record-cgroup-native-switch-events",
            "test scope",
        )
        .unwrap();
        assert_eq!(report["scheduled_runtime_sec"], 0.3);
        assert_eq!(report["peak_simultaneous_logical_cores"], 2);
        assert_eq!(
            report["time_sec_by_simultaneous_logical_cores"],
            json!({"0": 0.2, "1": 0.1, "2": 0.1})
        );
    }

    #[test]
    fn counts_lost_events_and_surfaces_duplicate_cpu() {
        let parsed = parse_perf_script_switch_events(
            "perf 1/1 [000] 1.000000000: PERF_RECORD_LOST lost: 17\nroot 42/42 [000] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0\nchild 43/43 [000] 1.200000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 42/42\n",
        )
        .unwrap();
        assert_eq!(parsed.lost_events, 17);
        assert!(parsed.anomalies[0].contains("while recorded TID 42 was active"));
    }

    #[test]
    fn peak_ignores_zero_duration_cross_cpu_transition() {
        let report = summarize_scheduler_events(
            &[
                SchedulerEvent {
                    timestamp_ns: 10,
                    cpu: 0,
                    tid: 100,
                    entering: true,
                },
                SchedulerEvent {
                    timestamp_ns: 20,
                    cpu: 0,
                    tid: 100,
                    entering: false,
                },
                SchedulerEvent {
                    timestamp_ns: 20,
                    cpu: 1,
                    tid: 101,
                    entering: true,
                },
                SchedulerEvent {
                    timestamp_ns: 30,
                    cpu: 1,
                    tid: 101,
                    entering: false,
                },
            ],
            0,
            40,
            0,
            "test",
            "test",
        )
        .unwrap();
        assert_eq!(report["peak_simultaneous_logical_cores"], 1);
        assert_eq!(report["trace_complete"], true);
    }

    #[test]
    fn validation_is_symmetric_and_never_scales_with_event_count() {
        let report = json!({
            "scheduled_runtime_sec": 81.0,
            "events_observed": 10_000_000,
            "trace_complete": true,
        });
        let validated = add_cgroup_validation(&report, 100.0, 0.02, 0.005).unwrap();
        assert_eq!(validated["cgroup_validation"]["allowed_delta_sec"], 2.0);
        assert_eq!(validated["cgroup_validation"]["within_tolerance"], false);
        assert_eq!(
            validated["cgroup_validation"]["valid_for_concurrency_reporting"],
            false
        );
    }

    #[test]
    fn validation_uses_five_millisecond_floor_for_short_cpu_samples() {
        let within = add_cgroup_validation(
            &json!({"scheduled_runtime_sec": 0.104, "events_observed": 1, "trace_complete": true}),
            0.1,
            0.02,
            0.005,
        )
        .unwrap();
        assert_eq!(within["cgroup_validation"]["allowed_delta_sec"], 0.005);
        assert_eq!(within["cgroup_validation"]["within_tolerance"], true);

        let outside = add_cgroup_validation(
            &json!({"scheduled_runtime_sec": 0.106, "events_observed": 1, "trace_complete": true}),
            0.1,
            0.02,
            0.005,
        )
        .unwrap();
        assert_eq!(outside["cgroup_validation"]["within_tolerance"], false);
        assert_eq!(
            outside["cgroup_validation"]["valid_for_concurrency_reporting"],
            false
        );
    }

    #[test]
    fn validation_rejects_empty_trace_for_nonzero_cgroup_cpu() {
        let validated = add_cgroup_validation(
            &json!({
                "scheduled_runtime_sec": 0.0,
                "events_observed": 0,
                "events_ignored_before_window": 0,
                "events_ignored_after_window": 0,
                "trace_complete": true,
            }),
            0.004,
            0.02,
            0.005,
        )
        .unwrap();
        assert_eq!(validated["cgroup_validation"]["within_tolerance"], true);
        assert_eq!(
            validated["cgroup_validation"]["valid_for_concurrency_reporting"],
            false
        );
    }

    #[test]
    fn validation_allows_empty_trace_only_for_zero_cgroup_cpu() {
        let validated = add_cgroup_validation(
            &json!({
                "scheduled_runtime_sec": 0.0,
                "events_observed": 0,
                "events_ignored_before_window": 0,
                "events_ignored_after_window": 0,
                "trace_complete": true,
            }),
            0.0,
            0.02,
            0.005,
        )
        .unwrap();
        assert_eq!(
            validated["cgroup_validation"]["valid_for_concurrency_reporting"],
            true
        );
    }

    #[test]
    fn event_artifact_is_sorted_and_newline_terminated() {
        let directory = tempdir().unwrap();
        let path = directory.path().join("events.jsonl");
        write_scheduler_events(
            &path,
            &[
                SchedulerEvent {
                    timestamp_ns: 20,
                    cpu: 1,
                    tid: 2,
                    entering: false,
                },
                SchedulerEvent {
                    timestamp_ns: 10,
                    cpu: 1,
                    tid: 2,
                    entering: true,
                },
            ],
        )
        .unwrap();
        let text = fs::read_to_string(path).unwrap();
        assert!(text.ends_with('\n'));
        assert!(text.lines().next().unwrap().contains("\"timestamp_ns\":10"));
    }
}
