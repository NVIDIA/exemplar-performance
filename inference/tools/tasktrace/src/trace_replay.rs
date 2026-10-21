// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native replay for finalized shell-command action streams (`tasktrace-cpu-trace.v1` and its earlier ids).

use crate::timebase;
use serde_json::{json, Map, Number, Value};
use std::ffi::OsString;
use std::fs;
use std::io::{self, Read};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const TRACE_SCHEMA_VERSION: &str = "tasktrace-cpu-trace.v1";
/// Every id the shell-command trace has been finalized under. Traces are long-lived artefacts: those
/// produced before the TaskTrace rename carry `riptide-cpu-trace.v1`, and the capture package's own rename
/// emits `tasktrace-capture-cpu-trace.v1`. The record layout is identical across all three.
const ACCEPTED_TRACE_SCHEMA_VERSIONS: &[&str] = &[
    TRACE_SCHEMA_VERSION,
    "riptide-cpu-trace.v1",
    "tasktrace-capture-cpu-trace.v1",
];
const REPLAY_SCHEMA_VERSION: &str = "offline-trace-replay.v1";
const PREVIEW_LENGTH: usize = 1000;

#[derive(Debug)]
pub struct ReplayError(String);

impl std::fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl From<io::Error> for ReplayError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct TaskTraceTrace {
    source_path: PathBuf,
    header: Map<String, Value>,
    actions: Vec<Map<String, Value>>,
}

impl TaskTraceTrace {
    pub fn header(&self) -> &Map<String, Value> {
        &self.header
    }

    pub fn action_count(&self) -> usize {
        self.actions.len()
    }

    /// Whether the actions carry their recorded timestamps; read_trace allows all or none, and a
    /// trace scrubbed for publication has none.
    fn has_timestamps(&self) -> bool {
        self.actions
            .first()
            .is_some_and(|action| action.contains_key("started_at_epoch"))
    }

    fn action_timeout_sec(&self) -> Result<Option<f64>, ReplayError> {
        let timeout = self
            .header
            .get("environment")
            .and_then(Value::as_object)
            .and_then(|environment| environment.get("action_timeout_sec"));
        match timeout {
            None | Some(Value::Null) => Ok(None),
            Some(value) => {
                let timeout = value
                    .as_f64()
                    .filter(|value| value.is_finite() && *value > 0.0);
                timeout.map(Some).ok_or_else(|| {
                    ReplayError("trace environment action timeout must be positive".into())
                })
            }
        }
    }

    fn bash_env(&self) -> Result<Option<String>, ReplayError> {
        let value = self
            .header
            .get("environment")
            .and_then(Value::as_object)
            .and_then(|environment| environment.get("bash_env"));
        match value {
            None | Some(Value::Null) => Ok(None),
            Some(Value::String(path)) if !path.is_empty() => Ok(Some(path.clone())),
            Some(_) => Err(ReplayError(
                "trace environment bash_env must be a non-empty string".into(),
            )),
        }
    }

    /// Variables the source's tool commands ran with: the recorded `command_env`, or for traces
    /// captured before it existed, the agent's configured `env`. Returns the field used.
    fn command_env(&self) -> Result<(Option<&'static str>, Vec<(String, String)>), ReplayError> {
        let environment = self.header.get("environment").and_then(Value::as_object);
        for field in ["command_env", "env"] {
            match environment.and_then(|environment| environment.get(field)) {
                None | Some(Value::Null) => continue,
                Some(Value::Object(variables)) => {
                    let mut pairs = Vec::with_capacity(variables.len());
                    for (name, value) in variables {
                        let value = value.as_str().ok_or_else(|| {
                            ReplayError(format!("trace environment {field} values must be strings"))
                        })?;
                        pairs.push((name.clone(), value.to_owned()));
                    }
                    return Ok((Some(field), pairs));
                }
                Some(_) => {
                    return Err(ReplayError(format!(
                        "trace environment {field} must be an object"
                    )))
                }
            }
        }
        Ok((None, Vec::new()))
    }
}

pub fn read_trace(path: &Path) -> Result<TaskTraceTrace, ReplayError> {
    if !path.is_file() {
        return Err(ReplayError(format!(
            "TaskTrace trace does not exist: {}",
            path.display()
        )));
    }
    let text = String::from_utf8_lossy(&fs::read(path)?).into_owned();
    let mut records = Vec::new();
    for (line_index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let value: Value = serde_json::from_str(line).map_err(|error| {
            ReplayError(format!(
                "trace line {} is invalid JSON: {}",
                line_index + 1,
                error
            ))
        })?;
        let object = value.as_object().cloned().ok_or_else(|| {
            ReplayError(format!(
                "trace line {} is not a JSON object",
                line_index + 1
            ))
        })?;
        records.push(object);
    }

    let header = records
        .iter()
        .find(|record| record.get("record_type").and_then(Value::as_str) == Some("trace"))
        .cloned()
        .ok_or_else(|| {
            ReplayError(
                "trace has no record_type=trace header; finalize the TaskTrace trace first".into(),
            )
        })?;
    if !header
        .get("schema_version")
        .and_then(Value::as_str)
        .is_some_and(|id| ACCEPTED_TRACE_SCHEMA_VERSIONS.contains(&id))
    {
        return Err(ReplayError(format!(
            "unsupported TaskTrace trace schema: {}",
            display_json(header.get("schema_version"))
        )));
    }
    let actions: Vec<_> = records
        .into_iter()
        .filter(|record| record.get("record_type").and_then(Value::as_str) == Some("action"))
        .collect();
    if actions.is_empty() {
        return Err(ReplayError("trace has no replayable action records".into()));
    }

    let mut last_index = 0_i64;
    let mut timed = None;
    for action in &actions {
        let index = action.get("action_index").and_then(Value::as_i64);
        if index.is_none() || index.unwrap() <= last_index {
            return Err(ReplayError(
                "trace action indexes must be increasing positive integers".into(),
            ));
        }
        let index = index.unwrap();
        if action.get("command").and_then(Value::as_str).is_none() {
            return Err(ReplayError(format!(
                "trace action {index} has no string command"
            )));
        }
        let has_timestamps =
            action.contains_key("started_at_epoch") || action.contains_key("finished_at_epoch");
        if *timed.get_or_insert(has_timestamps) != has_timestamps {
            return Err(ReplayError(format!(
                "trace action {index} {} timestamps; a trace's actions must all have them or none",
                if has_timestamps { "has" } else { "has no" }
            )));
        }
        if has_timestamps {
            let started = action.get("started_at_epoch").and_then(Value::as_f64);
            let finished = action.get("finished_at_epoch").and_then(Value::as_f64);
            if !matches!((started, finished), (Some(a), Some(b)) if a.is_finite() && b.is_finite())
            {
                return Err(ReplayError(format!(
                    "trace action {index} has invalid timestamps"
                )));
            }
            if finished.unwrap() < started.unwrap() {
                return Err(ReplayError(format!(
                    "trace action {index} finishes before it starts"
                )));
            }
        }
        last_index = index;
    }
    Ok(TaskTraceTrace {
        source_path: path.to_owned(),
        header,
        actions,
    })
}

fn display_json(value: Option<&Value>) -> String {
    match value {
        Some(Value::String(value)) => format!("{value:?}"),
        Some(value) => value.to_string(),
        None => "None".into(),
    }
}

fn preview(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .take(PREVIEW_LENGTH)
        .collect()
}

fn source_status(action: &Map<String, Value>) -> &'static str {
    let exception = action
        .get("exception_info")
        .and_then(Value::as_str)
        .unwrap_or("");
    if exception == "action was not executed" {
        "skipped_not_executed"
    } else if exception.to_lowercase().contains("timed out") {
        "timed_out"
    } else if action.get("returncode").and_then(Value::as_i64) == Some(0) {
        "ok"
    } else {
        "failed"
    }
}

fn terminate_group(child: &mut Child) {
    let process_group = -(child.id() as i32);
    unsafe {
        libc::kill(process_group, libc::SIGTERM);
    }
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut leader_reaped = false;
    while Instant::now() < deadline {
        match child.try_wait() {
            Ok(Some(_)) => {
                leader_reaped = true;
                break;
            }
            Ok(None) => thread::sleep(Duration::from_millis(10)),
            Err(_) => break,
        }
    }
    // The shell leader can exit before a descendant that ignored SIGTERM.
    // Always signal the whole process group at the hard-cleanup boundary.
    unsafe {
        libc::kill(process_group, libc::SIGKILL);
    }
    if !leader_reaped {
        let _ = child.wait();
    }
}

const ORPHAN_SWEEP_INTERVAL: Duration = Duration::from_millis(100);
const ORPHAN_KILL_TIMEOUT: Duration = Duration::from_secs(5);

/// Helpers that actions orphan, when the current-container profiler makes this replay their child
/// subreaper: exited ones are reaped as the replay runs, and those still running after the last
/// action are killed, as container teardown kills them under docker and enroot.
#[derive(Debug, Default)]
struct OrphanAdoption {
    active: bool,
    reaped: u64,
    killed: Vec<Value>,
    last_sweep: Option<Instant>,
}

impl OrphanAdoption {
    fn detect() -> Self {
        Self {
            active: is_child_subreaper(),
            ..Self::default()
        }
    }

    /// Reap adopted helpers that have exited. `keep` is the running action's own child, which its
    /// `Child` handle must reap: exited children are only peeked (WNOWAIT) until one is known not to
    /// be `keep`.
    fn reap_exited(&mut self, keep: Option<u32>) {
        if !self.active {
            return;
        }
        loop {
            let mut info: libc::siginfo_t = unsafe { std::mem::zeroed() };
            let peeked = unsafe {
                libc::waitid(
                    libc::P_ALL,
                    0,
                    &mut info,
                    libc::WEXITED | libc::WNOHANG | libc::WNOWAIT,
                )
            };
            if peeked != 0 {
                return;
            }
            let pid = unsafe { info.si_pid() };
            if pid <= 0 || u32::try_from(pid).ok() == keep {
                return;
            }
            let mut status = 0;
            if unsafe { libc::waitpid(pid, &mut status, libc::WNOHANG) } != pid {
                return;
            }
            self.reaped += 1;
        }
    }

    fn sweep_while_running(&mut self, action_child: u32) {
        if !self.active {
            return;
        }
        let now = Instant::now();
        if self
            .last_sweep
            .is_some_and(|last| now.duration_since(last) < ORPHAN_SWEEP_INTERVAL)
        {
            return;
        }
        self.last_sweep = Some(now);
        self.reap_exited(Some(action_child));
    }

    /// After the last action every remaining child is an adopted helper: kill it and reap it.
    fn kill_leftovers(&mut self) {
        if !self.active {
            return;
        }
        self.reap_exited(None);
        let mut pending = Vec::new();
        for pid in child_pids() {
            let name = fs::read_to_string(format!("/proc/{pid}/comm"))
                .ok()
                .map(|name| name.trim().to_owned());
            if let Ok(raw) = i32::try_from(pid) {
                unsafe { libc::kill(raw, libc::SIGKILL) };
                pending.push(raw);
            }
            self.killed.push(json!({"pid": pid, "name": name}));
        }
        let deadline = Instant::now() + ORPHAN_KILL_TIMEOUT;
        while !pending.is_empty() && Instant::now() < deadline {
            // waitpid returns 0 while a killed helper is still exiting; anything else means it is gone.
            pending.retain(|pid| {
                let mut status = 0;
                let waited = unsafe { libc::waitpid(*pid, &mut status, libc::WNOHANG) };
                waited == 0
            });
            if !pending.is_empty() {
                thread::sleep(Duration::from_millis(10));
            }
        }
    }

    fn report(&self) -> Option<Value> {
        self.active.then(|| {
            json!({
                "child_subreaper": true,
                "exited_helpers_reaped": self.reaped,
                "helpers_killed_after_last_action": self.killed,
            })
        })
    }
}

fn is_child_subreaper() -> bool {
    let mut value: libc::c_int = 0;
    let unused: libc::c_ulong = 0;
    let result = unsafe {
        libc::prctl(
            libc::PR_GET_CHILD_SUBREAPER,
            &mut value as *mut libc::c_int as libc::c_ulong,
            unused,
            unused,
            unused,
        )
    };
    result == 0 && value != 0
}

fn child_pids() -> Vec<u32> {
    let own = std::process::id();
    let Ok(entries) = fs::read_dir("/proc") else {
        return Vec::new();
    };
    entries
        .filter_map(Result::ok)
        .filter_map(|entry| entry.file_name().to_str()?.parse::<u32>().ok())
        .filter(|pid| parent_pid(*pid) == Some(own))
        .collect()
}

fn parent_pid(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let closing = stat.rfind(')')?;
    stat.get(closing + 2..)?
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

fn read_pipe<T: Read + Send + 'static>(mut pipe: T) -> thread::JoinHandle<Vec<u8>> {
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let _ = pipe.read_to_end(&mut bytes);
        bytes
    })
}

fn execute_action(
    action: &Map<String, Value>,
    cwd: &Path,
    timeout_sec: Option<f64>,
    command_env: &[(String, String)],
    replace_env: bool,
    bash_env: Option<&str>,
    adoption: &mut OrphanAdoption,
) -> Result<Map<String, Value>, ReplayError> {
    let command = action["command"].as_str().unwrap();
    let mut process = Command::new("/bin/bash");
    // A recorded command_env is the whole environment the command ran with, not an overlay.
    if replace_env {
        process.env_clear();
    }
    process
        .arg("-c")
        .arg(command)
        .current_dir(cwd)
        .envs(command_env.iter().map(|(name, value)| (name, value)))
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    if let Some(path) = bash_env {
        process.env("BASH_ENV", path);
    }
    unsafe {
        process.pre_exec(|| {
            if libc::setsid() == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok(())
        });
    }
    let mut child = process
        .spawn()
        .map_err(|error| ReplayError(format!("unable to execute trace action command: {error}")))?;
    let stdout_thread = read_pipe(child.stdout.take().unwrap());
    let stderr_thread = read_pipe(child.stderr.take().unwrap());
    let deadline = timeout_sec.map(|seconds| Instant::now() + Duration::from_secs_f64(seconds));
    let mut timed_out = false;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            timed_out = true;
            terminate_group(&mut child);
            break None;
        }
        adoption.sweep_while_running(child.id());
        thread::sleep(Duration::from_millis(2));
    };
    let stdout = stdout_thread.join().unwrap_or_default();
    let stderr = stderr_thread.join().unwrap_or_default();
    let mut result = Map::new();
    if timed_out {
        result.insert("status".into(), json!("timed_out"));
    } else {
        let status = status.unwrap();
        let returncode = status
            .code()
            .map(i64::from)
            .or_else(|| status.signal().map(|signal| -(signal as i64)))
            .unwrap_or(1);
        result.insert(
            "status".into(),
            json!(if status.success() { "ok" } else { "failed" }),
        );
        result.insert("returncode".into(), json!(returncode));
    }
    result.insert("stdout_preview".into(), json!(preview(&stdout)));
    result.insert("stderr_preview".into(), json!(preview(&stderr)));
    Ok(result)
}

fn rounded(value: f64, places: i32) -> Value {
    let scale = 10_f64.powi(places);
    Number::from_f64((value * scale).round() / scale)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

/// The tool side's half of a loop-order replay (`--loop-sync-dir`): a concurrent LLM-side replay and this one
/// keep the recorded agent loop's order, without its timings, through small files in one directory. Before an
/// action that followed LLM call k (its `after_llm_index`) this side waits until `llm_done` reaches k or
/// `llm_end` exists; after every action it writes the action's index to `tool_done`, and `tool_end` when done.
struct LoopSync {
    dir: PathBuf,
}

impl LoopSync {
    const POLL: Duration = Duration::from_millis(5);

    fn start(dir: &Path) -> Result<Self, ReplayError> {
        fs::create_dir_all(dir).map_err(|error| {
            ReplayError(format!(
                "cannot create the loop-sync directory {}: {error}",
                dir.display()
            ))
        })?;
        Ok(Self {
            dir: dir.to_path_buf(),
        })
    }

    fn counter(&self, name: &str) -> Option<i64> {
        fs::read_to_string(self.dir.join(name))
            .ok()?
            .trim()
            .parse()
            .ok()
    }

    /// Wait until the LLM side has finished call `llm_index` (or ended); returns the seconds waited.
    fn wait_for_llm(&self, llm_index: i64) -> f64 {
        let started = Instant::now();
        while self
            .counter("llm_done")
            .map_or(true, |done| done < llm_index)
            && !self.dir.join("llm_end").exists()
        {
            thread::sleep(Self::POLL);
        }
        started.elapsed().as_secs_f64()
    }

    fn tool_done(&self, action_index: &Value) -> Result<(), ReplayError> {
        let partial = self.dir.join(".tool_done.partial");
        fs::write(&partial, format!("{action_index}\n"))
            .and_then(|()| fs::rename(&partial, self.dir.join("tool_done")))
            .map_err(|error| ReplayError(format!("cannot write the loop-sync progress: {error}")))
    }
}

impl Drop for LoopSync {
    fn drop(&mut self) {
        let _ = fs::write(self.dir.join("tool_end"), b"");
    }
}

pub fn replay_trace(
    trace: &TaskTraceTrace,
    cwd: &Path,
    pacing: &str,
    timeout_sec: Option<f64>,
) -> Result<Value, ReplayError> {
    replay_trace_with_loop_sync(trace, cwd, pacing, timeout_sec, None)
}

pub fn replay_trace_with_loop_sync(
    trace: &TaskTraceTrace,
    cwd: &Path,
    pacing: &str,
    timeout_sec: Option<f64>,
    loop_sync_dir: Option<&Path>,
) -> Result<Value, ReplayError> {
    let pacing = crate::pacing::canonical(pacing);
    if !matches!(pacing, "recorded" | "unpaced") {
        return Err(ReplayError(format!("unsupported pacing mode: {pacing:?}")));
    }
    if !cwd.is_dir() {
        return Err(ReplayError(format!(
            "replay working directory does not exist: {}",
            cwd.display()
        )));
    }
    let effective_timeout = match timeout_sec {
        Some(value) if value.is_finite() && value > 0.0 => Some(value),
        Some(_) => return Err(ReplayError("action timeout must be positive".into())),
        None => trace.action_timeout_sec()?,
    };
    let bash_env = trace.bash_env()?;
    let (environment_source, command_env) = trace.command_env()?;
    let executable: Vec<_> = trace
        .actions
        .iter()
        .filter(|action| source_status(action) != "skipped_not_executed")
        .collect();
    if executable.is_empty() {
        return Err(ReplayError(
            "trace has no source-executed actions to replay".into(),
        ));
    }
    let timed = trace.has_timestamps();
    if pacing == "recorded" && !timed {
        return Err(ReplayError(
            "recorded pacing needs the trace's timestamps, and this trace has none; use unpaced"
                .into(),
        ));
    }
    if loop_sync_dir.is_some() {
        if pacing != "unpaced" {
            return Err(ReplayError(
                "a loop-sync replay keeps the recorded order, not the schedule; use unpaced pacing"
                    .into(),
            ));
        }
        if let Some(action) = executable
            .iter()
            .find(|action| !action.contains_key("after_llm_index"))
        {
            return Err(ReplayError(format!(
                "trace action {} has no after_llm_index; a loop-sync replay needs a trace written with tasktrace-capture --trajectory-timings omit",
                action["action_index"]
            )));
        }
    }
    let first_start = executable[0]
        .get("started_at_epoch")
        .and_then(Value::as_f64);
    let source_end = executable
        .last()
        .unwrap()
        .get("finished_at_epoch")
        .and_then(Value::as_f64);
    let loop_sync = loop_sync_dir.map(LoopSync::start).transpose()?;
    let replay_started = Instant::now();
    let mut adoption = OrphanAdoption::detect();
    let mut results = Vec::new();
    for action in &trace.actions {
        let expected = source_status(action);
        let mut result = Map::new();
        for field in ["action_index", "command"] {
            result.insert(field.into(), action[field].clone());
        }
        result.insert("source_status".into(), json!(expected));
        result.insert(
            "source_wall_time_sec".into(),
            action.get("wall_time_sec").cloned().unwrap_or(Value::Null),
        );
        if expected == "skipped_not_executed" {
            result.insert("status".into(), json!("skipped_not_executed"));
            result.insert("outcome_matches_source".into(), json!(true));
            results.push(Value::Object(result));
            if let Some(sync) = &loop_sync {
                sync.tool_done(&action["action_index"])?;
            }
            continue;
        }
        let source_offset = action
            .get("started_at_epoch")
            .and_then(Value::as_f64)
            .zip(first_start)
            .map(|(started, first)| started - first);
        result.insert(
            "source_start_offset_sec".into(),
            source_offset.map_or(Value::Null, |offset| rounded(offset, 9)),
        );
        if let (Some(offset), "recorded") = (source_offset, pacing) {
            let sleep = offset - replay_started.elapsed().as_secs_f64();
            if sleep > 0.0 {
                thread::sleep(Duration::from_secs_f64(sleep));
            }
        }
        if let Some(sync) = &loop_sync {
            let waited = action
                .get("after_llm_index")
                .and_then(Value::as_i64)
                .map_or(0.0, |llm_index| sync.wait_for_llm(llm_index));
            result.insert("loop_wait_sec".into(), rounded(waited, 9));
        }
        let action_started = Instant::now();
        let action_started_ns = timebase::monotonic_ns();
        result.extend(execute_action(
            action,
            cwd,
            effective_timeout,
            &command_env,
            environment_source == Some("command_env"),
            bash_env.as_deref(),
            &mut adoption,
        )?);
        result.insert(
            "wall_time_sec".into(),
            rounded(action_started.elapsed().as_secs_f64(), 9),
        );
        result.insert("replay_start_monotonic_ns".into(), json!(action_started_ns));
        result.insert(
            "replay_end_monotonic_ns".into(),
            json!(timebase::monotonic_ns()),
        );
        let observed = result["status"].as_str().unwrap();
        result.insert("outcome_matches_source".into(), json!(observed == expected));
        results.push(Value::Object(result));
        adoption.reap_exited(None);
        if let Some(sync) = &loop_sync {
            sync.tool_done(&action["action_index"])?;
        }
    }
    let replay_wall_sec = replay_started.elapsed().as_secs_f64();
    adoption.kill_leftovers();
    let mismatches: Vec<_> = results
        .iter()
        .filter(|result| result["outcome_matches_source"] == false)
        .map(|result| result["action_index"].clone())
        .collect();
    let mismatch_count = mismatches.len();
    let count = |status: &str| {
        results
            .iter()
            .filter(|result| result["status"].as_str() == Some(status))
            .count()
    };
    let source = trace
        .source_path
        .canonicalize()
        .unwrap_or_else(|_| trace.source_path.clone());
    let mut report = json!({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_trace": source.to_string_lossy(),
        "source_schema_version": trace.header.get("schema_version").cloned().unwrap_or(Value::Null),
        "source_trial": trace.header.get("source_trial").cloned().unwrap_or(Value::Null),
        "task_name": trace.header.get("task_name").cloned().unwrap_or(Value::Null),
        "source_reward": trace.header.get("reward").cloned().unwrap_or(Value::Null),
        "source_limitations": trace.header.get("limitations").cloned().unwrap_or_else(|| json!([])),
        "pacing": pacing,
        "loop_sync_dir": loop_sync_dir.map_or(Value::Null, |dir| json!(dir.to_string_lossy())),
        "action_timeout_sec": effective_timeout,
        "bash_env": bash_env,
        "environment_source": environment_source,
        "clock_reference": timebase::clock_reference(),
        "actions": results,
        "summary": {
            "actions_recorded": trace.actions.len(),
            "actions_executed": executable.len(),
            "actions_skipped_not_executed": trace.actions.len() - executable.len(),
            "actions_ok": count("ok"),
            "actions_failed": count("failed"),
            "actions_timed_out": count("timed_out"),
            "source_outcome_mismatch_action_indexes": mismatches,
            "source_outcome_mismatch_count": mismatch_count,
            "source_interval_sec": source_end
                .zip(first_start)
                .map_or(Value::Null, |(end, start)| rounded(end - start, 9)),
            "wall_time_sec": rounded(replay_wall_sec, 9),
        },
    });
    if let Some(orphan_adoption) = adoption.report() {
        report["orphan_adoption"] = orphan_adoption;
    }
    Ok(report)
}

#[derive(Debug)]
struct Cli {
    trace: PathBuf,
    output: PathBuf,
    cwd: PathBuf,
    pacing: String,
    timeout_sec: Option<f64>,
    loop_sync_dir: Option<PathBuf>,
}

fn usage(program: &str, output: &mut dyn io::Write) {
    let _ = writeln!(
        output,
        "usage: {program} [-h] --output OUTPUT [--cwd CWD] [--pacing {{recorded,unpaced}}] [--timeout-sec TIMEOUT_SEC] [--loop-sync-dir LOOP_SYNC_DIR] trace"
    );
}

fn help(program: &str) {
    usage(program, &mut io::stdout());
    println!();
    println!("Replay action records from a finalized TaskTrace CPU trace.");
    println!();
    println!("positional arguments:");
    println!("  trace                 Finalized tasktrace_shell_commands.jsonl.");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --output OUTPUT       Replay report JSON path.");
    println!("  --cwd CWD             Replay working directory.");
    println!("  --pacing {{recorded,unpaced}}");
    println!("  --timeout-sec TIMEOUT_SEC");
    println!(
        "                        Override the per-action timeout recorded in the source trace."
    );
    println!("  --loop-sync-dir LOOP_SYNC_DIR");
    println!(
        "                        Keep the recorded agent loop's order with a concurrent LLM-side replay, through"
    );
    println!(
        "                        files in this directory (unpaced; the trace's actions need after_llm_index)."
    );
}

fn cli_error(program: &str, message: impl std::fmt::Display) -> i32 {
    usage(program, &mut io::stderr());
    eprintln!("{program}: error: {message}");
    2
}

fn option_value(
    arguments: &[OsString],
    index: &mut usize,
    option: &str,
) -> Result<OsString, String> {
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| format!("argument {option}: expected one argument"))
}

fn parse_cli(arguments: &[OsString]) -> Result<Option<Cli>, String> {
    let mut trace = None;
    let mut output = None;
    let mut cwd = std::env::current_dir().map_err(|error| error.to_string())?;
    let mut pacing = "recorded".to_owned();
    let mut timeout_sec = None;
    let mut loop_sync_dir = None;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        let text = argument.to_string_lossy();
        match text.as_ref() {
            "-h" | "--help" => return Ok(None),
            "--output" => {
                output = Some(PathBuf::from(option_value(
                    arguments, &mut index, "--output",
                )?))
            }
            "--cwd" => cwd = PathBuf::from(option_value(arguments, &mut index, "--cwd")?),
            "--pacing" => {
                pacing = option_value(arguments, &mut index, "--pacing")?
                    .to_string_lossy()
                    .into_owned()
            }
            "--timeout-sec" => {
                let raw = option_value(arguments, &mut index, "--timeout-sec")?;
                timeout_sec = Some(raw.to_string_lossy().parse::<f64>().map_err(|_| {
                    format!("argument --timeout-sec: invalid float value: {:?}", raw)
                })?);
            }
            "--loop-sync-dir" => {
                loop_sync_dir = Some(PathBuf::from(option_value(
                    arguments,
                    &mut index,
                    "--loop-sync-dir",
                )?))
            }
            value if value.starts_with('-') => {
                return Err(format!("unrecognized arguments: {value}"))
            }
            _ if trace.is_none() => trace = Some(PathBuf::from(argument)),
            _ => return Err(format!("unrecognized arguments: {text}")),
        }
        index += 1;
    }
    let trace = trace.ok_or_else(|| "the following arguments are required: trace".to_owned())?;
    let output =
        output.ok_or_else(|| "the following arguments are required: --output".to_owned())?;
    let pacing = crate::pacing::canonical(&pacing).to_owned();
    if !matches!(pacing.as_str(), "recorded" | "unpaced") {
        return Err(format!(
            "argument --pacing: invalid choice: {pacing:?} (choose from 'recorded', 'unpaced')"
        ));
    }
    if timeout_sec.is_some_and(|value| !value.is_finite() || value <= 0.0) {
        return Err("--timeout-sec must be positive".into());
    }
    Ok(Some(Cli {
        trace,
        output,
        cwd,
        pacing,
        timeout_sec,
        loop_sync_dir,
    }))
}

pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let arguments = match parse_cli(arguments) {
        Ok(Some(arguments)) => arguments,
        Ok(None) => {
            help(public_program);
            return 0;
        }
        Err(error) => return cli_error(public_program, error),
    };
    let report = match read_trace(&arguments.trace).and_then(|trace| {
        replay_trace_with_loop_sync(
            &trace,
            &arguments.cwd,
            &arguments.pacing,
            arguments.timeout_sec,
            arguments.loop_sync_dir.as_deref(),
        )
    }) {
        Ok(report) => report,
        Err(error) => return cli_error(public_program, error),
    };
    if let Some(parent) = arguments
        .output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        if let Err(error) = fs::create_dir_all(parent) {
            return cli_error(public_program, error);
        }
    }
    let mut serialized = serde_json::to_string_pretty(&report).unwrap();
    serialized.push('\n');
    if let Err(error) = fs::write(&arguments.output, serialized) {
        return cli_error(public_program, error);
    }
    let summary = &report["summary"];
    println!("Task: {}", display_report_value(&report["task_name"]));
    println!(
        "Source trial: {}",
        display_report_value(&report["source_trial"])
    );
    println!(
        "Actions: {}/{} executed, {} timed out, {} source-outcome mismatches",
        summary["actions_executed"],
        summary["actions_recorded"],
        summary["actions_timed_out"],
        summary["source_outcome_mismatch_count"],
    );
    println!(
        "Wall time: {:.9} sec",
        summary["wall_time_sec"].as_f64().unwrap()
    );
    println!("Report: {}", arguments.output.display());
    (summary["source_outcome_mismatch_count"].as_u64().unwrap() > 0) as i32
}

fn display_report_value(value: &Value) -> String {
    match value {
        Value::Null => "None".into(),
        Value::String(value) => value.clone(),
        value => value.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn trace(actions: &[Value]) -> (tempfile::TempDir, TaskTraceTrace) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut file = fs::File::create(&path).unwrap();
        writeln!(
            file,
            "{}",
            json!({
                "record_type": "trace",
                "schema_version": TRACE_SCHEMA_VERSION,
                "source_trial": "task__trial",
                "task_name": "task",
                "environment": {"action_timeout_sec": 0.1},
                "limitations": [],
            })
        )
        .unwrap();
        for action in actions {
            writeln!(file, "{action}").unwrap();
        }
        let parsed = read_trace(&path).unwrap();
        (directory, parsed)
    }

    fn action(index: i64, command: &str, start: f64, end: f64) -> Value {
        json!({
            "record_type": "action",
            "action_index": index,
            "command": command,
            "started_at_epoch": start,
            "finished_at_epoch": end,
            "wall_time_sec": end - start,
            "returncode": 0,
            "exception_info": Value::Null,
        })
    }

    #[test]
    fn executes_and_skips_source_action() {
        let skipped = json!({
            "record_type": "action", "action_index": 2,
            "command": "printf bad > skipped", "started_at_epoch": 1.02,
            "finished_at_epoch": 1.03, "wall_time_sec": 0.01,
            "returncode": -1, "exception_info": "action was not executed",
        });
        let (directory, trace) = trace(&[action(1, "printf done > output", 1.0, 1.01), skipped]);
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(
            fs::read_to_string(directory.path().join("output")).unwrap(),
            "done"
        );
        assert!(!directory.path().join("skipped").exists());
        assert_eq!(report["summary"]["actions_executed"], 1);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
    }

    fn untimed(index: i64, command: &str) -> Value {
        json!({
            "record_type": "action", "action_index": index, "command": command,
            "returncode": 0, "exception_info": Value::Null,
        })
    }

    #[test]
    fn untimed_trace_replays_unpaced_and_refuses_recorded_pacing() {
        let (directory, trace) = trace(&[untimed(1, "printf a > a"), untimed(2, "printf b > b")]);
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["summary"]["actions_executed"], 2);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
        assert_eq!(report["summary"]["source_interval_sec"], Value::Null);
        assert_eq!(report["actions"][0]["source_start_offset_sec"], Value::Null);
        assert_eq!(report["actions"][0]["source_wall_time_sec"], Value::Null);
        let error = replay_trace(&trace, directory.path(), "recorded", None).unwrap_err();
        assert!(error
            .0
            .contains("recorded pacing needs the trace's timestamps"));
    }

    #[test]
    fn trace_mixing_timed_and_untimed_actions_is_rejected() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let header = json!({"record_type": "trace", "schema_version": TRACE_SCHEMA_VERSION});
        let rows = [header, action(1, "true", 1.0, 1.1), untimed(2, "true")];
        let body: Vec<String> = rows.iter().map(Value::to_string).collect();
        fs::write(&path, body.join("\n") + "\n").unwrap();
        let error = read_trace(&path).unwrap_err();
        assert!(
            error.0.contains("must all have them or none"),
            "{}",
            error.0
        );
    }

    fn in_loop(index: i64, after: Option<i64>, command: &str) -> Value {
        let mut action = untimed(index, command);
        action["after_llm_index"] = after.map_or(Value::Null, Value::from);
        action
    }

    fn wait_for_counter(dir: &Path, name: &str, at_least: i64) {
        let deadline = Instant::now() + Duration::from_secs(10);
        loop {
            let value = fs::read_to_string(dir.join(name))
                .ok()
                .and_then(|text| text.trim().parse::<i64>().ok());
            if value.is_some_and(|value| value >= at_least) {
                return;
            }
            assert!(Instant::now() < deadline, "{name} never reached {at_least}");
            thread::sleep(Duration::from_millis(2));
        }
    }

    #[test]
    fn loop_sync_keeps_the_recorded_agent_loop_order() {
        let (directory, trace) = trace(&[
            in_loop(1, None, "printf a >> order"),
            in_loop(2, Some(1), "printf b >> order"),
            in_loop(3, Some(2), "printf c >> order"),
        ]);
        let sync = directory.path().join("sync");
        let order = directory.path().join("order");
        let llm_side = {
            let (sync, order) = (sync.clone(), order.clone());
            thread::spawn(move || {
                for (llm_index, after_action) in [(1, 1), (2, 2)] {
                    wait_for_counter(&sync, "tool_done", after_action);
                    let mut file = fs::OpenOptions::new().append(true).open(&order).unwrap();
                    write!(file, "L{llm_index}").unwrap();
                    fs::write(sync.join("llm_done"), format!("{llm_index}\n")).unwrap();
                }
                fs::write(sync.join("llm_end"), b"").unwrap();
            })
        };
        let report =
            replay_trace_with_loop_sync(&trace, directory.path(), "unpaced", None, Some(&sync))
                .unwrap();
        llm_side.join().unwrap();
        assert_eq!(fs::read_to_string(&order).unwrap(), "aL1bL2c");
        assert_eq!(report["loop_sync_dir"], json!(sync.to_string_lossy()));
        assert!(report["actions"][1]["loop_wait_sec"].as_f64().is_some());
        assert!(sync.join("tool_end").exists());
        assert_eq!(
            fs::read_to_string(sync.join("tool_done")).unwrap().trim(),
            "3"
        );
    }

    #[test]
    fn loop_sync_stops_waiting_once_the_llm_side_has_ended() {
        let (directory, trace) = trace(&[in_loop(1, Some(5), "true")]);
        let sync = directory.path().join("sync");
        fs::create_dir_all(&sync).unwrap();
        fs::write(sync.join("llm_end"), b"").unwrap();
        let report =
            replay_trace_with_loop_sync(&trace, directory.path(), "unpaced", None, Some(&sync))
                .unwrap();
        assert_eq!(report["summary"]["actions_executed"], 1);
    }

    #[test]
    fn loop_sync_needs_unpaced_pacing_and_the_loop_order() {
        let (directory, in_order) = trace(&[in_loop(1, None, "true")]);
        let sync = directory.path().join("sync");
        let error =
            replay_trace_with_loop_sync(&in_order, directory.path(), "recorded", None, Some(&sync))
                .unwrap_err();
        assert!(error.0.contains("use unpaced"), "{}", error.0);
        let (directory, unordered) = trace(&[untimed(1, "true")]);
        let error =
            replay_trace_with_loop_sync(&unordered, directory.path(), "unpaced", None, Some(&sync))
                .unwrap_err();
        assert!(error.0.contains("has no after_llm_index"), "{}", error.0);
    }

    #[test]
    fn timeout_matches_recorded_timeout_and_cleans_descendant() {
        let timed_out = json!({
            "record_type": "action", "action_index": 1,
            "command": "sleep 5 & echo $! > child.pid; wait",
            "started_at_epoch": 1.0, "finished_at_epoch": 1.01,
            "wall_time_sec": 0.01, "returncode": -1,
            "exception_info": "command timed out after 0.01 seconds",
        });
        let (directory, trace) = trace(&[timed_out]);
        let report = replay_trace(&trace, directory.path(), "unpaced", Some(0.02)).unwrap();
        assert_eq!(report["actions"][0]["status"], "timed_out");
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
        let pid: i32 = fs::read_to_string(directory.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let alive = unsafe { libc::kill(pid, 0) } == 0;
        assert!(!alive, "timed-out descendant {pid} survived replay");
    }

    #[test]
    fn header_bash_env_is_sourced_by_every_replayed_command() {
        let directory = tempfile::tempdir().unwrap();
        let rc = directory.path().join("image.bashrc");
        fs::write(&rc, "export REPLAY_MARK=activated\n").unwrap();
        let path = directory.path().join("trace.jsonl");
        let rows = [
            json!({"record_type": "trace", "schema_version": "tasktrace-cpu-trace.v1", "source_trial": "t",
                   "task_name": "task", "limitations": [],
                   "environment": {"action_timeout_sec": 5.0, "bash_env": rc.to_string_lossy()}}),
            action(1, "test \"$REPLAY_MARK\" = activated", 1.0, 1.01),
        ];
        let body: Vec<String> = rows.iter().map(|row| row.to_string()).collect();
        fs::write(&path, body.join("\n") + "\n").unwrap();
        let trace = read_trace(&path).unwrap();
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["bash_env"], json!(rc.to_string_lossy()));
        assert_eq!(report["actions"][0]["returncode"], 0);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);

        let rows = [
            json!({"record_type": "trace", "schema_version": "tasktrace-cpu-trace.v1", "source_trial": "t",
                   "task_name": "task", "limitations": [], "environment": {"action_timeout_sec": 5.0}}),
            action(1, "test \"$REPLAY_MARK\" = activated", 1.0, 1.01),
        ];
        let body: Vec<String> = rows.iter().map(|row| row.to_string()).collect();
        fs::write(&path, body.join("\n") + "\n").unwrap();
        let trace = read_trace(&path).unwrap();
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["bash_env"], Value::Null);
        assert_eq!(report["actions"][0]["returncode"], 1);
    }

    fn trace_with_environment(
        environment: Value,
        actions: &[Value],
    ) -> (tempfile::TempDir, TaskTraceTrace) {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        let mut rows = vec![
            json!({"record_type": "trace", "schema_version": TRACE_SCHEMA_VERSION,
                   "source_trial": "t", "task_name": "task", "limitations": [],
                   "environment": environment}),
        ];
        rows.extend(actions.iter().cloned());
        let body: Vec<String> = rows.iter().map(|row| row.to_string()).collect();
        fs::write(&path, body.join("\n") + "\n").unwrap();
        let trace = read_trace(&path).unwrap();
        (directory, trace)
    }

    #[test]
    fn recorded_command_env_is_applied_in_place_of_configured_env() {
        let (directory, trace) = trace_with_environment(
            json!({"action_timeout_sec": 5.0,
                   "command_env": {"REPLAY_MARK": "recorded", "LC_CTYPE": "C.UTF-8"},
                   "env": {"REPLAY_MARK": "configured", "CONFIGURED_ONLY": "x"}}),
            &[action(1, "test \"$REPLAY_MARK\" = recorded && test \"$LC_CTYPE\" = C.UTF-8 && test -z \"${CONFIGURED_ONLY-}\"", 1.0, 1.01)],
        );
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["environment_source"], "command_env");
        assert_eq!(report["actions"][0]["returncode"], 0);
    }

    #[test]
    fn recorded_command_env_replaces_the_inherited_environment() {
        assert!(std::env::var_os("HOME").is_some(), "needs an inherited HOME to prove it is dropped");
        let (directory, trace) = trace_with_environment(
            json!({"action_timeout_sec": 5.0, "command_env": {"REPLAY_MARK": "recorded"}}),
            &[action(1, "test \"$REPLAY_MARK\" = recorded && test -z \"${HOME-}\"", 1.0, 1.01)],
        );
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["actions"][0]["returncode"], 0);
    }

    #[test]
    fn configured_env_is_applied_when_command_env_was_not_recorded() {
        let (directory, trace) = trace_with_environment(
            json!({"action_timeout_sec": 5.0, "command_env": null, "env": {"REPLAY_MARK": "configured"}}),
            &[action(1, "test \"$REPLAY_MARK\" = configured", 1.0, 1.01)],
        );
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["environment_source"], "env");
        assert_eq!(report["actions"][0]["returncode"], 0);

        let (directory, trace) = trace_with_environment(
            json!({"action_timeout_sec": 5.0}),
            &[action(1, "true", 1.0, 1.01)],
        );
        let report = replay_trace(&trace, directory.path(), "unpaced", None).unwrap();
        assert_eq!(report["environment_source"], Value::Null);
    }

    #[test]
    fn rejects_non_string_command_env_values() {
        let (directory, trace) = trace_with_environment(
            json!({"action_timeout_sec": 5.0, "command_env": {"N": 1}}),
            &[action(1, "true", 1.0, 1.01)],
        );
        let error = replay_trace(&trace, directory.path(), "unpaced", None).unwrap_err();
        assert!(error
            .to_string()
            .contains("command_env values must be strings"));
    }

    #[test]
    fn accepts_every_finalized_schema_id() {
        for id in ["tasktrace-cpu-trace.v1", "riptide-cpu-trace.v1", "tasktrace-capture-cpu-trace.v1"] {
            let directory = tempfile::tempdir().unwrap();
            let path = directory.path().join("trace.jsonl");
            let rows = [
                json!({"record_type": "trace", "schema_version": id, "source_trial": "t", "task_name": "task",
                       "environment": {"action_timeout_sec": 0.1}, "limitations": []}),
                action(1, "true", 1.0, 1.01),
            ];
            let body: Vec<String> = rows.iter().map(|row| row.to_string()).collect();
            fs::write(&path, body.join("\n") + "\n").unwrap();
            let trace = read_trace(&path).unwrap_or_else(|err| panic!("{id}: {err}"));
            assert_eq!(trace.header["schema_version"], id);
            assert_eq!(trace.actions.len(), 1);
        }
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("trace.jsonl");
        fs::write(&path, json!({"record_type": "trace", "schema_version": "something-else.v9"}).to_string() + "\n").unwrap();
        assert!(read_trace(&path).unwrap_err().to_string().contains("unsupported TaskTrace trace schema"));
    }

    #[test]
    fn rejects_unfinalized_trace() {
        let directory = tempfile::tempdir().unwrap();
        let path = directory.path().join("raw.jsonl");
        fs::write(&path, "{\"command_shell_quoted\":\"true\"}\n").unwrap();
        assert!(read_trace(&path)
            .unwrap_err()
            .to_string()
            .contains("finalize the TaskTrace trace"));
    }
}
