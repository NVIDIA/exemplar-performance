// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transcript JSONL extraction and deterministic CPU-side tool replay.
//!
//! This is the native implementation of the public `replay transcript` leaf.
//! It intentionally supports only the three tool families implemented by the
//! original adapter: shell execution, workspace reads, and workspace writes.

use crate::timebase;
use serde_json::{json, Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufRead, BufReader, BufWriter, Read, Write};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Component, Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const TOOL_MARKER_PREFIX: &str = "[Tool Call:";
const PREVIEW_LENGTH: usize = 1000;
const SCHEMA_VERSION: &str = "offline-tool-replay.v1";
const POLL_INTERVAL: Duration = Duration::from_millis(2);

static CGROUP_SEQUENCE: AtomicU64 = AtomicU64::new(0);

/// An extraction or replay failure that is safe to show at the CLI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayError(String);

impl ReplayError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ReplayError {}

impl From<io::Error> for ReplayError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ReplayError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

/// Typed actions recovered from a single transcript task/trial selection.
#[derive(Clone, Debug, PartialEq)]
pub struct Extraction {
    pub task_id: String,
    pub trial_name: Option<String>,
    pub actions: Vec<Map<String, Value>>,
    pub warnings: Vec<String>,
}

/// Runtime options for [`replay_actions`].
#[derive(Clone, Debug)]
pub struct ReplayConfig {
    pub workspace: PathBuf,
    pub source_workspace: Option<PathBuf>,
    pub initial_workspace: Option<PathBuf>,
    pub timeout_sec: f64,
    pub strict: bool,
    pub cgroup_cpu: bool,
    pub cgroup_ready_file: Option<PathBuf>,
    pub cgroup_start_delay_sec: f64,
}

impl ReplayConfig {
    pub fn new(workspace: impl Into<PathBuf>) -> Self {
        Self {
            workspace: workspace.into(),
            source_workspace: None,
            initial_workspace: None,
            timeout_sec: 60.0,
            strict: false,
            cgroup_cpu: false,
            cgroup_ready_file: None,
            cgroup_start_delay_sec: 0.0,
        }
    }
}

#[derive(Clone, Debug)]
struct Marker {
    end: usize,
    raw_tool: String,
}

fn tool_markers(text: &str) -> Vec<Marker> {
    let mut markers = Vec::new();
    let mut search_from = 0;
    while let Some(relative_start) = text[search_from..].find(TOOL_MARKER_PREFIX) {
        let start = search_from + relative_start;
        let after_prefix = start + TOOL_MARKER_PREFIX.len();
        let Some(relative_end) = text[after_prefix..].find(']') else {
            break;
        };
        let end = after_prefix + relative_end;
        // The Python expression uses `\s*([^\]]+)`; a marker containing only
        // whitespace still matches after regex backtracking and normalizes to
        // an empty tool name.
        if end > after_prefix {
            markers.push(Marker {
                end: end + 1,
                raw_tool: text[after_prefix..end].trim_start().trim().to_owned(),
            });
        }
        search_from = end + 1;
    }
    markers
}

fn answer_has_tool_marker(record: &Map<String, Value>) -> bool {
    record
        .get("answer")
        .and_then(Value::as_str)
        .map(|answer| !tool_markers(answer).is_empty())
        .unwrap_or(false)
}

fn python_string(value: &Value) -> Option<String> {
    match value {
        Value::Null => None,
        Value::String(value) => Some(value.clone()),
        Value::Bool(value) => Some(if *value { "True" } else { "False" }.to_owned()),
        Value::Number(value) => Some(value.to_string()),
        other => Some(other.to_string()),
    }
}

fn python_repr(value: &str) -> String {
    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
}

fn selected_string(record: &Map<String, Value>, key: &str, selected: &str) -> bool {
    matches!(record.get(key), Some(Value::String(value)) if value == selected)
}

fn decode_object_after(text: &str, position: usize) -> Result<Map<String, Value>, String> {
    let remaining = &text[position..];
    let mut values = serde_json::Deserializer::from_str(remaining).into_iter::<Value>();
    let Some(decoded) = values.next() else {
        return Err("expected value".to_owned());
    };
    let decoded = decoded.map_err(|error| error.to_string())?;
    match decoded {
        Value::Object(object) => Ok(object),
        _ => Err("tool-call payload is not a JSON object".to_owned()),
    }
}

/// Extract supported and unsupported tool calls in transcript source order.
pub fn extract_actions(
    transcript_path: &Path,
    task_id: Option<&str>,
    trial_name: Option<&str>,
) -> Result<Extraction, ReplayError> {
    let bytes = fs::read(transcript_path)?;
    let text = String::from_utf8_lossy(&bytes);
    let mut records = Vec::<Map<String, Value>>::new();
    let mut warnings = Vec::new();

    for (index, line) in text.lines().enumerate() {
        let line_number = index + 1;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str::<Value>(line) {
            Ok(Value::Object(mut record)) => {
                record.insert(
                    "_line_number".to_owned(),
                    Value::Number(Number::from(line_number as u64)),
                );
                records.push(record);
            }
            Ok(_) => {}
            Err(error) => warnings.push(format!(
                "line {line_number}: invalid JSON skipped ({error})"
            )),
        }
    }

    let available: BTreeSet<String> = records
        .iter()
        .filter(|record| answer_has_tool_marker(record))
        .filter_map(|record| record.get("task_id").and_then(python_string))
        .collect();
    let available_display = || {
        if available.is_empty() {
            "(none)".to_owned()
        } else {
            available.iter().cloned().collect::<Vec<_>>().join(", ")
        }
    };
    let selected_task = match task_id {
        Some(task_id) if !available.contains(task_id) => {
            return Err(ReplayError::new(format!(
                "task {} has no extractable tool calls; available: {}",
                python_repr(task_id),
                available_display()
            )));
        }
        Some(task_id) => task_id.to_owned(),
        None if available.len() != 1 => {
            return Err(ReplayError::new(format!(
                "provide --task-id because the transcript contains tool calls for {} tasks: {}",
                available.len(),
                available_display()
            )));
        }
        None => available.iter().next().expect("one available task").clone(),
    };

    let available_trials: BTreeSet<String> = records
        .iter()
        .filter(|record| selected_string(record, "task_id", &selected_task))
        .filter(|record| answer_has_tool_marker(record))
        .filter_map(|record| record.get("trial_name").and_then(python_string))
        .collect();
    if let Some(trial_name) = trial_name {
        if !available_trials.contains(trial_name) {
            let available = if available_trials.is_empty() {
                "(none)".to_owned()
            } else {
                available_trials
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            };
            return Err(ReplayError::new(format!(
                "trial {} has no extractable tool calls for task {}; available: {available}",
                python_repr(trial_name),
                python_repr(&selected_task)
            )));
        }
    }

    let mut actions = Vec::new();
    for record in &records {
        if !selected_string(record, "task_id", &selected_task) {
            continue;
        }
        let Some(answer) = record.get("answer").and_then(Value::as_str) else {
            continue;
        };
        if let Some(trial_name) = trial_name {
            if !selected_string(record, "trial_name", trial_name) {
                continue;
            }
        }
        for marker in tool_markers(answer) {
            let source_tool = marker.raw_tool.to_lowercase();
            let tool = match source_tool.as_str() {
                "shell" | "bash" => "exec".to_owned(),
                _ => source_tool.clone(),
            };
            let payload = match decode_object_after(answer, marker.end) {
                Ok(payload) => payload,
                Err(error) => {
                    let line_number = record
                        .get("_line_number")
                        .and_then(Value::as_u64)
                        .unwrap_or(0);
                    warnings.push(format!(
                        "line {line_number}: failed to parse {} tool payload ({error})",
                        python_repr(&source_tool)
                    ));
                    continue;
                }
            };
            let mut action = Map::new();
            action.insert(
                "action_index".to_owned(),
                Value::Number(Number::from((actions.len() + 1) as u64)),
            );
            action.insert("tool".to_owned(), Value::String(tool.clone()));
            action.insert("source_tool".to_owned(), Value::String(source_tool));
            action.insert("payload".to_owned(), Value::Object(payload));
            action.insert(
                "source_line".to_owned(),
                record.get("_line_number").cloned().unwrap_or(Value::Null),
            );
            action.insert(
                "turn".to_owned(),
                record.get("turn").cloned().unwrap_or(Value::Null),
            );
            action.insert(
                "trial_name".to_owned(),
                record.get("trial_name").cloned().unwrap_or(Value::Null),
            );
            action.insert(
                "supported".to_owned(),
                Value::Bool(matches!(tool.as_str(), "exec" | "read" | "write")),
            );
            actions.push(action);
        }
    }

    if actions.is_empty() {
        return Err(ReplayError::new(format!(
            "no tool calls extracted for task {}",
            python_repr(&selected_task)
        )));
    }
    Ok(Extraction {
        task_id: selected_task,
        trial_name: trial_name.map(str::to_owned),
        actions,
        warnings,
    })
}

fn resolve_allow_missing(path: &Path) -> io::Result<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut pending: VecDeque<OsString> = absolute
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_os_string()),
            Component::ParentDir => Some(OsString::from("..")),
            _ => None,
        })
        .collect();
    let mut resolved = PathBuf::from("/");
    let mut links_followed = 0_u32;
    while let Some(component) = pending.pop_front() {
        if component == OsStr::new("..") {
            resolved.pop();
            continue;
        }
        resolved.push(&component);
        match fs::symlink_metadata(&resolved) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                links_followed += 1;
                if links_followed > 40 {
                    return Err(io::Error::from_raw_os_error(libc::ELOOP));
                }
                let target = fs::read_link(&resolved)?;
                resolved.pop();
                if target.is_absolute() {
                    resolved = PathBuf::from("/");
                }
                let target_components: Vec<_> = target
                    .components()
                    .filter_map(|component| match component {
                        Component::Normal(part) => Some(part.to_os_string()),
                        Component::ParentDir => Some(OsString::from("..")),
                        _ => None,
                    })
                    .collect();
                for component in target_components.into_iter().rev() {
                    pending.push_front(component);
                }
            }
            Ok(_) => {}
            Err(error) if error.kind() == io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(resolved)
}

fn map_workspace_path(
    captured_path: &str,
    workspace: &Path,
    source_workspace: Option<&Path>,
) -> Result<PathBuf, ReplayError> {
    let raw_path = Path::new(captured_path);
    let relative = if raw_path.is_absolute() {
        let explicit = source_workspace.and_then(|source| raw_path.strip_prefix(source).ok());
        if let Some(relative) = explicit {
            relative.to_path_buf()
        } else {
            let components: Vec<_> = raw_path.components().collect();
            if let Some(index) = components.iter().rposition(|component| {
                matches!(component, Component::Normal(part) if *part == OsStr::new("agent_workspace"))
            }) {
                components[index + 1..]
                    .iter()
                    .fold(PathBuf::new(), |mut path, component| {
                        path.push(component.as_os_str());
                        path
                    })
            } else {
                return Err(ReplayError::new(format!(
                    "absolute tool path is outside the captured agent workspace: {captured_path}"
                )));
            }
        }
    } else {
        raw_path.to_path_buf()
    };

    let resolved_workspace = fs::canonicalize(workspace)?;
    let target = resolve_allow_missing(&resolved_workspace.join(relative))?;
    if target.strip_prefix(&resolved_workspace).is_err() {
        return Err(ReplayError::new(format!(
            "tool path escapes replay workspace: {captured_path}"
        )));
    }
    Ok(target)
}

fn remap_command(command: &str, workspace: &Path, source_workspace: Option<&Path>) -> String {
    match source_workspace {
        Some(source) => {
            let source = source.as_os_str().to_string_lossy();
            let workspace = workspace.as_os_str().to_string_lossy();
            command.replace(source.as_ref(), workspace.as_ref())
        }
        None => command.to_owned(),
    }
}

fn output_preview(bytes: &[u8]) -> String {
    String::from_utf8_lossy(bytes)
        .chars()
        .take(PREVIEW_LENGTH)
        .collect()
}

fn spawn_pipe_reader<R>(mut reader: R) -> Receiver<io::Result<Vec<u8>>>
where
    R: Read + Send + 'static,
{
    let (sender, receiver) = mpsc::channel();
    thread::spawn(move || {
        let mut bytes = Vec::new();
        let result = reader.read_to_end(&mut bytes).map(|_| bytes);
        let _ = sender.send(result);
    });
    receiver
}

#[derive(Debug)]
struct ExecOutput {
    status: ExitStatus,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
    timed_out: bool,
}

fn kill_process_group(child: &mut Child) -> io::Result<()> {
    let pid = i32::try_from(child.id()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "child PID does not fit in pid_t",
        )
    })?;
    // The child calls setpgid(0, 0) before exec, so a negative PID targets the
    // entire replay action rather than leaving grandchildren behind.
    let result = unsafe { libc::kill(-pid, libc::SIGKILL) };
    if result == 0 {
        Ok(())
    } else {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

fn receive_pipe(receiver: Receiver<io::Result<Vec<u8>>>) -> io::Result<Vec<u8>> {
    receiver.recv().map_err(|_| {
        io::Error::new(
            io::ErrorKind::BrokenPipe,
            "subprocess output reader terminated unexpectedly",
        )
    })?
}

fn run_shell(command: &str, workspace: &Path, timeout_sec: f64) -> io::Result<ExecOutput> {
    if !timeout_sec.is_finite() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "timeout must be finite",
        ));
    }
    let timeout = if timeout_sec <= 0.0 {
        Duration::ZERO
    } else {
        Duration::from_secs_f64(timeout_sec)
    };
    let mut process = Command::new("/bin/bash");
    process
        .arg("-c")
        .arg(command)
        .current_dir(workspace)
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    unsafe {
        process.pre_exec(|| {
            if libc::setpgid(0, 0) == 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        });
    }
    let mut child = process.spawn()?;
    let stdout = spawn_pipe_reader(child.stdout.take().expect("stdout configured as piped"));
    let stderr = spawn_pipe_reader(child.stderr.take().expect("stderr configured as piped"));
    let started = Instant::now();
    let mut status = None;
    let mut stdout_result = None;
    let mut stderr_result = None;

    loop {
        if status.is_none() {
            status = child.try_wait()?;
        }
        if stdout_result.is_none() {
            match stdout.try_recv() {
                Ok(result) => stdout_result = Some(result),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    stdout_result = Some(Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "stdout reader disconnected",
                    )))
                }
            }
        }
        if stderr_result.is_none() {
            match stderr.try_recv() {
                Ok(result) => stderr_result = Some(result),
                Err(TryRecvError::Empty) => {}
                Err(TryRecvError::Disconnected) => {
                    stderr_result = Some(Err(io::Error::new(
                        io::ErrorKind::BrokenPipe,
                        "stderr reader disconnected",
                    )))
                }
            }
        }
        if status.is_some() && stdout_result.is_some() && stderr_result.is_some() {
            return Ok(ExecOutput {
                status: status.expect("checked above"),
                stdout: stdout_result.take().expect("checked above")?,
                stderr: stderr_result.take().expect("checked above")?,
                timed_out: false,
            });
        }
        if started.elapsed() >= timeout {
            let group_kill = kill_process_group(&mut child);
            let waited = match status {
                Some(status) => Ok(status),
                None => child.wait(),
            };
            if let Err(error) = group_kill {
                let _ = child.kill();
                let _ = child.wait();
                return Err(error);
            }
            let status = waited?;
            let stdout = match stdout_result {
                Some(result) => result?,
                None => receive_pipe(stdout)?,
            };
            let stderr = match stderr_result {
                Some(result) => result?,
                None => receive_pipe(stderr)?,
            };
            return Ok(ExecOutput {
                status,
                stdout,
                stderr,
                timed_out: true,
            });
        }
        let remaining = timeout.saturating_sub(started.elapsed());
        thread::sleep(POLL_INTERVAL.min(remaining));
    }
}

fn payload_object(action: &Map<String, Value>) -> Result<&Map<String, Value>, ReplayError> {
    action
        .get("payload")
        .and_then(Value::as_object)
        .ok_or_else(|| ReplayError::new("tool payload is not a mapping"))
}

fn optional_positive_integer(
    payload: &Map<String, Value>,
    field: &str,
) -> Result<Option<u64>, ReplayError> {
    let Some(value) = payload.get(field) else {
        return Ok(None);
    };
    if value.is_null() {
        return Ok(None);
    }
    let Some(value) = value.as_u64().filter(|value| *value >= 1) else {
        return Err(ReplayError::new(format!(
            "read tool payload '{field}' must be a positive integer"
        )));
    };
    Ok(Some(value))
}

fn relative_report_path(target: &Path, workspace: &Path) -> Result<String, ReplayError> {
    let workspace = fs::canonicalize(workspace)?;
    target
        .strip_prefix(workspace)
        .map(|relative| relative.to_string_lossy().into_owned())
        .map_err(|_| ReplayError::new("mapped tool path is outside replay workspace"))
}

fn run_action(
    action: &Map<String, Value>,
    workspace: &Path,
    source_workspace: Option<&Path>,
    timeout_sec: f64,
) -> Result<Map<String, Value>, ReplayError> {
    let tool = action
        .get("tool")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .unwrap_or_else(|| {
            action
                .get("tool")
                .cloned()
                .unwrap_or(Value::Null)
                .to_string()
        });
    let payload = payload_object(action)?;

    if tool == "exec" {
        let command = payload
            .get("command")
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ReplayError::new("exec tool payload is missing string field 'command'")
            })?;
        let command = remap_command(command, workspace, source_workspace);
        let completed = run_shell(&command, workspace, timeout_sec)?;
        let mut outcome = Map::new();
        outcome.insert(
            "status".to_owned(),
            Value::String(
                if completed.timed_out {
                    "timed_out"
                } else if completed.status.success() {
                    "ok"
                } else {
                    "failed"
                }
                .to_owned(),
            ),
        );
        outcome.insert("command".to_owned(), Value::String(command));
        if !completed.timed_out {
            let return_code = completed
                .status
                .code()
                .unwrap_or_else(|| -completed.status.signal().unwrap_or(0));
            outcome.insert(
                "returncode".to_owned(),
                Value::Number(Number::from(i64::from(return_code))),
            );
        }
        outcome.insert(
            "stdout_preview".to_owned(),
            Value::String(output_preview(&completed.stdout)),
        );
        outcome.insert(
            "stderr_preview".to_owned(),
            Value::String(output_preview(&completed.stderr)),
        );
        return Ok(outcome);
    }

    if tool == "write" {
        let captured_path = payload.get("path").and_then(Value::as_str);
        let content = payload.get("content").and_then(Value::as_str);
        let (Some(captured_path), Some(content)) = (captured_path, content) else {
            return Err(ReplayError::new(
                "write tool payload requires string 'path' and 'content'",
            ));
        };
        let target = map_workspace_path(captured_path, workspace, source_workspace)?;
        if let Some(parent) = target.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::write(&target, content.as_bytes())?;
        return Ok(Map::from_iter([
            ("status".to_owned(), Value::String("ok".to_owned())),
            (
                "path".to_owned(),
                Value::String(relative_report_path(&target, workspace)?),
            ),
            (
                "bytes_written".to_owned(),
                Value::Number(Number::from(content.len() as u64)),
            ),
        ]));
    }

    if tool == "read" {
        let captured_path = payload
            .get("path")
            .and_then(Value::as_str)
            .ok_or_else(|| ReplayError::new("read tool payload requires string field 'path'"))?;
        let target = map_workspace_path(captured_path, workspace, source_workspace)?;
        let offset = optional_positive_integer(payload, "offset")?;
        let limit = optional_positive_integer(payload, "limit")?;
        let data = if offset.is_none() && limit.is_none() {
            fs::read(&target)?
        } else {
            let first_line = offset.unwrap_or(1);
            let stop_line = limit.map(|limit| first_line.saturating_sub(1).saturating_add(limit));
            let mut reader = BufReader::new(File::open(&target)?);
            let mut data = Vec::new();
            let mut line = Vec::new();
            let mut line_number = 1_u64;
            loop {
                line.clear();
                let read = reader.read_until(b'\n', &mut line)?;
                if read == 0 {
                    break;
                }
                if line_number >= first_line
                    && stop_line.map(|stop| line_number <= stop).unwrap_or(true)
                {
                    data.extend_from_slice(&line);
                }
                if stop_line.map(|stop| line_number >= stop).unwrap_or(false) {
                    break;
                }
                line_number = line_number.saturating_add(1);
            }
            data
        };
        let mut outcome = Map::from_iter([
            ("status".to_owned(), Value::String("ok".to_owned())),
            (
                "path".to_owned(),
                Value::String(relative_report_path(&target, workspace)?),
            ),
            (
                "bytes_read".to_owned(),
                Value::Number(Number::from(data.len() as u64)),
            ),
        ]);
        if let Some(offset) = offset {
            outcome.insert("offset".to_owned(), Value::Number(Number::from(offset)));
        }
        if let Some(limit) = limit {
            outcome.insert("limit".to_owned(), Value::Number(Number::from(limit)));
        }
        return Ok(outcome);
    }

    Ok(Map::from_iter([
        ("status".to_owned(), Value::String("unsupported".to_owned())),
        (
            "reason".to_owned(),
            Value::String(format!("unsupported tool: {tool}")),
        ),
    ]))
}

fn copy_tree(source: &Path, destination: &Path) -> Result<(), ReplayError> {
    for entry in fs::read_dir(source)? {
        let entry = entry?;
        let source_path = entry.path();
        let destination_path = destination.join(entry.file_name());
        let metadata = fs::metadata(&source_path)?;
        if metadata.is_dir() {
            fs::create_dir_all(&destination_path)?;
            copy_tree(&source_path, &destination_path)?;
        } else if metadata.is_file() {
            fs::copy(&source_path, &destination_path)?;
        } else {
            return Err(ReplayError::new(format!(
                "unsupported initial workspace entry: {}",
                source_path.display()
            )));
        }
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
struct CpuUsage {
    user_sec: f64,
    system_sec: f64,
}

impl CpuUsage {
    fn difference(self, earlier: CpuUsage) -> CpuUsage {
        CpuUsage {
            user_sec: self.user_sec - earlier.user_sec,
            system_sec: self.system_sec - earlier.system_sec,
        }
    }

    fn total(self) -> f64 {
        self.user_sec + self.system_sec
    }
}

fn timeval_seconds(value: libc::timeval) -> f64 {
    value.tv_sec as f64 + value.tv_usec as f64 / 1_000_000.0
}

fn get_rusage(who: libc::c_int) -> io::Result<libc::rusage> {
    let mut usage = std::mem::MaybeUninit::<libc::rusage>::uninit();
    if unsafe { libc::getrusage(who, usage.as_mut_ptr()) } == 0 {
        Ok(unsafe { usage.assume_init() })
    } else {
        Err(io::Error::last_os_error())
    }
}

fn cpu_usage() -> Result<CpuUsage, ReplayError> {
    let own = get_rusage(libc::RUSAGE_SELF)?;
    let children = get_rusage(libc::RUSAGE_CHILDREN)?;
    Ok(CpuUsage {
        user_sec: timeval_seconds(own.ru_utime) + timeval_seconds(children.ru_utime),
        system_sec: timeval_seconds(own.ru_stime) + timeval_seconds(children.ru_stime),
    })
}

fn round_places(value: f64, places: i32) -> f64 {
    let scale = 10_f64.powi(places);
    (value * scale).round() / scale
}

fn json_number(value: f64) -> Value {
    Number::from_f64(value)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn read_cpu_stat(path: &Path) -> Result<BTreeMap<String, i64>, ReplayError> {
    let text = fs::read_to_string(path.join("cpu.stat"))?;
    let mut stats = BTreeMap::new();
    for line in text.lines() {
        let fields: Vec<_> = line.split_whitespace().collect();
        if fields.len() == 2 {
            let value = fields[1].parse::<i64>().map_err(|error| {
                ReplayError::new(format!("invalid cpu.stat value for {}: {error}", fields[0]))
            })?;
            stats.insert(fields[0].to_owned(), value);
        }
    }
    Ok(stats)
}

fn current_cgroup_path() -> Result<PathBuf, ReplayError> {
    for line in fs::read_to_string("/proc/self/cgroup")?.lines() {
        if let Some(relative) = line.strip_prefix("0::") {
            return Ok(Path::new("/sys/fs/cgroup").join(relative.trim_start_matches('/')));
        }
    }
    Err(ReplayError::new(
        "cgroup v2 membership not found in /proc/self/cgroup",
    ))
}

fn scheduler_affinity() -> Result<Vec<u64>, ReplayError> {
    let mut set = unsafe { std::mem::zeroed::<libc::cpu_set_t>() };
    let result = unsafe {
        libc::sched_getaffinity(
            0,
            std::mem::size_of::<libc::cpu_set_t>(),
            &mut set as *mut libc::cpu_set_t,
        )
    };
    if result != 0 {
        return Err(io::Error::last_os_error().into());
    }
    let mut cpus = Vec::new();
    for cpu in 0..libc::CPU_SETSIZE as usize {
        if unsafe { libc::CPU_ISSET(cpu, &set) } {
            cpus.push(cpu as u64);
        }
    }
    Ok(cpus)
}

struct CgroupCpuMeter {
    parent_path: PathBuf,
    path: PathBuf,
    active: bool,
    baseline_stats: Option<BTreeMap<String, i64>>,
    allowed_cpu_ids: Option<Vec<u64>>,
}

impl CgroupCpuMeter {
    fn new() -> Result<Self, ReplayError> {
        let parent_path = current_cgroup_path()?;
        let sequence = CGROUP_SEQUENCE.fetch_add(1, Ordering::Relaxed);
        let entropy = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64
            ^ sequence;
        let path = parent_path.join(format!(
            "offline-replayer-{}-{:08x}",
            std::process::id(),
            entropy & 0xffff_ffff
        ));
        Ok(Self {
            parent_path,
            path,
            active: false,
            baseline_stats: None,
            allowed_cpu_ids: None,
        })
    }

    fn start(&mut self) -> Result<(), ReplayError> {
        let result = (|| -> io::Result<()> {
            fs::create_dir(&self.path)?;
            fs::write(
                self.path.join("cgroup.procs"),
                std::process::id().to_string(),
            )?;
            Ok(())
        })();
        if let Err(error) = result {
            let _ = fs::remove_dir(&self.path);
            return Err(ReplayError::new(format!(
                "unable to create replay CPU cgroup: {error}"
            )));
        }
        self.active = true;
        Ok(())
    }

    fn begin_workload(&mut self) -> Result<(), ReplayError> {
        if !self.active {
            return Err(ReplayError::new("cgroup CPU meter was not started"));
        }
        self.allowed_cpu_ids = Some(scheduler_affinity()?);
        self.baseline_stats = Some(read_cpu_stat(&self.path)?);
        Ok(())
    }

    fn finish(&mut self, wall_time_sec: f64) -> Result<Value, ReplayError> {
        if !self.active || self.baseline_stats.is_none() {
            return Err(ReplayError::new(
                "cgroup CPU meter workload window was not started",
            ));
        }
        let final_stats = read_cpu_stat(&self.path)?;
        let baseline = self.baseline_stats.as_ref().expect("checked above");
        let stats: BTreeMap<_, _> = final_stats
            .iter()
            .map(|(key, value)| {
                (
                    key.clone(),
                    *value - baseline.get(key).copied().unwrap_or(0),
                )
            })
            .collect();
        fs::write(
            self.parent_path.join("cgroup.procs"),
            std::process::id().to_string(),
        )
        .map_err(|error| ReplayError::new(format!("unable to leave replay CPU cgroup: {error}")))?;

        let remaining_pids: Vec<u64> = fs::read_to_string(self.path.join("cgroup.procs"))?
            .split_whitespace()
            .filter_map(|pid| pid.parse::<u64>().ok())
            .collect();
        let cpuset_path = self.path.join("cpuset.cpus.effective");
        let allowed_cpu_list = if cpuset_path.exists() {
            Some(fs::read_to_string(cpuset_path)?.trim().to_owned())
        } else {
            None
        };
        let mut retained = !remaining_pids.is_empty();
        if !retained && fs::remove_dir(&self.path).is_err() {
            retained = true;
        }

        let usage_sec = stats.get("usage_usec").copied().unwrap_or(0) as f64 / 1_000_000.0;
        let average = (wall_time_sec > 0.0).then(|| round_places(usage_sec / wall_time_sec, 9));
        let percent =
            (wall_time_sec > 0.0).then(|| round_places(100.0 * usage_sec / wall_time_sec, 6));
        let allowed_ids = self.allowed_cpu_ids.clone();
        let mut result = Map::new();
        result.insert(
            "method".to_owned(),
            Value::String("cgroup-v2-cpu.stat".to_owned()),
        );
        result.insert(
            "scope_path".to_owned(),
            Value::String(self.path.to_string_lossy().into_owned()),
        );
        result.insert(
            "usage_sec".to_owned(),
            json_number(round_places(usage_sec, 9)),
        );
        result.insert(
            "user_sec".to_owned(),
            json_number(round_places(
                stats.get("user_usec").copied().unwrap_or(0) as f64 / 1_000_000.0,
                9,
            )),
        );
        result.insert(
            "system_sec".to_owned(),
            json_number(round_places(
                stats.get("system_usec").copied().unwrap_or(0) as f64 / 1_000_000.0,
                9,
            )),
        );
        result.insert(
            "average_logical_cores_used".to_owned(),
            average.map(json_number).unwrap_or(Value::Null),
        );
        result.insert(
            "logical_core_percent".to_owned(),
            percent.map(json_number).unwrap_or(Value::Null),
        );
        result.insert(
            "allowed_logical_cpu_list".to_owned(),
            allowed_cpu_list.map(Value::String).unwrap_or(Value::Null),
        );
        result.insert(
            "allowed_logical_cpu_count".to_owned(),
            allowed_ids
                .as_ref()
                .map(|ids| Value::Number(Number::from(ids.len() as u64)))
                .unwrap_or(Value::Null),
        );
        result.insert(
            "allowed_logical_cpu_ids".to_owned(),
            allowed_ids
                .map(|ids| Value::Array(ids.into_iter().map(|id| json!(id)).collect()))
                .unwrap_or(Value::Null),
        );
        result.insert(
            "allowed_logical_cpu_source".to_owned(),
            Value::String("sched_getaffinity".to_owned()),
        );
        result.insert("scope_retained".to_owned(), Value::Bool(retained));
        result.insert(
            "remaining_pids".to_owned(),
            Value::Array(remaining_pids.into_iter().map(|pid| json!(pid)).collect()),
        );
        result.insert(
            "meaning".to_owned(),
            Value::String(
                "Exact accumulated CPU time for work in this cgroup during the bounded replay window; it does not identify per-CPU residency."
                    .to_owned(),
            ),
        );
        if retained {
            result.insert(
                "warning".to_owned(),
                Value::String(
                    "The replay left processes in the CPU cgroup; reported CPU time does not include work that occurs after replay completion."
                        .to_owned(),
                ),
            );
        }
        self.active = false;
        Ok(Value::Object(result))
    }
}

/// Replay extracted actions and return the stable `offline-tool-replay.v1` report.
pub fn replay_actions(
    extraction: &Extraction,
    config: &ReplayConfig,
) -> Result<Value, ReplayError> {
    if config.workspace.exists() {
        if !config.workspace.is_dir() {
            return Err(ReplayError::new(format!(
                "replay workspace must be a directory: {}",
                config.workspace.display()
            )));
        }
        if fs::read_dir(&config.workspace)?
            .next()
            .transpose()?
            .is_some()
        {
            return Err(ReplayError::new(format!(
                "replay workspace must be empty: {}",
                config.workspace.display()
            )));
        }
    }
    fs::create_dir_all(&config.workspace)?;
    if let Some(initial_workspace) = &config.initial_workspace {
        if !initial_workspace.is_dir() {
            return Err(ReplayError::new(format!(
                "initial workspace does not exist: {}",
                initial_workspace.display()
            )));
        }
        copy_tree(initial_workspace, &config.workspace)?;
    }

    let mut cgroup_meter = if config.cgroup_cpu {
        let mut meter = CgroupCpuMeter::new()?;
        meter.start()?;
        if let Some(ready_file) = &config.cgroup_ready_file {
            fs::write(ready_file, format!("{}\n", meter.path.display()))?;
        }
        if config.cgroup_start_delay_sec > 0.0 {
            thread::sleep(Duration::from_secs_f64(config.cgroup_start_delay_sec));
        }
        meter.begin_workload()?;
        Some(meter)
    } else {
        None
    };

    let total_started = Instant::now();
    let total_cpu_before = cpu_usage()?;
    let mut results = Vec::<Value>::new();
    for action in &extraction.actions {
        let cpu_before = cpu_usage()?;
        let action_started = Instant::now();
        let action_start_monotonic_ns = timebase::monotonic_ns();
        let outcome = match run_action(
            action,
            &config.workspace,
            config.source_workspace.as_deref(),
            config.timeout_sec,
        ) {
            Ok(outcome) => outcome,
            Err(error) => Map::from_iter([
                ("status".to_owned(), Value::String("error".to_owned())),
                ("error".to_owned(), Value::String(error.to_string())),
            ]),
        };
        let wall_time_sec = action_started.elapsed().as_secs_f64();
        let action_end_monotonic_ns = timebase::monotonic_ns();
        let cpu_delta = cpu_usage()?.difference(cpu_before);
        let mut result = action.clone();
        result.extend(outcome);
        result.insert(
            "wall_time_sec".to_owned(),
            json_number(round_places(wall_time_sec, 9)),
        );
        result.insert(
            "replay_start_monotonic_ns".to_owned(),
            json!(action_start_monotonic_ns),
        );
        result.insert(
            "replay_end_monotonic_ns".to_owned(),
            json!(action_end_monotonic_ns),
        );
        result.insert(
            "user_cpu_sec".to_owned(),
            json_number(round_places(cpu_delta.user_sec, 9)),
        );
        result.insert(
            "system_cpu_sec".to_owned(),
            json_number(round_places(cpu_delta.system_sec, 9)),
        );
        result.insert(
            "cpu_time_sec".to_owned(),
            json_number(round_places(cpu_delta.total(), 9)),
        );
        let should_stop =
            config.strict && result.get("status").and_then(Value::as_str) != Some("ok");
        results.push(Value::Object(result));
        if should_stop {
            break;
        }
    }

    let total_wall_sec = total_started.elapsed().as_secs_f64();
    let total_cpu_delta = cpu_usage()?.difference(total_cpu_before);
    let cgroup_measurement = match cgroup_meter.as_mut() {
        Some(meter) => meter.finish(total_wall_sec)?,
        None => Value::Null,
    };
    let status_count = |status: &str| {
        results
            .iter()
            .filter(|result| result.get("status").and_then(Value::as_str) == Some(status))
            .count() as u64
    };
    let actions_failed = results
        .iter()
        .filter(|result| {
            matches!(
                result.get("status").and_then(Value::as_str),
                Some("failed" | "error" | "timed_out")
            )
        })
        .count() as u64;
    let actions_replayed = results.len();
    let actions_ok = status_count("ok");
    let actions_unsupported = status_count("unsupported");

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "task_id": extraction.task_id,
        "trial_name": extraction.trial_name,
        "workspace": config.workspace.to_string_lossy(),
        "initial_workspace": config.initial_workspace.as_ref().map(|path| path.to_string_lossy().into_owned()),
        "source_workspace": config.source_workspace.as_ref().map(|path| path.to_string_lossy().into_owned()),
        "extraction_warnings": extraction.warnings,
        "clock_reference": timebase::clock_reference(),
        "actions": results,
        "summary": {
            "actions_extracted": extraction.actions.len(),
            "actions_replayed": actions_replayed,
            "actions_ok": actions_ok,
            "actions_unsupported": actions_unsupported,
            "actions_failed": actions_failed,
            "wall_time_sec": round_places(total_wall_sec, 9),
            "user_cpu_sec": round_places(total_cpu_delta.user_sec, 9),
            "system_cpu_sec": round_places(total_cpu_delta.system_sec, 9),
            "cpu_time_sec": round_places(total_cpu_delta.total(), 9),
            "cgroup_cpu": cgroup_measurement,
        },
    }))
}

/// Write the compact JSONL action stream produced before replay begins.
pub fn write_extracted_actions(path: &Path, extraction: &Extraction) -> Result<(), ReplayError> {
    let mut output = BufWriter::new(File::create(path)?);
    let trace = json!({
        "record_type": "trace",
        "schema_version": SCHEMA_VERSION,
        "task_id": extraction.task_id,
        "trial_name": extraction.trial_name,
        "warnings": extraction.warnings,
    });
    serde_json::to_writer(&mut output, &trace)?;
    output.write_all(b"\n")?;
    for action in &extraction.actions {
        let mut record = Map::new();
        record.insert("record_type".to_owned(), Value::String("action".to_owned()));
        record.extend(action.clone());
        serde_json::to_writer(&mut output, &Value::Object(record))?;
        output.write_all(b"\n")?;
    }
    output.flush()?;
    Ok(())
}

/// Write a pretty, newline-terminated replay report.
pub fn write_replay_report(path: &Path, report: &Value) -> Result<(), ReplayError> {
    let mut output = BufWriter::new(File::create(path)?);
    serde_json::to_writer_pretty(&mut output, report)?;
    output.write_all(b"\n")?;
    output.flush()?;
    Ok(())
}

#[derive(Debug)]
struct CliOptions {
    transcript: PathBuf,
    task_id: Option<String>,
    trial_name: Option<String>,
    replay: ReplayConfig,
    output_dir: PathBuf,
}

#[derive(Debug)]
enum CliParse {
    Help,
    Run(Box<CliOptions>),
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

fn split_long_option(argument: &OsStr) -> Option<(&str, OsString)> {
    let value = argument.to_str()?;
    let (name, value) = value.split_once('=')?;
    name.starts_with("--")
        .then(|| (name, OsString::from(value)))
}

fn parse_float(option: &str, value: OsString) -> Result<f64, String> {
    let display = value.to_string_lossy();
    display
        .parse::<f64>()
        .map_err(|_| format!("argument {option}: invalid float value: {display:?}"))
}

fn default_output_dir(workspace: &Path) -> PathBuf {
    let name = workspace
        .file_name()
        .unwrap_or_else(|| OsStr::new(""))
        .to_string_lossy();
    let output_name = format!("{name}-output");
    workspace
        .parent()
        .unwrap_or_else(|| Path::new(""))
        .join(output_name)
}

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut task_id = None;
    let mut trial_name = None;
    let mut workspace = None;
    let mut initial_workspace = None;
    let mut source_workspace = None;
    let mut timeout_sec = 60.0;
    let mut strict = false;
    let mut cgroup_cpu = false;
    let mut cgroup_ready_file = None;
    let mut cgroup_start_delay_sec = 0.0;
    let mut output_dir = None;
    let mut positional = Vec::new();
    let mut options = true;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if options && matches!(argument.to_str(), Some("-h" | "--help")) {
            return Ok(CliParse::Help);
        }
        if options && argument.as_os_str() == OsStr::new("--") {
            options = false;
            index += 1;
            continue;
        }
        let inline = options.then(|| split_long_option(argument)).flatten();
        let option_name = inline
            .as_ref()
            .map(|(name, _)| *name)
            .or_else(|| options.then(|| argument.to_str()).flatten());
        let inline_value = inline.map(|(_, value)| value);
        let value_for = |index: &mut usize, name: &str| {
            inline_value
                .clone()
                .map(Ok)
                .unwrap_or_else(|| option_value(arguments, index, name))
        };
        match option_name {
            Some("--task-id") => {
                task_id = Some(
                    value_for(&mut index, "--task-id")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            Some("--trial-name") => {
                trial_name = Some(
                    value_for(&mut index, "--trial-name")?
                        .to_string_lossy()
                        .into_owned(),
                )
            }
            Some("--workspace") => {
                workspace = Some(PathBuf::from(value_for(&mut index, "--workspace")?))
            }
            Some("--initial-workspace") => {
                initial_workspace =
                    Some(PathBuf::from(value_for(&mut index, "--initial-workspace")?))
            }
            Some("--source-workspace") => {
                source_workspace = Some(PathBuf::from(value_for(&mut index, "--source-workspace")?))
            }
            Some("--timeout-sec") => {
                timeout_sec = parse_float("--timeout-sec", value_for(&mut index, "--timeout-sec")?)?
            }
            Some("--strict") if inline_value.is_none() => strict = true,
            Some("--cgroup-cpu") if inline_value.is_none() => cgroup_cpu = true,
            Some("--cgroup-ready-file") => {
                cgroup_ready_file =
                    Some(PathBuf::from(value_for(&mut index, "--cgroup-ready-file")?))
            }
            Some("--cgroup-start-delay-sec") => {
                cgroup_start_delay_sec = parse_float(
                    "--cgroup-start-delay-sec",
                    value_for(&mut index, "--cgroup-start-delay-sec")?,
                )?
            }
            Some("--output-dir") => {
                output_dir = Some(PathBuf::from(value_for(&mut index, "--output-dir")?))
            }
            Some(option) if options && option.starts_with('-') => {
                return Err(format!(
                    "unrecognized arguments: {}",
                    argument.to_string_lossy()
                ))
            }
            _ => positional.push(PathBuf::from(argument.as_os_str())),
        }
        index += 1;
    }

    if positional.is_empty() {
        return Err("the following arguments are required: transcript".to_owned());
    }
    if positional.len() > 1 {
        return Err(format!(
            "unrecognized arguments: {}",
            positional[1..]
                .iter()
                .map(|path| path.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    let Some(workspace) = workspace else {
        return Err("the following arguments are required: --workspace".to_owned());
    };
    if (cgroup_ready_file.is_some() || cgroup_start_delay_sec != 0.0) && !cgroup_cpu {
        return Err(
            "--cgroup-ready-file and --cgroup-start-delay-sec require --cgroup-cpu".to_owned(),
        );
    }
    if cgroup_start_delay_sec < 0.0 {
        return Err("--cgroup-start-delay-sec must not be negative".to_owned());
    }
    let output_dir = output_dir.unwrap_or_else(|| default_output_dir(&workspace));
    let mut replay = ReplayConfig::new(workspace);
    replay.source_workspace = source_workspace;
    replay.initial_workspace = initial_workspace;
    replay.timeout_sec = timeout_sec;
    replay.strict = strict;
    replay.cgroup_cpu = cgroup_cpu;
    replay.cgroup_ready_file = cgroup_ready_file;
    replay.cgroup_start_delay_sec = cgroup_start_delay_sec;
    Ok(CliParse::Run(Box::new(CliOptions {
        transcript: positional.remove(0),
        task_id,
        trial_name,
        replay,
        output_dir,
    })))
}

fn print_help(public_program: &str) {
    println!("Extract transcript tool calls and replay supported CPU-side actions.");
    println!();
    println!("usage: {public_program} TRANSCRIPT --workspace WORKSPACE [options]");
    println!();
    println!("positional arguments:");
    println!("  TRANSCRIPT                    Transcript JSONL to import.");
    println!();
    println!("options:");
    println!("  -h, --help                    Show this help message and exit.");
    println!("  --task-id TASK_ID             Task ID when input includes multiple tasks.");
    println!("  --trial-name TRIAL_NAME       Replay one recorded run for the task.");
    println!("  --workspace WORKSPACE         New or empty replay workspace (required).");
    println!("  --initial-workspace PATH      Contents copied before action timing.");
    println!("  --source-workspace PATH       Captured workspace prefix to remap.");
    println!("  --timeout-sec SECONDS         Per-exec timeout (default: 60.0).");
    println!("  --strict                      Stop on the first non-ok action.");
    println!("  --cgroup-cpu                  Use a temporary cgroup v2 CPU scope.");
    println!("  --cgroup-ready-file PATH      Publish the temporary cgroup path.");
    println!("  --cgroup-start-delay-sec SEC  Wait before the measured workload.");
    println!("  --output-dir OUTPUT_DIR       Report output directory.");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    eprintln!("usage: {public_program} TRANSCRIPT --workspace WORKSPACE [options]");
    eprintln!("{public_program}: error: {message}");
    2
}

fn required_u64(value: &Value, pointer: &str) -> u64 {
    value.pointer(pointer).and_then(Value::as_u64).unwrap_or(0)
}

fn required_f64(value: &Value, pointer: &str) -> f64 {
    value
        .pointer(pointer)
        .and_then(Value::as_f64)
        .unwrap_or(0.0)
}

/// Native CLI entry point used by the top-level dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let options = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(options)) => *options,
        Err(error) => return cli_error(public_program, &error),
    };
    let result = (|| -> Result<Value, ReplayError> {
        let extraction = extract_actions(
            &options.transcript,
            options.task_id.as_deref(),
            options.trial_name.as_deref(),
        )?;
        fs::create_dir_all(&options.output_dir)?;
        write_extracted_actions(
            &options.output_dir.join("extracted_actions.jsonl"),
            &extraction,
        )?;
        let report = replay_actions(&extraction, &options.replay)?;
        write_replay_report(&options.output_dir.join("replay_report.json"), &report)?;
        Ok(report)
    })();
    let report = match result {
        Ok(report) => report,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };

    println!("Task: {}", report["task_id"].as_str().unwrap_or(""));
    if let Some(trial_name) = report["trial_name"].as_str() {
        println!("Trial: {trial_name}");
    }
    println!(
        "Actions: {}/{} replayed, {} ok, {} unsupported, {} failed",
        required_u64(&report, "/summary/actions_replayed"),
        required_u64(&report, "/summary/actions_extracted"),
        required_u64(&report, "/summary/actions_ok"),
        required_u64(&report, "/summary/actions_unsupported"),
        required_u64(&report, "/summary/actions_failed"),
    );
    println!(
        "CPU time: {:.9} sec",
        required_f64(&report, "/summary/cpu_time_sec")
    );
    println!(
        "Wall time: {:.9} sec",
        required_f64(&report, "/summary/wall_time_sec")
    );
    if !report["summary"]["cgroup_cpu"].is_null() {
        let cgroup = &report["summary"]["cgroup_cpu"];
        println!(
            "Cgroup CPU: {:.9} sec, average logical cores used: {:.6}",
            cgroup["usage_sec"].as_f64().unwrap_or(0.0),
            cgroup["average_logical_cores_used"].as_f64().unwrap_or(0.0),
        );
        if let Some(warning) = cgroup["warning"].as_str() {
            println!("Warning: {warning}");
        }
    }
    let report_path = options.output_dir.join("replay_report.json");
    println!("Report: {}", report_path.display());
    if options.replay.strict
        && (required_u64(&report, "/summary/actions_unsupported") > 0
            || required_u64(&report, "/summary/actions_failed") > 0)
    {
        1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_jsonl(path: &Path, records: &[Value]) {
        let mut output = BufWriter::new(File::create(path).unwrap());
        for record in records {
            serde_json::to_writer(&mut output, record).unwrap();
            output.write_all(b"\n").unwrap();
        }
        output.flush().unwrap();
    }

    fn action_payload(extraction: &Extraction, index: usize) -> &Value {
        extraction.actions[index].get("payload").unwrap()
    }

    #[test]
    fn extraction_filters_task_trial_and_normalizes_shell_aliases() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        let records = [
            json!({
                "task_id": "ignored",
                "answer": "[Tool Call: exec]\n{\"command\":\"false\"}"
            }),
            json!({
                "task_id": "calendar",
                "trial_name": "calendar.1",
                "turn": 2,
                "answer": "[Tool Call: Bash]\n{\"command\":\"printf first\"}"
            }),
            json!({
                "task_id": "calendar",
                "trial_name": "calendar.2",
                "turn": 3,
                "answer": concat!(
                    "[Tool Call: shell]\n{\"command\":\"printf second\"}\n",
                    "[Tool Call: write] {\"path\":\"event.ics\",\"content\":\"calendar\"}"
                )
            }),
        ];
        write_jsonl(&transcript, &records);

        let extraction =
            extract_actions(&transcript, Some("calendar"), Some("calendar.2")).unwrap();
        assert_eq!(extraction.task_id, "calendar");
        assert_eq!(extraction.trial_name.as_deref(), Some("calendar.2"));
        assert_eq!(extraction.actions.len(), 2);
        assert_eq!(extraction.actions[0]["tool"], "exec");
        assert_eq!(extraction.actions[0]["source_tool"], "shell");
        assert_eq!(action_payload(&extraction, 0)["command"], "printf second");
        assert_eq!(extraction.actions[1]["tool"], "write");
        assert_eq!(extraction.actions[1]["action_index"], 2);
        assert_eq!(extraction.actions[1]["source_line"], 3);
    }

    #[test]
    fn extraction_retains_lossy_import_warnings_and_skips_bad_payloads() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        fs::write(
            &transcript,
            concat!(
                "not json\n",
                "{\"task_id\":\"task\",\"answer\":\"[Tool Call: read] []\"}\n",
                "{\"task_id\":\"task\",\"answer\":\"[Tool Call: exec] {\\\"command\\\":\\\"true\\\"}\"}\n"
            ),
        )
        .unwrap();

        let extraction = extract_actions(&transcript, Some("task"), None).unwrap();
        assert_eq!(extraction.actions.len(), 1);
        assert_eq!(extraction.actions[0]["tool"], "exec");
        assert_eq!(extraction.warnings.len(), 2);
        assert!(extraction.warnings[0].starts_with("line 1: invalid JSON skipped"));
        assert!(extraction.warnings[1].contains(
            "failed to parse 'read' tool payload (tool-call payload is not a JSON object)"
        ));
    }

    #[test]
    fn extraction_requires_unambiguous_task_and_valid_trial() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        write_jsonl(
            &transcript,
            &[
                json!({"task_id":"a", "answer":"[Tool Call: exec] {\"command\":\"true\"}"}),
                json!({"task_id":"b", "trial_name":"b.1", "answer":"[Tool Call: exec] {\"command\":\"true\"}"}),
            ],
        );
        assert!(extract_actions(&transcript, None, None)
            .unwrap_err()
            .to_string()
            .contains("provide --task-id"));
        assert!(extract_actions(&transcript, Some("missing"), None)
            .unwrap_err()
            .to_string()
            .contains("available: a, b"));
        assert!(extract_actions(&transcript, Some("b"), Some("b.2"))
            .unwrap_err()
            .to_string()
            .contains("available: b.1"));
    }

    #[test]
    fn workspace_mapping_handles_agent_workspace_and_rejects_escapes() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let mapped = map_workspace_path(
            "/tmp/pinchbench/one/agent_workspace/nested/file.txt",
            &workspace,
            None,
        )
        .unwrap();
        assert_eq!(
            mapped,
            fs::canonicalize(&workspace)
                .unwrap()
                .join("nested/file.txt")
        );
        let error = map_workspace_path("../outside", &workspace, None).unwrap_err();
        assert!(error.to_string().contains("escapes replay workspace"));
        let error = map_workspace_path("/etc/passwd", &workspace, None).unwrap_err();
        assert!(error
            .to_string()
            .contains("outside the captured agent workspace"));
    }

    #[test]
    fn replay_exec_write_read_and_unsupported_preserve_report_schema() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        let source_workspace = Path::new("/captured/agent_workspace");
        write_jsonl(
            &transcript,
            &[
                json!({
                    "task_id":"task",
                    "answer":"[Tool Call: exec] {\"command\":\"printf generated > /captured/agent_workspace/generated.txt\"}"
                }),
                json!({
                    "task_id":"task",
                    "answer":"[Tool Call: write] {\"path\":\"/captured/agent_workspace/input.txt\",\"content\":\"first\\nlonger-second\\nthird\\n\"}"
                }),
                json!({
                    "task_id":"task",
                    "answer":"[Tool Call: read] {\"path\":\"input.txt\",\"offset\":2,\"limit\":1}"
                }),
                json!({
                    "task_id":"task",
                    "answer":"[Tool Call: browser] {\"action\":\"open\"}"
                }),
            ],
        );
        let extraction = extract_actions(&transcript, Some("task"), None).unwrap();
        let workspace = temporary.path().join("workspace");
        let mut config = ReplayConfig::new(&workspace);
        config.source_workspace = Some(source_workspace.to_path_buf());
        let report = replay_actions(&extraction, &config).unwrap();

        assert_eq!(report["schema_version"], SCHEMA_VERSION);
        assert_eq!(report["summary"]["actions_extracted"], 4);
        assert_eq!(report["summary"]["actions_replayed"], 4);
        assert_eq!(report["summary"]["actions_ok"], 3);
        assert_eq!(report["summary"]["actions_unsupported"], 1);
        assert_eq!(report["summary"]["actions_failed"], 0);
        assert_eq!(report["actions"][2]["bytes_read"], 14);
        assert_eq!(report["actions"][2]["offset"], 2);
        assert_eq!(report["actions"][2]["limit"], 1);
        assert_eq!(
            fs::read_to_string(workspace.join("generated.txt")).unwrap(),
            "generated"
        );
        assert_eq!(report["clock_reference"]["clock_id"], "CLOCK_MONOTONIC");
        for action in report["actions"].as_array().unwrap() {
            assert!(
                action["replay_end_monotonic_ns"].as_u64().unwrap()
                    >= action["replay_start_monotonic_ns"].as_u64().unwrap()
            );
            assert!(action["cpu_time_sec"].as_f64().unwrap() >= 0.0);
        }
    }

    #[test]
    fn replay_replaces_non_utf8_output_and_limits_preview_by_characters() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        write_jsonl(
            &transcript,
            &[json!({
                "task_id":"binary",
                "answer":"[Tool Call: exec] {\"command\":\"printf '\\\\377'\"}"
            })],
        );
        let extraction = extract_actions(&transcript, Some("binary"), None).unwrap();
        let report = replay_actions(
            &extraction,
            &ReplayConfig::new(temporary.path().join("workspace")),
        )
        .unwrap();
        assert_eq!(report["actions"][0]["status"], "ok");
        assert_eq!(report["actions"][0]["stdout_preview"], "�");
        serde_json::to_string(&report).unwrap();
    }

    #[test]
    fn strict_replay_stops_after_first_non_ok_action() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        write_jsonl(
            &transcript,
            &[
                json!({"task_id":"strict", "answer":"[Tool Call: browser] {\"action\":\"open\"}"}),
                json!({"task_id":"strict", "answer":"[Tool Call: exec] {\"command\":\"touch should-not-exist\"}"}),
            ],
        );
        let extraction = extract_actions(&transcript, Some("strict"), None).unwrap();
        let workspace = temporary.path().join("workspace");
        let mut config = ReplayConfig::new(&workspace);
        config.strict = true;
        let report = replay_actions(&extraction, &config).unwrap();
        assert_eq!(report["summary"]["actions_replayed"], 1);
        assert_eq!(report["summary"]["actions_unsupported"], 1);
        assert!(!workspace.join("should-not-exist").exists());
    }

    #[test]
    fn timeout_kills_the_entire_process_group_and_retains_partial_output() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        write_jsonl(
            &transcript,
            &[json!({
                "task_id":"slow",
                "answer": concat!(
                    "[Tool Call: exec] {\"command\":\"printf partial; ",
                    "(sleep 0.20; touch escaped) & wait\"}"
                )
            })],
        );
        let extraction = extract_actions(&transcript, Some("slow"), None).unwrap();
        let workspace = temporary.path().join("workspace");
        let mut config = ReplayConfig::new(&workspace);
        config.timeout_sec = 0.02;
        let report = replay_actions(&extraction, &config).unwrap();
        assert_eq!(report["actions"][0]["status"], "timed_out");
        assert_eq!(report["actions"][0]["stdout_preview"], "partial");
        thread::sleep(Duration::from_millis(300));
        assert!(!workspace.join("escaped").exists());
    }

    #[test]
    fn initial_workspace_is_seeded_before_actions_and_must_exist() {
        let temporary = tempfile::tempdir().unwrap();
        let initial = temporary.path().join("initial");
        fs::create_dir(&initial).unwrap();
        fs::write(initial.join("input.txt"), "payload").unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        write_jsonl(
            &transcript,
            &[json!({"task_id":"seed", "answer":"[Tool Call: read] {\"path\":\"input.txt\"}"})],
        );
        let extraction = extract_actions(&transcript, Some("seed"), None).unwrap();
        let mut config = ReplayConfig::new(temporary.path().join("workspace"));
        config.initial_workspace = Some(initial);
        let report = replay_actions(&extraction, &config).unwrap();
        assert_eq!(report["actions"][0]["bytes_read"], 7);

        let mut bad = ReplayConfig::new(temporary.path().join("other-workspace"));
        bad.initial_workspace = Some(temporary.path().join("missing"));
        assert!(replay_actions(&extraction, &bad)
            .unwrap_err()
            .to_string()
            .contains("initial workspace does not exist"));
    }

    #[test]
    fn writers_emit_parseable_trace_and_pretty_report() {
        let temporary = tempfile::tempdir().unwrap();
        let transcript = temporary.path().join("trace.jsonl");
        write_jsonl(
            &transcript,
            &[json!({"task_id":"task", "answer":"[Tool Call: exec] {\"command\":\"true\"}"})],
        );
        let extraction = extract_actions(&transcript, Some("task"), None).unwrap();
        let actions_path = temporary.path().join("actions.jsonl");
        write_extracted_actions(&actions_path, &extraction).unwrap();
        let lines: Vec<_> = fs::read_to_string(actions_path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str::<Value>(line).unwrap())
            .collect();
        assert_eq!(lines[0]["record_type"], "trace");
        assert_eq!(lines[1]["record_type"], "action");

        let report_path = temporary.path().join("report.json");
        let report = json!({"schema_version": SCHEMA_VERSION});
        write_replay_report(&report_path, &report).unwrap();
        let text = fs::read_to_string(report_path).unwrap();
        assert!(text.ends_with('\n'));
        assert!(text.contains("\n  \"schema_version\""));
    }

    #[test]
    fn cpu_stat_parser_and_cli_validation_match_contract() {
        let temporary = tempfile::tempdir().unwrap();
        fs::write(
            temporary.path().join("cpu.stat"),
            "usage_usec 5000000\nuser_usec 4000000\nsystem_usec 1000000\n",
        )
        .unwrap();
        assert_eq!(
            read_cpu_stat(temporary.path()).unwrap(),
            BTreeMap::from([
                ("system_usec".to_owned(), 1_000_000),
                ("usage_usec".to_owned(), 5_000_000),
                ("user_usec".to_owned(), 4_000_000),
            ])
        );

        let arguments = [
            OsString::from("trace.jsonl"),
            OsString::from("--workspace"),
            OsString::from("workspace"),
            OsString::from("--cgroup-ready-file"),
            OsString::from("ready"),
        ];
        assert!(parse_cli(&arguments)
            .unwrap_err()
            .contains("require --cgroup-cpu"));
    }
}
