// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native renderer for validated scheduler-concurrency report batches.
//!
//! The JSON, CSV, and SVG artifacts intentionally retain the Python public
//! schemas.  PNG rendering is implemented with a small deterministic raster
//! backend so the public `--png` surface does not retain a matplotlib/Python
//! runtime dependency.

use flate2::write::ZlibEncoder;
use flate2::Compression;
use regex::Regex;
use serde_json::{Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

pub const SUMMARY_SCHEMA_VERSION: &str = "scheduler-batch-summary.v1";
const CORE_COLORS: [&str; 9] = [
    "#e5e7eb", "#2563eb", "#14b8a6", "#84cc16", "#f59e0b", "#f97316", "#ef4444", "#a855f7",
    "#111827",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct BatchReportError(String);

impl BatchReportError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for BatchReportError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for BatchReportError {}

#[derive(Clone, Debug, PartialEq)]
pub struct TaskSummary {
    pub task_name: String,
    pub source_trial: String,
    pub report_path: PathBuf,
    pub wall_sec: f64,
    pub cpu_usage_sec: f64,
    pub average_logical_cores_used: f64,
    pub scheduler_average_logical_cores_used: f64,
    pub peak_simultaneous_logical_cores: i64,
    pub allocated_logical_cpus: f64,
    pub time_sec_by_cores: BTreeMap<i64, f64>,
    pub source_outcome_mismatch_count: i64,
    pub lost_perf_events: i64,
    pub replay_report_path: Option<PathBuf>,
    pub scheduler_events_path: Option<PathBuf>,
}

impl TaskSummary {
    fn utilization_percent_of_allocation(&self) -> f64 {
        self.average_logical_cores_used / self.allocated_logical_cpus * 100.0
    }

    fn time_above_one_core_sec(&self) -> f64 {
        self.time_sec_by_cores
            .iter()
            .filter(|(cores, _)| **cores > 1)
            .map(|(_, seconds)| seconds)
            .sum()
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ShellCommandSummary {
    pub task_name: String,
    pub action_index: i64,
    pub session_id: String,
    pub poll_count: usize,
    pub command: String,
    pub command_preview: String,
    pub command_family: String,
    pub lifecycle_start_monotonic_ns: i64,
    pub lifecycle_end_monotonic_ns: i64,
    pub lifecycle_wall_sec: f64,
    pub status: String,
    pub source_status: String,
    pub peak_simultaneous_logical_cores: i64,
    pub average_simultaneous_logical_cores: f64,
    pub time_at_or_above_2_cores_sec: f64,
    pub time_at_or_above_4_cores_sec: f64,
    pub time_at_or_above_6_cores_sec: f64,
    pub time_at_or_above_8_cores_sec: f64,
    time_at_or_above_2_has_bucket: bool,
    time_at_or_above_4_has_bucket: bool,
    time_at_or_above_6_has_bucket: bool,
    time_at_or_above_8_has_bucket: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CommandFrequencySummary {
    pub command: String,
    pub command_preview: String,
    pub invocation_count: usize,
    pub task_count: usize,
    pub task_names: Vec<String>,
    pub total_lifecycle_wall_sec: f64,
    pub total_time_at_or_above_2_cores_sec: f64,
    pub max_peak_simultaneous_logical_cores: i64,
    total_time_at_or_above_2_has_bucket: bool,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ToolFamilySummary {
    pub command_family: String,
    pub invocation_count: usize,
    pub invocation_percent_of_commands: f64,
    pub task_count: usize,
    pub task_names: Vec<String>,
    pub summed_invocation_lifecycle_wall_sec: f64,
    pub active_lifecycle_wall_sec: f64,
    pub overlap_collapsed_wall_sec: f64,
    pub active_lifecycle_percent_of_replay_wall: f64,
    pub active_time_at_or_above_2_cores_sec: f64,
    pub active_time_at_or_above_4_cores_sec: f64,
    pub active_time_at_or_above_8_cores_sec: f64,
    pub max_peak_simultaneous_logical_cores: i64,
    pub command_examples: Vec<String>,
}

#[derive(Clone, Debug)]
struct SchedulerEvent {
    timestamp_ns: i64,
    cpu: i64,
    entering: bool,
}

fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|number| number != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn py_string(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) | Some(Value::Bool(false)) => String::new(),
        Some(Value::Bool(true)) => "True".to_owned(),
        Some(Value::Number(value)) if value.as_f64() == Some(0.0) => String::new(),
        Some(Value::String(value)) if value.is_empty() => String::new(),
        Some(Value::Array(value)) if value.is_empty() => String::new(),
        Some(Value::Object(value)) if value.is_empty() => String::new(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

fn py_display(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_owned(),
        Some(Value::Bool(true)) => "True".to_owned(),
        Some(Value::Bool(false)) => "False".to_owned(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

fn number(value: Option<&Value>, field: &str, path: &Path) -> Result<f64, BatchReportError> {
    let number = value
        .filter(|value| !value.is_boolean())
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite())
        .ok_or_else(|| {
            BatchReportError::new(format!(
                "{} has invalid numeric field '{field}'",
                path.display()
            ))
        })?;
    Ok(number)
}

fn integer(value: Option<&Value>, field: &str) -> Result<i64, BatchReportError> {
    let value = value.ok_or_else(|| BatchReportError::new(format!("missing {field}")))?;
    if let Some(value) = value.as_bool() {
        return Ok(i64::from(value));
    }
    if let Some(value) = value.as_i64() {
        return Ok(value);
    }
    if let Some(value) = value.as_u64().and_then(|value| i64::try_from(value).ok()) {
        return Ok(value);
    }
    if let Some(value) = value.as_f64().filter(|value| value.is_finite()) {
        if value >= i64::MIN as f64 && value <= i64::MAX as f64 {
            return Ok(value.trunc() as i64);
        }
    }
    if let Some(value) = value
        .as_str()
        .and_then(|value| value.trim().parse::<i64>().ok())
    {
        return Ok(value);
    }
    Err(BatchReportError::new(format!(
        "invalid integer field {field}"
    )))
}

fn absolute_path(path: &Path) -> Result<PathBuf, BatchReportError> {
    if let Ok(path) = fs::canonicalize(path) {
        return Ok(path);
    }
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| {
                BatchReportError::new(format!("unable to resolve {}: {error}", path.display()))
            })?
            .join(path)
    };
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                normalized.pop();
            }
            other => normalized.push(other.as_os_str()),
        }
    }
    Ok(normalized)
}

fn read_json_object(path: &Path, kind: &str) -> Result<Map<String, Value>, BatchReportError> {
    let text = fs::read_to_string(path).map_err(|error| {
        BatchReportError::new(format!("unable to read {kind} {}: {error}", path.display()))
    })?;
    let value: Value = serde_json::from_str(&text).map_err(|error| {
        BatchReportError::new(format!("unable to read {kind} {}: {error}", path.display()))
    })?;
    value.as_object().cloned().ok_or_else(|| {
        BatchReportError::new(format!("{kind} is not an object: {}", path.display()))
    })
}

pub fn load_summary(path: &Path) -> Result<TaskSummary, BatchReportError> {
    let payload = read_json_object(path, "pipeline report")?;
    if payload
        .get("status")
        .is_some_and(|status| status.as_str() != Some("valid"))
    {
        return Err(BatchReportError::new(format!(
            "pipeline report is not valid for profiling: {}",
            path.display()
        )));
    }
    let replay = payload.get("replay_summary").and_then(Value::as_object);
    let scheduler = payload
        .get("scheduler_concurrency")
        .and_then(Value::as_object);
    let cpu = payload.get("container_cpu").and_then(Value::as_object);
    let (replay, scheduler, cpu) = match (replay, scheduler, cpu) {
        (Some(replay), Some(scheduler), Some(cpu)) => (replay, scheduler, cpu),
        _ => {
            return Err(BatchReportError::new(format!(
                "pipeline report lacks replay, CPU, or scheduler data: {}",
                path.display()
            )))
        }
    };
    let mismatch = replay.get("source_outcome_mismatch_count");
    let mismatch_is_zero = matches!(mismatch, Some(Value::Bool(false)))
        || mismatch.and_then(Value::as_f64) == Some(0.0);
    if !mismatch_is_zero {
        return Err(BatchReportError::new(format!(
            "pipeline report has {} replay outcome mismatches: {}",
            py_display(mismatch),
            path.display()
        )));
    }
    if replay
        .get("source_outcome_comparison_unavailable_count")
        .is_some_and(|count| count.as_u64() != Some(0))
    {
        return Err(BatchReportError::new(format!(
            "pipeline report has replay source outcome comparisons unavailable: {}",
            path.display()
        )));
    }
    for field in [
        "actions_timed_out",
        "actions_launch_failed",
        "actions_source_argument_parse_failed",
        "actions_skipped_after_timeout",
    ] {
        if replay
            .get(field)
            .is_some_and(|count| count.as_u64() != Some(0))
        {
            return Err(BatchReportError::new(format!(
                "pipeline replay {field} is nonzero or invalid: {}",
                path.display()
            )));
        }
    }
    if let Some(recorded) = replay.get("tool_actions_recorded") {
        if !recorded.as_u64().is_some_and(|count| {
            count > 0 && replay.get("tool_actions_replayed").and_then(Value::as_u64) == Some(count)
        }) {
            return Err(BatchReportError::new(format!(
                "pipeline replay tool actions are incomplete or invalid: {}",
                path.display()
            )));
        }
    }
    let valid = scheduler
        .get("cgroup_validation")
        .and_then(Value::as_object)
        .and_then(|validation| validation.get("valid_for_concurrency_reporting"))
        .is_some_and(py_truthy);
    if !valid {
        return Err(BatchReportError::new(format!(
            "scheduler concurrency is not cgroup-validated: {}",
            path.display()
        )));
    }
    let distribution = scheduler
        .get("time_sec_by_simultaneous_logical_cores")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            BatchReportError::new(format!(
                "pipeline report lacks scheduler distribution: {}",
                path.display()
            ))
        })?;
    let mut time_sec_by_cores = BTreeMap::new();
    for (key, value) in distribution {
        let cores = key.parse::<i64>().map_err(|_| {
            BatchReportError::new(format!(
                "{} has invalid concurrency bucket {key:?}",
                path.display()
            ))
        })?;
        time_sec_by_cores.insert(
            cores,
            number(Some(value), &format!("distribution[{key}]"), path)?,
        );
    }
    let referenced_path = |value: Option<&Value>| -> Result<Option<PathBuf>, BatchReportError> {
        match value.filter(|value| py_truthy(value)) {
            Some(value) => absolute_path(Path::new(&py_string(Some(value)))).map(Some),
            None => Ok(None),
        }
    };
    Ok(TaskSummary {
        task_name: py_string(payload.get("task_name")),
        source_trial: py_string(payload.get("source_trial")),
        report_path: absolute_path(path)?,
        wall_sec: number(cpu.get("wall_sec"), "container_cpu.wall_sec", path)?,
        cpu_usage_sec: number(cpu.get("usage_sec"), "container_cpu.usage_sec", path)?,
        average_logical_cores_used: number(
            cpu.get("average_logical_cores_used"),
            "container_cpu.average_logical_cores_used",
            path,
        )?,
        scheduler_average_logical_cores_used: number(
            scheduler.get("average_logical_cores_used"),
            "scheduler_concurrency.average_logical_cores_used",
            path,
        )?,
        peak_simultaneous_logical_cores: number(
            scheduler.get("peak_simultaneous_logical_cores"),
            "scheduler_concurrency.peak_simultaneous_logical_cores",
            path,
        )? as i64,
        allocated_logical_cpus: number(
            cpu.get("allocated_logical_cpus"),
            "container_cpu.allocated_logical_cpus",
            path,
        )?,
        time_sec_by_cores,
        source_outcome_mismatch_count: 0,
        lost_perf_events: match scheduler
            .get("lost_perf_events")
            .filter(|value| py_truthy(value))
        {
            Some(value) => integer(Some(value), "scheduler_concurrency.lost_perf_events")?,
            None => 0,
        },
        replay_report_path: referenced_path(payload.get("replay_report"))?,
        scheduler_events_path: referenced_path(scheduler.get("events_path"))?,
    })
}

pub fn build_summary(paths: &[PathBuf]) -> Result<Vec<TaskSummary>, BatchReportError> {
    if paths.is_empty() {
        return Err(BatchReportError::new(
            "provide at least one pipeline report",
        ));
    }
    let mut tasks = paths
        .iter()
        .map(|path| load_summary(path))
        .collect::<Result<Vec<_>, _>>()?;
    let mut counts = BTreeMap::new();
    for task in &tasks {
        *counts.entry(task.task_name.clone()).or_insert(0usize) += 1;
    }
    let duplicates = counts
        .into_iter()
        .filter_map(|(name, count)| (count > 1).then_some(name))
        .collect::<Vec<_>>();
    if !duplicates.is_empty() {
        return Err(BatchReportError::new(format!(
            "duplicate task reports: {}",
            duplicates.join(", ")
        )));
    }
    tasks.sort_by(|left, right| left.task_name.cmp(&right.task_name));
    Ok(tasks)
}

fn load_scheduler_events(task: &TaskSummary) -> Result<Vec<SchedulerEvent>, BatchReportError> {
    let path = task.scheduler_events_path.as_ref().ok_or_else(|| {
        BatchReportError::new(format!("{} lacks scheduler-events path", task.task_name))
    })?;
    let text = fs::read_to_string(path).map_err(|error| {
        BatchReportError::new(format!(
            "unable to read scheduler events for {}: {error}",
            task.task_name
        ))
    })?;
    let mut events = Vec::new();
    for line in text.lines().filter(|line| !line.trim().is_empty()) {
        let event: Value = serde_json::from_str(line).map_err(|error| {
            BatchReportError::new(format!(
                "unable to read scheduler events for {}: {error}",
                task.task_name
            ))
        })?;
        let event = event.as_object().ok_or_else(|| {
            BatchReportError::new(format!(
                "unable to read scheduler events for {}: event is not an object",
                task.task_name
            ))
        })?;
        events.push(SchedulerEvent {
            timestamp_ns: integer(event.get("timestamp_ns"), "timestamp_ns")?,
            cpu: integer(event.get("cpu"), "cpu")?,
            entering: event
                .get("entering")
                .map(py_truthy)
                .ok_or_else(|| BatchReportError::new("missing entering"))?,
        });
    }
    events.sort_by_key(|event| event.timestamp_ns);
    Ok(events)
}

fn load_replay_actions(task: &TaskSummary) -> Result<Vec<Map<String, Value>>, BatchReportError> {
    let path = task.replay_report_path.as_ref().ok_or_else(|| {
        BatchReportError::new(format!("{} lacks replay-report path", task.task_name))
    })?;
    let replay = read_json_object(path, "replay report").map_err(|error| {
        BatchReportError::new(format!(
            "unable to read replay report for {}: {error}",
            task.task_name
        ))
    })?;
    let raw_actions = match replay.get("actions") {
        None => &[][..],
        Some(Value::Array(actions)) => actions.as_slice(),
        Some(_) => {
            return Err(BatchReportError::new(format!(
                "{} replay actions are not a list",
                task.task_name
            )))
        }
    };
    if raw_actions.iter().any(|action| !action.is_object()) {
        return Err(BatchReportError::new(format!(
            "{} replay report contains a non-object action",
            task.task_name
        )));
    }
    let actions = raw_actions
        .iter()
        .filter_map(Value::as_object)
        .filter(|action| {
            action.get("status").and_then(Value::as_str) != Some("ignored_metadata")
                && action.contains_key("replay_start_monotonic_ns")
                && action.contains_key("replay_end_monotonic_ns")
        })
        .cloned()
        .collect::<Vec<_>>();
    if actions.is_empty() {
        return Err(BatchReportError::new(format!(
            "{} replay report has no timed executable actions",
            task.task_name
        )));
    }
    Ok(actions)
}

fn integrate_event_window(
    events: &[SchedulerEvent],
    window_start_ns: i64,
    window_end_ns: i64,
) -> BTreeMap<i64, f64> {
    if window_end_ns <= window_start_ns {
        return BTreeMap::from([(0, 0.0)]);
    }
    let mut active_cpus = BTreeSet::new();
    for event in events
        .iter()
        .take_while(|event| event.timestamp_ns < window_start_ns)
    {
        if event.entering {
            active_cpus.insert(event.cpu);
        } else {
            active_cpus.remove(&event.cpu);
        }
    }
    let mut duration_ns_by_cores = BTreeMap::<i64, i128>::new();
    let mut cursor_ns = window_start_ns;
    for event in events {
        if event.timestamp_ns < window_start_ns {
            continue;
        }
        if event.timestamp_ns > window_end_ns {
            break;
        }
        if event.timestamp_ns > cursor_ns {
            *duration_ns_by_cores
                .entry(active_cpus.len() as i64)
                .or_default() += i128::from(event.timestamp_ns) - i128::from(cursor_ns);
        }
        if event.entering {
            active_cpus.insert(event.cpu);
        } else {
            active_cpus.remove(&event.cpu);
        }
        cursor_ns = event.timestamp_ns;
    }
    if window_end_ns > cursor_ns {
        *duration_ns_by_cores
            .entry(active_cpus.len() as i64)
            .or_default() += i128::from(window_end_ns) - i128::from(cursor_ns);
    }
    duration_ns_by_cores
        .into_iter()
        .map(|(cores, duration)| (cores, duration as f64 / 1_000_000_000.0))
        .collect()
}

fn normalized_command(command: &str) -> String {
    command.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn command_preview(action: &Map<String, Value>) -> (String, String) {
    let command = py_string(action.get("command"));
    if command.is_empty() {
        return (String::new(), String::new());
    }
    let command = normalized_command(&command);
    let preview = command
        .chars()
        .take(110)
        .collect::<String>()
        .trim_end()
        .to_owned();
    (command, preview)
}

fn shell_tokens(command: &str) -> Result<Vec<String>, ()> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut chars = command.chars().peekable();
    let mut quote = None;
    let mut token_started = false;
    while let Some(character) = chars.next() {
        if let Some(active_quote) = quote {
            if character == active_quote {
                quote = None;
            } else if character == '\\' && active_quote == '"' {
                match chars.peek().copied() {
                    Some('"' | '\\' | '$' | '`' | '\n') => {
                        current.push(chars.next().expect("peeked shell escape"));
                    }
                    _ => current.push(character),
                }
            } else {
                current.push(character);
            }
            continue;
        }
        match character {
            '\'' | '"' => {
                quote = Some(character);
                token_started = true;
            }
            '\\' => match chars.next() {
                Some(escaped) => {
                    current.push(escaped);
                    token_started = true;
                }
                None => return Err(()),
            },
            character if character.is_whitespace() => {
                if token_started {
                    tokens.push(std::mem::take(&mut current));
                    token_started = false;
                }
            }
            '|' | '&' | ';' => {
                if token_started {
                    tokens.push(std::mem::take(&mut current));
                    token_started = false;
                }
                let mut punctuation = String::from(character);
                while chars
                    .peek()
                    .is_some_and(|character| matches!(character, '|' | '&' | ';'))
                {
                    punctuation.push(chars.next().expect("peeked punctuation"));
                }
                tokens.push(punctuation);
            }
            other => {
                current.push(other);
                token_started = true;
            }
        }
    }
    if quote.is_some() {
        return Err(());
    }
    if token_started {
        tokens.push(current);
    }
    Ok(tokens)
}

fn shell_segments(command: &str) -> (Vec<Vec<String>>, Vec<String>) {
    let heredoc = Regex::new(r#"(?m)^.*<<-?\s*['"]?[A-Za-z_][A-Za-z0-9_]*['"]?.*$"#)
        .expect("static heredoc regex");
    let command = heredoc
        .find(command)
        .map_or(command, |matched| &command[..matched.end()]);
    let command = command.replace('\n', ";");
    let Ok(tokens) = shell_tokens(&command) else {
        return (
            vec![command.split_whitespace().map(str::to_owned).collect()],
            Vec::new(),
        );
    };
    let mut segments = Vec::new();
    let mut operators = Vec::new();
    let mut current = Vec::new();
    for token in tokens {
        if !token.is_empty()
            && token
                .chars()
                .all(|character| matches!(character, '|' | '&' | ';'))
        {
            if !current.is_empty() {
                segments.push(std::mem::take(&mut current));
            }
            operators.push(token);
        } else {
            current.push(token);
        }
    }
    if !current.is_empty() {
        segments.push(current);
    }
    (segments, operators)
}

fn segment_family(mut tokens: Vec<String>) -> String {
    if tokens.is_empty() {
        return String::new();
    }
    let assignment = Regex::new(r"^[A-Za-z_][A-Za-z0-9_]*=").expect("static assignment regex");
    while tokens
        .first()
        .is_some_and(|token| assignment.is_match(token))
    {
        tokens.remove(0);
    }
    if tokens.is_empty() {
        return "environment assignment".to_owned();
    }
    if matches!(tokens[0].as_str(), "time" | "/usr/bin/time") {
        let mut index = 1;
        while tokens
            .get(index)
            .is_some_and(|token| token.starts_with('-'))
        {
            index += 1;
        }
        tokens.drain(..index);
        if tokens.is_empty() {
            return "time".to_owned();
        }
    }
    let executable = Path::new(&tokens[0])
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("")
        .to_owned();
    let rest = &tokens[1..];
    if executable == "set" {
        return "set".to_owned();
    }
    let python_version = Regex::new(r"^python\d+(?:\.\d+)?$").expect("static Python regex");
    if executable == "python" || executable == "python3" || python_version.is_match(&executable) {
        if rest.len() >= 3 && rest[0] == "-m" && rest[1] == "pip" && rest[2] == "install" {
            return "python -m pip install".to_owned();
        }
        if rest.len() >= 2 && rest[0] == "-m" && rest[1] == "pytest" {
            return "python -m pytest".to_owned();
        }
        if rest.len() >= 2 && rest[0] == "-m" && rest[1] == "py_compile" {
            return "python -m py_compile".to_owned();
        }
        if rest.len() >= 2
            && Path::new(&rest[0])
                .file_name()
                .and_then(|name| name.to_str())
                == Some("setup.py")
            && rest[1] == "build_ext"
        {
            return "python setup.py build_ext".to_owned();
        }
        if rest.first().is_some_and(|value| value == "-") {
            return "python inline script".to_owned();
        }
        if rest.first().is_some_and(|value| value.ends_with(".py")) {
            let name = Path::new(&rest[0])
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&rest[0]);
            return format!("python {name}");
        }
        return "python".to_owned();
    }
    if matches!(executable.as_str(), "pip" | "pip3")
        && rest.first().is_some_and(|value| value == "install")
    {
        return "pip install".to_owned();
    }
    if executable == "go"
        && rest.first().is_some_and(|value| {
            matches!(
                value.as_str(),
                "build" | "fmt" | "generate" | "install" | "run" | "test" | "vet"
            )
        })
    {
        return format!("go {}", rest[0]);
    }
    if executable == "cargo"
        && rest.first().is_some_and(|value| {
            matches!(value.as_str(), "bench" | "build" | "check" | "run" | "test")
        })
    {
        return format!("cargo {}", rest[0]);
    }
    if executable == "npm" {
        if rest.first().is_some_and(|value| value == "run") && rest.len() > 1 {
            return format!("npm run {}", rest[1]);
        }
        if rest
            .first()
            .is_some_and(|value| matches!(value.as_str(), "ci" | "install" | "test"))
        {
            return format!("npm {}", rest[0]);
        }
    }
    if executable == "git" && !rest.is_empty() {
        return format!("git {}", rest[0]);
    }
    if executable == "node" {
        if rest.first().is_some_and(|value| value == "--check") {
            return "node --check".to_owned();
        }
        if rest.first().is_some_and(|value| value.ends_with(".js")) {
            let name = Path::new(&rest[0])
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&rest[0]);
            return format!("node {name}");
        }
        return "node".to_owned();
    }
    if executable == "uv" && rest.len() >= 2 && rest[0] == "run" && rest[1] == "python" {
        if rest.get(2).is_some_and(|value| value.ends_with(".py")) {
            let name = Path::new(&rest[2])
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(&rest[2]);
            return format!("uv run python {name}");
        }
        return "uv run python".to_owned();
    }
    if matches!(executable.as_str(), "g++" | "gcc" | "clang" | "clang++") {
        return format!("{executable} compile");
    }
    if executable == "vim" {
        return "vim batch edit".to_owned();
    }
    if matches!(
        executable.as_str(),
        "make" | "cmake" | "valgrind" | "sqlite3" | "pytest"
    ) {
        return executable;
    }
    executable
}

pub fn classify_command_family(command: &str) -> String {
    if normalized_command(command).is_empty() {
        return "<empty>".to_owned();
    }
    let (segments, operators) = shell_segments(command);
    let families = segments
        .into_iter()
        .map(segment_family)
        .filter(|family| !family.is_empty())
        .collect::<Vec<_>>();
    const HOUSEKEEPING: [&str; 12] = [
        ":", "cd", "chmod", "cp", "false", "mkdir", "printf", "rm", "set", "touch", "true", "",
    ];
    let meaningful = families
        .iter()
        .filter(|family| !HOUSEKEEPING.contains(&family.as_str()))
        .cloned()
        .collect::<Vec<_>>();
    let selected = if meaningful.is_empty() {
        families
    } else {
        meaningful
    };
    let mut unique = Vec::new();
    for family in selected {
        if !unique.contains(&family) {
            unique.push(family);
        }
    }
    if unique.is_empty() {
        return "<unclassified>".to_owned();
    }
    if unique.len() == 1 {
        return unique.remove(0);
    }
    if !operators.is_empty() && operators.iter().all(|operator| operator == "|") {
        return format!("{} pipeline", unique[0]);
    }
    let suffix = if unique.len() > 3 { " + ..." } else { "" };
    format!(
        "multi-tool: {}{suffix}",
        unique[..unique.len().min(3)].join(" + ")
    )
}

pub fn build_command_summary(
    tasks: &[TaskSummary],
) -> Result<Vec<ShellCommandSummary>, BatchReportError> {
    let mut rows = Vec::new();
    for task in tasks {
        let events = load_scheduler_events(task)?;
        let actions = load_replay_actions(task)?;
        for action in &actions {
            if action.get("tool").and_then(Value::as_str) != Some("exec_command") {
                continue;
            }
            let start_ns = integer(
                action.get("replay_start_monotonic_ns"),
                "replay_start_monotonic_ns",
            )?;
            let mut end_ns = integer(
                action.get("replay_end_monotonic_ns"),
                "replay_end_monotonic_ns",
            )?;
            let session_id = py_string(action.get("session_id"));
            let action_index = integer(action.get("action_index"), "action_index")?;
            let mut polls = Vec::new();
            for followup in &actions {
                if !session_id.is_empty()
                    && followup.get("tool").and_then(Value::as_str) == Some("write_stdin")
                    && py_string(followup.get("session_id")) == session_id
                    && integer(followup.get("action_index"), "action_index")? > action_index
                {
                    polls.push(followup);
                }
            }
            for poll in &polls {
                end_ns = end_ns.max(integer(
                    poll.get("replay_end_monotonic_ns"),
                    "replay_end_monotonic_ns",
                )?);
            }
            let durations = integrate_event_window(&events, start_ns, end_ns);
            let wall_sec =
                ((i128::from(end_ns) - i128::from(start_ns)) as f64 / 1_000_000_000.0).max(0.0);
            let scheduled_sec: f64 = durations
                .iter()
                .map(|(cores, seconds)| *cores as f64 * seconds)
                .sum();
            let (command, preview) = command_preview(action);
            let terminal = polls.last().copied().unwrap_or(action);
            let sum_at = |threshold: i64| {
                durations
                    .iter()
                    .filter(|(cores, _)| **cores >= threshold)
                    .map(|(_, seconds)| seconds)
                    .sum()
            };
            let has_at = |threshold: i64| durations.keys().any(|cores| *cores >= threshold);
            rows.push(ShellCommandSummary {
                task_name: task.task_name.clone(),
                action_index,
                session_id,
                poll_count: polls.len(),
                command,
                command_preview: preview,
                command_family: classify_command_family(&py_string(action.get("command"))),
                lifecycle_start_monotonic_ns: start_ns,
                lifecycle_end_monotonic_ns: end_ns,
                lifecycle_wall_sec: wall_sec,
                status: py_string(terminal.get("status")),
                source_status: py_string(terminal.get("source_status")),
                peak_simultaneous_logical_cores: *durations.keys().max().unwrap_or(&0),
                average_simultaneous_logical_cores: if wall_sec == 0.0 {
                    0.0
                } else {
                    scheduled_sec / wall_sec
                },
                time_at_or_above_2_cores_sec: sum_at(2),
                time_at_or_above_4_cores_sec: sum_at(4),
                time_at_or_above_6_cores_sec: sum_at(6),
                time_at_or_above_8_cores_sec: sum_at(8),
                time_at_or_above_2_has_bucket: has_at(2),
                time_at_or_above_4_has_bucket: has_at(4),
                time_at_or_above_6_has_bucket: has_at(6),
                time_at_or_above_8_has_bucket: has_at(8),
            });
        }
    }
    Ok(rows)
}

fn merge_intervals(intervals: &[(i64, i64)]) -> Vec<(i64, i64)> {
    let mut intervals = intervals.to_vec();
    intervals.sort();
    let mut merged: Vec<(i64, i64)> = Vec::new();
    for (start, end) in intervals {
        if merged
            .last()
            .map_or(true, |(_, previous_end)| start > *previous_end)
        {
            merged.push((start, end));
        } else if let Some(last) = merged.last_mut() {
            last.1 = last.1.max(end);
        }
    }
    merged
}

pub fn build_tool_family_summary(
    tasks: &[TaskSummary],
    commands: &[ShellCommandSummary],
) -> Result<Vec<ToolFamilySummary>, BatchReportError> {
    let mut grouped: BTreeMap<String, Vec<&ShellCommandSummary>> = BTreeMap::new();
    for command in commands {
        grouped
            .entry(command.command_family.clone())
            .or_default()
            .push(command);
    }
    let task_by_name = tasks
        .iter()
        .map(|task| (task.task_name.as_str(), task))
        .collect::<BTreeMap<_, _>>();
    let total_wall_sec: f64 = tasks.iter().map(|task| task.wall_sec).sum();
    let mut summaries = Vec::new();
    for (family, invocations) in grouped {
        let mut intervals_by_task: BTreeMap<&str, Vec<(i64, i64)>> = BTreeMap::new();
        for invocation in &invocations {
            intervals_by_task
                .entry(&invocation.task_name)
                .or_default()
                .push((
                    invocation.lifecycle_start_monotonic_ns,
                    invocation.lifecycle_end_monotonic_ns,
                ));
        }
        let summed_invocation_sec: f64 =
            invocations.iter().map(|item| item.lifecycle_wall_sec).sum();
        let mut active_sec = 0.0;
        let mut burst_2_sec = 0.0;
        let mut burst_4_sec = 0.0;
        let mut burst_8_sec = 0.0;
        let mut peak = 0;
        for (task_name, intervals) in &intervals_by_task {
            let task = task_by_name.get(task_name).ok_or_else(|| {
                BatchReportError::new(format!("missing task summary for {task_name}"))
            })?;
            let events = load_scheduler_events(task)?;
            for (start, end) in merge_intervals(intervals) {
                active_sec += (i128::from(end) - i128::from(start)) as f64 / 1_000_000_000.0;
                let durations = integrate_event_window(&events, start, end);
                peak = peak.max(*durations.keys().max().unwrap_or(&0));
                burst_2_sec += durations
                    .iter()
                    .filter(|(cores, _)| **cores >= 2)
                    .map(|(_, seconds)| seconds)
                    .sum::<f64>();
                burst_4_sec += durations
                    .iter()
                    .filter(|(cores, _)| **cores >= 4)
                    .map(|(_, seconds)| seconds)
                    .sum::<f64>();
                burst_8_sec += durations
                    .iter()
                    .filter(|(cores, _)| **cores >= 8)
                    .map(|(_, seconds)| seconds)
                    .sum::<f64>();
            }
        }
        let mut ranked = invocations.clone();
        ranked.sort_by(|left, right| {
            right
                .lifecycle_wall_sec
                .partial_cmp(&left.lifecycle_wall_sec)
                .unwrap_or(Ordering::Equal)
        });
        let mut examples = Vec::new();
        for invocation in ranked {
            if !examples.contains(&invocation.command_preview) {
                examples.push(invocation.command_preview.clone());
            }
            if examples.len() == 3 {
                break;
            }
        }
        summaries.push(ToolFamilySummary {
            command_family: family,
            invocation_count: invocations.len(),
            invocation_percent_of_commands: if commands.is_empty() {
                0.0
            } else {
                invocations.len() as f64 / commands.len() as f64 * 100.0
            },
            task_count: intervals_by_task.len(),
            task_names: intervals_by_task
                .keys()
                .map(|name| (*name).to_owned())
                .collect(),
            summed_invocation_lifecycle_wall_sec: summed_invocation_sec,
            active_lifecycle_wall_sec: active_sec,
            overlap_collapsed_wall_sec: (summed_invocation_sec - active_sec).max(0.0),
            active_lifecycle_percent_of_replay_wall: if total_wall_sec == 0.0 {
                0.0
            } else {
                active_sec / total_wall_sec * 100.0
            },
            active_time_at_or_above_2_cores_sec: burst_2_sec,
            active_time_at_or_above_4_cores_sec: burst_4_sec,
            active_time_at_or_above_8_cores_sec: burst_8_sec,
            max_peak_simultaneous_logical_cores: peak,
            command_examples: examples,
        });
    }
    summaries.sort_by(|left, right| {
        right
            .active_lifecycle_wall_sec
            .partial_cmp(&left.active_lifecycle_wall_sec)
            .unwrap_or(Ordering::Equal)
            .then_with(|| right.invocation_count.cmp(&left.invocation_count))
            .then_with(|| left.command_family.cmp(&right.command_family))
    });
    Ok(summaries)
}

pub fn build_command_frequency(commands: &[ShellCommandSummary]) -> Vec<CommandFrequencySummary> {
    let mut grouped: BTreeMap<String, Vec<&ShellCommandSummary>> = BTreeMap::new();
    for command in commands {
        grouped
            .entry(command.command.clone())
            .or_default()
            .push(command);
    }
    let mut rows = grouped
        .into_iter()
        .map(|(command, invocations)| {
            let task_names = invocations
                .iter()
                .map(|invocation| invocation.task_name.clone())
                .collect::<BTreeSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            CommandFrequencySummary {
                command_preview: command
                    .chars()
                    .take(110)
                    .collect::<String>()
                    .trim_end()
                    .to_owned(),
                command,
                invocation_count: invocations.len(),
                task_count: task_names.len(),
                task_names,
                total_lifecycle_wall_sec: invocations
                    .iter()
                    .map(|invocation| invocation.lifecycle_wall_sec)
                    .sum(),
                total_time_at_or_above_2_cores_sec: invocations
                    .iter()
                    .map(|invocation| invocation.time_at_or_above_2_cores_sec)
                    .sum(),
                max_peak_simultaneous_logical_cores: invocations
                    .iter()
                    .map(|invocation| invocation.peak_simultaneous_logical_cores)
                    .max()
                    .unwrap_or(0),
                total_time_at_or_above_2_has_bucket: invocations
                    .iter()
                    .any(|invocation| invocation.time_at_or_above_2_has_bucket),
            }
        })
        .collect::<Vec<_>>();
    rows.sort_by(|left, right| {
        right
            .invocation_count
            .cmp(&left.invocation_count)
            .then_with(|| {
                right
                    .total_lifecycle_wall_sec
                    .partial_cmp(&left.total_lifecycle_wall_sec)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left.command.cmp(&right.command))
    });
    rows
}

fn html_escape(value: &str) -> String {
    let mut escaped = String::new();
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' => escaped.push_str("&quot;"),
            '\'' => escaped.push_str("&#x27;"),
            other => escaped.push(other),
        }
    }
    escaped
}

fn svg_text(x: f64, y: f64, text: &str, attributes: &[(&str, &str)]) -> String {
    let attributes = attributes
        .iter()
        .map(|(key, value)| format!("{}=\"{}\"", key.replace('_', "-"), html_escape(value)))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<text x=\"{x:.1}\" y=\"{y:.1}\" {attributes}>{}</text>",
        html_escape(text)
    )
}

fn write_text(path: &Path, text: String) -> Result<(), BatchReportError> {
    fs::write(path, text).map_err(|error| {
        BatchReportError::new(format!("unable to write {}: {error}", path.display()))
    })
}

fn render_average_vs_peak(tasks: &[TaskSummary], output: &Path) -> Result<(), BatchReportError> {
    let allocation = tasks
        .iter()
        .map(|task| task.allocated_logical_cpus)
        .reduce(f64::max)
        .ok_or_else(|| BatchReportError::new("provide at least one pipeline report"))?;
    if !allocation.is_finite() || allocation <= 0.0 {
        return Err(BatchReportError::new(
            "allocated logical CPUs must be positive for chart rendering",
        ));
    }
    let (left, top, chart_width, row_height) = (245i64, 70i64, 680i64, 42i64);
    let width = 1080;
    let height = 70 + 42 * tasks.len() as i64 + 74;
    let scale = chart_width as f64 / allocation;
    let mut svg = vec![
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"),
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>".to_owned(),
        svg_text(22.0, 30.0, "Average CPU Use vs Peak Simultaneous Logical CPUs", &[("fill", "#111827"), ("font_size", "19"), ("font_weight", "bold")]),
        svg_text(22.0, 53.0, "Validated one-at-a-time BCC replays; blue bar = average, red marker = observed peak", &[("fill", "#4b5563"), ("font_size", "12")]),
    ];
    for tick in 0..=(allocation as i64) {
        let x = left as f64 + tick as f64 * scale;
        svg.push(format!(
            "<line x1=\"{x:.1}\" y1=\"{}\" x2=\"{x:.1}\" y2=\"{}\" stroke=\"#e5e7eb\"/>",
            top - 14,
            height - 50
        ));
        svg.push(svg_text(
            x - 3.0,
            height as f64 - 28.0,
            &tick.to_string(),
            &[("fill", "#374151"), ("font_size", "11")],
        ));
    }
    for (index, task) in tasks.iter().enumerate() {
        let center_y = top + index as i64 * row_height + 15;
        svg.push(svg_text(
            22.0,
            (center_y + 4) as f64,
            &task.task_name,
            &[("fill", "#111827"), ("font_size", "12")],
        ));
        svg.push(format!(
            "<rect x=\"{left}\" y=\"{}\" width=\"{:.1}\" height=\"19\" rx=\"3\" fill=\"#2563eb\"/>",
            center_y - 10,
            task.average_logical_cores_used * scale
        ));
        let peak_x = left as f64 + task.peak_simultaneous_logical_cores as f64 * scale;
        svg.push(format!(
            "<circle cx=\"{peak_x:.1}\" cy=\"{}\" r=\"6\" fill=\"#dc2626\"/>",
            center_y - 1
        ));
        svg.push(svg_text(
            (left + chart_width + 17) as f64,
            (center_y + 4) as f64,
            &format!(
                "avg {:.2}  peak {}",
                task.average_logical_cores_used, task.peak_simultaneous_logical_cores
            ),
            &[("fill", "#374151"), ("font_size", "11")],
        ));
    }
    svg.push(svg_text(
        (left + chart_width / 2 - 45) as f64,
        height as f64 - 8.0,
        "Logical CPUs",
        &[("fill", "#374151"), ("font_size", "12")],
    ));
    svg.push("</svg>\n".to_owned());
    write_text(output, svg.join("\n"))
}

fn render_distribution(tasks: &[TaskSummary], output: &Path) -> Result<(), BatchReportError> {
    let max_bucket = tasks
        .iter()
        .filter_map(|task| task.time_sec_by_cores.keys().max().copied())
        .max()
        .ok_or_else(|| BatchReportError::new("pipeline report lacks scheduler distribution"))?;
    if max_bucket < 0 {
        return Err(BatchReportError::new(
            "scheduler distribution has no nonnegative concurrency bucket",
        ));
    }
    let (left, top, chart_width, row_height) = (245.0, 90.0, 700.0, 43.0);
    let width = 1110;
    let height = 90 + 43 * tasks.len() as i64 + 86;
    let mut svg = vec![
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">"),
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>".to_owned(),
        svg_text(22.0, 30.0, "Per-Task Simultaneous Logical-CPU Distribution", &[("fill", "#111827"), ("font_size", "19"), ("font_weight", "bold")]),
        svg_text(22.0, 52.0, "Each bar is 100% of measured wall time; segments indicate concurrent scheduled logical CPUs", &[("fill", "#4b5563"), ("font_size", "12")]),
    ];
    let mut legend_x = 22.0;
    for cores in 0..=max_bucket {
        let color = CORE_COLORS[(cores as usize).min(CORE_COLORS.len() - 1)];
        svg.push(format!(
            "<rect x=\"{legend_x}\" y=\"65\" width=\"13\" height=\"13\" fill=\"{color}\"/>"
        ));
        svg.push(svg_text(
            legend_x + 18.0,
            76.0,
            &cores.to_string(),
            &[("fill", "#374151"), ("font_size", "11")],
        ));
        legend_x += 42.0;
    }
    for (index, task) in tasks.iter().enumerate() {
        let y = top + index as f64 * row_height;
        svg.push(svg_text(
            22.0,
            y + 19.0,
            &task.task_name,
            &[("fill", "#111827"), ("font_size", "12")],
        ));
        let total: f64 = task.time_sec_by_cores.values().sum();
        let mut offset = 0.0;
        for cores in 0..=max_bucket {
            let value = task.time_sec_by_cores.get(&cores).copied().unwrap_or(0.0);
            let segment_width = if total == 0.0 {
                0.0
            } else {
                value / total * chart_width
            };
            if segment_width > 0.0 {
                let color = CORE_COLORS[(cores as usize).min(CORE_COLORS.len() - 1)];
                svg.push(format!(
                    "<rect x=\"{:.2}\" y=\"{}\" width=\"{segment_width:.2}\" height=\"22\" fill=\"{color}\"/>",
                    left + offset,
                    y + 4.0
                ));
            }
            offset += segment_width;
        }
        svg.push(svg_text(
            left + chart_width + 15.0,
            y + 19.0,
            &format!(">1: {:.1} ms", task.time_above_one_core_sec() * 1000.0),
            &[("fill", "#374151"), ("font_size", "11")],
        ));
    }
    svg.push(svg_text(
        left + chart_width / 2.0 - 40.0,
        height as f64 - 14.0,
        "Wall time",
        &[("fill", "#374151"), ("font_size", "12")],
    ));
    svg.push("</svg>\n".to_owned());
    write_text(output, svg.join("\n"))
}

fn json_number(value: f64) -> Result<Value, BatchReportError> {
    serde_json::Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| BatchReportError::new("report contains a non-finite calculated value"))
}

fn json_string(value: &str) -> String {
    let mut rendered = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => rendered.push_str("\\\""),
            '\\' => rendered.push_str("\\\\"),
            '\u{0008}' => rendered.push_str("\\b"),
            '\u{000c}' => rendered.push_str("\\f"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            character if character <= '\u{001f}' => {
                rendered.push_str(&format!("\\u{:04x}", character as u32));
            }
            character if character.is_ascii() => rendered.push(character),
            character if character as u32 <= 0xffff => {
                rendered.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => {
                let scalar = character as u32 - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                rendered.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    rendered.push('"');
    rendered
}

fn render_json_value(value: &Value, depth: usize, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&python_number_string(&value.to_string())),
        Value::String(value) => output.push_str(&json_string(value)),
        Value::Array(values) => {
            if values.is_empty() {
                output.push_str("[]");
                return;
            }
            output.push_str("[\n");
            for (index, value) in values.iter().enumerate() {
                output.push_str(&"  ".repeat(depth + 1));
                render_json_value(value, depth + 1, output);
                if index + 1 != values.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&"  ".repeat(depth));
            output.push(']');
        }
        Value::Object(values) => {
            if values.is_empty() {
                output.push_str("{}");
                return;
            }
            output.push_str("{\n");
            let mut entries = values.iter().collect::<Vec<_>>();
            entries.sort_by(|(left, _), (right, _)| left.cmp(right));
            let count = entries.len();
            for (index, (key, value)) in entries.into_iter().enumerate() {
                output.push_str(&"  ".repeat(depth + 1));
                output.push_str(&json_string(key));
                output.push_str(": ");
                render_json_value(value, depth + 1, output);
                if index + 1 != count {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&"  ".repeat(depth));
            output.push('}');
        }
    }
}

fn python_pretty_json(value: &Value) -> String {
    let mut output = String::new();
    render_json_value(value, 0, &mut output);
    output.push('\n');
    output
}

fn csv_quote(value: &str) -> String {
    if value
        .chars()
        .any(|character| matches!(character, ',' | '"' | '\r' | '\n'))
    {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

fn csv_line<'a>(values: impl IntoIterator<Item = &'a str>) -> String {
    let mut line = values
        .into_iter()
        .map(csv_quote)
        .collect::<Vec<_>>()
        .join(",");
    line.push('\n');
    line
}

fn float_string(value: f64) -> String {
    python_number_string(&serde_json::to_string(&value).unwrap_or_else(|_| value.to_string()))
}

fn python_number_string(value: &str) -> String {
    if !value.contains('e') {
        let (sign, unsigned) = value
            .strip_prefix('-')
            .map_or(("", value), |unsigned| ("-", unsigned));
        if let Some(fraction) = unsigned.strip_prefix("0.") {
            let zeroes = fraction.bytes().take_while(|byte| *byte == b'0').count();
            if zeroes >= 4 {
                let significant = &fraction[zeroes..];
                if !significant.is_empty() {
                    let mut digits = significant.chars();
                    let first = digits.next().expect("nonempty significant digits");
                    let tail = digits.as_str();
                    let mantissa = if tail.is_empty() {
                        first.to_string()
                    } else {
                        format!("{first}.{tail}")
                    };
                    return format!("{sign}{mantissa}e-{:02}", zeroes + 1);
                }
            }
        }
        return value.to_owned();
    }
    let (mantissa, exponent) = value.split_once('e').expect("checked exponent");
    let (sign, digits) = if let Some(digits) = exponent.strip_prefix('-') {
        ("-", digits)
    } else if let Some(digits) = exponent.strip_prefix('+') {
        ("+", digits)
    } else {
        ("+", exponent)
    };
    format!("{mantissa}e{sign}{digits:0>2}")
}

fn shell_command_json(command: &ShellCommandSummary) -> Result<Value, BatchReportError> {
    let mut row = Map::new();
    row.insert(
        "task_name".to_owned(),
        Value::String(command.task_name.clone()),
    );
    row.insert("action_index".to_owned(), Value::from(command.action_index));
    row.insert(
        "session_id".to_owned(),
        Value::String(command.session_id.clone()),
    );
    row.insert("poll_count".to_owned(), Value::from(command.poll_count));
    row.insert("command".to_owned(), Value::String(command.command.clone()));
    row.insert(
        "command_preview".to_owned(),
        Value::String(command.command_preview.clone()),
    );
    row.insert(
        "command_family".to_owned(),
        Value::String(command.command_family.clone()),
    );
    row.insert(
        "lifecycle_start_monotonic_ns".to_owned(),
        Value::from(command.lifecycle_start_monotonic_ns),
    );
    row.insert(
        "lifecycle_end_monotonic_ns".to_owned(),
        Value::from(command.lifecycle_end_monotonic_ns),
    );
    row.insert(
        "lifecycle_wall_sec".to_owned(),
        json_number(command.lifecycle_wall_sec)?,
    );
    row.insert("status".to_owned(), Value::String(command.status.clone()));
    row.insert(
        "source_status".to_owned(),
        Value::String(command.source_status.clone()),
    );
    row.insert(
        "peak_simultaneous_logical_cores".to_owned(),
        Value::from(command.peak_simultaneous_logical_cores),
    );
    row.insert(
        "average_simultaneous_logical_cores".to_owned(),
        json_number(command.average_simultaneous_logical_cores)?,
    );
    let duration_value = |value: f64, has_bucket: bool| {
        if !has_bucket {
            Ok(Value::from(0))
        } else {
            json_number(value)
        }
    };
    row.insert(
        "time_at_or_above_2_cores_sec".to_owned(),
        duration_value(
            command.time_at_or_above_2_cores_sec,
            command.time_at_or_above_2_has_bucket,
        )?,
    );
    row.insert(
        "time_at_or_above_4_cores_sec".to_owned(),
        duration_value(
            command.time_at_or_above_4_cores_sec,
            command.time_at_or_above_4_has_bucket,
        )?,
    );
    row.insert(
        "time_at_or_above_6_cores_sec".to_owned(),
        duration_value(
            command.time_at_or_above_6_cores_sec,
            command.time_at_or_above_6_has_bucket,
        )?,
    );
    row.insert(
        "time_at_or_above_8_cores_sec".to_owned(),
        duration_value(
            command.time_at_or_above_8_cores_sec,
            command.time_at_or_above_8_has_bucket,
        )?,
    );
    Ok(Value::Object(row))
}

fn frequency_json(row: &CommandFrequencySummary) -> Result<Value, BatchReportError> {
    let mut value = Map::new();
    value.insert("command".to_owned(), Value::String(row.command.clone()));
    value.insert(
        "command_preview".to_owned(),
        Value::String(row.command_preview.clone()),
    );
    value.insert(
        "invocation_count".to_owned(),
        Value::from(row.invocation_count),
    );
    value.insert("task_count".to_owned(), Value::from(row.task_count));
    value.insert(
        "task_names".to_owned(),
        Value::Array(row.task_names.iter().cloned().map(Value::String).collect()),
    );
    value.insert(
        "total_lifecycle_wall_sec".to_owned(),
        json_number(row.total_lifecycle_wall_sec)?,
    );
    value.insert(
        "total_time_at_or_above_2_cores_sec".to_owned(),
        if row.total_time_at_or_above_2_has_bucket {
            json_number(row.total_time_at_or_above_2_cores_sec)?
        } else {
            Value::from(0)
        },
    );
    value.insert(
        "max_peak_simultaneous_logical_cores".to_owned(),
        Value::from(row.max_peak_simultaneous_logical_cores),
    );
    Ok(Value::Object(value))
}

fn family_json(row: &ToolFamilySummary) -> Result<Value, BatchReportError> {
    let mut value = Map::new();
    value.insert(
        "command_family".to_owned(),
        Value::String(row.command_family.clone()),
    );
    value.insert(
        "invocation_count".to_owned(),
        Value::from(row.invocation_count),
    );
    value.insert(
        "invocation_percent_of_commands".to_owned(),
        json_number(row.invocation_percent_of_commands)?,
    );
    value.insert("task_count".to_owned(), Value::from(row.task_count));
    value.insert(
        "task_names".to_owned(),
        Value::Array(row.task_names.iter().cloned().map(Value::String).collect()),
    );
    value.insert(
        "summed_invocation_lifecycle_wall_sec".to_owned(),
        json_number(row.summed_invocation_lifecycle_wall_sec)?,
    );
    value.insert(
        "active_lifecycle_wall_sec".to_owned(),
        json_number(row.active_lifecycle_wall_sec)?,
    );
    value.insert(
        "overlap_collapsed_wall_sec".to_owned(),
        json_number(row.overlap_collapsed_wall_sec)?,
    );
    value.insert(
        "active_lifecycle_percent_of_replay_wall".to_owned(),
        json_number(row.active_lifecycle_percent_of_replay_wall)?,
    );
    value.insert(
        "active_time_at_or_above_2_cores_sec".to_owned(),
        json_number(row.active_time_at_or_above_2_cores_sec)?,
    );
    value.insert(
        "active_time_at_or_above_4_cores_sec".to_owned(),
        json_number(row.active_time_at_or_above_4_cores_sec)?,
    );
    value.insert(
        "active_time_at_or_above_8_cores_sec".to_owned(),
        json_number(row.active_time_at_or_above_8_cores_sec)?,
    );
    value.insert(
        "max_peak_simultaneous_logical_cores".to_owned(),
        Value::from(row.max_peak_simultaneous_logical_cores),
    );
    value.insert(
        "command_examples".to_owned(),
        Value::Array(
            row.command_examples
                .iter()
                .cloned()
                .map(Value::String)
                .collect(),
        ),
    );
    Ok(Value::Object(value))
}

fn resolved_output(path: &Path) -> Result<PathBuf, BatchReportError> {
    absolute_path(path)
}

fn write_command_outputs(
    tasks: &[TaskSummary],
    output_dir: &Path,
    render_png: bool,
) -> Result<BTreeMap<String, PathBuf>, BatchReportError> {
    let commands = build_command_summary(tasks)?;
    let frequency = build_command_frequency(&commands);
    let tool_families = build_tool_family_summary(tasks, &commands)?;
    let json_path = output_dir.join("bash_commands.json");
    let csv_path = output_dir.join("bash_commands.csv");
    let frequency_json_path = output_dir.join("bash_command_frequency.json");
    let frequency_csv_path = output_dir.join("bash_command_frequency.csv");
    let family_json_path = output_dir.join("tool_family_distribution.json");
    let family_csv_path = output_dir.join("tool_family_distribution.csv");

    let command_rows = commands
        .iter()
        .map(shell_command_json)
        .collect::<Result<Vec<_>, _>>()?;
    let command_document = serde_json::json!({
        "schema_version": "scheduler-bash-command-summary.v1",
        "note": "Async write_stdin poll intervals are collapsed into the originating exec_command lifecycle; overlapping command lifetimes are not additive.",
        "bash_commands": command_rows,
    });
    write_text(&json_path, python_pretty_json(&command_document))?;
    const COMMAND_FIELDS: [&str; 18] = [
        "task_name",
        "action_index",
        "session_id",
        "poll_count",
        "command",
        "command_preview",
        "command_family",
        "lifecycle_start_monotonic_ns",
        "lifecycle_end_monotonic_ns",
        "lifecycle_wall_sec",
        "status",
        "source_status",
        "peak_simultaneous_logical_cores",
        "average_simultaneous_logical_cores",
        "time_at_or_above_2_cores_sec",
        "time_at_or_above_4_cores_sec",
        "time_at_or_above_6_cores_sec",
        "time_at_or_above_8_cores_sec",
    ];
    let mut csv = csv_line(COMMAND_FIELDS);
    for row in &commands {
        let values = vec![
            row.task_name.clone(),
            row.action_index.to_string(),
            row.session_id.clone(),
            row.poll_count.to_string(),
            row.command.clone(),
            row.command_preview.clone(),
            row.command_family.clone(),
            row.lifecycle_start_monotonic_ns.to_string(),
            row.lifecycle_end_monotonic_ns.to_string(),
            float_string(row.lifecycle_wall_sec),
            row.status.clone(),
            row.source_status.clone(),
            row.peak_simultaneous_logical_cores.to_string(),
            float_string(row.average_simultaneous_logical_cores),
            if row.time_at_or_above_2_has_bucket {
                float_string(row.time_at_or_above_2_cores_sec)
            } else {
                "0".to_owned()
            },
            if row.time_at_or_above_4_has_bucket {
                float_string(row.time_at_or_above_4_cores_sec)
            } else {
                "0".to_owned()
            },
            if row.time_at_or_above_6_has_bucket {
                float_string(row.time_at_or_above_6_cores_sec)
            } else {
                "0".to_owned()
            },
            if row.time_at_or_above_8_has_bucket {
                float_string(row.time_at_or_above_8_cores_sec)
            } else {
                "0".to_owned()
            },
        ];
        csv.push_str(&csv_line(values.iter().map(String::as_str)));
    }
    write_text(&csv_path, csv)?;

    let frequency_rows = frequency
        .iter()
        .map(frequency_json)
        .collect::<Result<Vec<_>, _>>()?;
    let frequency_document = serde_json::json!({
        "schema_version": "scheduler-bash-command-frequency.v1",
        "note": "Rows group exact normalized exec_command shell strings; aggregate lifecycle totals may overlap for concurrently active commands.",
        "bash_command_frequency": frequency_rows,
    });
    write_text(
        &frequency_json_path,
        python_pretty_json(&frequency_document),
    )?;
    const FREQUENCY_FIELDS: [&str; 8] = [
        "command",
        "command_preview",
        "invocation_count",
        "task_count",
        "task_names",
        "total_lifecycle_wall_sec",
        "total_time_at_or_above_2_cores_sec",
        "max_peak_simultaneous_logical_cores",
    ];
    let mut frequency_csv = csv_line(FREQUENCY_FIELDS);
    for row in &frequency {
        let values = vec![
            row.command.clone(),
            row.command_preview.clone(),
            row.invocation_count.to_string(),
            row.task_count.to_string(),
            row.task_names.join(";"),
            float_string(row.total_lifecycle_wall_sec),
            if row.total_time_at_or_above_2_has_bucket {
                float_string(row.total_time_at_or_above_2_cores_sec)
            } else {
                "0".to_owned()
            },
            row.max_peak_simultaneous_logical_cores.to_string(),
        ];
        frequency_csv.push_str(&csv_line(values.iter().map(String::as_str)));
    }
    write_text(&frequency_csv_path, frequency_csv)?;

    let family_rows = tool_families
        .iter()
        .map(family_json)
        .collect::<Result<Vec<_>, _>>()?;
    let family_document = serde_json::json!({
        "schema_version": "scheduler-tool-family-distribution.v1",
        "classification": "Semantic top-level shell-action family; categories retain multi-tool compound actions when subcommand time is not observed.",
        "timing_note": "Active lifecycle time is unioned within each family and task; overlap_collapsed_wall_sec reports duplicate same-family interval time. Different families can still overlap when asynchronous commands are active.",
        "denominator": {
            "actual_bash_commands": commands.len(),
            "measured_replay_wall_sec": tasks.iter().map(|task| task.wall_sec).sum::<f64>(),
        },
        "tool_families": family_rows,
    });
    write_text(&family_json_path, python_pretty_json(&family_document))?;
    const FAMILY_FIELDS: [&str; 14] = [
        "command_family",
        "invocation_count",
        "invocation_percent_of_commands",
        "task_count",
        "task_names",
        "summed_invocation_lifecycle_wall_sec",
        "active_lifecycle_wall_sec",
        "overlap_collapsed_wall_sec",
        "active_lifecycle_percent_of_replay_wall",
        "active_time_at_or_above_2_cores_sec",
        "active_time_at_or_above_4_cores_sec",
        "active_time_at_or_above_8_cores_sec",
        "max_peak_simultaneous_logical_cores",
        "command_examples",
    ];
    let mut family_csv = csv_line(FAMILY_FIELDS);
    for row in &tool_families {
        let values = vec![
            row.command_family.clone(),
            row.invocation_count.to_string(),
            float_string(row.invocation_percent_of_commands),
            row.task_count.to_string(),
            row.task_names.join(";"),
            float_string(row.summed_invocation_lifecycle_wall_sec),
            float_string(row.active_lifecycle_wall_sec),
            float_string(row.overlap_collapsed_wall_sec),
            float_string(row.active_lifecycle_percent_of_replay_wall),
            float_string(row.active_time_at_or_above_2_cores_sec),
            float_string(row.active_time_at_or_above_4_cores_sec),
            float_string(row.active_time_at_or_above_8_cores_sec),
            row.max_peak_simultaneous_logical_cores.to_string(),
            row.command_examples.join(" ; "),
        ];
        family_csv.push_str(&csv_line(values.iter().map(String::as_str)));
    }
    write_text(&family_csv_path, family_csv)?;

    let mut outputs = BTreeMap::from([
        (
            "bash_commands_json".to_owned(),
            resolved_output(&json_path)?,
        ),
        ("bash_commands_csv".to_owned(), resolved_output(&csv_path)?),
        (
            "bash_command_frequency_json".to_owned(),
            resolved_output(&frequency_json_path)?,
        ),
        (
            "bash_command_frequency_csv".to_owned(),
            resolved_output(&frequency_csv_path)?,
        ),
        (
            "tool_family_distribution_json".to_owned(),
            resolved_output(&family_json_path)?,
        ),
        (
            "tool_family_distribution_csv".to_owned(),
            resolved_output(&family_csv_path)?,
        ),
    ]);
    if render_png {
        let duration_path = output_dir.join("longest_bash_commands.png");
        let burst_path = output_dir.join("burstiest_bash_commands.png");
        let repeated_path = output_dir.join("bash_command_frequency.png");
        let families_path = output_dir.join("tool_family_distribution.png");
        render_command_duration_png(&commands, &duration_path)?;
        render_burstiest_commands_png(&commands, &burst_path)?;
        render_command_frequency_png(&frequency, &repeated_path)?;
        render_tool_family_distribution_png(&tool_families, &families_path)?;
        outputs.insert(
            "longest_bash_commands_png".to_owned(),
            resolved_output(&duration_path)?,
        );
        outputs.insert(
            "burstiest_bash_commands_png".to_owned(),
            resolved_output(&burst_path)?,
        );
        outputs.insert(
            "bash_command_frequency_png".to_owned(),
            resolved_output(&repeated_path)?,
        );
        outputs.insert(
            "tool_family_distribution_png".to_owned(),
            resolved_output(&families_path)?,
        );
    }
    Ok(outputs)
}

#[derive(Clone)]
struct Canvas {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

impl Canvas {
    fn new(width: usize, height: usize) -> Self {
        Self {
            width,
            height,
            pixels: vec![255; width * height * 3],
        }
    }

    fn set(&mut self, x: i64, y: i64, color: [u8; 3]) {
        if x < 0 || y < 0 || x >= self.width as i64 || y >= self.height as i64 {
            return;
        }
        let offset = (y as usize * self.width + x as usize) * 3;
        self.pixels[offset..offset + 3].copy_from_slice(&color);
    }

    fn rect(&mut self, x: i64, y: i64, width: i64, height: i64, color: [u8; 3]) {
        for row in y.max(0)..(y + height).min(self.height as i64) {
            for column in x.max(0)..(x + width).min(self.width as i64) {
                self.set(column, row, color);
            }
        }
    }

    fn line(&mut self, mut x0: i64, mut y0: i64, x1: i64, y1: i64, color: [u8; 3]) {
        let dx = (x1 - x0).abs();
        let sx = if x0 < x1 { 1 } else { -1 };
        let dy = -(y1 - y0).abs();
        let sy = if y0 < y1 { 1 } else { -1 };
        let mut error = dx + dy;
        loop {
            self.set(x0, y0, color);
            self.set(x0, y0 + 1, color);
            if x0 == x1 && y0 == y1 {
                break;
            }
            let twice = 2 * error;
            if twice >= dy {
                error += dy;
                x0 += sx;
            }
            if twice <= dx {
                error += dx;
                y0 += sy;
            }
        }
    }

    fn text(&mut self, x: i64, y: i64, text: &str, scale: i64, color: [u8; 3]) {
        let mut cursor = x;
        for character in text.chars() {
            let glyph = glyph(character.to_ascii_uppercase());
            for (row, bits) in glyph.iter().enumerate() {
                for column in 0..5 {
                    if bits & (1 << (4 - column)) != 0 {
                        self.rect(
                            cursor + column * scale,
                            y + row as i64 * scale,
                            scale,
                            scale,
                            color,
                        );
                    }
                }
            }
            cursor += 6 * scale;
        }
    }
}

fn glyph(character: char) -> [u8; 7] {
    match character {
        'A' => [14, 17, 17, 31, 17, 17, 17],
        'B' => [30, 17, 17, 30, 17, 17, 30],
        'C' => [15, 16, 16, 16, 16, 16, 15],
        'D' => [30, 17, 17, 17, 17, 17, 30],
        'E' => [31, 16, 16, 30, 16, 16, 31],
        'F' => [31, 16, 16, 30, 16, 16, 16],
        'G' => [15, 16, 16, 19, 17, 17, 15],
        'H' => [17, 17, 17, 31, 17, 17, 17],
        'I' => [31, 4, 4, 4, 4, 4, 31],
        'J' => [7, 2, 2, 2, 18, 18, 12],
        'K' => [17, 18, 20, 24, 20, 18, 17],
        'L' => [16, 16, 16, 16, 16, 16, 31],
        'M' => [17, 27, 21, 21, 17, 17, 17],
        'N' => [17, 25, 21, 19, 17, 17, 17],
        'O' => [14, 17, 17, 17, 17, 17, 14],
        'P' => [30, 17, 17, 30, 16, 16, 16],
        'Q' => [14, 17, 17, 17, 21, 18, 13],
        'R' => [30, 17, 17, 30, 20, 18, 17],
        'S' => [15, 16, 16, 14, 1, 1, 30],
        'T' => [31, 4, 4, 4, 4, 4, 4],
        'U' => [17, 17, 17, 17, 17, 17, 14],
        'V' => [17, 17, 17, 17, 17, 10, 4],
        'W' => [17, 17, 17, 21, 21, 21, 10],
        'X' => [17, 17, 10, 4, 10, 17, 17],
        'Y' => [17, 17, 10, 4, 4, 4, 4],
        'Z' => [31, 1, 2, 4, 8, 16, 31],
        '0' => [14, 17, 19, 21, 25, 17, 14],
        '1' => [4, 12, 4, 4, 4, 4, 14],
        '2' => [14, 17, 1, 2, 4, 8, 31],
        '3' => [30, 1, 1, 14, 1, 1, 30],
        '4' => [2, 6, 10, 18, 31, 2, 2],
        '5' => [31, 16, 16, 30, 1, 1, 30],
        '6' => [14, 16, 16, 30, 17, 17, 14],
        '7' => [31, 1, 2, 4, 8, 8, 8],
        '8' => [14, 17, 17, 14, 17, 17, 14],
        '9' => [14, 17, 17, 15, 1, 1, 14],
        '-' => [0, 0, 0, 31, 0, 0, 0],
        '_' => [0, 0, 0, 0, 0, 0, 31],
        '.' => [0, 0, 0, 0, 0, 12, 12],
        ':' => [0, 12, 12, 0, 12, 12, 0],
        '/' => [1, 2, 2, 4, 8, 8, 16],
        '%' => [17, 2, 4, 8, 17, 0, 0],
        '+' => [0, 4, 4, 31, 4, 4, 0],
        '=' => [0, 0, 31, 0, 31, 0, 0],
        '>' => [16, 8, 4, 2, 4, 8, 16],
        _ => [0; 7],
    }
}

fn crc32(bytes: &[u8]) -> u32 {
    let mut crc = 0xffff_ffffu32;
    for byte in bytes {
        crc ^= u32::from(*byte);
        for _ in 0..8 {
            crc = if crc & 1 == 1 {
                (crc >> 1) ^ 0xedb8_8320
            } else {
                crc >> 1
            };
        }
    }
    !crc
}

fn png_chunk(output: &mut Vec<u8>, kind: &[u8; 4], payload: &[u8]) {
    output.extend_from_slice(&(payload.len() as u32).to_be_bytes());
    output.extend_from_slice(kind);
    output.extend_from_slice(payload);
    let mut crc_input = Vec::with_capacity(4 + payload.len());
    crc_input.extend_from_slice(kind);
    crc_input.extend_from_slice(payload);
    output.extend_from_slice(&crc32(&crc_input).to_be_bytes());
}

fn write_png(path: &Path, canvas: &Canvas) -> Result<(), BatchReportError> {
    let mut raw = Vec::with_capacity((canvas.width * 3 + 1) * canvas.height);
    for row in 0..canvas.height {
        raw.push(0);
        let start = row * canvas.width * 3;
        raw.extend_from_slice(&canvas.pixels[start..start + canvas.width * 3]);
    }
    let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
    encoder
        .write_all(&raw)
        .map_err(|error| BatchReportError::new(format!("unable to encode PNG: {error}")))?;
    let compressed = encoder
        .finish()
        .map_err(|error| BatchReportError::new(format!("unable to encode PNG: {error}")))?;
    let mut png = b"\x89PNG\r\n\x1a\n".to_vec();
    let mut header = Vec::with_capacity(13);
    header.extend_from_slice(&(canvas.width as u32).to_be_bytes());
    header.extend_from_slice(&(canvas.height as u32).to_be_bytes());
    header.extend_from_slice(&[8, 2, 0, 0, 0]);
    png_chunk(&mut png, b"IHDR", &header);
    png_chunk(&mut png, b"IDAT", &compressed);
    png_chunk(&mut png, b"IEND", &[]);
    fs::write(path, png).map_err(|error| {
        BatchReportError::new(format!("unable to write {}: {error}", path.display()))
    })
}

fn color(hex: &str) -> [u8; 3] {
    let byte = |start| u8::from_str_radix(&hex[start..start + 2], 16).unwrap_or(0);
    [byte(1), byte(3), byte(5)]
}

fn chart_grid(canvas: &mut Canvas, left: i64, top: i64, right: i64, bottom: i64) {
    let gray = color("#e5e7eb");
    for tick in 0..=10 {
        let x = left + (right - left) * tick / 10;
        canvas.line(x, top, x, bottom, gray);
    }
    canvas.line(left, bottom, right, bottom, color("#374151"));
}

fn render_average_vs_peak_png(
    tasks: &[TaskSummary],
    output: &Path,
) -> Result<(), BatchReportError> {
    let height = (tasks.len() * 76 + 150).max(300);
    let mut canvas = Canvas::new(1200, height);
    canvas.text(
        30,
        20,
        "AVERAGE CPU USE VS PEAK SIMULTANEOUS LOGICAL CPUS",
        2,
        color("#111827"),
    );
    let allocation = tasks
        .iter()
        .map(|task| task.allocated_logical_cpus)
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(1.0);
    chart_grid(&mut canvas, 250, 60, 1150, height as i64 - 50);
    for (index, task) in tasks.iter().enumerate() {
        let y = 80 + index as i64 * 76;
        canvas.text(10, y + 7, &task.task_name, 1, color("#111827"));
        let average = (task.average_logical_cores_used / allocation * 900.0).max(0.0) as i64;
        canvas.rect(250, y, average.min(900), 28, color("#2563eb"));
        let peak = (task.peak_simultaneous_logical_cores as f64 / allocation * 900.0) as i64;
        canvas.rect(250 + peak.clamp(0, 900) - 4, y - 5, 8, 38, color("#dc2626"));
    }
    write_png(output, &canvas)
}

fn render_distribution_png(tasks: &[TaskSummary], output: &Path) -> Result<(), BatchReportError> {
    let height = (tasks.len() * 76 + 150).max(300);
    let mut canvas = Canvas::new(1200, height);
    canvas.text(
        30,
        20,
        "PER-TASK SIMULTANEOUS LOGICAL-CPU DISTRIBUTION",
        2,
        color("#111827"),
    );
    chart_grid(&mut canvas, 250, 60, 1150, height as i64 - 50);
    for (index, task) in tasks.iter().enumerate() {
        let y = 80 + index as i64 * 76;
        canvas.text(10, y + 7, &task.task_name, 1, color("#111827"));
        let total: f64 = task.time_sec_by_cores.values().sum();
        let mut x = 250i64;
        for (cores, seconds) in &task.time_sec_by_cores {
            let width = if total == 0.0 {
                0
            } else {
                (seconds / total * 900.0).round() as i64
            };
            let index = (*cores).max(0) as usize;
            canvas.rect(
                x,
                y,
                width.max(0),
                28,
                color(CORE_COLORS[index.min(CORE_COLORS.len() - 1)]),
            );
            x += width;
        }
    }
    write_png(output, &canvas)
}

fn timeline(task: &TaskSummary) -> Result<(Vec<(f64, i64)>, f64), BatchReportError> {
    if task.replay_report_path.is_none() || task.scheduler_events_path.is_none() {
        return Err(BatchReportError::new(format!(
            "{} lacks replay-report or scheduler-events paths for timeline rendering",
            task.task_name
        )));
    }
    let replay = read_json_object(
        task.replay_report_path
            .as_ref()
            .expect("validated replay report path"),
        "replay report",
    )?;
    let raw_actions = match replay.get("actions") {
        None => &[][..],
        Some(Value::Array(actions)) => actions.as_slice(),
        Some(_) => {
            return Err(BatchReportError::new(format!(
                "{} replay actions are not a list",
                task.task_name
            )))
        }
    };
    if raw_actions.iter().any(|action| !action.is_object()) {
        return Err(BatchReportError::new(format!(
            "{} replay report contains a non-object action",
            task.task_name
        )));
    }
    let actions = raw_actions
        .iter()
        .filter_map(Value::as_object)
        .filter(|action| {
            action.contains_key("replay_start_monotonic_ns")
                && action.contains_key("replay_end_monotonic_ns")
        })
        .cloned()
        .collect::<Vec<_>>();
    if actions.is_empty() {
        return Err(BatchReportError::new(format!(
            "{} replay report has no timed action windows",
            task.task_name
        )));
    }
    let start = actions
        .iter()
        .map(|action| {
            integer(
                action.get("replay_start_monotonic_ns"),
                "replay_start_monotonic_ns",
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .min()
        .ok_or_else(|| {
            BatchReportError::new(format!(
                "{} replay report has no timed action windows",
                task.task_name
            ))
        })?;
    let end = actions
        .iter()
        .map(|action| {
            integer(
                action.get("replay_end_monotonic_ns"),
                "replay_end_monotonic_ns",
            )
        })
        .collect::<Result<Vec<_>, _>>()?
        .into_iter()
        .max()
        .ok_or_else(|| {
            BatchReportError::new(format!(
                "{} replay report has no timed action windows",
                task.task_name
            ))
        })?;
    if end <= start {
        return Err(BatchReportError::new(format!(
            "{} action timeline has no positive duration",
            task.task_name
        )));
    }
    let events = load_scheduler_events(task)?;
    let mut active = BTreeSet::new();
    for event in events.iter().take_while(|event| event.timestamp_ns < start) {
        if event.entering {
            active.insert(event.cpu);
        } else {
            active.remove(&event.cpu);
        }
    }
    let mut points = vec![(0.0, active.len() as i64)];
    for event in events
        .iter()
        .filter(|event| event.timestamp_ns >= start && event.timestamp_ns <= end)
    {
        let elapsed = (i128::from(event.timestamp_ns) - i128::from(start)) as f64 / 1_000_000_000.0;
        points.push((elapsed, active.len() as i64));
        if event.entering {
            active.insert(event.cpu);
        } else {
            active.remove(&event.cpu);
        }
        points.push((elapsed, active.len() as i64));
    }
    let duration = (i128::from(end) - i128::from(start)) as f64 / 1_000_000_000.0;
    points.push((duration, active.len() as i64));
    Ok((points, duration))
}

fn render_timeline_png(tasks: &[TaskSummary], output: &Path) -> Result<(), BatchReportError> {
    let panel_height = 150usize;
    let height = (tasks.len() * panel_height + 80).max(300);
    let mut canvas = Canvas::new(1500, height);
    canvas.text(
        30,
        8,
        "PER-TASK SCHEDULED LOGICAL-CPU TIMELINE",
        2,
        color("#111827"),
    );
    let allocation = tasks
        .iter()
        .map(|task| task.allocated_logical_cpus)
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(1.0);
    for (index, task) in tasks.iter().enumerate() {
        let top = 30 + index as i64 * panel_height as i64;
        let bottom = top + 110;
        canvas.text(10, top + 50, &task.task_name, 1, color("#111827"));
        chart_grid(&mut canvas, 250, top, 1450, bottom);
        let (points, duration) = timeline(task)?;
        let mut previous = None;
        for (seconds, running) in points {
            let x = 250 + (seconds / duration * 1200.0).clamp(0.0, 1200.0) as i64;
            let y = bottom - (running as f64 / allocation * 105.0).clamp(0.0, 105.0) as i64;
            if let Some((previous_x, previous_y)) = previous {
                canvas.line(previous_x, previous_y, x, y, color("#2563eb"));
            }
            previous = Some((x, y));
        }
    }
    write_png(output, &canvas)
}

fn render_ranked_bars(
    values: &[f64],
    secondary: Option<&[f64]>,
    tertiary: Option<&[f64]>,
    labels: &[String],
    title: &str,
    output: &Path,
) -> Result<(), BatchReportError> {
    let row_count = values.len().max(1);
    let height = (row_count * 32 + 100).max(300);
    let mut canvas = Canvas::new(1500, height);
    canvas.text(30, 12, title, 2, color("#111827"));
    let maximum = values
        .iter()
        .copied()
        .chain(
            secondary
                .into_iter()
                .flat_map(|values| values.iter().copied()),
        )
        .chain(
            tertiary
                .into_iter()
                .flat_map(|values| values.iter().copied()),
        )
        .reduce(f64::max)
        .unwrap_or(1.0)
        .max(f64::EPSILON);
    chart_grid(&mut canvas, 400, 40, 1450, height as i64 - 40);
    for (index, value) in values.iter().enumerate() {
        let y = 55 + index as i64 * 32;
        if let Some(label) = labels.get(index) {
            canvas.text(10, y + 6, label, 1, color("#111827"));
        }
        let width = (*value / maximum * 1050.0).max(0.0) as i64;
        canvas.rect(400, y, width, 20, color("#2563eb"));
        if let Some(secondary) = secondary {
            let width =
                (secondary.get(index).copied().unwrap_or(0.0) / maximum * 1050.0).max(0.0) as i64;
            canvas.rect(400, y + 10, width, 10, color("#f97316"));
        }
        if let Some(tertiary) = tertiary {
            let width =
                (tertiary.get(index).copied().unwrap_or(0.0) / maximum * 1050.0).max(0.0) as i64;
            canvas.rect(400, y + 15, width, 5, color("#dc2626"));
        }
    }
    write_png(output, &canvas)
}

fn render_command_duration_png(
    commands: &[ShellCommandSummary],
    output: &Path,
) -> Result<(), BatchReportError> {
    let mut ranked = commands.to_vec();
    ranked.sort_by(|left, right| {
        right
            .lifecycle_wall_sec
            .partial_cmp(&left.lifecycle_wall_sec)
            .unwrap_or(Ordering::Equal)
    });
    ranked.truncate(25);
    ranked.reverse();
    let values = ranked
        .iter()
        .map(|row| row.lifecycle_wall_sec)
        .collect::<Vec<_>>();
    let burst = ranked
        .iter()
        .map(|row| row.time_at_or_above_2_cores_sec)
        .collect::<Vec<_>>();
    let labels = ranked
        .iter()
        .map(|row| format!("{}: {}", row.task_name, row.command_preview))
        .collect::<Vec<_>>();
    render_ranked_bars(
        &values,
        Some(&burst),
        None,
        &labels,
        "LONGEST BASH COMMANDS AND CONCURRENT-CORE BURST TIME",
        output,
    )
}

fn render_burstiest_commands_png(
    commands: &[ShellCommandSummary],
    output: &Path,
) -> Result<(), BatchReportError> {
    let mut ranked = commands.to_vec();
    ranked.sort_by(|left, right| {
        right
            .time_at_or_above_2_cores_sec
            .partial_cmp(&left.time_at_or_above_2_cores_sec)
            .unwrap_or(Ordering::Equal)
    });
    ranked.truncate(25);
    ranked.reverse();
    let values = ranked
        .iter()
        .map(|row| row.time_at_or_above_2_cores_sec)
        .collect::<Vec<_>>();
    let burst = ranked
        .iter()
        .map(|row| row.time_at_or_above_4_cores_sec)
        .collect::<Vec<_>>();
    let burst_8 = ranked
        .iter()
        .map(|row| row.time_at_or_above_8_cores_sec)
        .collect::<Vec<_>>();
    let labels = ranked
        .iter()
        .map(|row| format!("{}: {}", row.task_name, row.command_preview))
        .collect::<Vec<_>>();
    render_ranked_bars(
        &values,
        Some(&burst),
        Some(&burst_8),
        &labels,
        "BASH COMMANDS WITH THE MOST CONCURRENT-CORE DEMAND",
        output,
    )
}

fn render_command_frequency_png(
    frequency: &[CommandFrequencySummary],
    output: &Path,
) -> Result<(), BatchReportError> {
    let mut repeated = frequency
        .iter()
        .filter(|row| row.invocation_count > 1)
        .take(25)
        .collect::<Vec<_>>();
    repeated.reverse();
    let values = repeated
        .iter()
        .map(|row| row.invocation_count as f64)
        .collect::<Vec<_>>();
    let labels = repeated
        .iter()
        .map(|row| row.command_preview.clone())
        .collect::<Vec<_>>();
    render_ranked_bars(
        &values,
        None,
        None,
        &labels,
        "REPEATED EXACT BASH COMMANDS",
        output,
    )
}

fn render_tool_family_distribution_png(
    families: &[ToolFamilySummary],
    output: &Path,
) -> Result<(), BatchReportError> {
    let values = families
        .iter()
        .take(20)
        .rev()
        .map(|row| row.active_lifecycle_percent_of_replay_wall)
        .collect::<Vec<_>>();
    let invocation = families
        .iter()
        .take(20)
        .rev()
        .map(|row| row.invocation_percent_of_commands)
        .collect::<Vec<_>>();
    let burst = families
        .iter()
        .take(20)
        .rev()
        .map(|row| row.active_time_at_or_above_2_cores_sec)
        .collect::<Vec<_>>();
    let labels = families
        .iter()
        .take(20)
        .rev()
        .map(|row| row.command_family.clone())
        .collect::<Vec<_>>();
    render_ranked_bars(
        &values,
        Some(&invocation),
        Some(&burst),
        &labels,
        "SEMANTIC COMMAND FAMILIES ACROSS VALIDATED REPLAYS",
        output,
    )
}

fn task_json(task: &TaskSummary) -> Result<Value, BatchReportError> {
    let distribution = task
        .time_sec_by_cores
        .iter()
        .map(|(cores, seconds)| Ok((cores.to_string(), json_number(*seconds)?)))
        .collect::<Result<Map<String, Value>, BatchReportError>>()?;
    let mut row = Map::new();
    row.insert(
        "task_name".to_owned(),
        Value::String(task.task_name.clone()),
    );
    row.insert(
        "source_trial".to_owned(),
        Value::String(task.source_trial.clone()),
    );
    row.insert(
        "pipeline_report".to_owned(),
        Value::String(task.report_path.display().to_string()),
    );
    row.insert("wall_sec".to_owned(), json_number(task.wall_sec)?);
    row.insert("cpu_usage_sec".to_owned(), json_number(task.cpu_usage_sec)?);
    row.insert(
        "average_logical_cores_used".to_owned(),
        json_number(task.average_logical_cores_used)?,
    );
    row.insert(
        "scheduler_average_logical_cores_used".to_owned(),
        json_number(task.scheduler_average_logical_cores_used)?,
    );
    row.insert(
        "allocated_logical_cpus".to_owned(),
        json_number(task.allocated_logical_cpus)?,
    );
    row.insert(
        "utilization_percent_of_allocation".to_owned(),
        json_number(task.utilization_percent_of_allocation())?,
    );
    row.insert(
        "peak_simultaneous_logical_cores".to_owned(),
        Value::from(task.peak_simultaneous_logical_cores),
    );
    row.insert(
        "time_above_one_core_sec".to_owned(),
        if task.time_sec_by_cores.keys().any(|cores| *cores > 1) {
            json_number(task.time_above_one_core_sec())?
        } else {
            Value::from(0)
        },
    );
    row.insert(
        "time_sec_by_simultaneous_logical_cores".to_owned(),
        Value::Object(distribution),
    );
    row.insert(
        "lost_perf_events".to_owned(),
        Value::from(task.lost_perf_events),
    );
    Ok(Value::Object(row))
}

pub fn write_outputs(
    tasks: &[TaskSummary],
    output_dir: &Path,
    render_png: bool,
) -> Result<BTreeMap<String, PathBuf>, BatchReportError> {
    fs::create_dir_all(output_dir).map_err(|error| {
        BatchReportError::new(format!(
            "unable to create output directory {}: {error}",
            output_dir.display()
        ))
    })?;
    let json_path = output_dir.join("batch_summary.json");
    let csv_path = output_dir.join("batch_summary.csv");
    let average_path = output_dir.join("average_vs_peak.svg");
    let distribution_path = output_dir.join("core_distribution.svg");
    let rows = tasks.iter().map(task_json).collect::<Result<Vec<_>, _>>()?;
    let document = serde_json::json!({
        "schema_version": SUMMARY_SCHEMA_VERSION,
        "tasks": rows,
    });
    write_text(&json_path, python_pretty_json(&document))?;
    const FIELDS: [&str; 12] = [
        "task_name",
        "source_trial",
        "pipeline_report",
        "wall_sec",
        "cpu_usage_sec",
        "average_logical_cores_used",
        "scheduler_average_logical_cores_used",
        "allocated_logical_cpus",
        "utilization_percent_of_allocation",
        "peak_simultaneous_logical_cores",
        "time_above_one_core_sec",
        "lost_perf_events",
    ];
    let mut csv = csv_line(FIELDS);
    for task in tasks {
        let values = vec![
            task.task_name.clone(),
            task.source_trial.clone(),
            task.report_path.display().to_string(),
            float_string(task.wall_sec),
            float_string(task.cpu_usage_sec),
            float_string(task.average_logical_cores_used),
            float_string(task.scheduler_average_logical_cores_used),
            float_string(task.allocated_logical_cpus),
            float_string(task.utilization_percent_of_allocation()),
            task.peak_simultaneous_logical_cores.to_string(),
            if task.time_sec_by_cores.keys().any(|cores| *cores > 1) {
                float_string(task.time_above_one_core_sec())
            } else {
                "0".to_owned()
            },
            task.lost_perf_events.to_string(),
        ];
        csv.push_str(&csv_line(values.iter().map(String::as_str)));
    }
    write_text(&csv_path, csv)?;
    render_average_vs_peak(tasks, &average_path)?;
    render_distribution(tasks, &distribution_path)?;
    let mut outputs = BTreeMap::from([
        ("json".to_owned(), resolved_output(&json_path)?),
        ("csv".to_owned(), resolved_output(&csv_path)?),
        (
            "average_vs_peak_svg".to_owned(),
            resolved_output(&average_path)?,
        ),
        (
            "core_distribution_svg".to_owned(),
            resolved_output(&distribution_path)?,
        ),
    ]);
    outputs.extend(write_command_outputs(tasks, output_dir, render_png)?);
    if render_png {
        let average_png = output_dir.join("average_vs_peak.png");
        let distribution_png = output_dir.join("core_distribution.png");
        let timeline_png = output_dir.join("core_timeline.png");
        render_average_vs_peak_png(tasks, &average_png)?;
        render_distribution_png(tasks, &distribution_png)?;
        render_timeline_png(tasks, &timeline_png)?;
        outputs.insert(
            "average_vs_peak_png".to_owned(),
            resolved_output(&average_png)?,
        );
        outputs.insert(
            "core_distribution_png".to_owned(),
            resolved_output(&distribution_png)?,
        );
        outputs.insert(
            "core_timeline_png".to_owned(),
            resolved_output(&timeline_png)?,
        );
    }
    Ok(outputs)
}

#[derive(Debug)]
struct Cli {
    pipeline_reports: Vec<PathBuf>,
    output_dir: PathBuf,
    render_png: bool,
}

enum CliParse {
    Help,
    Run(Cli),
}

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut reports = Vec::new();
    let mut output_dir = None;
    let mut render_png = false;
    let mut options = true;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if options && matches!(argument.to_str(), Some("-h" | "--help")) {
            return Ok(CliParse::Help);
        }
        if options && argument == "--" {
            options = false;
            index += 1;
            continue;
        }
        if options {
            let text = argument.to_string_lossy();
            if text.starts_with("--") && "--png".starts_with(text.as_ref()) {
                render_png = true;
                index += 1;
                continue;
            }
            if let Some((name, value)) = text.split_once('=') {
                if "--output-dir".starts_with(name) {
                    if value.is_empty() {
                        return Err("argument --output-dir: expected one argument".to_owned());
                    }
                    output_dir = Some(PathBuf::from(value));
                    index += 1;
                    continue;
                }
                if "--png".starts_with(name) {
                    return Err(format!(
                        "argument --png: ignored explicit argument {value:?}"
                    ));
                }
            }
            if text.starts_with("--") && "--output-dir".starts_with(text.as_ref()) {
                index += 1;
                let value = arguments
                    .get(index)
                    .ok_or_else(|| "argument --output-dir: expected one argument".to_owned())?;
                output_dir = Some(PathBuf::from(value));
                index += 1;
                continue;
            }
            if text.starts_with('-') {
                return Err(format!("unrecognized arguments: {text}"));
            }
        }
        reports.push(PathBuf::from(argument));
        index += 1;
    }
    let mut missing = Vec::new();
    if output_dir.is_none() {
        missing.push("--output-dir");
    }
    if reports.is_empty() {
        missing.push("pipeline_reports");
    }
    if !missing.is_empty() {
        return Err(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    Ok(CliParse::Run(Cli {
        pipeline_reports: reports,
        output_dir: output_dir.expect("validated output directory"),
        render_png,
    }))
}

fn print_usage(public_program: &str, output: &mut dyn Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] --output-dir OUTPUT_DIR [--png] pipeline_reports [pipeline_reports ...]"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!("Render validated scheduler-concurrency pipeline reports into batch SVG charts.");
    println!();
    println!("positional arguments:");
    println!("  pipeline_reports");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --output-dir OUTPUT_DIR");
    println!("  --png                 Also render PNG charts using the native raster renderer.");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
}

/// Native CLI entry point expected by the top-level dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let cli = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(cli)) => cli,
        Err(error) => return cli_error(public_program, &error),
    };
    let tasks = match build_summary(&cli.pipeline_reports) {
        Ok(tasks) => tasks,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    let outputs = match write_outputs(&tasks, &cli.output_dir, cli.render_png) {
        Ok(outputs) => outputs,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    println!("Summary JSON: {}", outputs["json"].display());
    println!("Summary CSV: {}", outputs["csv"].display());
    println!(
        "Average vs peak chart: {}",
        outputs["average_vs_peak_svg"].display()
    );
    println!(
        "Core distribution chart: {}",
        outputs["core_distribution_svg"].display()
    );
    println!(
        "Bash commands JSON: {}",
        outputs["bash_commands_json"].display()
    );
    println!(
        "Bash commands CSV: {}",
        outputs["bash_commands_csv"].display()
    );
    println!(
        "Bash command frequency JSON: {}",
        outputs["bash_command_frequency_json"].display()
    );
    println!(
        "Bash command frequency CSV: {}",
        outputs["bash_command_frequency_csv"].display()
    );
    println!(
        "Tool family distribution JSON: {}",
        outputs["tool_family_distribution_json"].display()
    );
    println!(
        "Tool family distribution CSV: {}",
        outputs["tool_family_distribution_csv"].display()
    );
    if cli.render_png {
        println!(
            "Average vs peak PNG: {}",
            outputs["average_vs_peak_png"].display()
        );
        println!(
            "Core distribution PNG: {}",
            outputs["core_distribution_png"].display()
        );
        println!(
            "Core timeline PNG: {}",
            outputs["core_timeline_png"].display()
        );
        println!(
            "Longest bash commands PNG: {}",
            outputs["longest_bash_commands_png"].display()
        );
        println!(
            "Burstiest bash commands PNG: {}",
            outputs["burstiest_bash_commands_png"].display()
        );
        println!(
            "Bash command frequency PNG: {}",
            outputs["bash_command_frequency_png"].display()
        );
        println!(
            "Tool family distribution PNG: {}",
            outputs["tool_family_distribution_png"].display()
        );
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn command_family_examples_match_public_contract() {
        let examples = [
            ("go test ./...", "go test"),
            ("python -m pytest tests/test_api.py", "python -m pytest"),
            (
                "set +e; valgrind /app/debug; printf '%s\\n' \"$?\"",
                "valgrind",
            ),
            ("rm -f /tmp/frame.bmp; node /app/vm.js", "node vm.js"),
            (
                "python -m py_compile app.py && python app.py",
                "multi-tool: python -m py_compile + python app.py",
            ),
            (
                "grep -E 'np\\.(int|float)' src.py | sed -n '1,20p'",
                "grep pipeline",
            ),
            (
                "python - <<'PY'\nif left | right:\n    print('ok')\nPY",
                "python inline script",
            ),
        ];
        for (command, expected) in examples {
            assert_eq!(classify_command_family(command), expected, "{command}");
        }
    }

    #[test]
    fn native_png_encoder_writes_valid_signature_and_chunks() {
        let directory = TempDir::new().unwrap();
        let path = directory.path().join("chart.png");
        let mut canvas = Canvas::new(8, 8);
        canvas.rect(1, 1, 4, 4, color("#2563eb"));
        write_png(&path, &canvas).unwrap();
        let bytes = fs::read(path).unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"));
        assert!(bytes.windows(4).any(|window| window == b"IHDR"));
        assert!(bytes.windows(4).any(|window| window == b"IDAT"));
        assert!(bytes.ends_with(&[0xae, 0x42, 0x60, 0x82]));
    }

    #[test]
    fn parser_accepts_interspersed_options_and_last_output_wins() {
        let parsed = parse_cli(&[
            OsString::from("a.json"),
            OsString::from("--out"),
            OsString::from("old"),
            OsString::from("b.json"),
            OsString::from("--output-dir=new"),
            OsString::from("--p"),
        ])
        .unwrap();
        let CliParse::Run(parsed) = parsed else {
            panic!("expected runnable CLI")
        };
        assert_eq!(
            parsed.pipeline_reports,
            [PathBuf::from("a.json"), PathBuf::from("b.json")]
        );
        assert_eq!(parsed.output_dir, PathBuf::from("new"));
        assert!(parsed.render_png);
    }

    #[test]
    fn public_main_signature_is_stable() {
        let _entry: fn(&[OsString], &str) -> i32 = main;
    }
    #[test]
    fn batch_rejects_invalid_or_unavailable_replay_before_comparison() {
        let root = tempfile::tempdir().unwrap();
        let path = root.path().join("pipeline.json");
        for payload in [
            serde_json::json!({"status": "invalid"}),
            serde_json::json!({"replay_summary": {
                "source_outcome_mismatch_count": 0,
                "source_outcome_comparison_unavailable_count": 1,
            }, "scheduler_concurrency": {}, "container_cpu": {}}),
        ] {
            fs::write(&path, serde_json::to_vec(&payload).unwrap()).unwrap();
            let error = load_summary(&path).unwrap_err().to_string();
            assert!(error.contains("not valid") || error.contains("comparisons unavailable"));
        }
    }
}
