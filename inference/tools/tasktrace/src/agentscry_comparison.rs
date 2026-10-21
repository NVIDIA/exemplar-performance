// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compare container cgroup CPU accounting with an AgentScry process trace.
//!
//! AgentScry's filtered CSV attributes `USER` and `SYS` time at process exit.
//! The accounting is not automatically equivalent to container `cpu.stat`:
//! descendants can be captured without an exit record, and `asfilter
//! --duration` can apportion one PID's exit total across several exec rows.
//! This module therefore retains every accounted row and reports numerical
//! agreement separately from complete descendant exit accounting.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: &str = "offline-agentscry-comparison.v1";

type CsvRow = BTreeMap<String, String>;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ComparisonError(String);

impl ComparisonError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ComparisonError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ComparisonError {}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcessAggregation {
    pub root_pid: i64,
    pub descendant_pids: BTreeSet<i64>,
    pub exit_accounted_pids: BTreeSet<i64>,
    cpu_accounted_rows: Vec<CsvRow>,
    pub missing_rows: Vec<Value>,
    pub user_sec: f64,
    pub system_sec: f64,
}

impl ProcessAggregation {
    pub fn usage_sec(&self) -> f64 {
        self.user_sec + self.system_sec
    }

    pub fn cpu_accounted_row_count(&self) -> usize {
        self.cpu_accounted_rows.len()
    }
}

/// Parse CSV records using the standard comma/quote rules used by
/// `csv.DictReader`. Blank physical records are ignored, while newlines inside
/// quoted fields remain part of the field.
fn parse_csv_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut characters = text.chars().peekable();
    let mut quoted = false;
    let mut record_started = false;
    while let Some(character) = characters.next() {
        if quoted {
            if character == '"' {
                if characters.peek() == Some(&'"') {
                    characters.next();
                    field.push('"');
                } else {
                    quoted = false;
                }
            } else {
                field.push(character);
            }
            record_started = true;
            continue;
        }
        match character {
            '"' if field.is_empty() => {
                quoted = true;
                record_started = true;
            }
            ',' => {
                record.push(std::mem::take(&mut field));
                record_started = true;
            }
            '\n' => {
                if record_started || !field.is_empty() || !record.is_empty() {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                record_started = false;
            }
            '\r' => {
                if characters.peek() == Some(&'\n') {
                    characters.next();
                }
                if record_started || !field.is_empty() || !record.is_empty() {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                }
                record_started = false;
            }
            other => {
                field.push(other);
                record_started = true;
            }
        }
    }
    if record_started || !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

fn read_filtered_csv(path: &Path) -> Result<Vec<CsvRow>, ComparisonError> {
    if !path.is_file() {
        return Err(ComparisonError::new(format!(
            "AgentScry filtered CSV does not exist: {}",
            path.display()
        )));
    }
    let text = fs::read_to_string(path).map_err(|error| {
        ComparisonError::new(format!(
            "unable to read AgentScry filtered CSV {}: {error}",
            path.display()
        ))
    })?;
    let mut records = parse_csv_records(&text).into_iter();
    let headers = records.next().unwrap_or_default();
    let rows: Vec<CsvRow> = records
        .map(|fields| {
            headers
                .iter()
                .enumerate()
                .map(|(index, header)| {
                    (
                        header.clone(),
                        fields.get(index).cloned().unwrap_or_default(),
                    )
                })
                .collect()
        })
        .collect();

    let required = ["ARGS", "PID", "PPID", "SYS", "USER"];
    // The Python implementation derives fields from the first data row, so a
    // header-only CSV reports every required field as missing.
    let fields: BTreeSet<&str> = if rows.is_empty() {
        BTreeSet::new()
    } else {
        headers.iter().map(String::as_str).collect()
    };
    let missing: Vec<_> = required
        .iter()
        .copied()
        .filter(|field| !fields.contains(field))
        .collect();
    if rows.is_empty() || !missing.is_empty() {
        return Err(ComparisonError::new(format!(
            "AgentScry CSV is empty or missing required columns: {}",
            missing.join(", ")
        )));
    }
    Ok(rows)
}

fn parse_pid(value: &str) -> Option<i64> {
    let value = value.trim();
    (!value.is_empty() && value.bytes().all(|byte| byte.is_ascii_digit()))
        .then(|| value.parse().ok())
        .flatten()
}

fn parse_cpu_seconds(value: &str) -> Option<f64> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.parse().ok()).flatten()
}

fn python_repr_string(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut rendered = String::new();
    rendered.push(quote);
    for character in value.chars() {
        match character {
            '\\' => rendered.push_str("\\\\"),
            '\n' => rendered.push_str("\\n"),
            '\r' => rendered.push_str("\\r"),
            '\t' => rendered.push_str("\\t"),
            character if character == quote => {
                rendered.push('\\');
                rendered.push(character);
            }
            character => rendered.push(character),
        }
    }
    rendered.push(quote);
    rendered
}

fn aggregate_process_tree(
    rows: &[CsvRow],
    root_command_substring: &str,
) -> Result<ProcessAggregation, ComparisonError> {
    let root_pids: BTreeSet<i64> = rows
        .iter()
        .filter(|row| {
            row.get("ARGS")
                .is_some_and(|arguments| arguments.contains(root_command_substring))
        })
        .filter_map(|row| row.get("PID").and_then(|pid| parse_pid(pid)))
        .collect();
    if root_pids.len() != 1 {
        return Err(ComparisonError::new(format!(
            "expected exactly one root PID matching {}; found {:?}",
            python_repr_string(root_command_substring),
            root_pids.iter().copied().collect::<Vec<_>>()
        )));
    }
    let root_pid = *root_pids.iter().next().expect("one root PID validated");

    let mut children: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    let mut rows_by_pid: BTreeMap<i64, Vec<&CsvRow>> = BTreeMap::new();
    for row in rows {
        let Some(pid) = row.get("PID").and_then(|value| parse_pid(value)) else {
            continue;
        };
        rows_by_pid.entry(pid).or_default().push(row);
        if let Some(ppid) = row.get("PPID").and_then(|value| parse_pid(value)) {
            children.entry(ppid).or_default().insert(pid);
        }
    }

    let mut descendant_pids = BTreeSet::from([root_pid]);
    let mut pending = VecDeque::from([root_pid]);
    while let Some(parent) = pending.pop_front() {
        for child in children.get(&parent).into_iter().flatten() {
            if descendant_pids.insert(*child) {
                pending.push_back(*child);
            }
        }
    }

    let mut exit_accounted_pids = BTreeSet::new();
    let mut cpu_accounted_rows = Vec::new();
    let mut missing_rows = Vec::new();
    for pid in &descendant_pids {
        let process_rows = rows_by_pid
            .get(pid)
            .expect("every discovered PID has at least one CSV row");
        let exit_rows: Vec<_> = process_rows
            .iter()
            .filter(|row| {
                row.get("USER")
                    .and_then(|value| parse_cpu_seconds(value))
                    .is_some()
                    && row
                        .get("SYS")
                        .and_then(|value| parse_cpu_seconds(value))
                        .is_some()
            })
            .collect();
        if !exit_rows.is_empty() {
            // asfilter apportions a PID's exit total over all retained exec
            // rows. Every portion is required to reconstruct that total.
            exit_accounted_pids.insert(*pid);
            cpu_accounted_rows.extend(exit_rows.into_iter().map(|row| (*row).clone()));
            continue;
        }
        let last = process_rows.last().expect("process has a row");
        missing_rows.push(json!({
            "pid": pid,
            "comm": last.get("COMM"),
            "args": last.get("ARGS"),
            "duration": last.get("DURATION"),
        }));
    }

    let user_sec = cpu_accounted_rows
        .iter()
        .map(|row| {
            parse_cpu_seconds(row.get("USER").expect("accounted row has USER"))
                .expect("accounted USER was validated")
        })
        .sum();
    let system_sec = cpu_accounted_rows
        .iter()
        .map(|row| {
            parse_cpu_seconds(row.get("SYS").expect("accounted row has SYS"))
                .expect("accounted SYS was validated")
        })
        .sum();
    Ok(ProcessAggregation {
        root_pid,
        descendant_pids,
        exit_accounted_pids,
        cpu_accounted_rows,
        missing_rows,
        user_sec,
        system_sec,
    })
}

fn pipeline_usage_seconds(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(number)) => number.as_f64(),
        // Python's bool is an int subclass and passes the original isinstance
        // check, even though real reports always contain a number.
        Some(Value::Bool(value)) => Some(u8::from(*value) as f64),
        _ => None,
    }
}

fn python_float(value: Option<&Value>) -> Option<f64> {
    match value {
        Some(Value::Number(number)) => number.as_f64(),
        Some(Value::String(value)) => value.trim().parse().ok(),
        Some(Value::Bool(value)) => Some(u8::from(*value) as f64),
        _ => None,
    }
}

fn skip_json_whitespace(source: &str, index: &mut usize) {
    let bytes = source.as_bytes();
    while *index < bytes.len() && bytes[*index].is_ascii_whitespace() {
        *index += 1;
    }
}

fn json_string_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    if bytes.get(start) != Some(&b'"') {
        return None;
    }
    let mut index = start + 1;
    while index < bytes.len() {
        match bytes[index] {
            b'\\' => index = index.checked_add(2)?,
            b'"' => return Some(index + 1),
            _ => index += 1,
        }
    }
    None
}

fn json_value_end(source: &str, start: usize) -> Option<usize> {
    let bytes = source.as_bytes();
    match *bytes.get(start)? {
        b'"' => json_string_end(source, start),
        b'{' | b'[' => {
            let mut levels = Vec::new();
            let mut index = start;
            while index < bytes.len() {
                match bytes[index] {
                    b'"' => index = json_string_end(source, index)?,
                    b'{' => {
                        levels.push(b'}');
                        index += 1;
                    }
                    b'[' => {
                        levels.push(b']');
                        index += 1;
                    }
                    closing @ (b'}' | b']') => {
                        if levels.pop()? != closing {
                            return None;
                        }
                        index += 1;
                        if levels.is_empty() {
                            return Some(index);
                        }
                    }
                    _ => index += 1,
                }
            }
            None
        }
        _ => {
            let mut index = start;
            while index < bytes.len()
                && !bytes[index].is_ascii_whitespace()
                && !matches!(bytes[index], b',' | b'}' | b']')
            {
                index += 1;
            }
            (index > start).then_some(index)
        }
    }
}

/// Return a top-level object's raw value token for one decoded JSON key.
/// The complete document is validated by serde_json first; this scanner only
/// retains the original decimal token so Rust's accurate `str::parse` can
/// reproduce Python's binary float at the CPU-report boundary.
fn raw_object_field<'a>(source: &'a str, field: &str) -> Option<&'a str> {
    let bytes = source.as_bytes();
    let mut index = 0;
    let mut found = None;
    skip_json_whitespace(source, &mut index);
    if bytes.get(index) != Some(&b'{') {
        return None;
    }
    index += 1;
    loop {
        skip_json_whitespace(source, &mut index);
        if bytes.get(index) == Some(&b'}') {
            return found;
        }
        let key_start = index;
        let key_end = json_string_end(source, key_start)?;
        let key: String = serde_json::from_str(&source[key_start..key_end]).ok()?;
        index = key_end;
        skip_json_whitespace(source, &mut index);
        if bytes.get(index) != Some(&b':') {
            return None;
        }
        index += 1;
        skip_json_whitespace(source, &mut index);
        let value_start = index;
        let value_end = json_value_end(source, value_start)?;
        if key == field {
            // json.loads retains the final value for a duplicate object key.
            found = Some(source[value_start..value_end].trim());
        }
        index = value_end;
        skip_json_whitespace(source, &mut index);
        match bytes.get(index) {
            Some(b',') => index += 1,
            Some(b'}') => return found,
            _ => return None,
        }
    }
}

fn raw_usage_seconds(value: &str) -> Option<f64> {
    match value {
        "true" => Some(1.0),
        "false" => Some(0.0),
        value if !value.starts_with('"') => value.parse().ok(),
        _ => None,
    }
}

fn raw_python_float(value: &str) -> Option<f64> {
    match value {
        "true" => Some(1.0),
        "false" => Some(0.0),
        value if value.starts_with('"') => serde_json::from_str::<String>(value)
            .ok()
            .and_then(|value| value.trim().parse().ok()),
        value => value.parse().ok(),
    }
}

fn corrected_json_float(value: &str) -> Option<Value> {
    if !value
        .bytes()
        .any(|character| matches!(character, b'.' | b'e' | b'E'))
    {
        return None;
    }
    serde_json::Number::from_f64(value.parse().ok()?).map(Value::Number)
}

fn round_nine(value: f64) -> f64 {
    if value.is_finite() {
        // Rust's fixed-precision formatter rounds the represented binary float
        // directly, matching Python's `round(value, 9)` more closely than
        // multiplying by 1e9 before rounding.
        format!("{value:.9}").parse().unwrap_or(value)
    } else {
        value
    }
}

fn resolved(path: &Path) -> String {
    fs::canonicalize(path)
        .unwrap_or_else(|_| {
            if path.is_absolute() {
                path.to_owned()
            } else {
                env::current_dir()
                    .unwrap_or_else(|_| PathBuf::from("."))
                    .join(path)
            }
        })
        .to_string_lossy()
        .into_owned()
}

/// Build the schema-compatible AgentScry comparison report.
pub fn build_comparison_report(
    pipeline_report_path: &Path,
    filtered_csv_path: &Path,
    root_command_substring: &str,
    tolerance_fraction: f64,
) -> Result<Value, ComparisonError> {
    let pipeline_text = fs::read_to_string(pipeline_report_path).map_err(|_| {
        ComparisonError::new(format!(
            "unable to read pipeline report: {}",
            pipeline_report_path.display()
        ))
    })?;
    let pipeline: Value = serde_json::from_str(&pipeline_text).map_err(|_| {
        ComparisonError::new(format!(
            "unable to read pipeline report: {}",
            pipeline_report_path.display()
        ))
    })?;
    let pipeline = pipeline.as_object().ok_or_else(|| {
        ComparisonError::new(format!(
            "unable to read pipeline report: {}",
            pipeline_report_path.display()
        ))
    })?;
    let cpu = pipeline
        .get("container_cpu")
        .and_then(Value::as_object)
        .ok_or_else(|| ComparisonError::new("pipeline report has no container_cpu.usage_sec"))?;
    let raw_cpu = raw_object_field(&pipeline_text, "container_cpu");
    let usage_sec = raw_cpu
        .and_then(|cpu| raw_object_field(cpu, "usage_sec"))
        .and_then(raw_usage_seconds)
        .or_else(|| pipeline_usage_seconds(cpu.get("usage_sec")))
        .ok_or_else(|| ComparisonError::new("pipeline report has no container_cpu.usage_sec"))?;
    let wall_sec = raw_cpu
        .and_then(|cpu| raw_object_field(cpu, "wall_sec"))
        .and_then(raw_python_float)
        .or_else(|| python_float(cpu.get("wall_sec")))
        .ok_or_else(|| {
            ComparisonError::new("pipeline report has no numeric container_cpu.wall_sec")
        })?;
    let average_logical_cores_used = raw_cpu
        .and_then(|cpu| raw_object_field(cpu, "average_logical_cores_used"))
        .and_then(corrected_json_float)
        .or_else(|| cpu.get("average_logical_cores_used").cloned())
        .unwrap_or(Value::Null);
    let aggregation = aggregate_process_tree(
        &read_filtered_csv(filtered_csv_path)?,
        root_command_substring,
    )?;
    let agent_usage_sec = aggregation.usage_sec();
    let delta_sec = agent_usage_sec - usage_sec;
    let delta_fraction = (usage_sec != 0.0).then_some(delta_sec / usage_sec);
    let cpu_within_tolerance =
        delta_fraction.is_some_and(|delta| delta.abs() <= tolerance_fraction);
    let complete_exit_accounting = aggregation.missing_rows.is_empty();
    let strict_equivalence = cpu_within_tolerance && complete_exit_accounting;
    let reason = if strict_equivalence {
        "AgentScry CPU is within tolerance and every captured replay descendant has exit accounting."
    } else if cpu_within_tolerance && !complete_exit_accounting {
        "AgentScry aggregate CPU is within tolerance, but replay descendants without exit CPU accounting prevent strict equivalence."
    } else if !complete_exit_accounting {
        "AgentScry CPU differs from container cgroup CPU beyond tolerance and replay descendants lack exit CPU accounting."
    } else {
        "AgentScry CPU differs from container cgroup CPU beyond tolerance."
    };

    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "pipeline_report": resolved(pipeline_report_path),
        "agentscry_filtered_csv": resolved(filtered_csv_path),
        "measurement_scope": {
            "container_cpu": "Disposable replay container cgroup v2 cpu.stat command interval.",
            "agentscry_cpu": "Privileged host-side AgentScry USER+SYS for captured descendant PIDs rooted at the replay container entrypoint; sums CPU portions apportioned across retained exec rows by asfilter --duration.",
        },
        "task_id": pipeline.get("task_id").cloned().unwrap_or(Value::Null),
        "trial_name": pipeline.get("trial_name").cloned().unwrap_or(Value::Null),
        "image_identity": pipeline.get("image_identity").cloned().unwrap_or(Value::Null),
        "container_cpu": {
            "usage_sec": usage_sec,
            "wall_sec": wall_sec,
            "average_logical_cores_used": average_logical_cores_used,
        },
        "agentscry_cpu": {
            "root_pid": aggregation.root_pid,
            "captured_descendant_pid_count": aggregation.descendant_pids.len(),
            "exit_accounted_pid_count": aggregation.exit_accounted_pids.len(),
            "cpu_accounted_row_count": aggregation.cpu_accounted_row_count(),
            "missing_exit_accounting_pid_count": aggregation.missing_rows.len(),
            "missing_exit_accounting": aggregation.missing_rows,
            "user_sec": round_nine(aggregation.user_sec),
            "system_sec": round_nine(aggregation.system_sec),
            "usage_sec": round_nine(agent_usage_sec),
            "average_logical_cores_on_container_wall": if wall_sec != 0.0 {
                Some(round_nine(agent_usage_sec / wall_sec))
            } else {
                None
            },
        },
        "comparison": {
            "agentscry_minus_container_usage_sec": round_nine(delta_sec),
            "agentscry_minus_container_fraction": delta_fraction.map(round_nine),
            "tolerance_fraction": tolerance_fraction,
            "cpu_within_tolerance": cpu_within_tolerance,
            "complete_exit_accounting": complete_exit_accounting,
            "equivalent_for_utilization": strict_equivalence,
            "reason": reason,
        },
    }))
}

fn ensure_ascii_json(serialized: &str) -> String {
    let mut escaped = String::with_capacity(serialized.len());
    for character in serialized.chars() {
        if character.is_ascii() {
            escaped.push(character);
        } else {
            let mut units = [0_u16; 2];
            for unit in character.encode_utf16(&mut units).iter() {
                escaped.push_str(&format!("\\u{unit:04x}"));
            }
        }
    }
    escaped
}

fn python_float_repr(value: f64) -> String {
    let rendered = format!("{value:?}");
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return rendered;
    };
    let Ok(exponent) = exponent.parse::<i32>() else {
        return rendered;
    };
    format!("{mantissa}e{exponent:+03}")
}

/// Serialize with the formatting used by `json.dumps(..., indent=2,
/// sort_keys=True)`. In particular, serde_json 1.0.114's number formatter does
/// not always choose CPython's shortest float representation.
fn write_python_json(value: &Value, indent: usize, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                output.push_str(&value.to_string());
            } else if let Some(value) = number.as_u64() {
                output.push_str(&value.to_string());
            } else if let Some(value) = number.as_f64() {
                output.push_str(&python_float_repr(value));
            }
        }
        Value::String(value) => output
            .push_str(&serde_json::to_string(value).expect("a JSON string is always serializable")),
        Value::Array(values) => {
            if values.is_empty() {
                output.push_str("[]");
                return;
            }
            output.push_str("[\n");
            for (index, value) in values.iter().enumerate() {
                output.push_str(&" ".repeat(indent + 2));
                write_python_json(value, indent + 2, output);
                if index + 1 != values.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&" ".repeat(indent));
            output.push(']');
        }
        Value::Object(values) => {
            if values.is_empty() {
                output.push_str("{}");
                return;
            }
            output.push_str("{\n");
            let mut fields: Vec<_> = values.iter().collect();
            fields.sort_by(|(left, _), (right, _)| left.cmp(right));
            for (index, (key, value)) in fields.iter().enumerate() {
                output.push_str(&" ".repeat(indent + 2));
                output.push_str(
                    &serde_json::to_string(key).expect("a JSON object key is always serializable"),
                );
                output.push_str(": ");
                write_python_json(value, indent + 2, output);
                if index + 1 != fields.len() {
                    output.push(',');
                }
                output.push('\n');
            }
            output.push_str(&" ".repeat(indent));
            output.push('}');
        }
    }
}

pub fn write_comparison_report(report: &Value, output: &Path) -> Result<(), ComparisonError> {
    if let Some(parent) = output
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            ComparisonError::new(format!(
                "unable to create output directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let mut serialized = String::new();
    write_python_json(report, 0, &mut serialized);
    fs::write(output, ensure_ascii_json(&serialized) + "\n").map_err(|error| {
        ComparisonError::new(format!(
            "unable to write comparison report {}: {error}",
            output.display()
        ))
    })
}

#[derive(Debug)]
struct Cli {
    pipeline_report: PathBuf,
    agentscry_filtered_csv: PathBuf,
    output: PathBuf,
    root_command_substring: String,
    tolerance_fraction: f64,
}

#[derive(Debug)]
enum CliParse {
    Help,
    Run(Cli),
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
        .trim()
        .parse()
        .map_err(|_| format!("argument {option}: invalid float value: {display:?}"))
}

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut positional = Vec::new();
    let mut output = None;
    let mut root_command_substring = "/tasktrace-profile-entrypoint.sh".to_owned();
    let mut tolerance_fraction = 0.05;
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
            Some("--output") => {
                output = Some(PathBuf::from(value_for(&mut index, "--output")?));
            }
            Some("--root-command-substring") => {
                let value = value_for(&mut index, "--root-command-substring")?;
                root_command_substring = value
                    .into_string()
                    .map_err(|_| "argument --root-command-substring: invalid UTF-8".to_owned())?;
            }
            Some("--tolerance-fraction") => {
                tolerance_fraction = parse_float(
                    "--tolerance-fraction",
                    value_for(&mut index, "--tolerance-fraction")?,
                )?;
            }
            Some(value) if options && value.starts_with('-') => {
                return Err(format!(
                    "unrecognized arguments: {}",
                    argument.to_string_lossy()
                ));
            }
            _ => positional.push(PathBuf::from(argument)),
        }
        index += 1;
    }

    let mut missing = Vec::new();
    if positional.is_empty() {
        missing.push("pipeline_report");
    }
    if positional.len() < 2 {
        missing.push("agentscry_filtered_csv");
    }
    if output.is_none() {
        missing.push("--output");
    }
    if !missing.is_empty() {
        return Err(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    if positional.len() > 2 {
        return Err(format!(
            "unrecognized arguments: {}",
            positional[2..]
                .iter()
                .map(|path| path.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    if tolerance_fraction < 0.0 {
        return Err("--tolerance-fraction must not be negative".to_owned());
    }
    Ok(CliParse::Run(Cli {
        pipeline_report: positional.remove(0),
        agentscry_filtered_csv: positional.remove(0),
        output: output.expect("required output validated"),
        root_command_substring,
        tolerance_fraction,
    }))
}

fn print_usage(public_program: &str, output: &mut dyn io::Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] --output OUTPUT [--root-command-substring ROOT_COMMAND_SUBSTRING] [--tolerance-fraction TOLERANCE_FRACTION] pipeline_report agentscry_filtered_csv"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!(
        "Compare a profiled offline replay's container CPU report with a filtered AgentScry process trace from that same run."
    );
    println!();
    println!("positional arguments:");
    println!("  pipeline_report");
    println!("  agentscry_filtered_csv");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --output OUTPUT       JSON comparison report.");
    println!("  --root-command-substring ROOT_COMMAND_SUBSTRING");
    println!(
        "                        Substring identifying the measured replay container entry shell."
    );
    println!("  --tolerance-fraction TOLERANCE_FRACTION");
    println!("                        Maximum fractional CPU delta accepted only when no descendant accounting is missing.");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
}

fn python_bool(value: bool) -> &'static str {
    if value {
        "True"
    } else {
        "False"
    }
}

/// Native CLI entry point used by the public dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let options = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(options)) => options,
        Err(error) => return cli_error(public_program, &error),
    };
    let report = match build_comparison_report(
        &options.pipeline_report,
        &options.agentscry_filtered_csv,
        &options.root_command_substring,
        options.tolerance_fraction,
    ) {
        Ok(report) => report,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    if let Err(error) = write_comparison_report(&report, &options.output) {
        return cli_error(public_program, &error.to_string());
    }
    let agent = &report["agentscry_cpu"];
    let comparison = &report["comparison"];
    println!(
        "Container CPU seconds: {:.9}",
        report["container_cpu"]["usage_sec"].as_f64().unwrap_or(0.0)
    );
    println!(
        "AgentScry CPU seconds: {:.9}",
        agent["usage_sec"].as_f64().unwrap_or(0.0)
    );
    println!(
        "AgentScry missing exit-accounted PIDs: {}",
        agent["missing_exit_accounting_pid_count"]
    );
    println!(
        "CPU within tolerance: {}",
        python_bool(
            comparison["cpu_within_tolerance"]
                .as_bool()
                .unwrap_or(false)
        )
    );
    println!(
        "Strict equivalent for utilization: {}",
        python_bool(
            comparison["equivalent_for_utilization"]
                .as_bool()
                .unwrap_or(false)
        )
    );
    println!("Reason: {}", comparison["reason"].as_str().unwrap_or(""));
    println!("Report: {}", options.output.display());
    i32::from(
        !comparison["equivalent_for_utilization"]
            .as_bool()
            .unwrap_or(false),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    fn write_pipeline(path: &Path, usage_sec: f64, wall_sec: f64) {
        fs::write(
            path,
            serde_json::to_vec(&json!({
                "task_id": "task",
                "trial_name": "task.1",
                "image_identity": {"id": "sha256:image"},
                "container_cpu": {
                    "usage_sec": usage_sec,
                    "wall_sec": wall_sec,
                    "average_logical_cores_used": usage_sec / wall_sec,
                },
            }))
            .unwrap(),
        )
        .unwrap();
    }

    #[test]
    fn sums_every_apportioned_exec_row() {
        let directory = tempdir().unwrap();
        let pipeline = directory.path().join("pipeline.json");
        let filtered = directory.path().join("agentscry.csv");
        write_pipeline(&pipeline, 1.0, 1.0);
        fs::write(
            &filtered,
            concat!(
                "EPOCH,DURATION,COMM,PPID,PID,CPU,USER,SYS,NTHD,ARGS\n",
                ",,,,1,0,9,9,,/unrelated\n",
                ",,,1,10,0,0.100,0.100,,/bin/sh /tasktrace-profile-entrypoint.sh\n",
                ",,,10,11,0,0.200,0.200,,/bin/bash -c python3 -m replay\n",
                ",,,10,11,0,0.200,0.200,,/usr/bin/python3 -m replay\n",
            ),
        )
        .unwrap();

        let report =
            build_comparison_report(&pipeline, &filtered, "/tasktrace-profile-entrypoint.sh", 0.01)
                .unwrap();
        assert_eq!(report["agentscry_cpu"]["usage_sec"], json!(1.0));
        assert_eq!(
            report["agentscry_cpu"]["exit_accounted_pid_count"],
            json!(2)
        );
        assert_eq!(report["agentscry_cpu"]["cpu_accounted_row_count"], json!(3));
        assert_eq!(report["comparison"]["cpu_within_tolerance"], json!(true));
        assert_eq!(
            report["comparison"]["equivalent_for_utilization"],
            json!(true)
        );
    }

    #[test]
    fn missing_exit_accounting_prevents_strict_equivalence() {
        let directory = tempdir().unwrap();
        let pipeline = directory.path().join("pipeline.json");
        let filtered = directory.path().join("agentscry.csv");
        write_pipeline(&pipeline, 1.0, 1.0);
        fs::write(
            &filtered,
            concat!(
                "EPOCH,DURATION,COMM,PPID,PID,CPU,USER,SYS,NTHD,ARGS\n",
                ",,,1,10,0,0.500,0.500,,/bin/sh /tasktrace-profile-entrypoint.sh\n",
                ",12,grep,10,12,0,,,,/usr/bin/grep input.log\n",
            ),
        )
        .unwrap();

        let report =
            build_comparison_report(&pipeline, &filtered, "/tasktrace-profile-entrypoint.sh", 0.01)
                .unwrap();
        assert_eq!(
            report["agentscry_cpu"]["missing_exit_accounting_pid_count"],
            json!(1)
        );
        assert_eq!(report["comparison"]["cpu_within_tolerance"], json!(true));
        assert_eq!(
            report["comparison"]["equivalent_for_utilization"],
            json!(false)
        );
        assert!(report["comparison"]["reason"]
            .as_str()
            .unwrap()
            .contains("without exit CPU accounting prevent strict equivalence"));
    }

    #[test]
    fn rejects_ambiguous_root_and_empty_csv() {
        let rows = vec![
            BTreeMap::from([
                ("PID".to_owned(), "10".to_owned()),
                ("PPID".to_owned(), "1".to_owned()),
                ("USER".to_owned(), "0".to_owned()),
                ("SYS".to_owned(), "0".to_owned()),
                ("ARGS".to_owned(), "root-marker".to_owned()),
            ]),
            BTreeMap::from([
                ("PID".to_owned(), "11".to_owned()),
                ("PPID".to_owned(), "1".to_owned()),
                ("USER".to_owned(), "0".to_owned()),
                ("SYS".to_owned(), "0".to_owned()),
                ("ARGS".to_owned(), "root-marker".to_owned()),
            ]),
        ];
        let error = aggregate_process_tree(&rows, "root-marker").unwrap_err();
        assert!(error.to_string().contains("found [10, 11]"));

        let directory = tempdir().unwrap();
        let csv = directory.path().join("empty.csv");
        fs::write(&csv, "PID,PPID,USER,SYS,ARGS\n").unwrap();
        let error = read_filtered_csv(&csv).unwrap_err();
        assert_eq!(
            error.to_string(),
            "AgentScry CSV is empty or missing required columns: ARGS, PID, PPID, SYS, USER"
        );
    }

    #[test]
    fn serializer_matches_python_scientific_notation() {
        let mut rendered = String::new();
        write_python_json(
            &json!({"large": 1e20_f64, "small": 1e-5_f64, "tiny": 1e-9_f64}),
            0,
            &mut rendered,
        );
        assert_eq!(
            rendered,
            concat!(
                "{\n",
                "  \"large\": 1e+20,\n",
                "  \"small\": 1e-05,\n",
                "  \"tiny\": 1e-09\n",
                "}"
            )
        );
    }
}
