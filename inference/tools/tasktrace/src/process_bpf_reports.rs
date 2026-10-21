// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed analysis for externally captured BPF process evidence.
//!
//! The collector remains an external privileged component.  This module owns
//! the unprivileged, deterministic proof construction exposed by the two
//! `report process-inventory-bpf*` command leaves.

use regex::Regex;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashSet};
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs::{self, File};
use std::io::Read;
use std::path::{Path, PathBuf};

pub const PROCESS_BPF_SCHEMA_VERSION: &str = "tasktrace-bpf-process-inventory.v1";
pub const EXPECTED_SCHEMA_VERSION: &str = "tasktrace-representative-process-expectation.v1";
pub const REPEATABILITY_SCHEMA_VERSION: &str = "tasktrace-representative-process-repeatability.v1";

const REQUIRED_COLUMNS: &[&str] = &["EPOCH", "COMM", "PPID", "PID", "CPU", "RET", "ARGS"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessBpfEvidenceError(String);

impl ProcessBpfEvidenceError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProcessBpfEvidenceError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ProcessBpfEvidenceError {}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EventKind {
    Fork,
    ExitGroup,
    Exec,
}

#[derive(Clone, Debug)]
struct Event {
    index: usize,
    timestamp_ns: i64,
    ppid: i64,
    pid: i64,
    cpu: i64,
    retval: Option<i64>,
    args: String,
}

impl Event {
    fn kind(&self) -> EventKind {
        match self.args.as_str() {
            "[fork]" => EventKind::Fork,
            "[exit_group]" => EventKind::ExitGroup,
            _ => EventKind::Exec,
        }
    }

    fn binary(&self) -> Option<&str> {
        (self.kind() == EventKind::Exec && !self.args.is_empty()).then(|| {
            self.args
                .split_once(' ')
                .map_or(self.args.as_str(), |item| item.0)
        })
    }
}

fn sha256(path: &Path) -> Result<String, ProcessBpfEvidenceError> {
    let mut source = File::open(path).map_err(|error| {
        ProcessBpfEvidenceError::new(format!(
            "unable to read artifact {}: {error}",
            path.display()
        ))
    })?;
    let mut digest = Sha256::new();
    let mut block = vec![0_u8; 1024 * 1024];
    loop {
        let count = source.read(&mut block).map_err(|error| {
            ProcessBpfEvidenceError::new(format!(
                "unable to read artifact {}: {error}",
                path.display()
            ))
        })?;
        if count == 0 {
            break;
        }
        digest.update(&block[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn resolved(path: &Path) -> Result<String, ProcessBpfEvidenceError> {
    let resolved = fs::canonicalize(path).unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_owned()
        } else {
            env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        }
    });
    resolved
        .into_os_string()
        .into_string()
        .map_err(|_| ProcessBpfEvidenceError::new("artifact path is not valid UTF-8"))
}

fn artifact(path: &Path) -> Result<Value, ProcessBpfEvidenceError> {
    let metadata = fs::metadata(path).map_err(|error| {
        ProcessBpfEvidenceError::new(format!(
            "unable to inspect artifact {}: {error}",
            path.display()
        ))
    })?;
    Ok(json!({
        "path": resolved(path)?,
        "sha256": sha256(path)?,
        "size_bytes": metadata.len(),
    }))
}

fn read_object(path: &Path, label: &str) -> Result<Value, ProcessBpfEvidenceError> {
    let text = fs::read_to_string(path).map_err(|_| {
        ProcessBpfEvidenceError::new(format!("unable to read {label}: {}", path.display()))
    })?;
    let value: Value = serde_json::from_str(&text).map_err(|_| {
        ProcessBpfEvidenceError::new(format!("unable to read {label}: {}", path.display()))
    })?;
    if !value.is_object() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "{label} must contain a JSON object"
        )));
    }
    Ok(value)
}

fn decimal_epoch_ns(value: &str, row_number: usize) -> Result<i64, ProcessBpfEvidenceError> {
    let raw = value.trim();
    let (negative, unsigned) = match raw.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, raw.strip_prefix('+').unwrap_or(raw)),
    };
    let (significand, exponent) = match unsigned.find(['e', 'E']) {
        Some(index) => {
            let exponent = unsigned[index + 1..].parse::<i64>().map_err(|_| {
                ProcessBpfEvidenceError::new(format!(
                    "row {row_number}: EPOCH is not a decimal timestamp"
                ))
            })?;
            (&unsigned[..index], exponent)
        }
        None => (unsigned, 0),
    };
    let mut digits = String::new();
    let mut decimal_places = 0_i64;
    let mut seen_dot = false;
    for character in significand.chars() {
        match character {
            '0'..='9' => {
                digits.push(character);
                if seen_dot {
                    decimal_places += 1;
                }
            }
            '.' if !seen_dot => seen_dot = true,
            _ => {
                return Err(ProcessBpfEvidenceError::new(format!(
                    "row {row_number}: EPOCH is not a decimal timestamp"
                )))
            }
        }
    }
    if digits.is_empty() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "row {row_number}: EPOCH is not a decimal timestamp"
        )));
    }
    if digits.bytes().all(|digit| digit == b'0') {
        return Ok(0);
    }
    let scale = exponent
        .checked_add(9)
        .and_then(|value| value.checked_sub(decimal_places))
        .ok_or_else(|| {
            ProcessBpfEvidenceError::new(format!(
                "row {row_number}: EPOCH is not a decimal timestamp"
            ))
        })?;
    let integral_digits = if scale < 0 {
        let removed = scale.unsigned_abs();
        if removed > digits.len() as u64 {
            return Err(ProcessBpfEvidenceError::new(format!(
                "row {row_number}: EPOCH has sub-nanosecond precision"
            )));
        }
        let split = digits.len() - removed as usize;
        if !digits.as_bytes()[split..]
            .iter()
            .all(|digit| *digit == b'0')
        {
            return Err(ProcessBpfEvidenceError::new(format!(
                "row {row_number}: EPOCH has sub-nanosecond precision"
            )));
        }
        digits[..split].to_owned()
    } else {
        let significant_len = digits.trim_start_matches('0').len();
        let result_len = (significant_len as u64).saturating_add(scale as u64);
        if result_len > 20 {
            return Err(ProcessBpfEvidenceError::new(format!(
                "row {row_number}: EPOCH is not a decimal timestamp"
            )));
        }
        let mut result = digits;
        result.extend(std::iter::repeat('0').take(scale as usize));
        result
    };
    let mut scaled = integral_digits.parse::<i128>().map_err(|_| {
        ProcessBpfEvidenceError::new(format!(
            "row {row_number}: EPOCH is not a decimal timestamp"
        ))
    })?;
    if negative {
        scaled = -scaled;
    }
    i64::try_from(scaled).map_err(|_| {
        ProcessBpfEvidenceError::new(format!(
            "row {row_number}: EPOCH is not a decimal timestamp"
        ))
    })
}

fn integer(
    value: &str,
    field: &str,
    row_number: usize,
    optional: bool,
) -> Result<Option<i64>, ProcessBpfEvidenceError> {
    if optional && value.is_empty() {
        return Ok(None);
    }
    let result = value.parse::<i64>().map_err(|_| {
        ProcessBpfEvidenceError::new(format!("row {row_number}: {field} is not an integer"))
    })?;
    if result < 0 && matches!(field, "PID" | "PPID" | "CPU") {
        return Err(ProcessBpfEvidenceError::new(format!(
            "row {row_number}: {field} must be non-negative"
        )));
    }
    Ok(Some(result))
}

fn parse_csv_records(text: &str) -> Vec<Vec<String>> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut characters = text.chars().peekable();
    let mut quoted = false;
    let mut started = false;
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
            started = true;
            continue;
        }
        match character {
            '"' if field.is_empty() => {
                quoted = true;
                started = true;
            }
            ',' => {
                record.push(std::mem::take(&mut field));
                started = true;
            }
            '\n' => {
                if started || !field.is_empty() || !record.is_empty() {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                } else if records.is_empty() {
                    // csv.DictReader binds even an empty first record as its
                    // header; later blank records are ignored.
                    records.push(Vec::new());
                }
                started = false;
            }
            '\r' => {
                if characters.peek() == Some(&'\n') {
                    characters.next();
                }
                if started || !field.is_empty() || !record.is_empty() {
                    record.push(std::mem::take(&mut field));
                    records.push(std::mem::take(&mut record));
                } else if records.is_empty() {
                    records.push(Vec::new());
                }
                started = false;
            }
            other => {
                field.push(other);
                started = true;
            }
        }
    }
    if started || !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }
    records
}

fn read_events(path: &Path) -> Result<Vec<Event>, ProcessBpfEvidenceError> {
    let text = fs::read_to_string(path).map_err(|_| {
        ProcessBpfEvidenceError::new(format!("unable to read process trace: {}", path.display()))
    })?;
    let mut records = parse_csv_records(&text).into_iter();
    let headers = records.next().unwrap_or_default();
    let positions: BTreeMap<&str, usize> = headers
        .iter()
        .enumerate()
        .map(|(index, header)| (header.as_str(), index))
        .collect();
    let missing: Vec<_> = REQUIRED_COLUMNS
        .iter()
        .filter(|column| !positions.contains_key(**column))
        .copied()
        .collect();
    if !missing.is_empty() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "process trace is missing columns: {}",
            missing.join(", ")
        )));
    }
    let field = |row: &[String], name: &str| -> String {
        positions
            .get(name)
            .and_then(|index| row.get(*index))
            .cloned()
            .unwrap_or_default()
    };
    let mut events = Vec::new();
    for (offset, row) in records.enumerate() {
        let row_number = offset + 2;
        if row.len() > headers.len() {
            return Err(ProcessBpfEvidenceError::new(format!(
                "row {row_number}: malformed extra CSV fields"
            )));
        }
        let ppid = integer(&field(&row, "PPID"), "PPID", row_number, false)?.unwrap();
        let pid = integer(&field(&row, "PID"), "PID", row_number, false)?.unwrap();
        let cpu = integer(&field(&row, "CPU"), "CPU", row_number, false)?.unwrap();
        let retval = integer(&field(&row, "RET"), "RET", row_number, true)?;
        events.push(Event {
            index: row_number,
            timestamp_ns: decimal_epoch_ns(&field(&row, "EPOCH"), row_number)?,
            ppid,
            pid,
            cpu,
            retval,
            args: field(&row, "ARGS"),
        });
    }
    if events.is_empty() {
        return Err(ProcessBpfEvidenceError::new("process trace has no events"));
    }
    Ok(events)
}

fn cpuset(value: &str) -> Result<BTreeSet<i64>, ProcessBpfEvidenceError> {
    let mut result = BTreeSet::new();
    for raw_item in value.split(',') {
        let item = raw_item.trim();
        if item.is_empty() {
            return Err(ProcessBpfEvidenceError::new(format!(
                "invalid empty cpuset component in {value:?}"
            )));
        }
        if let Some((first, last)) = item.split_once('-') {
            let first = first
                .parse::<i64>()
                .map_err(|_| ProcessBpfEvidenceError::new(format!("invalid cpuset {value:?}")))?;
            let last = last
                .parse::<i64>()
                .map_err(|_| ProcessBpfEvidenceError::new(format!("invalid cpuset {value:?}")))?;
            if first < 0 || last < first {
                return Err(ProcessBpfEvidenceError::new(format!(
                    "invalid cpuset {value:?}"
                )));
            }
            result.extend(first..=last);
        } else {
            let cpu = item
                .parse::<i64>()
                .map_err(|_| ProcessBpfEvidenceError::new(format!("invalid cpuset {value:?}")))?;
            if cpu < 0 {
                return Err(ProcessBpfEvidenceError::new(format!(
                    "invalid cpuset {value:?}"
                )));
            }
            result.insert(cpu);
        }
    }
    Ok(result)
}

fn loss_count(path: &Path) -> Result<u64, ProcessBpfEvidenceError> {
    let text = fs::read_to_string(path).map_err(|_| {
        ProcessBpfEvidenceError::new(format!(
            "unable to read tracer diagnostics: {}",
            path.display()
        ))
    })?;
    let marker = Regex::new(r"(?m)^AGENTSCRY_LOST_EVENTS=([0-9]+)$").unwrap();
    let captures: Vec<_> = marker.captures_iter(&text).collect();
    if captures.len() != 1 {
        return Err(ProcessBpfEvidenceError::new(
            "tracer diagnostics must contain exactly one terminal lost-event marker",
        ));
    }
    let digits = &captures[0][1];
    let nonzero = digits.bytes().any(|digit| digit != b'0');
    if nonzero || text.contains("perf-buffer loss:") {
        let count = digits.trim_start_matches('0');
        let count = if count.is_empty() { "0" } else { count };
        return Err(ProcessBpfEvidenceError::new(format!(
            "process trace dropped {count} event(s)"
        )));
    }
    Ok(0)
}

fn nested<'a>(value: &'a Value, keys: &[&str]) -> Option<&'a Value> {
    keys.iter()
        .try_fold(value, |current, key| current.as_object()?.get(*key))
}

fn is_zero(value: Option<&Value>) -> bool {
    value
        .and_then(Value::as_f64)
        .is_some_and(|number| number == 0.0)
}

fn pipeline_invariants(
    pipeline: &Value,
) -> Result<(String, BTreeSet<i64>), ProcessBpfEvidenceError> {
    let summary = pipeline
        .get("replay_summary")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProcessBpfEvidenceError::new("pipeline report has no replay_summary object")
        })?;
    let checks = [
        "actions_timed_out",
        "source_outcome_mismatch_count",
        "actions_skipped_after_timeout",
    ];
    let nonzero: Vec<_> = checks
        .iter()
        .filter(|name| !is_zero(summary.get(**name)))
        .map(|name| format!("'{name}': {}", py_repr(summary.get(*name))))
        .collect();
    if !nonzero.is_empty() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "replay fidelity gate failed: {{{}}}",
            nonzero.join(", ")
        )));
    }
    if pipeline.get("status") != Some(&Value::String("valid".to_owned()))
        || pipeline.get("profiling_valid") != Some(&Value::Bool(true))
    {
        return Err(ProcessBpfEvidenceError::new("pipeline report is not valid"));
    }
    let effective_network_mode = pipeline
        .get("network_policy")
        .and_then(Value::as_object)
        .and_then(|policy| policy.get("effective_mode"))
        .cloned()
        .unwrap_or_else(|| {
            Value::String(
                if pipeline.get("network") == Some(&Value::String("none".to_owned())) {
                    "no-network"
                } else {
                    "public"
                }
                .to_owned(),
            )
        });
    if effective_network_mode != Value::String("no-network".to_owned()) {
        return Err(ProcessBpfEvidenceError::new(
            "representative replay must use effective no-network policy",
        ));
    }
    let cpuset_text = nested(pipeline, &["execution_plan", "extensions", "cpuset_cpus"])
        .and_then(Value::as_str)
        .ok_or_else(|| ProcessBpfEvidenceError::new("execution plan has no fixed cpuset"))?;
    let source_sha = pipeline
        .get("source_trajectory_sha256")
        .and_then(Value::as_str)
        .filter(|value| value.len() == 64)
        .ok_or_else(|| {
            ProcessBpfEvidenceError::new("pipeline report has no source trajectory digest")
        })?;
    Ok((source_sha.to_owned(), cpuset(cpuset_text)?))
}

fn python_string(value: &str) -> String {
    let mut output = String::with_capacity(value.len() + 2);
    output.push('"');
    for character in value.chars() {
        match character {
            '"' => output.push_str("\\\""),
            '\\' => output.push_str("\\\\"),
            '\u{08}' => output.push_str("\\b"),
            '\u{0c}' => output.push_str("\\f"),
            '\n' => output.push_str("\\n"),
            '\r' => output.push_str("\\r"),
            '\t' => output.push_str("\\t"),
            value if (' '..='~').contains(&value) => output.push(value),
            value if (value as u32) <= 0xffff => {
                output.push_str(&format!("\\u{:04x}", value as u32));
            }
            value => {
                let scalar = value as u32 - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                output.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    output.push('"');
    output
}

fn render_json(value: &Value, indent: Option<usize>, depth: usize, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) => output.push_str(&value.to_string()),
        Value::String(value) => output.push_str(&python_string(value)),
        Value::Array(values) => {
            if values.is_empty() {
                output.push_str("[]");
                return;
            }
            output.push('[');
            for (index, item) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                if let Some(width) = indent {
                    output.push('\n');
                    output.push_str(&" ".repeat((depth + 1) * width));
                }
                render_json(item, indent, depth + 1, output);
            }
            if let Some(width) = indent {
                output.push('\n');
                output.push_str(&" ".repeat(depth * width));
            }
            output.push(']');
        }
        Value::Object(values) => {
            if values.is_empty() {
                output.push_str("{}");
                return;
            }
            output.push('{');
            let mut entries: Vec<_> = values.iter().collect();
            entries.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, item)) in entries.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                if let Some(width) = indent {
                    output.push('\n');
                    output.push_str(&" ".repeat((depth + 1) * width));
                }
                output.push_str(&python_string(key));
                output.push_str(if indent.is_some() { ": " } else { ":" });
                render_json(item, indent, depth + 1, output);
            }
            if let Some(width) = indent {
                output.push('\n');
                output.push_str(&" ".repeat(depth * width));
            }
            output.push('}');
        }
    }
}

fn canonical_json(value: &Value) -> String {
    let mut output = String::new();
    render_json(value, None, 0, &mut output);
    output
}

fn py_repr(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_owned(),
        Some(Value::Bool(value)) => (if *value { "True" } else { "False" }).to_owned(),
        Some(Value::String(value)) => {
            format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'"))
        }
        Some(value) => canonical_json(value),
    }
}

fn canonical_tree(
    root_pid: i64,
    children: &BTreeMap<i64, BTreeSet<i64>>,
    successful_execs: &BTreeMap<i64, Vec<Event>>,
    ancestors: &HashSet<i64>,
) -> Result<Value, ProcessBpfEvidenceError> {
    if ancestors.contains(&root_pid) {
        return Err(ProcessBpfEvidenceError::new(
            "captured fork graph contains a cycle",
        ));
    }
    let mut next_ancestors = ancestors.clone();
    next_ancestors.insert(root_pid);
    let mut child_nodes = Vec::new();
    if let Some(child_ids) = children.get(&root_pid) {
        for child in child_ids {
            child_nodes.push(canonical_tree(
                *child,
                children,
                successful_execs,
                &next_ancestors,
            )?);
        }
    }
    child_nodes.sort_by_cached_key(canonical_json);
    Ok(json!({
        "exec_chain": successful_execs
            .get(&root_pid)
            .into_iter()
            .flatten()
            .map(|event| event.binary().map(str::to_owned))
            .collect::<Vec<_>>(),
        "children": child_nodes,
    }))
}

fn expected_inventory(value: Option<&Value>) -> Option<BTreeMap<String, i64>> {
    let values = value?.as_array()?;
    let mut result = BTreeMap::new();
    for item in values {
        let item = item.as_object()?;
        let binary = item.get("binary")?.as_str()?.to_owned();
        let count = item.get("exec_count")?.as_i64()?;
        if count < 1 || result.insert(binary, count).is_some() {
            return None;
        }
    }
    Some(result)
}

fn bound_artifact_error(value: Option<&Value>, label: &str) -> Option<String> {
    let value = match value.and_then(Value::as_object) {
        Some(value) => value,
        None => return Some(format!("{label} identity is missing")),
    };
    let path = value.get("path").and_then(Value::as_str);
    let expected_sha = value.get("sha256").and_then(Value::as_str);
    let expected_size = value.get("size_bytes").and_then(Value::as_u64);
    if path.is_none()
        || expected_sha.map_or(true, |sha| {
            sha.len() != 64
                || !sha
                    .bytes()
                    .all(|byte| byte.is_ascii_hexdigit() && !byte.is_ascii_uppercase())
        })
        || expected_size.is_none()
    {
        return Some(format!("{label} identity is malformed"));
    }
    let path = Path::new(path.unwrap());
    if !path.is_file() {
        return Some(format!("{label} artifact is missing: {}", path.display()));
    }
    let actual_size = fs::metadata(path).ok().map(|metadata| metadata.len());
    let actual_sha = sha256(path).ok();
    if actual_size != expected_size || actual_sha.as_deref() != expected_sha {
        return Some(format!(
            "{label} artifact no longer matches its bound content identity"
        ));
    }
    None
}

fn validate_expected(report: &Value, expected: &Value) -> Vec<String> {
    let mut reasons = Vec::new();
    if expected.get("schema_version") != Some(&Value::String(EXPECTED_SCHEMA_VERSION.to_owned())) {
        reasons.push("expected manifest schema_version is unsupported".to_owned());
    }
    let comparisons = [
        (
            "source_trajectory_sha256",
            &["source", "trajectory_sha256"][..],
        ),
        ("image_id", &["source", "image_id"][..]),
        ("image_repo_digest", &["source", "image_repo_digest"][..]),
        ("cpuset_cpus", &["scope", "cpuset_cpus"][..]),
        ("root_argv_prefix", &["scope", "root_argv_prefix"][..]),
        ("task_count", &["summary", "task_count"][..]),
        (
            "successful_exec_count",
            &["summary", "successful_exec_count"][..],
        ),
        (
            "failed_exec_attempt_count",
            &["summary", "failed_exec_attempt_count"][..],
        ),
        ("canonical_tree_sha256", &["canonical_tree_sha256"][..]),
    ];
    for (field, path) in comparisons {
        let actual = nested(report, path);
        let wanted = expected.get(field);
        if wanted != actual {
            reasons.push(format!(
                "expected {field}={}, observed {}",
                py_repr(wanted),
                py_repr(actual)
            ));
        }
    }
    match expected_inventory(expected.get("binary_inventory")) {
        None => reasons.push("expected binary_inventory is malformed".to_owned()),
        Some(wanted) => {
            let actual: BTreeMap<String, i64> = report["binary_inventory"]
                .as_array()
                .unwrap()
                .iter()
                .map(|item| {
                    (
                        item["binary"].as_str().unwrap().to_owned(),
                        item["exec_count"].as_i64().unwrap(),
                    )
                })
                .collect();
            if wanted != actual {
                reasons.push("binary inventory differs from the pinned expectation".to_owned());
            }
        }
    }
    reasons
}

/// Build one exact, ancestry-scoped binary inventory proof.
pub fn build_process_inventory_report(
    raw_csv_path: &Path,
    tracer_stderr_path: &Path,
    pipeline_report_path: &Path,
    root_argv_prefix: &str,
    expected_manifest_path: Option<&Path>,
    collector_path: Option<&Path>,
) -> Result<Value, ProcessBpfEvidenceError> {
    if !root_argv_prefix.starts_with('/') {
        return Err(ProcessBpfEvidenceError::new(
            "root_argv_prefix must begin with an absolute path",
        ));
    }
    let pipeline = read_object(pipeline_report_path, "pipeline report")?;
    let (source_sha, allocated_cpus) = pipeline_invariants(&pipeline)?;
    let loss_count = loss_count(tracer_stderr_path)?;
    let events = read_events(raw_csv_path)?;
    let observed_cpus: BTreeSet<_> = events.iter().map(|event| event.cpu).collect();
    let outside: Vec<_> = observed_cpus.difference(&allocated_cpus).copied().collect();
    if !outside.is_empty() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "tracer emitted events outside the execution plan cpuset: {outside:?}"
        )));
    }

    let roots: Vec<_> = events
        .iter()
        .filter(|event| {
            event.kind() == EventKind::Exec
                && event.retval == Some(0)
                && event.args.starts_with(root_argv_prefix)
        })
        .collect();
    if roots.len() != 1 {
        return Err(ProcessBpfEvidenceError::new(format!(
            "expected one successful root exec matching {root_argv_prefix:?}; observed {}",
            roots.len()
        )));
    }
    let root_pid = roots[0].pid;

    let mut fork_by_child = BTreeMap::new();
    let mut children: BTreeMap<i64, BTreeSet<i64>> = BTreeMap::new();
    for event in &events {
        if event.kind() != EventKind::Fork {
            continue;
        }
        if fork_by_child.insert(event.pid, event.clone()).is_some() {
            return Err(ProcessBpfEvidenceError::new(format!(
                "task ID {} has duplicate fork events; PID reuse is unsupported",
                event.pid
            )));
        }
        children.entry(event.ppid).or_default().insert(event.pid);
    }

    let mut reachable = BTreeSet::from([root_pid]);
    let mut pending = vec![root_pid];
    while let Some(parent) = pending.pop() {
        let child_ids: Vec<_> = children
            .get(&parent)
            .into_iter()
            .flatten()
            .filter(|child| !reachable.contains(child))
            .copied()
            .collect();
        reachable.extend(&child_ids);
        pending.extend(child_ids);
    }
    let unlinked: BTreeSet<_> = events
        .iter()
        .filter(|event| {
            reachable.contains(&event.ppid)
                && !reachable.contains(&event.pid)
                && event.pid != event.ppid
                && event.kind() != EventKind::Fork
        })
        .map(|event| event.pid)
        .collect();
    if !unlinked.is_empty() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "descendant events lack captured fork edges: {}",
            unlinked
                .iter()
                .map(ToString::to_string)
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }

    let scoped_events: Vec<_> = events
        .iter()
        .filter(|event| reachable.contains(&event.pid))
        .collect();
    let empty_args: Vec<_> = scoped_events
        .iter()
        .filter(|event| event.args.is_empty())
        .map(|event| event.index.to_string())
        .collect();
    if !empty_args.is_empty() {
        return Err(ProcessBpfEvidenceError::new(format!(
            "scoped process events have empty ARGS at CSV rows: {}",
            empty_args.join(", ")
        )));
    }
    let mut successful: BTreeMap<i64, Vec<Event>> = BTreeMap::new();
    let mut failed: BTreeMap<i64, Vec<Event>> = BTreeMap::new();
    let mut exit_groups: BTreeMap<i64, Vec<Event>> = BTreeMap::new();
    for event in scoped_events {
        match event.kind() {
            EventKind::Exec if event.retval == Some(0) => {
                successful.entry(event.pid).or_default().push(event.clone());
            }
            EventKind::Exec => failed.entry(event.pid).or_default().push(event.clone()),
            EventKind::ExitGroup => exit_groups
                .entry(event.pid)
                .or_default()
                .push(event.clone()),
            EventKind::Fork => {}
        }
    }
    for events in successful.values_mut().chain(failed.values_mut()) {
        events.sort_by_key(|event| event.index);
    }

    let mut inventory: BTreeMap<String, usize> = BTreeMap::new();
    for event in successful.values().flatten() {
        let binary = event.binary().ok_or_else(|| {
            ProcessBpfEvidenceError::new("a successful exec has no parseable binary path")
        })?;
        *inventory.entry(binary.to_owned()).or_default() += 1;
    }
    let tree = canonical_tree(root_pid, &children, &successful, &HashSet::new())?;
    let tree_hash = format!("{:x}", Sha256::digest(canonical_json(&tree).as_bytes()));

    let image = pipeline
        .get("image_identity")
        .and_then(Value::as_object)
        .ok_or_else(|| ProcessBpfEvidenceError::new("pipeline report has no image identity"))?;
    let repo_digests = image
        .get("repo_digests")
        .and_then(Value::as_array)
        .filter(|values| values.len() == 1)
        .ok_or_else(|| {
            ProcessBpfEvidenceError::new("representative image must have one pinned repo digest")
        })?;
    let mut process_rows = Vec::new();
    for pid in &reachable {
        let creation = fork_by_child.get(pid);
        process_rows.push(json!({
            "task_id": pid,
            "parent_task_id": creation.map(|event| event.ppid),
            "creation_observed": creation.is_some(),
            "successful_execs": successful
                .get(pid)
                .into_iter()
                .flatten()
                .map(|event| json!({
                    "binary": event.binary(),
                    "flat_args": event.args,
                    "callback_epoch_ns": event.timestamp_ns,
                    "cpu": event.cpu,
                }))
                .collect::<Vec<_>>(),
            "failed_exec_attempts": failed
                .get(pid)
                .into_iter()
                .flatten()
                .map(|event| json!({
                    "binary": event.binary(),
                    "flat_args": event.args,
                    "retval": event.retval,
                    "callback_epoch_ns": event.timestamp_ns,
                    "cpu": event.cpu,
                }))
                .collect::<Vec<_>>(),
            "exit_group_observed": exit_groups.contains_key(pid),
        }));
    }

    let cpuset_text = nested(&pipeline, &["execution_plan", "extensions", "cpuset_cpus"])
        .cloned()
        .unwrap();
    let successful_count: usize = successful.values().map(Vec::len).sum();
    let failed_count: usize = failed.values().map(Vec::len).sum();
    let without_exit = reachable
        .iter()
        .filter(|pid| !exit_groups.contains_key(pid))
        .count();
    let mut artifacts = Map::new();
    artifacts.insert("raw_csv".to_owned(), artifact(raw_csv_path)?);
    artifacts.insert("tracer_stderr".to_owned(), artifact(tracer_stderr_path)?);
    artifacts.insert(
        "pipeline_report".to_owned(),
        artifact(pipeline_report_path)?,
    );
    if let Some(path) = collector_path {
        artifacts.insert("collector".to_owned(), artifact(path)?);
    }
    let mut report = json!({
        "schema_version": PROCESS_BPF_SCHEMA_VERSION,
        "method": "external-bcc-execve-sched_process_fork-exit_group",
        "valid": true,
        "invalid_reasons": [],
        "source": {
            "trajectory_sha256": source_sha,
            "image_id": image.get("id").cloned().unwrap_or(Value::Null),
            "image_repo_digest": repo_digests[0],
            "network": pipeline.get("network").cloned().unwrap_or(Value::Null),
            "network_policy": pipeline.get("network_policy").cloned().unwrap_or(Value::Null),
            "pipeline_status": pipeline.get("status").cloned().unwrap_or(Value::Null),
        },
        "scope": {
            "kind": "captured-fork-descendants-of-unique-replay-root",
            "root_task_id": root_pid,
            "root_argv_prefix": root_argv_prefix,
            "cpuset_cpus": cpuset_text,
            "observed_cpu_ids": observed_cpus,
        },
        "loss": {
            "perf_buffer_lost_event_count": loss_count,
            "terminal_marker_observed": true,
        },
        "summary": {
            "task_count": reachable.len(),
            "successful_exec_count": successful_count,
            "failed_exec_attempt_count": failed_count,
            "distinct_binary_count": inventory.len(),
            "exit_group_observed_task_count": exit_groups.len(),
            "task_count_without_exit_group": without_exit,
        },
        "binary_inventory": inventory
            .iter()
            .map(|(binary, count)| json!({"binary": binary, "exec_count": count}))
            .collect::<Vec<_>>(),
        "canonical_tree": tree,
        "canonical_tree_sha256": tree_hash,
        "tasks": process_rows,
        "limitations": {
            "event_timestamp": "EPOCH is userspace callback time, not kernel event time; it is not eligible for canonical timeline alignment",
            "argv": "ARGS is flat and limited by the external collector; argument boundaries and arbitrary long values are not proven",
            "cwd": "not collected",
            "execveat": "not collected by this external collector",
            "exit": "exit_group is observed, but signal-terminated tasks need sched_process_exit for complete lifecycle evidence",
            "claim": "exact successful execve binary inventory and captured task ancestry only",
        },
        "artifacts": artifacts,
    });
    if let Some(path) = expected_manifest_path {
        let expected = read_object(path, "expected manifest")?;
        let reasons = validate_expected(&report, &expected);
        let manifest = artifact(path)?;
        report["expected_validation"] = json!({
            "valid": reasons.is_empty(),
            "reasons": reasons,
            "manifest": manifest,
        });
        report["artifacts"]["expected_manifest"] = artifact(path)?;
        if !reasons.is_empty() {
            report["valid"] = Value::Bool(false);
            report["invalid_reasons"] = json!(reasons);
        }
    }
    Ok(report)
}

fn proof_comparison<F>(proofs: &[Value], mut comparison: F) -> bool
where
    F: FnMut(&Value, &Value) -> bool,
{
    let reference = &proofs[0];
    proofs[1..].iter().all(|proof| comparison(proof, reference))
}

/// Compare independent proofs without requiring PID or callback-time equality.
pub fn build_process_repeatability_report(
    proof_paths: &[PathBuf],
) -> Result<Value, ProcessBpfEvidenceError> {
    if proof_paths.len() < 2 {
        return Err(ProcessBpfEvidenceError::new(
            "repeatability requires at least two process proofs",
        ));
    }
    let proofs: Vec<_> = proof_paths
        .iter()
        .map(|path| read_object(path, "process proof"))
        .collect::<Result<_, _>>()?;
    let mut reasons = Vec::new();
    for (offset, proof) in proofs.iter().enumerate() {
        let run = offset + 1;
        if proof.get("schema_version")
            != Some(&Value::String(PROCESS_BPF_SCHEMA_VERSION.to_owned()))
        {
            reasons.push(format!("run {run} has an unsupported process proof schema"));
        }
        if proof.get("valid") != Some(&Value::Bool(true)) {
            reasons.push(format!("run {run} process proof is invalid"));
        }
        let expected = proof.get("expected_validation").and_then(Value::as_object);
        let expected_manifest = expected.and_then(|value| value.get("manifest"));
        if expected.and_then(|value| value.get("valid")) != Some(&Value::Bool(true))
            || expected_manifest
                .and_then(Value::as_object)
                .and_then(|value| value.get("sha256"))
                .and_then(Value::as_str)
                .is_none()
            || expected_manifest
                .and_then(Value::as_object)
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .is_none()
        {
            reasons.push(format!("run {run} did not pass its pinned expectation"));
        } else if let Some(error) =
            bound_artifact_error(expected_manifest, &format!("run {run} expected manifest"))
        {
            reasons.push(error);
        }
        let collector = nested(proof, &["artifacts", "collector"]);
        if collector
            .and_then(Value::as_object)
            .and_then(|value| value.get("sha256"))
            .and_then(Value::as_str)
            .is_none()
            || collector
                .and_then(Value::as_object)
                .and_then(|value| value.get("path"))
                .and_then(Value::as_str)
                .is_none()
        {
            reasons.push(format!("run {run} has no content-identified collector"));
        } else if let Some(error) = bound_artifact_error(collector, &format!("run {run} collector"))
        {
            reasons.push(error);
        }
        if nested(proof, &["artifacts", "expected_manifest"]) != expected_manifest {
            reasons.push(format!(
                "run {run} expected-manifest identities are internally inconsistent"
            ));
        }
        if nested(proof, &["loss", "perf_buffer_lost_event_count"]).and_then(Value::as_i64)
            != Some(0)
            || nested(proof, &["loss", "terminal_marker_observed"]) != Some(&Value::Bool(true))
        {
            reasons.push(format!("run {run} has incomplete or lossy acquisition"));
        }
    }

    let mut comparisons = Map::new();
    comparisons.insert(
        "source_identity_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            left.get("source") == right.get("source")
        })),
    );
    comparisons.insert(
        "scope_configuration_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            ["kind", "root_argv_prefix", "cpuset_cpus"]
                .iter()
                .all(|key| nested(left, &["scope", key]) == nested(right, &["scope", key]))
        })),
    );
    comparisons.insert(
        "summary_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            left.get("summary") == right.get("summary")
        })),
    );
    comparisons.insert(
        "binary_inventory_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            left.get("binary_inventory") == right.get("binary_inventory")
        })),
    );
    comparisons.insert(
        "canonical_tree_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            left.get("canonical_tree_sha256") == right.get("canonical_tree_sha256")
        })),
    );
    comparisons.insert(
        "collector_sha256_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            nested(left, &["artifacts", "collector", "sha256"])
                == nested(right, &["artifacts", "collector", "sha256"])
        })),
    );
    comparisons.insert(
        "expected_manifest_sha256_equal".to_owned(),
        Value::Bool(proof_comparison(&proofs, |left, right| {
            nested(left, &["expected_validation", "manifest", "sha256"])
                == nested(right, &["expected_validation", "manifest", "sha256"])
        })),
    );
    for (name, value) in &comparisons {
        if value != &Value::Bool(true) {
            reasons.push(format!("repeatability comparison failed: {name}"));
        }
    }
    let comparison = |name: &str| comparisons.get(name) == Some(&Value::Bool(true));
    let shared_summary = if reasons.is_empty() {
        proofs[0].get("summary").cloned().unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let shared_inventory = if comparison("binary_inventory_equal") {
        proofs[0]
            .get("binary_inventory")
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let shared_tree = if comparison("canonical_tree_equal") {
        proofs[0]
            .get("canonical_tree_sha256")
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let shared_collector = if comparison("collector_sha256_equal") {
        nested(&proofs[0], &["artifacts", "collector"])
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let shared_expected = if comparison("expected_manifest_sha256_equal") {
        nested(&proofs[0], &["expected_validation", "manifest"])
            .cloned()
            .unwrap_or(Value::Null)
    } else {
        Value::Null
    };
    let proof_artifacts = proof_paths
        .iter()
        .map(|path| artifact(path))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(json!({
        "schema_version": REPEATABILITY_SCHEMA_VERSION,
        "valid": reasons.is_empty(),
        "reasons": reasons,
        "run_count": proofs.len(),
        "comparisons": comparisons,
        "shared_summary": shared_summary,
        "shared_binary_inventory": shared_inventory,
        "shared_canonical_tree_sha256": shared_tree,
        "shared_collector": shared_collector,
        "shared_expected_manifest": shared_expected,
        "proof_artifacts": proof_artifacts,
    }))
}

pub fn write_process_inventory_report(
    report: &Value,
    path: &Path,
) -> Result<(), ProcessBpfEvidenceError> {
    if let Some(parent) = path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            ProcessBpfEvidenceError::new(format!(
                "unable to create output directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let mut output = String::new();
    render_json(report, Some(2), 0, &mut output);
    output.push('\n');
    fs::write(path, output).map_err(|error| {
        ProcessBpfEvidenceError::new(format!(
            "unable to write report {}: {error}",
            path.display()
        ))
    })
}

fn root_prefix(path: &Path) -> Result<String, ProcessBpfEvidenceError> {
    let value = read_object(path, "expected manifest")?;
    if value.get("schema_version") != Some(&Value::String(EXPECTED_SCHEMA_VERSION.to_owned())) {
        return Err(ProcessBpfEvidenceError::new(
            "expected manifest schema_version is unsupported",
        ));
    }
    value
        .get("root_argv_prefix")
        .and_then(Value::as_str)
        .filter(|prefix| prefix.starts_with('/'))
        .map(str::to_owned)
        .ok_or_else(|| {
            ProcessBpfEvidenceError::new(
                "expected manifest root_argv_prefix must begin with an absolute path",
            )
        })
}

fn parse_cli(arguments: &[OsString]) -> Result<(Vec<PathBuf>, PathBuf), String> {
    let mut positional = Vec::new();
    let mut output = None;
    let mut options = true;
    let mut index = 0;
    while index < arguments.len() {
        if options && arguments[index] == "--" {
            options = false;
        } else if options
            && arguments[index].to_str().is_some_and(|argument| {
                argument.starts_with("--")
                    && argument.len() >= 3
                    && "--output".starts_with(argument)
            })
        {
            index += 1;
            let value = arguments
                .get(index)
                .ok_or_else(|| "argument --output: expected one argument".to_owned())?;
            output = Some(PathBuf::from(value));
        } else if let Some(value) = options
            .then(|| arguments[index].to_str())
            .flatten()
            .and_then(|argument| argument.split_once('='))
            .filter(|(name, _)| {
                name.len() >= 3 && name.starts_with("--") && "--output".starts_with(name)
            })
            .map(|(_, value)| value)
        {
            if value.is_empty() {
                return Err("argument --output: expected one argument".to_owned());
            }
            output = Some(PathBuf::from(value));
        } else if options
            && arguments[index]
                .to_str()
                .is_some_and(|argument| argument.starts_with('-'))
        {
            return Err(format!(
                "unrecognized arguments: {}",
                arguments[index].to_string_lossy()
            ));
        } else {
            positional.push(PathBuf::from(&arguments[index]));
        }
        index += 1;
    }
    let output =
        output.ok_or_else(|| "the following arguments are required: --output".to_owned())?;
    Ok((positional, output))
}

fn help_requested(arguments: &[OsString]) -> bool {
    arguments
        .iter()
        .take_while(|argument| argument.as_os_str() != "--")
        .any(|argument| matches!(argument.to_str(), Some("-h" | "--help")))
}

fn inventory_help(program: &str) {
    println!(
        "usage: {program} RAW_CSV TRACER_STDERR PIPELINE_REPORT EXPECTED_MANIFEST COLLECTOR --output OUTPUT"
    );
    println!();
    println!("Build a fail-closed process-tree and binary-inventory proof from one loss-audited external BPF capture.");
    println!();
    println!("positional arguments:");
    println!("  raw_csv");
    println!("  tracer_stderr");
    println!("  pipeline_report");
    println!("  expected_manifest");
    println!("  collector");
    println!();
    println!("options:");
    println!("  -h, --help       show this help message and exit");
    println!("  --output OUTPUT  Output proof JSON.");
}

fn repeatability_help(program: &str) {
    println!("usage: {program} PROOF [PROOF ...] --output OUTPUT");
    println!();
    println!("Compare two or more content-bound BPF process inventory proofs.");
    println!();
    println!("positional arguments:");
    println!("  proof");
    println!();
    println!("options:");
    println!("  -h, --help       show this help message and exit");
    println!("  --output OUTPUT  Output repeatability JSON.");
}

pub fn inventory_main(arguments: &[OsString], public_program: &str) -> i32 {
    if help_requested(arguments) {
        inventory_help(public_program);
        return 0;
    }
    let result = (|| {
        let (positional, output) = parse_cli(arguments).map_err(ProcessBpfEvidenceError::new)?;
        if positional.len() != 5 {
            return Err(ProcessBpfEvidenceError::new(format!(
                "expected 5 positional arguments, observed {}",
                positional.len()
            )));
        }
        let prefix = root_prefix(&positional[3])?;
        let report = build_process_inventory_report(
            &positional[0],
            &positional[1],
            &positional[2],
            &prefix,
            Some(&positional[3]),
            Some(&positional[4]),
        )?;
        write_process_inventory_report(&report, &output)?;
        Ok((report, output))
    })();
    match result {
        Ok((report, output)) => {
            println!(
                "Process inventory valid: {}",
                report["valid"].as_bool().unwrap()
            );
            println!("Captured tasks: {}", report["summary"]["task_count"]);
            println!(
                "Successful execs: {}",
                report["summary"]["successful_exec_count"]
            );
            println!(
                "Distinct binaries: {}",
                report["summary"]["distinct_binary_count"]
            );
            println!("Report: {}", output.display());
            if report["valid"] == Value::Bool(true) {
                0
            } else {
                1
            }
        }
        Err(error) => {
            eprintln!("error: {error}");
            2
        }
    }
}

pub fn repeatability_main(arguments: &[OsString], public_program: &str) -> i32 {
    if help_requested(arguments) {
        repeatability_help(public_program);
        return 0;
    }
    let result = (|| {
        let (proofs, output) = parse_cli(arguments).map_err(ProcessBpfEvidenceError::new)?;
        if proofs.is_empty() {
            return Err(ProcessBpfEvidenceError::new(
                "the following arguments are required: proof",
            ));
        }
        let report = build_process_repeatability_report(&proofs)?;
        write_process_inventory_report(&report, &output)?;
        Ok((report, output))
    })();
    match result {
        Ok((report, output)) => {
            println!(
                "Process repeatability valid: {}",
                report["valid"].as_bool().unwrap()
            );
            println!("Compared runs: {}", report["run_count"]);
            println!("Report: {}", output.display());
            if report["valid"] == Value::Bool(true) {
                0
            } else {
                1
            }
        }
        Err(error) => {
            eprintln!("error: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn exact_decimal_conversion_rejects_sub_nanoseconds() {
        assert_eq!(decimal_epoch_ns("100.000001", 2).unwrap(), 100_000_001_000);
        assert_eq!(decimal_epoch_ns("1e-9", 2).unwrap(), 1);
        assert_eq!(decimal_epoch_ns("0e2147483647", 2).unwrap(), 0);
        assert!(decimal_epoch_ns("1.0000000001", 2)
            .unwrap_err()
            .to_string()
            .contains("sub-nanosecond"));
    }

    #[test]
    fn python_json_renderer_sorts_and_ascii_escapes() {
        let value = json!({"z": "caf\u{e9}", "a": [1, true]});
        assert_eq!(canonical_json(&value), r#"{"a":[1,true],"z":"caf\u00e9"}"#);
    }

    #[test]
    fn repeatability_requires_two_proofs() {
        let error = build_process_repeatability_report(&[]).unwrap_err();
        assert!(error.to_string().contains("at least two"));
    }

    #[test]
    fn oversized_lost_event_count_fails_without_panicking() {
        let root = TempDir::new().unwrap();
        let diagnostics = root.path().join("tracer.stderr");
        fs::write(&diagnostics, "AGENTSCRY_LOST_EVENTS=18446744073709551616\n").unwrap();
        assert_eq!(
            loss_count(&diagnostics).unwrap_err().to_string(),
            "process trace dropped 18446744073709551616 event(s)"
        );
    }

    #[test]
    fn leading_blank_csv_record_is_not_skipped_as_a_header() {
        let root = TempDir::new().unwrap();
        let trace = root.path().join("events.csv");
        fs::write(
            &trace,
            "\nEPOCH,COMM,PPID,PID,CPU,RET,ARGS\n1,x,0,1,0,0,/bin/x\n",
        )
        .unwrap();
        assert!(read_events(&trace)
            .unwrap_err()
            .to_string()
            .contains("missing columns"));
    }

    #[test]
    fn writer_matches_sorted_pretty_shape() {
        let root = TempDir::new().unwrap();
        let output = root.path().join("proof.json");
        write_process_inventory_report(&json!({"z": 2, "a": 1}), &output).unwrap();
        assert_eq!(
            fs::read_to_string(output).unwrap(),
            "{\n  \"a\": 1,\n  \"z\": 2\n}\n"
        );
    }

    #[test]
    fn cli_accepts_argparse_abbreviation_repetition_and_option_terminator() {
        let (positional, output) = parse_cli(&[
            OsString::from("proof-one"),
            OsString::from("--out=first.json"),
            OsString::from("--output"),
            OsString::from("second.json"),
            OsString::from("--"),
            OsString::from("-proof-two"),
        ])
        .unwrap();
        assert_eq!(output, PathBuf::from("second.json"));
        assert_eq!(
            positional,
            [PathBuf::from("proof-one"), PathBuf::from("-proof-two")]
        );
        assert!(help_requested(&[
            OsString::from("proof"),
            OsString::from("--help")
        ]));
    }
}
