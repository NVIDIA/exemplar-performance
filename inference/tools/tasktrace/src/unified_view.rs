// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native renderer for the public `report unified-view` leaf.
//!
//! The renderer is deliberately downstream-only. It validates and normalizes a
//! retained typed evidence bundle, then derives every export from one canonical
//! view model without executing a workload.

use serde_json::{Map, Number, Value};
use sha2::{Digest, Sha256};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::OsString;
use std::fmt;
use std::fmt::Write as _;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};

pub const VIEW_MODEL_SCHEMA_VERSION: &str = "tasktrace-unified-profile-view-model.v1";
const EVIDENCE_BUNDLE_SCHEMA_VERSION: &str = "tasktrace-profile-evidence-bundle.v1";
const PLOT_WIDTH: f64 = 1000.0;
const PLOT_HEIGHT: f64 = 112.0;
const PLOT_LEFT: f64 = 24.0;
const PLOT_RIGHT: f64 = 12.0;
const PLOT_TOP: f64 = 14.0;
const PLOT_BOTTOM: f64 = 20.0;

const STATUSES: [&str; 5] = ["valid", "invalid", "missing", "dropped", "unalignable"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnifiedViewError(String);

impl UnifiedViewError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for UnifiedViewError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for UnifiedViewError {}

fn shape_registration(shape: &str) -> Option<(&'static str, &'static str)> {
    Some(match shape {
        "scalar" => ("summary", "summary_fact"),
        "point_event" => ("timeline", "point_ticks"),
        "interval" => ("timeline", "interval_bands"),
        "sampled_gauge" => ("timeline", "sample_series"),
        "cumulative_counter" => ("timeline", "counter_series"),
        "window_aggregate" => ("timeline", "window_bands"),
        _ => return None,
    })
}

fn status_color(status: &str) -> &'static str {
    match status {
        "valid" => "#15803d",
        "partial" => "#a16207",
        "invalid" => "#b91c1c",
        "missing" => "#6b7280",
        "dropped" => "#c2410c",
        "unalignable" => "#7e22ce",
        _ => "#374151",
    }
}

fn object<'a>(value: &'a Value, field: &str) -> Result<&'a Map<String, Value>, UnifiedViewError> {
    value
        .as_object()
        .ok_or_else(|| UnifiedViewError::new(format!("{field} must be a JSON object")))
}

fn array<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a Vec<Value>, UnifiedViewError> {
    value
        .and_then(Value::as_array)
        .ok_or_else(|| UnifiedViewError::new(format!("{field} must be a list")))
}

fn text<'a>(value: Option<&'a Value>, field: &str) -> Result<&'a str, UnifiedViewError> {
    value
        .and_then(Value::as_str)
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| UnifiedViewError::new(format!("{field} must be a non-empty string")))
}

fn integer(value: Option<&Value>, field: &str) -> Result<i64, UnifiedViewError> {
    value
        .and_then(Value::as_i64)
        .ok_or_else(|| UnifiedViewError::new(format!("{field} must be an integer")))
}

fn nonnegative(value: Option<&Value>, field: &str) -> Result<i64, UnifiedViewError> {
    let value = integer(value, field)?;
    if value < 0 {
        return Err(UnifiedViewError::new(format!("{field} must be at least 0")));
    }
    Ok(value)
}

fn bool_default(
    value: Option<&Value>,
    default: bool,
    field: &str,
) -> Result<bool, UnifiedViewError> {
    match value {
        None => Ok(default),
        Some(Value::Bool(value)) => Ok(*value),
        _ => Err(UnifiedViewError::new(format!("{field} must be boolean"))),
    }
}

fn mapping_default(value: Option<&Value>, field: &str) -> Result<Value, UnifiedViewError> {
    match value {
        None => Ok(Value::Object(Map::new())),
        Some(Value::Object(value)) => Ok(Value::Object(value.clone())),
        _ => Err(UnifiedViewError::new(format!(
            "{field} must be a JSON object"
        ))),
    }
}

fn list_default(value: Option<&Value>, field: &str) -> Result<Vec<Value>, UnifiedViewError> {
    match value {
        None => Ok(Vec::new()),
        Some(Value::Array(value)) => Ok(value.clone()),
        _ => Err(UnifiedViewError::new(format!("{field} must be a list"))),
    }
}

fn sorted_text_list(value: Option<&Value>, field: &str) -> Result<Vec<Value>, UnifiedViewError> {
    let values = list_default(value, field)?;
    let mut result = Vec::with_capacity(values.len());
    let mut seen = BTreeSet::new();
    for value in values {
        let item = text(Some(&value), field)?.to_owned();
        if !seen.insert(item.clone()) {
            return Err(UnifiedViewError::new(format!(
                "{field} must not contain duplicates"
            )));
        }
        result.push(item);
    }
    result.sort();
    Ok(result.into_iter().map(Value::String).collect())
}

fn normalized_reasons(
    status: &str,
    value: Option<&Value>,
    field: &str,
) -> Result<Vec<Value>, UnifiedViewError> {
    if !STATUSES.contains(&status) {
        return Err(UnifiedViewError::new(format!(
            "{field}.status must be one of: dropped, invalid, missing, unalignable, valid"
        )));
    }
    let reasons = sorted_text_list(value, &format!("{field}.reasons"))?;
    if status == "valid" && !reasons.is_empty() {
        return Err(UnifiedViewError::new(format!(
            "{field} valid evidence cannot have failure reasons"
        )));
    }
    if status != "valid" && reasons.is_empty() {
        return Err(UnifiedViewError::new(format!(
            "{field} {status} evidence requires a reason"
        )));
    }
    Ok(reasons)
}

fn only_keys(
    value: &Map<String, Value>,
    allowed: &[&str],
    required: &[&str],
    field: &str,
) -> Result<(), UnifiedViewError> {
    let mut missing = required
        .iter()
        .filter(|key| !value.contains_key(**key))
        .copied()
        .collect::<Vec<_>>();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(UnifiedViewError::new(format!(
            "{field} is missing: {}",
            missing.join(", ")
        )));
    }
    let allowed = allowed.iter().copied().collect::<BTreeSet<_>>();
    let mut unknown = value
        .keys()
        .filter(|key| !allowed.contains(key.as_str()))
        .map(String::as_str)
        .collect::<Vec<_>>();
    unknown.sort_unstable();
    if !unknown.is_empty() {
        return Err(UnifiedViewError::new(format!(
            "{field} has unknown fields: {}",
            unknown.join(", ")
        )));
    }
    Ok(())
}

fn valid_uuid(value: &str) -> bool {
    let value = value.strip_prefix("urn:uuid:").unwrap_or(value);
    let value = value
        .strip_prefix('{')
        .and_then(|value| value.strip_suffix('}'))
        .unwrap_or(value);
    if value.contains('-') {
        if value.len() != 36 {
            return false;
        }
        for (index, byte) in value.bytes().enumerate() {
            if matches!(index, 8 | 13 | 18 | 23) {
                if byte != b'-' {
                    return false;
                }
            } else if !byte.is_ascii_hexdigit() {
                return false;
            }
        }
        true
    } else {
        value.len() == 32 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
    }
}

fn map(entries: impl IntoIterator<Item = (&'static str, Value)>) -> Value {
    Value::Object(
        entries
            .into_iter()
            .map(|(key, value)| (key.to_owned(), value))
            .collect(),
    )
}

fn clone_or_null(value: Option<&Value>) -> Value {
    value.cloned().unwrap_or(Value::Null)
}

fn number_f64(value: f64) -> Value {
    Value::Number(Number::from_f64(value).expect("finite number"))
}

fn python_float_repr(value: f64) -> String {
    let rendered = format!("{value:?}");
    let Some((mantissa, exponent)) = rendered.split_once('e') else {
        return rendered;
    };
    let exponent = exponent.parse::<i32>().expect("Rust float exponent");
    format!("{mantissa}e{exponent:+03}")
}

fn render_python_json(value: &Value, indent: Option<usize>, depth: usize, output: &mut String) {
    match value {
        Value::Null => output.push_str("null"),
        Value::Bool(value) => output.push_str(if *value { "true" } else { "false" }),
        Value::Number(value) if value.is_i64() || value.is_u64() => {
            output.push_str(&value.to_string());
        }
        Value::Number(value) => {
            output.push_str(&python_float_repr(
                value.as_f64().expect("finite JSON number"),
            ));
        }
        Value::String(value) => {
            output.push_str(&serde_json::to_string(value).expect("JSON string"));
        }
        Value::Array(values) => {
            if values.is_empty() {
                output.push_str("[]");
                return;
            }
            output.push('[');
            for (index, value) in values.iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                if let Some(width) = indent {
                    output.push('\n');
                    output.push_str(&" ".repeat((depth + 1) * width));
                }
                render_python_json(value, indent, depth + 1, output);
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
            let mut values = values.iter().collect::<Vec<_>>();
            values.sort_by(|left, right| left.0.cmp(right.0));
            for (index, (key, value)) in values.into_iter().enumerate() {
                if index > 0 {
                    output.push(',');
                }
                if let Some(width) = indent {
                    output.push('\n');
                    output.push_str(&" ".repeat((depth + 1) * width));
                }
                output.push_str(&serde_json::to_string(key).expect("JSON object key"));
                output.push_str(if indent.is_some() { ": " } else { ":" });
                render_python_json(value, indent, depth + 1, output);
            }
            if let Some(width) = indent {
                output.push('\n');
                output.push_str(&" ".repeat(depth * width));
            }
            output.push('}');
        }
    }
}

fn python_json(value: &Value, indent: Option<usize>) -> String {
    let mut output = String::new();
    render_python_json(value, indent, 0, &mut output);
    output
}

pub fn view_model_bytes(view_model: &Value) -> Result<Vec<u8>, UnifiedViewError> {
    if view_model.get("schema_version").and_then(Value::as_str) != Some(VIEW_MODEL_SCHEMA_VERSION) {
        return Err(UnifiedViewError::new(
            "unsupported or missing unified view-model schema",
        ));
    }
    let mut bytes = python_json(view_model, None).into_bytes();
    bytes.push(b'\n');
    Ok(bytes)
}

pub fn view_model_sha256(view_model: &Value) -> Result<String, UnifiedViewError> {
    Ok(format!(
        "{:x}",
        Sha256::digest(view_model_bytes(view_model)?)
    ))
}

fn artifact_view(artifact: &Map<String, Value>) -> Result<Value, UnifiedViewError> {
    let artifact_id = text(artifact.get("artifact_id"), "artifact.artifact_id")?;
    let relative_path = text(artifact.get("relative_path"), "artifact.relative_path")?;
    let path = Path::new(relative_path);
    if path.is_absolute()
        || relative_path == "."
        || path
            .components()
            .any(|part| matches!(part, Component::ParentDir))
    {
        return Err(UnifiedViewError::new(
            "artifact.relative_path must be a safe relative path",
        ));
    }
    let sha256 = text(artifact.get("sha256"), "artifact.sha256")?;
    if sha256.len() != 64
        || !sha256
            .bytes()
            .all(|character| character.is_ascii_digit() || (b'a'..=b'f').contains(&character))
    {
        return Err(UnifiedViewError::new(
            "artifact.sha256 must be a lowercase SHA-256 digest",
        ));
    }
    let size = nonnegative(artifact.get("size_bytes"), "artifact.size_bytes")?;
    let media = text(artifact.get("media_type"), "artifact.media_type")?;
    Ok(map([
        ("artifact_id", Value::String(artifact_id.to_owned())),
        ("relative_path", Value::String(relative_path.to_owned())),
        ("href", Value::String(percent_quote(relative_path))),
        ("sha256", Value::String(sha256.to_owned())),
        ("size_bytes", Value::from(size)),
        ("media_type", Value::String(media.to_owned())),
        (
            "provenance",
            mapping_default(artifact.get("provenance"), "artifact.provenance")?,
        ),
    ]))
}

fn percent_quote(value: &str) -> String {
    let mut result = String::new();
    const HEX: &[u8; 16] = b"0123456789ABCDEF";
    for byte in value.bytes() {
        if byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.' | b'~' | b'/') {
            result.push(byte as char);
        } else {
            result.push('%');
            result.push(HEX[(byte >> 4) as usize] as char);
            result.push(HEX[(byte & 15) as usize] as char);
        }
    }
    result
}

fn record_sort_key(record: &Map<String, Value>) -> (u8, i64, String) {
    let id = record
        .get("record_id")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .to_owned();
    if let Some(value) = record.get("canonical_timestamp_ns").and_then(Value::as_i64) {
        return (0, value, id);
    }
    if let Some(value) = record
        .get("acquisition_bounds")
        .and_then(Value::as_object)
        .and_then(|bounds| bounds.get("canonical_start_ns"))
        .and_then(Value::as_i64)
    {
        return (0, value, id);
    }
    if let Some(value) = record.get("canonical_start_ns").and_then(Value::as_i64) {
        return (0, value, id);
    }
    if let Some(value) = record.get("native_timestamp_ns").and_then(Value::as_i64) {
        return (1, value, id);
    }
    if let Some(value) = record.get("native_start_ns").and_then(Value::as_i64) {
        return (1, value, id);
    }
    (2, 0, id)
}

fn record_view(
    record: &Map<String, Value>,
    origin_ns: i64,
    artifacts: &BTreeMap<String, Value>,
    expected: (&str, &str, &str, &str),
) -> Result<Value, UnifiedViewError> {
    let record_id = text(record.get("record_id"), "record.record_id")?;
    let run_id = text(record.get("run_id"), "record.run_id")?;
    let scope_id = text(record.get("scope_id"), "record.scope_id")?;
    let profiler_id = text(record.get("profiler_id"), "record.profiler_id")?;
    let stream_id = text(record.get("stream_id"), "record.stream_id")?;
    let shape = text(record.get("shape"), "record.shape")?;
    if (run_id, scope_id, profiler_id, stream_id) != expected {
        return Err(UnifiedViewError::new(
            "record run/scope/stream/profiler identity does not match its bundle",
        ));
    }
    if shape_registration(shape).is_none() {
        return Err(UnifiedViewError::new("record.shape is unsupported"));
    }
    let status = text(record.get("status"), "record.status")?;
    if status == "missing" {
        return Err(UnifiedViewError::new(
            "missing evidence belongs on a stream, not in a fabricated record",
        ));
    }
    let reasons = normalized_reasons(status, record.get("reasons"), "record")?;
    let value = record
        .get("value")
        .cloned()
        .ok_or_else(|| UnifiedViewError::new("record is missing: value"))?;
    if matches!(status, "valid" | "unalignable") && value.is_null() {
        return Err(UnifiedViewError::new(format!(
            "{status} record must contain a value"
        )));
    }
    if status == "dropped" && !value.is_null() {
        return Err(UnifiedViewError::new(
            "dropped record must not fabricate a value",
        ));
    }
    let raw_ids = sorted_text_list(record.get("raw_artifact_ids"), "record.raw_artifact_ids")?;
    let mut raw_artifacts = Vec::new();
    for id in &raw_ids {
        let id = id.as_str().expect("text list");
        raw_artifacts.push(artifacts.get(id).cloned().ok_or_else(|| {
            UnifiedViewError::new(format!("evidence references unknown raw artifacts: {id}"))
        })?);
    }
    let mut result = Map::new();
    result.insert("record_id".into(), Value::String(record_id.to_owned()));
    result.insert("status".into(), Value::String(status.to_owned()));
    result.insert("reasons".into(), Value::Array(reasons));
    result.insert("value".into(), value);
    result.insert(
        "provenance".into(),
        mapping_default(record.get("provenance"), "record.provenance")?,
    );
    result.insert(
        "attributes".into(),
        mapping_default(record.get("attributes"), "record.attributes")?,
    );
    result.insert("raw_artifacts".into(), Value::Array(raw_artifacts));
    for key in [
        "source_clock_id",
        "native_timestamp_ns",
        "native_start_ns",
        "native_end_ns",
        "alignment_error_ns",
    ] {
        if let Some(value) = record.get(key).filter(|value| !value.is_null()) {
            result.insert(key.to_owned(), value.clone());
        }
    }
    if let Some(value) = record
        .get("canonical_timestamp_ns")
        .filter(|value| !value.is_null())
    {
        let timestamp = nonnegative(Some(value), "record.canonical_timestamp_ns")?;
        result.insert(
            "canonical_timestamp_offset_ns".into(),
            Value::from(timestamp - origin_ns),
        );
    }
    if let Some(value) = record
        .get("canonical_start_ns")
        .filter(|value| !value.is_null())
    {
        let start = nonnegative(Some(value), "record.canonical_start_ns")?;
        let end = nonnegative(record.get("canonical_end_ns"), "record.canonical_end_ns")?;
        if end <= start {
            return Err(UnifiedViewError::new(
                "canonical interval end must be greater than start",
            ));
        }
        result.insert(
            "canonical_start_offset_ns".into(),
            Value::from(start - origin_ns),
        );
        result.insert(
            "canonical_end_offset_ns".into(),
            Value::from(end - origin_ns),
        );
    }
    if let Some(bounds) = record
        .get("acquisition_bounds")
        .filter(|value| !value.is_null())
    {
        let bounds = object(bounds, "acquisition_bounds")?;
        let start = nonnegative(
            bounds.get("canonical_start_ns"),
            "acquisition_bounds.canonical_start_ns",
        )?;
        let end = nonnegative(
            bounds.get("canonical_end_ns"),
            "acquisition_bounds.canonical_end_ns",
        )?;
        if end < start {
            return Err(UnifiedViewError::new(
                "acquisition bounds end before they start",
            ));
        }
        result.insert(
            "acquisition_start_offset_ns".into(),
            Value::from(start - origin_ns),
        );
        result.insert(
            "acquisition_end_offset_ns".into(),
            Value::from(end - origin_ns),
        );
    }

    validate_record_timing(record, shape, status)?;
    Ok(Value::Object(result))
}

fn validate_record_timing(
    record: &Map<String, Value>,
    shape: &str,
    status: &str,
) -> Result<(), UnifiedViewError> {
    let present = |key: &str| record.get(key).is_some_and(|value| !value.is_null());
    let timing_keys = [
        "source_clock_id",
        "native_timestamp_ns",
        "native_start_ns",
        "native_end_ns",
        "canonical_timestamp_ns",
        "canonical_start_ns",
        "canonical_end_ns",
        "acquisition_bounds",
        "alignment_error_ns",
    ];
    if shape == "scalar" {
        if timing_keys.iter().any(|key| present(key)) {
            return Err(UnifiedViewError::new("scalar record must remain untimed"));
        }
        return Ok(());
    }
    text(record.get("source_clock_id"), "record.source_clock_id")?;
    for key in [
        "native_timestamp_ns",
        "native_start_ns",
        "native_end_ns",
        "canonical_timestamp_ns",
        "canonical_start_ns",
        "canonical_end_ns",
    ] {
        if let Some(value) = record.get(key).filter(|value| !value.is_null()) {
            nonnegative(Some(value), &format!("record.{key}"))?;
        }
    }
    let point = matches!(
        shape,
        "point_event" | "sampled_gauge" | "cumulative_counter"
    );
    if point {
        if !present("native_timestamp_ns") || present("native_start_ns") || present("native_end_ns")
        {
            return Err(UnifiedViewError::new(format!(
                "{shape} requires one native timestamp"
            )));
        }
        if present("canonical_start_ns") || present("canonical_end_ns") {
            return Err(UnifiedViewError::new(format!(
                "{shape} cannot contain a canonical range"
            )));
        }
    } else if present("native_timestamp_ns")
        || !present("native_start_ns")
        || !present("native_end_ns")
    {
        return Err(UnifiedViewError::new(format!(
            "{shape} requires a native start/end range"
        )));
    } else {
        let native_start = nonnegative(record.get("native_start_ns"), "record.native_start_ns")?;
        let native_end = nonnegative(record.get("native_end_ns"), "record.native_end_ns")?;
        if native_end <= native_start {
            return Err(UnifiedViewError::new(
                "native interval end must be greater than start",
            ));
        }
        if present("canonical_timestamp_ns") || present("acquisition_bounds") {
            return Err(UnifiedViewError::new(format!(
                "{shape} requires a canonical range, not a point/acquisition bound"
            )));
        }
    }
    if status == "unalignable" {
        if [
            "canonical_timestamp_ns",
            "canonical_start_ns",
            "canonical_end_ns",
            "acquisition_bounds",
        ]
        .iter()
        .any(|key| present(key))
        {
            return Err(UnifiedViewError::new(
                "unalignable record must retain native timing without canonical timing",
            ));
        }
        if present("alignment_error_ns") {
            return Err(UnifiedViewError::new(
                "unalignable record cannot claim a canonical alignment error",
            ));
        }
        return Ok(());
    }
    nonnegative(
        record.get("alignment_error_ns"),
        "record.alignment_error_ns",
    )?;
    if point {
        if present("canonical_timestamp_ns") == present("acquisition_bounds") {
            return Err(UnifiedViewError::new(format!(
                "{shape} requires exactly one canonical timestamp or acquisition bounds"
            )));
        }
    } else if !present("canonical_start_ns") || !present("canonical_end_ns") {
        return Err(UnifiedViewError::new(format!(
            "{shape} requires a canonical start/end range"
        )));
    }
    Ok(())
}

fn numeric_domain(records: &[Value]) -> Option<Value> {
    let mut values = records
        .iter()
        .filter_map(|record| record.get("value"))
        .filter(|value| !value.is_boolean())
        .filter_map(Value::as_f64)
        .filter(|value| value.is_finite())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return None;
    }
    values.sort_by(|left, right| left.partial_cmp(right).unwrap_or(Ordering::Equal));
    let mut minimum = values[0];
    let mut maximum = *values.last().expect("nonempty");
    if minimum == maximum {
        let padding = (minimum.abs() * 0.05).max(1.0);
        minimum -= padding;
        maximum += padding;
    }
    Some(map([
        ("minimum", number_f64(minimum)),
        ("maximum", number_f64(maximum)),
    ]))
}

fn stream_view(
    stream: &Map<String, Value>,
    origin_ns: i64,
    run_id: &str,
    scope_id: &str,
    artifacts: &BTreeMap<String, Value>,
    profiler: &Value,
) -> Result<Value, UnifiedViewError> {
    let stream_id = text(stream.get("stream_id"), "stream.stream_id")?;
    let profiler_id = text(stream.get("profiler_id"), "stream.profiler_id")?;
    let metric = text(stream.get("metric"), "stream.metric")?;
    let shape = text(stream.get("shape"), "stream.shape")?;
    let (placement, render_kind) = shape_registration(shape)
        .ok_or_else(|| UnifiedViewError::new("stream.shape is unsupported"))?;
    let unit = text(stream.get("unit"), "stream.unit")?;
    let status = text(stream.get("status"), "stream.status")?;
    let reasons = normalized_reasons(status, stream.get("reasons"), "stream")?;
    let mut raw_records = list_default(stream.get("records"), "stream.records")?;
    raw_records.sort_by(|left, right| {
        let left = left.as_object();
        let right = right.as_object();
        match (left, right) {
            (Some(left), Some(right)) => record_sort_key(left).cmp(&record_sort_key(right)),
            _ => Ordering::Equal,
        }
    });
    let mut record_ids = BTreeSet::new();
    let mut records = Vec::with_capacity(raw_records.len());
    for record in raw_records {
        let record = object(&record, "record")?;
        let record_id = text(record.get("record_id"), "record.record_id")?;
        if !record_ids.insert(record_id.to_owned()) {
            return Err(UnifiedViewError::new(format!(
                "stream {stream_id:?} has duplicate record IDs"
            )));
        }
        if text(record.get("shape"), "record.shape")? != shape {
            return Err(UnifiedViewError::new(
                "record shape does not match its stream",
            ));
        }
        records.push(record_view(
            record,
            origin_ns,
            artifacts,
            (run_id, scope_id, profiler_id, stream_id),
        )?);
    }
    let record_statuses = records
        .iter()
        .filter_map(|record| record.get("status").and_then(Value::as_str))
        .collect::<Vec<_>>();
    match status {
        "missing" if !records.is_empty() => {
            return Err(UnifiedViewError::new(
                "missing stream must not contain fabricated records",
            ));
        }
        "valid" if records.is_empty() || record_statuses.iter().any(|item| *item != "valid") => {
            return Err(UnifiedViewError::new(
                "valid stream requires only valid records",
            ));
        }
        "invalid" if !record_statuses.contains(&"invalid") => {
            return Err(UnifiedViewError::new(
                "invalid stream requires an invalid record",
            ));
        }
        "dropped" if !records.is_empty() && !record_statuses.contains(&"dropped") => {
            return Err(UnifiedViewError::new(
                "dropped stream records must expose a dropped record",
            ));
        }
        "unalignable"
            if records.is_empty() || record_statuses.iter().any(|item| *item != "unalignable") =>
        {
            return Err(UnifiedViewError::new(
                "unalignable stream requires only unalignable records",
            ));
        }
        _ => {}
    }

    let display = mapping_default(stream.get("display"), "stream.display")?;
    let display_object = display.as_object().expect("mapping default");
    let label = display_object
        .get("label")
        .and_then(Value::as_str)
        .filter(|label| !label.is_empty())
        .unwrap_or(metric)
        .to_owned();
    let group = display_object
        .get("group")
        .and_then(Value::as_str)
        .filter(|group| !group.is_empty())
        .unwrap_or(profiler_id)
        .to_owned();
    let order = display_object
        .get("order")
        .and_then(|value| {
            if value.is_boolean() {
                None
            } else {
                value.as_f64()
            }
        })
        .filter(|value| value.is_finite())
        .unwrap_or(0.0);

    let mut raw_gaps = list_default(stream.get("gaps"), "stream.gaps")?;
    raw_gaps.sort_by(|left, right| {
        let key = |value: &Value| {
            let gap = value.as_object();
            (
                gap.and_then(|gap| gap.get("canonical"))
                    .and_then(Value::as_bool)
                    .map(|canonical| !canonical)
                    .unwrap_or(false),
                gap.and_then(|gap| gap.get("start_ns"))
                    .and_then(Value::as_i64)
                    .unwrap_or(0),
                gap.and_then(|gap| gap.get("end_ns"))
                    .and_then(Value::as_i64)
                    .unwrap_or(0),
                gap.and_then(|gap| gap.get("reason"))
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            )
        };
        key(left).cmp(&key(right))
    });
    if shape == "scalar" && !raw_gaps.is_empty() {
        return Err(UnifiedViewError::new(
            "untimed scalar stream cannot contain timeline gaps",
        ));
    }
    let mut gaps = Vec::new();
    for gap in raw_gaps {
        let gap = object(&gap, "gap")?;
        let clock_id = text(gap.get("clock_id"), "gap.clock_id")?;
        let start = nonnegative(gap.get("start_ns"), "gap.start_ns")?;
        let end = nonnegative(gap.get("end_ns"), "gap.end_ns")?;
        if end <= start {
            return Err(UnifiedViewError::new(
                "gap end_ns must be greater than start_ns",
            ));
        }
        let reason = text(gap.get("reason"), "gap.reason")?;
        let canonical = bool_default(gap.get("canonical"), true, "gap.canonical")?;
        let mut gap_view = Map::new();
        gap_view.insert("clock_id".into(), Value::String(clock_id.to_owned()));
        gap_view.insert("start_ns".into(), Value::from(start));
        gap_view.insert("end_ns".into(), Value::from(end));
        gap_view.insert("reason".into(), Value::String(reason.to_owned()));
        gap_view.insert("canonical".into(), Value::Bool(canonical));
        gap_view.insert(
            "dropped_records".into(),
            clone_or_null(gap.get("dropped_records")),
        );
        if canonical {
            gap_view.insert("start_offset_ns".into(), Value::from(start - origin_ns));
            gap_view.insert("end_offset_ns".into(), Value::from(end - origin_ns));
        }
        gaps.push(Value::Object(gap_view));
    }

    let mut status_counts = Map::new();
    for record_status in STATUSES {
        let count = records
            .iter()
            .filter(|record| record.get("status").and_then(Value::as_str) == Some(record_status))
            .count();
        if count > 0 {
            status_counts.insert(record_status.to_owned(), Value::from(count));
        }
    }
    let maximum_alignment_error = records
        .iter()
        .filter_map(|record| record.get("alignment_error_ns"))
        .filter_map(Value::as_i64)
        .max()
        .map(Value::from)
        .unwrap_or(Value::Null);
    let stream_raw_ids =
        sorted_text_list(stream.get("raw_artifact_ids"), "stream.raw_artifact_ids")?;
    let mut artifact_ids = BTreeSet::new();
    for id in stream_raw_ids.iter().filter_map(Value::as_str) {
        artifact_ids.insert(id.to_owned());
    }
    for raw_record in list_default(stream.get("records"), "stream.records")? {
        let record = object(&raw_record, "record")?;
        for id in sorted_text_list(record.get("raw_artifact_ids"), "record.raw_artifact_ids")?
            .iter()
            .filter_map(Value::as_str)
        {
            artifact_ids.insert(id.to_owned());
        }
    }
    let mut raw_artifacts = Vec::new();
    for id in artifact_ids {
        raw_artifacts.push(artifacts.get(&id).cloned().ok_or_else(|| {
            UnifiedViewError::new(format!("evidence references unknown raw artifacts: {id}"))
        })?);
    }
    let cadence = match stream.get("cadence").filter(|value| !value.is_null()) {
        None => Value::Null,
        Some(value) => {
            if !matches!(shape, "sampled_gauge" | "cumulative_counter") {
                return Err(UnifiedViewError::new(
                    "cadence is only valid for sampled gauge/counter streams",
                ));
            }
            let cadence = object(value, "cadence")?;
            let requested = integer(
                cadence.get("requested_interval_ns"),
                "cadence.requested_interval_ns",
            )?;
            let maximum = integer(
                cadence.get("maximum_acceptable_gap_ns"),
                "cadence.maximum_acceptable_gap_ns",
            )?;
            if requested < 1 || maximum < requested {
                return Err(UnifiedViewError::new("cadence intervals are invalid"));
            }
            map([
                ("requested_interval_ns", Value::from(requested)),
                ("maximum_acceptable_gap_ns", Value::from(maximum)),
                (
                    "method",
                    Value::String(text(cadence.get("method"), "cadence.method")?.to_owned()),
                ),
            ])
        }
    };
    let domain = numeric_domain(&records).unwrap_or(Value::Null);
    Ok(map([
        ("stream_id", Value::String(stream_id.to_owned())),
        ("profiler_id", Value::String(profiler_id.to_owned())),
        ("profiler", profiler.clone()),
        ("metric", Value::String(metric.to_owned())),
        ("label", Value::String(label)),
        ("group", Value::String(group)),
        ("order", number_f64(order)),
        ("shape", Value::String(shape.to_owned())),
        ("placement", Value::String(placement.to_owned())),
        ("render_kind", Value::String(render_kind.to_owned())),
        ("unit", Value::String(unit.to_owned())),
        ("status", Value::String(status.to_owned())),
        ("reasons", Value::Array(reasons)),
        ("records", Value::Array(records)),
        ("record_status_counts", Value::Object(status_counts)),
        ("cadence", cadence),
        ("gaps", Value::Array(gaps)),
        ("maximum_alignment_error_ns", maximum_alignment_error),
        ("raw_artifacts", Value::Array(raw_artifacts)),
        (
            "provenance",
            mapping_default(stream.get("provenance"), "stream.provenance")?,
        ),
        ("display", display),
        (
            "extensions",
            mapping_default(stream.get("extensions"), "stream.extensions")?,
        ),
        ("numeric_value_domain", domain),
    ]))
}

fn profiler_view(profiler: &Map<String, Value>) -> Result<Value, UnifiedViewError> {
    let profiler_id = text(profiler.get("profiler_id"), "profiler.profiler_id")?;
    let version = text(
        profiler.get("profiler_version"),
        "profiler.profiler_version",
    )?;
    let status = text(profiler.get("status"), "profiler.status")?;
    let reasons = normalized_reasons(status, profiler.get("reasons"), "profiler")?;
    Ok(map([
        ("profiler_id", Value::String(profiler_id.to_owned())),
        ("profiler_version", Value::String(version.to_owned())),
        ("status", Value::String(status.to_owned())),
        ("reasons", Value::Array(reasons)),
        (
            "stream_ids",
            Value::Array(sorted_text_list(
                profiler.get("stream_ids"),
                "profiler.stream_ids",
            )?),
        ),
        (
            "intrusive",
            Value::Bool(bool_default(
                profiler.get("intrusive"),
                false,
                "profiler.intrusive",
            )?),
        ),
        (
            "provenance",
            mapping_default(profiler.get("provenance"), "profiler.provenance")?,
        ),
        (
            "configuration",
            mapping_default(profiler.get("configuration"), "profiler.configuration")?,
        ),
        (
            "extensions",
            mapping_default(profiler.get("extensions"), "profiler.extensions")?,
        ),
    ]))
}

fn validate_schema_fields(bundle: &Map<String, Value>) -> Result<(), UnifiedViewError> {
    only_keys(
        bundle,
        &[
            "schema_version",
            "run_id",
            "scope",
            "timeline",
            "clocks",
            "clock_mappings",
            "profilers",
            "streams",
            "artifacts",
            "provenance",
            "extensions",
        ],
        &[
            "schema_version",
            "run_id",
            "scope",
            "timeline",
            "clocks",
            "clock_mappings",
            "profilers",
            "streams",
        ],
        "bundle",
    )?;
    let scope = object(bundle.get("scope").expect("required"), "scope")?;
    only_keys(
        scope,
        &[
            "scope_id",
            "kind",
            "identity",
            "includes_descendants",
            "description",
            "extensions",
        ],
        &["scope_id", "kind", "identity"],
        "scope",
    )?;
    if let Some(description) = scope.get("description").filter(|value| !value.is_null()) {
        text(Some(description), "scope.description")?;
    }
    let timeline = object(bundle.get("timeline").expect("required"), "timeline")?;
    only_keys(
        timeline,
        &[
            "clock_id",
            "clock_name",
            "origin_ns",
            "start_ns",
            "end_ns",
            "lifecycle_markers",
        ],
        &["clock_id", "clock_name", "origin_ns", "start_ns", "end_ns"],
        "timeline",
    )?;
    for marker in list_default(
        timeline.get("lifecycle_markers"),
        "timeline.lifecycle_markers",
    )? {
        let marker = object(&marker, "marker")?;
        only_keys(
            marker,
            &[
                "name",
                "canonical_timestamp_ns",
                "alignment_error_ns",
                "provenance",
            ],
            &["name", "canonical_timestamp_ns"],
            "marker",
        )?;
    }
    for clock in array(bundle.get("clocks"), "bundle.clocks")? {
        let clock = object(clock, "clock")?;
        only_keys(
            clock,
            &[
                "clock_id",
                "clock_name",
                "resolution_ns",
                "namespace_identity",
                "provenance",
            ],
            &["clock_id", "clock_name", "resolution_ns"],
            "clock",
        )?;
        if let Some(namespace) = clock
            .get("namespace_identity")
            .filter(|value| !value.is_null())
        {
            text(Some(namespace), "clock.namespace_identity")?;
        }
        mapping_default(clock.get("provenance"), "clock.provenance")?;
    }
    for mapping in array(bundle.get("clock_mappings"), "bundle.clock_mappings")? {
        let mapping = object(mapping, "clock_mapping")?;
        only_keys(
            mapping,
            &[
                "source_clock_id",
                "canonical_clock_id",
                "status",
                "method",
                "scale_numerator",
                "scale_denominator",
                "offset_ns",
                "max_error_ns",
                "reasons",
                "provenance",
            ],
            &["source_clock_id", "canonical_clock_id", "status", "method"],
            "clock_mapping",
        )?;
    }
    for artifact in list_default(bundle.get("artifacts"), "bundle.artifacts")? {
        let artifact = object(&artifact, "artifact")?;
        only_keys(
            artifact,
            &[
                "artifact_id",
                "relative_path",
                "sha256",
                "size_bytes",
                "media_type",
                "provenance",
            ],
            &[
                "artifact_id",
                "relative_path",
                "sha256",
                "size_bytes",
                "media_type",
            ],
            "artifact",
        )?;
    }
    for profiler in array(bundle.get("profilers"), "bundle.profilers")? {
        let profiler = object(profiler, "profiler")?;
        only_keys(
            profiler,
            &[
                "profiler_id",
                "profiler_version",
                "status",
                "stream_ids",
                "reasons",
                "intrusive",
                "provenance",
                "configuration",
                "extensions",
            ],
            &["profiler_id", "profiler_version", "status"],
            "profiler",
        )?;
    }
    for stream in array(bundle.get("streams"), "bundle.streams")? {
        let stream = object(stream, "stream")?;
        only_keys(
            stream,
            &[
                "stream_id",
                "profiler_id",
                "metric",
                "shape",
                "unit",
                "status",
                "records",
                "reasons",
                "cadence",
                "gaps",
                "raw_artifact_ids",
                "provenance",
                "display",
                "extensions",
            ],
            &[
                "stream_id",
                "profiler_id",
                "metric",
                "shape",
                "unit",
                "status",
            ],
            "stream",
        )?;
        if let Some(cadence) = stream.get("cadence").filter(|value| !value.is_null()) {
            let cadence = object(cadence, "cadence")?;
            only_keys(
                cadence,
                &[
                    "requested_interval_ns",
                    "maximum_acceptable_gap_ns",
                    "method",
                ],
                &[
                    "requested_interval_ns",
                    "maximum_acceptable_gap_ns",
                    "method",
                ],
                "cadence",
            )?;
        }
        for gap in list_default(stream.get("gaps"), "stream.gaps")? {
            let gap = object(&gap, "gap")?;
            only_keys(
                gap,
                &[
                    "clock_id",
                    "start_ns",
                    "end_ns",
                    "reason",
                    "canonical",
                    "dropped_records",
                ],
                &["clock_id", "start_ns", "end_ns", "reason"],
                "gap",
            )?;
        }
        for record in list_default(stream.get("records"), "stream.records")? {
            let record = object(&record, "record")?;
            only_keys(
                record,
                &[
                    "record_id",
                    "run_id",
                    "scope_id",
                    "profiler_id",
                    "stream_id",
                    "shape",
                    "status",
                    "value",
                    "reasons",
                    "source_clock_id",
                    "native_timestamp_ns",
                    "native_start_ns",
                    "native_end_ns",
                    "canonical_timestamp_ns",
                    "canonical_start_ns",
                    "canonical_end_ns",
                    "acquisition_bounds",
                    "alignment_error_ns",
                    "raw_artifact_ids",
                    "provenance",
                    "attributes",
                ],
                &[
                    "record_id",
                    "run_id",
                    "scope_id",
                    "profiler_id",
                    "stream_id",
                    "shape",
                    "status",
                    "value",
                ],
                "record",
            )?;
            if let Some(bounds) = record
                .get("acquisition_bounds")
                .filter(|value| !value.is_null())
            {
                let bounds = object(bounds, "acquisition_bounds")?;
                only_keys(
                    bounds,
                    &["canonical_start_ns", "canonical_end_ns"],
                    &["canonical_start_ns", "canonical_end_ns"],
                    "acquisition_bounds",
                )?;
            }
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ClockMappingInfo {
    status: String,
    numerator: Option<i128>,
    denominator: Option<i128>,
    offset: Option<i128>,
    max_error: Option<i128>,
}

fn i128_integer(value: Option<&Value>, field: &str) -> Result<i128, UnifiedViewError> {
    let value =
        value.ok_or_else(|| UnifiedViewError::new(format!("{field} must be an integer")))?;
    if let Some(value) = value.as_i64() {
        Ok(value as i128)
    } else if let Some(value) = value.as_u64() {
        Ok(value as i128)
    } else {
        Err(UnifiedViewError::new(format!("{field} must be an integer")))
    }
}

fn validate_bundle_relationships(
    bundle: &Map<String, Value>,
    canonical_id: &str,
    start: i64,
    end: i64,
) -> Result<(), UnifiedViewError> {
    let run_id = text(bundle.get("run_id"), "bundle.run_id")?;
    if !valid_uuid(run_id) {
        return Err(UnifiedViewError::new("bundle.run_id must be a UUID"));
    }
    let scope_id = text(
        bundle.get("scope").and_then(|scope| scope.get("scope_id")),
        "scope.scope_id",
    )?;
    let mut clock_ids = BTreeSet::new();
    for clock in array(bundle.get("clocks"), "bundle.clocks")? {
        let clock = object(clock, "clock")?;
        clock_ids.insert(text(clock.get("clock_id"), "clock.clock_id")?.to_owned());
    }
    let mut mapping_by_source = BTreeMap::new();
    for mapping in array(bundle.get("clock_mappings"), "bundle.clock_mappings")? {
        let mapping = object(mapping, "clock_mapping")?;
        let source = text(
            mapping.get("source_clock_id"),
            "clock_mapping.source_clock_id",
        )?;
        let target = text(
            mapping.get("canonical_clock_id"),
            "clock_mapping.canonical_clock_id",
        )?;
        if !clock_ids.contains(source) {
            return Err(UnifiedViewError::new(
                "clock mapping references an unknown source clock",
            ));
        }
        if target != canonical_id {
            return Err(UnifiedViewError::new(
                "clock mapping targets a non-canonical clock",
            ));
        }
        if source == canonical_id {
            return Err(UnifiedViewError::new(
                "canonical clock must not map to itself",
            ));
        }
        let status = text(mapping.get("status"), "clock_mapping.status")?;
        let transform_values = [
            "scale_numerator",
            "scale_denominator",
            "offset_ns",
            "max_error_ns",
        ];
        let all_present = transform_values
            .iter()
            .all(|key| mapping.get(*key).is_some_and(|value| !value.is_null()));
        let any_present = transform_values
            .iter()
            .any(|key| mapping.get(*key).is_some_and(|value| !value.is_null()));
        let info = if status == "valid" {
            if !all_present {
                return Err(UnifiedViewError::new(
                    "valid clock mapping requires a complete affine transform",
                ));
            }
            let numerator = i128_integer(
                mapping.get("scale_numerator"),
                "clock_mapping.scale_numerator",
            )?;
            let denominator = i128_integer(
                mapping.get("scale_denominator"),
                "clock_mapping.scale_denominator",
            )?;
            let offset = i128_integer(mapping.get("offset_ns"), "clock_mapping.offset_ns")?;
            let max_error =
                i128_integer(mapping.get("max_error_ns"), "clock_mapping.max_error_ns")?;
            if numerator < 1 || denominator < 1 || max_error < 0 {
                return Err(UnifiedViewError::new(
                    "valid clock mapping transform is invalid",
                ));
            }
            ClockMappingInfo {
                status: status.to_owned(),
                numerator: Some(numerator),
                denominator: Some(denominator),
                offset: Some(offset),
                max_error: Some(max_error),
            }
        } else {
            if status != "unalignable" {
                return Err(UnifiedViewError::new(
                    "clock_mapping.status must be valid or unalignable",
                ));
            }
            if any_present {
                return Err(UnifiedViewError::new(
                    "unalignable clock mapping must not contain a fabricated transform",
                ));
            }
            ClockMappingInfo {
                status: status.to_owned(),
                numerator: None,
                denominator: None,
                offset: None,
                max_error: None,
            }
        };
        if mapping_by_source.insert(source.to_owned(), info).is_some() {
            return Err(UnifiedViewError::new(format!(
                "duplicate clock mapping identity: {source}"
            )));
        }
    }
    for clock in &clock_ids {
        if clock != canonical_id && !mapping_by_source.contains_key(clock) {
            return Err(UnifiedViewError::new(format!(
                "source clock {clock:?} has no clock mapping"
            )));
        }
    }

    for stream in array(bundle.get("streams"), "bundle.streams")? {
        let stream = object(stream, "stream")?;
        for gap in list_default(stream.get("gaps"), "stream.gaps")? {
            let gap = object(&gap, "gap")?;
            let clock = text(gap.get("clock_id"), "gap.clock_id")?;
            if !clock_ids.contains(clock) {
                return Err(UnifiedViewError::new("gap references an unknown clock"));
            }
            if bool_default(gap.get("canonical"), true, "gap.canonical")? {
                if clock != canonical_id {
                    return Err(UnifiedViewError::new(
                        "canonical gap must use the canonical clock",
                    ));
                }
                let gap_start = nonnegative(gap.get("start_ns"), "gap.start_ns")?;
                let gap_end = nonnegative(gap.get("end_ns"), "gap.end_ns")?;
                if gap_start < start || gap_end > end {
                    return Err(UnifiedViewError::new(
                        "canonical gap falls outside the run window",
                    ));
                }
            }
            if let Some(dropped) = gap.get("dropped_records").filter(|value| !value.is_null()) {
                if integer(Some(dropped), "gap.dropped_records")? < 1 {
                    return Err(UnifiedViewError::new(
                        "gap.dropped_records must be at least 1",
                    ));
                }
            }
        }
        for record in list_default(stream.get("records"), "stream.records")? {
            let record = object(&record, "record")?;
            if text(record.get("run_id"), "record.run_id")? != run_id {
                return Err(UnifiedViewError::new("record belongs to a different run"));
            }
            if text(record.get("scope_id"), "record.scope_id")? != scope_id {
                return Err(UnifiedViewError::new(
                    "record belongs to a different measured scope",
                ));
            }
            validate_record_relationships(
                record,
                canonical_id,
                start,
                end,
                &clock_ids,
                &mapping_by_source,
            )?;
        }
    }
    Ok(())
}

fn validate_record_relationships(
    record: &Map<String, Value>,
    canonical_id: &str,
    start: i64,
    end: i64,
    clocks: &BTreeSet<String>,
    mappings: &BTreeMap<String, ClockMappingInfo>,
) -> Result<(), UnifiedViewError> {
    let shape = text(record.get("shape"), "record.shape")?;
    let status = text(record.get("status"), "record.status")?;
    let source = record.get("source_clock_id").and_then(Value::as_str);
    if shape != "scalar" {
        let source = source
            .ok_or_else(|| UnifiedViewError::new("temporal record requires source_clock_id"))?;
        if !clocks.contains(source) {
            return Err(UnifiedViewError::new(
                "record references an unknown source clock",
            ));
        }
        let mapping = mappings.get(source);
        if source == canonical_id && status == "unalignable" {
            return Err(UnifiedViewError::new(
                "canonical-clock record cannot be unalignable",
            ));
        }
        if source != canonical_id {
            let mapping = mapping
                .ok_or_else(|| UnifiedViewError::new("record source clock has no mapping"))?;
            if mapping.status == "unalignable" && status != "unalignable" {
                return Err(UnifiedViewError::new(
                    "record on an unalignable clock must be marked unalignable",
                ));
            }
            if mapping.status == "valid" && status == "unalignable" {
                return Err(UnifiedViewError::new(
                    "unalignable record contradicts its valid clock mapping",
                ));
            }
        }
        if status != "unalignable" {
            validate_projection(record, mapping)?;
        }
    }
    let canonical_values = [
        record.get("canonical_timestamp_ns").and_then(Value::as_i64),
        record.get("canonical_start_ns").and_then(Value::as_i64),
        record.get("canonical_end_ns").and_then(Value::as_i64),
        record
            .get("acquisition_bounds")
            .and_then(|bounds| bounds.get("canonical_start_ns"))
            .and_then(Value::as_i64),
        record
            .get("acquisition_bounds")
            .and_then(|bounds| bounds.get("canonical_end_ns"))
            .and_then(Value::as_i64),
    ]
    .into_iter()
    .flatten()
    .collect::<Vec<_>>();
    if canonical_values
        .iter()
        .any(|timestamp| *timestamp < start || *timestamp > end)
    {
        return Err(UnifiedViewError::new(
            "record canonical timing falls outside the run window",
        ));
    }
    Ok(())
}

fn validate_projection(
    record: &Map<String, Value>,
    mapping: Option<&ClockMappingInfo>,
) -> Result<(), UnifiedViewError> {
    let alignment = i128_integer(
        record.get("alignment_error_ns"),
        "record.alignment_error_ns",
    )?;
    if alignment < 0 {
        return Err(UnifiedViewError::new(
            "record.alignment_error_ns must be at least 0",
        ));
    }
    let mapping_error = mapping.and_then(|mapping| mapping.max_error).unwrap_or(0);
    if alignment < mapping_error {
        return Err(UnifiedViewError::new(
            "record alignment error understates its clock mapping error",
        ));
    }
    let project = |native: i128| -> Result<i128, UnifiedViewError> {
        if let Some(mapping) = mapping {
            let numerator = mapping.numerator.expect("valid mapping");
            let denominator = mapping.denominator.expect("valid mapping");
            let scaled = native.checked_mul(numerator).ok_or_else(|| {
                UnifiedViewError::new("clock projection exceeds native integer range")
            })?;
            let mut quotient = scaled / denominator;
            let remainder = scaled % denominator;
            if remainder * 2 >= denominator {
                quotient += 1;
            }
            quotient
                .checked_add(mapping.offset.expect("valid mapping"))
                .ok_or_else(|| {
                    UnifiedViewError::new("clock projection exceeds native integer range")
                })
        } else {
            Ok(native)
        }
    };
    if let Some(native) = record
        .get("native_timestamp_ns")
        .filter(|value| !value.is_null())
    {
        let expected = project(i128_integer(Some(native), "record.native_timestamp_ns")?)?;
        if let Some(canonical) = record
            .get("canonical_timestamp_ns")
            .filter(|value| !value.is_null())
        {
            let canonical = i128_integer(Some(canonical), "record.canonical_timestamp_ns")?;
            if (canonical - expected).abs() > alignment {
                return Err(UnifiedViewError::new(
                    "record canonical timestamp contradicts its clock mapping",
                ));
            }
        } else {
            let lower = i128_integer(
                record
                    .get("acquisition_bounds")
                    .and_then(|bounds| bounds.get("canonical_start_ns")),
                "acquisition_bounds.canonical_start_ns",
            )?;
            let upper = i128_integer(
                record
                    .get("acquisition_bounds")
                    .and_then(|bounds| bounds.get("canonical_end_ns")),
                "acquisition_bounds.canonical_end_ns",
            )?;
            if expected + alignment < lower || expected - alignment > upper {
                return Err(UnifiedViewError::new(
                    "record acquisition bounds contradict its clock mapping",
                ));
            }
        }
    } else {
        let native_start = i128_integer(record.get("native_start_ns"), "record.native_start_ns")?;
        let native_end = i128_integer(record.get("native_end_ns"), "record.native_end_ns")?;
        if native_end <= native_start {
            return Err(UnifiedViewError::new(
                "native interval end must be greater than start",
            ));
        }
        let canonical_start = i128_integer(
            record.get("canonical_start_ns"),
            "record.canonical_start_ns",
        )?;
        let canonical_end =
            i128_integer(record.get("canonical_end_ns"), "record.canonical_end_ns")?;
        if (canonical_start - project(native_start)?).abs() > alignment
            || (canonical_end - project(native_end)?).abs() > alignment
        {
            return Err(UnifiedViewError::new(
                "record canonical range contradicts its clock mapping",
            ));
        }
    }
    Ok(())
}

pub fn assemble_view_model(bundle: &Value) -> Result<Value, UnifiedViewError> {
    let bundle = object(bundle, "bundle")?;
    validate_schema_fields(bundle)?;
    if text(bundle.get("schema_version"), "bundle.schema_version")?
        != EVIDENCE_BUNDLE_SCHEMA_VERSION
    {
        return Err(UnifiedViewError::new("unsupported evidence bundle schema"));
    }
    let run_id = text(bundle.get("run_id"), "bundle.run_id")?;
    let scope = object(
        bundle
            .get("scope")
            .ok_or_else(|| UnifiedViewError::new("bundle is missing: scope"))?,
        "scope",
    )?;
    let scope_id = text(scope.get("scope_id"), "scope.scope_id")?;
    text(scope.get("kind"), "scope.kind")?;
    let identity = mapping_default(scope.get("identity"), "scope.identity")?;
    if identity.as_object().is_some_and(Map::is_empty) {
        return Err(UnifiedViewError::new("scope.identity must not be empty"));
    }
    let normalized_scope = map([
        ("scope_id", Value::String(scope_id.to_owned())),
        (
            "kind",
            Value::String(text(scope.get("kind"), "scope.kind")?.to_owned()),
        ),
        ("identity", identity),
        (
            "includes_descendants",
            Value::Bool(bool_default(
                scope.get("includes_descendants"),
                true,
                "scope.includes_descendants",
            )?),
        ),
        ("description", clone_or_null(scope.get("description"))),
        (
            "extensions",
            mapping_default(scope.get("extensions"), "scope.extensions")?,
        ),
    ]);
    let timeline = object(
        bundle
            .get("timeline")
            .ok_or_else(|| UnifiedViewError::new("bundle is missing: timeline"))?,
        "timeline",
    )?;
    let clock_id = text(timeline.get("clock_id"), "timeline.clock_id")?;
    if text(timeline.get("clock_name"), "timeline.clock_name")? != "CLOCK_MONOTONIC" {
        return Err(UnifiedViewError::new(
            "timeline.clock_name must be CLOCK_MONOTONIC",
        ));
    }
    let origin = nonnegative(timeline.get("origin_ns"), "timeline.origin_ns")?;
    let start = nonnegative(timeline.get("start_ns"), "timeline.start_ns")?;
    let end = nonnegative(timeline.get("end_ns"), "timeline.end_ns")?;
    if end <= start {
        return Err(UnifiedViewError::new(
            "timeline end_ns must be greater than start_ns",
        ));
    }
    if origin < start || origin > end {
        return Err(UnifiedViewError::new(
            "timeline origin_ns must fall within the run window",
        ));
    }
    validate_bundle_relationships(bundle, clock_id, start, end)?;

    let mut artifact_entries = Vec::new();
    for artifact in list_default(bundle.get("artifacts"), "bundle.artifacts")? {
        let artifact = object(&artifact, "artifact")?;
        artifact_entries.push((
            text(artifact.get("artifact_id"), "artifact.artifact_id")?.to_owned(),
            artifact_view(artifact)?,
        ));
    }
    artifact_entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut artifact_by_id = BTreeMap::new();
    for (id, artifact) in artifact_entries {
        if artifact_by_id.insert(id.clone(), artifact).is_some() {
            return Err(UnifiedViewError::new(format!(
                "duplicate artifact identity: {id}"
            )));
        }
    }

    let mut profiler_entries = Vec::new();
    for profiler in array(bundle.get("profilers"), "bundle.profilers")? {
        let profiler = object(profiler, "profiler")?;
        let id = text(profiler.get("profiler_id"), "profiler.profiler_id")?.to_owned();
        profiler_entries.push((id, profiler_view(profiler)?));
    }
    profiler_entries.sort_by(|left, right| left.0.cmp(&right.0));
    let mut profiler_by_id = BTreeMap::new();
    for (id, profiler) in profiler_entries {
        if profiler_by_id.insert(id.clone(), profiler).is_some() {
            return Err(UnifiedViewError::new(format!(
                "duplicate profiler identity: {id}"
            )));
        }
    }

    let mut raw_streams = array(bundle.get("streams"), "bundle.streams")?.clone();
    raw_streams.sort_by(|left, right| {
        left.get("stream_id")
            .and_then(Value::as_str)
            .cmp(&right.get("stream_id").and_then(Value::as_str))
    });
    let mut seen_streams = BTreeSet::new();
    let mut streams = Vec::new();
    let mut actual_by_profiler: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for raw_stream in raw_streams {
        let stream = object(&raw_stream, "stream")?;
        let stream_id = text(stream.get("stream_id"), "stream.stream_id")?.to_owned();
        if !seen_streams.insert(stream_id.clone()) {
            return Err(UnifiedViewError::new(format!(
                "duplicate stream identity: {stream_id}"
            )));
        }
        let profiler_id = text(stream.get("profiler_id"), "stream.profiler_id")?;
        let profiler = profiler_by_id
            .get(profiler_id)
            .ok_or_else(|| UnifiedViewError::new("stream references an unknown profiler"))?;
        actual_by_profiler
            .entry(profiler_id.to_owned())
            .or_default()
            .push(stream_id);
        streams.push(stream_view(
            stream,
            origin,
            run_id,
            scope_id,
            &artifact_by_id,
            profiler,
        )?);
    }
    for (profiler_id, profiler) in &profiler_by_id {
        let expected = profiler
            .get("stream_ids")
            .and_then(Value::as_array)
            .expect("profiler view stream ids");
        let actual = actual_by_profiler
            .get(profiler_id)
            .cloned()
            .unwrap_or_default();
        let expected = expected
            .iter()
            .filter_map(Value::as_str)
            .map(str::to_owned)
            .collect::<Vec<_>>();
        if actual != expected {
            return Err(UnifiedViewError::new(format!(
                "profiler {profiler_id:?} manifest stream IDs do not match bundle streams"
            )));
        }
        if profiler.get("status").and_then(Value::as_str) == Some("missing")
            && streams.iter().any(|stream| {
                stream.get("profiler_id").and_then(Value::as_str) == Some(profiler_id.as_str())
                    && stream.get("status").and_then(Value::as_str) != Some("missing")
            })
        {
            return Err(UnifiedViewError::new(
                "missing profiler may contain only missing streams",
            ));
        }
    }
    streams.sort_by(|left, right| {
        let key = |stream: &Value| {
            (
                stream
                    .get("group")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                stream.get("order").and_then(Value::as_f64).unwrap_or(0.0),
                stream
                    .get("label")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
                stream
                    .get("stream_id")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            )
        };
        let left_key = key(left);
        let right_key = key(right);
        left_key
            .0
            .cmp(&right_key.0)
            .then_with(|| {
                left_key
                    .1
                    .partial_cmp(&right_key.1)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| left_key.2.cmp(&right_key.2))
            .then_with(|| left_key.3.cmp(&right_key.3))
    });
    let summaries = streams
        .iter()
        .filter(|stream| stream.get("placement").and_then(Value::as_str) == Some("summary"))
        .cloned()
        .collect::<Vec<_>>();
    let tracks = streams
        .iter()
        .filter(|stream| stream.get("placement").and_then(Value::as_str) == Some("timeline"))
        .cloned()
        .collect::<Vec<_>>();

    let mut markers_raw = list_default(
        timeline.get("lifecycle_markers"),
        "timeline.lifecycle_markers",
    )?;
    markers_raw.sort_by(|left, right| {
        let key = |marker: &Value| {
            (
                marker
                    .get("canonical_timestamp_ns")
                    .and_then(Value::as_i64)
                    .unwrap_or(0),
                marker
                    .get("name")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
                    .to_owned(),
            )
        };
        key(left).cmp(&key(right))
    });
    let mut marker_names = BTreeSet::new();
    let mut markers = Vec::new();
    for marker in markers_raw {
        let marker = object(&marker, "marker")?;
        let name = text(marker.get("name"), "marker.name")?;
        if !marker_names.insert(name.to_owned()) {
            return Err(UnifiedViewError::new(
                "timeline lifecycle marker names must be unique",
            ));
        }
        let timestamp = nonnegative(
            marker.get("canonical_timestamp_ns"),
            "marker.canonical_timestamp_ns",
        )?;
        if timestamp < start || timestamp > end {
            return Err(UnifiedViewError::new(format!(
                "lifecycle marker {name:?} falls outside the run window"
            )));
        }
        markers.push(map([
            ("name", Value::String(name.to_owned())),
            ("canonical_timestamp_ns", Value::from(timestamp)),
            ("offset_ns", Value::from(timestamp - origin)),
            (
                "alignment_error_ns",
                Value::from(match marker.get("alignment_error_ns") {
                    Some(value) => nonnegative(Some(value), "marker.alignment_error_ns")?,
                    None => 0,
                }),
            ),
            (
                "provenance",
                mapping_default(marker.get("provenance"), "marker.provenance")?,
            ),
        ]));
    }

    validate_clocks(bundle, clock_id)?;
    let mut mappings_raw = array(bundle.get("clock_mappings"), "bundle.clock_mappings")?.clone();
    mappings_raw.sort_by(|left, right| {
        left.get("source_clock_id")
            .and_then(Value::as_str)
            .cmp(&right.get("source_clock_id").and_then(Value::as_str))
    });
    let mappings = mappings_raw
        .iter()
        .map(normalize_clock_mapping)
        .collect::<Result<Vec<_>, _>>()?;

    let extensions = mapping_default(bundle.get("extensions"), "bundle.extensions")?;
    let extension_object = extensions.as_object().expect("mapping");
    let declared = extension_object
        .get("evidence_status")
        .and_then(Value::as_str);
    let profiler_statuses = profiler_by_id
        .values()
        .filter_map(|profiler| profiler.get("status").and_then(Value::as_str))
        .collect::<BTreeSet<_>>();
    let evidence_status = if declared == Some("invalid")
        || profiler_statuses.contains("invalid")
        || profiler_statuses.contains("dropped")
    {
        "invalid"
    } else if declared == Some("partial")
        || profiler_statuses.contains("missing")
        || profiler_statuses.contains("unalignable")
    {
        "partial"
    } else {
        "valid"
    };
    let mut evidence_reasons = BTreeSet::new();
    for profiler in profiler_by_id.values() {
        if profiler.get("status").and_then(Value::as_str) != Some("valid") {
            for reason in profiler
                .get("reasons")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
            {
                evidence_reasons.insert(reason.to_owned());
            }
        }
    }
    if let Some(reasons) = extension_object
        .get("evidence_invalid_reasons")
        .and_then(Value::as_array)
    {
        for reason in reasons
            .iter()
            .filter_map(Value::as_str)
            .filter(|reason| !reason.is_empty())
        {
            evidence_reasons.insert(reason.to_owned());
        }
    }

    let mut shape_registry = Map::new();
    for shape in [
        "cumulative_counter",
        "interval",
        "point_event",
        "sampled_gauge",
        "scalar",
        "window_aggregate",
    ] {
        let (placement, render_kind) = shape_registration(shape).expect("known shape");
        shape_registry.insert(
            shape.to_owned(),
            map([
                ("placement", Value::String(placement.to_owned())),
                ("render_kind", Value::String(render_kind.to_owned())),
            ]),
        );
    }
    Ok(map([
        (
            "schema_version",
            Value::String(VIEW_MODEL_SCHEMA_VERSION.to_owned()),
        ),
        ("run_id", Value::String(run_id.to_owned())),
        ("scope", normalized_scope),
        (
            "axis",
            map([
                ("clock_id", Value::String(clock_id.to_owned())),
                ("clock_name", Value::String("CLOCK_MONOTONIC".to_owned())),
                ("origin_ns", Value::from(origin)),
                ("start_ns", Value::from(start)),
                ("end_ns", Value::from(end)),
                ("minimum_offset_ns", Value::from(start - origin)),
                ("maximum_offset_ns", Value::from(end - origin)),
                ("duration_ns", Value::from(end - start)),
                (
                    "meaning",
                    Value::String(
                        "run-relative nanoseconds from the declared canonical origin".to_owned(),
                    ),
                ),
            ]),
        ),
        ("lifecycle_markers", Value::Array(markers)),
        (
            "evidence_status",
            map([
                ("status", Value::String(evidence_status.to_owned())),
                (
                    "reasons",
                    Value::Array(evidence_reasons.into_iter().map(Value::String).collect()),
                ),
                (
                    "all_selected_profilers_valid",
                    extension_object
                        .get("all_selected_profilers_valid")
                        .or_else(|| extension_object.get("all_selected_profilers_usable"))
                        .cloned()
                        .unwrap_or(Value::Null),
                ),
            ]),
        ),
        ("shape_registry", Value::Object(shape_registry)),
        (
            "profilers",
            Value::Array(profiler_by_id.into_values().collect()),
        ),
        ("summaries", Value::Array(summaries)),
        ("tracks", Value::Array(tracks)),
        (
            "artifacts",
            Value::Array(artifact_by_id.into_values().collect()),
        ),
        ("clock_mappings", Value::Array(mappings)),
        (
            "provenance",
            mapping_default(bundle.get("provenance"), "bundle.provenance")?,
        ),
        ("extensions", extensions),
    ]))
}

fn validate_clocks(
    bundle: &Map<String, Value>,
    canonical_id: &str,
) -> Result<(), UnifiedViewError> {
    let clocks = array(bundle.get("clocks"), "bundle.clocks")?;
    let mut ids = BTreeSet::new();
    let mut canonical_seen = false;
    for clock in clocks {
        let clock = object(clock, "clock")?;
        let id = text(clock.get("clock_id"), "clock.clock_id")?;
        if !ids.insert(id.to_owned()) {
            return Err(UnifiedViewError::new(format!(
                "duplicate clock identity: {id}"
            )));
        }
        let name = text(clock.get("clock_name"), "clock.clock_name")?;
        if integer(clock.get("resolution_ns"), "clock.resolution_ns")? < 1 {
            return Err(UnifiedViewError::new(
                "clock.resolution_ns must be at least 1",
            ));
        }
        if id == canonical_id {
            canonical_seen = true;
            if name != "CLOCK_MONOTONIC" {
                return Err(UnifiedViewError::new(
                    "canonical timeline clock must be CLOCK_MONOTONIC",
                ));
            }
        }
    }
    if !canonical_seen {
        return Err(UnifiedViewError::new(
            "canonical timeline clock is absent from clocks",
        ));
    }
    Ok(())
}

fn normalize_clock_mapping(value: &Value) -> Result<Value, UnifiedViewError> {
    let mapping = object(value, "clock_mapping")?;
    let source = text(
        mapping.get("source_clock_id"),
        "clock_mapping.source_clock_id",
    )?;
    let canonical = text(
        mapping.get("canonical_clock_id"),
        "clock_mapping.canonical_clock_id",
    )?;
    let status = text(mapping.get("status"), "clock_mapping.status")?;
    if !matches!(status, "valid" | "unalignable") {
        return Err(UnifiedViewError::new(
            "clock_mapping.status must be valid or unalignable",
        ));
    }
    let reasons = normalized_reasons(status, mapping.get("reasons"), "clock_mapping")?;
    Ok(map([
        ("source_clock_id", Value::String(source.to_owned())),
        ("canonical_clock_id", Value::String(canonical.to_owned())),
        ("status", Value::String(status.to_owned())),
        (
            "method",
            Value::String(text(mapping.get("method"), "clock_mapping.method")?.to_owned()),
        ),
        (
            "scale_numerator",
            clone_or_null(mapping.get("scale_numerator")),
        ),
        (
            "scale_denominator",
            clone_or_null(mapping.get("scale_denominator")),
        ),
        ("offset_ns", clone_or_null(mapping.get("offset_ns"))),
        ("max_error_ns", clone_or_null(mapping.get("max_error_ns"))),
        ("reasons", Value::Array(reasons)),
        (
            "provenance",
            mapping_default(mapping.get("provenance"), "clock_mapping.provenance")?,
        ),
    ]))
}

fn html_escape(value: &str, quote: bool) -> String {
    let mut escaped = String::with_capacity(value.len());
    for character in value.chars() {
        match character {
            '&' => escaped.push_str("&amp;"),
            '<' => escaped.push_str("&lt;"),
            '>' => escaped.push_str("&gt;"),
            '"' if quote => escaped.push_str("&quot;"),
            '\'' if quote => escaped.push_str("&#x27;"),
            _ => escaped.push(character),
        }
    }
    escaped
}

fn comma_integer(value: i64) -> String {
    let negative = value < 0;
    let digits = value.unsigned_abs().to_string();
    let mut result = String::new();
    for (index, character) in digits.chars().enumerate() {
        if index > 0 && (digits.len() - index) % 3 == 0 {
            result.push(',');
        }
        result.push(character);
    }
    if negative {
        result.insert(0, '-');
    }
    result
}

fn compact_json(value: &Value) -> String {
    python_json(value, None)
}

fn pretty_json(value: &Value) -> String {
    python_json(value, Some(2))
}

fn display_text(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "unavailable".to_owned(),
        Some(Value::Bool(value)) => value.to_string(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) if value.is_i64() || value.is_u64() => value.to_string(),
        Some(Value::Number(value)) => format_python_9g(value.as_f64().unwrap_or_default()),
        Some(value) => compact_json(value),
    }
}

// Python's `.9g` formatting is used only for displayed scalar floating-point
// evidence. This covers ordinary and scientific output while keeping exponent
// spelling compatible with CPython.
fn format_python_9g(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0" } else { "0" }.to_owned();
    }
    // First round in scientific notation to discover the exponent *after*
    // nine-significant-digit rounding. Python's `g` selection uses that
    // exponent, which matters at boundaries such as 999_999_999.9.
    let scientific = format!("{value:.8e}");
    let (mantissa, exponent) = scientific.split_once('e').expect("scientific float");
    let exponent = exponent.parse::<i32>().expect("scientific exponent");
    if !(-4..9).contains(&exponent) {
        let mantissa = mantissa.trim_end_matches('0').trim_end_matches('.');
        format!("{mantissa}e{exponent:+03}")
    } else {
        let decimals = (8 - exponent).max(0) as usize;
        let rendered = format!("{value:.decimals$}");
        if rendered.contains('.') {
            rendered
                .trim_end_matches('0')
                .trim_end_matches('.')
                .to_owned()
        } else {
            rendered
        }
    }
}

fn status_badge(status: &str) -> String {
    format!(
        "<span class=\"status status-{}\" style=\"--status-color:{}\">{}</span>",
        html_escape(status, true),
        status_color(status),
        html_escape(&status.to_uppercase(), true)
    )
}

fn artifacts(value: Option<&Value>) -> &[Value] {
    value
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[])
}

fn artifact_links_html(value: Option<&Value>) -> String {
    let artifacts = artifacts(value);
    if artifacts.is_empty() {
        return "<span class=\"muted\">raw evidence: none declared</span>".to_owned();
    }
    let links = artifacts
        .iter()
        .map(|artifact| {
            format!(
                "<a href=\"{}\" data-sha256=\"{}\">{}</a>",
                html_escape(
                    artifact
                        .get("href")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
                html_escape(
                    artifact
                        .get("sha256")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
                html_escape(
                    artifact
                        .get("artifact_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    format!("raw evidence: {links}")
}

fn reasons_html(value: Option<&Value>) -> String {
    let Some(reasons) = value.and_then(Value::as_array) else {
        return String::new();
    };
    if reasons.is_empty() {
        return String::new();
    }
    let items = reasons.iter().fold(String::new(), |mut output, reason| {
        write!(
            &mut output,
            "<li>{}</li>",
            html_escape(reason.as_str().unwrap_or_default(), true)
        )
        .expect("writing to a String cannot fail");
        output
    });
    format!("<ul class=\"reasons\">{items}</ul>")
}

fn stream_metadata_html(stream: &Value) -> String {
    let cadence_text = if let Some(cadence) = stream.get("cadence").and_then(Value::as_object) {
        format!(
            "requested {} ns; maximum gap {} ns; {}",
            comma_integer(
                cadence
                    .get("requested_interval_ns")
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
            ),
            comma_integer(
                cadence
                    .get("maximum_acceptable_gap_ns")
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
            ),
            html_escape(
                cadence
                    .get("method")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                true
            ),
        )
    } else {
        "unsampled / event-driven".to_owned()
    };
    let gap_values = stream
        .get("gaps")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let gap_text = if gap_values.is_empty() {
        "none".to_owned()
    } else {
        gap_values
            .iter()
            .map(|gap| {
                let duration = gap
                    .get("end_ns")
                    .and_then(Value::as_i64)
                    .unwrap_or_default()
                    - gap
                        .get("start_ns")
                        .and_then(Value::as_i64)
                        .unwrap_or_default();
                format!(
                    "{} ({} ns)",
                    html_escape(
                        gap.get("reason")
                            .and_then(Value::as_str)
                            .unwrap_or_default(),
                        true
                    ),
                    comma_integer(duration)
                )
            })
            .collect::<Vec<_>>()
            .join("; ")
    };
    let uncertainty = match stream.get("maximum_alignment_error_ns") {
        Some(Value::Number(value)) => {
            format!("±{} ns", comma_integer(value.as_i64().unwrap_or_default()))
        }
        _ => "not applicable / unaligned".to_owned(),
    };
    let provenance = pretty_json(stream.get("provenance").unwrap_or(&Value::Null));
    let native_only = stream
        .get("records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|record| {
            record.get("native_timestamp_ns").is_some()
                && record.get("canonical_timestamp_offset_ns").is_none()
                && record.get("acquisition_start_offset_ns").is_none()
        })
        .map(|record| {
            format!(
                "{} @ {} ns ({})",
                html_escape(
                    record
                        .get("record_id")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
                comma_integer(
                    record
                        .get("native_timestamp_ns")
                        .and_then(Value::as_i64)
                        .unwrap_or_default()
                ),
                html_escape(
                    record
                        .get("source_clock_id")
                        .and_then(Value::as_str)
                        .unwrap_or("unknown clock"),
                    true
                ),
            )
        })
        .collect::<Vec<_>>();
    let native_text = if native_only.is_empty() {
        String::new()
    } else {
        format!(
            "<div><b>native-only timing</b>: {}</div>",
            native_only.join(", ")
        )
    };
    format!(
        "\n<div class=\"stream-meta\">\n  <div><b>cadence</b>: {cadence_text}</div>\n  <div><b>gaps</b>: {gap_text}</div>\n  <div><b>alignment uncertainty</b>: {uncertainty}</div>\n  {native_text}\n  <div>{}</div>\n  <details class=\"provenance\"><summary>provenance</summary><pre>{}</pre></details>\n</div>",
        artifact_links_html(stream.get("raw_artifacts")),
        html_escape(&provenance, true),
    )
}

fn x_for_offset(offset: i64, axis: &Value) -> f64 {
    let minimum = axis
        .get("minimum_offset_ns")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let maximum = axis
        .get("maximum_offset_ns")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let fraction = (offset - minimum) as f64 / (maximum - minimum).max(1) as f64;
    PLOT_LEFT + (PLOT_WIDTH - PLOT_LEFT - PLOT_RIGHT) * fraction
}

fn y_for_value(value: f64, domain: &Value) -> f64 {
    let minimum = domain
        .get("minimum")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let maximum = domain
        .get("maximum")
        .and_then(Value::as_f64)
        .unwrap_or_default();
    let fraction = (value - minimum) / (maximum - minimum).max(1e-30);
    PLOT_HEIGHT - PLOT_BOTTOM - fraction * (PLOT_HEIGHT - PLOT_TOP - PLOT_BOTTOM)
}

fn marker_lines(model: &Value) -> String {
    let axis = &model["axis"];
    model
        .get("lifecycle_markers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .fold(String::new(), |mut output, marker| {
            let x = x_for_offset(marker.get("offset_ns").and_then(Value::as_i64).unwrap_or_default(), axis);
            write!(
                &mut output,
                "<line class=\"lifecycle-marker\" data-marker=\"{}\" x1=\"{x:.3}\" y1=\"0\" x2=\"{x:.3}\" y2=\"{PLOT_HEIGHT:.3}\"/>",
                html_escape(marker.get("name").and_then(Value::as_str).unwrap_or_default(), true),
            )
            .expect("writing to a String cannot fail");
            output
        })
}

fn gap_bands(stream: &Value, axis: &Value) -> String {
    stream
        .get("gaps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter(|gap| gap.get("canonical").and_then(Value::as_bool).unwrap_or(false))
        .fold(String::new(), |mut output, gap| {
            let x1 = x_for_offset(gap.get("start_offset_ns").and_then(Value::as_i64).unwrap_or_default(), axis);
            let x2 = x_for_offset(gap.get("end_offset_ns").and_then(Value::as_i64).unwrap_or_default(), axis);
            write!(
                &mut output,
                "<rect class=\"evidence-gap\" x=\"{x1:.3}\" y=\"0\" width=\"{:.3}\" height=\"{PLOT_HEIGHT:.3}\"><title>{}</title></rect>",
                (x2 - x1).max(1.0),
                html_escape(gap.get("reason").and_then(Value::as_str).unwrap_or_default(), true),
            )
            .expect("writing to a String cannot fail");
            output
        })
}

fn record_x(record: &Value, axis: &Value) -> Option<f64> {
    if let Some(offset) = record
        .get("canonical_timestamp_offset_ns")
        .and_then(Value::as_i64)
    {
        return Some(x_for_offset(offset, axis));
    }
    let start = record
        .get("acquisition_start_offset_ns")
        .and_then(Value::as_i64)?;
    let end = record
        .get("acquisition_end_offset_ns")
        .and_then(Value::as_i64)?;
    Some(x_for_offset((start + end) / 2, axis))
}

fn uncertainty_glyph(record: &Value, axis: &Value, y: f64) -> String {
    let range = if let (Some(start), Some(end)) = (
        record
            .get("acquisition_start_offset_ns")
            .and_then(Value::as_i64),
        record
            .get("acquisition_end_offset_ns")
            .and_then(Value::as_i64),
    ) {
        Some((start, end))
    } else if let (Some(timestamp), Some(error)) = (
        record
            .get("canonical_timestamp_offset_ns")
            .and_then(Value::as_i64),
        record.get("alignment_error_ns").and_then(Value::as_i64),
    ) {
        Some((timestamp - error, timestamp + error))
    } else {
        None
    };
    let Some((start, end)) = range else {
        return String::new();
    };
    let x1 = x_for_offset(start, axis);
    let x2 = x_for_offset(end, axis);
    format!(
        "<line class=\"alignment-uncertainty\" x1=\"{x1:.3}\" y1=\"{y:.3}\" x2=\"{x2:.3}\" y2=\"{y:.3}\"/>"
    )
}

fn render_point_ticks(stream: &Value, axis: &Value) -> String {
    stream
        .get("records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, record)| {
            let x = record_x(record, axis)?;
            let y1 = 26.0 + (index % 3) as f64 * 18.0;
            let y2 = y1 + 28.0;
            let status = record.get("status").and_then(Value::as_str).unwrap_or_default();
            let glyph = uncertainty_glyph(record, axis, (y1 + y2) / 2.0);
            let rendered = if status == "dropped" {
                format!(
                    "<path class=\"record record-dropped\" d=\"M{:.3},{:.3} L{:.3},{:.3} M{:.3},{:.3} L{:.3},{:.3}\"/>",
                    x - 5.0, y1 + 9.0, x + 5.0, y1 + 19.0,
                    x + 5.0, y1 + 9.0, x - 5.0, y1 + 19.0,
                )
            } else {
                format!(
                    "<line class=\"record record-{}\" x1=\"{x:.3}\" y1=\"{y1:.3}\" x2=\"{x:.3}\" y2=\"{y2:.3}\"><title>{}</title></line>",
                    html_escape(status, true),
                    html_escape(&display_text(record.get("value")), true),
                )
            };
            Some(glyph + &rendered)
        })
        .collect()
}

fn render_bands(stream: &Value, axis: &Value, window: bool) -> String {
    stream
        .get("records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .enumerate()
        .filter_map(|(index, record)| {
            let start = record.get("canonical_start_offset_ns").and_then(Value::as_i64)?;
            let end = record.get("canonical_end_offset_ns").and_then(Value::as_i64)?;
            let x1 = x_for_offset(start, axis);
            let x2 = x_for_offset(end, axis);
            let y = 25.0 + (index % 3) as f64 * 23.0;
            let class = if window { "window-band" } else { "interval-band" };
            Some(format!(
                "<rect class=\"record {class} record-{}\" x=\"{x1:.3}\" y=\"{y:.3}\" width=\"{:.3}\" height=\"16\"><title>{}</title></rect>",
                html_escape(record.get("status").and_then(Value::as_str).unwrap_or_default(), true),
                (x2 - x1).max(1.0),
                html_escape(&display_text(record.get("value")), true),
            ))
        })
        .collect()
}

fn render_series(stream: &Value, axis: &Value) -> String {
    let Some(domain) = stream
        .get("numeric_value_domain")
        .filter(|value| value.is_object())
    else {
        return String::new();
    };
    let points = stream
        .get("records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|record| {
            let value = record.get("value")?;
            if value.is_boolean() {
                return None;
            }
            let value = value.as_f64().filter(|value| value.is_finite())?;
            let x = record_x(record, axis)?;
            Some((x, y_for_value(value, domain), record))
        })
        .collect::<Vec<_>>();
    let valid = points
        .iter()
        .filter(|(_, _, record)| record.get("status").and_then(Value::as_str) == Some("valid"))
        .map(|(x, y, _)| (*x, *y))
        .collect::<Vec<_>>();
    let mut result = String::new();
    if valid.len() >= 2 {
        result.push_str(&format!(
            "<polyline class=\"series-line\" points=\"{}\"/>",
            valid
                .iter()
                .map(|(x, y)| format!("{x:.3},{y:.3}"))
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    for (x, y, record) in points {
        result.push_str(&uncertainty_glyph(record, axis, y));
        result.push_str(&format!(
            "<circle class=\"record series-point record-{}\" cx=\"{x:.3}\" cy=\"{y:.3}\" r=\"3.5\"><title>{}</title></circle>",
            html_escape(record.get("status").and_then(Value::as_str).unwrap_or_default(), true),
            html_escape(&display_text(record.get("value")), true),
        ));
    }
    result
}

fn track_status_message(stream: &Value) -> String {
    let status = stream
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let aligned = stream
        .get("records")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .any(|record| {
            record.get("canonical_timestamp_offset_ns").is_some()
                || record.get("canonical_start_offset_ns").is_some()
                || record.get("acquisition_start_offset_ns").is_some()
        });
    if status == "valid" && aligned {
        return String::new();
    }
    let mut detail = stream
        .get("reasons")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join("; ");
    if detail.is_empty() {
        detail = "no aligned records".to_owned();
    }
    if status == "unalignable" {
        let count = stream
            .get("records")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        detail = format!("{detail}; {count} native-timed record(s) retained off-axis");
    }
    format!(
        "<text class=\"track-status-message status-text-{}\" x=\"{:.3}\" y=\"{:.3}\">{}</text>",
        html_escape(status, true),
        PLOT_LEFT + 8.0,
        PLOT_HEIGHT / 2.0,
        html_escape(&format!("{}: {detail}", status.to_uppercase()), true),
    )
}

fn track_plot_svg(stream: &Value, model: &Value) -> Result<String, UnifiedViewError> {
    let axis = &model["axis"];
    let records = match stream.get("render_kind").and_then(Value::as_str) {
        Some("point_ticks") => render_point_ticks(stream, axis),
        Some("interval_bands") => render_bands(stream, axis, false),
        Some("window_bands") => render_bands(stream, axis, true),
        Some("sample_series" | "counter_series") => render_series(stream, axis),
        other => {
            return Err(UnifiedViewError::new(format!(
                "unsupported timeline render kind: {other:?}"
            )));
        }
    };
    let minimum = axis
        .get("minimum_offset_ns")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let maximum = axis
        .get("maximum_offset_ns")
        .and_then(Value::as_i64)
        .unwrap_or_default();
    let mut ticks = String::new();
    for index in 0..5 {
        let offset = minimum + (maximum - minimum) * index / 4;
        let x = x_for_offset(offset, axis);
        ticks.push_str(&format!(
            "<line class=\"axis-grid\" x1=\"{x:.3}\" y1=\"0\" x2=\"{x:.3}\" y2=\"{PLOT_HEIGHT:.3}\"/>"
        ));
        ticks.push_str(&format!(
            "<text class=\"axis-tick\" x=\"{x:.3}\" y=\"{:.3}\" text-anchor=\"middle\">{:.3}s</text>",
            PLOT_HEIGHT - 4.0,
            offset as f64 / 1_000_000_000.0,
        ));
    }
    Ok(format!(
        "{ticks}{}{}{records}{}<line class=\"shared-cursor\" x1=\"{PLOT_LEFT:.3}\" y1=\"0\" x2=\"{PLOT_LEFT:.3}\" y2=\"{PLOT_HEIGHT:.3}\"/>",
        gap_bands(stream, axis),
        marker_lines(model),
        track_status_message(stream),
    ))
}

fn summary_card_html(summary: &Value) -> String {
    let records = summary
        .get("records")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let mut values = records
        .iter()
        .map(|record| {
            format!(
                "<span class=\"summary-value record-{}\">{} <small>{}</small></span>",
                html_escape(
                    record
                        .get("status")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
                html_escape(&display_text(record.get("value")), true),
                html_escape(
                    summary
                        .get("unit")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
            )
        })
        .collect::<Vec<_>>()
        .join("<br>");
    if values.is_empty() {
        values = "<span class=\"summary-value muted\">no value</span>".to_owned();
    }
    let exact = !records.is_empty()
        && summary.get("status").and_then(Value::as_str) == Some("valid")
        && records.iter().all(|record| {
            record
                .get("attributes")
                .and_then(|value| value.get("exact"))
                == Some(&Value::Bool(true))
        });
    let timing = if exact {
        "exact untimed fact"
    } else {
        "untimed evidence"
    };
    let provenance = pretty_json(summary.get("provenance").unwrap_or(&Value::Null));
    let status = summary
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "\n<article class=\"summary-card status-border-{}\" data-stream-id=\"{}\">\n  <header><span>{}</span>{}</header>\n  {values}\n  {}\n  <div class=\"summary-meta\">{timing}; never placed on the timeline</div>\n  <div>{}</div>\n  <details class=\"provenance\"><summary>provenance</summary><pre>{}</pre></details>\n</article>",
        html_escape(status, true),
        html_escape(summary.get("stream_id").and_then(Value::as_str).unwrap_or_default(), true),
        html_escape(summary.get("label").and_then(Value::as_str).unwrap_or_default(), true),
        status_badge(status),
        reasons_html(summary.get("reasons")),
        artifact_links_html(summary.get("raw_artifacts")),
        html_escape(&provenance, true),
    )
}

fn profiler_card_html(profiler: &Value) -> String {
    let status = profiler
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let count = profiler
        .get("stream_ids")
        .and_then(Value::as_array)
        .map(Vec::len)
        .unwrap_or(0);
    let intrusive = if profiler
        .get("intrusive")
        .and_then(Value::as_bool)
        .unwrap_or(false)
    {
        " · intrusive"
    } else {
        ""
    };
    let id = profiler
        .get("profiler_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    format!(
        "\n<article class=\"profiler-card status-border-{}\" data-profiler-id=\"{}\">\n  <header><span>{}</span>{}</header>\n  <div class=\"profiler-meta\">{count} stream(s){intrusive}</div>\n  {}\n</article>",
        html_escape(status, true),
        html_escape(id, true),
        html_escape(id, true),
        status_badge(status),
        reasons_html(profiler.get("reasons")),
    )
}

fn embedded_model_json(view_model: &Value) -> Result<String, UnifiedViewError> {
    let model = String::from_utf8(view_model_bytes(view_model)?)
        .map_err(|error| UnifiedViewError::new(error.to_string()))?;
    Ok(model
        .trim_end_matches('\n')
        .replace('&', "\\u0026")
        .replace('<', "\\u003c")
        .replace('>', "\\u003e"))
}

pub fn render_unified_html(view_model: &Value) -> Result<String, UnifiedViewError> {
    let digest = view_model_sha256(view_model)?;
    let mut summaries = view_model
        .get("summaries")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(summary_card_html)
        .collect::<String>();
    if summaries.is_empty() {
        summaries = "<p class=\"muted\">No scalar summaries were emitted.</p>".to_owned();
    }
    let profilers = view_model
        .get("profilers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(profiler_card_html)
        .collect::<String>();
    let evidence_status = &view_model["evidence_status"];
    let evidence_status_text = evidence_status
        .get("status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let evidence_reason_text = evidence_status
        .get("reasons")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .collect::<Vec<_>>()
        .join("; ");
    let mut tracks = String::new();
    for track in view_model
        .get("tracks")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let status = track
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let label = track
            .get("label")
            .and_then(Value::as_str)
            .unwrap_or_default();
        tracks.push_str(&format!(
            "\n<details class=\"track status-border-{}\" open data-stream-id=\"{}\">\n  <summary><span class=\"track-group\">{}</span> · {} <small>({})</small> {}</summary>\n  {}\n  <svg class=\"track-plot\" viewBox=\"0 0 {:.0} {:.0}\" role=\"img\" aria-label=\"{} timeline\">{}</svg>\n  {}\n</details>",
            html_escape(status, true),
            html_escape(track.get("stream_id").and_then(Value::as_str).unwrap_or_default(), true),
            html_escape(track.get("group").and_then(Value::as_str).unwrap_or_default(), true),
            html_escape(label, true),
            html_escape(track.get("unit").and_then(Value::as_str).unwrap_or_default(), true),
            status_badge(status),
            reasons_html(track.get("reasons")),
            PLOT_WIDTH,
            PLOT_HEIGHT,
            html_escape(label, true),
            track_plot_svg(track, view_model)?,
            stream_metadata_html(track),
        ));
    }
    let marker_values = view_model
        .get("lifecycle_markers")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|marker| {
            format!(
                "{} ({:.6}s ±{}ns)",
                html_escape(
                    marker
                        .get("name")
                        .and_then(Value::as_str)
                        .unwrap_or_default(),
                    true
                ),
                marker
                    .get("offset_ns")
                    .and_then(Value::as_i64)
                    .unwrap_or_default() as f64
                    / 1_000_000_000.0,
                comma_integer(
                    marker
                        .get("alignment_error_ns")
                        .and_then(Value::as_i64)
                        .unwrap_or_default()
                ),
            )
        })
        .collect::<Vec<_>>()
        .join(", ");
    let markers = if marker_values.is_empty() {
        "none"
    } else {
        &marker_values
    };
    let embedded = embedded_model_json(view_model)?;
    let reason_suffix = if evidence_reason_text.is_empty() {
        String::new()
    } else {
        format!(" · {}", html_escape(&evidence_reason_text, true))
    };
    let run_id = html_escape(
        view_model
            .get("run_id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        true,
    );
    let scope_id = html_escape(
        view_model
            .pointer("/scope/scope_id")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        true,
    );
    let clock_name = html_escape(
        view_model
            .pointer("/axis/clock_name")
            .and_then(Value::as_str)
            .unwrap_or_default(),
        true,
    );
    Ok(format!(
        r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="view-model-sha256" content="{digest}">
<title>Unified sandbox profile</title>
<style>
:root{{--bg:#f8fafc;--panel:#fff;--ink:#111827;--muted:#64748b;--line:#cbd5e1}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:14px/1.45 system-ui,sans-serif}}
main{{max-width:1280px;margin:auto;padding:24px}} h1{{margin:0 0 4px}} .subtitle,.muted,.summary-meta{{color:var(--muted)}}
.controls{{position:sticky;top:0;z-index:5;display:flex;gap:18px;align-items:center;padding:10px 12px;margin:18px 0;background:#fffffff2;border:1px solid var(--line);border-radius:8px}}
.controls label{{display:flex;gap:8px;align-items:center}} #cursor-readout{{font-variant-numeric:tabular-nums}}
.summaries,.profiler-statuses{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px;margin:14px 0 22px}}
.summary-card,.profiler-card,.track{{background:var(--panel);border:1px solid var(--line);border-left-width:5px;border-radius:8px;padding:11px}}
.summary-card header{{display:flex;justify-content:space-between;gap:8px}} .summary-value{{display:block;font-size:22px;font-weight:700;margin:10px 0}}
.profiler-card header{{display:flex;justify-content:space-between;gap:8px;font-weight:700}} .profiler-meta{{margin-top:7px;color:var(--muted)}}
.status{{display:inline-block;padding:2px 7px;border:1px solid var(--status-color);border-radius:999px;color:var(--status-color);font-size:10px;font-weight:800;vertical-align:middle}}
.status-border-valid{{border-left-color:#15803d}}.status-border-partial{{border-left-color:#a16207}}.status-border-invalid{{border-left-color:#b91c1c}}.status-border-missing{{border-left-color:#6b7280}}.status-border-dropped{{border-left-color:#c2410c}}.status-border-unalignable{{border-left-color:#7e22ce}}
.track{{margin:10px 0;padding:0}} .track>summary{{cursor:pointer;padding:10px 12px;font-weight:650}} .track-group{{color:#475569}} .track-plot{{display:block;width:100%;height:128px;border-top:1px solid #e2e8f0;border-bottom:1px solid #e2e8f0;background:#fff;touch-action:none}}
.stream-meta{{display:flex;flex-wrap:wrap;gap:8px 18px;padding:8px 12px;color:#475569;font-size:12px}} .provenance{{width:100%}} pre{{white-space:pre-wrap;overflow-wrap:anywhere}}
.reasons{{margin:4px 12px 8px;color:#991b1b}} a{{color:#1d4ed8}} .axis-grid{{stroke:#e5e7eb;stroke-width:1}} .axis-tick{{fill:#64748b;font-size:10px}}
.lifecycle-marker{{stroke:#0f766e;stroke-width:1.2;stroke-dasharray:5 4}} .evidence-gap{{fill:#fee2e2;opacity:.75}}
.record{{stroke-width:2.5}} .record-valid{{stroke:#15803d;fill:#15803d}}.record-invalid{{stroke:#b91c1c;fill:#b91c1c;stroke-dasharray:4 2}}.record-dropped{{stroke:#c2410c;fill:none}}.record-unalignable{{stroke:#7e22ce;fill:#7e22ce}}
.series-line{{fill:none;stroke:#2563eb;stroke-width:2}} .series-point{{stroke:#fff;stroke-width:1}} .interval-band{{fill:#2563eb;opacity:.7}} .window-band{{fill:#7c3aed;opacity:.45;stroke-dasharray:4 2}} .alignment-uncertainty{{stroke:#111827;stroke-width:1;opacity:.65}} .shared-cursor{{stroke:#e11d48;stroke-width:1.5;visibility:hidden;pointer-events:none}} .track-status-message{{font-size:12px;fill:#475569}} .status-text-invalid{{fill:#b91c1c}}.status-text-missing{{fill:#6b7280}}.status-text-dropped{{fill:#c2410c}}.status-text-unalignable{{fill:#7e22ce}}
</style>
</head>
<body data-view-model-sha256="{digest}"><main>
<h1>Unified sandbox profile</h1>
<div class="subtitle">run {run_id} · scope {scope_id} · canonical {clock_name}</div>
<div class="subtitle">lifecycle markers: {markers}</div>
<div class="summary-card status-border-{status_class}"><b>Evidence status: {status_upper}</b>{reason_suffix}</div>
<div class="controls"><label>Zoom <input id="zoom" type="range" min="1" max="16" value="1" step="1"></label><button id="reset" type="button">Reset view</button><output id="cursor-readout">Move over a track to inspect aligned time</output></div>
<h2>Profiler status</h2><section class="profiler-statuses">{profilers}</section>
<h2>Untimed summaries</h2><section class="summaries">{summaries}</section>
<h2>Timeline tracks</h2><p class="muted">All aligned tracks share the axis below; native-only evidence remains labeled off-axis.</p><section id="tracks">{tracks}</section>
<script id="view-model" type="application/json">{embedded}</script>
<script>
(()=>{{
  const model=JSON.parse(document.getElementById('view-model').textContent);
  const plots=[...document.querySelectorAll('.track-plot')];
  const zoom=document.getElementById('zoom'); const readout=document.getElementById('cursor-readout');
  const baseWidth=1000.0; let visibleWidth=baseWidth;
  function applyZoom(){{visibleWidth=baseWidth/Number(zoom.value); const x=(baseWidth-visibleWidth)/2; plots.forEach(p=>p.setAttribute('viewBox',`${{x}} 0 ${{visibleWidth}} 112.0`));}}
  function move(event){{const source=event.currentTarget; const rect=source.getBoundingClientRect(); const vb=source.viewBox.baseVal; const x=vb.x+Math.max(0,Math.min(1,(event.clientX-rect.left)/rect.width))*vb.width; plots.forEach(p=>{{const c=p.querySelector('.shared-cursor');c.setAttribute('x1',x);c.setAttribute('x2',x);c.style.visibility='visible';}}); const fraction=(x-24.0)/(964.0); const min=model.axis.minimum_offset_ns; const max=model.axis.maximum_offset_ns; const ns=Math.round(min+fraction*(max-min)); readout.value=`cursor: ${{(ns/1e9).toFixed(9)}} s from origin`;}}
  plots.forEach(p=>p.addEventListener('pointermove',move)); zoom.addEventListener('input',applyZoom); document.getElementById('reset').addEventListener('click',()=>{{zoom.value='1';applyZoom();}}); applyZoom();
}})();
</script>
</main></body></html>
"#,
        status_class = html_escape(evidence_status_text, true),
        status_upper = html_escape(&evidence_status_text.to_uppercase(), true),
    ))
}

fn svg_text(x: f64, y: f64, text: &str, attributes: &[(&str, &str)]) -> String {
    let attrs = attributes
        .iter()
        .map(|(key, value)| format!("{}=\"{}\"", key.replace('_', "-"), html_escape(value, true)))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<text x=\"{x:.3}\" y=\"{y:.3}\" {attrs}>{}</text>",
        html_escape(text, true)
    )
}

fn svg_raw_links(value: Option<&Value>, x: f64, y: f64) -> String {
    let artifacts = artifacts(value);
    if artifacts.is_empty() {
        return svg_text(
            x,
            y,
            "raw: none declared",
            &[("fill", "#64748b"), ("font_size", "10")],
        );
    }
    let mut parts = String::new();
    let mut cursor = x;
    for (index, artifact) in artifacts.iter().enumerate() {
        let label = format!(
            "{}{}",
            if index == 0 { "raw: " } else { ", " },
            artifact
                .get("artifact_id")
                .and_then(Value::as_str)
                .unwrap_or_default()
        );
        parts.push_str(&format!(
            "<a href=\"{}\" data-sha256=\"{}\">{}</a>",
            html_escape(
                artifact
                    .get("href")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                true
            ),
            html_escape(
                artifact
                    .get("sha256")
                    .and_then(Value::as_str)
                    .unwrap_or_default(),
                true
            ),
            svg_text(
                cursor,
                y,
                &label,
                &[("fill", "#1d4ed8"), ("font_size", "10")]
            ),
        ));
        cursor += ((label.chars().count() as f64) * 6.2).max(20.0);
    }
    parts
}

fn render_unified_svg_inner(
    view_model: &Value,
    animated: bool,
) -> Result<String, UnifiedViewError> {
    let digest = view_model_sha256(view_model)?;
    let profilers = view_model
        .get("profilers")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let summaries = view_model
        .get("summaries")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let tracks = view_model
        .get("tracks")
        .and_then(Value::as_array)
        .map(Vec::as_slice)
        .unwrap_or(&[]);
    let profiler_rows = profilers.len().div_ceil(3).max(1);
    let profiler_height = profiler_rows * 76;
    let summary_rows = summaries.len().div_ceil(3).max(1);
    let summary_height = summary_rows * 100;
    let profiler_start = 146usize;
    let summary_title_y = profiler_start + profiler_height + 18;
    let summary_start = summary_title_y + 12;
    let track_height = 154usize;
    let track_start = summary_start + summary_height + 32;
    let height = track_start + tracks.len().max(1) * track_height + 48;
    let width = 1200usize;
    let chart_left = 170usize;
    let chart_width = 1000usize;
    let model_json = String::from_utf8(view_model_bytes(view_model)?)
        .map_err(|error| UnifiedViewError::new(error.to_string()))?;
    let model_json = html_escape(model_json.trim_end_matches('\n'), false);
    let run_id = view_model
        .get("run_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let scope_id = view_model
        .pointer("/scope/scope_id")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let clock = view_model
        .pointer("/axis/clock_name")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let evidence_status = view_model
        .pointer("/evidence_status/status")
        .and_then(Value::as_str)
        .unwrap_or_default();
    let mut parts = vec![
        format!("<svg xmlns=\"http://www.w3.org/2000/svg\" xmlns:xlink=\"http://www.w3.org/1999/xlink\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" data-view-model-sha256=\"{digest}\">"),
        format!("<metadata id=\"view-model\" data-sha256=\"{digest}\">{model_json}</metadata>"),
        "<rect width=\"100%\" height=\"100%\" fill=\"#f8fafc\"/>".to_owned(),
        svg_text(24.0, 34.0, "Unified sandbox profile", &[("fill", "#111827"), ("font_size", "22"), ("font_weight", "700")]),
        svg_text(24.0, 58.0, &format!("run {run_id} · scope {scope_id}"), &[("fill", "#475569"), ("font_size", "12")]),
        svg_text(24.0, 78.0, &format!("canonical {clock} · view model sha256 {digest}"), &[("fill", "#64748b"), ("font_size", "10")]),
        svg_text(24.0, 100.0, &format!("Evidence status: {}", evidence_status.to_uppercase()), &[("fill", status_color(evidence_status)), ("font_size", "12"), ("font_weight", "700")]),
        svg_text(24.0, 132.0, "Profiler status", &[("fill", "#111827"), ("font_size", "15"), ("font_weight", "700")]),
    ];
    for (index, profiler) in profilers.iter().enumerate() {
        let column = index % 3;
        let row = index / 3;
        let x = 24 + column * 382;
        let y = profiler_start + row * 76;
        let status = profiler
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let reason = profiler
            .get("reasons")
            .and_then(Value::as_array)
            .and_then(|reasons| reasons.first())
            .and_then(Value::as_str)
            .unwrap_or_default();
        let id = profiler
            .get("profiler_id")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let count = profiler
            .get("stream_ids")
            .and_then(Value::as_array)
            .map(Vec::len)
            .unwrap_or(0);
        let intrusive = if profiler
            .get("intrusive")
            .and_then(Value::as_bool)
            .unwrap_or(false)
        {
            " · intrusive"
        } else {
            ""
        };
        parts.extend([
            format!("<rect x=\"{x}\" y=\"{y}\" width=\"360\" height=\"64\" rx=\"6\" fill=\"#fff\" stroke=\"#cbd5e1\"/>"),
            format!("<rect x=\"{x}\" y=\"{y}\" width=\"5\" height=\"64\" fill=\"{}\"/>", status_color(status)),
            svg_text((x + 14) as f64, (y + 22) as f64, id, &[("fill", "#111827"), ("font_size", "12"), ("font_weight", "700")]),
            svg_text((x + 268) as f64, (y + 22) as f64, &status.to_uppercase(), &[("fill", status_color(status)), ("font_size", "10"), ("font_weight", "700")]),
            svg_text((x + 14) as f64, (y + 41) as f64, &format!("{count} stream(s){intrusive}"), &[("fill", "#64748b"), ("font_size", "9")]),
        ]);
        if !reason.is_empty() {
            parts.push(svg_text(
                (x + 14) as f64,
                (y + 56) as f64,
                reason,
                &[("fill", status_color(status)), ("font_size", "8")],
            ));
        }
    }
    if profilers.is_empty() {
        parts.push(svg_text(
            24.0,
            (profiler_start + 24) as f64,
            "No profiler manifests emitted",
            &[("fill", "#64748b"), ("font_size", "12")],
        ));
    }
    parts.push(svg_text(
        24.0,
        summary_title_y as f64,
        "Untimed summaries",
        &[
            ("fill", "#111827"),
            ("font_size", "15"),
            ("font_weight", "700"),
        ],
    ));
    for (index, summary) in summaries.iter().enumerate() {
        let column = index % 3;
        let row = index / 3;
        let x = 24 + column * 382;
        let y = summary_start + row * 100;
        let status = summary
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let values = summary
            .get("records")
            .and_then(Value::as_array)
            .map(Vec::as_slice)
            .unwrap_or(&[]);
        let value = values
            .first()
            .map(|record| display_text(record.get("value")))
            .unwrap_or_else(|| "no value".to_owned());
        let unit = summary
            .get("unit")
            .and_then(Value::as_str)
            .unwrap_or_default();
        let exact = !values.is_empty()
            && status == "valid"
            && values
                .iter()
                .all(|record| record.pointer("/attributes/exact") == Some(&Value::Bool(true)));
        let timing = if exact {
            "exact untimed fact"
        } else {
            "untimed evidence"
        };
        let reason = summary
            .get("reasons")
            .and_then(Value::as_array)
            .and_then(|reasons| reasons.first())
            .and_then(Value::as_str)
            .unwrap_or_default();
        parts.push(format!("<rect x=\"{x}\" y=\"{y}\" width=\"360\" height=\"88\" rx=\"6\" fill=\"#fff\" stroke=\"#cbd5e1\" stroke-width=\"1\"/>"));
        parts.push(format!(
            "<rect x=\"{x}\" y=\"{y}\" width=\"5\" height=\"88\" fill=\"{}\"/>",
            status_color(status)
        ));
        parts.push(svg_text(
            (x + 14) as f64,
            (y + 21) as f64,
            summary
                .get("label")
                .and_then(Value::as_str)
                .unwrap_or_default(),
            &[
                ("fill", "#111827"),
                ("font_size", "12"),
                ("font_weight", "700"),
            ],
        ));
        parts.push(svg_text(
            (x + 14) as f64,
            (y + 47) as f64,
            &format!("{value} {unit}"),
            &[
                ("fill", "#111827"),
                ("font_size", "17"),
                ("font_weight", "700"),
            ],
        ));
        parts.push(svg_text(
            (x + 250) as f64,
            (y + 21) as f64,
            &status.to_uppercase(),
            &[
                ("fill", status_color(status)),
                ("font_size", "10"),
                ("font_weight", "700"),
            ],
        ));
        parts.push(svg_text(
            (x + 14) as f64,
            (y + 64) as f64,
            &format!("{timing} · not plotted"),
            &[("fill", "#64748b"), ("font_size", "9")],
        ));
        if !reason.is_empty() {
            parts.push(svg_text(
                (x + 14) as f64,
                (y + 80) as f64,
                reason,
                &[("fill", status_color(status)), ("font_size", "8")],
            ));
        }
        parts.push(svg_raw_links(
            summary.get("raw_artifacts"),
            (x + 220) as f64,
            (y + 64) as f64,
        ));
    }
    if summaries.is_empty() {
        parts.push(svg_text(
            24.0,
            (summary_start + 24) as f64,
            "No scalar summaries emitted",
            &[("fill", "#64748b"), ("font_size", "12")],
        ));
    }
    parts.push(svg_text(
        24.0,
        (track_start - 16) as f64,
        "Shared run-relative timeline · native-only evidence retained off-axis",
        &[
            ("fill", "#111827"),
            ("font_size", "15"),
            ("font_weight", "700"),
        ],
    ));
    for (index, track) in tracks.iter().enumerate() {
        let y = track_start + index * track_height;
        let status = track
            .get("status")
            .and_then(Value::as_str)
            .unwrap_or_default();
        parts.extend([
            format!("<g class=\"track\" data-stream-id=\"{}\" transform=\"translate(0,{y})\">", html_escape(track.get("stream_id").and_then(Value::as_str).unwrap_or_default(), true)),
            "<rect x=\"16\" y=\"0\" width=\"1168\" height=\"142\" rx=\"6\" fill=\"#fff\" stroke=\"#cbd5e1\"/>".to_owned(),
            format!("<rect x=\"16\" y=\"0\" width=\"5\" height=\"142\" fill=\"{}\"/>", status_color(status)),
            svg_text(30.0, 22.0, track.get("group").and_then(Value::as_str).unwrap_or_default(), &[("fill", "#64748b"), ("font_size", "10")]),
            svg_text(30.0, 42.0, track.get("label").and_then(Value::as_str).unwrap_or_default(), &[("fill", "#111827"), ("font_size", "12"), ("font_weight", "700")]),
            svg_text(30.0, 59.0, &format!("{} · {}", track.get("shape").and_then(Value::as_str).unwrap_or_default(), track.get("unit").and_then(Value::as_str).unwrap_or_default()), &[("fill", "#475569"), ("font_size", "9")]),
            svg_text(30.0, 76.0, &status.to_uppercase(), &[("fill", status_color(status)), ("font_size", "10"), ("font_weight", "700")]),
            format!("<svg class=\"track-plot\" x=\"{chart_left}\" y=\"8\" width=\"{chart_width}\" height=\"112\" viewBox=\"0 0 {:.0} {:.0}\">{}</svg>", PLOT_WIDTH, PLOT_HEIGHT, track_plot_svg(track, view_model)?),
            svg_raw_links(track.get("raw_artifacts"), 30.0, 126.0),
            "</g>".to_owned(),
        ]);
    }
    if tracks.is_empty() {
        parts.push(svg_text(
            24.0,
            (track_start + 30) as f64,
            "No temporal streams emitted",
            &[("fill", "#64748b"), ("font_size", "12")],
        ));
    }
    if animated && !tracks.is_empty() {
        let animation_top = track_start + 8;
        let animation_bottom = track_start + (tracks.len() - 1) * track_height + 120;
        let x1 = chart_left as f64 + PLOT_LEFT;
        let x2 = chart_left as f64 + PLOT_WIDTH - PLOT_RIGHT;
        let animation_top = animation_top as f64;
        let animation_bottom = animation_bottom as f64;
        parts.push(format!(
            "<line class=\"animated-shared-cursor\" x1=\"{x1:.3}\" y1=\"{animation_top:.3}\" x2=\"{x1:.3}\" y2=\"{animation_bottom:.3}\" stroke=\"#e11d48\" stroke-width=\"2\"><animate attributeName=\"x1\" from=\"{x1:.3}\" to=\"{x2:.3}\" dur=\"8s\" repeatCount=\"indefinite\"/><animate attributeName=\"x2\" from=\"{x1:.3}\" to=\"{x2:.3}\" dur=\"8s\" repeatCount=\"indefinite\"/></line>"
        ));
    }
    parts.push("</svg>".to_owned());
    Ok(parts.join("\n") + "\n")
}

pub fn render_unified_svg(view_model: &Value) -> Result<String, UnifiedViewError> {
    render_unified_svg_inner(view_model, false)
}

pub fn render_unified_animated_svg(view_model: &Value) -> Result<String, UnifiedViewError> {
    render_unified_svg_inner(view_model, true)
}

pub fn write_unified_view_from_bundle(
    bundle_path: &Path,
    output_dir: &Path,
) -> Result<BTreeMap<String, String>, UnifiedViewError> {
    let bytes = fs::read(bundle_path).map_err(|error| {
        UnifiedViewError::new(format!(
            "unable to read evidence bundle {}: {error}",
            bundle_path.display()
        ))
    })?;
    let bundle: Value = serde_json::from_slice(&bytes).map_err(|error| {
        UnifiedViewError::new(format!("unable to decode evidence bundle JSON: {error}"))
    })?;
    write_unified_view_artifacts(&bundle, output_dir)
}

pub fn write_unified_view_artifacts(
    bundle: &Value,
    output_dir: &Path,
) -> Result<BTreeMap<String, String>, UnifiedViewError> {
    fs::create_dir_all(output_dir).map_err(|error| UnifiedViewError::new(error.to_string()))?;
    let paths = [
        ("view_model", output_dir.join("view_model.json")),
        ("html", output_dir.join("unified_profile.html")),
        ("svg", output_dir.join("unified_profile.svg")),
        (
            "animated_svg",
            output_dir.join("unified_profile.animated.svg"),
        ),
    ];
    let existing = paths
        .iter()
        .filter(|(_, path)| path.exists())
        .map(|(_, path)| path.display().to_string())
        .collect::<Vec<_>>();
    if !existing.is_empty() {
        return Err(UnifiedViewError::new(format!(
            "refusing to overwrite unified view artifacts: {}",
            existing.join(", ")
        )));
    }
    let model = assemble_view_model(bundle)?;
    let model_bytes = view_model_bytes(&model)?;
    let html = render_unified_html(&model)?;
    let svg = render_unified_svg(&model)?;
    let animated = render_unified_animated_svg(&model)?;
    for ((_, path), content) in paths.iter().zip([
        model_bytes.as_slice(),
        html.as_bytes(),
        svg.as_bytes(),
        animated.as_bytes(),
    ]) {
        fs::write(path, content).map_err(|error| UnifiedViewError::new(error.to_string()))?;
    }
    let mut result = BTreeMap::new();
    for (name, path) in paths {
        result.insert(name.to_owned(), path.display().to_string());
    }
    result.insert("view_model_sha256".to_owned(), view_model_sha256(&model)?);
    Ok(result)
}

#[derive(Debug)]
struct Cli {
    bundle: PathBuf,
    output_dir: PathBuf,
}

enum CliParse {
    Help,
    Run(Cli),
}

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut positional = Vec::new();
    let mut output_dir = None;
    let mut index = 0;
    let mut options = true;
    while index < arguments.len() {
        let argument = &arguments[index];
        if options && argument == "--" {
            options = false;
        } else if options && (argument == "-h" || argument == "--help") {
            return Ok(CliParse::Help);
        } else if options
            && argument
                .to_str()
                .is_some_and(|argument| argument.len() > 2 && "--output-dir".starts_with(argument))
        {
            index += 1;
            if index == arguments.len() {
                return Err("argument --output-dir: expected one argument".to_owned());
            }
            output_dir = Some(PathBuf::from(&arguments[index]));
        } else if options {
            let lossy = argument.to_string_lossy();
            let abbreviated_output = lossy
                .split_once('=')
                .filter(|(option, _)| option.len() > 2 && "--output-dir".starts_with(option));
            if let Some((_option, value)) = abbreviated_output {
                if value.is_empty() {
                    return Err("argument --output-dir: expected one argument".to_owned());
                }
                output_dir = Some(PathBuf::from(value));
            } else if lossy.starts_with('-') {
                return Err(format!("unrecognized arguments: {lossy}"));
            } else {
                positional.push(PathBuf::from(argument));
            }
        } else {
            positional.push(PathBuf::from(argument));
        }
        index += 1;
    }
    let mut missing = Vec::new();
    if positional.is_empty() {
        missing.push("bundle");
    }
    if output_dir.is_none() {
        missing.push("--output-dir");
    }
    if !missing.is_empty() {
        return Err(format!(
            "the following arguments are required: {}",
            missing.join(", ")
        ));
    }
    if positional.len() > 1 {
        return Err(format!(
            "unrecognized arguments: {}",
            positional[1..]
                .iter()
                .map(|value| value.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    Ok(CliParse::Run(Cli {
        bundle: positional.pop().expect("validated positional"),
        output_dir: output_dir.expect("validated output"),
    }))
}

fn print_usage(public_program: &str, output: &mut dyn Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] --output-dir OUTPUT_DIR bundle"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!("Render one retained typed evidence bundle as synchronized HTML, static SVG,");
    println!("and animated SVG without rerunning the workload.");
    println!();
    println!("positional arguments:");
    println!("  bundle                Typed evidence_bundle.json path.");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --output-dir OUTPUT_DIR");
    println!("                        New or reusable-empty directory for deterministic view");
    println!("                        artifacts.");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
}

pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let options = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(options)) => options,
        Err(error) => return cli_error(public_program, &error),
    };
    let outputs = match write_unified_view_from_bundle(&options.bundle, &options.output_dir) {
        Ok(outputs) => outputs,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    println!("Unified profile HTML: {}", outputs["html"]);
    println!("Unified profile SVG: {}", outputs["svg"]);
    println!("Unified animated SVG: {}", outputs["animated_svg"]);
    println!("View-model SHA-256: {}", outputs["view_model_sha256"]);
    0
}
