// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Render named scheduler-concurrency report bundles from a strict manifest.
//!
//! Manifest validation, diagnostic projections, and validated rendering are
//! native. Validated sets invoke the public native `report render-batch` leaf
//! through this same executable so the two command contracts stay aligned.

use serde_json::{json, Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Component, Path, PathBuf};
use std::process::Command;

pub const MANIFEST_SCHEMA_VERSION: &str = "scheduler-report-sets.v1";

const DIAGNOSTIC_FIELDS: &[&str] = &[
    "task_name",
    "source_trial",
    "pipeline_report",
    "classification",
    "source_outcome_mismatch_count",
    "source_outcome_comparison_unavailable_count",
    "observed_window_wall_sec",
    "observed_window_cpu_usage_sec",
    "observed_window_average_logical_cores_used",
    "observed_window_utilization_percent_of_allocation",
    "scheduler_observed_peak_logical_cores",
    "scheduler_peak_interpretation",
    "scheduler_lost_perf_events",
    "scheduler_anomaly_count",
    "scheduler_valid_for_concurrency_reporting",
    "full_replay_average_inference_allowed",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReportSetError(String);

impl ReportSetError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ReportSetError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ReportSetError {}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReportSetConfig {
    pub name: String,
    pub output_dir: PathBuf,
    pub pipeline_reports: Vec<PathBuf>,
    pub diagnostic_pipeline_reports: Vec<PathBuf>,
}

pub type OutputPaths = BTreeMap<String, String>;

fn lexical_normalize(path: &Path) -> PathBuf {
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::CurDir => {}
            Component::ParentDir => {
                if !normalized.pop() && !normalized.has_root() {
                    normalized.push(component.as_os_str());
                }
            }
            component => normalized.push(component.as_os_str()),
        }
    }
    normalized
}

/// Mirror `Path.resolve(strict=False)`: canonicalize the longest existing
/// prefix, then retain any not-yet-created suffix.
fn resolved(path: &Path) -> Result<PathBuf, ReportSetError> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()
            .map_err(|error| {
                ReportSetError::new(format!("unable to resolve current directory: {error}"))
            })?
            .join(path)
    };
    if let Ok(canonical) = fs::canonicalize(&absolute) {
        return Ok(canonical);
    }

    let mut existing = absolute.as_path();
    let mut suffix = Vec::new();
    while !existing.exists() {
        let Some(name) = existing.file_name() else {
            return Ok(lexical_normalize(&absolute));
        };
        suffix.push(name.to_os_string());
        let Some(parent) = existing.parent() else {
            return Ok(lexical_normalize(&absolute));
        };
        existing = parent;
    }
    let mut result = fs::canonicalize(existing).unwrap_or_else(|_| lexical_normalize(existing));
    for component in suffix.iter().rev() {
        result.push(component);
    }
    Ok(lexical_normalize(&result))
}

fn resolve_manifest_path(
    value: Option<&Value>,
    base_dir: &Path,
    field: &str,
) -> Result<PathBuf, ReportSetError> {
    let Some(value) = value
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
    else {
        return Err(ReportSetError::new(format!(
            "{field} must be a non-empty path string"
        )));
    };
    let path = Path::new(value);
    let candidate = if path.is_absolute() {
        path.to_path_buf()
    } else {
        base_dir.join(path)
    };
    resolved(&candidate)
}

fn python_repr(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_repr)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        other => other.to_string(),
    }
}

fn schema_repr(value: Option<&Value>) -> String {
    value.map_or_else(|| "None".to_owned(), python_repr)
}

/// Load a report-set manifest while retaining the validated/diagnostic split.
pub fn load_report_sets(manifest_path: &Path) -> Result<Vec<ReportSetConfig>, ReportSetError> {
    let text = fs::read_to_string(manifest_path).map_err(|error| {
        ReportSetError::new(format!(
            "unable to read report-set manifest {}: {error}",
            manifest_path.display()
        ))
    })?;
    let payload: Value = serde_json::from_str(&text).map_err(|error| {
        ReportSetError::new(format!(
            "unable to read report-set manifest {}: {error}",
            manifest_path.display()
        ))
    })?;
    let root = payload
        .as_object()
        .ok_or_else(|| ReportSetError::new("report-set manifest root must be a JSON object"))?;
    if root.get("schema_version").and_then(Value::as_str) != Some(MANIFEST_SCHEMA_VERSION) {
        return Err(ReportSetError::new(format!(
            "unsupported report-set manifest schema: {}",
            schema_repr(root.get("schema_version"))
        )));
    }
    let reports = root
        .get("reports")
        .and_then(Value::as_object)
        .filter(|reports| !reports.is_empty())
        .ok_or_else(|| {
            ReportSetError::new("report-set manifest must define a non-empty reports object")
        })?;
    let sets = root
        .get("sets")
        .and_then(Value::as_array)
        .filter(|sets| !sets.is_empty())
        .ok_or_else(|| {
            ReportSetError::new("report-set manifest must define a non-empty sets list")
        })?;
    let empty_diagnostics = Map::new();
    let diagnostic_reports = match root.get("diagnostic_reports") {
        None => &empty_diagnostics,
        Some(Value::Object(reports)) => reports,
        Some(_) => {
            return Err(ReportSetError::new(
                "diagnostic_reports must be an object when defined",
            ))
        }
    };

    let manifest = resolved(manifest_path)?;
    let base_dir = manifest
        .parent()
        .ok_or_else(|| ReportSetError::new("report-set manifest has no parent directory"))?;
    let mut validated_paths = BTreeMap::new();
    for (report_id, path) in reports {
        if report_id.is_empty() {
            return Err(ReportSetError::new("report IDs must be non-empty strings"));
        }
        validated_paths.insert(
            report_id.clone(),
            resolve_manifest_path(Some(path), base_dir, &format!("reports['{report_id}']"))?,
        );
    }
    let mut diagnostic_paths = BTreeMap::new();
    for (report_id, path) in diagnostic_reports {
        if report_id.is_empty() {
            return Err(ReportSetError::new(
                "diagnostic report IDs must be non-empty strings",
            ));
        }
        diagnostic_paths.insert(
            report_id.clone(),
            resolve_manifest_path(
                Some(path),
                base_dir,
                &format!("diagnostic_reports['{report_id}']"),
            )?,
        );
    }

    let mut loaded = Vec::new();
    let mut names = BTreeSet::new();
    let mut output_dirs = BTreeSet::new();
    for (index, raw_set) in sets.iter().enumerate() {
        let raw_set = raw_set
            .as_object()
            .ok_or_else(|| ReportSetError::new(format!("sets[{index}] must be an object")))?;
        let name = raw_set
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                ReportSetError::new(format!("sets[{index}].name must be a non-empty string"))
            })?;
        if !names.insert(name.to_owned()) {
            return Err(ReportSetError::new(format!(
                "duplicate report-set name: {name}"
            )));
        }
        let output_dir = resolve_manifest_path(
            raw_set.get("output_dir"),
            base_dir,
            &format!("sets[{index}].output_dir"),
        )?;
        if !output_dirs.insert(output_dir.clone()) {
            return Err(ReportSetError::new(format!(
                "duplicate report-set output directory: {}",
                output_dir.display()
            )));
        }

        let report_ids = raw_set
            .get("report_ids")
            .and_then(Value::as_array)
            .filter(|ids| !ids.is_empty())
            .ok_or_else(|| {
                ReportSetError::new(format!("sets[{index}].report_ids must be a non-empty list"))
            })?;
        let unknown: Vec<_> = report_ids
            .iter()
            .filter(|report_id| {
                report_id
                    .as_str()
                    .map_or(true, |report_id| !validated_paths.contains_key(report_id))
            })
            .cloned()
            .collect();
        if !unknown.is_empty() {
            return Err(ReportSetError::new(format!(
                "report set {} references unknown report IDs: {}",
                python_repr(&Value::String(name.to_owned())),
                python_repr(&Value::Array(unknown))
            )));
        }
        let unique_report_ids: BTreeSet<_> = report_ids.iter().filter_map(Value::as_str).collect();
        if unique_report_ids.len() != report_ids.len() {
            return Err(ReportSetError::new(format!(
                "report set {} repeats a pipeline report",
                python_repr(&Value::String(name.to_owned()))
            )));
        }

        let empty_ids = Vec::new();
        let diagnostic_report_ids = match raw_set.get("diagnostic_report_ids") {
            None => &empty_ids,
            Some(Value::Array(ids)) => ids,
            Some(_) => {
                return Err(ReportSetError::new(format!(
                    "sets[{index}].diagnostic_report_ids must be a list"
                )))
            }
        };
        let unknown_diagnostic: Vec<_> = diagnostic_report_ids
            .iter()
            .filter(|report_id| {
                report_id
                    .as_str()
                    .map_or(true, |report_id| !diagnostic_paths.contains_key(report_id))
            })
            .cloned()
            .collect();
        if !unknown_diagnostic.is_empty() {
            return Err(ReportSetError::new(format!(
                "report set {} references unknown diagnostic report IDs: {}",
                python_repr(&Value::String(name.to_owned())),
                python_repr(&Value::Array(unknown_diagnostic))
            )));
        }
        let unique_diagnostic_ids: BTreeSet<_> = diagnostic_report_ids
            .iter()
            .filter_map(Value::as_str)
            .collect();
        if unique_diagnostic_ids.len() != diagnostic_report_ids.len() {
            return Err(ReportSetError::new(format!(
                "report set {} repeats a diagnostic pipeline report",
                python_repr(&Value::String(name.to_owned()))
            )));
        }

        loaded.push(ReportSetConfig {
            name: name.to_owned(),
            output_dir,
            pipeline_reports: report_ids
                .iter()
                .filter_map(Value::as_str)
                .map(|report_id| validated_paths[report_id].clone())
                .collect(),
            diagnostic_pipeline_reports: diagnostic_report_ids
                .iter()
                .filter_map(Value::as_str)
                .map(|report_id| diagnostic_paths[report_id].clone())
                .collect(),
        });
    }
    Ok(loaded)
}

fn python_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64() != Some(0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn python_string_or_empty(value: Option<&Value>) -> String {
    if !python_truthy(value) {
        return String::new();
    }
    match value.expect("truthy optional JSON value") {
        Value::String(value) => value.clone(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        value => value.to_string(),
    }
}

fn integer_or_zero(value: Option<&Value>, field: &str) -> Result<i64, ReportSetError> {
    if !python_truthy(value) {
        return Ok(0);
    }
    let value = value.expect("truthy optional JSON value");
    match value {
        Value::Bool(value) => Ok(if *value { 1 } else { 0 }),
        Value::Number(value) => value
            .as_i64()
            .or_else(|| value.as_u64().and_then(|value| i64::try_from(value).ok()))
            .or_else(|| value.as_f64().map(|value| value.trunc() as i64))
            .ok_or_else(|| ReportSetError::new(format!("{field} is not an integer"))),
        Value::String(value) => value
            .parse::<i64>()
            .map_err(|_| ReportSetError::new(format!("{field} is not an integer: {value:?}"))),
        _ => Err(ReportSetError::new(format!(
            "{field} is not an integer: {}",
            python_repr(value)
        ))),
    }
}

fn float_or_zero(value: Option<&Value>, field: &str) -> Result<f64, ReportSetError> {
    if !python_truthy(value) {
        return Ok(0.0);
    }
    let value = value.expect("truthy optional JSON value");
    let number = match value {
        Value::Bool(value) => Some(f64::from(u8::from(*value))),
        Value::Number(value) => value.as_f64(),
        Value::String(value) => value.parse::<f64>().ok(),
        _ => None,
    }
    .filter(|value| value.is_finite())
    .ok_or_else(|| {
        ReportSetError::new(format!(
            "{field} is not a finite number: {}",
            python_repr(value)
        ))
    })?;
    Ok(number)
}

fn json_float(value: f64, field: &str) -> Result<Value, ReportSetError> {
    Number::from_f64(value)
        .map(Value::Number)
        .ok_or_else(|| ReportSetError::new(format!("{field} is not a finite number")))
}

fn object_field<'a>(value: Option<&'a Value>, field: &str) -> Option<&'a Value> {
    value
        .and_then(Value::as_object)
        .and_then(|value| value.get(field))
}

fn sorted_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sorted_json).collect()),
        Value::Object(values) => {
            let mut fields: Vec<_> = values.iter().collect();
            fields.sort_by(|(left, _), (right, _)| left.cmp(right));
            let mut sorted = Map::new();
            for (key, value) in fields {
                sorted.insert(key.clone(), sorted_json(value));
            }
            Value::Object(sorted)
        }
        value => value.clone(),
    }
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

fn csv_scalar(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(value)) => if *value { "True" } else { "False" }.to_owned(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::String(value)) => value.clone(),
        Some(value) => value.to_string(),
    }
}

fn csv_quote(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
    }
}

/// Write explicitly non-comparable observed-window diagnostics.  These rows
/// never enter the validated batch renderer.
pub fn write_diagnostic_outputs(
    paths: &[PathBuf],
    output_dir: &Path,
) -> Result<OutputPaths, ReportSetError> {
    let mut rows = Vec::new();
    for path in paths {
        let text = fs::read_to_string(path).map_err(|error| {
            ReportSetError::new(format!(
                "unable to read diagnostic pipeline report {}: {error}",
                path.display()
            ))
        })?;
        let payload: Value = serde_json::from_str(&text).map_err(|error| {
            ReportSetError::new(format!(
                "unable to read diagnostic pipeline report {}: {error}",
                path.display()
            ))
        })?;
        let payload = payload.as_object().ok_or_else(|| {
            ReportSetError::new(format!(
                "diagnostic pipeline report is not an object: {}",
                path.display()
            ))
        })?;
        let replay = payload.get("replay_summary");
        let cpu = payload.get("container_cpu");
        let scheduler = payload.get("scheduler_concurrency");
        let validation = object_field(scheduler, "cgroup_validation");
        let scheduler_valid =
            python_truthy(object_field(validation, "valid_for_concurrency_reporting"));
        let peak = object_field(scheduler, "peak_simultaneous_logical_cores")
            .cloned()
            .unwrap_or(Value::Null);
        let peak_interpretation = if scheduler_valid {
            "exact_for_observed_non_comparable_window"
        } else if !peak.is_null() {
            "observed_lower_bound_only"
        } else {
            "unavailable"
        };
        let anomaly_count = match object_field(scheduler, "anomalies") {
            None | Some(Value::Null) | Some(Value::Bool(false)) => 0,
            Some(Value::Array(values)) => values.len(),
            Some(Value::String(value)) => value.chars().count(),
            Some(value) => {
                return Err(ReportSetError::new(format!(
                    "scheduler_concurrency.anomalies is not iterable: {}",
                    python_repr(value)
                )))
            }
        };
        let mut row = Map::new();
        row.insert(
            "task_name".to_owned(),
            Value::String(python_string_or_empty(payload.get("task_name"))),
        );
        row.insert(
            "source_trial".to_owned(),
            Value::String(python_string_or_empty(payload.get("source_trial"))),
        );
        row.insert(
            "pipeline_report".to_owned(),
            Value::String(resolved(path)?.display().to_string()),
        );
        row.insert(
            "classification".to_owned(),
            Value::String("diagnostic_non_comparable_replay".to_owned()),
        );
        row.insert(
            "source_outcome_mismatch_count".to_owned(),
            Value::Number(
                integer_or_zero(
                    object_field(replay, "source_outcome_mismatch_count"),
                    "replay_summary.source_outcome_mismatch_count",
                )?
                .into(),
            ),
        );
        row.insert(
            "source_outcome_comparison_unavailable_count".to_owned(),
            Value::Number(
                integer_or_zero(
                    object_field(replay, "source_outcome_comparison_unavailable_count"),
                    "replay_summary.source_outcome_comparison_unavailable_count",
                )?
                .into(),
            ),
        );
        for (field, source_field) in [
            ("observed_window_wall_sec", "wall_sec"),
            ("observed_window_cpu_usage_sec", "usage_sec"),
            (
                "observed_window_average_logical_cores_used",
                "average_logical_cores_used",
            ),
            (
                "observed_window_utilization_percent_of_allocation",
                "utilization_percent_of_allocation",
            ),
        ] {
            row.insert(
                field.to_owned(),
                json_float(
                    float_or_zero(
                        object_field(cpu, source_field),
                        &format!("container_cpu.{source_field}"),
                    )?,
                    field,
                )?,
            );
        }
        row.insert("scheduler_observed_peak_logical_cores".to_owned(), peak);
        row.insert(
            "scheduler_peak_interpretation".to_owned(),
            Value::String(peak_interpretation.to_owned()),
        );
        row.insert(
            "scheduler_lost_perf_events".to_owned(),
            Value::Number(
                integer_or_zero(
                    object_field(scheduler, "lost_perf_events"),
                    "scheduler_concurrency.lost_perf_events",
                )?
                .into(),
            ),
        );
        row.insert(
            "scheduler_anomaly_count".to_owned(),
            Value::Number((anomaly_count as u64).into()),
        );
        row.insert(
            "scheduler_valid_for_concurrency_reporting".to_owned(),
            Value::Bool(scheduler_valid),
        );
        row.insert(
            "full_replay_average_inference_allowed".to_owned(),
            Value::Bool(false),
        );
        rows.push(Value::Object(row));
    }
    rows.sort_by(|left, right| {
        left.get("task_name")
            .and_then(Value::as_str)
            .cmp(&right.get("task_name").and_then(Value::as_str))
    });

    fs::create_dir_all(output_dir).map_err(|error| {
        ReportSetError::new(format!(
            "unable to create output directory {}: {error}",
            output_dir.display()
        ))
    })?;
    let json_path = output_dir.join("diagnostic_observations.json");
    let csv_path = output_dir.join("diagnostic_observations.csv");
    let report = json!({
        "schema_version": "scheduler-diagnostic-observations.v1",
        "interpretation": "Observed-window cgroup averages from non-comparable replays. These rows are excluded from validated charts. Scheduler peaks are lower bounds when the scheduler trace is invalid.",
        "tasks": rows,
    });
    let mut encoded = serde_json::to_string_pretty(&sorted_json(&report))
        .map_err(|error| ReportSetError::new(format!("unable to encode diagnostics: {error}")))?;
    encoded = ensure_ascii_json(&encoded);
    encoded.push('\n');
    fs::write(&json_path, encoded).map_err(|error| {
        ReportSetError::new(format!("unable to write {}: {error}", json_path.display()))
    })?;

    let fields: Vec<&str> = if report["tasks"]
        .as_array()
        .is_some_and(|rows| rows.is_empty())
    {
        vec![
            "task_name",
            "classification",
            "observed_window_average_logical_cores_used",
        ]
    } else {
        DIAGNOSTIC_FIELDS.to_vec()
    };
    let mut csv = String::new();
    csv.push_str(&fields.join(","));
    csv.push('\n');
    for row in report["tasks"].as_array().expect("diagnostic tasks array") {
        let values = fields
            .iter()
            .map(|field| csv_quote(&csv_scalar(row.get(*field))))
            .collect::<Vec<_>>();
        csv.push_str(&values.join(","));
        csv.push('\n');
    }
    fs::write(&csv_path, csv).map_err(|error| {
        ReportSetError::new(format!("unable to write {}: {error}", csv_path.display()))
    })?;

    Ok(BTreeMap::from([
        (
            "diagnostic_json".to_owned(),
            resolved(&json_path)?.display().to_string(),
        ),
        (
            "diagnostic_csv".to_owned(),
            resolved(&csv_path)?.display().to_string(),
        ),
    ]))
}

fn delegated_error(stderr: &[u8], status: std::process::ExitStatus) -> String {
    let stderr = String::from_utf8_lossy(stderr);
    for line in stderr.lines().rev() {
        if let Some((_, message)) = line.rsplit_once(": error: ") {
            return message.to_owned();
        }
    }
    let stderr = stderr.trim();
    if stderr.is_empty() {
        format!("report render-batch exited with {status}")
    } else {
        stderr.to_owned()
    }
}

fn render_validated_reports(
    report_set: &ReportSetConfig,
    render_png: bool,
) -> Result<OutputPaths, ReportSetError> {
    let executable = env::current_exe().map_err(|error| {
        ReportSetError::new(format!("unable to locate tasktrace executable: {error}"))
    })?;
    let mut command = Command::new(executable);
    command
        .arg("report")
        .arg("render-batch")
        .args(&report_set.pipeline_reports)
        .arg("--output-dir")
        .arg(&report_set.output_dir);
    if render_png {
        command.arg("--png");
    }
    let output = command.output().map_err(|error| {
        ReportSetError::new(format!("unable to start report render-batch: {error}"))
    })?;
    if !output.status.success() {
        return Err(ReportSetError::new(delegated_error(
            &output.stderr,
            output.status,
        )));
    }

    let path = |name: &str| report_set.output_dir.join(name).display().to_string();
    let mut outputs = BTreeMap::from([
        ("json".to_owned(), path("batch_summary.json")),
        ("csv".to_owned(), path("batch_summary.csv")),
        (
            "average_vs_peak_svg".to_owned(),
            path("average_vs_peak.svg"),
        ),
        (
            "core_distribution_svg".to_owned(),
            path("core_distribution.svg"),
        ),
    ]);
    if render_png {
        outputs.insert("core_timeline_png".to_owned(), path("core_timeline.png"));
        outputs.insert(
            "tool_family_distribution_png".to_owned(),
            path("tool_family_distribution.png"),
        );
    }
    Ok(outputs)
}

fn render_report_sets_with<F>(
    manifest_path: &Path,
    render_png: bool,
    selected_names: &[String],
    mut render_validated: F,
) -> Result<Vec<(String, OutputPaths)>, ReportSetError>
where
    F: FnMut(&ReportSetConfig, bool) -> Result<OutputPaths, ReportSetError>,
{
    let report_sets = load_report_sets(manifest_path)?;
    let available: BTreeSet<_> = report_sets.iter().map(|set| set.name.as_str()).collect();
    let requested: BTreeSet<_> = selected_names.iter().map(String::as_str).collect();
    let missing: Vec<_> = requested.difference(&available).copied().collect();
    if !missing.is_empty() {
        return Err(ReportSetError::new(format!(
            "unknown requested report sets: {}",
            missing.join(", ")
        )));
    }
    let mut rendered = Vec::new();
    for report_set in report_sets {
        if !requested.is_empty() && !requested.contains(report_set.name.as_str()) {
            continue;
        }
        let mut outputs = render_validated(&report_set, render_png)?;
        if !report_set.diagnostic_pipeline_reports.is_empty() {
            outputs.extend(write_diagnostic_outputs(
                &report_set.diagnostic_pipeline_reports,
                &report_set.output_dir,
            )?);
        }
        rendered.push((report_set.name, outputs));
    }
    Ok(rendered)
}

/// Render all selected report sets through the native `report render-batch` leaf.
pub fn render_report_sets(
    manifest_path: &Path,
    render_png: bool,
    selected_names: &[String],
) -> Result<Vec<(String, OutputPaths)>, ReportSetError> {
    render_report_sets_with(
        manifest_path,
        render_png,
        selected_names,
        render_validated_reports,
    )
}

#[derive(Debug, Eq, PartialEq)]
struct Cli {
    manifest: PathBuf,
    selected_names: Vec<String>,
    render_png: bool,
}

#[derive(Debug)]
enum CliParse {
    Help,
    Run(Cli),
}

fn split_long_option(argument: &OsStr) -> Option<(&str, OsString)> {
    let argument = argument.to_str()?;
    let (name, value) = argument.split_once('=')?;
    name.starts_with("--")
        .then(|| (name, OsString::from(value)))
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

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut positional = Vec::new();
    let mut selected_names = Vec::new();
    let mut render_png = false;
    let mut parse_options = true;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if parse_options && matches!(argument.to_str(), Some("-h" | "--help")) {
            return Ok(CliParse::Help);
        }
        if parse_options && argument == "--" {
            parse_options = false;
            index += 1;
            continue;
        }
        let inline = parse_options.then(|| split_long_option(argument)).flatten();
        let option_name = inline
            .as_ref()
            .map(|(name, _)| *name)
            .or_else(|| parse_options.then(|| argument.to_str()).flatten());
        match option_name {
            Some("--set") => {
                let value = inline
                    .map(|(_, value)| value)
                    .map(Ok)
                    .unwrap_or_else(|| option_value(arguments, &mut index, "--set"))?;
                selected_names.push(
                    value
                        .into_string()
                        .map_err(|_| "argument --set: value is not valid UTF-8".to_owned())?,
                );
            }
            Some("--png") if inline.is_none() => render_png = true,
            Some(value) if parse_options && value.starts_with('-') => {
                return Err(format!(
                    "unrecognized arguments: {}",
                    argument.to_string_lossy()
                ));
            }
            _ => positional.push(PathBuf::from(argument)),
        }
        index += 1;
    }
    if positional.is_empty() {
        return Err("the following arguments are required: manifest".to_owned());
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
    Ok(CliParse::Run(Cli {
        manifest: positional.remove(0),
        selected_names,
        render_png,
    }))
}

fn print_usage(public_program: &str, output: &mut dyn Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] [--set SELECTED_NAMES] [--png] manifest"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!("Regenerate scheduler-concurrency report bundles declared in a JSON manifest.");
    println!();
    println!("positional arguments:");
    println!("  manifest              Report-set manifest JSON path.");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --set SELECTED_NAMES  Regenerate only the named set; may be supplied more");
    println!("                        than once.");
    println!("  --png                 Also render native PNG charts.");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
}

/// Native CLI entry point expected by the public dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let cli = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(cli)) => cli,
        Err(error) => return cli_error(public_program, &error),
    };
    let rendered = match render_report_sets(&cli.manifest, cli.render_png, &cli.selected_names) {
        Ok(rendered) => rendered,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    for (name, outputs) in rendered {
        println!("Report set: {name}");
        println!(
            "  Summary JSON: {}",
            outputs
                .get("json")
                .expect("validated renderer returns JSON")
        );
        if cli.render_png {
            println!(
                "  Core timeline PNG: {}",
                outputs
                    .get("core_timeline_png")
                    .expect("PNG renderer returns core timeline")
            );
            println!(
                "  Tool family distribution PNG: {}",
                outputs
                    .get("tool_family_distribution_png")
                    .expect("PNG renderer returns family chart")
            );
        }
        if let Some(path) = outputs.get("diagnostic_csv") {
            println!("  Diagnostic observations CSV: {path}");
        }
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    fn write_manifest(directory: &Path) -> PathBuf {
        let path = directory.join("sets.json");
        fs::write(
            &path,
            serde_json::to_string(&json!({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "reports": {"a": "raw/a.json", "b": "/var/reports/b.json"},
                "diagnostic_reports": {"bad": "raw/bad.json"},
                "sets": [{
                    "name": "all",
                    "output_dir": "outputs/all",
                    "report_ids": ["a", "b"],
                    "diagnostic_report_ids": ["bad"],
                }],
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    #[test]
    fn manifest_paths_are_relative_to_the_resolved_manifest() {
        let directory = TempDir::new().unwrap();
        let manifest = write_manifest(directory.path());
        let sets = load_report_sets(&manifest).unwrap();
        assert_eq!(sets.len(), 1);
        assert_eq!(sets[0].name, "all");
        assert_eq!(
            sets[0].pipeline_reports,
            vec![
                directory.path().join("raw/a.json"),
                PathBuf::from("/var/reports/b.json"),
            ]
        );
        assert_eq!(
            sets[0].diagnostic_pipeline_reports,
            vec![directory.path().join("raw/bad.json")]
        );
        assert_eq!(sets[0].output_dir, directory.path().join("outputs/all"));
    }

    #[test]
    fn manifest_validation_rejects_unknown_and_duplicate_references() {
        let directory = TempDir::new().unwrap();
        let manifest = directory.path().join("sets.json");
        fs::write(
            &manifest,
            serde_json::to_string(&json!({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "reports": {"a": "a.json"},
                "sets": [{"name": "broken", "output_dir": "out", "report_ids": ["missing"]}],
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(load_report_sets(&manifest)
            .unwrap_err()
            .to_string()
            .contains("unknown report IDs"));

        fs::write(
            &manifest,
            serde_json::to_string(&json!({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "reports": {"a": "a.json"},
                "sets": [{"name": "broken", "output_dir": "out", "report_ids": ["a", "a"]}],
            }))
            .unwrap(),
        )
        .unwrap();
        assert!(load_report_sets(&manifest)
            .unwrap_err()
            .to_string()
            .contains("repeats a pipeline report"));
    }

    #[test]
    fn diagnostics_are_sorted_labeled_and_use_unix_csv() {
        let directory = TempDir::new().unwrap();
        let report = directory.path().join("invalid.json");
        fs::write(
            &report,
            serde_json::to_string(&json!({
                "task_name": "caffe-cifar-10",
                "source_trial": "caffe-cifar-10__trial",
                "replay_summary": {"source_outcome_mismatch_count": 7, "source_outcome_comparison_unavailable_count": 2},
                "container_cpu": {
                    "wall_sec": 288.4,
                    "usage_sec": 1183.6,
                    "average_logical_cores_used": 4.104,
                    "utilization_percent_of_allocation": 51.3,
                },
                "scheduler_concurrency": {
                    "peak_simultaneous_logical_cores": 8,
                    "lost_perf_events": 3063,
                    "anomalies": ["lost ordering"],
                    "cgroup_validation": {"valid_for_concurrency_reporting": false},
                },
            }))
            .unwrap(),
        )
        .unwrap();
        let output_dir = directory.path().join("out");
        let outputs = write_diagnostic_outputs(&[report], &output_dir).unwrap();
        let summary: Value =
            serde_json::from_str(&fs::read_to_string(&outputs["diagnostic_json"]).unwrap())
                .unwrap();
        let row = &summary["tasks"][0];
        assert_eq!(row["classification"], "diagnostic_non_comparable_replay");
        assert_eq!(row["source_outcome_comparison_unavailable_count"], 2);
        assert_eq!(row["observed_window_average_logical_cores_used"], 4.104);
        assert_eq!(
            row["scheduler_peak_interpretation"],
            "observed_lower_bound_only"
        );
        assert_eq!(row["full_replay_average_inference_allowed"], false);
        assert!(!fs::read(&outputs["diagnostic_csv"])
            .unwrap()
            .contains(&b'\r'));
    }

    #[test]
    fn selection_preserves_manifest_order_and_deduplicates_requested_names() {
        let directory = TempDir::new().unwrap();
        let manifest = directory.path().join("sets.json");
        fs::write(
            &manifest,
            serde_json::to_string(&json!({
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "reports": {"a": "a.json"},
                "sets": [
                    {"name": "second", "output_dir": "second", "report_ids": ["a"]},
                    {"name": "first", "output_dir": "first", "report_ids": ["a"]},
                ],
            }))
            .unwrap(),
        )
        .unwrap();
        let selected = vec!["first".to_owned(), "first".to_owned()];
        let rendered = render_report_sets_with(&manifest, true, &selected, |set, png| {
            assert!(png);
            Ok(BTreeMap::from([
                (
                    "json".to_owned(),
                    set.output_dir
                        .join("batch_summary.json")
                        .display()
                        .to_string(),
                ),
                ("core_timeline_png".to_owned(), "timeline.png".to_owned()),
                (
                    "tool_family_distribution_png".to_owned(),
                    "families.png".to_owned(),
                ),
            ]))
        })
        .unwrap();
        assert_eq!(rendered.len(), 1);
        assert_eq!(rendered[0].0, "first");

        let error = render_report_sets_with(
            &manifest,
            false,
            &["missing".to_owned()],
            |_, _| unreachable!(),
        )
        .unwrap_err();
        assert_eq!(error.to_string(), "unknown requested report sets: missing");
    }

    #[test]
    fn cli_accepts_repeated_and_inline_set_options() {
        let parsed = parse_cli(&[
            OsString::from("--set=all"),
            OsString::from("--set"),
            OsString::from("fast"),
            OsString::from("--png"),
            OsString::from("sets.json"),
        ])
        .unwrap();
        let CliParse::Run(parsed) = parsed else {
            panic!("expected runnable CLI");
        };
        assert_eq!(parsed.manifest, Path::new("sets.json"));
        assert_eq!(parsed.selected_names, ["all", "fast"]);
        assert!(parsed.render_png);
        assert!(matches!(
            parse_cli(&[OsString::from("--help")]).unwrap(),
            CliParse::Help
        ));
        assert_eq!(
            parse_cli(&[]).unwrap_err(),
            "the following arguments are required: manifest"
        );
    }

    #[cfg(unix)]
    #[test]
    fn delegated_argparse_error_is_reframed_for_render_set() {
        use std::os::unix::process::ExitStatusExt;
        let status = std::process::ExitStatus::from_raw(2 << 8);
        assert_eq!(
            delegated_error(
                b"usage: child ...\nchild: error: scheduler concurrency is not cgroup-validated: x.json\n",
                status,
            ),
            "scheduler concurrency is not cgroup-validated: x.json"
        );
    }
}
