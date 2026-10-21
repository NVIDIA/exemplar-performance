// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compare validated wide-allocation ceiling hitters with a smaller baseline.
//!
//! Peak concurrency alone is a weak sizing signal. This analysis retains
//! ceiling duration, baseline comparability, and command attribution for each
//! validated wide-run ceiling hitter.

use serde_json::{json, Map, Value};
use std::cmp::Ordering;
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: &str = "scheduler-ceiling-hitters.v1";

const REQUIRED_COMMAND_COLUMNS: &[&str] = &[
    "task_name",
    "command_family",
    "command_preview",
    "lifecycle_wall_sec",
    "average_simultaneous_logical_cores",
    "peak_simultaneous_logical_cores",
    "time_at_or_above_8_cores_sec",
];

const OUTPUT_FIELDS: &[&str] = &[
    "task_name",
    "wide_average_logical_cores_used",
    "wide_wall_sec",
    "wide_peak_simultaneous_logical_cores",
    "wide_time_at_ceiling_sec",
    "wide_fraction_at_ceiling",
    "wide_time_at_or_above_8_cores_sec",
    "wide_peak_duration_class",
    "baseline_report_available",
    "baseline_average_logical_cores_used",
    "baseline_peak_simultaneous_logical_cores",
    "baseline_time_at_ceiling_sec",
    "baseline_also_reached_ceiling",
    "wide_dominant_high_core_command_family",
    "wide_dominant_high_core_command_preview",
    "wide_dominant_high_core_command_time_at_or_above_8_sec",
    "wide_dominant_high_core_command_peak",
    "baseline_dominant_high_core_command_family",
    "same_dominant_command_family_as_baseline",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CeilingAnalysisError(String);

impl CeilingAnalysisError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CeilingAnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for CeilingAnalysisError {}

#[derive(Clone, Debug, PartialEq)]
pub struct DominantCommand {
    pub command_family: String,
    pub command_preview: String,
    pub lifecycle_wall_sec: f64,
    pub average_cores: f64,
    pub peak_cores: i64,
    pub time_at_or_above_high_core_sec: f64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct CeilingOutputPaths {
    pub json: PathBuf,
    pub csv: PathBuf,
}

type CsvRow = BTreeMap<String, String>;

fn number(value: Option<&Value>, field: &str, path: &Path) -> Result<f64, CeilingAnalysisError> {
    match value {
        Some(Value::Number(value)) => value.as_f64().ok_or_else(|| {
            CeilingAnalysisError::new(format!("{} lacks numeric field {field}", path.display()))
        }),
        _ => Err(CeilingAnalysisError::new(format!(
            "{} lacks numeric field {field}",
            path.display()
        ))),
    }
}

fn load_summary(path: &Path) -> Result<BTreeMap<String, Map<String, Value>>, CeilingAnalysisError> {
    let text = fs::read_to_string(path).map_err(|error| {
        CeilingAnalysisError::new(format!(
            "unable to read summary {}: {error}",
            path.display()
        ))
    })?;
    let payload: Value = serde_json::from_str(&text).map_err(|error| {
        CeilingAnalysisError::new(format!(
            "unable to read summary {}: {error}",
            path.display()
        ))
    })?;
    let tasks = payload
        .as_object()
        .and_then(|payload| payload.get("tasks"))
        .and_then(Value::as_array)
        .ok_or_else(|| {
            CeilingAnalysisError::new(format!("{} has no tasks list", path.display()))
        })?;
    let mut loaded = BTreeMap::new();
    for task in tasks {
        let Some(task) = task.as_object() else {
            return Err(CeilingAnalysisError::new(format!(
                "{} contains an invalid task summary",
                path.display()
            )));
        };
        let Some(task_name) = task.get("task_name").and_then(Value::as_str) else {
            return Err(CeilingAnalysisError::new(format!(
                "{} contains an invalid task summary",
                path.display()
            )));
        };
        loaded.insert(task_name.to_owned(), task.clone());
    }
    Ok(loaded)
}

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

fn load_commands(path: &Path) -> Result<BTreeMap<String, Vec<CsvRow>>, CeilingAnalysisError> {
    let text = fs::read_to_string(path).map_err(|error| {
        CeilingAnalysisError::new(format!(
            "unable to read command CSV {}: {error}",
            path.display()
        ))
    })?;
    let mut records = parse_csv_records(&text).into_iter();
    let headers = records.next().unwrap_or_default();
    let rows: Vec<Vec<String>> = records.collect();
    let header_set: BTreeSet<_> = headers.iter().map(String::as_str).collect();
    if rows.is_empty()
        || !REQUIRED_COMMAND_COLUMNS
            .iter()
            .all(|column| header_set.contains(*column))
    {
        return Err(CeilingAnalysisError::new(format!(
            "{} lacks required command attribution columns",
            path.display()
        )));
    }
    let mut by_task: BTreeMap<String, Vec<CsvRow>> = BTreeMap::new();
    for fields in rows {
        let row: CsvRow = headers
            .iter()
            .enumerate()
            .map(|(index, header)| {
                (
                    header.clone(),
                    fields.get(index).cloned().unwrap_or_default(),
                )
            })
            .collect();
        let task_name = row.get("task_name").cloned().unwrap_or_default();
        by_task.entry(task_name).or_default().push(row);
    }
    Ok(by_task)
}

fn distribution(
    task: &Map<String, Value>,
    path: &Path,
) -> Result<BTreeMap<i64, f64>, CeilingAnalysisError> {
    let raw = task
        .get("time_sec_by_simultaneous_logical_cores")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            let task_name = task
                .get("task_name")
                .map(Value::to_string)
                .unwrap_or_else(|| "None".to_owned());
            CeilingAnalysisError::new(format!(
                "{} task {task_name} lacks distribution",
                path.display()
            ))
        })?;
    let mut result = BTreeMap::new();
    for (cores, seconds) in raw {
        let core_count = cores.trim().parse::<i64>().map_err(|error| {
            CeilingAnalysisError::new(format!(
                "{} has invalid distribution core {cores:?}: {error}",
                path.display()
            ))
        })?;
        result.insert(
            core_count,
            number(Some(seconds), &format!("distribution[{cores}]"), path)?,
        );
    }
    Ok(result)
}

fn parse_command_float(row: &CsvRow, field: &str) -> Result<f64, CeilingAnalysisError> {
    let raw = row.get(field).map(String::as_str).unwrap_or("");
    raw.parse::<f64>().map_err(|error| {
        CeilingAnalysisError::new(format!(
            "invalid command attribution field {field}={raw:?}: {error}"
        ))
    })
}

fn dominant_command(rows: &[CsvRow]) -> Result<Option<DominantCommand>, CeilingAnalysisError> {
    let Some(first) = rows.first() else {
        return Ok(None);
    };
    let mut selected = first;
    let mut selected_key = (
        parse_command_float(first, "time_at_or_above_8_cores_sec")?,
        parse_command_float(first, "average_simultaneous_logical_cores")?,
        parse_command_float(first, "lifecycle_wall_sec")?,
    );
    for candidate in &rows[1..] {
        let key = (
            parse_command_float(candidate, "time_at_or_above_8_cores_sec")?,
            parse_command_float(candidate, "average_simultaneous_logical_cores")?,
            parse_command_float(candidate, "lifecycle_wall_sec")?,
        );
        let ordering = key
            .0
            .partial_cmp(&selected_key.0)
            .unwrap_or(Ordering::Equal)
            .then_with(|| {
                key.1
                    .partial_cmp(&selected_key.1)
                    .unwrap_or(Ordering::Equal)
            })
            .then_with(|| {
                key.2
                    .partial_cmp(&selected_key.2)
                    .unwrap_or(Ordering::Equal)
            });
        if ordering == Ordering::Greater {
            selected = candidate;
            selected_key = key;
        }
    }
    let peak_raw = selected
        .get("peak_simultaneous_logical_cores")
        .map(String::as_str)
        .unwrap_or("");
    let peak_cores = peak_raw.parse::<i64>().map_err(|error| {
        CeilingAnalysisError::new(format!(
            "invalid command attribution field peak_simultaneous_logical_cores={peak_raw:?}: {error}"
        ))
    })?;
    Ok(Some(DominantCommand {
        command_family: selected.get("command_family").cloned().unwrap_or_default(),
        command_preview: selected.get("command_preview").cloned().unwrap_or_default(),
        lifecycle_wall_sec: selected_key.2,
        average_cores: selected_key.1,
        peak_cores,
        time_at_or_above_high_core_sec: selected_key.0,
    }))
}

fn duration_class(average_cores: f64, wall_sec: f64, at_ceiling_sec: f64) -> &'static str {
    if at_ceiling_sec < 0.01 {
        return "sub-10ms ceiling touch";
    }
    if average_cores >= 2.0 {
        return "elevated average multi-core use";
    }
    if at_ceiling_sec / wall_sec < 0.01 {
        return "brief ceiling burst in mostly narrow run";
    }
    "moderate multi-core burst"
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

/// Analyze with the Python command's default 16-wide and 8-baseline ceilings.
#[allow(dead_code)] // Convenience API retained for the future native report library.
pub fn analyze(
    wide_summary_path: &Path,
    baseline_summary_path: &Path,
    wide_commands_path: &Path,
    baseline_commands_path: &Path,
) -> Result<Value, CeilingAnalysisError> {
    analyze_with_ceilings(
        wide_summary_path,
        baseline_summary_path,
        wide_commands_path,
        baseline_commands_path,
        16,
        8,
    )
}

/// Compare wide-run ceiling hitters against completed baseline summaries.
pub fn analyze_with_ceilings(
    wide_summary_path: &Path,
    baseline_summary_path: &Path,
    wide_commands_path: &Path,
    baseline_commands_path: &Path,
    wide_ceiling: i64,
    baseline_ceiling: i64,
) -> Result<Value, CeilingAnalysisError> {
    let wide_tasks = load_summary(wide_summary_path)?;
    let baseline_tasks = load_summary(baseline_summary_path)?;
    let wide_commands = load_commands(wide_commands_path)?;
    let baseline_commands = load_commands(baseline_commands_path)?;
    let mut rows = Vec::new();
    for (task_name, wide) in &wide_tasks {
        let wide_peak = number(
            wide.get("peak_simultaneous_logical_cores"),
            "peak",
            wide_summary_path,
        )?
        .trunc() as i64;
        if wide_peak != wide_ceiling {
            continue;
        }
        let wide_distribution = distribution(wide, wide_summary_path)?;
        let wide_wall = number(wide.get("wall_sec"), "wall_sec", wide_summary_path)?;
        let wide_average = number(
            wide.get("average_logical_cores_used"),
            "average_logical_cores_used",
            wide_summary_path,
        )?;
        let wide_at_ceiling = wide_distribution.get(&wide_ceiling).copied().unwrap_or(0.0);
        let wide_at_or_above_baseline: f64 = wide_distribution
            .iter()
            .filter(|(cores, _)| **cores >= baseline_ceiling)
            .map(|(_, seconds)| *seconds)
            .sum();
        let baseline = baseline_tasks.get(task_name);
        let baseline_distribution = match baseline {
            Some(task) => distribution(task, baseline_summary_path)?,
            None => BTreeMap::new(),
        };
        let baseline_peak = match baseline {
            Some(task) => Some(
                number(
                    task.get("peak_simultaneous_logical_cores"),
                    "peak",
                    baseline_summary_path,
                )?
                .trunc() as i64,
            ),
            None => None,
        };
        let wide_command = dominant_command(
            wide_commands
                .get(task_name)
                .map(Vec::as_slice)
                .unwrap_or_default(),
        )?;
        let baseline_command = dominant_command(
            baseline_commands
                .get(task_name)
                .map(Vec::as_slice)
                .unwrap_or_default(),
        )?;
        let baseline_average = match baseline {
            Some(task) => Some(number(
                task.get("average_logical_cores_used"),
                "average_logical_cores_used",
                baseline_summary_path,
            )?),
            None => None,
        };
        let baseline_time_at_ceiling = baseline.map(|_| {
            baseline_distribution
                .get(&baseline_ceiling)
                .copied()
                .unwrap_or(0.0)
        });
        let baseline_also_reached_ceiling = baseline_peak.map(|peak| peak == baseline_ceiling);
        let same_family = match (&wide_command, &baseline_command) {
            (Some(wide), Some(baseline)) => Some(wide.command_family == baseline.command_family),
            _ => None,
        };
        rows.push(json!({
            "task_name": task_name,
            "wide_average_logical_cores_used": wide_average,
            "wide_wall_sec": wide_wall,
            "wide_peak_simultaneous_logical_cores": wide_peak,
            "wide_time_at_ceiling_sec": wide_at_ceiling,
            "wide_fraction_at_ceiling": if wide_wall != 0.0 {
                wide_at_ceiling / wide_wall
            } else {
                0.0
            },
            "wide_time_at_or_above_8_cores_sec": wide_at_or_above_baseline,
            "wide_peak_duration_class": duration_class(
                wide_average,
                wide_wall,
                wide_at_ceiling,
            ),
            "baseline_report_available": baseline.is_some(),
            "baseline_average_logical_cores_used": baseline_average,
            "baseline_peak_simultaneous_logical_cores": baseline_peak,
            "baseline_time_at_ceiling_sec": baseline_time_at_ceiling,
            "baseline_also_reached_ceiling": baseline_also_reached_ceiling,
            "wide_dominant_high_core_command_family": wide_command
                .as_ref()
                .map(|command| command.command_family.as_str())
                .unwrap_or(""),
            "wide_dominant_high_core_command_preview": wide_command
                .as_ref()
                .map(|command| command.command_preview.as_str())
                .unwrap_or(""),
            "wide_dominant_high_core_command_time_at_or_above_8_sec": wide_command
                .as_ref()
                .map(|command| command.time_at_or_above_high_core_sec),
            "wide_dominant_high_core_command_peak": wide_command
                .as_ref()
                .map(|command| command.peak_cores),
            "baseline_dominant_high_core_command_family": baseline_command
                .as_ref()
                .map(|command| command.command_family.as_str())
                .unwrap_or(""),
            "same_dominant_command_family_as_baseline": same_family,
        }));
    }

    let with_baseline: Vec<_> = rows
        .iter()
        .filter(|row| row["baseline_report_available"].as_bool() == Some(true))
        .collect();
    let also_hit_baseline = with_baseline
        .iter()
        .filter(|row| row["baseline_also_reached_ceiling"].as_bool() == Some(true))
        .count();
    let same_family = with_baseline
        .iter()
        .filter(|row| row["same_dominant_command_family_as_baseline"].as_bool() == Some(true))
        .count();
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "wide_summary": resolved(wide_summary_path),
        "baseline_summary": resolved(baseline_summary_path),
        "wide_ceiling": wide_ceiling,
        "baseline_ceiling": baseline_ceiling,
        "summary": {
            "wide_ceiling_hitter_count": rows.len(),
            "with_comparable_baseline_count": with_baseline.len(),
            "also_hit_baseline_ceiling_count": also_hit_baseline,
            "same_dominant_command_family_count": same_family,
        },
        "tasks": rows,
    }))
}

fn csv_value(value: &Value) -> String {
    match value {
        Value::Null => String::new(),
        Value::Bool(value) => if *value { "True" } else { "False" }.to_owned(),
        Value::String(value) => value.clone(),
        Value::Number(value) => value.to_string(),
        value => value.to_string(),
    }
}

fn csv_field(value: &str) -> String {
    if value.contains(',') || value.contains('"') || value.contains('\r') || value.contains('\n') {
        format!("\"{}\"", value.replace('"', "\"\""))
    } else {
        value.to_owned()
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

/// Write sorted JSON plus a Unix-newline CSV projection.
pub fn write_outputs(
    analysis: &Value,
    output_dir: &Path,
) -> Result<CeilingOutputPaths, CeilingAnalysisError> {
    fs::create_dir_all(output_dir).map_err(|error| {
        CeilingAnalysisError::new(format!(
            "unable to create output directory {}: {error}",
            output_dir.display()
        ))
    })?;
    let json_path = output_dir.join("ceiling_hitters_comparison.json");
    let csv_path = output_dir.join("ceiling_hitters_comparison.csv");
    let serialized = serde_json::to_string_pretty(&sorted_json(analysis)).map_err(|error| {
        CeilingAnalysisError::new(format!("unable to serialize ceiling analysis: {error}"))
    })?;
    let mut serialized = ensure_ascii_json(&serialized);
    serialized.push('\n');
    fs::write(&json_path, serialized).map_err(|error| {
        CeilingAnalysisError::new(format!("unable to write {}: {error}", json_path.display()))
    })?;

    let rows = analysis
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| CeilingAnalysisError::new("ceiling analysis has no tasks list"))?;
    let fields: &[&str] = if rows.is_empty() {
        &["task_name"]
    } else {
        OUTPUT_FIELDS
    };
    let mut csv = String::new();
    csv.push_str(&fields.join(","));
    csv.push('\n');
    for row in rows {
        let row = row
            .as_object()
            .ok_or_else(|| CeilingAnalysisError::new("ceiling analysis task is not an object"))?;
        let values = fields
            .iter()
            .map(|field| csv_field(&csv_value(row.get(*field).unwrap_or(&Value::Null))))
            .collect::<Vec<_>>();
        csv.push_str(&values.join(","));
        csv.push('\n');
    }
    fs::write(&csv_path, csv).map_err(|error| {
        CeilingAnalysisError::new(format!("unable to write {}: {error}", csv_path.display()))
    })?;
    Ok(CeilingOutputPaths {
        json: json_path,
        csv: csv_path,
    })
}

#[derive(Debug)]
struct Cli {
    wide_summary: PathBuf,
    baseline_summary: PathBuf,
    wide_commands: PathBuf,
    baseline_commands: PathBuf,
    output_dir: PathBuf,
    wide_ceiling: i64,
    baseline_ceiling: i64,
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

fn parse_integer(option: &str, value: OsString) -> Result<i64, String> {
    let display = value.to_string_lossy();
    display
        .trim()
        .parse::<i64>()
        .map_err(|_| format!("argument {option}: invalid int value: {display:?}"))
}

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut wide_summary = None;
    let mut baseline_summary = None;
    let mut wide_commands = None;
    let mut baseline_commands = None;
    let mut output_dir = None;
    let mut wide_ceiling = 16;
    let mut baseline_ceiling = 8;
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
            Some("--wide-summary") => {
                wide_summary = Some(PathBuf::from(value_for(&mut index, "--wide-summary")?));
            }
            Some("--baseline-summary") => {
                baseline_summary =
                    Some(PathBuf::from(value_for(&mut index, "--baseline-summary")?));
            }
            Some("--wide-commands") => {
                wide_commands = Some(PathBuf::from(value_for(&mut index, "--wide-commands")?));
            }
            Some("--baseline-commands") => {
                baseline_commands =
                    Some(PathBuf::from(value_for(&mut index, "--baseline-commands")?));
            }
            Some("--output-dir") => {
                output_dir = Some(PathBuf::from(value_for(&mut index, "--output-dir")?));
            }
            Some("--wide-ceiling") => {
                wide_ceiling =
                    parse_integer("--wide-ceiling", value_for(&mut index, "--wide-ceiling")?)?;
            }
            Some("--baseline-ceiling") => {
                baseline_ceiling = parse_integer(
                    "--baseline-ceiling",
                    value_for(&mut index, "--baseline-ceiling")?,
                )?;
            }
            _ => {
                return Err(format!(
                    "unrecognized arguments: {}",
                    argument.to_string_lossy()
                ));
            }
        }
        index += 1;
    }
    let mut missing = Vec::new();
    if wide_summary.is_none() {
        missing.push("--wide-summary");
    }
    if baseline_summary.is_none() {
        missing.push("--baseline-summary");
    }
    if wide_commands.is_none() {
        missing.push("--wide-commands");
    }
    if baseline_commands.is_none() {
        missing.push("--baseline-commands");
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
    Ok(CliParse::Run(Cli {
        wide_summary: wide_summary.expect("validated above"),
        baseline_summary: baseline_summary.expect("validated above"),
        wide_commands: wide_commands.expect("validated above"),
        baseline_commands: baseline_commands.expect("validated above"),
        output_dir: output_dir.expect("validated above"),
        wide_ceiling,
        baseline_ceiling,
    }))
}

fn print_usage(public_program: &str, output: &mut dyn io::Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] --wide-summary WIDE_SUMMARY --baseline-summary BASELINE_SUMMARY --wide-commands WIDE_COMMANDS --baseline-commands BASELINE_COMMANDS --output-dir OUTPUT_DIR [--wide-ceiling WIDE_CEILING] [--baseline-ceiling BASELINE_CEILING]"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!("Compare wide-allocation CPU ceiling hitters.");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --wide-summary WIDE_SUMMARY");
    println!("  --baseline-summary BASELINE_SUMMARY");
    println!("  --wide-commands WIDE_COMMANDS");
    println!("  --baseline-commands BASELINE_COMMANDS");
    println!("  --output-dir OUTPUT_DIR");
    println!("  --wide-ceiling WIDE_CEILING");
    println!("  --baseline-ceiling BASELINE_CEILING");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
}

fn summary_count(analysis: &Value, field: &str) -> u64 {
    analysis["summary"][field].as_u64().unwrap_or(0)
}

/// Native CLI entry point used by the top-level dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let options = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(options)) => options,
        Err(error) => return cli_error(public_program, &error),
    };
    let analysis = match analyze_with_ceilings(
        &options.wide_summary,
        &options.baseline_summary,
        &options.wide_commands,
        &options.baseline_commands,
        options.wide_ceiling,
        options.baseline_ceiling,
    ) {
        Ok(analysis) => analysis,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    let outputs = match write_outputs(&analysis, &options.output_dir) {
        Ok(outputs) => outputs,
        Err(error) => return cli_error(public_program, &error.to_string()),
    };
    println!(
        "Wide-allocation ceiling hitters: {}",
        summary_count(&analysis, "wide_ceiling_hitter_count")
    );
    println!(
        "With comparable baseline: {}",
        summary_count(&analysis, "with_comparable_baseline_count")
    );
    println!(
        "Also reached baseline ceiling: {}",
        summary_count(&analysis, "also_hit_baseline_ceiling_count")
    );
    println!(
        "Same dominant high-core command family as baseline: {}",
        summary_count(&analysis, "same_dominant_command_family_count")
    );
    println!("JSON: {}", outputs.json.display());
    println!("CSV: {}", outputs.csv.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering as AtomicOrdering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, AtomicOrdering::Relaxed);
            let path = env::temp_dir().join(format!(
                "tasktrace-ceiling-hitters-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).unwrap();
            Self(path)
        }

        fn join(&self, path: impl AsRef<Path>) -> PathBuf {
            self.0.join(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_summary(path: &Path, tasks: Value) {
        fs::write(path, serde_json::to_vec(&json!({"tasks": tasks})).unwrap()).unwrap();
    }

    fn write_commands(path: &Path, rows: &[&str]) {
        let mut text = REQUIRED_COMMAND_COLUMNS.join(",");
        text.push('\n');
        for row in rows {
            text.push_str(row);
            text.push('\n');
        }
        fs::write(path, text).unwrap();
    }

    #[test]
    fn analysis_matches_baseline_and_retains_missing_baseline() {
        let directory = TestDirectory::new();
        let wide = directory.join("wide.json");
        write_summary(
            &wide,
            json!([
                {
                    "task_name": "numeric", "wall_sec": 2.0,
                    "average_logical_cores_used": 3.0,
                    "peak_simultaneous_logical_cores": 16,
                    "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "16": 1.0},
                },
                {
                    "task_name": "new-task", "wall_sec": 10.0,
                    "average_logical_cores_used": 1.0,
                    "peak_simultaneous_logical_cores": 16,
                    "time_sec_by_simultaneous_logical_cores": {"1": 9.999, "16": 0.001},
                },
                {
                    "task_name": "narrow", "wall_sec": 1.0,
                    "average_logical_cores_used": 1.0,
                    "peak_simultaneous_logical_cores": 4,
                    "time_sec_by_simultaneous_logical_cores": {"1": 1.0},
                },
            ]),
        );
        let baseline = directory.join("baseline.json");
        write_summary(
            &baseline,
            json!([{
                "task_name": "numeric", "wall_sec": 2.0,
                "average_logical_cores_used": 2.0,
                "peak_simultaneous_logical_cores": 8,
                "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "8": 1.0},
            }]),
        );
        let wide_commands = directory.join("wide.csv");
        write_commands(
            &wide_commands,
            &[
                "numeric,numpy,python compute.py,2.0,3.0,16,1.0",
                "new-task,startup,python smoke.py,10.0,1.0,16,0.001",
            ],
        );
        let baseline_commands = directory.join("baseline.csv");
        write_commands(
            &baseline_commands,
            &["numeric,numpy,python compute.py,2.0,2.0,8,1.0"],
        );

        let analysis = analyze(&wide, &baseline, &wide_commands, &baseline_commands).unwrap();
        assert_eq!(
            analysis["summary"],
            json!({
                "wide_ceiling_hitter_count": 2,
                "with_comparable_baseline_count": 1,
                "also_hit_baseline_ceiling_count": 1,
                "same_dominant_command_family_count": 1,
            })
        );
        let rows = analysis["tasks"].as_array().unwrap();
        let new_task = rows
            .iter()
            .find(|row| row["task_name"] == "new-task")
            .unwrap();
        let numeric = rows
            .iter()
            .find(|row| row["task_name"] == "numeric")
            .unwrap();
        assert_eq!(
            numeric["wide_peak_duration_class"],
            "elevated average multi-core use"
        );
        assert_eq!(numeric["baseline_also_reached_ceiling"], true);
        assert_eq!(
            new_task["wide_peak_duration_class"],
            "sub-10ms ceiling touch"
        );
        assert_eq!(new_task["baseline_report_available"], false);

        let outputs = write_outputs(&analysis, &directory.join("out")).unwrap();
        assert!(outputs.json.is_file());
        assert!(outputs.csv.is_file());
        let csv = fs::read(&outputs.csv).unwrap();
        assert!(!csv.contains(&b'\r'));
        assert!(String::from_utf8(csv)
            .unwrap()
            .starts_with("task_name,wide_average_logical_cores_used,"));
    }

    #[test]
    fn dominant_command_uses_high_core_time_then_average_then_wall() {
        let rows = [
            ("first", "0.5", "3.0", "1.0", "16"),
            ("second", "0.5", "4.0", "0.5", "8"),
            ("third", "0.5", "4.0", "2.0", "8"),
        ];
        let mapped: Vec<CsvRow> = rows
            .iter()
            .map(|(family, at_eight, average, wall, peak)| {
                BTreeMap::from([
                    ("command_family".to_owned(), (*family).to_owned()),
                    ("command_preview".to_owned(), format!("{family} command")),
                    (
                        "time_at_or_above_8_cores_sec".to_owned(),
                        (*at_eight).to_owned(),
                    ),
                    (
                        "average_simultaneous_logical_cores".to_owned(),
                        (*average).to_owned(),
                    ),
                    ("lifecycle_wall_sec".to_owned(), (*wall).to_owned()),
                    (
                        "peak_simultaneous_logical_cores".to_owned(),
                        (*peak).to_owned(),
                    ),
                ])
            })
            .collect();
        let dominant = dominant_command(&mapped).unwrap().unwrap();
        assert_eq!(dominant.command_family, "third");
        assert_eq!(dominant.lifecycle_wall_sec, 2.0);
        assert_eq!(dominant.average_cores, 4.0);
    }

    #[test]
    fn duration_classes_cover_each_sizing_boundary() {
        assert_eq!(duration_class(10.0, 1.0, 0.009), "sub-10ms ceiling touch");
        assert_eq!(
            duration_class(2.0, 1.0, 0.1),
            "elevated average multi-core use"
        );
        assert_eq!(
            duration_class(1.0, 100.0, 0.5),
            "brief ceiling burst in mostly narrow run"
        );
        assert_eq!(duration_class(1.0, 1.0, 0.5), "moderate multi-core burst");
    }

    #[test]
    fn input_validation_is_fail_closed() {
        let directory = TestDirectory::new();
        let invalid_summary = directory.join("invalid.json");
        fs::write(&invalid_summary, "{\"tasks\":[{\"task_name\":1}]}").unwrap();
        assert!(load_summary(&invalid_summary)
            .unwrap_err()
            .to_string()
            .contains("invalid task summary"));

        let commands = directory.join("commands.csv");
        fs::write(&commands, "task_name,command_family\ntask,command\n").unwrap();
        assert!(load_commands(&commands)
            .unwrap_err()
            .to_string()
            .contains("lacks required command attribution columns"));

        let task = json!({"task_name": "missing-distribution"});
        assert!(distribution(task.as_object().unwrap(), &invalid_summary)
            .unwrap_err()
            .to_string()
            .contains("lacks distribution"));
        assert!(number(Some(&Value::Bool(true)), "peak", &invalid_summary).is_err());
    }

    #[test]
    fn csv_writer_quotes_values_and_uses_python_scalar_spelling() {
        let directory = TestDirectory::new();
        let mut task = Map::new();
        for field in OUTPUT_FIELDS {
            task.insert((*field).to_owned(), Value::Null);
        }
        task.insert(
            "task_name".to_owned(),
            Value::String("tâsk,quoted".to_owned()),
        );
        task.insert("baseline_report_available".to_owned(), Value::Bool(true));
        let analysis = json!({"tasks": [task]});
        let outputs = write_outputs(&analysis, &directory.join("out")).unwrap();
        let csv = fs::read_to_string(outputs.csv).unwrap();
        let row = csv.lines().nth(1).unwrap();
        assert!(row.starts_with("\"tâsk,quoted\","));
        assert!(row.contains(",True,"));
        assert!(!csv.contains('\r'));
        let json = fs::read_to_string(outputs.json).unwrap();
        assert!(json.contains("t\\u00e2sk,quoted"));
    }

    #[test]
    fn cli_parser_supports_defaults_overrides_and_required_errors() {
        let parsed = parse_cli(&[
            OsString::from("--wide-summary=wide.json"),
            OsString::from("--baseline-summary"),
            OsString::from("baseline.json"),
            OsString::from("--wide-commands"),
            OsString::from("wide.csv"),
            OsString::from("--baseline-commands=baseline.csv"),
            OsString::from("--output-dir"),
            OsString::from("out"),
            OsString::from("--wide-ceiling=32"),
            OsString::from("--baseline-ceiling"),
            OsString::from("16"),
        ])
        .unwrap();
        let CliParse::Run(parsed) = parsed else {
            panic!("expected run options");
        };
        assert_eq!(parsed.wide_ceiling, 32);
        assert_eq!(parsed.baseline_ceiling, 16);
        assert_eq!(parsed.output_dir, Path::new("out"));
        assert!(parse_cli(&[])
            .unwrap_err()
            .contains("--wide-summary, --baseline-summary"));
        assert!(matches!(
            parse_cli(&[OsString::from("--help")]).unwrap(),
            CliParse::Help
        ));
    }
}
