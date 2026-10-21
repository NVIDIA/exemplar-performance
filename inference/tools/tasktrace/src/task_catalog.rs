// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Browser-friendly Terminal-Bench task and CPU evidence catalog export.
//!
//! This module is the native implementation of the public
//! `report task-catalog` leaf. It preserves diagnostic tasks while keeping
//! exact concurrency and reviewed CPU explanations tied to validated evidence.

use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::BTreeMap;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

pub const SCHEMA_VERSION: &str = "tb2-task-catalog.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TaskCatalogError(String);

impl TaskCatalogError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for TaskCatalogError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for TaskCatalogError {}

type CsvRow = BTreeMap<String, String>;

fn read_json(path: &Path) -> Result<Map<String, Value>, TaskCatalogError> {
    let text = fs::read_to_string(path).map_err(|error| {
        TaskCatalogError::new(format!("unable to read JSON {}: {error}", path.display()))
    })?;
    let value: Value = serde_json::from_str(&text).map_err(|error| {
        TaskCatalogError::new(format!("unable to read JSON {}: {error}", path.display()))
    })?;
    value.as_object().cloned().ok_or_else(|| {
        TaskCatalogError::new(format!("JSON source is not an object: {}", path.display()))
    })
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

fn load_commands(path: Option<&Path>) -> Result<BTreeMap<String, Vec<CsvRow>>, TaskCatalogError> {
    let Some(path) = path else {
        return Ok(BTreeMap::new());
    };
    let text = fs::read_to_string(path).map_err(|error| {
        TaskCatalogError::new(format!(
            "unable to read command CSV {}: {error}",
            path.display()
        ))
    })?;
    let mut records = parse_csv_records(&text).into_iter();
    let Some(headers) = records.next() else {
        return Ok(BTreeMap::new());
    };
    let mut by_task: BTreeMap<String, Vec<CsvRow>> = BTreeMap::new();
    for fields in records {
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
        if let Some(task_name) = row.get("task_name").filter(|value| !value.is_empty()) {
            by_task.entry(task_name.clone()).or_default().push(row);
        }
    }
    Ok(by_task)
}

fn instruction_from_pipeline_report(
    report_path: &Path,
) -> Result<(String, String), TaskCatalogError> {
    let report = read_json(report_path)?;
    let Some(source) = report
        .get("source_trajectory")
        .and_then(Value::as_str)
        .filter(|source| !source.is_empty())
    else {
        return Ok((String::new(), String::new()));
    };
    let trajectory_path = PathBuf::from(source);
    let trajectory = read_json(&trajectory_path)?;
    let instruction = trajectory
        .get("steps")
        .and_then(Value::as_array)
        .and_then(|steps| {
            steps.iter().find_map(|step| {
                let step = step.as_object()?;
                if step.get("source").and_then(Value::as_str) != Some("user") {
                    return None;
                }
                let message = step.get("message").and_then(Value::as_str)?;
                (!message.starts_with("<environment_context>")).then(|| message.to_owned())
            })
        })
        .unwrap_or_default();
    Ok((trajectory_path.to_string_lossy().into_owned(), instruction))
}

fn instruction_excerpt(instruction: &str, limit: usize) -> String {
    let paragraph_separator = Regex::new(r"\n\s*\n").expect("valid paragraph regex");
    let whitespace = Regex::new(r"\s+").expect("valid whitespace regex");
    let paragraphs: Vec<String> = paragraph_separator
        .split(instruction)
        .filter(|paragraph| !paragraph.trim().is_empty())
        .map(|paragraph| {
            let paragraph = paragraph.trim().trim_start_matches('#').trim();
            whitespace.replace_all(paragraph, " ").into_owned()
        })
        .collect();
    let Some(first) = paragraphs.first() else {
        return String::new();
    };
    let mut selected = first.clone();
    if (selected.chars().count() < 80 || selected.ends_with(':')) && paragraphs.len() > 1 {
        let separator = if selected.ends_with(':') { " " } else { ": " };
        selected.push_str(separator);
        selected.push_str(&paragraphs[1]);
    }
    if selected.chars().count() > limit {
        selected = selected
            .chars()
            .take(limit.saturating_sub(3))
            .collect::<String>()
            .trim_end()
            .to_owned();
        selected.push_str("...");
    }
    selected
}

fn display_name(task_name: &str) -> String {
    task_name
        .replace('-', " ")
        .split(' ')
        .map(|word| {
            let mut characters = word.chars();
            let Some(first) = characters.next() else {
                return String::new();
            };
            first
                .to_uppercase()
                .chain(characters.flat_map(char::to_lowercase))
                .collect()
        })
        .collect::<Vec<_>>()
        .join(" ")
}

fn row_float(row: &CsvRow, field: &str) -> f64 {
    row.get(field)
        .filter(|value| !value.is_empty())
        .and_then(|value| value.parse::<f64>().ok())
        .unwrap_or(0.0)
}

fn row_int(row: &CsvRow, field: &str) -> i64 {
    row_float(row, field).trunc() as i64
}

#[derive(Clone, Debug)]
struct RankedCommand {
    command_family: String,
    command_preview: String,
    lifecycle_wall_sec: f64,
    average_simultaneous_logical_cores: f64,
    peak_simultaneous_logical_cores: i64,
    time_at_or_above_4_cores_sec: f64,
    time_at_or_above_8_cores_sec: f64,
}

impl RankedCommand {
    fn from_row(row: &CsvRow) -> Self {
        Self {
            command_family: row.get("command_family").cloned().unwrap_or_default(),
            command_preview: row.get("command_preview").cloned().unwrap_or_default(),
            lifecycle_wall_sec: row_float(row, "lifecycle_wall_sec"),
            average_simultaneous_logical_cores: row_float(
                row,
                "average_simultaneous_logical_cores",
            ),
            peak_simultaneous_logical_cores: row_int(row, "peak_simultaneous_logical_cores"),
            time_at_or_above_4_cores_sec: row_float(row, "time_at_or_above_4_cores_sec"),
            time_at_or_above_8_cores_sec: row_float(row, "time_at_or_above_8_cores_sec"),
        }
    }

    fn to_value(&self) -> Value {
        json!({
            "command_family": self.command_family,
            "command_preview": self.command_preview,
            "lifecycle_wall_sec": self.lifecycle_wall_sec,
            "average_simultaneous_logical_cores": self.average_simultaneous_logical_cores,
            "peak_simultaneous_logical_cores": self.peak_simultaneous_logical_cores,
            "time_at_or_above_4_cores_sec": self.time_at_or_above_4_cores_sec,
            "time_at_or_above_8_cores_sec": self.time_at_or_above_8_cores_sec,
        })
    }
}

fn ranked_commands(rows: &[CsvRow]) -> Vec<RankedCommand> {
    let mut ranked: Vec<_> = rows.iter().map(RankedCommand::from_row).collect();
    ranked.sort_by(|left, right| {
        right
            .time_at_or_above_8_cores_sec
            .total_cmp(&left.time_at_or_above_8_cores_sec)
            .then_with(|| {
                right
                    .time_at_or_above_4_cores_sec
                    .total_cmp(&left.time_at_or_above_4_cores_sec)
            })
            .then_with(|| {
                right
                    .average_simultaneous_logical_cores
                    .total_cmp(&left.average_simultaneous_logical_cores)
            })
            .then_with(|| right.lifecycle_wall_sec.total_cmp(&left.lifecycle_wall_sec))
    });
    ranked.truncate(3);
    ranked
}

fn reviewed_cpu_explanation(task_name: &str) -> Option<(&'static str, &'static str)> {
    match task_name {
        "bn-fit-modify" => Some((
            "statistics and Bayesian-network fitting",
            "NumPy/SciPy statistical kernels and Bayesian-network operations create a real short CPU burst, although the full run is approximately single-core.",
        )),
        "break-filter-js-from-html" => Some((
            "browser validation and HTML security testing",
            "The measured ceiling touch occurs during Selenium/Chromium validation and is too short to treat as meaningful 16-core demand.",
        )),
        "build-cython-ext" => Some((
            "native extension build and validation",
            "Cython extension compilation, installation, and import/test verification produce short native build bursts; the overall build remains narrow.",
        )),
        "distribution-search" => Some((
            "numerical optimization",
            "SciPy root solving and NumPy operations over a 150,000-element distribution produce a repeatable, meaningful wide numerical burst.",
        )),
        "largest-eigenval" => Some((
            "linear algebra optimization",
            "Repeated NumPy eigenvalue/eigenvector evaluation during performance validation produces a real native linear-algebra burst.",
        )),
        "model-extraction-relu-logits" => Some((
            "neural-network probing and numerical reconstruction",
            "NumPy-based black-box model queries and verification produce a real numerical burst, but the total task is very short.",
        )),
        "modernize-scientific-stack" => Some((
            "dataframe climate analysis",
            "Pandas/NumPy startup and dataframe computation cause a repeatable scientific library burst, but it is brief.",
        )),
        "multi-source-data-merger" => Some((
            "dataframe and parquet processing",
            "Pandas/PyArrow parsing, merging, and Parquet serialization execute native code and produce meaningful short multi-core activity.",
        )),
        "portfolio-optimization" => Some((
            "native numerical extension and benchmark",
            "Extension compilation and numerical benchmark commands produce a repeatable native burst, not sustained sixteen-core optimizer demand.",
        )),
        "pytorch-model-recovery" => Some((
            "machine-learning tensor computation",
            "PyTorch model loading, output-layer fitting, and verification use CPU tensor kernels and create a material multi-core burst.",
        )),
        "rstan-to-pystan" => Some((
            "probabilistic sampling",
            "PyStan model compilation and posterior sampling create real parallel work; the run is primarily a sustained four-core workload with a brief 16-core touch.",
        )),
        "sanitize-git-repo" => Some((
            "repository secret scanning",
            "The observed burst is concentrated in repository/secret scans such as git grep and lasts only milliseconds, so it is negligible for capacity planning.",
        )),
        "tune-mjcf" => Some((
            "physics simulation tuning",
            "Repeated MuJoCo/NumPy simulation evaluation produces genuine numeric bursts, but peak usage is tiny relative to the long task duration.",
        )),
        "video-processing" => Some((
            "computer vision video analysis",
            "OpenCV video decoding and frame analysis create a repeatable, meaningful multi-core burst while identifying takeoff and landing frames.",
        )),
        _ => None,
    }
}

fn reviewed_mechanism_only_reason(task_name: &str) -> Option<&'static str> {
    match task_name {
        "bn-fit-modify" => Some(
            "CPU work comes from NumPy/SciPy statistics plus Bayesian-network fitting, intervention, and sampling.",
        ),
        "break-filter-js-from-html" => Some(
            "CPU work comes from Selenium/Chromium loading and validating the crafted HTML payload.",
        ),
        "build-cython-ext" => Some(
            "CPU work comes from Cython compilation, native extension linking, installation, and test execution.",
        ),
        "distribution-search" => Some(
            "CPU work comes from SciPy root solving and NumPy calculations over a 150,000-element probability distribution.",
        ),
        "largest-eigenval" => Some(
            "CPU work comes from repeated eigenvalue/eigenvector calculations and benchmark validation.",
        ),
        "model-extraction-relu-logits" => Some(
            "CPU work comes from repeated neural-network queries and NumPy reconstruction checks.",
        ),
        "modernize-scientific-stack" => Some(
            "CPU work comes from pandas CSV processing and NumPy-backed climate aggregation.",
        ),
        "multi-source-data-merger" => Some(
            "CPU work comes from Pandas/PyArrow parsing, record merging, and Parquet serialization.",
        ),
        "portfolio-optimization" => Some(
            "CPU work comes from native extension compilation and numerical risk/return benchmarking.",
        ),
        "pytorch-model-recovery" => Some(
            "CPU work comes from PyTorch model reconstruction, output-layer fitting, and tensor validation.",
        ),
        "rstan-to-pystan" => {
            Some("CPU work comes from PyStan model compilation and posterior sampling.")
        }
        "sanitize-git-repo" => Some(
            "CPU work comes from scanning repository contents for secret values and verifying replacements.",
        ),
        "tune-mjcf" => Some(
            "CPU work comes from repeated MuJoCo physics simulations and numerical correctness evaluation.",
        ),
        "video-processing" => {
            Some("CPU work comes from OpenCV video decoding and per-frame motion analysis.")
        }
        _ => None,
    }
}

fn workload_category(task_name: &str, instruction: &str, command_text: &str) -> &'static str {
    let text = format!("{task_name} {instruction} {command_text}").to_lowercase();
    let patterns: &[(&[&str], &str)] = &[
        (&["stan", "mcmc", "posterior"], "probabilistic sampling"),
        (
            &["torch", "pytorch", "caffe", "cifar", "inference", "model"],
            "machine-learning computation",
        ),
        (
            &["video", "cv2", "image", "cell-seg"],
            "media or image processing",
        ),
        (
            &["cython", "compile", "build", "gcc", "compcert"],
            "native compilation or build",
        ),
        (
            &["numpy", "scipy", "eigen", "portfolio", "bayesian", "raman"],
            "numerical or statistical computation",
        ),
        (&["qemu", "windows 3.11"], "virtualization or emulation"),
        (&["git", "repository", "secret"], "repository processing"),
        (&["sqlite", "database", "wal"], "database processing"),
        (&["nginx", "server", "grpc"], "server or network workload"),
        (
            &["regex", "html", "text", "log", "cobol", "gcode"],
            "text or document processing",
        ),
    ];
    for &(keywords, category) in patterns {
        if keywords.iter().any(|&keyword| text.contains(keyword)) {
            return category;
        }
    }
    "general command-line workload"
}

fn generic_reason(category: &str, dominant: Option<&RankedCommand>, valid: bool) -> String {
    if !valid {
        return format!(
            "The task concerns {category}. CPU observations are diagnostic only because the replay failed validation, so its recorded peak should not be presented as an exact concurrency measurement."
        );
    }
    let Some(dominant) = dominant else {
        return format!(
            "The task concerns {category}; no command-level driver was available in the validated command report."
        );
    };
    let family = if dominant.command_family.is_empty() {
        "an executed command"
    } else {
        &dominant.command_family
    };
    if dominant.time_at_or_above_8_cores_sec >= 0.01 {
        return format!(
            "The task concerns {category}. Measured high-core activity is concentrated in `{family}`, which reached peak {} and spent {:.3} s at eight or more simultaneous logical cores.",
            dominant.peak_simultaneous_logical_cores,
            dominant.time_at_or_above_8_cores_sec,
        );
    }
    format!(
        "The task concerns {category}. The highest-ranked measured command is `{family}` (peak {}), but it did not spend material time at eight or more simultaneous logical cores.",
        dominant.peak_simultaneous_logical_cores,
    )
}

fn mechanism_only_reason(category: &str, dominant: Option<&RankedCommand>) -> String {
    let reason = match category {
        "probabilistic sampling" => {
            "CPU work comes from probabilistic-model compilation and sampling."
        }
        "media or image processing" => {
            "CPU work comes from image/video decoding and pixel-level processing."
        }
        "machine-learning computation" => {
            "CPU work comes from neural-network execution, training, or tensor kernels."
        }
        "native compilation or build" => {
            "CPU work comes from compiling, linking, installing, and validating native software."
        }
        "numerical or statistical computation" => {
            "CPU work comes from numerical array, linear-algebra, or statistical kernels."
        }
        "virtualization or emulation" => {
            "CPU work comes from virtual-machine or emulator execution."
        }
        "repository processing" => {
            "CPU work comes from scanning, comparing, and rewriting repository contents."
        }
        "database processing" => {
            "CPU work comes from database parsing, recovery, query, and verification operations."
        }
        "server or network workload" => {
            "CPU work comes from server execution, request handling, and validation."
        }
        "text or document processing" => {
            "CPU work comes from parsing, transforming, searching, and validating text files."
        }
        _ => "CPU work comes from the programs executed to solve and validate the task.",
    };
    match dominant.filter(|command| !command.command_family.is_empty()) {
        Some(command) => format!(
            "{reason} Primary measured driver: `{}`.",
            command.command_family
        ),
        None => reason.to_owned(),
    }
}

fn value_float(value: Option<&Value>) -> f64 {
    match value {
        Some(Value::Number(number)) => number.as_f64().unwrap_or(0.0),
        Some(Value::String(value)) => value.parse().unwrap_or(0.0),
        Some(Value::Bool(value)) => u8::from(*value) as f64,
        _ => 0.0,
    }
}

fn value_int(value: Option<&Value>) -> i64 {
    value_float(value).trunc() as i64
}

fn python_truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(number)) => number.as_f64().unwrap_or(0.0) != 0.0,
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn python_string(value: Option<&Value>, null_default: &str) -> String {
    match value {
        None => null_default.to_owned(),
        Some(Value::Null) => "None".to_owned(),
        Some(Value::Bool(value)) => if *value { "True" } else { "False" }.to_owned(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Number(value)) => value.to_string(),
        Some(value) => value.to_string(),
    }
}

fn duration_class(
    task: &Map<String, Value>,
    validated_summary: &BTreeMap<String, Map<String, Value>>,
) -> &'static str {
    if task.get("status").and_then(Value::as_str) != Some("passed") {
        return "diagnostic-only observation";
    }
    let task_name = python_string(task.get("task_name"), "None");
    let peak = value_int(task.get("scheduler_peak_simultaneous_logical_cores"));
    let average = value_float(task.get("average_logical_cores_used"));
    let wall = value_float(task.get("wall_sec"));
    let at_ceiling = validated_summary
        .get(&task_name)
        .and_then(|summary| summary.get("time_sec_by_simultaneous_logical_cores"))
        .and_then(Value::as_object)
        .map(|distribution| value_float(distribution.get("16")))
        .unwrap_or(0.0);
    if peak == 16 && at_ceiling < 0.01 {
        return "transient 16-core touch";
    }
    if average >= 2.0 {
        return "elevated average multi-core use";
    }
    if peak >= 8 && wall != 0.0 && at_ceiling / wall < 0.01 {
        return "brief high-core burst in mostly narrow run";
    }
    if peak >= 8 {
        return "measured high-core burst";
    }
    "mostly narrow measured run"
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

fn cloned_or_null(object: &Map<String, Value>, field: &str) -> Value {
    object.get(field).cloned().unwrap_or(Value::Null)
}

/// Join task prompts, validation state, metrics, and display explanations.
pub fn build_catalog(
    analysis_path: &Path,
    validated_summary_path: Option<&Path>,
    commands_path: Option<&Path>,
) -> Result<Value, TaskCatalogError> {
    let analysis = read_json(analysis_path)?;
    let tasks = analysis
        .get("tasks")
        .and_then(Value::as_array)
        .ok_or_else(|| {
            TaskCatalogError::new(format!("{} has no task list", analysis_path.display()))
        })?;

    let mut validated_summary = BTreeMap::new();
    if let Some(path) = validated_summary_path {
        let payload = read_json(path)?;
        if let Some(rows) = payload.get("tasks").and_then(Value::as_array) {
            for row in rows {
                let Some(row) = row.as_object() else {
                    continue;
                };
                let Some(task_name) = row.get("task_name").and_then(Value::as_str) else {
                    continue;
                };
                validated_summary.insert(task_name.to_owned(), row.clone());
            }
        }
    }
    let commands = load_commands(commands_path)?;

    let mut task_objects: Vec<&Map<String, Value>> = tasks
        .iter()
        .map(|task| {
            task.as_object().ok_or_else(|| {
                TaskCatalogError::new(format!(
                    "{} task list contains a non-object entry",
                    analysis_path.display()
                ))
            })
        })
        .collect::<Result<_, _>>()?;
    task_objects.sort_by_key(|task| python_string(task.get("task_name"), ""));

    let mut output_tasks = Vec::new();
    for task in task_objects {
        let task_name = if python_truthy(task.get("task_name")) {
            python_string(task.get("task_name"), "")
        } else {
            String::new()
        };
        let (trajectory_path, instruction) = match task
            .get("pipeline_report")
            .and_then(Value::as_str)
            .filter(|path| !path.is_empty())
        {
            Some(path) => instruction_from_pipeline_report(Path::new(path))?,
            None => (String::new(), String::new()),
        };
        let top_commands = ranked_commands(
            commands
                .get(&task_name)
                .map(Vec::as_slice)
                .unwrap_or_default(),
        );
        let dominant = top_commands.first();
        let command_text = dominant
            .map(|command| command.command_preview.as_str())
            .unwrap_or("");
        let valid = task.get("status").and_then(Value::as_str) == Some("passed");
        let reviewed = reviewed_cpu_explanation(&task_name);
        let excerpt = instruction_excerpt(&instruction, 360);
        let category = reviewed
            .map(|(category, _)| category)
            .unwrap_or_else(|| workload_category(&task_name, &instruction, command_text));
        let reason = match (reviewed, valid) {
            (Some((_, reason)), true) => reason.to_owned(),
            _ => generic_reason(category, dominant, valid),
        };
        let mechanism_reason = reviewed_mechanism_only_reason(&task_name)
            .map(str::to_owned)
            .unwrap_or_else(|| mechanism_only_reason(category, dominant));
        let top_command_values: Vec<_> = top_commands.iter().map(RankedCommand::to_value).collect();
        output_tasks.push(json!({
            "task_name": task_name,
            "display_name": display_name(&task_name),
            "status": cloned_or_null(task, "status"),
            "measurement_quality": cloned_or_null(task, "distribution_quality"),
            "task": {
                "what_it_does": excerpt,
                "instruction_excerpt": excerpt,
                "instruction": instruction,
                "source_trajectory": trajectory_path,
            },
            "cpu": {
                "allocated_logical_cpus": cloned_or_null(task, "allocated_logical_cpus"),
                "wall_sec": cloned_or_null(task, "wall_sec"),
                "cpu_usage_sec": cloned_or_null(task, "cpu_usage_sec"),
                "average_logical_cores_used": cloned_or_null(task, "average_logical_cores_used"),
                "peak_simultaneous_logical_cores": cloned_or_null(task, "scheduler_peak_simultaneous_logical_cores"),
                "exact_concurrency_valid": python_truthy(task.get("included_in_exact_concurrency_distribution")),
                "failure_reasons": task.get("failure_reasons").cloned().unwrap_or_else(|| json!([])),
                "burst_class": duration_class(task, &validated_summary),
                "top_command_drivers": top_command_values,
            },
            "explanation": {
                "workload_category": category,
                "why_it_uses_cores": reason,
                "why_it_uses_cores_mechanism_only": mechanism_reason,
                "method": if reviewed.is_some() && valid {
                    "reviewed_from_prompt_and_command_evidence"
                } else {
                    "rule_based_from_prompt_and_command_evidence"
                },
            },
        }));
    }

    let passed = output_tasks
        .iter()
        .filter(|task| task["status"].as_str() == Some("passed"))
        .count();
    let command_attribution = output_tasks
        .iter()
        .filter(|task| {
            task["cpu"]["top_command_drivers"]
                .as_array()
                .is_some_and(|commands| !commands.is_empty())
        })
        .count();
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "sources": {
            "task_analysis": resolved(analysis_path),
            "validated_summary": validated_summary_path.map(resolved),
            "validated_commands": commands_path.map(resolved),
        },
        "summary": {
            "task_count": output_tasks.len(),
            "validated_task_count": passed,
            "diagnostic_only_task_count": output_tasks.len() - passed,
            "tasks_with_command_attribution": command_attribution,
            "notes": [
                "Exact concurrency and command attribution are intended for passed validated replays.",
                "Failed-validation tasks retain their task instruction and observation-window CPU fields, but their peaks are diagnostic only.",
            ],
        },
        "tasks": output_tasks,
    }))
}

#[derive(Debug)]
struct Cli {
    task_analysis: PathBuf,
    validated_summary: Option<PathBuf>,
    commands: Option<PathBuf>,
    output: PathBuf,
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

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut task_analysis = None;
    let mut validated_summary = None;
    let mut commands = None;
    let mut output = None;
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
            Some("--task-analysis") => {
                task_analysis = Some(PathBuf::from(value_for(&mut index, "--task-analysis")?));
            }
            Some("--validated-summary") => {
                validated_summary =
                    Some(PathBuf::from(value_for(&mut index, "--validated-summary")?));
            }
            Some("--commands") => {
                commands = Some(PathBuf::from(value_for(&mut index, "--commands")?));
            }
            Some("--output") => {
                output = Some(PathBuf::from(value_for(&mut index, "--output")?));
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
    if task_analysis.is_none() {
        missing.push("--task-analysis");
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
    Ok(CliParse::Run(Cli {
        task_analysis: task_analysis.expect("validated above"),
        validated_summary,
        commands,
        output: output.expect("validated above"),
    }))
}

fn print_usage(public_program: &str, output: &mut dyn io::Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] --task-analysis TASK_ANALYSIS [--validated-summary VALIDATED_SUMMARY] [--commands COMMANDS] --output OUTPUT"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!("Export Terminal-Bench task and CPU explanation data.");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --task-analysis TASK_ANALYSIS");
    println!("  --validated-summary VALIDATED_SUMMARY");
    println!("  --commands COMMANDS");
    println!("  --output OUTPUT");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
}

fn summary_count(catalog: &Value, field: &str) -> u64 {
    catalog["summary"][field].as_u64().unwrap_or(0)
}

/// Native CLI entry point expected by the top-level dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let options = match parse_cli(arguments) {
        Ok(CliParse::Help) => {
            print_help(public_program);
            return 0;
        }
        Ok(CliParse::Run(options)) => options,
        Err(error) => return cli_error(public_program, &error),
    };
    let catalog = match build_catalog(
        &options.task_analysis,
        options.validated_summary.as_deref(),
        options.commands.as_deref(),
    ) {
        Ok(catalog) => catalog,
        Err(error) => {
            eprintln!("error: {error}");
            return 2;
        }
    };
    if let Some(parent) = options.output.parent() {
        if let Err(error) = fs::create_dir_all(parent) {
            eprintln!(
                "error: unable to create output directory {}: {error}",
                parent.display()
            );
            return 2;
        }
    }
    let mut serialized = serde_json::to_string_pretty(&catalog)
        .expect("task catalog contains only JSON-serializable values");
    serialized.push('\n');
    if let Err(error) = fs::write(&options.output, serialized) {
        eprintln!(
            "error: unable to write JSON {}: {error}",
            options.output.display()
        );
        return 2;
    }
    println!("Tasks: {}", summary_count(&catalog, "task_count"));
    println!(
        "Validated: {}",
        summary_count(&catalog, "validated_task_count")
    );
    println!(
        "Diagnostic only: {}",
        summary_count(&catalog, "diagnostic_only_task_count")
    );
    println!("JSON: {}", options.output.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = env::temp_dir().join(format!(
                "tasktrace-task-catalog-test-{}-{sequence}",
                std::process::id()
            ));
            fs::create_dir(&path).expect("create test directory");
            Self(path)
        }

        fn join(&self, path: &str) -> PathBuf {
            self.0.join(path)
        }
    }

    impl Drop for TestDirectory {
        fn drop(&mut self) {
            let _ = fs::remove_dir_all(&self.0);
        }
    }

    fn write_json(path: &Path, value: &Value) {
        fs::write(path, serde_json::to_vec(value).unwrap()).unwrap();
    }

    fn trajectory(path: &Path, instruction: &str) {
        write_json(
            path,
            &json!({
                "steps": [
                    {"source": "user", "message": "<environment_context>ignored</environment_context>"},
                    {"source": "user", "message": instruction},
                ]
            }),
        );
    }

    fn pipeline_report(path: &Path, trajectory: &Path) {
        write_json(
            path,
            &json!({"source_trajectory": trajectory.to_string_lossy()}),
        );
    }

    #[test]
    fn builds_catalog_with_prompts_validated_metrics_and_diagnostic_separation() {
        let directory = TestDirectory::new();
        let numeric_trajectory = directory.join("numeric-trajectory.json");
        trajectory(
            &numeric_trajectory,
            "Find the dominant eigenvalue using numpy and validate the numeric answer.",
        );
        let failed_trajectory = directory.join("failed-trajectory.json");
        trajectory(
            &failed_trajectory,
            "Compile a large C project and run its test suite.",
        );
        let numeric_report = directory.join("numeric-report.json");
        pipeline_report(&numeric_report, &numeric_trajectory);
        let failed_report = directory.join("failed-report.json");
        pipeline_report(&failed_report, &failed_trajectory);
        let analysis = directory.join("analysis.json");
        write_json(
            &analysis,
            &json!({
                "tasks": [
                    {
                        "task_name": "largest-eigenval",
                        "status": "passed",
                        "distribution_quality": "validated_replay",
                        "pipeline_report": numeric_report.to_string_lossy(),
                        "allocated_logical_cpus": 16,
                        "wall_sec": 2.0,
                        "cpu_usage_sec": 3.0,
                        "average_logical_cores_used": 1.5,
                        "scheduler_peak_simultaneous_logical_cores": 16,
                        "included_in_exact_concurrency_distribution": true,
                        "failure_reasons": [],
                    },
                    {
                        "task_name": "compile-compcert",
                        "status": "failed_validation",
                        "distribution_quality": "observed_peak_lower_bound_only",
                        "pipeline_report": failed_report.to_string_lossy(),
                        "allocated_logical_cpus": 16,
                        "wall_sec": 30.0,
                        "cpu_usage_sec": 10.0,
                        "average_logical_cores_used": 0.3,
                        "scheduler_peak_simultaneous_logical_cores": 8,
                        "included_in_exact_concurrency_distribution": false,
                        "failure_reasons": ["scheduler_validation_failed"],
                    },
                ]
            }),
        );
        let validated = directory.join("validated.json");
        write_json(
            &validated,
            &json!({
                "tasks": [{
                    "task_name": "largest-eigenval",
                    "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "16": 1.0},
                }]
            }),
        );
        let commands = directory.join("commands.csv");
        fs::write(
            &commands,
            "task_name,command_family,command_preview,lifecycle_wall_sec,average_simultaneous_logical_cores,peak_simultaneous_logical_cores,time_at_or_above_4_cores_sec,time_at_or_above_8_cores_sec\n\
             largest-eigenval,python eval.py,python /app/eval.py,2.0,3.0,16,1.0,0.5\n",
        )
        .unwrap();

        let catalog = build_catalog(&analysis, Some(&validated), Some(&commands)).unwrap();
        assert_eq!(catalog["schema_version"], SCHEMA_VERSION);
        assert_eq!(catalog["summary"]["task_count"], 2);
        assert_eq!(catalog["summary"]["validated_task_count"], 1);
        assert_eq!(catalog["summary"]["diagnostic_only_task_count"], 1);
        assert_eq!(catalog["summary"]["tasks_with_command_attribution"], 1);
        let tasks = catalog["tasks"].as_array().unwrap();
        assert_eq!(tasks[0]["task_name"], "compile-compcert");
        assert_eq!(tasks[1]["task_name"], "largest-eigenval");
        let failed = &tasks[0];
        let numeric = &tasks[1];
        assert!(numeric["task"]["instruction_excerpt"]
            .as_str()
            .unwrap()
            .contains("dominant eigenvalue"));
        assert_eq!(
            numeric["task"]["what_it_does"],
            numeric["task"]["instruction_excerpt"]
        );
        assert_eq!(numeric["cpu"]["burst_class"], "measured high-core burst");
        assert_eq!(
            numeric["explanation"]["method"],
            "reviewed_from_prompt_and_command_evidence"
        );
        assert_eq!(failed["cpu"]["exact_concurrency_valid"], false);
        assert_eq!(failed["cpu"]["burst_class"], "diagnostic-only observation");
        assert!(failed["explanation"]["why_it_uses_cores"]
            .as_str()
            .unwrap()
            .contains("diagnostic only"));
        assert!(!failed["explanation"]["why_it_uses_cores_mechanism_only"]
            .as_str()
            .unwrap()
            .contains("did not spend material time"));
        assert!(numeric["explanation"]["why_it_uses_cores_mechanism_only"]
            .as_str()
            .unwrap()
            .contains("CPU work comes from"));
    }

    #[test]
    fn csv_parser_handles_quotes_and_ranking_prefers_high_core_duration() {
        let rows = parse_csv_records(
            "task_name,command_family,command_preview,lifecycle_wall_sec,average_simultaneous_logical_cores,peak_simultaneous_logical_cores,time_at_or_above_4_cores_sec,time_at_or_above_8_cores_sec\r\n\
             task,slow,slow command,100,1,16,0,0\r\n\
             task,\"python, hot\",\"python -c \"\"print(1, 2)\"\"\",1,2,8,.2,.1\r\n\
             task,invalid,invalid,bad,bad,bad,bad,bad\r\n",
        );
        let headers = &rows[0];
        let mapped: Vec<CsvRow> = rows[1..]
            .iter()
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
        let ranked = ranked_commands(&mapped);
        assert_eq!(ranked.len(), 3);
        assert_eq!(ranked[0].command_family, "python, hot");
        assert_eq!(ranked[0].command_preview, "python -c \"print(1, 2)\"");
        assert_eq!(ranked[0].time_at_or_above_8_cores_sec, 0.1);
        assert_eq!(ranked[2].lifecycle_wall_sec, 0.0);
    }

    #[test]
    fn excerpt_collapses_markdown_paragraphs_and_uses_character_limit() {
        assert_eq!(
            instruction_excerpt("# Heading:\n\n  Second   paragraph\nline", 360),
            "Heading: Second paragraph line"
        );
        assert_eq!(display_name("largest-EIGENVAL"), "Largest Eigenval");
        let excerpt = instruction_excerpt(&"é".repeat(400), 10);
        assert_eq!(excerpt, format!("{}...", "é".repeat(7)));
    }

    #[test]
    fn duration_classes_match_validation_and_ceiling_rules() {
        let mut summary = BTreeMap::new();
        summary.insert(
            "transient".to_owned(),
            json!({"time_sec_by_simultaneous_logical_cores": {"16": 0.001}})
                .as_object()
                .unwrap()
                .clone(),
        );
        let transient = json!({
            "task_name": "transient", "status": "passed",
            "scheduler_peak_simultaneous_logical_cores": 16,
            "average_logical_cores_used": 1.0, "wall_sec": 10.0,
        });
        assert_eq!(
            duration_class(transient.as_object().unwrap(), &summary),
            "transient 16-core touch"
        );
        let elevated = json!({
            "task_name": "elevated", "status": "passed",
            "scheduler_peak_simultaneous_logical_cores": 8,
            "average_logical_cores_used": 2.0, "wall_sec": 10.0,
        });
        assert_eq!(
            duration_class(elevated.as_object().unwrap(), &summary),
            "elevated average multi-core use"
        );
        let failed = json!({"task_name": "failed", "status": "failed_validation"});
        assert_eq!(
            duration_class(failed.as_object().unwrap(), &summary),
            "diagnostic-only observation"
        );
    }

    #[test]
    fn rejects_non_object_json_and_missing_task_list() {
        let directory = TestDirectory::new();
        let array = directory.join("array.json");
        fs::write(&array, "[]").unwrap();
        assert!(read_json(&array)
            .unwrap_err()
            .to_string()
            .contains("not an object"));
        let missing = directory.join("missing.json");
        fs::write(&missing, "{}").unwrap();
        assert!(build_catalog(&missing, None, None)
            .unwrap_err()
            .to_string()
            .contains("has no task list"));
    }

    #[test]
    fn cli_parser_supports_inline_options_and_reports_required_arguments() {
        let parsed = parse_cli(&[
            OsString::from("--task-analysis=analysis.json"),
            OsString::from("--validated-summary"),
            OsString::from("validated.json"),
            OsString::from("--commands=commands.csv"),
            OsString::from("--output"),
            OsString::from("catalog.json"),
        ])
        .unwrap();
        let CliParse::Run(parsed) = parsed else {
            panic!("expected run options");
        };
        assert_eq!(parsed.task_analysis, Path::new("analysis.json"));
        assert_eq!(
            parsed.validated_summary.as_deref(),
            Some(Path::new("validated.json"))
        );
        assert_eq!(parsed.commands.as_deref(), Some(Path::new("commands.csv")));
        assert_eq!(parsed.output, Path::new("catalog.json"));
        assert_eq!(
            parse_cli(&[]).unwrap_err(),
            "the following arguments are required: --task-analysis, --output"
        );
        assert!(matches!(
            parse_cli(&[OsString::from("--help")]).unwrap(),
            CliParse::Help
        ));
    }
}
