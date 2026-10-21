// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Terminal-Bench profile coverage and aggregate analysis.

use regex::Regex;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::UNIX_EPOCH;

pub const SCHEMA_VERSION: &str = "terminal-bench-profile-analysis.v1";
pub const DEFAULT_REPORT_GLOB: &str = "**/pipeline_report.json";
pub const DEFAULT_STARTED_GLOB: &str = "**/container_profile";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnalysisError(String);

impl AnalysisError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for AnalysisError {}

#[derive(Clone, Debug)]
struct ProfileObservation {
    task_name: String,
    report_path: PathBuf,
    wall_sec: f64,
    cpu_usage_sec: f64,
    average_cores: f64,
    allocated_cpus: f64,
    mismatches: i64,
    unavailable_comparisons: u64,
    replay_valid: bool,
    timed_out_actions: i64,
    scheduler_valid: bool,
    scheduler_peak: Option<i64>,
    lost_perf_events: i64,
    scheduler_anomalies: usize,
    distribution: BTreeMap<i64, f64>,
}

impl ProfileObservation {
    fn passed(&self) -> bool {
        self.mismatches == 0
            && self.unavailable_comparisons == 0
            && self.timed_out_actions == 0
            && self.replay_valid
            && self.scheduler_valid
    }

    fn failure_reasons(&self) -> Vec<Value> {
        let mut reasons = Vec::new();
        if self.mismatches != 0 {
            reasons.push(Value::String("source_outcome_mismatch".to_owned()));
        }
        if self.unavailable_comparisons != 0 {
            reasons.push(Value::String(
                "source_outcome_comparison_unavailable".to_owned(),
            ));
        }
        if self.timed_out_actions != 0 {
            reasons.push(Value::String("timed_out_action".to_owned()));
        }
        if !self.replay_valid {
            reasons.push(Value::String("incomplete_replay".to_owned()));
        }
        if !self.scheduler_valid {
            reasons.push(Value::String("scheduler_validation_failed".to_owned()));
        }
        reasons
    }
}

fn number(value: Option<&Value>, field: &str, path: &Path) -> Result<f64, AnalysisError> {
    value.and_then(Value::as_f64).ok_or_else(|| {
        AnalysisError::new(format!("{} lacks numeric field {field}", path.display()))
    })
}

fn truthy(value: Option<&Value>) -> bool {
    match value {
        None | Some(Value::Null) => false,
        Some(Value::Bool(value)) => *value,
        Some(Value::Number(value)) => value.as_f64().map_or(true, |number| number != 0.0),
        Some(Value::String(value)) => !value.is_empty(),
        Some(Value::Array(value)) => !value.is_empty(),
        Some(Value::Object(value)) => !value.is_empty(),
    }
}

fn python_int(value: Option<&Value>, field: &str, path: &Path) -> Result<i64, AnalysisError> {
    let Some(value) = value.filter(|value| truthy(Some(value))) else {
        return Ok(0);
    };
    match value {
        Value::Bool(value) => Ok(i64::from(*value)),
        Value::Number(value) => value
            .as_f64()
            .filter(|number| number.is_finite())
            .map(|number| number.trunc() as i64)
            .ok_or_else(|| {
                AnalysisError::new(format!(
                    "{} has invalid integer field {field}",
                    path.display()
                ))
            }),
        Value::String(value) => value.parse::<i64>().map_err(|_| {
            AnalysisError::new(format!(
                "{} has invalid integer field {field}",
                path.display()
            ))
        }),
        _ => Err(AnalysisError::new(format!(
            "{} has invalid integer field {field}",
            path.display()
        ))),
    }
}

fn python_len(value: Option<&Value>) -> usize {
    match value.filter(|value| truthy(Some(value))) {
        None => 0,
        Some(Value::String(value)) => value.chars().count(),
        Some(Value::Array(value)) => value.len(),
        Some(Value::Object(value)) => value.len(),
        _ => 0,
    }
}

#[derive(Clone)]
enum GlobComponent {
    Recursive,
    Match(Regex),
}

fn component_regex(component: &str) -> Result<Regex, AnalysisError> {
    let characters: Vec<_> = component.chars().collect();
    let mut pattern = String::from("^");
    let mut index = 0;
    while index < characters.len() {
        match characters[index] {
            '*' => pattern.push_str(".*"),
            '?' => pattern.push('.'),
            '[' => {
                let Some(relative_end) = characters[index + 1..]
                    .iter()
                    .position(|character| *character == ']')
                else {
                    pattern.push_str(r"\[");
                    index += 1;
                    continue;
                };
                let end = index + relative_end + 1;
                pattern.push('[');
                let mut inner = index + 1;
                if inner < end && matches!(characters[inner], '!' | '^') {
                    pattern.push('^');
                    inner += 1;
                }
                for character in &characters[inner..end] {
                    if matches!(*character, '\\' | ']') {
                        pattern.push('\\');
                    }
                    pattern.push(*character);
                }
                pattern.push(']');
                index = end;
            }
            character => pattern.push_str(&regex::escape(&character.to_string())),
        }
        index += 1;
    }
    pattern.push('$');
    Regex::new(&pattern).map_err(|error| {
        AnalysisError::new(format!("invalid glob component {component:?}: {error}"))
    })
}

fn compile_glob(pattern: &str) -> Result<Vec<GlobComponent>, AnalysisError> {
    if pattern.is_empty() || Path::new(pattern).is_absolute() {
        return Err(AnalysisError::new(format!(
            "invalid relative glob {pattern:?}"
        )));
    }
    pattern
        .split('/')
        .filter(|component| !component.is_empty())
        .map(|component| {
            if component == "**" {
                Ok(GlobComponent::Recursive)
            } else {
                component_regex(component).map(GlobComponent::Match)
            }
        })
        .collect()
}

fn walk_glob(
    current: &Path,
    components: &[GlobComponent],
    index: usize,
    output: &mut BTreeSet<PathBuf>,
) -> Result<(), AnalysisError> {
    if index == components.len() {
        if current.exists() {
            output.insert(current.to_owned());
        }
        return Ok(());
    }
    match &components[index] {
        GlobComponent::Recursive => {
            walk_glob(current, components, index + 1, output)?;
            let entries = match fs::read_dir(current) {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => {
                    return Err(AnalysisError::new(format!(
                        "unable to traverse {}: {error}",
                        current.display()
                    )))
                }
            };
            for entry in entries {
                let entry = entry.map_err(|error| {
                    AnalysisError::new(format!("unable to traverse {}: {error}", current.display()))
                })?;
                if entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                    walk_glob(&entry.path(), components, index, output)?;
                }
            }
        }
        GlobComponent::Match(pattern) => {
            let entries = match fs::read_dir(current) {
                Ok(entries) => entries,
                Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Ok(()),
                Err(error) => {
                    return Err(AnalysisError::new(format!(
                        "unable to traverse {}: {error}",
                        current.display()
                    )))
                }
            };
            for entry in entries {
                let entry = entry.map_err(|error| {
                    AnalysisError::new(format!("unable to traverse {}: {error}", current.display()))
                })?;
                let name = entry.file_name();
                if !pattern.is_match(&name.to_string_lossy()) {
                    continue;
                }
                if index + 1 == components.len() {
                    output.insert(entry.path());
                } else if entry.file_type().map(|kind| kind.is_dir()).unwrap_or(false) {
                    walk_glob(&entry.path(), components, index + 1, output)?;
                }
            }
        }
    }
    Ok(())
}

fn glob_paths(root: &Path, pattern: &str) -> Result<Vec<PathBuf>, AnalysisError> {
    let components = compile_glob(pattern)?;
    let mut output = BTreeSet::new();
    walk_glob(root, &components, 0, &mut output)?;
    Ok(output.into_iter().collect())
}

fn resolve(path: &Path) -> PathBuf {
    fs::canonicalize(path).unwrap_or_else(|_| {
        if path.is_absolute() {
            path.to_owned()
        } else {
            env::current_dir()
                .unwrap_or_else(|_| PathBuf::from("."))
                .join(path)
        }
    })
}

fn load_inventory(trajectory_root: &Path) -> Result<BTreeSet<String>, AnalysisError> {
    let mut tasks = BTreeSet::new();
    for path in glob_paths(trajectory_root, "*/agent/trajectory.json")? {
        if let Some(dirname) = path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .and_then(|name| name.to_str())
        {
            tasks.insert(
                dirname
                    .split_once("__")
                    .map_or(dirname, |value| value.0)
                    .to_owned(),
            );
        }
    }
    if tasks.is_empty() {
        return Err(AnalysisError::new(format!(
            "no Terminal-Bench trajectories found under {}",
            trajectory_root.display()
        )));
    }
    Ok(tasks)
}

fn task_name(value: Option<&Value>) -> String {
    match value.filter(|value| truthy(Some(value))) {
        Some(Value::String(value)) => value.clone(),
        Some(Value::Bool(value)) => (if *value { "True" } else { "False" }).to_owned(),
        Some(Value::Number(value)) => value.to_string(),
        _ => String::new(),
    }
}

fn load_observation(
    path: &Path,
    inventory: &BTreeSet<String>,
) -> Result<Option<ProfileObservation>, AnalysisError> {
    let text = fs::read_to_string(path).map_err(|error| {
        AnalysisError::new(format!(
            "unable to read pipeline report {}: {error}",
            path.display()
        ))
    })?;
    let payload: Value = serde_json::from_str(&text).map_err(|error| {
        AnalysisError::new(format!(
            "unable to read pipeline report {}: {error}",
            path.display()
        ))
    })?;
    let payload = payload.as_object().ok_or_else(|| {
        AnalysisError::new(format!(
            "pipeline report must be an object: {}",
            path.display()
        ))
    })?;
    let task_name = task_name(payload.get("task_name"));
    if !inventory.contains(&task_name) {
        return Ok(None);
    }
    let replay = payload.get("replay_summary").and_then(Value::as_object);
    let cpu = payload.get("container_cpu").and_then(Value::as_object);
    let scheduler = payload
        .get("scheduler_concurrency")
        .and_then(Value::as_object);
    let (Some(replay), Some(cpu), Some(scheduler)) = (replay, cpu, scheduler) else {
        return Err(AnalysisError::new(format!(
            "pipeline report lacks replay, CPU, or scheduler data: {}",
            path.display()
        )));
    };
    let validation = scheduler
        .get("cgroup_validation")
        .filter(|value| truthy(Some(value)))
        .and_then(Value::as_object);
    let mut distribution = BTreeMap::new();
    if let Some(raw) = scheduler
        .get("time_sec_by_simultaneous_logical_cores")
        .filter(|value| truthy(Some(value)))
        .and_then(Value::as_object)
    {
        for (cores, seconds) in raw {
            let cores = cores.parse::<i64>().map_err(|error| {
                AnalysisError::new(format!(
                    "{} has invalid scheduler distribution core {cores:?}: {error}",
                    path.display()
                ))
            })?;
            distribution.insert(
                cores,
                number(
                    Some(seconds),
                    &format!("scheduler distribution[{cores}]"),
                    path,
                )?,
            );
        }
    }
    let peak = scheduler.get("peak_simultaneous_logical_cores");
    let scheduler_peak = match peak {
        Some(Value::Bool(value)) => Some(i64::from(*value)),
        Some(Value::Number(value)) => value.as_f64().map(|number| number.trunc() as i64),
        _ => None,
    };
    Ok(Some(ProfileObservation {
        task_name,
        report_path: resolve(path),
        wall_sec: number(cpu.get("wall_sec"), "container_cpu.wall_sec", path)?,
        cpu_usage_sec: number(cpu.get("usage_sec"), "container_cpu.usage_sec", path)?,
        average_cores: number(
            cpu.get("average_logical_cores_used"),
            "container_cpu.average_logical_cores_used",
            path,
        )?,
        allocated_cpus: number(
            cpu.get("allocated_logical_cpus"),
            "container_cpu.allocated_logical_cpus",
            path,
        )?,
        mismatches: python_int(
            replay.get("source_outcome_mismatch_count"),
            "replay_summary.source_outcome_mismatch_count",
            path,
        )?,
        unavailable_comparisons: replay
            .get("source_outcome_comparison_unavailable_count")
            .map_or(Ok(0), |value| {
                value.as_u64().ok_or_else(|| AnalysisError::new(format!(
                    "{} has invalid unsigned integer field replay_summary.source_outcome_comparison_unavailable_count",
                    path.display()
                )))
            })?,
        replay_valid: payload
            .get("status")
            .map_or(true, |status| status.as_str() == Some("valid"))
            && [
                "actions_launch_failed",
                "actions_source_argument_parse_failed",
                "actions_skipped_after_timeout",
            ]
            .iter()
            .all(|field| {
                replay
                    .get(*field)
                    .map_or(true, |value| value.as_u64() == Some(0))
            }),
        timed_out_actions: python_int(
            replay.get("actions_timed_out"),
            "replay_summary.actions_timed_out",
            path,
        )?,
        scheduler_valid: truthy(
            validation.and_then(|value| value.get("valid_for_concurrency_reporting")),
        ),
        scheduler_peak,
        lost_perf_events: python_int(
            scheduler.get("lost_perf_events"),
            "scheduler_concurrency.lost_perf_events",
            path,
        )?,
        scheduler_anomalies: python_len(scheduler.get("anomalies")),
        distribution,
    }))
}

fn task_from_started_dir(path: &Path, inventory: &BTreeSet<String>) -> Option<String> {
    let dirname = path.parent()?.file_name()?.to_str()?;
    inventory
        .iter()
        .filter(|task| dirname == task.as_str() || dirname.starts_with(&format!("{task}-")))
        .max_by_key(|task| task.len())
        .cloned()
}

fn modified_ns(path: &Path) -> u128 {
    fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .ok()
        .and_then(|modified| modified.duration_since(UNIX_EPOCH).ok())
        .map_or(0, |duration| duration.as_nanos())
}

fn choose_observation(observations: &[ProfileObservation]) -> ProfileObservation {
    let passed: Vec<_> = observations
        .iter()
        .filter(|observation| observation.passed())
        .collect();
    let candidates: Vec<_> = if passed.is_empty() {
        observations.iter().collect()
    } else {
        passed
    };
    candidates
        .into_iter()
        .max_by_key(|observation| modified_ns(&observation.report_path))
        .unwrap()
        .clone()
}

pub fn analyze(
    trajectory_root: &Path,
    reports_root: &Path,
    include_failed: bool,
    report_glob: &str,
    started_glob: &str,
) -> Result<Value, AnalysisError> {
    let inventory = load_inventory(trajectory_root)?;
    let mut observations_by_task: BTreeMap<String, Vec<ProfileObservation>> = BTreeMap::new();
    for path in glob_paths(reports_root, report_glob)? {
        if let Some(observation) = load_observation(&path, &inventory)? {
            observations_by_task
                .entry(observation.task_name.clone())
                .or_default()
                .push(observation);
        }
    }
    let started_tasks: BTreeSet<_> = glob_paths(reports_root, started_glob)?
        .iter()
        .filter_map(|path| task_from_started_dir(path, &inventory))
        .collect();
    let selected: BTreeMap<_, _> = observations_by_task
        .iter()
        .map(|(task, observations)| (task.clone(), choose_observation(observations)))
        .collect();

    let mut task_rows = Vec::new();
    let mut exact_distribution: BTreeMap<i64, f64> = BTreeMap::new();
    let mut included = Vec::new();
    let mut exact_distribution_tasks = 0;
    for task in &inventory {
        let Some(observation) = selected.get(task) else {
            task_rows.push(json!({
                "task_name": task,
                "status": if started_tasks.contains(task) { "incomplete" } else { "not_profiled" },
                "attempt_count": 0,
                "included_in_metrics": false,
                "included_in_exact_concurrency_distribution": false,
            }));
            continue;
        };
        let passed = observation.passed();
        let is_included = passed || include_failed;
        let include_distribution = is_included && observation.scheduler_valid;
        if is_included {
            included.push(observation);
        }
        if include_distribution {
            exact_distribution_tasks += 1;
            for (cores, seconds) in &observation.distribution {
                *exact_distribution.entry(*cores).or_default() += seconds;
            }
        }
        let distribution_quality = if observation.scheduler_valid {
            if passed {
                "validated_replay"
            } else {
                "exact_observed_window_non_comparable"
            }
        } else {
            "observed_peak_lower_bound_only"
        };
        task_rows.push(json!({
            "task_name": task,
            "status": if passed { "passed" } else { "failed_validation" },
            "attempt_count": observations_by_task[task].len(),
            "pipeline_report": observation.report_path.to_string_lossy(),
            "failure_reasons": observation.failure_reasons(),
            "timed_out_actions": observation.timed_out_actions,
            "source_outcome_mismatch_count": observation.mismatches,
            "source_outcome_comparison_unavailable_count": observation.unavailable_comparisons,
            "wall_sec": observation.wall_sec,
            "cpu_usage_sec": observation.cpu_usage_sec,
            "average_logical_cores_used": observation.average_cores,
            "allocated_logical_cpus": observation.allocated_cpus,
            "scheduler_peak_simultaneous_logical_cores": observation.scheduler_peak,
            "scheduler_lost_perf_events": observation.lost_perf_events,
            "scheduler_anomaly_count": observation.scheduler_anomalies,
            "scheduler_valid_for_concurrency_reporting": observation.scheduler_valid,
            "distribution_quality": distribution_quality,
            "included_in_metrics": is_included,
            "included_in_exact_concurrency_distribution": include_distribution,
        }));
    }
    let count = |status: &str| {
        task_rows
            .iter()
            .filter(|row| row["status"] == Value::String(status.to_owned()))
            .count()
    };
    let passed = count("passed");
    let failed = count("failed_validation");
    let incomplete = count("incomplete");
    let not_profiled = count("not_profiled");
    let timed_out_failed = task_rows
        .iter()
        .filter(|row| {
            row["status"] == Value::String("failed_validation".to_owned())
                && row
                    .get("timed_out_actions")
                    .and_then(Value::as_i64)
                    .unwrap_or(0)
                    > 0
        })
        .count();
    let total_wall: f64 = included
        .iter()
        .map(|observation| observation.wall_sec)
        .sum();
    let total_cpu: f64 = included
        .iter()
        .map(|observation| observation.cpu_usage_sec)
        .sum();
    let distribution_total: f64 = exact_distribution.values().sum();
    let distribution_rows: Vec<_> = exact_distribution
        .iter()
        .map(|(cores, seconds)| {
            json!({
                "simultaneous_logical_cores": cores,
                "observed_time_sec": seconds,
                "fraction_of_exact_distribution_window": if distribution_total != 0.0 { seconds / distribution_total } else { 0.0 },
            })
        })
        .collect();
    Ok(json!({
        "schema_version": SCHEMA_VERSION,
        "trajectory_root": resolve(trajectory_root).to_string_lossy(),
        "reports_root": resolve(reports_root).to_string_lossy(),
        "report_glob": report_glob,
        "include_failed": include_failed,
        "selection_rule": "One row per Terminal-Bench task; prefer any passed report, otherwise select the most recent completed failed-validation report.",
        "counts": {
            "passed": passed,
            "failed_validation": failed,
            "incomplete": incomplete,
            "not_profiled": not_profiled,
            "total_terminal_bench_tasks": inventory.len(),
            "profiled_with_completed_report": passed + failed,
            "failed_validation_with_timed_out_actions": timed_out_failed,
        },
        "metrics": {
            "included_task_count": included.len(),
            "observed_window_total_wall_sec": total_wall,
            "observed_window_total_cpu_usage_sec": total_cpu,
            "observed_window_weighted_average_logical_cores_used": if total_wall != 0.0 { total_cpu / total_wall } else { 0.0 },
            "exact_concurrency_distribution_task_count": exact_distribution_tasks,
            "exact_concurrency_distribution_excludes_invalid_scheduler_traces": true,
        },
        "concurrency_distribution": distribution_rows,
        "tasks": task_rows,
    }))
}

fn py_list(value: &[Value]) -> String {
    format!(
        "[{}]",
        value
            .iter()
            .map(|item| match item {
                Value::String(value) =>
                    format!("'{}'", value.replace('\\', "\\\\").replace('\'', "\\'")),
                other => csv_scalar(Some(other)),
            })
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn csv_scalar(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => String::new(),
        Some(Value::Bool(value)) => (if *value { "True" } else { "False" }).to_owned(),
        Some(Value::Number(value)) => value.to_string(),
        Some(Value::String(value)) => value.clone(),
        Some(Value::Array(value)) => py_list(value),
        Some(Value::Object(_)) => value.unwrap().to_string(),
    }
}

fn csv_quote(value: &str) -> String {
    if value.contains([',', '"', '\r', '\n']) {
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

pub fn write_outputs(
    analysis: &Value,
    output_dir: &Path,
) -> Result<BTreeMap<&'static str, PathBuf>, AnalysisError> {
    fs::create_dir_all(output_dir).map_err(|error| {
        AnalysisError::new(format!(
            "unable to create {}: {error}",
            output_dir.display()
        ))
    })?;
    let json_path = output_dir.join("tb2_profile_analysis.json");
    let task_csv_path = output_dir.join("tb2_task_status.csv");
    let distribution_csv_path = output_dir.join("tb2_concurrency_distribution.csv");
    crate::process_bpf_reports::write_process_inventory_report(analysis, &json_path)
        .map_err(|error| AnalysisError::new(error.to_string()))?;
    let tasks = analysis["tasks"]
        .as_array()
        .ok_or_else(|| AnalysisError::new("analysis has no tasks list"))?;
    let fields: BTreeSet<_> = tasks
        .iter()
        .filter_map(Value::as_object)
        .flat_map(|row| row.keys().cloned())
        .collect();
    let mut task_csv = csv_line(fields.iter().map(String::as_str));
    for row in tasks {
        let row = row.as_object().unwrap();
        let values: Vec<_> = fields
            .iter()
            .map(|field| csv_scalar(row.get(field)))
            .collect();
        task_csv.push_str(&csv_line(values.iter().map(String::as_str)));
    }
    fs::write(&task_csv_path, task_csv).map_err(|error| {
        AnalysisError::new(format!(
            "unable to write {}: {error}",
            task_csv_path.display()
        ))
    })?;
    let distribution_fields = [
        "simultaneous_logical_cores",
        "observed_time_sec",
        "fraction_of_exact_distribution_window",
    ];
    let mut distribution_csv = csv_line(distribution_fields);
    for row in analysis["concurrency_distribution"].as_array().unwrap() {
        let values: Vec<_> = distribution_fields
            .iter()
            .map(|field| csv_scalar(row.get(field)))
            .collect();
        distribution_csv.push_str(&csv_line(values.iter().map(String::as_str)));
    }
    fs::write(&distribution_csv_path, distribution_csv).map_err(|error| {
        AnalysisError::new(format!(
            "unable to write {}: {error}",
            distribution_csv_path.display()
        ))
    })?;
    Ok(BTreeMap::from([
        ("json", json_path),
        ("task_csv", task_csv_path),
        ("distribution_csv", distribution_csv_path),
    ]))
}

#[derive(Debug)]
struct Cli {
    trajectory_root: PathBuf,
    reports_root: PathBuf,
    output_dir: PathBuf,
    include_failed: bool,
    report_glob: String,
    started_glob: String,
}

fn parse_cli(arguments: &[OsString]) -> Result<Cli, AnalysisError> {
    let mut trajectory_root = None;
    let mut reports_root = None;
    let mut output_dir = None;
    let mut include_failed = false;
    let mut report_glob = DEFAULT_REPORT_GLOB.to_owned();
    let mut started_glob = DEFAULT_STARTED_GLOB.to_owned();
    let mut index = 0;
    while index < arguments.len() {
        let argument = arguments[index].to_string_lossy();
        if argument == "--include-failed" {
            include_failed = true;
            index += 1;
            continue;
        }
        let (name, inline) = argument
            .split_once('=')
            .map_or((argument.as_ref(), None), |(name, value)| {
                (name, Some(value))
            });
        let target_path = match name {
            "--trajectory-root" => Some(&mut trajectory_root),
            "--reports-root" => Some(&mut reports_root),
            "--output-dir" => Some(&mut output_dir),
            _ => None,
        };
        if let Some(target) = target_path {
            let value = match inline {
                Some(value) if !value.is_empty() => OsString::from(value),
                Some(_) => {
                    return Err(AnalysisError::new(format!(
                        "argument {name}: expected one argument"
                    )))
                }
                None => {
                    index += 1;
                    arguments.get(index).cloned().ok_or_else(|| {
                        AnalysisError::new(format!("argument {name}: expected one argument"))
                    })?
                }
            };
            *target = Some(PathBuf::from(value));
            index += 1;
            continue;
        }
        let target_string = match name {
            "--report-glob" => Some(&mut report_glob),
            "--started-glob" => Some(&mut started_glob),
            _ => None,
        };
        if let Some(target) = target_string {
            let value = match inline {
                Some(value) if !value.is_empty() => value.to_owned(),
                Some(_) => {
                    return Err(AnalysisError::new(format!(
                        "argument {name}: expected one argument"
                    )))
                }
                None => {
                    index += 1;
                    arguments
                        .get(index)
                        .and_then(|value| value.to_str())
                        .ok_or_else(|| {
                            AnalysisError::new(format!("argument {name}: expected one argument"))
                        })?
                        .to_owned()
                }
            };
            *target = value;
            index += 1;
            continue;
        }
        return Err(AnalysisError::new(format!(
            "unrecognized arguments: {argument}"
        )));
    }
    Ok(Cli {
        trajectory_root: trajectory_root.ok_or_else(|| {
            AnalysisError::new("the following arguments are required: --trajectory-root")
        })?,
        reports_root: reports_root.ok_or_else(|| {
            AnalysisError::new("the following arguments are required: --reports-root")
        })?,
        output_dir: output_dir.ok_or_else(|| {
            AnalysisError::new("the following arguments are required: --output-dir")
        })?,
        include_failed,
        report_glob,
        started_glob,
    })
}

fn help(program: &str) {
    println!("usage: {program} --trajectory-root DIR --reports-root DIR --output-dir DIR [--include-failed]");
    println!();
    println!("Analyze CPU profile coverage once per Terminal-Bench task.");
}

pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    if arguments
        .first()
        .and_then(|argument| argument.to_str())
        .is_some_and(|argument| matches!(argument, "-h" | "--help"))
    {
        help(public_program);
        return 0;
    }
    let result: Result<(Cli, Value, BTreeMap<&'static str, PathBuf>), AnalysisError> = (|| {
        let cli = parse_cli(arguments)?;
        let analysis = analyze(
            &cli.trajectory_root,
            &cli.reports_root,
            cli.include_failed,
            &cli.report_glob,
            &cli.started_glob,
        )?;
        let outputs = write_outputs(&analysis, &cli.output_dir)?;
        Ok((cli, analysis, outputs))
    })();
    match result {
        Ok((cli, analysis, outputs)) => {
            let counts = &analysis["counts"];
            let metrics = &analysis["metrics"];
            println!(
                "Terminal-Bench tasks: {}",
                counts["total_terminal_bench_tasks"]
            );
            println!(
                "Coverage: {} passed, {} failed validation ({} with timed-out actions), {} incomplete, {} not profiled",
                counts["passed"],
                counts["failed_validation"],
                counts["failed_validation_with_timed_out_actions"],
                counts["incomplete"],
                counts["not_profiled"],
            );
            println!(
                "Metrics mode: {}",
                if cli.include_failed {
                    "passed plus failed-validation windows"
                } else {
                    "passed only"
                }
            );
            println!(
                "Observed-window weighted average logical cores: {:.6} across {} tasks",
                metrics["observed_window_weighted_average_logical_cores_used"]
                    .as_f64()
                    .unwrap(),
                metrics["included_task_count"],
            );
            println!(
                "Exact concurrency distributions included for: {} tasks",
                metrics["exact_concurrency_distribution_task_count"]
            );
            println!("JSON: {}", outputs["json"].display());
            println!("Task CSV: {}", outputs["task_csv"].display());
            println!("Concurrency CSV: {}", outputs["distribution_csv"].display());
            0
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

    #[test]
    fn glob_component_supports_wildcards_and_ranges() {
        let expression = component_regex("cpuset[12]-bcc*").unwrap();
        assert!(expression.is_match("cpuset1-bcc-set"));
        assert!(!expression.is_match("cpuset3-bcc-set"));
    }

    #[test]
    fn csv_scalar_matches_python_booleans_lists_and_nulls() {
        assert_eq!(csv_scalar(Some(&Value::Bool(true))), "True");
        assert_eq!(csv_scalar(Some(&json!(["failure"]))), "['failure']");
        assert_eq!(csv_scalar(Some(&Value::Null)), "");
    }
}
