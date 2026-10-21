// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Cgroup-v2 charged-memory observation for one gated disposable container.

use crate::timebase::monotonic_ns;
use serde_json::{json, Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};

pub const TIMESERIES_SCHEMA_VERSION: &str = "container-cgroup-resource-timeseries.v1";
pub const MEMORY_REPORT_SCHEMA_VERSION: &str = "container-cgroup-memory.v2";
pub const MEMORY_SOURCE_SCHEMA_VERSION: &str = "container-cgroup-memory-source-snapshots.v1";
const REQUIRED_EVENTS: &[&str] = &["high", "max", "oom", "oom_kill", "oom_group_kill"];
const SAMPLE_STAT_KEYS: &[&str] = &[
    "anon",
    "file",
    "kernel",
    "kernel_stack",
    "pagetables",
    "percpu",
    "sock",
    "shmem",
    "zswap",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MemoryError(String);

impl MemoryError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for MemoryError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for MemoryError {}

impl From<std::io::Error> for MemoryError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct Limit {
    bytes: Option<u64>,
    state: &'static str,
}

impl Limit {
    fn unavailable() -> Self {
        Self {
            bytes: None,
            state: "unavailable",
        }
    }
}

#[derive(Clone, Debug)]
struct ScopeSnapshot {
    path: String,
    acquisition_start_ns: u64,
    acquisition_end_ns: u64,
    memory_current: Option<u64>,
    memory_peak: Option<u64>,
    swap_current: Option<u64>,
    swap_peak: Option<u64>,
    limits: BTreeMap<String, Limit>,
    events: BTreeMap<String, u64>,
    events_local: BTreeMap<String, u64>,
    pressure: Value,
    memory_stat: BTreeMap<String, u64>,
    errors: Vec<String>,
}

#[derive(Clone, Debug)]
struct GateSnapshot {
    leaf: ScopeSnapshot,
    ancestors: Vec<ScopeSnapshot>,
}

#[derive(Clone, Debug)]
struct Sample {
    timestamp_ns: u64,
    acquisition_start_ns: u64,
    acquisition_end_ns: u64,
    memory_current_bytes: u64,
    memory_swap_current_bytes: Option<u64>,
    memory_stat: BTreeMap<String, u64>,
    cpu_usage_usec: Option<u64>,
    cpu_user_usec: Option<u64>,
    cpu_system_usec: Option<u64>,
}

#[derive(Clone, Debug)]
pub struct MemoryOutput {
    pub report: Value,
    pub timeseries: Value,
    pub raw_sources: Value,
}

#[derive(Debug)]
pub struct MemorySampler {
    cgroup: PathBuf,
    root: PathBuf,
    ancestors: Vec<PathBuf>,
    interval_sec: f64,
    interval_ns: u64,
    next_sample_ns: Option<u64>,
    start_ns: Option<u64>,
    samples: Vec<Sample>,
    markers: Vec<Value>,
    anomalies: Vec<String>,
    missed_deadlines: u64,
    max_lateness_ns: u64,
    start: Option<GateSnapshot>,
    raw_reads: Vec<Value>,
}

impl MemorySampler {
    pub fn new(cgroup: &Path, root: &Path, interval_sec: f64) -> Result<Self, MemoryError> {
        if !interval_sec.is_finite() || interval_sec <= 0.0 {
            return Err(MemoryError::new("memory sample interval must be positive"));
        }
        let cgroup = fs::canonicalize(cgroup)?;
        let root = fs::canonicalize(root)?;
        if cgroup == root || !cgroup.starts_with(&root) {
            return Err(MemoryError::new(format!(
                "memory cgroup {} is not a leaf below {}",
                cgroup.display(),
                root.display()
            )));
        }
        let mut ancestors = Vec::new();
        let mut current = cgroup
            .parent()
            .ok_or_else(|| MemoryError::new("memory cgroup has no parent"))?;
        loop {
            if current == root {
                if current.join("memory.current").is_file() {
                    ancestors.push(current.to_path_buf());
                }
                break;
            }
            ancestors.push(current.to_path_buf());
            current = current.parent().ok_or_else(|| {
                MemoryError::new("memory ancestor chain does not reach cgroup root")
            })?;
        }
        let interval_ns = (interval_sec * 1_000_000_000.0).round() as u64;
        Ok(Self {
            cgroup,
            root,
            ancestors,
            interval_sec,
            interval_ns: interval_ns.max(1),
            next_sample_ns: None,
            start_ns: None,
            samples: Vec::new(),
            markers: Vec::new(),
            anomalies: Vec::new(),
            missed_deadlines: 0,
            max_lateness_ns: 0,
            start: None,
            raw_reads: Vec::new(),
        })
    }

    fn relative(&self, path: &Path) -> String {
        path.strip_prefix(&self.root)
            .ok()
            .map(|path| {
                let value = path.to_string_lossy();
                if value.is_empty() {
                    ".".to_owned()
                } else {
                    value.into_owned()
                }
            })
            .unwrap_or_else(|| "<outside-root>".to_owned())
    }

    fn read_source(
        &mut self,
        directory: &Path,
        filename: &str,
        phase: &str,
        scope: &str,
    ) -> Option<String> {
        let started = monotonic_ns();
        let result = fs::read_to_string(directory.join(filename));
        let ended = monotonic_ns();
        match result {
            Ok(text) => {
                self.raw_reads.push(json!({
                    "read_index": self.raw_reads.len(),
                    "phase": phase,
                    "source_scope": scope,
                    "filename": filename,
                    "acquisition_start_ns": started,
                    "acquisition_end_ns": ended,
                    "available": true,
                    "raw_text": text,
                    "error_type": Value::Null,
                }));
                Some(text)
            }
            Err(error) => {
                self.raw_reads.push(json!({
                    "read_index": self.raw_reads.len(),
                    "phase": phase,
                    "source_scope": scope,
                    "filename": filename,
                    "acquisition_start_ns": started,
                    "acquisition_end_ns": ended,
                    "available": false,
                    "raw_text": Value::Null,
                    "error_type": format!("{}", error.kind()),
                }));
                None
            }
        }
    }

    fn parse_integer(
        text: Option<String>,
        source: &str,
        errors: &mut Vec<String>,
        required: bool,
    ) -> Option<u64> {
        let Some(text) = text else {
            if required {
                errors.push(format!("required memory source is unavailable: {source}"));
            }
            return None;
        };
        match text.trim().parse::<u64>() {
            Ok(value) => Some(value),
            Err(_) => {
                errors.push(format!("invalid nonnegative integer in {source}: {text:?}"));
                None
            }
        }
    }

    fn parse_limit(
        text: Option<String>,
        source: &str,
        errors: &mut Vec<String>,
        required: bool,
    ) -> Limit {
        let Some(text) = text else {
            if required {
                errors.push(format!("required memory limit is unavailable: {source}"));
            }
            return Limit::unavailable();
        };
        let raw = text.trim();
        if raw == "max" {
            return Limit {
                bytes: None,
                state: "unlimited",
            };
        }
        match raw.parse::<u64>() {
            Ok(value) => Limit {
                bytes: Some(value),
                state: "finite",
            },
            Err(_) => {
                errors.push(format!("invalid memory limit in {source}: {raw:?}"));
                Limit {
                    bytes: None,
                    state: "invalid",
                }
            }
        }
    }

    fn parse_flat(
        text: Option<String>,
        source: &str,
        errors: &mut Vec<String>,
        required: bool,
    ) -> BTreeMap<String, u64> {
        let Some(text) = text else {
            if required {
                errors.push(format!("required memory source is unavailable: {source}"));
            }
            return BTreeMap::new();
        };
        let mut values = BTreeMap::new();
        for line in text.lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            if fields.len() != 2 {
                errors.push(format!("invalid {source} row: {line:?}"));
                continue;
            }
            match fields[1].parse::<u64>() {
                Ok(value) => {
                    values.insert(fields[0].to_owned(), value);
                }
                Err(_) => errors.push(format!("invalid {source} value: {line:?}")),
            }
        }
        values
    }

    fn parse_pressure(text: Option<String>, source: &str, errors: &mut Vec<String>) -> Value {
        let Some(text) = text else {
            return json!({});
        };
        let mut output = Map::new();
        for line in text.lines() {
            let fields: Vec<_> = line.split_whitespace().collect();
            if fields.is_empty() {
                continue;
            }
            let mut row = Map::new();
            for token in &fields[1..] {
                let Some((key, raw)) = token.split_once('=') else {
                    errors.push(format!("invalid {source} token: {token:?}"));
                    continue;
                };
                let value = if key == "total" {
                    raw.parse::<u64>().ok().map(Value::from)
                } else {
                    raw.parse::<f64>()
                        .ok()
                        .and_then(Number::from_f64)
                        .map(Value::Number)
                };
                match value {
                    Some(value) => {
                        row.insert(key.to_owned(), value);
                    }
                    None => errors.push(format!("invalid {source} value: {token:?}")),
                }
            }
            output.insert(fields[0].to_owned(), Value::Object(row));
        }
        Value::Object(output)
    }

    fn capture_scope(&mut self, path: &Path, phase: &str, leaf: bool) -> ScopeSnapshot {
        let started = monotonic_ns();
        let scope = if leaf {
            "leaf".to_owned()
        } else {
            format!("ancestor:{}", self.relative(path))
        };
        let mut errors = Vec::new();
        let memory_current = Self::parse_integer(
            self.read_source(path, "memory.current", phase, &scope),
            "memory.current",
            &mut errors,
            true,
        );
        let memory_peak = if leaf {
            Self::parse_integer(
                self.read_source(path, "memory.peak", phase, &scope),
                "memory.peak",
                &mut errors,
                false,
            )
        } else {
            None
        };
        let swap_current = if leaf {
            Self::parse_integer(
                self.read_source(path, "memory.swap.current", phase, &scope),
                "memory.swap.current",
                &mut errors,
                false,
            )
        } else {
            None
        };
        let swap_peak = if leaf {
            Self::parse_integer(
                self.read_source(path, "memory.swap.peak", phase, &scope),
                "memory.swap.peak",
                &mut errors,
                false,
            )
        } else {
            None
        };
        let mut limits = BTreeMap::new();
        for (key, filename, required) in if leaf {
            vec![
                ("memory_high", "memory.high", true),
                ("memory_max", "memory.max", true),
                ("swap_high", "memory.swap.high", false),
                ("swap_max", "memory.swap.max", true),
            ]
        } else {
            vec![
                ("memory_high", "memory.high", true),
                ("memory_max", "memory.max", true),
            ]
        } {
            limits.insert(
                key.to_owned(),
                Self::parse_limit(
                    self.read_source(path, filename, phase, &scope),
                    filename,
                    &mut errors,
                    required,
                ),
            );
        }
        let events = if leaf {
            Self::parse_flat(
                self.read_source(path, "memory.events", phase, &scope),
                "memory.events",
                &mut errors,
                true,
            )
        } else {
            BTreeMap::new()
        };
        let events_local = Self::parse_flat(
            self.read_source(path, "memory.events.local", phase, &scope),
            "memory.events.local",
            &mut errors,
            true,
        );
        for key in REQUIRED_EVENTS {
            if leaf && !events.contains_key(*key) {
                errors.push(format!("memory.events lacks required counter {key:?}"));
            }
            if !events_local.contains_key(*key) {
                errors.push(format!(
                    "memory.events.local lacks required counter {key:?}"
                ));
            }
        }
        let pressure = Self::parse_pressure(
            self.read_source(path, "memory.pressure", phase, &scope),
            "memory.pressure",
            &mut errors,
        );
        let memory_stat = if leaf {
            Self::parse_flat(
                self.read_source(path, "memory.stat", phase, &scope),
                "memory.stat",
                &mut errors,
                true,
            )
        } else {
            BTreeMap::new()
        };
        deduplicate(&mut errors);
        ScopeSnapshot {
            path: self.relative(path),
            acquisition_start_ns: started,
            acquisition_end_ns: monotonic_ns(),
            memory_current,
            memory_peak,
            swap_current,
            swap_peak,
            limits,
            events,
            events_local,
            pressure,
            memory_stat,
            errors,
        }
    }

    fn capture_gate(&mut self, phase: &str) -> GateSnapshot {
        let cgroup = self.cgroup.clone();
        let leaf = self.capture_scope(&cgroup, phase, true);
        let paths = self.ancestors.clone();
        let ancestors = paths
            .iter()
            .map(|path| self.capture_scope(path, phase, false))
            .collect();
        GateSnapshot { leaf, ancestors }
    }

    fn capture_sample(&mut self, timestamp_ns: u64) {
        let started = timestamp_ns;
        let current = match fs::read_to_string(self.cgroup.join("memory.current"))
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok())
        {
            Some(value) => value,
            None => {
                self.anomalies
                    .push("memory.current sample was unavailable or malformed".to_owned());
                return;
            }
        };
        let swap = fs::read_to_string(self.cgroup.join("memory.swap.current"))
            .ok()
            .and_then(|value| value.trim().parse::<u64>().ok());
        let mut parse_errors = Vec::new();
        let stat = Self::parse_flat(
            fs::read_to_string(self.cgroup.join("memory.stat")).ok(),
            "memory.stat sample",
            &mut parse_errors,
            true,
        );
        let cpu = Self::parse_flat(
            fs::read_to_string(self.cgroup.join("cpu.stat")).ok(),
            "cpu.stat sample",
            &mut parse_errors,
            false,
        );
        self.anomalies.extend(parse_errors);
        let sample = Sample {
            timestamp_ns,
            acquisition_start_ns: started,
            acquisition_end_ns: monotonic_ns(),
            memory_current_bytes: current,
            memory_swap_current_bytes: swap,
            memory_stat: SAMPLE_STAT_KEYS
                .iter()
                .filter_map(|key| {
                    stat.get(*key)
                        .copied()
                        .map(|value| ((*key).to_owned(), value))
                })
                .collect(),
            cpu_usage_usec: cpu.get("usage_usec").copied(),
            cpu_user_usec: cpu.get("user_usec").copied(),
            cpu_system_usec: cpu.get("system_usec").copied(),
        };
        if self
            .samples
            .last()
            .is_some_and(|previous| timestamp_ns < previous.timestamp_ns)
        {
            self.anomalies
                .push("sampling clock moved backwards".to_owned());
            return;
        }
        if self
            .samples
            .last()
            .is_some_and(|previous| timestamp_ns == previous.timestamp_ns)
        {
            self.samples.pop();
        }
        self.samples.push(sample);
    }

    fn add_marker(&mut self, name: &str, timestamp_ns: u64) {
        let start = self.start_ns.unwrap_or(timestamp_ns);
        self.markers.push(json!({
            "name": name,
            "timestamp_ns": timestamp_ns,
            "elapsed_sec": round9((timestamp_ns.saturating_sub(start)) as f64 / 1_000_000_000.0),
        }));
    }

    pub fn start(&mut self) -> Result<(), MemoryError> {
        if self.start.is_some() {
            return Err(MemoryError::new("memory sampling already started"));
        }
        self.raw_reads.clear();
        let gate = self.capture_gate("observer_start");
        let timestamp = monotonic_ns();
        self.start_ns = Some(timestamp);
        self.capture_sample(timestamp);
        self.add_marker("sandbox_observation_start", timestamp);
        self.next_sample_ns = Some(timestamp.saturating_add(self.interval_ns));
        self.start = Some(gate);
        Ok(())
    }

    pub fn poll(&mut self) -> Result<(), MemoryError> {
        let Some(deadline) = self.next_sample_ns else {
            return Err(MemoryError::new("memory sampling is not active"));
        };
        let timestamp = monotonic_ns();
        if timestamp < deadline {
            return Ok(());
        }
        let lateness = timestamp - deadline;
        self.max_lateness_ns = self.max_lateness_ns.max(lateness);
        self.missed_deadlines = self
            .missed_deadlines
            .saturating_add(lateness / self.interval_ns);
        self.capture_sample(timestamp);
        self.next_sample_ns = Some(timestamp.saturating_add(self.interval_ns));
        Ok(())
    }

    pub fn mark(&mut self, name: &str) -> Result<u64, MemoryError> {
        if self.start.is_none() {
            return Err(MemoryError::new("memory sampling is not active"));
        }
        let timestamp = monotonic_ns();
        self.capture_sample(timestamp);
        self.add_marker(name, timestamp);
        self.next_sample_ns = Some(timestamp.saturating_add(self.interval_ns));
        Ok(timestamp)
    }

    pub fn finish(
        &mut self,
        command_exit_code: i32,
        remaining_descendant_pids: &[u64],
    ) -> Result<MemoryOutput, MemoryError> {
        let start = self
            .start
            .clone()
            .ok_or_else(|| MemoryError::new("memory sampling is not active"))?;
        let end_ns = self.mark("sandbox_observation_end")?;
        let finish = self.capture_gate("controlled_finish_gate");
        self.next_sample_ns = None;
        Ok(build_output(
            self,
            start,
            finish,
            end_ns,
            command_exit_code,
            remaining_descendant_pids,
        ))
    }
}

fn deduplicate(values: &mut Vec<String>) {
    let mut seen = BTreeSet::new();
    values.retain(|value| seen.insert(value.clone()));
}

fn round9(value: f64) -> f64 {
    (value * 1_000_000_000.0).round() / 1_000_000_000.0
}

fn counter_delta(
    before: &BTreeMap<String, u64>,
    after: &BTreeMap<String, u64>,
    source: &str,
    errors: &mut Vec<String>,
) -> BTreeMap<String, u64> {
    let mut keys: BTreeSet<&String> = before.keys().collect();
    keys.extend(after.keys());
    let mut output = BTreeMap::new();
    for key in keys {
        match (before.get(key), after.get(key)) {
            (Some(start), Some(end)) if end >= start => {
                output.insert(key.clone(), end - start);
            }
            (Some(start), Some(end)) => errors.push(format!(
                "{source} counter {key:?} moved backwards: {start} -> {end}"
            )),
            _ => errors.push(format!(
                "{source} counter {key:?} was absent at one boundary"
            )),
        }
    }
    output
}

fn pressure_total(value: &Value, kind: &str) -> Option<u64> {
    value.get(kind)?.get("total")?.as_u64()
}

fn pressure_delta(
    before: &Value,
    after: &Value,
    duration_sec: f64,
    source: &str,
    errors: &mut Vec<String>,
) -> Value {
    let mut output = Map::new();
    for kind in ["some", "full"] {
        match (pressure_total(before, kind), pressure_total(after, kind)) {
            (Some(start), Some(end)) if end >= start => {
                let delta = end - start;
                output.insert(
                    kind.to_owned(),
                    json!({
                        "total_stall_usec": delta,
                        "stall_fraction_of_wall": if duration_sec > 0.0 { Value::from(round9(delta as f64 / (duration_sec * 1_000_000.0))) } else { Value::Null },
                    }),
                );
            }
            (Some(start), Some(end)) => errors.push(format!(
                "{source} {kind}.total moved backwards: {start} -> {end}"
            )),
            _ => {}
        }
    }
    Value::Object(output)
}

fn limit_json(limits: &BTreeMap<String, Limit>, key: &str) -> (Value, Value) {
    limits.get(key).map_or(
        (Value::Null, Value::String("unavailable".to_owned())),
        |limit| {
            (
                limit.bytes.map(Value::from).unwrap_or(Value::Null),
                Value::String(limit.state.to_owned()),
            )
        },
    )
}

fn snapshot_json(snapshot: &ScopeSnapshot) -> Value {
    let (high, high_state) = limit_json(&snapshot.limits, "memory_high");
    let (max, max_state) = limit_json(&snapshot.limits, "memory_max");
    json!({
        "path": snapshot.path,
        "acquisition_start_ns": snapshot.acquisition_start_ns,
        "acquisition_end_ns": snapshot.acquisition_end_ns,
        "memory_current_bytes": snapshot.memory_current,
        "memory_high_bytes": high,
        "memory_high_state": high_state,
        "memory_max_bytes": max,
        "memory_max_state": max_state,
        "events_local": snapshot.events_local,
        "pressure": snapshot.pressure,
        "pressure_available": snapshot.pressure.as_object().is_some_and(|value| !value.is_empty()),
        "errors": snapshot.errors,
    })
}

fn summarize(samples: &[Sample], start_ns: u64, end_ns: u64) -> Value {
    let selected: Vec<_> = samples
        .iter()
        .filter(|sample| start_ns <= sample.timestamp_ns && sample.timestamp_ns <= end_ns)
        .collect();
    if selected.is_empty() {
        return json!({
            "sample_count": 0,
            "duration_sec": 0.0,
            "start_bytes": Value::Null,
            "end_bytes": Value::Null,
            "sampled_peak_bytes": Value::Null,
            "sampled_peak_elapsed_sec": Value::Null,
            "sampled_time_weighted_average_bytes": Value::Null,
            "sampled_peak_swap_bytes": Value::Null,
            "max_sample_gap_sec": Value::Null,
            "sampled_cpu_usage_sec": Value::Null,
            "sampled_cpu_average_logical_cores": Value::Null,
        });
    }
    let peak = selected
        .iter()
        .max_by_key(|sample| sample.memory_current_bytes)
        .unwrap();
    let gaps: Vec<_> = selected
        .windows(2)
        .map(|pair| pair[1].timestamp_ns - pair[0].timestamp_ns)
        .collect();
    let weighted = if selected.len() == 1 {
        selected[0].memory_current_bytes
    } else {
        let numerator: u128 = selected
            .windows(2)
            .map(|pair| {
                (pair[0].memory_current_bytes as u128 + pair[1].memory_current_bytes as u128)
                    * (pair[1].timestamp_ns - pair[0].timestamp_ns) as u128
            })
            .sum();
        let denominator =
            2u128 * (selected.last().unwrap().timestamp_ns - selected[0].timestamp_ns) as u128;
        if denominator == 0 {
            selected.last().unwrap().memory_current_bytes
        } else {
            ((numerator + denominator / 2) / denominator) as u64
        }
    };
    let cpu_delta = selected
        .first()
        .and_then(|first| first.cpu_usage_usec)
        .zip(selected.last().and_then(|last| last.cpu_usage_usec))
        .filter(|(first, last)| last >= first)
        .map(|(first, last)| last - first);
    let duration_sec = (end_ns.saturating_sub(start_ns)) as f64 / 1_000_000_000.0;
    json!({
        "sample_count": selected.len(),
        "duration_sec": round9(duration_sec),
        "start_bytes": selected.first().unwrap().memory_current_bytes,
        "end_bytes": selected.last().unwrap().memory_current_bytes,
        "sampled_peak_bytes": peak.memory_current_bytes,
        "sampled_peak_elapsed_sec": round9(peak.timestamp_ns.saturating_sub(start_ns) as f64 / 1_000_000_000.0),
        "sampled_time_weighted_average_bytes": weighted,
        "sampled_peak_swap_bytes": selected.iter().filter_map(|sample| sample.memory_swap_current_bytes).max(),
        "max_sample_gap_sec": gaps.iter().max().map(|gap| round9(*gap as f64 / 1_000_000_000.0)).unwrap_or(0.0),
        "sampled_cpu_usage_sec": cpu_delta.map(|value| round9(value as f64 / 1_000_000.0)),
        "sampled_cpu_average_logical_cores": cpu_delta.filter(|_| duration_sec > 0.0).map(|value| round9(value as f64 / 1_000_000.0 / duration_sec)),
        "memory_stat_at_sampled_peak": peak.memory_stat,
    })
}

fn samples_payload(sampler: &MemorySampler) -> Value {
    let start = sampler.start_ns.unwrap_or(0);
    let mut previous: Option<&Sample> = None;
    let rows: Vec<_> = sampler.samples.iter().map(|sample| {
        let interval = previous.map(|value| (sample.timestamp_ns - value.timestamp_ns) as f64 / 1_000_000_000.0);
        let cpu_average = previous.and_then(|value| {
            value.cpu_usage_usec.zip(sample.cpu_usage_usec)
                .filter(|(before, after)| after >= before)
                .zip(interval)
                .filter(|(_, interval)| *interval > 0.0)
                .map(|((before, after), interval)| round9((after - before) as f64 / 1_000_000.0 / interval))
        });
        let acquisition = sample.acquisition_end_ns - sample.acquisition_start_ns;
        let row = json!({
            "timestamp_ns": sample.timestamp_ns,
            "acquisition_start_ns": sample.acquisition_start_ns,
            "acquisition_end_ns": sample.acquisition_end_ns,
            "memory_current_bytes": sample.memory_current_bytes,
            "memory_swap_current_bytes": sample.memory_swap_current_bytes,
            "memory_stat": sample.memory_stat,
            "cpu_usage_usec": sample.cpu_usage_usec,
            "cpu_user_usec": sample.cpu_user_usec,
            "cpu_system_usec": sample.cpu_system_usec,
            "elapsed_sec": round9(sample.timestamp_ns.saturating_sub(start) as f64 / 1_000_000_000.0),
            "acquisition_start_elapsed_sec": round9(sample.acquisition_start_ns.saturating_sub(start) as f64 / 1_000_000_000.0),
            "acquisition_end_elapsed_sec": round9(sample.acquisition_end_ns.saturating_sub(start) as f64 / 1_000_000_000.0),
            "acquisition_duration_sec": round9(acquisition as f64 / 1_000_000_000.0),
            "timestamp_alignment_uncertainty_sec": round9(acquisition as f64 / 1_000_000_000.0),
            "interval_sec": interval.map(round9),
            "cpu_average_logical_cores": cpu_average,
        });
        previous = Some(sample);
        row
    }).collect();
    json!({
        "schema_version": TIMESERIES_SCHEMA_VERSION,
        "scope": sampler.cgroup,
        "clock": "host-monotonic",
        "timestamp_semantics": "acquisition-start",
        "timestamp_alignment_uncertainty_semantics": "The cgroup files are read serially after timestamp_ns. Each value was acquired no later than acquisition_end_ns; the full acquisition duration is the conservative alignment uncertainty.",
        "sample_interval_sec": sampler.interval_sec,
        "max_sample_acquisition_duration_sec": sampler.samples.iter().map(|sample| sample.acquisition_end_ns - sample.acquisition_start_ns).max().map(|value| round9(value as f64 / 1_000_000_000.0)),
        "markers": sampler.markers,
        "samples": rows,
    })
}

fn build_output(
    sampler: &MemorySampler,
    start: GateSnapshot,
    finish: GateSnapshot,
    end_ns: u64,
    command_exit_code: i32,
    remaining_descendant_pids: &[u64],
) -> MemoryOutput {
    let start_ns = sampler.start_ns.unwrap_or(end_ns);
    let duration_sec = end_ns.saturating_sub(start_ns) as f64 / 1_000_000_000.0;
    let mut instrumentation_reasons = sampler.anomalies.clone();
    instrumentation_reasons.extend(start.leaf.errors.clone());
    instrumentation_reasons.extend(finish.leaf.errors.clone());
    let mut policy_drift = Vec::new();
    for key in ["memory_high", "memory_max", "swap_high", "swap_max"] {
        if start.leaf.limits.get(key) != finish.leaf.limits.get(key) {
            policy_drift.push(format!("leaf {key} changed during observation"));
        }
    }
    let events_delta = counter_delta(
        &start.leaf.events,
        &finish.leaf.events,
        "memory.events",
        &mut instrumentation_reasons,
    );
    let events_local_delta = counter_delta(
        &start.leaf.events_local,
        &finish.leaf.events_local,
        "memory.events.local",
        &mut instrumentation_reasons,
    );
    let pressure = pressure_delta(
        &start.leaf.pressure,
        &finish.leaf.pressure,
        duration_sec,
        "memory.pressure",
        &mut instrumentation_reasons,
    );
    let mut ancestor_rows = Vec::new();
    let mut ancestor_invalid = Vec::new();
    let mut ancestor_oom_kill = false;
    if start.ancestors.len() != finish.ancestors.len() {
        instrumentation_reasons.push("ancestor memory cgroup chain length changed".to_owned());
    }
    for (before, after) in start.ancestors.iter().zip(&finish.ancestors) {
        let mut errors = before.errors.clone();
        errors.extend(after.errors.clone());
        if before.path != after.path {
            errors.push(format!(
                "ancestor memory path changed from {:?} to {:?}",
                before.path, after.path
            ));
        }
        for key in ["memory_high", "memory_max"] {
            if before.limits.get(key) != after.limits.get(key) {
                let reason = format!(
                    "ancestor {:?} {key} changed during observation",
                    before.path
                );
                errors.push(reason.clone());
                policy_drift.push(reason);
            }
        }
        let delta = counter_delta(
            &before.events_local,
            &after.events_local,
            &format!("ancestor {:?} memory.events.local", before.path),
            &mut errors,
        );
        let pressure_delta = pressure_delta(
            &before.pressure,
            &after.pressure,
            duration_sec,
            &format!("ancestor {:?} memory.pressure", before.path),
            &mut errors,
        );
        let constraints: BTreeMap<_, _> = REQUIRED_EVENTS
            .iter()
            .filter_map(|key| {
                delta
                    .get(*key)
                    .copied()
                    .filter(|value| *value > 0)
                    .map(|value| ((*key).to_owned(), value))
            })
            .collect();
        ancestor_oom_kill |= delta.get("oom_kill").copied().unwrap_or(0) > 0
            || delta.get("oom_group_kill").copied().unwrap_or(0) > 0;
        let positive_pressure: BTreeMap<_, _> = ["some", "full"]
            .into_iter()
            .filter_map(|kind| {
                pressure_delta
                    .get(kind)?
                    .get("total_stall_usec")?
                    .as_u64()
                    .filter(|value| *value > 0)
                    .map(|value| (kind.to_owned(), value))
            })
            .collect();
        let mut invalid = errors.clone();
        if !constraints.is_empty() {
            invalid.push(format!(
                "ancestor {:?} recorded constraint/OOM events",
                before.path
            ));
        }
        if !positive_pressure.is_empty() {
            invalid.push(format!(
                "ancestor {:?} recorded memory pressure",
                before.path
            ));
        }
        deduplicate(&mut errors);
        deduplicate(&mut invalid);
        instrumentation_reasons.extend(errors.clone());
        ancestor_invalid.extend(invalid.clone());
        ancestor_rows.push(json!({
            "path": before.path,
            "start": snapshot_json(before),
            "finish": snapshot_json(after),
            "events_local_delta": delta,
            "pressure_delta": pressure_delta,
            "constraint_event_deltas": constraints,
            "positive_pressure_total_usec": positive_pressure,
            "finite_limits_present": before.limits.values().any(|limit| limit.state == "finite"),
            "evidence_valid": errors.is_empty(),
            "attribution_ambiguous": !invalid.is_empty(),
            "natural_demand_trustworthy": invalid.is_empty(),
            "invalid_reasons": invalid,
        }));
    }
    instrumentation_reasons.extend(policy_drift.clone());
    deduplicate(&mut instrumentation_reasons);
    deduplicate(&mut ancestor_invalid);
    let observation = summarize(&sampler.samples, start_ns, end_ns);
    let marker_time = |name: &str| {
        sampler
            .markers
            .iter()
            .find(|marker| marker["name"] == name)
            .and_then(|marker| marker["timestamp_ns"].as_u64())
    };
    let command_window = marker_time("command_start")
        .zip(marker_time("command_end"))
        .map(|(start, end)| summarize(&sampler.samples, start, end));
    let sampled_peak = observation["sampled_peak_bytes"].as_u64();
    let peak = finish.leaf.memory_peak;
    let mut peak_invalid = Vec::new();
    if peak.is_none() {
        peak_invalid.push("memory.peak is unavailable".to_owned());
    }
    if peak
        .zip(sampled_peak)
        .is_some_and(|(peak, sample)| peak < sample)
    {
        peak_invalid
            .push("memory.peak was smaller than an observed memory.current sample".to_owned());
    }
    let max_gap = observation["max_sample_gap_sec"].as_f64();
    let mut timeline_invalid = Vec::new();
    if sampler.samples.len() < 2 {
        timeline_invalid.push("fewer than two resource samples".to_owned());
    }
    if max_gap.is_some_and(|gap| gap > 2.0 * sampler.interval_sec) {
        timeline_invalid.push(format!(
            "maximum sample gap {:.6}s exceeded twice the requested interval",
            max_gap.unwrap()
        ));
    }
    let oom_kill = events_delta.get("oom_kill").copied().unwrap_or(0) > 0
        || events_delta.get("oom_group_kill").copied().unwrap_or(0) > 0
        || ancestor_oom_kill;
    let censoring = if events_delta.get("oom").copied().unwrap_or(0) > 0 || oom_kill {
        "oom"
    } else if events_delta.get("max").copied().unwrap_or(0) > 0 {
        "memory_max"
    } else if events_delta.get("high").copied().unwrap_or(0) > 0 {
        "memory_high"
    } else if finish.leaf.swap_peak.unwrap_or(0) > start.leaf.swap_peak.unwrap_or(0)
        || observation["sampled_peak_swap_bytes"].as_u64().unwrap_or(0) > 0
    {
        "swap_activity"
    } else if command_exit_code != 0 {
        "functional_failure"
    } else {
        "none"
    };
    let observation_complete = remaining_descendant_pids.is_empty();
    let instrumentation_valid = instrumentation_reasons.is_empty() && !sampler.samples.is_empty();
    let peak_valid = peak.is_some() && peak_invalid.is_empty();
    let timeline_valid = timeline_invalid.is_empty();
    let valid_for_observed_memory_reporting = instrumentation_valid
        && peak_valid
        && timeline_valid
        && observation_complete
        && !oom_kill
        && policy_drift.is_empty();
    let mut natural_invalid = instrumentation_reasons.clone();
    natural_invalid.extend(ancestor_invalid.clone());
    natural_invalid.extend(peak_invalid.clone());
    natural_invalid.extend(timeline_invalid.clone());
    let swap_limit = start.leaf.limits.get("swap_max");
    let swap_disabled =
        swap_limit.is_some_and(|limit| limit.state == "finite" && limit.bytes == Some(0));
    let swap_peak_observable = start.leaf.swap_peak.is_some() && finish.leaf.swap_peak.is_some();
    if swap_limit.map_or(true, |limit| limit.state == "unavailable")
        || (!swap_disabled && !swap_peak_observable)
    {
        natural_invalid.push(
            "swap activity was not provably observable: require readable memory.swap.max and either disabled swap or memory.swap.peak"
                .to_owned(),
        );
    }
    if start
        .leaf
        .limits
        .get("memory_max")
        .map_or(true, |limit| limit.state == "unavailable")
    {
        natural_invalid.push("memory.max was unavailable".to_owned());
    }
    if censoring != "none" {
        natural_invalid.push(format!("memory demand is censored by {censoring}"));
    }
    if !observation_complete {
        natural_invalid.push("descendants remained after the bounded finish drain".to_owned());
    }
    deduplicate(&mut natural_invalid);
    let natural_peak_trustworthy = natural_invalid.is_empty();
    let valid_for_sizing = valid_for_observed_memory_reporting && natural_peak_trustworthy;
    let (memory_max, memory_max_state) = limit_json(&start.leaf.limits, "memory_max");
    let finite_limit = memory_max.as_u64().filter(|_| memory_max_state == "finite");
    let remaining = finite_limit
        .zip(peak)
        .map(|(limit, peak)| limit.saturating_sub(peak));
    let fraction = finite_limit
        .filter(|limit| *limit > 0)
        .zip(peak)
        .map(|(limit, peak)| peak as f64 / limit as f64);
    let (memory_high, memory_high_state) = limit_json(&start.leaf.limits, "memory_high");
    let (swap_high, swap_high_state) = limit_json(&start.leaf.limits, "swap_high");
    let (swap_max, swap_max_state) = limit_json(&start.leaf.limits, "swap_max");
    let ancestor_report = json!({
        "cgroup_root": sampler.root,
        "cgroup_root_memory_controller_files_available": sampler.root.join("memory.current").is_file(),
        "leaf_path": start.leaf.path,
        "chain_order": "leaf-parent-to-root",
        "ancestor_count": ancestor_rows.len(),
        "ancestors": ancestor_rows,
        "evidence_complete": ancestor_invalid.iter().all(|reason| !reason.contains("unavailable") && !reason.contains("invalid") && !reason.contains("changed")),
        "finite_limits_present": start.ancestors.iter().any(|row| row.limits.values().any(|limit| limit.state == "finite")),
        "constraint_events_observed": ancestor_invalid.iter().any(|reason| reason.contains("events")),
        "pressure_observed": ancestor_invalid.iter().any(|reason| reason.contains("pressure")),
        "attribution_ambiguous": !ancestor_invalid.is_empty(),
        "natural_demand_trustworthy": ancestor_invalid.is_empty(),
        "natural_demand_untrustworthy": !ancestor_invalid.is_empty(),
        "evidence_errors": instrumentation_reasons,
        "invalid_reasons": ancestor_invalid,
        "meaning": "Ancestor-local counters and pressure cover the measured leaf plus possible sibling cgroups. Positive events or stalls invalidate unconstrained-demand interpretation even when attribution is ambiguous; finite unused ancestor limits are provenance only.",
    });
    let lifetime = json!({
        "peak_bytes": peak,
        "peak_at_observer_start_bytes": start.leaf.memory_peak,
        "peak_source": "memory.peak-since-cgroup-creation-read-at-finish-gate",
        "exact_through_controlled_finish_gate": peak_valid,
        "exact_for_fresh_cgroup_lifetime": false,
        "container_init_exit_observed": false,
    });
    let limits = json!({
        "memory_high_bytes": memory_high,
        "memory_high_state": memory_high_state,
        "memory_max_bytes": memory_max,
        "memory_max_state": memory_max_state,
        "swap_high_bytes": swap_high,
        "swap_high_state": swap_high_state,
        "swap_max_bytes": swap_max,
        "swap_max_state": swap_max_state,
    });
    let margin = json!({
        "limit_scope": "leaf-cgroup-memory.max-at-observer-start",
        "configured_limit_state": memory_max_state,
        "configured_limit_bytes": finite_limit,
        "observed_peak_bytes": peak,
        "remaining_to_limit_bytes": remaining,
        "peak_fraction_of_limit": fraction,
        "arithmetic_valid": finite_limit.is_some() && peak_valid,
        "meaning": "Descriptive arithmetic for this run only: configured leaf memory.max minus the observed exact peak. This is not recommended safety headroom and does not account for a constrained ancestor or future-run variance.",
    });
    let capabilities = json!({
        "memory_events_at_observer_start": !start.leaf.events.is_empty(),
        "memory_events_at_finish_gate": !finish.leaf.events.is_empty(),
        "memory_events_local_at_observer_start": !start.leaf.events_local.is_empty(),
        "memory_events_local_at_finish_gate": !finish.leaf.events_local.is_empty(),
        "memory_peak_at_observer_start": start.leaf.memory_peak.is_some(),
        "memory_peak_at_finish_gate": finish.leaf.memory_peak.is_some(),
        "memory_swap_current_at_observer_start": start.leaf.swap_current.is_some(),
        "memory_swap_current_at_finish_gate": finish.leaf.swap_current.is_some(),
        "memory_swap_peak_at_observer_start": start.leaf.swap_peak.is_some(),
        "memory_swap_peak_at_finish_gate": finish.leaf.swap_peak.is_some(),
        "memory_limit": start.leaf.limits.get("memory_max").is_some_and(|limit| limit.state != "unavailable"),
        "memory_swap_limit": start.leaf.limits.get("swap_max").is_some_and(|limit| limit.state != "unavailable"),
        "ancestor_memory_evidence_complete": ancestor_report["evidence_complete"],
    });
    let mut report = json!({
        "schema_version": MEMORY_REPORT_SCHEMA_VERSION,
        "method": "container-cgroup-v2-memory-and-cpu-stat-sampling",
        "scope": "Measured cgroup v2 subtree, including descendant anonymous, file-cache, and kernel memory charges. This is not process RSS.",
        "clock": "host-monotonic",
        "sample_interval_sec": sampler.interval_sec,
        "sample_count": sampler.samples.len(),
        "raw_source_snapshot_schema_version": MEMORY_SOURCE_SCHEMA_VERSION,
        "raw_source_snapshot_read_count": sampler.raw_reads.len(),
        "missed_sample_deadlines": sampler.missed_deadlines,
        "max_sample_lateness_sec": round9(sampler.max_lateness_ns as f64 / 1_000_000_000.0),
        "max_sample_acquisition_duration_sec": sampler.samples.iter().map(|sample| sample.acquisition_end_ns - sample.acquisition_start_ns).max().map(|value| round9(value as f64 / 1_000_000_000.0)),
        "markers": sampler.markers,
        "observation_window": observation,
        "command_window": command_window,
        "lifetime": lifetime,
        "swap": {"lifetime_peak_bytes": finish.leaf.swap_peak, "sampled_peak_bytes": observation["sampled_peak_swap_bytes"]},
        "limits": limits,
        "observed_limit_margin": margin,
        "capabilities": capabilities,
    });
    let report_object = report.as_object_mut().expect("memory report object");
    report_object.extend(
        json!({
            "observation_boundary": "controlled-finish-gate",
            "observation_boundary_complete": observation_complete,
            "full_cgroup_lifetime_observed": false,
            "events_at_observer_start": start.leaf.events,
            "events_lifetime": finish.leaf.events,
            "events_delta": events_delta,
            "events_local_at_observer_start": start.leaf.events_local,
            "events_local_lifetime": finish.leaf.events_local,
            "events_local_delta": events_local_delta,
            "pressure_delta": pressure,
            "ancestor_memory": ancestor_report,
            "memory_stat_before": start.leaf.memory_stat,
            "memory_stat_after": finish.leaf.memory_stat,
            "censoring": censoring,
            "oom_kill_observed": oom_kill,
            "policy_stable": policy_drift.is_empty(),
            "policy_drift_reasons": policy_drift,
        })
        .as_object()
        .expect("memory report details")
        .clone(),
    );
    report_object.extend(json!({
        "instrumentation_valid": instrumentation_valid,
        "instrumentation_invalid_reasons": instrumentation_reasons,
        "peak_valid": peak_valid,
        "peak_invalid_reasons": peak_invalid,
        "natural_peak_trustworthy": natural_peak_trustworthy,
        "natural_peak_invalid_reasons": natural_invalid,
        "timeline_valid_for_shape": timeline_valid,
        "timeline_invalid_reasons": timeline_invalid,
        "valid_for_observed_memory_reporting": valid_for_observed_memory_reporting,
        "valid_for_sizing": valid_for_sizing,
        "remaining_descendant_pids_at_finish_gate": remaining_descendant_pids,
        "meaning": "memory.peak through the controlled finish gate is the exact observed maximum charge. Sampled memory and interval CPU values explain shape and may miss short spikes. A constrained run retains its observed peak while natural_peak_trustworthy becomes false.",
    }).as_object().expect("memory report validity").clone());
    let timeseries = samples_payload(sampler);
    let raw_sources = json!({
        "schema_version": MEMORY_SOURCE_SCHEMA_VERSION,
        "scope": {"kind": "cgroup-v2-subtree", "leaf_path": start.leaf.path, "path_representation": "relative-to-configured-cgroup-root"},
        "phases": ["observer_start", "controlled_finish_gate"],
        "controllers": ["memory"],
        "reads": sampler.raw_reads,
        "meaning": "Exact text consumed from cgroup memory controller files at both gates. Derived deltas and validity decisions live in the parsed report.",
    });
    MemoryOutput {
        report,
        timeseries,
        raw_sources,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_source(directory: &Path, name: &str, value: &str) {
        fs::write(directory.join(name), value).unwrap();
    }

    fn events(oom_kill: u64) -> String {
        format!("low 0\nhigh 0\nmax 0\noom {oom_kill}\noom_kill {oom_kill}\noom_group_kill 0\n")
    }

    fn synthetic_cgroup() -> (tempfile::TempDir, PathBuf) {
        let root = tempfile::tempdir().unwrap();
        let leaf = root.path().join("leaf");
        fs::create_dir(&leaf).unwrap();
        write_source(&leaf, "memory.current", "10\n");
        write_source(&leaf, "memory.peak", "10\n");
        write_source(&leaf, "memory.swap.current", "0\n");
        write_source(&leaf, "memory.swap.peak", "0\n");
        write_source(&leaf, "memory.high", "max\n");
        write_source(&leaf, "memory.max", "100\n");
        write_source(&leaf, "memory.swap.high", "max\n");
        write_source(&leaf, "memory.swap.max", "0\n");
        write_source(&leaf, "memory.events", &events(0));
        write_source(&leaf, "memory.events.local", &events(0));
        write_source(
            &leaf,
            "memory.pressure",
            "some avg10=0.00 avg60=0.00 avg300=0.00 total=0\nfull avg10=0.00 avg60=0.00 avg300=0.00 total=0\n",
        );
        write_source(
            &leaf,
            "memory.stat",
            "anon 8\nfile 1\nkernel 1\nkernel_stack 0\npagetables 0\npercpu 0\nsock 0\nshmem 0\nzswap 0\n",
        );
        write_source(
            &leaf,
            "cpu.stat",
            "usage_usec 10\nuser_usec 7\nsystem_usec 3\n",
        );
        (root, leaf)
    }

    #[test]
    fn parses_limits_and_flat_counters_fail_closed() {
        let mut errors = Vec::new();
        assert_eq!(
            MemorySampler::parse_limit(Some("max\n".to_owned()), "memory.max", &mut errors, true),
            Limit {
                bytes: None,
                state: "unlimited"
            }
        );
        assert_eq!(
            MemorySampler::parse_limit(
                Some("1048576\n".to_owned()),
                "memory.max",
                &mut errors,
                true
            ),
            Limit {
                bytes: Some(1_048_576),
                state: "finite"
            }
        );
        let values = MemorySampler::parse_flat(
            Some("high 0\noom_kill 2\n".to_owned()),
            "memory.events",
            &mut errors,
            true,
        );
        assert_eq!(values["oom_kill"], 2);
        let _ = MemorySampler::parse_flat(
            Some("bad not-an-int\n".to_owned()),
            "memory.events",
            &mut errors,
            true,
        );
        assert!(errors
            .iter()
            .any(|error| error.contains("invalid memory.events value")));
    }

    #[test]
    fn summarizes_sampled_shape_without_claiming_exact_peak() {
        let samples = vec![
            Sample {
                timestamp_ns: 1_000_000_000,
                acquisition_start_ns: 1_000_000_000,
                acquisition_end_ns: 1_000_000_100,
                memory_current_bytes: 10,
                memory_swap_current_bytes: Some(0),
                memory_stat: BTreeMap::from([("anon".to_owned(), 8)]),
                cpu_usage_usec: Some(100),
                cpu_user_usec: Some(70),
                cpu_system_usec: Some(30),
            },
            Sample {
                timestamp_ns: 2_000_000_000,
                acquisition_start_ns: 2_000_000_000,
                acquisition_end_ns: 2_000_000_100,
                memory_current_bytes: 30,
                memory_swap_current_bytes: Some(0),
                memory_stat: BTreeMap::from([("anon".to_owned(), 24)]),
                cpu_usage_usec: Some(1_000_100),
                cpu_user_usec: Some(800_070),
                cpu_system_usec: Some(200_030),
            },
        ];
        let summary = summarize(&samples, 1_000_000_000, 2_000_000_000);
        assert_eq!(summary["sampled_peak_bytes"], 30);
        assert_eq!(summary["sampled_time_weighted_average_bytes"], 20);
        assert_eq!(summary["sampled_cpu_average_logical_cores"], 1.0);
    }

    #[test]
    fn event_and_pressure_regressions_are_invalid() {
        let mut errors = Vec::new();
        let delta = counter_delta(
            &BTreeMap::from([("oom_kill".to_owned(), 3)]),
            &BTreeMap::from([("oom_kill".to_owned(), 2)]),
            "memory.events",
            &mut errors,
        );
        assert!(delta.is_empty());
        let pressure = pressure_delta(
            &json!({"some": {"total": 10}}),
            &json!({"some": {"total": 9}}),
            1.0,
            "memory.pressure",
            &mut errors,
        );
        assert_eq!(pressure, json!({}));
        assert_eq!(errors.len(), 2);
    }

    #[test]
    fn synthetic_scope_retains_exact_peak_timeline_and_observed_margin() {
        let (root, leaf) = synthetic_cgroup();
        let mut sampler = MemorySampler::new(&leaf, root.path(), 0.05).unwrap();
        sampler.start().unwrap();
        sampler.mark("command_start").unwrap();
        write_source(&leaf, "memory.current", "30\n");
        write_source(&leaf, "memory.peak", "30\n");
        write_source(
            &leaf,
            "memory.stat",
            "anon 24\nfile 3\nkernel 3\nkernel_stack 0\npagetables 0\npercpu 0\nsock 0\nshmem 0\nzswap 0\n",
        );
        sampler.mark("command_end").unwrap();
        let output = sampler.finish(0, &[]).unwrap();
        assert_eq!(output.report["lifetime"]["peak_bytes"], 30);
        assert_eq!(
            output.report["observed_limit_margin"]["remaining_to_limit_bytes"],
            70
        );
        assert_eq!(output.report["peak_valid"], true);
        assert_eq!(output.report["timeline_valid_for_shape"], true);
        assert_eq!(output.report["natural_peak_trustworthy"], true);
        assert_eq!(output.report["valid_for_observed_memory_reporting"], true);
        assert_eq!(output.report["valid_for_sizing"], true);
        assert!(output.timeseries["samples"].as_array().unwrap().len() >= 4);
        assert!(output.raw_sources["reads"]
            .as_array()
            .unwrap()
            .iter()
            .any(|read| {
                read["phase"] == "controlled_finish_gate"
                    && read["filename"] == "memory.peak"
                    && read["raw_text"] == "30\n"
            }));
    }

    #[test]
    fn oom_and_policy_drift_fail_closed_while_retaining_peak() {
        let (root, leaf) = synthetic_cgroup();
        let mut sampler = MemorySampler::new(&leaf, root.path(), 0.05).unwrap();
        sampler.start().unwrap();
        sampler.mark("command_start").unwrap();
        write_source(&leaf, "memory.current", "90\n");
        write_source(&leaf, "memory.peak", "90\n");
        write_source(&leaf, "memory.max", "80\n");
        write_source(&leaf, "memory.events", &events(1));
        write_source(&leaf, "memory.events.local", &events(1));
        sampler.mark("command_end").unwrap();
        let output = sampler.finish(137, &[]).unwrap();
        assert_eq!(output.report["lifetime"]["peak_bytes"], 90);
        assert_eq!(output.report["oom_kill_observed"], true);
        assert_eq!(output.report["policy_stable"], false);
        assert_eq!(output.report["valid_for_observed_memory_reporting"], false);
        assert_eq!(output.report["valid_for_sizing"], false);
        assert_eq!(output.report["natural_peak_trustworthy"], false);
        assert!(output.report["policy_drift_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason.as_str().unwrap().contains("memory_max changed")));
    }

    #[test]
    fn remaining_descendants_invalidate_sizing_scope() {
        let (root, leaf) = synthetic_cgroup();
        let mut sampler = MemorySampler::new(&leaf, root.path(), 0.05).unwrap();
        sampler.start().unwrap();
        sampler.mark("command_start").unwrap();
        sampler.mark("command_end").unwrap();
        let output = sampler.finish(0, &[42]).unwrap();
        assert_eq!(output.report["observation_boundary_complete"], false);
        assert_eq!(output.report["valid_for_observed_memory_reporting"], false);
        assert_eq!(output.report["valid_for_sizing"], false);
        assert_eq!(
            output.report["remaining_descendant_pids_at_finish_gate"][0],
            42
        );
    }

    #[test]
    fn finite_limit_event_keeps_observed_peak_valid_but_rejects_natural_sizing() {
        let (root, leaf) = synthetic_cgroup();
        let mut sampler = MemorySampler::new(&leaf, root.path(), 0.05).unwrap();
        sampler.start().unwrap();
        sampler.mark("command_start").unwrap();
        write_source(&leaf, "memory.current", "90\n");
        write_source(&leaf, "memory.peak", "90\n");
        let constrained = "low 0\nhigh 0\nmax 1\noom 0\noom_kill 0\noom_group_kill 0\n";
        write_source(&leaf, "memory.events", constrained);
        write_source(&leaf, "memory.events.local", constrained);
        sampler.mark("command_end").unwrap();
        let output = sampler.finish(0, &[]).unwrap();
        assert_eq!(output.report["censoring"], "memory_max");
        assert_eq!(output.report["instrumentation_valid"], true);
        assert_eq!(output.report["peak_valid"], true);
        assert_eq!(output.report["valid_for_observed_memory_reporting"], true);
        assert_eq!(output.report["natural_peak_trustworthy"], false);
        assert_eq!(output.report["valid_for_sizing"], false);
        assert_eq!(output.report["lifetime"]["peak_bytes"], 90);
    }
}
