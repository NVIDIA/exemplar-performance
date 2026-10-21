// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Low-privilege sampled host-CPU context for profiled command windows.
//!
//! This intentionally matches TaskTrace's `/proc/stat` busy-time definition:
//! user + nice + system + irq + softirq divided by that busy time plus idle +
//! iowait. The result is host pressure projected onto selected logical CPU IDs,
//! never task-attributed CPU demand or scheduler concurrency.

use serde_json::{json, Number, Value};
use std::collections::{BTreeMap, HashMap};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::thread;
use std::time::{Duration, Instant};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcStatSamplingError(String);

impl ProcStatSamplingError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProcStatSamplingError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ProcStatSamplingError {}

impl From<std::io::Error> for ProcStatSamplingError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ProcStatSamplingError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProcCpuCounters {
    pub user: i128,
    pub system: i128,
    pub idle: i128,
    pub iowait: i128,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ProcStatInterval {
    pub duration_sec: f64,
    pub busy_logical_cores: f64,
    pub active_logical_cpus: usize,
    pub utilization_percent_by_logical_cpu: BTreeMap<String, f64>,
}

fn round_to(value: f64, places: u32) -> f64 {
    let scale = 10_f64.powi(places as i32);
    (value * scale).round() / scale
}

fn json_float(value: f64) -> Value {
    Value::Number(Number::from_f64(value).expect("finite proc-stat metric"))
}

fn parse_counter(value: &str, line: &str) -> Result<i128, ProcStatSamplingError> {
    value.parse::<i128>().map_err(|_| {
        ProcStatSamplingError::new(format!(
            "invalid /proc/stat CPU counter {value:?} in row: {line:?}"
        ))
    })
}

/// Parse per-logical-CPU rows from Linux `/proc/stat` content.
pub fn parse_proc_stat(text: &str) -> Result<HashMap<u32, ProcCpuCounters>, ProcStatSamplingError> {
    let mut counters = HashMap::new();
    for line in text.lines() {
        let fields: Vec<&str> = line.split_whitespace().collect();
        let Some(name) = fields.first().copied() else {
            continue;
        };
        let Some(cpu_text) = name.strip_prefix("cpu") else {
            continue;
        };
        if cpu_text.is_empty() || !cpu_text.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        if fields.len() < 9 {
            return Err(ProcStatSamplingError::new(format!(
                "short /proc/stat CPU row: {line:?}"
            )));
        }
        let cpu = cpu_text.parse::<u32>().map_err(|_| {
            ProcStatSamplingError::new(format!("invalid logical CPU ID: {cpu_text:?}"))
        })?;
        let values: Vec<i128> = fields[1..9]
            .iter()
            .map(|value| parse_counter(value, line))
            .collect::<Result<_, _>>()?;
        let user = values[0].checked_add(values[1]).ok_or_else(|| {
            ProcStatSamplingError::new(format!("/proc/stat user counter overflow: {line:?}"))
        })?;
        let system = values[2]
            .checked_add(values[5])
            .and_then(|value| value.checked_add(values[6]))
            .ok_or_else(|| {
                ProcStatSamplingError::new(format!("/proc/stat system counter overflow: {line:?}"))
            })?;
        counters.insert(
            cpu,
            ProcCpuCounters {
                user,
                system,
                idle: values[3],
                iowait: values[4],
            },
        );
    }
    if counters.is_empty() {
        return Err(ProcStatSamplingError::new(
            "no logical CPU rows found in /proc/stat",
        ));
    }
    Ok(counters)
}

/// Calculate one TaskTrace-style utilization sample on selected CPU IDs.
pub fn calculate_interval(
    previous: &HashMap<u32, ProcCpuCounters>,
    current: &HashMap<u32, ProcCpuCounters>,
    cpu_ids: &[u32],
    duration_sec: f64,
    active_threshold_percent: f64,
) -> Result<ProcStatInterval, ProcStatSamplingError> {
    if !duration_sec.is_finite() || duration_sec <= 0.0 {
        return Err(ProcStatSamplingError::new(
            "sample interval must have positive duration",
        ));
    }
    let mut utilization = BTreeMap::new();
    for cpu in cpu_ids {
        let before = previous.get(cpu).ok_or_else(|| {
            ProcStatSamplingError::new(format!("logical CPU {cpu} missing from /proc/stat"))
        })?;
        let after = current.get(cpu).ok_or_else(|| {
            ProcStatSamplingError::new(format!("logical CPU {cpu} missing from /proc/stat"))
        })?;
        let delta_user = after.user - before.user;
        let delta_system = after.system - before.system;
        let delta_idle = after.idle - before.idle;
        let delta_iowait = after.iowait - before.iowait;
        if [delta_user, delta_system, delta_idle, delta_iowait]
            .iter()
            .any(|value| *value < 0)
        {
            return Err(ProcStatSamplingError::new(format!(
                "logical CPU {cpu} counters moved backwards"
            )));
        }
        let total = delta_user + delta_system + delta_idle + delta_iowait;
        let busy = delta_user + delta_system;
        let percent = if total == 0 {
            0.0
        } else {
            round_to(100.0 * busy as f64 / total as f64, 1)
        };
        utilization.insert(cpu.to_string(), percent);
    }
    let busy_logical_cores = round_to(utilization.values().sum::<f64>() / 100.0, 9);
    let active_logical_cpus = utilization
        .values()
        .filter(|percent| **percent > active_threshold_percent)
        .count();
    Ok(ProcStatInterval {
        duration_sec,
        busy_logical_cores,
        active_logical_cpus,
        utilization_percent_by_logical_cpu: utilization,
    })
}

/// Summarize samples while weighting a terminal partial bucket by duration.
pub fn summarize_intervals(intervals: &[ProcStatInterval]) -> Value {
    let duration_sec: f64 = intervals.iter().map(|interval| interval.duration_sec).sum();
    if !duration_sec.is_finite() || duration_sec <= 0.0 {
        return json!({
            "sample_count": 0,
            "duration_sec": 0.0,
            "average_busy_logical_cores": Value::Null,
            "average_active_logical_cpus": Value::Null,
            "peak_active_logical_cpus": Value::Null,
            "peak_busy_logical_cores_in_sample": Value::Null,
        });
    }
    let average_busy = intervals
        .iter()
        .map(|interval| interval.busy_logical_cores * interval.duration_sec)
        .sum::<f64>()
        / duration_sec;
    let average_active = intervals
        .iter()
        .map(|interval| interval.active_logical_cpus as f64 * interval.duration_sec)
        .sum::<f64>()
        / duration_sec;
    let peak_active = intervals
        .iter()
        .map(|interval| interval.active_logical_cpus)
        .max()
        .expect("positive duration requires at least one interval");
    let peak_busy = intervals
        .iter()
        .map(|interval| interval.busy_logical_cores)
        .fold(f64::NEG_INFINITY, f64::max);
    json!({
        "sample_count": intervals.len(),
        "duration_sec": round_to(duration_sec, 9),
        "average_busy_logical_cores": round_to(average_busy, 9),
        "average_active_logical_cpus": round_to(average_active, 9),
        "peak_active_logical_cpus": peak_active,
        "peak_busy_logical_cores_in_sample": peak_busy,
    })
}

type CounterReader =
    Box<dyn FnMut() -> Result<HashMap<u32, ProcCpuCounters>, ProcStatSamplingError>>;
type Clock = Box<dyn FnMut() -> f64>;
type Sleeper = Box<dyn FnMut(f64)>;

pub struct ProcStatComparisonSampler {
    pub cpu_ids: Vec<u32>,
    pub interval_sec: f64,
    pub baseline_sec: f64,
    pub active_threshold_percent: f64,
    pub proc_stat_path: PathBuf,
    pub baseline_intervals: Vec<ProcStatInterval>,
    pub measurement_intervals: Vec<ProcStatInterval>,
    reader: CounterReader,
    clock: Clock,
    sleep: Sleeper,
    previous: Option<HashMap<u32, ProcCpuCounters>>,
    previous_time: Option<f64>,
    next_sample_time: Option<f64>,
}

impl fmt::Debug for ProcStatComparisonSampler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("ProcStatComparisonSampler")
            .field("cpu_ids", &self.cpu_ids)
            .field("interval_sec", &self.interval_sec)
            .field("baseline_sec", &self.baseline_sec)
            .field("active_threshold_percent", &self.active_threshold_percent)
            .field("proc_stat_path", &self.proc_stat_path)
            .field("baseline_intervals", &self.baseline_intervals)
            .field("measurement_intervals", &self.measurement_intervals)
            .finish_non_exhaustive()
    }
}

impl ProcStatComparisonSampler {
    pub fn new(cpu_ids: Vec<u32>) -> Result<Self, ProcStatSamplingError> {
        Self::with_path(cpu_ids, 0.5, 5.0, 3.0, PathBuf::from("/proc/stat"))
    }

    pub fn with_path(
        cpu_ids: Vec<u32>,
        interval_sec: f64,
        baseline_sec: f64,
        active_threshold_percent: f64,
        proc_stat_path: PathBuf,
    ) -> Result<Self, ProcStatSamplingError> {
        let read_path = proc_stat_path.clone();
        let origin = Instant::now();
        Self::with_sources(
            cpu_ids,
            interval_sec,
            baseline_sec,
            active_threshold_percent,
            proc_stat_path,
            move || origin.elapsed().as_secs_f64(),
            |seconds| thread::sleep(Duration::from_secs_f64(seconds)),
            move || {
                let text = fs::read_to_string(&read_path)?;
                parse_proc_stat(&text)
            },
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_sources<C, S, R>(
        cpu_ids: Vec<u32>,
        interval_sec: f64,
        baseline_sec: f64,
        active_threshold_percent: f64,
        proc_stat_path: PathBuf,
        clock: C,
        sleep: S,
        reader: R,
    ) -> Result<Self, ProcStatSamplingError>
    where
        C: FnMut() -> f64 + 'static,
        S: FnMut(f64) + 'static,
        R: FnMut() -> Result<HashMap<u32, ProcCpuCounters>, ProcStatSamplingError> + 'static,
    {
        if cpu_ids.is_empty() {
            return Err(ProcStatSamplingError::new(
                "at least one logical CPU ID is required",
            ));
        }
        if !interval_sec.is_finite()
            || !baseline_sec.is_finite()
            || interval_sec <= 0.0
            || baseline_sec <= 0.0
        {
            return Err(ProcStatSamplingError::new(
                "sampling interval and baseline duration must be positive",
            ));
        }
        if !active_threshold_percent.is_finite() || active_threshold_percent < 0.0 {
            return Err(ProcStatSamplingError::new(
                "active CPU threshold must not be negative",
            ));
        }
        Ok(Self {
            cpu_ids,
            interval_sec,
            baseline_sec,
            active_threshold_percent,
            proc_stat_path,
            baseline_intervals: Vec::new(),
            measurement_intervals: Vec::new(),
            reader: Box::new(reader),
            clock: Box::new(clock),
            sleep: Box::new(sleep),
            previous: None,
            previous_time: None,
            next_sample_time: None,
        })
    }

    fn read(&mut self) -> Result<HashMap<u32, ProcCpuCounters>, ProcStatSamplingError> {
        (self.reader)()
    }

    fn now(&mut self) -> Result<f64, ProcStatSamplingError> {
        let value = (self.clock)();
        if value.is_finite() {
            Ok(value)
        } else {
            Err(ProcStatSamplingError::new(
                "proc-stat sampling clock returned a non-finite value",
            ))
        }
    }

    pub fn collect_baseline(
        &mut self,
        mut poll_callback: Option<&mut dyn FnMut() -> Result<(), ProcStatSamplingError>>,
        poll_interval_sec: Option<f64>,
    ) -> Result<(), ProcStatSamplingError> {
        if poll_interval_sec.map(|value| !value.is_finite() || value <= 0.0) == Some(true) {
            return Err(ProcStatSamplingError::new(
                "baseline callback interval must be positive",
            ));
        }
        let mut previous = self.read()?;
        let mut previous_time = self.now()?;
        let deadline = previous_time + self.baseline_sec;
        while previous_time < deadline {
            let sample_deadline = (previous_time + self.interval_sec).min(deadline);
            while self.now()? < sample_deadline {
                let remaining = sample_deadline - self.now()?;
                let sleep_for = poll_interval_sec.unwrap_or(remaining).min(remaining);
                (self.sleep)(sleep_for);
                if let Some(callback) = poll_callback.as_mut() {
                    callback()?;
                }
            }
            let current_time = self.now()?;
            if current_time <= previous_time {
                return Err(ProcStatSamplingError::new(
                    "proc-stat baseline clock did not advance",
                ));
            }
            let current = self.read()?;
            let interval = calculate_interval(
                &previous,
                &current,
                &self.cpu_ids,
                current_time - previous_time,
                self.active_threshold_percent,
            )?;
            self.baseline_intervals.push(interval);
            previous = current;
            previous_time = current_time;
        }
        Ok(())
    }

    pub fn start_measurement(&mut self) -> Result<(), ProcStatSamplingError> {
        self.previous = Some(self.read()?);
        let now = self.now()?;
        self.previous_time = Some(now);
        self.next_sample_time = Some(now + self.interval_sec);
        Ok(())
    }

    pub fn poll(&mut self) -> Result<(), ProcStatSamplingError> {
        let (previous_time, next_sample_time) = match (self.previous_time, self.next_sample_time) {
            (Some(previous), Some(next)) if self.previous.is_some() => (previous, next),
            _ => {
                return Err(ProcStatSamplingError::new(
                    "measurement sampling was not started",
                ))
            }
        };
        let current_time = self.now()?;
        if current_time < next_sample_time {
            return Ok(());
        }
        let current = self.read()?;
        let previous = self.previous.take().expect("validated above");
        if current_time > previous_time {
            self.measurement_intervals.push(calculate_interval(
                &previous,
                &current,
                &self.cpu_ids,
                current_time - previous_time,
                self.active_threshold_percent,
            )?);
        }
        self.previous = Some(current);
        self.previous_time = Some(current_time);
        self.next_sample_time = Some(current_time + self.interval_sec);
        Ok(())
    }

    pub fn finish(&mut self) -> Result<Value, ProcStatSamplingError> {
        let previous_time = self
            .previous_time
            .ok_or_else(|| ProcStatSamplingError::new("measurement sampling was not started"))?;
        let previous = self
            .previous
            .take()
            .ok_or_else(|| ProcStatSamplingError::new("measurement sampling was not started"))?;
        let current_time = self.now()?;
        let current = self.read()?;
        if current_time <= previous_time {
            return Err(ProcStatSamplingError::new(
                "proc-stat measurement clock did not advance",
            ));
        }
        self.measurement_intervals.push(calculate_interval(
            &previous,
            &current,
            &self.cpu_ids,
            current_time - previous_time,
            self.active_threshold_percent,
        )?);
        self.previous = Some(current);
        self.previous_time = Some(current_time);

        let baseline = summarize_intervals(&self.baseline_intervals);
        let measurement = summarize_intervals(&self.measurement_intervals);
        let baseline_average = optional_f64(&baseline["average_busy_logical_cores"])?;
        let measured_average = optional_f64(&measurement["average_busy_logical_cores"])?;
        let baseline_peak = optional_i64(&baseline["peak_active_logical_cpus"])?;
        let measured_peak = optional_i64(&measurement["peak_active_logical_cpus"])?;
        Ok(json!({
            "method": "tasktrace-style-proc-stat-sampling",
            "scope": "Host /proc/stat activity projected onto allocated logical CPU IDs; not container- or task-attributed.",
            "allocated_logical_cpu_ids": self.cpu_ids,
            "sample_interval_sec": self.interval_sec,
            "active_threshold_percent": self.active_threshold_percent,
            "baseline": baseline,
            "measurement": measurement,
            "baseline_subtracted_estimate": {
                "average_busy_logical_cores": match (measured_average, baseline_average) {
                    (Some(measured), Some(baseline)) => json_float(round_to((measured - baseline).max(0.0), 9)),
                    _ => Value::Null,
                },
                "peak_active_logical_cpus": match (measured_peak, baseline_peak) {
                    (Some(measured), Some(baseline)) => Value::from((measured - baseline).max(0)),
                    _ => Value::Null,
                },
            },
            "meaning": "Sampled host-pressure context only. The active-CPU threshold can overstate true simultaneous task execution after CPU migration or unrelated host work.",
        }))
    }

    pub fn write_samples(&self, path: &Path) -> Result<(), ProcStatSamplingError> {
        let intervals = |values: &[ProcStatInterval]| {
            values.iter().map(interval_payload).collect::<Vec<Value>>()
        };
        let payload = json!({
            "baseline": intervals(&self.baseline_intervals),
            "measurement": intervals(&self.measurement_intervals),
        });
        let mut encoded = serde_json::to_string_pretty(&payload)?;
        encoded.push('\n');
        fs::write(path, encoded)?;
        Ok(())
    }
}

fn interval_payload(interval: &ProcStatInterval) -> Value {
    json!({
        "duration_sec": interval.duration_sec,
        "busy_logical_cores": interval.busy_logical_cores,
        "active_logical_cpus": interval.active_logical_cpus,
        "utilization_percent_by_logical_cpu": interval.utilization_percent_by_logical_cpu,
    })
}

fn optional_f64(value: &Value) -> Result<Option<f64>, ProcStatSamplingError> {
    if value.is_null() {
        Ok(None)
    } else {
        value.as_f64().map(Some).ok_or_else(|| {
            ProcStatSamplingError::new("proc-stat report metric is not numeric or null")
        })
    }
}

fn optional_i64(value: &Value) -> Result<Option<i64>, ProcStatSamplingError> {
    if value.is_null() {
        Ok(None)
    } else {
        value.as_i64().map(Some).ok_or_else(|| {
            ProcStatSamplingError::new("proc-stat report metric is not an integer or null")
        })
    }
}

/// Attach differences to authoritative same-window cgroup/scheduler evidence.
pub fn add_reference_comparison(
    proc_report: &Value,
    cgroup_average_logical_cores: Option<f64>,
    scheduler_peak_simultaneous_logical_cores: Option<i64>,
    scheduler_peak_valid: bool,
) -> Result<Value, ProcStatSamplingError> {
    let mut report = proc_report
        .as_object()
        .cloned()
        .ok_or_else(|| ProcStatSamplingError::new("proc-stat report must be a JSON object"))?;
    let measurement = report
        .get("measurement")
        .and_then(Value::as_object)
        .ok_or_else(|| ProcStatSamplingError::new("proc-stat report has no measurement object"))?;
    let measured_average = optional_f64(
        measurement
            .get("average_busy_logical_cores")
            .ok_or_else(|| {
                ProcStatSamplingError::new(
                    "proc-stat measurement has no average_busy_logical_cores",
                )
            })?,
    )?;
    let sampled_peak = measurement
        .get("peak_active_logical_cpus")
        .cloned()
        .ok_or_else(|| {
            ProcStatSamplingError::new("proc-stat measurement has no peak_active_logical_cpus")
        })?;
    let subtracted = report
        .get("baseline_subtracted_estimate")
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ProcStatSamplingError::new(
                "proc-stat report has no baseline_subtracted_estimate object",
            )
        })?;
    let subtracted_average = optional_f64(
        subtracted
            .get("average_busy_logical_cores")
            .ok_or_else(|| {
                ProcStatSamplingError::new(
                    "baseline-subtracted estimate has no average_busy_logical_cores",
                )
            })?,
    )?;
    let difference = |value: Option<f64>| match (value, cgroup_average_logical_cores) {
        (Some(proc_value), Some(cgroup_value)) => {
            json_float(round_to(proc_value - cgroup_value, 9))
        }
        _ => Value::Null,
    };
    report.insert(
        "reference_comparison".to_owned(),
        json!({
            "cgroup_average_logical_cores": cgroup_average_logical_cores,
            "raw_proc_minus_cgroup_average_logical_cores": difference(measured_average),
            "baseline_subtracted_proc_minus_cgroup_average_logical_cores": difference(subtracted_average),
            "scheduler_peak_simultaneous_logical_cores": if scheduler_peak_valid {
                scheduler_peak_simultaneous_logical_cores.map(Value::from).unwrap_or(Value::Null)
            } else {
                Value::Null
            },
            "sampled_proc_peak_active_logical_cpus": sampled_peak,
            "scheduler_peak_valid": scheduler_peak_valid,
            "peak_values_are_equivalent": false,
        }),
    );
    Ok(Value::Object(report))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::collections::VecDeque;
    use std::rc::Rc;

    fn stat(
        cpu0_user: i128,
        cpu0_idle: i128,
        cpu1_user: i128,
        cpu1_idle: i128,
    ) -> HashMap<u32, ProcCpuCounters> {
        parse_proc_stat(&format!(
            "cpu0 {cpu0_user} 0 0 {cpu0_idle} 0 0 0 0\n\
             cpu1 {cpu1_user} 0 0 {cpu1_idle} 0 0 0 0\n"
        ))
        .unwrap()
    }

    #[test]
    fn interval_matches_tasktrace_active_cpu_definition() {
        let interval = calculate_interval(
            &stat(0, 100, 0, 100),
            &stat(25, 175, 0, 200),
            &[0, 1],
            0.5,
            3.0,
        )
        .unwrap();
        assert_eq!(
            interval.utilization_percent_by_logical_cpu,
            BTreeMap::from([("0".to_owned(), 25.0), ("1".to_owned(), 0.0)])
        );
        assert_eq!(interval.busy_logical_cores, 0.25);
        assert_eq!(interval.active_logical_cpus, 1);
    }

    #[test]
    fn summary_weights_terminal_partial_bucket() {
        let first = calculate_interval(
            &stat(0, 100, 0, 100),
            &stat(25, 175, 0, 200),
            &[0, 1],
            0.5,
            3.0,
        )
        .unwrap();
        let terminal = calculate_interval(
            &stat(25, 175, 0, 200),
            &stat(25, 275, 25, 200),
            &[0, 1],
            0.25,
            3.0,
        )
        .unwrap();
        let summary = summarize_intervals(&[first, terminal]);
        assert_eq!(summary["duration_sec"], 0.75);
        assert_eq!(summary["average_busy_logical_cores"], 0.5);
        assert_eq!(summary["peak_active_logical_cpus"], 1);
    }

    #[test]
    fn sampler_reports_subtracted_estimate_and_reference_delta() {
        let clock = Rc::new(Cell::new(0.0));
        let sleep_clock = Rc::clone(&clock);
        let read_clock = Rc::clone(&clock);
        let snapshots = Rc::new(RefCell::new(VecDeque::from([
            stat(0, 100, 0, 100),
            stat(0, 150, 0, 150),
            stat(0, 150, 0, 150),
            stat(50, 150, 0, 200),
            stat(75, 175, 0, 225),
        ])));
        let read_snapshots = Rc::clone(&snapshots);
        let mut sampler = ProcStatComparisonSampler::with_sources(
            vec![0, 1],
            0.5,
            0.5,
            3.0,
            PathBuf::from("/proc/stat"),
            move || read_clock.get(),
            move |seconds| sleep_clock.set(sleep_clock.get() + seconds),
            move || {
                read_snapshots
                    .borrow_mut()
                    .pop_front()
                    .ok_or_else(|| ProcStatSamplingError::new("test snapshot sequence exhausted"))
            },
        )
        .unwrap();
        sampler.collect_baseline(None, None).unwrap();
        sampler.start_measurement().unwrap();
        clock.set(clock.get() + 0.5);
        sampler.poll().unwrap();
        clock.set(clock.get() + 0.25);
        let report = sampler.finish().unwrap();
        let report = add_reference_comparison(&report, Some(0.75), Some(1), true).unwrap();
        assert_eq!(report["baseline"]["average_busy_logical_cores"], 0.0);
        assert_eq!(
            report["measurement"]["average_busy_logical_cores"],
            0.833333333
        );
        assert_eq!(
            report["baseline_subtracted_estimate"]["average_busy_logical_cores"],
            0.833333333
        );
        assert_eq!(
            report["reference_comparison"]["raw_proc_minus_cgroup_average_logical_cores"],
            0.083333333
        );
        assert_eq!(
            report["reference_comparison"]["peak_values_are_equivalent"],
            false
        );
    }

    #[test]
    fn baseline_can_poll_faster_observer() {
        let clock = Rc::new(Cell::new(0.0));
        let sleep_clock = Rc::clone(&clock);
        let read_clock = Rc::clone(&clock);
        let snapshots = Rc::new(RefCell::new(VecDeque::from([
            stat(0, 100, 0, 100),
            stat(0, 150, 0, 150),
        ])));
        let read_snapshots = Rc::clone(&snapshots);
        let mut sampler = ProcStatComparisonSampler::with_sources(
            vec![0, 1],
            0.5,
            0.5,
            3.0,
            PathBuf::from("/proc/stat"),
            move || read_clock.get(),
            move |seconds| sleep_clock.set(sleep_clock.get() + seconds),
            move || {
                read_snapshots
                    .borrow_mut()
                    .pop_front()
                    .ok_or_else(|| ProcStatSamplingError::new("test snapshot sequence exhausted"))
            },
        )
        .unwrap();
        let callback_times = Rc::new(RefCell::new(Vec::new()));
        let callback_values = Rc::clone(&callback_times);
        let callback_clock = Rc::clone(&clock);
        let mut callback = move || {
            callback_values.borrow_mut().push(callback_clock.get());
            Ok(())
        };
        sampler
            .collect_baseline(Some(&mut callback), Some(0.1))
            .unwrap();
        assert_eq!(callback_times.borrow().len(), 5);
        assert_eq!(*callback_times.borrow().last().unwrap(), 0.5);
    }

    #[test]
    fn baseline_callback_and_non_finite_clock_fail_closed() {
        let clock = Rc::new(Cell::new(0.0));
        let sleep_clock = Rc::clone(&clock);
        let read_clock = Rc::clone(&clock);
        let mut sampler = ProcStatComparisonSampler::with_sources(
            vec![0, 1],
            0.5,
            0.5,
            3.0,
            PathBuf::from("/proc/stat"),
            move || read_clock.get(),
            move |seconds| sleep_clock.set(sleep_clock.get() + seconds),
            move || Ok(stat(0, 100, 0, 100)),
        )
        .unwrap();
        let mut cancelled = || Err(ProcStatSamplingError::new("cancelled by test"));
        assert!(sampler
            .collect_baseline(Some(&mut cancelled), Some(0.1))
            .unwrap_err()
            .to_string()
            .contains("cancelled by test"));

        let mut invalid_clock = ProcStatComparisonSampler::with_sources(
            vec![0],
            0.5,
            0.5,
            3.0,
            PathBuf::from("/proc/stat"),
            || f64::NAN,
            |_| {},
            || Ok(stat(0, 100, 0, 100)),
        )
        .unwrap();
        assert!(invalid_clock
            .start_measurement()
            .unwrap_err()
            .to_string()
            .contains("non-finite"));
    }

    #[test]
    fn measurement_finish_rejects_non_advancing_clock() {
        let snapshots = Rc::new(RefCell::new(VecDeque::from([
            stat(0, 100, 0, 100),
            stat(1, 101, 0, 100),
        ])));
        let read_snapshots = Rc::clone(&snapshots);
        let mut sampler = ProcStatComparisonSampler::with_sources(
            vec![0],
            0.5,
            0.5,
            3.0,
            PathBuf::from("/proc/stat"),
            || 0.0,
            |_| {},
            move || {
                read_snapshots
                    .borrow_mut()
                    .pop_front()
                    .ok_or_else(|| ProcStatSamplingError::new("test snapshots exhausted"))
            },
        )
        .unwrap();
        sampler.start_measurement().unwrap();
        assert!(sampler
            .finish()
            .unwrap_err()
            .to_string()
            .contains("clock did not advance"));
    }

    #[test]
    fn strict_parser_and_state_guards_fail_closed() {
        assert!(parse_proc_stat("cpu all aggregate\n").is_err());
        assert!(parse_proc_stat("cpu0 1 2 3\n").is_err());
        assert!(calculate_interval(
            &stat(10, 100, 0, 100),
            &stat(9, 101, 0, 101),
            &[0],
            0.5,
            3.0,
        )
        .unwrap_err()
        .to_string()
        .contains("moved backwards"));
        let mut sampler = ProcStatComparisonSampler::new(vec![0]).unwrap();
        assert!(sampler
            .poll()
            .unwrap_err()
            .to_string()
            .contains("not started"));
        assert!(sampler
            .finish()
            .unwrap_err()
            .to_string()
            .contains("not started"));
    }
}
