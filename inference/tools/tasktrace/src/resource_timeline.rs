// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dependency-free SVG rendering for aligned cgroup CPU and memory evidence.
//!
//! This is the native implementation of the public `report resource-timeline`
//! leaf.  The exact cgroup peak remains an untimed annotation: only sampled
//! memory values determine the chart's timed y-axis.

use serde_json::{Map, Value};
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::io::{self, Write};
use std::path::{Path, PathBuf};

pub const TIMESERIES_SCHEMA_VERSION: &str = "container-cgroup-resource-timeseries.v1";
const MIB: f64 = (1024 * 1024) as f64;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ResourceTimelineError(String);

impl ResourceTimelineError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ResourceTimelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for ResourceTimelineError {}

fn load_json(path: &Path) -> Result<Map<String, Value>, ResourceTimelineError> {
    let text = fs::read_to_string(path).map_err(|error| {
        ResourceTimelineError::new(format!(
            "unable to read JSON report {}: {error}",
            path.display()
        ))
    })?;
    let value: Value = serde_json::from_str(&text).map_err(|error| {
        ResourceTimelineError::new(format!(
            "unable to read JSON report {}: {error}",
            path.display()
        ))
    })?;
    value.as_object().cloned().ok_or_else(|| {
        ResourceTimelineError::new(format!("expected a JSON object in {}", path.display()))
    })
}

fn py_truthy(value: &Value) -> bool {
    match value {
        Value::Null => false,
        Value::Bool(value) => *value,
        Value::Number(value) => value.as_f64().is_some_and(|value| value != 0.0),
        Value::String(value) => !value.is_empty(),
        Value::Array(value) => !value.is_empty(),
        Value::Object(value) => !value.is_empty(),
    }
}

fn python_string(value: &Value) -> String {
    match value {
        Value::Null => "None".to_owned(),
        Value::Bool(true) => "True".to_owned(),
        Value::Bool(false) => "False".to_owned(),
        Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn python_repr(value: Option<&Value>) -> String {
    match value {
        None | Some(Value::Null) => "None".to_owned(),
        Some(Value::Bool(true)) => "True".to_owned(),
        Some(Value::Bool(false)) => "False".to_owned(),
        Some(Value::String(value)) => {
            let escaped = value.replace('\\', "\\\\").replace('\'', "\\'");
            format!("'{escaped}'")
        }
        Some(other) => other.to_string(),
    }
}

fn nonnegative_finite_number(
    value: Option<&Value>,
    field: &str,
) -> Result<f64, ResourceTimelineError> {
    let number = value
        .and_then(Value::as_f64)
        .filter(|number| number.is_finite() && *number >= 0.0)
        .ok_or_else(|| {
            ResourceTimelineError::new(format!("{field} must be a finite nonnegative number"))
        })?;
    Ok(number)
}

fn select_memory(report: &Map<String, Value>) -> Option<&Value> {
    report
        .get("container_memory")
        .filter(|value| py_truthy(value))
        .or_else(|| report.get("memory"))
}

fn select_cpu(report: &Map<String, Value>) -> Option<&Value> {
    report
        .get("container_cpu")
        .filter(|value| py_truthy(value))
        .or_else(|| report.get("cpu").filter(|value| py_truthy(value)))
}

fn invalid_reasons(memory: &Map<String, Value>) -> String {
    let Some(reasons) = memory
        .get("timeline_invalid_reasons")
        .filter(|value| py_truthy(value))
    else {
        return String::new();
    };
    match reasons {
        Value::Array(reasons) => reasons
            .iter()
            .map(python_string)
            .collect::<Vec<_>>()
            .join("; "),
        Value::String(reason) => reason
            .chars()
            .map(|character| character.to_string())
            .collect::<Vec<_>>()
            .join("; "),
        Value::Object(reasons) => reasons.keys().cloned().collect::<Vec<_>>().join("; "),
        other => python_string(other),
    }
}

fn validate_resource_timeline(
    memory: &Map<String, Value>,
    timeline: &Map<String, Value>,
) -> Result<(), ResourceTimelineError> {
    if memory.get("timeline_valid_for_shape") == Some(&Value::Bool(false)) {
        let detail = invalid_reasons(memory);
        return Err(ResourceTimelineError::new(format!(
            "container memory timeline is invalid for shape reporting{}",
            if detail.is_empty() {
                String::new()
            } else {
                format!(": {detail}")
            }
        )));
    }

    let interval =
        nonnegative_finite_number(timeline.get("sample_interval_sec"), "sample_interval_sec")?;
    if interval == 0.0 {
        return Err(ResourceTimelineError::new(
            "sample_interval_sec must be positive",
        ));
    }
    let samples = timeline
        .get("samples")
        .and_then(Value::as_array)
        .filter(|samples| samples.len() >= 2)
        .ok_or_else(|| {
            ResourceTimelineError::new("resource time series requires at least two samples")
        })?;

    let mut first_elapsed = None;
    let mut previous_elapsed = None;
    for (index, sample) in samples.iter().enumerate() {
        let sample = sample.as_object().ok_or_else(|| {
            ResourceTimelineError::new(format!("samples[{index}] must be an object"))
        })?;
        let elapsed = nonnegative_finite_number(
            sample.get("elapsed_sec"),
            &format!("samples[{index}].elapsed_sec"),
        )?;
        if previous_elapsed.is_some_and(|previous| elapsed <= previous) {
            return Err(ResourceTimelineError::new(
                "sample elapsed_sec values must be strictly increasing",
            ));
        }
        if previous_elapsed.is_some_and(|previous| elapsed - previous > 2.0 * interval) {
            return Err(ResourceTimelineError::new(format!(
                "samples[{index}] exceeds twice the requested sample interval"
            )));
        }
        first_elapsed.get_or_insert(elapsed);
        previous_elapsed = Some(elapsed);
        nonnegative_finite_number(
            sample.get("memory_current_bytes"),
            &format!("samples[{index}].memory_current_bytes"),
        )?;
        if sample.get("cpu_average_logical_cores") != Some(&Value::Null) {
            nonnegative_finite_number(
                sample.get("cpu_average_logical_cores"),
                &format!("samples[{index}].cpu_average_logical_cores"),
            )?;
        }
        let memory_stat = sample
            .get("memory_stat")
            .and_then(Value::as_object)
            .ok_or_else(|| {
                ResourceTimelineError::new(format!(
                    "samples[{index}].memory_stat must be an object"
                ))
            })?;
        for key in ["anon", "file", "kernel"] {
            if let Some(value) = memory_stat.get(key) {
                nonnegative_finite_number(
                    Some(value),
                    &format!("samples[{index}].memory_stat.{key}"),
                )?;
            }
        }
    }

    let first_elapsed = first_elapsed.expect("two validated samples have a first value");
    let last_elapsed = previous_elapsed.expect("two validated samples have a last value");
    let markers = timeline
        .get("markers")
        .and_then(Value::as_array)
        .ok_or_else(|| ResourceTimelineError::new("resource time series markers must be a list"))?;
    let mut previous_marker_elapsed = None;
    let mut command_start = None;
    let mut command_end = None;
    for (index, marker) in markers.iter().enumerate() {
        let marker = marker.as_object().ok_or_else(|| {
            ResourceTimelineError::new(format!("markers[{index}] must be an object"))
        })?;
        let name = marker
            .get("name")
            .and_then(Value::as_str)
            .filter(|name| !name.is_empty())
            .ok_or_else(|| {
                ResourceTimelineError::new(format!("markers[{index}].name must be nonempty"))
            })?;
        let elapsed = nonnegative_finite_number(
            marker.get("elapsed_sec"),
            &format!("markers[{index}].elapsed_sec"),
        )?;
        if elapsed < first_elapsed || elapsed > last_elapsed {
            return Err(ResourceTimelineError::new(format!(
                "markers[{index}].elapsed_sec falls outside the sample window"
            )));
        }
        if previous_marker_elapsed.is_some_and(|previous| elapsed < previous) {
            return Err(ResourceTimelineError::new(
                "marker elapsed_sec values must be monotonic",
            ));
        }
        previous_marker_elapsed = Some(elapsed);
        match name {
            "command_start" => command_start = Some(elapsed),
            "command_end" => command_end = Some(elapsed),
            _ => {}
        }
    }
    if command_start
        .zip(command_end)
        .is_some_and(|(start, end)| end < start)
    {
        return Err(ResourceTimelineError::new(
            "command_end marker precedes command_start",
        ));
    }

    if let Some(peak) = memory
        .get("lifetime")
        .and_then(Value::as_object)
        .and_then(|lifetime| lifetime.get("peak_bytes"))
        .filter(|peak| !peak.is_null())
    {
        nonnegative_finite_number(Some(peak), "memory.lifetime.peak_bytes")?;
    }
    Ok(())
}

/// Load a profile/pipeline memory summary and its external raw samples.
type ResourceTimeline = (Map<String, Value>, Map<String, Value>);

pub fn load_resource_timeline(
    report_path: &Path,
) -> Result<ResourceTimeline, ResourceTimelineError> {
    let report = load_json(report_path)?;
    let memory = select_memory(&report)
        .and_then(Value::as_object)
        .ok_or_else(|| {
            ResourceTimelineError::new(format!(
                "report has no container memory profile: {}",
                report_path.display()
            ))
        })?;
    let raw_samples_path = memory
        .get("samples_path")
        .and_then(Value::as_str)
        .filter(|path| !path.is_empty())
        .ok_or_else(|| {
            ResourceTimelineError::new("container memory profile has no samples_path")
        })?;
    let raw_samples_path = Path::new(raw_samples_path);
    let samples_path = if raw_samples_path.is_absolute() {
        raw_samples_path.to_owned()
    } else {
        report_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join(raw_samples_path)
    };
    let timeline = load_json(&samples_path)?;
    if timeline.get("schema_version").and_then(Value::as_str) != Some(TIMESERIES_SCHEMA_VERSION) {
        return Err(ResourceTimelineError::new(format!(
            "unsupported resource time-series schema: {}",
            python_repr(timeline.get("schema_version"))
        )));
    }
    validate_resource_timeline(memory, &timeline)?;

    if let Some(cpu) = select_cpu(&report) {
        let cpu = cpu
            .as_object()
            .ok_or_else(|| ResourceTimelineError::new("container CPU summary must be an object"))?;
        if let Some(allocated) = cpu
            .get("allocated_logical_cpus")
            .filter(|value| !value.is_null())
        {
            nonnegative_finite_number(Some(allocated), "allocated_logical_cpus")?;
        }
    }
    Ok((report, timeline))
}

fn escape_html(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len());
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

fn svg_text(x: f64, y: f64, value: &str, attributes: &[(&str, &str)]) -> String {
    let attributes = attributes
        .iter()
        .map(|(key, value)| format!("{}=\"{}\"", key.replace('_', "-"), escape_html(value)))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "<text x=\"{x:.1}\" y=\"{y:.1}\" {attributes}>{}</text>",
        escape_html(value)
    )
}

fn numeric(value: Option<&Value>) -> Option<f64> {
    value.and_then(Value::as_f64)
}

fn sample_elapsed(sample: &Map<String, Value>) -> f64 {
    numeric(sample.get("elapsed_sec")).expect("validated elapsed value")
}

fn polyline<F, X, Y>(
    samples: &[&Map<String, Value>],
    value_for: F,
    x_for: &X,
    y_for: &Y,
    color: &str,
    width: f64,
) -> String
where
    F: Fn(&Map<String, Value>) -> Option<f64>,
    X: Fn(f64) -> f64,
    Y: Fn(f64) -> f64,
{
    let points = samples
        .iter()
        .filter_map(|sample| {
            value_for(sample)
                .map(|value| format!("{:.2},{:.2}", x_for(sample_elapsed(sample)), y_for(value)))
        })
        .collect::<Vec<_>>();
    if points.len() < 2 {
        return String::new();
    }
    format!(
        "<polyline points=\"{}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"{width:.1}\"/>",
        points.join(" ")
    )
}

fn interval_steps<X, Y>(
    samples: &[&Map<String, Value>],
    x_for: &X,
    y_for: &Y,
    color: &str,
    width: f64,
) -> String
where
    X: Fn(f64) -> f64,
    Y: Fn(f64) -> f64,
{
    let mut lines = Vec::new();
    let mut previous_value = None;
    for pair in samples.windows(2) {
        let previous = pair[0];
        let current = pair[1];
        let Some(value) = numeric(current.get("cpu_average_logical_cores")) else {
            previous_value = None;
            continue;
        };
        let x_start = x_for(sample_elapsed(previous));
        let x_end = x_for(sample_elapsed(current));
        let y = y_for(value);
        if let Some(previous_value) = previous_value {
            let previous_y = y_for(previous_value);
            lines.push(format!(
                "<line class=\"cpu-interval-transition\" x1=\"{x_start:.2}\" y1=\"{previous_y:.2}\" x2=\"{x_start:.2}\" y2=\"{y:.2}\" stroke=\"{color}\" stroke-width=\"{width:.1}\"/>"
            ));
        }
        lines.push(format!(
            "<line class=\"cpu-interval\" x1=\"{x_start:.2}\" y1=\"{y:.2}\" x2=\"{x_end:.2}\" y2=\"{y:.2}\" stroke=\"{color}\" stroke-width=\"{width:.1}\"/>"
        ));
        previous_value = Some(value);
    }
    lines.join("\n")
}

fn integer_number(value: Option<&Value>) -> Option<f64> {
    let number = value?.as_number()?;
    let rendered = number.to_string();
    (!rendered.contains(['.', 'e', 'E']))
        .then(|| number.as_f64())
        .flatten()
}

fn value_display(value: Option<&Value>) -> String {
    value
        .map(python_string)
        .unwrap_or_else(|| "None".to_owned())
}

/// Render aligned interval-CPU and charged-memory panels.
pub fn render_resource_timeline_svg(
    report_path: &Path,
    output_path: &Path,
) -> Result<(), ResourceTimelineError> {
    let (report, timeline) = load_resource_timeline(report_path)?;
    let memory = select_memory(&report)
        .and_then(Value::as_object)
        .expect("loader validated the memory profile");
    let samples = timeline["samples"]
        .as_array()
        .expect("loader validated samples")
        .iter()
        .map(|sample| sample.as_object().expect("loader validated sample object"))
        .collect::<Vec<_>>();
    let markers = timeline["markers"]
        .as_array()
        .expect("loader validated markers");
    let duration = samples
        .iter()
        .map(|sample| sample_elapsed(sample))
        .fold(0.0_f64, f64::max)
        .max(0.001);
    let cpu_values = samples
        .iter()
        .filter_map(|sample| numeric(sample.get("cpu_average_logical_cores")))
        .collect::<Vec<_>>();
    let memory_values = samples
        .iter()
        .map(|sample| numeric(sample.get("memory_current_bytes")).expect("validated memory"))
        .collect::<Vec<_>>();
    let cpu_capacity = select_cpu(&report)
        .and_then(Value::as_object)
        .and_then(|cpu| numeric(cpu.get("allocated_logical_cpus")))
        .unwrap_or(0.0);
    let cpu_max = cpu_values
        .iter()
        .copied()
        .chain([cpu_capacity, 1.0])
        .fold(0.0_f64, f64::max)
        * 1.08;
    let mut memory_max_mib = memory_values.iter().copied().fold(0.0_f64, f64::max) / MIB;

    let lifetime_peak = memory
        .get("lifetime")
        .and_then(Value::as_object)
        .and_then(|lifetime| numeric(lifetime.get("peak_bytes")));
    let lifetime_peak_text = if let Some(lifetime_peak) = lifetime_peak {
        let peak_status = if memory.get("peak_valid") == Some(&Value::Bool(true)) {
            "exact"
        } else {
            "invalid"
        };
        format!(
            "Untimed controlled-finish-gate peak: {:.1} MiB · {peak_status} (annotated only; not on the timed y-axis)",
            lifetime_peak / MIB
        )
    } else {
        "Untimed controlled-finish-gate peak: unavailable".to_owned()
    };

    let margin = memory
        .get("observed_limit_margin")
        .and_then(Value::as_object);
    let configured_limit =
        margin.and_then(|margin| integer_number(margin.get("configured_limit_bytes")));
    let remaining_to_limit =
        margin.and_then(|margin| integer_number(margin.get("remaining_to_limit_bytes")));
    let limit_margin_text = if let (Some(configured), Some(remaining)) =
        (configured_limit, remaining_to_limit)
    {
        if margin.and_then(|margin| margin.get("arithmetic_valid")) == Some(&Value::Bool(true)) {
            let fraction_text = margin
                .and_then(|margin| numeric(margin.get("peak_fraction_of_limit")))
                .map(|fraction| format!(" · peak/limit={:.1}%", fraction * 100.0))
                .unwrap_or_default();
            format!(
                "Leaf memory.max: {:.1} MiB · observed margin: {:.1} MiB{fraction_text} · not a recommendation",
                configured / MIB,
                remaining / MIB
            )
        } else {
            format!(
                "Leaf memory.max: {:.1} MiB · observed margin invalid because the peak is invalid",
                configured / MIB
            )
        }
    } else if let Some(configured) = configured_limit {
        format!(
            "Leaf memory.max: {:.1} MiB · observed margin invalid because the peak is invalid",
            configured / MIB
        )
    } else if margin.and_then(|margin| margin.get("configured_limit_state"))
        == Some(&Value::String("unlimited".to_owned()))
    {
        "Leaf memory.max: unlimited · observed margin unavailable".to_owned()
    } else {
        "Leaf memory.max and observed margin: unavailable".to_owned()
    };
    memory_max_mib = (memory_max_mib * 1.08).max(1.0);

    let (width, height) = (1080, 720);
    let (left, right) = (86, 38);
    let chart_width = width - left - right;
    let (cpu_top, panel_height) = (132, 220);
    let memory_top = 420;
    let x_for = |elapsed: f64| left as f64 + chart_width as f64 * elapsed / duration;
    let cpu_y =
        |value: f64| cpu_top as f64 + panel_height as f64 - panel_height as f64 * value / cpu_max;
    let memory_y_bytes = |value: f64| {
        let value_mib = value / MIB;
        memory_top as f64 + panel_height as f64 - panel_height as f64 * value_mib / memory_max_mib
    };

    let report_name = report_path
        .file_name()
        .unwrap_or_else(|| OsStr::new(""))
        .to_string_lossy();
    let timeline_valid = match memory.get("timeline_valid_for_shape") {
        None => true,
        Some(value) => py_truthy(value),
    };
    let natural_peak = memory
        .get("natural_peak_trustworthy")
        .is_some_and(py_truthy);
    let mut svg = vec![
        format!(
            "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\" data-timed-memory-max-mib=\"{memory_max_mib:.6}\">"
        ),
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>".to_owned(),
        svg_text(
            28.0,
            34.0,
            "Sandbox CPU and memory over time",
            &[("fill", "#111827"), ("font_size", "20"), ("font_weight", "bold")],
        ),
        svg_text(
            28.0,
            58.0,
            &format!(
                "{report_name} · cgroup subtree · {} samples · interval={}s · timeline={} · natural-peak={}",
                samples.len(),
                value_display(timeline.get("sample_interval_sec")),
                if timeline_valid { "valid" } else { "invalid" },
                if natural_peak { "trustworthy" } else { "constrained" }
            ),
            &[("fill", "#4b5563"), ("font_size", "12")],
        ),
        svg_text(
            28.0,
            80.0,
            &lifetime_peak_text,
            &[("fill", "#6b7280"), ("font_size", "11")],
        ),
        svg_text(
            28.0,
            100.0,
            &limit_margin_text,
            &[("fill", "#6b7280"), ("font_size", "11")],
        ),
        svg_text(
            left as f64,
            (cpu_top - 18) as f64,
            "Sampled interval CPU demand",
            &[("fill", "#111827"), ("font_size", "14"), ("font_weight", "bold")],
        ),
        svg_text(
            left as f64,
            (memory_top - 18) as f64,
            "Cgroup-charged memory (MiB)",
            &[("fill", "#111827"), ("font_size", "14"), ("font_weight", "bold")],
        ),
    ];
    for top in [cpu_top, memory_top] {
        svg.push(format!(
            "<rect x=\"{left}\" y=\"{top}\" width=\"{chart_width}\" height=\"{panel_height}\" fill=\"#f9fafb\" stroke=\"#d1d5db\"/>"
        ));
    }
    for index in 0..6 {
        let elapsed = duration * index as f64 / 5.0;
        let x = x_for(elapsed);
        for top in [cpu_top, memory_top] {
            svg.push(format!(
                "<line x1=\"{x:.2}\" y1=\"{top}\" x2=\"{x:.2}\" y2=\"{}\" stroke=\"#e5e7eb\" stroke-width=\"1\"/>",
                top + panel_height
            ));
        }
        svg.push(svg_text(
            x - 10.0,
            (memory_top + panel_height + 22) as f64,
            &format!("{elapsed:.2}"),
            &[("fill", "#4b5563"), ("font_size", "10")],
        ));
    }
    for index in 0..5 {
        let fraction = index as f64 / 4.0;
        let cpu_value = cpu_max * (1.0 - fraction);
        let memory_value = memory_max_mib * (1.0 - fraction);
        let cpu_line_y = cpu_top as f64 + panel_height as f64 * fraction;
        let memory_line_y = memory_top as f64 + panel_height as f64 * fraction;
        svg.extend([
            format!(
                "<line x1=\"{left}\" y1=\"{cpu_line_y:.2}\" x2=\"{}\" y2=\"{cpu_line_y:.2}\" stroke=\"#e5e7eb\"/>",
                left + chart_width
            ),
            format!(
                "<line x1=\"{left}\" y1=\"{memory_line_y:.2}\" x2=\"{}\" y2=\"{memory_line_y:.2}\" stroke=\"#e5e7eb\"/>",
                left + chart_width
            ),
            svg_text(
                18.0,
                cpu_line_y + 4.0,
                &format!("{cpu_value:.2}"),
                &[("fill", "#4b5563"), ("font_size", "10")],
            ),
            svg_text(
                18.0,
                memory_line_y + 4.0,
                &format!("{memory_value:.1}"),
                &[("fill", "#4b5563"), ("font_size", "10")],
            ),
        ]);
    }
    svg.push(interval_steps(&samples, &x_for, &cpu_y, "#2563eb", 2.4));
    svg.push(polyline(
        &samples,
        |sample| numeric(sample.get("memory_current_bytes")),
        &x_for,
        &memory_y_bytes,
        "#dc2626",
        2.5,
    ));
    for (key, color) in [
        ("anon", "#f59e0b"),
        ("file", "#16a34a"),
        ("kernel", "#7c3aed"),
    ] {
        svg.push(polyline(
            &samples,
            |sample| {
                sample
                    .get("memory_stat")
                    .and_then(Value::as_object)
                    .and_then(|memory_stat| numeric(memory_stat.get(key)))
            },
            &x_for,
            &memory_y_bytes,
            color,
            1.4,
        ));
    }
    for marker in markers {
        let marker = marker.as_object().expect("loader validated marker object");
        let name = marker["name"]
            .as_str()
            .expect("loader validated marker name");
        let color = match name {
            "command_start" => "#059669",
            "command_end" => "#9333ea",
            _ => continue,
        };
        let elapsed = numeric(marker.get("elapsed_sec")).expect("loader validated marker time");
        let x = x_for(elapsed);
        svg.push(format!(
            "<line x1=\"{x:.2}\" y1=\"{cpu_top}\" x2=\"{x:.2}\" y2=\"{}\" stroke=\"{color}\" stroke-width=\"1.5\" stroke-dasharray=\"5 4\"/>",
            memory_top + panel_height
        ));
        svg.push(svg_text(
            x + 4.0,
            (cpu_top + 14) as f64,
            &name.replace('_', " "),
            &[("fill", color), ("font_size", "10")],
        ));
    }
    svg.extend([
        svg_text(
            left as f64,
            (height - 24) as f64,
            "Elapsed seconds (shared monotonic clock)",
            &[("fill", "#374151"), ("font_size", "11")],
        ),
        svg_text(
            (left + 280) as f64,
            (cpu_top - 18) as f64,
            "blue: interval-average logical cores",
            &[("fill", "#2563eb"), ("font_size", "11")],
        ),
        svg_text(
            (left + 250) as f64,
            (memory_top - 18) as f64,
            "red: total",
            &[("fill", "#dc2626"), ("font_size", "11")],
        ),
        svg_text(
            (left + 325) as f64,
            (memory_top - 18) as f64,
            "orange: anon",
            &[("fill", "#f59e0b"), ("font_size", "11")],
        ),
        svg_text(
            (left + 420) as f64,
            (memory_top - 18) as f64,
            "green: file",
            &[("fill", "#16a34a"), ("font_size", "11")],
        ),
        svg_text(
            (left + 505) as f64,
            (memory_top - 18) as f64,
            "purple: kernel",
            &[("fill", "#7c3aed"), ("font_size", "11")],
        ),
        "</svg>".to_owned(),
    ]);

    if let Some(parent) = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
    {
        fs::create_dir_all(parent).map_err(|error| {
            ResourceTimelineError::new(format!(
                "unable to create output directory {}: {error}",
                parent.display()
            ))
        })?;
    }
    let output = svg
        .into_iter()
        .filter(|part| !part.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    fs::write(output_path, output).map_err(|error| {
        ResourceTimelineError::new(format!(
            "unable to write resource timeline {}: {error}",
            output_path.display()
        ))
    })
}

#[derive(Debug)]
struct Cli {
    report: PathBuf,
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

fn parse_cli(arguments: &[OsString]) -> Result<CliParse, String> {
    let mut positional = Vec::new();
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
        if options && argument == "--output" {
            output = Some(PathBuf::from(option_value(
                arguments, &mut index, "--output",
            )?));
        } else if options {
            if let Some(value) = argument
                .to_str()
                .and_then(|argument| argument.strip_prefix("--output="))
            {
                if value.is_empty() {
                    return Err("argument --output: expected one argument".to_owned());
                }
                output = Some(PathBuf::from(value));
            } else if argument
                .to_str()
                .is_some_and(|argument| argument.starts_with('-'))
            {
                return Err(format!(
                    "unrecognized arguments: {}",
                    argument.to_string_lossy()
                ));
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
        missing.push("report");
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
    if positional.len() > 1 {
        return Err(format!(
            "unrecognized arguments: {}",
            positional[1..]
                .iter()
                .map(|argument| argument.to_string_lossy())
                .collect::<Vec<_>>()
                .join(" ")
        ));
    }
    Ok(CliParse::Run(Cli {
        report: positional.pop().expect("validated positional report"),
        output: output.expect("validated output"),
    }))
}

fn print_usage(public_program: &str, output: &mut dyn Write) {
    let _ = writeln!(
        output,
        "usage: {public_program} [-h] --output OUTPUT report"
    );
}

fn print_help(public_program: &str) {
    print_usage(public_program, &mut io::stdout());
    println!();
    println!("Render one profile's aligned cgroup CPU and memory timeline.");
    println!();
    println!("positional arguments:");
    println!("  report           pipeline_report.json or profile_report.json");
    println!();
    println!("options:");
    println!("  -h, --help       show this help message and exit");
    println!("  --output OUTPUT  New SVG output path.");
}

fn cli_error(public_program: &str, message: &str) -> i32 {
    print_usage(public_program, &mut io::stderr());
    eprintln!("{public_program}: error: {message}");
    2
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
    if options.output.exists() {
        return cli_error(
            public_program,
            &format!("output path already exists: {}", options.output.display()),
        );
    }
    if let Err(error) = render_resource_timeline_svg(&options.report, &options.output) {
        return cli_error(public_program, &error.to_string());
    }
    println!("Resource timeline: {}", options.output.display());
    0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::TempDir;

    fn fixture() -> (TempDir, PathBuf, PathBuf) {
        let directory = TempDir::new().unwrap();
        let samples_path = directory.path().join("resource_timeseries.json");
        fs::write(
            &samples_path,
            serde_json::to_string(&json!({
                "schema_version": TIMESERIES_SCHEMA_VERSION,
                "sample_interval_sec": 0.5,
                "markers": [
                    {"name": "command_start", "elapsed_sec": 0.1},
                    {"name": "command_end", "elapsed_sec": 0.9}
                ],
                "samples": [
                    {
                        "elapsed_sec": 0.0,
                        "cpu_average_logical_cores": null,
                        "memory_current_bytes": 10 * 1024 * 1024,
                        "memory_stat": {"anon": 6 * 1024 * 1024, "file": 2 * 1024 * 1024, "kernel": 2 * 1024 * 1024}
                    },
                    {
                        "elapsed_sec": 0.5,
                        "cpu_average_logical_cores": 1.5,
                        "memory_current_bytes": 70 * 1024 * 1024,
                        "memory_stat": {"anon": 60 * 1024 * 1024, "file": 5 * 1024 * 1024, "kernel": 5 * 1024 * 1024}
                    },
                    {
                        "elapsed_sec": 1.0,
                        "cpu_average_logical_cores": 0.2,
                        "memory_current_bytes": 20 * 1024 * 1024,
                        "memory_stat": {"anon": 10 * 1024 * 1024, "file": 5 * 1024 * 1024, "kernel": 5 * 1024 * 1024}
                    }
                ]
            }))
            .unwrap(),
        )
        .unwrap();
        let report_path = directory.path().join("pipeline_report.json");
        fs::write(
            &report_path,
            serde_json::to_string(&json!({
                "container_cpu": {"allocated_logical_cpus": 2},
                "container_memory": {
                    "samples_path": samples_path,
                    "timeline_valid_for_shape": true,
                    "peak_valid": true,
                    "natural_peak_trustworthy": true,
                    "lifetime": {"peak_bytes": 800 * 1024 * 1024},
                    "observed_limit_margin": {
                        "configured_limit_state": "finite",
                        "configured_limit_bytes": 1024 * 1024 * 1024,
                        "remaining_to_limit_bytes": 224 * 1024 * 1024,
                        "peak_fraction_of_limit": 800.0 / 1024.0,
                        "arithmetic_valid": true
                    }
                }
            }))
            .unwrap(),
        )
        .unwrap();
        let output_path = directory.path().join("timeline.svg");
        (directory, report_path, output_path)
    }

    #[test]
    fn renders_aligned_step_cpu_and_sampled_memory() {
        let (_directory, report, output) = fixture();
        render_resource_timeline_svg(&report, &output).unwrap();
        let svg = fs::read_to_string(output).unwrap();
        assert!(svg.contains("Sandbox CPU and memory over time"));
        assert!(svg.contains("natural-peak=trustworthy"));
        assert!(svg.contains("Leaf memory.max: 1024.0 MiB"));
        assert!(svg.contains("observed margin: 224.0 MiB"));
        assert!(svg.contains("Untimed controlled-finish-gate peak: 800.0 MiB · exact"));
        assert!(svg.contains("data-timed-memory-max-mib=\"75.600000\""));
        assert_eq!(svg.matches("class=\"cpu-interval\"").count(), 2);
        assert!(svg.contains("class=\"cpu-interval-transition\""));
        assert!(svg.ends_with("</svg>\n"));
    }

    #[test]
    fn rejects_invalid_samples_and_markers() {
        let (_directory, report, _output) = fixture();
        let (_, mut timeline) = load_resource_timeline(&report).unwrap();
        timeline["samples"][1]["memory_current_bytes"] = json!(-1);
        let memory = select_memory(&load_json(&report).unwrap())
            .and_then(Value::as_object)
            .unwrap()
            .clone();
        assert!(validate_resource_timeline(&memory, &timeline)
            .unwrap_err()
            .to_string()
            .contains("memory_current_bytes must be a finite nonnegative"));

        timeline["samples"][1]["memory_current_bytes"] = json!(70 * 1024 * 1024);
        timeline["markers"][1]["elapsed_sec"] = json!(1.1);
        assert!(validate_resource_timeline(&memory, &timeline)
            .unwrap_err()
            .to_string()
            .contains("outside the sample window"));
    }

    #[test]
    fn rejects_summary_marked_invalid_with_reason() {
        let (_directory, report, _output) = fixture();
        let (loaded_report, timeline) = load_resource_timeline(&report).unwrap();
        let mut memory = select_memory(&loaded_report)
            .and_then(Value::as_object)
            .unwrap()
            .clone();
        memory.insert("timeline_valid_for_shape".to_owned(), Value::Bool(false));
        memory.insert("timeline_invalid_reasons".to_owned(), json!(["sample gap"]));
        assert!(validate_resource_timeline(&memory, &timeline)
            .unwrap_err()
            .to_string()
            .contains("sample gap"));
    }

    #[test]
    fn cli_refuses_to_overwrite_output() {
        let (_directory, report, output) = fixture();
        fs::write(&output, "sentinel").unwrap();
        let arguments = vec![
            report.into_os_string(),
            OsString::from("--output"),
            output.clone().into_os_string(),
        ];
        assert_eq!(
            main(&arguments, "tasktrace report resource-timeline"),
            2
        );
        assert_eq!(fs::read_to_string(output).unwrap(), "sentinel");
    }

    #[test]
    fn parser_accepts_output_before_or_after_report() {
        let parsed = parse_cli(&[
            OsString::from("--output=chart.svg"),
            OsString::from("report.json"),
        ])
        .unwrap();
        let CliParse::Run(parsed) = parsed else {
            panic!("expected runnable CLI")
        };
        assert_eq!(parsed.report, PathBuf::from("report.json"));
        assert_eq!(parsed.output, PathBuf::from("chart.svg"));
    }

    #[test]
    fn html_escaping_matches_python_html_escape() {
        assert_eq!(
            svg_text(
                1.0,
                2.0,
                "<&\"'",
                &[("data_value", "<&\"'")]
            ),
            "<text x=\"1.0\" y=\"2.0\" data-value=\"&lt;&amp;&quot;&#x27;\">&lt;&amp;&quot;&#x27;</text>"
        );
    }

    #[test]
    fn schema_repr_uses_python_style_quotes() {
        assert_eq!(python_repr(Some(&json!("wrong"))), "'wrong'");
        assert_eq!(python_repr(None), "None");
    }

    #[test]
    fn public_main_signature_is_stable() {
        let _entry: fn(&[OsString], &str) -> i32 = main;
    }
}
