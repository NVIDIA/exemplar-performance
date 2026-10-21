// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared native pipeline envelope and replay-evidence augmentation.
//!
//! Container profiling owns execution and resource evidence.  Replay adapters
//! own format-specific reports.  This module joins those retained artifacts
//! without launching a second workload.

use crate::unified_view;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::error::Error;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Read, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

const AUGMENTATION_VERSION: &str = "tasktrace-coordinator-replay-augmentation.v1";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PipelineError(String);

impl PipelineError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for PipelineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for PipelineError {}

impl From<io::Error> for PipelineError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for PipelineError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

fn object<'a>(
    value: &'a Value,
    description: &str,
) -> Result<&'a Map<String, Value>, PipelineError> {
    value
        .as_object()
        .ok_or_else(|| PipelineError::new(format!("{description} must be a JSON object")))
}

fn read_object(path: &Path, description: &str) -> Result<Value, PipelineError> {
    let bytes = fs::read(path).map_err(|error| {
        PipelineError::new(format!(
            "unable to read {description} {}: {error}",
            path.display()
        ))
    })?;
    let value: Value = serde_json::from_slice(&bytes).map_err(|error| {
        PipelineError::new(format!(
            "unable to decode {description} {}: {error}",
            path.display()
        ))
    })?;
    object(&value, description)?;
    Ok(value)
}

fn sha256_file(path: &Path) -> Result<String, PipelineError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(format!("{:x}", digest.finalize()))
}

fn sorted_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sorted_json).collect()),
        Value::Object(values) => {
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            let mut output = Map::new();
            for key in keys {
                output.insert(key.clone(), sorted_json(&values[key]));
            }
            Value::Object(output)
        }
        value => value.clone(),
    }
}

static WRITE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

fn write_json(path: &Path, value: &Value) -> Result<(), PipelineError> {
    let parent = path.parent().unwrap_or_else(|| Path::new("."));
    let name = path
        .file_name()
        .and_then(OsStr::to_str)
        .ok_or_else(|| PipelineError::new(format!("invalid output path: {}", path.display())))?;
    let temporary = parent.join(format!(
        ".{name}.tmp.{}.{}",
        std::process::id(),
        WRITE_SEQUENCE.fetch_add(1, Ordering::Relaxed)
    ));
    let mut options = OpenOptions::new();
    options.create_new(true).write(true).mode(0o600);
    let mut file = options.open(&temporary)?;
    let mut encoded = serde_json::to_vec_pretty(&sorted_json(value))?;
    encoded.push(b'\n');
    file.write_all(&encoded)?;
    file.sync_all()?;
    drop(file);
    fs::rename(&temporary, path).map_err(|error| {
        let _ = fs::remove_file(&temporary);
        PipelineError::new(format!("unable to publish {}: {error}", path.display()))
    })
}

fn safe_relative(path: &Path, description: &str) -> Result<PathBuf, PipelineError> {
    if path.is_absolute() {
        return Err(PipelineError::new(format!(
            "{description} must be relative: {}",
            path.display()
        )));
    }
    let mut normalized = PathBuf::new();
    for component in path.components() {
        match component {
            Component::Normal(value) => normalized.push(value),
            Component::CurDir => {}
            _ => {
                return Err(PipelineError::new(format!(
                    "{description} escapes its evidence root: {}",
                    path.display()
                )))
            }
        }
    }
    if normalized.as_os_str().is_empty() {
        return Err(PipelineError::new(format!(
            "{description} must not be empty"
        )));
    }
    Ok(normalized)
}

fn validated_output_file(
    output_dir: &Path,
    path: &Path,
    description: &str,
) -> Result<PathBuf, PipelineError> {
    let metadata = fs::symlink_metadata(path).map_err(|error| {
        PipelineError::new(format!(
            "required {description} is unavailable at {}: {error}",
            path.display()
        ))
    })?;
    if metadata.file_type().is_symlink() || !metadata.is_file() {
        return Err(PipelineError::new(format!(
            "required {description} must be a regular non-symlink file: {}",
            path.display()
        )));
    }
    let canonical = path.canonicalize()?;
    if !canonical.starts_with(output_dir.canonicalize()?) {
        return Err(PipelineError::new(format!(
            "required {description} is outside pipeline output: {}",
            canonical.display()
        )));
    }
    Ok(canonical)
}

fn timeline_bounds(bundle: &Value) -> Result<(i128, i128), PipelineError> {
    let timeline = object(
        bundle
            .get("timeline")
            .ok_or_else(|| PipelineError::new("coordinator bundle has no timeline"))?,
        "coordinator timeline",
    )?;
    let start = timeline
        .get("start_ns")
        .and_then(Value::as_u64)
        .ok_or_else(|| PipelineError::new("coordinator timeline has no valid start_ns"))?;
    let end = timeline
        .get("end_ns")
        .and_then(Value::as_u64)
        .ok_or_else(|| PipelineError::new("coordinator timeline has no valid end_ns"))?;
    if end <= start {
        return Err(PipelineError::new(
            "coordinator timeline is not a positive monotonic interval",
        ));
    }
    Ok((i128::from(start), i128::from(end)))
}

fn monotonic_mapping(source: &Value, canonical: &Value) -> Value {
    let source_boot = source.get("boot_id").and_then(Value::as_str);
    let canonical_boot = canonical.get("boot_id").and_then(Value::as_str);
    let source_offset = source
        .get("time_namespace_monotonic_offset_ns")
        .and_then(Value::as_i64);
    let canonical_offset = canonical
        .get("time_namespace_monotonic_offset_ns")
        .and_then(Value::as_i64);
    if source_boot.is_some()
        && source_boot == canonical_boot
        && source_offset.is_some()
        && canonical_offset.is_some()
    {
        return json!({
            "source_clock_id": "replay-monotonic",
            "canonical_clock_id": "host-monotonic",
            "status": "valid",
            "method": "linux-time-namespace-offset",
            "scale_numerator": 1,
            "scale_denominator": 1,
            "offset_ns": canonical_offset.unwrap() - source_offset.unwrap(),
            "max_error_ns": 0,
            "reasons": [],
            "provenance": {
                "boot_id": source_boot,
                "source_time_namespace": source.get("time_namespace").cloned().unwrap_or(Value::Null),
                "canonical_time_namespace": canonical.get("time_namespace").cloned().unwrap_or(Value::Null),
                "source_monotonic_offset_ns": source_offset,
                "canonical_monotonic_offset_ns": canonical_offset,
            },
        });
    }
    let source_namespace = source.get("time_namespace").and_then(Value::as_str);
    let canonical_namespace = canonical.get("time_namespace").and_then(Value::as_str);
    if source_boot.is_some()
        && source_boot == canonical_boot
        && source_namespace.is_some()
        && source_namespace == canonical_namespace
    {
        return json!({
            "source_clock_id": "replay-monotonic",
            "canonical_clock_id": "host-monotonic",
            "status": "valid",
            "method": "linux-time-namespace-identity",
            "scale_numerator": 1,
            "scale_denominator": 1,
            "offset_ns": 0,
            "max_error_ns": 0,
            "reasons": [],
            "provenance": {"boot_id": source_boot, "time_namespace": source_namespace},
        });
    }
    json!({
        "source_clock_id": "replay-monotonic",
        "canonical_clock_id": "host-monotonic",
        "status": "unalignable",
        "method": "linux-monotonic-reference-insufficient",
        "scale_numerator": Value::Null,
        "scale_denominator": Value::Null,
        "offset_ns": Value::Null,
        "max_error_ns": Value::Null,
        "reasons": ["source and canonical monotonic clocks lack a proven common boot and time-namespace offset mapping"],
        "provenance": {
            "source_boot_id": source.get("boot_id").cloned().unwrap_or(Value::Null),
            "canonical_boot_id": canonical.get("boot_id").cloned().unwrap_or(Value::Null),
            "source_time_namespace": source.get("time_namespace").cloned().unwrap_or(Value::Null),
            "canonical_time_namespace": canonical.get("time_namespace").cloned().unwrap_or(Value::Null),
        },
    })
}

fn canonical_reference(bundle: &Value) -> Result<&Value, PipelineError> {
    bundle
        .get("clocks")
        .and_then(Value::as_array)
        .and_then(|clocks| {
            clocks.iter().find(|clock| {
                clock.get("clock_id").and_then(Value::as_str) == Some("host-monotonic")
            })
        })
        .and_then(|clock| clock.get("provenance"))
        .filter(|value| value.is_object())
        .or_else(|| bundle.pointer("/provenance/canonical_clock_reference"))
        .filter(|value| value.is_object())
        .ok_or_else(|| {
            PipelineError::new("coordinator bundle has no canonical monotonic clock reference")
        })
}

fn rebase_and_validate_artifacts(
    bundle: &mut Value,
    direct_bundle_dir: &Path,
    output_dir: &Path,
) -> Result<(), PipelineError> {
    let prefix = direct_bundle_dir
        .canonicalize()?
        .strip_prefix(output_dir.canonicalize()?)
        .map_err(|_| PipelineError::new("coordinator evidence is outside pipeline output"))?
        .to_path_buf();
    let artifacts = bundle
        .get_mut("artifacts")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("coordinator bundle artifacts must be a list"))?;
    let mut ids = BTreeSet::new();
    for artifact in artifacts {
        let artifact = artifact
            .as_object_mut()
            .ok_or_else(|| PipelineError::new("coordinator raw artifact must be an object"))?;
        let id = artifact
            .get("artifact_id")
            .and_then(Value::as_str)
            .ok_or_else(|| PipelineError::new("coordinator raw artifact has no artifact_id"))?;
        if !ids.insert(id.to_owned()) {
            return Err(PipelineError::new(format!(
                "duplicate coordinator raw artifact id: {id}"
            )));
        }
        let relative = artifact
            .get("relative_path")
            .and_then(Value::as_str)
            .ok_or_else(|| PipelineError::new(format!("raw artifact {id} has no relative_path")))?;
        let relative = safe_relative(Path::new(relative), "raw artifact path")?;
        let source = direct_bundle_dir.join(&relative);
        let metadata = source.metadata().map_err(|error| {
            PipelineError::new(format!(
                "required raw artifact {} is unavailable: {error}",
                source.display()
            ))
        })?;
        if !metadata.is_file() {
            return Err(PipelineError::new(format!(
                "required raw artifact is not a file: {}",
                source.display()
            )));
        }
        let expected_size = artifact.get("size_bytes").and_then(Value::as_u64);
        let expected_sha = artifact.get("sha256").and_then(Value::as_str);
        let observed_sha = sha256_file(&source)?;
        if expected_size != Some(metadata.len()) || expected_sha != Some(observed_sha.as_str()) {
            return Err(PipelineError::new(format!(
                "raw artifact content changed after coordinator publication: {}",
                source.display()
            )));
        }
        artifact.insert(
            "relative_path".to_owned(),
            Value::String(prefix.join(relative).to_string_lossy().into_owned()),
        );
    }
    Ok(())
}

fn action_value(action: &Map<String, Value>) -> Value {
    const FIELDS: &[&str] = &[
        "action_index",
        "step_id",
        "tool",
        "source_tool",
        "command",
        "source_status",
        "source_returncode",
        "status",
        "returncode",
        "outcome_matches_source",
        "command_executed",
        "source_argument_parse_failed",
        "launch_failed",
        "completion_timing_drift",
        "session_id",
    ];
    let mut output = Map::new();
    for field in FIELDS {
        if let Some(value) = action.get(*field) {
            output.insert((*field).to_owned(), value.clone());
        }
    }
    Value::Object(output)
}

fn replay_evidence(
    bundle: &mut Value,
    replay: &Value,
    replay_path: &Path,
    output_dir: &Path,
) -> Result<(), PipelineError> {
    let replay_object = object(replay, "replay report")?;
    let reference = replay_object
        .get("clock_reference")
        .filter(|value| value.is_object())
        .ok_or_else(|| PipelineError::new("replay report has no structured clock_reference"))?;
    let mapping = monotonic_mapping(reference, canonical_reference(bundle)?);
    let mapping_valid = mapping.get("status").and_then(Value::as_str) == Some("valid");
    let offset = mapping
        .get("offset_ns")
        .and_then(Value::as_i64)
        .unwrap_or(0) as i128;
    let (timeline_start, timeline_end) = timeline_bounds(bundle)?;
    let run_id = bundle
        .get("run_id")
        .and_then(Value::as_str)
        .ok_or_else(|| PipelineError::new("coordinator bundle has no run_id"))?;
    let scope_id = bundle
        .pointer("/scope/scope_id")
        .and_then(Value::as_str)
        .ok_or_else(|| PipelineError::new("coordinator bundle has no scope_id"))?;
    let actions = replay_object
        .get("actions")
        .and_then(Value::as_array)
        .ok_or_else(|| PipelineError::new("replay report actions must be a list"))?;
    let mut indexes = BTreeSet::new();
    let mut timed = Vec::new();
    let mut untimed = Vec::new();
    for (position, raw) in actions.iter().enumerate() {
        let action = object(raw, "replay action")?;
        let index = action
            .get("action_index")
            .and_then(Value::as_u64)
            .filter(|value| *value > 0)
            .ok_or_else(|| {
                PipelineError::new(format!(
                    "replay action {position} has no valid action_index"
                ))
            })?;
        if !indexes.insert(index) {
            return Err(PipelineError::new(format!(
                "replay action index {index} is duplicated"
            )));
        }
        let status = action
            .get("status")
            .and_then(Value::as_str)
            .filter(|value| !value.is_empty())
            .ok_or_else(|| PipelineError::new(format!("replay action {index} has no status")))?;
        let start = action
            .get("replay_start_monotonic_ns")
            .and_then(Value::as_u64);
        let end = action
            .get("replay_end_monotonic_ns")
            .and_then(Value::as_u64);
        if start.is_some() != end.is_some() {
            return Err(PipelineError::new(format!(
                "replay action {index} has a partial monotonic interval"
            )));
        }
        let value = action_value(action);
        if let (Some(start), Some(end)) = (start, end) {
            if end <= start {
                return Err(PipelineError::new(format!(
                    "replay action {index} has a non-positive monotonic interval"
                )));
            }
            let canonical_start = i128::from(start) + offset;
            let canonical_end = i128::from(end) + offset;
            if mapping_valid && (canonical_start < timeline_start || canonical_end > timeline_end) {
                return Err(PipelineError::new(format!(
                    "replay action {index} falls outside the coordinator timeline"
                )));
            }
            timed.push(json!({
                "record_id": format!("replay-action:{position:06}"),
                "run_id": run_id,
                "scope_id": scope_id,
                "profiler_id": "replay-actions",
                "stream_id": "replay-actions",
                "shape": "interval",
                "status": if mapping_valid { "valid" } else { "unalignable" },
                "reasons": if mapping_valid { Vec::<String>::new() } else { vec!["replay monotonic clock cannot be aligned to the coordinator clock".to_owned()] },
                "value": value,
                "source_clock_id": "replay-monotonic",
                "native_timestamp_ns": Value::Null,
                "native_start_ns": start,
                "native_end_ns": end,
                "canonical_timestamp_ns": Value::Null,
                "canonical_start_ns": if mapping_valid { Value::from(u64::try_from(canonical_start).map_err(|_| PipelineError::new("mapped replay timestamp is negative"))?) } else { Value::Null },
                "canonical_end_ns": if mapping_valid { Value::from(u64::try_from(canonical_end).map_err(|_| PipelineError::new("mapped replay timestamp is negative"))?) } else { Value::Null },
                "alignment_error_ns": if mapping_valid { Value::from(0) } else { Value::Null },
                "acquisition_bounds": Value::Null,
                "attributes": {"lane": "replay-actions"},
                "raw_artifact_ids": ["raw-replay-report"],
                "provenance": {"source_schema_version": replay_object.get("schema_version").cloned().unwrap_or(Value::Null)},
            }));
        } else {
            if !matches!(status, "ignored_metadata" | "skipped_not_executed") {
                return Err(PipelineError::new(format!(
                    "replay action {index} has no complete monotonic interval"
                )));
            }
            untimed.push(json!({
                "record_id": format!("replay-action-fact:{position:06}"),
                "run_id": run_id,
                "scope_id": scope_id,
                "profiler_id": "replay-actions",
                "stream_id": "replay-action-facts",
                "shape": "scalar",
                "status": "valid",
                "reasons": [],
                "value": value,
                "source_clock_id": Value::Null,
                "native_timestamp_ns": Value::Null,
                "native_start_ns": Value::Null,
                "native_end_ns": Value::Null,
                "canonical_timestamp_ns": Value::Null,
                "canonical_start_ns": Value::Null,
                "canonical_end_ns": Value::Null,
                "alignment_error_ns": Value::Null,
                "acquisition_bounds": Value::Null,
                "attributes": {"lane": "replay-action-facts", "exact": true},
                "raw_artifact_ids": ["raw-replay-report"],
                "provenance": {"source_schema_version": replay_object.get("schema_version").cloned().unwrap_or(Value::Null)},
            }));
        }
    }
    if actions.is_empty() {
        return Err(PipelineError::new("replay report contains no actions"));
    }

    let relative = replay_path
        .canonicalize()?
        .strip_prefix(output_dir.canonicalize()?)
        .map_err(|_| PipelineError::new("replay report is outside pipeline output"))?
        .to_path_buf();
    let metadata = replay_path.metadata()?;
    let artifacts = bundle
        .get_mut("artifacts")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("coordinator artifacts must be a list"))?;
    if artifacts.iter().any(|artifact| {
        artifact.get("artifact_id").and_then(Value::as_str) == Some("raw-replay-report")
    }) {
        return Err(PipelineError::new(
            "coordinator bundle already defines replay evidence",
        ));
    }
    artifacts.push(json!({
        "artifact_id": "raw-replay-report",
        "relative_path": relative.to_string_lossy(),
        "media_type": "application/json",
        "sha256": sha256_file(replay_path)?,
        "size_bytes": metadata.len(),
        "provenance": {"augmentation_role": "replay-report", "augmentation_schema_version": AUGMENTATION_VERSION},
    }));
    bundle
        .get_mut("clocks")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("coordinator clocks must be a list"))?
        .push(json!({
            "clock_id": "replay-monotonic",
            "clock_name": "CLOCK_MONOTONIC",
            "resolution_ns": reference.get("storage_precision_ns").and_then(Value::as_u64).unwrap_or(1),
            "namespace_identity": match (reference.get("time_namespace").and_then(Value::as_str), reference.get("boot_id").and_then(Value::as_str)) {
                (Some(namespace), Some(boot)) => Value::String(format!("{namespace}|boot:{boot}")),
                _ => Value::Null,
            },
            "provenance": reference,
        }));
    bundle
        .get_mut("clock_mappings")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("coordinator clock_mappings must be a list"))?
        .push(mapping.clone());
    let mut stream_ids = vec![Value::String("replay-actions".to_owned())];
    if !untimed.is_empty() {
        stream_ids.push(Value::String("replay-action-facts".to_owned()));
    }
    bundle
        .get_mut("profilers")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("coordinator profilers must be a list"))?
        .push(json!({
            "profiler_id": "replay-actions",
            "profiler_version": AUGMENTATION_VERSION,
            "status": if mapping_valid || timed.is_empty() { "valid" } else { "unalignable" },
            "reasons": if mapping_valid || timed.is_empty() { Vec::<String>::new() } else { vec!["replay monotonic clock cannot be aligned to the coordinator clock".to_owned()] },
            "stream_ids": stream_ids,
            "intrusive": false,
            "configuration": {},
            "provenance": {"method": "read-only-retained-replay-augmentation", "does_not_execute_workload": true},
            "extensions": {"measurement_domains": [], "measurement_perturbations": [], "invalidates_measurement_domains": [], "invalidates_metrics": []},
        }));
    let streams = bundle
        .get_mut("streams")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("coordinator streams must be a list"))?;
    streams.push(json!({
        "stream_id": "replay-actions",
        "profiler_id": "replay-actions",
        "metric": "replay.action",
        "shape": "interval",
        "unit": "action",
        "status": if timed.is_empty() { "missing" } else if mapping_valid { "valid" } else { "unalignable" },
        "reasons": if timed.is_empty() { vec!["replay report contains no actions with measured intervals".to_owned()] } else if mapping_valid { Vec::<String>::new() } else { vec!["replay monotonic clock cannot be aligned to the coordinator clock".to_owned()] },
        "records": timed,
        "raw_artifact_ids": ["raw-replay-report"],
        "cadence": Value::Null,
        "gaps": [],
        "display": {"group": "Replay", "label": "Replay actions"},
        "provenance": {"augmentation_schema_version": AUGMENTATION_VERSION, "source_schema_version": replay_object.get("schema_version").cloned().unwrap_or(Value::Null)},
        "extensions": {"measurement_domains": [], "measurement_perturbations": []},
    }));
    if !untimed.is_empty() {
        streams.push(json!({
            "stream_id": "replay-action-facts",
            "profiler_id": "replay-actions",
            "metric": "replay.action.untimed",
            "shape": "scalar",
            "unit": "action",
            "status": "valid",
            "reasons": [],
            "records": untimed,
            "raw_artifact_ids": ["raw-replay-report"],
            "cadence": Value::Null,
            "gaps": [],
            "display": {"group": "Replay", "label": "Untimed replay action facts"},
            "provenance": {"augmentation_schema_version": AUGMENTATION_VERSION, "source_schema_version": replay_object.get("schema_version").cloned().unwrap_or(Value::Null)},
            "extensions": {"measurement_domains": [], "measurement_perturbations": []},
        }));
    }
    Ok(())
}

fn direct_bundle(
    container_report: &Value,
    output_dir: &Path,
    replay: &Value,
    replay_path: &Path,
) -> Result<(Value, Value, PathBuf, String), PipelineError> {
    let attachment = container_report
        .get("evidence_bundle")
        .filter(|value| value.is_object())
        .ok_or_else(|| PipelineError::new("coordinated container report has no evidence_bundle"))?;
    let path = attachment
        .get("path")
        .and_then(Value::as_str)
        .map(PathBuf::from)
        .ok_or_else(|| PipelineError::new("coordinator evidence attachment has no path"))?;
    let path = if path.is_absolute() {
        path
    } else {
        output_dir.join("container_profile").join(path)
    };
    let path = validated_output_file(output_dir, &path, "coordinator evidence bundle")?;
    let expected_sha = attachment.get("sha256").and_then(Value::as_str);
    let observed_sha = sha256_file(&path)?;
    if expected_sha != Some(observed_sha.as_str()) {
        return Err(PipelineError::new(
            "coordinator evidence bundle changed after publication",
        ));
    }
    let mut bundle = read_object(&path, "coordinator evidence bundle")?;
    rebase_and_validate_artifacts(
        &mut bundle,
        path.parent().unwrap_or_else(|| Path::new(".")),
        output_dir,
    )?;
    replay_evidence(&mut bundle, replay, replay_path, output_dir)?;
    if bundle.get("run_id") != container_report.get("run_id") {
        return Err(PipelineError::new(
            "container report and coordinator evidence bundle have different run IDs",
        ));
    }
    Ok((attachment.clone(), bundle, path, observed_sha))
}

fn raw_artifact(
    artifact_id: &str,
    path: &Path,
    output_dir: &Path,
    role: &str,
) -> Result<Value, PipelineError> {
    let canonical = path.canonicalize()?;
    let root = output_dir.canonicalize()?;
    let relative = canonical.strip_prefix(&root).map_err(|_| {
        PipelineError::new(format!(
            "required retained artifact is outside pipeline output: {}",
            canonical.display()
        ))
    })?;
    let metadata = canonical.metadata()?;
    if !metadata.is_file() {
        return Err(PipelineError::new(format!(
            "required retained artifact is not a file: {}",
            canonical.display()
        )));
    }
    Ok(json!({
        "artifact_id": artifact_id,
        "relative_path": relative.to_string_lossy(),
        "media_type": "application/json",
        "sha256": sha256_file(&canonical)?,
        "size_bytes": metadata.len(),
        "provenance": {
            "augmentation_role": role,
            "augmentation_schema_version": AUGMENTATION_VERSION,
        },
    }))
}

fn finish_bundle_provenance(
    bundle: &mut Value,
    output_dir: &Path,
    pipeline_path: &Path,
    direct_path: &Path,
    direct_sha256: &str,
    container_profile_path: &Path,
) -> Result<(), PipelineError> {
    let artifacts = bundle
        .get_mut("artifacts")
        .and_then(Value::as_array_mut)
        .ok_or_else(|| PipelineError::new("augmented bundle artifacts must be a list"))?;
    artifacts.extend([
        raw_artifact(
            "raw-direct-evidence-bundle",
            direct_path,
            output_dir,
            "authoritative-coordinator-bundle",
        )?,
        raw_artifact(
            "raw-source-report",
            pipeline_path,
            output_dir,
            "pipeline-report",
        )?,
        raw_artifact(
            "raw-container-profile-report",
            container_profile_path,
            output_dir,
            "container-profile-report",
        )?,
    ]);
    let mut ids = BTreeSet::new();
    for artifact in artifacts.iter() {
        let id = artifact
            .get("artifact_id")
            .and_then(Value::as_str)
            .ok_or_else(|| PipelineError::new("augmented raw artifact has no artifact_id"))?;
        if !ids.insert(id.to_owned()) {
            return Err(PipelineError::new(format!(
                "augmented bundle has duplicate artifact id: {id}"
            )));
        }
    }
    let direct_relative = direct_path
        .canonicalize()?
        .strip_prefix(output_dir.canonicalize()?)
        .map_err(|_| PipelineError::new("direct bundle is outside pipeline output"))?
        .to_path_buf();
    let source_sha256 = sha256_file(pipeline_path)?;
    bundle
        .get_mut("provenance")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| PipelineError::new("coordinator provenance must be an object"))?
        .insert(
            "pipeline_augmentation".to_owned(),
            json!({
                "schema_version": AUGMENTATION_VERSION,
                "method": "verified-direct-bundle-plus-retained-replay",
                "does_not_execute_workload": true,
                "workload_executions_added": 0,
                "source_report": pipeline_path.file_name().and_then(OsStr::to_str),
                "source_report_sha256": source_sha256,
                "source_attachment_field": "coordinator_evidence_bundle",
                "direct_bundle": direct_relative.to_string_lossy(),
                "direct_bundle_sha256": direct_sha256,
                "phase_log": Value::Null,
            }),
        );
    let replay_status = bundle
        .get("profilers")
        .and_then(Value::as_array)
        .and_then(|profilers| {
            profilers.iter().find(|profiler| {
                profiler.get("profiler_id").and_then(Value::as_str) == Some("replay-actions")
            })
        })
        .and_then(|profiler| profiler.get("status"))
        .cloned()
        .unwrap_or_else(|| Value::String("missing".to_owned()));
    let extensions = bundle
        .get_mut("extensions")
        .and_then(Value::as_object_mut)
        .ok_or_else(|| PipelineError::new("coordinator extensions must be an object"))?;
    let coordinator_status = extensions
        .get("evidence_status")
        .cloned()
        .unwrap_or(Value::Null);
    let combined = match (coordinator_status.as_str(), replay_status.as_str()) {
        (Some("invalid"), _) | (_, Some("invalid" | "dropped")) => "invalid",
        (Some("partial"), _) | (_, Some("missing" | "unalignable")) => "partial",
        _ => "valid",
    };
    extensions.insert(
        "augmentation_schema_version".to_owned(),
        Value::String(AUGMENTATION_VERSION.to_owned()),
    );
    extensions.insert(
        "augmentation_profiler_statuses".to_owned(),
        json!({"replay-actions": replay_status}),
    );
    extensions.insert("coordinator_evidence_status".to_owned(), coordinator_status);
    extensions.insert(
        "evidence_status".to_owned(),
        Value::String(combined.to_owned()),
    );
    extensions.insert(
        "all_bundle_evidence_valid".to_owned(),
        Value::Bool(combined == "valid"),
    );
    Ok(())
}

fn transcript_summary_is_valid(
    replay: &Value,
    summary: &Map<String, Value>,
    report_fields: &Map<String, Value>,
) -> bool {
    let count = |field: &str| summary.get(field).and_then(Value::as_u64);
    let Some(extracted) = count("actions_extracted") else {
        return false;
    };
    let Some(replayed) = count("actions_replayed") else {
        return false;
    };
    let Some(ok) = count("actions_ok") else {
        return false;
    };
    let Some(unsupported) = count("actions_unsupported") else {
        return false;
    };
    let Some(failed) = count("actions_failed") else {
        return false;
    };
    let Some(allow_unsupported) = report_fields
        .get("allow_unsupported")
        .and_then(Value::as_bool)
    else {
        return false;
    };
    if replay.get("task_id") != report_fields.get("task_id")
        || replay.get("trial_name") != report_fields.get("trial_name")
        || report_fields
            .get("actions_validated")
            .and_then(Value::as_u64)
            != Some(extracted)
        || report_fields
            .get("unsupported_actions_validated")
            .and_then(Value::as_u64)
            != Some(unsupported)
    {
        return false;
    }
    let Some(actions) = replay.get("actions").and_then(Value::as_array) else {
        return false;
    };
    let mut observed_ok = 0_u64;
    let mut observed_unsupported = 0_u64;
    let mut observed_failed = 0_u64;
    for (position, action) in actions.iter().enumerate() {
        if action.get("action_index").and_then(Value::as_u64) != Some(position as u64 + 1) {
            return false;
        }
        match action.get("status").and_then(Value::as_str) {
            Some("ok") => observed_ok += 1,
            Some("unsupported") => observed_unsupported += 1,
            Some("failed" | "error" | "timed_out") => observed_failed += 1,
            _ => return false,
        }
    }
    extracted > 0
        && replayed == extracted
        && actions.len() as u64 == replayed
        && failed == 0
        && (unsupported == 0 || allow_unsupported)
        && (observed_ok, observed_unsupported, observed_failed) == (ok, unsupported, failed)
        && ok
            .checked_add(unsupported)
            .and_then(|value| value.checked_add(failed))
            == Some(replayed)
}

/// New ATIF adapters distinguish executed tools from metadata and reproduced
/// input-validation failures. Zero mismatches alone cannot prove completion.
pub(crate) fn atif_summary_is_valid(summary: &Map<String, Value>) -> bool {
    let count = |field| summary.get(field).and_then(Value::as_u64);
    let recorded = count("tool_actions_recorded");
    [
        "actions_timed_out",
        "actions_skipped_after_timeout",
        "actions_launch_failed",
        "actions_source_argument_parse_failed",
        "source_outcome_mismatch_count",
        "source_outcome_comparison_unavailable_count",
    ]
    .iter()
    .all(|field| count(field) == Some(0))
        && recorded.is_some_and(|count| count > 0)
        && count("tool_actions_replayed") == recorded
        && count("actions_executed").and_then(|executed| {
            let reproduced = summary
                .get("tool_actions_reproduced_without_execution")
                .map_or(Some(0), Value::as_u64)?;
            executed.checked_add(reproduced)
        }) == recorded
}

fn summary_is_valid(
    replay: &Value,
    summary: &Map<String, Value>,
    report_fields: &Map<String, Value>,
) -> bool {
    match replay.get("schema_version").and_then(Value::as_str) {
        Some("offline-tool-replay.v1") => {
            transcript_summary_is_valid(replay, summary, report_fields)
        }
        Some("offline-atif-trajectory-replay.v1") => atif_summary_is_valid(summary),
        // Trace replay runs every action; a timeout the source did not have is a mismatch, so a
        // replay that reproduced every recorded outcome is valid even when some were timeouts.
        Some("offline-trace-replay.v1") => {
            summary
                .get("source_outcome_mismatch_count")
                .and_then(Value::as_u64)
                == Some(0)
        }
        _ => false,
    }
}

fn copy_optional(report: &mut Map<String, Value>, source: &Value, field: &str) {
    if let Some(value) = source.get(field).filter(|value| !value.is_null()) {
        report.insert(field.to_owned(), value.clone());
    }
}

/// Join one completed container profile with one retained replay report.
///
/// The function never executes workload code.  For coordinated profiles it
/// verifies and augments the direct coordinator bundle, writes the authoritative
/// root bundle, and renders the unified artifacts from that retained evidence.
pub fn finish_profiled_replay(
    container_report: &Value,
    output_dir: &Path,
    replay_report_relative_path: &Path,
    report_fields: Value,
    profile_sets: &[String],
    coordinated_baseline: bool,
) -> Result<Value, PipelineError> {
    let replay_path = output_dir.join(replay_report_relative_path);
    let pipeline_path = output_dir.join("pipeline_report.json");
    let fields = object(&report_fields, "adapter report fields")?;
    let mut report = fields.clone();
    report.insert(
        "image_identity".to_owned(),
        container_report
            .get("image_identity")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "replay_report".to_owned(),
        Value::String(
            replay_path
                .canonicalize()
                .unwrap_or_else(|_| replay_path.clone())
                .to_string_lossy()
                .into_owned(),
        ),
    );
    report.insert(
        "container_profile_report".to_owned(),
        Value::String(
            output_dir
                .join("container_profile/profile_report.json")
                .canonicalize()
                .unwrap_or_else(|_| output_dir.join("container_profile/profile_report.json"))
                .to_string_lossy()
                .into_owned(),
        ),
    );
    report.insert("replay_summary".to_owned(), Value::Null);
    report.insert(
        "container_cpu".to_owned(),
        container_report.get("cpu").cloned().unwrap_or(Value::Null),
    );
    if let Some(memory) = container_report
        .get("memory")
        .filter(|value| !value.is_null())
    {
        report.insert("container_memory".to_owned(), memory.clone());
    }
    report.insert(
        "scheduler_concurrency".to_owned(),
        container_report
            .get("scheduler_concurrency")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "proc_stat_comparison".to_owned(),
        container_report
            .get("proc_stat_comparison")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "strace".to_owned(),
        container_report
            .get("strace")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert(
        "warning".to_owned(),
        container_report
            .get("warning")
            .cloned()
            .unwrap_or(Value::Null),
    );
    report.insert("status".to_owned(), Value::String("valid".to_owned()));
    report.insert(
        "profile_provenance".to_owned(),
        json!({
            "mode": if profile_sets.is_empty() { "baseline" } else { "profile" },
            "profile_sets": profile_sets,
        }),
    );
    if container_report.get("runtime").and_then(Value::as_str) == Some("current") {
        report.insert("runtime".to_owned(), Value::String("current".to_owned()));
        report.insert(
            "measurement_scope".to_owned(),
            container_report
                .get("scope")
                .cloned()
                .unwrap_or(Value::Null),
        );
        for field in [
            "environment_identity",
            "task_environment",
            "measurement_validity",
        ] {
            report.insert(
                field.to_owned(),
                container_report.get(field).cloned().unwrap_or(Value::Null),
            );
        }
    }
    for field in [
        "network_policy",
        "network",
        "enroot_lifecycle",
        "runtime_report",
        "run_id",
        "profile_started_at_utc",
        "canonical_clock_reference",
        "canonical_lifecycle",
        "hardware_perf",
        "process_exec",
        "verification",
        "cancellation",
        "profiling_valid",
        "profiling_invalid_reasons",
    ] {
        copy_optional(&mut report, container_report, field);
    }
    if report.get("runtime").and_then(Value::as_str) == Some("current")
        && report
            .get("measurement_validity")
            .and_then(|value| value.get("valid_for_average_cpu_reporting"))
            != Some(&Value::Bool(true))
    {
        let reason = json!({
            "profiler_id": "current-container-cgroup-cpu",
            "status": "invalid",
            "reasons": ["current-container measurement is not valid for average CPU reporting"],
        });
        let reasons = report
            .entry("profiling_invalid_reasons".to_owned())
            .or_insert_with(|| Value::Array(Vec::new()));
        if !reasons.is_array() {
            *reasons = Value::Array(Vec::new());
        }
        let reasons = reasons.as_array_mut().expect("normalized invalid reasons");
        if !reasons.contains(&reason) {
            reasons.push(reason);
        }
        report.insert("profiling_valid".to_owned(), Value::Bool(false));
    }
    let coordinated = coordinated_baseline
        || !profile_sets.is_empty()
        || container_report
            .get("evidence_bundle")
            .is_some_and(Value::is_object);
    if !replay_path.is_file() {
        if let Some(memory) = report
            .get_mut("container_memory")
            .and_then(Value::as_object_mut)
        {
            if memory.get("censoring").and_then(Value::as_str) == Some("none") {
                memory.insert(
                    "censoring".to_owned(),
                    Value::String("functional_failure".to_owned()),
                );
            }
            memory.insert("natural_peak_trustworthy".to_owned(), Value::Bool(false));
            memory.insert("valid_for_sizing".to_owned(), Value::Bool(false));
            let reason = Value::String(
                "container command completed without the required replay report".to_owned(),
            );
            let reasons = memory
                .entry("natural_peak_invalid_reasons".to_owned())
                .or_insert_with(|| Value::Array(Vec::new()));
            if let Some(reasons) = reasons.as_array_mut() {
                if !reasons.contains(&reason) {
                    reasons.push(reason);
                }
            }
        }
        report.insert(
            "pipeline_status".to_owned(),
            Value::String("failed_missing_replay_report".to_owned()),
        );
        report.insert("status".to_owned(), Value::String("invalid".to_owned()));
        report.insert(
            "pipeline_error".to_owned(),
            Value::String(format!(
                "container command completed without a replay report: {}",
                replay_path.display()
            )),
        );
        report.insert(
            "container_command_exit_code".to_owned(),
            container_report
                .get("command_exit_code")
                .cloned()
                .unwrap_or(Value::Null),
        );
        let value = Value::Object(report);
        write_json(&pipeline_path, &value)?;
        return Err(PipelineError::new(format!(
            "container command completed without a replay report: {}",
            replay_path.display()
        )));
    }
    let replay_path = validated_output_file(output_dir, &replay_path, "replay report")?;
    let replay = read_object(&replay_path, "replay report")?;
    let summary = replay
        .get("summary")
        .and_then(Value::as_object)
        .ok_or_else(|| PipelineError::new("replay report summary must be an object"))?;
    report.insert("replay_summary".to_owned(), Value::Object(summary.clone()));
    if !summary_is_valid(&replay, summary, fields)
        || report.get("profiling_valid") == Some(&Value::Bool(false))
    {
        report.insert("status".to_owned(), Value::String("invalid".to_owned()));
    }

    let mut augmented_bundle = None;
    if coordinated {
        let (direct_attachment, bundle, direct_path, direct_sha256) =
            direct_bundle(container_report, output_dir, &replay, &replay_path)?;
        let bundle_path = output_dir.join("evidence_bundle.json");
        report.insert(
            "coordinator_evidence_bundle".to_owned(),
            direct_attachment.clone(),
        );
        for (source, target) in [
            ("execution_plan_digest", "execution_plan_digest"),
            ("scope", "scope"),
        ] {
            if let Some(value) = direct_attachment
                .get(source)
                .filter(|value| !value.is_null())
            {
                report.insert(target.to_owned(), value.clone());
            }
        }
        if let Some(scope) = bundle.get("scope").filter(|value| value.is_object()) {
            report.insert("scope".to_owned(), scope.clone());
        }
        if let Some(plan) = bundle.pointer("/provenance/execution_plan") {
            report.insert("execution_plan".to_owned(), plan.clone());
        }
        report.insert(
            "evidence_bundle".to_owned(),
            json!({
                "path": bundle_path.to_string_lossy(),
                "authoritative": true,
                "generation_method": "verified-direct-coordinator-bundle-augmentation",
                "coordinator_source": "coordinator_evidence_bundle",
                "coordination_status": direct_attachment.get("coordination_status").cloned().unwrap_or(Value::Null),
                "evidence_status": direct_attachment.get("evidence_status").cloned().unwrap_or(Value::Null),
                "all_selected_profilers_valid": direct_attachment.get("all_selected_profilers_valid").cloned().unwrap_or(Value::Null),
                "profile_sets": profile_sets,
                "sha256_manifest": output_dir.join("unified_profile_artifacts.json").to_string_lossy(),
            }),
        );
        report.insert(
            "unified_profile_artifacts".to_owned(),
            json!({
                "evidence_bundle": bundle_path.to_string_lossy(),
                "view_model": output_dir.join("view_model.json").to_string_lossy(),
                "html": output_dir.join("unified_profile.html").to_string_lossy(),
                "svg": output_dir.join("unified_profile.svg").to_string_lossy(),
                "animated_svg": output_dir.join("unified_profile.animated.svg").to_string_lossy(),
                "manifest": output_dir.join("unified_profile_artifacts.json").to_string_lossy(),
                "workload_executions": 1,
                "generation_method": "verified-direct-coordinator-bundle-augmentation",
            }),
        );
        augmented_bundle = Some((
            bundle,
            bundle_path,
            direct_path,
            direct_sha256,
            output_dir.join("container_profile/profile_report.json"),
        ));
    }
    let mut value = Value::Object(report);
    write_json(&pipeline_path, &value)?;
    if let Some((mut bundle, bundle_path, direct_path, direct_sha256, container_path)) =
        augmented_bundle
    {
        finish_bundle_provenance(
            &mut bundle,
            output_dir,
            &pipeline_path,
            &direct_path,
            &direct_sha256,
            &container_path,
        )?;
        write_json(&bundle_path, &bundle)?;
        let rendered =
            unified_view::write_unified_view_artifacts(&bundle, output_dir).map_err(|error| {
                PipelineError::new(format!(
                    "profile completed but unified evidence rendering failed: {error}"
                ))
            })?;
        let manifest = json!({
            "schema_version": "tasktrace-unified-profile-artifacts.v1",
            "run_id": value.get("run_id").cloned().unwrap_or(Value::Null),
            "workload_executions": 1,
            "evidence_bundle": bundle_path.canonicalize()?.to_string_lossy(),
            "evidence_bundle_sha256": sha256_file(&bundle_path)?,
            "pipeline_report_sha256": sha256_file(&pipeline_path)?,
            "generation_method": "verified-direct-coordinator-bundle-augmentation",
            "view_model": rendered.get("view_model").cloned(),
            "html": rendered.get("html").cloned(),
            "svg": rendered.get("svg").cloned(),
            "animated_svg": rendered.get("animated_svg").cloned(),
        });
        write_json(
            &output_dir.join("unified_profile_artifacts.json"),
            &manifest,
        )?;
    }
    // Re-read the published representation so callers and disk consumers see
    // precisely the same sorted JSON value.
    value = read_object(&pipeline_path, "pipeline report")?;
    Ok(value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::fs::symlink;

    fn transcript_replay(statuses: &[&str]) -> Value {
        json!({
            "schema_version": "offline-tool-replay.v1",
            "task_id": "task",
            "trial_name": "task.1",
            "actions": statuses.iter().enumerate().map(|(index, status)| json!({
                "action_index": index + 1,
                "status": status,
            })).collect::<Vec<_>>(),
        })
    }

    fn transcript_fields(action_count: u64, unsupported: u64, allowed: bool) -> Value {
        json!({
            "task_id": "task",
            "trial_name": "task.1",
            "actions_validated": action_count,
            "unsupported_actions_validated": unsupported,
            "allow_unsupported": allowed,
        })
    }

    #[test]
    fn clock_mapping_uses_exact_namespace_offsets() {
        let source = json!({
            "boot_id": "boot",
            "time_namespace": "time:[2]",
            "time_namespace_monotonic_offset_ns": 7,
        });
        let canonical = json!({
            "boot_id": "boot",
            "time_namespace": "time:[1]",
            "time_namespace_monotonic_offset_ns": 11,
        });
        let mapping = monotonic_mapping(&source, &canonical);
        assert_eq!(mapping["status"], "valid");
        assert_eq!(mapping["offset_ns"], 4);
    }

    #[test]
    fn clock_mapping_fails_closed_without_common_identity() {
        let mapping = monotonic_mapping(
            &json!({"boot_id": "one", "time_namespace": "time:[1]"}),
            &json!({"boot_id": "two", "time_namespace": "time:[1]"}),
        );
        assert_eq!(mapping["status"], "unalignable");
        assert!(mapping["offset_ns"].is_null());
    }

    #[test]
    fn relative_artifact_paths_cannot_escape() {
        assert!(safe_relative(Path::new("cpu.stat"), "artifact").is_ok());
        assert!(safe_relative(Path::new("../outside"), "artifact").is_err());
        assert!(safe_relative(Path::new("/outside"), "artifact").is_err());
    }

    #[test]
    fn retained_reports_must_not_be_symlinks() {
        let root = tempfile::tempdir().unwrap();
        let target = root.path().join("target.json");
        let link = root.path().join("replay.json");
        fs::write(&target, "{}\n").unwrap();
        symlink(&target, &link).unwrap();
        assert!(validated_output_file(root.path(), &link, "replay report")
            .unwrap_err()
            .to_string()
            .contains("non-symlink"));
    }

    #[test]
    fn transcript_summary_requires_complete_accounting_and_no_failures() {
        let replay = transcript_replay(&["ok", "ok", "ok"]);
        let fields = transcript_fields(3, 0, false);
        let fields = fields.as_object().unwrap();
        let valid = json!({
            "actions_extracted": 3,
            "actions_replayed": 3,
            "actions_ok": 3,
            "actions_unsupported": 0,
            "actions_failed": 0,
        });
        assert!(summary_is_valid(
            &replay,
            valid.as_object().unwrap(),
            fields
        ));

        let failed = json!({
            "actions_extracted": 3,
            "actions_replayed": 3,
            "actions_ok": 2,
            "actions_unsupported": 0,
            "actions_failed": 1,
        });
        assert!(!summary_is_valid(
            &transcript_replay(&["ok", "ok", "failed"]),
            failed.as_object().unwrap(),
            fields
        ));

        let inconsistent = json!({
            "actions_extracted": 3,
            "actions_replayed": 3,
            "actions_ok": 3,
            "actions_unsupported": 1,
            "actions_failed": 0,
        });
        assert!(!summary_is_valid(
            &replay,
            inconsistent.as_object().unwrap(),
            fields
        ));
    }

    #[test]
    fn transcript_summary_surfaces_explicit_partial_replay() {
        let replay = transcript_replay(&["ok", "unsupported"]);
        let partial = json!({
            "actions_extracted": 2,
            "actions_replayed": 2,
            "actions_ok": 1,
            "actions_unsupported": 1,
            "actions_failed": 0,
        });
        let denied = transcript_fields(2, 1, false);
        assert!(!summary_is_valid(
            &replay,
            partial.as_object().unwrap(),
            denied.as_object().unwrap(),
        ));
        let allowed = transcript_fields(2, 1, true);
        assert!(summary_is_valid(
            &replay,
            partial.as_object().unwrap(),
            allowed.as_object().unwrap(),
        ));
    }

    #[test]
    fn trace_replay_with_reproduced_timeouts_is_valid_but_atif_is_not() {
        let timed_out = json!({"actions_timed_out": 1, "source_outcome_mismatch_count": 0});
        let mismatch = json!({"actions_timed_out": 0, "source_outcome_mismatch_count": 1});
        let no_fields = json!({});
        let trace = json!({"schema_version": "offline-trace-replay.v1"});
        let atif = json!({"schema_version": "offline-atif-trajectory-replay.v1"});
        assert!(summary_is_valid(
            &trace,
            timed_out.as_object().unwrap(),
            no_fields.as_object().unwrap(),
        ));
        assert!(!summary_is_valid(
            &trace,
            mismatch.as_object().unwrap(),
            no_fields.as_object().unwrap(),
        ));
        assert!(!summary_is_valid(
            &atif,
            timed_out.as_object().unwrap(),
            no_fields.as_object().unwrap(),
        ));
    }

    #[test]
    fn summary_validation_fails_closed_for_unknown_schema() {
        let summary = json!({
            "actions_timed_out": 0,
            "source_outcome_mismatch_count": 0,
        });
        assert!(!summary_is_valid(
            &json!({"schema_version": "unknown-replay.v1"}),
            summary.as_object().unwrap(),
            json!({}).as_object().unwrap(),
        ));
    }

    #[test]
    fn pipeline_projects_memory_and_marks_missing_replay_as_functional_failure() {
        let root = tempfile::tempdir().unwrap();
        let container = json!({
            "image_identity": {"id": "sha256:image"},
            "cpu": {"valid_for_cpu_performance": true},
            "memory": {
                "schema_version": "container-cgroup-memory.v2",
                "censoring": "none",
                "natural_peak_trustworthy": true,
                "natural_peak_invalid_reasons": [],
                "lifetime": {"peak_bytes": 1234},
            },
            "strace": {"mode": "none"},
            "warning": "memory test",
            "command_exit_code": 0,
        });
        let error = finish_profiled_replay(
            &container,
            root.path(),
            Path::new("missing-replay.json"),
            json!({}),
            &[],
            false,
        )
        .unwrap_err();
        assert!(error.to_string().contains("without a replay report"));
        let report =
            read_object(&root.path().join("pipeline_report.json"), "pipeline report").unwrap();
        assert_eq!(report["container_memory"]["lifetime"]["peak_bytes"], 1234);
        assert_eq!(
            report["container_memory"]["censoring"],
            "functional_failure"
        );
        assert_eq!(
            report["container_memory"]["natural_peak_trustworthy"],
            false
        );
        assert!(report["container_memory"]["natural_peak_invalid_reasons"]
            .as_array()
            .unwrap()
            .iter()
            .any(|reason| reason.as_str().unwrap().contains("required replay report")));
    }

    #[test]
    fn uncoordinated_pipeline_preserves_top_level_scheduler_report() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("replay")).unwrap();
        write_json(
            &root.path().join("replay/replay_report.json"),
            &json!({
                "schema_version": "offline-trace-replay.v1",
                "summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                },
            }),
        )
        .unwrap();
        let scheduler = json!({
            "backend": "perf",
            "peak_simultaneous_logical_cores": 2,
            "trace_complete": true,
            "cgroup_validation": {"valid_for_concurrency_reporting": true},
        });
        let container = json!({
            "image_identity": {"id": "sha256:image"},
            "cpu": {"valid_for_cpu_performance": true},
            "scheduler_concurrency": scheduler,
            "strace": {"mode": "none"},
            "warning": "scheduler test",
            "command_exit_code": 0,
        });
        let report = finish_profiled_replay(
            &container,
            root.path(),
            Path::new("replay/replay_report.json"),
            json!({}),
            &[],
            false,
        )
        .unwrap();
        assert_eq!(report["scheduler_concurrency"], scheduler);
        assert_eq!(report["status"], "valid");
        assert!(report.get("evidence_bundle").is_none());
    }

    #[test]
    fn current_runtime_projects_scope_and_fails_closed_on_invalid_exclusivity() {
        let root = tempfile::tempdir().unwrap();
        fs::create_dir(root.path().join("replay")).unwrap();
        write_json(
            &root.path().join("replay/replay_report.json"),
            &json!({
                "schema_version": "offline-atif-trajectory-replay.v1",
                "summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                },
            }),
        )
        .unwrap();
        let container = json!({
            "runtime": "current",
            "scope": "current-container",
            "image_identity": null,
            "cpu": {"usage_sec": 1.0},
            "environment_identity": {"runtime": "current"},
            "task_environment": {"unset_variables": ["VIRTUAL_ENV"]},
            "measurement_validity": {
                "valid_for_average_cpu_reporting": false,
                "exclusive_single_task_container_asserted": false,
            },
            "strace": {"mode": "none"},
            "warning": "diagnostic",
            "command_exit_code": 0,
        });
        let report = finish_profiled_replay(
            &container,
            root.path(),
            Path::new("replay/replay_report.json"),
            json!({"schema_version": "atif-trajectory-profile.v1"}),
            &[],
            false,
        )
        .unwrap();
        assert_eq!(report["runtime"], "current");
        assert_eq!(report["measurement_scope"], "current-container");
        assert_eq!(report["environment_identity"]["runtime"], "current");
        assert_eq!(report["profiling_valid"], false);
        assert_eq!(report["status"], "invalid");
        assert_eq!(
            report["profiling_invalid_reasons"][0]["profiler_id"],
            "current-container-cgroup-cpu"
        );
    }
    #[test]
    fn atif_summary_requires_complete_comparable_tool_replay() {
        let valid = json!({
            "actions_timed_out": 0, "actions_skipped_after_timeout": 0,
            "actions_launch_failed": 0, "actions_source_argument_parse_failed": 0,
            "source_outcome_mismatch_count": 0,
            "source_outcome_comparison_unavailable_count": 0,
            "tool_actions_recorded": 2, "tool_actions_replayed": 2, "actions_executed": 2,
        });
        assert!(atif_summary_is_valid(valid.as_object().unwrap()));
        for field in [
            "actions_timed_out",
            "actions_skipped_after_timeout",
            "actions_launch_failed",
            "actions_source_argument_parse_failed",
            "source_outcome_mismatch_count",
            "source_outcome_comparison_unavailable_count",
        ] {
            for value in [json!(1), Value::Null, json!(true), json!("0"), json!(-1)] {
                let mut invalid = valid.clone();
                invalid[field] = value;
                assert!(
                    !atif_summary_is_valid(invalid.as_object().unwrap()),
                    "{field}: {invalid}"
                );
            }
            let mut missing = valid.clone();
            missing.as_object_mut().unwrap().remove(field);
            assert!(!atif_summary_is_valid(missing.as_object().unwrap()));
        }
        for field in [
            "tool_actions_recorded",
            "tool_actions_replayed",
            "actions_executed",
        ] {
            for value in [json!(0), json!(1), Value::Null, json!(true)] {
                let mut invalid = valid.clone();
                invalid[field] = value;
                assert!(
                    !atif_summary_is_valid(invalid.as_object().unwrap()),
                    "{field}: {invalid}"
                );
            }
        }
        let mut reproduced = valid.clone();
        reproduced["actions_executed"] = json!(1);
        reproduced["tool_actions_reproduced_without_execution"] = json!(1);
        assert!(atif_summary_is_valid(reproduced.as_object().unwrap()));
        for value in [
            Value::Null,
            json!(true),
            json!("1"),
            json!(-1),
            json!(u64::MAX),
        ] {
            reproduced["tool_actions_reproduced_without_execution"] = value;
            assert!(!atif_summary_is_valid(reproduced.as_object().unwrap()));
        }
    }
}
