// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Strict, content-bound `tasktrace-run.v1`/`.v2`/`.v3` manifests.
//!
//! Parsing is deliberately separate from execution. A successfully loaded
//! document contains canonical CPU sets, resolved paths, source digests, and
//! (for v3 compact authoring) the fully expanded launch plan.

use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use serde_yaml::{Mapping, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeSet, VecDeque};
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs;
use std::net::IpAddr;
use std::path::{Component, Path, PathBuf};

pub const SCHEMA_VERSION_V1: &str = "tasktrace-run.v1";
pub const SCHEMA_VERSION_V2: &str = "tasktrace-run.v2";
pub const SCHEMA_VERSION_V3: &str = "tasktrace-run.v3";
pub const SCHEMA_VERSION: &str = SCHEMA_VERSION_V1;

const ROUND_ROBIN_SINGLE_CPU: &str = "round-robin-single-cpu";
const PACING_MODES: &[&str] = &["recorded", "unpaced", "llm-gaps"];
const PROFILE_SETS: &[&str] = &[
    "cpu-average",
    "cpu-peak",
    "memory-detail",
    "cpu-compare",
    "software-deep",
    "hardware-deep",
];
const NATIVE_CONTROLLER_RUNTIME: &str = "rust-host-mount";
const RETIRED_CONTROLLER_RUNTIME_MODES: &[&str] = &["image", "host-mount"];
const CONTAINER_RUNTIMES: &[&str] = &["docker", "enroot"];
const NETWORK_MODES: &[&str] = &["public", "no-network", "allowlist"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RunSpecError(String);

impl RunSpecError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for RunSpecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for RunSpecError {}

impl From<std::io::Error> for RunSpecError {
    fn from(error: std::io::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunTaskSpec {
    pub task_id: String,
    pub atif: PathBuf,
    pub source_sha256: String,
    pub image: String,
    pub expected_image_id: Option<String>,
    pub action_timeout_sec: Option<f64>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunSpecV1 {
    pub schema_version: String,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub output: PathBuf,
    pub cpus: String,
    pub cpu_ids: Vec<u32>,
    pub concurrency: usize,
    pub repetitions: usize,
    pub tasks: Vec<RunTaskSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunSpecV2 {
    pub schema_version: String,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub output: PathBuf,
    pub cpus: String,
    pub cpu_ids: Vec<u32>,
    pub concurrency: usize,
    pub repetitions: usize,
    pub tasks: Vec<RunTaskSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum MetadataScalar {
    String(String),
    Boolean(bool),
    Integer(i128),
    Number(f64),
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunDefaultsSpec {
    pub pacing: String,
    pub network_policy: String,
    pub allowed_hosts: Vec<String>,
    pub runtime: String,
    pub profile_sets: Vec<String>,
    pub controller_runtime: String,
    pub controller_cpus: String,
    pub controller_cpu_ids: Vec<u32>,
    pub action_timeout_sec: f64,
    pub worker_cpus: Option<String>,
    pub worker_cpu_ids: Vec<u32>,
    pub placement: Option<String>,
    /// Optional container working directory for every launch. The historical
    /// profile default is `/app` (Terminal-Bench images); datasets such as
    /// SWE-bench place the task checkout elsewhere (`/testbed`).
    pub workdir: Option<String>,
    /// Optional NAME=VALUE assignments applied inside every launch's
    /// container, pinning the recording environment across sites and enroot
    /// versions (image-ENV application is site/version dependent).
    pub environment: Vec<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub struct RunLaunchSpec {
    pub task_id: String,
    pub cpus: String,
    pub cpu_ids: Vec<u32>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunWaveSpec {
    pub wave_id: String,
    pub scored: bool,
    pub metadata: Vec<(String, MetadataScalar)>,
    pub launches: Vec<RunLaunchSpec>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct RunSpecV3 {
    pub schema_version: String,
    pub manifest_path: PathBuf,
    pub manifest_sha256: String,
    pub output: PathBuf,
    pub input_root: PathBuf,
    pub defaults: RunDefaultsSpec,
    pub tasks: Vec<RunTaskSpec>,
    pub waves: Vec<RunWaveSpec>,
}

pub type OrderedRunSpec = RunSpecV3;

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(untagged)]
pub enum RunSpec {
    V1(RunSpecV1),
    V2(RunSpecV2),
    V3(Box<RunSpecV3>),
}

impl RunSpec {
    pub fn schema_version(&self) -> &str {
        match self {
            Self::V1(spec) => &spec.schema_version,
            Self::V2(spec) => &spec.schema_version,
            Self::V3(spec) => &spec.schema_version,
        }
    }

    pub fn manifest_path(&self) -> &Path {
        match self {
            Self::V1(spec) => &spec.manifest_path,
            Self::V2(spec) => &spec.manifest_path,
            Self::V3(spec) => &spec.manifest_path,
        }
    }

    pub fn output(&self) -> &Path {
        match self {
            Self::V1(spec) => &spec.output,
            Self::V2(spec) => &spec.output,
            Self::V3(spec) => &spec.output,
        }
    }

    pub fn tasks(&self) -> &[RunTaskSpec] {
        match self {
            Self::V1(spec) => &spec.tasks,
            Self::V2(spec) => &spec.tasks,
            Self::V3(spec) => &spec.tasks,
        }
    }
}

/// A recursive YAML value whose map visitor rejects duplicate keys before a
/// normal `Mapping` can overwrite them.
#[derive(Clone, Debug, PartialEq)]
struct StrictValue(Value);

impl<'de> Deserialize<'de> for StrictValue {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        deserializer.deserialize_any(StrictValueVisitor)
    }
}

struct StrictValueVisitor;

impl<'de> Visitor<'de> for StrictValueVisitor {
    type Value = StrictValue;

    fn expecting(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("a YAML value")
    }

    fn visit_unit<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_none<E>(self) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Null))
    }

    fn visit_bool<E>(self, value: bool) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Bool(value)))
    }

    fn visit_i64<E>(self, value: i64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_u64<E>(self, value: u64) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_f64<E>(self, value: f64) -> Result<Self::Value, E>
    where
        E: de::Error,
    {
        Ok(StrictValue(Value::Number(Number::from(value))))
    }

    fn visit_str<E>(self, value: &str) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value.to_owned())))
    }

    fn visit_string<E>(self, value: String) -> Result<Self::Value, E> {
        Ok(StrictValue(Value::String(value)))
    }

    fn visit_seq<A>(self, mut sequence: A) -> Result<Self::Value, A::Error>
    where
        A: SeqAccess<'de>,
    {
        let mut values = Vec::new();
        while let Some(value) = sequence.next_element::<StrictValue>()? {
            values.push(value.0);
        }
        Ok(StrictValue(Value::Sequence(values)))
    }

    fn visit_map<A>(self, mut source: A) -> Result<Self::Value, A::Error>
    where
        A: MapAccess<'de>,
    {
        let mut mapping = Mapping::new();
        let mut seen = Vec::<Value>::new();
        while let Some((key, value)) = source.next_entry::<StrictValue, StrictValue>()? {
            if seen.iter().any(|existing| existing == &key.0) {
                return Err(de::Error::custom(format!(
                    "found duplicate key {}",
                    yaml_key_repr(&key.0)
                )));
            }
            seen.push(key.0.clone());
            mapping.insert(key.0, value.0);
        }
        Ok(StrictValue(Value::Mapping(mapping)))
    }
}

fn yaml_key_repr(value: &Value) -> String {
    match value {
        Value::String(value) => format!("'{value}'"),
        other => serde_yaml::to_string(other)
            .unwrap_or_else(|_| "<unprintable>".to_owned())
            .trim()
            .to_owned(),
    }
}

fn mapping<'a>(value: &'a Value, field: &str) -> Result<&'a Mapping, RunSpecError> {
    value
        .as_mapping()
        .ok_or_else(|| RunSpecError::new(format!("{field} must be an object")))
}

fn map_get<'a>(mapping: &'a Mapping, key: &str) -> Option<&'a Value> {
    mapping
        .iter()
        .find_map(|(candidate, value)| match candidate {
            Value::String(candidate) if candidate == key => Some(value),
            _ => None,
        })
}

fn object<'a>(
    value: &'a Value,
    field: &str,
    required: &[&str],
    optional: &[&str],
) -> Result<&'a Mapping, RunSpecError> {
    let mapping = mapping(value, field)?;
    let mut keys = Vec::with_capacity(mapping.len());
    for key in mapping.keys() {
        let Value::String(key) = key else {
            return Err(RunSpecError::new(format!("{field} keys must be strings")));
        };
        keys.push(key.as_str());
    }
    let mut missing: Vec<_> = required
        .iter()
        .copied()
        .filter(|required| !keys.contains(required))
        .collect();
    missing.sort_unstable();
    if !missing.is_empty() {
        return Err(RunSpecError::new(format!(
            "{field} is missing: {}",
            missing.join(", ")
        )));
    }
    let mut unknown: Vec<_> = keys
        .into_iter()
        .filter(|key| !required.contains(key) && !optional.contains(key))
        .collect();
    unknown.sort_unstable();
    if !unknown.is_empty() {
        return Err(RunSpecError::new(format!(
            "{field} has unknown fields: {}",
            unknown.join(", ")
        )));
    }
    Ok(mapping)
}

fn required<'a>(mapping: &'a Mapping, key: &str) -> &'a Value {
    map_get(mapping, key).expect("required field checked by object")
}

fn text(value: &Value, field: &str) -> Result<String, RunSpecError> {
    let Value::String(value) = value else {
        return Err(RunSpecError::new(format!(
            "{field} must be a non-empty trimmed string"
        )));
    };
    if value.is_empty() || value.trim() != value {
        return Err(RunSpecError::new(format!(
            "{field} must be a non-empty trimmed string"
        )));
    }
    Ok(value.clone())
}

fn identifier(value: &Value, field: &str) -> Result<String, RunSpecError> {
    let value = text(value, field)?;
    let mut bytes = value.bytes();
    let valid = (1..=128).contains(&value.len())
        && bytes
            .next()
            .map(|byte| byte.is_ascii_alphanumeric())
            .unwrap_or(false)
        && bytes.all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'.' | b'_' | b'-'))
        && !matches!(value.as_str(), "." | "..");
    if !valid {
        return Err(RunSpecError::new(format!(
            "{field} must be a safe 1-128 character identifier"
        )));
    }
    Ok(value)
}

fn resolved_path(path: &Path) -> Result<PathBuf, std::io::Error> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        env::current_dir()?.join(path)
    };
    let mut pending: VecDeque<OsString> = absolute
        .components()
        .filter_map(|component| match component {
            Component::Normal(part) => Some(part.to_os_string()),
            Component::ParentDir => Some(OsString::from("..")),
            _ => None,
        })
        .collect();
    let mut resolved = PathBuf::from("/");
    let mut followed = 0_u32;
    while let Some(component) = pending.pop_front() {
        if component == OsStr::new("..") {
            resolved.pop();
            continue;
        }
        resolved.push(&component);
        match fs::symlink_metadata(&resolved) {
            Ok(metadata) if metadata.file_type().is_symlink() => {
                followed += 1;
                if followed > 40 {
                    return Err(std::io::Error::from_raw_os_error(libc::ELOOP));
                }
                let target = fs::read_link(&resolved)?;
                resolved.pop();
                if target.is_absolute() {
                    resolved = PathBuf::from("/");
                }
                let components: Vec<_> = target
                    .components()
                    .filter_map(|component| match component {
                        Component::Normal(part) => Some(part.to_os_string()),
                        Component::ParentDir => Some(OsString::from("..")),
                        _ => None,
                    })
                    .collect();
                for component in components.into_iter().rev() {
                    pending.push_front(component);
                }
            }
            Ok(_) => {}
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error),
        }
    }
    Ok(resolved)
}

fn path_value(value: &Value, field: &str, base: &Path) -> Result<PathBuf, RunSpecError> {
    let raw = PathBuf::from(text(value, field)?);
    let candidate = if raw.is_absolute() {
        raw
    } else {
        base.join(raw)
    };
    resolved_path(&candidate).map_err(Into::into)
}

fn sha256_bytes(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn file_sha256(path: &Path, field: &str) -> Result<String, RunSpecError> {
    fs::read(path)
        .map(|bytes| sha256_bytes(&bytes))
        .map_err(|error| {
            RunSpecError::new(format!(
                "unable to read {field} {}: {error}",
                path.display()
            ))
        })
}

fn parse_cpuset(raw: &str) -> Result<Vec<u32>, String> {
    let mut cpu_ids = BTreeSet::new();
    for raw_part in raw.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            return Err(format!("invalid empty CPU set component in '{raw}'"));
        }
        if let Some((start, end)) = part.split_once('-') {
            let start = start
                .parse::<i64>()
                .map_err(|_| format!("invalid CPU set range: '{part}'"))?;
            let end = end
                .parse::<i64>()
                .map_err(|_| format!("invalid CPU set range: '{part}'"))?;
            if start < 0 || end < start || end > u32::MAX as i64 {
                return Err(format!("invalid CPU set range: '{part}'"));
            }
            cpu_ids.extend(start as u32..=end as u32);
        } else {
            let cpu_id = part
                .parse::<i64>()
                .map_err(|_| format!("invalid CPU ID: '{part}'"))?;
            if cpu_id < 0 || cpu_id > u32::MAX as i64 {
                return Err(format!("invalid CPU ID: '{part}'"));
            }
            cpu_ids.insert(cpu_id as u32);
        }
    }
    if cpu_ids.is_empty() {
        return Err("CPU set must not be empty".to_owned());
    }
    Ok(cpu_ids.into_iter().collect())
}

fn canonical_cpuset(value: &Value, field: &str) -> Result<(String, Vec<u32>), RunSpecError> {
    let raw = text(value, field)?;
    let cpu_ids = parse_cpuset(&raw)
        .map_err(|error| RunSpecError::new(format!("invalid {field}: {error}")))?;
    let mut ranges = Vec::new();
    let mut start = cpu_ids[0];
    let mut previous = start;
    for &cpu in &cpu_ids[1..] {
        if cpu == previous.saturating_add(1) {
            previous = cpu;
        } else {
            ranges.push((start, previous));
            start = cpu;
            previous = cpu;
        }
    }
    ranges.push((start, previous));
    let canonical = ranges
        .into_iter()
        .map(|(start, end)| {
            if start == end {
                start.to_string()
            } else {
                format!("{start}-{end}")
            }
        })
        .collect::<Vec<_>>()
        .join(",");
    Ok((canonical, cpu_ids))
}

fn positive_timeout(value: &Value, field: &str) -> Result<f64, RunSpecError> {
    if matches!(value, Value::Bool(_)) {
        return Err(RunSpecError::new(format!("{field} must be a number")));
    }
    let Some(timeout) = value.as_f64() else {
        return Err(RunSpecError::new(format!("{field} must be a number")));
    };
    if !timeout.is_finite() || timeout <= 0.0 {
        return Err(RunSpecError::new(format!(
            "{field} must be a finite positive number"
        )));
    }
    Ok(timeout)
}

fn positive_integer(value: &Value, field: &str) -> Result<usize, RunSpecError> {
    let integer = match value {
        Value::Number(number) => number.as_u64(),
        _ => None,
    };
    let Some(integer) = integer.filter(|integer| *integer > 0) else {
        return Err(RunSpecError::new(format!(
            "{field} must be a positive integer"
        )));
    };
    usize::try_from(integer)
        .map_err(|_| RunSpecError::new(format!("{field} must be a positive integer")))
}

fn normalize_allowed_host(value: &str) -> Result<String, String> {
    if value.is_empty() {
        return Err("allowed hosts must not be empty".to_owned());
    }
    if value.trim() != value {
        return Err(format!(
            "allowed host '{value}' must not contain surrounding whitespace"
        ));
    }
    if !value.is_ascii() {
        return Err(format!(
            "allowed host '{value}' must be an ASCII DNS hostname"
        ));
    }
    let host = value.to_ascii_lowercase();
    if host.ends_with('.') {
        return Err(format!(
            "allowed host '{value}' must not use a trailing DNS root dot"
        ));
    }
    if host.len() > 253 {
        return Err(format!(
            "allowed host '{value}' exceeds the 253-byte DNS hostname limit"
        ));
    }
    if host.parse::<IpAddr>().is_ok() {
        return Err(format!(
            "allowed host '{value}' must be a DNS hostname, not an IP literal"
        ));
    }
    let labels: Vec<_> = host.split('.').collect();
    let valid_label = |label: &str| {
        !label.is_empty()
            && label.len() <= 63
            && label
                .bytes()
                .all(|byte| byte.is_ascii_lowercase() || byte.is_ascii_digit() || byte == b'-')
            && label
                .as_bytes()
                .first()
                .is_some_and(u8::is_ascii_alphanumeric)
            && label
                .as_bytes()
                .last()
                .is_some_and(u8::is_ascii_alphanumeric)
    };
    if labels.iter().any(|label| !valid_label(label)) {
        return Err(format!(
            "allowed host '{value}' is not an exact ASCII DNS hostname; URLs, wildcards, ports, underscores, and empty labels are unsupported"
        ));
    }
    if host == "localhost" || host.ends_with(".localhost") {
        return Err(format!(
            "allowed host '{value}' uses the prohibited localhost namespace"
        ));
    }
    if host == "local" || host.ends_with(".local") {
        return Err(format!(
            "allowed host '{value}' uses the prohibited local namespace"
        ));
    }
    if labels.len() < 2 {
        return Err(format!(
            "allowed host '{value}' must be a multi-label DNS hostname"
        ));
    }
    Ok(host)
}

fn load_tasks(
    value: &Value,
    base: &Path,
    image_base: Option<&Path>,
) -> Result<Vec<RunTaskSpec>, RunSpecError> {
    let Some(raw_tasks) = value.as_sequence() else {
        return Err(RunSpecError::new("tasks must be a non-empty list"));
    };
    if raw_tasks.is_empty() {
        return Err(RunSpecError::new("tasks must be a non-empty list"));
    }
    let mut tasks = Vec::with_capacity(raw_tasks.len());
    let mut seen = BTreeSet::new();
    for (index, raw_task) in raw_tasks.iter().enumerate() {
        let field = format!("tasks[{index}]");
        let item = object(
            raw_task,
            &field,
            &["id", "atif", "image"],
            &[
                "expected_source_sha256",
                "expected_image_id",
                "action_timeout_sec",
            ],
        )?;
        let task_id = identifier(required(item, "id"), &format!("{field}.id"))?;
        if !seen.insert(task_id.clone()) {
            return Err(RunSpecError::new(format!("duplicate task id: {task_id}")));
        }
        let atif = path_value(required(item, "atif"), &format!("{field}.atif"), base)?;
        if !atif.is_file() {
            return Err(RunSpecError::new(format!(
                "{field}.atif is not a file: {}",
                atif.display()
            )));
        }
        let source_sha256 = file_sha256(&atif, &format!("{field}.atif"))?;
        if let Some(expected) =
            map_get(item, "expected_source_sha256").filter(|value| !value.is_null())
        {
            let expected =
                text(expected, &format!("{field}.expected_source_sha256"))?.to_ascii_lowercase();
            if expected.len() != 64
                || !expected
                    .bytes()
                    .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
            {
                return Err(RunSpecError::new(format!(
                    "{field}.expected_source_sha256 must be a SHA-256 hex digest"
                )));
            }
            if expected != source_sha256 {
                return Err(RunSpecError::new(format!(
                    "{field}.atif SHA-256 mismatch: expected {expected}, got {source_sha256}"
                )));
            }
        }
        let expected_image_id = map_get(item, "expected_image_id")
            .filter(|value| !value.is_null())
            .map(|value| text(value, &format!("{field}.expected_image_id")))
            .transpose()?;
        let action_timeout_sec = map_get(item, "action_timeout_sec")
            .filter(|value| !value.is_null())
            .map(|value| positive_timeout(value, &format!("{field}.action_timeout_sec")))
            .transpose()?;
        let image = if let Some(image_base) = image_base {
            let resolved = path_value(
                required(item, "image"),
                &format!("{field}.image"),
                image_base,
            )?;
            resolved.into_os_string().into_string().map_err(|_| {
                RunSpecError::new(format!("{field}.image must resolve to a UTF-8 path"))
            })?
        } else {
            text(required(item, "image"), &format!("{field}.image"))?
        };
        tasks.push(RunTaskSpec {
            task_id,
            atif,
            source_sha256,
            image,
            expected_image_id,
            action_timeout_sec,
        });
    }
    Ok(tasks)
}

fn load_defaults(value: &Value, input_root: &Path) -> Result<RunDefaultsSpec, RunSpecError> {
    let field = "defaults";
    let item = object(
        value,
        field,
        &[
            "pacing",
            "network_policy",
            "profile_sets",
            "controller_runtime",
            "controller_cpus",
            "action_timeout_sec",
        ],
        &[
            "allowed_hosts",
            "runtime",
            "worker_cpus",
            "placement",
            "workdir",
            "environment",
            "environment_file",
            "environment_file_sha256",
        ],
    )?;
    let pacing =
        crate::pacing::canonical(&text(required(item, "pacing"), "defaults.pacing")?).to_owned();
    if !PACING_MODES.contains(&pacing.as_str()) {
        return Err(RunSpecError::new(format!(
            "defaults.pacing must be one of: {}",
            PACING_MODES.join(", ")
        )));
    }

    let network_policy = text(required(item, "network_policy"), "defaults.network_policy")?;
    if !NETWORK_MODES.contains(&network_policy.as_str()) {
        return Err(RunSpecError::new(format!(
            "invalid defaults.network_policy: network mode must be one of: {}",
            NETWORK_MODES.join(", ")
        )));
    }
    let allowed_values = match map_get(item, "allowed_hosts") {
        None => &[][..],
        Some(Value::Sequence(values)) => values,
        Some(_) => return Err(RunSpecError::new("defaults.allowed_hosts must be a list")),
    };
    let mut allowed_hosts = BTreeSet::new();
    for (index, host) in allowed_values.iter().enumerate() {
        let host = text(host, &format!("defaults.allowed_hosts[{index}]"))?;
        let host = normalize_allowed_host(&host).map_err(|error| {
            RunSpecError::new(format!("invalid defaults.network_policy: {error}"))
        })?;
        allowed_hosts.insert(host);
    }
    if network_policy != "allowlist" && !allowed_hosts.is_empty() {
        return Err(RunSpecError::new(format!(
            "invalid defaults.network_policy: allowed hosts require mode='allowlist'; mode='{network_policy}' does not accept them"
        )));
    }

    let runtime = map_get(item, "runtime")
        .map(|value| text(value, "defaults.runtime"))
        .transpose()?
        .unwrap_or_else(|| "docker".to_owned());
    if !CONTAINER_RUNTIMES.contains(&runtime.as_str()) {
        return Err(RunSpecError::new(format!(
            "defaults.runtime must be one of: {}",
            CONTAINER_RUNTIMES.join(", ")
        )));
    }
    if runtime == "enroot" && !matches!(network_policy.as_str(), "public" | "no-network") {
        return Err(RunSpecError::new(
            "defaults.network_policy must be public or no-network for enroot runtime",
        ));
    }

    let Some(raw_profile_sets) = required(item, "profile_sets").as_sequence() else {
        return Err(RunSpecError::new("defaults.profile_sets must be a list"));
    };
    if runtime == "docker" && raw_profile_sets.is_empty() {
        return Err(RunSpecError::new(
            "defaults.profile_sets must be a non-empty list",
        ));
    }
    let profile_sets: Vec<_> = raw_profile_sets
        .iter()
        .enumerate()
        .map(|(index, value)| text(value, &format!("defaults.profile_sets[{index}]")))
        .collect::<Result<_, _>>()?;
    if profile_sets.iter().collect::<BTreeSet<_>>().len() != profile_sets.len() {
        return Err(RunSpecError::new(
            "defaults.profile_sets must not contain duplicates",
        ));
    }
    let unknown: BTreeSet<_> = profile_sets
        .iter()
        .filter(|value| !PROFILE_SETS.contains(&value.as_str()))
        .cloned()
        .collect();
    if !unknown.is_empty() {
        return Err(RunSpecError::new(format!(
            "defaults.profile_sets contains unsupported values: {}",
            unknown.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    if runtime == "enroot" && !profile_sets.is_empty() {
        return Err(RunSpecError::new(
            "defaults.profile_sets must be empty for enroot runtime",
        ));
    }
    if runtime == "docker"
        && !profile_sets
            .iter()
            .any(|profile_set| profile_set == "cpu-average")
    {
        return Err(RunSpecError::new(
            "defaults.profile_sets must include cpu-average",
        ));
    }

    let controller_runtime = text(
        required(item, "controller_runtime"),
        "defaults.controller_runtime",
    )?;
    if RETIRED_CONTROLLER_RUNTIME_MODES.contains(&controller_runtime.as_str()) {
        return Err(RunSpecError::new(format!(
            "defaults.controller_runtime '{controller_runtime}' is a retired Python-backed run controller mode; native-only tasktrace run requires '{NATIVE_CONTROLLER_RUNTIME}'; migrate the manifest before retrying"
        )));
    }
    if controller_runtime != NATIVE_CONTROLLER_RUNTIME {
        return Err(RunSpecError::new(format!(
            "defaults.controller_runtime must be '{NATIVE_CONTROLLER_RUNTIME}' for native-only run execution"
        )));
    }
    let (controller_cpus, controller_cpu_ids) = canonical_cpuset(
        required(item, "controller_cpus"),
        "defaults.controller_cpus",
    )?;
    let has_worker_cpus = map_get(item, "worker_cpus").is_some();
    let has_placement = map_get(item, "placement").is_some();
    if has_worker_cpus != has_placement {
        return Err(RunSpecError::new(
            "defaults.worker_cpus and defaults.placement must be specified together",
        ));
    }
    let mut worker_cpus = None;
    let mut worker_cpu_ids = Vec::new();
    let mut placement = None;
    if has_worker_cpus {
        let (canonical, ids) = canonical_cpuset(
            map_get(item, "worker_cpus").expect("present"),
            "defaults.worker_cpus",
        )?;
        let selected = text(
            map_get(item, "placement").expect("present"),
            "defaults.placement",
        )?;
        if selected != ROUND_ROBIN_SINGLE_CPU {
            return Err(RunSpecError::new(format!(
                "defaults.placement must be '{ROUND_ROBIN_SINGLE_CPU}'"
            )));
        }
        let overlap: Vec<_> = controller_cpu_ids
            .iter()
            .copied()
            .filter(|cpu| ids.contains(cpu))
            .collect();
        if !overlap.is_empty() {
            return Err(RunSpecError::new(format!(
                "defaults.worker_cpus overlaps defaults.controller_cpus: {}",
                overlap
                    .into_iter()
                    .map(|cpu| cpu.to_string())
                    .collect::<Vec<_>>()
                    .join(",")
            )));
        }
        worker_cpus = Some(canonical);
        worker_cpu_ids = ids;
        placement = Some(selected);
    }
    let action_timeout_sec = positive_timeout(
        required(item, "action_timeout_sec"),
        "defaults.action_timeout_sec",
    )?;
    let workdir = map_get(item, "workdir")
        .map(|value| text(value, "defaults.workdir"))
        .transpose()?;
    let mut environment = Vec::new();
    // File-sourced contract first (the collection's sealed recording
    // environment); inline manifest entries follow and therefore override.
    let has_file = map_get(item, "environment_file").is_some();
    let has_file_sha = map_get(item, "environment_file_sha256").is_some();
    if has_file_sha && !has_file {
        return Err(RunSpecError::new(
            "defaults.environment_file_sha256 requires defaults.environment_file",
        ));
    }
    if has_file {
        let file = path_value(
            map_get(item, "environment_file").expect("present"),
            "defaults.environment_file",
            input_root,
        )?;
        let raw_bytes = fs::read(&file).map_err(|error| {
            RunSpecError::new(format!(
                "unable to read defaults.environment_file {}: {error}",
                file.display()
            ))
        })?;
        if let Some(expected) = map_get(item, "environment_file_sha256") {
            let expected =
                text(expected, "defaults.environment_file_sha256")?.to_ascii_lowercase();
            let actual = sha256_bytes(&raw_bytes);
            if expected != actual {
                return Err(RunSpecError::new(format!(
                    "defaults.environment_file SHA-256 mismatch: expected {expected}, got {actual}"
                )));
            }
        }
        let contents = String::from_utf8(raw_bytes).map_err(|_| {
            RunSpecError::new("defaults.environment_file must be UTF-8")
        })?;
        for (line_number, line) in contents.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let name = line.split('=').next().unwrap_or("");
            if name.is_empty() || name.contains(char::is_whitespace) {
                return Err(RunSpecError::new(format!(
                    "defaults.environment_file line {} must be NAME=VALUE or bare NAME",
                    line_number + 1
                )));
            }
            environment.push(line.to_owned());
        }
    }
    if let Some(raw) = map_get(item, "environment") {
        let Some(entries) = raw.as_sequence() else {
            return Err(RunSpecError::new("defaults.environment must be a list"));
        };
        for (index, entry) in entries.iter().enumerate() {
            let assignment = text(entry, &format!("defaults.environment[{index}]"))?;
            let name = assignment.split('=').next().unwrap_or("");
            if name.is_empty() {
                return Err(RunSpecError::new(format!(
                    "defaults.environment[{index}] must be NAME=VALUE (or bare NAME to unset)"
                )));
            }
            environment.push(assignment);
        }
    }
    if let Some(workdir) = &workdir {
        if !workdir.starts_with('/') {
            return Err(RunSpecError::new(
                "defaults.workdir must be an absolute in-container path",
            ));
        }
    }
    Ok(RunDefaultsSpec {
        pacing,
        network_policy,
        allowed_hosts: allowed_hosts.into_iter().collect(),
        runtime,
        profile_sets,
        controller_runtime,
        controller_cpus,
        controller_cpu_ids,
        action_timeout_sec,
        worker_cpus,
        worker_cpu_ids,
        placement,
        workdir,
        environment,
    })
}

fn metadata_scalar(value: &Value, field: &str) -> Result<MetadataScalar, RunSpecError> {
    match value {
        Value::String(_) => Ok(MetadataScalar::String(text(value, field)?)),
        Value::Bool(value) => Ok(MetadataScalar::Boolean(*value)),
        Value::Number(number) if number.as_i64().is_some() => Ok(MetadataScalar::Integer(
            number.as_i64().expect("checked") as i128,
        )),
        Value::Number(number) if number.as_u64().is_some() => Ok(MetadataScalar::Integer(
            number.as_u64().expect("checked") as i128,
        )),
        Value::Number(number) if number.as_f64().is_some_and(f64::is_finite) => {
            Ok(MetadataScalar::Number(number.as_f64().expect("checked")))
        }
        _ => Err(RunSpecError::new(format!(
            "{field} must be a string, boolean, integer, or finite number"
        ))),
    }
}

fn load_waves(
    value: &Value,
    tasks: &[RunTaskSpec],
    defaults: &RunDefaultsSpec,
) -> Result<Vec<RunWaveSpec>, RunSpecError> {
    let Some(raw_waves) = value.as_sequence() else {
        return Err(RunSpecError::new("waves must be a non-empty list"));
    };
    if raw_waves.is_empty() {
        return Err(RunSpecError::new("waves must be a non-empty list"));
    }
    let task_ids: BTreeSet<_> = tasks.iter().map(|task| task.task_id.clone()).collect();
    let controller_cpu_ids: BTreeSet<_> = defaults.controller_cpu_ids.iter().copied().collect();
    let compact = defaults.placement.is_some();
    let mut seen_wave_ids = BTreeSet::new();
    let mut referenced_task_ids = BTreeSet::new();
    let mut waves = Vec::with_capacity(raw_waves.len());
    for (wave_index, raw_wave) in raw_waves.iter().enumerate() {
        let field = format!("waves[{wave_index}]");
        if let Some(raw_mapping) = raw_wave.as_mapping() {
            let has = |key| map_get(raw_mapping, key).is_some();
            if compact && (has("metadata") || has("launches")) {
                return Err(RunSpecError::new(
                    "compact defaults cannot be combined with explicit waves; all waves must use id/allocation/repeat form",
                ));
            }
            if !compact && (has("allocation") || has("repeat")) {
                return Err(RunSpecError::new(
                    "compact waves require defaults.worker_cpus and defaults.placement; all waves must use one form",
                ));
            }
        }
        let item = if compact {
            object(
                raw_wave,
                &field,
                &["id", "allocation", "repeat"],
                &["scored"],
            )?
        } else {
            object(
                raw_wave,
                &field,
                &["id", "scored", "metadata", "launches"],
                &[],
            )?
        };
        let wave_id = identifier(required(item, "id"), &format!("{field}.id"))?;
        if !seen_wave_ids.insert(wave_id.clone()) {
            return Err(RunSpecError::new(format!("duplicate wave id: {wave_id}")));
        }
        let scored = match map_get(item, "scored") {
            None => true,
            Some(Value::Bool(scored)) => *scored,
            Some(_) => {
                return Err(RunSpecError::new(format!(
                    "{field}.scored must be a boolean"
                )))
            }
        };
        if compact {
            let allocation =
                positive_integer(required(item, "allocation"), &format!("{field}.allocation"))?;
            let repeat = positive_integer(required(item, "repeat"), &format!("{field}.repeat"))?;
            if allocation > defaults.worker_cpu_ids.len() {
                return Err(RunSpecError::new(format!(
                    "{field}.allocation ({allocation}) exceeds the number of defaults.worker_cpus ({})",
                    defaults.worker_cpu_ids.len()
                )));
            }
            let launches = tasks
                .iter()
                .enumerate()
                .map(|(index, task)| {
                    let cpu = defaults.worker_cpu_ids[index % allocation];
                    referenced_task_ids.insert(task.task_id.clone());
                    RunLaunchSpec {
                        task_id: task.task_id.clone(),
                        cpus: cpu.to_string(),
                        cpu_ids: vec![cpu],
                    }
                })
                .collect();
            waves.push(RunWaveSpec {
                wave_id,
                scored,
                metadata: vec![
                    (
                        "allocation".to_owned(),
                        MetadataScalar::Integer(allocation as i128),
                    ),
                    ("repeat".to_owned(), MetadataScalar::Integer(repeat as i128)),
                ],
                launches,
            });
            continue;
        }

        let raw_metadata = required(item, "metadata");
        let Some(metadata_mapping) = raw_metadata.as_mapping() else {
            return Err(RunSpecError::new(format!(
                "{field}.metadata must be an object"
            )));
        };
        let mut metadata = Vec::with_capacity(metadata_mapping.len());
        for (raw_key, raw_value) in metadata_mapping {
            let Value::String(key) = raw_key else {
                return Err(RunSpecError::new(format!(
                    "{field}.metadata keys must be strings"
                )));
            };
            let metadata_key = identifier(
                &Value::String(key.clone()),
                &format!("{field}.metadata key"),
            )?;
            metadata.push((
                metadata_key.clone(),
                metadata_scalar(raw_value, &format!("{field}.metadata.{metadata_key}"))?,
            ));
        }

        let Some(raw_launches) = required(item, "launches").as_sequence() else {
            return Err(RunSpecError::new(format!(
                "{field}.launches must be a non-empty list"
            )));
        };
        if raw_launches.is_empty() {
            return Err(RunSpecError::new(format!(
                "{field}.launches must be a non-empty list"
            )));
        }
        let mut launches = Vec::with_capacity(raw_launches.len());
        let mut wave_task_ids = BTreeSet::new();
        for (launch_index, raw_launch) in raw_launches.iter().enumerate() {
            let launch_field = format!("{field}.launches[{launch_index}]");
            let launch = object(raw_launch, &launch_field, &["task", "cpus"], &[])?;
            let task_id = identifier(required(launch, "task"), &format!("{launch_field}.task"))?;
            if !task_ids.contains(&task_id) {
                return Err(RunSpecError::new(format!(
                    "{launch_field}.task references unknown task: {task_id}"
                )));
            }
            if !wave_task_ids.insert(task_id.clone()) {
                return Err(RunSpecError::new(format!(
                    "{field} launches task more than once: {task_id}"
                )));
            }
            referenced_task_ids.insert(task_id.clone());
            let (cpus, cpu_ids) =
                canonical_cpuset(required(launch, "cpus"), &format!("{launch_field}.cpus"))?;
            let overlap: Vec<_> = cpu_ids
                .iter()
                .copied()
                .filter(|cpu| controller_cpu_ids.contains(cpu))
                .collect();
            if !overlap.is_empty() {
                return Err(RunSpecError::new(format!(
                    "{launch_field}.cpus overlaps defaults.controller_cpus: {}",
                    overlap
                        .into_iter()
                        .map(|cpu| cpu.to_string())
                        .collect::<Vec<_>>()
                        .join(",")
                )));
            }
            launches.push(RunLaunchSpec {
                task_id,
                cpus,
                cpu_ids,
            });
        }
        waves.push(RunWaveSpec {
            wave_id,
            scored,
            metadata,
            launches,
        });
    }
    let unused: Vec<_> = task_ids.difference(&referenced_task_ids).cloned().collect();
    if !unused.is_empty() {
        return Err(RunSpecError::new(format!(
            "task catalog contains unreferenced tasks: {}",
            unused.join(", ")
        )));
    }
    Ok(waves)
}

fn load_v3(payload: &Value, manifest_path: PathBuf, raw: &[u8]) -> Result<RunSpecV3, RunSpecError> {
    let root = object(
        payload,
        "run",
        &["schema_version", "output", "defaults", "tasks", "waves"],
        &["input_root"],
    )?;
    let schema_version = text(required(root, "schema_version"), "schema_version")?;
    if schema_version != SCHEMA_VERSION_V3 {
        return Err(RunSpecError::new(format!(
            "unsupported run schema: '{schema_version}'"
        )));
    }
    let base = manifest_path
        .parent()
        .expect("resolved manifest always has parent");
    let input_root = match map_get(root, "input_root") {
        Some(value) => path_value(value, "input_root", base)?,
        None => base.to_path_buf(),
    };
    let defaults = load_defaults(required(root, "defaults"), &input_root)?;
    let tasks = load_tasks(
        required(root, "tasks"),
        &input_root,
        (defaults.runtime == "enroot").then_some(base),
    )?;
    let waves = load_waves(required(root, "waves"), &tasks, &defaults)?;
    let output = path_value(required(root, "output"), "output", base)?;
    Ok(RunSpecV3 {
        schema_version,
        manifest_path,
        manifest_sha256: sha256_bytes(raw),
        output,
        input_root,
        defaults,
        tasks,
        waves,
    })
}

/// Load, normalize, and content-bind one supported run YAML file.
pub fn load_run_spec(manifest_path: &Path) -> Result<RunSpec, RunSpecError> {
    let manifest_path = resolved_path(manifest_path)?;
    let raw = fs::read(&manifest_path).map_err(|error| {
        RunSpecError::new(format!(
            "unable to read run file {}: {error}",
            manifest_path.display()
        ))
    })?;
    let payload = serde_yaml::from_slice::<StrictValue>(&raw)
        .map(|value| value.0)
        .map_err(|error| {
            RunSpecError::new(format!(
                "unable to read run file {}: {error}",
                manifest_path.display()
            ))
        })?;
    let declared_version = payload
        .as_mapping()
        .and_then(|mapping| map_get(mapping, "schema_version"));
    if declared_version.and_then(Value::as_str) == Some(SCHEMA_VERSION_V3) {
        return load_v3(&payload, manifest_path, &raw)
            .map(Box::new)
            .map(RunSpec::V3);
    }

    let mut required_fields = vec!["schema_version", "output", "cpus", "tasks"];
    if declared_version.and_then(Value::as_str) == Some(SCHEMA_VERSION_V2) {
        required_fields.push("concurrency");
    }
    let root = object(&payload, "run", &required_fields, &["repetitions"])?;
    let schema_version = text(required(root, "schema_version"), "schema_version")?;
    if !matches!(
        schema_version.as_str(),
        SCHEMA_VERSION_V1 | SCHEMA_VERSION_V2
    ) {
        return Err(RunSpecError::new(format!(
            "unsupported run schema: '{schema_version}'"
        )));
    }
    let concurrency = if schema_version == SCHEMA_VERSION_V1 {
        1
    } else {
        positive_integer(required(root, "concurrency"), "concurrency")?
    };
    let repetitions = map_get(root, "repetitions")
        .map(|value| positive_integer(value, "repetitions"))
        .transpose()?
        .unwrap_or(1);
    let (cpus, cpu_ids) = canonical_cpuset(required(root, "cpus"), "cpus")?;
    let base = manifest_path
        .parent()
        .expect("resolved manifest always has parent");
    let tasks = load_tasks(required(root, "tasks"), base, None)?;
    let attempt_count = tasks.len().saturating_mul(repetitions);
    if concurrency > attempt_count {
        return Err(RunSpecError::new(format!(
            "concurrency cannot exceed the declared task-attempt count ({attempt_count})"
        )));
    }
    let manifest_sha256 = sha256_bytes(&raw);
    let output = path_value(required(root, "output"), "output", base)?;
    if schema_version == SCHEMA_VERSION_V1 {
        Ok(RunSpec::V1(RunSpecV1 {
            schema_version,
            manifest_path,
            manifest_sha256,
            output,
            cpus,
            cpu_ids,
            concurrency,
            repetitions,
            tasks,
        }))
    } else {
        Ok(RunSpec::V2(RunSpecV2 {
            schema_version,
            manifest_path,
            manifest_sha256,
            output,
            cpus,
            cpu_ids,
            concurrency,
            repetitions,
            tasks,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_yaml::Value;
    use std::io::Write;

    fn yaml(source: &str) -> Value {
        serde_yaml::from_str(source).unwrap()
    }

    fn map(value: &Value) -> &Mapping {
        value.as_mapping().unwrap()
    }

    fn map_mut(value: &mut Value) -> &mut Mapping {
        value.as_mapping_mut().unwrap()
    }

    fn get_mut<'a>(value: &'a mut Value, key: &str) -> &'a mut Value {
        map_mut(value)
            .iter_mut()
            .find_map(|(candidate, value)| match candidate {
                Value::String(candidate) if candidate == key => Some(value),
                _ => None,
            })
            .unwrap()
    }

    fn set(value: &mut Value, key: &str, replacement: Value) {
        map_mut(value).insert(Value::String(key.to_owned()), replacement);
    }

    fn remove(value: &mut Value, key: &str) {
        map_mut(value).remove(&Value::String(key.to_owned()));
    }

    fn write_manifest(root: &Path, payload: &Value) -> PathBuf {
        let path = root.join("run.yaml");
        fs::write(&path, serde_yaml::to_string(payload).unwrap()).unwrap();
        path
    }

    fn base_v1(root: &Path) -> Value {
        fs::write(root.join("one.json"), b"{\"task\": 1}\n").unwrap();
        fs::write(root.join("two.json"), b"{\"task\": 2}\n").unwrap();
        yaml(
            r#"
schema_version: tasktrace-run.v1
output: results
cpus: 23,20-22,22
tasks:
  - id: one
    atif: one.json
    image: image-one
  - id: two
    atif: two.json
    image: image-two
"#,
        )
    }

    fn base_v3(root: &Path) -> Value {
        let inputs = root.join("inputs");
        fs::create_dir_all(&inputs).unwrap();
        fs::write(inputs.join("one.json"), b"{\"task\": 1}\n").unwrap();
        fs::write(inputs.join("two.json"), b"{\"task\": 2}\n").unwrap();
        let one_digest = file_sha256(&inputs.join("one.json"), "fixture").unwrap();
        let two_digest = file_sha256(&inputs.join("two.json"), "fixture").unwrap();
        serde_yaml::from_str(&format!(
            r#"
schema_version: tasktrace-run.v3
output: results/v3
defaults:
  pacing: llm-gaps
  network_policy: public
  profile_sets: [cpu-average]
  controller_runtime: rust-host-mount
  controller_cpus: 3,0-2
  action_timeout_sec: 900
tasks:
  - id: one
    atif: inputs/one.json
    expected_source_sha256: {one_digest}
    image: image-one@sha256:one
    expected_image_id: sha256:one
  - id: two
    atif: inputs/two.json
    expected_source_sha256: {two_digest}
    image: image-two@sha256:two
    expected_image_id: sha256:two
    action_timeout_sec: 1200
waves:
  - id: r1-n1
    scored: true
    metadata:
      repeat: 1
      allocation: 1
    launches:
      - task: one
        cpus: '4'
      - task: two
        cpus: '4'
  - id: r2-n2
    scored: false
    metadata:
      repeat: 2
      allocation: 2
    launches:
      - task: two
        cpus: '6'
      - task: one
        cpus: '5'
"#
        ))
        .unwrap()
    }

    fn compact_v3(root: &Path) -> Value {
        let mut payload = base_v3(root);
        let defaults = get_mut(&mut payload, "defaults");
        set(defaults, "worker_cpus", Value::String("6,4-5".to_owned()));
        set(
            defaults,
            "placement",
            Value::String(ROUND_ROBIN_SINGLE_CPU.to_owned()),
        );
        set(
            &mut payload,
            "waves",
            yaml(
                r#"
- id: r1-n1
  allocation: 1
  repeat: 1
- id: r2-n2
  allocation: 2
  repeat: 2
  scored: false
"#,
            ),
        );
        payload
    }

    fn error(root: &Path, payload: &Value) -> String {
        load_run_spec(&write_manifest(root, payload))
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn loads_v1_with_defaults_content_identity_and_order() {
        let root = tempfile::tempdir().unwrap();
        let manifest = write_manifest(root.path(), &base_v1(root.path()));
        let loaded = load_run_spec(&manifest).unwrap();
        let RunSpec::V1(spec) = loaded else {
            panic!("expected v1")
        };
        assert_eq!(spec.schema_version, SCHEMA_VERSION_V1);
        assert_eq!(spec.manifest_path, fs::canonicalize(&manifest).unwrap());
        assert_eq!(
            spec.manifest_sha256,
            file_sha256(&manifest, "manifest").unwrap()
        );
        assert_eq!(spec.output, root.path().join("results"));
        assert_eq!(spec.cpus, "20-23");
        assert_eq!(spec.cpu_ids, [20, 21, 22, 23]);
        assert_eq!(spec.concurrency, 1);
        assert_eq!(spec.repetitions, 1);
        assert_eq!(
            spec.tasks
                .iter()
                .map(|task| task.task_id.as_str())
                .collect::<Vec<_>>(),
            ["one", "two"]
        );
        assert_eq!(spec.tasks[0].atif, root.path().join("one.json"));
        assert_eq!(
            spec.tasks[0].source_sha256,
            file_sha256(&root.path().join("one.json"), "source").unwrap()
        );
        assert_eq!(spec.tasks[0].expected_image_id, None);
        assert_eq!(spec.tasks[0].action_timeout_sec, None);
    }

    #[test]
    fn loads_v1_task_assertions_and_v2_concurrency() {
        let root = tempfile::tempdir().unwrap();
        let mut payload = base_v1(root.path());
        set(&mut payload, "repetitions", yaml("3"));
        let digest = file_sha256(&root.path().join("one.json"), "source").unwrap();
        set(
            &mut payload,
            "tasks",
            serde_yaml::from_str(&format!(
                r#"
- id: task.one
  atif: one.json
  image: image@sha256:abc
  expected_source_sha256: {}
  expected_image_id: sha256:resolved
  action_timeout_sec: 12.5
"#,
                digest.to_ascii_uppercase()
            ))
            .unwrap(),
        );
        let RunSpec::V1(spec) = load_run_spec(&write_manifest(root.path(), &payload)).unwrap()
        else {
            panic!("expected v1")
        };
        assert_eq!(spec.repetitions, 3);
        assert_eq!(
            spec.tasks[0].expected_image_id.as_deref(),
            Some("sha256:resolved")
        );
        assert_eq!(spec.tasks[0].action_timeout_sec, Some(12.5));

        let mut payload = base_v1(root.path());
        set(
            &mut payload,
            "schema_version",
            Value::String(SCHEMA_VERSION_V2.to_owned()),
        );
        set(&mut payload, "concurrency", yaml("2"));
        let RunSpec::V2(spec) = load_run_spec(&write_manifest(root.path(), &payload)).unwrap()
        else {
            panic!("expected v2")
        };
        assert_eq!(spec.concurrency, 2);
        assert_eq!(spec.repetitions, 1);
    }

    #[test]
    fn v2_requires_valid_bounded_concurrency() {
        let root = tempfile::tempdir().unwrap();
        let mut payload = base_v1(root.path());
        set(
            &mut payload,
            "schema_version",
            Value::String(SCHEMA_VERSION_V2.to_owned()),
        );
        assert!(error(root.path(), &payload).contains("run is missing: concurrency"));
        for invalid in [
            yaml("true"),
            yaml("0"),
            yaml("-1"),
            yaml("1.5"),
            yaml("'3'"),
        ] {
            set(&mut payload, "concurrency", invalid);
            assert!(error(root.path(), &payload).contains("concurrency must be a positive integer"));
        }
        set(&mut payload, "concurrency", yaml("3"));
        assert!(error(root.path(), &payload)
            .contains("cannot exceed the declared task-attempt count (2)"));
    }

    #[test]
    fn loads_v3_explicit_ordered_waves() {
        let root = tempfile::tempdir().unwrap();
        let manifest = write_manifest(root.path(), &base_v3(root.path()));
        let RunSpec::V3(spec) = load_run_spec(&manifest).unwrap() else {
            panic!("expected v3")
        };
        assert_eq!(spec.schema_version, SCHEMA_VERSION_V3);
        assert_eq!(spec.output, root.path().join("results/v3"));
        assert_eq!(spec.input_root, root.path());
        assert_eq!(spec.defaults.pacing, "llm-gaps");
        assert_eq!(spec.defaults.network_policy, "public");
        assert!(spec.defaults.allowed_hosts.is_empty());
        assert_eq!(spec.defaults.runtime, "docker");
        assert_eq!(spec.defaults.profile_sets, ["cpu-average"]);
        assert_eq!(spec.defaults.controller_runtime, "rust-host-mount");
        assert_eq!(spec.defaults.controller_cpus, "0-3");
        assert_eq!(spec.defaults.controller_cpu_ids, [0, 1, 2, 3]);
        assert_eq!(spec.defaults.action_timeout_sec, 900.0);
        assert_eq!(spec.tasks[1].action_timeout_sec, Some(1200.0));
        assert_eq!(spec.waves[0].wave_id, "r1-n1");
        assert!(spec.waves[0].scored);
        assert!(!spec.waves[1].scored);
        assert_eq!(
            spec.waves[0].metadata,
            [
                ("repeat".to_owned(), MetadataScalar::Integer(1)),
                ("allocation".to_owned(), MetadataScalar::Integer(1)),
            ]
        );
        assert_eq!(spec.waves[0].launches[0].task_id, "one");
        assert_eq!(spec.waves[0].launches[1].cpus, "4");
        assert_eq!(spec.waves[1].launches[0].cpu_ids, [6]);
        assert_eq!(spec.tasks[0].image, "image-one@sha256:one");
    }

    #[test]
    fn compact_v3_expands_round_robin_launches() {
        let root = tempfile::tempdir().unwrap();
        let RunSpec::V3(spec) =
            load_run_spec(&write_manifest(root.path(), &compact_v3(root.path()))).unwrap()
        else {
            panic!("expected v3")
        };
        assert_eq!(spec.defaults.worker_cpus.as_deref(), Some("4-6"));
        assert_eq!(spec.defaults.worker_cpu_ids, [4, 5, 6]);
        assert_eq!(
            spec.defaults.placement.as_deref(),
            Some(ROUND_ROBIN_SINGLE_CPU)
        );
        assert!(spec.waves[0].scored);
        assert!(!spec.waves[1].scored);
        assert_eq!(
            spec.waves[0].metadata,
            [
                ("allocation".to_owned(), MetadataScalar::Integer(1)),
                ("repeat".to_owned(), MetadataScalar::Integer(1)),
            ]
        );
        assert_eq!(
            spec.waves[1]
                .launches
                .iter()
                .map(|launch| launch.cpus.as_str())
                .collect::<Vec<_>>(),
            ["4", "5"]
        );
    }

    #[test]
    fn v3_input_root_allowlist_and_profile_order_are_resolved() {
        let root = tempfile::tempdir().unwrap();
        let mut payload = base_v3(root.path());
        let portable = root.path().join("portable-inputs");
        fs::create_dir(&portable).unwrap();
        for name in ["one.json", "two.json"] {
            fs::copy(root.path().join("inputs").join(name), portable.join(name)).unwrap();
        }
        let tasks = get_mut(&mut payload, "tasks").as_sequence_mut().unwrap();
        for (index, task) in tasks.iter_mut().enumerate() {
            let name = if index == 0 { "one.json" } else { "two.json" };
            set(task, "atif", Value::String(name.to_owned()));
        }
        set(
            &mut payload,
            "input_root",
            Value::String("portable-inputs".to_owned()),
        );
        let defaults = get_mut(&mut payload, "defaults");
        set(
            defaults,
            "network_policy",
            Value::String("allowlist".to_owned()),
        );
        set(
            defaults,
            "allowed_hosts",
            yaml("[B.EXAMPLE.COM, a.example.com, b.example.com]"),
        );
        set(
            defaults,
            "profile_sets",
            yaml("[cpu-average, hardware-deep]"),
        );
        let RunSpec::V3(spec) = load_run_spec(&write_manifest(root.path(), &payload)).unwrap()
        else {
            panic!("expected v3")
        };
        assert_eq!(spec.input_root, portable);
        assert!(spec
            .tasks
            .iter()
            .all(|task| task.atif.parent() == Some(spec.input_root.as_path())));
        assert_eq!(
            spec.defaults.allowed_hosts,
            ["a.example.com", "b.example.com"]
        );
        assert_eq!(spec.defaults.profile_sets, ["cpu-average", "hardware-deep"]);
    }

    #[test]
    fn v3_enroot_accepts_supported_network_modes_and_anchors_images_to_manifest() {
        for network_policy in ["public", "no-network"] {
            let root = tempfile::tempdir().unwrap();
            let mut payload = base_v3(root.path());
            let inputs = root.path().join("portable-inputs");
            fs::create_dir(&inputs).unwrap();
            for name in ["one.json", "two.json"] {
                fs::copy(root.path().join("inputs").join(name), inputs.join(name)).unwrap();
            }
            set(
                &mut payload,
                "input_root",
                Value::String("portable-inputs".to_owned()),
            );
            let defaults = get_mut(&mut payload, "defaults");
            set(defaults, "runtime", Value::String("enroot".to_owned()));
            set(
                defaults,
                "network_policy",
                Value::String(network_policy.to_owned()),
            );
            set(defaults, "profile_sets", Value::Sequence(Vec::new()));
            let tasks = get_mut(&mut payload, "tasks").as_sequence_mut().unwrap();
            for (index, task) in tasks.iter_mut().enumerate() {
                let name = if index == 0 { "one" } else { "two" };
                set(task, "atif", Value::String(format!("{name}.json")));
                set(task, "image", Value::String(format!("images/{name}.sqsh")));
            }

            let RunSpec::V3(spec) = load_run_spec(&write_manifest(root.path(), &payload)).unwrap()
            else {
                panic!("expected v3")
            };
            assert_eq!(spec.defaults.runtime, "enroot");
            assert_eq!(spec.defaults.network_policy, network_policy);
            assert!(spec.defaults.profile_sets.is_empty());
            assert!(spec
                .tasks
                .iter()
                .all(|task| task.atif.parent() == Some(inputs.as_path())));
            assert_eq!(
                spec.tasks
                    .iter()
                    .map(|task| task.image.clone())
                    .collect::<Vec<_>>(),
                [
                    root.path()
                        .join("images/one.sqsh")
                        .to_string_lossy()
                        .into_owned(),
                    root.path()
                        .join("images/two.sqsh")
                        .to_string_lossy()
                        .into_owned(),
                ]
            );
        }
    }

    #[test]
    fn v3_enroot_rejects_allowlist_and_nonempty_profile_sets() {
        let root = tempfile::tempdir().unwrap();
        let mut payload = base_v3(root.path());
        let defaults = get_mut(&mut payload, "defaults");
        set(defaults, "runtime", Value::String("enroot".to_owned()));
        set(
            defaults,
            "network_policy",
            Value::String("allowlist".to_owned()),
        );
        set(defaults, "allowed_hosts", yaml("[example.com]"));
        assert!(error(root.path(), &payload)
            .contains("network_policy must be public or no-network for enroot runtime"));

        let mut payload = base_v3(root.path());
        set(
            get_mut(&mut payload, "defaults"),
            "runtime",
            Value::String("enroot".to_owned()),
        );
        assert!(
            error(root.path(), &payload).contains("profile_sets must be empty for enroot runtime")
        );
    }

    #[test]
    fn compact_defaults_reject_missing_pair_overlap_and_bad_placement() {
        let root = tempfile::tempdir().unwrap();
        let mut payload = compact_v3(root.path());
        remove(get_mut(&mut payload, "defaults"), "placement");
        assert!(error(root.path(), &payload)
            .contains("worker_cpus and defaults.placement must be specified together"));

        let mut payload = compact_v3(root.path());
        set(
            get_mut(&mut payload, "defaults"),
            "placement",
            Value::String("packed".to_owned()),
        );
        assert!(error(root.path(), &payload)
            .contains("defaults.placement must be 'round-robin-single-cpu'"));

        let mut payload = compact_v3(root.path());
        set(
            get_mut(&mut payload, "defaults"),
            "worker_cpus",
            Value::String("2-4".to_owned()),
        );
        assert!(error(root.path(), &payload)
            .contains("defaults.worker_cpus overlaps defaults.controller_cpus: 2,3"));

        let mut payload = compact_v3(root.path());
        set(
            get_mut(&mut payload, "defaults"),
            "worker_cpus",
            Value::String("worker-four".to_owned()),
        );
        assert!(error(root.path(), &payload).contains("invalid defaults.worker_cpus"));
    }

    #[test]
    fn compact_waves_reject_invalid_counts_capacity_and_mixed_forms() {
        let root = tempfile::tempdir().unwrap();
        for (field, invalid) in [
            ("allocation", yaml("true")),
            ("allocation", yaml("0")),
            ("allocation", yaml("1.5")),
            ("repeat", yaml("'2'")),
        ] {
            let mut payload = compact_v3(root.path());
            let wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
            set(wave, field, invalid);
            assert!(error(root.path(), &payload)
                .contains(&format!("{field} must be a positive integer")));
        }
        let mut payload = compact_v3(root.path());
        set(
            &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0],
            "allocation",
            yaml("4"),
        );
        assert!(error(root.path(), &payload)
            .contains("allocation (4) exceeds the number of defaults.worker_cpus (3)"));

        let mut payload = compact_v3(root.path());
        let mut explicit_payload = base_v3(root.path());
        let explicit_wave = get_mut(&mut explicit_payload, "waves")
            .as_sequence()
            .unwrap()[0]
            .clone();
        get_mut(&mut payload, "waves")
            .as_sequence_mut()
            .unwrap()
            .push(explicit_wave);
        assert!(error(root.path(), &payload)
            .contains("compact defaults cannot be combined with explicit waves"));

        let mut payload = compact_v3(root.path());
        set(
            &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0],
            "note",
            Value::String("nope".to_owned()),
        );
        assert!(error(root.path(), &payload).contains("waves[0] has unknown fields: note"));
    }

    #[test]
    fn v3_rejects_invalid_defaults() {
        let root = tempfile::tempdir().unwrap();
        let cases = [
            ("pacing", yaml("fast-ish"), "defaults.pacing must be one of"),
            (
                "runtime",
                yaml("podman"),
                "defaults.runtime must be one of: docker, enroot",
            ),
            (
                "network_policy",
                yaml("private"),
                "invalid defaults.network_policy",
            ),
            (
                "allowed_hosts",
                yaml("example.com"),
                "defaults.allowed_hosts must be a list",
            ),
            (
                "profile_sets",
                yaml("[]"),
                "defaults.profile_sets must be a non-empty list",
            ),
            (
                "profile_sets",
                yaml("[cpu-average, cpu-average]"),
                "must not contain duplicates",
            ),
            (
                "profile_sets",
                yaml("[hardware-deep]"),
                "must include cpu-average",
            ),
            (
                "controller_runtime",
                yaml("local"),
                "defaults.controller_runtime must be 'rust-host-mount' for native-only run execution",
            ),
            (
                "controller_cpus",
                yaml("three"),
                "invalid defaults.controller_cpus",
            ),
            (
                "action_timeout_sec",
                yaml("0"),
                "defaults.action_timeout_sec",
            ),
        ];
        for (field, value, expected) in cases {
            let mut payload = base_v3(root.path());
            set(get_mut(&mut payload, "defaults"), field, value);
            assert!(
                error(root.path(), &payload).contains(expected),
                "missing {expected:?} for {field}"
            );
        }
        let mut payload = base_v3(root.path());
        set(
            get_mut(&mut payload, "defaults"),
            "extra",
            Value::Bool(true),
        );
        assert!(error(root.path(), &payload).contains("defaults has unknown fields"));

        let mut payload = base_v3(root.path());
        set(
            get_mut(&mut payload, "defaults"),
            "allowed_hosts",
            yaml("[example.com]"),
        );
        assert!(error(root.path(), &payload).contains("allowed hosts require mode='allowlist'"));
    }

    #[test]
    fn v3_rejects_retired_python_controller_modes_with_a_migration_error() {
        for retired in RETIRED_CONTROLLER_RUNTIME_MODES {
            let root = tempfile::tempdir().unwrap();
            let mut payload = base_v3(root.path());
            set(
                get_mut(&mut payload, "defaults"),
                "controller_runtime",
                Value::String((*retired).to_owned()),
            );
            let message = error(root.path(), &payload);
            assert!(
                message.contains(&format!(
                    "defaults.controller_runtime '{retired}' is a retired Python-backed run controller mode"
                )),
                "{message}"
            );
            assert!(
                message.contains("native-only tasktrace run requires 'rust-host-mount'"),
                "{message}"
            );
            assert!(message.contains("migrate the manifest"), "{message}");
        }
    }

    #[test]
    fn v3_rejects_invalid_waves_and_unreferenced_tasks() {
        let root = tempfile::tempdir().unwrap();

        let mut payload = base_v3(root.path());
        let second = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[1];
        set(second, "id", Value::String("r1-n1".to_owned()));
        assert!(error(root.path(), &payload).contains("duplicate wave id"));

        let mut payload = base_v3(root.path());
        let first = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        set(first, "scored", yaml("1"));
        assert!(error(root.path(), &payload).contains("scored must be a boolean"));

        let mut payload = base_v3(root.path());
        let first = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        set(first, "metadata", yaml("{dimensions: {repeat: 1}}"));
        assert!(error(root.path(), &payload).contains("metadata.dimensions must be"));

        let mut payload = base_v3(root.path());
        let first = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        set(first, "metadata", yaml("{'bad key': 1}"));
        assert!(error(root.path(), &payload).contains("metadata key must be a safe"));

        let mut payload = base_v3(root.path());
        let first = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        set(first, "extra", Value::Bool(true));
        assert!(error(root.path(), &payload).contains("waves[0] has unknown fields"));

        let mut payload = base_v3(root.path());
        let first_wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        let launch = &mut get_mut(first_wave, "launches").as_sequence_mut().unwrap()[0];
        set(launch, "extra", Value::Bool(true));
        assert!(error(root.path(), &payload).contains("launches[0] has unknown fields"));

        let mut payload = base_v3(root.path());
        let first_wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        let launch = &mut get_mut(first_wave, "launches").as_sequence_mut().unwrap()[0];
        set(launch, "task", Value::String("missing".to_owned()));
        assert!(error(root.path(), &payload).contains("references unknown task"));

        let mut payload = base_v3(root.path());
        let first_wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        let launches = get_mut(first_wave, "launches").as_sequence_mut().unwrap();
        set(&mut launches[1], "task", Value::String("one".to_owned()));
        assert!(error(root.path(), &payload).contains("launches task more than once"));

        for (cpus, expected) in [
            ("2", "overlaps defaults.controller_cpus"),
            ("cpu-four", "invalid waves[0].launches[0].cpus"),
        ] {
            let mut payload = base_v3(root.path());
            let first_wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
            let launch = &mut get_mut(first_wave, "launches").as_sequence_mut().unwrap()[0];
            set(launch, "cpus", Value::String(cpus.to_owned()));
            assert!(error(root.path(), &payload).contains(expected));
        }

        let mut payload = base_v3(root.path());
        set(&mut payload, "waves", Value::Sequence(Vec::new()));
        assert!(error(root.path(), &payload).contains("waves must be a non-empty list"));

        let mut payload = base_v3(root.path());
        let first_wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
        set(first_wave, "launches", Value::Sequence(Vec::new()));
        assert!(error(root.path(), &payload).contains("launches must be a non-empty list"));

        let mut payload = base_v3(root.path());
        for wave in get_mut(&mut payload, "waves").as_sequence_mut().unwrap() {
            get_mut(wave, "launches")
                .as_sequence_mut()
                .unwrap()
                .retain(|launch| {
                    map_get(map(launch), "task").and_then(Value::as_str) != Some("two")
                });
        }
        assert!(error(root.path(), &payload).contains("unreferenced tasks: two"));
    }

    #[test]
    fn root_task_cpuset_and_timeout_validation_is_strict() {
        let root = tempfile::tempdir().unwrap();
        for field in ["runtime", "concurrency", "strategy", "name", "measure"] {
            let mut payload = base_v1(root.path());
            set(&mut payload, field, Value::String("nope".to_owned()));
            assert!(error(root.path(), &payload).contains("run has unknown fields"));
        }
        for field in ["schema_version", "output", "cpus", "tasks"] {
            let mut payload = base_v1(root.path());
            remove(&mut payload, field);
            assert!(error(root.path(), &payload).contains("run is missing"));
        }
        let mut payload = base_v1(root.path());
        set(
            &mut payload,
            "schema_version",
            Value::String("other".to_owned()),
        );
        assert!(error(root.path(), &payload).contains("unsupported run schema"));
        for non_text in [Value::Sequence(Vec::new()), Value::Mapping(Mapping::new())] {
            let mut payload = base_v1(root.path());
            set(&mut payload, "schema_version", non_text);
            assert!(error(root.path(), &payload)
                .contains("schema_version must be a non-empty trimmed string"));
        }

        for invalid in ["", "4-2", "1,,2", "one"] {
            let mut payload = base_v1(root.path());
            set(&mut payload, "cpus", Value::String(invalid.to_owned()));
            assert!(error(root.path(), &payload).contains("cpus"));
        }
        for invalid in [
            yaml("true"),
            yaml("0"),
            yaml("-1"),
            yaml("1.5"),
            yaml("'3'"),
        ] {
            let mut payload = base_v1(root.path());
            set(&mut payload, "repetitions", invalid);
            assert!(error(root.path(), &payload).contains("positive integer"));
        }

        let mut payload = base_v1(root.path());
        let tasks = get_mut(&mut payload, "tasks").as_sequence_mut().unwrap();
        set(&mut tasks[1], "id", Value::String("one".to_owned()));
        assert!(error(root.path(), &payload).contains("duplicate task id"));

        let mut payload = base_v1(root.path());
        let task = &mut get_mut(&mut payload, "tasks").as_sequence_mut().unwrap()[0];
        set(task, "id", Value::String("../one".to_owned()));
        assert!(error(root.path(), &payload).contains("must be a safe"));

        let mut payload = base_v1(root.path());
        let task = &mut get_mut(&mut payload, "tasks").as_sequence_mut().unwrap()[0];
        set(task, "format", Value::String("atif".to_owned()));
        assert!(error(root.path(), &payload).contains("tasks[0] has unknown fields"));

        let mut payload = base_v1(root.path());
        let task = &mut get_mut(&mut payload, "tasks").as_sequence_mut().unwrap()[0];
        set(task, "atif", Value::String("missing".to_owned()));
        assert!(error(root.path(), &payload).contains("is not a file"));

        let mut payload = base_v1(root.path());
        let task = &mut get_mut(&mut payload, "tasks").as_sequence_mut().unwrap()[0];
        set(
            task,
            "expected_source_sha256",
            Value::String("0".repeat(64)),
        );
        assert!(error(root.path(), &payload).contains("SHA-256 mismatch"));

        for invalid in [
            yaml("true"),
            yaml("0"),
            yaml("-1"),
            yaml(".inf"),
            yaml("'10'"),
        ] {
            let mut payload = base_v1(root.path());
            let task = &mut get_mut(&mut payload, "tasks").as_sequence_mut().unwrap()[0];
            set(task, "action_timeout_sec", invalid);
            assert!(error(root.path(), &payload).contains("action_timeout_sec"));
        }
    }

    #[test]
    fn duplicate_yaml_keys_are_rejected_at_every_mapping_depth() {
        let root = tempfile::tempdir().unwrap();
        fs::write(root.path().join("one.json"), b"{}\n").unwrap();
        for (source, expected) in [
            (
                r#"
schema_version: tasktrace-run.v1
output: results
cpus: 20-23
cpus: 24-27
tasks: []
"#,
                "found duplicate key 'cpus'",
            ),
            (
                r#"
schema_version: tasktrace-run.v1
output: results
cpus: 20-23
tasks:
  - id: one
    atif: one.json
    image: first
    image: second
"#,
                "found duplicate key 'image'",
            ),
        ] {
            let manifest = root.path().join("duplicate.yaml");
            fs::write(&manifest, source).unwrap();
            let error = load_run_spec(&manifest).unwrap_err().to_string();
            assert!(error.contains(expected), "{error}");
        }
    }

    #[test]
    fn metadata_rejects_non_scalar_nonfinite_and_unsafe_keys() {
        let root = tempfile::tempdir().unwrap();
        for (metadata, expected) in [
            ("{nested: {value: 1}}", "metadata.nested must be"),
            ("{'bad key': 1}", "metadata key must be a safe"),
            ("{infinite: .inf}", "metadata.infinite must be"),
            (
                "{empty: ''}",
                "metadata.empty must be a non-empty trimmed string",
            ),
        ] {
            let mut payload = base_v3(root.path());
            let wave = &mut get_mut(&mut payload, "waves").as_sequence_mut().unwrap()[0];
            set(wave, "metadata", yaml(metadata));
            assert!(error(root.path(), &payload).contains(expected));
        }
    }

    #[test]
    fn missing_manifest_error_contains_resolved_path() {
        let root = tempfile::tempdir().unwrap();
        let missing = root.path().join("missing/../run.yaml");
        let error = load_run_spec(&missing).unwrap_err().to_string();
        assert!(error.contains("unable to read run file"));
        assert!(error.contains(&root.path().join("run.yaml").display().to_string()));
    }

    #[test]
    fn strict_yaml_reader_accepts_comments_and_preserves_metadata_order() {
        let root = tempfile::tempdir().unwrap();
        let mut payload = base_v3(root.path());
        let manifest = write_manifest(root.path(), &payload);
        let mut file = fs::OpenOptions::new().append(true).open(&manifest).unwrap();
        writeln!(file, "# trailing comment").unwrap();
        let RunSpec::V3(spec) = load_run_spec(&manifest).unwrap() else {
            panic!("expected v3")
        };
        assert_eq!(
            spec.waves[0]
                .metadata
                .iter()
                .map(|(key, _)| key.as_str())
                .collect::<Vec<_>>(),
            ["repeat", "allocation"]
        );
        // Keep this mutable binding exercised so clippy does not suggest
        // replacing the fixture setup with an unrelated static string.
        set(
            &mut payload,
            "output",
            Value::String("elsewhere".to_owned()),
        );
    }
}
