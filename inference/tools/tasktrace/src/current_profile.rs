// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Read-only profiling inside the controller's current cgroup-v2 container.
//!
//! This backend cannot create or enforce an allocation.  It observes the
//! scheduler-created cgroup, validates that allocation and process scope stay
//! stable, and reports task-scoped CPU only when the caller explicitly asserts
//! a fresh exclusive container.

use crate::timebase::monotonic_ns;
use serde::Serialize;
use serde_json::{json, Map, Number, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::env;
use std::error::Error;
use std::ffi::{CString, OsStr, OsString};
use std::fmt;
use std::fs;
use std::io::{self, Read};
use std::mem;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const PROC_SELF_CGROUP: &str = "/proc/self/cgroup";
const PROC_SELF_MOUNTINFO: &str = "/proc/self/mountinfo";
const SCOPE_OBSERVATION_INTERVAL: Duration = Duration::from_millis(10);
const PROCESS_GROUP_CLEANUP_TIMEOUT: Duration = Duration::from_secs(2);
const KUBERNETES_IDENTITY_ENV: &[&str] = &[
    "HOSTNAME",
    "KUBERNETES_SERVICE_HOST",
    "KUBERNETES_SERVICE_PORT",
    "POD_NAME",
    "POD_NAMESPACE",
    "POD_UID",
    "CONTAINER_NAME",
    "NODE_NAME",
];

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct CurrentProfileError(String);

impl CurrentProfileError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for CurrentProfileError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for CurrentProfileError {}

impl From<io::Error> for CurrentProfileError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for CurrentProfileError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct CgroupV2Location {
    pub(crate) mount_point: PathBuf,
    pub(crate) mount_root: String,
    pub(crate) membership_path: String,
    pub(crate) path: PathBuf,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub(crate) struct AllocationState {
    pub(crate) cpu_max: Option<String>,
    pub(crate) cpu_max_hierarchy: Vec<Value>,
    pub(crate) quota_cores: Option<f64>,
    pub(crate) cpuset: Option<String>,
    pub(crate) cpuset_ids: Vec<u32>,
    pub(crate) affinity_ids: Vec<u32>,
}

#[derive(Clone, Debug)]
pub(crate) struct CurrentProfileConfig {
    pub(crate) command: String,
    pub(crate) prepare_command: Option<String>,
    pub(crate) output_dir: PathBuf,
    pub(crate) workdir: PathBuf,
    pub(crate) shell: String,
    pub(crate) environment: Vec<String>,
    pub(crate) unset_environment: Vec<String>,
    pub(crate) strace_mode: String,
    pub(crate) string_limit: u64,
    pub(crate) allocated_cpus: Option<f64>,
    pub(crate) exclusive: bool,
    /// Make the measured command a child subreaper so helpers its actions orphan stay in its process
    /// tree. Trace replays only: ATIF replays keep long-lived session children.
    pub(crate) adopt_orphans: bool,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SanitizedTaskEnvironment {
    pub(crate) assignments: Vec<String>,
    pub(crate) unset_names: Vec<String>,
}

fn normalized_environment_path(value: &str, cwd: &Path) -> PathBuf {
    let path = if value.is_empty() {
        Path::new(".")
    } else {
        Path::new(value)
    };
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        cwd.join(path)
    };
    fs::canonicalize(&path).unwrap_or(path)
}

fn path_is_within(path: &Path, root: &Path) -> bool {
    path == root || path.starts_with(root)
}

fn sanitized_controller_environment_from(
    values: &BTreeMap<String, String>,
    controller_executable: &Path,
    controller_source: Option<&Path>,
    invocation_cwd: &Path,
) -> Result<SanitizedTaskEnvironment, CurrentProfileError> {
    let executable = fs::canonicalize(controller_executable)
        .unwrap_or_else(|_| controller_executable.to_path_buf());
    let source =
        controller_source.map(|path| fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf()));
    let controller_virtualenv = values.get("VIRTUAL_ENV").and_then(|value| {
        let candidate = normalized_environment_path(value, invocation_cwd);
        path_is_within(&executable, &candidate).then_some(candidate)
    });
    let mut assignments = Vec::new();
    let mut unset = BTreeSet::new();

    if let (Some(path), Some(virtualenv)) = (values.get("PATH"), &controller_virtualenv) {
        let mut kept = Vec::new();
        let mut changed = false;
        for entry in path.split(':') {
            let resolved = normalized_environment_path(entry, invocation_cwd);
            if path_is_within(&resolved, virtualenv) {
                changed = true;
            } else {
                kept.push(entry);
            }
        }
        if changed {
            if kept.is_empty() {
                return Err(CurrentProfileError::new(
                    "current-container profiling cannot derive a task PATH after removing the controller virtualenv",
                ));
            }
            assignments.push(format!("PATH={}", kept.join(":")));
        }
    }

    if let Some(pythonpath) = values.get("PYTHONPATH") {
        let mut kept = Vec::new();
        let mut changed = false;
        for entry in pythonpath.split(':') {
            let resolved = normalized_environment_path(entry, invocation_cwd);
            let from_source = source
                .as_ref()
                .is_some_and(|root| path_is_within(&resolved, root));
            let from_virtualenv = controller_virtualenv
                .as_ref()
                .is_some_and(|root| path_is_within(&resolved, root));
            if from_source || from_virtualenv {
                changed = true;
            } else {
                kept.push(entry);
            }
        }
        if changed {
            unset.insert("PYTHONPATH".to_owned());
            if !kept.is_empty() {
                assignments.push(format!("PYTHONPATH={}", kept.join(":")));
            }
        }
    }
    if values.contains_key("UV_RUN_RECURSION_DEPTH") {
        unset.insert("UV_RUN_RECURSION_DEPTH".to_owned());
    }
    if controller_virtualenv.is_some() {
        unset.insert("VIRTUAL_ENV".to_owned());
        if values.contains_key("VIRTUAL_ENV_PROMPT") {
            unset.insert("VIRTUAL_ENV_PROMPT".to_owned());
        }
    }
    Ok(SanitizedTaskEnvironment {
        assignments,
        unset_names: unset.into_iter().collect(),
    })
}

pub(crate) fn sanitized_controller_environment(
    controller_executable: &Path,
    controller_source: Option<&Path>,
    invocation_cwd: &Path,
) -> Result<SanitizedTaskEnvironment, CurrentProfileError> {
    // Only inspect variables that can carry controller state.  `env::vars()`
    // panics when *any unrelated* variable contains non-UTF-8 bytes, which is
    // both unnecessary and an avoidable availability failure for this
    // fail-closed isolation check.  A non-UTF-8 value in one of the variables
    // we must reason about is still rejected explicitly.
    let mut values = BTreeMap::new();
    for name in [
        "PATH",
        "PYTHONPATH",
        "VIRTUAL_ENV",
        "VIRTUAL_ENV_PROMPT",
        "UV_RUN_RECURSION_DEPTH",
    ] {
        if let Some(value) = env::var_os(name) {
            let value = value.into_string().map_err(|_| {
                CurrentProfileError::new(format!(
                    "current-container profiling cannot safely inspect non-UTF-8 {name}"
                ))
            })?;
            values.insert(name.to_owned(), value);
        }
    }
    sanitized_controller_environment_from(
        &values,
        controller_executable,
        controller_source,
        invocation_cwd,
    )
}

pub(crate) fn discover_controller_source(executable: &Path) -> Option<PathBuf> {
    let executable = fs::canonicalize(executable).ok()?;
    executable.ancestors().find_map(|ancestor| {
        (ancestor.join("Cargo.toml").is_file() && ancestor.join("src/main.rs").is_file())
            .then(|| ancestor.to_path_buf())
    })
}

fn decode_mountinfo_path(mut value: String) -> String {
    for (escaped, decoded) in [
        ("\\040", " "),
        ("\\011", "\t"),
        ("\\012", "\n"),
        ("\\134", "\\"),
    ] {
        value = value.replace(escaped, decoded);
    }
    value
}

fn unified_membership(path: &Path) -> Result<String, CurrentProfileError> {
    let text = fs::read_to_string(path).map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to read cgroup membership from {}: {error}",
            path.display()
        ))
    })?;
    for line in text.lines() {
        if let Some(membership) = line.strip_prefix("0::") {
            return Ok(if membership.starts_with('/') {
                membership.to_owned()
            } else {
                format!("/{membership}")
            });
        }
    }
    Err(CurrentProfileError::new(format!(
        "cgroup v2 membership not found in {}",
        path.display()
    )))
}

fn cgroup2_mounts(path: &Path) -> Result<Vec<(String, PathBuf)>, CurrentProfileError> {
    let text = fs::read_to_string(path).map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to read mount information from {}: {error}",
            path.display()
        ))
    })?;
    let mut mounts = Vec::new();
    for line in text.lines() {
        let Some((before, after)) = line.split_once(" - ") else {
            continue;
        };
        let before = before.split_whitespace().collect::<Vec<_>>();
        let after = after.split_whitespace().collect::<Vec<_>>();
        if after.first() != Some(&"cgroup2") || before.len() < 5 {
            continue;
        }
        mounts.push((
            decode_mountinfo_path(before[3].to_owned()),
            PathBuf::from(decode_mountinfo_path(before[4].to_owned())),
        ));
    }
    if mounts.is_empty() {
        return Err(CurrentProfileError::new(format!(
            "cgroup v2 mount not found in {}",
            path.display()
        )));
    }
    Ok(mounts)
}

fn safe_relative(value: &str) -> Option<PathBuf> {
    let path = Path::new(value.trim_start_matches('/'));
    if path
        .components()
        .any(|component| !matches!(component, std::path::Component::Normal(_)))
        && !path.as_os_str().is_empty()
    {
        return None;
    }
    Some(path.to_path_buf())
}

fn mount_candidates(mount_root: &str, mount_point: &Path, membership: &str) -> Vec<PathBuf> {
    let mut relative = Vec::new();
    let root = Path::new(mount_root);
    let member = Path::new(membership);
    if let Ok(value) = member.strip_prefix(root) {
        if let Some(value) = safe_relative(&value.to_string_lossy()) {
            relative.push(value);
        }
    }
    if let Some(value) = safe_relative(membership) {
        if !relative.contains(&value) {
            relative.push(value);
        }
    }
    relative
        .into_iter()
        .map(|value| mount_point.join(value))
        .collect()
}

pub(crate) fn resolve_current_cgroup_from(
    proc_self_cgroup: &Path,
    proc_self_mountinfo: &Path,
) -> Result<CgroupV2Location, CurrentProfileError> {
    let membership = unified_membership(proc_self_cgroup)?;
    let mut attempted = Vec::new();
    for (mount_root, mount_point) in cgroup2_mounts(proc_self_mountinfo)? {
        for candidate in mount_candidates(&mount_root, &mount_point, &membership) {
            attempted.push(candidate.clone());
            if candidate.join("cpu.stat").is_file() && candidate.join("cgroup.procs").is_file() {
                return Ok(CgroupV2Location {
                    mount_point,
                    mount_root,
                    membership_path: membership,
                    path: candidate,
                });
            }
        }
    }
    Err(CurrentProfileError::new(format!(
        "current-container profiling requires readable cgroup v2 cpu.stat and cgroup.procs; checked: {}",
        attempted
            .iter()
            .map(|path| path.display().to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )))
}

pub(crate) fn resolve_current_cgroup() -> Result<CgroupV2Location, CurrentProfileError> {
    resolve_current_cgroup_from(Path::new(PROC_SELF_CGROUP), Path::new(PROC_SELF_MOUNTINFO))
}

pub(crate) fn read_cpu_stat(
    path: &Path,
) -> Result<(String, BTreeMap<String, u64>), CurrentProfileError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to read cgroup CPU statistics from {}: {error}",
            path.display()
        ))
    })?;
    let mut stats = BTreeMap::new();
    for line in raw.lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() != 2 {
            continue;
        }
        let value = fields[1].parse::<u64>().map_err(|_| {
            CurrentProfileError::new(format!(
                "invalid cgroup CPU statistic in {}: {line:?}",
                path.display()
            ))
        })?;
        stats.insert(fields[0].to_owned(), value);
    }
    if !stats.contains_key("usage_usec") {
        return Err(CurrentProfileError::new(format!(
            "cgroup CPU statistics have no usage_usec counter: {}",
            path.display()
        )));
    }
    Ok((raw, stats))
}

fn read_timed_cpu_stat_with(
    path: &Path,
    mut clock: impl FnMut() -> u64,
) -> Result<(String, BTreeMap<String, u64>, u64, u64), CurrentProfileError> {
    let started = clock();
    let (raw, stats) = read_cpu_stat(path)?;
    let finished = clock();
    let duration = finished.checked_sub(started).ok_or_else(|| {
        CurrentProfileError::new(
            "monotonic clock moved backward while reading cgroup CPU statistics",
        )
    })?;
    Ok((raw, stats, started + duration / 2, duration))
}

fn read_timed_cpu_stat(
    path: &Path,
) -> Result<(String, BTreeMap<String, u64>, u64, u64), CurrentProfileError> {
    read_timed_cpu_stat_with(path, monotonic_ns)
}

pub(crate) fn parse_cpu_max(value: Option<&str>) -> Result<Option<f64>, CurrentProfileError> {
    let Some(value) = value else {
        return Ok(None);
    };
    let fields = value.split_whitespace().collect::<Vec<_>>();
    if fields.len() != 2 {
        return Err(CurrentProfileError::new(format!(
            "invalid cgroup cpu.max value: {value:?}"
        )));
    }
    let period = fields[1].parse::<u64>().map_err(|_| {
        CurrentProfileError::new(format!("invalid cgroup cpu.max period: {value:?}"))
    })?;
    if period == 0 {
        return Err(CurrentProfileError::new(format!(
            "invalid cgroup cpu.max period: {value:?}"
        )));
    }
    if fields[0] == "max" {
        return Ok(None);
    }
    let quota = fields[0].parse::<u64>().map_err(|_| {
        CurrentProfileError::new(format!("invalid cgroup cpu.max quota: {value:?}"))
    })?;
    if quota == 0 {
        return Err(CurrentProfileError::new(format!(
            "invalid cgroup cpu.max quota: {value:?}"
        )));
    }
    Ok(Some(quota as f64 / period as f64))
}

fn read_optional(path: &Path) -> Result<Option<String>, CurrentProfileError> {
    match fs::read_to_string(path) {
        Ok(value) => Ok((!value.trim().is_empty()).then(|| value.trim().to_owned())),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(None),
        Err(error) => Err(CurrentProfileError::new(format!(
            "unable to read cgroup allocation metadata from {}: {error}",
            path.display()
        ))),
    }
}

fn ancestor_paths(location: &CgroupV2Location) -> Result<Vec<PathBuf>, CurrentProfileError> {
    if !location.path.starts_with(&location.mount_point) {
        return Err(CurrentProfileError::new(
            "resolved current cgroup lies outside its cgroup-v2 mount",
        ));
    }
    let mut paths = Vec::new();
    let mut current = location.path.clone();
    loop {
        paths.push(current.clone());
        if current == location.mount_point {
            break;
        }
        current = current
            .parent()
            .ok_or_else(|| {
                CurrentProfileError::new("cgroup ancestor chain ended before its mount root")
            })?
            .to_path_buf();
    }
    Ok(paths)
}

fn read_inherited(
    location: &CgroupV2Location,
    filename: &str,
) -> Result<Option<String>, CurrentProfileError> {
    for path in ancestor_paths(location)? {
        if let Some(value) = read_optional(&path.join(filename))? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn parse_cpuset(value: &str) -> Result<Vec<u32>, CurrentProfileError> {
    let mut ids = BTreeSet::new();
    for segment in value.split(',') {
        let segment = segment.trim();
        if segment.is_empty() {
            return Err(CurrentProfileError::new(format!(
                "invalid cgroup cpuset value: {value:?}"
            )));
        }
        if let Some((start, end)) = segment.split_once('-') {
            let start = start.parse::<u32>().map_err(|_| {
                CurrentProfileError::new(format!("invalid cgroup cpuset value: {value:?}"))
            })?;
            let end = end.parse::<u32>().map_err(|_| {
                CurrentProfileError::new(format!("invalid cgroup cpuset value: {value:?}"))
            })?;
            if end < start {
                return Err(CurrentProfileError::new(format!(
                    "invalid cgroup cpuset value: {value:?}"
                )));
            }
            ids.extend(start..=end);
        } else {
            ids.insert(segment.parse::<u32>().map_err(|_| {
                CurrentProfileError::new(format!("invalid cgroup cpuset value: {value:?}"))
            })?);
        }
    }
    Ok(ids.into_iter().collect())
}

fn process_affinity() -> Result<Vec<u32>, CurrentProfileError> {
    let mut set = unsafe { mem::zeroed::<libc::cpu_set_t>() };
    let result = unsafe {
        libc::sched_getaffinity(
            0,
            mem::size_of::<libc::cpu_set_t>(),
            &mut set as *mut libc::cpu_set_t,
        )
    };
    if result != 0 {
        return Err(CurrentProfileError::new(format!(
            "unable to read current process CPU affinity: {}",
            io::Error::last_os_error()
        )));
    }
    let mut ids = Vec::new();
    for cpu in 0..libc::CPU_SETSIZE as usize {
        if unsafe { libc::CPU_ISSET(cpu, &set) } {
            ids.push(cpu as u32);
        }
    }
    if ids.is_empty() {
        return Err(CurrentProfileError::new(
            "current process CPU affinity is empty",
        ));
    }
    Ok(ids)
}

fn read_allocation_state_with_affinity(
    location: &CgroupV2Location,
    affinity_ids: Vec<u32>,
) -> Result<AllocationState, CurrentProfileError> {
    if affinity_ids.is_empty() {
        return Err(CurrentProfileError::new(
            "current process CPU affinity is empty",
        ));
    }
    let mut hierarchy = Vec::new();
    let mut finite_quotas = Vec::new();
    for path in ancestor_paths(location)? {
        if let Some(raw) = read_optional(&path.join("cpu.max"))? {
            let quota = parse_cpu_max(Some(&raw))?;
            if let Some(quota) = quota {
                finite_quotas.push(quota);
            }
            hierarchy.push(json!({
                "path": path.join("cpu.max"),
                "value": raw,
                "quota_cores": quota,
            }));
        }
    }
    let cpu_max = hierarchy
        .first()
        .and_then(|row| row.get("value"))
        .and_then(Value::as_str)
        .map(ToOwned::to_owned);
    let cpuset = read_inherited(location, "cpuset.cpus.effective")?;
    let cpuset_ids = cpuset
        .as_deref()
        .map(parse_cpuset)
        .transpose()?
        .unwrap_or_default();
    Ok(AllocationState {
        cpu_max,
        cpu_max_hierarchy: hierarchy,
        quota_cores: finite_quotas.into_iter().reduce(f64::min),
        cpuset,
        cpuset_ids,
        affinity_ids,
    })
}

pub(crate) fn read_allocation_state(
    location: &CgroupV2Location,
) -> Result<AllocationState, CurrentProfileError> {
    read_allocation_state_with_affinity(location, process_affinity()?)
}

fn capacity(
    allocation: &AllocationState,
    override_capacity: Option<f64>,
) -> (Option<f64>, Option<String>, Vec<u32>) {
    let cpuset = allocation
        .cpuset_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let affinity = allocation
        .affinity_ids
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let (effective, cpu_source) = if !cpuset.is_empty() && !affinity.is_empty() {
        (
            cpuset.intersection(&affinity).copied().collect::<Vec<_>>(),
            Some("cpuset+affinity"),
        )
    } else if !cpuset.is_empty() {
        (cpuset.into_iter().collect(), Some("cpuset"))
    } else if !affinity.is_empty() {
        (affinity.into_iter().collect(), Some("affinity"))
    } else {
        (Vec::new(), None)
    };
    if let Some(value) = override_capacity {
        return (Some(value), Some("user-supplied".to_owned()), effective);
    }
    let effective_count = (!effective.is_empty()).then_some(effective.len() as f64);
    match (allocation.quota_cores, effective_count) {
        (Some(quota), Some(count)) => (
            Some(quota.min(count)),
            Some(format!(
                "cgroup-v2-cpu.max+{}",
                cpu_source.expect("effective IDs have source")
            )),
            effective,
        ),
        (Some(quota), None) => (Some(quota), Some("cgroup-v2-cpu.max".to_owned()), effective),
        (None, Some(count)) => (Some(count), cpu_source.map(ToOwned::to_owned), effective),
        (None, None) => (None, None, effective),
    }
}

fn read_cgroup_pids(path: &Path, scope: &str) -> Result<BTreeSet<u32>, CurrentProfileError> {
    let text = fs::read_to_string(path).map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to inspect processes in {scope} from {}: {error}",
            path.display()
        ))
    })?;
    let mut pids = BTreeSet::new();
    for value in text.split_whitespace() {
        let pid = value.parse::<u32>().map_err(|_| {
            CurrentProfileError::new(format!(
                "invalid process ID in {}: {value:?}",
                path.display()
            ))
        })?;
        if pid == 0 {
            return Err(CurrentProfileError::new(format!(
                "invalid process ID in {}: {value:?}",
                path.display()
            )));
        }
        pids.insert(pid);
    }
    Ok(pids)
}

fn visit_descendant_cgroups(
    directory: &Path,
    root: &Path,
    pids: &mut BTreeSet<u32>,
) -> Result<(), CurrentProfileError> {
    let entries = fs::read_dir(directory).map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to enumerate descendant cgroups under {}: {error}",
            root.display()
        ))
    })?;
    for entry in entries {
        let entry = entry.map_err(|error| {
            CurrentProfileError::new(format!(
                "unable to enumerate descendant cgroups under {}: {error}",
                root.display()
            ))
        })?;
        let kind = entry.file_type().map_err(|error| {
            CurrentProfileError::new(format!(
                "unable to enumerate descendant cgroups under {}: {error}",
                root.display()
            ))
        })?;
        if !kind.is_dir() || kind.is_symlink() {
            continue;
        }
        let child = entry.path();
        let process_file = child.join("cgroup.procs");
        if process_file.is_file() {
            pids.extend(read_cgroup_pids(&process_file, "descendant cgroup")?);
        }
        visit_descendant_cgroups(&child, root, pids)?;
    }
    Ok(())
}

fn cgroup_pids(path: &Path) -> Result<Vec<u32>, CurrentProfileError> {
    let mut pids = read_cgroup_pids(&path.join("cgroup.procs"), "current cgroup")?;
    visit_descendant_cgroups(path, path, &mut pids)?;
    Ok(pids.into_iter().collect())
}

fn cgroup_filesystem_read_only(path: &Path) -> Result<bool, CurrentProfileError> {
    let path = CString::new(path.as_os_str().as_bytes()).map_err(|_| {
        CurrentProfileError::new(format!(
            "cgroup-v2 mount path contains a NUL byte: {}",
            path.display()
        ))
    })?;
    let mut state = unsafe { mem::zeroed::<libc::statvfs>() };
    if unsafe { libc::statvfs(path.as_ptr(), &mut state) } != 0 {
        return Err(CurrentProfileError::new(format!(
            "unable to inspect cgroup-v2 mount flags: {}",
            io::Error::last_os_error()
        )));
    }
    Ok(state.f_flag & libc::ST_RDONLY != 0)
}

fn visible_cgroup_filesystems_read_only(
    proc_self_mountinfo: &Path,
) -> Result<bool, CurrentProfileError> {
    for (_, mount_point) in cgroup2_mounts(proc_self_mountinfo)? {
        if !cgroup_filesystem_read_only(&mount_point)? {
            return Ok(false);
        }
    }
    Ok(true)
}

fn parent_pid(pid: u32) -> Option<u32> {
    let stat = fs::read_to_string(format!("/proc/{pid}/stat")).ok()?;
    let closing = stat.rfind(')')?;
    let fields = stat
        .get(closing + 2..)?
        .split_whitespace()
        .collect::<Vec<_>>();
    fields.get(1)?.parse().ok()
}

fn ancestor_pids(pid: u32) -> BTreeSet<u32> {
    let mut ancestors = BTreeSet::new();
    let mut current = pid;
    while current > 0 && ancestors.insert(current) {
        let Some(parent) = parent_pid(current) else {
            break;
        };
        current = parent;
    }
    ancestors
}

fn process_details(pids: &[u32]) -> Vec<Value> {
    pids.iter()
        .map(|pid| {
            let name = fs::read_to_string(format!("/proc/{pid}/comm"))
                .ok()
                .map(|value| value.trim().to_owned());
            json!({"pid": pid, "name": name})
        })
        .collect()
}

fn unrelated_processes(pids: &[u32]) -> Vec<u32> {
    let ancestors = ancestor_pids(std::process::id());
    pids.iter()
        .copied()
        .filter(|pid| !ancestors.contains(pid))
        .collect()
}

/// How a process sampled from the container cgroup relates to the measured command.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Ancestry {
    Descendant,
    Unrelated,
    /// Exited before its parent chain could be read.
    Vanished,
}

fn process_exists(pid: u32) -> bool {
    Path::new(&format!("/proc/{pid}")).exists()
}

/// Walks of a chain broken by an exiting ancestor before the process is left unattributed.
const ANCESTRY_WALK_ATTEMPTS: usize = 3;

/// Relate `pid` to the measured command `root` by its parent chain. A pid gone mid-walk is
/// `Vanished`, not `Unrelated`: replays fork many helpers that exit within milliseconds.
fn classify_ancestry(pid: u32, root: Option<u32>) -> (Ancestry, Vec<u32>) {
    classify_ancestry_with(pid, root, parent_pid, process_exists)
}

/// `classify_ancestry` over injectable /proc lookups. Only a chain that ends at pid 0 (or loops)
/// proves a process is outside `root`; one broken by an exiting ancestor is walked again.
fn classify_ancestry_with(
    pid: u32,
    root: Option<u32>,
    parent_of: impl Fn(u32) -> Option<u32>,
    exists: impl Fn(u32) -> bool,
) -> (Ancestry, Vec<u32>) {
    for _ in 0..ANCESTRY_WALK_ATTEMPTS {
        let mut chain = Vec::new();
        let mut current = pid;
        let mut visited = BTreeSet::new();
        let mut broken = false;
        while current > 0 && visited.insert(current) {
            if Some(current) == root {
                return (Ancestry::Descendant, chain);
            }
            chain.push(current);
            match parent_of(current) {
                Some(parent) => current = parent,
                None => {
                    broken = true;
                    break;
                }
            }
        }
        if !exists(pid) {
            return (Ancestry::Vanished, chain);
        }
        if !broken {
            return (Ancestry::Unrelated, chain);
        }
    }
    (Ancestry::Vanished, Vec::new())
}

fn process_name(pid: u32) -> Option<String> {
    fs::read_to_string(format!("/proc/{pid}/comm"))
        .ok()
        .map(|value| value.trim().to_owned())
}

#[derive(Debug)]
struct ScopeObservation<'a> {
    expected_location: &'a CgroupV2Location,
    proc_self_cgroup: &'a Path,
    proc_self_mountinfo: &'a Path,
    controller_ancestry: BTreeSet<u32>,
    samples: u64,
    first_sample_ns: Option<u64>,
    last_sample_ns: Option<u64>,
    maximum_sample_gap_ns: u64,
    cgroup_path_stable: bool,
    command_membership_stable: bool,
    cgroup_filesystem_read_only: bool,
    observed_membership_paths: BTreeSet<String>,
    observed_unrelated: BTreeMap<u32, (Option<String>, Vec<u32>)>,
    observed_transient: BTreeSet<u32>,
    errors: Vec<String>,
}

impl<'a> ScopeObservation<'a> {
    fn new(
        expected_location: &'a CgroupV2Location,
        proc_self_cgroup: &'a Path,
        proc_self_mountinfo: &'a Path,
    ) -> Self {
        Self {
            expected_location,
            proc_self_cgroup,
            proc_self_mountinfo,
            controller_ancestry: ancestor_pids(std::process::id()),
            samples: 0,
            first_sample_ns: None,
            last_sample_ns: None,
            maximum_sample_gap_ns: 0,
            cgroup_path_stable: true,
            command_membership_stable: true,
            cgroup_filesystem_read_only: true,
            observed_membership_paths: BTreeSet::new(),
            observed_unrelated: BTreeMap::new(),
            observed_transient: BTreeSet::new(),
            errors: Vec::new(),
        }
    }

    fn record_error(&mut self, message: impl Into<String>) {
        let message = message.into();
        if !self.errors.contains(&message) {
            self.errors.push(message);
        }
    }

    fn sample(&mut self, command_root: Option<u32>) {
        let now = monotonic_ns();
        if let Some(previous) = self.last_sample_ns {
            let Some(gap) = now.checked_sub(previous) else {
                self.record_error("monotonic clock moved backward during cgroup scope observation");
                return;
            };
            self.maximum_sample_gap_ns = self.maximum_sample_gap_ns.max(gap);
        } else {
            self.first_sample_ns = Some(now);
        }
        self.last_sample_ns = Some(now);
        self.samples += 1;

        match resolve_current_cgroup_from(self.proc_self_cgroup, self.proc_self_mountinfo) {
            Ok(location) => {
                self.observed_membership_paths
                    .insert(location.membership_path.clone());
                if location != *self.expected_location {
                    self.cgroup_path_stable = false;
                }
            }
            Err(error) => {
                self.cgroup_path_stable = false;
                self.record_error(format!(
                    "unable to resolve the controller cgroup during measured-window observation: {error}"
                ));
            }
        }

        match visible_cgroup_filesystems_read_only(self.proc_self_mountinfo) {
            Ok(true) => {}
            Ok(false) => self.cgroup_filesystem_read_only = false,
            Err(error) => {
                self.cgroup_filesystem_read_only = false;
                self.record_error(error.to_string());
            }
        }

        if let Some(command_root) = command_root {
            let command_membership = PathBuf::from(format!("/proc/{command_root}/cgroup"));
            match unified_membership(&command_membership) {
                Ok(membership) => {
                    self.observed_membership_paths.insert(membership.clone());
                    if membership != self.expected_location.membership_path {
                        self.command_membership_stable = false;
                    }
                }
                Err(error) => {
                    self.command_membership_stable = false;
                    self.record_error(format!(
                        "unable to resolve the measured command cgroup during scope observation: {error}"
                    ));
                }
            }
        }

        match cgroup_pids(&self.expected_location.path) {
            Ok(pids) => {
                for pid in pids {
                    if self.controller_ancestry.contains(&pid) {
                        continue;
                    }
                    match classify_ancestry(pid, command_root) {
                        (Ancestry::Descendant, _) => {}
                        (Ancestry::Vanished, _) => {
                            if !self.observed_unrelated.contains_key(&pid) {
                                self.observed_transient.insert(pid);
                            }
                        }
                        (Ancestry::Unrelated, chain) => {
                            self.observed_transient.remove(&pid);
                            self.observed_unrelated
                                .entry(pid)
                                .or_insert_with(|| (process_name(pid), chain));
                        }
                    }
                }
            }
            Err(error) => self.record_error(format!(
                "unable to inspect processes during measured-window scope observation: {error}"
            )),
        }
    }

    fn covers_window(&self, start_ns: u64, end_ns: u64) -> bool {
        self.first_sample_ns
            .zip(self.last_sample_ns)
            .is_some_and(|(first, last)| first <= start_ns && last >= end_ns)
    }

    fn complete(&self, start_ns: u64, end_ns: u64) -> bool {
        self.errors.is_empty() && self.samples >= 2 && self.covers_window(start_ns, end_ns)
    }

    fn observed_unrelated_details(&self) -> Vec<Value> {
        self.observed_unrelated
            .iter()
            .map(|(pid, (name, chain))| json!({"pid": pid, "name": name, "parent_chain": chain}))
            .collect()
    }

    fn observed_transient_pids(&self) -> Vec<u32> {
        self.observed_transient.iter().copied().collect()
    }
}

fn hostname() -> Option<String> {
    let mut buffer = [0_u8; 256];
    let result =
        unsafe { libc::gethostname(buffer.as_mut_ptr().cast::<libc::c_char>(), buffer.len()) };
    if result != 0 {
        return None;
    }
    let length = buffer
        .iter()
        .position(|byte| *byte == 0)
        .unwrap_or(buffer.len());
    String::from_utf8(buffer[..length].to_vec()).ok()
}

fn os_release() -> Option<Value> {
    let text = fs::read_to_string("/etc/os-release").ok()?;
    let mut result = Map::new();
    for line in text.lines() {
        let Some((key, value)) = line.split_once('=') else {
            continue;
        };
        let output_key = match key {
            "ID" => "id",
            "VERSION_ID" => "version_id",
            "PRETTY_NAME" => "pretty_name",
            _ => continue,
        };
        result.insert(
            output_key.to_owned(),
            Value::String(value.trim().trim_matches('"').to_owned()),
        );
    }
    (!result.is_empty()).then_some(Value::Object(result))
}

fn environment_identity(location: &CgroupV2Location) -> Value {
    let kubernetes = KUBERNETES_IDENTITY_ENV
        .iter()
        .filter_map(|key| {
            env::var(key)
                .ok()
                .filter(|value| !value.is_empty())
                .map(|value| ((*key).to_owned(), Value::String(value)))
        })
        .collect::<Map<_, _>>();
    json!({
        "runtime": "current",
        "hostname": hostname(),
        "controller_executable": env::current_exe().ok(),
        "controller_language": "rust",
        "controller_version": env!("CARGO_PKG_VERSION"),
        "cgroup_v2": {
            "mount_point": location.mount_point,
            "mount_root": location.mount_root,
            "membership_path": location.membership_path,
            "resolved_path": location.path,
            "private_namespace_root": location.membership_path == "/",
        },
        "kubernetes": if kubernetes.is_empty() { Value::Null } else { Value::Object(kubernetes) },
        "os_release": os_release(),
    })
}

type CommandEnvironment = BTreeMap<OsString, OsString>;

fn command_environment(
    assignments: &[String],
    unset_names: &[String],
) -> Result<CommandEnvironment, CurrentProfileError> {
    let mut environment = env::vars_os().collect::<CommandEnvironment>();
    for name in unset_names {
        if name.is_empty() || name.contains('=') || name.contains('\0') {
            return Err(CurrentProfileError::new(format!(
                "invalid environment variable name to unset: {name:?}"
            )));
        }
        environment.remove(OsStr::new(name));
    }
    for assignment in assignments {
        let (key, value) = assignment.split_once('=').ok_or_else(|| {
            CurrentProfileError::new(format!(
                "environment assignment must use KEY=VALUE: {assignment:?}"
            ))
        })?;
        if key.is_empty() || key.contains('\0') || value.contains('\0') {
            return Err(CurrentProfileError::new(format!(
                "environment assignment has an invalid key or value: {assignment:?}"
            )));
        }
        environment.insert(OsString::from(key), OsString::from(value));
    }
    Ok(environment)
}

fn executable_in_environment(command: &str, environment: &CommandEnvironment) -> bool {
    let candidate = Path::new(command);
    if candidate.components().count() > 1 {
        return candidate.metadata().ok().is_some_and(|metadata| {
            metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
        });
    }
    let Some(path) = environment.get(OsStr::new("PATH")) else {
        return false;
    };
    env::split_paths(path).any(|directory| {
        directory
            .join(command)
            .metadata()
            .ok()
            .is_some_and(|metadata| {
                metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
            })
    })
}

fn validate_config(
    config: &CurrentProfileConfig,
    environment: &CommandEnvironment,
) -> Result<(), CurrentProfileError> {
    if config.command.is_empty() {
        return Err(CurrentProfileError::new(
            "current-container command is empty",
        ));
    }
    if !config.workdir.is_dir() {
        return Err(CurrentProfileError::new(format!(
            "current-container workdir does not exist: {}",
            config.workdir.display()
        )));
    }
    if !matches!(config.strace_mode.as_str(), "none" | "execve" | "process") {
        return Err(CurrentProfileError::new(format!(
            "unsupported strace mode: {:?}",
            config.strace_mode
        )));
    }
    if config.string_limit == 0 {
        return Err(CurrentProfileError::new(
            "strace string limit must be positive",
        ));
    }
    if config
        .allocated_cpus
        .is_some_and(|value| !value.is_finite() || value <= 0.0)
    {
        return Err(CurrentProfileError::new(
            "allocated CPU count must be positive",
        ));
    }
    if !executable_in_environment(&config.shell, environment) {
        return Err(CurrentProfileError::new(format!(
            "current-container shell is not executable: {}",
            config.shell
        )));
    }
    if config.strace_mode != "none" && !executable_in_environment("strace", environment) {
        return Err(CurrentProfileError::new(
            "strace is not installed in the current container",
        ));
    }
    Ok(())
}

fn prepare_output(path: &Path) -> Result<(), CurrentProfileError> {
    if path.exists() {
        if !path.is_dir() {
            return Err(CurrentProfileError::new(format!(
                "profile output path is not a directory: {}",
                path.display()
            )));
        }
        if fs::read_dir(path)?.next().transpose()?.is_some() {
            return Err(CurrentProfileError::new(format!(
                "profile output directory must be new or empty: {}",
                path.display()
            )));
        }
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::create_dir(path)?;
    Ok(())
}

fn preflight_output(path: &Path) -> Result<(), CurrentProfileError> {
    match fs::symlink_metadata(path) {
        Ok(metadata) => {
            if metadata.file_type().is_symlink() || !metadata.is_dir() {
                return Err(CurrentProfileError::new(format!(
                    "profile output path must be a nonsymlink directory: {}",
                    path.display()
                )));
            }
            if fs::read_dir(path)?.next().transpose()?.is_some() {
                return Err(CurrentProfileError::new(format!(
                    "profile output directory must be new or empty: {}",
                    path.display()
                )));
            }
        }
        Err(error) if error.kind() == io::ErrorKind::NotFound => {
            let mut ancestor = path.parent().unwrap_or_else(|| Path::new("."));
            while !ancestor.exists() {
                ancestor = ancestor.parent().ok_or_else(|| {
                    CurrentProfileError::new(format!(
                        "profile output has no existing ancestor: {}",
                        path.display()
                    ))
                })?;
            }
            if !ancestor.is_dir() {
                return Err(CurrentProfileError::new(format!(
                    "profile output ancestor is not a directory: {}",
                    ancestor.display()
                )));
            }
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

pub(crate) fn preflight_current_profile(
    config: &CurrentProfileConfig,
) -> Result<Value, CurrentProfileError> {
    let environment = command_environment(&config.environment, &config.unset_environment)?;
    validate_config(config, &environment)?;
    preflight_output(&config.output_dir)?;
    let location = resolve_current_cgroup()?;
    let allocation = read_allocation_state(&location)?;
    let pids = cgroup_pids(&location.path)?;
    let unrelated = unrelated_processes(&pids);
    let cgroup_read_only = visible_cgroup_filesystems_read_only(Path::new(PROC_SELF_MOUNTINFO))?;
    let (_, stats) = read_cpu_stat(&location.path.join("cpu.stat"))?;
    Ok(json!({
        "schema_version": "tasktrace-profile-preflight.v1",
        "ready": cgroup_read_only,
        "runtime": "current",
        "image_identity": Value::Null,
        "profile_sets": [],
        "cpuset_cpus": Value::Null,
        "allocated_cpus": config.allocated_cpus,
        "checks": [
            {"name": "profile_config", "status": "passed"},
            {"name": "output_directory", "status": "passed"},
            {"name": "runtime", "status": "passed", "runtime": "current"},
            {"name": "cgroup_v2", "status": "passed", "required": true, "path": location.path},
            {
                "name": "cgroup_v2_read_only",
                "status": if cgroup_read_only { "passed" } else { "failed" },
                "required": true,
                "reason": if cgroup_read_only { Value::Null } else { json!("The cgroup-v2 filesystem is writable; current-container scope cannot be accepted as immutable.") },
            },
            {"name": "cgroup_cpu_stat", "status": "passed", "usage_usec": stats["usage_usec"]},
            {"name": "allocation_observation", "status": "passed", "allocation": allocation},
            {
                "name": "exclusive_process_scope",
                "status": "pending",
                "exclusive_asserted": config.exclusive,
                "potential_unrelated_processes_now": process_details(&unrelated),
                "reason": "Final exclusivity requires measured-window scope observation plus survivor scans.",
            },
        ],
        "limitations": [
            "The current runtime observes but cannot enforce the scheduler-created allocation.",
            "The in-container controller cannot establish a portable immutable image identity; retain the pod manifest and image digest externally.",
            "Process membership is sampled throughout the measured window; the report retains the observed resolution and maximum sample gap.",
        ],
    }))
}

fn exit_code(status: ExitStatus) -> i32 {
    status
        .code()
        .unwrap_or_else(|| 128 + status.signal().unwrap_or(0))
}

fn configured_command(
    config: &CurrentProfileConfig,
    environment: &CommandEnvironment,
) -> Result<Command, CurrentProfileError> {
    let mut command = if config.strace_mode == "none" {
        let mut command = Command::new(&config.shell);
        command.args(["-c", &config.command]);
        command
    } else {
        let path = environment
            .get(OsStr::new("PATH"))
            .ok_or_else(|| CurrentProfileError::new("task PATH is unavailable"))?;
        let strace = env::split_paths(path)
            .map(|directory| directory.join("strace"))
            .find(|path| {
                path.metadata().ok().is_some_and(|metadata| {
                    metadata.is_file() && metadata.permissions().mode() & 0o111 != 0
                })
            })
            .ok_or_else(|| {
                CurrentProfileError::new("strace is not installed in the current container")
            })?;
        let trace_filter = if config.strace_mode == "execve" {
            "execve"
        } else {
            "%process"
        };
        let mut command = Command::new(strace);
        command
            .args(["-f", "-ttt", "-s"])
            .arg(config.string_limit.to_string())
            .args(["-e", &format!("trace={trace_filter}"), "-o"])
            .arg(config.output_dir.join("process.strace.log"))
            .args([&config.shell, "-c", &config.command]);
        command
    };
    command
        .current_dir(&config.workdir)
        .env_clear()
        .envs(environment)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    if config.adopt_orphans {
        // Orphaned helpers are reparented to the nearest child subreaper, so this keeps them in the
        // measured tree instead of under the container's init, where they would look foreign.
        unsafe {
            command.pre_exec(|| {
                let enable: libc::c_ulong = 1;
                let unused: libc::c_ulong = 0;
                // Best effort: PR_SET_CHILD_SUBREAPER only fails on kernels older than 3.4.
                libc::prctl(libc::PR_SET_CHILD_SUBREAPER, enable, unused, unused, unused);
                Ok(())
            });
        }
    }
    Ok(command)
}

#[derive(Debug)]
struct ControlledOutput {
    output: Output,
    cancellation_requested: bool,
    sent_sigterm: bool,
    sent_sigkill: bool,
}

fn read_child_pipe(mut pipe: impl Read) -> io::Result<Vec<u8>> {
    let mut bytes = Vec::new();
    pipe.read_to_end(&mut bytes)?;
    Ok(bytes)
}

fn signal_process_group(process_group: i32, signal: i32) -> io::Result<()> {
    let result = unsafe { libc::kill(-process_group, signal) };
    if result == 0 {
        return Ok(());
    }
    let error = io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(())
    } else {
        Err(error)
    }
}

/// Returns `true` if at least one process in the given process group is still
/// alive, `false` if the group is gone (`ESRCH`), and `Err` for unexpected
/// failures.  `EPERM` is treated as "exists" — the group is there but owned
/// by another user.
fn process_group_exists(process_group: i32) -> io::Result<bool> {
    if process_group <= 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "invalid process group",
        ));
    }
    if unsafe { libc::kill(-process_group, 0) } == 0 {
        return Ok(true);
    }
    let error = io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(false),
        Some(libc::EPERM) => Ok(true),
        _ => Err(error),
    }
}

/// Kill any surviving descendants in `process_group` and wait for the group to
/// disappear.  This must be called after the leader exits but before joining
/// reader threads, because a surviving background descendant inherits the
/// stdout/stderr pipes and keeps blocking `read_to_end` calls alive forever.
fn reap_surviving_descendants(
    process_group: i32,
) -> Result<(), CurrentProfileError> {
    match process_group_exists(process_group) {
        Ok(false) => return Ok(()),
        Ok(true) => {}
        Err(error) => {
            return Err(CurrentProfileError::new(format!(
                "unable to verify measured command process group after leader exit: {error}"
            )));
        }
    }
    // Background descendants survived — SIGKILL the group and wait.
    signal_process_group(process_group, libc::SIGKILL).map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to kill surviving descendants of measured command: {error}"
        ))
    })?;
    let deadline = Instant::now() + PROCESS_GROUP_CLEANUP_TIMEOUT;
    loop {
        match process_group_exists(process_group) {
            Ok(false) => return Ok(()),
            Ok(true) if Instant::now() < deadline => {
                thread::sleep(SCOPE_OBSERVATION_INTERVAL);
            }
            Ok(true) => {
                return Err(CurrentProfileError::new(
                    "measured command process group did not exit after SIGKILL cleanup; \
                     reader threads may hang",
                ));
            }
            Err(error) => {
                return Err(CurrentProfileError::new(format!(
                    "unable to verify measured command process group cleanup: {error}"
                )));
            }
        }
    }
}

fn execute_controlled_command(
    mut command: Command,
    cancel_requested: Option<&dyn Fn() -> bool>,
    mut scope_observer: Option<&mut ScopeObservation<'_>>,
) -> Result<ControlledOutput, CurrentProfileError> {
    let mut child = command.spawn().map_err(|error| {
        CurrentProfileError::new(format!(
            "unable to execute current-container measured command: {error}"
        ))
    })?;
    let process_group = child.id() as i32;
    let stdout = child
        .stdout
        .take()
        .ok_or_else(|| CurrentProfileError::new("measured command stdout pipe is unavailable"))?;
    let stderr = child
        .stderr
        .take()
        .ok_or_else(|| CurrentProfileError::new("measured command stderr pipe is unavailable"))?;
    let stdout_reader = thread::spawn(move || read_child_pipe(stdout));
    let stderr_reader = thread::spawn(move || read_child_pipe(stderr));

    let mut cancellation_requested = false;
    let mut sent_sigterm = false;
    let mut sent_sigkill = false;
    let mut termination_deadline = None;
    let mut kill_deadline = None;
    let status = loop {
        if let Some(observer) = &mut scope_observer {
            observer.sample(Some(child.id()));
        }
        if let Some(status) = child.try_wait().map_err(|error| {
            CurrentProfileError::new(format!(
                "unable to wait for current-container measured command: {error}"
            ))
        })? {
            break status;
        }
        if !cancellation_requested && cancel_requested.is_some_and(|callback| callback()) {
            cancellation_requested = true;
            signal_process_group(process_group, libc::SIGTERM).map_err(|error| {
                CurrentProfileError::new(format!(
                    "unable to terminate cancelled current-container command group: {error}"
                ))
            })?;
            sent_sigterm = true;
            termination_deadline = Some(Instant::now() + Duration::from_secs(2));
        }
        if sent_sigterm
            && !sent_sigkill
            && termination_deadline.is_some_and(|deadline| Instant::now() >= deadline)
        {
            signal_process_group(process_group, libc::SIGKILL).map_err(|error| {
                CurrentProfileError::new(format!(
                    "unable to kill cancelled current-container command group: {error}"
                ))
            })?;
            sent_sigkill = true;
            kill_deadline = Some(Instant::now() + Duration::from_secs(5));
        }
        if sent_sigkill && kill_deadline.is_some_and(|deadline| Instant::now() >= deadline) {
            return Err(CurrentProfileError::new(
                "cancelled current-container command group did not exit after SIGKILL; process scope is uncertain",
            ));
        }
        thread::sleep(SCOPE_OBSERVATION_INTERVAL);
    };

    // The leader exited, but a surviving background descendant may still hold
    // the stdout/stderr pipes open.  Kill the owned process group before
    // joining the reader threads so read_to_end() can reach EOF.
    reap_surviving_descendants(process_group)?;

    let stdout = stdout_reader
        .join()
        .map_err(|_| CurrentProfileError::new("measured command stdout reader panicked"))??;
    let stderr = stderr_reader
        .join()
        .map_err(|_| CurrentProfileError::new("measured command stderr reader panicked"))??;
    Ok(ControlledOutput {
        output: Output {
            status,
            stdout,
            stderr,
        },
        cancellation_requested,
        sent_sigterm,
        sent_sigkill,
    })
}

fn run_prepare(
    config: &CurrentProfileConfig,
    environment: &CommandEnvironment,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<(), CurrentProfileError> {
    let Some(prepare) = &config.prepare_command else {
        return Ok(());
    };
    let mut command = Command::new(&config.shell);
    command
        .args(["-c", prepare])
        .current_dir(&config.workdir)
        .env_clear()
        .envs(environment)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .process_group(0);
    let controlled =
        execute_controlled_command(command, cancel_requested, None).map_err(|error| {
            CurrentProfileError::new(format!(
                "unable to execute current-container prepare command: {error}"
            ))
        })?;
    let output = controlled.output;
    fs::write(config.output_dir.join("prepare.stdout.txt"), &output.stdout)?;
    fs::write(config.output_dir.join("prepare.stderr.txt"), &output.stderr)?;
    fs::write(
        config.output_dir.join("prepare_exit_code"),
        format!("{}\n", exit_code(output.status)),
    )?;
    if controlled.cancellation_requested {
        write_json(
            &config.output_dir.join("prepare_cancellation.json"),
            &json!({
                "requested": true,
                "sent_sigterm_to_owned_command_group": controlled.sent_sigterm,
                "sent_sigkill_to_owned_command_group": controlled.sent_sigkill,
            }),
        )?;
        return Err(CurrentProfileError::new(
            "current-container prepare command was cancelled",
        ));
    }
    if !output.status.success() {
        return Err(CurrentProfileError::new(format!(
            "current-container prepare command failed with exit code {}: {}",
            exit_code(output.status),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    Ok(())
}

fn cpu_delta(
    before: &BTreeMap<String, u64>,
    after: &BTreeMap<String, u64>,
) -> Result<BTreeMap<String, u64>, CurrentProfileError> {
    let keys = before
        .keys()
        .chain(after.keys())
        .cloned()
        .collect::<BTreeSet<_>>();
    let mut delta = BTreeMap::new();
    let mut decreased = Vec::new();
    for key in keys {
        let before = before.get(&key).copied().unwrap_or(0);
        let after = after.get(&key).copied().unwrap_or(0);
        if after < before {
            decreased.push(key);
        } else {
            delta.insert(key, after - before);
        }
    }
    if !decreased.is_empty() {
        return Err(CurrentProfileError::new(format!(
            "current cgroup CPU counters decreased during the measured window: {}",
            decreased.join(", ")
        )));
    }
    Ok(delta)
}

fn rounded(value: f64, digits: i32) -> f64 {
    let factor = 10_f64.powi(digits);
    (value * factor).round() / factor
}

fn numeric(value: f64) -> Value {
    Value::Number(Number::from_f64(value).expect("finite current-container measurement"))
}

fn write_json(path: &Path, value: &Value) -> Result<(), CurrentProfileError> {
    let mut bytes = serde_json::to_vec_pretty(value)?;
    bytes.push(b'\n');
    fs::write(path, bytes)?;
    Ok(())
}

pub(crate) fn run_current_profile(
    config: &CurrentProfileConfig,
) -> Result<Value, CurrentProfileError> {
    run_current_profile_with_cancel(config, None)
}

pub(crate) fn run_current_profile_with_cancel(
    config: &CurrentProfileConfig,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<Value, CurrentProfileError> {
    run_current_profile_from(
        config,
        Path::new(PROC_SELF_CGROUP),
        Path::new(PROC_SELF_MOUNTINFO),
        cancel_requested,
    )
}

fn run_current_profile_from(
    config: &CurrentProfileConfig,
    proc_self_cgroup: &Path,
    proc_self_mountinfo: &Path,
    cancel_requested: Option<&dyn Fn() -> bool>,
) -> Result<Value, CurrentProfileError> {
    let environment = command_environment(&config.environment, &config.unset_environment)?;
    validate_config(config, &environment)?;
    prepare_output(&config.output_dir)?;
    run_prepare(config, &environment, cancel_requested)?;

    let location_before = resolve_current_cgroup_from(proc_self_cgroup, proc_self_mountinfo)?;
    let allocation_before = read_allocation_state(&location_before)?;
    let identity = environment_identity(&location_before);
    let pids_before = cgroup_pids(&location_before.path)?;
    let unrelated_before = unrelated_processes(&pids_before);
    let mut scope_observation =
        ScopeObservation::new(&location_before, proc_self_cgroup, proc_self_mountinfo);
    scope_observation.sample(None);

    let (raw_before, stats_before, start_ns, before_read_duration_ns) =
        read_timed_cpu_stat(&location_before.path.join("cpu.stat"))?;
    let controlled = execute_controlled_command(
        configured_command(config, &environment)?,
        cancel_requested,
        Some(&mut scope_observation),
    )?;
    let output = controlled.output;
    let (raw_after, stats_after, end_ns, after_read_duration_ns) =
        read_timed_cpu_stat(&location_before.path.join("cpu.stat"))?;
    scope_observation.sample(None);
    if end_ns <= start_ns {
        return Err(CurrentProfileError::new(
            "current-container measured window has non-positive duration",
        ));
    }

    // Scope and survivors are inspected before any profile artifact writes.
    // Otherwise writing the first artifact could hide a short-lived sibling.
    let location_after = resolve_current_cgroup_from(proc_self_cgroup, proc_self_mountinfo)?;
    let pids_after = cgroup_pids(&location_after.path)?;
    let unrelated_after = unrelated_processes(&pids_after);
    let allocation_after = read_allocation_state(&location_after)?;

    fs::write(config.output_dir.join("cpu.stat.before"), raw_before)?;
    fs::write(config.output_dir.join("cpu.stat.after"), raw_after)?;
    fs::write(config.output_dir.join("command.stdout.txt"), &output.stdout)?;
    fs::write(config.output_dir.join("command.stderr.txt"), &output.stderr)?;
    fs::write(config.output_dir.join("start_ns"), format!("{start_ns}\n"))?;
    fs::write(config.output_dir.join("end_ns"), format!("{end_ns}\n"))?;
    fs::write(
        config.output_dir.join("exit_code"),
        format!("{}\n", exit_code(output.status)),
    )?;
    fs::write(
        config.output_dir.join("strace_mode"),
        format!("{}\n", config.strace_mode),
    )?;

    let delta = cpu_delta(&stats_before, &stats_after)?;
    let wall_sec = (end_ns - start_ns) as f64 / 1_000_000_000.0;
    let usage_sec = delta["usage_usec"] as f64 / 1_000_000.0;
    let average_cores = usage_sec / wall_sec;
    let (capacity, capacity_source, effective_cpu_ids) =
        capacity(&allocation_before, config.allocated_cpus);
    let utilization = capacity
        .filter(|value| *value > 0.0)
        .map(|value| average_cores / value);
    let endpoint_location_stable = location_before == location_after;
    let location_stable = endpoint_location_stable
        && scope_observation.cgroup_path_stable
        && scope_observation.command_membership_stable;
    let allocation_stable = allocation_before == allocation_after;
    let scope_observation_complete = scope_observation.complete(start_ns, end_ns);
    let scope_observation_covers_window = scope_observation.covers_window(start_ns, end_ns);
    let observed_unrelated = scope_observation.observed_unrelated_details();
    let observed_transient = scope_observation.observed_transient_pids();
    let valid = config.exclusive
        && !controlled.cancellation_requested
        && location_stable
        && allocation_stable
        && scope_observation_complete
        && scope_observation.cgroup_filesystem_read_only
        && observed_unrelated.is_empty()
        && unrelated_before.is_empty()
        && unrelated_after.is_empty();

    let mut warnings = vec![
        "Current-container cgroup accounting includes every process sharing this container. Comparable results require a fresh, single-task sandbox.".to_owned(),
        "Process and cgroup membership sampling is periodic; validity relies on the exclusivity assertion and read-only cgroup mounts and does not prove absence of a privileged external process entirely between samples.".to_owned(),
    ];
    if !unrelated_before.is_empty() {
        warnings.push("Potential unrelated processes existed before replay.".to_owned());
    }
    if !unrelated_after.is_empty() {
        warnings
            .push("Processes outside the controller ancestry remained after replay.".to_owned());
    }
    if !location_stable {
        warnings.push(
            "The controller or measured command changed cgroup scope during replay.".to_owned(),
        );
    }
    if !allocation_stable {
        warnings.push("The cgroup CPU allocation changed during replay.".to_owned());
    }
    if config.strace_mode != "none" {
        warnings.push("This run includes strace overhead.".to_owned());
    }
    if !config.exclusive {
        warnings.push("Container exclusivity was not asserted; CPU data is diagnostic rather than approved for comparison.".to_owned());
    }
    if controlled.cancellation_requested {
        warnings.push(
            "The measured replay was cancelled; CPU data is diagnostic rather than approved for comparison."
                .to_owned(),
        );
    }
    if !scope_observation.cgroup_filesystem_read_only {
        warnings.push(
            "The cgroup-v2 filesystem was writable; immutable current-container scope was not established."
                .to_owned(),
        );
    }
    if !scope_observation_complete {
        warnings.push(
            "Measured-window process and cgroup scope observation was incomplete.".to_owned(),
        );
    }
    if !observed_unrelated.is_empty() {
        warnings.push(
            "A process outside the controller or measured-command ancestry was observed during replay."
                .to_owned(),
        );
    }
    if !observed_transient.is_empty() {
        warnings.push(format!(
            "{} process(es) in the container cgroup exited before their ancestry could be read (lifetimes shorter than the {}ms sampling interval); they are reported as transient and do not affect validity.",
            observed_transient.len(),
            SCOPE_OBSERVATION_INTERVAL.as_millis(),
        ));
    }

    let task_environment = json!({
        "unset_variables": config.unset_environment,
        "overridden_variables": config.environment.iter().filter_map(|value| value.split_once('=').map(|(key, _)| key)).collect::<Vec<_>>(),
    });
    let report = json!({
        "schema_version": "container-command-profile.v1",
        "runtime": "current",
        "scope": "current-container",
        "image": Value::Null,
        "image_identity": Value::Null,
        "container": Value::Null,
        "environment_identity": identity,
        "task_environment": task_environment,
        "measurement_validity": {
            "valid_for_average_cpu_reporting": valid,
            "cancelled": controlled.cancellation_requested,
            "requires_exclusive_single_task_container": true,
            "exclusive_single_task_container_asserted": config.exclusive,
            "cgroup_path_stable": location_stable,
            "cgroup_path_stable_at_endpoints": endpoint_location_stable,
            "controller_cgroup_path_stable_during_replay": scope_observation.cgroup_path_stable,
            "measured_command_cgroup_path_stable_during_replay": scope_observation.command_membership_stable,
            "cgroup_filesystem_read_only": scope_observation.cgroup_filesystem_read_only,
            "allocation_stable": allocation_stable,
            "scope_observation_complete": scope_observation_complete,
            "scope_observation_covers_measured_window": scope_observation_covers_window,
            "potential_unrelated_processes_before": process_details(&unrelated_before),
            "processes_outside_controller_or_command_ancestry_during": observed_unrelated,
            "transient_unattributed_processes_during": observed_transient,
            "processes_outside_controller_ancestry_after": process_details(&unrelated_after),
            "scope_observation": {
                "method": "in-window-cgroup-membership-polling+kernel-read-only-cgroup-v2",
                "orphan_adoption": if config.adopt_orphans { "measured-command-child-subreaper" } else { "off" },
                "absence_between_samples_proven": false,
                "external_injector_excluded_by_assertion_not_observation": true,
                "requested_interval_ns": SCOPE_OBSERVATION_INTERVAL.as_nanos(),
                "sample_count": scope_observation.samples,
                "first_sample_ns": scope_observation.first_sample_ns,
                "last_sample_ns": scope_observation.last_sample_ns,
                "maximum_sample_gap_ns": scope_observation.maximum_sample_gap_ns,
                "observed_membership_paths": scope_observation.observed_membership_paths,
                "errors": scope_observation.errors,
            },
        },
        "command": config.command,
        "prepare_command": config.prepare_command,
        "shell": config.shell,
        "strace": {
            "mode": config.strace_mode,
            "log_path": if config.output_dir.join("process.strace.log").is_file() { json!(config.output_dir.join("process.strace.log")) } else { Value::Null },
            "string_limit": config.string_limit,
        },
        "command_exit_code": exit_code(output.status),
        "runtime_returncode": exit_code(output.status),
        "docker_returncode": Value::Null,
        "cancellation": {
            "requested": controlled.cancellation_requested,
            "sent_sigterm_to_owned_command_group": controlled.sent_sigterm,
            "sent_sigkill_to_owned_command_group": controlled.sent_sigkill,
        },
        "cpu": {
            "method": "current-container-cgroup-v2-cpu.stat",
            "scope_path": location_before.path,
            "wall_sec": rounded(wall_sec, 9),
            "usage_sec": rounded(usage_sec, 9),
            "user_sec": rounded(delta.get("user_usec").copied().unwrap_or(0) as f64 / 1_000_000.0, 9),
            "system_sec": rounded(delta.get("system_usec").copied().unwrap_or(0) as f64 / 1_000_000.0, 9),
            "average_logical_cores_used": rounded(average_cores, 9),
            "allocated_logical_cpus": capacity.map(numeric).unwrap_or(Value::Null),
            "allocated_cpu_source": capacity_source,
            "effective_logical_cpu_ids": effective_cpu_ids,
            "effective_logical_cpu_count": if effective_cpu_ids.is_empty() { Value::Null } else { json!(effective_cpu_ids.len()) },
            "cgroup_cpuset": allocation_before.cpuset,
            "cgroup_cpuset_logical_cpu_ids": allocation_before.cpuset_ids,
            "process_affinity_logical_cpu_ids": allocation_before.affinity_ids,
            "enforced_logical_cpu_limit": allocation_before.quota_cores.map(numeric).unwrap_or(Value::Null),
            "enforced_cpu_limit_source": if allocation_before.quota_cores.is_some() { json!("cgroup-v2-cpu.max") } else { Value::Null },
            "cpu_max": allocation_before.cpu_max,
            "cpu_max_hierarchy": allocation_before.cpu_max_hierarchy,
            "utilization_fraction_of_allocation": utilization.map(|value| numeric(rounded(value, 9))).unwrap_or(Value::Null),
            "utilization_percent_of_allocation": utilization.map(|value| numeric(rounded(value * 100.0, 6))).unwrap_or(Value::Null),
            "throttling": {
                "periods": delta.get("nr_periods").copied().unwrap_or(0),
                "throttled_periods": delta.get("nr_throttled").copied().unwrap_or(0),
                "throttled_sec": rounded(delta.get("throttled_usec").copied().unwrap_or(0) as f64 / 1_000_000.0, 9),
                "burst_count": delta.get("nr_bursts").copied().unwrap_or(0),
                "burst_sec": rounded(delta.get("burst_usec").copied().unwrap_or(0) as f64 / 1_000_000.0, 9),
            },
            "raw_cpu_stat_delta": delta,
            "snapshot_read_duration_ns": {"before": before_read_duration_ns, "after": after_read_duration_ns},
        },
        "warning": warnings.join(" "),
    });
    let environment_report = json!({
        "environment_identity": report["environment_identity"],
        "task_environment": report["task_environment"],
        "allocation_before": allocation_before,
        "allocation_after": allocation_after,
        "pids_before": process_details(&pids_before),
        "pids_after": process_details(&pids_after),
        "scope_observation": report["measurement_validity"]["scope_observation"],
        "processes_outside_controller_or_command_ancestry_during": report["measurement_validity"]["processes_outside_controller_or_command_ancestry_during"],
        "transient_unattributed_processes_during": report["measurement_validity"]["transient_unattributed_processes_during"],
    });
    write_json(
        &config.output_dir.join("current_environment.json"),
        &environment_report,
    )?;
    write_json(&config.output_dir.join("profile_report.json"), &report)?;
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use tempfile::{tempdir, NamedTempFile};

    fn fake_mount() -> (tempfile::TempDir, CgroupV2Location) {
        let directory = tempdir().unwrap();
        let cgroup = directory.path().join("cgroup");
        fs::create_dir(&cgroup).unwrap();
        fs::write(cgroup.join("cpu.stat"), "usage_usec 10\n").unwrap();
        fs::write(
            cgroup.join("cgroup.procs"),
            format!("{}\n", std::process::id()),
        )
        .unwrap();
        let location = CgroupV2Location {
            mount_point: cgroup.clone(),
            mount_root: "/".to_owned(),
            membership_path: "/".to_owned(),
            path: cgroup,
        };
        (directory, location)
    }

    fn fake_proc_cgroup_files(
        directory: &Path,
        mount: &Path,
        membership_value: &str,
    ) -> (PathBuf, PathBuf) {
        let membership = directory.join("membership");
        fs::write(&membership, format!("0::{membership_value}\n")).unwrap();
        let mountinfo = directory.join("mountinfo");
        fs::write(
            &mountinfo,
            format!("29 23 0:26 / {} rw - cgroup2 cgroup rw\n", mount.display()),
        )
        .unwrap();
        (membership, mountinfo)
    }

    #[test]
    fn resolves_private_and_host_relative_cgroup_namespaces() {
        let directory = tempdir().unwrap();
        let mount = directory.path().join("cgroup");
        fs::create_dir(&mount).unwrap();
        fs::write(mount.join("cpu.stat"), "usage_usec 0\n").unwrap();
        fs::write(mount.join("cgroup.procs"), "1\n").unwrap();
        let membership = directory.path().join("membership");
        fs::write(&membership, "0::/\n").unwrap();
        let mountinfo = directory.path().join("mountinfo");
        fs::write(
            &mountinfo,
            format!(
                "29 23 0:26 / {} rw,nosuid,nodev,noexec,relatime - cgroup2 cgroup rw\n",
                mount.display()
            ),
        )
        .unwrap();
        assert_eq!(
            resolve_current_cgroup_from(&membership, &mountinfo)
                .unwrap()
                .path,
            mount
        );

        let child = directory.path().join("other").join("container");
        fs::create_dir_all(&child).unwrap();
        fs::write(child.join("cpu.stat"), "usage_usec 0\n").unwrap();
        fs::write(child.join("cgroup.procs"), "1\n").unwrap();
        fs::write(&membership, "0::/kubepods/pod/container\n").unwrap();
        fs::write(
            &mountinfo,
            format!(
                "29 23 0:26 /kubepods/pod {} rw - cgroup2 cgroup rw\n",
                directory.path().join("other").display()
            ),
        )
        .unwrap();
        assert_eq!(
            resolve_current_cgroup_from(&membership, &mountinfo)
                .unwrap()
                .path,
            child
        );
    }

    #[test]
    fn cpu_max_and_timed_snapshot_contracts() {
        assert_eq!(parse_cpu_max(Some("100000 100000")).unwrap(), Some(1.0));
        assert_eq!(parse_cpu_max(Some("250000 100000")).unwrap(), Some(2.5));
        assert_eq!(parse_cpu_max(Some("max 100000")).unwrap(), None);
        let mut file = NamedTempFile::new().unwrap();
        writeln!(file, "usage_usec 10").unwrap();
        let mut times = [100_u64, 200].into_iter();
        let (_, stats, midpoint, duration) =
            read_timed_cpu_stat_with(file.path(), || times.next().unwrap()).unwrap();
        assert_eq!(stats["usage_usec"], 10);
        assert_eq!(midpoint, 150);
        assert_eq!(duration, 100);
    }

    #[test]
    fn allocation_uses_tightest_visible_ancestor_quota() {
        let directory = tempdir().unwrap();
        let mount = directory.path().join("cgroup");
        let child = mount.join("child");
        fs::create_dir_all(&child).unwrap();
        fs::write(mount.join("cpu.max"), "200000 100000\n").unwrap();
        fs::write(child.join("cpu.max"), "max 100000\n").unwrap();
        fs::write(child.join("cpuset.cpus.effective"), "0-3\n").unwrap();
        let allocation = read_allocation_state_with_affinity(
            &CgroupV2Location {
                mount_point: mount,
                mount_root: "/".to_owned(),
                membership_path: "/child".to_owned(),
                path: child,
            },
            vec![0, 1, 2, 3],
        )
        .unwrap();
        assert_eq!(allocation.cpu_max.as_deref(), Some("max 100000"));
        assert_eq!(allocation.quota_cores, Some(2.0));
        let (capacity, source, ids) = capacity(&allocation, None);
        assert_eq!(capacity, Some(2.0));
        assert_eq!(source.as_deref(), Some("cgroup-v2-cpu.max+cpuset+affinity"));
        assert_eq!(ids, vec![0, 1, 2, 3]);
    }

    #[test]
    fn invalid_process_ids_and_descendant_read_failures_fail_closed() {
        let (_directory, location) = fake_mount();
        fs::write(location.path.join("cgroup.procs"), "not-a-pid\n").unwrap();
        assert!(cgroup_pids(&location.path)
            .unwrap_err()
            .to_string()
            .contains("invalid process ID"));
    }

    #[test]
    fn scope_observer_reports_a_vanished_pid_as_transient_not_unrelated() {
        const IMPOSSIBLE_LINUX_PID: u32 = 4_000_000_000;

        let (directory, location) = fake_mount();
        let (membership, mountinfo) =
            fake_proc_cgroup_files(directory.path(), &location.mount_point, "/");
        let mut observation = ScopeObservation::new(&location, &membership, &mountinfo);

        observation.sample(None);
        fs::write(
            location.path.join("cgroup.procs"),
            format!("{}\n{IMPOSSIBLE_LINUX_PID}\n", std::process::id()),
        )
        .unwrap();
        observation.sample(None);
        fs::write(
            location.path.join("cgroup.procs"),
            format!("{}\n", std::process::id()),
        )
        .unwrap();
        observation.sample(None);

        assert_eq!(
            cgroup_pids(&location.path).unwrap(),
            vec![std::process::id()]
        );
        assert!(observation.observed_unrelated.is_empty());
        assert_eq!(
            observation.observed_transient_pids(),
            vec![IMPOSSIBLE_LINUX_PID]
        );
    }

    #[test]
    fn scope_observer_flags_a_live_process_outside_the_command_tree_by_name() {
        let (directory, location) = fake_mount();
        let (membership, mountinfo) =
            fake_proc_cgroup_files(directory.path(), &location.mount_point, "/");
        let mut observation = ScopeObservation::new(&location, &membership, &mountinfo);
        let mut bystander = Command::new("sleep").arg("30").spawn().unwrap();

        fs::write(
            location.path.join("cgroup.procs"),
            format!("{}\n{}\n", std::process::id(), bystander.id()),
        )
        .unwrap();
        // Not a descendant of the (fictitious) command root.
        observation.sample(Some(4_000_000_000));
        bystander.kill().unwrap();
        bystander.wait().unwrap();

        assert!(observation.observed_transient.is_empty());
        let (name, chain) = observation.observed_unrelated.get(&bystander.id()).unwrap();
        assert_eq!(name, &Some("sleep".to_owned()));
        assert_eq!(chain.first(), Some(&bystander.id()));
    }

    #[test]
    fn scope_observer_accepts_descendants_of_the_command_root() {
        let (directory, location) = fake_mount();
        let (membership, mountinfo) =
            fake_proc_cgroup_files(directory.path(), &location.mount_point, "/");
        let mut observation = ScopeObservation::new(&location, &membership, &mountinfo);
        let mut helper = Command::new("sleep").arg("30").spawn().unwrap();

        fs::write(
            location.path.join("cgroup.procs"),
            format!("{}\n{}\n", std::process::id(), helper.id()),
        )
        .unwrap();
        observation.sample(Some(std::process::id()));
        helper.kill().unwrap();
        helper.wait().unwrap();

        assert!(observation.observed_unrelated.is_empty());
        assert!(observation.observed_transient.is_empty());
    }

    #[test]
    fn classify_ancestry_distinguishes_descendant_unrelated_and_vanished() {
        let mut child = Command::new("sleep").arg("30").spawn().unwrap();
        assert_eq!(
            classify_ancestry(child.id(), Some(std::process::id())).0,
            Ancestry::Descendant
        );
        let (ancestry, chain) = classify_ancestry(child.id(), Some(4_000_000_000));
        assert_eq!(ancestry, Ancestry::Unrelated);
        assert_eq!(chain.first(), Some(&child.id()));
        assert!(chain.contains(&std::process::id()));
        assert_eq!(classify_ancestry(child.id(), None).0, Ancestry::Unrelated);
        child.kill().unwrap();
        child.wait().unwrap();
        assert_eq!(
            classify_ancestry(4_000_000_000, Some(std::process::id())).0,
            Ancestry::Vanished
        );
    }

    #[test]
    fn an_ancestor_exiting_mid_walk_does_not_make_a_live_process_unrelated() {
        // 30 -> 20 -> 10 (root); 20 exits mid-walk and 30 is reparented to the root.
        let walks = std::cell::Cell::new(0);
        let parent_of = |pid: u32| match pid {
            30 if walks.get() == 0 => {
                walks.set(1);
                Some(20)
            }
            30 => Some(10),
            _ => None,
        };
        assert_eq!(
            classify_ancestry_with(30, Some(10), parent_of, |_| true).0,
            Ancestry::Descendant
        );

        assert_eq!(
            classify_ancestry_with(30, Some(10), |pid| (pid == 30).then_some(20), |_| true).0,
            Ancestry::Vanished
        );

        let (ancestry, chain) = classify_ancestry_with(
            30,
            Some(10),
            |pid| match pid {
                30 => Some(1),
                1 => Some(0),
                _ => None,
            },
            |_| true,
        );
        assert_eq!(ancestry, Ancestry::Unrelated);
        assert_eq!(chain, vec![30, 1]);
    }

    #[test]
    fn scope_observer_retains_a_transient_controller_cgroup_migration() {
        let (directory, location) = fake_mount();
        let migrated = location.mount_point.join("migrated");
        fs::create_dir(&migrated).unwrap();
        fs::write(migrated.join("cpu.stat"), "usage_usec 10\n").unwrap();
        fs::write(
            migrated.join("cgroup.procs"),
            format!("{}\n", std::process::id()),
        )
        .unwrap();
        let (membership, mountinfo) =
            fake_proc_cgroup_files(directory.path(), &location.mount_point, "/");
        let mut observation = ScopeObservation::new(&location, &membership, &mountinfo);

        observation.sample(None);
        fs::write(&membership, "0::/migrated\n").unwrap();
        observation.sample(None);
        fs::write(&membership, "0::/\n").unwrap();
        observation.sample(None);

        assert!(!observation.cgroup_path_stable);
        assert_eq!(
            observation
                .observed_membership_paths
                .into_iter()
                .collect::<Vec<_>>(),
            vec!["/".to_owned(), "/migrated".to_owned()]
        );
    }

    #[test]
    fn writable_cgroup_filesystem_is_rejected_by_the_validity_gate() {
        let (directory, location) = fake_mount();
        let (_, mountinfo) = fake_proc_cgroup_files(directory.path(), &location.mount_point, "/");
        assert!(!cgroup_filesystem_read_only(&location.mount_point).unwrap());
        assert!(!visible_cgroup_filesystems_read_only(&mountinfo).unwrap());
    }

    #[test]
    fn environment_overrides_and_unsets_without_leaking_values_to_summary() {
        let environment = command_environment(
            &["TASKTRACE_CURRENT_TEST=task".to_owned()],
            &["VIRTUAL_ENV".to_owned()],
        )
        .unwrap();
        assert_eq!(
            environment.get(OsStr::new("TASKTRACE_CURRENT_TEST")),
            Some(&OsString::from("task"))
        );
        assert!(!environment.contains_key(OsStr::new("VIRTUAL_ENV")));
    }

    #[test]
    fn controller_environment_sanitizer_preserves_unrelated_task_paths() {
        let directory = tempdir().unwrap();
        let controller = directory.path().join("controller");
        let virtualenv = controller.join(".venv");
        let executable = virtualenv.join("bin/tasktrace");
        fs::create_dir_all(executable.parent().unwrap()).unwrap();
        fs::write(&executable, b"binary").unwrap();
        let task_python = directory.path().join("task-python");
        fs::create_dir(&task_python).unwrap();
        let mut values = BTreeMap::new();
        values.insert(
            "PATH".to_owned(),
            format!("{}:/usr/bin:/bin", virtualenv.join("bin").display()),
        );
        values.insert(
            "PYTHONPATH".to_owned(),
            format!("{}:{}", controller.display(), task_python.display()),
        );
        values.insert("VIRTUAL_ENV".to_owned(), virtualenv.display().to_string());
        values.insert("VIRTUAL_ENV_PROMPT".to_owned(), "controller".to_owned());
        values.insert("UV_RUN_RECURSION_DEPTH".to_owned(), "1".to_owned());

        let sanitized = sanitized_controller_environment_from(
            &values,
            &executable,
            Some(&controller),
            directory.path(),
        )
        .unwrap();
        assert_eq!(
            sanitized.assignments,
            vec![
                "PATH=/usr/bin:/bin".to_owned(),
                format!("PYTHONPATH={}", task_python.display()),
            ]
        );
        assert_eq!(
            sanitized.unset_names,
            [
                "PYTHONPATH",
                "UV_RUN_RECURSION_DEPTH",
                "VIRTUAL_ENV",
                "VIRTUAL_ENV_PROMPT",
            ]
        );
    }

    #[test]
    fn controlled_command_cancellation_terminates_only_its_owned_process_group() {
        let directory = tempdir().unwrap();
        let config = CurrentProfileConfig {
            command: "sleep 30".to_owned(),
            prepare_command: None,
            output_dir: directory.path().join("output"),
            workdir: directory.path().to_path_buf(),
            shell: "/bin/bash".to_owned(),
            environment: Vec::new(),
            unset_environment: Vec::new(),
            strace_mode: "none".to_owned(),
            string_limit: 4096,
            allocated_cpus: None,
            exclusive: true,
            adopt_orphans: false,
        };
        fs::create_dir(&config.output_dir).unwrap();
        let environment = command_environment(&[], &[]).unwrap();
        let polls = AtomicUsize::new(0);
        let cancel = || polls.fetch_add(1, Ordering::SeqCst) >= 2;
        let result = execute_controlled_command(
            configured_command(&config, &environment).unwrap(),
            Some(&cancel),
            None,
        )
        .unwrap();
        assert!(result.cancellation_requested);
        assert!(result.sent_sigterm);
        assert!(!result.sent_sigkill);
        assert_eq!(result.output.status.signal(), Some(libc::SIGTERM));
        assert!(result.output.stdout.is_empty());
        assert!(result.output.stderr.is_empty());
    }

    #[test]
    fn current_profile_runs_one_command_and_retains_raw_cpu_math() {
        let directory = tempdir().unwrap();
        let cgroup = directory.path().join("cgroup");
        fs::create_dir(&cgroup).unwrap();
        fs::write(
            cgroup.join("cpu.stat"),
            "usage_usec 1000000\nuser_usec 600000\nsystem_usec 400000\nnr_periods 10\nnr_throttled 1\nthrottled_usec 100000\n",
        )
        .unwrap();
        fs::write(
            cgroup.join("cgroup.procs"),
            format!("{}\n", std::process::id()),
        )
        .unwrap();
        fs::write(cgroup.join("cpu.max"), "100000 100000\n").unwrap();
        let affinity = process_affinity().unwrap();
        fs::write(
            cgroup.join("cpuset.cpus.effective"),
            affinity
                .iter()
                .map(u32::to_string)
                .collect::<Vec<_>>()
                .join(","),
        )
        .unwrap();
        let membership = directory.path().join("membership");
        fs::write(&membership, "0::/\n").unwrap();
        let mountinfo = directory.path().join("mountinfo");
        fs::write(
            &mountinfo,
            format!("29 23 0:26 / {} rw - cgroup2 cgroup rw\n", cgroup.display()),
        )
        .unwrap();
        let after = "usage_usec 3000000\\nuser_usec 2100000\\nsystem_usec 900000\\nnr_periods 30\\nnr_throttled 6\\nthrottled_usec 1600000\\n";
        let config = CurrentProfileConfig {
            command: format!("printf '{after}' > {}", cgroup.join("cpu.stat").display()),
            prepare_command: None,
            output_dir: directory.path().join("profile"),
            workdir: directory.path().to_path_buf(),
            shell: "/bin/bash".to_owned(),
            environment: Vec::new(),
            unset_environment: Vec::new(),
            strace_mode: "none".to_owned(),
            string_limit: 4096,
            allocated_cpus: None,
            exclusive: true,
            adopt_orphans: true,
        };

        let report = run_current_profile_from(&config, &membership, &mountinfo, None).unwrap();
        assert_eq!(report["runtime"], "current");
        assert_eq!(report["scope"], "current-container");
        assert_eq!(report["cpu"]["usage_sec"], 2.0);
        assert_eq!(report["cpu"]["throttling"]["throttled_periods"], 5);
        assert_eq!(
            report["measurement_validity"]["valid_for_average_cpu_reporting"],
            false
        );
        assert_eq!(
            report["measurement_validity"]["cgroup_filesystem_read_only"],
            false
        );
        assert!(config.output_dir.join("cpu.stat.before").is_file());
        assert!(config.output_dir.join("cpu.stat.after").is_file());
        assert!(config.output_dir.join("current_environment.json").is_file());
        assert!(config.output_dir.join("profile_report.json").is_file());
        assert_eq!(
            report["measurement_validity"]["scope_observation"]["orphan_adoption"],
            "measured-command-child-subreaper"
        );
    }

    /// Runs a measured command whose child backgrounds a helper and exits, orphaning it, and returns
    /// (measured shell pid, helper's parent pid after orphaning).
    fn orphaned_helper_parent(adopt_orphans: bool) -> (String, String) {
        let directory = tempdir().unwrap();
        let config = CurrentProfileConfig {
            command: "helper=$(sh -c 'sleep 5 >/dev/null 2>&1 & echo $!'); sleep 0.3; \
                      echo \"$$ $(cut -d' ' -f4 /proc/$helper/stat)\"; kill $helper"
                .to_owned(),
            prepare_command: None,
            output_dir: directory.path().join("output"),
            workdir: directory.path().to_path_buf(),
            shell: "/bin/bash".to_owned(),
            environment: Vec::new(),
            unset_environment: Vec::new(),
            strace_mode: "none".to_owned(),
            string_limit: 4096,
            allocated_cpus: None,
            exclusive: true,
            adopt_orphans,
        };
        let environment = command_environment(&[], &[]).unwrap();
        let result = execute_controlled_command(
            configured_command(&config, &environment).unwrap(),
            None,
            None,
        )
        .unwrap();
        let stdout = String::from_utf8(result.output.stdout).unwrap();
        let fields: Vec<&str> = stdout.split_whitespace().collect();
        assert_eq!(fields.len(), 2, "unexpected measured output {stdout:?}");
        (fields[0].to_owned(), fields[1].to_owned())
    }

    #[test]
    fn measured_command_adopts_helpers_its_children_orphan() {
        let (shell, helper_parent) = orphaned_helper_parent(true);
        assert_eq!(
            helper_parent, shell,
            "the orphaned helper must be reparented to the measured command"
        );
    }

    #[test]
    fn without_adoption_orphaned_helpers_leave_the_measured_tree() {
        let (shell, helper_parent) = orphaned_helper_parent(false);
        assert_ne!(helper_parent, shell);
    }
}
