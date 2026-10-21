// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Small, explicit host identity for cross-machine CPU comparisons.
//!
//! Selected logical CPUs are inspected individually. Missing or inconsistent
//! topology is an error rather than a partially populated comparison identity.

use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::ffi::CStr;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

pub const HOST_SCHEMA_VERSION: &str = "tasktrace-run-host.v1";
pub const SYS_CPU_ROOT: &str = "/sys/devices/system/cpu";
pub const CPUINFO_PATH: &str = "/proc/cpuinfo";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostIdentityError(String);

impl HostIdentityError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for HostIdentityError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl std::error::Error for HostIdentityError {}

/// Platform facts supplied separately from sysfs so topology tests and callers
/// reconstructing a host snapshot can bind the same JSON contract explicitly.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct HostSystemIdentity {
    pub machine: String,
    pub kernel_release: String,
    pub processor: String,
    pub logical_cpu_count: Option<usize>,
}

impl HostSystemIdentity {
    /// Read the native machine, kernel release, processor fallback, and online
    /// logical-CPU count from the current Linux host.
    pub fn detect() -> Result<Self, HostIdentityError> {
        let mut name = std::mem::MaybeUninit::<libc::utsname>::zeroed();
        let result = unsafe { libc::uname(name.as_mut_ptr()) };
        if result != 0 {
            return Err(HostIdentityError::new(format!(
                "unable to read host platform identity: {}",
                io::Error::last_os_error()
            )));
        }
        let name = unsafe { name.assume_init() };
        let machine = c_string(&name.machine);
        let kernel_release = c_string(&name.release);
        let count = unsafe { libc::sysconf(libc::_SC_NPROCESSORS_ONLN) };
        Ok(Self {
            processor: machine.clone(),
            machine,
            kernel_release,
            logical_cpu_count: (count > 0).then_some(count as usize),
        })
    }
}

fn c_string(buffer: &[libc::c_char]) -> String {
    unsafe { CStr::from_ptr(buffer.as_ptr()) }
        .to_string_lossy()
        .into_owned()
}

fn cpuinfo_fields(path: &Path) -> Result<BTreeMap<String, String>, HostIdentityError> {
    let text = fs::read_to_string(path).map_err(|error| {
        HostIdentityError::new(format!(
            "unable to read CPU identity from {}: {error}",
            path.display()
        ))
    })?;
    // Path.read_text() uses universal-newline handling in Python. Preserve
    // that behavior for captured cpuinfo fixtures with CRLF or CR endings.
    let normalized = text.replace("\r\n", "\n").replace('\r', "\n");
    let first_block = normalized
        .split_once("\n\n")
        .map_or(normalized.as_str(), |(first, _)| first);
    let mut fields = BTreeMap::new();
    for line in first_block.lines() {
        let Some((key, value)) = line.split_once(':') else {
            continue;
        };
        fields.insert(key.trim().to_lowercase(), value.trim().to_owned());
    }
    Ok(fields)
}

fn integer(path: &Path, field: &str) -> Result<i64, HostIdentityError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        HostIdentityError::new(format!(
            "unable to read {field} from {}: {error}",
            path.display()
        ))
    })?;
    raw.trim().parse::<i64>().map_err(|error| {
        HostIdentityError::new(format!(
            "unable to read {field} from {}: {error}",
            path.display()
        ))
    })
}

fn parse_cpuset_cpu_ids(cpuset: &str) -> Result<Vec<i64>, HostIdentityError> {
    let mut cpu_ids = BTreeSet::new();
    for raw_part in cpuset.split(',') {
        let part = raw_part.trim();
        if part.is_empty() {
            return Err(HostIdentityError::new(format!(
                "invalid empty CPU set component in {cpuset:?}"
            )));
        }
        if let Some((start_raw, end_raw)) = part.split_once('-') {
            let start = start_raw
                .trim()
                .parse::<i64>()
                .map_err(|_| HostIdentityError::new(format!("invalid CPU set range: {part:?}")))?;
            let end = end_raw
                .trim()
                .parse::<i64>()
                .map_err(|_| HostIdentityError::new(format!("invalid CPU set range: {part:?}")))?;
            if start < 0 || end < start {
                return Err(HostIdentityError::new(format!(
                    "invalid CPU set range: {part:?}"
                )));
            }
            cpu_ids.extend(start..=end);
        } else {
            let cpu_id = part
                .parse::<i64>()
                .map_err(|_| HostIdentityError::new(format!("invalid CPU ID: {part:?}")))?;
            if cpu_id < 0 {
                return Err(HostIdentityError::new(format!("invalid CPU ID: {part:?}")));
            }
            cpu_ids.insert(cpu_id);
        }
    }
    if cpu_ids.is_empty() {
        return Err(HostIdentityError::new("CPU set must not be empty"));
    }
    Ok(cpu_ids.into_iter().collect())
}

fn siblings(path: &Path) -> Result<Vec<i64>, HostIdentityError> {
    let raw = fs::read_to_string(path).map_err(|error| {
        HostIdentityError::new(format!(
            "unable to read thread siblings from {}: {error}",
            path.display()
        ))
    })?;
    parse_cpuset_cpu_ids(raw.trim()).map_err(|error| {
        HostIdentityError::new(format!(
            "unable to read thread siblings from {}: {error}",
            path.display()
        ))
    })
}

fn numa_node(cpu_dir: &Path) -> Result<Option<i64>, HostIdentityError> {
    let entries = fs::read_dir(cpu_dir).map_err(|error| {
        HostIdentityError::new(format!(
            "unable to inspect NUMA-node links in {}: {error}",
            cpu_dir.display()
        ))
    })?;
    let mut nodes = Vec::new();
    for entry in entries {
        let entry = entry.map_err(|error| {
            HostIdentityError::new(format!(
                "unable to inspect NUMA-node links in {}: {error}",
                cpu_dir.display()
            ))
        })?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(suffix) = name.strip_prefix("node") else {
            continue;
        };
        if suffix.is_empty() || !suffix.bytes().all(|byte| byte.is_ascii_digit()) {
            continue;
        }
        let node = suffix.parse::<i64>().map_err(|error| {
            HostIdentityError::new(format!(
                "unable to parse NUMA-node link {}: {error}",
                entry.path().display()
            ))
        })?;
        nodes.push(node);
    }
    nodes.sort_unstable();
    if nodes.len() > 1 {
        return Err(HostIdentityError::new(format!(
            "CPU has multiple NUMA-node links: {}",
            cpu_dir.display()
        )));
    }
    Ok(nodes.first().copied())
}

/// Collect CPU identity using explicit filesystem roots and platform facts.
///
/// The order and multiplicity of `cpu_ids` are retained. This function never
/// substitutes host-wide topology for a missing selected-CPU record.
pub fn collect_host_cpu_identity_with(
    cpu_ids: &[i64],
    sys_cpu_root: &Path,
    cpuinfo_path: &Path,
    system: &HostSystemIdentity,
) -> Result<Value, HostIdentityError> {
    let fields = cpuinfo_fields(cpuinfo_path)?;
    let model = fields
        .get("model name")
        .filter(|value| !value.is_empty())
        .or_else(|| fields.get("hardware").filter(|value| !value.is_empty()))
        .or_else(|| fields.get("processor").filter(|value| !value.is_empty()))
        .cloned()
        .or_else(|| (!system.processor.is_empty()).then(|| system.processor.clone()))
        .ok_or_else(|| HostIdentityError::new("host CPU model is unavailable"))?;
    let vendor = fields
        .get("vendor_id")
        .filter(|value| !value.is_empty())
        .or_else(|| {
            fields
                .get("cpu implementer")
                .filter(|value| !value.is_empty())
        })
        .cloned();

    let mut selected = Vec::with_capacity(cpu_ids.len());
    for &cpu_id in cpu_ids {
        let cpu_dir = sys_cpu_root.join(format!("cpu{cpu_id}"));
        let topology = cpu_dir.join("topology");
        let thread_siblings = siblings(&topology.join("thread_siblings_list"))?;
        if !thread_siblings.contains(&cpu_id) {
            return Err(HostIdentityError::new(format!(
                "CPU {cpu_id} is absent from its thread-sibling topology"
            )));
        }
        selected.push(json!({
            "logical_cpu": cpu_id,
            "package_id": integer(&topology.join("physical_package_id"), "physical package ID")?,
            "core_id": integer(&topology.join("core_id"), "physical core ID")?,
            "thread_siblings": thread_siblings,
            "numa_node": numa_node(&cpu_dir)?,
        }));
    }

    Ok(json!({
        "schema_version": HOST_SCHEMA_VERSION,
        "machine": system.machine,
        "kernel_release": system.kernel_release,
        "cpu_vendor": vendor,
        "cpu_model": model,
        "microcode": fields.get("microcode"),
        "host_logical_cpu_count": system.logical_cpu_count,
        "selected_cpus": selected,
    }))
}

/// Collect CPU identity from explicit sysfs/cpuinfo roots on the current host.
pub fn collect_host_cpu_identity_from(
    cpu_ids: &[i64],
    sys_cpu_root: &Path,
    cpuinfo_path: &Path,
) -> Result<Value, HostIdentityError> {
    let system = HostSystemIdentity::detect()?;
    collect_host_cpu_identity_with(cpu_ids, sys_cpu_root, cpuinfo_path, &system)
}

/// Collect CPU identity and selected-CPU topology from the current Linux host.
pub fn collect_host_cpu_identity(cpu_ids: &[i64]) -> Result<Value, HostIdentityError> {
    collect_host_cpu_identity_from(cpu_ids, Path::new(SYS_CPU_ROOT), Path::new(CPUINFO_PATH))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    static TEST_SEQUENCE: AtomicU64 = AtomicU64::new(0);

    struct TestDirectory(PathBuf);

    impl TestDirectory {
        fn new() -> Self {
            let sequence = TEST_SEQUENCE.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "tasktrace-run-host-test-{}-{sequence}",
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

    fn cpu(root: &Path, cpu_id: i64, package: i64, core: i64, siblings: &str) {
        let topology = root.join(format!("cpu{cpu_id}/topology"));
        fs::create_dir_all(&topology).unwrap();
        fs::write(topology.join("physical_package_id"), format!("{package}\n")).unwrap();
        fs::write(topology.join("core_id"), format!("{core}\n")).unwrap();
        fs::write(
            topology.join("thread_siblings_list"),
            format!("{siblings}\n"),
        )
        .unwrap();
    }

    fn system() -> HostSystemIdentity {
        HostSystemIdentity {
            machine: "x86_64".to_owned(),
            kernel_release: "test-kernel".to_owned(),
            processor: "fallback-processor".to_owned(),
            logical_cpu_count: Some(192),
        }
    }

    #[test]
    fn collects_selected_cpu_model_and_smt_topology() {
        let directory = TestDirectory::new();
        let cpuinfo = directory.join("cpuinfo");
        fs::write(
            &cpuinfo,
            "processor : 0\nvendor_id : GenuineTest\nmodel name : Test CPU 9000\nmicrocode : 0x123\n\n",
        )
        .unwrap();
        let sys_cpu = directory.join("sys-cpu");
        cpu(&sys_cpu, 20, 0, 10, "20,92");
        cpu(&sys_cpu, 21, 0, 11, "21,93");
        fs::create_dir(sys_cpu.join("cpu20/node0")).unwrap();
        fs::create_dir(sys_cpu.join("cpu21/node0")).unwrap();

        let result =
            collect_host_cpu_identity_with(&[20, 21], &sys_cpu, &cpuinfo, &system()).unwrap();
        assert_eq!(
            result,
            json!({
                "schema_version": "tasktrace-run-host.v1",
                "machine": "x86_64",
                "kernel_release": "test-kernel",
                "cpu_vendor": "GenuineTest",
                "cpu_model": "Test CPU 9000",
                "microcode": "0x123",
                "host_logical_cpu_count": 192,
                "selected_cpus": [
                    {
                        "logical_cpu": 20,
                        "package_id": 0,
                        "core_id": 10,
                        "thread_siblings": [20, 92],
                        "numa_node": 0,
                    },
                    {
                        "logical_cpu": 21,
                        "package_id": 0,
                        "core_id": 11,
                        "thread_siblings": [21, 93],
                        "numa_node": 0,
                    },
                ],
            })
        );
    }

    #[test]
    fn rejects_missing_or_inconsistent_selected_cpu_topology() {
        let directory = TestDirectory::new();
        let cpuinfo = directory.join("cpuinfo");
        fs::write(&cpuinfo, "processor : 0\nmodel name : Test CPU\n\n").unwrap();
        let missing =
            collect_host_cpu_identity_with(&[20], &directory.join("missing"), &cpuinfo, &system())
                .unwrap_err();
        assert!(missing.to_string().contains("thread siblings"));

        let sys_cpu = directory.join("sys-cpu");
        cpu(&sys_cpu, 20, 0, 10, "21,93");
        let inconsistent =
            collect_host_cpu_identity_with(&[20], &sys_cpu, &cpuinfo, &system()).unwrap_err();
        assert!(inconsistent
            .to_string()
            .contains("absent from its thread-sibling topology"));
    }

    #[test]
    fn cpuset_parser_sorts_deduplicates_and_rejects_invalid_components() {
        assert_eq!(parse_cpuset_cpu_ids("3,1-2,3").unwrap(), vec![1, 2, 3]);
        assert_eq!(parse_cpuset_cpu_ids(" 3 , 1 - 2 ").unwrap(), vec![1, 2, 3]);
        assert!(parse_cpuset_cpu_ids("4-2")
            .unwrap_err()
            .to_string()
            .contains("invalid CPU set range"));
        assert!(parse_cpuset_cpu_ids("1,,2")
            .unwrap_err()
            .to_string()
            .contains("invalid empty CPU set component"));
        assert!(parse_cpuset_cpu_ids("-1")
            .unwrap_err()
            .to_string()
            .contains("invalid CPU set range"));
    }

    #[test]
    fn rejects_multiple_numa_links_and_preserves_selected_cpu_order() {
        let directory = TestDirectory::new();
        let cpuinfo = directory.join("cpuinfo");
        fs::write(&cpuinfo, "model name : Test CPU\n\n").unwrap();
        let sys_cpu = directory.join("sys-cpu");
        cpu(&sys_cpu, 4, 0, 2, "4,8");
        fs::create_dir(sys_cpu.join("cpu4/node0")).unwrap();
        fs::create_dir(sys_cpu.join("cpu4/node1")).unwrap();
        let error =
            collect_host_cpu_identity_with(&[4], &sys_cpu, &cpuinfo, &system()).unwrap_err();
        assert!(error.to_string().contains("multiple NUMA-node links"));

        fs::remove_dir(sys_cpu.join("cpu4/node1")).unwrap();
        cpu(&sys_cpu, 8, 0, 2, "4,8");
        fs::create_dir(sys_cpu.join("cpu8/node0")).unwrap();
        let result =
            collect_host_cpu_identity_with(&[8, 4, 8], &sys_cpu, &cpuinfo, &system()).unwrap();
        assert_eq!(result["selected_cpus"][0]["logical_cpu"], 8);
        assert_eq!(result["selected_cpus"][1]["logical_cpu"], 4);
        assert_eq!(result["selected_cpus"][2]["logical_cpu"], 8);
    }

    #[test]
    fn uses_processor_fallback_but_rejects_unavailable_model() {
        let directory = TestDirectory::new();
        let cpuinfo = directory.join("cpuinfo");
        fs::write(&cpuinfo, "vendor_id : TestVendor\n\n").unwrap();
        let sys_cpu = directory.join("sys-cpu");
        cpu(&sys_cpu, 0, 0, 0, "0");
        let result = collect_host_cpu_identity_with(&[0], &sys_cpu, &cpuinfo, &system()).unwrap();
        assert_eq!(result["cpu_model"], "fallback-processor");

        let unavailable = HostSystemIdentity {
            processor: String::new(),
            ..system()
        };
        let error =
            collect_host_cpu_identity_with(&[0], &sys_cpu, &cpuinfo, &unavailable).unwrap_err();
        assert_eq!(error.to_string(), "host CPU model is unavailable");
    }

    #[test]
    fn first_cpuinfo_block_and_identity_precedence_match_python_contract() {
        let directory = TestDirectory::new();
        let cpuinfo = directory.join("cpuinfo");
        fs::write(
            &cpuinfo,
            "processor : first\r\nhardware : First Hardware\r\ncpu implementer : 0x41\r\n\r\nmodel name : Ignored Second CPU\r\n",
        )
        .unwrap();
        let sys_cpu = directory.join("sys-cpu");
        cpu(&sys_cpu, 0, 0, 0, "0");
        let result = collect_host_cpu_identity_with(&[0], &sys_cpu, &cpuinfo, &system()).unwrap();
        assert_eq!(result["cpu_model"], "First Hardware");
        assert_eq!(result["cpu_vendor"], "0x41");
        assert_eq!(result["microcode"], Value::Null);
    }
}
