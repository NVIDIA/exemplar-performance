// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in real-Docker gate for native cgroup-scoped hardware counters.
//!
//! This is deliberately ignored: the host must permit cgroup perf events and
//! provide a runner-owned CPU set plus a preloaded immutable image.
//!
//! ```text
//! TASKTRACE_RUST_DOCKER_TEST_IMAGE=sha256:<64-hex> \
//! TASKTRACE_RUST_DOCKER_CPUSET=16-19 \
//! cargo test --test rust_hardware_perf_docker_e2e -- --ignored --nocapture
//! ```

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::Path;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn immutable_digest(value: &str) -> bool {
    let digest = value.rsplit_once('@').map_or(value, |(_, digest)| digest);
    let Some(hex) = digest.strip_prefix("sha256:") else {
        return false;
    };
    hex.len() == 64
        && hex
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
}

fn file_sha256(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(fs::read(path).expect("read retained hardware artifact"))
    )
}

#[test]
#[ignore = "requires real Docker, cgroup v2 perf-event access, reserved CPUs, and a preloaded immutable image"]
fn cgroup_perf_hardware_deep_is_scope_exact_and_fails_closed() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image digest");
    assert!(
        immutable_digest(&image),
        "Docker test image must be immutable"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-reserved logical CPUs");
    let root = tempfile::tempdir().expect("create hardware Docker gate root");
    let output_dir = root.path().join("profile");
    let missing_python = root.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args([
            "profile",
            "container",
            "--image",
            &image,
            "--command",
            "python3 -c 'print(sum(i * i for i in range(10000000)))'",
            "--cpuset-cpus",
            &cpuset,
            "--profile-set",
            "hardware-deep",
            "--hardware-perf-event",
            "cycles",
            "--hardware-perf-event",
            "instructions",
            "--hardware-perf-minimum-running-ratio",
            "0.95",
            "--network-policy",
            "no-network",
            "--output-dir",
        ])
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native hardware perf Docker gate");
    let report_path = output_dir.join("profile_report.json");
    let retained = fs::read_to_string(report_path).ok();
    assert!(
        output.status.success(),
        "native hardware Docker gate failed with {:?}\nstdout:\n{}\nstderr:\n{}\nretained report:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
        retained.as_deref().unwrap_or("<none>"),
    );
    assert!(!missing_python.exists());
    let report: Value = serde_json::from_str(retained.as_deref().expect("retained report"))
        .expect("parse native hardware profile report");
    let hardware = &report["hardware_perf"];
    assert_eq!(report["profiling_valid"], true);
    assert_eq!(hardware["instrumentation_valid"], true);
    assert_eq!(hardware["all_requested_metrics_valid"], true);
    assert_eq!(hardware["hybrid_pmu_counts_merged"], true);
    assert_eq!(hardware["cpuset_cpus"], cpuset);
    assert_eq!(hardware["counters"].as_array().unwrap().len(), 2);
    assert!(hardware["counters"]
        .as_array()
        .unwrap()
        .iter()
        .all(|counter| {
            counter["valid"] == true
                && counter["value"].as_u64().is_some_and(|value| value > 0)
                && counter["observed_cgroup"] == hardware["perf_cgroup"]
        }));
    for action in ["attach_ready", "enable", "disable"] {
        let bounds = &hardware["control_bounds"][action];
        assert_eq!(bounds["clock"], "CLOCK_MONOTONIC");
        assert!(
            bounds["acknowledgement_monotonic_ns"].as_u64().unwrap()
                >= bounds["request_monotonic_ns"].as_u64().unwrap()
        );
    }
    assert!(
        hardware["definite_active_window_start_ns"]
            .as_u64()
            .unwrap()
            <= hardware["definite_active_window_end_ns"].as_u64().unwrap()
    );
    assert_eq!(
        report["evidence_bundle"]["resolved_profiler_ids"],
        serde_json::json!(["hardware-perf"])
    );
    assert!(!report["cpu"].is_null());

    let bundle_path = report["evidence_bundle"]["path"]
        .as_str()
        .map(Path::new)
        .expect("evidence path");
    let bundle: Value = serde_json::from_slice(&fs::read(bundle_path).expect("read evidence"))
        .expect("parse evidence");
    assert_eq!(bundle["profilers"].as_array().unwrap().len(), 1);
    assert_eq!(bundle["profilers"][0]["profiler_id"], "hardware-perf");
    assert_eq!(
        bundle["extensions"]["profiler_statuses"]["cgroup-cpu"],
        "not-selected"
    );
    assert_eq!(bundle["artifacts"].as_array().unwrap().len(), 3);
    for artifact in bundle["artifacts"].as_array().unwrap() {
        let relative = artifact["relative_path"].as_str().unwrap();
        let path = output_dir.join(relative);
        assert_eq!(artifact["sha256"], file_sha256(&path));
        assert_eq!(artifact["size_bytes"], path.metadata().unwrap().len());
    }
    let container_name = bundle["extensions"]["runtime_report"]["container_name"]
        .as_str()
        .expect("retained container identity");
    let inspect = Command::new("docker")
        .args(["inspect", container_name])
        .output()
        .expect("verify Docker cleanup");
    assert!(
        !inspect.status.success(),
        "profile container survived cleanup: {container_name}"
    );
}
