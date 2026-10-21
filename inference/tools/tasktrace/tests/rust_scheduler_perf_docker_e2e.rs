// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real Docker gate for native cgroup-scoped perf scheduler concurrency.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
}

fn immutable_image(value: &str) -> bool {
    let digest = value.rsplit_once('@').map_or(value, |(_, digest)| digest);
    digest.strip_prefix("sha256:").is_some_and(|hex| {
        hex.len() == 64
            && hex
                .bytes()
                .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    })
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

#[test]
#[ignore = "requires a privileged cgroup-v2 Docker host, permissive perf_event policy, immutable image, and an exclusive cpuset with at least two logical CPUs"]
fn native_perf_scheduler_capture_is_complete_reconciled_and_typed() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to an immutable image ID or repo digest");
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to an exclusive multi-CPU allocation");
    assert!(
        immutable_image(&image),
        "Docker test image must be immutable"
    );
    assert!(
        cpuset.contains(',')
            || cpuset
                .split_once('-')
                .and_then(|(start, end)| Some((
                    start.parse::<u32>().ok()?,
                    end.parse::<u32>().ok()?
                )))
                .is_some_and(|(start, end)| end > start),
        "scheduler Docker gate requires at least two logical CPUs"
    );

    let root = tempfile::tempdir().unwrap();
    let output_dir = root.path().join("cpu-peak");
    let missing_python = root.path().join("python-fallback-must-not-run");
    let workload = "python3 -c 'import os,time; end=time.monotonic()+1.0; pid=os.fork(); x=0; exec(\"while time.monotonic() < end: x += 1\\nif pid: os.waitpid(pid, 0)\\nelse: os._exit(0)\")'";
    let output = Command::new(binary())
        .args(["profile", "container", "--image", &image])
        .args(["--cpuset-cpus", &cpuset])
        .args(["--network-policy", "no-network"])
        .args(["--command", workload])
        .args(["--profile-set", "cpu-peak"])
        .args(["--scheduler-backend", "perf"])
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native scheduler Docker profile");
    assert!(
        output.status.success(),
        "scheduler Docker gate failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(!missing_python.exists());

    let report = read_json(&output_dir.join("profile_report.json"));
    assert_eq!(report["profiling_valid"], true);
    let scheduler = &report["scheduler_concurrency"];
    assert_eq!(scheduler["backend"], "perf");
    assert_eq!(scheduler["perf_clockid"], "mono");
    assert_eq!(scheduler["trace_complete"], true);
    assert_eq!(
        scheduler["cgroup_validation"]["valid_for_concurrency_reporting"],
        true
    );
    assert!(scheduler["events_observed"].as_u64().unwrap() > 0);
    assert!(
        scheduler["peak_simultaneous_logical_cores"]
            .as_u64()
            .unwrap()
            >= 2
    );

    let mut artifact_ids = Vec::new();
    for artifact in scheduler["artifacts"].as_array().unwrap() {
        artifact_ids.push(artifact["artifact_id"].as_str().unwrap().to_owned());
        let path = artifact["path"]
            .as_str()
            .map(PathBuf::from)
            .unwrap_or_else(|| output_dir.join(artifact["relative_path"].as_str().unwrap()));
        let bytes = fs::read(path).unwrap();
        assert_eq!(artifact["size_bytes"].as_u64(), Some(bytes.len() as u64));
        assert_eq!(
            artifact["sha256"].as_str(),
            Some(format!("{:x}", Sha256::digest(&bytes)).as_str())
        );
    }
    artifact_ids.sort();
    assert_eq!(
        artifact_ids,
        [
            "scheduler-events",
            "scheduler-perf-data",
            "scheduler-perf-decoded",
            "scheduler-perf-stderr",
        ]
    );

    let evidence = read_json(&output_dir.join("evidence_bundle.json"));
    assert!(evidence["profilers"]
        .as_array()
        .unwrap()
        .iter()
        .any(|profiler| {
            profiler["profiler_id"] == "scheduler-concurrency" && profiler["status"] == "valid"
        }));
    for (stream_id, shape) in [
        ("cpu-peak-simultaneous", "scalar"),
        ("cpu-scheduled-runtime", "scalar"),
        ("scheduler-concurrency", "interval"),
        ("scheduler-per-core", "interval"),
    ] {
        assert!(evidence["streams"]
            .as_array()
            .unwrap()
            .iter()
            .any(|stream| {
                stream["stream_id"] == stream_id
                    && stream["shape"] == shape
                    && stream["status"] == "valid"
            }));
    }
}
