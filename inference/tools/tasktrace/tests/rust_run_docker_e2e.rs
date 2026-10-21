// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in real-Docker gate for the native v1/v2/v3 run orchestrator.
//!
//! All three schemas inject the content-bound Rust controller with the retired
//! Python fallback disabled. The gate proves both host orchestration and the
//! measured replay command use the native binary.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn command_output(command: &mut Command) -> Output {
    command
        .output()
        .expect("run native run-orchestrator E2E command")
}

fn assert_success(output: &Output, context: &str) {
    assert!(
        output.status.success(),
        "{context} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn assert_run_success(result: &Output, context: &str, output: &Path) {
    if result.status.success() {
        return;
    }
    let rows = fs::read_to_string(output.join("results.csv"))
        .unwrap_or_else(|error| format!("<unavailable: {error}>"));
    let summary = fs::read_to_string(output.join("summary.json"))
        .unwrap_or_else(|error| format!("<unavailable: {error}>"));
    fn reports(path: &Path, collected: &mut String) {
        let Ok(entries) = fs::read_dir(path) else {
            return;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                reports(&path, collected);
            } else if path.file_name().and_then(|name| name.to_str())
                == Some("pipeline_report.json")
            {
                collected.push_str(&format!(
                    "\n{}:\n{}",
                    path.display(),
                    fs::read_to_string(&path).unwrap_or_else(|error| error.to_string())
                ));
            }
        }
    }
    let mut pipeline_reports = String::new();
    reports(output, &mut pipeline_reports);
    panic!(
        "{context} failed with {:?}\nstdout:\n{}\nstderr:\n{}\nresults.csv:\n{rows}\nsummary.json:\n{summary}\npipeline reports:{pipeline_reports}",
        result.status.code(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
}

fn image_id(image: &str) -> String {
    let output = command_output(
        Command::new("docker").args(["image", "inspect", "--format", "{{.Id}}", image]),
    );
    assert_success(&output, "image inspection");
    String::from_utf8(output.stdout).unwrap().trim().to_owned()
}

fn cpus(value: &str) -> Vec<u32> {
    let mut result = Vec::new();
    for component in value.split(',') {
        if let Some((first, last)) = component.split_once('-') {
            result.extend(first.parse::<u32>().unwrap()..=last.parse::<u32>().unwrap());
        } else {
            result.push(component.parse().unwrap());
        }
    }
    result.sort_unstable();
    result.dedup();
    result
}

fn write_atif(path: &Path, task: &str, workload_sec: f64) -> String {
    let command = format!(
        "python3 -c \"import time; end=time.monotonic()+{workload_sec}; exec('while time.monotonic() < end: pass')\""
    );
    let payload = json!({
        "schema_version": "ATIF-v1.5",
        "session_id": task,
        "agent": {"name": "rust-run-docker-e2e"},
        "steps": [{
            "step_id": 1,
            "timestamp": "2026-07-06T00:00:00+00:00",
            "tool_calls": [{
                "function_name": "exec_command",
                "arguments": {"cmd": command},
            }],
            "observation": {"content": "Process exited with code 0"},
        }],
    });
    let mut bytes = serde_json::to_vec(&payload).unwrap();
    bytes.push(b'\n');
    fs::write(path, &bytes).unwrap();
    format!("{:x}", Sha256::digest(&bytes))
}

fn task(root: &Path, name: &str, image: &str, expected_image: &str, seconds: f64) -> Value {
    let path = root.join(format!("{name}.json"));
    let digest = write_atif(&path, name, seconds);
    json!({
        "id": name,
        "atif": path.file_name().unwrap().to_string_lossy(),
        "image": image,
        "expected_source_sha256": digest,
        "expected_image_id": expected_image,
        "action_timeout_sec": 20,
    })
}

fn run_manifest(manifest: &Path) -> Output {
    command_output(
        Command::new("timeout")
            .args(["--signal=INT", "--kill-after=10s", "180s"])
            .arg(binary())
            .arg("run")
            .arg(manifest)
            .current_dir(repository())
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                manifest
                    .parent()
                    .expect("run manifest has a parent")
                    .join("python-fallback-must-not-run"),
            )
            .env_remove("PYTHONPATH")
            .env("PYTHONDONTWRITEBYTECODE", "1"),
    )
}

fn run_environment(command: &mut Command) {
    command
        .current_dir(repository())
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            repository().join("python-fallback-must-not-run"),
        )
        .env_remove("PYTHONPATH")
        .env("PYTHONDONTWRITEBYTECODE", "1");
}

fn object(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

fn assert_profile_evidence(report: &Value, expected_image: &str, expected_cpuset: &str) -> String {
    assert_eq!(report["status"], "valid");
    assert_eq!(report["profiling_valid"], true);
    assert_eq!(report["image_identity"]["id"], expected_image);
    assert_eq!(
        report["source_trajectory_sha256"].as_str().unwrap().len(),
        64
    );
    assert_eq!(report["execution_plan_digest"].as_str().unwrap().len(), 64);
    assert_eq!(
        report["canonical_clock_reference"]["clock_id"],
        "CLOCK_MONOTONIC"
    );
    let cpu = &report["container_cpu"];
    assert_eq!(cpu["valid_for_cpu_performance"], true);
    assert!(cpu["usage_sec"].as_f64().unwrap() > 0.0);
    assert!(cpu["average_logical_cores"].as_f64().unwrap() > 0.0);
    assert_eq!(cpu["allocated_cpu_source"], "cpuset-cpus");
    assert_eq!(
        cpu["allocated_logical_cpus"].as_f64().unwrap() as usize,
        cpus(expected_cpuset).len()
    );
    assert_eq!(
        cpus(
            report["execution_plan"]["extensions"]["cpuset_cpus"]
                .as_str()
                .unwrap()
        ),
        cpus(expected_cpuset)
    );
    let placement = &report["scope"]["identity"]["cpu_placement"];
    assert_eq!(placement["validation_status"], "passed");
    for field in [
        "requested_cpuset_cpus",
        "docker_host_config_cpuset_cpus",
        "container_init_cpus_allowed_list",
        "cgroup_cpuset_cpus_effective",
    ] {
        assert_eq!(
            cpus(placement[field].as_str().unwrap()),
            cpus(expected_cpuset),
            "placement field {field}"
        );
    }
    assert_eq!(report["controller_runtime"], "rust-host-mount");
    let controller_digest = report["controller_identity"]["sha256"]
        .as_str()
        .expect("profile retains a Rust controller digest");
    assert_eq!(controller_digest.len(), 64);
    assert_eq!(
        report["execution_plan"]["source_identity"]["rust_controller_binary_sha256"],
        report["controller_identity"]["sha256"]
    );
    controller_digest.to_owned()
}

fn assert_native_summary_controller(summary: &Value, expected_digest: &str) {
    assert_eq!(summary["controller_runtime"], "rust-host-mount");
    assert_eq!(summary["controller_runtime_content_bound"], true);
    assert!(summary["controller_runtime_limitation"].is_null());
    assert_eq!(summary["controller_binary_sha256"], expected_digest);
}

fn scoped_containers(output: &Path) -> Vec<String> {
    let listed = command_output(Command::new("docker").args([
        "ps",
        "--all",
        "--filter",
        "name=tasktrace-profile-",
        "--format",
        "{{.ID}}",
    ]));
    assert_success(&listed, "container listing");
    let Ok(output) = fs::canonicalize(output) else {
        return Vec::new();
    };
    let mut found = Vec::new();
    for id in String::from_utf8_lossy(&listed.stdout).lines() {
        let inspected = command_output(Command::new("docker").args(["inspect", id]));
        if !inspected.status.success() {
            continue;
        }
        let value: Value = serde_json::from_slice(&inspected.stdout).unwrap();
        let scoped = value[0]["Mounts"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|mount| mount["Source"].as_str())
            .filter_map(|source| fs::canonicalize(source).ok())
            .any(|source| source == output || source.starts_with(&output));
        if scoped {
            found.push(id.to_owned());
        }
    }
    found
}

fn exercise_native_v2_signal_cleanup(signal: libc::c_int, signal_name: &str) {
    let image =
        env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE").expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE");
    let available =
        cpus(&env::var("TASKTRACE_RUST_DOCKER_CPUSET").expect("set TASKTRACE_RUST_DOCKER_CPUSET"));
    assert!(available.len() >= 2);
    let expected_image = image_id(&image);
    let root = TempDir::new().unwrap();
    let tasks: Vec<_> = (0..4)
        .map(|index| {
            task(
                root.path(),
                &format!("interrupt-{index}"),
                &image,
                &expected_image,
                30.0,
            )
        })
        .collect();
    let manifest = root.path().join("interrupt.yaml");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "tasktrace-run.v2",
            "output": "results",
            "cpus": format!("{},{}", available[0], available[1]),
            "concurrency": 2,
            "repetitions": 1,
            "tasks": tasks,
        }))
        .unwrap(),
    )
    .unwrap();
    let output = root.path().join("results");
    let mut command = Command::new(binary());
    command
        .arg("run")
        .arg(&manifest)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    run_environment(&mut command);
    let mut child = command.spawn().unwrap();
    let startup_deadline = Instant::now() + Duration::from_secs(45);
    while Instant::now() < startup_deadline {
        if scoped_containers(&output).len() == 2 {
            break;
        }
        if let Some(status) = child.try_wait().unwrap() {
            panic!("native interrupted run exited before cancellation: {status}");
        }
        thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(scoped_containers(&output).len(), 2);
    let signal_result = unsafe { libc::kill(child.id() as i32, signal) };
    assert_eq!(signal_result, 0);
    let cleanup_deadline = Instant::now() + Duration::from_secs(35);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= cleanup_deadline {
            let _ = child.kill();
            panic!("native run did not complete cleanup within 35 seconds of {signal_name}");
        }
        thread::sleep(Duration::from_millis(50));
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert_eq!(
        status.code(),
        Some(130),
        "stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let summary = object(&output.join("summary.json"));
    assert_eq!(summary["overall"]["interrupted"], true);
    assert_eq!(summary["overall"]["valid_for_throughput"], false);
    assert_eq!(summary["overall"]["attempt_count"], 4);
    let csv = fs::read_to_string(output.join("results.csv")).unwrap();
    assert_eq!(csv.lines().count(), 5, "{csv}");
    let container_deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < container_deadline && !scoped_containers(&output).is_empty() {
        thread::sleep(Duration::from_millis(100));
    }
    assert!(scoped_containers(&output).is_empty());
}

#[test]
#[ignore = "requires Docker, cgroup v2, a preloaded immutable image, and two runner-owned CPUs"]
fn native_v2_sigint_cooperatively_cleans_profiles_and_retains_rows() {
    exercise_native_v2_signal_cleanup(libc::SIGINT, "SIGINT");
}

#[test]
#[ignore = "requires Docker, cgroup v2, a preloaded immutable image, and two runner-owned CPUs"]
fn native_v2_sigterm_cooperatively_cleans_profiles_and_retains_rows() {
    exercise_native_v2_signal_cleanup(libc::SIGTERM, "SIGTERM");
}

#[test]
#[ignore = "requires Docker, cgroup v2, a preloaded immutable image, and three runner-owned CPUs"]
fn native_v3_sigint_retains_undispatched_wave_and_restores_affinity() {
    let image =
        env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE").expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE");
    let available =
        cpus(&env::var("TASKTRACE_RUST_DOCKER_CPUSET").expect("set TASKTRACE_RUST_DOCKER_CPUSET"));
    assert!(available.len() >= 3);
    let expected_image = image_id(&image);
    let root = TempDir::new().unwrap();
    let tasks = vec![
        task(root.path(), "wave-a", &image, &expected_image, 30.0),
        task(root.path(), "wave-b", &image, &expected_image, 30.0),
    ];
    let manifest = root.path().join("interrupt-v3.yaml");
    fs::write(
        &manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "tasktrace-run.v3",
            "output": "results",
            "defaults": {
                "pacing": "unpaced",
                "network_policy": "no-network",
                "profile_sets": ["cpu-average"],
                "controller_runtime": "rust-host-mount",
                "controller_cpus": available[0].to_string(),
                "action_timeout_sec": 60,
            },
            "tasks": tasks,
            "waves": [
                {
                    "id": "active-wave",
                    "scored": true,
                    "metadata": {"repeat": 1},
                    "launches": [
                        {"task": "wave-a", "cpus": available[1].to_string()},
                        {"task": "wave-b", "cpus": available[2].to_string()},
                    ],
                },
                {
                    "id": "undispatched-wave",
                    "scored": true,
                    "metadata": {"repeat": 2},
                    "launches": [
                        {"task": "wave-a", "cpus": available[1].to_string()},
                        {"task": "wave-b", "cpus": available[2].to_string()},
                    ],
                },
            ],
        }))
        .unwrap(),
    )
    .unwrap();
    let output = root.path().join("results");
    let mut command = Command::new(binary());
    command
        .arg("run")
        .arg(&manifest)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    run_environment(&mut command);
    let mut child = command.spawn().unwrap();
    let startup_deadline = Instant::now() + Duration::from_secs(45);
    while Instant::now() < startup_deadline {
        if scoped_containers(&output).len() == 2 {
            break;
        }
        if let Some(status) = child.try_wait().unwrap() {
            let mut stdout = String::new();
            let mut stderr = String::new();
            child
                .stdout
                .take()
                .unwrap()
                .read_to_string(&mut stdout)
                .unwrap();
            child
                .stderr
                .take()
                .unwrap()
                .read_to_string(&mut stderr)
                .unwrap();
            panic!(
                "native v3 interrupted run exited before cancellation: {status}\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
        }
        thread::sleep(Duration::from_millis(100));
    }
    assert_eq!(scoped_containers(&output).len(), 2);
    assert_eq!(unsafe { libc::kill(child.id() as i32, libc::SIGINT) }, 0);
    let cleanup_deadline = Instant::now() + Duration::from_secs(35);
    let status = loop {
        if let Some(status) = child.try_wait().unwrap() {
            break status;
        }
        if Instant::now() >= cleanup_deadline {
            let _ = child.kill();
            panic!("native v3 run did not complete cleanup within 35 seconds of SIGINT");
        }
        thread::sleep(Duration::from_millis(50));
    };
    let mut stdout = String::new();
    let mut stderr = String::new();
    child
        .stdout
        .take()
        .unwrap()
        .read_to_string(&mut stdout)
        .unwrap();
    child
        .stderr
        .take()
        .unwrap()
        .read_to_string(&mut stderr)
        .unwrap();
    assert_eq!(
        status.code(),
        Some(130),
        "stdout:\n{stdout}\nstderr:\n{stderr}"
    );
    let summary = object(&output.join("summary.json"));
    assert_eq!(summary["overall"]["interrupted"], true);
    assert_eq!(summary["overall"]["attempt_count"], 4);
    assert_eq!(summary["overall"]["valid_for_analysis"], false);
    assert_eq!(
        summary["controller_affinity"]["caller_affinity_restored"],
        true
    );
    let csv = fs::read_to_string(output.join("results.csv")).unwrap();
    assert_eq!(csv.lines().count(), 5, "{csv}");
    assert_eq!(
        csv.lines()
            .skip(1)
            .filter(|line| line.contains(",undispatched-wave,"))
            .count(),
        2,
        "{csv}"
    );
    assert!(
        csv.contains("run interrupted before attempt dispatch"),
        "{csv}"
    );
    let container_deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < container_deadline && !scoped_containers(&output).is_empty() {
        thread::sleep(Duration::from_millis(100));
    }
    assert!(scoped_containers(&output).is_empty());
}

fn assert_clock_and_cleanup(output: &Path, summary: &Value) {
    assert_eq!(
        summary["canonical_clock_reference"]["clock_id"],
        "CLOCK_MONOTONIC"
    );
    let (start, end) = match summary["schema_version"].as_str().unwrap() {
        "tasktrace-run-summary.v1" => {
            assert!(summary["clock_resolution_ns"].as_u64().unwrap() >= 1);
            (
                summary["outer_window_start_ns"].as_u64().unwrap(),
                summary["outer_window_end_ns"].as_u64().unwrap(),
            )
        }
        _ => {
            assert!(summary["overall"]["clock_resolution_ns"].as_u64().unwrap() >= 1);
            (
                summary["overall"]["outer_window_start_ns"]
                    .as_u64()
                    .unwrap(),
                summary["overall"]["outer_window_end_ns"].as_u64().unwrap(),
            )
        }
    };
    assert!(end > start);
    assert!(scoped_containers(output).is_empty());
}

#[test]
#[ignore = "requires Docker, cgroup v2, a preloaded immutable image, and three runner-owned CPUs"]
fn native_run_v1_v2_v3_preserves_measurement_and_wave_contracts() {
    let image =
        env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE").expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE");
    let available =
        cpus(&env::var("TASKTRACE_RUST_DOCKER_CPUSET").expect("set TASKTRACE_RUST_DOCKER_CPUSET"));
    assert!(available.len() >= 3);
    let expected_image = image_id(&image);
    let root = TempDir::new().unwrap();

    let v1_root = root.path().join("v1");
    fs::create_dir(&v1_root).unwrap();
    let v1_tasks = vec![task(&v1_root, "v1-task", &image, &expected_image, 0.1)];
    let v1_manifest = v1_root.join("run.yaml");
    fs::write(
        &v1_manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "tasktrace-run.v1",
            "output": "results",
            "cpus": available[1].to_string(),
            "repetitions": 1,
            "tasks": v1_tasks,
        }))
        .unwrap(),
    )
    .unwrap();
    let v1_output = v1_root.join("results");
    let result = run_manifest(&v1_manifest);
    assert_run_success(&result, "native v1 run", &v1_output);
    let v1_summary = object(&v1_output.join("summary.json"));
    assert_eq!(v1_summary["overall"]["valid_duration_count"], 1);
    let v1_controller_digest = assert_profile_evidence(
        &object(&v1_output.join("tasks/v1-task/rep-0001/profile/pipeline_report.json")),
        &expected_image,
        &available[1].to_string(),
    );
    assert_native_summary_controller(&v1_summary, &v1_controller_digest);
    assert_clock_and_cleanup(&v1_output, &v1_summary);

    let v2_root = root.path().join("v2");
    fs::create_dir(&v2_root).unwrap();
    let v2_tasks = vec![
        task(&v2_root, "v2-a", &image, &expected_image, 0.35),
        task(&v2_root, "v2-b", &image, &expected_image, 0.35),
    ];
    let v2_manifest = v2_root.join("run.yaml");
    fs::write(
        &v2_manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "tasktrace-run.v2",
            "output": "results",
            "cpus": format!("{},{}", available[1], available[2]),
            "concurrency": 2,
            "repetitions": 1,
            "tasks": v2_tasks,
        }))
        .unwrap(),
    )
    .unwrap();
    let v2_output = v2_root.join("results");
    let result = run_manifest(&v2_manifest);
    assert_run_success(&result, "native v2 run", &v2_output);
    let v2_summary = object(&v2_output.join("summary.json"));
    assert_eq!(v2_summary["overall"]["valid_for_throughput"], true);
    assert_eq!(v2_summary["overall"]["achieved_peak_measured_attempts"], 2);
    assert!(
        v2_summary["overall"]["total_cgroup_cpu_sec"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert!(
        v2_summary["overall"]["pool_utilization_percent"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    let v2_cpuset = format!("{},{}", available[1], available[2]);
    let mut v2_controller_digests = Vec::new();
    for name in ["v2-a", "v2-b"] {
        v2_controller_digests.push(assert_profile_evidence(
            &object(&v2_output.join(format!(
                "tasks/{name}/rep-0001/profile/pipeline_report.json"
            ))),
            &expected_image,
            &v2_cpuset,
        ));
    }
    assert!(v2_controller_digests
        .iter()
        .all(|digest| digest == &v2_controller_digests[0]));
    assert_native_summary_controller(&v2_summary, &v2_controller_digests[0]);
    assert_clock_and_cleanup(&v2_output, &v2_summary);

    let v3_root = root.path().join("v3");
    fs::create_dir(&v3_root).unwrap();
    let v3_tasks = vec![
        task(&v3_root, "v3-a", &image, &expected_image, 0.25),
        task(&v3_root, "v3-b", &image, &expected_image, 0.25),
    ];
    let v3_manifest = v3_root.join("run.yaml");
    fs::write(
        &v3_manifest,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "tasktrace-run.v3",
            "output": "results",
            "defaults": {
                "pacing": "unpaced",
                "network_policy": "no-network",
                "profile_sets": ["cpu-average"],
                "controller_runtime": "rust-host-mount",
                "controller_cpus": available[0].to_string(),
                "worker_cpus": format!("{},{}", available[1], available[2]),
                "placement": "round-robin-single-cpu",
                "action_timeout_sec": 20,
            },
            "tasks": v3_tasks,
            "waves": [
                {"id": "r1-n1", "allocation": 1, "repeat": 1},
                {"id": "r1-n2", "allocation": 2, "repeat": 1},
            ],
        }))
        .unwrap(),
    )
    .unwrap();
    let v3_output = v3_root.join("results");
    let result = run_manifest(&v3_manifest);
    assert_run_success(&result, "native v3 run", &v3_output);
    let v3_summary = object(&v3_output.join("summary.json"));
    assert_eq!(v3_summary["overall"]["valid_for_analysis"], true);
    assert_eq!(
        v3_summary["waves"][0]["overall"]["valid_for_throughput"],
        true
    );
    assert_eq!(
        v3_summary["waves"][1]["overall"]["valid_for_throughput"],
        true
    );
    assert_eq!(
        v3_summary["scheduling"]["authoring_mode"],
        "compact-worker-pool"
    );
    assert!(
        v3_summary["waves"][0]["overall"]["outer_window_end_ns"]
            .as_u64()
            .unwrap()
            <= v3_summary["waves"][1]["overall"]["outer_window_start_ns"]
                .as_u64()
                .unwrap(),
        "the second wave must start only after the first wave's host window closes"
    );
    for wave in v3_summary["waves"].as_array().unwrap() {
        assert!(wave["overall"]["total_cgroup_cpu_sec"].as_f64().unwrap() > 0.0);
        assert!(
            wave["overall"]["pool_utilization_percent"]
                .as_f64()
                .unwrap()
                > 0.0
        );
    }
    assert!(v3_output.join("resolved-run.yaml").is_file());
    let resolved: serde_yaml::Value =
        serde_yaml::from_slice(&fs::read(v3_output.join("resolved-run.yaml")).unwrap()).unwrap();
    assert!(resolved["defaults"].get("worker_cpus").is_none());
    assert!(resolved["defaults"].get("placement").is_none());
    let expected_placements = [
        (1, "r1-n1", 1, "v3-a", available[1]),
        (1, "r1-n1", 2, "v3-b", available[1]),
        (2, "r1-n2", 1, "v3-a", available[1]),
        (2, "r1-n2", 2, "v3-b", available[2]),
    ];
    let mut v3_controller_digests = Vec::new();
    for (wave, wave_id, launch, name, cpu) in expected_placements {
        let report = object(&v3_output.join(format!(
            "waves/{wave:04}-{wave_id}/launch-{launch:04}-{name}/profile/pipeline_report.json"
        )));
        v3_controller_digests.push(assert_profile_evidence(
            &report,
            &expected_image,
            &cpu.to_string(),
        ));
    }
    assert!(v3_controller_digests
        .iter()
        .all(|digest| digest == &v3_controller_digests[0]));
    assert_native_summary_controller(&v3_summary, &v3_controller_digests[0]);
    let csv = fs::read_to_string(v3_output.join("results.csv")).unwrap();
    let mut completion_indexes: Vec<usize> = csv
        .lines()
        .skip(1)
        .map(|line| line.split(',').nth(1).unwrap().parse().unwrap())
        .collect();
    completion_indexes.sort_unstable();
    assert_eq!(completion_indexes, vec![1, 2, 3, 4]);
    assert_clock_and_cleanup(&v3_output, &v3_summary);
}
