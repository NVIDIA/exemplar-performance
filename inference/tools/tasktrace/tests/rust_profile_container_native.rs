// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-binary gates for native disposable-container resource profiling.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
}

fn disabled_fallback() -> PathBuf {
    PathBuf::from("/tasktrace-test/python-fallback-must-not-run")
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

fn successful_output(command: &mut Command, description: &str) -> Output {
    let output = command
        .output()
        .unwrap_or_else(|error| panic!("unable to {description}: {error}"));
    assert!(
        output.status.success(),
        "{description} failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn local_image_id(image: &str) -> String {
    let output = successful_output(
        Command::new("docker").args(["image", "inspect", "--format", "{{.Id}}", image]),
        "inspect the immutable Docker image",
    );
    String::from_utf8(output.stdout)
        .expect("Docker image ID is UTF-8")
        .trim()
        .to_owned()
}

fn assert_bundle_artifacts(output_dir: &Path, evidence: &Value) {
    let mut ids = BTreeSet::new();
    for artifact in evidence["artifacts"]
        .as_array()
        .expect("evidence artifacts")
    {
        let id = artifact["artifact_id"].as_str().expect("artifact ID");
        assert!(ids.insert(id.to_owned()), "duplicate artifact ID {id}");
        let relative = artifact["relative_path"]
            .as_str()
            .expect("artifact relative path");
        let bytes = fs::read(output_dir.join(relative)).expect("read evidence artifact");
        assert_eq!(artifact["size_bytes"].as_u64(), Some(bytes.len() as u64));
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(artifact["sha256"].as_str(), Some(digest.as_str()));
    }
    assert!(ids.contains("cgroup-resource-timeseries"));
    assert!(ids.contains("cgroup-memory-source-snapshots"));
}

fn scoped_profile_containers(output_dir: &Path) -> Vec<String> {
    let listed = successful_output(
        Command::new("docker").args([
            "ps",
            "--all",
            "--filter",
            "name=tasktrace-profile-",
            "--format",
            "{{.ID}}",
        ]),
        "list profiler containers",
    );
    let expected_source = output_dir
        .canonicalize()
        .expect("canonicalize profile output directory");
    let mut matched = Vec::new();
    for id in String::from_utf8_lossy(&listed.stdout)
        .lines()
        .filter(|line| !line.is_empty())
    {
        let inspected = Command::new("docker")
            .arg("inspect")
            .arg(id)
            .output()
            .unwrap();
        if !inspected.status.success() {
            continue;
        }
        let payload: Value =
            serde_json::from_slice(&inspected.stdout).expect("parse container inspection");
        let scoped = payload
            .as_array()
            .and_then(|rows| rows.first())
            .and_then(|row| row["Mounts"].as_array())
            .into_iter()
            .flatten()
            .any(|mount| {
                mount["Destination"].as_str() == Some("/tasktrace-profile-output")
                    && mount["Source"]
                        .as_str()
                        .map(PathBuf::from)
                        .and_then(|source| source.canonicalize().ok())
                        .as_deref()
                        == Some(expected_source.as_path())
            });
        if scoped {
            matched.push(id.to_owned());
        }
    }
    matched
}

fn docker_inputs() -> (OsString, OsString) {
    (
        env::var_os("TASKTRACE_RUST_DOCKER_TEST_IMAGE").expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE"),
        env::var_os("TASKTRACE_RUST_DOCKER_CPUSET").expect("set TASKTRACE_RUST_DOCKER_CPUSET"),
    )
}

fn run_profile(output: &Path, extra: &[&str]) -> Output {
    let (image, cpuset) = docker_inputs();
    let mut command = Command::new(binary());
    command
        .args(["profile", "container", "--image"])
        .arg(image)
        .arg("--cpuset-cpus")
        .arg(cpuset)
        .args(["--network-policy", "no-network"])
        .arg("--output-dir")
        .arg(output)
        .args(extra)
        .env("TASKTRACE_LEGACY_PYTHON", disabled_fallback());
    command.output().expect("run native public profile")
}

fn report(output: &Path) -> Value {
    serde_json::from_slice(&fs::read(output.join("profile_report.json")).unwrap()).unwrap()
}

#[test]
fn public_help_does_not_start_python_fallback() {
    let output = Command::new(binary())
        .args(["profile", "container", "--help"])
        .env("TASKTRACE_LEGACY_PYTHON", disabled_fallback())
        .output()
        .unwrap();
    assert!(output.status.success());
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.contains("native measurement modes:"));
    assert!(stdout.contains("--profile-set cpu-average"));
    assert!(stdout.contains("--memory-profile"));
    assert!(stdout.contains("native existing-Docker-container diagnostic mode"));
}

#[test]
#[ignore = "requires a local cgroup-v2 Docker daemon and explicit image/cpuset"]
fn disposable_profile_keeps_publication_read_only_and_exchange_writable() {
    let (image, cpuset) = docker_inputs();
    let root = tempfile::tempdir().unwrap();
    let output_dir = root.path().join("mount-boundary");
    let child = Command::new(binary())
        .args(["profile", "container", "--image"])
        .arg(image)
        .arg("--cpuset-cpus")
        .arg(cpuset)
        .args(["--network-policy", "no-network"])
        .arg("--output-dir")
        .arg(&output_dir)
        .args(["--command", "sleep 5", "--profile-set", "cpu-average"])
        .env("TASKTRACE_LEGACY_PYTHON", disabled_fallback())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("start native disposable profile");

    let deadline = Instant::now() + Duration::from_secs(10);
    let inspect = loop {
        let expected = output_dir.canonicalize().ok();
        let listed = Command::new("docker")
            .args([
                "ps",
                "--filter",
                "name=tasktrace-profile-",
                "--format",
                "{{.ID}}",
            ])
            .output()
            .expect("list profiler containers");
        assert!(listed.status.success());
        let mut matched = None;
        for id in String::from_utf8_lossy(&listed.stdout)
            .lines()
            .filter(|line| !line.is_empty())
        {
            let inspected = Command::new("docker")
                .args(["inspect", id])
                .output()
                .expect("inspect profiler container");
            if !inspected.status.success() {
                continue;
            }
            let payload: Value = serde_json::from_slice(&inspected.stdout).unwrap();
            let Some(row) = payload.as_array().and_then(|rows| rows.first()) else {
                continue;
            };
            let scoped = row["Mounts"].as_array().into_iter().flatten().any(|mount| {
                mount["Destination"] == "/tasktrace-profile-output"
                    && mount["Source"]
                        .as_str()
                        .and_then(|source| Path::new(source).canonicalize().ok())
                        == expected
            });
            if scoped {
                matched = Some(row.clone());
                break;
            }
        }
        if let Some(inspect) = matched {
            break inspect;
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for output-scoped profiler container"
        );
        thread::sleep(Duration::from_millis(25));
    };

    let mounts = inspect["Mounts"].as_array().unwrap();
    let publication = mounts
        .iter()
        .find(|mount| mount["Destination"] == "/tasktrace-profile-output")
        .expect("read-only publication mount");
    assert_eq!(publication["RW"], false);
    assert_eq!(
        Path::new(publication["Source"].as_str().unwrap())
            .canonicalize()
            .unwrap(),
        output_dir.canonicalize().unwrap()
    );
    let exchange = mounts
        .iter()
        .find(|mount| mount["Destination"] == "/tasktrace-profile-exchange")
        .expect("writable exchange mount");
    assert_eq!(exchange["RW"], true);
    assert_eq!(
        Path::new(exchange["Source"].as_str().unwrap())
            .canonicalize()
            .unwrap(),
        output_dir
            .join(".tasktrace-task-exchange")
            .canonicalize()
            .unwrap()
    );
    let environment = inspect["Config"]["Env"].as_array().unwrap();
    for expected in [
        "TASKTRACE_PROFILE_DIR=/tasktrace-profile-exchange",
        "TASKTRACE_RESOURCE_GATE_DIR=/tasktrace-profile-exchange",
        "TASKTRACE_SCHEDULER_GATE_DIR=/tasktrace-profile-exchange",
    ] {
        assert!(
            environment.iter().any(|value| value == expected),
            "{expected}"
        );
    }

    let result = child.wait_with_output().expect("wait for native profile");
    assert!(
        result.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr)
    );
    assert!(!output_dir.join(".tasktrace-task-exchange").exists());
    assert!(scoped_profile_containers(&output_dir).is_empty());
}

#[test]
fn existing_container_validation_is_native_with_python_fallback_disabled() {
    let root = tempfile::tempdir().unwrap();
    let output_dir = root.path().join("existing");
    let output = Command::new(binary())
        .args(["profile", "container", "--container", "caller-owned"])
        .args(["--command", "true", "--mount", "/host:/target"])
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", disabled_fallback())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("unsupported options: --mount"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
}

#[test]
fn allowlist_validation_is_native_with_python_fallback_disabled() {
    let root = tempfile::tempdir().unwrap();
    let output = Command::new(binary())
        .args(["profile", "container", "--image", "missing-image"])
        .args(["--command", "true", "--cpuset-cpus", "0"])
        .args(["--network-policy", "allowlist"])
        .args(["--allowed-host", "Allowed.TEST"])
        .args(["--network-policy-proxy-image", "missing-proxy-image"])
        .args(["--network-policy-external-network", "fixture-external"])
        .args([
            "--mount",
            "type=bind,src=/var/run/docker.sock,dst=/tmp/runtime.sock",
        ])
        .arg("--output-dir")
        .arg(root.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", disabled_fallback())
        .output()
        .unwrap();
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("container-runtime socket mounts"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
}

#[test]
#[ignore = "requires a local cgroup-v2 Docker daemon and explicit image/cpuset"]
fn public_memory_detail_is_native_enforced_and_directly_evidenced() {
    const MIB: u64 = 1024 * 1024;
    const LIMIT: u64 = 128 * MIB;
    let (image, _) = docker_inputs();
    let image = image.to_str().expect("Docker image reference is UTF-8");
    assert!(
        immutable_digest(image),
        "TASKTRACE_RUST_DOCKER_TEST_IMAGE must be an immutable sha256 image ID or repo digest"
    );
    let expected_image_id = local_image_id(image);
    let root = tempfile::tempdir().unwrap();
    let output_dir = root.path().join("memory-detail");
    let profiled = run_profile(
        &output_dir,
        &[
            "--command",
            "python3 -c 'import time; x=bytearray(32*1024*1024); x[::4096]=b\"x\"*8192; time.sleep(0.3)'",
            "--profile-set",
            "memory-detail",
            "--memory-sample-interval-sec",
            "0.02",
            "--memory-limit-bytes",
            "134217728",
            "--memory-swap-total-limit-bytes",
            "134217728",
        ],
    );
    assert!(
        profiled.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&profiled.stdout),
        String::from_utf8_lossy(&profiled.stderr)
    );
    let profile = report(&output_dir);
    let memory = &profile["memory"];
    assert_eq!(
        profile["image_identity"]["id"].as_str(),
        Some(expected_image_id.as_str())
    );
    assert_eq!(profile["profiling_valid"], true);
    assert_eq!(memory["schema_version"], "container-cgroup-memory.v2");
    assert_eq!(memory["instrumentation_valid"], true);
    assert_eq!(memory["peak_valid"], true);
    assert_eq!(memory["timeline_valid_for_shape"], true);
    assert_eq!(memory["observation_boundary_complete"], true);
    assert_eq!(memory["valid_for_observed_memory_reporting"], true);
    assert_eq!(
        memory["valid_for_sizing"],
        memory["natural_peak_trustworthy"]
    );
    let peak = memory["lifetime"]["peak_bytes"].as_u64().unwrap();
    assert!(peak >= 32 * MIB, "peak={peak}");
    assert!(peak < LIMIT, "peak={peak}");
    assert_eq!(memory["limits"]["memory_max_bytes"], LIMIT);
    assert_eq!(memory["limits"]["memory_max_state"], "finite");
    assert_eq!(memory["limits"]["swap_max_bytes"], 0);
    assert_eq!(
        memory["observed_limit_margin"]["remaining_to_limit_bytes"],
        LIMIT - peak
    );
    assert_eq!(
        memory["configured_enforcement"]["docker_host_config_memory_bytes"],
        LIMIT
    );
    assert_eq!(
        memory["configured_enforcement"]["docker_host_config_memory_swap_total_bytes"],
        LIMIT
    );
    assert!(memory["sample_count"].as_u64().unwrap() >= 2);
    assert!(output_dir.join("resource_timeseries.json").is_file());
    assert!(output_dir
        .join("cgroup_memory_source_snapshots.json")
        .is_file());

    assert_eq!(
        profile["evidence_bundle"]["resolved_profiler_ids"],
        serde_json::json!(["cgroup-memory"])
    );
    let evidence: Value =
        serde_json::from_slice(&fs::read(output_dir.join("evidence_bundle.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["profilers"].as_array().unwrap().len(), 1);
    assert_eq!(evidence["profilers"][0]["profiler_id"], "cgroup-memory");
    assert_eq!(evidence["extensions"]["all_selected_profilers_valid"], true);
    let stream_ids = evidence["streams"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|stream| stream["stream_id"].as_str())
        .collect::<Vec<_>>();
    assert!(stream_ids.contains(&"memory-current"));
    assert!(stream_ids.contains(&"memory-peak"));
    let artifact_ids = evidence["artifacts"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|artifact| artifact["artifact_id"].as_str())
        .collect::<Vec<_>>();
    assert!(artifact_ids.contains(&"cgroup-resource-timeseries"));
    assert!(artifact_ids.contains(&"cgroup-memory-source-snapshots"));
    assert_bundle_artifacts(&output_dir, &evidence);
    assert_eq!(
        scoped_profile_containers(&output_dir),
        Vec::<String>::new(),
        "an output-scoped profiler container survived cleanup"
    );
}

#[test]
#[ignore = "requires a local cgroup-v2 Docker daemon and explicit image/cpuset"]
fn public_cpu_average_and_coordinated_baseline_are_native() {
    let root = tempfile::tempdir().unwrap();
    let average_dir = root.path().join("average");
    let average = run_profile(
        &average_dir,
        &[
            "--command",
            "python3 -c 'sum(i*i for i in range(1500000))'",
            "--profile-set",
            "cpu-average",
        ],
    );
    assert!(
        average.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&average.stderr)
    );
    let average_stdout = String::from_utf8(average.stdout).unwrap();
    assert!(average_stdout.contains("Report:"));
    assert!(average_stdout.contains("Evidence bundle:"));
    let average_report = report(&average_dir);
    assert_eq!(average_report["profiling_valid"], true);
    assert_eq!(average_report["evidence_bundle"]["mode"], "profile");
    assert_eq!(
        average_report["evidence_bundle"]["scope"]["identity"]["cpu_placement"]
            ["validation_status"],
        "passed"
    );

    let baseline_dir = root.path().join("baseline");
    let baseline = run_profile(
        &baseline_dir,
        &["--command", "true", "--coordinated-baseline"],
    );
    assert!(
        baseline.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&baseline.stderr)
    );
    let baseline_report = report(&baseline_dir);
    assert_eq!(baseline_report["profiling_valid"], true);
    assert_eq!(baseline_report["evidence_bundle"]["mode"], "baseline");
}

#[test]
#[ignore = "requires a local cgroup-v2 Docker daemon and explicit image/cpuset"]
fn public_exit_status_fails_for_workload_or_cpu_validity() {
    let root = tempfile::tempdir().unwrap();
    let command_dir = root.path().join("command-failed");
    let command = run_profile(
        &command_dir,
        &["--command", "exit 7", "--profile-set", "cpu-average"],
    );
    assert_eq!(command.status.code(), Some(1));
    let command_report = report(&command_dir);
    assert_eq!(command_report["command_exit_code"], 7);
    assert_eq!(command_report["profiling_valid"], true);

    let invalid_dir = root.path().join("cpu-invalid");
    let invalid = run_profile(
        &invalid_dir,
        &[
            "--command",
            "true",
            "--cpu-limit",
            "1",
            "--profile-set",
            "cpu-average",
        ],
    );
    assert_eq!(
        invalid.status.code(),
        Some(1),
        "stderr={}",
        String::from_utf8_lossy(&invalid.stderr)
    );
    let invalid_report = report(&invalid_dir);
    assert_eq!(invalid_report["command_exit_code"], 0);
    assert_eq!(invalid_report["profiling_valid"], false);
    assert!(invalid_report["cpu"]["invalid_reasons"]
        .as_array()
        .unwrap()
        .iter()
        .any(|reason| reason.as_str().unwrap().contains("finite cpu.max")));
}
