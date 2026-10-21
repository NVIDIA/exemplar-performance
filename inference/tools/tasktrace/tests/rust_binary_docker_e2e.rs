// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in real-Docker gate for the native Rust container controller.
//!
//! The gate is ignored by default because it needs daemon access, cgroup v2,
//! a runner-owned cpuset, and a preloaded immutable native image containing
//! Bash. Run it explicitly against the frozen Rust implementation:
//!
//! ```text
//! TASKTRACE_RUST_DOCKER_TEST_IMAGE=sha256:<64-lowercase-hex> \
//! TASKTRACE_RUST_DOCKER_CPUSET=16-19 \
//! cargo test --test rust_binary_docker_e2e -- --ignored --nocapture
//! ```
//!
//! `TASKTRACE_LEGACY_PYTHON` is forced to a nonexistent executable as a negative
//! canary: the native public path must never try to start it.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn file_sha256(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(fs::read(path).expect("read artifact for SHA-256"))
    )
}

fn representative_atif_fixture() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("offline_replayer/tests/fixtures/representative_atif")
        .join("swebenchpro-codex__fixture/agent/trajectory.json")
}

fn command_output(command: &mut Command, description: &str) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("unable to {description}: {error}"))
}

fn assert_success(output: &Output, description: &str) {
    assert!(
        output.status.success(),
        "{description} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
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

fn local_image_id(image: &str) -> String {
    let output = command_output(
        Command::new("docker")
            .arg("image")
            .arg("inspect")
            .arg("--format")
            .arg("{{.Id}}")
            .arg(image),
        "inspect the preloaded Docker fixture image",
    );
    assert_success(&output, "Docker image inspection");
    String::from_utf8(output.stdout)
        .expect("Docker image ID is UTF-8")
        .trim()
        .to_owned()
}

fn scoped_profile_containers(output_dir: &Path) -> Vec<String> {
    let listed = command_output(
        Command::new("docker")
            .arg("ps")
            .arg("--all")
            .arg("--filter")
            .arg("name=tasktrace-profile-")
            .arg("--format")
            .arg("{{.ID}}"),
        "list profiler containers",
    );
    assert_success(&listed, "Docker profiler-container listing");
    let expected_source = output_dir
        .canonicalize()
        .expect("canonicalize Rust Docker E2E output directory");
    let mut matched = Vec::new();
    for id in String::from_utf8_lossy(&listed.stdout)
        .lines()
        .filter(|line| !line.is_empty())
    {
        let inspected = command_output(
            Command::new("docker").arg("inspect").arg(id),
            "inspect a profiler container",
        );
        if !inspected.status.success() {
            // A correctly cleaned container can disappear between list and
            // inspect; any still-observable scoped container is checked below.
            continue;
        }
        let payload: Value = serde_json::from_slice(&inspected.stdout)
            .expect("parse Docker container inspection JSON");
        let Some(container) = payload.as_array().and_then(|items| items.first()) else {
            continue;
        };
        let scoped = container["Mounts"]
            .as_array()
            .into_iter()
            .flatten()
            .any(|mount| {
                mount["Destination"].as_str() == Some("/tasktrace-profile-output")
                    && mount["Source"]
                        .as_str()
                        .map(Path::new)
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

#[test]
#[ignore = "requires Docker and a native Rust profile-container controller"]
fn rust_controller_profiles_one_container_and_removes_its_sandbox() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "TASKTRACE_RUST_DOCKER_TEST_IMAGE must be repo@sha256:<64 lowercase hex> or a local image ID"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");

    let server = command_output(
        Command::new("docker")
            .arg("version")
            .arg("--format")
            .arg("{{.Server.Version}}"),
        "query the Docker server",
    );
    assert_success(&server, "Docker server preflight");
    let expected_image_id = local_image_id(&image);

    let temp = TempDir::new().expect("create temporary Docker E2E root");
    let output_dir = temp.path().join("container-profile");
    let result = command_output(
        Command::new(binary())
            .arg("profile")
            .arg("container")
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .arg("--workdir")
            .arg("/tmp")
            .arg("--network-policy")
            .arg("no-network")
            .arg("--strace-mode")
            .arg("none")
            .arg("--profile-set")
            .arg("cpu-average")
            .arg("--command")
            .arg("python -c 'import time; end=time.monotonic()+0.2; exec(\"while time.monotonic() < end: pass\")'")
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temp.path().join("python-fallback-must-not-run"),
            ),
        "run the native Rust Docker profile",
    );
    assert_success(&result, "native Rust Docker profile");

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("profile_report.json")).expect("read Docker profile report"),
    )
    .expect("parse Docker profile report JSON");
    assert_eq!(
        report["schema_version"].as_str(),
        Some("container-command-profile.v1")
    );
    assert_eq!(report["runtime"].as_str(), Some("docker"));
    assert_eq!(report["scope"].as_str(), Some("disposable-image-container"));
    assert_eq!(report["command_exit_code"].as_i64(), Some(0));
    assert_eq!(
        report["image_identity"]["id"].as_str(),
        Some(expected_image_id.as_str())
    );
    assert_eq!(
        report["cpu"]["method"].as_str(),
        Some("container-cgroup-v2-cpu.stat")
    );
    assert!(report["cpu"]["usage_sec"].as_f64().unwrap_or_default() > 0.0);
    assert!(report["cpu"]["wall_sec"].as_f64().unwrap_or_default() > 0.0);
    assert!(
        scoped_profile_containers(&output_dir).is_empty(),
        "native Rust Docker controller left its disposable container behind"
    );
}

#[test]
#[ignore = "requires Docker, cgroup v2, and runner-owned logical CPUs"]
fn rust_proc_stat_context_profiles_one_workload_and_cleans_up() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "TASKTRACE_RUST_DOCKER_TEST_IMAGE must be repo@sha256:<64 lowercase hex> or a local image ID"
    );
    let expected_image_id = local_image_id(&image);
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let temp = TempDir::new().expect("create temporary proc-stat Docker E2E root");
    let output_dir = temp.path().join("proc-stat-profile");
    let result = command_output(
        Command::new(binary())
            .arg("profile")
            .arg("container")
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .arg("--workdir")
            .arg("/tmp")
            .arg("--network-policy")
            .arg("no-network")
            .arg("--profile-set")
            .arg("cpu-average")
            .arg("--proc-stat-comparison")
            .arg("--proc-stat-interval-sec")
            .arg("0.05")
            .arg("--proc-stat-baseline-sec")
            .arg("0.15")
            .arg("--command")
            .arg("python -c 'import time; end=time.monotonic()+0.25; exec(\"while time.monotonic() < end: pass\")'")
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temp.path().join("python-fallback-must-not-run"),
            ),
        "run the native Rust proc-stat comparison profile",
    );
    assert_success(&result, "native Rust proc-stat comparison profile");

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("profile_report.json")).expect("read proc-stat profile report"),
    )
    .expect("parse proc-stat profile report");
    let proc_stat = &report["proc_stat_comparison"];
    assert_eq!(
        report["image_identity"]["id"].as_str(),
        Some(expected_image_id.as_str())
    );
    assert_eq!(
        proc_stat["method"].as_str(),
        Some("tasktrace-style-proc-stat-sampling")
    );
    assert_eq!(proc_stat["instrumentation_valid"].as_bool(), Some(true));
    assert_eq!(
        proc_stat["reference_comparison"]["peak_values_are_equivalent"].as_bool(),
        Some(false)
    );
    assert!(proc_stat["baseline"]["sample_count"].as_u64().unwrap_or(0) >= 1);
    assert!(
        proc_stat["measurement"]["sample_count"]
            .as_u64()
            .unwrap_or(0)
            >= 1
    );
    assert!(
        proc_stat["baseline_window"]["end_ns"].as_u64().unwrap()
            <= report["cpu"]["window_start_ns"].as_u64().unwrap()
    );
    assert!(
        proc_stat["measurement_window"]["start_ns"]
            .as_u64()
            .unwrap()
            >= report["cpu"]["window_start_ns"].as_u64().unwrap()
    );
    assert!(
        proc_stat["measurement_window"]["end_ns"].as_u64().unwrap()
            >= report["cpu"]["window_end_ns"].as_u64().unwrap()
    );
    assert!(report["warning"]
        .as_str()
        .unwrap_or_default()
        .contains("not task peak concurrency"));
    assert!(output_dir.join("proc_stat_samples.json").is_file());

    let evidence_path = PathBuf::from(
        report["evidence_bundle"]["path"]
            .as_str()
            .expect("proc-stat profile retains evidence path"),
    );
    let evidence: Value =
        serde_json::from_slice(&fs::read(evidence_path).expect("read proc-stat evidence bundle"))
            .expect("parse proc-stat evidence bundle");
    assert!(evidence["profilers"]
        .as_array()
        .expect("proc-stat profiler list")
        .iter()
        .any(|profiler| {
            profiler["profiler_id"].as_str() == Some("proc-stat-host-context")
                && profiler["provenance"]["task_attributed"].as_bool() == Some(false)
        }));
    let proc_artifact = evidence["artifacts"]
        .as_array()
        .expect("proc-stat artifact list")
        .iter()
        .find(|artifact| artifact["artifact_id"].as_str() == Some("proc-stat-samples"))
        .expect("proc-stat raw artifact descriptor");
    let proc_samples = output_dir.join(
        proc_artifact["relative_path"]
            .as_str()
            .expect("proc-stat artifact path is textual"),
    );
    assert_eq!(
        proc_artifact["size_bytes"].as_u64(),
        Some(fs::metadata(&proc_samples).unwrap().len())
    );
    assert_eq!(
        proc_artifact["sha256"].as_str(),
        Some(file_sha256(&proc_samples).as_str())
    );
    let rendered = command_output(
        Command::new(binary())
            .arg("report")
            .arg("unified-view")
            .arg(
                report["evidence_bundle"]["path"]
                    .as_str()
                    .expect("proc-stat evidence path is textual"),
            )
            .arg("--output-dir")
            .arg(temp.path().join("proc-stat-view"))
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temp.path().join("python-fallback-must-not-run"),
            ),
        "render the proc-stat evidence bundle natively",
    );
    assert_success(&rendered, "native proc-stat unified view");
    assert!(
        scoped_profile_containers(&output_dir).is_empty(),
        "native proc-stat controller left its disposable container behind"
    );
}

#[test]
#[ignore = "requires Docker, cgroup v2, and four runner-owned logical CPUs"]
fn rust_atif_controller_runs_inside_measured_container_and_cleans_up() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "TASKTRACE_RUST_DOCKER_TEST_IMAGE must be repo@sha256:<64 lowercase hex> or a local image ID"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let expected_image_id = local_image_id(&image);
    let temp = TempDir::new().expect("create temporary Rust ATIF Docker E2E root");
    let output_dir = temp.path().join("atif-profile");

    let result = command_output(
        Command::new("timeout")
            .arg("--signal=TERM")
            .arg("--kill-after=5s")
            .arg("60s")
            .arg(binary())
            .arg("profile")
            .arg("atif")
            .arg(representative_atif_fixture())
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .arg("--controller-runtime")
            .arg("rust-host-mount")
            .arg("--pacing")
            .arg("unpaced")
            .arg("--network-policy")
            .arg("no-network")
            .arg("--workdir")
            .arg("/tmp")
            .arg("--patch-root")
            .arg("/tmp")
            .arg("--strace-mode")
            .arg("none")
            .arg("--profile-set")
            .arg("cpu-average")
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temp.path().join("python-fallback-must-not-run"),
            ),
        "run the measured native Rust ATIF controller",
    );
    assert_success(&result, "measured native Rust ATIF controller");

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("pipeline_report.json")).expect("read ATIF pipeline report"),
    )
    .expect("parse ATIF pipeline report JSON");
    assert_eq!(
        report["schema_version"].as_str(),
        Some("container-atif-trajectory-profile.v1")
    );
    assert_eq!(report["status"].as_str(), Some("valid"));
    assert_eq!(report["profiling_valid"].as_bool(), Some(true));
    assert_eq!(
        report["controller_runtime"].as_str(),
        Some("rust-host-mount")
    );
    assert_eq!(
        report["controller_identity"]["kind"].as_str(),
        Some("rust-host-mounted-binary")
    );
    let controller_sha = report["controller_identity"]["sha256"]
        .as_str()
        .expect("controller identity has SHA-256");
    assert_eq!(controller_sha.len(), 64);
    assert_eq!(
        report["execution_plan"]["source_identity"]["rust_controller_binary_sha256"].as_str(),
        Some(controller_sha)
    );
    let controller_mount = report["execution_plan"]["mounts"]
        .as_array()
        .expect("execution plan retains mounts")
        .iter()
        .find(|mount| mount["target"].as_str() == Some("/tasktrace-controller/tasktrace"))
        .expect("execution plan retains Rust controller mount");
    assert_eq!(
        controller_mount["source_identity"]["kind"].as_str(),
        Some("rust-controller-binary")
    );
    assert_eq!(
        controller_mount["source_identity"]["sha256"].as_str(),
        Some(controller_sha)
    );
    let command = report["execution_plan"]["command"][2]
        .as_str()
        .expect("execution plan retains shell command");
    assert!(command.contains("/tasktrace-controller/tasktrace replay atif"));
    assert!(!command.contains("python"));
    assert_eq!(
        report["image_identity"]["id"].as_str(),
        Some(expected_image_id.as_str())
    );
    assert_eq!(
        report["container_cpu"]["method"].as_str(),
        Some("container-cgroup-v2-cpu.stat")
    );
    assert_eq!(
        report["container_cpu"]["valid_for_cpu_performance"].as_bool(),
        Some(true)
    );
    assert!(
        report["container_cpu"]["usage_sec"]
            .as_f64()
            .unwrap_or_default()
            > 0.0
    );
    assert_eq!(
        report["replay_summary"]["source_outcome_mismatch_count"].as_u64(),
        Some(0)
    );
    assert_eq!(
        report["evidence_bundle"]["authoritative"].as_bool(),
        Some(true)
    );
    assert_eq!(
        report["evidence_bundle"]["generation_method"].as_str(),
        Some("verified-direct-coordinator-bundle-augmentation")
    );
    assert_eq!(
        report["unified_profile_artifacts"]["workload_executions"].as_u64(),
        Some(1)
    );
    let evidence_path = PathBuf::from(
        report["evidence_bundle"]["path"]
            .as_str()
            .expect("pipeline report retains authoritative evidence path"),
    );
    assert_eq!(evidence_path, output_dir.join("evidence_bundle.json"));
    let evidence: Value = serde_json::from_slice(
        &fs::read(&evidence_path).expect("read replay-augmented evidence bundle"),
    )
    .expect("parse replay-augmented evidence bundle");
    assert!(evidence["profilers"]
        .as_array()
        .expect("evidence profiler list")
        .iter()
        .any(|profiler| profiler["profiler_id"].as_str() == Some("replay-actions")));
    assert!(evidence["artifacts"]
        .as_array()
        .expect("evidence artifact list")
        .iter()
        .any(|artifact| artifact["artifact_id"].as_str() == Some("raw-replay-report")));
    assert!(output_dir.join("unified_profile_artifacts.json").is_file());
    assert!(
        scoped_profile_containers(&output_dir).is_empty(),
        "native Rust ATIF controller left its disposable container behind"
    );
}
