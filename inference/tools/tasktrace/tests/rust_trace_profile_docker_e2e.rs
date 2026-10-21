// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in real-Docker gate for native finalized-TaskTrace-trace profiling.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
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

fn write_trace(path: &Path) {
    let records = [
        json!({
            "record_type": "trace",
            "schema_version": "tasktrace-cpu-trace.v1",
            "task_name": "rust-tasktrace-profile-e2e",
            "source_trial": "rust-tasktrace-profile-e2e__trial",
            "reward": 1,
            "limitations": ["synthetic Docker contract fixture"],
            "environment": {"action_timeout_sec": 5.0},
        }),
        json!({
            "record_type": "action",
            "action_index": 1,
            "command": "sleep 0.2; printf native-tasktrace-profile > /tmp/native-tasktrace-profile.txt",
            "started_at_epoch": 1.0,
            "finished_at_epoch": 1.01,
            "wall_time_sec": 0.01,
            "returncode": 0,
            "exception_info": null,
        }),
    ];
    fs::write(
        path,
        records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write finalized TaskTrace trace");
}

fn write_memory_trace(path: &Path) {
    let records = [
        json!({
            "record_type": "trace",
            "schema_version": "tasktrace-cpu-trace.v1",
            "task_name": "rust-tasktrace-memory-profile-e2e",
            "source_trial": "rust-tasktrace-memory-profile-e2e__trial",
            "reward": 1,
            "limitations": ["synthetic finite-limit memory fixture"],
            "environment": {"action_timeout_sec": 5.0},
        }),
        json!({
            "record_type": "action",
            "action_index": 1,
            "command": "python3 -c 'import time; x=bytearray(32*1024*1024); x[::4096]=b\"x\"*8192; time.sleep(0.3)'",
            "started_at_epoch": 1.0,
            "finished_at_epoch": 1.01,
            "wall_time_sec": 0.01,
            "returncode": 0,
            "exception_info": null,
        }),
    ];
    fs::write(
        path,
        records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write finalized memory TaskTrace trace");
}

fn assert_memory_artifacts_validate(output_dir: &Path, bundle: &Value) {
    let artifacts = bundle["artifacts"]
        .as_array()
        .expect("direct memory bundle has raw artifacts");
    let mut ids = std::collections::BTreeSet::new();
    for artifact in artifacts {
        let id = artifact["artifact_id"]
            .as_str()
            .expect("memory artifact ID");
        assert!(
            ids.insert(id.to_owned()),
            "duplicate memory artifact ID {id}"
        );
        let path = output_dir.join(
            artifact["relative_path"]
                .as_str()
                .expect("memory artifact relative path"),
        );
        let bytes = fs::read(&path).expect("read retained memory artifact");
        assert_eq!(artifact["size_bytes"].as_u64(), Some(bytes.len() as u64));
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(artifact["sha256"].as_str(), Some(digest.as_str()));
    }
    assert_eq!(
        ids,
        std::collections::BTreeSet::from([
            "cgroup-memory-source-snapshots".to_owned(),
            "cgroup-resource-timeseries".to_owned(),
        ])
    );
}

fn assert_bundle_artifacts_validate(output_dir: &Path, bundle: &Value) {
    let artifacts = bundle["artifacts"]
        .as_array()
        .expect("evidence bundle has raw artifacts");
    let mut ids = std::collections::BTreeSet::new();
    for artifact in artifacts {
        let id = artifact["artifact_id"]
            .as_str()
            .expect("raw artifact has an ID");
        assert!(ids.insert(id.to_owned()), "duplicate raw artifact ID {id}");
        let relative = artifact["relative_path"]
            .as_str()
            .expect("raw artifact has a relative path");
        let path = output_dir.join(relative);
        let bytes = fs::read(&path).unwrap_or_else(|error| {
            panic!("read retained raw artifact {}: {error}", path.display())
        });
        assert_eq!(
            artifact["size_bytes"].as_u64(),
            Some(bytes.len() as u64),
            "raw artifact size drifted for {id}"
        );
        assert_eq!(
            artifact["sha256"].as_str(),
            Some(format!("{:x}", Sha256::digest(&bytes)).as_str()),
            "raw artifact digest drifted for {id}"
        );
    }
    for required in [
        "raw-replay-report",
        "raw-direct-evidence-bundle",
        "raw-source-report",
        "raw-container-profile-report",
        "cgroup-cpu-before",
        "cgroup-cpu-after",
        "cgroup-cpu-source-snapshots",
    ] {
        assert!(ids.contains(required), "missing raw artifact {required}");
    }
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
        .join("container_profile")
        .canonicalize()
        .expect("canonicalize profile output directory");
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
            continue;
        }
        let payload: Value =
            serde_json::from_slice(&inspected.stdout).expect("parse container inspection JSON");
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

#[test]
#[ignore = "requires Docker, cgroup v2, and runner-owned logical CPUs"]
fn native_tasktrace_trace_profile_is_content_bound_and_cleans_up() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "Docker test image must be an immutable sha256 image ID or repo digest"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let expected_image_id = local_image_id(&image);
    let temp = TempDir::new().expect("create TaskTrace profile E2E root");
    let trace_path = temp.path().join("tasktrace_shell_commands.jsonl");
    let output_dir = temp.path().join("profile");
    write_trace(&trace_path);
    let trace_sha = format!("{:x}", Sha256::digest(fs::read(&trace_path).unwrap()));

    let output = command_output(
        Command::new("timeout")
            .arg("--signal=TERM")
            .arg("--kill-after=5s")
            .arg("60s")
            .arg(binary())
            .arg("profile")
            .arg("trace")
            .arg(&trace_path)
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .arg("--workdir")
            .arg("/tmp")
            .arg("--pacing")
            .arg("unpaced")
            .arg("--network-policy")
            .arg("no-network")
            .arg("--strace-mode")
            .arg("none")
            .arg("--profile-set")
            .arg("cpu-average")
            .arg("--proc-stat-comparison")
            .arg("--proc-stat-interval-sec")
            .arg("0.05")
            .arg("--proc-stat-baseline-sec")
            .arg("0.15")
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temp.path().join("python-fallback-must-not-run"),
            ),
        "run the native finalized-TaskTrace-trace profile",
    );
    assert_success(&output, "native finalized-TaskTrace-trace profile");
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("Sampled /proc active-CPU peak (comparison only):"));

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("pipeline_report.json")).expect("read pipeline report"),
    )
    .expect("parse pipeline report JSON");
    assert_eq!(
        report["schema_version"].as_str(),
        Some("container-trace-profile.v1")
    );
    assert_eq!(report["status"].as_str(), Some("valid"));
    assert_eq!(
        report["task_name"].as_str(),
        Some("rust-tasktrace-profile-e2e")
    );
    assert_eq!(
        report["source_trial"].as_str(),
        Some("rust-tasktrace-profile-e2e__trial")
    );
    assert_eq!(
        report["source_trace_sha256"].as_str(),
        Some(trace_sha.as_str())
    );
    assert_eq!(
        report["image_identity"]["id"].as_str(),
        Some(expected_image_id.as_str())
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
    assert_eq!(
        report["execution_plan"]["source_identity"]["sha256"].as_str(),
        Some(trace_sha.as_str())
    );
    let command = report["execution_plan"]["command"][2]
        .as_str()
        .expect("execution plan has shell command");
    assert!(command.contains("/tasktrace-controller/tasktrace replay trace"));
    assert!(!command.contains("python"));
    assert_eq!(
        report["replay_summary"]["source_outcome_mismatch_count"].as_u64(),
        Some(0)
    );
    assert_eq!(
        report["replay_summary"]["actions_timed_out"].as_u64(),
        Some(0)
    );
    assert_eq!(
        report["container_cpu"]["method"].as_str(),
        Some("container-cgroup-v2-cpu.stat")
    );
    assert_eq!(
        report["scope"]["identity"]["cpu_placement"]["validation_status"].as_str(),
        Some("passed")
    );
    assert!(
        report["container_cpu"]["usage_sec"]
            .as_f64()
            .unwrap_or_default()
            > 0.0
    );
    assert_eq!(report["profiling_valid"].as_bool(), Some(true));
    assert_eq!(
        report["unified_profile_artifacts"]["workload_executions"].as_u64(),
        Some(1)
    );
    let proc_stat = &report["proc_stat_comparison"];
    assert_eq!(
        proc_stat["method"].as_str(),
        Some("tasktrace-style-proc-stat-sampling")
    );
    assert_eq!(proc_stat["instrumentation_valid"].as_bool(), Some(true));
    assert_eq!(
        proc_stat["reference_comparison"]["peak_values_are_equivalent"].as_bool(),
        Some(false)
    );
    assert!(
        proc_stat["baseline"]["sample_count"]
            .as_u64()
            .unwrap_or_default()
            >= 1
    );
    assert!(
        proc_stat["measurement"]["sample_count"]
            .as_u64()
            .unwrap_or_default()
            >= 1
    );
    assert!(proc_stat["measurement"]["peak_active_logical_cpus"]
        .as_u64()
        .is_some());
    assert!(report["warning"]
        .as_str()
        .unwrap_or_default()
        .contains("not task peak concurrency"));
    assert!(
        report["evidence_bundle"].get("sha256").is_none(),
        "pipeline must not embed the digest of a bundle that hashes the pipeline"
    );
    let bundle_path = PathBuf::from(
        report["evidence_bundle"]["path"]
            .as_str()
            .expect("pipeline has authoritative bundle path"),
    );
    let bundle: Value = serde_json::from_slice(
        &fs::read(bundle_path).expect("read authoritative augmented bundle"),
    )
    .expect("parse authoritative augmented bundle");
    assert_eq!(
        bundle["provenance"]["pipeline_augmentation"]["workload_executions_added"].as_u64(),
        Some(0)
    );
    assert!(bundle["profilers"]
        .as_array()
        .expect("evidence profiler list")
        .iter()
        .any(|profiler| {
            profiler["profiler_id"].as_str() == Some("proc-stat-host-context")
                && profiler["provenance"]["task_attributed"].as_bool() == Some(false)
        }));
    assert!(bundle["artifacts"]
        .as_array()
        .expect("evidence artifact list")
        .iter()
        .any(|artifact| artifact["artifact_id"].as_str() == Some("proc-stat-samples")));
    assert_bundle_artifacts_validate(&output_dir, &bundle);
    assert!(
        scoped_profile_containers(&output_dir).is_empty(),
        "native TaskTrace profile left its disposable container behind"
    );
}

#[test]
#[ignore = "requires Docker, cgroup v2, and runner-owned logical CPUs"]
fn native_tasktrace_memory_detail_is_single_run_content_bound_and_cleans_up() {
    const MIB: u64 = 1024 * 1024;
    const LIMIT: u64 = 128 * MIB;
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "Docker test image must be an immutable sha256 image ID or repo digest"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let expected_image_id = local_image_id(&image);
    let temp = TempDir::new().expect("create TaskTrace memory profile E2E root");
    let trace_path = temp.path().join("tasktrace_memory_action.jsonl");
    let output_dir = temp.path().join("memory-profile");
    write_memory_trace(&trace_path);
    let trace_sha = format!("{:x}", Sha256::digest(fs::read(&trace_path).unwrap()));

    let output = command_output(
        Command::new("timeout")
            .arg("--signal=TERM")
            .arg("--kill-after=5s")
            .arg("60s")
            .arg(binary())
            .arg("profile")
            .arg("trace")
            .arg(&trace_path)
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .arg("--workdir")
            .arg("/tmp")
            .arg("--pacing")
            .arg("unpaced")
            .arg("--network-policy")
            .arg("no-network")
            .arg("--profile-set")
            .arg("memory-detail")
            .arg("--memory-sample-interval-sec")
            .arg("0.02")
            .arg("--memory-limit-bytes")
            .arg(LIMIT.to_string())
            .arg("--memory-swap-total-limit-bytes")
            .arg(LIMIT.to_string())
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temp.path().join("python-fallback-must-not-run"),
            ),
        "run native finalized-TaskTrace memory profile",
    );
    assert_success(&output, "native finalized-TaskTrace memory profile");

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("pipeline_report.json")).expect("read memory pipeline report"),
    )
    .expect("parse memory pipeline report");
    assert_eq!(report["status"], "valid");
    assert_eq!(report["source_trace_sha256"], trace_sha);
    assert_eq!(report["image_identity"]["id"], expected_image_id);
    assert_eq!(
        report["controller_identity"]["sha256"],
        report["execution_plan"]["source_identity"]["rust_controller_binary_sha256"]
    );
    assert_eq!(
        report["execution_plan"]["source_identity"]["sha256"],
        trace_sha
    );
    assert_eq!(report["replay_summary"]["source_outcome_mismatch_count"], 0);
    assert_eq!(report["replay_summary"]["actions_timed_out"], 0);
    assert_eq!(
        report["unified_profile_artifacts"]["workload_executions"],
        1
    );

    let memory = &report["container_memory"];
    assert_eq!(memory["schema_version"], "container-cgroup-memory.v2");
    assert_eq!(memory["valid_for_observed_memory_reporting"], true);
    assert_eq!(memory["peak_valid"], true);
    assert_eq!(memory["timeline_valid_for_shape"], true);
    assert_eq!(memory["observation_boundary_complete"], true);
    assert_eq!(
        memory["valid_for_sizing"],
        memory["natural_peak_trustworthy"]
    );
    let peak = memory["lifetime"]["peak_bytes"].as_u64().unwrap();
    assert!(peak >= 32 * MIB, "peak={peak}");
    assert!(peak < LIMIT, "peak={peak}");
    assert_eq!(memory["limits"]["memory_max_bytes"], LIMIT);
    assert_eq!(memory["limits"]["swap_max_bytes"], 0);
    assert_eq!(
        memory["observed_limit_margin"]["remaining_to_limit_bytes"],
        LIMIT - peak
    );
    assert!(memory["sample_count"].as_u64().unwrap() >= 2);

    let direct_path = output_dir.join("container_profile/evidence_bundle.json");
    let direct: Value = serde_json::from_slice(&fs::read(direct_path).unwrap()).unwrap();
    assert_eq!(direct["profilers"].as_array().unwrap().len(), 1);
    assert_eq!(direct["profilers"][0]["profiler_id"], "cgroup-memory");
    assert_eq!(direct["extensions"]["all_selected_profilers_valid"], true);
    assert_eq!(
        direct["extensions"]["raw_profiler_reports"]["cgroup-memory"]["lifetime"]["peak_bytes"],
        peak
    );
    let stream_ids = direct["streams"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|stream| stream["stream_id"].as_str())
        .collect::<Vec<_>>();
    assert!(stream_ids.contains(&"memory-current"));
    assert!(stream_ids.contains(&"memory-peak"));
    assert_memory_artifacts_validate(&output_dir.join("container_profile"), &direct);
    assert!(
        scoped_profile_containers(&output_dir).is_empty(),
        "native TaskTrace memory profile left its disposable container behind"
    );
}
