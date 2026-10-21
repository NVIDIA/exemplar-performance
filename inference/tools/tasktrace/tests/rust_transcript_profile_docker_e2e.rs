// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in real-Docker proof for native transcript profiling.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
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

fn write_transcript(path: &Path) {
    let records = [
        json!({
            "task_id": "rust-transcript-profile-e2e",
            "trial_name": "rust-transcript-profile-e2e.1-of-1.default",
            "turn": 1,
            "answer": "[Tool Call: read]\n{\"path\":\"input.txt\"}",
        }),
        json!({
            "task_id": "rust-transcript-profile-e2e",
            "trial_name": "rust-transcript-profile-e2e.1-of-1.default",
            "turn": 2,
            "answer": concat!(
                "[Tool Call: exec]\n",
                "{\"command\":\"i=0; while [ \\\"$i\\\" -lt 150000 ]; do i=$((i+1)); done; ",
                "printf native-transcript-profile > /captured/agent_workspace/result.txt\"}"
            ),
        }),
        json!({
            "task_id": "rust-transcript-profile-e2e",
            "trial_name": "rust-transcript-profile-e2e.1-of-1.default",
            "turn": 3,
            "answer": "[Tool Call: read]\n{\"path\":\"/captured/agent_workspace/result.txt\"}",
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
    .expect("write transcript profile fixture");
}

fn assert_bundle_artifacts_validate(output_dir: &Path, bundle: &Value) {
    let artifacts = bundle["artifacts"]
        .as_array()
        .expect("evidence bundle has raw artifacts");
    let mut ids = BTreeSet::new();
    for artifact in artifacts {
        let id = artifact["artifact_id"]
            .as_str()
            .expect("raw artifact has an ID");
        assert!(ids.insert(id.to_owned()), "duplicate raw artifact ID {id}");
        let relative = artifact["relative_path"]
            .as_str()
            .expect("raw artifact has a relative path");
        let path = output_dir.join(relative);
        let bytes = fs::read(&path)
            .unwrap_or_else(|error| panic!("read raw artifact {}: {error}", path.display()));
        assert_eq!(artifact["size_bytes"].as_u64(), Some(bytes.len() as u64));
        let digest = format!("{:x}", Sha256::digest(&bytes));
        assert_eq!(artifact["sha256"].as_str(), Some(digest.as_str()));
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
fn native_transcript_profile_is_content_bound_correct_and_cleans_up() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "Docker test image must be an immutable sha256 image ID or repo digest"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let expected_image_id = local_image_id(&image);
    let expected_controller_sha = format!(
        "{:x}",
        Sha256::digest(fs::read(binary()).expect("read native controller binary"))
    );
    let temporary = TempDir::new().expect("create transcript profile E2E root");
    let transcript = temporary.path().join("transcript.jsonl");
    write_transcript(&transcript);
    let transcript_bytes = fs::read(&transcript).expect("read transcript fixture");
    let transcript_sha = format!("{:x}", Sha256::digest(&transcript_bytes));
    let initial = temporary.path().join("initial-workspace");
    fs::create_dir(&initial).expect("create initial workspace");
    fs::write(initial.join("input.txt"), "seeded-input\n").expect("write seeded input");
    let output_dir = temporary.path().join("profile");

    let output = command_output(
        Command::new("timeout")
            .arg("--signal=TERM")
            .arg("--kill-after=5s")
            .arg("60s")
            .arg(binary())
            .args(["profile", "transcript"])
            .arg(&transcript)
            .args([
                "--task-id",
                "rust-transcript-profile-e2e",
                "--trial-name",
                "rust-transcript-profile-e2e.1-of-1.default",
                "--image",
                &image,
                "--cpuset-cpus",
                &cpuset,
                "--source-workspace",
                "/captured/agent_workspace",
                "--workdir",
                "/tmp",
                "--network-policy",
                "no-network",
                "--strace-mode",
                "none",
                "--profile-set",
                "cpu-average",
                "--verification-command",
                "test \"$(cat /tasktrace/workspace/result.txt)\" = native-transcript-profile",
            ])
            .arg("--initial-workspace")
            .arg(&initial)
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temporary.path().join("python-fallback-must-not-run"),
            ),
        "run native transcript profile",
    );
    assert_success(&output, "native transcript profile");

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("pipeline_report.json")).expect("read pipeline report"),
    )
    .expect("parse pipeline report JSON");
    assert_eq!(
        report["schema_version"].as_str(),
        Some("container-transcript-profile.v1")
    );
    assert_eq!(report["status"].as_str(), Some("valid"));
    assert_eq!(report["profiling_valid"].as_bool(), Some(true));
    assert_eq!(report["allow_unsupported"].as_bool(), Some(false));
    assert_eq!(report["unsupported_actions_validated"].as_u64(), Some(0));
    assert_eq!(
        report["transcript_sha256"].as_str(),
        Some(transcript_sha.as_str())
    );
    let snapshot = PathBuf::from(
        report["transcript_snapshot"]
            .as_str()
            .expect("report retains transcript snapshot"),
    );
    assert_eq!(
        fs::read(snapshot).expect("read transcript snapshot"),
        transcript_bytes
    );
    assert_eq!(
        report["image_identity"]["id"].as_str(),
        Some(expected_image_id.as_str())
    );
    assert_eq!(
        report["controller_identity"]["sha256"].as_str(),
        Some(expected_controller_sha.as_str())
    );
    assert_eq!(
        report["execution_plan"]["source_identity"]["sha256"].as_str(),
        Some(transcript_sha.as_str())
    );
    assert_eq!(
        report["execution_plan"]["source_identity"]["rust_controller_binary_sha256"].as_str(),
        Some(expected_controller_sha.as_str())
    );
    assert_eq!(
        report["execution_plan"]["source_identity"]["initial_workspace_sha256"],
        report["initial_workspace_sha256"]
    );
    let command = report["execution_plan"]["command"][2]
        .as_str()
        .expect("execution plan retains measured command");
    assert!(command.contains("/tasktrace-controller/tasktrace replay transcript"));
    assert!(!command.contains("python"));
    let mounts = report["execution_plan"]["mounts"]
        .as_array()
        .expect("execution plan retains mounts");
    let controller_mount = mounts
        .iter()
        .find(|mount| mount["target"].as_str() == Some("/tasktrace-controller/tasktrace"))
        .expect("controller mount is retained");
    assert_eq!(controller_mount["read_only"].as_bool(), Some(true));
    assert_eq!(
        controller_mount["source_identity"]["sha256"].as_str(),
        Some(expected_controller_sha.as_str())
    );
    assert!(mounts.iter().any(|mount| {
        mount["target"].as_str() == Some("/tasktrace-input")
            && mount["read_only"].as_bool() == Some(true)
    }));
    assert!(mounts.iter().any(|mount| {
        mount["target"].as_str() == Some("/tasktrace-initial-workspace")
            && mount["read_only"].as_bool() == Some(true)
    }));
    assert!(report["initial_workspace_sha256"]
        .as_str()
        .is_some_and(|value| value.len() == 64));
    assert_eq!(
        report["source_workspace_provenance_state"].as_str(),
        Some("unavailable")
    );
    assert_eq!(
        report["replay_summary"]["actions_extracted"].as_u64(),
        Some(3)
    );
    assert_eq!(
        report["replay_summary"]["actions_replayed"].as_u64(),
        Some(3)
    );
    assert_eq!(report["replay_summary"]["actions_ok"].as_u64(), Some(3));
    assert_eq!(report["replay_summary"]["actions_failed"].as_u64(), Some(0));
    assert_eq!(
        report["replay_summary"]["actions_unsupported"].as_u64(),
        Some(0)
    );
    assert_eq!(
        fs::read_to_string(output_dir.join("replay/workspace/result.txt"))
            .expect("read replay output"),
        "native-transcript-profile"
    );
    assert_eq!(report["verification"]["valid"].as_bool(), Some(true));
    assert_eq!(
        report["container_cpu"]["method"].as_str(),
        Some("container-cgroup-v2-cpu.stat")
    );
    assert!(
        report["container_cpu"]["usage_sec"]
            .as_f64()
            .unwrap_or_default()
            > 0.0
    );
    assert_eq!(
        report["scope"]["identity"]["cpu_placement"]["validation_status"].as_str(),
        Some("passed")
    );
    assert_eq!(
        report["scope"]["identity"]["cpu_placement"]["requested_cpuset_cpus"].as_str(),
        Some(cpuset.as_str())
    );

    let bundle_path = PathBuf::from(
        report["evidence_bundle"]["path"]
            .as_str()
            .expect("pipeline retains authoritative bundle"),
    );
    let bundle: Value =
        serde_json::from_slice(&fs::read(bundle_path).expect("read authoritative evidence bundle"))
            .expect("parse authoritative evidence bundle");
    assert_eq!(
        bundle["provenance"]["pipeline_augmentation"]["workload_executions_added"].as_u64(),
        Some(0)
    );
    assert_bundle_artifacts_validate(&output_dir, &bundle);
    assert!(
        scoped_profile_containers(&output_dir).is_empty(),
        "native transcript profile left its disposable container behind"
    );
}
