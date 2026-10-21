// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in real-Docker gate for native execve/process tracing.

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

fn output(command: &mut Command, description: &str) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("unable to {description}: {error}"))
}

fn assert_success(result: &Output, description: &str) {
    assert!(
        result.status.success(),
        "{description} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        result.status.code(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
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

fn image_has_strace(image: &str) -> bool {
    output(
        Command::new("docker")
            .args(["run", "--rm", "--entrypoint", "/bin/sh", image, "-c"])
            .arg("command -v strace >/dev/null 2>&1"),
        "check the fixture image for strace",
    )
    .status
    .success()
}

fn local_image_id(image: &str) -> String {
    let inspected = output(
        Command::new("docker").args(["image", "inspect", "--format", "{{.Id}}", image]),
        "inspect the strace fixture image",
    );
    assert_success(&inspected, "Docker image inspection");
    String::from_utf8(inspected.stdout)
        .expect("Docker image ID is UTF-8")
        .trim()
        .to_owned()
}

fn scoped_profile_containers(output_directories: &[PathBuf]) -> Vec<String> {
    let listed = output(
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
    assert_success(&listed, "Docker profiler-container listing");
    let expected = output_directories
        .iter()
        .filter_map(|path| path.canonicalize().ok())
        .collect::<Vec<_>>();
    String::from_utf8_lossy(&listed.stdout)
        .lines()
        .filter(|line| !line.is_empty())
        .filter_map(|id| {
            let inspected = output(
                Command::new("docker").arg("inspect").arg(id),
                "inspect a profiler container",
            );
            if !inspected.status.success() {
                return None;
            }
            let payload: Value = serde_json::from_slice(&inspected.stdout).ok()?;
            let scoped = payload
                .as_array()?
                .first()?
                .get("Mounts")?
                .as_array()?
                .iter()
                .any(|mount| {
                    mount["Destination"].as_str() == Some("/tasktrace-profile-output")
                        && mount["Source"]
                            .as_str()
                            .map(PathBuf::from)
                            .and_then(|source| source.canonicalize().ok())
                            .is_some_and(|source| expected.contains(&source))
                });
            scoped.then(|| id.to_owned())
        })
        .collect()
}

fn assert_artifact_descriptor(descriptor: &Value) {
    let path = PathBuf::from(
        descriptor["path"]
            .as_str()
            .expect("artifact descriptor has a path"),
    );
    let bytes = fs::read(&path)
        .unwrap_or_else(|error| panic!("read retained artifact {}: {error}", path.display()));
    assert_eq!(descriptor["size_bytes"].as_u64(), Some(bytes.len() as u64));
    let digest = format!("{:x}", Sha256::digest(&bytes));
    assert_eq!(descriptor["sha256"].as_str(), Some(digest.as_str()));
}

fn assert_mode_output(directory: &Path, mode: &str, expected_image_id: &str, coordinated: bool) {
    let report: Value = serde_json::from_slice(
        &fs::read(directory.join("profile_report.json")).expect("read profile report"),
    )
    .expect("parse profile report");
    assert_eq!(report["strace"]["mode"], mode);
    assert_eq!(report["command_exit_code"], 0);
    assert_eq!(report["image_identity"]["id"], expected_image_id);
    assert!(directory.join("process.strace.log").is_file());
    let raw = fs::read_to_string(directory.join("process.strace.log"))
        .expect("read retained raw strace log");
    assert!(raw.contains("execve("), "raw trace omitted execve records");
    assert_artifact_descriptor(&report["strace"]["raw_artifact"]);
    if mode == "process" {
        assert!(directory.join("process_exec_report.json").is_file());
        assert_eq!(
            report["process_exec"]["schema_version"],
            "software-deep-process-exec.v1"
        );
        assert_eq!(report["process_exec"]["valid"], true);
        assert!(report["process_exec"]["summary"]["process_count"]
            .as_u64()
            .is_some_and(|count| count >= 1));
        assert!(report["process_exec"]["summary"]["successful_exec_count"]
            .as_u64()
            .is_some_and(|count| count >= 1));
        assert!(report["strace"]["structured_report_path"].is_string());
        assert_artifact_descriptor(&report["strace"]["structured_artifact"]);
    } else {
        assert!(!directory.join("process_exec_report.json").exists());
        assert!(report["process_exec"].is_null());
        assert!(report["strace"]["structured_report_path"].is_null());
        assert!(report["strace"]["structured_artifact"].is_null());
    }
    if coordinated {
        assert_eq!(
            report["evidence_bundle"]["profile_sets"],
            json!(["software-deep"])
        );
        assert!(report["evidence_bundle"]["resolved_profiler_ids"]
            .as_array()
            .is_some_and(|ids| ids.iter().any(|id| id == "process-exec")));
    } else {
        assert!(report.get("evidence_bundle").is_none());
    }
}

fn write_adapter_fixtures(root: &Path) -> (PathBuf, PathBuf, PathBuf) {
    let atif = root.join("trajectory.json");
    fs::write(
        &atif,
        serde_json::to_vec(&json!({
            "schema_version": "ATIF-v1.5",
            "session_id": "process-adapter-e2e",
            "agent": {"name": "codex"},
            "steps": [{
                "step_id": 1,
                "timestamp": "2026-07-06T00:00:00+00:00",
                "tool_calls": [{
                    "function_name": "exec_command",
                    "arguments": {"cmd": "printf atif-process-adapter >/tmp/atif-process-adapter"},
                }],
                "observation": {"content": "Process exited with code 0"},
            }],
        }))
        .expect("encode ATIF process fixture"),
    )
    .expect("write ATIF process fixture");

    let trace = root.join("trace.jsonl");
    let tasktrace_records = [
        json!({
            "record_type": "trace",
            "schema_version": "tasktrace-cpu-trace.v1",
            "task_name": "process-adapter-e2e",
            "source_trial": "process-adapter-e2e__trial",
            "environment": {"action_timeout_sec": 5.0},
        }),
        json!({
            "record_type": "action",
            "action_index": 1,
            "command": "printf tasktrace-process-adapter >/tmp/tasktrace-process-adapter",
            "started_at_epoch": 1.0,
            "finished_at_epoch": 1.01,
            "wall_time_sec": 0.01,
            "returncode": 0,
            "exception_info": null,
        }),
    ];
    fs::write(
        &trace,
        tasktrace_records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .expect("write TaskTrace process fixture");

    let transcript = root.join("transcript.jsonl");
    fs::write(
        &transcript,
        json!({
            "task_id": "process-adapter-e2e",
            "trial_name": "process-adapter-e2e.1-of-1.default",
            "answer": "[Tool Call: exec] {\"command\":\"printf transcript-process-adapter >/tmp/transcript-process-adapter\"}",
        })
        .to_string()
            + "\n",
    )
    .expect("write transcript process fixture");
    (atif, trace, transcript)
}

#[test]
#[ignore = "requires Docker, cgroup v2, runner-owned CPUs, and an image containing strace"]
fn native_disposable_execve_and_process_baselines_retain_trace_artifacts() {
    let image = env::var("TASKTRACE_RUST_STRACE_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_STRACE_DOCKER_TEST_IMAGE to a preloaded immutable image containing strace");
    assert!(
        immutable_digest(&image),
        "Docker test image must be an immutable sha256 image ID or repo digest"
    );
    assert!(
        image_has_strace(&image),
        "strace fixture image has no strace executable"
    );
    let expected_image_id = local_image_id(&image);
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let temporary = TempDir::new().expect("create process trace E2E root");
    let output_directories = [
        temporary.path().join("execve"),
        temporary.path().join("process"),
    ];

    for mode in ["execve", "process"] {
        let directory = temporary.path().join(mode);
        let result = output(
            Command::new("timeout")
                .args(["--signal=TERM", "--kill-after=5s", "60s"])
                .arg(binary())
                .args(["profile", "container", "--image"])
                .arg(&image)
                .args([
                    "--command",
                    "/bin/sh -c 'printf traced >/tmp/tasktrace-process-trace-e2e'",
                    "--cpuset-cpus",
                ])
                .arg(&cpuset)
                .args([
                    "--workdir",
                    "/tmp",
                    "--shell",
                    "/bin/sh",
                    "--network-policy",
                    "no-network",
                    "--strace-mode",
                    mode,
                    "--output-dir",
                ])
                .arg(&directory)
                .env(
                    "TASKTRACE_LEGACY_PYTHON",
                    temporary.path().join("python-fallback-must-not-run"),
                ),
            &format!("run native {mode} baseline"),
        );
        assert_success(&result, &format!("native {mode} baseline"));
        assert_mode_output(&directory, mode, &expected_image_id, false);
    }

    assert!(
        scoped_profile_containers(&output_directories).is_empty(),
        "native process tracing leaked a container scoped to its output mounts"
    );
}

#[test]
#[ignore = "requires Docker, cgroup v2, runner-owned CPUs, and an image containing strace"]
fn native_adapter_process_modes_reuse_the_container_tracing_lifecycle() {
    let image = env::var("TASKTRACE_RUST_STRACE_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_STRACE_DOCKER_TEST_IMAGE to a preloaded immutable image containing strace");
    assert!(immutable_digest(&image));
    assert!(image_has_strace(&image));
    let expected_image_id = local_image_id(&image);
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let temporary = TempDir::new().expect("create adapter process E2E root");
    let (atif, trace, transcript) = write_adapter_fixtures(temporary.path());
    let atif_output = temporary.path().join("atif-output");
    let tasktrace_output = temporary.path().join("tasktrace-output");
    let transcript_output = temporary.path().join("transcript-output");
    let missing_python = temporary.path().join("python-fallback-must-not-run");

    let atif_result = output(
        Command::new("timeout")
            .args(["--signal=TERM", "--kill-after=5s", "60s"])
            .arg(binary())
            .args(["profile", "atif"])
            .arg(&atif)
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .args([
                "--controller-runtime",
                "rust-host-mount",
                "--pacing",
                "unpaced",
                "--workdir",
                "/tmp",
                "--shell",
                "/bin/sh",
                "--network-policy",
                "no-network",
                "--profile-set",
                "software-deep",
                "--output-dir",
            ])
            .arg(&atif_output)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python),
        "run native ATIF software-deep profile",
    );
    assert_success(&atif_result, "native ATIF software-deep profile");
    assert_mode_output(
        &atif_output.join("container_profile"),
        "process",
        &expected_image_id,
        true,
    );

    let tasktrace_result = output(
        Command::new("timeout")
            .args(["--signal=TERM", "--kill-after=5s", "60s"])
            .arg(binary())
            .args(["profile", "trace"])
            .arg(&trace)
            .arg("--image")
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .args([
                "--pacing",
                "unpaced",
                "--workdir",
                "/tmp",
                "--shell",
                "/bin/sh",
                "--network-policy",
                "no-network",
                "--strace-mode",
                "process",
                "--output-dir",
            ])
            .arg(&tasktrace_output)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python),
        "run native TaskTrace process profile",
    );
    assert_success(&tasktrace_result, "native TaskTrace process profile");
    assert_mode_output(
        &tasktrace_output.join("container_profile"),
        "process",
        &expected_image_id,
        false,
    );

    let transcript_result = output(
        Command::new("timeout")
            .args(["--signal=TERM", "--kill-after=5s", "60s"])
            .arg(binary())
            .args(["profile", "transcript"])
            .arg(&transcript)
            .args([
                "--task-id",
                "process-adapter-e2e",
                "--trial-name",
                "process-adapter-e2e.1-of-1.default",
                "--image",
            ])
            .arg(&image)
            .arg("--cpuset-cpus")
            .arg(&cpuset)
            .args([
                "--workdir",
                "/tmp",
                "--shell",
                "/bin/sh",
                "--network-policy",
                "no-network",
                "--output-dir",
            ])
            .arg(&transcript_output)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python),
        "run native transcript default-execve profile",
    );
    assert_success(
        &transcript_result,
        "native transcript default-execve profile",
    );
    assert_mode_output(
        &transcript_output.join("container_profile"),
        "execve",
        &expected_image_id,
        false,
    );

    assert!(!missing_python.exists());
    let profile_outputs = [
        atif_output.join("container_profile"),
        tasktrace_output.join("container_profile"),
        transcript_output.join("container_profile"),
    ];
    assert!(scoped_profile_containers(&profile_outputs).is_empty());
}
