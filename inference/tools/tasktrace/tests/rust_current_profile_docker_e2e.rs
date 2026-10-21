// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in restricted-container gate for the native current-container runtime.
//!
//! This is deliberately separate from the Kubernetes gate: it proves the
//! non-root/read-only cgroup contract in a real container, while Kubernetes
//! remains responsible for the scheduler-created pod and immutable manifest.

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
        .canonicalize()
        .expect("canonicalize native tasktrace binary")
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

fn run(command: &mut Command, description: &str) -> Output {
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

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(path)
            .unwrap_or_else(|error| panic!("unable to read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("unable to parse {}: {error}", path.display()))
}

fn cpu_stat_usage(path: &Path) -> u64 {
    fs::read_to_string(path)
        .expect("read retained cpu.stat")
        .lines()
        .find_map(|line| {
            let (name, value) = line.split_once(' ')?;
            (name == "usage_usec").then(|| value.trim().parse::<u64>().ok())?
        })
        .expect("retained cpu.stat has usage_usec")
}

#[test]
#[ignore = "requires Docker, cgroup v2, and a preloaded immutable native image"]
fn native_current_profile_is_valid_nonroot_readonly_and_python_independent() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    assert!(
        immutable_digest(&image),
        "TASKTRACE_RUST_DOCKER_TEST_IMAGE must be an immutable sha256 digest"
    );
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let server = run(
        Command::new("docker").args(["version", "--format", "{{.Server.Version}}"]),
        "query Docker server",
    );
    assert_success(&server, "Docker server preflight");

    let root = TempDir::new().expect("create current-container E2E root");
    fs::set_permissions(root.path(), fs::Permissions::from_mode(0o755)).unwrap();
    let workspace = root.path().join("workspace");
    let results = root.path().join("results");
    fs::create_dir(&workspace).unwrap();
    fs::create_dir(&results).unwrap();
    fs::set_permissions(&workspace, fs::Permissions::from_mode(0o777)).unwrap();
    fs::set_permissions(&results, fs::Permissions::from_mode(0o777)).unwrap();
    let trajectory = root.path().join("trajectory.json");
    fs::write(
        &trajectory,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "ATIF-v1.5",
            "session_id": "native-current-container-e2e",
            "agent": {"name": "fixture", "version": "1", "model_name": "none"},
            "steps": [{
                "step_id": 1,
                "timestamp": "2026-07-06T00:00:00Z",
                "tool_calls": [{
                    "tool_call_id": "busy-shell",
                    "function_name": "exec_command",
                    "arguments": {
                        "cmd": "i=0; while [ \"$i\" -lt 250000 ]; do i=$((i+1)); done; printf 'native-current\\n' > result.txt",
                        "yield_time_ms": 1000,
                        "max_output_tokens": 12000
                    }
                }],
                "observation": {"results": [{
                    "source_call_id": "busy-shell",
                    "content": "Chunk ID: fixture\\nWall time: 0.2 seconds\\nProcess exited with code 0\\nFinal output:\\n"
                }]}
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    fs::set_permissions(&trajectory, fs::Permissions::from_mode(0o644)).unwrap();

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let name = format!("tasktrace-current-e2e-{}-{nonce}", std::process::id());
    let host_uid = unsafe { libc::geteuid() };
    let host_gid = unsafe { libc::getegid() };
    assert_ne!(
        host_uid, 0,
        "the restricted gate must execute as a non-root UID"
    );
    let container_user = format!("{host_uid}:{host_gid}");
    let output = run(
        Command::new("docker").args([
            "run",
            "--rm",
            "--name",
            &name,
            "--label",
            "tasktrace.test=current-container",
            "--read-only",
            "--user",
            &container_user,
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges",
            "--network",
            "none",
            "--pids-limit",
            "64",
            "--memory",
            "256m",
            "--cpus",
            "1",
            "--cpuset-cpus",
            &cpuset,
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=16m",
            "--mount",
            &format!(
                "type=bind,src={},dst=/controller/bin/tasktrace,readonly",
                binary().display()
            ),
            "--mount",
            &format!(
                "type=bind,src={},dst=/inputs/trajectory.json,readonly",
                trajectory.display()
            ),
            "--mount",
            &format!("type=bind,src={},dst=/workspace", workspace.display()),
            "--mount",
            &format!("type=bind,src={},dst=/results", results.display()),
            "--workdir",
            "/workspace",
            "--env",
            "PATH=/controller/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin",
            "--env",
            "VIRTUAL_ENV=/controller",
            "--env",
            "VIRTUAL_ENV_PROMPT=(controller)",
            "--env",
            "PYTHONPATH=/controller/source:/task-preserve",
            "--env",
            "UV_RUN_RECURSION_DEPTH=1",
            "--env",
            "TASKTRACE_LEGACY_PYTHON=/python-must-not-run",
            "--entrypoint",
            "/controller/bin/tasktrace",
            &image,
            "profile",
            "atif",
            "/inputs/trajectory.json",
            "--runtime",
            "current",
            "--current-container-exclusive",
            "--allocated-cpus",
            "1",
            "--pacing",
            "unpaced",
            "--workdir",
            "/workspace",
            "--shell",
            "/bin/bash",
            "--strace-mode",
            "none",
            "--timeout-sec",
            "10",
            "--patch-root",
            "/workspace",
            "--output-dir",
            "/results/profile",
        ]),
        "run restricted current-container profile",
    );
    assert_success(&output, "restricted current-container profile");

    assert_eq!(
        fs::read_to_string(workspace.join("result.txt")).unwrap(),
        "native-current\n"
    );
    let pipeline = read_json(&results.join("profile/pipeline_report.json"));
    let profile = read_json(&results.join("profile/container_profile/profile_report.json"));
    assert_eq!(
        pipeline["schema_version"],
        "container-atif-trajectory-profile.v1"
    );
    assert_eq!(pipeline["runtime"], "current");
    assert_eq!(pipeline["measurement_scope"], "current-container");
    assert_eq!(pipeline["status"], "valid");
    assert_eq!(
        pipeline["controller_identity"]["kind"],
        "rust-current-container-binary"
    );
    assert_eq!(
        profile["measurement_validity"]["valid_for_average_cpu_reporting"],
        true
    );
    assert_eq!(
        profile["measurement_validity"]["exclusive_single_task_container_asserted"],
        true
    );
    assert_eq!(profile["measurement_validity"]["cgroup_path_stable"], true);
    assert_eq!(profile["measurement_validity"]["allocation_stable"], true);
    assert_eq!(
        profile["task_environment"]["unset_variables"],
        json!([
            "PYTHONPATH",
            "UV_RUN_RECURSION_DEPTH",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT"
        ])
    );
    assert_eq!(
        profile["task_environment"]["overridden_variables"],
        json!(["PATH", "PYTHONPATH"])
    );
    assert!(profile["command"]
        .as_str()
        .unwrap()
        .contains("/controller/bin/tasktrace replay atif"));
    assert!(!profile["command"].as_str().unwrap().contains("python"));

    let before = cpu_stat_usage(&results.join("profile/container_profile/cpu.stat.before"));
    let after = cpu_stat_usage(&results.join("profile/container_profile/cpu.stat.after"));
    let retained_delta = after
        .checked_sub(before)
        .expect("monotonic CPU usage counter");
    assert!(retained_delta > 0);
    let reported_delta = profile["cpu"]["raw_cpu_stat_delta"]["usage_usec"]
        .as_u64()
        .unwrap();
    assert_eq!(reported_delta, retained_delta);
    let usage_sec = profile["cpu"]["usage_sec"].as_f64().unwrap();
    assert!((usage_sec - retained_delta as f64 / 1_000_000.0).abs() <= 0.000_000_5);
    assert_eq!(profile["cpu"]["allocated_logical_cpus"], 1.0);

    let survivors = run(
        Command::new("docker").args([
            "ps",
            "--all",
            "--quiet",
            "--filter",
            &format!("name=^{name}$"),
        ]),
        "check current-container E2E cleanup",
    );
    assert_success(&survivors, "Docker cleanup inspection");
    assert!(
        survivors.stdout.is_empty(),
        "Docker left the current-container test sandbox behind"
    );
}
