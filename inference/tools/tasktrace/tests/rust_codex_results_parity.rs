// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Release 26.10 rejects Codex exec before runtime or filesystem side effects.
use serde_json::{json, Value};
use std::{env, fs, path::Path, process::Command};

fn codex(tool: &str, arguments: Value) -> Value {
    json!({"schema_version":"ATIF-v1.7", "trajectory_id":"codex",
        "agent":{"name":"codex","extra":{"originator":"codex_exec"}},
        "steps":[{"timestamp":0, "tool_calls":[{
            "tool_call_id":"a", "function_name":tool, "arguments":arguments
        }]}]
    })
}

fn nested(child: Value) -> Value {
    json!({"schema_version":"ATIF-v1.7", "trajectory_id":"root", "agent":{"name":"generic"},
        "steps":[{"timestamp":0, "tool_calls":[{
            "function_name":"exec_command", "arguments":{"cmd":"touch actual"}
        }]}], "subagent_trajectories":[child]
    })
}

fn cases() -> Vec<Value> {
    vec![
        codex("exec_command", json!({"cmd":"touch actual"})),
        codex(
            "write_stdin",
            json!({"session_id":44,"chars":"touch actual\n"}),
        ),
        codex(
            "apply_patch",
            json!({"input":"*** Begin Patch\n*** Add File: actual\n+changed\n*** End Patch"}),
        ),
        codex(
            "exec",
            json!({"input":"await tools.exec_command({cmd:'touch actual'});"}),
        ),
        codex(
            "exec",
            json!({"input":"await Promise.all([tools.exec_command({cmd:'touch actual'})]);"}),
        ),
        codex("update_plan", json!({"plan":[]})),
    ]
}

fn assert_replay_rejected(payload: &Value, native: bool) {
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("trajectory.json");
    fs::write(&input, payload.to_string()).unwrap();
    let mut command = if native {
        let mut command = Command::new(env!("CARGO_BIN_EXE_tasktrace"));
        command.args(["replay", "atif"]);
        command.env("TASKTRACE_LEGACY_PYTHON", "/must-not-use-python-fallback");
        command
    } else {
        let mut command = Command::new(
            env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| "python3".into()),
        );
        command.args(["-m", "offline_replayer.adapters.atif_replay"]);
        command.env("PYTHONPATH", env!("CARGO_MANIFEST_DIR"));
        command
    };
    let output = command
        .arg(input)
        .arg("--output")
        .arg(root.path().join("report.json"))
        .arg("--cwd")
        .arg(root.path())
        .args(["--pacing", "unpaced"])
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("Codex exec ATIF replay and profiling are not supported in release 26.10"),
        "{:?}",
        output
    );
    assert_eq!(
        fs::read_dir(root.path()).unwrap().count(),
        1,
        "rejection must not create files or artifacts"
    );
}

#[test]
fn codex_exec_root_and_nested_calls_are_rejected_in_both_readers() {
    for payload in cases() {
        for document in [payload.clone(), nested(payload)] {
            for native in [false, true] {
                assert_replay_rejected(&document, native);
            }
        }
    }
    let mut empty = codex("exec_command", json!({"cmd":"touch actual"}));
    empty["steps"] = json!([]);
    for native in [false, true] {
        assert_replay_rejected(&nested(empty.clone()), native);
    }
}

#[test]
fn direct_codex_exec_fixture_is_rejected_in_both_readers() {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("offline_replayer/tests/fixtures/representative_atif/swebenchpro-codex__fixture/agent/trajectory.json");
    let payload: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    for native in [false, true] {
        assert_replay_rejected(&payload, native);
    }
}

#[test]
fn codex_exec_profile_is_rejected_before_runtime_or_output_creation() {
    use std::os::unix::fs::PermissionsExt;
    for payload in [cases().remove(0), nested(cases().remove(0))] {
        let root = tempfile::tempdir().unwrap();
        let input = root.path().join("trajectory.json");
        fs::write(&input, payload.to_string()).unwrap();
        let docker = root.path().join("docker");
        let marker = root.path().join("docker-called");
        fs::write(
            &docker,
            format!("#!/bin/sh\ntouch '{}'\nexit 99\n", marker.display()),
        )
        .unwrap();
        fs::set_permissions(&docker, fs::Permissions::from_mode(0o755)).unwrap();
        let output_dir = root.path().join("profile");
        let output = Command::new(env!("CARGO_BIN_EXE_tasktrace"))
            .args(["profile", "atif"])
            .arg(input)
            .args([
                "--image",
                "unused:latest",
                "--allocated-cpus",
                "1",
                "--output-dir",
            ])
            .arg(&output_dir)
            .arg("--docker")
            .arg(docker)
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr).contains(
                "Codex exec ATIF replay and profiling are not supported in release 26.10"
            ),
            "{:?}",
            output
        );
        assert!(
            !marker.exists(),
            "unsupported replay must not launch Docker"
        );
        assert!(
            !output_dir.exists(),
            "unsupported replay must not create output"
        );
    }
}
