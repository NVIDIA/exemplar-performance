// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-call mini-SWE evidence and unsupported-agent rejection agree in both readers.
use serde_json::{json, Value};
use std::{env, fs, path::Path, process::Command};

fn mini() -> Value {
    json!({
        "schema_version":"ATIF-v1.5",
        "agent":{"name":"mini-swe-agent","extra":{"original_format":"mini-swe-agent-1.1"}},
        "steps":[{"step_id":1,"timestamp":0,
            "tool_calls":[
                {"tool_call_id":"a","function_name":"bash","arguments":{"command":"exit 127"}},
                {"tool_call_id":"b","function_name":"bash","arguments":{"command":"printf done > actual"}}
            ],
            "observation":{"results":[
                {"source_call_id":"b","content":"{\"returncode\":0,\"output\":\"\"}"},
                {"source_call_id":"a","content":"{\"returncode\":127,\"output\":\"\"}"}
            ]}
        }]
    })
}

fn run(root: &Path, payload: &Value, native: bool, accepted: bool) -> Value {
    let work = root.join(if native { "rust" } else { "python" });
    fs::create_dir_all(&work).unwrap();
    let input = work.join("trajectory.json");
    let report = work.join("report.json");
    fs::write(&input, payload.to_string()).unwrap();
    let mut command = Command::new("timeout");
    command.args(["--signal=KILL", "--kill-after=2s", "15s"]);
    if native {
        command
            .arg(env!("CARGO_BIN_EXE_tasktrace"))
            .args(["replay", "atif"]);
        command.env("TASKTRACE_LEGACY_PYTHON", "/must-not-use-python-fallback");
    } else {
        command
            .arg(env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| "python3".into()))
            .args(["-m", "offline_replayer.adapters.atif_replay"])
            .env("PYTHONPATH", env!("CARGO_MANIFEST_DIR"));
    }
    let output = command
        .arg(input)
        .arg("--output")
        .arg(&report)
        .arg("--cwd")
        .arg(&work)
        .args(["--pacing", "unpaced", "--timeout-sec", "2"])
        .output()
        .unwrap();
    assert_eq!(
        output.status.success(),
        accepted,
        "native={native}: {}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    if !accepted {
        assert!(
            !report.exists(),
            "invalid evidence must fail before executing commands"
        );
        assert!(!work.join("actual").exists());
        return json!({"error": String::from_utf8_lossy(&output.stderr)});
    }
    let report: Value = serde_json::from_slice(&fs::read(report).unwrap()).unwrap();
    assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
    assert_eq!(
        report["summary"]["source_outcome_comparison_unavailable_count"],
        0
    );
    json!({
        "actions":report["actions"].as_array().unwrap().iter().map(|action| json!({
            "source_status":action["source_status"],"source_returncode":action["source_returncode"],
            "outcome_matches_source":action["outcome_matches_source"]
        })).collect::<Vec<_>>(),
        "actual":fs::read_to_string(work.join("actual")).ok()
    })
}

#[test]
fn mini_swe_matches_call_ids_and_retains_legacy_singletons() {
    let mut extra = mini();
    extra["steps"][0]["observation"]["results"]
        .as_array_mut()
        .unwrap()
        .push(json!({"content":"unbound diagnostic"}));
    let mut singleton = mini();
    singleton["schema_version"] = json!("ATIF-v1.2");
    singleton["steps"][0]["tool_calls"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    let results = singleton["steps"][0]["observation"]["results"]
        .as_array_mut()
        .unwrap();
    results.pop();
    results[0].as_object_mut().unwrap().remove("source_call_id");
    for payload in [mini(), extra, singleton] {
        let root = tempfile::tempdir().unwrap();
        let native = run(root.path(), &payload, true, true);
        let python = run(root.path(), &payload, false, true);
        assert_eq!(native, python);
        assert_eq!(native["actual"], "done");
        let actions = native["actions"].as_array().unwrap();
        assert_eq!(actions.last().unwrap()["source_returncode"], 0);
        if actions.len() == 2 {
            assert_eq!(actions[0]["source_returncode"], 127);
        }
    }
}

#[test]
fn mini_swe_rejects_ambiguous_and_invalid_evidence_before_replay() {
    let mut cases = Vec::new();
    let mut duplicate = mini();
    let result = duplicate["steps"][0]["observation"]["results"][0].clone();
    duplicate["steps"][0]["observation"]["results"]
        .as_array_mut()
        .unwrap()
        .push(result);
    cases.push(duplicate);
    let mut missing = mini();
    missing["steps"][0]["observation"]["results"]
        .as_array_mut()
        .unwrap()
        .remove(0);
    cases.push(missing);
    let mut missing_id = mini();
    missing_id["steps"][0]["observation"]["results"][0]
        .as_object_mut()
        .unwrap()
        .remove("source_call_id");
    cases.push(missing_id);
    let mut duplicate_call = mini();
    duplicate_call["steps"][0]["tool_calls"][1]["tool_call_id"] = json!("a");
    cases.push(duplicate_call);
    for content in [
        "not JSON",
        "[]",
        "{}",
        "{\"returncode\":true}",
        "{\"returncode\":\"0\"}",
        "{\"returncode\":0.5}",
        "{\"returncode\":null}",
        "{\"returncode\":0,\"exception_info\":\"timed out\"}",
        "{\"returncode\":0,\"exception_info\":\"action was not executed\"}",
        "{\"returncode\":127,\"exception_info\":\"action was not executed\"}",
        "{\"returncode\":1,\"exception_info\":false}",
    ] {
        let mut invalid = mini();
        invalid["steps"][0]["observation"]["results"][0]["content"] = json!(content);
        cases.push(invalid);
    }
    for payload in cases {
        let root = tempfile::tempdir().unwrap();
        run(root.path(), &payload, true, false);
        run(root.path(), &payload, false, false);
    }
}

fn claude(command: &str, result: Value) -> Value {
    json!({"schema_version":"ATIF-v1.5","agent":{"name":"claude-code"},"steps":[{
        "step_id":1,"timestamp":0,
        "tool_calls":[{"tool_call_id":"a","function_name":"Bash","arguments":{"command":command}}],
        "observation":{"results":[result]}
    }]})
}

#[test]
fn claude_replay_is_rejected_before_execution_in_both_readers() {
    let payload = claude(
        "touch actual",
        json!({"source_call_id":"a", "extra":{"tool_result_is_error":false}}),
    );
    for nested in [false, true] {
        let mut document = payload.clone();
        if nested {
            document["trajectory_id"] = json!("child");
            document = json!({"schema_version":"ATIF-v1.7", "trajectory_id":"root",
                "agent":{"name":"codex"}, "steps":[], "subagent_trajectories":[document]});
        }
        for native in [false, true] {
            let root = tempfile::tempdir().unwrap();
            let result = run(root.path(), &document, native, false);
            assert!(result["error"]
                .as_str()
                .unwrap()
                .contains("not supported in release 26.10"));
        }
    }
}

#[test]
fn claude_profile_is_rejected_before_runtime_or_output_creation() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let input = root.path().join("trajectory.json");
    fs::write(
        &input,
        claude("touch actual", json!({"source_call_id":"a","content":"ok"})).to_string(),
    )
    .unwrap();
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
    assert!(String::from_utf8_lossy(&output.stderr).contains("not supported in release 26.10"));
    assert!(
        !marker.exists(),
        "unsupported replay must not launch Docker"
    );
    assert!(
        !output_dir.exists(),
        "unsupported replay must not create output"
    );
}
