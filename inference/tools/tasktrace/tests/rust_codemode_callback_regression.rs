// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use std::{fs, process::Command};

#[test]
fn rejects_effectful_mapped_callbacks_before_any_replay_work() {
    for body in [
        "cmd='printf SOURCE > marker'; return tools.exec_command({cmd});",
        "return; tools.exec_command({cmd});",
        "return\n tools.exec_command({cmd});",
        "throw new Error('abort'); tools.exec_command({cmd});",
        "tools.exec_command({cmd}); cmd='changed';",
        "const r=await tools.exec_command({cmd}); return `${process.exit()}`;",
        "const cmd=await tools.exec_command({cmd}); return cmd;",
        "false && tools.exec_command({cmd});",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let trajectory = tmp.path().join("trajectory.json");
        let code = format!("await tools.exec_command({{cmd:'touch earlier_marker'}}); const cmds=['printf ORIGINAL > marker']; await Promise.all(cmds.map(async cmd=>{{{body}}}));");
        fs::write(
            &trajectory,
            serde_json::to_vec(&json!({
                "schema_version":"ATIF-v1.5", "session_id":"session",
                "agent":{"name":"codex","extra":{"originator":"codex_exec"}},
                "steps":[{"step_id":1,"timestamp":0,"tool_calls":[{
                    "tool_call_id":"call-0","function_name":"exec","arguments":{"input":code}
                }],"observation":null}]
            }))
            .unwrap(),
        )
        .unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_tasktrace"))
            .args(["replay", "atif"])
            .arg(&trajectory)
            .arg("--output")
            .arg(tmp.path().join("report.json"))
            .arg("--cwd")
            .arg(tmp.path())
            .args(["--pacing", "unpaced"])
            .output()
            .unwrap();
        assert!(!output.status.success(), "accepted {body}");
        assert!(!tmp.path().join("marker").exists(), "executed {body}");
        assert!(
            !tmp.path().join("earlier_marker").exists(),
            "partially executed {body}"
        );
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("command batch"),
            "{:?}",
            output
        );
    }
}

#[test]
fn literal_mapped_callbacks_are_rejected_before_execution() {
    for body in [
        "return tools.exec_command({cmd});",
        "const r=await tools.exec_command({cmd}); return {output:r.output,label:'literal'};",
        "const r=await tools.exec_command({cmd}); return `${cmd}:${r.output}`;",
    ] {
        let tmp = tempfile::tempdir().unwrap();
        let trajectory = tmp.path().join("trajectory.json");
        fs::write(&trajectory, serde_json::to_vec(&json!({
            "schema_version":"ATIF-v1.5", "session_id":"session",
            "agent":{"name":"codex","extra":{"originator":"codex_exec"}},
            "steps":[{"step_id":1,"timestamp":0,"tool_calls":[{
                "tool_call_id":"call-0","function_name":"exec",
                "arguments":{"input":format!("const cmds=['printf ORIGINAL > marker']; await Promise.all(cmds.map(async cmd=>{{{body}}}));")}
            }],"observation":null}]
        })).unwrap()).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_tasktrace"))
            .args(["replay", "atif"])
            .arg(&trajectory)
            .arg("--output")
            .arg(tmp.path().join("report.json"))
            .arg("--cwd")
            .arg(tmp.path())
            .args(["--pacing", "unpaced"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("not supported in release 26.10"));
        assert!(!tmp.path().join("report.json").exists());
        assert!(!tmp.path().join("marker").exists());
    }
}
