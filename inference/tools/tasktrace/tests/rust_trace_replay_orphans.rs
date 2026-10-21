// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `tasktrace replay trace` adopting the helpers its actions orphan when it runs as a child
//! subreaper, which is how the current-container profiler launches it.

use serde_json::{json, Value};
use std::fs;
use std::os::unix::process::CommandExt;
use std::path::Path;
use std::process::Command;
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn write_trace(directory: &Path, commands: &[&str]) {
    let mut rows = vec![json!({
        "record_type": "trace",
        "schema_version": "tasktrace-cpu-trace.v1",
        "source_trial": "orphans__trial",
        "task_name": "orphans",
        "environment": {"action_timeout_sec": 30, "shell": "/bin/bash", "cwd": ""},
        "limitations": [],
        "replay_supported": true,
    })];
    for (index, command) in commands.iter().enumerate() {
        let started = 10.0 + index as f64;
        rows.push(json!({
            "record_type": "action",
            "action_index": index + 1,
            "command": command,
            "started_at_epoch": started,
            "finished_at_epoch": started + 0.5,
            "wall_time_sec": 0.5,
            "returncode": 0,
            "exception_info": null,
        }));
    }
    let lines: Vec<String> = rows.iter().map(Value::to_string).collect();
    fs::write(directory.join("trace.jsonl"), lines.join("\n") + "\n").unwrap();
}

fn replay(directory: &TempDir, as_subreaper: bool) -> Value {
    let root = directory.path();
    let mut command = Command::new(binary());
    command
        .args(["replay", "trace"])
        .arg(root.join("trace.jsonl"))
        .arg("--output")
        .arg(root.join("report.json"))
        .arg("--cwd")
        .arg(root)
        .args(["--pacing", "unpaced"]);
    if as_subreaper {
        unsafe {
            command.pre_exec(|| {
                let enable: libc::c_ulong = 1;
                let unused: libc::c_ulong = 0;
                libc::prctl(libc::PR_SET_CHILD_SUBREAPER, enable, unused, unused, unused);
                Ok(())
            });
        }
    }
    let output = command.output().unwrap();
    assert!(
        output.status.success(),
        "replay failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_str(&fs::read_to_string(root.join("report.json")).unwrap()).unwrap()
}

fn alive(pid: i32) -> bool {
    unsafe { libc::kill(pid, 0) == 0 }
}

#[test]
fn replay_as_subreaper_reaps_exited_helpers_and_kills_leftovers() {
    let directory = TempDir::new().unwrap();
    write_trace(
        directory.path(),
        &[
            "sh -c 'sleep 0.05 >/dev/null 2>&1 &'",
            "sleep 0.4",
            "sh -c 'sleep 30 >/dev/null 2>&1 & echo $! > leftover.pid'",
        ],
    );
    let report = replay(&directory, true);
    assert_eq!(report["summary"]["actions_ok"], 3);
    let adoption = &report["orphan_adoption"];
    assert_eq!(adoption["child_subreaper"], true);
    assert!(
        adoption["exited_helpers_reaped"].as_u64().unwrap() >= 1,
        "the short helper orphaned by action 1 was not reaped: {adoption}"
    );
    let leftover: i32 = fs::read_to_string(directory.path().join("leftover.pid"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let killed = adoption["helpers_killed_after_last_action"]
        .as_array()
        .unwrap();
    assert!(
        killed
            .iter()
            .any(|entry| entry["pid"] == leftover && entry["name"] == "sleep"),
        "leftover helper {leftover} missing from {adoption}"
    );
    assert!(
        !alive(leftover),
        "leftover helper {leftover} survived the replay"
    );
}

#[test]
fn replay_without_subreaper_reports_no_adoption() {
    let directory = TempDir::new().unwrap();
    write_trace(directory.path(), &["true"]);
    let report = replay(&directory, false);
    assert_eq!(report["summary"]["actions_ok"], 1);
    assert!(report.get("orphan_adoption").is_none());
}
