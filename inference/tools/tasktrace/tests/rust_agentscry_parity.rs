// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public-dispatch differential gate for the native AgentScry comparison.

use serde_json::json;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::PathBuf;
use std::process::{Command, Output};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python_path() -> OsString {
    let mut paths = vec![repository()];
    if let Some(existing) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).unwrap()
}

fn run(command: &mut Command) -> Output {
    command.output().expect("run AgentScry comparison")
}

#[test]
fn public_agentscry_report_matches_python_without_fallback() {
    let root = TempDir::new().unwrap();
    let pipeline = root.path().join("pipeline.json");
    fs::write(
        &pipeline,
        serde_json::to_vec(&json!({
            "task_id": "task",
            "trial_name": "task.1",
            "image_identity": {"id": "sha256:image"},
            "container_cpu": {
                "usage_sec": 1.0,
                "wall_sec": 2.0,
                "average_logical_cores_used": 0.5,
            },
        }))
        .unwrap(),
    )
    .unwrap();
    let csv = root.path().join("agentscry.csv");
    fs::write(
        &csv,
        concat!(
            "EPOCH,DURATION,COMM,PPID,PID,CPU,USER,SYS,NTHD,ARGS\n",
            "1,,sh,1,10,0,0.1,0.1,,/bin/sh /tasktrace-profile-entrypoint.sh\n",
            "2,,python,10,11,0,0.2,0.2,,/usr/bin/python3 replay.py\n",
            "3,,python,10,11,0,0.2,0.2,,/usr/bin/python3 replay.py --second-exec\n",
        ),
    )
    .unwrap();
    let native_output = root.path().join("native.json");
    let python_output = root.path().join("python.json");
    let native = run(Command::new(binary())
        .args([
            OsStr::new("report"),
            OsStr::new("compare-agentscry"),
            pipeline.as_os_str(),
            csv.as_os_str(),
            OsStr::new("--output"),
            native_output.as_os_str(),
            OsStr::new("--tolerance-fraction"),
            OsStr::new("0.01"),
        ])
        .current_dir(repository())
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            root.path().join("fallback-must-not-run"),
        ));
    let python = env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| OsString::from("python3"));
    let oracle = run(Command::new(python)
        .args([
            OsStr::new("-m"),
            OsStr::new("offline_replayer.reporting.agentscry_comparison"),
            pipeline.as_os_str(),
            csv.as_os_str(),
            OsStr::new("--output"),
            python_output.as_os_str(),
            OsStr::new("--tolerance-fraction"),
            OsStr::new("0.01"),
        ])
        .current_dir(repository())
        .env("PYTHONPATH", python_path()));
    assert_eq!(
        native.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&native.stderr)
    );
    assert_eq!(
        oracle.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&oracle.stderr)
    );
    assert_eq!(
        fs::read(native_output).unwrap(),
        fs::read(python_output).unwrap()
    );
    let native_stdout = String::from_utf8(native.stdout).unwrap().replace(
        root.path().join("native.json").to_string_lossy().as_ref(),
        "<REPORT>",
    );
    let oracle_stdout = String::from_utf8(oracle.stdout).unwrap().replace(
        root.path().join("python.json").to_string_lossy().as_ref(),
        "<REPORT>",
    );
    assert_eq!(native_stdout, oracle_stdout);
}
