// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../src/scheduler_batch.rs"]
mod scheduler_batch;

use serde_json::json;
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;
use tempfile::TempDir;

fn python() -> OsString {
    env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn python_path() -> OsString {
    let mut paths = vec![PathBuf::from(env!("CARGO_MANIFEST_DIR"))];
    if let Some(existing) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).unwrap()
}

fn report(directory: &Path, task: &str, offset: i64) -> PathBuf {
    let replay = directory.join(format!("{task}-replay.json"));
    let events = directory.join(format!("{task}-events.jsonl"));
    fs::write(
        &replay,
        serde_json::to_string(&json!({
            "actions": [
                {
                    "action_index": 1,
                    "tool": "exec_command",
                    "command": "python3 work.py",
                    "session_id": "session",
                    "status": "running",
                    "source_status": "running",
                    "replay_start_monotonic_ns": 100 + offset,
                    "replay_end_monotonic_ns": 250 + offset,
                },
                {
                    "action_index": 2,
                    "tool": "write_stdin",
                    "session_id": "session",
                    "status": "ok",
                    "source_status": "ok",
                    "replay_start_monotonic_ns": 300 + offset,
                    "replay_end_monotonic_ns": 600 + offset,
                }
            ]
        }))
        .unwrap(),
    )
    .unwrap();
    let event_rows = [
        json!({"timestamp_ns": 150 + offset, "cpu": 1, "tid": 1, "entering": true}),
        json!({"timestamp_ns": 200 + offset, "cpu": 2, "tid": 2, "entering": true}),
        json!({"timestamp_ns": 400 + offset, "cpu": 2, "tid": 2, "entering": false}),
        json!({"timestamp_ns": 550 + offset, "cpu": 1, "tid": 1, "entering": false}),
    ];
    fs::write(
        &events,
        event_rows
            .iter()
            .map(|row| serde_json::to_string(row).unwrap())
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    let pipeline = directory.join(format!("{task}-pipeline.json"));
    fs::write(
        &pipeline,
        serde_json::to_string(&json!({
            "task_name": task,
            "source_trial": format!("{task}__trial"),
            "replay_report": replay,
            "replay_summary": {"source_outcome_mismatch_count": 0},
            "container_cpu": {
                "wall_sec": 2.0,
                "usage_sec": 2.4,
                "average_logical_cores_used": 1.2,
                "allocated_logical_cpus": 8.0,
            },
            "scheduler_concurrency": {
                "average_logical_cores_used": 1.19,
                "peak_simultaneous_logical_cores": 3,
                "time_sec_by_simultaneous_logical_cores": {
                    "0": 0.1,
                    "1": 1.4,
                    "2": 0.4,
                    "3": 0.1,
                },
                "lost_perf_events": 0,
                "events_path": events,
                "cgroup_validation": {"valid_for_concurrency_reporting": true},
            },
        }))
        .unwrap(),
    )
    .unwrap();
    pipeline
}

#[test]
fn json_csv_and_svg_outputs_match_python_bytes() {
    let directory = TempDir::new().unwrap();
    let a = report(directory.path(), "task-a", 0);
    let b = report(directory.path(), "task-b", 1_000);
    let python_output = directory.path().join("python");
    let rust_output = directory.path().join("rust");
    let status = Command::new(python())
        .arg("-m")
        .arg("offline_replayer.reporting.scheduler_batch")
        .arg(&a)
        .arg(&b)
        .arg("--output-dir")
        .arg(&python_output)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .env("PYTHONPATH", python_path())
        .status()
        .unwrap();
    assert!(status.success());
    let native = Command::new(env!("CARGO_BIN_EXE_tasktrace"))
        .args(["report", "render-batch"])
        .arg(&a)
        .arg(&b)
        .arg("--output-dir")
        .arg(&rust_output)
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            "/definitely/missing/python-for-native-render-batch",
        )
        .output()
        .unwrap();
    assert!(
        native.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&native.stdout),
        String::from_utf8_lossy(&native.stderr)
    );

    for name in [
        "batch_summary.json",
        "batch_summary.csv",
        "average_vs_peak.svg",
        "core_distribution.svg",
        "bash_commands.json",
        "bash_commands.csv",
        "bash_command_frequency.json",
        "bash_command_frequency.csv",
        "tool_family_distribution.json",
        "tool_family_distribution.csv",
    ] {
        assert_eq!(
            fs::read(rust_output.join(name)).unwrap(),
            fs::read(python_output.join(name)).unwrap(),
            "artifact differs: {name}"
        );
    }
}

#[test]
fn png_mode_writes_all_seven_native_png_artifacts() {
    let directory = TempDir::new().unwrap();
    let report = report(directory.path(), "task", 0);
    let output = directory.path().join("output");
    let tasks = scheduler_batch::build_summary(&[report]).unwrap();
    let outputs = scheduler_batch::write_outputs(&tasks, &output, true).unwrap();
    for key in [
        "average_vs_peak_png",
        "core_distribution_png",
        "core_timeline_png",
        "longest_bash_commands_png",
        "burstiest_bash_commands_png",
        "bash_command_frequency_png",
        "tool_family_distribution_png",
    ] {
        let bytes = fs::read(&outputs[key]).unwrap();
        assert!(bytes.starts_with(b"\x89PNG\r\n\x1a\n"), "{key}");
        assert!(bytes.len() > 100, "{key}");
    }
}

#[test]
fn batch_rejects_incomplete_replay_without_outcome_mismatches() {
    let root = TempDir::new().unwrap();
    let path = report(root.path(), "task", 0);
    let baseline: serde_json::Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
    for (field, value) in [
        ("source_outcome_comparison_unavailable_count", json!(1)),
        ("source_outcome_comparison_unavailable_count", json!(false)),
        ("actions_launch_failed", json!(1)),
        ("actions_source_argument_parse_failed", json!(1)),
        ("actions_skipped_after_timeout", json!(1)),
        ("tool_actions_recorded", json!(2)),
    ] {
        let mut invalid = baseline.clone();
        invalid["replay_summary"][field] = value;
        fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
        assert!(scheduler_batch::load_summary(&path).is_err(), "{field}");
    }
    let mut invalid = baseline;
    invalid["status"] = json!("invalid");
    fs::write(&path, serde_json::to_vec(&invalid).unwrap()).unwrap();
    assert!(scheduler_batch::load_summary(&path).is_err());
}
