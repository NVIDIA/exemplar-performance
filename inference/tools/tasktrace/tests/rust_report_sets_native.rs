// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde_json::json;
use std::env;
use std::fs;
use std::process::Command;
use tempfile::TempDir;

#[test]
fn public_render_set_and_nested_render_batch_are_python_independent() {
    let root = TempDir::new().unwrap();
    let replay = root.path().join("replay.json");
    fs::write(
        &replay,
        serde_json::to_vec(&json!({
            "actions": [{
                "action_index": 1,
                "tool": "exec_command",
                "command": "python3 work.py",
                "session_id": "session",
                "status": "ok",
                "source_status": "ok",
                "replay_start_monotonic_ns": 100,
                "replay_end_monotonic_ns": 900,
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let events = root.path().join("events.jsonl");
    fs::write(
        &events,
        b"{\"timestamp_ns\":200,\"cpu\":1,\"tid\":1,\"entering\":true}\n{\"timestamp_ns\":800,\"cpu\":1,\"tid\":1,\"entering\":false}\n",
    )
    .unwrap();
    let pipeline = root.path().join("pipeline.json");
    fs::write(
        pipeline,
        serde_json::to_vec(&json!({
            "task_name": "task",
            "source_trial": "task__trial",
            "replay_report": replay,
            "replay_summary": {"source_outcome_mismatch_count": 0},
            "container_cpu": {
                "wall_sec": 1.0,
                "usage_sec": 0.8,
                "average_logical_cores_used": 0.8,
                "allocated_logical_cpus": 1.0,
            },
            "scheduler_concurrency": {
                "average_logical_cores_used": 0.8,
                "peak_simultaneous_logical_cores": 1,
                "time_sec_by_simultaneous_logical_cores": {"0": 0.2, "1": 0.8},
                "lost_perf_events": 0,
                "events_path": events,
                "cgroup_validation": {"valid_for_concurrency_reporting": true},
            },
        }))
        .unwrap(),
    )
    .unwrap();
    let manifest = root.path().join("sets.json");
    fs::write(
        &manifest,
        serde_json::to_vec(&json!({
            "schema_version": "scheduler-report-sets.v1",
            "reports": {"task": "pipeline.json"},
            "sets": [{
                "name": "all",
                "output_dir": "rendered",
                "report_ids": ["task"],
            }],
        }))
        .unwrap(),
    )
    .unwrap();

    let output = Command::new(env!("CARGO_BIN_EXE_tasktrace"))
        .args(["report", "render-set"])
        .arg(&manifest)
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            "/definitely/missing/python-for-native-render-set",
        )
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "stdout:\n{}\nstderr:\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout).contains("Report set: all"));
    for name in [
        "batch_summary.json",
        "batch_summary.csv",
        "average_vs_peak.svg",
        "core_distribution.svg",
    ] {
        assert!(root.path().join("rendered").join(name).is_file(), "{name}");
    }
}
