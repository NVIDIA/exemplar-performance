// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Differential BCC program and scheduler-math contract against Python.

#[path = "../src/scheduler_bcc.rs"]
mod scheduler_bcc;
#[path = "../src/scheduler_perf.rs"]
mod scheduler_perf;

use scheduler_bcc::{summarize_bcc_events, BPF_PROGRAM, EVENT_BOUNDARY_TOLERANCE_SEC};
use scheduler_perf::SchedulerEvent;
use serde_json::{json, Value};
use std::process::Command;

#[test]
fn bpf_source_and_report_shape_match_python_oracle() {
    let script = r#"
import json
from offline_replayer.measurement.scheduler_bcc import (
    BPF_PROGRAM,
    SchedulerEvent,
    summarize_scheduler_events,
)

events = [
    SchedulerEvent(100, 0, 40, True),
    SchedulerEvent(120, 1, 41, True),
    SchedulerEvent(180, 1, 41, False),
    SchedulerEvent(200, 0, 40, False),
]
report = summarize_scheduler_events(
    events,
    window_start_ns=0,
    window_end_ns=300,
    lost_events=0,
    method="ebpf-bcc-finish-task-switch-cgroup-subtree",
    scope="all tasks scheduled from the measured container cgroup or descendant cgroups",
)
print(json.dumps({"bpf_program": BPF_PROGRAM, "report": report}, sort_keys=True))
"#;
    let output = Command::new("python3")
        .args(["-c", script])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Python BCC oracle");
    assert!(
        output.status.success(),
        "Python BCC oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let oracle: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(oracle["bpf_program"], BPF_PROGRAM);

    let events = [
        SchedulerEvent {
            timestamp_ns: 100,
            cpu: 0,
            tid: 40,
            entering: true,
        },
        SchedulerEvent {
            timestamp_ns: 120,
            cpu: 1,
            tid: 41,
            entering: true,
        },
        SchedulerEvent {
            timestamp_ns: 180,
            cpu: 1,
            tid: 41,
            entering: false,
        },
        SchedulerEvent {
            timestamp_ns: 200,
            cpu: 0,
            tid: 40,
            entering: false,
        },
    ];
    let report = summarize_bcc_events(&events, 0, 300, 0).unwrap();
    assert_eq!(report, oracle["report"]);
    assert_eq!(report["peak_simultaneous_logical_cores"], json!(2));
}

#[test]
fn bcc_cgroup_reconciliation_shape_matches_python_oracle() {
    let script = r#"
import json
from offline_replayer.measurement.scheduler_bcc import add_cgroup_validation
print(json.dumps(add_cgroup_validation(
    {"scheduled_runtime_sec": 0.8, "events_observed": 4, "trace_complete": True},
    cgroup_usage_sec=0.8,
    tolerance_fraction=0.02,
    tolerance_sec=0.005,
    per_observed_event_tolerance_sec=0.000002,
), sort_keys=True))
"#;
    let output = Command::new("python3")
        .args(["-c", script])
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Python BCC cgroup-validation oracle");
    assert!(output.status.success());
    let oracle: Value = serde_json::from_slice(&output.stdout).unwrap();
    let report = scheduler_perf::add_cgroup_validation_with_event_tolerance(
        &json!({
            "scheduled_runtime_sec": 0.8,
            "events_observed": 4,
            "trace_complete": true,
        }),
        0.8,
        0.02,
        0.005,
        EVENT_BOUNDARY_TOLERANCE_SEC,
    )
    .unwrap();
    assert_eq!(report, oracle);
}
