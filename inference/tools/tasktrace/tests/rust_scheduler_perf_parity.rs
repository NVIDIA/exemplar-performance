// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../src/scheduler_perf.rs"]
mod scheduler_perf;

use scheduler_perf::{
    add_cgroup_validation, parse_perf_script_switch_events, summarize_scheduler_events,
};
use serde_json::{json, Value};
use std::fs;
use std::process::Command;

fn python_oracle(trace: &str, start_ns: u64, end_ns: u64, cgroup_usage: f64) -> Value {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("perf-script.txt");
    fs::write(&path, trace).unwrap();
    let script = r#"
import json
import sys
from pathlib import Path
from offline_replayer.measurement.scheduler_perf import parse_perf_script_switch_events
from offline_replayer.measurement.scheduler_bcc import summarize_scheduler_events, add_cgroup_validation

events, lost, anomalies = parse_perf_script_switch_events(Path(sys.argv[1]).read_text())
report = summarize_scheduler_events(
    events,
    window_start_ns=int(sys.argv[2]),
    window_end_ns=int(sys.argv[3]),
    lost_events=lost,
    method="perf-record-cgroup-native-switch-events",
    scope="test scope",
)
report = add_cgroup_validation(
    report,
    cgroup_usage_sec=float(sys.argv[4]),
    tolerance_fraction=0.02,
    tolerance_sec=0.005,
)
print(json.dumps({
    "events": [event.__dict__ for event in events],
    "lost_events": lost,
    "decoder_anomalies": anomalies,
    "report": report,
}, sort_keys=True))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .arg(start_ns.to_string())
        .arg(end_ns.to_string())
        .arg(cgroup_usage.to_string())
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Python scheduler perf oracle");
    assert!(
        output.status.success(),
        "Python oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}

fn rust_report(trace: &str, start_ns: u64, end_ns: u64, cgroup_usage: f64) -> Value {
    let parsed = parse_perf_script_switch_events(trace).unwrap();
    let report = summarize_scheduler_events(
        &parsed.events,
        start_ns,
        end_ns,
        parsed.lost_events,
        "perf-record-cgroup-native-switch-events",
        "test scope",
    )
    .unwrap();
    let report = add_cgroup_validation(&report, cgroup_usage, 0.02, 0.005).unwrap();
    json!({
        "events": parsed.events,
        "lost_events": parsed.lost_events,
        "decoder_anomalies": parsed.anomalies,
        "report": report,
    })
}

#[test]
fn parser_summary_and_symmetric_cgroup_gate_match_python() {
    let valid = concat!(
        "root 42/42 [000] 1.100000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0\n",
        "child 43/43 [001] 1.150000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0\n",
        ":-1 -1/-1 [001] 1.250000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0\n",
        "root 42/42 [000] 1.300000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0\n",
    );
    assert_eq!(
        rust_report(valid, 1_000_000_000, 1_400_000_000, 0.3),
        python_oracle(valid, 1_000_000_000, 1_400_000_000, 0.3)
    );

    let lossy = concat!(
        "perf 1/1 [000] 1.010000000: PERF_RECORD_LOST lost: 7\n",
        ":-1 -1/-1 [002] 1.020000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0\n",
        "root 52/52 [002] 1.030000000: PERF_RECORD_SWITCH_CPU_WIDE IN prev pid/tid: 0/0\n",
        "root 52/52 [002] 1.040000000: PERF_RECORD_SWITCH_CPU_WIDE OUT next pid/tid: 0/0\n",
    );
    assert_eq!(
        rust_report(lossy, 1_000_000_000, 1_100_000_000, 0.01),
        python_oracle(lossy, 1_000_000_000, 1_100_000_000, 0.01)
    );
}

#[test]
fn malformed_direction_and_nonpositive_window_fail_closed() {
    assert!(parse_perf_script_switch_events(
        "root 1/1 [000] 1.0: PERF_RECORD_SWITCH_CPU_WIDE MAYBE\n"
    )
    .is_err());
    assert!(summarize_scheduler_events(&[], 10, 10, 0, "test", "test").is_err());
}
