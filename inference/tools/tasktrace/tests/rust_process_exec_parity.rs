// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#[path = "../src/profile_container/process_exec.rs"]
mod process_exec;

use process_exec::{
    load_process_exec_trace, parse_process_exec_trace, serialize_process_exec_report,
    write_process_exec_report,
};
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

fn fixture(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/process_exec")
        .join(name)
}

fn python_report(path: &Path, initial_cwd: &str) -> String {
    let script = r#"
import sys
from pathlib import Path
from offline_replayer.measurement.process_exec import parse_process_exec_trace, serialize_process_exec_report

text = Path(sys.argv[1]).read_text(encoding="utf-8")
report = parse_process_exec_trace(text, initial_cwd=sys.argv[2])
sys.stdout.write(serialize_process_exec_report(report))
"#;
    let output = Command::new("python3")
        .arg("-c")
        .arg(script)
        .arg(path)
        .arg(initial_cwd)
        .current_dir(env!("CARGO_MANIFEST_DIR"))
        .output()
        .expect("run Python process-exec oracle");
    assert!(
        output.status.success(),
        "Python oracle failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    String::from_utf8(output.stdout).expect("Python emits UTF-8")
}

#[test]
fn rust_reports_match_python_for_representative_strace_fixtures() {
    for (name, initial_cwd) in [
        ("realistic.strace", "/app"),
        ("split_vfork_thread.strace", "/srv"),
        ("malformed.strace", "/"),
        ("quoting.strace", "//srv/./root"),
    ] {
        let path = fixture(name);
        let text = fs::read_to_string(&path).unwrap();
        let rust = parse_process_exec_trace(&text, Some(initial_cwd), None);
        let rust = serialize_process_exec_report(&rust).unwrap();
        let python = python_report(&path, initial_cwd);
        assert_eq!(rust, python, "Rust/Python report mismatch for {name}");
    }
}

#[test]
fn malformed_and_empty_inputs_fail_closed_without_panicking() {
    for text in [
        "",
        "not strace\n",
        "1 0.0000000001 execve(\"/x\", [\"x\"], 0) = 0\n",
        "1 0.0 execve(\"unterminated, [\"x\"], 0) = 0\n",
        "1 0.0 clone(flags=CLONE_VM <unfinished ...>\n",
    ] {
        let report = parse_process_exec_trace(text, Some("/"), None);
        assert_eq!(report["valid"], Value::Bool(false));
        assert!(report["summary"]["anomaly_count"].as_u64().unwrap() > 0);
    }
}

#[test]
fn invalid_utf8_is_a_loss_report_and_publication_errors_remain_results() {
    let directory = tempfile::tempdir().unwrap();
    let raw = directory.path().join("raw.strace");
    fs::write(&raw, [0xff, 0xfe]).unwrap();
    let report = load_process_exec_trace(&raw, Some("/"), None);
    assert_eq!(report["valid"], false);
    assert_eq!(report["loss"]["detected"], true);
    assert_eq!(report["anomalies"][0]["code"], "invalid_trace_encoding");

    let output = directory.path().join("report.json");
    write_process_exec_report(&report, &output).unwrap();
    assert_eq!(
        fs::read_to_string(&output).unwrap(),
        serialize_process_exec_report(&report).unwrap()
    );
    assert!(write_process_exec_report(&report, directory.path()).is_err());
}
