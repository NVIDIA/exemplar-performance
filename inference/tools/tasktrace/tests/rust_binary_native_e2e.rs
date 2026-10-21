// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Black-box end-to-end coverage for the compiled Rust `tasktrace` binary.
//!
//! These tests deliberately do not import crate internals. They execute the
//! Cargo-built public binary with the legacy Python executable set to a path
//! that cannot exist, so a passing result proves that the selected leaf ran
//! natively rather than through the migration fallback.

use serde_json::{json, Value};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

const REPORT_SCHEMA: &str = "offline-atif-trajectory-replay.v1";

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("offline_replayer/tests/fixtures/representative_atif")
        .join(name)
        .join("agent/trajectory.json")
}

fn run_atif(
    trajectory: &Path,
    workspace: &Path,
    report: &Path,
    action_output_dir: Option<&Path>,
) -> Output {
    fs::create_dir_all(workspace).expect("create replay workspace");
    let mut command = Command::new("timeout");
    command
        .arg("--signal=KILL")
        .arg("--kill-after=2s")
        .arg("15s")
        .arg(binary())
        .arg("replay")
        .arg("atif")
        .arg(trajectory)
        .arg("--output")
        .arg(report)
        .arg("--cwd")
        .arg(workspace)
        .arg("--pacing")
        .arg("unpaced")
        .arg("--timeout-sec")
        .arg("2")
        // If the dispatcher regresses to Python delegation, execution must
        // fail instead of accidentally validating the retained implementation.
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            workspace.join("python-fallback-must-not-run"),
        );
    if let Some(path) = action_output_dir {
        command.arg("--action-output-dir").arg(path);
    }
    command.output().expect("start compiled tasktrace")
}

fn assert_success(output: &Output) {
    assert!(
        output.status.success(),
        "tasktrace failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn read_report(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).expect("read replay report"))
        .expect("parse replay report JSON")
}

fn summary_u64(report: &Value, field: &str) -> u64 {
    report["summary"][field]
        .as_u64()
        .unwrap_or_else(|| panic!("summary field {field:?} is not an unsigned integer"))
}

#[test]
fn native_deepswe_replay_preserves_outcomes_and_skips_unexecuted_actions() {
    let temp = TempDir::new().expect("create temporary E2E root");
    let workspace = temp.path().join("workspace");
    let report_path = temp.path().join("replay-report.json");
    let action_outputs = temp.path().join("action-outputs");

    let output = run_atif(
        &fixture("deepswe-mini-swe__fixture"),
        &workspace,
        &report_path,
        Some(&action_outputs),
    );
    assert_success(&output);

    let report = read_report(&report_path);
    assert_eq!(report["schema_version"].as_str(), Some(REPORT_SCHEMA));
    assert_eq!(report["source_schema_version"].as_str(), Some("ATIF-v1.2"));
    assert_eq!(
        report["source_agent_original_format"].as_str(),
        Some("mini-swe-agent-1.1")
    );
    assert_eq!(summary_u64(&report, "actions_recorded"), 4);
    assert_eq!(summary_u64(&report, "actions_executed"), 3);
    assert_eq!(summary_u64(&report, "actions_failed"), 2);
    assert_eq!(summary_u64(&report, "actions_skipped_not_executed"), 1);
    assert_eq!(summary_u64(&report, "source_outcome_mismatch_count"), 0);
    assert_eq!(report["actions"][1]["returncode"].as_i64(), Some(2));
    assert_eq!(report["actions"][2]["returncode"].as_i64(), Some(7));

    assert!(!workspace.join("skipped.txt").exists());
    assert_eq!(
        fs::read_to_string(action_outputs.join("action-0001.output.txt"))
            .expect("read combined command output"),
        "deep-stdout\ndeep-stderr\n"
    );
    assert_eq!(
        fs::read_to_string(action_outputs.join("action-0003.output.txt"))
            .expect("read recorded failure output"),
        "source-failure\n"
    );
}

#[test]
fn native_codex_exec_fixtures_are_rejected_before_execution() {
    for name in [
        "swebenchpro-codex__fixture",
        "swebenchpro-malformed-call__fixture",
    ] {
        let temp = TempDir::new().expect("create temporary E2E root");
        let workspace = temp.path().join("workspace");
        let report_path = temp.path().join("replay-report.json");
        let action_outputs = temp.path().join("action-outputs");
        let output = run_atif(
            &fixture(name),
            &workspace,
            &report_path,
            Some(&action_outputs),
        );
        assert!(!output.status.success());
        assert!(String::from_utf8_lossy(&output.stderr).contains("not supported in release 26.10"));
        assert!(!report_path.exists());
        assert!(!action_outputs.exists());
        assert_eq!(fs::read_dir(&workspace).unwrap().count(), 0);
    }
}

fn process_exists(pid: i32) -> bool {
    let result = unsafe { libc::kill(pid, 0) };
    if result == 0 {
        return true;
    }
    std::io::Error::last_os_error().raw_os_error() != Some(libc::ESRCH)
}

struct LeakedProcessGuard(i32);

impl Drop for LeakedProcessGuard {
    fn drop(&mut self) {
        if process_exists(self.0) {
            unsafe {
                libc::kill(self.0, libc::SIGKILL);
            }
        }
    }
}

#[test]
fn native_replay_terminates_an_unfinished_interactive_session() {
    let temp = TempDir::new().expect("create temporary E2E root");
    let workspace = temp.path().join("workspace");
    fs::create_dir_all(&workspace).expect("create replay workspace");
    let trajectory_path = temp.path().join("unfinished-session.json");
    let report_path = temp.path().join("replay-report.json");
    let trajectory = json!({
        "schema_version": "ATIF-v1.5",
        "session_id": "rust-cleanup-e2e",
        "agent": {"name": "codex", "model_name": "fixture"},
        "steps": [{
            "step_id": 1,
            "timestamp": "2026-07-06T00:00:00Z",
            "tool_calls": [{
                "tool_call_id": "unfinished-session",
                "function_name": "exec_command",
                "arguments": {
                    "cmd": "printf '%s\\n' \"$$\" > sleeper.pid; exec sleep 120",
                    // Give the shell enough time to publish the PID before the
                    // replay's end-of-trajectory cleanup terminates it.
                    "yield_time_ms": 100,
                    "max_output_tokens": 12000
                }
            }],
            "observation": {"results": [{
                "source_call_id": "unfinished-session",
                "content": "Chunk ID: fixture\\nWall time: 0.001 seconds\\nProcess running with session ID 991\\nOriginal token count: 0\\nOutput:\\n"
            }]}
        }]
    });
    fs::write(
        &trajectory_path,
        serde_json::to_vec_pretty(&trajectory).expect("serialize cleanup trajectory"),
    )
    .expect("write cleanup trajectory");

    let output = run_atif(&trajectory_path, &workspace, &report_path, None);
    let leaked_process = fs::read_to_string(workspace.join("sleeper.pid"))
        .ok()
        .and_then(|value| value.trim().parse::<i32>().ok())
        .map(LeakedProcessGuard);
    assert_success(&output);
    let report = read_report(&report_path);
    assert_eq!(report["schema_version"].as_str(), Some(REPORT_SCHEMA));
    assert_eq!(summary_u64(&report, "source_outcome_mismatch_count"), 0);
    assert_eq!(
        report["summary"]["sessions_terminated_at_end"],
        json!(["991"])
    );

    let pid = leaked_process
        .as_ref()
        .expect("unfinished replay session did not write its PID")
        .0;
    let deadline = Instant::now() + Duration::from_secs(2);
    while process_exists(pid) && Instant::now() < deadline {
        thread::sleep(Duration::from_millis(20));
    }
    assert!(
        !process_exists(pid),
        "unfinished replay session process {pid} survived binary exit"
    );
}
