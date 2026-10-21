// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Black-box differential coverage for the native replay leaves.
//!
//! Python remains the compatibility oracle during the Rust migration. These
//! tests execute both public implementations in symmetric fresh directories,
//! normalize only explicitly nondeterministic measurement/path fields, and
//! then compare the complete remaining JSON value.

use serde_json::{json, Value};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs::{self, File};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const COMMAND_TIMEOUT: &str = "30s";

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn repository() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

fn python() -> OsString {
    env::var_os("TASKTRACE_PARITY_PYTHON").unwrap_or_else(|| OsString::from("python3"))
}

fn python_path() -> OsString {
    let mut paths = vec![repository()];
    if let Some(existing) = env::var_os("PYTHONPATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).expect("construct parity-test PYTHONPATH")
}

fn wrapped_command(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new("timeout");
    command
        .arg("--signal=KILL")
        .arg("--kill-after=2s")
        .arg(COMMAND_TIMEOUT)
        .arg(program);
    command
}

fn rust_command(root: &Path) -> Command {
    let mut command = wrapped_command(binary());
    // Any regression to the migration dispatcher fails closed instead of
    // silently running the same Python code as the oracle.
    command.env(
        "TASKTRACE_LEGACY_PYTHON",
        root.join("python-fallback-must-not-run"),
    );
    command
}

fn python_command(module: &str) -> Command {
    let mut command = wrapped_command(python());
    command
        .arg("-m")
        .arg(module)
        .current_dir(repository())
        .env("PYTHONPATH", python_path());
    command
}

fn replay_command(root: &Path, native: bool, command: &str, module: &str) -> Command {
    if native {
        let mut replay = rust_command(root);
        replay.args(["replay", command]);
        replay
    } else {
        python_command(&format!("offline_replayer.adapters.{module}"))
    }
}

fn run(command: &mut Command, description: &str) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("unable to {description}: {error}"))
}

fn status_description(output: &Output) -> String {
    output
        .status
        .code()
        .map(|code| format!("exit {code}"))
        .or_else(|| {
            output
                .status
                .signal()
                .map(|signal| format!("signal {signal}"))
        })
        .unwrap_or_else(|| "unknown exit status".to_owned())
}

fn diagnostic(output: &Output) -> String {
    format!(
        "{}\nstdout:\n{}\nstderr:\n{}",
        status_description(output),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

fn assert_exit_parity(rust: &Output, python: &Output, context: &str) {
    assert_eq!(
        rust.status.code(),
        python.status.code(),
        "{context}: native/Python exit mismatch\nRust {}\nPython {}",
        diagnostic(rust),
        diagnostic(python),
    );
    assert_eq!(
        rust.status.signal(),
        python.status.signal(),
        "{context}: native/Python signal mismatch\nRust {}\nPython {}",
        diagnostic(rust),
        diagnostic(python),
    );
}

fn read_json(path: &Path) -> Value {
    serde_json::from_slice(
        &fs::read(path).unwrap_or_else(|error| panic!("read {}: {error}", path.display())),
    )
    .unwrap_or_else(|error| panic!("parse {}: {error}", path.display()))
}

fn read_jsonl(path: &Path) -> Vec<Value> {
    BufReader::new(
        File::open(path).unwrap_or_else(|error| panic!("open {}: {error}", path.display())),
    )
    .lines()
    .enumerate()
    .filter_map(|(index, line)| {
        let line = line
            .unwrap_or_else(|error| panic!("read {} line {}: {error}", path.display(), index + 1));
        (!line.trim().is_empty()).then(|| {
            serde_json::from_str(&line).unwrap_or_else(|error| {
                panic!("parse {} line {}: {error}", path.display(), index + 1)
            })
        })
    })
    .collect()
}

fn normalize_report(value: &mut Value, temp_root: &Path) {
    fn visit(value: &mut Value, temp_root: &str) {
        match value {
            Value::Object(fields) => {
                for (key, value) in fields {
                    match key.as_str() {
                        // These are runtime monotonic measurements, not source
                        // trajectory timing fields such as source_wall_time_sec.
                        "wall_time_sec" => *value = json!("<MONOTONIC_DURATION>"),
                        "replay_start_monotonic_ns" | "replay_end_monotonic_ns" => {
                            *value = json!("<MONOTONIC_TIMESTAMP>")
                        }
                        "replay_start_realtime_ns" => *value = json!("<REALTIME_TIMESTAMP>"),
                        "replay_start_realtime_uncertainty_ns" => {
                            *value = json!("<REALTIME_UNCERTAINTY>")
                        }
                        "process_root_pid" => *value = json!("<PROCESS_ROOT_PID>"),
                        // Transcript replay retains process CPU deltas. Their
                        // presence/schema is compared; the sampled values are not.
                        "user_cpu_sec" | "system_cpu_sec" | "cpu_time_sec" => {
                            *value = json!("<CPU_TIMING>")
                        }
                        // Interactive ATIF session filenames are implementation
                        // temporaries and have no fidelity meaning.
                        "session_log" if value.is_string() => *value = json!("<SESSION_LOG_PATH>"),
                        _ => visit(value, temp_root),
                    }
                }
            }
            Value::Array(values) => {
                for value in values {
                    visit(value, temp_root);
                }
            }
            Value::String(text) if !temp_root.is_empty() && text.contains(temp_root) => {
                *text = text.replace(temp_root, "<TMP>");
            }
            _ => {}
        }
    }

    visit(value, &temp_root.to_string_lossy());
}

fn assert_report_parity(
    mut rust: Value,
    rust_root: &Path,
    mut python: Value,
    python_root: &Path,
    context: &str,
) {
    normalize_report(&mut rust, rust_root);
    normalize_report(&mut python, python_root);
    assert_eq!(
        rust,
        python,
        "{context}: normalized reports differ\nRust:\n{}\nPython:\n{}",
        serde_json::to_string_pretty(&rust).unwrap(),
        serde_json::to_string_pretty(&python).unwrap(),
    );
}

fn atif_supported_tool(tool: &str) -> bool {
    matches!(
        tool,
        "exec_command"
            | "write_stdin"
            | "apply_patch"
            | "claude_write"
            | "claude_edit"
            | "claude_read"
            | "claude_grep"
            | "claude_glob"
            | "claude_task_stop"
            | "claude_task_output"
    )
}

fn assert_rust_atif_status_contract(report: &Value, output: &Output, context: &str) {
    let actions = report["actions"]
        .as_array()
        .expect("ATIF actions are a list");
    let summary = report["summary"]
        .as_object()
        .expect("ATIF summary is an object");
    let mut recorded = 0_u64;
    let mut replayed = 0_u64;
    let mut status_counts = [0_u64; 3];
    let mut outcome_counts = [0_u64; 3];
    let mut outcome_different = Vec::new();
    let mut outcome_unavailable = Vec::new();

    for action in actions {
        let index = action["action_index"]
            .as_u64()
            .expect("ATIF action index is an integer");
        let tool = action["tool"].as_str().expect("ATIF action has a tool");
        let source = action["source_status"]
            .as_str()
            .expect("ATIF action has a source status");
        let actual = action["status"]
            .as_str()
            .expect("ATIF action has a replay status");
        assert!(
            matches!(
                source,
                "ok" | "failed"
                    | "running"
                    | "timed_out"
                    | "unknown"
                    | "skipped_not_executed"
                    | "ignored_metadata"
            ) && matches!(
                actual,
                "ok" | "failed"
                    | "running"
                    | "timed_out"
                    | "skipped_not_executed"
                    | "ignored_metadata"
            ),
            "{context}: action {index} has an unnormalized status"
        );
        assert!(
            action.get("source_returncode").is_some(),
            "{context}: action {index} has no source return code field"
        );
        let launch_failed = action.get("launch_failed") == Some(&Value::Bool(true));
        let expected_status = if source == "unknown" || launch_failed {
            Value::Null
        } else {
            Value::Bool(source == actual)
        };
        assert_eq!(
            action.get("status_matches_source"),
            Some(&expected_status),
            "{context}: action {index} status comparison drifted"
        );

        let source_code = action["source_returncode"].as_i64();
        let replay_code = action.get("returncode").and_then(Value::as_i64);
        let expected_outcome = if source == "unknown" || launch_failed {
            None
        } else if source == "running" && matches!(actual, "ok" | "failed") {
            Some(true)
        } else if source != actual {
            Some(false)
        } else if source == "failed" && source_code == Some(-1) {
            Some(replay_code.is_some_and(|code| code != 0))
        } else {
            Some(source_code.map_or(true, |code| replay_code == Some(code)))
        };
        assert_eq!(
            action.get("outcome_matches_source"),
            Some(&expected_outcome.map_or(Value::Null, Value::Bool)),
            "{context}: action {index} outcome comparison drifted"
        );

        let supported = atif_supported_tool(tool);
        recorded += u64::from(supported && source != "skipped_not_executed");
        if !(supported
            && action.get("command_executed") != Some(&Value::Bool(false))
            && matches!(actual, "ok" | "failed" | "running" | "timed_out"))
        {
            if expected_outcome.is_none()
                && action.get("source_outcome_comparison_unavailable") == Some(&Value::Bool(true))
            {
                outcome_unavailable.push(json!(index));
            }
            continue;
        }
        replayed += 1;
        match expected_status.as_bool() {
            Some(true) => status_counts[0] += 1,
            Some(false) => status_counts[1] += 1,
            None => status_counts[2] += 1,
        }
        match expected_outcome {
            Some(true) => outcome_counts[0] += 1,
            Some(false) => {
                outcome_counts[1] += 1;
                outcome_different.push(json!(index));
            }
            None => {
                outcome_counts[2] += 1;
                outcome_unavailable.push(json!(index));
            }
        }
    }

    let count = |field| {
        summary[field]
            .as_u64()
            .unwrap_or_else(|| panic!("{context}: summary {field} is not an integer"))
    };
    assert_eq!(count("tool_actions_recorded"), recorded, "{context}");
    assert_eq!(count("tool_actions_replayed"), replayed, "{context}");
    assert_eq!(
        count("tool_action_status_equal"),
        status_counts[0],
        "{context}"
    );
    assert_eq!(
        count("tool_action_status_different"),
        status_counts[1],
        "{context}"
    );
    assert_eq!(
        count("tool_action_status_unavailable"),
        status_counts[2],
        "{context}"
    );
    assert_eq!(
        count("tool_action_status_comparable"),
        status_counts[0] + status_counts[1],
        "{context}"
    );
    assert_eq!(
        count("tool_action_outcome_equal"),
        outcome_counts[0],
        "{context}"
    );
    assert_eq!(
        count("tool_action_outcome_different"),
        outcome_counts[1],
        "{context}"
    );
    assert_eq!(
        count("tool_action_outcome_unavailable"),
        outcome_counts[2],
        "{context}"
    );
    assert_eq!(
        count("tool_action_outcome_comparable"),
        outcome_counts[0] + outcome_counts[1],
        "{context}"
    );
    assert_eq!(
        summary["source_outcome_mismatch_action_indexes"]
            .as_array()
            .expect("outcome mismatch indexes are a list"),
        &outcome_different,
        "{context}"
    );
    assert_eq!(
        summary["source_outcome_comparison_unavailable_action_indexes"]
            .as_array()
            .expect("outcome unavailable indexes are a list"),
        &outcome_unavailable,
        "{context}"
    );
    assert_eq!(
        count("source_outcome_mismatch_count"),
        outcome_counts[1],
        "{context}"
    );
    assert_eq!(
        count("source_outcome_comparison_unavailable_count"),
        outcome_unavailable.len() as u64,
        "{context}"
    );

    let expected_exit = i32::from(
        count("source_outcome_mismatch_count") > 0
            || count("source_outcome_comparison_unavailable_count") > 0
            || count("actions_launch_failed") > 0
            || count("actions_timed_out") > 0
            || count("actions_skipped_after_timeout") > 0,
    );
    assert_eq!(output.status.code(), Some(expected_exit), "{context}");
    assert_eq!(output.status.signal(), None, "{context}");
    let headline = format!(
        "Tool actions: {replayed}/{recorded} replayed; source status: {} equal, {} different, {} unavailable",
        status_counts[0], status_counts[1], status_counts[2]
    );
    assert!(
        String::from_utf8_lossy(&output.stdout).contains(&headline),
        "{context}: missing status headline {headline:?}\n{}",
        diagnostic(output)
    );
}

fn project_atif_compatibility_fields(report: &mut Value) {
    let fields = report.as_object_mut().expect("ATIF report is an object");
    fields.remove("subagent_trajectories_flattened");
    fields.remove("path_policy");
    for action in fields["actions"].as_array_mut().unwrap() {
        let action = action.as_object_mut().unwrap();
        for name in [
            "source_trajectory_id",
            "source_item_id",
            "parallel_group_id",
            "source_reported_wall_time_sec",
            "session_log",
            "session_output_preview",
        ] {
            action.remove(name);
        }
    }
    fields["limitations"]
        .as_array_mut()
        .unwrap()
        .retain(|value| {
            !value
                .as_str()
                .unwrap_or("")
                .starts_with("Recorded file-tool paths")
        });
}

fn atif_fixture(name: &str) -> PathBuf {
    repository()
        .join("offline_replayer/tests/fixtures/representative_atif")
        .join(name)
        .join("agent/trajectory.json")
}

fn run_atif_side(root: &Path, fixture: &Path, native: bool) -> (Output, PathBuf) {
    let workspace = root.join("workspace");
    let report = root.join("output/replay-report.json");
    let action_outputs = root.join("action-outputs");
    fs::create_dir_all(&workspace).expect("create ATIF parity workspace");
    let mut command = replay_command(root, native, "atif", "atif_replay");
    command
        .arg(fixture)
        .arg("--output")
        .arg(&report)
        .arg("--cwd")
        .arg(&workspace)
        .arg("--pacing")
        .arg("unpaced")
        .arg("--timeout-sec")
        .arg("2")
        .arg("--action-output-dir")
        .arg(action_outputs);
    (run(&mut command, "run ATIF parity side"), report)
}

#[test]
fn representative_atif_fixtures_match_python_oracle() {
    for fixture_name in [
        "deepswe-mini-swe__fixture",
        "claude-code__fixture",
        "generic-bare-bash__fixture",
        "swebenchpro-codex__fixture",
        "terminalbench-codex-exec__fixture",
        "terminalbench-codex-exec-promise-all__fixture",
        "swebenchpro-malformed-call__fixture",
    ] {
        let rust_root = TempDir::new().expect("create native ATIF parity root");
        let python_root = TempDir::new().expect("create Python ATIF parity root");
        let fixture = atif_fixture(fixture_name);
        let (rust_output, rust_report) = run_atif_side(rust_root.path(), &fixture, true);
        let (python_output, python_report) = run_atif_side(python_root.path(), &fixture, false);
        assert_eq!(
            rust_report.exists(),
            python_report.exists(),
            "{fixture_name}: only one implementation wrote a report\nRust {}\nPython {}",
            diagnostic(&rust_output),
            diagnostic(&python_output),
        );
        if !rust_report.exists() {
            assert_exit_parity(&rust_output, &python_output, fixture_name);
            continue;
        }
        let mut rust_report = read_json(&rust_report);
        let mut python_report = read_json(&python_report);
        assert_exit_parity(&rust_output, &python_output, fixture_name);
        assert_rust_atif_status_contract(&rust_report, &rust_output, fixture_name);
        project_atif_compatibility_fields(&mut rust_report);
        project_atif_compatibility_fields(&mut python_report);
        assert_report_parity(
            rust_report,
            rust_root.path(),
            python_report,
            python_root.path(),
            fixture_name,
        );
    }
}

#[test]
fn atif_launch_failure_unavailable_accounting_matches_python() {
    let fixture_root = TempDir::new().unwrap();
    let fixture = fixture_root.path().join("trajectory.json");
    fs::write(
        &fixture,
        serde_json::to_vec(&json!({
            "schema_version": "ATIF-v1.5",
            "session_id": "launch-failure",
            "agent": {"name": "codex"},
            "steps": [{
                "step_id": 1, "timestamp": 0,
                "tool_calls": [{
                    "tool_call_id": "call-1", "function_name": "exec_command",
                    "arguments": {"cmd": "true", "workdir": fixture_root.path().join("missing")}
                }],
                "observation": {"content": "Process exited with code 1"}
            }]
        }))
        .unwrap(),
    )
    .unwrap();
    let rust_root = TempDir::new().unwrap();
    let python_root = TempDir::new().unwrap();
    let (rust_output, rust_path) = run_atif_side(rust_root.path(), &fixture, true);
    let (python_output, python_path) = run_atif_side(python_root.path(), &fixture, false);
    assert_exit_parity(&rust_output, &python_output, "launch failure");
    assert_eq!(rust_output.status.code(), Some(1));
    let mut rust_report = read_json(&rust_path);
    let mut python_report = read_json(&python_path);
    assert_rust_atif_status_contract(&rust_report, &rust_output, "launch failure");
    assert_eq!(rust_report["summary"]["actions_launch_failed"], 1);
    assert_eq!(rust_report["summary"]["tool_actions_replayed"], 0);
    assert_eq!(rust_report["summary"]["tool_action_outcome_unavailable"], 0);
    assert_eq!(
        rust_report["summary"]["source_outcome_comparison_unavailable_action_indexes"],
        json!([1])
    );
    assert_eq!(
        rust_report["summary"]["source_outcome_comparison_unavailable_count"],
        1
    );
    project_atif_compatibility_fields(&mut rust_report);
    project_atif_compatibility_fields(&mut python_report);
    assert_report_parity(
        rust_report,
        rust_root.path(),
        python_report,
        python_root.path(),
        "launch failure",
    );
}

fn prepend_path(directory: &Path) -> OsString {
    let mut paths = vec![directory.to_path_buf()];
    if let Some(existing) = env::var_os("PATH") {
        paths.extend(env::split_paths(&existing));
    }
    env::join_paths(paths).expect("construct deterministic transcript PATH")
}

fn run_transcript_side(root: &Path, fake_bin: &Path, native: bool) -> (Output, PathBuf) {
    let transcript = repository().join("offline_replayer/examples/productivity_calendar.jsonl");
    let workspace = root.join("workspace");
    let output_dir = root.join("output");
    let mut command = replay_command(root, native, "transcript", "transcript_replay");
    command
        .arg(transcript)
        .arg("--task-id")
        .arg("productivity__task_calendar")
        .arg("--trial-name")
        .arg("productivity__task_calendar.1-of-3.default")
        .arg("--workspace")
        .arg(workspace)
        .arg("--source-workspace")
        .arg("/tmp/pinchbench/0001/agent_workspace")
        .arg("--timeout-sec")
        .arg("2")
        .arg("--output-dir")
        .arg(&output_dir)
        .env("PATH", prepend_path(fake_bin));
    (run(&mut command, "run transcript parity side"), output_dir)
}

#[test]
fn transcript_quickstart_matches_python_oracle() {
    let fixture_root = TempDir::new().expect("create deterministic-tool root");
    let fake_bin = fixture_root.path().join("bin");
    fs::create_dir(&fake_bin).expect("create deterministic-tool bin directory");
    let date = fake_bin.join("date");
    fs::write(
        &date,
        "#!/bin/sh\nprintf '%s\\n' '2026-05-28 18:28:55 Thursday'\n",
    )
    .expect("write deterministic date command");
    let mut permissions = fs::metadata(&date).unwrap().permissions();
    permissions.set_mode(0o755);
    fs::set_permissions(&date, permissions).expect("make deterministic date executable");

    let rust_root = TempDir::new().expect("create native transcript parity root");
    let python_root = TempDir::new().expect("create Python transcript parity root");
    let (rust_output, rust_output_dir) = run_transcript_side(rust_root.path(), &fake_bin, true);
    let (python_output, python_output_dir) =
        run_transcript_side(python_root.path(), &fake_bin, false);
    assert_exit_parity(&rust_output, &python_output, "transcript quickstart");
    assert!(
        rust_output.status.success(),
        "transcript quickstart failed\nRust {}\nPython {}",
        diagnostic(&rust_output),
        diagnostic(&python_output),
    );
    assert_report_parity(
        read_json(&rust_output_dir.join("replay_report.json")),
        rust_root.path(),
        read_json(&python_output_dir.join("replay_report.json")),
        python_root.path(),
        "transcript quickstart",
    );
    assert_eq!(
        read_jsonl(&rust_output_dir.join("extracted_actions.jsonl")),
        read_jsonl(&python_output_dir.join("extracted_actions.jsonl")),
        "transcript extracted action streams differ",
    );
}

fn write_tasktrace_trace(path: &Path, mismatch: bool) {
    let mut output = File::create(path).expect("create synthetic TaskTrace trace");
    writeln!(
        output,
        "{}",
        json!({
            "record_type": "trace",
            "schema_version": "tasktrace-cpu-trace.v1",
            "source_trial": if mismatch { "parity__mismatch" } else { "parity__matching" },
            "task_name": "parity-task",
            "reward": 1,
            "limitations": ["synthetic deterministic parity fixture"],
            "environment": {"action_timeout_sec": 2.0},
        })
    )
    .unwrap();
    let actions = if mismatch {
        vec![json!({
            "record_type": "action",
            "action_index": 1,
            "command": "printf 'mismatch-output\\n'; exit 9",
            "started_at_epoch": 100.0,
            "finished_at_epoch": 100.01,
            "wall_time_sec": 0.01,
            "returncode": 0,
            "exception_info": null,
        })]
    } else {
        vec![
            json!({
                "record_type": "action",
                "action_index": 1,
                "command": "printf 'alpha\\n'",
                "started_at_epoch": 100.0,
                "finished_at_epoch": 100.01,
                "wall_time_sec": 0.01,
                "returncode": 0,
                "exception_info": null,
            }),
            json!({
                "record_type": "action",
                "action_index": 2,
                "command": "printf 'expected failure\\n' >&2; exit 7",
                "started_at_epoch": 100.10,
                "finished_at_epoch": 100.12,
                "wall_time_sec": 0.02,
                "returncode": 7,
                "exception_info": null,
            }),
            json!({
                "record_type": "action",
                "action_index": 3,
                "command": "touch must-not-run",
                "started_at_epoch": 100.20,
                "finished_at_epoch": 100.20,
                "wall_time_sec": 0.0,
                "returncode": null,
                "exception_info": "action was not executed",
            }),
        ]
    };
    for action in actions {
        writeln!(output, "{action}").unwrap();
    }
}

fn run_tasktrace_side(root: &Path, trace: &Path, native: bool) -> (Output, PathBuf) {
    let workspace = root.join("workspace");
    let report = root.join("output/replay-report.json");
    fs::create_dir_all(&workspace).expect("create TaskTrace parity workspace");
    let mut command = replay_command(root, native, "trace", "trace_replay");
    command
        .arg(trace)
        .arg("--output")
        .arg(&report)
        .arg("--cwd")
        .arg(workspace)
        .arg("--pacing")
        .arg("unpaced")
        .arg("--timeout-sec")
        .arg("2");
    (run(&mut command, "run TaskTrace parity side"), report)
}

#[test]
fn tasktrace_trace_reports_and_mismatch_exit_match_python_oracle() {
    for mismatch in [false, true] {
        let fixture = TempDir::new().expect("create synthetic TaskTrace fixture root");
        let trace = fixture.path().join("trace.jsonl");
        write_tasktrace_trace(&trace, mismatch);
        let rust_root = TempDir::new().expect("create native TaskTrace parity root");
        let python_root = TempDir::new().expect("create Python TaskTrace parity root");
        let (rust_output, rust_report) = run_tasktrace_side(rust_root.path(), &trace, true);
        let (python_output, python_report) = run_tasktrace_side(python_root.path(), &trace, false);
        let context = if mismatch {
            "TaskTrace deliberate mismatch"
        } else {
            "TaskTrace matching trace"
        };
        assert_exit_parity(&rust_output, &python_output, context);
        assert_eq!(
            rust_output.status.code(),
            Some(if mismatch { 1 } else { 0 }),
            "{context}: unexpected exit behavior\nRust {}\nPython {}",
            diagnostic(&rust_output),
            diagnostic(&python_output),
        );
        let rust_report_value = read_json(&rust_report);
        let python_report_value = read_json(&python_report);
        assert_eq!(
            rust_report_value["summary"]["source_outcome_mismatch_count"],
            if mismatch { json!(1) } else { json!(0) },
        );
        assert_report_parity(
            rust_report_value,
            rust_root.path(),
            python_report_value,
            python_root.path(),
            context,
        );
    }
}
