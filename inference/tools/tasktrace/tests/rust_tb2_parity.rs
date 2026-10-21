// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Byte-level compatibility gate for the native Terminal-Bench analyzer.

use serde_json::json;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

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
    env::join_paths(paths).unwrap()
}

fn trajectory(root: &Path, task: &str) {
    let path = root
        .join(format!("{task}__trial"))
        .join("agent/trajectory.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(path, "{}").unwrap();
}

fn report(
    root: &Path,
    dirname: &str,
    task: &str,
    mismatches: i64,
    timed_out: i64,
    scheduler_valid: bool,
) {
    let path = root
        .join("campaign-run")
        .join(dirname)
        .join("pipeline_report.json");
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(
        path,
        serde_json::to_vec(&json!({
            "task_name": task,
            "replay_summary": {
                "source_outcome_mismatch_count": mismatches,
                "actions_timed_out": timed_out,
            },
            "container_cpu": {
                "wall_sec": 2.0,
                "usage_sec": 4.0,
                "average_logical_cores_used": 2.0,
                "allocated_logical_cpus": 8,
            },
            "scheduler_concurrency": {
                "peak_simultaneous_logical_cores": 4,
                "lost_perf_events": if scheduler_valid { 0 } else { 2 },
                "anomalies": if scheduler_valid { Vec::<&str>::new() } else { vec!["drop"] },
                "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "3": 1.0},
                "cgroup_validation": {
                    "valid_for_concurrency_reporting": scheduler_valid,
                },
            },
        }))
        .unwrap(),
    )
    .unwrap();
}

fn run_native(arguments: &[&OsStr], root: &Path) -> Output {
    Command::new(binary())
        .args(arguments)
        .current_dir(repository())
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            root.join("python-fallback-must-not-run"),
        )
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap()
}

fn run_python(arguments: &[&OsStr]) -> Output {
    Command::new(python())
        .args(["-m", "offline_replayer.reporting.tb2_profiles"])
        .args(arguments)
        .current_dir(repository())
        .env("PYTHONPATH", python_path())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .unwrap()
}

fn diagnostic(output: &Output) -> String {
    format!(
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

#[test]
fn native_tb2_analysis_matches_python_artifacts_and_stdout() {
    let root = TempDir::new().unwrap();
    let trajectories = root.path().join("trajectories");
    let reports = root.path().join("reports");
    for task in [
        "rerun-task",
        "failed-valid-scheduler",
        "failed-invalid-scheduler",
        "incomplete-task",
        "unseen-task",
    ] {
        trajectory(&trajectories, task);
    }
    report(&reports, "rerun-old", "rerun-task", 1, 0, true);
    report(&reports, "rerun-pass", "rerun-task", 0, 0, true);
    report(
        &reports,
        "failed-valid",
        "failed-valid-scheduler",
        1,
        0,
        true,
    );
    report(
        &reports,
        "failed-invalid",
        "failed-invalid-scheduler",
        1,
        1,
        false,
    );
    fs::create_dir_all(reports.join("campaign-run/incomplete-task/container_profile")).unwrap();

    for (task, field, value) in [
        (
            "unavailable",
            "source_outcome_comparison_unavailable_count",
            1,
        ),
        ("launch-failed", "actions_launch_failed", 1),
        ("parse-failed", "actions_source_argument_parse_failed", 1),
    ] {
        trajectory(&trajectories, task);
        report(&reports, task, task, 0, 0, true);
        let path = reports
            .join("campaign-run")
            .join(task)
            .join("pipeline_report.json");
        let mut payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        payload["replay_summary"][field] = json!(value);
        fs::write(path, serde_json::to_vec(&payload).unwrap()).unwrap();
    }

    let native_output = root.path().join("native-output");
    let python_output = root.path().join("python-output");
    let native_arguments = [
        OsStr::new("report"),
        OsStr::new("analyze-tb2"),
        OsStr::new("--trajectory-root"),
        trajectories.as_os_str(),
        OsStr::new("--reports-root"),
        reports.as_os_str(),
        OsStr::new("--output-dir"),
        native_output.as_os_str(),
        OsStr::new("--include-failed"),
    ];
    let python_arguments = [
        OsStr::new("--trajectory-root"),
        trajectories.as_os_str(),
        OsStr::new("--reports-root"),
        reports.as_os_str(),
        OsStr::new("--output-dir"),
        python_output.as_os_str(),
        OsStr::new("--include-failed"),
    ];
    let native = run_native(&native_arguments, root.path());
    let python = run_python(&python_arguments);
    assert_eq!(native.status.code(), Some(0), "{}", diagnostic(&native));
    assert_eq!(python.status.code(), Some(0), "{}", diagnostic(&python));
    for name in [
        "tb2_profile_analysis.json",
        "tb2_task_status.csv",
        "tb2_concurrency_distribution.csv",
    ] {
        assert_eq!(
            fs::read(native_output.join(name)).unwrap(),
            fs::read(python_output.join(name)).unwrap(),
            "artifact differs: {name}"
        );
    }
    let analysis: serde_json::Value =
        serde_json::from_slice(&fs::read(native_output.join("tb2_profile_analysis.json")).unwrap())
            .unwrap();
    for (task, reason) in [
        ("unavailable", "source_outcome_comparison_unavailable"),
        ("launch-failed", "incomplete_replay"),
        ("parse-failed", "incomplete_replay"),
    ] {
        let row = analysis["tasks"]
            .as_array()
            .unwrap()
            .iter()
            .find(|row| row["task_name"] == task)
            .unwrap();
        assert!(row["failure_reasons"]
            .as_array()
            .unwrap()
            .contains(&json!(reason)));
    }
    let native_stdout = String::from_utf8(native.stdout)
        .unwrap()
        .replace(native_output.to_string_lossy().as_ref(), "<OUTPUT>");
    let python_stdout = String::from_utf8(python.stdout)
        .unwrap()
        .replace(python_output.to_string_lossy().as_ref(), "<OUTPUT>");
    assert_eq!(native_stdout, python_stdout);
}

#[test]
fn unavailable_comparison_counts_reject_non_unsigned_integers() {
    for value in [
        json!(null),
        json!(false),
        json!(true),
        json!("0"),
        json!(0.0),
        json!(-1),
        json!([]),
        json!({}),
    ] {
        let root = TempDir::new().unwrap();
        let trajectories = root.path().join("trajectories");
        let reports = root.path().join("reports");
        trajectory(&trajectories, "task");
        report(&reports, "task", "task", 0, 0, true);
        let path = reports.join("campaign-run/task/pipeline_report.json");
        let mut payload: serde_json::Value =
            serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        payload["replay_summary"]["source_outcome_comparison_unavailable_count"] = value.clone();
        fs::write(path, serde_json::to_vec(&payload).unwrap()).unwrap();
        let output = root.path().join("output");
        let arguments = [
            OsStr::new("--trajectory-root"),
            trajectories.as_os_str(),
            OsStr::new("--reports-root"),
            reports.as_os_str(),
            OsStr::new("--output-dir"),
            output.as_os_str(),
        ];
        let mut native_arguments = vec![OsStr::new("report"), OsStr::new("analyze-tb2")];
        native_arguments.extend(arguments);
        for result in [
            run_native(&native_arguments, root.path()),
            run_python(&arguments),
        ] {
            assert_eq!(
                result.status.code(),
                Some(2),
                "{value}: {}",
                diagnostic(&result)
            );
            assert!(
                String::from_utf8_lossy(&result.stderr).contains("invalid unsigned integer field"),
                "{value}: {}",
                diagnostic(&result)
            );
        }
    }
}
