// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Black-box differential coverage for native reporting leaves.
//!
//! Python remains the compatibility oracle during the Rust migration. Each
//! implementation receives an equivalent fixture in a separate temporary
//! root. Artifact comparison replaces only that absolute root; JSON key order,
//! whitespace, number spelling, CSV row order, and line endings remain exact.

use serde_json::{json, Value};
use std::env;
use std::ffi::OsString;
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
    env::join_paths(paths).expect("construct reporting parity PYTHONPATH")
}

fn run_native(root: &Path, arguments: &[OsString]) -> Output {
    Command::new(binary())
        .args(arguments)
        .current_dir(repository())
        // A dispatcher regression must fail instead of silently invoking the
        // same Python leaf used as this test's oracle.
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            root.join("python-fallback-must-not-run"),
        )
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run native reporting leaf")
}

fn run_python(module: &str, arguments: &[OsString]) -> Output {
    Command::new(python())
        .arg("-m")
        .arg(module)
        .args(arguments)
        .current_dir(repository())
        .env("PYTHONPATH", python_path())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run Python reporting leaf")
}

fn diagnostic(output: &Output) -> String {
    format!(
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

fn assert_successful_exit_parity(native: &Output, python: &Output, context: &str) {
    assert_eq!(
        native.status.code(),
        python.status.code(),
        "{context}: native/Python exit mismatch\nNative {}\nPython {}",
        diagnostic(native),
        diagnostic(python),
    );
    assert_eq!(
        python.status.code(),
        Some(0),
        "{context}: both implementations must succeed\nNative {}\nPython {}",
        diagnostic(native),
        diagnostic(python),
    );
}

fn normalized_artifact(path: &Path, temp_root: &Path) -> String {
    let artifact = fs::read_to_string(path)
        .unwrap_or_else(|error| panic!("read emitted artifact {}: {error}", path.display()));
    let absolute_root = fs::canonicalize(temp_root)
        .unwrap_or_else(|error| panic!("resolve temporary root {}: {error}", temp_root.display()));
    artifact.replace(absolute_root.to_string_lossy().as_ref(), "<TEMP_ROOT>")
}

fn assert_artifact_parity(
    native_path: &Path,
    native_root: &Path,
    python_path: &Path,
    python_root: &Path,
    context: &str,
) {
    assert_eq!(
        normalized_artifact(native_path, native_root),
        normalized_artifact(python_path, python_root),
        "{context}: emitted artifacts differ after absolute temporary-root normalization",
    );
}

fn write_json(path: &Path, value: Value) {
    fs::write(
        path,
        serde_json::to_vec(&value).expect("serialize fixture JSON"),
    )
    .unwrap_or_else(|error| panic!("write fixture {}: {error}", path.display()));
}

struct CeilingFixture {
    wide_summary: PathBuf,
    baseline_summary: PathBuf,
    wide_commands: PathBuf,
    baseline_commands: PathBuf,
    output_dir: PathBuf,
}

impl CeilingFixture {
    fn create(root: &Path) -> Self {
        let wide_summary = root.join("wide.json");
        write_json(
            &wide_summary,
            json!({
                "tasks": [
                    {
                        "task_name": "numeric",
                        "wall_sec": 2.0,
                        "average_logical_cores_used": 3.0,
                        "peak_simultaneous_logical_cores": 16,
                        "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "16": 1.0},
                    },
                    {
                        "task_name": "new-task",
                        "wall_sec": 10.0,
                        "average_logical_cores_used": 1.0,
                        "peak_simultaneous_logical_cores": 16,
                        "time_sec_by_simultaneous_logical_cores": {"1": 9.999, "16": 0.001},
                    },
                    {
                        "task_name": "narrow",
                        "wall_sec": 1.0,
                        "average_logical_cores_used": 1.0,
                        "peak_simultaneous_logical_cores": 4,
                        "time_sec_by_simultaneous_logical_cores": {"1": 1.0},
                    },
                ],
            }),
        );

        let baseline_summary = root.join("baseline.json");
        write_json(
            &baseline_summary,
            json!({
                "tasks": [{
                    "task_name": "numeric",
                    "wall_sec": 2.0,
                    "average_logical_cores_used": 2.0,
                    "peak_simultaneous_logical_cores": 8,
                    "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "8": 1.0},
                }],
            }),
        );

        let headers = concat!(
            "task_name,command_family,command_preview,lifecycle_wall_sec,",
            "average_simultaneous_logical_cores,peak_simultaneous_logical_cores,",
            "time_at_or_above_8_cores_sec\n",
        );
        let wide_commands = root.join("wide.csv");
        fs::write(
            &wide_commands,
            format!(
                "{headers}numeric,numpy,python compute.py,2.0,3.0,16,1.0\n\
                 new-task,startup,python smoke.py,10.0,1.0,16,0.001\n"
            ),
        )
        .expect("write wide command fixture");
        let baseline_commands = root.join("baseline.csv");
        fs::write(
            &baseline_commands,
            format!("{headers}numeric,numpy,python compute.py,2.0,2.0,8,1.0\n"),
        )
        .expect("write baseline command fixture");

        Self {
            wide_summary,
            baseline_summary,
            wide_commands,
            baseline_commands,
            output_dir: root.join("output"),
        }
    }

    fn arguments(&self, native: bool) -> Vec<OsString> {
        let mut arguments = Vec::new();
        if native {
            arguments.extend([OsString::from("report"), OsString::from("ceiling-hitters")]);
        }
        arguments.extend([
            OsString::from("--wide-summary"),
            self.wide_summary.as_os_str().to_owned(),
            OsString::from("--baseline-summary"),
            self.baseline_summary.as_os_str().to_owned(),
            OsString::from("--wide-commands"),
            self.wide_commands.as_os_str().to_owned(),
            OsString::from("--baseline-commands"),
            self.baseline_commands.as_os_str().to_owned(),
            OsString::from("--output-dir"),
            self.output_dir.as_os_str().to_owned(),
        ]);
        arguments
    }
}

#[test]
fn ceiling_hitters_artifacts_exactly_match_python_leaf() {
    let native_root = TempDir::new().expect("create native ceiling-hitters fixture root");
    let python_root = TempDir::new().expect("create Python ceiling-hitters fixture root");
    let native_fixture = CeilingFixture::create(native_root.path());
    let python_fixture = CeilingFixture::create(python_root.path());

    let native = run_native(native_root.path(), &native_fixture.arguments(true));
    let python = run_python(
        "offline_replayer.reporting.ceiling_hitters",
        &python_fixture.arguments(false),
    );
    assert_successful_exit_parity(&native, &python, "report ceiling-hitters");

    for artifact_name in [
        "ceiling_hitters_comparison.json",
        "ceiling_hitters_comparison.csv",
    ] {
        assert_artifact_parity(
            &native_fixture.output_dir.join(artifact_name),
            native_root.path(),
            &python_fixture.output_dir.join(artifact_name),
            python_root.path(),
            &format!("report ceiling-hitters {artifact_name}"),
        );
    }
}

struct TaskCatalogFixture {
    analysis: PathBuf,
    validated_summary: PathBuf,
    commands: PathBuf,
    output: PathBuf,
}

impl TaskCatalogFixture {
    fn create(root: &Path) -> Self {
        let numeric_trajectory = root.join("numeric_trajectory.json");
        write_json(
            &numeric_trajectory,
            json!({
                "steps": [
                    {
                        "source": "user",
                        "message": "<environment_context>ignored</environment_context>",
                    },
                    {
                        "source": "user",
                        "message": "Find the dominant eigenvalue using numpy and validate the numeric answer.",
                    },
                ],
            }),
        );
        let failed_trajectory = root.join("failed_trajectory.json");
        write_json(
            &failed_trajectory,
            json!({
                "steps": [
                    {
                        "source": "user",
                        "message": "<environment_context>ignored</environment_context>",
                    },
                    {
                        "source": "user",
                        "message": "Compile a large C project and run its test suite.",
                    },
                ],
            }),
        );

        let numeric_report = root.join("numeric_report.json");
        write_json(
            &numeric_report,
            json!({"source_trajectory": numeric_trajectory.to_string_lossy()}),
        );
        let failed_report = root.join("failed_report.json");
        write_json(
            &failed_report,
            json!({"source_trajectory": failed_trajectory.to_string_lossy()}),
        );

        let analysis = root.join("analysis.json");
        write_json(
            &analysis,
            json!({
                "tasks": [
                    {
                        "task_name": "largest-eigenval",
                        "status": "passed",
                        "distribution_quality": "validated_replay",
                        "pipeline_report": numeric_report.to_string_lossy(),
                        "allocated_logical_cpus": 16,
                        "wall_sec": 2.0,
                        "cpu_usage_sec": 3.0,
                        "average_logical_cores_used": 1.5,
                        "scheduler_peak_simultaneous_logical_cores": 16,
                        "included_in_exact_concurrency_distribution": true,
                        "failure_reasons": [],
                    },
                    {
                        "task_name": "compile-compcert",
                        "status": "failed_validation",
                        "distribution_quality": "observed_peak_lower_bound_only",
                        "pipeline_report": failed_report.to_string_lossy(),
                        "allocated_logical_cpus": 16,
                        "wall_sec": 30.0,
                        "cpu_usage_sec": 10.0,
                        "average_logical_cores_used": 0.3,
                        "scheduler_peak_simultaneous_logical_cores": 8,
                        "included_in_exact_concurrency_distribution": false,
                        "failure_reasons": ["scheduler_validation_failed"],
                    },
                ],
            }),
        );

        let validated_summary = root.join("validated.json");
        write_json(
            &validated_summary,
            json!({
                "tasks": [{
                    "task_name": "largest-eigenval",
                    "time_sec_by_simultaneous_logical_cores": {"1": 1.0, "16": 1.0},
                }],
            }),
        );

        let commands = root.join("commands.csv");
        fs::write(
            &commands,
            concat!(
                "task_name,command_family,command_preview,lifecycle_wall_sec,",
                "average_simultaneous_logical_cores,peak_simultaneous_logical_cores,",
                "time_at_or_above_4_cores_sec,time_at_or_above_8_cores_sec\n",
                "largest-eigenval,python eval.py,python /app/eval.py,2.0,3.0,16,1.0,0.5\n",
            ),
        )
        .expect("write task-catalog command fixture");

        Self {
            analysis,
            validated_summary,
            commands,
            output: root.join("catalog.json"),
        }
    }

    fn arguments(&self, native: bool) -> Vec<OsString> {
        let mut arguments = Vec::new();
        if native {
            arguments.extend([OsString::from("report"), OsString::from("task-catalog")]);
        }
        arguments.extend([
            OsString::from("--task-analysis"),
            self.analysis.as_os_str().to_owned(),
            OsString::from("--validated-summary"),
            self.validated_summary.as_os_str().to_owned(),
            OsString::from("--commands"),
            self.commands.as_os_str().to_owned(),
            OsString::from("--output"),
            self.output.as_os_str().to_owned(),
        ]);
        arguments
    }
}

#[test]
fn task_catalog_artifact_exactly_matches_python_leaf() {
    let native_root = TempDir::new().expect("create native task-catalog fixture root");
    let python_root = TempDir::new().expect("create Python task-catalog fixture root");
    let native_fixture = TaskCatalogFixture::create(native_root.path());
    let python_fixture = TaskCatalogFixture::create(python_root.path());

    let native = run_native(native_root.path(), &native_fixture.arguments(true));
    let python = run_python(
        "offline_replayer.reporting.task_catalog",
        &python_fixture.arguments(false),
    );
    assert_successful_exit_parity(&native, &python, "report task-catalog");
    assert_artifact_parity(
        &native_fixture.output,
        native_root.path(),
        &python_fixture.output,
        python_root.path(),
        "report task-catalog catalog.json",
    );
}
