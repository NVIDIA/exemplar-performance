// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fallback-disabled public CLI gates for native scheduler profiling.

use std::fs;
use std::process::Command;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

#[test]
fn perf_scheduler_validation_is_native_for_explicit_and_named_modes() {
    let root = tempfile::tempdir().expect("create scheduler CLI root");
    let missing_python = root.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("explicit", vec!["--scheduler-concurrency"]),
        ("cpu-peak", vec!["--profile-set", "cpu-peak"]),
        ("cpu-compare", vec!["--profile-set", "cpu-compare"]),
    ] {
        let output_dir = root.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args([
                "profile",
                "container",
                "--image",
                "unused-image",
                "--command",
                "true",
                "--cpuset-cpus",
                "0",
                "--scheduler-backend",
                "perf",
                "--scheduler-perf-mmap-pages",
                "0",
                "--output-dir",
            ])
            .arg(&output_dir)
            .args(mode)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command.output().expect("run native scheduler validation");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("--scheduler-perf-mmap-pages must be positive"),
            "{label}: {stderr}"
        );
        assert!(
            !stderr.contains("unable to start legacy"),
            "{label}: {stderr}"
        );
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
    assert!(fs::read_dir(root.path()).unwrap().next().is_none());
}

#[test]
fn bcc_scheduler_requests_are_native_and_fail_before_output_when_unavailable() {
    let root = tempfile::tempdir().expect("create native BCC scheduler CLI root");
    let missing_python = root.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("explicit", vec!["--scheduler-concurrency"]),
        ("cpu-peak", vec!["--profile-set", "cpu-peak"]),
    ] {
        let output_dir = root.path().join(format!("must-not-exist-{label}"));
        let output = Command::new(binary())
            .args([
                "profile",
                "container",
                "--image",
                "unused-image",
                "--command",
                "true",
                "--cpuset-cpus",
                "0",
                "--output-dir",
            ])
            .arg(&output_dir)
            .args(mode)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
            .output()
            .expect("run native BCC scheduler request");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            !stderr.contains("unable to start legacy"),
            "{label}: {stderr}"
        );
        assert!(!output_dir.exists(), "{label}");
    }
    assert!(!missing_python.exists());
}

#[test]
fn bcc_buffer_validation_is_local_and_fallback_disabled() {
    let root = tempfile::tempdir().expect("create BCC buffer validation root");
    let output_dir = root.path().join("must-not-exist");
    let missing_python = root.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args([
            "profile",
            "container",
            "--image",
            "unused-image",
            "--command",
            "true",
            "--cpuset-cpus",
            "0",
            "--scheduler-concurrency",
            "--scheduler-backend",
            "bcc",
            "--scheduler-bcc-buffer-pages",
            "3",
            "--output-dir",
        ])
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native BCC buffer validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("positive power of two"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}

#[test]
fn perf_executable_is_preflighted_before_output_creation() {
    let root = tempfile::tempdir().expect("create scheduler preflight root");
    let output_dir = root.path().join("must-not-exist");
    let missing_perf = root.path().join("missing-perf");
    let missing_python = root.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args([
            "profile",
            "container",
            "--image",
            "unused-image",
            "--command",
            "true",
            "--cpuset-cpus",
            "0",
            "--scheduler-concurrency",
            "--scheduler-backend",
            "perf",
            "--scheduler-perf-binary",
        ])
        .arg(&missing_perf)
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run scheduler executable preflight");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing-perf"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}
