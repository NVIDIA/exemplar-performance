// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fallback-disabled public CLI gates for native transcript profiling.

use serde_json::json;
use std::fs;
use std::process::Command;
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

#[test]
fn public_help_and_source_validation_do_not_start_python() {
    let temporary = TempDir::new().expect("create transcript profile CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    let help = Command::new(binary())
        .args(["profile", "transcript", "--help"])
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native transcript profile help");
    assert!(
        help.status.success(),
        "help failed:\n{}",
        String::from_utf8_lossy(&help.stderr)
    );
    let stdout = String::from_utf8_lossy(&help.stdout);
    assert!(stdout.contains("native replay controller is mounted read-only"));
    assert!(stdout.contains("--profile-set cpu-average"));

    let output_dir = temporary.path().join("missing-output");
    let missing = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(temporary.path().join("missing.jsonl"))
        .args([
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "unused-image",
            "--allocated-cpus",
            "2",
            "--strace-mode",
            "none",
            "--profile-set",
            "cpu-average",
        ])
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native transcript profile validation");
    assert_eq!(missing.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&missing.stderr);
    assert!(stderr.contains("transcript does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}

#[test]
fn bare_default_and_process_modes_do_not_start_python_fallback() {
    let temporary = TempDir::new().expect("create transcript process-routing root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("bare-default", Vec::new()),
        ("execve", vec!["--strace-mode", "execve"]),
        ("process", vec!["--strace-mode", "process"]),
        ("software-deep", vec!["--profile-set", "software-deep"]),
    ] {
        let output_dir = temporary.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "transcript"])
            .arg(temporary.path().join(format!("missing-{label}.jsonl")))
            .args([
                "--task-id",
                "task",
                "--trial-name",
                "task.1",
                "--image",
                "unused-image",
                "--allocated-cpus",
                "2",
            ])
            .args(mode)
            .arg("--output-dir")
            .arg(&output_dir)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command
            .output()
            .expect("run native transcript process mode");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("transcript does not exist"),
            "{label}: {stderr}"
        );
        assert!(
            !stderr.contains("unable to start legacy"),
            "{label}: {stderr}"
        );
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn enroot_wall_only_validation_does_not_start_python() {
    let temporary = TempDir::new().expect("create transcript Enroot CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(temporary.path().join("missing-enroot.jsonl"))
        .args([
            "--runtime",
            "enroot",
            "--enroot",
            "/opt/test-enroot",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "task.sqsh",
            "--allocated-cpus",
            "1",
            "--strace-mode",
            "none",
        ])
        .arg("--output-dir")
        .arg(temporary.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native transcript Enroot validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("transcript does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!missing_python.exists());

    let unsupported_output = temporary.path().join("unsupported-output");
    let unsupported = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(temporary.path().join("missing-enroot-profile.jsonl"))
        .args([
            "--runtime",
            "enroot",
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "task.sqsh",
            "--allocated-cpus",
            "1",
            "--strace-mode",
            "none",
            "--profile-set",
            "cpu-average",
        ])
        .arg("--output-dir")
        .arg(&unsupported_output)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run unsupported transcript Enroot profiler mode");
    assert_eq!(unsupported.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&unsupported.stderr);
    assert!(stderr.contains("requires the Docker runtime"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!unsupported_output.exists());
}

#[test]
fn allowlist_and_backend_overrides_do_not_start_python() {
    let temporary = TempDir::new().expect("create transcript allowlist CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(temporary.path().join("missing-allowlist.jsonl"))
        .args([
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "unused-image",
            "--cpuset-cpus",
            "0",
            "--strace-mode",
            "none",
            "--profile-set",
            "cpu-average",
            "--network-policy",
            "allowlist",
            "--allowed-host",
            "Allowed.TEST",
            "--network-policy-proxy-image",
            "fixture-proxy",
            "--network-policy-external-network",
            "fixture-external",
        ])
        .arg("--output-dir")
        .arg(temporary.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native transcript allowlist validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("transcript does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!missing_python.exists());
}

#[test]
fn unsupported_trial_fails_before_output_or_docker_without_python() {
    let temporary = TempDir::new().expect("create unsupported transcript CLI root");
    let transcript = temporary.path().join("trace.jsonl");
    fs::write(
        &transcript,
        json!({
            "task_id": "task",
            "trial_name": "task.1",
            "answer": "[Tool Call: browser] {\"url\":\"https://example.com\"}",
        })
        .to_string()
            + "\n",
    )
    .expect("write unsupported transcript");
    let output_dir = temporary.path().join("output");
    let output = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(&transcript)
        .args([
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "unused-image",
            "--allocated-cpus",
            "2",
            "--strace-mode",
            "none",
            "--profile-set",
            "cpu-average",
        ])
        .arg("--output-dir")
        .arg(&output_dir)
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-fallback-must-not-run"),
        )
        .output()
        .expect("run unsupported transcript validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("selected trial contains unsupported tools"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
}

#[test]
fn proc_stat_cpuset_validation_does_not_start_python() {
    let temporary = TempDir::new().expect("create transcript proc-stat CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(temporary.path().join("missing.jsonl"))
        .args([
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "unused-image",
            "--allocated-cpus",
            "2",
            "--proc-stat-comparison",
            "--proc-stat-interval-sec",
            "0.05",
            "--proc-stat-baseline-sec",
            "0.15",
        ])
        .arg("--output-dir")
        .arg(temporary.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native transcript proc-stat validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--proc-stat-comparison requires --cpuset-cpus"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!missing_python.exists());
}

#[test]
fn memory_flag_and_memory_detail_do_not_start_python() {
    let temporary = TempDir::new().expect("create transcript memory CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("flag", vec!["--memory-profile"]),
        ("detail", vec!["--profile-set", "memory-detail"]),
    ] {
        let output_dir = temporary.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "transcript"])
            .arg(temporary.path().join(format!("missing-{label}.jsonl")))
            .args([
                "--task-id",
                "task",
                "--trial-name",
                "task.1",
                "--image",
                "unused-image",
                "--cpuset-cpus",
                "0",
            ])
            .args(mode)
            .args([
                "--memory-sample-interval-sec",
                "0.02",
                "--memory-limit-bytes",
                "134217728",
                "--memory-swap-total-limit-bytes",
                "134217728",
            ])
            .arg("--output-dir")
            .arg(&output_dir)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command
            .output()
            .expect("run native transcript memory validation");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("transcript does not exist"), "{stderr}");
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn perf_and_bcc_scheduler_modes_are_native() {
    let temporary = TempDir::new().expect("create transcript scheduler CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("explicit", vec!["--scheduler-concurrency"]),
        ("cpu-peak", vec!["--profile-set", "cpu-peak"]),
        ("cpu-compare", vec!["--profile-set", "cpu-compare"]),
    ] {
        let output_dir = temporary.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "transcript"])
            .arg(temporary.path().join(format!("missing-{label}.jsonl")))
            .args([
                "--task-id",
                "task",
                "--trial-name",
                "task.1",
                "--image",
                "unused-image",
                "--cpuset-cpus",
                "0",
                "--scheduler-backend",
                "perf",
                "--scheduler-perf-mmap-pages",
                "0",
            ])
            .args(mode)
            .arg("--output-dir")
            .arg(&output_dir)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command
            .output()
            .expect("run native transcript scheduler validation");
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

    for mode in [
        vec!["--scheduler-concurrency"],
        vec!["--profile-set", "cpu-peak"],
    ] {
        let output = Command::new(binary())
            .args(["profile", "transcript"])
            .arg(temporary.path().join("missing-bcc.jsonl"))
            .args([
                "--task-id",
                "task",
                "--trial-name",
                "task.1",
                "--image",
                "unused-image",
                "--cpuset-cpus",
                "0",
            ])
            .args(mode)
            .arg("--output-dir")
            .arg(temporary.path().join("bcc-output"))
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
            .output()
            .expect("run native transcript BCC scheduler request");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("transcript does not exist"), "{stderr}");
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    }
    assert!(!missing_python.exists());
}

#[test]
fn hardware_deep_does_not_start_python() {
    let temporary = TempDir::new().expect("create transcript hardware CLI root");
    let missing_python = temporary.path().join("python-fallback-must-not-run");
    let output_dir = temporary.path().join("output-hardware");
    let output = Command::new(binary())
        .args(["profile", "transcript"])
        .arg(temporary.path().join("missing-hardware.jsonl"))
        .args([
            "--task-id",
            "task",
            "--trial-name",
            "task.1",
            "--image",
            "unused-image",
            "--cpuset-cpus",
            "0",
            "--profile-set",
            "hardware-deep",
            "--hardware-perf-binary",
            "missing-test-perf",
            "--hardware-perf-event",
            "cycles",
            "--hardware-perf-event",
            "custom-test-event",
            "--hardware-perf-attach-timeout-sec",
            "0.25",
            "--hardware-perf-minimum-running-ratio",
            "0.9",
        ])
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native transcript hardware validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("transcript does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}
