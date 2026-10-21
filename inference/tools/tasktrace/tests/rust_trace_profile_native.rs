// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fallback-disabled public CLI gates for native TaskTrace trace profiling.

use std::fs;
use std::process::Command;
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

#[test]
fn public_help_and_process_modes_are_native_with_fallback_disabled() {
    let temp = TempDir::new().expect("create native profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    let help = Command::new(binary())
        .args(["profile", "trace", "--help"])
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native profile help");
    assert!(help.status.success());
    let help_text = String::from_utf8_lossy(&help.stdout);
    assert!(help_text.contains("native replay controller is mounted read-only"));
    assert!(help_text.contains("--profile-set cpu-average"));

    for (label, mode) in [
        ("execve", vec!["--strace-mode", "execve"]),
        ("process", vec!["--strace-mode", "process"]),
        ("software-deep", vec!["--profile-set", "software-deep"]),
    ] {
        let output_dir = temp.path().join(format!("must-not-exist-{label}"));
        let output = Command::new(binary())
            .args(["profile", "trace"])
            .arg(temp.path().join(format!("missing-{label}.jsonl")))
            .args(["--image", "unused-image", "--allocated-cpus", "2"])
            .arg("--output-dir")
            .arg(&output_dir)
            .args(mode)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
            .output()
            .expect("run native process profile request");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("TaskTrace trace does not exist"), "{stderr}");
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }

    let output_dir = temp.path().join("must-not-exist-baseline");
    let missing_trace = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing.jsonl"))
        .args(["--image", "unused-image", "--allocated-cpus", "2"])
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run missing-trace native profile request");
    assert_eq!(missing_trace.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&missing_trace.stderr).contains("TaskTrace trace does not exist"));
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
    assert!(fs::read_dir(temp.path()).unwrap().next().is_none());
}

#[test]
fn proc_stat_cpuset_validation_never_invokes_python_fallback() {
    let temp = TempDir::new().expect("create proc-stat profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing.jsonl"))
        .args([
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
        .arg(temp.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native TaskTrace proc-stat validation");
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
fn enroot_wall_only_validation_never_invokes_python_fallback() {
    let temp = TempDir::new().expect("create Enroot profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing-enroot.jsonl"))
        .args([
            "--runtime",
            "enroot",
            "--enroot",
            "/opt/test-enroot",
            "--image",
            "task.sqsh",
            "--allocated-cpus",
            "1",
            "--strace-mode",
            "none",
        ])
        .arg("--output-dir")
        .arg(temp.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native TaskTrace Enroot validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("TaskTrace trace does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!missing_python.exists());

    let unsupported_output = temp.path().join("unsupported-output");
    let unsupported = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing-enroot-profile.jsonl"))
        .args([
            "--runtime",
            "enroot",
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
        .expect("run unsupported TaskTrace Enroot profiler mode");
    assert_eq!(unsupported.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&unsupported.stderr);
    assert!(stderr.contains("requires the Docker runtime"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!unsupported_output.exists());
}

#[test]
fn allowlist_and_backend_overrides_never_invoke_python_fallback() {
    let temp = TempDir::new().expect("create allowlist profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing-allowlist.jsonl"))
        .args(["--image", "unused-image", "--cpuset-cpus", "0"])
        .args(["--network-policy", "allowlist"])
        .args(["--allowed-host", "Allowed.TEST"])
        .args(["--network-policy-proxy-image", "fixture-proxy"])
        .args(["--network-policy-external-network", "fixture-external"])
        .arg("--output-dir")
        .arg(temp.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native TaskTrace allowlist validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("TaskTrace trace does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!missing_python.exists());
}

#[test]
fn memory_flag_and_memory_detail_never_invoke_python_fallback() {
    let temp = TempDir::new().expect("create memory profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("flag", vec!["--memory-profile"]),
        ("detail", vec!["--profile-set", "memory-detail"]),
    ] {
        let output_dir = temp.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "trace"])
            .arg(temp.path().join(format!("missing-{label}.jsonl")))
            .args(["--image", "unused-image", "--cpuset-cpus", "0"])
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
            .expect("run native TaskTrace memory validation");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("TaskTrace trace does not exist"), "{stderr}");
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn hardware_deep_never_invokes_python_fallback() {
    let temp = TempDir::new().expect("create hardware profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    let output_dir = temp.path().join("output-hardware");
    let output = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing-hardware.jsonl"))
        .args(["--image", "unused-image", "--cpuset-cpus", "0"])
        .args(["--profile-set", "hardware-deep"])
        .args([
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
        .expect("run native TaskTrace hardware validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("TaskTrace trace does not exist"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}

#[test]
fn perf_scheduler_modes_never_invoke_python_fallback() {
    let temp = TempDir::new().expect("create scheduler profile CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    for (label, mode) in [
        ("explicit", vec!["--scheduler-concurrency"]),
        ("cpu-peak", vec!["--profile-set", "cpu-peak"]),
        ("cpu-compare", vec!["--profile-set", "cpu-compare"]),
    ] {
        let output_dir = temp.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "trace"])
            .arg(temp.path().join(format!("missing-{label}.jsonl")))
            .args([
                "--image",
                "unused-image",
                "--cpuset-cpus",
                "0",
                "--scheduler-backend",
                "perf",
                "--scheduler-perf-binary",
                "/opt/test-perf",
                "--scheduler-perf-mmap-pages",
                "128",
                "--scheduler-attach-timeout-sec",
                "0.25",
                "--scheduler-cpu-tolerance-fraction",
                "0.03",
                "--scheduler-cpu-tolerance-sec",
                "0.007",
            ])
            .args(mode)
            .arg("--output-dir")
            .arg(&output_dir)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command
            .output()
            .expect("run native TaskTrace scheduler validation");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("TaskTrace trace does not exist"),
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
fn bcc_scheduler_modes_are_native_with_fallback_disabled() {
    let temp = TempDir::new().expect("create native BCC scheduler CLI root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    for mode in [
        vec!["--scheduler-concurrency"],
        vec!["--profile-set", "cpu-peak"],
        vec!["--profile-set", "cpu-compare"],
        vec!["--scheduler-concurrency", "--scheduler-backend", "bcc"],
    ] {
        let output_dir = temp
            .path()
            .join(format!("must-not-exist-{}", mode.join("-")));
        let output = Command::new(binary())
            .args(["profile", "trace"])
            .arg(temp.path().join("missing.jsonl"))
            .args(["--image", "unused-image", "--cpuset-cpus", "0"])
            .args(mode)
            .arg("--output-dir")
            .arg(&output_dir)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
            .output()
            .expect("run native TaskTrace BCC scheduler request");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("TaskTrace trace does not exist"), "{stderr}");
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn current_runtime_validation_stays_native_without_python() {
    let temp = TempDir::new().expect("create current-runtime validation root");
    let missing_python = temp.path().join("python-fallback-must-not-run");
    let output_dir = temp.path().join("output-current");
    let output = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing-current.jsonl"))
        .args(["--runtime", "current", "--allocated-cpus", "4"])
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--workdir")
        .arg(temp.path())
        .arg("--current-container-exclusive")
        .args(["--profile-set", "cpu-average"])
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native trace current validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--profile-set is not supported by current-container runtime"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());

    let output = Command::new(binary())
        .args(["profile", "trace"])
        .arg(temp.path().join("missing-current.jsonl"))
        .args(["--runtime", "current", "--image", "unused-image"])
        .arg("--output-dir")
        .arg(&output_dir)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native trace current image rejection");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--runtime current does not accept --image"),
        "{stderr}"
    );
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}
