// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fallback-disabled public CLI checks for the native ATIF profile adapter.

use std::env;
use std::process::Command;
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

#[test]
fn atif_profile_help_does_not_start_python() {
    let temporary = TempDir::new().expect("create fallback-disabled CLI root");
    let output = Command::new(binary())
        .arg("profile")
        .arg("atif")
        .arg("--help")
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-must-not-run"),
        )
        .output()
        .expect("run native ATIF profile help");
    assert!(
        output.status.success(),
        "native help failed:\n{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(String::from_utf8_lossy(&output.stdout)
        .contains("Profile supported actions from an ATIF trajectory"));
}

#[test]
fn enroot_wall_only_validation_stays_native_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled Enroot root");
    let missing_python = temporary.path().join("python-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "atif"])
        .arg(temporary.path().join("missing-enroot-trajectory.json"))
        .args([
            "--runtime",
            "enroot",
            "--enroot",
            "/opt/test-enroot",
            "--controller-runtime",
            "rust-host-mount",
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
        .expect("run native ATIF Enroot validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing-enroot-trajectory.json"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!missing_python.exists());
}

#[test]
fn enroot_profiler_requests_fail_locally_without_python() {
    let temporary = TempDir::new().expect("create unsupported Enroot root");
    let missing_python = temporary.path().join("python-must-not-run");
    let output = Command::new(binary())
        .args(["profile", "atif"])
        .arg(temporary.path().join("missing.json"))
        .args([
            "--runtime",
            "enroot",
            "--controller-runtime",
            "rust-host-mount",
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
        .arg(temporary.path().join("output"))
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run unsupported ATIF Enroot profiler request");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("wall-time-only"), "{stderr}");
    assert!(stderr.contains("unsupported"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!temporary.path().join("output").exists());
    assert!(!missing_python.exists());
}

#[test]
fn rust_host_mount_validation_fails_before_docker_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled validation root");
    let output = Command::new(binary())
        .arg("profile")
        .arg("atif")
        .arg(temporary.path().join("missing-trajectory.json"))
        .arg("--image")
        .arg("sha256:".to_owned() + &"a".repeat(64))
        .arg("--output-dir")
        .arg(temporary.path().join("output"))
        .arg("--cpuset-cpus")
        .arg("0")
        .arg("--controller-runtime")
        .arg("rust-host-mount")
        .arg("--profile-set")
        .arg("cpu-average")
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-must-not-run"),
        )
        .output()
        .expect("run native ATIF profile validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing-trajectory.json"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
}

#[test]
fn docker_defaults_to_native_controller_and_old_controller_choices_fail_locally() {
    let temporary = TempDir::new().expect("create controller migration root");
    let missing_python = temporary.path().join("python-must-not-run");
    let default_output = temporary.path().join("default-output");
    let default = Command::new(binary())
        .args(["profile", "atif"])
        .arg(temporary.path().join("missing-default-controller.json"))
        .args(["--image", "unused-image", "--allocated-cpus", "2"])
        .arg("--output-dir")
        .arg(&default_output)
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run default native-controller validation");
    assert_eq!(default.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&default.stderr);
    assert!(
        stderr.contains("missing-default-controller.json"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!default_output.exists());

    for controller in ["image", "host-mount"] {
        let output_dir = temporary.path().join(format!("old-{controller}"));
        let output = Command::new(binary())
            .args(["profile", "atif"])
            .arg(temporary.path().join(format!("missing-{controller}.json")))
            .args(["--image", "unused-image", "--allocated-cpus", "2"])
            .args(["--controller-runtime", controller])
            .arg("--output-dir")
            .arg(&output_dir)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
            .output()
            .expect("run old controller validation");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("is no longer supported"),
            "{controller}: {stderr}"
        );
        assert!(stderr.contains("rust-host-mount"), "{controller}: {stderr}");
        assert!(
            !stderr.contains("unable to start legacy"),
            "{controller}: {stderr}"
        );
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn process_modes_stay_native_with_fallback_disabled() {
    let temporary = TempDir::new().expect("create fallback-disabled process root");
    let missing_python = temporary.path().join("python-must-not-run");
    for (label, mode) in [
        ("execve", vec!["--strace-mode", "execve"]),
        ("process", vec!["--strace-mode", "process"]),
        ("software-deep", vec!["--profile-set", "software-deep"]),
    ] {
        let output_dir = temporary.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "atif"])
            .arg(temporary.path().join(format!("missing-{label}.json")))
            .arg("--image")
            .arg("sha256:".to_owned() + &"a".repeat(64))
            .arg("--output-dir")
            .arg(&output_dir)
            .args(["--allocated-cpus", "2"])
            .args(["--controller-runtime", "rust-host-mount"])
            .args(mode)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command.output().expect("run native ATIF process mode");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("missing-{label}.json")),
            "{stderr}"
        );
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn bcc_scheduler_modes_stay_native_with_fallback_disabled() {
    let temporary = TempDir::new().expect("create fallback-disabled BCC root");
    let missing_python = temporary.path().join("python-must-not-run");
    for (label, mode) in [
        ("explicit", vec!["--scheduler-concurrency"]),
        ("cpu-peak", vec!["--profile-set", "cpu-peak"]),
        ("cpu-compare", vec!["--profile-set", "cpu-compare"]),
    ] {
        let output_dir = temporary.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .args(["profile", "atif"])
            .arg(temporary.path().join(format!("missing-bcc-{label}.json")))
            .arg("--image")
            .arg("sha256:".to_owned() + &"a".repeat(64))
            .arg("--output-dir")
            .arg(&output_dir)
            .args(["--allocated-cpus", "2"])
            .args(["--controller-runtime", "rust-host-mount"])
            .args(["--scheduler-backend", "bcc"])
            .args(["--scheduler-bcc-buffer-pages", "64"])
            .args(mode)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command.output().expect("run native ATIF BCC mode");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("missing-bcc-{label}.json")),
            "{stderr}"
        );
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn allowlist_and_backend_overrides_stay_native_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled allowlist root");
    let output = Command::new(binary())
        .arg("profile")
        .arg("atif")
        .arg(temporary.path().join("missing-allowlist-trajectory.json"))
        .arg("--image")
        .arg("sha256:".to_owned() + &"a".repeat(64))
        .arg("--output-dir")
        .arg(temporary.path().join("output"))
        .arg("--cpuset-cpus")
        .arg("0")
        .arg("--controller-runtime")
        .arg("rust-host-mount")
        .arg("--profile-set")
        .arg("cpu-average")
        .arg("--network-policy")
        .arg("allowlist")
        .arg("--allowed-host")
        .arg("Allowed.TEST")
        .arg("--network-policy-proxy-image")
        .arg("fixture-proxy")
        .arg("--network-policy-external-network")
        .arg("fixture-external")
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-must-not-run"),
        )
        .output()
        .expect("run native ATIF allowlist validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("missing-allowlist-trajectory.json"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
}

#[test]
fn proc_stat_cpuset_validation_stays_native_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled proc-stat root");
    let output = Command::new(binary())
        .arg("profile")
        .arg("atif")
        .arg(temporary.path().join("missing-trajectory.json"))
        .arg("--image")
        .arg("sha256:".to_owned() + &"a".repeat(64))
        .arg("--output-dir")
        .arg(temporary.path().join("output"))
        .arg("--allocated-cpus")
        .arg("1")
        .arg("--controller-runtime")
        .arg("rust-host-mount")
        .arg("--proc-stat-comparison")
        .arg("--proc-stat-interval-sec")
        .arg("0.05")
        .arg("--proc-stat-baseline-sec")
        .arg("0.15")
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-must-not-run"),
        )
        .output()
        .expect("run native ATIF proc-stat validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--proc-stat-comparison requires --cpuset-cpus"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
}

#[test]
fn memory_flag_and_memory_detail_validation_stay_native_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled memory root");
    let missing_python = temporary.path().join("python-must-not-run");
    for (label, mode) in [
        ("flag", vec!["--memory-profile"]),
        ("detail", vec!["--profile-set", "memory-detail"]),
    ] {
        let output_dir = temporary.path().join(format!("output-{label}"));
        let mut command = Command::new(binary());
        command
            .arg("profile")
            .arg("atif")
            .arg(temporary.path().join(format!("missing-{label}.json")))
            .arg("--image")
            .arg("sha256:".to_owned() + &"a".repeat(64))
            .arg("--output-dir")
            .arg(&output_dir)
            .arg("--cpuset-cpus")
            .arg("0")
            .arg("--controller-runtime")
            .arg("rust-host-mount")
            .args(mode)
            .args([
                "--memory-sample-interval-sec",
                "0.02",
                "--memory-limit-bytes",
                "134217728",
                "--memory-swap-total-limit-bytes",
                "134217728",
            ])
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python);
        let output = command.output().expect("run native ATIF memory validation");
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!("missing-{label}.json")),
            "{stderr}"
        );
        assert!(!stderr.contains("unable to start legacy"), "{stderr}");
        assert!(!output_dir.exists());
    }
    assert!(!missing_python.exists());
}

#[test]
fn hardware_deep_validation_stays_native_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled hardware root");
    let missing_python = temporary.path().join("python-must-not-run");
    let output_dir = temporary.path().join("output-hardware");
    let output = Command::new(binary())
        .arg("profile")
        .arg("atif")
        .arg(temporary.path().join("missing-hardware.json"))
        .arg("--image")
        .arg("sha256:".to_owned() + &"a".repeat(64))
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--cpuset-cpus")
        .arg("0")
        .arg("--controller-runtime")
        .arg("rust-host-mount")
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
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native ATIF hardware validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("missing-hardware.json"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}

#[test]
fn current_runtime_validation_stays_native_without_python() {
    let temporary = TempDir::new().expect("create fallback-disabled current root");
    let missing_python = temporary.path().join("python-must-not-run");
    let output_dir = temporary.path().join("output-current");
    let output = Command::new(binary())
        .arg("profile")
        .arg("atif")
        .arg(temporary.path().join("missing-current.json"))
        .args(["--runtime", "current"])
        .arg("--output-dir")
        .arg(&output_dir)
        .arg("--workdir")
        .arg(temporary.path())
        .arg("--current-container-exclusive")
        .args(["--profile-set", "cpu-average"])
        .env("TASKTRACE_LEGACY_PYTHON", &missing_python)
        .output()
        .expect("run native ATIF current validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("--profile-set is not supported by current-container runtime"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
    assert!(!output_dir.exists());
    assert!(!missing_python.exists());
}
