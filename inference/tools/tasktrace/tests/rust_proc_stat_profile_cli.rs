// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fallback-disabled public routing checks for native `/proc/stat` comparison.

use std::env;
use std::process::Command;
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

#[test]
fn proc_stat_request_is_validated_natively_without_python() {
    let temporary = TempDir::new().expect("create proc-stat CLI root");
    let output = Command::new(binary())
        .arg("profile")
        .arg("container")
        .arg("--image")
        .arg("sha256:".to_owned() + &"a".repeat(64))
        .arg("--command")
        .arg("true")
        .arg("--output-dir")
        .arg(temporary.path().join("profile"))
        .arg("--allocated-cpus")
        .arg("1")
        .arg("--proc-stat-comparison")
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-fallback-must-not-run"),
        )
        .output()
        .expect("run native proc-stat validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("requires --cpuset-cpus"), "{stderr}");
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
}

#[test]
fn proc_stat_timing_options_fail_closed_before_docker() {
    let temporary = TempDir::new().expect("create proc-stat timing CLI root");
    let output = Command::new(binary())
        .arg("profile")
        .arg("container")
        .arg("--image")
        .arg("sha256:".to_owned() + &"a".repeat(64))
        .arg("--command")
        .arg("true")
        .arg("--output-dir")
        .arg(temporary.path().join("profile"))
        .arg("--cpuset-cpus")
        .arg("0")
        .arg("--proc-stat-comparison")
        .arg("--proc-stat-interval-sec")
        .arg("0")
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-fallback-must-not-run"),
        )
        .output()
        .expect("run native proc-stat timing validation");
    assert_eq!(output.status.code(), Some(2));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("proc-stat interval must be positive"),
        "{stderr}"
    );
    assert!(!stderr.contains("unable to start legacy"), "{stderr}");
}
