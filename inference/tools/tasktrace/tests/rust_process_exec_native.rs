// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::process::{Command, Output};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn run_native(arguments: &[&str], fallback: &std::path::Path) -> Output {
    Command::new(binary())
        .args(["profile", "container"])
        .args(arguments)
        .env("TASKTRACE_LEGACY_PYTHON", fallback)
        .output()
        .expect("run native process profile routing probe")
}

#[test]
fn baseline_process_and_software_deep_route_natively_without_python_fallback() {
    let temporary = TempDir::new().unwrap();
    let fallback = temporary.path().join("python-fallback-must-not-run");
    for selector in [
        vec!["--strace-mode", "process"],
        vec!["--profile-set", "software-deep"],
    ] {
        let mut arguments = vec![
            "--image",
            "fixture",
            "--command",
            "true",
            "--output-dir",
            "/tmp/tasktrace-process-native-routing",
            "--cpuset-cpus",
            "0",
            "--strace-string-limit",
            "0",
        ];
        arguments.extend(selector);
        let result = run_native(&arguments, &fallback);
        assert_eq!(result.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&result.stderr);
        assert!(
            stderr.contains("--strace-string-limit must be positive"),
            "native validation was not reached: {stderr}"
        );
        assert!(!stderr.contains("legacy command implementation"));
        assert!(!stderr.contains("python-fallback-must-not-run"));
    }
}
