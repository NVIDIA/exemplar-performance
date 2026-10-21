// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Black-box native-only gates for the top-level `run` command.

use std::fs;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn available_cpu() -> u32 {
    let mut set: libc::cpu_set_t = unsafe { std::mem::zeroed() };
    let result =
        unsafe { libc::sched_getaffinity(0, std::mem::size_of::<libc::cpu_set_t>(), &mut set) };
    assert_eq!(result, 0, "read test-process CPU affinity");
    (0..libc::CPU_SETSIZE as usize)
        .find(|cpu| unsafe { libc::CPU_ISSET(*cpu, &set) })
        .map(|cpu| cpu as u32)
        .expect("test process has at least one allowed CPU")
}

fn fallback_probe(root: &Path) -> (PathBuf, PathBuf) {
    let executable = root.join("legacy-python-must-not-run");
    let marker = root.join("legacy-python-was-started");
    fs::write(
        &executable,
        format!(
            "#!/bin/sh\nprintf started > '{}'\nexit 97\n",
            marker.display()
        ),
    )
    .unwrap();
    fs::set_permissions(&executable, fs::Permissions::from_mode(0o700)).unwrap();
    (executable, marker)
}

fn run_manifest(manifest: &Path, fallback: &Path) -> Output {
    Command::new(binary())
        .args(["run"])
        .arg(manifest)
        .env("TASKTRACE_LEGACY_PYTHON", fallback)
        .output()
        .expect("start native tasktrace run command")
}

fn diagnostic(output: &Output) -> String {
    format!(
        "status={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

#[test]
fn v1_and_v2_preflight_natively_with_legacy_fallback_disabled() {
    for version in [1, 2] {
        let root = tempfile::tempdir().unwrap();
        let (fallback, marker) = fallback_probe(root.path());
        fs::write(root.path().join("trajectory.json"), b"{}\n").unwrap();
        let concurrency = if version == 2 { "concurrency: 1\n" } else { "" };
        let manifest = root.path().join("run.yaml");
        fs::write(
            &manifest,
            format!(
                "schema_version: tasktrace-run.v{version}\noutput: results\ncpus: '{}'\n{concurrency}tasks:\n  - id: native-only\n    atif: trajectory.json\n    image: unused-image\n",
                available_cpu()
            ),
        )
        .unwrap();

        let output = run_manifest(&manifest, &fallback);
        assert_eq!(output.status.code(), Some(2), "{}", diagnostic(&output));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains("native profile preflight failed"),
            "{}",
            diagnostic(&output)
        );
        assert!(!marker.exists(), "legacy Python fallback was started");
        assert!(
            !root.path().join("results").exists(),
            "preflight failure must not claim the output"
        );
    }
}

#[test]
fn retired_v3_controller_modes_fail_with_migration_error_before_output_claim() {
    for retired in ["image", "host-mount"] {
        let root = tempfile::tempdir().unwrap();
        let (fallback, marker) = fallback_probe(root.path());
        fs::write(root.path().join("trajectory.json"), b"{}\n").unwrap();
        let manifest = root.path().join("run.yaml");
        fs::write(
            &manifest,
            format!(
                "schema_version: tasktrace-run.v3\noutput: results\ndefaults:\n  pacing: unpaced\n  network_policy: no-network\n  profile_sets: [cpu-average]\n  controller_runtime: {retired}\n  controller_cpus: '{}'\n  action_timeout_sec: 5\ntasks:\n  - id: native-only\n    atif: trajectory.json\n    image: unused-image\nwaves:\n  - id: wave\n    launches:\n      - task: native-only\n        cpus: '{}'\n",
                available_cpu(),
                available_cpu()
            ),
        )
        .unwrap();

        let output = run_manifest(&manifest, &fallback);
        assert_eq!(output.status.code(), Some(2), "{}", diagnostic(&output));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(
            stderr.contains(&format!(
                "defaults.controller_runtime '{retired}' is a retired Python-backed run controller mode"
            )),
            "{}",
            diagnostic(&output)
        );
        assert!(
            stderr.contains("native-only tasktrace run requires 'rust-host-mount'"),
            "{}",
            diagnostic(&output)
        );
        assert!(
            stderr.contains("migrate the manifest"),
            "{}",
            diagnostic(&output)
        );
        assert!(!marker.exists(), "legacy Python fallback was started");
        assert!(
            !root.path().join("results").exists(),
            "retired controller mode must fail before output claim"
        );
    }
}
