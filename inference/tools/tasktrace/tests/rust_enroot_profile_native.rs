// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Public fallback-disabled gates for the native wall-time-only Enroot path.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::io::Write;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
}

fn shell_literal(path: &Path) -> String {
    format!("'{}'", path.to_string_lossy().replace('\'', "'\"'\"'"))
}

struct FakeEnroot {
    executable: PathBuf,
    state: PathBuf,
}

impl FakeEnroot {
    fn create(root: &Path) -> Self {
        let state = root.join("fake-enroot-state");
        fs::create_dir(&state).unwrap();
        let executable = root.join("fake-enroot");
        let state_literal = shell_literal(&state);
        let script = format!(
            r#"#!/bin/sh
set -eu
state={state_literal}
{{
    printf 'CALL'
    for argument in "$@"; do
        printf '\t%s' "$argument"
    done
    printf '\n'
}} >> "$state/calls"
case "${{1-}}" in
    version)
        printf 'enroot version 3.5.0\n'
        exit 0
        ;;
    list)
        if [ -e "$state/owned-container" ]; then
            cat "$state/owned-container"
        fi
        printf 'named-container RUNNING\n'
        exit 0
        ;;
    create)
        printf '%s\n' "$3" > "$state/owned-container"
        exit 0
        ;;
    remove)
        if [ -e "$state/fail-remove" ]; then
            printf 'synthetic cleanup failure\n' >&2
            exit 41
        fi
        rm -f "$state/owned-container"
        exit 0
        ;;
    start)
        if [ -e "$state/start-sleep" ]; then
            sleep 30 &
            child="$!"
            host_child="$(awk '/^NSpid:/ {{ print $2 }}' "/proc/$child/status")"
            printf '%s\n' "$host_child" > "$state/start-child-pid"
            : > "$state/start-entered"
            wait "$child"
            exit "$?"
        fi
        shift
        entrypoint=''
        output_dir=''
        while [ "$#" -gt 0 ]; do
            case "$1" in
                --rw|--root)
                    shift
                    ;;
                --mount)
                    mount="$2"
                    source_path="${{mount%%:*}}"
                    source_path="$(printf '%b' "$source_path")"
                    case "$mount" in
                        *:/tasktrace-profile-entrypoint.sh:none:x-create=auto,bind,ro)
                            entrypoint="$source_path"
                            ;;
                        *:/tasktrace-profile-output:none:x-create=auto,bind,rw)
                            output_dir="$source_path"
                            ;;
                    esac
                    shift 2
                    ;;
                --env)
                    export "$2"
                    shift 2
                    ;;
                --)
                    shift
                    break
                    ;;
                *)
                    printf 'unexpected enroot start option: %s\n' "$1" >&2
                    exit 90
                    ;;
            esac
        done
        if [ -z "$entrypoint" ] || [ -z "$output_dir" ]; then
            printf 'missing profiler mounts\n' >&2
            exit 91
        fi
        TASKTRACE_PROFILE_DIR="$output_dir"
        export TASKTRACE_PROFILE_DIR
        sleep 0.02
        exec /bin/sh "$entrypoint"
        ;;
    *)
        printf 'unexpected fake enroot operation: %s\n' "${{1-}}" >&2
        exit 92
        ;;
esac
"#
        );
        let staged = executable.with_extension("staged");
        let mut file = fs::File::create(&staged).unwrap();
        file.write_all(script.as_bytes()).unwrap();
        file.sync_all().unwrap();
        drop(file);
        let mut permissions = fs::metadata(&staged).unwrap().permissions();
        permissions.set_mode(0o755);
        fs::set_permissions(&staged, permissions).unwrap();
        fs::rename(staged, &executable).unwrap();
        Self { executable, state }
    }

    fn calls(&self) -> Vec<Vec<String>> {
        fs::read_to_string(self.state.join("calls"))
            .unwrap_or_default()
            .lines()
            .map(|line| line.split('\t').skip(1).map(ToOwned::to_owned).collect())
            .collect()
    }
}

fn run_profile(fake: &FakeEnroot, image: &str, output_dir: &Path, extra: &[&str]) -> Output {
    Command::new(binary())
        .args(["profile", "container", "--runtime", "enroot", "--enroot"])
        .arg(&fake.executable)
        .args(["--image", image, "--allocated-cpus", "1"])
        .args([
            "--strace-mode",
            "none",
            "--command",
            "sleep 0.02; printf native-enroot",
        ])
        .arg("--output-dir")
        .arg(output_dir)
        .args(extra)
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            output_dir.with_extension("python-must-not-run"),
        )
        .output()
        .unwrap()
}

fn wait_for_path(path: &Path) -> bool {
    for _ in 0..500 {
        if path.exists() {
            return true;
        }
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    false
}

#[test]
fn public_sqsh_profile_is_native_wall_only_and_cleans_up() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    let image_bytes = b"immutable public enroot image";
    fs::write(&image, image_bytes).unwrap();
    let output_dir = temporary.path().join("profile");

    let output = run_profile(
        &fake,
        image.to_str().unwrap(),
        &output_dir,
        &["--prepare-command", "printf prepared"],
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    assert!(stdout.contains("CPU time: n/a"), "{stdout}");
    assert!(
        stdout.contains("Average logical cores used: n/a"),
        "{stdout}"
    );
    assert!(!stdout.contains("unable to start legacy"), "{stdout}");
    assert_eq!(
        fs::read_to_string(output_dir.join("command.stdout.txt")).unwrap(),
        "native-enroot"
    );
    let report: Value =
        serde_json::from_slice(&fs::read(output_dir.join("profile_report.json")).unwrap()).unwrap();
    assert_eq!(report["runtime"], "enroot");
    assert_eq!(report["scope"], "disposable-image-container");
    assert_eq!(report["profiling_valid"], true);
    assert_eq!(report["cpu"]["method"], "wall-clock-only-no-cgroup");
    assert!(report["cpu"]["wall_sec"].as_f64().unwrap() > 0.0);
    assert!(report["cpu"]["usage_sec"].is_null());
    assert!(report["cpu"]["average_logical_cores_used"].is_null());
    assert_eq!(report["cpu"]["valid_for_wall_time_reporting"], true);
    assert_eq!(report["cpu"]["valid_for_cpu_performance"], false);
    assert_eq!(report["cpu"]["cpu_containment_enforced"], false);
    assert_eq!(report["cpu"]["cpu_affinity_enforced"], false);
    assert_eq!(report["execution_plan"]["runtime"], "enroot");
    assert_eq!(
        report["execution_plan"]["extensions"]["cpuset_contract"],
        "requested-taskset-launch-affinity-not-cpu-containment"
    );
    assert_eq!(
        report["image_identity"]["sha256"],
        format!("{:x}", Sha256::digest(image_bytes))
    );
    assert_eq!(
        report["canonical_lifecycle"]["clock_name"],
        "CLOCK_BOOTTIME"
    );
    assert!(
        report["canonical_lifecycle"]["end_ns"].as_u64().unwrap()
            > report["canonical_lifecycle"]["start_ns"].as_u64().unwrap()
    );
    assert_eq!(report["runtime_report"]["cleanup_succeeded"], true);
    assert_eq!(report["runtime_report"]["cleanup_verified_absent"], true);
    assert_eq!(
        report["runtime_report"]["create_process_group_cleanup"]["verified_gone"],
        true
    );
    assert_eq!(
        report["runtime_report"]["start_process_group_cleanup"]["verified_gone"],
        true
    );
    assert!(!output_dir.join("evidence_bundle.json").exists());
    assert!(!fake.state.join("owned-container").exists());
    let operations = fake
        .calls()
        .into_iter()
        .map(|call| call[0].clone())
        .collect::<Vec<_>>();
    assert_eq!(
        operations,
        ["version", "version", "create", "start", "remove", "list"]
    );
}

#[test]
fn public_named_enroot_container_stays_unowned() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let output_dir = temporary.path().join("named-profile");
    let output = run_profile(&fake, "named-container", &output_dir, &[]);
    assert!(
        output.status.success(),
        "stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    let report: Value =
        serde_json::from_slice(&fs::read(output_dir.join("profile_report.json")).unwrap()).unwrap();
    assert_eq!(report["image_identity"]["kind"], "created-container");
    assert_eq!(report["image_identity"]["immutable"], false);
    assert_eq!(report["scope"], "caller-owned-enroot-container");
    assert_eq!(report["runtime_report"]["cleanup_required"], false);
    assert!(report["runtime_report"]["create_process_group_cleanup"].is_null());
    assert_eq!(
        report["runtime_report"]["start_process_group_cleanup"]["verified_gone"],
        true
    );
    assert_eq!(
        fake.calls()
            .into_iter()
            .map(|call| call[0].clone())
            .collect::<Vec<_>>(),
        ["version", "list", "version", "list", "start"]
    );
}

#[test]
fn public_sigint_cancels_the_enroot_group_and_retains_invalid_cleanup_evidence() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    fs::write(fake.state.join("start-sleep"), "1\n").unwrap();
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"cancellation image").unwrap();
    let output_dir = temporary.path().join("cancelled-profile");

    let mut child = Command::new(binary())
        .args(["profile", "container", "--runtime", "enroot", "--enroot"])
        .arg(&fake.executable)
        .args(["--image", image.to_str().unwrap(), "--allocated-cpus", "1"])
        .args(["--strace-mode", "none", "--command", "printf unreachable"])
        .arg("--output-dir")
        .arg(&output_dir)
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            temporary.path().join("python-must-not-run"),
        )
        .spawn()
        .unwrap();
    assert!(
        wait_for_path(&fake.state.join("start-entered")),
        "fake Enroot never entered start"
    );
    assert_eq!(
        unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGINT) },
        0
    );
    let status = child.wait().unwrap();
    assert_eq!(status.code(), Some(130));

    let evidence: Value =
        serde_json::from_slice(&fs::read(output_dir.join("enroot_runtime_failure.json")).unwrap())
            .unwrap();
    assert_eq!(evidence["status"], "cancelled");
    assert_eq!(evidence["phase"], "start");
    assert_eq!(evidence["profiling_valid"], false);
    assert_eq!(evidence["cleanup_required"], true);
    assert_eq!(evidence["cleanup_succeeded"], true);
    assert_eq!(evidence["cleanup"]["verified_absent"], true);
    assert_eq!(evidence["host_process_group"]["verified_gone"], true);
    assert!(!output_dir.join("profile_report.json").exists());
    assert!(!fake.state.join("owned-container").exists());
    assert_eq!(
        fake.calls()
            .into_iter()
            .map(|call| call[0].clone())
            .collect::<Vec<_>>(),
        ["version", "version", "create", "start", "remove", "list"]
    );
}

#[test]
fn unsupported_enroot_modes_fail_locally_without_python_fallback() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"squashfs").unwrap();
    for (label, arguments) in [
        ("cpu-limit", vec!["--cpu-limit", "1"]),
        ("profile-set", vec!["--profile-set", "cpu-average"]),
        ("scheduler", vec!["--scheduler-concurrency"]),
    ] {
        let output_dir = temporary.path().join(label);
        let output = run_profile(&fake, image.to_str().unwrap(), &output_dir, &arguments);
        assert_eq!(output.status.code(), Some(2));
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("Docker"), "{label}: {stderr}");
        assert!(
            !stderr.contains("unable to start legacy"),
            "{label}: {stderr}"
        );
        assert!(!output_dir.exists());
    }
}
