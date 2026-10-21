// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fake-runtime end-to-end proof for all native Enroot profile adapters.

use serde_json::{json, Value};
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
    for argument in "$@"; do printf '\t%s' "$argument"; done
    printf '\n'
}} >> "$state/calls"
case "${{1-}}" in
    version)
        printf 'enroot version 3.5.0\n'
        ;;
    list)
        if [ -e "$state/owned-container" ]; then cat "$state/owned-container"; fi
        ;;
    create)
        printf '%s\n' "$3" > "$state/owned-container"
        ;;
    remove)
        rm -f "$state/owned-container"
        ;;
    start)
        shift
        : > "$state/mappings"
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
                    remainder="${{mount#*:}}"
                    destination="${{remainder%%:*}}"
                    printf '%s|%s\n' "$source_path" "$destination" >> "$state/mappings"
                    case "$destination" in
                        /tasktrace-profile-entrypoint.sh) entrypoint="$source_path" ;;
                        /tasktrace-profile-output) output_dir="$source_path" ;;
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
                    printf 'unexpected start option: %s\n' "$1" >&2
                    exit 90
                    ;;
            esac
        done
        if [ -z "$entrypoint" ] || [ -z "$output_dir" ]; then
            printf 'missing profiler mounts\n' >&2
            exit 91
        fi
        awk -F'|' '{{ print length($2) "|" $1 "|" $2 }}' "$state/mappings" | sort -t'|' -k1,1nr > "$state/ordered-mappings"
        : > "$state/resolved-mappings"
        index=0
        while IFS='|' read -r _ source_path destination; do
            index=$((index + 1))
            token="__TASKTRACE_FAKE_MOUNT_${{index}}__"
            TASKTRACE_PROFILE_COMMAND="$(printf '%s' "$TASKTRACE_PROFILE_COMMAND" | sed "s|$destination|$token|g")"
            if [ -n "${{TASKTRACE_PROFILE_PREPARE_COMMAND:-}}" ]; then
                TASKTRACE_PROFILE_PREPARE_COMMAND="$(printf '%s' "$TASKTRACE_PROFILE_PREPARE_COMMAND" | sed "s|$destination|$token|g")"
            fi
            printf '%s|%s\n' "$token" "$source_path" >> "$state/resolved-mappings"
        done < "$state/ordered-mappings"
        while IFS='|' read -r token source_path; do
            TASKTRACE_PROFILE_COMMAND="$(printf '%s' "$TASKTRACE_PROFILE_COMMAND" | sed "s|$token|$source_path|g")"
            if [ -n "${{TASKTRACE_PROFILE_PREPARE_COMMAND:-}}" ]; then
                TASKTRACE_PROFILE_PREPARE_COMMAND="$(printf '%s' "$TASKTRACE_PROFILE_PREPARE_COMMAND" | sed "s|$token|$source_path|g")"
            fi
        done < "$state/resolved-mappings"
        export TASKTRACE_PROFILE_COMMAND TASKTRACE_PROFILE_PREPARE_COMMAND
        TASKTRACE_PROFILE_DIR="$output_dir"
        export TASKTRACE_PROFILE_DIR
        exec /bin/sh "$entrypoint"
        ;;
    *)
        printf 'unexpected fake Enroot operation: %s\n' "${{1-}}" >&2
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
}

fn run(command: &mut Command, label: &str, output_dir: &Path) -> Output {
    let output = command.output().unwrap();
    let retained = if output.status.success() {
        String::new()
    } else {
        [
            "container_profile/command.stderr.txt",
            "container_profile/command.stdout.txt",
            "container_profile/enroot_start.stderr.txt",
            "container_profile/enroot_start.stdout.txt",
        ]
        .iter()
        .map(|relative| {
            let path = output_dir.join(relative);
            format!(
                "{relative}: {}",
                fs::read_to_string(path).unwrap_or_default()
            )
        })
        .collect::<Vec<_>>()
        .join("\n")
    };
    assert!(
        output.status.success(),
        "{label} failed\nstdout={}\nstderr={}\n{retained}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    output
}

fn assert_wall_only_pipeline(output_dir: &Path) {
    let report: Value =
        serde_json::from_slice(&fs::read(output_dir.join("pipeline_report.json")).unwrap())
            .unwrap();
    assert_eq!(report["runtime"], "enroot", "pipeline report: {report}");
    assert_eq!(
        report["container_cpu"]["method"],
        "wall-clock-only-no-cgroup"
    );
    assert!(report["container_cpu"]["usage_sec"].is_null());
    assert_eq!(
        report["container_cpu"]["valid_for_wall_time_reporting"],
        true
    );
    assert_eq!(report["container_cpu"]["cpu_containment_enforced"], false);
    assert_eq!(report["profiling_valid"], true);
    let container: Value = serde_json::from_slice(
        &fs::read(output_dir.join("container_profile/profile_report.json")).unwrap(),
    )
    .unwrap();
    assert_eq!(container["runtime_report"]["cleanup_succeeded"], true);
    assert_eq!(container["runtime_report"]["cleanup_verified_absent"], true);
    assert!(!output_dir
        .join("container_profile/evidence_bundle.json")
        .exists());
}

#[test]
fn all_native_adapters_execute_through_fake_enroot_without_python() {
    let temporary = tempfile::tempdir().unwrap();
    let fake = FakeEnroot::create(temporary.path());
    let image = temporary.path().join("task.sqsh");
    fs::write(&image, b"adapter fake squashfs").unwrap();
    let workspace = temporary.path().join("workspace");
    fs::create_dir(&workspace).unwrap();
    let missing_python = temporary.path().join("python-must-not-run");

    let trace = temporary.path().join("trace.jsonl");
    let trace_records = [
        json!({
            "record_type": "trace",
            "schema_version": "tasktrace-cpu-trace.v1",
            "task_name": "enroot-tasktrace",
            "source_trial": "enroot-tasktrace__trial",
            "reward": 1,
            "limitations": [],
            "environment": {"action_timeout_sec": 5.0},
        }),
        json!({
            "record_type": "action",
            "action_index": 1,
            "command": "sleep 0.02; printf tasktrace-enroot > tasktrace.txt",
            "started_at_epoch": 1.0,
            "finished_at_epoch": 1.01,
            "wall_time_sec": 0.01,
            "returncode": 0,
            "exception_info": null,
        }),
    ];
    fs::write(
        &trace,
        trace_records
            .iter()
            .map(Value::to_string)
            .collect::<Vec<_>>()
            .join("\n")
            + "\n",
    )
    .unwrap();
    let tasktrace_output = temporary.path().join("tasktrace-output");
    let trace_run = run(
        Command::new(binary())
            .args(["profile", "trace"])
            .arg(&trace)
            .args(["--runtime", "enroot", "--enroot"])
            .arg(&fake.executable)
            .arg("--image")
            .arg(&image)
            .args(["--allocated-cpus", "1", "--strace-mode", "none"])
            .args(["--pacing", "unpaced", "--workdir"])
            .arg(&workspace)
            .arg("--output-dir")
            .arg(&tasktrace_output)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python),
        "TaskTrace Enroot profile",
        &tasktrace_output,
    );
    assert!(String::from_utf8_lossy(&trace_run.stdout).contains("CPU time: n/a"));
    assert_wall_only_pipeline(&tasktrace_output);

    let trajectory = temporary.path().join("trajectory.json");
    fs::write(
        &trajectory,
        serde_json::to_vec_pretty(&json!({
            "schema_version": "ATIF-v1.5",
            "session_id": "enroot-atif",
            "agent": {"name": "codex", "model_name": "fixture"},
            "steps": [{
                "step_id": 1,
                "timestamp": "2026-07-06T00:00:00Z",
                "tool_calls": [{
                    "tool_call_id": "atif-call",
                    "function_name": "exec_command",
                    "arguments": {"cmd": "sleep 0.02; printf atif-enroot > atif.txt"},
                }],
                "observation": {"results": [{
                    "source_call_id": "atif-call",
                    "content": "Chunk ID: fixture\\nProcess exited with code 0\\nFinal output:\\n",
                }]},
            }],
        }))
        .unwrap(),
    )
    .unwrap();
    let atif_output = temporary.path().join("atif-output");
    run(
        Command::new(binary())
            .args(["profile", "atif"])
            .arg(&trajectory)
            .args(["--runtime", "enroot", "--enroot"])
            .arg(&fake.executable)
            .args(["--controller-runtime", "rust-host-mount", "--image"])
            .arg(&image)
            .args(["--allocated-cpus", "1", "--strace-mode", "none"])
            .args(["--pacing", "unpaced", "--workdir"])
            .arg(&workspace)
            .arg("--patch-root")
            .arg(&workspace)
            .arg("--output-dir")
            .arg(&atif_output)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python),
        "ATIF Enroot profile",
        &atif_output,
    );
    assert_wall_only_pipeline(&atif_output);

    let transcript = temporary.path().join("transcript.jsonl");
    fs::write(
        &transcript,
        json!({
            "task_id": "enroot-transcript",
            "trial_name": "enroot-transcript.1",
            "turn": 1,
            "answer": "[Tool Call: exec]\n{\"command\":\"sleep 0.02; printf transcript-enroot > result.txt\"}",
        })
        .to_string()
            + "\n",
    )
    .unwrap();
    let transcript_output = temporary.path().join("transcript-output");
    run(
        Command::new(binary())
            .args(["profile", "transcript"])
            .arg(&transcript)
            .args(["--runtime", "enroot", "--enroot"])
            .arg(&fake.executable)
            .args([
                "--task-id",
                "enroot-transcript",
                "--trial-name",
                "enroot-transcript.1",
                "--image",
            ])
            .arg(&image)
            .args(["--allocated-cpus", "1", "--strace-mode", "none"])
            .arg("--workdir")
            .arg(&workspace)
            .arg("--output-dir")
            .arg(&transcript_output)
            .env("TASKTRACE_LEGACY_PYTHON", &missing_python),
        "transcript Enroot profile",
        &transcript_output,
    );
    assert_wall_only_pipeline(&transcript_output);

    assert!(!missing_python.exists());
    assert!(!fake.state.join("owned-container").exists());
}
