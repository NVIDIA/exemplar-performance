// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Differential gates for the native external-BPF process proof commands.

use serde_json::json;
use sha2::{Digest, Sha256};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use tempfile::TempDir;

const ROOT_PREFIX: &str = "/tmp/tasktrace-python -m offline_replayer.adapters.atif_replay ";

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
    env::join_paths(paths).expect("construct PYTHONPATH")
}

fn native(arguments: &[&OsStr], root: &Path) -> Output {
    Command::new(binary())
        .args(arguments)
        .current_dir(repository())
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            root.join("python-fallback-must-not-run"),
        )
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run native BPF report")
}

fn python_module(module: &str, arguments: &[&OsStr]) -> Output {
    Command::new(python())
        .arg("-m")
        .arg(module)
        .args(arguments)
        .current_dir(repository())
        .env("PYTHONPATH", python_path())
        .env("PYTHONDONTWRITEBYTECODE", "1")
        .output()
        .expect("run Python BPF report")
}

fn diagnostic(output: &Output) -> String {
    format!(
        "exit {:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    )
}

struct Fixture {
    raw: PathBuf,
    stderr: PathBuf,
    pipeline: PathBuf,
    expected: PathBuf,
    collector: PathBuf,
}

impl Fixture {
    fn create(root: &Path) -> Self {
        let raw = root.join("events.csv");
        fs::write(
            &raw,
            concat!(
                "EPOCH,COMM,PPID,PID,CPU,RET,ARGS\n",
                "100.000001,noise,1,90,2,0,\n",
                "100.000002,python,10,100,2,0,/tmp/tasktrace-python -m offline_replayer.adapters.atif_replay /input\n",
                "100.000003,python,100,101,2,,[fork]\n",
                "100.000004,sh,100,101,3,-2,/missing/tool arg\n",
                "100.000005,echo,100,101,3,0,/bin/echo hello\n",
                "100.000006,echo,100,101,3,,[exit_group]\n",
                "100.000007,python,10,100,2,,[exit_group]\n",
            ),
        )
        .unwrap();
        let stderr = root.join("tracer.stderr");
        fs::write(&stderr, "AGENTSCRY_LOST_EVENTS=0\n").unwrap();
        let pipeline = root.join("pipeline.json");
        fs::write(
            &pipeline,
            serde_json::to_vec(&json!({
                "status": "valid",
                "profiling_valid": true,
                "network": "none",
                "source_trajectory_sha256": "a".repeat(64),
                "replay_summary": {
                    "actions_timed_out": 0,
                    "source_outcome_mismatch_count": 0,
                    "actions_skipped_after_timeout": 0,
                },
                "execution_plan": {"extensions": {"cpuset_cpus": "2-3"}},
                "image_identity": {
                    "id": "sha256:image",
                    "repo_digests": ["example/task@sha256:digest"],
                },
            }))
            .unwrap(),
        )
        .unwrap();
        let tree = concat!(
            "{\"children\":[{\"children\":[],\"exec_chain\":[\"/bin/echo\"]}],",
            "\"exec_chain\":[\"/tmp/tasktrace-python\"]}"
        );
        let tree_hash = format!("{:x}", Sha256::digest(tree.as_bytes()));
        let expected = root.join("expected.json");
        fs::write(
            &expected,
            serde_json::to_vec(&json!({
                "schema_version": "tasktrace-representative-process-expectation.v1",
                "source_trajectory_sha256": "a".repeat(64),
                "image_id": "sha256:image",
                "image_repo_digest": "example/task@sha256:digest",
                "cpuset_cpus": "2-3",
                "root_argv_prefix": ROOT_PREFIX,
                "task_count": 2,
                "successful_exec_count": 2,
                "failed_exec_attempt_count": 1,
                "canonical_tree_sha256": tree_hash,
                "binary_inventory": [
                    {"binary": "/bin/echo", "exec_count": 1},
                    {"binary": "/tmp/tasktrace-python", "exec_count": 1},
                ],
            }))
            .unwrap(),
        )
        .unwrap();
        let collector = root.join("collector.py");
        fs::write(&collector, "# reviewed collector\n").unwrap();
        Self {
            raw,
            stderr,
            pipeline,
            expected,
            collector,
        }
    }

    fn inventory_arguments<'a>(&'a self, output: &'a Path) -> Vec<&'a OsStr> {
        vec![
            OsStr::new("report"),
            OsStr::new("process-inventory-bpf"),
            self.raw.as_os_str(),
            self.stderr.as_os_str(),
            self.pipeline.as_os_str(),
            self.expected.as_os_str(),
            self.collector.as_os_str(),
            OsStr::new("--output"),
            output.as_os_str(),
        ]
    }

    fn python_inventory_arguments<'a>(&'a self, output: &'a Path) -> Vec<&'a OsStr> {
        let mut arguments = self.inventory_arguments(output);
        arguments.drain(..2);
        arguments
    }
}

#[test]
fn inventory_and_repeatability_artifacts_match_python_byte_for_byte() {
    let root = TempDir::new().expect("create BPF parity root");
    let fixture = Fixture::create(root.path());
    let native_proof = root.path().join("native-proof.json");
    let python_proof = root.path().join("python-proof.json");
    let native_result = native(&fixture.inventory_arguments(&native_proof), root.path());
    let python_result = python_module(
        "offline_replayer.reporting.process_bpf_inventory",
        &fixture.python_inventory_arguments(&python_proof),
    );
    assert_eq!(
        native_result.status.code(),
        Some(0),
        "native inventory failed: {}",
        diagnostic(&native_result)
    );
    assert_eq!(
        python_result.status.code(),
        Some(0),
        "Python inventory failed: {}",
        diagnostic(&python_result)
    );
    assert_eq!(
        fs::read(&native_proof).unwrap(),
        fs::read(&python_proof).unwrap(),
        "inventory proof bytes differ"
    );

    let proof_copy = root.path().join("proof-copy.json");
    fs::copy(&native_proof, &proof_copy).unwrap();
    let native_repeated = root.path().join("native-repeatability.json");
    let python_repeated = root.path().join("python-repeatability.json");
    let native_repeat_args = [
        OsStr::new("report"),
        OsStr::new("process-inventory-bpf-repeatability"),
        native_proof.as_os_str(),
        proof_copy.as_os_str(),
        OsStr::new("--output"),
        native_repeated.as_os_str(),
    ];
    let python_repeat_args = [
        native_proof.as_os_str(),
        proof_copy.as_os_str(),
        OsStr::new("--output"),
        python_repeated.as_os_str(),
    ];
    let native_result = native(&native_repeat_args, root.path());
    let python_result = python_module(
        "offline_replayer.reporting.process_bpf_repeatability",
        &python_repeat_args,
    );
    assert_eq!(
        native_result.status.code(),
        Some(0),
        "native repeatability failed: {}",
        diagnostic(&native_result)
    );
    assert_eq!(
        python_result.status.code(),
        Some(0),
        "Python repeatability failed: {}",
        diagnostic(&python_result)
    );
    assert_eq!(
        fs::read(&native_repeated).unwrap(),
        fs::read(&python_repeated).unwrap(),
        "repeatability report bytes differ"
    );
}

#[test]
fn lossy_capture_fails_closed_without_python_fallback() {
    let root = TempDir::new().expect("create lossy BPF root");
    let fixture = Fixture::create(root.path());
    fs::write(&fixture.stderr, "AGENTSCRY_LOST_EVENTS=3\n").unwrap();
    let output = root.path().join("must-not-exist.json");
    let result = native(&fixture.inventory_arguments(&output), root.path());
    assert_eq!(result.status.code(), Some(2), "{}", diagnostic(&result));
    assert!(
        String::from_utf8_lossy(&result.stderr).contains("process trace dropped 3 event(s)"),
        "{}",
        diagnostic(&result)
    );
    assert!(!output.exists());
}

#[test]
fn expectation_mismatch_matches_python_and_returns_one() {
    let root = TempDir::new().expect("create mismatch BPF root");
    let fixture = Fixture::create(root.path());
    let mut expected: serde_json::Value =
        serde_json::from_slice(&fs::read(&fixture.expected).unwrap()).unwrap();
    expected["task_count"] = json!(999);
    fs::write(&fixture.expected, serde_json::to_vec(&expected).unwrap()).unwrap();
    let native_proof = root.path().join("native-mismatch.json");
    let python_proof = root.path().join("python-mismatch.json");
    let native_result = native(&fixture.inventory_arguments(&native_proof), root.path());
    let python_result = python_module(
        "offline_replayer.reporting.process_bpf_inventory",
        &fixture.python_inventory_arguments(&python_proof),
    );
    assert_eq!(
        native_result.status.code(),
        Some(1),
        "{}",
        diagnostic(&native_result)
    );
    assert_eq!(
        python_result.status.code(),
        Some(1),
        "{}",
        diagnostic(&python_result)
    );
    assert_eq!(
        fs::read(native_proof).unwrap(),
        fs::read(python_proof).unwrap()
    );
}
