// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in restricted Kind/Kubernetes gate for the native current-container runtime.
//!
//! Required environment:
//!
//! - `TASKTRACE_RUST_KIND` (defaults to `kind`)
//! - `TASKTRACE_RUST_KUBECTL` (defaults to `kubectl`)
//! - `TASKTRACE_RUST_KIND_TEST_IMAGE` (a preloaded Linux/amd64 task image)
//! - `TASKTRACE_RUST_KIND_CPUSET` (runner-owned host logical CPUs)
//!
//! The gate writes its manifest, external image/controller identities, reports,
//! and teardown proof beneath `TASKTRACE_RUST_KIND_ARTIFACT_DIR` when supplied.

use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::env;
use std::ffi::{OsStr, OsString};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const NODE_IMAGE: &str =
    "kindest/node:v1.36.1@sha256:3489c7674813ba5d8b1a9977baea8a6e553784dab7b84759d1014dbd78f7ebd5";
const READY_MARKER: &str = "TASKTRACE_RUST_CURRENT_KIND_READY";

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
        .canonicalize()
        .expect("canonicalize native tasktrace binary")
}

fn file_sha256(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(fs::read(path).expect("read file for SHA-256"))
    )
}

fn output(command: &mut Command, description: &str) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("unable to {description}: {error}"))
}

fn diagnostic(output: &Output) -> String {
    format!(
        "exit={:?}\nstdout:\n{}\nstderr:\n{}",
        output.status.code(),
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_success(output: &Output, description: &str) {
    assert!(
        output.status.success(),
        "{description} failed: {}",
        diagnostic(output)
    );
}

fn write_output(root: &Path, stem: &str, output: &Output) {
    fs::write(root.join(format!("{stem}.stdout.txt")), &output.stdout).unwrap();
    fs::write(root.join(format!("{stem}.stderr.txt")), &output.stderr).unwrap();
}

fn command_with_args(
    program: &Path,
    arguments: impl IntoIterator<Item = impl AsRef<OsStr>>,
) -> Command {
    let mut command = Command::new(program);
    command.args(arguments);
    command
}

struct ClusterGuard {
    kind: PathBuf,
    name: String,
    artifacts: PathBuf,
    deleted: bool,
}

impl ClusterGuard {
    fn delete(&mut self) {
        if self.deleted {
            return;
        }
        let result = output(
            &mut command_with_args(&self.kind, ["delete", "cluster", "--name", &self.name]),
            "delete Kind cluster",
        );
        write_output(&self.artifacts, "kind-delete", &result);
        self.deleted = result.status.success();
        assert_success(&result, "Kind cluster teardown");
    }
}

impl Drop for ClusterGuard {
    fn drop(&mut self) {
        if self.deleted {
            return;
        }
        let result = Command::new(&self.kind)
            .args(["delete", "cluster", "--name", &self.name])
            .output();
        if let Ok(result) = result {
            let _ = fs::write(
                self.artifacts.join("kind-delete-drop.stdout.txt"),
                result.stdout,
            );
            let _ = fs::write(
                self.artifacts.join("kind-delete-drop.stderr.txt"),
                result.stderr,
            );
        }
    }
}

fn kubectl(
    executable: &Path,
    kubeconfig: &Path,
    arguments: impl IntoIterator<Item = impl AsRef<OsStr>>,
) -> Command {
    let mut command = command_with_args(executable, arguments);
    command.env("KUBECONFIG", kubeconfig);
    command
}

fn kubectl_cat(executable: &Path, kubeconfig: &Path, pod: &str, path: &str) -> Vec<u8> {
    let result = output(
        &mut kubectl(executable, kubeconfig, ["exec", pod, "--", "cat", path]),
        &format!("read {path} from Kubernetes pod"),
    );
    assert_success(&result, &format!("kubectl exec cat {path}"));
    result.stdout
}

fn wait_for_pod_marker(executable: &Path, kubeconfig: &Path, pod: &str, marker: &str) -> Output {
    let deadline = Instant::now() + Duration::from_secs(240);
    loop {
        let logs = output(
            &mut kubectl(executable, kubeconfig, ["logs", pod]),
            "read profiling pod logs",
        );
        if logs.status.success() && String::from_utf8_lossy(&logs.stdout).contains(marker) {
            return logs;
        }
        let status = output(
            &mut kubectl(executable, kubeconfig, ["get", "pod", pod, "-o", "json"]),
            "inspect profiling pod status",
        );
        if status.status.success() {
            let status_json = parse_json(&status.stdout, "profiling pod status");
            assert_ne!(
                status_json["status"]["phase"],
                "Failed",
                "pod failed: {}",
                diagnostic(&logs)
            );
        }
        assert!(
            Instant::now() < deadline,
            "timed out waiting for profiling pod: {}",
            diagnostic(&logs)
        );
        thread::sleep(Duration::from_secs(1));
    }
}

fn delete_pod(executable: &Path, kubeconfig: &Path, pod: &str) -> Output {
    output(
        &mut kubectl(
            executable,
            kubeconfig,
            ["delete", "pod", pod, "--wait=true", "--timeout=60s"],
        ),
        "delete profiling pod",
    )
}

fn derived_pod(base: &Value, name: &str, script: &str) -> Value {
    let mut pod = base.clone();
    pod["metadata"]["name"] = Value::String(name.to_owned());
    pod["spec"]["containers"][0]["command"] = json!(["/bin/sh", "-c", script]);
    pod
}

fn parse_json(bytes: &[u8], description: &str) -> Value {
    serde_json::from_slice(bytes).unwrap_or_else(|error| {
        panic!(
            "unable to parse {description}: {error}\n{}",
            String::from_utf8_lossy(bytes)
        )
    })
}

fn cpu_stat(bytes: &[u8]) -> Value {
    let mut result = serde_json::Map::new();
    for line in String::from_utf8_lossy(bytes).lines() {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        if fields.len() == 2 {
            result.insert(
                fields[0].to_owned(),
                json!(fields[1].parse::<u64>().expect("numeric cpu.stat value")),
            );
        }
    }
    Value::Object(result)
}

#[test]
#[ignore = "requires Docker, pinned Kind/Kubernetes binaries, cgroup v2, and runner-owned CPUs"]
fn restricted_kind_proves_native_current_container_contract_and_teardown() {
    let kind = PathBuf::from(env::var_os("TASKTRACE_RUST_KIND").unwrap_or_else(|| "kind".into()));
    let kubectl_bin =
        PathBuf::from(env::var_os("TASKTRACE_RUST_KUBECTL").unwrap_or_else(|| "kubectl".into()));
    let image = env::var("TASKTRACE_RUST_KIND_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_KIND_TEST_IMAGE to a preloaded task image");
    let cpuset = env::var("TASKTRACE_RUST_KIND_CPUSET")
        .expect("set TASKTRACE_RUST_KIND_CPUSET to runner-owned host CPUs");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let cluster = format!("tasktrace-rust-{}-{nonce}", std::process::id());
    let cluster = cluster.chars().take(48).collect::<String>();
    let artifacts = env::var_os("TASKTRACE_RUST_KIND_ARTIFACT_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(format!("/tmp/{cluster}-artifacts")));
    fs::create_dir(&artifacts).expect("create new Kind artifact directory");
    let kubeconfig = artifacts.join("kubeconfig");
    let controller = binary();

    for (stem, mut command) in [
        ("kind-version", command_with_args(&kind, ["version"])),
        (
            "kubectl-version",
            command_with_args(&kubectl_bin, ["version", "--client", "-o", "json"]),
        ),
        ("docker-server-version", {
            let mut command = Command::new("docker");
            command.args(["version", "--format", "{{.Server.Version}}"]);
            command
        }),
    ] {
        let result = output(&mut command, stem);
        write_output(&artifacts, stem, &result);
        assert_success(&result, stem);
    }
    let kind_version = fs::read_to_string(artifacts.join("kind-version.stdout.txt")).unwrap();
    assert!(
        kind_version.contains("v0.32.0"),
        "unexpected Kind version: {kind_version}"
    );
    let kubectl_version: Value = parse_json(
        &fs::read(artifacts.join("kubectl-version.stdout.txt")).unwrap(),
        "kubectl client version",
    );
    assert_eq!(kubectl_version["clientVersion"]["gitVersion"], "v1.36.1");

    let image_inspect = output(
        Command::new("docker").args(["image", "inspect", &image]),
        "inspect task image",
    );
    write_output(&artifacts, "task-image-inspect", &image_inspect);
    assert_success(&image_inspect, "task image inspection");
    let image_metadata = parse_json(&image_inspect.stdout, "task image inspection");
    let expected_image_id = image_metadata[0]["Id"]
        .as_str()
        .expect("task image has immutable local ID")
        .to_owned();
    assert!(expected_image_id.starts_with("sha256:") && expected_image_id.len() == 71);
    assert_eq!(image_metadata[0]["Architecture"], "amd64");
    assert_eq!(image_metadata[0]["Os"], "linux");
    fs::write(
        artifacts.join("controller-identity.json"),
        serde_json::to_vec_pretty(&json!({
            "path": controller,
            "sha256": file_sha256(&controller),
            "architecture": env::consts::ARCH,
        }))
        .unwrap(),
    )
    .unwrap();

    let kind_config = json!({
        "kind": "Cluster",
        "apiVersion": "kind.x-k8s.io/v1alpha4",
        "nodes": [{
            "role": "control-plane",
            "extraMounts": [{
                "hostPath": controller,
                "containerPath": "/tasktrace-host/tasktrace",
                "readOnly": true,
            }]
        }]
    });
    let kind_config_path = artifacts.join("kind-config.json");
    fs::write(
        &kind_config_path,
        serde_json::to_vec_pretty(&kind_config).unwrap(),
    )
    .unwrap();
    // Install the teardown guard before creation so a partially initialized
    // API server or node container cannot survive a failed `kind create`.
    let mut guard = ClusterGuard {
        kind: kind.clone(),
        name: cluster.clone(),
        artifacts: artifacts.clone(),
        deleted: false,
    };
    let create = output(
        command_with_args(
            &kind,
            [
                OsString::from("create"),
                OsString::from("cluster"),
                OsString::from("--name"),
                OsString::from(&cluster),
                OsString::from("--image"),
                OsString::from(NODE_IMAGE),
                OsString::from("--config"),
                kind_config_path.as_os_str().to_owned(),
                OsString::from("--kubeconfig"),
                kubeconfig.as_os_str().to_owned(),
                OsString::from("--wait"),
                OsString::from("180s"),
            ],
        )
        .env("KIND_EXPERIMENTAL_PROVIDER", "docker"),
        "create restricted Kind cluster",
    );
    write_output(&artifacts, "kind-create", &create);
    assert_success(&create, "Kind cluster creation");
    let node = format!("{cluster}-control-plane");
    let update = output(
        Command::new("docker").args(["update", "--cpuset-cpus", &cpuset, "--cpus", "4", &node]),
        "constrain Kind node CPUs",
    );
    write_output(&artifacts, "kind-node-cpu-update", &update);
    assert_success(&update, "Kind node CPU constraint");
    let node_inspect = output(
        Command::new("docker").args(["inspect", &node]),
        "inspect Kind node",
    );
    write_output(&artifacts, "kind-node-docker-inspect", &node_inspect);
    assert_success(&node_inspect, "Kind node inspection");

    let load = output(
        &mut command_with_args(&kind, ["load", "docker-image", &image, "--name", &cluster]),
        "load task image into Kind",
    );
    write_output(&artifacts, "kind-load-image", &load);
    assert_success(&load, "Kind image load");

    let trajectory = json!({
        "schema_version": "ATIF-v1.5",
        "session_id": "rust-current-kind-e2e",
        "agent": {"name": "native-rust-e2e"},
        "steps": [{
            "step_id": 1,
            "timestamp": "2026-07-06T00:00:00Z",
            "tool_calls": [{
                "tool_call_id": "quota-contention",
                "function_name": "exec_command",
                "arguments": {
                    "cmd": "for n in 1 2 3 4; do (i=0; while [ \"$i\" -lt 300000 ]; do i=$((i+1)); done) & done; wait; printf 'kind-native-rust\\n' > result.txt",
                    "yield_time_ms": 1000,
                    "max_output_tokens": 12000
                }
            }],
            "observation": {"results": [{
                "source_call_id": "quota-contention",
                "content": "Chunk ID: fixture\\nWall time: 3 seconds\\nProcess exited with code 0\\nFinal output:\\n"
            }]}
        }]
    });
    let config_map = json!({
        "apiVersion": "v1",
        "kind": "ConfigMap",
        "metadata": {"name": "tasktrace-rust-current-input"},
        "data": {"trajectory.json": format!("{}\n", serde_json::to_string_pretty(&trajectory).unwrap())},
    });
    let config_map_path = artifacts.join("input-configmap.json");
    fs::write(
        &config_map_path,
        serde_json::to_vec_pretty(&config_map).unwrap(),
    )
    .unwrap();
    let apply_input = output(
        &mut kubectl(
            &kubectl_bin,
            &kubeconfig,
            ["apply", "-f", config_map_path.to_str().unwrap()],
        ),
        "apply trajectory ConfigMap",
    );
    write_output(&artifacts, "kubectl-apply-input", &apply_input);
    assert_success(&apply_input, "trajectory ConfigMap creation");

    let script = format!(
        "set -eu\n/usr/local/bin/tasktrace-rust profile atif /input/trajectory.json --runtime current --current-container-exclusive --allocated-cpus 1 --pacing unpaced --workdir /workspace --shell /bin/bash --strace-mode none --timeout-sec 30 --patch-root /workspace --output-dir /results/profile > /results/cli.stdout.txt 2> /results/cli.stderr.txt\nprintf '%s\\n' {READY_MARKER}\nsleep 600\n"
    );
    let pod = json!({
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "tasktrace-rust-current",
            "labels": {"app": "tasktrace-rust-current-e2e"},
        },
        "spec": {
            "automountServiceAccountToken": false,
            "enableServiceLinks": false,
            "restartPolicy": "Never",
            "terminationGracePeriodSeconds": 1,
            "securityContext": {
                "runAsNonRoot": true,
                "runAsUser": 65534,
                "runAsGroup": 65534,
                "fsGroup": 65534,
                "fsGroupChangePolicy": "OnRootMismatch",
            },
            "containers": [{
                "name": "profiler",
                "image": image,
                "imagePullPolicy": "Never",
                "command": ["/bin/sh", "-c", script],
                "workingDir": "/workspace",
                "env": [
                    {"name": "PATH", "value": "/usr/local/bin:/usr/bin:/bin"},
                    {"name": "VIRTUAL_ENV", "value": "/usr/local"},
                    {"name": "VIRTUAL_ENV_PROMPT", "value": "(controller)"},
                    {"name": "PYTHONPATH", "value": "/usr/local/source:/task-preserve"},
                    {"name": "UV_RUN_RECURSION_DEPTH", "value": "1"},
                    {"name": "TASKTRACE_LEGACY_PYTHON", "value": "/python-must-not-run"},
                    {"name": "CONTAINER_NAME", "value": "profiler"},
                    {"name": "POD_NAME", "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}}},
                    {"name": "POD_NAMESPACE", "valueFrom": {"fieldRef": {"fieldPath": "metadata.namespace"}}},
                    {"name": "POD_UID", "valueFrom": {"fieldRef": {"fieldPath": "metadata.uid"}}},
                    {"name": "NODE_NAME", "valueFrom": {"fieldRef": {"fieldPath": "spec.nodeName"}}},
                ],
                "resources": {
                    "requests": {"cpu": "1", "memory": "128Mi"},
                    "limits": {"cpu": "1", "memory": "256Mi"},
                },
                "securityContext": {
                    "allowPrivilegeEscalation": false,
                    "readOnlyRootFilesystem": true,
                    "capabilities": {"drop": ["ALL"]},
                    "seccompProfile": {"type": "RuntimeDefault"},
                },
                "volumeMounts": [
                    {"name": "controller", "mountPath": "/usr/local/bin/tasktrace-rust", "readOnly": true},
                    {"name": "input", "mountPath": "/input", "readOnly": true},
                    {"name": "workspace", "mountPath": "/workspace"},
                    {"name": "results", "mountPath": "/results"},
                    {"name": "tmp", "mountPath": "/tmp"},
                ],
            }],
            "volumes": [
                {"name": "controller", "hostPath": {"path": "/tasktrace-host/tasktrace", "type": "File"}},
                {"name": "input", "configMap": {"name": "tasktrace-rust-current-input"}},
                {"name": "workspace", "emptyDir": {}},
                {"name": "results", "emptyDir": {}},
                {"name": "tmp", "emptyDir": {}},
            ],
        },
    });
    let pod_path = artifacts.join("pod-manifest.json");
    fs::write(&pod_path, serde_json::to_vec_pretty(&pod).unwrap()).unwrap();
    let apply_pod = output(
        &mut kubectl(
            &kubectl_bin,
            &kubeconfig,
            ["apply", "-f", pod_path.to_str().unwrap()],
        ),
        "apply restricted profiling pod",
    );
    write_output(&artifacts, "kubectl-apply-pod", &apply_pod);
    assert_success(&apply_pod, "restricted profiling pod creation");

    let last_logs = wait_for_pod_marker(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        READY_MARKER,
    );
    write_output(&artifacts, "pod-logs", &last_logs);

    let pod_status_output = output(
        &mut kubectl(
            &kubectl_bin,
            &kubeconfig,
            ["get", "pod", "tasktrace-rust-current", "-o", "json"],
        ),
        "capture profiling pod status",
    );
    assert_success(&pod_status_output, "profiling pod status capture");
    fs::write(artifacts.join("pod-status.json"), &pod_status_output.stdout).unwrap();
    let pod_status = parse_json(&pod_status_output.stdout, "profiling pod status");
    let observed_image_id = pod_status["status"]["containerStatuses"][0]["imageID"]
        .as_str()
        .expect("pod has observed imageID");
    // `kind load docker-image` imports the Docker config into a new OCI
    // manifest, so Kubernetes reports the imported *manifest* digest rather
    // than Docker's config/image ID.  Prove the content-bound relationship by
    // reading that exact manifest from the node's containerd store and
    // requiring its config digest to equal the externally recorded image ID.
    let observed_manifest_digest = observed_image_id
        .rsplit_once('@')
        .map_or(observed_image_id, |(_, digest)| digest);
    assert!(
        observed_manifest_digest.starts_with("sha256:"),
        "pod has no content-addressed image ID: {observed_image_id:?}"
    );
    let loaded_manifest = output(
        Command::new("docker").args([
            "exec",
            &node,
            "ctr",
            "--namespace",
            "k8s.io",
            "content",
            "get",
            observed_manifest_digest,
        ]),
        "read loaded OCI image manifest from Kind node",
    );
    write_output(&artifacts, "kind-loaded-image-manifest", &loaded_manifest);
    assert_success(&loaded_manifest, "Kind loaded-image manifest inspection");
    let loaded_root_json = parse_json(&loaded_manifest.stdout, "Kind loaded-image manifest root");
    let concrete_manifest_json = if loaded_root_json.get("config").is_some() {
        loaded_root_json
    } else {
        let concrete_digest = loaded_root_json["manifests"][0]["digest"]
            .as_str()
            .expect("loaded OCI index has a concrete manifest digest");
        let concrete_manifest = output(
            Command::new("docker").args([
                "exec",
                &node,
                "ctr",
                "--namespace",
                "k8s.io",
                "content",
                "get",
                concrete_digest,
            ]),
            "read concrete OCI image manifest from Kind node",
        );
        write_output(
            &artifacts,
            "kind-loaded-image-concrete-manifest",
            &concrete_manifest,
        );
        assert_success(
            &concrete_manifest,
            "Kind concrete loaded-image manifest inspection",
        );
        parse_json(
            &concrete_manifest.stdout,
            "Kind concrete loaded-image manifest",
        )
    };
    assert_eq!(
        concrete_manifest_json["config"]["digest"], expected_image_id,
        "Kind's pod manifest does not reference the externally inspected Docker image config"
    );

    let pipeline_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/results/profile/pipeline_report.json",
    );
    let profile_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/results/profile/container_profile/profile_report.json",
    );
    let replay_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/results/profile/replay/replay_report.json",
    );
    let before_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/results/profile/container_profile/cpu.stat.before",
    );
    let after_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/results/profile/container_profile/cpu.stat.after",
    );
    for (name, bytes) in [
        ("pipeline_report.json", &pipeline_bytes),
        ("profile_report.json", &profile_bytes),
        ("replay_report.json", &replay_bytes),
        ("cpu.stat.before", &before_bytes),
        ("cpu.stat.after", &after_bytes),
    ] {
        fs::write(artifacts.join(name), bytes).unwrap();
    }
    let pipeline = parse_json(&pipeline_bytes, "pipeline report");
    let profile = parse_json(&profile_bytes, "container profile report");
    let replay = parse_json(&replay_bytes, "replay report");
    assert_eq!(pipeline["runtime"], "current");
    assert_eq!(pipeline["measurement_scope"], "current-container");
    assert_eq!(pipeline["status"], "valid");
    assert_eq!(pipeline["image_identity"], Value::Null);
    assert_eq!(
        pipeline["controller_identity"]["kind"],
        "rust-current-container-binary"
    );
    assert_eq!(replay["summary"]["source_outcome_mismatch_count"], 0);
    assert_eq!(
        profile["environment_identity"]["controller_language"],
        "rust"
    );
    assert_eq!(
        profile["measurement_validity"]["valid_for_average_cpu_reporting"],
        true
    );
    assert_eq!(
        profile["measurement_validity"]["exclusive_single_task_container_asserted"],
        true
    );
    assert_eq!(profile["measurement_validity"]["cgroup_path_stable"], true);
    assert_eq!(profile["measurement_validity"]["allocation_stable"], true);
    assert_eq!(
        profile["measurement_validity"]["potential_unrelated_processes_before"],
        json!([])
    );
    assert_eq!(
        profile["measurement_validity"]["processes_outside_controller_ancestry_after"],
        json!([])
    );
    assert_eq!(
        profile["task_environment"]["unset_variables"],
        json!([
            "PYTHONPATH",
            "UV_RUN_RECURSION_DEPTH",
            "VIRTUAL_ENV",
            "VIRTUAL_ENV_PROMPT"
        ])
    );
    assert_eq!(
        profile["task_environment"]["overridden_variables"],
        json!(["PATH", "PYTHONPATH"])
    );
    assert!(profile["command"]
        .as_str()
        .unwrap()
        .starts_with("/usr/local/bin/tasktrace-rust replay atif"));
    assert!(!profile["command"].as_str().unwrap().contains("python"));
    assert_eq!(profile["cpu"]["enforced_logical_cpu_limit"], 1.0);
    assert_eq!(profile["cpu"]["allocated_logical_cpus"], 1.0);
    let average = profile["cpu"]["average_logical_cores_used"]
        .as_f64()
        .unwrap();
    assert!(
        (0.75..=1.25).contains(&average),
        "unexpected quota-bound average core use: {average}"
    );
    assert!(
        profile["cpu"]["throttling"]["throttled_periods"]
            .as_u64()
            .unwrap_or(0)
            > 0
    );
    assert!(
        profile["cpu"]["throttling"]["throttled_sec"]
            .as_f64()
            .unwrap_or(0.0)
            > 0.0
    );
    assert_eq!(
        profile["environment_identity"]["cgroup_v2"]["private_namespace_root"],
        true
    );
    assert_eq!(
        profile["environment_identity"]["cgroup_v2"]["membership_path"],
        "/"
    );
    for key in [
        "POD_NAME",
        "POD_NAMESPACE",
        "POD_UID",
        "NODE_NAME",
        "CONTAINER_NAME",
    ] {
        assert!(
            profile["environment_identity"]["kubernetes"][key].is_string(),
            "missing Kubernetes identity {key}"
        );
    }

    let before = cpu_stat(&before_bytes);
    let after = cpu_stat(&after_bytes);
    let raw_delta = after["usage_usec"].as_u64().unwrap() - before["usage_usec"].as_u64().unwrap();
    assert_eq!(
        profile["cpu"]["raw_cpu_stat_delta"]["usage_usec"],
        raw_delta
    );
    assert!(raw_delta > 0);
    let reported_usage = profile["cpu"]["usage_sec"].as_f64().unwrap();
    assert!((reported_usage - raw_delta as f64 / 1_000_000.0).abs() <= 0.000_000_5);

    let process_status = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/proc/1/status",
    );
    let mountinfo = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current",
        "/proc/1/mountinfo",
    );
    fs::write(artifacts.join("pod-init-status.txt"), &process_status).unwrap();
    fs::write(artifacts.join("pod-init-mountinfo.txt"), &mountinfo).unwrap();
    let status_text = String::from_utf8_lossy(&process_status);
    assert!(status_text
        .lines()
        .any(|line| line.starts_with("Uid:") && line.split_whitespace().nth(1) == Some("65534")));
    assert!(status_text.lines().any(|line| line.starts_with("CapEff:")
        && line.split_whitespace().nth(1) == Some("0000000000000000")));
    assert!(
        status_text
            .lines()
            .any(|line| line.starts_with("NoNewPrivs:")
                && line.split_whitespace().nth(1) == Some("1"))
    );
    assert!(status_text
        .lines()
        .any(|line| line.starts_with("Seccomp:") && line.split_whitespace().nth(1) == Some("2")));
    let mount_text = String::from_utf8_lossy(&mountinfo);
    assert!(mount_text.lines().any(|line| {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        fields.get(4) == Some(&"/")
            && fields
                .get(5)
                .is_some_and(|options| options.split(',').any(|value| value == "ro"))
    }));
    assert!(mount_text.lines().any(|line| {
        let fields = line.split_whitespace().collect::<Vec<_>>();
        fields.get(4) == Some(&"/sys/fs/cgroup")
            && fields
                .get(5)
                .is_some_and(|options| options.split(',').any(|value| value == "ro"))
    }));

    let main_delete = delete_pod(&kubectl_bin, &kubeconfig, "tasktrace-rust-current");
    write_output(&artifacts, "kubectl-delete-main-pod", &main_delete);
    assert_success(&main_delete, "main profiling pod deletion");

    // The deployment assertion is a validity gate, not a documentation hint.
    // Prove that the same restricted pod returns an invalid report and exit 1
    // when exclusivity is not explicitly asserted.
    let unasserted_marker = "TASKTRACE_RUST_CURRENT_UNASSERTED_READY";
    let unasserted_script = format!(
        "set +e\n/usr/local/bin/tasktrace-rust profile atif /input/trajectory.json --runtime current --allocated-cpus 1 --pacing unpaced --workdir /workspace --shell /bin/bash --strace-mode none --timeout-sec 30 --patch-root /workspace --output-dir /results/profile > /results/cli.stdout.txt 2> /results/cli.stderr.txt\nstatus=$?\nset -e\ntest \"$status\" -eq 1\nprintf '%s\\n' {unasserted_marker}\nsleep 600\n"
    );
    let unasserted_pod = derived_pod(&pod, "tasktrace-rust-current-unasserted", &unasserted_script);
    let unasserted_path = artifacts.join("pod-unasserted-manifest.json");
    fs::write(
        &unasserted_path,
        serde_json::to_vec_pretty(&unasserted_pod).unwrap(),
    )
    .unwrap();
    let unasserted_apply = output(
        &mut kubectl(
            &kubectl_bin,
            &kubeconfig,
            ["apply", "-f", unasserted_path.to_str().unwrap()],
        ),
        "apply unasserted validity-guard pod",
    );
    write_output(&artifacts, "kubectl-apply-unasserted", &unasserted_apply);
    assert_success(&unasserted_apply, "unasserted validity-guard pod creation");
    let unasserted_logs = wait_for_pod_marker(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current-unasserted",
        unasserted_marker,
    );
    write_output(&artifacts, "pod-unasserted-logs", &unasserted_logs);
    let unasserted_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current-unasserted",
        "/results/profile/pipeline_report.json",
    );
    fs::write(
        artifacts.join("pipeline_report.unasserted.json"),
        &unasserted_bytes,
    )
    .unwrap();
    let unasserted_report = parse_json(&unasserted_bytes, "unasserted pipeline report");
    assert_eq!(unasserted_report["status"], "invalid");
    assert_eq!(
        unasserted_report["measurement_validity"]["exclusive_single_task_container_asserted"],
        false
    );
    assert_eq!(
        unasserted_report["measurement_validity"]["valid_for_average_cpu_reporting"],
        false
    );
    let unasserted_delete =
        delete_pod(&kubectl_bin, &kubeconfig, "tasktrace-rust-current-unasserted");
    write_output(
        &artifacts,
        "kubectl-delete-unasserted-pod",
        &unasserted_delete,
    );
    assert_success(&unasserted_delete, "unasserted pod deletion");

    // Prove the cgroup process-scope gate with a live sibling that is neither
    // the controller nor one of its ancestors. The profiler may diagnose it,
    // but must not signal or hide it and must reject the sample.
    let contaminated_marker = "TASKTRACE_RUST_CURRENT_CONTAMINATED_READY";
    let contaminated_script = format!(
        "set -u\n/bin/sleep 60 &\nsibling=$!\nsleep 0.1\nset +e\n/usr/local/bin/tasktrace-rust profile atif /input/trajectory.json --runtime current --current-container-exclusive --allocated-cpus 1 --pacing unpaced --workdir /workspace --shell /bin/bash --strace-mode none --timeout-sec 30 --patch-root /workspace --output-dir /results/profile > /results/cli.stdout.txt 2> /results/cli.stderr.txt\nstatus=$?\nkill \"$sibling\" 2>/dev/null || true\nwait \"$sibling\" 2>/dev/null || true\nset -e\ntest \"$status\" -eq 1\nprintf '%s\\n' {contaminated_marker}\nsleep 600\n"
    );
    let contaminated_pod = derived_pod(
        &pod,
        "tasktrace-rust-current-contaminated",
        &contaminated_script,
    );
    let contaminated_path = artifacts.join("pod-contaminated-manifest.json");
    fs::write(
        &contaminated_path,
        serde_json::to_vec_pretty(&contaminated_pod).unwrap(),
    )
    .unwrap();
    let contaminated_apply = output(
        &mut kubectl(
            &kubectl_bin,
            &kubeconfig,
            ["apply", "-f", contaminated_path.to_str().unwrap()],
        ),
        "apply contaminated validity-guard pod",
    );
    write_output(
        &artifacts,
        "kubectl-apply-contaminated",
        &contaminated_apply,
    );
    assert_success(
        &contaminated_apply,
        "contaminated validity-guard pod creation",
    );
    let contaminated_logs = wait_for_pod_marker(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current-contaminated",
        contaminated_marker,
    );
    write_output(&artifacts, "pod-contaminated-logs", &contaminated_logs);
    let contaminated_bytes = kubectl_cat(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current-contaminated",
        "/results/profile/pipeline_report.json",
    );
    fs::write(
        artifacts.join("pipeline_report.contaminated.json"),
        &contaminated_bytes,
    )
    .unwrap();
    let contaminated_report = parse_json(&contaminated_bytes, "contaminated pipeline report");
    assert_eq!(contaminated_report["status"], "invalid");
    assert_eq!(
        contaminated_report["measurement_validity"]["valid_for_average_cpu_reporting"],
        false
    );
    assert!(
        contaminated_report["measurement_validity"]["potential_unrelated_processes_before"]
            .as_array()
            .is_some_and(|processes| !processes.is_empty()),
        "live sibling was not retained in the invalidity evidence"
    );
    let contaminated_delete = delete_pod(
        &kubectl_bin,
        &kubeconfig,
        "tasktrace-rust-current-contaminated",
    );
    write_output(
        &artifacts,
        "kubectl-delete-contaminated-pod",
        &contaminated_delete,
    );
    assert_success(&contaminated_delete, "contaminated pod deletion");

    let artifact_manifest = json!({
        "schema_version": "tasktrace-rust-current-kind-e2e.v1",
        "cluster": cluster,
        "kind_node_image": NODE_IMAGE,
        "task_image_reference": image,
        "task_image_id": expected_image_id,
        "pod_observed_image_id": observed_image_id,
        "controller_sha256": file_sha256(&controller),
        "pod_manifest": "pod-manifest.json",
        "reports": {
            "pipeline_report.json": format!("{:x}", Sha256::digest(&pipeline_bytes)),
            "profile_report.json": format!("{:x}", Sha256::digest(&profile_bytes)),
            "replay_report.json": format!("{:x}", Sha256::digest(&replay_bytes)),
            "cpu.stat.before": format!("{:x}", Sha256::digest(&before_bytes)),
            "cpu.stat.after": format!("{:x}", Sha256::digest(&after_bytes)),
            "pipeline_report.unasserted.json": format!("{:x}", Sha256::digest(&unasserted_bytes)),
            "pipeline_report.contaminated.json": format!("{:x}", Sha256::digest(&contaminated_bytes)),
        },
        "measurement": {
            "average_logical_cores_used": average,
            "usage_sec": reported_usage,
            "throttled_periods": profile["cpu"]["throttling"]["throttled_periods"],
            "throttled_sec": profile["cpu"]["throttling"]["throttled_sec"],
            "valid": true,
        },
        "validity_guards": {
            "missing_exclusive_assertion_rejected": true,
            "live_sibling_process_rejected": true,
        },
    });
    fs::write(
        artifacts.join("validation-summary.json"),
        serde_json::to_vec_pretty(&artifact_manifest).unwrap(),
    )
    .unwrap();

    guard.delete();
    let clusters = output(
        &mut command_with_args(&kind, ["get", "clusters"]),
        "verify Kind teardown",
    );
    write_output(&artifacts, "kind-clusters-after", &clusters);
    assert_success(&clusters, "Kind teardown verification");
    assert!(
        !String::from_utf8_lossy(&clusters.stdout)
            .lines()
            .any(|name| name == cluster),
        "Kind cluster survived teardown"
    );
}
