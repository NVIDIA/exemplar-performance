// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Real-Docker gate for native profiling of a caller-owned running container.

use serde_json::{json, Value};
use std::env;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};
use std::time::{SystemTime, UNIX_EPOCH};

const IMAGE: &str = "sha256:d28f698d6895b4863769ef7ec47f013529c4d83c30e0b6b74a85a239052ada95";
const CPUSET: &str = "16-19";

fn binary() -> PathBuf {
    PathBuf::from(env!("CARGO_BIN_EXE_tasktrace"))
}

fn docker(arguments: &[&str]) -> Output {
    Command::new("docker")
        .args(arguments)
        .output()
        .unwrap_or_else(|error| panic!("unable to run docker {arguments:?}: {error}"))
}

fn assert_success(output: &Output, description: &str) {
    assert!(
        output.status.success(),
        "{description} failed\nstdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
}

fn inspect(reference: &str) -> Value {
    let output = docker(&["container", "inspect", reference]);
    assert_success(&output, "inspect existing-container fixture");
    serde_json::from_slice::<Value>(&output.stdout).expect("Docker inspection JSON")[0].clone()
}

fn stable_projection(value: &Value) -> Value {
    let networks = value["NetworkSettings"]["Networks"]
        .as_object()
        .expect("Docker network attachments")
        .iter()
        .map(|(name, network)| {
            (
                name.clone(),
                json!({
                    "NetworkID": network["NetworkID"],
                    "EndpointID": network["EndpointID"],
                    "IPAddress": network["IPAddress"],
                    "GlobalIPv6Address": network["GlobalIPv6Address"],
                }),
            )
        })
        .collect::<serde_json::Map<_, _>>();
    json!({
        "Id": value["Id"],
        "Name": value["Name"],
        "Image": value["Image"],
        "Created": value["Created"],
        "RestartCount": value["RestartCount"],
        "State": {
            "Running": value["State"]["Running"],
            "Paused": value["State"]["Paused"],
            "Restarting": value["State"]["Restarting"],
            "Pid": value["State"]["Pid"],
            "StartedAt": value["State"]["StartedAt"],
        },
        "HostConfig": {
            "CpusetCpus": value["HostConfig"]["CpusetCpus"],
            "NanoCpus": value["HostConfig"]["NanoCpus"],
            "CpuQuota": value["HostConfig"]["CpuQuota"],
            "CpuPeriod": value["HostConfig"]["CpuPeriod"],
            "CpuShares": value["HostConfig"]["CpuShares"],
            "CgroupParent": value["HostConfig"]["CgroupParent"],
            "NetworkMode": value["HostConfig"]["NetworkMode"],
        },
        "Networks": networks,
    })
}

fn cpu_stat(path: &Path) -> std::collections::BTreeMap<String, u64> {
    fs::read_to_string(path)
        .expect("read retained cpu.stat")
        .lines()
        .map(|line| {
            let mut fields = line.split_whitespace();
            let name = fields.next().expect("cpu.stat field").to_owned();
            let value = fields
                .next()
                .expect("cpu.stat value")
                .parse::<u64>()
                .expect("numeric cpu.stat value");
            assert!(fields.next().is_none(), "malformed cpu.stat line: {line}");
            (name, value)
        })
        .collect()
}

struct Fixture {
    name: String,
    removed: bool,
}

impl Fixture {
    fn remove(&mut self) {
        if self.removed {
            return;
        }
        let output = docker(&["container", "rm", "--force", &self.name]);
        assert_success(&output, "remove only the owned existing-container fixture");
        self.removed = true;
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        if !self.removed {
            let _ = docker(&["container", "rm", "--force", &self.name]);
        }
    }
}

#[test]
#[ignore = "requires a local cgroup-v2 Docker daemon and reserved CPUs 16-19"]
fn public_existing_container_profile_is_native_stable_and_non_owning() {
    let image = docker(&["image", "inspect", IMAGE]);
    assert_success(&image, "inspect pre-pulled immutable fixture image");

    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let name = format!("tasktrace-existing-e2e-{}-{nonce}", std::process::id());
    let artifact_root = PathBuf::from(format!(
        "/tmp/tasktrace-existing-container-e2e-{}-{nonce}",
        std::process::id()
    ));
    fs::create_dir(&artifact_root).expect("create retained /tmp proof directory");
    let output_dir = artifact_root.join("profile");

    let started = docker(&[
        "run",
        "--detach",
        "--name",
        &name,
        "--label",
        "io.tasktrace.test=existing-container-e2e",
        "--cpuset-cpus",
        CPUSET,
        "--network",
        "none",
        IMAGE,
        "/bin/sh",
        "-c",
        "while :; do sleep 3600 & wait $!; done",
    ]);
    assert_success(&started, "start caller-owned existing-container fixture");
    let mut fixture = Fixture {
        name: name.clone(),
        removed: false,
    };

    let before = inspect(&name);
    let before_projection = stable_projection(&before);
    let exact_id = before["Id"].as_str().expect("full container ID").to_owned();
    assert_eq!(before["Image"], IMAGE);
    assert_eq!(before["HostConfig"]["CpusetCpus"], CPUSET);
    assert_eq!(before["HostConfig"]["NetworkMode"], "none");
    assert_eq!(before["State"]["Running"], true);

    let profiled = Command::new(binary())
        .args(["profile", "container", "--container", &name])
        .args(["--allocated-cpus", "4", "--strace-mode", "none"])
        .args([
            "--command",
            "python3 -c 'x=sum((i*i)%97 for i in range(12000000)); print(\"tasktrace-existing-ok\", x)'",
        ])
        .arg("--output-dir")
        .arg(&output_dir)
        .env(
            "TASKTRACE_LEGACY_PYTHON",
            "/tasktrace-test/python-fallback-must-not-run",
        )
        .output()
        .expect("run native public existing-container profile");
    assert_success(&profiled, "profile caller-owned existing container");
    assert!(
        String::from_utf8_lossy(&profiled.stdout).contains("Report:"),
        "public output omitted native report path: {}",
        String::from_utf8_lossy(&profiled.stdout)
    );
    assert!(
        !String::from_utf8_lossy(&profiled.stderr).contains("legacy"),
        "Python fallback was consulted: {}",
        String::from_utf8_lossy(&profiled.stderr)
    );

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("profile_report.json")).expect("read native profile report"),
    )
    .expect("parse native profile report");
    assert_eq!(report["schema_version"], "container-command-profile.v1");
    assert_eq!(report["runtime"], "docker");
    assert_eq!(report["scope"], "existing-container");
    assert_eq!(report["container"], name);
    assert_eq!(report["image_identity"]["id"], IMAGE);
    assert_eq!(report["profiling_valid"], true);
    assert_eq!(report["command_exit_code"], 0);
    assert_eq!(
        report["existing_container"]["resolved_container"]["container_id"],
        exact_id
    );
    assert_eq!(
        report["existing_container"]["observed_allocation"]["docker"]
            ["docker_host_config_cpuset_cpus"],
        CPUSET
    );
    assert_eq!(
        report["existing_container"]["observed_allocation"]["kernel"]
            ["cgroup_cpuset_cpus_effective"],
        CPUSET
    );
    assert_eq!(
        report["existing_container"]["network_inheritance"]["docker_host_config_network_mode"],
        "none"
    );
    assert_eq!(
        report["existing_container"]["network_inheritance"]["topology_mutated"],
        false
    );
    assert_eq!(report["existing_container"]["stability"]["valid"], true);
    assert_eq!(
        report["existing_container"]["post_artifact_stability"]["valid"],
        true
    );
    assert_eq!(
        report["existing_container"]["survivor_handling"]["remaining_pids_at_finish_gate"],
        json!([])
    );
    assert_eq!(
        report["existing_container"]["lifecycle_ownership"],
        json!({
            "owner": "caller",
            "container_started": false,
            "container_updated": false,
            "container_restarted": false,
            "container_stopped": false,
            "container_removed": false,
            "network_topology_mutated": false,
        })
    );
    assert!(report["cpu"]["wall_sec"].as_f64().unwrap() > 0.0);
    assert!(report["cpu"]["usage_sec"].as_f64().unwrap() > 0.0);
    assert!(
        report["cpu"]["average_logical_cores_used"]
            .as_f64()
            .unwrap()
            > 0.0
    );
    assert_eq!(report["cpu"]["allocated_logical_cpus"], 4.0);
    assert_eq!(report["cpu"]["scope_includes_descendants"], true);

    let before_cpu = cpu_stat(&output_dir.join("cpu.stat.before"));
    let after_cpu = cpu_stat(&output_dir.join("cpu.stat.after"));
    let usage_delta = after_cpu["usage_usec"] - before_cpu["usage_usec"];
    let user_delta = after_cpu["user_usec"] - before_cpu["user_usec"];
    let system_delta = after_cpu["system_usec"] - before_cpu["system_usec"];
    assert!(usage_delta > 0);
    assert_eq!(
        report["cpu"]["usage_sec"].as_f64().unwrap(),
        usage_delta as f64 / 1_000_000.0
    );
    assert_eq!(
        report["cpu"]["user_sec"].as_f64().unwrap(),
        user_delta as f64 / 1_000_000.0
    );
    assert_eq!(
        report["cpu"]["system_sec"].as_f64().unwrap(),
        system_delta as f64 / 1_000_000.0
    );
    assert!(fs::read_to_string(output_dir.join("command.stdout.txt"))
        .unwrap()
        .contains("tasktrace-existing-ok"));
    let start_ns: u64 = fs::read_to_string(output_dir.join("start_ns"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    let end_ns: u64 = fs::read_to_string(output_dir.join("end_ns"))
        .unwrap()
        .trim()
        .parse()
        .unwrap();
    assert!(end_ns > start_ns);

    let after = inspect(&exact_id);
    assert_eq!(stable_projection(&after), before_projection);
    assert_eq!(after["State"]["Running"], true);

    fixture.remove();
    let absent = docker(&["container", "inspect", &exact_id]);
    assert!(
        !absent.status.success(),
        "owned fixture remained after explicit test cleanup"
    );
    eprintln!(
        "retained existing-container E2E proof: {}",
        artifact_root.display()
    );
}
