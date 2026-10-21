// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Opt-in local-Docker proof for native allowlist enforcement.
//!
//! The test builds an isolated, globally numbered fixture network so the
//! production private-address guard remains enabled. It never contacts the
//! Internet. Run with the same immutable native Bash image used by the other
//! Docker gates and runner-owned CPUs.

use serde_json::Value;
use sha2::{Digest, Sha256};
use std::env;
use std::fs;
use std::path::Path;
use std::process::{Command, Output};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tempfile::TempDir;

fn binary() -> &'static str {
    env!("CARGO_BIN_EXE_tasktrace")
}

fn output(command: &mut Command, description: &str) -> Output {
    command
        .output()
        .unwrap_or_else(|error| panic!("unable to {description}: {error}"))
}

fn assert_success(result: &Output, description: &str) {
    assert!(
        result.status.success(),
        "{description} failed with {:?}\nstdout:\n{}\nstderr:\n{}",
        result.status.code(),
        String::from_utf8_lossy(&result.stdout),
        String::from_utf8_lossy(&result.stderr),
    );
}

fn image_id(image: &str) -> String {
    let inspected = output(
        Command::new("docker").args(["image", "inspect", "--format", "{{.Id}}", image]),
        "inspect fixture image",
    );
    assert_success(&inspected, "fixture image inspection");
    String::from_utf8(inspected.stdout)
        .expect("image ID is UTF-8")
        .trim()
        .to_owned()
}

fn file_sha256(path: &Path) -> String {
    format!(
        "{:x}",
        Sha256::digest(fs::read(path).expect("read SHA-256 input"))
    )
}

struct Fixture {
    network: String,
    allowed: String,
    blocked: String,
    allowed_ip: String,
}

impl Fixture {
    fn new(image: &str) -> Self {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let token = format!("{}-{:x}", std::process::id(), nonce & 0xfffffff);
        let fixture = Self {
            network: format!("tasktrace-rust-policy-fixture-{token}"),
            allowed: format!("tasktrace-rust-policy-allowed-{token}"),
            blocked: format!("tasktrace-rust-policy-blocked-{token}"),
            allowed_ip: "8.223.17.10".to_owned(),
        };
        let created = output(
            Command::new("docker").args([
                "network",
                "create",
                "--driver",
                "bridge",
                "--subnet",
                "8.223.17.0/24",
                "--gateway",
                "8.223.17.1",
                &fixture.network,
            ]),
            "create local policy fixture network",
        );
        assert_success(&created, "policy fixture network creation");
        for (name, address, alias) in [
            (
                &fixture.allowed,
                fixture.allowed_ip.as_str(),
                "allowed.test",
            ),
            (&fixture.blocked, "8.223.17.11", "blocked.test"),
        ] {
            let started = output(
                Command::new("docker").args([
                    "run",
                    "--detach",
                    "--name",
                    name,
                    "--network",
                    &fixture.network,
                    "--ip",
                    address,
                    "--network-alias",
                    alias,
                    image,
                    "python",
                    "-m",
                    "http.server",
                    "80",
                    "--bind",
                    "0.0.0.0",
                ]),
                "start local HTTP fixture",
            );
            assert_success(&started, "local HTTP fixture startup");
        }
        for name in [&fixture.allowed, &fixture.blocked] {
            let mut ready = false;
            for _ in 0..100 {
                let checked = output(
                    Command::new("docker").args([
                        "exec",
                        name,
                        "python",
                        "-c",
                        "import socket; s=socket.create_connection(('127.0.0.1',80),.2); s.close()",
                    ]),
                    "probe local HTTP fixture",
                );
                if checked.status.success() {
                    ready = true;
                    break;
                }
                thread::sleep(Duration::from_millis(50));
            }
            assert!(ready, "local HTTP fixture {name} did not become ready");
        }
        fixture
    }
}

impl Drop for Fixture {
    fn drop(&mut self) {
        for name in [&self.allowed, &self.blocked] {
            let _ = Command::new("docker")
                .args(["container", "rm", "--force", name])
                .output();
        }
        let _ = Command::new("docker")
            .args(["network", "rm", &self.network])
            .output();
    }
}

#[test]
#[ignore = "requires local Docker, cgroup v2, an immutable Python image, and runner-owned CPUs"]
fn native_allowlist_allows_exact_http_and_removes_every_resource() {
    let image = env::var("TASKTRACE_RUST_DOCKER_TEST_IMAGE")
        .expect("set TASKTRACE_RUST_DOCKER_TEST_IMAGE to a preloaded immutable image");
    let cpuset = env::var("TASKTRACE_RUST_DOCKER_CPUSET")
        .expect("set TASKTRACE_RUST_DOCKER_CPUSET to runner-owned logical CPUs");
    let expected_image_id = image_id(&image);
    let fixture = Fixture::new(&image);
    let temporary = TempDir::new().expect("create allowlist E2E root");
    let output_dir = temporary.path().join("profile");
    let probe = temporary.path().join("probe.py");
    fs::write(
        &probe,
        format!(
            r#"import socket
import urllib.error
import urllib.request

secret = "must-not-appear-in-policy-evidence"
response = urllib.request.urlopen("http://allowed.test/?token=" + secret, timeout=5)
assert response.status == 200
response.read()

def denied(url, status):
    try:
        urllib.request.urlopen(url, timeout=5)
    except urllib.error.HTTPError as error:
        assert error.code == status, (url, error.code)
    else:
        raise AssertionError("unexpectedly allowed " + url)

denied("http://blocked.test/", 403)
denied("http://sub.allowed.test/", 403)
denied("http://allowed.test:8080/", 403)
denied("http://{allowed_ip}/", 403)

def raw(request, expected):
    stream = socket.create_connection(("tasktrace-egress-proxy", 3128), timeout=5)
    stream.sendall(request)
    response = stream.recv(4096)
    stream.close()
    assert response.startswith(expected), response

raw(b"GET http://allowed.test/ HTTP/1.1\r\nHost: blocked.test\r\n\r\n", b"HTTP/1.1 400")
raw(b"CONNECT {allowed_ip}:443 HTTP/1.1\r\nHost: {allowed_ip}:443\r\n\r\n", b"HTTP/1.1 403")

try:
    socket.create_connection(("{allowed_ip}", 80), timeout=.5)
except OSError:
    pass
else:
    raise AssertionError("direct TCP egress bypassed the internal policy network")

print("native-allowlist-probe-ok")
"#,
            allowed_ip = fixture.allowed_ip,
        ),
    )
    .expect("write allowlist probe");
    let mount = format!(
        "type=bind,src={},dst=/tmp/tasktrace-allowlist-probe.py,readonly",
        probe.canonicalize().unwrap().display()
    );
    let profiled = output(
        Command::new("timeout")
            .args(["--signal=TERM", "--kill-after=10s", "90s", binary()])
            .args(["profile", "container", "--image", &image])
            .args(["--cpuset-cpus", &cpuset])
            .args(["--workdir", "/tmp"])
            .args(["--network-policy", "allowlist"])
            .args(["--allowed-host", "Allowed.TEST"])
            .args(["--network-policy-proxy-image", &image])
            .args(["--network-policy-external-network", &fixture.network])
            .args(["--profile-set", "cpu-average"])
            .args(["--mount", &mount])
            .args(["--command", "python /tmp/tasktrace-allowlist-probe.py"])
            .arg("--output-dir")
            .arg(&output_dir)
            .env(
                "TASKTRACE_LEGACY_PYTHON",
                temporary.path().join("python-fallback-must-not-run"),
            ),
        "run native allowlist profile",
    );
    assert_success(&profiled, "native allowlist profile");

    let enforcement: Value = serde_json::from_slice(
        &fs::read(output_dir.join("network_enforcement.json"))
            .expect("read network enforcement report"),
    )
    .expect("parse network enforcement report");
    assert_eq!(enforcement["state"], "closed");
    assert_eq!(enforcement["backend"], "docker-forward-proxy-v1");
    assert_eq!(
        enforcement["policy"]["allowed_hosts"],
        serde_json::json!(["allowed.test"])
    );
    assert_eq!(
        enforcement["proxy"]["image_identity"]["id"],
        expected_image_id
    );
    assert_eq!(
        enforcement["proxy"]["binary_sha256"],
        file_sha256(Path::new(binary()))
    );
    for key in [
        "external_network_inspected",
        "internal_network_inspected",
        "proxy_container_inspected",
        "proxy_ready",
        "proxy_bound_to_internal_address",
        "state_disjoint_from_workload_mounts",
        "workload_container_inspected",
        "workload_isolated_on_internal_network",
        "proxy_environment_validated",
        "decision_log_validated",
        "proxy_healthy_at_cleanup",
    ] {
        assert_eq!(enforcement["validation"][key], true, "validation {key}");
    }
    for key in [
        "attempted",
        "proxy_stopped",
        "proxy_removed",
        "proxy_absence_verified",
        "network_removed",
        "network_absence_verified",
        "decision_log_archived",
        "host_state_removed",
        "host_state_absence_verified",
    ] {
        assert_eq!(enforcement["cleanup"][key], true, "cleanup {key}");
    }

    let decisions = fs::read_to_string(output_dir.join("network_proxy_decisions.jsonl"))
        .expect("read archived decisions");
    assert!(!decisions.contains("must-not-appear-in-policy-evidence"));
    let records = decisions
        .lines()
        .map(|line| serde_json::from_str::<Value>(line).expect("parse decision record"))
        .collect::<Vec<_>>();
    assert!(records.iter().any(|record| {
        record["event"] == "request_decision" && record["decision"] == "allowed"
    }));
    assert!(records
        .iter()
        .any(|record| { record["event"] == "request_decision" && record["decision"] == "denied" }));

    let report: Value = serde_json::from_slice(
        &fs::read(output_dir.join("profile_report.json")).expect("read profile report"),
    )
    .expect("parse profile report");
    assert_eq!(report["network_policy"]["effective_mode"], "allowlist");
    assert_eq!(report["network_policy"]["enforcement"]["state"], "closed");
    let network = enforcement["ephemeral_internal_network"].as_str().unwrap();
    let proxy = enforcement["proxy"]["container_name"].as_str().unwrap();
    assert!(!output(
        Command::new("docker").args(["container", "inspect", proxy]),
        "verify proxy absence",
    )
    .status
    .success());
    assert!(!output(
        Command::new("docker").args(["network", "inspect", network]),
        "verify internal network absence",
    )
    .status
    .success());
    let token = network.strip_prefix("tasktrace-policy-").unwrap();
    assert!(!env::temp_dir()
        .join(format!(".tasktrace-network-policy-state-{token}"))
        .exists());
}
