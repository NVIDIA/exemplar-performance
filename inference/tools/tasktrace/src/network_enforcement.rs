// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed Docker enforcement for semantic replay network policies.
//!
//! A nonempty allowlist receives a fresh internal bridge and a hardened proxy
//! sidecar. The measured workload joins only that bridge; the proxy alone also
//! joins the configured external network. The sidecar executes a content-bound
//! copy of the current native binary in its hidden proxy mode.

use crate::network_policy::{NetworkMode, NetworkPolicy};
use crate::network_proxy::{globally_routable, PROXY_SCHEMA_VERSION};
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fmt::Write as _;
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Read, Write};
use std::os::unix::fs::{OpenOptionsExt, PermissionsExt};
use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};
use std::thread;
use std::time::{Duration, Instant};

pub const NETWORK_ENFORCEMENT_SCHEMA_VERSION: &str = "tasktrace-docker-network-enforcement.v1";
pub const NETWORK_ENFORCEMENT_COMPARABILITY_SCHEMA_VERSION: &str =
    "tasktrace-docker-network-enforcement-comparability.v1";
pub const DEFAULT_PROXY_IMAGE: &str = "python:3.12-slim";
pub const DEFAULT_EXTERNAL_NETWORK: &str = "bridge";
pub const PROXY_ALIAS: &str = "tasktrace-egress-proxy";
pub const DEFAULT_PROXY_PORT: u16 = 3128;
const INHIBIT_IPV4_OPTION: &str = "com.docker.network.bridge.inhibit_ipv4";
const MAX_DECISION_LOG_BYTES: u64 = 64 * 1024 * 1024;
const MAX_DECISION_LOG_RECORDS: u64 = 1_000_000;
const PINNED_PROXY_PATH: &str = "/opt/tasktrace/tasktrace";
const PROXY_STATE_PATH: &str = "/var/run/tasktrace-policy";
const POLICY_COMPONENT_LABEL: &str = "io.tasktrace.component";
const POLICY_DIGEST_LABEL: &str = "io.tasktrace.policy-digest";
const OUTPUT_DIGEST_LABEL: &str = "io.tasktrace.output-digest";

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct NetworkEnforcementError(String);

impl NetworkEnforcementError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for NetworkEnforcementError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for NetworkEnforcementError {}

impl From<io::Error> for NetworkEnforcementError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for NetworkEnforcementError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DockerNetworkBinding {
    pub docker_network: Option<String>,
    pub proxy_environment: BTreeMap<String, String>,
}

impl DockerNetworkBinding {
    fn public() -> Self {
        Self {
            docker_network: None,
            proxy_environment: BTreeMap::new(),
        }
    }

    fn no_network() -> Self {
        Self {
            docker_network: Some("none".to_owned()),
            proxy_environment: BTreeMap::new(),
        }
    }

    pub fn applied_environment(&self, configured: &[String]) -> Vec<String> {
        if self.proxy_environment.is_empty() {
            return configured.to_vec();
        }
        const POLICY_PROXY_KEYS: &[&str] = &[
            "ALL_PROXY",
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "all_proxy",
            "http_proxy",
            "https_proxy",
            "no_proxy",
        ];
        let mut environment = configured
            .iter()
            .filter(|assignment| {
                let name = assignment
                    .split_once('=')
                    .map_or(assignment.as_str(), |(name, _)| name);
                !POLICY_PROXY_KEYS.contains(&name)
            })
            .cloned()
            .collect::<Vec<_>>();
        environment.extend(
            self.proxy_environment
                .iter()
                .map(|(name, value)| format!("{name}={value}")),
        );
        environment
    }
}

#[derive(Clone, Debug)]
pub struct DockerNetworkPolicyConfig {
    pub policy: NetworkPolicy,
    pub output_dir: PathBuf,
    pub docker: String,
    pub proxy_image: String,
    pub external_network: String,
    pub workload_mount_sources: Vec<PathBuf>,
    pub proxy_executable: PathBuf,
    pub state_root: PathBuf,
    pub proxy_port: u16,
    pub allowed_ports: Vec<u16>,
    pub readiness_timeout: Duration,
    token: Option<String>,
}

impl DockerNetworkPolicyConfig {
    pub fn new(
        policy: NetworkPolicy,
        output_dir: impl Into<PathBuf>,
    ) -> Result<Self, NetworkEnforcementError> {
        Ok(Self {
            policy,
            output_dir: output_dir.into(),
            docker: "docker".to_owned(),
            proxy_image: DEFAULT_PROXY_IMAGE.to_owned(),
            external_network: DEFAULT_EXTERNAL_NETWORK.to_owned(),
            workload_mount_sources: Vec::new(),
            proxy_executable: std::env::current_exe()?.canonicalize()?,
            state_root: std::env::temp_dir(),
            proxy_port: DEFAULT_PROXY_PORT,
            allowed_ports: vec![80, 443],
            readiness_timeout: Duration::from_secs(10),
            token: None,
        })
    }

    #[cfg(test)]
    fn with_token(mut self, token: &str) -> Self {
        self.token = Some(token.to_owned());
        self
    }
}

pub trait DockerCommandRunner: Send {
    fn run(&mut self, program: &OsStr, arguments: &[OsString]) -> io::Result<Output>;
}

struct SystemDockerRunner;

impl DockerCommandRunner for SystemDockerRunner {
    fn run(&mut self, program: &OsStr, arguments: &[OsString]) -> io::Result<Output> {
        Command::new(program)
            .args(arguments)
            .stdin(Stdio::null())
            .output()
    }
}

pub struct DockerNetworkPolicySession {
    config: DockerNetworkPolicyConfig,
    runner: Box<dyn DockerCommandRunner>,
    token: String,
    network_name: String,
    proxy_name: String,
    state_dir: PathBuf,
    ready_path: PathBuf,
    decision_log_path: PathBuf,
    pinned_proxy_path: PathBuf,
    archived_decision_log_path: PathBuf,
    report_path: PathBuf,
    output_scope_sha256: String,
    state: &'static str,
    binding: Option<DockerNetworkBinding>,
    network_created: bool,
    proxy_created: bool,
    proxy_ever_started: bool,
    proxy_container_id: Option<String>,
    proxy_binary_sha256: Option<String>,
    proxy_image_identity: Option<Value>,
    external_network_identity: Option<Value>,
    proxy_internal_ip: Option<String>,
    decision_log_evidence: Option<Value>,
    validation: BTreeMap<String, Option<bool>>,
    cleanup: BTreeMap<String, bool>,
    cleanup_errors: Vec<String>,
    closed: bool,
}

impl DockerNetworkPolicySession {
    pub fn new(config: DockerNetworkPolicyConfig) -> Result<Self, NetworkEnforcementError> {
        Self::with_runner(config, Box::new(SystemDockerRunner))
    }

    fn with_runner(
        config: DockerNetworkPolicyConfig,
        runner: Box<dyn DockerCommandRunner>,
    ) -> Result<Self, NetworkEnforcementError> {
        validate_configuration(&config)?;
        let token = match &config.token {
            Some(token) => normalize_token(token)?,
            None => random_token()?,
        };
        let network_name = format!("tasktrace-policy-{token}");
        let proxy_name = format!("tasktrace-policy-proxy-{token}");
        let state_dir = config
            .state_root
            .join(format!(".tasktrace-network-policy-state-{token}"));
        ensure_state_is_disjoint(
            &state_dir,
            &config.output_dir,
            &config.workload_mount_sources,
        )?;
        let output_scope_sha256 = digest_domain(
            b"tasktrace-network-policy-output-scope-v1\0",
            config
                .output_dir
                .canonicalize()?
                .to_string_lossy()
                .as_bytes(),
        );
        let validation = BTreeMap::from([
            ("external_network_inspected".to_owned(), Some(false)),
            ("internal_network_inspected".to_owned(), Some(false)),
            ("proxy_container_inspected".to_owned(), Some(false)),
            ("proxy_ready".to_owned(), Some(false)),
            ("proxy_bound_to_internal_address".to_owned(), Some(false)),
            ("state_disjoint_from_workload_mounts".to_owned(), Some(true)),
            ("workload_container_inspected".to_owned(), Some(false)),
            (
                "workload_isolated_on_internal_network".to_owned(),
                Some(false),
            ),
            ("proxy_environment_validated".to_owned(), Some(false)),
            ("decision_log_validated".to_owned(), None),
            ("proxy_healthy_at_cleanup".to_owned(), None),
        ]);
        let cleanup = BTreeMap::from([
            ("attempted".to_owned(), false),
            ("proxy_stopped".to_owned(), false),
            ("proxy_removed".to_owned(), false),
            ("proxy_absence_verified".to_owned(), false),
            ("network_removed".to_owned(), false),
            ("network_absence_verified".to_owned(), false),
            ("decision_log_archived".to_owned(), false),
            ("host_state_removed".to_owned(), false),
            ("host_state_absence_verified".to_owned(), false),
        ]);
        Ok(Self {
            ready_path: state_dir.join("proxy.ready.json"),
            decision_log_path: state_dir.join("proxy-decisions.jsonl"),
            pinned_proxy_path: state_dir.join("tasktrace"),
            archived_decision_log_path: config.output_dir.join("network_proxy_decisions.jsonl"),
            report_path: config.output_dir.join("network_enforcement.json"),
            config,
            runner,
            token,
            network_name,
            proxy_name,
            state_dir,
            output_scope_sha256,
            state: "configured",
            binding: None,
            network_created: false,
            proxy_created: false,
            proxy_ever_started: false,
            proxy_container_id: None,
            proxy_binary_sha256: None,
            proxy_image_identity: None,
            external_network_identity: None,
            proxy_internal_ip: None,
            decision_log_evidence: None,
            validation,
            cleanup,
            cleanup_errors: Vec::new(),
            closed: false,
        })
    }

    pub fn prepare(&mut self) -> Result<DockerNetworkBinding, NetworkEnforcementError> {
        if self.state != "configured" || self.closed {
            return Err(NetworkEnforcementError::new(
                "network policy session is single-use",
            ));
        }
        let effective = self.config.policy.effective_mode();
        if effective == NetworkMode::Public {
            self.binding = Some(DockerNetworkBinding::public());
            self.state = "active";
            self.write_report(false)?;
            return Ok(self.binding.clone().expect("stored public binding"));
        }
        if effective == NetworkMode::NoNetwork {
            self.binding = Some(DockerNetworkBinding::no_network());
            self.state = "active";
            self.write_report(false)?;
            return Ok(self.binding.clone().expect("stored deny-all binding"));
        }
        let result = self.prepare_allowlist().and_then(|binding| {
            self.binding = Some(binding.clone());
            self.state = "active";
            self.write_report(false)?;
            Ok(binding)
        });
        if let Err(error) = result {
            self.state = "preparation-failed";
            let cleanup = self.cleanup_allowlist(false);
            self.cleanup_errors.extend(cleanup);
            self.closed = true;
            let _ = self.write_report(true);
            let suffix = if self.cleanup_errors.is_empty() {
                String::new()
            } else {
                format!("; cleanup also failed: {}", self.cleanup_errors.join("; "))
            };
            return Err(NetworkEnforcementError::new(format!(
                "unable to enforce Docker allowlist: {error}{suffix}"
            )));
        }
        result
    }

    pub fn binding(&self) -> Result<&DockerNetworkBinding, NetworkEnforcementError> {
        self.binding
            .as_ref()
            .ok_or_else(|| NetworkEnforcementError::new("network policy session is not active"))
    }

    pub fn report_path(&self) -> &Path {
        &self.report_path
    }

    pub fn is_closed(&self) -> bool {
        self.closed
    }

    pub fn report_payload(&self) -> Value {
        let allowlist = self.config.policy.effective_mode() == NetworkMode::Allowlist;
        let proxy_url =
            allowlist.then(|| format!("http://{PROXY_ALIAS}:{}", self.config.proxy_port));
        json!({
            "schema_version": NETWORK_ENFORCEMENT_SCHEMA_VERSION,
            "state": self.state,
            "policy": self.config.policy.to_report_payload(),
            "policy_comparability_sha256": self.config.policy.comparability_digest(),
            "comparability": self.comparability_payload(),
            "comparability_sha256": self.comparability_digest(),
            "output_scope_sha256": self.output_scope_sha256,
            "allowed_ports": if allowlist { self.allowed_ports() } else { Vec::<u16>::new() },
            "backend": if allowlist { "docker-forward-proxy-v1" } else { "docker-native" },
            "docker_network_override": self.binding.as_ref().and_then(|binding| binding.docker_network.clone()),
            "external_network": if allowlist { Some(self.config.external_network.as_str()) } else { None },
            "external_network_identity": if allowlist { self.external_network_identity.clone().unwrap_or(Value::Null) } else { Value::Null },
            "ephemeral_internal_network": if allowlist { Some(self.network_name.as_str()) } else { None },
            "proxy": {
                "container_name": if allowlist { Some(self.proxy_name.as_str()) } else { None },
                "container_id": self.proxy_container_id,
                "stable_alias": if allowlist { Some(PROXY_ALIAS) } else { None },
                "url": proxy_url,
                "configured_image": if allowlist { Some(self.config.proxy_image.as_str()) } else { None },
                "image_identity": self.proxy_image_identity,
                "binary_sha256": self.proxy_binary_sha256,
                "source_sha256": self.proxy_binary_sha256,
                "automatic_image_pull": false,
                "decision_log": if allowlist { Some(self.archived_decision_log_path.to_string_lossy()) } else { None },
                "decision_log_evidence": self.decision_log_evidence,
                "measurement_scope": if allowlist { Some("outside-sandbox-cgroup") } else { None },
                "included_in_sandbox_cgroup_measurement": if allowlist { Some(false) } else { None },
                "bound_internal_ip": if allowlist { self.proxy_internal_ip.clone() } else { None },
            },
            "validation": self.validation,
            "cleanup": {
                "attempted": self.cleanup["attempted"],
                "proxy_stopped": self.cleanup["proxy_stopped"],
                "proxy_removed": self.cleanup["proxy_removed"],
                "proxy_absence_verified": self.cleanup["proxy_absence_verified"],
                "network_removed": self.cleanup["network_removed"],
                "network_absence_verified": self.cleanup["network_absence_verified"],
                "decision_log_archived": self.cleanup["decision_log_archived"],
                "host_state_removed": self.cleanup["host_state_removed"],
                "host_state_absence_verified": self.cleanup["host_state_absence_verified"],
                "errors": self.cleanup_errors,
            },
        })
    }

    pub fn comparability_payload(&self) -> Value {
        let allowlist = self.config.policy.effective_mode() == NetworkMode::Allowlist;
        json!({
            "schema_version": NETWORK_ENFORCEMENT_COMPARABILITY_SCHEMA_VERSION,
            "policy": self.config.policy.to_comparability_payload(),
            "backend": if allowlist { "docker-forward-proxy-v1" } else { "docker-native" },
            "allowed_ports": if allowlist { self.allowed_ports() } else { Vec::<u16>::new() },
            "external_network": if allowlist { Some(self.config.external_network.as_str()) } else { None },
            "external_network_identity": if allowlist { self.external_network_identity.clone().unwrap_or(Value::Null) } else { Value::Null },
            "proxy_source_sha256": if allowlist { self.proxy_binary_sha256.clone() } else { None },
            "proxy_image_identity": if allowlist { self.proxy_image_identity.clone().unwrap_or(Value::Null) } else { Value::Null },
        })
    }

    pub fn comparability_digest(&self) -> String {
        digest_domain(
            b"tasktrace-docker-network-enforcement-comparability-v1\0",
            canonical_json(&self.comparability_payload()).as_bytes(),
        )
    }

    pub fn validate_before_workload_release(
        &mut self,
        workload_container_name: &str,
    ) -> Result<(), NetworkEnforcementError> {
        if self.config.policy.effective_mode() != NetworkMode::Allowlist {
            return Ok(());
        }
        if self.state != "active" {
            return Err(NetworkEnforcementError::new(
                "allowlist enforcement is not active before workload release",
            ));
        }
        let external = self.inspect_external_network()?;
        if self.external_network_identity.as_ref() != Some(&external) {
            return Err(NetworkEnforcementError::new(
                "proxy external network identity changed before workload release",
            ));
        }
        self.inspect_internal_network()?;
        self.inspect_proxy_container()?;
        self.inspect_workload_container(workload_container_name)?;
        self.write_report(false)?;
        Ok(())
    }

    pub fn close(&mut self) -> Result<(), NetworkEnforcementError> {
        if self.closed {
            if self.cleanup_errors.is_empty() {
                return Ok(());
            }
            return Err(NetworkEnforcementError::new(self.cleanup_errors.join("; ")));
        }
        self.closed = true;
        if self.config.policy.effective_mode() != NetworkMode::Allowlist {
            self.state = "closed";
            self.write_report(false)?;
            return Ok(());
        }
        let errors = self.cleanup_allowlist(true);
        self.cleanup_errors.extend(errors);
        self.state = if self.cleanup_errors.is_empty() {
            "closed"
        } else {
            "cleanup-failed"
        };
        let report_result = self.write_report(!self.cleanup_errors.is_empty());
        if let Err(error) = report_result {
            self.cleanup_errors
                .push(format!("write final network enforcement report: {error}"));
            self.state = "cleanup-failed";
        }
        if self.cleanup_errors.is_empty() {
            Ok(())
        } else {
            Err(NetworkEnforcementError::new(format!(
                "Docker network-policy cleanup failed: {}",
                self.cleanup_errors.join("; ")
            )))
        }
    }

    fn allowed_ports(&self) -> Vec<u16> {
        self.config
            .allowed_ports
            .iter()
            .copied()
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect()
    }

    fn prepare_allowlist(&mut self) -> Result<DockerNetworkBinding, NetworkEnforcementError> {
        self.prepare_state()?;
        self.proxy_image_identity = Some(self.inspect_image()?);
        self.external_network_identity = Some(self.inspect_external_network()?);
        let policy_digest = self.config.policy.comparability_digest();
        self.checked(
            "create internal policy network",
            strings(&[
                "network",
                "create",
                "--driver",
                "bridge",
                "--internal",
                "--ipv6=false",
                "--opt",
                &format!("{INHIBIT_IPV4_OPTION}=true"),
                "--label",
                "io.tasktrace.component=network-policy",
                "--label",
                &format!("{POLICY_DIGEST_LABEL}={policy_digest}"),
                "--label",
                &format!("{OUTPUT_DIGEST_LABEL}={}", self.output_scope_sha256),
                &self.network_name,
            ]),
        )?;
        self.network_created = true;
        self.inspect_internal_network()?;
        let output = self.checked("start allowlist proxy", self.proxy_run_arguments())?;
        self.proxy_created = true;
        self.proxy_ever_started = true;
        let id = String::from_utf8_lossy(&output.stdout).trim().to_owned();
        if !id.is_empty() {
            self.proxy_container_id = Some(id);
        }
        self.checked(
            "connect allowlist proxy to external network",
            strings(&[
                "network",
                "connect",
                &self.config.external_network,
                &self.proxy_name,
            ]),
        )?;
        self.inspect_proxy_container()?;
        self.wait_until_ready()?;
        self.inspect_proxy_container()?;
        let proxy_url = format!("http://{PROXY_ALIAS}:{}", self.config.proxy_port);
        Ok(DockerNetworkBinding {
            docker_network: Some(self.network_name.clone()),
            proxy_environment: BTreeMap::from([
                ("HTTP_PROXY".to_owned(), proxy_url.clone()),
                ("HTTPS_PROXY".to_owned(), proxy_url.clone()),
                ("http_proxy".to_owned(), proxy_url.clone()),
                ("https_proxy".to_owned(), proxy_url),
                ("NO_PROXY".to_owned(), "localhost,127.0.0.1,::1".to_owned()),
                ("no_proxy".to_owned(), "localhost,127.0.0.1,::1".to_owned()),
                ("ALL_PROXY".to_owned(), String::new()),
                ("all_proxy".to_owned(), String::new()),
            ]),
        })
    }

    fn prepare_state(&mut self) -> Result<(), NetworkEnforcementError> {
        fs::create_dir(&self.state_dir).map_err(|error| {
            NetworkEnforcementError::new(format!(
                "unable to create private network-policy state {}: {error}",
                self.state_dir.display()
            ))
        })?;
        fs::set_permissions(&self.state_dir, fs::Permissions::from_mode(0o755))?;
        let metadata = fs::symlink_metadata(&self.config.proxy_executable)?;
        if !metadata.file_type().is_file() || metadata.file_type().is_symlink() {
            return Err(NetworkEnforcementError::new(format!(
                "native proxy executable is not a regular non-symlink file: {}",
                self.config.proxy_executable.display()
            )));
        }
        let source_digest = sha256_file(&self.config.proxy_executable)?;
        let mut source = File::open(&self.config.proxy_executable)?;
        let mut destination = OpenOptions::new()
            .create_new(true)
            .write(true)
            .mode(0o555)
            .open(&self.pinned_proxy_path)?;
        io::copy(&mut source, &mut destination)?;
        destination.sync_all()?;
        fs::set_permissions(&self.pinned_proxy_path, fs::Permissions::from_mode(0o555))?;
        let pinned_digest = sha256_file(&self.pinned_proxy_path)?;
        if pinned_digest != source_digest {
            return Err(NetworkEnforcementError::new(
                "content-bound native proxy copy changed during preparation",
            ));
        }
        self.proxy_binary_sha256 = Some(pinned_digest);
        for path in [&self.ready_path, &self.decision_log_path] {
            OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o666)
                .open(path)?
                .sync_all()?;
            fs::set_permissions(path, fs::Permissions::from_mode(0o666))?;
        }
        Ok(())
    }

    fn inspect_image(&mut self) -> Result<Value, NetworkEnforcementError> {
        let image = self.config.proxy_image.clone();
        let output = self.checked(
            "inspect local proxy image",
            strings(&["image", "inspect", &image]),
        )?;
        let item = one_object(&output.stdout, "proxy image inspection")?;
        let image_id = item
            .get("Id")
            .and_then(Value::as_str)
            .filter(|value| value.starts_with("sha256:") && value.len() == 71)
            .ok_or_else(|| {
                NetworkEnforcementError::new(
                    "proxy image inspection did not return an immutable sha256 image ID",
                )
            })?;
        let mut repo_digests = item
            .get("RepoDigests")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .map(|value| {
                        value.as_str().map(ToOwned::to_owned).ok_or_else(|| {
                            NetworkEnforcementError::new("proxy image RepoDigests are malformed")
                        })
                    })
                    .collect::<Result<Vec<_>, _>>()
            })
            .transpose()?
            .unwrap_or_default();
        repo_digests.sort();
        Ok(json!({
            "id": image_id,
            "repo_digests": repo_digests,
            "os": item.get("Os").cloned().unwrap_or(Value::Null),
            "architecture": item.get("Architecture").cloned().unwrap_or(Value::Null),
        }))
    }

    fn inspect_external_network(&mut self) -> Result<Value, NetworkEnforcementError> {
        let external = self.config.external_network.clone();
        let output = self.checked(
            "inspect external proxy network",
            strings(&["network", "inspect", &external]),
        )?;
        let item = one_object(&output.stdout, "external network inspection")?;
        let valid = item.get("Name").and_then(Value::as_str) == Some(external.as_str())
            && item.get("Internal").and_then(Value::as_bool) == Some(false)
            && item
                .get("Id")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
            && item
                .get("Driver")
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
            && item.get("IPAM").is_some_and(Value::is_object)
            && item
                .get("Options")
                .map_or(true, |value| value.is_null() || value.is_object());
        if !valid {
            return Err(NetworkEnforcementError::new(format!(
                "proxy external network {:?} is malformed or internal",
                self.config.external_network
            )));
        }
        let configuration = json!({
            "name": item["Name"],
            "id": item["Id"],
            "driver": item["Driver"],
            "scope": item.get("Scope").cloned().unwrap_or(Value::Null),
            "internal": item["Internal"],
            "attachable": item.get("Attachable").cloned().unwrap_or(Value::Null),
            "ingress": item.get("Ingress").cloned().unwrap_or(Value::Null),
            "enable_ipv4": item.get("EnableIPv4").cloned().unwrap_or(Value::Null),
            "enable_ipv6": item.get("EnableIPv6").cloned().unwrap_or(Value::Null),
            "ipam": item["IPAM"],
            "options": item.get("Options").cloned().unwrap_or_else(|| json!({})),
        });
        let mut identity = configuration
            .as_object()
            .expect("network configuration is an object")
            .clone();
        identity.insert(
            "configuration_sha256".to_owned(),
            Value::String(hex_digest(Sha256::digest(
                canonical_json(&configuration).as_bytes(),
            ))),
        );
        self.validation
            .insert("external_network_inspected".to_owned(), Some(true));
        Ok(Value::Object(identity))
    }

    fn inspect_internal_network(&mut self) -> Result<(), NetworkEnforcementError> {
        let network = self.network_name.clone();
        let output = self.checked(
            "inspect internal policy network",
            strings(&["network", "inspect", &network]),
        )?;
        let item = one_object(&output.stdout, "internal network inspection")?;
        let options = item.get("Options").and_then(Value::as_object);
        let labels = item.get("Labels").and_then(Value::as_object);
        let policy_digest = self.config.policy.comparability_digest();
        let valid = item.get("Name").and_then(Value::as_str) == Some(network.as_str())
            && item.get("Driver").and_then(Value::as_str) == Some("bridge")
            && item.get("Internal").and_then(Value::as_bool) == Some(true)
            && item.get("EnableIPv6").and_then(Value::as_bool) == Some(false)
            && options
                .and_then(|values| values.get(INHIBIT_IPV4_OPTION))
                .and_then(Value::as_str)
                .is_some_and(|value| value.eq_ignore_ascii_case("true"))
            && labels
                .and_then(|values| values.get(POLICY_COMPONENT_LABEL))
                .and_then(Value::as_str)
                == Some("network-policy")
            && labels
                .and_then(|values| values.get(POLICY_DIGEST_LABEL))
                .and_then(Value::as_str)
                == Some(policy_digest.as_str())
            && labels
                .and_then(|values| values.get(OUTPUT_DIGEST_LABEL))
                .and_then(Value::as_str)
                == Some(self.output_scope_sha256.as_str());
        if !valid {
            return Err(NetworkEnforcementError::new(
                "created Docker network did not inspect as the required output-scoped internal bridge",
            ));
        }
        self.validation
            .insert("internal_network_inspected".to_owned(), Some(true));
        Ok(())
    }

    fn proxy_process_arguments(&self) -> Vec<String> {
        let mut arguments = vec![
            "__network-proxy".to_owned(),
            "--listen-host".to_owned(),
            PROXY_ALIAS.to_owned(),
            "--listen-port".to_owned(),
            self.config.proxy_port.to_string(),
            "--ready-file".to_owned(),
            format!("{PROXY_STATE_PATH}/proxy.ready.json"),
            "--decision-log".to_owned(),
            format!("{PROXY_STATE_PATH}/proxy-decisions.jsonl"),
        ];
        for host in self.config.policy.allowed_hosts() {
            arguments.extend(["--allow-host".to_owned(), host.clone()]);
        }
        for port in self.allowed_ports() {
            arguments.extend(["--allow-port".to_owned(), port.to_string()]);
        }
        arguments
    }

    fn proxy_run_arguments(&self) -> Vec<OsString> {
        let image_id = self
            .proxy_image_identity
            .as_ref()
            .and_then(|value| value.get("id"))
            .and_then(Value::as_str)
            .expect("proxy image inspected before command construction");
        let binary_mount = format!(
            "type=bind,src={},dst={PINNED_PROXY_PATH},readonly",
            self.pinned_proxy_path.display()
        );
        let state_mount = format!(
            "type=bind,src={},dst={PROXY_STATE_PATH}",
            self.state_dir.display()
        );
        let policy_digest = self.config.policy.comparability_digest();
        let mut arguments = strings(&[
            "run",
            "--detach",
            "--pull=never",
            "--name",
            &self.proxy_name,
            "--network",
            &self.network_name,
            "--network-alias",
            PROXY_ALIAS,
            "--label",
            "io.tasktrace.component=network-policy-proxy",
            "--label",
            &format!("{POLICY_DIGEST_LABEL}={policy_digest}"),
            "--label",
            &format!("{OUTPUT_DIGEST_LABEL}={}", self.output_scope_sha256),
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--restart=no",
            "--sysctl",
            "net.ipv4.ip_forward=0",
            "--sysctl",
            "net.ipv6.conf.all.forwarding=0",
            "--user",
            "65534:65534",
            "--pids-limit",
            "64",
            "--memory",
            "128m",
            "--cpus",
            "1",
            "--tmpfs",
            "/tmp:rw,noexec,nosuid,nodev,size=16m",
            "--mount",
            &binary_mount,
            "--mount",
            &state_mount,
            "--entrypoint",
            PINNED_PROXY_PATH,
            image_id,
        ]);
        arguments.extend(
            self.proxy_process_arguments()
                .into_iter()
                .map(OsString::from),
        );
        arguments
    }

    fn inspect_proxy_container(&mut self) -> Result<(), NetworkEnforcementError> {
        let proxy = self.proxy_name.clone();
        let output = self.checked(
            "inspect allowlist proxy container",
            strings(&["container", "inspect", &proxy]),
        )?;
        let item = one_object(&output.stdout, "proxy container inspection")?;
        let state = item.get("State").and_then(Value::as_object);
        let config = item.get("Config").and_then(Value::as_object);
        let host = item.get("HostConfig").and_then(Value::as_object);
        let networks = item
            .pointer("/NetworkSettings/Networks")
            .and_then(Value::as_object);
        let internal = networks.and_then(|values| values.get(&self.network_name));
        let internal_ip = internal
            .and_then(|value| value.get("IPAddress"))
            .and_then(Value::as_str);
        let aliases = internal
            .and_then(|value| value.get("Aliases"))
            .and_then(Value::as_array);
        let labels = config
            .and_then(|values| values.get("Labels"))
            .and_then(Value::as_object);
        let cap_drop = host
            .and_then(|values| values.get("CapDrop"))
            .and_then(Value::as_array);
        let security = host
            .and_then(|values| values.get("SecurityOpt"))
            .and_then(Value::as_array);
        let expected_image = self
            .proxy_image_identity
            .as_ref()
            .and_then(|value| value.get("id").and_then(Value::as_str));
        let expected_command = self.proxy_process_arguments();
        let command_matches = config
            .and_then(|values| values.get("Cmd"))
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values.iter().map(Value::as_str).collect::<Option<Vec<_>>>()
                    == Some(expected_command.iter().map(String::as_str).collect())
            });
        let entrypoint_matches = config
            .and_then(|values| values.get("Entrypoint"))
            .and_then(Value::as_array)
            .is_some_and(|values| {
                values.len() == 1 && values[0].as_str() == Some(PINNED_PROXY_PATH)
            });
        let sysctls = host
            .and_then(|values| values.get("Sysctls"))
            .and_then(Value::as_object);
        let restart = host
            .and_then(|values| values.get("RestartPolicy"))
            .and_then(Value::as_object);
        let port_bindings_absent = host
            .and_then(|values| values.get("PortBindings"))
            .map_or(true, |value| {
                value.is_null() || value.as_object().is_some_and(Map::is_empty)
            });
        let mounts_valid = self.proxy_mounts_valid(item.get("Mounts"));
        let policy_digest = self.config.policy.comparability_digest();
        let valid = state
            .and_then(|values| values.get("Running"))
            .and_then(Value::as_bool)
            == Some(true)
            && config
                .and_then(|values| values.get("Image"))
                .and_then(Value::as_str)
                == expected_image
            && config
                .and_then(|values| values.get("User"))
                .and_then(Value::as_str)
                == Some("65534:65534")
            && command_matches
            && entrypoint_matches
            && labels
                .and_then(|values| values.get(POLICY_COMPONENT_LABEL))
                .and_then(Value::as_str)
                == Some("network-policy-proxy")
            && labels
                .and_then(|values| values.get(POLICY_DIGEST_LABEL))
                .and_then(Value::as_str)
                == Some(policy_digest.as_str())
            && labels
                .and_then(|values| values.get(OUTPUT_DIGEST_LABEL))
                .and_then(Value::as_str)
                == Some(self.output_scope_sha256.as_str())
            && host
                .and_then(|values| values.get("ReadonlyRootfs"))
                .and_then(Value::as_bool)
                == Some(true)
            && host
                .and_then(|values| values.get("PidsLimit"))
                .and_then(Value::as_i64)
                == Some(64)
            && host
                .and_then(|values| values.get("Memory"))
                .and_then(Value::as_u64)
                == Some(128 * 1024 * 1024)
            && host
                .and_then(|values| values.get("NanoCpus"))
                .and_then(Value::as_u64)
                == Some(1_000_000_000)
            && restart
                .and_then(|values| values.get("Name"))
                .and_then(Value::as_str)
                == Some("no")
            && sysctls
                .and_then(|values| values.get("net.ipv4.ip_forward"))
                .and_then(Value::as_str)
                == Some("0")
            && sysctls
                .and_then(|values| values.get("net.ipv6.conf.all.forwarding"))
                .and_then(Value::as_str)
                == Some("0")
            && cap_drop
                .is_some_and(|values| values.iter().any(|value| value.as_str() == Some("ALL")))
            && security.is_some_and(|values| {
                values.iter().any(|value| {
                    value
                        .as_str()
                        .is_some_and(|value| value.contains("no-new-privileges"))
                })
            })
            && networks.is_some_and(|values| {
                values.len() == 2
                    && values.contains_key(&self.network_name)
                    && values.contains_key(&self.config.external_network)
            })
            && aliases.is_some_and(|values| {
                values
                    .iter()
                    .any(|value| value.as_str() == Some(PROXY_ALIAS))
            })
            && internal_ip.is_some_and(|value| !value.is_empty())
            && port_bindings_absent
            && mounts_valid;
        if !valid {
            return Err(NetworkEnforcementError::new(
                "allowlist proxy inspection did not match required running, content-bound, hardened topology",
            ));
        }
        if let (Some(previous), Some(current)) = (&self.proxy_internal_ip, internal_ip) {
            if previous != current {
                return Err(NetworkEnforcementError::new(
                    "allowlist proxy internal address changed during the policy session",
                ));
            }
        }
        self.proxy_internal_ip = internal_ip.map(ToOwned::to_owned);
        self.validation
            .insert("proxy_container_inspected".to_owned(), Some(true));
        Ok(())
    }

    fn proxy_mounts_valid(&self, mounts: Option<&Value>) -> bool {
        let Some(mounts) = mounts.and_then(Value::as_array) else {
            return false;
        };
        let binary_source = self.pinned_proxy_path.canonicalize().ok();
        let state_source = self.state_dir.canonicalize().ok();
        let mut binary = false;
        let mut state = false;
        for mount in mounts {
            let destination = mount.get("Destination").and_then(Value::as_str);
            let source = mount
                .get("Source")
                .and_then(Value::as_str)
                .and_then(|value| Path::new(value).canonicalize().ok());
            let read_write = mount.get("RW").and_then(Value::as_bool);
            if destination == Some(PINNED_PROXY_PATH)
                && source == binary_source
                && read_write == Some(false)
            {
                binary = true;
            }
            if destination == Some(PROXY_STATE_PATH)
                && source == state_source
                && read_write == Some(true)
            {
                state = true;
            }
        }
        binary && state
    }

    fn wait_until_ready(&mut self) -> Result<(), NetworkEnforcementError> {
        let deadline = Instant::now() + self.config.readiness_timeout;
        loop {
            if let Ok(raw) = fs::read_to_string(&self.ready_path) {
                if !raw.trim().is_empty() {
                    let payload: Value = serde_json::from_str(&raw).map_err(|_| {
                        NetworkEnforcementError::new(
                            "allowlist proxy wrote malformed readiness JSON",
                        )
                    })?;
                    let expected_ports = self.allowed_ports();
                    let valid = payload.get("ready").and_then(Value::as_bool) == Some(true)
                        && payload.get("listen_host").and_then(Value::as_str)
                            == self.proxy_internal_ip.as_deref()
                        && payload.get("listen_port").and_then(Value::as_u64)
                            == Some(self.config.proxy_port as u64)
                        && payload.get("allowed_ports") == Some(&json!(expected_ports));
                    if valid {
                        self.validation.insert("proxy_ready".to_owned(), Some(true));
                        self.validation
                            .insert("proxy_bound_to_internal_address".to_owned(), Some(true));
                        return Ok(());
                    }
                    return Err(NetworkEnforcementError::new(
                        "allowlist proxy readiness did not prove the internal listener and configured ports",
                    ));
                }
            }
            if Instant::now() >= deadline {
                return Err(NetworkEnforcementError::new(
                    "timed out waiting for allowlist proxy readiness",
                ));
            }
            thread::sleep(Duration::from_millis(50));
        }
    }

    fn inspect_workload_container(
        &mut self,
        workload_container_name: &str,
    ) -> Result<(), NetworkEnforcementError> {
        let output = self.checked(
            "inspect measured workload network policy",
            strings(&["container", "inspect", workload_container_name]),
        )?;
        let item = one_object(&output.stdout, "workload container inspection")?;
        let state_running = item.pointer("/State/Running").and_then(Value::as_bool) == Some(true);
        let network_mode = item
            .pointer("/HostConfig/NetworkMode")
            .and_then(Value::as_str);
        let cap_drop = item
            .pointer("/HostConfig/CapDrop")
            .and_then(Value::as_array);
        let networks = item
            .pointer("/NetworkSettings/Networks")
            .and_then(Value::as_object);
        let isolated = network_mode == Some(self.network_name.as_str())
            && networks
                .is_some_and(|values| values.len() == 1 && values.contains_key(&self.network_name))
            && cap_drop.is_some_and(|values| {
                values.iter().any(|value| {
                    value.as_str().is_some_and(|value| {
                        value
                            .strip_prefix("CAP_")
                            .unwrap_or(value)
                            .eq_ignore_ascii_case("NET_RAW")
                    })
                })
            });
        let environment = item
            .pointer("/Config/Env")
            .and_then(Value::as_array)
            .map(|values| {
                values
                    .iter()
                    .filter_map(Value::as_str)
                    .filter_map(|assignment| assignment.split_once('='))
                    .map(|(name, value)| (name.to_owned(), value.to_owned()))
                    .collect::<BTreeMap<_, _>>()
            })
            .unwrap_or_default();
        let expected = self
            .binding
            .as_ref()
            .map(|binding| &binding.proxy_environment)
            .ok_or_else(|| NetworkEnforcementError::new("allowlist binding is missing"))?;
        let environment_valid = expected
            .iter()
            .all(|(name, value)| environment.get(name) == Some(value))
            && environment.get("ALL_PROXY").map_or(true, String::is_empty)
            && environment.get("all_proxy").map_or(true, String::is_empty);
        if !state_running || !isolated || !environment_valid {
            return Err(NetworkEnforcementError::new(format!(
                "measured workload did not inspect as an active proxy-only internal-network sandbox: running={state_running}, network_mode_matches={}, network_membership_matches={}, net_raw_dropped={}, proxy_environment_matches={environment_valid}",
                network_mode == Some(self.network_name.as_str()),
                networks.is_some_and(|values| {
                    values.len() == 1 && values.contains_key(&self.network_name)
                }),
                cap_drop.is_some_and(|values| {
                    values.iter().any(|value| {
                        value
                            .as_str()
                            .is_some_and(|value| {
                                value
                                    .strip_prefix("CAP_")
                                    .unwrap_or(value)
                                    .eq_ignore_ascii_case("NET_RAW")
                            })
                    })
                }),
            )));
        }
        self.validation
            .insert("workload_container_inspected".to_owned(), Some(true));
        self.validation.insert(
            "workload_isolated_on_internal_network".to_owned(),
            Some(true),
        );
        self.validation
            .insert("proxy_environment_validated".to_owned(), Some(true));
        Ok(())
    }

    fn cleanup_allowlist(&mut self, inspect_health: bool) -> Vec<String> {
        let mut errors = Vec::new();
        if self.proxy_created || self.network_created || self.state_dir.exists() {
            self.cleanup.insert("attempted".to_owned(), true);
        }
        if self.proxy_created && inspect_health {
            match self.inspect_proxy_container() {
                Ok(()) => {
                    self.validation
                        .insert("proxy_healthy_at_cleanup".to_owned(), Some(true));
                }
                Err(error) => {
                    self.validation
                        .insert("proxy_healthy_at_cleanup".to_owned(), Some(false));
                    errors.push(format!("proxy unhealthy before cleanup: {error}"));
                }
            }
        }
        if self.proxy_created {
            let output = self.invoke(strings(&[
                "container",
                "stop",
                "--time",
                "5",
                &self.proxy_name,
            ]));
            match output {
                Ok(output)
                    if output.status.success()
                        || output_diagnostic(&output).contains("is not running")
                        || output_diagnostic(&output).contains("No such container") =>
                {
                    self.cleanup.insert("proxy_stopped".to_owned(), true);
                }
                Ok(output) => errors.push(format!(
                    "stop proxy container: {}",
                    output_diagnostic(&output)
                )),
                Err(error) => errors.push(format!("stop proxy container: {error}")),
            }
            let output = self.invoke(strings(&["container", "rm", "--force", &self.proxy_name]));
            match output {
                Ok(output)
                    if output.status.success()
                        || output_diagnostic(&output).contains("No such container") =>
                {
                    self.cleanup.insert("proxy_removed".to_owned(), true);
                    self.proxy_created = false;
                }
                Ok(output) => errors.push(format!(
                    "remove proxy container: {}",
                    output_diagnostic(&output)
                )),
                Err(error) => errors.push(format!("remove proxy container: {error}")),
            }
        }
        let proxy_name = self.proxy_name.clone();
        match self.verify_docker_absent("container", &proxy_name) {
            Ok(true) => {
                self.cleanup
                    .insert("proxy_absence_verified".to_owned(), true);
            }
            Ok(false) if !self.proxy_ever_started => {}
            Ok(false) => errors.push("proxy container survived cleanup".to_owned()),
            Err(error) => errors.push(format!("verify proxy removal: {error}")),
        }
        if self.network_created {
            let output = self.invoke(strings(&["network", "rm", &self.network_name]));
            match output {
                Ok(output)
                    if output.status.success()
                        || output_diagnostic(&output).contains("No such network") =>
                {
                    self.cleanup.insert("network_removed".to_owned(), true);
                    self.network_created = false;
                }
                Ok(output) => errors.push(format!(
                    "remove internal network: {}",
                    output_diagnostic(&output)
                )),
                Err(error) => errors.push(format!("remove internal network: {error}")),
            }
        }
        let network_name = self.network_name.clone();
        match self.verify_docker_absent("network", &network_name) {
            Ok(true) => {
                self.cleanup
                    .insert("network_absence_verified".to_owned(), true);
            }
            Ok(false) if !self.network_created && !self.proxy_ever_started => {}
            Ok(false) => errors.push("internal policy network survived cleanup".to_owned()),
            Err(error) => errors.push(format!("verify network removal: {error}")),
        }
        errors.extend(self.archive_decisions_and_remove_state());
        errors
    }

    fn verify_docker_absent(
        &mut self,
        kind: &str,
        name: &str,
    ) -> Result<bool, NetworkEnforcementError> {
        let output = self.invoke(strings(&[kind, "inspect", name]))?;
        if output.status.success() {
            return Ok(false);
        }
        let diagnostic = output_diagnostic(&output);
        let missing = diagnostic.contains("No such container")
            || diagnostic.contains("No such network")
            || diagnostic.contains("No such object")
            || (kind == "network"
                && diagnostic.starts_with("Error response from daemon: network ")
                && diagnostic.ends_with(" not found"));
        if missing {
            Ok(true)
        } else {
            Err(NetworkEnforcementError::new(format!(
                "Docker {kind} absence inspection was ambiguous: {diagnostic}"
            )))
        }
    }

    fn archive_decisions_and_remove_state(&mut self) -> Vec<String> {
        if !self.state_dir.exists() {
            return Vec::new();
        }
        let mut errors = Vec::new();
        if self.proxy_ever_started {
            match self.validate_decision_log() {
                Ok(evidence) => {
                    self.decision_log_evidence = Some(evidence);
                    self.validation
                        .insert("decision_log_validated".to_owned(), Some(true));
                }
                Err(error) => {
                    self.validation
                        .insert("decision_log_validated".to_owned(), Some(false));
                    errors.push(format!("validate proxy decision log: {error}"));
                }
            }
        }
        let temporary = self
            .config
            .output_dir
            .parent()
            .unwrap_or(Path::new("/tmp"))
            .join(format!(".tasktrace-network-decisions-{}.tmp", self.token));
        let archive_result = (|| -> Result<(), NetworkEnforcementError> {
            if temporary.exists() {
                fs::remove_file(&temporary)?;
            }
            fs::copy(&self.decision_log_path, &temporary)?;
            fs::set_permissions(&temporary, fs::Permissions::from_mode(0o644))?;
            fs::rename(&temporary, &self.archived_decision_log_path)?;
            if let Some(expected) = self
                .decision_log_evidence
                .as_ref()
                .and_then(|value| value.get("sha256"))
                .and_then(Value::as_str)
            {
                let observed = sha256_file(&self.archived_decision_log_path)?;
                if observed != expected {
                    return Err(NetworkEnforcementError::new(
                        "archived proxy decision log digest changed during copy",
                    ));
                }
            }
            Ok(())
        })();
        let _ = fs::remove_file(&temporary);
        match archive_result {
            Ok(()) => {
                self.cleanup
                    .insert("decision_log_archived".to_owned(), true);
            }
            Err(error) => errors.push(format!("archive proxy decision log: {error}")),
        }
        match fs::remove_dir_all(&self.state_dir) {
            Ok(()) => {
                self.cleanup.insert("host_state_removed".to_owned(), true);
            }
            Err(error) => errors.push(format!("remove host-only proxy state: {error}")),
        }
        if self.state_dir.exists() {
            errors.push(format!(
                "host-only proxy state survived cleanup: {}",
                self.state_dir.display()
            ));
        } else {
            self.cleanup
                .insert("host_state_absence_verified".to_owned(), true);
        }
        errors
    }

    fn validate_decision_log(&self) -> Result<Value, NetworkEnforcementError> {
        let file = File::open(&self.decision_log_path)?;
        let mut reader = BufReader::new(file);
        let mut digest = Sha256::new();
        let mut bytes = 0_u64;
        let mut records = 0_u64;
        let mut counts = BTreeMap::<String, u64>::new();
        let mut allowed_ips = BTreeSet::<String>::new();
        loop {
            let mut line = Vec::new();
            let count = reader.read_until(b'\n', &mut line)?;
            if count == 0 {
                break;
            }
            bytes = bytes.saturating_add(count as u64);
            records = records.saturating_add(1);
            if bytes > MAX_DECISION_LOG_BYTES {
                return Err(NetworkEnforcementError::new(
                    "proxy decision log exceeds the 64 MiB evidence bound",
                ));
            }
            if records > MAX_DECISION_LOG_RECORDS {
                return Err(NetworkEnforcementError::new(
                    "proxy decision log exceeds the record-count evidence bound",
                ));
            }
            if !line.ends_with(b"\n") {
                return Err(NetworkEnforcementError::new(
                    "proxy decision log contains an incomplete terminal record",
                ));
            }
            digest.update(&line);
            let record: Value = serde_json::from_slice(&line).map_err(|_| {
                NetworkEnforcementError::new("proxy decision log contains malformed JSON")
            })?;
            let object = record.as_object().ok_or_else(|| {
                NetworkEnforcementError::new("proxy decision log record must be a JSON object")
            })?;
            let event = record.get("event").and_then(Value::as_str).ok_or_else(|| {
                NetworkEnforcementError::new("proxy decision record does not name an event")
            })?;
            if record.get("schema_version").and_then(Value::as_str) != Some(PROXY_SCHEMA_VERSION)
                || !record
                    .get("timestamp")
                    .and_then(Value::as_str)
                    .is_some_and(|value| !value.is_empty())
                || !matches!(
                    event,
                    "proxy_ready" | "proxy_stopped" | "request_decision" | "request_error"
                )
            {
                return Err(NetworkEnforcementError::new(
                    "proxy decision log record does not match its schema",
                ));
            }
            let secret_bearing_field = object.keys().any(|name| {
                let lower = name.to_ascii_lowercase();
                lower.contains("url")
                    || lower.contains("path")
                    || lower.contains("header")
                    || lower.contains("authorization")
                    || lower.contains("cookie")
                    || lower.contains("body")
                    || lower.contains("query")
                    || lower.contains("target")
            });
            if secret_bearing_field {
                return Err(NetworkEnforcementError::new(
                    "proxy decision log contains a prohibited request-content field",
                ));
            }
            let base_keys = ["schema_version", "timestamp", "event"];
            let event_keys: &[&str] = match event {
                "proxy_ready" | "proxy_stopped" => &[
                    "ready",
                    "listen_host",
                    "listen_port",
                    "allowed_ports",
                    "allowed_hosts",
                ],
                "request_decision"
                    if record.get("decision").and_then(Value::as_str) == Some("allowed") =>
                {
                    &[
                        "request_id",
                        "decision",
                        "reason",
                        "method",
                        "host",
                        "port",
                        "destination_ip",
                    ]
                }
                "request_decision" => &["request_id", "decision", "reason"],
                "request_error" => &["request_id", "reason"],
                _ => unreachable!("validated proxy event"),
            };
            if object.len() != base_keys.len() + event_keys.len()
                || object.keys().any(|name| {
                    !base_keys.contains(&name.as_str()) && !event_keys.contains(&name.as_str())
                })
            {
                return Err(NetworkEnforcementError::new(
                    "proxy decision log record contains missing or unsupported fields",
                ));
            }
            if matches!(event, "proxy_ready" | "proxy_stopped") {
                let ready = event == "proxy_ready";
                let lifecycle_valid = record.get("ready").and_then(Value::as_bool) == Some(ready)
                    && record.get("listen_host").and_then(Value::as_str)
                        == self.proxy_internal_ip.as_deref()
                    && record.get("listen_port").and_then(Value::as_u64)
                        == Some(self.config.proxy_port as u64)
                    && record.get("allowed_ports") == Some(&json!(self.allowed_ports()))
                    && record.get("allowed_hosts")
                        == Some(&json!(self.config.policy.allowed_hosts()));
                if !lifecycle_valid {
                    return Err(NetworkEnforcementError::new(
                        "proxy lifecycle decision record does not bind the active policy",
                    ));
                }
            }
            *counts.entry(event.to_owned()).or_default() += 1;
            if event == "request_decision"
                && record.get("decision").and_then(Value::as_str) == Some("allowed")
            {
                let host = record.get("host").and_then(Value::as_str);
                let port = record.get("port").and_then(Value::as_u64);
                let destination = record
                    .get("destination_ip")
                    .and_then(Value::as_str)
                    .and_then(|value| value.parse::<std::net::IpAddr>().ok());
                let valid = host.is_some_and(|host| {
                    self.config
                        .policy
                        .allowed_hosts()
                        .iter()
                        .any(|value| value == host)
                }) && port.is_some_and(|port| {
                    port <= u16::MAX as u64 && self.allowed_ports().contains(&(port as u16))
                }) && destination.is_some_and(globally_routable)
                    && record
                        .get("request_id")
                        .and_then(Value::as_str)
                        .is_some_and(|value| {
                            value.len() == 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
                    && record
                        .get("method")
                        .and_then(Value::as_str)
                        .is_some_and(|value| {
                            !value.is_empty()
                                && value.bytes().all(|byte| {
                                    !byte.is_ascii_lowercase()
                                        && (byte.is_ascii_alphanumeric()
                                            || b"!#$%&'*+-.^_`|~".contains(&byte))
                                })
                        });
                if !valid {
                    return Err(NetworkEnforcementError::new(
                        "allowed proxy decision violates the enforced policy",
                    ));
                }
                allowed_ips.insert(destination.expect("validated destination").to_string());
            } else if event == "request_decision" {
                let denied_valid = record.get("decision").and_then(Value::as_str) == Some("denied")
                    && record
                        .get("request_id")
                        .and_then(Value::as_str)
                        .is_some_and(|value| {
                            value.len() == 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                        })
                    && record
                        .get("reason")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty() && value.len() <= 128);
                if !denied_valid {
                    return Err(NetworkEnforcementError::new(
                        "denied proxy decision does not match the secret-free schema",
                    ));
                }
            } else if event == "request_error" {
                let error_valid = record
                    .get("request_id")
                    .and_then(Value::as_str)
                    .is_some_and(|value| {
                        value.len() == 16 && value.bytes().all(|byte| byte.is_ascii_hexdigit())
                    })
                    && record
                        .get("reason")
                        .and_then(Value::as_str)
                        .is_some_and(|value| !value.is_empty() && value.len() <= 128);
                if !error_valid {
                    return Err(NetworkEnforcementError::new(
                        "proxy request-error record does not match the secret-free schema",
                    ));
                }
            }
        }
        if counts.get("proxy_ready") != Some(&1) {
            return Err(NetworkEnforcementError::new(
                "proxy decision log must contain exactly one ready event",
            ));
        }
        if counts.get("proxy_stopped") != Some(&1) {
            return Err(NetworkEnforcementError::new(
                "proxy decision log must contain exactly one graceful-stop event",
            ));
        }
        Ok(json!({
            "schema_version": "tasktrace-network-proxy-decision-evidence.v1",
            "valid": true,
            "sha256": hex_digest(digest.finalize()),
            "bytes": bytes,
            "records": records,
            "event_counts": counts,
            "allowed_destination_ips": allowed_ips,
        }))
    }

    fn checked(
        &mut self,
        action: &str,
        arguments: Vec<OsString>,
    ) -> Result<Output, NetworkEnforcementError> {
        let output = self.invoke(arguments)?;
        if output.status.success() {
            Ok(output)
        } else {
            Err(NetworkEnforcementError::new(format!(
                "{action} failed: {}",
                output_diagnostic(&output)
            )))
        }
    }

    fn invoke(&mut self, arguments: Vec<OsString>) -> Result<Output, NetworkEnforcementError> {
        self.runner
            .run(OsStr::new(&self.config.docker), &arguments)
            .map_err(|error| {
                NetworkEnforcementError::new(format!(
                    "unable to execute Docker network-policy command: {error}"
                ))
            })
    }

    fn write_report(&self, best_effort: bool) -> Result<(), NetworkEnforcementError> {
        let parent = self.config.output_dir.parent().ok_or_else(|| {
            NetworkEnforcementError::new("network enforcement output has no parent directory")
        })?;
        let temporary = parent.join(format!(".tasktrace-network-enforcement-{}.tmp", self.token));
        let result = (|| -> Result<(), NetworkEnforcementError> {
            if temporary.exists() {
                fs::remove_file(&temporary)?;
            }
            let mut file = OpenOptions::new()
                .create_new(true)
                .write(true)
                .mode(0o600)
                .open(&temporary)?;
            serde_json::to_writer_pretty(&mut file, &self.report_payload())?;
            file.write_all(b"\n")?;
            file.sync_all()?;
            fs::rename(&temporary, &self.report_path)?;
            Ok(())
        })();
        let _ = fs::remove_file(&temporary);
        match result {
            Err(_) if best_effort => Ok(()),
            other => other,
        }
    }
}

impl Drop for DockerNetworkPolicySession {
    fn drop(&mut self) {
        if !self.closed {
            let _ = self.close();
        }
    }
}

fn validate_configuration(
    config: &DockerNetworkPolicyConfig,
) -> Result<(), NetworkEnforcementError> {
    if !config.output_dir.is_dir() {
        return Err(NetworkEnforcementError::new(format!(
            "network enforcement output directory does not exist: {}",
            config.output_dir.display()
        )));
    }
    if config.docker.trim().is_empty()
        || config.proxy_image.trim().is_empty()
        || config.external_network.trim().is_empty()
    {
        return Err(NetworkEnforcementError::new(
            "Docker, proxy image, and external network must be nonempty",
        ));
    }
    if config.proxy_port == 0 {
        return Err(NetworkEnforcementError::new(
            "proxy port must be from 1 through 65535",
        ));
    }
    let ports = config
        .allowed_ports
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    if config.policy.effective_mode() == NetworkMode::Allowlist
        && (ports.is_empty() || ports.contains(&0))
    {
        return Err(NetworkEnforcementError::new(
            "a nonempty allowlist requires valid allowed ports",
        ));
    }
    if config.readiness_timeout.is_zero() {
        return Err(NetworkEnforcementError::new(
            "proxy readiness timeout must be positive",
        ));
    }
    if config.policy.effective_mode() == NetworkMode::Allowlist {
        validate_workload_mount_sources(&config.workload_mount_sources)?;
    }
    Ok(())
}

pub fn validate_workload_mount_sources(sources: &[PathBuf]) -> Result<(), NetworkEnforcementError> {
    const SOCKET_NAMES: &[&str] = &[
        "containerd.sock",
        "cri-dockerd.sock",
        "crio.sock",
        "docker.sock",
        "podman.sock",
    ];
    const SOCKET_PATHS: &[&str] = &[
        "/run/containerd/containerd.sock",
        "/run/crio/crio.sock",
        "/run/docker.sock",
        "/run/podman/podman.sock",
        "/var/run/docker.sock",
        "/var/run/podman/podman.sock",
    ];
    for source in sources {
        let candidate = absolute_normalized(source)?;
        let named_socket = candidate
            .file_name()
            .and_then(OsStr::to_str)
            .is_some_and(|name| SOCKET_NAMES.contains(&name));
        let exposes_socket = SOCKET_PATHS.iter().any(|path| {
            absolute_normalized(Path::new(path))
                .is_ok_and(|socket| socket == candidate || socket.starts_with(&candidate))
        });
        if named_socket || exposes_socket {
            return Err(NetworkEnforcementError::new(format!(
                "allowlist policy rejects container-runtime socket mounts: {}",
                source.display()
            )));
        }
    }
    Ok(())
}

pub fn load_network_enforcement_report(
    output_dir: &Path,
) -> Result<Value, NetworkEnforcementError> {
    let path = output_dir.join("network_enforcement.json");
    let payload: Value = serde_json::from_slice(&fs::read(&path).map_err(|error| {
        NetworkEnforcementError::new(format!(
            "unable to read network enforcement report {}: {error}",
            path.display()
        ))
    })?)?;
    if !payload.is_object()
        || payload.get("schema_version").and_then(Value::as_str)
            != Some(NETWORK_ENFORCEMENT_SCHEMA_VERSION)
    {
        return Err(NetworkEnforcementError::new(
            "network enforcement report does not match the native schema",
        ));
    }
    Ok(payload)
}

fn ensure_state_is_disjoint(
    state: &Path,
    output: &Path,
    mounts: &[PathBuf],
) -> Result<(), NetworkEnforcementError> {
    let state = absolute_normalized(state)?;
    let output = absolute_normalized(output)?;
    if state.starts_with(&output) || output.starts_with(&state) {
        return Err(NetworkEnforcementError::new(
            "network policy state root must be outside the sandbox output",
        ));
    }
    for mount in mounts {
        let mount = absolute_normalized(mount)?;
        if state.starts_with(&mount) || mount.starts_with(&state) {
            return Err(NetworkEnforcementError::new(format!(
                "network policy host state must be disjoint from workload bind mount {}",
                mount.display()
            )));
        }
    }
    Ok(())
}

fn absolute_normalized(path: &Path) -> Result<PathBuf, NetworkEnforcementError> {
    if let Ok(path) = path.canonicalize() {
        return Ok(path);
    }
    if path.is_absolute() {
        Ok(path.to_path_buf())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn normalize_token(value: &str) -> Result<String, NetworkEnforcementError> {
    let token = value.to_ascii_lowercase();
    if !(8..=16).contains(&token.len()) || !token.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(NetworkEnforcementError::new(
            "network policy token must contain 8 through 16 hexadecimal characters",
        ));
    }
    Ok(token)
}

fn random_token() -> Result<String, NetworkEnforcementError> {
    let mut bytes = [0_u8; 8];
    File::open("/dev/urandom")?.read_exact(&mut bytes)?;
    Ok(hex_digest(bytes))
}

fn strings(values: &[&str]) -> Vec<OsString> {
    values.iter().map(OsString::from).collect()
}

fn one_object(raw: &[u8], description: &str) -> Result<Value, NetworkEnforcementError> {
    let payload: Value = serde_json::from_slice(raw)
        .map_err(|_| NetworkEnforcementError::new(format!("invalid {description} JSON")))?;
    let values = payload.as_array().ok_or_else(|| {
        NetworkEnforcementError::new(format!("{description} must contain exactly one object"))
    })?;
    if values.len() != 1 || !values[0].is_object() {
        return Err(NetworkEnforcementError::new(format!(
            "{description} must contain exactly one object"
        )));
    }
    Ok(values[0].clone())
}

fn output_diagnostic(output: &Output) -> String {
    let bytes = if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    };
    let value = String::from_utf8_lossy(bytes);
    let value = value.trim();
    if value.is_empty() {
        format!(
            "Docker exited with {} and no diagnostic output",
            display_status(output.status)
        )
    } else {
        value.chars().take(2000).collect()
    }
}

fn display_status(status: ExitStatus) -> String {
    status
        .code()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "signal".to_owned())
}

fn sha256_file(path: &Path) -> Result<String, NetworkEnforcementError> {
    let mut file = File::open(path)?;
    let mut digest = Sha256::new();
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        let count = file.read(&mut buffer)?;
        if count == 0 {
            break;
        }
        digest.update(&buffer[..count]);
    }
    Ok(hex_digest(digest.finalize()))
}

fn digest_domain(domain: &[u8], value: &[u8]) -> String {
    let mut digest = Sha256::new();
    digest.update(domain);
    digest.update(value);
    hex_digest(digest.finalize())
}

fn hex_digest(bytes: impl AsRef<[u8]>) -> String {
    let bytes = bytes.as_ref();
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        write!(&mut encoded, "{byte:02x}").expect("writing to a String cannot fail");
    }
    encoded
}

fn canonical_json(value: &Value) -> String {
    match value {
        Value::Null | Value::Bool(_) | Value::Number(_) => value.to_string(),
        Value::String(value) => serde_json::to_string(value).expect("serialize JSON string"),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(canonical_json)
                .collect::<Vec<_>>()
                .join(",")
        ),
        Value::Object(values) => {
            let mut fields = values.iter().collect::<Vec<_>>();
            fields.sort_by(|(left, _), (right, _)| left.cmp(right));
            format!(
                "{{{}}}",
                fields
                    .into_iter()
                    .map(|(name, value)| format!(
                        "{}:{}",
                        serde_json::to_string(name).expect("serialize JSON field"),
                        canonical_json(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(",")
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::os::unix::process::ExitStatusExt;
    use std::sync::{Arc, Mutex};
    use tempfile::TempDir;

    const IMAGE_ID: &str =
        "sha256:1111111111111111111111111111111111111111111111111111111111111111";
    const INTERNAL_IP: &str = "172.29.0.2";

    #[derive(Default)]
    struct FakeState {
        calls: Vec<Vec<String>>,
        internal_name: Option<String>,
        internal_labels: BTreeMap<String, String>,
        internal_exists: bool,
        proxy_name: Option<String>,
        proxy_exists: bool,
        proxy_running: bool,
        external_connected: bool,
        proxy_command: Vec<String>,
        proxy_labels: BTreeMap<String, String>,
        binary_source: Option<String>,
        state_source: Option<String>,
        ready_path: PathBuf,
        decision_log_path: PathBuf,
        workload_valid: bool,
        ambiguous_absence: bool,
    }

    struct FakeDocker {
        shared: Arc<Mutex<FakeState>>,
    }

    impl FakeDocker {
        fn success(value: impl AsRef<[u8]>) -> Output {
            Output {
                status: ExitStatus::from_raw(0),
                stdout: value.as_ref().to_vec(),
                stderr: Vec::new(),
            }
        }

        fn failure(message: &str) -> Output {
            Output {
                status: ExitStatus::from_raw(256),
                stdout: Vec::new(),
                stderr: message.as_bytes().to_vec(),
            }
        }

        fn option(arguments: &[String], option: &str) -> Option<String> {
            arguments
                .iter()
                .position(|value| value == option)
                .and_then(|index| arguments.get(index + 1))
                .cloned()
        }

        fn repeated(arguments: &[String], option: &str) -> Vec<String> {
            arguments
                .iter()
                .enumerate()
                .filter_map(|(index, value)| {
                    (value == option)
                        .then(|| arguments.get(index + 1).cloned())
                        .flatten()
                })
                .collect()
        }

        fn write_event(path: &Path, event: &str, command: &[String]) -> io::Result<()> {
            let hosts = Self::repeated(command, "--allow-host");
            let ports = Self::repeated(command, "--allow-port")
                .into_iter()
                .map(|value| value.parse::<u16>().unwrap())
                .collect::<Vec<_>>();
            let record = json!({
                "schema_version": PROXY_SCHEMA_VERSION,
                "timestamp": "unix:1.000000000Z",
                "event": event,
                "ready": event == "proxy_ready",
                "listen_host": INTERNAL_IP,
                "listen_port": DEFAULT_PROXY_PORT,
                "allowed_ports": ports,
                "allowed_hosts": hosts,
            });
            let mut file = OpenOptions::new().append(true).open(path)?;
            serde_json::to_writer(&mut file, &record)?;
            file.write_all(b"\n")
        }

        fn proxy_inspection(state: &FakeState) -> Value {
            let internal = state.internal_name.as_deref().unwrap();
            let external_network = if state.external_connected {
                json!({"IPAddress": "8.8.8.12", "Aliases": [state.proxy_name]})
            } else {
                Value::Null
            };
            let mut networks = Map::new();
            networks.insert(
                internal.to_owned(),
                json!({"IPAddress": INTERNAL_IP, "Aliases": [PROXY_ALIAS]}),
            );
            if state.external_connected {
                networks.insert("fixture-external".to_owned(), external_network);
            }
            json!({
                "State": {"Running": state.proxy_running},
                "Config": {
                    "Image": IMAGE_ID,
                    "User": "65534:65534",
                    "Cmd": state.proxy_command,
                    "Entrypoint": [PINNED_PROXY_PATH],
                    "Labels": state.proxy_labels,
                },
                "HostConfig": {
                    "ReadonlyRootfs": true,
                    "PidsLimit": 64,
                    "Memory": 128 * 1024 * 1024_u64,
                    "NanoCpus": 1_000_000_000_u64,
                    "RestartPolicy": {"Name": "no"},
                    "Sysctls": {
                        "net.ipv4.ip_forward": "0",
                        "net.ipv6.conf.all.forwarding": "0",
                    },
                    "CapDrop": ["ALL"],
                    "SecurityOpt": ["no-new-privileges:true"],
                    "PortBindings": {},
                },
                "NetworkSettings": {"Networks": networks},
                "Mounts": [
                    {
                        "Source": state.binary_source,
                        "Destination": PINNED_PROXY_PATH,
                        "RW": false,
                    },
                    {
                        "Source": state.state_source,
                        "Destination": PROXY_STATE_PATH,
                        "RW": true,
                    },
                ],
            })
        }
    }

    impl DockerCommandRunner for FakeDocker {
        fn run(&mut self, _program: &OsStr, raw: &[OsString]) -> io::Result<Output> {
            let arguments = raw
                .iter()
                .map(|value| value.to_string_lossy().into_owned())
                .collect::<Vec<_>>();
            let mut state = self.shared.lock().unwrap();
            state.calls.push(arguments.clone());
            let command = arguments.iter().map(String::as_str).collect::<Vec<_>>();
            match command.as_slice() {
                ["image", "inspect", "fixture-proxy"] => {
                    Ok(Self::success(serde_json::to_vec(&vec![json!({
                        "Id": IMAGE_ID,
                        "RepoDigests": ["fixture@sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"],
                        "Os": "linux",
                        "Architecture": "amd64",
                    })])?))
                }
                ["network", "inspect", "fixture-external"] => {
                    Ok(Self::success(serde_json::to_vec(&vec![json!({
                        "Name": "fixture-external",
                        "Id": "external-id",
                        "Driver": "bridge",
                        "Scope": "local",
                        "Internal": false,
                        "Attachable": false,
                        "Ingress": false,
                        "EnableIPv4": true,
                        "EnableIPv6": false,
                        "IPAM": {"Driver": "default", "Config": [{"Subnet": "8.8.8.0/24"}]},
                        "Options": {},
                    })])?))
                }
                ["network", "create", ..] => {
                    let name = arguments.last().cloned().unwrap();
                    let mut labels = BTreeMap::new();
                    for assignment in Self::repeated(&arguments, "--label") {
                        if let Some((key, value)) = assignment.split_once('=') {
                            labels.insert(key.to_owned(), value.to_owned());
                        }
                    }
                    state.internal_name = Some(name);
                    state.internal_labels = labels;
                    state.internal_exists = true;
                    Ok(Self::success("internal-id\n"))
                }
                ["network", "inspect", name] if state.internal_name.as_deref() == Some(*name) => {
                    if !state.internal_exists {
                        return Ok(if state.ambiguous_absence {
                            Self::failure("daemon inspection failed")
                        } else {
                            Self::failure("Error: No such network")
                        });
                    }
                    Ok(Self::success(serde_json::to_vec(&vec![json!({
                        "Name": name,
                        "Driver": "bridge",
                        "Internal": true,
                        "EnableIPv6": false,
                        "Options": {INHIBIT_IPV4_OPTION: "true"},
                        "Labels": state.internal_labels,
                    })])?))
                }
                ["run", ..] => {
                    let name = Self::option(&arguments, "--name").unwrap();
                    state.proxy_name = Some(name);
                    state.proxy_exists = true;
                    state.proxy_running = true;
                    state.proxy_labels = Self::repeated(&arguments, "--label")
                        .into_iter()
                        .filter_map(|value| {
                            value
                                .split_once('=')
                                .map(|(key, value)| (key.to_owned(), value.to_owned()))
                        })
                        .collect();
                    let mounts = Self::repeated(&arguments, "--mount");
                    state.binary_source = mounts.iter().find_map(|mount| {
                        mount
                            .contains(&format!("dst={PINNED_PROXY_PATH}"))
                            .then(|| mount.split(',').find_map(|item| item.strip_prefix("src=")))
                            .flatten()
                            .map(ToOwned::to_owned)
                    });
                    state.state_source = mounts.iter().find_map(|mount| {
                        mount
                            .contains(&format!("dst={PROXY_STATE_PATH}"))
                            .then(|| mount.split(',').find_map(|item| item.strip_prefix("src=")))
                            .flatten()
                            .map(ToOwned::to_owned)
                    });
                    let image = arguments
                        .iter()
                        .position(|value| value == IMAGE_ID)
                        .unwrap();
                    state.proxy_command = arguments[image + 1..].to_vec();
                    fs::write(
                        &state.ready_path,
                        serde_json::to_vec(&json!({
                            "ready": true,
                            "listen_host": INTERNAL_IP,
                            "listen_port": DEFAULT_PROXY_PORT,
                            "allowed_ports": [80, 443],
                        }))?,
                    )?;
                    Self::write_event(
                        &state.decision_log_path,
                        "proxy_ready",
                        &state.proxy_command,
                    )?;
                    Ok(Self::success("proxy-container-id\n"))
                }
                ["network", "connect", "fixture-external", name]
                    if state.proxy_name.as_deref() == Some(*name) =>
                {
                    state.external_connected = true;
                    Ok(Self::success(""))
                }
                ["container", "inspect", name] if state.proxy_name.as_deref() == Some(*name) => {
                    if !state.proxy_exists {
                        return Ok(if state.ambiguous_absence {
                            Self::failure("daemon inspection failed")
                        } else {
                            Self::failure("Error: No such container")
                        });
                    }
                    Ok(Self::success(serde_json::to_vec(&vec![
                        Self::proxy_inspection(&state),
                    ])?))
                }
                ["container", "inspect", "workload"] => {
                    let network = state.internal_name.as_deref().unwrap();
                    let proxy = format!("http://{PROXY_ALIAS}:{DEFAULT_PROXY_PORT}");
                    let mut environment = vec![
                        format!("HTTP_PROXY={proxy}"),
                        format!("HTTPS_PROXY={proxy}"),
                        format!("http_proxy={proxy}"),
                        format!("https_proxy={proxy}"),
                        "NO_PROXY=localhost,127.0.0.1,::1".to_owned(),
                        "no_proxy=localhost,127.0.0.1,::1".to_owned(),
                        "ALL_PROXY=".to_owned(),
                        "all_proxy=".to_owned(),
                    ];
                    if !state.workload_valid {
                        environment.push("ALL_PROXY=http://bypass.test:9999".to_owned());
                    }
                    Ok(Self::success(serde_json::to_vec(&vec![json!({
                        "State": {"Running": true},
                        "HostConfig": {"NetworkMode": network, "CapDrop": ["NET_RAW"]},
                        "NetworkSettings": {"Networks": {network: {"IPAddress": "172.29.0.3"}}},
                        "Config": {"Env": environment},
                    })])?))
                }
                ["container", "stop", "--time", "5", name]
                    if state.proxy_name.as_deref() == Some(*name) =>
                {
                    Self::write_event(
                        &state.decision_log_path,
                        "proxy_stopped",
                        &state.proxy_command,
                    )?;
                    state.proxy_running = false;
                    Ok(Self::success(format!("{name}\n")))
                }
                ["container", "rm", "--force", name]
                    if state.proxy_name.as_deref() == Some(*name) =>
                {
                    state.proxy_exists = false;
                    Ok(Self::success(format!("{name}\n")))
                }
                ["network", "rm", name] if state.internal_name.as_deref() == Some(*name) => {
                    state.internal_exists = false;
                    Ok(Self::success(format!("{name}\n")))
                }
                ["container", "inspect", _] => Ok(Self::failure("Error: No such container")),
                ["network", "inspect", _] => Ok(Self::failure("Error: No such network")),
                _ => panic!("unexpected fake Docker command: {arguments:?}"),
            }
        }
    }

    fn fixture(
        mode: NetworkMode,
        hosts: &[&str],
    ) -> (TempDir, DockerNetworkPolicyConfig, Arc<Mutex<FakeState>>) {
        let temporary = TempDir::new().unwrap();
        let output = temporary.path().join("output");
        fs::create_dir(&output).unwrap();
        let executable = temporary.path().join("tasktrace");
        fs::write(&executable, b"native-proxy-test-binary").unwrap();
        fs::set_permissions(&executable, fs::Permissions::from_mode(0o755)).unwrap();
        let policy = NetworkPolicy::new(mode, hosts).unwrap();
        let mut config = DockerNetworkPolicyConfig::new(policy, &output)
            .unwrap()
            .with_token("a1b2c3d4");
        config.proxy_image = "fixture-proxy".to_owned();
        config.external_network = "fixture-external".to_owned();
        config.proxy_executable = executable;
        config.state_root = temporary.path().to_path_buf();
        config.readiness_timeout = Duration::from_millis(100);
        let state_dir = temporary
            .path()
            .join(".tasktrace-network-policy-state-a1b2c3d4");
        let shared = Arc::new(Mutex::new(FakeState {
            ready_path: state_dir.join("proxy.ready.json"),
            decision_log_path: state_dir.join("proxy-decisions.jsonl"),
            workload_valid: true,
            ..FakeState::default()
        }));
        (temporary, config, shared)
    }

    fn session_with_fake(
        config: DockerNetworkPolicyConfig,
        shared: &Arc<Mutex<FakeState>>,
    ) -> DockerNetworkPolicySession {
        DockerNetworkPolicySession::with_runner(
            config,
            Box::new(FakeDocker {
                shared: Arc::clone(shared),
            }),
        )
        .unwrap()
    }

    #[test]
    fn allowlist_lifecycle_binds_validates_and_removes_every_resource() {
        let (temporary, config, shared) =
            fixture(NetworkMode::Allowlist, &["Allowed.Test", "other.example"]);
        let output = config.output_dir.clone();
        let state_dir = temporary
            .path()
            .join(".tasktrace-network-policy-state-a1b2c3d4");
        let mut session = session_with_fake(config, &shared);
        let binding = session.prepare().unwrap();
        assert_eq!(
            binding.docker_network.as_deref(),
            Some("tasktrace-policy-a1b2c3d4")
        );
        assert_eq!(
            binding
                .proxy_environment
                .get("HTTPS_PROXY")
                .map(String::as_str),
            Some("http://tasktrace-egress-proxy:3128")
        );
        assert_eq!(
            binding.applied_environment(&[
                "KEEP=value".to_owned(),
                "HTTP_PROXY=http://untrusted.test:9".to_owned(),
                "ALL_PROXY=http://bypass.test:9".to_owned(),
            ]),
            vec![
                "KEEP=value",
                "ALL_PROXY=",
                "HTTPS_PROXY=http://tasktrace-egress-proxy:3128",
                "HTTP_PROXY=http://tasktrace-egress-proxy:3128",
                "NO_PROXY=localhost,127.0.0.1,::1",
                "all_proxy=",
                "http_proxy=http://tasktrace-egress-proxy:3128",
                "https_proxy=http://tasktrace-egress-proxy:3128",
                "no_proxy=localhost,127.0.0.1,::1",
            ]
        );
        session
            .validate_before_workload_release("workload")
            .unwrap();
        session.close().unwrap();

        let report = session.report_payload();
        assert_eq!(report["state"], "closed");
        assert_eq!(
            report["policy"]["allowed_hosts"],
            json!(["allowed.test", "other.example"])
        );
        assert_eq!(report["proxy"]["image_identity"]["id"], IMAGE_ID);
        assert_eq!(report["cleanup"]["proxy_removed"], true);
        assert_eq!(report["cleanup"]["network_removed"], true);
        assert_eq!(report["cleanup"]["host_state_absence_verified"], true);
        assert_eq!(report["validation"]["decision_log_validated"], true);
        assert!(!state_dir.exists());
        assert!(output.join("network_proxy_decisions.jsonl").is_file());
        assert!(output.join("network_enforcement.json").is_file());

        let state = shared.lock().unwrap();
        assert!(!state.proxy_exists);
        assert!(!state.internal_exists);
        let run = state
            .calls
            .iter()
            .find(|call| call.first().map(String::as_str) == Some("run"))
            .unwrap();
        assert!(run
            .windows(2)
            .any(|pair| pair == ["--allow-host", "allowed.test"]));
        assert!(run
            .windows(2)
            .any(|pair| pair == ["--allow-host", "other.example"]));
        assert!(run.iter().any(|value| value == "--pull=never"));
        assert!(run.iter().any(|value| value == "--cap-drop=ALL"));
    }

    #[test]
    fn workload_bypass_environment_fails_before_release_and_cleanup_still_closes() {
        let (_temporary, config, shared) = fixture(NetworkMode::Allowlist, &["allowed.test"]);
        shared.lock().unwrap().workload_valid = false;
        let mut session = session_with_fake(config, &shared);
        session.prepare().unwrap();
        let error = session
            .validate_before_workload_release("workload")
            .unwrap_err()
            .to_string();
        assert!(error.contains("proxy-only internal-network sandbox"));
        session.close().unwrap();
        assert_eq!(session.report_payload()["state"], "closed");
    }

    #[test]
    fn ambiguous_absence_is_a_cleanup_failure_not_success() {
        let (_temporary, config, shared) = fixture(NetworkMode::Allowlist, &["allowed.test"]);
        let mut session = session_with_fake(config, &shared);
        session.prepare().unwrap();
        session
            .validate_before_workload_release("workload")
            .unwrap();
        shared.lock().unwrap().ambiguous_absence = true;
        let error = session.close().unwrap_err().to_string();
        assert!(error.contains("absence inspection was ambiguous"));
        assert_eq!(session.report_payload()["state"], "cleanup-failed");
        assert_eq!(
            session.report_payload()["cleanup"]["proxy_absence_verified"],
            false
        );
        assert_eq!(
            session.report_payload()["cleanup"]["network_absence_verified"],
            false
        );
    }

    #[test]
    fn secret_bearing_decision_fields_fail_closed_during_archival() {
        let (_temporary, config, shared) = fixture(NetworkMode::Allowlist, &["allowed.test"]);
        let mut session = session_with_fake(config, &shared);
        session.prepare().unwrap();
        session
            .validate_before_workload_release("workload")
            .unwrap();
        let path = shared.lock().unwrap().decision_log_path.clone();
        let mut file = OpenOptions::new().append(true).open(path).unwrap();
        serde_json::to_writer(
            &mut file,
            &json!({
                "schema_version": PROXY_SCHEMA_VERSION,
                "timestamp": "unix:2.000000000Z",
                "event": "request_decision",
                "request_id": "0000000000000001",
                "decision": "denied",
                "reason": "host_not_allowlisted",
                "path": "/private?token=must-not-be-retained",
            }),
        )
        .unwrap();
        file.write_all(b"\n").unwrap();
        let error = session.close().unwrap_err().to_string();
        assert!(
            error.contains("prohibited request-content field"),
            "{error}"
        );
        assert_eq!(session.report_payload()["state"], "cleanup-failed");
        assert_eq!(
            session.report_payload()["validation"]["decision_log_validated"],
            false
        );
    }

    #[test]
    fn public_and_empty_allowlist_preserve_native_network_modes_without_docker() {
        for (mode, hosts, expected) in [
            (NetworkMode::Public, Vec::<&str>::new(), None),
            (NetworkMode::Allowlist, Vec::<&str>::new(), Some("none")),
        ] {
            let (_temporary, config, shared) = fixture(mode, &hosts);
            let mut session = session_with_fake(config, &shared);
            assert_eq!(
                session.prepare().unwrap().docker_network.as_deref(),
                expected
            );
            session.close().unwrap();
            assert!(shared.lock().unwrap().calls.is_empty());
            assert_eq!(session.report_payload()["state"], "closed");
        }
    }

    #[test]
    fn runtime_socket_mounts_and_overlapping_state_fail_closed() {
        let error = validate_workload_mount_sources(&[PathBuf::from("/var/run/docker.sock")])
            .unwrap_err()
            .to_string();
        assert!(error.contains("socket mounts"));

        let (temporary, mut config, _shared) = fixture(NetworkMode::Allowlist, &["allowed.test"]);
        config.state_root = config.output_dir.clone();
        let error = DockerNetworkPolicySession::new(config)
            .err()
            .expect("overlap must fail")
            .to_string();
        assert!(error.contains("outside the sandbox output"));
        drop(temporary);
    }

    #[test]
    fn comparability_ignores_ephemeral_token_but_includes_ports() {
        let (_one, config, shared) = fixture(NetworkMode::Allowlist, &["allowed.test"]);
        let first = session_with_fake(config.clone().with_token("11111111"), &shared);
        let second = session_with_fake(config.clone().with_token("22222222"), &shared);
        assert_eq!(first.comparability_digest(), second.comparability_digest());
        let mut changed = config.with_token("33333333");
        changed.allowed_ports = vec![443];
        let third = session_with_fake(changed, &shared);
        assert_ne!(first.comparability_digest(), third.comparability_digest());
    }
}
