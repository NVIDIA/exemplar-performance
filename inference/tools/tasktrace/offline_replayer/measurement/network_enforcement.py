# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fail-closed Docker enforcement for offline-replay network policy.

The measured sandbox is the only consumer of :class:`NetworkPolicy` exposed
to callers.  Allowlist infrastructure is deliberately created in separate
Docker resources, outside the sandbox cgroup, and is torn down when the
session exits.  The proxy has an external network attachment; the measured
sandbox receives only an ephemeral ``--internal`` bridge attachment and
therefore has no direct egress route.

This module never pulls an image.  It resolves a configured local proxy image
to its immutable Docker image ID and runs that ID with ``--pull=never``.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .network_policy import NetworkMode, NetworkPolicy

NETWORK_ENFORCEMENT_SCHEMA_VERSION = "tasktrace-docker-network-enforcement.v1"
NETWORK_ENFORCEMENT_COMPARABILITY_SCHEMA_VERSION = "tasktrace-docker-network-enforcement-comparability.v1"
DEFAULT_PROXY_IMAGE = "python:3.12-slim"
DEFAULT_PROXY_PORT = 3128
PROXY_ALIAS = "tasktrace-egress-proxy"
_INHIBIT_IPV4_OPTION = "com.docker.network.bridge.inhibit_ipv4"
_PROXY_LOG_SCHEMA_VERSION = "tasktrace-network-proxy.v1"
_MAX_DECISION_LOG_BYTES = 64 * 1024 * 1024
_MAX_DECISION_LOG_RECORDS = 1_000_000

CommandRunner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


class NetworkEnforcementError(RuntimeError):
    """Raised when a requested Docker network policy is not proven active."""


@dataclass(frozen=True)
class DockerNetworkBinding:
    """The small runtime-facing result of one prepared policy session."""

    docker_network: str | None
    proxy_environment: Mapping[str, str]

    @property
    def docker_run_args(self) -> tuple[str, ...]:
        """Return the Docker override without inventing one for public mode."""
        if self.docker_network is None:
            return ()
        return ("--network", self.docker_network)


def _default_command_runner(
    command: Sequence[str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(command), text=True, capture_output=True, check=False)


class DockerNetworkPolicySession:
    """Prepare and clean up Docker resources enforcing one network policy.

    The object is a single-use context manager.  Use ``docker_run_args`` and
    ``proxy_environment`` when constructing the measured sandbox command.
    Public mode intentionally contributes no Docker network argument.  Both
    explicit no-network and an empty allowlist contribute ``--network none``.

    ``output_dir/network_enforcement.json`` is updated as the lifecycle moves
    through preparation and cleanup.  It records policy, local proxy image
    identity, validation, and the fact that the sidecar is outside the
    measured sandbox cgroup.
    """

    def __init__(
        self,
        policy: NetworkPolicy,
        output_dir: Path,
        *,
        docker: str = "docker",
        proxy_image: str = DEFAULT_PROXY_IMAGE,
        proxy_source: Path | None = None,
        state_root: Path | None = None,
        workload_mount_sources: Sequence[str] = (),
        external_network: str = "bridge",
        proxy_port: int = DEFAULT_PROXY_PORT,
        allowed_ports: Sequence[int] = (80, 443),
        readiness_timeout_seconds: float = 10.0,
        command_runner: CommandRunner | None = None,
        token_factory: Callable[[], str] | None = None,
        monotonic: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        if not isinstance(policy, NetworkPolicy):
            raise TypeError("policy must be a NetworkPolicy")
        if not proxy_image:
            raise ValueError("proxy_image must not be empty")
        if not external_network:
            raise ValueError("external_network must not be empty")
        if not 1 <= proxy_port <= 65535:
            raise ValueError("proxy_port must be between 1 and 65535")
        if any(not isinstance(port, int) or isinstance(port, bool) or not 1 <= port <= 65535 for port in allowed_ports):
            raise ValueError("allowed_ports must contain TCP ports from 1 to 65535")
        normalized_ports = tuple(sorted(set(allowed_ports)))
        if policy.effective_mode is NetworkMode.ALLOWLIST and not normalized_ports:
            raise ValueError("a non-empty allowlist requires at least one allowed port")
        if readiness_timeout_seconds < 0:
            raise ValueError("readiness_timeout_seconds must be non-negative")

        self.policy = policy
        self.output_dir = Path(output_dir)
        self.docker = docker
        self.proxy_image = proxy_image
        self.proxy_source = Path(proxy_source or Path(__file__).with_name("network_proxy.py"))
        self.external_network = external_network
        self.proxy_port = proxy_port
        self.allowed_ports = normalized_ports
        self.readiness_timeout_seconds = readiness_timeout_seconds
        self._run_command = command_runner or _default_command_runner
        self._monotonic = monotonic
        self._sleep = sleep

        token = (token_factory or (lambda: uuid.uuid4().hex))()
        token = "".join(character for character in token.lower() if character in "0123456789abcdef")
        if len(token) < 8:
            raise ValueError("token_factory must return at least eight hexadecimal characters")
        token = token[:16]
        self._session_token = token
        self._network_name = f"tasktrace-policy-{token}"
        self._proxy_name = f"tasktrace-policy-proxy-{token}"
        state_parent = Path(state_root) if state_root is not None else Path(tempfile.gettempdir())
        self._state_dir = state_parent / f".tasktrace-network-policy-state-{token}"
        state_dir_resolved = self._state_dir.resolve()
        if state_dir_resolved.is_relative_to(self.output_dir.resolve()):
            raise ValueError("network policy state_root must be outside the sandbox-mounted output_dir")
        for source in workload_mount_sources:
            mount_source = Path(source).expanduser().resolve(strict=False)
            if state_dir_resolved.is_relative_to(mount_source) or mount_source.is_relative_to(state_dir_resolved):
                raise ValueError(
                    "network policy host state must be disjoint from every " f"workload bind mount: {mount_source}"
                )
        self._ready_path = self._state_dir / "proxy.ready.json"
        self._decision_log_path = self._state_dir / "proxy-decisions.jsonl"
        self._pinned_proxy_source_path = self._state_dir / "network_proxy.py"
        self._archived_decision_log_path = self.output_dir / "network_proxy_decisions.jsonl"
        self._report_path = self.output_dir / "network_enforcement.json"

        self._binding: DockerNetworkBinding | None = None
        self._entered = False
        self._closed = False
        self._network_created = False
        self._proxy_created = False
        self._proxy_ever_started = False
        self._proxy_image_identity: dict[str, Any] | None = None
        self._proxy_container_id: str | None = None
        self._proxy_source_sha256: str | None = None
        self._proxy_internal_ip: str | None = None
        self._external_network_identity: dict[str, Any] | None = None
        self._decision_log_evidence: dict[str, Any] | None = None
        self._state = "configured"
        self._validation: dict[str, bool | None] = {
            "external_network_inspected": False,
            "internal_network_inspected": False,
            "proxy_container_inspected": False,
            "proxy_ready": False,
            "proxy_bound_to_internal_address": False,
            "state_disjoint_from_workload_mounts": True,
            "decision_log_validated": None,
            "proxy_healthy_at_cleanup": None,
        }
        self._cleanup: dict[str, Any] = {
            "attempted": False,
            "proxy_removed": False,
            "proxy_stopped": False,
            "network_removed": False,
            "decision_log_archived": False,
            "host_state_removed": False,
            "errors": [],
        }

    def __enter__(self) -> DockerNetworkPolicySession:
        if self._entered or self._closed:
            raise NetworkEnforcementError("network policy session is single-use")
        self._entered = True
        self.output_dir.mkdir(parents=True, exist_ok=True)

        effective = self.policy.effective_mode
        if effective is NetworkMode.PUBLIC:
            self._binding = DockerNetworkBinding(None, {})
            self._state = "active"
            self._write_report()
            return self
        if effective is NetworkMode.NO_NETWORK:
            self._binding = DockerNetworkBinding("none", {})
            self._state = "active"
            self._write_report()
            return self

        try:
            self._prepare_allowlist()
            self._state = "active"
            # A report failure is still a preparation failure: callers must
            # not receive an active sandbox binding without durable policy
            # provenance, and __enter__ failures do not invoke __exit__.
            self._write_report()
        except BaseException as exc:
            self._state = "preparation-failed"
            cleanup_errors = self._cleanup_resources()
            cleanup_errors.extend(self._archive_and_remove_state())
            self._cleanup["errors"] = cleanup_errors
            # __enter__ failures do not invoke __exit__. Mark this terminal so
            # an outer finally calling close() cannot erase failure evidence
            # or retry cleanup against resources already handled here.
            self._closed = True
            self._write_report(best_effort=True)
            message = f"unable to enforce Docker allowlist: {exc}"
            if cleanup_errors:
                message += "; cleanup also failed: " + "; ".join(cleanup_errors)
            raise NetworkEnforcementError(message) from exc
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: Any,
    ) -> bool:
        del exc_type, traceback
        cleanup_errors = self.close()
        if cleanup_errors:
            message = "Docker network-policy cleanup failed: " + "; ".join(cleanup_errors)
            if exc is not None:
                raise NetworkEnforcementError(message) from exc
            raise NetworkEnforcementError(message)
        return False

    @property
    def binding(self) -> DockerNetworkBinding:
        if self._binding is None:
            raise NetworkEnforcementError("network policy session is not active")
        return self._binding

    @property
    def docker_network(self) -> str | None:
        return self.binding.docker_network

    @property
    def sandbox_network(self) -> str | None:
        """Alias naming the network to attach to the measured sandbox."""
        return self.binding.docker_network

    @property
    def docker_run_args(self) -> tuple[str, ...]:
        return self.binding.docker_run_args

    @property
    def proxy_environment(self) -> dict[str, str]:
        return dict(self.binding.proxy_environment)

    @property
    def report_path(self) -> Path:
        return self._report_path

    def report_payload(self) -> dict[str, Any]:
        """Return serializable enforcement and lifecycle provenance."""
        binding = self._binding
        proxy_url = (
            f"http://{PROXY_ALIAS}:{self.proxy_port}" if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
        )
        return {
            "schema_version": NETWORK_ENFORCEMENT_SCHEMA_VERSION,
            "state": self._state,
            "policy": self.policy.to_report_payload(),
            "policy_comparability_sha256": self.policy.comparability_digest,
            "comparability": self.to_comparability_payload(),
            "comparability_sha256": self.comparability_digest,
            "allowed_ports": (list(self.allowed_ports) if self.policy.effective_mode is NetworkMode.ALLOWLIST else []),
            "backend": (
                "docker-forward-proxy-v1" if self.policy.effective_mode is NetworkMode.ALLOWLIST else "docker-native"
            ),
            "docker_network_override": (binding.docker_network if binding is not None else None),
            "external_network": (
                self.external_network if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
            ),
            "external_network_identity": (
                self._external_network_identity if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
            ),
            "ephemeral_internal_network": (
                self._network_name if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
            ),
            "proxy": {
                "container_name": (self._proxy_name if self.policy.effective_mode is NetworkMode.ALLOWLIST else None),
                "container_id": self._proxy_container_id,
                "stable_alias": (PROXY_ALIAS if self.policy.effective_mode is NetworkMode.ALLOWLIST else None),
                "url": proxy_url,
                "configured_image": (self.proxy_image if self.policy.effective_mode is NetworkMode.ALLOWLIST else None),
                "image_identity": self._proxy_image_identity,
                "source_sha256": self._proxy_source_sha256,
                "automatic_image_pull": False,
                "decision_log": (
                    str(self._archived_decision_log_path.resolve())
                    if self.policy.effective_mode is NetworkMode.ALLOWLIST
                    else None
                ),
                "decision_log_evidence": self._decision_log_evidence,
                "measurement_scope": (
                    "outside-sandbox-cgroup" if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
                ),
                "included_in_sandbox_cgroup_measurement": (
                    False if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
                ),
                "bound_internal_ip": (
                    self._proxy_internal_ip if self.policy.effective_mode is NetworkMode.ALLOWLIST else None
                ),
            },
            "validation": dict(self._validation),
            "cleanup": {
                **self._cleanup,
                "errors": list(self._cleanup["errors"]),
            },
        }

    def to_comparability_payload(self) -> dict[str, Any]:
        """Return enforced behavior, excluding ephemeral Docker resources."""
        allowlist = self.policy.effective_mode is NetworkMode.ALLOWLIST
        return {
            "schema_version": NETWORK_ENFORCEMENT_COMPARABILITY_SCHEMA_VERSION,
            "policy": self.policy.to_comparability_payload(),
            "backend": "docker-forward-proxy-v1" if allowlist else "docker-native",
            "allowed_ports": list(self.allowed_ports) if allowlist else [],
            "external_network": self.external_network if allowlist else None,
            "external_network_identity": (self._external_network_identity if allowlist else None),
            "proxy_source_sha256": self._proxy_source_sha256 if allowlist else None,
            "proxy_image_identity": self._proxy_image_identity if allowlist else None,
        }

    @property
    def comparability_digest(self) -> str:
        encoded = json.dumps(
            self.to_comparability_payload(),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        digest = hashlib.sha256()
        digest.update(b"tasktrace-docker-network-enforcement-comparability-v1\0")
        digest.update(encoded)
        return digest.hexdigest()

    def close(self) -> list[str]:
        """Idempotently remove sidecar then network and return cleanup errors."""
        if self._closed:
            return list(self._cleanup["errors"])
        self._closed = True
        health_errors: list[str] = []
        if self._proxy_created and self._state == "active":
            try:
                self._inspect_proxy_container()
            except NetworkEnforcementError as exc:
                self._validation["proxy_healthy_at_cleanup"] = False
                health_errors.append(f"proxy unhealthy before cleanup: {exc}")
            else:
                self._validation["proxy_healthy_at_cleanup"] = True
        cleanup_errors = health_errors + self._cleanup_resources()
        cleanup_errors.extend(self._archive_and_remove_state())
        self._cleanup["errors"] = cleanup_errors
        self._state = "closed" if not cleanup_errors else "cleanup-failed"
        self._write_report(best_effort=bool(cleanup_errors))
        return cleanup_errors

    def _prepare_allowlist(self) -> None:
        if not self.proxy_source.is_file():
            raise NetworkEnforcementError(f"network proxy source does not exist: {self.proxy_source}")
        proxy_source_bytes = self.proxy_source.read_bytes()
        self._proxy_source_sha256 = hashlib.sha256(proxy_source_bytes).hexdigest()
        self._prepare_state_files(proxy_source_bytes)
        self._proxy_image_identity = self._inspect_image()
        self._inspect_external_network()

        self._checked(
            "create internal policy network",
            [
                self.docker,
                "network",
                "create",
                "--driver",
                "bridge",
                "--internal",
                "--ipv6=false",
                "--opt",
                f"{_INHIBIT_IPV4_OPTION}=true",
                "--label",
                "io.tasktrace.component=network-policy",
                "--label",
                f"io.tasktrace.policy-digest={self.policy.comparability_digest}",
                self._network_name,
            ],
        )
        self._network_created = True
        self._inspect_internal_network()

        completed = self._checked("start allowlist proxy", self._proxy_run_command())
        self._proxy_created = True
        self._proxy_ever_started = True
        self._proxy_container_id = completed.stdout.strip() or None
        self._checked(
            "connect allowlist proxy to external network",
            [
                self.docker,
                "network",
                "connect",
                self.external_network,
                self._proxy_name,
            ],
        )
        self._inspect_proxy_container()
        self._wait_until_ready()
        # A ready-file proves the listener initialized; this second inspect
        # proves it did not exit immediately after publishing readiness.
        self._inspect_proxy_container()

        proxy_url = f"http://{PROXY_ALIAS}:{self.proxy_port}"
        self._binding = DockerNetworkBinding(
            self._network_name,
            {
                "HTTP_PROXY": proxy_url,
                "HTTPS_PROXY": proxy_url,
                "http_proxy": proxy_url,
                "https_proxy": proxy_url,
                "NO_PROXY": "localhost,127.0.0.1,::1",
                "no_proxy": "localhost,127.0.0.1,::1",
            },
        )

    def _prepare_state_files(self, proxy_source_bytes: bytes) -> None:
        self._state_dir.mkdir(parents=True, exist_ok=False)
        self._pinned_proxy_source_path.write_bytes(proxy_source_bytes)
        os.chmod(self._pinned_proxy_source_path, 0o444)
        for path in (self._ready_path, self._decision_log_path):
            path.touch(exist_ok=False)
            os.chmod(path, 0o666)

    def _inspect_image(self) -> dict[str, Any]:
        completed = self._checked(
            "inspect local proxy image",
            [self.docker, "image", "inspect", self.proxy_image],
        )
        payload = self._json_list(completed.stdout, "proxy image inspection")
        item = payload[0]
        image_id = item.get("Id")
        if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
            raise NetworkEnforcementError("proxy image inspection did not return an immutable sha256 image ID")
        repo_digests = item.get("RepoDigests") or []
        if not isinstance(repo_digests, list) or not all(isinstance(value, str) for value in repo_digests):
            raise NetworkEnforcementError("proxy image RepoDigests are malformed")
        return {
            "id": image_id,
            "repo_digests": sorted(repo_digests),
            "os": item.get("Os"),
            "architecture": item.get("Architecture"),
        }

    def _inspect_external_network(self) -> None:
        completed = self._checked(
            "inspect external proxy network",
            [self.docker, "network", "inspect", self.external_network],
        )
        item = self._json_list(completed.stdout, "external network inspection")[0]
        network_id = item.get("Id")
        driver = item.get("Driver")
        ipam = item.get("IPAM")
        options = item.get("Options") or {}
        if (
            item.get("Name") != self.external_network
            or item.get("Internal") is not False
            or not isinstance(network_id, str)
            or not network_id
            or not isinstance(driver, str)
            or not driver
            or not isinstance(ipam, dict)
            or not isinstance(options, dict)
        ):
            raise NetworkEnforcementError(f"proxy external network {self.external_network!r} is malformed or internal")
        configuration = {
            "name": item["Name"],
            "id": network_id,
            "driver": driver,
            "scope": item.get("Scope"),
            "internal": item["Internal"],
            "attachable": item.get("Attachable"),
            "ingress": item.get("Ingress"),
            "enable_ipv4": item.get("EnableIPv4"),
            "enable_ipv6": item.get("EnableIPv6"),
            "ipam": ipam,
            "options": options,
        }
        encoded = json.dumps(
            configuration,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        self._external_network_identity = {
            **configuration,
            "configuration_sha256": hashlib.sha256(encoded).hexdigest(),
        }
        self._validation["external_network_inspected"] = True

    def _inspect_internal_network(self) -> None:
        completed = self._checked(
            "inspect internal policy network",
            [self.docker, "network", "inspect", self._network_name],
        )
        item = self._json_list(completed.stdout, "internal network inspection")[0]
        options = item.get("Options")
        labels = item.get("Labels")
        if (
            item.get("Name") != self._network_name
            or item.get("Driver") != "bridge"
            or item.get("Internal") is not True
            or item.get("EnableIPv6") is not False
            or not isinstance(options, dict)
            or str(options.get(_INHIBIT_IPV4_OPTION, "")).lower() != "true"
            or not isinstance(labels, dict)
            or labels.get("io.tasktrace.component") != "network-policy"
            or labels.get("io.tasktrace.policy-digest") != self.policy.comparability_digest
        ):
            raise NetworkEnforcementError("created Docker network did not inspect as the required internal bridge")
        self._validation["internal_network_inspected"] = True

    def _inspect_proxy_container(self) -> None:
        completed = self._checked(
            "inspect allowlist proxy container",
            [self.docker, "container", "inspect", self._proxy_name],
        )
        item = self._json_list(completed.stdout, "proxy container inspection")[0]
        state = item.get("State")
        config = item.get("Config")
        host_config = item.get("HostConfig")
        network_settings = item.get("NetworkSettings")
        networks = network_settings.get("Networks") if isinstance(network_settings, dict) else None
        internal_attachment = networks.get(self._network_name) if isinstance(networks, dict) else None
        internal_ip = internal_attachment.get("IPAddress") if isinstance(internal_attachment, dict) else None
        aliases = internal_attachment.get("Aliases") if isinstance(internal_attachment, dict) else None
        security_options = (host_config.get("SecurityOpt") if isinstance(host_config, dict) else None) or []
        cap_drop = (host_config.get("CapDrop") if isinstance(host_config, dict) else None) or []
        labels = config.get("Labels") if isinstance(config, dict) else None
        entrypoint = config.get("Entrypoint") if isinstance(config, dict) else None
        container_command = config.get("Cmd") if isinstance(config, dict) else None
        restart_policy = host_config.get("RestartPolicy") if isinstance(host_config, dict) else None
        sysctls = host_config.get("Sysctls") if isinstance(host_config, dict) else None
        if (
            not isinstance(state, dict)
            or state.get("Running") is not True
            or not isinstance(config, dict)
            or config.get("Image") != self._proxy_image_identity["id"]
            or config.get("User") != "65534:65534"
            or entrypoint != ["python"]
            or container_command != self._proxy_process_arguments()
            or not isinstance(labels, dict)
            or labels.get("io.tasktrace.component") != "network-policy-proxy"
            or labels.get("io.tasktrace.policy-digest") != self.policy.comparability_digest
            or not isinstance(host_config, dict)
            or host_config.get("ReadonlyRootfs") is not True
            or host_config.get("PidsLimit") != 64
            or host_config.get("Memory") != 128 * 1024 * 1024
            or host_config.get("NanoCpus") != 1_000_000_000
            or not isinstance(restart_policy, dict)
            or restart_policy.get("Name") != "no"
            or not isinstance(sysctls, dict)
            or str(sysctls.get("net.ipv4.ip_forward")) != "0"
            or str(sysctls.get("net.ipv6.conf.all.forwarding")) != "0"
            or "ALL" not in cap_drop
            or not any("no-new-privileges" in str(value) for value in security_options)
            or not isinstance(networks, dict)
            or self.external_network not in networks
            or self._network_name not in networks
            or not isinstance(aliases, list)
            or PROXY_ALIAS not in aliases
            or not isinstance(internal_ip, str)
            or not internal_ip
        ):
            raise NetworkEnforcementError("allowlist proxy inspection did not match required running/hardened topology")
        self._validation["proxy_container_inspected"] = True
        if self._proxy_internal_ip is not None and self._proxy_internal_ip != internal_ip:
            raise NetworkEnforcementError("allowlist proxy internal address changed during the policy session")
        self._proxy_internal_ip = internal_ip

    def _proxy_run_command(self) -> list[str]:
        assert self._proxy_image_identity is not None
        state_mount = f"type=bind,src={self._state_dir.resolve()}," "dst=/var/run/tasktrace-policy"
        source_mount = (
            f"type=bind,src={self._pinned_proxy_source_path.resolve()}," "dst=/opt/tasktrace/network_proxy.py,readonly"
        )
        command = [
            self.docker,
            "run",
            "--detach",
            "--pull=never",
            "--name",
            self._proxy_name,
            "--network",
            self._network_name,
            "--network-alias",
            PROXY_ALIAS,
            "--label",
            "io.tasktrace.component=network-policy-proxy",
            "--label",
            f"io.tasktrace.policy-digest={self.policy.comparability_digest}",
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
            source_mount,
            "--mount",
            state_mount,
            "--entrypoint",
            "python",
            self._proxy_image_identity["id"],
            *self._proxy_process_arguments(),
        ]
        return command

    def _proxy_process_arguments(self) -> list[str]:
        """Return the exact trusted proxy argv, independent of image metadata."""
        command = [
            "-B",
            "/opt/tasktrace/network_proxy.py",
            "--listen-host",
            PROXY_ALIAS,
            "--listen-port",
            str(self.proxy_port),
            "--ready-file",
            "/var/run/tasktrace-policy/proxy.ready.json",
            "--decision-log",
            "/var/run/tasktrace-policy/proxy-decisions.jsonl",
        ]
        for host in self.policy.allowed_hosts:
            command.extend(("--allow-host", host))
        for port in self.allowed_ports:
            command.extend(("--allow-port", str(port)))
        return command

    def _wait_until_ready(self) -> None:
        deadline = self._monotonic() + self.readiness_timeout_seconds
        while True:
            try:
                raw = self._ready_path.read_text(encoding="utf-8")
            except OSError:
                raw = ""
            if raw.strip():
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError as exc:
                    raise NetworkEnforcementError("allowlist proxy wrote malformed readiness JSON") from exc
                if (
                    isinstance(payload, dict)
                    and payload.get("ready") is True
                    and self._proxy_internal_ip is not None
                    and payload.get("listen_host") == self._proxy_internal_ip
                    and payload.get("listen_port") == self.proxy_port
                    and payload.get("allowed_ports") == list(self.allowed_ports)
                ):
                    self._validation["proxy_ready"] = True
                    self._validation["proxy_bound_to_internal_address"] = True
                    return
                raise NetworkEnforcementError(
                    "allowlist proxy readiness payload did not prove the " "configured listener and ports"
                )
            if self._monotonic() >= deadline:
                raise NetworkEnforcementError("timed out waiting for allowlist proxy readiness")
            self._sleep(min(0.05, max(0.0, deadline - self._monotonic())))

    def _cleanup_resources(self) -> list[str]:
        errors: list[str] = []
        if not self._proxy_created and not self._network_created:
            return errors
        self._cleanup["attempted"] = True

        if self._proxy_created:
            stopped = self._invoke(
                [
                    self.docker,
                    "container",
                    "stop",
                    "--time",
                    "5",
                    self._proxy_name,
                ]
            )
            if stopped.returncode == 0 or "is not running" in stopped.stderr:
                self._cleanup["proxy_stopped"] = True
            else:
                errors.append("stop proxy container: " + self._failure_detail(stopped))
            completed = self._invoke([self.docker, "container", "rm", "--force", self._proxy_name])
            if completed.returncode == 0 or "No such container" in completed.stderr:
                self._cleanup["proxy_removed"] = True
                self._proxy_created = False
            else:
                errors.append("remove proxy container: " + self._failure_detail(completed))
        if self._network_created:
            completed = self._invoke([self.docker, "network", "rm", self._network_name])
            if completed.returncode == 0 or "No such network" in completed.stderr:
                self._cleanup["network_removed"] = True
                self._network_created = False
            else:
                errors.append("remove internal network: " + self._failure_detail(completed))
        self._cleanup["errors"] = errors
        return errors

    def _archive_and_remove_state(self) -> list[str]:
        """Archive proxy decisions after shutdown and remove host-only state."""
        if not self._state_dir.exists():
            return []
        errors: list[str] = []
        archive_temporary = self.output_dir.parent / (f".tasktrace-network-decisions-{self._session_token}.tmp")
        try:
            if self._proxy_ever_started:
                try:
                    self._decision_log_evidence = self._validate_decision_log(self._decision_log_path)
                except (OSError, NetworkEnforcementError) as exc:
                    self._validation["decision_log_validated"] = False
                    errors.append("validate proxy decision log: " f"{type(exc).__name__}: {exc}")
                else:
                    self._validation["decision_log_validated"] = True
            archive_temporary.unlink(missing_ok=True)
            shutil.copyfile(self._decision_log_path, archive_temporary)
            os.chmod(archive_temporary, 0o644)
            os.replace(archive_temporary, self._archived_decision_log_path)
            self._cleanup["decision_log_archived"] = True
            if self._decision_log_evidence is not None:
                archived_sha256 = hashlib.sha256(self._archived_decision_log_path.read_bytes()).hexdigest()
                if archived_sha256 != self._decision_log_evidence["sha256"]:
                    raise NetworkEnforcementError("archived proxy decision log digest changed during copy")
        except (OSError, NetworkEnforcementError) as exc:
            self._cleanup["decision_log_archived"] = False
            errors.append("archive proxy decision log: " f"{type(exc).__name__}: {exc}")
        finally:
            try:
                archive_temporary.unlink(missing_ok=True)
            except OSError:
                pass
        try:
            shutil.rmtree(self._state_dir)
            self._cleanup["host_state_removed"] = True
        except OSError as exc:
            errors.append(f"remove host-only proxy state: {type(exc).__name__}: {exc}")
        return errors

    def _validate_decision_log(self, path: Path) -> dict[str, Any]:
        """Validate complete proxy JSONL and return secret-safe integrity facts."""
        digest = hashlib.sha256()
        byte_count = 0
        record_count = 0
        event_counts: dict[str, int] = {}
        allowed_destination_ips: set[str] = set()
        with path.open("rb") as stream:
            for line in stream:
                byte_count += len(line)
                record_count += 1
                if byte_count > _MAX_DECISION_LOG_BYTES:
                    raise NetworkEnforcementError("proxy decision log exceeds the 64 MiB evidence bound")
                if record_count > _MAX_DECISION_LOG_RECORDS:
                    raise NetworkEnforcementError("proxy decision log exceeds the record-count evidence bound")
                if not line.endswith(b"\n"):
                    raise NetworkEnforcementError("proxy decision log contains an incomplete terminal record")
                digest.update(line)
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise NetworkEnforcementError("proxy decision log contains malformed JSON") from exc
                if (
                    not isinstance(record, dict)
                    or record.get("schema_version") != _PROXY_LOG_SCHEMA_VERSION
                    or not isinstance(record.get("timestamp"), str)
                    or not isinstance(record.get("event"), str)
                ):
                    raise NetworkEnforcementError("proxy decision log record does not match its schema")
                event = record["event"]
                if event not in {
                    "proxy_ready",
                    "proxy_stopped",
                    "readiness_write_failed",
                    "request_decision",
                    "request_error",
                }:
                    raise NetworkEnforcementError(f"proxy decision log contains unknown event {event!r}")
                event_counts[event] = event_counts.get(event, 0) + 1
                if event == "request_decision" and record.get("decision") == "allowed":
                    host = record.get("host")
                    port = record.get("port")
                    destination_ip = record.get("destination_ip")
                    try:
                        address = ipaddress.ip_address(destination_ip)
                    except (TypeError, ValueError) as exc:
                        raise NetworkEnforcementError("allowed proxy decision lacks a numeric destination") from exc
                    if host not in self.policy.allowed_hosts or port not in self.allowed_ports or not address.is_global:
                        raise NetworkEnforcementError("allowed proxy decision violates the enforced policy")
                    allowed_destination_ips.add(str(address))
        if event_counts.get("proxy_ready") != 1:
            raise NetworkEnforcementError("proxy decision log must contain exactly one ready event")
        if event_counts.get("proxy_stopped") != 1:
            raise NetworkEnforcementError("proxy decision log must contain exactly one graceful-stop event")
        return {
            "schema_version": "tasktrace-network-proxy-decision-evidence.v1",
            "valid": True,
            "sha256": digest.hexdigest(),
            "bytes": byte_count,
            "records": record_count,
            "event_counts": dict(sorted(event_counts.items())),
            "allowed_destination_ips": sorted(allowed_destination_ips),
        }

    def _checked(self, action: str, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        completed = self._invoke(command)
        if completed.returncode != 0:
            raise NetworkEnforcementError(f"{action} failed: {self._failure_detail(completed)}")
        return completed

    def _invoke(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        try:
            return self._run_command(tuple(command))
        except (OSError, subprocess.SubprocessError) as exc:
            return subprocess.CompletedProcess(list(command), 127, "", f"{type(exc).__name__}: {exc}")

    @staticmethod
    def _failure_detail(completed: subprocess.CompletedProcess[str]) -> str:
        detail = (completed.stderr or completed.stdout or "no diagnostic output").strip()
        return detail[:2000]

    @staticmethod
    def _json_list(raw: str, description: str) -> list[dict[str, Any]]:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise NetworkEnforcementError(f"invalid {description} JSON") from exc
        if not isinstance(payload, list) or len(payload) != 1 or not isinstance(payload[0], dict):
            raise NetworkEnforcementError(f"{description} must contain exactly one object")
        return payload

    def _write_report(self, *, best_effort: bool = False) -> None:
        # The deterministic destination lives in the sandbox-mounted results
        # directory, but the temporary is a host-only sibling.  ``replace``
        # replaces (rather than follows) a malicious destination symlink.
        temporary = self.output_dir.parent / (f".tasktrace-network-enforcement-{self._session_token}.tmp")
        try:
            temporary.unlink(missing_ok=True)
            temporary.write_text(
                json.dumps(
                    self.report_payload(),
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                )
                + "\n",
                encoding="utf-8",
            )
            temporary.replace(self._report_path)
        except OSError:
            if not best_effort:
                raise
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass
