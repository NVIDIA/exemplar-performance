# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Sequence

import pytest
from offline_replayer.measurement.network_enforcement import (
    PROXY_ALIAS,
    DockerNetworkPolicySession,
    NetworkEnforcementError,
)
from offline_replayer.measurement.network_policy import NetworkPolicy

_IMAGE_ID = "sha256:" + "a" * 64


class FakeDocker:
    def __init__(
        self,
        *,
        ready: bool = True,
        fail_action: tuple[str, ...] | None = None,
        fail_cleanup: bool = False,
        die_on_cleanup: bool = False,
    ) -> None:
        self.commands: list[tuple[str, ...]] = []
        self.ready = ready
        self.fail_action = fail_action
        self.fail_cleanup = fail_cleanup
        self.die_on_cleanup = die_on_cleanup
        self.container_inspects = 0
        self.network_name = ""
        self.proxy_name = ""
        self.policy_digest = ""
        self.proxy_entrypoint: list[str] = []
        self.proxy_command: list[str] = []
        self.state_dir: Path | None = None

    def __call__(self, command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        command = tuple(command)
        self.commands.append(command)
        if self.fail_action and command[1 : 1 + len(self.fail_action)] == self.fail_action:
            return subprocess.CompletedProcess(command, 1, "", "injected failure")

        if command[1:3] == ("image", "inspect"):
            return self._ok(
                command,
                json.dumps(
                    [
                        {
                            "Id": _IMAGE_ID,
                            "RepoDigests": ["python@sha256:" + "b" * 64],
                            "Os": "linux",
                            "Architecture": "amd64",
                        }
                    ]
                ),
            )
        if command[1:3] == ("network", "create"):
            self.network_name = command[-1]
            labels = [command[index + 1] for index, value in enumerate(command) if value == "--label"]
            self.policy_digest = next(
                label.split("=", 1)[1] for label in labels if label.startswith("io.tasktrace.policy-digest=")
            )
            return self._ok(command, "network-id\n")
        if command[1:3] == ("network", "inspect"):
            name = command[-1]
            if name == "fixture-external":
                return self._ok(
                    command,
                    json.dumps(
                        [
                            {
                                "Name": name,
                                "Id": "external-network-id",
                                "Driver": "bridge",
                                "Scope": "local",
                                "Internal": False,
                                "Attachable": False,
                                "Ingress": False,
                                "EnableIPv4": True,
                                "EnableIPv6": False,
                                "IPAM": {
                                    "Driver": "default",
                                    "Config": [
                                        {
                                            "Subnet": "172.29.0.0/16",
                                            "Gateway": "172.29.0.1",
                                        }
                                    ],
                                },
                                "Options": {},
                            }
                        ]
                    ),
                )
            return self._ok(
                command,
                json.dumps(
                    [
                        {
                            "Name": name,
                            "Driver": "bridge",
                            "Internal": True,
                            "EnableIPv6": False,
                            "Options": {"com.docker.network.bridge.inhibit_ipv4": "true"},
                            "Labels": {
                                "io.tasktrace.component": "network-policy",
                                "io.tasktrace.policy-digest": self.policy_digest,
                            },
                        }
                    ]
                ),
            )
        if command[1] == "run":
            self.proxy_name = command[command.index("--name") + 1]
            self.proxy_entrypoint = [command[command.index("--entrypoint") + 1]]
            image_index = command.index(_IMAGE_ID)
            self.proxy_command = list(command[image_index + 1 :])
            allowed_ports = [int(command[index + 1]) for index, value in enumerate(command) if value == "--allow-port"]
            labels = [command[index + 1] for index, value in enumerate(command) if value == "--label"]
            self.policy_digest = next(
                label.split("=", 1)[1] for label in labels if label.startswith("io.tasktrace.policy-digest=")
            )
            mount = next(
                command[index + 1]
                for index, value in enumerate(command)
                if value == "--mount" and "dst=/var/run/tasktrace-policy" in command[index + 1]
            )
            source = next(part.removeprefix("src=") for part in mount.split(",") if part.startswith("src="))
            self.state_dir = Path(source)
            if self.ready:
                (Path(source) / "proxy.ready.json").write_text(
                    json.dumps(
                        {
                            "ready": True,
                            "listen_host": "172.30.0.2",
                            "listen_port": 3128,
                            "allowed_ports": allowed_ports,
                        }
                    ),
                    encoding="utf-8",
                )
                self._append_proxy_event("proxy_ready")
            return self._ok(command, "proxy-container-id\n")
        if command[1:3] == ("network", "connect"):
            return self._ok(command)
        if command[1:3] == ("container", "inspect"):
            self.container_inspects += 1
            running = not (self.die_on_cleanup and self.container_inspects >= 3)
            return self._ok(command, json.dumps([self._container_inspect(running=running)]))
        if command[1:3] == ("container", "stop"):
            if self.die_on_cleanup and self.container_inspects >= 3:
                return subprocess.CompletedProcess(command, 1, "", "container is not running")
            self._append_proxy_event("proxy_stopped")
            return self._ok(command, self.proxy_name + "\n")
        if command[1:4] == ("container", "rm", "--force"):
            if self.fail_cleanup:
                return subprocess.CompletedProcess(command, 1, "", "container busy")
            return self._ok(command)
        if command[1:3] == ("network", "rm"):
            return self._ok(command)
        raise AssertionError(f"unexpected Docker command: {command}")

    def _append_proxy_event(self, event: str) -> None:
        assert self.state_dir is not None
        record = {
            "schema_version": "tasktrace-network-proxy.v1",
            "timestamp": "2026-07-01T00:00:00Z",
            "event": event,
        }
        with (self.state_dir / "proxy-decisions.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")

    @staticmethod
    def _ok(command: Sequence[str], stdout: str = "") -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout, "")

    def _container_inspect(self, *, running: bool) -> dict[str, Any]:
        return {
            "State": {"Running": running},
            "Config": {
                "Image": _IMAGE_ID,
                "User": "65534:65534",
                "Entrypoint": self.proxy_entrypoint,
                "Cmd": self.proxy_command,
                "Labels": {
                    "io.tasktrace.component": "network-policy-proxy",
                    "io.tasktrace.policy-digest": self.policy_digest,
                },
            },
            "HostConfig": {
                "ReadonlyRootfs": True,
                "CapDrop": ["ALL"],
                "SecurityOpt": ["no-new-privileges"],
                "PidsLimit": 64,
                "Memory": 128 * 1024 * 1024,
                "NanoCpus": 1_000_000_000,
                "RestartPolicy": {"Name": "no"},
                "Sysctls": {
                    "net.ipv4.ip_forward": "0",
                    "net.ipv6.conf.all.forwarding": "0",
                },
            },
            "NetworkSettings": {
                "Networks": {
                    self.network_name: {
                        "Aliases": [PROXY_ALIAS],
                        "IPAddress": "172.30.0.2",
                    },
                    "fixture-external": {"Aliases": [self.proxy_name]},
                }
            },
        }


def _proxy_source(tmp_path: Path) -> Path:
    source = tmp_path / "network_proxy.py"
    source.write_text("# test proxy\n", encoding="utf-8")
    return source


@pytest.mark.parametrize(
    ("policy", "expected_network"),
    [
        (NetworkPolicy("public"), None),
        (NetworkPolicy("no-network"), "none"),
        (NetworkPolicy("allowlist"), "none"),
    ],
)
def test_native_modes_do_not_create_infrastructure(
    tmp_path: Path, policy: NetworkPolicy, expected_network: str | None
) -> None:
    docker = FakeDocker()

    with DockerNetworkPolicySession(
        policy,
        tmp_path / "result",
        command_runner=docker,
        token_factory=lambda: "12345678",
    ) as session:
        assert session.docker_network == expected_network
        assert session.docker_run_args == (() if expected_network is None else ("--network", expected_network))
        assert session.proxy_environment == {}

    assert docker.commands == []
    report = json.loads(session.report_path.read_text(encoding="utf-8"))
    assert report["state"] == "closed"
    assert report["policy"]["requested_mode"] == policy.mode.value
    assert report["policy"]["effective_mode"] == policy.effective_mode.value


def test_allowlist_builds_hardened_isolated_topology_and_reports_it(
    tmp_path: Path,
) -> None:
    docker = FakeDocker()
    proxy_source = _proxy_source(tmp_path)
    policy = NetworkPolicy("allowlist", ("PyPI.org", "files.pythonhosted.org"))

    with DockerNetworkPolicySession(
        policy,
        tmp_path / "result",
        command_runner=docker,
        proxy_source=proxy_source,
        external_network="fixture-external",
        allowed_ports=(443, 80, 443),
        token_factory=lambda: "abcdef1234567890",
    ) as session:
        host_state_dir = session._state_dir
        assert host_state_dir.exists()
        assert session._pinned_proxy_source_path.read_text() == "# test proxy\n"
        assert session._pinned_proxy_source_path.stat().st_mode & 0o777 == 0o444
        assert not host_state_dir.is_relative_to(tmp_path / "result")
        assert not (tmp_path / "result" / "proxy.ready.json").exists()
        assert session.docker_network == "tasktrace-policy-abcdef1234567890"
        assert session.docker_run_args == (
            "--network",
            "tasktrace-policy-abcdef1234567890",
        )
        assert session.proxy_environment["HTTPS_PROXY"] == (f"http://{PROXY_ALIAS}:3128")
        active_report = session.report_payload()
        assert active_report["state"] == "active"
        assert active_report["external_network"] == "fixture-external"
        assert active_report["allowed_ports"] == [80, 443]
        assert active_report["comparability"]["allowed_ports"] == [80, 443]
        assert active_report["comparability"]["external_network_identity"]["id"] == "external-network-id"
        assert "ephemeral_internal_network" not in active_report["comparability"]
        assert active_report["proxy"]["image_identity"]["id"] == _IMAGE_ID
        assert active_report["proxy"]["automatic_image_pull"] is False
        assert active_report["proxy"]["measurement_scope"] == ("outside-sandbox-cgroup")
        assert active_report["proxy"]["included_in_sandbox_cgroup_measurement"] is False
        assert active_report["validation"]["external_network_inspected"] is True
        assert active_report["external_network_identity"]["driver"] == "bridge"
        assert len(active_report["external_network_identity"]["configuration_sha256"]) == 64
        assert active_report["validation"]["internal_network_inspected"] is True
        assert active_report["validation"]["proxy_container_inspected"] is True
        assert active_report["validation"]["proxy_ready"] is True
        assert active_report["validation"]["proxy_bound_to_internal_address"] is True
        assert active_report["proxy"]["bound_internal_ip"] == "172.30.0.2"
        assert active_report["validation"]["proxy_healthy_at_cleanup"] is None

    create = next(command for command in docker.commands if command[1:3] == ("network", "create"))
    assert "--internal" in create
    assert "--ipv6=false" in create
    assert "com.docker.network.bridge.inhibit_ipv4=true" in create
    create_labels = [create[index + 1] for index, value in enumerate(create) if value == "--label"]
    assert create_labels == [
        "io.tasktrace.component=network-policy",
        f"io.tasktrace.policy-digest={policy.comparability_digest}",
    ]

    run = next(command for command in docker.commands if command[1] == "run")
    assert "--pull=never" in run
    assert run[run.index("--entrypoint") + 1] == "python"
    assert "--read-only" in run
    assert "--cap-drop=ALL" in run
    assert "--security-opt=no-new-privileges" in run
    assert "--restart=no" in run
    assert run[run.index("net.ipv4.ip_forward=0") - 1] == "--sysctl"
    assert run[run.index("net.ipv6.conf.all.forwarding=0") - 1] == "--sysctl"
    assert run[run.index("--user") + 1] == "65534:65534"
    assert run[run.index("--network") + 1] == ("tasktrace-policy-abcdef1234567890")
    assert run[run.index("--network-alias") + 1] == PROXY_ALIAS
    assert run[run.index("--listen-host") + 1] == PROXY_ALIAS
    source_mount = next(
        run[index + 1]
        for index, value in enumerate(run)
        if value == "--mount" and "dst=/opt/tasktrace/network_proxy.py" in run[index + 1]
    )
    assert f"src={host_state_dir / 'network_proxy.py'}" in source_mount
    assert _IMAGE_ID in run
    assert "-p" not in run and "--publish" not in run
    assert run.count("--allow-host") == 2
    assert run.count("--allow-port") == 2
    assert run[run.index("--allow-port") + 1] == "80"
    second_port = run.index("--allow-port", run.index("--allow-port") + 1)
    assert run[second_port + 1] == "443"

    connect = next(command for command in docker.commands if command[1:3] == ("network", "connect"))
    assert connect[3:] == (
        "fixture-external",
        "tasktrace-policy-proxy-abcdef1234567890",
    )
    assert docker.commands[-3][1:3] == ("container", "stop")
    assert docker.commands[-2][1:4] == ("container", "rm", "--force")
    assert docker.commands[-1][1:3] == ("network", "rm")
    report = json.loads(session.report_path.read_text(encoding="utf-8"))
    assert report["state"] == "closed"
    assert report["cleanup"]["proxy_removed"] is True
    assert report["cleanup"]["proxy_stopped"] is True
    assert report["cleanup"]["network_removed"] is True
    assert report["cleanup"]["decision_log_archived"] is True
    assert report["cleanup"]["host_state_removed"] is True
    assert report["validation"]["proxy_healthy_at_cleanup"] is True
    assert report["validation"]["decision_log_validated"] is True
    assert report["proxy"]["decision_log_evidence"]["records"] == 2
    assert len(report["proxy"]["decision_log_evidence"]["sha256"]) == 64
    assert not host_state_dir.exists()
    assert (tmp_path / "result" / "network_proxy_decisions.jsonl").is_file()


def test_preparation_failure_removes_container_and_network(tmp_path: Path) -> None:
    docker = FakeDocker(fail_action=("network", "connect"))
    session = DockerNetworkPolicySession(
        NetworkPolicy("allowlist", ("example.com",)),
        tmp_path / "result",
        command_runner=docker,
        proxy_source=_proxy_source(tmp_path),
        external_network="fixture-external",
        token_factory=lambda: "11112222",
    )

    with pytest.raises(NetworkEnforcementError, match="injected failure"):
        session.__enter__()

    assert any(command[1:4] == ("container", "rm", "--force") for command in docker.commands)
    assert any(command[1:3] == ("network", "rm") for command in docker.commands)
    report = json.loads((tmp_path / "result" / "network_enforcement.json").read_text())
    assert report["state"] == "preparation-failed"
    assert report["cleanup"]["proxy_removed"] is True
    assert report["cleanup"]["network_removed"] is True
    assert report["cleanup"]["host_state_removed"] is True
    assert session.close() == []
    assert json.loads(session.report_path.read_text())["state"] == "preparation-failed"


def test_readiness_timeout_fails_closed(tmp_path: Path) -> None:
    docker = FakeDocker(ready=False)

    with pytest.raises(NetworkEnforcementError, match="readiness"):
        with DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("example.com",)),
            tmp_path / "result",
            command_runner=docker,
            proxy_source=_proxy_source(tmp_path),
            external_network="fixture-external",
            readiness_timeout_seconds=0,
            token_factory=lambda: "33334444",
        ):
            raise AssertionError("session must not become active")

    assert any(command[1:4] == ("container", "rm", "--force") for command in docker.commands)
    assert any(command[1:3] == ("network", "rm") for command in docker.commands)


def test_cleanup_failure_is_not_silently_accepted(tmp_path: Path) -> None:
    docker = FakeDocker(fail_cleanup=True)

    with pytest.raises(NetworkEnforcementError, match="cleanup failed"):
        with DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("example.com",)),
            tmp_path / "result",
            command_runner=docker,
            proxy_source=_proxy_source(tmp_path),
            external_network="fixture-external",
            token_factory=lambda: "55556666",
        ):
            pass

    report = json.loads((tmp_path / "result" / "network_enforcement.json").read_text())
    assert report["state"] == "cleanup-failed"
    assert report["cleanup"]["errors"]
    assert report["cleanup"]["host_state_removed"] is True


def test_proxy_death_during_workload_invalidates_session_but_still_cleans_up(
    tmp_path: Path,
) -> None:
    docker = FakeDocker(die_on_cleanup=True)

    with pytest.raises(NetworkEnforcementError, match="proxy unhealthy"):
        with DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("example.com",)),
            tmp_path / "result",
            command_runner=docker,
            proxy_source=_proxy_source(tmp_path),
            external_network="fixture-external",
            token_factory=lambda: "77778888",
        ):
            pass

    report = json.loads((tmp_path / "result" / "network_enforcement.json").read_text())
    assert report["state"] == "cleanup-failed"
    assert report["validation"]["proxy_healthy_at_cleanup"] is False
    assert report["cleanup"]["proxy_removed"] is True
    assert report["cleanup"]["network_removed"] is True


def test_state_root_must_not_be_inside_sandbox_output(tmp_path: Path) -> None:
    output_dir = tmp_path / "result"

    with pytest.raises(ValueError, match="outside"):
        DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("example.com",)),
            output_dir,
            state_root=output_dir / "live-state",
            proxy_source=_proxy_source(tmp_path),
            token_factory=lambda: "99990000",
        )


def test_host_state_must_be_disjoint_from_workload_mounts(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="disjoint"):
        DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("example.com",)),
            tmp_path / "result",
            state_root=tmp_path / "host-state",
            workload_mount_sources=(str(tmp_path),),
            proxy_source=_proxy_source(tmp_path),
            token_factory=lambda: "12121212",
        )


def test_malformed_decision_log_invalidates_cleanup_but_is_retained(
    tmp_path: Path,
) -> None:
    docker = FakeDocker()
    session = DockerNetworkPolicySession(
        NetworkPolicy("allowlist", ("example.com",)),
        tmp_path / "result",
        command_runner=docker,
        proxy_source=_proxy_source(tmp_path),
        external_network="fixture-external",
        token_factory=lambda: "34343434",
    )
    session.__enter__()
    session._decision_log_path.write_text("{malformed\n", encoding="utf-8")

    errors = session.close()

    assert any("validate proxy decision log" in error for error in errors)
    report = json.loads(session.report_path.read_text())
    assert report["state"] == "cleanup-failed"
    assert report["validation"]["decision_log_validated"] is False
    assert report["cleanup"]["decision_log_archived"] is True
    assert (tmp_path / "result" / "network_proxy_decisions.jsonl").read_text() == (
        "{malformed\n"
        + json.dumps(
            {
                "event": "proxy_stopped",
                "schema_version": "tasktrace-network-proxy.v1",
                "timestamp": "2026-07-01T00:00:00Z",
            },
            sort_keys=True,
        )
        + "\n"
    )


@pytest.mark.parametrize("ports", [(), (0,), (65536,), (True,), ("80",)])
def test_allowlist_rejects_empty_or_invalid_allowed_ports(tmp_path: Path, ports: tuple[object, ...]) -> None:
    with pytest.raises(ValueError, match="allowed port|allowed_ports"):
        DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("example.com",)),
            tmp_path / "result",
            allowed_ports=ports,  # type: ignore[arg-type]
            proxy_source=_proxy_source(tmp_path),
            token_factory=lambda: "abcddcba",
        )


def test_comparability_includes_ports_but_not_ephemeral_names(tmp_path: Path) -> None:
    policy = NetworkPolicy("allowlist", ("example.com",))
    first = DockerNetworkPolicySession(
        policy,
        tmp_path / "first",
        allowed_ports=(80, 443),
        proxy_source=_proxy_source(tmp_path),
        token_factory=lambda: "10101010",
    )
    same_behavior = DockerNetworkPolicySession(
        policy,
        tmp_path / "second",
        allowed_ports=(443, 80),
        proxy_source=tmp_path / "network_proxy.py",
        token_factory=lambda: "20202020",
    )
    different_ports = DockerNetworkPolicySession(
        policy,
        tmp_path / "third",
        allowed_ports=(443,),
        proxy_source=tmp_path / "network_proxy.py",
        token_factory=lambda: "30303030",
    )

    assert first.comparability_digest == same_behavior.comparability_digest
    assert first.comparability_digest != different_ports.comparability_digest
    assert "tasktrace-policy-10101010" not in json.dumps(first.to_comparability_payload())


def test_report_write_failure_after_setup_still_cleans_infrastructure(
    tmp_path: Path,
) -> None:
    docker = FakeDocker()

    class FailFirstReport(DockerNetworkPolicySession):
        writes = 0

        def _write_report(self, *, best_effort: bool = False) -> None:
            self.writes += 1
            if self.writes == 1:
                raise OSError("injected report failure")
            super()._write_report(best_effort=best_effort)

    with pytest.raises(NetworkEnforcementError, match="injected report failure"):
        with FailFirstReport(
            NetworkPolicy("allowlist", ("example.com",)),
            tmp_path / "result",
            command_runner=docker,
            proxy_source=_proxy_source(tmp_path),
            external_network="fixture-external",
            token_factory=lambda: "aaaabbbb",
        ):
            raise AssertionError("session must not become active")

    assert any(command[1:4] == ("container", "rm", "--force") for command in docker.commands)
    assert any(command[1:3] == ("network", "rm") for command in docker.commands)
    report = json.loads((tmp_path / "result" / "network_enforcement.json").read_text())
    assert report["state"] == "preparation-failed"
    assert report["cleanup"]["host_state_removed"] is True
