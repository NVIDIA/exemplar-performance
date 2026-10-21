# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in real-Docker E2E tests for semantic network policies.

The suite is intentionally local-only: it creates Python HTTP fixtures on
temporary Docker resources and never depends on public DNS or Internet access.

Run explicitly with a pre-pulled image, for example::

    TASKTRACE_DOCKER_NETWORK_POLICY_E2E=1 \
      TASKTRACE_DOCKER_NETWORK_POLICY_TEST_IMAGE=python:3.12-slim \
      uv run pytest -q offline_replayer/tests/test_network_policy_docker.py
"""

from __future__ import annotations

import contextlib
import json
import os
import shlex
import shutil
import socket
import subprocess
import textwrap
import time
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest
from offline_replayer.measurement.container import ProfileConfig, run_profile
from offline_replayer.measurement.network_enforcement import (
    DockerNetworkPolicySession,
)
from offline_replayer.measurement.network_policy import NetworkPolicy

ENABLED = os.environ.get("TASKTRACE_DOCKER_NETWORK_POLICY_E2E") == "1"
IMAGE = os.environ.get("TASKTRACE_DOCKER_NETWORK_POLICY_TEST_IMAGE", "python:3.12-slim")
DOCKER = "docker"

pytestmark = pytest.mark.skipif(
    not ENABLED,
    reason="set TASKTRACE_DOCKER_NETWORK_POLICY_E2E=1 for real-Docker policy E2E",
)

_FIXTURE_SERVER = textwrap.dedent("""
    import os
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    marker = os.environ["TASKTRACE_TEST_MARKER"].encode("ascii") + b"\\n"

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(marker)))
            self.end_headers()
            self.wfile.write(marker)

        def log_message(self, _format, *args):
            del args

    ThreadingHTTPServer(("0.0.0.0", 80), Handler).serve_forever()
    """).strip()


def _docker(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        [DOCKER, *arguments],
        text=True,
        capture_output=True,
        check=False,
    )
    if check and completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip()
        raise AssertionError(
            f"Docker command failed ({completed.returncode}): " f"{shlex.join([DOCKER, *arguments])}: {detail}"
        )
    return completed


@pytest.fixture(scope="module", autouse=True)
def _require_local_image() -> None:
    if not ENABLED:
        return
    inspected = _docker("image", "inspect", IMAGE, check=False)
    assert inspected.returncode == 0, (
        "network-policy E2E never pulls images; pre-pull "
        f"{IMAGE!r} or set TASKTRACE_DOCKER_NETWORK_POLICY_TEST_IMAGE"
    )


def _profile_config(
    output_dir: Path,
    *,
    command: str,
    policy: NetworkPolicy,
    external_network: str = "bridge",
) -> ProfileConfig:
    return ProfileConfig(
        command=command,
        prepare_command=None,
        output_dir=output_dir,
        image=IMAGE,
        container=None,
        cpuset_cpus=None,
        cpu_limit=None,
        allocated_cpus=None,
        workdir="/tmp",
        user=None,
        env=(),
        mounts=(),
        strace_mode="none",
        string_limit=4096,
        shell="/bin/sh",
        network_policy=policy,
        network_policy_proxy_image=IMAGE,
        network_policy_external_network=external_network,
        docker=DOCKER,
    )


def _url_probe_command(url: str, marker: str) -> str:
    expected = marker.encode("ascii") + b"\n"
    source = textwrap.dedent(f"""
        import sys
        import urllib.request

        try:
            body = urllib.request.urlopen({url!r}, timeout=1.5).read()
        except Exception:
            raise SystemExit(23)
        raise SystemExit(0 if body == {expected!r} else 24)
        """)
    return "python -c " + shlex.quote(source)


def _wait_for_host_http(host: str, port: int) -> None:
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=0.2) as stream:
                stream.sendall(b"GET / HTTP/1.0\r\nHost: fixture\r\n\r\n")
                if b"200 OK" in stream.recv(512):
                    return
        except OSError:
            time.sleep(0.05)
    raise AssertionError(f"fixture did not become ready at {host}:{port}")


def _remove_container(name: str) -> None:
    removed = _docker("container", "rm", "--force", name, check=False)
    assert removed.returncode == 0 or "No such container" in removed.stderr


def _remove_network(name: str) -> None:
    removed = _docker("network", "rm", name, check=False)
    assert removed.returncode == 0 or "No such network" in removed.stderr


@contextlib.contextmanager
def _host_published_fixture() -> Iterator[tuple[str, str]]:
    token = uuid.uuid4().hex[:12]
    name = f"tasktrace-policy-e2e-host-{token}"
    marker = "host-published-fixture"
    bridge = json.loads(_docker("network", "inspect", "bridge").stdout)[0]
    gateway = bridge["IPAM"]["Config"][0]["Gateway"]
    created = False
    try:
        _docker(
            "run",
            "--detach",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--publish",
            f"{gateway}::80",
            "--env",
            f"TASKTRACE_TEST_MARKER={marker}",
            IMAGE,
            "python",
            "-B",
            "-c",
            _FIXTURE_SERVER,
        )
        created = True
        inspected = json.loads(_docker("container", "inspect", name).stdout)[0]
        published = inspected["NetworkSettings"]["Ports"]["80/tcp"][0]
        assert published["HostIp"] == gateway
        port = int(published["HostPort"])
        _wait_for_host_http(gateway, port)
        yield f"http://{gateway}:{port}/", marker
    finally:
        if created:
            _remove_container(name)


def _wait_for_container_http(name: str) -> None:
    probe = (
        "import socket; "
        "s=socket.create_connection(('127.0.0.1',80),0.5); "
        "s.sendall(b'GET / HTTP/1.0\\r\\nHost: fixture\\r\\n\\r\\n'); "
        "assert b'200 OK' in s.recv(512); s.close()"
    )
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        completed = _docker("container", "exec", name, "python", "-c", probe, check=False)
        if completed.returncode == 0:
            return
        time.sleep(0.05)
    raise AssertionError(f"fixture container {name!r} did not become ready")


@contextlib.contextmanager
def _allowlist_fixtures() -> Iterator[dict[str, str]]:
    token = uuid.uuid4().hex[:12]
    network = f"tasktrace-policy-e2e-external-{token}"
    allowed = f"tasktrace-policy-e2e-allowed-{token}"
    blocked = f"tasktrace-policy-e2e-blocked-{token}"
    network_created = False
    containers: list[str] = []
    try:
        # These globally classified addresses are local Docker fixtures. They
        # deliberately exercise the proxy's public-address guard without any
        # packet leaving this host.
        _docker(
            "network",
            "create",
            "--driver",
            "bridge",
            "--subnet",
            "8.8.8.0/24",
            "--gateway",
            "8.8.8.1",
            "--label",
            "io.tasktrace.component=network-policy-e2e-fixture",
            network,
        )
        network_created = True
        for name, address, alias, marker in (
            (allowed, "8.8.8.10", "allowed.test", "allowed-fixture"),
            (blocked, "8.8.8.11", "blocked.test", "blocked-fixture"),
        ):
            _docker(
                "run",
                "--detach",
                "--rm",
                "--pull=never",
                "--name",
                name,
                "--network",
                network,
                "--ip",
                address,
                "--network-alias",
                alias,
                "--env",
                f"TASKTRACE_TEST_MARKER={marker}",
                IMAGE,
                "python",
                "-B",
                "-c",
                _FIXTURE_SERVER,
            )
            containers.append(name)
            _wait_for_container_http(name)
        yield {"network": network, "allowed": allowed, "blocked": blocked}
    finally:
        for name in reversed(containers):
            _remove_container(name)
        if network_created:
            _remove_network(network)


def _assert_docker_object_absent(kind: str, name: str) -> None:
    inspected = _docker(kind, "inspect", name, check=False)
    assert inspected.returncode != 0, f"temporary Docker {kind} remains: {name}"


def test_public_no_network_and_empty_allowlist_semantics(tmp_path: Path) -> None:
    with _host_published_fixture() as (url, marker):
        command = _url_probe_command(url, marker)
        public = run_profile(
            _profile_config(
                tmp_path / "public",
                command=command,
                policy=NetworkPolicy("public"),
            )
        )
        isolated = run_profile(
            _profile_config(
                tmp_path / "no-network",
                command=command,
                policy=NetworkPolicy("no-network"),
            )
        )
        empty = run_profile(
            _profile_config(
                tmp_path / "empty-allowlist",
                command=command,
                policy=NetworkPolicy("allowlist"),
            )
        )

    assert public["command_exit_code"] == 0
    assert public["network_policy"]["requested_mode"] == "public"
    assert public["network_policy"]["effective_mode"] == "public"
    assert isolated["command_exit_code"] == 23
    assert isolated["network_policy"]["requested_mode"] == "no-network"
    assert isolated["network_policy"]["effective_mode"] == "no-network"
    assert empty["command_exit_code"] == 23
    assert empty["network_policy"]["requested_mode"] == "allowlist"
    assert empty["network_policy"]["effective_mode"] == "no-network"
    assert empty["network_policy"]["allowed_hosts"] == []
    assert empty["network_policy"]["enforcement"]["backend"] == "docker-native"
    assert empty["network_policy"]["enforcement"]["state"] == "closed"


def test_allowlist_enforces_hosts_and_archives_final_evidence(
    tmp_path: Path,
) -> None:
    source = textwrap.dedent("""
        import socket
        import urllib.error
        import urllib.request

        allowed = urllib.request.urlopen("http://allowed.test/", timeout=2).read()
        if allowed != b"allowed-fixture\\n":
            raise SystemExit(31)

        try:
            urllib.request.urlopen("http://blocked.test/", timeout=2)
        except urllib.error.HTTPError as exc:
            if exc.code != 403:
                raise SystemExit(32)
        else:
            raise SystemExit(33)

        direct = socket.socket()
        direct.settimeout(1)
        try:
            direct.connect(("8.8.8.10", 80))
        except OSError:
            pass
        else:
            raise SystemExit(34)
        finally:
            direct.close()
        """)
    command = "python -c " + shlex.quote(source)

    with _allowlist_fixtures() as fixtures:
        output = tmp_path / "allowlist"
        report = run_profile(
            _profile_config(
                output,
                command=command,
                policy=NetworkPolicy("allowlist", ("allowed.test",)),
                external_network=fixtures["network"],
            )
        )

        assert report["command_exit_code"] == 0
        policy = report["network_policy"]
        assert policy["requested_mode"] == "allowlist"
        assert policy["effective_mode"] == "allowlist"
        assert policy["allowed_hosts"] == ["allowed.test"]

        enforcement = policy["enforcement"]
        assert enforcement["state"] == "closed"
        assert enforcement["backend"] == "docker-forward-proxy-v1"
        assert enforcement["external_network_identity"]["name"] == fixtures["network"]
        assert len(enforcement["external_network_identity"]["configuration_sha256"]) == 64
        assert enforcement["validation"] == {
            "external_network_inspected": True,
            "internal_network_inspected": True,
            "proxy_container_inspected": True,
            "proxy_healthy_at_cleanup": True,
            "proxy_ready": True,
            "proxy_bound_to_internal_address": True,
            "state_disjoint_from_workload_mounts": True,
            "decision_log_validated": True,
        }
        assert enforcement["cleanup"] == {
            "attempted": True,
            "decision_log_archived": True,
            "errors": [],
            "host_state_removed": True,
            "network_removed": True,
            "proxy_removed": True,
            "proxy_stopped": True,
        }

        decision_path = Path(enforcement["proxy"]["decision_log"])
        assert decision_path == output / "network_proxy_decisions.jsonl"
        decisions = [
            json.loads(line) for line in decision_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ]
        log_evidence = enforcement["proxy"]["decision_log_evidence"]
        assert log_evidence["valid"] is True
        assert len(log_evidence["sha256"]) == 64
        assert log_evidence["records"] == len(decisions)
        assert log_evidence["event_counts"]["proxy_ready"] == 1
        assert log_evidence["event_counts"]["proxy_stopped"] == 1
        assert log_evidence["allowed_destination_ips"] == ["8.8.8.10"]
        assert any(
            event.get("event") == "request_decision"
            and event.get("decision") == "allowed"
            and event.get("host") == "allowed.test"
            for event in decisions
        )
        assert any(
            event.get("event") == "request_decision"
            and event.get("decision") == "denied"
            and event.get("host") == "blocked.test"
            and event.get("reason") == "host_not_allowlisted"
            for event in decisions
        )

        final_enforcement = json.loads((output / "network_enforcement.json").read_text(encoding="utf-8"))
        assert final_enforcement == enforcement
        _assert_docker_object_absent("container", enforcement["proxy"]["container_name"])
        _assert_docker_object_absent("network", enforcement["ephemeral_internal_network"])


def test_allowlist_proxy_is_not_a_relay_on_its_external_network(
    tmp_path: Path,
) -> None:
    """An unrelated external-network peer must not reach the proxy listener."""
    with _allowlist_fixtures() as fixtures:
        output = tmp_path / "external-peer-isolation"
        session = DockerNetworkPolicySession(
            NetworkPolicy("allowlist", ("allowed.test",)),
            output,
            docker=DOCKER,
            proxy_image=IMAGE,
            external_network=fixtures["network"],
        )
        with session:
            active = session.report_payload()
            proxy_name = active["proxy"]["container_name"]
            inspected = json.loads(_docker("container", "inspect", proxy_name).stdout)[0]
            external_ip = inspected["NetworkSettings"]["Networks"][fixtures["network"]]["IPAddress"]
            assert external_ip
            probe = textwrap.dedent(f"""
                import socket

                try:
                    stream = socket.create_connection(({external_ip!r}, 3128), 1)
                except OSError:
                    raise SystemExit(0)
                else:
                    stream.close()
                    raise SystemExit(41)
                """)
            peer = _docker(
                "container",
                "exec",
                fixtures["allowed"],
                "python",
                "-c",
                probe,
                check=False,
            )
            assert peer.returncode == 0, "allowlist proxy accepted a connection on its external NIC: " + (
                peer.stderr or peer.stdout
            )
            assert active["proxy"]["bound_internal_ip"]
            assert active["proxy"]["bound_internal_ip"] != external_ip
            assert active["validation"]["proxy_bound_to_internal_address"] is True

        final = session.report_payload()
        assert final["state"] == "closed"
        assert final["cleanup"]["proxy_removed"] is True
        assert final["cleanup"]["network_removed"] is True


def test_public_atif_profile_cli_applies_allowlist_end_to_end(
    tmp_path: Path,
) -> None:
    """Exercise policy routing through the installed customer-facing command."""
    source = textwrap.dedent("""
        import urllib.request

        body = urllib.request.urlopen("http://allowed.test/", timeout=2).read()
        raise SystemExit(0 if body == b"allowed-fixture\\n" else 51)
        """)
    action = "python -c " + shlex.quote(source)
    trajectory = tmp_path / "trajectory.json"
    trajectory.write_text(
        json.dumps(
            {
                "schema_version": "ATIF-v1.5",
                "session_id": "network-policy-e2e",
                "agent": {"name": "network-policy-e2e"},
                "steps": [
                    {
                        "step_id": 1,
                        "timestamp": "2026-07-01T00:00:00Z",
                        "tool_calls": [
                            {
                                "function_name": "exec_command",
                                "arguments": {"cmd": action},
                            }
                        ],
                        "observation": {"content": "Process exited with code 0"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    executable = shutil.which("tasktrace")
    if executable is None:
        candidate = Path(__file__).resolve().parents[2] / ".venv/bin/tasktrace"
        assert candidate.is_file(), "public tasktrace entry point is unavailable"
        executable = str(candidate)

    with _allowlist_fixtures() as fixtures:
        output = tmp_path / "public-atif-profile"
        completed = subprocess.run(
            [
                executable,
                "profile",
                "atif",
                str(trajectory),
                "--image",
                IMAGE,
                "--allocated-cpus",
                "1",
                "--workdir",
                "/tmp",
                "--pacing",
                "unpaced",
                "--strace-mode",
                "none",
                "--network-policy",
                "allowlist",
                "--allowed-host",
                "allowed.test",
                "--network-policy-proxy-image",
                IMAGE,
                "--network-policy-external-network",
                fixtures["network"],
                "--output-dir",
                str(output),
            ],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            capture_output=True,
            check=False,
            timeout=60,
        )

        assert completed.returncode == 0, completed.stderr or completed.stdout
        report = json.loads((output / "pipeline_report.json").read_text())
        assert report["status"] == "valid"
        assert report["replay_summary"]["actions_executed"] == 1
        assert report["replay_summary"]["source_outcome_mismatch_count"] == 0
        assert report["network"] == "tasktrace-allowlist"
        policy = report["network_policy"]
        assert policy["effective_mode"] == "allowlist"
        assert policy["allowed_hosts"] == ["allowed.test"]
        assert policy["enforcement"]["state"] == "closed"
        assert policy["enforcement"]["validation"]["decision_log_validated"] is True
