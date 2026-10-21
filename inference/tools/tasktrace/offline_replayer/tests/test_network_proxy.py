# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import contextlib
import http.server
import io
import json
import socket
import socketserver
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import pytest
from offline_replayer.measurement.network_proxy import (
    ALLOWED_HOSTS_ENV,
    AllowlistProxyServer,
    DecisionLogger,
    ProxyConfig,
    ProxyConfigurationError,
    canonical_allowed_host,
)


class _OriginHandler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    records: list[dict[str, object]] = []

    def _record_and_reply(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        type(self).records.append({"method": self.command, "path": self.path, "headers": self.headers, "body": body})
        response = b"origin-response"
        self.send_response(200)
        self.send_header("Content-Length", str(len(response)))
        self.send_header("Connection", "X-Origin-Secret")
        self.send_header("X-Origin-Secret", "must-not-escape")
        self.send_header("Proxy-Authenticate", "Basic secret-realm")
        self.end_headers()
        self.wfile.write(response)

    do_GET = _record_and_reply  # noqa: N815
    do_POST = _record_and_reply  # noqa: N815

    def do_HEAD(self) -> None:
        response = b"origin-response"
        self.send_response(200)
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()

    def log_message(self, _format: str, *_args: object) -> None:
        return


class _EchoHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        while True:
            data = self.request.recv(65536)
            if not data:
                return
            self.request.sendall(data)


class _ThreadingTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True


class _RawResponseHandler(socketserver.BaseRequestHandler):
    response = b""

    def handle(self) -> None:
        self.request.recv(64 * 1024)
        self.request.sendall(type(self).response)


@contextlib.contextmanager
def _raw_origin(response: bytes) -> Iterator[tuple[str, int]]:
    _RawResponseHandler.response = response
    server = _ThreadingTCPServer(("127.0.0.1", 0), _RawResponseHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield str(server.server_address[0]), int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def origin() -> tuple[str, int]:
    _OriginHandler.records = []
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _OriginHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield str(server.server_address[0]), int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.fixture
def echo_server() -> tuple[str, int]:
    server = _ThreadingTCPServer(("127.0.0.1", 0), _EchoHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield str(server.server_address[0]), int(server.server_address[1])
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def _install_fake_dns(monkeypatch: pytest.MonkeyPatch, host: str = "allowed.test") -> list[str]:
    original = socket.getaddrinfo
    calls: list[str] = []

    def fake_getaddrinfo(name: object, port: object, *args: object, **kwargs: object):
        if name == host:
            calls.append(str(name))
            return [
                (
                    socket.AF_INET,
                    socket.SOCK_STREAM,
                    socket.IPPROTO_TCP,
                    "",
                    ("127.0.0.1", int(port)),
                )
            ]
        return original(name, port, *args, **kwargs)

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return calls


def _proxy(
    *,
    allowed_hosts: tuple[str, ...] = ("allowed.test",),
    logger_stream: io.StringIO | None = None,
    allow_loopback: bool = True,
    ready_file: Path | None = None,
    allowed_ports: tuple[int, ...] = (80, 443),
    dns_timeout_seconds: float = 2,
    max_response_body_bytes: int = 8 * 1024 * 1024 * 1024,
) -> AllowlistProxyServer:
    logger = DecisionLogger(logger_stream or io.StringIO())
    return AllowlistProxyServer(
        ProxyConfig(
            allowed_hosts=allowed_hosts,
            allowed_ports=allowed_ports,
            listen_host="127.0.0.1",
            listen_port=0,
            ready_file=ready_file,
            test_allowed_destination_cidrs=("127.0.0.0/8",) if allow_loopback else (),
            idle_timeout_seconds=2,
            connect_timeout_seconds=2,
            dns_timeout_seconds=dns_timeout_seconds,
            max_connection_seconds=5,
            max_response_body_bytes=max_response_body_bytes,
        ),
        logger=logger,
    )


def _exchange(address: tuple[str, int], request: bytes) -> bytes:
    # Use a numeric connect so DNS monkeypatches exercise only the proxy.
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as client:
        client.settimeout(2)
        client.connect(address)
        client.sendall(request)
        client.shutdown(socket.SHUT_WR)
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)


def test_http_absolute_form_forwards_and_strips_hop_and_proxy_auth_headers(
    monkeypatch: pytest.MonkeyPatch, origin: tuple[str, int]
) -> None:
    calls = _install_fake_dns(monkeypatch)
    log = io.StringIO()
    _host, port = origin
    secret = "never-log-this-token"
    with _proxy(logger_stream=log, allowed_ports=(port,)) as proxy:
        response = _exchange(
            proxy.address,
            (
                f"GET http://allowed.test:{port}/private?token={secret} HTTP/1.1\r\n"
                f"Host: allowed.test:{port}\r\n"
                f"Proxy-Authorization: Bearer {secret}\r\n"
                "Connection: X-Client-Secret\r\n"
                f"X-Client-Secret: {secret}\r\n"
                "X-End-To-End: preserved\r\n\r\n"
            ).encode("ascii"),
        )

    assert response.startswith(b"HTTP/1.1 200")
    response_head, response_body = response.split(b"\r\n\r\n", 1)
    assert response_body == b"origin-response"
    assert b"Proxy-Authenticate" not in response_head
    assert b"X-Origin-Secret" not in response_head
    assert calls == ["allowed.test"]
    record = _OriginHandler.records[-1]
    headers = record["headers"]
    assert isinstance(headers, http.client.HTTPMessage)
    assert headers["X-End-To-End"] == "preserved"
    assert headers.get("Proxy-Authorization") is None
    assert headers.get("X-Client-Secret") is None
    assert record["path"] == f"/private?token={secret}"
    assert secret not in log.getvalue()
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    allowed = next(item for item in records if item.get("decision") == "allowed")
    assert allowed["destination_ip"] == "127.0.0.1"


def test_head_preserves_origin_content_length(monkeypatch: pytest.MonkeyPatch, origin: tuple[str, int]) -> None:
    _install_fake_dns(monkeypatch)
    _host, port = origin
    with _proxy(allowed_ports=(port,)) as proxy:
        response = _exchange(
            proxy.address,
            (f"HEAD http://allowed.test:{port}/ HTTP/1.1\r\n" f"Host: allowed.test:{port}\r\n\r\n").encode("ascii"),
        )

    head, body = response.split(b"\r\n\r\n", 1)
    assert b"Content-Length: 15" in head
    assert body == b""


def test_post_body_may_arrive_in_same_packet_as_headers(
    monkeypatch: pytest.MonkeyPatch, origin: tuple[str, int]
) -> None:
    _install_fake_dns(monkeypatch)
    _host, port = origin
    body = b"one-packet-body"
    with _proxy(allowed_ports=(port,)) as proxy:
        response = _exchange(
            proxy.address,
            (
                f"POST http://allowed.test:{port}/upload HTTP/1.1\r\n"
                f"Host: allowed.test:{port}\r\n"
                f"Content-Length: {len(body)}\r\n\r\n"
            ).encode("ascii")
            + body,
        )
    assert response.startswith(b"HTTP/1.1 200")
    assert _OriginHandler.records[-1]["body"] == body


def test_connect_authorizes_then_relays_bidirectionally(
    monkeypatch: pytest.MonkeyPatch, echo_server: tuple[str, int]
) -> None:
    calls = _install_fake_dns(monkeypatch)
    _host, port = echo_server
    with _proxy(allowed_ports=(port,)) as proxy:
        with socket.create_connection(proxy.address, timeout=2) as client:
            client.sendall(f"CONNECT allowed.test:{port} HTTP/1.1\r\nHost: allowed.test:{port}\r\n\r\n".encode("ascii"))
            head = b""
            while b"\r\n\r\n" not in head:
                head += client.recv(1024)
            assert head == b"HTTP/1.1 200 Connection Established\r\n\r\n"
            client.sendall(b"tunnel-payload")
            assert client.recv(1024) == b"tunnel-payload"
    assert calls == ["allowed.test"]


def test_shutdown_closes_an_active_tunnel_cleanly(
    monkeypatch: pytest.MonkeyPatch, echo_server: tuple[str, int]
) -> None:
    _install_fake_dns(monkeypatch)
    _host, port = echo_server
    proxy = _proxy(allowed_ports=(port,)).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(2)
    try:
        client.connect(proxy.address)
        client.sendall(f"CONNECT allowed.test:{port} HTTP/1.1\r\nHost: allowed.test:{port}\r\n\r\n".encode("ascii"))
        head = b""
        while b"\r\n\r\n" not in head:
            head += client.recv(1024)
        started = time.monotonic()
        proxy.shutdown()
        assert time.monotonic() - started < 2
        try:
            assert client.recv(1) == b""
        except (ConnectionResetError, OSError):
            pass
    finally:
        client.close()
        proxy.shutdown()


@pytest.mark.parametrize(
    "target,host_header",
    [
        ("http://127.0.0.1/", "127.0.0.1"),
        ("http://sub.allowed.test/", "sub.allowed.test"),
        ("http://allowed.test/", "other.test"),
    ],
)
def test_ip_literal_subdomain_and_host_mismatch_fail_closed(target: str, host_header: str) -> None:
    with _proxy() as proxy:
        response = _exchange(
            proxy.address,
            f"GET {target} HTTP/1.1\r\nHost: {host_header}\r\n\r\n".encode("ascii"),
        )
    assert response.startswith((b"HTTP/1.1 400", b"HTTP/1.1 403"))


def test_non_global_dns_result_is_denied_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_dns(monkeypatch)
    with _proxy(allow_loopback=False) as proxy:
        response = _exchange(
            proxy.address,
            b"GET http://allowed.test/ HTTP/1.1\r\nHost: allowed.test\r\n\r\n",
        )
    assert response.startswith(b"HTTP/1.1 403")
    assert b"destination address is forbidden" in response


def test_unlisted_port_is_denied_before_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_dns(*_args: object, **_kwargs: object):
        pytest.fail("a denied port must not trigger DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    with _proxy(allowed_ports=(80, 443)) as proxy:
        response = _exchange(
            proxy.address,
            b"GET http://allowed.test:8080/ HTTP/1.1\r\nHost: allowed.test:8080\r\n\r\n",
        )
    assert response.startswith(b"HTTP/1.1 403")
    assert b"destination port is not allowed" in response


def test_explicit_zero_port_is_rejected_without_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_dns(*_args: object, **_kwargs: object):
        pytest.fail("an invalid port must not trigger DNS")

    monkeypatch.setattr(socket, "getaddrinfo", unexpected_dns)
    with _proxy() as proxy:
        response = _exchange(
            proxy.address,
            b"GET http://allowed.test:0/ HTTP/1.1\r\nHost: allowed.test\r\n\r\n",
        )
    assert response.startswith(b"HTTP/1.1 400")


def test_dns_resolution_has_a_deadline(monkeypatch: pytest.MonkeyPatch) -> None:
    def stuck_dns(*_args: object, **_kwargs: object):
        time.sleep(0.2)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", stuck_dns)
    started = time.monotonic()
    with _proxy(dns_timeout_seconds=0.02) as proxy:
        response = _exchange(
            proxy.address,
            b"GET http://allowed.test/ HTTP/1.1\r\nHost: allowed.test\r\n\r\n",
        )
        elapsed = time.monotonic() - started
    assert response.startswith(b"HTTP/1.1 504")
    assert elapsed < 0.15


def test_shutdown_cancels_a_request_waiting_on_dns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entered = threading.Event()
    release = threading.Event()

    def stuck_dns(*_args: object, **_kwargs: object):
        entered.set()
        release.wait(5)
        return []

    monkeypatch.setattr(socket, "getaddrinfo", stuck_dns)
    proxy = _proxy(dns_timeout_seconds=10).start()
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(2)
    client.connect(proxy.address)
    client.sendall(b"GET http://allowed.test/ HTTP/1.1\r\nHost: allowed.test\r\n\r\n")
    assert entered.wait(1)
    started = time.monotonic()
    try:
        proxy.shutdown()
    finally:
        release.set()
        client.close()

    assert time.monotonic() - started < 1


@pytest.mark.parametrize(
    ("origin_response", "body_limit", "reason"),
    [
        (
            b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n" + b"x" * 32,
            8,
            "response_body_too_large",
        ),
        (
            b"HTTP/1.1 200 OK\r\nContent-Length: 10\r\nConnection: close\r\n\r\nabc",
            1024,
            "incomplete_upstream_response",
        ),
    ],
)
def test_response_transfer_failures_are_logged(
    monkeypatch: pytest.MonkeyPatch,
    origin_response: bytes,
    body_limit: int,
    reason: str,
) -> None:
    log = io.StringIO()
    with _raw_origin(origin_response) as (_host, port):
        _install_fake_dns(monkeypatch)
        with _proxy(
            logger_stream=log,
            allowed_ports=(port,),
            max_response_body_bytes=body_limit,
        ) as proxy:
            response = _exchange(
                proxy.address,
                (f"GET http://allowed.test:{port}/ HTTP/1.1\r\n" f"Host: allowed.test:{port}\r\n\r\n").encode("ascii"),
            )

    assert response.startswith(b"HTTP/1.1 200")
    records = [json.loads(line) for line in log.getvalue().splitlines()]
    assert any(record.get("event") == "request_error" and record.get("reason") == reason for record in records)


def test_control_character_in_connect_authority_is_rejected() -> None:
    with _proxy() as proxy:
        response = _exchange(
            proxy.address,
            b"CONNECT allowed.test\t:443 HTTP/1.1\r\n\r\n",
        )
    assert response.startswith(b"HTTP/1.1 400")


def test_any_non_global_member_rejects_entire_dns_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mixed_answer(*_args: object, **_kwargs: object):
        return [
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("8.8.8.8", 80)),
            (socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", ("127.0.0.1", 80)),
        ]

    monkeypatch.setattr(socket, "getaddrinfo", mixed_answer)
    with _proxy(allow_loopback=False) as proxy:
        response = _exchange(
            proxy.address,
            b"GET http://allowed.test/ HTTP/1.1\r\nHost: allowed.test\r\n\r\n",
        )
    assert response.startswith(b"HTTP/1.1 403")


def test_ready_file_has_stable_contract_and_is_cleared_on_shutdown(tmp_path: Path) -> None:
    ready = tmp_path / "ready.json"
    ready.write_text("", encoding="utf-8")
    proxy = _proxy(ready_file=ready).start()
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            payload = json.loads(ready.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            time.sleep(0.01)
            continue
        if payload.get("ready") is True:
            break
    else:
        pytest.fail("proxy did not become ready")
    assert payload == {
        "ready": True,
        "listen_host": "127.0.0.1",
        "listen_port": proxy.address[1],
        "allowed_ports": [80, 443],
    }
    proxy.shutdown()
    assert json.loads(ready.read_text(encoding="utf-8")) == {
        "ready": False,
        "listen_host": "127.0.0.1",
        "listen_port": proxy.address[1],
        "allowed_ports": [80, 443],
    }


def test_configuration_and_environment_are_canonical_and_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ProxyConfig(("B.Example", "a.example", "A.EXAMPLE"), listen_port=0)
    assert config.allowed_hosts == ("a.example", "b.example")
    assert canonical_allowed_host("PyPI.ORG") == "pypi.org"
    with pytest.raises(ProxyConfigurationError):
        ProxyConfig(("127.0.0.1",))
    with pytest.raises(ProxyConfigurationError):
        ProxyConfig(("*.example.com",))

    monkeypatch.setenv(ALLOWED_HOSTS_ENV, '["PyPI.org", "files.pythonhosted.org"]')
    # Exercise the standalone CLI parser without starting the long-running server.
    source = Path(__file__).parents[1] / "measurement" / "network_proxy.py"
    completed = subprocess.run(
        [sys.executable, str(source), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert completed.returncode == 0
    assert "--allow-host" in completed.stdout
    assert "TEST ONLY" in completed.stdout
