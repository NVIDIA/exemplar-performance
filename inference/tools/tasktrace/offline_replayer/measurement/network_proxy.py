# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small deny-by-default HTTP(S) egress proxy for offline replay.

The module deliberately has no TaskTrace imports so the single file can be bind
mounted into a minimal Python sidecar and executed directly::

    python /tasktrace-network-proxy.py \
        --allow-host pypi.org --allow-host files.pythonhosted.org

Allowlist entries are exact ASCII DNS names.  They do not include subdomains,
IP literals are never valid authorities, and every request is resolved anew.
All resolved addresses must be globally routable.  The test-only destination
CIDR escape hatch exists solely so socket tests can use loopback fixtures; it
must never be enabled by production orchestration.

This proxy supports ordinary HTTP absolute-form requests and HTTPS (or other
TCP protocols) through CONNECT.  It is an enforcement component, not a TLS
interceptor: it can authorize the CONNECT authority but cannot see the TLS
request inside the resulting tunnel.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as _datetime
import http.client
import ipaddress
import json
import os
import queue
import re
import secrets
import select
import signal
import socket
import socketserver
import sys
import threading
import time
import urllib.parse
from pathlib import Path
from typing import IO, Sequence

PROXY_SCHEMA_VERSION = "tasktrace-network-proxy.v1"
ALLOWED_HOSTS_ENV = "TASKTRACE_PROXY_ALLOWED_HOSTS"
READY_FILE_ENV = "TASKTRACE_PROXY_READY_FILE"

_DNS_LABEL = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?$")
_TOKEN = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_REQUEST_LINE_LIMIT = 16 * 1024
_STANDARD_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
    }
)


class ProxyConfigurationError(ValueError):
    """Raised when proxy configuration could broaden access ambiguously."""


class _RequestRejectedError(Exception):
    def __init__(self, status: int, reason_code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.reason_code = reason_code
        self.message = message


def canonical_allowed_host(value: str) -> str:
    """Canonicalize one exact allowlist hostname without package imports."""
    if not isinstance(value, str):
        raise ProxyConfigurationError("allowed hosts must be strings")
    if not value or value != value.strip() or not value.isascii():
        raise ProxyConfigurationError(
            f"allowed host {value!r} must be a non-empty ASCII DNS hostname " "without surrounding whitespace"
        )
    host = value.lower()
    if host.endswith(".") or len(host) > 253:
        raise ProxyConfigurationError(
            f"allowed host {value!r} must not have a root dot and must be at " "most 253 bytes"
        )
    try:
        ipaddress.ip_address(host)
    except ValueError:
        pass
    else:
        raise ProxyConfigurationError(f"allowed host {value!r} must not be an IP literal")
    labels = host.split(".")
    if len(labels) < 2 or any(not _DNS_LABEL.fullmatch(label) for label in labels):
        raise ProxyConfigurationError(f"allowed host {value!r} must be an exact multi-label ASCII DNS name")
    if host == "localhost" or host.endswith(".localhost"):
        raise ProxyConfigurationError("localhost names cannot be allowlisted")
    if host == "local" or host.endswith(".local"):
        raise ProxyConfigurationError("local names cannot be allowlisted")
    return host


@dataclasses.dataclass(frozen=True)
class ProxyConfig:
    """Resource bounds and exact host policy for one proxy process.

    ``test_allowed_destination_cidrs`` is an internal test hook.  Production
    callers must leave it empty; accepting private, loopback, link-local, or
    otherwise non-global destinations defeats the proxy's SSRF boundary.
    """

    allowed_hosts: tuple[str, ...]
    allowed_ports: tuple[int, ...] = (80, 443)
    listen_host: str = "0.0.0.0"
    listen_port: int = 3128
    ready_file: Path | None = None
    header_timeout_seconds: float = 10.0
    connect_timeout_seconds: float = 10.0
    dns_timeout_seconds: float = 10.0
    idle_timeout_seconds: float = 120.0
    max_connection_seconds: float = 3600.0
    max_header_bytes: int = 64 * 1024
    max_request_body_bytes: int = 512 * 1024 * 1024
    max_response_body_bytes: int = 8 * 1024 * 1024 * 1024
    max_tunnel_bytes: int = 8 * 1024 * 1024 * 1024
    test_allowed_destination_cidrs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            hosts = tuple(sorted({canonical_allowed_host(item) for item in self.allowed_hosts}))
        except TypeError as exc:
            raise ProxyConfigurationError("allowed_hosts must be iterable") from exc
        object.__setattr__(self, "allowed_hosts", hosts)

        try:
            ports = tuple(sorted(set(self.allowed_ports)))
        except TypeError as exc:
            raise ProxyConfigurationError("allowed_ports must be iterable") from exc
        if any(not isinstance(port, int) or isinstance(port, bool) for port in ports):
            raise ProxyConfigurationError("allowed_ports must contain integers")
        if any(not 1 <= port <= 65535 for port in ports):
            raise ProxyConfigurationError("allowed_ports must be between 1 and 65535")
        object.__setattr__(self, "allowed_ports", ports)

        if not isinstance(self.listen_host, str) or not self.listen_host:
            raise ProxyConfigurationError("listen_host must be non-empty")
        if not isinstance(self.listen_port, int) or not 0 <= self.listen_port <= 65535:
            raise ProxyConfigurationError("listen_port must be between 0 and 65535")

        positive_numbers = {
            "header_timeout_seconds": self.header_timeout_seconds,
            "connect_timeout_seconds": self.connect_timeout_seconds,
            "dns_timeout_seconds": self.dns_timeout_seconds,
            "idle_timeout_seconds": self.idle_timeout_seconds,
            "max_connection_seconds": self.max_connection_seconds,
            "max_header_bytes": self.max_header_bytes,
            "max_request_body_bytes": self.max_request_body_bytes,
            "max_response_body_bytes": self.max_response_body_bytes,
            "max_tunnel_bytes": self.max_tunnel_bytes,
        }
        for name, value in positive_numbers.items():
            if not isinstance(value, (int, float)) or value <= 0:
                raise ProxyConfigurationError(f"{name} must be positive")
        if self.max_header_bytes < 1024:
            raise ProxyConfigurationError("max_header_bytes must be at least 1024")

        networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
        for raw in self.test_allowed_destination_cidrs:
            try:
                network = ipaddress.ip_network(raw, strict=True)
            except (TypeError, ValueError) as exc:
                raise ProxyConfigurationError(f"invalid test-only destination CIDR {raw!r}") from exc
            networks.append(network)
        object.__setattr__(self, "test_allowed_destination_cidrs", tuple(networks))


class DecisionLogger:
    """Thread-safe JSONL logger that never receives URLs, headers, or bodies."""

    def __init__(self, stream: IO[str], *, owns_stream: bool = False) -> None:
        self._stream = stream
        self._owns_stream = owns_stream
        self._lock = threading.Lock()

    @classmethod
    def from_destination(cls, destination: str | None) -> "DecisionLogger":
        if destination in (None, "-"):
            return cls(sys.stdout)
        # Orchestration may pre-create a writable file in a read-only mount.
        stream = open(destination, "a", encoding="utf-8", buffering=1)
        return cls(stream, owns_stream=True)

    def emit(self, event: str, **fields: object) -> None:
        record = {
            "schema_version": PROXY_SCHEMA_VERSION,
            "timestamp": _datetime.datetime.now(_datetime.timezone.utc).isoformat().replace("+00:00", "Z"),
            "event": event,
            **fields,
        }
        line = json.dumps(
            record,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        with self._lock:
            self._stream.write(line + "\n")
            self._stream.flush()

    def close(self) -> None:
        if self._owns_stream:
            with self._lock:
                self._stream.close()


class _ThreadingProxyServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    allow_reuse_address = True
    daemon_threads = True
    block_on_close = False

    def __init__(
        self,
        server_address: tuple[str, int],
        config: ProxyConfig,
        logger: DecisionLogger,
    ) -> None:
        self.config = config
        self.decision_logger = logger
        self._ready_lock = threading.Lock()
        self._active_lock = threading.Lock()
        self._active_requests: set[socket.socket] = set()
        self.shutdown_requested = threading.Event()
        super().__init__(server_address, _ProxyRequestHandler)

    def register_request(self, request: socket.socket) -> None:
        with self._active_lock:
            self._active_requests.add(request)

    def unregister_request(self, request: socket.socket) -> None:
        with self._active_lock:
            self._active_requests.discard(request)

    def close_active_requests(self) -> None:
        self.shutdown_requested.set()
        with self._active_lock:
            requests = tuple(self._active_requests)
        for request in requests:
            try:
                request.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                request.close()
            except OSError:
                pass

    def wait_for_active_requests(self, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self._active_lock:
                if not self._active_requests:
                    return True
            time.sleep(0.01)
        with self._active_lock:
            return not self._active_requests

    def _write_readiness(self, ready: bool) -> None:
        path = self.config.ready_file
        payload = {
            "ready": ready,
            # Report the numeric address selected by bind(), not merely the
            # configured hostname.  The Docker enforcement layer uses this to
            # prove the listener is confined to the sandbox-facing NIC rather
            # than exposed on the proxy's outbound NIC as a shared relay.
            "listen_host": str(self.server_address[0]),
            "listen_port": int(self.server_address[1]),
            "allowed_ports": list(self.config.allowed_ports),
        }
        if path is not None:
            # Open in place: sidecars may be given a pre-created writable file
            # in a directory they cannot modify, so atomic rename is impossible.
            with self._ready_lock:
                with path.open("w", encoding="utf-8") as stream:
                    json.dump(payload, stream, sort_keys=True, separators=(",", ":"))
                    stream.write("\n")
                    stream.flush()
                    os.fsync(stream.fileno())
        self.decision_logger.emit(
            "proxy_ready" if ready else "proxy_stopped",
            **payload,
            allowed_hosts=list(self.config.allowed_hosts),
        )

    def serve_forever(self, poll_interval: float = 0.2) -> None:
        self._write_readiness(True)
        try:
            super().serve_forever(poll_interval=poll_interval)
        finally:
            try:
                self._write_readiness(False)
            except OSError as exc:
                self.decision_logger.emit("readiness_write_failed", error_type=type(exc).__name__)


class AllowlistProxyServer:
    """Programmatic lifecycle wrapper used by orchestration and socket tests."""

    def __init__(
        self,
        config: ProxyConfig,
        *,
        logger: DecisionLogger | None = None,
    ) -> None:
        self.config = config
        self.logger = logger or DecisionLogger.from_destination(None)
        self._server = _ThreadingProxyServer((config.listen_host, config.listen_port), config, self.logger)
        self._thread: threading.Thread | None = None

    @property
    def address(self) -> tuple[str, int]:
        host, port = self._server.server_address[:2]
        return str(host), int(port)

    def serve_forever(self) -> None:
        self._server.serve_forever()

    def start(self) -> "AllowlistProxyServer":
        if self._thread is not None:
            raise RuntimeError("proxy has already been started")
        self._thread = threading.Thread(target=self.serve_forever, name="tasktrace-network-proxy", daemon=True)
        self._thread.start()
        return self

    def shutdown(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            self._server.shutdown()
            self._server.close_active_requests()
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                raise RuntimeError("proxy did not stop within five seconds")
            if not self._server.wait_for_active_requests(2.0):
                raise RuntimeError("proxy request handlers did not stop within two seconds")
        self._server.server_close()

    def close(self) -> None:
        self.shutdown()

    def __enter__(self) -> "AllowlistProxyServer":
        return self.start()

    def __exit__(self, *_args: object) -> None:
        self.shutdown()


@dataclasses.dataclass(frozen=True)
class _ParsedRequest:
    method: str
    target: str
    version: str
    headers: tuple[tuple[str, str], ...]
    body_prefix: bytes


class _ProxyRequestHandler(socketserver.BaseRequestHandler):
    server: _ThreadingProxyServer

    def setup(self) -> None:
        self.server.register_request(self.request)

    def finish(self) -> None:
        self.server.unregister_request(self.request)

    def handle(self) -> None:
        request_id = secrets.token_hex(8)
        self._request_host: str | None = None
        self._request_port: int | None = None
        self._request_method: str | None = None
        self._decision_logged = False
        self._response_started = False
        self.request.settimeout(self.server.config.header_timeout_seconds)
        try:
            parsed = self._read_request()
            self._request_method = parsed.method
            deadline = time.monotonic() + self.server.config.max_connection_seconds
            if parsed.method == "CONNECT":
                self._handle_connect(parsed, request_id, deadline)
            else:
                self._handle_http(parsed, request_id, deadline)
        except _RequestRejectedError as exc:
            self._log_failure(
                request_id,
                reason=exc.reason_code,
                decision=("denied" if exc.status in (400, 403, 405, 407, 413, 417) else "error"),
            )
            self._send_error(exc.status, exc.message)
        except (OSError, http.client.HTTPException) as exc:
            self._log_failure(
                request_id,
                reason="io_error",
                decision="error",
                error_type=type(exc).__name__,
            )
            self._send_error(502, "upstream connection failed")
        except Exception as exc:  # fail closed; do not expose details to clients
            self._log_failure(
                request_id,
                reason="internal_error",
                decision="error",
                error_type=type(exc).__name__,
            )
            self._send_error(500, "proxy request failed")

    def _log_failure(
        self,
        request_id: str,
        *,
        reason: str,
        decision: str,
        error_type: str | None = None,
    ) -> None:
        fields: dict[str, object] = {
            "request_id": request_id,
            "reason": reason,
            "method": self._request_method,
            "host": self._request_host,
            "port": self._request_port,
        }
        if error_type is not None:
            fields["error_type"] = error_type
        if self._decision_logged:
            self.server.decision_logger.emit("request_error", **fields)
        else:
            self._decision_logged = True
            self.server.decision_logger.emit("request_decision", decision=decision, **fields)

    def _read_request(self) -> _ParsedRequest:
        block, body_prefix = _read_header_block(
            self.request,
            max_bytes=self.server.config.max_header_bytes,
            first_line_limit=_REQUEST_LINE_LIMIT,
        )
        lines = block.split(b"\r\n")
        try:
            request_line = lines[0].decode("ascii")
        except UnicodeDecodeError as exc:
            raise _RequestRejectedError(400, "non_ascii_request_line", "bad request") from exc
        parts = request_line.split(" ")
        if len(parts) != 3:
            raise _RequestRejectedError(400, "malformed_request_line", "bad request")
        method, target, version = parts
        if not _TOKEN.fullmatch(method) or method != method.upper():
            raise _RequestRejectedError(400, "invalid_method", "bad request method")
        if version not in ("HTTP/1.0", "HTTP/1.1"):
            raise _RequestRejectedError(400, "unsupported_http_version", "bad HTTP version")
        if not target or not target.isascii():
            raise _RequestRejectedError(400, "invalid_request_target", "bad request target")
        headers = _parse_headers(lines[1:-2] if lines[-2:] == [b"", b""] else lines[1:])
        return _ParsedRequest(method, target, version, tuple(headers), body_prefix)

    def _handle_http(self, parsed: _ParsedRequest, request_id: str, deadline: float) -> tuple[str, int]:
        if any(ord(character) < 0x20 or ord(character) == 0x7F for character in parsed.target):
            raise _RequestRejectedError(400, "invalid_request_target", "bad request target")
        split = urllib.parse.urlsplit(parsed.target)
        if split.scheme.lower() != "http" or not split.netloc:
            raise _RequestRejectedError(
                400,
                "absolute_http_url_required",
                "proxy requests require an absolute http:// URL",
            )
        if split.username is not None or split.password is not None or split.fragment:
            raise _RequestRejectedError(400, "invalid_url_authority", "bad URL authority")
        try:
            raw_host = split.hostname
            port = split.port if split.port is not None else 80
        except ValueError as exc:
            raise _RequestRejectedError(400, "invalid_url_port", "bad URL port") from exc
        host = _canonical_request_host(raw_host)
        _validate_port(port)
        self._request_host, self._request_port = host, port
        self._require_allowed_port(port)

        host_values = [value for name, value in parsed.headers if name.lower() == "host"]
        if len(host_values) != 1:
            raise _RequestRejectedError(400, "invalid_host_header_count", "one Host header is required")
        header_host, header_port = _parse_authority(host_values[0], default_port=80, require_port=False)
        if (header_host, header_port) != (host, port):
            raise _RequestRejectedError(400, "host_header_mismatch", "Host header does not match URL")

        content_length = _request_content_length(parsed.headers)
        if content_length > self.server.config.max_request_body_bytes:
            raise _RequestRejectedError(413, "request_body_too_large", "request body is too large")
        if len(parsed.body_prefix) > content_length:
            raise _RequestRejectedError(400, "unexpected_request_bytes", "unexpected bytes after request")
        if any(name.lower() == "expect" for name, _value in parsed.headers):
            raise _RequestRejectedError(417, "expectation_unsupported", "Expect is unsupported")

        upstream = self._connect_authorized(host, port, request_id, parsed.method, deadline)
        try:
            origin_target = urllib.parse.urlunsplit(("", "", split.path or "/", split.query, ""))
            outgoing = _filtered_request_headers(parsed.headers)
            authority = host if port == 80 else f"{host}:{port}"
            had_content_length = any(name.lower() == "content-length" for name, _value in outgoing)
            outgoing = [(name, value) for name, value in outgoing if name.lower() not in ("host", "content-length")]
            if had_content_length:
                outgoing.append(("Content-Length", str(content_length)))
            outgoing.extend([("Host", authority), ("Connection", "close")])
            request_head = [f"{parsed.method} {origin_target} HTTP/1.1\r\n"]
            request_head.extend(f"{name}: {value}\r\n" for name, value in outgoing)
            request_head.append("\r\n")
            upstream.sendall("".join(request_head).encode("latin-1"))
            _copy_exact(
                self.request,
                upstream,
                content_length,
                idle_timeout=self.server.config.idle_timeout_seconds,
                deadline=deadline,
                initial=parsed.body_prefix,
            )
            upstream.settimeout(
                min(
                    self.server.config.idle_timeout_seconds,
                    _remaining_seconds(deadline),
                )
            )
            response = http.client.HTTPResponse(upstream, method=parsed.method)
            response.begin()
            self._forward_response(response, deadline=deadline)
        finally:
            self.server.unregister_request(upstream)
            upstream.close()

        return host, port

    def _handle_connect(self, parsed: _ParsedRequest, request_id: str, deadline: float) -> tuple[str, int]:
        if parsed.headers and any(
            name.lower() in ("transfer-encoding", "content-length") for name, _value in parsed.headers
        ):
            raise _RequestRejectedError(400, "connect_body_forbidden", "CONNECT body is unsupported")
        if parsed.body_prefix:
            raise _RequestRejectedError(400, "early_tunnel_bytes", "wait for the CONNECT response")
        host, port = _parse_authority(parsed.target, default_port=None, require_port=True)
        self._request_host, self._request_port = host, port
        self._require_allowed_port(port)
        upstream = self._connect_authorized(host, port, request_id, "CONNECT", deadline)
        try:
            self._response_started = True
            self.request.sendall(b"HTTP/1.1 200 Connection Established\r\n\r\n")
            relay_result = _relay_tunnel(
                self.request,
                upstream,
                idle_timeout=self.server.config.idle_timeout_seconds,
                max_seconds=_remaining_seconds(deadline),
                max_bytes=self.server.config.max_tunnel_bytes,
            )
            if relay_result != "clean-eof":
                raise _RequestRejectedError(
                    504,
                    "tunnel_" + relay_result.replace("-", "_"),
                    "proxy tunnel terminated by a resource bound",
                )
        finally:
            self.server.unregister_request(upstream)
            upstream.close()
        return host, port

    def _connect_authorized(
        self,
        host: str,
        port: int,
        request_id: str,
        method: str,
        deadline: float,
    ) -> socket.socket:
        if host not in self.server.config.allowed_hosts:
            raise _RequestRejectedError(403, "host_not_allowlisted", "destination is not allowed")
        addresses = _resolve_validated(
            host,
            port,
            self.server.config,
            timeout=min(
                self.server.config.dns_timeout_seconds,
                _remaining_seconds(deadline),
            ),
            cancel_event=self.server.shutdown_requested,
        )
        errors: list[OSError] = []
        for family, socktype, proto, sockaddr in addresses:
            candidate = socket.socket(family, socktype, proto)
            self.server.register_request(candidate)
            try:
                candidate.settimeout(
                    min(
                        self.server.config.connect_timeout_seconds,
                        _remaining_seconds(deadline),
                    )
                )
                # sockaddr is the already-validated numeric result.  Passing it
                # here, rather than the hostname, avoids a second DNS lookup.
                candidate.connect(sockaddr)
                candidate.settimeout(self.server.config.idle_timeout_seconds)
                self._decision_logged = True
                self.server.decision_logger.emit(
                    "request_decision",
                    request_id=request_id,
                    decision="allowed",
                    reason="exact_host_and_port_allowed",
                    method=method,
                    host=host,
                    port=port,
                    destination_ip=str(sockaddr[0]).split("%", 1)[0],
                )
                return candidate
            except OSError as exc:
                errors.append(exc)
                self.server.unregister_request(candidate)
                candidate.close()
            except Exception:
                self.server.unregister_request(candidate)
                candidate.close()
                raise
        raise _RequestRejectedError(502, "upstream_connect_failed", "upstream connection failed")

    def _require_allowed_port(self, port: int) -> None:
        if port not in self.server.config.allowed_ports:
            raise _RequestRejectedError(403, "port_not_allowlisted", "destination port is not allowed")

    def _forward_response(self, response: http.client.HTTPResponse, *, deadline: float) -> None:
        if response.length is not None and response.length > self.server.config.max_response_body_bytes:
            raise _RequestRejectedError(502, "response_body_too_large", "upstream response is too large")

        headers = response.getheaders()
        content_length_values = [value for name, value in headers if name.lower() == "content-length"]
        connection_tokens = _connection_tokens(headers)
        stripped = _STANDARD_HOP_HEADERS | connection_tokens | {"content-length"}
        outgoing = [(name, value) for name, value in headers if name.lower() not in stripped]
        if (
            self._request_method == "HEAD"
            and len(set(content_length_values)) == 1
            and content_length_values[0].isdigit()
        ):
            # A HEAD response carries no body, but its Content-Length describes
            # the representation a corresponding GET would have returned.
            outgoing.append(("Content-Length", content_length_values[0]))
        elif response.length is not None and not response.chunked:
            outgoing.append(("Content-Length", str(response.length)))
        outgoing.append(("Connection", "close"))
        reason = (response.reason or "").replace("\r", " ").replace("\n", " ")
        head = [f"HTTP/1.1 {response.status} {reason}\r\n"]
        head.extend(f"{name}: {value}\r\n" for name, value in outgoing)
        head.append("\r\n")
        self._response_started = True
        self.request.sendall("".join(head).encode("latin-1", errors="replace"))

        total = 0
        while True:
            self.request.settimeout(
                min(
                    self.server.config.idle_timeout_seconds,
                    _remaining_seconds(deadline),
                )
            )
            try:
                chunk = response.read(64 * 1024)
            except http.client.IncompleteRead as exc:
                raise _RequestRejectedError(
                    502,
                    "incomplete_upstream_response",
                    "upstream response ended before Content-Length",
                ) from exc
            if not chunk:
                if response.length not in (None, 0):
                    raise _RequestRejectedError(
                        502,
                        "incomplete_upstream_response",
                        "upstream response ended before Content-Length",
                    )
                return
            total += len(chunk)
            if total > self.server.config.max_response_body_bytes:
                raise _RequestRejectedError(
                    502,
                    "response_body_too_large",
                    "upstream response exceeded the configured limit",
                )
            self.request.sendall(chunk)

    def _send_error(self, status: int, message: str) -> None:
        if self._response_started:
            return
        reason = {
            400: "Bad Request",
            403: "Forbidden",
            405: "Method Not Allowed",
            407: "Proxy Authentication Required",
            413: "Content Too Large",
            417: "Expectation Failed",
            500: "Internal Server Error",
            502: "Bad Gateway",
            504: "Gateway Timeout",
        }.get(status, "Proxy Error")
        safe_message = re.sub(r"[^ -~]", "?", message)[:256]
        body = (safe_message + "\n").encode("ascii", errors="replace")
        payload = (
            f"HTTP/1.1 {status} {reason}\r\n"
            "Content-Type: text/plain; charset=us-ascii\r\n"
            f"Content-Length: {len(body)}\r\n"
            "Connection: close\r\n\r\n"
        ).encode("ascii") + body
        try:
            self.request.sendall(payload)
        except OSError:
            pass


def _read_header_block(sock: socket.socket, *, max_bytes: int, first_line_limit: int) -> tuple[bytes, bytes]:
    data = bytearray()
    while b"\r\n\r\n" not in data:
        if len(data) >= max_bytes:
            raise _RequestRejectedError(413, "headers_too_large", "request headers are too large")
        chunk = sock.recv(min(4096, max_bytes - len(data)))
        if not chunk:
            raise _RequestRejectedError(400, "incomplete_headers", "incomplete request headers")
        data.extend(chunk)
        first_break = data.find(b"\r\n")
        if first_break < 0 and len(data) > first_line_limit:
            raise _RequestRejectedError(413, "request_line_too_large", "request line is too large")
        if first_break > first_line_limit:
            raise _RequestRejectedError(413, "request_line_too_large", "request line is too large")
    end = data.find(b"\r\n\r\n") + 4
    return bytes(data[:end]), bytes(data[end:])


def _parse_headers(raw_lines: Sequence[bytes]) -> list[tuple[str, str]]:
    headers: list[tuple[str, str]] = []
    for raw in raw_lines:
        if not raw:
            continue
        if raw[:1] in (b" ", b"\t") or b":" not in raw:
            raise _RequestRejectedError(400, "malformed_header", "bad request header")
        name_bytes, value_bytes = raw.split(b":", 1)
        try:
            name = name_bytes.decode("ascii")
            value = value_bytes.decode("latin-1").strip(" \t")
        except UnicodeDecodeError as exc:
            raise _RequestRejectedError(400, "malformed_header", "bad request header") from exc
        if not _TOKEN.fullmatch(name):
            raise _RequestRejectedError(400, "invalid_header_name", "bad request header")
        if any(ord(char) < 32 and char != "\t" for char in value) or "\x7f" in value:
            raise _RequestRejectedError(400, "invalid_header_value", "bad request header")
        headers.append((name, value))
    return headers


def _canonical_request_host(value: str | None) -> str:
    if value is None:
        raise _RequestRejectedError(400, "missing_destination_host", "destination host is required")
    try:
        return canonical_allowed_host(value)
    except ProxyConfigurationError as exc:
        raise _RequestRejectedError(403, "invalid_destination_host", "destination host is forbidden") from exc


def _parse_authority(value: str, *, default_port: int | None, require_port: bool) -> tuple[str, int]:
    if (
        not value
        or not value.isascii()
        or any(ord(char) < 0x20 or ord(char) == 0x7F for char in value)
        or any(char in value for char in "/?#@")
    ):
        raise _RequestRejectedError(400, "invalid_authority", "bad destination authority")
    try:
        split = urllib.parse.urlsplit("//" + value)
        raw_host = split.hostname
        explicit_port = split.port
    except ValueError as exc:
        raise _RequestRejectedError(400, "invalid_authority", "bad destination authority") from exc
    if split.username is not None or split.password is not None:
        raise _RequestRejectedError(400, "invalid_authority", "bad destination authority")
    if require_port and explicit_port is None:
        raise _RequestRejectedError(400, "missing_connect_port", "CONNECT requires host:port")
    port = explicit_port if explicit_port is not None else default_port
    if port is None:
        raise _RequestRejectedError(400, "missing_destination_port", "destination port is required")
    _validate_port(port)
    return _canonical_request_host(raw_host), port


def _validate_port(port: int) -> None:
    if not isinstance(port, int) or not 1 <= port <= 65535:
        raise _RequestRejectedError(400, "invalid_destination_port", "destination port is invalid")


def _request_content_length(headers: Sequence[tuple[str, str]]) -> int:
    if any(name.lower() == "transfer-encoding" for name, _value in headers):
        raise _RequestRejectedError(
            400,
            "transfer_encoding_unsupported",
            "chunked requests are unsupported",
        )
    values = [value for name, value in headers if name.lower() == "content-length"]
    if not values:
        return 0
    if len(set(values)) != 1 or not values[0].isdigit():
        raise _RequestRejectedError(400, "invalid_content_length", "bad Content-Length")
    return int(values[0])


def _connection_tokens(headers: Sequence[tuple[str, str]]) -> frozenset[str]:
    tokens: set[str] = set()
    for name, value in headers:
        if name.lower() in ("connection", "proxy-connection"):
            for raw in value.split(","):
                token = raw.strip().lower()
                if token and _TOKEN.fullmatch(token):
                    tokens.add(token)
    return frozenset(tokens)


def _filtered_request_headers(
    headers: Sequence[tuple[str, str]],
) -> list[tuple[str, str]]:
    stripped = _STANDARD_HOP_HEADERS | _connection_tokens(headers)
    return [(name, value) for name, value in headers if name.lower() not in stripped]


def _address_is_permitted(address: ipaddress.IPv4Address | ipaddress.IPv6Address, config: ProxyConfig) -> bool:
    if address.is_global:
        return True
    for network in config.test_allowed_destination_cidrs:
        if address.version == network.version and address in network:
            return True
    return False


def _resolve_validated(
    host: str,
    port: int,
    config: ProxyConfig,
    *,
    timeout: float | None = None,
    cancel_event: threading.Event | None = None,
) -> list[tuple[int, int, int, tuple[object, ...]]]:
    result: queue.Queue[tuple[bool, object]] = queue.Queue(maxsize=1)

    def resolve() -> None:
        try:
            value: object = socket.getaddrinfo(
                host,
                port,
                family=socket.AF_UNSPEC,
                type=socket.SOCK_STREAM,
                proto=socket.IPPROTO_TCP,
            )
        except BaseException as exc:  # delivered back to the request thread
            result.put((False, exc))
        else:
            result.put((True, value))

    resolver = threading.Thread(
        target=resolve,
        name="tasktrace-proxy-dns",
        daemon=True,
    )
    resolver.start()
    deadline = time.monotonic() + (timeout or config.dns_timeout_seconds)
    while True:
        if cancel_event is not None and cancel_event.is_set():
            raise _RequestRejectedError(504, "proxy_shutdown", "proxy is shutting down")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise _RequestRejectedError(504, "dns_resolution_timeout", "destination resolution timed out")
        try:
            succeeded, value = result.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        break
    if not succeeded:
        if isinstance(value, socket.gaierror):
            cause = value
        elif isinstance(value, BaseException):
            raise value
        else:  # pragma: no cover - queue contract defense
            raise RuntimeError("resolver returned an invalid failure")
        raise _RequestRejectedError(502, "dns_resolution_failed", "destination resolution failed") from cause
    if not isinstance(value, list):
        raise _RequestRejectedError(502, "dns_resolution_failed", "destination resolution failed")
    resolved = value
    if not resolved:
        raise _RequestRejectedError(502, "dns_no_addresses", "destination resolution returned no addresses")

    accepted: list[tuple[int, int, int, tuple[object, ...]]] = []
    seen: set[tuple[int, tuple[object, ...]]] = set()
    for family, socktype, proto, _canonname, raw_sockaddr in resolved:
        sockaddr = tuple(raw_sockaddr)
        if (
            family not in (socket.AF_INET, socket.AF_INET6)
            or socktype != socket.SOCK_STREAM
            or proto not in (0, socket.IPPROTO_TCP)
            or len(sockaddr) < 2
        ):
            raise _RequestRejectedError(403, "unsupported_address_family", "destination address is forbidden")
        if not isinstance(sockaddr[1], int) or sockaddr[1] != port:
            raise _RequestRejectedError(403, "dns_port_mismatch", "destination address is forbidden")
        try:
            address = ipaddress.ip_address(str(sockaddr[0]).split("%", 1)[0])
        except ValueError as exc:
            raise _RequestRejectedError(403, "non_numeric_dns_result", "destination address is forbidden") from exc
        if not _address_is_permitted(address, config):
            # Reject the entire resolution set.  Choosing only a global member
            # would permit rebinding/fallback behavior that depends on order.
            raise _RequestRejectedError(403, "non_global_destination", "destination address is forbidden")
        key = (family, sockaddr)
        if key not in seen:
            seen.add(key)
            accepted.append((family, socktype, proto, sockaddr))
    return accepted


def _copy_exact(
    source: socket.socket,
    destination: socket.socket,
    length: int,
    *,
    idle_timeout: float,
    deadline: float,
    initial: bytes = b"",
) -> None:
    remaining = length
    source.settimeout(min(idle_timeout, _remaining_seconds(deadline)))
    destination.settimeout(min(idle_timeout, _remaining_seconds(deadline)))
    if initial:
        if len(initial) > remaining:
            raise _RequestRejectedError(400, "unexpected_request_bytes", "unexpected request bytes")
        destination.sendall(initial)
        remaining -= len(initial)
    while remaining:
        source.settimeout(min(idle_timeout, _remaining_seconds(deadline)))
        destination.settimeout(min(idle_timeout, _remaining_seconds(deadline)))
        chunk = source.recv(min(64 * 1024, remaining))
        if not chunk:
            raise _RequestRejectedError(400, "incomplete_request_body", "incomplete request body")
        destination.sendall(chunk)
        remaining -= len(chunk)


def _relay_tunnel(
    left: socket.socket,
    right: socket.socket,
    *,
    idle_timeout: float,
    max_seconds: float,
    max_bytes: int,
) -> str:
    peers = {left: right, right: left}
    readable = {left, right}
    started = last_activity = time.monotonic()
    total = 0
    while readable:
        now = time.monotonic()
        if now - started >= max_seconds:
            return "lifetime-limit"
        timeout = min(idle_timeout - (now - last_activity), max_seconds - (now - started))
        if timeout <= 0:
            return "idle-timeout"
        ready, _write, _error = select.select(list(readable), [], list(readable), timeout)
        if not ready:
            return "idle-timeout"
        for source in ready:
            try:
                data = source.recv(64 * 1024)
            except OSError:
                return "io-error"
            if not data:
                readable.discard(source)
                try:
                    peers[source].shutdown(socket.SHUT_WR)
                except OSError:
                    pass
                continue
            total += len(data)
            if total > max_bytes:
                return "byte-limit"
            last_activity = time.monotonic()
            peers[source].settimeout(idle_timeout)
            peers[source].sendall(data)
    return "clean-eof"


def _remaining_seconds(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _RequestRejectedError(
            504,
            "connection_lifetime_exceeded",
            "proxy connection exceeded its lifetime limit",
        )
    return remaining


def _hosts_from_environment() -> tuple[str, ...]:
    raw = os.environ.get(ALLOWED_HOSTS_ENV)
    if raw is None or raw == "":
        return ()
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProxyConfigurationError(f"{ALLOWED_HOSTS_ENV} must be a JSON array of exact DNS hostnames") from exc
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ProxyConfigurationError(f"{ALLOWED_HOSTS_ENV} must be a JSON array of strings")
    return tuple(value)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=3128)
    parser.add_argument(
        "--allow-host",
        action="append",
        default=[],
        help=(
            "Exact ASCII DNS host to permit; repeat as needed. If omitted, "
            f"read the JSON array in ${ALLOWED_HOSTS_ENV}."
        ),
    )
    parser.add_argument(
        "--allow-port",
        action="append",
        type=int,
        default=None,
        help="Destination TCP port to permit; repeat as needed (default: 80 and 443).",
    )
    parser.add_argument(
        "--ready-file",
        default=os.environ.get(READY_FILE_ENV),
        help="Write readiness JSON in place (supports a pre-created writable file).",
    )
    parser.add_argument(
        "--decision-log",
        default="-",
        help="JSONL decision log path, or - for stdout (default).",
    )
    parser.add_argument(
        "--test-allow-destination-cidr",
        action="append",
        default=[],
        help="TEST ONLY: permit a non-global fixture CIDR; never use in production.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        env_hosts = _hosts_from_environment()
        if args.allow_host and env_hosts:
            raise ProxyConfigurationError(f"do not combine --allow-host with {ALLOWED_HOSTS_ENV}")
        hosts = tuple(args.allow_host) if args.allow_host else env_hosts
        config = ProxyConfig(
            allowed_hosts=hosts,
            allowed_ports=tuple(args.allow_port) if args.allow_port is not None else (80, 443),
            listen_host=args.listen_host,
            listen_port=args.listen_port,
            ready_file=Path(args.ready_file) if args.ready_file else None,
            test_allowed_destination_cidrs=tuple(args.test_allow_destination_cidr),
        )
        logger = DecisionLogger.from_destination(args.decision_log)
        proxy = AllowlistProxyServer(config, logger=logger)
    except (OSError, ProxyConfigurationError) as exc:
        parser.error(str(exc))

    stop = threading.Event()

    def request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    previous_handlers: dict[int, object] = {}
    for signum in (signal.SIGINT, signal.SIGTERM):
        previous_handlers[signum] = signal.signal(signum, request_stop)

    proxy.start()
    try:
        while not stop.wait(0.2):
            thread = proxy._thread
            if thread is None or not thread.is_alive():
                break
    finally:
        proxy.shutdown()
        logger.close()
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
