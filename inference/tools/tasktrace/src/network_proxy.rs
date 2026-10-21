// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Deny-by-default HTTP/HTTPS proxy used by Docker allowlist enforcement.
//!
//! This module deliberately depends only on the standard library and the
//! crate's canonical network-policy parser. The same native executable is
//! mounted read-only into the proxy sidecar and enters this implementation
//! through a hidden internal command.

use crate::network_policy::normalize_allowed_host;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::ffi::OsString;
use std::fmt;
use std::fs::{File, OpenOptions};
use std::io::{self, Read, Write};
use std::net::{
    IpAddr, Ipv4Addr, Ipv6Addr, Shutdown, SocketAddr, TcpListener, TcpStream, ToSocketAddrs,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::thread;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

pub const PROXY_SCHEMA_VERSION: &str = "tasktrace-network-proxy.v1";
const MAX_HEADER_BYTES: usize = 64 * 1024;
const MAX_REQUEST_LINE_BYTES: usize = 16 * 1024;
const MAX_REQUEST_BODY_BYTES: u64 = 512 * 1024 * 1024;
const MAX_RESPONSE_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const MAX_TUNNEL_BYTES: u64 = 8 * 1024 * 1024 * 1024;
const IO_TIMEOUT: Duration = Duration::from_secs(10);
const DNS_TIMEOUT: Duration = Duration::from_secs(10);
const CONNECT_TIMEOUT: Duration = Duration::from_secs(10);
const ACCEPT_POLL: Duration = Duration::from_millis(25);
const MAX_ACTIVE_CONNECTIONS: usize = 12;
const MAX_ACTIVE_RESOLVERS: u64 = 8;

static SIGNAL_STOP: AtomicBool = AtomicBool::new(false);
static ACTIVE_RESOLVERS: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProxyError(String);

impl ProxyError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ProxyError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ProxyError {}

impl From<io::Error> for ProxyError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ProxyError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug)]
pub struct ProxyConfig {
    pub allowed_hosts: BTreeSet<String>,
    pub allowed_ports: BTreeSet<u16>,
    pub listen_host: String,
    pub listen_port: u16,
    pub ready_file: PathBuf,
    pub decision_log: PathBuf,
    #[cfg(test)]
    allow_non_global_for_tests: bool,
}

impl ProxyConfig {
    pub fn new<I, S>(
        allowed_hosts: I,
        allowed_ports: impl IntoIterator<Item = u16>,
        listen_host: impl Into<String>,
        listen_port: u16,
        ready_file: impl Into<PathBuf>,
        decision_log: impl Into<PathBuf>,
    ) -> Result<Self, ProxyError>
    where
        I: IntoIterator<Item = S>,
        S: AsRef<str>,
    {
        let allowed_hosts = allowed_hosts
            .into_iter()
            .map(|host| {
                normalize_allowed_host(host.as_ref())
                    .map_err(|error| ProxyError::new(error.to_string()))
            })
            .collect::<Result<BTreeSet<_>, _>>()?;
        let allowed_ports = allowed_ports.into_iter().collect::<BTreeSet<_>>();
        if allowed_ports.is_empty() || allowed_ports.contains(&0) {
            return Err(ProxyError::new(
                "allowed ports must contain TCP ports from 1 through 65535",
            ));
        }
        let listen_host = listen_host.into();
        if listen_host.trim().is_empty() || !listen_host.is_ascii() {
            return Err(ProxyError::new("listen host must be nonempty ASCII"));
        }
        Ok(Self {
            allowed_hosts,
            allowed_ports,
            listen_host,
            listen_port,
            ready_file: ready_file.into(),
            decision_log: decision_log.into(),
            #[cfg(test)]
            allow_non_global_for_tests: false,
        })
    }

    #[cfg(test)]
    fn allowing_non_global_for_tests(mut self) -> Self {
        self.allow_non_global_for_tests = true;
        self
    }

    fn permits_address(&self, address: IpAddr) -> bool {
        #[cfg(test)]
        if self.allow_non_global_for_tests {
            return true;
        }
        globally_routable(address)
    }
}

#[derive(Debug)]
struct Rejection {
    status: u16,
    reason: &'static str,
    message: &'static str,
}

impl Rejection {
    fn new(status: u16, reason: &'static str, message: &'static str) -> Self {
        Self {
            status,
            reason,
            message,
        }
    }
}

#[derive(Clone)]
struct DecisionLogger {
    stream: Arc<Mutex<File>>,
}

impl DecisionLogger {
    fn open(path: &PathBuf) -> Result<Self, ProxyError> {
        let stream = OpenOptions::new()
            .append(true)
            .open(path)
            .map_err(|error| {
                ProxyError::new(format!(
                    "unable to open pre-created proxy decision log {}: {error}",
                    path.display()
                ))
            })?;
        Ok(Self {
            stream: Arc::new(Mutex::new(stream)),
        })
    }

    fn emit(&self, event: &str, fields: Map<String, Value>) -> Result<(), ProxyError> {
        let mut record = Map::new();
        record.insert(
            "schema_version".to_owned(),
            Value::String(PROXY_SCHEMA_VERSION.to_owned()),
        );
        record.insert("timestamp".to_owned(), Value::String(timestamp()));
        record.insert("event".to_owned(), Value::String(event.to_owned()));
        record.extend(fields);
        let mut line = serde_json::to_vec(&Value::Object(record))?;
        line.push(b'\n');
        let mut stream = self
            .stream
            .lock()
            .map_err(|_| ProxyError::new("proxy decision-log lock was poisoned"))?;
        stream.write_all(&line)?;
        stream.flush()?;
        Ok(())
    }
}

#[derive(Clone)]
struct ActiveConnections {
    next_id: Arc<AtomicU64>,
    streams: Arc<Mutex<BTreeMap<u64, TcpStream>>>,
}

impl ActiveConnections {
    fn new() -> Self {
        Self {
            next_id: Arc::new(AtomicU64::new(1)),
            streams: Arc::new(Mutex::new(BTreeMap::new())),
        }
    }

    fn register(&self, stream: &TcpStream) -> Result<u64, ProxyError> {
        let id = self.next_id.fetch_add(1, Ordering::Relaxed);
        self.streams
            .lock()
            .map_err(|_| ProxyError::new("active-connection lock was poisoned"))?
            .insert(id, stream.try_clone()?);
        Ok(id)
    }

    fn unregister(&self, id: u64) {
        if let Ok(mut streams) = self.streams.lock() {
            streams.remove(&id);
        }
    }

    fn shutdown_all(&self) {
        if let Ok(streams) = self.streams.lock() {
            for stream in streams.values() {
                let _ = stream.shutdown(Shutdown::Both);
            }
        }
    }
}

#[derive(Debug)]
struct ParsedRequest {
    method: String,
    target: String,
    version: String,
    headers: Vec<(String, String)>,
    body_prefix: Vec<u8>,
}

#[derive(Debug, Eq, PartialEq)]
struct HttpTarget {
    host: String,
    port: u16,
    origin_form: String,
}

pub fn serve(config: ProxyConfig, shutdown: Arc<AtomicBool>) -> Result<(), ProxyError> {
    let logger = DecisionLogger::open(&config.decision_log)?;
    let listener = TcpListener::bind((config.listen_host.as_str(), config.listen_port))?;
    listener.set_nonblocking(true)?;
    let address = listener.local_addr()?;
    write_readiness(&config, &logger, address, true)?;
    let config = Arc::new(config);
    let active = ActiveConnections::new();
    let mut workers: Vec<thread::JoinHandle<()>> = Vec::new();
    while !shutdown.load(Ordering::SeqCst) && !SIGNAL_STOP.load(Ordering::SeqCst) {
        match listener.accept() {
            Ok((mut stream, _peer)) => {
                let mut index = 0;
                while index < workers.len() {
                    if workers[index].is_finished() {
                        let worker = workers.swap_remove(index);
                        let _ = worker.join();
                    } else {
                        index += 1;
                    }
                }
                if workers.len() >= MAX_ACTIVE_CONNECTIONS {
                    let mut fields = Map::new();
                    fields.insert("request_id".to_owned(), Value::String(request_id()));
                    fields.insert(
                        "reason".to_owned(),
                        Value::String("connection_capacity_exhausted".to_owned()),
                    );
                    let _ = logger.emit("request_error", fields);
                    let _ = send_error(&mut stream, 503, "proxy connection capacity is exhausted");
                    let _ = stream.shutdown(Shutdown::Both);
                    continue;
                }
                let worker_config = Arc::clone(&config);
                let worker_logger = logger.clone();
                let worker_active = active.clone();
                let worker_shutdown = Arc::clone(&shutdown);
                let worker = thread::Builder::new().spawn(move || {
                    let id = match worker_active.register(&stream) {
                        Ok(id) => id,
                        Err(_) => {
                            let mut fields = Map::new();
                            fields.insert("request_id".to_owned(), Value::String(request_id()));
                            fields.insert(
                                "reason".to_owned(),
                                Value::String("connection_tracking_failed".to_owned()),
                            );
                            let _ = worker_logger.emit("request_error", fields);
                            return;
                        }
                    };
                    handle_connection(
                        stream,
                        &worker_config,
                        &worker_logger,
                        &worker_active,
                        &worker_shutdown,
                    );
                    worker_active.unregister(id);
                });
                if let Ok(worker) = worker {
                    workers.push(worker);
                } else {
                    let mut fields = Map::new();
                    fields.insert("request_id".to_owned(), Value::String(request_id()));
                    fields.insert(
                        "reason".to_owned(),
                        Value::String("connection_worker_failed".to_owned()),
                    );
                    let _ = logger.emit("request_error", fields);
                }
            }
            Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                let mut index = 0;
                while index < workers.len() {
                    if workers[index].is_finished() {
                        let worker = workers.swap_remove(index);
                        let _ = worker.join();
                    } else {
                        index += 1;
                    }
                }
                thread::sleep(ACCEPT_POLL);
            }
            Err(error) => return Err(error.into()),
        }
    }
    active.shutdown_all();
    for worker in workers {
        let _ = worker.join();
    }
    write_readiness(&config, &logger, address, false)?;
    Ok(())
}

fn handle_connection(
    mut client: TcpStream,
    config: &ProxyConfig,
    logger: &DecisionLogger,
    active: &ActiveConnections,
    shutdown: &Arc<AtomicBool>,
) {
    let request_id = request_id();
    let _ = client.set_read_timeout(Some(IO_TIMEOUT));
    let _ = client.set_write_timeout(Some(IO_TIMEOUT));
    let result = (|| -> Result<(), Rejection> {
        let request = read_request(&mut client)?;
        if request.method == "CONNECT" {
            handle_connect(
                &mut client,
                request,
                config,
                logger,
                active,
                shutdown,
                &request_id,
            )
        } else {
            handle_http(
                &mut client,
                request,
                config,
                logger,
                active,
                shutdown,
                &request_id,
            )
        }
    })();
    if let Err(rejection) = result {
        let mut fields = Map::new();
        fields.insert("request_id".to_owned(), Value::String(request_id));
        fields.insert("decision".to_owned(), Value::String("denied".to_owned()));
        fields.insert(
            "reason".to_owned(),
            Value::String(rejection.reason.to_owned()),
        );
        let _ = logger.emit("request_decision", fields);
        let _ = send_error(&mut client, rejection.status, rejection.message);
    }
}

fn handle_http(
    client: &mut TcpStream,
    request: ParsedRequest,
    config: &ProxyConfig,
    logger: &DecisionLogger,
    active: &ActiveConnections,
    shutdown: &Arc<AtomicBool>,
    request_id: &str,
) -> Result<(), Rejection> {
    let target = parse_http_target(&request.target)?;
    require_allowed(config, &target.host, target.port)?;
    let host_headers = request
        .headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("host"))
        .map(|(_, value)| value)
        .collect::<Vec<_>>();
    if host_headers.len() != 1 {
        return Err(Rejection::new(
            400,
            "invalid_host_header_count",
            "one Host header is required",
        ));
    }
    let (header_host, header_port) = parse_authority(host_headers[0], Some(80), false)?;
    if header_host != target.host || header_port != target.port {
        return Err(Rejection::new(
            400,
            "host_header_mismatch",
            "Host header does not match URL",
        ));
    }
    if request.headers.iter().any(|(name, _)| {
        name.eq_ignore_ascii_case("transfer-encoding") || name.eq_ignore_ascii_case("expect")
    }) {
        return Err(Rejection::new(
            400,
            "unsupported_request_framing",
            "unsupported proxy request framing",
        ));
    }
    let content_length = content_length(&request.headers)?;
    if content_length > MAX_REQUEST_BODY_BYTES || request.body_prefix.len() as u64 > content_length
    {
        return Err(Rejection::new(
            413,
            "request_body_too_large",
            "request body is too large",
        ));
    }
    let (mut upstream, destination) =
        connect_authorized(config, &target.host, target.port, shutdown)?;
    let upstream_id = active
        .register(&upstream)
        .map_err(|_| Rejection::new(502, "upstream_tracking_failed", "upstream failed"))?;
    log_allowed(
        logger,
        request_id,
        &request.method,
        &target.host,
        target.port,
        destination,
    );
    let result = (|| -> Result<(), Rejection> {
        let connection_tokens = connection_tokens(&request.headers);
        let mut head = format!(
            "{} {} {}\r\n",
            request.method, target.origin_form, request.version
        );
        for (name, value) in &request.headers {
            let lower = name.to_ascii_lowercase();
            if is_hop_header(&lower)
                || connection_tokens.contains(&lower)
                || lower == "host"
                || lower == "content-length"
            {
                continue;
            }
            head.push_str(name);
            head.push_str(": ");
            head.push_str(value);
            head.push_str("\r\n");
        }
        let authority = if target.port == 80 {
            target.host.clone()
        } else {
            format!("{}:{}", target.host, target.port)
        };
        head.push_str(&format!("Host: {authority}\r\n"));
        if content_length > 0 {
            head.push_str(&format!("Content-Length: {content_length}\r\n"));
        }
        head.push_str("Connection: close\r\n\r\n");
        upstream
            .write_all(head.as_bytes())
            .map_err(|_| Rejection::new(502, "upstream_write_failed", "upstream failed"))?;
        upstream
            .write_all(&request.body_prefix)
            .map_err(|_| Rejection::new(502, "upstream_write_failed", "upstream failed"))?;
        let remaining = content_length.saturating_sub(request.body_prefix.len() as u64);
        copy_exact(client, &mut upstream, remaining)?;
        copy_bounded(&mut upstream, client, MAX_RESPONSE_BYTES)?;
        Ok(())
    })();
    active.unregister(upstream_id);
    let _ = upstream.shutdown(Shutdown::Both);
    result
}

fn handle_connect(
    client: &mut TcpStream,
    request: ParsedRequest,
    config: &ProxyConfig,
    logger: &DecisionLogger,
    active: &ActiveConnections,
    shutdown: &Arc<AtomicBool>,
    request_id: &str,
) -> Result<(), Rejection> {
    if request.headers.iter().any(|(name, _)| {
        name.eq_ignore_ascii_case("transfer-encoding")
            || name.eq_ignore_ascii_case("content-length")
    }) || !request.body_prefix.is_empty()
    {
        return Err(Rejection::new(
            400,
            "connect_body_forbidden",
            "CONNECT body is unsupported",
        ));
    }
    let (host, port) = parse_authority(&request.target, None, true)?;
    require_allowed(config, &host, port)?;
    let (mut upstream, destination) = connect_authorized(config, &host, port, shutdown)?;
    let upstream_id = active
        .register(&upstream)
        .map_err(|_| Rejection::new(502, "upstream_tracking_failed", "upstream failed"))?;
    log_allowed(logger, request_id, "CONNECT", &host, port, destination);
    client
        .write_all(b"HTTP/1.1 200 Connection Established\r\n\r\n")
        .map_err(|_| Rejection::new(502, "client_write_failed", "proxy tunnel failed"))?;
    let result = relay_tunnel(client, &mut upstream, shutdown);
    active.unregister(upstream_id);
    let _ = upstream.shutdown(Shutdown::Both);
    result
}

fn read_request(stream: &mut TcpStream) -> Result<ParsedRequest, Rejection> {
    let mut bytes = Vec::new();
    let header_end = loop {
        if bytes.len() >= MAX_HEADER_BYTES {
            return Err(Rejection::new(
                413,
                "headers_too_large",
                "request headers are too large",
            ));
        }
        let mut buffer = [0_u8; 4096];
        let count = stream
            .read(&mut buffer)
            .map_err(|_| Rejection::new(400, "incomplete_headers", "incomplete headers"))?;
        if count == 0 {
            return Err(Rejection::new(
                400,
                "incomplete_headers",
                "incomplete headers",
            ));
        }
        bytes.extend_from_slice(&buffer[..count]);
        if let Some(position) = find_bytes(&bytes, b"\r\n\r\n") {
            break position + 4;
        }
        if find_bytes(&bytes, b"\r\n").is_none() && bytes.len() > MAX_REQUEST_LINE_BYTES {
            return Err(Rejection::new(
                413,
                "request_line_too_large",
                "request line is too large",
            ));
        }
    };
    let header = &bytes[..header_end];
    let body_prefix = bytes[header_end..].to_vec();
    let text = std::str::from_utf8(header)
        .map_err(|_| Rejection::new(400, "non_ascii_headers", "bad request headers"))?;
    if !text.is_ascii() {
        return Err(Rejection::new(
            400,
            "non_ascii_headers",
            "bad request headers",
        ));
    }
    let mut lines = text[..text.len() - 4].split("\r\n");
    let request_line = lines.next().unwrap_or_default();
    if request_line.len() > MAX_REQUEST_LINE_BYTES {
        return Err(Rejection::new(
            413,
            "request_line_too_large",
            "request line is too large",
        ));
    }
    let parts = request_line.split(' ').collect::<Vec<_>>();
    if parts.len() != 3
        || !valid_token(parts[0])
        || parts[0].bytes().any(|byte| byte.is_ascii_lowercase())
        || !matches!(parts[2], "HTTP/1.0" | "HTTP/1.1")
        || parts[1].is_empty()
        || has_control(parts[1])
    {
        return Err(Rejection::new(
            400,
            "malformed_request_line",
            "bad request line",
        ));
    }
    let mut headers = Vec::new();
    for line in lines {
        if line.starts_with(' ') || line.starts_with('\t') {
            return Err(Rejection::new(
                400,
                "malformed_header",
                "bad request header",
            ));
        }
        let Some((name, raw_value)) = line.split_once(':') else {
            return Err(Rejection::new(
                400,
                "malformed_header",
                "bad request header",
            ));
        };
        if !valid_token(name) {
            return Err(Rejection::new(
                400,
                "invalid_header_name",
                "bad request header",
            ));
        }
        let value = raw_value.trim_matches(|character| matches!(character, ' ' | '\t'));
        if has_control_except_tab(value) {
            return Err(Rejection::new(
                400,
                "invalid_header_value",
                "bad request header",
            ));
        }
        headers.push((name.to_owned(), value.to_owned()));
    }
    Ok(ParsedRequest {
        method: parts[0].to_owned(),
        target: parts[1].to_owned(),
        version: parts[2].to_owned(),
        headers,
        body_prefix,
    })
}

fn parse_http_target(value: &str) -> Result<HttpTarget, Rejection> {
    if has_control(value) || value.contains('#') || value.contains('@') {
        return Err(Rejection::new(
            400,
            "invalid_url_authority",
            "bad URL authority",
        ));
    }
    let Some((scheme, rest)) = value.split_once("://") else {
        return Err(Rejection::new(
            400,
            "absolute_http_url_required",
            "proxy requests require an absolute http URL",
        ));
    };
    if !scheme.eq_ignore_ascii_case("http") || rest.is_empty() {
        return Err(Rejection::new(
            400,
            "absolute_http_url_required",
            "proxy requests require an absolute http URL",
        ));
    }
    let authority_end = rest
        .find(|character| matches!(character, '/' | '?'))
        .unwrap_or(rest.len());
    let authority = &rest[..authority_end];
    let suffix = &rest[authority_end..];
    let (host, port) = parse_authority(authority, Some(80), false)?;
    let origin_form = if suffix.is_empty() {
        "/".to_owned()
    } else if suffix.starts_with('?') {
        format!("/{suffix}")
    } else {
        suffix.to_owned()
    };
    Ok(HttpTarget {
        host,
        port,
        origin_form,
    })
}

fn parse_authority(
    value: &str,
    default_port: Option<u16>,
    require_port: bool,
) -> Result<(String, u16), Rejection> {
    if value.is_empty()
        || !value.is_ascii()
        || has_control(value)
        || value.bytes().any(|byte| b"/?#@".contains(&byte))
        || value.starts_with('[')
        || value.ends_with(']')
    {
        return Err(Rejection::new(
            400,
            "invalid_authority",
            "bad destination authority",
        ));
    }
    let (raw_host, port) = if let Some((host, raw_port)) = value.rsplit_once(':') {
        if host.contains(':')
            || raw_port.is_empty()
            || !raw_port.bytes().all(|byte| byte.is_ascii_digit())
        {
            return Err(Rejection::new(
                400,
                "invalid_destination_port",
                "destination port is invalid",
            ));
        }
        let parsed = raw_port
            .parse::<u16>()
            .ok()
            .filter(|port| *port > 0)
            .ok_or_else(|| {
                Rejection::new(
                    400,
                    "invalid_destination_port",
                    "destination port is invalid",
                )
            })?;
        (host, parsed)
    } else {
        if require_port {
            return Err(Rejection::new(
                400,
                "missing_connect_port",
                "CONNECT requires host:port",
            ));
        }
        (
            value,
            default_port.ok_or_else(|| {
                Rejection::new(
                    400,
                    "missing_destination_port",
                    "destination port is required",
                )
            })?,
        )
    };
    let host = normalize_allowed_host(raw_host).map_err(|_| {
        Rejection::new(
            403,
            "invalid_destination_host",
            "destination host is forbidden",
        )
    })?;
    Ok((host, port))
}

fn require_allowed(config: &ProxyConfig, host: &str, port: u16) -> Result<(), Rejection> {
    if !config.allowed_ports.contains(&port) {
        return Err(Rejection::new(
            403,
            "port_not_allowlisted",
            "destination port is not allowed",
        ));
    }
    if !config.allowed_hosts.contains(host) {
        return Err(Rejection::new(
            403,
            "host_not_allowlisted",
            "destination is not allowed",
        ));
    }
    Ok(())
}

fn connect_authorized(
    config: &ProxyConfig,
    host: &str,
    port: u16,
    shutdown: &Arc<AtomicBool>,
) -> Result<(TcpStream, IpAddr), Rejection> {
    let addresses = resolve_with_timeout(host, port, shutdown)?;
    if addresses
        .iter()
        .any(|address| !config.permits_address(address.ip()))
    {
        return Err(Rejection::new(
            403,
            "non_global_destination",
            "destination address is forbidden",
        ));
    }
    for address in addresses {
        if shutdown.load(Ordering::SeqCst) || SIGNAL_STOP.load(Ordering::SeqCst) {
            return Err(Rejection::new(
                504,
                "proxy_shutdown",
                "proxy is shutting down",
            ));
        }
        if let Ok(stream) = TcpStream::connect_timeout(&address, CONNECT_TIMEOUT) {
            let _ = stream.set_read_timeout(Some(IO_TIMEOUT));
            let _ = stream.set_write_timeout(Some(IO_TIMEOUT));
            return Ok((stream, address.ip()));
        }
    }
    Err(Rejection::new(
        502,
        "upstream_connect_failed",
        "upstream connection failed",
    ))
}

fn resolve_with_timeout(
    host: &str,
    port: u16,
    shutdown: &Arc<AtomicBool>,
) -> Result<Vec<SocketAddr>, Rejection> {
    let mut current = ACTIVE_RESOLVERS.load(Ordering::SeqCst);
    loop {
        if current >= MAX_ACTIVE_RESOLVERS {
            return Err(Rejection::new(
                504,
                "dns_resolver_capacity",
                "destination resolution capacity is exhausted",
            ));
        }
        match ACTIVE_RESOLVERS.compare_exchange(
            current,
            current + 1,
            Ordering::SeqCst,
            Ordering::SeqCst,
        ) {
            Ok(_) => break,
            Err(observed) => current = observed,
        }
    }
    struct ResolverSlot;
    impl Drop for ResolverSlot {
        fn drop(&mut self) {
            ACTIVE_RESOLVERS.fetch_sub(1, Ordering::SeqCst);
        }
    }
    let host = host.to_owned();
    let (sender, receiver) = mpsc::sync_channel(1);
    thread::Builder::new()
        .spawn(move || {
            let _slot = ResolverSlot;
            let result = (host.as_str(), port)
                .to_socket_addrs()
                .map(|values| values.collect::<Vec<_>>());
            let _ = sender.send(result);
        })
        .map_err(|_| {
            ACTIVE_RESOLVERS.fetch_sub(1, Ordering::SeqCst);
            Rejection::new(
                502,
                "dns_resolver_thread_failed",
                "destination resolution failed",
            )
        })?;
    let deadline = std::time::Instant::now() + DNS_TIMEOUT;
    loop {
        if shutdown.load(Ordering::SeqCst) || SIGNAL_STOP.load(Ordering::SeqCst) {
            return Err(Rejection::new(
                504,
                "proxy_shutdown",
                "proxy is shutting down",
            ));
        }
        let now = std::time::Instant::now();
        if now >= deadline {
            return Err(Rejection::new(
                504,
                "dns_resolution_timeout",
                "destination resolution timed out",
            ));
        }
        match receiver.recv_timeout((deadline - now).min(Duration::from_millis(50))) {
            Ok(Ok(values)) => {
                let mut unique = values
                    .into_iter()
                    .collect::<BTreeSet<_>>()
                    .into_iter()
                    .collect::<Vec<_>>();
                unique.sort();
                if unique.is_empty() {
                    return Err(Rejection::new(
                        502,
                        "dns_no_addresses",
                        "destination resolution returned no addresses",
                    ));
                }
                if unique.iter().any(|address| address.port() != port) {
                    return Err(Rejection::new(
                        403,
                        "dns_port_mismatch",
                        "destination address is forbidden",
                    ));
                }
                return Ok(unique);
            }
            Ok(Err(_)) => {
                return Err(Rejection::new(
                    502,
                    "dns_resolution_failed",
                    "destination resolution failed",
                ))
            }
            Err(mpsc::RecvTimeoutError::Timeout) => {}
            Err(mpsc::RecvTimeoutError::Disconnected) => {
                return Err(Rejection::new(
                    502,
                    "dns_resolution_failed",
                    "destination resolution failed",
                ))
            }
        }
    }
}

fn content_length(headers: &[(String, String)]) -> Result<u64, Rejection> {
    let values = headers
        .iter()
        .filter(|(name, _)| name.eq_ignore_ascii_case("content-length"))
        .map(|(_, value)| value.as_str())
        .collect::<Vec<_>>();
    if values.is_empty() {
        return Ok(0);
    }
    if values.iter().any(|value| *value != values[0])
        || !values[0].bytes().all(|byte| byte.is_ascii_digit())
    {
        return Err(Rejection::new(
            400,
            "invalid_content_length",
            "bad Content-Length",
        ));
    }
    values[0]
        .parse::<u64>()
        .map_err(|_| Rejection::new(400, "invalid_content_length", "bad Content-Length"))
}

fn connection_tokens(headers: &[(String, String)]) -> BTreeSet<String> {
    headers
        .iter()
        .filter(|(name, _)| {
            name.eq_ignore_ascii_case("connection") || name.eq_ignore_ascii_case("proxy-connection")
        })
        .flat_map(|(_, value)| value.split(','))
        .map(|value| value.trim().to_ascii_lowercase())
        .filter(|value| valid_token(value))
        .collect()
}

fn is_hop_header(name: &str) -> bool {
    matches!(
        name,
        "connection"
            | "keep-alive"
            | "proxy-authenticate"
            | "proxy-authorization"
            | "proxy-connection"
            | "te"
            | "trailer"
            | "transfer-encoding"
            | "upgrade"
    )
}

fn copy_exact(
    source: &mut TcpStream,
    destination: &mut TcpStream,
    mut remaining: u64,
) -> Result<(), Rejection> {
    let mut buffer = [0_u8; 64 * 1024];
    while remaining > 0 {
        let wanted = remaining.min(buffer.len() as u64) as usize;
        let count = source
            .read(&mut buffer[..wanted])
            .map_err(|_| Rejection::new(400, "incomplete_request_body", "incomplete body"))?;
        if count == 0 {
            return Err(Rejection::new(
                400,
                "incomplete_request_body",
                "incomplete body",
            ));
        }
        destination
            .write_all(&buffer[..count])
            .map_err(|_| Rejection::new(502, "upstream_write_failed", "upstream failed"))?;
        remaining -= count as u64;
    }
    Ok(())
}

fn copy_bounded(
    source: &mut TcpStream,
    destination: &mut TcpStream,
    limit: u64,
) -> Result<(), Rejection> {
    let mut buffer = [0_u8; 64 * 1024];
    let mut total = 0_u64;
    loop {
        let count = source
            .read(&mut buffer)
            .map_err(|_| Rejection::new(502, "upstream_read_failed", "upstream failed"))?;
        if count == 0 {
            return Ok(());
        }
        total = total.saturating_add(count as u64);
        if total > limit {
            return Err(Rejection::new(
                502,
                "response_body_too_large",
                "upstream response is too large",
            ));
        }
        destination
            .write_all(&buffer[..count])
            .map_err(|_| Rejection::new(502, "client_write_failed", "client failed"))?;
    }
}

fn relay_tunnel(
    client: &mut TcpStream,
    upstream: &mut TcpStream,
    shutdown: &Arc<AtomicBool>,
) -> Result<(), Rejection> {
    let mut client_read = client
        .try_clone()
        .map_err(|_| Rejection::new(502, "tunnel_clone_failed", "proxy tunnel failed"))?;
    let mut client_write = client
        .try_clone()
        .map_err(|_| Rejection::new(502, "tunnel_clone_failed", "proxy tunnel failed"))?;
    let mut upstream_read = upstream
        .try_clone()
        .map_err(|_| Rejection::new(502, "tunnel_clone_failed", "proxy tunnel failed"))?;
    let mut upstream_write = upstream
        .try_clone()
        .map_err(|_| Rejection::new(502, "tunnel_clone_failed", "proxy tunnel failed"))?;
    let stop_left = Arc::clone(shutdown);
    let stop_right = Arc::clone(shutdown);
    let left = thread::Builder::new()
        .spawn(move || tunnel_copy(&mut client_read, &mut upstream_write, &stop_left))
        .map_err(|_| Rejection::new(502, "tunnel_thread_failed", "proxy tunnel failed"))?;
    let right = match thread::Builder::new()
        .spawn(move || tunnel_copy(&mut upstream_read, &mut client_write, &stop_right))
    {
        Ok(right) => right,
        Err(_) => {
            let _ = client.shutdown(Shutdown::Both);
            let _ = upstream.shutdown(Shutdown::Both);
            let _ = left.join();
            return Err(Rejection::new(
                502,
                "tunnel_thread_failed",
                "proxy tunnel failed",
            ));
        }
    };
    let left = left.join().unwrap_or(Err(()));
    let right = right.join().unwrap_or(Err(()));
    if left.is_err() || right.is_err() {
        Err(Rejection::new(
            504,
            "tunnel_resource_bound",
            "proxy tunnel terminated",
        ))
    } else {
        Ok(())
    }
}

fn tunnel_copy(
    source: &mut TcpStream,
    destination: &mut TcpStream,
    shutdown: &Arc<AtomicBool>,
) -> Result<(), ()> {
    let _ = source.set_read_timeout(Some(Duration::from_millis(250)));
    let mut total = 0_u64;
    let mut buffer = [0_u8; 64 * 1024];
    loop {
        if shutdown.load(Ordering::SeqCst) || SIGNAL_STOP.load(Ordering::SeqCst) {
            return Ok(());
        }
        match source.read(&mut buffer) {
            Ok(0) => {
                let _ = destination.shutdown(Shutdown::Write);
                return Ok(());
            }
            Ok(count) => {
                total = total.saturating_add(count as u64);
                if total > MAX_TUNNEL_BYTES || destination.write_all(&buffer[..count]).is_err() {
                    return Err(());
                }
            }
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::WouldBlock | io::ErrorKind::TimedOut
                ) => {}
            Err(_) => return Err(()),
        }
    }
}

fn log_allowed(
    logger: &DecisionLogger,
    request_id: &str,
    method: &str,
    host: &str,
    port: u16,
    destination: IpAddr,
) {
    let fields = json!({
        "request_id": request_id,
        "decision": "allowed",
        "reason": "exact_host_and_port_allowed",
        "method": method,
        "host": host,
        "port": port,
        "destination_ip": destination.to_string(),
    })
    .as_object()
    .expect("decision fields are an object")
    .clone();
    let _ = logger.emit("request_decision", fields);
}

fn send_error(stream: &mut TcpStream, status: u16, message: &str) -> io::Result<()> {
    let reason = match status {
        400 => "Bad Request",
        403 => "Forbidden",
        413 => "Content Too Large",
        500 => "Internal Server Error",
        502 => "Bad Gateway",
        504 => "Gateway Timeout",
        _ => "Proxy Error",
    };
    let body = format!(
        "{}\n",
        message.replace(|character| matches!(character, '\r' | '\n'), " ")
    );
    write!(
        stream,
        "HTTP/1.1 {status} {reason}\r\nContent-Type: text/plain; charset=us-ascii\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn write_readiness(
    config: &ProxyConfig,
    logger: &DecisionLogger,
    address: SocketAddr,
    ready: bool,
) -> Result<(), ProxyError> {
    let payload = json!({
        "ready": ready,
        "listen_host": address.ip().to_string(),
        "listen_port": address.port(),
        "allowed_ports": config.allowed_ports,
    });
    let mut file = OpenOptions::new()
        .write(true)
        .truncate(true)
        .open(&config.ready_file)?;
    serde_json::to_writer(&mut file, &payload)?;
    file.write_all(b"\n")?;
    file.sync_all()?;
    let mut fields = payload
        .as_object()
        .expect("readiness payload is an object")
        .clone();
    fields.insert("allowed_hosts".to_owned(), json!(config.allowed_hosts));
    logger.emit(
        if ready {
            "proxy_ready"
        } else {
            "proxy_stopped"
        },
        fields,
    )
}

pub(crate) fn globally_routable(address: IpAddr) -> bool {
    match address {
        IpAddr::V4(address) => globally_routable_v4(address),
        IpAddr::V6(address) => globally_routable_v6(address),
    }
}

fn globally_routable_v4(address: Ipv4Addr) -> bool {
    let value = u32::from(address);
    let in_prefix = |base: u32, bits: u32| value >> (32 - bits) == base >> (32 - bits);
    !(in_prefix(u32::from(Ipv4Addr::new(0, 0, 0, 0)), 8)
        || in_prefix(u32::from(Ipv4Addr::new(10, 0, 0, 0)), 8)
        || in_prefix(u32::from(Ipv4Addr::new(100, 64, 0, 0)), 10)
        || in_prefix(u32::from(Ipv4Addr::new(127, 0, 0, 0)), 8)
        || in_prefix(u32::from(Ipv4Addr::new(169, 254, 0, 0)), 16)
        || in_prefix(u32::from(Ipv4Addr::new(172, 16, 0, 0)), 12)
        || in_prefix(u32::from(Ipv4Addr::new(192, 0, 0, 0)), 24)
        || in_prefix(u32::from(Ipv4Addr::new(192, 0, 2, 0)), 24)
        || in_prefix(u32::from(Ipv4Addr::new(192, 88, 99, 0)), 24)
        || in_prefix(u32::from(Ipv4Addr::new(192, 168, 0, 0)), 16)
        || in_prefix(u32::from(Ipv4Addr::new(198, 18, 0, 0)), 15)
        || in_prefix(u32::from(Ipv4Addr::new(198, 51, 100, 0)), 24)
        || in_prefix(u32::from(Ipv4Addr::new(203, 0, 113, 0)), 24)
        || in_prefix(u32::from(Ipv4Addr::new(224, 0, 0, 0)), 4)
        || in_prefix(u32::from(Ipv4Addr::new(240, 0, 0, 0)), 4))
}

fn globally_routable_v6(address: Ipv6Addr) -> bool {
    if let Some(mapped) = address.to_ipv4_mapped() {
        return globally_routable_v4(mapped);
    }
    let value = u128::from(address);
    let in_prefix =
        |base: Ipv6Addr, bits: u32| value >> (128 - bits) == u128::from(base) >> (128 - bits);
    if !in_prefix("2000::".parse().expect("constant IPv6 prefix"), 3)
        || in_prefix("2001:db8::".parse().expect("constant IPv6 prefix"), 32)
        || in_prefix("2002::".parse().expect("constant IPv6 prefix"), 16)
        || in_prefix("3fff::".parse().expect("constant IPv6 prefix"), 20)
    {
        return false;
    }
    if in_prefix("2001::".parse().expect("constant IPv6 prefix"), 23) {
        let exception = address
            == "2001:1::1"
                .parse::<Ipv6Addr>()
                .expect("constant IPv6 address")
            || address
                == "2001:1::2"
                    .parse::<Ipv6Addr>()
                    .expect("constant IPv6 address")
            || in_prefix("2001:3::".parse().expect("constant IPv6 prefix"), 32)
            || in_prefix("2001:4:112::".parse().expect("constant IPv6 prefix"), 48)
            || in_prefix("2001:20::".parse().expect("constant IPv6 prefix"), 28)
            || in_prefix("2001:30::".parse().expect("constant IPv6 prefix"), 28);
        return exception;
    }
    true
}

fn valid_token(value: &str) -> bool {
    !value.is_empty()
        && value
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || b"!#$%&'*+-.^_`|~".contains(&byte))
}

fn has_control(value: &str) -> bool {
    value.bytes().any(|byte| byte < 0x20 || byte == 0x7f)
}

fn has_control_except_tab(value: &str) -> bool {
    value
        .bytes()
        .any(|byte| (byte < 0x20 && byte != b'\t') || byte == 0x7f)
}

fn find_bytes(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack
        .windows(needle.len())
        .position(|window| window == needle)
}

fn request_id() -> String {
    static NEXT: AtomicU64 = AtomicU64::new(1);
    format!("{:016x}", NEXT.fetch_add(1, Ordering::Relaxed))
}

fn timestamp() -> String {
    let duration = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default();
    let Ok(seconds) = libc::time_t::try_from(duration.as_secs()) else {
        return format!("unix:{}.{:09}", duration.as_secs(), duration.subsec_nanos());
    };
    let mut broken_down = std::mem::MaybeUninit::<libc::tm>::uninit();
    let converted = unsafe { libc::gmtime_r(&seconds, broken_down.as_mut_ptr()) };
    if converted.is_null() {
        return format!("unix:{}.{:09}", duration.as_secs(), duration.subsec_nanos());
    }
    let broken_down = unsafe { broken_down.assume_init() };
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:09}Z",
        broken_down.tm_year + 1900,
        broken_down.tm_mon + 1,
        broken_down.tm_mday,
        broken_down.tm_hour,
        broken_down.tm_min,
        broken_down.tm_sec,
        duration.subsec_nanos(),
    )
}

extern "C" fn stop_handler(_signal: libc::c_int) {
    SIGNAL_STOP.store(true, Ordering::SeqCst);
}

fn parse_internal_arguments(arguments: &[OsString]) -> Result<ProxyConfig, ProxyError> {
    let mut listen_host = "0.0.0.0".to_owned();
    let mut listen_port = 3128_u16;
    let mut ready_file = None;
    let mut decision_log = None;
    let mut allowed_hosts = Vec::new();
    let mut allowed_ports = Vec::new();
    let mut index = 0;
    while index < arguments.len() {
        let raw = arguments[index]
            .to_str()
            .ok_or_else(|| ProxyError::new("proxy arguments must be valid UTF-8"))?;
        let (option, attached) = raw
            .split_once('=')
            .map_or((raw, None), |(name, value)| (name, Some(value)));
        let mut take = || -> Result<String, ProxyError> {
            if let Some(value) = attached {
                return Ok(value.to_owned());
            }
            index += 1;
            arguments
                .get(index)
                .and_then(|value| value.to_str())
                .map(ToOwned::to_owned)
                .ok_or_else(|| ProxyError::new(format!("{option} requires one argument")))
        };
        match option {
            "--listen-host" => listen_host = take()?,
            "--listen-port" => {
                listen_port = take()?
                    .parse::<u16>()
                    .map_err(|_| ProxyError::new("--listen-port must be a TCP port"))?
            }
            "--ready-file" => ready_file = Some(PathBuf::from(take()?)),
            "--decision-log" => decision_log = Some(PathBuf::from(take()?)),
            "--allow-host" => allowed_hosts.push(take()?),
            "--allow-port" => allowed_ports.push(
                take()?
                    .parse::<u16>()
                    .ok()
                    .filter(|port| *port > 0)
                    .ok_or_else(|| ProxyError::new("--allow-port must be a TCP port"))?,
            ),
            _ => return Err(ProxyError::new(format!("unknown proxy option: {raw}"))),
        }
        index += 1;
    }
    ProxyConfig::new(
        allowed_hosts,
        allowed_ports,
        listen_host,
        listen_port,
        ready_file.ok_or_else(|| ProxyError::new("--ready-file is required"))?,
        decision_log.ok_or_else(|| ProxyError::new("--decision-log is required"))?,
    )
}

pub fn internal_main(arguments: &[OsString]) -> i32 {
    let config = match parse_internal_arguments(arguments) {
        Ok(config) => config,
        Err(error) => {
            eprintln!("native network proxy: {error}");
            return 2;
        }
    };
    SIGNAL_STOP.store(false, Ordering::SeqCst);
    let previous_int = unsafe { libc::signal(libc::SIGINT, stop_handler as libc::sighandler_t) };
    let previous_term = unsafe { libc::signal(libc::SIGTERM, stop_handler as libc::sighandler_t) };
    if previous_int == libc::SIG_ERR || previous_term == libc::SIG_ERR {
        eprintln!("native network proxy: unable to install signal handlers");
        return 2;
    }
    let result = serve(config, Arc::new(AtomicBool::new(false)));
    unsafe {
        libc::signal(libc::SIGINT, previous_int);
        libc::signal(libc::SIGTERM, previous_term);
    }
    match result {
        Ok(()) => 0,
        Err(error) => {
            eprintln!("native network proxy: {error}");
            2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(root: &std::path::Path) -> ProxyConfig {
        File::create(root.join("ready.json")).unwrap();
        File::create(root.join("decisions.jsonl")).unwrap();
        ProxyConfig::new(
            ["Allowed.TEST"],
            [80, 443],
            "127.0.0.1",
            0,
            root.join("ready.json"),
            root.join("decisions.jsonl"),
        )
        .unwrap()
        .allowing_non_global_for_tests()
    }

    #[test]
    fn authorities_are_exact_case_insensitive_dns_names_with_explicit_ports() {
        assert_eq!(
            parse_authority("Allowed.TEST:443", None, true).unwrap(),
            ("allowed.test".to_owned(), 443)
        );
        assert_eq!(
            parse_http_target("http://Allowed.TEST/path?q=1").unwrap(),
            HttpTarget {
                host: "allowed.test".to_owned(),
                port: 80,
                origin_form: "/path?q=1".to_owned(),
            }
        );
        for authority in [
            "allowed.test",
            "127.0.0.1:443",
            "sub.allowed.test:443",
            "allowed.test:0",
            "allowed.test:443/path",
            "user@allowed.test:443",
        ] {
            let result = parse_authority(authority, None, true);
            if authority == "sub.allowed.test:443" {
                assert_eq!(result.unwrap().0, "sub.allowed.test");
            } else {
                assert!(result.is_err(), "accepted {authority:?}");
            }
        }
    }

    #[test]
    fn policy_check_does_not_imply_subdomains_or_arbitrary_ports() {
        let root = tempfile::tempdir().unwrap();
        let config = config(root.path());
        assert!(require_allowed(&config, "allowed.test", 80).is_ok());
        assert_eq!(
            require_allowed(&config, "sub.allowed.test", 80)
                .unwrap_err()
                .reason,
            "host_not_allowlisted"
        );
        assert_eq!(
            require_allowed(&config, "allowed.test", 8080)
                .unwrap_err()
                .reason,
            "port_not_allowlisted"
        );
    }

    #[test]
    fn address_guard_rejects_private_loopback_documentation_and_mixed_bypass_space() {
        for address in [
            "127.0.0.1",
            "10.0.0.1",
            "172.16.0.1",
            "192.168.0.1",
            "169.254.1.1",
            "192.0.2.1",
            "198.51.100.1",
            "203.0.113.1",
            "192.88.99.1",
            "::1",
            "fc00::1",
            "fe80::1",
            "2001:db8::1",
            "2001::1",
            "2001:2::1",
            "2002::1",
            "3fff::1",
        ] {
            assert!(
                !globally_routable(address.parse().unwrap()),
                "accepted {address}"
            );
        }
        for address in ["8.8.8.8", "1.1.1.1", "2001:3::1", "2606:4700:4700::1111"] {
            assert!(
                globally_routable(address.parse().unwrap()),
                "rejected {address}"
            );
        }
    }

    #[test]
    fn readiness_and_decisions_never_include_request_urls_or_headers() {
        let root = tempfile::tempdir().unwrap();
        let config = config(root.path());
        let logger = DecisionLogger::open(&config.decision_log).unwrap();
        log_allowed(
            &logger,
            "request",
            "GET",
            "allowed.test",
            80,
            "8.8.8.8".parse().unwrap(),
        );
        let raw = std::fs::read_to_string(&config.decision_log).unwrap();
        assert!(raw.contains("allowed.test"));
        assert!(!raw.contains("http://"));
        assert!(!raw.contains("authorization"));
    }

    #[test]
    fn hidden_parser_normalizes_hosts_and_ports() {
        let root = tempfile::tempdir().unwrap();
        File::create(root.path().join("ready")).unwrap();
        File::create(root.path().join("log")).unwrap();
        let parsed = parse_internal_arguments(&[
            "--listen-host".into(),
            "127.0.0.1".into(),
            "--listen-port=3128".into(),
            "--ready-file".into(),
            root.path().join("ready").into_os_string(),
            "--decision-log".into(),
            root.path().join("log").into_os_string(),
            "--allow-host".into(),
            "Allowed.TEST".into(),
            "--allow-port".into(),
            "443".into(),
        ])
        .unwrap();
        assert_eq!(
            parsed.allowed_hosts,
            BTreeSet::from(["allowed.test".to_owned()])
        );
        assert_eq!(parsed.allowed_ports, BTreeSet::from([443]));
    }
}
