// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Fail-closed process-lifecycle evidence from PID-prefixed `strace` logs.
//!
//! Acquisition is intentionally separate from this parser.  The supported
//! input is emitted by `strace -f -ttt -e trace=%process,chdir,fchdir` into one
//! shared output file.  Decimal realtime timestamps are converted exactly to
//! integer nanoseconds.  Missing, malformed, truncated, escaped, or ambiguous
//! lifecycle facts remain explicit anomalies and invalidate the report.

use regex::Regex;
use serde_json::{json, Map, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;
use std::sync::OnceLock;

pub const PROCESS_EXEC_SCHEMA_VERSION: &str = "software-deep-process-exec.v1";
pub const PROCESS_EXEC_METHOD: &str = "strace-f-ttt-process-chdir";

const RELEVANT_SYSCALLS: &[&str] = &[
    "execve", "execveat", "clone", "clone3", "fork", "vfork", "chdir", "fchdir",
];
const CREATION_SYSCALLS: &[&str] = &["clone", "clone3", "fork", "vfork"];

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProcessExecError(String);

impl fmt::Display for ProcessExecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ProcessExecError {}

impl From<io::Error> for ProcessExecError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ProcessExecError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug)]
struct TaskState {
    tid: u32,
    process_pid: u32,
    cwd_id: u64,
    first_seen_ns: u64,
    start_ns: Option<u64>,
    end_ns: Option<u64>,
    exit_code: Option<i64>,
    exit_signal: Option<String>,
}

#[derive(Clone, Debug)]
struct ProcessState {
    pid: u32,
    parent_pid: Option<u32>,
    parent_resolution: String,
    start_ns: u64,
    start_source: String,
    first_seen_ns: u64,
    cwd_id: u64,
    creation_syscall: Option<String>,
    creation_completed_ns: Option<u64>,
    binary: Option<String>,
    argv: Option<Vec<String>>,
    exec_cwd: Option<String>,
    binary_source: Option<String>,
    end_ns: Option<u64>,
    exit_code: Option<i64>,
    exit_signal: Option<String>,
    execs: Vec<Value>,
    failed_execs: Vec<Value>,
    cwd_changes: Vec<Value>,
}

#[derive(Clone, Debug)]
struct PendingSyscall {
    pid: u32,
    name: String,
    prefix: String,
    started_ns: u64,
    line_number: u64,
}

#[derive(Clone, Debug)]
struct Anomaly {
    code: String,
    message: String,
    line_number: Option<u64>,
    pid: Option<u32>,
    timestamp_ns: Option<u64>,
}

impl Anomaly {
    fn to_json(&self) -> Value {
        let mut value = Map::from_iter([
            ("code".to_owned(), Value::String(self.code.clone())),
            ("message".to_owned(), Value::String(self.message.clone())),
        ]);
        if let Some(line_number) = self.line_number {
            value.insert("line_number".to_owned(), json!(line_number));
        }
        if let Some(pid) = self.pid {
            value.insert("pid".to_owned(), json!(pid));
        }
        if let Some(timestamp_ns) = self.timestamp_ns {
            value.insert("timestamp_ns".to_owned(), json!(timestamp_ns));
        }
        Value::Object(value)
    }
}

struct Parser {
    initial_cwd: Option<String>,
    expected_root_pid: Option<u32>,
    processes: BTreeMap<u32, ProcessState>,
    tasks: BTreeMap<u32, TaskState>,
    cwd_values: BTreeMap<u64, Option<String>>,
    next_cwd_id: u64,
    pending: BTreeMap<(u32, String), PendingSyscall>,
    anomalies: Vec<Anomaly>,
    split_syscalls: u64,
    parsed_line_count: u64,
    total_line_count: u64,
}

impl Parser {
    fn new(initial_cwd: Option<&str>, expected_root_pid: Option<u32>) -> Self {
        Self {
            initial_cwd: normalize_absolute_cwd(initial_cwd),
            expected_root_pid,
            processes: BTreeMap::new(),
            tasks: BTreeMap::new(),
            cwd_values: BTreeMap::new(),
            next_cwd_id: 0,
            pending: BTreeMap::new(),
            anomalies: Vec::new(),
            split_syscalls: 0,
            parsed_line_count: 0,
            total_line_count: 0,
        }
    }

    fn new_cwd(&mut self, value: Option<String>) -> u64 {
        let id = self.next_cwd_id;
        self.next_cwd_id += 1;
        self.cwd_values.insert(id, value);
        id
    }

    fn cwd(&self, id: u64) -> Option<String> {
        self.cwd_values.get(&id).cloned().flatten()
    }

    fn set_cwd(&mut self, id: u64, value: Option<String>) {
        self.cwd_values.insert(id, value);
    }

    fn anomaly(
        &mut self,
        code: &str,
        message: impl Into<String>,
        line_number: Option<u64>,
        pid: Option<u32>,
        timestamp_ns: Option<u64>,
    ) {
        self.anomalies.push(Anomaly {
            code: code.to_owned(),
            message: message.into(),
            line_number,
            pid,
            timestamp_ns,
        });
    }

    fn ensure_task(&mut self, tid: u32, timestamp_ns: u64) -> u32 {
        if let Some(existing) = self.tasks.get_mut(&tid) {
            existing.first_seen_ns = existing.first_seen_ns.min(timestamp_ns);
            let process_pid = existing.process_pid;
            if let Some(process) = self.processes.get_mut(&process_pid) {
                process.first_seen_ns = process.first_seen_ns.min(timestamp_ns);
            }
            return process_pid;
        }
        let root_like = self
            .expected_root_pid
            .map_or(self.processes.is_empty(), |root| tid == root);
        let cwd_id = self.new_cwd(root_like.then(|| self.initial_cwd.clone()).flatten());
        self.processes.insert(
            tid,
            ProcessState {
                pid: tid,
                parent_pid: None,
                parent_resolution: "unresolved".to_owned(),
                start_ns: timestamp_ns,
                start_source: "first-observed-event".to_owned(),
                first_seen_ns: timestamp_ns,
                cwd_id,
                creation_syscall: None,
                creation_completed_ns: None,
                binary: None,
                argv: None,
                exec_cwd: None,
                binary_source: None,
                end_ns: None,
                exit_code: None,
                exit_signal: None,
                execs: Vec::new(),
                failed_execs: Vec::new(),
                cwd_changes: Vec::new(),
            },
        );
        self.tasks.insert(
            tid,
            TaskState {
                tid,
                process_pid: tid,
                cwd_id,
                first_seen_ns: timestamp_ns,
                start_ns: Some(timestamp_ns),
                end_ns: None,
                exit_code: None,
                exit_signal: None,
            },
        );
        tid
    }

    fn parse(mut self, text: &str) -> Value {
        let lines: Vec<&str> = text.lines().collect();
        self.total_line_count = lines.len() as u64;
        for (position, line) in lines.into_iter().enumerate() {
            let line_number = position as u64 + 1;
            if line.trim().is_empty() {
                continue;
            }
            let Some((pid, timestamp, body)) = parse_line(line) else {
                self.anomaly(
                    "malformed_trace_line",
                    "trace line lacks a numeric task ID, decimal timestamp, or body",
                    Some(line_number),
                    None,
                    None,
                );
                continue;
            };
            let Some(timestamp_ns) = timestamp_ns(timestamp) else {
                self.anomaly(
                    "invalid_timestamp",
                    format!(
                        "timestamp is not exactly representable in nanoseconds: {}",
                        python_repr(timestamp)
                    ),
                    Some(line_number),
                    Some(pid),
                    None,
                );
                continue;
            };
            self.parsed_line_count += 1;
            self.parse_body(pid, timestamp_ns, body, line_number);
        }
        let mut pending: Vec<_> = self.pending.values().cloned().collect();
        pending.sort_by(|left, right| {
            (left.line_number, left.pid, left.name.as_str()).cmp(&(
                right.line_number,
                right.pid,
                right.name.as_str(),
            ))
        });
        for item in pending {
            self.anomaly(
                "unfinished_syscall_without_resume",
                format!("{} syscall was unfinished at end of trace", item.name),
                Some(item.line_number),
                Some(item.pid),
                Some(item.started_ns),
            );
        }
        self.pending.clear();
        self.finish()
    }

    fn parse_body(&mut self, pid: u32, timestamp_ns: u64, body: &str, line_number: u64) {
        if let Some(exit_code) = parse_exit(body) {
            self.handle_exit(pid, timestamp_ns, Some(exit_code), None, line_number);
            return;
        }
        if let Some(signal) = parse_killed(body) {
            self.handle_exit(pid, timestamp_ns, None, Some(signal), line_number);
            return;
        }
        if let Some((name, prefix)) = parse_unfinished(body) {
            if !is_relevant(name) {
                return;
            }
            let key = (pid, name.to_owned());
            if self.pending.contains_key(&key) {
                self.anomaly(
                    "overlapping_unfinished_syscall",
                    format!("a second unfinished {name} appeared before its resume"),
                    Some(line_number),
                    Some(pid),
                    Some(timestamp_ns),
                );
                return;
            }
            self.pending.insert(
                key,
                PendingSyscall {
                    pid,
                    name: name.to_owned(),
                    prefix: format!("{name}({prefix}"),
                    started_ns: timestamp_ns,
                    line_number,
                },
            );
            return;
        }
        if let Some((name, suffix)) = parse_resumed(body) {
            if !is_relevant(name) {
                return;
            }
            let Some(pending) = self.pending.remove(&(pid, name.to_owned())) else {
                self.anomaly(
                    "resumed_syscall_without_start",
                    format!("{name} resumed without a matching unfinished syscall"),
                    Some(line_number),
                    Some(pid),
                    Some(timestamp_ns),
                );
                return;
            };
            self.split_syscalls += 1;
            self.handle_syscall(
                pid,
                pending.started_ns,
                timestamp_ns,
                &(pending.prefix + suffix),
                pending.line_number,
            );
            return;
        }
        if let Some((name, arguments, result)) = parse_syscall(body) {
            if is_relevant(name) {
                self.handle_parsed_syscall(
                    pid,
                    timestamp_ns,
                    timestamp_ns,
                    name,
                    arguments,
                    result,
                    line_number,
                );
            }
            return;
        }
        let possible_name = body.split_once('(').map_or(body, |(name, _)| name);
        if is_relevant(possible_name) {
            self.anomaly(
                "malformed_relevant_syscall",
                format!("unable to parse {possible_name} syscall record"),
                Some(line_number),
                Some(pid),
                Some(timestamp_ns),
            );
        }
    }

    fn handle_syscall(
        &mut self,
        pid: u32,
        started_ns: u64,
        completed_ns: u64,
        body: &str,
        line_number: u64,
    ) {
        let Some((name, arguments, result)) = parse_syscall(body) else {
            self.anomaly(
                "malformed_resumed_syscall",
                "unfinished/resumed records could not be reconstructed",
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
            return;
        };
        if !is_relevant(name) {
            self.anomaly(
                "malformed_resumed_syscall",
                "unfinished/resumed records could not be reconstructed",
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
            return;
        }
        self.handle_parsed_syscall(
            pid,
            started_ns,
            completed_ns,
            name,
            arguments,
            result,
            line_number,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_parsed_syscall(
        &mut self,
        pid: u32,
        started_ns: u64,
        completed_ns: u64,
        name: &str,
        arguments: &str,
        result: &str,
        line_number: u64,
    ) {
        match name {
            "execve" | "execveat" => self.handle_exec(
                pid,
                started_ns,
                completed_ns,
                name,
                arguments,
                result,
                line_number,
            ),
            name if CREATION_SYSCALLS.contains(&name) => self.handle_creation(
                pid,
                started_ns,
                completed_ns,
                name,
                arguments,
                result,
                line_number,
            ),
            "chdir" => self.handle_chdir(pid, completed_ns, arguments, result, line_number),
            "fchdir" => self.handle_fchdir(pid, completed_ns, arguments, result, line_number),
            _ => {}
        }
    }
}

fn line_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^\s*(?P<pid>[0-9]+)\s+(?P<timestamp>[0-9]+(?:\.[0-9]+)?)\s+(?P<body>.*)$")
            .expect("static line regex")
    })
}

fn syscall_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<arguments>.*)\)\s+=\s+(?P<result>.+)$")
            .expect("static syscall regex")
    })
}

fn unfinished_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<prefix>.*)<unfinished \.\.\.>$")
            .expect("static unfinished-syscall regex")
    })
}

fn resumed_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^<\.\.\. (?P<name>[A-Za-z_][A-Za-z0-9_]*) resumed>(?P<suffix>.*)$")
            .expect("static resumed-syscall regex")
    })
}

fn exit_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^\+\+\+ exited with (?P<status>-?[0-9]+) \+\+\+$").expect("static exit regex")
    })
}

fn killed_regex() -> &'static Regex {
    static REGEX: OnceLock<Regex> = OnceLock::new();
    REGEX.get_or_init(|| {
        Regex::new(r"^\+\+\+ killed by (?P<signal>SIG[A-Z0-9]+)(?: \([^)]*\))? \+\+\+$")
            .expect("static killed regex")
    })
}

fn parse_line(line: &str) -> Option<(u32, &str, &str)> {
    let captures = line_regex().captures(line)?;
    Some((
        captures.name("pid")?.as_str().parse().ok()?,
        captures.name("timestamp")?.as_str(),
        captures.name("body")?.as_str(),
    ))
}

/// Convert decimal epoch seconds to nanoseconds without floating-point rounding.
fn timestamp_ns(text: &str) -> Option<u64> {
    let (seconds, fraction) = text.split_once('.').unwrap_or((text, ""));
    if seconds.is_empty()
        || !seconds.bytes().all(|byte| byte.is_ascii_digit())
        || !fraction.bytes().all(|byte| byte.is_ascii_digit())
    {
        return None;
    }
    let seconds: u64 = seconds.parse().ok()?;
    let (nanosecond_digits, subnanosecond_digits) = if fraction.len() > 9 {
        fraction.split_at(9)
    } else {
        (fraction, "")
    };
    if !subnanosecond_digits.bytes().all(|byte| byte == b'0') {
        return None;
    }
    let mut nanos = if nanosecond_digits.is_empty() {
        0
    } else {
        nanosecond_digits.parse::<u64>().ok()?
    };
    for _ in nanosecond_digits.len()..9 {
        nanos = nanos.checked_mul(10)?;
    }
    seconds.checked_mul(1_000_000_000)?.checked_add(nanos)
}

fn parse_exit(body: &str) -> Option<i64> {
    exit_regex()
        .captures(body)?
        .name("status")?
        .as_str()
        .parse()
        .ok()
}

fn parse_killed(body: &str) -> Option<String> {
    Some(
        killed_regex()
            .captures(body)?
            .name("signal")?
            .as_str()
            .to_owned(),
    )
}

fn parse_unfinished(body: &str) -> Option<(&str, &str)> {
    let captures = unfinished_regex().captures(body)?;
    Some((
        captures.name("name")?.as_str(),
        captures.name("prefix")?.as_str(),
    ))
}

fn parse_resumed(body: &str) -> Option<(&str, &str)> {
    let captures = resumed_regex().captures(body)?;
    Some((
        captures.name("name")?.as_str(),
        captures.name("suffix")?.as_str(),
    ))
}

fn parse_syscall(body: &str) -> Option<(&str, &str, &str)> {
    let captures = syscall_regex().captures(body)?;
    Some((
        captures.name("name")?.as_str(),
        captures.name("arguments")?.as_str(),
        captures.name("result")?.as_str(),
    ))
}

fn is_relevant(name: &str) -> bool {
    RELEVANT_SYSCALLS.contains(&name)
}

fn integer_result(result: &str) -> Option<i64> {
    let result = result.trim();
    let mut characters = result.char_indices().peekable();
    if characters
        .peek()
        .is_some_and(|(_, character)| *character == '-')
    {
        characters.next();
    }
    let mut saw_digit = false;
    let mut end = 0;
    while let Some((index, character)) = characters.peek().copied() {
        if !character.is_ascii_digit() {
            break;
        }
        saw_digit = true;
        end = index + character.len_utf8();
        characters.next();
    }
    if !saw_digit
        || result[end..]
            .chars()
            .next()
            .is_some_and(|character| !character.is_whitespace())
    {
        return None;
    }
    result[..end].parse().ok()
}

fn errno(result: &str) -> Option<String> {
    let result = result.trim();
    let remainder = result.strip_prefix("-1")?;
    if !remainder.chars().next().is_some_and(char::is_whitespace) {
        return None;
    }
    let token = remainder.split_whitespace().next()?;
    if token.is_empty()
        || !token.as_bytes()[0].is_ascii_uppercase()
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_uppercase() || byte.is_ascii_digit() || byte == b'_')
    {
        return None;
    }
    Some(token.to_owned())
}

fn python_repr(value: &str) -> String {
    let quote = if value.contains('\'') && !value.contains('"') {
        '"'
    } else {
        '\''
    };
    let mut output = String::new();
    output.push(quote);
    for character in value.chars() {
        match character {
            character if character == quote => {
                output.push('\\');
                output.push(character);
            }
            '\\' => output.push_str("\\\\"),
            '\u{0007}' => output.push_str("\\x07"),
            '\u{0008}' => output.push_str("\\x08"),
            '\t' => output.push_str("\\t"),
            '\n' => output.push_str("\\n"),
            '\u{000b}' => output.push_str("\\x0b"),
            '\u{000c}' => output.push_str("\\x0c"),
            '\r' => output.push_str("\\r"),
            character if character.is_control() && u32::from(character) <= 0xff => {
                output.push_str(&format!("\\x{:02x}", u32::from(character)));
            }
            character if character.is_control() && u32::from(character) <= 0xffff => {
                output.push_str(&format!("\\u{:04x}", u32::from(character)));
            }
            character if character.is_control() => {
                output.push_str(&format!("\\U{:08x}", u32::from(character)));
            }
            character => output.push(character),
        }
    }
    output.push(quote);
    output
}

fn split_top_level(text: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut start = 0;
    let mut stack = Vec::new();
    let mut quote = None;
    let mut escaped = false;
    let mut in_comment = false;
    let mut previous = None;
    for (index, character) in text.char_indices() {
        if in_comment {
            if character == '/' && previous == Some('*') {
                in_comment = false;
            }
            previous = Some(character);
            continue;
        }
        if let Some(active_quote) = quote {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == active_quote {
                quote = None;
            }
            previous = Some(character);
            continue;
        }
        match character {
            '\'' | '"' => quote = Some(character),
            '*' if previous == Some('/') => in_comment = true,
            '(' | '[' | '{' => stack.push(character),
            ')' | ']' | '}' => {
                let expected = match character {
                    ')' => '(',
                    ']' => '[',
                    '}' => '{',
                    _ => unreachable!(),
                };
                if stack.pop() != Some(expected) {
                    return None;
                }
            }
            ',' if stack.is_empty() => {
                parts.push(text[start..index].trim().to_owned());
                start = index + character.len_utf8();
            }
            _ => {}
        }
        previous = Some(character);
    }
    if quote.is_some() || !stack.is_empty() || in_comment {
        return None;
    }
    parts.push(text[start..].trim().to_owned());
    Some(parts)
}

fn parse_c_string(token: &str) -> Option<String> {
    let token = token.trim();
    if token.is_empty() || token.ends_with("...") {
        return None;
    }
    let mut input = token;
    let mut output = String::new();
    loop {
        input = input.trim_start();
        let quote = input.chars().next()?;
        if quote != '\'' && quote != '"' {
            return None;
        }
        let mut index = quote.len_utf8();
        let mut closed = false;
        while index < input.len() {
            let character = input[index..].chars().next()?;
            index += character.len_utf8();
            if character == quote {
                closed = true;
                break;
            }
            if character != '\\' {
                output.push(character);
                continue;
            }
            let escaped = input[index..].chars().next()?;
            index += escaped.len_utf8();
            match escaped {
                '\\' => output.push('\\'),
                '\'' => output.push('\''),
                '"' => output.push('"'),
                'a' => output.push('\u{0007}'),
                'b' => output.push('\u{0008}'),
                'f' => output.push('\u{000c}'),
                'n' => output.push('\n'),
                'r' => output.push('\r'),
                't' => output.push('\t'),
                'v' => output.push('\u{000b}'),
                '0'..='7' => {
                    let mut digits = String::from(escaped);
                    for _ in 1..3 {
                        let Some(next) = input[index..].chars().next() else {
                            break;
                        };
                        if !matches!(next, '0'..='7') {
                            break;
                        }
                        digits.push(next);
                        index += next.len_utf8();
                    }
                    output.push(char::from_u32(u32::from_str_radix(&digits, 8).ok()?)?);
                }
                'x' => {
                    let (value, consumed) = parse_fixed_hex(&input[index..], 2)?;
                    index += consumed;
                    output.push(char::from_u32(value)?);
                }
                'u' => {
                    let (value, consumed) = parse_fixed_hex(&input[index..], 4)?;
                    index += consumed;
                    output.push(char::from_u32(value)?);
                }
                'U' => {
                    let (value, consumed) = parse_fixed_hex(&input[index..], 8)?;
                    index += consumed;
                    output.push(char::from_u32(value)?);
                }
                '\n' => {}
                '\r' => {
                    if input[index..].starts_with('\n') {
                        index += 1;
                    }
                }
                other => {
                    output.push('\\');
                    output.push(other);
                }
            }
        }
        if !closed {
            return None;
        }
        input = &input[index..];
        if input.trim().is_empty() {
            return Some(output);
        }
        if !input.chars().next().is_some_and(char::is_whitespace) {
            return None;
        }
    }
}

fn parse_fixed_hex(text: &str, count: usize) -> Option<(u32, usize)> {
    let mut value = 0u32;
    let mut consumed = 0;
    let mut digits = 0;
    for character in text.chars().take(count) {
        value = value
            .checked_mul(16)?
            .checked_add(character.to_digit(16)?)?;
        consumed += character.len_utf8();
        digits += 1;
    }
    (digits == count).then_some((value, consumed))
}

fn parse_argv(token: &str) -> Option<Vec<String>> {
    let token = token.trim();
    let interior = token.strip_prefix('[')?.strip_suffix(']')?;
    let items = split_top_level(interior)?;
    if items.len() == 1 && items[0].is_empty() {
        return Some(Vec::new());
    }
    let mut result = Vec::with_capacity(items.len());
    for item in items {
        if item.trim() == "..." {
            return None;
        }
        result.push(parse_c_string(&item)?);
    }
    Some(result)
}

fn normalize_absolute_cwd(value: Option<&str>) -> Option<String> {
    value
        .filter(|value| value.starts_with('/'))
        .map(posix_normpath)
}

fn posix_normpath(path: &str) -> String {
    let initial_slashes = if path.starts_with("//") && !path.starts_with("///") {
        "//"
    } else if path.starts_with('/') {
        "/"
    } else {
        ""
    };
    let absolute = !initial_slashes.is_empty();
    let mut components: Vec<&str> = Vec::new();
    for component in path.split('/') {
        match component {
            "" | "." => {}
            ".." if components.last().is_some_and(|last| *last != "..") => {
                components.pop();
            }
            ".." if absolute => {}
            _ => components.push(component),
        }
    }
    let joined = components.join("/");
    if joined.is_empty() {
        if absolute {
            initial_slashes.to_owned()
        } else {
            ".".to_owned()
        }
    } else {
        format!("{initial_slashes}{joined}")
    }
}

fn resolve_cwd(path: &str, cwd: Option<&str>) -> Option<String> {
    if path.starts_with('/') {
        return Some(posix_normpath(path));
    }
    let cwd = cwd?;
    Some(posix_normpath(&format!("{cwd}/{path}")))
}

fn resolve_exec_path(
    path: Option<&str>,
    cwd: Option<&str>,
    syscall: &str,
    dirfd: Option<&str>,
) -> (Option<String>, &'static str) {
    let Some(path) = path else {
        return (None, "unresolved");
    };
    if path.starts_with('/') {
        return (Some(posix_normpath(path)), "absolute");
    }
    if syscall == "execve" || dirfd == Some("AT_FDCWD") {
        let resolved = resolve_cwd(path, cwd);
        let resolution = if resolved.is_some() {
            "cwd-relative"
        } else {
            "unresolved"
        };
        return (resolved, resolution);
    }
    (None, "unresolved-dirfd")
}

fn clone_flags(name: &str, arguments: &str) -> Option<BTreeSet<String>> {
    if !matches!(name, "clone" | "clone3") {
        return Some(BTreeSet::new());
    }
    for (index, _) in arguments.match_indices("flags=") {
        let allowed_prefix = index == 0
            || arguments[..index]
                .chars()
                .next_back()
                .is_some_and(|character| matches!(character, ',' | ' ' | '{'));
        if !allowed_prefix {
            continue;
        }
        let value = &arguments[index + "flags=".len()..];
        let end = value
            .find(|character| matches!(character, ',' | '}' | ')'))
            .unwrap_or(value.len());
        return Some(
            value[..end]
                .split('|')
                .map(str::trim)
                .filter(|part| !part.is_empty())
                .map(str::to_owned)
                .collect(),
        );
    }
    None
}

fn sorted_json(value: &Value) -> Value {
    match value {
        Value::Array(values) => Value::Array(values.iter().map(sorted_json).collect()),
        Value::Object(values) => {
            let mut keys: Vec<_> = values.keys().collect();
            keys.sort_unstable();
            let mut sorted = Map::new();
            for key in keys {
                sorted.insert(key.clone(), sorted_json(&values[key]));
            }
            Value::Object(sorted)
        }
        _ => value.clone(),
    }
}

/// Parse one PID-prefixed strace log into deterministic process evidence.
pub fn parse_process_exec_trace(
    text: &str,
    initial_cwd: Option<&str>,
    expected_root_pid: Option<u32>,
) -> Value {
    Parser::new(initial_cwd, expected_root_pid).parse(text)
}

/// Read and parse one strace log without replacing undecodable bytes.
pub fn load_process_exec_trace(
    path: &Path,
    initial_cwd: Option<&str>,
    expected_root_pid: Option<u32>,
) -> Value {
    match fs::read(path) {
        Ok(bytes) => match std::str::from_utf8(&bytes) {
            Ok(text) => parse_process_exec_trace(text, initial_cwd, expected_root_pid),
            Err(error) => {
                let mut parser = Parser::new(initial_cwd, expected_root_pid);
                parser.anomaly(
                    "invalid_trace_encoding",
                    format!("unable to read process trace safely: {error}"),
                    None,
                    None,
                    None,
                );
                parser.finish()
            }
        },
        Err(error) => {
            let mut parser = Parser::new(initial_cwd, expected_root_pid);
            parser.anomaly(
                "trace_read_error",
                format!("unable to read process trace safely: {error}"),
                None,
                None,
                None,
            );
            parser.finish()
        }
    }
}

/// Return canonical stable JSON compatible with Python's sorted report output.
pub fn serialize_process_exec_report(report: &Value) -> Result<String, ProcessExecError> {
    let mut encoded = serde_json::to_string_pretty(&sorted_json(report))?;
    encoded.push('\n');
    Ok(encoded)
}

/// Write a process evidence report deterministically.
pub fn write_process_exec_report(report: &Value, path: &Path) -> Result<(), ProcessExecError> {
    fs::write(path, serialize_process_exec_report(report)?)?;
    Ok(())
}

impl Parser {
    fn reachable_processes(&mut self, root_pid: Option<u32>) -> BTreeSet<u32> {
        let Some(root_pid) = root_pid.filter(|pid| self.processes.contains_key(pid)) else {
            return BTreeSet::new();
        };
        let mut children: BTreeMap<u32, Vec<u32>> = self
            .processes
            .keys()
            .map(|pid| (*pid, Vec::new()))
            .collect();
        for process in self.processes.values() {
            if let Some(items) = process.parent_pid.and_then(|pid| children.get_mut(&pid)) {
                items.push(process.pid);
            }
        }
        let mut reachable = BTreeSet::new();
        let mut stack = vec![root_pid];
        while let Some(pid) = stack.pop() {
            if !reachable.insert(pid) {
                self.anomaly(
                    "process_parent_cycle",
                    format!("process parent graph contains a cycle at PID {pid}"),
                    None,
                    Some(pid),
                    None,
                );
                continue;
            }
            if let Some(items) = children.get(&pid) {
                let mut items = items.clone();
                items.sort_unstable_by(|left, right| right.cmp(left));
                stack.extend(items);
            }
        }
        reachable
    }

    fn finish(mut self) -> Value {
        let (root_pid, root_inference) = if let Some(expected) = self.expected_root_pid {
            if self.processes.contains_key(&expected) {
                (Some(expected), "caller-supplied")
            } else {
                self.anomaly(
                    "expected_root_not_observed",
                    format!("expected root PID {expected} was not observed"),
                    None,
                    Some(expected),
                    None,
                );
                (None, "unresolved")
            }
        } else {
            let candidate = self
                .processes
                .values()
                .filter(|process| process.parent_pid.is_none())
                .min_by_key(|process| (process.first_seen_ns, process.pid))
                .map(|process| process.pid);
            if candidate.is_none() {
                self.anomaly(
                    "trace_root_unresolved",
                    "no trace root could be identified",
                    None,
                    None,
                    None,
                );
                (None, "unresolved")
            } else {
                (candidate, "earliest-unparented-task")
            }
        };
        if let Some(root_pid) = root_pid {
            let root = self.processes.get_mut(&root_pid).expect("observed root");
            root.parent_pid = None;
            root.parent_resolution = "trace-root".to_owned();
        }

        let mut unresolved_parent_pids = Vec::new();
        let process_ids: Vec<u32> = self.processes.keys().copied().collect();
        for pid in process_ids.iter().copied() {
            if Some(pid) == root_pid {
                continue;
            }
            let parent = self.processes[&pid].parent_pid;
            if parent.is_none()
                || !parent.is_some_and(|parent| self.processes.contains_key(&parent))
            {
                unresolved_parent_pids.push(pid);
                self.anomaly(
                    "unresolved_process_parent",
                    format!("process {pid} has no resolved parent in the trace"),
                    None,
                    Some(pid),
                    None,
                );
            }
        }
        let reachable = self.reachable_processes(root_pid);
        let process_escape_pids: Vec<u32> = process_ids
            .iter()
            .copied()
            .filter(|pid| !reachable.contains(pid))
            .collect();
        for pid in &process_escape_pids {
            self.anomaly(
                "process_escape_or_unscoped_task",
                format!("process {pid} is not connected to the trace root"),
                None,
                Some(*pid),
                None,
            );
        }
        let process_snapshots: Vec<ProcessState> = self.processes.values().cloned().collect();
        for process in &process_snapshots {
            if process.end_ns.is_none() {
                self.anomaly(
                    "unresolved_process_lifecycle",
                    format!(
                        "process {} has no terminal exit or signal record",
                        process.pid
                    ),
                    None,
                    Some(process.pid),
                    None,
                );
            } else if process.end_ns.is_some_and(|end| end < process.start_ns) {
                self.anomaly(
                    "invalid_process_interval",
                    format!("process {} ends before it starts", process.pid),
                    None,
                    Some(process.pid),
                    None,
                );
            }
            if process.binary.is_none() {
                self.anomaly(
                    "unresolved_process_binary",
                    format!(
                        "process {} has neither a successful nor inherited executable",
                        process.pid
                    ),
                    None,
                    Some(process.pid),
                    None,
                );
            }
            if process.argv.is_none() {
                self.anomaly(
                    "unresolved_process_argv",
                    format!("process {} has no complete argv evidence", process.pid),
                    None,
                    Some(process.pid),
                    None,
                );
            }
            if process.exec_cwd.is_none() {
                self.anomaly(
                    "unresolved_process_cwd",
                    format!("process {} has no resolved exec/inherited cwd", process.pid),
                    None,
                    Some(process.pid),
                    None,
                );
            }
        }
        let task_snapshots: Vec<TaskState> = self.tasks.values().cloned().collect();
        for task in &task_snapshots {
            if task.tid != task.process_pid && task.end_ns.is_none() {
                self.anomaly(
                    "unresolved_thread_lifecycle",
                    format!("thread task {} has no terminal record", task.tid),
                    None,
                    Some(task.tid),
                    None,
                );
            }
        }

        let mut children: BTreeMap<u32, Vec<u32>> = self
            .processes
            .keys()
            .map(|pid| (*pid, Vec::new()))
            .collect();
        for process in self.processes.values() {
            if let Some(items) = process.parent_pid.and_then(|pid| children.get_mut(&pid)) {
                items.push(process.pid);
            }
        }
        for values in children.values_mut() {
            values.sort_unstable();
        }
        let process_rows: Vec<Value> = self
            .processes
            .values()
            .map(|process| {
                json!({
                    "pid": process.pid,
                    "parent_pid": process.parent_pid,
                    "parent_resolution": process.parent_resolution,
                    "in_scope": reachable.contains(&process.pid),
                    "process_escape": process_escape_pids.contains(&process.pid),
                    "children": children.get(&process.pid).cloned().unwrap_or_default(),
                    "creation_syscall": process.creation_syscall,
                    "start_ns": process.start_ns,
                    "start_source": process.start_source,
                    "creation_completed_ns": process.creation_completed_ns,
                    "end_ns": process.end_ns,
                    "exit_code": process.exit_code,
                    "exit_signal": process.exit_signal,
                    "end_reason": if process.exit_code.is_some() {
                        Some("exit")
                    } else if process.exit_signal.is_some() {
                        Some("signal")
                    } else {
                        None
                    },
                    "binary": process.binary,
                    "argv": process.argv,
                    "cwd": process.exec_cwd,
                    "final_observed_cwd": self.cwd(process.cwd_id),
                    "binary_source": process.binary_source,
                    "execs": process.execs,
                    "failed_execs": process.failed_execs,
                    "cwd_changes": process.cwd_changes,
                })
            })
            .collect();
        let thread_rows: Vec<Value> = self
            .tasks
            .values()
            .filter(|task| task.tid != task.process_pid)
            .map(|task| {
                json!({
                    "tid": task.tid,
                    "process_pid": task.process_pid,
                    "start_ns": task.start_ns,
                    "end_ns": task.end_ns,
                    "exit_code": task.exit_code,
                    "exit_signal": task.exit_signal,
                })
            })
            .collect();

        self.anomalies.sort_by(|left, right| {
            (
                left.line_number.unwrap_or(i64::MAX as u64),
                left.timestamp_ns.unwrap_or(i64::MAX as u64),
                left.pid.map(u64::from).unwrap_or(i64::MAX as u64),
                left.code.as_str(),
                left.message.as_str(),
            )
                .cmp(&(
                    right.line_number.unwrap_or(i64::MAX as u64),
                    right.timestamp_ns.unwrap_or(i64::MAX as u64),
                    right.pid.map(u64::from).unwrap_or(i64::MAX as u64),
                    right.code.as_str(),
                    right.message.as_str(),
                ))
        });
        let codes: Vec<String> = self
            .anomalies
            .iter()
            .map(|anomaly| anomaly.code.clone())
            .collect();
        let anomalies: Vec<Value> = self.anomalies.iter().map(Anomaly::to_json).collect();
        let mut inventory: BTreeMap<String, (u64, BTreeSet<u32>)> = BTreeMap::new();
        for process in self.processes.values() {
            for event in &process.execs {
                let Some(binary) = event.get("binary").and_then(Value::as_str) else {
                    continue;
                };
                let item = inventory.entry(binary.to_owned()).or_default();
                item.0 += 1;
                item.1.insert(process.pid);
            }
        }
        let binary_inventory: Vec<Value> = inventory
            .into_iter()
            .map(|(binary, (exec_count, process_pids))| {
                json!({
                    "binary": binary,
                    "exec_count": exec_count,
                    "process_pids": process_pids,
                })
            })
            .collect();
        let loss_codes = [
            "invalid_trace_encoding",
            "trace_read_error",
            "malformed_trace_line",
            "invalid_timestamp",
            "overlapping_unfinished_syscall",
            "unfinished_syscall_without_resume",
            "resumed_syscall_without_start",
            "malformed_resumed_syscall",
            "malformed_relevant_syscall",
            "truncated_or_invalid_exec_path",
            "truncated_or_invalid_argv",
            "truncated_or_invalid_chdir_path",
        ];
        let lifecycle_codes = [
            "trace_root_unresolved",
            "expected_root_not_observed",
            "unresolved_process_parent",
            "process_escape_or_unscoped_task",
            "unresolved_process_lifecycle",
            "invalid_process_interval",
            "unresolved_process_binary",
            "unresolved_process_argv",
            "unresolved_process_cwd",
            "unresolved_thread_lifecycle",
        ];
        let is_loss = |code: &String| loss_codes.contains(&code.as_str());
        let is_lifecycle = |code: &String| lifecycle_codes.contains(&code.as_str());
        let successful_exec_count: usize = self
            .processes
            .values()
            .map(|process| process.execs.len())
            .sum();
        let failed_exec_count: usize = self
            .processes
            .values()
            .map(|process| process.failed_execs.len())
            .sum();
        json!({
            "schema_version": PROCESS_EXEC_SCHEMA_VERSION,
            "method": PROCESS_EXEC_METHOD,
            "source_clock": "realtime",
            "timestamp_encoding": "decimal-epoch-seconds-converted-exactly-to-nanoseconds",
            "scope": {
                "kind": "trace-root-and-descendants",
                "root_pid": root_pid,
                "root_inference": root_inference,
                "process_escape_detected": !process_escape_pids.is_empty(),
                "process_escape_pids": process_escape_pids,
                "unresolved_parent_pids": unresolved_parent_pids,
            },
            "valid": anomalies.is_empty(),
            "trace_complete": !codes.iter().any(|code| is_loss(code) || is_lifecycle(code)),
            "loss": {
                "detected": codes.iter().any(is_loss),
                "split_syscall_count": self.split_syscalls,
                "unfinished_syscall_count": codes.iter().filter(|code| code.as_str() == "unfinished_syscall_without_resume").count(),
                "resumed_without_start_count": codes.iter().filter(|code| code.as_str() == "resumed_syscall_without_start").count(),
                "malformed_or_truncated_count": codes.iter().filter(|code| is_loss(code)).count(),
                "unresolved_process_count": codes.iter().filter(|code| code.as_str() == "unresolved_process_lifecycle").count(),
            },
            "summary": {
                "line_count": self.total_line_count,
                "pid_prefixed_line_count": self.parsed_line_count,
                "process_count": process_rows.len(),
                "thread_count": thread_rows.len(),
                "successful_exec_count": successful_exec_count,
                "failed_exec_count": failed_exec_count,
                "anomaly_count": anomalies.len(),
            },
            "anomalies": anomalies,
            "binary_inventory": binary_inventory,
            "processes": process_rows,
            "threads": thread_rows,
        })
    }
}

impl Parser {
    #[allow(clippy::too_many_arguments)]
    fn handle_exec(
        &mut self,
        pid: u32,
        started_ns: u64,
        completed_ns: u64,
        name: &str,
        arguments: &str,
        result: &str,
        line_number: u64,
    ) {
        let process_pid = self.ensure_task(pid, started_ns);
        let task = self.tasks.get(&pid).expect("ensured task").clone();
        if task.tid != task.process_pid {
            self.anomaly(
                "exec_from_nonleader_thread",
                "exec from a non-leader thread cannot be mapped safely without de-thread evidence",
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
        }
        let parts = split_top_level(arguments);
        let minimum = if name == "execve" { 2 } else { 3 };
        let Some(parts) = parts.filter(|parts| parts.len() >= minimum) else {
            self.anomaly(
                "malformed_exec_arguments",
                format!("unable to split {name} arguments"),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
            return;
        };
        let path_index = usize::from(name == "execveat");
        let argv_index = path_index + 1;
        let invoked_path = parse_c_string(&parts[path_index]);
        let argv = parse_argv(&parts[argv_index]);
        if invoked_path.is_none() {
            self.anomaly(
                "truncated_or_invalid_exec_path",
                format!("{name} path is truncated or is not a complete quoted string"),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
        }
        if argv.is_none() {
            self.anomaly(
                "truncated_or_invalid_argv",
                format!("{name} argv is truncated or malformed"),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
        }
        let cwd = self.cwd(task.cwd_id);
        let dirfd = (name == "execveat").then(|| parts[0].trim().to_owned());
        let flags = (name == "execveat" && parts.len() >= 5).then(|| parts[4].trim().to_owned());
        let (binary, binary_resolution) = resolve_exec_path(
            invoked_path.as_deref(),
            cwd.as_deref(),
            name,
            dirfd.as_deref(),
        );
        if invoked_path.is_some() && binary.is_none() {
            self.anomaly(
                "unresolved_exec_binary",
                format!(
                    "cannot resolve {name} path {} against the observed cwd/dirfd",
                    python_repr(invoked_path.as_deref().unwrap_or_default())
                ),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
        }
        if cwd.is_none() {
            self.anomaly(
                "unresolved_exec_cwd",
                "working directory at exec time is unknown",
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
        }
        let mut event = Map::from_iter([
            ("syscall".to_owned(), Value::String(name.to_owned())),
            ("started_ns".to_owned(), json!(started_ns)),
            ("completed_ns".to_owned(), json!(completed_ns)),
            ("invoked_path".to_owned(), json!(invoked_path)),
            ("binary".to_owned(), json!(binary)),
            (
                "binary_resolution".to_owned(),
                Value::String(binary_resolution.to_owned()),
            ),
            ("argv".to_owned(), json!(argv)),
            ("cwd".to_owned(), json!(cwd)),
        ]);
        if name == "execveat" {
            event.insert("dirfd".to_owned(), json!(dirfd));
            event.insert("flags".to_owned(), json!(flags));
        }
        let result_value = integer_result(result);
        if result_value == Some(0) {
            let process = self
                .processes
                .get_mut(&process_pid)
                .expect("ensured process");
            process.execs.push(Value::Object(event));
            process.binary = binary;
            process.argv = argv;
            process.exec_cwd = cwd;
            process.binary_source = Some("successful-exec".to_owned());
        } else if result.starts_with("-1") {
            event.insert("errno".to_owned(), json!(errno(result)));
            event.insert("result".to_owned(), Value::String(result.to_owned()));
            self.processes
                .get_mut(&process_pid)
                .expect("ensured process")
                .failed_execs
                .push(Value::Object(event));
        } else {
            self.anomaly(
                "ambiguous_exec_result",
                format!(
                    "cannot determine whether {name} succeeded from result {}",
                    python_repr(result)
                ),
                Some(line_number),
                Some(pid),
                Some(completed_ns),
            );
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn handle_creation(
        &mut self,
        pid: u32,
        started_ns: u64,
        completed_ns: u64,
        name: &str,
        arguments: &str,
        result: &str,
        line_number: u64,
    ) {
        let caller_process_pid = self.ensure_task(pid, started_ns);
        let caller = self.tasks.get(&pid).expect("ensured task").clone();
        let Some(child_value) = integer_result(result) else {
            self.anomaly(
                "ambiguous_process_creation_result",
                format!("cannot parse {name} result {}", python_repr(result)),
                Some(line_number),
                Some(pid),
                Some(completed_ns),
            );
            return;
        };
        if child_value < 0 {
            return;
        }
        if child_value == 0 {
            self.anomaly(
                "child_side_creation_result",
                format!(
                    "{name} returned zero in a task-prefixed trace; parent relation is unresolved"
                ),
                Some(line_number),
                Some(pid),
                Some(completed_ns),
            );
            return;
        }
        let Ok(child_tid) = u32::try_from(child_value) else {
            self.anomaly(
                "ambiguous_process_creation_result",
                format!("cannot parse {name} result {}", python_repr(result)),
                Some(line_number),
                Some(pid),
                Some(completed_ns),
            );
            return;
        };
        let flags = clone_flags(name, arguments);
        if matches!(name, "clone" | "clone3") && flags.is_none() {
            self.anomaly(
                "unresolved_clone_flags",
                format!(
                    "cannot determine whether {name} result {child_tid} is a process or thread"
                ),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
        }
        let is_thread = flags
            .as_ref()
            .is_some_and(|flags| flags.contains("CLONE_THREAD"));
        let share_cwd = flags
            .as_ref()
            .is_some_and(|flags| flags.contains("CLONE_FS"));
        let caller_process = self
            .processes
            .get(&caller_process_pid)
            .expect("ensured process")
            .clone();
        let caller_cwd = self.cwd(caller.cwd_id);

        if is_thread {
            if let Some(existing) = self.tasks.get(&child_tid).cloned() {
                let provisional = self.processes.get(&existing.process_pid).cloned();
                let referenced_as_parent = self
                    .processes
                    .values()
                    .any(|process| process.parent_pid == Some(child_tid));
                let reconcilable = existing.process_pid == child_tid
                    && provisional.as_ref().is_some_and(|process| {
                        process.parent_pid.is_none()
                            && process.execs.is_empty()
                            && process.failed_execs.is_empty()
                            && process.cwd_changes.is_empty()
                    })
                    && !referenced_as_parent;
                if reconcilable {
                    let provisional = provisional.expect("checked above");
                    self.processes.remove(&child_tid);
                    if self.cwd(provisional.cwd_id).is_none() {
                        self.set_cwd(provisional.cwd_id, caller_cwd.clone());
                    }
                    for task in self.tasks.values_mut() {
                        if task.process_pid == child_tid {
                            task.process_pid = caller_process_pid;
                        }
                    }
                    let cwd_id = if share_cwd {
                        caller.cwd_id
                    } else {
                        self.new_cwd(caller_cwd)
                    };
                    let task = self.tasks.get_mut(&child_tid).expect("existing task");
                    task.cwd_id = cwd_id;
                    task.start_ns = Some(task.start_ns.unwrap_or(started_ns).min(started_ns));
                    task.first_seen_ns = task.first_seen_ns.min(started_ns);
                    return;
                }
                self.anomaly(
                    "late_or_conflicting_thread_relation",
                    format!("thread {child_tid} was observed before or conflicts with its clone relation"),
                    Some(line_number),
                    Some(pid),
                    Some(started_ns),
                );
                return;
            }
            let cwd_id = if share_cwd {
                caller.cwd_id
            } else {
                self.new_cwd(caller_cwd)
            };
            self.tasks.insert(
                child_tid,
                TaskState {
                    tid: child_tid,
                    process_pid: caller_process_pid,
                    cwd_id,
                    first_seen_ns: started_ns,
                    start_ns: Some(started_ns),
                    end_ns: None,
                    exit_code: None,
                    exit_signal: None,
                },
            );
            return;
        }

        let mut parent_pid = Some(caller_process_pid);
        let mut parent_resolution = "creation-edge".to_owned();
        if flags
            .as_ref()
            .is_some_and(|flags| flags.contains("CLONE_PARENT"))
        {
            if caller_process.parent_pid.is_none() {
                parent_pid = None;
                parent_resolution = "unresolved-clone-parent".to_owned();
                self.anomaly(
                    "unresolved_clone_parent",
                    format!("CLONE_PARENT child {child_tid} has no resolvable traced parent"),
                    Some(line_number),
                    Some(pid),
                    Some(started_ns),
                );
            } else {
                parent_pid = caller_process.parent_pid;
                parent_resolution = "clone-parent-edge".to_owned();
            }
        }
        let child_cwd_id = if share_cwd {
            caller.cwd_id
        } else {
            self.new_cwd(caller_cwd.clone())
        };
        let existing_task = self.tasks.get(&child_tid).cloned();
        if existing_task
            .as_ref()
            .is_some_and(|task| task.process_pid != child_tid)
        {
            self.anomaly(
                "pid_task_collision",
                format!("process result {child_tid} collides with a known thread task"),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
            return;
        }
        if let std::collections::btree_map::Entry::Vacant(entry) = self.processes.entry(child_tid) {
            entry.insert(ProcessState {
                pid: child_tid,
                parent_pid,
                parent_resolution,
                start_ns: started_ns,
                start_source: format!("{name}-result"),
                first_seen_ns: started_ns,
                cwd_id: child_cwd_id,
                creation_syscall: Some(name.to_owned()),
                creation_completed_ns: Some(completed_ns),
                binary: caller_process.binary.clone(),
                argv: caller_process.argv.clone(),
                exec_cwd: caller_cwd,
                binary_source: caller_process
                    .binary
                    .is_some()
                    .then(|| "inherited-from-parent".to_owned()),
                end_ns: None,
                exit_code: None,
                exit_signal: None,
                execs: Vec::new(),
                failed_execs: Vec::new(),
                cwd_changes: Vec::new(),
            });
            self.tasks.insert(
                child_tid,
                TaskState {
                    tid: child_tid,
                    process_pid: child_tid,
                    cwd_id: child_cwd_id,
                    first_seen_ns: started_ns,
                    start_ns: Some(started_ns),
                    end_ns: None,
                    exit_code: None,
                    exit_signal: None,
                },
            );
            return;
        }

        let conflicting_parent = self
            .processes
            .get(&child_tid)
            .and_then(|process| process.parent_pid)
            .is_some_and(|existing| Some(existing) != parent_pid);
        if conflicting_parent {
            self.anomaly(
                "conflicting_parent_relation",
                format!("process {child_tid} has conflicting parent relations"),
                Some(line_number),
                Some(pid),
                Some(started_ns),
            );
            return;
        }
        let successful_cwd_change = self
            .processes
            .get(&child_tid)
            .expect("existing child")
            .cwd_changes
            .iter()
            .any(|event| event.get("success").and_then(Value::as_bool) == Some(true));
        {
            let child = self.processes.get_mut(&child_tid).expect("existing child");
            child.parent_pid = parent_pid;
            child.parent_resolution = parent_resolution;
            child.start_ns = child.start_ns.min(started_ns);
            child.start_source = format!("{name}-result");
            child.creation_syscall = Some(name.to_owned());
            child.creation_completed_ns = Some(completed_ns);
        }
        if name == "vfork"
            && existing_task
                .as_ref()
                .is_some_and(|task| self.cwd(task.cwd_id).is_none())
            && caller_cwd.is_some()
            && !successful_cwd_change
        {
            let mut repaired = BTreeSet::new();
            let child = self.processes.get_mut(&child_tid).expect("existing child");
            for event in child.failed_execs.iter_mut().chain(child.execs.iter_mut()) {
                let event_started = event.get("started_ns").and_then(Value::as_u64);
                if event.get("cwd").is_some_and(Value::is_null)
                    && event_started.is_some_and(|timestamp| {
                        started_ns <= timestamp && timestamp <= completed_ns
                    })
                {
                    event
                        .as_object_mut()
                        .expect("event object")
                        .insert("cwd".to_owned(), json!(caller_cwd));
                    repaired.insert(event_started.expect("checked above"));
                }
            }
            if !repaired.is_empty() {
                self.anomalies.retain(|anomaly| {
                    !(anomaly.code == "unresolved_exec_cwd"
                        && anomaly.pid == Some(child_tid)
                        && anomaly
                            .timestamp_ns
                            .is_some_and(|time| repaired.contains(&time)))
                });
                if let Some(cwd) = child
                    .execs
                    .last()
                    .and_then(|event| event.get("cwd"))
                    .and_then(Value::as_str)
                {
                    child.exec_cwd = Some(cwd.to_owned());
                }
            }
        }
        {
            let child = self.processes.get_mut(&child_tid).expect("existing child");
            if child.binary.is_none() && child.execs.is_empty() {
                child.binary = caller_process.binary.clone();
                child.argv = caller_process.argv.clone();
                child.exec_cwd = caller_cwd.clone();
                child.binary_source = caller_process
                    .binary
                    .is_some()
                    .then(|| "inherited-from-parent".to_owned());
            }
        }
        if existing_task
            .as_ref()
            .is_some_and(|task| self.cwd(task.cwd_id).is_none())
            && !successful_cwd_change
        {
            let task = self.tasks.get_mut(&child_tid).expect("existing task");
            task.cwd_id = child_cwd_id;
            self.processes
                .get_mut(&child_tid)
                .expect("existing child")
                .cwd_id = child_cwd_id;
        }
    }

    fn handle_chdir(
        &mut self,
        pid: u32,
        timestamp_ns: u64,
        arguments: &str,
        result: &str,
        line_number: u64,
    ) {
        let process_pid = self.ensure_task(pid, timestamp_ns);
        let task = self.tasks.get(&pid).expect("ensured task").clone();
        let parts = split_top_level(arguments);
        let path = parts
            .as_ref()
            .filter(|parts| parts.len() == 1)
            .and_then(|parts| parse_c_string(&parts[0]));
        let Some(path) = path else {
            self.anomaly(
                "truncated_or_invalid_chdir_path",
                "chdir path is truncated or malformed",
                Some(line_number),
                Some(pid),
                Some(timestamp_ns),
            );
            return;
        };
        let result_value = integer_result(result);
        let old_cwd = self.cwd(task.cwd_id);
        let new_cwd = resolve_cwd(&path, old_cwd.as_deref());
        let event = json!({
            "syscall": "chdir",
            "timestamp_ns": timestamp_ns,
            "path": path,
            "from_cwd": old_cwd,
            "to_cwd": new_cwd,
            "success": result_value == Some(0),
        });
        if result_value == Some(0) {
            if new_cwd.is_none() {
                self.anomaly(
                    "unresolved_chdir_cwd",
                    format!(
                        "cannot resolve relative chdir path {} without a known cwd",
                        python_repr(&path)
                    ),
                    Some(line_number),
                    Some(pid),
                    Some(timestamp_ns),
                );
            }
            self.set_cwd(task.cwd_id, new_cwd);
        } else if result_value.is_none() {
            self.anomaly(
                "ambiguous_chdir_result",
                format!("cannot determine chdir result from {}", python_repr(result)),
                Some(line_number),
                Some(pid),
                Some(timestamp_ns),
            );
        }
        self.processes
            .get_mut(&process_pid)
            .expect("ensured process")
            .cwd_changes
            .push(event);
    }

    fn handle_fchdir(
        &mut self,
        pid: u32,
        timestamp_ns: u64,
        arguments: &str,
        result: &str,
        line_number: u64,
    ) {
        let process_pid = self.ensure_task(pid, timestamp_ns);
        let task = self.tasks.get(&pid).expect("ensured task").clone();
        let result_value = integer_result(result);
        let old_cwd = self.cwd(task.cwd_id);
        let event = json!({
            "syscall": "fchdir",
            "timestamp_ns": timestamp_ns,
            "fd": arguments.trim(),
            "from_cwd": old_cwd,
            "to_cwd": Value::Null,
            "success": result_value == Some(0),
        });
        if result_value == Some(0) {
            self.set_cwd(task.cwd_id, None);
            self.anomaly(
                "unresolved_fchdir_cwd",
                "successful fchdir cannot be resolved without file-descriptor evidence",
                Some(line_number),
                Some(pid),
                Some(timestamp_ns),
            );
        } else if result_value.is_none() {
            self.anomaly(
                "ambiguous_fchdir_result",
                format!(
                    "cannot determine fchdir result from {}",
                    python_repr(result)
                ),
                Some(line_number),
                Some(pid),
                Some(timestamp_ns),
            );
        }
        self.processes
            .get_mut(&process_pid)
            .expect("ensured process")
            .cwd_changes
            .push(event);
    }

    fn handle_exit(
        &mut self,
        pid: u32,
        timestamp_ns: u64,
        exit_code: Option<i64>,
        exit_signal: Option<String>,
        line_number: u64,
    ) {
        let process_pid = self.ensure_task(pid, timestamp_ns);
        if self
            .tasks
            .get(&pid)
            .is_some_and(|task| task.end_ns.is_some())
        {
            self.anomaly(
                "duplicate_task_exit",
                format!("task {pid} has more than one terminal record"),
                Some(line_number),
                Some(pid),
                Some(timestamp_ns),
            );
            return;
        }
        let leader = pid == process_pid;
        let task = self.tasks.get_mut(&pid).expect("ensured task");
        task.end_ns = Some(timestamp_ns);
        task.exit_code = exit_code;
        task.exit_signal = exit_signal.clone();
        if leader {
            let process = self
                .processes
                .get_mut(&process_pid)
                .expect("ensured process");
            process.end_ns = Some(timestamp_ns);
            process.exit_code = exit_code;
            process.exit_signal = exit_signal;
        }
    }
}
