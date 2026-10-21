// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Native ATIF trajectory replay.
//!
//! The module deliberately mirrors the established Python adapter instead of
//! inventing a more permissive ATIF interpreter.  It executes only the tool
//! dialects whose offline semantics are defined by this repository, preserves
//! source-outcome mismatches as evidence, and stops after the first timeout.

use crate::timebase::{
    clock_reference, monotonic_ns, realtime_midpoint_and_uncertainty, realtime_ns,
};
use flate2::read::GzDecoder;
use ignore::{overrides::OverrideBuilder, WalkBuilder};
use regex::{Regex, RegexBuilder};
use serde_json::{json, Map, Number, Value};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::env;
use std::error::Error;
use std::ffi::{OsStr, OsString};
use std::fmt;
use std::fs::{self, File};
use std::io::{self, Read, Write};
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use std::thread;
use std::time::{Duration, Instant};

pub const REPLAY_SCHEMA_VERSION: &str = "offline-atif-trajectory-replay.v1";
pub const PACING_MODES: &[&str] = &["recorded", "unpaced", "llm-gaps"];
const DEFAULT_WRITE_STDIN_YIELD_SEC: f64 = 5.0;
const PREVIEW_LENGTH: usize = 1000;
const MINI_SWE_AGENT_ORIGINAL_FORMAT: &str = "mini-swe-agent-1.1";
const CLAUDE_CODE_AGENT_NAME: &str = "claude-code";
const CONTROLLER_ENV_VARS_MARKER: &str = "TASKTRACE_CONTROLLER_ENV_VARS";
const WORKLOAD_ENV_PREFIX: &str = "TASKTRACE_WORKLOAD_ENV_";
const WORKLOAD_ENV_PRESENT_PREFIX: &str = "TASKTRACE_WORKLOAD_ENV_PRESENT_";
const POLL_INTERVAL: Duration = Duration::from_millis(2);
const TIMEOUT_TERMINATION_GRACE: Duration = Duration::from_secs(2);
const TIMEOUT_KILL_SETTLE: Duration = Duration::from_secs(2);

pub(crate) fn is_supported_tool(tool: &str) -> bool {
    matches!(
        tool,
        "exec_command"
            | "write_stdin"
            | "apply_patch"
            | "claude_write"
            | "claude_edit"
            | "claude_read"
            | "claude_grep"
            | "claude_glob"
            | "claude_task_stop"
            | "claude_task_output"
    )
}

pub(crate) fn is_source_executed_supported_action(action: &AtifAction) -> bool {
    is_supported_tool(&action.tool) && source_outcome(action).status != "skipped_not_executed"
}

fn is_metadata_tool(tool: &str) -> bool {
    matches!(tool, "update_plan" | "view_image" | "web_search_call")
}

fn is_code_mode_metadata_tool(tool: &str) -> bool {
    is_metadata_tool(tool)
        || matches!(
            tool,
            "mcp__codex_apps__github_fetch_issue"
                | "mcp__codex_apps__github_fetch_issue_comments"
                | "mcp__codex_apps__github_fetch_pr"
                | "mcp__codex_apps__github_fetch_pr_comments"
                | "mcp__codex_apps__github_fetch_pr_patch"
                | "mcp__codex_apps__github_search_issues"
                | "mcp__codex_apps__github_search_prs"
                | "web__run"
                | "spawn_agent"
                | "send_message"
                | "followup_task"
                | "wait_agent"
                | "list_agents"
                | "interrupt_agent"
                | "wait"
                | "get_goal"
                | "create_goal"
                | "update_goal"
                | "notify"
                | "yield_control"
        )
}

fn is_claude_code_metadata_tool(tool: &str) -> bool {
    matches!(
        tool,
        "TaskCreate"
            | "TaskUpdate"
            | "TaskList"
            | "TaskGet"
            | "ToolSearch"
            | "ScheduleWakeup"
            | "WebFetch"
            | "WebSearch"
            | "TodoWrite"
            | "AskUserQuestion"
            | "Agent"
            | "Monitor"
            | "claude_orchestration"
    )
}

/// A replay failure that is safe to display at the public CLI boundary.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ReplayError(String);

impl ReplayError {
    fn new(message: impl Into<String>) -> Self {
        Self(message.into())
    }
}

impl fmt::Display for ReplayError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for ReplayError {}

impl From<io::Error> for ReplayError {
    fn from(error: io::Error) -> Self {
        Self(error.to_string())
    }
}

impl From<serde_json::Error> for ReplayError {
    fn from(error: serde_json::Error) -> Self {
        Self(error.to_string())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtifAction {
    pub index: usize,
    pub step_id: Value,
    pub timestamp_sec: f64,
    pub source_trajectory_id: String,
    pub source_item_id: Option<String>,
    pub tool: String,
    pub source_tool: String,
    pub code_mode: bool,
    pub parallel_group_id: Option<String>,
    pub source_agent_original_format: Option<String>,
    pub arguments: Map<String, Value>,
    pub observation: Value,
    pub observation_text: String,
}

#[derive(Clone)]
struct CodeModeAction {
    tool: String,
    arguments: Map<String, Value>,
    parallel_group_id: Option<String>,
    result_print: ResultPrint,
}

struct CodeModeActions {
    actions: Vec<CodeModeAction>,
    // Each proven single text() print identifies an action or display-only data.
    // None means a conditional/derived presentation has no proven print layout.
    direct_prints: Option<Vec<Option<usize>>>,
}

#[derive(Clone, Copy, PartialEq)]
enum ResultPrint {
    None,
    Direct,
    Indexed,
    Settled,
}

fn is_metadata_action(action: &AtifAction) -> bool {
    is_metadata_tool(&action.tool)
        || (action.code_mode && is_code_mode_metadata_tool(&action.tool))
        || (action.source_agent_original_format.as_deref() == Some(CLAUDE_CODE_AGENT_NAME)
            && is_claude_code_metadata_tool(&action.tool))
}

#[derive(Clone, Debug, PartialEq)]
pub struct AtifTrajectory {
    pub source_path: PathBuf,
    pub schema_version: String,
    pub agent: Map<String, Value>,
    pub source_agent_original_format: Option<String>,
    pub session_id: Option<String>,
    pub subagent_trajectories_flattened: usize,
    pub actions: Vec<AtifAction>,
    // Retain node provenance even when a nested Codex exec node has no actions.
    pub contains_codex_exec: bool,
}

impl AtifTrajectory {
    pub fn trial_name(&self) -> String {
        self.source_path
            .parent()
            .and_then(Path::parent)
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default()
    }

    pub fn task_name(&self) -> String {
        let trial = self.trial_name();
        trial
            .rsplit_once("__")
            .map(|(task, _)| task.to_owned())
            .unwrap_or(trial)
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Pacing {
    Recorded,
    Unpaced,
    LlmGaps,
}

impl Pacing {
    fn parse(value: &str) -> Result<Self, ReplayError> {
        let value = crate::pacing::canonical(value);
        if !PACING_MODES.contains(&value) {
            return Err(ReplayError::new(format!(
                "unsupported pacing mode: {value:?}"
            )));
        }
        match value {
            "recorded" => Ok(Self::Recorded),
            "unpaced" => Ok(Self::Unpaced),
            "llm-gaps" => Ok(Self::LlmGaps),
            _ => unreachable!("validated ATIF pacing mode"),
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Recorded => "recorded",
            Self::Unpaced => "unpaced",
            Self::LlmGaps => "llm-gaps",
        }
    }
}

/// Runtime options for [`replay_trajectory`].
#[derive(Clone, Debug)]
pub struct ReplayOptions {
    pub cwd: PathBuf,
    pub pacing: Pacing,
    pub timeout_sec: Option<f64>,
    pub artifacts_dir: Option<PathBuf>,
    pub action_output_dir: Option<PathBuf>,
    pub patch_roots: Vec<PathBuf>,
}

impl ReplayOptions {
    pub fn new(cwd: impl Into<PathBuf>) -> Self {
        Self {
            cwd: cwd.into(),
            pacing: Pacing::Recorded,
            timeout_sec: None,
            artifacts_dir: None,
            action_output_dir: None,
            patch_roots: Vec::new(),
        }
    }
}

fn parse_iso_timestamp(value: &str) -> Result<f64, ReplayError> {
    // ATIF recordings use RFC3339 timestamps. Python's fromisoformat accepts
    // either `T` or a space and optional offsets, which this expression keeps.
    let expression = Regex::new(
        r"^(\d{4})-(\d{2})-(\d{2})[T ](\d{2}):(\d{2}):(\d{2}(?:\.\d+)?)(Z|[+-]\d{2}(?::?\d{2})?)?$",
    )
    .expect("static timestamp regex");
    let captures = expression
        .captures(value)
        .ok_or_else(|| ReplayError::new("invalid ISO timestamp"))?;
    let number = |index: usize| -> Result<i64, ReplayError> {
        captures[index]
            .parse::<i64>()
            .map_err(|_| ReplayError::new("invalid ISO timestamp"))
    };
    let year = number(1)?;
    let month = number(2)?;
    let day = number(3)?;
    let hour = number(4)?;
    let minute = number(5)?;
    let second = captures[6]
        .parse::<f64>()
        .map_err(|_| ReplayError::new("invalid ISO timestamp"))?;
    if !(1..=12).contains(&month)
        || !(1..=31).contains(&day)
        || !(0..=23).contains(&hour)
        || !(0..=59).contains(&minute)
        || !(0.0..61.0).contains(&second)
    {
        return Err(ReplayError::new("invalid ISO timestamp"));
    }

    // Howard Hinnant's civil-date conversion, yielding days since 1970-01-01.
    let adjusted_year = year - i64::from(month <= 2);
    let era = adjusted_year.div_euclid(400);
    let year_of_era = adjusted_year - era * 400;
    let shifted_month = month + if month > 2 { -3 } else { 9 };
    let day_of_year = (153 * shifted_month + 2) / 5 + day - 1;
    let day_of_era = year_of_era * 365 + year_of_era / 4 - year_of_era / 100 + day_of_year;
    let days = era * 146_097 + day_of_era - 719_468;
    let mut timestamp =
        days as f64 * 86_400.0 + hour as f64 * 3_600.0 + minute as f64 * 60.0 + second;
    if let Some(zone) = captures.get(7).map(|capture| capture.as_str()) {
        if zone != "Z" {
            let sign = if zone.starts_with('-') { -1.0 } else { 1.0 };
            let digits = &zone[1..];
            let (hours, minutes) = if let Some((hours, minutes)) = digits.split_once(':') {
                (hours, minutes)
            } else if digits.len() == 4 {
                (&digits[..2], &digits[2..])
            } else {
                (digits, "0")
            };
            let hours = hours
                .parse::<u32>()
                .map_err(|_| ReplayError::new("invalid ISO timestamp"))?;
            let minutes = minutes
                .parse::<u32>()
                .map_err(|_| ReplayError::new("invalid ISO timestamp"))?;
            if hours > 23 || minutes > 59 {
                return Err(ReplayError::new("invalid ISO timestamp"));
            }
            timestamp -= sign * (hours * 3_600 + minutes * 60) as f64;
        }
    }
    Ok(timestamp)
}

fn timestamp_sec(value: &Value, action_index: usize) -> Result<f64, ReplayError> {
    if let Some(value) = value.as_f64() {
        return Ok(value);
    }
    if let Some(value) = value.as_str() {
        return parse_iso_timestamp(value).map_err(|_| {
            ReplayError::new(format!(
                "ATIF action {action_index} has invalid timestamp {value:?}"
            ))
        });
    }
    Err(ReplayError::new(format!(
        "ATIF action {action_index} has no valid timestamp"
    )))
}

fn python_json_string(value: &str) -> String {
    let mut encoded = String::from("\"");
    for character in value.chars() {
        match character {
            '"' => encoded.push_str("\\\""),
            '\\' => encoded.push_str("\\\\"),
            '\u{08}' => encoded.push_str("\\b"),
            '\u{0c}' => encoded.push_str("\\f"),
            '\n' => encoded.push_str("\\n"),
            '\r' => encoded.push_str("\\r"),
            '\t' => encoded.push_str("\\t"),
            character if (' '..='~').contains(&character) => encoded.push(character),
            character if (character as u32) <= 0xffff => {
                encoded.push_str(&format!("\\u{:04x}", character as u32));
            }
            character => {
                let scalar = character as u32 - 0x1_0000;
                let high = 0xd800 + (scalar >> 10);
                let low = 0xdc00 + (scalar & 0x3ff);
                encoded.push_str(&format!("\\u{high:04x}\\u{low:04x}"));
            }
        }
    }
    encoded.push('"');
    encoded
}

fn python_json_dumps(value: &Value) -> String {
    match value {
        Value::Null => "null".to_owned(),
        Value::Bool(value) => if *value { "true" } else { "false" }.to_owned(),
        Value::Number(value) => value.to_string(),
        Value::String(value) => python_json_string(value),
        Value::Array(values) => format!(
            "[{}]",
            values
                .iter()
                .map(python_json_dumps)
                .collect::<Vec<_>>()
                .join(", ")
        ),
        Value::Object(values) => {
            let mut fields: Vec<_> = values.iter().collect();
            fields.sort_by(|(left, _), (right, _)| left.cmp(right));
            format!(
                "{{{}}}",
                fields
                    .into_iter()
                    .map(|(key, value)| format!(
                        "{}: {}",
                        python_json_string(key),
                        python_json_dumps(value)
                    ))
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        }
    }
}

fn text_content(value: &Value) -> String {
    match value {
        Value::String(value) => value.clone(),
        Value::Object(value) => ["content", "text", "output"]
            .iter()
            .find_map(|key| value.get(*key).and_then(Value::as_str))
            .map(str::to_owned)
            .unwrap_or_else(|| python_json_dumps(&Value::Object(value.clone()))),
        Value::Array(values) => values
            .iter()
            .map(text_content)
            .collect::<Vec<_>>()
            .join("\n"),
        Value::Null => String::new(),
        value => python_json_dumps(value),
    }
}

fn source_agent_original_format(agent: &Map<String, Value>) -> Option<String> {
    agent
        .get("extra")
        .and_then(Value::as_object)
        .and_then(|extra| extra.get("original_format"))
        .and_then(Value::as_str)
        .filter(|value| !value.is_empty())
        .map(str::to_owned)
}

fn is_codex_code_mode(agent: &Map<String, Value>) -> bool {
    agent.get("name").and_then(Value::as_str) == Some("codex")
        && agent
            .get("extra")
            .and_then(Value::as_object)
            .and_then(|extra| extra.get("originator"))
            .and_then(Value::as_str)
            == Some("codex_exec")
}

fn is_claude_code(agent: &Map<String, Value>) -> bool {
    agent.get("name").and_then(Value::as_str) == Some(CLAUDE_CODE_AGENT_NAME)
}

fn contains_identifier(text: &str, wanted: &str) -> bool {
    let bytes = text.as_bytes();
    let identifier = |byte: u8| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'$');
    let mut at = 0usize;
    let mut quote = None;
    while at < bytes.len() {
        if let Some(mark) = quote {
            if bytes[at] == b'\\' {
                at += 2;
                continue;
            }
            if bytes[at] == mark {
                quote = None;
            }
            at += 1;
            continue;
        }
        if matches!(bytes[at], b'\'' | b'"' | b'`') {
            quote = Some(bytes[at]);
            at += 1;
            continue;
        }
        if bytes.get(at..at + wanted.len()) == Some(wanted.as_bytes())
            && !at
                .checked_sub(1)
                .and_then(|index| bytes.get(index))
                .is_some_and(|byte| identifier(*byte))
            && !bytes
                .get(at + wanted.len())
                .is_some_and(|byte| identifier(*byte))
        {
            return true;
        }
        at += 1;
    }
    false
}

/// Strict parser for sequential Code Mode statements with literal arguments.
struct CodeModeCell<'a> {
    input: &'a str,
    at: usize,
    strings: HashMap<String, String>,
    command_batches: HashMap<String, Vec<Vec<String>>>,
    image_path_results: HashMap<String, String>,
    parallel_groups: usize,
    outer_call_identity: String,
    data_literal: bool,
    indexed_results: bool,
    literal_depth: usize,
}

impl<'a> CodeModeCell<'a> {
    fn new(input: &'a str, outer_call_identity: String) -> Self {
        Self {
            input,
            at: 0,
            strings: HashMap::new(),
            command_batches: HashMap::new(),
            image_path_results: HashMap::new(),
            parallel_groups: 0,
            outer_call_identity,
            data_literal: false,
            indexed_results: false,
            literal_depth: 0,
        }
    }

    fn error(&self, message: &str) -> ReplayError {
        ReplayError::new(format!("unsupported Codex exec payload: {message}"))
    }

    fn rest(&self) -> &'a str {
        &self.input[self.at..]
    }

    fn skip_space(&mut self) {
        while let Some(byte) = self.input.as_bytes().get(self.at) {
            if byte.is_ascii_whitespace() {
                self.at += 1;
            } else {
                break;
            }
        }
    }

    fn eat(&mut self, value: &str) -> bool {
        if self.rest().starts_with(value) {
            self.at += value.len();
            true
        } else {
            false
        }
    }

    fn expect(&mut self, value: &str, message: &str) -> Result<(), ReplayError> {
        self.skip_space();
        if self.eat(value) {
            Ok(())
        } else {
            Err(self.error(message))
        }
    }

    // The opening delimiter has already been consumed. Quoted delimiters do not nest.
    fn balanced_end(
        &mut self,
        opening: u8,
        closing: u8,
        message: &str,
    ) -> Result<usize, ReplayError> {
        let mut depth = 1usize;
        let mut quote = None;
        while let Some(byte) = self.input.as_bytes().get(self.at).copied() {
            self.at += 1;
            if let Some(mark) = quote {
                if byte == b'\\' {
                    self.at += 1;
                } else if byte == mark {
                    quote = None;
                }
            } else if matches!(byte, b'\'' | b'"' | b'`') {
                quote = Some(byte);
            } else if byte == opening {
                depth += 1;
            } else if byte == closing {
                depth -= 1;
                if depth == 0 {
                    return Ok(self.at - 1);
                }
            }
        }
        Err(self.error(message))
    }

    fn identifier(&mut self) -> Option<String> {
        let start = self.at;
        let bytes = self.input.as_bytes();
        if !bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_alphabetic() || *b == b'_' || *b == b'$')
        {
            return None;
        }
        self.at += 1;
        while bytes
            .get(self.at)
            .is_some_and(|b| b.is_ascii_alphanumeric() || *b == b'_' || *b == b'$')
        {
            self.at += 1;
        }
        Some(self.input[start..self.at].to_owned())
    }

    fn escaped_scalar(&mut self, width: usize, error: &str) -> Result<char, ReplayError> {
        let end = self.at + width;
        let digits = self
            .input
            .get(self.at..end)
            .ok_or_else(|| self.error(error))?;
        let scalar = u32::from_str_radix(digits, 16).map_err(|_| self.error(error))?;
        let character = char::from_u32(scalar).ok_or_else(|| self.error(error))?;
        self.at = end;
        Ok(character)
    }

    fn string(&mut self) -> Result<String, ReplayError> {
        let quote = self
            .rest()
            .chars()
            .next()
            .ok_or_else(|| self.error("unterminated string"))?;
        if !matches!(quote, '\'' | '\"' | '`') {
            return Err(self.error("string literal expected"));
        }
        self.at += quote.len_utf8();
        let mut result = String::new();
        while let Some(character) = self.rest().chars().next() {
            self.at += character.len_utf8();
            if character == quote {
                return Ok(result);
            }
            if character == '$' && quote == '`' && self.rest().starts_with('{') {
                return Err(self.error("template interpolation is not supported"));
            }
            if character != '\\' {
                result.push(character);
                continue;
            }
            let escaped = self
                .rest()
                .chars()
                .next()
                .ok_or_else(|| self.error("unterminated escape"))?;
            self.at += escaped.len_utf8();
            match escaped {
                'n' => result.push('\n'),
                'r' => result.push('\r'),
                't' => result.push('\t'),
                'b' => result.push('\u{0008}'),
                'f' => result.push('\u{000c}'),
                'v' => result.push('\u{000b}'),
                '0' => result.push('\0'),
                '\\' => result.push('\\'),
                '/' => result.push('/'),
                '\'' => result.push('\''),
                '\"' => result.push('\"'),
                '`' => result.push('`'),
                'x' => result.push(self.escaped_scalar(2, "invalid hex escape")?),
                'u' => result.push(self.escaped_scalar(4, "invalid unicode escape")?),
                'U' if self.data_literal => {
                    result.push(self.escaped_scalar(8, "invalid unicode escape")?)
                }
                _ => return Err(self.error("unsupported string escape")),
            }
        }
        Err(self.error("unterminated string"))
    }

    fn literal(&mut self) -> Result<Value, ReplayError> {
        self.skip_space();
        self.literal_depth += 1;
        if self.literal_depth > 128 {
            return Err(self.error("literal nesting exceeds limit"));
        }
        let result = match self.input.as_bytes().get(self.at).copied() {
            Some(b'\'') | Some(b'\"') | Some(b'`') => Ok(Value::String(self.string()?)),
            Some(b'{') => self.object(),
            Some(b'[') => self.array(),
            Some(b'-' | b'0'..=b'9') => self.number(),
            _ => {
                let word = self
                    .identifier()
                    .ok_or_else(|| self.error("arguments must be literals"))?;
                match word.as_str() {
                    "true" => Ok(Value::Bool(true)),
                    "false" => Ok(Value::Bool(false)),
                    "null" => Ok(Value::Null),
                    "True" if self.data_literal => Ok(Value::Bool(true)),
                    "False" if self.data_literal => Ok(Value::Bool(false)),
                    "None" if self.data_literal => Ok(Value::Null),
                    _ => self
                        .strings
                        .get(&word)
                        .cloned()
                        .map(Value::String)
                        .ok_or_else(|| self.error("arguments must be literals")),
                }
            }
        };
        self.literal_depth -= 1;
        result
    }

    fn number(&mut self) -> Result<Value, ReplayError> {
        let start = self.at;
        while self
            .input
            .as_bytes()
            .get(self.at)
            .is_some_and(|b| matches!(*b, b'-' | b'+' | b'.' | b'e' | b'E' | b'0'..=b'9'))
        {
            self.at += 1;
        }
        serde_json::from_str(&self.input[start..self.at])
            .map_err(|_| self.error("invalid number literal"))
    }

    fn object(&mut self) -> Result<Value, ReplayError> {
        self.at += 1;
        let mut map = Map::new();
        loop {
            self.skip_space();
            if self.eat("}") {
                return Ok(Value::Object(map));
            }
            let key = match self.input.as_bytes().get(self.at).copied() {
                Some(b'\'') | Some(b'\"') => self.string()?,
                _ => self
                    .identifier()
                    .ok_or_else(|| self.error("object key must be literal"))?,
            };
            self.skip_space();
            let value = if self.eat(":") {
                self.literal()?
            } else {
                self.strings
                    .get(&key)
                    .cloned()
                    .map(Value::String)
                    .ok_or_else(|| self.error("object shorthand must reference a literal string"))?
            };
            if map.insert(key, value).is_some() && self.data_literal {
                return Err(self.error("duplicate result field"));
            }
            self.skip_space();
            if self.eat("}") {
                return Ok(Value::Object(map));
            }
            if !self.eat(",") {
                return Err(self.error("object property separator expected"));
            }
        }
    }

    fn array(&mut self) -> Result<Value, ReplayError> {
        self.at += 1;
        let mut values = Vec::new();
        loop {
            self.skip_space();
            if self.eat("]") {
                return Ok(Value::Array(values));
            }
            values.push(self.literal()?);
            self.skip_space();
            if self.eat("]") {
                return Ok(Value::Array(values));
            }
            if !self.eat(",") {
                return Err(self.error("array separator expected"));
            }
        }
    }

    fn literal_command_batch(&mut self) -> Result<Option<Vec<Vec<String>>>, ReplayError> {
        if !self.rest().starts_with("[") {
            return Ok(None);
        }
        let Value::Array(entries) = self.array()? else {
            unreachable!()
        };
        let mut rows = Vec::with_capacity(entries.len());
        for entry in entries {
            match entry {
                Value::String(command) => rows.push(vec![command]),
                Value::Array(pair) => {
                    if pair.len() != 2 || pair.iter().any(|value| !value.is_string()) {
                        return Err(self.error("command batch entries must be literal pairs"));
                    }
                    rows.push(
                        pair.into_iter()
                            .map(|value| value.as_str().unwrap().to_owned())
                            .collect(),
                    );
                }
                _ => {
                    return Err(self.error("command batch entries must be literal strings or pairs"))
                }
            }
        }
        if rows.is_empty() {
            return Err(self.error("command batch must be nonempty"));
        }
        Ok(Some(rows))
    }

    fn semicolon(&mut self) -> Result<(), ReplayError> {
        self.skip_space();
        if self.eat(";") || self.at == self.input.len() {
            Ok(())
        } else {
            Err(self.error("statement separator expected"))
        }
    }

    fn ignored_empty_catch(&mut self) -> Result<bool, ReplayError> {
        self.skip_space();
        if !self.eat(".catch") {
            return Ok(false);
        }
        for (token, error) in [
            ("(", "catch missing '('"),
            ("(", "catch handler must have no parameters"),
            (")", "catch handler must have no parameters"),
            ("=>", "catch handler must be an arrow function"),
            ("{", "catch handler must have an empty body"),
            ("}", "catch handler must have an empty body"),
            (")", "catch missing ')'"),
        ] {
            self.expect(token, error)?;
        }
        Ok(true)
    }

    fn ignored_output_statement(&mut self) -> Result<bool, ReplayError> {
        let start = self.at;
        let Some(name) = self.identifier() else {
            return Ok(false);
        };
        if !matches!(name.as_str(), "text" | "image" | "audio") {
            self.at = start;
            return Ok(false);
        }
        self.skip_space();
        if !self.eat("(") {
            self.at = start;
            return Ok(false);
        }
        let contents = self.at;
        let end = self.balanced_end(b'(', b')', "unterminated output helper")?;
        if self.input[contents..end].contains("tools.") {
            return Err(self.error("nested tools call in output helper"));
        }
        self.semicolon()?;
        Ok(true)
    }

    fn data_only_print(statement: &str) -> bool {
        let Some(expression) = statement
            .trim_end_matches(';')
            .strip_prefix("text(")
            .and_then(|text| text.strip_suffix(')'))
        else {
            return false;
        };
        let listing = format!("{};", expression.trim());
        let mut reader = CodeModeCell::new(&listing, String::new());
        if reader.ignored_tool_listing_initializer().unwrap_or(false) {
            return reader.rest().is_empty();
        }
        let mut reader = CodeModeCell::new(expression, String::new());
        if reader.literal().is_err() {
            return false;
        }
        reader.skip_space();
        reader.rest().is_empty()
    }

    fn ignored_tool_listing_initializer(&mut self) -> Result<bool, ReplayError> {
        let start = self.at;
        if !self.eat("ALL_TOOLS") {
            return Ok(false);
        }
        self.expect(".", "ALL_TOOLS listing is missing an operation")?;
        let operation = self
            .identifier()
            .ok_or_else(|| self.error("ALL_TOOLS listing has an invalid operation"))?;
        if !matches!(operation.as_str(), "find" | "filter") {
            return Err(self.error("ALL_TOOLS listing operation is unsupported"));
        }
        self.expect("(", "ALL_TOOLS listing is missing '('")?;
        self.skip_space();
        let parameter = self
            .identifier()
            .ok_or_else(|| self.error("ALL_TOOLS listing callback has an invalid parameter"))?;
        self.expect("=>", "ALL_TOOLS listing must use a direct arrow callback")?;
        self.skip_space();
        if operation == "filter"
            && self
                .rest()
                .starts_with(&format!("{parameter}.name.startsWith"))
        {
            let end = self
                .rest()
                .find(';')
                .ok_or_else(|| self.error("unterminated ALL_TOOLS filter"))?;
            let statement = self.input[start..self.at + end]
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            let prefix = format!("ALL_TOOLS.filter({parameter}=>{parameter}.name.startsWith(\"");
            let after_prefix = statement
                .strip_prefix(&prefix)
                .ok_or_else(|| self.error("ALL_TOOLS listing callback must be display-only"))?;
            let (name_prefix, predicate) = after_prefix
                .split_once("\")&&/")
                .ok_or_else(|| self.error("ALL_TOOLS listing callback must be display-only"))?;
            let insensitive =
                format!("/i.test({parameter}.name)).map({parameter}=>{parameter}.name)");
            let sensitive = format!("/.test({parameter}.name)).map({parameter}=>{parameter}.name)");
            let pattern = predicate
                .strip_suffix(&insensitive)
                .or_else(|| predicate.strip_suffix(&sensitive))
                .ok_or_else(|| self.error("ALL_TOOLS listing callback must be display-only"))?;
            let safe_name_prefix = !name_prefix.is_empty()
                && name_prefix
                    .bytes()
                    .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.'));
            let safe_pattern = !pattern.is_empty()
                && pattern.chars().all(|character| {
                    character.is_ascii_alphanumeric()
                        || matches!(character, '.' | '|' | '*' | '_' | '-' | '(' | ')')
                });
            if !safe_name_prefix || !safe_pattern {
                return Err(self.error("ALL_TOOLS listing callback is not display-only"));
            }
            self.at += end;
            self.semicolon()?;
            return Ok(true);
        }
        if operation == "filter" && self.rest().starts_with('/') {
            let end = self
                .rest()
                .find(';')
                .ok_or_else(|| self.error("unterminated ALL_TOOLS filter"))?;
            let statement = self.input[start..self.at + end]
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            let prefix = format!("ALL_TOOLS.filter({parameter}=>/");
            let inspected_values = [
                format!("{parameter}.name"),
                format!("{parameter}.description"),
                format!("{parameter}.name+\"\"+{parameter}.description"),
                format!("({parameter}.name+\"\"+{parameter}.description).toLowerCase()"),
            ];
            let mut predicate = statement
                .strip_prefix(&prefix)
                .ok_or_else(|| self.error("ALL_TOOLS regex filter has an unsupported predicate"))?;
            loop {
                let (pattern, after_pattern) = predicate.split_once('/').ok_or_else(|| {
                    self.error("ALL_TOOLS regex filter has an unsupported predicate")
                })?;
                let safe_pattern = !pattern.is_empty()
                    && pattern.chars().all(|character| {
                        character.is_ascii_alphanumeric()
                            || matches!(character, '.' | '|' | '*' | '_' | '-' | '(' | ')')
                    });
                if !safe_pattern {
                    return Err(self.error("ALL_TOOLS regex filter is not display-only"));
                }
                let after_flags = after_pattern
                    .strip_prefix("i.test(")
                    .or_else(|| after_pattern.strip_prefix(".test("))
                    .ok_or_else(|| {
                        self.error("ALL_TOOLS regex filter has an unsupported predicate")
                    })?;
                predicate = inspected_values
                    .iter()
                    .find_map(|inspected| after_flags.strip_prefix(inspected)?.strip_prefix(')'))
                    .ok_or_else(|| {
                        self.error("ALL_TOOLS regex filter has an unsupported predicate")
                    })?;
                if let Some(next) = predicate.strip_prefix("&&/") {
                    predicate = next;
                    continue;
                }
                break;
            }
            let suffix = predicate
                .strip_prefix(')')
                .ok_or_else(|| self.error("ALL_TOOLS regex filter has an unsupported predicate"))?;
            let catalog_projection = suffix
                .strip_prefix(&format!(
                    ".map({parameter}=>({{name:{parameter}.name,description:{parameter}.description.slice(0,"
                ))
                .and_then(|value| value.split_once(")})).slice(0,"))
                .is_some_and(|(description_limit, after)| {
                    after.strip_suffix(')').is_some_and(|result_limit| {
                        [description_limit, result_limit].iter().all(|limit| {
                            !limit.is_empty() && limit.bytes().all(|byte| byte.is_ascii_digit())
                        })
                    })
                });
            let safe_suffix = catalog_projection
                || suffix.is_empty()
                || suffix
                    .strip_prefix(".slice(")
                    .and_then(|value| value.strip_suffix(')'))
                    .is_some_and(|range| {
                        range.split_once(',').is_some_and(|(start, end)| {
                            !start.is_empty()
                                && !end.is_empty()
                                && start.bytes().all(|byte| byte.is_ascii_digit())
                                && end.bytes().all(|byte| byte.is_ascii_digit())
                        })
                    })
                || suffix
                    .strip_prefix(&format!(
                        ".map({parameter}=>{parameter}.name+\"::\"+{parameter}.description.slice(0,"
                    ))
                    .and_then(|value| value.strip_suffix("))"))
                    .is_some_and(|limit| {
                        !limit.is_empty() && limit.bytes().all(|byte| byte.is_ascii_digit())
                    });
            if !safe_suffix {
                return Err(self.error("ALL_TOOLS regex filter is not display-only"));
            }
            self.at += end;
            self.semicolon()?;
            return Ok(true);
        }
        if self.identifier().as_deref() != Some(parameter.as_str()) {
            return Err(self.error("ALL_TOOLS listing callback must inspect its parameter"));
        }
        self.skip_space();
        if !self.eat(".") || self.identifier().as_deref() != Some("name") {
            return Err(self.error("ALL_TOOLS listing callback may inspect only the tool name"));
        }
        self.skip_space();
        if self.eat("===") || self.eat("==") {
            self.skip_space();
            if self.string()?.is_empty() {
                return Err(self.error("ALL_TOOLS listing tool name must be nonempty"));
            }
            self.skip_space();
        } else if operation == "find" {
            return Err(self.error("ALL_TOOLS.find requires a literal tool-name comparison"));
        }
        if !self.eat(")") {
            return Err(self.error("ALL_TOOLS listing callback must be display-only"));
        }
        self.semicolon()?;
        Ok(true)
    }

    fn ignored_session_id_presentation(
        &mut self,
        binding: Option<&str>,
    ) -> Result<bool, ReplayError> {
        let Some(binding) = binding else {
            return Ok(false);
        };
        let start = self.at;
        if !self.eat("if") {
            return Ok(false);
        }
        let Some(end) = self.rest().find(';') else {
            self.at = start;
            return Ok(false);
        };
        self.at += end + 1;
        let statement = self.input[start..self.at]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let expected = format!(
            "if({binding}.session_id)text(JSON.stringify({{session_id:{binding}.session_id}}));"
        );
        // Codex's normal compact session presentation is display-only.  It
        // carries no control flow into the replay, so accept this exact
        // alternative alongside the structured JSON form above.
        let session_line =
            format!("if({binding}.session_id)text(`SESSION_ID=${{{binding}.session_id}}`);");
        if statement == expected || statement == session_line {
            Ok(true)
        } else {
            self.at = start;
            Ok(false)
        }
    }

    fn ignored_tool_listing_loop(&mut self) -> Result<bool, ReplayError> {
        let start = self.at;
        if !self.eat("for") {
            return Ok(false);
        }
        self.skip_space();
        if !self.eat("(") {
            self.at = start;
            return Ok(false);
        }
        self.skip_space();
        if !self.eat("const") {
            self.at = start;
            return Ok(false);
        }
        self.skip_space();
        let item = self
            .identifier()
            .ok_or_else(|| self.error("tool listing loop has an invalid item binding"))?;
        self.expect("of", "tool listing loop must iterate literal tool names")?;
        self.skip_space();
        let names = self.literal()?;
        let Value::Array(names) = names else {
            return Err(self.error("tool listing loop must iterate literal tool names"));
        };
        if names.is_empty()
            || !names.iter().all(|name| {
                name.as_str().is_some_and(|name| {
                    !name.is_empty()
                        && name.bytes().all(|byte| {
                            byte.is_ascii_alphanumeric() || matches!(byte, b'_' | b'-' | b'.')
                        })
                })
            })
        {
            return Err(self.error("tool listing loop has an invalid tool name"));
        }
        self.expect(")", "tool listing loop is missing ')'")?;
        self.expect("{", "tool listing loop must have a braced body")?;
        self.expect("const", "tool listing loop must bind its result")?;
        self.skip_space();
        let binding = self
            .identifier()
            .ok_or_else(|| self.error("tool listing loop has an invalid result binding"))?;
        self.skip_space();
        if !self.eat("=") || !self.eat("ALL_TOOLS.find(") {
            return Err(self.error("tool listing loop must use ALL_TOOLS.find"));
        }
        let parameter = self
            .identifier()
            .ok_or_else(|| self.error("tool listing loop has an invalid callback"))?;
        if !self.eat("=>")
            || self.identifier().as_deref() != Some(parameter.as_str())
            || !self.eat(".name===")
            || self.identifier().as_deref() != Some(item.as_str())
            || !self.eat(")")
        {
            return Err(self.error("tool listing loop callback is not display-only"));
        }
        self.semicolon()?;
        self.skip_space();
        if !self.eat("text(")
            || self.identifier().as_deref() != Some(binding.as_str())
            || !self.eat(")")
        {
            return Err(self.error("tool listing loop result must be displayed"));
        }
        self.semicolon()?;
        self.expect("}", "tool listing loop has trailing work")?;
        Ok(true)
    }

    fn ignored_metadata_query_loop(&mut self) -> Result<bool, ReplayError> {
        let start = self.at;
        if !self.eat("const") {
            return Ok(false);
        }
        self.skip_space();
        let Some(batch) = self.identifier() else {
            self.at = start;
            return Ok(false);
        };
        self.skip_space();
        if !self.eat("=") {
            self.at = start;
            return Ok(false);
        }
        self.skip_space();
        if !self.rest().starts_with('[') {
            self.at = start;
            return Ok(false);
        }
        let values = self.literal()?;
        let Value::Array(values) = values else {
            self.at = start;
            return Ok(false);
        };
        if values.is_empty() || !values.iter().all(Value::is_string) {
            self.at = start;
            return Ok(false);
        }
        self.semicolon()?;
        self.skip_space();
        if !self.eat("for") {
            self.at = start;
            return Ok(false);
        }
        self.expect("(", "metadata query loop is missing '('")?;
        self.expect("const", "metadata query loop must bind a const item")?;
        self.skip_space();
        let item = self
            .identifier()
            .ok_or_else(|| self.error("metadata query loop has an invalid item"))?;
        self.expect("of", "metadata query loop must iterate its literal batch")?;
        self.skip_space();
        if self.identifier().as_deref() != Some(batch.as_str()) {
            return Err(self.error("metadata query loop must iterate its literal batch"));
        }
        self.expect(")", "metadata query loop must have a braced body")?;
        self.expect("{", "metadata query loop must have a braced body")?;
        self.expect("const", "metadata query loop must bind its result")?;
        self.skip_space();
        let result = self
            .identifier()
            .ok_or_else(|| self.error("metadata query loop has an invalid result"))?;
        self.expect("=", "metadata query loop must await a read-only tool")?;
        self.expect("await", "metadata query loop must await a read-only tool")?;
        self.expect("tools.", "metadata query loop must await a read-only tool")?;
        let tool = self
            .identifier()
            .ok_or_else(|| self.error("metadata query loop has an invalid tool"))?;
        if tool == "view_image" {
            self.at = start;
            return Ok(false);
        }
        if !is_code_mode_metadata_tool(&tool) || !self.eat("(") {
            return Err(self.error("metadata query loop tool is not read-only"));
        }
        let arguments = self.at;
        let mut quote = None;
        while let Some(byte) = self.input.as_bytes().get(self.at).copied() {
            if let Some(mark) = quote {
                self.at += 1;
                if byte == b'\\' {
                    self.at += 1;
                } else if byte == mark {
                    quote = None;
                }
            } else if matches!(byte, b'\'' | b'"') {
                quote = Some(byte);
                self.at += 1;
            } else if byte == b')' {
                break;
            } else if matches!(byte, b'(' | b'[' | b']' | b';') {
                return Err(self.error("metadata query loop arguments are not data-only"));
            } else {
                self.at += 1;
            }
        }
        let arguments = &self.input[arguments..self.at];
        if !arguments.trim_start().starts_with('{')
            || !arguments.trim_end().ends_with('}')
            || !contains_identifier(arguments, &item)
            || arguments.contains("tools.")
            || arguments.contains("await")
        {
            return Err(self.error("metadata query loop arguments are not data-only"));
        }
        if !self.eat(")") {
            return Err(self.error("metadata query loop tool call is unterminated"));
        }
        self.semicolon()?;
        self.skip_space();
        for name in [&item, &result] {
            if !self.eat("text(")
                || self.identifier().as_deref() != Some(name.as_str())
                || !self.eat(")")
            {
                return Err(self.error("metadata query loop may only display its item and result"));
            }
            self.semicolon()?;
            self.skip_space();
        }
        if !self.eat("}") {
            return Err(self.error("metadata query loop has trailing work"));
        }
        Ok(true)
    }

    fn presentation_for_each(&mut self, results: &str) -> Result<bool, ReplayError> {
        let start = self.at;
        if !self.eat(results) || !self.eat(".forEach") || !self.eat("(") {
            self.at = start;
            return Ok(false);
        }
        self.balanced_end(b'(', b')', "unterminated Promise.all presentation forEach")?;
        self.expect(";", "unterminated Promise.all presentation forEach")?;
        let statement = self.input[start..self.at]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let prefix = format!("{results}.forEach((r,i)=>");
        let Some(callback) = statement.strip_prefix(&prefix) else {
            return Err(self.error("Promise.all presentation forEach must bind (r, i)"));
        };
        let body = if callback.starts_with('{') {
            callback
                .strip_prefix('{')
                .and_then(|value| value.strip_suffix("});"))
                // `});` already consumes the presentation statement's
                // terminating semicolon.  Keep the body validation below
                // unchanged: it still accepts display-only `text(...)`.
                .map(|value| value.strip_suffix(';').unwrap_or(value))
        } else {
            callback.strip_suffix(");")
        };
        let Some(body) = body else {
            return Err(self.error("invalid Promise.all presentation forEach"));
        };
        if !body.starts_with("text(")
            || !body.ends_with(')')
            || (!body.contains("r.output")
                && body != "text(JSON.stringify({i,...r}))"
                && body != "text({i,...r})")
            || body.contains("tools.")
            || body.contains("await")
        {
            return Err(self.error("Promise.all presentation forEach must only display r.output"));
        }
        self.indexed_results = matches!(body, "text(JSON.stringify({i,...r}))" | "text({i,...r})");
        Ok(true)
    }

    fn mapped_json_presentation(&mut self, results: &str) -> bool {
        let start = self.at;
        let Some(end) = self.rest().find(';') else {
            return false;
        };
        self.at += end + 1;
        let statement = self.input[start..self.at]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let expected =
            format!("text(JSON.stringify({results}.map((r,i)=>({{i,output:r.output}}))));");
        if statement == expected {
            true
        } else {
            self.at = start;
            false
        }
    }

    fn destructured_presentation(&mut self, bindings: &str) -> Result<bool, ReplayError> {
        let mut names = bindings
            .split(',')
            .filter(|name| !name.is_empty())
            .map(str::to_owned)
            .collect::<Vec<_>>();
        self.skip_space();
        if self.rest().starts_with("for") {
            let start = self.at;
            if let Some(end) = self.rest().find('}') {
                self.at += end + 1;
                let statement = self.input[start..self.at]
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .collect::<String>();
                let rows = statement
                    .strip_prefix("for(const[label,r]of[[")
                    .and_then(|value| value.strip_suffix("]]){text(label);text(r);}"));
                let safe = rows.is_some_and(|rows| {
                    !rows.is_empty()
                        && rows.split("],[").all(|row| {
                            row.split_once(',').is_some_and(|(label, result)| {
                                label.len() >= 2
                                    && label.starts_with('"')
                                    && label.ends_with('"')
                                    && label[1..label.len() - 1].bytes().all(|byte| {
                                        byte.is_ascii_alphanumeric() || matches!(byte, b' ' | b'-')
                                    })
                                    && names.iter().any(|name| name == result)
                            })
                        })
                });
                if safe {
                    return Ok(true);
                }
                self.at = start;
            }
        }
        loop {
            self.skip_space();
            let start = self.at;
            if !(self.eat("const") || self.eat("let")) {
                self.at = start;
                break;
            }
            self.skip_space();
            let name = self.identifier().ok_or_else(|| {
                self.error("Promise.all presentation binding has an invalid name")
            })?;
            self.expect("=", "Promise.all presentation binding is missing '='")?;
            let expression = self.at;
            let mut quote = None;
            let mut depth = 0usize;
            while let Some(byte) = self.input.as_bytes().get(self.at).copied() {
                self.at += 1;
                if let Some(mark) = quote {
                    if byte == b'\\' {
                        self.at += 1;
                    } else if byte == mark {
                        quote = None;
                    }
                } else {
                    match byte {
                        b'\'' | b'\"' | b'`' => quote = Some(byte),
                        b'(' | b'[' | b'{' => depth += 1,
                        b')' | b']' | b'}' if depth > 0 => depth -= 1,
                        b';' if depth == 0 => break,
                        _ => {}
                    }
                }
            }
            if self.input.as_bytes().get(self.at.saturating_sub(1)) != Some(&b';') {
                return Err(self.error("unterminated Promise.all presentation binding"));
            }
            let expression = &self.input[expression..self.at - 1];
            if expression.contains(['(', ')', '`', '='])
                || expression.contains("tools.")
                || expression.contains("await")
                || !names
                    .iter()
                    .any(|bound| contains_identifier(expression, bound))
            {
                return Err(self.error(
                    "Promise.all presentation binding must derive from a bound result without calls",
                ));
            }
            names.push(name);
        }
        self.skip_space();
        let output = self.at;
        if !self.ignored_output_statement()? {
            return Ok(false);
        }
        let statement = &self.input[output..self.at];
        if !names
            .iter()
            .any(|name| contains_identifier(statement, name))
        {
            return Err(
                self.error("Promise.all destructured presentation does not use its results")
            );
        }
        Ok(true)
    }

    fn presentation_loop(&mut self, results: &str) -> Result<bool, ReplayError> {
        if self.mapped_json_presentation(results) || self.presentation_for_each(results)? {
            return Ok(true);
        }
        if let Some(paths) = self.image_path_results.get(results).cloned() {
            let start = self.at;
            if let Some(end) = self.rest().find('}') {
                self.at += end + 1;
                self.skip_space();
                self.eat(";");
                let statement = self.input[start..self.at]
                    .chars()
                    .filter(|character| !character.is_whitespace())
                    .collect::<String>();
                let expected = format!(
                    "for(leti=0;i<{results}.length;i++){{text({paths}[i]);image({results}[i].image_url);}}"
                );
                if statement.trim_end_matches(';') == expected {
                    return Ok(true);
                }
                self.at = start;
            }
        }
        if !self.eat("for") {
            return Ok(false);
        }
        self.expect("(", "invalid Promise.all presentation loop")?;
        let header_start = self.at;
        let mut depth = 1usize;
        while let Some(byte) = self.input.as_bytes().get(self.at).copied() {
            self.at += 1;
            if byte == b')' {
                depth -= 1;
                if depth == 0 {
                    break;
                }
            } else if byte == b'(' {
                depth += 1;
            }
        }
        if depth != 0 {
            return Err(self.error("unterminated Promise.all presentation loop"));
        }
        let header = self.input[header_start..self.at - 1]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        if let Some(item) = header
            .strip_prefix("const")
            .and_then(|value| value.strip_suffix(&format!("of{results}")))
            .filter(|value| !value.is_empty())
        {
            self.skip_space();
            let braced = self.eat("{");
            self.skip_space();
            let body_start = self.at;
            if !self.ignored_output_statement()? {
                return Err(
                    self.error("Promise.all presentation loop must contain only output calls")
                );
            }
            if !self.input[body_start..self.at].contains(item) {
                return Err(self.error("Promise.all presentation loop must use its bound results"));
            }
            if braced {
                self.skip_space();
                if !self.eat("}") {
                    return Err(
                        self.error("Promise.all presentation loop must contain only output calls")
                    );
                }
            }
            self.skip_space();
            self.eat(";");
            return Ok(true);
        }
        let parts = header.split(';').collect::<Vec<_>>();
        let Some((index, initial)) = parts
            .first()
            .and_then(|part| part.strip_prefix("let"))
            .and_then(|part| part.split_once('='))
        else {
            return Err(self.error("Promise.all presentation loop must use its bound results"));
        };
        if !index.chars().next().is_some_and(|character| {
            character.is_ascii_alphabetic() || matches!(character, '_' | '$')
        }) || !index
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '_' | '$'))
            || parts.len() != 3
            || initial != "0"
            || parts[1] != format!("{index}<{results}.length")
            || parts[2] != format!("{index}++")
        {
            return Err(self.error("Promise.all presentation loop must use its bound results"));
        }
        self.skip_space();
        let braced = self.eat("{");
        self.skip_space();
        let body_start = self.at;
        if !self.ignored_output_statement()? {
            return Err(self.error("Promise.all presentation loop must contain only text calls"));
        }
        let body = self.input[body_start..self.at]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        let reference = format!("{results}[{index}]");
        self.indexed_results = index == "i"
            && (body == format!("text({{{index},...{reference}}});")
                || body == format!("text(JSON.stringify({{{index},...{reference}}}));"));
        let interpolation = format!("${{JSON.stringify({reference})}}");
        let output_interpolation = format!("${{{reference}.output}}");
        if !body.starts_with(&format!("text({reference}"))
            && body != format!("text({{{index},...{reference}}});")
            && body != format!("text(JSON.stringify({{{index},...{reference}}}));")
            && !(body.starts_with("text(`")
                && body.contains(&interpolation)
                && !body.contains(&format!("\\{interpolation}")))
            // This is the corresponding Codex human-readable form, e.g.
            // `text(`---RESULT ${i + 1}---\n${results[i].output}`)`.  It
            // remains presentation-only: the parser already consumed one
            // output helper and rejects nested tools calls there.
            && !(body.starts_with("text(`")
                && body.contains(&output_interpolation)
                && !body.contains("tools.")
                && !body.contains("await"))
        {
            return Err(self.error("Promise.all presentation loop must use its bound results"));
        }
        self.skip_space();
        if braced && !self.eat("}") {
            return Err(self.error("Promise.all presentation loop must contain only text calls"));
        }
        self.skip_space();
        self.eat(";");
        Ok(true)
    }

    fn static_image_loop(&mut self) -> Result<Option<Vec<CodeModeAction>>, ReplayError> {
        let start = self.at;
        if !self.eat("for") {
            return Ok(None);
        }
        self.expect("(", "static image loop is missing '('")?;
        self.expect("const", "static image loop must bind a const path")?;
        self.skip_space();
        let path_name = self
            .identifier()
            .ok_or_else(|| self.error("static image loop has an invalid path binding"))?;
        self.expect("of", "static image loop must iterate a literal path batch")?;
        self.skip_space();
        let Some(batch_name) = self.identifier() else {
            self.at = start;
            return Ok(None);
        };
        let Some(rows) = self.command_batches.get(&batch_name).cloned() else {
            self.at = start;
            return Ok(None);
        };
        let paths = rows
            .into_iter()
            .map(|row| {
                if row.len() == 1 {
                    Ok(row.into_iter().next().unwrap())
                } else {
                    Err(self.error("static image loop requires a literal path list"))
                }
            })
            .collect::<Result<Vec<_>, _>>()?;
        self.expect(")", "static image loop is missing ')'")?;
        self.expect("{", "static image loop must have a braced body")?;
        self.expect("const", "static image loop must bind each result")?;
        self.skip_space();
        let result_name = self
            .identifier()
            .ok_or_else(|| self.error("static image loop has an invalid result binding"))?;
        self.expect("=", "static image loop result is missing '='")?;
        self.expect("await", "static image loop must await view_image")?;
        self.skip_space();
        self.strings.insert(path_name, paths[0].clone());
        let template = self.action()?;
        if template.tool != "view_image"
            || template.arguments.len() != 2
            || template.arguments.get("path").and_then(Value::as_str) != Some(paths[0].as_str())
            || template.arguments.get("detail").and_then(Value::as_str) != Some("original")
            || paths.iter().any(|path| !path.starts_with('/'))
        {
            return Err(
                self.error("static image loop requires absolute literal paths and original detail")
            );
        }
        self.semicolon()?;
        self.skip_space();
        let output = self.at;
        if !self.ignored_output_statement()? {
            return Err(self.error("static image loop must display each image"));
        }
        let statement = self.input[output..self.at]
            .chars()
            .filter(|character| !character.is_whitespace())
            .collect::<String>();
        if statement != format!("image({result_name}.image_url);") {
            return Err(self.error("static image loop may only display each image result"));
        }
        self.expect("}", "static image loop must contain one display call")?;
        self.skip_space();
        self.eat(";");
        debug_assert!(self.at > start);
        Ok(Some(
            paths
                .into_iter()
                .map(|path| {
                    let mut action = template.clone();
                    action
                        .arguments
                        .insert("path".to_owned(), Value::String(path));
                    action
                })
                .collect(),
        ))
    }

    fn action(&mut self) -> Result<CodeModeAction, ReplayError> {
        if !self.eat("tools.") {
            return Err(self.error("expected tools call"));
        }
        let tool = self
            .identifier()
            .ok_or_else(|| self.error("missing tool name"))?;
        self.expect("(", "tool call missing '('")?;
        let value = self.literal()?;
        self.expect(")", "tool call missing ')'")?;
        if !is_supported_tool(&tool) && !is_code_mode_metadata_tool(&tool) {
            return Err(self.error(&format!("unsupported nested tool {tool:?}")));
        }
        let arguments = match (tool.as_str(), value) {
            ("apply_patch", Value::String(patch)) => {
                Map::from_iter([("input".to_owned(), Value::String(patch))])
            }
            ("apply_patch", Value::Object(arguments)) => Map::from_iter([(
                "input".to_owned(),
                arguments
                    .get("input")
                    .or_else(|| arguments.get("patch"))
                    .filter(|patch| patch.is_string())
                    .cloned()
                    .ok_or_else(|| self.error("apply_patch requires string input"))?,
            )]),
            (_, Value::Object(arguments)) => arguments,
            _ => return Err(self.error("tool arguments must be an object")),
        };
        Ok(CodeModeAction {
            tool,
            arguments,
            parallel_group_id: None,
            result_print: ResultPrint::None,
        })
    }

    fn validate_mapped_callback(
        &self,
        prefix: &str,
        suffix: &str,
        bindings: &[String],
    ) -> Result<(), ReplayError> {
        let prefix_pattern = Regex::new(
            r"^\s*(?:(?:return[ \t]+)?(?:await\s+)?|(?:const|let)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:await\s+)?)$",
        ).expect("static mapped callback prefix regex");
        let captured = prefix_pattern.captures(prefix).ok_or_else(|| {
            self.error("command batch callback contains unsupported work before its tool call")
        })?;
        let result = captured.get(1).map(|value| value.as_str());
        if result.is_some_and(|name| bindings.iter().any(|binding| binding == name)) {
            return Err(self.error("command batch result shadows a callback binding"));
        }
        let mut references = bindings
            .iter()
            .map(|name| regex::escape(name))
            .collect::<Vec<_>>();
        if let Some(name) = result {
            references.push(format!(r"{}(?:\s*\.\s*output)?", regex::escape(name)));
        }
        let reference = format!("(?:{})", references.join("|"));
        let quoted = r#"(?:"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*')"#;
        let template = format!(r"`(?:[^`$\\]|\\.|\$\{{\s*{reference}\s*\}})*`");
        let value =
            format!(r"(?:{quoted}|{template}|{reference}|true|false|null|-?[0-9]+(?:\.[0-9]+)?)");
        let property = format!(r"(?:[A-Za-z_$][A-Za-z0-9_$]*|{quoted})\s*:\s*{value}|{reference}");
        let object = format!(r"\{{\s*(?:(?:{property})\s*(?:,\s*(?:{property})\s*)*,?)?\}}");
        let suffix_pattern = Regex::new(&format!(
            r"^\s*;?\s*(?:return[ \t]+(?:{value}|{object})\s*;?\s*)?$"
        ))
        .expect("mapped callback presentation regex");
        if !suffix_pattern.is_match(suffix)
            || (prefix.trim_start().starts_with("return")
                && !suffix.trim().trim_end_matches(';').is_empty())
        {
            return Err(
                self.error("command batch callback contains unsupported work after its tool call")
            );
        }
        Ok(())
    }

    fn parse(mut self) -> Result<CodeModeActions, ReplayError> {
        let mut actions: Vec<CodeModeAction> = Vec::new();
        let mut direct_prints = Some(Vec::new());
        let mut presentation_loop_allowed: Option<String> = None;
        let mut direct_await_binding: Option<String> = None;
        let mut batch_start = 0;
        let mut settled = false;
        while self.at < self.input.len() {
            self.skip_space();
            if self.at == self.input.len() {
                break;
            }
            if let Some(results) = presentation_loop_allowed.take() {
                direct_prints = None;
                self.indexed_results = false;
                let presented = if let Some(bindings) = results.strip_prefix('\0') {
                    self.destructured_presentation(bindings)?
                } else {
                    self.presentation_loop(&results)?
                };
                if !presented {
                    return Err(
                        self.error("Promise.all result binding requires its presentation loop")
                    );
                }
                if self.indexed_results {
                    for action in &mut actions[batch_start..] {
                        action.result_print = if settled {
                            ResultPrint::Settled
                        } else {
                            ResultPrint::Indexed
                        };
                    }
                }
                continue;
            }
            if self.ignored_metadata_query_loop()? {
                direct_prints = None;
                continue;
            }
            if let Some(batch) = self.static_image_loop()? {
                direct_prints = None;
                actions.extend(batch);
                continue;
            }
            if self.ignored_tool_listing_loop()? {
                direct_prints = None;
                continue;
            }
            if self.ignored_session_id_presentation(direct_await_binding.as_deref())? {
                direct_prints = None;
                continue;
            }
            let output_wrapper = self.rest().starts_with("text(await ");
            if output_wrapper {
                self.at += "text(".len();
            } else {
                let start = self.at;
                if self.ignored_output_statement()? {
                    let display: String = self.input[start..self.at]
                        .chars()
                        .filter(|c| !c.is_whitespace())
                        .collect();
                    let full_result = direct_await_binding.as_ref().is_some_and(|binding| {
                        display.trim_end_matches(';') == format!("text({binding})")
                            || display.trim_end_matches(';')
                                == format!("text(JSON.stringify({binding}))")
                    });
                    if full_result {
                        if let Some(action) = actions.last_mut() {
                            action.result_print = ResultPrint::Direct;
                        }
                    }
                    if full_result || Self::data_only_print(self.input[start..self.at].trim()) {
                        if let Some(prints) = &mut direct_prints {
                            prints.push(full_result.then(|| actions.len() - 1));
                        }
                    } else {
                        direct_prints = None;
                    }
                    continue;
                }
            }
            let promise_binding = if self.eat("const") || self.eat("let") {
                self.skip_space();
                let destructured_output =
                    self.eat("{output,...rest}") || self.eat("{ output, ...rest }");
                let name = if destructured_output {
                    "__discarded_output".to_owned()
                } else if self.eat("[") {
                    let mut names = Vec::new();
                    loop {
                        self.skip_space();
                        names.push(self.identifier().ok_or_else(|| {
                            self.error("invalid Promise.all destructured binding")
                        })?);
                        self.skip_space();
                        if self.eat("]") {
                            break;
                        }
                        if !self.eat(",") {
                            return Err(
                                self.error("invalid Promise.all destructured binding separator")
                            );
                        }
                    }
                    format!("\0{}", names.join(","))
                } else {
                    self.identifier()
                        .ok_or_else(|| self.error("invalid literal binding"))?
                };
                self.expect("=", "invalid literal binding")?;
                self.skip_space();
                if !self.rest().starts_with("await") {
                    direct_await_binding = None;
                    if let Some(batch) = self.literal_command_batch()? {
                        self.semicolon()?;
                        self.command_batches.insert(name, batch);
                        continue;
                    }
                    if self.ignored_tool_listing_initializer()? {
                        continue;
                    }
                    if !matches!(
                        self.input.as_bytes().get(self.at),
                        Some(b'\'') | Some(b'\"') | Some(b'`')
                    ) {
                        return Err(self.error("const bindings must be string literals"));
                    }
                    let value = self.string()?;
                    self.strings.insert(name, value);
                    self.semicolon()?;
                    continue;
                }
                Some(name)
            } else {
                None
            };
            if !self.eat("await") {
                return Err(self.error("expected direct awaited tools call"));
            }
            self.skip_space();
            settled = self.eat("Promise.allSettled");
            if settled || self.eat("Promise.all") {
                batch_start = actions.len();
                self.expect("(", "Promise.all missing '('")?;
                self.skip_space();
                let argument_start = self.at;
                if let Some(batch_name) = self.identifier() {
                    if let Some(commands) = self.command_batches.get(&batch_name).cloned() {
                        if !self.eat(".map") || !self.eat("(") {
                            return Err(self.error("command batch must use a literal map"));
                        }
                        let map_body = self.at;
                        self.skip_space();
                        if let Some(path_name) = self.identifier() {
                            self.skip_space();
                            if self.eat("=>") {
                                self.skip_space();
                                if self.rest().starts_with("tools.view_image(") {
                                    let paths = commands
                                        .iter()
                                        .map(|row| match row.as_slice() {
                                            [path] => Ok(path.clone()),
                                            _ => Err(self.error(
                                                "image path batch must use a literal path list",
                                            )),
                                        })
                                        .collect::<Result<Vec<_>, _>>()?;
                                    self.strings.insert(path_name.clone(), paths[0].clone());
                                    let template = self.action()?;
                                    if template.tool != "view_image"
                                        || template.arguments.len() != 2
                                        || template.arguments.get("path").and_then(Value::as_str)
                                            != Some(paths[0].as_str())
                                        || template.arguments.get("detail").and_then(Value::as_str)
                                            != Some("original")
                                        || paths.iter().any(|path| !path.starts_with('/'))
                                    {
                                        return Err(self.error(
                                            "image path batch must use absolute literal paths and original detail",
                                        ));
                                    }
                                    self.expect(")", "image path map is missing ')'")?;
                                    self.expect(")", "image Promise.all is missing ')'")?;
                                    self.semicolon()?;
                                    for path in paths {
                                        let mut action = template.clone();
                                        action
                                            .arguments
                                            .insert("path".to_owned(), Value::String(path));
                                        actions.push(action);
                                    }
                                    let results = promise_binding.as_deref().ok_or_else(|| {
                                        self.error("image path batch requires a result binding")
                                    })?;
                                    if results.starts_with('\0') {
                                        return Err(self.error(
                                            "image path batch requires one result-array binding",
                                        ));
                                    }
                                    self.image_path_results
                                        .insert(results.to_owned(), batch_name);
                                    presentation_loop_allowed = promise_binding;
                                    continue;
                                }
                            }
                        }
                        self.at = map_body;
                        self.eat("async");
                        self.skip_space();
                        let mut bindings = Vec::new();
                        if self.eat("(") {
                            self.skip_space();
                            if self.eat("[") {
                                loop {
                                    self.skip_space();
                                    bindings.push(self.identifier().ok_or_else(|| {
                                        self.error("invalid command batch map binding")
                                    })?);
                                    self.skip_space();
                                    if self.eat("]") {
                                        break;
                                    }
                                    if !self.eat(",") {
                                        return Err(self.error("invalid command batch map binding"));
                                    }
                                }
                            } else {
                                bindings.push(self.identifier().ok_or_else(|| {
                                    self.error("invalid command batch map binding")
                                })?);
                            }
                            self.expect(")", "invalid command batch map binding")?;
                        } else {
                            bindings.push(
                                self.identifier().ok_or_else(|| {
                                    self.error("invalid command batch map binding")
                                })?,
                            );
                        }
                        self.expect("=>", "invalid command batch map")?;
                        self.skip_space();
                        if bindings.is_empty() || bindings.len() > 2 {
                            return Err(self.error("invalid command batch map binding"));
                        }
                        let braced = self.eat("{");
                        let body_start = self.at;
                        let body_end = if braced {
                            self.balanced_end(b'{', b'}', "unterminated command batch map")?
                        } else {
                            self.input.len()
                        };
                        let body = &self.input[body_start..body_end];
                        let tool_offset = body.find("tools.").ok_or_else(|| {
                            self.error("command batch must execute commands directly")
                        })?;
                        if body.matches("tools.").count() != 1 {
                            return Err(self.error(
                                "command batch map must contain exactly one unconditional tools call",
                            ));
                        }
                        if bindings.iter().collect::<BTreeSet<_>>().len() != bindings.len()
                            || commands.iter().any(|row| row.len() != bindings.len())
                        {
                            return Err(self.error("command batch item does not match its binding"));
                        }
                        let tool_start = body_start + tool_offset;
                        let original_strings = self.strings.clone();
                        let parsed = (|| -> Result<(Vec<CodeModeAction>, usize), ReplayError> {
                            self.strings = original_strings.clone();
                            let markers = bindings
                                .iter()
                                .enumerate()
                                .map(|(index, binding)| {
                                    let marker = format!("\0tasktrace-command-binding-{index}\0");
                                    self.strings.insert(binding.clone(), marker.clone());
                                    marker
                                })
                                .collect::<Vec<_>>();
                            self.at = tool_start;
                            let marker_action = self.action()?;
                            if marker_action.tool != "exec_command" {
                                return Err(self.error("command batch must use exec_command"));
                            }
                            let command = marker_action
                                .arguments
                                .get("cmd")
                                .and_then(Value::as_str)
                                .ok_or_else(|| self.error("command batch must bind cmd"))?;
                            let columns = markers
                                .iter()
                                .enumerate()
                                .filter_map(|(index, marker)| (command == marker).then_some(index))
                                .collect::<Vec<_>>();
                            let [command_column] = columns.as_slice() else {
                                return Err(self.error(
                                    "command batch cmd must reference one callback binding",
                                ));
                            };
                            let tool_end = self.at;
                            if braced {
                                self.validate_mapped_callback(
                                    &body[..tool_offset],
                                    &self.input[tool_end..body_end],
                                    &bindings,
                                )?;
                            } else if !body[..tool_offset].trim().is_empty() {
                                return Err(
                                    self.error("command batch must execute commands directly")
                                );
                            }
                            let mut batch = Vec::with_capacity(commands.len());
                            for row in &commands {
                                self.strings = original_strings.clone();
                                for (binding, value) in bindings.iter().zip(row) {
                                    self.strings.insert(binding.clone(), value.clone());
                                }
                                self.at = tool_start;
                                let action = self.action()?;
                                if action.tool != "exec_command"
                                    || self.at != tool_end
                                    || action.arguments.get("cmd").and_then(Value::as_str)
                                        != Some(row[*command_column].as_str())
                                {
                                    return Err(self.error(
                                        "command batch cmd does not match its callback binding",
                                    ));
                                }
                                batch.push(action);
                            }
                            Ok((batch, tool_end))
                        })();
                        self.strings = original_strings;
                        let (mut batch, tool_end) = parsed?;
                        self.at = if braced { body_end + 1 } else { tool_end };
                        self.skip_space();
                        if !self.eat(")") || !self.eat(")") {
                            return Err(self.error("unterminated command batch"));
                        }
                        self.semicolon()?;
                        self.parallel_groups += 1;
                        let parallel_group_id = format!(
                            "{}:{}:{}",
                            self.outer_call_identity,
                            if settled {
                                "promise_all_settled"
                            } else {
                                "promise_all"
                            },
                            self.parallel_groups
                        );
                        for mut action in batch.drain(..) {
                            action.parallel_group_id = Some(parallel_group_id.clone());
                            actions.push(action);
                        }
                        presentation_loop_allowed = promise_binding;
                        continue;
                    }
                    self.at = argument_start;
                }
                if !self.eat("[") {
                    return Err(self.error("Promise.all requires a literal array"));
                }
                let mut batch = Vec::new();
                loop {
                    self.skip_space();
                    if self.eat("]") {
                        break;
                    }
                    batch.push(self.action()?);
                    self.skip_space();
                    if self.eat("]") {
                        break;
                    }
                    if !self.eat(",") {
                        return Err(self.error("Promise.all array separator expected"));
                    }
                }
                if batch.is_empty() {
                    return Err(self.error("Promise.all requires a nonempty literal array"));
                }
                if let Some(bindings) = promise_binding
                    .as_deref()
                    .and_then(|value| value.strip_prefix('\0'))
                {
                    let names = bindings.split(',').collect::<Vec<_>>();
                    if names.len() != batch.len()
                        || names
                            .iter()
                            .enumerate()
                            .any(|(index, name)| names[..index].contains(name))
                    {
                        return Err(self.error(
                            "Promise.all destructured binding must match its unique members",
                        ));
                    }
                }
                let executable = batch.iter().all(|action| action.tool == "exec_command");
                let metadata = batch
                    .iter()
                    .all(|action| is_code_mode_metadata_tool(&action.tool));
                let mixed_settled = settled
                    && batch.iter().all(|action| {
                        action.tool == "exec_command" || is_code_mode_metadata_tool(&action.tool)
                    });
                if !executable && !metadata && !mixed_settled {
                    return Err(self.error(
                        "Promise.all members must be homogeneous exec_command or metadata calls",
                    ));
                }
                if executable || mixed_settled {
                    self.parallel_groups += 1;
                    let parallel_group_id = format!(
                        "{}:{}:{}",
                        self.outer_call_identity,
                        if settled {
                            "promise_all_settled"
                        } else {
                            "promise_all"
                        },
                        self.parallel_groups
                    );
                    for action in &mut batch {
                        action.parallel_group_id = Some(parallel_group_id.clone());
                    }
                }
                actions.extend(batch);
                self.expect(")", "Promise.all missing ')'")?;
                if output_wrapper {
                    return Err(self.error("Promise.all cannot be wrapped by an output helper"));
                }
                self.semicolon()?;
                presentation_loop_allowed = promise_binding;
                continue;
            }
            // Accept a basic parenthesized direct await, for example
            // `const result = await (tools.exec_command({cmd: "true"}))`.
            // Do not use this to broaden the Promise.all grammar: that form
            // remains deliberately literal and separately checked above.
            let direct_await_wrapper = self.eat("(");
            self.skip_space();
            let mut action = self.action()?;
            if self.ignored_empty_catch()? && !is_code_mode_metadata_tool(&action.tool) {
                return Err(self.error("catch may wrap only an ignored metadata tool"));
            }
            if direct_await_wrapper {
                self.expect(")", "direct awaited tools wrapper missing ')'")?;
            }
            if output_wrapper {
                self.expect(")", "output helper missing ')'")?;
                action.result_print = ResultPrint::Direct;
                if let Some(prints) = &mut direct_prints {
                    prints.push(Some(actions.len()));
                }
            }
            self.semicolon()?;
            actions.push(action);
            direct_await_binding = promise_binding;
        }
        Ok(CodeModeActions {
            actions,
            direct_prints,
        })
    }
}

fn codex_exec_actions(
    arguments: &Map<String, Value>,
    outer_call_identity: String,
) -> Result<CodeModeActions, ReplayError> {
    let mut input = arguments
        .get("input")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ReplayError::new("unsupported Codex exec payload: missing string field: input")
        })?;
    if input.starts_with("// @exec:") {
        input = input.split_once('\n').map_or("", |(_, body)| body);
    }
    CodeModeCell::new(input, outer_call_identity).parse()
}

fn canonical_tool(tool: &str, original_format: Option<&str>, claude_code: bool) -> String {
    if claude_code {
        return match tool {
            "Bash" => "exec_command",
            "Write" => "claude_write",
            "Edit" => "claude_edit",
            "Read" => "claude_read",
            "Grep" => "claude_grep",
            "Glob" => "claude_glob",
            "TaskStop" => "claude_task_stop",
            "TaskOutput" => "claude_task_output",
            _ => tool,
        }
        .to_owned();
    }
    if original_format == Some(MINI_SWE_AGENT_ORIGINAL_FORMAT) && tool == "bash" {
        "exec_command".to_owned()
    } else {
        tool.to_owned()
    }
}

fn canonical_arguments(
    mut arguments: Map<String, Value>,
    source_tool: &str,
    original_format: Option<&str>,
    claude_code: bool,
) -> Map<String, Value> {
    if ((claude_code && source_tool == "Bash")
        || (original_format == Some(MINI_SWE_AGENT_ORIGINAL_FORMAT) && source_tool == "bash"))
        && !arguments.contains_key("cmd")
    {
        if let Some(Value::String(command)) = arguments.get("command") {
            arguments.insert("cmd".to_owned(), Value::String(command.clone()));
        }
    }
    arguments
}

fn read_trajectory_bytes(requested_path: &Path) -> Result<Vec<u8>, ReplayError> {
    let source = if requested_path.is_file() {
        requested_path.to_path_buf()
    } else {
        let gzip_sibling = PathBuf::from(format!("{}.gz", requested_path.to_string_lossy()));
        if gzip_sibling.is_file() {
            gzip_sibling
        } else {
            return Err(ReplayError::new(format!(
                "ATIF trajectory does not exist: {}",
                requested_path.display()
            )));
        }
    };
    let raw = fs::read(source)?;
    if raw.starts_with(&[0x1f, 0x8b]) {
        let mut decoder = GzDecoder::new(raw.as_slice());
        let mut decoded = Vec::new();
        decoder.read_to_end(&mut decoded)?;
        Ok(decoded)
    } else {
        Ok(raw)
    }
}

fn call_observation(
    step: &Map<String, Value>,
    call_id: Option<&str>,
) -> Result<Value, ReplayError> {
    let Some(call_id) = call_id.filter(|id| !id.is_empty()) else {
        return Err(ReplayError::new(
            "ATIF tool call has no exact observation binding",
        ));
    };
    if step
        .get("tool_calls")
        .and_then(Value::as_array)
        .map_or(true, |calls| {
            calls
                .iter()
                .filter(|call| call.get("tool_call_id").and_then(Value::as_str) == Some(call_id))
                .count()
                != 1
        })
    {
        return Err(ReplayError::new("ATIF tool call identity is duplicated"));
    }
    let matches = step
        .get("observation")
        .and_then(|observation| observation.get("results"))
        .and_then(Value::as_array)
        .map(|results| {
            results
                .iter()
                .filter(|result| {
                    result.get("source_call_id").and_then(Value::as_str) == Some(call_id)
                })
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();
    if matches.len() == 1 {
        return Ok(matches[0].clone());
    }
    Err(ReplayError::new(format!(
        "ATIF tool call {call_id:?} has no exact observation binding"
    )))
}

fn mini_swe_call_observation(
    step: &Map<String, Value>,
    call_id: Option<&str>,
) -> Result<Value, ReplayError> {
    // ATIF-v1.2 mini-SWE recordings may omit result IDs for singleton steps.
    if step
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| calls.len() == 1)
    {
        if let Some(results) = step
            .get("observation")
            .and_then(|value| value.get("results"))
            .and_then(Value::as_array)
        {
            if results.len() == 1
                && results[0]
                    .get("source_call_id")
                    .map_or(true, Value::is_null)
            {
                return Ok(results[0].clone());
            }
        }
    }
    call_observation(step, call_id)
}

fn validate_claude_agent_parent(
    steps: &[Value],
    step_offset: usize,
    arguments: &Map<String, Value>,
    observation: &Value,
) -> Result<(), ReplayError> {
    const KEYS: [&str; 3] = ["description", "prompt", "subagent_type"];
    if arguments.len() != KEYS.len()
        || arguments.keys().any(|key| !KEYS.contains(&key.as_str()))
        || KEYS.iter().any(|key| {
            !arguments
                .get(*key)
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty())
        })
        || claude_consistent_result_is_error(observation)? == Some(true)
    {
        return Err(ReplayError::new(
            "Claude Agent parent is not an exact successful call",
        ));
    }
    let result = observation
        .pointer("/extra/tool_result_metadata/tool_use_result")
        .and_then(Value::as_object)
        .ok_or_else(|| ReplayError::new("Claude Agent parent has no structured result"))?;
    let child_text = result
        .get("content")
        .and_then(Value::as_array)
        .filter(|content| content.len() == 1)
        .and_then(|content| content[0].as_object())
        .filter(|content| {
            content.len() == 2 && content.get("type").and_then(Value::as_str) == Some("text")
        })
        .and_then(|content| content.get("text"))
        .and_then(Value::as_str)
        .filter(|text| !text.is_empty())
        .ok_or_else(|| ReplayError::new("Claude Agent result has no single child text"))?;
    if result.get("status").and_then(Value::as_str) != Some("completed")
        || result.get("prompt") != arguments.get("prompt")
        || result.get("agentType") != arguments.get("subagent_type")
    {
        return Err(ReplayError::new(
            "Claude Agent result does not identify the requested child",
        ));
    }
    let mut sidechain = steps[step_offset + 1..].iter().take_while(|step| {
        step.get("extra")
            .and_then(|extra| extra.get("is_sidechain"))
            .and_then(Value::as_bool)
            == Some(true)
    });
    let first = sidechain
        .next()
        .ok_or_else(|| ReplayError::new("Claude Agent parent has no captured child sidechain"))?;
    if first.get("source").and_then(Value::as_str) != Some("user")
        || first.get("message") != arguments.get("prompt")
    {
        return Err(ReplayError::new(
            "Claude Agent child sidechain does not start with its prompt",
        ));
    }
    let mut child_calls = 0usize;
    let mut final_message = None;
    let sidechain = std::iter::once(first).chain(sidechain).collect::<Vec<_>>();
    for step in &sidechain {
        child_calls += step
            .get("tool_calls")
            .and_then(Value::as_array)
            .map_or(0, Vec::len);
        if step.get("source").and_then(Value::as_str) == Some("agent") {
            if let Some(message) = step
                .get("message")
                .and_then(Value::as_str)
                .filter(|v| !v.is_empty())
            {
                final_message = Some(message);
            }
        }
    }
    let terminal = sidechain.last().unwrap();
    let one_parent = steps[step_offset]
        .get("tool_calls")
        .and_then(Value::as_array)
        .is_some_and(|calls| {
            calls
                .iter()
                .filter(|call| call.get("function_name").and_then(Value::as_str) == Some("Agent"))
                .count()
                == 1
        });
    if !one_parent
        || child_calls == 0
        || final_message != Some(child_text)
        || terminal.get("source").and_then(Value::as_str) != Some("agent")
        || terminal.get("message").and_then(Value::as_str) != Some(child_text)
        || terminal
            .get("tool_calls")
            .and_then(Value::as_array)
            .is_some_and(|calls| !calls.is_empty())
    {
        return Err(ReplayError::new(
            "Claude Agent child sidechain is incomplete",
        ));
    }
    Ok(())
}

fn claude_monitor_poll(
    arguments: &Map<String, Value>,
    observation: &Value,
    background_task_ids: &BTreeSet<String>,
) -> Result<String, ReplayError> {
    const KEYS: [&str; 4] = ["command", "description", "persistent", "timeout_ms"];
    let timeout = arguments
        .get("timeout_ms")
        .and_then(Value::as_u64)
        .filter(|timeout| *timeout > 0)
        .ok_or_else(|| ReplayError::new("Claude Monitor timeout must be a positive integer"))?;
    if arguments.len() != KEYS.len()
        || arguments.keys().any(|key| !KEYS.contains(&key.as_str()))
        || arguments.get("persistent").and_then(Value::as_bool) != Some(false)
        || !arguments
            .get("description")
            .and_then(Value::as_str)
            .is_some_and(|value| !value.is_empty())
        || claude_consistent_result_is_error(observation)? == Some(true)
    {
        return Err(ReplayError::new(
            "Claude Monitor is not an exact non-persistent success",
        ));
    }
    let command = arguments
        .get("command")
        .and_then(Value::as_str)
        .unwrap_or("");
    let tail_grep = Regex::new(
        r#"^tail -f (/tmp/claude-[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+/tasks/([A-Za-z0-9._-]+)\.output) \| grep -E --line-buffered (["'])([^"'\r\n]+)(["'])$"#,
    )
    .unwrap()
    .captures(command);
    let wait_tail = Regex::new(
        r#"^until ! pgrep -f (["'])([^"'\r\n]+)(["']) > /dev/null 2>&1; do sleep [1-9][0-9]*; done; echo (["'])([^"'\r\n]+)(["']); tail -[1-9][0-9]* (/tmp/claude-[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+/tasks/([A-Za-z0-9._-]+)\.output)$"#,
    )
    .unwrap()
    .captures(command);
    let task_id = tail_grep
        .as_ref()
        .filter(|capture| {
            capture.get(3).unwrap().as_str() == capture.get(5).unwrap().as_str()
                && safe_claude_monitor_literal(capture.get(4).unwrap().as_str())
        })
        .and_then(|capture| capture.get(2))
        .or_else(|| {
            wait_tail
                .as_ref()
                .filter(|capture| {
                    capture.get(1).unwrap().as_str() == capture.get(3).unwrap().as_str()
                        && capture.get(4).unwrap().as_str() == capture.get(6).unwrap().as_str()
                        && safe_claude_monitor_literal(capture.get(2).unwrap().as_str())
                        && safe_claude_monitor_literal(capture.get(5).unwrap().as_str())
                })
                .and_then(|capture| capture.get(8))
        })
        .map(|capture| capture.as_str())
        .ok_or_else(|| ReplayError::new("Claude Monitor command is unsupported"))?;
    let result = observation
        .pointer("/extra/tool_result_metadata/tool_use_result")
        .and_then(Value::as_object);
    let orchestration_id = result
        .and_then(|value| value.get("taskId"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty());
    if !background_task_ids.contains(task_id)
        || orchestration_id.is_some_and(|id| background_task_ids.contains(id))
        || result
            .and_then(|value| value.get("timeoutMs"))
            .and_then(Value::as_u64)
            != Some(timeout)
        || result
            .and_then(|value| value.get("persistent"))
            .and_then(Value::as_bool)
            != Some(false)
        || orchestration_id.is_none()
    {
        return Err(ReplayError::new(
            "Claude Monitor does not prove one owned non-persistent poll",
        ));
    }
    Ok(orchestration_id.unwrap().to_owned())
}

fn safe_claude_monitor_literal(value: &str) -> bool {
    let mut bytes = value.bytes();
    let mut nonempty = false;
    while let Some(byte) = bytes.next() {
        nonempty = true;
        if byte == b'\\' {
            if !matches!(bytes.next(), Some(b'[' | b']')) {
                return false;
            }
        } else if !(byte.is_ascii_alphanumeric()
            || byte.is_ascii_whitespace()
            || matches!(byte, b'|' | b':' | b'.' | b'_' | b'/' | b'-' | b'[' | b']'))
        {
            return false;
        }
    }
    nonempty
}

fn claude_result_is_error(observation: &Value) -> Option<bool> {
    observation
        .get("extra")
        .and_then(|extra| extra.get("tool_result_is_error"))
        .or_else(|| {
            observation
                .get("extra")
                .and_then(|extra| extra.get("tool_result_metadata"))
                .and_then(|metadata| metadata.get("raw_tool_result"))
                .and_then(|result| result.get("is_error"))
        })
        .or_else(|| {
            observation
                .get("tool_result")
                .and_then(|result| result.get("is_error"))
        })
        .or_else(|| observation.get("is_error"))
        .or_else(|| observation.pointer("/extra/tool_result_metadata/is_error"))
        .and_then(Value::as_bool)
}

fn claude_consistent_result_is_error(observation: &Value) -> Result<Option<bool>, ReplayError> {
    let values = [
        observation.get("is_error"),
        observation.pointer("/tool_result/is_error"),
        observation.pointer("/extra/tool_result_is_error"),
        observation.pointer("/extra/tool_result_metadata/is_error"),
        observation.pointer("/extra/tool_result_metadata/raw_tool_result/is_error"),
    ];
    if values.iter().flatten().any(|value| !value.is_boolean()) {
        return Err(ReplayError::new(
            "Claude auxiliary outcome has a non-boolean error flag",
        ));
    }
    let explicit = values
        .into_iter()
        .flatten()
        .filter_map(Value::as_bool)
        .collect::<BTreeSet<_>>();
    if explicit.len() > 1 {
        return Err(ReplayError::new(
            "Claude auxiliary outcome has conflicting error flags",
        ));
    }
    Ok(explicit.into_iter().next())
}

fn claude_background_task_id(observation: &Value) -> Option<&str> {
    observation
        .get("extra")
        .and_then(|extra| extra.get("tool_result_metadata"))
        .and_then(|metadata| metadata.get("tool_use_result"))
        .and_then(|result| result.get("backgroundTaskId"))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
}

fn claude_task_result_id(observation: &Value) -> Option<&str> {
    let result = observation.pointer("/extra/tool_result_metadata/tool_use_result")?;
    result
        .get("task_id")
        .or_else(|| result.get("task").and_then(|task| task.get("task_id")))
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
}

fn claude_bash_timeout(arguments: &Map<String, Value>) -> Result<Option<f64>, ReplayError> {
    let Some(value) = arguments.get("timeout") else {
        return Ok(None);
    };
    let timeout = value
        .as_f64()
        .filter(|timeout| timeout.is_finite() && *timeout > 0.0)
        .ok_or_else(|| ReplayError::new("Claude Code Bash timeout must be a positive number"))?;
    Ok(Some(timeout / 1000.0))
}

struct TrajectoryNode<'a> {
    id: String,
    value: &'a Map<String, Value>,
}

fn collect_trajectory_nodes<'a>(
    value: &'a Map<String, Value>,
    root: bool,
    preflattened_ids: Option<&BTreeSet<String>>,
    seen: &mut BTreeSet<String>,
    nodes: &mut Vec<TrajectoryNode<'a>>,
) -> Result<(), ReplayError> {
    let id = value
        .get("trajectory_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .map(str::to_owned)
        .or_else(|| root.then(|| "root".to_owned()))
        .ok_or_else(|| ReplayError::new("embedded ATIF trajectory has no trajectory_id"))?;
    if !seen.insert(id.clone()) {
        return Err(ReplayError::new(format!(
            "ATIF trajectory_id is duplicated: {id:?}"
        )));
    }
    nodes.push(TrajectoryNode {
        id: id.clone(),
        value,
    });
    let children = match value.get("subagent_trajectories") {
        None | Some(Value::Null) => &[][..],
        Some(Value::Array(children)) => children.as_slice(),
        Some(_) => {
            return Err(ReplayError::new(
                "ATIF subagent_trajectories is not an array",
            ))
        }
    };
    let mut child_ids = BTreeSet::new();
    for child in children {
        let child = child
            .as_object()
            .ok_or_else(|| ReplayError::new("embedded ATIF trajectory is not an object"))?;
        let child_id = child
            .get("trajectory_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| ReplayError::new("embedded ATIF trajectory has no trajectory_id"))?;
        child_ids.insert(child_id.to_owned());
    }
    for step in value
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for result in step
            .pointer("/observation/results")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            for reference in result
                .get("subagent_trajectory_ref")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
            {
                let referenced = reference
                    .get("trajectory_id")
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .ok_or_else(|| {
                        ReplayError::new("ATIF subagent reference has no trajectory_id")
                    })?;
                if !child_ids.contains(referenced)
                    && !preflattened_ids.is_some_and(|ids| ids.contains(referenced))
                {
                    return Err(ReplayError::new(format!(
                        "ATIF trajectory {id:?} references unembedded subagent {referenced:?}"
                    )));
                }
            }
        }
    }
    for child in children {
        collect_trajectory_nodes(
            child.as_object().expect("children were validated"),
            false,
            None,
            seen,
            nodes,
        )?;
    }
    Ok(())
}

struct CodexNormalization {
    source_ids: BTreeSet<String>,
    executable_events: usize,
    bound_outcomes: usize,
}

fn value_contains_text(value: &Value, needle: &str) -> bool {
    match value {
        Value::String(value) => value.contains(needle),
        Value::Array(values) => values
            .iter()
            .any(|value| value_contains_text(value, needle)),
        Value::Object(values) => values
            .values()
            .any(|value| value_contains_text(value, needle)),
        _ => false,
    }
}

fn claude_empty_memory_normalization(
    root: &Map<String, Value>,
    claude_code: bool,
) -> Result<bool, ReplayError> {
    let Some(value) = root
        .get("extra")
        .and_then(|extra| extra.get("riptide_normalization"))
    else {
        return Ok(false);
    };
    let normalization = value
        .as_object()
        .ok_or_else(|| ReplayError::new("riptide_normalization is not an object"))?;
    if normalization.get("kind").and_then(Value::as_str) != Some("claude-empty-memory-remapped") {
        return Ok(false);
    }
    const KEYS: [&str; 5] = [
        "initial_state",
        "kind",
        "remapped_action_count",
        "replay_root",
        "source_root_sha256",
    ];
    const CONFIG_ROOT: &str = "/tmp/riptide-claude-config/";
    const EMPTY_PROBE_ROOT: &str = "/tmp/riptide-claude-config/empty-memory-probe";
    const REPLAY_ROOT: &str = "/tmp/riptide-claude-config/replay-memory";
    if !claude_code
        || normalization.len() != KEYS.len()
        || normalization
            .keys()
            .any(|key| !KEYS.contains(&key.as_str()))
        || normalization.get("initial_state").and_then(Value::as_str)
            != Some("trajectory-proven-empty-or-absent")
        || normalization.get("replay_root").and_then(Value::as_str) != Some(REPLAY_ROOT)
    {
        return Err(ReplayError::new(
            "Claude empty-memory normalization has an unsupported schema",
        ));
    }
    normalization
        .get("source_root_sha256")
        .and_then(Value::as_str)
        .and_then(|value| value.strip_prefix("sha256:"))
        .filter(|value| value.len() == 64 && value.bytes().all(|byte| byte.is_ascii_hexdigit()))
        .ok_or_else(|| {
            ReplayError::new("Claude empty-memory normalization has an invalid source hash")
        })?;
    let expected = normalization
        .get("remapped_action_count")
        .and_then(Value::as_u64)
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| {
            ReplayError::new("Claude empty-memory normalization has an invalid action count")
        })?;
    if root
        .get("subagent_trajectories")
        .and_then(Value::as_array)
        .is_some_and(|children| !children.is_empty())
    {
        return Err(ReplayError::new(
            "Claude empty-memory normalization cannot embed subagent trajectories",
        ));
    }
    let mut actual = 0usize;
    for step in root
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        for call in step
            .get("tool_calls")
            .and_then(Value::as_array)
            .into_iter()
            .flatten()
        {
            let arguments = call.get("arguments").unwrap_or(&Value::Null);
            if value_contains_text(arguments, "/logs/agent")
                || value_contains_text(arguments, "/logs/verifier")
            {
                return Err(ReplayError::new(
                    "Claude empty-memory normalization retained a private log path",
                ));
            }
            let replay_memory = value_contains_text(arguments, REPLAY_ROOT);
            let empty_probe = value_contains_text(arguments, EMPTY_PROBE_ROOT);
            if value_contains_text(arguments, CONFIG_ROOT) && !replay_memory && !empty_probe {
                return Err(ReplayError::new(
                    "Claude empty-memory normalization retained an unknown config path",
                ));
            }
            actual += usize::from(replay_memory || empty_probe);
        }
    }
    if actual != expected {
        return Err(ReplayError::new(
            "Claude empty-memory normalization action count is inconsistent",
        ));
    }
    Ok(true)
}

fn codex_normalization(
    root: &Map<String, Value>,
) -> Result<Option<CodexNormalization>, ReplayError> {
    let Some(value) = root
        .get("extra")
        .and_then(|extra| extra.get("riptide_normalization"))
    else {
        return Ok(None);
    };
    let normalization = value
        .as_object()
        .ok_or_else(|| ReplayError::new("Codex riptide_normalization is not an object"))?;
    const KEYS: [&str; 8] = [
        "kind",
        "native_session_count",
        "normalized_session_count",
        "native_executable_event_count",
        "extracted_executable_event_count",
        "outcome_bound_executable_event_count",
        "outcome_unavailable_executable_event_count",
        "unaccounted_executable_events",
    ];
    if normalization.len() != KEYS.len()
        || normalization
            .keys()
            .any(|key| !KEYS.contains(&key.as_str()))
        || normalization.get("kind").and_then(Value::as_str)
            != Some("codex-native-sessions-flattened")
    {
        return Err(ReplayError::new(
            "Codex riptide_normalization has an unsupported schema",
        ));
    }
    let count = |name: &str| {
        normalization
            .get(name)
            .and_then(Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                ReplayError::new(format!("Codex riptide_normalization has invalid {name}"))
            })
    };
    let native_sessions = count("native_session_count")?;
    let normalized_sessions = count("normalized_session_count")?;
    let native_events = count("native_executable_event_count")?;
    let executable_events = count("extracted_executable_event_count")?;
    let bound_outcomes = count("outcome_bound_executable_event_count")?;
    let unavailable_outcomes = count("outcome_unavailable_executable_event_count")?;
    let unaccounted = count("unaccounted_executable_events")?;
    if native_sessions == 0
        || native_sessions != normalized_sessions
        || native_events != executable_events
        || unaccounted != 0
        || bound_outcomes.checked_add(unavailable_outcomes) != Some(executable_events)
    {
        return Err(ReplayError::new(
            "Codex riptide_normalization counts are inconsistent",
        ));
    }
    if root
        .get("subagent_trajectories")
        .and_then(Value::as_array)
        .is_some_and(|children| !children.is_empty())
    {
        return Err(ReplayError::new(
            "flattened Codex trajectory still embeds subagent trajectories",
        ));
    }
    let mut ids = BTreeSet::new();
    for step in root
        .get("steps")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
    {
        let id = step
            .pointer("/extra/riptide_source_trajectory_id")
            .and_then(Value::as_str)
            .filter(|id| !id.is_empty())
            .ok_or_else(|| ReplayError::new("flattened Codex step has no source trajectory id"))?;
        ids.insert(id.to_owned());
    }
    if ids.len() != normalized_sessions {
        return Err(ReplayError::new(format!(
            "flattened Codex trajectory accounts for {}/{} sessions",
            ids.len(),
            normalized_sessions
        )));
    }
    Ok(Some(CodexNormalization {
        source_ids: ids,
        executable_events,
        bound_outcomes,
    }))
}

struct AuthoritativeInvocation {
    action: CodeModeAction,
    observation: Value,
    source_item_id: Option<String>,
}

// Decode the retained text blocks as data, never as executable JavaScript or Python.
// Harbor versions may store either JSON blocks or Python's repr of those blocks.
fn code_mode_result_payloads(observation: &Value) -> Result<Vec<Value>, ReplayError> {
    let content = observation.get("content").unwrap_or(observation);
    let decoded;
    let content = if let Some(text) = content.as_str() {
        let text = text.trim();
        if text.starts_with('[') || text.starts_with('{') {
            let mut reader = CodeModeCell::new(text, String::new());
            reader.data_literal = true;
            decoded = reader.literal()?;
            reader.skip_space();
            if !reader.rest().is_empty() {
                return Err(ReplayError::new("Codex result has trailing data"));
            }
            &decoded
        } else {
            return Ok(Vec::new());
        }
    } else {
        content
    };
    let Some(blocks) = content.as_array() else {
        return Ok(if content.is_object() {
            vec![content.clone()]
        } else {
            vec![]
        });
    };
    let mut payloads = Vec::new();
    for (index, block) in blocks.iter().enumerate() {
        if !matches!(
            block.get("type").and_then(Value::as_str),
            Some("input_text" | "text")
        ) {
            return Ok(Vec::new());
        }
        let Some(text) = block.get("text").and_then(Value::as_str) else {
            return Ok(Vec::new());
        };
        if index == 0 && text.starts_with("Script completed\n") && text.ends_with("Output:\n") {
            continue;
        }
        // Only whole JSON printouts bind results. Never mine command stdout for JSON.
        // Preserve non-JSON text's slot: a web response must not erase a
        // preceding command result or shift later results to another call.
        payloads
            .push(serde_json::from_str(text).unwrap_or_else(|_| Value::String(text.to_owned())));
    }
    Ok(payloads)
}

fn printed_tool_outcome(payload: &Value, settled: bool) -> Result<Value, ReplayError> {
    let invalid =
        || ReplayError::new("Codex printed tool result has contradictory or invalid fields");
    let payload = if settled {
        match payload.get("status").and_then(Value::as_str) {
            Some("fulfilled")
                if payload.get("value").is_some() && payload.get("reason").is_none() =>
            {
                &payload["value"]
            }
            Some("rejected")
                if payload.get("reason").is_some() && payload.get("value").is_none() =>
            {
                // A rejected tool promise does not establish a shell outcome. Replay
                // the recorded command, but never certify its exit as matching.
                return Ok(
                    json!({"status": "unknown", "returncode": null, "session_id": null,
                    "reason": payload["reason"]}),
                );
            }
            _ => return Err(invalid()),
        }
    } else {
        payload
    };
    let Some(fields) = payload.as_object() else {
        return Ok(Value::Null);
    };
    let exit = fields.get("exit_code").filter(|v| !v.is_null());
    let session = fields.get("session_id").filter(|v| !v.is_null());
    if exit.is_some_and(|v| v.as_i64().is_none()) || (exit.is_some() && session.is_some()) {
        return Err(invalid());
    }
    let session = match session {
        Some(Value::String(id)) if !id.is_empty() => Some(id.clone()),
        Some(value) if value.as_u64().is_some() => Some(value.to_string()),
        Some(_) => return Err(invalid()),
        None => None,
    };
    let code = exit.and_then(Value::as_i64);
    if code.is_none() && session.is_none() {
        return Ok(Value::Null);
    }
    if fields.get("output").is_some_and(|v| !v.is_string()) {
        return Err(invalid());
    }
    Ok(json!({
        "status": match code { Some(0) => "ok", Some(_) => "failed", None => "running" },
        "returncode": code, "session_id": session,
        "output": fields.get("output").cloned().unwrap_or(Value::String(String::new()))
    }))
}

fn trajectory_code_mode_invocations(
    arguments: &Map<String, Value>,
    identity: String,
    observation: &Value,
) -> Result<Vec<AuthoritativeInvocation>, ReplayError> {
    let parsed = codex_exec_actions(arguments, identity)?;
    let actions = parsed.actions;
    let mut outcomes = vec![Value::Null; actions.len()];
    // Older adapters retained a structured outcome directly for one nested call.
    // This is distinct from the outer cell's textual execution summary.
    if actions.len() == 1
        && actions[0].parallel_group_id.is_none()
        && observation.get("status").is_some_and(Value::is_string)
        && (observation.get("returncode").is_some() || observation.get("session_id").is_some())
    {
        outcomes[0] = observation.clone();
    }
    let print = actions
        .first()
        .map_or(ResultPrint::None, |a| a.result_print);
    let indexed = matches!(print, ResultPrint::Indexed | ResultPrint::Settled);
    // A mixed/partial display, or multiple index namespaces, has no proven mapping.
    let bindable = print != ResultPrint::None
        && actions.iter().all(|a| a.result_print == print)
        && (!indexed
            || actions
                .iter()
                .all(|a| a.parallel_group_id == actions[0].parallel_group_id));
    if bindable && (indexed || parsed.direct_prints.is_some()) {
        let payloads = code_mode_result_payloads(observation)?;
        if !payloads.is_empty() {
            let expected = if indexed {
                actions.len()
            } else {
                parsed.direct_prints.as_ref().unwrap().len()
            };
            if payloads.len() != expected {
                return Err(ReplayError::new(
                    "Codex printed result count does not match nested calls",
                ));
            }
            let mut seen = BTreeSet::new();
            for (position, payload) in payloads.iter().enumerate() {
                let index = if indexed {
                    payload
                        .get("i")
                        .and_then(Value::as_u64)
                        .and_then(|n| usize::try_from(n).ok())
                        .filter(|n| *n < actions.len())
                        .ok_or_else(|| ReplayError::new("Codex printed result has invalid index"))?
                } else {
                    let Some(index) = parsed.direct_prints.as_ref().unwrap()[position] else {
                        continue;
                    };
                    index
                };
                if !seen.insert(index) {
                    return Err(ReplayError::new("Codex printed results repeat an index"));
                }
                outcomes[index] = printed_tool_outcome(payload, print == ResultPrint::Settled)?;
            }
        }
    }
    Ok(actions
        .into_iter()
        .zip(outcomes)
        .map(|(action, observation)| AuthoritativeInvocation {
            action,
            observation,
            source_item_id: None,
        })
        .collect())
}

fn authoritative_code_mode_invocations(
    tool_call: &Map<String, Value>,
    seen_source_items: &mut BTreeSet<String>,
    seen_parallel_groups: &mut BTreeSet<String>,
) -> Result<Option<Vec<AuthoritativeInvocation>>, ReplayError> {
    let extra = tool_call.get("extra").and_then(Value::as_object);
    if extra.is_some_and(|extra| extra.contains_key("riptide_nested_outcomes")) {
        return Err(ReplayError::new(
            "Codex riptide_nested_outcomes was replaced by riptide_nested_invocations",
        ));
    }
    let Some(value) = extra.and_then(|extra| extra.get("riptide_nested_invocations")) else {
        return Ok(None);
    };
    let values = value
        .as_array()
        .ok_or_else(|| ReplayError::new("Codex riptide_nested_invocations is not an array"))?;
    if values.is_empty() {
        return Err(ReplayError::new(
            "Codex riptide_nested_invocations must not be empty",
        ));
    }
    let mut invocations = Vec::with_capacity(values.len());
    let mut local_groups = BTreeMap::<String, (usize, usize, String, BTreeSet<String>)>::new();
    for (index, value) in values.iter().enumerate() {
        let invocation = value.as_object().ok_or_else(|| {
            ReplayError::new("Codex riptide_nested_invocations entry is not an object")
        })?;
        const KEYS: [&str; 6] = [
            "index",
            "tool",
            "arguments",
            "source_item_id",
            "outcome",
            "parallel_group_id",
        ];
        let expected_len = 5 + usize::from(invocation.contains_key("parallel_group_id"));
        if invocation.len() != expected_len
            || invocation.keys().any(|key| !KEYS.contains(&key.as_str()))
            || invocation.get("index").and_then(Value::as_u64) != Some(index as u64)
        {
            return Err(ReplayError::new(
                "Codex authoritative invocation has invalid keys or index",
            ));
        }
        let tool = invocation
            .get("tool")
            .and_then(Value::as_str)
            .filter(|tool| matches!(*tool, "exec_command" | "write_stdin" | "apply_patch"))
            .ok_or_else(|| ReplayError::new("Codex authoritative invocation has invalid tool"))?;
        let arguments = invocation
            .get("arguments")
            .and_then(Value::as_object)
            .cloned()
            .ok_or_else(|| {
                ReplayError::new("Codex authoritative invocation arguments are not an object")
            })?;
        let valid_arguments = match tool {
            "exec_command" => arguments.get("cmd").is_some_and(Value::is_string),
            "write_stdin" => arguments.get("session_id").is_some_and(|value| {
                value.as_u64().is_some()
                    || value.as_str().is_some_and(|session| !session.is_empty())
            }),
            "apply_patch" => {
                arguments.len() == 1 && arguments.get("input").is_some_and(Value::is_string)
            }
            _ => unreachable!(),
        };
        if !valid_arguments {
            return Err(ReplayError::new(
                "Codex authoritative invocation has invalid canonical arguments",
            ));
        }
        let source_item_id = invocation
            .get("source_item_id")
            .and_then(Value::as_str)
            .filter(|value| {
                !value.is_empty() && !value.chars().any(|character| character.is_control())
            })
            .ok_or_else(|| {
                ReplayError::new("Codex authoritative invocation has invalid source_item_id")
            })?
            .to_owned();
        if !seen_source_items.insert(source_item_id.clone()) {
            return Err(ReplayError::new(
                "Codex authoritative invocation repeats a source_item_id",
            ));
        }
        let parallel_group_id = match invocation.get("parallel_group_id") {
            None => None,
            Some(Value::String(value))
                if !value.is_empty()
                    && !value.chars().any(|character| character.is_control())
                    && ((tool == "exec_command"
                        && arguments.get("run_in_background") != Some(&Value::Bool(true))
                        && !arguments.contains_key("session_id"))
                        || (tool == "write_stdin"
                            && arguments.get("chars").and_then(Value::as_str) == Some(""))) =>
            {
                let entry = local_groups
                    .entry(value.clone())
                    .or_insert_with(|| (0, index, tool.to_owned(), BTreeSet::new()));
                if entry.1 + entry.0 != index {
                    return Err(ReplayError::new(
                        "Codex authoritative parallel group is not contiguous",
                    ));
                }
                if entry.2 != tool {
                    return Err(ReplayError::new(
                        "Codex authoritative parallel group mixes operation types",
                    ));
                }
                if tool == "write_stdin" {
                    let session = arguments["session_id"].to_string();
                    if !entry.3.insert(session) {
                        return Err(ReplayError::new(
                            "Codex authoritative parallel poll group repeats a session",
                        ));
                    }
                }
                entry.0 += 1;
                Some(value.clone())
            }
            Some(_) => {
                return Err(ReplayError::new(
                    "Codex authoritative invocation has an unsafe parallel group",
                ))
            }
        };
        let observation = match invocation.get("outcome") {
            Some(Value::Null) => Value::Null,
            Some(Value::Object(outcome)) => {
                const REQUIRED_OUTCOME_KEYS: [&str; 3] = ["status", "returncode", "session_id"];
                const OUTCOME_KEYS: [&str; 4] = ["status", "returncode", "session_id", "output"];
                if !REQUIRED_OUTCOME_KEYS
                    .iter()
                    .all(|key| outcome.contains_key(*key))
                    || outcome
                        .keys()
                        .any(|key| !OUTCOME_KEYS.contains(&key.as_str()))
                    || outcome
                        .get("output")
                        .is_some_and(|output| !output.is_string())
                {
                    return Err(ReplayError::new(
                        "Codex authoritative outcome has invalid keys",
                    ));
                }
                let status = outcome
                    .get("status")
                    .and_then(Value::as_str)
                    .filter(|status| {
                        matches!(
                            *status,
                            "ok" | "failed" | "running" | "timed_out" | "skipped_not_executed"
                        )
                    })
                    .ok_or_else(|| {
                        ReplayError::new("Codex authoritative outcome has invalid status")
                    })?;
                let returncode = outcome.get("returncode").ok_or_else(|| {
                    ReplayError::new("Codex authoritative outcome has no returncode")
                })?;
                let session_id = outcome.get("session_id").ok_or_else(|| {
                    ReplayError::new("Codex authoritative outcome has no session_id")
                })?;
                if (!returncode.is_null() && returncode.as_i64().is_none())
                    || (!session_id.is_null()
                        && !session_id
                            .as_str()
                            .is_some_and(|session| !session.is_empty()))
                    || (status == "ok" && returncode.as_i64().is_some_and(|code| code != 0))
                    || (status == "failed" && returncode.as_i64() == Some(0))
                    || (status == "running"
                        && (returncode.as_i64().is_some() || session_id.is_null()))
                    || (matches!(status, "timed_out" | "skipped_not_executed")
                        && !returncode.is_null())
                {
                    return Err(ReplayError::new(
                        "Codex authoritative outcome has contradictory fields",
                    ));
                }
                Value::Object(outcome.clone())
            }
            _ => {
                return Err(ReplayError::new(
                    "Codex authoritative invocation has invalid outcome",
                ))
            }
        };
        invocations.push(AuthoritativeInvocation {
            action: CodeModeAction {
                tool: tool.to_owned(),
                arguments,
                parallel_group_id,
                result_print: ResultPrint::None,
            },
            observation,
            source_item_id: Some(source_item_id),
        });
    }
    for (group, (count, _, _, _)) in local_groups {
        if count < 2 || !seen_parallel_groups.insert(group) {
            return Err(ReplayError::new(
                "Codex authoritative parallel group is singleton or reused",
            ));
        }
    }
    Ok(Some(invocations))
}

/// Load executable tool calls from an ATIF trajectory document.
pub fn read_trajectory(path: &Path) -> Result<AtifTrajectory, ReplayError> {
    let raw = read_trajectory_bytes(path)?;
    let payload: Value = serde_json::from_str(&String::from_utf8_lossy(&raw))
        .map_err(|error| ReplayError::new(format!("ATIF trajectory is invalid JSON: {}", error)))?;
    let root = payload
        .as_object()
        .ok_or_else(|| ReplayError::new("ATIF trajectory root is not a JSON object"))?;
    let schema = root
        .get("schema_version")
        .and_then(Value::as_str)
        .filter(|schema| schema.starts_with("ATIF-v1."))
        .ok_or_else(|| {
            ReplayError::new(format!(
                "unsupported ATIF trajectory schema: {}",
                root.get("schema_version").unwrap_or(&Value::Null)
            ))
        })?
        .to_owned();
    let agent = root
        .get("agent")
        .and_then(Value::as_object)
        .cloned()
        .unwrap_or_default();
    let root_claude_code = is_claude_code(&agent);
    let original_format = if root_claude_code {
        Some(CLAUDE_CODE_AGENT_NAME.to_owned())
    } else {
        source_agent_original_format(&agent)
    };
    let claude_memory_normalized = claude_empty_memory_normalization(root, root_claude_code)?;
    let normalization = if claude_memory_normalized {
        None
    } else {
        codex_normalization(root)?
    };
    if normalization.is_some() && !is_codex_code_mode(&agent) {
        return Err(ReplayError::new(
            "Codex normalization requires a Code Mode trajectory",
        ));
    }
    let mut nodes = Vec::new();
    collect_trajectory_nodes(
        root,
        true,
        normalization.as_ref().map(|value| &value.source_ids),
        &mut BTreeSet::new(),
        &mut nodes,
    )?;
    let subagent_trajectories_flattened = normalization
        .as_ref()
        .map_or(nodes.len() - 1, |value| value.source_ids.len() - 1);
    let contains_codex_exec = nodes.iter().any(|node| {
        node.value
            .get("agent")
            .and_then(Value::as_object)
            .is_some_and(is_codex_code_mode)
    });
    let mut actions = Vec::new();
    let mut background_task_ids = BTreeSet::new();
    let mut claude_orchestration_ids = BTreeSet::new();
    let mut claude_auxiliary_call_ids = BTreeSet::new();
    let mut source_item_ids = BTreeSet::new();
    let mut parallel_group_ids = BTreeSet::new();
    let mut authoritative_events = 0usize;
    let mut authoritative_outcomes = 0usize;
    for node in nodes {
        let node_agent = node
            .value
            .get("agent")
            .and_then(Value::as_object)
            .cloned()
            .unwrap_or_default();
        let code_mode = is_codex_code_mode(&node_agent);
        let claude_code = is_claude_code(&node_agent);
        let node_original_format = if claude_code {
            Some(CLAUDE_CODE_AGENT_NAME.to_owned())
        } else {
            source_agent_original_format(&node_agent)
        };
        let steps = node
            .value
            .get("steps")
            .and_then(Value::as_array)
            .ok_or_else(|| ReplayError::new("ATIF trajectory has no steps list"))?;
        for (step_offset, step) in steps.iter().enumerate() {
            let step_number = step_offset + 1;
            let step = step.as_object().ok_or_else(|| {
                ReplayError::new(format!("ATIF step {step_number} is not a JSON object"))
            })?;
            let empty = Vec::new();
            let tool_calls = match step.get("tool_calls") {
                None | Some(Value::Null) => &empty,
                Some(Value::Array(tool_calls)) => tool_calls,
                Some(_) => {
                    return Err(ReplayError::new(format!(
                        "ATIF step {step_number} has invalid tool_calls"
                    )))
                }
            };
            let telemetry_only = match step
                .get("extra")
                .and_then(|extra| extra.get("riptide_telemetry_only"))
            {
                None => false,
                Some(Value::Bool(true)) if normalization.is_some() && tool_calls.len() == 1 => true,
                Some(_) => {
                    return Err(ReplayError::new(
                        "Codex telemetry-only step has an invalid marker or tool count",
                    ))
                }
            };
            for (tool_call_offset, tool_call) in tool_calls.iter().enumerate() {
                let tool_call = tool_call.as_object().ok_or_else(|| {
                    ReplayError::new(format!(
                        "ATIF step {step_number} contains an invalid tool call"
                    ))
                })?;
                let source_tool = tool_call
                    .get("function_name")
                    .and_then(Value::as_str)
                    .filter(|value| !value.is_empty())
                    .or_else(|| {
                        tool_call
                            .get("name")
                            .and_then(Value::as_str)
                            .filter(|value| !value.is_empty())
                    })
                    .ok_or_else(|| {
                        ReplayError::new(format!(
                            "ATIF step {step_number} tool call has no function name"
                        ))
                    })?;
                let arguments = match tool_call.get("arguments") {
                    None | Some(Value::Null) => Map::new(),
                    Some(Value::Object(arguments)) => arguments.clone(),
                    Some(_) => {
                        return Err(ReplayError::new(format!(
                            "ATIF tool {source_tool:?} arguments are not an object"
                        )))
                    }
                };
                let has_authoritative_data = tool_call
                    .get("extra")
                    .and_then(Value::as_object)
                    .is_some_and(|extra| {
                        extra.contains_key("riptide_nested_invocations")
                            || extra.contains_key("riptide_nested_outcomes")
                    });
                if has_authoritative_data && !(code_mode && source_tool == "exec") {
                    return Err(ReplayError::new(
                        "Codex authoritative invocations require an outer Code Mode exec",
                    ));
                }
                if telemetry_only && !(code_mode && source_tool == "exec") {
                    return Err(ReplayError::new(
                        "Codex telemetry-only step requires one outer Code Mode exec",
                    ));
                }
                if claude_code
                    && !matches!(
                        source_tool,
                        "Bash"
                            | "Write"
                            | "Edit"
                            | "Read"
                            | "Grep"
                            | "Glob"
                            | "TaskCreate"
                            | "TaskUpdate"
                            | "TaskList"
                            | "TaskGet"
                            | "TaskStop"
                            | "TaskOutput"
                            | "ToolSearch"
                            | "Monitor"
                            | "ScheduleWakeup"
                            | "WebFetch"
                            | "WebSearch"
                            | "TodoWrite"
                            | "AskUserQuestion"
                            | "Agent"
                    )
                {
                    return Err(ReplayError::new(format!(
                        "unsupported Claude Code tool: {source_tool}"
                    )));
                }
                if claude_code
                    && source_tool == "Bash"
                    && arguments.get("dangerouslyDisableSandbox") == Some(&Value::Bool(true))
                {
                    return Err(ReplayError::new(
                        "unsupported Claude Code Bash payload: dangerouslyDisableSandbox=true",
                    ));
                }
                let mini_swe = node_original_format.as_deref()
                    == Some(MINI_SWE_AGENT_ORIGINAL_FORMAT)
                    && source_tool == "bash";
                let observation = if mini_swe {
                    mini_swe_call_observation(
                        step,
                        tool_call.get("tool_call_id").and_then(Value::as_str),
                    )?
                } else if claude_code {
                    call_observation(step, tool_call.get("tool_call_id").and_then(Value::as_str))?
                } else {
                    step.get("observation").cloned().unwrap_or(Value::Null)
                };
                if mini_swe {
                    mini_swe_result(&observation)?;
                }
                let consistent_result_error = if claude_code {
                    claude_consistent_result_is_error(&observation)?
                } else {
                    None
                };
                if claude_code && matches!(source_tool, "Agent" | "Monitor") {
                    let call_id = tool_call
                        .get("tool_call_id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .ok_or_else(|| ReplayError::new("Claude auxiliary call has no identity"))?;
                    if !claude_auxiliary_call_ids.insert(call_id.to_owned()) {
                        return Err(ReplayError::new(
                            "Claude auxiliary call reuses a tool_call_id",
                        ));
                    }
                }
                if claude_code && source_tool == "Agent" {
                    validate_claude_agent_parent(steps, step_offset, &arguments, &observation)?;
                }
                if claude_code && source_tool == "Monitor" {
                    let id = claude_monitor_poll(&arguments, &observation, &background_task_ids)?;
                    if background_task_ids.contains(&id) || !claude_orchestration_ids.insert(id) {
                        return Err(ReplayError::new(
                            "Claude Monitor reuses a background or orchestration task identity",
                        ));
                    }
                }
                let background_id =
                    (claude_code && source_tool == "Bash" && consistent_result_error != Some(true))
                        .then(|| claude_background_task_id(&observation))
                        .flatten();
                claude_bash_timeout(&arguments)?;
                if claude_code
                    && matches!(source_tool, "TaskStop" | "TaskOutput")
                    && consistent_result_error != Some(true)
                {
                    let task_id = arguments
                        .get("task_id")
                        .and_then(Value::as_str)
                        .filter(|id| !id.is_empty())
                        .ok_or_else(|| {
                            ReplayError::new(format!("Claude Code {source_tool} has no task_id"))
                        })?;
                    if claude_task_result_id(&observation) != Some(task_id) {
                        return Err(ReplayError::new(format!(
                            "Claude Code {source_tool} task_id does not match its observation"
                        )));
                    }
                }
                if claude_code
                    && source_tool == "Bash"
                    && (background_id.is_some()
                        || arguments.get("run_in_background") == Some(&Value::Bool(true)))
                    && consistent_result_error != Some(true)
                {
                    let id = background_id.ok_or_else(|| {
                        ReplayError::new("Claude Code background Bash has no backgroundTaskId")
                    })?;
                    if claude_orchestration_ids.contains(id)
                        || !background_task_ids.insert(id.to_owned())
                    {
                        return Err(ReplayError::new(format!(
                            "Claude Code background Bash reuses backgroundTaskId {id:?}"
                        )));
                    }
                }
                let step_id = step
                    .get("step_id")
                    .cloned()
                    .unwrap_or_else(|| Value::Number(Number::from(step_number as u64)));
                let source_trajectory_id = step
                    .get("extra")
                    .and_then(|extra| extra.get("riptide_source_trajectory_id"))
                    .and_then(Value::as_str)
                    .filter(|id| !id.is_empty())
                    .unwrap_or(&node.id)
                    .to_owned();
                let timestamp = timestamp_sec(
                    step.get("timestamp").unwrap_or(&Value::Null),
                    actions.len() + 1,
                )?;
                if code_mode && source_tool == "exec" {
                    let authoritative = authoritative_code_mode_invocations(
                        tool_call,
                        &mut source_item_ids,
                        &mut parallel_group_ids,
                    )?;
                    let nested = match (normalization.is_some(), authoritative) {
                        (true, Some(invocations)) => invocations,
                        (true, None) => {
                            let structural_identity =
                                format!("step-{step_number}-call-{}", tool_call_offset + 1);
                            codex_exec_actions(&arguments, structural_identity)?
                                .actions
                                .into_iter()
                                .map(|action| AuthoritativeInvocation {
                                    action,
                                    observation: Value::Null,
                                    source_item_id: None,
                                })
                                .collect()
                        }
                        (false, Some(_)) => {
                            return Err(ReplayError::new(
                                "authoritative Codex invocations require normalized provenance",
                            ))
                        }
                        (false, None) => {
                            let structural_identity =
                                format!("step-{step_number}-call-{}", tool_call_offset + 1);
                            let outer_call_identity = tool_call
                                .get("tool_call_id")
                                .and_then(Value::as_str)
                                .filter(|id| !id.is_empty())
                                .map(|id| format!("{structural_identity}:{id}"))
                                .unwrap_or(structural_identity);
                            let observation = if step
                                .get("observation")
                                .and_then(|v| v.get("results"))
                                .is_some()
                            {
                                call_observation(
                                    step,
                                    tool_call.get("tool_call_id").and_then(Value::as_str),
                                )?
                            } else if tool_calls.len() == 1 {
                                observation.clone()
                            } else {
                                Value::Null
                            };
                            trajectory_code_mode_invocations(
                                &arguments,
                                outer_call_identity,
                                &observation,
                            )?
                        }
                    };
                    let input = arguments.get("input").and_then(Value::as_str);
                    if telemetry_only
                        && (input != Some("")
                            || nested.len() != 1
                            || nested[0].action.parallel_group_id.is_some())
                    {
                        return Err(ReplayError::new(
                            "Codex telemetry-only step is not one sequential synthetic invocation",
                        ));
                    }
                    if normalization.is_some() && input.is_none() {
                        return Err(ReplayError::new(
                            "normalized Codex exec has no string input",
                        ));
                    }
                    for invocation in nested {
                        authoritative_events += usize::from(invocation.source_item_id.is_some());
                        authoritative_outcomes += usize::from(
                            invocation.source_item_id.is_some()
                                && !invocation.observation.is_null(),
                        );
                        let nested_action = invocation.action;
                        let nested_observation = invocation.observation;
                        let index = actions.len() + 1;
                        actions.push(AtifAction {
                            index,
                            step_id: step_id.clone(),
                            timestamp_sec: timestamp,
                            source_trajectory_id: source_trajectory_id.clone(),
                            source_item_id: invocation.source_item_id,
                            tool: nested_action.tool.clone(),
                            source_tool: format!("exec.{}", nested_action.tool),
                            code_mode: true,
                            parallel_group_id: nested_action.parallel_group_id,
                            source_agent_original_format: node_original_format.clone(),
                            arguments: nested_action.arguments,
                            observation_text: text_content(&nested_observation),
                            observation: nested_observation,
                        });
                    }
                } else {
                    let index = actions.len() + 1;
                    let code_mode_metadata = code_mode && is_code_mode_metadata_tool(source_tool);
                    let orchestration_metadata = claude_code
                        && matches!(source_tool, "TaskStop" | "TaskOutput")
                        && arguments
                            .get("task_id")
                            .and_then(Value::as_str)
                            .is_some_and(|id| claude_orchestration_ids.contains(id));
                    actions.push(AtifAction {
                        index,
                        step_id,
                        timestamp_sec: timestamp,
                        source_trajectory_id,
                        source_item_id: None,
                        tool: if orchestration_metadata {
                            "claude_orchestration".to_owned()
                        } else {
                            canonical_tool(
                                source_tool,
                                node_original_format.as_deref(),
                                claude_code,
                            )
                        },
                        source_tool: if code_mode_metadata {
                            format!("exec.{source_tool}")
                        } else {
                            source_tool.to_owned()
                        },
                        code_mode: code_mode_metadata,
                        parallel_group_id: None,
                        source_agent_original_format: node_original_format.clone(),
                        arguments: canonical_arguments(
                            arguments,
                            source_tool,
                            node_original_format.as_deref(),
                            claude_code,
                        ),
                        observation_text: text_content(&observation),
                        observation,
                    });
                }
            }
        }
    }
    if let Some(normalization) = &normalization {
        let replayable_events = actions
            .iter()
            .filter(|action| is_supported_tool(&action.tool))
            .count();
        if authoritative_events != normalization.executable_events
            || replayable_events != normalization.executable_events
            || authoritative_outcomes != normalization.bound_outcomes
        {
            return Err(ReplayError::new(
                "normalized Codex trajectory counts do not match Rust-expanded actions",
            ));
        }
    }
    if actions.is_empty() {
        return Err(ReplayError::new("ATIF trajectory contains no tool actions"));
    }
    actions.sort_by(|left, right| {
        left.timestamp_sec
            .total_cmp(&right.timestamp_sec)
            .then_with(|| left.index.cmp(&right.index))
    });
    for (offset, action) in actions.iter_mut().enumerate() {
        action.index = offset + 1;
    }
    Ok(AtifTrajectory {
        source_path: path.to_path_buf(),
        schema_version: schema,
        agent,
        source_agent_original_format: original_format,
        session_id: root
            .get("session_id")
            .and_then(Value::as_str)
            .map(str::to_owned),
        subagent_trajectories_flattened,
        actions,
        contains_codex_exec,
    })
}

pub fn validate_supported_actions(trajectory: &AtifTrajectory) -> Result<(), ReplayError> {
    if trajectory.contains_codex_exec
        || is_codex_code_mode(&trajectory.agent)
        || trajectory.actions.iter().any(|action| action.code_mode)
    {
        return Err(ReplayError::new(
            "Codex exec ATIF replay and profiling are not supported in release 26.10",
        ));
    }
    if is_claude_code(&trajectory.agent)
        || trajectory.source_agent_original_format.as_deref() == Some(CLAUDE_CODE_AGENT_NAME)
        || trajectory.actions.iter().any(|action| {
            action.source_agent_original_format.as_deref() == Some(CLAUDE_CODE_AGENT_NAME)
                || action.tool.starts_with("claude_")
        })
    {
        return Err(ReplayError::new(
            "Claude Code ATIF replay and profiling are not supported in release 26.10",
        ));
    }
    for action in trajectory
        .actions
        .iter()
        .filter(|action| action.tool == "exec_command")
    {
        execution_shell(action)?;
    }
    let unsafe_parallel_groups: BTreeSet<&str> = trajectory
        .actions
        .iter()
        .filter(|action| {
            action.parallel_group_id.is_some()
                && !(action.code_mode
                    && (is_metadata_action(action)
                        || (action.tool == "exec_command"
                            && action.arguments.get("run_in_background")
                                != Some(&Value::Bool(true))
                            && !action.arguments.contains_key("session_id"))
                        || (action.tool == "write_stdin"
                            && action.arguments.get("chars").and_then(Value::as_str) == Some("")
                            && action.arguments.contains_key("session_id"))))
        })
        .filter_map(|action| action.parallel_group_id.as_deref())
        .collect();
    if !unsafe_parallel_groups.is_empty() {
        return Err(ReplayError::new(format!(
            "ATIF trajectory contains parallel groups that cannot be replayed sequentially: {}",
            unsafe_parallel_groups
                .into_iter()
                .collect::<Vec<_>>()
                .join(", ")
        )));
    }
    let mut parallel_poll_sessions = BTreeMap::<&str, BTreeSet<String>>::new();
    for action in trajectory
        .actions
        .iter()
        .filter(|action| action.parallel_group_id.is_some() && action.tool == "write_stdin")
    {
        let group = action.parallel_group_id.as_deref().unwrap();
        if !parallel_poll_sessions
            .entry(group)
            .or_default()
            .insert(action.arguments["session_id"].to_string())
        {
            return Err(ReplayError::new(format!(
                "ATIF parallel poll group repeats a session: {group}"
            )));
        }
    }
    let unsupported: BTreeSet<&str> = trajectory
        .actions
        .iter()
        .filter(|action| !is_supported_tool(&action.tool) && !is_metadata_action(action))
        .map(|action| action.tool.as_str())
        .collect();
    if !unsupported.is_empty() {
        return Err(ReplayError::new(format!(
            "ATIF trajectory contains unsupported executable tool calls: {}",
            unsupported.into_iter().collect::<Vec<_>>().join(", ")
        )));
    }
    if !trajectory
        .actions
        .iter()
        .any(|action| is_supported_tool(&action.tool))
    {
        return Err(ReplayError::new(
            "ATIF trajectory contains no supported executable tool calls",
        ));
    }
    Ok(())
}

fn preview(value: &str) -> String {
    value.chars().take(PREVIEW_LENGTH).collect()
}

fn recorded_tool_wall_sec(action: &AtifAction) -> Option<f64> {
    let expression = RegexBuilder::new(r"Wall time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds")
        .case_insensitive(true)
        .build()
        .expect("static wall-time regex");
    expression
        .captures(&action.observation_text)
        .and_then(|captures| captures[1].parse::<f64>().ok())
        .filter(|value| value.is_finite() && *value >= 0.0)
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SourceOutcome {
    status: String,
    returncode: Option<i64>,
    session_id: Option<String>,
}

fn mini_swe_result(observation: &Value) -> Result<Map<String, Value>, ReplayError> {
    let content = observation
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| ReplayError::new("mini-SWE result has no structured content"))?;
    let payload: Map<String, Value> = serde_json::from_str(content)
        .map_err(|_| ReplayError::new("mini-SWE result content is not a JSON object"))?;
    let returncode = payload
        .get("returncode")
        .ok_or_else(|| ReplayError::new("mini-SWE result has no returncode"))?;
    let exception = payload
        .get("exception_info")
        .filter(|value| !value.is_null());
    let special_exception = exception.and_then(Value::as_str).is_some_and(|value| {
        value == "action was not executed" || value.to_ascii_lowercase().contains("timed out")
    });
    if (!returncode.is_null() && returncode.as_i64().is_none())
        || exception.is_some_and(|value| !value.is_string())
        || (returncode.is_null() && !special_exception)
        || (returncode.as_i64() == Some(0)
            && exception
                .and_then(Value::as_str)
                .is_some_and(|value| !value.is_empty()))
        || (special_exception && !returncode.is_null() && returncode.as_i64() != Some(-1))
    {
        return Err(ReplayError::new(
            "mini-SWE result has invalid or conflicting outcome evidence",
        ));
    }
    Ok(payload)
}

/// Use shell diagnostics only when the tool did not preserve a structured outcome.
fn shell_command_not_found(action: &AtifAction) -> bool {
    if action.source_tool != "Bash" && action.tool != "exec_command" {
        return false;
    }
    action.observation_text.lines().map(str::trim).any(|line| {
        line.contains("command not found")
            && (line.starts_with("/bin/bash:")
                || line.starts_with("/bin/sh:")
                || line.starts_with("bash:")
                || line.starts_with("sh:"))
    })
}

fn source_argument_parse_failed(action: &AtifAction) -> bool {
    if !action
        .observation_text
        .to_ascii_lowercase()
        .contains("failed to parse function arguments")
    {
        return false;
    }
    if !matches!(
        action.source_agent_original_format.as_deref(),
        Some(CLAUDE_CODE_AGENT_NAME) | Some(MINI_SWE_AGENT_ORIGINAL_FORMAT)
    ) {
        return matches!(
            action.tool.as_str(),
            "exec_command" | "write_stdin" | "apply_patch"
        );
    }
    match action.tool.as_str() {
        "exec_command" => !action
            .arguments
            .get("cmd")
            .or_else(|| action.arguments.get("command"))
            .is_some_and(Value::is_string),
        "write_stdin" => !action.arguments.get("session_id").is_some_and(|value| {
            value.as_u64().is_some() || value.as_str().is_some_and(|id| !id.is_empty())
        }),
        "apply_patch" => !action.arguments.get("input").is_some_and(Value::is_string),
        _ => false,
    }
}

fn source_outcome(action: &AtifAction) -> SourceOutcome {
    if action.source_agent_original_format.as_deref() == Some(MINI_SWE_AGENT_ORIGINAL_FORMAT)
        && action.source_tool == "bash"
    {
        if let Ok(result) = mini_swe_result(&action.observation) {
            if let Some(exception) = result.get("exception_info").and_then(Value::as_str) {
                if exception == "action was not executed" {
                    return SourceOutcome {
                        status: "skipped_not_executed".to_owned(),
                        returncode: None,
                        session_id: None,
                    };
                }
                if exception.to_ascii_lowercase().contains("timed out") {
                    return SourceOutcome {
                        status: "timed_out".to_owned(),
                        returncode: None,
                        session_id: None,
                    };
                }
            }
            if let Some(returncode) = result.get("returncode").and_then(Value::as_i64) {
                return SourceOutcome {
                    status: if returncode == 0 { "ok" } else { "failed" }.to_owned(),
                    returncode: Some(returncode),
                    session_id: None,
                };
            }
        }
        return SourceOutcome {
            status: "unknown".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    if is_metadata_action(action) {
        return SourceOutcome {
            status: "ignored_metadata".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    if action.code_mode {
        if let Some(status) = action.observation.get("status").and_then(Value::as_str) {
            return SourceOutcome {
                status: status.to_owned(),
                returncode: action.observation.get("returncode").and_then(Value::as_i64),
                session_id: action
                    .observation
                    .get("session_id")
                    .and_then(session_id_text),
            };
        }
        return SourceOutcome {
            status: "unknown".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    if action.source_agent_original_format.as_deref() == Some(CLAUDE_CODE_AGENT_NAME)
        && action.source_tool == "Bash"
        && claude_background_task_id(&action.observation).is_some()
        && claude_result_is_error(&action.observation) != Some(true)
    {
        return SourceOutcome {
            status: "running".to_owned(),
            returncode: None,
            session_id: claude_background_task_id(&action.observation).map(str::to_owned),
        };
    }
    if action.source_agent_original_format.as_deref() == Some(CLAUDE_CODE_AGENT_NAME)
        && action.source_tool == "TaskOutput"
        && claude_result_is_error(&action.observation) != Some(true)
    {
        let task = action
            .observation
            .pointer("/extra/tool_result_metadata/tool_use_result/task");
        if task
            .and_then(|task| task.get("status"))
            .and_then(Value::as_str)
            == Some("running")
        {
            return SourceOutcome {
                status: "running".to_owned(),
                returncode: None,
                session_id: task
                    .and_then(|task| task.get("task_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            };
        }
        if let Some(returncode) = task
            .and_then(|task| task.get("exitCode"))
            .and_then(Value::as_i64)
        {
            return SourceOutcome {
                status: if returncode == 0 { "ok" } else { "failed" }.to_owned(),
                returncode: Some(returncode),
                session_id: task
                    .and_then(|task| task.get("task_id"))
                    .and_then(Value::as_str)
                    .map(str::to_owned),
            };
        }
    }
    if action.source_agent_original_format.as_deref() == Some(CLAUDE_CODE_AGENT_NAME) {
        let is_error = claude_result_is_error(&action.observation).unwrap_or_else(|| {
            action
                .observation_text
                .to_ascii_lowercase()
                .contains("[error] tool reported failure")
                || shell_command_not_found(action)
        });
        return SourceOutcome {
            status: if is_error { "failed" } else { "ok" }.to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    if shell_command_not_found(action) {
        return SourceOutcome {
            status: "failed".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    if source_argument_parse_failed(action) {
        return SourceOutcome {
            status: "failed".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    let lower = action.observation_text.to_ascii_lowercase();
    if action.tool == "exec_command" && lower.contains("exec_command failed") {
        return SourceOutcome {
            status: "failed".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    if action.tool == "apply_patch"
        && action.observation_text.lines().any(|line| {
            let line = line.to_ascii_lowercase();
            line.contains("apply_patch")
                && ["failed", "failure", "rejected"]
                    .iter()
                    .any(|word| line.contains(word))
        })
    {
        return SourceOutcome {
            status: "failed".to_owned(),
            returncode: None,
            session_id: None,
        };
    }
    let session =
        Regex::new(r"Process running with session ID\s+(\d+)").expect("static session regex");
    if let Some(captures) = session.captures(&action.observation_text) {
        return SourceOutcome {
            status: "running".to_owned(),
            returncode: None,
            session_id: Some(captures[1].to_owned()),
        };
    }
    for pattern in [
        r"Process exited with code\s+(-?\d+)",
        r"Exit code:\s*(-?\d+)",
    ] {
        let expression = Regex::new(pattern).expect("static exit-code regex");
        if let Some(returncode) = expression
            .captures(&action.observation_text)
            .and_then(|captures| captures[1].parse::<i64>().ok())
        {
            return SourceOutcome {
                status: if returncode == 0 { "ok" } else { "failed" }.to_owned(),
                returncode: Some(returncode),
                session_id: None,
            };
        }
    }
    SourceOutcome {
        status: "unknown".to_owned(),
        returncode: None,
        session_id: None,
    }
}

fn outcome_matches(
    expected_status: &str,
    expected_returncode: Option<i64>,
    status: &str,
    returncode: Option<i64>,
) -> Option<bool> {
    if expected_status == "unknown" {
        return None;
    }
    if expected_status == "running" && matches!(status, "ok" | "failed") {
        return Some(true);
    }
    if expected_status != status {
        return Some(false);
    }
    if expected_status == "failed" && expected_returncode == Some(-1) {
        return Some(returncode.map(|code| code != 0).unwrap_or(false));
    }
    Some(
        expected_returncode
            .map(|expected| returncode == Some(expected))
            .unwrap_or(true),
    )
}

fn status_matches(expected_status: &str, status: &str) -> Option<bool> {
    (expected_status != "unknown").then_some(expected_status == status)
}

#[derive(Clone, Debug, Eq, PartialEq)]
enum PatchKind {
    Add,
    Update,
    Delete,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct PatchOperation {
    kind: PatchKind,
    path: String,
    lines: Vec<String>,
    move_to: Option<String>,
}

fn parse_patch(patch_text: &str) -> Result<Vec<PatchOperation>, ReplayError> {
    let lines: Vec<&str> = patch_text.lines().collect();
    if lines.first() != Some(&"*** Begin Patch") || lines.last() != Some(&"*** End Patch") {
        return Err(ReplayError::new(
            "apply_patch payload is missing Begin Patch or End Patch markers",
        ));
    }
    let mut operations = Vec::new();
    let mut index = 1;
    while index < lines.len() - 1 {
        let header = lines[index];
        index += 1;
        let (kind, path) = if let Some(path) = header.strip_prefix("*** Add File: ") {
            (PatchKind::Add, path)
        } else if let Some(path) = header.strip_prefix("*** Update File: ") {
            (PatchKind::Update, path)
        } else if let Some(path) = header.strip_prefix("*** Delete File: ") {
            operations.push(PatchOperation {
                kind: PatchKind::Delete,
                path: path.to_owned(),
                lines: Vec::new(),
                move_to: None,
            });
            continue;
        } else {
            return Err(ReplayError::new(format!(
                "unsupported apply_patch operation: {header:?}"
            )));
        };
        let mut move_to = None;
        if kind == PatchKind::Update && index < lines.len() - 1 {
            if let Some(destination) = lines[index].strip_prefix("*** Move to: ") {
                move_to = Some(destination.to_owned());
                index += 1;
            }
        }
        let mut operation_lines = Vec::new();
        while index < lines.len() - 1
            && !lines[index].starts_with("*** Add File: ")
            && !lines[index].starts_with("*** Update File: ")
            && !lines[index].starts_with("*** Delete File: ")
        {
            operation_lines.push(lines[index].to_owned());
            index += 1;
        }
        operations.push(PatchOperation {
            kind,
            path: path.to_owned(),
            lines: operation_lines,
            move_to,
        });
    }
    Ok(operations)
}

fn resolve_allow_missing(path: &Path) -> io::Result<PathBuf> {
    if let Ok(path) = fs::canonicalize(path) {
        return Ok(path);
    }
    let mut cursor = path;
    let mut suffix = Vec::<OsString>::new();
    loop {
        if cursor.exists() {
            let mut resolved = fs::canonicalize(cursor)?;
            for part in suffix.iter().rev() {
                resolved.push(part);
            }
            return Ok(resolved);
        }
        let name = cursor.file_name().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("unable to resolve path: {}", path.display()),
            )
        })?;
        suffix.push(name.to_os_string());
        cursor = cursor.parent().ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("unable to resolve path: {}", path.display()),
            )
        })?;
    }
}

#[derive(Clone, Copy)]
enum PathAccess {
    Read,
    Write,
    ReadWrite,
}

#[derive(Clone, Debug)]
struct ReplayPathPolicy {
    workspace_root: PathBuf,
    readable_roots: Vec<PathBuf>,
    writable_roots: Vec<PathBuf>,
    exact_readable_paths: BTreeSet<PathBuf>,
    exact_writable_paths: BTreeSet<PathBuf>,
}

impl ReplayPathPolicy {
    fn from_roots(cwd: &Path, roots: &[PathBuf]) -> Result<Self, ReplayError> {
        let workspace_root = resolve_allow_missing(cwd)?;
        let roots = roots
            .iter()
            .map(|root| resolve_allow_missing(root).map_err(ReplayError::from))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            workspace_root: workspace_root.clone(),
            readable_roots: std::iter::once(workspace_root.clone())
                .chain(roots.iter().cloned())
                .collect(),
            writable_roots: std::iter::once(workspace_root).chain(roots).collect(),
            exact_readable_paths: BTreeSet::new(),
            exact_writable_paths: BTreeSet::new(),
        })
    }

    fn for_trajectory(
        cwd: &Path,
        roots: &[PathBuf],
        trajectory: &AtifTrajectory,
    ) -> Result<Self, ReplayError> {
        let mut policy = Self::from_roots(cwd, roots)?;
        for action in &trajectory.actions {
            match action.tool.as_str() {
                "claude_read" => {
                    if let Some(path) = action.arguments.get("file_path").and_then(Value::as_str) {
                        if claude_task_output_path(path).is_none() {
                            policy.grant_recorded_absolute(path, cwd, PathAccess::Read, false)?;
                        }
                    }
                }
                "claude_write" => {
                    if let Some(path) = action.arguments.get("file_path").and_then(Value::as_str) {
                        policy.grant_recorded_absolute(path, cwd, PathAccess::Write, false)?;
                    }
                }
                "claude_edit" => {
                    if let Some(path) = action.arguments.get("file_path").and_then(Value::as_str) {
                        policy.grant_recorded_absolute(path, cwd, PathAccess::ReadWrite, false)?;
                    }
                }
                "claude_grep" | "claude_glob" => {
                    if let Some(path) = action.arguments.get("path").and_then(Value::as_str) {
                        policy.grant_recorded_absolute(path, cwd, PathAccess::Read, true)?;
                    }
                }
                "apply_patch" => {
                    let Some(patch) = action
                        .arguments
                        .get("input")
                        .or_else(|| action.arguments.get("patch"))
                        .and_then(Value::as_str)
                    else {
                        continue;
                    };
                    let Ok(operations) = parse_patch(patch) else {
                        continue;
                    };
                    for operation in operations {
                        let access = if operation.kind == PatchKind::Add {
                            PathAccess::Write
                        } else {
                            PathAccess::ReadWrite
                        };
                        policy.grant_recorded_absolute(&operation.path, cwd, access, false)?;
                        if let Some(destination) = operation.move_to {
                            policy.grant_recorded_absolute(
                                &destination,
                                cwd,
                                PathAccess::Write,
                                false,
                            )?;
                        }
                    }
                }
                _ => {}
            }
        }
        Ok(policy)
    }

    fn grant_recorded_absolute(
        &mut self,
        raw: &str,
        cwd: &Path,
        access: PathAccess,
        subtree: bool,
    ) -> Result<(), ReplayError> {
        let raw = Path::new(raw);
        if !raw.is_absolute() || raw.starts_with(cwd) || raw.starts_with(&self.workspace_root) {
            return Ok(());
        }
        let path = resolve_allow_missing(raw)?;
        let (read, write) = match access {
            PathAccess::Read => (true, false),
            PathAccess::Write => (false, true),
            PathAccess::ReadWrite => (true, true),
        };
        if subtree {
            if read {
                self.readable_roots.push(path.clone());
            }
            if write {
                self.writable_roots.push(path);
            }
        } else {
            if read {
                self.exact_readable_paths.insert(path.clone());
            }
            if write {
                self.exact_writable_paths.insert(path);
            }
        }
        Ok(())
    }

    fn resolve(&self, raw: &str, cwd: &Path, access: PathAccess) -> Result<PathBuf, ReplayError> {
        let raw_path = Path::new(raw);
        let candidate = if raw_path.is_absolute() {
            raw_path.to_path_buf()
        } else {
            cwd.join(raw_path)
        };
        let path = resolve_allow_missing(&candidate)?;
        let allowed = |roots: &[PathBuf], exact: &BTreeSet<PathBuf>| {
            exact.contains(&path)
                || roots
                    .iter()
                    .any(|root| path == *root || path.starts_with(root))
        };
        let readable = allowed(&self.readable_roots, &self.exact_readable_paths);
        let writable = allowed(&self.writable_roots, &self.exact_writable_paths);
        if match access {
            PathAccess::Read => readable,
            PathAccess::Write => writable,
            PathAccess::ReadWrite => readable && writable,
        } {
            Ok(path)
        } else {
            Err(ReplayError::new(format!(
                "path leaves the permitted replay roots: {raw}"
            )))
        }
    }
}

fn claude_task_output_path(raw: &str) -> Option<String> {
    let path = Path::new(raw);
    let file = path.file_name()?.to_str()?.strip_suffix(".output")?;
    (path.parent()?.file_name()? == OsStr::new("tasks")
        && !file.is_empty()
        && file
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_' | b'.')))
    .then(|| file.to_owned())
}

fn claude_recorded_task_output_path(action: &AtifAction, task_id: &str) -> Option<PathBuf> {
    const MARKER: &str = "Output is being written to: ";
    action.observation_text.lines().find_map(|line| {
        let (_, raw) = line.split_once(MARKER)?;
        let raw = raw.trim();
        let path = PathBuf::from(raw);
        (Regex::new(
            r"^/tmp/(?:claude|claude-[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*)/tasks/[A-Za-z0-9._-]+\.output$",
        )
        .expect("static Claude task output path regex")
        .is_match(raw)
            && !raw.split('/').any(|part| part == "..")
            && resolve_allow_missing(path.parent()?).ok().as_deref() == path.parent()
            && claude_task_output_path(raw).as_deref() == Some(task_id))
        .then_some(path)
    })
}

fn find_subsequence(lines: &[String], wanted: &[String], start: usize) -> Option<usize> {
    if wanted.is_empty() {
        return Some(start);
    }
    if wanted.len() > lines.len() {
        return None;
    }
    (start..=lines.len() - wanted.len())
        .find(|index| lines[*index..*index + wanted.len()] == *wanted)
}

fn split_lines(text: &str) -> Vec<String> {
    text.lines().map(str::to_owned).collect()
}

fn apply_update(path: &Path, lines: &[String]) -> Result<(), ReplayError> {
    if !path.is_file() {
        return Err(ReplayError::new(format!(
            "cannot update missing file: {}",
            path.display()
        )));
    }
    let source_text = fs::read_to_string(path)?;
    let mut source_ends_newline = source_text.ends_with('\n');
    let mut content = split_lines(&source_text);
    let mut hunks: Vec<Vec<String>> = Vec::new();
    let mut current: Option<Vec<String>> = None;
    for line in lines {
        if line.starts_with("@@") {
            if let Some(previous) = current.replace(Vec::new()) {
                hunks.push(previous);
            }
        } else if let Some(current) = current.as_mut() {
            current.push(line.clone());
        } else {
            return Err(ReplayError::new(format!(
                "update for {} contains text before its first hunk",
                path.display()
            )));
        }
    }
    if let Some(current) = current {
        hunks.push(current);
    }
    if hunks.is_empty() {
        return Err(ReplayError::new(format!(
            "update for {} contains no hunks",
            path.display()
        )));
    }
    let mut cursor = 0;
    for hunk in hunks {
        let mut before = Vec::new();
        let mut after = Vec::new();
        for line in hunk {
            if let Some(line) = line.strip_prefix(' ') {
                before.push(line.to_owned());
                after.push(line.to_owned());
            } else if let Some(line) = line.strip_prefix('-') {
                before.push(line.to_owned());
            } else if let Some(line) = line.strip_prefix('+') {
                after.push(line.to_owned());
            } else if line == r"\ No newline at end of file" {
                source_ends_newline = false;
            } else {
                return Err(ReplayError::new(format!(
                    "invalid update hunk line for {}: {line:?}",
                    path.display()
                )));
            }
        }
        let location = find_subsequence(&content, &before, cursor).ok_or_else(|| {
            ReplayError::new(format!(
                "update hunk context was not found in {}",
                path.display()
            ))
        })?;
        content.splice(location..location + before.len(), after.iter().cloned());
        cursor = location + after.len();
    }
    let mut output = content.join("\n");
    if source_ends_newline {
        output.push('\n');
    }
    fs::write(path, output)?;
    Ok(())
}

fn apply_patch_payload_with_policy(
    patch_text: &str,
    cwd: &Path,
    policy: &ReplayPathPolicy,
) -> Result<(), ReplayError> {
    for operation in parse_patch(patch_text)? {
        let access = if operation.kind == PatchKind::Add {
            PathAccess::Write
        } else {
            PathAccess::ReadWrite
        };
        let path = policy.resolve(&operation.path, cwd, access)?;
        match operation.kind {
            PatchKind::Add => {
                if path.exists() {
                    return Err(ReplayError::new(format!(
                        "cannot add existing file: {}",
                        path.display()
                    )));
                }
                let mut added = Vec::new();
                for line in operation.lines {
                    let line = line.strip_prefix('+').ok_or_else(|| {
                        ReplayError::new(format!(
                            "invalid add-file patch line for {}: {line:?}",
                            path.display()
                        ))
                    })?;
                    added.push(line.to_owned());
                }
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                let mut output = added.join("\n");
                if !added.is_empty() {
                    output.push('\n');
                }
                fs::write(path, output)?;
            }
            PatchKind::Delete => {
                if !path.exists() {
                    return Err(ReplayError::new(format!(
                        "cannot delete missing file: {}",
                        path.display()
                    )));
                }
                fs::remove_file(path)?;
            }
            PatchKind::Update => {
                apply_update(&path, &operation.lines)?;
                if let Some(destination) = operation.move_to {
                    let destination = policy.resolve(&destination, cwd, PathAccess::Write)?;
                    if let Some(parent) = destination.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    fs::rename(path, destination)?;
                }
            }
        }
    }
    Ok(())
}

fn session_yield_sec(arguments: &Map<String, Value>) -> f64 {
    arguments
        .get("yield_time_ms")
        .and_then(Value::as_f64)
        .filter(|value| *value > 0.0)
        .map(|value| value / 1_000.0)
        .unwrap_or(0.0)
}

fn command_cwd(action: &AtifAction, default_cwd: &Path) -> Result<PathBuf, ReplayError> {
    let Some(requested) = action.arguments.get("workdir") else {
        return Ok(default_cwd.to_path_buf());
    };
    let requested = requested.as_str().ok_or_else(|| {
        ReplayError::new(format!(
            "ATIF action {} workdir is not a string",
            action.index
        ))
    })?;
    let requested = Path::new(requested);
    let workdir = if requested.is_absolute() {
        requested.to_path_buf()
    } else {
        default_cwd.join(requested)
    };
    if !workdir.is_dir() {
        return Err(ReplayError::new(format!(
            "ATIF action {} workdir does not exist: {}",
            action.index,
            workdir.display()
        )));
    }
    Ok(workdir)
}

fn workload_environment() -> Option<BTreeMap<OsString, OsString>> {
    let marker = env::var_os(CONTROLLER_ENV_VARS_MARKER)?;
    if marker.is_empty() {
        return None;
    }
    let mut environment: BTreeMap<OsString, OsString> = env::vars_os().collect();
    environment.remove(OsStr::new(CONTROLLER_ENV_VARS_MARKER));
    for name in marker.to_string_lossy().split(':') {
        let original_name = OsString::from(format!("{WORKLOAD_ENV_PREFIX}{name}"));
        let presence_name = OsString::from(format!("{WORKLOAD_ENV_PRESENT_PREFIX}{name}"));
        let presence_marker = environment.remove(&presence_name);
        let original_value = environment.remove(&original_name);
        let had_preserved_value = original_value.is_some();
        let was_present = match presence_marker {
            None => {
                had_preserved_value
                    && original_value
                        .as_ref()
                        .map(|value| !value.is_empty())
                        .unwrap_or(false)
            }
            Some(value) => value == OsStr::new("1"),
        };
        if was_present {
            environment.insert(OsString::from(name), original_value.unwrap_or_default());
        } else {
            environment.remove(OsStr::new(name));
        }
    }
    environment.retain(|name, _| !name.to_string_lossy().starts_with(WORKLOAD_ENV_PREFIX));
    Some(environment)
}

fn configure_environment(command: &mut Command) {
    if let Some(environment) = workload_environment() {
        command.env_clear().envs(environment);
    }
}

fn configure_new_session(command: &mut Command) {
    unsafe {
        command.pre_exec(|| {
            if libc::setsid() >= 0 {
                Ok(())
            } else {
                Err(io::Error::last_os_error())
            }
        });
    }
}

fn process_returncode(status: ExitStatus) -> i64 {
    status
        .code()
        .map(i64::from)
        .or_else(|| status.signal().map(|signal| -i64::from(signal)))
        .unwrap_or(-1)
}

fn kill_process_group(child: &Child, signal: i32) -> io::Result<()> {
    let pid = i32::try_from(child.id()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "child PID does not fit in pid_t",
        )
    })?;
    let result = unsafe { libc::kill(-pid, signal) };
    if result == 0 {
        Ok(())
    } else {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct ProcIdentity {
    pid: i32,
    parent_pid: i32,
    process_group: i32,
    start_time_ticks: u64,
}

fn parse_proc_stat(pid: i32, contents: &str) -> io::Result<ProcIdentity> {
    // The comm field is parenthesized and may itself contain spaces or `)`.
    // Everything after its final `) ` has fixed whitespace-delimited fields:
    // state is field 3, ppid field 4, and starttime field 22.
    let close = contents.rfind(") ").ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("/proc/{pid}/stat has no closing comm delimiter"),
        )
    })?;
    let fields: Vec<&str> = contents[close + 2..].split_whitespace().collect();
    if fields.len() <= 19 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("/proc/{pid}/stat is missing required fields"),
        ));
    }
    let parent_pid = fields[1].parse::<i32>().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("/proc/{pid}/stat has invalid ppid: {error}"),
        )
    })?;
    let start_time_ticks = fields[19].parse::<u64>().map_err(|error| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("/proc/{pid}/stat has invalid starttime: {error}"),
        )
    })?;
    Ok(ProcIdentity {
        pid,
        parent_pid,
        process_group: fields[2]
            .parse()
            .map_err(|error| io::Error::new(io::ErrorKind::InvalidData, error))?,
        start_time_ticks,
    })
}

fn read_proc_identity(pid: i32) -> io::Result<Option<ProcIdentity>> {
    match fs::read_to_string(format!("/proc/{pid}/stat")) {
        Ok(contents) => parse_proc_stat(pid, &contents).map(Some),
        Err(error)
            if error.kind() == io::ErrorKind::NotFound
                || error.raw_os_error() == Some(libc::ESRCH) =>
        {
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn read_proc_table() -> io::Result<BTreeMap<i32, ProcIdentity>> {
    let mut table = BTreeMap::new();
    for entry in fs::read_dir("/proc")? {
        let entry = entry?;
        let Some(pid) = entry
            .file_name()
            .to_str()
            .and_then(|name| name.parse::<i32>().ok())
            .filter(|pid| *pid > 0)
        else {
            continue;
        };
        match read_proc_identity(pid) {
            Ok(Some(identity)) => {
                table.insert(pid, identity);
            }
            Ok(None) => {}
            // Processes can disappear or become inaccessible between the
            // directory walk and stat read. An unreadable unrelated process
            // cannot be established as part of this action's ancestry.
            Err(error)
                if matches!(
                    error.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::PermissionDenied
                ) => {}
            Err(error) => return Err(error),
        }
    }
    Ok(table)
}

fn open_pidfd(identity: ProcIdentity) -> io::Result<Option<OwnedFd>> {
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_open as libc::c_long,
            identity.pid as libc::pid_t,
            0u32,
        )
    };
    if result < 0 {
        let error = io::Error::last_os_error();
        return match error.raw_os_error() {
            Some(libc::ESRCH) => Ok(None),
            _ => Err(error),
        };
    }
    let descriptor = unsafe { OwnedFd::from_raw_fd(result as i32) };
    match read_proc_identity(identity.pid)? {
        Some(current) if current.start_time_ticks == identity.start_time_ticks => {
            Ok(Some(descriptor))
        }
        // The numeric PID exited or was reused around pidfd_open. Closing the
        // descriptor without signaling is the only safe response.
        _ => Ok(None),
    }
}

fn pidfd_send_signal(descriptor: &OwnedFd, signal: i32) -> io::Result<()> {
    let result = unsafe {
        libc::syscall(
            libc::SYS_pidfd_send_signal as libc::c_long,
            descriptor.as_raw_fd(),
            signal,
            std::ptr::null::<libc::siginfo_t>(),
            0u32,
        )
    };
    if result == 0 {
        Ok(())
    } else {
        let error = io::Error::last_os_error();
        if error.raw_os_error() == Some(libc::ESRCH) {
            Ok(())
        } else {
            Err(error)
        }
    }
}

fn pidfd_has_exited(descriptor: &OwnedFd) -> io::Result<bool> {
    let mut pollfd = libc::pollfd {
        fd: descriptor.as_raw_fd(),
        events: libc::POLLIN,
        revents: 0,
    };
    let result = unsafe { libc::poll(&mut pollfd, 1, 0) };
    if result < 0 {
        Err(io::Error::last_os_error())
    } else {
        Ok(result > 0 && pollfd.revents & (libc::POLLIN | libc::POLLHUP) != 0)
    }
}

#[derive(Debug)]
struct TrackedProcess {
    identity: ProcIdentity,
    pidfd: OwnedFd,
    term_signaled: bool,
    kill_signaled: bool,
}

#[derive(Debug)]
struct ProcessTreeTracker {
    processes: BTreeMap<i32, TrackedProcess>,
    leader_reaped: bool,
}

impl ProcessTreeTracker {
    fn capture(root_pid: i32) -> io::Result<Self> {
        let mut tracker = Self {
            processes: BTreeMap::new(),
            leader_reaped: false,
        };
        let table = read_proc_table()?;
        let Some(root) = table.get(&root_pid).copied() else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("live subprocess root {root_pid} is absent from /proc"),
            ));
        };
        tracker.insert_identity(root)?;
        if !tracker.processes.contains_key(&root_pid) {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                format!("live subprocess root {root_pid} exited during pidfd capture"),
            ));
        }
        tracker.discover_from_table(&table)?;
        Ok(tracker)
    }

    fn insert_identity(&mut self, identity: ProcIdentity) -> io::Result<()> {
        if self
            .processes
            .get(&identity.pid)
            .is_some_and(|tracked| tracked.identity.start_time_ticks == identity.start_time_ticks)
        {
            return Ok(());
        }
        if let Some(pidfd) = open_pidfd(identity)? {
            self.processes.insert(
                identity.pid,
                TrackedProcess {
                    identity,
                    pidfd,
                    term_signaled: false,
                    kill_signaled: false,
                },
            );
        }
        Ok(())
    }

    fn discover_from_table(&mut self, table: &BTreeMap<i32, ProcIdentity>) -> io::Result<()> {
        let mut reachable = BTreeSet::new();
        for tracked in self.processes.values() {
            if table.get(&tracked.identity.pid).is_some_and(|current| {
                current.start_time_ticks == tracked.identity.start_time_ticks
            }) && !pidfd_has_exited(&tracked.pidfd)?
            {
                reachable.insert(tracked.identity.pid);
            }
        }
        if reachable.is_empty() {
            return Ok(());
        }

        // Repeated passes intentionally follow the snapshot's full ancestry,
        // including grandchildren whose direct parent exits during capture.
        loop {
            let discovered: Vec<ProcIdentity> = table
                .values()
                .filter(|identity| {
                    !reachable.contains(&identity.pid) && reachable.contains(&identity.parent_pid)
                })
                .copied()
                .collect();
            if discovered.is_empty() {
                break;
            }
            for identity in discovered {
                reachable.insert(identity.pid);
                self.insert_identity(identity)?;
            }
        }
        Ok(())
    }

    fn refresh(&mut self) -> io::Result<()> {
        self.discover_from_table(&read_proc_table()?)
    }

    fn poll_child(&mut self, child: &mut Child) -> io::Result<Option<ExitStatus>> {
        let table = read_proc_table()?;
        if !self.leader_reaped {
            // Even an exited, unreaped leader reserves its PGID. Capture its
            // group members before try_wait can release that numeric identity.
            for identity in table
                .values()
                .filter(|identity| identity.process_group == child.id() as i32)
            {
                self.insert_identity(*identity)?;
            }
        }
        self.discover_from_table(&table)?;
        let status = child.try_wait()?;
        self.leader_reaped |= status.is_some();
        Ok(status)
    }

    fn signal_unsignaled(&mut self, signal: i32) -> io::Result<()> {
        for process in self.processes.values_mut() {
            let already_signaled = match signal {
                libc::SIGTERM => &mut process.term_signaled,
                libc::SIGKILL => &mut process.kill_signaled,
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        "process-tree cleanup accepts only SIGTERM or SIGKILL",
                    ))
                }
            };
            if !*already_signaled {
                pidfd_send_signal(&process.pidfd, signal)?;
                *already_signaled = true;
            }
        }
        Ok(())
    }

    fn all_exited(&self) -> io::Result<bool> {
        for process in self.processes.values() {
            if !pidfd_has_exited(&process.pidfd)? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

fn wait_without_kill(
    child: &mut Child,
    timeout: Option<Duration>,
    mut tracker: Option<&mut ProcessTreeTracker>,
) -> io::Result<Option<ExitStatus>> {
    let started = Instant::now();
    loop {
        let status = match tracker.as_deref_mut() {
            Some(tracker) => tracker.poll_child(child)?,
            None => child.try_wait()?,
        };
        if let Some(status) = status {
            return Ok(Some(status));
        }
        if timeout
            .map(|timeout| started.elapsed() >= timeout)
            .unwrap_or(false)
        {
            return Ok(None);
        }
        let sleep = timeout
            .map(|timeout| timeout.saturating_sub(started.elapsed()).min(POLL_INTERVAL))
            .unwrap_or(POLL_INTERVAL);
        thread::sleep(sleep);
    }
}

fn signal_contained_processes(
    child: &Child,
    leader_unreaped: bool,
    tracker: &mut ProcessTreeTracker,
    signal: i32,
) -> io::Result<()> {
    // An unreaped child reserves its PID. After reaping, only retained pidfds
    // prove ownership; the old numeric PGID may belong to an unrelated process.
    let group_result = if leader_unreaped {
        kill_process_group(child, signal)
    } else {
        Ok(())
    };
    let pidfd_result = tracker.signal_unsignaled(signal);
    group_result.and(pidfd_result)
}

fn wait_for_child_and_containment(
    child: &mut Child,
    status: &mut Option<ExitStatus>,
    tracker: &mut ProcessTreeTracker,
    phase_signal: i32,
    timeout: Duration,
) -> io::Result<bool> {
    let started = Instant::now();
    loop {
        if status.is_none() {
            *status = tracker.poll_child(child)?;
        } else {
            tracker.refresh()?;
        }
        tracker.signal_unsignaled(phase_signal)?;
        if status.is_some() && tracker.all_exited()? {
            return Ok(true);
        }
        if started.elapsed() >= timeout {
            return Ok(false);
        }
        thread::sleep(timeout.saturating_sub(started.elapsed()).min(POLL_INTERVAL));
    }
}

fn terminate_and_wait_for_containment(
    child: &mut Child,
    status: Option<ExitStatus>,
) -> io::Result<ExitStatus> {
    let root_pid = i32::try_from(child.id()).map_err(|_| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            "child PID does not fit in pid_t",
        )
    })?;
    // Capture ancestry before sending TERM, while the action/session leader is
    // normally still alive and the kernel still exposes parent relationships.
    let mut tracker = if status.is_none() {
        ProcessTreeTracker::capture(root_pid)?
    } else {
        // Never derive ancestry from a numeric PID after its leader was reaped.
        ProcessTreeTracker {
            processes: BTreeMap::new(),
            leader_reaped: true,
        }
    };
    terminate_tracked_processes(child, status, &mut tracker)
}

fn terminate_tracked_processes(
    child: &mut Child,
    mut status: Option<ExitStatus>,
    tracker: &mut ProcessTreeTracker,
) -> io::Result<ExitStatus> {
    signal_contained_processes(child, status.is_none(), tracker, libc::SIGTERM)?;
    if wait_for_child_and_containment(
        child,
        &mut status,
        tracker,
        libc::SIGTERM,
        TIMEOUT_TERMINATION_GRACE,
    )? {
        return Ok(status.expect("settled child has an exit status"));
    }
    tracker.refresh()?;
    signal_contained_processes(child, status.is_none(), tracker, libc::SIGKILL)?;
    if !wait_for_child_and_containment(
        child,
        &mut status,
        tracker,
        libc::SIGKILL,
        TIMEOUT_KILL_SETTLE,
    )? {
        return Err(io::Error::new(
            io::ErrorKind::TimedOut,
            "timed-out subprocess containment remained populated after SIGKILL",
        ));
    }
    Ok(status.expect("settled child has an exit status"))
}

/// Timeout cleanup combines the original process group with a /proc ancestry
/// snapshot and starttime-revalidated pidfds. This catches observed descendants
/// that call `setsid()` without risking signals to a reused numeric PID.
///
/// This is the strongest safe in-process fallback, not a containment boundary:
/// a double-forked descendant can reparent before the first snapshot and erase
/// its ancestry without a delegated per-action cgroup or dedicated supervisor.
/// A descendant hidden by /proc access controls likewise cannot be discovered.
/// The disposable Docker cgroup remains the exact, authoritative boundary for
/// the complete profiled replay.
fn wait_with_timeout_cleanup(
    child: &mut Child,
    timeout: Option<Duration>,
) -> io::Result<(ExitStatus, bool)> {
    if let Some(status) = wait_without_kill(child, timeout, None)? {
        return Ok((status, false));
    }
    Ok((terminate_and_wait_for_containment(child, None)?, true))
}

#[derive(Debug)]
struct CapturedOutput {
    returncode: i64,
    process_root_pid: u32,
    stdout: String,
    stderr: String,
    timed_out: bool,
}

struct CapturedCommand<'a> {
    shell: &'a str,
    shell_flag: &'a str,
    command_text: &'a str,
    cwd: &'a Path,
    timeout: Option<Duration>,
    combine_output: bool,
    tty: bool,
    artifacts_dir: &'a Path,
    action_index: usize,
}

fn run_captured_command(request: CapturedCommand<'_>) -> Result<CapturedOutput, ReplayError> {
    let CapturedCommand {
        shell,
        shell_flag,
        command_text,
        cwd,
        timeout,
        combine_output,
        tty,
        artifacts_dir,
        action_index,
    } = request;
    let mut command = Command::new(shell);
    command.arg(shell_flag).arg(command_text).current_dir(cwd);
    configure_environment(&mut command);
    configure_new_session(&mut command);

    let combined_path = artifacts_dir.join(format!(
        ".mini-swe-action-{action_index:04}-{}.capture",
        std::process::id()
    ));
    let stdout_path = artifacts_dir.join(format!(
        ".atif-action-{action_index:04}-{}.stdout.capture",
        std::process::id()
    ));
    let stderr_path = artifacts_dir.join(format!(
        ".atif-action-{action_index:04}-{}.stderr.capture",
        std::process::id()
    ));
    let mut terminal = if tty {
        Some(PtyCapture::attach(
            &mut command,
            File::create(&stdout_path)?,
        )?)
    } else {
        None
    };
    if tty {
        // PTYs merge stdout and stderr at the terminal, as in the source tool.
        File::create(&stderr_path)?;
    } else if combine_output {
        let output = File::create(&combined_path)?;
        command
            .stdout(Stdio::from(output.try_clone()?))
            .stderr(Stdio::from(output));
    } else {
        // Capture into files rather than pipes. A foreground shell may exit after
        // intentionally starting a background service that inherits stdout or
        // stderr. Pipe readers would then wait for that service to exit even
        // though the recorded tool action already completed, preventing later
        // actions from interacting with it and bypassing the action timeout.
        let stdout = File::create(&stdout_path)?;
        let stderr = File::create(&stderr_path)?;
        command
            .stdout(Stdio::from(stdout))
            .stderr(Stdio::from(stderr));
    }
    let mut child = command.spawn()?;
    drop(command); // Close the parent's slave descriptors so terminal EOF works.
    let process_root_pid = child.id();
    let (status, timed_out) = wait_with_timeout_cleanup(&mut child, timeout)?;
    if let Some(terminal) = terminal.as_mut() {
        terminal.finish()?;
    }
    let (stdout, stderr) = if combine_output && !tty {
        let output = fs::read(&combined_path)?;
        let _ = fs::remove_file(&combined_path);
        (output, Vec::new())
    } else {
        let stdout = fs::read(&stdout_path)?;
        let stderr = fs::read(&stderr_path)?;
        let _ = fs::remove_file(&stdout_path);
        let _ = fs::remove_file(&stderr_path);
        (stdout, stderr)
    };
    Ok(CapturedOutput {
        returncode: process_returncode(status),
        process_root_pid,
        stdout: String::from_utf8_lossy(&stdout).into_owned(),
        stderr: String::from_utf8_lossy(&stderr).into_owned(),
        timed_out,
    })
}

#[derive(Debug)]
struct PtyCapture {
    master: File,
    stop: Arc<AtomicBool>,
    reader: Option<thread::JoinHandle<io::Result<()>>>,
}

impl PtyCapture {
    fn attach(command: &mut Command, mut output: File) -> io::Result<Self> {
        let (mut master, mut slave) = (-1, -1);
        let size = libc::winsize {
            ws_row: 24,
            ws_col: 80,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        if unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null(),
                &size,
            )
        } < 0
        {
            return Err(io::Error::last_os_error());
        }
        let master = unsafe { File::from_raw_fd(master) };
        let slave = unsafe { File::from_raw_fd(slave) };
        for fd in [&master, &slave] {
            if unsafe { libc::fcntl(fd.as_raw_fd(), libc::F_SETFD, libc::FD_CLOEXEC) } < 0 {
                return Err(io::Error::last_os_error());
            }
        }
        if unsafe { libc::fcntl(master.as_raw_fd(), libc::F_SETFL, libc::O_NONBLOCK) } < 0 {
            return Err(io::Error::last_os_error());
        }
        command
            .stdin(Stdio::from(slave.try_clone()?))
            .stdout(Stdio::from(slave.try_clone()?))
            .stderr(Stdio::from(slave));
        // configure_new_session runs first. Acquire the slave as that session's
        // controlling terminal; kernel line discipline then handles Ctrl-C/EOF.
        unsafe {
            command.pre_exec(|| {
                if libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    Err(io::Error::last_os_error())
                } else {
                    Ok(())
                }
            });
        }
        let mut input = master.try_clone()?;
        let stop = Arc::new(AtomicBool::new(false));
        let stopping = stop.clone();
        let reader = thread::Builder::new()
            .name("replay-pty".into())
            .spawn(move || {
                let mut buffer = [0u8; 65536];
                let mut finishing = None;
                let mut capture_error = None;
                loop {
                    if stopping.load(Ordering::Acquire) {
                        let since = finishing.get_or_insert_with(Instant::now);
                        // Drain buffered output, but never wait on a background
                        // descendant that still owns a slave descriptor.
                        if since.elapsed() >= Duration::from_millis(50) {
                            return capture_error.map_or(Ok(()), Err);
                        }
                    }
                    let mut poll = libc::pollfd {
                        fd: input.as_raw_fd(),
                        events: libc::POLLIN,
                        revents: 0,
                    };
                    let ready = unsafe {
                        libc::poll(&mut poll, 1, if finishing.is_some() { 0 } else { 50 })
                    };
                    if ready < 0 {
                        let error = io::Error::last_os_error();
                        if error.kind() == io::ErrorKind::Interrupted {
                            continue;
                        }
                        return Err(error);
                    }
                    if ready == 0 {
                        if finishing.is_some() {
                            return capture_error.map_or(Ok(()), Err);
                        }
                        continue;
                    }
                    match input.read(&mut buffer) {
                        Ok(0) => return capture_error.map_or(Ok(()), Err),
                        Ok(count) => {
                            // Keep draining after a disk error, otherwise a full
                            // terminal buffer could block the child forever.
                            if capture_error.is_none() {
                                capture_error = output.write_all(&buffer[..count]).err();
                            }
                        }
                        Err(error) if error.raw_os_error() == Some(libc::EIO) => {
                            return capture_error.map_or(Ok(()), Err)
                        }
                        Err(error)
                            if matches!(
                                error.kind(),
                                io::ErrorKind::Interrupted | io::ErrorKind::WouldBlock
                            ) =>
                        {
                            continue
                        }
                        Err(error) => return Err(error),
                    }
                }
            })?;
        Ok(Self {
            master,
            stop,
            reader: Some(reader),
        })
    }

    fn write_input(&mut self, mut bytes: &[u8], timeout: Duration) -> io::Result<bool> {
        let started = Instant::now();
        while !bytes.is_empty() {
            if started.elapsed() >= timeout {
                return Ok(false);
            }
            match self.master.write(bytes) {
                Ok(0) => return Err(io::Error::new(io::ErrorKind::WriteZero, "PTY input closed")),
                Ok(count) => bytes = &bytes[count..],
                Err(error) if error.kind() == io::ErrorKind::Interrupted => continue,
                Err(error) if error.kind() == io::ErrorKind::WouldBlock => {
                    let mut poll = libc::pollfd {
                        fd: self.master.as_raw_fd(),
                        events: libc::POLLOUT,
                        revents: 0,
                    };
                    let wait = timeout
                        .saturating_sub(started.elapsed())
                        .as_millis()
                        .min(50) as i32;
                    if unsafe { libc::poll(&mut poll, 1, wait) } < 0 {
                        let error = io::Error::last_os_error();
                        if error.kind() != io::ErrorKind::Interrupted {
                            return Err(error);
                        }
                    }
                }
                Err(error) => return Err(error),
            }
        }
        Ok(true)
    }

    fn finish(&mut self) -> io::Result<()> {
        self.stop.store(true, Ordering::Release);
        if let Some(reader) = self.reader.take() {
            reader.join().map_err(|_| {
                io::Error::new(io::ErrorKind::Other, "PTY output reader panicked")
            })??;
        }
        Ok(())
    }
}

impl Drop for PtyCapture {
    fn drop(&mut self) {
        let _ = self.finish();
    }
}

#[derive(Debug)]
struct RunningSession {
    child: Child,
    log_path: PathBuf,
    terminal: Option<PtyCapture>,
    tracker: ProcessTreeTracker,
}

impl RunningSession {
    fn poll(&mut self) -> io::Result<Option<ExitStatus>> {
        self.tracker.poll_child(&mut self.child)
    }
}

fn terminate_session(session: &mut RunningSession) -> io::Result<()> {
    let status = session.poll()?;
    if status.is_none() || !session.tracker.all_exited()? {
        terminate_tracked_processes(&mut session.child, status, &mut session.tracker)?;
    }
    session.child.stdin.take();
    if let Some(terminal) = session.terminal.as_mut() {
        terminal.finish()?;
    }
    Ok(())
}

fn session_output_preview(session: &mut RunningSession) -> Result<Value, ReplayError> {
    if session.poll()?.is_some() {
        if let Some(terminal) = session.terminal.as_mut() {
            terminal.finish()?;
        }
    }
    Ok(Value::String(preview(&String::from_utf8_lossy(&fs::read(
        &session.log_path,
    )?))))
}

fn claude_task_id(action: &AtifAction) -> Result<&str, ReplayError> {
    action
        .arguments
        .get("task_id")
        .and_then(Value::as_str)
        .filter(|id| !id.is_empty())
        .ok_or_else(|| ReplayError::new(format!("ATIF action {} has no task_id", action.index)))
}

fn process_status(returncode: Option<i64>) -> &'static str {
    match returncode {
        None => "running",
        Some(0) => "ok",
        Some(_) => "failed",
    }
}

fn claude_task_output(
    action: &AtifAction,
    timeout_sec: Option<f64>,
    sessions: &mut HashMap<String, RunningSession>,
) -> Result<Map<String, Value>, ReplayError> {
    let task_id = claude_task_id(action)?;
    let block = action
        .arguments
        .get("block")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let timeout = action.arguments.get("timeout").map_or(Ok(None), |value| {
        value
            .as_f64()
            .filter(|timeout| timeout.is_finite() && *timeout > 0.0)
            .map(|timeout| Some(timeout / 1000.0))
            .ok_or_else(|| {
                ReplayError::new("Claude Code TaskOutput timeout must be a positive number")
            })
    })?;
    let session = sessions.get_mut(task_id).ok_or_else(|| {
        ReplayError::new(format!(
            "Claude Code TaskOutput references unknown background task {task_id:?}"
        ))
    })?;
    let wait = match (timeout, timeout_sec) {
        (Some(task), Some(global)) => Some(task.min(global)),
        (Some(task), None) => Some(task),
        (None, Some(global)) => Some(global),
        (None, None) => Some(DEFAULT_WRITE_STDIN_YIELD_SEC),
    };
    let status = if block {
        wait_without_kill(
            &mut session.child,
            wait.map(Duration::from_secs_f64),
            Some(&mut session.tracker),
        )?
    } else {
        session.poll()?
    };
    let returncode = status.map(process_returncode);
    Ok(Map::from_iter([
        ("status".to_owned(), json!(process_status(returncode))),
        ("returncode".to_owned(), json!(returncode)),
        ("session_id".to_owned(), json!(task_id)),
        (
            "session_output_preview".to_owned(),
            session_output_preview(session)?,
        ),
    ]))
}

fn insert(map: &mut Map<String, Value>, key: &str, value: impl Into<Value>) {
    map.insert(key.to_owned(), value.into());
}

fn deterministic_mktemp_command(
    action: &AtifAction,
    command_text: &str,
    expected: &SourceOutcome,
) -> Result<Option<String>, ReplayError> {
    let expression =
        Regex::new(r"^mktemp -d (/tmp/[A-Za-z0-9._/-]*XXXXXX)$").expect("static mktemp regex");
    let Some(captures) = expression.captures(command_text) else {
        return Ok(None);
    };
    if expected.status != "ok" || expected.returncode != Some(0) {
        return Ok(None);
    }
    let template = &captures[1];
    let prefix = template
        .strip_suffix("XXXXXX")
        .expect("regex requires suffix");
    let mut candidates = action
        .observation_text
        .match_indices(prefix)
        .filter_map(|(start, _)| {
            let suffix = action.observation_text[start + prefix.len()..].as_bytes();
            (suffix.len() >= 6
                && suffix[..6].iter().all(u8::is_ascii_alphanumeric)
                && suffix
                    .get(6)
                    .map_or(true, |byte| !byte.is_ascii_alphanumeric()))
            .then(|| &action.observation_text[start..start + prefix.len() + 6])
        })
        .collect::<Vec<_>>();
    candidates.sort_unstable();
    candidates.dedup();
    if candidates.len() != 1 {
        return Err(ReplayError::new(format!(
            "ATIF action {} successful mktemp has no unique recorded /tmp result",
            action.index
        )));
    }
    let recorded = candidates[0];
    Ok(Some(format!(
        "mkdir -m 700 -- '{recorded}' && printf '%s\\n' '{recorded}'"
    )))
}

fn execution_shell(action: &AtifAction) -> Result<(&str, &str), ReplayError> {
    let mini_swe = action.source_agent_original_format.as_deref()
        == Some(MINI_SWE_AGENT_ORIGINAL_FORMAT)
        && action.source_tool == "bash";
    let default_shell = if mini_swe { "/bin/sh" } else { "/bin/bash" };
    let shell = match action.arguments.get("shell") {
        None => default_shell,
        Some(Value::String(shell))
            if !shell.contains('\0')
                && Path::new(shell).is_absolute()
                && matches!(
                    Path::new(shell).file_name().and_then(OsStr::to_str),
                    Some("bash" | "sh" | "dash" | "zsh")
                ) =>
        {
            shell
        }
        Some(_) => {
            return Err(ReplayError::new(
                "exec_command shell must be an absolute bash, sh, dash, or zsh path",
            ))
        }
    };
    let login = match action.arguments.get("login") {
        None => {
            !mini_swe
                && action.source_agent_original_format.as_deref() != Some(CLAUDE_CODE_AGENT_NAME)
        }
        Some(Value::Bool(login)) => *login,
        Some(_) => return Err(ReplayError::new("exec_command login must be boolean")),
    };
    match action.arguments.get("tty") {
        None | Some(Value::Bool(_)) => {}
        Some(_) => return Err(ReplayError::new("exec_command tty must be boolean")),
    }
    Ok((shell, if login { "-lc" } else { "-c" }))
}

fn execute_command(
    action: &AtifAction,
    cwd: &Path,
    timeout_sec: Option<f64>,
    expected: &SourceOutcome,
    sessions: &mut HashMap<String, RunningSession>,
    artifacts_dir: &Path,
    action_output_dir: Option<&Path>,
) -> Result<Map<String, Value>, ReplayError> {
    let command_text = action
        .arguments
        .get("cmd")
        .or_else(|| action.arguments.get("command"))
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ReplayError::new(format!(
                "ATIF action {} exec_command has no string cmd",
                action.index
            ))
        })?;
    let mini_swe = action.source_agent_original_format.as_deref()
        == Some(MINI_SWE_AGENT_ORIGINAL_FORMAT)
        && action.source_tool == "bash";
    let (shell, shell_flag) = execution_shell(action)?;
    let tty = action
        .arguments
        .get("tty")
        .and_then(Value::as_bool)
        .unwrap_or(false);
    let command_cwd = match command_cwd(action, cwd) {
        Ok(workdir) => workdir,
        Err(error) => {
            // A source failure without an exit code does not prove that its
            // working-directory validation failed before the command ran.
            let result = Map::from_iter([
                ("command".to_owned(), json!(command_text)),
                ("status".to_owned(), json!("failed")),
                ("returncode".to_owned(), Value::Null),
                ("stderr_preview".to_owned(), json!(error.to_string())),
                ("launch_failed".to_owned(), json!(true)),
                ("command_executed".to_owned(), json!(false)),
            ]);
            return Ok(result);
        }
    };
    if expected.status == "running" {
        let session_id = expected.session_id.as_deref().ok_or_else(|| {
            ReplayError::new(format!(
                "ATIF action {} has no source session id",
                action.index
            ))
        })?;
        let log_path = artifacts_dir.join(format!("session-{session_id}.log"));
        let log = File::options()
            .create(true)
            .truncate(true)
            .read(true)
            .write(true)
            .open(&log_path)?;
        if let Some(recorded_path) = claude_recorded_task_output_path(action, session_id) {
            let parent = recorded_path.parent().ok_or_else(|| {
                ReplayError::new("Claude Code background task output has no parent directory")
            })?;
            fs::create_dir_all(parent)?;
            if recorded_path.exists() || recorded_path.is_symlink() {
                return Err(ReplayError::new(format!(
                    "Claude Code background task output path already exists: {}",
                    recorded_path.display()
                )));
            }
            std::os::unix::fs::symlink(fs::canonicalize(&log_path)?, &recorded_path)?;
        }
        let recorded_timeout = claude_bash_timeout(&action.arguments)?;
        let mut command = if let Some(timeout) = recorded_timeout {
            let mut command = Command::new("timeout");
            command
                .arg("--signal=TERM")
                .arg(format!(
                    "--kill-after={}s",
                    TIMEOUT_TERMINATION_GRACE.as_secs_f64()
                ))
                .arg(format!("{timeout}s"))
                .arg(shell)
                .arg(shell_flag)
                .arg(command_text);
            command
        } else {
            let mut command = Command::new(shell);
            command.arg(shell_flag).arg(command_text);
            command
        };
        command
            .current_dir(command_cwd)
            .stdin(Stdio::piped())
            .stdout(Stdio::from(log.try_clone()?))
            .stderr(Stdio::from(log.try_clone()?));
        configure_environment(&mut command);
        configure_new_session(&mut command);
        let terminal = if tty {
            Some(PtyCapture::attach(&mut command, log)?)
        } else {
            None
        };
        let mut child = command.spawn()?;
        drop(command);
        let process_root_pid = child.id();
        let tracker = match ProcessTreeTracker::capture(process_root_pid as i32) {
            Ok(tracker) => tracker,
            Err(error) => {
                kill_process_group(&child, libc::SIGKILL)?;
                child.wait()?;
                return Err(error.into());
            }
        };
        sessions.insert(
            session_id.to_owned(),
            RunningSession {
                child,
                log_path: log_path.clone(),
                terminal,
                tracker,
            },
        );
        let session = sessions
            .get_mut(session_id)
            .expect("session inserted immediately above");
        let yielded = session_yield_sec(&action.arguments);
        let status = if yielded > 0.0 {
            wait_without_kill(
                &mut session.child,
                Some(Duration::from_secs_f64(yielded)),
                Some(&mut session.tracker),
            )?
        } else {
            session.poll()?
        };
        let returncode = status.map(process_returncode);
        let status = process_status(returncode);
        return Ok(Map::from_iter([
            ("command".to_owned(), json!(command_text)),
            ("status".to_owned(), json!(status)),
            ("returncode".to_owned(), json!(returncode)),
            ("process_root_pid".to_owned(), json!(process_root_pid)),
            ("session_id".to_owned(), json!(session_id)),
            (
                "session_output_preview".to_owned(),
                session_output_preview(session)?,
            ),
        ]));
    }

    let recorded_timeout = if action.source_agent_original_format.as_deref()
        == Some(CLAUDE_CODE_AGENT_NAME)
        && action.source_tool == "Bash"
    {
        claude_bash_timeout(&action.arguments)?
    } else {
        None
    };
    let timeout = match (timeout_sec, recorded_timeout) {
        (Some(global), Some(recorded)) => Some(global.min(recorded)),
        (global, recorded) => global.or(recorded),
    }
    .map(Duration::from_secs_f64);
    let replay_command = deterministic_mktemp_command(action, command_text, expected)?;
    let output = run_captured_command(CapturedCommand {
        shell,
        shell_flag,
        command_text: replay_command.as_deref().unwrap_or(command_text),
        cwd: &command_cwd,
        timeout,
        combine_output: mini_swe,
        tty,
        artifacts_dir,
        action_index: action.index,
    })?;
    let mut result = Map::new();
    insert(&mut result, "command", command_text.to_owned());
    insert(&mut result, "process_root_pid", output.process_root_pid);
    if let Some(output_dir) = action_output_dir {
        fs::create_dir_all(output_dir)?;
        if mini_swe {
            let path = output_dir.join(format!("action-{:04}.output.txt", action.index));
            fs::write(&path, &output.stdout)?;
            insert(
                &mut result,
                "output_path",
                path.to_string_lossy().into_owned(),
            );
        } else {
            let stdout_path = output_dir.join(format!("action-{:04}.stdout.txt", action.index));
            let stderr_path = output_dir.join(format!("action-{:04}.stderr.txt", action.index));
            fs::write(&stdout_path, &output.stdout)?;
            fs::write(&stderr_path, &output.stderr)?;
            insert(
                &mut result,
                "stdout_path",
                stdout_path.to_string_lossy().into_owned(),
            );
            insert(
                &mut result,
                "stderr_path",
                stderr_path.to_string_lossy().into_owned(),
            );
        }
    }
    if output.timed_out {
        insert(&mut result, "status", "timed_out");
    } else {
        insert(
            &mut result,
            "status",
            if output.returncode == 0 {
                "ok"
            } else {
                "failed"
            },
        );
        insert(&mut result, "returncode", output.returncode);
    }
    insert(&mut result, "stdout_preview", preview(&output.stdout));
    insert(&mut result, "stderr_preview", preview(&output.stderr));
    Ok(result)
}

fn session_id_text(value: &Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value.clone()),
        Value::Number(value) => Some(value.to_string()),
        _ => None,
    }
}

fn write_stdin(
    action: &AtifAction,
    timeout_sec: Option<f64>,
    _expected: &SourceOutcome,
    sessions: &mut HashMap<String, RunningSession>,
) -> Result<Map<String, Value>, ReplayError> {
    let session_id = action
        .arguments
        .get("session_id")
        .and_then(session_id_text)
        .ok_or_else(|| {
            ReplayError::new(format!(
                "ATIF action {} write_stdin has no session_id",
                action.index
            ))
        })?;
    let Some(session) = sessions.get_mut(&session_id) else {
        // Earlier replay divergence can prevent a source-recorded session from
        // being created. Retain that as an action failure so the report still
        // accounts for the remainder of the trajectory.
        return Ok(Map::from_iter([
            ("status".to_owned(), json!("failed")),
            ("returncode".to_owned(), json!(-1)),
            ("session_id".to_owned(), json!(session_id.clone())),
            (
                "stderr_preview".to_owned(),
                json!(format!(
                    "ATIF action {} references unknown session {session_id}",
                    action.index
                )),
            ),
        ]));
    };
    let chars = match action.arguments.get("chars") {
        None => String::new(),
        Some(Value::String(chars)) => chars.clone(),
        Some(_) => {
            return Err(ReplayError::new(format!(
                "ATIF action {} write_stdin chars are not a string",
                action.index
            )))
        }
    };
    let mut status = session.poll()?;
    let interrupted = chars == "\u{3}";
    let input_started = Instant::now();
    let mut input_timed_out = false;
    if let Some(terminal) = session
        .terminal
        .as_mut()
        .filter(|_| status.is_none() && !chars.is_empty())
    {
        input_timed_out = !terminal.write_input(
            chars.as_bytes(),
            Duration::from_secs_f64(timeout_sec.unwrap_or(DEFAULT_WRITE_STDIN_YIELD_SEC)),
        )?;
    } else if interrupted && status.is_none() {
        // Codex write_stdin sessions are PTYs: a lone Ctrl-C is interpreted by
        // the terminal driver and delivered as SIGINT.  Our contained sessions
        // use pipes, where writing byte 0x03 would be inert, so reproduce the
        // PTY operation by signalling only the owned process group.
        kill_process_group(&session.child, libc::SIGINT)?;
    } else if !chars.is_empty() && status.is_none() {
        if let Some(stdin) = session.child.stdin.as_mut() {
            if let Err(error) = stdin
                .write_all(chars.as_bytes())
                .and_then(|_| stdin.flush())
            {
                if error.kind() != io::ErrorKind::BrokenPipe {
                    return Err(error.into());
                }
            }
        }
    }
    // write_stdin is a bounded observation of an owned interactive session.
    // Source terminal state is only the comparison target: timing drift must
    // not kill the live replay session or abort the remaining trajectory.
    let yielded = session_yield_sec(&action.arguments);
    let observation_sec = if yielded > 0.0 {
        yielded
    } else {
        DEFAULT_WRITE_STDIN_YIELD_SEC
    };
    let remaining_timeout =
        timeout_sec.map(|timeout| (timeout - input_started.elapsed().as_secs_f64()).max(0.0));
    let hard_timeout_wins = remaining_timeout.is_some_and(|timeout| timeout <= observation_sec);
    status = if input_timed_out {
        None
    } else {
        wait_without_kill(
            &mut session.child,
            Some(Duration::from_secs_f64(
                remaining_timeout.map_or(observation_sec, |timeout| timeout.min(observation_sec)),
            )),
            Some(&mut session.tracker),
        )?
    };
    if input_timed_out || (status.is_none() && hard_timeout_wins) {
        terminate_session(session)?;
        return Ok(Map::from_iter([
            ("status".to_owned(), json!("timed_out")),
            ("returncode".to_owned(), json!(124)),
            ("process_root_pid".to_owned(), json!(session.child.id())),
            ("session_id".to_owned(), json!(session_id)),
            (
                "session_output_preview".to_owned(),
                session_output_preview(session)?,
            ),
        ]));
    }
    let returncode = status.map(|status| {
        if interrupted && status.signal() == Some(libc::SIGINT) {
            i64::from(128 + libc::SIGINT)
        } else {
            process_returncode(status)
        }
    });
    let actual_status = process_status(returncode);
    Ok(Map::from_iter([
        ("status".to_owned(), json!(actual_status)),
        ("returncode".to_owned(), json!(returncode)),
        ("process_root_pid".to_owned(), json!(session.child.id())),
        ("session_id".to_owned(), json!(session_id)),
        (
            "session_output_preview".to_owned(),
            session_output_preview(session)?,
        ),
    ]))
}

fn claude_file_path(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
    access: PathAccess,
) -> Result<PathBuf, ReplayError> {
    let path = action
        .arguments
        .get("file_path")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ReplayError::new(format!(
                "ATIF action {} has no string file_path",
                action.index
            ))
        })?;
    policy.resolve(path, cwd, access)
}

fn claude_write(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
) -> Result<Map<String, Value>, ReplayError> {
    let content = action
        .arguments
        .get("content")
        .and_then(Value::as_str)
        .ok_or_else(|| {
            ReplayError::new(format!(
                "ATIF action {} Write has no string content",
                action.index
            ))
        })?;
    let path = claude_file_path(action, cwd, policy, PathAccess::Write)?;
    fs::write(&path, content)?;
    Ok(Map::from_iter([
        ("status".to_owned(), json!("ok")),
        ("returncode".to_owned(), json!(0)),
        ("path".to_owned(), json!(path.to_string_lossy())),
        ("bytes_written".to_owned(), json!(content.len() as u64)),
    ]))
}

fn claude_read(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
) -> Result<Map<String, Value>, ReplayError> {
    let path = claude_file_path(action, cwd, policy, PathAccess::Read)?;
    claude_read_path(action, &path)
}

fn claude_read_path(action: &AtifAction, path: &Path) -> Result<Map<String, Value>, ReplayError> {
    let positive = |key, default| match action.arguments.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .filter(|value| *value > 0)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                ReplayError::new(format!(
                    "ATIF action {} Read offset and limit must be positive integers",
                    action.index
                ))
            }),
    };
    let offset = positive("offset", 1)?;
    let limit = positive("limit", 2000)?;
    if !path.is_file() {
        return Err(ReplayError::new(format!(
            "cannot read missing file: {}",
            path.display()
        )));
    }
    let source = fs::read(&path)?;
    let mut result = Map::from_iter([
        ("status".to_owned(), json!("ok")),
        ("returncode".to_owned(), json!(0)),
        ("path".to_owned(), json!(path.to_string_lossy())),
        ("offset".to_owned(), json!(offset as u64)),
        ("limit".to_owned(), json!(limit as u64)),
        ("bytes_read".to_owned(), json!(source.len() as u64)),
        (
            "content_sha256".to_owned(),
            json!(format!("{:x}", Sha256::digest(&source))),
        ),
    ]);
    match String::from_utf8(source) {
        Ok(source) => {
            let lines: Vec<_> = source.split_inclusive('\n').collect();
            let start = offset.saturating_sub(1).min(lines.len());
            let content = lines[start..start.saturating_add(limit).min(lines.len())].concat();
            result.insert(
                "lines_read".to_owned(),
                json!(content.lines().count() as u64),
            );
            result.insert("bytes_read".to_owned(), json!(content.len() as u64));
            result.insert(
                "content_sha256".to_owned(),
                json!(format!("{:x}", Sha256::digest(content.as_bytes()))),
            );
        }
        Err(_) => {
            // Claude's Read tool accepts PDFs and images; replaying its transport
            // success must not require that the payload is UTF-8 source text.
            result.insert("binary".to_owned(), json!(true));
            result.insert("lines_read".to_owned(), Value::Null);
        }
    }
    Ok(result)
}

fn grep_files(root: &Path, filter: Option<&str>) -> Result<Vec<PathBuf>, ReplayError> {
    if root.is_file() {
        return Ok(vec![root.to_path_buf()]);
    }
    if !root.is_dir() {
        return Err(ReplayError::new(format!(
            "Grep path does not exist: {}",
            root.display()
        )));
    }
    let mut walker = WalkBuilder::new(root);
    walker.add_custom_ignore_filename(".rgignore");
    if let Some(filter) = filter {
        let mut overrides = OverrideBuilder::new(root);
        overrides
            .add(filter)
            .map_err(|error| ReplayError::new(error.to_string()))?;
        walker.overrides(
            overrides
                .build()
                .map_err(|error| ReplayError::new(error.to_string()))?,
        );
    }
    let mut files = Vec::new();
    for entry in walker.build() {
        let entry = entry.map_err(|error| ReplayError::new(error.to_string()))?;
        if entry.file_type().is_some_and(|kind| kind.is_file()) {
            files.push(entry.into_path());
        }
    }
    files.sort();
    Ok(files)
}

fn grep_type_matches(path: &Path, kind: &str) -> Result<bool, ReplayError> {
    let extensions: &[&str] = match kind {
        "rust" => &["rs"],
        "go" => &["go"],
        "py" | "python" => &["py", "pyi"],
        "js" => &["js", "jsx", "mjs", "cjs"],
        "ts" => &["ts", "tsx", "mts", "cts"],
        "java" => &["java"],
        "c" => &["c", "h"],
        "cpp" => &["cc", "cpp", "cxx", "hh", "hpp", "hxx"],
        "json" => &["json"],
        "markdown" | "md" => &["md", "markdown"],
        "yaml" => &["yaml", "yml"],
        other => {
            return Err(ReplayError::new(format!(
                "Claude Code Grep has unsupported type {other:?}"
            )))
        }
    };
    Ok(path
        .extension()
        .and_then(OsStr::to_str)
        .is_some_and(|extension| extensions.contains(&extension)))
}

fn claude_grep_with_policy(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
    timeout_sec: Option<f64>,
) -> Result<Map<String, Value>, ReplayError> {
    let string = |key| {
        action
            .arguments
            .get(key)
            .and_then(Value::as_str)
            .filter(|value| !value.contains('\0'))
    };
    let pattern = string("pattern").ok_or_else(|| {
        ReplayError::new(format!(
            "ATIF action {} Grep has no string pattern",
            action.index
        ))
    })?;
    let path = match string("path") {
        Some(path) => policy.resolve(path, cwd, PathAccess::Read)?,
        None => resolve_allow_missing(cwd)?,
    };
    let mode = action
        .arguments
        .get("output_mode")
        .and_then(Value::as_str)
        .unwrap_or("files_with_matches");
    match mode {
        "files_with_matches" | "content" | "count" => {}
        _ => {
            return Err(ReplayError::new(format!(
                "ATIF action {} Grep has invalid output_mode",
                action.index
            )))
        }
    };
    let integer = |key, default| match action.arguments.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_u64()
            .and_then(|value| usize::try_from(value).ok())
            .ok_or_else(|| {
                ReplayError::new(format!(
                    "ATIF action {} Grep {key} must be a non-negative integer",
                    action.index
                ))
            }),
    };
    let offset = integer("offset", 0)?;
    let head_limit = integer("head_limit", usize::MAX)?;
    let case_insensitive = match action.arguments.get("-i") {
        None => false,
        Some(Value::Bool(value)) => *value,
        Some(_) => return Err(ReplayError::new("Claude Code Grep -i must be a boolean")),
    };
    let multiline = action.arguments.get("multiline") == Some(&Value::Bool(true));
    let expression = RegexBuilder::new(pattern)
        .multi_line(multiline)
        .case_insensitive(case_insensitive)
        .build()
        .map_err(|error| ReplayError::new(format!("invalid Grep pattern: {error}")))?;
    let started = Instant::now();
    let mut rendered_matches = Vec::new();
    let mut total_matches = 0_u64;
    for file in grep_files(&path, string("glob"))? {
        if timeout_sec.is_some_and(|timeout| started.elapsed().as_secs_f64() >= timeout) {
            return Ok(Map::from_iter([
                ("status".to_owned(), json!("timed_out")),
                ("returncode".to_owned(), json!(124)),
                ("path".to_owned(), json!(path.to_string_lossy())),
                ("output_mode".to_owned(), json!(mode)),
                ("matches".to_owned(), json!(0)),
                ("total_matches".to_owned(), Value::Null),
            ]));
        }
        if let Some(kind) = string("type") {
            if !grep_type_matches(&file, kind)? {
                continue;
            }
        }
        let Ok(content) = fs::read_to_string(&file) else {
            continue;
        };
        let display = file.to_string_lossy();
        let file_matches = if multiline {
            expression
                .find_iter(&content)
                .map(|found| {
                    let line = content[..found.start()]
                        .bytes()
                        .filter(|byte| *byte == b'\n')
                        .count()
                        + 1;
                    format!("{display}:{line}:{}", found.as_str().replace('\n', "\\n"))
                })
                .collect::<Vec<_>>()
        } else {
            content
                .lines()
                .enumerate()
                .filter(|(_, line)| expression.is_match(line))
                .map(|(line, content)| format!("{display}:{}:{content}", line + 1))
                .collect::<Vec<_>>()
        };
        if file_matches.is_empty() {
            continue;
        }
        total_matches += file_matches.len() as u64;
        match mode {
            "files_with_matches" => rendered_matches.push(display.into_owned()),
            "count" => rendered_matches.push(format!("{display}:{}", file_matches.len())),
            "content" => rendered_matches.extend(file_matches),
            _ => unreachable!("Grep output mode was validated"),
        }
    }
    let returncode = if rendered_matches.is_empty() { 1 } else { 0 };
    let matches = rendered_matches
        .into_iter()
        .skip(offset)
        .take(head_limit)
        .collect::<Vec<_>>();
    let rendered = matches.join("\n");
    Ok(Map::from_iter([
        ("status".to_owned(), json!("ok")),
        ("returncode".to_owned(), json!(returncode)),
        ("path".to_owned(), json!(path.to_string_lossy())),
        ("output_mode".to_owned(), json!(mode)),
        (
            "matches_sha256".to_owned(),
            json!(format!("{:x}", Sha256::digest(rendered.as_bytes()))),
        ),
        ("matches".to_owned(), json!(matches.len() as u64)),
        (
            "total_matches".to_owned(),
            if mode == "count" {
                json!(total_matches)
            } else {
                Value::Null
            },
        ),
        ("stderr_preview".to_owned(), json!("")),
    ]))
}

fn claude_grep_input_validation_parameter(action: &AtifAction) -> Option<&str> {
    const RECOGNIZED: &[&str] = &[
        "pattern",
        "path",
        "glob",
        "output_mode",
        "offset",
        "head_limit",
        "multiline",
        "type",
        "-i",
    ];
    let parameter = action
        .observation_text
        .strip_prefix("<tool_use_error>InputValidationError: Grep failed due to the following issue:\nAn unexpected parameter `")?
        .strip_suffix("` was provided</tool_use_error>\n\n[error] tool reported failure")?;
    let unsupported = action
        .arguments
        .keys()
        .filter(|key| !RECOGNIZED.contains(&key.as_str()))
        .collect::<Vec<_>>();
    (unsupported.len() == 1 && unsupported[0] == parameter).then_some(parameter)
}

fn claude_file_action(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
    sessions: &HashMap<String, RunningSession>,
) -> Result<Map<String, Value>, ReplayError> {
    match action.tool.as_str() {
        "claude_write" => claude_write(action, cwd, policy),
        "claude_edit" => claude_edit(action, cwd, policy),
        "claude_read" => {
            let task_id = action
                .arguments
                .get("file_path")
                .and_then(Value::as_str)
                .and_then(claude_task_output_path);
            if let Some(task_id) = task_id {
                let session = sessions.get(&task_id).ok_or_else(|| {
                    ReplayError::new(format!(
                        "Claude Code Read references unknown background task {task_id:?}"
                    ))
                })?;
                claude_read_path(action, &session.log_path)
            } else {
                claude_read(action, cwd, policy)
            }
        }
        _ => unreachable!("caller checks Claude file action"),
    }
}

fn failed_file_operation(error: ReplayError) -> Map<String, Value> {
    Map::from_iter([
        ("status".to_owned(), json!("failed")),
        ("returncode".to_owned(), json!(1)),
        ("stderr_preview".to_owned(), json!(error.to_string())),
    ])
}

fn claude_glob_patterns(pattern: &str) -> Result<Vec<String>, ReplayError> {
    let Some(open) = pattern.find('{') else {
        if pattern.contains('}') {
            return Err(ReplayError::new("Claude Code Glob has an unmatched brace"));
        }
        return Ok(vec![pattern.to_owned()]);
    };
    let close = pattern[open + 1..]
        .find('}')
        .map(|index| open + index + 1)
        .ok_or_else(|| ReplayError::new("Claude Code Glob has an unclosed brace"))?;
    let choices = &pattern[open + 1..close];
    if choices.is_empty() {
        return Err(ReplayError::new("Claude Code Glob has an empty brace"));
    }
    let mut output = Vec::new();
    for choice in choices.split(',') {
        output.extend(claude_glob_patterns(&format!(
            "{}{}{}",
            &pattern[..open],
            choice,
            &pattern[close + 1..]
        ))?);
    }
    Ok(output)
}

fn claude_glob(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
) -> Result<Map<String, Value>, ReplayError> {
    let string = |key| {
        action
            .arguments
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ReplayError::new(format!(
                    "ATIF action {} Glob has no string {key}",
                    action.index
                ))
            })
    };
    let pattern = string("pattern")?;
    if pattern.is_empty()
        || pattern.contains('\0')
        || Path::new(pattern).components().any(|component| {
            !matches!(
                component,
                std::path::Component::Normal(_) | std::path::Component::CurDir
            )
        })
    {
        return Err(ReplayError::new(
            "Claude Code Glob pattern is empty or contains a null byte",
        ));
    }
    let root = match action.arguments.get("path") {
        None => cwd.to_path_buf(),
        Some(Value::String(path)) if !path.contains('\0') => {
            policy.resolve(path, cwd, PathAccess::Read)?
        }
        _ => {
            return Err(ReplayError::new(
                "Claude Code Glob path is not a string or contains a null byte",
            ))
        }
    };
    if !root.is_dir() {
        return Err(ReplayError::new(format!(
            "Glob path is not a directory: {}",
            root.display()
        )));
    }
    let mut matches = BTreeSet::new();
    for pattern in claude_glob_patterns(pattern)? {
        let full_pattern = root.join(pattern).to_string_lossy().into_owned();
        for path in
            glob::glob(&full_pattern).map_err(|error| ReplayError::new(error.to_string()))?
        {
            let path = path.map_err(|error| ReplayError::new(error.to_string()))?;
            if path.is_file() {
                matches.insert(policy.resolve(&path.to_string_lossy(), cwd, PathAccess::Read)?);
            }
        }
    }
    let mut matches: Vec<_> = matches
        .into_iter()
        .map(|path| {
            let modified = fs::metadata(&path).and_then(|meta| meta.modified()).ok();
            (path, modified)
        })
        .collect();
    matches.sort_by(|(left, left_time), (right, right_time)| {
        right_time.cmp(left_time).then_with(|| left.cmp(right))
    });
    let truncated = matches.len() > 100;
    matches.truncate(100);
    Ok(Map::from_iter([
        ("status".to_owned(), json!("ok")),
        ("returncode".to_owned(), json!(0)),
        ("path".to_owned(), json!(root.to_string_lossy())),
        ("pattern".to_owned(), json!(pattern)),
        (
            "matches".to_owned(),
            Value::Array(
                matches
                    .into_iter()
                    .map(|(path, _)| json!(path.to_string_lossy()))
                    .collect(),
            ),
        ),
        ("truncated".to_owned(), json!(truncated)),
    ]))
}

fn claude_edit(
    action: &AtifAction,
    cwd: &Path,
    policy: &ReplayPathPolicy,
) -> Result<Map<String, Value>, ReplayError> {
    let string = |key| {
        action
            .arguments
            .get(key)
            .and_then(Value::as_str)
            .ok_or_else(|| {
                ReplayError::new(format!(
                    "ATIF action {} Edit has no string {key}",
                    action.index
                ))
            })
    };
    let old = string("old_string")?;
    let new = string("new_string")?;
    if old.is_empty() {
        return Err(ReplayError::new(format!(
            "ATIF action {} Edit old_string is empty",
            action.index
        )));
    }
    let path = claude_file_path(action, cwd, policy, PathAccess::ReadWrite)?;
    let source = fs::read_to_string(&path)?;
    let count = source.match_indices(old).count();
    let replace_all = action
        .arguments
        .get("replace_all")
        .and_then(Value::as_bool)
        .ok_or_else(|| {
            ReplayError::new(format!(
                "ATIF action {} Edit has no boolean replace_all",
                action.index
            ))
        })?;
    if count == 0 || (!replace_all && count != 1) {
        return Err(ReplayError::new(format!(
            "ATIF action {} Edit old_string matched {count} times in {}",
            action.index,
            path.display()
        )));
    }
    fs::write(
        &path,
        if replace_all {
            source.replace(old, new)
        } else {
            source.replacen(old, new, 1)
        },
    )?;
    Ok(Map::from_iter([
        ("status".to_owned(), json!("ok")),
        ("returncode".to_owned(), json!(0)),
        ("path".to_owned(), json!(path.to_string_lossy())),
    ]))
}

fn rounded(value: f64) -> Value {
    let value = (value * 1_000_000_000.0).round() / 1_000_000_000.0;
    Value::Number(Number::from_f64(value).expect("finite replay duration"))
}

fn temp_artifacts_dir() -> PathBuf {
    env::temp_dir().join(format!(
        "atif-replay-sessions-{}-{}",
        std::process::id(),
        monotonic_ns()
    ))
}

fn result_status(result: &Map<String, Value>) -> Option<&str> {
    result.get("status").and_then(Value::as_str)
}

/// Replay all supported source actions and return the stable v1 JSON report.
pub fn replay_trajectory(
    trajectory: &AtifTrajectory,
    options: &ReplayOptions,
) -> Result<Value, ReplayError> {
    validate_supported_actions(trajectory)?;
    if !options.cwd.is_dir() {
        return Err(ReplayError::new(format!(
            "replay working directory does not exist: {}",
            options.cwd.display()
        )));
    }
    if let Some(timeout) = options.timeout_sec {
        if !timeout.is_finite() || timeout <= 0.0 {
            return Err(ReplayError::new("action timeout must be positive"));
        }
    }
    let path_policy =
        ReplayPathPolicy::for_trajectory(&options.cwd, &options.patch_roots, trajectory)?;
    let artifacts_dir = options
        .artifacts_dir
        .clone()
        .unwrap_or_else(temp_artifacts_dir);
    fs::create_dir_all(&artifacts_dir)?;

    let executable: Vec<&AtifAction> = trajectory
        .actions
        .iter()
        .filter(|action| is_source_executed_supported_action(action))
        .collect();
    if executable.is_empty() {
        return Err(ReplayError::new(
            "ATIF trajectory has no source-executed supported actions to replay",
        ));
    }
    let first_timestamp = executable[0].timestamp_sec;
    let source_end = executable
        .last()
        .expect("nonempty executable list")
        .timestamp_sec;
    let replay_started = Instant::now();
    let mut previous_source_finish: Option<f64> = None;
    let mut previous_replay_finish: Option<Instant> = None;
    let mut results = Vec::<Map<String, Value>>::new();
    let mut sessions = HashMap::<String, RunningSession>::new();
    let mut session_order = Vec::<String>::new();
    let mut terminated_session_ids = Vec::<String>::new();
    let mut aborted_after_timeout_action_index: Option<usize> = None;

    let replay_result = (|| -> Result<(), ReplayError> {
        for action in &trajectory.actions {
            let expected = source_outcome(action);
            let mut result = Map::new();
            insert(&mut result, "action_index", action.index as u64);
            insert(&mut result, "step_id", action.step_id.clone());
            insert(
                &mut result,
                "source_trajectory_id",
                action.source_trajectory_id.clone(),
            );
            if let Some(source_item_id) = action.source_item_id.as_ref() {
                insert(&mut result, "source_item_id", source_item_id.clone());
            }
            insert(&mut result, "tool", action.tool.clone());
            insert(&mut result, "source_tool", action.source_tool.clone());
            insert(&mut result, "code_mode", action.code_mode);
            if let Some(parallel_group_id) = action.parallel_group_id.as_ref() {
                insert(&mut result, "parallel_group_id", parallel_group_id.clone());
                insert(
                    &mut result,
                    "parallel_execution",
                    "sequentialized_for_per_command_cpu",
                );
            }
            insert(&mut result, "source_status", expected.status.clone());
            insert(
                &mut result,
                "source_returncode",
                expected.returncode.map(Value::from).unwrap_or(Value::Null),
            );
            insert(
                &mut result,
                "source_reported_wall_time_sec",
                recorded_tool_wall_sec(action)
                    .map(rounded)
                    .unwrap_or(Value::Null),
            );
            if is_metadata_action(action) {
                insert(&mut result, "status", "ignored_metadata");
                insert(
                    &mut result,
                    "status_matches_source",
                    status_matches(&expected.status, "ignored_metadata")
                        .map(Value::Bool)
                        .unwrap_or(Value::Null),
                );
                insert(&mut result, "outcome_matches_source", true);
                results.push(result);
                continue;
            }
            if expected.status == "skipped_not_executed" {
                insert(&mut result, "status", "skipped_not_executed");
                insert(&mut result, "status_matches_source", true);
                insert(&mut result, "outcome_matches_source", true);
                results.push(result);
                continue;
            }
            let source_offset_sec = action.timestamp_sec - first_timestamp;
            insert(
                &mut result,
                "source_start_offset_sec",
                rounded(source_offset_sec),
            );
            match options.pacing {
                Pacing::Recorded => {
                    let sleep = source_offset_sec - replay_started.elapsed().as_secs_f64();
                    if sleep > 0.0 {
                        thread::sleep(Duration::from_secs_f64(sleep));
                    }
                }
                Pacing::LlmGaps => {
                    if let (Some(source_finish), Some(replay_finish)) =
                        (previous_source_finish, previous_replay_finish)
                    {
                        let gap = (action.timestamp_sec - source_finish).max(0.0);
                        insert(&mut result, "source_llm_gap_sec", rounded(gap));
                        let sleep = gap - replay_finish.elapsed().as_secs_f64();
                        if sleep > 0.0 {
                            thread::sleep(Duration::from_secs_f64(sleep));
                        }
                    }
                }
                Pacing::Unpaced => {}
            }
            let action_started = Instant::now();
            let realtime_before_ns = realtime_ns();
            let action_started_ns = monotonic_ns();
            let realtime_after_ns = realtime_ns();
            let (action_started_realtime_ns, action_started_realtime_uncertainty_ns) =
                realtime_midpoint_and_uncertainty(realtime_before_ns, realtime_after_ns);
            let mut outcome = if source_argument_parse_failed(action) {
                Map::from_iter([
                    ("command".to_owned(), Value::Null),
                    ("status".to_owned(), Value::String("failed".to_owned())),
                    ("returncode".to_owned(), Value::Null),
                    ("source_argument_parse_failed".to_owned(), Value::Bool(true)),
                    ("command_executed".to_owned(), Value::Bool(false)),
                    (
                        "source_error_preview".to_owned(),
                        Value::String(preview(&action.observation_text)),
                    ),
                ])
            } else if action.tool == "exec_command" {
                let outcome = execute_command(
                    action,
                    &options.cwd,
                    options.timeout_sec,
                    &expected,
                    &mut sessions,
                    &artifacts_dir,
                    options.action_output_dir.as_deref(),
                )?;
                if expected.status == "running" {
                    if let Some(session_id) = expected.session_id.as_ref() {
                        if sessions.contains_key(session_id) && !session_order.contains(session_id)
                        {
                            session_order.push(session_id.clone());
                        }
                    }
                }
                outcome
            } else if action.tool == "write_stdin" {
                write_stdin(action, options.timeout_sec, &expected, &mut sessions)?
            } else if action.tool == "claude_task_stop" {
                let task_id = claude_task_id(action)?;
                let mut session = sessions.remove(task_id).ok_or_else(|| {
                    ReplayError::new(format!(
                        "Claude Code TaskStop references unknown background task {task_id:?}"
                    ))
                })?;
                terminate_session(&mut session)?;
                Map::from_iter([
                    ("status".to_owned(), Value::String("ok".to_owned())),
                    ("returncode".to_owned(), Value::from(0)),
                    ("session_id".to_owned(), Value::String(task_id.to_owned())),
                ])
            } else if action.tool == "claude_task_output" {
                claude_task_output(action, options.timeout_sec, &mut sessions)?
            } else if action.tool == "claude_grep" {
                let validation_parameter = (expected.status == "failed")
                    .then(|| claude_grep_input_validation_parameter(action))
                    .flatten();
                if let Some(parameter) = validation_parameter {
                    Map::from_iter([
                        ("status".to_owned(), Value::String("failed".to_owned())),
                        ("returncode".to_owned(), Value::Null),
                        ("command_executed".to_owned(), Value::Bool(false)),
                        (
                            "source_input_validation_reproduced".to_owned(),
                            Value::Bool(true),
                        ),
                        (
                            "unsupported_arguments".to_owned(),
                            Value::Array(vec![Value::String(parameter.to_owned())]),
                        ),
                    ])
                } else {
                    match claude_grep_with_policy(
                        action,
                        &options.cwd,
                        &path_policy,
                        options.timeout_sec,
                    ) {
                        Ok(outcome) => outcome,
                        Err(error) => failed_file_operation(error),
                    }
                }
            } else if action.tool == "apply_patch" {
                let patch = action
                    .arguments
                    .get("input")
                    .or_else(|| action.arguments.get("patch"))
                    .and_then(Value::as_str)
                    .ok_or_else(|| {
                        ReplayError::new(format!(
                            "ATIF action {} apply_patch has no string input",
                            action.index
                        ))
                    })?;
                match apply_patch_payload_with_policy(patch, &options.cwd, &path_policy) {
                    Ok(()) => Map::from_iter([
                        ("status".to_owned(), Value::String("ok".to_owned())),
                        ("returncode".to_owned(), Value::from(0)),
                    ]),
                    Err(error) => failed_file_operation(error),
                }
            } else if action.tool == "claude_glob" {
                match claude_glob(action, &options.cwd, &path_policy) {
                    Ok(outcome) => outcome,
                    Err(error) => failed_file_operation(error),
                }
            } else if matches!(
                action.tool.as_str(),
                "claude_write" | "claude_edit" | "claude_read"
            ) {
                match claude_file_action(action, &options.cwd, &path_policy, &sessions) {
                    Ok(outcome) => outcome,
                    Err(error) => failed_file_operation(error),
                }
            } else {
                return Err(ReplayError::new(format!(
                    "unexpected supported action tool: {:?}",
                    action.tool
                )));
            };
            result.append(&mut outcome);
            insert(
                &mut result,
                "wall_time_sec",
                rounded(action_started.elapsed().as_secs_f64()),
            );
            insert(&mut result, "replay_start_monotonic_ns", action_started_ns);
            insert(
                &mut result,
                "replay_start_realtime_ns",
                action_started_realtime_ns,
            );
            insert(
                &mut result,
                "replay_start_realtime_uncertainty_ns",
                action_started_realtime_uncertainty_ns,
            );
            insert(&mut result, "replay_end_monotonic_ns", monotonic_ns());
            previous_source_finish =
                Some(action.timestamp_sec + recorded_tool_wall_sec(action).unwrap_or(0.0));
            previous_replay_finish = Some(Instant::now());
            let actual_status = result_status(&result)
                .expect("every supported action outcome has a status")
                .to_owned();
            if expected.status == "running" && matches!(actual_status.as_str(), "ok" | "failed") {
                insert(
                    &mut result,
                    "completion_timing_drift",
                    "completed_before_recorded_poll",
                );
            }
            let actual_returncode = result.get("returncode").and_then(Value::as_i64);
            insert(
                &mut result,
                "status_matches_source",
                status_matches(&expected.status, &actual_status)
                    .map(Value::Bool)
                    .unwrap_or(Value::Null),
            );
            insert(
                &mut result,
                "outcome_matches_source",
                outcome_matches(
                    &expected.status,
                    expected.returncode,
                    &actual_status,
                    actual_returncode,
                )
                .map(Value::Bool)
                .unwrap_or(Value::Null),
            );
            if result.get("launch_failed") == Some(&Value::Bool(true)) {
                // No command completion occurred, so a generic failed source
                // status is not evidence of an equivalent execution outcome.
                insert(&mut result, "status_matches_source", Value::Null);
                insert(&mut result, "outcome_matches_source", Value::Null);
                insert(&mut result, "source_outcome_comparison_unavailable", true);
            }
            let timed_out = actual_status == "timed_out";
            results.push(result);
            if timed_out {
                aborted_after_timeout_action_index = Some(action.index);
                break;
            }
        }
        Ok(())
    })();

    let mut cleanup_error = None;
    for session_id in session_order {
        if let Some(mut session) = sessions.remove(&session_id) {
            match session.poll() {
                Ok(None) => terminated_session_ids.push(session_id.clone()),
                Ok(Some(_)) => {}
                Err(error) => {
                    cleanup_error.get_or_insert_with(|| ReplayError::from(error));
                }
            }
            if let Err(error) = terminate_session(&mut session) {
                cleanup_error.get_or_insert_with(|| ReplayError::from(error));
            }
        }
    }
    for (session_id, mut session) in sessions {
        if session.poll()?.is_none() {
            terminated_session_ids.push(session_id);
        }
        terminate_session(&mut session)?;
    }
    replay_result?;
    if let Some(error) = cleanup_error {
        return Err(error);
    }
    if options.action_output_dir.is_none() {
        for result in &mut results {
            for field in [
                "stdout_preview",
                "stderr_preview",
                "session_output_preview",
                "source_error_preview",
            ] {
                result.remove(field);
            }
        }
    }

    let replayed = |result: &&Map<String, Value>| {
        let faithfully_reproduced_failure = result.get("source_input_validation_reproduced")
            == Some(&Value::Bool(true))
            && match result.get("tool").and_then(Value::as_str) {
                Some("claude_grep") => result
                    .get("unsupported_arguments")
                    .and_then(Value::as_array)
                    .is_some_and(|arguments| {
                        arguments.len() == 1
                            && arguments[0]
                                .as_str()
                                .is_some_and(|argument| !argument.is_empty())
                    }),
                _ => false,
            }
            && result.get("command_executed") == Some(&Value::Bool(false))
            && result.get("source_status").and_then(Value::as_str) == Some("failed")
            && result_status(result) == Some("failed")
            && result.get("status_matches_source") == Some(&Value::Bool(true))
            && result.get("outcome_matches_source") == Some(&Value::Bool(true))
            && result.get("source_argument_parse_failed") != Some(&Value::Bool(true))
            && result.get("launch_failed") != Some(&Value::Bool(true));
        result
            .get("tool")
            .and_then(Value::as_str)
            .is_some_and(is_supported_tool)
            && (result.get("command_executed") != Some(&Value::Bool(false))
                || faithfully_reproduced_failure)
            && matches!(
                result_status(result),
                Some("ok" | "failed" | "timed_out" | "running")
            )
    };
    let replayed_results = results.iter().filter(replayed).collect::<Vec<_>>();
    let reproduced_without_execution = replayed_results
        .iter()
        .filter(|result| {
            result.get("source_input_validation_reproduced") == Some(&Value::Bool(true))
        })
        .count();
    let status_mismatches: Vec<Value> = replayed_results
        .iter()
        .filter(|result| result.get("status_matches_source") == Some(&Value::Bool(false)))
        .filter_map(|result| result.get("action_index").cloned())
        .collect();
    let status_unavailable: Vec<Value> = replayed_results
        .iter()
        .filter(|result| result.get("status_matches_source") == Some(&Value::Null))
        .filter_map(|result| result.get("action_index").cloned())
        .collect();
    let outcome_mismatches: Vec<Value> = replayed_results
        .iter()
        .filter(|result| result.get("outcome_matches_source") == Some(&Value::Bool(false)))
        .filter_map(|result| result.get("action_index").cloned())
        .collect();
    let outcome_unavailable: Vec<Value> = results
        .iter()
        .filter(|result| result.get("outcome_matches_source") == Some(&Value::Null))
        .filter(|result| {
            replayed(result)
                || result.get("source_outcome_comparison_unavailable") == Some(&Value::Bool(true))
        })
        .filter_map(|result| result.get("action_index").cloned())
        .collect();
    let actions_executed = results
        .iter()
        .filter(|result| result.get("command_executed") != Some(&Value::Bool(false)))
        .filter(|result| {
            matches!(
                result_status(result),
                Some("ok" | "failed" | "timed_out" | "running")
            )
        })
        .count();
    let actions_skipped_after_timeout = aborted_after_timeout_action_index
        .map(|aborted| {
            trajectory
                .actions
                .iter()
                .filter(|action| action.index > aborted)
                .filter(|action| is_supported_tool(&action.tool))
                .filter(|action| source_outcome(action).status != "skipped_not_executed")
                .count()
        })
        .unwrap_or(0);
    let count_status = |status: &str| {
        results
            .iter()
            .filter(|result| result_status(result) == Some(status))
            .count()
    };
    let source_path = resolve_allow_missing(&trajectory.source_path)?;
    let outcome_mismatch_count = outcome_mismatches.len();
    let outcome_unavailable_count = outcome_unavailable.len();
    let tool_actions_recorded = executable.len();
    let tool_actions_replayed = replayed_results.len();
    let tool_action_status_equal = replayed_results
        .iter()
        .filter(|result| result.get("status_matches_source") == Some(&Value::Bool(true)))
        .count();
    let tool_action_status_different = status_mismatches.len();
    let tool_action_status_comparable = tool_action_status_equal + tool_action_status_different;
    let tool_action_status_unavailable = status_unavailable.len();
    let tool_action_outcome_equal = replayed_results
        .iter()
        .filter(|result| result.get("outcome_matches_source") == Some(&Value::Bool(true)))
        .count();
    let tool_action_outcome_different = outcome_mismatch_count;
    let tool_action_outcome_comparable = tool_action_outcome_equal + tool_action_outcome_different;
    let tool_action_outcome_unavailable = replayed_results
        .iter()
        .filter(|result| result.get("outcome_matches_source") == Some(&Value::Null))
        .count();
    let mut summary = json!({
        "actions_recorded": trajectory.actions.len(),
        "actions_executed": actions_executed,
        "actions_skipped_after_timeout": actions_skipped_after_timeout,
        "aborted_after_timeout_action_index": aborted_after_timeout_action_index,
        "metadata_actions_ignored": trajectory.actions.iter().filter(|action| is_metadata_action(action)).count(),
        "actions_skipped_not_executed": count_status("skipped_not_executed"),
        "actions_ok": count_status("ok"),
        "actions_failed": count_status("failed"),
        "actions_launch_failed": results.iter().filter(|result| result.get("launch_failed") == Some(&Value::Bool(true))).count(),
        "actions_source_argument_parse_failed": results.iter().filter(|result| result.get("source_argument_parse_failed") == Some(&Value::Bool(true))).count(),
        "actions_timed_out": count_status("timed_out"),
        "tool_actions_recorded": tool_actions_recorded,
        "tool_actions_replayed": tool_actions_replayed,
        "tool_action_status_comparable": tool_action_status_comparable,
        "tool_action_status_equal": tool_action_status_equal,
        "tool_action_status_different": tool_action_status_different,
        "tool_action_status_unavailable": tool_action_status_unavailable,
        "tool_action_outcome_comparable": tool_action_outcome_comparable,
        "tool_action_outcome_equal": tool_action_outcome_equal,
        "tool_action_outcome_different": tool_action_outcome_different,
        "tool_action_outcome_unavailable": tool_action_outcome_unavailable,
        "source_outcome_mismatch_action_indexes": outcome_mismatches,
        "source_outcome_mismatch_count": outcome_mismatch_count,
        "source_outcome_comparison_unavailable_action_indexes": outcome_unavailable,
        "source_outcome_comparison_unavailable_count": outcome_unavailable_count,
        "actions_completed_before_recorded_poll": results.iter().filter(|result| result.get("completion_timing_drift").and_then(Value::as_str) == Some("completed_before_recorded_poll")).count(),
        "sessions_terminated_at_end": terminated_session_ids,
        "source_interval_sec": rounded(source_end - first_timestamp),
        "wall_time_sec": rounded(replay_started.elapsed().as_secs_f64()),
    });
    if reproduced_without_execution > 0 {
        summary["tool_actions_reproduced_without_execution"] =
            Value::from(reproduced_without_execution);
    }
    Ok(json!({
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_trajectory": source_path.to_string_lossy(),
        "source_schema_version": trajectory.schema_version,
        "source_agent_original_format": trajectory.source_agent_original_format,
        "source_session_id": trajectory.session_id,
        "subagent_trajectories_flattened": trajectory.subagent_trajectories_flattened,
        "agent": trajectory.agent,
        "task_name": trajectory.task_name(),
        "source_trial": trajectory.trial_name(),
        "pacing": options.pacing.as_str(),
        "action_timeout_sec": options.timeout_sec,
        "path_policy": {
            "workspace_root": path_policy.workspace_root,
            "readable_roots": path_policy.readable_roots,
            "writable_roots": path_policy.writable_roots,
            "recorded_external_read_paths": path_policy.exact_readable_paths.len(),
            "recorded_external_write_paths": path_policy.exact_writable_paths.len(),
        },
        "clock_reference": clock_reference(),
        "actions": results,
        "limitations": [
            "ATIF replay executes supported recorded tool calls only; it does not invoke the model.",
            "Replay stops after the first action timeout because later state is no longer comparable.",
            "Source-outcome matching compares recorded tool completion states and return codes where available.",
            "Recorded file-tool paths are allowed by a read/write path policy; shell commands are not filesystem-sandboxed by that policy.",
        ],
        "summary": summary,
    }))
}

#[derive(Debug)]
struct CliOptions {
    trajectory: PathBuf,
    output: PathBuf,
    cwd: PathBuf,
    patch_roots: Vec<PathBuf>,
    pacing: Pacing,
    timeout_sec: Option<f64>,
    action_output_dir: Option<PathBuf>,
}

fn print_cli_help(public_program: &str) {
    println!(
        "usage: {public_program} [-h] --output OUTPUT [--cwd CWD]\n\
         \x20      [--patch-root PATCH_ROOT]\n\
         \x20      [--pacing {{recorded,unpaced,llm-gaps}}]\n\
         \x20      [--timeout-sec TIMEOUT_SEC]\n\
         \x20      [--action-output-dir ACTION_OUTPUT_DIR]\n\
         \x20      trajectory"
    );
    println!();
    println!("Replay supported tool actions from an ATIF trajectory.");
    println!();
    println!("positional arguments:");
    println!("  trajectory            ATIF trajectory.json path.");
    println!();
    println!("options:");
    println!("  -h, --help            show this help message and exit");
    println!("  --output OUTPUT       Replay report JSON path.");
    println!("  --cwd CWD             Replay working directory.");
    println!("  --patch-root PATCH_ROOT");
    println!("                        Additional writable path root permitted for recorded apply_patch actions.");
    println!("  --pacing {{recorded,unpaced,llm-gaps}}");
    println!("  --timeout-sec TIMEOUT_SEC");
    println!("                        Per-command/session completion timeout.");
    println!("  --action-output-dir ACTION_OUTPUT_DIR");
    println!("                        Optional directory for full source-equivalent action output artifacts.");
}

fn cli_error(public_program: &str, message: impl fmt::Display) -> i32 {
    eprintln!("usage: {public_program} [-h] --output OUTPUT [options] trajectory");
    eprintln!("{public_program}: error: {message}");
    2
}

fn option_value(
    arguments: &[OsString],
    index: &mut usize,
    option: &str,
    inline: Option<OsString>,
) -> Result<OsString, String> {
    if let Some(value) = inline {
        if value.is_empty() {
            return Err(format!("argument {option}: expected one argument"));
        }
        return Ok(value);
    }
    *index += 1;
    arguments
        .get(*index)
        .cloned()
        .ok_or_else(|| format!("argument {option}: expected one argument"))
}

fn split_long_option(argument: &OsStr) -> (String, Option<OsString>) {
    let argument = argument.to_string_lossy();
    if let Some((name, value)) = argument.split_once('=') {
        (name.to_owned(), Some(OsString::from(value)))
    } else {
        (argument.into_owned(), None)
    }
}

fn parse_cli(arguments: &[OsString]) -> Result<Option<CliOptions>, String> {
    let mut trajectory = None;
    let mut output = None;
    let mut cwd = env::current_dir().map_err(|error| error.to_string())?;
    let mut patch_roots = Vec::new();
    let mut pacing = Pacing::Recorded;
    let mut timeout_sec = None;
    let mut action_output_dir = None;
    let mut positional_only = false;
    let mut index = 0;
    while index < arguments.len() {
        let argument = &arguments[index];
        if !positional_only && matches!(argument.to_str(), Some("-h" | "--help")) {
            return Ok(None);
        }
        if !positional_only && argument == OsStr::new("--") {
            positional_only = true;
            index += 1;
            continue;
        }
        let starts_option = !positional_only
            && argument
                .to_str()
                .map(|argument| argument.starts_with('-'))
                .unwrap_or(false);
        if starts_option {
            let (name, inline) = split_long_option(argument);
            match name.as_str() {
                "--output" => {
                    output = Some(PathBuf::from(option_value(
                        arguments, &mut index, "--output", inline,
                    )?));
                }
                "--cwd" => {
                    cwd = PathBuf::from(option_value(arguments, &mut index, "--cwd", inline)?);
                }
                "--patch-root" => {
                    patch_roots.push(PathBuf::from(option_value(
                        arguments,
                        &mut index,
                        "--patch-root",
                        inline,
                    )?));
                }
                "--pacing" => {
                    let value = option_value(arguments, &mut index, "--pacing", inline)?;
                    let value = value
                        .to_str()
                        .ok_or_else(|| "argument --pacing contains non-Unicode text".to_owned())?;
                    pacing = Pacing::parse(value).map_err(|_| {
                        format!(
                            "argument --pacing: invalid choice: {value:?} (choose from 'recorded', 'unpaced', 'llm-gaps')"
                        )
                    })?;
                }
                "--timeout-sec" => {
                    let value = option_value(arguments, &mut index, "--timeout-sec", inline)?;
                    let displayed = value.to_string_lossy();
                    timeout_sec = Some(displayed.parse::<f64>().map_err(|_| {
                        format!("argument --timeout-sec: invalid float value: {displayed:?}")
                    })?);
                }
                "--action-output-dir" => {
                    action_output_dir = Some(PathBuf::from(option_value(
                        arguments,
                        &mut index,
                        "--action-output-dir",
                        inline,
                    )?));
                }
                _ => {
                    return Err(format!(
                        "unrecognized arguments: {}",
                        argument.to_string_lossy()
                    ))
                }
            }
        } else if trajectory.is_none() {
            trajectory = Some(PathBuf::from(argument.as_os_str()));
        } else {
            return Err(format!(
                "unrecognized arguments: {}",
                argument.to_string_lossy()
            ));
        }
        index += 1;
    }
    let trajectory =
        trajectory.ok_or_else(|| "the following arguments are required: trajectory".to_owned())?;
    let output =
        output.ok_or_else(|| "the following arguments are required: --output".to_owned())?;
    if timeout_sec.map(|value| !value.is_finite() || value <= 0.0) == Some(true) {
        return Err("--timeout-sec must be positive".to_owned());
    }
    Ok(Some(CliOptions {
        trajectory,
        output,
        cwd,
        patch_roots,
        pacing,
        timeout_sec,
        action_output_dir,
    }))
}

fn run_cli(options: CliOptions) -> Result<i32, ReplayError> {
    let mut replay_options = ReplayOptions::new(options.cwd);
    replay_options.pacing = options.pacing;
    replay_options.timeout_sec = options.timeout_sec;
    replay_options.artifacts_dir = Some(
        options
            .output
            .parent()
            .unwrap_or_else(|| Path::new("."))
            .join("session_logs"),
    );
    replay_options.action_output_dir = options.action_output_dir;
    replay_options.patch_roots = options.patch_roots;
    let trajectory = read_trajectory(&options.trajectory)?;
    let report = replay_trajectory(&trajectory, &replay_options)?;
    if let Some(parent) = options.output.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let mut encoded = serde_json::to_string_pretty(&report)?;
    encoded.push('\n');
    fs::write(&options.output, encoded)?;
    let summary = &report["summary"];
    println!("Task: {}", report["task_name"].as_str().unwrap_or(""));
    println!(
        "Source trial: {}",
        report["source_trial"].as_str().unwrap_or("")
    );
    println!(
        "Tool actions: {}/{} replayed; source status: {} equal, {} different, {} unavailable",
        summary["tool_actions_replayed"].as_u64().unwrap_or(0),
        summary["tool_actions_recorded"].as_u64().unwrap_or(0),
        summary["tool_action_status_equal"].as_u64().unwrap_or(0),
        summary["tool_action_status_different"]
            .as_u64()
            .unwrap_or(0),
        summary["tool_action_status_unavailable"]
            .as_u64()
            .unwrap_or(0),
    );
    println!(
        "Wall time: {:.9} sec",
        summary["wall_time_sec"].as_f64().unwrap_or(0.0)
    );
    println!("Report: {}", options.output.display());
    Ok(
        if [
            "source_outcome_mismatch_count",
            "source_outcome_comparison_unavailable_count",
            "actions_timed_out",
            "actions_skipped_after_timeout",
            "actions_launch_failed",
        ]
        .iter()
        .any(|field| summary[*field].as_u64().unwrap_or(0) > 0)
        {
            1
        } else {
            0
        },
    )
}

/// Public leaf entry point used by the crate dispatcher.
pub fn main(arguments: &[OsString], public_program: &str) -> i32 {
    let options = match parse_cli(arguments) {
        Ok(None) => {
            print_cli_help(public_program);
            return 0;
        }
        Ok(Some(options)) => options,
        Err(error) => return cli_error(public_program, error),
    };
    match run_cli(options) {
        Ok(status) => status,
        Err(error) => cli_error(public_program, error),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::write::GzEncoder;
    use flate2::Compression;

    fn write_trajectory(root: &Path, actions: Vec<(&str, Value, Value)>, agent: Value) -> PathBuf {
        let directory = root.join("task-name__trial-id").join("agent");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("trajectory.json");
        let claude_code = agent.get("name").and_then(Value::as_str) == Some(CLAUDE_CODE_AGENT_NAME);
        let steps: Vec<Value> = actions
            .into_iter()
            .enumerate()
            .map(|(index, (tool, arguments, observation))| {
                let call_id = format!("call-{index}");
                json!({
                    "step_id": index + 1,
                    "timestamp": index as f64,
                    "tool_calls": [{"tool_call_id": call_id, "function_name": tool, "arguments": arguments}],
                    "observation": if claude_code {
                        json!({"results": [{"source_call_id": call_id, "content": observation.clone()}]})
                    } else {
                        observation
                    },
                })
            })
            .collect();
        fs::write(
            &path,
            serde_json::to_vec(&json!({
                "schema_version": "ATIF-v1.5",
                "session_id": "session",
                "agent": agent,
                "steps": steps,
            }))
            .unwrap(),
        )
        .unwrap();
        path
    }

    fn default_agent() -> Value {
        json!({"name": "codex"})
    }

    fn replay_options(cwd: &Path) -> ReplayOptions {
        let mut options = ReplayOptions::new(cwd);
        options.pacing = Pacing::Unpaced;
        options.artifacts_dir = Some(cwd.join("session-logs"));
        options
    }

    fn path_policy(cwd: &Path) -> ReplayPathPolicy {
        ReplayPathPolicy::from_roots(cwd, &[]).unwrap()
    }

    fn add_nested_invocations(path: &Path, invocations: Vec<Value>) {
        let mut trajectory: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
        let steps = trajectory["steps"].as_array_mut().unwrap();
        assert_eq!(steps.len(), invocations.len());
        let mut extracted = 0u64;
        let mut bound = 0u64;
        for (step, invocation) in steps.iter_mut().zip(invocations) {
            let values = invocation.as_array().unwrap();
            extracted += values.len() as u64;
            bound += values
                .iter()
                .filter(|value| !value["outcome"].is_null())
                .count() as u64;
            step["extra"]["riptide_source_trajectory_id"] = json!("session");
            step["tool_calls"][0]["extra"]["riptide_nested_invocations"] = invocation;
        }
        trajectory["extra"]["riptide_normalization"] = json!({
            "kind": "codex-native-sessions-flattened",
            "native_session_count": 1,
            "normalized_session_count": 1,
            "native_executable_event_count": extracted,
            "extracted_executable_event_count": extracted,
            "outcome_bound_executable_event_count": bound,
            "outcome_unavailable_executable_event_count": extracted - bound,
            "unaccounted_executable_events": 0
        });
        fs::write(path, serde_json::to_vec(&trajectory).unwrap()).unwrap();
    }

    #[test]
    fn reads_gzip_directly_and_through_plain_path_fallback() {
        let temporary = tempfile::tempdir().unwrap();
        let plain = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": "true"}),
                json!({"content": "Process exited with code 0"}),
            )],
            default_agent(),
        );
        let raw = fs::read(&plain).unwrap();
        fs::remove_file(&plain).unwrap();
        let gzip = PathBuf::from(format!("{}.gz", plain.to_string_lossy()));
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(&raw).unwrap();
        fs::write(&gzip, encoder.finish().unwrap()).unwrap();

        assert_eq!(
            read_trajectory(&gzip).unwrap().actions[0].tool,
            "exec_command"
        );
        let fallback = read_trajectory(&plain).unwrap();
        assert_eq!(fallback.source_path, plain);
        assert_eq!(fallback.actions[0].tool, "exec_command");
    }

    #[test]
    fn flattens_recursive_subagents_by_timestamp_and_rejects_dangling_refs() {
        let temporary = tempfile::tempdir().unwrap();
        let path = temporary.path().join("trajectory.json");
        let node = |id: &str, timestamp: f64, children: Value, reference: Value| {
            json!({
                "schema_version": "ATIF-v1.7",
                "trajectory_id": id,
                "agent": claude_code_agent(),
                "steps": [{
                    "step_id": 1,
                    "timestamp": timestamp,
                    "tool_calls": [{"tool_call_id": id, "function_name": "Bash", "arguments": {"command": "true"}}],
                    "observation": {"results": [{
                        "source_call_id": id,
                        "content": "done",
                        "subagent_trajectory_ref": reference,
                        "extra": {"tool_result_is_error": false}
                    }]}
                }],
                "subagent_trajectories": children
            })
        };
        let grandchild = node("grandchild", 1.0, json!([]), json!([]));
        let child = node(
            "child",
            2.0,
            json!([grandchild]),
            json!([{"trajectory_id": "grandchild"}]),
        );
        let root = node(
            "root-id",
            3.0,
            json!([child]),
            json!([{"trajectory_id": "child"}]),
        );
        fs::write(&path, serde_json::to_vec(&root).unwrap()).unwrap();
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.subagent_trajectories_flattened, 2);
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.source_trajectory_id.as_str())
                .collect::<Vec<_>>(),
            ["grandchild", "child", "root-id"]
        );

        let mut flattened = root.clone();
        let mut steps = Vec::new();
        for (id, node) in [
            ("root-id", &root),
            ("child", &root["subagent_trajectories"][0]),
            (
                "grandchild",
                &root["subagent_trajectories"][0]["subagent_trajectories"][0],
            ),
        ] {
            let mut step = node["steps"][0].clone();
            step["extra"]["riptide_source_trajectory_id"] = json!(id);
            step["tool_calls"][0]["function_name"] = json!("exec");
            step["tool_calls"][0]["arguments"] =
                json!({"input": "await tools.exec_command({cmd: 'true'});"});
            step["tool_calls"][0]["extra"]["riptide_nested_invocations"] = json!([{
                "index": 0,
                "tool": "exec_command",
                "arguments": {"cmd": "true"},
                "source_item_id": format!("item-{id}"),
                "outcome": null
            }]);
            steps.push(step);
        }
        flattened["agent"] = codex_exec_agent();
        flattened["steps"] = Value::Array(steps);
        flattened
            .as_object_mut()
            .unwrap()
            .remove("subagent_trajectories");
        flattened["extra"]["riptide_normalization"] = json!({
            "kind": "codex-native-sessions-flattened",
            "native_session_count": 3,
            "normalized_session_count": 3,
            "native_executable_event_count": 3,
            "extracted_executable_event_count": 3,
            "outcome_bound_executable_event_count": 0,
            "outcome_unavailable_executable_event_count": 3,
            "unaccounted_executable_events": 0
        });
        fs::write(&path, serde_json::to_vec(&flattened).unwrap()).unwrap();
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.subagent_trajectories_flattened, 2);
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.source_trajectory_id.as_str())
                .collect::<Vec<_>>(),
            ["grandchild", "child", "root-id"]
        );

        let mut dangling = root.clone();
        dangling["steps"][0]["observation"]["results"][0]["subagent_trajectory_ref"][0]
            ["trajectory_id"] = json!("missing");
        fs::write(&path, serde_json::to_vec(&dangling).unwrap()).unwrap();
        assert!(read_trajectory(&path)
            .unwrap_err()
            .to_string()
            .contains("unembedded subagent"));

        let mut duplicate = root;
        duplicate["subagent_trajectories"][0]["subagent_trajectories"][0]["trajectory_id"] =
            json!("child");
        duplicate["subagent_trajectories"][0]["steps"][0]["observation"]["results"][0]
            ["subagent_trajectory_ref"][0]["trajectory_id"] = json!("child");
        fs::write(&path, serde_json::to_vec(&duplicate).unwrap()).unwrap();
        assert!(read_trajectory(&path)
            .unwrap_err()
            .to_string()
            .contains("duplicated"));
    }

    fn printed_blocks(values: &[Value]) -> Value {
        let mut blocks = vec![
            json!({"type": "input_text", "text": "Script completed\nWall time 0.1 seconds\nOutput:\n"}),
        ];
        blocks.extend(
            values
                .iter()
                .map(|v| json!({"type": "input_text", "text": v.to_string()})),
        );
        json!({"content": blocks})
    }

    #[test]
    fn trajectory_results_reject_unaccounted_missing_and_duplicate_direct_prints() {
        let duplicate = "const r=await tools.exec_command({cmd:'true'}); text(r); text(r);";
        for (input, results) in [
            ("text(await tools.exec_command({cmd:'true'}));", vec![0,0]),
            ("text(await tools.exec_command({cmd:'true'}));", vec![0,127]),
            ("text(await tools.exec_command({cmd:'true'})); text(await tools.exec_command({cmd:'true'}));", vec![0]),
            (duplicate, vec![0,0]),
            (duplicate, vec![0,127]),
        ] {
            let args = json!({"input":input});
            let values: Vec<_> = results.into_iter().map(|code| json!({"exit_code":code,"output":""})).collect();
            assert!(trajectory_code_mode_invocations(args.as_object().unwrap(), "cell".into(), &printed_blocks(&values)).is_err(), "{input}");
        }
        // A dynamic display can emit multiple blocks or rewrite the binding.
        // It cannot establish which printed JSON belongs to the tool result.
        for diagnostic in [
            "text(text({exit_code:127}));",
            "text((r={exit_code:127}));",
            "text(ALL_TOOLS.map(x=>text(x)));",
            "if(r.session_id)text(JSON.stringify({session_id:r.session_id}));",
        ] {
            let args = json!({"input":format!("const r=await tools.exec_command({{cmd:'true'}}); {diagnostic} text(r);")});
            let values = trajectory_code_mode_invocations(
                args.as_object().unwrap(),
                "cell".into(),
                &printed_blocks(&[json!({"exit_code":0,"output":""})]),
            )
            .unwrap();
            assert!(values[0].observation.is_null(), "{diagnostic}");
        }
    }

    #[test]
    fn code_mode_authoritative_outcome_accepts_only_optional_string_output() {
        for output in [
            None,
            Some(json!("captured output")),
            Some(Value::Null),
            Some(json!(42)),
            Some(json!({"stdout":"x"})),
        ] {
            let mut outcome = json!({"status":"ok","returncode":0,"session_id":null});
            if let Some(value) = &output {
                outcome["output"] = value.clone();
            }
            let call = json!({"extra":{"riptide_nested_invocations":[{
                "index":0,"tool":"exec_command","arguments":{"cmd":"true"},"source_item_id":"source-0","outcome":outcome
            }]}});
            let parsed = authoritative_code_mode_invocations(
                call.as_object().unwrap(),
                &mut BTreeSet::new(),
                &mut BTreeSet::new(),
            );
            assert_eq!(
                parsed.is_ok(),
                output.as_ref().map_or(true, Value::is_string)
            );
            if let Ok(Some(invocations)) = parsed {
                assert_eq!(invocations[0].observation, outcome);
            }
        }
        for invalid in [
            json!({"status":"ok","returncode":0,"output":"missing session"}),
            json!({"status":"ok","returncode":0,"session_id":null,"output":"x","unexpected":true}),
            json!({"status":"ok","returncode":127,"session_id":null,"output":"contradiction"}),
        ] {
            let call = json!({"extra":{"riptide_nested_invocations":[{
                "index":0,"tool":"exec_command","arguments":{"cmd":"true"},"source_item_id":"source-0","outcome":invalid
            }]}});
            assert!(authoritative_code_mode_invocations(
                call.as_object().unwrap(),
                &mut BTreeSet::new(),
                &mut BTreeSet::new()
            )
            .is_err());
        }
    }

    #[test]
    fn trajectory_results_rejections_and_sessions_are_not_invented_exit_codes() {
        let input = json!({"input":"const r = await Promise.allSettled([tools.exec_command({cmd:'a'}),tools.exec_command({cmd:'b'})]); for(let i=0;i<r.length;i++){text({i,...r[i]});}"});
        let result = trajectory_code_mode_invocations(input.as_object().unwrap(), "cell".into(), &printed_blocks(&[
            json!({"i":0,"status":"rejected","reason":{"message":"tool unavailable"}}),
            json!({"i":1,"status":"fulfilled","value":{"exit_code":null,"session_id":42,"output":"ready"}}),
        ])).unwrap();
        assert_eq!(result[0].observation["status"], "unknown");
        assert!(result[0].observation["returncode"].is_null());
        assert_eq!(result[1].observation["status"], "running");
        assert_eq!(result[1].observation["session_id"], "42");
        for payload in [
            json!({"exit_code":0,"session_id":42}),
            json!({"exit_code":"0"}),
            json!({"exit_code":0,"output":null}),
            json!({"session_id":false}),
        ] {
            assert!(printed_tool_outcome(&payload, false).is_err());
        }
        assert!(
            printed_tool_outcome(&json!({"status":"rejected","value":{},"reason":"x"}), true)
                .is_err()
        );
    }

    #[test]
    fn trajectory_results_reject_duplicate_missing_and_invalid_indices() {
        let input = json!({"input":"const r = await Promise.allSettled([tools.exec_command({cmd:'a'}),tools.exec_command({cmd:'b'})]); for(let i=0;i<r.length;i++){text({i,...r[i]});}"});
        for indexes in [vec![0, 0], vec![0, 2], vec![0], vec![-1, 1]] {
            let values: Vec<Value> = indexes
                .into_iter()
                .map(|i| json!({"i":i,"status":"fulfilled","value":{"exit_code":0,"output":""}}))
                .collect();
            assert!(trajectory_code_mode_invocations(
                input.as_object().unwrap(),
                "cell".into(),
                &printed_blocks(&values)
            )
            .is_err());
        }
        let malformed = json!({"content":"[{'type':'input_text','text':'unfinished"});
        assert!(trajectory_code_mode_invocations(
            input.as_object().unwrap(),
            "cell".into(),
            &malformed
        )
        .is_err());
    }

    #[test]
    fn trajectory_results_do_not_bind_derived_or_partial_stdout() {
        for input in [
            "const r = await tools.exec_command({cmd:'a'}); text(r.output);",
            "text(await tools.exec_command({cmd:'a'})); await tools.exec_command({cmd:'b'});",
            "const r = await Promise.allSettled([tools.exec_command({cmd:'a'})]); for(let i=0;i<r.length;i++){text(r[i].output);}",
        ] {
            let args = json!({"input":input});
            let values = trajectory_code_mode_invocations(args.as_object().unwrap(), "cell".into(), &printed_blocks(&[json!({"exit_code":0,"output":"fake result in stdout"})])).unwrap();
            assert!(values.iter().all(|v| v.observation.is_null()));
        }
        let args = json!({"input":"text(await tools.exec_command({cmd:'a'}));"});
        let values = trajectory_code_mode_invocations(
            args.as_object().unwrap(),
            "cell".into(),
            &Value::Null,
        )
        .unwrap();
        assert!(values[0].observation.is_null());
        // Data decoding cannot run Python expressions, trailing code, or interpolation.
        for content in [
            "[__import__('os').system('false')]",
            "[]; tools.exec_command({cmd:'false'})",
            "[{'text':`hello ${danger()}`}]",
        ] {
            assert!(code_mode_result_payloads(&json!({"content":content})).is_err());
        }
    }

    #[test]
    fn replay_honors_recorded_shell_and_login_for_foreground_and_sessions() {
        for background in [false, true] {
            for (settings, command, expected) in [
                (
                    json!({"login": false, "tty": true}),
                    "test -t 0 && test -t 1 && test -t 2 && test -r /dev/tty && printf terminal > actual",
                    "terminal",
                ),
                (
                    json!({"shell": "/bin/sh", "login": false, "tty": false}),
                    "printf '%s' \"${BASH_VERSION-unset}\" > actual",
                    "unset",
                ),
                (
                    json!({"shell": "/bin/bash", "login": false}),
                    "if shopt -q login_shell; then printf login; else printf nonlogin; fi > actual",
                    "nonlogin",
                ),
                (
                    json!({"shell": "/bin/bash", "login": true}),
                    "if shopt -q login_shell; then printf login; else printf nonlogin; fi > actual",
                    "login",
                ),
                (
                    json!({}),
                    "if shopt -q login_shell; then printf login; else printf nonlogin; fi > actual",
                    "login",
                ),
            ] {
                let temporary = tempfile::tempdir().unwrap();
                let mut arguments = settings;
                arguments["cmd"] = json!(command);
                let mut actions = vec![(
                    "exec_command",
                    arguments,
                    json!({"content":
                    if background { "Process running with session ID 41" }
                    else { "Process exited with code 0" }}),
                )];
                if background {
                    actions.push((
                        "write_stdin",
                        json!({"session_id": 41, "chars": "", "yield_time_ms": 1000}),
                        json!({"content": "Process exited with code 0"}),
                    ));
                }
                let path = write_trajectory(temporary.path(), actions, default_agent());
                let report = replay_trajectory(
                    &read_trajectory(&path).unwrap(),
                    &replay_options(temporary.path()),
                )
                .unwrap();
                assert_eq!(
                    fs::read_to_string(temporary.path().join("actual")).unwrap(),
                    expected
                );
                assert_eq!(report["summary"]["actions_failed"], 0);
            }
        }
    }

    #[test]
    fn tty_capture_drains_large_merged_output_and_cleans_up_timeout() {
        let temporary = tempfile::tempdir().unwrap();
        for timeout in [false, true] {
            let output = run_captured_command(CapturedCommand {
                shell: "/bin/bash",
                shell_flag: "-c",
                command_text: if timeout {
                    "sleep 30"
                } else {
                    "head -c 262144 /dev/zero | tr '\\0' x; printf '\\nlast\\n' >&2"
                },
                cwd: temporary.path(),
                timeout: Some(Duration::from_millis(if timeout { 50 } else { 3000 })),
                combine_output: false,
                tty: true,
                artifacts_dir: temporary.path(),
                action_index: 0,
            })
            .unwrap();
            assert_eq!(output.timed_out, timeout);
            if !timeout {
                assert_eq!(output.returncode, 0);
                assert!(output.stdout.len() >= 262144);
                assert!(output.stdout.ends_with("last\r\n"));
                assert!(output.stderr.is_empty());
            } else {
                assert_eq!(
                    unsafe { libc::kill(-(output.process_root_pid as i32), 0) },
                    -1
                );
                assert_eq!(io::Error::last_os_error().raw_os_error(), Some(libc::ESRCH));
            }
        }
    }

    #[test]
    fn tty_input_backpressure_respects_timeout() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "stty raw -echo; sleep 30", "tty": true, "login": false, "yield_time_ms": 100}),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "chars": "x".repeat(1024 * 1024), "yield_time_ms": 1000}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.1);
        let started = Instant::now();
        let report = replay_trajectory(&read_trajectory(&path).unwrap(), &options).unwrap();
        assert_eq!(report["actions"][1]["status"], "timed_out");
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn tty_capture_disk_error_does_not_block_child_output() {
        let mut command = Command::new("/bin/bash");
        command.args(["-c", "head -c 1048576 /dev/zero"]);
        configure_new_session(&mut command);
        let mut terminal = PtyCapture::attach(
            &mut command,
            File::options().write(true).open("/dev/full").unwrap(),
        )
        .unwrap();
        let mut child = command.spawn().unwrap();
        drop(command);
        let (status, timed_out) =
            wait_with_timeout_cleanup(&mut child, Some(Duration::from_secs(3))).unwrap();
        assert!(!timed_out);
        assert!(status.success());
        assert_eq!(
            terminal.finish().unwrap_err().raw_os_error(),
            Some(libc::ENOSPC)
        );
    }

    #[test]
    fn tty_session_handles_terminal_input_and_eof() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "test -t 0 && test -t 1 && test -t 2 && cat > actual", "tty": true, "login": false, "yield_time_ms": 100}),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "chars": "hello\n\u{4}", "yield_time_ms": 1000}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let report = replay_trajectory(
            &read_trajectory(&path).unwrap(),
            &replay_options(temporary.path()),
        )
        .unwrap();
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
        assert_eq!(
            fs::read_to_string(temporary.path().join("actual")).unwrap(),
            "hello\n"
        );
    }

    #[test]
    fn replay_rejects_unsupported_execution_options_before_any_action() {
        for (option, value) in [
            ("tty", json!("true")),
            ("login", json!("false")),
            ("login", Value::Null),
            ("shell", json!("/bin/fish")),
            ("shell", json!("bash")),
            ("shell", json!("/bin/bash\0")),
            ("shell", Value::Null),
        ] {
            let temporary = tempfile::tempdir().unwrap();
            let mut arguments = json!({"cmd": "true"});
            arguments[option] = value;
            let path = write_trajectory(
                temporary.path(),
                vec![
                    (
                        "exec_command",
                        json!({"cmd": "touch executed-before-preflight"}),
                        json!({"content": "Process exited with code 0"}),
                    ),
                    (
                        "exec_command",
                        arguments,
                        json!({"content": "Process exited with code 0"}),
                    ),
                ],
                default_agent(),
            );
            let error = replay_trajectory(
                &read_trajectory(&path).unwrap(),
                &replay_options(temporary.path()),
            )
            .unwrap_err();
            assert!(error.to_string().contains(option));
            assert!(!temporary.path().join("executed-before-preflight").exists());
        }
    }

    #[test]
    fn missing_replay_workdir_does_not_reproduce_source_command_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let source = temporary.path().join("source");
        fs::create_dir(&source).unwrap();
        let command =
            "printf source-ran > executed-marker; definitely_missing_replay_audit_command";
        let original = Command::new("/bin/bash")
            .args(["-c", command])
            .current_dir(&source)
            .output()
            .unwrap();
        assert_eq!(original.status.code(), Some(127));
        assert!(source.join("executed-marker").exists());
        fs::rename(&source, temporary.path().join("preserved-source")).unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": command, "workdir": source}),
                json!({"content": String::from_utf8(original.stderr).unwrap()}),
            )],
            default_agent(),
        );
        let report = replay_trajectory(
            &read_trajectory(&path).unwrap(),
            &replay_options(temporary.path()),
        )
        .unwrap();
        assert_eq!(report["actions"][0]["launch_failed"], true);
        assert_eq!(report["actions"][0]["command_executed"], false);
        assert!(report["actions"][0]
            .get("source_input_validation_reproduced")
            .is_none());
        assert_eq!(report["summary"]["actions_executed"], 0);
        assert_eq!(report["summary"]["tool_actions_replayed"], 0);
        assert_eq!(report["summary"]["actions_launch_failed"], 1);
        assert_eq!(report["actions"][0]["status_matches_source"], Value::Null);
        assert_eq!(report["actions"][0]["outcome_matches_source"], Value::Null);

        let output = temporary.path().join("cli-report.json");
        let status = main(
            &[
                path.into_os_string(),
                "--cwd".into(),
                temporary.path().as_os_str().to_owned(),
                "--output".into(),
                output.clone().into_os_string(),
                "--pacing".into(),
                "unpaced".into(),
            ],
            "tasktrace replay atif",
        );
        assert_eq!(status, 1);
        let cli_report: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(cli_report["summary"]["actions_launch_failed"], 1);
        assert_eq!(cli_report["summary"]["tool_actions_replayed"], 0);
        assert_eq!(
            cli_report["actions"][0]["outcome_matches_source"],
            Value::Null
        );
    }

    #[test]
    fn executed_source_failure_remains_comparable_and_cli_successful() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": "exit 1"}),
                json!({"content": "Process exited with code 1"}),
            )],
            default_agent(),
        );
        let output = temporary.path().join("cli-report.json");
        assert_eq!(
            main(
                &[
                    path.into_os_string(),
                    "--cwd".into(),
                    temporary.path().as_os_str().to_owned(),
                    "--output".into(),
                    output.clone().into_os_string(),
                    "--pacing".into(),
                    "unpaced".into()
                ],
                "tasktrace replay atif"
            ),
            0
        );
        let report: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(report["actions"][0]["outcome_matches_source"], true);
        assert_eq!(report["summary"]["actions_launch_failed"], 0);
        assert_eq!(report["summary"]["actions_executed"], 1);
    }

    #[test]
    fn authoritative_invocations_preserve_order_parallelism_and_synthetic_events() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![("exec", json!({"input": ""}), Value::Null)],
            codex_exec_agent(),
        );
        add_nested_invocations(
            &path,
            vec![json!([{
                "index": 0,
                "tool": "apply_patch",
                "arguments": {"input": "*** Begin Patch\n*** End Patch"},
                "source_item_id": "telemetry-call",
                "outcome": null
            }])],
        );
        let mut synthetic: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        synthetic["steps"][0]["extra"]["riptide_telemetry_only"] = json!(true);
        fs::write(&path, serde_json::to_vec(&synthetic).unwrap()).unwrap();
        let action = &read_trajectory(&path).unwrap().actions[0];
        assert_eq!(action.source_tool, "exec.apply_patch");
        assert!(action.code_mode);
        assert_eq!(action.source_item_id.as_deref(), Some("telemetry-call"));

        let grouped = write_trajectory(
            &temporary.path().join("grouped"),
            vec![(
                "exec",
                json!({"input": "await invokeCommandsKnownOnlyAtRuntime();"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        add_nested_invocations(
            &grouped,
            vec![json!([
                {
                    "index": 0, "tool": "exec_command", "arguments": {"cmd": "printf first"},
                    "source_item_id": "group-call-1", "outcome": null,
                    "parallel_group_id": "source-promise"
                },
                {
                    "index": 1, "tool": "exec_command", "arguments": {"cmd": "printf second"},
                    "source_item_id": "group-call-2", "outcome": null,
                    "parallel_group_id": "source-promise"
                }
            ])],
        );
        let actions = read_trajectory(&grouped).unwrap().actions;
        assert_eq!(actions[0].arguments["cmd"], "printf first");
        assert_eq!(actions[1].arguments["cmd"], "printf second");
        assert_eq!(
            actions[0].parallel_group_id.as_deref(),
            Some("source-promise")
        );
        assert_eq!(
            actions[1].parallel_group_id.as_deref(),
            Some("source-promise")
        );
    }

    #[test]
    fn authoritative_invocations_fail_closed_on_schema_and_count_drift() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "await tools.exec_command({cmd: 'true'});"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        add_nested_invocations(
            &path,
            vec![json!([{
                "index": 0, "tool": "exec_command", "arguments": {"cmd": "true"},
                "source_item_id": "call-1", "outcome": {"status": "ok", "returncode": 0, "session_id": null}
            }])],
        );
        let valid: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        let assert_error = |mutated: Value, expected: &str| {
            fs::write(&path, serde_json::to_vec(&mutated).unwrap()).unwrap();
            let error = read_trajectory(&path).unwrap_err().to_string();
            assert!(
                error.contains(expected),
                "{error:?} did not contain {expected:?}"
            );
        };

        let mut empty = valid.clone();
        empty["steps"][0]["tool_calls"][0]["extra"]["riptide_nested_invocations"] = json!([]);
        assert_error(empty, "must not be empty");

        let mut extra_key = valid.clone();
        extra_key["steps"][0]["tool_calls"][0]["extra"]["riptide_nested_invocations"][0]
            ["unexpected"] = json!(true);
        assert_error(extra_key, "invalid keys or index");

        let mut bad_outcome = valid.clone();
        bad_outcome["steps"][0]["tool_calls"][0]["extra"]["riptide_nested_invocations"][0]
            ["outcome"]["status"] = json!("success");
        assert_error(bad_outcome, "invalid status");

        let mut count_drift = valid.clone();
        count_drift["extra"]["riptide_normalization"]["extracted_executable_event_count"] =
            json!(2);
        assert_error(count_drift, "counts are inconsistent");

        let mut legacy = valid;
        let call = legacy["steps"][0]["tool_calls"][0].as_object_mut().unwrap();
        let extra = call["extra"].as_object_mut().unwrap();
        extra.remove("riptide_nested_invocations");
        extra.insert("riptide_nested_outcomes".to_owned(), json!([]));
        assert_error(legacy, "was replaced");
    }

    #[test]
    fn status_and_outcome_comparisons_are_accounted_separately() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "false"}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "false"}),
                    json!({"content": "Process exited with code 2"}),
                ),
            ],
            default_agent(),
        );
        let output = temporary.path().join("report.json");
        assert_eq!(
            run_cli(CliOptions {
                trajectory,
                output: output.clone(),
                cwd: temporary.path().to_path_buf(),
                patch_roots: Vec::new(),
                pacing: Pacing::Unpaced,
                timeout_sec: None,
                action_output_dir: None,
            })
            .unwrap(),
            1
        );
        let report: Value = serde_json::from_slice(&fs::read(output).unwrap()).unwrap();
        assert_eq!(report["summary"]["tool_action_status_different"], 1);
        assert_eq!(report["summary"]["tool_action_status_equal"], 1);
        assert_eq!(report["summary"]["tool_action_outcome_different"], 2);
        assert_eq!(report["summary"]["tool_action_outcome_equal"], 0);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 2);
        assert_eq!(report["actions"][1]["status_matches_source"], true);
        assert_eq!(report["actions"][1]["outcome_matches_source"], false);
    }

    #[test]
    fn recorded_agent_wall_time_is_nullable_and_parsed_from_observation_text() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": "true"}),
                json!({"content": "Process exited with code 0\nWall time: 2.75 seconds"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(recorded_tool_wall_sec(&trajectory.actions[0]), Some(2.75));
        let report = replay_trajectory(&trajectory, &replay_options(temporary.path())).unwrap();
        assert_eq!(report["actions"][0]["source_reported_wall_time_sec"], 2.75);
        assert!(report["actions"][0].get("agent_output_preview").is_none());
        assert!(report["actions"][0].get("stdout_preview").is_none());

        let missing = write_trajectory(
            &temporary.path().join("missing"),
            vec![(
                "exec_command",
                json!({"cmd": "true"}),
                json!({"content": "Process exited with code 0"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&missing).unwrap();
        assert_eq!(recorded_tool_wall_sec(&trajectory.actions[0]), None);
    }

    #[test]
    fn normalizes_only_provenance_gated_mini_swe_bash() {
        let temporary = tempfile::tempdir().unwrap();
        let mini = write_trajectory(
            temporary.path(),
            vec![(
                "bash",
                json!({"command": "true"}),
                json!({"results": [{"content": "{\"returncode\": 0}"}]}),
            )],
            json!({
                "name": "mini-swe-agent",
                "extra": {"original_format": MINI_SWE_AGENT_ORIGINAL_FORMAT},
            }),
        );
        let trajectory = read_trajectory(&mini).unwrap();
        assert_eq!(trajectory.actions[0].tool, "exec_command");
        assert_eq!(trajectory.actions[0].source_tool, "bash");
        assert_eq!(trajectory.actions[0].arguments["cmd"], "true");

        let generic_root = temporary.path().join("generic");
        let generic = write_trajectory(
            &generic_root,
            vec![(
                "bash",
                json!({"command": "true"}),
                json!({"content": "Process exited with code 0"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&generic).unwrap();
        assert!(validate_supported_actions(&trajectory)
            .unwrap_err()
            .to_string()
            .contains("unsupported executable tool calls: bash"));
    }

    fn codex_exec_agent() -> Value {
        json!({"name": "codex", "extra": {"originator": "codex_exec"}})
    }

    fn claude_code_agent() -> Value {
        json!({"name": CLAUDE_CODE_AGENT_NAME})
    }

    #[test]
    fn claude_read_honors_window_and_rejects_symlink_escape() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::write(workspace.join("state.txt"), "first\nsecond\nthird\n").unwrap();
        let read = AtifAction {
            index: 1,
            step_id: Value::Null,
            timestamp_sec: 0.0,
            source_trajectory_id: "root".to_owned(),
            source_item_id: None,
            tool: "claude_read".to_owned(),
            source_tool: "Read".to_owned(),
            code_mode: false,
            parallel_group_id: None,
            source_agent_original_format: Some(CLAUDE_CODE_AGENT_NAME.to_owned()),
            arguments: Map::from_iter([
                (
                    "file_path".to_owned(),
                    Value::String("state.txt".to_owned()),
                ),
                ("offset".to_owned(), Value::from(2)),
                ("limit".to_owned(), Value::from(1)),
            ]),
            observation: Value::Null,
            observation_text: String::new(),
        };
        let result = claude_read(&read, &workspace, &path_policy(&workspace)).unwrap();
        assert_eq!(result["offset"], 2);
        assert_eq!(result["limit"], 1);
        assert_eq!(result["lines_read"], 1);
        assert_eq!(result["bytes_read"], 7);
        let outside = temporary.path().join("outside.txt");
        fs::write(&outside, "outside\n").unwrap();
        std::os::unix::fs::symlink(&outside, workspace.join("outside-link")).unwrap();
        let escaped = AtifAction {
            arguments: Map::from_iter([(
                "file_path".to_owned(),
                Value::String("outside-link".to_owned()),
            )]),
            ..read
        };
        assert!(claude_read(&escaped, &workspace, &path_policy(&workspace)).is_err());
    }

    #[test]
    fn claude_read_accepts_binary_files() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::write(workspace.join("document.pdf"), [0xff, 0xd8, 0xff, 0x00]).unwrap();
        let read = AtifAction {
            index: 1,
            step_id: Value::Null,
            timestamp_sec: 0.0,
            source_trajectory_id: "root".to_owned(),
            source_item_id: None,
            tool: "claude_read".to_owned(),
            source_tool: "Read".to_owned(),
            code_mode: false,
            parallel_group_id: None,
            source_agent_original_format: Some(CLAUDE_CODE_AGENT_NAME.to_owned()),
            arguments: Map::from_iter([(
                "file_path".to_owned(),
                Value::String("document.pdf".to_owned()),
            )]),
            observation: Value::Null,
            observation_text: String::new(),
        };
        let result = claude_read(&read, &workspace, &path_policy(&workspace)).unwrap();
        assert_eq!(result["status"], "ok");
        assert_eq!(result["binary"], true);
        assert_eq!(result["bytes_read"], 4);
        assert!(result["lines_read"].is_null());
    }

    #[test]
    fn claude_glob_matches_braces_caps_and_contains_paths() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        fs::create_dir(workspace.join("nested")).unwrap();
        fs::write(workspace.join("nested/one.rs"), "").unwrap();
        fs::write(workspace.join("nested/two.py"), "").unwrap();
        fs::write(workspace.join(".ignored.rs"), "").unwrap();
        for index in 0..101 {
            fs::write(workspace.join(format!("item-{index:03}.txt")), "").unwrap();
        }
        let action = AtifAction {
            index: 1,
            step_id: Value::Null,
            timestamp_sec: 0.0,
            source_trajectory_id: "root".to_owned(),
            source_item_id: None,
            tool: "claude_glob".to_owned(),
            source_tool: "Glob".to_owned(),
            code_mode: false,
            parallel_group_id: None,
            source_agent_original_format: Some(CLAUDE_CODE_AGENT_NAME.to_owned()),
            arguments: Map::from_iter([(
                "pattern".to_owned(),
                Value::String("**/*.{rs,py}".to_owned()),
            )]),
            observation: Value::Null,
            observation_text: String::new(),
        };
        let result = claude_glob(&action, &workspace, &path_policy(&workspace)).unwrap();
        assert_eq!(result["matches"].as_array().unwrap().len(), 3);
        assert!(result["matches"]
            .as_array()
            .unwrap()
            .iter()
            .any(|path| path.as_str().unwrap().ends_with(".ignored.rs")));
        let capped = AtifAction {
            arguments: Map::from_iter([(
                "pattern".to_owned(),
                Value::String("item-*.txt".to_owned()),
            )]),
            ..action
        };
        let result = claude_glob(&capped, &workspace, &path_policy(&workspace)).unwrap();
        assert_eq!(result["matches"].as_array().unwrap().len(), 100);
        assert_eq!(result["truncated"], true);
        let invalid = AtifAction {
            arguments: Map::from_iter([(
                "pattern".to_owned(),
                Value::String("bad\0pattern".to_owned()),
            )]),
            ..capped
        };
        assert!(claude_glob(&invalid, &workspace, &path_policy(&workspace)).is_err());
    }

    #[test]
    fn claude_grep_honors_nested_ignore_rules_and_explicit_overrides() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path();
        fs::create_dir_all(workspace.join(".git/info")).unwrap();
        fs::create_dir(workspace.join("nested")).unwrap();
        fs::write(
            workspace.join(".gitignore"),
            "ignored.txt\nnested/*.txt\n!nested/kept.txt\n",
        )
        .unwrap();
        fs::write(workspace.join(".ignore"), "ignored-by-ignore.txt\n").unwrap();
        fs::write(workspace.join(".rgignore"), "ignored-by-rgignore.txt\n").unwrap();
        fs::write(workspace.join(".git/info/exclude"), "local.txt\n").unwrap();
        for name in [
            "visible.txt",
            "ignored.txt",
            "ignored-by-ignore.txt",
            "ignored-by-rgignore.txt",
            "local.txt",
            ".hidden.txt",
            "nested/kept.txt",
            "nested/dropped.txt",
        ] {
            fs::write(workspace.join(name), "HELLO\n").unwrap();
        }
        let expected = [
            workspace.join("nested/kept.txt"),
            workspace.join("visible.txt"),
        ];
        assert_eq!(grep_files(workspace, None).unwrap(), expected);
        let overridden = grep_files(workspace, Some("*.txt")).unwrap();
        assert!(overridden.contains(&workspace.join("ignored.txt")));
        assert!(overridden.contains(&workspace.join(".hidden.txt")));
        assert!(overridden.contains(&workspace.join("nested/dropped.txt")));
        assert_eq!(
            grep_files(&workspace.join("ignored.txt"), None).unwrap(),
            [workspace.join("ignored.txt")]
        );

        let path = write_trajectory(
            &workspace.join("source"),
            vec![(
                "Grep",
                json!({"pattern": "hello", "path": "visible.txt", "output_mode": "content", "-i": true}),
                json!({"extra": {"tool_result_is_error": false}}),
            )],
            claude_code_agent(),
        );
        let action = &read_trajectory(&path).unwrap().actions[0];
        assert_eq!(
            claude_grep_with_policy(action, workspace, &path_policy(workspace), None).unwrap()
                ["matches"],
            1
        );
        let mut sensitive = action.clone();
        sensitive.arguments.insert("-i".to_owned(), json!(false));
        assert_eq!(
            claude_grep_with_policy(&sensitive, workspace, &path_policy(workspace), None).unwrap()
                ["matches"],
            0
        );
        sensitive.arguments.insert("-i".to_owned(), json!("true"));
        assert!(
            claude_grep_with_policy(&sensitive, workspace, &path_policy(workspace), None).is_err()
        );
    }

    #[test]
    fn claude_code_pairs_result_ids_and_fails_closed_for_unsafe_tools() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().join("task-name__trial-id").join("agent");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("trajectory.json");
        fs::write(&path, serde_json::to_vec(&json!({
            "schema_version": "ATIF-v1.5", "agent": claude_code_agent(), "steps": [{
                "step_id": 1, "timestamp": 0,
                "tool_calls": [
                    {"tool_call_id": "write", "function_name": "Write", "arguments": {"file_path": "a", "content": "a"}},
                    {"tool_call_id": "bash", "function_name": "Bash", "arguments": {"command": "true"}}
                ],
                "observation": {"results": [
                    {"source_call_id": "bash", "content": "bash result", "extra": {"tool_result_is_error": false}},
                    {"source_call_id": "write", "content": "write result", "extra": {"tool_result_metadata": {"raw_tool_result": {"is_error": true}}}}
                ]}
            }]
        })).unwrap()).unwrap();
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions[0].observation_text, "write result");
        assert_eq!(trajectory.actions[1].observation_text, "bash result");
        assert_eq!(source_outcome(&trajectory.actions[0]).status, "failed");
        assert_eq!(source_outcome(&trajectory.actions[1]).status, "ok");

        let unsafe_path = write_trajectory(
            &temporary.path().join("unsafe"),
            vec![("NotebookEdit", json!({"notebook_path": "x"}), Value::Null)],
            claude_code_agent(),
        );
        assert!(read_trajectory(&unsafe_path).is_err());

        let task = write_trajectory(
            &temporary.path().join("task"),
            vec![(
                "Agent",
                json!({"prompt": "change the workspace"}),
                Value::Null,
            )],
            claude_code_agent(),
        );
        assert!(read_trajectory(&task).is_err());

        let background = write_trajectory(
            &temporary.path().join("background"),
            vec![(
                "Bash",
                json!({"command": "sleep 1", "run_in_background": true}),
                Value::Null,
            )],
            claude_code_agent(),
        );
        assert!(read_trajectory(&background)
            .unwrap_err()
            .to_string()
            .contains("backgroundTaskId"));

        let unsandboxed = write_trajectory(
            &temporary.path().join("unsandboxed"),
            vec![(
                "Bash",
                json!({"command": "true", "dangerouslyDisableSandbox": true}),
                Value::Null,
            )],
            claude_code_agent(),
        );
        assert!(read_trajectory(&unsandboxed)
            .unwrap_err()
            .to_string()
            .contains("dangerouslyDisableSandbox=true"));
    }

    #[test]
    fn claude_code_rejects_alias_collision_and_task_lifecycle_control() {
        let temporary = tempfile::tempdir().unwrap();
        let alias = write_trajectory(
            &temporary.path().join("alias"),
            vec![("exec_command", json!({}), Value::Null)],
            claude_code_agent(),
        );
        assert!(read_trajectory(&alias)
            .unwrap_err()
            .to_string()
            .contains("unsupported Claude Code tool"));

        let lifecycle = write_trajectory(
            &temporary.path().join("lifecycle"),
            vec![("TaskStop", json!({"task_id": "task-1"}), Value::Null)],
            claude_code_agent(),
        );
        assert!(read_trajectory(&lifecycle).is_err());
    }

    #[test]
    fn claude_code_honors_top_level_result_is_error() {
        let temporary = tempfile::tempdir().unwrap();
        let directory = temporary.path().join("task-name__trial-id").join("agent");
        fs::create_dir_all(&directory).unwrap();
        let path = directory.join("trajectory.json");
        fs::write(&path, serde_json::to_vec(&json!({
            "schema_version": "ATIF-v1.7", "agent": claude_code_agent(), "steps": [{
                "timestamp": 0,
                "tool_calls": [{"tool_call_id": "write", "function_name": "Write", "arguments": {"file_path": "state.txt", "content": "written"}}],
                "observation": {"results": [{"source_call_id": "write", "content": "", "is_error": true}]}
            }]
        })).unwrap()).unwrap();
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(source_outcome(&trajectory.actions[0]).status, "failed");
    }

    #[test]
    fn recorded_task_logs_are_confined_to_claude_temp_paths() {
        let temporary = tempfile::Builder::new()
            .prefix("claude-")
            .tempdir_in("/tmp")
            .unwrap();
        let outside = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![("exec_command", json!({"cmd":"true"}), Value::Null)],
            default_agent(),
        );
        let mut action = read_trajectory(&path).unwrap().actions.remove(0);
        let valid = temporary.path().join("run/tasks/task.1.output");
        action.observation_text = format!("Output is being written to: {}", valid.display());
        assert_eq!(
            claude_recorded_task_output_path(&action, "task.1"),
            Some(valid)
        );
        assert_eq!(
            claude_task_output_path("/tmp/claude/tasks/task.1.output").as_deref(),
            Some("task.1")
        );
        action.observation_text =
            "Output is being written to: /tmp/claude/tasks/task.1.output".to_owned();
        assert_eq!(
            claude_recorded_task_output_path(&action, "task.1"),
            Some(PathBuf::from("/tmp/claude/tasks/task.1.output"))
        );
        std::os::unix::fs::symlink(outside.path(), temporary.path().join("alias")).unwrap();
        for raw in [
            outside
                .path()
                .join("tasks/task.1.output")
                .display()
                .to_string(),
            temporary
                .path()
                .join("../escape/tasks/task.1.output")
                .display()
                .to_string(),
            temporary
                .path()
                .join("alias/tasks/task.1.output")
                .display()
                .to_string(),
            temporary
                .path()
                .join("tasks/other.output")
                .display()
                .to_string(),
            "relative/tasks/task.1.output".to_owned(),
        ] {
            action.observation_text = format!("Output is being written to: {raw}");
            assert_eq!(
                claude_recorded_task_output_path(&action, "task.1"),
                None,
                "{raw}"
            );
        }
    }

    #[test]
    fn reaped_leader_does_not_authorize_numeric_group_signals() {
        let mut command = Command::new("sleep");
        command.arg("30");
        configure_new_session(&mut command);
        let mut unrelated = command.spawn().unwrap();
        let mut tracker = ProcessTreeTracker {
            processes: BTreeMap::new(),
            leader_reaped: true,
        };
        let result = signal_contained_processes(&unrelated, false, &mut tracker, libc::SIGKILL);
        let status =
            wait_without_kill(&mut unrelated, Some(Duration::from_millis(100)), None).unwrap();
        let _ = unrelated.kill();
        unrelated.wait().unwrap();
        result.unwrap();
        assert!(
            status.is_none(),
            "cleanup signaled a group without live ownership evidence"
        );
    }

    #[test]
    fn session_cleanup_uses_retained_pidfds_after_leader_is_reaped() {
        let temporary = tempfile::tempdir().unwrap();
        let mut command = Command::new("/bin/sh");
        command
            .args([
                "-c",
                "setsid sh -c 'echo $$ > escaped.pid; exec sleep 30' & read -r line",
            ])
            .current_dir(temporary.path())
            .stdin(Stdio::piped());
        configure_new_session(&mut command);
        let child = command.spawn().unwrap();
        let tracker = ProcessTreeTracker::capture(child.id() as i32).unwrap();
        let mut session = RunningSession {
            child,
            tracker,
            log_path: temporary.path().join("log"),
            terminal: None,
        };
        let deadline = Instant::now() + Duration::from_secs(2);
        while !temporary.path().join("escaped.pid").exists() && Instant::now() < deadline {
            session.poll().unwrap();
            thread::sleep(POLL_INTERVAL);
        }
        let escaped: i32 = fs::read_to_string(temporary.path().join("escaped.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        assert!(session.poll().unwrap().is_none());
        assert!(session.tracker.processes.contains_key(&escaped));
        session.child.kill().unwrap();
        while session.poll().unwrap().is_none() {
            thread::sleep(POLL_INTERVAL);
        }
        assert!(session.tracker.leader_reaped);
        terminate_session(&mut session).unwrap();
        assert!(pidfd_has_exited(&session.tracker.processes[&escaped].pidfd).unwrap());
    }

    #[test]
    fn claude_code_write_and_edit_cannot_escape_or_ambiguously_replace() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        fs::create_dir(&workspace).unwrap();
        let escaping = AtifAction {
            index: 1,
            step_id: Value::Null,
            timestamp_sec: 0.0,
            source_trajectory_id: "root".to_owned(),
            source_item_id: None,
            tool: "claude_write".to_owned(),
            source_tool: "Write".to_owned(),
            code_mode: false,
            parallel_group_id: None,
            source_agent_original_format: None,
            arguments: Map::from_iter([
                (
                    "file_path".to_owned(),
                    Value::String("../escape".to_owned()),
                ),
                ("content".to_owned(), Value::String("x".to_owned())),
            ]),
            observation: Value::Null,
            observation_text: String::new(),
        };
        assert!(claude_write(&escaping, &workspace, &path_policy(&workspace)).is_err());
        fs::write(workspace.join("state.txt"), "aa").unwrap();
        let ambiguous = AtifAction {
            arguments: Map::from_iter([
                (
                    "file_path".to_owned(),
                    Value::String("state.txt".to_owned()),
                ),
                ("old_string".to_owned(), Value::String("a".to_owned())),
                ("new_string".to_owned(), Value::String("b".to_owned())),
            ]),
            ..escaping
        };
        assert!(claude_edit(&ambiguous, &workspace, &path_policy(&workspace)).is_err());
    }

    #[test]
    fn path_policy_grants_only_recorded_external_file_access() {
        let temporary = tempfile::tempdir().unwrap();
        let workspace = temporary.path().join("workspace");
        let external = temporary.path().join("external.txt");
        let sibling = temporary.path().join("sibling.txt");
        fs::create_dir(&workspace).unwrap();
        fs::write(&external, "allowed").unwrap();
        fs::write(&sibling, "denied").unwrap();
        let path = write_trajectory(
            &temporary.path().join("source"),
            vec![(
                "Read",
                json!({"file_path": external.to_string_lossy()}),
                Value::Null,
            )],
            claude_code_agent(),
        );
        let mut trajectory = read_trajectory(&path).unwrap();
        let patch_target = temporary.path().join("recorded-patch.txt");
        let recorded_patch = format!(
            "*** Begin Patch\n*** Add File: {}\n+recorded\n*** End Patch",
            patch_target.display()
        );
        let mut patch_action = trajectory.actions[0].clone();
        patch_action.tool = "apply_patch".to_owned();
        patch_action.arguments =
            Map::from_iter([("input".to_owned(), Value::String(recorded_patch.clone()))]);
        trajectory.actions.push(patch_action);
        let policy = ReplayPathPolicy::for_trajectory(&workspace, &[], &trajectory).unwrap();
        assert_eq!(
            policy
                .resolve(&external.to_string_lossy(), &workspace, PathAccess::Read)
                .unwrap(),
            external
        );
        assert!(policy
            .resolve(&external.to_string_lossy(), &workspace, PathAccess::Write)
            .is_err());
        assert!(policy
            .resolve(&sibling.to_string_lossy(), &workspace, PathAccess::Read)
            .is_err());
        apply_patch_payload_with_policy(&recorded_patch, &workspace, &policy).unwrap();
        assert_eq!(fs::read_to_string(&patch_target).unwrap(), "recorded\n");
        let unrecorded_patch = format!(
            "*** Begin Patch\n*** Update File: {}\n@@\n-allowed\n+changed\n*** End Patch",
            external.display()
        );
        assert!(apply_patch_payload_with_policy(&unrecorded_patch, &workspace, &policy).is_err());
    }

    #[test]
    fn codex_exec_delimiters_preserve_quoted_content_and_reject_incomplete_statements() {
        for input in [
            r#"text(JSON.stringify({message: "π ) \" ("})); await tools.exec_command({cmd:'true'});"#,
            r#"const rs=await Promise.all([tools.exec_command({cmd:'true'})]); rs.forEach((r,i)=>text("π ) \" (" + r.output));"#,
            r#"const cmds=['true']; await Promise.all(cmds.map(cmd=>{const r=await tools.exec_command({cmd}); return {output:r.output,label:"π } \" {"};}));"#,
        ] {
            let parsed = CodeModeCell::new(input, "call".to_owned())
                .parse()
                .unwrap_or_else(|error| panic!("{input}: {error}"));
            assert_eq!(parsed.actions.len(), 1, "{input}");
            assert_eq!(parsed.actions[0].arguments["cmd"], "true", "{input}");
        }
        for (input, message) in [
            (r#"text("unterminated\"#, "unterminated output helper"),
            (
                "const rs=await Promise.all([tools.exec_command({cmd:'true'})]); rs.forEach((r,i)=>text(r.output);",
                "unterminated Promise.all presentation forEach",
            ),
            (
                "const cmds=['true']; await Promise.all(cmds.map(cmd=>{tools.exec_command({cmd});",
                "unterminated command batch map",
            ),
            (
                "text(tools.exec_command({cmd:'true'}));",
                "nested tools call in output helper",
            ),
        ] {
            let error = CodeModeCell::new(input, "call".to_owned())
                .parse()
                .err()
                .unwrap_or_else(|| panic!("accepted {input}"));
            assert_eq!(error.to_string(), format!("unsupported Codex exec payload: {message}"));
        }
    }

    #[test]
    fn codex_exec_sequentializes_literal_mapped_command_batches() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": r#"const specs = [["first", "printf first"], ["second", "printf second"]]; const results = await Promise.all(specs.map(async ([name, cmd]) => { const r = await tools.exec_command({cmd, workdir: "/app", yield_time_ms: 1}); return {name, output: r.output}; })); for (const r of results) text(r.output);"#}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.arguments["cmd"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["printf first", "printf second"]
        );
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_some()));

        let reversed = CodeModeCell::new(
            r#"const specs = [["git status", "status"]]; await Promise.all(specs.map(([cmd, label]) => tools.exec_command({cmd})));"#,
            "call".to_owned(),
        )
        .parse()
        .unwrap();
        assert_eq!(reversed.actions[0].arguments["cmd"], "git status");
    }

    #[test]
    fn codex_exec_sequentializes_static_string_command_batches() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const cmds = ['printf first', 'printf second']; const rs = await Promise.all(cmds.map(cmd => tools.exec_command({cmd, workdir: '/app'}))); rs.forEach((r,i)=>text(`--- ${i+1} --- $${r.output}`));"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.arguments["cmd"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["printf first", "printf second"]
        );
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_some()));
    }

    #[test]
    fn codex_exec_rejects_effectful_or_conditional_mapped_command_batches() {
        let temporary = tempfile::tempdir().unwrap();
        for (name, input) in [
            (
                "two-tools",
                "const cmds=['a']; await Promise.all(cmds.map(cmd => { tools.exec_command({cmd}); tools.exec_command({cmd}); }));",
            ),
            (
                "conditional-tool",
                "const cmds=['a']; await Promise.all(cmds.map(cmd => { if (cmd) tools.exec_command({cmd}); }));",
            ),
            (
                "unbound-command",
                "const cmds=['a']; await Promise.all(cmds.map(cmd => tools.exec_command({cmd:'a'})));",
            ),
            (
                "binding-width",
                "const cmds=[['a','b']]; await Promise.all(cmds.map(cmd => tools.exec_command({cmd})));",
            ),
        ] {
            let path = write_trajectory(
                &temporary.path().join(name),
                vec![("exec", json!({"input": input}), Value::Null)],
                codex_exec_agent(),
            );
            assert!(read_trajectory(&path).is_err(), "{name}");
        }
    }

    #[test]
    fn codex_exec_accepts_braced_promise_all_for_each_result_presentation() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const results = await Promise.all([tools.exec_command({cmd: 'printf release'}), tools.exec_command({cmd: 'printf debug'})]); results.forEach((r,i)=>{text(`BUILD${i}\n${r.output}\nexit=${r.exit_code}`)});"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.arguments["cmd"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["printf release", "printf debug"]
        );
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_some()));
    }

    #[test]
    fn codex_exec_accepts_mapped_output_only_promise_all_presentation() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const results = await Promise.all([tools.exec_command({cmd: 'printf first'}), tools.exec_command({cmd: 'printf second'})]); text(JSON.stringify(results.map((r,i)=>({i,output:r.output}))));"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions.len(), 2);
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_some()));
    }

    #[test]
    fn codex_exec_accepts_full_result_json_for_each_presentation() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const results = await Promise.all([tools.exec_command({cmd: 'printf first'}), tools.exec_command({cmd: 'printf second'})]); results.forEach((r,i)=>{text(JSON.stringify({i,...r}));});"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions.len(), 2);
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_some()));
    }

    #[test]
    fn codex_exec_replays_bounded_result_destructuring_and_ignores_tool_listing() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec",
                    json!({"input": "let{output,...rest}=await tools.exec_command({cmd: 'true'}); text(rest); text(output);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const xs = ALL_TOOLS.filter(x => x.name); text(xs);"}),
                    Value::Null,
                ),
            ],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions.len(), 1);
        assert_eq!(trajectory.actions[0].tool, "exec_command");
        assert_eq!(trajectory.actions[0].arguments["cmd"], "true");
    }

    #[test]
    fn codex_exec_accepts_exact_destructured_metadata_and_tool_listing_forms() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec",
                    json!({"input": "const x=ALL_TOOLS.find(x=>x.name===\"mcp__codex_apps__github_fetch_pr\"); text(x.description);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const xs=ALL_TOOLS.filter(x=>/github|pull request/i.test(x.name+\" \"+x.description)).slice(0,30); text(xs);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const names = ALL_TOOLS.filter(x => /web|fetch|http|url|search/.test(x.name+\" \"+x.description)).slice(0,20); text(names);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const hits = ALL_TOOLS.filter(x => /github|pull request|issue/.test((x.name+\" \"+x.description).toLowerCase())); text(hits);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const candidates = ALL_TOOLS.filter(x => /search/i.test(x.name + \" \" + x.description) && /web|github|internet/i.test(x.description)); text(candidates);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const hits = ALL_TOOLS.filter(x => /github_(get|fetch).*pull|github_search_prs|github_get_issue|github_fetch_issue/.test(x.name)).map(x => x.name+\" :: \"+x.description.slice(0,240)); text(hits);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const names = ALL_TOOLS.filter(x => x.name.startsWith(\"mcp__codex_apps__github_\") && /(get|fetch|list|search).*(issue|pull|pr|comment|review)|issue.*(get|fetch|list)|pull.*(get|fetch|list)/i.test(x.name)).map(x=>x.name); text(names);"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "for (const n of [\"mcp__codex_apps__github_fetch_issue\",\"mcp__codex_apps__github_fetch_issue_comments\",\"mcp__codex_apps__github_fetch_pr\",\"mcp__codex_apps__github_fetch_pr_comments\",\"mcp__codex_apps__github_fetch_pr_patch\",\"mcp__codex_apps__github_search_issues\",\"mcp__codex_apps__github_search_prs\"]) { const x=ALL_TOOLS.find(y=>y.name===n); text(x); }"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "await tools.mcp__codex_apps__github_search_issues({repo: \"gin-gonic/gin\", query: \"multipart\"});"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const [pr, comments] = await Promise.all([tools.mcp__codex_apps__github_fetch_pr({repo_full_name: 'apache/druid', pr_number: 1}), tools.mcp__codex_apps__github_fetch_pr_comments({repo_full_name: 'apache/druid', pr_number: 1})]); const p=pr.structuredContent?.pull_request||{}; text(JSON.stringify({p,comments}));"}),
                    Value::Null,
                ),
            ],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions.len(), 3);
        assert!(trajectory.actions.iter().all(is_metadata_action));
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_none()));
    }

    #[test]
    fn codex_exec_rejects_effectful_tool_listing_and_unbound_destructuring() {
        let temporary = tempfile::tempdir().unwrap();
        for (name, input) in [
            (
                "effectful-listing",
                "const x=ALL_TOOLS.find(x=>tools.exec_command({cmd:'true'})); text(x);",
            ),
            (
                "effectful-regex-listing",
                "const x=ALL_TOOLS.filter(x=>/github/i.test(tools.exec_command({cmd:'true'}))); text(x);",
            ),
            (
                "unsupported-regex-flags",
                "const x=ALL_TOOLS.filter(x=>/github/g.test(x.name+\" \"+x.description)); text(x);",
            ),
            (
                "effectful-lowercase-regex-listing",
                "const x=ALL_TOOLS.filter(x=>/github/.test((tools.exec_command({cmd:'true'})+x.description).toLowerCase())); text(x);",
            ),
            (
                "effectful-conjunctive-regex-listing",
                "const x=ALL_TOOLS.filter(x=>/github/i.test(x.name)&&/web/i.test(tools.exec_command({cmd:'true'}))); text(x);",
            ),
            (
                "effectful-prefixed-listing",
                "const x=ALL_TOOLS.filter(x=>x.name.startsWith(\"mcp__\")&&/github/i.test(tools.exec_command({cmd:'true'}))).map(x=>x.name); text(x);",
            ),
            (
                "unsupported-prefixed-listing-flags",
                "const x=ALL_TOOLS.filter(x=>x.name.startsWith(\"mcp__\")&&/github/g.test(x.name)).map(x=>x.name); text(x);",
            ),
            (
                "effectful-github-tool",
                "await tools.mcp__codex_apps__github_create_issue({repo:'gin-gonic/gin'});",
            ),
            (
                "unknown-read-prefixed-github-tool",
                "await tools.mcp__codex_apps__github_fetch_repository_secret({repo:'gin-gonic/gin'});",
            ),
            (
                "wrong-cardinality",
                "const [a] = await Promise.all([tools.exec_command({cmd:'true'}), tools.exec_command({cmd:'true'})]); text(a);",
            ),
            (
                "duplicate-binding",
                "const [a,a] = await Promise.all([tools.exec_command({cmd:'true'}), tools.exec_command({cmd:'true'})]); text(a);",
            ),
            (
                "effectful-derived-binding",
                "const [a] = await Promise.all([tools.exec_command({cmd:'true'})]); const x=run(); text(a);",
            ),
            (
                "substring-only-presentation",
                "const [a] = await Promise.all([tools.exec_command({cmd:'true'})]); text(data);",
            ),
            (
                "quoted-only-presentation",
                "const [a] = await Promise.all([tools.exec_command({cmd:'true'})]); text('a');",
            ),
            (
                "unicode-quoted-only-presentation",
                "const [a] = await Promise.all([tools.exec_command({cmd:'true'})]); text('é');",
            ),
            (
                "quoted-only-derived-binding",
                "const [a] = await Promise.all([tools.exec_command({cmd:'true'})]); const x='a'; text(x);",
            ),
        ] {
            let path = write_trajectory(
                &temporary.path().join(name),
                vec![("exec", json!({"input": input}), Value::Null)],
                codex_exec_agent(),
            );
            let error = read_trajectory(&path).unwrap_err().to_string();
            assert!(error.contains("unsupported Codex exec payload"), "{name}: {error}");
        }
    }

    #[test]
    fn codex_exec_accepts_only_static_view_image_batches() {
        let temporary = tempfile::tempdir().unwrap();
        for (name, input) in [
            (
                "literal",
                "const results = await Promise.all([tools.view_image({path:'/app/a.png',detail:'original'}),tools.view_image({path:'/app/b.png',detail:'original'})]); for (const im of results) image(im.image_url);",
            ),
            (
                "mapped",
                "const paths=['/app/a.png','/app/b.png']; const rs=await Promise.all(paths.map(path=>tools.view_image({path,detail:'original'}))); for(let i=0;i<rs.length;i++){text(paths[i]);image(rs[i].image_url);}",
            ),
            (
                "sequential-loop",
                "const paths=['/app/a.png','/app/b.png']; for(const p of paths){const r = await tools.view_image({path:p,detail:'original'});image(r.image_url);}",
            ),
        ] {
            let path = write_trajectory(
                &temporary.path().join(name),
                vec![("exec", json!({"input": input}), Value::Null)],
                codex_exec_agent(),
            );
            let trajectory = read_trajectory(&path)
                .unwrap_or_else(|error| panic!("{name}: {error}"));
            assert_eq!(trajectory.actions.len(), 2);
            assert!(trajectory
                .actions
                .iter()
                .all(|action| action.tool == "view_image"
                    && action.parallel_group_id.is_none()
                    && is_metadata_action(action)));
        }

        for (name, input) in [
            (
                "dynamic",
                "const paths=getPaths(); const rs=await Promise.all(paths.map(path=>tools.view_image({path,detail:'original'}))); for(let i=0;i<rs.length;i++){text(paths[i]);image(rs[i].image_url);}",
            ),
            (
                "wrong-detail",
                "const paths=['/app/a.png']; const rs=await Promise.all(paths.map(path=>tools.view_image({path,detail:'high'}))); for(let i=0;i<rs.length;i++){text(paths[i]);image(rs[i].image_url);}",
            ),
            (
                "effectful-presentation",
                "const paths=['/app/a.png']; const rs=await Promise.all(paths.map(path=>tools.view_image({path,detail:'original'}))); for(let i=0;i<rs.length;i++){tools.exec_command({cmd:'true'});image(rs[i].image_url);}",
            ),
        ] {
            let path = write_trajectory(
                &temporary.path().join(name),
                vec![("exec", json!({"input": input}), Value::Null)],
                codex_exec_agent(),
            );
            let error = read_trajectory(&path).unwrap_err().to_string();
            assert!(error.contains("unsupported Codex exec payload"), "{name}: {error}");
        }
    }

    #[test]
    fn codex_exec_requires_originator_and_fails_closed_for_dynamic_or_parallel_cells() {
        let temporary = tempfile::tempdir().unwrap();
        let generic = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "await tools.exec_command({cmd: 'true'});"}),
                Value::Null,
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&generic).unwrap();
        assert_eq!(trajectory.actions[0].tool, "exec");
        assert!(validate_supported_actions(&trajectory).is_err());

        for (name, input) in [
            (
                "control",
                "if (ok) await tools.exec_command({cmd: 'true'});",
            ),
            (
                "assigned-control",
                "if (false) x = await tools.exec_command({cmd: 'true'});",
            ),
            (
                "line-comment",
                "// x = await tools.exec_command({cmd: 'true'});",
            ),
            (
                "block-comment",
                "/* x = await tools.exec_command({cmd: 'true'}); */",
            ),
            ("dynamic", "await tools.exec_command({cmd: command});"),
            (
                "invalid-patch-object",
                "await tools.apply_patch({input: 1, patch: 'valid'});",
            ),
            ("unknown", "await tools.rm_rf({path: '/'});"),
        ] {
            let path = write_trajectory(
                &temporary.path().join(name),
                vec![("exec", json!({"input": input}), Value::Null)],
                codex_exec_agent(),
            );
            let error = read_trajectory(&path).unwrap_err().to_string();
            assert!(
                error.contains("unsupported Codex exec payload"),
                "{name}: {error}"
            );
        }
    }

    #[test]
    fn codex_authoritative_parallel_polls_require_distinct_sessions() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![("exec", json!({"input": ""}), Value::Null)],
            codex_exec_agent(),
        );
        let poll = |index: usize, session: u64| {
            json!({
                "index": index,
                "tool": "write_stdin",
                "arguments": {"session_id": session, "chars": "", "yield_time_ms": 1000},
                "source_item_id": format!("poll-{index}"),
                "outcome": {"status": "running", "returncode": null, "session_id": session.to_string()},
                "parallel_group_id": "parallel-polls"
            })
        };
        add_nested_invocations(&path, vec![json!([poll(0, 41), poll(1, 42)])]);
        let trajectory = read_trajectory(&path).unwrap();
        assert!(validate_supported_actions(&trajectory)
            .unwrap_err()
            .to_string()
            .contains("not supported in release 26.10"));

        add_nested_invocations(&path, vec![json!([poll(0, 41), poll(1, 41)])]);
        let error = read_trajectory(&path).unwrap_err().to_string();
        assert!(
            error.contains("parallel poll group repeats a session"),
            "{error}"
        );

        add_nested_invocations(
            &path,
            vec![json!([
                {
                    "index": 0, "tool": "exec_command", "arguments": {"cmd": "kill 1"},
                    "source_item_id": "mixed-exec", "outcome": null,
                    "parallel_group_id": "mixed-mutation"
                },
                {
                    "index": 1, "tool": "apply_patch", "arguments": {"input": "*** Begin Patch\n*** End Patch"},
                    "source_item_id": "mixed-patch", "outcome": null,
                    "parallel_group_id": "mixed-mutation"
                }
            ])],
        );
        let error = read_trajectory(&path).unwrap_err().to_string();
        assert!(error.contains("unsafe parallel group"), "{error}");
    }

    #[test]
    fn codex_exec_accepts_static_parallel_output_presentation() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const results = await Promise.all([tools.exec_command({cmd: 'printf first'}), tools.exec_command({cmd: 'printf second'})]); for (let i = 0; i < results.length; i++) { text(`---RESULT ${i+1}---\n${results[i].output}\n(exit ${results[i].exit_code})`); }"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.arguments["cmd"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["printf first", "printf second"]
        );
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.parallel_group_id.is_some()));
    }

    #[test]
    fn codex_exec_ignores_exact_session_id_presentation() {
        let temporary = tempfile::tempdir().unwrap();
        let valid = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec",
                    json!({"input": "const r = await tools.exec_command({cmd: 'printf first'}); text(r.output); if(r.session_id) text(JSON.stringify({session_id:r.session_id}));"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const r = await tools.exec_command({cmd: 'printf second'});\ntext(r.output);\nif (r.session_id) text(JSON.stringify({session_id:r.session_id}));"}),
                    Value::Null,
                ),
                (
                    "exec",
                    json!({"input": "const r = await tools.exec_command({cmd: 'printf third'}); text(r.output); if (r.session_id) text(`SESSION_ID=${r.session_id}`);"}),
                    Value::Null,
                ),
            ],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&valid).unwrap();
        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.arguments["cmd"].as_str().unwrap())
                .collect::<Vec<_>>(),
            ["printf first", "printf second", "printf third"]
        );

        let invalid = write_trajectory(
            &temporary.path().join("invalid"),
            vec![(
                "exec",
                json!({"input": "const r = await tools.exec_command({cmd: 'true'}); if(r.session_id) text(r.output);"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        assert!(read_trajectory(&invalid)
            .unwrap_err()
            .to_string()
            .contains("unsupported Codex exec payload"));
    }

    #[test]
    fn codex_exec_accepts_parenthesized_direct_await_only_when_static() {
        let temporary = tempfile::tempdir().unwrap();
        let valid = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const result = await (tools.exec_command({cmd: 'printf wrapped'})); text(result.output);"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&valid).unwrap();
        assert_eq!(trajectory.actions.len(), 1);
        assert_eq!(trajectory.actions[0].arguments["cmd"], "printf wrapped");

        let dynamic = write_trajectory(
            &temporary.path().join("dynamic"),
            vec![(
                "exec",
                json!({"input": "await (tools.exec_command({cmd: command}));"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        assert!(read_trajectory(&dynamic)
            .unwrap_err()
            .to_string()
            .contains("unsupported Codex exec payload"));
    }

    #[test]
    fn codex_exec_accepts_metadata_only_and_rejects_empty_or_mixed_promise_all() {
        let temporary = tempfile::tempdir().unwrap();
        for (name, input) in [
            ("empty", "await Promise.all([]);"),
            (
                "mixed",
                "await Promise.all([tools.update_plan({plan: []}), tools.exec_command({cmd: 'true'})]);",
            ),
        ] {
            let path = write_trajectory(
                &temporary.path().join(name),
                vec![("exec", json!({"input": input}), Value::Null)],
                codex_exec_agent(),
            );
            let error = read_trajectory(&path).unwrap_err().to_string();
            assert!(error.contains("unsupported Codex exec payload"), "{error}");
        }
        let metadata = write_trajectory(
            &temporary.path().join("metadata"),
            vec![(
                "exec",
                json!({"input": "await Promise.all([tools.update_plan({plan: []})]);"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&metadata).unwrap();
        assert_eq!(trajectory.actions.len(), 1);
        assert_eq!(trajectory.actions[0].tool, "update_plan");
        assert!(is_metadata_action(&trajectory.actions[0]));
    }

    #[test]
    fn codex_exec_parallel_group_ids_are_unique_and_never_inherit_outer_outcomes() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec",
                    json!({"input": "await Promise.all ([tools.exec_command({cmd: 'one'})]);"}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec",
                    json!({"input": "await Promise.all([tools.exec_command({cmd: 'two'})]);"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            codex_exec_agent(),
        );
        let mut payload: Value = serde_json::from_slice(&fs::read(&path).unwrap()).unwrap();
        for step in payload["steps"].as_array_mut().unwrap() {
            step["tool_calls"][0]["tool_call_id"] = json!("reused");
        }
        fs::write(&path, serde_json::to_vec(&payload).unwrap()).unwrap();

        let trajectory = read_trajectory(&path).unwrap();

        assert_eq!(
            trajectory
                .actions
                .iter()
                .map(|action| action.parallel_group_id.as_deref())
                .collect::<Vec<_>>(),
            [
                Some("step-1-call-1:reused:promise_all:1"),
                Some("step-2-call-1:reused:promise_all:1")
            ]
        );
        assert!(trajectory
            .actions
            .iter()
            .all(|action| action.observation.is_null()));
    }

    #[test]
    fn codex_exec_metadata_is_retained_as_ignored_metadata() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "await tools.wait({cell_id: 'missing'}).catch(()=>{}); text('ignored'); await tools.update_plan({plan: []}); await tools.spawn_agent({task_name: 'x', message: 'y'}); await tools.exec_command({cmd: 'true'});"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions.len(), 4);
        assert!(validate_supported_actions(&trajectory)
            .unwrap_err()
            .to_string()
            .contains("not supported in release 26.10"));
        assert_eq!(
            source_outcome(&trajectory.actions[0]).status,
            "ignored_metadata"
        );
    }

    #[test]
    fn codex_exec_direct_coordination_does_not_weaken_generic_validation() {
        let temporary = tempfile::tempdir().unwrap();
        let code_mode = write_trajectory(
            &temporary.path().join("code-mode"),
            vec![
                (
                    "spawn_agent",
                    json!({"task_name": "audit", "message": "inspect"}),
                    Value::Null,
                ),
                (
                    "exec_command",
                    json!({"cmd": "true"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            codex_exec_agent(),
        );
        let generic = write_trajectory(
            &temporary.path().join("generic"),
            vec![
                (
                    "spawn_agent",
                    json!({"task_name": "audit", "message": "inspect"}),
                    Value::Null,
                ),
                (
                    "exec_command",
                    json!({"cmd": "true"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );

        assert!(
            validate_supported_actions(&read_trajectory(&code_mode).unwrap())
                .unwrap_err()
                .to_string()
                .contains("not supported in release 26.10")
        );
        assert!(validate_supported_actions(&read_trajectory(&generic).unwrap()).is_err());
    }

    #[test]
    fn codex_exec_accepts_static_patch_template_in_text_wrapper() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "const patch = `*** Begin Patch\\n*** End Patch`; text(await tools.apply_patch(patch));"}),
                Value::Null,
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(trajectory.actions[0].tool, "apply_patch");
        assert_eq!(
            trajectory.actions[0].arguments["input"],
            "*** Begin Patch\n*** End Patch"
        );
    }

    #[test]
    fn code_mode_source_outcome_preserves_numeric_legacy_session_id() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "exec",
                json!({"input": "await tools.exec_command({cmd: 'sleep 1', yield_time_ms: 1});"}),
                json!({"status": "running", "returncode": null, "session_id": 44}),
            )],
            codex_exec_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        assert_eq!(
            source_outcome(&trajectory.actions[0]).session_id.as_deref(),
            Some("44")
        );
    }

    #[test]
    fn applies_add_update_move_delete_and_rejects_escape() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = temporary.path().join("workspace");
        fs::create_dir(&cwd).unwrap();
        fs::write(cwd.join("original.txt"), "old\n").unwrap();
        fs::write(cwd.join("remove.txt"), "remove\n").unwrap();
        let patch = "*** Begin Patch\n\
                     *** Add File: added.txt\n\
                     +created\n\
                     *** Update File: original.txt\n\
                     *** Move to: moved.txt\n\
                     @@\n\
                     -old\n\
                     +new\n\
                     *** Delete File: remove.txt\n\
                     *** End Patch";
        apply_patch_payload_with_policy(patch, &cwd, &path_policy(&cwd)).unwrap();
        assert_eq!(
            fs::read_to_string(cwd.join("added.txt")).unwrap(),
            "created\n"
        );
        assert_eq!(fs::read_to_string(cwd.join("moved.txt")).unwrap(), "new\n");
        assert!(!cwd.join("original.txt").exists());
        assert!(!cwd.join("remove.txt").exists());

        let escape = format!(
            "*** Begin Patch\n*** Add File: {}\n+bad\n*** End Patch",
            temporary.path().join("escaped.txt").display()
        );
        assert!(
            apply_patch_payload_with_policy(&escape, &cwd, &path_policy(&cwd))
                .unwrap_err()
                .to_string()
                .contains("leaves the permitted replay roots")
        );
    }

    #[test]
    fn tmp_patch_root_allows_normalized_tmp_and_rejects_symlink_and_other_escapes() {
        let temporary = tempfile::tempdir().unwrap();
        let cwd = temporary.path().join("app");
        let tmp = temporary.path().join("tmp");
        fs::create_dir(&cwd).unwrap();
        fs::create_dir(&tmp).unwrap();
        let policy = ReplayPathPolicy::from_roots(&cwd, std::slice::from_ref(&tmp)).unwrap();
        assert_eq!(
            policy
                .resolve("../tmp/reproduce.cjs", &cwd, PathAccess::Write)
                .unwrap(),
            tmp.join("reproduce.cjs")
        );
        assert!(policy
            .resolve("../etc/passwd", &cwd, PathAccess::Write)
            .unwrap_err()
            .to_string()
            .contains("leaves the permitted replay roots"));

        let link = tmp.join("outside");
        std::os::unix::fs::symlink("/etc", &link).unwrap();
        let path = link.join("passwd");
        assert!(policy
            .resolve(path.to_str().unwrap(), &cwd, PathAccess::Write)
            .unwrap_err()
            .to_string()
            .contains("leaves the permitted replay roots"));
    }

    #[test]
    fn replays_commands_metadata_and_source_argument_failure() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "printf done > result.txt"}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "update_plan",
                    json!({"plan": []}),
                    json!({"content": "Plan updated"}),
                ),
                (
                    "exec_command",
                    json!({"not_cmd": "printf bad > bad.txt"}),
                    json!({"content": "failed to parse function arguments: missing field `cmd`"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "printf rejected > rejected.txt", "yield_time_ms": "1000"}),
                    json!({"content": "failed to parse function arguments: invalid type"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let report = replay_trajectory(&trajectory, &replay_options(temporary.path())).unwrap();
        assert_eq!(
            fs::read_to_string(temporary.path().join("result.txt")).unwrap(),
            "done"
        );
        assert!(!temporary.path().join("bad.txt").exists());
        assert!(!temporary.path().join("rejected.txt").exists());
        assert_eq!(report["schema_version"], REPLAY_SCHEMA_VERSION);
        assert_eq!(report["task_name"], "task-name");
        assert_eq!(report["summary"]["actions_executed"], 1);
        assert_eq!(report["summary"]["metadata_actions_ignored"], 1);
        assert_eq!(report["summary"]["actions_source_argument_parse_failed"], 2);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
        assert!(report["actions"][0]["process_root_pid"]
            .as_u64()
            .is_some_and(|pid| pid > 0));
        for index in [0, 2, 3] {
            assert!(report["actions"][index]["replay_start_realtime_ns"]
                .as_u64()
                .is_some_and(|timestamp| timestamp > 0));
            assert!(
                report["actions"][index]["replay_start_realtime_uncertainty_ns"]
                    .as_u64()
                    .is_some_and(|uncertainty| uncertainty <= 1_000_000_000)
            );
        }
        assert!(report["actions"][1]
            .get("replay_start_realtime_ns")
            .is_none());
        assert!(report["actions"][1]
            .get("replay_start_realtime_uncertainty_ns")
            .is_none());
    }

    #[test]
    fn skipped_actions_do_not_receive_realtime_correlation() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "bash",
                    json!({"command": "touch must-not-exist"}),
                    json!({"results": [{"content": "{\"returncode\": null, \"exception_info\": \"action was not executed\"}"}]}),
                ),
                (
                    "bash",
                    json!({"command": "true"}),
                    json!({"results": [{"content": "{\"returncode\": 0}"}]}),
                ),
            ],
            json!({
                "name": "mini-swe-agent",
                "extra": {"original_format": MINI_SWE_AGENT_ORIGINAL_FORMAT},
            }),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let report = replay_trajectory(&trajectory, &replay_options(temporary.path())).unwrap();

        assert_eq!(report["actions"][0]["status"], "skipped_not_executed");
        assert!(report["actions"][0]
            .get("replay_start_realtime_ns")
            .is_none());
        assert!(report["actions"][0]
            .get("replay_start_realtime_uncertainty_ns")
            .is_none());
        assert!(report["actions"][1]["replay_start_realtime_ns"]
            .as_u64()
            .is_some_and(|timestamp| timestamp > 0));
        assert!(!temporary.path().join("must-not-exist").exists());
    }

    #[test]
    fn supports_interactive_command_sessions() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({
                        "cmd": "read value; printf '%s' \"$value\" > interactive.txt",
                        "yield_time_ms": 1,
                    }),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "chars": "done\n"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(2.0);
        let report = replay_trajectory(&trajectory, &options).unwrap();
        assert_eq!(
            fs::read_to_string(temporary.path().join("interactive.txt")).unwrap(),
            "done"
        );
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
        assert_eq!(
            report["actions"][0]["process_root_pid"],
            report["actions"][1]["process_root_pid"]
        );
    }

    #[test]
    fn final_session_cleanup_kills_term_ignoring_descendant_after_leader_exits() {
        struct KillPidOnDrop(u32);

        impl Drop for KillPidOnDrop {
            fn drop(&mut self) {
                unsafe {
                    libc::kill(i32::try_from(self.0).unwrap_or(i32::MAX), libc::SIGKILL);
                }
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({
                    "cmd": "(trap '' TERM; exec </dev/null >/dev/null 2>&1; touch session-descendant.ready; exec sleep 30) & descendant=$!; printf '%s\n' \"$descendant\" > session-descendant.pid.tmp; mv session-descendant.pid.tmp session-descendant.pid; while [ ! -f session-descendant.ready ]; do sleep 0.01; done; wait",
                    "yield_time_ms": 100,
                }),
                json!({"content": "Process running with session ID 44"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let replay_result = replay_trajectory(&trajectory, &replay_options(temporary.path()));

        let descendant: u32 = fs::read_to_string(temporary.path().join("session-descendant.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let _cleanup = KillPidOnDrop(descendant);
        let report = replay_result.unwrap();
        assert_eq!(report["actions"][0]["status"], "running");

        let descendant_proc = format!("/proc/{descendant}");
        for _ in 0..100 {
            if !Path::new(&descendant_proc).exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !Path::new(&descendant_proc).exists(),
            "TERM-ignoring interactive descendant {descendant} survived final session cleanup"
        );
    }

    #[test]
    fn write_stdin_observation_expiry_keeps_session_and_replay_continues() {
        struct KillPidOnDrop(u32);

        impl Drop for KillPidOnDrop {
            fn drop(&mut self) {
                unsafe {
                    libc::kill(i32::try_from(self.0).unwrap_or(i32::MAX), libc::SIGKILL);
                }
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({
                        "cmd": "(trap '' TERM; exec </dev/null >/dev/null 2>&1; touch poll-descendant.ready; exec sleep 30) & descendant=$!; printf '%s\n' \"$descendant\" > poll-descendant.pid.tmp; mv poll-descendant.pid.tmp poll-descendant.pid; while [ ! -f poll-descendant.ready ]; do sleep 0.01; done; wait",
                        "yield_time_ms": 100,
                    }),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "yield_time_ms": 50}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "touch continued-after-session-observation"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.5);
        options.action_output_dir = Some(temporary.path().join("outputs"));
        let replay_result = replay_trajectory(&trajectory, &options);

        let descendant: u32 = fs::read_to_string(temporary.path().join("poll-descendant.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let _cleanup = KillPidOnDrop(descendant);
        let report = replay_result.unwrap();
        assert_eq!(report["actions"][1]["status"], "running");
        assert_eq!(report["actions"][1]["session_id"], "44");
        assert_eq!(
            report["actions"][1]["process_root_pid"],
            report["actions"][0]["process_root_pid"]
        );
        assert_eq!(
            report["actions"][1]["session_output_preview"],
            report["actions"][0]["session_output_preview"]
        );
        assert_eq!(report["summary"]["actions_timed_out"], 0);
        assert_eq!(report["summary"]["tool_actions_replayed"], 3);
        assert!(temporary
            .path()
            .join("continued-after-session-observation")
            .is_file());

        let descendant_proc = format!("/proc/{descendant}");
        for _ in 0..100 {
            if !Path::new(&descendant_proc).exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !Path::new(&descendant_proc).exists(),
            "TERM-ignoring interactive descendant {descendant} survived write_stdin timeout cleanup"
        );
    }

    #[test]
    fn terminal_write_stdin_poll_expiry_is_running_and_replay_continues() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "sleep 5", "yield_time_ms": 1}),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "yield_time_ms": 1}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "touch continued-after-poll"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(2.0);
        let report = replay_trajectory(&trajectory, &options).unwrap();

        assert_eq!(report["actions"][1]["status"], "running");
        assert_eq!(report["actions"][1]["outcome_matches_source"], false);
        assert_eq!(report["summary"]["actions_timed_out"], 0);
        assert_eq!(
            report["summary"]["aborted_after_timeout_action_index"],
            Value::Null
        );
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 1);
        assert_eq!(report["summary"]["tool_actions_replayed"], 3);
        assert!(temporary.path().join("continued-after-poll").is_file());
    }

    #[test]
    fn write_stdin_hard_timeout_caps_long_observation_and_aborts() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "sleep 30", "yield_time_ms": 1}),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "yield_time_ms": 1000}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "touch should-not-run-after-write-timeout"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.05);
        let report =
            replay_trajectory(&read_trajectory(&trajectory_path).unwrap(), &options).unwrap();

        assert_eq!(report["actions"][1]["status"], "timed_out");
        assert_eq!(report["actions"][1]["returncode"], 124);
        assert_eq!(report["summary"]["actions_timed_out"], 1);
        assert_eq!(report["summary"]["actions_skipped_after_timeout"], 1);
        assert!(!temporary
            .path()
            .join("should-not-run-after-write-timeout")
            .exists());
    }

    #[test]
    fn write_stdin_ctrl_c_signals_the_contained_session() {
        for tty in [false, true] {
            let temporary = tempfile::tempdir().unwrap();
            let trajectory_path = write_trajectory(
                temporary.path(),
                vec![
                    (
                        "exec_command",
                        json!({"cmd": "sleep 30", "yield_time_ms": 100, "tty": tty}),
                        json!({"content": "Process running with session ID 44"}),
                    ),
                    (
                        "write_stdin",
                        json!({"session_id": 44, "chars": "\u{3}", "yield_time_ms": 100}),
                        json!({"content": "Process exited with code 130"}),
                    ),
                    (
                        "exec_command",
                        json!({"cmd": "touch continued-after-interrupt"}),
                        json!({"content": "Process exited with code 0"}),
                    ),
                ],
                default_agent(),
            );
            let mut options = replay_options(temporary.path());
            options.timeout_sec = Some(2.0);
            let report =
                replay_trajectory(&read_trajectory(&trajectory_path).unwrap(), &options).unwrap();

            assert_eq!(report["actions"][1]["status"], "failed");
            assert_eq!(report["actions"][1]["returncode"], 130);
            assert_eq!(report["actions"][1]["outcome_matches_source"], true);
            assert_eq!(report["summary"]["actions_timed_out"], 0);
            assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
            assert!(temporary.path().join("continued-after-interrupt").is_file());
        }
    }

    #[test]
    fn running_write_stdin_retains_single_poll_behavior() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "read -r line", "yield_time_ms": 1}),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "yield_time_ms": 1}),
                    json!({"content": "Process running with session ID 44"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 44, "chars": "done\n"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(2.0);
        let report =
            replay_trajectory(&read_trajectory(&trajectory_path).unwrap(), &options).unwrap();

        assert_eq!(report["actions"][1]["status"], "running");
        assert_eq!(report["actions"][2]["status"], "ok");
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
    }

    #[test]
    fn early_session_completion_is_an_outcome_match_with_timing_drift() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "true", "yield_time_ms": 100}),
                    json!({"content": "Process running with session ID 41"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "false", "yield_time_ms": 100}),
                    json!({"content": "Process running with session ID 42"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "sleep 30", "yield_time_ms": 1}),
                    json!({"content": "Process running with session ID 43"}),
                ),
            ],
            default_agent(),
        );
        let report = replay_trajectory(
            &read_trajectory(&trajectory_path).unwrap(),
            &replay_options(temporary.path()),
        )
        .unwrap();

        assert_eq!(report["actions"][0]["status"], "ok");
        assert_eq!(report["actions"][1]["status"], "failed");
        assert_eq!(report["actions"][2]["status"], "running");
        assert_eq!(report["actions"][0]["outcome_matches_source"], true);
        assert_eq!(report["actions"][1]["outcome_matches_source"], true);
        assert_eq!(report["actions"][2]["outcome_matches_source"], true);
        assert_eq!(report["actions"][0]["status_matches_source"], false);
        assert_eq!(report["actions"][1]["status_matches_source"], false);
        assert_eq!(report["actions"][2]["status_matches_source"], true);
        for index in [0, 1] {
            assert_eq!(
                report["actions"][index]["completion_timing_drift"],
                "completed_before_recorded_poll"
            );
        }
        assert!(report["actions"][2]["completion_timing_drift"].is_null());
        assert_eq!(report["summary"]["tool_action_status_equal"], 1);
        assert_eq!(report["summary"]["tool_action_status_different"], 2);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
    }

    #[test]
    fn unknown_write_stdin_session_is_reported_and_replay_continues() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({
                        "cmd": "sleep 30",
                        "workdir": "missing-workdir",
                        "yield_time_ms": 1,
                    }),
                    json!({"content": "Process running with session ID 13846"}),
                ),
                (
                    "write_stdin",
                    json!({"session_id": 13846}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "touch continued-after-missing-session"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.action_output_dir = Some(temporary.path().join("outputs"));
        let report = replay_trajectory(&trajectory, &options).unwrap();

        assert_eq!(report["actions"][0]["status"], "failed");
        assert_eq!(report["actions"][0]["launch_failed"], true);
        assert_eq!(report["actions"][0]["command_executed"], false);
        assert_eq!(report["actions"][1]["status"], "failed");
        assert_eq!(report["actions"][1]["returncode"], -1);
        assert_eq!(report["actions"][1]["session_id"], "13846");
        assert_eq!(
            report["actions"][1]["stderr_preview"],
            "ATIF action 2 references unknown session 13846"
        );
        assert_eq!(report["actions"][1]["outcome_matches_source"], false);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 1);
        assert_eq!(report["summary"]["tool_actions_recorded"], 3);
        assert_eq!(report["summary"]["tool_actions_replayed"], 2);
        assert_eq!(report["summary"]["actions_executed"], 2);
        assert!(temporary
            .path()
            .join("continued-after-missing-session")
            .is_file());
    }

    #[test]
    fn successful_mktemp_recreates_the_recorded_directory_deterministically() {
        let temporary = tempfile::tempdir().unwrap();
        let recorded = temporary.path().join("recorded.A1b2C3");
        let template = temporary.path().join("recorded.XXXXXX");
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": format!("mktemp -d {}", template.display())}),
                json!({"content": format!(
                    "Process exited with code 0\n[{{'type': 'input_text', 'text': 'Script completed\\nOutput:\\n'}}, {{'type': 'input_text', 'text': '{}\\n'}}]",
                    recorded.display()
                )}),
            )],
            default_agent(),
        );

        let report = replay_trajectory(
            &read_trajectory(&trajectory_path).unwrap(),
            &replay_options(temporary.path()),
        )
        .unwrap();

        assert!(recorded.is_dir());
        assert_eq!(report["actions"][0]["status"], "ok");
        assert_eq!(
            report["actions"][0]["command"],
            format!("mktemp -d {}", template.display())
        );
    }

    #[test]
    fn successful_mktemp_without_one_safe_recorded_result_fails_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let template = temporary.path().join("recorded.XXXXXX");
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": format!("mktemp -d {}", template.display())}),
                json!({"content": "Process exited with code 0\nOutput:\n/tmp/unrelated.A1b2C3\n"}),
            )],
            default_agent(),
        );

        let error = replay_trajectory(
            &read_trajectory(&trajectory_path).unwrap(),
            &replay_options(temporary.path()),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("successful mktemp has no unique recorded /tmp result"));
    }

    #[test]
    fn successful_mktemp_with_two_distinct_recorded_results_fails_closed() {
        let temporary = tempfile::tempdir().unwrap();
        let template = temporary.path().join("recorded.XXXXXX");
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({"cmd": format!("mktemp -d {}", template.display())}),
                json!({"content": format!(
                    "Process exited with code 0\n{}\\n{}\\n",
                    temporary.path().join("recorded.A1b2C3").display(),
                    temporary.path().join("recorded.D4e5F6").display()
                )}),
            )],
            default_agent(),
        );

        let error = replay_trajectory(
            &read_trajectory(&trajectory_path).unwrap(),
            &replay_options(temporary.path()),
        )
        .unwrap_err();
        assert!(error
            .to_string()
            .contains("successful mktemp has no unique recorded /tmp result"));
    }

    #[test]
    fn timeout_kills_process_group_and_stops_later_actions() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "sleep 30 & child=$!; printf '%s\n' \"$child\" > child.pid.tmp; mv child.pid.tmp child.pid; wait"}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "touch should-not-run"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        // Leave enough startup time for the shell to publish the child PID
        // atomically even when the full test suite is compiling concurrently.
        options.timeout_sec = Some(0.5);
        let report = replay_trajectory(&trajectory, &options).unwrap();
        assert_eq!(report["summary"]["actions_timed_out"], 1);
        assert_eq!(report["summary"]["actions_skipped_after_timeout"], 1);
        assert_eq!(report["summary"]["aborted_after_timeout_action_index"], 1);
        assert!(!temporary.path().join("should-not-run").exists());

        let child: u32 = fs::read_to_string(temporary.path().join("child.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        for _ in 0..100 {
            if !Path::new(&format!("/proc/{child}")).exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(!Path::new(&format!("/proc/{child}")).exists());
    }

    #[test]
    fn foreground_action_does_not_wait_for_background_descendant_output_handles() {
        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![
                (
                    "exec_command",
                    json!({"cmd": "sleep 3 & printf '%s\n' \"$!\" > background.pid"}),
                    json!({"content": "Process exited with code 0"}),
                ),
                (
                    "exec_command",
                    json!({"cmd": "kill $(cat background.pid)"}),
                    json!({"content": "Process exited with code 0"}),
                ),
            ],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.5);
        let report = replay_trajectory(&trajectory, &options).unwrap();

        assert_eq!(report["actions"][0]["status"], "ok");
        assert!(report["actions"][0]["wall_time_sec"].as_f64().unwrap() < 1.0);
        assert_eq!(report["actions"][1]["status"], "ok");
        assert_eq!(report["summary"]["actions_timed_out"], 0);
        assert_eq!(report["summary"]["source_outcome_mismatch_count"], 0);
    }

    #[test]
    fn timeout_kills_term_ignoring_descendant_after_leader_exits() {
        struct KillPidOnDrop(u32);

        impl Drop for KillPidOnDrop {
            fn drop(&mut self) {
                unsafe {
                    libc::kill(i32::try_from(self.0).unwrap_or(i32::MAX), libc::SIGKILL);
                }
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({
                    "cmd": "(trap '' TERM; exec </dev/null >/dev/null 2>&1; touch descendant.ready; exec sleep 30) & descendant=$!; printf '%s\n' \"$descendant\" > descendant.pid.tmp; mv descendant.pid.tmp descendant.pid; while [ ! -f descendant.ready ]; do sleep 0.01; done; wait"
                }),
                json!({"content": "Process exited with code 0"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.5);
        let replay_result = replay_trajectory(&trajectory, &options);

        let descendant: u32 = fs::read_to_string(temporary.path().join("descendant.pid"))
            .unwrap()
            .trim()
            .parse()
            .unwrap();
        let _cleanup = KillPidOnDrop(descendant);
        let report = replay_result.unwrap();
        assert_eq!(report["summary"]["actions_timed_out"], 1);
        assert_eq!(report["actions"][0]["status"], "timed_out");

        let descendant_proc = format!("/proc/{descendant}");
        for _ in 0..100 {
            if !Path::new(&descendant_proc).exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !Path::new(&descendant_proc).exists(),
            "TERM-ignoring descendant {descendant} survived timeout cleanup"
        );
    }

    #[test]
    fn timeout_kills_setsid_term_ignoring_closed_io_descendant_via_pidfd() {
        struct KillPidOnDrop(u32);

        impl Drop for KillPidOnDrop {
            fn drop(&mut self) {
                unsafe {
                    libc::kill(i32::try_from(self.0).unwrap_or(i32::MAX), libc::SIGKILL);
                }
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({
                    "cmd": r#"/usr/bin/setsid /bin/sh -c 'trap "" TERM; printf "%s %s\n" "$$" "$(ps -o sid= -p $$)" > setsid-descendant.pid.tmp; mv setsid-descendant.pid.tmp setsid-descendant.pid; exec </dev/null >/dev/null 2>&1; exec sleep 30' & while [ ! -s setsid-descendant.pid ]; do sleep 0.01; done; wait"#
                }),
                json!({"content": "Process exited with code 0"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.5);
        let replay_result = replay_trajectory(&trajectory, &options);

        let identity = fs::read_to_string(temporary.path().join("setsid-descendant.pid")).unwrap();
        let fields: Vec<u32> = identity
            .split_whitespace()
            .map(|field| field.parse().unwrap())
            .collect();
        assert_eq!(fields.len(), 2);
        let descendant = fields[0];
        let _cleanup = KillPidOnDrop(descendant);
        assert_eq!(
            fields[1], descendant,
            "adversarial descendant did not enter its own session"
        );

        let report = replay_result.unwrap();
        assert_eq!(report["summary"]["actions_timed_out"], 1);
        assert_eq!(report["actions"][0]["status"], "timed_out");

        let descendant_proc = format!("/proc/{descendant}");
        for _ in 0..100 {
            if !Path::new(&descendant_proc).exists() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert!(
            !Path::new(&descendant_proc).exists(),
            "setsid TERM-ignoring closed-I/O descendant {descendant} survived pidfd cleanup"
        );
    }

    #[test]
    fn timeout_refreshes_pidfds_for_descendant_forked_during_term_grace() {
        struct KillPidOnDrop(u32);

        impl Drop for KillPidOnDrop {
            fn drop(&mut self) {
                unsafe {
                    libc::kill(i32::try_from(self.0).unwrap_or(i32::MAX), libc::SIGKILL);
                }
            }
        }

        let temporary = tempfile::tempdir().unwrap();
        fs::write(
            temporary.path().join("spawn-after-term.py"),
            r#"import os
import signal
import time

os.setsid()

def close_stdio():
    for descriptor in (0, 1, 2):
        try:
            os.close(descriptor)
        except OSError:
            pass

def on_term(_signal, _frame):
    child = os.fork()
    if child == 0:
        os.setsid()
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        close_stdio()
        with open("term-grandchild.pid.tmp", "w", encoding="ascii") as output:
            output.write(f"{os.getpid()} {os.getsid(0)}\n")
        os.replace("term-grandchild.pid.tmp", "term-grandchild.pid")
        while True:
            time.sleep(30)
    time.sleep(0.25)

signal.signal(signal.SIGTERM, on_term)
with open("term-parent.pid.tmp", "w", encoding="ascii") as output:
    output.write(f"{os.getpid()} {os.getsid(0)}\n")
os.replace("term-parent.pid.tmp", "term-parent.pid")
close_stdio()
while True:
    time.sleep(30)
"#,
        )
        .unwrap();
        let trajectory_path = write_trajectory(
            temporary.path(),
            vec![(
                "exec_command",
                json!({
                    "cmd": "python3 spawn-after-term.py & while [ ! -s term-parent.pid ]; do sleep 0.01; done; wait"
                }),
                json!({"content": "Process exited with code 0"}),
            )],
            default_agent(),
        );
        let trajectory = read_trajectory(&trajectory_path).unwrap();
        let mut options = replay_options(temporary.path());
        options.timeout_sec = Some(0.5);
        let replay_result = replay_trajectory(&trajectory, &options);

        let read_identity = |name: &str| -> (u32, u32) {
            let identity = fs::read_to_string(temporary.path().join(name)).unwrap();
            let fields: Vec<u32> = identity
                .split_whitespace()
                .map(|field| field.parse().unwrap())
                .collect();
            assert_eq!(fields.len(), 2);
            (fields[0], fields[1])
        };
        let (parent, parent_session) = read_identity("term-parent.pid");
        let _parent_cleanup = KillPidOnDrop(parent);
        let (grandchild, grandchild_session) = read_identity("term-grandchild.pid");
        let _grandchild_cleanup = KillPidOnDrop(grandchild);
        assert_eq!(parent_session, parent);
        assert_eq!(grandchild_session, grandchild);

        let report = replay_result.unwrap();
        assert_eq!(report["summary"]["actions_timed_out"], 1);
        for (description, pid) in [("TERM handler parent", parent), ("grandchild", grandchild)] {
            let process_path = format!("/proc/{pid}");
            for _ in 0..100 {
                if !Path::new(&process_path).exists() {
                    break;
                }
                thread::sleep(Duration::from_millis(10));
            }
            assert!(
                !Path::new(&process_path).exists(),
                "{description} {pid} survived refreshed pidfd cleanup"
            );
        }
    }

    #[test]
    fn cli_parser_accepts_repeated_patch_roots_and_equals_options() {
        let parsed = parse_cli(&[
            OsString::from("trajectory.json"),
            OsString::from("--output=report.json"),
            OsString::from("--patch-root"),
            OsString::from("one"),
            OsString::from("--patch-root=two"),
            OsString::from("--pacing=llm-gaps"),
        ])
        .unwrap()
        .unwrap();
        assert_eq!(parsed.output, Path::new("report.json"));
        assert_eq!(
            parsed.patch_roots,
            [PathBuf::from("one"), PathBuf::from("two")]
        );
        assert_eq!(parsed.pacing, Pacing::LlmGaps);
    }

    #[test]
    fn outcome_matching_accepts_early_completion_and_preserves_returncode_rules() {
        assert_eq!(status_matches("running", "running"), Some(true));
        assert_eq!(status_matches("running", "ok"), Some(false));
        assert_eq!(status_matches("running", "failed"), Some(false));
        assert_eq!(status_matches("unknown", "ok"), None);
        assert_eq!(
            outcome_matches("running", None, "running", None),
            Some(true)
        );
        assert_eq!(outcome_matches("running", None, "ok", Some(0)), Some(true));
        assert_eq!(
            outcome_matches("running", None, "failed", Some(1)),
            Some(true)
        );
        assert_eq!(
            outcome_matches("failed", Some(-1), "failed", Some(23)),
            Some(true)
        );
        assert_eq!(
            outcome_matches("failed", Some(1), "failed", Some(2)),
            Some(false)
        );
        assert_eq!(outcome_matches("unknown", None, "ok", Some(0)), None);
    }

    #[test]
    fn source_outcome_marks_claude_bash_command_not_found_failed() {
        let temporary = tempfile::tempdir().unwrap();
        let path = write_trajectory(
            temporary.path(),
            vec![(
                "Bash",
                json!({"command": "python -c \"print('ok')\""}),
                Value::String("/bin/bash: line 1: python: command not found\n".to_owned()),
            )],
            claude_code_agent(),
        );
        let trajectory = read_trajectory(&path).unwrap();
        let outcome = source_outcome(&trajectory.actions[0]);
        assert_eq!(outcome.status, "failed");
        assert_eq!(outcome.returncode, None);
    }
}
