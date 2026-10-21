#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay supported tool actions from a supported ATIF agent trajectory.

Terminal-Bench agent trajectories contain tool calls rather than a flattened
shell tape.  This module preserves the executable tool semantics required by
Codex-style ATIF trajectories: shell commands, interactive command sessions,
and workspace patches.  It also supports the provenance-gated
``mini-swe-agent-1.1`` wrapper that preserves mini-swe's ``bash`` tool shape.
Non-executing planning calls are retained in the report but do not alter the
replay environment.
"""

from __future__ import annotations

import argparse
import errno
import fcntl
import gzip
import hashlib
import json
import os
import re
import select
import signal
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, TypeVar

from ..timebase import (
    bounded_realtime_monotonic_sample,
    monotonic_clock_reference,
)
from .atif_contract import (
    CONTROLLER_ENV_VARS_MARKER,
    WORKLOAD_ENV_PREFIX,
    workload_environment_presence_variable,
    workload_environment_variable,
)
from .pacing import UNPACED
from .transcript_replay import ReplayError

REPLAY_SCHEMA_VERSION = "offline-atif-trajectory-replay.v1"
# - ``recorded``: re-inject the original wall-clock schedule (start offsets);
#   faster offline tool calls are padded with a longer sleep, so per-action
#   speedups never shrink total wall time.
# - ``unpaced``: no sleeps; pure back-to-back tool throughput.
# - ``llm-gaps``: preserve only the recorded inter-action gap (the model
#   round-trip between one tool finishing and the next starting) and run each
#   tool at its real offline speed. The recorded finish of the previous tool is
#   its issue timestamp plus the tool's recorded ``Wall time`` (read from the
#   observation), so the LLM half of each turn is held fixed while the tool half
#   is free to compress or expand under contention.
PACING_MODES = ("recorded", UNPACED, "llm-gaps")
PREVIEW_LENGTH = 1000
PROCESS_GROUP_TERMINATION_GRACE_SEC = 2.0
PROCESS_GROUP_KILL_SETTLE_SEC = 2.0
PROCESS_GROUP_POLL_SEC = 0.01
DEFAULT_WRITE_STDIN_YIELD_SEC = 5.0
_LeaderWaitResult = TypeVar("_LeaderWaitResult")
_WALL_TIME_PATTERN = re.compile(r"Wall time:\s*([0-9]+(?:\.[0-9]+)?)\s*seconds", re.IGNORECASE)
SUPPORTED_EXECUTING_TOOLS = frozenset(
    {
        "exec_command",
        "write_stdin",
        "apply_patch",
        "claude_write",
        "claude_edit",
        "claude_read",
        "claude_glob",
        "claude_grep",
        "claude_task_output",
        "claude_task_stop",
    }
)
IGNORED_METADATA_TOOLS = frozenset({"update_plan", "view_image", "web_search_call"})
CODEX_EXEC_ORIGINATOR = "codex_exec"
# These calls produce presentation or orchestration state, not task state.  The
# Code Mode wrapper is intentionally admitted only for this closed list plus
# the three existing replayable tools below.
CODEX_EXEC_METADATA_TOOLS = IGNORED_METADATA_TOOLS | frozenset(
    {
        "mcp__codex_apps__github_fetch_issue",
        "mcp__codex_apps__github_fetch_issue_comments",
        "mcp__codex_apps__github_fetch_pr",
        "mcp__codex_apps__github_fetch_pr_comments",
        "mcp__codex_apps__github_fetch_pr_patch",
        "mcp__codex_apps__github_search_issues",
        "mcp__codex_apps__github_search_prs",
        "web__run",
        "text",
        "image",
        "audio",
        "notify",
        "yield_control",
        "spawn_agent",
        "send_message",
        "followup_task",
        "wait_agent",
        "list_agents",
        "interrupt_agent",
        "wait",
        "get_goal",
        "create_goal",
        "update_goal",
    }
)


def _is_code_mode_metadata_tool(tool: str) -> bool:
    return tool in CODEX_EXEC_METADATA_TOOLS


MINI_SWE_AGENT_ORIGINAL_FORMAT = "mini-swe-agent-1.1"
MINI_SWE_TOOL_ALIASES = {"bash": "exec_command"}
CLAUDE_CODE_AGENT_NAME = "claude-code"
CLAUDE_CODE_TOOL_ALIASES = {
    "Bash": "exec_command",
    "Write": "claude_write",
    "Edit": "claude_edit",
    "Read": "claude_read",
    "Glob": "claude_glob",
    "Grep": "claude_grep",
    "TaskOutput": "claude_task_output",
    "TaskStop": "claude_task_stop",
}
# These prompts maintain agent-side planning/conversation state only; their
# later recorded executable actions already contain the resulting task work.
CLAUDE_CODE_METADATA_TOOLS = frozenset(
    {
        "TaskCreate",
        "TaskUpdate",
        "TaskList",
        "TaskGet",
        "ToolSearch",
        "ScheduleWakeup",
        "WebFetch",
        "WebSearch",
        "TodoWrite",
        "AskUserQuestion",
    }
)
CLAUDE_CODE_LIFECYCLE_TOOLS = frozenset({"TaskStop", "TaskOutput", "Monitor"})
CLAUDE_CODE_AUXILIARY_METADATA_TOOLS = frozenset({"Agent", "Monitor"})
CLAUDE_CODE_TOOLS = (
    frozenset(CLAUDE_CODE_TOOL_ALIASES)
    | CLAUDE_CODE_METADATA_TOOLS
    | CLAUDE_CODE_LIFECYCLE_TOOLS
    | CLAUDE_CODE_AUXILIARY_METADATA_TOOLS
)
_EXIT_CODE_PATTERNS = (
    re.compile(r"Process exited with code\s+(-?\d+)"),
    re.compile(r"Exit code:\s*(-?\d+)"),
)
_SESSION_ID_PATTERN = re.compile(r"Process running with session ID\s+(\d+)")
_EXEC_COMMAND_FAILURE_PATTERN = re.compile(r"\bexec_command failed\b", re.IGNORECASE)
_TOOL_ARGUMENT_PARSE_FAILURE_PATTERN = re.compile(r"\bfailed to parse function arguments\b", re.IGNORECASE)
_APPLY_PATCH_FAILURE_PATTERN = re.compile(r"\bapply_patch\b[^\n]*\b(?:failed|failure|rejected)\b", re.IGNORECASE)


def _workload_environment() -> dict[str, str] | None:
    """Restore task-image values shadowed by the replay controller."""
    controller_variables = os.environ.get(CONTROLLER_ENV_VARS_MARKER)
    if not controller_variables:
        return None
    environment = os.environ.copy()
    environment.pop(CONTROLLER_ENV_VARS_MARKER, None)
    for name in controller_variables.split(":"):
        original_name = workload_environment_variable(name)
        presence_name = workload_environment_presence_variable(name)
        presence_marker = environment.pop(presence_name, None)
        had_preserved_value = original_name in environment
        original_value = environment.pop(original_name, None)
        if presence_marker is None:
            # Backward compatibility for launchers that predate the explicit
            # presence marker and could not distinguish unset from empty.
            was_present = had_preserved_value and bool(original_value)
        else:
            was_present = presence_marker == "1"
        if was_present:
            environment[name] = original_value or ""
        else:
            environment.pop(name, None)
    for name in tuple(environment):
        if name.startswith(WORKLOAD_ENV_PREFIX):
            environment.pop(name, None)
    return environment


@dataclass(frozen=True)
class AtifAction:
    index: int
    step_id: Any
    timestamp_sec: float
    tool: str
    source_tool: str
    source_agent_original_format: str | None
    arguments: dict[str, Any]
    observation: Any
    observation_text: str
    code_mode: bool = False
    parallel_group_id: str | None = None
    source_trajectory_id: str = "root"


@dataclass(frozen=True)
class AtifTrajectory:
    source_path: Path
    schema_version: str
    agent: dict[str, Any]
    source_agent_original_format: str | None
    session_id: str | None
    actions: list[AtifAction]
    subagent_trajectories_flattened: int = 0
    contains_codex_exec: bool = False

    @property
    def trial_name(self) -> str:
        return self.source_path.parent.parent.name

    @property
    def task_name(self) -> str:
        return self.trial_name.rsplit("__", 1)[0]


@dataclass
class _RunningSession:
    source_id: str
    process: subprocess.Popen[str]
    log_path: Path
    log_handle: Any
    terminal: _PtyOutput | None = None


class _PtyOutput:
    """Drain a terminal even between recorded polls, when its child may write."""

    def __init__(self, master: int, output: Any):
        self.master = master
        self.output = output
        self.stopped = threading.Event()
        self.error: OSError | None = None
        self.reader = threading.Thread(target=self._drain, daemon=True)
        self.reader.start()

    def _drain(self) -> None:
        try:
            while not self.stopped.is_set():
                ready, _, _ = select.select([self.master], [], [], 0.05)
                if not ready:
                    if self.stopped.is_set():
                        return
                    continue
                try:
                    data = os.read(self.master, 65536)
                except BlockingIOError:
                    continue
                except OSError as exc:
                    if exc.errno == errno.EIO:  # Linux PTY slave has closed.
                        return
                    raise
                if not data:
                    return
                if self.error is None:
                    try:
                        while data:
                            data = data[os.write(self.output.fileno(), data) :]
                    except OSError as exc:
                        # Keep draining after a full/failed sink so the child
                        # can exit; report the capture failure at completion.
                        self.error = exc
        except OSError as exc:
            if self.error is None:
                self.error = exc

    def snapshot(self) -> str:
        # A live background child may retain the slave after its shell exits.
        # Do not close its terminal or wait indefinitely for terminal EOF.
        self.reader.join(timeout=0.05)
        if self.error is not None:
            raise ReplayError(f"PTY output capture failed: {self.error}")
        return os.pread(self.output.fileno(), os.fstat(self.output.fileno()).st_size, 0).decode(
            "utf-8", errors="replace"
        )

    def close(self) -> None:
        if self.stopped.is_set():
            return
        self.reader.join(timeout=0.05)
        self.stopped.set()
        self.reader.join()
        os.close(self.master)
        if self.error is not None:
            raise ReplayError(f"PTY output capture failed: {self.error}")


@dataclass(frozen=True)
class _PatchOperation:
    kind: str
    path: str
    lines: tuple[str, ...] = ()
    move_to: str | None = None


def _timestamp_sec(value: Any, *, action_index: int) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if isinstance(value, str):
        try:
            return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()
        except ValueError as exc:
            raise ReplayError(f"ATIF action {action_index} has invalid timestamp {value!r}") from exc
    raise ReplayError(f"ATIF action {action_index} has no valid timestamp")


def _text_content(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for key in ("content", "text", "output"):
            if isinstance(value.get(key), str):
                return value[key]
    if isinstance(value, list):
        return "\n".join(_text_content(item) for item in value)
    return json.dumps(value, sort_keys=True) if value is not None else ""


def _source_agent_original_format(agent: dict[str, Any]) -> str | None:
    extra = agent.get("extra")
    if not isinstance(extra, dict):
        return None
    original_format = extra.get("original_format")
    return original_format if isinstance(original_format, str) and original_format else None


def _codex_exec_originator(agent: dict[str, Any]) -> bool:
    extra = agent.get("extra")
    return agent.get("name") == "codex" and isinstance(extra, dict) and extra.get("originator") == CODEX_EXEC_ORIGINATOR


def _claude_code_agent(agent: dict[str, Any]) -> bool:
    return agent.get("name") == CLAUDE_CODE_AGENT_NAME


def _is_code_mode_action(action: AtifAction) -> bool:
    return action.code_mode


def _is_metadata_action(action: AtifAction) -> bool:
    return (
        action.tool in IGNORED_METADATA_TOOLS
        or (_is_code_mode_action(action) and _is_code_mode_metadata_tool(action.tool))
        or (
            action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME
            and action.tool in CLAUDE_CODE_METADATA_TOOLS | CLAUDE_CODE_AUXILIARY_METADATA_TOOLS
        )
    )


class _CodeModeParser:
    """Strict sequential parser for the literal Code Mode replay contract."""

    def __init__(self, source: str) -> None:
        if source.startswith("// @exec:"):
            _, _, source = source.partition("\n")
        self.source = source
        self.at = 0
        # A deliberately tiny data environment, not a JavaScript interpreter.
        # It holds literal ``const`` values and bindings for closed literal
        # array ``.map`` batches emitted by Codex Code Mode.
        self.values: dict[str, Any] = {}
        self.data_literal = False
        self.literal_depth = 0
        self.result_prints: list[str | None] = []
        self.direct_prints: list[int | None] | None = []
        self.indexed_results = False

    def error(self, message: str) -> ReplayError:
        return ReplayError(f"Codex exec contains {message}")

    def skip_space(self) -> None:
        while self.at < len(self.source) and self.source[self.at].isspace():
            self.at += 1

    def eat(self, value: str) -> bool:
        if self.source.startswith(value, self.at):
            self.at += len(value)
            return True
        return False

    def expect(self, value: str, message: str) -> None:
        self.skip_space()
        if not self.eat(value):
            raise self.error(message)

    def identifier(self) -> str | None:
        match = re.match(r"[A-Za-z_$][A-Za-z0-9_$]*", self.source[self.at :])
        if match is None:
            return None
        self.at += len(match.group())
        return match.group()

    def string(self) -> str:
        if self.at == len(self.source) or self.source[self.at] not in "'\"`":
            raise self.error("a non-string literal")
        quote = self.source[self.at]
        self.at += 1
        value: list[str] = []
        escapes = {
            "n": "\n",
            "r": "\r",
            "t": "\t",
            "b": "\b",
            "f": "\f",
            "v": "\v",
            "0": "\0",
            "\\": "\\",
            "/": "/",
            "'": "'",
            "\"": "\"",
            "`": "`",
        }
        while self.at < len(self.source):
            character = self.source[self.at]
            self.at += 1
            if character == quote:
                return "".join(value)
            if quote == "`" and character == "$" and self.source.startswith("{", self.at):
                raise self.error("template interpolation is not supported")
            if character != "\\":
                value.append(character)
                continue
            if self.at == len(self.source):
                break
            escaped = self.source[self.at]
            self.at += 1
            if escaped in {"x", "u"} or (self.data_literal and escaped == "U"):
                width = {"x": 2, "u": 4, "U": 8}[escaped]
                digits = self.source[self.at : self.at + width]
                if len(digits) != width:
                    raise self.error("an invalid string escape")
                try:
                    value.append(chr(int(digits, 16)))
                except ValueError as exc:
                    raise self.error("an invalid string escape") from exc
                self.at += width
            elif escaped in escapes:
                value.append(escapes[escaped])
            else:
                raise self.error("an unsupported string escape")
        raise self.error("an unterminated string literal")

    def literal(self) -> Any:
        self.literal_depth += 1
        try:
            if self.literal_depth > 128:
                raise self.error("literal nesting exceeds limit")
            return self._literal()
        finally:
            self.literal_depth -= 1

    def _literal(self) -> Any:
        self.skip_space()
        if self.at == len(self.source):
            raise self.error("a missing tool argument")
        character = self.source[self.at]
        if character in "'\"`":
            return self.string()
        if character == "{":
            return self.object()
        if character == "[":
            return self.array()
        if character in "-0123456789":
            start = self.at
            while self.at < len(self.source) and self.source[self.at] in "-+.eE0123456789":
                self.at += 1
            try:
                return json.loads(self.source[start : self.at])
            except json.JSONDecodeError as exc:
                raise self.error("an invalid number literal") from exc
        word = self.identifier()
        if word in ("true", "false", "null"):
            return json.loads(word)
        if self.data_literal and word in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[word]
        if word in self.values:
            return self.values[word]
        raise self.error("a dynamic tool argument")

    def object(self) -> dict[str, Any]:
        self.at += 1
        result: dict[str, Any] = {}
        while True:
            self.skip_space()
            if self.eat("}"):
                return result
            if self.at < len(self.source) and self.source[self.at] in "'\"":
                key = self.string()
            else:
                key = self.identifier()
            if key is None:
                raise self.error("a non-literal object key")
            if self.data_literal and key in result:
                raise self.error("duplicate result field")
            self.skip_space()
            if self.eat(":"):
                result[key] = self.literal()
            elif key in self.values:
                # JavaScript object shorthand: ``{cmd, workdir}``.
                result[key] = self.values[key]
            else:
                raise self.error("an object literal missing ':'")
            self.skip_space()
            if self.eat("}"):
                return result
            self.expect(",", "an object literal missing ','")

    def array(self) -> list[Any]:
        self.at += 1
        result: list[Any] = []
        while True:
            self.skip_space()
            if self.eat("]"):
                return result
            result.append(self.literal())
            self.skip_space()
            if self.eat("]"):
                return result
            self.expect(",", "an array literal missing ','")

    def semicolon(self) -> None:
        self.skip_space()
        if not self.eat(";") and self.at != len(self.source):
            raise self.error("a statement separator")

    def ignored_empty_catch(self) -> bool:
        self.skip_space()
        if not self.eat(".catch"):
            return False
        for token, error in (
            ("(", "catch missing '('"),
            ("(", "catch handler parameters"),
            (")", "catch handler parameters"),
            ("=>", "catch handler arrow"),
            ("{", "catch handler body"),
            ("}", "catch handler body"),
            (")", "catch missing ')'"),
        ):
            self.skip_space()
            self.expect(token, error)
        return True

    def output(self) -> bool:
        start = self.at
        name = self.identifier()
        self.skip_space()
        if name not in {"text", "image", "audio"} or not self.eat("("):
            self.at = start
            return False
        contents = self.at
        depth = 1
        quote = ""
        while self.at < len(self.source):
            character = self.source[self.at]
            self.at += 1
            if quote:
                if character == "\\":
                    self.at += 1
                elif character == quote:
                    quote = ""
            elif character in "'\"`":
                quote = character
            elif character == "(":
                depth += 1
            elif character == ")":
                depth -= 1
                if not depth:
                    if "tools." in self.source[contents : self.at - 1]:
                        raise self.error("a nested tools call in an output helper")
                    self.semicolon()
                    return True
        raise self.error("an unterminated output helper")

    def _output_conditional(self) -> bool:
        """Accept an output-only conditional after a recorded tool call."""
        start = self.at
        if not self.eat("if"):
            return False
        self.skip_space()
        condition = self._consume_balanced("(", ")")
        if "tools." in condition:
            raise self.error("a tools call in an output conditional")
        self.skip_space()
        if self.at < len(self.source) and self.source[self.at] == "{":
            body = self._consume_balanced("{", "}")
            if "tools." not in body and "text" in body:
                self.semicolon()
                return True
        elif self.output():
            return True
        self.at = start
        return False

    def presentation_loop(self, bound_name: str) -> bool:
        if self._presentation_loop_variant(bound_name):
            return True
        if not self.eat("for"):
            return False
        self.skip_space()
        self.expect("(", "an invalid Promise.all presentation loop")
        start = self.at
        while self.at < len(self.source) and self.source[self.at] != ")":
            self.at += 1
        binding = "".join(self.source[start : self.at].split())
        parts = binding.split(";")
        declaration = parts[0][len("let") :] if parts and parts[0].startswith("let") else ""
        index, separator, initial = declaration.partition("=")
        self.expect(")", "an unterminated Promise.all presentation loop")
        if (
            not index
            or not (index[0].isascii() and (index[0].isalpha() or index[0] in "_$"))
            or not all(character.isascii() and (character.isalnum() or character in "_$") for character in index)
            or len(parts) != 3
            or not separator
            or initial != "0"
            or parts[1] != f"{index}<{bound_name}.length"
            or parts[2] != f"{index}++"
        ):
            raise self.error("an invalid Promise.all presentation loop")
        self.skip_space()
        braced = self.eat("{")
        self.skip_space()
        start = self.at
        if not self.output():
            raise self.error("Promise.all presentation loop must contain only text calls")
        body = "".join(self.source[start : self.at].split())
        reference = f"{bound_name}[{index}]"
        self.indexed_results = index == "i" and body in {
            f"text({{i,...{reference}}});",
            f"text(JSON.stringify({{i,...{reference}}}));",
        }
        if (
            reference not in body
            or body.startswith('text("')
            or body.startswith("text('")
            or "tools." in body
            or "text" not in body
        ):
            raise self.error("an invalid Promise.all presentation loop")
        if braced:
            self.expect("}", "Promise.all presentation loop must contain only text calls")
        self.semicolon()
        return True

    def tool_call(self) -> tuple[str, dict[str, Any]]:
        self.expect("tools.", "a tools call")
        tool = self.identifier()
        if tool is None:
            raise self.error("a tool call missing '('")
        self.expect("(", "a tool call missing '('")
        value = self.literal()
        self.expect(")", "a tool call missing ')'")
        if tool not in SUPPORTED_EXECUTING_TOOLS and not _is_code_mode_metadata_tool(tool):
            raise self.error(f"unsupported tool call: {tool}")
        if tool == "apply_patch":
            patch = value.get("input", value.get("patch")) if isinstance(value, dict) else value
            if not isinstance(patch, str):
                raise self.error("apply_patch arguments must be a string or object with string input")
            value = {"input": patch}
        if not isinstance(value, dict):
            raise self.error("tool arguments must be a literal object")
        return tool, value

    def _consume_balanced(self, opening: str, closing: str) -> str:
        """Consume one balanced source fragment without evaluating it."""
        if self.at == len(self.source) or self.source[self.at] != opening:
            raise self.error(f"expected {opening!r}")
        start = self.at
        depth = 0
        quote = ""
        while self.at < len(self.source):
            character = self.source[self.at]
            self.at += 1
            if quote:
                if character == "\\":
                    self.at += 1
                elif character == quote:
                    quote = ""
            elif character in "'\"`":
                quote = character
            elif character == opening:
                depth += 1
            elif character == closing:
                depth -= 1
                if not depth:
                    return self.source[start : self.at]
        raise self.error(f"an unterminated {opening}{closing} fragment")

    def _validate_mapped_callback(self, prefix: str, suffix: str, names: list[str]) -> None:
        captured = re.fullmatch(
            r"\s*(?:(?:return[ \t]+)?(?:await\s+)?|(?:const|let)\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:await\s+)?)",
            prefix,
        )
        if captured is None:
            raise self.error("command batch callback contains unsupported work before its tool call")
        result = captured.group(1)
        if result in names:
            raise self.error("command batch result shadows a callback binding")
        references = [re.escape(name) for name in names]
        if result:
            references.append(rf"{re.escape(result)}(?:\s*\.\s*output)?")
        reference = "(?:" + "|".join(references) + ")"
        quoted = r"(?:\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*')"
        template = rf"`(?:[^`$\\]|\\.|\$\{{\s*{reference}\s*\}})*`"
        value = rf"(?:{quoted}|{template}|{reference}|true|false|null|-?[0-9]+(?:\.[0-9]+)?)"
        prop = rf"(?:[A-Za-z_$][A-Za-z0-9_$]*|{quoted})\s*:\s*{value}|{reference}"
        obj = rf"\{{\s*(?:(?:{prop})\s*(?:,\s*(?:{prop})\s*)*,?)?\}}"
        if re.fullmatch(rf"\s*;?\s*(?:return[ \t]+(?:{value}|{obj})\s*;?\s*)?", suffix) is None or (
            prefix.lstrip().startswith("return") and suffix.strip().rstrip(";")
        ):
            raise self.error("command batch callback contains unsupported work after its tool call")

    def _map_promise_all(self, group_id: str) -> list[tuple[str, dict[str, Any], str]]:
        """Parse a closed literal-array ``.map`` Promise.all batch.

        This supports recorded data substitution only.  The callback may contain
        one supported tool call plus result-display code; it cannot branch,
        create timers, or issue another tool call.
        """
        array_name = self.identifier()
        items = self.values.get(array_name or "")
        if not isinstance(items, list):
            raise self.error("a dynamic Promise.all input")
        self.expect(".map", "Promise.all input must be an array literal or literal array map")
        self.expect("(", "an invalid Promise.all map")
        self.skip_space()
        if self.source.startswith("async", self.at):
            self.at += len("async")
            self.skip_space()
        if self.eat("("):
            self.skip_space()
            if self.eat("["):
                names: list[str] = []
                while True:
                    self.skip_space()
                    name = self.identifier()
                    if name is None:
                        raise self.error("an invalid Promise.all map binding")
                    names.append(name)
                    self.skip_space()
                    if self.eat("]"):
                        break
                    self.expect(",", "an invalid Promise.all map binding")
            else:
                name = self.identifier()
                if name is None:
                    raise self.error("an invalid Promise.all map binding")
                names = [name]
            self.expect(")", "an invalid Promise.all map binding")
        else:
            name = self.identifier()
            if name is None:
                raise self.error("an invalid Promise.all map binding")
            names = [name]
        self.expect("=>", "an invalid Promise.all map")
        self.skip_space()
        braced = self.at < len(self.source) and self.source[self.at] == "{"
        direct_start = self.at
        if braced:
            body = self._consume_balanced("{", "}")
            if len(re.findall(r"tools\.", body)) != 1:
                raise self.error("Promise.all map must contain exactly one unconditional tools call")
            marker = body.find("tools.")
        else:
            # The concise ``items.map(x => tools.exec_command({...}))`` form
            # has no callback block.  Parse the one call directly from the
            # original source instead of trying to find its nested ')'.
            if not self.source.startswith("tools.", direct_start):
                raise self.error("Promise.all map must contain an unconditional tools call")
            body = ""
            marker = 0
        actions: list[tuple[str, dict[str, Any], str]] = []
        original_values = self.values.copy()
        try:
            for item in items:
                if len(names) == 1:
                    bound = [item]
                elif isinstance(item, list) and len(item) == len(names):
                    bound = item
                else:
                    raise self.error("Promise.all map item does not match its binding")
                self.values = original_values | dict(zip(names, bound, strict=True))
                # Parse the tool-call expression from the isolated callback
                # fragment, so arbitrary presentation text remains inert.
                saved_source, saved_at = self.source, self.at
                self.source, self.at = (body[marker:], 0) if braced else (saved_source, direct_start)
                try:
                    tool, arguments = self.tool_call()
                    direct_end = self.at
                    if braced:
                        self._validate_mapped_callback(body[1:marker], body[marker + direct_end : -1], names)
                finally:
                    self.source, self.at = saved_source, saved_at
                if tool not in SUPPORTED_EXECUTING_TOOLS:
                    raise self.error(f"Promise.all only supports stateful tool calls: {tool}")
                actions.append((tool, arguments, group_id))
        finally:
            self.values = original_values
        if not braced:
            self.at = direct_end
        self.expect(")", "an invalid Promise.all map")
        return actions

    def _presentation_loop_variant(self, bound_name: str) -> bool:
        """Accept pure result-printing variants used by recorded Code Mode."""
        if self.source.startswith(f"{bound_name}.forEach", self.at):
            self.at += len(f"{bound_name}.forEach")
            self.skip_space()
            body = self._consume_balanced("(", ")")
            self.indexed_results = "".join(body.split()) in {
                "((r,i)=>{text(JSON.stringify({i,...r}));})",
                "((r,i)=>{text({i,...r});})",
                "((r,i)=>text(JSON.stringify({i,...r})))",
                "((r,i)=>text({i,...r}))",
            }
            if "tools." in body or "text" not in body:
                raise self.error("Promise.all presentation loop must contain only text calls")
            self.semicolon()
            return True
        start = self.at
        if self.source.startswith("for", self.at):
            self.at += len("for")
            self.skip_space()
            header = self._consume_balanced("(", ")")
            if re.fullmatch(r"\(const\s+[A-Za-z_$][A-Za-z0-9_$]*\s+of\s+" + re.escape(bound_name) + r"\)", header):
                self.skip_space()
                if self.at < len(self.source) and self.source[self.at] == "{":
                    body = self._consume_balanced("{", "}")
                    if "tools." in body or "text" not in body:
                        raise self.error("Promise.all presentation loop must contain only text calls")
                    self.semicolon()
                    return True
                if not self.output():
                    raise self.error("Promise.all presentation loop must contain only text calls")
                return True
        if self.output():
            return True
        self.at = start
        return False

    def promise_all(self, group_id: str, settled: bool) -> list[tuple[str, dict[str, Any], str]]:
        self.expect("Promise.allSettled" if settled else "Promise.all", "control flow or parallel work")
        self.expect("(", "Promise.all missing '('")
        self.skip_space()
        if self.at < len(self.source) and self.source[self.at] != "[":
            actions = self._map_promise_all(group_id)
            self.expect(")", "Promise.all missing ')'")
            return actions
        self.expect("[", "Promise.all arguments must be an array literal")
        actions: list[tuple[str, dict[str, Any], str]] = []
        while True:
            self.skip_space()
            if self.eat("]"):
                break
            tool, arguments = self.tool_call()
            if tool not in SUPPORTED_EXECUTING_TOOLS and not (settled and _is_code_mode_metadata_tool(tool)):
                raise self.error(f"Promise.all only supports stateful tool calls: {tool}")
            actions.append((tool, arguments, group_id))
            self.skip_space()
            if self.eat("]"):
                break
            self.expect(",", "Promise.all members must be comma-separated")
        if not actions:
            raise self.error("Promise.all must contain a stateful tool call")
        self.expect(")", "Promise.all missing ')'")
        return actions

    def parse(self, outer_call_identity: str) -> list[tuple[str, dict[str, Any], str | None]]:
        actions: list[tuple[str, dict[str, Any], str | None]] = []
        promise_ordinal = 0
        direct_binding: str | None = None
        while True:
            self.skip_space()
            if self.at == len(self.source):
                return actions
            wrapper = self.source.startswith("text(await ", self.at)
            if wrapper:
                self.at += len("text(")
            start = self.at
            if not wrapper and self.output():
                display = "".join(self.source[start : self.at].split())
                full_result = bool(direct_binding) and display.rstrip(";") in {
                    f"text({direct_binding})",
                    f"text(JSON.stringify({direct_binding}))",
                }
                if full_result:
                    self.result_prints[-1] = "direct"
                if full_result or _data_only_print(self.source[start : self.at].strip()):
                    if self.direct_prints is not None:
                        self.direct_prints.append(len(actions) - 1 if full_result else None)
                else:
                    self.direct_prints = None
                continue
            if not wrapper and self._output_conditional():
                self.direct_prints = None
                continue
            metadata_loop = re.match(
                r'const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*'
                r'\[\s*"(?:\\.|[^"\\])*"(?:\s*,\s*"(?:\\.|[^"\\])*")+\s*\]\s*;\s*'
                r'for\s*\(\s*const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+of\s+\1\s*\)\s*\{\s*'
                r'const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*await\s+tools\.'
                r'(mcp__codex_apps__github_(?:fetch|search|list)_[A-Za-z0-9_]+)'
                r'\(\s*(\{[^();]*\})\s*\)\s*;\s*text\(\s*\2\s*\)\s*;\s*'
                r'text\(\s*\3\s*\)\s*;\s*\}',
                self.source[self.at :],
            )
            if metadata_loop is not None:
                if not _is_code_mode_metadata_tool(metadata_loop.group(4)) or not re.search(
                    rf"\b{re.escape(metadata_loop.group(2))}\b", metadata_loop.group(5)
                ):
                    raise self.error("a dynamic metadata query loop")
                self.direct_prints = None
                self.at += len(metadata_loop.group())
                continue
            listing_loop = re.match(
                r'for\s*\(\s*const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s+of\s+'
                r'(\[\s*"[A-Za-z0-9_.\-]+"(?:\s*,\s*"[A-Za-z0-9_.\-]+")*\s*\])\s*\)\s*\{\s*'
                r'const\s+([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*ALL_TOOLS\.find\('
                r'([A-Za-z_$][A-Za-z0-9_$]*)\s*=>\s*\4\.name\s*===\s*\1\s*\)\s*;\s*'
                r'text\(\s*\3\s*\)\s*;\s*\}',
                self.source[self.at :],
            )
            if listing_loop is not None:
                self.direct_prints = None
                self.at += len(listing_loop.group())
                continue
            destructured = re.match(
                r"let\s*\{\s*output\s*,\s*\.\.\.rest\s*\}\s*=\s*",
                self.source[self.at :],
            )
            if destructured is not None:
                self.at += len(destructured.group())
                first = None
            else:
                first = self.identifier()
            bound_name: str | None = None
            if first in {"const", "let"}:
                self.skip_space()
                name = self.identifier()
                self.skip_space()
                if name is None or not self.eat("="):
                    raise self.error("an invalid const binding")
                self.skip_space()
                if not self.source.startswith("await", self.at):
                    direct_binding = None
                    # Tool-catalog filtering is display-only agent plumbing;
                    # the later recorded tool calls contain all task state.
                    if self.source.startswith("ALL_TOOLS.filter", self.at):
                        end = self.source.find(";", self.at)
                        listing = re.sub(r"\s+", "", self.source[self.at : end])
                        parameter = r"([A-Za-z_$][A-Za-z0-9_$]*)"
                        safe_pattern = r"[A-Za-z0-9.|*_()\-]+"
                        simple = re.fullmatch(rf"ALL_TOOLS\.filter\({parameter}=>\1\.name\)", listing)
                        inspected = (
                            r'(?:\1\.name|\1\.description|\1\.name\+\"\"\+\1\.description|'
                            r'\(\1\.name\+\"\"\+\1\.description\)\.toLowerCase\(\))'
                        )
                        regex_test = rf'/{safe_pattern}/(?:i)?\.test\({inspected}\)'
                        regex_filter = re.fullmatch(
                            rf'ALL_TOOLS\.filter\({parameter}=>({regex_test}(?:&&{regex_test})*)\)(?:\.slice\([0-9]+,[0-9]+\)|\.map\(\1=>\1\.name\+\"::\"\+\1\.description\.slice\(0,[0-9]+\)\)|\.map\(\1=>\(\{{name:\1\.name,description:\1\.description\.slice\(0,[0-9]+\)\}}\)\)\.slice\(0,[0-9]+\))?',
                            listing,
                        )
                        prefixed = re.fullmatch(
                            rf'ALL_TOOLS\.filter\({parameter}=>\1\.name\.startsWith\("[A-Za-z0-9_.\-]+"\)&&/{safe_pattern}/(?:i)?\.test\(\1\.name\)\)\.map\(\1=>\1\.name\)',
                            listing,
                        )
                        if end < 0 or not (simple or regex_filter or prefixed):
                            raise self.error("a dynamic const binding")
                        self.at = end + 1
                        self.values[name] = []
                        continue
                    if self.at == len(self.source) or self.source[self.at] not in "'\"`[{":
                        raise self.error("a dynamic const binding")
                    self.values[name] = self.literal()
                    self.semicolon()
                    continue
                bound_name = name
            elif destructured is None:
                self.at = start
            if not self.eat("await"):
                raise self.error("control flow or parallel work")
            self.skip_space()
            if self.source.startswith("Promise.all", self.at):
                self.direct_prints = None
                promise_ordinal += 1
                settled = self.source.startswith("Promise.allSettled", self.at)
                kind = "promise_all_settled" if settled else "promise_all"
                group_id = f"{outer_call_identity}:{kind}:{promise_ordinal}"
                batch_start = len(actions)
                actions.extend(self.promise_all(group_id, settled))
                self.result_prints.extend([None] * (len(actions) - batch_start))
                self.semicolon()
                if wrapper:
                    raise self.error("Promise.all cannot be wrapped by an output helper")
                if bound_name is not None:
                    self.skip_space()
                    self.indexed_results = False
                    if not self.presentation_loop(bound_name):
                        raise self.error("Promise.all result binding requires its presentation loop")
                    if self.indexed_results:
                        self.result_prints[batch_start:] = ["settled" if settled else "indexed"] * (
                            len(actions) - batch_start
                        )
                direct_binding = None
                continue
            tool, value = self.tool_call()
            if self.ignored_empty_catch() and not _is_code_mode_metadata_tool(tool):
                raise self.error("catch around an executable tool")
            if wrapper:
                self.expect(")", "an output helper missing ')'")
                if self.direct_prints is not None:
                    self.direct_prints.append(len(actions))
            self.semicolon()
            actions.append((tool, value, None))
            self.result_prints.append("direct" if wrapper else None)
            direct_binding = bound_name


def _data_only_print(statement: str) -> bool:
    statement = statement.rstrip(";")
    if not statement.startswith("text(") or not statement.endswith(")"):
        return False
    expression = statement[5:-1].strip()
    reader = _CodeModeParser(expression)
    try:
        reader.literal()
        reader.skip_space()
        return reader.at == len(reader.source)
    except ReplayError:
        pass
    if not expression.startswith("ALL_TOOLS.filter"):
        return False
    reader = _CodeModeParser(f"const listing = {expression};")
    try:
        return not reader.parse("")
    except ReplayError:
        return False


def _codex_exec_actions(source: str, outer_call_identity: str) -> list[tuple[str, dict[str, Any], str | None]]:
    return _CodeModeParser(source).parse(outer_call_identity)


def _code_mode_result_payloads(observation: Any) -> list[Any]:
    """Decode retained printouts as data; never execute JavaScript or Python."""
    content = observation.get("content", observation) if isinstance(observation, dict) else observation
    if isinstance(content, str):
        content = content.strip()
        if not content.startswith(("[", "{")):
            return []
        reader = _CodeModeParser(content)
        reader.data_literal = True
        content = reader.literal()
        reader.skip_space()
        if reader.at != len(reader.source):
            raise ReplayError("Codex result has trailing data")
    if not isinstance(content, list):
        return [content] if isinstance(content, dict) else []
    payloads = []
    for index, block in enumerate(content):
        if not isinstance(block, dict) or block.get("type") not in {"input_text", "text"}:
            return []
        text = block.get("text")
        if not isinstance(text, str):
            return []
        if index == 0 and text.startswith("Script completed\n") and text.endswith("Output:\n"):
            continue
        try:
            payloads.append(json.loads(text))
        except json.JSONDecodeError:
            # Keep the print position; never shift another call's result into it.
            payloads.append(text)
    return payloads


def _printed_tool_outcome(payload: Any, settled: bool) -> Any:
    invalid = ReplayError("Codex printed tool result has contradictory or invalid fields")
    if settled:
        if not isinstance(payload, dict):
            raise invalid
        if payload.get("status") == "fulfilled" and "value" in payload and "reason" not in payload:
            payload = payload["value"]
        elif payload.get("status") == "rejected" and "reason" in payload and "value" not in payload:
            # Replay known command text, but rejection does not prove a shell outcome.
            return {"status": "unknown", "returncode": None, "session_id": None, "reason": payload["reason"]}
        else:
            raise invalid
    if not isinstance(payload, dict):
        return None
    code, session = payload.get("exit_code"), payload.get("session_id")
    if code is not None and (type(code) is not int or not -(2**63) <= code < 2**63 or session is not None):
        raise invalid
    if session is not None:
        if type(session) is int and 0 <= session < 2**64:
            session = str(session)
        elif not isinstance(session, str) or not session:
            raise invalid
    if code is None and session is None:
        return None
    if "output" in payload and not isinstance(payload["output"], str):
        raise invalid
    return {
        "status": "running" if code is None else ("ok" if code == 0 else "failed"),
        "returncode": code,
        "session_id": session,
        "output": payload.get("output", ""),
    }


def _trajectory_code_mode_invocations(
    source: str, identity: str, observation: Any
) -> list[tuple[str, dict[str, Any], str | None, Any]]:
    parser = _CodeModeParser(source)
    actions = parser.parse(identity)
    outcomes: list[Any] = [None] * len(actions)
    print_kind = parser.result_prints[0] if actions else None
    indexed = print_kind in {"indexed", "settled"}
    bindable = print_kind is not None and all(p == print_kind for p in parser.result_prints)
    bindable = bindable and (not indexed or all(a[2] == actions[0][2] for a in actions))
    if bindable and (indexed or parser.direct_prints is not None):
        payloads = _code_mode_result_payloads(observation)
        if payloads:
            expected = len(actions) if indexed else len(parser.direct_prints)
            if len(payloads) != expected:
                raise ReplayError("Codex printed result count does not match nested calls")
            seen = set()
            for position, payload in enumerate(payloads):
                index = payload.get("i") if indexed and isinstance(payload, dict) else position
                if not indexed:
                    index = parser.direct_prints[position]
                    if index is None:
                        continue
                if indexed and (
                    not isinstance(payload, dict) or type(index) is not int or not 0 <= index < len(actions)
                ):
                    raise ReplayError("Codex printed result has invalid index")
                if index in seen:
                    raise ReplayError("Codex printed results repeat an index")
                seen.add(index)
                outcomes[index] = _printed_tool_outcome(payload, print_kind == "settled")
    if (
        len(actions) == 1
        and actions[0][2] is None
        and outcomes[0] is None
        and isinstance(observation, dict)
        and isinstance(observation.get("status"), str)
        and ("returncode" in observation or "session_id" in observation)
    ):
        outcomes[0] = observation
    return [(*action, outcome) for action, outcome in zip(actions, outcomes, strict=True)]


def _canonical_tool(tool: str, *, source_agent_original_format: str | None) -> str:
    """Normalize known wrapper aliases without weakening generic ATIF validation."""
    if source_agent_original_format == MINI_SWE_AGENT_ORIGINAL_FORMAT:
        return MINI_SWE_TOOL_ALIASES.get(tool, tool)
    if source_agent_original_format == CLAUDE_CODE_AGENT_NAME:
        return CLAUDE_CODE_TOOL_ALIASES.get(tool, tool)
    return tool


def _canonical_arguments(
    arguments: dict[str, Any],
    *,
    source_tool: str,
    source_agent_original_format: str | None,
) -> dict[str, Any]:
    """Preserve mini-swe's raw payload while exposing canonical shell fields."""
    if source_agent_original_format != MINI_SWE_AGENT_ORIGINAL_FORMAT or source_tool != "bash":
        if source_agent_original_format != CLAUDE_CODE_AGENT_NAME or source_tool != "Bash":
            return arguments
        if arguments.get("dangerouslyDisableSandbox") is True:
            raise ReplayError("Claude Code dangerouslyDisableSandbox Bash is unsupported")
        command = arguments.get("command")
        return arguments if "cmd" in arguments or not isinstance(command, str) else {**arguments, "cmd": command}
    command = arguments.get("command")
    if "cmd" in arguments or not isinstance(command, str):
        return arguments
    normalized = dict(arguments)
    normalized["cmd"] = command
    return normalized


def _read_trajectory_text(path: Path) -> str:
    """Read a trajectory document, transparently handling gzip.

    Large trajectories may be stored gzipped (``trajectory.json.gz``) to stay
    under hosting file-size limits. A caller that passes the plain ``.json``
    path still works: if it is absent we fall back to the ``.gz`` sibling. The
    on-disk form is detected by gzip magic bytes, so extension and content can
    disagree without breaking the load.
    """
    if not path.is_file():
        gz = path.with_name(path.name + ".gz")
        if gz.is_file():
            path = gz
        else:
            raise ReplayError(f"ATIF trajectory does not exist: {path}")
    raw = path.read_bytes()
    if raw[:2] == b"\x1f\x8b":  # gzip magic
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="replace")


def _call_observation(step: dict[str, Any], call: dict[str, Any]) -> Any:
    """Only the exactly bound per-call result is an executable action's outcome."""
    call_id = call.get("tool_call_id")
    results = (step.get("observation") or {}).get("results")
    if not isinstance(call_id, str) or not call_id or not isinstance(results, list):
        raise ReplayError("ATIF tool call has no exact observation binding")
    if sum(isinstance(item, dict) and item.get("tool_call_id") == call_id for item in step.get("tool_calls", [])) != 1:
        raise ReplayError("ATIF tool call identity is duplicated")
    matching = [result for result in results if isinstance(result, dict) and result.get("source_call_id") == call_id]
    if len(matching) != 1:
        raise ReplayError(f"ATIF tool call {call_id!r} has no exact observation binding")
    return matching[0]


def _mini_swe_call_observation(step: dict[str, Any], call: dict[str, Any]) -> Any:
    # ATIF-v1.2 singleton captures may omit result IDs; multiple calls never may.
    observation = step.get("observation")
    results = observation.get("results") if isinstance(observation, dict) else None
    if (
        len(step["tool_calls"]) == 1
        and isinstance(results, list)
        and len(results) == 1
        and isinstance(results[0], dict)
        and results[0].get("source_call_id") is None
    ):
        return results[0]
    return _call_observation(step, call)


def _claude_background_task_id(observation: Any) -> str | None:
    metadata = (observation.get("extra") or {}).get("tool_result_metadata") if isinstance(observation, dict) else None
    result = metadata.get("tool_use_result") if isinstance(metadata, dict) else None
    task_id = result.get("backgroundTaskId") if isinstance(result, dict) else None
    return task_id if isinstance(task_id, str) and task_id else None


def _claude_consistent_result_is_error(observation: Any) -> bool | None:
    if not isinstance(observation, dict):
        return None
    extra = observation.get("extra")
    metadata = extra.get("tool_result_metadata") if isinstance(extra, dict) else None
    raw = metadata.get("raw_tool_result") if isinstance(metadata, dict) else None
    tool_result = observation.get("tool_result")
    values = [
        container[key]
        for container, key in (
            (observation, "is_error"),
            (tool_result, "is_error"),
            (extra, "tool_result_is_error"),
            (metadata, "is_error"),
            (raw, "is_error"),
        )
        if isinstance(container, dict) and key in container
    ]
    if any(type(value) is not bool for value in values):
        raise ReplayError("Claude auxiliary outcome has a non-boolean error flag")
    explicit = {value for value in values if type(value) is bool}
    if len(explicit) > 1:
        raise ReplayError("Claude auxiliary outcome has conflicting error flags")
    return next(iter(explicit), None)


def _validate_claude_agent_parent(
    steps: list[Any],
    step_offset: int,
    arguments: dict[str, Any],
    observation: Any,
) -> None:
    keys = {"description", "prompt", "subagent_type"}
    if (
        set(arguments) != keys
        or any(not isinstance(arguments[key], str) or not arguments[key] for key in keys)
        or _claude_consistent_result_is_error(observation) is True
    ):
        raise ReplayError("Claude Agent parent is not an exact successful call")
    extra = observation.get("extra") if isinstance(observation, dict) else None
    metadata = extra.get("tool_result_metadata") if isinstance(extra, dict) else None
    result = metadata.get("tool_use_result") if isinstance(metadata, dict) else None
    content = result.get("content") if isinstance(result, dict) else None
    if (
        not isinstance(result, dict)
        or result.get("status") != "completed"
        or result.get("prompt") != arguments["prompt"]
        or result.get("agentType") != arguments["subagent_type"]
        or not isinstance(content, list)
        or len(content) != 1
        or not isinstance(content[0], dict)
        or set(content[0]) != {"type", "text"}
        or content[0].get("type") != "text"
        or not isinstance(content[0].get("text"), str)
        or not content[0]["text"]
    ):
        raise ReplayError("Claude Agent result does not identify the requested child")
    sidechain: list[dict[str, Any]] = []
    for child in steps[step_offset + 1 :]:
        if (
            not isinstance(child, dict)
            or not isinstance(child.get("extra"), dict)
            or child["extra"].get("is_sidechain") is not True
        ):
            break
        sidechain.append(child)
    parent_calls = steps[step_offset].get("tool_calls")
    one_parent = (
        isinstance(parent_calls, list)
        and sum(isinstance(call, dict) and call.get("function_name") == "Agent" for call in parent_calls) == 1
    )
    child_call_count = sum(
        len(child.get("tool_calls")) for child in sidechain if isinstance(child.get("tool_calls"), list)
    )
    final_message = next(
        (
            child["message"]
            for child in reversed(sidechain)
            if child.get("source") == "agent" and isinstance(child.get("message"), str) and child["message"]
        ),
        None,
    )
    terminal = sidechain[-1] if sidechain else None
    if (
        not one_parent
        or child_call_count == 0
        or final_message != content[0]["text"]
        or not isinstance(terminal, dict)
        or terminal.get("source") != "agent"
        or terminal.get("message") != content[0]["text"]
        or (isinstance(terminal.get("tool_calls"), list) and bool(terminal["tool_calls"]))
    ):
        raise ReplayError("Claude Agent child sidechain is incomplete")


def _safe_claude_monitor_literal(value: str) -> bool:
    offset = 0
    while offset < len(value):
        character = value[offset]
        if character == "\\":
            offset += 1
            if offset == len(value) or value[offset] not in "[]":
                return False
        elif not (character.isascii() and (character.isalnum() or character.isspace()) or character in "|:._/-[]"):
            return False
        offset += 1
    return bool(value)


def _validate_claude_monitor_poll(
    arguments: dict[str, Any],
    observation: Any,
    background_task_ids: set[str],
) -> None:
    if (
        set(arguments) != {"command", "description", "persistent", "timeout_ms"}
        or arguments.get("persistent") is not False
        or not isinstance(arguments.get("description"), str)
        or not arguments["description"]
        or type(arguments.get("timeout_ms")) is not int
        or arguments["timeout_ms"] <= 0
        or _claude_consistent_result_is_error(observation) is True
    ):
        raise ReplayError("Claude Monitor is not an exact non-persistent success")
    command = arguments.get("command")
    tail_grep = re.fullmatch(
        r"tail -f (/tmp/claude-[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+/tasks/([A-Za-z0-9._-]+)\.output) "
        r"\| grep -E --line-buffered (['\"])([^'\"\r\n]+)\3",
        command if isinstance(command, str) else "",
    )
    wait_tail = re.fullmatch(
        r"until ! pgrep -f (['\"])([^'\"\r\n]+)\1 > /dev/null 2>&1; "
        r"do sleep [1-9][0-9]*; done; echo (['\"])([^'\"\r\n]+)\3; "
        r"tail -[1-9][0-9]* "
        r"(/tmp/claude-[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)+/tasks/([A-Za-z0-9._-]+)\.output)",
        command if isinstance(command, str) else "",
    )
    if tail_grep is not None:
        task_id = tail_grep.group(2)
        literals = [tail_grep.group(4)]
    elif wait_tail is not None:
        task_id = wait_tail.group(6)
        literals = [wait_tail.group(2), wait_tail.group(4)]
    else:
        raise ReplayError("Claude Monitor command is unsupported")
    extra = observation.get("extra") if isinstance(observation, dict) else None
    metadata = extra.get("tool_result_metadata") if isinstance(extra, dict) else None
    result = metadata.get("tool_use_result") if isinstance(metadata, dict) else None
    if (
        task_id not in background_task_ids
        or not all(_safe_claude_monitor_literal(value) for value in literals)
        or not isinstance(result, dict)
        or not isinstance(result.get("taskId"), str)
        or not result["taskId"]
        or result.get("timeoutMs") != arguments["timeout_ms"]
        or result.get("persistent") is not False
    ):
        raise ReplayError("Claude Monitor does not prove one owned non-persistent poll")


def _claude_bash_timeout(arguments: dict[str, Any]) -> float | None:
    value = arguments.get("timeout")
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool) or value <= 0:
        raise ReplayError("Claude Code Bash timeout must be a positive number")
    return float(value) / 1000.0


def _codex_normalization(payload: dict[str, Any]) -> dict[str, int] | None:
    value = (payload.get("extra") or {}).get("riptide_normalization")
    if value is None:
        return None
    keys = {
        "kind",
        "native_session_count",
        "normalized_session_count",
        "native_executable_event_count",
        "extracted_executable_event_count",
        "outcome_bound_executable_event_count",
        "outcome_unavailable_executable_event_count",
        "unaccounted_executable_events",
    }
    if not isinstance(value, dict) or set(value) != keys or value.get("kind") != "codex-native-sessions-flattened":
        raise ReplayError("Codex riptide_normalization has an unsupported schema")
    counts = {key: value[key] for key in keys - {"kind"}}
    if any(not isinstance(count, int) or isinstance(count, bool) or count < 0 for count in counts.values()):
        raise ReplayError("Codex riptide_normalization has an invalid count")
    if (
        counts["native_session_count"] == 0
        or counts["native_session_count"] != counts["normalized_session_count"]
        or counts["native_executable_event_count"] != counts["extracted_executable_event_count"]
        or counts["unaccounted_executable_events"] != 0
        or counts["outcome_bound_executable_event_count"] + counts["outcome_unavailable_executable_event_count"]
        != counts["extracted_executable_event_count"]
    ):
        raise ReplayError("Codex riptide_normalization counts are inconsistent")
    if payload.get("subagent_trajectories"):
        raise ReplayError("flattened Codex trajectory still embeds subagent trajectories")
    source_ids = {
        (step.get("extra") or {}).get("riptide_source_trajectory_id")
        for step in payload.get("steps", [])
        if isinstance(step, dict)
    }
    if None in source_ids or "" in source_ids or len(source_ids) != counts["normalized_session_count"]:
        raise ReplayError("flattened Codex trajectory does not account for every session")
    return counts


def _authoritative_codex_invocations(
    tool_call: dict[str, Any],
    seen_source_items: set[str],
    seen_parallel_groups: set[str],
) -> list[tuple[str, dict[str, Any], str | None, Any]] | None:
    extra = tool_call.get("extra")
    extra = extra if isinstance(extra, dict) else {}
    if "riptide_nested_outcomes" in extra:
        raise ReplayError("Codex riptide_nested_outcomes was replaced by riptide_nested_invocations")
    if "riptide_nested_invocations" not in extra:
        return None
    values = extra["riptide_nested_invocations"]
    if not isinstance(values, list) or not values:
        raise ReplayError("Codex riptide_nested_invocations must be a nonempty array")
    result: list[tuple[str, dict[str, Any], str | None, Any]] = []
    local_groups: dict[str, tuple[int, int, str, set[str]]] = {}
    for index, invocation in enumerate(values):
        allowed = {"index", "tool", "arguments", "source_item_id", "outcome", "parallel_group_id"}
        if (
            not isinstance(invocation, dict)
            or set(invocation) - allowed
            or len(invocation) != 5 + ("parallel_group_id" in invocation)
            or invocation.get("index") != index
        ):
            raise ReplayError("Codex authoritative invocation has invalid keys or index")
        tool = invocation.get("tool")
        arguments = invocation.get("arguments")
        if tool not in {"exec_command", "write_stdin", "apply_patch"} or not isinstance(arguments, dict):
            raise ReplayError("Codex authoritative invocation has invalid tool or arguments")
        valid_arguments = (
            (tool == "exec_command" and isinstance(arguments.get("cmd"), str))
            or (
                tool == "write_stdin"
                and isinstance(arguments.get("session_id"), (str, int))
                and not isinstance(arguments.get("session_id"), bool)
            )
            or (tool == "apply_patch" and set(arguments) == {"input"} and isinstance(arguments.get("input"), str))
        )
        if not valid_arguments:
            raise ReplayError("Codex authoritative invocation has invalid canonical arguments")
        source_item_id = invocation.get("source_item_id")
        if not isinstance(source_item_id, str) or not source_item_id or any(ord(c) < 32 for c in source_item_id):
            raise ReplayError("Codex authoritative invocation has invalid source_item_id")
        if source_item_id in seen_source_items:
            raise ReplayError("Codex authoritative invocation repeats a source_item_id")
        seen_source_items.add(source_item_id)
        group = invocation.get("parallel_group_id")
        if group is not None:
            safe_poll = tool == "write_stdin" and arguments.get("chars") == "" and "session_id" in arguments
            if (
                not isinstance(group, str)
                or not group
                or any(ord(c) < 32 for c in group)
                or not (
                    (
                        tool == "exec_command"
                        and arguments.get("run_in_background") is not True
                        and "session_id" not in arguments
                    )
                    or safe_poll
                )
            ):
                raise ReplayError("Codex authoritative invocation has an unsafe parallel group")
            count, start, group_tool, sessions = local_groups.get(group, (0, index, tool, set()))
            if start + count != index:
                raise ReplayError("Codex authoritative parallel group is not contiguous")
            if group_tool != tool:
                raise ReplayError("Codex authoritative parallel group mixes operation types")
            if safe_poll:
                session = str(arguments["session_id"])
                if session in sessions:
                    raise ReplayError("Codex authoritative parallel poll group repeats a session")
                sessions.add(session)
            local_groups[group] = (count + 1, start, group_tool, sessions)
        outcome = invocation.get("outcome")
        if outcome is not None:
            required = {"status", "returncode", "session_id"}
            if (
                not isinstance(outcome, dict)
                or not required <= set(outcome)
                or not set(outcome) <= required | {"output"}
                or ("output" in outcome and not isinstance(outcome["output"], str))
            ):
                raise ReplayError("Codex authoritative outcome has invalid keys")
            status, returncode, session_id = outcome["status"], outcome["returncode"], outcome["session_id"]
            if (
                status not in {"ok", "failed", "running", "timed_out", "skipped_not_executed"}
                or (returncode is not None and (not isinstance(returncode, int) or isinstance(returncode, bool)))
                or (session_id is not None and (not isinstance(session_id, str) or not session_id))
                or (status == "ok" and returncode not in (None, 0))
                or (status == "failed" and returncode == 0)
                or (status == "running" and (returncode is not None or session_id is None))
                or (status in {"timed_out", "skipped_not_executed"} and returncode is not None)
            ):
                raise ReplayError("Codex authoritative outcome has contradictory fields")
        result.append((tool, dict(arguments), group, outcome))
    for group, (count, _, _, _) in local_groups.items():
        if count < 2 or group in seen_parallel_groups:
            raise ReplayError("Codex authoritative parallel group is singleton or reused")
        seen_parallel_groups.add(group)
    return result


def _trajectory_nodes(payload: dict[str, Any], preflattened_ids: set[str] | None) -> list[tuple[str, dict[str, Any]]]:
    nodes: list[tuple[str, dict[str, Any]]] = []
    seen: set[str] = set()

    def collect(node: dict[str, Any], root: bool = False) -> None:
        identity = node.get("trajectory_id")
        if not isinstance(identity, str) or not identity:
            if not root:
                raise ReplayError("embedded ATIF trajectory has no trajectory_id")
            identity = "root"
        if identity in seen:
            raise ReplayError(f"ATIF trajectory_id is duplicated: {identity!r}")
        seen.add(identity)
        nodes.append((identity, node))
        children = node.get("subagent_trajectories")
        if children is None:
            children = []
        if not isinstance(children, list):
            raise ReplayError("ATIF subagent_trajectories is not an array")
        child_ids = set()
        for child in children:
            if not isinstance(child, dict):
                raise ReplayError("embedded ATIF trajectory is not an object")
            child_id = child.get("trajectory_id")
            if not isinstance(child_id, str) or not child_id:
                raise ReplayError("embedded ATIF trajectory has no trajectory_id")
            child_ids.add(child_id)
        steps = node.get("steps")
        if not isinstance(steps, list):
            raise ReplayError("ATIF trajectory has no steps list")
        for step in steps:
            observation = step.get("observation") if isinstance(step, dict) else None
            results = observation.get("results") if isinstance(observation, dict) else None
            for result in results if isinstance(results, list) else []:
                refs = result.get("subagent_trajectory_ref") if isinstance(result, dict) else None
                for ref in refs if isinstance(refs, list) else []:
                    referenced = ref.get("trajectory_id") if isinstance(ref, dict) else None
                    if not isinstance(referenced, str) or not referenced:
                        raise ReplayError("ATIF subagent reference has no trajectory_id")
                    if referenced not in child_ids and not (
                        root and preflattened_ids and referenced in preflattened_ids
                    ):
                        raise ReplayError(f"ATIF trajectory {identity!r} references unembedded subagent {referenced!r}")
        for child in children:
            collect(child)

    collect(payload, True)
    return nodes


def read_trajectory(path: Path) -> AtifTrajectory:
    """Load executable tool calls from an ATIF trajectory JSON document."""
    try:
        payload = json.loads(_read_trajectory_text(path))
    except json.JSONDecodeError as exc:
        raise ReplayError(f"ATIF trajectory is invalid JSON: {exc.msg}") from exc
    if not isinstance(payload, dict):
        raise ReplayError("ATIF trajectory root is not a JSON object")
    schema = payload.get("schema_version")
    if not isinstance(schema, str) or not schema.startswith("ATIF-v1."):
        raise ReplayError(f"unsupported ATIF trajectory schema: {schema!r}")
    agent = payload.get("agent") if isinstance(payload.get("agent"), dict) else {}
    source_agent_original_format = _source_agent_original_format(agent)
    codex_exec_originator = _codex_exec_originator(agent)
    claude_code = _claude_code_agent(agent)
    normalization = None if claude_code else _codex_normalization(payload)
    if normalization is not None and not codex_exec_originator:
        raise ReplayError("Codex normalization requires a Code Mode trajectory")
    if claude_code:
        source_agent_original_format = CLAUDE_CODE_AGENT_NAME
    root_original_format = source_agent_original_format
    preflattened_ids = (
        {(step.get("extra") or {}).get("riptide_source_trajectory_id") for step in payload["steps"]}
        if normalization is not None
        else None
    )
    nodes = _trajectory_nodes(payload, preflattened_ids)
    actions: list[AtifAction] = []
    background_task_ids: set[str] = set()
    claude_auxiliary_call_ids: set[str] = set()
    source_item_ids: set[str] = set()
    parallel_group_ids: set[str] = set()
    authoritative_events = 0
    authoritative_outcomes = 0
    for node_id, node, step_number, step in (
        (node_id, source_node, number, step)
        for node_id, source_node in nodes
        for number, step in enumerate(source_node["steps"], start=1)
    ):
        node_agent = node.get("agent") if isinstance(node.get("agent"), dict) else {}
        source_agent_original_format = _source_agent_original_format(node_agent)
        codex_exec_originator = _codex_exec_originator(node_agent)
        claude_code = _claude_code_agent(node_agent)
        if claude_code:
            source_agent_original_format = CLAUDE_CODE_AGENT_NAME
        steps = node["steps"]
        if not isinstance(step, dict):
            raise ReplayError(f"ATIF step {step_number} is not a JSON object")
        source_trajectory_id = (step.get("extra") or {}).get("riptide_source_trajectory_id")
        if not isinstance(source_trajectory_id, str) or not source_trajectory_id:
            source_trajectory_id = node_id
        tool_calls = step.get("tool_calls") or []
        if not isinstance(tool_calls, list):
            raise ReplayError(f"ATIF step {step_number} has invalid tool_calls")
        for call_number, tool_call in enumerate(tool_calls, start=1):
            if not isinstance(tool_call, dict):
                raise ReplayError(f"ATIF step {step_number} contains an invalid tool call")
            source_tool = tool_call.get("function_name") or tool_call.get("name")
            arguments = tool_call.get("arguments") or {}
            if not isinstance(source_tool, str) or not source_tool:
                raise ReplayError(f"ATIF step {step_number} tool call has no function name")
            if not isinstance(arguments, dict):
                raise ReplayError(f"ATIF tool {source_tool!r} arguments are not an object")
            if claude_code and source_tool not in CLAUDE_CODE_TOOLS:
                raise ReplayError(f"unsupported Claude Code tool: {source_tool}")
            if claude_code and source_tool == "Bash" and arguments.get("dangerouslyDisableSandbox") is True:
                raise ReplayError("unsupported Claude Code Bash payload: dangerouslyDisableSandbox=true")
            if claude_code and source_tool == "Bash":
                _claude_bash_timeout(arguments)
            if codex_exec_originator and source_tool == "exec":
                input_source = arguments.get("input")
                if not isinstance(input_source, str):
                    raise ReplayError("Codex exec tool call has no string input")
                structural_identity = f"step-{step_number}-call-{call_number}"
                source_call_id = tool_call.get("tool_call_id")
                outer_call_identity = (
                    f"{structural_identity}:{source_call_id}"
                    if isinstance(source_call_id, str) and source_call_id
                    else structural_identity
                )
                authoritative = _authoritative_codex_invocations(tool_call, source_item_ids, parallel_group_ids)
                if authoritative is not None and normalization is None:
                    raise ReplayError("authoritative Codex invocations require normalized provenance")
                observation = step.get("observation") if normalization is None else None
                if authoritative is None and isinstance(observation, dict) and "results" in observation:
                    observation = _call_observation(step, tool_call)
                elif len(tool_calls) != 1:
                    observation = None
                nested = (
                    authoritative
                    if authoritative is not None
                    else _trajectory_code_mode_invocations(input_source, outer_call_identity, observation)
                )
                for nested_tool, nested_arguments, parallel_group_id, observation in nested:
                    authoritative_events += authoritative is not None
                    authoritative_outcomes += authoritative is not None and observation is not None
                    index = len(actions) + 1
                    actions.append(
                        AtifAction(
                            index=index,
                            step_id=step.get("step_id", step_number),
                            timestamp_sec=_timestamp_sec(step.get("timestamp"), action_index=index),
                            tool=nested_tool,
                            source_tool=f"exec.{nested_tool}",
                            code_mode=True,
                            source_agent_original_format=source_agent_original_format,
                            arguments=nested_arguments,
                            observation=observation,
                            observation_text=_text_content(observation),
                            parallel_group_id=parallel_group_id,
                            source_trajectory_id=source_trajectory_id,
                        )
                    )
                continue
            tool = _canonical_tool(source_tool, source_agent_original_format=source_agent_original_format)
            arguments = _canonical_arguments(
                arguments,
                source_tool=source_tool,
                source_agent_original_format=source_agent_original_format,
            )
            mini_swe = source_agent_original_format == MINI_SWE_AGENT_ORIGINAL_FORMAT and source_tool == "bash"
            observation = (
                _mini_swe_call_observation(step, tool_call)
                if mini_swe
                else _call_observation(step, tool_call) if claude_code else step.get("observation")
            )
            if mini_swe:
                _mini_swe_result(observation)
            consistent_result_error = _claude_consistent_result_is_error(observation) if claude_code else None
            if claude_code and source_tool in CLAUDE_CODE_AUXILIARY_METADATA_TOOLS:
                call_id = tool_call.get("tool_call_id")
                if not isinstance(call_id, str) or not call_id:
                    raise ReplayError("Claude auxiliary call has no identity")
                if call_id in claude_auxiliary_call_ids:
                    raise ReplayError("Claude auxiliary call reuses a tool_call_id")
                claude_auxiliary_call_ids.add(call_id)
            if claude_code and source_tool == "Agent":
                _validate_claude_agent_parent(steps, step_number - 1, arguments, observation)
            if claude_code and source_tool == "Monitor":
                _validate_claude_monitor_poll(arguments, observation, background_task_ids)
            if (
                claude_code
                and source_tool == "Bash"
                and consistent_result_error is not True
                and (arguments.get("run_in_background") is True or _claude_background_task_id(observation) is not None)
            ):
                task_id = _claude_background_task_id(observation)
                if task_id is None:
                    raise ReplayError("Claude Code background Bash has no backgroundTaskId")
                if task_id in background_task_ids:
                    raise ReplayError(f"Claude Code background Bash reuses backgroundTaskId {task_id!r}")
                background_task_ids.add(task_id)
                # Claude can auto-background Bash even when its call arguments do
                # not carry run_in_background. A successful structured task ID is
                # the exact source-side evidence that this call created a session.
                arguments = {**arguments, "run_in_background": True}
            index = len(actions) + 1
            code_mode = codex_exec_originator and _is_code_mode_metadata_tool(source_tool)
            actions.append(
                AtifAction(
                    index=index,
                    step_id=step.get("step_id", step_number),
                    timestamp_sec=_timestamp_sec(step.get("timestamp"), action_index=index),
                    tool=tool,
                    source_tool=f"exec.{source_tool}" if code_mode else source_tool,
                    code_mode=code_mode,
                    source_agent_original_format=source_agent_original_format,
                    arguments=arguments,
                    observation=observation,
                    observation_text=_text_content(observation),
                    source_trajectory_id=source_trajectory_id,
                )
            )
    if normalization is not None:
        executable = sum(action.tool in SUPPORTED_EXECUTING_TOOLS for action in actions)
        if (
            authoritative_events != normalization["extracted_executable_event_count"]
            or executable != normalization["extracted_executable_event_count"]
            or authoritative_outcomes != normalization["outcome_bound_executable_event_count"]
        ):
            raise ReplayError("normalized Codex trajectory counts do not match Python-expanded actions")
    if not actions:
        raise ReplayError("ATIF trajectory contains no tool actions")
    actions = [
        replace(action, index=index)
        for index, action in enumerate(sorted(actions, key=lambda action: action.timestamp_sec), 1)
    ]
    return AtifTrajectory(
        source_path=path,
        schema_version=schema,
        agent=agent,
        source_agent_original_format=root_original_format,
        session_id=(payload.get("session_id") if isinstance(payload.get("session_id"), str) else None),
        actions=actions,
        subagent_trajectories_flattened=(len(preflattened_ids) - 1 if preflattened_ids else len(nodes) - 1),
        contains_codex_exec=any(
            _codex_exec_originator(node["agent"]) for _, node in nodes if isinstance(node.get("agent"), dict)
        ),
    )


_PYTHON3_ABSENT_PATTERN = re.compile(r"python3:\s*(?:command\s+)?not found", re.IGNORECASE)


def source_observed_python3_absent(trajectory: AtifTrajectory) -> bool:
    """Whether the recorded run observed ``python3`` missing from PATH.

    The Terminal-Bench base image ships python3, but several tasks remove it so
    the agent must solve without it. When the source observed ``python3: command
    not found``, a faithful replay must also hide the image's python3 (the replay
    driver itself uses a host-mounted python at a separate path, off PATH).
    """
    return any(_PYTHON3_ABSENT_PATTERN.search(action.observation_text or "") for action in trajectory.actions)


def validate_supported_actions(trajectory: AtifTrajectory) -> None:
    """Reject an incomplete replay rather than omitting unimplemented work.

    CPU numbers are comparable only when every executable action in the source
    trajectory is either reproduced or explicitly known to be metadata.
    """
    if (
        trajectory.contains_codex_exec
        or _codex_exec_originator(trajectory.agent)
        or any(action.code_mode for action in trajectory.actions)
    ):
        raise ReplayError("Codex exec ATIF replay and profiling are not supported in release 26.10")
    if (
        _claude_code_agent(trajectory.agent)
        or trajectory.source_agent_original_format == CLAUDE_CODE_AGENT_NAME
        or any(
            action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME or action.tool.startswith("claude_")
            for action in trajectory.actions
        )
    ):
        raise ReplayError("Claude Code ATIF replay and profiling are not supported in release 26.10")
    parallel_groups: dict[str, list[AtifAction]] = {}
    for action in trajectory.actions:
        if action.tool == "exec_command":
            _execution_shell(action)
        if action.parallel_group_id:
            parallel_groups.setdefault(action.parallel_group_id, []).append(action)
    # Sequentialization is intentional for per-command CPU replay; unsafe
    # session/patch group members remain fail-closed.
    for group_id, actions in parallel_groups.items():
        poll_sessions: set[str] = set()
        for action in actions:
            safe_poll = (
                action.tool == "write_stdin"
                and action.arguments.get("chars") == ""
                and "session_id" in action.arguments
            )
            if not action.code_mode or not (
                (
                    action.tool == "exec_command"
                    and action.arguments.get("run_in_background") is not True
                    and "session_id" not in action.arguments
                )
                or safe_poll
                or _is_metadata_action(action)
            ):
                raise ReplayError(
                    "ATIF trajectory contains a parallel group that cannot be safely " f"sequentialized: {group_id}"
                )
            if safe_poll:
                session = str(action.arguments["session_id"])
                if session in poll_sessions:
                    raise ReplayError(f"ATIF parallel poll group repeats a session: {group_id}")
                poll_sessions.add(session)
    unsupported = sorted(
        {
            action.tool
            for action in trajectory.actions
            if action.tool not in SUPPORTED_EXECUTING_TOOLS and not _is_metadata_action(action)
        }
    )
    if unsupported:
        raise ReplayError("ATIF trajectory contains unsupported executable tool calls: " + ", ".join(unsupported))
    if not any(action.tool in SUPPORTED_EXECUTING_TOOLS for action in trajectory.actions):
        raise ReplayError("ATIF trajectory contains no supported executable tool calls")


def _preview(value: str | bytes) -> str:
    if isinstance(value, bytes):
        value = value.decode(errors="replace")
    return value[:PREVIEW_LENGTH]


def _recorded_tool_wall_sec(action: AtifAction) -> float:
    """Recorded execution time of a tool call, read from its observation.

    Codex ``exec_command`` observations report ``Wall time: N seconds``. For the
    ``llm-gaps`` pacing this is added to the action's issue timestamp to recover
    when the tool actually finished, so the inter-action model gap can be
    isolated from tool-execution time. Returns 0.0 when no wall time is recorded
    (e.g. ``apply_patch``), which collapses that turn's gap to issue-to-issue.
    """
    match = _WALL_TIME_PATTERN.search(action.observation_text)
    if match is None:
        return 0.0
    try:
        return max(0.0, float(match.group(1)))
    except ValueError:
        return 0.0


def _source_argument_parse_failed(action: AtifAction) -> bool:
    if _TOOL_ARGUMENT_PARSE_FAILURE_PATTERN.search(action.observation_text) is None:
        return False
    if action.source_agent_original_format not in {CLAUDE_CODE_AGENT_NAME, MINI_SWE_AGENT_ORIGINAL_FORMAT}:
        return action.tool in {"exec_command", "write_stdin", "apply_patch"}
    if action.tool == "exec_command":
        return not isinstance(action.arguments.get("cmd", action.arguments.get("command")), str)
    if action.tool == "write_stdin":
        session_id = action.arguments.get("session_id")
        return not (
            isinstance(session_id, str) and bool(session_id) or type(session_id) is int and 0 <= session_id < 2**64
        )
    if action.tool == "apply_patch":
        return not isinstance(action.arguments.get("input", action.arguments.get("patch")), str)
    return False


def _source_outcome(action: AtifAction) -> tuple[str, int | None, str | None]:
    if action.source_agent_original_format == MINI_SWE_AGENT_ORIGINAL_FORMAT and action.source_tool == "bash":
        mini_swe_result = _mini_swe_result(action.observation)
        if mini_swe_result is not None:
            exception_info = mini_swe_result.get("exception_info")
            if isinstance(exception_info, str):
                if exception_info == "action was not executed":
                    return "skipped_not_executed", None, None
                if "timed out" in exception_info.lower():
                    return "timed_out", None, None
            returncode = mini_swe_result.get("returncode")
            if isinstance(returncode, int) and not isinstance(returncode, bool):
                return ("ok" if returncode == 0 else "failed"), returncode, None
    if _is_metadata_action(action):
        return "ignored_metadata", None, None
    if action.code_mode:
        if isinstance(action.observation, dict) and isinstance(action.observation.get("status"), str):
            return (
                action.observation["status"],
                action.observation.get("returncode"),
                action.observation.get("session_id"),
            )
        return "unknown", None, None
    if (
        action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME
        and action.source_tool == "Bash"
        and action.arguments.get("run_in_background") is True
        and _claude_code_result(action) is not True
    ):
        return "running", None, _claude_background_task_id(action.observation)
    if action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME and action.source_tool == "TaskOutput":
        metadata = (
            (action.observation.get("extra") or {}).get("tool_result_metadata")
            if isinstance(action.observation, dict)
            else None
        )
        task = (metadata.get("tool_use_result") or {}).get("task") if isinstance(metadata, dict) else None
        if isinstance(task, dict) and task.get("status") == "running":
            return "running", None, task.get("task_id") if isinstance(task.get("task_id"), str) else None
        returncode = task.get("exitCode") if isinstance(task, dict) else None
        if isinstance(returncode, int) and not isinstance(returncode, bool):
            return (
                ("ok" if returncode == 0 else "failed"),
                returncode,
                task.get("task_id") if isinstance(task.get("task_id"), str) else None,
            )
    if action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME:
        result = _claude_code_result(action)
        is_error = (
            result
            if result is not None
            else (
                "[error] tool reported failure" in action.observation_text.lower()
                or (
                    action.source_tool == "Bash"
                    and any(
                        line.strip().startswith(("/bin/bash:", "/bin/sh:", "bash:", "sh:"))
                        and "command not found" in line
                        for line in action.observation_text.splitlines()
                    )
                )
            )
        )
        return ("failed" if is_error else "ok"), None, None
    if _source_argument_parse_failed(action):
        return "failed", None, None
    if action.tool == "exec_command" and _EXEC_COMMAND_FAILURE_PATTERN.search(action.observation_text):
        return "failed", None, None
    if action.tool == "apply_patch" and _APPLY_PATCH_FAILURE_PATTERN.search(action.observation_text):
        return "failed", None, None
    match = _SESSION_ID_PATTERN.search(action.observation_text)
    if match is not None:
        return "running", None, match.group(1)
    for pattern in _EXIT_CODE_PATTERNS:
        match = pattern.search(action.observation_text)
        if match is not None:
            returncode = int(match.group(1))
            return ("ok" if returncode == 0 else "failed"), returncode, None
    if action.tool == "apply_patch" and action.observation_text:
        return "ok", 0, None
    return "unknown", None, None


def _mini_swe_result(observation: Any) -> dict[str, Any]:
    """Validate the already bound mini-SWE structured result before execution."""
    content = observation.get("content") if isinstance(observation, dict) else None
    if not isinstance(content, str):
        raise ReplayError("mini-SWE result has no structured content")
    try:
        payload = json.loads(content)
    except json.JSONDecodeError as error:
        raise ReplayError("mini-SWE result content is not a JSON object") from error
    if not isinstance(payload, dict):
        raise ReplayError("mini-SWE result content is not a JSON object")
    if "returncode" not in payload:
        raise ReplayError("mini-SWE result has no returncode")
    returncode = payload["returncode"]
    exception = payload.get("exception_info")
    special_exception = isinstance(exception, str) and (
        exception == "action was not executed" or "timed out" in exception.lower()
    )
    if (
        (returncode is not None and (type(returncode) is not int or not -(2**63) <= returncode < 2**63))
        or (exception is not None and not isinstance(exception, str))
        or (returncode is None and not special_exception)
        or (returncode == 0 and bool(exception))
        or (special_exception and returncode not in (None, -1))
    ):
        raise ReplayError("mini-SWE result has invalid or conflicting outcome evidence")
    return payload


def _claude_result_is_error(observation: Any) -> bool | None:
    if not isinstance(observation, dict):
        return None
    extra = observation.get("extra")
    if not isinstance(extra, dict):
        extra = {}
    direct = extra.get("tool_result_is_error")
    if isinstance(direct, bool):
        return direct
    raw = (extra.get("tool_result_metadata") or {}).get("raw_tool_result")
    if isinstance(raw, dict) and isinstance(raw.get("is_error"), bool):
        return raw["is_error"]
    tool_result = observation.get("tool_result")
    if isinstance(tool_result, dict) and isinstance(tool_result.get("is_error"), bool):
        return tool_result["is_error"]
    if isinstance(observation.get("is_error"), bool):
        return observation["is_error"]
    metadata = extra.get("tool_result_metadata")
    return (
        metadata.get("is_error") if isinstance(metadata, dict) and isinstance(metadata.get("is_error"), bool) else None
    )


def _claude_code_result(action: AtifAction) -> bool | None:
    """Return Claude's recorded error flag, preserving its per-call pairing."""
    return _claude_result_is_error(action.observation)


def _outcome_matches(
    expected_status: str,
    expected_returncode: int | None,
    status: str,
    returncode: int | None = None,
) -> bool | None:
    if expected_status == "unknown":
        return None
    if expected_status == "running" and status in {"ok", "failed"}:
        # A replayed asynchronous command may finish before the source poll.
        # Its recorded terminal poll will compare the eventual exit status.
        return True
    if expected_status != status:
        return False
    if expected_status == "failed" and expected_returncode == -1:
        # ATIF uses -1 as a generic failure/signal result; local subprocess
        # reports concrete signal values such as -11 for SIGSEGV.
        return returncode is not None and returncode != 0
    return expected_returncode is None or expected_returncode == returncode


def _parse_patch(patch_text: str) -> list[_PatchOperation]:
    lines = patch_text.splitlines()
    if not lines or lines[0] != "*** Begin Patch" or lines[-1] != "*** End Patch":
        raise ReplayError("apply_patch payload is missing Begin Patch or End Patch markers")
    operations: list[_PatchOperation] = []
    index = 1
    while index < len(lines) - 1:
        header = lines[index]
        index += 1
        if header.startswith("*** Add File: "):
            kind = "add"
            path = header[len("*** Add File: ") :]
        elif header.startswith("*** Update File: "):
            kind = "update"
            path = header[len("*** Update File: ") :]
        elif header.startswith("*** Delete File: "):
            operations.append(_PatchOperation(kind="delete", path=header[len("*** Delete File: ") :]))
            continue
        else:
            raise ReplayError(f"unsupported apply_patch operation: {header!r}")
        move_to: str | None = None
        if kind == "update" and index < len(lines) - 1 and lines[index].startswith("*** Move to: "):
            move_to = lines[index][len("*** Move to: ") :]
            index += 1
        operation_lines: list[str] = []
        while index < len(lines) - 1 and not (
            lines[index].startswith("*** Add File: ")
            or lines[index].startswith("*** Update File: ")
            or lines[index].startswith("*** Delete File: ")
        ):
            operation_lines.append(lines[index])
            index += 1
        operations.append(_PatchOperation(kind=kind, path=path, lines=tuple(operation_lines), move_to=move_to))
    return operations


def _workspace_path(raw_path: str, *, cwd: Path, patch_roots: tuple[Path, ...]) -> Path:
    """Resolve patch paths while keeping mutation inside disposable task roots."""
    path = Path(raw_path)
    candidate = (path if path.is_absolute() else cwd / path).resolve()
    roots = (cwd.resolve(), *(root.resolve() for root in patch_roots))
    if not any(candidate == root or root in candidate.parents for root in roots):
        raise ReplayError(f"apply_patch path leaves permitted replay patch roots: {raw_path}")
    return candidate


def _find_subsequence(lines: list[str], wanted: list[str], start: int) -> int:
    if not wanted:
        return start
    for index in range(start, len(lines) - len(wanted) + 1):
        if lines[index : index + len(wanted)] == wanted:
            return index
    return -1


def _apply_update(path: Path, lines: tuple[str, ...]) -> None:
    if not path.is_file():
        raise ReplayError(f"cannot update missing file: {path}")
    source_text = path.read_text()
    source_ends_newline = source_text.endswith("\n")
    content = source_text.splitlines()
    hunks: list[list[str]] = []
    current: list[str] | None = None
    for line in lines:
        if line.startswith("@@"):
            if current is not None:
                hunks.append(current)
            current = []
        elif current is None:
            raise ReplayError(f"update for {path} contains text before its first hunk")
        else:
            current.append(line)
    if current is not None:
        hunks.append(current)
    if not hunks:
        raise ReplayError(f"update for {path} contains no hunks")
    cursor = 0
    for hunk in hunks:
        before: list[str] = []
        after: list[str] = []
        for line in hunk:
            if line.startswith(" "):
                before.append(line[1:])
                after.append(line[1:])
            elif line.startswith("-"):
                before.append(line[1:])
            elif line.startswith("+"):
                after.append(line[1:])
            elif line == r"\ No newline at end of file":
                source_ends_newline = False
            else:
                raise ReplayError(f"invalid update hunk line for {path}: {line!r}")
        location = _find_subsequence(content, before, cursor)
        if location < 0:
            raise ReplayError(f"update hunk context was not found in {path}")
        content[location : location + len(before)] = after
        cursor = location + len(after)
    output = "\n".join(content)
    if source_ends_newline:
        output += "\n"
    path.write_text(output)


def apply_patch_payload(patch_text: str, *, cwd: Path, patch_roots: tuple[Path, ...] = ()) -> None:
    """Apply Codex patch operations within explicitly permitted replay roots."""
    for operation in _parse_patch(patch_text):
        path = _workspace_path(operation.path, cwd=cwd, patch_roots=patch_roots)
        if operation.kind == "add":
            if path.exists():
                raise ReplayError(f"cannot add existing file: {path}")
            added: list[str] = []
            for line in operation.lines:
                if not line.startswith("+"):
                    raise ReplayError(f"invalid add-file patch line for {path}: {line!r}")
                added.append(line[1:])
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("\n".join(added) + ("\n" if added else ""))
        elif operation.kind == "delete":
            if not path.exists():
                raise ReplayError(f"cannot delete missing file: {path}")
            path.unlink()
        elif operation.kind == "update":
            _apply_update(path, operation.lines)
            if operation.move_to is not None:
                destination = _workspace_path(operation.move_to, cwd=cwd, patch_roots=patch_roots)
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
        else:
            raise ReplayError(f"unsupported patch kind: {operation.kind!r}")


def _claude_write(action: AtifAction, *, cwd: Path, patch_roots: tuple[Path, ...]) -> dict[str, Any]:
    path, content = action.arguments.get("file_path"), action.arguments.get("content")
    if not isinstance(path, str) or not isinstance(content, str):
        raise ReplayError(f"ATIF action {action.index} Write requires string file_path and content")
    destination = _workspace_path(path, cwd=cwd, patch_roots=patch_roots)
    with destination.open("w", encoding="utf-8", newline="") as target:
        target.write(content)
    return {"status": "ok", "returncode": 0, "path": str(destination), "bytes_written": len(content.encode())}


def _claude_read(action: AtifAction, *, cwd: Path, patch_roots: tuple[Path, ...]) -> dict[str, Any]:
    path, offset, limit = (
        action.arguments.get("file_path"),
        action.arguments.get("offset", 1),
        action.arguments.get("limit", 2000),
    )
    if not isinstance(path, str):
        raise ReplayError(f"ATIF action {action.index} Read requires string file_path")
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 1
        or not isinstance(limit, int)
        or isinstance(limit, bool)
        or limit < 1
    ):
        raise ReplayError(f"ATIF action {action.index} Read offset and limit must be positive integers")
    path = _workspace_path(path, cwd=cwd, patch_roots=patch_roots)
    if not path.is_file():
        raise ReplayError(f"cannot read missing file: {path}")
    source = path.read_bytes()
    try:
        text = source.decode("utf-8")
    except UnicodeDecodeError:
        content = source
        lines_read = None
    else:
        content = "".join(re.findall(r"[^\n]*\n|[^\n]+", text)[offset - 1 : offset - 1 + limit]).encode()
        lines_read = len(content.split(b"\n")) - int(content.endswith(b"\n")) if content else 0
    result = {
        "status": "ok",
        "returncode": 0,
        "path": str(path),
        "offset": offset,
        "limit": limit,
        "lines_read": lines_read,
        "bytes_read": len(content),
        "content_sha256": hashlib.sha256(content).hexdigest(),
    }
    if lines_read is None:
        result["binary"] = True
    return result


def _claude_search_root(action: AtifAction, *, cwd: Path, patch_roots: tuple[Path, ...]) -> Path:
    path = action.arguments.get("path", ".")
    if not isinstance(path, str) or "\0" in path:
        raise ReplayError(f"ATIF action {action.index} {action.source_tool} path must be a non-NUL string")
    return _workspace_path(path, cwd=cwd, patch_roots=patch_roots)


def _claude_glob(action: AtifAction, *, cwd: Path, patch_roots: tuple[Path, ...]) -> dict[str, Any]:
    pattern = action.arguments.get("pattern")
    if (
        not isinstance(pattern, str)
        or not pattern
        or "\0" in pattern
        or any(part == ".." for part in Path(pattern).parts)
    ):
        raise ReplayError(f"ATIF action {action.index} Glob pattern must be a nonempty non-NUL string")
    root = _claude_search_root(action, cwd=cwd, patch_roots=patch_roots)
    if not root.is_dir():
        raise ReplayError(f"Glob path is not a directory: {root}")

    def expand(pattern: str) -> list[str]:
        try:
            left, rest = pattern.split("{", 1)
            choices, right = rest.split("}", 1)
        except ValueError:
            if "{" in pattern or "}" in pattern:
                raise ReplayError("Claude Code Glob has an unclosed brace") from None
            return [pattern]
        if not choices:
            raise ReplayError("Claude Code Glob has an empty brace")
        return sum((expand(f"{left}{choice}{right}") for choice in choices.split(",")), [])

    matches = {
        path.resolve()
        for expanded in expand(pattern)
        for path in root.glob(expanded)
        if path.is_file() and _workspace_path(str(path), cwd=cwd, patch_roots=patch_roots)
    }
    matches = sorted(matches, key=lambda path: (-path.stat().st_mtime, str(path)))
    return {
        "status": "ok",
        "returncode": 0,
        "path": str(root),
        "pattern": pattern,
        "matches": [str(path) for path in matches[:100]],
        "truncated": len(matches) > 100,
    }


def _claude_grep(
    action: AtifAction, *, cwd: Path, patch_roots: tuple[Path, ...], timeout_sec: float | None = None
) -> dict[str, Any]:
    pattern = action.arguments.get("pattern")
    output_mode = action.arguments.get("output_mode", "files_with_matches")
    glob = action.arguments.get("glob")
    if (
        not isinstance(pattern, str)
        or "\0" in pattern
        or output_mode not in {"content", "files_with_matches", "count"}
        or glob is not None
        and (not isinstance(glob, str) or "\0" in glob)
    ):
        raise ReplayError(f"ATIF action {action.index} Grep has invalid pattern, glob, or output_mode")
    offset, head_limit = action.arguments.get("offset", 0), action.arguments.get("head_limit")
    if (
        isinstance(offset, bool)
        or not isinstance(offset, int)
        or offset < 0
        or head_limit is not None
        and (isinstance(head_limit, bool) or not isinstance(head_limit, int) or head_limit < 0)
    ):
        raise ReplayError(f"ATIF action {action.index} Grep offset and head_limit must be non-negative integers")
    root = _claude_search_root(action, cwd=cwd, patch_roots=patch_roots)
    if not root.is_dir() and not root.is_file():
        raise ReplayError(f"Grep path does not exist: {root}")
    mode_flag = {"content": "--line-number", "files_with_matches": "--files-with-matches", "count": "--count"}[
        output_mode
    ]
    command = ["rg", "--color", "never", "--no-messages", mode_flag]
    case_insensitive = action.arguments.get("-i", False)
    if not isinstance(case_insensitive, bool):
        raise ReplayError("Claude Code Grep -i must be a boolean")
    if case_insensitive:
        command.append("--ignore-case")
    if action.arguments.get("multiline") is True:
        command.append("--multiline")
    for key, flag in (("glob", "--glob"), ("type", "--type")):
        if isinstance(action.arguments.get(key), str):
            command += [flag, action.arguments[key]]
    try:
        result = subprocess.run(
            command + ["--", pattern, str(root)], cwd=cwd, capture_output=True, text=True, timeout=timeout_sec
        )
    except subprocess.TimeoutExpired:
        return {
            "status": "timed_out",
            "returncode": 124,
            "path": str(root),
            "output_mode": output_mode,
            "matches_sha256": hashlib.sha256(b"").hexdigest(),
            "matches": 0,
            "total_matches": None,
            "stderr_preview": "",
        }
    except OSError as exc:
        raise ReplayError(f"ATIF action {action.index} Grep could not run ripgrep: {exc}") from exc
    ok = result.returncode <= 1
    matches = result.stdout.splitlines()
    total_matches = (
        sum(int(line.rsplit(":", 1)[1]) for line in matches if ":" in line) if output_mode == "count" else None
    )
    matches = matches[offset : None if head_limit is None else offset + head_limit]
    rendered = "\n".join(matches)
    return {
        "status": "ok" if ok else "failed",
        "returncode": result.returncode,
        "path": str(root),
        "output_mode": output_mode,
        "matches_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "matches": len(matches),
        "total_matches": total_matches,
        "stderr_preview": _preview(result.stderr),
    }


def _claude_grep_input_validation_parameter(action: AtifAction) -> str | None:
    recognized = {"pattern", "path", "glob", "output_mode", "offset", "head_limit", "multiline", "type", "-i"}
    prefix = (
        "<tool_use_error>InputValidationError: Grep failed due to the following issue:\n" "An unexpected parameter `"
    )
    suffix = "` was provided</tool_use_error>\n\n[error] tool reported failure"
    text = action.observation_text
    if not text.startswith(prefix) or not text.endswith(suffix):
        return None
    parameter = text[len(prefix) : -len(suffix)]
    unsupported = [name for name in action.arguments if name not in recognized]
    return parameter if unsupported == [parameter] else None


def _claude_edit(action: AtifAction, *, cwd: Path, patch_roots: tuple[Path, ...]) -> dict[str, Any]:
    path = action.arguments.get("file_path")
    old, new = action.arguments.get("old_string"), action.arguments.get("new_string")
    replace_all = action.arguments.get("replace_all")
    if (
        not isinstance(path, str)
        or not isinstance(old, str)
        or not isinstance(new, str)
        or not isinstance(replace_all, bool)
    ):
        raise ReplayError(f"ATIF action {action.index} Edit has invalid arguments")
    if not old:
        raise ReplayError(f"ATIF action {action.index} Edit old_string is empty")
    destination = _workspace_path(path, cwd=cwd, patch_roots=patch_roots)
    if not destination.is_file():
        raise ReplayError(f"cannot edit missing file: {destination}")
    with destination.open(encoding="utf-8", newline="") as target:
        source = target.read()
    count = source.count(old)
    if not count or (count > 1 and not replace_all):
        raise ReplayError(f"Edit replacement is not unique in {destination}")
    with destination.open("w", encoding="utf-8", newline="") as target:
        target.write(source.replace(old, new, -1 if replace_all else 1))
    return {"status": "ok", "returncode": 0, "path": str(destination)}


def _session_yield_sec(arguments: dict[str, Any]) -> float:
    value = arguments.get("yield_time_ms")
    if isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0:
        return float(value) / 1000.0
    return 0.0


def _command_cwd(action: AtifAction, *, default_cwd: Path) -> Path:
    requested = action.arguments.get("workdir")
    if requested is None:
        return default_cwd
    if not isinstance(requested, str):
        raise ReplayError(f"ATIF action {action.index} workdir is not a string")
    workdir = Path(requested)
    if not workdir.is_absolute():
        workdir = default_cwd / workdir
    if not workdir.is_dir():
        raise ReplayError(f"ATIF action {action.index} workdir does not exist: {workdir}")
    return workdir


def _terminate_session(session: _RunningSession) -> None:
    try:
        if session.process.poll() is None or _process_group_exists(session.process.pid):
            _terminate_and_verify_process_group(
                session.process,
                wait_for_leader=lambda timeout: session.process.wait(timeout=timeout),
            )
    finally:
        if session.process.stdin is not None:
            session.process.stdin.close()
        try:
            if session.terminal is not None:
                session.terminal.close()
        finally:
            session.log_handle.close()


def _process_group_exists(process_group_id: int) -> bool:
    """Return whether the session-scoped process group still has members."""
    try:
        os.killpg(process_group_id, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        # Treat an inaccessible group as live so cleanup never reports success
        # merely because the liveness probe could not inspect it.
        return True
    return True


def _proc_identity(pid: int) -> tuple[int, int] | None:
    """Return ``(parent_pid, start_ticks)`` from one Linux procfs stat row."""
    try:
        text = Path(f"/proc/{pid}/stat").read_text()
    except (FileNotFoundError, ProcessLookupError):
        return None
    closing = text.rfind(")")
    if closing < 0:
        return None
    fields = text[closing + 2 :].split()
    if len(fields) <= 19:
        return None
    try:
        return int(fields[1]), int(fields[19])
    except ValueError:
        return None


def _proc_process_snapshot() -> dict[int, tuple[int, int]] | None:
    snapshot: dict[int, tuple[int, int]] = {}
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return None
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        try:
            identity = _proc_identity(pid)
        except OSError:
            return None
        if identity is not None:
            snapshot[pid] = identity
    return snapshot


class _DescendantPidfds:
    """Track one live root's observed descendants without numeric-PID signals."""

    def __init__(self, root_pid: int):
        self.root_pid = root_pid
        try:
            root_identity = _proc_identity(root_pid)
        except OSError:
            root_identity = None
        self.root_start_ticks = root_identity[1] if root_identity is not None else None
        self.processes: dict[int, tuple[int, int]] = {}
        self.error: str | None = (
            None if root_identity is not None else f"unable to read procfs identity for action root {root_pid}"
        )

    def _record_error(self, message: str) -> None:
        if self.error is None:
            self.error = message

    def refresh(self, signal_number: int | None = None) -> None:
        snapshot = _proc_process_snapshot()
        if snapshot is None:
            self._record_error("unable to enumerate /proc for descendant containment")
            return
        children: dict[int, list[int]] = {}
        for pid, (parent_pid, _start_ticks) in snapshot.items():
            children.setdefault(parent_pid, []).append(pid)

        roots: list[int] = []
        root_identity = snapshot.get(self.root_pid)
        if (
            root_identity is not None
            and self.root_start_ticks is not None
            and root_identity[1] == self.root_start_ticks
        ):
            roots.append(self.root_pid)
        for pid, (start_ticks, pidfd) in tuple(self.processes.items()):
            identity = snapshot.get(pid)
            if identity is not None and identity[1] == start_ticks and self._pidfd_alive(pidfd):
                roots.append(pid)

        queue = list(dict.fromkeys(roots))
        seen = set(queue)
        discovered: list[int] = []
        while queue:
            parent_pid = queue.pop()
            for pid in children.get(parent_pid, []):
                if pid in seen:
                    continue
                seen.add(pid)
                queue.append(pid)
                discovered.append(pid)

        for pid in discovered:
            if pid in self.processes:
                continue
            before = snapshot.get(pid)
            if before is None:
                continue
            try:
                pidfd = os.pidfd_open(pid)
            except AttributeError:
                self._record_error("os.pidfd_open is unavailable")
                continue
            except ProcessLookupError:
                continue
            except OSError as exc:
                self._record_error(f"unable to open pidfd for descendant {pid}: {exc}")
                continue
            try:
                after = _proc_identity(pid)
            except OSError as exc:
                os.close(pidfd)
                self._record_error(f"unable to revalidate procfs identity for descendant {pid}: {exc}")
                continue
            if after != before:
                os.close(pidfd)
                continue
            self.processes[pid] = (before[1], pidfd)
            if signal_number is not None:
                self._signal(pid, pidfd, signal_number)

    def _pidfd_alive(self, pidfd: int) -> bool:
        try:
            poller = select.poll()
            poller.register(pidfd, select.POLLIN | select.POLLHUP | select.POLLERR)
            readable = poller.poll(0)
        except (OSError, ValueError) as exc:
            self._record_error(f"unable to poll descendant pidfd: {exc}")
            return False
        return not readable

    def _signal(self, pid: int, pidfd: int, signal_number: int) -> None:
        try:
            signal.pidfd_send_signal(pidfd, signal_number)
        except AttributeError:
            self._record_error("signal.pidfd_send_signal is unavailable")
        except ProcessLookupError:
            pass
        except OSError as exc:
            self._record_error(f"unable to signal tracked descendant {pid} through pidfd: {exc}")

    def signal_all(self, signal_number: int) -> None:
        for pid, (_start_ticks, pidfd) in tuple(self.processes.items()):
            if self._pidfd_alive(pidfd):
                self._signal(pid, pidfd, signal_number)

    def all_exited(self) -> bool:
        return all(not self._pidfd_alive(pidfd) for _start_ticks, pidfd in self.processes.values())

    def close(self) -> None:
        for _start_ticks, pidfd in self.processes.values():
            try:
                os.close(pidfd)
            except OSError:
                pass
        self.processes.clear()


def _wait_for_containment_exit(
    process_group_id: int,
    descendants: _DescendantPidfds,
    timeout_sec: float,
    *,
    signal_new_descendants: int,
) -> bool:
    deadline = time.monotonic() + max(0.0, timeout_sec)
    while True:
        descendants.refresh(signal_new_descendants)
        if not _process_group_exists(process_group_id) and descendants.all_exited():
            return True
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            return False
        time.sleep(min(PROCESS_GROUP_POLL_SEC, remaining))


def _terminate_and_verify_process_group(
    process: subprocess.Popen[str],
    *,
    wait_for_leader: Callable[[float | None], _LeaderWaitResult],
) -> _LeaderWaitResult:
    """Terminate, reap, and verify the original session process group.

    Waiting for the leader alone does not prove that descendants exited.  The
    spawned leader is also the process-group id because it used
    ``start_new_session``; retain that stable id and explicitly wait for the
    group to disappear.  A descendant that deliberately calls ``setsid()`` or
    moves to another process group is retained when it remains visible in a
    procfs ancestry snapshot and is signalled through a pidfd, avoiding
    numeric-PID reuse.  Without a delegated cgroup, a descendant that
    double-forks and is reparented before the first timeout snapshot can erase
    that ancestry; unreadable procfs makes the audit fail closed.  The enclosing
    container cleanup gate remains the exact containment boundary for those
    cases.
    """
    process_group_id = process.pid
    descendants = _DescendantPidfds(process_group_id)
    descendants.refresh()
    grace_deadline = time.monotonic() + PROCESS_GROUP_TERMINATION_GRACE_SEC
    try:
        os.killpg(process_group_id, signal.SIGTERM)
    except ProcessLookupError:
        pass
    descendants.signal_all(signal.SIGTERM)

    leader_reaped = False
    leader_result: _LeaderWaitResult
    try:
        leader_result = wait_for_leader(max(0.0, grace_deadline - time.monotonic()))
        leader_reaped = True
    except subprocess.TimeoutExpired:
        pass

    group_exited = _wait_for_containment_exit(
        process_group_id,
        descendants,
        max(0.0, grace_deadline - time.monotonic()),
        signal_new_descendants=signal.SIGTERM,
    )
    if not group_exited:
        descendants.refresh(signal.SIGKILL)
        try:
            os.killpg(process_group_id, signal.SIGKILL)
        except ProcessLookupError:
            pass
        descendants.signal_all(signal.SIGKILL)
        if not leader_reaped:
            leader_result = wait_for_leader(None)
            leader_reaped = True
        if not _wait_for_containment_exit(
            process_group_id,
            descendants,
            PROCESS_GROUP_KILL_SETTLE_SEC,
            signal_new_descendants=signal.SIGKILL,
        ):
            descendants.close()
            raise ReplayError(
                "timed-out command descendant containment remained populated " "after SIGKILL: " f"{process_group_id}"
            )

    if not leader_reaped:
        leader_result = wait_for_leader(None)
    tracking_error = descendants.error
    descendants.close()
    if tracking_error is not None:
        raise ReplayError("timed-out command descendant containment was not provable: " f"{tracking_error}")
    return leader_result


def _communicate_after_timeout(
    process: subprocess.Popen[str],
) -> tuple[str | None, str | None]:
    return _terminate_and_verify_process_group(
        process,
        wait_for_leader=lambda timeout: process.communicate(timeout=timeout),
    )


def _execution_shell(action: AtifAction) -> tuple[str, str]:
    mini_swe = action.source_agent_original_format == MINI_SWE_AGENT_ORIGINAL_FORMAT and action.source_tool == "bash"
    shell = action.arguments.get("shell", "/bin/sh" if mini_swe else "/bin/bash")
    if (
        not isinstance(shell, str)
        or "\0" in shell
        or not Path(shell).is_absolute()
        or Path(shell).name not in {"bash", "sh", "dash", "zsh"}
    ):
        raise ReplayError("exec_command shell must be an absolute bash, sh, dash, or zsh path")
    login = action.arguments.get(
        "login", not mini_swe and action.source_agent_original_format != CLAUDE_CODE_AGENT_NAME
    )
    if not isinstance(login, bool):
        raise ReplayError("exec_command login must be boolean")
    tty = action.arguments.get("tty", False)
    if not isinstance(tty, bool):
        raise ReplayError("exec_command tty must be boolean")
    return shell, "-lc" if login else "-c"


def _spawn_pty(argv: list[str], cwd: Path, output: Any) -> tuple[subprocess.Popen[str], _PtyOutput]:
    master, slave = os.openpty()
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 24, 80, 0, 0))
        process = subprocess.Popen(
            argv,
            cwd=cwd,
            env=_workload_environment(),
            stdin=slave,
            stdout=slave,
            stderr=slave,
            start_new_session=True,
            preexec_fn=lambda: fcntl.ioctl(0, termios.TIOCSCTTY, 0),
            text=True,
            errors="replace",
        )
        os.set_blocking(master, False)
        terminal = _PtyOutput(master, output)
    except BaseException:
        os.close(master)
        raise
    finally:
        os.close(slave)
    return process, terminal


def _execute_command(
    action: AtifAction,
    *,
    cwd: Path,
    timeout_sec: float | None,
    expected_status: str,
    expected_session_id: str | None,
    sessions: dict[str, _RunningSession],
    artifacts_dir: Path,
    action_output_dir: Path | None,
) -> dict[str, Any]:
    command = action.arguments.get("cmd", action.arguments.get("command"))
    if not isinstance(command, str):
        raise ReplayError(f"ATIF action {action.index} exec_command has no string cmd")
    is_mini_swe = action.source_agent_original_format == MINI_SWE_AGENT_ORIGINAL_FORMAT and action.source_tool == "bash"
    shell, shell_flag = _execution_shell(action)
    tty = action.arguments.get("tty", False)
    try:
        command_cwd = _command_cwd(action, default_cwd=cwd)
    except ReplayError as exc:
        return {
            "command": command,
            "status": "failed",
            "returncode": None,
            "stderr_preview": str(exc),
            "launch_failed": True,
            "command_executed": False,
        }
    if expected_status == "running":
        if expected_session_id is None:
            raise ReplayError(f"ATIF action {action.index} has no source session id")
        log_path = artifacts_dir / f"session-{expected_session_id}.log"
        log_handle = log_path.open("w+")
        marker = "Output is being written to: "
        for line in action.observation_text.splitlines():
            if marker not in line:
                continue
            recorded_path = Path(line.split(marker, 1)[1].strip())
            if (
                not re.fullmatch(
                    r"/tmp/claude-[A-Za-z0-9._-]+(?:/[A-Za-z0-9._-]+)*/tasks/[A-Za-z0-9._-]+\.output",
                    str(recorded_path),
                )
                or ".." in recorded_path.parts
                or recorded_path.parent.resolve() != recorded_path.parent
                or recorded_path.parent.name != "tasks"
                or recorded_path.name != f"{expected_session_id}.output"
            ):
                continue
            recorded_path.parent.mkdir(parents=True, exist_ok=True)
            if recorded_path.exists() or recorded_path.is_symlink():
                raise ReplayError(f"Claude Code background task output path already exists: {recorded_path}")
            recorded_path.symlink_to(log_path.resolve())
            break
        recorded_timeout = _claude_bash_timeout(action.arguments)
        argv = [shell, shell_flag, command]
        if recorded_timeout is not None:
            argv = [
                "timeout",
                "--signal=TERM",
                "--kill-after=2s",
                f"{recorded_timeout}s",
                *argv,
            ]
        terminal = None
        try:
            if tty:
                process, terminal = _spawn_pty(argv, command_cwd, log_handle)
            else:
                process = subprocess.Popen(
                    argv,
                    cwd=command_cwd,
                    env=_workload_environment(),
                    stdin=subprocess.PIPE,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    text=True,
                    errors="replace",
                    start_new_session=True,
                )
        except BaseException:
            log_handle.close()
            raise
        session = _RunningSession(expected_session_id, process, log_path, log_handle, terminal)
        sessions[expected_session_id] = session
        yielded = _session_yield_sec(action.arguments)
        if yielded:
            try:
                process.wait(timeout=yielded)
            except subprocess.TimeoutExpired:
                pass
        returncode = process.poll()
        return {
            "command": command,
            "status": "running" if returncode is None else ("ok" if returncode == 0 else "failed"),
            "returncode": returncode,
            "process_root_pid": process.pid,
            "session_id": expected_session_id,
            "session_log": str(log_path),
        }
    # Files avoid waiting forever when a completed foreground shell leaves an
    # intentional background service holding its inherited output descriptors.
    # The next recorded action can then interact with that service as it did in
    # the source run.
    with tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace") as stdout_file:
        stderr_file = (
            stdout_file if is_mini_swe else tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
        )
        terminal = None
        try:
            if tty:
                process, terminal = _spawn_pty([shell, shell_flag, command], command_cwd, stdout_file)
            else:
                process = subprocess.Popen(
                    [shell, shell_flag, command],
                    cwd=command_cwd,
                    env=_workload_environment(),
                    text=True,
                    errors="replace",
                    stdout=stdout_file,
                    stderr=stderr_file,
                    start_new_session=True,
                )
            process_root_pid = process.pid
            timed_out = False
            recorded_timeout = (
                _claude_bash_timeout(action.arguments)
                if action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME and action.source_tool == "Bash"
                else None
            )
            effective_timeout = (
                min(timeout_sec, recorded_timeout)
                if timeout_sec is not None and recorded_timeout is not None
                else timeout_sec or recorded_timeout
            )
            try:
                process.communicate(timeout=effective_timeout)
            except subprocess.TimeoutExpired:
                timed_out = True
                _communicate_after_timeout(process)
            if terminal is not None:
                stdout = terminal.snapshot()
            else:
                stdout_file.seek(0)
                stdout = stdout_file.read()
            if is_mini_swe:
                stderr = None
            else:
                stderr_file.seek(0)
                stderr = stderr_file.read()
        finally:
            if terminal is not None:
                terminal.close()
            if not is_mini_swe:
                stderr_file.close()
    stdout = stdout or ""
    stderr = stderr or ""
    output_paths: dict[str, str] = {}
    if action_output_dir is not None:
        action_output_dir.mkdir(parents=True, exist_ok=True)
        if is_mini_swe:
            output_path = action_output_dir / f"action-{action.index:04d}.output.txt"
            output_path.write_text(stdout)
            output_paths["output_path"] = str(output_path)
        else:
            stdout_path = action_output_dir / f"action-{action.index:04d}.stdout.txt"
            stderr_path = action_output_dir / f"action-{action.index:04d}.stderr.txt"
            stdout_path.write_text(stdout)
            stderr_path.write_text(stderr)
            output_paths.update({"stdout_path": str(stdout_path), "stderr_path": str(stderr_path)})
    if timed_out:
        return {
            "command": command,
            "status": "timed_out",
            "process_root_pid": process_root_pid,
            "stdout_preview": _preview(stdout),
            "stderr_preview": _preview(stderr),
            **output_paths,
        }
    return {
        "command": command,
        "status": "ok" if process.returncode == 0 else "failed",
        "returncode": process.returncode,
        "process_root_pid": process_root_pid,
        "stdout_preview": _preview(stdout),
        "stderr_preview": _preview(stderr),
        **output_paths,
    }


def _write_stdin(
    action: AtifAction,
    *,
    timeout_sec: float | None,
    expected_status: str,
    sessions: dict[str, _RunningSession],
) -> dict[str, Any]:
    source_id = action.arguments.get("session_id")
    if not isinstance(source_id, (str, int)):
        raise ReplayError(f"ATIF action {action.index} write_stdin has no session_id")
    session_id = str(source_id)
    session = sessions.get(session_id)
    if session is None:
        raise ReplayError(f"ATIF action {action.index} references unknown session {session_id}")
    chars = action.arguments.get("chars", "")
    if not isinstance(chars, str):
        raise ReplayError(f"ATIF action {action.index} write_stdin chars are not a string")
    process = session.process
    input_started = time.monotonic()
    input_timed_out = False
    interrupted = chars == "\x03"
    if chars and session.terminal is not None and process.poll() is None:
        deadline = input_started + (timeout_sec if timeout_sec is not None else DEFAULT_WRITE_STDIN_YIELD_SEC)
        try:
            data = chars.encode("utf-8")
            while data:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    input_timed_out = True
                    break
                _, writable, _ = select.select([], [session.terminal.master], [], remaining)
                if not writable:
                    input_timed_out = True
                    break
                try:
                    data = data[os.write(session.terminal.master, data) :]
                except BlockingIOError:
                    continue
        except OSError as exc:
            if exc.errno != errno.EIO:
                raise
    elif interrupted and process.poll() is None:
        # Codex sessions are PTYs, where a lone Ctrl-C becomes SIGINT.  This
        # replay session uses a pipe, so writing byte 0x03 would do nothing.
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
    elif chars and process.poll() is None and process.stdin is not None:
        try:
            process.stdin.write(chars)
            process.stdin.flush()
        except BrokenPipeError:
            pass
    del expected_status
    # A write_stdin poll observes the owned session for its recorded yield. The
    # source terminal state is a comparison target, not permission to kill a
    # replay session whose timing differs and abort all later actions.
    yielded = _session_yield_sec(action.arguments) or DEFAULT_WRITE_STDIN_YIELD_SEC
    remaining_timeout = max(0.0, timeout_sec - (time.monotonic() - input_started)) if timeout_sec is not None else None
    wait = min(yielded, remaining_timeout) if remaining_timeout is not None else yielded
    try:
        if input_timed_out:
            raise subprocess.TimeoutExpired(process.args, timeout_sec)
        process.wait(timeout=wait)
    except subprocess.TimeoutExpired:
        if input_timed_out or (remaining_timeout is not None and remaining_timeout <= yielded):
            _terminate_session(session)
            return {
                "status": "timed_out",
                "returncode": 124,
                "process_root_pid": process.pid,
                "session_id": session_id,
                "session_log": str(session.log_path),
            }
    returncode = process.poll()
    if interrupted and returncode == -signal.SIGINT:
        returncode = 128 + signal.SIGINT
    status = "running" if returncode is None else ("ok" if returncode == 0 else "failed")
    return {
        "status": status,
        "returncode": returncode,
        "process_root_pid": process.pid,
        "session_id": session_id,
        "session_log": str(session.log_path),
    }


def _claude_background_session(action: AtifAction, sessions: dict[str, _RunningSession]) -> tuple[str, _RunningSession]:
    task_id = action.arguments.get("task_id")
    if not isinstance(task_id, str) or not task_id:
        raise ReplayError(f"ATIF action {action.index} {action.source_tool} has no string task_id")
    session = sessions.get(task_id)
    if session is None:
        raise ReplayError(f"ATIF action {action.index} references unknown Claude background task {task_id!r}")
    return task_id, session


def _claude_task_output(
    action: AtifAction, sessions: dict[str, _RunningSession], timeout_sec: float | None
) -> dict[str, Any]:
    """Observe only the Bash process named by Claude's recorded backgroundTaskId."""
    task_id, session = _claude_background_session(action, sessions)
    block = action.arguments.get("block", False)
    timeout = action.arguments.get("timeout")
    if not isinstance(block, bool):
        raise ReplayError("Claude Code TaskOutput block must be boolean")
    if timeout is not None and (not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0):
        raise ReplayError("Claude Code TaskOutput timeout must be a positive number")
    wait = (
        min(timeout / 1000, timeout_sec)
        if timeout is not None and timeout_sec is not None
        else (timeout / 1000 if timeout is not None else timeout_sec)
    )
    try:
        returncode = (
            session.process.wait(timeout=wait if wait is not None else DEFAULT_WRITE_STDIN_YIELD_SEC)
            if block
            else session.process.poll()
        )
    except subprocess.TimeoutExpired:
        returncode = None
    output = session.log_path.read_bytes()
    return {
        "status": "running" if returncode is None else ("ok" if returncode == 0 else "failed"),
        "returncode": returncode,
        "session_id": task_id,
        "session_log": str(session.log_path),
        "output_sha256": hashlib.sha256(output).hexdigest(),
    }


def _claude_task_stop(action: AtifAction, sessions: dict[str, _RunningSession]) -> dict[str, Any]:
    """Stop only the Bash process named by Claude's recorded backgroundTaskId."""
    task_id, session = _claude_background_session(action, sessions)
    _terminate_session(session)
    sessions.pop(task_id)
    return {
        "status": "ok",
        "returncode": 0,
        "session_id": task_id,
    }


def replay_trajectory(
    trajectory: AtifTrajectory,
    *,
    cwd: Path,
    pacing: str = "recorded",
    timeout_sec: float | None = None,
    artifacts_dir: Path | None = None,
    action_output_dir: Path | None = None,
    patch_roots: tuple[Path, ...] = (),
) -> dict[str, Any]:
    """Replay supported ATIF actions and compare observed tool outcomes.

    Outcome matching is the fidelity gate consumed by the reporting pipeline:
    a replay with divergent command/patch outcomes may still be useful as a
    diagnostic observation, but is not accepted into comparable charts.
    """
    validate_supported_actions(trajectory)
    if pacing not in PACING_MODES:
        raise ReplayError(f"unsupported pacing mode: {pacing!r}")
    if not cwd.is_dir():
        raise ReplayError(f"replay working directory does not exist: {cwd}")
    if timeout_sec is not None and timeout_sec <= 0:
        raise ReplayError("action timeout must be positive")
    if artifacts_dir is None:
        artifacts_dir = Path(tempfile.mkdtemp(prefix="atif-replay-sessions-"))
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    executable = [
        action
        for action in trajectory.actions
        if action.tool in SUPPORTED_EXECUTING_TOOLS and _source_outcome(action)[0] != "skipped_not_executed"
    ]
    if not executable:
        raise ReplayError("ATIF trajectory has no source-executed supported actions to replay")
    first_timestamp = executable[0].timestamp_sec
    source_end = executable[-1].timestamp_sec
    replay_started = time.monotonic()
    # For ``llm-gaps`` pacing: the recorded finish epoch of the previous executed
    # action (its issue timestamp plus its recorded tool wall time) and the
    # monotonic time it actually finished offline. The model round-trip is then
    # re-injected relative to the real offline finish, keeping the tool half of
    # each turn at its true speed.
    prev_source_finish: float | None = None
    prev_replay_finish: float | None = None
    results: list[dict[str, Any]] = []
    sessions: dict[str, _RunningSession] = {}
    terminated_session_ids: list[str] = []
    aborted_after_timeout_action_index: int | None = None
    try:
        for action in trajectory.actions:
            expected_status, expected_returncode, expected_session_id = _source_outcome(action)
            result: dict[str, Any] = {
                "action_index": action.index,
                "step_id": action.step_id,
                "tool": action.tool,
                "source_tool": action.source_tool,
                "code_mode": action.code_mode,
                "source_status": expected_status,
                "source_returncode": expected_returncode,
                "source_trajectory_id": action.source_trajectory_id,
            }
            if action.parallel_group_id is not None:
                result["parallel_group_id"] = action.parallel_group_id
                result["parallel_execution"] = "sequentialized_for_per_command_cpu"
            if _is_metadata_action(action):
                result["status"] = "ignored_metadata"
                result["outcome_matches_source"] = True
                results.append(result)
                continue
            if expected_status == "skipped_not_executed":
                result["status"] = "skipped_not_executed"
                result["outcome_matches_source"] = True
                results.append(result)
                continue
            source_offset_sec = action.timestamp_sec - first_timestamp
            result["source_start_offset_sec"] = round(source_offset_sec, 9)
            if pacing == "recorded":
                sleep_sec = source_offset_sec - (time.monotonic() - replay_started)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            elif pacing == "llm-gaps" and prev_source_finish is not None:
                llm_gap_sec = max(0.0, action.timestamp_sec - prev_source_finish)
                result["source_llm_gap_sec"] = round(llm_gap_sec, 9)
                sleep_sec = llm_gap_sec - (time.monotonic() - prev_replay_finish)
                if sleep_sec > 0:
                    time.sleep(sleep_sec)
            action_started = time.monotonic()
            action_clock_sample = bounded_realtime_monotonic_sample()
            action_started_ns = action_clock_sample["monotonic_ns"]
            if _source_argument_parse_failed(action):
                outcome = {
                    "command": None,
                    "status": "failed",
                    "returncode": None,
                    "source_argument_parse_failed": True,
                    "command_executed": False,
                    "source_error_preview": _preview(action.observation_text),
                }
            elif action.tool == "exec_command":
                outcome = _execute_command(
                    action,
                    cwd=cwd,
                    timeout_sec=timeout_sec,
                    expected_status=expected_status,
                    expected_session_id=expected_session_id,
                    sessions=sessions,
                    artifacts_dir=artifacts_dir,
                    action_output_dir=action_output_dir,
                )
            elif action.tool == "write_stdin":
                outcome = _write_stdin(
                    action,
                    timeout_sec=timeout_sec,
                    expected_status=expected_status,
                    sessions=sessions,
                )
            elif action.tool == "claude_task_output":
                outcome = _claude_task_output(action, sessions, timeout_sec)
            elif action.tool == "claude_task_stop":
                outcome = _claude_task_stop(action, sessions)
            elif action.tool == "apply_patch":
                patch = action.arguments.get("input", action.arguments.get("patch"))
                if not isinstance(patch, str):
                    raise ReplayError(f"ATIF action {action.index} apply_patch has no string input")
                try:
                    apply_patch_payload(patch, cwd=cwd, patch_roots=patch_roots)
                    outcome = {"status": "ok", "returncode": 0}
                except ReplayError as exc:
                    outcome = {"status": "failed", "returncode": 1, "stderr_preview": str(exc)}
            elif (
                action.tool == "claude_grep"
                and expected_status == "failed"
                and (parameter := _claude_grep_input_validation_parameter(action))
            ):
                outcome = {
                    "status": "failed",
                    "returncode": None,
                    "command_executed": False,
                    "source_input_validation_reproduced": True,
                    "unsupported_arguments": [parameter],
                }
            elif action.tool in {"claude_write", "claude_edit", "claude_read", "claude_glob", "claude_grep"}:
                try:
                    operation = {
                        "claude_write": _claude_write,
                        "claude_edit": _claude_edit,
                        "claude_read": _claude_read,
                        "claude_glob": _claude_glob,
                        "claude_grep": _claude_grep,
                    }[action.tool]
                    kwargs = {"cwd": cwd, "patch_roots": patch_roots}
                    if action.tool == "claude_grep":
                        kwargs["timeout_sec"] = timeout_sec
                    outcome = operation(action, **kwargs)
                except ReplayError as exc:
                    outcome = {"status": "failed", "returncode": 1, "stderr_preview": str(exc)}
            else:
                raise ReplayError(f"unexpected supported action tool: {action.tool!r}")
            result.update(outcome)
            result["wall_time_sec"] = round(time.monotonic() - action_started, 9)
            result["replay_start_monotonic_ns"] = action_started_ns
            result["replay_start_realtime_ns"] = action_clock_sample["realtime_ns"]
            result["replay_start_realtime_uncertainty_ns"] = action_clock_sample["realtime_uncertainty_ns"]
            result["replay_end_monotonic_ns"] = time.monotonic_ns()
            # Recorded finish = issue timestamp + recorded tool wall; anchors the
            # next turn's llm-gap to the real offline finish of this one.
            prev_source_finish = action.timestamp_sec + _recorded_tool_wall_sec(action)
            prev_replay_finish = time.monotonic()
            if expected_status == "running" and result.get("status") in {"ok", "failed"}:
                result["completion_timing_drift"] = "completed_before_recorded_poll"
            background_unverified = (
                action.source_agent_original_format == CLAUDE_CODE_AGENT_NAME
                and action.source_tool == "Bash"
                and action.arguments.get("run_in_background") is True
            )
            result["outcome_matches_source"] = (
                None
                if background_unverified
                else _outcome_matches(
                    expected_status, expected_returncode, str(result["status"]), result.get("returncode")
                )
            )
            if result.get("launch_failed") is True:
                result["outcome_matches_source"] = None
                result["source_outcome_comparison_unavailable"] = True
            if background_unverified:
                result["source_outcome_comparison_unavailable"] = True
            results.append(result)
            if result.get("status") == "timed_out":
                aborted_after_timeout_action_index = action.index
                break
    finally:
        for session_id, session in sessions.items():
            if session.process.poll() is None or _process_group_exists(session.process.pid):
                terminated_session_ids.append(session_id)
            _terminate_session(session)

    if action_output_dir is None:
        for result in results:
            for field in (
                "stdout_preview",
                "stderr_preview",
                "session_output_preview",
                "source_error_preview",
            ):
                result.pop(field, None)

    for result in results:
        result["status_matches_source"] = (
            None
            if result["source_status"] == "unknown" or result.get("launch_failed") is True
            else result["source_status"] == result["status"]
        )
    reproduced_without_execution = [
        result
        for result in results
        if result.get("source_input_validation_reproduced") is True
        and result.get("unsupported_arguments")
        and result.get("command_executed") is False
        and result.get("source_status") == result.get("status") == "failed"
        and result.get("outcome_matches_source") is True
    ]
    replayed = [
        result
        for result in results
        if (result.get("command_executed") is not False or result in reproduced_without_execution)
        and result.get("status") in {"ok", "failed", "timed_out", "running"}
    ]
    mismatches = [result["action_index"] for result in replayed if result.get("outcome_matches_source") is False]
    comparison_unavailable = [
        result["action_index"]
        for result in results
        if result.get("outcome_matches_source") is None
        and (result in replayed or result.get("source_outcome_comparison_unavailable"))
    ]
    actions_executed = sum(
        result.get("command_executed") is not False and result.get("status") in {"ok", "failed", "timed_out", "running"}
        for result in results
    )
    actions_skipped_after_timeout = (
        sum(
            action.index > aborted_after_timeout_action_index
            and action.tool in SUPPORTED_EXECUTING_TOOLS
            and _source_outcome(action)[0] != "skipped_not_executed"
            for action in trajectory.actions
        )
        if aborted_after_timeout_action_index is not None
        else 0
    )
    wall_time_sec = time.monotonic() - replay_started
    return {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "source_trajectory": str(trajectory.source_path.resolve()),
        "source_schema_version": trajectory.schema_version,
        "source_agent_original_format": trajectory.source_agent_original_format,
        "source_session_id": trajectory.session_id,
        "subagent_trajectories_flattened": trajectory.subagent_trajectories_flattened,
        "agent": trajectory.agent,
        "task_name": trajectory.task_name,
        "source_trial": trajectory.trial_name,
        "pacing": pacing,
        "action_timeout_sec": timeout_sec,
        "clock_reference": monotonic_clock_reference(),
        "actions": results,
        "limitations": [
            "ATIF replay executes supported recorded tool calls only; it does not invoke the model.",
            "Replay stops after the first action timeout because later state is no longer comparable.",
            "Source-outcome matching compares recorded tool completion states and return codes where available.",
        ],
        "summary": {
            "actions_recorded": len(trajectory.actions),
            "actions_executed": actions_executed,
            "tool_actions_recorded": len(executable),
            "tool_actions_replayed": len(replayed),
            **(
                {"tool_actions_reproduced_without_execution": len(reproduced_without_execution)}
                if reproduced_without_execution
                else {}
            ),
            "tool_action_status_equal": sum(result["status_matches_source"] is True for result in replayed),
            "tool_action_status_different": sum(result["status_matches_source"] is False for result in replayed),
            "tool_action_status_unavailable": sum(result["status_matches_source"] is None for result in replayed),
            "tool_action_outcome_equal": sum(result["outcome_matches_source"] is True for result in replayed),
            "tool_action_outcome_different": sum(result["outcome_matches_source"] is False for result in replayed),
            "tool_action_outcome_unavailable": sum(result["outcome_matches_source"] is None for result in replayed),
            "tool_action_status_comparable": sum(result["status_matches_source"] is not None for result in replayed),
            "tool_action_outcome_comparable": sum(result["outcome_matches_source"] is not None for result in replayed),
            "actions_skipped_after_timeout": actions_skipped_after_timeout,
            "aborted_after_timeout_action_index": aborted_after_timeout_action_index,
            "metadata_actions_ignored": sum(_is_metadata_action(action) for action in trajectory.actions),
            "actions_skipped_not_executed": sum(result.get("status") == "skipped_not_executed" for result in results),
            "actions_ok": sum(result.get("status") == "ok" for result in results),
            "actions_failed": sum(result.get("status") == "failed" for result in results),
            "actions_launch_failed": sum(result.get("launch_failed") is True for result in results),
            "actions_source_argument_parse_failed": sum(
                bool(result.get("source_argument_parse_failed")) for result in results
            ),
            "actions_timed_out": sum(result.get("status") == "timed_out" for result in results),
            "source_outcome_mismatch_action_indexes": mismatches,
            "source_outcome_mismatch_count": len(mismatches),
            "source_outcome_comparison_unavailable_action_indexes": comparison_unavailable,
            "source_outcome_comparison_unavailable_count": len(comparison_unavailable),
            "actions_completed_before_recorded_poll": sum(
                result.get("completion_timing_drift") == "completed_before_recorded_poll" for result in results
            ),
            "sessions_terminated_at_end": terminated_session_ids,
            "source_interval_sec": round(source_end - first_timestamp, 9),
            "wall_time_sec": round(wall_time_sec, 9),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Replay supported tool actions from an ATIF trajectory.")
    parser.add_argument("trajectory", type=Path, help="ATIF trajectory.json path.")
    parser.add_argument("--output", type=Path, required=True, help="Replay report JSON path.")
    parser.add_argument("--cwd", type=Path, default=Path.cwd(), help="Replay working directory.")
    parser.add_argument(
        "--patch-root",
        action="append",
        default=[],
        type=Path,
        help="Additional writable path root permitted for recorded apply_patch actions.",
    )
    parser.add_argument("--pacing", choices=PACING_MODES, default="recorded")
    parser.add_argument("--timeout-sec", type=float, help="Per-command/session completion timeout.")
    parser.add_argument(
        "--action-output-dir",
        type=Path,
        help="Optional directory for full source-equivalent action output artifacts.",
    )
    args = parser.parse_args(argv)
    if args.timeout_sec is not None and args.timeout_sec <= 0:
        parser.error("--timeout-sec must be positive")
    try:
        report = replay_trajectory(
            read_trajectory(args.trajectory),
            cwd=args.cwd,
            pacing=args.pacing,
            timeout_sec=args.timeout_sec,
            artifacts_dir=args.output.parent / "session_logs",
            action_output_dir=args.action_output_dir,
            patch_roots=tuple(args.patch_root),
        )
    except (OSError, ReplayError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    summary = report["summary"]
    print(f"Task: {report['task_name']}")
    print(f"Source trial: {report['source_trial']}")
    print(
        "Actions: "
        f"{summary['actions_executed']}/{summary['actions_recorded']} executed, "
        f"{summary['actions_timed_out']} timed out, "
        f"{summary['source_outcome_mismatch_count']} source-outcome mismatches, "
        f"{summary['source_outcome_comparison_unavailable_count']} "
        "source-outcome comparisons unavailable"
    )
    print(f"Wall time: {summary['wall_time_sec']:.9f} sec")
    print(f"Report: {args.output}")
    return int(
        summary["source_outcome_mismatch_count"] > 0
        or summary["source_outcome_comparison_unavailable_count"] > 0
        or summary["actions_launch_failed"] > 0
        or summary["actions_timed_out"] > 0
        or summary["actions_skipped_after_timeout"] > 0
    )


if __name__ == "__main__":
    sys.exit(main())
