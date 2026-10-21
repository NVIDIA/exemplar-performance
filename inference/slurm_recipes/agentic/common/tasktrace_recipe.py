#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Python half of the agentic (TaskTrace) recipes' launch scripts.

The launch scripts are bash; everything that needs JSON, hashing or HTTP lives
here as a subcommand so it can be unit-tested and shared by both recipes:

  cpuset            expand a taskset range list and pick the first N CPUs
  image-cache-key   the file name tasktrace-capture's enroot image cache expects
  resolve-trace     map a selector (MODEL_SIZE) to a trajectory's trace.json and
                    print the shell variables the launch scripts need
  endpoint-check    one-token request against the live endpoint (key from $TT_API_KEY)
  stage             copy a captured trajectory into the trajectories-tree layout,
                    validate it and write provenance.json (inputs from $TT_* variables)
  secret-scan       redact the API key and nvapi- patterns from a run directory
  summarize-capture write summary.json for the capture + replay recipe
  summarize-replay  write summary.json for the replay-only recipe
  verdict           styled PASS/FAIL report (plus the reasons on FAIL) for one run, or a
                    table for every run under a directory; exits non-zero on any FAIL
  image-file-name   the .sqsh name llmb-install gives a container.images entry
  server-info       write server.json describing the vLLM server a *-gpu run brought up
                    (inputs from $TT_SERVER_* variables)
  llm-replay        replay the LLM side of a trajectory against a live OpenAI-compatible
                    server with synthetic prompts of the recorded token shapes (*-gpu recipes)

Standard library only; it runs with the compute node's python3.
"""

import argparse
import hashlib
import json
import os
import pathlib
import random
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
import urllib.error
import urllib.request

ACCEPTED_SCHEMAS = {"tasktrace-cpu-trace.v1", "riptide-cpu-trace.v1", "tasktrace-capture-cpu-trace.v1"}
# Name of a staged/shipped trajectory in the trajectories tree. tasktrace-capture writes its
# finalized trace as tasktrace_capture_replay_trajectory.jsonl (CAPTURE_TRAJECTORY_NAME); stage
# renames it on the way in.
TRAJECTORY_NAME = "trajectory.jsonl"
CAPTURE_TRAJECTORY_NAME = "tasktrace_capture_replay_trajectory.jsonl"
TRACE_MANIFEST = "trace.json"
NOT_EXECUTED = "action was not executed"
DEFAULT_CPUS_PER_TASK = 8  # what both launch scripts request via #SBATCH --cpus-per-task


# --------------------------------------------------------------------------- helpers


def read_json(path):
    """The JSON object at `path`; {} when the file is missing, unreadable, malformed or not an object."""
    try:
        data = json.loads(pathlib.Path(path).read_text())
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _num(value):
    """A number from JSON, or None when the field is missing or not numeric."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return value
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def kv_file(path):
    out = {}
    try:
        for line in pathlib.Path(path).read_text().splitlines():
            key, _, value = line.partition("=")
            if key:
                out[key] = value
    except OSError:
        pass
    return out


def deep_get(d, *keys):
    for key in keys:
        if not isinstance(d, dict):
            return None
        d = d.get(key)
    return d


def find_replay_report(profile_dir):
    for candidate in pathlib.Path(profile_dir).rglob("replay_report.json"):
        return read_json(candidate)
    return {}


def fidelity_verdict(report, replay_ran=True):
    """PASS when the replay executed every recorded action with no outcome mismatch."""
    summary = report.get("summary", {}) if report else {}
    mismatches = summary.get("source_outcome_mismatch_count")
    executed = summary.get("actions_executed")
    recorded = summary.get("actions_recorded")
    skipped = summary.get("actions_skipped_not_executed", 0)
    ok = replay_ran and bool(report) and mismatches == 0 and executed is not None and executed + skipped == recorded
    return "PASS" if ok else "FAIL"


def tool_time_comparison(report, compare=True):
    """Summed per-command wall time over the actions the replay executed: the replay's and, with compare, the
    recording's and their ratio. Only the capture+replay recipes compare; the replay-only recipes report."""
    actions = [a for a in report.get("actions", []) if a.get("status") != "skipped_not_executed"]
    replay_busy = sum(float(a.get("wall_time_sec") or 0.0) for a in actions)
    if not compare:
        return {"tool_busy_replay_sec": round(replay_busy, 3)}
    live_busy = sum(float(a.get("source_wall_time_sec") or 0.0) for a in actions)
    ratio = round(replay_busy / live_busy, 4) if live_busy > 0 and actions else None
    comparison = {
        "tool_busy_live_sec": round(live_busy, 3),
        "tool_busy_replay_sec": round(replay_busy, 3),
        "replay_to_live_tool_ratio": ratio,
        "actions_compared": len(actions),
    }
    return comparison


NETWORK_LIMITATION_RE = re.compile(r"Action (\d+) performs package or network-dependent work")
COMMAND_PREVIEW = 90


def _one_line(text, limit=COMMAND_PREVIEW):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


ERROR_LINE_RE = re.compile(
    r"Error\b|error:|not found|No such file|not created|Permission denied|No module named|Failing command|"
    r"fatal:|FAILED|failed with|Traceback",
    re.IGNORECASE,
)
PREVIEW_LIMIT = 1000  # tasktrace truncates stdout/stderr previews here; a preview this long may end mid-line


def _first_line(*texts):
    """The line that tells the reader what happened.

    Tracebacks are named by their last line. Otherwise the first line matching an error pattern
    wins (the venv/apt/pip messages); failing that, the last complete line, which is where pytest,
    make and most tools print their summary."""
    for text in texts:
        text = text or ""
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if not lines:
            continue
        lines = [line.strip(" =-") or line for line in lines]  # "=== 1 skipped in 0.01s ===" -> "1 skipped in 0.01s"
        if lines[0].startswith("Traceback (most recent call last)"):
            return lines[-1]
        for line in lines:
            if ERROR_LINE_RE.search(line):
                return line
        return lines[0] if len(text) >= PREVIEW_LIMIT else lines[-1]
    return ""


ENVIRONMENT_RE = re.compile(
    r"No module named|command not found|No such file or directory|not found|Permission denied|"
    r"is not recognized|No such container",
    re.IGNORECASE,
)
FAST_FAILURE_SEC = 0.1  # a command that fails this fast never did real work: it could not start


def _mismatch_cause(action, live_output):
    """Why one non-timeout action ended differently live and in the replay, if the outputs say.

    Only one cause is claimed with confidence: the two shells resolved the environment differently
    (interpreter, PATH, cwd, installed packages), visible as a 'No module named' / 'not found' style
    error on one side only. A live failure that took under 0.1s points the same way with less
    certainty. Anything else stays unexplained rather than guessed."""
    replay_detail = _first_line(action.get("stderr_preview"), action.get("stdout_preview"))
    live_detail = _first_line(live_output) if live_output is not None else None
    env_side = None
    if (
        live_detail
        and ENVIRONMENT_RE.search(live_detail)
        and not (replay_detail and ENVIRONMENT_RE.search(replay_detail))
    ):
        env_side = "live"
    elif (
        replay_detail
        and ENVIRONMENT_RE.search(replay_detail)
        and not (live_detail and ENVIRONMENT_RE.search(live_detail))
    ):
        env_side = "replay"
    if env_side:
        return [
            f"cause: the {env_side} shell could not find a program, module or path the other shell had; "
            "the live agent's shell and the replay shell resolved PATH, the interpreter or cwd differently "
            "(known case: the capture's per-container /root mount hid the image's .bashrc, "
            "so conda was never activated live)",
            "fix: capture with tasktrace-capture that preserves the image's /root dotfiles, "
            "then re-capture; the trace records only a few env vars, so a replay cannot repair this after the fact",
        ]
    recorded = _num(action.get("source_wall_time_sec"))
    if action.get("source_status") == "failed" and action.get("status") == "ok" and recorded is not None:
        if 0 < recorded < FAST_FAILURE_SEC:
            return [
                f"cause: probably an environment difference; the live command failed within {recorded * 1000:.0f} ms "
                "and the replay ran it normally, which fits a command that resolved to a different program or cwd "
                "live (a fast content difference, e.g. grep finding nothing, would look the same)",
                "fix: compare the live tool output with the replay's (see summary.json diagnostics) "
                "and align the environments",
            ]
    return []


INTERACTIVE_RE = re.compile(
    r"--pdb\b|breakpoint\(|pdb\.set_trace|\binput\(|(^|[;&|]\s*)(read\s+-|less|vim?|nano|more)\b"
)
CLOSE_TO_LIMIT = 0.5  # recorded time above this fraction of the limit: a slower node alone explains the timeout
RETRY_HINT = "retry with RUN_CONF_TASKTRACE_TIMEOUT=300 (per-action seconds)"


def _timeout_cause(action, network_dependent, timeout, earlier_divergences):
    """Best explanation for one timed-out action, most certain cause first, plus what to do about it.

    Causes we can tell apart from the report: the trace flags the action as network-dependent;
    the recording was already close to the limit (CPU-bound, slower node); the command can wait
    for interactive input and the environment had already diverged; or a generic consequence of
    an earlier divergence. Anything else is reported as not identifiable rather than guessed."""
    recorded = _num(action.get("source_wall_time_sec"))
    command = str(action.get("command") or "")
    close_to_limit = (
        timeout is not None and timeout > 0 and recorded is not None and recorded >= CLOSE_TO_LIMIT * timeout
    )
    if network_dependent:
        return [
            "cause: trace marks this action as package/network-dependent; "
            "this node reached the network resource slower than the capture host did",
            f"fix: {RETRY_HINT}, or remove the network dependency from the task image",
        ]
    if close_to_limit:
        return [
            f"cause: the recording already used {recorded / timeout:.0%} of the limit; "
            "a slower core or fewer CPUs is enough to exceed it",
            f"fix: {RETRY_HINT}, or replay with the recorded CPU count (RUN_CONF_TASKTRACE_CPUS)",
        ]
    if INTERACTIVE_RE.search(command) and earlier_divergences:
        return [
            f"cause: the command can wait for interactive input and the environment had already diverged "
            f"at action {earlier_divergences[-1]}; it likely stopped at a prompt the recording never reached",
            "fix: resolve the earlier divergence; a longer timeout will not help",
        ]
    if earlier_divergences:
        return [
            f"cause: not identifiable from the report; the environment had already diverged at action "
            f"{earlier_divergences[-1]}, so this command may have waited on something that never happened",
            f"fix: resolve the earlier divergence first; if it still times out, {RETRY_HINT} and compare",
        ]
    return [
        "cause: not identifiable from the report (not network-flagged, recording was well under the limit, "
        "no earlier divergence); the stdout preview in replay_report.json shows how far it got",
        f"fix: {RETRY_HINT} to learn whether it completes at all; if it does, the node is slower for this command",
    ]


def live_tool_outputs(capture_dir):
    """The live run's tool outputs in execution order, from the harbor trajectory the capture wrote.

    Returns a list of (command, output) so callers can pair them with replay actions by position and
    verify by command text; an empty list when the capture left no trajectory."""
    if not capture_dir:
        return []
    for path in sorted(pathlib.Path(capture_dir).rglob("agent/trajectory.json")):
        try:
            steps = read_json(path).get("steps") or []
        except (OSError, ValueError):
            continue
        outputs = []
        for step in steps:
            calls = step.get("tool_calls") or []
            if not calls:
                continue
            command = (calls[0].get("arguments") or {}).get("command", "")
            results = (step.get("observation") or {}).get("results") or []
            content = results[0].get("content") if results and isinstance(results[0], dict) else None
            output = None
            if isinstance(content, str):
                try:
                    output = json.loads(content).get("output")
                except ValueError:
                    output = content
            outputs.append((command, output))
        if outputs:
            return outputs
    return []


def _pair_live_outputs(actions, live_outputs):
    """action_index -> live output, only where the k-th live tool call is verifiably the k-th replayed action."""
    paired = {}
    for k, a in enumerate(actions):
        if k < len(live_outputs) and live_outputs[k][0] == a.get("command"):
            paired[a.get("action_index")] = live_outputs[k][1]
    return paired


def replay_diagnostics(report, live_outputs=()):
    """Why a replay diverged, one short line per cause, derived from the replay report.

    Timed-out actions come first (they are root causes); a mismatch that follows a
    timed-out action is marked as such so a cascade reads as one problem, not three.
    A timed-out action the trace itself flags as package/network-dependent gets the
    network explanation and the two remediations. Other mismatches show what the replay
    printed and, when the capture's live tool outputs are available, what the live run
    printed; a cause line is added when the two outputs identify one."""
    if not report:
        return ["replay report not found: the replay did not run or exited before writing it"]
    summary = report.get("summary", {})
    actions = report.get("actions", [])
    timeout = _num(report.get("action_timeout_sec"))
    network_actions = {
        int(m.group(1)) for m in map(NETWORK_LIMITATION_RE.match, report.get("source_limitations") or []) if m
    }
    lines = []
    recorded, executed = _num(summary.get("actions_recorded")), _num(summary.get("actions_executed"))
    skipped = _num(summary.get("actions_skipped_not_executed")) or 0
    if None not in (recorded, executed) and executed + skipped != recorded:
        lines.append(f"replay stopped after {executed:g} of {recorded - skipped:g} executable actions")
    timed_out = []
    diverged = []  # indexes of actions whose outcome differed, in order; a later timeout may be their consequence
    for a in actions:
        idx = a.get("action_index")
        if a.get("status") != "timed_out" or a.get("outcome_matches_source"):  # a matched timeout is reproduced
            if not a.get("outcome_matches_source", True):
                diverged.append(idx)
            continue
        limit = f"{timeout:g}s" if isinstance(timeout, (int, float)) else "the action timeout"
        recorded_sec = _num(a.get("source_wall_time_sec"))
        recorded_txt = f" (recorded {recorded_sec:.1f}s)" if recorded_sec is not None else ""
        lines.append(f"action {idx} timed out after {limit}{recorded_txt}: {_one_line(a.get('command'))}")
        lines.extend(
            f"  {line}"
            for line in _timeout_cause(a, idx in network_actions, timeout, [d for d in diverged + timed_out if d < idx])
        )
        timed_out.append(idx)
        diverged.append(idx)
    live_by_index = _pair_live_outputs(actions, list(live_outputs))
    seen_causes = {}
    env_cause_at = None
    for a in actions:
        if a.get("outcome_matches_source", True) or a.get("status") == "timed_out":
            continue
        idx = a.get("action_index")
        earlier = [t for t in timed_out if t is not None and idx is not None and t < idx]
        cascade = f", after timed-out action {earlier[-1]}" if earlier else ""
        replay_detail = _one_line(_first_line(a.get("stderr_preview"), a.get("stdout_preview")))
        live_output = live_by_index.get(idx)
        live_detail = _one_line(_first_line(live_output)) if live_output else ""
        if live_detail:
            detail = f": replay said: {replay_detail or '(no output)'} | live said: {live_detail}"
        else:
            detail = f": {replay_detail}" if replay_detail else ""
        lines.append(f"action {idx} {a.get('source_status')} -> {a.get('status')}{cascade}{detail}")
        if earlier:
            continue  # the cascade marker is the explanation
        cause = _mismatch_cause(a, live_output)
        if cause:
            if cause[0] in seen_causes:
                lines.append(f"  cause: same as action {seen_causes[cause[0]]}")
            else:
                seen_causes[cause[0]] = idx
                if "resolved PATH, the interpreter or cwd differently" in cause[0]:
                    env_cause_at = idx
                lines.extend(f"  {line}" for line in cause)
        elif env_cause_at is not None and a.get("source_status") == "failed" and a.get("status") == "ok":
            lines.append(
                f"  cause: likely the same environment difference as action {env_cause_at}; "
                "the live shell's interpreter or tooling ran this command differently"
            )
    return lines


def compared_with_recording(summary, llm=False):
    """Whether the run compared its tool (or LLM) time with the recording's: only a capture+replay of a timed
    recording does. The replay-only recipes, and a capture written without timings, report their own time."""
    comparison = summary.get("llm_comparison" if llm else "comparison") or {}
    return (
        is_capture_workload(summary)
        and ("replay_to_live_llm_ratio" if llm else "replay_to_live_tool_ratio") in comparison
    )


def tool_time_ratio(comparison):
    """(replay/live tool-time ratio, the recording's tool seconds); both None when the run did not compare."""
    comparison = comparison or {}
    return comparison.get("replay_to_live_tool_ratio"), comparison.get("tool_busy_live_sec")


def llm_time_comparison(report, compare=True):
    """Summed per-call wall time over the LLM calls the llm-replay completed, with the call counts: the
    replay's and, with compare, the recording's and their ratio (see tool_time_comparison)."""
    summary = (report or {}).get("summary") or {}
    if not compare:
        return {
            "llm_wait_replay_sec": summary.get("llm_wait_replay_sec"),
            "calls_compared": summary.get("calls_replayed"),
            "calls_failed": summary.get("calls_failed"),
            "calls_skipped": summary.get("calls_skipped"),
        }
    comparison = {
        "llm_wait_live_sec": summary.get("llm_wait_live_sec"),
        "llm_wait_replay_sec": summary.get("llm_wait_replay_sec"),
        "replay_to_live_llm_ratio": summary.get("replay_to_live_llm_ratio"),
        "calls_compared": summary.get("calls_replayed"),
        "calls_failed": summary.get("calls_failed"),
        "calls_skipped": summary.get("calls_skipped"),
    }
    return comparison


def llm_time_ratio(comparison):
    """(replay/live LLM-time ratio, the recording's LLM wait seconds); both None when the run did not compare."""
    comparison = comparison or {}
    return comparison.get("replay_to_live_llm_ratio"), comparison.get("llm_wait_live_sec")


def llm_calls_ok(summary):
    """For runs that replayed the LLM side (the gpu recipes): every recorded call replayed, none failed
    and none skipped for lack of recorded token counts. Runs without an LLM side pass it."""
    comparison = summary.get("llm_comparison")
    if comparison is None:
        return True
    return not comparison.get("calls_failed") and not comparison.get("calls_skipped")


def llm_calls_diagnostic(summary):
    """The FAIL reason when the LLM side did not replay every recorded call, else None."""
    if llm_calls_ok(summary):
        return None
    comparison = summary.get("llm_comparison") or {}
    if comparison.get("calls_failed"):
        return (
            f"llm calls: {comparison['calls_failed']} of the replayed LLM calls failed "
            f"(see llm-replay/{LLM_REPLAY_REPORT} in the run directory)"
        )
    return (
        f"llm calls: {comparison['calls_skipped']} recorded LLM calls carry no token counts and were not "
        "replayed; the recording cannot be replayed in full"
    )


def is_capture_workload(summary):
    """tasktrace_capture_replay and its gpu variant, as opposed to the replay-only recipes."""
    return str(summary.get("workload") or "").startswith("tasktrace_capture_replay")


def summary_ok(summary):
    """The job-level verdict: fidelity PASS, every recorded LLM call replayed (gpu recipes), and every
    phase the recipe ran exited cleanly (for capture+replay: the capture tool, the stage step, the
    replay; for the gpu recipes: the LLM replay too). Tool and LLM time are reported, not judged.
    Shared by summarize-* (exit status) and verdict (printed line) so they cannot disagree."""
    if summary.get("fidelity") != "PASS" or not llm_calls_ok(summary):
        return False
    llm = summary.get("llm_replay")
    if llm is not None and (not llm.get("ran") or llm.get("exit_status") != 0):
        return False
    if is_capture_workload(summary):
        replay, capture = summary.get("replay") or {}, summary.get("capture") or {}
        return bool(
            replay.get("ran")
            and replay.get("tasktrace_exit_status") == 0
            and capture.get("stage_exit_status") == 0
            and capture.get("tasktrace_capture_exit_status") == 0
            and (summary.get("image_identity_check") or {}).get("match")
        )
    return summary.get("tasktrace_exit_status") == 0


def write_json(path, data):
    """Write via a temp file and rename, so a concurrent `verdict` never reads a half-written file."""
    path = pathlib.Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, indent=2) + "\n")
    os.replace(tmp, path)


# --------------------------------------------------------------------------- cpuset


def cmd_cpuset(args):
    """stdin: taskset's allowed list, e.g. "0-3,8-11". Prints the first --count CPUs after skipping the
    first --skip, comma-separated; with --all-remaining, every CPU after the first --skip (at least --min
    of them: the gpu recipes give the replay the first CPUs and the model server the rest)."""
    text = sys.stdin.read().strip()
    cpus = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        a, _, b = part.partition("-")
        cpus.extend(range(int(a), int(b or a) + 1))
    if args.skip < 0:
        sys.exit("[launch] ERROR: cpuset --skip must be 0 or more.")
    if args.all_remaining and args.min < 1:
        sys.exit("[launch] ERROR: cpuset --min must be at least 1 with --all-remaining.")
    rest = cpus[args.skip :]
    if args.all_remaining:
        if len(rest) < args.min:
            sys.exit(
                f"[launch] ERROR: allocation exposes {len(cpus)} CPUs; after the {args.skip} pinned for the replay, "
                f"{len(rest)} are left for the server, need at least {args.min}. Raise the request with "
                f"--slurm-arg cpus-per-task=N on 'llmb-run submit', or lower RUN_CONF_TASKTRACE_CPUS."
            )
        print(",".join(str(c) for c in rest))
        return 0
    if args.count <= 0:
        sys.exit("[launch] ERROR: cpuset needs --count N or --all-remaining.")
    if len(rest) < args.count:
        sys.exit(
            f"[launch] ERROR: allocation exposes {len(cpus)} CPUs, need {args.count + args.skip}. The launch script asks "
            f"for {DEFAULT_CPUS_PER_TASK} per task (#SBATCH --cpus-per-task); raise it with "
            f"--slurm-arg cpus-per-task=N on 'llmb-run submit', or lower RUN_CONF_TASKTRACE_CPUS."
        )
    print(",".join(str(c) for c in rest[: args.count]))
    return 0


# --------------------------------------------------------------------------- elf check

ELF_MACHINES = {"aarch64": 183, "x86_64": 62}  # e_machine values
PT_INTERP = 3


def static_elf_problems(path, machine):
    """Reasons `path` is not a static 64-bit ELF for e_machine `machine` (empty when it is).
    A cross-compiled artifact cannot be executed on the build host, so the headers are inspected."""
    with open(path, "rb") as fh:
        hdr = fh.read(64)
    if len(hdr) < 64 or hdr[:4] != b"\x7fELF" or hdr[4] != 2:
        return ["not a 64-bit ELF file"]
    end = "<" if hdr[5] == 1 else ">"
    problems = []
    e_machine = struct.unpack_from(end + "H", hdr, 18)[0]
    if e_machine != machine:
        problems.append(f"ELF machine {e_machine}, expected {machine}")
    (e_phoff,) = struct.unpack_from(end + "Q", hdr, 32)
    e_phentsize, e_phnum = struct.unpack_from(end + "HH", hdr, 54)
    with open(path, "rb") as fh:
        fh.seek(e_phoff)
        for _ in range(e_phnum):
            entry = fh.read(e_phentsize)
            if len(entry) < 4:
                problems.append("truncated program header table")
                break
            if struct.unpack_from(end + "I", entry, 0)[0] == PT_INTERP:
                problems.append("has a dynamic interpreter (PT_INTERP); expected a static binary")
                break
    return problems


def cmd_elf_check(args):
    tag = args.tag or "[elf-check]"
    machine = ELF_MACHINES[args.arch]
    problems = static_elf_problems(args.path, machine)
    for problem in problems:
        print(f"{tag} {args.path}: {problem}", file=sys.stderr)
    if problems:
        return 1
    print(f"{tag} verified: static ELF for {args.arch} (machine {machine})")
    return 0


# --------------------------------------------------------------------------- image cache key


def image_cache_key(image_ref):
    """tasktrace-capture names cache entries <ref with non [A-Za-z0-9._-] -> _>_<sha256(ref)[:8]>.sqsh."""
    return (
        re.sub(r"[^a-zA-Z0-9._-]", "_", image_ref) + "_" + hashlib.sha256(image_ref.encode()).hexdigest()[:8] + ".sqsh"
    )


def cmd_image_cache_key(args):
    print(image_cache_key(args.image_ref))
    return 0


# --------------------------------------------------------------------------- trace manifests


def image_file_name(image_ref):
    """The .sqsh name llmb-install gives a container.images entry: <parent>+<name>+<tag>.sqsh
    (mirrors llmb_install.downloads.image._generate_image_filename; '#' and '/' both separate)."""
    parts = image_ref.replace("#", "/").split("/")
    if len(parts) < 2:
        return image_ref.replace("/", "_").replace(":", "_").replace("#", "_") + ".sqsh"
    parent, name_and_tag = parts[-2], parts[-1]
    if ":" in name_and_tag:
        name, tag = name_and_tag.split(":", 1)
        version = "sha256-" + tag.split(":", 1)[1][:12] if tag.startswith("sha256:") else tag
    else:
        name, version = name_and_tag, "latest"
    return f"{parent}+{name}+{version}.sqsh"


def cmd_image_file_name(args):
    print(image_file_name(args.image_ref))
    return 0


# --------------------------------------------------------------------------- server info (the *-gpu recipes)


def cmd_server_info(args):
    """server.json for a *-gpu run: the vLLM server the launch script brought up, from the TT_SERVER_*
    variables common/tasktrace_gpu_lib.sh exports. Read back by `stage` (into provenance.json) and by
    summarize-* (the summary's `server` block and the verdict's Server row)."""
    env = os.environ

    def text(key):
        return env.get(key) or None

    def integer(key):
        value = _num(env.get(key))
        return int(value) if value is not None else None

    def lines(key):
        return [line for line in (env.get(key) or "").splitlines() if line.strip()]

    info = {
        "schema_version": 1,
        "kind": "tasktrace-llm-server",
        "config": text("TT_SERVER_CONFIG"),
        "engine": text("TT_SERVER_ENGINE"),
        "version": text("TT_SERVER_VERSION"),
        "served_model": text("TT_SERVER_SERVED_MODEL"),
        "image": {"reference": text("TT_SERVER_IMAGE_REF"), "path": text("TT_SERVER_IMAGE_PATH")},
        "model": {
            "repo": text("TT_SERVER_MODEL_REPO"),
            "revision": text("TT_SERVER_MODEL_REVISION"),
            "path": text("TT_SERVER_MODEL_DIR"),
        },
        "tp": integer("TT_SERVER_TP"),
        "max_model_len": integer("TT_SERVER_MAX_MODEL_LEN"),
        "gpu_memory_utilization": _num(env.get("TT_SERVER_GPU_MEM_UTIL")),
        "extra_args": lines("TT_SERVER_EXTRA_ARGS"),
        "env": lines("TT_SERVER_ENV"),
        "api_base": text("TT_SERVER_API_BASE"),
        "port": integer("TT_SERVER_PORT"),
        "gpu_count": integer("TT_SERVER_GPU_COUNT"),
        "gpu_name": text("TT_SERVER_GPU_NAME"),
        "cpuset": text("TT_SERVER_CPUSET"),
        "startup_sec": integer("TT_SERVER_STARTUP_SEC"),
        "started_at_utc": text("TT_SERVER_STARTED_AT"),
        "log": text("TT_SERVER_LOG"),
        "cache_home": text("TT_SERVER_HOME"),
        "node": text("TT_SERVER_NODE"),
    }
    write_json(args.out, info)
    print(f"[server] info: {args.out}")
    return 0


def load_trace_manifests(root):
    """trace.json files under <root>/<model>/<agent-harness>/<dataset>/<task>/."""
    out = []
    for path in sorted(pathlib.Path(root).glob("*/*/*/*/" + TRACE_MANIFEST)):
        manifest = read_json(path)
        if manifest:
            out.append((path, manifest))
    return out


def cmd_resolve_trace(args):
    """Print `NAME=value` lines (shell-quoted) for the trajectory whose trace.json carries the
    selector: tree coordinates, agent, default live model, task image reference, the .sqsh
    file llmb-install imports for it, and the image architecture."""
    root = pathlib.Path(args.root)
    manifests = load_trace_manifests(root)
    selectors = [m.get("selector") for _, m in manifests]
    for path, m in manifests:
        if m.get("selector") != args.selector:
            continue
        required = ("model", "agent_harness", "dataset", "task")
        missing = (
            [k for k in required if not m.get(k)]
            + ([] if deep_get(m, "image", "reference") else ["image.reference"])
            + ([] if deep_get(m, "image", "arch") else ["image.arch"])
        )
        if missing:
            print(f"[launch] ERROR: {path} is missing {', '.join(missing)}.", file=sys.stderr)
            return 1
        expected_dir = root / m["model"] / m["agent_harness"] / m["dataset"] / m["task"]
        if path.parent.resolve() != expected_dir.resolve():
            print(
                f"[launch] ERROR: {path} declares {m['model']}/{m['agent_harness']}/{m['dataset']}/{m['task']} but lives in {path.parent}.",
                file=sys.stderr,
            )
            return 1
        values = {
            "TRACE_MODEL": m["model"],
            "TRACE_HARNESS": m["agent_harness"],
            "TRACE_DATASET": m["dataset"],
            "TRACE_ID": m["task"],
            "TRACE_DIR": str(expected_dir.parent),
            "TRACE": str(expected_dir / TRAJECTORY_NAME),
            "AGENT": m.get("agent") or m["agent_harness"],
            "LIVE_MODEL_DEFAULT": m.get("live_model_default") or "",
            "IMAGE_REF": m["image"]["reference"],
            "IMAGE_FILE": image_file_name(m["image"]["reference"]),
            "IMAGE_ARCH": m["image"]["arch"],
            # 1 when the trajectory carries its recorded timings, 0 when it was written without them
            "TRACE_TIMED": "0" if trajectory_timed(expected_dir / TRAJECTORY_NAME) is False else "1",
            # Set by `stage` for trajectories a gpu capture produced; empty for the reference trajectories.
            "SERVING_CONFIG": m.get("serving_config") or "",
            "LLM_MODE": m.get("llm_mode") or "",
        }
        for key, value in values.items():
            print(f"{key}={shlex.quote(str(value))}")
        return 0
    print(
        f"[launch] ERROR: no trajectory declares selector '{args.selector}' under {root}; available: {', '.join(s for s in selectors if s) or 'none'}",
        file=sys.stderr,
    )
    return 1


# --------------------------------------------------------------------------- endpoint check


def cmd_endpoint_check(args, urlopen=urllib.request.urlopen):
    """One-token chat completion; catches a bad key (401/403), model id (404) or missing egress.

    429 is a warning: the key is accepted, the endpoint is rate limited. The key is read
    from $TT_API_KEY and never appears on a command line.
    """
    base, model = args.base, args.model
    key = os.environ.get("TT_API_KEY", "")
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": "hi"}], "max_tokens": 1}).encode()
    req = urllib.request.Request(
        base.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=90) as resp:
            payload = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        served = payload.get("model") or model
        print(f"[launch] endpoint check: OK (HTTP {resp.status}, served model {served!r})")
        return 0
    except urllib.error.HTTPError as exc:
        detail = exc.read()[:300].decode("utf-8", errors="replace").replace("\n", " ")
        if key:
            detail = detail.replace(key, "<REDACTED>")
        if exc.code == 429:
            print(
                f"[launch] endpoint check: WARNING HTTP 429 (rate limited); the key is accepted, continuing: {detail}",
                file=sys.stderr,
            )
            return 0
        hint = {
            401: "the key was rejected; for build.nvidia.com generate it while signed in under the NV-Developer org",
            403: "the key is not allowed to call this endpoint",
            404: _hint_404(base, model, key, urlopen),
            502: "the endpoint's model backend is down; retry later",
            503: "the endpoint's model backend is overloaded or down; retry later",
            504: "the endpoint's model backend did not answer in time; retry later",
        }.get(exc.code, "see the response body")
        print(f"[launch] ERROR: endpoint check failed with HTTP {exc.code}: {hint}. Body: {detail}", file=sys.stderr)
        return 1
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        if _is_read_timeout(exc):
            # The TCP connection succeeded and the model produced nothing within the deadline:
            # the endpoint is silent (overloaded or down), not unreachable from this node.
            print(
                f"[launch] ERROR: endpoint check connected to {base} but got no response within 90s: {exc}. "
                "The model backend is not serving (overloaded or down); retry later. Egress is fine.",
                file=sys.stderr,
            )
        else:
            print(
                f"[launch] ERROR: endpoint check could not connect to {base}: {exc}. "
                "The compute node needs egress to the endpoint (DNS, route or firewall).",
                file=sys.stderr,
            )
        return 1


def _hint_404(base, model, key, urlopen):
    """Tell 'model id is wrong' apart from 'model is in the catalog but its backend is gone'.

    integrate.api.nvidia.com keeps a catalog entry per model id that points at a deployed
    function; when that function is deleted or redeployed, chat completions 404 while
    GET /models still lists the id. Checking the config against the catalog then leads
    nowhere, so say which case this is. The catalog lookup is best effort."""
    listed = None
    try:
        req = urllib.request.Request(base.rstrip("/") + "/models", headers={"Authorization": "Bearer " + key})
        with urlopen(req, timeout=30) as resp:
            catalog = json.loads(resp.read().decode("utf-8", errors="replace") or "{}")
        listed = model in {m.get("id") for m in catalog.get("data", []) if isinstance(m, dict)}
    except Exception:
        listed = None
    if listed is True:
        return (
            f"model id {model!r} is listed by GET {base}/models but its backend deployment is gone "
            "(the endpoint's function id no longer exists); the service is being redeployed, retry later"
        )
    if listed is False:
        return f"model id {model!r} is not in GET {base}/models; check RUN_CONF_MODEL"
    return f"model id {model!r} is not served at this base URL; check RUN_CONF_MODEL against GET {base}/models"


def _is_read_timeout(exc):
    """True when the failure happened after connecting: the socket timed out waiting for the response.

    urllib raises the read timeout bare ('The read operation timed out'); a connect timeout comes
    wrapped in URLError(reason=TimeoutError('timed out')), which means no route or a firewall, not a
    silent backend, so it must not be read as one."""
    if isinstance(exc, urllib.error.URLError):
        return "read operation timed out" in str(exc.reason).lower()
    return isinstance(exc, TimeoutError) or "read operation timed out" in str(exc).lower()


# --------------------------------------------------------------------------- stage


def read_trajectory(path):
    """Split a finalized trajectory into its header, action records and llm records.
    Raises ValueError naming the line when the JSONL is malformed."""
    header, actions, llm_calls = {}, [], []
    data = pathlib.Path(path).read_bytes()
    for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"trajectory line {lineno} is not valid JSON: {exc.msg}") from exc
        if not isinstance(rec, dict):
            raise ValueError(f"trajectory line {lineno} is not a JSON object")
        kind = rec.get("record_type")
        if kind == "trace" and not header:
            header = rec
        elif kind == "action":
            actions.append(rec)
        elif kind == "llm":
            llm_calls.append(rec)
    return header, actions, llm_calls, hashlib.sha256(data).hexdigest()


def trajectory_timed(path):
    """True when the trajectory's actions carry their recorded timestamps, False when it was written without
    them (tasktrace-capture --trajectory-timings omit), None when the file has no action to tell from."""
    try:
        with open(path) as handle:
            for line in handle:
                record = json.loads(line) if line.strip() else {}
                if record.get("record_type") == "action":
                    return "started_at_epoch" in record
    except (OSError, ValueError):
        return None
    return None


def cmd_trace_timed(args):
    """TRACE_TIMED=1|0 for one trajectory file, as resolve-trace reports it for a shipped one. The capture
    recipes pick their replay from what the capture wrote, whichever setting asked for it."""
    print(f"TRACE_TIMED={'0' if trajectory_timed(args.trace) is False else '1'}")
    return 0


def cmd_stage(args):
    """Copy the captured trajectory into the trajectories-tree layout with its checksum and a
    provenance record, validate the header, and extract the live run's metrics.
    Inputs beyond the paths arrive as TT_* environment variables (set by launch.sh).
    Exits non-zero when the trace cannot be replayed."""
    env = os.environ
    staged = pathlib.Path(args.staged_dir)
    problems = []
    header, actions, llm_calls = {}, [], []
    trace_sha256 = None
    trial_dir = None
    if args.trace:
        trace = pathlib.Path(args.trace)
        trial_dir = trace.parent.parent
        try:
            header, actions, llm_calls, trace_sha256 = read_trajectory(trace)
        except ValueError as exc:
            # Still stage the file and write provenance so the failure is inspectable.
            problems.append(str(exc))
            trace_sha256 = hashlib.sha256(trace.read_bytes()).hexdigest()
        staged.mkdir(parents=True, exist_ok=True)
        dest = staged / TRAJECTORY_NAME
        shutil.copyfile(trace, dest)
        (staged / (TRAJECTORY_NAME + ".sha256")).write_text(f"{trace_sha256}  {dest.name}\n")
        if problems:
            pass  # the parse error explains the missing header and actions
        elif not header:
            problems.append("trace has no finalized 'trace' header record")
        else:
            schema = header.get("schema_version")
            if schema not in ACCEPTED_SCHEMAS:
                problems.append(f"unsupported trace schema_version {schema!r}")
            if header.get("replay_supported") is not True:
                problems.append("trace header says replay_supported is not true")
            if not actions:
                problems.append("trace has no action records")
    else:
        problems.append("capture produced no finalized trajectory")

    executed = [a for a in actions if a.get("exception_info") != NOT_EXECUTED]
    timed = any("started_at_epoch" in a for a in actions)  # False for a trajectory written without timings
    live_tool_busy = sum(float(a.get("wall_time_sec") or 0.0) for a in executed) if timed else None
    live_llm_wait = sum(float(c.get("wall_time_sec") or 0.0) for c in llm_calls) if timed else None
    live_interval = None
    if executed:
        starts = [float(a["started_at_epoch"]) for a in executed if a.get("started_at_epoch") is not None]
        ends = [float(a["finished_at_epoch"]) for a in executed if a.get("finished_at_epoch") is not None]
        if starts and ends:
            live_interval = max(ends) - min(starts)

    result = read_json(trial_dir / "result.json") if trial_dir else {}
    capture_report = (
        read_json(pathlib.Path(env.get("TT_CAPTURE_DIR", "")) / "report.json") if env.get("TT_CAPTURE_DIR") else {}
    )
    level = deep_get(capture_report, "concurrency", "1") or {}
    server = (read_json(env["TT_SERVER_INFO"]) if env.get("TT_SERVER_INFO") else {}) or None
    if not timed and server:
        # A trajectory written without its timings gets a provenance record without measured time either:
        # no capture start or finish, no capture timing or phases, and no server start-up.
        server = {**server, "startup_sec": None, "started_at_utc": None}
    venv_version = kv_file(env.get("TT_VENV_VERSION_FILE", ""))
    tasktrace_version = kv_file(env.get("TT_TASKTRACE_VERSION_FILE", ""))

    agent_version = None
    for candidate in (
        deep_get(result, "agent_info", "version"),
        deep_get(result, "agent_info", "agent_version"),
        deep_get(result, "agent", "version"),
    ):
        if candidate:
            agent_version = candidate
            break
    if agent_version is None and trial_dir is not None:
        mswea = read_json(trial_dir / "agent" / "mini-swe-agent.trajectory.json")
        agent_version = deep_get(mswea, "info", "mini_version") or deep_get(mswea, "info", "version")

    provenance = {
        "schema_version": 1,
        "kind": "tasktrace-capture-provenance",
        "selector": env.get("TT_SELECTOR"),
        "trace": {
            "model": env.get("TT_TRACE_MODEL"),
            "agent_harness": env.get("TT_TRACE_HARNESS"),
            "dataset": env.get("TT_TRACE_DATASET"),
            "task": env.get("TT_TRACE_ID"),
            "file": TRAJECTORY_NAME,
            "sha256": trace_sha256,
            "timings": "recorded" if timed else "omitted",
            "schema_version": header.get("schema_version"),
            "replay_supported": header.get("replay_supported"),
            "source_trial": header.get("source_trial"),
            "task_checksum": header.get("task_checksum"),
            "harbor_task": header.get("task"),
            "limitations": header.get("limitations"),
            "source_path": args.trace or None,
        },
        "live_run": {
            "model_id": env.get("TT_LIVE_MODEL"),
            "endpoint_model_id": env.get("TT_ENDPOINT_MODEL_ID"),
            "api_base": env.get("TT_API_BASE"),
            "api_key_env_var": env.get("TT_API_KEY_VAR") or None,
            # external-endpoint: the agent talked to a remote OpenAI-compatible endpoint (the CPU recipe).
            # live-server: to a vLLM server the job ran on its own GPUs (the gpu recipe); `server` is that
            # server's server.json.
            "llm_mode": env.get("TT_LLM_MODE") or "external-endpoint",
            "server": server,
            "agent": {"name": env.get("TT_AGENT"), "version": agent_version},
            "agent_timeout_sec": int(env.get("TT_AGENT_TIMEOUT") or 0),
            "concurrency": 1,
            "trials": 1,
            "shell_tracing": True,
            "perfetto_depth": 1,
            # enroot on Slurm; `current` when tasktrace-capture ran inside the task container.
            "runtime": env.get("TT_RUNTIME") or capture_report.get("runtime") or "enroot",
            "cpuset": env.get("TT_CPUSET") or None,
            "allocated_cpus": int(env["TT_CPUS"]) if env.get("TT_CPUS") else None,
            "tasktrace_capture_args": (env.get("TT_CAPTURE_ARGS") or "").split(),
            "tasktrace_capture_exit_status": int(env.get("TT_CAPTURE_EXIT") or 0),
            "started_at_utc": env.get("TT_CAPTURE_STARTED_AT") if timed else None,
            "finished_at_utc": env.get("TT_CAPTURE_FINISHED_AT") if timed else None,
            "reward": header.get("reward", deep_get(result, "verifier_result", "rewards", "reward")),
            "verifier_result": header.get("verifier_result"),
            "llm_calls": len(llm_calls),
            "tool_actions_recorded": len(actions),
            "tool_actions_executed": len(executed),
            "llm_wait_sec": round(live_llm_wait, 3) if timed else None,
            "tool_busy_sec": round(live_tool_busy, 3) if timed else None,
            "tool_interval_sec": round(live_interval, 3) if live_interval is not None else None,
            "capture_report": {
                "timing": level.get("timing") if timed else None,
                "phases": level.get("phases") if timed else None,
                "reward": level.get("reward"),
                "runtime": capture_report.get("runtime"),
                "hostname": capture_report.get("hostname"),
                "system_info": capture_report.get("system_info"),
            },
        },
        "tools": {
            "tasktrace_capture": {
                "commit": venv_version.get("commit"),
                "source": venv_version.get("source"),
                "python": venv_version.get("python"),
                "harbor_version": venv_version.get("harbor"),
                "arch": venv_version.get("arch"),
            },
            "tasktrace": {
                "source": tasktrace_version.get("source"),
                "target": tasktrace_version.get("target"),
            },
        },
        "task_image": {
            "reference": env.get("TT_IMAGE_REF"),
            "path": env.get("TT_IMAGE_PATH") or None,
            "sha256": env.get("TT_IMAGE_SHA256"),
            # The .sqsh file (enroot) or the image digest the container was created from (current runtime).
            "identity": env.get("TT_IMAGE_IDENTITY") or ("sqsh-file" if env.get("TT_IMAGE_PATH") else None),
            "arch": env.get("TT_IMAGE_ARCH"),
        },
        "host": {
            "node": env.get("TT_NODE"),
            "slurm_job_id": env.get("TT_JOB_ID") or None,
            "gpu_type": env.get("TT_GPU_TYPE") or None,
        },
        "problems": problems,
    }
    staged.mkdir(parents=True, exist_ok=True)
    write_json(args.provenance, provenance)
    # The manifest the launch scripts resolve selectors against, so a promoted
    # run needs no launch.sh edit: copy the directory, register it in metadata.
    manifest = {
        "schema_version": 1,
        "kind": "tasktrace-trace",
        "selector": env.get("TT_SELECTOR"),
        "model": env.get("TT_TRACE_MODEL"),
        "agent_harness": env.get("TT_TRACE_HARNESS"),
        "dataset": env.get("TT_TRACE_DATASET"),
        "task": env.get("TT_TRACE_ID"),
        "agent": env.get("TT_AGENT"),
        # The endpoint model id the CPU capture recipe defaults to. A gpu capture keeps the reference
        # trajectory's (TT_LIVE_MODEL_DEFAULT): its own model id names the local server and is recorded
        # in provenance.json instead.
        "live_model_default": env.get("TT_LIVE_MODEL_DEFAULT") or env.get("TT_LIVE_MODEL"),
        "image": {"reference": env.get("TT_IMAGE_REF"), "arch": env.get("TT_IMAGE_ARCH")},
    }
    if env.get("TT_LLM_MODE"):
        manifest["llm_mode"] = env["TT_LLM_MODE"]
    if env.get("TT_SERVER_CONFIG"):
        manifest["serving_config"] = env["TT_SERVER_CONFIG"]  # common/serving/<name>.env that reproduces it
    write_json(staged / TRACE_MANIFEST, manifest)

    print(f"[stage] trajectory: {staged / TRAJECTORY_NAME}" if args.trace else "[stage] no trajectory")
    print(f"[stage] provenance: {args.provenance}")
    print(
        f"[stage] live run: reward={provenance['live_run']['reward']} llm_calls={len(llm_calls)} "
        f"tool_actions={len(executed)}/{len(actions)} llm_wait={_fmt_sec(live_llm_wait)} tool_busy={_fmt_sec(live_tool_busy)}"
    )
    for problem in problems:
        print(f"[stage] PROBLEM: {problem}", file=sys.stderr)
    return 1 if problems else 0


# --------------------------------------------------------------------------- secret scan

NVAPI = re.compile(rb"nvapi-[A-Za-z0-9_-]{20,}")
SCAN_LIMIT = 512 * 1024 * 1024


def cmd_secret_scan(args):
    """No API key may land in any file the job wrote. Redact the configured key's exact value
    (from $TT_SECRET) and anything shaped like a build.nvidia.com key, in place, and record
    the affected paths (never the values)."""
    root = pathlib.Path(args.root)
    secret = os.environ.get("TT_SECRET", "").encode()
    redacted, pattern_hits = [], []
    scanned = skipped_large = 0
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            if path.stat().st_size > SCAN_LIMIT:
                skipped_large += 1
                continue
            data = path.read_bytes()
        except OSError:
            continue
        scanned += 1
        cleaned = data
        if secret and secret in cleaned:
            cleaned = cleaned.replace(secret, b"<REDACTED>")
            redacted.append(str(path.relative_to(root)))
        if NVAPI.search(cleaned):
            cleaned = NVAPI.sub(b"nvapi-<REDACTED>", cleaned)
            pattern_hits.append(str(path.relative_to(root)))
        if cleaned != data:
            path.write_bytes(cleaned)
    hits = sorted(set(redacted) | set(pattern_hits))
    write_json(
        args.report,
        {
            "files_scanned": scanned,
            "files_skipped_over_limit": skipped_large,
            "files_redacted": hits,
            "files_with_configured_key": redacted,
            "files_with_nvapi_pattern": pattern_hits,
        },
    )
    if hits:
        print(f"[scan] WARNING: API key material found and redacted in {len(hits)} file(s):", file=sys.stderr)
        for p in hits:
            print(f"[scan]   {p}", file=sys.stderr)
    else:
        print(f"[scan] no API key material in {scanned} files under {root}")
    return 0


# --------------------------------------------------------------------------- llm replay (the *-gpu recipes)

# Synthetic prompts are token ids, not text: the completions API accepts them directly, so a request
# carries exactly the recorded number of prompt tokens. Ids in this range are ordinary vocabulary entries
# for every tokenizer the recipes serve (special tokens sit below it, vocabularies extend far beyond it).
LLM_TOKEN_ID_LOW, LLM_TOKEN_ID_HIGH = 1000, 30000
LLM_REPLAY_REPORT = "llm_replay_report.json"


# Untimed requests issued before the timed phases: a fresh server spends tens of seconds compiling
# kernels on its first prefills, which is a cold-start cost, not a property of the workload. Prompt
# lengths in tokens, each with a short completion; the trajectory's largest prompt is added when it
# is longer. Smaller one-off compiles (up to about 1.5 s) can still happen the first time some other
# prompt size appears; no fixed list of sizes avoids them, and they hit the live run the same way.
WARMUP_PROMPT_TOKENS = (2048, 8192, 32768)
WARMUP_COMPLETION_TOKENS = 16
WARMUP_TIMEOUT_SEC = 600  # a warm-up that takes longer than this is a broken server, not a slow compile
ALIGN_POLL_SEC = 0.01


def synthetic_prompt(previous, prompt_tokens, rng):
    """Token ids for one call: the previous call's prompt as the prefix, fresh ids for the rest. An
    agent conversation only ever grows (the last prompt plus the model's answer plus the tool output),
    so this is the recorded structure, and the server under test gets the prefix-cache hits that
    structure gives it. A prompt that shrank (the agent truncated) keeps only what still fits."""
    reuse = max(0, min(len(previous), prompt_tokens))
    ids = list(previous[:reuse])
    ids.extend(rng.randrange(LLM_TOKEN_ID_LOW, LLM_TOKEN_ID_HIGH) for _ in range(prompt_tokens - reuse))
    return ids


class StreamEndedEarlyError(OSError):
    """The completion stream closed without its final usage record: the server gave up mid-request."""


def _post_completion(api_base, model, ids, max_tokens, timeout, urlopen, clock):
    """One streamed completion of exactly `max_tokens` tokens (ignore_eos). Returns the client-side wall
    time, the time to the first generated token, and the server's usage record from the final chunk.
    A stream that ends without that record (an engine that died mid-request) is a failure."""
    body = {
        "model": model,
        "prompt": ids,
        "max_tokens": max_tokens,
        "ignore_eos": True,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        api_base.rstrip("/") + "/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = clock()
    ttft = usage = None
    with urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[len("data:") :].strip()
            if payload == "[DONE]":
                break
            try:
                chunk = json.loads(payload)
            except ValueError:
                continue
            if not isinstance(chunk, dict):
                continue
            if ttft is None and any(c.get("text") for c in chunk.get("choices") or [] if isinstance(c, dict)):
                ttft = clock() - t0
            if isinstance(chunk.get("usage"), dict):
                usage = chunk["usage"]
    if usage is None:
        raise StreamEndedEarlyError(f"stream ended after {clock() - t0:.1f}s without a usage record (server gave up)")
    return {"wall_time_sec": clock() - t0, "ttft_sec": ttft, "usage": usage}


def _percentile(values, fraction):
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(fraction * (len(ordered) - 1))))]


def llm_replay_summary(rows, pacing, wall):
    ok = [r for r in rows if r["status"] == "ok"]
    failed = [r for r in rows if r["status"] == "failed"]
    # a trajectory written without its timings recorded no call times: the live wait is unknown, not zero
    untimed = any(r.get("source_wall_time_sec") is None for r in ok)
    live = None if untimed else sum(r["source_wall_time_sec"] for r in ok)
    replay = sum(r.get("wall_time_sec") or 0.0 for r in ok)
    generated = sum(r.get("completion_tokens") or 0 for r in ok)
    ttfts = [r["ttft_sec"] for r in ok if isinstance(r.get("ttft_sec"), (int, float))]
    lags = [r["schedule_lag_sec"] for r in rows if isinstance(r.get("schedule_lag_sec"), (int, float))]
    return {
        "calls_recorded": len(rows),
        "calls_replayed": len(ok),
        "calls_failed": len(failed),
        "calls_skipped": len(rows) - len(ok) - len(failed),
        "llm_wait_live_sec": round(live, 3) if live is not None else None,
        "llm_wait_replay_sec": round(replay, 3),
        "replay_to_live_llm_ratio": round(replay / live, 4) if live and ok else None,
        "prompt_tokens_recorded": sum(r.get("source_prompt_tokens") or 0 for r in ok),
        "prompt_tokens_replayed": sum(r.get("prompt_tokens") or 0 for r in ok),
        "cached_tokens_recorded": sum(r.get("source_cached_tokens") or 0 for r in ok),
        "cached_tokens_replayed": sum(r.get("cached_tokens") or 0 for r in ok),
        "completion_tokens_recorded": sum(r.get("source_completion_tokens") or 0 for r in ok),
        "completion_tokens_replayed": generated,
        "generation_tokens_per_sec": round(generated / replay, 2) if replay > 0 else None,
        "ttft_p50_sec": round(_percentile(ttfts, 0.5), 4) if ttfts else None,
        "ttft_max_sec": round(max(ttfts), 4) if ttfts else None,
        "schedule_lag_max_sec": round(max(lags), 3) if lags else None,
        "pacing": pacing,
        "wall_time_sec": round(wall, 3),
    }


def _call_row(call):
    """The recorded side of one LLM call's report row."""
    prompt_tokens = int(_num(call.get("prompt_tokens")) or 0)
    completion_tokens = int(_num(call.get("completion_tokens")) or 0)
    return {
        "llm_index": call.get("llm_index"),
        "source_prompt_tokens": prompt_tokens or None,
        "source_cached_tokens": int(_num(call.get("cached_prompt_tokens")) or 0),
        "source_completion_tokens": completion_tokens or None,
        "source_wall_time_sec": _num(call.get("wall_time_sec")),
        "status": "skipped",
    }


def _health_url(api_base):
    base = api_base.rstrip("/")
    return (base[: -len("/v1")] if base.endswith("/v1") else base) + "/health"


def _server_alive(api_base, urlopen, timeout=20):
    try:
        with urlopen(_health_url(api_base), timeout=timeout):
            return True
    except Exception:  # any failure to answer /health is "not alive" for the caller
        return False


def _first_start(records):
    for record in records:
        value = _num(record.get("started_at_epoch"))
        if value is not None:
            return value
    return None


def _print_start_offset(calls, actions):
    """--lead-only: seconds from the recording's first tool action to its first LLM call (negative when
    the LLM call came first), the --start-delay that lines the LLM side up with the tool side."""
    origin = _first_start(calls)
    first_action = _first_start([a for a in actions if a.get("exception_info") != NOT_EXECUTED])
    offset = (origin - first_action) if origin is not None and first_action is not None else 0.0
    print(f"LLM_START_OFFSET={offset:.3f}")
    return 0


def _process_alive(pid):
    """False once `pid` has exited (gone, or a zombie its parent has not reaped yet)."""
    try:
        return pathlib.Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0] != "Z"
    except (OSError, IndexError):
        return False


def _wait_for_start(path, pid, timeout, sleep, clock):
    """--align-file: the clock reading at which the tool replay wrote `path` (its task container does so as
    its first action starts), or None when the tool replay (`pid`) ended first or `timeout` passed."""
    marker, deadline = pathlib.Path(path), clock() + timeout
    while clock() < deadline:
        if marker.is_file() and marker.stat().st_size > 0:
            return clock()
        if pid and not _process_alive(pid):
            return None
        sleep(ALIGN_POLL_SEC)
    return None


class _LoopHandoff:
    """--loop-sync-dir: llm-replay's half of a loop-order replay, for a trajectory without recorded timings.
    The tool replay (`tasktrace profile trace --loop-order`) creates the directory, writes the index of each
    action it finished to `tool_done` and `tool_end` when done; this side waits, before each LLM call, until
    every action recorded before that call has run, and writes the call's index to `llm_done` after it and
    `llm_end` when done. Each side stops waiting once the other has ended, so neither can hang the other."""

    def __init__(self, directory, actions, pid, sleep, clock):
        self.dir = pathlib.Path(directory)
        self.actions = [(a.get("after_llm_index"), int(a["action_index"])) for a in actions]
        self.pid, self.sleep, self.clock = pid, sleep, clock

    def _tool_side_gone(self):
        return (self.dir / "tool_end").exists() or bool(self.pid and not _process_alive(self.pid))

    def started(self, timeout):
        """Wait for the tool replay to create the directory; False when it ended first or `timeout` passed."""
        deadline = self.clock() + timeout
        while self.clock() < deadline:
            if self.dir.is_dir():
                return True
            if self.pid and not _process_alive(self.pid):
                return False
            self.sleep(ALIGN_POLL_SEC)
        return False

    def before(self, llm_index):
        """Wait until the tool side has run every action recorded before LLM call `llm_index`; returns the
        seconds waited."""
        needed = max((index for after, index in self.actions if after is None or after < llm_index), default=0)
        waited_from = self.clock()
        while not self._tool_side_gone():
            try:
                if int((self.dir / "tool_done").read_text().strip()) >= needed:
                    break
            except (OSError, ValueError):
                if needed <= 0:
                    break
            self.sleep(ALIGN_POLL_SEC)
        return self.clock() - waited_from

    def done(self, llm_index):
        partial = self.dir / ".llm_done.partial"
        partial.write_text(f"{llm_index}\n")
        partial.replace(self.dir / "llm_done")

    def end(self):
        if self.dir.is_dir():
            (self.dir / "llm_end").write_bytes(b"")


def _warm_up(args, calls, rng, urlopen, clock):
    """--warmup-only: the untimed requests, the fixed shapes plus the trajectory's largest prompt."""
    largest = max((int(_num(c.get("prompt_tokens")) or 0) for c in calls), default=0)
    shapes = list(WARMUP_PROMPT_TOKENS) + ([largest] if largest > WARMUP_PROMPT_TOKENS[-1] else [])
    for prompt_tokens in shapes:
        ids = synthetic_prompt([], prompt_tokens, rng)
        try:
            result = _post_completion(
                args.api_base,
                args.model,
                ids,
                WARMUP_COMPLETION_TOKENS,
                min(args.timeout_sec, WARMUP_TIMEOUT_SEC),
                urlopen,
                clock,
            )
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            detail = (
                exc.read()[:300].decode("utf-8", errors="replace") if isinstance(exc, urllib.error.HTTPError) else exc
            )
            print(
                f"[llm-replay] ERROR: warm-up request of {prompt_tokens} prompt tokens failed: {detail}",
                file=sys.stderr,
            )
            return 1
        print(
            f"[llm-replay] warm-up: prompt {prompt_tokens} completion {WARMUP_COMPLETION_TOKENS} took "
            f"{_fmt_sec(result['wall_time_sec'])} (ttft {_fmt_sec(result['ttft_sec'])})",
            flush=True,
        )
    return 0


def cmd_llm_replay(args, urlopen=urllib.request.urlopen, sleep=time.sleep, clock=time.monotonic):
    """Replay the LLM side of a trajectory against a live OpenAI-compatible server (the gpu recipes).

    Every recorded call is re-issued in order with a synthetic prompt of the recorded prompt length
    (the previous call's prompt as its prefix, as in the recorded conversation) and an output forced
    to the recorded completion length, so the server sees the recorded token shapes; the text is
    not reproduced. `recorded` pacing issues each call at its recorded offset from the first one, never
    before the previous call returned; `unpaced` issues them back to back. Writes a report
    with one row per call and the summed LLM wait, live vs replay. Exits 1 when any call failed.

    --lead-only prints LLM_START_OFFSET, the seconds from the recording's first tool action to its first
    LLM call. The launch script starts the tool replay first and passes that offset as --start-delay with
    --align-file, the file the tool replay's task container writes as its first action starts: the LLM
    timeline is then counted from that moment, so container setup does not shift one side against the
    other. --align-pid is the tool replay's process; when it ends before the file appears, no call is
    attempted and all count as failed. --warmup-only
    sends the untimed warm-up requests (WARMUP_PROMPT_TOKENS plus the trajectory's largest prompt) and
    exits. After a failed call the server's /health is probed; an unresponsive server aborts the
    remaining calls (they count as failed) instead of waiting out one timeout per call."""
    try:
        _, actions, llm_calls, _ = read_trajectory(args.trace)
    except (OSError, ValueError) as exc:
        print(f"[llm-replay] ERROR: {exc}", file=sys.stderr)
        return 2
    calls = sorted(llm_calls, key=lambda c: (_num(c.get("started_at_epoch")) or 0.0, c.get("llm_index") or 0))
    origin = _first_start(calls)
    if args.lead_only:
        return _print_start_offset(calls, actions)
    if not (args.api_base and args.model and (args.output or args.warmup_only)):
        print("[llm-replay] ERROR: --api-base, --model and --output are required.", file=sys.stderr)
        return 2
    if args.warmup_only:
        # its own token stream: a warm-up prompt that began like the first timed prompt would prime the
        # server's prefix cache for it, a hit the recorded call did not get
        return _warm_up(args, calls, random.Random(args.seed + 1), urlopen, clock)
    loop = None
    if args.loop_sync_dir:
        if args.pacing != "unpaced":
            print(
                "[llm-replay] ERROR: --loop-sync-dir keeps the recorded order, not the schedule; use --pacing unpaced.",
                file=sys.stderr,
            )
            return 2
        missing = [a.get("action_index") for a in actions if "after_llm_index" not in a]
        if missing:
            print(
                f"[llm-replay] ERROR: trajectory action {missing[0]} has no after_llm_index; loop order needs a "
                "trajectory written with tasktrace-capture --trajectory-timings omit.",
                file=sys.stderr,
            )
            return 2
        loop = _LoopHandoff(args.loop_sync_dir, actions, args.align_pid, sleep, clock)
    rng = random.Random(args.seed)
    rows, replay = [], calls
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    t_start = clock()
    if args.align_file:
        t_start = _wait_for_start(args.align_file, args.align_pid, args.align_timeout_sec, sleep, clock)
        if t_start is None:
            reason = f"not attempted: the tool replay ended or timed out before its first action ({args.align_file})"
            print(f"[llm-replay] ERROR: {reason}", file=sys.stderr, flush=True)
            rows, replay, t_start = [dict(_call_row(c), status="failed", error=reason) for c in calls], [], clock()
    if loop is not None and replay and not loop.started(args.align_timeout_sec):
        reason = f"not attempted: the tool replay ended or timed out before it started ({args.loop_sync_dir})"
        print(f"[llm-replay] ERROR: {reason}", file=sys.stderr, flush=True)
        rows, replay = [dict(_call_row(c), status="failed", error=reason) for c in calls], []
    try:
        return _replay_llm_calls(
            args, calls, replay, rows, origin, t_start, started_at, rng, loop, urlopen, sleep, clock
        )
    finally:
        if loop is not None:
            loop.end()


def _replay_llm_calls(args, calls, replay, rows, origin, t_start, started_at, rng, loop, urlopen, sleep, clock):
    """cmd_llm_replay's timed part: issue the calls (paced, or in loop order), then write the report."""
    previous = []
    if args.pacing == "recorded" and replay:
        t_start += args.start_delay  # negative: the recording's first LLM call came before its first tool action
        if t_start > clock():
            sleep(t_start - clock())
    for position, call in enumerate(replay):
        llm_index = call.get("llm_index")
        waited = loop.before(int(llm_index)) if loop is not None and llm_index is not None else None
        prompt_tokens = int(_num(call.get("prompt_tokens")) or 0)
        completion_tokens = int(_num(call.get("completion_tokens")) or 0)
        row = _call_row(call)
        if waited is not None:
            row["loop_wait_sec"] = round(waited, 3)
        if prompt_tokens <= 0 or completion_tokens <= 0:
            row["reason"] = "no recorded token counts"
            rows.append(row)
            if loop is not None:
                loop.done(llm_index)
            continue
        start = _num(call.get("started_at_epoch"))
        if args.pacing == "recorded" and origin is not None and start is not None:
            wait = t_start + (start - origin) - clock()
            row["schedule_lag_sec"] = round(max(0.0, -wait), 3)
            if wait > 0:
                sleep(wait)
        ids = synthetic_prompt(previous, prompt_tokens, rng)
        previous = ids
        row["offset_sec"] = round(clock() - t_start, 3)
        try:
            result = _post_completion(
                args.api_base, args.model, ids, completion_tokens, args.timeout_sec, urlopen, clock
            )
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", errors="replace").replace("\n", " ")
            row.update(status="failed", error=f"HTTP {exc.code}: {detail}")
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            row.update(status="failed", error=str(exc)[:300])
        else:
            usage = result["usage"] or {}
            details = usage.get("prompt_tokens_details") if isinstance(usage.get("prompt_tokens_details"), dict) else {}
            row.update(
                status="ok",
                wall_time_sec=round(result["wall_time_sec"], 4),
                ttft_sec=round(result["ttft_sec"], 4) if result["ttft_sec"] is not None else None,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                cached_tokens=details.get("cached_tokens"),
            )
            if usage.get("completion_tokens") != completion_tokens:
                row.update(
                    status="failed",
                    error=f"the server generated {usage.get('completion_tokens')} of {completion_tokens} completion tokens",
                )
        rows.append(row)
        if loop is not None:
            loop.done(llm_index)
        print(
            f"[llm-replay] call {row['llm_index']}: {row['status']}  prompt {prompt_tokens}  completion "
            f"{completion_tokens}  live {_fmt_sec(row['source_wall_time_sec'])}  replay {_fmt_sec(row.get('wall_time_sec'))}"
            + (f"  {row['error']}" if row.get("error") else ""),
            flush=True,
        )
        if row["status"] == "failed" and not _server_alive(args.api_base, urlopen):
            reason = f"not attempted: the server stopped answering /health after call {row['llm_index']} failed"
            print(f"[llm-replay] ERROR: {reason}; aborting the remaining calls", file=sys.stderr, flush=True)
            for later in calls[position + 1 :]:
                rows.append(dict(_call_row(later), status="failed", error=reason))
            break
    summary = llm_replay_summary(rows, args.pacing, clock() - t_start)
    pathlib.Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    report = {
        "schema_version": 1,
        "kind": "tasktrace-llm-replay",
        "trace": args.trace,
        "api_base": args.api_base,
        "model": args.model,
        "pacing": args.pacing,
        "seed": args.seed,
        # CLOCK_MONOTONIC, the clock of the tool replay's replay_start_monotonic_ns: a call went out at
        # timeline_start_monotonic_sec + offset_sec, so the two sides' alignment can be checked
        "timeline_start_monotonic_sec": round(t_start, 6),
        "aligned_to": args.align_file or None,
        "loop_sync_dir": args.loop_sync_dir or None,
        "started_at_utc": started_at,
        "finished_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "summary": summary,
        "calls": rows,
    }
    write_json(args.output, report)
    print(
        f"[llm-replay] {summary['calls_replayed']}/{summary['calls_recorded']} calls replayed, "
        f"{summary['calls_failed']} failed; llm wait replay {_fmt_sec(summary['llm_wait_replay_sec'])} vs live "
        f"{_fmt_sec(summary['llm_wait_live_sec'])} (ratio {_fmt_ratio(summary['replay_to_live_llm_ratio'])}); "
        f"report: {args.output}",
        flush=True,
    )
    return 1 if summary["calls_failed"] else 0


# --------------------------------------------------------------------------- summaries


CPU_FIELDS = (
    "method",
    "wall_sec",
    "usage_sec",
    "user_sec",
    "system_sec",
    "average_logical_cores_used",
    "allocated_logical_cpus",
    "allocated_cpu_source",
    "utilization_percent_of_allocation",
    "enforced_logical_cpu_limit",
    "enforced_cpu_limit_source",
    "cpu_max",
    "cgroup_cpuset",
    "process_affinity_logical_cpu_ids",
    "effective_logical_cpu_count",
    "throttling",
)


def replay_cpu(pipeline):
    """The replay's CPU measurement from the pipeline report, with the pinning regime (cpuset, cpu.max,
    affinity) so shared-core and exclusive-core runs can be told apart. None when the replay did not run."""
    cpu = pipeline.get("container_cpu")
    if not isinstance(cpu, dict):
        return None
    block = {field: cpu.get(field) for field in CPU_FIELDS if field in cpu}
    block["runtime"] = pipeline.get("runtime")
    validity = pipeline.get("measurement_validity")
    if isinstance(validity, dict):
        block["valid_for_average_cpu_reporting"] = validity.get("valid_for_average_cpu_reporting")
        block["exclusive_single_task_container_asserted"] = validity.get("exclusive_single_task_container_asserted")
    return block


def image_identity_check(capture_sha, pipeline, replay_image_sha256=None):
    """Did capture and replay use the same task image? Under docker/enroot the pipeline report identifies the
    replay image; under the current runtime the launcher supplies its digest (`--replay-image-sha256` or
    $TT_IMAGE_SHA256)."""
    replay_sha = (pipeline.get("image_identity") or {}).get("sha256")
    source = "pipeline_report"
    if replay_sha is None and pipeline.get("runtime") == "current":
        replay_sha = replay_image_sha256 or os.environ.get("TT_IMAGE_SHA256") or None
        source = (
            "current-container: image digest supplied by the launcher"
            if replay_sha
            else "current-container: no image digest supplied"
        )
    return {
        "capture_image_sha256": capture_sha,
        "replay_image_sha256": replay_sha,
        "match": bool(capture_sha) and capture_sha == replay_sha,
        "source": source,
    }


def _llm_side(args, compare=True):
    """The LLM side of a gpu run for the summary: (llm_replay block, llm_comparison, server), from the
    llm-replay report and server.json. (None, None, None) for the CPU recipes, which pass no --llm-report.
    """
    if not args.llm_report:
        return None, None, None
    report = read_json(args.llm_report)
    llm = {
        "ran": bool(report),
        "exit_status": int(args.llm_exit) if args.llm_exit != "" else None,
        "pacing": report.get("pacing"),
        "api_base": report.get("api_base"),
        "model": report.get("model"),
        "report": args.llm_report,
        "summary": report.get("summary", {}),
    }
    server = read_json(args.server_info) if args.server_info else {}
    return llm, llm_time_comparison(report, compare), server or None


def _llm_diagnostics(summary):
    """FAIL reasons for the LLM side of a gpu run; empty for runs without one."""
    llm = summary.get("llm_replay")
    if llm is None:
        return []
    out = []
    if not llm.get("ran"):
        out.append("llm replay did not run (no report)")
    elif llm.get("exit_status") is None:
        out.append("llm replay: exit status unknown")
    elif llm["exit_status"] not in (0, 1):  # 1 means calls failed, which the check below spells out
        out.append(f"llm replay: exited {llm['exit_status']}")
    calls = llm_calls_diagnostic(summary)
    if calls:
        out.append(calls)
    return out


def cmd_summarize_capture(args):
    """Live-run facts from the provenance record, fidelity + wall from the replay report,
    image identity from the pipeline report (or, under the current-container runtime, from the
    launcher-supplied image digest)."""
    provenance = read_json(args.provenance)
    # a trajectory captured without timings (--trajectory-timings omit) has nothing to compare with
    recording_timed = (provenance.get("trace") or {}).get("timings") != "omitted"
    scrub = read_json(args.scrub_report)
    profile = pathlib.Path(args.profile_dir)
    report = find_replay_report(profile)
    pipeline = read_json(profile / "pipeline_report.json")
    replay_ran = args.replay_exit != ""
    fidelity = fidelity_verdict(report, replay_ran)
    capture_sha = (provenance.get("task_image") or {}).get("sha256")
    identity = image_identity_check(capture_sha, pipeline, getattr(args, "replay_image_sha256", None))
    replay_sha = identity["replay_image_sha256"]
    live = provenance.get("live_run") or {}
    trace = provenance.get("trace") or {}
    summary = {
        "schema_version": 1,
        "workload": args.workload,
        "selector": provenance.get("selector"),
        "trace_id": trace.get("task"),
        "trajectory": {
            "path": str(pathlib.Path(args.provenance).parent / TRAJECTORY_NAME),
            "sha256": trace.get("sha256"),
            "schema_version": trace.get("schema_version"),
            "source_trial": trace.get("source_trial"),
            "provenance": args.provenance,
        },
        "capture": {
            "tasktrace_capture_exit_status": live.get("tasktrace_capture_exit_status"),
            "stage_exit_status": int(args.stage_exit),
            "model_id": live.get("model_id"),
            "endpoint_model_id": live.get("endpoint_model_id"),
            "api_base": live.get("api_base"),
            "llm_mode": live.get("llm_mode") or "external-endpoint",
            "agent": live.get("agent"),
            "agent_timeout_sec": live.get("agent_timeout_sec"),
            "cpuset": live.get("cpuset"),
            "allocated_cpus": live.get("allocated_cpus"),
            "reward": live.get("reward"),
            "llm_calls": live.get("llm_calls"),
            "tool_actions_recorded": live.get("tool_actions_recorded"),
            "tool_actions_executed": live.get("tool_actions_executed"),
            "llm_wait_sec": live.get("llm_wait_sec"),
            "tool_busy_sec": live.get("tool_busy_sec"),
            "tool_interval_sec": live.get("tool_interval_sec"),
            "timing": (live.get("capture_report") or {}).get("timing"),
            "phases": (live.get("capture_report") or {}).get("phases"),
            "problems": provenance.get("problems"),
        },
        "replay": {
            "ran": replay_ran,
            "tasktrace_exit_status": int(args.replay_exit) if replay_ran else None,
            "pacing": args.pacing,
            "allocated_cpus": int(args.cpus),
            "cpuset": args.cpuset or None,
            "summary": report.get("summary", {}),
            "pipeline_status": pipeline.get("status"),
            "pipeline_invalid_reasons": pipeline.get("profiling_invalid_reasons"),
            "image_identity": pipeline.get("image_identity"),
        },
        "comparison": tool_time_comparison(report, compare=recording_timed),
        "cpu": replay_cpu(pipeline),
        "task_image": provenance.get("task_image"),
        "image_identity_check": identity,
        "fidelity": fidelity,
        "secret_scan": scrub,
    }
    llm, llm_comparison, server = _llm_side(args, compare=recording_timed)
    if llm is not None:
        summary["llm_replay"] = llm
        summary["llm_comparison"] = llm_comparison
    if server or live.get("server"):
        summary["server"] = server or live.get("server")
    diagnostics = []
    if int(args.stage_exit) != 0:
        diagnostics.append(f"capture: stage exit {int(args.stage_exit)}; the live run left no replayable trajectory")
    if live.get("tasktrace_capture_exit_status") not in (None, 0):
        diagnostics.append(f"capture: tasktrace-capture exited {live['tasktrace_capture_exit_status']}")
    diagnostics.extend(f"capture: {problem}" for problem in provenance.get("problems") or [])
    if not identity["match"]:
        diagnostics.append(f"image identity mismatch: capture {capture_sha} vs replay {replay_sha}")
    if not replay_ran:
        diagnostics.append("replay did not run")
    else:
        if int(args.replay_exit) != 0:
            diagnostics.append(f"replay: tasktrace exited {int(args.replay_exit)}")
        if fidelity != "PASS":
            diagnostics.extend(replay_diagnostics(report, live_tool_outputs(args.capture_dir)))
    diagnostics.extend(_llm_diagnostics(summary))
    summary["diagnostics"] = diagnostics
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary_ok(summary) else 1


def cmd_summarize_replay(args):
    """The replay-only recipes: fidelity + wall from the replay report, the replay's tool (and LLM) time,
    reported without comparing it with the recording, image identity from the pipeline report."""
    out = pathlib.Path(args.out_dir)
    report = find_replay_report(out)
    pipeline = read_json(out / "pipeline_report.json")
    fidelity = fidelity_verdict(report)
    summary = {
        "schema_version": 1,
        "workload": args.workload,
        "trace_id": args.trace_id,
        "pacing": args.pacing,
        "allocated_cpus": int(args.cpus),
        "cpuset": args.cpuset or None,
        "tasktrace_exit_status": int(args.replay_exit),
        "fidelity": fidelity,
        "replay": report.get("summary", {}),
        "comparison": tool_time_comparison(report, compare=False),
        "cpu": replay_cpu(pipeline),
        "pipeline_status": pipeline.get("status"),
        "pipeline_invalid_reasons": pipeline.get("profiling_invalid_reasons"),
        "image_identity": pipeline.get("image_identity"),
    }
    llm, llm_comparison, server = _llm_side(args, compare=False)
    if llm is not None:
        summary["llm_replay"] = llm
        summary["llm_comparison"] = llm_comparison
    if server:
        summary["server"] = server
    diagnostics = []
    if int(args.replay_exit) != 0:
        diagnostics.append(f"tasktrace exited {int(args.replay_exit)}")
    if fidelity != "PASS":
        diagnostics.extend(replay_diagnostics(report))
    diagnostics.extend(_llm_diagnostics(summary))
    summary["diagnostics"] = diagnostics
    write_json(args.summary, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary_ok(summary) else 1


# --------------------------------------------------------------------------- verdict


def _fmt_ratio(value):
    return f"{value:.3f}" if isinstance(value, (int, float)) else "n/a"


def _fmt_count(value):
    return "?" if value is None else str(value)


def _replay_phrase(replay_summary, comparison, compared=True):
    executed, recorded = replay_summary.get("actions_executed"), replay_summary.get("actions_recorded")
    parts = [
        f"{_fmt_count(executed)}/{_fmt_count(recorded)} actions",
        f"{_fmt_count(replay_summary.get('source_outcome_mismatch_count'))} mismatches",
    ]
    if replay_summary.get("actions_timed_out"):
        parts.append(f"{replay_summary['actions_timed_out']} timed out")
    if not compared:
        return f"tool time {_fmt_sec(comparison.get('tool_busy_replay_sec'))}, " + ", ".join(parts)
    ratio, _ = tool_time_ratio(comparison)
    tag = " (live)" if ratio is not None else ""
    return f"tool time {_fmt_ratio(ratio)}{tag}, " + ", ".join(parts)


def _llm_phrase(summary):
    comparison = summary.get("llm_comparison")
    if comparison is None:
        return ""
    if compared_with_recording(summary, llm=True):
        ratio, _ = llm_time_ratio(comparison)
        timing = _fmt_ratio(ratio) + (" (live)" if ratio is not None else "")
    else:
        timing = _fmt_sec(comparison.get("llm_wait_replay_sec"))
    return f"  llm time {timing}" + (
        f", {comparison['calls_failed']} calls failed" if comparison.get("calls_failed") else ""
    )


def verdict_line(summary):
    """One line: the outcome (PASS/FAIL), the trace, and the numbers a reader needs to judge the run."""
    verdict = f"Outcome: {'PASS' if summary_ok(summary) else 'FAIL'}"
    trace = summary.get("trace_id") or "?"
    comparison = summary.get("comparison") or {}
    if is_capture_workload(summary):
        capture, replay = summary.get("capture") or {}, summary.get("replay") or {}
        if capture.get("stage_exit_status") not in (None, 0):
            capture_txt = f"stage exit {capture['stage_exit_status']}"
        else:
            capture_txt = (
                f"reward {capture.get('reward')}, {_fmt_count(capture.get('llm_calls'))} llm calls, "
                f"{_fmt_count(capture.get('tool_actions_executed'))} tool actions"
            )
        replay_txt = (
            _replay_phrase(replay.get("summary") or {}, comparison, compared=compared_with_recording(summary))
            if replay.get("ran")
            else "did not run"
        )
        return f"{verdict}  {trace}  capture: {capture_txt}  replay: {replay_txt}{_llm_phrase(summary)}"
    return (
        f"{verdict}  {trace}  replay: {_replay_phrase(summary.get('replay') or {}, comparison, compared=False)}"
        f"{_llm_phrase(summary)}"
    )


LAUNCH_ERRORS_SHOWN = 3


FATAL_LINE_RE = re.compile(
    r"\[launch\] ERROR|\bERROR:|allocation exposes \d+ CPUs|Traceback \(most recent call last\)|sbatch: error|"
    r"srun: error|slurmstepd: error|Exec format error|DUE TO TIME LIMIT|CANCELLED AT"
)
KILLED_RE = re.compile(r"DUE TO TIME LIMIT|CANCELLED AT")


def launch_error_header(errors):
    """How to introduce the quoted fatal lines: a Slurm kill mid-run is not an early abort."""
    if any(KILLED_RE.search(e) for e in errors):
        return "Slurm ended the job before it wrote summary.json:"
    return "the launch script aborted before the run:"


def _strip_cause(text):
    text = str(text).strip()
    return text[len("cause: ") :] if text.startswith("cause: ") else text


def launch_errors(experiment_dir):
    """The last few distinct fatal lines from the job's Slurm output: why a launch script aborted early.

    Besides '[launch] ERROR' lines this catches the helper's own exits (the cpuset check), Python
    tracebacks, Slurm's own errors, wrong-architecture binaries, and time-limit kills."""
    found = []
    for log in sorted(pathlib.Path(experiment_dir).glob("slurm-*.out")):
        try:
            text = log.read_text(errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if FATAL_LINE_RE.search(line) and line not in found:
                found.append(line)
    return found[-LAUNCH_ERRORS_SHOWN:]


def slurm_states(job_ids):
    """job id -> Slurm state (e.g. RUNNING, COMPLETED, FAILED) via sacct; {} when sacct is unavailable.

    Lets the report tell 'still running' from 'died without writing summary.json', which the
    experiment directory alone cannot."""
    ids = [j for j in job_ids if j and j.isdigit()]
    if not ids or not shutil.which("sacct"):
        return {}
    try:
        out = subprocess.run(
            ["sacct", "-j", ",".join(ids), "-n", "-P", "-X", "-o", "JobID,State"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    states = {}
    for line in out.splitlines():
        job, _, state = line.strip().partition("|")
        if job and state:
            states[job] = state.split()[0]  # "CANCELLED by 1234" -> CANCELLED
    return states


OPEN_STATES = {
    "PENDING",
    "RUNNING",
    "COMPLETING",
    "CONFIGURING",
    "SUSPENDED",
    "REQUEUED",
    "REQUEUE_HOLD",
    "REQUEUE_FED",
    "RESV_DEL_HOLD",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
    "STOPPED",
}
NO_SACCT_NOTE = " (sacct unavailable: state inferred from the directory, not from Slurm)"


def _last_log_line(experiment_dir):
    for log in sorted(pathlib.Path(experiment_dir).glob("slurm-*.out")):
        try:
            lines = [ln.strip() for ln in log.read_text(errors="replace").splitlines() if ln.strip()]
        except OSError:
            continue
        if lines:
            return lines[-1]
    return ""


def unfinished_status(experiment_dir, job, slurm_state):
    """(outcome, reason) for a run without summary.json and without a recognised fatal line."""
    started = any(pathlib.Path(experiment_dir).glob("slurm-*.out"))
    note = NO_SACCT_NOTE if slurm_state is None and not shutil.which("sacct") else ""
    if slurm_state in OPEN_STATES or (slurm_state is None and started):
        return ("RUNNING" if started else "PENDING"), (
            "no summary.json yet; the job is in progress" if started else "no Slurm log yet; the job has not started"
        ) + note
    if slurm_state is None:
        return "PENDING", "no Slurm log yet; the job has not started" + note
    # Slurm says it ended, but the launch script never wrote summary.json and printed no recognised error
    tail = _last_log_line(experiment_dir)
    return "FAIL", f"job {job} ended ({slurm_state}) without writing summary.json" + (
        f"; last log line: {_one_line(tail, 120)}" if tail else ""
    )


def launch_cause(errors):
    """One-line reading of an early launcher abort, for the cases that recur; empty when unsure."""
    text = " ".join(errors)
    if "endpoint check" in text:
        if "HTTP 503" in text or "overloaded" in text.lower() or "HTTP 502" in text or "HTTP 504" in text:
            return (
                "cause: the inference endpoint's model backend was overloaded or down; "
                "not a cluster or recipe problem, retry later"
            )
        if "could not connect" in text or "could not reach" in text:
            return "cause: the compute node has no route to the endpoint (DNS, route or firewall)"
        if "no response within" in text or "timed out" in text.lower():
            return (
                "cause: the inference endpoint accepted the connection but never answered; "
                "the model backend is not serving, retry later"
            )
        if "HTTP 401" in text or "HTTP 403" in text:
            return "cause: the API key was rejected; fix PUBLIC_NVIDIA_API_KEY in cluster_config.yaml"
        if "HTTP 404" in text:
            if "backend deployment is gone" in text:
                return (
                    "cause: the model is still in the endpoint's catalog but its backend deployment was removed "
                    "or is being redeployed; not a config problem, retry later"
                )
            return (
                "cause: the model id is not served at this endpoint; "
                "check the trajectory's model against GET <base>/models"
            )
    if "[launch] ERROR: server" in text:
        if "exposes" in text and "GPUs" in text:
            return (
                "cause: the allocation exposes fewer GPUs than the server's tensor parallelism; submit the gpu "
                "recipe with --scale 4 (one node) on the GPU partition"
            )
        if "serving image missing" in text:
            return "cause: the vLLM image is not installed; re-run llmb-install for the gpu workload"
        if "model snapshot missing" in text:
            return (
                "cause: the model weights are not installed; re-run llmb-install for the gpu workload "
                "(downloads.huggingface.repos)"
            )
        if "did not become ready" in text or "exited before it became ready" in text:
            return (
                "cause: the vLLM server did not come up; read server/vllm.log in the run directory "
                "(model load, GPU memory, or a flag the serving image does not support)"
            )
        if "already in use" in text:
            return "cause: another process holds the server port on this node; set RUN_CONF_SERVER_PORT"
    if "allocation exposes" in text:
        return (
            "cause: the Slurm allocation has fewer CPUs than RUN_CONF_TASKTRACE_CPUS asks for (typical on a CPU "
            "partition); raise the request with --slurm-arg cpus-per-task=N or lower RUN_CONF_TASKTRACE_CPUS"
        )
    if "image" in text.lower() and ("missing" in text.lower() or "not found" in text.lower()):
        return "cause: the task image is not installed; re-run llmb-install or point RUN_CONF_IMAGE at the .sqsh"
    return ""


# Styling mirrors `llmb-run jobs`: `Key : value` rows like `jobs show`, a rule under the
# header like the jobs table, color only on a terminal (and never when NO_COLOR is set).
_KEY_WIDTH = 12
_RULE_WIDTH = 100


class _Style:
    def __init__(self, enabled):
        self.enabled = enabled

    def _wrap(self, code, text):
        return f"\033[{code}m{text}\033[0m" if self.enabled else text

    def outcome(self, text):
        code = {"PASS": "1;32", "PENDING": "1;33", "RUNNING": "1;33"}.get(text, "1;31")
        return self._wrap(code, text)

    def bold(self, text):
        return self._wrap("1", text)

    def dim(self, text):
        return self._wrap("2", text)

    def cause(self, text):
        return self._wrap("33", text)

    def fix(self, text):
        return self._wrap("36", text)


def _use_color(args):
    if getattr(args, "no_color", False) or os.environ.get("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def _row(key, value):
    return f"{key:<{_KEY_WIDTH}}: {value}"


def _job_id(experiment_dir):
    for log in sorted(pathlib.Path(experiment_dir).glob("slurm-*.out")):
        return log.stem.split("-", 1)[1]
    for cfg in sorted(pathlib.Path(experiment_dir).glob("llmb-config_*.yaml")):
        return cfg.stem.split("_", 1)[1]
    return ""


def _fmt_sec(value):
    return f"{float(value):.1f}s" if isinstance(value, (int, float)) else "n/a"


def run_diagnostics(summary, run_dir):
    """The reasons to show for a run. Empty on PASS. A summary.json without a diagnostics field gets
    them derived from the replay report next to it, and the LLM-call check's reason is appended when
    missing. Stored "tool time:" and "llm time:" reasons are not shown: time is reported, not judged."""
    if summary_ok(summary):
        return []
    diagnostics = summary.get("diagnostics")
    if diagnostics is None:
        diagnostics = replay_diagnostics(find_replay_report(run_dir), live_tool_outputs(run_dir / "capture"))
    diagnostics = diagnostics if isinstance(diagnostics, list) else []
    diagnostics = [str(d) for d in diagnostics if not str(d).startswith(("tool time:", "llm time:"))]
    calls = llm_calls_diagnostic(summary)
    if calls and not any(d.startswith("llm calls:") for d in diagnostics):
        diagnostics.append(calls)
    return diagnostics


def _server_row(server):
    """One line on the vLLM server a gpu run brought up."""
    engine = " ".join(p for p in (server.get("engine"), server.get("version")) if p) or "server"
    parts = [engine]
    if server.get("served_model"):
        parts.append(server["served_model"])
    if server.get("tp"):
        parts.append(f"TP{server['tp']}")
    if server.get("gpu_count"):
        parts.append(f"{server['gpu_count']}x {server.get('gpu_name') or 'GPU'}")
    if server.get("startup_sec") is not None:
        parts.append(f"ready in {_fmt_sec(server['startup_sec'])}")
    return "  |  ".join(parts)


def _llm_rows(summary):
    """The LLM time row of a gpu run: the same shape as the Tool time row, plus the call counts."""
    comparison = summary.get("llm_comparison")
    if comparison is None:
        return []
    llm = summary.get("llm_replay") or {}
    if not llm.get("ran"):
        return [("LLM time", "llm replay did not run")]
    calls = llm.get("summary") or {}
    compared = compared_with_recording(summary, llm=True)
    if compared:
        ratio, live_sec = llm_time_ratio(comparison)
        text = (
            f"{_fmt_ratio(ratio)} replay/live  "
            f"(replay {_fmt_sec(comparison.get('llm_wait_replay_sec'))}, live {_fmt_sec(live_sec)})"
        )
    else:
        text = f"{_fmt_sec(comparison.get('llm_wait_replay_sec'))} replay"
    text += f"  |  {_fmt_count(calls.get('calls_replayed'))}/{_fmt_count(calls.get('calls_recorded'))} calls"
    if comparison.get("calls_failed"):
        text += f"  |  {comparison['calls_failed']} failed"
    if llm.get("pacing"):
        text += f"  |  pacing {llm['pacing']}"
    return [("LLM time", text)]


def verdict_rows(summary, path):
    """`Key : value` rows describing the run; the Outcome row comes first."""
    ok = summary_ok(summary)
    rows = [("Outcome", "PASS" if ok else "FAIL"), ("Task", summary.get("trace_id") or "?")]
    job = _job_id(path.parent)
    if job:
        rows.append(("Job", job))
    if summary.get("server"):
        rows.append(("Server", _server_row(summary["server"])))
    comparison = summary.get("comparison") or {}
    if is_capture_workload(summary):
        capture, replay = summary.get("capture") or {}, summary.get("replay") or {}
        if capture.get("model_id"):
            rows.append(("Model", capture["model_id"]))
        if capture.get("stage_exit_status") not in (None, 0):
            rows.append(("Capture", f"stage exit {capture['stage_exit_status']} (no replayable trajectory)"))
        else:
            rows.append(
                (
                    "Capture",
                    f"reward {capture.get('reward')}  |  {_fmt_count(capture.get('llm_calls'))} llm calls  |  "
                    f"{_fmt_count(capture.get('tool_actions_executed'))}/"
                    f"{_fmt_count(capture.get('tool_actions_recorded'))} "
                    f"tool actions  |  llm wait {_fmt_sec(capture.get('llm_wait_sec'))}  |  "
                    f"tool busy {_fmt_sec(capture.get('tool_busy_sec'))}",
                )
            )
        if not replay.get("ran"):
            rows.append(("Replay", "did not run"))
            return rows
        replay_summary = replay.get("summary") or {}
        pacing = replay.get("pacing")
        identity = (summary.get("image_identity_check") or {}).get("match")
        rows.append(
            ("Image", "same sha256 in capture and replay" if identity else "sha256 differs between capture and replay")
        )
    else:
        replay_summary = summary.get("replay") or {}
        pacing = summary.get("pacing")
    rows.append(
        (
            "Replay",
            f"{_fmt_count(replay_summary.get('actions_executed'))}/"
            f"{_fmt_count(replay_summary.get('actions_recorded'))} "
            f"actions  |  {_fmt_count(replay_summary.get('source_outcome_mismatch_count'))} mismatches  |  "
            f"{_fmt_count(replay_summary.get('actions_timed_out', 0))} timed out"
            + (f"  |  pacing {pacing}" if pacing else ""),
        )
    )
    compared = compared_with_recording(summary)  # only capture+replay of a timed recording compares
    if compared:
        ratio, live_sec = tool_time_ratio(comparison)
        tool_text = (
            f"{_fmt_ratio(ratio)} replay/live  "
            f"(replay {_fmt_sec(comparison.get('tool_busy_replay_sec'))}, live {_fmt_sec(live_sec)})"
        )
    else:
        tool_text = f"{_fmt_sec(comparison.get('tool_busy_replay_sec'))} replay"
    rows.append(("Tool time", tool_text))
    rows.extend(_llm_rows(summary))
    wall = replay_summary.get("wall_time_sec")
    recorded = replay_summary.get("source_interval_sec") if compared else None
    if wall is not None:
        rows.append(
            ("Wall time", f"{_fmt_sec(wall)} replay" + (f"  (recorded {_fmt_sec(recorded)})" if recorded else ""))
        )
    return rows


EXIT_NOT_FINISHED = 2  # distinct from PASS (0) and FAIL (1) so scripts never mistake an unfinished run for either


def render_verdict(rows, diagnostics, style, path=None, why_title="Why it failed"):
    """The styled report: Outcome header, rule, detail rows, and a 'Why' section on FAIL."""
    lines = []
    for key, value in rows:
        if key == "Outcome":
            lines.append(_row(style.bold(key) if style.enabled else key, style.outcome(value)))
            lines.append(style.dim("─" * _RULE_WIDTH))
        else:
            lines.append(_row(key, value))
    if path is not None:
        lines.append(_row("Summary", str(path)))
    if diagnostics:
        lines.append("")
        lines.append(style.bold(why_title))
        lines.append(style.dim("─" * _RULE_WIDTH))
        for line in diagnostics:
            line = str(line)
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith("cause:"):
                stripped = style.cause(stripped)
            elif stripped.startswith("fix:"):
                stripped = style.fix(stripped)
            lines.append(f"  {indent}{stripped}")
    return "\n".join(lines)


def locate_summary(target, style=None, oneline=False):
    """Accept the experiment directory (what `llmb-run jobs log <id> --dir` prints) or summary.json itself."""
    style = style or _Style(False)
    path = pathlib.Path(target)
    if path.is_dir():
        path = path / "summary.json"
    elif path.name != "summary.json":
        sys.exit(
            f"[verdict] {path} is neither an experiment directory nor a summary.json "
            "(use the directory `llmb-run jobs log <job_id> --dir` prints)"
        )
    summary = read_json(path) if path.is_file() else None
    if not summary:
        job = _job_id(path.parent)
        state = slurm_states([job]).get(job)
        errors = launch_errors(path.parent)
        if summary is not None:  # the file exists but is empty, truncated or not a JSON object
            if state in OPEN_STATES:
                outcome, code = "RUNNING", EXIT_NOT_FINISHED
                diagnostics = ["summary.json is incomplete; the job is still writing it"]
            else:
                outcome, code = "FAIL", 1
                diagnostics = ["summary.json is empty or not valid JSON; the run ended while writing it"]
        elif errors and state not in OPEN_STATES:
            outcome, code = "FAIL", 1
            diagnostics = [launch_error_header(errors)] + [f"  {line}" for line in errors]
            cause = launch_cause(errors)
            if cause:
                diagnostics.append(cause)
        else:
            outcome, reason = unfinished_status(path.parent, job, state)
            code = 1 if outcome == "FAIL" else EXIT_NOT_FINISHED
            diagnostics = [reason]
            if errors:  # the job is still running; the lines may be benign, so they are notes, not a verdict
                diagnostics.append("error lines so far (the job has not finished, so not yet a verdict):")
                diagnostics.extend(f"  {line}" for line in errors)
        what = "no summary.json" if summary is None else "unreadable summary.json"
        if oneline:
            print(f"Outcome: {outcome}  {what} at {path}")
            for line in diagnostics:
                print(f"  {line}")
        else:
            rows = [("Outcome", outcome), ("Task", f"unknown ({what})")]
            if job:
                rows.append(("Job", job))
            title = "Why it failed" if outcome == "FAIL" else "Status"
            print(render_verdict(rows, diagnostics, style, path=path, why_title=title))
        sys.exit(code)
    return path


RUN_MARKER_PREFIXES = ("slurm-", "llmb-config_")
MAX_SCAN_DEPTH = 6  # workloads/<workload>/experiments/<name>/<id> is depth 4 from an install root


def _run_markers(entries):
    """Whether a directory listing identifies an experiment directory: summary.json, a Slurm
    log of an aborted launch, or just the llmb-config of a job that has not started yet."""
    for name in entries:
        if name == "summary.json" or name.startswith(RUN_MARKER_PREFIXES):
            return True
    return False


def _is_run_dir(path):
    try:
        return path.is_dir() and _run_markers(os.listdir(path))
    except OSError:
        return False


def find_run_dirs(root):
    """Every run directory at or below root, oldest first (llmb-run names them by submit epoch).

    Each directory is listed once and the walk stops at a run directory, so the large trees
    inside a run (capture/, profile/, trajectories/) are never entered; on a networked
    filesystem that is the difference between milliseconds and a full crawl."""
    root = pathlib.Path(root)
    found = []
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            with os.scandir(directory) as it:
                entries = list(it)
        except OSError:
            continue
        if _run_markers(e.name for e in entries):
            found.append(directory)
            continue
        if depth >= MAX_SCAN_DEPTH:
            continue
        for e in entries:
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append((pathlib.Path(e.path), depth + 1))
            except OSError:
                continue
    return sorted(found, key=lambda d: (d.name, str(d)))


GPU_WORKLOAD_RE = re.compile(r"_gpu(?![a-z0-9])")  # the *_gpu workloads; not the `_gpus4` of an experiment dir name


def run_mode(run_dir, summary=None):
    """'capture+replay' or 'replay', plus ' gpu' for the gpu recipes, from summary.json, else the
    llmb-config, else the path."""
    workload = (summary or {}).get("workload") or ""
    if not workload:
        for cfg in sorted(run_dir.glob("llmb-config_*.yaml")):
            try:
                text = cfg.read_text()
            except OSError:
                continue
            for line in text.splitlines():
                if "model_name:" in line:
                    workload = line.split(":", 1)[1].strip().strip("'\"")
                    break
        # llmb-run names the experiment directory <workload key>_<model size>_<dtype>_gpus<N>
        workload = workload or f"{run_dir.parent.name}/{run_dir.name}"
    mode = "capture+replay" if "capture" in workload else "replay"
    return mode + " gpu" if GPU_WORKLOAD_RE.search(workload) else mode


def run_row(run_dir, states=None):
    """One table row for a run: (job, outcome, mode, llm wait, tool busy, replay, tool time, llm time, notes).

    `states` is the sacct job-id -> state map from `slurm_states`, when available."""
    summary_path = run_dir / "summary.json"
    job = _job_id(run_dir) or run_dir.name
    mode = run_mode(run_dir)
    state = (states or {}).get(job)
    summary = read_json(summary_path) if summary_path.is_file() else None
    if not summary:
        if summary is not None:  # exists but empty, truncated or not a JSON object
            if state in OPEN_STATES:
                return job, "RUNNING", mode, "-", "-", "in progress", "-", "-", "summary.json is still being written"
            return job, "FAIL", mode, "-", "-", "did not finish", "-", "-", "summary.json is empty or not valid JSON"
        errors = launch_errors(run_dir)
        if not errors or state in OPEN_STATES:
            outcome, reason = unfinished_status(run_dir, job, state)
            replay_txt = {"PENDING": "not started", "RUNNING": "in progress"}.get(outcome, "did not run")
            return job, outcome, mode, "-", "-", replay_txt, "-", "-", reason
        reason = launch_cause(errors) or errors[0]
        return job, "FAIL", mode, "-", "-", "did not run", "-", "-", _strip_cause(reason)
    ok = summary_ok(summary)
    mode = run_mode(run_dir, summary)
    if is_capture_workload(summary):
        capture, replay = summary.get("capture") or {}, summary.get("replay") or {}
        if capture.get("stage_exit_status") not in (None, 0):
            llm_txt, tool_txt = "-", "-"  # the live run left nothing; the reason goes in Notes
        else:
            llm_txt = _fmt_sec(capture.get("llm_wait_sec"))
            tool_txt = _fmt_sec(capture.get("tool_busy_sec"))
        rs = replay.get("summary") or {} if replay.get("ran") else {}
    else:
        llm_txt, tool_txt = "-", "-"  # replay-only: the trace was recorded elsewhere
        rs = summary.get("replay") or {}
    replay_txt = (
        f"{_fmt_count(rs.get('actions_executed'))}/{_fmt_count(rs.get('actions_recorded'))} act, "
        f"{_fmt_count(rs.get('source_outcome_mismatch_count'))} mism"
        + (f", {rs['actions_timed_out']} t/o" if rs.get("actions_timed_out") else "")
        if rs
        else "did not run"
    )
    ratio, llm_ratio = "-", "-"  # the replay-only recipes report their time without comparing it
    if compared_with_recording(summary):
        ratio_value, _ = tool_time_ratio(summary.get("comparison"))
        ratio = f"{_fmt_ratio(ratio_value)} (live)" if ratio_value is not None else "-"
        llm_ratio_value, _ = llm_time_ratio(summary.get("llm_comparison"))
        llm_ratio = f"{_fmt_ratio(llm_ratio_value)} (live)" if llm_ratio_value is not None else "-"
    diagnostics = run_diagnostics(summary, run_dir)
    reason = ""
    if not ok:
        first = next((d.strip() for d in diagnostics if not d.lstrip().startswith(("cause:", "fix:"))), "")
        cause = next((_strip_cause(d) for d in diagnostics if d.lstrip().startswith("cause:")), "")
        reason = cause or first or "see report"
    return job, "PASS" if ok else "FAIL", mode, llm_txt, tool_txt, replay_txt, ratio, llm_ratio, reason


def render_table(rows, style):
    """A `llmb-run jobs`-style table: bold header, a rule, no edges."""
    headers = ("Job ID", "Outcome", "Mode", "LLM wait", "Tool busy", "Replay", "Tool time", "LLM time", "Notes")
    right = {3, 4, 6, 7}  # numeric columns: LLM wait, Tool busy, Tool time, LLM time
    fixed = len(headers) - 1  # every column but the last is padded to its widest cell
    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row[:fixed]):
            widths[i] = max(widths[i], len(str(cell)))

    def fmt(cells, styled=None):
        out = []
        for i, cell in enumerate(cells):
            text = str(cell)
            pad = " " * (widths[i] - len(text)) if i < fixed else ""
            cell = styled[i] if styled else text
            out.append(pad + cell if i in right else cell + pad)
        return "  ".join(out).rstrip()

    lines = [
        style.bold(fmt(headers)) if style.enabled else fmt(headers),
        style.dim("─" * min(_RULE_WIDTH, sum(widths) + 2 * 5 + 40)),
    ]
    for row in rows:
        styled = [str(c) for c in row]
        styled[1] = style.outcome(row[1])
        lines.append(fmt(row, styled))
    passed = sum(1 for r in rows if r[1] == "PASS")
    open_states = {state: sum(1 for r in rows if r[1] == state) for state in ("RUNNING", "PENDING")}
    failed = len(rows) - passed - sum(open_states.values())
    lines.append("")
    tail = "".join(f", {n} {state}" for state, n in open_states.items() if n)
    lines.append(f"{len(rows)} runs: {passed} PASS, {failed} FAIL{tail}")
    return "\n".join(lines)


def cmd_verdict(args):
    style = _Style(_use_color(args))
    targets = [pathlib.Path(t) for t in args.target]
    # A single run (directory or its summary.json) gets the full report.
    if len(targets) == 1 and (targets[0].is_file() or _is_run_dir(targets[0])):
        path = locate_summary(targets[0], style=style, oneline=args.oneline)
        summary = read_json(path)
        diagnostics = run_diagnostics(summary, path.parent)
        if args.oneline:
            print(verdict_line(summary))
            for line in diagnostics or []:
                print(f"  {line}")
        else:
            print(render_verdict(verdict_rows(summary, path), diagnostics, style, path=path))
        return 0 if summary_ok(summary) else 1
    # Several targets, or a directory holding runs: one row per run.
    run_dirs = []
    for t in targets:
        if t.is_file():
            t = t.parent
        found = find_run_dirs(t)
        if not found:
            print(f"[verdict] no runs found under {t}", file=sys.stderr)
        run_dirs.extend(d for d in found if d not in run_dirs)
    if not run_dirs:
        return 1
    states = slurm_states([_job_id(d) for d in run_dirs])
    rows = [run_row(d, states) for d in run_dirs]
    if args.oneline:
        for row in rows:
            print("  ".join(str(c) for c in row))
    else:
        print(render_table(rows, style))
    return 0 if all(r[1] in ("PASS", "PENDING", "RUNNING") for r in rows) else 1


# --------------------------------------------------------------------------- cli


def _add_gpu_args(parser, workload):
    """summarize-* inputs the gpu recipes add: their workload name and the LLM side of the run."""
    parser.add_argument(
        "--workload", default=workload, help="general.workload of the recipe (the gpu recipes pass theirs)"
    )
    parser.add_argument("--llm-report", default="", help="llm-replay report; the summary then carries the LLM side")
    parser.add_argument("--llm-exit", default="", help="llm-replay exit status")
    parser.add_argument("--server-info", default="", help="server.json written by server-info")


def build_parser():
    p = argparse.ArgumentParser(
        prog="tasktrace_recipe.py", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("cpuset")
    s.add_argument("--count", type=int, default=0, help="CPUs to print (after --skip)")
    s.add_argument("--skip", type=int, default=0, help="CPUs at the front of the list to leave out")
    s.add_argument("--all-remaining", action="store_true", help="print every CPU after the first --skip")
    s.add_argument("--min", type=int, default=1, help="with --all-remaining: fail below this many")
    s.set_defaults(fn=cmd_cpuset)
    s = sub.add_parser("image-cache-key")
    s.add_argument("--image-ref", required=True)
    s.set_defaults(fn=cmd_image_cache_key)
    s = sub.add_parser("image-file-name")
    s.add_argument("--image-ref", required=True)
    s.set_defaults(fn=cmd_image_file_name)
    s = sub.add_parser("server-info")
    s.add_argument("--out", required=True)
    s.set_defaults(fn=cmd_server_info)
    s = sub.add_parser("llm-replay", help="replay the LLM side of a trajectory against a live server (gpu recipes)")
    s.add_argument("--trace", required=True)
    s.add_argument("--api-base", default="")
    s.add_argument("--model", default="", help="the served model id")
    s.add_argument("--pacing", choices=("recorded", "unpaced"), default="recorded")
    s.add_argument("--output", default="", help="where to write the report (" + LLM_REPLAY_REPORT + ")")
    s.add_argument("--timeout-sec", type=int, default=1800, help="per-call client timeout")
    s.add_argument("--seed", type=int, default=20260921, help="seed for the synthetic prompt tokens")
    s.add_argument(
        "--start-delay",
        type=float,
        default=0.0,
        help="seconds from the start (or from --align-file appearing) to the first call, recorded pacing; may be negative",
    )
    s.add_argument("--align-file", default="", help="time the replay from this file's appearance (see --lead-only)")
    s.add_argument("--align-pid", type=int, default=0, help="stop waiting for --align-file once this process ended")
    s.add_argument("--align-timeout-sec", type=float, default=1800.0, help="how long to wait for --align-file")
    s.add_argument(
        "--loop-sync-dir",
        default="",
        help="replay in the recorded agent loop's order with the tool replay, through this directory (unpaced; "
        "for a trajectory without timings, see tasktrace profile trace --loop-order); --align-pid stops the wait",
    )
    s.add_argument(
        "--lead-only",
        action="store_true",
        help="print LLM_START_OFFSET (a shell assignment), the recorded first LLM call's offset from the first tool action, and exit",
    )
    s.add_argument("--warmup-only", action="store_true", help="send the untimed warm-up requests and exit")
    s.set_defaults(fn=cmd_llm_replay)
    s = sub.add_parser("elf-check")
    s.add_argument("--path", required=True)
    s.add_argument("--arch", choices=sorted(ELF_MACHINES), required=True)
    s.add_argument("--tag", default="")
    s.set_defaults(fn=cmd_elf_check)
    s = sub.add_parser("resolve-trace")
    s.add_argument("--root", required=True)
    s.add_argument("--selector", required=True)
    s.set_defaults(fn=cmd_resolve_trace)
    s = sub.add_parser("trace-timed")
    s.add_argument("--trace", required=True)
    s.set_defaults(fn=cmd_trace_timed)
    s = sub.add_parser("endpoint-check")
    s.add_argument("--base", required=True)
    s.add_argument("--model", required=True)
    s.set_defaults(fn=cmd_endpoint_check)
    s = sub.add_parser("stage")
    s.add_argument("--trace", default="")
    s.add_argument("--staged-dir", required=True)
    s.add_argument("--provenance", required=True)
    s.set_defaults(fn=cmd_stage)
    s = sub.add_parser("secret-scan")
    s.add_argument("--root", required=True)
    s.add_argument("--report", required=True)
    s.set_defaults(fn=cmd_secret_scan)
    s = sub.add_parser("summarize-capture")
    for name in ("--provenance", "--profile-dir", "--summary", "--scrub-report", "--pacing", "--cpus", "--stage-exit"):
        s.add_argument(name, required=True)
    s.add_argument("--replay-exit", default="")
    s.add_argument("--cpuset", default="")
    s.add_argument(
        "--capture-dir",
        default="",
        help="tasktrace-capture output dir; its harbor trajectory supplies live tool outputs",
    )
    s.add_argument(
        "--replay-image-sha256",
        default="",
        help="current-container runtime only: the digest of the image the replay's container was created from "
        "(the pipeline report has no image identity when nothing was created); default $TT_IMAGE_SHA256",
    )

    _add_gpu_args(s, "tasktrace_capture_replay")
    s.set_defaults(fn=cmd_summarize_capture)
    s = sub.add_parser("summarize-replay")
    for name in ("--out-dir", "--summary", "--trace-id", "--pacing", "--cpus", "--replay-exit"):
        s.add_argument(name, required=True)
    s.add_argument("--cpuset", default="")
    _add_gpu_args(s, "tasktrace_replay")
    s.set_defaults(fn=cmd_summarize_replay)
    s = sub.add_parser("verdict", help="print the outcome (PASS/FAIL) and the reasons for a finished run")
    s.add_argument(
        "target",
        nargs="+",
        help="a run: experiment directory (from `llmb-run jobs log <id> --dir`) or its summary.json, for the full "
        "report; or one or more directories holding runs (e.g. a workload's experiments/ tree), for a table",
    )
    s.add_argument(
        "--oneline", action="store_true", help="single 'Outcome: PASS/FAIL ...' line plus reasons; for scripts"
    )
    s.add_argument("--no-color", action="store_true", help="disable ANSI color (also honours NO_COLOR)")
    s.set_defaults(fn=cmd_verdict)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
