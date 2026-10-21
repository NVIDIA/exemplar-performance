# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for tasktrace_recipe.py (run: python -m pytest inference/slurm_recipes/agentic/common/tests)."""

import hashlib
import importlib.util
import io
import json
import os
import pathlib
import re
import shutil
import urllib.error

import pytest

MODULE = pathlib.Path(__file__).resolve().parents[1] / "tasktrace_recipe.py"
spec = importlib.util.spec_from_file_location("tasktrace_recipe", MODULE)
tr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tr)


def run(argv, monkeypatch=None, stdin=""):
    if monkeypatch is not None and stdin is not None:
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
    return tr.main(argv)


# ---------------------------------------------------------------- cpuset / cache key


def test_cpuset_expands_ranges(monkeypatch, capsys):
    assert run(["cpuset", "--count", "6"], monkeypatch, "0-3,8-11\n") == 0
    assert capsys.readouterr().out.strip() == "0,1,2,3,8,9"


def test_cpuset_too_few(monkeypatch):
    with pytest.raises(SystemExit) as info:
        run(["cpuset", "--count", "4"], monkeypatch, "5,7")
    msg = str(info.value.code)
    assert msg.startswith("[launch] ERROR: allocation exposes 2 CPUs, need 4")
    assert "--slurm-arg cpus-per-task=N" in msg and "RUN_CONF_TASKTRACE_CPUS" in msg


def test_image_cache_key_matches_tasktrace_capture_rule():
    ref = "ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11490:latest"
    expected = (
        "ghcr.io_epoch-research_swe-bench.eval.arm64.django__django-11490_latest_"
        + hashlib.sha256(ref.encode()).hexdigest()[:8]
        + ".sqsh"
    )
    assert tr.image_cache_key(ref) == expected


# ---------------------------------------------------------------- endpoint check


class _Resp:
    status = 200

    def __init__(self, body):
        self._b = body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self):
        return self._b


def _http_error(code, body=b'{"error":"x"}'):
    return urllib.error.HTTPError("u", code, "msg", {}, io.BytesIO(body))


def test_endpoint_check_ok(monkeypatch, capsys):
    monkeypatch.setenv("TT_API_KEY", "nvapi-secretvalue")
    args = tr.build_parser().parse_args(["endpoint-check", "--base", "https://x/v1", "--model", "m"])
    assert tr.cmd_endpoint_check(args, urlopen=lambda req, timeout: _Resp(b'{"model":"m-served"}')) == 0
    assert "OK (HTTP 200, served model 'm-served')" in capsys.readouterr().out


@pytest.mark.parametrize(
    "code,rc,needle",
    [
        (429, 0, "WARNING HTTP 429"),
        (401, 1, "NV-Developer"),
        (404, 1, "not served"),
        (503, 1, "overloaded or down"),
        (500, 1, "see the response body"),
    ],
)
def test_endpoint_check_http_errors(monkeypatch, capsys, code, rc, needle):
    monkeypatch.setenv("TT_API_KEY", "nvapi-secretvalue")
    args = tr.build_parser().parse_args(["endpoint-check", "--base", "https://x/v1", "--model", "m"])

    def raise_it(req, timeout):
        raise _http_error(code, b"body with nvapi-secretvalue inside")

    assert tr.cmd_endpoint_check(args, urlopen=raise_it) == rc
    err = capsys.readouterr().err
    assert needle in err and "nvapi-secretvalue" not in err and "<REDACTED>" in err


@pytest.mark.parametrize(
    "catalog,needle",
    [
        (b'{"data":[{"id":"m"},{"id":"other"}]}', "backend deployment is gone"),  # listed -> backend removed
        (b'{"data":[{"id":"other"}]}', "is not in GET https://x/v1/models"),  # not listed -> wrong id
        (None, "check RUN_CONF_MODEL against GET"),  # catalog unreachable -> generic
    ],
)
def test_endpoint_check_404_consults_the_catalog(monkeypatch, capsys, catalog, needle):
    """A 404 from chat completions is diagnosed by the catalog: model listed in /models means its backend
    deployment is gone, not listed means a wrong model id, catalog unreachable means the generic hint."""
    monkeypatch.setenv("TT_API_KEY", "k")
    args = tr.build_parser().parse_args(["endpoint-check", "--base", "https://x/v1", "--model", "m"])

    def fake_urlopen(req, timeout):
        if req.full_url.endswith("/models"):
            if catalog is None:
                raise urllib.error.URLError("down")
            return _Resp(catalog)
        raise _http_error(404, b'{"detail":"Function id x version null: Specified function is not found"}')

    assert tr.cmd_endpoint_check(args, urlopen=fake_urlopen) == 1
    assert needle in capsys.readouterr().err


def test_endpoint_check_unreachable(monkeypatch, capsys):
    monkeypatch.setenv("TT_API_KEY", "k")
    args = tr.build_parser().parse_args(["endpoint-check", "--base", "https://x/v1", "--model", "m"])

    def raise_it(req, timeout):
        raise urllib.error.URLError("no route")

    assert tr.cmd_endpoint_check(args, urlopen=raise_it) == 1
    err = capsys.readouterr().err
    assert "could not connect" in err and "needs egress" in err


def test_endpoint_check_read_timeout_is_not_blamed_on_egress(monkeypatch, capsys):
    """Connected but silent: the backend is down, not the route."""
    import socket

    monkeypatch.setenv("TT_API_KEY", "k")
    args = tr.build_parser().parse_args(["endpoint-check", "--base", "https://x/v1", "--model", "m"])
    for exc in (TimeoutError("The read operation timed out"), socket.timeout("The read operation timed out")):

        def raise_it(req, timeout, exc=exc):
            raise exc

        assert tr.cmd_endpoint_check(args, urlopen=raise_it) == 1
        err = capsys.readouterr().err
        assert "no response within 90s" in err and "Egress is fine" in err and "needs egress" not in err

    def raise_connect_timeout(req, timeout):
        raise urllib.error.URLError(socket.timeout("timed out"))  # no route / firewall: never connected

    assert tr.cmd_endpoint_check(args, urlopen=raise_connect_timeout) == 1
    err = capsys.readouterr().err
    assert "could not connect" in err and "needs egress" in err and "Egress is fine" not in err


# ---------------------------------------------------------------- stage


def _trace_lines(schema="tasktrace-capture-cpu-trace.v1", replay_supported=True):
    rows = [
        {
            "record_type": "trace",
            "schema_version": schema,
            "replay_supported": replay_supported,
            "source_trial": "t__abc",
            "task": "swebench-verified/django__django-11490",
            "task_checksum": "sha",
            "limitations": ["Action 2 network"],
            "reward": 0.0,
        },
        {"record_type": "llm", "llm_index": 1, "wall_time_sec": 2.5},
        {
            "record_type": "action",
            "action_index": 1,
            "command": "ls",
            "wall_time_sec": 0.5,
            "started_at_epoch": 100.0,
            "finished_at_epoch": 100.5,
        },
        {"record_type": "llm", "llm_index": 2, "wall_time_sec": 3.5},
        {
            "record_type": "action",
            "action_index": 2,
            "command": "pip install x",
            "wall_time_sec": 4.0,
            "started_at_epoch": 104.0,
            "finished_at_epoch": 108.0,
        },
        {
            "record_type": "action",
            "action_index": 3,
            "command": "echo SUBMIT",
            "exception_info": "action was not executed",
        },
    ]
    return "\n".join(json.dumps(r) for r in rows) + "\n"


def _capture_layout(tmp_path, **kw):
    trial = tmp_path / "capture" / "runs" / "run1" / "trial1"
    (trial / "agent").mkdir(parents=True)
    trace = trial / "agent" / tr.TRAJECTORY_NAME
    trace.write_text(_trace_lines(**kw))
    (trial / "result.json").write_text(json.dumps({"agent_info": {"version": "1.2.3"}}))
    (tmp_path / "capture" / "report.json").write_text(
        json.dumps(
            {"runtime": "enroot", "hostname": "n1", "concurrency": {"1": {"timing": {"wall": 9}, "phases": {"a": 1}}}}
        )
    )
    return trace


def _stage_env(monkeypatch, tmp_path):
    vals = {
        "TT_SELECTOR": "sel",
        "TT_TRACE_MODEL": "m",
        "TT_TRACE_HARNESS": "h",
        "TT_TRACE_DATASET": "d",
        "TT_TRACE_ID": "t",
        "TT_AGENT": "mini-swe-agent",
        "TT_LIVE_MODEL": "openai/x",
        "TT_ENDPOINT_MODEL_ID": "x",
        "TT_API_BASE": "https://b/v1",
        "TT_API_KEY_VAR": "K",
        "TT_AGENT_TIMEOUT": "2400",
        "TT_CAPTURE_EXIT": "0",
        "TT_CAPTURE_STARTED_AT": "s",
        "TT_CAPTURE_FINISHED_AT": "f",
        "TT_CAPTURE_ARGS": "--runtime enroot",
        "TT_CAPTURE_DIR": str(tmp_path / "capture"),
        "TT_IMAGE_REF": "ghcr.io/x:latest",
        "TT_IMAGE_PATH": "/img.sqsh",
        "TT_IMAGE_SHA256": "abc",
        "TT_IMAGE_ARCH": "aarch64",
        "TT_VENV_VERSION_FILE": str(tmp_path / "venv.version"),
        "TT_TASKTRACE_VERSION_FILE": str(tmp_path / "tt.version"),
        "TT_NODE": "n1",
        "TT_JOB_ID": "42",
        "TT_GPU_TYPE": "gb300",
        "TT_CPUSET": "0,1,2,3",
        "TT_CPUS": "4",
    }
    for k, v in vals.items():
        monkeypatch.setenv(k, v)
    (tmp_path / "venv.version").write_text(
        "source=dir:1\ncommit=unknown\npython=3.12.12\nharbor=0.13.2\narch=aarch64\n"
    )
    (tmp_path / "tt.version").write_text("source=dir:2\ntarget=aarch64-unknown-linux-musl\n")


def test_stage_happy_path(tmp_path, monkeypatch, capsys):
    trace = _capture_layout(tmp_path)
    _stage_env(monkeypatch, tmp_path)
    staged = tmp_path / "staged"
    prov = staged / "provenance.json"
    rc = run(["stage", "--trace", str(trace), "--staged-dir", str(staged), "--provenance", str(prov)])
    assert rc == 0
    p = json.loads(prov.read_text())
    assert p["problems"] == []
    assert p["trace"]["sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert (staged / (tr.TRAJECTORY_NAME + ".sha256")).read_text().startswith(p["trace"]["sha256"])
    live = p["live_run"]
    assert live["llm_calls"] == 2 and live["tool_actions_recorded"] == 3 and live["tool_actions_executed"] == 2
    assert live["llm_wait_sec"] == 6.0 and live["tool_busy_sec"] == 4.5 and live["tool_interval_sec"] == 8.0
    assert live["agent"] == {"name": "mini-swe-agent", "version": "1.2.3"} and live["reward"] == 0.0
    assert live["cpuset"] == "0,1,2,3" and live["allocated_cpus"] == 4
    assert live["capture_report"]["timing"] == {"wall": 9} and live["capture_report"]["hostname"] == "n1"
    assert p["tools"]["tasktrace_capture"]["harbor_version"] == "0.13.2" and p["tools"]["tasktrace"]["target"].endswith(
        "musl"
    )
    assert "[stage] live run: reward=0.0 llm_calls=2 tool_actions=2/3" in capsys.readouterr().out


def test_stage_rejects_bad_schema_and_missing_trace(tmp_path, monkeypatch, capsys):
    _stage_env(monkeypatch, tmp_path)
    trace = _capture_layout(tmp_path, schema="something.v9")
    staged = tmp_path / "s1"
    assert (
        run(["stage", "--trace", str(trace), "--staged-dir", str(staged), "--provenance", str(staged / "p.json")]) == 1
    )
    assert "unsupported trace schema_version 'something.v9'" in capsys.readouterr().err
    staged2 = tmp_path / "s2"
    assert run(["stage", "--trace", "", "--staged-dir", str(staged2), "--provenance", str(staged2 / "p.json")]) == 1
    assert json.loads((staged2 / "p.json").read_text())["problems"] == ["capture produced no finalized trajectory"]


def test_stage_reports_malformed_trajectory(tmp_path, monkeypatch, capsys):
    _stage_env(monkeypatch, tmp_path)
    trace = _capture_layout(tmp_path)
    trace.write_bytes(trace.read_bytes() + b'{"record_type": "action", "truncated\n')
    staged = tmp_path / "s3"
    prov = staged / "p.json"
    assert run(["stage", "--trace", str(trace), "--staged-dir", str(staged), "--provenance", str(prov)]) == 1
    p = json.loads(prov.read_text())
    assert len(p["problems"]) == 1 and "is not valid JSON" in p["problems"][0]
    assert p["trace"]["sha256"] == hashlib.sha256(trace.read_bytes()).hexdigest()
    assert (staged / tr.TRAJECTORY_NAME).exists()
    assert "is not valid JSON" in capsys.readouterr().err


# ---------------------------------------------------------------- elf check


def _fake_elf(machine, interp=False, little=True):
    """A minimal 64-bit ELF: header, then a program header table with one LOAD and optionally one INTERP."""
    import struct

    end = "<" if little else ">"
    phdrs = [1] + ([3] if interp else [])  # PT_LOAD, PT_INTERP
    e_phoff, e_phentsize = 64, 56
    ident = b"\x7fELF" + bytes([2, 1 if little else 2, 1]) + b"\0" * 9
    hdr = ident + struct.pack(
        end + "HHIQQQIHHHHHH", 2, machine, 1, 0x400000, e_phoff, 0, 0, 64, e_phentsize, len(phdrs), 0, 0, 0
    )
    assert len(hdr) == 64
    table = b"".join(struct.pack(end + "IIQQQQQQ", p_type, 5, 0, 0, 0, 0, 0, 0) for p_type in phdrs)
    return hdr + table


def test_elf_check_accepts_static_binary_for_target(tmp_path, capsys):
    exe = tmp_path / "tasktrace"
    exe.write_bytes(_fake_elf(183))
    assert run(["elf-check", "--path", str(exe), "--arch", "aarch64", "--tag", "[t]"]) == 0
    assert "[t] verified: static ELF for aarch64 (machine 183)" in capsys.readouterr().out


@pytest.mark.parametrize(
    "payload, arch, needle",
    [
        (_fake_elf(183), "x86_64", "ELF machine 183, expected 62"),
        (_fake_elf(62, interp=True), "x86_64", "dynamic interpreter"),
        (_fake_elf(62, little=False), "x86_64", None),  # big-endian header is parsed correctly
        (b"#!/bin/sh\necho not an elf\n", "aarch64", "not a 64-bit ELF file"),
    ],
)
def test_elf_check_rejects_wrong_machine_dynamic_or_non_elf(tmp_path, capsys, payload, arch, needle):
    exe = tmp_path / "bin"
    exe.write_bytes(payload)
    rc = run(["elf-check", "--path", str(exe), "--arch", arch])
    err = capsys.readouterr().err
    if needle is None:
        assert rc == 0 and err == ""
    else:
        assert rc == 1 and needle in err


# ---------------------------------------------------------------- secret scan


def test_secret_scan_redacts_key_and_pattern(tmp_path, monkeypatch, capsys):
    root = tmp_path / "exp"
    (root / "sub").mkdir(parents=True)
    (root / "a.log").write_text("Authorization: Bearer nvapi-r-THISISTHECONFIGUREDKEY0001\n")
    (root / "sub" / "b.json").write_text(json.dumps({"k": "nvapi-Zz9_anotherkeythatlooksreal-XYZ"}))
    (root / "clean.txt").write_text("nothing here")
    monkeypatch.setenv("TT_SECRET", "nvapi-r-THISISTHECONFIGUREDKEY0001")
    assert run(["secret-scan", "--root", str(root), "--report", str(tmp_path / "scan.json")]) == 0
    rep = json.loads((tmp_path / "scan.json").read_text())
    assert (
        rep["files_scanned"] == 3
        and rep["files_with_configured_key"] == ["a.log"]
        and rep["files_with_nvapi_pattern"] == ["sub/b.json"]
    )
    assert "THISISTHECONFIGUREDKEY" not in (root / "a.log").read_text() and "<REDACTED>" in (root / "a.log").read_text()
    assert (root / "sub" / "b.json").read_text() == json.dumps({"k": "nvapi-<REDACTED>"})
    assert (root / "clean.txt").read_text() == "nothing here"
    assert "redacted in 2 file(s)" in capsys.readouterr().err


def test_secret_scan_skips_symlinks_without_following_them(tmp_path):
    # uv's cache in the mounted home links to /root/..., which is mode 700 on some nodes.
    root = tmp_path / "exp"
    root.mkdir()
    private = tmp_path / "private"
    private.mkdir()
    (root / "wheel").symlink_to(private / "archive")
    private.chmod(0o000)
    try:
        assert run(["secret-scan", "--root", str(root), "--report", str(tmp_path / "scan.json")]) == 0
    finally:
        private.chmod(0o700)
    assert json.loads((tmp_path / "scan.json").read_text())["files_scanned"] == 0


# ---------------------------------------------------------------- summaries


def _replay_report(mismatch=0):
    actions = [
        {
            "action_index": 1,
            "status": "ok",
            "source_status": "ok",
            "wall_time_sec": 0.4,
            "source_wall_time_sec": 0.5,
            "outcome_matches_source": True,
        },
        {
            "action_index": 2,
            "command": "cd /testbed && python run.py",
            "status": "failed" if mismatch else "ok",
            "source_status": "ok",
            "wall_time_sec": 3.0,
            "source_wall_time_sec": 4.0,
            "stdout_preview": "",
            "stderr_preview": "Traceback (most recent call last):\n  File x\n"
            "ModuleNotFoundError: No module named 'attr'",
            "outcome_matches_source": not mismatch,
        },
        {
            "action_index": 3,
            "status": "skipped_not_executed",
            "source_status": "skipped_not_executed",
            "outcome_matches_source": True,
        },
    ]
    return {
        "summary": {
            "actions_recorded": 3,
            "actions_executed": 2,
            "actions_skipped_not_executed": 1,
            "actions_ok": 2 - mismatch,
            "actions_failed": mismatch,
            "actions_timed_out": 0,
            "source_outcome_mismatch_count": mismatch,
            "wall_time_sec": 8.0,
        },
        "actions": actions,
    }


def _profile_dir(tmp_path, mismatch=0, sha="imgsha"):
    prof = tmp_path / "profile"
    (prof / "replay").mkdir(parents=True)
    (prof / "replay" / "replay_report.json").write_text(json.dumps(_replay_report(mismatch)))
    (prof / "pipeline_report.json").write_text(
        json.dumps(
            {"status": "valid", "profiling_invalid_reasons": [], "image_identity": {"sha256": sha, "runtime": "enroot"}}
        )
    )
    return prof


def test_summarize_replay_reports_tool_time_without_comparing(tmp_path, capsys):
    prof = _profile_dir(tmp_path)
    rc = run(
        [
            "summarize-replay",
            "--out-dir",
            str(prof),
            "--summary",
            str(tmp_path / "s.json"),
            "--trace-id",
            "t",
            "--pacing",
            "recorded",
            "--cpus",
            "4",
            "--cpuset",
            "0,1,2,3",
            "--replay-exit",
            "0",
        ]
    )
    assert rc == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["fidelity"] == "PASS" and s["replay"]["actions_executed"] == 2 and s["cpuset"] == "0,1,2,3"
    assert s["comparison"] == {"tool_busy_replay_sec": 3.4}


def test_summarize_replay_reports_a_slower_replay_without_failing(tmp_path):
    prof = _profile_dir(tmp_path)
    report_path = prof / "replay" / "replay_report.json"
    report = json.loads(report_path.read_text())
    for action in report["actions"]:
        if action.get("source_wall_time_sec"):
            action["wall_time_sec"] = round(action["source_wall_time_sec"] * 1.2, 3)
    report_path.write_text(json.dumps(report))
    argv = ["summarize-replay", "--out-dir", str(prof), "--summary", str(tmp_path / "s.json"), "--trace-id", "t"]
    argv += ["--pacing", "recorded", "--cpus", "4", "--cpuset", "", "--replay-exit", "0"]
    assert run(argv) == 0  # 20% slower than the recording: reported, not compared
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["fidelity"] == "PASS" and s["tasktrace_exit_status"] == 0
    assert s["comparison"] == {"tool_busy_replay_sec": 5.4}
    assert s["diagnostics"] == []


def test_verdict_replay_only_shows_no_comparison_when_the_summary_has_ratios(tmp_path, capsys):
    """A replay-only summary that carries ratios: the verdict shows only the replay's own time."""
    comparison = {
        "replay_to_live_tool_ratio": 0.9066,
        "tool_busy_replay_sec": 26.6,
        "tool_busy_live_sec": 28.8,
        "reference": "Grace median",
        "reference_tool_busy_sec": 26.1,
        "replay_to_reference_tool_ratio": 1.0192,
    }
    replay = {
        "actions_recorded": 114,
        "actions_executed": 113,
        "source_outcome_mismatch_count": 0,
        "wall_time_sec": 25.9,
        "source_interval_sec": 1662.0,
    }
    exp = _write_summary(tmp_path, comparison=comparison, replay=replay)
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert capsys.readouterr().out == (
        "Outcome: PASS  django__django-11490  replay: tool time 26.6s, 113/114 actions, 0 mismatches\n"
    )
    assert run(["verdict", "--no-color", str(exp)]) == 0
    styled = capsys.readouterr().out
    assert "Tool time   : 26.6s replay\n" in styled and "Wall time   : 25.9s replay\n" in styled
    assert "replay/" not in styled and "recorded" not in styled


def _timeout_cascade_report():
    """A network-dependent action times out and two later actions diverge because of it."""
    actions = [
        {"action_index": 91, "status": "failed", "source_status": "failed", "outcome_matches_source": True},
        {
            "action_index": 92,
            "command": "apt-get update && apt-get install -y python3.10-venv 2>&1 | tail -10",
            "status": "timed_out",
            "source_status": "ok",
            "source_wall_time_sec": 5.343,
            "wall_time_sec": 30.01,
            "stdout_preview": "Get:1 http://ports.ubuntu.com ...",
            "outcome_matches_source": False,
        },
        {
            "action_index": 93,
            "command": "cd /testbed && python3.10 -m venv /tmp/venv",
            "status": "failed",
            "source_status": "ok",
            "returncode": 1,
            "stdout_preview": "The virtual environment was not created successfully because ensurepip is not\n"
            "available.",
            "stderr_preview": "",
            "outcome_matches_source": False,
        },
        {
            "action_index": 110,
            "command": "/tmp/venv/bin/python /tmp/simple_test2.py",
            "status": "failed",
            "source_status": "ok",
            "stdout_preview": 'Traceback (most recent call last):\n  File "/tmp/simple_test2.py"\n'
            "ModuleNotFoundError: No module named 'attr'\n",
            "stderr_preview": "",
            "outcome_matches_source": False,
        },
        {"action_index": 114, "status": "skipped_not_executed", "source_status": "skipped_not_executed"},
    ]
    return {
        "action_timeout_sec": 30.0,
        "source_limitations": [
            "Action 91 performs package or network-dependent work",
            "Action 92 performs package or network-dependent work",
        ],
        "summary": {
            "actions_recorded": 114,
            "actions_executed": 113,
            "actions_skipped_not_executed": 1,
            "actions_timed_out": 1,
            "source_outcome_mismatch_count": 3,
        },
        "actions": actions,
    }


def test_replay_diagnostics_explains_timeout_cascade():
    lines = tr.replay_diagnostics(_timeout_cascade_report())
    assert lines[0] == (
        "action 92 timed out after 30s (recorded 5.3s): "
        "apt-get update && apt-get install -y python3.10-venv 2>&1 | tail -10"
    )
    assert lines[1].startswith("  cause: trace marks this action as package/network-dependent")
    assert lines[2].startswith("  fix: retry with RUN_CONF_TASKTRACE_TIMEOUT=300")
    assert lines[3].startswith(
        "action 93 ok -> failed, after timed-out action 92: The virtual environment was not created"
    )
    # a traceback is reported by its last line, not its first
    assert lines[4] == "action 110 ok -> failed, after timed-out action 92: ModuleNotFoundError: No module named 'attr'"
    assert len(lines) == 5


def _timed_out(idx, command, recorded, **extra):
    return dict(
        action_index=idx,
        command=command,
        status="timed_out",
        source_status="ok",
        source_wall_time_sec=recorded,
        wall_time_sec=30.0,
        outcome_matches_source=False,
        **extra,
    )


def test_timeout_causes_are_told_apart():
    timeout = 30.0

    def cause(action, network=False, earlier=()):
        return tr._timeout_cause(action, network, timeout, list(earlier))

    # recorded time near the limit -> slower node explanation
    c = cause(_timed_out(5, "python -m pytest testing/", 18.0))
    assert c[0].startswith("cause: the recording already used 60% of the limit") and "RUN_CONF_TASKTRACE_CPUS" in c[1]
    # an interactive command after a divergence -> prompt explanation, and a longer timeout is called out as useless
    c = cause(_timed_out(6, "python -m pytest test_repro.py --pdb", 0.4), earlier=[3])
    assert "wait for interactive input" in c[0] and "diverged at action 3" in c[0] and "will not help" in c[1]
    # the same interactive command with no earlier divergence is not blamed on a prompt
    c = cause(_timed_out(6, "python -m pytest test_repro.py --pdb", 0.4))
    assert c[0].startswith("cause: not identifiable from the report (not network-flagged")
    # a plain command after a divergence
    c = cause(_timed_out(7, "curl localhost:8000/health", 0.2), earlier=[2, 4])
    assert "diverged at action 4" in c[0] and "resolve the earlier divergence first" in c[1]
    # network flag wins over everything else
    c = cause(_timed_out(8, "pip install -e .", 29.0), network=True, earlier=[1])
    assert c[0].startswith("cause: trace marks this action as package/network-dependent")


def test_replay_diagnostics_timeout_after_divergence_is_linked():
    report = {
        "action_timeout_sec": 30.0,
        "summary": {"actions_recorded": 3, "actions_executed": 3, "actions_skipped_not_executed": 0},
        "actions": [
            {
                "action_index": 1,
                "status": "failed",
                "source_status": "ok",
                "outcome_matches_source": False,
                "stderr_preview": "boom",
            },
            _timed_out(2, "python -m pytest x.py --pdb", 0.3),
            {"action_index": 3, "status": "ok", "source_status": "ok", "outcome_matches_source": True},
        ],
    }
    lines = tr.replay_diagnostics(report)
    assert lines[0].startswith("action 2 timed out after 30s (recorded 0.3s)")
    assert "diverged at action 1" in lines[1]
    assert lines[3] == "action 1 ok -> failed: boom"


def test_timeout_matching_the_recording_is_reproduced_not_a_failure():
    matched = _timed_out(2, "cd /testbed && python tests/runtests.py 2>&1 | tail -50", 30.2)
    matched.update(source_status="timed_out", outcome_matches_source=True)
    report = {
        "action_timeout_sec": 30.0,
        "summary": {
            "actions_recorded": 3,
            "actions_executed": 3,
            "actions_skipped_not_executed": 0,
            "actions_timed_out": 1,
            "source_outcome_mismatch_count": 0,
        },
        "actions": [
            {"action_index": 1, "status": "ok", "source_status": "ok", "outcome_matches_source": True},
            matched,
            {"action_index": 3, "status": "failed", "source_status": "ok", "outcome_matches_source": False},
        ],
    }
    assert tr.fidelity_verdict({**report, "actions": report["actions"][:2]}) == "PASS"
    # an unrelated failure is not blamed on the reproduced timeout
    assert tr.replay_diagnostics(report) == ["action 3 ok -> failed"]


def test_replay_diagnostics_edge_cases():
    assert tr.replay_diagnostics({}) == ["replay report not found: the replay did not run or exited before writing it"]
    # no timeout limit recorded, no network flag, no output -> still one line per cause, nothing invented
    report = {
        "summary": {"actions_recorded": 5, "actions_executed": 2, "actions_skipped_not_executed": 0},
        "actions": [
            {"action_index": 1, "command": "x" * 200, "status": "timed_out", "source_status": "ok"},
            {"action_index": 2, "status": "failed", "source_status": "ok", "outcome_matches_source": False},
        ],
    }
    lines = tr.replay_diagnostics(report)
    assert lines[0] == "replay stopped after 2 of 5 executable actions"
    assert lines[1].startswith("action 1 timed out after the action timeout: xxx") and lines[1].endswith("...")
    assert lines[2].startswith("  cause: not identifiable") and lines[3].startswith("  fix: ")
    assert lines[4] == "action 2 ok -> failed, after timed-out action 1"
    assert tr.replay_diagnostics(_replay_report(mismatch=0)) == []


def test_summarize_replay_fail_on_mismatch(tmp_path):
    prof = _profile_dir(tmp_path, mismatch=1)
    rc = run(
        [
            "summarize-replay",
            "--out-dir",
            str(prof),
            "--summary",
            str(tmp_path / "s.json"),
            "--trace-id",
            "t",
            "--pacing",
            "recorded",
            "--cpus",
            "4",
            "--replay-exit",
            "0",
        ]
    )
    s = json.loads((tmp_path / "s.json").read_text())
    assert rc == 1 and s["fidelity"] == "FAIL"
    assert s["diagnostics"][0] == "action 2 ok -> failed: ModuleNotFoundError: No module named 'attr'"
    # a module missing on the replay side only is an environment difference worth naming
    assert s["diagnostics"][1].startswith("  cause: the replay shell could not find a program, module or path")
    assert len(s["diagnostics"]) == 3


# ---------------------------------------------------------------- verdict


def _write_summary(tmp_path, **overrides):
    summary = {
        "workload": "tasktrace_replay",
        "trace_id": "django__django-11490",
        "tasktrace_exit_status": 0,
        "fidelity": "PASS",
        "replay": {
            "actions_recorded": 114,
            "actions_executed": 113,
            "source_outcome_mismatch_count": 0,
            "actions_timed_out": 0,
        },
        "comparison": {"tool_busy_replay_sec": 24.0},
        "diagnostics": [],
    }
    summary.update(overrides)
    exp = tmp_path / "exp"
    exp.mkdir(parents=True, exist_ok=True)
    (exp / "summary.json").write_text(json.dumps(summary))
    return exp


def test_verdict_replay_pass_and_fail(tmp_path, capsys):
    exp = _write_summary(tmp_path)
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert (
        capsys.readouterr().out
        == "Outcome: PASS  django__django-11490  replay: tool time 24.0s, 113/114 actions, 0 mismatches\n"
    )
    # summary.json path works as well as the directory
    assert run(["verdict", "--oneline", str(exp / "summary.json")]) == 0
    capsys.readouterr()
    exp = _write_summary(
        tmp_path,
        fidelity="FAIL",
        tasktrace_exit_status=1,
        replay={
            "actions_recorded": 114,
            "actions_executed": 113,
            "source_outcome_mismatch_count": 3,
            "actions_timed_out": 1,
        },
        comparison={"replay_to_live_tool_ratio": 1.3127, "tool_busy_replay_sec": 26.3, "tool_busy_live_sec": 20.0},
        diagnostics=[
            "tasktrace exited 1",
            "action 92 timed out after 30s (recorded 5.3s): apt-get update",
            "tool time: replay 26.3s is 31.3% above live 20.0s (limit +5%)",
        ],
    )
    assert run(["verdict", "--oneline", str(exp)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert (
        out[0] == "Outcome: FAIL  django__django-11490  replay: tool time 26.3s, 113/114 actions, "
        "3 mismatches, 1 timed out"
    )
    # a stored "tool time:" reason is not shown: time is reported, not judged
    assert out[1:] == [
        "  tasktrace exited 1",
        "  action 92 timed out after 30s (recorded 5.3s): apt-get update",
    ]
    # a diagnostics field that isn't a list is ignored rather than printed character by character
    exp = _write_summary(tmp_path, fidelity="FAIL", tasktrace_exit_status=1, diagnostics="tasktrace exited 1")
    assert run(["verdict", "--oneline", str(exp)]) == 1
    assert capsys.readouterr().out.splitlines()[1:] == []


def test_verdict_reports_tool_time_without_judging_it(tmp_path, capsys):
    """Capture+replay shows its same-run comparison as information; replay-only shows only its own time."""

    def comparison(ratio, replay_sec):
        return {"replay_to_live_tool_ratio": ratio, "tool_busy_replay_sec": replay_sec, "tool_busy_live_sec": 20.0}

    capture = {
        "workload": "tasktrace_capture_replay",
        "capture": {
            "stage_exit_status": 0,
            "tasktrace_capture_exit_status": 0,
            "reward": 1.0,
            "llm_calls": 3,
            "tool_actions_executed": 2,
        },
        "replay": {
            "ran": True,
            "tasktrace_exit_status": 0,
            "summary": {"actions_recorded": 2, "actions_executed": 2, "source_outcome_mismatch_count": 0},
        },
        "image_identity_check": {"match": True},
    }
    # a capture+replay whose replay was slower than its capture still passes; the ratio is shown
    exp = _write_summary(tmp_path, comparison=comparison(1.5, 30.0), **capture)
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert "replay: tool time 1.500 (live), 2/2 actions, 0 mismatches" in capsys.readouterr().out
    assert run(["verdict", "--no-color", str(exp)]) == 0
    styled = capsys.readouterr().out
    assert "Tool time   : 1.500 replay/live  (replay 30.0s, live 20.0s)\n" in styled
    assert "limit" not in styled and "Why it failed" not in styled
    # no ratio at all (nothing executed carried a live time) is reported as such
    exp = _write_summary(tmp_path, comparison={"replay_to_live_tool_ratio": None, "actions_compared": 0}, **capture)
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert "replay: tool time n/a, 2/2 actions, 0 mismatches" in capsys.readouterr().out
    # replay-only: its own time, never compared
    exp = _write_summary(tmp_path, comparison={"tool_busy_replay_sec": 30.0})
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert "replay: tool time 30.0s, 113/114 actions" in capsys.readouterr().out


def test_verdict_styled_report_pass(tmp_path, capsys):
    """Default output mirrors `llmb-run jobs show`: Key : value rows, a rule under Outcome, no ANSI when piped."""
    exp = _write_summary(tmp_path, pacing="recorded")
    (exp / "slurm-3064753.out").write_text("[launch] ok\n")
    assert run(["verdict", str(exp)]) == 0
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Outcome     : PASS" and set(out[1]) == {"\u2500"}
    assert out[2] == "Task        : django__django-11490"
    assert out[3] == "Job         : 3064753"
    assert out[4] == "Replay      : 113/114 actions  |  0 mismatches  |  0 timed out  |  pacing recorded"
    assert out[5] == "Tool time   : 24.0s replay"
    assert not any(line.startswith("Note") for line in out)
    assert out[-1] == f"Summary     : {exp / 'summary.json'}"
    assert "\033[" not in "\n".join(out)  # not a tty -> plain
    assert "Why it failed" not in out


def test_verdict_styled_report_fail_and_color(tmp_path, capsys, monkeypatch):
    exp = _write_summary(
        tmp_path,
        fidelity="FAIL",
        tasktrace_exit_status=1,
        replay={
            "actions_recorded": 114,
            "actions_executed": 113,
            "source_outcome_mismatch_count": 3,
            "actions_timed_out": 1,
        },
        comparison={"replay_to_live_tool_ratio": 1.3127, "tool_busy_replay_sec": 44.4, "tool_busy_live_sec": 33.8},
        diagnostics=["action 92 timed out after 30s: apt-get update", "  cause: network", "  fix: retry"],
    )
    assert run(["verdict", str(exp)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Outcome     : FAIL"
    assert "Why it failed" in out
    i = out.index("Why it failed")
    assert set(out[i + 1]) == {"\u2500"}
    assert out[i + 2] == "  action 92 timed out after 30s: apt-get update"
    assert out[i + 3] == "    cause: network" and out[i + 4] == "    fix: retry"
    # color when stdout is a tty and NO_COLOR is unset
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setattr("sys.stdout.isatty", lambda: True)
    run(["verdict", str(exp)])
    colored = capsys.readouterr().out
    assert "\033[1;31mFAIL\033[0m" in colored and "\033[33mcause: network\033[0m" in colored
    # ... and not with --no-color or NO_COLOR
    run(["verdict", "--no-color", str(exp)])
    assert "\033[" not in capsys.readouterr().out
    monkeypatch.setenv("NO_COLOR", "1")
    run(["verdict", str(exp)])
    assert "\033[" not in capsys.readouterr().out


def test_verdict_styled_report_no_summary(tmp_path, capsys):
    (tmp_path / "slurm-77.out").write_text("[launch] ERROR: endpoint check failed with HTTP 503: overloaded\n")
    with pytest.raises(SystemExit) as info:
        run(["verdict", str(tmp_path)])
    assert info.value.code == 1
    out = capsys.readouterr().out.splitlines()
    assert (
        out[0] == "Outcome     : FAIL"
        and out[2] == "Task        : unknown (no summary.json)"
        and out[3] == "Job         : 77"
    )
    i = out.index("Why it failed")
    assert out[i + 2] == "  the launch script aborted before the run:"
    assert out[i + 3].startswith("    [launch] ERROR: endpoint check failed with HTTP 503")
    assert out[i + 4].startswith("  cause: the inference endpoint's model backend was overloaded")


def test_verdict_directory_of_runs_prints_a_table(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {})  # no sacct in this test
    """Pointing at a workload's experiments tree lists every run beneath it, one row each."""
    root = tmp_path / "experiments"
    ok = _write_summary(root / "wl_a")
    (ok / "slurm-101.out").write_text("ok\n")
    bad = _write_summary(
        root / "wl_a2",
        fidelity="FAIL",
        tasktrace_exit_status=1,
        replay={
            "actions_recorded": 114,
            "actions_executed": 113,
            "source_outcome_mismatch_count": 3,
            "actions_timed_out": 1,
        },
        comparison={"replay_to_live_tool_ratio": 1.3127},
        diagnostics=["action 92 timed out after 30s: apt-get", "  cause: network slow", "  fix: retry"],
    )
    (bad / "slurm-102.out").write_text("x\n")
    aborted = root / "wl_b" / "1789587241"
    aborted.mkdir(parents=True)
    (aborted / "slurm-103.out").write_text("[launch] ERROR: endpoint check failed with HTTP 503: overloaded\n")
    (root / "wl_c" / "not_a_run").mkdir(parents=True)  # ignored: no summary.json, no slurm log, no config
    cap = _write_summary(
        root / "wl_e",
        workload="tasktrace_capture_replay",
        capture={
            "stage_exit_status": 0,
            "tasktrace_capture_exit_status": 0,
            "reward": 1.0,
            "llm_calls": 34,
            "llm_wait_sec": 193.8,
            "tool_busy_sec": 1.9,
        },
        replay={
            "ran": True,
            "tasktrace_exit_status": 0,
            "summary": {"actions_recorded": 35, "actions_executed": 34, "source_outcome_mismatch_count": 0},
        },
        image_identity_check={"match": True},
        comparison={"replay_to_live_tool_ratio": 0.835},
    )
    (cap / "slurm-105.out").write_text("ok\n")
    pending = root / "wl_d" / "1789599999"
    pending.mkdir(parents=True)
    (pending / "llmb-config_104.yaml").write_text("job_info: {}\n")  # submitted, not started
    running = root / "wl_f" / "1789599998"
    running.mkdir(parents=True)
    (running / "slurm-106.out").write_text("[launch] endpoint check: OK\n")  # started, no summary, no errors
    assert run(["verdict", str(root)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].split() == [
        "Job",
        "ID",
        "Outcome",
        "Mode",
        "LLM",
        "wait",
        "Tool",
        "busy",
        "Replay",
        "Tool",
        "time",
        "LLM",
        "time",
        "Notes",
    ]
    assert set(out[1]) == {"\u2500"}
    body = out[2:8]
    assert any(r.startswith("101") and "PASS" in r and re.search(r"113/114 act, 0 mism\s+-\s+-\s*$", r) for r in body)
    assert any(r.startswith("102") and "FAIL" in r and "3 mism, 1 t/o" in r and "network slow" in r for r in body)
    assert any(r.startswith("103") and "FAIL" in r and "did not run" in r and "overloaded or down" in r for r in body)
    assert any(r.startswith("104") and "PENDING" in r and "not started" in r for r in body)
    assert any(
        r.startswith("105") and "PASS" in r and "capture+replay" in r and "193.8s" in r and "1.9s" in r and "0.835" in r
        for r in body
    )
    # replay-only rows have no capture phase
    assert any(r.startswith("101") and re.search(r"\sreplay\s+-\s+-\s+113/114", r) for r in body)
    assert any(r.startswith("106") and "RUNNING" in r and "in progress" in r for r in body)
    assert out[-1] == "6 runs: 2 PASS, 2 FAIL, 1 RUNNING, 1 PENDING"
    # mode is known even without summary.json, from the llmb-config's model_name
    (pending / "llmb-config_104.yaml").write_text("model_info:\n  model_name: tasktrace_capture_replay\n")
    run(["verdict", str(pending.parent)])
    assert re.search(r"^104\s+PENDING\s+capture\+replay", capsys.readouterr().out, re.M)
    # all-pass tree exits 0; --oneline gives plain rows; several targets are merged
    assert run(["verdict", str(root / "wl_a")]) == 0  # single run dir -> full report, exit 0
    capsys.readouterr()
    assert run(["verdict", "--oneline", str(root / "wl_a"), str(root / "wl_b")]) == 1
    rows = capsys.readouterr().out.splitlines()
    assert len(rows) == 2 and rows[0].startswith("101  PASS") and rows[1].startswith("103  FAIL")
    # an empty directory: message on stderr, exit 1
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(["verdict", str(empty)]) == 1
    assert "no runs found" in capsys.readouterr().err


def test_verdict_pass_fidelity_but_tasktrace_failed_is_fail(tmp_path, capsys):
    exp = _write_summary(tmp_path, tasktrace_exit_status=1, diagnostics=["tasktrace exited 1"])
    assert run(["verdict", "--oneline", str(exp)]) == 1
    assert capsys.readouterr().out.startswith("Outcome: FAIL  ")


def test_verdict_derives_reasons_for_summaries_without_diagnostics(tmp_path, capsys):
    exp = _write_summary(tmp_path, fidelity="FAIL", tasktrace_exit_status=1)
    json_path = exp / "summary.json"
    s = json.loads(json_path.read_text())
    del s["diagnostics"]
    json_path.write_text(json.dumps(s))
    (exp / "profile" / "replay").mkdir(parents=True)
    (exp / "profile" / "replay" / "replay_report.json").write_text(json.dumps(_timeout_cascade_report()))
    assert run(["verdict", "--oneline", str(exp)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("Outcome: FAIL  ") and out[1].startswith("  action 92 timed out after 30s")


def test_launch_errors_catches_untagged_fatal_lines(tmp_path):
    (tmp_path / "slurm-1.out").write_text(
        "[launch] node: x\n"
        "allocation exposes 1 CPUs, need 4\n"
        "some noise\n"
        "srun: error: task 0: Exited with exit code 1\n"
        "slurmstepd: error: *** JOB 1 ON n1 CANCELLED AT 2026-09-17T01:00:00 DUE TO TIME LIMIT ***\n"
    )
    errors = tr.launch_errors(tmp_path)
    assert errors[0] == "allocation exposes 1 CPUs, need 4"
    assert any("DUE TO TIME LIMIT" in e for e in errors) and "some noise" not in errors
    assert tr.launch_cause(errors).startswith("cause: the Slurm allocation has fewer CPUs")


def test_unfinished_runs_use_slurm_state(tmp_path, monkeypatch):
    """A finished job with no summary and no recognised error is a FAIL when sacct says it ended."""
    run_dir = tmp_path / "1789000000"
    run_dir.mkdir()
    (run_dir / "slurm-772341.out").write_text("[launch] node: x\nsomething unexpected happened\n")
    # sacct unavailable -> the log-only heuristic: started, so RUNNING
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {})
    assert tr.run_row(run_dir, {})[1] == "RUNNING"
    # sacct says RUNNING -> RUNNING; says FAILED -> FAIL with the last log line as the reason
    assert tr.run_row(run_dir, {"772341": "RUNNING"})[1] == "RUNNING"
    row = tr.run_row(run_dir, {"772341": "FAILED"})
    assert row[1] == "FAIL" and "ended (FAILED) without writing summary.json" in row[-1]
    assert "something unexpected happened" in row[-1]
    # the single-run report follows the same rule: exit 1 for the ended job, 2 while running
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"772341": "COMPLETED"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(run_dir)])
    assert info.value.code == 1
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"772341": "RUNNING"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(run_dir)])
    assert info.value.code == 2


def test_slurm_states_parses_sacct_and_degrades(monkeypatch):
    class R:
        stdout = "772341|FAILED\n772346|CANCELLED by 1001\n"

    monkeypatch.setattr(tr.shutil, "which", lambda name: "/usr/bin/sacct")
    monkeypatch.setattr(tr.subprocess, "run", lambda *a, **k: R())
    assert tr.slurm_states(["772341", "772346", "notajob"]) == {"772341": "FAILED", "772346": "CANCELLED"}
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    assert tr.slurm_states(["772341"]) == {}


def test_verdict_rejects_a_file_that_is_not_summary_json(tmp_path, capsys):
    stray = tmp_path / "notes.txt"
    stray.write_text("x")
    with pytest.raises(SystemExit) as info:
        run(["verdict", str(stray)])
    assert "neither an experiment directory nor a summary.json" in str(info.value.code)


def test_verdict_missing_summary(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {})  # no sacct in this test
    (tmp_path / "llmb-config_42.yaml").write_text("job_info: {}\n")  # submitted, no Slurm log yet
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(tmp_path)])
    assert info.value.code == 2
    out = capsys.readouterr().out.splitlines()
    assert out[0].startswith("Outcome: PENDING  no summary.json at") and out[1].startswith("  no Slurm log yet")
    # started, no summary, no launcher error -> RUNNING, exit 2, styled 'Status' section
    (tmp_path / "slurm-42.out").write_text("[launch] endpoint check: OK\n")
    with pytest.raises(SystemExit) as info:
        run(["verdict", str(tmp_path)])
    assert info.value.code == 2
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Outcome     : RUNNING" and "Status" in out and "Why it failed" not in out


@pytest.mark.parametrize(
    "log,needle",
    [
        (
            "[launch] ERROR: endpoint check failed with HTTP 503: the endpoint's model backend is overloaded or down; "
            'retry later. Body: {"error":{"message":"Service temporarily overloaded"}}\n',
            "overloaded or down; not a cluster or recipe problem",
        ),
        (
            "[launch] ERROR: endpoint check connected to https://x/v1 but got no response within 90s: timed out.\n",
            "accepted the connection but never answered",
        ),
        ("[launch] ERROR: endpoint check could not connect to https://x/v1: no route.\n", "no route to the endpoint"),
        ("[launch] ERROR: endpoint check failed with HTTP 401: the key was rejected.\n", "API key was rejected"),
        (
            "[launch] ERROR: endpoint check failed with HTTP 404: model id 'm' is listed by GET https://x/v1/models "
            "but its backend deployment is gone (the endpoint's function id no longer exists).\n",
            "still in the endpoint's catalog but its backend deployment was removed",
        ),
        ("[launch] ERROR: allocation exposes 2 CPUs, need 4\n", "fewer CPUs"),
    ],
)
def test_verdict_quotes_launch_errors_and_names_the_cause(tmp_path, capsys, log, needle):
    (tmp_path / "slurm-42.out").write_text(
        "[launch] node: x\n"
        + log
        + "[launch] ERROR: aborting before the capture; set RUN_CONF_SKIP_ENDPOINT_CHECK=1.\n"
    )
    with pytest.raises(SystemExit):
        run(["verdict", "--oneline", str(tmp_path)])
    out = capsys.readouterr().out.splitlines()
    assert out[1] == "  the launch script aborted before the run:"
    assert out[2].startswith("    [launch] ERROR:") and "aborting" not in out[2]  # the informative line comes first
    assert out[3].startswith("    [launch] ERROR: aborting")
    assert out[4].startswith("  cause: ") and needle in out[4]


# ---------------------------------------------------------------- live outputs / environment mismatch


def _atif(tmp_path, steps):
    """A harbor trajectory.json like the capture writes: tool-call steps carry the live tool output."""
    agent = tmp_path / "capture" / "runs" / "r" / "trial" / "agent"
    agent.mkdir(parents=True)
    atif_steps = [{"step_id": 1, "source": "user", "message": "task"}]
    for i, (command, rc, output) in enumerate(steps, start=2):
        atif_steps.append(
            {
                "step_id": i,
                "source": "agent",
                "tool_calls": [{"function_name": "bash", "arguments": {"command": command}}],
                "observation": {"results": [{"content": json.dumps({"returncode": rc, "output": output})}]},
            }
        )
    (agent / "trajectory.json").write_text(json.dumps({"schema_version": "ATIF-v1.5", "steps": atif_steps}))
    return tmp_path / "capture"


def test_live_tool_outputs_pairs_by_position_and_verifies_command(tmp_path):
    live_err = "/opt/miniconda3/bin/python: No module named pytest\n"
    cap = _atif(tmp_path, [("ls", 0, "a\n"), ("python -m pytest t.py", 1, live_err)])
    outputs = tr.live_tool_outputs(cap)
    assert outputs == [("ls", "a\n"), ("python -m pytest t.py", live_err)]
    actions = [
        {"action_index": 1, "command": "ls"},
        {"action_index": 2, "command": "python -m pytest t.py"},
        {"action_index": 3, "command": "echo extra"},
    ]
    assert tr._pair_live_outputs(actions, outputs) == {1: "a\n", 2: live_err}
    # a command mismatch at some position drops only that pairing
    actions[0]["command"] = "ls -la"
    assert tr._pair_live_outputs(actions, outputs) == {2: live_err}
    assert tr.live_tool_outputs("") == [] and tr.live_tool_outputs(tmp_path / "nowhere") == []


def _pytest_mismatch(idx, command, recorded=0.02):
    return {
        "action_index": idx,
        "command": command,
        "status": "ok",
        "source_status": "failed",
        "returncode": 0,
        "source_wall_time_sec": recorded,
        "wall_time_sec": 0.19,
        "stdout_preview": "===== test session starts =====\ncollected 1 item\n\n===== 1 skipped in 0.01s =====\n",
        "stderr_preview": "",
        "outcome_matches_source": False,
    }


def test_replay_diagnostics_names_environment_mismatch_from_live_output():
    """A capture+replay run whose live python had no pytest while the replay's did, so four runs flipped failed->ok."""
    pytest_cmd = "cd /testbed && python -m pytest test_repro.py -v"
    live_err = "/opt/miniconda3/bin/python: No module named pytest\n"
    live_tb = 'Traceback (most recent call last):\n  File x\nTypeError: required field "lineno" missing from alias\n'
    actions = [
        {"action_index": 6, "command": "pwd", "status": "ok", "source_status": "ok", "outcome_matches_source": True},
        _pytest_mismatch(7, pytest_cmd),
        {"action_index": 8, "command": "pip install -e .", "status": "ok", "source_status": "ok"},
        _pytest_mismatch(9, pytest_cmd, recorded=0.23),
    ]
    live = [("pwd", "/testbed\n"), (pytest_cmd, live_err), ("pip install -e .", "ok"), (pytest_cmd, live_tb)]
    report = {
        "summary": {"actions_recorded": 4, "actions_executed": 4, "actions_skipped_not_executed": 0},
        "actions": actions,
    }
    lines = tr.replay_diagnostics(report, live)
    assert lines[0] == (
        "action 7 failed -> ok: replay said: 1 skipped in 0.01s | "
        "live said: /opt/miniconda3/bin/python: No module named pytest"
    )
    assert lines[1].startswith("  cause: the live shell could not find a program, module or path the other shell had")
    assert "/root mount hid the image's .bashrc" in lines[1]
    assert lines[2].startswith("  fix: capture with tasktrace-capture that preserves the image's /root dotfiles")
    assert lines[3] == (
        "action 9 failed -> ok: replay said: 1 skipped in 0.01s | "
        'live said: TypeError: required field "lineno" missing from alias'
    )
    assert lines[4].startswith("  cause: likely the same environment difference as action 7")
    assert len(lines) == 5
    # without live outputs, a sub-0.1s live failure that the replay ran normally is still called out, less certainly
    lines = tr.replay_diagnostics(report)
    assert lines[0] == "action 7 failed -> ok: 1 skipped in 0.01s"
    assert lines[1].startswith("  cause: probably an environment difference; the live command failed within 20 ms")
    # and a 0.23s live failure with no live output gets no invented cause
    assert lines[3] == "action 9 failed -> ok: 1 skipped in 0.01s" and len(lines) == 4


def test_verdict_capture_line(tmp_path, capsys):
    base = {
        "workload": "tasktrace_capture_replay",
        "trace_id": "django__django-11490",
        "fidelity": "PASS",
        "capture": {
            "stage_exit_status": 0,
            "tasktrace_capture_exit_status": 0,
            "reward": 0.0,
            "llm_calls": 57,
            "tool_actions_executed": 113,
        },
        "replay": {
            "ran": True,
            "tasktrace_exit_status": 0,
            "summary": {"actions_recorded": 114, "actions_executed": 113, "source_outcome_mismatch_count": 0},
        },
        "comparison": {"replay_to_live_tool_ratio": 0.98},
        "image_identity_check": {"match": True},
        "diagnostics": [],
    }
    exp = _write_summary(tmp_path, **base)
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert capsys.readouterr().out == (
        "Outcome: PASS  django__django-11490  capture: reward 0.0, 57 llm calls, 113 tool actions"
        "  replay: tool time 0.980 (live), 113/114 actions, 0 mismatches\n"
    )
    # the live run produced nothing replayable
    broken = dict(base, fidelity="FAIL", capture=dict(base["capture"], stage_exit_status=1), replay={"ran": False})
    broken["diagnostics"] = ["capture: stage exit 1; the live run left no replayable trajectory", "replay did not run"]
    exp = _write_summary(tmp_path, **broken)
    assert run(["verdict", "--oneline", str(exp)]) == 1
    out = capsys.readouterr().out.splitlines()
    assert out[0] == "Outcome: FAIL  django__django-11490  capture: stage exit 1  replay: did not run"
    assert out[1] == "  capture: stage exit 1; the live run left no replayable trajectory"
    # image mismatch alone flips the verdict even though fidelity is PASS
    exp = _write_summary(tmp_path, **dict(base, image_identity_check={"match": False}))
    assert run(["verdict", "--oneline", str(exp)]) == 1
    assert capsys.readouterr().out.startswith("Outcome: FAIL  ")
    # the capture tool exiting non-zero (e.g. killed by a signal after finalizing the trajectory) is a FAIL
    # even though the stage step and the replay went through
    capture_died = dict(base, capture=dict(base["capture"], tasktrace_capture_exit_status=143))
    capture_died["diagnostics"] = ["capture: tasktrace-capture exited 143"]
    exp = _write_summary(tmp_path, **capture_died)
    assert run(["verdict", "--oneline", str(exp)]) == 1
    assert capsys.readouterr().out.splitlines()[1] == "  capture: tasktrace-capture exited 143"


def _provenance(tmp_path, sha="imgsha"):
    prov = tmp_path / "staged" / "provenance.json"
    prov.parent.mkdir(parents=True)
    prov.write_text(
        json.dumps(
            {
                "selector": "sel",
                "trace": {"task": "t", "sha256": "tsha", "schema_version": "v1", "source_trial": "st"},
                "live_run": {
                    "reward": 0.0,
                    "tasktrace_capture_exit_status": 0,
                    "llm_calls": 2,
                    "tool_busy_sec": 4.5,
                    "capture_report": {"timing": {"wall": 9}},
                },
                "task_image": {"sha256": sha},
                "problems": [],
            }
        )
    )
    return prov


def test_summarize_capture_ok_and_identity(tmp_path):
    prov = _provenance(tmp_path)
    prof = _profile_dir(tmp_path)
    (tmp_path / "scan.json").write_text(json.dumps({"files_redacted": []}))
    base = [
        "summarize-capture",
        "--provenance",
        str(prov),
        "--profile-dir",
        str(prof),
        "--summary",
        str(tmp_path / "s.json"),
        "--scrub-report",
        str(tmp_path / "scan.json"),
        "--pacing",
        "recorded",
        "--cpus",
        "4",
        "--stage-exit",
        "0",
    ]
    assert run(base + ["--replay-exit", "0"]) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert (
        s["fidelity"] == "PASS"
        and s["image_identity_check"]["match"] is True
        and s["comparison"]["actions_compared"] == 2
    )
    assert s["capture"]["llm_calls"] == 2 and s["capture"]["timing"] == {"wall": 9} and s["replay"]["ran"] is True
    # replay never ran -> FAIL, exit 1, replay.ran false
    assert run(base + ["--replay-exit", ""]) == 1
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["fidelity"] == "FAIL" and s["replay"]["ran"] is False and s["replay"]["tasktrace_exit_status"] is None
    assert s["diagnostics"] == ["replay did not run"]
    # image mismatch -> exit 1 even with PASS fidelity
    prov2 = _provenance(tmp_path / "other", sha="different")
    assert run(base[:2] + [str(prov2)] + base[3:] + ["--replay-exit", "0"]) == 1
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["fidelity"] == "PASS" and s["diagnostics"] == [
        "image identity mismatch: capture different vs replay imgsha"
    ]


# ---------------------------------------------------------------- trace manifests / resolve-trace


@pytest.mark.parametrize(
    "ref,expected",
    [
        (
            "ghcr.io#epoch-research/swe-bench.eval.arm64.django__django-11490:latest",
            "epoch-research+swe-bench.eval.arm64.django__django-11490+latest.sqsh",
        ),
        (
            "ghcr.io/epoch-research/swe-bench.eval.arm64.django__django-11490:latest",
            "epoch-research+swe-bench.eval.arm64.django__django-11490+latest.sqsh",
        ),
        ("nvcr.io/nvidia/nemo:25.02.01", "nvidia+nemo+25.02.01.sqsh"),
        ("nvcr.io/nvidia/nemo", "nvidia+nemo+latest.sqsh"),
        ("nvcr.io#nvidia/nemo:sha256:abcdef0123456789ffff", "nvidia+nemo+sha256-abcdef012345.sqsh"),
    ],
)
def test_image_file_name_matches_llmb_install(ref, expected):
    assert tr.image_file_name(ref) == expected


def _manifest(root, selector="sel-a", model="m", harness="h", dataset="d", task="t", **extra):
    d = root / model / harness / dataset / task
    d.mkdir(parents=True, exist_ok=True)
    data = {
        "schema_version": 1,
        "selector": selector,
        "model": model,
        "agent_harness": harness,
        "dataset": dataset,
        "task": task,
        "agent": "mini-swe-agent",
        "live_model_default": "openai/x",
        "image": {"reference": "ghcr.io/org/img:tag", "arch": "aarch64"},
    }
    data.update(extra)
    (d / tr.TRACE_MANIFEST).write_text(json.dumps(data))
    return d


def test_resolve_trace_prints_shell_assignments(tmp_path, capsys):
    root = tmp_path / "trajectories"
    d = _manifest(root)
    _manifest(root, selector="sel-b", task="t2")
    assert run(["resolve-trace", "--root", str(root), "--selector", "sel-a"]) == 0
    out = capsys.readouterr().out
    vals = dict(line.split("=", 1) for line in out.strip().splitlines())
    assert vals["TRACE_MODEL"] == "m" and vals["TRACE_ID"] == "t" and vals["AGENT"] == "mini-swe-agent"
    assert vals["TRACE_DIR"] == str(d.parent) and vals["TRACE"] == str(d / tr.TRAJECTORY_NAME)
    assert (
        vals["IMAGE_FILE"] == "org+img+tag.sqsh"
        and vals["IMAGE_ARCH"] == "aarch64"
        and vals["LIVE_MODEL_DEFAULT"] == "openai/x"
    )
    # values are shell-quoted when needed
    _manifest(root, selector="sel-c", task="t3", live_model_default="weird value$")
    run(["resolve-trace", "--root", str(root), "--selector", "sel-c"])
    assert "LIVE_MODEL_DEFAULT='weird value$'" in capsys.readouterr().out


def test_resolve_trace_unknown_and_inconsistent(tmp_path, capsys):
    root = tmp_path / "trajectories"
    _manifest(root)
    assert run(["resolve-trace", "--root", str(root), "--selector", "nope"]) == 1
    assert "available: sel-a" in capsys.readouterr().err
    # manifest whose coordinates disagree with its directory
    d = _manifest(root, selector="sel-x", task="t9")
    m = json.loads((d / tr.TRACE_MANIFEST).read_text())
    m["task"] = "elsewhere"
    (d / tr.TRACE_MANIFEST).write_text(json.dumps(m))
    assert run(["resolve-trace", "--root", str(root), "--selector", "sel-x"]) == 1
    assert "but lives in" in capsys.readouterr().err
    # missing image
    d = _manifest(root, selector="sel-y", task="t8")
    m = json.loads((d / tr.TRACE_MANIFEST).read_text())
    del m["image"]
    (d / tr.TRACE_MANIFEST).write_text(json.dumps(m))
    assert run(["resolve-trace", "--root", str(root), "--selector", "sel-y"]) == 1
    assert "image.reference" in capsys.readouterr().err


def test_stage_writes_trace_manifest(tmp_path, monkeypatch):
    trace = _capture_layout(tmp_path)
    _stage_env(monkeypatch, tmp_path)
    staged = tmp_path / "staged"
    assert (
        run(
            [
                "stage",
                "--trace",
                str(trace),
                "--staged-dir",
                str(staged),
                "--provenance",
                str(staged / "provenance.json"),
            ]
        )
        == 0
    )
    m = json.loads((staged / tr.TRACE_MANIFEST).read_text())
    assert m == {
        "schema_version": 1,
        "kind": "tasktrace-trace",
        "selector": "sel",
        "model": "m",
        "agent_harness": "h",
        "dataset": "d",
        "task": "t",
        "agent": "mini-swe-agent",
        "live_model_default": "openai/x",
        "image": {"reference": "ghcr.io/x:latest", "arch": "aarch64"},
    }


def test_shipped_manifest_resolves(capsys):
    """The shipped trajectory must resolve with the values the recipes were validated with, and carry no
    measured time."""
    root = pathlib.Path(__file__).resolve().parents[4] / "trajectories"
    assert run(["resolve-trace", "--root", str(root), "--selector", "nemotron-3-ultra_miniswe_django-11490"]) == 0
    vals = dict(line.split("=", 1) for line in capsys.readouterr().out.strip().splitlines())
    assert vals["IMAGE_FILE"] == "epoch-research+swe-bench.eval.arm64.django__django-11490+sha256-58f9edb89a62.sqsh"
    assert vals["TRACE_MODEL"] == "nemotron-3-ultra-nvfp4" and vals["IMAGE_ARCH"] == "aarch64"
    assert pathlib.Path(vals["TRACE"]).is_file()
    assert tr.trajectory_timed(pathlib.Path(vals["TRACE"])) is False


# ---------------------------------------------------------------- review follow-ups


def test_read_json_returns_dict_only(tmp_path):
    for body in ("[]", "[1, 2]", '"x"', "3", "", "{not json"):
        (tmp_path / "f.json").write_text(body)
        assert tr.read_json(tmp_path / "f.json") == {}
    (tmp_path / "f.json").write_text('{"a": 1}')
    assert tr.read_json(tmp_path / "f.json") == {"a": 1}
    assert not (tmp_path / "f.json.tmp").exists()


def test_write_json_is_atomic(tmp_path):
    tr.write_json(tmp_path / "s.json", {"k": 1})
    assert tr.read_json(tmp_path / "s.json") == {"k": 1}
    assert sorted(p.name for p in tmp_path.iterdir()) == ["s.json"]


def test_running_job_with_error_lines_is_not_a_verdict(tmp_path, capsys, monkeypatch):
    """A '[launch] ERROR' the capture prints and recovers from must not fail a job Slurm says is running."""
    (tmp_path / "slurm-777.out").write_text(
        "[launch] node: x\n[launch] ERROR: capture produced no finalized trajectory\n"
    )
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"777": "RUNNING"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(tmp_path)])
    assert info.value.code == tr.EXIT_NOT_FINISHED
    out = capsys.readouterr().out
    assert out.startswith("Outcome: RUNNING") and "not yet a verdict" in out and "capture produced" in out
    # table mode agrees
    row = tr.run_row(tmp_path, {"777": "RUNNING"})
    assert row[1] == "RUNNING" and row[5] == "in progress"
    # once Slurm says it ended, the same lines become the verdict
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"777": "FAILED"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(tmp_path)])
    assert info.value.code == 1
    assert "aborted before the run" in capsys.readouterr().out
    assert tr.run_row(tmp_path, {"777": "FAILED"})[1] == "FAIL"


def test_time_limit_kill_is_not_called_an_early_abort(tmp_path, capsys, monkeypatch):
    (tmp_path / "slurm-778.out").write_text(
        "[launch] replay: starting\nslurmstepd: error: *** JOB 778 ON n1 CANCELLED AT 2026-09-17T01:00:00 DUE TO TIME LIMIT ***\n"
    )
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"778": "TIMEOUT"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(tmp_path)])
    assert info.value.code == 1
    out = capsys.readouterr().out
    assert "Outcome: FAIL" in out and "Slurm ended the job" in out and "aborted before" not in out


def test_incomplete_summary_json(tmp_path, capsys, monkeypatch):
    (tmp_path / "slurm-779.out").write_text("[launch] node: x\n")
    (tmp_path / "summary.json").write_text('{"fidelity": "PA')  # being written
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"779": "RUNNING"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(tmp_path)])
    assert info.value.code == tr.EXIT_NOT_FINISHED and "RUNNING" in capsys.readouterr().out
    assert tr.run_row(tmp_path, {"779": "RUNNING"})[1] == "RUNNING"
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {"779": "COMPLETED"})
    with pytest.raises(SystemExit) as info:
        run(["verdict", "--oneline", str(tmp_path)])
    assert info.value.code == 1 and "not valid JSON" in capsys.readouterr().out
    (tmp_path / "summary.json").write_text("[1, 2]")  # valid JSON, wrong shape: no traceback
    assert tr.run_row(tmp_path, {"779": "COMPLETED"})[1] == "FAIL"


def test_fatal_line_re_ignores_error_inside_identifiers():
    assert not tr.FATAL_LINE_RE.search("CUDA_ERROR_NONE FOO_ERROR_COUNT=0 no errors here")
    assert tr.FATAL_LINE_RE.search("[launch] ERROR: x") and tr.FATAL_LINE_RE.search("ERROR: boom")
    assert tr.FATAL_LINE_RE.search("allocation exposes 1 CPUs, need 4")


def test_diagnostics_tolerate_odd_report_values():
    report = {
        "summary": {"actions_recorded": "114", "actions_executed": 5, "actions_skipped_not_executed": None},
        "action_timeout_sec": "30",
        "actions": [
            {"action_index": 1, "status": "timed_out", "source_wall_time_sec": "12", "command": "make"},
            {
                "action_index": 2,
                "status": "ok",
                "outcome_matches_source": False,
                "source_status": "failed",
                "source_wall_time_sec": "abc",
                "command": "ls",
            },
        ],
    }
    lines = tr.replay_diagnostics(report)
    assert lines[0] == "replay stopped after 5 of 114 executable actions"
    assert any("timed out after 30s (recorded 12.0s)" in ln for ln in lines)
    out = tr.render_verdict([("Outcome", "FAIL")], ["x", 3, None], tr._Style(False))
    assert "3" in out and "None" in out


def test_interactive_re_matches_commands_not_arguments():
    assert not tr.INTERACTIVE_RE.search("grep -r 'vi' src/")
    assert not tr.INTERACTIVE_RE.search("sed -i 's/less/more/' f")
    assert tr.INTERACTIVE_RE.search("vim setup.py") and tr.INTERACTIVE_RE.search("cat f | less")
    assert tr.INTERACTIVE_RE.search("python -m pytest --pdb")


def test_connect_timeout_line_names_egress_not_silent_backend():
    line = "[launch] ERROR: endpoint check could not connect to https://x/v1: <urlopen error timed out>. needs egress"
    assert "no route" in tr.launch_cause([line])
    assert "never answered" in tr.launch_cause(
        ["[launch] ERROR: endpoint check connected to x but got no response within 90s"]
    )


def test_open_states_include_hold_and_stageout(tmp_path):
    (tmp_path / "slurm-780.out").write_text("[launch] node: x\n")
    for state in ("REQUEUE_HOLD", "STAGE_OUT", "SIGNALING"):
        assert tr.unfinished_status(tmp_path, "780", state)[0] == "RUNNING"


def test_missing_sacct_is_flagged(tmp_path, monkeypatch):
    (tmp_path / "slurm-781.out").write_text("[launch] node: x\n")
    monkeypatch.setattr(tr.shutil, "which", lambda name: None)
    outcome, reason = tr.unfinished_status(tmp_path, "781", None)
    assert outcome == "RUNNING" and "sacct unavailable" in reason


# ---------------------------------------------------------------- gpu recipes: cpuset split, image name, server info


def test_cpuset_skip_and_all_remaining(monkeypatch, capsys):
    assert run(["cpuset", "--count", "2", "--skip", "4"], monkeypatch, "0-7\n") == 0
    assert capsys.readouterr().out.strip() == "4,5"
    assert run(["cpuset", "--skip", "4", "--all-remaining", "--min", "2"], monkeypatch, "0-7\n") == 0
    assert capsys.readouterr().out.strip() == "4,5,6,7"
    with pytest.raises(SystemExit) as info:
        run(["cpuset", "--skip", "4", "--all-remaining", "--min", "8"], monkeypatch, "0-7\n")
    assert "4 are left for the server, need at least 8" in str(info.value.code)
    with pytest.raises(SystemExit) as info:
        run(["cpuset", "--skip", "4", "--count", "6"], monkeypatch, "0-7\n")
    assert "exposes 8 CPUs, need 10" in str(info.value.code)
    with pytest.raises(SystemExit):
        run(["cpuset"], monkeypatch, "0-7\n")
    with pytest.raises(SystemExit) as info:
        run(["cpuset", "--count", "2", "--skip", "-1"], monkeypatch, "0-7\n")
    assert "--skip must be 0 or more" in str(info.value.code)
    with pytest.raises(SystemExit) as info:
        run(["cpuset", "--all-remaining", "--min", "0"], monkeypatch, "0-7\n")
    assert "--min must be at least 1" in str(info.value.code)


def test_image_file_name_command(capsys):
    ref = (
        "nvcr.io#nvidia/ai-dynamo/vllm-runtime:sha256:9fefbd10b63c52030326e22ed0eb9d87082fee53709fe327b5b2773e267792bc"
    )
    assert run(["image-file-name", "--image-ref", ref]) == 0
    assert capsys.readouterr().out.strip() == "ai-dynamo+vllm-runtime+sha256-9fefbd10b63c.sqsh"


def _server_env(monkeypatch, **overrides):
    vals = {
        "TT_SERVER_CONFIG": "nemotron-3-ultra-nvfp4",
        "TT_SERVER_ENGINE": "vllm",
        "TT_SERVER_VERSION": "0.11.0",
        "TT_SERVER_IMAGE_REF": "nvcr.io#nvidia/ai-dynamo/vllm-runtime:sha256:abc",
        "TT_SERVER_IMAGE_PATH": "/img/vllm.sqsh",
        "TT_SERVER_MODEL_REPO": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "TT_SERVER_MODEL_REVISION": "1839",
        "TT_SERVER_MODEL_DIR": "/datasets/model",
        "TT_SERVER_SERVED_MODEL": "nemotron-3-ultra-nvfp4",
        "TT_SERVER_TP": "4",
        "TT_SERVER_MAX_MODEL_LEN": "262144",
        "TT_SERVER_GPU_MEM_UTIL": "0.9",
        "TT_SERVER_EXTRA_ARGS": '--enable-prefix-caching\n--override-generation-config\n{"temperature": 1.0}\n',
        "TT_SERVER_ENV": "HF_HUB_OFFLINE=1\n",
        "TT_SERVER_API_BASE": "http://127.0.0.1:8000/v1",
        "TT_SERVER_PORT": "8000",
        "TT_SERVER_GPU_COUNT": "4",
        "TT_SERVER_GPU_NAME": "NVIDIA GB300",
        "TT_SERVER_CPUSET": "4-31",
        "TT_SERVER_STARTUP_SEC": "412",
        "TT_SERVER_STARTED_AT": "2026-09-21T10:00:00Z",
        "TT_SERVER_LOG": "/exp/server/vllm.log",
        "TT_SERVER_HOME": "/wl/server-home",
        "TT_SERVER_NODE": "n1",
    }
    vals.update(overrides)
    for k, v in vals.items():
        monkeypatch.setenv(k, v)


def test_server_info_from_env(tmp_path, monkeypatch, capsys):
    _server_env(monkeypatch)
    out = tmp_path / "server.json"
    assert run(["server-info", "--out", str(out)]) == 0
    info = json.loads(out.read_text())
    assert info["engine"] == "vllm" and info["version"] == "0.11.0" and info["served_model"] == "nemotron-3-ultra-nvfp4"
    assert info["tp"] == 4 and info["max_model_len"] == 262144 and info["gpu_memory_utilization"] == 0.9
    assert info["extra_args"] == ["--enable-prefix-caching", "--override-generation-config", '{"temperature": 1.0}']
    assert info["env"] == ["HF_HUB_OFFLINE=1"] and info["gpu_count"] == 4 and info["startup_sec"] == 412
    assert info["model"] == {
        "repo": "nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4",
        "revision": "1839",
        "path": "/datasets/model",
    }
    assert "server.json" in capsys.readouterr().out
    # unset or empty inputs become null rather than crashing the launch script
    for k in ("TT_SERVER_VERSION", "TT_SERVER_STARTUP_SEC", "TT_SERVER_GPU_MEM_UTIL"):
        monkeypatch.setenv(k, "")
    run(["server-info", "--out", str(out)])
    info = json.loads(out.read_text())
    assert info["version"] is None and info["startup_sec"] is None and info["gpu_memory_utilization"] is None


def test_stage_records_the_live_server(tmp_path, monkeypatch, capsys):
    trace = _capture_layout(tmp_path)
    _stage_env(monkeypatch, tmp_path)
    server = tmp_path / "server.json"
    server.write_text(
        json.dumps({"engine": "vllm", "served_model": "nemotron-3-ultra-nvfp4", "tp": 4, "startup_sec": 300})
    )
    monkeypatch.setenv("TT_LLM_MODE", "live-server")
    monkeypatch.setenv("TT_SERVER_INFO", str(server))
    monkeypatch.setenv("TT_SERVER_CONFIG", "nemotron-3-ultra-nvfp4")
    monkeypatch.setenv("TT_LIVE_MODEL_DEFAULT", "openai/nvidia/public-id")
    monkeypatch.setenv("TT_API_KEY_VAR", "")
    staged = tmp_path / "staged"
    rc = run(["stage", "--trace", str(trace), "--staged-dir", str(staged), "--provenance", str(staged / "p.json")])
    assert rc == 0
    live = json.loads((staged / "p.json").read_text())["live_run"]
    assert live["llm_mode"] == "live-server" and live["server"]["tp"] == 4 and live["api_key_env_var"] is None
    assert live["server"]["startup_sec"] == 300 and live["capture_report"]["timing"] == {"wall": 9}
    m = json.loads((staged / tr.TRACE_MANIFEST).read_text())
    assert m["llm_mode"] == "live-server" and m["serving_config"] == "nemotron-3-ultra-nvfp4"
    assert m["live_model_default"] == "openai/nvidia/public-id"  # the reference id, not the local server's
    # resolve-trace hands both new fields to the launch script
    root = tmp_path / "trajectories"
    d = root / "m" / "h" / "d" / "t"
    d.mkdir(parents=True)
    shutil.copy(staged / tr.TRACE_MANIFEST, d / tr.TRACE_MANIFEST)
    capsys.readouterr()
    assert run(["resolve-trace", "--root", str(root), "--selector", "sel"]) == 0
    vals = dict(line.split("=", 1) for line in capsys.readouterr().out.strip().splitlines())
    assert vals["SERVING_CONFIG"] == "nemotron-3-ultra-nvfp4" and vals["LLM_MODE"] == "live-server"


# ---------------------------------------------------------------- gpu recipes: llm replay


def _llm_trace(tmp_path, calls, actions=()):
    rows = [{"record_type": "trace", "schema_version": "tasktrace-capture-cpu-trace.v1", "replay_supported": True}]
    for i, (start, wall, prompt, cached, completion) in enumerate(calls, 1):
        rows.append(
            {
                "record_type": "llm",
                "llm_index": i,
                "started_at_epoch": start,
                "finished_at_epoch": start + wall,
                "wall_time_sec": wall,
                "prompt_tokens": prompt,
                "cached_prompt_tokens": cached,
                "completion_tokens": completion,
            }
        )
    for j, start in enumerate(actions, 1):
        rows.append(
            {
                "record_type": "action",
                "action_index": j,
                "command": "ls",
                "started_at_epoch": start,
                "finished_at_epoch": start + 0.5,
                "wall_time_sec": 0.5,
            }
        )
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "trajectory.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


class _Stream:
    """A fake urlopen response: the SSE lines of a streamed completion, each one advancing the clock."""

    def __init__(self, lines, tick):
        self._lines, self._tick = lines, tick

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __iter__(self):
        for line in self._lines:
            self._tick()
            yield line


def _sse(prompt_tokens, completion_tokens, cached=0):
    usage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    return [
        b": keep-alive\n",
        b"\n",
        b'data: {"choices": [{"text": "", "index": 0}]}\n',  # an empty first chunk is not the first token
        b'data: {"choices": [{"text": "tok", "index": 0}]}\n',
        b"data: not json\n",
        b'data: {"choices": [{"text": "tok", "index": 0}]}\n',
        ("data: " + json.dumps({"choices": [], "usage": usage}) + "\n").encode(),
        b"data: [DONE]\n",
    ]


def _llm_args(trace, out, pacing="recorded", extra=()):
    argv = ["llm-replay", "--trace", str(trace), "--api-base", "http://127.0.0.1:8000/v1", "--model", "m"]
    argv += ["--pacing", pacing, "--output", str(out), "--timeout-sec", "7", *extra]
    return tr.build_parser().parse_args(argv)


def test_llm_replay_recorded_pacing_shapes_and_report(tmp_path, capsys):
    calls = [(100.0, 2.0, 3000, 0, 40), (110.0, 3.0, 3200, 3000, 60), (120.0, 1.0, 0, 0, 10)]
    trace = _llm_trace(tmp_path, calls, actions=[102.5])
    requests, slept, now = [], [], [0.0]

    def clock():
        return now[0]

    def sleep(sec):
        slept.append(round(sec, 3))
        now[0] += sec

    def tick():
        now[0] += 0.25

    def urlopen(req, timeout):
        body = json.loads(req.data)
        requests.append((req.full_url, body, timeout))
        cached = 3000 if len(requests) == 2 else 0
        return _Stream(_sse(len(body["prompt"]), body["max_tokens"], cached), tick)

    out = tmp_path / "llm" / tr.LLM_REPLAY_REPORT
    assert tr.cmd_llm_replay(_llm_args(trace, out), urlopen=urlopen, sleep=sleep, clock=clock) == 0
    assert len(requests) == 2  # the third call has no token counts and is skipped
    url, body, timeout = requests[0]
    assert url == "http://127.0.0.1:8000/v1/completions" and timeout == 7
    assert body["model"] == "m" and body["max_tokens"] == 40 and body["ignore_eos"] is True and body["stream"] is True
    assert body["stream_options"] == {"include_usage": True} and len(body["prompt"]) == 3000
    assert all(tr.LLM_TOKEN_ID_LOW <= t < tr.LLM_TOKEN_ID_HIGH for t in body["prompt"])
    second = requests[1][1]["prompt"]
    assert len(second) == 3200 and second[:3000] == body["prompt"]  # the recorded cache hits: a shared prefix
    # recorded pacing: the second call is issued 10 s after the first, whatever the first took
    assert slept and requests and slept[0] == pytest.approx(10.0 - 8 * 0.25, abs=1e-6)
    report = json.loads(out.read_text())
    rows = report["calls"]
    assert [r["status"] for r in rows] == ["ok", "ok", "skipped"] and rows[2]["reason"] == "no recorded token counts"
    assert rows[0]["offset_sec"] == 0.0 and rows[1]["offset_sec"] == 10.0 and rows[1]["schedule_lag_sec"] == 0.0
    assert rows[0]["wall_time_sec"] == 2.0 and rows[0]["ttft_sec"] == 1.0  # 8 lines x 0.25 s; first token at line 4
    assert rows[1]["cached_tokens"] == 3000 and rows[1]["prompt_tokens"] == 3200 and rows[1]["completion_tokens"] == 60
    s = report["summary"]
    assert s["calls_recorded"] == 3 and s["calls_replayed"] == 2 and s["calls_failed"] == 0 and s["calls_skipped"] == 1
    assert s["llm_wait_live_sec"] == 5.0 and s["llm_wait_replay_sec"] == 4.0 and s["replay_to_live_llm_ratio"] == 0.8
    assert s["prompt_tokens_recorded"] == 6200 and s["prompt_tokens_replayed"] == 6200
    assert s["cached_tokens_recorded"] == 3000 and s["cached_tokens_replayed"] == 3000
    assert s["completion_tokens_recorded"] == 100 and s["completion_tokens_replayed"] == 100
    assert s["generation_tokens_per_sec"] == 25.0 and s["ttft_p50_sec"] == 1.0 and s["pacing"] == "recorded"
    assert report["model"] == "m" and report["api_base"] == "http://127.0.0.1:8000/v1"
    out_text = capsys.readouterr().out
    assert "[llm-replay] call 1: ok  prompt 3000  completion 40  live 2.0s  replay 2.0s" in out_text
    assert "2/3 calls replayed, 0 failed; llm wait replay 4.0s vs live 5.0s (ratio 0.800)" in out_text
    # --lead-only: the first LLM call's offset from the first tool action, negative when it came first
    assert tr.cmd_llm_replay(_llm_args(trace, out, extra=["--lead-only"])) == 0
    assert capsys.readouterr().out.splitlines() == ["LLM_START_OFFSET=-2.500"]
    early_tools = _llm_trace(tmp_path / "early", [(100.0, 2.0, 30, 0, 4)], actions=[99.25])
    assert tr.cmd_llm_replay(_llm_args(early_tools, out, extra=["--lead-only"])) == 0
    assert capsys.readouterr().out.splitlines() == ["LLM_START_OFFSET=0.750"]
    # --start-delay holds the first call back under recorded pacing and is ignored when going flat out
    slept.clear()
    requests.clear()
    args = _llm_args(trace, out, extra=["--start-delay", "0.75"])
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=sleep, clock=clock) == 0
    assert slept[0] == 0.75 and len(requests) == 2
    slept.clear()
    args = _llm_args(trace, out, "unpaced", extra=["--start-delay", "0.75"])
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=sleep, clock=clock) == 0
    assert slept == []
    # unpaced never sleeps
    slept.clear()
    requests.clear()
    assert tr.cmd_llm_replay(_llm_args(trace, out, "unpaced"), urlopen=urlopen, sleep=sleep, clock=clock) == 0
    assert slept == [] and len(requests) == 2


def test_llm_replay_failures_and_bad_inputs(tmp_path, capsys):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 300, 0, 4), (101.0, 2.0, 400, 300, 6), (102.0, 2.0, 500, 400, 8)])
    now = [0.0]

    def clock():
        return now[0]

    def tick():
        now[0] += 0.1

    def urlopen(req, timeout):
        if isinstance(req, str):  # the /health probe after a failure: the server itself is fine
            assert req == "http://127.0.0.1:8000/health" and timeout == 20
            return _Stream([], tick)
        body = json.loads(req.data)
        if len(body["prompt"]) == 400:
            raise urllib.error.HTTPError(req.full_url, 400, "Bad Request", {}, io.BytesIO(b'{"error": "too long"}'))
        if len(body["prompt"]) == 500:
            raise urllib.error.URLError("connection refused")
        return _Stream(_sse(len(body["prompt"]), body["max_tokens"]), tick)

    out = tmp_path / tr.LLM_REPLAY_REPORT
    assert tr.cmd_llm_replay(_llm_args(trace, out), urlopen=urlopen, sleep=lambda s: None, clock=clock) == 1
    report = json.loads(out.read_text())
    assert [r["status"] for r in report["calls"]] == ["ok", "failed", "failed"]
    assert report["calls"][1]["error"] == 'HTTP 400: {"error": "too long"}'
    assert "connection refused" in report["calls"][2]["error"]
    assert report["summary"]["calls_failed"] == 2 and report["summary"]["replay_to_live_llm_ratio"] is not None
    assert "1/3 calls replayed, 2 failed" in capsys.readouterr().out
    # a malformed trajectory, and missing server arguments, are usage errors (exit 2)
    trace.write_bytes(trace.read_bytes() + b"{not json\n")
    assert tr.cmd_llm_replay(_llm_args(trace, out), urlopen=urlopen, sleep=lambda s: None, clock=clock) == 2
    assert "is not valid JSON" in capsys.readouterr().err
    good = _llm_trace(tmp_path / "g", [(1.0, 1.0, 10, 0, 1)]) if (tmp_path / "g").mkdir() is None else None
    args = tr.build_parser().parse_args(["llm-replay", "--trace", str(good), "--output", str(out)])
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=lambda s: None, clock=clock) == 2
    assert "--api-base, --model and --output are required" in capsys.readouterr().err


def test_synthetic_prompt_extends_the_previous_prompt():
    import random

    rng = random.Random(1)
    first = tr.synthetic_prompt([], 5, rng)  # nothing to reuse yet
    assert len(first) == 5
    second = tr.synthetic_prompt(first, 8, rng)  # the conversation grew: the whole previous prompt is the prefix
    assert second[:5] == first and len(second) == 8
    shrunk = tr.synthetic_prompt(second, 2, rng)  # the agent truncated: keep what still fits
    assert shrunk == second[:2]


def test_llm_replay_warmup_only(tmp_path, capsys):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 3000, 0, 40), (110.0, 3.0, 50000, 0, 60)])
    seen = []

    def urlopen(req, timeout):
        body = json.loads(req.data)
        seen.append((len(body["prompt"]), body["max_tokens"]))
        return _Stream(_sse(len(body["prompt"]), body["max_tokens"]), lambda: None)

    args = tr.build_parser().parse_args(
        ["llm-replay", "--trace", str(trace), "--api-base", "http://127.0.0.1:8000/v1", "--model", "m", "--warmup-only"]
    )
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=lambda s: None, clock=lambda: 0.0) == 0
    # the fixed shapes, then the trajectory's largest prompt because it exceeds them
    assert seen == [(n, 16) for n in tr.WARMUP_PROMPT_TOKENS] + [(50000, 16)]
    out = capsys.readouterr().out
    assert out.count("[llm-replay] warm-up: prompt") == len(tr.WARMUP_PROMPT_TOKENS) + 1
    assert "prompt 50000 completion 16" in out
    # nothing is written: no report, and a server that refuses fails the warm-up
    assert not list(tmp_path.glob("*.json"))

    def refuse(req, timeout):
        raise urllib.error.URLError("refused")

    assert tr.cmd_llm_replay(args, urlopen=refuse, sleep=lambda s: None, clock=lambda: 0.0) == 1
    assert f"warm-up request of {tr.WARMUP_PROMPT_TOKENS[0]} prompt tokens failed" in capsys.readouterr().err

    # a stream that closes without its usage record (the engine died) fails the warm-up too
    def truncated(req, timeout):
        assert timeout == tr.WARMUP_TIMEOUT_SEC  # capped below the per-call replay timeout
        return _Stream([b'data: {"choices": [{"text": "", "index": 0}]}\n'], lambda: None)

    assert tr.cmd_llm_replay(args, urlopen=truncated, sleep=lambda s: None, clock=lambda: 0.0) == 1
    assert "without a usage record" in capsys.readouterr().err


def test_llm_replay_stream_without_usage_is_a_failed_call(tmp_path):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 300, 0, 4), (101.0, 2.0, 400, 300, 6)])

    def urlopen(req, timeout):
        if isinstance(req, str):
            return _Stream([], lambda: None)  # /health still answers
        body = json.loads(req.data)
        if len(body["prompt"]) == 400:
            return _Stream([b'data: {"choices": [{"text": "x", "index": 0}]}\n'], lambda: None)  # no usage, no DONE
        return _Stream(_sse(len(body["prompt"]), body["max_tokens"]), lambda: None)

    out = tmp_path / tr.LLM_REPLAY_REPORT
    args = _llm_args(trace, out, "unpaced")
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=lambda s: None, clock=lambda: 0.0) == 1
    rows = json.loads(out.read_text())["calls"]
    assert rows[0]["status"] == "ok" and rows[1]["status"] == "failed"
    assert "without a usage record" in rows[1]["error"]


# ---------------------------------------------------------------- gpu recipes: summaries, gate, verdict


def _llm_report(tmp_path, replay=4.0, live=5.0, failed=0, calls=2, skipped=0):
    report = {
        "pacing": "recorded",
        "api_base": "http://127.0.0.1:8000/v1",
        "model": "nemotron-3-ultra-nvfp4",
        "summary": {
            "calls_recorded": calls + failed + skipped,
            "calls_replayed": calls,
            "calls_failed": failed,
            "calls_skipped": skipped,
            "llm_wait_live_sec": live,
            "llm_wait_replay_sec": replay,
            "replay_to_live_llm_ratio": round(replay / live, 4),
        },
    }
    path = tmp_path / tr.LLM_REPLAY_REPORT
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report))
    return path


def _server_json(tmp_path):
    path = tmp_path / "server.json"
    path.write_text(
        json.dumps(
            {
                "engine": "vllm",
                "version": "0.11.0",
                "served_model": "nemotron-3-ultra-nvfp4",
                "tp": 4,
                "gpu_count": 4,
                "gpu_name": "NVIDIA GB300",
                "startup_sec": 400,
            }
        )
    )
    return path


def test_summarize_replay_gpu_llm_calls_and_verdict(tmp_path, capsys):
    prof = _profile_dir(tmp_path)
    server = _server_json(tmp_path)

    def argv(report, llm_exit="0"):
        return [
            "summarize-replay",
            "--out-dir",
            str(prof),
            "--summary",
            str(tmp_path / "s.json"),
            "--trace-id",
            "t",
            "--pacing",
            "recorded",
            "--cpus",
            "4",
            "--replay-exit",
            "0",
            "--workload",
            "tasktrace_replay_gpu",
            "--server-info",
            str(server),
            "--llm-report",
            str(report),
            "--llm-exit",
            llm_exit,
        ]

    good = _llm_report(tmp_path / "good")
    assert run(argv(good)) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["workload"] == "tasktrace_replay_gpu" and s["llm_replay"]["ran"] and s["llm_replay"]["exit_status"] == 0
    assert s["llm_comparison"] == {
        "llm_wait_replay_sec": 4.0,
        "calls_compared": 2,
        "calls_failed": 0,
        "calls_skipped": 0,
    }
    assert s["server"]["tp"] == 4 and s["diagnostics"] == [] and s["fidelity"] == "PASS"
    capsys.readouterr()
    # the verdict shows the server and the LLM side's time, reported without comparing it
    exp = tmp_path / "exp"
    exp.mkdir()
    shutil.copy(tmp_path / "s.json", exp / "summary.json")
    (exp / "slurm-777.out").write_text("ok\n")
    assert run(["verdict", "--no-color", str(exp)]) == 0
    out = capsys.readouterr().out
    assert "Server      : vllm 0.11.0  |  nemotron-3-ultra-nvfp4  |  TP4  |  4x NVIDIA GB300  |  ready in 400.0s" in out
    assert "LLM time    : 4.0s replay  |  2/2 calls  |  pacing recorded\n" in out
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert capsys.readouterr().out.splitlines()[0].endswith(" actions, 0 mismatches  llm time 4.0s")
    # an LLM side slower than the recording is reported, not compared
    slow = _llm_report(tmp_path / "slow", replay=5.5)
    assert run(argv(slow)) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["llm_comparison"]["llm_wait_replay_sec"] == 5.5 and s["diagnostics"] == []
    # a failed call fails the run
    bad = _llm_report(tmp_path / "bad", replay=1.0, failed=1)
    assert run(argv(bad, llm_exit="1")) == 1
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["diagnostics"] == [
        f"llm calls: 1 of the replayed LLM calls failed (see llm-replay/{tr.LLM_REPLAY_REPORT} in the run directory)"
    ]
    # a recorded call without token counts cannot be replayed: the recording is incomplete, FAIL
    partial = _llm_report(tmp_path / "partial", skipped=1)
    assert run(argv(partial)) == 1
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["diagnostics"] == [
        "llm calls: 1 recorded LLM calls carry no token counts and were not replayed; the recording cannot be replayed in full"
    ]
    # the replayer crashing (exit 2) or leaving no report is a FAIL too
    assert run(argv(good, llm_exit="2")) == 1
    assert json.loads((tmp_path / "s.json").read_text())["diagnostics"] == ["llm replay: exited 2"]
    assert run(argv(tmp_path / "missing" / tr.LLM_REPLAY_REPORT)) == 1
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["llm_replay"]["ran"] is False and s["diagnostics"][0] == "llm replay did not run (no report)"
    shutil.copy(tmp_path / "s.json", exp / "summary.json")
    run(["verdict", "--no-color", str(exp)])
    assert "LLM time    : llm replay did not run" in capsys.readouterr().out


def test_summarize_capture_gpu_carries_server_and_llm(tmp_path):
    prov = _provenance(tmp_path)
    p = json.loads(prov.read_text())
    p["live_run"].update(llm_mode="live-server", server={"engine": "vllm", "tp": 4})
    prov.write_text(json.dumps(p))
    prof = _profile_dir(tmp_path)
    (tmp_path / "scan.json").write_text(json.dumps({"files_redacted": []}))
    good = _llm_report(tmp_path / "good")
    args = [
        "summarize-capture",
        "--provenance",
        str(prov),
        "--profile-dir",
        str(prof),
        "--summary",
        str(tmp_path / "s.json"),
        "--scrub-report",
        str(tmp_path / "scan.json"),
        "--pacing",
        "recorded",
        "--cpus",
        "4",
        "--stage-exit",
        "0",
        "--replay-exit",
        "0",
        "--workload",
        "tasktrace_capture_replay_gpu",
        "--llm-report",
        str(good),
        "--llm-exit",
        "0",
    ]
    assert run(args) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["workload"] == "tasktrace_capture_replay_gpu" and s["capture"]["llm_mode"] == "live-server"
    assert s["server"] == {"engine": "vllm", "tp": 4}  # from provenance when no --server-info is given
    assert s["llm_comparison"]["calls_compared"] == 2 and s["diagnostics"] == []
    # the capture-side checks still gate a gpu run
    prov2 = _provenance(tmp_path / "other", sha="different")
    assert run(args[:2] + [str(prov2)] + args[3:]) == 1
    # a slower LLM side is reported, not judged; a failed call still fails the run
    slow = _llm_report(tmp_path / "slow", replay=6.0)
    args[args.index("--llm-report") + 1] = str(slow)
    assert run(args) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["fidelity"] == "PASS" and s["llm_comparison"]["replay_to_live_llm_ratio"] == 1.2
    assert s["diagnostics"] == []
    bad = _llm_report(tmp_path / "bad", failed=1)
    args[args.index("--llm-report") + 1] = str(bad)
    args[args.index("--llm-exit") + 1] = "1"
    assert run(args) == 1


def test_verdict_table_gpu_row(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(tr, "slurm_states", lambda ids: {})
    root = tmp_path / "experiments"
    gpu = _write_summary(
        root / "wl_g",
        workload="tasktrace_replay_gpu",
        llm_replay={"ran": True, "exit_status": 0, "pacing": "recorded", "summary": {"calls_recorded": 3}},
        llm_comparison={"llm_wait_replay_sec": 8.12, "calls_compared": 3, "calls_failed": 0},
    )
    (gpu / "slurm-201.out").write_text("ok\n")
    cpu = _write_summary(root / "wl_c")
    (cpu / "slurm-202.out").write_text("ok\n")
    assert run(["verdict", str(root)]) == 0
    out = capsys.readouterr().out
    assert re.search(
        r"^201\s+PASS\s+replay gpu\s+-\s+-\s+113/114 act, 0 mism\s+-\s+-\s*$",
        out,
        re.M,
    )
    assert re.search(
        r"^202\s+PASS\s+replay\s+-\s+-\s+113/114 act, 0 mism\s+-\s+-\s*$",
        out,
        re.M,
    )
    # a summary that stored an "llm time:" reason passes: time is reported, not judged
    slow = {
        "llm_replay": {"ran": True, "exit_status": 0, "summary": {}},
        "llm_comparison": {"replay_to_live_llm_ratio": 1.2, "llm_wait_live_sec": 10.0, "llm_wait_replay_sec": 12.0},
        "diagnostics": ["llm time: replay 12.0s is 20.0% above live 10.0s (limit +5%)"],
    }
    exp = _write_summary(tmp_path / "old", workload="tasktrace_replay_gpu", **slow)
    assert run(["verdict", "--oneline", str(exp)]) == 0
    assert capsys.readouterr().out.splitlines()[0].startswith("Outcome: PASS")
    # one whose LLM side lost calls derives that reason
    lost = dict(slow, llm_comparison=dict(slow["llm_comparison"], calls_failed=2), diagnostics=None)
    exp = _write_summary(tmp_path / "lost", workload="tasktrace_replay_gpu", **lost)
    assert run(["verdict", "--oneline", str(exp)]) == 1
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("Outcome: FAIL")
    assert any(line.strip().startswith("llm calls: 2 of the replayed LLM calls failed") for line in lines[1:])


def test_run_mode_detects_gpu_recipes(tmp_path):
    assert tr.run_mode(tmp_path, {"workload": "tasktrace_replay_gpu"}) == "replay gpu"
    assert tr.run_mode(tmp_path, {"workload": "tasktrace_capture_replay_gpu"}) == "capture+replay gpu"
    assert tr.run_mode(tmp_path, {"workload": "tasktrace_replay"}) == "replay"
    cpu = tmp_path / "inference_tasktrace_replay_nemotron-3-ultra_miniswe_pytest-7236_nvfp4_gpus4" / "1"
    cpu.mkdir(parents=True)
    assert tr.run_mode(cpu) == "replay"  # the experiment directory's `gpus4` is a scale, not the gpu recipe
    gpu = tmp_path / "inference_tasktrace_replay_gpu_nemotron-3-ultra_miniswe_pytest-7236_nvfp4_gpus4" / "1"
    gpu.mkdir(parents=True)
    assert tr.run_mode(gpu) == "replay gpu"
    (cpu / "llmb-config_1.yaml").write_text("model_info:\n  model_name: tasktrace_capture_replay_gpu\n")
    assert tr.run_mode(cpu) == "capture+replay gpu"


@pytest.mark.parametrize(
    "line,needle",
    [
        ("[launch] ERROR: server allocation exposes 0 GPUs, need 4 for tensor parallelism 4.", "fewer GPUs"),
        ("[launch] ERROR: server serving image missing at /x.sqsh.", "vLLM image is not installed"),
        ("[launch] ERROR: server model snapshot missing at /d (no config.json).", "weights are not installed"),
        ("[launch] ERROR: server did not become ready within 3600s; see /log.", "did not come up"),
        ("[launch] ERROR: server exited before it became ready (pid 5); see /log.", "did not come up"),
        ("[launch] ERROR: server port 8000 on 127.0.0.1 is already in use on n1.", "RUN_CONF_SERVER_PORT"),
    ],
)
def test_launch_cause_names_server_failures(line, needle):
    assert needle in tr.launch_cause([line])


def test_llm_replay_aborts_when_the_server_stops_answering(tmp_path, capsys):
    calls = [(100.0, 2.0, 300, 0, 4), (101.0, 2.0, 400, 300, 6), (102.0, 2.0, 500, 400, 8), (103.0, 2.0, 600, 500, 8)]
    trace = _llm_trace(tmp_path, calls)
    seen = []
    alive = [True]

    def urlopen(req, timeout):
        url = req if isinstance(req, str) else req.full_url
        seen.append(url)
        if url.endswith("/health"):
            if alive[0]:
                return _Stream([], lambda: None)
            raise urllib.error.URLError("refused")
        body = json.loads(req.data)
        if len(body["prompt"]) == 400:  # a plain request failure while the server is fine: carry on
            raise urllib.error.HTTPError(url, 400, "Bad Request", {}, io.BytesIO(b"too long"))
        if len(body["prompt"]) == 500:  # the server wedges here
            alive[0] = False
            raise TimeoutError("timed out")
        return _Stream(_sse(len(body["prompt"]), body["max_tokens"]), lambda: None)

    out = tmp_path / tr.LLM_REPLAY_REPORT
    args = _llm_args(trace, out, "unpaced")
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=lambda s: None, clock=lambda: 0.0) == 1
    report = json.loads(out.read_text())
    assert [r["status"] for r in report["calls"]] == ["ok", "failed", "failed", "failed"]
    assert report["calls"][3]["error"].startswith("not attempted: the server stopped answering /health after call 3")
    assert report["calls"][3]["source_prompt_tokens"] == 600 and "wall_time_sec" not in report["calls"][3]
    assert report["summary"]["calls_failed"] == 3 and report["summary"]["calls_replayed"] == 1
    # the health probe ran after each failure (twice) and the fourth completion was never sent
    assert seen.count("http://127.0.0.1:8000/health") == 2 and seen.count("http://127.0.0.1:8000/v1/completions") == 3
    assert "aborting the remaining calls" in capsys.readouterr().err


def test_shipped_gpu_manifest_resolves(capsys):
    """The gpu recipes' reference: the capture the capture gpu recipe recorded against the local vLLM
    server without its timings. It must resolve, name its serving config, carry its checksum,
    and carry no measured time."""
    root = pathlib.Path(__file__).resolve().parents[4] / "trajectories"
    selector = "nemotron-3-ultra-vllm_miniswe_django-11490"
    assert run(["resolve-trace", "--root", str(root), "--selector", selector]) == 0
    vals = dict(line.split("=", 1) for line in capsys.readouterr().out.strip().splitlines())
    assert vals["TRACE_MODEL"] == "nemotron-3-ultra-vllm" and vals["IMAGE_ARCH"] == "aarch64"
    assert vals["SERVING_CONFIG"] == "nemotron-3-ultra-nvfp4" and vals["LLM_MODE"] == "live-server"
    assert vals["IMAGE_FILE"] == "epoch-research+swe-bench.eval.arm64.django__django-11490+sha256-58f9edb89a62.sqsh"
    trace = pathlib.Path(vals["TRACE"])
    recorded = (trace.parent / (tr.TRAJECTORY_NAME + ".sha256")).read_text().split()[0]
    assert hashlib.sha256(trace.read_bytes()).hexdigest() == recorded
    header, actions, llm_calls, _ = tr.read_trajectory(trace)
    assert header["replay_supported"] is True and len(llm_calls) == 111 and len(actions) == 112
    assert tr.trajectory_timed(trace) is False
    # the serving config it names exists
    assert (pathlib.Path(__file__).resolve().parents[1] / "serving" / "nemotron-3-ultra-nvfp4.env").is_file()


def _recording_server(requests, completion=None):
    """A fake urlopen that records each request body and answers with `completion` tokens (default: as asked)."""

    def urlopen(req, timeout):
        body = json.loads(req.data)
        requests.append(body)
        return _Stream(_sse(len(body["prompt"]), completion or body["max_tokens"]), lambda: None)

    return urlopen


def test_llm_replay_warm_up_does_not_share_a_prefix_with_the_timed_calls(tmp_path):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 3000, 0, 40)])
    warm, timed = [], []
    base = ["llm-replay", "--trace", str(trace), "--api-base", "http://127.0.0.1:8000/v1", "--model", "m"]
    args = tr.build_parser().parse_args([*base, "--warmup-only"])
    assert tr.cmd_llm_replay(args, urlopen=_recording_server(warm), sleep=lambda s: None, clock=lambda: 0.0) == 0
    out = tmp_path / tr.LLM_REPLAY_REPORT
    args = _llm_args(trace, out, "unpaced")
    assert tr.cmd_llm_replay(args, urlopen=_recording_server(timed), sleep=lambda s: None, clock=lambda: 0.0) == 0
    first = timed[0]["prompt"]
    # no warm-up prompt begins like the first timed prompt, so the warm-up cannot prime its prefix cache
    assert all(body["prompt"][:16] != first[:16] for body in warm)


def test_llm_replay_short_completion_is_a_failed_call(tmp_path, capsys):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 300, 0, 40)])
    out = tmp_path / tr.LLM_REPLAY_REPORT
    args = _llm_args(trace, out, "unpaced")
    urlopen = _recording_server([], completion=25)
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=lambda s: None, clock=lambda: 0.0) == 1
    row = json.loads(out.read_text())["calls"][0]
    assert row["status"] == "failed" and row["error"] == "the server generated 25 of 40 completion tokens"
    assert json.loads(out.read_text())["summary"]["calls_failed"] == 1


def test_llm_replay_aligns_to_the_tool_side_start(tmp_path, capsys):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 300, 0, 4), (104.0, 2.0, 400, 300, 6)], actions=[99.0])
    marker = tmp_path / "container_profile" / "start_ns"
    marker.parent.mkdir()
    now, polls, requests = [0.0], [], []

    def clock():
        return now[0]

    def sleep(sec):
        polls.append(round(sec, 3))
        now[0] += sec
        if len(polls) == 3:  # the tool replay's container starts during the third poll
            marker.write_text("123\n")

    def urlopen(req, timeout):
        requests.append(round(now[0], 3))
        return _Stream(_sse(len(json.loads(req.data)["prompt"]), json.loads(req.data)["max_tokens"]), lambda: None)

    out = tmp_path / tr.LLM_REPLAY_REPORT
    extra = ["--align-file", str(marker), "--start-delay", "1.0"]
    assert tr.cmd_llm_replay(_llm_args(trace, out, extra=extra), urlopen=urlopen, sleep=sleep, clock=clock) == 0
    # the marker appeared at 0.03 s; the first call follows it by the recorded 1.0 s, the second 4 s later
    assert polls[:3] == [tr.ALIGN_POLL_SEC] * 3 and requests == [1.03, 5.03]
    report = json.loads(out.read_text())
    assert report["aligned_to"] == str(marker) and report["timeline_start_monotonic_sec"] == 1.03
    # a negative offset (the recording's LLM call came first) starts the timeline in the past: no wait,
    # and the lag is reported
    now[0], polls[:], requests[:] = 0.0, [], []
    marker.write_text("123\n")
    extra = ["--align-file", str(marker), "--start-delay", "-0.5"]
    assert tr.cmd_llm_replay(_llm_args(trace, out, extra=extra), urlopen=urlopen, sleep=sleep, clock=clock) == 0
    assert requests[0] == 0.0 and json.loads(out.read_text())["calls"][0]["schedule_lag_sec"] == 0.5


def test_llm_replay_gives_up_when_the_tool_side_never_starts(tmp_path, capsys, monkeypatch):
    trace = _llm_trace(tmp_path, [(100.0, 2.0, 300, 0, 4), (104.0, 2.0, 400, 300, 6)])
    out = tmp_path / tr.LLM_REPLAY_REPORT
    extra = ["--align-file", str(tmp_path / "never"), "--align-pid", "4242", "--start-delay", "3"]
    args = _llm_args(trace, out, extra=extra)
    now = [0.0]

    def sleep(sec):
        now[0] += sec

    assert tr._process_alive(os.getpid()) and not tr._process_alive(2**22 + 1)  # the real check
    monkeypatch.setattr(tr, "_process_alive", lambda pid: now[0] < 0.05)  # the tool replay exits after a few polls
    rc = tr.cmd_llm_replay(args, urlopen=_recording_server([]), sleep=sleep, clock=lambda: now[0])
    assert rc == 1 and now[0] < 1.0  # no start delay slept out once nothing will be replayed
    report = json.loads(out.read_text())
    assert [r["status"] for r in report["calls"]] == ["failed", "failed"]
    assert report["calls"][0]["error"].startswith("not attempted: the tool replay ended or timed out")
    assert "ERROR: not attempted" in capsys.readouterr().err


# ---------------------------------------------------------------- runtime-aware provenance / summaries


def test_stage_records_launcher_runtime_and_image_identity(tmp_path, monkeypatch):
    """Slurm (enroot, .sqsh file) and the current-container runtime (Kubernetes pod = task image, OCI digest)
    are both recorded faithfully instead of the historical hardcoded 'enroot'."""
    trace = _capture_layout(tmp_path)
    _stage_env(monkeypatch, tmp_path)
    prov = tmp_path / "staged" / "provenance.json"
    argv = ["stage", "--trace", str(trace), "--staged-dir", str(tmp_path / "staged"), "--provenance", str(prov)]
    assert run(argv, monkeypatch) == 0
    p = json.loads(prov.read_text())
    assert p["live_run"]["runtime"] == "enroot"  # from the capture report when the launcher says nothing
    assert p["task_image"] == {
        "reference": "ghcr.io/x:latest",
        "path": "/img.sqsh",
        "sha256": "abc",
        "identity": "sqsh-file",
        "arch": "aarch64",
    }
    monkeypatch.setenv("TT_RUNTIME", "current")
    monkeypatch.setenv("TT_IMAGE_PATH", "")
    monkeypatch.setenv("TT_IMAGE_IDENTITY", "oci-digest")
    monkeypatch.setenv("TT_IMAGE_SHA256", "945da72e")
    assert run(argv, monkeypatch) == 0
    p = json.loads(prov.read_text())
    assert p["live_run"]["runtime"] == "current"
    assert p["task_image"]["path"] is None
    assert p["task_image"]["identity"] == "oci-digest" and p["task_image"]["sha256"] == "945da72e"


def _current_pipeline(prof, sha=None, valid=True):
    (prof / "pipeline_report.json").write_text(
        json.dumps(
            {
                "status": "valid",
                "profiling_invalid_reasons": [],
                "runtime": "current",
                "image_identity": {"sha256": sha} if sha else None,
                "container_cpu": {
                    "method": "current-container-cgroup-v2-cpu.stat",
                    "wall_sec": 0.12,
                    "usage_sec": 0.052,
                    "user_sec": 0.021,
                    "system_sec": 0.031,
                    "average_logical_cores_used": 0.435,
                    "allocated_logical_cpus": 1.0,
                    "allocated_cpu_source": "user-supplied",
                    "utilization_percent_of_allocation": 43.5,
                    "enforced_logical_cpu_limit": 1.0,
                    "enforced_cpu_limit_source": "cgroup-v2-cpu.max",
                    "cpu_max": "100000 100000",
                    "cgroup_cpuset": "0-11",
                    "process_affinity_logical_cpu_ids": [0, 1, 2, 3],
                    "effective_logical_cpu_count": 12,
                    "throttling": {"throttled_periods": 0, "throttled_sec": 0.0},
                    "raw_cpu_stat_delta": {"usage_usec": 52175},
                },
                "measurement_validity": {
                    "valid_for_average_cpu_reporting": valid,
                    "exclusive_single_task_container_asserted": True,
                },
            }
        )
    )


def test_summarize_capture_current_runtime_takes_the_launcher_image_digest(tmp_path, monkeypatch):
    prov = _provenance(tmp_path, sha="945da72e")
    prof = _profile_dir(tmp_path)
    _current_pipeline(prof)  # current mode: the pipeline carries no image identity
    (tmp_path / "scan.json").write_text(json.dumps({"files_redacted": []}))
    base = [
        "summarize-capture",
        "--provenance",
        str(prov),
        "--profile-dir",
        str(prof),
        "--summary",
        str(tmp_path / "s.json"),
        "--scrub-report",
        str(tmp_path / "scan.json"),
        "--pacing",
        "recorded",
        "--cpus",
        "4",
        "--stage-exit",
        "0",
        "--replay-exit",
        "0",
    ]
    monkeypatch.delenv("TT_IMAGE_SHA256", raising=False)
    assert run(base, monkeypatch) == 1
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["image_identity_check"]["match"] is False
    assert "no image digest supplied" in s["image_identity_check"]["source"]
    assert any("image identity mismatch" in d for d in s["diagnostics"])
    assert run(base + ["--replay-image-sha256", "945da72e"], monkeypatch) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["image_identity_check"] == {
        "capture_image_sha256": "945da72e",
        "replay_image_sha256": "945da72e",
        "match": True,
        "source": "current-container: image digest supplied by the launcher",
    }
    assert s["fidelity"] == "PASS" and s["diagnostics"] == []
    monkeypatch.setenv("TT_IMAGE_SHA256", "945da72e")
    assert run(base, monkeypatch) == 0
    cpu = json.loads((tmp_path / "s.json").read_text())["cpu"]
    assert cpu["runtime"] == "current" and cpu["method"] == "current-container-cgroup-v2-cpu.stat"
    assert cpu["usage_sec"] == 0.052 and cpu["utilization_percent_of_allocation"] == 43.5
    assert cpu["cpu_max"] == "100000 100000" and cpu["cgroup_cpuset"] == "0-11"
    assert cpu["process_affinity_logical_cpu_ids"] == [0, 1, 2, 3]
    assert cpu["valid_for_average_cpu_reporting"] is True
    assert "raw_cpu_stat_delta" not in cpu  # raw counters stay in the pipeline report
    (prof / "pipeline_report.json").write_text(
        json.dumps(
            {
                "status": "valid",
                "profiling_invalid_reasons": [],
                "image_identity": {"sha256": "945da72e", "runtime": "enroot"},
            }
        )
    )
    assert run(base + ["--replay-image-sha256", "ignored"], monkeypatch) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["image_identity_check"]["source"] == "pipeline_report" and s["image_identity_check"]["match"] is True
    assert s["cpu"] is None  # that fixture carries no container_cpu block


def test_summarize_replay_carries_the_cpu_block(tmp_path):
    prof = _profile_dir(tmp_path)
    _current_pipeline(prof, valid=False)
    argv = [
        "summarize-replay",
        "--out-dir",
        str(prof),
        "--summary",
        str(tmp_path / "r.json"),
        "--trace-id",
        "t",
        "--pacing",
        "recorded",
        "--cpus",
        "4",
        "--replay-exit",
        "0",
    ]
    assert run(argv) == 0
    s = json.loads((tmp_path / "r.json").read_text())
    assert s["fidelity"] == "PASS"
    assert s["cpu"]["usage_sec"] == 0.052 and s["cpu"]["valid_for_average_cpu_reporting"] is False
    assert s["cpu"]["exclusive_single_task_container_asserted"] is True
    assert s["image_identity"] is None


def test_trajectory_timed_tells_an_untimed_trajectory_from_a_recording(tmp_path):
    action = {"record_type": "action", "action_index": 1, "command": "ls", "returncode": 0}
    header = {"record_type": "trace", "schema_version": "tasktrace-cpu-trace.v1"}
    timed, untimed, empty = tmp_path / "timed.jsonl", tmp_path / "untimed.jsonl", tmp_path / "empty.jsonl"
    timed.write_text(json.dumps(header) + "\n" + json.dumps(dict(action, started_at_epoch=1.0)) + "\n")
    untimed.write_text(json.dumps(header) + "\n" + json.dumps(dict(action, after_llm_index=None)) + "\n")
    empty.write_text(json.dumps(header) + "\n")
    assert tr.trajectory_timed(timed) is True
    assert tr.trajectory_timed(untimed) is False
    assert tr.trajectory_timed(empty) is None and tr.trajectory_timed(tmp_path / "missing.jsonl") is None


def test_trace_timed_reports_what_the_capture_wrote(tmp_path, capsys):
    action = {"record_type": "action", "action_index": 1, "command": "ls", "returncode": 0}
    header = {"record_type": "trace", "schema_version": "tasktrace-cpu-trace.v1"}
    for name, row, expected in (("timed", dict(action, started_at_epoch=1.0), "1"), ("untimed", action, "0")):
        trace = tmp_path / f"{name}.jsonl"
        trace.write_text(json.dumps(header) + "\n" + json.dumps(row) + "\n")
        assert run(["trace-timed", "--trace", str(trace)]) == 0
        assert capsys.readouterr().out.strip() == f"TRACE_TIMED={expected}"


def test_stage_of_a_recording_without_timings_leaves_the_live_times_unknown(tmp_path, monkeypatch, capsys):
    trace = _capture_layout(tmp_path)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    for row in rows:
        for field in ("started_at_epoch", "finished_at_epoch", "wall_time_sec"):
            row.pop(field, None)
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))
    _stage_env(monkeypatch, tmp_path)
    server = tmp_path / "server.json"
    server.write_text(json.dumps({"engine": "vllm", "tp": 4, "startup_sec": 300, "started_at_utc": "s"}))
    monkeypatch.setenv("TT_SERVER_INFO", str(server))
    staged = tmp_path / "staged"
    prov = staged / "provenance.json"
    assert run(["stage", "--trace", str(trace), "--staged-dir", str(staged), "--provenance", str(prov)]) == 0
    p = json.loads(prov.read_text())
    assert p["trace"]["timings"] == "omitted"
    live = p["live_run"]
    assert live["tool_busy_sec"] is None and live["llm_wait_sec"] is None
    # the record staged beside an untimed trajectory carries no measured time either
    assert live["started_at_utc"] is None and live["finished_at_utc"] is None
    assert live["capture_report"]["timing"] is None and live["capture_report"]["phases"] is None
    assert live["server"]["startup_sec"] is None and live["server"]["started_at_utc"] is None
    assert live["capture_report"]["hostname"] == "n1" and live["server"]["tp"] == 4
    assert "llm_wait=n/a tool_busy=n/a" in capsys.readouterr().out


def test_summarize_capture_of_an_untimed_recording_reports_without_comparing(tmp_path, capsys):
    prov = _provenance(tmp_path)
    p = json.loads(prov.read_text())
    p["trace"]["timings"] = "omitted"
    p["live_run"]["tool_busy_sec"] = None
    prov.write_text(json.dumps(p))
    prof = _profile_dir(tmp_path)
    (tmp_path / "scan.json").write_text(json.dumps({"files_redacted": []}))
    argv = ["summarize-capture", "--provenance", str(prov), "--profile-dir", str(prof), "--summary"]
    argv += [str(tmp_path / "s.json"), "--scrub-report", str(tmp_path / "scan.json"), "--pacing", "unpaced"]
    argv += ["--cpus", "4", "--stage-exit", "0", "--replay-exit", "0"]
    assert run(argv) == 0
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["comparison"] == {"tool_busy_replay_sec": 3.4}
    assert not tr.compared_with_recording(s)
    exp = tmp_path / "exp"
    exp.mkdir()
    (exp / "summary.json").write_text((tmp_path / "s.json").read_text())
    (exp / "slurm-9.out").write_text("ok\n")
    capsys.readouterr()
    assert run(["verdict", "--no-color", str(exp)]) == 0
    out = capsys.readouterr().out
    assert "Tool time   : 3.4s replay" in out and "replay/live" not in out


def _untimed_llm_trace(tmp_path, afters):
    """A trajectory written without timings: two LLM calls, and actions that record the call they followed."""
    rows = [{"record_type": "trace", "schema_version": "tasktrace-capture-cpu-trace.v1", "replay_supported": True}]
    rows += [
        {
            "record_type": "llm",
            "llm_index": k,
            "prompt_tokens": 100 * k,
            "cached_prompt_tokens": 0,
            "completion_tokens": 4,
        }
        for k in (1, 2)
    ]
    rows += [
        {"record_type": "action", "action_index": j, "command": "ls", "after_llm_index": after}
        for j, after in enumerate(afters, 1)
    ]
    path = tmp_path / "trajectory.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def test_llm_replay_loop_order_alternates_with_the_tool_side(tmp_path):
    afters = [None, 1, 2]
    trace = _untimed_llm_trace(tmp_path, afters)
    sync = tmp_path / "sync"
    sync.mkdir()  # the tool replay creates it when it starts
    order, now, next_action = [], [0.0], [1]

    def clock():
        return now[0]

    def sleep(sec):
        # the fake tool side runs its next action once the LLM call that action followed is done
        now[0] += sec
        while next_action[0] <= len(afters):
            after = afters[next_action[0] - 1]
            done = int((sync / "llm_done").read_text()) if (sync / "llm_done").exists() else 0
            if after is not None and done < after:
                break
            order.append(f"a{next_action[0]}")
            (sync / "tool_done").write_text(f"{next_action[0]}\n")
            next_action[0] += 1

    def urlopen(req, timeout):
        order.append(f"L{sum(o.startswith('L') for o in order) + 1}")
        body = json.loads(req.data)
        return _Stream(_sse(len(body["prompt"]), body["max_tokens"]), lambda: None)

    out = tmp_path / tr.LLM_REPLAY_REPORT
    args = _llm_args(trace, out, pacing="unpaced", extra=["--loop-sync-dir", str(sync)])
    assert tr.cmd_llm_replay(args, urlopen=urlopen, sleep=sleep, clock=clock) == 0
    assert order[:4] == ["a1", "L1", "a2", "L2"]  # the agent loop's order: no call before the actions it followed
    report = json.loads(out.read_text())
    assert report["loop_sync_dir"] == str(sync) and all("loop_wait_sec" in call for call in report["calls"])
    # an untimed trajectory recorded no call times: the live wait is unknown, not zero
    assert report["summary"]["llm_wait_live_sec"] is None and report["summary"]["replay_to_live_llm_ratio"] is None
    assert (sync / "llm_end").exists() and (sync / "llm_done").read_text().strip() == "2"


def test_llm_replay_loop_order_gives_up_when_the_tool_side_never_starts(tmp_path, capsys):
    trace = _untimed_llm_trace(tmp_path, [None, 1])
    out = tmp_path / tr.LLM_REPLAY_REPORT
    extra = ["--loop-sync-dir", str(tmp_path / "never"), "--align-pid", str(2**22 + 1)]
    args = _llm_args(trace, out, pacing="unpaced", extra=extra)
    now = [0.0]

    def sleep(sec):
        now[0] += sec

    assert tr.cmd_llm_replay(args, urlopen=None, sleep=sleep, clock=lambda: now[0]) == 1
    calls = json.loads(out.read_text())["calls"]
    assert [c["status"] for c in calls] == ["failed", "failed"]
    assert "the tool replay ended or timed out before it started" in calls[0]["error"]


def test_llm_replay_loop_order_needs_unpaced_pacing_and_the_loop_order(tmp_path, capsys):
    out = tmp_path / tr.LLM_REPLAY_REPORT
    trace = _untimed_llm_trace(tmp_path, [None])
    args = _llm_args(trace, out, pacing="recorded", extra=["--loop-sync-dir", str(tmp_path / "s")])
    assert tr.cmd_llm_replay(args) == 2
    assert "use --pacing unpaced" in capsys.readouterr().err
    timed = _llm_trace(tmp_path / "timed", [(100.0, 2.0, 300, 0, 4)], actions=[99.0])
    args = _llm_args(timed, out, pacing="unpaced", extra=["--loop-sync-dir", str(tmp_path / "s")])
    assert tr.cmd_llm_replay(args) == 2
    assert "has no after_llm_index" in capsys.readouterr().err
