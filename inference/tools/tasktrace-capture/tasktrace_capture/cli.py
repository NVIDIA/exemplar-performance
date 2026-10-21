#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Agentic CPU scaling sweep for Harbor on Slurm/Enroot.

Thin CLI that orchestrates: monitor -> runner -> plot.

Usage:
    # hello-world, 1 agent (default)
    python cli.py --api-base http://<vllm-host>:8000/v1

    # scaling sweep
    python cli.py -d swebench --sweep 1 2 4 8 \\
        --api-base http://<vllm-host>:8000/v1

    # single task from registry
    python cli.py --task terminal-bench/compile-compcert \\
        --concurrency 1,2,4 --api-base http://<vllm-host>:8000/v1

    # suite mode: multiple tasks, concurrency is a multiplier
    python cli.py \\
        --task terminal-bench/cobol-modernization,sqlite-with-gcov,qemu-startup \\
        --concurrency 1,2,4 --api-base http://<vllm-host>:8000/v1

    # dry run: preview experiment plan without running
    python cli.py --task terminal-bench/qemu-startup \\
        --concurrency 1,2,4 --dry-run
"""

import argparse
import ipaddress
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import yaml

from .models import BaselineCpu, CpuInfo, SweepData
from .monitor import collect_system_info, get_cpu_info
from .runner import (
    _compute_trials,
    find_latest_harbor_job,
    get_active_harbor_pgid,
    get_found_suite_tasks,
    normalize_registry_task_selection,
    parse_agent_time_split,
    run_experiment,
)


def _materialize_adhoc_task(prompt_body: str, dest_dir: Path, image: str, agent_timeout_sec: int) -> Path:
    """Write a minimal Harbor task directory under *dest_dir* and return its path.

    Layout matches harbor/examples/tasks/hello-world/ but with a no-op verifier
    that always rewards 1.0 — tasktrace-capture's value here is telemetry, not pass/fail.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    (dest_dir / "environment").mkdir(exist_ok=True)
    (dest_dir / "solution").mkdir(exist_ok=True)
    (dest_dir / "tests").mkdir(exist_ok=True)

    (dest_dir / "task.toml").write_text(
        'version = "1.0"\n'
        '\n'
        '[metadata]\n'
        'author_name = "tasktrace-capture"\n'
        'difficulty = "n/a"\n'
        'category = "ad-hoc"\n'
        'tags = ["tasktrace-capture-adhoc"]\n'
        '\n'
        f'[agent]\ntimeout_sec = {agent_timeout_sec}\n'
        '\n'
        '[verifier]\ntimeout_sec = 60\n'
        '\n'
        '[environment]\n'
        f'docker_image = {json.dumps(image)}\n'
        'build_timeout_sec = 600\n'
        'cpus = 1\n'
        'memory = "2G"\n'
        'storage = "10G"\n'
    )

    (dest_dir / "instruction.md").write_text(prompt_body if prompt_body.endswith("\n") else prompt_body + "\n")

    (dest_dir / "environment" / "Dockerfile").write_text(f"FROM {image}\n")

    solve_sh = dest_dir / "solution" / "solve.sh"
    solve_sh.write_text("#!/bin/bash\nexit 0\n")
    solve_sh.chmod(0o755)

    test_sh = dest_dir / "tests" / "test.sh"
    test_sh.write_text('#!/bin/bash\n' 'echo 1 > /logs/verifier/reward.txt\n' 'exit 0\n')
    test_sh.chmod(0o755)

    return dest_dir


def parse_task_arg(task_str):
    """Parse --task value into single-task or suite mode.

    Returns one of:
        ("single", task_ref)                  -- e.g. "terminal-bench/compile-compcert"
        ("suite", {"dataset": ..., "tasks": [...]})  -- comma-separated tasks
    """
    if "," not in task_str:
        return "single", task_str

    parts = [p.strip() for p in task_str.split(",") if p.strip()]
    if not parts:
        return "single", task_str

    # First entry must have dataset/ prefix
    if "/" not in parts[0]:
        print("ERROR: First task in suite must include dataset prefix (e.g. terminal-bench/task-name)")
        print(f"  Got: {parts[0]}")
        sys.exit(1)

    dataset, first_task = parts[0].split("/", 1)
    task_names = [first_task]

    for p in parts[1:]:
        if "/" in p:
            d, t = p.split("/", 1)
            if d != dataset:
                print("ERROR: All tasks must be from the same dataset.")
                print(f"  Expected dataset: {dataset}")
                print(f"  Got: {d}/{t}")
                sys.exit(1)
            task_names.append(t)
        else:
            task_names.append(p)

    # Deduplicate tasks while preserving order (issue #21)
    seen = set()
    unique_tasks = []
    for t in task_names:
        if t not in seen:
            seen.add(t)
            unique_tasks.append(t)
    task_names = unique_tasks

    return "suite", {"dataset": dataset, "tasks": task_names}


def _aggregate_splits(splits):
    """Aggregate multiple repeat time-split dicts into median/min/max summary."""

    def _stat(key):
        vals = [s[key] for s in splits if key in s and s[key] is not None]
        if not vals:
            return None
        return {
            "median": round(statistics.median(vals), 2),
            "min": round(min(vals), 2),
            "max": round(max(vals), 2),
            "values": vals,
        }

    total_full = sum(s.get("n_full", 0) for s in splits)
    total_trials = sum(s.get("n_full", 0) + s.get("n_partial", 0) + s.get("n_invalid", 0) for s in splits)

    return {
        "pct_cpu": _stat("pct_cpu"),
        "pct_llm": _stat("pct_llm"),
        "tool_time_sec": _stat("tool_time_sec"),
        "wall_time_sec": _stat("wall_time_sec"),
        "llm_time_sec": _stat("llm_time_sec"),
        "n_llm_calls": _stat("n_llm_calls"),
        "avg_reward": _stat("avg_reward"),
        "avg_env_setup_sec": _stat("avg_env_setup_sec"),
        "avg_agent_setup_sec": _stat("avg_agent_setup_sec"),
        "avg_agent_exec_sec": _stat("avg_agent_exec_sec"),
        "avg_verifier_sec": _stat("avg_verifier_sec"),
        "n_full": total_full,
        "n_partial": sum(s.get("n_partial", 0) for s in splits),
        "n_invalid": sum(s.get("n_invalid", 0) for s in splits),
        "n_verified": sum(s.get("n_verified", 0) for s in splits),
        "n_verifier_errors": sum(s.get("n_verifier_errors", 0) for s in splits),
        # Collect unique error types across all repeats
        "verifier_errors": sorted({e for s in splits for e in s.get("verifier_errors", [])}),
        "survival_rate": f"{total_full}/{total_trials}",
        "n_repeats": len(splits),
        "per_repeat": splits,
    }


def _is_private_url(url: str) -> bool:
    """Return True if *url* points to a loopback or RFC-1918 private address.

    Checks for:
    - localhost / 127.0.0.0/8
    - 10.0.0.0/8, 172.16.0.0/12, 192.168.0.0/16 (RFC 1918)
    - ::1, fe80::/10 (IPv6 loopback / link-local)
    - 169.254.0.0/16 (link-local)
    - 0.0.0.0

    Returns False if the hostname cannot be resolved to an IP (e.g. a
    regular domain name that DNS hasn't resolved yet).
    """
    import socket

    try:
        parsed = urlparse(url)
        hostname = parsed.hostname
        if not hostname:
            return False

        # Resolve hostname to IP address(es)
        try:
            addr_infos = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            # Cannot resolve -- not obviously private
            return False

        for addr_info in addr_infos:
            ip_str = addr_info[4][0]
            try:
                ip = ipaddress.ip_address(ip_str)
                if ip.is_loopback or ip.is_private or ip.is_link_local:
                    return True
            except ValueError:
                continue

        return False
    except Exception:
        return False


def _user_supplied_flags(parser, args, ignore=frozenset()):
    """Return a list of human-readable flag tokens the user passed on the CLI.

    Walks ``parser._actions`` and reports every dest whose current value
    differs from its argparse default (i.e. the user typed it). Used by
    the ``--config`` strict-mode check to refuse mixing yaml with CLI flags.

    For ``action="append"`` lists each value is reported as a separate
    ``--flag value`` entry so the message names every concrete CLI token.
    """
    out = []
    for action in parser._actions:
        dest = action.dest
        if dest in ignore or dest == "help" or not action.option_strings:
            continue
        current = getattr(args, dest, None)
        default = parser.get_default(dest)
        if current == default:
            continue
        flag = next(
            (s for s in action.option_strings if s.startswith("--")),
            action.option_strings[0],
        )
        if isinstance(default, bool):
            out.append(flag)
        elif isinstance(default, list) and isinstance(current, list):
            for v in current:
                out.append(f"{flag} {v}")
        else:
            out.append(f"{flag} {current}")
    return out


def _resolve_api_base(args, parser):
    url = args.api_base or os.environ.get("AGENT_API_BASE")
    if not url:
        if args.agent.startswith("terminus"):
            parser.error("Provide --api-base or set AGENT_API_BASE env var")
        url = ""  # non-terminus agents don't need an API base
    return url


def _run_shutdown(runtime: str, signum: int, docker_owner: str | None = None) -> None:
    """Kill harbor's process group and clean up containers for the runtime.

    Called from the SIGINT/SIGTERM handler installed in main().
    """
    pgid = get_active_harbor_pgid()
    if pgid:
        try:
            os.killpg(pgid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    try:
        from tasktrace_capture.runtimes import runtime_cleanup

        runtime_cleanup(runtime, docker_owner)
    except Exception:
        pass
    sys.exit(128 + signum)


def _environment_label_for_runtime(runtime: str) -> str:
    """Return the 'Environment' row string for the report config panel."""
    if runtime == "enroot":
        return "EnrootEnvironment (enroot_env.py)"
    if runtime == "docker":
        return "TasktraceCaptureDockerEnv (docker_env.py)"
    if runtime == "apptainer":
        return "ApptainerEnvironment (apptainer_env.py)"
    if runtime == "opensandbox":
        return "OpenSandboxEnvironment (opensandbox_env.py)"
    if runtime == "current":
        return "CurrentEnvironment (current_env.py)"
    return runtime  # future-proof fallback


def current_runtime_concurrency_error(runtime: str, concurrency_levels) -> str | None:
    """The refusal for `--runtime current` with any concurrency level other than 1, else None. Pure."""
    if runtime != "current":
        return None
    bad = [n for n in (concurrency_levels or [1]) if n != 1]
    if not bad:
        return None
    return (
        "--runtime current runs trials in THIS container, one at a time (/logs/agent is a single path); "
        f"use --n-concurrent 1 / --concurrency 1 (got {','.join(str(n) for n in concurrency_levels)})"
    )


def _validate_output_dir_whitespace(output_dir: str, runtime: str) -> bool:
    """True if output_dir is usable for this runtime.

    Enroot's mount parser has no quoting mechanism, so whitespace in paths
    breaks it. Docker/compose handle whitespace correctly.
    """
    if runtime != "enroot":
        return True
    return not any(c.isspace() for c in output_dir)


# Runtimes with no local host for the node guards (Slurm allocation, one coordinator per node, foreign
# container processes) to protect.
_REMOTE_OR_IN_PLACE_RUNTIMES = frozenset({"opensandbox", "current"})


def _preflight(
    api_base: str,
    quiet: bool = False,
    agent: str = "terminus-2",
    allow_private_api: bool = False,
    runtime: str = "enroot",
    allow_parallel_docker: bool = False,
) -> None:
    """Run preflight checks: runtime-specific, harbor CLI, API endpoint reachable."""
    import shutil as _shutil
    import urllib.error
    import urllib.request

    from rich.console import Console

    console = Console(stderr=True)
    if not quiet:
        console.print("\n[bold]Pre-flight checks[/]")

    results = []

    def _check(label, ok, fix=None):
        if not quiet:
            status = "[green]\u2713[/]" if ok else "[red]\u2717[/]"
            console.print(f"  {status} {label}")
            if not ok and fix:
                console.print(f"      {fix}")
        results.append(ok)
        return ok

    # Warn if a node-local runtime is being launched outside a Slurm
    # allocation. OpenSandbox only coordinates remote sandboxes here, so the
    # local process is not consuming a compute node.
    if runtime not in _REMOTE_OR_IN_PLACE_RUNTIMES and not os.environ.get("SLURM_JOB_ID"):
        if not quiet:
            console.print("  [yellow]![/] [yellow]No SLURM_JOB_ID detected — are you on a login node?[/]")
            console.print("      TaskTrace Capture spawns containers and monitors /proc — use a compute node.")
            console.print("      Example: srun --partition=<part> --nodes=1 --time=4:00:00 --pty bash")

    # Node-local runtimes share containers and /proc metrics, so concurrent
    # TaskTrace Capture coordinators would race and pollute measurements. OpenSandbox
    # workers create isolated remote sandboxes and may safely share a service
    # host; applying this node-local guard would prevent remote fan-out.
    # Fail if another tasktrace-capture process is already running on this node.
    # Concurrent instances cause container races and polluted metrics.
    # Exclude our own process tree (current PID + parent uv process).
    if runtime not in _REMOTE_OR_IN_PLACE_RUNTIMES and not allow_parallel_docker:
        import subprocess as _sp

        try:
            my_pid = os.getpid()
            my_ppid = os.getppid()
            out = _sp.run(
                ["pgrep", "-f", "bin/tasktrace-capture"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            other_pids = [int(p) for p in out.stdout.strip().split() if p and int(p) != my_pid and int(p) != my_ppid]
            _check(
                "No other tasktrace-capture process on this node",
                len(other_pids) == 0,
                "Concurrent tasktrace-capture instances cause container races and inaccurate metrics.\n"
                "      Wait for the other process to finish or use a different node.",
            )
        except Exception:
            pass

    # List container processes in non-host mount namespaces (issue #43).
    # Could be orphans from a previous tasktrace-capture run OR legitimate services
    # (vLLM server, other enroot containers). We can't tell — just list them.
    from .monitor import _iter_container_pids

    container_procs = [] if runtime in _REMOTE_OR_IN_PLACE_RUNTIMES else list(_iter_container_pids(skip_idle=False))
    if container_procs and not quiet:
        pids = " ".join(str(p) for p, _ in container_procs)
        console.print(f"  [yellow]![/] [yellow]{len(container_procs)} process(es) in " f"non-host mount namespace[/]")
        for pid, cmd in container_procs[:5]:
            console.print(f"      PID {pid}: {cmd[:80]}")
        if len(container_procs) > 5:
            console.print(f"      ... and {len(container_procs) - 5} more")
        console.print("      If orphans from a previous run, they will pollute CPU metrics.")
        console.print(f"      To kill: [bold]kill -9 {pids}[/]")

    from tasktrace_capture.runtimes import runtime_preflight_checks

    for label, ok, fix in runtime_preflight_checks(runtime):
        _check(label, ok, fix)
    _check("harbor CLI on PATH", _shutil.which("harbor") is not None, "Run `uv sync` first")
    # Warn if /home is nearly full — enroot/harbor default to ~/.cache which
    # will corrupt sqsh imports on tiny NFS /home mounts.
    try:
        _home_stat = os.statvfs(Path.home())
        _home_free_gb = (_home_stat.f_bavail * _home_stat.f_frsize) / (1024**3)
        if _home_free_gb < 2.0 and not quiet:
            console.print(f"  [yellow]![/] [yellow]/home has only {_home_free_gb:.1f}GB free[/]")
            console.print("      Enroot cache redirected to .enroot/ in working directory")
    except OSError:
        pass

    # Check github.com connectivity — verifiers download uv/pytest at runtime.
    # Flaky github access causes silent verifier failures (reward=0) that look
    # like agent failures.  Issue #53.
    try:
        urllib.request.urlopen("https://github.com", timeout=5)
        github_ok = True
    except Exception:
        github_ok = False
    if not github_ok and not quiet:
        console.print("  [yellow]![/] [yellow]github.com unreachable (5s timeout)[/]")
        console.print("      Verifiers download uv/pytest from github.com at runtime.")
        console.print("      Flaky connectivity will cause false reward=0 failures.")
        console.print("      Run: curl --connect-timeout 5 https://github.com")

    # SSRF check: warn when --api-base points to private/loopback addresses
    # (Issue #24). This is a warning, not a hard block -- use --allow-private-api
    # to suppress it.
    if api_base and not allow_private_api and _is_private_url(api_base):
        if not quiet:
            console.print("  [yellow]![/] [yellow]--api-base points to a private/loopback address[/]")
            console.print(f"      URL: {api_base}")
            console.print("      This may be an SSRF risk if the URL is user-controlled.")
            console.print("      Use --allow-private-api to suppress this warning.")

    if api_base:
        try:
            url = api_base.rstrip("/").replace("/v1", "") + "/v1/models"
            urllib.request.urlopen(url, timeout=5)
            endpoint_ok = True
        except Exception:
            endpoint_ok = False
        _check(
            f"API endpoint reachable at {api_base}",
            endpoint_ok,
            "Check that your inference server is running and accessible",
        )

    if not quiet:
        console.print()

    if not all(results):
        raise SystemExit(1)


def _fmt_or_na(v):
    """Format a float as 'X.Xs' or 'n/a' if None."""
    return f"{v:.1f}s" if v is not None else "n/a"


def _split_val(v, default=None):
    """Extract a scalar from a time-split field.

    In single-run mode, fields are plain numbers.
    In repeat mode, _aggregate_splits wraps them as {median, min, max, values};
    we return the median.
    """
    if v is None:
        return default
    if isinstance(v, dict):
        return v.get("median", default)
    return v


def _write_report_json(data, out_dir):
    """Write report.json — clean per-concurrency summary without raw samples."""
    time_splits = data.get("time_splits", {})

    results_by_n = {}
    for r in data.get("results", []):
        for entry in (r if isinstance(r, list) else [r]):
            n = entry.get("n_concurrent")
            if n is not None and n not in results_by_n:
                results_by_n[n] = entry

    levels = {}
    for n_key in sorted(time_splits, key=lambda x: int(x)):
        split = time_splits[n_key]
        res = results_by_n.get(int(n_key), {})
        p95 = res.get("p95_cores_active")
        n_calls_raw = _split_val(split.get("n_llm_calls"))
        llm_sec = _split_val(split.get("llm_time_sec"), 0)
        # When both n_llm_calls and llm_msec are missing/zero, timing data
        # is unavailable — report None instead of fake Calls:1 (issue #25).
        has_llm_data = bool(n_calls_raw) or bool(llm_sec)
        n_calls = n_calls_raw or (1 if has_llm_data else None)
        levels[n_key] = {
            "timing": {
                "pct_tool": _split_val(split.get("pct_cpu")),
                "pct_llm": _split_val(split.get("pct_llm")),
                "avg_wall_sec": _split_val(split.get("wall_time_sec")),
                "avg_llm_sec": llm_sec if has_llm_data else None,
                "avg_tool_sec": _split_val(split.get("tool_time_sec")),
                "avg_llm_calls": n_calls,
                "avg_llm_latency_sec": round(llm_sec / n_calls, 2) if n_calls else None,
            },
            "phases": {
                "avg_env_setup_sec": _split_val(split.get("avg_env_setup_sec")),
                "avg_agent_setup_sec": _split_val(split.get("avg_agent_setup_sec")),
                "avg_agent_exec_sec": _split_val(split.get("avg_agent_exec_sec")),
                "avg_verifier_sec": _split_val(split.get("avg_verifier_sec")),
            },
            "cpu": {
                "avg_pct": res.get("avg_cpu"),
                "peak_pct": res.get("peak_cpu"),
                "avg_cores": res.get("avg_cores_active"),
                "p95_cores": p95,
                "peak_cores": res.get("peak_cores_active"),
                "n_cores": res.get("n_cores"),
                "avg_procs": res.get("avg_procs"),
                "peak_procs": res.get("peak_procs"),
            },
            "disk": {
                "avg_read_mb_s": res.get("avg_disk_read_mb_s"),
                "avg_write_mb_s": res.get("avg_disk_write_mb_s"),
                "peak_write_mb_s": res.get("peak_disk_write_mb_s"),
            },
            "reward": {
                "avg": split.get("avg_reward"),
                "survival_rate": split.get("survival_rate"),
                "n_full": split.get("n_full"),
                "n_partial": split.get("n_partial"),
                "n_invalid": split.get("n_invalid"),
                "n_verified": split.get("n_verified"),
                "n_verifier_errors": split.get("n_verifier_errors"),
            },
            "latency": {
                "llm": {
                    "median": _split_val(split.get("llm_latency_median")),
                    "p95": _split_val(split.get("llm_latency_p95")),
                    "p99": _split_val(split.get("llm_latency_p99")),
                    "max": _split_val(split.get("llm_latency_max")),
                    "n_calls": split.get("n_llm_calls_tracked"),
                },
                "tool": {
                    "median": _split_val(split.get("tool_latency_median")),
                    "p95": _split_val(split.get("tool_latency_p95")),
                    "p99": _split_val(split.get("tool_latency_p99")),
                    "max": _split_val(split.get("tool_latency_max")),
                    "n_calls": split.get("n_tool_calls_tracked"),
                },
            },
            "inference": res.get("inference"),
        }

    report = {
        "dataset": data.get("suite_dataset") or data.get("dataset"),
        "include_tasks": data.get("suite_tasks"),
        "model": data.get("model"),
        "api_base": data.get("api_base"),
        "timestamp": data.get("timestamp"),
        "hostname": data.get("hostname"),
        "runtime": data.get("runtime"),
        "enroot_network_policy": data.get("enroot_network_policy"),
        "cpu": data.get("cpu_info"),
        "baseline": data.get("baseline_cpu", {}),
        "system_info": data.get("system_info", {}),
        "concurrency": levels,
    }
    path = out_dir / "report.json"
    with open(path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Saved {path}")


def _write_report(data, out_dir):
    """Write report.txt — human-readable summary across concurrency levels."""
    time_splits = data.get("time_splits", {})
    if not time_splits:
        return

    results_by_n = {}
    for r in data.get("results", []):
        for entry in (r if isinstance(r, list) else [r]):
            n = entry.get("n_concurrent")
            if n is not None and n not in results_by_n:
                results_by_n[n] = entry

    baseline = data.get("baseline_cpu", {})
    cpu_info = data.get("cpu_info", {})
    n_keys = sorted(time_splits, key=lambda x: int(x))

    w = 72
    lines = []
    lines.append("=" * w)
    lines.append("tasktrace-capture report")
    lines.append("=" * w)
    lines.append(f"Dataset   : {data.get('dataset', '')}")
    lines.append(f"Model     : {data.get('model', '')}")
    lines.append(f"API       : {data.get('api_base', '')}")
    lines.append(f"Host      : {data.get('hostname', '')}")
    lines.append(f"Runtime   : {data.get('runtime', 'enroot')}")
    lines.append(f"Timestamp : {data.get('timestamp', '')}")
    lines.append(f"CPU       : {cpu_info.get('model', '')} ({cpu_info.get('cores_logical', '?')} cores)")
    if baseline:
        lines.append(
            f"Baseline  : avg {baseline.get('avg_cpu_pct', '?')}% CPU  "
            f"avg {baseline.get('avg_cores_active', '?'):.0f} cores  "
            f"peak {baseline.get('peak_cores_active', '?')} cores"
        )

    # System info section
    sysinfo = data.get("system_info", {})
    if sysinfo:
        lines.append("")
        lines.append("-" * w)
        lines.append("System Info")
        lines.append("-" * w)
        si_cpu = sysinfo.get("cpu")
        if si_cpu:
            sockets = si_cpu.get("sockets", "?")
            cores = si_cpu.get("cores_per_socket", "?")
            freq = f"  freq {si_cpu['current_mhz_avg']:.0f} MHz" if si_cpu.get("current_mhz_avg") else ""
            cache = f"  L3 {si_cpu['l3_cache']}" if si_cpu.get("l3_cache") else ""
            lines.append(f"  CPU     : {si_cpu.get('model', '?')}  ({sockets}S x {cores}C){freq}{cache}")
        si_gpu = sysinfo.get("gpu")
        if si_gpu:
            gpu_list = si_gpu.get("summary", si_gpu) if isinstance(si_gpu, dict) else si_gpu
            if gpu_list:
                g = gpu_list[0]
                lines.append(
                    f"  GPU     : {len(gpu_list)}x {g['name']}  vBIOS {g.get('vbios', '?')}  driver {g.get('driver', '?')}  {g.get('memory_mb', '?')} MB"
                )
        si_mem = sysinfo.get("memory")
        if si_mem:
            lines.append(f"  Memory  : {si_mem.get('total_gb', '?')} GB total")
        si_os = sysinfo.get("os")
        if si_os:
            lines.append(f"  OS      : {si_os.get('distro', '?')}  kernel {si_os.get('kernel', '?')}")
        si_slurm = sysinfo.get("slurm")
        if si_slurm:
            lines.append(
                f"  SLURM   : cluster {si_slurm.get('cluster', '?')}  job {si_slurm.get('job_id', '?')}  partition {si_slurm.get('partition', '?')}"
            )
        si_virt = sysinfo.get("virtualization")
        if si_virt:
            lines.append(f"  Virt    : {si_virt}")
        si_ib = sysinfo.get("infiniband")
        if si_ib:
            lines.append(f"  IB      : {len(si_ib)} device(s)  {si_ib[0].get('hca_id', '?')}")
        si_ver = sysinfo.get("versions")
        if si_ver:
            parts = [f"{k} {v}" for k, v in si_ver.items()]
            lines.append(f"  Versions: {', '.join(parts)}")

    lines.append("")

    # ── Time split ──────────────────────────────────────────────────────────
    lines.append("=" * w)
    lines.append("Time Split  (% wall time: tool calls on client vs LLM inference on server)")
    lines.append("=" * w)
    hdr = f"{'Concurrency':>12}  {'Tool%':>6}  {'LLM%':>6}  {'Tool':>7}  {'LLM':>7}  {'Wall':>7}  {'Calls':>6}  {'Avg lat':>7}"
    lines.append(hdr)
    lines.append("-" * w)
    for n_key in n_keys:
        split = time_splits[n_key]
        pct_cpu = _split_val(split.get("pct_cpu"))
        if pct_cpu is None:
            lines.append(
                f"{n_key:>12}  {'n/a':>6}  {'n/a':>6}  {'n/a':>7}  {'n/a':>7}  {'n/a':>7}  {'n/a':>6}  {'n/a':>7}"
            )
        else:
            n_calls_raw = _split_val(split.get("n_llm_calls"))
            tool_sec = _split_val(split.get("tool_time_sec")) or 0
            llm_sec = _split_val(split.get("llm_time_sec")) or 0
            wall_sec = _split_val(split.get("wall_time_sec")) or 0
            # Show dash instead of fake Calls:1 when LLM data missing (issue #25)
            has_llm_data = bool(n_calls_raw) or bool(llm_sec)
            if has_llm_data:
                n_calls = n_calls_raw or 1
                avg_lat = llm_sec / n_calls if n_calls else 0
                calls_str = f"{n_calls:>6.0f}"
                lat_str = f"{avg_lat:>6.2f}s"
            else:
                calls_str = f"{'—':>6}"
                lat_str = f"{'—':>7}"
            lines.append(
                f"{n_key:>12}  "
                f"{pct_cpu:>5.1f}%  "
                f"{_split_val(split.get('pct_llm'), 0):>5.1f}%  "
                f"{tool_sec:>6.1f}s  "
                f"{llm_sec:>6.1f}s  "
                f"{wall_sec:>6.1f}s  "
                f"{calls_str}  "
                f"{lat_str}"
            )
    lines.append("")

    # ── Per-call latency distribution ─────────────────────────────────────
    has_latency = any(_split_val(time_splits[k].get("llm_latency_median")) is not None for k in n_keys)
    if has_latency:
        lines.append("=" * w)
        lines.append("Per-Call Latency  (from session JSONL timestamps)")
        lines.append("=" * w)
        hdr = f"{'Concurrency':>12}  " f"{'':>5}  {'Median':>7}  {'P95':>7}  {'P99':>7}  {'Max':>8}  {'Calls':>6}"
        lines.append(hdr)
        lines.append("-" * w)
        for n_key in n_keys:
            split = time_splits[n_key]
            for kind, prefix in [("llm", "LLM"), ("tool", "Tool")]:
                med = _split_val(split.get(f"{kind}_latency_median"))
                if med is None:
                    continue
                p95 = _split_val(split.get(f"{kind}_latency_p95"), 0)
                p99 = _split_val(split.get(f"{kind}_latency_p99"), 0)
                mx = _split_val(split.get(f"{kind}_latency_max"), 0)
                nc = split.get(f"n_{kind}_calls_tracked", 0) or 0
                lines.append(
                    f"{n_key if kind == 'llm' else '':>12}  "
                    f"{prefix:>5}  "
                    f"{med:>6.2f}s  "
                    f"{p95:>6.2f}s  "
                    f"{p99:>6.2f}s  "
                    f"{mx:>7.2f}s  "
                    f"{nc:>6}"
                )
        lines.append("")

    # ── Phase breakdown ────────────────────────────────────────────────────
    lines.append("=" * w)
    lines.append("Phase Breakdown  (avg seconds per trial)")
    lines.append("=" * w)
    hdr = f"{'Concurrency':>12}  {'Env setup':>9}  {'Agent setup':>11}  {'Execution':>9}  {'Verifier':>8}  {'Total':>7}"
    lines.append(hdr)
    lines.append("-" * w)
    for n_key in n_keys:
        split = time_splits[n_key]
        env = _split_val(split.get("avg_env_setup_sec"))
        ags = _split_val(split.get("avg_agent_setup_sec"))
        exe = _split_val(split.get("avg_agent_exec_sec"))
        ver = _split_val(split.get("avg_verifier_sec"))
        parts = [env, ags, exe, ver]
        total = sum(p for p in parts if p is not None)
        lines.append(
            f"{n_key:>12}  "
            f"{_fmt_or_na(env):>9}  "
            f"{_fmt_or_na(ags):>11}  "
            f"{_fmt_or_na(exe):>9}  "
            f"{_fmt_or_na(ver):>8}  "
            f"{total:>6.1f}s"
        )
    lines.append("")

    # ── CPU usage ───────────────────────────────────────────────────────────
    lines.append("=" * w)
    lines.append("CPU Usage  (during agent execution only)")
    lines.append("=" * w)
    hdr = f"{'Concurrency':>12}  {'Avg CPU%':>8}  {'Peak CPU%':>9}  {'Avg cores':>9}  {'P95 cores':>9}  {'Peak cores':>10}"
    lines.append(hdr)
    lines.append("-" * w)
    for n_key in n_keys:
        res = results_by_n.get(int(n_key), {})
        p95 = res.get("p95_cores_active", 0)
        n_cores = res.get("n_cores", 128)
        lines.append(
            f"{n_key:>12}  "
            f"{res.get('avg_cpu', 0):>7.1f}%  "
            f"{res.get('peak_cpu', 0):>8.1f}%  "
            f"{res.get('avg_cores_active', 0):>9.1f}  "
            f"{p95:>9}  "
            f"{res.get('peak_cores_active', 0):>6}/{n_cores:<4}"
        )
    lines.append("")

    # ── Disk I/O ────────────────────────────────────────────────────────────
    lines.append("=" * w)
    lines.append("Disk I/O  (during benchmark)")
    lines.append("=" * w)
    hdr = f"{'Concurrency':>12}  {'Avg read MB/s':>13}  {'Avg write MB/s':>14}  {'Peak write MB/s':>15}"
    lines.append(hdr)
    lines.append("-" * w)
    for n_key in n_keys:
        res = results_by_n.get(int(n_key), {})
        lines.append(
            f"{n_key:>12}  "
            f"{res.get('avg_disk_read_mb_s', 0):>13.1f}  "
            f"{res.get('avg_disk_write_mb_s', 0):>14.1f}  "
            f"{res.get('peak_disk_write_mb_s', 0):>15.1f}"
        )
    lines.append("")

    # ── Rewards ─────────────────────────────────────────────────────────────
    lines.append("=" * w)
    lines.append("Rewards")
    lines.append("=" * w)
    hdr = f"{'Concurrency':>12}  {'Avg reward':>10}  {'Verified':>10}  {'Completed':>10}"
    lines.append(hdr)
    lines.append("-" * w)
    for n_key in n_keys:
        split = time_splits[n_key]
        avg_r = _split_val(split.get("avg_reward"), None)
        n_verr = split.get("n_verifier_errors", 0)
        verr_types = split.get("verifier_errors", [])
        n_verified = split.get("n_verified", 0)
        # Build the verified denominator from full + partial + invalid
        n_total = split.get("n_full", 0) + split.get("n_partial", 0) + split.get("n_invalid", 0)
        verified_str = f"{n_verified}/{n_total}" if n_total else "?"
        if avg_r is not None:
            # Show error count alongside reward so users know the denominator
            if n_verr:
                reward_str = f"{avg_r:.3f} ({n_verr} err)"
            else:
                reward_str = f"{avg_r:.3f}"
        elif n_verr:
            reward_str = f"n/a ({n_verr} verifier err: {', '.join(verr_types)})"
        else:
            reward_str = "n/a"
        lines.append(
            f"{n_key:>12}  " f"{reward_str:>10}  " f"{verified_str:>10}  " f"{split.get('survival_rate', '?'):>10}"
        )
    lines.append("")

    # ── Inference metrics (optional) ───────────────────────────────────────
    has_inference = any(results_by_n.get(int(k), {}).get("inference") for k in n_keys)
    if has_inference:
        lines.append("=" * w)
        lines.append("Inference Server  (vLLM metrics)")
        lines.append("=" * w)
        hdr = f"{'Concurrency':>12}  {'KV$':>5}  {'Queue':>5}  {'Prompt t/s':>10}  {'Gen t/s':>7}  {'Cache%':>6}  {'TTFT':>6}  {'TPOT':>6}  {'Preempt':>7}"
        lines.append(hdr)
        lines.append("-" * w)
        for n_key in n_keys:
            inf = results_by_n.get(int(n_key), {}).get("inference")
            if not inf:
                lines.append(
                    f"{n_key:>12}  {'n/a':>5}  {'n/a':>5}  {'n/a':>10}  {'n/a':>7}  {'n/a':>6}  {'n/a':>6}  {'n/a':>6}  {'n/a':>7}"
                )
            else:
                kv = inf.get("kv_cache", {})
                q = inf.get("queue", {})
                tp = inf.get("throughput", {})
                lat = inf.get("latency") or {}
                cache_pct = inf.get("prefix_cache_hit_pct")
                lines.append(
                    f"{n_key:>12}  "
                    f"{kv.get('avg_pct', 0):>4.1f}%  "
                    f"{q.get('avg_running', 0) + q.get('avg_waiting', 0):>5.1f}  "
                    f"{tp.get('prompt_tok_per_sec', 0) or 0:>10.0f}  "
                    f"{tp.get('gen_tok_per_sec', 0) or 0:>7.0f}  "
                    f"{cache_pct if cache_pct is not None else 0:>5.1f}%  "
                    f"{_fmt_or_na(lat.get('avg_ttft_sec')):>6}  "
                    f"{_fmt_or_na(lat.get('avg_tpot_sec')):>6}  "
                    f"{inf.get('preemptions', 0):>7}"
                )
        lines.append("")

    path = out_dir / "report.txt"
    path.write_text("\n".join(lines) + "\n")
    print(f"  Saved {path}")


def _fmt_time(sec):
    """Format seconds as mm:ss."""
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def _print_summary_table(data):
    """Print a Rich summary table of all trial results, grouped by concurrency."""
    from rich import box
    from rich.console import Console
    from rich.rule import Rule
    from rich.table import Table

    time_splits = data.get("time_splits", {})
    if not time_splits:
        return

    # Build concurrency → result lookup (first match wins for repeats)
    results_by_n = {}
    for r in data.get("results", []):
        entries = r if isinstance(r, list) else [r]
        for entry in entries:
            n = entry.get("n_concurrent")
            if n is not None and n not in results_by_n:
                results_by_n[n] = entry

    console = Console()
    console.print(f"\n[bold]Run Summary[/bold] — {data.get('dataset', '')}  " f"[dim]{data.get('model', '')}[/dim]")

    for n_key in sorted(time_splits.keys(), key=lambda x: int(x)):
        split = time_splits[n_key]
        result = results_by_n.get(int(n_key), {})

        p95 = result.get("p95_cores_active", "—")
        avg_c = result.get("avg_cores_active", "—")
        peak_c = result.get("peak_cores_active", "—")
        n_total = result.get("n_cores", 128)
        avg_c_str = f"{avg_c:.0f}" if isinstance(avg_c, float) else str(avg_c)

        console.print(
            Rule(
                f"[bold cyan]Concurrency {n_key}[/bold cyan]"
                f"   avg {avg_c_str} cores · p95 {p95} · peak {peak_c}/{n_total}"
            )
        )

        table = Table(show_header=True, show_lines=False, box=box.SIMPLE_HEAVY, padding=(0, 1))
        table.add_column("Trial", justify="center", no_wrap=True)
        table.add_column("Outcome", justify="center")
        table.add_column("Wall", justify="right")
        table.add_column("Turns", justify="right")
        table.add_column("LLM time", justify="right")
        table.add_column("Tool time", justify="right")
        table.add_column("Avg LLM latency", justify="right")

        trials = split.get("trials", [])
        valid = [t for t in trials if t.get("status") in ("full", "partial")]
        if not valid:
            valid = [split]

        for i, t in enumerate(valid):
            reward = t.get("reward") if "reward" in t else split.get("avg_reward")
            wall = t.get("wall_time_sec", split.get("wall_time_sec", 0))
            llm = t.get("llm_time_sec", split.get("llm_time_sec", 0))
            tool = t.get("tool_time_sec", split.get("tool_time_sec", 0))
            n_calls_raw = t.get("n_llm_calls", split.get("n_llm_calls", 0))
            n_eps = t.get("n_episodes", "—")
            # Show dash when LLM timing data is unavailable (issue #25)
            has_llm_data = bool(n_calls_raw) or bool(llm)

            if reward is None:
                outcome = "[yellow]?[/yellow]"
            elif reward >= 1.0:
                outcome = "[green]PASS[/green]"
            elif reward > 0:
                outcome = f"[yellow]{reward:.2f}[/yellow]"
            else:
                outcome = "[red]FAIL[/red]"

            n_calls = (n_calls_raw or 1) if has_llm_data else 0
            table.add_row(
                f"#{i + 1}",
                outcome,
                _fmt_time(wall),
                str(n_eps),
                f"{llm:.1f}s" if has_llm_data else "—",
                f"{tool:.1f}s",
                f"{llm / n_calls:.1f}s" if has_llm_data and n_calls else "—",
            )

        console.print(table)


def _cleanup_containers():
    """Kill processes inside hb__* containers, remove them, then sweep orphans.

    Phase 1: enroot exec kill inside each hb__* container (scoped by name).
    Phase 2: kill_orphaned_container_procs() sweeps orphans that escaped
    removal by comparing mount namespaces against live containers.
    """
    import subprocess as _sp

    from .monitor import kill_orphaned_container_procs

    # Phase 1: remove hb__* containers.
    try:
        r = _sp.run(["enroot", "list"], capture_output=True, text=True, timeout=5)
        if r.returncode != 0:
            return
        for name in r.stdout.strip().split("\n"):
            name = name.strip()
            if name and name.startswith("hb__"):
                _sp.run(["enroot", "remove", "-f", name], capture_output=True, timeout=10)
    except Exception:
        pass

    # Phase 2: kill orphans from removed containers.
    try:
        kill_orphaned_container_procs()
    except Exception:
        pass


def _telemetry_interval(value):
    """argparse type for --node-telemetry-interval: finite and >= 0.5s.

    Anything smaller turns the sampler into a busy loop that contaminates
    the very CPU/disk measurements tasktrace-capture is taking (0 measured at ~3000
    samples/s); non-finite values would break the wait timeout.
    """
    import math

    if isinstance(value, bool):
        raise argparse.ArgumentTypeError(f"--node-telemetry-interval must be a finite value >= 0.5, got {value!r}")
    try:
        f = float(value)
    except (TypeError, ValueError) as err:
        raise argparse.ArgumentTypeError(f"invalid interval: {value!r}") from err
    if not math.isfinite(f) or f < 0.5:
        raise argparse.ArgumentTypeError(f"--node-telemetry-interval must be a finite value >= 0.5, got {value}")
    return f


def main():
    parser = argparse.ArgumentParser(
        description="Harbor + Enroot benchmark sweep with CPU monitoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # hello-world default
  %(prog)s --api-base http://<vllm-host>:8000/v1

  # single registry task
  %(prog)s --task terminal-bench/compile-compcert --concurrency 1,2,4

  # suite: 8 tasks, concurrency is a multiplier (1x=8 agents, 2x=16, 4x=32)
  %(prog)s --task terminal-bench/cobol-modernization,sqlite-with-gcov,qemu-startup,sanitize-git-repo,merge-diff-arc-agi-task,fix-code-vulnerability,code-from-image,headless-terminal --concurrency 1,2,4

  # dataset mode
  %(prog)s -d terminal-bench@2.0 --n-concurrent 64 -k 5

  # package-safe explicit suite
  %(prog)s -d datacurve/deep-swe --include-task datacurve/task-a --include-task datacurve/task-b --concurrency 1

  # dry run: see what will be launched
  %(prog)s --task terminal-bench/qemu-startup --concurrency 1,2,4 --dry-run
""",
    )
    parser.add_argument(
        "--config",
        "-c",
        dest="config",
        default=None,
        help="Path to a tasktrace-capture YAML experiment config. "
        "Exclusive: passing other CLI flags alongside "
        "--config errors out — edit the yaml or drop "
        "--config. (--skip-preflight, --check, --dry-run "
        "are exempt.)",
    )
    parser.add_argument(
        "--dataset", "-d", default=None, help="Harbor dataset name (e.g. hello-world, terminal-bench@2.0)"
    )
    parser.add_argument(
        "--n-tasks",
        "-l",
        type=int,
        default=None,
        help="Maximum tasks to select in dataset mode (forwarded to Harbor -l)",
    )
    parser.add_argument(
        "--include-task",
        dest="include_tasks",
        action="append",
        default=[],
        metavar="TASK",
        help=(
            "Task name to include from --dataset; repeat for multiple tasks. "
            "Unlike --task suites, this unambiguously supports package "
            "datasets and tasks whose names contain '/'."
        ),
    )
    parser.add_argument(
        "--task",
        "-p",
        default=None,
        help="Single task or comma-separated suite (e.g. terminal-bench/task1,task2,task3)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Free-text instruction. Materialises a throwaway Harbor task "
        "directory under the sweep output dir; reward is hardcoded to 1.0.",
    )
    parser.add_argument(
        "--prompt-file", default=None, help="Path to a file containing the instruction (alternative to --prompt)."
    )
    parser.add_argument(
        "--prompt-image",
        default="ubuntu:24.04",
        help="Base image for the ad-hoc task's environment (default: ubuntu:24.04).",
    )
    parser.add_argument(
        "--prompt-name",
        default="tasktrace-capture-adhoc",
        help="Bench name for the ad-hoc task (default: tasktrace-capture-adhoc).",
    )
    parser.add_argument("--model", "-m", default="hosted_vllm/model", help="Model name (default: hosted_vllm/model)")
    parser.add_argument(
        "--api-base",
        default=None,
        help="OpenAI-compatible API base URL "
        "(e.g. http://<vllm-host>:8000/v1, http://localhost:11434/v1). "
        "Env var: AGENT_API_BASE",
    )
    parser.add_argument(
        "--n-concurrent", type=int, default=None, help="Number of concurrent agents (default: same as -k)"
    )
    parser.add_argument(
        "-k", "--trials", type=int, default=None, help="Number of trials per task (default: same as --n-concurrent)"
    )
    parser.add_argument(
        "--sweep",
        nargs="+",
        type=int,
        default=None,
        help="Scaling sweep: run at each concurrency level (e.g. --sweep 1 2 4 8)",
    )
    parser.add_argument(
        "--concurrency",
        default=None,
        help="Comma-separated concurrency levels (e.g. --concurrency 1,2,4). Alias for --sweep",
    )
    parser.add_argument(
        "--agent",
        "-a",
        default="terminus-2",
        help="Harbor agent name (default: terminus-2). "
        "Options: terminus-2, claude-code, codex, aider, swe-agent, etc.",
    )
    parser.add_argument(
        "--agent-kwarg", action="append", default=[], help="Extra agent kwargs passed to harbor (repeatable)"
    )
    parser.add_argument(
        "--agent-env", action="append", default=[], help="Environment variables for the agent (KEY=VALUE, repeatable)"
    )
    parser.add_argument(
        "--api-key",
        action="append",
        default=[],
        help="API key(s) for the inference server (repeatable). "
        "Multiple keys are round-robined across containers to spread "
        "rate limits. Sets ANTHROPIC_API_KEY per container.",
    )
    parser.add_argument(
        "--agent-timeout",
        type=int,
        default=3600,
        help="Timeout budget in seconds (default: 3600). Trials exceeding 90%% "
        "of this are classified as 'partial'. The harbor subprocess is killed "
        "after timeout + 600s.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override output directory (default: <cwd>/jobs/<timestamp>__sweep/). "
        "Path must not contain spaces (enroot limitation).",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Re-run each concurrency level N times as independent batches (default: 1)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be run without actually running anything"
    )
    parser.add_argument("--check", action="store_true", help="Run preflight checks only, then exit 0/1")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip preflight checks")
    parser.add_argument(
        "--allow-parallel-docker",
        action="store_true",
        help="Allow multiple Docker tasktrace-capture experiments on this node",
    )
    parser.add_argument(
        "--allow-private-api",
        action="store_true",
        help="Suppress warning when --api-base points to a private/loopback address",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress preflight output (for scripted/job-array use)"
    )
    parser.add_argument(
        "--runtime",
        choices=["enroot", "docker", "apptainer", "opensandbox", "current"],
        default="enroot",
        help="Container runtime (default: enroot). "
        "Use 'docker' for Harbor Docker or 'apptainer' for "
        "rootless named Apptainer instances. Use "
        "'opensandbox' for remote OpenSandbox sandboxes. Use "
        "'current' when tasktrace-capture already runs INSIDE the task container "
        "(e.g. a Kubernetes Job whose container is the task image): no nested "
        "container is created; requires --n-concurrent 1.",
    )
    parser.add_argument(
        "--enroot-network-policy",
        choices=["strict", "best-effort"],
        default="strict",
        help=(
            "Handling for task network policies that Enroot cannot enforce. "
            "'strict' rejects them (default); 'best-effort' runs with public "
            "host networking and records the degraded isolation."
        ),
    )
    parser.add_argument(
        "--swebench-image-prefix",
        default=None,
        help="Override canonical SWE-bench task images with "
        "<prefix>:<instance-id>. Useful for architecture-specific "
        "registry mirrors with Enroot or Docker; Apptainer "
        "requires --swebench-sif-template instead.",
    )
    parser.add_argument(
        "--swebench-sif-template",
        default=None,
        help="Existing native SWE-bench SIF path template for the "
        "Apptainer runtime, e.g. "
        "'/path/to/images/sweb.eval.arm64.{instance_id}.sif'.",
    )
    parser.add_argument(
        "--override-cpus",
        type=int,
        default=None,
        help="Override the task CPU allocation passed to Harbor; use with "
        "--profile-cpuset-cpus for controlled live/replay comparisons.",
    )
    parser.add_argument(
        "--instrument-shell-commands",
        action="store_true",
        help="Capture individual Bash command wall-clock timings for "
        "instrumented runs (terminus, claude-code, mini-swe-agent).",
    )
    parser.add_argument(
        "--source-image-bashrc",
        action="store_true",
        help="Point BASH_ENV at the task image's /root/.bashrc for every command in the task "
        "container, so non-interactive Bash gets the environment the image sets up for "
        "interactive shells (SWE-bench images activate the `testbed` conda env there). "
        "Recorded in the trace header; the replay sets the same BASH_ENV.",
    )
    parser.add_argument(
        "--instrument-shell-perfetto-depth",
        choices=["1", "max"],
        default="1",
        help="Bash detail rendered in Perfetto for traced runs: "
        "'1' keeps the direct-command lane (default); "
        "'max' renders all nested traced Bash processes.",
    )
    parser.add_argument(
        "--trajectory-timings",
        choices=["keep", "omit"],
        default="keep",
        help="With --instrument-shell-commands: 'keep' (default) writes every command's start, finish and wall "
        "time into the finalized trajectory; 'omit' leaves all measured timings out, for a trajectory to publish "
        "without them. Each action then records the LLM call it followed (after_llm_index), and no Perfetto "
        "timeline is written.",
    )
    parser.add_argument(
        "--profile-live-cpu",
        action="store_true",
        help="Collect Docker cgroup CPU accounting for selected live trial windows.",
    )
    parser.add_argument(
        "--profile-window",
        dest="profile_windows",
        action="append",
        choices=["agent_execution", "container_lifecycle_to_verified"],
        help="Live CPU window to measure (repeat for both windows; " "default: agent_execution).",
    )
    parser.add_argument(
        "--profile-scheduler",
        choices=["off", "required"],
        default="off",
        help="Collect native-perf scheduler transitions and require a usable "
        "collector for exact peak concurrency (Docker; requires an "
        "explicit --profile-cpuset-cpus and one in-flight trial).",
    )
    parser.add_argument(
        "--profile-scheduler-collector-image",
        default=None,
        help="Docker image used as the privileged host-perf wrapper for " "--profile-scheduler required.",
    )
    parser.add_argument(
        "--profile-cpuset-cpus",
        default=None,
        help="Pin the profiled Docker main container and scheduler collector "
        "to a user-selected fixed logical CPU set such as 8-15; "
        "required by --profile-scheduler required.",
    )
    parser.add_argument(
        "--no-start-barrier",
        action="store_true",
        help="Do not synchronize concurrent agent starts; useful for observational traces",
    )
    parser.add_argument(
        "--no-node-telemetry",
        action="store_true",
        help="Disable the raw node telemetry recorder (telemetry.raw: per-CPU "
        "jiffies, PSI, diskstats, storage occupancy, NIC, MemAvailable).",
    )
    parser.add_argument(
        "--node-telemetry-interval",
        type=_telemetry_interval,
        default=None,
        help="Node telemetry sample interval in seconds " "(default: 5; must be a finite value >= 0.5).",
    )
    parser.add_argument(
        "--claude-code-version",
        default=None,
        help="Pin claude-code version (forwarded to harbor as "
        "--agent-kwarg version=<X.Y.Z>). Recommended for "
        "--runtime docker at concurrency >=2: pinning flips "
        "the install path from `curl|bash` native-build to "
        "`npm install` which avoids install-time OOM. "
        "Matches enroot's pin by default.",
    )
    parser.add_argument(
        "--swebench-images",
        default="auto",
        choices=["auto", "epoch-research", "default"],
        help="SWE-bench container image source. "
        "auto: use epoch-research GHCR images on arm64, default on x86_64. "
        "epoch-research: always use ghcr.io/epoch-research/swe-bench.eval.* "
        "(arm64 support for 1819/2294 tasks). "
        "default: use the image specified in the task Dockerfile.",
    )
    args = parser.parse_args()

    # --config is exclusive: yaml is the entire experiment definition.
    # If the user combined -c with any other CLI flag, hard-error before
    # doing any work. This avoids the ambiguity of CLI-vs-yaml precedence
    # (and the agent_env partial-replace foot-gun in particular).
    if args.config:
        from .config_loader import load_yaml_config

        # --skip-preflight, --check, --dry-run are operational/diagnostic
        # overrides, not experiment-config knobs. Yaml mode shouldn't reject
        # them or you can't troubleshoot a yaml file without first removing
        # --skip-preflight just to see the actual yaml-validation error.
        violators = _user_supplied_flags(
            parser,
            args,
            ignore={"config", "skip_preflight", "check", "dry_run"},
        )
        if violators:
            parser.error(
                f"--config: cannot mix yaml with CLI flags. "
                f"Saw: {', '.join(violators)}. Edit the yaml or drop --config."
            )

        try:
            cfg = load_yaml_config(args.config)
        except (FileNotFoundError, KeyError, ValueError, yaml.YAMLError) as e:
            parser.error(f"--config: {e}")

        for key, value in cfg.items():
            if not hasattr(args, key):
                parser.error(f"--config: unknown CLI option '{key}'")
            setattr(args, key, value)

    if args.runtime == "apptainer" and args.swebench_image_prefix:
        parser.error(
            "--runtime apptainer does not convert registry images. Use an "
            "existing SIF with --swebench-sif-template, or use --runtime enroot."
        )
    if args.allow_parallel_docker and args.runtime != "docker":
        parser.error("--allow-parallel-docker requires --runtime docker")
    if args.runtime != "apptainer" and args.swebench_sif_template:
        parser.error("--swebench-sif-template requires --runtime apptainer")
    if args.enroot_network_policy != "strict" and args.runtime != "enroot":
        parser.error("--enroot-network-policy best-effort requires --runtime enroot")

    # --claude-code-version: forward to harbor as --agent-kwarg version=<X.Y.Z>
    # for the claude-code agent. Pins the install version and flips the install
    # path from curl|bash native-build to `npm install` (avoids OOM at c>=2
    # on memory-constrained hosts like WSL docker).
    if args.claude_code_version and args.agent == "claude-code":
        existing_versions = [k for k in args.agent_kwarg if k.startswith("version=")]
        if not existing_versions:
            args.agent_kwarg.append(f"version={args.claude_code_version}")

    # --prompt / --prompt-file: ad-hoc task path. Mutually exclusive with each
    # other and with --task / --dataset.
    if args.prompt is not None and args.prompt_file is not None:
        parser.error("Use either --prompt or --prompt-file, not both.")
    prompt_mode = args.prompt is not None or args.prompt_file is not None
    if prompt_mode and (args.task or args.dataset or args.include_tasks):
        parser.error("--prompt / --prompt-file cannot be combined with --task, " "--dataset, or --include-task.")
    prompt_body = None
    if prompt_mode:
        if args.prompt_file:
            try:
                prompt_body = Path(args.prompt_file).read_text()
            except OSError as e:
                parser.error(f"--prompt-file {args.prompt_file!r} could not be read: {e}")
        else:
            prompt_body = args.prompt
        if not prompt_body.strip():
            parser.error("--prompt / --prompt-file is empty.")

    if args.include_tasks and not args.dataset:
        parser.error("--include-task requires --dataset (-d).")
    if not args.dataset and not args.task and not prompt_mode:
        args.dataset = "hello-world"
    if args.dataset and args.task:
        parser.error("Use either --dataset (-d) or --task (-p), not both.")
    if args.n_tasks is not None and (args.task or prompt_mode):
        parser.error("--n-tasks (-l) can only be used with --dataset (-d).")
    if args.n_tasks is not None and args.include_tasks:
        parser.error("--n-tasks cannot be combined with --include-task.")
    if args.n_tasks is not None and args.n_tasks <= 0:
        parser.error(f"--n-tasks must be a positive integer, got: {args.n_tasks}")
    if args.repeat <= 0:
        parser.error(f"--repeat must be a positive integer, got: {args.repeat}")
    if args.agent_timeout <= 0:
        parser.error(f"--agent-timeout must be a positive integer, got: {args.agent_timeout}")
    if args.override_cpus is not None and args.override_cpus <= 0:
        parser.error(f"--override-cpus must be a positive integer, got: {args.override_cpus}")
    if args.instrument_shell_commands:
        from .shell_capture import supports_shell_capture_agent

        if not supports_shell_capture_agent(args.agent):
            parser.error("--instrument-shell-commands currently supports " "terminus, claude-code, and mini-swe-agent")
    if (
        args.profile_windows or args.profile_scheduler != "off" or args.profile_cpuset_cpus
    ) and not args.profile_live_cpu:
        parser.error("--profile-window, --profile-scheduler, and --profile-cpuset-cpus " "require --profile-live-cpu")
    if args.profile_live_cpu and args.runtime != "docker":
        parser.error("--profile-live-cpu currently supports --runtime docker only")
    from .live_cpu_profile import LiveCpuProfileError, validate_profile_cpuset_cpus

    try:
        args.profile_cpuset_cpus = validate_profile_cpuset_cpus(
            profile_scheduler=args.profile_scheduler,
            requested_cpuset=args.profile_cpuset_cpus,
            override_cpus=args.override_cpus,
        )
    except LiveCpuProfileError as exc:
        parser.error(str(exc))

    # Claude Code agent validation (Issue #28)
    if args.agent == "claude-code":
        env_dict = {}
        for e in args.agent_env:
            if "=" in e:
                k, v = e.split("=", 1)
                env_dict[k] = v
        if "ANTHROPIC_BASE_URL" not in env_dict:
            parser.error(
                '--agent claude-code requires ANTHROPIC_BASE_URL via --agent-env\n'
                '  e.g. --agent-env "ANTHROPIC_BASE_URL=https://<your-inference-endpoint>"'
            )
        # Claude Code can authenticate with its subscription OAuth token.  Do
        # not require an API key in that mode; supplying a placeholder key
        # makes the CLI prefer it over CLAUDE_CODE_OAUTH_TOKEN.
        if "ANTHROPIC_API_KEY" not in env_dict and "CLAUDE_CODE_OAUTH_TOKEN" not in env_dict:
            parser.error(
                '--agent claude-code requires ANTHROPIC_API_KEY or '
                'CLAUDE_CODE_OAUTH_TOKEN via --agent-env\n'
                '  e.g. --agent-env "ANTHROPIC_API_KEY=<your-key>" or '
                '"CLAUDE_CODE_OAUTH_TOKEN=<oauth-token>"'
            )
        base_url = env_dict.get("ANTHROPIC_BASE_URL", "")
        if args.model and "/" in base_url:
            # Remind users to check their model prefix matches the inference endpoint.
            # Some endpoints require a provider prefix (e.g. 'azure/anthropic/') —
            # consult your endpoint's documentation for the correct model string.
            pass

    # Normalize every registry subset to dataset + included task identities.
    # Local task paths remain on the direct task execution path.
    suite_mode = False
    include_tasks = None
    if args.include_tasks:
        include_tasks = {
            "dataset": args.dataset,
            "tasks": list(dict.fromkeys(args.include_tasks)),
        }
        suite_mode = len(include_tasks["tasks"]) > 1
        bench_name = (
            f"suite-{len(include_tasks['tasks'])}tasks"
            if suite_mode
            else include_tasks["tasks"][0].rstrip("/").split("/")[-1]
        )
    elif args.task:
        mode, parsed = parse_task_arg(args.task)
        if mode == "suite":
            include_tasks = parsed
            suite_mode = len(include_tasks["tasks"]) > 1
            bench_name = f"suite-{len(include_tasks['tasks'])}tasks"
        else:
            include_tasks = normalize_registry_task_selection(parsed)
            if include_tasks is not None:
                bench_name = include_tasks["tasks"][0].rstrip("/").split("/")[-1]
            else:
                bench_name = args.task.rstrip("/").split("/")[-1]
    elif prompt_mode:
        bench_name = args.prompt_name
    else:
        bench_name = args.dataset

    if args.concurrency and args.sweep:
        parser.error("--concurrency and --sweep are aliases; use one or the other, not both.")

    if args.concurrency and not args.sweep:
        try:
            args.sweep = [int(x.strip()) for x in args.concurrency.split(",")]
        except ValueError:
            parser.error(f"--concurrency values must be positive integers, got: {args.concurrency}")
        for val in args.sweep:
            if val <= 0:
                parser.error(f"--concurrency values must be positive integers, got: {val}")
        if len(args.sweep) != len(set(args.sweep)):
            parser.error(f"--concurrency contains duplicate values: {args.concurrency}")

    if args.sweep:
        concurrency_levels = args.sweep
    else:
        concurrency_levels = [args.n_concurrent or 1]

    # --runtime current runs one trial at a time in this container: /logs/agent is a single path.
    if error := current_runtime_concurrency_error(args.runtime, concurrency_levels):
        parser.error(error)

    scheduler_profile_sequential = args.profile_scheduler == "required"
    if scheduler_profile_sequential and any(n != 1 for n in concurrency_levels):
        parser.error(
            "--profile-scheduler required runs sequentially; set --concurrency 1. "
            "Multiple selected tasks will run one at a time on the fixed cpuset."
        )

    trials = args.trials

    model_info = {
        "max_input_tokens": 131072,
        "max_output_tokens": 131072,
        "input_cost_per_token": 0.0,
        "output_cost_per_token": 0.0,
    }

    # Handle --check early, before api_base resolution which may error for terminus agents
    if args.check:
        api_base = args.api_base or os.environ.get("AGENT_API_BASE") or ""
        _preflight(
            api_base,
            quiet=args.quiet,
            agent=args.agent,
            allow_private_api=args.allow_private_api,
            runtime=args.runtime,
            allow_parallel_docker=args.allow_parallel_docker,
        )
        sys.exit(0)

    jobs_dir = Path(args.output_dir) if args.output_dir else Path.cwd() / "jobs"

    # Enroot mount parsing cannot handle whitespace in paths (no quoting mechanism).
    # Docker doesn't have this restriction.
    if args.output_dir and not _validate_output_dir_whitespace(args.output_dir, args.runtime):
        parser.error(
            f"--output-dir path must not contain whitespace (enroot limitation).\n"
            f"  Got: {args.output_dir!r}\n"
            f"  Use a path without whitespace or create a symlink."
        )

    cpu_info = get_cpu_info()

    api_base = _resolve_api_base(args, parser)

    # Resolve --swebench-images: 'auto' picks epoch-research on arm64, default elsewhere.
    import platform as _platform

    if args.swebench_images == "auto":
        swebench_images = "epoch-research" if _platform.machine() == "aarch64" else None
    elif args.swebench_images == "epoch-research":
        swebench_images = "epoch-research"
    else:
        swebench_images = None

    _supported_agents = {"terminus", "terminus-1", "terminus-2", "claude-code", "mini-swe-agent"}
    _unsupported_agents = {"openhands", "openhands_sdk"}  # 2-layer sandbox issues

    if args.agent in _unsupported_agents:
        parser.error(
            f"--agent {args.agent} is unsupported in tasktrace-capture. "
            f"OpenHands runs a 2-layer sandbox (trial container → its own "
            f"all-hands-ai/runtime sub-container), ignores the task's "
            f"--prompt-image, requires Docker-in-Docker, and hides tool "
            f"execution from tasktrace-capture's /proc telemetry — the task's WORKDIR "
            f"is dropped on the way through, so writes land at '/' and the "
            f"verifier silently reports reward=0. For SWE-bench-c-style "
            f"workloads, use --agent mini-swe-agent."
        )
    if args.agent not in _supported_agents:
        print(
            f"  WARNING: Agent '{args.agent}' is not validated; "
            f"LLM/tool time-split metrics may be missing.\n"
            f"  Validated agents: {', '.join(sorted(_supported_agents))}\n"
        )

    if not args.skip_preflight:
        _preflight(
            api_base,
            quiet=args.quiet,
            agent=args.agent,
            allow_private_api=args.allow_private_api,
            runtime=args.runtime,
            allow_parallel_docker=args.allow_parallel_docker,
        )

    # ── Rich pre-launch display ──────────────────────────────────
    from rich import box
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Build config key-value table
    cfg = Table(show_header=False, box=None, padding=(0, 2), show_edge=False, expand=True)
    cfg.add_column("key", style="bold cyan", no_wrap=True, ratio=1)
    cfg.add_column("value", ratio=3)

    if include_tasks:
        n_tasks = len(include_tasks["tasks"])
        if suite_mode:
            cfg.add_row("Mode", "[bold yellow]Suite[/] — multiple tasks per sweep step")
        else:
            cfg.add_row("Mode", "Single task")
        cfg.add_row("Dataset", include_tasks["dataset"])
        task_list = "\n".join(f"  • {t}" for t in include_tasks["tasks"])
        cfg.add_row("Tasks", f"{n_tasks} tasks\n{task_list}")
    elif prompt_mode:
        cfg.add_row("Mode", "[bold yellow]Ad-hoc prompt[/]")
        cfg.add_row("Prompt", prompt_body.strip().splitlines()[0][:80])
        cfg.add_row("Image", args.prompt_image)
    elif args.task:
        cfg.add_row("Mode", "Single task")
        cfg.add_row("Task", args.task)
    else:
        cfg.add_row("Mode", "Dataset")
        cfg.add_row("Dataset", args.dataset)
        if args.n_tasks is not None:
            cfg.add_row("Task limit", str(args.n_tasks))

    cfg.add_row("Agent", args.agent)
    cfg.add_row("Model", args.model)
    cfg.add_row("API Base", api_base)
    cfg.add_row("Environment", _environment_label_for_runtime(args.runtime))
    cfg.add_row("Runtime", args.runtime)
    if args.swebench_image_prefix:
        cfg.add_row("SWE-bench images", args.swebench_image_prefix)
    if args.swebench_sif_template:
        cfg.add_row("SWE-bench SIFs", args.swebench_sif_template)
    if args.instrument_shell_commands:
        cfg.add_row("Shell timing", "individual Bash commands")
    if args.source_image_bashrc:
        cfg.add_row("BASH_ENV", "image's /root/.bashrc")
    if args.trajectory_timings == "omit":
        cfg.add_row("Trajectory timings", "omitted")
    if args.profile_live_cpu:
        cfg.add_row("Live CPU windows", ", ".join(args.profile_windows or ["agent_execution"]))
        if scheduler_profile_sequential:
            cfg.add_row(
                "Scheduler peak",
                f"required; sequential on logical CPUs {args.profile_cpuset_cpus}",
            )
        elif args.profile_cpuset_cpus:
            cfg.add_row("Profile CPU set", args.profile_cpuset_cpus)
    if args.no_start_barrier:
        cfg.add_row("Start barrier", "disabled")
    cfg.add_row("Host", os.uname().nodename)
    cfg.add_row("CPU", f"{cpu_info['model']} ({cpu_info['cores_logical']} logical cores)")
    cfg.add_row("Agent timeout", f"{args.agent_timeout}s")
    if args.repeat > 1:
        cfg.add_row("Repeat", f"{args.repeat}x per concurrency level")

    if args.agent_kwarg:
        for i, kw in enumerate(args.agent_kwarg):
            label = "Agent kwargs" if i == 0 else ""
            cfg.add_row(label, kw)

    console.print()
    console.print(
        Panel(
            cfg,
            title="[bold white]Harbor Benchmark Config[/]",
            border_style="bright_blue",
            box=box.ROUNDED,
            padding=(1, 2),
        )
    )

    # Build sweep plan table
    plan = Table(
        title="Experiment Plan",
        box=box.SIMPLE_HEAVY,
        header_style="bold magenta",
        title_style="bold white",
        show_lines=False,
    )

    if include_tasks and suite_mode:
        plan.add_column("Multiplier", justify="center")
        plan.add_column("Tasks", justify="center")
        if scheduler_profile_sequential:
            plan.add_column("Concurrent trials", justify="center", style="bold green")
            plan.add_column("Notes")
        else:
            plan.add_column("Copies each", justify="center")
            plan.add_column("Total agents", justify="center", style="bold green")
        for n in concurrency_levels:
            n_t = len(include_tasks["tasks"])
            if scheduler_profile_sequential:
                plan.add_row(f"{n}x", str(n_t), "1", "fixed cpuset; tasks queued sequentially")
            else:
                total = n * n_t
                plan.add_row(f"{n}x", str(n_t), str(n), str(total))
    elif args.task or include_tasks or prompt_mode:
        plan.add_column("Concurrent agents", justify="center", style="bold green")
        plan.add_column("Runs/agent", justify="center")
        if args.repeat > 1:
            plan.add_column("Repeats", justify="center")
        plan.add_column("Notes", style="dim")
        for n in concurrency_levels:
            k = _compute_trials(trials, n)
            runs_per_agent = k // n if n else k
            note = ""
            if trials is not None and trials < n:
                note = f"[yellow]⚠ --trials {trials} < {n}, bumped to {k}[/]"
            row = [str(n), str(runs_per_agent)]
            if args.repeat > 1:
                row.append(str(args.repeat))
            row.append(note)
            plan.add_row(*row)
    else:
        plan.add_column("Concurrent agents", justify="center", style="bold green")
        plan.add_column("Attempts/task", justify="center")
        plan.add_column("Selected tasks", justify="center")
        for n in concurrency_levels:
            k = _compute_trials(trials, n, fill_slots=False)
            plan.add_row(str(n), str(k), str(args.n_tasks or "all"))

    console.print(plan)
    console.print()

    if args.dry_run:
        console.print("[bold yellow]Dry run[/] — exiting without running.\n")
        sys.exit(0)

    timestamp = time.strftime("%Y-%m-%d__%H-%M-%S")
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = jobs_dir / f"{timestamp}__sweep__{bench_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(exist_ok=True)
    print(f"Output: {out_dir}/\n")

    node_telemetry = None
    if not args.no_node_telemetry:
        from .node_telemetry import DEFAULT_INTERVAL_SEC, NodeTelemetryRecorder

        # YAML config bypasses argparse's type converter, so re-validate here
        # (config_loader also type-checks; this guards direct setattr paths).
        interval = args.node_telemetry_interval
        interval = DEFAULT_INTERVAL_SEC if interval is None else _telemetry_interval(interval)
        node_telemetry = NodeTelemetryRecorder(out_dir, interval_sec=interval, runtime=args.runtime).start()
        # stop() on every exit path: explicit call on the normal path below
        # (which then unregisters); atexit covers exceptions and the signal
        # handlers' sys.exit().
        import atexit

        atexit.register(node_telemetry.stop)
        print(f"  Node telemetry: {node_telemetry.path} ({interval:g}s cadence)\n")

    if prompt_mode:
        adhoc_dir = _materialize_adhoc_task(
            prompt_body=prompt_body,
            dest_dir=out_dir / "adhoc_task",
            image=args.prompt_image,
            agent_timeout_sec=args.agent_timeout,
        )
        args.task = str(adhoc_dir)
        print(f"  Ad-hoc task materialized at: {adhoc_dir}\n")

    print("  Collecting system info...")
    system_info = collect_system_info(runtime=args.runtime)
    si_gpu = system_info.get("gpu")
    if si_gpu:
        gpu_list = si_gpu.get("summary", si_gpu) if isinstance(si_gpu, dict) else si_gpu
        if gpu_list:
            print(f"  GPU: {len(gpu_list)}x {gpu_list[0]['name']}  driver {gpu_list[0].get('driver', '?')}")
    si_slurm = system_info.get("slurm")
    if si_slurm:
        print(f"  SLURM: cluster {si_slurm.get('cluster', '?')}  job {si_slurm.get('job_id', '?')}")
    print()

    print("  Measuring baseline CPU (5s)...")
    from .monitor import sample_baseline

    baseline_cpu = sample_baseline(duration_sec=5)
    if baseline_cpu:
        print(
            f"  Baseline: avg {baseline_cpu['avg_cpu_pct']}% CPU  "
            f"avg {baseline_cpu['avg_cores_active']:.0f} cores  "
            f"peak {baseline_cpu['peak_cores_active']} cores\n"
        )

    # Install signal handler to kill harbor's process group on SIGTERM/SIGINT.
    # Without this, killing tasktrace-capture orphans enroot containers and agent processes
    # that burn 100% CPU indefinitely (see issue #39).
    docker_owner = f"rt{os.urandom(6).hex()}" if args.runtime == "docker" else None
    if docker_owner:
        os.environ["_TASKTRACE_CAPTURE_DOCKER_OWNER"] = docker_owner

    def _shutdown_handler(signum, frame):
        _run_shutdown(args.runtime, signum=signum, docker_owner=docker_owner)

    signal.signal(signal.SIGTERM, _shutdown_handler)
    signal.signal(signal.SIGINT, _shutdown_handler)

    results = []
    time_splits = {}
    n_repeats = args.repeat

    from tasktrace_capture.runtimes import runtime_cleanup

    for n in concurrency_levels:
        # Clean up leftover containers from previous concurrency levels.
        runtime_cleanup(args.runtime, docker_owner)

        repeat_results = []
        repeat_splits = []

        for rep in range(1, n_repeats + 1):
            if n_repeats > 1:
                print(f"\n  ── Repeat {rep}/{n_repeats} for concurrency={n} ──")

            pre_job = find_latest_harbor_job(runs_dir)

            if include_tasks:
                n_tasks = len(include_tasks["tasks"])
                if scheduler_profile_sequential:
                    # Harbor receives the full task selection but only one
                    # worker, so every profiled trial reuses the fixed cpuset
                    # without overlapping another scheduler capture.
                    actual_concurrent = 1
                    actual_trials = 1
                    print(
                        f"\n  Sequential scheduler profile: {n_tasks} tasks, "
                        f"one at a time on CPUs {args.profile_cpuset_cpus}"
                    )
                else:
                    actual_concurrent = n * n_tasks
                    actual_trials = n
                    print(f"\n  Suite multiplier {n}x: {n_tasks} tasks x {n} copies = {actual_concurrent} agents")

                r, harbor_stdout = run_experiment(
                    include_tasks=include_tasks,
                    model=args.model,
                    api_base=api_base,
                    model_info=model_info,
                    n_concurrent=actual_concurrent,
                    trials=actual_trials,
                    agent_timeout=args.agent_timeout,
                    extra_agent_kwargs=args.agent_kwarg,
                    extra_agent_env=args.agent_env,
                    api_keys=args.api_key or None,
                    swebench_image_prefix=args.swebench_image_prefix,
                    swebench_sif_template=args.swebench_sif_template,
                    swebench_images=swebench_images,
                    jobs_dir=runs_dir,
                    agent=args.agent,
                    runtime=args.runtime,
                    enroot_network_policy=args.enroot_network_policy,
                    instrument_shell_commands=args.instrument_shell_commands,
                    instrument_shell_perfetto_depth=args.instrument_shell_perfetto_depth,
                    source_image_bashrc=args.source_image_bashrc,
                    trajectory_timings=args.trajectory_timings,
                    profile_live_cpu=args.profile_live_cpu,
                    profile_windows=args.profile_windows,
                    profile_scheduler=args.profile_scheduler,
                    profile_scheduler_collector_image=args.profile_scheduler_collector_image,
                    profile_cpuset_cpus=args.profile_cpuset_cpus,
                    override_cpus=args.override_cpus,
                    sync_start=not args.no_start_barrier,
                )
                # Store the multiplier as n_concurrent so report lookups
                # (results_by_n[int(n_key)]) match the time_splits key.
                r["n_concurrent"] = n
            else:
                r, harbor_stdout = run_experiment(
                    dataset=args.dataset,
                    task=args.task,
                    n_tasks=args.n_tasks,
                    model=args.model,
                    api_base=api_base,
                    model_info=model_info,
                    n_concurrent=n,
                    trials=trials,
                    agent_timeout=args.agent_timeout,
                    extra_agent_kwargs=args.agent_kwarg,
                    extra_agent_env=args.agent_env,
                    api_keys=args.api_key or None,
                    swebench_image_prefix=args.swebench_image_prefix,
                    swebench_sif_template=args.swebench_sif_template,
                    swebench_images=swebench_images,
                    jobs_dir=runs_dir,
                    agent=args.agent,
                    runtime=args.runtime,
                    enroot_network_policy=args.enroot_network_policy,
                    instrument_shell_commands=args.instrument_shell_commands,
                    instrument_shell_perfetto_depth=args.instrument_shell_perfetto_depth,
                    source_image_bashrc=args.source_image_bashrc,
                    trajectory_timings=args.trajectory_timings,
                    profile_live_cpu=args.profile_live_cpu,
                    profile_windows=args.profile_windows,
                    profile_scheduler=args.profile_scheduler,
                    profile_scheduler_collector_image=args.profile_scheduler_collector_image,
                    profile_cpuset_cpus=args.profile_cpuset_cpus,
                    override_cpus=args.override_cpus,
                    sync_start=not args.no_start_barrier,
                )
            repeat_results.append(r)

            # Check for harbor failure with no trial results
            if isinstance(r, dict) and "error" in r:
                print(f"\n  ERROR: {r['error']}")
                if r.get("harbor_stderr"):
                    print(f"  {r['harbor_stderr'][-200:]}")

            post_job = find_latest_harbor_job(runs_dir)
            harbor_job_dir = post_job if post_job != pre_job else None

            # Warn if suite mode tasks are missing from harbor results (issue #20, #30)
            if include_tasks:
                if harbor_job_dir:
                    found_tasks = get_found_suite_tasks(harbor_job_dir)
                    requested_tasks = set(include_tasks["tasks"])
                    missing = requested_tasks - found_tasks
                    if missing:
                        print(
                            f"\n  WARNING: {len(missing)} requested task(s) missing from results: "
                            f"{', '.join(sorted(missing))}"
                        )
                        print(
                            f"  These tasks may not exist in dataset '{include_tasks['dataset']}'. "
                            f"Harbor silently skips unknown -i filters."
                        )
                else:
                    print(
                        f"\n  WARNING: Harbor produced no job directory — all "
                        f"{len(include_tasks['tasks'])} suite task(s) missing: "
                        f"{', '.join(sorted(include_tasks['tasks']))}"
                    )

            if harbor_job_dir:
                split = parse_agent_time_split(harbor_job_dir, agent_timeout=args.agent_timeout, agent=args.agent)
                if split:
                    repeat_splits.append(split.model_dump())
                    if split.pct_cpu is not None:
                        pct_line = f"  Time split: {split.pct_cpu:.1f}% CPU / {split.pct_llm:.1f}% LLM"
                        if split.tool_time_sec is not None:
                            pct_line += (
                                f"  |  tool={split.tool_time_sec:.1f}s"
                                f"  llm={split.llm_time_sec:.1f}s"
                                f"  calls={split.n_llm_calls:.0f}"
                                f"  survival={split.survival_rate}"
                            )
                        print(pct_line)
                    else:
                        print(f"  Time split: all trials invalid (survival={split.survival_rate})")

        if n_repeats > 1 and repeat_splits:
            agg = _aggregate_splits(repeat_splits)
            time_splits[str(n)] = agg
            tool_stat = agg.get("tool_time_sec")
            if tool_stat:
                print(
                    f"\n  Aggregate ({len(repeat_splits)} repeats): "
                    f"tool={tool_stat['median']:.1f}s (median) "
                    f"[{tool_stat['min']:.1f}-{tool_stat['max']:.1f}]  "
                    f"survival={agg['survival_rate']}"
                )
            else:
                print(
                    f"\n  Aggregate ({len(repeat_splits)} repeats): "
                    f"no valid timing data  survival={agg['survival_rate']}"
                )
        elif repeat_splits:
            time_splits[str(n)] = repeat_splits[0]
        elif str(n) not in time_splits:
            # No time split data (e.g. harbor timed out, no result.json).
            # Insert a placeholder so this concurrency level still appears in the report.
            from .models import TimeSplit

            time_splits[str(n)] = TimeSplit(
                survival_rate="0/0",
            ).model_dump()

        results.append(repeat_results[0] if len(repeat_results) == 1 else repeat_results)

        sweep = SweepData(
            cpu_info=CpuInfo(**cpu_info),
            dataset=bench_name,
            model=args.model,
            api_base=api_base,
            timestamp=timestamp,
            hostname=os.uname().nodename,
            runtime=args.runtime,
            enroot_network_policy=args.enroot_network_policy,
            agent_timeout=args.agent_timeout,
            n_repeats=n_repeats,
            time_splits=time_splits,
            suite_mode=suite_mode,
            results=results,
            baseline_cpu=BaselineCpu(**baseline_cpu) if baseline_cpu else BaselineCpu(),
            system_info=system_info,
            suite_tasks=include_tasks["tasks"] if include_tasks else None,
            suite_dataset=include_tasks["dataset"] if include_tasks else None,
        )
        data = sweep.model_dump()

        json_path = out_dir / "scaling_results.json"
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"  Saved intermediate results to {json_path}")

        _write_report_json(data, out_dir)
        _write_report(data, out_dir)

    if node_telemetry is not None:
        node_telemetry.stop()
        import atexit

        atexit.unregister(node_telemetry.stop)

    # Check if ALL concurrency levels failed (every result has an error key)
    flat_results = []
    for r in results:
        if isinstance(r, list):
            flat_results.extend(r)
        else:
            flat_results.append(r)
    all_errored = flat_results and all(isinstance(r, dict) and "error" in r for r in flat_results)

    if all_errored:
        print(f"\n{'=' * 60}")
        print("  FAILED — all concurrency levels errored")
        for r in flat_results:
            print(f"    concurrency={r.get('n_concurrent')}: {r.get('error')}")
        print(f"{'=' * 60}")
        sys.exit(1)

    print(f"\n{'=' * 60}")
    print("  DONE")
    print(f"  Results: {out_dir}/")
    print("    scaling_results.json  (raw samples)")
    print("    report.json           (clean summary + system info)")
    print("    report.txt            (human-readable results)")
    print("    runs/                 (harbor trial data)")
    if node_telemetry is not None:
        print("    telemetry.raw         (raw node telemetry: CPU/PSI/disk/occupancy)")
    if args.instrument_shell_commands:
        print("      */agent/tasktrace_capture_raw_trajectory.tsv      (raw Bash event log)")
        print("      */agent/tasktrace_capture_replay_trajectory.jsonl (finalized replay trajectory + command timings)")
        print("      */agent/tasktrace_capture_replay_trajectory.perfetto.json (interactive timeline)")
    print(f"{'=' * 60}")

    # Force-exit to avoid SIGBUS from matplotlib/numpy cleanup on Lustre
    # filesystem (memory-mapped files on network FS). All data is already
    # written to disk at this point. os._exit skips flushing stdio, so flush
    # first or piped output (a pod log) is lost.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
