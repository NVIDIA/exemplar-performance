#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

# ruff: noqa

"""observe.py — read-only status / jobs / logs for llmb-k8s.

"What's running / is it hung / tail it / has the published record drifted?" without hand-built kubectl
selectors. Composes the modules that already know the answers (profile_resolver, provenance impact,
lane) with a thin kubectl layer. kubectl is pinned to the profile's KUBE_CONTEXT + NAMESPACE, so it's
safe under concurrent multi-cluster use and never touches global kubectl state.

CLI:
  observe.py status <cell> <profile>              one panel: render/publish/last-run + live server/job
  observe.py jobs   <profile> [--recipe <cell>] [--all]   list Jobs (state · run-id · age)
  observe.py logs   <cell> <profile> [run-id]     resolve the newest (or named) Job and stream it
  observe.py logs   <cell> <profile> [run-id] --artifacts   post-TTL run record from persisted results/ (no cluster)

The pure helpers (last_run, format_jobs, pick_job, status_rows) are unit-tested in selftest.py; only the
kubectl wrappers need a live cluster.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fetch_receipt as _fetch_receipt  # noqa: E402  — the ONE reader of `_fetch_status.json`
import profile_resolver as pr  # noqa: E402


# ──── loaders (pure) ──────────────────────────────────────────────────────────
def envelope(cell) -> dict:
    import yaml

    return (yaml.safe_load((Path(cell) / "recipe.yaml").read_text()) or {}).get("envelope") or {}


def recipe_name(cell) -> str:
    return envelope(cell).get("name", "")


def last_run(cell) -> dict | None:
    """The most recent entry in the cell's append-only runs.jsonl ledger (or None)."""
    f = Path(cell) / "runs.jsonl"
    if not f.exists():
        return None
    lines = [line for line in f.read_text().splitlines() if line.strip()]
    try:
        return json.loads(lines[-1]) if lines else None
    except json.JSONDecodeError:
        return None


def rendered_present(cell) -> bool:
    return any((Path(cell) / "rendered").glob("*.yaml"))


def _read_json(p: Path):
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def artifacts_summary(run_dir: Path) -> str:
    """Pure (B2-4) — a post-Job-TTL 'what happened' view from the PERSISTED local artifacts (run_meta.json +
    per-rung sweep_steps + the fetch receipt), for when the k8s Job is gone and `kubectl logs` returns nothing.
    Raw container logs aren't persisted to the PVC, but the structured run record is — surface that.
    """
    if not run_dir.is_dir():
        return f"artifacts: no local run dir at {run_dir} — fetch it first: scripts/fetch_results.sh {run_dir.name}"
    m = _read_json(run_dir / "run_meta.json") or {}
    out = [f"── run artifacts: {run_dir.name}  ({run_dir}) " + "─" * 8]
    if m:
        wall = m.get("wall_seconds_total")
        wall_h = f"{round(wall / 3600, 1)}h" if isinstance(wall, (int, float)) else "?"
        out.append(
            f"  model {m.get('model_name', '?')} · {m.get('server_runtime', '?')} · "
            f"cluster {m.get('cluster') or m.get('profile') or '?'}"
        )
        out.append(
            f"  started {m.get('started_at_utc') or '?'} · completed {m.get('completed_at_utc') or '?'} "
            f"· wall {wall_h}"
        )
        out.append(f"  sweep: executed {m.get('executed_sequence', '?')} · stop: {m.get('sweep_stop_reason', '?')}")
        if m.get("failure_reason"):  # recovery.py stamps this on a crash/hang
            out.append(f"  ⚠ failure_reason: {m['failure_reason']}")
            if m.get("resume_command"):
                out.append(f"    resume: {m['resume_command']}")
        for s in m.get("sweep_steps") or []:
            c, ex, br = (
                s.get("concurrency", "?"),
                s.get("aiperf_exit"),
                (s.get("breaches") or []),
            )
            tag = "✓ ok" if (ex == 0 and not br) else (f"✗ aiperf exit {ex}" if ex else "⚠ SLA breach")
            out.append(f"    c{c}: {tag}" + (f" · breaches {', '.join(map(str, br))}" if br else ""))
    else:
        out.append("  (no run_meta.json — partial or pre-metadata run)")
    signals = []
    if (run_dir / "metrics_summary.csv").is_file():
        signals.append("metrics_summary.csv ✓ (publishable)")
    fr = _read_json(run_dir / "_fetch_status.json")
    if fr is not None:
        # ONE reader for the receipt: a v1 receipt's `complete: true` is UNVERIFIED, not ✓ — it records
        # Validate extracted files rather than archive entry counts.
        _st, _ = _fetch_receipt.verdict(fr)
        signals.append(
            {
                _fetch_receipt.VERIFIED: "fetch complete ✓",
                _fetch_receipt.UNVERIFIED: "fetch UNVERIFIED ? (receipt lacks evidence fields)",
            }.get(_st, "fetch INCOMPLETE ✗")
        )
    if any(run_dir.glob("smoke_*.json")):
        signals.append("smoke ✓")
    if signals:
        out.append("  artifacts: " + " · ".join(signals))
    out.append("  (raw container logs aren't persisted to the PVC — this is the artifact-based run record)")
    return "\n".join(out)


def pick_job(jobs: list[tuple[str, str]], run_id: str | None, name: str) -> str | None:
    """Choose the run's Job from (jobname, creationTimestamp) pairs: prefer an exact
    <name>-bench-<run-id>, else the newest Job. Pure — mirrors idle_guard.find_job."""
    if run_id:
        for kind in ("bench",):
            cand = f"{name}-{kind}-{run_id}"
            if any(j == cand for j, _ in jobs):
                return cand
    return max(jobs, key=lambda jt: jt[1])[0] if jobs else None


def active_job_names(items: list[dict]) -> list[str]:
    """Names of Jobs with a live pod (`.status.active` truthy). Pure — the API rejects a
    `--field-selector status.active=1` on Jobs, so filtering must happen in code (Kubernetes API constraint).
    """
    return [j.get("metadata", {}).get("name", "?") for j in items if (j.get("status") or {}).get("active")]


def format_jobs(items: list[dict]) -> str:
    """Render `kubectl get jobs -o json` items → aligned table. Pure."""
    if not items:
        return "  (no jobs for this selector)"
    rows = []
    for j in items:
        st, meta = j.get("status", {}) or {}, j.get("metadata", {}) or {}
        state = (
            "active"
            if st.get("active")
            else ("succeeded" if st.get("succeeded") else "failed" if st.get("failed") else "pending")
        )
        rid = (meta.get("labels", {}) or {}).get("llmb.nvidia.com/run-id", "-")
        rows.append((meta.get("name", "?"), state, rid, meta.get("creationTimestamp", "")))
    w0 = max(len(r[0]) for r in rows)
    out = [f"  {'NAME':<{w0}}  {'STATE':<9}  {'RUN-ID':<20}  CREATED"]
    out += [f"  {n:<{w0}}  {s:<9}  {r:<20}  {c}" for n, s, r, c in rows]
    return "\n".join(out)


# ──── kubectl (pinned to the profile; the only cluster-touching part) ─────────
def _prof(profile) -> dict:
    return pr._read_env(pr.profile_env_path(profile))


def _base(prof: dict) -> list[str]:
    cmd = ["kubectl"]
    ctx = (prof.get("KUBE_CONTEXT") or "").strip()
    if ctx:
        cmd += ["--context", ctx]
    ns = (prof.get("NAMESPACE") or "").strip()
    if ns:
        cmd += ["-n", ns]
    return cmd


def _kctl(prof: dict, *args, timeout=20) -> tuple[int, str, str]:
    try:
        p = subprocess.run(
            [*_base(prof), "--request-timeout=15s", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


# ──── verbs ───────────────────────────────────────────────────────────────────
def status_rows(cell, prof, live) -> list[tuple[str, str]]:
    """Build the status panel rows. `live` is a dict of already-fetched cluster facts (or {} offline),
    so the composition is pure + testable; the CLI fills `live` via kubectl."""
    import provenance as prov

    env = envelope(cell)
    rows = [
        (
            "recipe",
            f"{env.get('name', '?')}  ({env.get('scenario', '?')}/{env.get('gpu_type', '?')}, "
            f"status={env.get('status', '?')})",
        )
    ]
    rows.append(
        (
            "rendered",
            "present" if rendered_present(cell) else "MISSING — run scripts/render.sh",
        )
    )
    st, _ = prov.impact_one(Path(cell).resolve())
    rows.append(
        (
            "published",
            {
                "MATCH": "record matches current recipe",
                "DRIFT": "⚠ DRIFT — published record ≠ recipe (see provenance --impact)",
                "UNPUBLISHED": "not published yet",
            }[st],
        )
    )
    lr = last_run(cell)
    rows.append(
        (
            "last run",
            (f"{lr.get('run_id')} · {lr.get('cluster', '?')} · {lr.get('date', '?')}" if lr else "none recorded"),
        )
    )
    if "server" in live:
        rows.append(("server", live["server"]))
    if "job" in live:
        rows.append(("active job", live["job"]))
    return rows


def _server_icon(ready_str: str) -> str:
    """'2/2' → '✅ 2/2 ready', '0/2' → '⚠ 0/2 ready', error → '❌ ...'"""
    if "unreachable" in ready_str or "not deployed" in ready_str:
        return f"❌  {ready_str}"
    try:
        r, _, t = ready_str.partition("/")
        t_val = int(t.replace(" ready", "").strip())
        r_val = int((r or "0").strip())
        if t_val == 0:
            return f"·  {ready_str}"  # scaled to 0 — intentional
        return f"{'✅' if r_val == t_val else '⚠ '}  {ready_str}"
    except ValueError:
        return ready_str


def _job_icon(job_str: str) -> str:
    if job_str in ("none", "unreachable"):
        return f"·  {job_str}"
    return f"🔄  {job_str}"


def cmd_status(cell, profile) -> int:
    prof = _prof(profile)
    nm = recipe_name(cell)
    live = {}

    # Server readiness
    rc, out, _ = _kctl(
        prof,
        "get",
        "deploy",
        f"{nm}-server",
        "-o",
        "jsonpath={.status.readyReplicas}/{.spec.replicas}",
    )
    _ready, _, _total = (out or "").partition("/")
    _server_raw = f"{_ready or '0'}/{_total or '0'} ready" if rc == 0 else "not deployed / unreachable"
    live["server"] = _server_icon(_server_raw)

    # Active bench jobs
    rc, out, _ = _kctl(prof, "get", "jobs", "-l", f"llmb.nvidia.com/recipe={nm}", "-o", "json")
    _active = (", ".join(active_job_names(json.loads(out).get("items", []))) or "none") if rc == 0 else "unreachable"
    live["job"] = _job_icon(_active)

    # Phases log for the most recent active run (local file — no cluster call)
    import time as _time

    results_root = Path(__file__).resolve().parent.parent / "results"
    active_phases = sorted(results_root.glob("*/phases.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    if active_phases and (_time.time() - active_phases[0].stat().st_mtime) < 7200:
        live["timeline"] = str(active_phases[0].parent.name)

    rows = status_rows(cell, prof, live)

    # Render as a box panel
    k_w = max(len(k) for k, _ in rows) + 2
    content = [f"  {k:<{k_w}} {v}" for k, v in rows]
    header = f"  status  {nm or cell}  @  {profile}"
    inner_w = max(len(header), max(len(c) for c in content))
    border = "═" * (inner_w + 1)

    print(f"\n╔{border}╗")
    print(f"║{header:<{inner_w + 1}}║")
    print(f"╠{border}╣")
    for c in content:
        print(f"║{c:<{inner_w + 1}}║")
    if "timeline" in live:
        print(f"╠{border}╣")
        hint = f"  phases log  results/{live['timeline']}/phases.log  →  llmb-k8s watch {live['timeline']}"
        print(f"║{hint:<{inner_w + 1}}║")
    print(f"╚{border}╝")
    return 0


def cmd_jobs(profile, cell=None, all_=False) -> int:
    prof = _prof(profile)
    sel = [] if all_ else (["-l", f"llmb.nvidia.com/recipe={recipe_name(cell)}"] if cell else [])
    rc, out, err = _kctl(prof, "get", "jobs", *sel, "-o", "json")
    if rc != 0:
        print(f"jobs: kubectl failed — {err.strip() or 'cluster unreachable?'}")
        return rc
    print(format_jobs(json.loads(out).get("items", [])))
    return 0


def cmd_logs(cell, profile, run_id=None, tail_all=False, artifacts=False) -> int:
    # --artifacts (B2-4): read the persisted local run record instead of live Job logs — works after Job TTL,
    # when `kubectl logs` returns "no Job found". Runs are keyed by run-id under results/ (not per-cell).
    if artifacts:
        root = Path(__file__).resolve().parent.parent / "results"
        if run_id:
            run_dir = root / run_id
        else:
            dirs = sorted(
                (d for d in root.glob("*") if d.is_dir()),
                key=lambda d: d.stat().st_mtime,
                reverse=True,
            )
            if not dirs:
                print(f"logs --artifacts: no runs under {root}")
                return 1
            run_dir = dirs[0]
        print(artifacts_summary(run_dir))
        return 0
    prof = _prof(profile)
    nm = recipe_name(cell)
    rc, out, err = _kctl(
        prof,
        "get",
        "jobs",
        "-l",
        f"llmb.nvidia.com/recipe={nm}",
        "-o",
        'jsonpath={range .items[*]}{.metadata.name} {.metadata.creationTimestamp}{"\\n"}{end}',
    )
    if rc != 0:
        print(f"logs: kubectl failed — {err.strip() or 'cluster unreachable?'}")
        return rc
    pairs = [(line.split()[0], line.split()[1]) for line in out.splitlines() if len(line.split()) == 2]
    job = pick_job(pairs, run_id, nm)
    if not job:
        print(f"logs: no Job found for recipe={nm}" + (f" run-id={run_id}" if run_id else ""))
        return 1
    # Default to the last 200 lines then follow — a completed sweep's log is thousands of lines; dumping the
    # Default to a bounded log tail; `--tail all` restores the full history.
    tail = "all" if tail_all else "200"
    print(f"logs: streaming job/{job}  (last {tail} lines, then follow; Ctrl-C detaches — the Job keeps running)")
    return subprocess.run([*_base(prof), "logs", f"--tail={tail}", "-f", f"job/{job}"]).returncode


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    verb, rest = argv[0], argv[1:]
    if verb == "status":
        if len(rest) < 2:
            sys.exit("usage: observe.py status <cell> <profile>")
        return cmd_status(rest[0], rest[1])
    if verb == "jobs":
        all_ = "--all" in rest
        cell = None
        pos = []
        i = 0
        while i < len(rest):
            if rest[i] == "--recipe" and i + 1 < len(rest):
                cell = rest[i + 1]
                i += 2
                continue
            if rest[i] == "--all":
                i += 1
                continue
            pos.append(rest[i])
            i += 1
        if not pos:
            sys.exit("usage: observe.py jobs <profile> [--recipe <cell>] [--all]")
        return cmd_jobs(pos[0], cell=cell, all_=all_)
    if verb == "logs":
        tail_all = "--tail-all" in rest
        artifacts = "--artifacts" in rest
        pos = [a for a in rest if a not in ("--tail-all", "--artifacts")]
        if len(pos) < 2:
            sys.exit("usage: observe.py logs <cell> <profile> [run-id] [--tail-all] [--artifacts]")
        return cmd_logs(
            pos[0],
            pos[1],
            pos[2] if len(pos) > 2 else None,
            tail_all=tail_all,
            artifacts=artifacts,
        )
    sys.exit(f"observe.py: unknown verb '{verb}' (status | jobs | logs)")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
