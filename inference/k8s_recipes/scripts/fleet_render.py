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

"""fleet_render.py — PURE renderer for the multi-cluster fleet view (no kubectl, no network).

`fleet.sh` does the cluster discovery + the (auth-robust) `kubectl get ... -o json` reads, drops each
cluster's JSON into a workdir, and hands the whole batch to this module to classify + align + colorize
into one k9s-like pane. Keeping every parse/classify/render decision here (not in bash) makes the hard
part unit-testable offline: classification, GPU sums, ordering, RBAC-degradation, and color-stripping are
asserted in scripts/selftest_fleet.py against canned fixtures.

Contract with fleet.sh — `--meta <tsv>`, one line per CONFIGURED cluster (never omit a cluster):
    name <TAB> context <TAB> namespace <TAB> status <TAB> errfile <TAB> connect_cmd
where status ∈ {OK, AUTH, UNREACH} and connect_cmd is the profile's optional CONNECT_CMD (may be empty →
renderer falls back to `tsh kube login <context>`). For OK clusters, `--workdir` holds:
    <name>.pods.json      get pods -n NS -o json          (ours; always readable)
    <name>.deploys.json   get deploy -n NS -o json        (ours servers)
    <name>.jobs.json      get jobs -n NS -o json          (ours bench/coord)
    <name>.nodes.json     get nodes -o json               (cluster-scoped; may be RBAC-Forbidden → absent)
    <name>.all.json       get pods -A -o json             (cluster-scoped; may be RBAC-Forbidden → absent)

GPU accounting per cluster (graceful degradation):
    OURS      = Σ nvidia.com/gpu over our llmb Running pods            (namespace-scoped — always works)
    TOTAL     = Σ allocatable nvidia.com/gpu over GPU nodes            (needs get nodes — else 'n/a')
    OCCUPIED  = Σ nvidia.com/gpu over ALL Running pods cluster-wide    (needs get pods -A — else 'n/a')
    AVAILABLE = TOTAL − OCCUPIED                                       (only when both known)

Read-only by construction: this file only ever *reads* JSON — it can neither list nor mutate a cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import cache_inventory as CI  # noqa: E402  — the MODEL LEDGER (pure; own module)
from model_cache import (
    parse_profile_env,
)  # noqa: E402  — THE claim rule's profile reader

# ── labels this recipe collection stamps (single source: serving/**/templates + submit.sh) ──────
MANAGED_BY = "app.kubernetes.io/managed-by"
MANAGED_BY_VAL = "llmb-recipe"
L_RECIPE = "llmb.nvidia.com/recipe"
L_CELL = "llmb.nvidia.com/cell"
L_RUN_ID = "llmb.nvidia.com/run-id"
L_SWEEP = "llmb.nvidia.com/sweep-id"
L_COMPONENT = "app.kubernetes.io/component"  # llmb.nvidia.com/component on some objects; k8s std here
L_COMPONENT2 = "llmb.nvidia.com/component"  # model-download Jobs stamp BOTH; check either
L_MODEL_NAME = "llmb.nvidia.com/model-name"  # model-download Job → which model is being cached
COMP_MODEL_DOWNLOAD = "model-download"
# Expected runtime is read from the annotation, container environment, run history, or deadline in that order.
ANN_EXPECTED = "llmb.nvidia.com/expected-runtime-s"
ENV_EXPECTED = "EXPECTED_RUNTIME_SECONDS"

# SWEEP (concurrency-rung) sources, in truth-order. Unlike the variance sweep-id label (submit-only), these
# exist for EVERY launch path (submit AND run.sh/sweep/live), so the SWEEP column is never blank for a
# run whose cell we can resolve:
#   LIVE_RUNGS   — per-launch rung SUBSET (`run.sh --rungs`); the exact rungs THIS run drives.
#   CONCURRENCIES — aiperf fixed-mode rung list (reflects `sweep.sh --rungs` overrides via its sed).
#   SWEEP_MODE_RECIPE=adaptive — no fixed list; the job ramps + binary-searches → shown as 'adaptive'.
#   recipe bench.sweep_concurrency — committed fallback (adaptive-off, env absent on older/other jobs).
ENV_LIVE_RUNGS = "LIVE_RUNGS"
ENV_CONCURRENCIES = "CONCURRENCIES"
ENV_SWEEP_MODE = "SWEEP_MODE_RECIPE"

# Recent-terminal history windows (a run's lifecycle is LOADING→RUNNING→done/FAILED; fleet shows the tail).
DEFAULT_HISTORY_SECS = 43200  # by default, terminal runs that ended within the last 12h are shown as rows
FAILED_HISTORY_SECS = 86400  # --failed widens the window to a day (a death matters longer than a success)

OVER_MEDIAN_PCT = 150  # elapsed ≥ 1.5× expected/median → ⚠ over-median (stall signal; governor kills at 2×)
NEAR_DEADLINE_PCT = 90  # elapsed ≥ 90% of the activeDeadlineSeconds ceiling → ⚠ near-deadline

# ── status vocabulary (drives color) ────────────────────────────────────────────────────────────
RUNNING = "RUNNING"  # server ready+owned, or job active and healthy
LOADING = "LOADING"  # pod up, server not-ready yet (cold model load)
# Kubernetes Pending includes both unscheduled and scheduled-but-starting pods, so the UI distinguishes them.
STARTING = "STARTING"  # scheduled onto a node; containers still coming up (image pull / init) — resolves
UNSCHED = "UNSCHEDULED"  # NO node accepted the pod: it has not started and will not until something changes
STUCK = "STUCK"  # CrashLoopBackOff / image pull / OOM-loop
ORPHAN = "ORPHAN"  # GPU-server replicas>0 whose run is GENUINELY gone: old past grace, no active/recent
# sibling in its run group, no control-plane up (read-only detection — never acted on)
PARKED = "PARKED"  # intentionally-held run: GPU workers 0/0 but control-plane up, or recently created —
# NOT an orphan (over-eager ORPHAN used to flag these)
INFRA = "INFRA"  # control-plane member (etcd/nats/frontend/router) — never a prominent RUNNING row;
# folded into its run group / collapsed to a count
COMPLETE = "COMPLETE"  # terminal Job succeeded
FAILED = "FAILED"  # terminal Job failed
IDLE = "IDLE"  # server scaled to 0 with no held run around it (benign leftover; holds nothing)
QUEUED = "QUEUED"  # waiting for a per-model-cache load slot: NOT loading, NOT stalled — queued behind
# another run's checkpoint load (concurrent loads off one PVC starve each other)

# RUNNING is reserved for a server DRIVEN by a live bench/coord Job (an active benchmark). A bare server or
# control-plane pod up with no driving Job is PARKED/INFRA/ORPHAN, never RUNNING.
LIVE_STATES = {RUNNING, LOADING, STARTING, UNSCHED, STUCK, ORPHAN}
ACTIVE_JOB_STATES = {
    RUNNING,
    STARTING,
    UNSCHED,
    STUCK,
}  # a live "run" (active>0), not terminal
GRACE_SECONDS = 600  # a run group younger than this is "recent" → never ORPHANed (new launch, job may lag)
# How long a pod may sit unscheduled before the row is BADGED rather than merely labelled. The scheduler
# re-evaluates every pending pod on every cluster change, so it does not get slower — past this, nothing in
# the CURRENT cluster state will place the pod. Kept short deliberately: the row already names the state and
# the reason, so the badge escalates emphasis, never correctness (on an autoscaling cluster a node may still
# arrive later, and the row will simply stop being unscheduled).
UNSCHED_WARN_SECS = 120
# Scheduled pods that remain in image pull, ContainerCreating, or volume setup beyond this threshold are flagged.
STARTING_WARN_SECS = 900

_ANSI = {
    RUNNING: "32",
    LOADING: "33",
    STARTING: "33",
    UNSCHED: "31",
    STUCK: "31",
    ORPHAN: "31",
    PARKED: "36",
    INFRA: "2",
    COMPLETE: "2",
    FAILED: "31",
    IDLE: "2",
    QUEUED: "2;36",
}  # queued = waiting for a model-load slot: real, but idle-by-design → quiet cyan

# control-plane component labels / name suffixes (up ⇒ the run is intentionally held, not orphaned)
_CONTROL_COMPONENTS = {
    "control",
    "frontend",
    "governor",
    "router",
    "etcd",
    "nats",
    "ingress",
}
_CONTROL_SUFFIXES = ("-etcd", "-nats", "-frontend", "-router", "-ingress")
_WORKER_SUFFIXES = ("-prefill", "-decode", "-worker", "-workers", "-server")

# Run-lifecycle HELPER Jobs — a live run spawns these ALONGSIDE its ONE benchmark driver Job: the
# run-owner watcher (component=run-owner; lifetime==run, the GC anchor the server's ownerReference points at),
# and coordinator/backstop/netscore sidecars. None is the benchmark. Each must fold into the run (a footer
# helper-count), NEVER become a second top-level ● RUNNING row next to the driver — that duplication is the
# "two independent runs render as four rows" confusion this pane must not create. Treated like the governor cron.
_HELPER_JOB_COMPONENTS = {"run-owner", "coord", "backstop", "netscore"}


class Paint:
    """Colorizer that no-ops when color is off (NO_COLOR / non-tty / piped) — so `watch`/pipe output
    is plain text and diff-stable across refreshes."""

    def __init__(self, enabled: bool):
        self.on = enabled

    def c(self, code: str, s: str) -> str:
        return f"\033[{code}m{s}\033[0m" if self.on else s

    def status(self, s: str) -> str:
        return self.c(_ANSI.get(s, "0"), s)

    def dim(self, s: str) -> str:
        return self.c("2", s)

    def bold(self, s: str) -> str:
        return self.c("1", s)

    def red(self, s: str) -> str:
        return self.c("31", s)

    def green(self, s: str) -> str:
        return self.c("32", s)

    def yellow(self, s: str) -> str:
        return self.c("33", s)

    # ── STRUCTURAL palette (the hierarchy tree) — deliberately DISJOINT from the status hues so structure is
    # never confused with state. Status owns green(32)/red(31)/yellow(33) [+ cyan(36) for PARKED]; structure
    # owns blue / magenta / bold-white / grey. Three visually-distinct levels + receding guides:
    #   blue    → the CLUSTER bar        (bold blue    1;34)
    #   magenta → the NAMESPACE bar      (bold magenta 1;35)
    #   white   → the SECTION headers    (bold white   1;37 · INSTALLED / RUN / SERVER)
    #   grey    → the │ ├─ └─ tree guides (grey 90 · they recede so the content leads)
    def blue(self, s: str) -> str:
        return self.c("1;34", s)

    def magenta(self, s: str) -> str:
        return self.c("1;35", s)

    def white(self, s: str) -> str:
        return self.c("1;37", s)

    def grey(self, s: str) -> str:
        return self.c("90", s)


# ── pure primitives ─────────────────────────────────────────────────────────────────────────────


def friendly_gpu(product: str) -> str:
    """`nvidia.com/gpu.product` / GPU_PRODUCT → the short family name shown in the header.
    'NVIDIA-B200'→'B200', 'NVIDIA-GB200'→'GB200', 'NVIDIA-H100-80GB-HBM3'→'H100', ''→''.
    """
    if not product:
        return ""
    s = product.strip()
    for pre in ("NVIDIA-", "nvidia-", "NVIDIA_"):
        if s.startswith(pre):
            s = s[len(pre) :]
            break
    # collapse a verbose SKU to the family token (B200 / GB200 / GB300 / H100 / H200 / A100 / L40S ...)
    head = s.split("-")[0].split("_")[0]
    return head or s


def _nodes_gpu_product(nodes_path) -> str:
    """Best-effort GPU product from a node's `nvidia.com/gpu.product` label (fallback when the profile has
    no GPU_PRODUCT). Reads the already-fetched nodes JSON; '' if unreadable/absent."""
    try:
        import json as _json

        data = _json.loads(Path(nodes_path).read_text())
    except (OSError, ValueError):
        return ""
    for n in (data or {}).get("items", []):
        p = ((n.get("metadata") or {}).get("labels") or {}).get("nvidia.com/gpu.product")
        if p:
            return p
    return ""


def is_llmb(obj: dict) -> bool:
    """True if this object is one this recipe collection owns (vs a foreign occupant)."""
    labels = (obj.get("metadata") or {}).get("labels") or {}
    return labels.get(MANAGED_BY) == MANAGED_BY_VAL or L_RECIPE in labels


def gpu_request(pod: dict) -> int:
    """GPUs a single pod requests (containers + initContainers) via the device plugin."""
    spec = pod.get("spec") or {}
    return sum(
        int(((c.get("resources") or {}).get("requests") or {}).get("nvidia.com/gpu", 0) or 0)
        for c in ((spec.get("containers") or []) + (spec.get("initContainers") or []))
    )


def _phase(pod: dict) -> str:
    return (pod.get("status") or {}).get("phase", "")


def running_gpu(pods: list) -> int:
    """GPUs *in use* = Σ requests over Running-phase pods. Pending pods aren't scheduled (hold nothing);
    Succeeded/Failed released theirs. A CrashLoopBackOff pod is phase=Running and still occupies its
    device, so it counts — matching what the operator sees on the node."""
    return sum(gpu_request(p) for p in pods if _phase(p) == "Running")


def _parse_cpu(v) -> float:
    """A k8s CPU quantity → cores. '500m'→0.5, '2'→2.0, '4000m'→4.0, ''→0."""
    if not v:
        return 0.0
    s = str(v)
    try:
        return int(s[:-1]) / 1000.0 if s.endswith("m") else float(s)
    except ValueError:
        return 0.0


def cpu_request(pod: dict) -> float:
    """CPU cores a single pod requests (containers + initContainers)."""
    spec = pod.get("spec") or {}
    return sum(
        _parse_cpu(((c.get("resources") or {}).get("requests") or {}).get("cpu"))
        for c in ((spec.get("containers") or []) + (spec.get("initContainers") or []))
    )


def running_cpu(pods: list) -> float:
    """CPU cores in use = Σ requests over Running-phase pods (same rule as running_gpu)."""
    return sum(cpu_request(p) for p in pods if _phase(p) == "Running")


def fmt_cores(c: float) -> str:
    """Cores → terse token, shown only when >0: 4.0→'4c', 0.5→'0.5c', 2.5→'2.5c'."""
    if c <= 0:
        return ""
    return (f"{c:.1f}".rstrip("0").rstrip(".")) + "c"


def _gpus(n) -> str:
    """A GPU count spelled OUT — '1 GPU' / 'N GPUs' — never the confusing bare `Ng` abbreviation. Every column
    and summary that states a GPU count routes through here so 'g' is never a suffix anywhere in the pane.
    """
    n = int(n or 0)
    return f"{n} GPU" if n == 1 else f"{n} GPUs"


def pod_is_stuck(pod: dict) -> bool:
    """Wedged container: crash/image-pull loop, or a repeated OOMKill."""
    for cs in (pod.get("status") or {}).get("containerStatuses") or []:
        w = (cs.get("state") or {}).get("waiting") or {}
        if w.get("reason") in (
            "CrashLoopBackOff",
            "ImagePullBackOff",
            "ErrImagePull",
            "CreateContainerError",
            "CreateContainerConfigError",
        ):
            return True
        term = (cs.get("lastState") or {}).get("terminated") or {}
        if term.get("reason") == "OOMKilled" and int(cs.get("restartCount", 0) or 0) >= 3:
            return True
    return False


def pod_is_unschedulable(pod: dict) -> bool:
    if _phase(pod) != "Pending":
        return False
    for c in (pod.get("status") or {}).get("conditions") or []:
        if c.get("type") == "PodScheduled" and c.get("status") == "False" and c.get("reason") == "Unschedulable":
            return True
    return False


_SCHED_RE = re.compile(r"(\d+/\d+) nodes are available:\s*(.*)", re.S)
_SCHED_NOISE = re.compile(r"\bnode\(s\)\s+")
UNSCHED_CAUSE_CAP = 3  # the top 3 causes explain it; the tail is scheduler bookkeeping


def unschedulable_reason(pod: dict) -> str:
    """PURE. The one-line WHY no node took this pod — '' when it is not unschedulable.

    THE DIAGNOSIS IS ALREADY IN OUR HANDS. `status.conditions[PodScheduled].message` reads
    `0/11 nodes are available: 1 node(s) didn't match pod affinity rules, 10 Insufficient cpu,
    3 Insufficient memory.` — the complete answer, sitting in a pod object fleet has ALREADY fetched.
    Rendering `● PENDING` and dropping it sent the operator to `kubectl describe` for a string the pane
    was holding. Causes are ordered by how many nodes each eliminated, because the biggest number is the
    one to fix (here: cpu — and a profile's WHOLE_NODE_CPU was indeed the cause).

    An UNPARSEABLE or ABSENT message degrades to naming that absence, never to silence: a row that says
    UNSCHEDULED and reports no reason tells the reader the reason is MISSING, not that there is none.
    """
    if not pod_is_unschedulable(pod):
        return ""
    msg = ""
    for c in (pod.get("status") or {}).get("conditions") or []:
        if c.get("type") == "PodScheduled" and c.get("status") == "False":
            msg = str(c.get("message") or "").strip()
            break
    if not msg:
        return "FailedScheduling — the scheduler reported no reason"
    m = _SCHED_RE.search(msg)
    if not m:
        return "FailedScheduling: " + _trunc(" ".join(msg.split()), 90, "head")
    causes = []
    for part in m.group(2).strip().rstrip(".").split(","):
        part = _SCHED_NOISE.sub("", " ".join(part.split())).strip()
        if part:
            lead = re.match(r"(\d+)\s", part)
            causes.append((int(lead.group(1)) if lead else 0, part))
    causes.sort(key=lambda c: -c[0])
    shown = [c[1] for c in causes[:UNSCHED_CAUSE_CAP]]
    more = f" · +{len(causes) - len(shown)} more" if len(causes) > len(shown) else ""
    return f"FailedScheduling: {m.group(1)} nodes — " + " · ".join(shown) + more


def unsched_note(pods: list, now=None) -> tuple:
    """PURE. (reason, seconds-waiting) for the first unschedulable pod in `pods`; ('', None) when none is.
    Age is measured from the pod creation timestamp. `now=None` asks for the reason alone.
    """
    for p in pods or []:
        why = unschedulable_reason(p)
        if why:
            secs = age_seconds((p.get("metadata") or {}).get("creationTimestamp", ""), now) if now is not None else None
            return why, secs
    return "", None


def max_restarts(pods: list) -> int:
    n = 0
    for p in pods:
        for cs in (p.get("status") or {}).get("containerStatuses") or []:
            n = max(n, int(cs.get("restartCount", 0) or 0))
    return n


def _parse_ts(ts: str):
    if not ts:
        return None
    try:
        return datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def age_seconds(start_ts: str, now: datetime):
    """Seconds since start_ts, or None if unparseable."""
    t = _parse_ts(start_ts)
    return None if not t else max(0, int((now - t).total_seconds()))


def human_age(start_ts: str, now: datetime) -> str:
    """`creationTimestamp`/`startTime` → compact age like `1h58m`, `3d4h`, `45s`. '' if unknown."""
    secs = age_seconds(start_ts, now)
    if secs is None:
        return ""
    d, r = divmod(secs, 86400)
    h, r = divmod(r, 3600)
    m, s = divmod(r, 60)
    if d:
        return f"{d}d{h}h"
    if h:
        return f"{h}h{m}m"
    if m:
        return f"{m}m{s}s"
    return f"{s}s"


def human_dur(secs: float) -> str:
    """A duration in seconds → approx human label: `~4.8h`, `~45m`, `~2.1d`, `~30s`."""
    secs = float(secs)
    if secs >= 86400:
        return f"~{secs/86400:.1f}d"
    if secs >= 3600:
        return f"~{secs/3600:.1f}h"
    if secs >= 60:
        return f"~{int(secs//60)}m"
    return f"~{int(secs)}s"


def _parse_duration(s: str):
    """A terse duration → seconds: '6h'→21600, '90m'→5400, '2d'→172800, '45s'/'45'→45. None if unparseable."""
    s = str(s).strip().lower()
    if not s:
        return None
    unit = s[-1]
    mult = {"s": 1, "m": 60, "h": 3600, "d": 86400}.get(unit)
    num = s[:-1] if mult else s
    if mult is None:
        mult = 1
    try:
        return int(float(num) * mult)
    except ValueError:
        return None


def _job_env_expected(job: dict):
    """EXPECTED_RUNTIME_SECONDS the submit path stamps on the bench container env (the runs.jsonl median).
    Returns int seconds >0, or None. This is the real source today (no Job annotation exists yet).
    """
    spec = ((job.get("spec") or {}).get("template") or {}).get("spec") or {}
    for c in spec.get("containers") or []:
        for e in c.get("env") or []:
            if e.get("name") == ENV_EXPECTED:
                try:
                    v = int(float(e.get("value", "0") or 0))
                    return v if v > 0 else None
                except ValueError:
                    return None
    return None


def resolve_expected(job: dict, median_lookup=None):
    """Best expected-runtime for a Job → (seconds:int|None, source:str). Order:
    (a) `llmb.nvidia.com/expected-runtime-s` annotation,
    (b) EXPECTED_RUNTIME_SECONDS bench-container environment,
    (c) median wall_seconds from the cell's runs.jsonl (via median_lookup(cell_name)),
    (d) activeDeadlineSeconds as a CEILING (source='deadline' → rendered as 'max', not a median).
    """
    ann = (job.get("metadata") or {}).get("annotations") or {}
    if ann.get(ANN_EXPECTED):
        try:
            v = int(float(ann[ANN_EXPECTED]))
            if v > 0:
                return v, "median"
        except ValueError:
            pass
    env = _job_env_expected(job)
    if env:
        return env, "median"
    if median_lookup is not None:
        cell = _labels(job).get(L_CELL) or _labels(job).get(L_RECIPE)
        med = median_lookup(cell) if cell else None
        if med and med > 0:
            return int(med), "median"
    dl = int((job.get("spec") or {}).get("activeDeadlineSeconds", 0) or 0)
    if dl > 0:
        return dl, "deadline"
    return None, ""


def _job_env_val(job: dict, name: str):
    """First value of env var `name` on the job's bench/first container ('' if absent)."""
    spec = ((job.get("spec") or {}).get("template") or {}).get("spec") or {}
    for c in spec.get("containers") or []:
        for e in c.get("env") or []:
            if e.get("name") == name:
                return e.get("value", "")
    return ""


def _fmt_rungs(tokens) -> str:
    """A whitespace/comma rung list → the compact `16·32·64` shown in the SWEEP column. Non-int tokens are
    dropped; order preserved; de-duplicated. '' if nothing usable."""
    out, seen = [], set()
    for t in tokens:
        t = str(t).strip().strip(",")
        if not t or t in seen:
            continue
        try:
            int(t)
        except ValueError:
            continue
        seen.add(t)
        out.append(t)
    return "·".join(out)


def resolve_rungs(job: dict, rungs_lookup=None) -> tuple[str, str]:
    """Best concurrency-rung string for a run → (text, source). Truthful for ANY launch path (see the
    ENV_LIVE_RUNGS block above). Order: LIVE_RUNGS env (the exact subset this run drives) → CONCURRENCIES env
    (aiperf fixed list, incl. --rungs override) → SWEEP_MODE_RECIPE=adaptive → committed bench.sweep_concurrency
    → '' (genuinely unknowable → the caller renders a '?' placeholder, never a silent blank).
    """
    live = _fmt_rungs((_job_env_val(job, ENV_LIVE_RUNGS) or "").replace(",", " ").split())
    if live:
        return live, "live"
    conc = _fmt_rungs((_job_env_val(job, ENV_CONCURRENCIES) or "").replace(",", " ").split())
    if conc:
        return conc, "env"
    if _job_env_val(job, ENV_SWEEP_MODE) == "adaptive":
        return "adaptive", "adaptive"
    if rungs_lookup is not None:
        cell = _labels(job).get(L_CELL) or _labels(job).get(L_RECIPE)
        rl = rungs_lookup(cell) if cell else None
        if rl:
            txt = _fmt_rungs(rl)
            if txt:
                return txt, "recipe"
    return "", ""


# ── SWEEP progress (dot bar) ──────────────────────────────────────────────────────────────────────
# A sweep marches through a fixed list of concurrency rungs (`16·32·64·…`). The SWEEP column can show how far
# it has got as a SMALL dot bar — ● done · ◐ current · ○ pending — plus a terse `done/total`. The TOTAL is the
# rung list this renderer already resolves (resolve_rungs). The DONE count is derived ONLY from what the pure,
# kubectl-only renderer already sees — never a heavy PVC/artifact read: a ✓ done Job ran every rung; a Job
# carrying COMPLETED_RUNGS (the resume/progress analog of LIVE_RUNGS) names the rungs already finished. A LIVE
# sweep with no such signal degrades to the plain rung list — we never fabricate progress we cannot observe
# (live per-rung state lives in the control-PVC heartbeat / concurrency_* artifact dirs, which fleet never reads).
ENV_COMPLETED_RUNGS = "COMPLETED_RUNGS"
# LIVE progress annotations the sweep follower (scripts/rung_progress.py, driven by sweep.sh) rewrites on the
# bench Job after EACH rung completes — so the dot-bar advances rung-by-rung WHILE the sweep runs. A pure
# annotation on an object fleet already lists (get jobs): fleet NEVER reads the control-PVC heartbeat or the
# concurrency_* artifact dirs. Absent (follower detached / not a sweep) → clean degrade to the plain rung list.
ANN_COMPLETED_RUNGS = "llmb.nvidia.com/completed-rungs"
ANN_TOTAL_RUNGS = "llmb.nvidia.com/total-rungs"


def resolve_rungs_done(job: dict) -> list:
    """Rungs a sweep has ALREADY finished, from the COMPLETED_RUNGS bench env (order-preserved, de-duped). ''
    /absent → []. PURE — reads only the Job the renderer already fetched (no artifact-dir / PVC read).
    """
    out, seen = [], set()
    for t in (_job_env_val(job, ENV_COMPLETED_RUNGS) or "").replace(",", " ").split():
        t = t.strip()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _ann_int(job: dict, key: str):
    """A non-negative int annotation value, or None (absent / unparseable). PURE."""
    v = ((job.get("metadata") or {}).get("annotations") or {}).get(key)
    try:
        n = int(str(v).strip())
    except (TypeError, ValueError):
        return None
    return n if n >= 0 else None


def resolve_sweep_progress(job: dict, rungs_text: str) -> tuple:
    """(done, total, live) for a run's SWEEP dot-bar, from signals the PURE kubectl renderer already sees — in
    truth order:
      1. the LIVE `llmb.nvidia.com/completed-rungs` (+ `/total-rungs`) annotation the sweep follower rewrites
         after EACH rung → the bar advances rung-by-rung while the sweep runs (live=True even at done=0),
      2. else the COMPLETED_RUNGS bench env (a resume/partial signal),
      3. else no signal → (0, token-count, live=False) → the caller keeps the plain rung list.
    Never reads the control-PVC heartbeat / concurrency_* dirs. PURE."""
    tokens = [t for t in (rungs_text or "").split("·") if t]
    ad = _ann_int(job, ANN_COMPLETED_RUNGS)
    if ad is not None:  # live rung-by-rung annotation → the real-time bar
        at = _ann_int(job, ANN_TOTAL_RUNGS)
        total = at if (at and at > 0) else len(tokens)
        return ad, total, True
    done_list = resolve_rungs_done(job)  # resume/partial env signal
    if done_list:
        return sum(1 for t in tokens if t in set(done_list)), len(tokens), True
    return 0, len(tokens), False


def sweep_dotbar(done: int, total: int, *, running: bool = False) -> str:
    """A SMALL at-a-glance sweep progress bar: ● = completed rung · ◐ = the rung currently running · ○ = pending,
    with a terse `done/total`. e.g. `●●◐○○ 2/5`. A wide sweep (>8 rungs) collapses to a fixed 8-segment
    proportional fill so the column never blows up. '' when total is unknown. PURE."""
    if total <= 0:
        return ""
    done = max(0, min(done, total))
    if total <= 8:
        cur = 1 if (running and done < total) else 0
        bar = "●" * done + "◐" * cur + "○" * (total - done - cur)
    else:
        segs = 8
        fill = min(segs, int(round(done / total * segs)))
        cur = 1 if (running and fill < segs) else 0
        bar = "●" * fill + "◐" * cur + "○" * (segs - fill - cur)
    return f"{bar} {done}/{total}"


def _sweep_field(sweep_text: str, done, total, status: str, *, live: bool = False) -> str:
    """The SWEEP column value for a run: a progress dot-bar when the completed-rung count is KNOWN (a ✓ done
    sweep → every rung; a live `completed-rungs` annotation or a COMPLETED_RUNGS env → that many), else the
    plain rung list (`adaptive`/`?`/the rung tokens) — we never fabricate live progress we can't observe.
    `done`/`total`/`live` come from resolve_sweep_progress. PURE."""
    txt = sweep_text or ""
    if txt in ("", "adaptive", "?"):
        return txt
    tokens = [t for t in txt.split("·") if t]
    tot = total if (total and total > 0) else len(tokens)
    if tot <= 0:
        return txt
    if status == COMPLETE:
        return sweep_dotbar(tot, tot, running=False)  # a settled sweep ran every rung
    if live:  # a real completed-rung signal (annotation / env)
        return sweep_dotbar(max(0, min(done or 0, tot)), tot, running=(status == RUNNING))
    return txt  # live sweep, no progress signal → the rung list


def is_infra_job(job: dict) -> bool:
    """A managed Job that is NOT a benchmark run — the governor observe/reap cron, OR a run-lifecycle helper
    (run-owner watcher / coord/backstop/netscore). These must never be counted as a benchmark
    'done'/'failed' (that conflated a governor cron with a real completed run) NOR shown as a top-level run
    row (that showed a run's own run-owner/coord as a second ● RUNNING row beside its driver).
    """
    comp = _labels(job).get("app.kubernetes.io/component", "")
    if comp in _CONTROL_COMPONENTS or comp in _HELPER_JOB_COMPONENTS or comp.startswith("governor"):
        return True
    name = (job.get("metadata") or {}).get("name", "")
    return name.startswith("llmb-governor") or "-governor-" in name or "-runowner-" in name


def job_finished_ts(job: dict) -> str:
    """When a terminal Job ended: completionTime (success) or the latest Complete/Failed condition transition
    (failure); falls back to startTime/creation. '' if not resolvable. Drives the 'ended Nm ago' history age.
    """
    st = job.get("status") or {}
    if st.get("completionTime"):
        return st["completionTime"]
    times = sorted(
        c.get("lastTransitionTime", "")
        for c in (st.get("conditions") or [])
        if c.get("type") in ("Complete", "Failed") and c.get("lastTransitionTime")
    )
    if times:
        return times[-1]
    return st.get("startTime") or (job.get("metadata") or {}).get("creationTimestamp", "")


# container-wedge waiting reasons that ARE the failure cause (crash/image-pull loops) — same set the STUCK
# classifier keys on, reused here so a ✗ FAILED history row names WHY it died in the operator's own vocabulary.
_FAIL_WAITING = (
    "CrashLoopBackOff",
    "ImagePullBackOff",
    "ErrImagePull",
    "CreateContainerError",
    "CreateContainerConfigError",
)


def job_fail_cause(job: dict, job_pods: list) -> tuple[str, str]:
    """A terse failure cause for a ✗ FAILED run → (short, full), pulled ONLY from live state fleet already
    reads. `short` trails the history row (the glance 'why'); `full` (a fuller condition/terminated message)
    surfaces under --detail. Truth-order, most-actionable first:
       1. a pod's OOMKilled lastState (the classic silent killer — name it explicitly, with the exit code),
       2. a pod's crash/image-pull WAITING reason (CrashLoopBackOff/ImagePullBackOff…),
       3. the Job's own Failed condition reason (DeadlineExceeded/BackoffLimitExceeded…) + message,
       4. else a pod's terminated reason + non-zero exit code (generic `Error exit1`).
    '' when the failure left no readable signal (pods reaped, no condition) → caller still emits the logs hint.
    """
    for p in job_pods:
        for cs in (p.get("status") or {}).get("containerStatuses") or []:
            term = (cs.get("lastState") or {}).get("terminated") or {}
            if term.get("reason") == "OOMKilled":
                code = term.get("exitCode")
                return "OOMKilled", (f"exit {code}" if code is not None else "")
            w = (cs.get("state") or {}).get("waiting") or {}
            if w.get("reason") in _FAIL_WAITING:
                return w["reason"], (w.get("message") or "")
    for c in (job.get("status") or {}).get("conditions") or []:
        if c.get("type") == "Failed" and c.get("status") == "True":
            return (c.get("reason") or "Failed"), (c.get("message") or "")
    for p in job_pods:
        for cs in (p.get("status") or {}).get("containerStatuses") or []:
            term = (cs.get("lastState") or {}).get("terminated") or {}
            reason, code = term.get("reason"), term.get("exitCode")
            if reason or code:
                short = reason or "Error"
                if code not in (None, 0):
                    short = f"{short} exit{code}"
                return short, (term.get("message") or "")
    return "", ""


def _labels(obj: dict) -> dict:
    return (obj.get("metadata") or {}).get("labels") or {}


def _selector_match(pod: dict, match_labels: dict) -> bool:
    labels = _labels(pod)
    return bool(match_labels) and all(labels.get(k) == v for k, v in match_labels.items())


def _job_pods(job_name: str, pods: list) -> list:
    out = []
    for p in pods:
        labels = _labels(p)
        if labels.get("job-name") == job_name or labels.get("batch.kubernetes.io/job-name") == job_name:
            out.append(p)
    return out


# ── classification (pure) ───────────────────────────────────────────────────────────────────────


def _server_recipe(dep: dict) -> str:
    """Recipe a server belongs to: its `llmb.nvidia.com/recipe` label, else the name minus `-server`."""
    labels = _labels(dep)
    if labels.get(L_RECIPE):
        return labels[L_RECIPE]
    name = (dep.get("metadata") or {}).get("name", "")
    return name[: -len("-server")] if name.endswith("-server") else name


def run_group_key(dep: dict) -> str:
    """The run/cell a Deployment belongs to, so a whole run group (a disagg group's etcd/nats/frontend/
    prefill/decode, or an agg server) is evaluated together. Prefer the cell/recipe label; else the shared
    name prefix with a known component suffix stripped."""
    labels = _labels(dep)
    if labels.get(L_CELL):
        return labels[L_CELL]
    if labels.get(L_RECIPE):
        return labels[L_RECIPE]
    name = (dep.get("metadata") or {}).get("name", "")
    for suf in _CONTROL_SUFFIXES + _WORKER_SUFFIXES:
        if name.endswith(suf):
            return name[: -len(suf)]
    return name


def is_control_plane(dep: dict) -> bool:
    """A control-plane member (etcd/nats/frontend/router/governor) — GPU-less, holds the run together."""
    comp = _labels(dep).get("app.kubernetes.io/component", "")
    if comp in _CONTROL_COMPONENTS:
        return True
    name = (dep.get("metadata") or {}).get("name", "")
    return name.endswith(_CONTROL_SUFFIXES)


def classify_server(
    dep: dict,
    dep_pods: list,
    live_recipes: set,
    *,
    group_recent: bool = False,
    group_held: bool = False,
    live_roots: set = frozenset(),
) -> str:
    """Group-aware status for one server Deployment. A ready GPU-server with no active owning Job is only an
    ORPHAN when its run is GENUINELY gone: NOT recently created (past the grace window) and NO control-plane
    sibling is up in its run group. A run whose control-plane is up but whose GPU workers are scaled to 0/0,
    or one that was just created, is PARKED — an intentionally-held run, not an orphan. When in doubt we
    downgrade ORPHAN→PARKED/RUNNING rather than over-alarm. Detection only — fleet never scales or deletes.

    `group_recent` = any member of this run group is younger than GRACE_SECONDS.
    `group_held`   = a control-plane sibling in this run group is ready (the run is being held up).
    `live_roots`   = run-roots (cell/recipe) of the ACTIVE driver Jobs — a per-run server's NAME carries
                     a `-<run-id>-server` suffix so its name-derived recipe won't be in `live_recipes`, but its
                     `llmb.nvidia.com/cell` IS an active root → it's owned (RUNNING), not a false ORPHAN.
    """
    spec = dep.get("spec") or {}
    status = dep.get("status") or {}
    desired = int(spec.get("replicas", 1) or 0)
    ready = int(status.get("readyReplicas", 0) or 0)
    owned = _server_recipe(dep) in live_recipes or bool(_run_root(dep) and _run_root(dep) in live_roots)

    if any(pod_is_stuck(p) for p in dep_pods):
        return STUCK
    if desired == 0:
        # scaled to 0: PARKED if its run is intentionally held (control-plane up / recent / active job),
        # else a benign IDLE leftover.
        return PARKED if (group_held or group_recent or owned) else IDLE
    if ready >= desired:
        if is_control_plane(dep):
            return INFRA  # etcd/nats/frontend up — never a prominent RUNNING row (folded/collapsed)
        if owned:
            return RUNNING  # DRIVEN by an active bench/coord Job → an actual benchmark is running
        if group_held or group_recent:
            return PARKED  # server/run intentionally held (control-plane up) or a fresh launch (grace)
        return ORPHAN  # replicas>0, ready, old, no driving Job, no control-plane → gone
    if any(pod_is_unschedulable(p) for p in dep_pods):
        return UNSCHED  # no node accepted it — this is not "coming up", it is not coming up
    if any(_phase(p) == "Running" for p in dep_pods):
        return LOADING  # pod up, server 0/1 not-ready → cold model load (a run coming up)
    return STARTING  # scheduled (or not yet created) — containers still coming up


def classify_job(job: dict, job_pods: list) -> str:
    status = job.get("status") or {}
    active = int(status.get("active", 0) or 0)
    succeeded = int(status.get("succeeded", 0) or 0)
    failed = int(status.get("failed", 0) or 0)
    if active > 0:
        if any(pod_is_stuck(p) for p in job_pods):
            return STUCK
        if any(pod_is_unschedulable(p) for p in job_pods):
            return UNSCHED  # NEVER folded into STARTING: one resolves itself, the other cannot
        if not any(_phase(p) == "Running" for p in job_pods):
            return STARTING
        return RUNNING
    if succeeded > 0:
        return COMPLETE
    if failed > 0:
        return FAILED
    return STARTING


# ── cluster-wide GPU accounting (pure; graceful degradation) ────────────────────────────────────


def gpu_total(nodes_j):
    """TOTAL allocatable GPUs across GPU nodes. None = unreadable (no cluster-scope RBAC)."""
    if not nodes_j or "items" not in nodes_j:
        return None
    return sum(
        int(((n.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0) or 0) for n in nodes_j["items"]
    )


def gpu_occupied_and_foreign(allpods_j, is_ours=is_llmb, our_namespaces=frozenset()):
    """(OCCUPIED cluster-wide, FOREIGN, UNATTRIBUTED-in-our-namespaces) GPUs over Running pods.
    (None, None, None) = unreadable.

    GPU use in managed namespaces that cannot be tied to a tracked run is
    reported separately from foreign workloads.

    `our_namespaces` = the namespaces we manage (the profile's configured ns + every ns holding our llmb
    workloads). Empty → behaves as before (everything not-ours is foreign)."""
    if not allpods_j or "items" not in allpods_j:
        return None, None, None
    occ = fgn = unattr = 0
    for p in allpods_j["items"]:
        if _phase(p) != "Running":
            continue
        g = gpu_request(p)
        occ += g
        if is_ours(p):
            continue
        ns = (p.get("metadata") or {}).get("namespace", "")
        if ns and ns in our_namespaces:
            unattr += g  # in OUR namespace but unattributable → its own bucket
        else:
            fgn += g
    return occ, fgn, unattr


def node_capacity(nodes_j, allpods_j, is_ours=is_llmb):
    """Per-NODE schedulability — the TRUTHFUL capacity signal. The launchable unit is a WHOLE free node: a
    normal run (tp=4 / tp=8) takes an ENTIRE node's GPUs, so fragmented free GPUs scattered across busy nodes
    are unlaunchable (`4 free` as 2+2 across two busy nodes can't place a tp=4 pod). We aggregate per GPU node
    from the allocatable node list + the pods -A occupancy (nodeName + gpu requests):
        free_on_node = allocatable_gpus(node) − Σ gpu-requests of RUNNING pods on that node
        node is FREE  when nothing GPU-requesting is on it (free_on_node == allocatable)
        biggest_free  = max(free_on_node) over GPU nodes = the biggest single run launchable RIGHT NOW
                        (0g → the pool is fragmented/full → nothing launchable, however many stray GPUs are 'free')
    Returns a dict (or None when the node list is unreadable — no cluster-scope RBAC):
        {total_nodes, free_nodes, biggest_free, ours_nodes}
    A GPU node = allocatable nvidia.com/gpu > 0. free_nodes / biggest_free / ours_nodes are None when pods -A
    is unreadable (we know the nodes exist but not their per-node occupancy). `is_ours` (owner-aware matcher)
    attributes our own GPU pods so `ours_nodes` counts nodes WE hold, not foreign occupants.
    """
    if not nodes_j or "items" not in nodes_j:
        return None
    alloc = {}
    for n in nodes_j["items"]:
        g = int(((n.get("status") or {}).get("allocatable") or {}).get("nvidia.com/gpu", 0) or 0)
        if g > 0:
            alloc[(n.get("metadata") or {}).get("name", "")] = g
    total_nodes = len(alloc)
    if not allpods_j or "items" not in allpods_j:
        return {
            "total_nodes": total_nodes,
            "free_nodes": None,
            "biggest_free": None,
            "ours_nodes": None,
        }
    used = {name: 0 for name in alloc}
    ours_on = set()
    for p in allpods_j["items"]:
        if _phase(p) != "Running":
            continue
        g = gpu_request(p)
        if g <= 0:
            continue
        node = (p.get("spec") or {}).get("nodeName", "")
        if node in used:  # ignore GPU pods on non-GPU / unknown nodes
            used[node] += g
            if is_ours(p):
                ours_on.add(node)
    free_nodes = sum(1 for name in alloc if used[name] == 0)
    biggest_free = max((alloc[name] - used[name] for name in alloc), default=0)
    return {
        "total_nodes": total_nodes,
        "free_nodes": free_nodes,
        "biggest_free": biggest_free,
        "ours_nodes": len(ours_on),
    }


def make_ours_matcher(deploys: list, jobs: list, namespace: str):
    """Owner-aware ownership test. A pod is OURS if its OWN labels say llmb (managed-by / recipe), OR — for a
    pod in OUR namespace — it belongs to one of our llmb Deployments/Jobs by selector, ownerReference, or
    name prefix. This fixes disagg servers whose pods carry only `{app, role, pod-template-hash}` (the
    `managed-by`/`llmb.nvidia.com/*` labels sit on the Deployment, not the pod template) being mis-counted as
    FOREIGN. NOTE: the deeper fix — propagate `managed-by` into the serving pod TEMPLATE — is a separate
    hash-impacting serving-template change, tracked separately; this is the hash-neutral fleet-side attribution.
    Foreign teams live in OTHER namespaces, so unlabeled pods are only attributed to us within our own ns.
    """
    selectors = [((d.get("spec") or {}).get("selector") or {}).get("matchLabels") or {} for d in deploys]
    selectors = [s for s in selectors if s]
    dep_names = [(d.get("metadata") or {}).get("name", "") for d in deploys]
    dep_names = [n for n in dep_names if n]
    job_names = {(j.get("metadata") or {}).get("name", "") for j in jobs}

    def is_ours(pod: dict) -> bool:
        if is_llmb(pod):
            return True
        pns = (pod.get("metadata") or {}).get("namespace", "")
        if namespace and pns and pns != namespace:
            return False  # foreign namespace → not ours
        labels = _labels(pod)
        for sel in selectors:  # matches an llmb Deployment's pod selector
            if all(labels.get(k) == v for k, v in sel.items()):
                return True
        for orf in (pod.get("metadata") or {}).get("ownerReferences") or []:
            k, on = orf.get("kind"), orf.get("name", "")
            if k == "Job" and on in job_names:
                return True
            if k in ("ReplicaSet", "Deployment") and any(on == d or on.startswith(d + "-") for d in dep_names):
                return True
        jn = labels.get("job-name") or labels.get("batch.kubernetes.io/job-name")
        if jn and jn in job_names:
            return True
        pn = (pod.get("metadata") or {}).get("name", "")
        return any(pn == d or pn.startswith(d + "-") for d in dep_names)

    return is_ours


# ── LOADING + CROSS-NAMESPACE run discovery (from `pods -A`) ─────────────────────────────────────────────
# Our runs launch in PER-WORKTREE namespaces (e.g. my-namespace-kvbm-…), NOT only the profile's one
# configured ns. The single-ns nsread misses them, so a cluster shows "0 runs" while `pods -A` plainly holds
# our GPUs. We DISCOVER our runs cluster-wide from the pods -A data ALREADY fetched: group OUR (is_ours) pods
# by (namespace, run-root) and classify each — RUNNING (a live bench/coord pod), LOADING (a server coming up:
# image-pull / ContainerCreating / weights still loading → server not Ready), STUCK (crash-loop), or ORPHAN (a
# ready GPU-holding server with no bench → held, reclaim). This surfaces the "loading in another namespace"
# effort a single-ns view is blind to (GLM-5's 704GB weight reload, a live launch). All PURE + selftested.
_BENCH_HINTS = ("-bench-", "-coord-", "-sweep-")
_SERVER_HINTS = ("-server", "-prefill", "-decode", "-worker", "-frontend", "-router")


def _pod_ready(pod: dict) -> bool:
    """True when a Running pod has all containers Ready (the k8s Ready condition, else all containerStatuses)."""
    if _phase(pod) != "Running":
        return False
    for c in (pod.get("status") or {}).get("conditions") or []:
        if c.get("type") == "Ready":
            return c.get("status") == "True"
    css = (pod.get("status") or {}).get("containerStatuses") or []
    return bool(css) and all(cs.get("ready") for cs in css)


def _pod_coming_up(pod: dict) -> bool:
    """True when a pod is our EFFORT still STARTING: Pending/scheduling, or Running-but-not-Ready (a cold weight
    load), or a container in ContainerCreating / PodInitializing. A crash/image-pull loop is STUCK, not this.
    """
    ph = _phase(pod)
    if ph == "Pending" and not pod_is_stuck(pod):
        return True
    if ph == "Running" and not _pod_ready(pod) and not pod_is_stuck(pod):
        return True
    for cs in (pod.get("status") or {}).get("containerStatuses") or []:
        if ((cs.get("state") or {}).get("waiting") or {}).get("reason") in (
            "ContainerCreating",
            "PodInitializing",
        ):
            return True
    return False


def _pod_role(pod: dict) -> str:
    """bench | server | other — from the component/`role` label (disagg server pods carry `role: prefill|decode`
    but no component), else the pod name / `app` label shape."""
    lab = _labels(pod)
    comp = lab.get(L_COMPONENT2) or lab.get(L_COMPONENT) or lab.get("role") or ""
    if comp in ("bench", "coord"):
        return "bench"
    if comp in ("server", "prefill", "decode", "worker", "frontend", "router"):
        return "server"
    if lab.get("job-name") or lab.get("batch.kubernetes.io/job-name"):
        return "bench"
    hay = (pod.get("metadata") or {}).get("name", "") + " " + lab.get("app", "")
    if any(h in hay for h in _BENCH_HINTS):
        return "bench"
    if any(h in hay for h in _SERVER_HINTS):
        return "server"
    return "other"


def _pod_run_root(pod: dict) -> str:
    """A stable run-root key for grouping a namespace's pods into ONE run: the cell/recipe label, else the pod
    name with a trailing component word + pod-hash/run-id suffix trimmed (…-server-abc123 / …-bench-<rid>-0).
    """
    lab = _labels(pod)
    root = lab.get(L_CELL) or lab.get(L_RECIPE)
    if root:
        return root
    name = (pod.get("metadata") or {}).get("name", "")
    return (
        re.sub(
            r"-(server|prefill|decode|worker|frontend|router|bench|coord|sweep)\b.*$",
            "",
            name,
        )
        or name
    )


def _deploy_index(all_deploys):
    """[{ns, name, selector, cell, control}] for every cluster-wide llmb Deployment (from `deploy -A -l
    managed-by`). Used to attribute a disagg SERVER pod (labels only on its Deployment) to us, name its run
    by the Deployment's cell label, and skip control-plane (etcd/nats/frontend) members. PURE.
    """
    out = []
    for d in all_deploys or []:
        md = d.get("metadata") or {}
        out.append(
            {
                "ns": md.get("namespace", ""),
                "name": md.get("name", ""),
                "selector": ((d.get("spec") or {}).get("selector") or {}).get("matchLabels") or {},
                "cell": _labels(d).get(L_CELL) or _labels(d).get(L_RECIPE) or "",
                "control": is_control_plane(d),
            }
        )
    return out


def _match_deploy(pod, deploys):
    """The cluster-wide llmb Deployment (same ns) whose selector matches this pod, or None. PURE."""
    pns = (pod.get("metadata") or {}).get("namespace", "")
    lab = _labels(pod)
    for d in deploys:
        if d["ns"] != pns or not d["selector"]:
            continue
        if all(lab.get(k) == v for k, v in d["selector"].items()):
            return d
    return None


def _xns_infra_pod(pod, infra_job_names):
    """True if this pod is a run-lifecycle HELPER (run-owner/coord/governor), not the benchmark — it must never
    become a cross-ns run row (it would duplicate the driver). PURE."""
    lab = _labels(pod)
    jn = lab.get("job-name") or lab.get("batch.kubernetes.io/job-name") or ""
    if jn and jn in infra_job_names:
        return True
    name = (pod.get("metadata") or {}).get("name", "")
    return "-runowner-" in name or "-run-owner-" in name or name.startswith("llmb-governor") or "-governor-" in name


def _ns_owner_state(all_jobs):
    """PURE. namespace → 'active' | 'terminal' from OUR Jobs, so a held server is judged owned-vs-abandoned:
       'active'   = a LIVE run-owner (or bench) Job in that ns → the server is IN USE / intentionally held.
       'terminal' = a run-owner that already FINISHED (Complete/Failed) while its server lingers → a genuine
                    orphan the GC hasn't freed (the only case where a reclaim hint is safe to surface).
    A namespace with NO owning Job at all → absent → treated CONSERVATIVELY (parked, no reclaim): a
    `--skip-server` server parked between runs looks identical to an abandoned one, so we never over-alarm.
    """
    st: dict = {}
    for j in all_jobs or []:
        md = j.get("metadata") or {}
        ns = md.get("namespace", "")
        if not ns:
            continue
        s = j.get("status") or {}
        active = int(s.get("active", 0) or 0) > 0
        terminal = int(s.get("succeeded", 0) or 0) > 0 or int(s.get("failed", 0) or 0) > 0
        name = md.get("name", "")
        is_owner = (
            "-runowner-" in name
            or "-run-owner-" in name
            or _labels(j).get("app.kubernetes.io/component") == "run-owner"
        )
        is_bench = not is_infra_job(j)  # a real benchmark driver (not owner/coord/governor)
        if active and (is_owner or is_bench):
            st[ns] = "active"
        elif terminal and is_owner and st.get(ns) != "active":
            st[ns] = "terminal"
    return st


def discover_cross_ns_runs(
    allpods_j,
    is_ours,
    configured_ns,
    cluster_name,
    now,
    all_deploys=None,
    all_jobs=None,
    waiting_runs=frozenset(),
):
    """PURE. Cluster-wide run discovery from `pods -A`: OUR pods in namespaces OTHER than the profile's single
    configured ns — where our per-worktree runs actually launch — grouped by (ns, cell/run-root) and classified
    RUNNING / LOADING / STUCK / ORPHAN. A pod is OURS via its own llmb labels OR — the key real-world case — a
    match to one of our cluster-wide llmb Deployment selectors (`all_deploys`), since a disagg SERVER pod carries
    only {app,role} while the llmb labels sit on its Deployment. Control-plane (etcd/nats/frontend) + run-owner/
    coord/governor INFRA are skipped. Returns RUN-table rows, each carrying its `ns` + a `suffix`. The configured
    ns is covered by the richer single-ns path (skipped here). `all_deploys`/`all_jobs` degrade gracefully to
    pod-label-only attribution when absent."""
    if not allpods_j or "items" not in allpods_j:
        return []
    deploys = _deploy_index(all_deploys) if all_deploys else []
    infra_job_names = {(j.get("metadata") or {}).get("name", "") for j in (all_jobs or []) if is_infra_job(j)}
    owner_state = _ns_owner_state(all_jobs)
    groups: dict = {}
    for p in allpods_j["items"]:
        ns = (p.get("metadata") or {}).get("namespace", "")
        if not ns or ns == configured_ns:
            continue  # configured ns → the single-ns path already renders it richly
        d = _match_deploy(p, deploys) if deploys else None
        if not (is_ours(p) or d is not None):
            continue  # not ours (foreign team, or an unlabeled pod we can't attribute)
        if d and d["control"]:
            continue  # control-plane member (etcd/nats/frontend) → not a run row
        if _xns_infra_pod(p, infra_job_names):
            continue  # run-owner/coord/governor helper → folds into the run, not a row
        root = (
            (d["cell"] if d and d["cell"] else "")
            or _labels(p).get(L_CELL)
            or _labels(p).get(L_RECIPE)
            or _pod_run_root(p)
        )
        groups.setdefault((ns, root), []).append(p)
    rows = []
    for (ns, root), pods in sorted(groups.items()):
        gpus = running_gpu(pods)
        bench_up = any(
            _pod_role(p) == "bench" and _phase(p) == "Running" and _pod_ready(p) and not pod_is_stuck(p) for p in pods
        )
        stuck = any(pod_is_stuck(p) for p in pods)
        coming_up = any(_pod_coming_up(p) for p in pods)
        held_server = any(_pod_role(p) in ("server", "other") and _pod_ready(p) and gpu_request(p) > 0 for p in pods)
        start = min(
            (
                (p.get("status") or {}).get("startTime") or (p.get("metadata") or {}).get("creationTimestamp", "")
                for p in pods
            ),
            default="",
        )
        age = human_age(start, now)
        secs = age_seconds(start, now)
        recent = secs is not None and secs < GRACE_SECONDS
        # The row's namespace is rendered as its NAMESPACE header (Cluster → Namespace → RUN), so the suffix
        # carries only the extra note (warming / held / reclaim), NOT a redundant `ns <ns>` tag.
        if bench_up:
            status, suffix, code = RUNNING, "", "32"
        elif stuck:
            status, suffix, code = STUCK, "crash-loop", "31"
        elif coming_up and root in waiting_runs:
            # This run is waiting for another checkpoint load to release the shared cache.
            status, suffix, code = QUEUED, "waiting for model-load slot", "2;36"
        elif coming_up:
            status, suffix, code = LOADING, "server warming (image/weights)", "33"
        elif held_server or gpus > 0:
            # A ready GPU server with no active bench. It is DANGEROUS to call this a reclaimable ORPHAN: a
            # `--skip-server` server parked between runs (or one a sweep is attaching to right now) looks
            # identical. Surface the reclaim hint ONLY when we're confident it's abandoned — a run-owner in
            # this ns that already FINISHED while the server lingers. An ACTIVE owner/bench, a recent launch,
            # or NO owning Job at all → ● PARKED (held, NO reclaim). Mirrors the single-ns ORPHAN→PARKED
            # downgrade: when in doubt, never tell the operator to reclaim a possibly-live server.
            if owner_state.get(ns) == "terminal" and not recent:
                status, code = ORPHAN, "31"
                suffix = f"holds {_gpus(gpus)} · run finished, server not freed · reclaim: llmb-k8s reclaim-gpu {cluster_name} -n {ns}"
            else:
                status, code = PARKED, "36"
                why = "server up, no active bench" if not recent else "server starting"
                suffix = f"{_gpus(gpus)} held · {why}"
        else:
            continue  # our pods present but no GPU + no run signal → skip (noise)
        rows.append(
            {
                "level": "job",
                "status": status,
                "name": root,
                "keep": "tail",
                "gpus": gpus,
                "timing": age,
                "flag": "",
                "sweep": "",
                "node": "",
                "image": "",
                "restarts": 0,
                "suffix": suffix,
                "suffix_code": code,
                "ns": ns,
                "cluster": cluster_name,
            }
        )
    return rows


# ── per-cluster assembly (pure over already-read JSON) ──────────────────────────────────────────


def _server_row(d, dp, live_recipes, *, group_recent=False, group_held=False, live_roots=frozenset()):
    start = min(
        ((p.get("status") or {}).get("startTime") or _labels_ts(p) for p in dp if _phase(p) == "Running"),
        default=(d.get("metadata") or {}).get("creationTimestamp", ""),
    )
    return {
        "kind": "server",
        "name": (d.get("metadata") or {}).get("name", ""),
        "status": classify_server(
            d,
            dp,
            live_recipes,
            group_recent=group_recent,
            group_held=group_held,
            live_roots=live_roots,
        ),
        "gpus": running_gpu(dp),
        "cpus": running_cpu(dp),
        "start": start,
        "expected": None,
        "expected_src": "",
        "ready": f"{int((d.get('status') or {}).get('readyReplicas', 0) or 0)}/{int((d.get('spec') or {}).get('replicas', 1) or 0)}",
        "run_id": _labels(d).get(L_RUN_ID, ""),
        "sweep": _labels(d).get(L_SWEEP, ""),
        "node": (dp[0].get("spec") or {}).get("nodeName", "") if dp else "",
        "image": _first_image(dp),
        "restarts": max_restarts(dp),
        # WHY no node took it, from the pod we already hold — see unschedulable_reason.
        "unsched_why": unsched_note(dp)[0],
    }


def _run_root(obj: dict) -> str:
    """The run/cell identity shared by a driver Job and its server Deployment — the cell label, else the
    recipe label. This is the key both stamp (`llmb.nvidia.com/cell` / `/recipe`), so it links a per-run server
    to its OWN run's driver regardless of the `-<run-id>-server` suffix in the server's NAME.
    """
    l = _labels(obj)
    return l.get(L_CELL) or l.get(L_RECIPE) or ""


def _owning_job_name(dep, jobs):
    """The ACTIVE benchmark driver Job that owns this server → the run its `└ svc` sub-line nests under.
    `jobs` are the candidate DRIVER jobs (helper/infra jobs are filtered out by the caller, so a server whose
    ownerReference points at the run-owner watcher still resolves to its benchmark driver, not the watcher).
    Only an ACTIVE driver adopts a server — a server with NO active driver returns None and stays a standalone
    row (an idle/orphan server is a real state we must not hide by nesting it under a finished run).

    Match order (most-specific first):
      1. ownerReference → an active driver Job,
      2. shared run-root (cell/recipe) AND shared run-id → the per-run server case: a recipe reused across
         concurrent runs (r149b, r249b) is disambiguated by run-id, so server-r149b never binds to r249b,
      3. recipe/cell / name-prefix match (servers with no run-id label) — preferring an active driver.
    Returns the driver Job name, or None."""
    active = [j for j in jobs if int((j.get("status") or {}).get("active", 0) or 0) > 0]
    act_names = {(j.get("metadata") or {}).get("name", "") for j in active}
    for orf in (dep.get("metadata") or {}).get("ownerReferences") or []:
        if orf.get("kind") == "Job" and orf.get("name") in act_names:
            return orf["name"]
    droot, drun = _run_root(dep), _labels(dep).get(L_RUN_ID)
    if droot and drun:
        run_matches = [j for j in active if _run_root(j) == droot and _labels(j).get(L_RUN_ID) == drun]
        if run_matches:
            return (run_matches[0].get("metadata") or {}).get("name", "")
    rec = _server_recipe(dep)
    matches = [
        j
        for j in active
        if _labels(j).get(L_RECIPE) == rec or _labels(j).get(L_CELL) == rec or (droot and _run_root(j) == droot)
    ]
    if matches:
        return (matches[0].get("metadata") or {}).get("name", "")
    return None


def build_cluster(
    name,
    context,
    namespace,
    pods_j,
    deploys_j,
    jobs_j,
    nodes_j,
    allpods_j,
    median_lookup=None,
    rungs_lookup=None,
    result_lookup=None,
    now=None,
    all_deploys=None,
    all_jobs=None,
    leases_j=None,
    pvcs_j=None,
    workloads_read=True,
) -> dict:
    """Assemble one CONNECTED cluster with the JOB as the primary unit: a "run" = a bench/coord/staging/
    download Job, with its associated server(s) nested under it. Servers not owned by any Job (orphans,
    or run.sh-path servers) are standalone; benign IDLE (scale-to-0) servers are collapsed. Server status
    is evaluated per RUN GROUP so a parked run (control-plane up, GPU workers 0/0) reads PARKED, not ORPHAN.
    PURE."""
    now = now or datetime.now(timezone.utc)
    deploys = [d for d in (deploys_j or {}).get("items", []) if is_llmb(d)]
    jobs = [j for j in (jobs_j or {}).get("items", []) if is_llmb(j)]
    # Do NOT pre-filter namespaced pods by their own labels — our disagg server pods carry the llmb labels
    # only on the Deployment, not the pod. Rows match pods by Deployment selector / Job name below (a strong
    # "ours" signal); the owner-aware matcher attributes cluster-wide -A pods for the foreign count.
    pods = (pods_j or {}).get("items", [])
    is_ours = make_ours_matcher(deploys, jobs, namespace)

    live_recipes = set()
    live_roots = set()  # run-roots (cell/recipe) of ACTIVE drivers — adopts per-run servers whose
    for j in jobs:  # name-derived recipe (…-<run-id>-server) isn't the recipe label (would false-ORPHAN)
        if int((j.get("status") or {}).get("active", 0) or 0) > 0:
            live_recipes.add(_labels(j).get(L_RECIPE) or (j.get("metadata") or {}).get("name", ""))
            r = _run_root(j)
            if r:
                live_roots.add(r)

    # ── run-group context: recency + a live control-plane make the whole group NOT-orphan ─────────
    group_recent, group_held = {}, {}
    for d in deploys:
        g = run_group_key(d)
        age = age_seconds((d.get("metadata") or {}).get("creationTimestamp", ""), now)
        if age is not None and age < GRACE_SECONDS:
            group_recent[g] = True
        if is_control_plane(d) and int((d.get("status") or {}).get("readyReplicas", 0) or 0) >= 1:
            group_held[g] = True

    # server rows keyed by name; associate each to its owning benchmark DRIVER Job (or standalone). Only the
    # real benchmark drivers are ownership candidates — a run's run-owner/coord helper Jobs (infra) never adopt
    # a server, so a server whose ownerReference points at the run-owner still nests under its driver, not the
    # watcher (which is folded to a footer count, not a row).
    driver_jobs = [j for j in jobs if not is_infra_job(j)]
    srow = {}
    owner_of = {}
    for d in deploys:
        sel = ((d.get("spec") or {}).get("selector") or {}).get("matchLabels") or {}
        dp = [p for p in pods if _selector_match(p, sel)]
        g = run_group_key(d)
        r = _server_row(
            d,
            dp,
            live_recipes,
            group_recent=group_recent.get(g, False),
            group_held=group_held.get(g, False),
            live_roots=live_roots,
        )
        srow[r["name"]] = r
        owner_of[r["name"]] = _owning_job_name(d, driver_jobs)

    # job rows (primary), each carrying its nested servers
    job_rows = []
    for j in jobs:
        jn = (j.get("metadata") or {}).get("name", "")
        jp = _job_pods(jn, pods)
        start = (j.get("status") or {}).get("startTime") or (j.get("metadata") or {}).get("creationTimestamp", "")
        expected, esrc = resolve_expected(j, median_lookup)
        rungs, rsrc = resolve_rungs(j, rungs_lookup)
        servers = sorted((srow[s] for s, o in owner_of.items() if o == jn), key=lambda r: r["name"])
        jstatus = classify_job(j, jp)
        jcell = _labels(j).get(L_CELL) or _labels(j).get(L_RECIPE) or ""
        jrun = _labels(j).get(L_RUN_ID, "")
        # TERMINAL-run triage context (computed only for the tail we actually show as history):
        #   ✗ FAILED → the terse cause + fuller message, straight from live Job conditions / pod lastState.
        #   ✓ done   → the headline (metric, value) IF the run has been collected into runs.jsonl, else None
        #              (renderer degrades to a `collect <run-id>` pointer — never a fabricated number).
        fail_short, fail_full, result = "", "", None
        unsched_why, _unsched_secs = unsched_note(jp)  # '' unless a pod of this Job is unschedulable
        if jstatus == FAILED:
            fail_short, fail_full = job_fail_cause(j, jp)
        elif jstatus == COMPLETE and result_lookup is not None:
            result = result_lookup(jcell, jrun)
        job_rows.append(
            {
                "kind": "job",
                "name": jn,
                "status": jstatus,
                "gpus": running_gpu(jp),
                "cpus": running_cpu(jp),
                "start": start,
                "expected": expected,
                "expected_src": esrc,
                "ready": _job_ready(j),
                "cell": jcell,
                "run_id": jrun,
                "fail_cause": fail_short,
                "fail_cause_full": fail_full,
                "result": result,
                "unsched_why": unsched_why,
                "sweep": rungs,
                "sweep_src": rsrc,  # SWEEP column = concurrency rungs (any launch path)
                # rungs done / total / has-a-live-signal → the SWEEP progress dot-bar (live annotation, resume env,
                # or a settled ✓ sweep; else no signal → the plain rung list).
                **dict(
                    zip(
                        ("sweep_done", "sweep_total", "sweep_live"),
                        resolve_sweep_progress(j, rungs),
                    )
                ),
                "variance_sweep": _labels(j).get(L_SWEEP, ""),  # submit --repeat grouping id (may be absent)
                "infra": is_infra_job(j),  # governor cron etc. — not a benchmark run
                "finished": job_finished_ts(j),
                "node": (jp[0].get("spec") or {}).get("nodeName", "") if jp else "",
                "image": _first_image(jp),
                "restarts": max_restarts(jp),
                "servers": servers,
            }
        )
    job_rows.sort(key=lambda r: r["name"])

    # ── ONE ROW PER RUN. A single run (shared run-root cell/recipe + run-id) can still surface >1 ACTIVE
    #    benchmark Job (e.g. a live driver plus a server-holder sibling). Render the run ONCE — keep the
    #    sweep-carrying DRIVER as the row and FOLD the sibling(s) into it (merge servers + GPUs, dedup) — so
    #    two independent runs read as two rows, never four. Only groups with a shared, non-empty run-id merge
    #    (distinct-run-id jobs are always distinct runs); terminal/infra jobs are untouched. ──────────────────
    groups = {}
    for i, jr in enumerate(job_rows):
        if jr["status"] in ACTIVE_JOB_STATES and not jr["infra"] and jr.get("run_id") and jr.get("cell"):
            groups.setdefault((jr["cell"], jr["run_id"]), []).append(i)
    drop = set()
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        # primary = the driver that carries a resolved sweep (the real bench driver); tiebreak by name.
        prim = min(idxs, key=lambda i: (job_rows[i]["sweep"] == "", job_rows[i]["name"]))
        p = job_rows[prim]
        have = {s["name"] for s in p["servers"]}
        for i in idxs:
            if i == prim:
                continue
            sib = job_rows[i]
            for s in sib["servers"]:
                if s["name"] not in have:
                    p["servers"].append(s)
                    have.add(s["name"])
            p["gpus"] += sib["gpus"]  # sibling's own (non-server) GPUs, counted once
            if not p["sweep"] and sib["sweep"]:
                p["sweep"], p["sweep_src"] = sib["sweep"], sib["sweep_src"]
            drop.add(i)
        p["servers"].sort(key=lambda r: r["name"])
    if drop:
        job_rows = [jr for i, jr in enumerate(job_rows) if i not in drop]

    standalone = sorted(
        (r for nm, r in srow.items() if owner_of[nm] is None and r["status"] != IDLE),
        key=lambda r: r["name"],
    )
    idle = sorted(
        (r for nm, r in srow.items() if owner_of[nm] is None and r["status"] == IDLE),
        key=lambda r: r["name"],
    )

    all_rows = job_rows + [s for j in job_rows for s in j["servers"]] + standalone + idle
    total = gpu_total(nodes_j)
    # Namespaces WE manage: the profile's configured ns + every ns holding our llmb workloads. THE single
    # notion of ownership on this cluster — it splits "unattributed in ours" out of the foreign GPU bucket AND
    # scopes the INSTALLED inventory (see discover_model_caches). Deliberately ONE set: a second, divergent
    # definition is how the pane ended up rendering 229 colleagues' namespaces off a cluster-wide PVC read.
    our_ns = {namespace} if namespace else set()
    for o in list(all_deploys or []) + list(all_jobs or []):
        _n = (o.get("metadata") or {}).get("namespace", "")
        if _n:
            our_ns.add(_n)
    occupied, foreign, unattributed = gpu_occupied_and_foreign(allpods_j, is_ours, our_ns)
    nodes = node_capacity(nodes_j, allpods_j, is_ours)  # free-NODE capacity (the schedulable unit)
    # CROSS-NAMESPACE run discovery from `pods -A`: OUR runs in per-worktree namespaces (LOADING / RUNNING /
    # ORPHAN) the single configured-ns read is blind to. The cluster-wide llmb Deployments (all_deploys, from
    # `deploy -A -l managed-by`) let us attribute a disagg SERVER pod (its llmb labels live on the Deployment,
    # not the pod) to us in another namespace — so a run LOADING there is discovered, not invisible. Empty when
    # pods -A is unreadable (no cluster RBAC).
    _mload = discover_model_load(leases_j, all_jobs, now)
    xns_runs = discover_cross_ns_runs(
        allpods_j,
        is_ours,
        namespace,
        name,
        now,
        all_deploys=all_deploys,
        all_jobs=all_jobs,
        waiting_runs=model_load_waiting_runs(_mload),
    )
    # OURS GPU is CLUSTER-WIDE: the single configured-ns rows PLUS our GPUs held in OTHER namespaces (per-worktree
    # runs). Without the cross-ns term the headline reads "0 GPU (ours)" while a run in another ns holds them.
    ours_gpu = sum(r["gpus"] for r in all_rows) + sum(r["gpus"] for r in xns_runs)
    # INSTALLED is CLUSTER-WIDE: discover from the UNION of the configured-ns read and our workloads across ALL
    # namespaces (`deploy,jobs -A -l managed-by`, a superset on a real cluster) so "what can be run here"
    # reflects every namespace. discover_installed dedups by cell/model, so the overlap collapses.
    inst_deploys = {"items": ((deploys_j or {}).get("items") or []) + (all_deploys or [])}
    inst_jobs = {"items": ((jobs_j or {}).get("items") or []) + (all_jobs or [])}
    # Which PVCs a LIVE workload still mounts — the only input that can age a per-run `<cell>-artifacts` volume
    # into "leaked". Built from objects already fetched (no extra call).
    #
    # None = we CANNOT TELL, and then the rollup asserts no leak count at all. The gate is deliberately strict:
    # `_split_llmb` turns a FAILED cluster-wide llmb read into ([], []), indistinguishable from "we own no
    # workloads" — so without `workloads_read` a single timed-out call would silently reclassify every
    # artifacts PVC on the cluster as leaked, and the pane would accuse a running campaign of leaking the very
    # volumes it is writing to. The read-did-not-land rule, applied to leaks: absence of evidence is not
    # evidence of absence.
    _live_claims = (
        referenced_claims(allpods_j, pods_j, inst_deploys, inst_jobs)
        if (workloads_read and (allpods_j is not None or pods_j is not None))
        else None
    )
    return {
        "state": "connected",
        "name": name,
        "context": context,
        "namespace": namespace,
        "jobs": job_rows,
        "standalone": standalone,
        "idle": idle,
        "ours_gpu": ours_gpu,
        "ours_cpu": sum(r["cpus"] for r in all_rows),
        "total": total,
        "occupied": occupied,
        "foreign": foreign,
        "unattributed_gpu": unattributed,
        "nodes": nodes,
        # LIVE-discovered installed inventory (ground truth for `--stages` INSTALLED when connected).
        # INSTALLED = deployed cells + model-download evidence + the model-cache PVCs that actually exist on
        # the CLUSTER (so a model downloaded elsewhere is still visible, and every cache in the ns is listed).
        "discovered": _merge_installed(
            discover_installed(inst_deploys, inst_jobs),
            discover_model_caches(
                pvcs_j,
                (inst_jobs or {}).get("items") or [],
                our_namespaces=our_ns,
                live_claims=_live_claims,
                pins=load_model_pins(),
            ),
        ),
        # The model-download Jobs, kept as OBJECTS for the MODEL LEDGER: a Job whose PVC is not in the read
        # (GC'd claim, PVC-only RBAC denial) is still evidence about a model, and dropping it here is how a
        # cached model became `○ MISSING`.
        "all_download_jobs": [
            j
            for j in ((inst_jobs or {}).get("items") or [])
            if COMP_MODEL_DOWNLOAD in (_labels(j).get(L_COMPONENT2), _labels(j).get(L_COMPONENT))
        ],
        # The namespaces this cluster's view is scoped to (ownership set above) — exposed so the scoping is
        # inspectable/testable rather than an invisible property of the rendered rows.
        "our_namespaces": sorted(our_ns),
        # Did the model-cache (PVC) read actually LAND? False = it did not: `--fast` skips it, or the call was
        # RBAC-Forbidden / timed out (fleet.sh writes no file on a failed read, so the renderer gets None).
        # Without this flag an UNAVAILABLE inventory is indistinguishable from an EMPTY one and the pane
        # states "— nothing installed —" over a namespace holding 700GB of weights. Absence of evidence is
        # not evidence of absence — the renderer says "inventory unknown" and WHY instead.
        "inventory_unavailable": pvcs_j is None,
        # MODEL-LOAD queue (per model-cache PVC). [] when nothing is loading OR Leases are unreadable.
        "model_load": _mload,
        "xns_runs": xns_runs,
    }


def _labels_ts(p):
    return (p.get("metadata") or {}).get("creationTimestamp", "")


def _first_image(pods: list) -> str:
    for p in pods:
        for c in (p.get("spec") or {}).get("containers") or []:
            img = c.get("image", "")
            return img.rsplit(":", 1)[-1] if ":" in img else img
    return ""


def _job_ready(job: dict) -> str:
    s = job.get("status") or {}
    return f"a{int(s.get('active',0) or 0)}s{int(s.get('succeeded',0) or 0)}f{int(s.get('failed',0) or 0)}"


# ── render ──────────────────────────────────────────────────────────────────────────────────────


def _cluster_gpu_line(c: dict, *, fast: bool = False) -> str:
    """The headline capacity for one connected cluster, HEADLINED by free WHOLE nodes (the schedulable unit —
    a tp=4/8 run takes a whole node's GPUs), not misleading free-GPU counts. In --fast/--mine mode the
    cluster-scoped capacity reads were skipped, so show OURS only (no capacity, no misleading n/a).
    Delegates to `_capacity_text` so the free-node semantics live in one place."""
    ours = c["ours_gpu"]
    cpu = fmt_cores(c.get("ours_cpu", 0))  # CPU shown only when non-zero (same rule as GPU)
    cputail = f" · cpu {cpu}" if cpu else ""
    if fast:
        return f"ours {_gpus(ours)} (fast: capacity skipped){cputail}"
    return _capacity_text(c) + cputail


def _fresh_note(stale: str) -> str:
    """A `· …refreshing · updated Ns ago` suffix for a --watch laggard shown from its last good frame."""
    if not stale:
        return ""
    try:
        secs = int(stale)
    except ValueError:
        return " · …refreshing"
    return f" · …refreshing · updated {human_dur(secs).lstrip('~')} ago"


def _stale_badge(stale: str) -> str:
    """PURE. The LOUD staleness marker for the hierarchy pane's CLUSTER bar — '' when the frame is live.

    A watch laggard uses its last good frame, which must be marked stale so old capacity is not presented as live.
    """
    if not stale:
        return ""
    if stale == "pending":
        return " · …refreshing"
    try:
        secs = int(stale)
    except ValueError:
        return " · ⚠ STALE"
    return f" · ⚠ STALE — last live {human_dur(secs).lstrip('~')} ago (refreshing)"


def _cluster_header(e: dict, *, fast: bool = False, bare: bool = False) -> str:
    """One clean cluster header line:  `── <name> · ns <ns> · [TYPE]  <gpu-line>  [N profiles] <fresh> ──`.
    `bare` (disconnected/refreshing) omits the GPU line. Deterministic and terminal-width-friendly.
    """
    parts = [f"── {e['name']}", f"ns {e['namespace']}"]
    gtype = e.get("gpu_type", "")
    if gtype:
        parts.append(f"[{gtype}]")
    head = " · ".join(parts)
    segs = [head]
    if not bare:
        segs.append(_cluster_gpu_line(e, fast=fast))
    tail = ""
    if e.get("profiles", 1) > 1:
        tail += f"  [{e['profiles']} profiles]"
    tail += _fresh_note(e.get("stale", ""))
    return "  ".join(segs) + tail + "  " + "─" * 3


def _type_breakdown(connected: list) -> str:
    """`2×B200 · 1×GB200 · 1×GB300` over connected clusters with a known GPU type (deterministic order)."""
    counts = {}
    for c in connected:
        t = c.get("gpu_type", "")
        if t:
            counts[t] = counts.get(t, 0) + 1
    if not counts:
        return ""
    return " · ".join(f"{counts[t]}×{t}" for t in sorted(counts))


def _remaining(row, now):
    """Seconds left for a run = expected − elapsed (or deadline − elapsed for a deadline ceiling). None if
    no expectation, unknown, or already over. Drives the soonest-ETA answer."""
    # A RUN THAT HAS NOT STARTED HAS NO ETA. `expected − elapsed` on a pod that is not running counts down
    # a job that is not running, so the fleet headline advertised `⏱ soonest … ~ETA 20m` for a run no node
    # had accepted — and it got SOONER the longer it stayed broken. Same lie as the per-row percentage, one
    # level up: no estimate at all is the honest answer until work begins. Covers STARTING as well as
    # UNSCHED — a wedged image pull is no more under way than a wedged schedule.
    if row.get("status") in (UNSCHED, STARTING):
        return None
    exp = row.get("expected")
    if not exp:
        return None
    secs = age_seconds(row["start"], now)
    if secs is None:
        return None
    rem = exp - secs
    return rem if rem > 0 else None


def _active_run(cluster, row, gpus, now):
    """A normalized 'active run' from a Job (or a starting/wedged server) for the answer-at-a-glance list.
    The display name reads as the operator thinks of the run: the cell/recipe (+ run-id), e.g. `modelx r1c1`,
    not the raw Job name. A starting/wedged server keeps its server name."""
    if row["kind"] == "job":
        base = row.get("cell") or row["name"]
        name = f"{base} {row['run_id']}" if row.get("run_id") and row["run_id"] not in base else base
    else:
        name = row["name"]
    return {
        "cluster": cluster["name"],
        "gpu_type": cluster.get("gpu_type", ""),
        "status": row["status"],
        "name": name,
        "sweep": row.get("sweep", ""),
        "gpus": gpus,
        "timing": _elapsed_expected(row, now),
        "remaining": _remaining(row, now),
        "remaining_ceiling": row.get("expected_src") == "deadline",
    }  # ETA is a deadline cap, not a median


def _collect_active(connected, now):
    """Every ACTIVE run across the fleet: a live bench/coord Job (its GPU = job pods + its driven servers),
    plus a standalone server that is LOADING (a run coming up) or STUCK (a wedged run). Bare RUNNING/PARKED/
    INFRA/ORPHAN servers are NOT active runs — they collapse to counts."""
    actives = []
    for c in connected:
        for j in c["jobs"]:
            if j.get("infra"):
                continue  # governor cron etc. — never a benchmark run row
            if j["status"] in ACTIVE_JOB_STATES:
                gpus = j["gpus"] + sum(s["gpus"] for s in j["servers"])
                actives.append(_active_run(c, j, gpus, now))
        for s in c["standalone"]:
            if s["status"] in (LOADING, STUCK):
                actives.append(_active_run(c, s, s["gpus"], now))
        # CROSS-NAMESPACE efforts (our per-worktree runs the single-ns read misses): a RUNNING/LOADING/STUCK
        # run in another namespace IS an active run — count it so the headline isn't a false "0 runs" while our
        # GPUs are held. ORPHAN (held, no owner) is a footprint, not an active run → excluded (as single-ns).
        for r in c.get("xns_runs") or []:
            if r["status"] in (RUNNING, LOADING, STUCK, STARTING, UNSCHED, QUEUED):
                actives.append(
                    {
                        "cluster": c["name"],
                        "gpu_type": c.get("gpu_type", ""),
                        "status": r["status"],
                        "name": f"{r['name']} ({r['ns']})",
                        "sweep": "",
                        "gpus": r["gpus"],
                        "timing": r["timing"],
                        "remaining": None,
                        "remaining_ceiling": False,
                    }
                )
    actives.sort(key=lambda a: (a["cluster"], a["name"]))
    return actives


def _soonest_eta(actives) -> str:
    rems = [a for a in actives if a["remaining"] is not None]
    if not rems:
        return ""
    a = min(rems, key=lambda x: x["remaining"])
    dur = human_dur(a["remaining"]).lstrip("~")
    # a deadline-sourced 'remaining' is a CEILING (time until the Job self-kills), not a median ETA — mark it
    # `≤ … cap` so the headline never reads a wall-clock cap as an expected finish.
    if a.get("remaining_ceiling"):
        return f"⏱ soonest {a['name']}@{a['cluster']} ≤{dur} to deadline cap"
    return f"⏱ soonest {a['name']}@{a['cluster']} ~ETA {dur}"


def _elapsed_expected(r, now) -> str:
    """`1h58m` alone, or `1h58m / ~4.8h exp (41%)` with a ⚠ flag when a run runs long. Servers (expected
    None) just show age; the run's expectation lives on its Job row."""
    elapsed = human_age(r["start"], now)
    # Progress percentages are shown only after work starts; pending rows show elapsed wait time.
    if r.get("status") in (UNSCHED, STARTING):
        secs = age_seconds(r["start"], now)
        if r["status"] == UNSCHED:
            s = f"waiting {elapsed}"
            if secs is not None and secs >= UNSCHED_WARN_SECS:
                s += f" ⚠ unscheduled {human_dur(secs)} — no node has accepted it"
            return s
        # Starting rows show elapsed startup time until a container reaches Running.
        s = f"starting {elapsed}"
        if secs is not None and secs >= STARTING_WARN_SECS:
            s += f" ⚠ starting {human_dur(secs)} — no container has reached Running"
        return s
    exp = r.get("expected")
    if not exp:
        return elapsed
    secs = age_seconds(r["start"], now)
    if secs is None:
        return elapsed
    pct = int(round(secs / exp * 100))
    if r.get("expected_src") == "deadline":
        s = f"{elapsed} / {human_dur(exp)} max ({pct}%)"
        if pct >= NEAR_DEADLINE_PCT:
            s += " ⚠ near-deadline"
    else:
        s = f"{elapsed} / {human_dur(exp)} exp ({pct}%)"
        if pct >= OVER_MEDIAN_PCT:
            s += " ⚠ over-median"
    return s


def _terminal_runs(e):
    """This cluster's non-infra terminal benchmark Jobs (COMPLETE or FAILED), newest-ended first."""
    term = [j for j in e["jobs"] if not j.get("infra") and j["status"] in (COMPLETE, FAILED)]
    return sorted(term, key=lambda j: j.get("finished", ""), reverse=True)


def _recent_terminal(e, now, *, window_secs, limit, failed_only):
    """Terminal runs to SHOW as history rows. `window_secs` bounds by end-age (None = any age); `limit` caps
    the count (None = uncapped); `failed_only` keeps only ✗ FAILED. Returns (shown, hidden_count) so the
    footer can honestly note runs older than the window instead of silently truncating.
    """
    term = _terminal_runs(e)
    if failed_only:
        term = [j for j in term if j["status"] == FAILED]
    if window_secs is not None:
        kept = []
        for j in term:
            a = age_seconds(j.get("finished", ""), now)
            if a is None or a <= window_secs:
                kept.append(j)
        term = kept
    shown = term[:limit] if limit is not None else term
    return shown, len(_terminal_runs(e)) - len(shown)


def _collapse_counts(e):
    """Per-cluster non-active tallies for the compact footer. done is SPLIT into succeeded/failed (a failure
    must never hide inside a generic 'done'), and governor/infra jobs are excluded from both.
    """
    servers = list(e["standalone"]) + [s for j in e["jobs"] for s in j["servers"]]
    parked = sum(1 for s in servers if s["status"] == PARKED)
    infra = sum(1 for s in servers if s["status"] == INFRA)
    orphans = [s for s in servers if s["status"] == ORPHAN]
    runs = [j for j in e["jobs"] if not j.get("infra")]
    succeeded = sum(1 for j in runs if j["status"] == COMPLETE)
    failed = sum(1 for j in runs if j["status"] == FAILED)
    infra_jobs = sum(1 for j in e["jobs"] if j.get("infra"))
    return {
        "idle": len(e["idle"]),
        "parked": parked,
        "infra": infra,
        "orphan": len(orphans),
        "orphan_gpu": sum(s["gpus"] for s in orphans),
        "succeeded": succeeded,
        "failed": failed,
        "infra_jobs": infra_jobs,
    }


# ── aligned nested-tree table (cluster → namespace → jobs → servers) ──────────────────────────────

NAME_CAP = 26  # truncate a long run/server name (ellipsis) so columns stay within a terminal
SWEEP_CAP = 14  # fits a rung list (16·32·64) OR a sweep progress dot-bar (●●◐○○ 2/5 · up to 8 segs + N/M)
STATUS_CAP = 13  # fits `● UNSCHEDULED`, the longest state — see the STARTING/UNSCHED split
_TREE_INDENT = "     "  # data rows sit 5 spaces in (under the level-2 namespace line at 3)


def _vlen(s: str) -> int:
    return len(s)  # monospace approximation (fixtures/CLI are ASCII+BMP; good enough for alignment)


def _trunc(text: str, cap: int, keep: str = "head") -> str:
    """Truncate to `cap` with a `…`. keep='head' → `abcde…` (jobs); keep='tail' → `…prefill` (servers,
    whose meaning is in the suffix like -prefill/-decode)."""
    if _vlen(text) <= cap:
        return text
    if keep == "tail":
        return "…" + text[-(cap - 1) :]
    return text[: cap - 1] + "…"


def _split_flag(timing: str):
    """Separate the ⚠ stall flag from the timing so it can trail the row instead of widening the AGE column
    (a `⚠ near-deadline` on one run shouldn't push every other row's SWEEP column to the right).
    """
    if " ⚠ " in timing:
        base, flag = timing.split(" ⚠ ", 1)
        return base, "⚠ " + flag
    return timing, ""


def _svc_trow(s, now, *, nested):
    """A server row. Nested (under its driving job) → level 'svc' (`└ svc`); standalone → level 'job'.
    The server's FULL Deployment name is shown un-truncated (e.g. `…-pareto-r149b-server`) — the NAME column
    widens to fit rather than abbreviating the identity. tail-kept so the meaningful `…-server`/`…-decode`
    suffix leads if the terminal is ever narrow."""
    age = human_age(s["start"], now)
    timing = (f"waiting {age}" if s["status"] == UNSCHED else f"{s['ready']}  {age}").strip()
    r = {
        "level": "svc" if nested else "job",
        "status": s["status"],
        "name": s["name"],
        "keep": "tail",
        "gpus": s["gpus"],
        "timing": timing,
        "flag": "",
        "sweep": s.get("sweep", ""),
        "node": s.get("node", ""),
        "image": s.get("image", ""),
        "restarts": s.get("restarts", 0),
        "waited": age_seconds(s["start"], now) if s["status"] == UNSCHED else None,
    }
    if s.get("unsched_why"):  # the diagnosis rides the row, not a `kubectl describe` the reader runs
        r["suffix"], r["suffix_code"] = "⚠ " + s["unsched_why"], "31"
    return r


def _run_name(j) -> str:
    """The run's FULL name, un-truncated: the cell/recipe joined with its run-id — the exact identifier the
    operator greps for (e.g. `nemotron-ultra-3-b200-pareto-r149b`). No elision/abbreviation;
    the NAME column widens to fit rather than cutting the identity. Falls back to the bare cell when there is
    no run-id, and never duplicates a run-id already embedded in the cell name."""
    cell = j.get("cell") or j["name"]
    rid = j.get("run_id", "") or ""
    return f"{cell}-{rid}" if rid and rid not in cell else cell


def _term_trow(j, now):
    """A terminal (✓ done / ✗ FAILED) run history row: shows WHAT ran, its rungs, when it ended, and — the
    triage payload — the ✗ cause + how-to-investigate one-liner, or the ✓ result / where-to-collect pointer
    (rendered as a trailing suffix by _term_action, so column alignment is untouched).
    """
    fin = human_age(j.get("finished", ""), now)
    return {
        "level": "term",
        "status": j["status"],
        "name": _run_name(j),
        "keep": "head",
        "gpus": 0,
        "timing": (f"ended {fin} ago" if fin else "ended"),
        "flag": "",
        "sweep": _sweep_field(
            j.get("sweep", ""),
            j.get("sweep_done"),
            j.get("sweep_total"),
            j["status"],
            live=j.get("sweep_live", False),
        ),
        "node": j.get("node", ""),
        "image": j.get("image", ""),
        "restarts": j.get("restarts", 0),
        "run_id": j.get("run_id", ""),
        "cause": j.get("fail_cause", ""),
        "cause_full": j.get("fail_cause_full", ""),
        "result": j.get("result"),
    }


def _cluster_trows(
    e,
    now,
    *,
    gpu_only,
    detail,
    show_idle,
    history_secs=DEFAULT_HISTORY_SECS,
    history_n=None,
    failed_only=False,
):
    """Rows for one cluster's table, in lifecycle order: ACTIVE runs (each job + its driven servers), then
    RECENT TERMINAL runs (✓ done / ✗ FAILED within the history window — so a just-crashed run is VISIBLE, not
    hidden as 'idle'), then (with --detail) the collapsed parked/infra/orphan(/idle) servers. Governor/infra
    jobs are never rows. Returns (rows, hidden_terminal_count) — the caller notes hidden older runs, never
    truncating silently."""
    rows = []
    for j in e["jobs"]:
        if j.get("infra") or j["status"] not in ACTIVE_JOB_STATES:
            continue
        total = j["gpus"] + sum(s["gpus"] for s in j["servers"])
        te, flag = _split_flag(_elapsed_expected(j, now))
        # a real run's SWEEP is never silently blank: if no LIVE_RUNGS/CONCURRENCIES env and no committed
        # recipe resolved the rungs, show '?' (genuinely unknowable) rather than an empty cell.
        rungs = _sweep_field(
            j.get("sweep") or "?",
            j.get("sweep_done"),
            j.get("sweep_total"),
            j["status"],
            live=j.get("sweep_live", False),
        )
        row = {
            "level": "job",
            "status": j["status"],
            "name": _run_name(j),
            "keep": "head",
            "gpus": total,
            "timing": te,
            "flag": flag,
            "sweep": rungs,
            "node": j.get("node", ""),
            "image": j.get("image", ""),
            "restarts": j.get("restarts", 0),
            "waited": age_seconds(j["start"], now) if j["status"] == UNSCHED else None,
        }
        if j.get("unsched_why"):
            # THE SAME DIAGNOSIS THE HIERARCHY PANE CARRIES. It was attached in `_stage_run_rows` only, and
            # `fleet.sh` routes to that pane only under `--watch` — so a bare `fleet.sh` still sent the
            # operator to `kubectl describe` for a string already in hand. One fact, both panes.
            row["suffix"], row["suffix_code"] = "⚠ " + j["unsched_why"], "31"
        rows.append(row)
        for s in j["servers"]:
            rows.append(_svc_trow(s, now, nested=True))
    for s in e["standalone"]:
        if s["status"] in (LOADING, STUCK) or detail:
            rows.append(_svc_trow(s, now, nested=False))
    # RECENT TERMINAL runs (the lifecycle tail). --detail / --history-n uncap the count & drop the window.
    shown, hidden = _recent_terminal(
        e,
        now,
        window_secs=(None if detail else history_secs),
        limit=(None if (detail or history_n is None) else history_n),
        failed_only=failed_only,
    )
    for j in shown:
        rows.append(_term_trow(j, now))
    if detail and show_idle:
        for s in e["idle"]:
            rows.append(_svc_trow(s, now, nested=False))
    if gpu_only:
        rows = [r for r in rows if r["gpus"] > 0 or r["level"] == "term"]  # keep terminal history even at 0g
    return rows, hidden


def _status_core(r) -> str:
    st = r["status"]
    if r["level"] == "svc":
        return "  └ svc"
    if st == COMPLETE:
        return "✓ done"
    if st == FAILED:
        return "✗ FAILED"
    if st == QUEUED:
        return "◔ QUEUED"  # deliberately NOT ● — it is not running and not loading, it is waiting its turn
    return f"● {st}"


def _status_code(r) -> str:
    # UNSCHEDULED is the one state whose SEVERITY is a function of the clock: for a few seconds it is the
    # scheduler working normally; after the threshold it becomes an actionable scheduling failure.
    if r["status"] == UNSCHED:
        waited = r.get("waited")
        return "31" if (waited is not None and waited >= UNSCHED_WARN_SECS) else "33"
    if r["level"] == "svc":
        return {"STUCK": "31", "LOADING": "33"}.get(r["status"], "2")  # nested servers quiet unless wedged
    return _ANSI.get(r["status"], "0")


def _gpu_txt(r) -> str:
    return _gpus(r["gpus"]) if r["gpus"] else "·"


def _col_widths(rows: list) -> dict:
    """Per-column max VISIBLE width across every row (+ the header labels) → straight vertical columns."""

    def w(vals, hdr, cap=None):
        m = max([_vlen(v) for v in vals] + [len(hdr)])
        return min(cap, m) if cap else m

    return {
        "status": w([_status_core(r) for r in rows], "STATUS", STATUS_CAP),
        # NAME is UN-CAPPED: run/server identities (long cell-clone + `-server` names) are shown in FULL — the
        # column widens to fit rather than truncating the exact string the operator greps for.
        "name": w([r["name"] for r in rows], "RUN / SERVER"),
        "gpu": w([_gpu_txt(r) for r in rows], "GPU"),
        "timing": w([r["timing"] for r in rows], "AGE / EXPECTED"),
        "sweep": w([_trunc(r["sweep"], SWEEP_CAP, "tail") for r in rows], "SWEEP", SWEEP_CAP),
    }


def _col_header(W: dict) -> str:
    # roomier inter-column gaps (2·2·3·2) so the row fields breathe horizontally — matched EXACTLY by _fmt_trow.
    return (
        f"{'STATUS'.ljust(W['status'])}  {'RUN / SERVER'.ljust(W['name'])}  {'GPU'.rjust(W['gpu'])}   "
        f"{'AGE / EXPECTED'.ljust(W['timing'])}  {'SWEEP'.ljust(W['sweep'])}"
    ).rstrip()


def _term_action(r, *, detail: bool = False) -> tuple[str, str]:
    """The TRIAGE suffix on a terminal history row → (text, ansi_code); '' for non-terminal rows. This is what
    turns a status board into an act-on-it view:
      ✗ FAILED → `✗ why: <cause> · logs: llmb-k8s logs <run-id>` (the glance cause + the exact dig-in command;
                 --detail appends the fuller condition/terminated message).
      ✓ done   → `→ <metric>=<value> · collect <run-id>` when the result is cached in runs.jsonl, else the
                 honest pointer `→ result: collect <run-id>` (+ `(metric not cached — analyze <run-id>)` under
                 --detail). Never a fabricated number.
    Rendered AFTER all aligned columns (like the ⚠ flag), so it never shifts a column offset.

    A stages-view ORPHAN row (level 'orph') gets a `⚠ holds Ng · reclaim: …` suffix — the same alignment-safe
    pattern, naming the actionable reclaim command (it holds GPU with its run gone)."""
    if r.get("level") == "orph":
        return (
            f"⚠ holds {_gpus(r['gpus'])} · reclaim: llmb-k8s reclaim-gpu {r.get('cluster', '')}".rstrip(),
            "31",
        )
    if r.get("level") != "term":
        return "", "0"
    rid = r.get("run_id") or ""
    tail = rid or "<run-id>"
    if r.get("status") == FAILED:
        cause = r.get("cause") or "cause unknown (pods reaped)"
        txt = f"✗ why: {cause} · logs: llmb-k8s logs {tail}"
        full = r.get("cause_full") or ""
        if detail and full:
            txt += f"  ({_trunc(full, 80)})"
        return txt, "31"
    if r.get("status") == COMPLETE:
        res = r.get("result")
        if res:
            metric, value = res
            txt = f"→ {metric}={value}"
            txt += f" · collect {tail}" if detail else ""
            return txt, "2"
        txt = f"→ result: collect {tail}"  # not collected into runs.jsonl yet → point, don't invent
        if detail:
            txt += f" (metric not cached — analyze {tail})"
        return txt, "2"
    return "", "0"


def _fmt_trow(r, W: dict, paint: Paint, *, wide: bool = False, detail: bool = False) -> str:
    """One aligned data row: STATUS | NAME | GPU(right) | AGE/EXPECTED | SWEEP — padded to the shared widths
    so every field lines up vertically. Color is applied to the status cell only; padding is plain so the
    ANSI codes never shift a column. `--wide` appends node/image/restarts as trailing detail columns. A
    terminal row also gets its triage suffix (cause+logs / result+collect) trailing, alignment-safe.
    """
    core = _status_core(r)
    status_cell = paint.c(_status_code(r), core) + " " * max(0, W["status"] - _vlen(core))
    name = _trunc(r["name"], W["name"], r["keep"]).ljust(W["name"])
    gpu = _gpu_txt(r).rjust(W["gpu"])
    timing = r["timing"].ljust(W["timing"])
    sweep = _trunc(r["sweep"], W["sweep"], "tail")
    line = f"{status_cell}  {name}  {gpu}   {timing}"  # aligned core (2·2·3 gaps, matched by _col_header)
    if wide:
        line += f"  {sweep.ljust(W['sweep'])}  node={r.get('node') or '-'} img={r.get('image') or '-'} rst={r.get('restarts', 0)}"
    elif sweep:
        line += f"  {sweep}"
    if r.get("flag"):  # ⚠ stall flag TRAILS the row (never widens a column)
        line += f"  {paint.c('33', r['flag'])}"
    action, code = _term_action(r, detail=detail)  # triage suffix on ✓ done / ✗ FAILED history rows
    if action:
        line += "  " + paint.c(code, action)
    if r.get("suffix"):  # cross-ns run: its namespace (+ loading/reclaim note)
        line += "  " + paint.c(r.get("suffix_code", "2"), r["suffix"])
    return line.rstrip()


_BAR_RUN_RE = re.compile("━+")


def _color_bars(s: str, color_fn) -> str:
    """Apply a structural color to ONLY the `━` rule segments of a bar, leaving the header TEXT in the default
    foreground so the words stay readable on ANY terminal background. The header text never contains `━`, so
    coloring every maximal `━`-run cleanly tints just the leading + trailing rule segments. PURE.
    """
    return _BAR_RUN_RE.sub(lambda m: color_fn(m.group(0)), s)


def _rule(name: str, gtype: str, *, extra: str = "", label: str = "") -> str:
    lbl = f"{label} " if label else ""  # an ALL-CAPS level label (e.g. CLUSTER) — uniform with INSTALLED/RUN
    prefix = f"━━ {lbl}{name}  " + (f"[{gtype}]  " if gtype else "") + (f"{extra}  " if extra else "")
    return prefix + "━" * max(3, 64 - _vlen(prefix))


def _capacity_text(e) -> str:
    """This cluster's capacity, HEADLINED by free WHOLE nodes (the schedulable unit — a tp=4/8 run takes an
    entire node's GPUs), NOT free GPUs. Free-GPU counts mislead: `4 free` fragmented as 2+2 across two busy
    nodes can't launch a single tp=4 run, so the biggest-free-node figure tells the operator the max run size
    that can actually schedule right now (0g → nothing launchable). Layout:
        nodes 5/17 free (biggest free: 8g) · ours 1 node (4g) · 136/140 gpu
    `nodes 5/17 free` = free whole-nodes / total GPU nodes; `biggest free: 8g` = the largest single-node free
    count; `ours …` = what WE hold (nodes + GPUs); the trailing `136/140 gpu` keeps the raw GPU total terse.
    Degrades gracefully: no node list → speak in GPUs; nodes known but pods -A forbidden → `free n/a`.
    """
    ours_gpu = e["ours_gpu"]
    total, occ = e.get("total"), e.get("occupied")
    nc = e.get("nodes")
    gpu_tail = f" · {occ}/{total} GPUs" if (total is not None and occ is not None) else ""
    if nc is None:
        # no node list at all (RBAC) → can't speak in nodes; fall back to the raw GPU total if we have it.
        if total is not None and occ is not None:
            return f"nodes n/a · {occ}/{total} GPUs · ours {_gpus(ours_gpu)}"
        return f"nodes n/a · ours {_gpus(ours_gpu)}"
    tn, fn, bf, on = (
        nc["total_nodes"],
        nc["free_nodes"],
        nc["biggest_free"],
        nc["ours_nodes"],
    )
    if fn is None:  # nodes known, per-node occupancy unreadable (pods -A ✗)
        head = f"nodes {tn} total (free n/a)"
    else:
        head = f"nodes {fn}/{tn} free (biggest free: {_gpus(bf)})"
    if on is not None:
        ours = f"ours {on} node{'s' if on != 1 else ''} ({_gpus(ours_gpu)})"
    else:
        ours = f"ours {_gpus(ours_gpu)}"
    return f"{head} · {ours}{gpu_tail}"


def _ns_line(e, *, fast: bool) -> str:
    """Level-2 line: the namespace + this cluster's capacity (a `·`-joined summary is fine for a header)."""
    if fast:
        cap = f"ours {_gpus(e['ours_gpu'])} (capacity skipped)"
    else:
        cap = _capacity_text(e)
    cpu = fmt_cores(e.get("ours_cpu", 0))
    if cpu:
        cap += f" · cpu {cpu}"
    prof = f" · {e['profiles']} profiles" if e.get("profiles", 1) > 1 else ""
    return f"ns {e['namespace'] or '—'} · {cap}{prof}{_fresh_note(e.get('stale', ''))}"


# ─── 3-STAGE JOURNEY (init → install → run) ──────────────────────────────────────────────────────
# The S-tier fleet view: one copy/paste command lets a user watch their WHOLE journey per cluster —
#   STAGE 1  cluster-init readiness  ← cluster-profiles/.state/<profile>.readiness.json  (wizard init)
#   STAGE 2  recipe install/staging  ← cluster-profiles/.state/<profile>.install.jsonl   (install + `run` stage)
#   STAGE 3  live execution          ← the live Jobs already gathered by fleet.sh (this renderer)
# Stages 1 & 2 read LOCAL stamp files, so they render even when the cluster context is UNREACHABLE
# (auth-robust by construction — that's the purpose of showing them). Stage 3 needs the cluster,
# so it reads `unknown` when a context can't be probed. All stage logic below is PURE + selftested.

# The step_key ordering that maps a per-cell install stamp to a coarse "is this cell set up?" verdict.
_INSTALL_OK_PREFLIGHT = (
    "pass",
    "warn",
)  # preflight passed (warn is a non-blocking advisory) → staged ✓


def read_readiness_stamp(profiles_dir, name):
    """S1 source: cluster-profiles/.state/<name>.readiness.json → dict|None. Never raises (a missing or
    malformed stamp just means init has not run for this profile)."""
    if not profiles_dir:
        return None
    p = Path(profiles_dir) / ".state" / f"{name}.readiness.json"
    return _read_json(p) if p.is_file() else None


def read_install_stamps(profiles_dir, name):
    """S2 source: cluster-profiles/.state/<name>.install.jsonl → {cell: latest_rec} (append-only log, the
    last line per cell wins — mirrors install.read_install_stamps). Never raises."""
    out: dict = {}
    if not profiles_dir:
        return out
    p = Path(profiles_dir) / ".state" / f"{name}.install.jsonl"
    if not p.is_file():
        return out
    try:
        for ln in p.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except (json.JSONDecodeError, ValueError):
                continue
            cell = rec.get("cell")
            if cell:
                out[cell] = rec  # later line wins (idempotent re-install)
    except OSError:
        return out
    return out


def stage1_state(readiness):
    """PURE. Reduce the readiness stamp to (key, glyph, text). key ∈ {ready, blocked, unknown}."""
    if not readiness:
        return ("unknown", "?", "init ? not run")
    counts = readiness.get("level_counts") or {}
    if readiness.get("run_ready"):
        warn = int(counts.get("WARN", 0) or 0)
        return ("ready", "✓", "init ✓ run-ready" + (f" ({warn} warn)" if warn else ""))
    fail = int(counts.get("FAIL", 0) or 0)
    return (
        "blocked",
        "❌",
        f"init ❌ NOT ready ({fail} fail)" if fail else "init ❌ NOT ready",
    )


def stage2_state(installs):
    """PURE. Reduce the {cell: install_rec} map to (key, glyph, text). key ∈ {ok, partial, none}. A cell
    counts as staged-✓ when its preflight passed (pass/warn); fail/skipped/blank → still-blocked.
    """
    total = len(installs)
    if total == 0:
        return ("none", "·", "install · none staged")
    ok = sum(1 for r in installs.values() if str(r.get("preflight", "")).lower() in _INSTALL_OK_PREFLIGHT)
    blocked = total - ok
    if blocked == 0:
        return ("ok", "✓", f"install ✓ {ok}/{total} cell{'' if total == 1 else 's'}")
    return ("partial", "⚠", f"install ⚠ {ok}/{total} ready ({blocked} blocked)")


def stage3_state(active_count, reachable):
    """PURE. Reduce live execution to (key, glyph, text). key ∈ {active, idle, unknown}."""
    if not reachable:
        return ("unknown", "?", "run ? cluster unreachable")
    if active_count > 0:
        return ("active", "●", f"run ● {active_count} active")
    return ("idle", "·", "run · idle")


def _stage_paint(key, text, paint):
    """Color a stage token by its key. Complete/idle stages are DIMMED (collapsed to a quiet ✓/·); the
    stage that needs attention — a live run, a block, an un-run step — is emphasized so the eye lands on
    'what's active' (fleet.sh's ethos)."""
    if key in ("ready", "ok"):
        return paint.green(text)
    if key == "active":
        return paint.bold(paint.green(text))
    if key == "blocked":
        return paint.red(text)
    if key == "partial":
        return paint.yellow(text)
    return paint.dim(text)  # idle / none / unknown → quiet


def journey_line(readiness, installs, active_count, reachable, paint):
    """PURE (bar `paint`). One compact `①②③` journey line for a cluster: init → install → run. Done/idle
    stages collapse to a quiet token; the active/blocked/actionable stage is emphasized + colored.
    """
    s1 = stage1_state(readiness)
    s2 = stage2_state(installs)
    s3 = stage3_state(active_count, reachable)
    segs = []
    for i, (key, _glyph, text) in enumerate((s1, s2, s3), start=1):
        segs.append(f"{'①②③'[i - 1]} " + _stage_paint(key, text, paint))
    return "journey  " + paint.dim(" · ").join(segs)


def fleet_journey_summary(stage_keys, paint):
    """PURE. stage_keys = list of (s1key, s2key, s3key) per cluster → one fleet-wide JOURNEY headline that
    rolls up the three stages: how many clusters are init-ready, how many cells are staged, how many runs
    are live. Answers 'where is my whole fleet in the journey' in one line."""
    n = len(stage_keys)
    ready = sum(1 for k1, _, _ in stage_keys if k1 == "ready")
    blocked1 = sum(1 for k1, _, _ in stage_keys if k1 == "blocked")
    unknown1 = sum(1 for k1, _, _ in stage_keys if k1 == "unknown")
    staged_clusters = sum(1 for _, k2, _ in stage_keys if k2 in ("ok", "partial"))
    active = sum(1 for _, _, k3 in stage_keys if k3 == "active")
    init_seg = f"init: {ready}/{n} ready"
    if blocked1:
        init_seg += " · " + paint.red(f"{blocked1} blocked")
    if unknown1:
        init_seg += paint.dim(f" · {unknown1} not-run")
    inst_seg = f"install: {staged_clusters}/{n} cluster{'' if staged_clusters == 1 else 's'} staged"
    run_seg = f"run: {active} active"
    return paint.bold("JOURNEY  ") + paint.dim(init_seg + "   |   " + inst_seg + "   |   " + run_seg)


def _cluster_active_count(entry, actives):
    """How many fleet-wide ACTIVE runs belong to this cluster (by name). Pure over the precomputed list."""
    return sum(1 for a in actives if a["cluster"] == entry["name"])


# ─── STAGES VIEW — signal-over-noise reorganization of `--stages` ─────────────────────────────────
# HIERARCHY: CLUSTER → NAMESPACE → the journey STAGES as labeled sub-sections (INIT / INSTALL / RUN).
# A cluster is shown IN FULL only when it has activity OF OURS in some stage — an active/failed run, an
# orphan holding GPU, a blocked init, or a partially-staged install. An idle-but-connected cluster collapses
# to ONE compact line (name · connected · free-capacity) so the operator still sees it exists; an idle +
# unreachable cluster folds into a `+N unreachable` tail (never a full noisy block). Under a full cluster the
# three stages render ONLY when they have content: INIT (only when blocked/not-ready — a ready cluster is
# silent), INSTALL (only when a cell is still blocked), RUN (active runs · FAILED runs with why+logs · orphans
# holding GPU). INIT/INSTALL read LOCAL .state stamps, so they render even for an UNREACHABLE cluster
# (auth-robust). The dropped noise: the `journey ①②③` line, idle-server / helper-job / parked counts, the
# "no active runs of ours" filler, and the raw node/gpu census on idle clusters. PURE.

_SEC_LABEL = "    "  # a stage label (INIT/INSTALL/RUN) — indent 4, under the level-2 `ns` line at 2
_SEC_ROW = "      "  # rows under a stage label — indent 6


def _install_blocked_cells(installs):
    """Cells whose preflight did NOT pass (fail/skipped/blank) — the ones still blocking install. PURE."""
    return sorted(
        c for c, r in (installs or {}).items() if str(r.get("preflight", "")).lower() not in _INSTALL_OK_PREFLIGHT
    )


# ── INSTALLED inventory ("what can be run here") — per-cluster, from the local install stamps ─────────────
# One row per cell staged on this cluster: ✓ ready (staged + preflight pass) · ⚠ warn/needs-input · ✗ FAILED.
# Reads ONLY the local .state/<cluster>.install.jsonl stamps (written by BOTH `install` and the `run` inline
# stage path), so it renders even when the cluster is unreachable. All PURE + selftested.
_INSTALL_GLYPH = {"ready": "✓", "warn": "⚠", "failed": "✗"}
_INSTALL_RANK = {
    "failed": 0,
    "warn": 1,
    "ready": 2,
}  # attention-first: FAILED, then warn, then ready


def _staged_ok_reason(rec):
    """PURE. Pull (ok, reason) for the single stage step in an install-stamp record. ok ∈ {True, False, None}
    (None = skipped / needs-input); reason is a short note when present (e.g. 'missing task-source').
    """
    for v in (rec.get("staged") or {}).values():
        if isinstance(v, dict):
            return v.get("ok"), (v.get("reason") or "")
    return None, ""


def _install_cell_state(rec):
    """PURE. One install-stamp record → (state, why, fix_kind). state ∈ {ready, warn, failed}:
    ready  = staged ok + preflight pass  → runnable now (✓)
    warn   = staged but advisory (preflight WARN) or needs-input (stage ok=None)  → ⚠
    failed = staging failed (stage ok=False) or preflight fail/blank  → ✗ (carries a why + fix hint).
    """
    ok, reason = _staged_ok_reason(rec)
    pf = str(rec.get("preflight", "")).lower()
    if ok is False:
        return ("failed", "staging failed", "install")
    if pf == "fail":
        return ("failed", "preflight blocked", "install")
    if pf == "warn":
        return ("warn", "preflight advisory (WARN)", "")
    if ok is None:
        return ("warn", reason or "needs input", "needs-input")
    if pf in _INSTALL_OK_PREFLIGHT:  # staged ok + preflight pass → the runnable set
        return ("ready", "", "")
    return (
        "failed",
        f"preflight {pf or 'unknown'}",
        "install",
    )  # staged but never validated → not run-ready


def _install_inventory(installs):
    """PURE. {cell: rec} → (rows, counts). rows sorted attention-first (FAILED → warn → ready, then by cell).
    Each row: {cell, state, glyph, why, fix_kind}. counts = {ready, warn, failed, total}.
    """
    rows, counts = [], {"ready": 0, "warn": 0, "failed": 0}
    for cell, rec in (installs or {}).items():
        state, why, fixk = _install_cell_state(rec)
        counts[state] += 1
        rows.append(
            {
                "cell": cell,
                "state": state,
                "glyph": _INSTALL_GLYPH[state],
                "why": why,
                "fix_kind": fixk,
            }
        )
    rows.sort(key=lambda r: (_INSTALL_RANK[r["state"]], r["cell"]))
    counts["total"] = len(rows)
    return rows, counts


def _install_disp(cell):
    """PURE. Compact display form of a cell _path for the INSTALLED list (drop the common 'recipes/' prefix)."""
    c = str(cell)
    return c[len("recipes/") :] if c.startswith("recipes/") else c


def _install_fix(fix_kind, cluster, cell):
    """PURE. The concrete fix hint for a non-ready INSTALLED row."""
    if fix_kind == "needs-input":
        return f"llmb-k8s install --cluster {cluster} --recipes {Path(cell).name}"
    return f"llmb-k8s install --cluster {cluster}"


def _install_summary(counts, source=""):
    """PURE. One-line INSTALLED header summary. e.g. '3 cells · 2 ready · 1 FAILED · live'.

    `cells`, not `installed`: this header sits under the word INSTALLED (so "installed" said it twice) and
    beside MODELS and STORAGE headers counting models and PVCs — three sections whose numbers are in three
    different units, which only reads if each names its own."""
    summ = f"{counts['total']} cell{'' if counts['total'] == 1 else 's'} · {counts['ready']} ready"
    if counts["warn"]:
        summ += f" · {counts['warn']} warn"
    if counts["failed"]:
        summ += f" · {counts['failed']} FAILED"
    if source:
        summ += f" · {source}"
    return summ


def inventory_note(unavailable: bool, fast: bool) -> str:
    """PURE. WHY the model-cache half of the INSTALLED inventory is missing — '' when the PVC read landed.

    A skipped or failed PVC read is reported as unavailable rather than as an empty inventory.
    """
    if not unavailable:
        return ""
    if fast:
        return "model caches NOT READ (--fast skips the PVC read) — drop --fast for the full inventory"
    return "model-cache read UNAVAILABLE (PVC list failed / RBAC-forbidden) — " "check: kubectl auth can-i list pvc -A"


# ── LIVE DISCOVERY — the ground truth "what's installed here" when a cluster is REACHABLE ─────────────────
# Stamps are written going-forward only, so a cluster installed BEFORE the stamp feature has none. When a
# cluster is connected we DISCOVER its recipe-installed resources from the SAME namespaced Deployments + Jobs
# fleet already fetched (no extra kubectl call — cheap, read-only): server Deployments labelled
# `llmb.nvidia.com/cell` (a cell deployed here, running or parked) and model-download Jobs labelled
# `component=model-download` (a model cached / downloading / failed). Live discovery is the source of truth
# when connected; the local install stamp is the offline/unreachable fallback. All PURE + selftested.
_CELL_NAME_MAP_CACHE = None
_CATALOG_CACHE = None


def load_catalog(catalog_path=None):
    """catalog.json as a list of cells — the EXPECTED-model set's source of truth (`llmb-k8s fleet` must be
    able to say "the collection needs 3 models" without a cluster). Missing/malformed → [] and the ledger
    then renders only evidence-derived rows; never raises. The default path is cached.
    """
    global _CATALOG_CACHE
    default = catalog_path is None
    if default and _CATALOG_CACHE is not None:
        return _CATALOG_CACHE
    p = Path(catalog_path) if catalog_path else (Path(__file__).resolve().parent.parent / "catalog.json")
    try:
        cat = json.loads(p.read_text())
        cat = cat if isinstance(cat, list) else []
    except Exception:
        cat = []
    if default:
        _CATALOG_CACHE = cat
    return cat


def load_cell_name_map(catalog_path=None):
    """{cell_name → _path} from catalog.json — normalizes a live-discovered cell's label (the recipe NAME) to
    the _path key install stamps use, so a live cell and its stamp dedup + display identically. Missing /
    malformed catalog → {} (discovery then keys by raw name). Never raises. The default path is cached.
    """
    global _CELL_NAME_MAP_CACHE
    default = catalog_path is None
    if default and _CELL_NAME_MAP_CACHE is not None:
        return _CELL_NAME_MAP_CACHE
    m = {it["name"]: it["_path"] for it in load_catalog(catalog_path) if it.get("name") and it.get("_path")}
    if default:
        _CELL_NAME_MAP_CACHE = m
    return m


L_MODEL_REVISION = "llmb.nvidia.com/model-revision"  # download Job AND the PVC it filled → the revision
L_DOWNLOAD_COMPLETE = "llmb.nvidia.com/download-complete"  # PVC → stamped ONLY after a successful download

# THE REVISION + STAMP-PROVENANCE RULES LIVE IN `cache_inventory`, and are re-exported here under the names
# the renderer uses. ONE implementation, deliberately: a second copy of "does this label agree with the pin?"
# Keep the inventory resolver aligned with the claim used by serving.
REV_DISPLAY = CI.REV_DISPLAY  # every revision renders at EXACTLY this width
_short_rev = CI.short_rev  # (12-char revision, 'WRONG REVISION' | '') — see cache_inventory
_stamp_provenance = CI.stamp_provenance  # 'job' | 'hand' | 'unknown' — the ONLY gate on a ✓ from a stamp


_MODEL_PIN_CACHE = None
_PIN_RE = re.compile(r"^\s*model_revision:\s*[\"']?([A-Za-z0-9._-]+)", re.M)


def load_model_pins(catalog_path=None, root=None):
    """{model: pinned revision} — the revision each catalog model's recipes PIN (`serving.model_revision`).

    Not in catalog.json (it carries the envelope, not the serving block), so it is read from the cells
    themselves — a regex, not a YAML parse, because this runs on every frame and the field is one flat
    scalar. A model whose recipe cannot be read gets an EXPLICIT empty pin (see below), and an empty pin
    makes the ledger REFUSE to certify a stamp rather than assume it agrees. The default path is cached.
    """
    global _MODEL_PIN_CACHE
    default = catalog_path is None and root is None
    if default and _MODEL_PIN_CACHE is not None:
        return _MODEL_PIN_CACHE
    base = Path(root) if root else Path(__file__).resolve().parent.parent
    p = Path(catalog_path) if catalog_path else (base / "catalog.json")
    pins: dict = {}
    unread: set = set()  # models whose recipe we could NOT read — an UNKNOWN pin, kept as one
    try:
        cat = json.loads(p.read_text())
    except Exception:
        cat = []
    seen: dict = {}  # model → every DISTINCT pin its cells declare
    for it in (cat if isinstance(cat, list) else []):
        m, pth = it.get("model"), it.get("_path")
        if not m or not pth:
            continue
        try:
            txt = (base / pth / "recipe.yaml").read_text()
        except Exception:
            unread.add(m)  # recorded, not dropped — see below
            continue
        hit = _PIN_RE.search(txt)
        if hit:
            seen.setdefault(m, set()).add(hit.group(1))
    for m, vals in seen.items():
        # ONE pin, or NO pin. Taking the first readable cell's revision meant that when two cells of a model
        # disagreed, every claim was certified against a revision half of them never asked for — rendering a
        # confident `WRONG REVISION` at the claim, when the disagreement is in the RECIPES. An ambiguous pin
        # is an unknown pin, and an unknown pin refuses to certify rather than accusing the cluster.
        pins[m] = next(iter(vals)) if len(vals) == 1 else ""
    # An unread recipe leaves an EXPLICIT empty pin rather than no entry at all. Downstream, an empty pin
    # means "cannot be certified against anything": _stamp_provenance returns 'unknown' and the ledger
    # renders ~ instead of ✓. Fail-safe by construction — a read that did not land can only ever REMOVE a
    # green tick here, never add one.
    for m in unread:
        pins.setdefault(m, "")
    if default:
        _MODEL_PIN_CACHE = pins
    return pins


def _pvc_size(pvc: dict) -> str:
    """PURE. A PVC's bound capacity ('721Gi'), preferring status (what was actually provisioned) over the
    request. '' when unknown."""
    st = (pvc.get("status") or {}).get("capacity") or {}
    rq = ((pvc.get("spec") or {}).get("resources") or {}).get("requests") or {}
    return str(st.get("storage") or rq.get("storage") or "")


_QTY_SUFFIX = {
    "Ki": 1024,
    "Mi": 1024**2,
    "Gi": 1024**3,
    "Ti": 1024**4,
    "Pi": 1024**5,
    "K": 1000,
    "M": 1000**2,
    "G": 1000**3,
    "T": 1000**4,
    "P": 1000**5,
}


def _qty_bytes(q) -> int:
    """PURE. A k8s storage quantity ('50Gi', '1200Gi', '2T') → bytes. 0 when unparseable — an unreadable size
    must not fabricate a total, and must never raise in the middle of a render."""
    s = str(q or "").strip()
    for suf in ("Ki", "Mi", "Gi", "Ti", "Pi", "K", "M", "G", "T", "P"):
        if s.endswith(suf):
            try:
                return int(float(s[: -len(suf)]) * _QTY_SUFFIX[suf])
            except ValueError:
                return 0
    try:
        return int(float(s))
    except ValueError:
        return 0


def _fmt_bytes(n: int) -> str:
    """PURE. bytes → a terse binary size ('3.7Ti', '450Gi'). '' for 0 (say nothing rather than '0')."""
    if not n:
        return ""
    for unit, div in (
        ("Pi", 1024**5),
        ("Ti", 1024**4),
        ("Gi", 1024**3),
        ("Mi", 1024**2),
    ):
        if n >= div:
            v = n / div
            return f"{v:.1f}{unit}" if v < 100 else f"{v:.0f}{unit}"
    return f"{n}B"


CONTROL_PVC_NAME = "llmb-control"  # the shared per-namespace RWX control PVC (submit.sh CONTROL_PVC)
# Every creator names the output claim ${NAME}-artifacts (submit.sh / sweep.sh); the
# -rwx variant is the profile-gated RWX form. MUST stay identical to reclaim_storage.ARTIFACT_SUFFIXES — the
# sweeper deletes exactly what this pane calls run output, and selftest_fleet pins the two together.
ARTIFACTS_PVC_SUFFIXES = ("-artifacts", "-artifacts-rwx")


def _pvc_role(name) -> str:
    """PURE. WHAT A PVC IS FOR — 'artifacts' | 'control' | 'cache'. The three roles are told apart by NAME
    because none of the three creators labels the volume (`submit.sh` and `sweep.sh`
    each `kubectl apply` a bare 4-line manifest), while the naming is a hard convention all of them share:
    `${NAME}-artifacts` for the per-run OUTPUT claim, the literal `llmb-control` for the wrapper's RWX control
    claim, anything else is the profile's MODEL_CACHE_PVC (the weights).

    Only cache volumes may receive model-download status or backfill guidance.

    RELATION TO reclaim_storage.classify (which SWEEPS what this pane REPORTS): the artifacts verdict is
    identical by construction (shared suffix list, pinned by selftest) — what fleet calls run output is
    exactly what the sweeper may delete. The two differ, deliberately, on the residue: the sweeper splits
    non-artifacts into model-cache / other and refuses to touch 'other', because for a DESTRUCTIVE tool an
    unrecognised volume must be left alone; fleet reports an unrecognised volume as a cache, because for a
    REPORTING tool the honest ⚠ 'contents unverified' is the safe verdict and silence is not.
    """
    n = str(name or "")
    if n.endswith(ARTIFACTS_PVC_SUFFIXES):
        return "artifacts"
    if n == CONTROL_PVC_NAME:
        return "control"
    return "cache"


def _obj_is_live(o) -> bool:
    """PURE. Is this workload object a run that still EXISTS, as opposed to its corpse?

    A Succeeded/Failed Pod and a completed Job are the RESIDUE of a finished run — the namespace keeps them
    around until GC. Counting them as live references was the difference between a leak report of 71 and one
    of 0: 128 lingering pods in the GB300 campaign namespace still name 82 artifacts claims between them, so
    'is anything pointing at this PVC?' answered yes for every single one and the signal vanished.

    A Deployment is live regardless of replica count: a PARKED server is an installed cell that will mount its
    artifacts volume again on the next run. Unknown kinds are treated as live (fail safe: never accuse).

    `kind` is inferred structurally when absent — some kubectl versions omit it on the items of a single-kind
    list, and defaulting those to live would silently switch the whole leak signal off (a pod phase is the one
    field only a Pod has). Silent-off is the failure mode this pane exists to not have.
    """
    kind, st = o.get("kind", ""), (o.get("status") or {})
    if not kind and st.get("phase") and "template" not in (o.get("spec") or {}):
        kind = "Pod"
    if kind == "Pod":
        return st.get("phase") not in ("Succeeded", "Failed")
    if kind == "Job":
        return int(st.get("active", 0) or 0) > 0 or not (
            int(st.get("succeeded", 0) or 0) or int(st.get("failed", 0) or 0)
        )
    return True


def referenced_claims(*objsets):
    """PURE. {(ns, claimName)} for every PVC mounted by a workload that is STILL A RUN — see _obj_is_live.
    Each argument is a list of objects or a {'items': [...]} wrapper (None tolerated).

    Used only to AGE artifacts volumes: a `<cell>-artifacts` claim no live workload references is left over
    from a run that has finished. Conservative in the direction that matters — a Job still marked active, and
    any object whose kind we do not recognise, counts as live — so an in-flight campaign is never accused of
    leaking the volume it is writing to."""
    out = set()
    for os_ in objsets:
        items = (os_ or {}).get("items") or [] if isinstance(os_, dict) else (os_ or [])
        for o in items:
            if not _obj_is_live(o):
                continue
            ns = (o.get("metadata") or {}).get("namespace") or ""
            spec = o.get("spec") or {}
            podspec = ((spec.get("template") or {}).get("spec") or {}) or spec
            for v in podspec.get("volumes") or []:
                claim = (v.get("persistentVolumeClaim") or {}).get("claimName") or ""
                if claim:
                    out.add((ns, claim))
    return out


def _artifacts_rollup(arts, live_claims):
    """PURE. The per-namespace ARTIFACTS rollup rows: [{kind:'artifacts', ns, name, state, why}], one per ns.

    WHY A ROLLUP AND NOT A FILTER. Every run creates `<cell>-artifacts` (50Gi) and NOTHING ever deletes it —
    so these volumes accumulate one per run, forever. Silently dropping them from INSTALLED would fix the 82
    rows and simultaneously hide a real, growing storage leak (one namespace on GB300 holds 75 of them, 71 of
    which no live workload references). Collapsing them to ONE line per namespace does both jobs: the pane
    stops lying about what they are, and the leak becomes a number you can act on.

    `live_claims` = {(ns, claim)} referenced by a live pod/Deployment/Job (see referenced_claims); None means
    we could not tell, and then NO leak count is asserted — an unknown is never reported as zero.

    States, on the same honest-signal rule as the caches:
      failed — an artifacts PVC a LIVE workload references is not Bound → that run cannot write results
      warn   — leftovers present (a leak worth reclaiming), or unbound leftovers
      ready  — every artifacts PVC here is Bound and accounted for by a live run"""
    by_ns: dict = {}
    for ns, nm, phase, size in arts:
        g = by_ns.setdefault(ns, {"n": 0, "bytes": 0, "orphan": 0, "unbound": 0, "live_unbound": 0})
        live = None if live_claims is None else ((ns, nm) in live_claims)
        g["n"] += 1
        g["bytes"] += _qty_bytes(size)
        if phase != "Bound":
            g["unbound"] += 1
            if live:
                g["live_unbound"] += 1
        if live is False:
            g["orphan"] += 1
    out = []
    for ns in sorted(by_ns):
        g = by_ns[ns]
        parts = [f"{g['n']} PVC{'' if g['n'] == 1 else 's'}"]
        if _fmt_bytes(g["bytes"]):
            parts.append(_fmt_bytes(g["bytes"]))
        parts.append("run OUTPUT, not model weights")
        if live_claims is None:
            state = "warn" if g["unbound"] else "ready"
        elif g["live_unbound"]:
            state = "failed"
        elif g["orphan"]:
            state = "warn"
        else:
            state = "ready"
        if g["live_unbound"]:
            parts.append(f"{g['live_unbound']} NOT Bound under a live run — results cannot be written")
        elif g["unbound"]:
            parts.append(f"{g['unbound']} not Bound")
        if live_claims is None:
            parts.append("age unknown (workload read did not land)")
        elif g["orphan"]:
            parts.append(f"{g['orphan']} from runs no longer present — leaked, reclaimable")
        out.append(
            {
                "kind": "artifacts",
                "name": "artifacts",
                "ns": ns,
                "state": state,
                "n": g["n"],
                "bytes": g["bytes"],
                "why": " · ".join(parts),
            }
        )
    return out


def discover_model_caches(pvcs_j, jobs, our_namespaces=None, live_claims=None, pins=None):
    """Build the model-cache PVC inventory for the selected namespaces.

    The inventory is derived from cluster state rather than local stamps. Only PVCs with the model-cache role
    are included; artifact and control volumes are handled separately. ``our_namespaces=None`` disables the
    namespace filter for unit tests, while production callers provide the managed namespace set.

    A claim is ready only when it is Bound and has a download-job stamp. Unstamped, manually stamped, or
    actively downloading claims are warnings; Pending, Lost, or failed-download claims are failures. The
    result describes the claim, not the model contents. If PVCs cannot be read, the function returns an empty
    inventory instead of inferring that caches are absent.
    """
    if pvcs_j is None or "items" not in (pvcs_j or {}):
        return []
    ours = None if our_namespaces is None else frozenset(our_namespaces)
    # (ns, pvc) → the most informative download Job we have for it
    by_claim = {}
    for j in jobs or []:
        lab = _labels(j)
        if COMP_MODEL_DOWNLOAD not in (lab.get(L_COMPONENT2), lab.get(L_COMPONENT)):
            continue
        md = j.get("metadata") or {}
        ns = md.get("namespace", "")
        st = j.get("status") or {}
        done = int(st.get("succeeded", 0) or 0) > 0
        failed = int(st.get("failed", 0) or 0) > 0 and not done
        info = {
            "model": lab.get(L_MODEL_NAME, ""),
            "rev": lab.get(L_MODEL_REVISION, ""),
            "done": done,
            "failed": failed,
            "running": int(st.get("active", 0) or 0) > 0,
        }
        spec = ((j.get("spec") or {}).get("template") or {}).get("spec") or {}
        for v in spec.get("volumes") or []:
            claim = (v.get("persistentVolumeClaim") or {}).get("claimName") or ""
            if claim:
                cur = by_claim.get((ns, claim))
                if cur is None or (info["done"] and not cur["done"]):
                    by_claim[(ns, claim)] = info
    out, arts = [], []
    for c in pvcs_j["items"]:
        md = c.get("metadata") or {}
        ns, nm = md.get("namespace", ""), md.get("name", "")
        if not nm:
            continue
        if ours is not None and ns not in ours:
            continue  # a colleague's PVC — not our inventory, and never our `kubectl label` hint
        phase = (c.get("status") or {}).get("phase", "")
        size = _pvc_size(c)
        role = _pvc_role(nm)
        if role == "artifacts":
            arts.append((ns, nm, phase, size))  # → ONE rollup row per ns, never 75 fake `model:` rows
            continue
        if role == "control":
            # The wrapper's RWX control volume. It holds run STATE, not weights — so it gets no download
            # verdict at all, only "is it usable?", and it is counted under STORAGE, never as an installed
            # model (it was one of the "3 installed · 3 ready" in the reported header). The `✗ llmb-control
            # FAILED — PVC Pending` signal (a Pending control PVC wedges every resilient submit in the ns) is
            # exactly what used to be buried among the 80 look-alike `⚠ model:` rows.
            out.append(
                {
                    "kind": "control",
                    "name": nm,
                    "ns": ns,
                    "bytes": _qty_bytes(size),
                    "state": "failed" if phase != "Bound" else "ready",
                    "why": (
                        f"PVC {phase or 'not Bound'} — resilient submit cannot write run state"
                        if phase != "Bound"
                        else f"Bound{(' · ' + size) if size else ''} · run control plane " f"— not model weights"
                    ),
                }
            )
            continue
        info = by_claim.get((ns, nm))
        lab = md.get("labels") or {}
        pin = (pins or {}).get(lab.get(L_MODEL_NAME) or "", "")
        # A CLAIM ROW DESCRIBES THE VOLUME, NOT A MODEL. It used to describe a model — and when the claim's
        # stamp named none, `model = st_model or info and info["model"] or nm` put the PVC's OWN NAME in the
        # model slot. That is how `✓ model: nemotron-ultra-nvfp4-cache … verified by PVC stamp` reached an
        # operator's screen. Which MODELS are on a claim is now answered by the ledger (cache_inventory),
        # which can represent two models in one claim and a model in no claim; this row answers only "is this
        # volume usable, and does anything attribute it?".
        prov = _stamp_provenance(lab, pin)
        fact = {
            "ns": ns,
            "name": nm,
            "phase": phase,
            "size": size,
            "labels": dict(lab),
            "job": dict(info) if info else None,
        }
        row = {
            "kind": "cache",
            "name": nm,
            "ns": ns,
            "fact": fact,
            "size": size,
            "bytes": _qty_bytes(size),
        }
        if phase != "Bound":
            row.update(state="failed", why=f"PVC {phase or 'not Bound'} — cannot serve weights")
        elif info and info["failed"]:
            # A FAILURE IS NEVER HIDDEN, even when an older revision is stamped complete: the stamp vouches
            # for the revision it names, not for the one that just failed.
            prev = (
                f" · previously stamped {lab.get(L_MODEL_NAME) or '?'} "
                f"@{_short_rev(lab.get(L_MODEL_REVISION, ''), pin)[0]}"
                if prov != "unknown" and lab.get(L_MODEL_REVISION)
                else ""
            )
            row.update(state="failed", why=f"model download FAILED into this claim{prev}")
        elif info and info["running"]:
            row.update(
                state="warn",
                why=f"downloading{(' · ' + size) if size else ''} · download Job in flight",
            )
        elif prov == "hand":
            # THE REPORTED ROW. `download-complete=true` with no model-name (or a revision that is not the
            # 12 chars the template writes) is PROOF a download Job did not write it — so it is named as a
            # CLAIM, reads ⚠, and says the stamp names no model. It never occupies the model slot again.
            row.update(
                state="warn",
                why=f"Bound{(' · ' + size) if size else ''} · stamped download-complete but the stamp "
                f"does NOT name a model — not written by a download Job",
            )
        elif prov == "job":
            row.update(
                state="ready",
                why=f"Bound{(' · ' + size) if size else ''} · downloaded · stamped by a download Job "
                f"at the pinned revision",
            )
        elif lab.get(L_DOWNLOAD_COMPLETE) == "true":
            # Job-SHAPED (it names a model, in the one format only the Job writes) but we hold no pin to check
            # it against. The CLAIM is attributed and usable; WHICH revision it holds is a MODEL question, and
            # the ledger answers that one `~`, not `✓`. Two questions, two answers, neither borrowed.
            row.update(
                state="ready",
                why=f"Bound{(' · ' + size) if size else ''} · downloaded · carries a download-Job "
                f"stamp (no pinned revision to check it against)",
            )
        else:
            # Bound, but nothing vouches for the contents. NOT green — this is exactly the empty-PVC trap.
            row.update(
                state="warn",
                why=f"Bound{(' · ' + size) if size else ''} · contents not attributed "
                f"(no download record — see MODELS)",
            )
        out.append(row)
    return out + _artifacts_rollup(arts, live_claims)


def claim_facts(discovered, ns):
    """PURE. The model-cache CLAIM facts for ONE namespace, in the shape `cache_inventory.reconcile` joins on.

    Returns None when the PVC read did not land (`discovered` is None) — that None is load-bearing: it is the
    difference between "no claim holds this model" (`○ MISSING`) and "we could not look" (`? UNKNOWN`), and
    collapsing the two is the absence-as-default this pane exists to refuse."""
    if discovered is None:
        return None
    return [d["fact"] for d in discovered if d.get("kind") == "cache" and d.get("fact") and d["fact"].get("ns") == ns]


def download_job_facts(jobs, our_namespaces=None):
    """PURE. {claim, model, rev, done, failed, running} per model-download Job — so a model whose Job is
    visible but whose PVC is not (GC'd claim, RBAC on PVCs only) still gets a ledger row instead of vanishing
    into `○ MISSING`."""
    ours = None if our_namespaces is None else frozenset(our_namespaces)
    out = []
    for j in jobs or []:
        lab = _labels(j)
        if COMP_MODEL_DOWNLOAD not in (lab.get(L_COMPONENT2), lab.get(L_COMPONENT)):
            continue
        jns = (j.get("metadata") or {}).get("namespace", "")
        if ours is not None and jns not in ours:
            continue
        st = j.get("status") or {}
        done = int(st.get("succeeded", 0) or 0) > 0
        spec = ((j.get("spec") or {}).get("template") or {}).get("spec") or {}
        claims = [((v.get("persistentVolumeClaim") or {}).get("claimName") or "") for v in (spec.get("volumes") or [])]
        out.append(
            {
                "ns": jns,
                "claim": next((c for c in claims if c), ""),
                "model": lab.get(L_MODEL_NAME, ""),
                "rev": lab.get(L_MODEL_REVISION, ""),
                "done": done,
                "failed": int(st.get("failed", 0) or 0) > 0 and not done,
                "running": int(st.get("active", 0) or 0) > 0,
            }
        )
    return out


def _merge_installed(cells_and_models, caches):
    """PURE. Join the Job-derived inventory with the PVC-derived storage rows.

    The Job-derived `kind: "model"` rows are DROPPED here, not merged: "which models are here?" is now
    answered once, by the catalog-keyed ledger (`cache_inventory.reconcile`), which reads the SAME download
    Jobs via `download_job_facts` plus the claims and the catalog. Keeping a second, Job-only model row
    alongside it re-created the duplicate this function used to spend its time de-duplicating — and that
    second row could only ever be keyed by whatever the Job's label said, which is exactly the key that
    cannot represent two models in one claim or a model in none."""
    return [d for d in cells_and_models if d.get("kind") != "model"] + caches


def discover_installed(deploys_j, jobs_j):
    """PURE. LIVE-discover recipe-installed resources on a CONNECTED cluster from its already-fetched
    namespaced Deployments + Jobs (no extra kubectl call). The ground truth for 'what's installed here':
      • server Deployments labelled `llmb.nvidia.com/cell` → that CELL is deployed (running or parked-at-0)
      • model-download Jobs (component=model-download) → that MODEL is cached (Complete) / downloading / failed
    Returns a list of {kind, name, ns, state, why}: kind ∈ {cell, model}, state ∈ {ready, warn, failed}. Each
    item carries the NAMESPACE it is deployed in, so the INSTALLED inventory can be grouped per namespace.
    """
    out, seen_cells, seen_models = [], set(), set()
    for d in (deploys_j or {}).get("items", []):
        md = d.get("metadata") or {}
        cell = _labels(d).get(L_CELL)
        ns = md.get("namespace", "")
        if not cell or (ns, cell) in seen_cells:
            continue
        seen_cells.add((ns, cell))
        spec_rep = (d.get("spec") or {}).get("replicas")
        ready = int((d.get("status") or {}).get("readyReplicas", 0) or 0)
        parked = (spec_rep == 0) or (ready == 0)
        out.append(
            {
                "kind": "cell",
                "name": cell,
                "ns": ns,
                "state": "ready",
                "why": "installed · server parked" if parked else "server deployed",
            }
        )
    for j in (jobs_j or {}).get("items", []):
        lab = _labels(j)
        if COMP_MODEL_DOWNLOAD not in (lab.get(L_COMPONENT2), lab.get(L_COMPONENT)):
            continue
        ns = (j.get("metadata") or {}).get("namespace", "")
        model = lab.get(L_MODEL_NAME) or ""
        if not model or (ns, model) in seen_models:
            continue
        seen_models.add((ns, model))
        st = j.get("status") or {}
        if int(st.get("succeeded", 0) or 0) > 0:
            out.append(
                {
                    "kind": "model",
                    "name": model,
                    "ns": ns,
                    "state": "ready",
                    "why": "model cached",
                }
            )
        elif int(st.get("failed", 0) or 0) > 0:
            out.append(
                {
                    "kind": "model",
                    "name": model,
                    "ns": ns,
                    "state": "failed",
                    "why": "model download FAILED",
                }
            )
        else:
            out.append(
                {
                    "kind": "model",
                    "name": model,
                    "ns": ns,
                    "state": "warn",
                    "why": "model downloading",
                }
            )
    return out


def build_install_inventory(installs, discovered, name_map=None, fallback_ns=""):
    """PURE. The unified INSTALLED inventory for one cluster: LIVE-discovered resources (ground truth on a
    connected cluster) MERGED with the local install stamps (offline/unreachable fallback). Live wins on a
    cell collision. `name_map` ({cell_name: _path}) normalizes a discovered cell's label (the recipe NAME) to
    the _path key stamps use so a live cell and its stamp dedup + display identically. Each row carries the
    NAMESPACE it belongs to (a live item's own ns; `fallback_ns` for a stamp-only item, whose ns isn't
    recorded). Returns (rows, counts, source): row {cell(key), disp, ns, state, glyph, why, fix_kind, source};
    source ∈ {live, live+stamp, stamp, ''}."""
    if name_map is None:
        name_map = load_cell_name_map()
    rows, live_cell_keys, have_live, have_stamp = [], set(), False, False
    for d in discovered or []:
        # COUNTS ARE KIND-AWARE. `INSTALLED` answers "what CELLS can I run here", and it used to count every
        # PVC in the namespace as one: `llmb-control` — the wrapper's run-state volume, not a recipe and not
        # a model — was one of the "3 installed · 3 ready" in the reported header. Storage is its own
        # section with its own arithmetic (`build_storage_inventory`); a volume never inflates this number.
        if d.get("kind") != "cell":
            continue
        key = name_map.get(d["name"], d["name"])
        live_cell_keys.add(key)
        have_live = True
        rows.append(
            {
                "cell": key,
                "disp": _install_disp(key),
                "ns": d.get("ns", "") or fallback_ns,
                "state": d["state"],
                "glyph": _INSTALL_GLYPH[d["state"]],
                "why": d.get("why", ""),
                "fix_kind": "",
                "hint_pvc": "",
                "source": "live",
            }
        )
    for cell, rec in (installs or {}).items():
        if cell in live_cell_keys:
            continue  # live discovery is ground truth for this cell
        state, why, fixk = _install_cell_state(rec)
        have_stamp = True
        rows.append(
            {
                "cell": cell,
                "disp": _install_disp(cell),
                "ns": fallback_ns,
                "state": state,
                "glyph": _INSTALL_GLYPH[state],
                "why": why,
                "fix_kind": fixk,
                "source": "stamp",
            }
        )
    counts = {"ready": 0, "warn": 0, "failed": 0}
    for r in rows:
        counts[r["state"]] += 1
    counts["total"] = len(rows)
    rows.sort(key=lambda r: (_INSTALL_RANK[r["state"]], r["cell"]))
    source = "live+stamp" if have_live and have_stamp else "live" if have_live else "stamp" if have_stamp else ""
    return rows, counts, source


def build_storage_inventory(discovered, fallback_ns="", ledger_claims=(), ledger_unknown=""):
    """PURE. The STORAGE rows: the VOLUMES in a namespace and whether each is usable — model-cache claims,
    the `llmb-control` wrapper-state PVC, and the per-run artifacts ROLLUP.

    Split out of INSTALLED because the two answer different questions and must count differently. A claim is
    named `claim: <pvc>` and never `model: <pvc>`: a claim is a place weights can live, not a model, and the
    row that blurred the two is the reported defect. WHICH models are in a claim is the MODELS ledger's job.

    `ledger_claims` = the claims the MODEL LEDGER already speaks for. A claim in that set is USABLE storage
    whatever its labels say — an unattributed or hand-stamped stamp is a MODEL question, answered once, over
    there. A claim in NO ledger row is a different finding (weights nothing in the catalog routes to) and
    keeps its ⚠. Splitting it this way is what stops one claim from being reported twice, in two vocabularies.

    Returns (rows, counts): counts = {caches, control, artifacts_pvcs, bytes, failed, warn, ready, total}.
    """
    known = frozenset(ledger_claims or ())
    rows = []
    for d in discovered or []:
        kind = d.get("kind")
        if kind not in ("cache", "control", "artifacts"):
            continue
        ns, size = d.get("ns", ""), d.get("size") or ""
        state, why = d["state"], d.get("why", "")
        if kind == "cache":
            key, disp = f"cache:{ns}:{d['name']}", "claim: " + d["name"]
            if state == "warn" and (ns, d["name"]) in known:
                state = "ready"  # a MODELS row speaks for it — usable storage, question answered there
                why = f"Bound{(' · ' + size) if size else ''} · holds model weights — see MODELS"
            elif state == "warn":
                # "NO catalog model routes here" is a NEGATIVE about the catalog+profile. When either was
                # unreadable we do not hold that fact, and saying it anyway pointed at a claim three models
                # actually route to. Absence of the ledger is not absence of a route.
                why = f"Bound{(' · ' + size) if size else ''} · " + (
                    f"cannot tell what routes here — {ledger_unknown}"
                    if ledger_unknown
                    else "no catalog model routes to this claim — unattributed storage"
                )
        elif kind == "control":
            key, disp = f"control:{ns}:{d['name']}", d["name"]
        else:
            key, disp = (
                "zz-artifacts:" + ns,
                "ARTIFACTS",
            )  # a footnote — sorts last in its state band
        rows.append(
            {
                "cell": key,
                "disp": disp,
                "kind": kind,
                "ns": ns or fallback_ns,
                "state": state,
                "glyph": _INSTALL_GLYPH[state],
                "why": why,
                "fix_kind": "",
                "source": "live",
                "bytes": d.get("bytes", 0) or 0,
                "n": d.get("n", 1 if kind != "artifacts" else 0),
            }
        )
    rows.sort(key=lambda r: (_INSTALL_RANK[r["state"]], r["cell"]))
    return rows, _storage_counts(rows)  # ONE bucketing rule, shared with the per-namespace regroup


def _storage_counts(rows) -> dict:
    """PURE. Per-namespace storage arithmetic, recomputed from the grouped rows (the cluster-level pass in
    build_storage_inventory spans every namespace)."""
    c = {
        "caches": 0,
        "control": 0,
        "artifacts_pvcs": 0,
        "bytes": 0,
        "ready": 0,
        "warn": 0,
        "failed": 0,
        "total": 0,
    }
    for r in rows or []:
        c["total"] += 1
        c[r["state"]] += 1
        c["bytes"] += r.get("bytes", 0) or 0
        if r["kind"] == "cache":
            c["caches"] += 1
        elif r["kind"] == "control":
            c["control"] += 1
        else:
            c["artifacts_pvcs"] += r.get("n", 0) or 0
    return c


def _storage_summary(counts) -> str:
    """PURE. The STORAGE header: how many volumes, how much they cost, and WHAT they are. Every PVC in the
    namespace is in exactly one bucket, so `caches + control + artifacts` is the whole storage bill.
    """
    n = counts["caches"] + counts["control"] + counts["artifacts_pvcs"]
    parts = [f"{n} PVC{'' if n == 1 else 's'}"]
    if _fmt_bytes(counts["bytes"]):
        parts.append(_fmt_bytes(counts["bytes"]))
    parts.append(f"{counts['caches']} model cache{'' if counts['caches'] == 1 else 's'}")
    parts.append(f"{counts['control']} control")
    parts.append(f"{counts['artifacts_pvcs']} run-output")
    return " · ".join(parts)


def _idle_models_note(mdl_rows) -> str:
    """PURE. The model state of a COLLAPSED cluster, in a few words — '' when every catalog model is there.

    An idle cluster gets one line, and that line used to say nothing about models at all: `no weights here`
    and `everything downloaded` looked the same. Says the shortfall only (the happy case stays silent, so
    this never grows the line for a cluster that is genuinely ready)."""
    rows = [r for r in (mdl_rows or []) if r.get("catalog", True)]
    if not rows:
        return ""
    miss = sum(1 for r in rows if r["state"] == "missing")
    unk = sum(1 for r in rows if r["state"] == "unknown")
    if any(CI.is_config_blocker(r) for r in rows):  # never reached today (a blocker EXPANDS the cluster)
        return "no model cache configured"
    if unk:
        return f"{unk}/{len(rows)} models UNKNOWN"
    if miss:
        return f"{miss}/{len(rows)} models not installed"
    return ""


def _free_note(e) -> str:
    """A terse free-capacity note for a connected cluster's compact idle line ('' when capacity was skipped
    /unreadable). Headlines free WHOLE NODES (the schedulable unit — a tp=4/8 run takes a whole node), else
    the raw free-GPU count, else ''."""
    nc = e.get("nodes")
    if nc and nc.get("free_nodes") is not None and nc.get("total_nodes"):
        return (
            f"{nc['free_nodes']}/{nc['total_nodes']} GPU nodes free "  # GPU nodes — see _capacity_line
            f"(biggest {_gpus(nc['biggest_free'])})"
        )
    tot, occ = e.get("total"), e.get("occupied")
    if tot:  # only when there are GPUs to speak of (skip empty clusters)
        return f"{max(0, tot - (occ or 0))}/{tot} GPUs free"
    return ""


# ── MODEL-LOAD QUEUE ─────────────────────────────────────────────────────────────────────────────
# Runs sharing a model-cache PVC serialize checkpoint loading behind a per-cache Lease. Fleet reads the Lease
# and run-owner annotations to distinguish queued runs from active model loading.
L_MODEL_CACHE = "llmb.nvidia.com/model-cache"  # Lease → which PVC this lock serializes
ANN_ML_WAIT = "llmb.nvidia.com/model-load-wait"  # waiting run-owner Job → the PVC it is queued on
ANN_ML_SINCE = "llmb.nvidia.com/model-load-since"  # waiting run-owner Job → RFC3339 queue-entry time
ANN_ML_HOLDER = "llmb.nvidia.com/model-load-holder"  # holding run-owner Job → the PVC it holds


def _lease_live(lease: dict, now) -> bool:
    """PURE. Is this Lease still HELD? renewTime + leaseDurationSeconds must be in the future. A Lease whose
    holder died leaves the object behind, so an expired lease is a STALE LOCK — never a live holder, and never
    grounds for reporting waiters. Missing renewTime/duration → treat as NOT live (never invent a holder).
    """
    spec = lease.get("spec") or {}
    if not spec.get("holderIdentity"):
        return False
    t = _parse_ts(spec.get("renewTime") or spec.get("acquireTime") or "")
    try:
        dur = int(spec.get("leaseDurationSeconds") or 0)
    except (TypeError, ValueError):
        dur = 0
    if t is None or dur <= 0:
        return False
    return (now - t).total_seconds() <= dur


def discover_model_load(leases_j, all_jobs, now):
    """PURE. The model-load queue per (namespace, PVC), from the managed Leases + run-owner Job annotations.
    Returns [] when there is nothing to say — no lock, or the Lease read was unavailable (RBAC) — so the
    caller renders NOTHING rather than implying "no queue".

    Each entry: {ns, pvc, holder, since, waiters[], stale}. Discipline (mirrors the CAPACITY degradation
    rules): waiters are reported ONLY alongside a LIVE holder. Annotations with no live holder are phantom
    state left by a crashed run — we surface `stale` and drop the waiters rather than inventing a queue.
    """
    if leases_j is None or "items" not in (leases_j or {}):
        return []  # unreadable (no RBAC) → say nothing at all
    by = {}
    for l in leases_j["items"]:
        md = l.get("metadata") or {}
        pvc = (md.get("labels") or {}).get(L_MODEL_CACHE)
        if not pvc:
            continue  # a managed Lease that is not a model-load lock
        ns = md.get("namespace", "")
        live = _lease_live(l, now)
        spec = l.get("spec") or {}
        by[(ns, pvc)] = {
            "ns": ns,
            "pvc": pvc,
            "waiters": [],
            "holder": (spec.get("holderIdentity") or "") if live else "",
            "since": spec.get("acquireTime") or "",
            "stale": not live,
        }
    if not by:
        return []
    for j in all_jobs or []:
        md = j.get("metadata") or {}
        ann = md.get("annotations") or {}
        ns = md.get("namespace", "")
        pvc = ann.get(ANN_ML_WAIT)
        if not pvc:
            continue
        ent = by.get((ns, pvc))
        if not ent or not ent["holder"]:
            continue  # no live holder → phantom waiter, do NOT report a queue
        rid = _labels(j).get(L_RUN_ID) or _labels(j).get(L_CELL) or md.get("name", "")
        ent["waiters"].append({"run": rid, "since": ann.get(ANN_ML_SINCE, "")})
    for e in by.values():
        e["waiters"].sort(key=lambda w: (w["since"] or "~", w["run"]))
    # keep only rows worth showing: a live holder, or a stale lock worth flagging
    return [e for _, e in sorted(by.items()) if e["holder"] or e["stale"]]


def model_load_waiting_runs(queue):
    """PURE. The set of run identifiers currently QUEUED on a model-load slot → used to re-badge their
    RUN/SERVER row from the misleading `● LOADING server warming` to `◔ QUEUED`."""
    return {w["run"] for e in (queue or []) for w in e["waiters"] if w["run"]}


def _model_load_line(entry, now) -> str:
    """PURE. One namespace/PVC's model-load state as a single line body:
        `model-cache · ● r1c1 loading (12m) · ⏳ queued: r2c1, r3c1`
    A stale lock says so plainly instead of reporting a holder that is already gone. Shard progress is
    deliberately NOT shown: it is not in any object we already read, and inventing an extra per-cycle call
    for a cosmetic `17/113` is not worth it."""
    if entry["stale"] and not entry["holder"]:
        return f"{entry['pvc']} · (stale lock — no live holder)"
    held = human_age(entry.get("since", ""), now)
    line = f"{entry['pvc']} · ● {entry['holder']} loading" + (f" ({held})" if held else "")
    if entry["waiters"]:
        line += " · ⏳ queued: " + ", ".join(w["run"] for w in entry["waiters"])
    return line


def _capacity_line(e, *, fast: bool = False) -> str:
    """PURE. The CAPACITY sub-branch body for one cluster — the hardware answer to "what can I launch here",
    shown for ACTIVE clusters too (not just idle ones):

        GB300 · 18 nodes · 18 free · biggest free 4 GPUs

    The GPU product LEADS (it moved off the cluster bar). Then the truthful launchable unit: a whole FREE node
    (a tp=4/8 run takes an ENTIRE node, so fragmented free GPUs across busy nodes are unlaunchable), plus
    `biggest free` = the largest single run that can schedule RIGHT NOW. All from `node_capacity()` — data fleet
    already fetched, no new cluster call. Degrades honestly rather than printing a bogus 0:
      • --fast (capacity reads skipped)        → `GB300 · capacity skipped (--fast)`
      • no node list (no cluster-scope RBAC)   → `GB300 · node capacity unknown (no node read)`
      • nodes known, pods -A unreadable        → `GB300 · 18 nodes · capacity unknown (no pod read)`
    '' only when there is genuinely nothing to say (no GPU product and no node data)."""
    gtype = e.get("gpu_type", "")
    parts = [gtype] if gtype else []
    nc = e.get("nodes")
    if fast:
        parts.append("capacity skipped (--fast)")
    elif nc is None:
        parts.append("node capacity unknown (no node read)")
    else:
        tn, fn, bf = nc.get("total_nodes"), nc.get("free_nodes"), nc.get("biggest_free")
        if fn is None:  # nodes known, per-node occupancy unreadable → never invent a 0
            parts.append(f"{tn} GPU node{'' if tn == 1 else 's'}")
            parts.append("capacity unknown (no pod read)")
        else:
            # Report capacity in GPU-node units consistently across the view.
            parts.append(f"{fn}/{tn} GPU node{'' if tn == 1 else 's'} free")
            parts.append(f"biggest free {_gpus(bf)}")
    # UNATTRIBUTED: GPUs held in OUR namespaces by pods fleet cannot tie to a known run. Surfaced here rather
    # than folded into the foreign count, which would read as "someone else is using the cluster" when it is
    # our own workload. Shown only when non-zero — it is an anomaly, not part of the happy path.
    if not fast and (e.get("unattributed_gpu") or 0) > 0:
        parts.append(f"{_gpus(e['unattributed_gpu'])} unattributed in ours")
    return " · ".join(parts)


def _whole_node_note(e, *, fast: bool = False) -> str:
    """PURE. A dim hint for the ONE capacity state that silently blocks a launch: ZERO fully-free nodes. A cell
    with `requires.gpu.whole_node` requests ${GPU_PER_NODE} and preflight demands a fully-free node, so this
    turns a confusing downstream preflight failure into an up-front 'why'. '' on the happy path (never clutter)
    and whenever occupancy is unknown (we must not claim a blocker we cannot see)."""
    if fast:
        return ""
    nc = e.get("nodes")
    if not nc or nc.get("free_nodes") is None or not nc.get("total_nodes"):
        return ""
    if nc["free_nodes"] > 0:
        return ""
    return "⚠ no fully-free node — whole-node cells cannot schedule until one frees up"


def _stage_run_rows(e, now, *, gpu_only):
    """RUN-section rows for one CONNECTED cluster, in triage order: ACTIVE runs (each job + its nested └svc),
    then recently-FAILED runs (carrying the why+logs suffix), then ORPHAN servers holding GPU (carrying a
    reclaim hint). ✓-done runs are intentionally omitted — a settled success is not 'happening', it's noise
    here (the default `fleet` view still shows history). PURE."""
    active, failed, orphans = [], [], []
    for j in e["jobs"]:
        if j.get("infra") or j["status"] not in ACTIVE_JOB_STATES:
            continue
        total = j["gpus"] + sum(s["gpus"] for s in j["servers"])
        te, flag = _split_flag(_elapsed_expected(j, now))
        row = {
            "level": "job",
            "status": j["status"],
            "name": _run_name(j),
            "keep": "head",
            "gpus": total,
            "timing": te,
            "flag": flag,
            "sweep": _sweep_field(
                j.get("sweep") or "?",
                j.get("sweep_done"),
                j.get("sweep_total"),
                j["status"],
                live=j.get("sweep_live", False),
            ),
            "node": j.get("node", ""),
            "image": j.get("image", ""),
            "restarts": j.get("restarts", 0),
            "waited": age_seconds(j["start"], now) if j["status"] == UNSCHED else None,
        }
        if j.get("unsched_why"):
            # THE WHOLE DIAGNOSIS, INLINE. The operator's confusion was that the run row summarised a DEAD
            # bench and a HEALTHY server into one vague word; naming what the scheduler refused ends it.
            row["suffix"], row["suffix_code"] = "⚠ " + j["unsched_why"], "31"
        active.append(row)
        for s in j["servers"]:
            active.append(_svc_trow(s, now, nested=True))
    for s in e["standalone"]:
        if s["status"] in (
            LOADING,
            STUCK,
        ):  # a run coming up / a wedged run → a live run row
            active.append(_svc_trow(s, now, nested=False))
        elif s["status"] == PARKED and s["gpus"] > 0:
            # A server UP holding GPUs with NO active bench driving it — the "GPUs in the air" case the operator
            # was confused by. It IS a running-but-idle server, so it belongs in RUN/SERVER (its model-cache /
            # staged cell shows separately in INSTALLED). A scale-to-0 PARKED (0 GPUs held) stays collapsed —
            # it holds nothing and is already represented as its cell in INSTALLED. Inline `why` names the state.
            r = _svc_trow(s, now, nested=False)
            r["suffix"] = f"{_gpus(s['gpus'])} held · server up, no active bench"
            r["suffix_code"] = "36"
            active.append(r)
        elif s["status"] == ORPHAN:  # ready GPU-server, run gone → actionable (holds GPU)
            age = human_age(s["start"], now)
            orphans.append(
                {
                    "level": "orph",
                    "status": ORPHAN,
                    "name": s["name"],
                    "keep": "tail",
                    "gpus": s["gpus"],
                    "timing": f"{s['ready']}  {age}".strip(),
                    "flag": "",
                    "sweep": "",
                    "cluster": e["name"],
                    "node": s.get("node", ""),
                    "image": s.get("image", ""),
                    "restarts": s.get("restarts", 0),
                }
            )
    shown, _ = _recent_terminal(e, now, window_secs=FAILED_HISTORY_SECS, limit=None, failed_only=True)
    for j in shown:
        failed.append(_term_trow(j, now))
    # CROSS-NAMESPACE runs (LOADING/RUNNING/ORPHAN in our per-worktree namespaces) — the runs a single-ns view
    # is blind to. Split so RUNNING/LOADING/STUCK join the active block and held-ORPHANs join the orphan block.
    xns = e.get("xns_runs") or []
    xns_active = [r for r in xns if r["status"] in (RUNNING, LOADING, STUCK, STARTING, UNSCHED, QUEUED)]
    xns_held = [r for r in xns if r["status"] in (ORPHAN, PARKED)]  # held GPUs (parked / genuine orphan)
    # Every single-ns row belongs to the profile's configured namespace; cross-ns rows carry their own `ns`.
    # The `ns` tag drives the CLUSTER → NAMESPACE grouping in render_stages.
    cfg_ns = e.get("namespace", "") or ""
    for r in active + failed + orphans:
        r.setdefault("ns", cfg_ns)
    rows = active + xns_active + failed + orphans + xns_held
    if gpu_only:
        rows = [r for r in rows if r["gpus"] > 0 or r["level"] in ("term", "orph")]
    return rows


def _stage_classify(e, now, *, gpu_only, fast=False):
    """One entry → its stages-view classification (PURE). init/install come from LOCAL stamps (rendered even
    when unreachable); RUN rows only for a connected cluster. `full` = show the cluster expanded; otherwise it
    collapses to a compact idle line (connected) or the unreachable tail."""
    reachable = e["state"] == "connected"
    r_key = stage1_state(e.get("readiness"))[0]
    i_key = stage2_state(e.get("installs") or {})[0]
    run_rows = _stage_run_rows(e, now, gpu_only=gpu_only) if reachable else []
    has_run = any(r["level"] == "job" for r in run_rows)
    has_fail = any(r["level"] == "term" for r in run_rows)
    has_orph = any(r["level"] == "orph" for r in run_rows)
    init_block = r_key == "blocked"
    inst_block = i_key == "partial"
    # The unified INSTALLED inventory: LIVE-discovered resources (ground truth when connected) merged with the
    # local install stamps (offline fallback). ANY installed item → show this cluster in full so its inventory
    # ("what can be run here") is visible even when idle. A cluster with NOTHING installed + no runs + no block
    # stays idle (collapsed to one line).
    inv_rows, inv_counts, inv_source = build_install_inventory(
        e.get("installs"), e.get("discovered"), fallback_ns=e.get("namespace", "") or ""
    )
    # An UNREACHABLE cluster's model state is unknown for the same reason its run state is, and the cluster
    # tail already says so once, in words. Three `? UNKNOWN — PVC read FORBIDDEN` rows per namespace would
    # restate one fact per model and drown the one line that matters.
    mdl_rows, mdl_unknown = build_model_ledger(e, fast=fast) if reachable else ([], "")
    sto_rows = build_storage_inventory(  # the cluster-wide counts are re-derived per NAMESPACE below
        e.get("discovered"),
        fallback_ns=e.get("namespace", "") or "",
        ledger_claims={(r["ns"], r["claim"]) for r in mdl_rows if r.get("claim")},
        ledger_unknown=mdl_unknown,
    )[0]
    # WHAT COUNTS AS A REASON TO EXPAND A CLUSTER. "Not installed here yet" is the ordinary state of every
    # cluster you have not installed on; expanding all of them over it would make this pane unreadable, so a
    # ledger of plain `○ MISSING` stays collapsed. The rows still render once the cluster opens for any other
    # reason, and the idle line below names the gap so the two are never indistinguishable.
    # A CONFIGURATION BLOCKER IS NOT THAT. A profile with no MODEL_CACHE_PVC can never hold weights anywhere
    # and `install` fails closed on it, so the cluster cannot run a single cell — yet because that row is
    # ALSO `state="missing"` it inherited the collapse, and a freshly-onboarded cluster rendered in its
    # ENTIRETY as `· b200 [B200] · 2/2 nodes free`. Reads as fine; is broken. It expands now, which is what
    # makes the section's `↳ N models have nowhere to live` line — the one backfill_hint ranks above every
    # advisory as a HARD blocker — reachable at all.
    blocked = any(CI.is_config_blocker(r) for r in mdl_rows)
    has_inst = bool(inv_rows) or bool(sto_rows) or blocked or any(r["state"] != "missing" for r in mdl_rows)
    # Keep clusters expanded while tracked workloads hold GPUs, including workloads
    # discovered outside the profile namespace.
    held = (e.get("ours_gpu", 0) or 0) > 0
    full = has_run or has_fail or has_orph or init_block or has_inst or held
    return {
        "e": e,
        "reachable": reachable,
        "r_key": r_key,
        "i_key": i_key,
        "run_rows": run_rows,
        "init_block": init_block,
        "inst_block": inst_block,
        "inv_rows": inv_rows,
        "inv_counts": inv_counts,
        "inv_source": inv_source,
        "sto_rows": sto_rows,
        "mdl_rows": mdl_rows,
        "mdl_unknown": mdl_unknown,
        "has_inst": has_inst,
        "held": held,
        "full": full,
    }


def build_model_ledger(e, *, fast=False, catalog=None, pins=None):
    """The MODEL LEDGER for every namespace of one cluster — catalog-keyed rows, not PVC-keyed ones.

    THE STRUCTURAL FIX. One row per PVC could not represent (a) two models sharing a claim — `qwen3-0-6b`
    lives inside `glm5-fp8-model-cache` and was invisible — nor (b) a catalog model with no claim at all,
    which rendered as an OMISSION rather than a row. Keyed by catalog model, both are rows.

    THE EXPECTED SET IS SCOPED TO THE PROFILES' NAMESPACES. A cluster's other owned namespaces (discovered
    per-worktree ones) get evidence-derived rows only: asserting `○ MISSING` for all 3 catalog models in a
    colleague-adjacent worktree ns nobody configured would be noise, not signal.

    UNION OVER PROFILES: several profiles collapse onto one cluster row, so every profile's resolved claim is
    matched (see cache_inventory.expected_models) — otherwise a second profile's MODEL_CACHE_PVC is hidden.
    """
    profs = [p for p in (e.get("profile_envs") or []) if isinstance(p, dict)]
    cat = load_catalog() if catalog is None else catalog
    pin_map = load_model_pins() if pins is None else pins
    # THE EXPECTED SET CAN BE UNKNOWN, AND THAT IS NOT THE SAME AS EMPTY. With no readable profile the
    # ledger cannot say which claim any model belongs in; with no readable catalog it cannot say which
    # models exist. Both used to produce `expected = []`, which rendered as NO MODELS SECTION AT ALL and
    # left STORAGE asserting `no catalog model routes to this claim` about a volume three models route to —
    # a confident negative computed from a file we could not read.
    unknown = "cluster profile unreadable" if not profs else "catalog unreadable" if not cat else ""
    expected = CI.expected_models(cat, profs, pin_map) if profs else []
    prof_ns = {str(p.get("NAMESPACE") or "").strip() for p in profs}
    prof_ns.discard("")
    prof_ns.add(e.get("namespace", "") or "")
    discovered = e.get("discovered")
    unread = bool(e.get("inventory_unavailable"))
    cause = "fast" if fast else "unread"  # a read that did not land is not proof of a DENIAL
    jobs = download_job_facts(e.get("all_download_jobs") or [], our_namespaces=e.get("our_namespaces"))
    out = []
    namespaces = sorted(
        {d.get("ns", "") for d in (discovered or []) if d.get("ns")}
        | {n for n in prof_ns if n}
        | {j["ns"] for j in jobs}
    )
    for ns in namespaces:
        facts = None if unread else claim_facts(discovered, ns)
        rows = CI.reconcile(
            expected if ns in prof_ns else [],
            facts,
            jobs=[j for j in jobs if j["ns"] == ns],
            unreadable=cause,
            ns=ns,
        )
        out.extend(rows)
    return out, unknown


# CLUSTER → NAMESPACE grouping: each namespace under a cluster gets its own INSTALLED + RUN sub-block. A ns
# is ordered by how active it is (a live RUNNING run first, then LOADING/failed, then held, then installed-
# only); namespaces with neither an install nor a run are omitted (no empty blocks). All PURE + selftested.
_NS_STATUS_RANK = {
    RUNNING: 0,
    LOADING: 1,
    STUCK: 1,
    STARTING: 1,
    UNSCHED: 1,
    FAILED: 1,
    PARKED: 2,
    ORPHAN: 2,
}


def _ns_groups(inv_rows, run_rows, sto_rows=None, mdl_rows=None):
    """PURE. Group a cluster's per-namespace sections → an ORDERED list of
    {ns, inv, models, storage, runs, counts, mcounts, scounts, source}. Most-active namespace first
    (RUNNING > LOADING/FAILED > held > installed-only), then by ns name. A namespace with none of the four
    is omitted (no empty ns block).

    THE COUNTS ARE PER SECTION AND KIND-AWARE. `counts` used to be taken over EVERY row in the namespace,
    so the `llmb-control` PVC and the per-run artifacts rollup were counted as installed recipes — the
    reported header read `3 installed · 3 ready` over one model, one claim and one control volume. Each
    section now counts only its own kind."""

    def _slot(ns):
        return {"ns": ns, "inv": [], "runs": [], "storage": [], "models": []}

    by: dict = {}
    for key, rowset in (
        ("inv", inv_rows),
        ("runs", run_rows),
        ("storage", sto_rows),
        ("models", mdl_rows),
    ):
        for r in rowset or []:
            ns = r.get("ns", "")
            by.setdefault(ns, _slot(ns))[key].append(r)

    def rank(g):
        return min([_NS_STATUS_RANK.get(r["status"], 1) for r in g["runs"]], default=3)

    groups = sorted(by.values(), key=lambda g: (rank(g), g["ns"]))
    for g in groups:
        counts = {"ready": 0, "warn": 0, "failed": 0}
        for r in g["inv"]:
            counts[r["state"]] += 1
        counts["total"] = len(g["inv"])
        g["inv"].sort(key=lambda r: (_INSTALL_RANK[r["state"]], r["cell"]))
        g["storage"].sort(key=lambda r: (_INSTALL_RANK[r["state"]], r["cell"]))
        g["counts"] = counts
        g["mcounts"] = CI.ledger_counts(g["models"])
        g["scounts"] = _storage_counts(g["storage"])
        srcs = {r.get("source", "") for r in g["inv"]} or {"live"}
        g["source"] = (
            "live+stamp"
            if {"live", "stamp"} <= srcs
            else "live" if "live" in srcs else "stamp" if "stamp" in srcs else ""
        )
    return groups


def _ns_rule(ns, summary) -> str:
    """An INDENTED namespace rule — `━━ NAMESPACE <ns>  · <summary> ━━━` — nested ONE tier under the full-width
    cluster bar. The ALL-CAPS `NAMESPACE` label is uniform with the CLUSTER / INSTALLED / RUN·SERVER labels.
    Deliberately SHORTER than the cluster bar so the two nesting tiers (cluster → namespace) read unmistakably
    as `━━ … ━━` rule-lines. PURE."""
    head = f"━━ NAMESPACE {ns or '—'}" + (f"  · {summary}" if summary else "") + "  "
    return head + "━" * max(3, 58 - _vlen(head))


def _ns_summary(g) -> str:
    """The namespace's own footprint for its bar header: GPUs OUR runs hold here + a run count. '' when the
    namespace is installed-only (no runs) so the bar stays terse. PURE."""
    gpus = sum(r.get("gpus", 0) or 0 for r in g.get("runs", []) if r.get("level") in ("job", "orph"))
    nruns = sum(1 for r in g.get("runs", []) if r.get("level") == "job")
    parts = []
    if gpus:
        parts.append(f"using {_gpus(gpus)}")
    if nruns:
        parts.append(f"{nruns} run{'' if nruns == 1 else 's'}")
    return " · ".join(parts)


# ── box-drawing TREE guides (grey, receding) that thread the hierarchy cluster → namespace → section → row ──
_G_MID = "├─ "  # a non-last child branch
_G_END = "└─ "  # the LAST child branch
_G_RAIL = "│  "  # a vertical rail carried down past a non-last ancestor
_G_GAP = "   "  # blank continuation past a LAST ancestor (nothing more hangs below on this rail)


def backfill_hint(model_rows, ns="") -> str:
    """PURE. The ONE remediation line under a namespace's MODELS ledger — '' when nothing is unvouched.

    THIS FUNCTION USED TO PRINT:
        ↳ vouch for a cache you know is complete:  kubectl -n <ns> label pvc <PVC>
          llmb.nvidia.com/download-complete=true

    Follow that verbatim and you get a PVC stamped complete with NO `model-name` and no revision — which the
    panel then rendered as `✓ model: <the PVC's own name> @<40-char label> … verified by PVC stamp`. The
    remedy MANUFACTURED the defect it was offered to explain, and it is the most likely provenance of the
    reported nemotron row. Closing that loop is not a wording fix: there is no `kubectl` one-liner that can
    make a label into evidence about 700 GiB of weights, so none is offered. The line states what is and is
    not proven, and stops.

    (The read that WOULD settle it — a sentinel listing from inside the volume — is deliberately not built
    here: it needs a pod that can mount the claim, and some RWX claims mount on only a subset of
    11 nodes. Saying `~ unvouched` honestly is strictly better than a probe that fails on 9 nodes and gets
    read as an absence.)

    ONE line, most-actionable first: an unconfigured claim is a HARD blocker (install fails closed on it, and
    every model is affected), so it outranks the advisory note about unvouched weights.
    """
    rows = model_rows or []
    blocked = sum(1 for r in rows if r.get("evidence") == CI.NO_CLAIM_EVIDENCE)
    if blocked:
        return (
            f"↳ {blocked} model{'' if blocked == 1 else 's'} have nowhere to live — set "
            f'MODEL_CACHE_PVC="<pvc>" in cluster-profiles/<cluster>.env '
            f"(or the per-model MODEL_CACHE_PVC_<MODEL> key)"
        )
    n = sum(1 for r in rows if r.get("state") == "present")
    if not n:
        return ""
    return f"↳ {n} model{'' if n == 1 else 's'} present but unvouched — " f"weights may be complete; nothing proves it."


# A healthy inventory list is folded into ONE line once it passes this length. WHY A THRESHOLD AT ALL: the
# fold line costs 1 row and the section header already states the count and the verdict, so the ONLY thing
# the rows add is WHICH cells — worth printing while you can take them in at a glance, worthless as a wall.
# Below the threshold the names ARE the summary (a 1-cell namespace reading `✓ all 1 ready` instead of the
# cell's name is strictly worse); at 30 they are 30 near-identical 110-char paths nobody reads. `--detail`
# lists them at any size, so nothing is unreachable.
_INSTALL_FOLD_MIN = 5


def _tree_installed(out, g, paint, gr, base, sec_branch, sec_cont, *, unavail="", detail=False):
    """One namespace's INSTALLED section as a tree branch (`base+sec_branch` → the INSTALLED header; its rows
    hang off `base+sec_cont` with ├─/└─). ALWAYS emitted (a managed ns always has at least a model-cache /
    staged cell; '— nothing installed —' when truly empty). PURE (bar paint; `gr` colors guides grey).

    `unavail` (from inventory_note) = the model-cache read did NOT land. Then an empty list is UNKNOWN, not
    empty, and a non-empty list is PARTIAL — both say so rather than passing for a complete inventory.

    ROWS NEEDING ACTION ARE ALWAYS LISTED; a long tail of HEALTHY ones folds into one line (see
    _INSTALL_FOLD_MIN). With a 31-cell catalog the ✓ list is inherently long and almost always uniform: on
    the reported cluster it was 30 identical `✓ <110-char path>  ready` rows — 30 of the namespace's 45
    lines, carrying one bit of information between them, and the reason the viewport had to collapse
    anything at all. The rows are SORTED attention-first, so the fold always replaces a trailing run.
    """
    _setp(out, P_STRUCT)  # the HEADER carries the verdict — it is never collapsed away
    if not g.get("inv"):
        body = paint.yellow(f"— inventory UNKNOWN — {unavail}") if unavail else paint.dim("— nothing installed —")
        out.append(gr(base + sec_branch) + paint.bold("INSTALLED") + "  " + body)
        return
    counts = g["counts"]
    summ = _install_summary(counts, g["source"])
    summ_c = paint.red(summ) if counts["failed"] else paint.yellow(summ) if counts["warn"] else paint.green(summ)
    if unavail:  # a PARTIAL list must not read as the whole inventory
        summ_c += paint.yellow(f"  ⚠ PARTIAL — {unavail}")
    out.append(gr(base + sec_branch) + paint.bold("INSTALLED") + "  " + summ_c)
    rb = base + sec_cont
    rows = g["inv"]
    ready = [r for r in rows if r["state"] == "ready"]
    fold = (not detail) and len(ready) >= _INSTALL_FOLD_MIN
    shown = [r for r in rows if not (fold and r["state"] == "ready")]
    cw = max((len(r["disp"]) for r in shown), default=0)
    last = len(shown) - 1 + (1 if fold else 0)  # the fold line, when present, is the last child
    for idx, r in enumerate(shown):
        _setp(out, P_INSTALLED if r["state"] == "ready" else P_ATTENTION, _VP_FOLDED_LABEL)
        pre = gr(rb + (_G_END if idx == last else _G_MID))
        disp = r["disp"].ljust(cw)
        if r["state"] == "ready":
            out.append(pre + paint.green("✓ ") + disp + paint.dim("   " + (r["why"] or "ready")))
        elif r["state"] == "warn":
            out.append(pre + paint.yellow("⚠ ") + disp + paint.dim("   " + (r["why"] or "warn")))
        elif r["fix_kind"]:
            fix = _install_fix(r["fix_kind"], g["cluster"], r["cell"])
            out.append(pre + paint.red("✗ ") + disp + paint.red("   FAILED") + paint.dim(f" — {r['why']} · fix: {fix}"))
        else:
            out.append(pre + paint.red("✗ ") + disp + paint.red("   FAILED") + paint.dim(f" — {r['why']}"))
    if fold:
        _setp(out, P_INSTALLED, _VP_FOLDED_LABEL)
        n = len(ready)
        body = (
            f"✓ all {n} ready — nothing needs attention"
            if n == len(rows)
            else f"{n} more ✓ ready — nothing needs attention"
        )
        out.append(gr(rb + _G_END) + paint.green(body) + paint.dim("   (--detail to list them)"))


def _models_summary(c, source="") -> str:
    """PURE. The MODELS header. `missing` is printed even at ZERO — "0 missing" is the answer to the question
    an operator actually has, and a silent zero is indistinguishable from a section that forgot to look.
    UNKNOWN is its OWN term and is never folded into a positive count."""
    # ATTESTED, not VERIFIED — see cache_inventory.ledger_counts. A ✓ here rests on a download Job's LABEL;
    # the only grade that has seen a byte is the in-volume sentinel, and nothing produces one yet. Saying
    # `0 verified (no in-volume check)` beside the attested count keeps the header from claiming the
    # stronger of the two.
    parts = [
        f"{c['catalog']} catalog model{'' if c['catalog'] == 1 else 's'}",
        f"{c.get('attested', c['verified'])} attested",
    ]
    parts.append(f"{c.get('sentinel', 0)} verified" + ("" if c.get("sentinel") else " (no in-volume check)"))
    if c["present"]:
        parts.append(f"{c['present']} UNVERIFIED")
    if c["downloading"]:
        parts.append(f"{c['downloading']} downloading")
    parts.append(f"{c['missing']} missing")
    if c["unknown"]:
        parts.append(f"{c['unknown']} UNKNOWN")
    if c["failed"]:
        parts.append(f"{c['failed']} FAILED")
    if c["extra"]:
        parts.append(f"{c['extra']} off-catalog")
    if source:
        parts.append(source)
    return " · ".join(parts)


_MODEL_PAINT = {
    "verified": "green",
    "present": "yellow",
    "downloading": "yellow",
    "missing": "yellow",
    "failed": "red",
    "unknown": "yellow",
}


def _tree_models(out, g, paint, gr, base, sec_branch, sec_cont, *, unknown=""):
    """One namespace's MODEL LEDGER as a tree branch. One row per CATALOG MODEL (plus any model the cluster
    holds that the catalog does not ask for) — never one row per PVC, which could represent neither two
    models in one claim nor a model in no claim.

    COLUMN DISCIPLINE. MODEL and REVISION are fixed-width by construction: `_short_rev` guarantees 12 chars
    so no label length can move the EVIDENCE column (a 40-char `model-revision` used to shove it 28 columns
    right). The size prints on the FIRST row of a claim only — three rows quoting `1200Gi` would read as
    3.6Ti of storage that does not exist — the rest say `shared`."""
    rows = g.get("models") or []
    if not rows and not unknown:
        return
    _setp(out, P_STRUCT)  # the HEADER carries the verdict — it is never collapsed away
    # A SECTION THAT VANISHES IS A FACT NOBODY CAN SEE. With an unreadable profile or catalog the expected
    # set is UNKNOWN, not empty — and rendering nothing here read as "this namespace has no models to speak
    # of", which is a confident negative computed from a file we could not open. ONE header either way: the
    # UNKNOWN caveat leads, and any evidence-derived rows still follow it.
    c = g["mcounts"]
    if unknown:
        summ_c = paint.yellow(
            f"— ledger UNKNOWN — {unknown}: cannot tell which models "
            f"this collection needs, nor which claim holds them"
        )
        if rows:
            summ_c += paint.dim(
                f"  ({len(rows)} model{'' if len(rows) == 1 else 's'} seen on the cluster,"
                f" none of them expected-set)"
            )
    else:
        summ = _models_summary(c, g.get("source", ""))
        summ_c = (
            paint.red(summ)
            if c["failed"]
            else (paint.yellow(summ) if (c["present"] or c["missing"] or c["unknown"]) else paint.green(summ))
        )
    out.append(gr(base + sec_branch) + paint.bold("MODELS") + "    " + summ_c)
    if not rows:
        return
    rb = base + sec_cont
    # Column widths include the HEADER text, or a table of short values pushes its own header out of line.
    mw = max([len("MODEL")] + [len(r["model"]) for r in rows])
    cl = max([len("CLAIM")] + [len(r["claim"] or "—") for r in rows])
    sz = max([0] + [len(r["size"] or "") for r in rows])
    hint = backfill_hint(rows, g.get("ns", ""))
    # `_G_GAP` matches the ├─/└─ branch width; the extra 2 spaces match the row GLYPH ("✓ "), so the header
    # sits exactly over its column instead of two characters to the left of every value beneath it.
    out.append(
        gr(rb + _G_GAP)
        + "  "
        + paint.dim(
            f"{'MODEL'.ljust(mw)}  {'REVISION'.ljust(REV_DISPLAY + 1)}  CELLS  "
            f"{'CLAIM'.ljust(cl)} {''.ljust(sz)}  EVIDENCE"
        )
    )
    last = len(rows) - 1 + (1 if hint else 0)
    for idx, r in enumerate(rows):
        # A model that is anything but ✓ VERIFIED is the reason this section exists — it can never be
        # laddered away, so an inventory marker's `all ✓` stays true by construction.
        _setp(out, P_INSTALLED if r["state"] == "verified" else P_ATTENTION, "model rows")
        pre = gr(rb + (_G_END if idx == last else _G_MID))
        col = getattr(paint, _MODEL_PAINT[r["state"]])
        cells = str(r["cells"]) if r["cells"] else "-"
        # `@?`, never a bare `@`: with an unreadable recipe there is no pin AND no label, and a lone `@`
        # reads as a rendering fault rather than as "the revision is unknown".
        body = (
            f"{r['model'].ljust(mw)}  {('@' + (r['rev'] or '?')).ljust(REV_DISPLAY + 1)}  {cells:>5}  "
            f"{(r['claim'] or '—').ljust(cl)} {(r['size'] or '').ljust(sz)}  "
        )
        out.append(pre + col(r["glyph"] + " ") + body + paint.dim(r["evidence"]))
    if hint:
        _setp(out, P_ATTENTION)  # a remediation line is by definition the actionable one
        out.append(gr(rb + _G_END) + paint.dim(hint))


def _tree_storage(out, g, paint, gr, base, sec_branch, sec_cont):
    """One namespace's STORAGE section: the VOLUMES and whether each is usable. Model-cache claims are
    COUNTED in the header and given a row only when they need one (not Bound, or holding something nothing
    can attribute) — a claim that is quietly doing its job is already named in the MODELS ledger's CLAIM
    column, and repeating it is how one namespace grew 82 look-alike rows."""
    rows = g.get("storage") or []
    if not rows:
        return
    _setp(out, P_STRUCT)  # the HEADER carries the verdict — it is never collapsed away
    c = g["scounts"]
    summ = _storage_summary(c)
    summ_c = paint.red(summ) if c["failed"] else paint.yellow(summ) if c["warn"] else paint.green(summ)
    out.append(gr(base + sec_branch) + paint.bold("STORAGE") + "   " + summ_c)
    shown = [r for r in rows if not (r["kind"] == "cache" and r["state"] == "ready")]
    if not shown:
        return
    rb = base + sec_cont
    cw = max(len(r["disp"]) for r in shown)
    last = len(shown) - 1
    for idx, r in enumerate(shown):
        _setp(out, P_INSTALLED if r["state"] == "ready" else P_ATTENTION, "storage rows")
        pre = gr(rb + (_G_END if idx == last else _G_MID))
        disp = r["disp"].ljust(cw)
        if r["state"] == "ready":
            out.append(pre + paint.green("✓ ") + disp + paint.dim("   " + (r["why"] or "ready")))
        elif r["state"] == "warn":
            out.append(pre + paint.yellow("⚠ ") + disp + paint.dim("   " + (r["why"] or "warn")))
        else:
            out.append(pre + paint.red("✗ ") + disp + paint.red("   FAILED") + paint.dim(f" — {r['why']}"))


def _tree_run(out, g, W, paint, gr, base, sec_branch, sec_cont, *, detail):
    """One namespace's RUN / SERVER section as a tree branch (`base+sec_branch` → the header; its rows hang off
    `base+sec_cont` with ├─/└─, 'no active runs' when idle). The column header gets a 3-space placeholder in
    place of a branch so it stays the SAME width as ├─/└─ and the table columns line up. PURE (bar paint).
    """
    _setp(out, P_RUN)  # NEVER collapsed by the viewport ladder — this is what --watch is for
    out.append(gr(base + sec_branch) + paint.bold("RUN / SERVER"))
    rb = base + sec_cont
    if g.get("runs") and W:
        out.append(gr(rb + _G_GAP) + paint.dim(_col_header(W)))  # 3-space placeholder = ├─/└─ width → aligned
        last = len(g["runs"]) - 1
        for idx, r in enumerate(g["runs"]):
            out.append(gr(rb + (_G_END if idx == last else _G_MID)) + _fmt_trow(r, W, paint, wide=False, detail=detail))
    else:
        out.append(gr(rb + _G_END) + paint.dim("no active runs"))


def _render_ns_tree(out, e, groups, W, paint, *, reachable, detail, unavail="", ledger_unknown=""):
    """Render one cluster's namespaces as a box-drawing TREE hanging off the cluster bar: each namespace is a
    `├─/└─` branch (└─ for the last), and beneath it INSTALLED + RUN / SERVER are `├─/└─` branches whose rows
    hang off with continuation │ and ├─/└─. Grey guides recede; the magenta namespace bars + white section
    headers lead. The vertical │ rails DOUBLE as the breathing-room separators between blocks (one guide line,
    not a bare blank) so the nesting reads at a glance. PURE (bar paint)."""
    gr = paint.grey
    n = len(groups)
    _setp(out, P_STRUCT)
    out.append(gr("│"))  # the cluster rail dropping to its first namespace
    for gi, g in enumerate(groups):
        g["cluster"] = e["name"]
        _setp(out, P_STRUCT)
        last_ns = gi == n - 1
        ns_cont = _G_GAP if last_ns else _G_RAIL  # the cluster rail carried DOWN through this ns's subtree
        # MAGENTA on ONLY the `━━` rule segments (label + name text stays default-foreground = readable).
        out.append(gr(_G_END if last_ns else _G_MID) + _color_bars(_ns_rule(g["ns"], _ns_summary(g)), paint.magenta))
        # FOUR QUESTIONS, FOUR SECTIONS. INSTALLED = which CELLS can run here · MODELS = which MODELS are
        # here and what proves it · STORAGE = which VOLUMES exist and are they usable · RUN = what is
        # happening. They used to be one list with one set of counts, which is how a control-plane PVC came
        # to be one of "3 installed · 3 ready". MODELS/STORAGE are omitted when they have nothing to say;
        # INSTALLED is always present (its empty/UNKNOWN distinction is itself the signal).
        secs = ["inst"]
        if g.get("models") or ledger_unknown:  # an UNKNOWN ledger is a section, not a missing one
            secs.append("models")
        if g.get("storage"):
            secs.append("storage")
        if reachable:  # unreachable ns → run state UNKNOWN, so no RUN branch
            secs.append("run")
        out.append(gr(ns_cont + "│"))  # rail below the ns bar (space above INSTALLED)
        for si, sec in enumerate(secs):
            last_sec = si == len(secs) - 1
            sb = _G_END if last_sec else _G_MID
            scont = _G_GAP if last_sec else _G_RAIL
            if sec == "inst":
                _tree_installed(
                    out,
                    g,
                    paint,
                    gr,
                    ns_cont,
                    sb,
                    scont,
                    unavail=unavail,
                    detail=detail,
                )
            elif sec == "models":
                _tree_models(out, g, paint, gr, ns_cont, sb, scont, unknown=ledger_unknown)
            elif sec == "storage":
                _tree_storage(out, g, paint, gr, ns_cont, sb, scont)
            else:
                _tree_run(out, g, W, paint, gr, ns_cont, sb, scont, detail=detail)
            _setp(out, P_STRUCT)
            if not last_sec:
                out.append(gr(ns_cont + "│"))  # rail between the two sections (INSTALLED · RUN/SERVER)
        if not last_ns:
            out.append(gr("│"))  # rail between namespaces (back at the cluster level)


# ── VIEWPORT FIT ─────────────────────────────────────────────────────────────────────────────────
# Watch mode must fit the terminal viewport. Lower-priority sections collapse first and the footer reports
# omitted lines. Run rows and their cluster/namespace structure remain highest priority.
P_RUN = 0  # headline, live run/server rows, failures, actionable hints  — never laddered away
P_STRUCT = 1  # cluster + namespace bars, SECTION HEADERS                   — never laddered away
P_ATTENTION = 2  # inventory rows that need action (✗ ○ ? ~ ⚠)                 — never laddered away
P_CAPACITY = 3  # CAPACITY / MODEL LOAD / INIT sub-branches
P_INSTALLED = 4  # HEALTHY inventory rows only (✓) — anything else is P_ATTENTION
P_IDLE = 5  # idle-connected one-liners + the unreachable/refreshing tail
P_CHROME = 6  # build stamp
P_BLANK = 7  # pure spacer lines (no information — collapsed with no marker, still counted)

# WHY P_ATTENTION EXISTS, AND WHY SECTION HEADERS ARE P_STRUCT.
# Priority used to be assigned per SECTION: every line of INSTALLED/MODELS/STORAGE — header included — was
# P_INSTALLED. On a 24-row terminal that produced, for a healthy namespace:
#       … 31 lines hidden (installed inventory)
#       … 6 lines hidden (installed inventory)
#       … 2 lines hidden (installed inventory)
# Three defects at once. (a) The HEADERS were collapsed with their rows, so the one line carrying the
# verdict — `30 cells · 30 ready`, `1 verified · 2 UNVERIFIED` — was the first thing thrown away. (b) The
# marker could equally have been concealing 30 ✓ or 30 ✗ and the reader could not tell, so a hidden row was
# an UNKNOWN; unknowns are precisely what this pane exists to eliminate. (c) Three markers with identical
# labels read as a rendering fault rather than as three sections.
# Now: headers are P_STRUCT (never hidden, so the verdict always survives and each marker sits under a
# header that names its section), and rows take their priority from their own STATE. Because the ladder
# never descends past P_CAPACITY, a row that needs action is structurally uncollapsible — which is what
# lets an inventory marker assert `all ✓` as a FACT rather than a hope. Pinned by `_unit_collapse_safety`.
_VP_LADDER = (P_BLANK, P_CHROME, P_IDLE, P_INSTALLED, P_CAPACITY)
_VP_LABEL = {
    P_RUN: "runs",
    P_STRUCT: "structure",
    P_ATTENTION: "needs attention",
    P_CAPACITY: "capacity",
    P_INSTALLED: "installed inventory",
    P_IDLE: "idle clusters",
    P_CHROME: "build stamp",
    P_BLANK: "spacing",
}
# The VERDICT a collapsed region of this priority is entitled to assert. Only P_INSTALLED has one, and only
# because P_ATTENTION guarantees nothing needing action can be inside it.
_VP_VERDICT = {P_INSTALLED: "all ✓, nothing needs attention"}
_VP_ANSI = re.compile(r"\033\[[0-9;?]*[a-zA-Z]")
VP_MARK = "\u2026 "  # per-region marker prefix  ("… N lines hidden (…)")
VP_FOOTER_MARK = "viewport:"  # the one global accounting line
_VP_FOLDED_LABEL = "installed cells"  # the ONE section `--detail` expands — see _tree_installed's fold
# The smallest region worth replacing with a marker. A marker costs one line, so collapsing a single line
# frees nothing and trades content for an apology — see _vp_collapse.
_VP_MIN_REGION = 2


class LineBuf(list):
    """A line buffer that records a DROP PRIORITY per appended line, so `fit_viewport` can collapse the
    least-actionable regions of a frame first. Behaves exactly like `list` for every existing caller;
    `prio` is a parallel list of the same length. Set `.cur` to change the priority of what follows.
    """

    def __init__(self, cur=P_RUN):
        super().__init__()
        self.prio: list = []
        self.lbl: list = []  # per-line SECTION name, so a marker can say WHICH list it replaced
        self.cur = cur
        self.curlbl = ""

    def append(self, line):
        super().append(line)
        blank = line is not None and not str(line).strip()
        self.prio.append(P_BLANK if blank else self.cur)
        self.lbl.append("" if blank else self.curlbl)


def _setp(out, prio, lbl=""):
    """Set the current drop-priority (and optional SECTION label) on a LineBuf. No-op for a plain list
    (unit tests pass lists). The label is what stops three different sections' markers from all calling
    themselves `installed inventory`, which is what made one namespace read as a rendering fault.
    """
    if isinstance(out, LineBuf):
        out.cur = prio
        out.curlbl = lbl


def _vp_cost(line: str, cols: int) -> int:
    """How many terminal ROWS this line actually occupies — a long line WRAPS, so a frame with 40 lines can
    need 55 rows. Ignoring wrap is how a 'it fits' calculation still scrolls the top away.
    """
    if cols and cols > 0:
        w = _vlen(_VP_ANSI.sub("", line or ""))
        return max(1, -(-w // cols))
    return 1


def _vp_labels(prios) -> str:
    seen = []
    for p in sorted(set(prios), reverse=True):
        lbl = _VP_LABEL.get(p, "")
        if lbl and lbl not in seen and p != P_BLANK:
            seen.append(lbl)
    return " · ".join(seen)


def _vp_collapse(lines, prios, level, lbls=None):
    """Drop every line whose priority is >= `level`, replacing each CONTIGUOUS dropped region with ONE
    explicit marker line. Returns (out_lines, out_prios, out_is_marker, labels): `out_is_marker[i]` is True
    for a line WE synthesised, so the caller can count originals without double-counting its own markers.
    A region that was pure spacing gets no marker (a blank line carries nothing) but is still counted by
    the caller's `total - originals_kept` arithmetic. PURE."""
    out, oprio, omark, dropped_p = [], [], [], []
    lbls = list(lbls or [""] * len(lines))
    i, n = 0, len(lines)
    while i < n:
        if prios[i] < level:
            out.append(lines[i])
            oprio.append(prios[i])
            omark.append(False)
            i += 1
            continue
        j = i
        region, first = [], lines[i]
        while j < n and prios[j] >= level:
            region.append(prios[j])
            j += 1
        own = next((lbls[k] for k in range(i, j) if lbls[k]), "")  # this region's OWN section, if it has one
        if any(p != P_BLANK for p in region) and len(region) < _VP_MIN_REGION:
            # A ONE-LINE REGION IS NOT WORTH A MARKER. Replacing one line with one `… 1 line hidden (…)`
            # line frees no rows and trades content for an apology — the pane rendered three of these in a
            # row on the reported cluster. If the marker cannot be cheaper than what it replaces, keep the
            # real line.
            out.extend(lines[i:j])
            oprio.extend(prios[i:j])
            omark.extend([False] * len(region))
            i = j
            continue
        dropped_p.extend(region)
        if any(p != P_BLANK for p in region):
            lbl = own or _vp_labels(region)
            # THE MARKER ANSWERS "DOES THIS MATTER?", not just "how much is gone". `… 32 lines hidden` is a
            # number an operator can do nothing with: it neither says what was hidden nor whether any of it
            # needed them. A region of uniform priority that has a VERDICT (only P_INSTALLED does, and only
            # because P_ATTENTION cannot be inside it) states that verdict and names the flag that expands
            # it. `--detail`, NOT `--all`: `--all` sets FLAT_ONLY_FLAG in fleet.sh and so switches --watch
            # to the flat pane — pointing a --watch user at it would answer their question by removing the
            # pane they asked for.
            verdict = _VP_VERDICT.get(region[0]) if len(set(region)) == 1 else None
            ind = re.match(r"[ ]*", _VP_ANSI.sub("", first or "")).group(0)  # keep the tree depth
            note = f"{lbl} — {verdict}" if (lbl and verdict) else lbl
            # …and the flag is offered ONLY where it does something. `--detail` expands the INSTALLED cell
            # fold; on a MODELS or STORAGE marker it is advice that changes nothing (`_tree_models` does not
            # even take the flag), and a remedy that does not work is worse than none.
            expandable = verdict and own == _VP_FOLDED_LABEL
            out.append(
                f"{ind}{VP_MARK}{len(region)} line{'' if len(region) == 1 else 's'} hidden"
                + (f" ({note})" if note else "")
                + ("   (--detail to show)" if expandable else "")
            )
            oprio.append(P_STRUCT)
            omark.append(True)
        i = j
    return out, oprio, omark, _vp_labels(dropped_p)


_VP_CLUSTER_RE = re.compile(r"━━ CLUSTER (\S+)")


def _vp_clusters(lines) -> list:
    """PURE. The cluster names whose `━━ CLUSTER <name>` bar appears in these lines, in order.
    Used to name the clusters a tail-cut removed ENTIRELY — the one loss a line count cannot convey.
    """
    out = []
    for ln in lines or []:
        m = _VP_CLUSTER_RE.search(_VP_ANSI.sub("", ln or ""))
        if m and m.group(1) not in out:
            out.append(m.group(1))
    return out


def _vp_footer(hidden: int, total: int, labels: str) -> str:
    return (
        f"  \u26a0 {VP_FOOTER_MARK} {hidden} of {total} lines hidden"
        + (f" ({labels})" if labels else "")
        + " \u2014 lengthen/widen the terminal, or run `fleet.sh` (no --watch) for the full render"
    )


def fit_viewport(lines, prios, rows: int, cols: int = 0, lbls=None) -> list:
    """Fit `lines` into a `rows`-tall (`cols`-wide, 0 = don't model wrap) viewport.

    INVARIANT: nothing is ever dropped SILENTLY. Every collapsed region leaves an explicit
    `… N lines hidden (…)` marker in place, and one `⚠ viewport: N of M lines hidden …` footer restates
    the total plus how to see the rest — where N is EXACTLY (original lines in) - (original lines kept),
    counting neither the markers nor the footer as content. `rows <= 0` means 'unbounded' → returned
    unchanged, so the one-shot pane and every scripted caller are byte-for-byte unaffected. PURE.
    """
    lines = list(lines)
    prios = list(prios)
    lbls = list(lbls) if lbls else None
    rows = int(rows or 0)
    if rows <= 0 or not lines:
        return lines
    total = len(lines)
    cost = lambda ls: sum(_vp_cost(l, cols) for l in ls)  # noqa: E731 — local, one expression
    if cost(lines) <= rows:
        return lines

    best = (lines, prios, [False] * total, "")
    for level in _VP_LADDER:
        out, oprio, omark, labels = _vp_collapse(lines, prios, level, lbls)
        kept = sum(1 for m in omark if not m)
        if kept == total:  # this level dropped nothing — try the next
            continue
        best = (out, oprio, omark, labels)
        cand = out + [_vp_footer(total - kept, total, labels)]
        if cost(cand) <= rows:
            return cand

    # If the high-priority content still overflows, retain the head and report the omitted tail.
    out, oprio, omark, labels = best
    budget = max(1, rows - _vp_cost(_vp_footer(total, total, labels + " · overflow"), cols))
    keep, keep_mark, kept_i, used = [], [], [], 0
    for idx, (ln, mk) in enumerate(zip(out, omark)):
        c = _vp_cost(ln, cols)
        if used + c > budget:
            break
        keep.append(ln)
        keep_mark.append(mk)
        kept_i.append(idx)
        used += c
    while True:
        # THE FOOTER MUST DESCRIBE WHAT WAS ACTUALLY CUT. `labels` came from the LADDER's dropped
        # priorities alone, so a cluster removed by this position-cut contributed only the word `overflow`
        # — a three-cluster frame lost every line of a BROKEN cluster while the footer read
        # `(installed inventory · capacity · overflow)`, naming two healthy categories and not the runs,
        # and the two survivors both read `all ✓`. This section's own comment says it exists to stop
        # "whole clusters vanished with no trace"; a label list that omits them is that trace missing.
        cut = set(range(len(out))) - set(kept_i)
        parts = [p for p in labels.split(" · ") if p]  # what the LADDER collapsed
        for p in _vp_labels([oprio[i] for i in cut if not omark[i]]).split(" · "):
            if p and p not in parts:  # …plus what the CUT removed
                parts.append(p)
        lbl = " · ".join(parts + ["overflow"])
        lost = [c for c in _vp_clusters(out) if c not in _vp_clusters(keep)]
        if lost:  # name them: a missing cluster is the one thing a count cannot convey
            lbl += " · CLUSTERS CUT: " + ", ".join(lost)
        foot = _vp_footer(total - sum(1 for m in keep_mark if not m), total, lbl)
        if used + _vp_cost(foot, cols) <= rows or not keep:
            return keep + [foot]
        used -= _vp_cost(keep.pop(), cols)
        keep_mark.pop()
        kept_i.pop()


def build_line(build: str) -> str:
    """PURE. The one-line BUILD STAMP: which fleet is talking. '' when unknown (not a git checkout).

    The build identifier makes output attributable when multiple checkouts are in use.
    """
    return f"fleet build {build}" if build.strip() else ""


def render_stages(
    entries,
    now,
    *,
    color,
    fast,
    gpu_only,
    detail,
    history_secs=DEFAULT_HISTORY_SECS,
    history_n=None,
    build="",
    viewport_rows: int = 0,
    viewport_cols: int = 0,
) -> str:
    """The redesigned `--stages` pane. Same answer-at-a-glance headline as the default view, then per-CLUSTER
    (only the active ones in full) → NAMESPACE → the INIT/INSTALL/RUN stage sub-sections, with idle clusters
    collapsed. See the module comment above for the full contract. PURE (bar `paint`).
    """
    paint = Paint(color)
    out = LineBuf(P_RUN)  # priority-tagged so a short --watch viewport collapses the RIGHT things
    connected = [e for e in entries if e["state"] == "connected"]
    disconnected = [e for e in entries if e["state"] in ("auth", "unreach")]
    refreshing = [e for e in entries if e["state"] == "refreshing"]
    n_conn, n_disc = len(connected), len(disconnected)
    ts = now.strftime("%Y-%m-%d %H:%M:%SZ")
    actives = _collect_active(connected, now)

    # ── LINE 1 — the SAME answer-at-a-glance headline as the default pane (reused signal, not new noise):
    #    active runs · my GPUs · soonest ETA · recent failures | fleet up/auth · GPU used · timestamp ──
    g_ours = sum(c["ours_gpu"] for c in connected)
    occ_vals = [c["occupied"] for c in connected if c["occupied"] is not None]
    tot_vals = [c["total"] for c in connected if c["total"] is not None]
    partial = (len(occ_vals) < n_conn) or (len(tot_vals) < n_conn)
    recent_failed = sum(
        len(_recent_terminal(c, now, window_secs=FAILED_HISTORY_SECS, limit=None, failed_only=True)[0])
        for c in connected
    )
    eta = _soonest_eta(actives)
    fleet = f"fleet {n_conn} up"
    if n_disc:
        fleet += f" · {n_disc} auth✗"
    if refreshing:
        fleet += f" · {len(refreshing)} refreshing"
    if not fast and tot_vals and occ_vals:
        fleet += f" · {sum(occ_vals)}/{sum(tot_vals)} GPUs used" + ("*" if partial else "")
    line1 = f"ACTIVE  {len(actives)} run{'' if len(actives) == 1 else 's'} · {_gpus(g_ours)} (ours)"
    if eta:
        line1 += f" · {eta}"
    if recent_failed:
        line1 += paint.red(f" · ⚠ {recent_failed} recently FAILED")
    line1 += f"   |   {fleet} · {ts}"
    out.append(paint.bold(line1))
    breakdown = _type_breakdown(connected)
    if breakdown:
        out.append(paint.dim("        " + breakdown + (" · fast (capacity skipped)" if fast else "")))

    # ── classify every configured cluster → full / idle-connected / idle-unreachable ──
    infos = [_stage_classify(e, now, gpu_only=gpu_only, fast=fast) for e in entries]
    full = [i for i in infos if i["full"]]
    idle_conn = [i for i in infos if not i["full"] and i["reachable"]]
    idle_unreach = [i for i in infos if not i["full"] and i["e"]["state"] in ("auth", "unreach")]

    # shared RUN column widths across every full cluster → straight columns cluster-to-cluster
    all_run = [r for i in full for r in i["run_rows"]]
    W = _col_widths(all_run) if all_run else None

    def _gtype(e):
        return f" [{e['gpu_type']}]" if e.get("gpu_type") else ""

    # ── FULL clusters: header → INIT → per-NAMESPACE (INSTALLED + RUN) sub-blocks ──
    # HIERARCHY: Cluster (deduped by context) → Namespace (grouped, empty ones collapsed) → INSTALLED/RUN.
    # Each namespace's own runs live under ITS header (a cross-ns run is placed under its real namespace, not
    # the configured ns with an inline tag), so the operator reads one section per physical cluster with every
    # namespace's inventory + runs grouped beneath it — no duplication, no inline `ns` tags.
    for i in full:
        e = i["e"]
        _setp(out, P_STRUCT)
        state_txt = (
            "connected"
            if i["reachable"]
            else ("refreshing" if e["state"] == "refreshing" else "auth✗" if e["state"] == "auth" else "unreachable")
        )
        out.append("")
        # STRUCTURAL COLOR, disjoint from status hues (green/red/yellow[/cyan]): the CLUSTER level is bold BLUE,
        # applied ONLY to the `━━` rule segments — the header TEXT stays default-foreground so the words are
        # readable on any background.
        # The GPU product is NO LONGER crammed onto the bar as `[TYPE]` — it leads the CAPACITY sub-branch below.
        stale_badge = _stale_badge(e.get("stale", ""))
        out.append(
            _color_bars(
                _rule(e["name"], "", extra="· " + state_txt + stale_badge, label="CLUSTER"),
                paint.blue,
            )
        )
        if stale_badge:
            # Everything below this bar — CAPACITY, INSTALLED, RUN / SERVER — is that old frame. Say it once,
            # in words, so no row underneath can be mistaken for a live reading.
            out.append(
                paint.grey(_G_RAIL) + paint.yellow("⚠ CAPACITY / RUN state below is from the last good frame, NOT live")
            )
        if e.get("profiles", 1) > 1:
            _setp(out, P_CAPACITY)
            out.append(paint.grey(_G_RAIL) + paint.dim(f"({e['profiles']} profiles map to this cluster)"))
        # CAPACITY — the cluster's hardware answer to "what can I launch here": GPU product · nodes · FREE whole
        # nodes · biggest launchable run. A CLUSTER-level CHILD on a `├─` branch (never last — the NAMESPACE
        # tree always follows), exactly the idiom the INSTALLED / RUN·SERVER sections use one level down; its
        # wrapped note keeps the `│  ` continuation. Shown for ACTIVE clusters too (it used to appear
        # only on the idle one-liner), from data fleet already fetched — no new cluster call.
        cap = _capacity_line(e, fast=fast) if i["reachable"] else ""
        _setp(out, P_CAPACITY)
        if cap:
            out.append(paint.grey("│"))  # rail: same breathing room the namespace branches get
            out.append(paint.grey(_G_MID) + paint.bold("CAPACITY") + "   " + cap)
            note = _whole_node_note(e, fast=fast)
            if note:  # dim: a real blocker, but not a status hue (structure/status stay disjoint)
                out.append(paint.grey(_G_RAIL) + "            " + paint.dim(note))
        # MODEL LOAD — the per-model-cache checkpoint-load queue, a CLUSTER-level sub-branch beside CAPACITY.
        # Rendered ONLY when a lock actually exists (live holder, or a stale lock worth flagging); it stays
        # entirely off the happy path, and stays absent when Leases are unreadable rather than implying
        # "nothing is loading".
        for _ml in (e.get("model_load") or []) if i["reachable"] else []:
            out.append(paint.grey(_G_RAIL) + paint.bold("MODEL LOAD") + " " + _model_load_line(_ml, now))
        # INIT — cluster-level readiness, shown only when blocked/not-ready (a ready cluster's init is silent).
        _setp(out, P_STRUCT)
        if i["init_block"]:
            txt = stage1_state(e.get("readiness"))[2]
            body = txt[len("init ") :] if txt.startswith("init ") else txt
            out.append(
                paint.grey(_G_MID)
                + paint.bold("INIT")
                + "  "
                + paint.red(body)
                + paint.dim(f"  — fix: llmb-k8s init {e['name']}")
            )
        # per-NAMESPACE sub-blocks, rendered as a box-drawing TREE hanging off the cluster bar. HIERARCHY:
        # CLUSTER bar (bold blue rule) → each NAMESPACE a magenta `━━ NAMESPACE … ━━` bar on a ├─/└─ branch → two
        # sections (white INSTALLED + RUN / SERVER headers) on ├─/└─ branches → rows on ├─/└─ leaves, all threaded
        # by grey │ guides that DOUBLE as the breathing-room separators. BOTH sections render for every namespace
        # fleet manages (INSTALLED always present, '— nothing installed —' when empty; RUN / SERVER 'no active
        # runs' when idle). An UNREACHABLE cluster's run state is UNKNOWN → its per-ns RUN branch is suppressed.
        groups = _ns_groups(i["inv_rows"], i["run_rows"], i["sto_rows"], i["mdl_rows"])
        _render_ns_tree(
            out,
            e,
            groups,
            W,
            paint,
            reachable=i["reachable"],
            detail=detail,
            unavail=inventory_note(i["reachable"] and e.get("inventory_unavailable", False), fast),
            ledger_unknown=i.get("mdl_unknown", ""),
        )
        _setp(out, P_RUN)  # the cluster tail below is actionable (unreachable / GPUs-held)
        # cluster-level tail: an unreachable cluster's RUN state is unknown (only its stamps rendered above);
        # a reachable cluster holding GPUs with no resolved run row still must not read idle.
        if not i["reachable"]:
            connect = e.get("connect") or (f"tsh kube login {e.get('context')}" if e.get("context") else "")
            hint = f"run `{connect}`" if connect else "connect to this cluster"
            out.append("  " + paint.dim(f"(run state unknown — cluster unreachable · {hint})"))
        elif i.get("held") and not any(g["runs"] for g in groups):
            out.append(
                "  "
                + paint.c(
                    "33",
                    f"● our GPUs held ({_gpus(e['ours_gpu'])}) — see `llmb-k8s fleet` for detail",
                )
            )

    # ── IDLE-but-CONNECTED clusters: ONE compact line each (so the operator sees it exists + its capacity) ──
    _setp(out, P_IDLE)
    if idle_conn:
        out.append("")
        out.append(paint.dim("idle · connected:"))
        for i in idle_conn:
            e = i["e"]
            parts = [f"{e['name']}{_gtype(e)}"]
            free = "" if fast else _free_note(e)
            if free:
                parts.append(free)
            if e["ours_gpu"] > 0:
                parts.append(f"ours {_gpus(e['ours_gpu'])} held")  # a held (parked) footprint — benign, not orphaned
            if i["r_key"] == "ready":
                parts.append("init ✓")
            if i["i_key"] == "ok":
                parts.append("staged ✓")
            # THE MODEL GAP, ON THE ONE LINE THIS CLUSTER GETS. A cluster with no weights and one whose
            # weights are all present rendered identically here, so "nothing installed" was indistinguish-
            # able from "ready to run". A collapsed cluster is not a reason to say nothing about it.
            mnote = _idle_models_note(i.get("mdl_rows") or [])
            if mnote:
                parts.append(mnote)
            if e.get("profiles", 1) > 1:
                parts.append(f"{e['profiles']} profiles")
            out.append("  " + paint.dim("· " + " · ".join(parts)))

    # ── IDLE + UNREACHABLE / refreshing → a single terse tail (hidden, not a full noisy block) ──
    tail = []
    if idle_unreach:
        names = ", ".join(i["e"]["name"] for i in idle_unreach)
        tail.append(f"+{len(idle_unreach)} unreachable/idle ({names})")
    if refreshing and not any(i["full"] and i["e"]["state"] == "refreshing" for i in infos):
        tail.append(f"+{len(refreshing)} refreshing")
    if tail:
        out.append("")
        out.append("  " + paint.dim(" · ".join(tail)))

    # No trailing legend/explainer block — the tree + labels are intended to read on their own. (Actionable
    # signals like `orphan(N GPUs held)` / a run's `reclaim: …` hint already live inline on the rows themselves.)
    _setp(out, P_CHROME)
    bl = build_line(build)
    if bl:
        out.append("")
        out.append("  " + paint.dim(bl))
    lines = fit_viewport(out, out.prio, viewport_rows, viewport_cols, out.lbl)
    return "\n".join(lines) + "\n"


def render(
    entries: list,
    now: datetime,
    *,
    wide: bool,
    gpu_only: bool,
    color: bool,
    show_idle: bool = False,
    fast: bool = False,
    detail: bool = False,
    history_secs: int = DEFAULT_HISTORY_SECS,
    history_n: int = None,
    failed_only: bool = False,
    stages: bool = False,
    build: str = "",
    viewport_rows: int = 0,
    viewport_cols: int = 0,
) -> str:
    # `--stages` is a DISTINCT, redesigned pane (cluster → namespace → INIT/INSTALL/RUN sub-sections, idle
    # clusters collapsed). The default active-runs-first pane below is untouched.
    if stages:
        return render_stages(
            entries,
            now,
            color=color,
            fast=fast,
            gpu_only=gpu_only,
            detail=detail,
            history_secs=history_secs,
            history_n=history_n,
            build=build,
            viewport_rows=viewport_rows,
            viewport_cols=viewport_cols,
        )
    paint = Paint(color)
    out = []

    connected = [e for e in entries if e["state"] == "connected"]
    disconnected = [e for e in entries if e["state"] in ("auth", "unreach")]
    refreshing = [e for e in entries if e["state"] == "refreshing"]
    n_conn, n_disc = len(connected), len(disconnected)
    g_ours = sum(c["ours_gpu"] for c in connected)
    occ_vals = [c["occupied"] for c in connected if c["occupied"] is not None]
    tot_vals = [c["total"] for c in connected if c["total"] is not None]
    partial = (len(occ_vals) < n_conn) or (len(tot_vals) < n_conn)
    ts = now.strftime("%Y-%m-%d %H:%M:%SZ")

    actives = _collect_active(connected, now)
    # fleet-wide RECENTLY-FAILED runs (within the default window) — so a cluster whose only news is a death
    # can never read as a bare 'idle': the count surfaces in the headline and the run shows as a ✗ row below.
    recent_failed = sum(
        len(_recent_terminal(c, now, window_secs=FAILED_HISTORY_SECS, limit=None, failed_only=True)[0])
        for c in connected
    )

    # ── LINE 1 — answer at a glance: active runs · my GPUs · soonest ETA · recent failures | fleet capacity ──
    eta = _soonest_eta(actives)
    fleet = f"fleet {n_conn} up"
    if n_disc:
        fleet += f" · {n_disc} auth✗"
    if refreshing:
        fleet += f" · {len(refreshing)} refreshing"
    if not fast and tot_vals and occ_vals:
        fleet += f" · {sum(occ_vals)}/{sum(tot_vals)} GPUs used" + ("*" if partial else "")
    line1 = f"ACTIVE  {len(actives)} run{'' if len(actives) == 1 else 's'} · {_gpus(g_ours)} (ours)"
    if eta:
        line1 += f" · {eta}"
    if recent_failed:
        line1 += paint.red(f" · ⚠ {recent_failed} recently FAILED")
    line1 += f"   |   {fleet} · {ts}"
    out.append(paint.bold(line1))
    breakdown = _type_breakdown(connected)
    if breakdown:
        out.append(paint.dim("        " + breakdown + (" · fast (capacity skipped)" if fast else "")))

    # ── 3-STAGE JOURNEY: per-cluster init→install→run state (LOCAL stamps + this frame's live runs). Stages
    #    1&2 come from local .state stamps, so a disconnected cluster STILL shows its init/install progress. ──
    journeys = {}  # name → (readiness, installs, active_count, reachable)
    if stages:
        for e in entries:
            reachable = e["state"] == "connected"
            ac = _cluster_active_count(e, actives) if reachable else 0
            journeys[e["name"]] = (
                e.get("readiness"),
                e.get("installs") or {},
                ac,
                reachable,
            )
        stage_keys = [
            (stage1_state(r)[0], stage2_state(inst)[0], stage3_state(ac, rc)[0])
            for (r, inst, ac, rc) in journeys.values()
        ]
        out.append(fleet_journey_summary(stage_keys, paint))

    def _emit_journey(e):
        if not stages:
            return
        r, inst, ac, rc = journeys[e["name"]]
        out.append("   " + journey_line(r, inst, ac, rc, paint))

    # ── per-cluster GLOBAL column widths so every cluster's table lines up ──
    blocks = {
        e["name"]: _cluster_trows(
            e,
            now,
            gpu_only=gpu_only,
            detail=detail,
            show_idle=show_idle,
            history_secs=history_secs,
            history_n=history_n,
            failed_only=failed_only,
        )
        for e in connected
    }
    allrows = [r for rows, _ in blocks.values() for r in rows]
    W = _col_widths(allrows) if allrows else None

    # ── per-cluster nested tree ──
    for e in entries:
        out.append("")
        if e["state"] == "connected":
            out.append(paint.bold(_rule(e["name"], e.get("gpu_type", ""))))
            _emit_journey(e)
            out.append("   " + _ns_line(e, fast=fast))
            rows, hidden = blocks[e["name"]]
            if rows and W:
                out.append(_TREE_INDENT + paint.dim(_col_header(W)))
                for r in rows:
                    out.append(_TREE_INDENT + _fmt_trow(r, W, paint, wide=wide, detail=detail))
            # ── footer roll-up. Vocabulary is explicit (see the legend line): every token names WHAT it counts,
            #    and outcomes are split ✓ succeeded / ✗ FAILED so a death never hides inside a generic 'done'. ──
            cc = _collapse_counts(e)
            has_active = any(r["level"] != "term" for r in rows)
            seg = [] if has_active else ["no active runs of ours"]
            for k, lbl in (
                ("idle", "idle-server"),
                ("parked", "parked-run"),
                ("infra", "infra-pod"),
            ):
                if cc[k]:
                    seg.append(f"{cc[k]} {lbl}{'s' if cc[k] != 1 else ''}")
            if cc["infra_jobs"]:
                seg.append(f"{cc['infra_jobs']} helper-job{'s' if cc['infra_jobs'] != 1 else ''}")
            footer = paint.dim(" · ".join(seg)) if seg else ""
            # terminal-outcome tally (all non-infra terminal jobs in-ns): ✓N succeeded · ✗M FAILED(red).
            tally = []
            if cc["succeeded"]:
                tally.append(f"✓{cc['succeeded']} done")
            fail_txt = paint.red(f"✗{cc['failed']} FAILED") if cc["failed"] else ""
            if tally or fail_txt:
                joined = " · ".join([paint.dim(t) for t in tally] + ([fail_txt] if fail_txt else []))
                footer = f"{footer} · {joined}" if footer else joined
            if cc["orphan"]:
                orph = paint.red(f"{cc['orphan']} orphan({_gpus(cc['orphan_gpu'])} held)")  # holds GPU → reap candidate
                footer = f"{footer} · {orph}" if footer else orph
            if hidden > 0:
                note = paint.dim(f"+{hidden} older run{'s' if hidden != 1 else ''} ended (--history to show)")
                footer = f"{footer} · {note}" if footer else note
            if footer:
                out.append(_TREE_INDENT + footer)
        elif e["state"] == "refreshing":
            out.append(paint.bold(_rule(e["name"], e.get("gpu_type", ""))))
            _emit_journey(e)  # init/install still known from local stamps while this frame refreshes
            out.append("   " + paint.dim("…refreshing (slow to respond this frame)"))
        else:
            ctx = e.get("context") or "(ambient)"
            connect = e.get("connect") or (f"tsh kube login {ctx}" if e.get("context") else "")
            reason = "auth✗" if e["state"] == "auth" else "unreachable"
            det = e.get("detail") or ("credentials missing/expired" if e["state"] == "auth" else "API not answering")
            hint = f"run `{connect}`" if connect else "connect to this cluster"
            out.append(paint.bold(_rule(e["name"], e.get("gpu_type", ""))))
            # AUTH-ROBUST: even with no cluster access, STAGES 1&2 render from the local .state stamps, so a
            # disconnected cluster still shows how far along init/install got — only STAGE 3 reads `unknown`.
            _emit_journey(e)
            out.append("   " + paint.red(f"{reason} ({det}) — {hint}"))

    # No trailing legend/explainer block — the active-runs-first pane is meant to read on its own. Actionable
    # signals stay INLINE where they matter: a run's ⚠ over-median/near-deadline flag, a ✗ row's `why: … · logs:`
    # one-liner, an orphan's held-GPU footer, the `+N older runs … (--history to show)` note.
    bl = build_line(build)
    if bl:
        out.append("")
        out.append("  " + paint.dim(bl))
    # FLAT pane: already ordered active-runs-first, so an all-P_RUN fit degrades to an honest
    # head-keep + "N of M lines hidden" footer rather than a silent scroll-off.
    lines = fit_viewport(out, [P_RUN] * len(out), viewport_rows, viewport_cols)
    return "\n".join(lines) + "\n"


# ── I/O shell (impure: reads files listed in the meta.tsv) ──────────────────────────────────────


def _read_json(path: Path):
    try:
        txt = path.read_text().strip()
        return json.loads(txt) if txt else None
    except (OSError, json.JSONDecodeError):
        return None


def _read_profile_envs(pdir: Path, spec: str) -> list:
    """The parsed cluster-profile dicts for one cluster row, from meta field 10 (comma-separated FILE names,
    resolved against --profiles-dir). ALL of them: several profiles collapse onto one physical-cluster row and
    each may name a different MODEL_CACHE_PVC, so the ledger matches on the UNION of their resolved claims.

    A profile that cannot be read contributes nothing rather than raising — the ledger then reports the
    affected models as having no configured claim, which is true and says so, instead of dying mid-frame.
    """
    out = []
    for nm in str(spec or "").split(","):
        nm = nm.strip()
        if not nm:
            continue
        p = Path(nm) if Path(nm).is_absolute() else (pdir / nm)
        try:
            out.append(parse_profile_env(p))
        except (OSError, UnicodeDecodeError):  # missing/unreadable, or not text at all
            continue
    return out


def _split_ns(nsread_j, wd, name):
    """Split the combined response into pod, deployment, and Job lists by resource kind. Use each
    item's `kind`. Falls back to the legacy separate files if nsread is absent (older last-good snapshots).
    """
    if nsread_j and "items" in nsread_j:
        pods, deps, jobs = [], [], []
        for it in nsread_j["items"]:
            k = it.get("kind", "")
            (pods if k == "Pod" else deps if k == "Deployment" else jobs if k == "Job" else []).append(it)
        return {"items": pods}, {"items": deps}, {"items": jobs}
    return (
        _read_json(wd / f"{name}.pods.json"),
        _read_json(wd / f"{name}.deploys.json"),
        _read_json(wd / f"{name}.jobs.json"),
    )


def _split_llmb(allllmb_j):
    """The cluster-wide `get deployments,jobs -A -l managed-by=llmb-recipe` read → (our_deploys, our_jobs)
    lists across ALL namespaces. Missing/empty → ([], []). PURE."""
    deps, jobs = [], []
    for it in (allllmb_j or {}).get("items") or []:
        k = it.get("kind", "")
        (deps if k == "Deployment" else jobs if k == "Job" else []).append(it)
    return deps, jobs


def make_median_lookup(root: Path = None):
    """A cached `cell_name -> median wall_seconds` resolver over the committed recipes/ tree (source (b) for
    expected-runtime). Best-effort: only resolves when EXACTLY ONE recipes/**/<cell>/runs.jsonl exists, so
    an ambiguous leaf name never yields a wrong number. Mirrors submit.sh's median filter. Returns None on
    any miss. Filesystem-touching but read-only; unresolvable cells simply fall through to the deadline.
    """
    root = root or Path(__file__).resolve().parent.parent
    cache: dict = {}

    def lookup(cell_name):
        if not cell_name:
            return None
        if cell_name in cache:
            return cache[cell_name]
        cache[cell_name] = None
        try:
            hits = list((root / "recipes").glob(f"**/{cell_name}/runs.jsonl"))
        except OSError:
            hits = []
        if len(hits) == 1:
            durs = []
            try:
                for line in hits[0].read_text().splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    w = r.get("wall_seconds")
                    if isinstance(w, (int, float)) and w > 0 and ("metric" in r or "value" in r or r.get("gpu_count")):
                        durs.append(float(w))
            except OSError:
                durs = []
            if durs:
                durs.sort()
                n = len(durs)
                cache[cell_name] = durs[n // 2] if n % 2 else (durs[n // 2 - 1] + durs[n // 2]) / 2
        return cache[cell_name]

    return lookup


def make_result_lookup(root: Path = None):
    """A cached `(cell, run_id) -> (metric, value)` resolver over committed recipes/**/<cell>/runs.jsonl — the
    HEADLINE result of a ✓ done run, IF it has already been collected/published there (each runs.jsonl row
    carries {run_id, metric, value}). Same read-only, single-hit-glob discipline as make_median_lookup (an
    ambiguous leaf name → no guess). Returns None when the (cell, run_id) isn't in a runs.jsonl yet — a fresh
    completion not yet collected — so the caller degrades to the `collect <run-id>` pointer instead of inventing
    a number. Honors FLEET_RECIPES_ROOT (a dir containing recipes/) for offline tests; else the repo tree.
    """
    if root is None:
        env = os.environ.get("FLEET_RECIPES_ROOT")
        root = Path(env) if env else Path(__file__).resolve().parent.parent
    cache: dict = {}

    def lookup(cell_name, run_id):
        if not cell_name or not run_id:
            return None
        key = (cell_name, str(run_id))
        if key in cache:
            return cache[key]
        cache[key] = None
        try:
            hits = list((root / "recipes").glob(f"**/{cell_name}/runs.jsonl"))
        except OSError:
            hits = []
        if len(hits) == 1:
            try:
                for line in hits[0].read_text().splitlines():
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if str(r.get("run_id")) == str(run_id) and r.get("metric") and r.get("value") is not None:
                        cache[key] = (str(r["metric"]), r["value"])
                        break
            except OSError:
                pass
        return cache[key]

    return lookup


_RUNGS_RE = None


def make_rungs_lookup(root: Path = None):
    """A cached `cell_name -> [rungs]` resolver over committed recipes/ (the committed fallback for the SWEEP
    column when no LIVE_RUNGS/CONCURRENCIES env is on the Job). Resolves only when EXACTLY ONE
    recipes/**/<cell>/recipe.yaml exists (ambiguous leaf → no guess). Regex-parses `bench.sweep_concurrency`
    off the recipe text (only NON-comment lines match), so this stays yaml-dependency-free like the rest of
    this pure renderer. Read-only."""
    import re as _re

    global _RUNGS_RE
    if _RUNGS_RE is None:
        _RUNGS_RE = _re.compile(r"^[ \t]*sweep_concurrency:[ \t]*\[([^\]]*)\]", _re.MULTILINE)
    root = root or Path(__file__).resolve().parent.parent
    cache: dict = {}

    def lookup(cell_name):
        if not cell_name:
            return None
        if cell_name in cache:
            return cache[cell_name]
        cache[cell_name] = None
        try:
            hits = list((root / "recipes").glob(f"**/{cell_name}/recipe.yaml"))
        except OSError:
            hits = []
        if len(hits) == 1:
            try:
                m = _RUNGS_RE.search(hits[0].read_text())
            except OSError:
                m = None
            if m:
                vals = [t.strip() for t in m.group(1).split(",") if t.strip()]
                if vals:
                    cache[cell_name] = vals
        return cache[cell_name]

    return lookup


AUTH_HINTS = (
    "unauthorized",
    "expired",
    "credential",
    "login",
    "exec plugin",
    "tsh",
    "x509",
    "certificate",
    "does not exist",
    "no configuration has been provided",
    "you must be logged in",
)
UNREACH_HINTS = (
    "no such host",
    "connection refused",
    "i/o timeout",
    "timeout",
    "timed out",
    "eof",
    "unable to connect",
    "server could not find",
    "tcp",
)


def _classify_fail(status: str, errfile: str) -> tuple[str, str]:
    """(state, detail) for a failed cluster. state ∈ {auth, unreach}; fleet.sh may pre-decide via status."""
    detail = ""
    if errfile and Path(errfile).is_file():
        t = Path(errfile).read_text().strip()
        if t:
            detail = t.splitlines()[-1].strip()
    if status == "AUTH":
        return "auth", detail[:70]
    if status == "UNREACH":
        return "unreach", detail[:70]
    low = detail.lower()
    if any(k in low for k in AUTH_HINTS):
        return "auth", detail[:70]
    if any(k in low for k in UNREACH_HINTS):
        return "unreach", detail[:70]
    return "unreach", detail[:70]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="render the multi-cluster fleet view (pure; fed by fleet.sh)")
    ap.add_argument("--meta", required=True, help="TSV: name<TAB>ctx<TAB>ns<TAB>status<TAB>errfile")
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--wide", action="store_true")
    ap.add_argument("--gpu-only", action="store_true")
    ap.add_argument(
        "--idle",
        dest="show_idle",
        action="store_true",
        help="list idle 0/0 servers individually (default: collapse to one summary line)",
    )
    ap.add_argument(
        "--fast",
        action="store_true",
        help="ours-only: cluster-scoped capacity reads were skipped, show OURS without the triple",
    )
    ap.add_argument(
        "--detail",
        action="store_true",
        help="expand each cluster's full job/server tree under its summary line",
    )
    ap.add_argument(
        "--history",
        default="",
        help="deeper terminal-run history: N (last N runs) or a duration (e.g. 6h, 90m, 2d)",
    )
    ap.add_argument(
        "--failed",
        action="store_true",
        help="show only ✗ FAILED terminal runs (widened window)",
    )
    ap.add_argument(
        "--all",
        action="store_true",
        help="show everything: full tree + all terminal history + idle servers",
    )
    ap.add_argument("--color", dest="color", action="store_true")
    ap.add_argument("--no-color", dest="color", action="store_false")
    ap.add_argument("--now", default="", help="override 'now' (RFC3339 UTC) for deterministic tests")
    ap.add_argument(
        "--build",
        default="",
        help="build stamp (sha · branch · checkout) footer — which fleet produced this pane",
    )
    ap.add_argument(
        "--stages",
        action="store_true",
        help="3-STAGE journey per cluster: ① init readiness → ② recipe install → ③ live run "
        "(① & ② from local .state stamps → shown even for unreachable clusters)",
    )
    ap.add_argument(
        "--profiles-dir",
        default="",
        help="cluster-profiles dir holding .state/<profile>.{readiness.json,install.jsonl} "
        "(the S1/S2 journey stamp sources; defaults next to this script)",
    )
    ap.add_argument(
        "--viewport-rows",
        type=int,
        default=0,
        help="fit the frame into N terminal ROWS (0 = unbounded). --watch passes the live "
        "terminal height: a taller frame is COLLAPSED BY PRIORITY with an explicit "
        "'N lines hidden' marker, never silently scrolled off the alt-screen.",
    )
    ap.add_argument(
        "--viewport-cols",
        type=int,
        default=0,
        help="terminal WIDTH, so line WRAP is counted against the row budget (0 = ignore wrap)",
    )
    ap.set_defaults(color=False)
    a = ap.parse_args(argv)

    # history selection: --all ⇒ everything; --failed ⇒ failures over a day; --history N|dur ⇒ deeper.
    history_secs, history_n, failed_only = DEFAULT_HISTORY_SECS, None, a.failed
    detail = a.detail or a.all
    show_idle = a.show_idle or a.all
    if a.all:
        history_secs = None
    elif a.failed:
        history_secs = FAILED_HISTORY_SECS
    if a.history:
        h = a.history.strip().lower()
        if h in ("all", "*"):
            history_secs = None  # every terminal run, any age
        elif h.isdigit():
            history_n, history_secs = int(h), None  # last-N runs, any age
        else:
            secs = _parse_duration(h)
            if secs is not None:
                history_secs = secs

    now = _parse_ts(a.now) or datetime.now(timezone.utc)
    wd = Path(a.workdir)
    median_lookup = make_median_lookup()
    rungs_lookup = make_rungs_lookup()
    result_lookup = make_result_lookup()

    # WHERE THE CLUSTER PROFILES LIVE. Resolved for EVERY pane, not just `--stages`: the MODEL LEDGER needs
    # `MODEL_CACHE_PVC` (via model_cache.resolve_cache_claim) to know which claim each model's weights belong
    # in, and without it the ledger cannot tell "the claim is empty" from "I don't know which claim".
    pdir = Path(a.profiles_dir) if a.profiles_dir else (Path(__file__).resolve().parent.parent / "cluster-profiles")

    entries = []
    for ln in Path(a.meta).read_text().splitlines():
        if not ln.strip():
            continue
        # name ctx ns status errfile connect stale gpu_product profile_count profile_files
        #   stale (7th) = "" fresh · int secs = shown from last-good frame (--watch laggard) · "pending" = new
        #   profile_files (10th) = EVERY profile that maps to this physical cluster, comma-separated. All of
        #   them, not just the winner: several profiles collapse to one cluster row and each may pin a
        #   different MODEL_CACHE_PVC — passing only the first hides the second's claim entirely.
        parts = (ln.split("\t") + [""] * 10)[:10]
        name, ctx, ns, status, errfile, connect, stale, gpu_product, pcount, pfiles = parts
        gpu_type = friendly_gpu(gpu_product)
        profile_envs = _read_profile_envs(pdir, pfiles)
        try:
            pcount_i = int(pcount)
        except ValueError:
            pcount_i = 1
        if status == "OK":
            pods_j, deploys_j, jobs_j = _split_ns(_read_json(wd / f"{name}.nsread.json"), wd, name)
            _allllmb = _read_json(wd / f"{name}.allllmb.json")  # None when the cluster-wide llmb read FAILED
            all_deploys, all_jobs = _split_llmb(_allllmb)
            leases_j = _read_json(wd / f"{name}.leases.json")  # None when RBAC-forbidden / --fast
            pvcs_j = _read_json(wd / f"{name}.pvcs.json")  # None when RBAC-forbidden / --fast
            e = build_cluster(
                name,
                ctx,
                ns,
                pods_j,
                deploys_j,
                jobs_j,
                _read_json(wd / f"{name}.nodes.json"),
                _read_json(wd / f"{name}.all.json"),
                median_lookup=median_lookup,
                rungs_lookup=rungs_lookup,
                result_lookup=result_lookup,
                now=now,
                all_deploys=all_deploys,
                all_jobs=all_jobs,
                leases_j=leases_j,
                pvcs_j=pvcs_j,
                workloads_read=_allllmb is not None,
            )
            e["stale"] = stale if (stale and stale != "0") else ""
            e["gpu_type"] = gpu_type or friendly_gpu(_nodes_gpu_product(wd / f"{name}.nodes.json"))
            e["profiles"] = pcount_i
            e["profile_envs"] = profile_envs
            entries.append(e)
        elif status == "PENDING":
            entries.append(
                {
                    "state": "refreshing",
                    "name": name,
                    "context": ctx,
                    "namespace": ns,
                    "connect": connect,
                    "gpu_type": gpu_type,
                    "profiles": pcount_i,
                    "profile_envs": profile_envs,
                }
            )
        else:
            state, fail_detail = _classify_fail(status, errfile)  # NB: not `detail` — that's the view flag
            entries.append(
                {
                    "state": state,
                    "name": name,
                    "context": ctx,
                    "namespace": ns,
                    "detail": fail_detail,
                    "connect": connect,
                    "stale": stale,
                    "gpu_type": gpu_type,
                    "profiles": pcount_i,
                    "profile_envs": profile_envs,
                }
            )

    entries.sort(key=lambda e: e["name"])  # every configured cluster, deterministic order

    # ── attach the STAGE-1/2 journey stamps (local, per-profile; auth-independent) to every entry ──
    if a.stages:
        for e in entries:
            e["readiness"] = read_readiness_stamp(pdir, e["name"])
            e["installs"] = read_install_stamps(pdir, e["name"])

    sys.stdout.write(
        render(
            entries,
            now,
            wide=a.wide,
            gpu_only=a.gpu_only,
            color=a.color,
            show_idle=show_idle,
            fast=a.fast,
            detail=detail,
            history_secs=history_secs,
            history_n=history_n,
            failed_only=failed_only,
            stages=a.stages,
            build=a.build,
            viewport_rows=a.viewport_rows,
            viewport_cols=a.viewport_cols,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
