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

"""selftest_parallel_repro_spread.py — offline unit tests for parallel_repro.sh
model-cache spread: auto-discovery, round-robin assignment, and stagger widening.

No cluster, no network. parallel_repro.sh is sourced with PARALLEL_REPRO_LIB_ONLY=1
(which returns before the tool body runs), exposing three PURE bash helpers we drive
with a MOCKED `kubectl get pvc` line list:

  A. mcp_select_replicas <base>  — the auto-spread set discovered by default when
     MODEL_CACHE_PVCS is unset: the base PVC first (only if Bound) then Bound
     <base>-r<N> replicas in ascending N. Non-Bound, artifacts, and unrelated PVCs
     are dropped. This is the reliability fix: spread is discovered, not opt-in.

  B. mcp_assign <N> <pvc...>    — round-robin one PVC per copy (i % npvcs), so N
     copies never all land on the same filesystem when >1 exists.

  C. mcp_eff_stagger <S> <N> <npvcs> — auto-widen: npvcs>=N → 5s token; 0<npvcs<N →
     S×ceil(N/npvcs) (busiest filesystem serves ceil(N/npvcs) loads, so serialize
     the pile-up); npvcs<=0 → unchanged (single profile PVC fallback).

…and the SELF-DETACH of the orchestrator itself (the fan-out loop is 15-30 min at N=8;
a session boundary inside it used to strand a sweep with every leg cloned and ZERO Jobs
applied — nothing detached existed yet to survive):

  D. should_self_detach — the decision matrix + the re-exec LOOP GUARD.
  E. spawn_detached     — really forks/setsid/execs a probe command and proves the child
     reparented to init (ppid==1) in its OWN process group, with its log written.
  F. end-to-end against a FAKE collection root (stub render.sh/run.sh, stub cell — no
     cluster, no network): default self-detaches and the child fires every leg; --dry-run
     and --foreground stay inline; the sentinel-carrying child never detaches again; and
     the durable plan.json/progress.jsonl record says which legs fired.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Mirrors spawn_detached's own helper in parallel_repro.sh, for the same reason: the CI image has no
# procps, so `ps` is not available to ask for another process's parent. procfs answers it with no
# binary; `ps` stays only as the macOS fallback. NOTE os.getpgid() is NOT a substitute here — it
# returns a process GROUP, and the assertion below is about the PARENT (reparent-to-init). Where the
# question really IS "my own group", this file uses os.getpgid(0) directly.
_HAVE_PROC = os.path.isdir("/proc/self")


def _ppid_of(pid: int) -> str:
    """PPID of `pid` as a string, or "" when the process is gone (the callers compare against "1"
    and treat "" as not-yet-observable, so the empty string must mean GONE, never 'unknown').
    """
    if _HAVE_PROC:
        try:
            data = open("/proc/%d/stat" % pid, "rb").read()
        except OSError:
            return ""
        try:
            # field 2 (comm) is parenthesised and may contain spaces AND ')' — split after the LAST ')'
            return str(int(data[data.rindex(b")") + 2 :].split()[1]))
        except (ValueError, IndexError):
            return ""
    return subprocess.run(["ps", "-o", "ppid=", "-p", str(pid)], capture_output=True, text=True).stdout.strip()


ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "parallel_repro.sh"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def _sh(snippet: str, stdin: str = "") -> str:
    """Source the tool lib-only, then run a bash snippet against the pure helpers."""
    code = f"PARALLEL_REPRO_LIB_ONLY=1 source {SCRIPT}\n{snippet}"
    p = subprocess.run(["bash", "-c", code], input=stdin, text=True, capture_output=True)
    assert p.returncode == 0, f"bash failed: {p.stderr}\n--snippet--\n{snippet}"
    return p.stdout.strip()


def select(base: str, pvcs: list[tuple[str, str]]) -> list[str]:
    blob = "".join(f"{n} {phase}\n" for (n, phase) in pvcs)
    out = _sh(f'printf "%s" "$(cat)" | mcp_select_replicas {base}', stdin=blob)
    return out.splitlines() if out else []


def assign(n: int, pvcs: list[str]) -> list[str]:
    out = _sh(f'mcp_assign {n} {" ".join(pvcs)}')
    return out.splitlines() if out else []


def eff_stagger(s: int, n: int, npvcs: int) -> int:
    return int(_sh(f"mcp_eff_stagger {s} {n} {npvcs}"))


# The file must at least parse.
p = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
check("parallel_repro.sh parses (bash -n)", p.returncode == 0, p.stderr.strip())

# ── A. auto-discovery / mcp_select_replicas ─────────────────────────────────
BASE = "serving-gb300-model-cache"
fleet = [
    (BASE, "Bound"),
    (f"{BASE}-r3", "Bound"),
    (f"{BASE}-r2", "Bound"),
    (f"{BASE}-r10", "Bound"),  # two-digit replica must sort AFTER r3, not after r1
    (
        f"{BASE}-r4",
        "Pending",
    ),  # not Bound → dropped (matches the FSx-still-provisioning case)
    (f"{BASE}-artifacts", "Bound"),  # not a -r<N> replica → dropped
    ("unrelated-model-cache", "Bound"),
    ("llmb-control", "Bound"),
]
got = select(BASE, fleet)
check(
    "discovery: base first, Bound replicas in ascending numeric order",
    got == [BASE, f"{BASE}-r2", f"{BASE}-r3", f"{BASE}-r10"],
    f"got={got}",
)
check("discovery: Pending replica excluded", f"{BASE}-r4" not in got)
check("discovery: -artifacts PVC excluded", f"{BASE}-artifacts" not in got)
check(
    "discovery: unrelated PVCs excluded",
    "unrelated-model-cache" not in got and "llmb-control" not in got,
)

# base itself not Bound (e.g. lost) ⇒ omitted; replicas still discovered
got2 = select(BASE, [(BASE, "Lost"), (f"{BASE}-r2", "Bound")])
check(
    "discovery: non-Bound base omitted, replica kept",
    got2 == [f"{BASE}-r2"],
    f"got={got2}",
)

# nothing matches ⇒ empty (caller then warns + falls back to the single profile PVC)
check("discovery: no matches → empty", select(BASE, [("other", "Bound")]) == [])

# ── B. round-robin / mcp_assign ─────────────────────────────────────────────
check(
    "round-robin: 5 copies across 2 PVCs cycles a,b,a,b,a",
    assign(5, ["a", "b"]) == ["a", "b", "a", "b", "a"],
)
check(
    "round-robin: N==npvcs is a bijection (one filesystem each)",
    assign(3, ["a", "b", "c"]) == ["a", "b", "c"],
)
check(
    "round-robin: single PVC → every copy same filesystem",
    assign(4, ["a"]) == ["a", "a", "a", "a"],
)
# no copy pair collides on a PVC when replicas>=copies
a = assign(4, ["p0", "p1", "p2", "p3"])
check("round-robin: replicas>=copies ⇒ all distinct", len(set(a)) == 4, f"got={a}")

# ── C. stagger auto-widening / mcp_eff_stagger ──────────────────────────────
check(
    "stagger: npvcs>=N → 5s token (one fs per load, no contention)",
    eff_stagger(60, 5, 5) == 5 and eff_stagger(60, 3, 8) == 5,
)
check(
    "stagger: npvcs<=0 → unchanged (single profile PVC fallback)",
    eff_stagger(60, 5, 0) == 60,
)
# 0<npvcs<N widens by ceil(N/npvcs); fewer replicas ⇒ wider stagger
check("stagger: 1 fs, 5 copies → 60×5 = 300s", eff_stagger(60, 5, 1) == 300)
check("stagger: 2 fs, 5 copies → 60×3 = 180s", eff_stagger(60, 5, 2) == 180)
check("stagger: 3 fs, 5 copies → 60×2 = 120s", eff_stagger(60, 5, 3) == 120)
# monotonic: as replicas grow toward N the effective stagger never increases
seq = [eff_stagger(60, 5, k) for k in (1, 2, 3, 4, 5)]
check(
    "stagger: monotonically non-increasing as replicas→N",
    seq == sorted(seq, reverse=True),
    f"seq={seq}",
)
# and it is strictly wider than base whenever there is any pile-up
check(
    "stagger: any pile-up (npvcs<N) widens beyond base 60s",
    all(eff_stagger(60, 5, k) > 60 for k in (1, 2, 3, 4)),
)

# ── C2. THE BASE CLAIM COMES FROM THE RESOLVER, NOT FROM RAW ${MODEL_CACHE_PVC} ─────────────
# Every other consumer resolves (model_cache.resolve_cache_claim); this one sourced the profile and read the
# GLOBAL MODEL_CACHE_PVC. On a cluster using a per-model key — which cluster-profiles/
# example-gpu-cluster.env.example now ships — the discovered spread set is then <global-claim>-rN: every leg
# mounts a replica of a filesystem that does not hold this model's weights while the model's own claim sits
# unused, and the legs fail model-not-found AFTER a full GPU allocation, on N nodes at once. Nothing
# downstream can catch it, because MODEL_CACHE_PVC_OVERRIDE outranks the profile inside the leg.
_PR = SCRIPT.read_text()
check(
    "parallel_repro derives _BASE_PVC from the resolver, not from the sourced profile var",
    'model_cache.py" resolve "$CELL"' in _PR and "_BASE_PVC=%q" not in _PR,
)
check(
    "parallel_repro captures the resolver's STDOUT only (a stderr banner must not join the name)",
    not re.search(r'_BASE_PVC="\$\(python3[^\n]*2>&1', _PR),
)
check(
    "parallel_repro does NOT fall back to the global claim when resolution fails "
    "(each leg then resolves for itself)",
    "NOT spreading across replicas" in _PR,
)

# BEHAVIOURAL, not textual: the two rules must actually disagree on the shipped example profile, or this
# would be a no-op refactor dressed up as a fix.
_EXPROF = ROOT / "cluster-profiles" / "example-gpu-cluster.env.example"
_nem = sorted(ROOT.glob("recipes/**/nemotron*/**/recipe.yaml"))
if _EXPROF.exists() and _nem:
    _cell = _nem[0].parent
    _old = subprocess.run(
        [
            "sh",
            "-c",
            f'set -a; . "{_EXPROF}" >/dev/null 2>&1; set +a; printf "%s" "${{MODEL_CACHE_PVC:-}}"',
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    _new = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "model_cache.py"),
            "resolve",
            str(_cell),
            str(_EXPROF),
        ],
        capture_output=True,
        text=True,
    ).stdout.strip()
    check(
        f"the old and new base-claim rules genuinely disagree for a per-model-key model "
        f"(profile var {_old!r} vs resolved {_new!r})",
        bool(_old) and bool(_new) and _old != _new,
        f"{_old!r} vs {_new!r}",
    )


# ── D. self-detach decision matrix + re-exec loop guard ─────────────────────
# should_self_detach <child_sentinel> <dry> <attach> <foreground>
def decide(child: int, dry: int, attach: int, fg: int) -> str:
    return _sh(f"should_self_detach {child} {dry} {attach} {fg}")


check("self-detach: DEFAULT invocation detaches", decide(0, 0, 0, 0) == "1")
check(
    "self-detach: --dry-run does NOT detach (its plan is read synchronously)",
    decide(0, 1, 0, 0) == "0",
)
check(
    "self-detach: --attach does NOT detach (it babysits legs inline)",
    decide(0, 0, 1, 0) == "0",
)
check("self-detach: --foreground escape hatch does NOT detach", decide(0, 0, 0, 1) == "0")
check(
    "self-detach: LOOP GUARD — the re-exec'd child never detaches again",
    decide(1, 0, 0, 0) == "0",
)
check(
    "self-detach: guard wins even for an otherwise-detaching child",
    decide(1, 0, 0, 0) == "0",
)


# ── E. spawn_detached really reparents to init ──────────────────────────────
# macOS has no setsid(1), and `nohup … & disown` leaves the child in the CALLER'S
# process group (a group-wide reap still takes it). The python double-fork +
# os.setsid() must give us: ppid==1 AND a process group of our own.
with tempfile.TemporaryDirectory() as td:
    log = Path(td) / "probe.log"
    # $PPID is a bash builtin (no binary). For the pgid, python's os.getpgid(0) reports the CALLING
    # process's group, and python inherits bash's group here — same answer as `ps -o pgid=`, minus the
    # procps dependency that the CI image does not satisfy.
    out = _sh(
        f'spawn_detached {log} bash -c \'echo ppid=$PPID pgid=$(python3 -c "import os;print(os.getpgid(0))"); '
        f'python3 -c "import time;time.sleep(20)"\''
    )
    parts = out.split()
    check(
        "spawn_detached: returns '<pid> <ppid>'",
        len(parts) == 2 and parts[0].isdigit(),
        f"out={out!r}",
    )
    if len(parts) == 2 and parts[0].isdigit():
        pid, reported = int(parts[0]), parts[1]
        check(
            "spawn_detached: reports the VERIFIED reparent (ppid=1), not a claim",
            reported == "1",
            out,
        )
        deadline, live_ppid, body = time.time() + 20, "", ""
        while time.time() < deadline:
            live_ppid = _ppid_of(pid)
            body = log.read_text() if log.exists() else ""
            if live_ppid and "ppid=" in body:
                break
            time.sleep(0.1)
        check(
            "spawn_detached: live child really has ppid=1 (reparented to init)",
            live_ppid == "1",
            f"ppid={live_ppid!r}",
        )
        check(
            "spawn_detached: log file created and written",
            "ppid=1" in body,
            f"log={body!r}",
        )
        # OUR OWN group — os.getpgid(0) answers this directly, no /proc parsing and no binary.
        my_pgid = str(os.getpgid(0))
        child_pgid = re.search(r"pgid=(\d+)", body)
        check(
            "spawn_detached: child is in its OWN process group (a killpg can't reach it)",
            bool(child_pgid) and child_pgid.group(1) != my_pgid,
            f"child={body!r} mine={my_pgid}",
        )
        # os.kill, not the `kill` BINARY: /bin/kill also ships in procps, so it is absent from the CI
        # image exactly like `ps` (the shell builtin masks this — `command -v kill` finds the builtin
        # and reports success, while subprocess needs the executable). Signal delivery is a syscall;
        # it never needed a process at all.
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass  # already gone, or not ours — nothing to clean up either way


# ── F. end-to-end against a FAKE collection root (no cluster, no network) ────
# scripts/parallel_repro.sh derives ROOT from its own location, so a temp tree with
# stub render.sh/run.sh exercises the REAL control flow — self-detach, clone, fire,
# durable record — while touching nothing outside the temp dir.
def fake_root(td: str) -> Path:
    root = Path(td) / "root"
    (root / "scripts").mkdir(parents=True)
    (root / "cell").mkdir()
    (root / "results").mkdir()
    shutil.copy2(SCRIPT, root / "scripts" / "parallel_repro.sh")
    (root / "cell" / "recipe.yaml").write_text("envelope:\n  name: stubcell\n")
    for name, body in (
        ("render.sh", 'echo "stub render $1"\n'),
        ("run.sh", 'echo "stub run.sh $1 $2 $3"\n'),
    ):
        p = root / "scripts" / name
        p.write_text("#!/usr/bin/env bash\n" + body)
        p.chmod(0o755)
    return root


def run_tool(root: Path, *args: str, env_extra: dict | None = None) -> subprocess.CompletedProcess:
    # NAMESPACE/KUBECONFIG neutered: the tool's best-effort sweep ConfigMap must never
    # reach a real cluster from a test (these selftests are strictly offline).
    env = {**os.environ, "NAMESPACE": "", "KUBE_CONTEXT": "", "KUBECONFIG": "/dev/null"}
    env.update(env_extra or {})
    return subprocess.run(
        [
            "bash",
            str(root / "scripts" / "parallel_repro.sh"),
            str(root / "cell"),
            "fakeprof",
            *args,
        ],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )


def sweep_dirs(root: Path) -> list[Path]:
    base = root / "results" / ".sweeps"
    return sorted(p for p in base.glob("rc*") if p.is_dir()) if base.exists() else []


with tempfile.TemporaryDirectory() as td:
    root = fake_root(td)

    # F1 — --dry-run stays INLINE (the caller must be able to read the plan now).
    p = run_tool(root, "2", "--dry-run")
    check("e2e --dry-run: exits 0 inline", p.returncode == 0, p.stderr[-400:])
    check(
        "e2e --dry-run: prints the plan to OUR stdout",
        "--dry-run" in p.stdout,
        p.stdout[-400:],
    )
    check(
        "e2e --dry-run: does NOT self-detach",
        "SELF-DETACHED" not in p.stdout,
        p.stdout[-300:],
    )

    # F2 — --foreground stays INLINE and completes the fan-out in-process.
    p = run_tool(root, "2", "--stagger", "1", "--foreground")
    check(
        "e2e --foreground: does NOT self-detach",
        "SELF-DETACHED" not in p.stdout,
        p.stdout[-300:],
    )
    check(
        "e2e --foreground: completes the fan-out inline",
        "2/2 legs applied" in p.stdout,
        p.stdout[-400:],
    )

with tempfile.TemporaryDirectory() as td:
    root = fake_root(td)

    # F3 — DEFAULT self-detaches: the parent returns a banner immediately, the child
    # (ppid=1) fires every leg and writes the durable record.
    p = run_tool(root, "3", "--stagger", "1")
    check("e2e default: parent exits 0 immediately", p.returncode == 0, p.stderr[-400:])
    check(
        "e2e default: banner announces SELF-DETACHED",
        "SELF-DETACHED" in p.stdout,
        p.stdout[-400:],
    )
    check(
        "e2e default: banner reports the VERIFIED ppid=1",
        "ppid=1" in p.stdout,
        p.stdout[-400:],
    )
    m_id = re.search(r"sweep-id:\s+(rc\S+)", p.stdout)
    m_pid = re.search(r"pid:\s+(\d+)", p.stdout)
    m_log = re.search(r"log:\s+(\S+orchestrator\.log)", p.stdout)
    check(
        "e2e default: banner prints sweep-id + pid + log path",
        bool(m_id and m_pid and m_log),
        p.stdout[-400:],
    )
    check(
        "e2e default: the fan-out did NOT run in the parent",
        "legs applied" not in p.stdout,
        p.stdout[-400:],
    )

    if m_id and m_log and m_pid:
        sid, log = m_id.group(1), Path(m_log.group(1))
        plan = root / "results" / ".sweeps" / sid / "plan.json"
        record = root / "results" / ".sweeps" / f"{sid}.json"
        deadline = time.time() + 120
        while time.time() < deadline:
            if plan.exists() and json.loads(plan.read_text() or "{}").get("state") == "fired":
                break
            time.sleep(0.2)
        d = json.loads(plan.read_text()) if plan.exists() else {}
        check(
            "e2e default: detached child wrote its log",
            log.exists() and log.stat().st_size > 0,
        )
        check(
            "e2e default: child completed the fan-out (plan state=fired)",
            d.get("state") == "fired",
            str(d),
        )
        check(
            "e2e default: plan records self_detached + pid + every planned leg",
            d.get("self_detached") is True
            and d.get("orchestrator_pid") == int(m_pid.group(1))
            and len(d.get("planned_run_ids", [])) == 3,
            str(d),
        )
        prog = root / "results" / ".sweeps" / sid / "progress.jsonl"
        lines = [json.loads(x) for x in prog.read_text().splitlines() if x.strip()] if prog.exists() else []
        check(
            "e2e default: progress.jsonl has one applied line per leg",
            len(lines) == 3 and all(x["state"] == "applied" for x in lines),
            str(lines),
        )
        check(
            "e2e default: durable sweep record written only after every leg",
            record.exists(),
        )
        check(
            "e2e default: exactly ONE sweep dir (no re-exec loop)",
            len(sweep_dirs(root)) == 1,
            str(sweep_dirs(root)),
        )

with tempfile.TemporaryDirectory() as td:
    root = fake_root(td)

    # F4 — the sentinel the parent sets on the child is a hard loop guard: a process
    # already carrying it runs INLINE, so re-exec can never recurse.
    p = run_tool(root, "2", "--stagger", "1", env_extra={"PARALLEL_REPRO_DETACHED_CHILD": "1"})
    check(
        "e2e loop guard: sentinel-carrying process runs inline",
        "SELF-DETACHED" not in p.stdout,
        p.stdout[-300:],
    )
    check(
        "e2e loop guard: it does the work itself",
        "2/2 legs applied" in p.stdout,
        p.stdout[-400:],
    )
    check(
        "e2e loop guard: no second sweep spawned",
        len(sweep_dirs(root)) == 1,
        str(sweep_dirs(root)),
    )


print()
if fails:
    print(f"FAIL — {len(fails)} check(s) failed: {', '.join(fails)}")
    sys.exit(1)
print(
    "OK — parallel_repro spread (discovery + round-robin + stagger widening) "
    "+ orchestrator self-detach (reparent-to-init, loop guard, durable progress) verified"
)
