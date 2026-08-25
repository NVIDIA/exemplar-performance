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

"""selftest_detach.py — offline guards for the DETACHED reproducibility launch flow.

A multi-hour repro launch must NOT stay attached to the operator's laptop (a session/context boundary reaps
the orchestrator ~hourly, losing the auto fetch/publish). So the launch does only the ~2-minute setup, then
EXITS; results are harvested later via `collect`.

  * run.sh --detach — setup (preflight → run-owner → stage → deploy → APPLY the bench Job) then EXIT 0;
    NO wait-ready / follow / fetch / publish / teardown. Writes the SAME durable run-id handles submit.sh
    writes (results/.submits/<id>.json + a llmb-submit-<id> index ConfigMap) so `collect` resolves it. The
    DEFAULT (no flag) and --attach/--wait keep the full attached lifecycle (it still WAITS).
  * parallel_repro.sh — DEFAULT detached: fire N `run.sh --detach` legs, write a durable sweep record
    (results/.sweeps/<id>.json + llmb-sweep-<id> CM), print `collect --sweep <id>`, EXIT. --attach = babysit.
  * collect (scripts/resilient_status.py) resolves a run.sh-detached run by its run-id handle.
  * the lane captures `launch_attestation.json` before it registers the Job and the detached index ConfigMap
    carries that exact hash for cross-machine collection.

Strategy: bash -n every edited script; drive run.sh --detach END-TO-END against a STUB scripts dir + a fake
kubectl (no cluster) and assert it applies the Job + adopts the run-owner, writes the handle, applies the
index CM, and NEVER calls fetch/publish; assert the DEFAULT path still reaches wait-ready; exercise
parallel_repro --dry-run (both modes) for real; and resolve a detached run through resilient_status.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def _chmodx(p: Path) -> None:
    p.chmod(p.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)


# ── 0. static lint: every edited script is valid bash ────────────────────────────────────────────────────
for s in ("run.sh", "parallel_repro.sh", "submit.sh"):
    r = subprocess.run(["bash", "-n", str(SCRIPTS / s)], capture_output=True, text=True)
    check(f"{s} is valid bash (bash -n)", r.returncode == 0, r.stderr.strip()[:200])


# ── shared harness: a STUB scripts dir + fake kubectl so run.sh runs offline ─────────────────────────────
# run.sh calls its sub-scripts by absolute $ROOT/scripts/… path, so we build a throwaway ROOT: the REAL run.sh
# (+ real pure helpers lane.py/run_id.py) beside 1-line STUBS for every heavy sub-script. A fake `kubectl` on
# PATH answers the handful of queries the detached path makes. This is a genuine end-to-end drive of run.sh,
# no cluster.
LLM_PERF_CELL = ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"

FAKE_KUBECTL = r"""#!/usr/bin/env bash
# Fake kubectl for the detached run.sh drive. Logs every call; answers the few reads run.sh makes.
# `get jobs` is served from a REGISTRY FILE ($JOBS_FILE, one `name|ownerUIDs|succeeded|failed` line per Job)
# that the stub bench script writes — so the Job's real NAME and OWNER are whatever the lane script chose,
# exactly as in-cluster. $KUBECTL_JOBS_FAIL=1 makes the list read ERROR (≠ "no such Job").
echo "$*" >> "$KLOG"
args="$*"
case "$args" in
  *"get jobs"*)
      [ "${KUBECTL_JOBS_FAIL:-0}" = 1 ] && exit 1
      cat "$JOBS_FILE" 2> /dev/null ; exit 0 ;;
  *"get namespace"*)       exit 0 ;;                     # namespace exists
  *"apply -f -"*)          cat >> "${APPLY_MANIFESTS:-/dev/null}"; echo '---' >> "${APPLY_MANIFESTS:-/dev/null}"; exit 0 ;;
  *"get job"*)             exit 0 ;;
  *)                       exit 0 ;;
esac
"""

STUB_EXIT0_SH = "#!/usr/bin/env bash\nexit 0\n"
# The bench/sweep stub: "applies" a Job into the fake cluster's registry under the name the LANE
# chose, stamped with the run-owner uid the real scripts stamp via `run_owner.sh adopt-job`, then blocks on
# its log-follow until run.sh kills it.
STUB_BENCH = r"""#!/usr/bin/env bash
CELL="$1"; RID="${3:-}"
NAME="$(python3 -c 'import sys,yaml;print(yaml.safe_load(open(sys.argv[1]+"/recipe.yaml"))["envelope"]["name"])' "$CELL")"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$ROOT/scripts/launch_attestation.py" "$CELL" "$RID" \
  --out "$ROOT/results/$RID/launch_attestation.json" >/dev/null
if [ "${STUB_APPLY_NOTHING:-0}" != 1 ]; then
  # STUB_UNOWNED_JOB=1 reproduces "applied but adopt-job never landed" (no ownerReference at all).
  OWNER="${RUN_OWNER_UID:-}"; [ "${STUB_UNOWNED_JOB:-0}" = 1 ] && OWNER=""
  echo "${NAME}-${STUB_KIND:-bench}-${STUB_RUN_ID_OVERRIDE:-$RID}|${OWNER}||" >> "$JOBS_FILE"
fi
[ -n "${STUB_EXTRA_JOB:-}" ] && echo "${STUB_EXTRA_JOB}|${RUN_OWNER_UID:-}||" >> "$JOBS_FILE"
[ "${STUB_EXIT_AFTER_APPLY:-0}" = 1 ] && exit 0
sleep 300
"""
STUB_EXIT0_PY = "#!/usr/bin/env python3\nimport sys\nsys.exit(0)\n"
# run_owner.sh: `ensure` must print eval-able RUN_OWNER_NAME/UID; every other verb is a no-op.
STUB_RUN_OWNER = (
    "#!/usr/bin/env bash\n"
    'echo "$*" >> "${RUN_OWNER_LOG:-/dev/null}"\n'
    'if [ "${1:-}" = "ensure" ]; then echo "RUN_OWNER_NAME=ro-test"; echo "RUN_OWNER_UID=uid-1"; fi\n'
    "exit 0\n"
)
# fetch_results.sh + publish.py: touch a marker if EVER called, so the detached test can assert they are NOT.
STUB_FETCH = '#!/usr/bin/env bash\ntouch "$FETCH_MARKER" 2>/dev/null || true\nexit 0\n'
STUB_PUBLISH = '#!/usr/bin/env python3\nimport os\nopen(os.environ.get("PUBLISH_MARKER","/dev/null"),"w").close()\n'
# wait_server_ready.sh: touch a marker (proves the ATTACHED path reached wait-ready), then FAIL fast so the
# attached drive stops without needing the whole sweep stubbed.
STUB_WAIT = '#!/usr/bin/env bash\ntouch "$WAIT_MARKER" 2>/dev/null || true\nexit 1\n'


def _build_root(tmp: Path, cell: Path) -> tuple[Path, dict]:
    sd = tmp / "scripts"
    sd.mkdir(parents=True)
    (tmp / "cluster-profiles").mkdir()
    (tmp / "results").mkdir()
    # profile: namespace + RDMA_FABRIC_PROBED (skip the fabric-probe block); no KUBE_CONTEXT.
    (tmp / "cluster-profiles" / "testprof.env").write_text("NAMESPACE=testns\nRDMA_FABRIC_PROBED=1\nOWNER=tester\n")
    # real run.sh under test + the pure helpers it shells out to
    for real in ("run.sh", "lane.py", "run_id.py", "launch_attestation.py", "recipe_hash.py"):
        (sd / real).write_text((SCRIPTS / real).read_text())
        _chmodx(sd / real)
    stubs = {
        "dryrun.sh": STUB_EXIT0_SH,
        "deploy.sh": STUB_EXIT0_SH,
        "stage-dataset.sh": STUB_EXIT0_SH,
        "sweep.sh": STUB_BENCH,
        "preflight.py": STUB_EXIT0_PY,
        "probe_fabric.py": STUB_EXIT0_PY,
        "provenance.py": STUB_EXIT0_PY,
        "recovery.py": STUB_EXIT0_PY,
        "run_summary.py": STUB_EXIT0_PY,
        "run_owner.sh": STUB_RUN_OWNER,
        "fetch_results.sh": STUB_FETCH,
        "publish.py": STUB_PUBLISH,
        "wait_server_ready.sh": STUB_WAIT,
    }
    for name, body in stubs.items():
        (sd / name).write_text(body)
        _chmodx(sd / name)
    kbin = tmp / "bin"
    kbin.mkdir()
    (kbin / "kubectl").write_text(FAKE_KUBECTL)
    _chmodx(kbin / "kubectl")
    # Some release-validation filesystems are mounted noexec. Source a tiny bash
    # function so the fake kubectl is interpreted by bash instead of executed directly.
    bash_env = kbin / "bash-env.sh"
    bash_env.write_text("kubectl() { bash \"" + str(kbin / "kubectl") + "\" \"$@\"; }\nexport -f kubectl\n")
    env = dict(os.environ)
    env["PATH"] = f"{kbin}:{env['PATH']}"
    env["BASH_ENV"] = str(bash_env)
    env["KLOG"] = str(tmp / "kubectl.log")
    env["APPLY_MANIFESTS"] = str(tmp / "applied-manifests.yaml")
    env["JOBS_FILE"] = str(tmp / "jobs.registry")
    env["RUN_OWNER_LOG"] = str(tmp / "run_owner.log")
    env["STUB_KIND"] = (
        subprocess.run(
            [sys.executable, str(sd / "lane.py"), str(cell), "kind"], capture_output=True, text=True
        ).stdout.strip()
        or "bench"
    )
    env["FETCH_MARKER"] = str(tmp / "FETCHED")
    env["PUBLISH_MARKER"] = str(tmp / "PUBLISHED")
    env["WAIT_MARKER"] = str(tmp / "WAITED")
    env.pop("KUBE_CONTEXT", None)
    return sd, env


# ── 1. run.sh --detach applies the Job + run-owner, writes the handle, and does NOT fetch/publish ─────────
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    sd, env = _build_root(tmp, LLM_PERF_CELL)
    p = subprocess.run(
        ["bash", str(sd / "run.sh"), str(LLM_PERF_CELL), "testprof", "rdet1", "--detach"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    out = p.stdout + p.stderr
    check("run.sh --detach exits 0 (setup-then-exit)", p.returncode == 0, out.strip()[-300:])
    check("run.sh --detach prints the DETACHED banner", "DETACHED" in out, out.strip()[-200:])
    records = list((tmp / "results" / ".submits").glob("*.json"))
    rec_path = records[0] if len(records) == 1 else tmp / "results" / ".submits" / "MISSING.json"
    check("run.sh --detach writes the local submit record", rec_path.is_file(), out.strip()[-500:])
    if rec_path.is_file():
        rec = json.loads(rec_path.read_text())
        need = {"run_id", "cell", "profile", "namespace", "recipe", "job_name", "artifacts_pvc"}
        check("submit record carries the collect-resolvable keys", need <= set(rec), str(need - set(rec)))
        check(
            "submit record run_id/profile correct",
            str(rec.get("run_id", "")).endswith("-rdet1") and rec.get("profile") == "testprof",
        )
    run_id = json.loads(rec_path.read_text()).get("run_id") if rec_path.is_file() else "MISSING"
    receipt_path = tmp / "results" / run_id / "launch_attestation.json"
    check(
        "run.sh lane captures a launch-time hash before registering its Job", receipt_path.is_file(), out.strip()[-500:]
    )
    if receipt_path.is_file():
        receipt = json.loads(receipt_path.read_text())
        check(
            "launch receipt names this run and is a full recipe hash",
            receipt.get("run_id") == run_id
            and receipt.get("kind") == "recipe_hash_at_launch"
            and len(receipt.get("recipe_hash") or "") == 64,
        )
        applied = (tmp / "applied-manifests.yaml").read_text() if (tmp / "applied-manifests.yaml").exists() else ""
        check(
            "detached index ConfigMap preserves the launch hash for cross-machine collection",
            f'recipe_hash_at_launch: "{receipt["recipe_hash"]}"' in applied,
            applied[-500:],
        )
    klog = (tmp / "kubectl.log").read_text() if (tmp / "kubectl.log").exists() else ""
    check("run.sh --detach applies the index ConfigMap (llmb-submit-<id>)", "apply -f -" in klog, klog[-200:])
    check("run.sh --detach checks the run-owner adoption of the Job", "ownerReferences" in klog)
    check("run.sh --detach does NOT fetch results (no babysitting)", not (tmp / "FETCHED").exists())
    check("run.sh --detach does NOT publish inline", not (tmp / "PUBLISHED").exists())
    check("run.sh --detach does NOT reach wait-ready", not (tmp / "WAITED").exists())


# ── 2. the DEFAULT (attached) path still WAITS — it reaches wait-ready, i.e. does NOT take the detached exit ─
for flag_args, label in (([], "default (no flag)"), (["--attach"], "--attach"), (["--wait"], "--wait")):
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        sd, env = _build_root(tmp, LLM_PERF_CELL)
        p = subprocess.run(
            ["bash", str(sd / "run.sh"), str(LLM_PERF_CELL), "testprof", "ratt1", *flag_args],
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
        # wait_server_ready stub fails fast (exit 1) → run.sh aborts after touching WAITED; the point is it
        # REACHED wait-ready rather than exiting clean at the detached seam.
        check(f"run.sh {label} reaches wait-ready (attached still waits)", (tmp / "WAITED").exists())
        check(
            f"run.sh {label} did NOT write a detached submit record",
            not list((tmp / "results" / ".submits").glob("*.json")),
        )


# ── 3b. the Job is matched by RUN-OWNER OWNERSHIP, not by run.sh's run-id ─────────────────────────────────
# The predicate for matching a Job is "the Job my run-owner adopted" (ownerReference uid), which
# cannot drift the way a name can. Asserted on every unidentifiable reading.
def _drive_detach(
    cell: Path, rid: str, extra_env: dict, args: list[str] | None = None, timeout: int = 180
) -> tuple[subprocess.CompletedProcess, Path, dict]:
    td = tempfile.mkdtemp()
    tmp = Path(td)
    sd, env = _build_root(tmp, cell)
    env.update(extra_env)
    p = subprocess.run(
        ["bash", str(sd / "run.sh"), str(cell), "testprof", rid, "--detach", *(args or ["--skip-stage"])],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    return p, tmp, env


def _tore_down(tmp: Path) -> bool:
    log = tmp / "run_owner.log"
    return log.is_file() and "teardown" in log.read_text()


# llm-perf lane: the bench Job DOES carry run.sh's run-id — unchanged behaviour, still matched, no teardown.
p, tmp, _ = _drive_detach(LLM_PERF_CELL, "rmat1", {}, args=[])
check(
    "llm-perf lane: the run-id-named bench Job still matches", p.returncode == 0, (p.stdout + p.stderr).strip()[-400:]
)
check("llm-perf lane: no teardown on a good apply", not _tore_down(tmp))


# ── 3c. FAIL-SAFE: an unidentifiable Job must NEVER trigger a teardown ────────────────────────────────────
# The asymmetric-cost rule: deleting a healthy multi-hour run to reclaim GPUs is far worse than holding GPUs.
# Only ONE reading justifies teardown — a SUCCESSFUL list showing no candidate Job, after the apply exited.
p, tmp, _ = _drive_detach(LLM_PERF_CELL, "rerr1", {"KUBECTL_JOBS_FAIL": "1", "STUB_EXIT_AFTER_APPLY": "1"}, args=[])
out = p.stdout + p.stderr
check("unreadable cluster: run.sh does NOT tear down", not _tore_down(tmp), out.strip()[-400:])
check("unreadable cluster: run.sh says everything was LEFT RUNNING", "NOTHING WAS TORN DOWN" in out, out.strip()[-400:])
check("unreadable cluster: run.sh still exits non-zero (honest)", p.returncode != 0)

# a Job of this cell is live but NOT (yet) run-owner-adopted → still not "gone" → still no teardown.
p, tmp, _ = _drive_detach(LLM_PERF_CELL, "runad1", {"STUB_UNOWNED_JOB": "1", "STUB_EXIT_AFTER_APPLY": "1"}, args=[])
out = p.stdout + p.stderr
check("unadopted Job: run.sh does NOT tear down", not _tore_down(tmp), out.strip()[-400:])
check("unadopted Job: run.sh leaves everything running", "NOTHING WAS TORN DOWN" in out, out.strip()[-400:])

# the one genuine failure: apply exited, the cluster answers cleanly, and there is NO Job → teardown IS right.
p, tmp, _ = _drive_detach(LLM_PERF_CELL, "rgone1", {"STUB_APPLY_NOTHING": "1", "STUB_EXIT_AFTER_APPLY": "1"}, args=[])
check("genuinely-gone Job: run.sh still fails the run", p.returncode != 0)
check(
    "genuinely-gone Job: run.sh still frees the GPU (teardown)", _tore_down(tmp), (p.stdout + p.stderr).strip()[-400:]
)


# ── 3d. the rc==143 (external SIGTERM) leave-running branch still works ───────────────────────────────────
td = tempfile.mkdtemp()
tmp143 = Path(td)
sd143, env143 = _build_root(tmp143, LLM_PERF_CELL)
env143["STUB_APPLY_NOTHING"] = "1"  # never matches → run.sh stays in the wait loop until we signal it
proc = subprocess.Popen(
    ["bash", str(sd143 / "run.sh"), str(LLM_PERF_CELL), "testprof", "rterm1", "--detach"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    env=env143,
)
for _ in range(60):
    if (tmp143 / "kubectl.log").is_file() and "get jobs" in (tmp143 / "kubectl.log").read_text():
        break
    __import__("time").sleep(0.5)
proc.terminate()  # external SIGTERM — a session/context boundary, NOT a benchmark failure
term_out = proc.communicate(timeout=60)[0]
check(
    "SIGTERM: run.sh takes the 143 leave-running branch", "externally terminated" in term_out, term_out.strip()[-300:]
)
check("SIGTERM: run.sh does NOT tear down", not _tore_down(tmp143), term_out.strip()[-300:])
check("SIGTERM: run.sh exits 143", proc.returncode == 143, str(proc.returncode))


# ── 4. parallel_repro.sh --dry-run: DEFAULT detached prints collect --sweep; --attach is the babysit path ─
def _parallel_dry(cell: Path, extra: list[str]) -> str:
    p = subprocess.run(
        ["bash", str(SCRIPTS / "parallel_repro.sh"), str(cell), "testprof", "2", "--dry-run", *extra],
        capture_output=True,
        text=True,
        timeout=120,
    )
    return p.stdout + p.stderr


dry_default = _parallel_dry(LLM_PERF_CELL, [])
check("parallel_repro --dry-run defaults DETACHED", "DETACHED" in dry_default, dry_default[-200:])
check(
    "parallel_repro --dry-run prints the collect --sweep handle", "collect --sweep" in dry_default, dry_default[-200:]
)
dry_attach = _parallel_dry(LLM_PERF_CELL, ["--attach"])
check(
    "parallel_repro --attach --dry-run is the babysit path",
    "attach" in dry_attach.lower() and "consolidate" in dry_attach.lower(),
    dry_attach[-200:],
)
check(
    "parallel_repro --attach --dry-run does NOT print a collect --sweep handle",
    "collect --sweep" not in dry_attach,
    dry_attach[-200:],
)


# ── 5. collect resolves a run.sh-detached run by its run-id handle ────────────────────────────────────────
import importlib.util  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rs = _load(SCRIPTS / "resilient_status.py", "resilient_status")
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    rs.SUBMITS_DIR = tmp / ".submits"
    rs.SUBMITS_DIR.mkdir()
    rec = {
        "run_id": "rperf9",
        "cell": str(LLM_PERF_CELL),
        "profile": "testprof",
        "namespace": "testns",
        "recipe": "qwen3-0-6b-b200-vllm-agg-kvbm-pareto",
        "job_name": "qwen3-0-6b-b200-vllm-agg-kvbm-pareto-bench-rperf9",
        "artifacts_pvc": "qwen3-0-6b-b200-vllm-agg-kvbm-pareto-artifacts",
        "lane": "bench",
    }
    (rs.SUBMITS_DIR / "rperf9.json").write_text(json.dumps(rec))
    resolved = rs._resolve("rperf9", None)
    check(
        "collect resolves a detached run -> its cell",
        resolved.get("cell") == str(LLM_PERF_CELL),
        str(resolved.get("cell")),
    )
    check("collect resolves the run's artifacts_pvc", "artifacts" in (resolved.get("artifacts_pvc") or ""))
    fetch_src = (SCRIPTS / "fetch_results.sh").read_text()
    collect_src = (SCRIPTS / "resilient_status.py").read_text()
    check(
        "bare fetch resolves the durable submit record instead of recipe.env",
        'results/.submits/${RUN_ID_ARG}.json' in fetch_src and 'ARTIFACTS_PVC_NAME' in fetch_src,
    )
    check(
        "collect passes the run record's exact artifacts PVC to fetch",
        '["--artifacts-pvc", artifacts_pvc]' in collect_src,
    )
    calls = []
    old_run = rs.subprocess.run
    old_read_env = rs.pr._read_env
    try:

        def fake_run(args, *unused_args, **unused_kwargs):
            calls.append(list(args))
            # Stop after fetch so the test exercises the target selection without publishing/deleting.
            rc = 0 if "fetch_results.sh" in " ".join(map(str, args)) else 1
            return subprocess.CompletedProcess(args, rc, stdout="", stderr="")

        rs.subprocess.run = fake_run
        rs.pr._read_env = lambda _path: {"NAMESPACE": "testns"}
        rc = rs.cmd_collect("rperf9", None, [])
    finally:
        rs.subprocess.run = old_run
        rs.pr._read_env = old_read_env
    fetch_calls = [c for c in calls if "fetch_results.sh" in " ".join(map(str, c))]
    check(
        "collect executes fetch with the durable exact PVC (not a profile default)",
        rc == 1
        and len(fetch_calls) == 1
        and fetch_calls[0][-5:] == ["--artifacts-pvc", rec["artifacts_pvc"], str(LLM_PERF_CELL), "testprof", "rperf9"],
        str(fetch_calls),
    )
    check(
        "resolved cell has a recipe.yaml (publishable by collect)",
        (Path(resolved["cell"]) / "recipe.yaml").exists() if resolved.get("cell") else False,
    )

    # a parallel_repro-style sweep record resolves through resilient_status too (collect --sweep).
    rs.SWEEPS_DIR = tmp / ".sweeps"
    rs.SWEEPS_DIR.mkdir()
    sweep = {
        "sweep_id": "rc-x",
        "cell": str(LLM_PERF_CELL),
        "profile": "testprof",
        "namespace": "testns",
        "recipe": "qwen3-0-6b-b200-vllm-agg-kvbm-pareto",
        "mode": "parallel-detached",
        "run_ids": ["rperf9"],
        "legs": [{"run_id": "rperf9", "scratch_cell": str(LLM_PERF_CELL)}],
    }
    (rs.SWEEPS_DIR / "rc-x.json").write_text(json.dumps(sweep))
    sw = rs._resolve_sweep("rc-x", None)
    check("collect --sweep resolves the sweep record -> original cell", sw.get("cell") == str(LLM_PERF_CELL))
    check("collect --sweep resolves a leg's scratch cell", rs._leg_cell("rperf9", sw) == str(LLM_PERF_CELL))


print(f"\nselftest_detach: {'OK' if not fails else 'FAIL ' + ', '.join(fails)}")
sys.exit(1 if fails else 0)
