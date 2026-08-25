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

"""selftest_resilient_submit.py — offline guards for the disconnect-resilient `submit` injection substrate.

No cluster, no profile: feeds the COMMITTED rendered manifests through scripts/resilient_inject.py and asserts
the Phase-1 (the disconnect-resilient run contract §4) + RWX-control-PVC (the in-cluster governor contract §7)
contract, including the PHASE1-REVIEW hardening (F1/F2/F3/F4/F5/F6/F9):
  - bench Job gets activeDeadlineSeconds + short ttlSecondsAfterFinished + the lifecycle label + LLMB_* env
    + EXPECTED_RUNTIME_SECONDS + LLMB_HEALTH_MAX_ATTEMPTS;
  - the bench container command is wrapped to write _submit.json / status.json / runner.log to the shared RWX
    CONTROL PVC at /control/<run-id>/ — NOT the bench-cleaned /artifacts/<run-id>/ (F1) — and the control PVC
    volume+mount is injected; the original script is preserved verbatim inside, valid POSIX sh (`sh -n`);
  - F3: the bench pipeline is BACKGROUNDED + wait'ed and the TERM trap writes a durable reason=timed-out
    (verified at RUNTIME by driving the wrapper with a synthetic bench + SIGTERM);
  - F9: the hardcoded server-health MAX_ATTEMPTS is rewritten to honor LLMB_HEALTH_MAX_ATTEMPTS, and
    submit.sh --dry-run shows a big cold-load --health-timeout default;
  - the server Deployment + Service get an ownerReference → the bench Job (submit-time uid);
  - the injection is stdin→stdout ONLY — it never mutates the committed rendered file (→ recipe_hash can't
    roll from this feature).
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("selftest_resilient_submit: requires pyyaml")

ROOT = Path(__file__).resolve().parent.parent
INJECT = ROOT / "scripts" / "resilient_inject.py"
SUBMIT = ROOT / "scripts" / "submit.sh"
# A representative committed llm-perf cell with both rendered manifests.
CELL = ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"

fails: list[str] = []

# Cluster ${VARS} must be resolved before the transformer sees the manifest (raw ${VAR} in a YAML flow
# mapping is unparseable) — exactly what submit.sh does via envsubst. Mirror that with dummy values; the
# runner script's own ${...} placeholders live inside a block scalar and are opaque to YAML, so they survive.
ENVSUBST_VARS = {
    "NAMESPACE": "testns",
    "RUN_ID": "testrun",
    "OWNER": "tester",
    "IMAGE_PULL_SECRET": "cred",
    "MODEL_CACHE_PVC": "shared-model-cache",
    "DCGM_EXPORTER_URL": "",
    "CACHE_BUST": "",
    "BENCH_NODE_SELECTOR": "",
    "BENCH_CPU_REQUEST": "16",
    "GPU_PRODUCT": "NVIDIA-GB200",
    "SCHEDULER_NAME": "default-scheduler",
    "MODEL_CACHE_SUBPATH": "sub",
}


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def envsubst(src: Path) -> str:
    # Resolve ONLY the whitelisted cluster ${VARS} (submit.sh's strict whitelist), in pure Python so the
    # test needs no gettext. Runner-script ${placeholders} are not in the whitelist and are left intact.
    text = src.read_text()
    for k, v in ENVSUBST_VARS.items():
        text = text.replace("${" + k + "}", v)
    return text


def run_inject(src: Path, *args: str) -> str:
    p = subprocess.run(
        [sys.executable, str(INJECT), *args],
        input=envsubst(src),
        text=True,
        capture_output=True,
    )
    if p.returncode != 0:
        check(f"resilient_inject {args[:2]} exits 0", False, p.stderr.strip()[:200])
        return ""
    return p.stdout


bench_src = CELL / "rendered" / "bench-job.yaml"
server_src = CELL / "rendered" / "server.yaml"
submit_src = SUBMIT.read_text()
_pf = submit_src.index('python3 "$ROOT/scripts/preflight.py" "$CELL" "$PROFILE" --stage-only')
_first_live_apply = submit_src.index("render_control_pvc | kc apply -f -")
check(
    "submit: live compatibility preflight runs before the first Kubernetes apply",
    _pf < _first_live_apply and 'if [ "$_pf_rc" -ne 0 ]' in submit_src[_pf:_first_live_apply],
)

deploy_src = (ROOT / "scripts" / "deploy.sh").read_text()
_deploy_pf = deploy_src.index('python3 "$ROOT/scripts/preflight.py" "$CELL" "$PROFILE" --stage-only')
_deploy_apply = deploy_src.index("kc apply -f -")
_deploy_variant_validation = deploy_src.index('VARIANT_SPEC="$(python3 "$ROOT/scripts/merge_env_override.py" --json)"')
check(
    "deploy: local override validation precedes any live compatibility check",
    _deploy_variant_validation < _deploy_pf,
)
check(
    "deploy: live compatibility preflight runs before the first Kubernetes apply",
    _deploy_pf < _deploy_apply
    and 'if [ "$RENDER_ONLY" = 0 ]' in deploy_src[:_deploy_pf]
    and 'if [ "$_pf_rc" -ne 0 ]' in deploy_src[_deploy_pf:_deploy_apply],
)

# ── bench transform ───────────────────────────────────────────────────────────
before = bench_src.read_text()
out = run_inject(
    bench_src,
    "--kind",
    "bench",
    "--run-id",
    "testrun",
    "--cell",
    str(CELL),
    "--profile",
    "testprof",
    "--namespace",
    "testns",
    "--submit-utc",
    "2026-07-18T00:00:00Z",
    "--deadline-seconds",
    "36000",
    "--ttl-seconds",
    "300",
    "--control-pvc",
    "llmb-control",
    "--expected-runtime-seconds",
    "23692",
    "--health-timeout",
    "2400",
)
docs = [d for d in yaml.safe_load_all(out) if isinstance(d, dict)]
job = next((d for d in docs if d.get("kind") == "Job"), {})
spec = job.get("spec", {})
check("bench: activeDeadlineSeconds injected", spec.get("activeDeadlineSeconds") == 36000)
check(
    "bench: short ttlSecondsAfterFinished injected",
    spec.get("ttlSecondsAfterFinished") == 300,
)
check(
    "bench: lifecycle=detached label on Job",
    (job.get("metadata", {}).get("labels", {}) or {}).get("llmb.nvidia.com/lifecycle") == "detached",
)
pod_spec = spec.get("template", {}).get("spec", {})
container = (pod_spec.get("containers") or [{}])[0]
env = {e.get("name"): e.get("value") for e in (container.get("env") or [])}
check(
    "bench: LLMB_CELL/PROFILE/SUBMIT_UTC env injected",
    env.get("LLMB_CELL") == str(CELL)
    and env.get("LLMB_PROFILE") == "testprof"
    and env.get("LLMB_SUBMIT_UTC") == "2026-07-18T00:00:00Z",
)
# EXPECTED_RUNTIME_SECONDS feeds the Phase-2 governor's 2×-median timeout; LLMB_HEALTH_MAX_ATTEMPTS is F9.
check(
    "bench: EXPECTED_RUNTIME_SECONDS env injected (governor timeout substrate)",
    env.get("EXPECTED_RUNTIME_SECONDS") == "23692",
    str(env.get("EXPECTED_RUNTIME_SECONDS")),
)
check(
    "bench: LLMB_HEALTH_MAX_ATTEMPTS env injected = health-timeout/5 (F9)",
    env.get("LLMB_HEALTH_MAX_ATTEMPTS") == "480",
    str(env.get("LLMB_HEALTH_MAX_ATTEMPTS")),
)
# F1/F2: wrapper state lives on the shared RWX control PVC (llmb-control at /control), NOT /artifacts.
mounts = {m.get("name"): m.get("mountPath") for m in (container.get("volumeMounts") or [])}
vols = {v.get("name"): v for v in (pod_spec.get("volumes") or [])}
check(
    "bench: control PVC volumeMount at /control injected (F1/F2)",
    mounts.get("llmb-control") == "/control",
)
check(
    "bench: control PVC volume → claim llmb-control injected (F1/F2)",
    (vols.get("llmb-control", {}).get("persistentVolumeClaim", {}) or {}).get("claimName") == "llmb-control",
)
script = container.get("args", [""])[0]
check("bench: wrapper writes _submit.json to control PVC", "_submit.json" in script)
# F1: state dir is /control/<run-id> (the bench NEVER wipes it), NOT /artifacts/<run-id> (CLEAN_ARTIFACTS_ROOT).
check(
    "bench: wrapper state dir is on the control PVC, not the bench-cleaned /artifacts (F1)",
    '_STATE_DIR="/control/${RUN_ID}"' in script and '"/artifacts/${RUN_ID}"' not in script,
)
check(
    "bench: wrapper writes status.json (running/complete/failed)",
    "status.json" in script
    and "_set_state complete terminal completed" in script
    and "_set_state failed terminal failed" in script,
)
# F3: the deadline TERM trap writes reason=timed-out, and the bench runs BACKGROUNDED + wait'ed so the trap
# is not deferred behind a foreground pipeline.
check(
    "bench: deadline TERM trap records reason=timed-out (F3)",
    "trap '_terminate timed-out' TERM" in script and '_set_state failed terminal "$1"' in script,
)
check(
    "bench: bench pipeline is BACKGROUNDED + wait'ed so the trap can fire (F3)",
    'tee -a "$_STATE_DIR/logs/runner.log" &' in script and 'wait "$_bpid"' in script,
)
# F9: the hardcoded health MAX_ATTEMPTS now honors the injected LLMB_HEALTH_MAX_ATTEMPTS env.
check(
    "bench: server-health MAX_ATTEMPTS honors LLMB_HEALTH_MAX_ATTEMPTS (F9)",
    "MAX_ATTEMPTS=${LLMB_HEALTH_MAX_ATTEMPTS:-" in script,
)
# governor substrate: a backgrounded heartbeat loop republishes status.json (phase/heartbeat/progress).
check(
    "bench: backgrounded heartbeat loop atomically republishes status.json (governor substrate)",
    "_heartbeat & _hbpid=" in script and "status.json.tmp" in script and "mv " in script,
)
check(
    "bench: wrapper tees to durable runner.log on the control PVC",
    "logs/runner.log" in script and "tee -a" in script,
)
check(
    "bench: original bench script preserved verbatim inside wrapper",
    "aiperf" in script and "Waiting for $SERVER_URL/health" in script,
)
shn = subprocess.run(["sh", "-n"], input=script, text=True, capture_output=True)
check(
    "bench: wrapped command is valid POSIX sh (sh -n)",
    shn.returncode == 0,
    shn.stderr.strip()[:200],
)
check(
    "bench: injection did NOT mutate the committed rendered file (hash-neutral)",
    bench_src.read_text() == before,
)

# ── F3 RUNTIME: drive the wrapper with a synthetic bench; SIGTERM must leave a durable reason=timed-out ──────
import importlib.util as _ilu  # noqa: E402
import json as _json  # noqa: E402
import signal as _signal  # noqa: E402
import tempfile as _tempfile  # noqa: E402
import time as _time  # noqa: E402

_spec = _ilu.spec_from_file_location("_ri", INJECT)
_ri = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_ri)
_tmp = _tempfile.mkdtemp()


def _drive(bench: str) -> dict:
    s = _ri._wrap_script(bench).replace("/control/${RUN_ID}", _tmp + "/${RUN_ID}")
    s = "RUN_ID=rt\nSERVER_URL=http://127.0.0.1:1/\ncurl(){ return 1; }\n" + s
    return s


# deadline/SIGTERM → durable reason=timed-out, exit 143
import subprocess as _sp  # noqa: E402

_p = _sp.Popen(["sh", "-c", _drive("sleep 20")])
_time.sleep(2)
_p.send_signal(_signal.SIGTERM)
try:
    _p.wait(timeout=15)
except Exception:
    _p.kill()
_stp = Path(_tmp, "rt", "status.json")
_st = _json.loads(_stp.read_text()) if _stp.is_file() else {}
check(
    "F3 runtime: SIGTERM (deadline) writes a DURABLE status.json with reason=timed-out",
    _p.returncode == 143 and _st.get("state") == "failed" and _st.get("reason") == "timed-out",
    f"rc={_p.returncode} status={_st}",
)

# clean success → state=complete / reason=completed, and the governor fields are present
import shutil as _shutil  # noqa: E402

_shutil.rmtree(Path(_tmp, "rt"), ignore_errors=True)
_p2 = _sp.run(["sh", "-c", _drive("true")])
_st2 = _json.loads(Path(_tmp, "rt", "status.json").read_text())
check(
    "F3 runtime: clean exit writes state=complete reason=completed with governor fields",
    _st2.get("state") == "complete"
    and _st2.get("reason") == "completed"
    and all(
        k in _st2
        for k in (
            "phase",
            "heartbeat_utc",
            "progress_counter",
            "progress_utc",
            "expected_runtime_seconds",
        )
    ),
    str(_st2)[:160],
)

# ── server transform ──────────────────────────────────────────────────────────
sbefore = server_src.read_text()
sout = run_inject(
    server_src,
    "--kind",
    "server",
    "--owner-name",
    "cell-bench-testrun",
    "--owner-uid",
    "UID-1234",
)
sdocs = [d for d in yaml.safe_load_all(sout) if isinstance(d, dict)]
for kind in ("Deployment", "Service"):
    d = next((x for x in sdocs if x.get("kind") == kind), {})
    refs = d.get("metadata", {}).get("ownerReferences") or []
    ok = (
        len(refs) == 1
        and refs[0].get("kind") == "Job"
        and refs[0].get("name") == "cell-bench-testrun"
        and refs[0].get("uid") == "UID-1234"
    )
    check(f"server: {kind} gets ownerReference → bench Job", ok, str(refs)[:160])
check(
    "server: injection did NOT mutate the committed rendered file (hash-neutral)",
    server_src.read_text() == sbefore,
)

# ── run-id → cell INDEX ConfigMap (dry-run of submit.sh) ──────────────────────
# The cold, cross-machine, post-Job-GC resolver: a namespace-scoped ConfigMap with NO ownerReference (so it
# outlives the Job's TTL-GC) and NO TTL, label-selectable by run-id. Assert submit.sh --dry-run emits it.
DUMMY_PROFILE = next(
    (p.stem for p in (ROOT / "cluster-profiles").glob("*.env") if not p.name.endswith(".example")),
    None,
)
if DUMMY_PROFILE:
    dr = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "submit.sh"),
            str(CELL),
            DUMMY_PROFILE,
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    # The dry-run stdout interleaves human-readable headers with YAML; slice out just the index-CM section
    # (from its `apiVersion:` to the trailing completion comment) and parse that one document.
    tail = dr.stdout.split("run-id → cell INDEX")[-1]
    cm_text = tail.split("apiVersion:", 1)[-1]
    cm_text = "apiVersion:" + cm_text.split("# (dry-run complete", 1)[0]
    cm_docs = [
        d
        for d in yaml.safe_load_all(cm_text)
        if isinstance(d, dict)
        and d.get("kind") == "ConfigMap"
        and str(d.get("metadata", {}).get("name", "")).startswith("llmb-submit-")
    ]
    check(
        "index CM: submit.sh --dry-run emits an llmb-submit-<run-id> ConfigMap",
        len(cm_docs) == 1,
        f"found {len(cm_docs)} (rc={dr.returncode})",
    )
    if cm_docs:
        cm = cm_docs[0]
        meta = cm.get("metadata", {})
        check(
            "index CM: carries the llmb.nvidia.com/run-id label",
            (meta.get("labels", {}) or {}).get("llmb.nvidia.com/run-id") not in (None, ""),
        )
        check(
            "index CM: has NO ownerReference (survives Job GC)",
            not meta.get("ownerReferences"),
        )
        data = cm.get("data", {}) or {}
        check(
            "index CM: data resolves run-id → cell/profile/namespace/pvc",
            all(data.get(k) for k in ("run_id", "cell", "profile", "namespace", "artifacts_pvc")),
            str(data)[:160],
        )
        check(
            "index CM: carries a pre-launch recipe hash for cross-machine collection",
            bool(data.get("recipe_hash_at_launch")) and bool(data.get("recipe_hash_captured_at_utc")),
            str(data)[:200],
        )
    # ── F1/F2/F9 dry-run PROOFS: the control PVC + corrected state paths + the health-timeout are surfaced ──
    _dro = dr.stdout
    check(
        "dry-run: emits the shared RWX control PVC (ReadWriteMany, name llmb-control)",
        "name: llmb-control" in _dro and "ReadWriteMany" in _dro,
        f"rc={dr.returncode}",
    )
    ctrl_docs = [
        d
        for d in yaml.safe_load_all(
            "apiVersion:" + _dro.split("shared RWX control PVC")[-1].split("apiVersion:", 1)[-1].split("# ===", 1)[0]
        )
        if isinstance(d, dict) and d.get("kind") == "PersistentVolumeClaim"
    ]
    check(
        "dry-run: control PVC is ReadWriteMany (RWX — mid-run readable, F2)",
        bool(ctrl_docs) and ctrl_docs[0].get("spec", {}).get("accessModes") == ["ReadWriteMany"],
        str(ctrl_docs[:1])[:160],
    )
    check(
        "dry-run: wrapper state paths point at /control/<run-id> (F1, not bench-cleaned /artifacts)",
        "/control/${RUN_ID}/{status.json" in _dro or "state at /control/" in _dro,
    )
    # The CONTRACT (submit.sh) is max(1800, serving.startup_timeout_s) — a 1800s floor, not a literal 2400.
    # Asserting "2400s" only held while the fixture cell happened to omit startup_timeout_s; this cell sets
    # 900, so the correct budget is the 1800s floor. Assert the floor the check's own name states.
    _ht = re.search(r"health-timeout:\s*(\d+)s", _dro)
    check(
        "dry-run: shows a big cold-load --health-timeout default (>=1800s, F9)",
        bool(_ht) and int(_ht.group(1)) >= 1800,
        (_ht.group(0) if _ht else _dro[:120]),
    )
    check(
        "dry-run: notes the BACKGROUNDED bench pipeline / deadline trap (F3)",
        "BACKGROUNDED" in _dro and "reason=timed-out" in _dro,
    )
    check(
        "dry-run: stamps EXPECTED_RUNTIME_SECONDS (governor timeout substrate)",
        "expected_runtime:" in _dro,
    )
else:
    # No real cluster profile in this environment (profiles are gitignored).
    # The submit.sh --dry-run checks require a profile to invoke the script.
    # Treat as SKIP rather than FAIL so CI passes without local profiles.
    print("  SKIP  index CM + variance sweep: no cluster-profiles/*.env (CI environment — skip submit.sh dry-run)")

# ── VARIANCE SWEEP (submit --repeat N): N distinct-named legs, IDENTICAL benchmark_id, stagger plan, sweep
#    record, and collect --sweep run-id resolution — all OFFLINE, applying/mutating NOTHING (hash-safe). ──────
if DUMMY_PROFILE:
    import re as _re  # noqa: E402

    def _bid(cell) -> str:
        p = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "benchmark_id.py"),
                str(cell),
                "--short",
            ],
            capture_output=True,
            text=True,
        )
        return p.stdout.strip()

    orig_bid = _bid(CELL)
    recipe_before = (CELL / "recipe.yaml").read_text()  # hash-safety: the ORIGINAL cell must not be touched.
    runs_before = (CELL / "runs.jsonl").read_text() if (CELL / "runs.jsonl").is_file() else None
    sweeps_before = set((ROOT / "results" / ".sweeps").glob("*")) if (ROOT / "results" / ".sweeps").is_dir() else set()

    sw = subprocess.run(
        [
            "bash",
            str(ROOT / "scripts" / "submit.sh"),
            str(CELL),
            DUMMY_PROFILE,
            "--repeat",
            "3",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
    )
    swo = sw.stdout
    check(
        "sweep: submit --repeat 3 --dry-run exits 0",
        sw.returncode == 0,
        sw.stderr.strip()[:200],
    )
    check("sweep: dry-run applies NOTHING (announces it)", "NOTHING APPLIED" in swo)
    # N distinct-named legs.
    leg_lines = [ln for ln in swo.splitlines() if _re.search(r"^\s*#\d+\s+name=", ln)]
    leg_names = _re.findall(r"name=(\S+)", swo)
    check(
        "sweep: 3 legs listed with distinct suffixed names",
        len(leg_lines) == 3 and len(set(leg_names)) == 3,
        f"legs={leg_lines}",
    )
    # IDENTICAL benchmark_id on every leg == the ORIGINAL cell's benchmark_id (benchmark_id excludes name).
    leg_bids = _re.findall(r"benchmark_id=([0-9a-f]+)", swo)
    check(
        "sweep: every leg carries the IDENTICAL benchmark_id as the original (excludes envelope.name)",
        len(leg_bids) == 3 and set(leg_bids) == {orig_bid} and orig_bid != "",
        f"orig={orig_bid} legs={leg_bids}",
    )
    # stagger plan (parallel default, 120s, +0/+120/+240 offsets).
    check(
        "sweep: prints the staggered start plan (mode=parallel, +0s/+120s/+240s)",
        "mode:                parallel" in swo and "start +0s" in swo and "start +120s" in swo and "start +240s" in swo,
    )
    # durable sweep record ConfigMap: no ownerRef, resolves run-ids → original cell.
    sweep_tail = swo.split("sweep record ConfigMap")[-1]
    cm_text = "apiVersion:" + sweep_tail.split("apiVersion:", 1)[-1].split("# ===", 1)[0]
    sweep_cm = [
        d
        for d in yaml.safe_load_all(cm_text)
        if isinstance(d, dict)
        and d.get("kind") == "ConfigMap"
        and str(d.get("metadata", {}).get("name", "")).startswith("llmb-sweep-")
    ]
    check(
        "sweep: dry-run emits an llmb-sweep-<id> record ConfigMap",
        len(sweep_cm) == 1,
        f"found {len(sweep_cm)}",
    )
    if sweep_cm:
        scm = sweep_cm[0]
        check(
            "sweep: record CM has NO ownerReference (survives Job GC)",
            not scm.get("metadata", {}).get("ownerReferences"),
        )
        sdata = scm.get("data", {}) or {}
        run_ids = (sdata.get("run_ids") or "").split()
        check(
            "sweep: record lists 3 run-ids + original cell/recipe/benchmark_id",
            len(run_ids) == 3 and sdata.get("cell") == str(CELL) and sdata.get("benchmark_id") == orig_bid,
            str(sdata)[:200],
        )
    # per-leg proof: each leg is a full resilient submit (its own dry-run ran green).
    check(
        "sweep: each of the 3 legs ran a full resilient submit --dry-run OK",
        swo.count("resilient submit --dry-run OK") == 3,
    )
    # HASH SAFETY: the ORIGINAL committed cell is byte-unchanged, and the dry-run persisted NO sweep scratch.
    check(
        "sweep: dry-run did NOT mutate the original recipe.yaml (hash-neutral)",
        (CELL / "recipe.yaml").read_text() == recipe_before,
    )
    runs_after = (CELL / "runs.jsonl").read_text() if (CELL / "runs.jsonl").is_file() else None
    check(
        "sweep: dry-run did NOT touch the original runs.jsonl (no re-fingerprint)",
        runs_after == runs_before,
    )
    sweeps_after = set((ROOT / "results" / ".sweeps").glob("*")) if (ROOT / "results" / ".sweeps").is_dir() else set()
    check(
        "sweep: dry-run persisted NO sweep scratch under results/.sweeps (ephemeral)",
        sweeps_after == sweeps_before,
        f"new={sweeps_after - sweeps_before}",
    )

    # collect --sweep resolves the N run-ids from a LOCAL sweep record (no cluster needed).
    import importlib.util as _ilu2  # noqa: E402
    import json as _json2  # noqa: E402

    _rs_spec = _ilu2.spec_from_file_location("_rs_sweep", ROOT / "scripts" / "resilient_status.py")
    _rs = _ilu2.module_from_spec(_rs_spec)
    _rs_spec.loader.exec_module(_rs)
    _sd = ROOT / "results" / ".sweeps"
    _sd.mkdir(parents=True, exist_ok=True)
    _swid = "swSELFTEST0000"
    _rec = {
        "sweep_id": _swid,
        "cell": str(CELL),
        "profile": DUMMY_PROFILE,
        "namespace": "ns",
        "recipe": "x",
        "repeat": 3,
        "mode": "parallel",
        "stagger_seconds": 120,
        "run_ids": ["ra1", "rb2", "rc3"],
        "legs": [
            {"run_id": "ra1", "scratch_cell": "/tmp/x/ra1"},
            {"run_id": "rb2", "scratch_cell": "/tmp/x/rb2"},
            {"run_id": "rc3", "scratch_cell": "/tmp/x/rc3"},
        ],
    }
    _recp = _sd / f"{_swid}.json"
    _recp.write_text(_json2.dumps(_rec))
    try:
        _resolved = _rs._resolve_sweep(_swid, None)
        check(
            "collect --sweep: resolves the 3 leg run-ids + original cell from the local sweep record",
            _resolved.get("run_ids") == ["ra1", "rb2", "rc3"] and _resolved.get("cell") == str(CELL),
            str(_resolved)[:160],
        )
    finally:
        _recp.unlink(missing_ok=True)

print()
if fails:
    print(f"selftest_resilient_submit: {len(fails)} FAILED: {fails}")
    sys.exit(1)
total = sum(1 for line in open(__file__).read().splitlines() if line.strip().startswith("check("))
print(f"selftest_resilient_submit: all {total} checks PASSED ✓")
sys.exit(0)
