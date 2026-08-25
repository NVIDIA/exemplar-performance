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

"""resilient_status.py — status / logs / collect for a DETACHED (`submit`) run, keyed by run-id.

The disconnect-durable reconnect side of Phase 1 (the disconnect-resilient run contract §2Z / §4). A `submit`
leaves its live control-plane state on the shared per-namespace RWX CONTROL PVC (`llmb-control`):
/control/<run-id>/status.json, .../governor.json, .../logs/runner.log, .../_submit.json — plus a LOCAL
pointer (results/.submits/<run-id>.json). These verbs read that state so it survives Job TTL-GC and a cold
`tsh` re-login:

  resilient_status.py status  <run-id> [--cluster <profile>]   control status.json + governor.json + Job .status
  resilient_status.py logs    <run-id> [--cluster <profile>] [--tail N]   control runner.log
  resilient_status.py collect <run-id> [--cluster <profile>] [publish-args...]   fetch_results.sh + publish.py
  resilient_status.py collect --sweep <sweep-id> [--cluster <profile>] [publish-args...]   collect ALL legs of a
      variance sweep (submit --repeat N), consolidate their values into the ORIGINAL cell's runs.jsonl
      (repro_consolidate.py, stamped with the original recipe_hash), and print compare.py --repro's spread band.

Run-id → cell/namespace resolution order:
  1. the LOCAL submit record  results/.submits/<run-id>.json  (written by submit.sh; same-laptop reconnect);
  2. else, with --cluster <profile>, the persistent index ConfigMap / live Job labelled
     llmb.nvidia.com/run-id=<run-id> → its namespace + llmb.nvidia.com/cell label.
The CONTROL PVC is ReadWriteMany, so the mounter Pod mounts it read-only CONCURRENTLY with the running bench
(no RWO Multi-Attach / wait-for-release dance — PHASE1-REVIEW F1/F2). Bulk results stay on the per-cell RWO
artifacts PVC and are fetched post-run by `collect` (unchanged). The mounter uses the RUN's namespace (F6).
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))
import profile_resolver as pr  # noqa: E402

SUBMITS_DIR = ROOT / "results" / ".submits"
SWEEPS_DIR = ROOT / "results" / ".sweeps"  # variance-sweep records (submit --repeat N) — run-ids + original cell
CONTROL_PVC = "llmb-control"  # shared per-namespace RWX control PVC (the in-cluster governor contract §1)
CONTROL_MOUNT = "/control"


def _local_record(run_id: str) -> dict | None:
    p = SUBMITS_DIR / f"{run_id}.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _local_sweep(sweep_id: str) -> dict | None:
    p = SWEEPS_DIR / f"{sweep_id}.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:
            return None
    return None


def _kbase(prof: dict, ns: str | None = None) -> list[str]:
    # F6: the namespace must be the RUN's namespace (resolved from the record/index CM), not the profile's
    # default — a run submitted into a non-default namespace would otherwise get its peek Pod / Job get in the
    # wrong namespace. Callers pass rec["namespace"]; fall back to the profile NAMESPACE only if unset.
    cmd = ["kubectl"]
    ctx = (prof.get("KUBE_CONTEXT") or "").strip()
    if ctx:
        cmd += ["--context", ctx]
    ns = (ns or prof.get("NAMESPACE") or "").strip()
    if ns:
        cmd += ["-n", ns]
    return cmd


def _resolve(run_id: str, cluster: str | None) -> dict:
    """Return {run_id, cell, profile, namespace, recipe, artifacts_pvc, job_name}. Resolution order:
      (a) the LOCAL submit record (same-laptop reconnect);
      (b) with --cluster, the persistent cluster-side INDEX ConfigMap `llmb-submit-<run-id>` (labelled
          llmb.nvidia.com/run-id) — the COLD, cross-machine, post-Job-GC path (the CM has no ownerRef/TTL);
      (c) with --cluster, the live Job label (works only until the Job TTL-GCs).
    Exit with guidance if none locate the run-id."""
    rec = _local_record(run_id)
    if rec:
        rec.setdefault("artifacts_pvc", f"{rec.get('recipe', '')}-artifacts")
        if cluster:
            rec["profile"] = cluster  # explicit override wins (e.g. reconnecting from a different profile)
        return rec
    if not cluster:
        sys.exit(
            f"resilient_status: no local submit record for run-id '{run_id}' at {SUBMITS_DIR}.\n"
            f"  Reconnecting from a different machine? pass --cluster <profile> so I can resolve it from the\n"
            f"  cluster-side index ConfigMap / live Job:\n"
            f"    llmb-k8s status {run_id} --cluster <profile>"
        )
    prof = pr._read_env(pr.profile_env_path(cluster))

    # (b) cluster-side index ConfigMap — survives Job GC, so this is the cold cross-machine resolver.
    cp = subprocess.run(
        [
            *_kbase(prof),
            "get",
            "configmap",
            "-l",
            f"llmb.nvidia.com/run-id={run_id}",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    cms = json.loads(cp.stdout).get("items", []) if cp.returncode == 0 and cp.stdout.strip() else []
    if cms:
        d = cms[0].get("data", {}) or {}
        ns = d.get("namespace") or cms[0].get("metadata", {}).get("namespace", "") or (prof.get("NAMESPACE") or "")
        recipe = d.get("recipe", "")
        return {
            "run_id": run_id,
            "cell": d.get("cell", ""),
            "profile": cluster,
            "namespace": ns,
            "recipe": recipe,
            "artifacts_pvc": d.get("artifacts_pvc") or f"{recipe}-artifacts",
            "job_name": d.get("job_name", ""),
        }

    # (c) live Job label — last resort (gone after TTL-GC).
    p = subprocess.run(
        [
            *_kbase(prof),
            "get",
            "jobs",
            "-l",
            f"llmb.nvidia.com/run-id={run_id}",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    items = json.loads(p.stdout).get("items", []) if p.returncode == 0 and p.stdout.strip() else []
    if not items:
        sys.exit(
            f"resilient_status: run-id '{run_id}' not found on '{cluster}' — no index ConfigMap "
            f"(llmb-submit-{run_id}) and no live Job. If you have the cell dir, fetch directly:\n"
            f"    scripts/fetch_results.sh <cell> {cluster} {run_id}"
        )
    meta = items[0].get("metadata", {})
    labels = meta.get("labels", {}) or {}
    recipe = labels.get("llmb.nvidia.com/cell") or labels.get("llmb.nvidia.com/recipe") or ""
    return {
        "run_id": run_id,
        "cell": "",
        "profile": cluster,
        "namespace": meta.get("namespace", "") or (prof.get("NAMESPACE") or ""),
        "recipe": recipe,
        "artifacts_pvc": f"{recipe}-artifacts",
        "job_name": meta.get("name", ""),
    }


def _mounter_read(rec: dict, remote_paths, prof: dict) -> list[tuple[int, str]]:
    """Spawn ONE transient read-only mounter Pod on the shared RWX CONTROL PVC, cat each of `remote_paths`
    inside it (so status.json + governor.json need only a single peek Pod — F8: no same-name pod races), and
    delete it. Returns a list of (rc, contents) aligned with `remote_paths`. Because the control PVC is
    ReadWriteMany, the mount SUCCEEDS concurrently with the running bench (F1/F2 — no RWO Multi-Attach,
    no wait-for-release). Uses the RUN's namespace (F6). Accepts a single path (str) for convenience.
    """
    if isinstance(remote_paths, str):
        remote_paths = [remote_paths]
    ns = (rec.get("namespace") or "").strip()
    pvc = rec.get("control_pvc") or CONTROL_PVC
    run_id = rec["run_id"]
    pod = f"{rec.get('recipe', 'llmb')}-peek-{run_id}"[:63].rstrip("-")  # F8: no trailing '-' (DNS-1123)
    base = _kbase(prof, ns)
    manifest = f"""apiVersion: v1
kind: Pod
metadata:
  name: {pod}
  labels: {{ app.kubernetes.io/managed-by: llmb-recipe, llmb.nvidia.com/run-id: {run_id}, app.kubernetes.io/component: peek }}
spec:
  restartPolicy: Never
  containers:
  - name: mounter
    image: {prof.get('UTIL_IMAGE') or 'alpine:3.20'}
    command: ["sh", "-c", "sleep 600"]
    volumeMounts: [{{ name: control, mountPath: {CONTROL_MOUNT}, readOnly: true }}]
  volumes:
  - name: control
    persistentVolumeClaim: {{ claimName: {pvc}, readOnly: true }}
"""
    try:
        subprocess.run(
            [*base, "apply", "--validate=false", "-f", "-"],
            input=manifest,
            text=True,
            capture_output=True,
        )
        w = subprocess.run(
            [*base, "wait", f"pod/{pod}", "--for=condition=Ready", "--timeout=120s"],
            capture_output=True,
            text=True,
        )
        if w.returncode != 0:
            msg = f"(mounter pod {pod} not Ready: {w.stderr.strip()})"
            return [(1, msg) for _ in remote_paths]
        out = []
        for path in remote_paths:
            r = subprocess.run(
                [*base, "exec", pod, "-c", "mounter", "--", "cat", path],
                capture_output=True,
                text=True,
            )
            out.append((r.returncode, (r.stdout if r.returncode == 0 else r.stderr)))
        return out
    finally:
        subprocess.run(
            [*base, "delete", "pod", pod, "--wait=false", "--ignore-not-found"],
            capture_output=True,
            text=True,
        )


def cmd_status(run_id: str, cluster: str | None) -> int:
    rec = _resolve(run_id, cluster)
    prof = pr._read_env(pr.profile_env_path(rec["profile"])) if rec.get("profile") else {}
    print(f"── detached run {run_id} ─────────────────────────────")
    print(f"  cell        {rec.get('cell') or rec.get('recipe') or '?'}")
    print(f"  profile     {rec.get('profile') or '?'}   namespace {rec.get('namespace') or '?'}")
    job = rec.get("job_name") or ""
    jstate = "gone (TTL-GC'd or not applied)"
    ns = rec.get("namespace")  # F6: read the Job in the RUN's namespace, not the profile default
    if job and prof:
        jp = subprocess.run(
            [
                *_kbase(prof, ns),
                "get",
                "job",
                job,
                "-o",
                "jsonpath={.status.active}|{.status.succeeded}|{.status.failed}|" "{.status.conditions[-1].reason}",
            ],
            capture_output=True,
            text=True,
        )
        if jp.returncode == 0 and jp.stdout.strip():
            a, s, f, reason = (jp.stdout.split("|") + ["", "", "", ""])[:4]
            jstate = f"active={a or 0} succeeded={s or 0} failed={f or 0}" + (f" reason={reason}" if reason else "")
    print(f"  Job         {job or '?'}  →  {jstate}")
    if prof:
        # RWX control PVC → this read succeeds mid-run (F1/F2). status.json is wrapper-owned; governor.json (if
        # present) is the AUTHORITATIVE halt reason written by the Phase-2 governor. One peek Pod reads both.
        (rc, body), (grc, gbody) = _mounter_read(
            rec,
            [
                f"{CONTROL_MOUNT}/{run_id}/status.json",
                f"{CONTROL_MOUNT}/{run_id}/governor.json",
            ],
            prof,
        )
        if rc == 0:
            try:
                st = json.loads(body)
                print(
                    f"  status.json state={st.get('state')} phase={st.get('phase')} "
                    f"reason={st.get('reason')} progress={st.get('progress_counter')} "
                    f"heartbeat={st.get('heartbeat_utc')} updated={st.get('updated_utc')}"
                )
                # The hang short-circuit's two halves, in the operator's own words. A stall is only ever
                # actioned when BOTH "tokens not advancing" and "work still outstanding" hold, so both must
                # be visible here; -1 means UNMEASURABLE (never read as "no work") and disarms the halt.
                _inf, _q = st.get("inflight_requests", -1), st.get("queued_requests", -1)

                def _fmt(v):
                    return "unknown" if v is None or v == -1 else v

                print(
                    f"  work        inflight={_fmt(_inf)} queued={_fmt(_q)} "
                    f"tokens_last_advanced={st.get('progress_utc')} "
                    f"work_outstanding_since={st.get('idle_utc')} "
                    f"metrics={st.get('progress_note', 'unknown')}"
                )
            except Exception:
                # Invalid JSON here means the governor's `jq` reads also fail → the run is UNSUPERVISED.
                print(
                    f"  status.json INVALID JSON → this run is NOT under governor stall/timeout "
                    f"supervision: {body.strip()[:200]}"
                )
        else:
            # No status.json yet but Job exists → the pod hasn't started writing → queued.
            print(f"  status.json not readable → likely queued (pod not yet running). {body.strip()[:120]}")
        if grc == 0 and gbody.strip():
            try:
                g = json.loads(gbody)
                _act = g.get("action")
                _tag = (
                    "(AUTHORITATIVE halt reason)"
                    if _act == "halt"
                    else ("(EARLY WARNING — nothing was deleted)" if _act == "warn" else "")
                )
                print(
                    f"  governor    action={_act} reason={g.get('reason')} "
                    f"detail={g.get('detail')} utc={g.get('utc')}   {_tag}"
                )
            except Exception:
                print(f"  governor.json (unparsed): {gbody.strip()[:200]}")
    else:
        print("  (no reachable profile — pass --cluster <profile> to read control-PVC status.json + Job .status)")
    return 0


def cmd_logs(run_id: str, cluster: str | None, tail: int | None) -> int:
    rec = _resolve(run_id, cluster)
    if not rec.get("profile"):
        sys.exit("resilient_status logs: need a reachable profile — pass --cluster <profile>")
    prof = pr._read_env(pr.profile_env_path(rec["profile"]))
    log_path = f"{CONTROL_MOUNT}/{run_id}/logs/runner.log"
    ((rc, body),) = _mounter_read(rec, log_path, prof)  # RWX → readable mid-run (F1/F2)
    if rc != 0:
        print(f"logs: could not read {log_path} — {body.strip()[:200]}")
        return 1
    lines = body.splitlines()
    if tail:
        lines = lines[-tail:]
    print("\n".join(lines))
    return 0


def cmd_collect(run_id: str, cluster: str | None, extra: list[str]) -> int:
    rec = _resolve(run_id, cluster)
    cell = rec.get("cell") or ""
    profile = rec.get("profile") or cluster or ""
    if not cell:
        sys.exit(
            "resilient_status collect: could not resolve the cell dir from the run-id "
            f"(no local record at {SUBMITS_DIR}/{run_id}.json). Run collect from the machine that submitted, "
            "or fetch directly: scripts/fetch_results.sh <cell> <profile> " + run_id
        )
    if not (Path(cell) / "recipe.yaml").exists():
        sys.exit(f"resilient_status collect: cell dir '{cell}' has no recipe.yaml (moved?)")
    # 1) idempotent, resumable fetch off the PVC.
    artifacts_pvc = (rec.get("artifacts_pvc") or "").strip()
    fetch_cmd = ["bash", str(ROOT / "scripts" / "fetch_results.sh")]
    if artifacts_pvc:
        fetch_cmd += ["--artifacts-pvc", artifacts_pvc]
    fetch_cmd += [cell, profile, run_id]
    fr = subprocess.run(fetch_cmd)
    if fr.returncode != 0:
        print(
            "collect: fetch_results.sh did not complete — re-run to resume.",
            file=sys.stderr,
        )
        return fr.returncode
    # 2) local publish (human-gated; publish.py finds the run under RESULTS_LOCAL_DIR/<run-id>).
    run_dir = ROOT / "results" / run_id
    pub = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "publish.py"),
            cell,
            str(run_dir),
            f"--cluster={profile}",
            *extra,
        ]
    )
    # 3) on success, delete the cluster-side index ConfigMap so they don't accumulate (idempotent).
    if pub.returncode == 0 and profile:
        prof = pr._read_env(pr.profile_env_path(profile))
        subprocess.run(
            [
                *_kbase(prof),
                "delete",
                "configmap",
                f"llmb-submit-{run_id}",
                "--ignore-not-found",
                "--wait=false",
            ],
            capture_output=True,
            text=True,
        )
    return pub.returncode


def _resolve_sweep(sweep_id: str, cluster: str | None) -> dict:
    """Resolve a variance sweep (submit --repeat N) → its record {sweep_id, cell, profile, namespace, recipe,
    run_ids, legs}. Order: (a) LOCAL record results/.sweeps/<id>.json (same-laptop reconnect); (b) with
    --cluster, the durable cluster-side sweep ConfigMap `llmb-sweep-<id>` (no ownerRef/TTL → survives Job GC).
    """
    rec = _local_sweep(sweep_id)
    if rec:
        if cluster:
            rec["profile"] = cluster
        return rec
    if not cluster:
        sys.exit(
            f"resilient_status: no local sweep record for '{sweep_id}' at {SWEEPS_DIR}.\n"
            f"  Reconnecting from another machine? pass --cluster <profile> to resolve it from the cluster-side\n"
            f"  sweep ConfigMap: llmb-k8s collect --sweep {sweep_id} --cluster <profile>"
        )
    prof = pr._read_env(pr.profile_env_path(cluster))
    cp = subprocess.run(
        [
            *_kbase(prof),
            "get",
            "configmap",
            "-l",
            f"llmb.nvidia.com/sweep-id={sweep_id}",
            "-o",
            "json",
        ],
        capture_output=True,
        text=True,
    )
    items = json.loads(cp.stdout).get("items", []) if cp.returncode == 0 and cp.stdout.strip() else []
    if not items:
        sys.exit(
            f"resilient_status: sweep '{sweep_id}' not found on '{cluster}' — no llmb-sweep-{sweep_id} "
            f"ConfigMap. If you have the local sweep record, collect from the machine that submitted."
        )
    d = items[0].get("data", {}) or {}
    ns = d.get("namespace") or items[0].get("metadata", {}).get("namespace", "") or (prof.get("NAMESPACE") or "")
    run_ids = [x for x in (d.get("run_ids", "") or "").split() if x]
    return {
        "sweep_id": sweep_id,
        "cell": d.get("cell", ""),
        "profile": cluster,
        "namespace": ns,
        "recipe": d.get("recipe", ""),
        "run_ids": run_ids,
        "legs": [{"run_id": r} for r in run_ids],
        "mode": d.get("mode", ""),
        "stagger_seconds": d.get("stagger_seconds", ""),
    }


def _latest_valued_run_id(cell: str) -> str:
    """The run_id of the leg cell's LATEST valued runs.jsonl entry — the exact id repro_consolidate merges
    into the original's ledger. Used to scope the variance band to THIS sweep's legs (D6).
    """
    jl = Path(cell) / "runs.jsonl"
    if not jl.is_file():
        return ""
    rid = ""
    for x in jl.read_text().splitlines():
        if not x.strip():
            continue
        try:
            row = json.loads(x)
        except Exception:
            continue
        if row.get("value") is not None and row.get("run_id"):
            rid = row["run_id"]
    return rid


def _leg_run_id(tag: str, sweep: dict, cluster: str | None = None) -> str:
    """Map a sweep LEG TAG (r174cf19) to the run-id submit actually minted for it (ttje16h).

    THESE ARE NOT THE SAME ID, and assuming they were broke `collect --sweep` outright. The sweep
    record stores leg tags; submit.sh passes the tag down only as a --label and run_id.py mints the
    real id. Collect looked up .submits/<tag>.json, which never exists, so every leg failed to
    resolve and the documented variance-sweep workflow could not be completed at all. Same root
    cause as the --serial wait and the run.sh fetch mismatch — three sites, one wrong assumption.

    Resolution order, most authoritative first:
      1. the sweep record, when submit recorded the minted id (sweeps taken after that fix);
      2. the leg's own .submits record, located by its cell path ending in /legs/<tag> — exact and
         offline, and it also recovers sweeps submitted BEFORE submit started recording the id;
      3. with --cluster, the leg's bench Job labels (cell -> run-id), for a cold reconnect from
         another machine.
    Returns "" when nothing resolves, so the caller can name that leg instead of skipping it.
    """
    for leg in sweep.get("legs", []) or []:
        if tag in (leg.get("tag"), leg.get("run_id")) and leg.get("submitted_run_id"):
            return leg["submitted_run_id"]
    if SUBMITS_DIR.is_dir():  # newest first, so a re-submitted leg wins
        for p in sorted(SUBMITS_DIR.glob("*.json"), key=lambda q: q.stat().st_mtime, reverse=True):
            try:
                rec = json.loads(p.read_text())
            except Exception:
                continue
            if (rec.get("cell") or "").rstrip("/").endswith(f"/legs/{tag}") and rec.get("run_id"):
                return rec["run_id"]
    if cluster:
        cellname = next(
            (
                leg.get("cell_name") or ""
                for leg in sweep.get("legs", []) or []
                if tag in (leg.get("tag"), leg.get("run_id"))
            ),
            "",
        )
        ns = sweep.get("namespace") or ""
        if cellname and ns:
            try:
                prof = pr._read_env(pr.profile_env_path(cluster))
                out = subprocess.run(
                    [
                        *_kbase(prof),
                        "-n",
                        ns,
                        "get",
                        "job",
                        "-l",
                        f"llmb.nvidia.com/cell={cellname}",
                        "-o",
                        r"jsonpath={.items[0].metadata.labels.llmb\.nvidia\.com/run-id}",
                    ],
                    capture_output=True,
                    text=True,
                )
                if out.returncode == 0 and out.stdout.strip():
                    return out.stdout.strip()
            except Exception:
                pass
    return ""


def _leg_cell(rid: str, sweep: dict) -> str:
    """The leg's cell dir: prefer its LOCAL submit record's cell; else the sweep record's scratch_cell entry."""
    rec = _local_record(rid)
    if rec and rec.get("cell"):
        return rec["cell"]
    for leg in sweep.get("legs", []) or []:
        if leg.get("run_id") == rid and leg.get("scratch_cell"):
            return leg["scratch_cell"]
    return ""


def cmd_collect_sweep(sweep_id: str, cluster: str | None, extra: list[str]) -> int:
    """collect ALL legs of a variance sweep, then consolidate their values into the ORIGINAL cell's runs.jsonl
    (repro_consolidate.py — stamped with the original recipe_hash) and print compare.py --repro's spread band.
    """
    sw = _resolve_sweep(sweep_id, cluster)
    orig_cell = sw.get("cell") or ""
    if not orig_cell or not (Path(orig_cell) / "recipe.yaml").exists():
        sys.exit(
            f"resilient_status collect --sweep: cannot resolve the ORIGINAL cell '{orig_cell}' for sweep "
            f"'{sweep_id}' (moved? run from the machine that submitted, where the sweep record points at it)."
        )
    run_ids = sw.get("run_ids") or [leg.get("run_id") for leg in sw.get("legs", []) if leg.get("run_id")]
    if not run_ids:
        sys.exit(f"resilient_status collect --sweep: sweep '{sweep_id}' lists no legs.")
    print(f"── variance sweep {sweep_id}: collecting {len(run_ids)} leg(s) → {orig_cell} ──")
    ok, leg_cells, unresolved = 0, [], []
    for tag in run_ids:
        # The sweep stores leg TAGS; submit minted a DIFFERENT run-id for each. Resolve before
        # collecting — passing the tag through was the bug that made this command always fail.
        rid = _leg_run_id(tag, sw, cluster)
        if not rid:
            unresolved.append(tag)
            print(f"\n── leg {tag} ─────────────────────────────")
            print(
                f"collect --sweep: cannot resolve leg '{tag}' to the run-id submit minted for it. "
                f"No .submits record points at .../legs/{tag}"
                + ("" if cluster else " — retry with --cluster <profile> to resolve from the Job labels")
                + ".",
                file=sys.stderr,
            )
            continue
        print(f"\n── leg {tag} → run-id {rid} ─────────────────────────────")
        rc = cmd_collect(rid, cluster, extra)
        legcell = _leg_cell(rid, sw)
        if legcell and (Path(legcell) / "runs.jsonl").exists():
            leg_cells.append(legcell)
        if rc == 0:
            ok += 1
        else:
            print(
                f"collect --sweep: leg {rid} did not publish cleanly (rc={rc}); continuing.",
                file=sys.stderr,
            )
    if not leg_cells:
        # Name the actual reason. "did the legs finish?" sent me hunting a cluster problem when the
        # truth was that no leg had even been RESOLVED to a run-id.
        if unresolved:
            print(
                f"collect --sweep: {len(unresolved)}/{len(run_ids)} leg(s) never resolved to a run-id "
                f"({', '.join(unresolved)}) — nothing was collected. The legs may have run fine.",
                file=sys.stderr,
            )
        else:
            print(
                "collect --sweep: no leg produced a valued run to consolidate (did the legs finish + publish?).",
                file=sys.stderr,
            )
        return 1
    if unresolved:
        print(
            f"collect --sweep: WARNING — {len(unresolved)} leg(s) unresolved and NOT in the band: "
            f"{', '.join(unresolved)}. The spread below covers {len(leg_cells)} leg(s), not {len(run_ids)}.",
            file=sys.stderr,
        )
    # consolidate the legs' values into the ORIGINAL cell's runs.jsonl (same benchmark_id → legit repeats).
    subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "repro_consolidate.py"),
            orig_cell,
            *leg_cells,
        ]
    )
    # the variance band — SCOPED to this sweep's own legs (D6). Without --run-ids, compare --repro would band
    # over EVERY same-cluster valued row in the original's ledger (unrelated historical runs), overstating or
    # understating the sweep's spread. Pass the leg run_ids so the reported band reflects only the N repeats.
    leg_run_ids = [rid for c in leg_cells if (rid := _latest_valued_run_id(c))]
    compare_cmd = [
        sys.executable,
        str(ROOT / "analysis" / "compare.py"),
        "--repro",
        orig_cell,
    ]
    if leg_run_ids:
        compare_cmd += ["--run-ids", " ".join(leg_run_ids)]
    subprocess.run(compare_cmd)
    # on success, delete the cluster-side sweep ConfigMap so they don't accumulate (idempotent).
    prof_name = sw.get("profile") or cluster
    if prof_name:
        prof = pr._read_env(pr.profile_env_path(prof_name))
        subprocess.run(
            [
                *_kbase(prof, sw.get("namespace")),
                "delete",
                "configmap",
                f"llmb-sweep-{sweep_id}",
                "--ignore-not-found",
                "--wait=false",
            ],
            capture_output=True,
            text=True,
        )
    print(
        f"\ncollect --sweep: {ok}/{len(run_ids)} legs collected; "
        f"{len(leg_cells)} consolidated into {orig_cell}/runs.jsonl (see compare --repro above)."
    )
    return 0 if ok == len(run_ids) else 1


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="resilient_status.py")
    ap.add_argument("verb", choices=["status", "logs", "collect"])
    ap.add_argument("run_id", nargs="?", default=None)
    ap.add_argument("--cluster", default=None)
    ap.add_argument("--tail", type=int, default=None)
    ap.add_argument(
        "--sweep",
        default=None,
        help="collect an ENTIRE variance sweep by sweep-id (submit --repeat N)",
    )
    args, extra = ap.parse_known_args(argv)
    if args.verb == "status":
        if not args.run_id:
            sys.exit("resilient_status status: need a <run-id>")
        return cmd_status(args.run_id, args.cluster)
    if args.verb == "logs":
        if not args.run_id:
            sys.exit("resilient_status logs: need a <run-id>")
        return cmd_logs(args.run_id, args.cluster, args.tail)
    # collect: a whole sweep (--sweep) or a single detached run (<run-id>).
    if args.sweep:
        return cmd_collect_sweep(args.sweep, args.cluster, extra)
    if not args.run_id:
        sys.exit("resilient_status collect: need a <run-id> (or --sweep <sweep-id>)")
    return cmd_collect(args.run_id, args.cluster, extra)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
