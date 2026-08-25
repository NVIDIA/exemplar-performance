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

"""preflight.py <cell-dir> <cluster-profile> [--wait-on-resources] [--stage-only] — does the cluster satisfy this recipe?

Merges the scenario BASELINE (schema/requires-baseline.yaml) with the recipe's envelope.requires and
checks the live cluster: namespace, GPU nodes (product + free count, DRA-aware, taint-tolerated),
PVCs, secrets, operators, NetworkPolicy support, DCGM. Cluster-specific values come from the profile.

  --wait-on-resources   Instead of FAILing when no node has enough FREE GPUs, poll the
                        live free-GPU counts every 60s (printing the 📊 resources line each cycle) until
                        a node frees up, then PASS and continue. Ctrl-C to stop waiting. Any position.

  --stage-only          INSTALL-TIME mode: `install` STAGES a cell (datasets/weights/config) for a run
                        LATER, so live GPU AVAILABILITY is not a precondition — it is re-checked (and
                        hard-gated) at run time. This flag makes preflight CPU-only:
                          · the free-GPU capacity checks report WARN instead of FAIL (a cell whose GPUs
                            are merely busy right now still stages fine — the old FAIL made `install`
                            unusable on a shared cluster for any multi-GPU cell);
                          · the nvlink-p2p LIVE probe is skipped — it schedules a real GPU pod, so a
                            "prepare only" install would briefly HOLD GPUs it doesn't need.
                        Every config/identity check (arch, image, secrets, PVCs, operators) still hard-FAILs.

Exit 0 if all hard checks PASS (WARNs are advisory), 1 on any FAIL, 2 on usage/cluster-unreachable.
"""

import base64
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("requires: pip install pyyaml")

ROOT = Path(__file__).resolve().parent.parent
rc_fail = 0


# Pinned from the selected profile so every cluster query uses the intended context.
_KUBE_CONTEXT = ""


def _kubectl_argv(args) -> list:
    """Full kubectl argv, pinned to the profile's KUBE_CONTEXT when set. Pure — testable without a cluster."""
    ctx = ["--context", _KUBE_CONTEXT] if _KUBE_CONTEXT else []
    return ["kubectl", *ctx, "--request-timeout=25s", *args]


def krun(args, timeout=30):
    try:
        p = subprocess.run(_kubectl_argv(args), capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except Exception as e:
        return 1, "", str(e)


def fix_line(cmd: str) -> str:
    """Render a one-line, copy-pasteable fix hint printed under a FAIL. Pure."""
    return f"       → fix: {cmd}"


def classify_artifacts_sc(art_phase: str, sc_rc: int, sc_stderr: str) -> tuple[str, str]:
    """Return the artifacts storage-class verdict without querying a cluster.

    Precedence:
      1. an already-Bound artifacts PVC makes the class MOOT (sweep.sh reuses it)          → PASS  'moot-bound'
      2. else a readable storage class that exists                                         → PASS  'exists'
      3. else an RBAC-forbidden lookup can't verify — don't block a namespaced run over it → WARN  'rbac-forbidden'
      4. else the class genuinely isn't there                                              → FAIL  'not-found'
    """
    if art_phase == "Bound":
        return "PASS", "moot-bound"
    if sc_rc == 0:
        return "PASS", "exists"
    if "forbidden" in (sc_stderr or "").lower():
        return "WARN", "rbac-forbidden"
    return "FAIL", "not-found"


_counts = {"PASS": 0, "WARN": 0, "FAIL": 0}


def section(title: str) -> None:
    """Print a labelled section divider to group related checks visually."""
    print(f"\n  ── {title} {'─' * max(2, 52 - len(title))}")


def line(status, msg, fix=None):
    global rc_fail
    icon = {"PASS": "✅", "WARN": "🟡", "FAIL": "❌", "SKIP": "·"}[status]
    if status == "FAIL":
        rc_fail = 1
    if status in _counts:
        _counts[status] += 1
    print(f"  {icon} {msg}")
    if fix and status == "FAIL":  # 0.7: only FAILs carry an actionable fix
        print(fix_line(fix))


def _cpu_millicores(v):
    """PURE. A k8s cpu quantity → millicores. '139580m'→139580, '118'→118000, ''/junk→None."""
    s = str(v or "").strip()
    if not s:
        return None
    try:
        return int(s[:-1]) if s.endswith("m") else int(float(s) * 1000)
    except ValueError:
        return None


_MEM_UNITS = {
    "Ki": 1 / (1024**2),
    "Mi": 1 / 1024,
    "Gi": 1.0,
    "Ti": 1024.0,
    "K": 1000 / (1024**3),
    "M": 1000**2 / (1024**3),
    "G": 1000**3 / (1024**3),
}


def _mem_gib(v):
    """PURE. A k8s memory quantity → whole GiB (floor). '948007936Ki'→903, '903Gi'→903, ''/junk→None."""
    s = str(v or "").strip()
    if not s:
        return None
    for unit, mult in sorted(_MEM_UNITS.items(), key=lambda kv: -len(kv[0])):
        if s.endswith(unit):
            try:
                return int(float(s[: -len(unit)]) * mult)
            except ValueError:
                return None
    try:
        return int(float(s) / (1024**3))  # bare bytes
    except ValueError:
        return None


def parse_env(path):
    """Minimal .env parser. Mirrors model_cache.parse_profile_env / profile_resolver._read_env — three
    copies of one dialect, so any divergence between them is a divergence about the same file.
    """
    out = {}
    for ln in Path(path).read_text().splitlines():
        ln = ln.strip()
        # `export KEY=value` is an assignment to `sh`, and deploy.sh sources this same file. Without this
        # the key parsed as `export KEY` and the real key read as absent — an empty MODEL_CACHE_PVC, which
        # preflight treats as a hard failure.
        if ln.startswith("export "):
            ln = ln[len("export ") :].lstrip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip()
        if v[:1] in (
            '"',
            "'",
        ):  # quoted value: take up to the closing quote (keeps '# ' inside)
            q = v[0]
            j = v.find(q, 1)
            v = v[1:j] if j != -1 else v[1:]
        else:  # bare value: drop an inline comment (KEY=val  # note)
            v = re.split(r"\s+#", v, maxsplit=1)[0].strip()
        out[k.strip()] = v
    return out


def merge(base, extra):
    m = {**(base or {})}
    for k, v in (extra or {}).items():
        if isinstance(v, dict) and isinstance(m.get(k), dict):
            m[k] = {**m[k], **v}
        elif isinstance(v, list):
            m[k] = sorted(set((m.get(k) or []) + v))
        else:
            m[k] = v
    return m


# Deploy-time profile vars that SILENTLY break scheduling/mounting if empty: envsubst leaves an empty
# schedulerName, a no-match nodeSelector, or a bad PVC/secret ref, and the pod never schedules with no error.
# (Runtime shell vars like ${STOP_STAT}/${KARCH}/${SERVER_URL} are NOT profile-sourced — excluded here, which
# also avoids the dryrun false-positive that flagged them.)
CRITICAL_PROFILE_VARS = {
    "NAMESPACE",
    "GPU_PRODUCT",
    "SCHEDULER_NAME",
    "MODEL_CACHE_PVC",
    "IMAGE_PULL_SECRET",
}


# ---------------------------------------------------------------------------
# Whole-node co-scheduling budget
# ---------------------------------------------------------------------------
# A cell can pin its bench pod to the SAME NODE as its server (bench.colocate_with_server -> a HARD
# podAffinity on kubernetes.io/hostname). When it does, the server's whole-node reservation and the bench
# pod's requests are competing for ONE node's capacity, and the existing WHOLE_NODE_* checks — each of
# which validates a single value in isolation — all pass on a combination that can never schedule.
#
# Observed: WHOLE_NODE_CPU=180 on a 192-cpu node, BENCH_CPU_REQUEST=16. Both individually valid. The
# server came up 3/3 Running and looked healthy; the bench pod sat FailedScheduling for 11 minutes on
# "0/11 nodes are available: 1 node(s) didn't match pod affinity rules, 10 Insufficient cpu". preflight
# had passed. The reservation left ~9.7 cpu free against a 16 cpu request.

_SAME_HOST_TOPOLOGY = "kubernetes.io/hostname"


_UNSET_TOKEN = "__llmb_unset__"

# RUNTIME DEFAULTS THE LAUNCHERS APPLY, mirrored here so preflight reads the manifest the way it will
# actually be rendered. BENCH_CPU_REQUEST is an OPTIONAL profile var — it is not in CRITICAL_PROFILE_VARS,
# profile_init writes it but a hand-written profile need not — and sweep.sh:83, submit.sh:95 and dryrun.sh:83
# all default it with `: "${BENCH_CPU_REQUEST:=16}"` before envsubst. Substituting from the profile ALONE
# left it UNKNOWN on every profile that omits it, which silently switched the co-scheduling CPU comparison
# off (see _coschedule_verdict) — the exact 180+16 > 192 case the gate exists for — and printed the raw
# _UNSET_TOKEN inside a PASS line. selftest_preflight_gates asserts this table against those three scripts,
# so the default cannot drift away from the one the launcher uses.
_RUNTIME_DEFAULTS = {"BENCH_CPU_REQUEST": "16"}


def _subst_profile_vars(text: str, prof: dict) -> str:
    """Replace ${VAR} with the profile's value (or the launcher's documented default); UNKNOWN vars become a
    harmless token, never left as `${VAR}`.

    Leaving them breaks the YAML outright: `persistentVolumeClaim: { claimName: ${MODEL_CACHE_PVC} }` has
    the `{` of `${...}` open a nested flow mapping, so the parse dies and — with a bare `except` — the
    whole check would silently no-op. The token also keeps the value UNPARSEABLE as a quantity, so a var
    with no value AND no launcher default reads as UNKNOWN rather than as zero."""

    def _val(name: str) -> str:
        if str(prof.get(name, "")).strip() != "":
            return str(prof[name])
        # `: "${VAR:=16}"` in the launcher replaces an ABSENT var and an EMPTY one alike, so both take the
        # default here. Vars with no launcher default keep the previous behaviour exactly: absent -> the
        # unparseable token (UNKNOWN), explicitly-empty -> "" (which is what envsubst would write).
        if name in _RUNTIME_DEFAULTS:
            return _RUNTIME_DEFAULTS[name]
        return str(prof.get(name, _UNSET_TOKEN))

    return re.sub(r"\$\{([A-Z_][A-Z0-9_]*)\}", lambda m: _val(m.group(1)), text)


def bench_coschedule_demand(cell, prof) -> dict:
    """PURE — what this cell's bench pod demands on the SERVER'S node, or {} when it is free to schedule
    elsewhere. Reads the RENDERED manifests, i.e. exactly what will be applied.

    Returns {} unless the bench pod carries a REQUIRED (not preferred) podAffinity on
    kubernetes.io/hostname — the check must not fire for cells whose bench can land anywhere, which is
    every cell but the KVBM one today.

    Otherwise: {"manifest", "cpu_m", "mem_gib", "cpu_raw", "mem_raw"}. Values are read after profile
    substitution because the two numbers come from different places: cpu is ${BENCH_CPU_REQUEST} (profile)
    while memory is a literal in the manifest. A quantity we cannot parse stays None — never 0, which would
    silently claim the bench pod needs nothing."""
    out: dict = {}
    for f in sorted((Path(cell) / "rendered").glob("*.yaml")):
        if f.name not in ("bench-job.yaml",):
            continue
        raw = f.read_text()
        if "podAffinity" not in raw or "requiredDuringSchedulingIgnoredDuringExecution" not in raw:
            continue
        # MULTI-DOCUMENT: a bench manifest bundles ServiceAccount/Role/RoleBinding alongside the Job, so
        # load_all and pick the Job — safe_load() alone silently returns only the first document.
        try:
            docs = [d for d in yaml.safe_load_all(_subst_profile_vars(raw, prof)) if d]
        except Exception as e:
            return {"parse_error": f"{f.name}: {type(e).__name__}: {str(e)[:120]}"}
        jobs = [d for d in docs if d.get("kind") == "Job"]
        if not jobs:
            continue
        spec = ((jobs[0].get("spec") or {}).get("template") or {}).get("spec") or {}
        terms = ((spec.get("affinity") or {}).get("podAffinity") or {}).get(
            "requiredDuringSchedulingIgnoredDuringExecution"
        ) or []
        if not any((tm or {}).get("topologyKey") == _SAME_HOST_TOPOLOGY for tm in terms):
            continue  # preferred-only, or a different topology → the bench can go elsewhere
        cpu_m = mem_gib = 0
        cpu_raw, mem_raw = [], []
        for c in spec.get("containers") or []:
            req = (c.get("resources") or {}).get("requests") or {}
            if req.get("cpu") is not None:
                cpu_raw.append(str(req["cpu"]))
                cpu_m = None if cpu_m is None else _add_or_none(cpu_m, _cpu_millicores(req.get("cpu")))
            if req.get("memory") is not None:
                mem_raw.append(str(req["memory"]))
                mem_gib = None if mem_gib is None else _add_or_none(mem_gib, _mem_gib(req.get("memory")))
        out = {
            "manifest": f.name,
            "cpu_m": cpu_m or None,
            "mem_gib": mem_gib or None,
            "cpu_raw": "+".join(cpu_raw),
            "mem_raw": "+".join(mem_raw),
        }
        break
    return out


def _coschedule_verdict(
    wn_cpu_raw,
    wn_mem_raw,
    wn_cpu_m,
    wn_mem_gib,
    cosched,
    alloc_cpu,
    alloc_mem,
    free_cpu,
    free_mem,
    node,
    profile,
):
    """PURE — does the whole-node reservation leave room for the bench pod that MUST share its node?
    Returns (message, fix) on failure, or () when it fits / cannot be judged.

    Checked against BOTH ceilings, and the tighter one wins:
      * ALLOCATABLE — a hard physical impossibility, true regardless of what else is running.
      * FREE        — allocatable minus what is already requested there (DaemonSets, co-tenants). This is
                      the number that actually bit: 180+16=196 was over free (~189.7) as well as over
                      allocatable (192), and free is what the scheduler compares against.

    The message carries BOTH variables and the arithmetic, because the fix is a TRADE-OFF between them:
    an operator who sees only "insufficient cpu" cannot tell which knob to turn, and the recommended
    ceiling has to come from somewhere they can check."""
    if not cosched:
        return ()
    b_cpu, b_mem = cosched.get("cpu_m"), cosched.get("mem_gib")
    fixhint = (
        f"lower WHOLE_NODE_CPU / WHOLE_NODE_MEM in cluster-profiles/{profile}.env, or lower "
        f"BENCH_CPU_REQUEST — they share one node, so they share one budget"
    )
    # TIGHTER CEILING FIRST. free <= allocatable always, and free is what the scheduler compares against,
    # so a recommendation derived from allocatable can still be unschedulable: on the live node,
    # allocatable said "lower to <=176" while DaemonSets had already taken ~2.3 cpu, making the real
    # ceiling 173. Advice that still does not fit is worse than no advice.
    for _label, ceil_cpu, ceil_mem, ceil_note in (
        (
            "free",
            free_cpu,
            free_mem,
            f"FREE on {node} (allocatable − already requested)",
        ),
        ("allocatable", alloc_cpu, alloc_mem, f"{node} allocatable"),
    ):
        if wn_cpu_m and b_cpu and ceil_cpu and (wn_cpu_m + b_cpu) > ceil_cpu:
            _max = (ceil_cpu - b_cpu) // 1000
            return (
                f"whole_node co-scheduling: this cell pins its bench pod to the SERVER'S node "
                f"(required podAffinity on kubernetes.io/hostname), so both must fit at once — "
                f"WHOLE_NODE_CPU={wn_cpu_raw} + BENCH_CPU_REQUEST={cosched.get('cpu_raw')} = "
                f"{(wn_cpu_m + b_cpu) / 1000:g} cpu > {ceil_cpu / 1000:g} {ceil_note}. "
                f"The server will come up and look healthy; the BENCH pod will sit FailedScheduling "
                f"('Insufficient cpu') until it times out.",
                f"lower WHOLE_NODE_CPU to ≤{_max}, or lower BENCH_CPU_REQUEST to "
                f"≤{(ceil_cpu - wn_cpu_m) / 1000:g} — {fixhint}",
            )
        if wn_mem_gib and b_mem and ceil_mem and (wn_mem_gib + b_mem) > ceil_mem:
            _maxm = ceil_mem - b_mem
            return (
                f"whole_node co-scheduling: this cell pins its bench pod to the SERVER'S node "
                f"(required podAffinity on kubernetes.io/hostname), so both must fit at once — "
                f"WHOLE_NODE_MEM={wn_mem_raw} + bench memory {cosched.get('mem_raw')} = "
                f"{wn_mem_gib + b_mem}Gi > {ceil_mem}Gi {ceil_note}. The server will come up and look "
                f"healthy; the BENCH pod will sit FailedScheduling ('Insufficient memory').",
                f"lower WHOLE_NODE_MEM to ≤{_maxm}Gi — {fixhint}",
            )
    return ()


def _add_or_none(acc, v):
    """Sum that PROPAGATES unknown: an unparseable request must not read as 0."""
    return None if (acc is None or v is None) else acc + v


def coschedule_unreadable(cosched: dict) -> list[str]:
    """PURE — which of the co-scheduled bench pod's requests could NOT be read. [] means both were.

    ANY of them, not ALL of them. The caller's guard was `cpu_m is None and mem_gib is None`, and the bench
    pod's memory is a LITERAL in the manifest (16Gi) while its cpu is ${BENCH_CPU_REQUEST} — so `mem_gib` is
    never None and the branch was unreachable. _coschedule_verdict then skipped the cpu comparison
    (`if wn_cpu_m and b_cpu and ceil_cpu`) and returned "fits", and preflight printed a PASS reading
    `180+__llmb_unset__ cpu` — a gate that did not run, rendered as a gate that passed.
    """
    return [n for n, v in (("cpu", cosched.get("cpu_m")), ("memory", cosched.get("mem_gib"))) if v is None]


def rendered_profile_refs(cell):
    """Profile-style ${VAR}s referenced by this cell's rendered manifests."""
    refs = set()
    for f in (Path(cell) / "rendered").glob("*.yaml"):
        refs |= set(re.findall(r"\$\{([A-Z_][A-Z0-9_]*)\}", f.read_text()))
    return refs


def cell_mounts_model_cache(cell) -> bool:
    """PURE — do THIS cell's rendered manifests actually reference ${MODEL_CACHE_PVC}?

    The gate that REQUIRES a resolved claim keys off this, not off the lane, because the property that
    matters is SUBSTITUTION. A lane's envsubst whitelist can name a var its manifest never mentions
    so a whitelist is not evidence. A non-serving driver Job mounts only
    `<name>-artifacts`; refusing a non-mounting cell would be incorrect. Every one of the 31 shipped cells DOES reference
    it, so this changes nothing for them — it fails closed exactly where the claim is really used.
    """
    return "MODEL_CACHE_PVC" in rendered_profile_refs(cell)


def missing_profile_vars(cell, prof):
    """Scheduling/mount-critical ${VAR}s the cell's manifests reference but the profile leaves empty/absent."""
    refs = rendered_profile_refs(cell)
    return sorted(v for v in (refs & CRITICAL_PROFILE_VARS) if not (prof.get(v) or "").strip())


def secret_content_issue(secret_json):
    """A secret can EXIST but be useless: an empty token value → the server 401s minutes into startup with a
    cryptic HfHubHTTPError. Return a problem string, or None if the content looks usable. (Catches empty-value;
    a *wrong* token would need a live HF auth call.)"""
    stype = secret_json.get("type", "Opaque")
    data = secret_json.get("data") or {}
    if stype == "kubernetes.io/dockerconfigjson":
        return (
            None if (data.get(".dockerconfigjson") or "").strip() else "empty .dockerconfigjson (image pull will fail)"
        )
    if not data:
        return "secret has no data keys"
    empty = []
    for k, v in data.items():
        try:
            if not base64.b64decode(v or "").strip():
                empty.append(k)
        except Exception:
            empty.append(k)
    return f"empty value for key(s): {', '.join(sorted(empty))}" if empty else None


def parse_cache_integrity_report(text: str, server_path: str = "") -> dict:
    """Delegates to model_cache.parse_cache_integrity_report — ONE parser, so install's "should I download
    this?" and preflight's "may this run start?" read the same evidence off the same paths. Kept as a
    module-level name because selftest_preflight_gates and capability_registry import it from here.
    """
    import model_cache as _mc

    return _mc.parse_cache_integrity_report(text, server_path)


def probe_model_cache_integrity(ns, prof, recipe):
    """Read-only verification of the exact Hugging Face snapshot used by the server.

    Requires configuration plus all resolved weight shards. Probe failures return UNKNOWN;
    incomplete snapshots return structured evidence for the capability gate.
    """
    import capability_registry as _cap

    serving = recipe.get("serving") or {}
    repo = str(serving.get("model_repo", "")).strip()
    rev = str(serving.get("model_revision", "")).strip()
    subpath = str(prof.get("MODEL_CACHE_SUBPATH", "")).strip()
    pvc = str(prof.get("MODEL_CACHE_PVC", "")).strip()
    if not (repo and rev and pvc):
        return {"probe_error": "missing serving.model_repo/model_revision or MODEL_CACHE_PVC"}
    import model_cache as _mc

    snap = _cap.server_snapshot_dir(subpath, repo, rev)  # relative to the cache mount root
    script = _mc.cache_probe_script(subpath, repo, rev)
    pod = f"llmb-preflight-cache-{rev[:8]}"
    # PLACEMENT: this pod MUST land where the claim actually mounts. It previously carried no tolerations
    # at all, so on a cluster whose only mountable nodes are tainted (nvidia.com/gpu=:NoSchedule) it could
    # never reach one -- the integrity check was unable to succeed by construction, and reported that as a
    # generic "cache mounter pod did not become ready".
    _sel, _tol, _ = _mc.cache_pod_placement(prof)
    _spec = {
        "restartPolicy": "Never",
        "tolerations": _tol,
        "volumes": [{"name": "c", "persistentVolumeClaim": {"claimName": pvc, "readOnly": True}}],
        "containers": [
            {
                "name": "p",
                "image": "busybox:1.36",
                "command": ["sleep", "120"],
                "volumeMounts": [{"name": "c", "mountPath": "/cache", "readOnly": True}],
            }
        ],
    }
    if _sel:
        _spec["nodeSelector"] = _sel
    ov = json.dumps({"spec": _spec})
    # Capture the CREATE error. classify_mounter_failure takes it as a third argument and needs it to tell
    # rbac-denied from timeout: when PodSecurity Admission or Kyverno rejects the pod, `run` fails outright
    # and NO pod is ever created — so there is no phase and no event, and the wait below simply times out.
    # Dropping this rc is why preflight reported a PSA rejection as "the mounter pod did not become ready",
    # while install.py (which passes it) correctly reported "rbac-denied: not allowed to create the probe
    # pod: <error>". Same helper, same cluster, two different diagnoses — one of them useless.
    _rc_c, _, _err_c = krun(
        [
            "-n",
            ns,
            "run",
            pod,
            "--image=busybox:1.36",
            "--restart=Never",
            f"--overrides={ov}",
        ],
        timeout=30,
    )
    try:
        rc_wait, _, _ = krun(
            ["-n", ns, "wait", "pod", pod, "--for=condition=ready", "--timeout=60s"],
            timeout=70,
        )
        if rc_wait != 0:
            _, _ev, _ = krun(
                [
                    "-n",
                    ns,
                    "get",
                    "events",
                    "--field-selector",
                    f"involvedObject.name={pod}",
                    "-o",
                    "jsonpath={.items[*].message}",
                ],
                timeout=20,
            )
            _, _ph, _ = krun(
                ["-n", ns, "get", "pod", pod, "-o", "jsonpath={.status.phase}"],
                timeout=20,
            )
            _code, _why = _mc.classify_mounter_failure(_ph or "", _ev or "", _err_c or "")
            return {
                "probe_error": f"{_code}: {_why}. {_mc.cache_placement_hint(prof)}",
                "server_path": snap,
            }
        rc_exec, out, _ = krun(["-n", ns, "exec", pod, "--", "sh", "-c", script], timeout=45)
        if rc_exec != 0:
            return {"probe_error": "could not read the cache path", "server_path": snap}
        return parse_cache_integrity_report(out, server_path=snap)
    finally:
        krun(
            ["-n", ns, "delete", "pod", pod, "--ignore-not-found", "--wait=false"],
            timeout=20,
        )


def probe_image_pull_access(ns, prof, recipe):
    """Best-effort, read-only. Reproduce the kubelet's pull AUTHORIZATION for each image the cell pins,
    using the namespace's IMAGE_PULL_SECRET dockerconfig cred — so a credential that authenticates to the
    registry but LACKS pull access to a private repo (a private-repo 403) FAILs at preflight
    (pre-GPU) instead of ImagePullBackOff holding GPUs for 30min. Returns the facts dict
    capability_registry._probe_image_pull_access consumes: {'secret':.., 'results': {ref:{...}}}, or
    results=None → UNKNOWN (safe-degrade) when the secret can't be read/parsed. The probe is auth + manifest
    HEAD only (no real pull), makes outbound registry calls, and never mutates the cluster.
    """
    import capability_registry as _cap

    secret = str(prof.get("IMAGE_PULL_SECRET") or "").strip()
    images = _cap.pinned_images(recipe)
    if not images:
        return {"secret": secret, "results": {}}
    dockercfg = None
    if secret:
        rc, out, err = krun(
            [
                "-n",
                ns,
                "get",
                "secret",
                secret,
                "-o",
                "jsonpath={.data.\\.dockerconfigjson}",
            ]
        )
        if rc != 0:
            # Unreadable secret (RBAC / absent) → UNKNOWN, never a false FAIL (present+non-empty is a
            # separate preflight check; here we only judge pull AUTHORIZATION when we can read the cred).
            return {
                "secret": secret,
                "results": None,
                "probe_error": f"pull secret unreadable: {(err or '').strip() or 'not found'}",
            }
        if (out or "").strip():
            try:
                dockercfg = base64.b64decode(out.strip())
            except Exception:
                return {
                    "secret": secret,
                    "results": None,
                    "probe_error": "unparseable .dockerconfigjson",
                }
    try:
        results = _cap.probe_image_pull_access(images, dockercfg, _cap._default_pull_http)
    except Exception as e:  # any probe failure → UNKNOWN, never a block
        return {"secret": secret, "results": None, "probe_error": str(e)}
    return {"secret": secret, "results": results}


# ── Gap 2: nvlink-p2p is INIT-TIME cached, not per-run ──────────────────────────────────────────────
# NVLINK_P2P is written into the profile only at `profile init` (probe_fabric) and, for disagg cells, at
# run.sh Phase-1.5 — so a TP>1 AGGREGATED cell reads a possibly-stale fact: a fabric that DEGRADED after
# init, or an init where the 2-GPU probe couldn't schedule (→ `unknown`), is NOT caught. Preflight therefore
# actively RE-PROBES the live P2P health for a recipe that requires it, rather than trusting only the cached
# profile fact. Bounded + safe-degrade (any probe error → keep the cached fact, never a false block).
def should_probe_p2p_live(required: bool, cached_fresh: bool) -> bool:
    """PURE. Re-probe NVLink P2P live in preflight when the recipe REQUIRES it (tp>1) — the profile fact is
    init-time cached and can't see a post-init fabric degrade. Skip only when a FRESH live result is already
    in hand (NVLINK_P2P_FRESH — e.g. run.sh Phase-1.5 wrote one for a disagg cell) so we don't double-probe.
    Never probe when the entry isn't required (respects the recipe-scoping spine)."""
    return bool(required) and not bool(cached_fresh)


def merge_p2p_fact(cached: dict, live) -> dict:
    """PURE. Prefer a DEFINITIVE live probe (healthy|disabled) over the init-cached profile fact so a fabric
    that degraded after profile-init is caught. Fall back to the cached fact when the live probe couldn't
    run (unknown / None) — safe-degrade: never lose a measured fact and never turn a probe blip into a
    false block."""
    if live:
        st = str(live.get("state", "")).strip().lower()
        if st in ("healthy", "disabled"):
            return {"state": st, "route_unhealthy": live.get("route_unhealthy")}
    return cached or {}


def first_gpu_node(gpu_product: str) -> str:
    """IMPURE. First node name labelled with the profile's GPU_PRODUCT, or '' (→ skip the live probe)."""
    sel = ["-l", f"nvidia.com/gpu.product={gpu_product}"] if gpu_product else []
    rc, out, _ = krun(["get", "nodes", *sel, "-o", "jsonpath={.items[0].metadata.name}"])
    return out.strip() if rc == 0 else ""


def live_probe_p2p(ns: str, prof: dict, node_name: str):
    """IMPURE. Run the live NVLink-P2P GPU probe (probe_fabric) so preflight doesn't trust only the cached
    profile fact (Gap 2). Bounded + safe-degrade: ANY error → None (keep the cached fact, never a false
    block). Returns probe_fabric.probe_nvlink_p2p's dict, or None.

    `node_name` is used only as an EXISTENCE GATE (≥1 GPU node with this GPU_PRODUCT exists before we
    spin a pod). The probe itself is SCHEDULER-PLACED via nodeSelector on GPU_PRODUCT — pinning the pod
    to one node bypasses the scheduler's resource gate and gets kubelet-rejected on a busy node
    (UnexpectedAdmissionError), returning a useless `unknown` instead of the real fabric verdict.
    """
    if not (ns and node_name):
        return None
    try:
        import probe_fabric as _pf

        _pf._KUBE_CONTEXT = _KUBE_CONTEXT  # pin the probe to the profile's cluster
        gpu_product = (prof.get("GPU_PRODUCT") or "").strip()
        image = (prof.get("P2P_PROBE_IMAGE") or "").strip() or _pf.P2P_PROBE_IMAGE
        pull_secret = (prof.get("IMAGE_PULL_SECRET") or "").strip() or None
        try:
            gpus = int((prof.get("P2P_PROBE_GPUS") or "").strip() or _pf.P2P_PROBE_GPUS)
        except ValueError:
            gpus = _pf.P2P_PROBE_GPUS
        return _pf.probe_nvlink_p2p(ns, gpu_product, image=image, pull_secret=pull_secret, gpu_count=gpus)
    except Exception:
        return None


def gpu_resource_summary(nodes, used, product) -> str:
    """Aggregate per-node GPU accounting into one resources line. Pure: `nodes` are kubectl
    node dicts, `used` is {node_name: gpus_in_use}. Surfaces total/in-use/free + the biggest single free node
    (what actually matters — a recipe needs its GPUs on ONE node), before the pass/fail verdict.
    """

    def alloc(n):
        return int((n.get("status", {}).get("allocatable", {}) or {}).get("nvidia.com/gpu", 0) or 0)

    total = sum(alloc(n) for n in nodes)
    inuse = sum(min(used.get(n["metadata"]["name"], 0), alloc(n)) for n in nodes)
    biggest = max((alloc(n) - used.get(n["metadata"]["name"], 0) for n in nodes), default=0)
    return (
        f"{product or 'GPU'}: {len(nodes)} node(s) · {total} total · {inuse} in use · "
        f"{total - inuse} free  (biggest free node: {biggest})"
    )


# Match RDMA/InfiniBand node-selector keys in rendered manifests.
# Rendered placeholders prevent a full YAML parse, so this scan is intentionally regex-based.
_RDMA_LABEL_RE = re.compile(
    r'^\s*([A-Za-z0-9][\w./-]*(?:rdma|infiniband)[\w./-]*)\s*:\s*"?([^"#\n]+?)"?\s*$',
    re.I,
)


def is_disagg(serving: dict) -> bool:
    """Does this recipe use a disaggregated (RDMA/InfiniBand KV-transfer) serving stack?"""
    return bool(serving.get("disagg")) or str(serving.get("stack", "")).endswith("-disagg")


def disagg_cache_access_issue(serving: dict, access_modes) -> str | None:
    """Explain why one cache claim cannot serve this disaggregated topology, or return None.

    Prefill and decode workers load weights and may land on different nodes while sharing MODEL_CACHE_PVC.
    ReadWriteOnce is unsafe because whichever worker attaches first can leave the others stuck in Init with
    Multi-Attach. RWX and ROX are both valid because the workers mount the cache read-only.
    """
    roles = [r for r in ("prefill", "decode") if (serving.get("disagg") or {}).get(r)]
    if not is_disagg(serving):
        return None
    modes = {str(m).strip() for m in (access_modes or []) if str(m).strip()}
    if modes & {"ReadWriteMany", "ReadOnlyMany"}:
        return None
    shown = ",".join(sorted(modes)) or "unknown"
    return (
        f"disaggregated workers {roles} share one model-cache claim, but its access mode is {shown}; "
        "prefill/decode may run on different nodes and a single-node RWO attachment will Multi-Attach"
    )


def disagg_rdma_selectors(cell) -> list[tuple[str, str]]:
    """RDMA nodeSelector (key,value) labels the rendered disagg manifests require, deduped + sorted. Pure."""
    seen: dict[str, str] = {}
    for f in sorted((Path(cell) / "rendered").glob("*.yaml")):
        for ln in f.read_text().splitlines():
            m = _RDMA_LABEL_RE.match(ln)
            if m:
                seen[m.group(1)] = m.group(2).strip()
    return sorted(seen.items())


def rdma_node_ok(selectors, node_label_dicts) -> bool:
    """True iff ≥1 node satisfies ALL required RDMA selectors (a k8s nodeSelector is an AND). Pure."""
    return any(all(nd.get(k) == v for k, v in selectors) for nd in node_label_dicts)


def profile_rdma_selectors(prof: dict) -> list[tuple[str, str]]:
    """Return profile-defined RDMA node selectors as sorted key-value pairs.

    Profiles keep cluster label conventions outside portable recipes.
    """
    raw = (prof.get("RDMA_NODE_SELECTOR") or "").strip()
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        if k:
            out[k] = v.strip().strip('"')
    return sorted(out.items())


def parse_args(argv):
    """Accept --wait-on-resources / --stage-only in any position; keep exactly 2 positionals.
    Pure/testable. Returns (cell, profile, opts); raises ValueError on misuse."""
    opts = {"wait_on_resources": False, "stage_only": False}
    pos = []
    for a in argv:
        if a == "--wait-on-resources":
            opts["wait_on_resources"] = True
        elif a == "--stage-only":
            opts["stage_only"] = True
        elif a.startswith("--"):
            raise ValueError(f"unknown flag: {a}")
        else:
            pos.append(a)
    if len(pos) != 2:
        raise ValueError("need exactly <cell-dir> <cluster-profile>")
    return pos[0], pos[1], opts


def _is_gpu_dra_device(d):
    """True iff a DRA allocation-result device is an actual GPU — NOT a ComputeDomain/IMEX channel.
    gpu-operator provisions ComputeDomain claims (NVLink/IMEX fabric) whose JSON also contains
    'nvidia'; counting those as GPUs over-reports usage and can push a node's `used` above its
    allocatable (the impossible `6 used / 4 alloc` row). The NVIDIA GPU DRA driver is `gpu.nvidia.com`;
    ComputeDomain is `compute-domain.nvidia.com` (no 'gpu' token) — so we exclude it explicitly and
    keep the broad 'nvidia' fallback only for genuine GPU devices."""
    blob = json.dumps(d).lower()
    if "computedomain" in blob or "compute-domain" in blob or "imex" in blob:
        return False
    return "gpu" in (d.get("driver", "") + d.get("pool", "")).lower() or "nvidia" in blob


def gpu_availability(nodes, pods, claims, tolerated, want_arch):
    """PURE — the DRA-aware free-GPU accounting, over already-parsed kubectl items.
    shared implementation re-used by preflight AND the --wait-on-resources poll (no duplication).

    `nodes`/`pods`/`claims` are the `.items` lists; `tolerated` is the recipe's tolerated taint keys;
    `want_arch` gates schedulability on kubernetes.io/arch. Returns:
      used         {node: gpus_in_use}   device-plugin requests + DRA-claimed devices
      dra_gpus     int  (GPUs counted via ResourceClaims — device-plugin request-sum alone under-counts)
      node_archs   set of kubernetes.io/arch seen on the product nodes
      schedulable  [[node, free], ...]   taint-tolerated AND arch-matched nodes, with free GPU count
    """

    def alloc(n):
        return int((n.get("status", {}).get("allocatable", {}) or {}).get("nvidia.com/gpu", 0) or 0)

    used, pod_node = {}, {}
    for p in pods:
        if p.get("status", {}).get("phase") in ("Succeeded", "Failed"):
            continue
        nn = p["spec"].get("nodeName")
        if not nn:
            continue
        pod_node[(p["metadata"]["namespace"], p["metadata"]["name"])] = nn
        # containers AND initContainers (an init GPU request still reserves the device)
        g = sum(
            int(c.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0) or 0)
            for c in (p["spec"].get("containers", []) + p["spec"].get("initContainers", []))
        )
        if g:
            used[nn] = used.get(nn, 0) + g
    dra_gpus = 0  # DRA: GPUs claimed via spec.resourceClaims carry NO container request → count them too
    for claim in claims:
        alloc_c = (claim.get("status") or {}).get("allocation") or {}
        devs = ((alloc_c.get("devices") or {}).get("results")) or []
        ndev = sum(1 for d in devs if _is_gpu_dra_device(d))
        if not ndev:
            continue
        for ref in (claim.get("status") or {}).get("reservedFor") or []:
            node = pod_node.get((claim["metadata"]["namespace"], ref.get("name")))
            if node:
                used[node] = used.get(node, 0) + ndev
                dra_gpus += ndev
                break
    node_archs = {n["metadata"].get("labels", {}).get("kubernetes.io/arch", "?") for n in nodes}
    schedulable = []
    for n in nodes:
        name = n["metadata"]["name"]
        node_arch = n["metadata"].get("labels", {}).get("kubernetes.io/arch", "?")
        free = alloc(n) - used.get(name, 0)
        taints = [t["key"] for t in n["spec"].get("taints", []) if t.get("effect", "").startswith("No")]
        blocking = [t for t in taints if t not in tolerated]
        arch_ok = (not want_arch) or (node_arch == want_arch)
        if not blocking and arch_ok:
            schedulable.append([name, free])
    return {
        "used": used,
        "dra_gpus": dra_gpus,
        "node_archs": node_archs,
        "schedulable": schedulable,
    }


def pick_single_node(schedulable, want):
    """PURE — first schedulable node with >= `want` free GPUs as [node, free], else None."""
    return next(([nm, fr] for nm, fr in schedulable if fr >= want), None)


def place_disagg(schedulable, placements):
    """PURE — greedy biggest-free-first placement of disagg role→need pairs across nodes. Returns the
    list of 'role:node(need)' assignments if every role placed, else None. (Mutates a local copy only.)
    """
    pool = sorted(([n, f] for n, f in schedulable), key=lambda x: x[1], reverse=True)
    assigned = []
    for role, need in sorted(placements, key=lambda x: x[1], reverse=True):
        slot = next((nd for nd in pool if nd[1] >= need), None)
        if not slot:
            return None
        slot[1] -= need
        assigned.append(f"{role}:{slot[0]}({need})")
    return assigned


def query_gpu_availability(gpu_product, tolerated, want_arch):
    """LIVE wrapper: run the three kubectl queries, then delegate to gpu_availability (the pure
    accounting). Re-callable each --wait-on-resources cycle. Returns the availability dict with an
    added 'nodes' key, or None if no product nodes are visible."""
    rc, nout, _ = krun(["get", "nodes", "-l", f"nvidia.com/gpu.product={gpu_product}", "-o", "json"])
    _, pout, _ = krun(["get", "pods", "-A", "-o", "json"])
    rc_dra, dcout, _ = krun(["get", "resourceclaims", "-A", "-o", "json"])
    if rc != 0 or not nout:
        return None
    nodes = json.loads(nout).get("items", [])
    pods = json.loads(pout or '{"items":[]}').get("items", [])
    claims = json.loads(dcout or '{"items":[]}').get("items", []) if rc_dra == 0 else []
    avail = gpu_availability(nodes, pods, claims, tolerated, want_arch)
    avail["nodes"] = nodes
    # Pods too: the whole-node cpu/memory gate needs what is ALREADY REQUESTED on the candidate node, not
    # merely its allocatable. free = allocatable - sum(requests); checking against allocatable alone
    # green-lights a request that can never schedule.
    avail["pods"] = pods
    return avail


def wait_for_gpus(gpu_product, tolerated, want_arch, satisfied):
    """--wait-on-resources: re-query free GPUs every 60s — printing the live 📊 resources
    line each cycle — until `satisfied(schedulable)` returns a truthy payload, then return it. Blocks;
    Ctrl-C to stop waiting (surfaces as KeyboardInterrupt to the caller)."""
    # #13: name the context we're watching + flush the FIRST resources summary immediately (query at the TOP of
    # the loop, before the first sleep) so the operator sees the live state right away and knows we're targeting
    # the right cluster — not a 60s silence before the first line. flush=True: stdout is block-buffered when piped.
    ctx = _KUBE_CONTEXT or "(ambient context)"
    print(
        f"  ⏳ --wait-on-resources: watching {gpu_product} on {ctx}; re-checking every 60s. "
        f"Press Ctrl-C to stop waiting.",
        flush=True,
    )
    first = True
    while True:
        if not first:
            time.sleep(60)
        first = False
        avail = query_gpu_availability(gpu_product, tolerated, want_arch)
        if avail is None:
            print(
                f"  📊 resources — no {gpu_product} nodes visible right now (retrying in 60s)",
                flush=True,
            )
            continue
        print(
            f"  📊 resources — {gpu_resource_summary(avail['nodes'], avail['used'], gpu_product)}",
            flush=True,
        )
        result = satisfied(avail["schedulable"])
        if result:
            return result


def main() -> int:
    try:
        cell, profile, opts = parse_args(sys.argv[1:])
    except ValueError:
        sys.exit(__doc__)
    wait_on_resources = opts["wait_on_resources"]
    # --stage-only: capacity is a RUN-time gate, not a staging precondition (see module docstring).
    stage_only = opts["stage_only"]
    # Severity for the free-GPU capacity verdicts: hard FAIL normally, advisory WARN while staging.
    cap_level = "WARN" if stage_only else "FAIL"
    cap_note = (
        "  (--stage-only: staging does not need free GPUs now; capacity is re-checked at run time)"
        if stage_only
        else ""
    )
    recipe = yaml.safe_load((Path(cell) / "recipe.yaml").read_text()) or {}
    env = recipe.get("envelope") or {}
    serving = recipe.get("serving") or {}
    scenario, gpu_type = env.get("scenario"), env.get("gpu_type")
    baseline = (yaml.safe_load((ROOT / "schema" / "requires-baseline.yaml").read_text()) or {}).get(scenario, {})
    req = merge(baseline, env.get("requires"))
    prof_path = ROOT / "cluster-profiles" / f"{profile}.env"
    if not prof_path.exists():
        sys.exit(f"no profile: {prof_path}")
    prof = parse_env(prof_path)
    # Resolve THIS cell's model-cache claim exactly as deploy.sh will (install.resolve_cache_claim over the
    # same profile file), so preflight probes the claim the server will ACTUALLY mount. Reading the raw
    # MODEL_CACHE_PVC meant preflight vouched for a different PVC than the one the deploy mounted whenever a
    # per-model override was in play. Substituting it here also makes the missing-var check below report the
    # real gap: "no cache configured for model X", not "MODEL_CACHE_PVC is empty" on a cluster where the
    # per-model key is the one that matters.
    # FAIL-CLOSED. Falling back to the raw MODEL_CACHE_PVC here would let preflight vouch for one claim
    # while deploy.sh mounts another — and preflight is the gate that certifies the cache BEFORE a run, so
    # a silent degrade turns "verified" into a guess. BaseException, not Exception: install.py's import
    # guard calls sys.exit(), which raises SystemExit and is NOT an Exception subclass.
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        import install as _install  # noqa: PLC0415

        _claim, _src = _install.resolve_cache_claim(
            {
                "model": env.get("model"),
                "requires": env.get("requires") or {},
                "_path": cell,
            },
            prof,
        )
    except BaseException as e:
        sys.exit(
            f"preflight: cannot resolve this cell's model-cache claim ({type(e).__name__}: {e}). "
            f"Refusing to preflight against an unverified claim — deploy.sh resolves it the same way, "
            f"so a fallback here could certify a PVC the server never mounts."
        )
    # Require a cache claim only when the rendered manifests mount one.
    # This keeps non-serving lanes valid while cache-consuming cells fail closed.
    _mounts_cache = cell_mounts_model_cache(cell)
    if not _claim and _mounts_cache:
        sys.exit(
            f"preflight: no model-cache PVC resolved for model '{env.get('model') or '?'}' — "
            f"{_install.cache_claim_fix_hint({'model': env.get('model')}, profile)}"
        )
    if _claim:
        prof["MODEL_CACHE_PVC"] = _claim
    ns = prof.get("NAMESPACE", "")
    gpu_product = prof.get("GPU_PRODUCT", "")
    global _KUBE_CONTEXT, rc_fail
    _KUBE_CONTEXT = (prof.get("KUBE_CONTEXT") or "").strip()  # pin every kubectl below to the profile's cluster
    # Reset per-run state so calling main() twice in-process (test harness) gives correct counts.
    rc_fail = 0
    _counts["PASS"] = _counts["WARN"] = _counts["FAIL"] = 0

    print(f"\npreflight: {Path(cell).name}  ·  profile={profile}  ·  ns={ns or '?'}")

    section("offline")
    # profile completeness (OFFLINE — runs even if the cluster is unreachable): the manifests' scheduling/mount
    # vars must resolve, or the pod silently never schedules (empty schedulerName / no-match nodeSelector).
    mp = missing_profile_vars(cell, prof)
    line(
        "FAIL" if mp else "PASS",
        (
            f"profile '{profile}' is missing/empty deploy vars the manifests need: {', '.join(mp)} (→ silent no-schedule/bad mount)"
            if mp
            else "profile provides all scheduling/mount-critical vars the manifests reference"
        ),
        fix=f"set {', '.join(mp)} in cluster-profiles/{profile}.env",
    )

    # ${VAR} reconciliation (the linchpin — catches the NO_INTERNET_DNS_IP="" case that killed 48 trials):
    # the static CRITICAL_PROFILE_VARS check above only covers 5 vars, so a manifest that references a REAL
    # profile key left EMPTY sailed past (envsubst → '' at runtime, not a leftover ${VAR} dryrun would flag).
    # Reconcile THIS cell's EXACT referenced ${VAR}s against the profile — required-but-empty → FAIL. Shared
    # with `profile validate`/`init` and the CI template-coverage guard via manifest_vars, so they can't drift.
    try:
        import manifest_vars as _mv

        _vgaps = _mv.reconcile(Path(cell), prof)
        _vfail = [g for g in _vgaps if g.level == "FAIL"]
        _vwarn = [g for g in _vgaps if g.level == "WARN"]
        for _g in _vfail:
            line("FAIL", f"var-reconcile: {_g.message}", fix=_g.fix)
        for _g in _vwarn:
            line("WARN", f"var-reconcile: {_g.message}")
        if not _vgaps:
            line(
                "PASS",
                "every ${VAR} the manifests reference resolves to a non-empty profile/runtime value",
            )
    except Exception as _e:
        line(
            "WARN",
            f"var-reconcile skipped ({_e}) — profile-completeness check above still applies",
        )

    # target compatibility (OFFLINE — runs even if the cluster is unreachable): a recipe carries
    # hardware-specific serving flags, so refuse a wrong-target pairing (e.g. a GB200 recipe on a GB300
    # cluster — both arm64, so the live node-arch guard below can't catch it). See profile_resolver 0.11.
    import profile_resolver as _pr

    _compat = _pr.check_target_compat(env, prof)
    for _i in _compat:
        line(
            "FAIL",
            f"target: {_i}",
            fix=f"run this recipe on a {gpu_type or 'matching'}-target cluster: scripts/llmb-k8s profile list",
        )
    if not _compat:
        line(
            "PASS",
            f"recipe target {gpu_type or '?'} matches cluster GPU {gpu_product or '?'}",
        )

    section("cluster reachability")
    rc, ctx, _ = krun(["config", "current-context"])
    if rc != 0:
        print("  ❌ cluster unreachable (kubectl). Fix auth/context, then retry.")
        return 2
    if _KUBE_CONTEXT:
        line(
            "PASS",
            f"targeting pinned context: {_KUBE_CONTEXT}  (profile KUBE_CONTEXT; ambient is {ctx.strip()})",
        )
    else:
        line(
            "WARN",
            f"no KUBE_CONTEXT in profile — using the AMBIENT context {ctx.strip()}; in a multi-cluster "
            f'shell this can query the WRONG cluster. Set KUBE_CONTEXT="<ctx>" in '
            f"cluster-profiles/{profile}.env (kubectl config get-contexts)",
        )

    # Probe API reachability before resource checks so authentication or context failures are reported directly.
    # Retry transient control-plane errors before failing.
    # A malformed override (e.g. PREFLIGHT_PROBE_TRIES=lots) must not crash preflight with an uncaught
    # ValueError — fall back to the defaults with a WARN so the check still runs.
    try:
        tries = max(1, int(os.environ.get("PREFLIGHT_PROBE_TRIES", "4")))
    except ValueError:
        tries = 4
        line(
            "WARN",
            f'PREFLIGHT_PROBE_TRIES="{os.environ.get("PREFLIGHT_PROBE_TRIES")}" is not an int — '
            f"using default {tries}",
        )
    try:
        backoff = max(
            0.0, float(os.environ.get("PREFLIGHT_PROBE_BACKOFF_S", "5"))
        )  # clamp: a negative would crash time.sleep()
    except ValueError:
        backoff = 5.0
        line(
            "WARN",
            f'PREFLIGHT_PROBE_BACKOFF_S="{os.environ.get("PREFLIGHT_PROBE_BACKOFF_S")}" is not a '
            f"number — using default {backoff}",
        )
    rc, verr = 1, ""
    for attempt in range(1, tries + 1):
        rc, _, verr = krun(["get", "--raw", "/version"], timeout=15)
        if rc == 0:
            if attempt > 1:
                line(
                    "WARN",
                    f"apiserver answered on probe {attempt}/{tries} — earlier attempt(s) failed; "
                    f"momentary blip absorbed (a single-probe check would have hard-aborted here)",
                )
            break
        if attempt < tries:
            time.sleep(backoff * attempt)  # linear backoff: 5s, 10s, 15s between the 4 probes
    if rc != 0:
        which = f'pinned context "{_KUBE_CONTEXT}"' if _KUBE_CONTEXT else f'ambient context "{ctx.strip()}"'
        last = verr.strip().splitlines()[-1][:160] if verr.strip() else ""
        # Detect-and-guide: print the EXACT copy/paste SSO/Teleport login command (profile CONNECT_CMD, else a
        # derived `tsh kube login <ctx>`) instead of a generic "re-auth" note — an expired session is a
        # one-liner, not a scavenger hunt. Shared with profile_resolver so the two can't drift. (_pr already
        # imported in the offline section above.)
        _login = _pr.connect_hint(prof, _KUBE_CONTEXT or ctx.strip())
        line(
            "FAIL",
            f"{which} is unreachable after {tries} probes — apiserver did not answer (stopping before "
            f"resource checks so you don't chase phantom missing-resource failures)"
            + (f"; last error: {last}" if last else ""),
            fix=(
                f"log in, then retry:  {_login}   "
                f'(verify: kubectl --context "{_KUBE_CONTEXT or ctx.strip()}" get --raw /version)'
            ),
        )
        return 2

    section("GPU resources")
    rc, _, _ = krun(["get", "ns", ns])
    ns_ok = rc == 0
    line(
        "PASS" if ns_ok else "FAIL",
        f"namespace '{ns}' exists" if ns_ok else f"namespace '{ns}' NOT found",
        fix=f"scripts/llmb-k8s install {profile}   (auto-creates the namespace + secrets idempotently); "
        f"or kubectl create namespace {ns}",
    )
    if not ns_ok:
        # #12: without this, every in-namespace check below FAILs ("PVC not found", "secret MISSING", …) — a
        # noisy cascade that all reduces to the one FAIL above. Collapse it into a single deferred note; the
        # cluster-scoped checks (GPUs, storage class) still run and are still meaningful.
        line(
            "SKIP",
            f"in-namespace checks (PVC bind · secrets · model-weights) deferred — namespace '{ns}' "
            "doesn't exist yet; create it (or run with --managed-ns), then re-run preflight",
        )

    # --- GPU nodes: product + free count (DRA-aware) + taints ---
    want = (req.get("gpu") or {}).get("count", 0)
    # WHOLE-NODE POLICY: a cell declaring requires.gpu.whole_node must land on an EXCLUSIVE, otherwise-empty
    # node — "enough free GPUs" is NOT sufficient, because a node with free GPUs can still host other tenants
    # whose CPU/NIC/memory traffic corrupts our latency (a shared-node run measured +68-73% ITL vs the same
    # config on an exclusive node). The gate therefore becomes: some schedulable node has free == allocatable.
    whole_node = bool((req.get("gpu") or {}).get("whole_node"))
    if want:
        tolerated = set(req.get("tolerations") or [])
        want_arch = (env.get("arch") or "").strip()
        avail = query_gpu_availability(gpu_product, tolerated, want_arch)  # DRA-aware; re-callable in the poll
        if avail is None:
            line(
                "FAIL",
                f"no nodes labelled nvidia.com/gpu.product={gpu_product}",
                fix=f"use a cluster with {gpu_product} nodes, or fix GPU_PRODUCT in cluster-profiles/{profile}.env",
            )
        else:
            nodes, used = avail["nodes"], avail["used"]
            schedulable, node_archs = avail["schedulable"], avail["node_archs"]
            if avail["dra_gpus"]:
                line(
                    "WARN",
                    f"counted {avail['dra_gpus']} GPU(s) allocated via DRA ResourceClaims (device-plugin "
                    "request-sum alone would over-report free)",
                )
            # Arch guard: a digest-pinned image is single-platform; scheduling an arm64 image on an
            # amd64 node (or vice-versa) fails at run, not schedule. Flag product-node arch mismatch.
            if want_arch and node_archs and want_arch not in node_archs:
                line(
                    "FAIL",
                    f"arch mismatch: envelope.arch={want_arch} but {gpu_product} nodes are "
                    f"{sorted(node_archs)} (image platform won't run here)",
                    fix=f"deploy on a {want_arch} cluster (these {gpu_product} nodes are {sorted(node_archs)}): "
                    f"scripts/llmb-k8s profile list",
                )

            # 0.6 resource-aware preflight: show what's actually free (total/in-use/free/biggest-node) before
            # the pass/fail verdict, so an insufficient-GPU failure is a number the operator can act on.
            print(f"  📊 resources — {gpu_resource_summary(nodes, used, gpu_product)}")

            reclaim_fix = (
                f"free GPUs: scripts/llmb-k8s reclaim {profile}  ·  "
                f"see what's holding them: scripts/llmb-k8s jobs {profile} --all"
            )
            # Disaggregated serving = SEPARATE per-worker placements (e.g. 1P1D = 8 prefill + 8 decode), which
            # can land on DIFFERENT nodes. The rendered worker pod requests role.tp GPUs; role.dp is an engine
            # data-parallelism knob in this stack, not an additional Kubernetes GPU-placement multiplier.
            disagg = serving.get("disagg")
            if disagg:
                placements = []
                for role in ("prefill", "decode"):
                    r = disagg.get(role) or {}
                    if r:
                        placements.append((role, int(r.get("tp", 0))))
                assigned = place_disagg(schedulable, placements)  # greedy biggest-free-first
                # 0.6: --wait-on-resources polls until the placement fits (insufficient FREE GPUs only)
                if assigned is None and wait_on_resources:
                    try:
                        assigned = wait_for_gpus(
                            gpu_product,
                            tolerated,
                            want_arch,
                            lambda s: place_disagg(s, placements),
                        )
                    except KeyboardInterrupt:
                        print("  ⏹  stopped waiting for resources (Ctrl-C)")
                if assigned is not None:
                    line(
                        "PASS",
                        f"GPU (disagg): placed {', '.join(assigned)} across {len({a.split(':')[1] for a in assigned})} "
                        f"node(s) — {sum(n for _, n in placements)} {gpu_product} GPUs total",
                    )
                else:
                    line(
                        cap_level,
                        f"GPU: cannot place disagg roles {placements} on {gpu_product} nodes "
                        f"(free now: {[(n, f) for n, f in schedulable] or 'none'}){cap_note}",
                        fix=reclaim_fix,
                    )
            elif whole_node:
                # EXCLUSIVE-NODE gate: free == allocatable (nothing else running on it), not merely `want` free.
                node_alloc = {
                    n["metadata"]["name"]: int(
                        (n.get("status", {}).get("allocatable", {}) or {}).get("nvidia.com/gpu", 0) or 0
                    )
                    for n in nodes
                }

                def pick_free(candidates):
                    return next(
                        ([nm, fr] for nm, fr in candidates if node_alloc.get(nm, 0) and fr == node_alloc[nm]),
                        None,
                    )

                best = pick_free(schedulable)
                if best is None and wait_on_resources:
                    try:
                        best = wait_for_gpus(gpu_product, tolerated, want_arch, pick_free)
                    except KeyboardInterrupt:
                        print("  ⏹  stopped waiting for resources (Ctrl-C)")
                if best is not None:
                    line(
                        "PASS",
                        f"GPU (whole-node): node {best[0]} is FULLY FREE ({best[1]}/{node_alloc[best[0]]} "
                        f"{gpu_product} GPUs) — exclusive, no co-tenants",
                    )
                else:
                    _busy = [(nm, f"{fr}/{node_alloc.get(nm, '?')}") for nm, fr in schedulable]
                    line(
                        cap_level,
                        f"whole_node: no FULLY FREE {gpu_product} node (this cell demands an "
                        f"EXCLUSIVE node — a shared node corrupts its latency). "
                        f"free/allocatable now: {_busy or 'none'}{cap_note}",
                        fix=reclaim_fix + "   (or wait for a node to drain: re-run with --wait-on-resources)",
                    )
                # WHOLE-NODE CPU/MEMORY: the rendered manifest substitutes ${WHOLE_NODE_CPU}/${WHOLE_NODE_MEM}
                # (requests == limits → Guaranteed QoS). If the profile lacks them, envsubst yields an EMPTY
                # value and the Deployment is rejected (or, worse, a too-large value leaves the pod Pending
                # forever). Fail LOUDLY here, naming the fix, rather than leaving a mysterious Pending pod.
                _wn_cpu = (prof.get("WHOLE_NODE_CPU") or "").strip()
                _wn_mem = (prof.get("WHOLE_NODE_MEM") or "").strip()
                _missing = [
                    k
                    for k, v in (
                        ("WHOLE_NODE_CPU", _wn_cpu),
                        ("WHOLE_NODE_MEM", _wn_mem),
                    )
                    if not v
                ]
                if _missing:
                    line(
                        "FAIL",
                        f"whole_node: {' and '.join(_missing)} unset in the cluster profile — this cell "
                        f"scales CPU/memory to the node, so the manifest cannot be substituted",
                        fix=f"re-run `llmb-k8s init {profile}` to auto-detect them, or set "
                        f"{'/'.join(_missing)} by hand in cluster-profiles/{profile}.env",
                    )
                else:
                    # The request must FIT — and "fit" means FREE on the node we actually picked, not merely
                    # <= allocatable. Allocatable is a static ceiling; the scheduler subtracts what is
                    # already REQUESTED there (DaemonSets, plus any co-tenant that landed on a GPU-free
                    # node). Checking allocatable alone is an absence-as-success bug: a request under the
                    # ceiling but over the free pool passes every check, then sits Pending forever while
                    # preflight reports all-green.
                    _cand = best[0] if best else None
                    _alloc_cpu = _alloc_mem = None
                    for n in nodes:
                        _nm = n["metadata"]["name"]
                        if _nm == _cand or (_cand is None and node_alloc.get(_nm, 0)):
                            _a = n.get("status", {}).get("allocatable", {}) or {}
                            _alloc_cpu, _alloc_mem = _cpu_millicores(_a.get("cpu")), _mem_gib(_a.get("memory"))
                            _cand = _nm
                            break
                    _req_cpu, _req_mem = _cpu_millicores(_wn_cpu), _mem_gib(_wn_mem)
                    # What is already requested on that node. UNKNOWN (None) when pods are unreadable —
                    # never silently treated as "nothing is running there".
                    _used_cpu = _used_mem = None
                    _holders = []
                    _pods = avail.get("pods")
                    if _pods is not None and _cand:
                        _used_cpu = _used_mem = 0
                        for _p in _pods:
                            if (_p.get("spec", {}) or {}).get("nodeName") != _cand:
                                continue
                            if (_p.get("status", {}) or {}).get("phase") not in (
                                "Running",
                                "Pending",
                            ):
                                continue
                            # _cpu_millicores/_mem_gib return None for an absent request (a BestEffort
                            # container legitimately has none) — coerce to 0 rather than poisoning the sum.
                            _pc = sum(
                                (
                                    _cpu_millicores(
                                        ((c.get("resources", {}) or {}).get("requests", {}) or {}).get("cpu")
                                    )
                                    or 0
                                )
                                for c in (_p["spec"].get("containers") or [])
                            )
                            _pm = sum(
                                (
                                    _mem_gib(((c.get("resources", {}) or {}).get("requests", {}) or {}).get("memory"))
                                    or 0
                                )
                                for c in (_p["spec"].get("containers") or [])
                            )
                            _used_cpu += _pc
                            _used_mem += _pm
                            if _pc >= 1000:  # >=1 core: a real co-tenant, worth naming in the failure
                                _holders.append(
                                    f"{_p['metadata']['namespace']}/{_p['metadata']['name']} " f"({_pc / 1000:g} cpu)"
                                )
                    _free_cpu = (_alloc_cpu - _used_cpu) if (_alloc_cpu and _used_cpu is not None) else None
                    _free_mem = (_alloc_mem - _used_mem) if (_alloc_mem and _used_mem is not None) else None
                    _reinit = (
                        f"re-run `llmb-k8s init {profile}` to re-derive with DaemonSet headroom, or "
                        f"lower it in cluster-profiles/{profile}.env"
                    )
                    # CO-SCHEDULING BUDGET. Only for a cell whose bench pod is REQUIRED onto the server's
                    # node; otherwise the two never compete and this must stay silent.
                    _cosched = bench_coschedule_demand(cell, prof)
                    _cosched_fail = _coschedule_verdict(
                        _wn_cpu,
                        _wn_mem,
                        _req_cpu,
                        _req_mem,
                        _cosched,
                        _alloc_cpu,
                        _alloc_mem,
                        _free_cpu,
                        _free_mem,
                        _cand,
                        profile,
                    )
                    if _alloc_cpu and _req_cpu and _req_cpu > _alloc_cpu:
                        line(
                            "FAIL",
                            f"whole_node: WHOLE_NODE_CPU={_wn_cpu} exceeds node allocatable "
                            f"({_alloc_cpu}m) — the pod would stay Pending forever",
                            fix=_reinit,
                        )
                    elif _alloc_mem and _req_mem and _req_mem > _alloc_mem:
                        line(
                            "FAIL",
                            f"whole_node: WHOLE_NODE_MEM={_wn_mem} exceeds node allocatable "
                            f"({_alloc_mem}Gi) — the pod would stay Pending forever",
                            fix=_reinit,
                        )
                    elif _free_cpu is not None and _req_cpu and _req_cpu > _free_cpu:
                        line(
                            "FAIL",
                            f"whole_node: WHOLE_NODE_CPU={_wn_cpu} ({_req_cpu}m) exceeds FREE cpu on "
                            f"{_cand} ({_free_cpu}m = {_alloc_cpu}m allocatable − {_used_cpu}m already "
                            f"requested) — the pod will sit Pending until that frees"
                            + (f"; held by {', '.join(_holders[:3])}" if _holders else ""),
                            fix="wait for those pods to finish, or " + _reinit,
                        )
                    elif _free_mem is not None and _req_mem and _req_mem > _free_mem:
                        line(
                            "FAIL",
                            f"whole_node: WHOLE_NODE_MEM={_wn_mem} exceeds FREE memory on {_cand} "
                            f"({_free_mem}Gi = {_alloc_mem}Gi allocatable − {_used_mem}Gi already "
                            f"requested) — the pod will sit Pending until that frees",
                            fix="wait for those pods to finish, or " + _reinit,
                        )
                    elif _cosched.get("parse_error"):
                        line(
                            "WARN",
                            f"whole_node co-scheduling: could not read the bench manifest "
                            f"({_cosched['parse_error']}) — cannot confirm the bench pod fits "
                            f"beside the server's reservation",
                        )
                    elif _cosched and coschedule_unreadable(_cosched):
                        # ANY unreadable request, not both (see coschedule_unreadable). A required same-host
                        # affinity plus an un-computable fit must say so rather than fall through to a PASS.
                        _unread = ", ".join(coschedule_unreadable(_cosched))
                        line(
                            "WARN",
                            f"whole_node co-scheduling: this cell pins its bench pod to the SERVER'S "
                            f"node, but its {_unread} request is unreadable "
                            f"(cpu={_cosched.get('cpu_raw') or '?'}, mem={_cosched.get('mem_raw') or '?'})"
                            f" — cannot verify both fit on one node. This is 'not checked', not "
                            f"'checked and fine'.",
                            fix=f"set BENCH_CPU_REQUEST in cluster-profiles/{profile}.env so the bench pod's "
                            f"demand on the server's node is knowable before the run starts",
                        )
                    elif _cosched and _cosched_fail:
                        line("FAIL", _cosched_fail[0], fix=_cosched_fail[1])
                    elif _free_cpu is None:
                        line(
                            "WARN",
                            f"whole_node resources: cpu={_wn_cpu} mem={_wn_mem} fit node allocatable, "
                            f"but the pod list was unreadable so FREE capacity is UNKNOWN — this is "
                            f"not a clean bill of health; the pod may still stay Pending",
                        )
                    else:
                        _hr = (
                            f" ({100 * _req_cpu // _alloc_cpu}% of allocatable cpu)" if _alloc_cpu and _req_cpu else ""
                        )
                        _co = (
                            f"; {len(_holders)} co-tenant(s) already on {_cand}: {', '.join(_holders[:3])}"
                            if _holders
                            else ""
                        )
                        _cos = ""
                        if _cosched:
                            _cos = (
                                f"; bench co-schedules on the same node and fits too "
                                f"({_wn_cpu}+{_cosched.get('cpu_raw')} cpu, "
                                f"{_wn_mem}+{_cosched.get('mem_raw')} mem)"
                            )
                        line(
                            "PASS",
                            f"whole_node resources: requests==limits cpu={_wn_cpu} mem={_wn_mem}"
                            f"{_hr}, fits FREE cpu {_free_cpu}m on {_cand} → Guaranteed QoS{_co}{_cos}",
                        )
            else:
                best = pick_single_node(schedulable, want)
                # 0.6: --wait-on-resources polls until a node has `want` free (insufficient FREE GPUs only)
                if best is None and wait_on_resources:
                    try:
                        best = wait_for_gpus(
                            gpu_product,
                            tolerated,
                            want_arch,
                            lambda s: pick_single_node(s, want),
                        )
                    except KeyboardInterrupt:
                        print("  ⏹  stopped waiting for resources (Ctrl-C)")
                if best is not None:
                    line(
                        "PASS",
                        f"GPU: node {best[0]} has {best[1]}/{want} free {gpu_product} GPUs (taints tolerated)",
                    )
                else:
                    line(
                        cap_level,
                        f"no schedulable {gpu_product} node with {want} free GPUs, arch={want_arch or 'any'}, "
                        f"tolerating {sorted(tolerated) or 'no taints'} "
                        f"(DRA note: device-plugin count can over-report — a run may still OOM)"
                        f"{cap_note}",
                        fix=reclaim_fix + "   (or wait: re-run preflight with --wait-on-resources)",
                    )
        if (req.get("gpu") or {}).get("min_memory_gib"):
            line(
                "WARN",
                f"gpu.min_memory_gib={req['gpu']['min_memory_gib']} not statically checkable "
                "(needs DCGM/pod); deploy fails fast if a device has less free memory",
            )

    # --- disagg RDMA capability (PROACTIVE FAILURE) ---
    if is_disagg(serving):
        section("RDMA fabric")
    # A disaggregated recipe transfers KV cache over RDMA/InfiniBand; its worker pods nodeSelect on RDMA
    # labels. If NO node carries those labels, the pods never schedule — or the transport silently degrades
    # and the run fails 45 min in with a cryptic NIXL error. Catch it here with the exact fix.
    if is_disagg(serving):
        # Prefer the cluster profile selector; fall back to portable rendered defaults.
        prof_sels = profile_rdma_selectors(prof)
        sels = prof_sels or disagg_rdma_selectors(cell)
        src = "profile RDMA_NODE_SELECTOR" if prof_sels else "recipe (baked in rendered manifests)"
        if not sels:
            line(
                "WARN",
                "disagg recipe but no RDMA nodeSelector — profile RDMA_NODE_SELECTOR unset AND none in "
                "the rendered manifests; cannot verify RDMA. Set RDMA_NODE_SELECTOR in the profile "
                '(e.g. "feature.node.kubernetes.io/rdma.available=true").',
            )
        else:
            sel_str = ",".join(f"{k}={v}" for k, v in sels)
            rc, out, _ = krun(["get", "nodes", "-l", sel_str, "-o", "name"])
            n = len([x for x in (out or "").splitlines() if x.strip()])
            if rc != 0:
                line(
                    "WARN",
                    f"could not query RDMA nodes (selector {sel_str}) — kubectl error; verify manually",
                )
            elif n == 0:
                line(
                    "FAIL",
                    f"disagg needs RDMA/InfiniBand nodes, but NO node matches the selector ({sel_str}, from {src})",
                    fix=(
                        "list what your nodes actually advertise:  kubectl get nodes --show-labels | "
                        "grep -iE 'rdma|infiniband'  —  then set RDMA_NODE_SELECTOR in the profile to the "
                        "label(s) this cluster uses, or, if the cluster has no "
                        "RDMA fabric, disagg cannot run here — pick an aggregated (non-disagg) recipe"
                    ),
                )
            else:
                line(
                    "PASS",
                    f"{n} node(s) carry the disagg RDMA labels ({sel_str}, from {src})",
                )

    # --- disagg UCX device config (ADVISORY) ---
    # RDMA_UCX_NET_DEVICES defaults to "all" (UCX auto-select), which may pick the wrong
    # interface or silently miss multi-rail IB. probe-fabric discovers the right list from
    # live node capacity (rdma/* extended resources) and writes it to the cluster profile.
    if is_disagg(serving):
        net_devs = (prof.get("RDMA_UCX_NET_DEVICES") or "").strip()
        if not net_devs or net_devs == "all":
            line(
                "WARN",
                "RDMA_UCX_NET_DEVICES is unset or 'all' — UCX will auto-select the transport "
                "device, which may pick the wrong interface or miss multi-rail IB on this cluster. "
                f"Run: llmb-k8s profile probe-fabric --cluster {profile} --write",
            )
        else:
            # DC transports are InfiniBand-only and cannot initialize on an Ethernet bond.
            # Reject that incompatible profile combination before launching workers.
            _devs = [d.strip() for d in net_devs.split(",") if d.strip()]
            _bonds = sorted({d.split(":")[0] for d in _devs if "bond" in d.split(":")[0].lower()})
            _tls_now = (prof.get("RDMA_UCX_TLS") or "rc,rc_x,dc,dc_x,cuda_copy").strip()  # run.sh's default
            _dc_now = sorted({t.strip() for t in _tls_now.split(",")} & {"dc", "dc_x"})
            if _bonds and _dc_now:
                line(
                    "FAIL",
                    f"RDMA_UCX_NET_DEVICES lists {','.join(_bonds)} — an Ethernet/RoCE bond — while "
                    f"RDMA_UCX_TLS includes {','.join(_dc_now)}, which is InfiniBand-only. DC cannot "
                    "initialize on the bond, so the disaggregated scheduler cannot start",
                    fix=(
                        f"drop {','.join(_bonds)} from RDMA_UCX_NET_DEVICES (keep the IB rails only) — or, "
                        'if this really is a RoCE fabric, set RDMA_UCX_TLS="rc,rc_x,cuda_copy" and '
                        'RDMA_UCX_IB_ADDR_TYPE="eth"'
                    ),
                )
            else:
                line("PASS", f"RDMA_UCX_NET_DEVICES={net_devs}")

    # Cross-check the profile transport set against its declared fabric type.
    # DC transports are valid for InfiniBand, not RoCE/Ethernet.
    if is_disagg(serving):
        tls = (prof.get("RDMA_UCX_TLS") or "").strip()
        addr = (prof.get("RDMA_UCX_IB_ADDR_TYPE") or "").strip().lower()
        dc = sorted({t.strip() for t in tls.split(",")} & {"dc", "dc_x"})
        if addr == "eth" and (dc or not tls):
            line(
                "FAIL",
                "RDMA_UCX_IB_ADDR_TYPE=eth says this is a RoCE/Ethernet fabric, but RDMA_UCX_TLS "
                + (f"includes {','.join(dc)}" if dc else "is unset (run.sh defaults to the InfiniBand list)")
                + " — DC is an InfiniBand-only transport that cannot initialise on a RoCE device, so UCX "
                "createBackend fails and the disagg scheduler dies at init (NIXL_ERR_BACKEND, exit -3)",
                fix='set RDMA_UCX_TLS="rc,rc_x,cuda_copy" in the cluster profile (RoCE-safe: no dc/dc_x)',
            )
        elif not tls:
            line(
                "WARN",
                "RDMA_UCX_TLS is unset — run.sh falls back to the InfiniBand transport list "
                "(rc,rc_x,dc,dc_x,cuda_copy), which is correct ONLY on native InfiniBand. Confirm the "
                "link layer (`ibv_devinfo | grep link_layer`) and set it explicitly in the profile.",
                fix='RDMA_UCX_TLS="rc,rc_x,dc,dc_x,cuda_copy" (InfiniBand) or "rc,rc_x,cuda_copy" (RoCE)',
            )
        else:
            line("PASS", f"RDMA_UCX_TLS={tls}")

    section("storage & secrets")
    # --- storage PVCs bound --- (in-namespace: skip the whole loop if the namespace is absent — #12)
    for var in (req.get("storage") or []) if ns_ok else []:
        pvc = prof.get(var, "")
        rc, pvc_out, _ = krun(["-n", ns, "get", "pvc", pvc, "-o", "json"])
        try:
            pvc_obj = json.loads(pvc_out) if rc == 0 and pvc_out else {}
        except Exception:
            pvc_obj = {}
        phase = str((pvc_obj.get("status") or {}).get("phase") or "")
        access_modes = list((pvc_obj.get("spec") or {}).get("accessModes") or [])
        line(
            "PASS" if phase == "Bound" else "FAIL",
            f"PVC {var}={pvc or '?'}: {phase or 'not found'}",
            fix=f"kubectl -n {ns} describe pvc {pvc}   (check the PVC exists + its StorageClass provisioner; "
            f"or fix {var} in cluster-profiles/{profile}.env)",
        )
        if var == "MODEL_CACHE_PVC" and phase == "Bound":
            cache_issue = disagg_cache_access_issue(serving, access_modes)
            if cache_issue:
                line(
                    "FAIL",
                    f"model cache access: {cache_issue}",
                    fix=f"point MODEL_CACHE_PVC (or its per-model override) in cluster-profiles/{profile}.env "
                    "at a populated ReadWriteMany/ReadOnlyMany claim, or provision an RWX cache with "
                    "MODEL_CACHE_RWX_CLASS and run install before retrying",
                )
            elif is_disagg(serving):
                line(
                    "PASS",
                    f"model cache access supports multi-node disagg: {','.join(access_modes) or '?'}",
                )

    # --- artifacts storage class exists (B9 fix) ---
    # sweep.sh creates a per-cell RWO artifacts PVC using ARTIFACTS_STORAGE_CLASS. A missing or
    # misspelled class causes the PVC to stay Pending indefinitely mid-sweep with no surfaced error.
    # StorageClass is cluster-scoped (no -n flag).
    artifacts_sc = prof.get("ARTIFACTS_STORAGE_CLASS", "")
    if artifacts_sc:
        # If the per-cell artifacts PVC already exists and is Bound, the storage class is MOOT — sweep.sh reuses
        # the existing PVC (it only creates one when absent). Check that first: it makes preflight pass on a
        # ready cluster and sidesteps a false "NOT found" when RBAC forbids listing cluster-scoped storageclasses
        # Existing PVCs may use a shared class while cluster-scoped `get storageclass` is forbidden, which
        # was blocking --skip-stage/--skip-server runs against an already-provisioned PVC).
        cell_name = (env.get("name") or "").strip()
        art_pvc = f"{cell_name}-artifacts" if cell_name else ""
        art_phase = ""
        if art_pvc and ns:
            _, art_phase, _ = krun(["-n", ns, "get", "pvc", art_pvc, "-o", "jsonpath={.status.phase}"])
        # Only probe the (cluster-scoped) storage class when we still need to — an already-Bound PVC settles it.
        sc_rc, sc_err = 1, ""
        if art_phase != "Bound":
            sc_rc, _, sc_err = krun(["get", "storageclass", artifacts_sc])
        _lvl, reason = classify_artifacts_sc(art_phase, sc_rc, sc_err)
        if reason == "moot-bound":
            line(
                "PASS",
                f"ARTIFACTS_STORAGE_CLASS={artifacts_sc}: moot — artifacts PVC {art_pvc} already Bound (reused)",
            )
        elif reason == "exists":
            line("PASS", f"ARTIFACTS_STORAGE_CLASS={artifacts_sc}: exists")
        elif reason == "rbac-forbidden":
            # Can't read cluster-scoped storageclasses (RBAC). Don't block a namespaced workflow over an authz
            # gap — if the class is actually wrong, the artifacts PVC surfaces a visible Pending mid-sweep.
            line(
                "WARN",
                f"ARTIFACTS_STORAGE_CLASS={artifacts_sc}: cannot verify (RBAC forbids listing "
                f"cluster-scoped storageclasses) — assuming valid",
                fix="ask a cluster admin to confirm the class, or pre-create the artifacts PVC in the namespace",
            )
        else:
            line(
                "FAIL",
                f"ARTIFACTS_STORAGE_CLASS={artifacts_sc}: NOT found — artifacts PVC will pend indefinitely",
                fix="kubectl get storageclass  # then set an existing RWO class in the profile",
            )
    else:
        line(
            "WARN",
            "ARTIFACTS_STORAGE_CLASS unset — sweep.sh will omit storageClassName and rely on the "
            "cluster default; set explicitly in the profile if the default is not RWO-compatible",
        )

    # Model-cache integrity is evaluated by the recipe-scoped capability gate below.
    rev = serving.get("model_revision")
    if not rev:
        line(
            "SKIP",
            "model_revision unset — cache-integrity check skipped (check_invariants warns to pin it)",
        )

    # --- secrets present AND usable (empty token → 401 minutes into startup) --- (in-namespace: skip if absent, #12)
    rendered_refs = rendered_profile_refs(cell)
    for var in (req.get("secrets") or []) if ns_ok else []:
        if var not in rendered_refs:
            sec = prof.get(var, "")
            line(
                "SKIP",
                f"secret {var}={sec or '?'} not referenced by this cell's rendered manifests",
            )
            continue
        sec = prof.get(var, "")
        rc, out, _ = krun(["-n", ns, "get", "secret", sec, "-o", "json"]) if sec else (1, "", "")
        if rc != 0:
            line(
                "FAIL",
                f"secret {var}={sec or '?'}: MISSING",
                fix=f"kubectl -n {ns} create secret generic {sec or '<name>'} --from-literal=token=<VALUE>   "
                f"(or point {var} at an existing secret in cluster-profiles/{profile}.env)",
            )
            continue
        try:
            issue = secret_content_issue(json.loads(out))
        except Exception:
            issue = None
        line(
            "FAIL" if issue else "PASS",
            (f"secret {var}={sec}: {issue}" if issue else f"secret {var}={sec}: present + non-empty"),
            fix=f"recreate with a non-empty value: kubectl -n {ns} delete secret {sec} && "
            f"kubectl -n {ns} create secret generic {sec} --from-literal=token=<VALUE>",
        )

    section("operators & network")
    # --- operators reconciling ---
    for op in req.get("operators") or []:
        rc, dout, _ = krun(
            [
                "get",
                "deploy",
                "-A",
                "-o",
                "jsonpath={range .items[*]}{.metadata.name}{'\\n'}{end}",
            ]
        )
        found = any(op.split("-")[0] in d for d in dout.splitlines()) if rc == 0 else False
        line(
            "PASS" if found else "FAIL",
            f"operator '{op}': {'present' if found else 'NOT found'}",
            fix=f"install the '{op}' operator on this cluster (cluster-admin), or use a cluster that has it",
        )

    # --- NetworkPolicy support ---
    if req.get("network_policy"):
        rc, api, _ = krun(["api-resources", "--api-group=networking.k8s.io", "--no-headers"])
        ok = "networkpolicies" in api
        line(
            "PASS" if ok else "FAIL",
            f"NetworkPolicy API {'available' if ok else 'NOT available'}",
            fix="install a NetworkPolicy-capable CNI (e.g. Calico/Cilium), or use a cluster that has one",
        )

    # --- DCGM ---
    dcgm = req.get("dcgm", "none")
    if dcgm in ("required", "optional"):
        rc, sout, _ = krun(["get", "svc", "-A", "--no-headers"])
        ok = "dcgm" in sout.lower()
        line(
            "PASS" if ok else ("FAIL" if dcgm == "required" else "WARN"),
            f"DCGM exporter {'reachable' if ok else 'not found'} ({dcgm}) — needed for GPU telemetry",
            fix="deploy the DCGM exporter (GPU Operator's dcgm-exporter), or use a cluster with GPU telemetry",
        )

    # Apply recipe-scoped capability checks not covered by the dedicated sections above.
    # Probe errors remain advisory; confirmed incompatibilities fail before resources are launched.
    try:
        import capability_registry as _cap

        _reg_ids = {
            "nvlink-imex",
            "no-internet-ips",
            "nvlink-p2p",
            "model-cache-integrity",
            "image-arch",
            "image-pull-access",
        }
        _reg = tuple(e for e in _cap.REGISTRY if e.id in _reg_ids and e.requires(recipe))
        if _reg:
            section("cluster capabilities")
            _facts = _cap.gather_facts(prof, krun, recipe=recipe)
            # Mount the model-cache PVC and verify the snapshot path used by the server.
            if ns_ok and any(e.id == "model-cache-integrity" for e in _reg):
                _facts["model_cache_integrity"] = probe_model_cache_integrity(ns, prof, recipe)
            # Check that the namespace pull secret can access every pinned image before reserving GPUs.
            if ns_ok and any(e.id == "image-pull-access" for e in _reg):
                _facts["image_pull_access"] = probe_image_pull_access(ns, prof, recipe)
            # nvlink-p2p (Gap 2): the profile fact is INIT-TIME cached — a TP>1 cell reads a possibly-stale
            # value (a fabric that degraded post-init, or an init-time `unknown`, would sail past). Actively
            # re-probe live now unless a fresh live result is already in hand (NVLINK_P2P_FRESH). Prefer a
            # DEFINITIVE live verdict; on any probe error keep the cached fact (safe-degrade, never a false
            # block). Keeps the registry's scenario-aware severity (FAIL perf/goodput, WARN net-score).
            # --stage-only: SKIP the live probe entirely — it schedules a real GPU pod, and a "prepare
            # this cell for a later run" install must not hold GPUs it doesn't need. The cached fact is
            # still evaluated, and the live re-probe happens at run time when GPUs are legitimately ours.
            _p2p_required = any(e.id == "nvlink-p2p" for e in _reg)
            _p2p_fresh = str(prof.get("NVLINK_P2P_FRESH") or "").strip().lower() == "true"
            if stage_only and _p2p_required and not _p2p_fresh:
                print("  · nvlink-p2p live probe SKIPPED (--stage-only: would hold GPUs; re-probed at run time)")
            if ns_ok and not stage_only and should_probe_p2p_live(_p2p_required, _p2p_fresh):
                _live = live_probe_p2p(ns, prof, first_gpu_node(gpu_product))
                _facts["nvlink_p2p"] = merge_p2p_fact(_facts.get("nvlink_p2p") or {}, _live)
                # Make an opaque `unknown` diagnostic: if the live probe couldn't run, say WHY (pool
                # saturated / ImagePull / admission) so a human knows the fabric wasn't actually verified.
                if _live and str(_live.get("state", "")).lower() == "unknown" and _live.get("error"):
                    print(f"  · nvlink-p2p live probe inconclusive — {_live['error']}")
            for _g in _cap.evaluate(recipe, _facts, registry=_reg):
                # This section only SURFACES the gap (WARN, recoverable). The actual crash-killer is the
                # apply-time strip in the deploy.sh pipeline: scripts/merge_imex_strip.py removes the forced
                # FLASHINFER env when NVLINK_MULTICAST_IMEX is not provisioned, so a forced-fusion recipe on
                # an IMEX-less cluster runs correct-but-slower instead of CrashLooping on cuMulticastCreate
                # code=800. no-internet-ips FAILs only when the IPs weren't auto-written (profile
                # init/validate normally resolves them).
                line(_g.level, _g.message, fix=_g.fix)  # line() prints the fix hint only on FAIL
    except Exception as _e:  # never let the registry break a preflight
        print(f"  · cluster-capability registry skipped ({_e})")

    p, w, f = _counts["PASS"], _counts["WARN"], _counts["FAIL"]
    summary = f"  {p} passed  {w} warned  {f} failed"
    verdict = "✅ all hard checks passed" if rc_fail == 0 else "❌ one or more checks failed"
    print(f"\npreflight: {verdict}  ·  {summary.strip()}")
    return rc_fail


if __name__ == "__main__":
    raise SystemExit(main())
