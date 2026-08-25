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

"""selftest_cluster_readiness.py — unit tests for the `profile validate`/`init` cluster readiness battery.

Covers cluster_readiness.py's PURE classifiers (no cluster/network) plus its IMPURE probes driven by a FAKE
krun (canned kubectl JSON) and a FAKE http (canned registry responses). The headline invariants:
  - safe-degrade: only a DEFINITIVE gap (StorageClass ProvisioningFailed, `kubectl cp` failing every retry,
    an invalid pull credential, a real arch mismatch) is FAIL; everything unverifiable is WARN/SKIP and
    never flips verdict() to not-ready.
  - throwaway-resource cleanup: the staging round-trip ALWAYS deletes its pod+PVC (even on failure) and leaves
    the cleanup tracker empty — a probe can't leak a resource.
  - --fast bypasses every live probe.

Pure/offline. Run `python3 scripts/selftest_cluster_readiness.py` or via `make test`. Exit 0 = all pass.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent


def _load(name):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


cr = _load("cluster_readiness")

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


# ── FAKES ────────────────────────────────────────────────────────────────────
class FakeKrun:
    """Records every kubectl invocation; returns a canned (rc, out, err) by matching a substring of the args.
    Rules are (needle_tuple, (rc, out, err)); first match wins. Accepts a `stdin=` kwarg (for apply -f -).
    """

    def __init__(self, rules):
        self.rules = rules
        self.calls = []

    def __call__(self, args, timeout=30, stdin=None):
        self.calls.append(list(args))
        for needles, resp in self.rules:
            if all(any(n == a or n in a for a in args) for n in needles):
                return resp
        return (0, "", "")

    def deletes(self):
        return [c for c in self.calls if "delete" in c]


_NODES_OK = json.dumps(
    {
        "items": [
            {
                "metadata": {
                    "labels": {
                        "nvidia.com/gpu.product": "NVIDIA-GB300",
                        "kubernetes.io/arch": "arm64",
                    }
                }
            },
            {
                "metadata": {
                    "labels": {
                        "nvidia.com/gpu.product": "NVIDIA-GB300",
                        "kubernetes.io/arch": "arm64",
                    }
                }
            },
        ]
    }
)

# A JWT with iss=https://oidc.eks.us-east-2.amazonaws.com/id/ABC (payload only; signature irrelevant here).
import base64 as _b64


def _jwt(iss):
    payload = _b64.urlsafe_b64encode(json.dumps({"iss": iss, "sub": "x"}).encode()).decode().rstrip("=")
    return f"aGVhZGVy.{payload}.c2ln"


_DOCKERCFG = _b64.b64encode(
    json.dumps({"auths": {"nvcr.io": {"username": "$oauthtoken", "password": "KEY"}}}).encode()
).decode()


# ── 1. classify_nodes ────────────────────────────────────────────────────────
c = cr.classify_nodes(json.loads(_NODES_OK)["items"], "NVIDIA-GB300", "arm64")
check(
    "classify_nodes: matching product+arch → PASS",
    c.level == cr.PASS and "2 ×" in c.message,
    c.message,
)
c = cr.classify_nodes([], "NVIDIA-GB300", "arm64")
check(
    "classify_nodes: zero matching nodes → WARN (autoscale-to-zero is legit, not a hard fail)",
    c.level == cr.WARN,
)
c = cr.classify_nodes(json.loads(_NODES_OK)["items"], "NVIDIA-GB300", "amd64")
check(
    "classify_nodes: arch mismatch on the target product → FAIL (exec format error)",
    c.level == cr.FAIL,
)

# ── 2. classify_staging ──────────────────────────────────────────────────────
check(
    "classify_staging: full round-trip ok → PASS",
    cr.classify_staging(
        {
            "sc": "ebs",
            "bound": True,
            "pod_ready": True,
            "cp_ok": True,
            "readback_ok": True,
        }
    ).level
    == cr.PASS,
)
check(
    "classify_staging: ProvisioningFailed → FAIL",
    cr.classify_staging({"sc": "bad", "provisioning_failed": True}).level == cr.FAIL,
)
check(
    "classify_staging: slow bind (no error) → WARN (unverified, not blocking)",
    cr.classify_staging({"sc": "slow", "bound": False}).level == cr.WARN,
)
check(
    "classify_staging: bound but pod not ready → WARN (mount+cp unverified)",
    cr.classify_staging({"sc": "ebs", "bound": True, "pod_ready": False}).level == cr.WARN,
)
check(
    "classify_staging: cp fails after retries → FAIL (the Teleport stall we catch pre-GPU)",
    cr.classify_staging({"sc": "ebs", "bound": True, "pod_ready": True, "cp_ok": False}).level == cr.FAIL,
)
check(
    "classify_staging: unset SC → SKIP",
    cr.classify_staging({"skipped": True, "reason": "unset"}).level == cr.SKIP,
)

# ── 4. classify_pvc_bind (control RWX) ───────────────────────────────────────
check(
    "classify_pvc_bind: RWX binds → PASS",
    cr.classify_pvc_bind({"sc": "efs", "bound": True}).level == cr.PASS,
)
check(
    "classify_pvc_bind: RWX ProvisioningFailed → FAIL",
    cr.classify_pvc_bind({"sc": "rwo-only", "provisioning_failed": True}).level == cr.FAIL,
)
check(
    "classify_pvc_bind: unset → SKIP",
    cr.classify_pvc_bind({"skipped": True}).level == cr.SKIP,
)

# ── 4b. QA fail-fast (#3): object-store provisioner → immediate FAIL (no Pending-forever hang) ──
_ff = cr.classify_unprovisionable_sc("staging-roundtrip", "s3-object", "s3.csi.aws.com")
check(
    "classify_unprovisionable_sc: s3.csi → FAIL naming class + provisioner + fix",
    _ff is not None
    and _ff.level == cr.FAIL
    and "s3-object" in _ff.message
    and "s3.csi.aws.com" in _ff.message
    and bool(_ff.fix)
    and ("ebs" in _ff.fix or "block" in _ff.fix),
    getattr(_ff, "message", None),
)
check(
    "classify_unprovisionable_sc: block provisioner (ebs.csi) → None (proceed to live probe)",
    cr.classify_unprovisionable_sc("staging-roundtrip", "ebs", "ebs.csi.aws.com") is None,
)
check(
    "classify_unprovisionable_sc: file provisioner (fsx) → None",
    cr.classify_unprovisionable_sc("control-rwx", "fsx-lustre", "fsx.csi.aws.com") is None,
)
check(
    "classify_unprovisionable_sc: unknown/empty provisioner → None (safe-degrade)",
    cr.classify_unprovisionable_sc("staging-roundtrip", "sc", "") is None,
)

# ── 5. classify_pull_secret ──────────────────────────────────────────────────
check(
    "classify_pull_secret: --recipe per-image 403 → FAIL (org gap)",
    cr.classify_pull_secret(
        "s",
        True,
        True,
        ["nvcr.io"],
        {},
        image_results={"nvcr.io/nvidian/serving-driver@sha256:a": {"status": "forbidden"}},
    ).level
    == cr.FAIL,
)
check(
    "classify_pull_secret: --recipe all pass → PASS",
    cr.classify_pull_secret(
        "s",
        True,
        True,
        ["nvcr.io"],
        {},
        image_results={"nvcr.io/x@sha256:a": {"status": "pass"}},
    ).level
    == cr.PASS,
)
check(
    "classify_pull_secret: cluster auth ok → PASS",
    cr.classify_pull_secret("s", True, True, ["nvcr.io"], {"nvcr.io": cr._AUTH_OK}).level == cr.PASS,
)
check(
    "classify_pull_secret: invalid credential (registry refuses token) → FAIL",
    cr.classify_pull_secret("s", True, True, ["nvcr.io"], {"nvcr.io": cr._AUTH_FORBIDDEN}).level == cr.FAIL,
)
check(
    "classify_pull_secret: unreadable secret → WARN (safe-degrade, preflight also checks presence)",
    cr.classify_pull_secret("s", False, False, [], {}).level == cr.WARN,
)
check(
    "classify_pull_secret: registry unreachable → WARN",
    cr.classify_pull_secret("s", True, True, ["nvcr.io"], {"nvcr.io": cr._AUTH_UNKNOWN}).level == cr.WARN,
)

# ── 6. verdict ───────────────────────────────────────────────────────────────
ok, _ = cr.verdict([cr.Check("a", cr.PASS, ""), cr.Check("b", cr.WARN, "")])
check("verdict: PASS+WARN only → run-ready (WARN never blocks)", ok)
ok, _ = cr.verdict([cr.Check("a", cr.PASS, ""), cr.Check("b", cr.FAIL, "")])
check("verdict: any FAIL → NOT run-ready", not ok)


# ── 7. probe_registry_auth (mocked http) ─────────────────────────────────────
def http_open(method, url, headers=None):
    return (200, {}, "")


def http_401_then_token(status_token):
    def _h(method, url, headers=None):
        if url.endswith("/v2/"):
            return (
                401,
                {"WWW-Authenticate": 'Bearer realm="https://auth.x/token",service="reg"'},
                "",
            )
        if "auth.x/token" in url:
            if status_token == 200:
                return (200, {}, json.dumps({"token": "T"}))
            return (status_token, {}, "")
        return (404, {}, "")

    return _h


def http_unreachable(method, url, headers=None):
    raise OSError("connection refused")


check(
    "probe_registry_auth: open registry (200) → ok",
    cr.probe_registry_auth("x", None, http_open) == cr._AUTH_OK,
)
check(
    "probe_registry_auth: 401→token 200 → ok (cred authenticates)",
    cr.probe_registry_auth("nvcr.io", ("u", "p"), http_401_then_token(200)) == cr._AUTH_OK,
)
check(
    "probe_registry_auth: 401→token 403 → forbidden (bad/expired key)",
    cr.probe_registry_auth("nvcr.io", ("u", "p"), http_401_then_token(403)) == cr._AUTH_FORBIDDEN,
)
check(
    "probe_registry_auth: registry unreachable → unknown (safe-degrade)",
    cr.probe_registry_auth("nvcr.io", ("u", "p"), http_unreachable) == cr._AUTH_UNKNOWN,
)

# ── 8. probe_staging_roundtrip: happy path + cleanup ─────────────────────────
krun = FakeKrun(
    [
        (("get", "pvc", "jsonpath={.status.phase}"), (0, "Bound", "")),
        (("wait", "pod"), (0, "", "")),
        (
            ("exec", "cat"),
            (0, "", ""),
        ),  # placeholder; real token compared below via a smarter fake
    ]
)


class StagingKrun:
    """Full staging happy-path fake (pod-triggered bind): pod ready → bound, cp ok, readback echoes the token."""

    def __init__(self):
        self.calls = []
        self.token = None

    def __call__(self, args, timeout=30, stdin=None):
        self.calls.append(list(args))
        if "cp" in args:
            # capture the token from the temp file so exec-cat can echo it back
            src = args[args.index("cp") + 1]
            try:
                self.token = Path(src).read_text()
            except Exception:
                self.token = ""
            return (0, "", "")
        if "wait" in args:  # pod becomes ready → WFFC bind implied
            return (0, "", "")
        if "exec" in args and "cat" in args:
            return (0, self.token or "", "")
        return (0, "", "")

    def deletes(self):
        return [c for c in self.calls if "delete" in c]


sk = StagingKrun()
c = cr.probe_staging_roundtrip(sk, "ns", "ebs", "pull-secret", "ReadWriteOnce", suffix="test01")
check(
    "probe_staging_roundtrip: bound+ready+cp+readback → PASS",
    c.level == cr.PASS,
    c.message,
)
# cleanup: pod AND pvc were deleted, and the module-level tracker is empty (no leak)
_del_targets = [" ".join(d) for d in sk.deletes()]
check(
    "staging cleanup: pod deleted",
    any("pod" in d and "llmb-readycheck-test01" in d for d in _del_targets),
)
check(
    "staging cleanup: pvc deleted",
    any("pvc" in d and "llmb-readycheck-test01" in d for d in _del_targets),
)
check("staging cleanup: tracker empty (no leaked resource)", cr._LIVE_RESOURCES == [])


# ── 9. probe_staging_roundtrip: cp fails every retry → FAIL, still cleaned up ─
class CpFailKrun(StagingKrun):
    def __call__(self, args, timeout=30, stdin=None):
        self.calls.append(list(args))
        if "cp" in args:
            return (1, "", "context deadline exceeded")
        if "wait" in args:  # pod ready (bind ok) — the failure is purely the cp stream
            return (0, "", "")
        return (0, "", "")


cf = CpFailKrun()
c = cr.probe_staging_roundtrip(cf, "ns", "ebs", "ps", "ReadWriteOnce", suffix="test02")
check(
    "probe_staging_roundtrip: cp fails all retries → FAIL",
    c.level == cr.FAIL,
    c.message,
)
check(
    "staging cleanup on failure: pod+pvc still deleted",
    sum(1 for d in cf.deletes() if "llmb-readycheck-test02" in " ".join(d)) >= 2,
)
check("staging cleanup on failure: tracker empty", cr._LIVE_RESOURCES == [])


# ── 10. probe_staging_roundtrip: pod never ready + ProvisioningFailed event → FAIL (WFFC-safe) ────────
class ProvFailKrun(StagingKrun):
    def __call__(self, args, timeout=30, stdin=None):
        self.calls.append(list(args))
        a = " ".join(args)
        if "wait" in args:  # pod never becomes ready (PVC can't provision)
            return (1, "", "timed out waiting for the condition")
        if "phase" in a:
            return (0, "Pending", "")
        if "events" in args:
            return (
                0,
                json.dumps(
                    {
                        "items": [
                            {
                                "reason": "ProvisioningFailed",
                                "message": "storageclass not found",
                            }
                        ]
                    }
                ),
                "",
            )
        return (0, "", "")


pf = ProvFailKrun()
c = cr.probe_staging_roundtrip(pf, "ns", "bogus-sc", "ps", "ReadWriteOnce", suffix="test03")
check(
    "probe_staging_roundtrip: pod-triggered bind + ProvisioningFailed → FAIL (bad StorageClass)",
    c.level == cr.FAIL,
    c.message,
)
check(
    "staging cleanup: pod+pvc deleted even when the PVC never provisioned",
    sum(1 for d in pf.deletes() if "llmb-readycheck-test03" in " ".join(d)) >= 2 and cr._LIVE_RESOURCES == [],
)

# ── 10b. probe_control_rwx: pod mounts RWX PVC → PASS; cleanup ───────────────
rwx = StagingKrun()
c = cr.probe_control_rwx(rwx, "ns", "efs-sc", "ps", suffix="rwx01")
check("probe_control_rwx: pod mounts RWX PVC → PASS", c.level == cr.PASS, c.message)
check(
    "control_rwx cleanup: pod+pvc deleted, tracker empty",
    any("llmb-readycheck-rwx-rwx01" in " ".join(d) for d in rwx.deletes()) and cr._LIVE_RESOURCES == [],
)


# ── 10c. QA fail-fast (#3) at the PROBE level: an s3.csi class errors out WITHOUT creating a doomed PVC ──
class ObjectStoreKrun(StagingKrun):
    def __call__(self, args, timeout=30, stdin=None):
        self.calls.append(list(args))
        if "storageclass" in args and any("provisioner" in x for x in args):
            return (0, "s3.csi.aws.com", "")
        return (0, "", "")


_obj = ObjectStoreKrun()
c = cr.probe_staging_roundtrip(_obj, "ns", "s3-object", "ps", "ReadWriteOnce", suffix="obj01")
check(
    "probe_staging_roundtrip: object-store class → FAIL FAST, message names the class",
    c.level == cr.FAIL and "s3-object" in c.message,
    c.message,
)
check(
    "probe_staging_roundtrip: object-store class → NO PVC/pod apply attempted (fail-fast before create)",
    not any("apply" in cc for cc in _obj.calls)
    and not any("run" in cc for cc in _obj.calls)
    and cr._LIVE_RESOURCES == [],
    str(_obj.calls),
)
_obj2 = ObjectStoreKrun()
c = cr.probe_control_rwx(_obj2, "ns", "s3-object", "ps", suffix="obj02")
check(
    "probe_control_rwx: object-store class → FAIL FAST, no PVC created",
    c.level == cr.FAIL and not any("apply" in cc for cc in _obj2.calls),
    c.message,
)

# ── 11. run_battery(fast=True) bypasses LIVE probes but still runs offline ${VAR}-reconcile ──────────
checks = cr.run_battery({"NAMESPACE": "n"}, recipe_cell=None, fast=True)
ids = [c.id for c in checks]
check(
    "run_battery: --fast → offline var-reconcile + a SKIP note, no LIVE probe",
    "var-reconcile" in ids
    and any(c.level == cr.SKIP for c in checks)
    and not any(i in ids for i in ("staging-roundtrip", "gpu-nodes", "pull-secret")),
    str(ids),
)

# ── 12. probe_var_reconcile: chosen cell with an empty referenced var → FAIL Check ───────────────────
import tempfile as _tf  # noqa: E402

with _tf.TemporaryDirectory() as _td:
    _cell = Path(_td) / "c"
    (_cell / "rendered").mkdir(parents=True)
    (_cell / "rendered" / "j.yaml").write_text("env:\n  - IP=${NO_INTERNET_DNS_IP}\n")
    (_cell / "recipe.yaml").write_text("envelope: {gpu_type: B200}\n")
    c = cr.probe_var_reconcile({"NO_INTERNET_DNS_IP": ""}, str(_cell))
    check(
        "probe_var_reconcile: chosen cell, referenced var empty → FAIL Check",
        c.level == cr.FAIL,
        c.message,
    )
    c2 = cr.probe_var_reconcile({"NO_INTERNET_DNS_IP": "10.0.0.1"}, str(_cell))
    check(
        "probe_var_reconcile: chosen cell, var populated → not FAIL",
        c2.level != cr.FAIL,
        c2.message,
    )


# ── done ─────────────────────────────────────────────────────────────────────
if fails:
    print(f"\n{len(fails)} FAILED: {fails}")
    sys.exit(1)
print("\nall cluster-readiness selftests passed")
