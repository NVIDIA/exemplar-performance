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

"""Offline tests for `--env-set` and `--env-unset`.

The tests verify that overrides update serving containers, mark the run as a variant, leave committed recipes unchanged, preserve ownership metadata, and cannot be published as pinned results. Malformed overrides must fail before apply.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import merge_env_override as meo  # noqa: E402
import recipe_hash as rh  # noqa: E402

MERGE = ROOT / "scripts" / "merge_env_override.py"
DEPLOY = ROOT / "scripts" / "deploy.sh"
RUN = ROOT / "scripts" / "run.sh"
PUBLISH = ROOT / "scripts" / "publish.py"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def merge(stdin: str, *args: str, **env: str) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e.setdefault("LLMB_ENV_SET", "")
    e.setdefault("LLMB_ENV_UNSET", "")
    e.update(env)
    return subprocess.run(
        ["python3", str(MERGE), *args],
        input=stdin,
        capture_output=True,
        text=True,
        env=e,
    )


# ── 0. static lint ────────────────────────────────────────────────────────────
for sh in (DEPLOY, RUN):
    p = subprocess.run(["bash", "-n", str(sh)], capture_output=True, text=True)
    check(f"{sh.name} is valid bash (bash -n)", p.returncode == 0, p.stderr.strip()[:300])


# ── 1. the pure spec parser ───────────────────────────────────────────────────
spec = meo.parse_spec(["A=1", "B=x=y"], ["C", "D", "C"])
check(
    "parse_spec: KEY=VALUE splits on the FIRST '=' (values may contain '=')",
    spec["set"] == {"A": "1", "B": "x=y"},
)
check("parse_spec: unset de-duplicates and sorts", spec["unset"] == ["C", "D"])
check(
    "parse_spec: variant_id is 8 hex chars",
    len(spec["variant_id"]) == 8 and int(spec["variant_id"], 16) >= 0,
)
check(
    "variant_id is order-independent (same override set → same id)",
    meo.parse_spec(["B=x=y", "A=1"], ["D", "C"])["variant_id"] == spec["variant_id"],
)
check(
    "variant_id moves when the override set changes",
    meo.parse_spec(["A=2"], [])["variant_id"] != meo.parse_spec(["A=1"], [])["variant_id"],
)
check(
    "describe: a compact one-line summary",
    meo.describe(meo.parse_spec(["A=1"], ["Z"])) == "-Z +A=1",
)

for bad, why in ((["NOEQUALS"], "no '='"), (["1BAD=v"], "invalid var name")):
    try:
        meo.parse_spec(bad, [])
        check(f"parse_spec rejects {why}", False, str(bad))
    except ValueError:
        check(f"parse_spec rejects {why}", True)
try:
    meo.parse_spec(["A=1"], ["A"])
    check("parse_spec rejects a key passed to BOTH --env-set and --env-unset", False)
except ValueError:
    check("parse_spec rejects a key passed to BOTH --env-set and --env-unset", True)


# ── 2. the apply-stream stage ─────────────────────────────────────────────────
DEP = """apiVersion: apps/v1
kind: Deployment
metadata:
  name: foo-server
  namespace: ${NAMESPACE}
  labels: {app.kubernetes.io/managed-by: llmb-recipe}
spec:
  replicas: 1
  template:
    metadata:
      labels: {app: foo-server}
    spec:
      initContainers:
        - name: stage
          env: [{name: KEEPME, value: "init"}]
      containers:
        - name: server
          image: img:1
          env:
            - {name: KEEPME, value: "1"}
            - {name: DROPME, value: "100000"}
            - {name: TWEAKME, value: "old"}
        - name: sidecar
          image: side:1
          env:
            - {name: DROPME, value: "100000"}
"""

p = merge(DEP, LLMB_ENV_UNSET="DROPME", LLMB_ENV_SET="TWEAKME=new\nNEWVAR=42")
out = p.stdout
check("stage: exits 0 on a valid override", p.returncode == 0, p.stderr)
check(
    "stage: --env-unset removes the var from EVERY container (both server and sidecar)",
    "name: DROPME" not in out and "100000" not in out,
    out,
)
check(
    "stage: --env-set replaces an existing value",
    "new" in out and "old" not in out,
    out,
)
check(
    "stage: --env-set appends a var that wasn't there",
    "NEWVAR" in out and "42" in out,
    out,
)
check("stage: untouched env entries survive", out.count("KEEPME") >= 2, out)
check(
    "stage: initContainers are NOT patched (weight/config staging, not the engine)",
    "init" in out,
    out,
)
check(
    "stage: preserves the rest of the spec (replicas, images, fleet labels)",
    "replicas: 1" in out and "img:1" in out and "llmb-recipe" in out,
    out,
)
check(
    "stage: marks the object llmb.nvidia.com/variant=true",
    "llmb.nvidia.com/variant: 'true'" in out or 'llmb.nvidia.com/variant: "true"' in out,
    out,
)
_vid = meo.parse_spec(["TWEAKME=new", "NEWVAR=42"], ["DROPME"])["variant_id"]
check(
    "stage: stamps the 8-hex variant-id label matching the spec digest",
    f"llmb.nvidia.com/variant-id: {_vid}" in out or f"llmb.nvidia.com/variant-id: '{_vid}'" in out,
    out,
)
check(
    "stage: annotates the EXACT overrides (so the object itself carries what changed)",
    "llmb.nvidia.com/env-overrides:" in out and "DROPME" in out and "NEWVAR" in out,
    out,
)
check(
    "stage: marks the POD TEMPLATE too (so the pods carry the variant marker, not just the Deployment)",
    out.count("llmb.nvidia.com/variant:") >= 2,
    out,
)
check(
    "stage: says VARIANT + 'NOT publishable' on stderr (never a silent variant)",
    "VARIANT" in p.stderr and "NOT publishable" in p.stderr,
    p.stderr,
)

p = merge(DEP, LLMB_ENV_UNSET="NEVER_PRESENT")
check(
    "stage: WARNs when an --env-unset key matched nothing (typo / set via args, not env:)",
    "WARN" in p.stderr and "NEVER_PRESENT" in p.stderr,
    p.stderr,
)

# valueFrom binding → a literal override replaces it (documented behaviour)
VF = (
    "kind: Deployment\nmetadata: {name: d}\nspec:\n  template:\n    spec:\n      containers:\n"
    "        - name: c\n          env:\n            - name: TOK\n              valueFrom:\n"
    "                secretKeyRef: {name: s, key: k}\n"
)
out = merge(VF, LLMB_ENV_SET="TOK=literal").stdout
check(
    "stage: --env-set over a valueFrom binding replaces it with the literal",
    "valueFrom" not in out and "literal" in out,
    out,
)

# non-workload kinds are left alone
out = merge("kind: Service\nmetadata: {name: svc}\n", LLMB_ENV_SET="A=1").stdout
check(
    "stage: non-workload kinds (Service) are not patched or marked",
    "variant" not in out and "A" not in out.replace("apiVersion", ""),
    out,
)


# ── 3. ZERO REGRESSION — no override → byte-identical passthrough ─────────────
for name, blob in (
    ("deployment", DEP),
    ("multi-doc", DEP + "---\nkind: Service\nmetadata: {name: s}\n"),
    ("not-even-yaml", "just some bytes\n\tand a tab\n"),
):
    r = merge(blob)
    check(
        f"no override: {name} passes through BYTE-IDENTICAL (zero regression for the normal path)",
        r.returncode == 0 and r.stdout == blob,
        repr(r.stdout[:120]),
    )


# ── 4. FAIL-CLOSED on a malformed spec ────────────────────────────────────────
r = merge(DEP, LLMB_ENV_SET="OOPS_NO_EQUALS")
check(
    "malformed --env-set: exits NON-ZERO and emits NO manifests (never a silent pinned-config run)",
    r.returncode != 0 and r.stdout.strip() == "",
    f"rc={r.returncode} out={r.stdout[:120]!r}",
)
r = merge("kind: Deployment\nmetadata: {name: d\n  bad yaml: [\n", LLMB_ENV_SET="A=1")
check(
    "unparseable stream + an override in play: REFUSE (exit non-zero, no passthrough)",
    r.returncode != 0 and r.stdout.strip() == "",
    f"rc={r.returncode}",
)


# ── 5. deploy.sh end-to-end (render + apply), with a fake kubectl/envsubst ────
def _fixture(td: Path) -> tuple[Path, Path, dict]:
    """A throwaway ROOT mirroring scripts/ + cluster-profiles/ + one cell, plus PATH shims."""
    root = td / "root"
    (root / "scripts").mkdir(parents=True)
    # _model_cache.sh + model_cache.py are copied because deploy.sh now resolves its model-cache claim
    # through them (the one definition every consumer shares). model_cache.py is deliberately
    # dependency-light — stdlib + PyYAML — so a sandbox root like this one can host it.
    for s in (
        "deploy.sh",
        "merge_env_override.py",
        "merge_rdma_selector.py",
        "merge_imex_claim.py",
        "merge_imex_strip.py",
        "merge_run_owner.py",
        "_model_cache.sh",
        "model_cache.py",
    ):
        shutil.copy2(ROOT / "scripts" / s, root / "scripts" / s)
    # This fixture exercises deploy rendering/patching, not the live cluster checks. The real deploy now
    # fail-closes through preflight before apply; stub that independent boundary as a successful check.
    (root / "scripts" / "preflight.py").write_text("import sys\nsys.exit(0)\n")
    (root / "cluster-profiles").mkdir()
    # MODEL_CACHE_PVC is REQUIRED now: deploy.sh fail-closes rather than rendering an empty claimName.
    (root / "cluster-profiles" / "fake.env").write_text(
        "NAMESPACE=testns\nKUBE_CONTEXT=\nMODEL_CACHE_PVC=fake-model-cache\n"
    )
    cell = root / "cell"
    (cell / "rendered").mkdir(parents=True)
    (cell / "recipe.yaml").write_text("envelope:\n  name: foo\n  model: m\n  scenario: llm-perf\nserving:\n  tp: 1\n")
    (cell / "rendered" / "server.yaml").write_text(DEP)
    bindir = td / "bin"
    bindir.mkdir()
    (bindir / "envsubst").write_text("#!/usr/bin/env bash\nexec cat\n")
    log = td / "kubectl.log"
    (bindir / "kubectl").write_text(
        f'#!/usr/bin/env bash\necho "ARGS: $*" >> "{log}"\n'
        f'case "$*" in *"apply -f -"*) cat >> "{log}" ;; esac\nexit 0\n'
    )
    for f in bindir.iterdir():
        f.chmod(0o755)
    env = dict(os.environ)
    env["PATH"] = f"{bindir}:{env['PATH']}"
    kubectl_shim = bindir / "kubectl"
    bash_env = bindir / "bash-env.sh"
    bash_env.write_text(
        f'envsubst() {{ cat; }}\nkubectl() {{ bash "{kubectl_shim}" "$@"; }}\nexport -f envsubst kubectl\n'
    )
    env["BASH_ENV"] = str(bash_env)
    for k in ("LLMB_ENV_SET", "LLMB_ENV_UNSET", "RUN_OWNER_NAME", "RUN_OWNER_UID"):
        env.pop(k, None)
    return root, cell, env


def _deploy(root: Path, cell: Path, env: dict, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", str(root / "scripts" / "deploy.sh"), str(cell), "fake", *args],
        capture_output=True,
        text=True,
        env=env,
    )


with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    root, cell, env = _fixture(tdp)
    rendered = cell / "rendered" / "server.yaml"
    before_hash, before_bytes = rh.recipe_hash(cell), rendered.read_bytes()

    clean = _deploy(root, cell, env, "--render-only")
    check(
        "deploy.sh --render-only (no override): exits 0",
        clean.returncode == 0,
        clean.stderr[:300],
    )
    check(
        "deploy.sh --render-only (no override): output carries NO variant marker",
        "variant" not in clean.stdout,
        clean.stdout[:300],
    )
    clean2 = _deploy(root, cell, env, "--render-only")
    check(
        "deploy.sh --render-only (no override): deterministic + unchanged by the new stage",
        clean.stdout == clean2.stdout,
    )

    var = _deploy(
        root,
        cell,
        env,
        "--render-only",
        "--env-unset",
        "DROPME",
        "--env-set",
        "TWEAKME=new",
    )
    check(
        "deploy.sh: --env-set/--env-unset are accepted alongside --render-only in any order",
        var.returncode == 0,
        var.stderr[:300],
    )
    check(
        "deploy.sh: the override reaches the rendered stream",
        "name: DROPME" not in var.stdout and "value: new" in var.stdout,
        var.stdout[:400],
    )
    check(
        "deploy.sh: banner announces the VARIANT + that recipe_hash is unchanged",
        "VARIANT" in var.stdout and "recipe_hash is UNCHANGED" in var.stdout,
        var.stdout[:400],
    )
    check(
        "deploy.sh: variant output is marked",
        "llmb.nvidia.com/variant" in var.stdout,
        var.stdout[:400],
    )

    # (2) THE SAFETY INVARIANT: an override moves neither the committed bytes nor the fingerprint.
    check(
        "recipe_hash is UNCHANGED by a variant deploy (a runtime override is not a recipe change)",
        rh.recipe_hash(cell) == before_hash,
        f"{before_hash} -> {rh.recipe_hash(cell)}",
    )
    check(
        "committed rendered/*.yaml is byte-identical after a variant deploy (nothing was edited)",
        rendered.read_bytes() == before_bytes,
    )

    # (5) run-owner GC still applies — the feature whose absence leaked GPUs.
    oenv = dict(env, RUN_OWNER_NAME="foo-runowner-r1", RUN_OWNER_UID="uid-abc")
    owned = _deploy(root, cell, oenv, "--render-only", "--env-unset", "DROPME")
    check(
        "VARIANT deploy STILL stamps the run-owner ownerReference (owned from birth → GC frees the GPU)",
        "ownerReferences" in owned.stdout and "uid: uid-abc" in owned.stdout and "controller: true" in owned.stdout,
        owned.stdout[:600],
    )
    check(
        "VARIANT deploy carries BOTH the ownerRef and the variant marker on the same object",
        "llmb.nvidia.com/variant" in owned.stdout,
        owned.stdout[:600],
    )
    check(
        "VARIANT deploy keeps the fleet attribution label (app.kubernetes.io/managed-by: llmb-recipe)",
        "llmb-recipe" in owned.stdout,
        owned.stdout[:600],
    )

    # (7) fail-closed: a malformed spec aborts BEFORE the cluster is touched.
    log = tdp / "kubectl.log"
    bad = _deploy(root, cell, env, "--env-set", "OOPS")
    check(
        "deploy.sh: a malformed --env-set aborts non-zero",
        bad.returncode != 0,
        bad.stdout[-300:],
    )
    check(
        "deploy.sh: a malformed --env-set applies NOTHING (no kubectl call at all)",
        not log.exists() or log.read_text().strip() == "",
        (log.read_text()[:200] if log.exists() else ""),
    )

    # live apply path (fake kubectl): the patched manifest is what gets applied
    applied = _deploy(root, cell, env, "--env-unset", "DROPME")
    text = log.read_text() if log.exists() else ""
    check(
        "deploy.sh (apply path): exits 0 under a variant",
        applied.returncode == 0,
        applied.stderr[:300],
    )
    check(
        "deploy.sh (apply path): the APPLIED manifest is the patched one",
        "name: DROPME" not in text and "llmb.nvidia.com/variant" in text,
        text[:400],
    )


# ── 6. publish.py REFUSES a variant run-dir (unconditionally) ─────────────────
def _publish(run_dir: Path, cell: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["python3", str(PUBLISH), str(cell), str(run_dir), *extra],
        capture_output=True,
        text=True,
    )


with tempfile.TemporaryDirectory() as td:
    tdp = Path(td)
    cell = tdp / "cell"
    (cell / "rendered").mkdir(parents=True)
    (cell / "recipe.yaml").write_text("envelope:\n  name: foo\n  model: m\n  scenario: llm-perf\n")

    # (a) the _variant.json marker (written by run.sh BEFORE the deploy — survives a crashed run)
    rd = tdp / "run-variant"
    rd.mkdir()
    (rd / "_variant.json").write_text(
        json.dumps({"set": {"TWEAKME": "new"}, "unset": ["DROPME"], "variant_id": "abc12345"})
    )
    for extra, label in (
        ((), "plain"),
        (("--force",), "--force"),
        (("--dry-run",), "--dry-run"),
    ):
        r = _publish(rd, cell, *extra)
        check(
            f"publish REFUSES a _variant.json run-dir ({label} — there is no escape hatch)",
            r.returncode != 0 and "VARIANT" in (r.stdout + r.stderr),
            (r.stdout + r.stderr)[:300],
        )
    r = _publish(rd, cell)
    msg = r.stdout + r.stderr
    check(
        "publish's refusal names the variant id AND the exact overrides",
        "abc12345" in msg and "DROPME" in msg and "TWEAKME" in msg,
        msg[:400],
    )
    check(
        "publish's refusal points at the honest fix (edit the recipe, re-run without overrides)",
        "change-recipe" in msg or "WITHOUT overrides" in msg,
        msg[:400],
    )

    # (b) the same facts carried inside the run's own provenance (run_meta.json)
    rd2 = tdp / "run-meta-variant"
    rd2.mkdir()
    (rd2 / "run_meta.json").write_text(
        json.dumps(
            {
                "run_id": "r1",
                "overrides": {"set": {}, "unset": ["DROPME"], "variant_id": "deadbeef"},
            }
        )
    )
    r = _publish(rd2, cell)
    check(
        "publish REFUSES when run_meta.json records overrides (cross-machine collect path)",
        r.returncode != 0 and "VARIANT" in (r.stdout + r.stderr),
        (r.stdout + r.stderr)[:300],
    )

    # (c) a CLEAN run is NOT refused by this guard (no false positives)
    rd3 = tdp / "run-clean"
    rd3.mkdir()
    (rd3 / "run_meta.json").write_text(json.dumps({"run_id": "r2", "model_name": "m"}))
    r = _publish(rd3, cell, "--dry-run")
    check(
        "a CLEAN run-dir is NOT flagged as a variant (no false positive)",
        "VARIANT" not in (r.stdout + r.stderr),
        (r.stdout + r.stderr)[:300],
    )

    # the pure predicate itself
    sys.path.insert(0, str(ROOT / "scripts"))
    import publish as _pub  # noqa: E402

    check(
        "variant_overrides(): None for a clean run-dir",
        _pub.variant_overrides(rd3) is None,
    )
    check(
        "variant_overrides(): the spec for a marked run-dir",
        (_pub.variant_overrides(rd) or {}).get("variant_id") == "abc12345",
    )
    check(
        "variant_overrides(): tolerates an empty/absent run-dir",
        _pub.variant_overrides(tdp / "nope") is None,
    )


# ── 7. run.sh wiring (the launcher keeps every guarantee + marks the run) ─────
run_sh = RUN.read_text()
for name, needle in [
    ("parses --env-set / --env-unset (both spellings)", "--env-set=*"),
    (
        "exports the override so deploy.sh's apply stream sees it",
        "export LLMB_ENV_SET LLMB_ENV_UNSET",
    ),
    ("validates the spec BEFORE touching the cluster", "nothing was launched"),
    ("writes results/<run-id>/_variant.json before the deploy", "_variant.json"),
    ("stamps the overrides into run_meta.json", "_stamp_variant_meta"),
    ("prints NO publish command for a variant run", "NOT publishable"),
    (
        "warns that --skip-server means the override never reached a container",
        "were NOT applied",
    ),
]:
    check(f"run.sh: {name}", needle in run_sh, name)

i_var = run_sh.find('VARIANT_JSON="$(python3')
i_owner = run_sh.find('run_owner.sh" ensure')
i_deploy = run_sh.find('deploy.sh" "$CELL" "$PROFILE"')
check(
    "run.sh: the override is validated BEFORE the run-owner and the deploy (fail before any GPU is held)",
    0 < i_var < i_owner < i_deploy,
    f"var@{i_var} owner@{i_owner} deploy@{i_deploy}",
)
check(
    "run.sh: a variant run still goes through the run-owner path (GC-backed GPU release preserved)",
    i_owner > 0 and "adopt-deploy" in run_sh,
)

deploy_sh = DEPLOY.read_text()
check(
    "deploy.sh: merge_env_override runs BEFORE merge_run_owner (the ownerRef stamps the patched object)",
    deploy_sh.find("merge_env_override.py") < deploy_sh.find("merge_run_owner.py"),
)
check(
    "deploy.sh: the env stage is in BOTH the --render-only and the apply pipe (what you preview is applied)",
    deploy_sh.count('scripts/merge_env_override.py" |') == 2,
    str(deploy_sh.count('scripts/merge_env_override.py" |')),
)

# The digest below is not load-bearing; it just proves the doc claim that ids are content-addressed.
check(
    "variant ids are content-addressed (a stable handle for the same override set)",
    meo.variant_id({"set": {"A": "1"}, "unset": []}) == hashlib.sha256(b'{"set":{"A":"1"},"unset":[]}').hexdigest()[:8],
)

print()
if fails:
    print(f"selftest_env_override: {len(fails)} FAILED: {fails}")
    sys.exit(1)
print("selftest_env_override: all checks passed")
