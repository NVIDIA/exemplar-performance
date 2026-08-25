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

"""Verify that model download, validation, and serving resolve the same cache claim.

The checks cover profile-only claim resolution, consistent mounts, download-to-mount agreement, rejected
recipe-level claim overrides, shell consumers, and unknown cluster-read failures. They run offline.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).parent))

import install  # type: ignore[import]  # noqa: E402

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def _cell(model, cache=None, path="recipes/x/y"):
    c = {"model": model, "_path": path, "requires": {"gpu": {"count": 4}}}
    if cache is not None:
        c["requires"]["cache"] = cache
    return c


# ---------------------------------------------------------------------------
# 1. THE RESOLVER — profile-only, total, and honest about "unresolved"
# ---------------------------------------------------------------------------
print("\n-- 1. resolve_cache_claim is profile-only and total --")

check(
    "MODEL_CACHE_PVC resolves",
    install.resolve_cache_claim(_cell("glm5-fp8"), {"MODEL_CACHE_PVC": "shared"}) == ("shared", "MODEL_CACHE_PVC"),
)

# The per-model override key — cluster truth, in the cluster file.
check(
    "per-model key beats the default",
    install.resolve_cache_claim(
        _cell("nemotron-ultra-nvfp4"),
        {
            "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
            "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4": "nemotron-ultra-nvfp4-cache",
        },
    )
    == ("nemotron-ultra-nvfp4-cache", "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4"),
)
check(
    "a per-model key for ANOTHER model does not leak",
    install.resolve_cache_claim(
        _cell("glm5-fp8"),
        {
            "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
            "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4": "nemotron-ultra-nvfp4-cache",
        },
    )[0]
    == "glm5-fp8-model-cache",
)
check(
    "key derivation: nemotron-ultra-nvfp4 → MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4",
    install.model_cache_env_key("nemotron-ultra-nvfp4") == "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4",
    install.model_cache_env_key("nemotron-ultra-nvfp4"),
)
check(
    "key derivation: qwen3-0-6b → MODEL_CACHE_PVC_QWEN3_0_6B",
    install.model_cache_env_key("qwen3-0-6b") == "MODEL_CACHE_PVC_QWEN3_0_6B",
)

# THE PRE-FIX BEHAVIOUR, now impossible: an empty profile var must NOT become an invented claim name.
# Before this change derive_recipe_cache returned "<model-slug>-model-cache" here, install downloaded into
# it, and the server mounted "" — 31 cells blocked AFTER ~300 GiB.
empty = install.resolve_cache_claim(_cell("glm5-fp8"), {"MODEL_CACHE_PVC": ""})
check(
    "empty MODEL_CACHE_PVC → UNRESOLVED, never an invented '<slug>-model-cache'",
    empty == ("", ""),
    str(empty),
)
check(
    "no cache keys at all → UNRESOLVED",
    install.resolve_cache_claim(_cell("m"), {}) == ("", ""),
)

# A recipe field must NOT influence the name — that is the category error that made download != mount.
check(
    "requires.cache.name is IGNORED by the resolver (cluster truth, not recipe truth)",
    install.resolve_cache_claim(
        _cell("nemotron-ultra-nvfp4", {"name": "recipe-named-cache"}),
        {"MODEL_CACHE_PVC": "profile-cache"},
    )[0]
    == "profile-cache",
)

# ---------------------------------------------------------------------------
# 2. THE MOUNT — every cache mount on disk is exactly ${MODEL_CACHE_PVC}
# ---------------------------------------------------------------------------
print("\n-- 2. every cache mount is ${MODEL_CACHE_PVC} --")

# The whole design rests on this: rendered/*.yaml stays cluster-PORTABLE (a var, not a literal), which is
# also why recipe_hash does not move. A template that hardcoded a claim name would silently reintroduce the
# bug AND roll every published recipe_hash, so scan the real bytes rather than trusting the convention.
CACHE_MOUNT_RE = re.compile(r"claimName:\s*(\$\{[A-Z_][A-Z0-9_]*\}|\S+)")
# WHICH mount is a MODEL-CACHE mount is decided by the VOLUME NAME, not by pattern-matching the claim
# string. The old rule inverted that — it exempted anything that LOOKED like an artifacts claim and flagged
# everything else — so widening the scan to all of serving/ immediately flagged the governor's own
# `control` volume (claimName: llmb-control), which has nothing to do with model weights. Volume names in
# this repo are `model-cache` / `model-store` for the weights, `artifacts` per cell, `control` for the
# governor; both halves are asserted below so neither the classification nor the corpus can drift silently.
_CACHE_VOLUMES = {"model-cache", "model-store"}
_VOL_NAME_RE = re.compile(r"name:\s*([A-Za-z0-9_.-]+)")

bad_mounts: list[str] = []
scanned = cache_mounts = 0
# ALL of serving/, not just serving/**/templates/*.j2. The hand-maintained serving/dynamo-disagg/*.yaml
# manifests mount ${MODEL_CACHE_PVC} and live OUTSIDE templates/, so the narrower glob never looked at them
# — a hardcoded claim name there would have been invisible to the one test that exists to see it.
_misfiled: list[str] = []
for f in sorted(
    list((ROOT / "recipes").glob("**/rendered/*.yaml"))
    + [p for p in (ROOT / "serving").rglob("*") if p.is_file() and p.suffix in (".j2", ".yaml", ".yml")]
):
    txt = f.read_text()
    for m in CACHE_MOUNT_RE.finditer(txt):
        claim = m.group(1)
        scanned += 1
        _names = _VOL_NAME_RE.findall(txt[max(0, m.start() - 260) : m.start()])
        _vol = _names[-1] if _names else "?"
        if _vol not in _CACHE_VOLUMES:
            # Not a weights mount — but if it carries ${MODEL_CACHE_PVC} anyway, the classification itself
            # is wrong and must be fixed rather than quietly widening the exemption.
            if claim == "${MODEL_CACHE_PVC}":
                _misfiled.append(f"{f.relative_to(ROOT)}: volume '{_vol}' mounts ${{MODEL_CACHE_PVC}}")
            continue
        cache_mounts += 1
        if claim != "${MODEL_CACHE_PVC}":
            bad_mounts.append(f"{f.relative_to(ROOT)}: volume '{_vol}' claimName: {claim}")
check(
    f"every model-cache mount is ${{MODEL_CACHE_PVC}} "
    f"({cache_mounts} weights mounts of {scanned} claimName refs scanned)",
    not bad_mounts,
    "; ".join(bad_mounts[:4]),
)
check(
    "no volume OUTSIDE the model-cache set mounts ${MODEL_CACHE_PVC} (the classification is complete)",
    not _misfiled,
    "; ".join(_misfiled[:4]),
)
# Keep a meaningful non-vacuity floor immediately below the current 71 legitimate weights consumers.
# Synthetic benchmark pods and disaggregated HTTP frontends intentionally omit the cache; serving workers,
# aggregate servers and downloaders retain it. A tiny >0 guard would pass if a whole worker stack lost its
# model volume, while 70 still catches even one additional accidental removal from today's catalog.
check(
    "the scan found at least 70 legitimate model-cache mounts and their claim references "
    "(synthetic bench and HTTP-only frontend pods intentionally omit the weights mount)",
    cache_mounts >= 70 and scanned >= cache_mounts,
    f"{cache_mounts}/{scanned}",
)

# ---------------------------------------------------------------------------
# 3. DOWNLOAD TARGET == MOUNT TARGET, over the REAL catalog
# ---------------------------------------------------------------------------
print("\n-- 3. download claim == mount claim, over the real catalog --")

catalog = install.load_catalog()
check("catalog loaded", bool(catalog), str(len(catalog or [])))

PROFILES = [
    # the QA cluster shape: one default claim + a per-model override for the model with its own PVC
    {
        "NAMESPACE": "example-benchmark",
        "GPU_PRODUCT": "NVIDIA-B200",
        "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
        "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4": "nemotron-ultra-nvfp4-cache",
    },
    # a single-claim cluster
    {
        "NAMESPACE": "bench",
        "GPU_PRODUCT": "NVIDIA-B200",
        "MODEL_CACHE_PVC": "shared-model-cache",
    },
]
for prof in PROFILES:
    cells = install.gpu_matching_cells(catalog, prof)
    # What the DOWNLOAD Job mounts: phase_c_download uses cache_by_repo[model_repo], falling back to
    # prof["MODEL_CACHE_PVC"] — reproduce that exact expression.
    by_repo, cbr_errs = install.cache_pvc_by_repo(cells, prof)
    check(
        f"[{prof.get('MODEL_CACHE_PVC')}] cache_pvc_by_repo reports no errors on a valid profile",
        cbr_errs == [],
        str(cbr_errs),
    )
    diverged: list[str] = []
    for cell in cells:
        mount, _ = install.resolve_cache_claim(cell, prof)  # what envsubst puts in the manifest
        spec_name = install.derive_recipe_cache(cell, prof)["name"]  # what install PROVISIONS
        rp = ROOT / cell.get("_path", "") / "recipe.yaml"
        repo = ""
        if rp.exists():
            import yaml as _y

            repo = ((_y.safe_load(rp.read_text()) or {}).get("serving") or {}).get("model_repo") or ""
        dl = by_repo.get(install._norm_repo(repo), "")  # what DOWNLOADS
        if not (mount == spec_name == dl):
            diverged.append(f"{install.cell_id(cell)}: mount={mount} provision={spec_name} download={dl}")
    tag = prof.get("MODEL_CACHE_PVC")
    check(
        f"[{tag}] {len(cells)} cells: mount == provision == download",
        not diverged,
        "; ".join(diverged[:3]),
    )
    check(
        f"[{tag}] every cell resolved to a non-empty claim",
        all(install.resolve_cache_claim(c, prof)[0] for c in cells),
    )

# The per-model override actually ROUTES (not a no-op that silently collapses onto the default).
_qa = PROFILES[0]
_nem = [c for c in install.gpu_matching_cells(catalog, _qa) if c.get("model") == "nemotron-ultra-nvfp4"]
check(
    "per-model override routes the 3 nemotron cells to their own claim",
    bool(_nem) and all(install.resolve_cache_claim(c, _qa)[0] == "nemotron-ultra-nvfp4-cache" for c in _nem),
    str(len(_nem)),
)

# ---------------------------------------------------------------------------
# 4. NO RECIPE MAY NAME A CLAIM
# ---------------------------------------------------------------------------
print("\n-- 4. requires.cache.name is rejected --")

on_disk = [
    str(p.relative_to(ROOT))
    for p in (ROOT / "recipes").glob("**/recipe.yaml")
    if re.search(r"cache:\s*\{[^}]*\bname\s*:", p.read_text())
]
check(
    "no on-disk recipe declares requires.cache.name",
    not on_disk,
    "; ".join(on_disk[:3]),
)

schema = (ROOT / "schema" / "envelope.yaml").read_text()
_cache_block = schema.split("          cache:", 1)[-1].split("          secrets:", 1)[0]
check(
    "schema/envelope.yaml drops the `name` property from requires.cache",
    not re.search(r"^\s{14}name:", _cache_block, re.M),
)
check(
    "schema/envelope.yaml keeps additionalProperties:false (so `name` is REJECTED, not ignored)",
    "additionalProperties: false" in _cache_block,
)

errs = install.validate_cache_config(
    [_cell("nemotron-ultra-nvfp4", {"name": "recipe-named-cache"})],
    {"MODEL_CACHE_PVC": "profile-cache"},
    "somecluster",
)
check(
    "validate_cache_config REFUSES a cell carrying requires.cache.name",
    bool(errs),
    str(errs),
)
check(
    "...and names the offending value + the profile key to use instead",
    any("recipe-named-cache" in e and "MODEL_CACHE_PVC" in e for e in errs),
    str(errs),
)

# THE 31-CELL QA CASE: an empty profile var must fail the gate LOUDLY, naming the exact fix.
qa_cells = install.gpu_matching_cells(catalog, {"GPU_PRODUCT": "NVIDIA-B200"})
qa_errs = install.validate_cache_config(qa_cells, {"NAMESPACE": "n", "MODEL_CACHE_PVC": ""}, "qa-profile")
check(
    "MODEL_CACHE_PVC='' → the gate FAILS (pre-download, not after ~300 GiB)",
    bool(qa_errs),
)
check(
    "...one error per MODEL, not per cell (3 models, not 31 lines of noise)",
    len(qa_errs) == 3,
    f"{len(qa_errs)}: {qa_errs[:1]}",
)
check(
    "...and each names MODEL_CACHE_PVC and the profile file",
    all("MODEL_CACHE_PVC" in e and "qa-profile.env" in e for e in qa_errs),
    str(qa_errs[:1]),
)
check(
    "a correctly-configured profile passes the gate cleanly",
    install.validate_cache_config(qa_cells, PROFILES[0], "qa") == [],
)

# ---------------------------------------------------------------------------
# 5. EVERY SHELL CONSUMER RESOLVES BEFORE IT SUBSTITUTES
# ---------------------------------------------------------------------------
print("\n-- 5. every ${MODEL_CACHE_PVC} consumer calls the resolver --")

# A script that envsubsts ${MODEL_CACHE_PVC} into a manifest without resolving first is the ORIGINAL bug.
# Enumerate them from the source rather than from a hand-maintained list, so a NEW consumer added later
# fails this test instead of silently reintroducing the divergence.
SCRIPTS = ROOT / "scripts"

# THIS SECTION HAD A BLIND SPOT, and it was the escape hatch rather than the scan. The comment above says
# consumers are enumerated FROM SOURCE so a new one cannot slip through — and then a hand-maintained
# `EXEMPT = {...}` name list reintroduced exactly the failure mode that enumeration exists to prevent.
# dryrun.sh sat in it under the rationale "only PRINTS the value (never applies)", which was FALSE: it
# envsubsts $MODEL_CACHE_PVC into bench-job.yaml and (via its auto-whitelist) server.yaml, then
# kubeconform-validates the result. The rationale conflated "does not kubectl apply" with "does not
# substitute" — but the invariant is about SUBSTITUTION, so dry-run validated manifests against a claim the
# deploy would never use. An exemption asserted in a comment is not checked by anything; the fix is to
# derive exemption from BEHAVIOUR, so a script cannot be waved through by a claim about itself that stops
# being true.
#
# THE RULE: a script that can put MODEL_CACHE_PVC's value INTO A MANIFEST must resolve it first. Note the
# property is "can render the claim", NOT "runs envsubst" — my first attempt at this rule used envsubst as
# the proxy and promptly mis-filed stage-dataset.sh as harmless, because they emit
# `claimName: ${MODEL_CACHE_PVC}` with a plain `echo` heredoc instead. Same mistake as the one being fixed:
# a proxy standing in for the property. Both rendering routes are therefore tested.
# Sourced libraries (_-prefixed) are the plumbing callers invoke AFTER resolving, excluded structurally.
_PVC_REF = re.compile(r"\$\{?MODEL_CACHE_PVC\}?(?![A-Z0-9_])")  # not _OVERRIDE, not _PVCS


def _live(txt: str) -> str:
    """Non-comment lines only. A comment cannot resolve, render, or mount anything."""
    return "\n".join(ln for ln in txt.splitlines() if not ln.strip().startswith("#"))


def _resolves(txt: str) -> bool:
    # Either route counts: the shell helper, or asking install.py / model_cache.py directly.
    # MUST read live code — stage-dataset.sh carries a COMMENT mentioning `--resolve-cache`, so matching
    # raw text let a script pass on its own documentation. That is the same mistake as the exemption this
    # section is being fixed for: a claim about behaviour standing in for the behaviour.
    live = _live(txt)
    return "llmb::resolve_model_cache_pvc" in live or "--resolve-cache" in live or "model_cache.py" in live


def _renders(txt: str) -> bool:
    # envsubst substitution OR direct interpolation into an emitted manifest.
    live = _live(txt)
    return "envsubst" in live or "claimName" in live


import io as _io  # noqa: E402
import tokenize as _tok  # noqa: E402


def _py_code(path) -> list[str]:
    """A python file's LIVE CODE, line by line — comments and MULTI-LINE strings blanked out.

    PROSE IS THE TRAP. Every module in this change set explains, in its docstring, the exact anti-pattern it
    was fixed to remove (`or prof.get("MODEL_CACHE_PVC", "")`), so a startswith('#') filter reports the
    explanation as the offence. Tokenizing is the only way to tell a code expression from a sentence about
    one. SINGLE-LINE strings are KEPT deliberately: the writer sites this scans are subscripts and dict
    keys — `prof["MODEL_CACHE_PVC"] = …` — so blanking every string would blank the thing being looked for.
    An unparseable file returns [] (report nothing) rather than a wrong answer."""
    src = path.read_text()
    out = src.splitlines()
    try:
        for t in _tok.generate_tokens(_io.StringIO(src).readline):
            if t.type == _tok.COMMENT or (t.type == _tok.STRING and t.end[0] != t.start[0]):
                for ln in range(t.start[0], t.end[0] + 1):
                    out[ln - 1] = out[ln - 1][: t.start[1]] if ln == t.start[0] else ""
    except Exception:
        return []
    return out


def _live_any(path) -> str:
    """Live code of a .py or .sh file — comments (and python prose blocks) removed."""
    return "\n".join(_py_code(path)) if path.suffix == ".py" else _live(path.read_text())


consumers, unresolved, non_rendering, libraries = [], [], [], []
for f in sorted(SCRIPTS.glob("*.sh")):
    txt = f.read_text()
    # Ignore mentions that live only in comments — a comment cannot render anything.
    if not _PVC_REF.search(_live(txt)):
        continue
    if f.name.startswith("_"):  # sourced library — checked separately below, NOT waved through
        libraries.append(f.name)
        continue
    if not _renders(txt):  # cannot put the claim into a manifest at all
        non_rendering.append(f.name)
        continue
    consumers.append(f.name)
    if not _resolves(txt):
        unresolved.append(f.name)

check(
    f"all {len(consumers)} manifest-rendering consumers resolve first: {', '.join(consumers)}",
    not unresolved,
    f"NOT resolving: {', '.join(unresolved)}",
)
check(
    "the consumer scan is non-vacuous (found the known renderers)",
    {"sweep.sh", "submit.sh", "stage-dataset.sh", "dryrun.sh"} <= set(consumers),
    str(consumers),
)

# deploy.sh does NOT appear above and that is correct — but NOT for the reason previously written here.
# The old comment said it "delegates rendering to _lib.sh"; it does not. deploy.sh renders the serving
# manifests itself, with an envsubst whitelist DERIVED from each file
# (`wl=$(grep -oE '\$\{[A-Z_][A-Z0-9_]*\}' "$f" ...)`), so it never names MODEL_CACHE_PVC in its own text
# and the _PVC_REF scan cannot see it. That is a scan limitation, not a delegation — and a rationale about
# behaviour that is not derived from the behaviour is exactly what let dryrun.sh through. Check it
# explicitly, and check the reason: it renders, and it resolves first.
_DEPLOY = (SCRIPTS / "deploy.sh").read_text()
check(
    "deploy.sh renders manifests itself with a DERIVED whitelist (not via _lib.sh)",
    "envsubst" in _live(_DEPLOY) and "grep -oE" in _live(_DEPLOY) and "_lib.sh" not in _live(_DEPLOY),
    "if deploy.sh stops deriving its own whitelist, this exemption's basis has changed",
)
for _entry in ("deploy.sh", "sweep.sh"):
    check(
        f"{_entry} (primary deploy path) calls the resolver",
        _resolves((SCRIPTS / _entry).read_text()),
        _entry,
    )
# dryrun.sh is the regression this section missed for real. Pin it as a SUBSTITUTING consumer explicitly,
# so it can never drift back into being treated as print-only.
check(
    "dryrun.sh is classified as a rendering consumer (it renders + kubeconform-validates)",
    "dryrun.sh" in consumers,
    f"consumers={consumers} non_rendering={non_rendering}",
)
# Both rendering ROUTES must stay in scope: envsubst (deploy/sweep) and echo-emitted claimName (stage-*).
check(
    "echo-emitted claimName scripts are in scope too (not just envsubst ones)",
    {"stage-dataset.sh"} <= set(consumers),
    f"consumers={consumers} non_rendering={non_rendering}",
)
# And the exemption that remains is EARNED, not asserted: anything called non-substituting must really
# contain no envsubst at all.
for _ns in non_rendering:
    check(
        f"{_ns} is exempt only because it genuinely cannot render a claim",
        not _renders((SCRIPTS / _ns).read_text()),
        _ns,
    )

lib = (SCRIPTS / "_model_cache.sh").read_text()
check(
    "_model_cache.sh exists and is fail-closed (no silent MODEL_CACHE_PVC fallback)",
    "return 1" in lib and "MODEL_CACHE_PVC_OVERRIDE" in lib,
)
check(
    "_model_cache.sh routes through model_cache.py (one implementation, not two)",
    "model_cache.py" in lib and "resolve" in lib,
)
check(
    "install.py RE-EXPORTS the resolver rather than defining a second copy",
    "from model_cache import (" in (SCRIPTS / "install.py").read_text()
    and "def resolve_cache_claim(" not in (SCRIPTS / "install.py").read_text(),
)
check(
    "model_cache.py is dependency-light (importable in a sandbox root: stdlib + yaml only)",
    not any(
        m in (SCRIPTS / "model_cache.py").read_text()
        for m in (
            "import install",
            "import profile_resolver",
            "import capability_registry",
        )
    ),
)
# The shell lib validates the resolver's output before exporting it -- a stderr line concatenated onto the
# claim name (rc still 0) once produced a two-line `claimName:` that envsubst wrote straight into a manifest.
_cmd = [line for line in lib.splitlines() if "model_cache.py" in line and "out=" in line]
check(
    "_model_cache.sh captures STDOUT ONLY (never 2>&1 into the value)",
    len(_cmd) == 1 and "2>&1" not in _cmd[0],
    str(_cmd),
)
check(
    "_model_cache.sh validates the claim is an RFC-1123 name before exporting",
    "llmb::_is_pvc_name" in lib,
    lib,
)
# The OVERRIDE is validated by the SAME predicate. It used to `return 0` before reaching the whitelist, so
# the one input that comes from OUTSIDE the profile — an env var any caller can set — was the only value
# that reached `claimName:` unchecked.
_ovr_block = lib.split("MODEL_CACHE_PVC_OVERRIDE:-", 1)[-1].split("fi", 1)[0]
check(
    "_model_cache.sh validates MODEL_CACHE_PVC_OVERRIDE through the same predicate",
    "llmb::_is_pvc_name" in _ovr_block,
    _ovr_block,
)
# BEHAVIOURAL, not textual: actually run the shell predicate. A PVC name is a DNS SUBDOMAIN — `cache.v2` is
# legal and the old `*[!a-z0-9-]*` whitelist rejected it, which would have aborted every deploy on a cluster
# that named its claim that way. And the character sets must be ENUMERATED, not `a-z` ranges: under
# en_US.UTF-8 collation `[a-z]` also matches uppercase, so the old form accepted `MY-CACHE`.
import subprocess as _sp  # noqa: E402

_PVC_CASES = [
    ("glm5-fp8-model-cache", 0),
    ("cache.v2", 0),
    ("a", 0),
    ("1", 0),
    ("bad name!", 1),
    ("-lead", 1),
    ("trail-", 1),
    (".dot", 1),
    ("dot.", 1),
    ("MY-CACHE", 1),
    ("a_b", 1),
    ("", 1),
    ("two\nlines", 1),
]
_pvc_fails = []
for _nm, _want in _PVC_CASES:
    _rc = _sp.run(
        [
            "bash",
            "-c",
            f'. "{SCRIPTS / "_model_cache.sh"}"; llmb::_is_pvc_name "$1"',
            "_",
            _nm,
        ],
        env={"PATH": "/usr/bin:/bin", "LC_ALL": "en_US.UTF-8"},
    ).returncode
    if (_rc != 0) != bool(_want):
        _pvc_fails.append(f"{_nm!r} rc={_rc} want={'reject' if _want else 'accept'}")
check(
    "the PVC-name predicate accepts DNS SUBDOMAINS (dots) and rejects uppercase/garbage",
    not _pvc_fails,
    "; ".join(_pvc_fails),
)


# ---------------------------------------------------------------------------
# 5b. SOURCED LIBRARIES are not structurally exempt — the exemption must be EARNED
# ---------------------------------------------------------------------------
# The scan above skips `_`-prefixed files because a library "runs after the caller has resolved". That is a
# claim about CALLERS, and it was checked by nothing — so scripts/_lib.sh was invisible while it (a) applies
# MODEL_CACHE_PVC_OVERRIDE with no resolution at all and (b) whitelists $MODEL_CACHE_PVC inside
# llmb::render_manifests. Both are dead today, which is the only reason it is not a live bug, and "dead
# today" is precisely the kind of fact that must be DERIVED rather than asserted in a comment.
print("\n-- 5b. a library's exemption is earned by having no caller, not by its filename --")


def _shell_functions(txt: str) -> dict:
    """{name: body} for `name() {` / `ns::name() {` blocks closed by a column-0 `}`."""
    out, cur, buf = {}, None, []
    for ln in txt.splitlines():
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_:]*)\(\)\s*\{", ln)
        if m:
            cur, buf = m.group(1), []
            continue
        if cur is not None:
            if ln.startswith("}"):
                out[cur] = "\n".join(buf)
                cur = None
            else:
                buf.append(ln)
    return out


# SHELL ONLY, and that is a real limit stated rather than glossed: a shell function cannot be invoked
# except from a process that SOURCED the library, so the caller set is the shell scripts. (Scanning .py too
# made this file its own caller — it names llmb::render_manifests in an assertion message — which is a false
# positive that would have masked the property being asserted.)
_TREE = sorted(SCRIPTS.glob("*.sh"))
for _libname in libraries:
    _libtxt = (SCRIPTS / _libname).read_text()
    if _resolves(_libtxt):
        check(f"{_libname}: touches the claim AND resolves it", True)
        continue
    _touching = {n: b for n, b in _shell_functions(_libtxt).items() if _PVC_REF.search(_live(b))}
    check(
        f"{_libname}: every claim-touching part is inside a named function (so callers are findable)",
        bool(_touching) or not _PVC_REF.search(_live(_libtxt)),
        f"live ${{MODEL_CACHE_PVC}} outside any function in {_libname}",
    )
    for _fn, _body in sorted(_touching.items()):
        # A CALLER IS A FILE THAT SOURCES THE LIBRARY AND NAMES THE FUNCTION — both, because a shell
        # function cannot be invoked without being sourced. Matching the name alone made this very test
        # its own "caller" (it names llmb::render_manifests in an assertion message), which would have hidden
        # the property being asserted behind a permanent failure. Read off live code, never a filename rule.
        _callers = sorted(
            p.name
            for p in _TREE
            if p.name != _libname
            and _libname in _live_any(p)
            and re.search(rf"(?<![\w:]){re.escape(_fn)}\b", _live_any(p))
        )
        # THE EXEMPTION, EARNED. A library function that hands out a claim without resolving is safe only
        # while every caller either resolves itself or never renders the claim. Both halves are read off
        # the callers' source, so the day someone renders from one of these it fails here.
        #   _lib.sh::llmb::load_env      — applies MODEL_CACHE_PVC_OVERRIDE straight onto the profile value,
        #                                  no resolution, no RFC-1123 validation. One caller:
        #                                  fetch_results.sh, which mounts the ARTIFACTS pvc and never
        #                                  references the claim.
        #   _lib.sh::llmb::render_manifests — whitelists $MODEL_CACHE_PVC for envsubst. ZERO callers today,
        #                                  which is the only reason it is not a live second resolution rule.
        _unsafe = []
        for _c in _callers:
            _ctxt = (SCRIPTS / _c).read_text()
            if _resolves(_ctxt):
                continue  # caller resolves before/after: fine
            if _PVC_REF.search(_live(_ctxt)) or _renders(_ctxt) and "MODEL_CACHE_PVC" in _live(_ctxt):
                _unsafe.append(_c)
        check(
            f"{_libname}::{_fn} hands out an UNRESOLVED claim, so every caller must resolve or never "
            f"render it ({len(_callers)} caller(s): {', '.join(_callers) or 'none'})",
            not _unsafe,
            f"unsafe caller(s): {', '.join(_unsafe)}",
        )
        if _fn == "llmb::render_manifests":
            check(
                "_lib.sh::llmb::render_manifests (a SECOND rendering rule with no resolver) still has " "zero callers",
                not _callers,
                f"called by: {', '.join(_callers)}",
            )


# ---------------------------------------------------------------------------
# 5c. PYTHON PRODUCERS — a module that WRITES the claim must take it from the resolver
# ---------------------------------------------------------------------------
# The scan was `SCRIPTS.glob("*.sh")` only, so the two places that put a claim name into a profile dict for
# rendering — install.phase_c_download and preflight.main — were outside it entirely. Both are correct
# today; neither was checked.
print("\n-- 5c. every python site that WRITES MODEL_CACHE_PVC takes it from the resolver --")

_RESOLVER_SYMS = (
    "resolve_cache_claim",
    "cache_pvc_by_repo",
    "cache_by_repo",
    "--resolve-cache",
)
_WRITE_RE = re.compile(r'\["MODEL_CACHE_PVC"\]\s*=\s*(\S+)|"MODEL_CACHE_PVC":\s*([A-Za-z_][\w.]*)')
_py_sites, _py_bad = [], []
for _p in sorted(SCRIPTS.glob("*.py")):
    if _p.name.startswith("selftest_") or _p.name in (
        "model_cache.py",
        "wizard_init.py",
    ):
        continue  # tests build fixtures; model_cache IS the rule; wizard_init AUTHORS the profile
    _lines = _py_code(_p)
    _code = "\n".join(_lines)
    for _i, _ln in enumerate(_lines):
        _m = _WRITE_RE.search(_ln)
        if not _m:
            continue
        _py_sites.append(f"{_p.name}:{_i + 1}")
        # TRACE THE SYMBOL, not the neighbourhood: the value assigned must itself be bound from a resolver
        # call somewhere in this module. (preflight's binding is ~20 lines above its use, behind a comment
        # block, so a fixed proximity window silently mis-scores it.)
        _sym = (_m.group(1) or _m.group(2) or "").strip().rstrip(",}")
        _bound = re.search(
            rf"(?<![\w.]){re.escape(_sym)}\b[^\n=]*=\s*[^\n]*" rf"({'|'.join(re.escape(s) for s in _RESOLVER_SYMS)})",
            _code,
        )
        if not (_bound or any(s in _ln for s in _RESOLVER_SYMS)):
            _py_bad.append(f"{_p.name}:{_i + 1}: {_ln.strip()[:90]} (symbol {_sym!r} not from the resolver)")
check(
    f"all {len(_py_sites)} python writer(s) derive the claim from the resolver: {', '.join(_py_sites)}",
    not _py_bad,
    "; ".join(_py_bad),
)
check(
    "the python scan is non-vacuous (it found the known writers)",
    any(s.startswith("install.py:") for s in _py_sites) and any(s.startswith("preflight.py:") for s in _py_sites),
    str(_py_sites),
)
# And NO module may fall back to the raw profile value. `or prof.get("MODEL_CACHE_PVC", "")` is the original
# bug in one expression: on a per-model-key profile the download takes the GLOBAL claim while deploy.sh
# mounts the override.
_fb_re = re.compile(r'or\s+prof(?:ile)?\.get\(\s*["\']MODEL_CACHE_PVC["\']')
for _p in sorted(SCRIPTS.glob("*.py")):
    if _p.name.startswith("selftest_"):
        continue
    _bad_fb = [f"{_i + 1}: {ln.strip()[:80]}" for _i, ln in enumerate(_py_code(_p)) if _fb_re.search(ln)]
    check(
        f"{_p.name}: no silent fallback to the raw profile MODEL_CACHE_PVC",
        not _bad_fb,
        "; ".join(_bad_fb[:2]),
    )

# LAZY RESOLUTION must be derived from the MANIFEST, never from a lane name. submit.sh drives
# lanes whose Job may mount no model cache at all, so an unconditional resolve took those lanes
# away entirely from a profile with no cache configured.
for _s, _mf in (("submit.sh", "$BENCH"),):
    _txt = _live((SCRIPTS / _s).read_text())
    check(
        f"{_s} resolves only when the manifest it substitutes into references the claim",
        re.search(rf'grep -q .MODEL_CACHE_PVC. "{re.escape(_mf)}"', _txt) is not None,
        _txt[:200],
    )
    check(
        "...and it still resolves (fail-closed) when it does",
        "llmb::resolve_model_cache_pvc" in _txt,
    )

# ---------------------------------------------------------------------------
# 6. ABSENCE IS NOT ZERO
# ---------------------------------------------------------------------------
print("\n-- 6. a failed probe reports UNKNOWN, never 'none found' --")


def _krun_broken(args, timeout=30):
    return (1, "", "error: You must be logged in to the server (Unauthorized)")


cands, perr = install.list_cache_candidates("ns", _krun_broken)
check(
    "list_cache_candidates returns a probe_error when the listing fails",
    cands == [] and bool(perr),
    perr,
)
advice = install.render_cache_candidates_advice(cands, perr, [_cell("m")], "c")
check(
    "the advice says 'could not look', NOT 'none found'",
    "UNKNOWN" in advice
    and "not 'there are none'" in advice.replace("NOT", "not").lower()
    or ("UNKNOWN" in advice and "none" not in advice.split("Add to")[0].replace("UNKNOWN", "")),
    advice[:300],
)


def _krun_empty(args, timeout=30):
    return (0, '{"items": []}', "")


cands2, perr2 = install.list_cache_candidates("ns", _krun_empty)
check(
    "a SUCCESSFUL listing with no caches reports none (and no probe_error)",
    cands2 == [] and perr2 == "",
    f"{cands2} {perr2}",
)

# The ≥1Ti floor must no longer HIDE a smaller dedicated cache (it hid the 800Gi nemotron claim).
_J = (
    '{"items": [' + '{"metadata": {"name": "nemotron-ultra-nvfp4-cache", "labels": {}},'
    ' "spec": {"accessModes": ["ReadWriteMany"], "resources": {"requests": {"storage": "800Gi"}}},'
    ' "status": {"phase": "Bound", "capacity": {"storage": "800Gi"}}},'
    + '{"metadata": {"name": "glm5-fp8-model-cache", "labels": {}},'
    ' "spec": {"accessModes": ["ReadWriteMany"], "resources": {"requests": {"storage": "1200Gi"}}},'
    ' "status": {"phase": "Bound", "capacity": {"storage": "1200Gi"}}}]}'
)
cands3, _ = install.list_cache_candidates("ns", lambda a, timeout=30: (0, _J, ""))
names3 = [c["name"] for c in cands3]
check(
    "the 800Gi dedicated cache is VISIBLE (no ≥1Ti visibility floor)",
    "nemotron-ultra-nvfp4-cache" in names3,
    str(names3),
)
check(
    "...alongside the 1200Gi one, largest-first",
    names3[0] == "glm5-fp8-model-cache",
    str(names3),
)

# --adopt-cache PLANNING: prefer a claim that already holds the model, else the smallest that fits.
stamped = [
    {
        "name": "nemotron-ultra-nvfp4-cache",
        "capacity_gib": 800,
        "phase": "Bound",
        "access_modes": ["ReadWriteMany"],
        "labels": {"llmb.nvidia.com/model-name": "nemotron-ultra-nvfp4"},
    },
    {
        "name": "glm5-fp8-model-cache",
        "capacity_gib": 1200,
        "phase": "Bound",
        "access_modes": ["ReadWriteMany"],
        "labels": {},
    },
]
keys, notes = install.plan_cache_adoption(
    [
        _cell(
            "nemotron-ultra-nvfp4",
            path="recipes/llm-perf/nemotron_ultra_disagg/sglang_dynamo/10k16k/"
            "nemotron-ultra-nvfp4-b200-sglang-dynamo14-10k16k-1p1d",
        )
    ],
    stamped,
    {"MODEL_CACHE_PVC": "glm5-fp8-model-cache"},
)
check(
    "--adopt-cache prefers the claim that ALREADY holds the model",
    keys.get("MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4") == "nemotron-ultra-nvfp4-cache",
    f"{keys} {notes}",
)
check(
    "--adopt-cache never overwrites an already-configured model",
    install.plan_cache_adoption([_cell("glm5-fp8")], stamped, {"MODEL_CACHE_PVC": "glm5-fp8-model-cache"})[0] == {},
    "should be empty",
)

# apply_profile_keys: replace in place, append the rest, preserve comments.

with tempfile.TemporaryDirectory() as td:
    p = Path(td) / "c.env"
    p.write_text('# a comment worth keeping\nNAMESPACE="bench"\nMODEL_CACHE_PVC=""\n')
    install.apply_profile_keys(
        p,
        {
            "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
            "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4": "nemotron-ultra-nvfp4-cache",
        },
    )
    out = p.read_text()
    check(
        "apply_profile_keys replaces an existing key in place",
        'MODEL_CACHE_PVC="glm5-fp8-model-cache"' in out and 'MODEL_CACHE_PVC=""' not in out,
        out,
    )
    check(
        "apply_profile_keys appends a new key",
        "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4=" in out,
        out,
    )
    check(
        "apply_profile_keys preserves comments + unrelated keys",
        "# a comment worth keeping" in out and 'NAMESPACE="bench"' in out,
        out,
    )

# cache_pvc_by_repo must REFUSE, never fall back. The old `or prof.get("MODEL_CACHE_PVC","")` restored the
# original bug on a per-model-override profile: the cell fell back to the GLOBAL claim for the download
# while deploy.sh resolved the OVERRIDE for the mount.
_bad_map, _bad_errs = install.cache_pvc_by_repo(
    [
        {
            "model": "glm5-fp8",
            "_path": "recipes/llm-perf/Glm5/B200_k8s/Agg/1k_1k/glm5-fp8-b200-sglang15-agg-c4-1024",
            "requires": {},
        }
    ],
    {"NAMESPACE": "n"},
)
check(
    "cache_pvc_by_repo REFUSES an unresolved cell (no silent global fallback)",
    _bad_map == {} and bool(_bad_errs),
    f"{_bad_map} {_bad_errs}",
)
_unread_map, _unread_errs = install.cache_pvc_by_repo(
    [{"model": "m", "_path": "does/not/exist", "requires": {}}],
    {"MODEL_CACHE_PVC": "c"},
)
check(
    "cache_pvc_by_repo reports an UNREADABLE recipe.yaml (not a silent skip)",
    bool(_unread_errs),
    f"{_unread_map} {_unread_errs}",
)

# A typo'd per-model key must not be a silent no-op.
_typo = install.validate_cache_config(
    [_cell("nemotron-ultra-nvfp4")],
    {
        "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
        "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP": "oops",
    },
    "c",
)
check(
    "a typo'd MODEL_CACHE_PVC_* key is reported, not silently ignored",
    any("NVFP" in e and "no-op" in e for e in _typo),
    str(_typo),
)

# apply_profile_keys must rewrite EVERY assignment (sourcing a .env is last-wins).
with tempfile.TemporaryDirectory() as _td:
    _p = Path(_td) / "dup.env"
    _p.write_text('MODEL_CACHE_PVC="old-a"\nNAMESPACE="n"\nMODEL_CACHE_PVC="old-b"\n')
    install.apply_profile_keys(_p, {"MODEL_CACHE_PVC": "new"})
    _last = [ln for ln in _p.read_text().splitlines() if ln.startswith("MODEL_CACHE_PVC=")][-1]
    check(
        "apply_profile_keys rewrites a DUPLICATE key too (last-wins would keep the old value)",
        _last == 'MODEL_CACHE_PVC="new"',
        _p.read_text(),
    )

# ---------------------------------------------------------------------------
# 7. WHERE a cache-mounting pod may run  (blocker: the claim mounts on 2 of 11 nodes)
# ---------------------------------------------------------------------------
print("\n-- 7. every cache-mounting pod gets the same placement --")

import model_cache as mc  # noqa: E402

check(
    "parse_node_selector handles the BENCH_NODE_SELECTOR syntax",
    mc.parse_node_selector('nvidia.com/gpu.present: "true"') == {"nvidia.com/gpu.present": "true"},
)
check(
    "parse_node_selector handles multiple terms",
    mc.parse_node_selector("kubernetes.io/arch: amd64, pool: gpu") == {"kubernetes.io/arch": "amd64", "pool": "gpu"},
)
check(
    "parse_node_selector: garbage -> {} (no constraint), never a PARTIAL selector",
    mc.parse_node_selector("nonsense") == {},
)

_sel, _tol, _src = mc.cache_pod_placement({"MODEL_CACHE_NODE_SELECTOR": 'nvidia.com/gpu.present: "true"'})
_TOL_EXPECT = [
    {"operator": "Exists", "effect": "NoSchedule"},
    {"operator": "Exists", "effect": "PreferNoSchedule"},
]
check(
    "cache_pod_placement returns the selector + tolerates SCHEDULING taints",
    _sel == {"nvidia.com/gpu.present": "true"} and _tol == _TOL_EXPECT,
    f"{_sel} {_tol}",
)
check(
    "cache_pod_placement tolerates taints even with NO selector (the only mountable nodes are tainted)",
    mc.cache_pod_placement({})[1] == _TOL_EXPECT,
    str(mc.cache_pod_placement({})[1]),
)
# ...and NOT NoExecute. A bare {"operator":"Exists"} tolerates every taint, which also SUPPRESSES the
# not-ready/unreachable NoExecute tolerations kubernetes defaults onto every pod — so a probe on a node that
# goes NotReady is never evicted, hangs to its wait budget and reports UNKNOWN. On this path UNKNOWN means
# "re-fetch hundreds of GiB" or "refuse the run", so the blanket form is not the cheap default it looks like.
check(
    "cache pods do NOT tolerate NoExecute (a NotReady node must still evict them)",
    all(t.get("effect") in ("NoSchedule", "PreferNoSchedule") for t in _tol)
    and not any(t == {"operator": "Exists"} for t in _tol),
    str(_tol),
)
check(
    "the shell-emitted stage pods carry the SAME tolerations (one placement rule, not two)",
    all(
        "operator: Exists, effect: NoSchedule" in (SCRIPTS / _s).read_text()
        and "operator: Exists }]" not in (SCRIPTS / _s).read_text()
        for _s in ("stage-dataset.sh",)
    ),
)

# The download Job must carry BOTH halves, and must render byte-identically when the knob is unset.
import yaml as _y  # noqa: E402


def _job(prof):
    return [
        d
        for d in _y.safe_load_all(
            install.render_download_job(
                {
                    "model_repo": "nvidia/N",
                    "model_revision": "a" * 40,
                    "model_name": "n",
                },
                prof,
                install.DOWNLOAD_TMPL,
            )
        )
        if d and d.get("kind") == "Job"
    ][0]["spec"]["template"]["spec"]


_with = _job(
    {
        "NAMESPACE": "n",
        "MODEL_CACHE_PVC": "c",
        "MODEL_CACHE_NODE_SELECTOR": 'nvidia.com/gpu.present: "true"',
    }
)
_without = _job({"NAMESPACE": "n", "MODEL_CACHE_PVC": "c"})
check(
    "download Job is PINNED when the profile sets a selector",
    _with.get("nodeSelector") == {"nvidia.com/gpu.present": "true"},
    str(_with.get("nodeSelector")),
)
check(
    "download Job still tolerates taints (the mountable nodes are tainted here)",
    _with.get("tolerations") == _TOL_EXPECT,
    str(_with.get("tolerations")),
)
check(
    "download Job is UNCHANGED when the knob is unset (clusters that mount everywhere are unaffected)",
    _without.get("nodeSelector") is None,
)

# Every cache-mounting pod builder must consult the shared placement -- a new one that forgets is the bug.
_INS = (SCRIPTS / "install.py").read_text()
check(
    "install's free-space probe uses the shared placement",
    "def probe_pvc_free_gib(" in _INS and _INS.count("_mc.cache_pod_placement(prof)") >= 2,
)
check(
    "preflight's integrity mounter uses it too (it previously had NO tolerations at all)",
    "_mc.cache_pod_placement(prof)" in (SCRIPTS / "preflight.py").read_text(),
)
for _s in ("stage-dataset.sh",):
    check(
        f"{_s} pins its helper pod",
        "MODEL_CACHE_NODE_SELECTOR" in (SCRIPTS / _s).read_text(),
    )

# A failed mounter must say WHICH failure. These are three different operator actions.
check(
    "classify: FailedScheduling -> unschedulable",
    mc.classify_mounter_failure("Pending", "0/11 nodes are available: FailedScheduling")[0] == "unschedulable",
)
check(
    "classify: the real NFS refusal -> mount-failed",
    mc.classify_mounter_failure("Pending", "mount.nfs: rpc.statd is not running")[0] == "mount-failed",
)
check(
    "classify: FailedMount -> mount-failed",
    mc.classify_mounter_failure("Pending", "Warning FailedMount timed out")[0] == "mount-failed",
)
check(
    "classify: a create refusal -> rbac-denied",
    mc.classify_mounter_failure("", "", "pods is forbidden: User cannot create resource")[0] == "rbac-denied",
)
check(
    "classify: nothing conclusive -> timeout (not a guess)",
    mc.classify_mounter_failure("Running", "")[0] == "timeout",
)
check(
    "the placement hint NAMES the knob when it is unset",
    "MODEL_CACHE_NODE_SELECTOR" in mc.cache_placement_hint({}) and "cluster profile" in mc.cache_placement_hint({}),
)

# ---------------------------------------------------------------------------
# 8. Unverified state — weights present, completion unproven
# ---------------------------------------------------------------------------
print("\n-- 8. one completeness predicate, three states --")

_names = [f"m-{i:05d}.safetensors" for i in range(1, 114)]


def _facts(**kw):
    # SHAPED LIKE A REAL REPORT. cache_probe_script always emits SHARD_N/SHARD_MIN/SHARD_MAX alongside
    # SHARD_BYTES, so a fixture that omits them describes output the probe cannot produce — and it would
    # exercise the one path where every byte guard is inoperative (now UNKNOWN; see "the sizing pass
    # returned nothing" checks below).
    f = {
        "exists": True,
        "sentinel": False,
        "config_json": True,
        "index_json": True,
        "shard_count": 113,
        "required_shards": 113,
        "missing_shards": 0,
        "incomplete_files": 0,
        "index_total_bytes": 1000,
        "shard_bytes": 1100,
        "shard_files": 113,
        "shard_min_bytes": 9,
        "shard_max_bytes": 11,
    }
    f.update(kw)
    return f


# THE LIVE NEMOTRON CASE: 113/113 shards, NO .llmb_download_done directory at all.
check(
    "complete sharded model with NO sentinel -> COMPLETE (not a 330 GiB re-download)",
    mc.cache_completeness(_facts())[0] == mc.STATE_COMPLETE,
    str(mc.cache_completeness(_facts())),
)
check(
    "a missing shard -> INCOMPLETE",
    mc.cache_completeness(_facts(missing_shards=1))[0] == mc.STATE_INCOMPLETE,
)
check(
    "an .incomplete blob (hf's real partial marker) -> INCOMPLETE",
    mc.cache_completeness(_facts(incomplete_files=1))[0] == mc.STATE_INCOMPLETE,
)
check(
    "gross byte shortfall -> INCOMPLETE (the aggregate veto)",
    mc.cache_completeness(_facts(shard_bytes=10))[0] == mc.STATE_INCOMPLETE,
)
check(
    "weights present but unverified, no index, no sentinel -> PRESENT_UNVERIFIED (never ABSENT)",
    mc.cache_completeness(
        _facts(
            config_json=False,
            index_json=False,
            required_shards=0,
            index_total_bytes=None,
            shard_bytes=None,
        )
    )[0]
    == mc.STATE_PRESENT_UNVERIFIED,
)
check(
    "a probe that could not look -> UNKNOWN, never ABSENT",
    mc.cache_completeness({"probe_error": "pod did not schedule"})[0] == mc.STATE_UNKNOWN,
)
check(
    "nothing on disk -> ABSENT",
    mc.cache_completeness({"exists": False})[0] == mc.STATE_ABSENT,
)
check(
    "single-file model (no index) -> COMPLETE",
    mc.cache_completeness(
        _facts(
            index_json=False,
            required_shards=0,
            shard_count=1,
            index_total_bytes=None,
            shard_bytes=None,
        )
    )[0]
    == mc.STATE_COMPLETE,
)
# Regression: Nemotron has no central safetensors index; model-NNNNN-of-MMMMM.json filenames declare the
# complete 113-shard set. A quota failure left 41 files, and the old single-file fallback plus a stale
# sentinel incorrectly made that partial cache permanent. Physical shard-set evidence must win.
_partial_meta = _facts(
    sentinel=True,
    index_json=False,
    required_shards=0,
    shard_count=41,
    shard_files=41,
    metadata_total_shards=113,
    metadata_shard_files=41,
    metadata_distinct_totals=1,
    index_total_bytes=None,
    shard_bytes=410,
    shard_min_bytes=10,
    shard_max_bytes=10,
)
_pm_state, _pm_why = mc.cache_completeness(_partial_meta)
check(
    "41/113 filename-declared shards override a stale sentinel -> INCOMPLETE",
    _pm_state == mc.STATE_INCOMPLETE and "41" in _pm_why and "113" in _pm_why,
    f"{_pm_state}: {_pm_why}",
)
check(
    "partial filename-declared shard set is never sentinel-worthy",
    mc.sentinel_worthy(_partial_meta)[0] is False,
)
_complete_meta = _facts(
    index_json=False,
    required_shards=0,
    shard_count=113,
    shard_files=113,
    metadata_total_shards=113,
    metadata_shard_files=113,
    metadata_distinct_totals=1,
    index_total_bytes=None,
    shard_bytes=1130,
    shard_min_bytes=10,
    shard_max_bytes=10,
)
check(
    "113/113 filename-declared shards -> COMPLETE and sentinel-worthy",
    mc.cache_completeness(_complete_meta)[0] == mc.STATE_COMPLETE and mc.sentinel_worthy(_complete_meta)[0] is True,
)
_unindexed_multi = _facts(
    sentinel=False,
    index_json=False,
    required_shards=0,
    shard_count=41,
    shard_files=41,
    index_total_bytes=None,
    shard_bytes=410,
    shard_min_bytes=10,
    shard_max_bytes=10,
)
check(
    "an unindexed multi-file snapshot without proof is never misclassified as single-file",
    mc.cache_completeness(_unindexed_multi)[0] == mc.STATE_PRESENT_UNVERIFIED,
)
check(
    "a successful sentinel remains authoritative when there is no contradictory shard-set metadata",
    mc.cache_completeness({**_unindexed_multi, "sentinel": True})[0] == mc.STATE_COMPLETE,
)
_meta_report = mc.parse_cache_integrity_report(
    "EXISTS=1\nSENTINEL=1\nCONFIG=1\nINDEX=0\nSHARDS=41\nMETA_FILES=41\n" "META_TOTALS=1\nMETA_TOTAL=113\nSHARD_N=41\n"
)
check(
    "cache report parser carries filename-declared shard evidence",
    _meta_report["metadata_total_shards"] == 113 and _meta_report["metadata_shard_files"] == 41,
)
_probe_text = mc.cache_probe_script(".", "nvidia/Nemotron", "rev")
check(
    "cache probe extracts model-NNNNN-of-MMMMM metadata",
    all(key in _probe_text for key in ("META_FILES=", "META_TOTALS=", "META_TOTAL=")),
)

_download_template = install.DOWNLOAD_TMPL.read_text()
check(
    "download Job never trusts a sentinel as a reason to skip resume-capable verification",
    'if [ -f "${SENTINEL}" ]' not in _download_template,
)
check(
    "download Job still uses snapshot_download and writes its sentinel only afterward",
    _download_template.index("local_path = snapshot_download(") < _download_template.index('> "${SENTINEL}"'),
)

# Execute the emitted shell probe against the reported Nemotron failure shape, not only synthetic facts.
with tempfile.TemporaryDirectory() as _td:
    _cache = Path(_td) / "cache"
    _snap = _cache / mc.snapshot_dir(".", "nvidia/Nemotron", "rev")
    _snap.mkdir(parents=True)
    (_snap / "config.json").write_text("{}")
    for _i in range(1, 42):
        (_snap / f"model-{_i:05d}-of-00113.safetensors").write_bytes(b"x")
        (_snap / f"model-{_i:05d}-of-00113.json").write_text("{}")
    _sentinel = _cache / mc.sentinel_path(".", "rev")
    _sentinel.parent.mkdir(parents=True)
    _sentinel.write_text("stale")
    _script = mc.cache_probe_script(".", "nvidia/Nemotron", "rev").replace("/cache/", f"{_cache}/")
    _probe = subprocess.run(["sh", "-c", _script], capture_output=True, text=True, check=True)
    _live_facts = mc.parse_cache_integrity_report(_probe.stdout)
    _live_state, _live_why = mc.cache_completeness(_live_facts)
    check(
        "real shell probe detects the 41/113 cache as incomplete despite its stale sentinel",
        _live_state == mc.STATE_INCOMPLETE and "41" in _live_why and "113" in _live_why,
        f"{_live_state}: {_live_why}",
    )

check(
    "a bare shard COUNT never certifies (no config, no index) ",
    mc.cache_completeness(
        {
            "exists": True,
            "shard_count": 113,
            "config_json": False,
            "required_shards": 0,
            "shard_files": 113,
            "shard_min_bytes": 9,
            "shard_max_bytes": 11,
        }
    )[0]
    == mc.STATE_PRESENT_UNVERIFIED,
)
check(
    "...and it is never COMPLETE however the sizes came back",
    all(
        mc.cache_completeness(
            {
                "exists": True,
                "shard_count": 113,
                "config_json": False,
                "required_shards": 0,
                **_sz,
            }
        )[0]
        != mc.STATE_COMPLETE
        for _sz in ({}, {"shard_files": 0}, {"shard_files": 113})
    ),
)

# ── Incomplete indexed snapshot ────────────────────────────────────────────────────────
# INDEX=1 with REQ_SHARDS=0 is "the index file EXISTS and nothing was enumerated from it", NOT "this model
# has no index". Read as the latter, a 113-shard model whose shard set was never checked verdicted
# ('complete', 'config.json + 113 resolved shard(s), no shard index (single-file model)') — a false proof
# string. install then STAMPED it, cache_completeness treats a sentinel as conclusive, and the download
# old template also short-circuited on the same file: one probe hiccup survived re-installs,
# while the server crash-loops on a shard that is not there. index_json was parsed and read by nothing.
_unread_idx = _facts(
    required_shards=0,
    index_total_bytes=None,
    shard_bytes=None,
    shard_min_bytes=None,
    shard_max_bytes=None,
)
_ui_state, _ui_why = mc.cache_completeness(_unread_idx)
check(
    "index PRESENT but nothing enumerated from it -> NOT complete",
    _ui_state == mc.STATE_PRESENT_UNVERIFIED,
    f"{_ui_state}: {_ui_why}",
)
check(
    "...and the proof string no longer calls a 113-shard model 'single-file'",
    "single-file" not in _ui_why,
    _ui_why,
)
check(
    "...and it says the index IS present (the two situations are told apart)",
    "IS PRESENT" in _ui_why,
    _ui_why,
)
import capability_registry as _cr0  # noqa: E402

check(
    "...preflight refuses to start a run on it",
    _cr0.model_cache_verdict(_unread_idx) == _cr0.ABSENT,
    _cr0.model_cache_verdict(_unread_idx),
)
check(
    "...and it is NOT sentinel-worthy (a permanent record needs the strong proof)",
    mc.sentinel_worthy(_unread_idx)[0] is False,
    str(mc.sentinel_worthy(_unread_idx)),
)
check(
    "a genuine single-file model (NO index file) is still COMPLETE and IS sentinel-worthy",
    mc.cache_completeness(
        _facts(
            index_json=False,
            required_shards=0,
            shard_count=1,
            shard_files=1,
            index_total_bytes=None,
            shard_bytes=None,
        )
    )[0]
    == mc.STATE_COMPLETE
    and mc.sentinel_worthy(
        _facts(
            index_json=False,
            required_shards=0,
            shard_count=1,
            shard_files=1,
            index_total_bytes=None,
            shard_bytes=None,
        )
    )[0]
    is True,
)

# ── THE SIZING PASS RETURNED NOTHING ────────────────────────────────────────────────────────────────
# SHARDS>0 with no per-file sizes switched off the plausibility gate, the zero-byte veto AND the aggregate
# byte veto at once, and the verdict fell through to COMPLETE. Reproduced against a real hub-layout fixture
# in busybox 1.36.1 with `stat` made unavailable: the probe exits 0 and emits SHARDS=10 + zeros, because the
# awk END block runs on empty input. Both shapes must be UNKNOWN — the new empty one AND the old zeros, so
# a stale probe reading cannot slip through.
for _tag, _sz in (
    (
        "SHARD_N unset (post-fix probe)",
        {
            "shard_files": None,
            "shard_bytes": None,
            "shard_min_bytes": None,
            "shard_max_bytes": None,
        },
    ),
    (
        "SHARD_N=0 (legacy zero values)",
        {
            "shard_files": 0,
            "shard_bytes": 0,
            "shard_min_bytes": 0,
            "shard_max_bytes": 0,
        },
    ),
):
    _f = _facts(index_total_bytes=352284061280, **_sz)
    _s, _w = mc.cache_completeness(_f)
    check(
        f"113 shards, {_tag} -> UNKNOWN (not COMPLETE)",
        _s == mc.STATE_UNKNOWN,
        f"{_s}: {_w}",
    )
    check(f"...{_tag}: nothing gets stamped", mc.sentinel_worthy(_f)[0] is False)
# The probe itself must emit EMPTY, not 0, when it measured nothing — the parser's documented
# "unparseable numbers become None, NOT 0" was unreachable while the shell printed `n+0`.
_awk = [line for line in mc.cache_probe_script(".", "a/b", "r").splitlines() if "SHARD_N=" in line]
check(
    "the probe emits an EMPTY SHARD_N when it sized nothing (never 0)",
    any("SHARD_N=\\n" in line and "n+0" not in line for line in _awk),
    str(_awk),
)

# ONE function, not three opinions.
check(
    "capability_registry delegates to cache_completeness",
    "cache_completeness" in (SCRIPTS / "capability_registry.py").read_text(),
)
check(
    "preflight delegates its parser to model_cache",
    "_mc.parse_cache_integrity_report" in (SCRIPTS / "preflight.py").read_text(),
)
check(
    "preflight and install run the SAME probe script",
    "_mc.cache_probe_script" in (SCRIPTS / "preflight.py").read_text() and "_mc.cache_probe_script" in _INS,
)
import capability_registry as _cr  # noqa: E402

check(
    "preflight maps PRESENT_UNVERIFIED -> ABSENT (a run must not start on an unproven cache)",
    _cr.model_cache_verdict(
        _facts(
            config_json=False,
            index_json=False,
            required_shards=0,
            index_total_bytes=None,
            shard_bytes=None,
        )
    )
    == _cr.ABSENT,
)
check(
    "preflight maps a probe failure -> UNKNOWN (safe-degrade, never a false block)",
    _cr.model_cache_verdict({"probe_error": "x"}) == _cr.UNKNOWN,
)

# install VERIFIES then STAMPS -- never stamps on faith, and the STAMP bar is STRICTER than the verdict.
# A verdict is re-derived from the disk every probe, so a wrong one costs one run. A sentinel is believed
# as conclusive in cache_completeness until physical evidence contradicts it, and
# nothing deletes it, so a probe hiccup that produced a plausible COMPLETE would be frozen in permanently.
check(
    "install stamps only after proving COMPLETE",
    "if _st == _mc.STATE_COMPLETE and not plan_only:" in _INS and "stamp_download_sentinel" in _INS,
)
check(
    "...and every stamp call is gated on sentinel_worthy, not on the verdict alone",
    _INS.count("stamp_download_sentinel(") == _INS.count("_mc.sentinel_worthy(") + 1,  # +1 = the def
    f"stamp calls={_INS.count('stamp_download_sentinel(')} " f"worthy checks={_INS.count('_mc.sentinel_worthy(')}",
)
check(
    "sentinel_worthy is STRICTER than COMPLETE (it refuses an unread index and unsized shards)",
    mc.sentinel_worthy(_facts(required_shards=0))[0] is False
    and mc.sentinel_worthy(_facts(shard_files=None))[0] is False
    and mc.sentinel_worthy(_facts())[0] is True,
)
check(
    "...and a COMPLETE that rests only on an EXISTING sentinel adds nothing (nothing to write)",
    mc.sentinel_worthy(
        {
            "exists": True,
            "sentinel": True,
            "config_json": False,
            "shard_count": 0,
            "required_shards": 0,
        }
    )[0]
    is False,
)

# The aggregate byte veto KEEPS its power to demote, but must not assert a cause it cannot know: an index
# that OVER-declares metadata.total_size (tied/shared tensors) is equally consistent with a short total.
# Measured margins on the two real caches are +0.0020% (glm5-fp8) and +0.0070% (nemotron) — thin.
_short = mc.cache_completeness(_facts(index_total_bytes=1000, shard_bytes=900))
check(
    "a byte shortfall still DEMOTES (this is the only check that sees a non-hf truncation)",
    _short[0] == mc.STATE_INCOMPLETE,
    str(_short),
)
check(
    "...but the message names BOTH explanations instead of asserting 'truncated' as fact",
    "OVER-declare" in _short[1] and "truncated shard" in _short[1],
    _short[1],
)

# ---------------------------------------------------------------------------
# 9. Misplaced weights reconciliation
# ---------------------------------------------------------------------------
print("\n-- 9. weights stamped on another claim are NOTICED --")

_pvcs = [
    {
        "name": "nemotron-ultra-nvfp4-cache",
        "labels": {"llmb.nvidia.com/model-revision": "183968f87ae4"},
    },
    {
        "name": "glm5-fp8-model-cache",
        "labels": {"llmb.nvidia.com/model-name": "glm5-fp8"},
    },
]
_other, _why9 = mc.find_misplaced_weights(
    "nemotron-ultra-nvfp4",
    "glm5-fp8-model-cache",
    _pvcs,
    "183968f87ae4cedce3039313cac1fd43d112c578",
)
check(
    "resolving to the WRONG claim is detected via the revision stamp",
    _other == "nemotron-ultra-nvfp4-cache",
    f"{_other} {_why9}",
)
check(
    "resolving to the claim that holds it -> no complaint",
    mc.find_misplaced_weights("glm5-fp8", "glm5-fp8-model-cache", _pvcs)[0] == "",
)
check("install runs the reconcile before acting", "find_misplaced_weights" in _INS)

# The tracked EXAMPLE profile must not teach the wrong answer -- it is what a new operator copies.
_EX_PATH = ROOT / "cluster-profiles" / "example-gpu-cluster.env.example"
if _EX_PATH.exists():
    _exprof = mc.parse_profile_env(ROOT / "cluster-profiles" / "example-gpu-cluster.env.example")
    check(
        "example profile gives nemotron its own claim (else a ~330 GiB re-download that SUCCEEDS)",
        _exprof.get("MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4") == "nemotron-ultra-nvfp4-cache",
        str(_exprof.get("MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4")),
    )
    check(
        "example profile pins cache-mounting pods to mountable nodes",
        bool(mc.parse_node_selector(_exprof.get("MODEL_CACHE_NODE_SELECTOR", ""))),
        _exprof.get("MODEL_CACHE_NODE_SELECTOR"),
    )
    check(
        "example profile pins the BENCH pod too (it mounts the cache to read the staged trace)",
        bool(mc.parse_node_selector(_exprof.get("BENCH_NODE_SELECTOR", ""))),
        _exprof.get("BENCH_NODE_SELECTOR"),
    )
else:
    print("  SKIP  cluster-specific example-profile checks (not shipped externally)")
_TPL = (ROOT / "cluster-profiles" / "_template.env.example").read_text()
check(
    "template documents MODEL_CACHE_PVC_<MODEL_SLUG> as a first-class option",
    "MODEL_CACHE_PVC_EXAMPLE_MODEL" in _TPL and "per-model" in _TPL,
)
check("template ships MODEL_CACHE_NODE_SELECTOR", "MODEL_CACHE_NODE_SELECTOR=" in _TPL)

# The .env parser must agree with what `sh` sources -- deploy.sh sources the same file, and any
# disagreement is the download-vs-mount divergence again by a new route. An INLINE COMMENT on the claim
# line silently produced 'glm5-fp8-model-cache"   # default: holds ...' as the claim name.
with tempfile.TemporaryDirectory() as _td2:
    _pe = Path(_td2) / "p.env"
    _pe.write_text(
        'MODEL_CACHE_PVC="glm5-fp8-model-cache"   # holds GLM-5 + Qwen3\n'
        "BARE=value   # trailing comment\n"
        'MODEL_CACHE_PVC_X="a"\nMODEL_CACHE_PVC_X="b"\n'
    )
    _parsed = mc.parse_profile_env(_pe)
    check(
        "parse_profile_env strips an INLINE COMMENT on a quoted value (as sh does)",
        _parsed["MODEL_CACHE_PVC"] == "glm5-fp8-model-cache",
        repr(_parsed.get("MODEL_CACHE_PVC")),
    )
    check(
        "parse_profile_env strips an inline comment on a BARE value",
        _parsed["BARE"] == "value",
        repr(_parsed.get("BARE")),
    )
    check(
        "parse_profile_env is LAST-WINS on a duplicate key (as sourcing is)",
        _parsed["MODEL_CACHE_PVC_X"] == "b",
        repr(_parsed.get("MODEL_CACHE_PVC_X")),
    )

# Every GPU-matching cell on the example profile resolves to a claim, and nemotron to its own.
if _EX_PATH.exists():
    _cat = install.load_catalog()
    _cells_ex = install.gpu_matching_cells(_cat, _exprof)
    check(
        f"example profile resolves ALL {len(_cells_ex)} GPU-matching cells",
        install.validate_cache_config(_cells_ex, _exprof, "example-gpu-cluster") == [],
        str(install.validate_cache_config(_cells_ex, _exprof, "example-gpu-cluster"))[:200],
    )
    _nem_ex = [c for c in _cells_ex if c.get("model") == "nemotron-ultra-nvfp4"]
    check(
        "...and nemotron cells resolve to the claim that holds their weights",
        bool(_nem_ex)
        and all(install.resolve_cache_claim(c, _exprof)[0] == "nemotron-ultra-nvfp4-cache" for c in _nem_ex),
        str(len(_nem_ex)),
    )

else:
    print("  SKIP  cluster-specific whole-catalog cache checks (not shipped externally)")
# ---------------------------------------------------------------------------
# 10. SHARED-CLAIM SIZING counts each MODEL once, not each CELL
# ---------------------------------------------------------------------------
print("\n-- 10. a claim shared by N cells of one model sizes to ONE copy --")

# Shared-cache sizing counts each distinct model once, regardless of how many cells reference it.
_prof_share = {
    "NAMESPACE": "bench",
    "GPU_PRODUCT": "NVIDIA-B200",
    "MODEL_CACHE_PVC": "glm5-fp8-model-cache",
}
_all = install.gpu_matching_cells(catalog, _prof_share)
_specs = install.ensure_recipe_cache_pvcs(_all, _prof_share, applier=lambda m, n: (0, ""), probe=False, plan_only=True)
_sz = {s["name"]: s["size_gib"] for _, _, s in _specs}
_shared = _sz.get("glm5-fp8-model-cache")

# One copy of each DISTINCT model: GLM-5 704 + Nemotron 400 (+ qwen3, unknown) + 25% headroom.
_distinct = {}
for c in _all:
    _distinct[install._cell_model_repo(c) or c.get("model")] = install._cell_model_size_gib(c)
_expected = install._auto_cache_size_gib(list(_distinct.values()))
check(
    f"{len(_all)} cells / {len(_distinct)} distinct models -> sized for ONE copy each ({_expected}Gi)",
    _shared == _expected,
    f"got {_shared}Gi, expected {_expected}Gi",
)
check(
    "...which is NOT the per-cell sum (the 25260Gi bug)",
    _shared != install._auto_cache_size_gib([install._cell_model_size_gib(c) for c in _all]),
    f"got {_shared}Gi",
)
check(
    "...and it comfortably fits a real 1200Gi claim (the advice must be actionable)",
    _shared is not None and _shared <= 1500,
    f"{_shared}Gi",
)

# Directly: N cells of ONE model must size as one copy, regardless of N.
_one = [c for c in _all if c.get("model") == "glm5-fp8"]
_s1 = install.ensure_recipe_cache_pvcs(
    _one[:1], _prof_share, applier=lambda m, n: (0, ""), probe=False, plan_only=True
)[0][2]["size_gib"]
_s_n = install.ensure_recipe_cache_pvcs(_one, _prof_share, applier=lambda m, n: (0, ""), probe=False, plan_only=True)[
    0
][2]["size_gib"]
check(
    f"1 cell and {len(_one)} cells of the SAME model size identically",
    _s1 == _s_n,
    f"1 cell -> {_s1}Gi, {len(_one)} cells -> {_s_n}Gi",
)

# ---------------------------------------------------------------------------
# 11. READY is not reachable while a model's weights sit on another claim
# ---------------------------------------------------------------------------
print("\n-- 11. misplaced weights block READY and name the remedy --")

_nem_cell = [c for c in catalog if c.get("model") == "nemotron-ultra-nvfp4"][0]
_res = install.phase_d_setup(
    [_nem_cell],
    "example-gpu-cluster",
    {"NAMESPACE": "n", "MODEL_CACHE_PVC": "glm5-fp8-model-cache"},
    True,  # dry_run: no staging, no cluster
    misplaced={
        "nemotron-ultra-nvfp4": (
            "nemotron-ultra-nvfp4-cache",
            "PVC 'nemotron-ultra-nvfp4-cache' is stamped model-revision=183968f87ae4",
        )
    },
)
check(
    "a cell whose weights are on ANOTHER claim is NOT ready",
    _res[0]["status"] != "ready",
    str(_res[0]["status"]),
)
check(
    "...and the fix names --adopt-cache and the per-model key",
    "--adopt-cache" in _res[0]["fix"] and "MODEL_CACHE_PVC_NEMOTRON_ULTRA_NVFP4" in _res[0]["fix"],
    _res[0]["fix"],
)
check(
    "...and names the claim that actually holds the weights",
    "nemotron-ultra-nvfp4-cache" in _res[0]["fix"],
    _res[0]["fix"],
)

# With nothing misplaced the SAME cell is ready — so the downgrade above is caused by the misplacement and
# nothing else. (Asserting only "not ready" would pass even if phase_d_setup were broken for every cell.)
_res_ok = install.phase_d_setup(
    [_nem_cell],
    "example-gpu-cluster",
    {"NAMESPACE": "n", "MODEL_CACHE_PVC": "glm5-fp8-model-cache"},
    True,
    misplaced={},
)
check(
    "no misplacement -> the SAME cell is ready (the downgrade is caused by the misplacement alone)",
    _res_ok[0]["status"] == "ready",
    str(_res_ok[0]["status"]),
)
# A misplacement for a DIFFERENT model must not touch this cell.
_res_other = install.phase_d_setup(
    [_nem_cell],
    "example-gpu-cluster",
    {"NAMESPACE": "n", "MODEL_CACHE_PVC": "glm5-fp8-model-cache"},
    True,
    misplaced={"some-other-model": ("x", "y")},
)
check(
    "a misplacement for another model does not downgrade this cell",
    _res_other[0]["status"] == "ready",
    str(_res_other[0]["status"]),
)

# ---------------------------------------------------------------------------
# 12. The panel answers PER MODEL, not per claim
# ---------------------------------------------------------------------------
print("\n-- 12. a claim's download stamp is not every model's verdict --")

_live = {
    "readable": True,
    "cells": {},
    "caches": {"glm5-fp8-model-cache": {"state": "ready", "why": "downloaded"}},
    "cache_labels": {
        "glm5-fp8-model-cache": {
            "llmb.nvidia.com/download-complete": "true",
            "llmb.nvidia.com/model-name": "glm5-fp8",
        }
    },
}
_p = {"MODEL_CACHE_PVC": "glm5-fp8-model-cache"}
_g_glm, _d_glm = install.cell_cluster_state({"model": "glm5-fp8", "_path": "x"}, _p, _live)
_g_nem, _d_nem = install.cell_cluster_state({"model": "nemotron-ultra-nvfp4", "_path": "x"}, _p, _live)
check(
    "the model the claim IS stamped for reads as downloaded",
    "downloaded" in _d_glm,
    _d_glm,
)
check(
    "a DIFFERENT model on the same claim does NOT inherit that verdict",
    "downloaded" not in _d_nem and "nothing on it vouches for" in _d_nem,
    _d_nem,
)
check(
    "...and it does not get the ready glyph",
    _g_nem != install.GLYPH_READY,
    repr(_g_nem),
)
# The accumulating per-model stamp satisfies it.
_live2 = {
    **_live,
    "cache_labels": {
        "glm5-fp8-model-cache": {
            "llmb.nvidia.com/model.glm5-fp8": "4f96cc5eec29",
            "llmb.nvidia.com/model.nemotron-ultra-nvfp4": "183968f87ae4",
        }
    },
}
check(
    "an accumulating per-model stamp DOES vouch for that model",
    "downloaded" in install.cell_cluster_state({"model": "nemotron-ultra-nvfp4", "_path": "x"}, _p, _live2)[1],
)
# The stamp-key parser: `split(".",1)` on a prefix that CONTAINS dots silently never matched, so the
# accumulating per-model label was written and never read. Lock the parse itself.
check(
    "model_from_stamp_key decodes the per-model stamp (prefix contains dots)",
    mc.model_from_stamp_key("llmb.nvidia.com/model.nemotron-ultra-nvfp4") == "nemotron-ultra-nvfp4",
    mc.model_from_stamp_key("llmb.nvidia.com/model.nemotron-ultra-nvfp4"),
)
check(
    "model_from_stamp_key ignores the single-valued legacy keys",
    mc.model_from_stamp_key("llmb.nvidia.com/model-name") == ""
    and mc.model_from_stamp_key("llmb.nvidia.com/download-complete") == "",
)
check(
    "find_misplaced_weights matches on the per-model stamp too",
    mc.find_misplaced_weights(
        "nemotron-ultra-nvfp4",
        "glm5-fp8-model-cache",
        [
            {
                "name": "other-cache",
                "labels": {"llmb.nvidia.com/model.nemotron-ultra-nvfp4": "183968f87ae4"},
            }
        ],
    )[0]
    == "other-cache",
)

check(
    "the panel says where its evidence comes from",
    "evidence:" in install.render_installed_panel([], {}, _p, _live, "c"),
    install.render_installed_panel([], {}, _p, _live, "c"),
)

# ---------------------------------------------------------------------------
# 13. THE PROBE MUST NOT READ FILE CONTENT TO SIZE THEM
# ---------------------------------------------------------------------------
print("\n-- 13. shard sizes come from metadata, not from streaming the bytes --")

# Shard sizing must use metadata operations rather than streaming model contents.
_probe = mc.cache_probe_script(".", "zai-org/GLM-5-FP8", "4f96cc5eec29")

# Assert the property on the command that sizes shards; other uses of byte-reading tools may be legitimate.)
_size_lines = [line for line in _probe.splitlines() if "SHARD_BYTES" in line or "-exec stat" in line]
check("the shard-sizing command exists", bool(_size_lines), _probe)
_sizing = " ".join(_size_lines)
for _bad in ("wc -c", "cat ", "md5sum", "sha256sum", "cksum", "od "):
    check(
        f"shard sizing never reads content ({_bad.strip()})",
        _bad not in _sizing,
        _sizing,
    )
# NOTE the -L: an earlier version of this very assertion demanded bare `stat -c %s`, which is the
# non-dereferencing form that caused the symlink regression. Section 14 covers why it must dereference.
check(
    "shard sizing uses stat -Lc %s (metadata only, follows blob symlinks)",
    "stat -Lc %s" in _sizing,
    _sizing,
)
# Nothing anywhere in the probe may stream a .safetensors file.
check("no .safetensors file is ever streamed", "wc -c" not in _probe, _probe)
# `find -printf` does NOT exist in busybox 1.36 — verified in the image, not assumed.
check(
    "probe does not use find -printf (absent from busybox 1.36)",
    "-printf" not in _probe,
    _probe,
)

# The parser must surface per-file facts, which is what makes a short shard nameable.
_rep = mc.parse_cache_integrity_report(
    "EXISTS=1\nSENTINEL=0\nCONFIG=1\nINDEX=1\nSHARDS=10\nREQ_SHARDS=10\nMISSING_SHARDS=0\n"
    "INDEX_TOTAL_BYTES=1258287104\nSHARD_BYTES=1258291200\nSHARD_MIN=125829120\n"
    "SHARD_MAX=125829120\nSHARD_N=10\nINCOMPLETE_FILES=0\nREF=\nREFSNAP=0"
)
check(
    "parser surfaces per-shard min/max/count",
    (_rep["shard_min_bytes"], _rep["shard_max_bytes"], _rep["shard_files"]) == (125829120, 125829120, 10),
    str(_rep),
)
check(
    "an intact 10-shard snapshot verdicts COMPLETE",
    mc.cache_completeness(_rep)[0] == mc.STATE_COMPLETE,
    str(mc.cache_completeness(_rep)),
)

# THE CASE THIS NEWLY CATCHES: all 10 files present and index-satisfied, one truncated 120MB -> 1MB.
_trunc = {**_rep, "shard_bytes": 1133510656, "shard_min_bytes": 1048576}
_st, _why = mc.cache_completeness(_trunc)
check(
    "one truncated shard (all files present) -> INCOMPLETE",
    _st == mc.STATE_INCOMPLETE,
    _why,
)
check(
    "...and the message names the outlier so it can be found",
    "smallest shard" in _why,
    _why,
)

# A zero-byte shard passes the missing-set check (the file exists) and is only visible per-file.
check(
    "a 0-byte shard -> INCOMPLETE",
    mc.cache_completeness({**_rep, "shard_min_bytes": 0})[0] == mc.STATE_INCOMPLETE,
)

# Regression guard: probe pods must be deleted synchronously.
_INS2 = (SCRIPTS / "install.py").read_text()
check(
    "transient probe pods are deleted with --now, not --wait=false (they leaked)",
    '"--ignore-not-found", "--wait=false"], timeout=20)' not in _INS2 and _INS2.count('"--now"') >= 3,
    str(_INS2.count('"--now"')),
)

# ---------------------------------------------------------------------------
# 14. SIZING MUST DEREFERENCE — the hub keeps bytes in blobs/ and symlinks in snapshots/
# ---------------------------------------------------------------------------
print("\n-- 14. shard sizes are read THROUGH the hub's blob symlinks --")

# Snapshot entries may be symlinks into the Hugging Face blob store, so both find and stat must
# dereference them. A flat-file fixture alone cannot verify this behavior.
_probe14 = mc.cache_probe_script(".", "zai-org/GLM-5-FP8", "4f96cc5eec29")
_sizing14 = " ".join(line for line in _probe14.splitlines() if "-exec stat" in line or "SHARD_BYTES" in line)
check("the shard-sizing stat dereferences (-L)", "stat -Lc %s" in _sizing14, _sizing14)
check("the shard-sizing find dereferences (-L)", 'find -L "$d"' in _sizing14, _sizing14)
check(
    "no un-dereferenced `stat -c %s` sizes a shard",
    "stat -c %s" not in _sizing14,
    _sizing14,
)

# A hub-layout fixture stores bytes in blobs/ and exposes snapshot symlinks.
with tempfile.TemporaryDirectory() as _td:
    _blobs = Path(_td) / "hub/models--zai-org--GLM-5-FP8/blobs"
    _snap = Path(_td) / "hub/models--zai-org--GLM-5-FP8/snapshots/REV1"
    _blobs.mkdir(parents=True)
    _snap.mkdir(parents=True)
    _per, _n = 4_000_000, 10
    for _i in range(_n):
        _sha = f"{_i:064d}"
        (_blobs / _sha).write_bytes(b"\0" * _per)
        os.symlink(f"../../blobs/{_sha}", _snap / f"model-{_i:05d}.safetensors")
    _declared = _per * _n - 4096  # index total_size: tensor bytes, just under the file bytes
    _deref = sum(os.stat(_f).st_size for _f in _snap.glob("*.safetensors"))
    _nodef = sum(os.lstat(_f).st_size for _f in _snap.glob("*.safetensors"))

    check(
        "fixture is genuinely symlinked (deref != no-deref)",
        _deref != _nodef,
        f"{_deref} vs {_nodef}",
    )
    check(
        "dereferenced sum clears the index total",
        _deref >= _declared,
        f"{_deref} vs {_declared}",
    )
    check(
        "un-dereferenced sum is absurdly small (an invalid measurement)",
        _nodef < _declared // 1000,
        f"{_nodef} vs {_declared}",
    )

    _facts = {
        "exists": True,
        "sentinel": False,
        "config_json": True,
        "index_json": True,
        "shard_count": _n,
        "required_shards": _n,
        "missing_shards": 0,
        "incomplete_files": 0,
        "index_total_bytes": _declared,
        "shard_files": _n,
    }
    _sizes_ok = [os.stat(_f).st_size for _f in _snap.glob("*.safetensors")]
    _sizes_bad = [os.lstat(_f).st_size for _f in _snap.glob("*.safetensors")]

    _st_ok, _ = mc.cache_completeness(
        {
            **_facts,
            "shard_bytes": _deref,
            "shard_min_bytes": min(_sizes_ok),
            "shard_max_bytes": max(_sizes_ok),
        }
    )
    check(
        "hub-layout cache read WITH deref -> COMPLETE",
        _st_ok == mc.STATE_COMPLETE,
        _st_ok,
    )

    # The same fixture measured the buggy way must NOT produce a confident "truncated".
    _st_bad, _why_bad = mc.cache_completeness(
        {
            **_facts,
            "shard_bytes": _nodef,
            "shard_min_bytes": min(_sizes_bad),
            "shard_max_bytes": max(_sizes_bad),
        }
    )
    check(
        "hub-layout cache read WITHOUT deref -> UNKNOWN, never a confident 'truncated'",
        _st_bad == mc.STATE_UNKNOWN,
        f"{_st_bad}: {_why_bad}",
    )
    check(
        "...and it says the MEASUREMENT is at fault, not the data",
        "not credible" in _why_bad and "MEASUREMENT is wrong" in _why_bad,
        _why_bad,
    )

# THE SANITY INVARIANT the code holds itself to: observed sizes within a couple of orders of magnitude of
# the index-implied mean, or the measurement — not the disk — is wrong.
_glm = {
    "exists": True,
    "sentinel": False,
    "config_json": True,
    "index_json": True,
    "shard_count": 142,
    "required_shards": 142,
    "missing_shards": 0,
    "incomplete_files": 0,
    "index_total_bytes": 756162687872,
    "shard_files": 142,
}
check(
    "production symlink numbers (54/76 B over 142 shards) -> UNKNOWN",
    mc.cache_completeness({**_glm, "shard_bytes": 7712, "shard_min_bytes": 54, "shard_max_bytes": 76})[0]
    == mc.STATE_UNKNOWN,
)
check(
    "real GLM-5 sizes -> COMPLETE",
    mc.cache_completeness(
        {
            **_glm,
            "shard_bytes": 756200000000,
            "shard_min_bytes": 4000000000,
            "shard_max_bytes": 5363940952,
        }
    )[0]
    == mc.STATE_COMPLETE,
)
# The gate must not swallow a GENUINE truncation: one shard at 1/50th is implausible-looking but in range.
check(
    "a real truncation is still reported INCOMPLETE (gate does not over-trigger)",
    mc.cache_completeness(
        {
            **_glm,
            "shard_bytes": 751000000000,
            "shard_min_bytes": 107278819,
            "shard_max_bytes": 5363940952,
        }
    )[0]
    == mc.STATE_INCOMPLETE,
)
# And Nemotron's real numbers — the non-hub claim that passed even with the bug — must stay COMPLETE.
check(
    "nemotron's real (non-hub) numbers -> COMPLETE",
    mc.cache_completeness(
        {
            "exists": True,
            "sentinel": False,
            "config_json": True,
            "index_json": True,
            "shard_count": 113,
            "required_shards": 113,
            "missing_shards": 0,
            "incomplete_files": 0,
            "index_total_bytes": 352284061280,
            "shard_bytes": 352308689576,
            "shard_files": 113,
            "shard_min_bytes": 2000000000,
            "shard_max_bytes": 3200000000,
        }
    )[0]
    == mc.STATE_COMPLETE,
)

print()
if fails:
    print(f"FAILED ({len(fails)}): " + ", ".join(fails))
    sys.exit(1)
print("selftest_cache_claim_agreement: all checks passed")
