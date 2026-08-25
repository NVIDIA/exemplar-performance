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

"""provenance.py — make every published result reproducible + provably tied to the recipe that produced it.

A number is only trustworthy if you can (a) see exactly what produced it and (b) regenerate it. This tool
closes that loop around the `recipe_hash` fingerprint:

  provenance.py <cell> --stamp [--run <run_meta.json>]
      Emit the `## Provenance` block to paste into RESULTS.md — recipe_hash + image digest + dataset sha256
      + git commit + the EXACT reproduce command. Machine-derivable fields are auto-filled; run_id / cluster
      / date come from --run (the sweep's run_meta.json) or are left as «fill in».

  provenance.py --check [<root>]
      CI gate. Every cell at runs/performant/exemplar must ship a RESULTS.md that cites the CURRENT
      recipe_hash AND an append-only runs/index.jsonl entry with that same hash. New entries carry the hash
      captured before launch; historical entries are reported explicitly as legacy. Missing evidence is UNKNOWN
      and fails; a stale hash means the recipe drifted since the run was posted (re-run or re-stamp).
      planned/wip are lenient (no committed run yet).

  provenance.py <cell> --verify
      Human check: does this cell's RESULTS still match the current recipe? Prints MATCH/DRIFT + reproduce cmd.

  provenance.py [<cell>] --impact   |   provenance.py --impact [<root>]
      Advisory (never fails): after editing a recipe, does its published record still match?
      Prints MATCH/DRIFT/UNPUBLISHED (single cell) or lists the drifted cells + exact remediation (--all).
      The soft counterpart to --check — surfaces silent recipe_hash churn at edit/run time.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe_hash as _rh  # the fingerprint everything hangs off
import benchmark_id as _bid  # stable benchmark identity (image/flag-invariant)

try:
    import yaml
except ImportError:
    sys.exit("provenance: requires pyyaml")

ROOT = Path(__file__).resolve().parent.parent
PUBLISHED = {
    "runs",
    "performant",
    "exemplar",
}  # a run exists -> provenance is mandatory
HASH_RE = re.compile(r"recipe_hash[^0-9a-fx]{0,24}([0-9a-f]{64})", re.I)
FILL = "«fill in»"


def load(cell: Path) -> dict:
    return yaml.safe_load((cell / "recipe.yaml").read_text()) or {}


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
        ).stdout.strip()
        return out or "?"
    except Exception:
        return "?"


def stamp(cell: Path, run_meta: str | None = None) -> str:
    r = load(cell)
    env = r.get("envelope") or {}
    prov = env.get("provenance") or {}
    rel = cell.relative_to(ROOT)
    h = _rh.recipe_hash(cell)
    bid = _bid.benchmark_id(cell)
    img = prov.get("image_digest") or prov.get("image_ref") or "«unpinned»"

    run_id, cluster, date = FILL, FILL, FILL
    wall_h: str = FILL
    gpu_h: str = FILL
    if run_meta:
        m = json.loads(Path(run_meta).read_text())
        run_id = m.get("run_id") or run_id
        cluster = m.get("cluster") or m.get("profile") or cluster
        date = (m.get("completed_at_utc") or m.get("started_at_utc") or "")[:10] or date
        secs = m.get("wall_seconds_total")
        gpus = m.get("gpu_count")
        if secs is not None:
            wh = round(secs / 3600, 1)
            wall_h = f"{wh}h"
            if gpus is not None:
                gpu_h = f"{round(wh * int(gpus), 1)} GPU·h"

    gc = git_commit()
    gpu = env.get("gpu_type") or "GPU"
    gcount = ((env.get("requires") or {}).get("gpu") or {}).get("count") or "?"
    ex = env.get("exemplar") or {}
    metric = ex.get("metric") or "exemplar"
    _ref, _unit = ex.get("reference"), (ex.get("unit") or "")
    ref_str = f"{_ref} {_unit}".strip() if _ref is not None else "<set on publish>"
    tol_str = f", tol ±{ex['tolerance_pct']}%" if ex.get("tolerance_pct") is not None else ""

    leaf = rel.name  # the leaf slug `install --recipes` matches on (basename of the recipe path)
    d = (r.get("bench") or {}).get("dataset") or {}
    dataset = f"`{d.get('id', '?')}` · sha256 `{d.get('sha256', '?')}`"
    sweep = (r.get("bench") or {}).get("sweep_concurrency") or []
    sweep_str = ", ".join(str(x) for x in sweep) if sweep else "<rungs>"
    repro = (
        f"git checkout {gc}                                          # the exact repo state\n"
        f"scripts/llmb-k8s init                                      # one-time: pick your cluster → profile\n"
        f"scripts/llmb-k8s install <profile> --recipes {leaf}        # ns + secrets + model cache → weights\n"
        f"scripts/llmb-k8s run {rel} <profile> --teardown --fetch    # preflight → serve → sweep → results (GPUs auto-freed)\n"
        f"scripts/llmb-k8s publish {rel} <profile> results/<run-id>\n"
        f"# verify you're on this recipe: scripts/recipe_hash.py {rel}  == recipe_hash above"
    )
    example = (
        "$ llmb-k8s preflight --recipe {r} --cluster <profile>\n"
        "  ✅ pinned context · namespace · {g} node with {c} free GPUs · pull-secret · model weights  → PASSED\n"
        "$ llmb-k8s run --recipe {r} --cluster <profile>\n"
        "  1/8 preflight ✓  2/8 stage ✓  4/8 server ✓  5/8 wait-ready ({g} ×{c}, Ready)\n"
        "  6/8 benchmark (sweep)  concurrency {s}\n"
        "  7/8 fetch ✓  8/8 done → results/<run-id>\n"
        "$ llmb-k8s analyze --recipe {r} --cluster <profile> results/<run-id>\n"
        "  {m}: <measured>  (reference {rf}{t})"
    ).format(r=rel, g=gpu, c=gcount, s=sweep_str, m=metric, rf=ref_str, t=tol_str)

    return f"""## Provenance

_A published page requires a `runs/index.jsonl` ledger entry carrying this `recipe_hash`; new runs capture it
before launch and CI verifies that it still matches. `scripts/recipe_hash.py {rel}` must still print this hash,
or the recipe has changed since._

| field | value |
|---|---|
| recipe_hash | `{h}` |
| benchmark_id | `{bid}` |
| image | `{img}` |
| dataset | {dataset} |
| cluster · profile | {cluster} |
| run_id | {run_id} |
| wall_h | {wall_h} |
| gpu_h | {gpu_h} |
| git commit | `{git_commit()}` |
| date (UTC) | {date} |

**Reproduce** — from scratch on any matching cluster (`<profile>` = your cluster profile, e.g. `example-gpu-cluster`):
```bash
{repro}
```

<details><summary>What this looks like in the terminal (abridged, illustrative)</summary>

```console
{example}
```
</details>
"""


def cited_hash(cell: Path):
    rp = cell / "RESULTS.md"
    if not rp.exists():
        return None
    m = HASH_RE.search(rp.read_text())
    return m.group(1) if m else None


def ledger_recipe_hashes(
    cell: Path,
) -> tuple[list[str], list[str], int, int, str | None]:
    """Return (hashes, launch_hashes, missing_field_count, row_count, read_error) from the durable run ledger.

    A missing index, malformed JSON, or a row without recipe_hash is an UNKNOWN observation, not proof that
    no run existed. Callers must fail closed rather than turning it into an empty, valid ledger.
    """
    index = cell / "runs" / "index.jsonl"
    if not index.exists():
        return [], [], 0, 0, "runs/index.jsonl is absent"
    hashes: list[str] = []
    launch_hashes: list[str] = []
    missing = 0
    rows = 0
    try:
        lines = index.read_text().splitlines()
    except OSError as exc:
        return [], [], 0, 0, f"cannot read runs/index.jsonl: {exc}"
    for line_no, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        rows += 1
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            return (
                hashes,
                launch_hashes,
                missing,
                rows,
                f"runs/index.jsonl line {line_no} is invalid JSON: {exc.msg}",
            )
        if not isinstance(entry, dict):
            return (
                hashes,
                launch_hashes,
                missing,
                rows,
                f"runs/index.jsonl line {line_no} is not an object",
            )
        value = entry.get("recipe_hash")
        if not value:
            missing += 1
        elif isinstance(value, str):
            hashes.append(value)
        else:
            return (
                hashes,
                launch_hashes,
                missing,
                rows,
                f"runs/index.jsonl line {line_no} recipe_hash is not a string",
            )
        launched = entry.get("recipe_hash_at_launch")
        if launched is not None:
            if not isinstance(launched, str):
                return (
                    hashes,
                    launch_hashes,
                    missing,
                    rows,
                    (f"runs/index.jsonl line {line_no} recipe_hash_at_launch is not a string"),
                )
            launch_hashes.append(launched)
    return hashes, launch_hashes, missing, rows, None


def format_ledger_hashes(hashes: list[str], missing: int) -> str:
    found = sorted(set(hashes))
    if missing:
        found.append(f"<missing> ×{missing}")
    return ", ".join(found) if found else "<none>"


def check_all(root: Path) -> int:
    cells = sorted({p.parent for p in (root / "recipes").glob("**/recipe.yaml")})
    fails = 0
    for c in cells:
        env = load(c).get("envelope") or {}
        status = env.get("status", "planned")
        rel = c.relative_to(root)
        if status not in PUBLISHED:
            print(f"skip {rel} (status={status}; no published run yet)")
            continue
        cur, cited = _rh.recipe_hash(c), cited_hash(c)
        cell_ok = True
        if not cited:
            print(
                f"FAIL {rel}: status={status} but RESULTS.md cites no recipe_hash "
                f"(run: scripts/provenance.py {rel} --stamp)"
            )
            cell_ok = False
        elif cited != cur:
            print(
                f"FAIL {rel}: RESULTS.md recipe_hash {cited[:12]} != current {cur[:12]} — "
                f"the recipe drifted since these numbers were posted (re-run or re-stamp)"
            )
            cell_ok = False
        hashes, launch_hashes, missing, rows, ledger_error = ledger_recipe_hashes(c)
        ledger_found = format_ledger_hashes(hashes, missing)
        if ledger_error:
            print(
                f"UNKNOWN {rel}: cannot establish a matching ledger run for current recipe_hash {cur}; "
                f"{ledger_error}; ledger hashes found: {ledger_found}"
            )
            cell_ok = False
        elif rows == 0:
            print(
                f"FAIL {rel}: zero ledger runs match current recipe_hash {cur}; " f"ledger hashes found: {ledger_found}"
            )
            cell_ok = False
        elif not hashes:
            print(
                f"UNKNOWN {rel}: {rows} ledger run(s) lack recipe_hash, so matching current recipe_hash "
                f"{cur} is UNKNOWN; ledger hashes found: {ledger_found}"
            )
            cell_ok = False
        elif cur not in hashes:
            print(
                f"FAIL {rel}: zero ledger runs match current recipe_hash {cur}; " f"ledger hashes found: {ledger_found}"
            )
            cell_ok = False
        if cell_ok:
            if cur in launch_hashes:
                print(f"OK   {rel} (RESULTS and launch-attested ledger recipe_hash {cur[:12]} match)")
            else:
                print(
                    f"LEGACY {rel}: matching ledger recipe_hash has no recipe_hash_at_launch; this historical "
                    "receipt is not a runtime attestation"
                )
                print(f"OK   {rel} (RESULTS and legacy ledger recipe_hash {cur[:12]} match)")
        else:
            fails += 1
    published = sum(1 for c in cells if load(c).get("envelope", {}).get("status") in PUBLISHED)
    print(
        f"provenance-check: {published - fails}/{published} published cells cite the current recipe_hash and "
        f"carry a matching ledger run "
        f"({len(cells) - published} not-yet-run skipped)"
    )
    return 1 if fails else 0


def _rel(cell: Path):
    try:
        return cell.relative_to(ROOT)
    except ValueError:
        return cell


def impact_one(cell: Path) -> tuple[str, str]:
    """Advisory: does a cell's published record still match its recipe?
    Returns (status, message), status ∈ {MATCH, DRIFT, UNPUBLISHED}. Never raises/exits — this is the
    guardrail that surfaces the silent recipe_hash churn (edit a recipe → the committed numbers no longer
    match) with the exact remediation, instead of waiting for `--check` to fail in CI.
    """
    env = load(cell).get("envelope") or {}
    status = env.get("status", "planned")
    cur = _rh.recipe_hash(cell)
    rel = _rel(cell)
    if status not in PUBLISHED:
        return (
            "UNPUBLISHED",
            f"{rel}: status={status} — no published record to invalidate (recipe_hash {cur[:12]}).",
        )
    cited = cited_hash(cell)
    if cited and cited == cur:
        return (
            "MATCH",
            f"{rel}: published record matches the current recipe (recipe_hash {cur[:12]}).",
        )
    return "DRIFT", (
        f"{rel}: published record cites {(cited or 'none')[:12] if cited else 'none'} but the recipe is now "
        f"{cur[:12]}.\n"
        f"       The committed numbers no longer match this recipe. To make the change official:\n"
        f"         • set envelope.status: wip  (until a fresh run re-publishes at the new hash), and/or\n"
        f"           re-publish this cell:  scripts/publish.py {rel} <run-dir>\n"
        f"         • refresh the matrix:    scripts/build_catalog.py"
    )


def impact_all(root: Path) -> int:
    cells = sorted({p.parent for p in (root / "recipes").glob("**/recipe.yaml")})
    drift = 0
    for c in cells:
        st, msg = impact_one(c)
        if st == "DRIFT":
            print(f"DRIFT  {msg}")
            drift += 1
    print(
        f"recipe-impact: {'nothing to reconcile — every published cell matches its recipe_hash' if not drift else f'{drift} cell(s) drifted from their published record (advisory — see remediation above)'}."
    )
    return 0  # advisory: never fails CI (that's --check's job)


def verify(cell: Path) -> int:
    cur, cited = _rh.recipe_hash(cell), cited_hash(cell)
    rel = cell.relative_to(ROOT)
    if not cited:
        print(f"DRIFT {rel}: RESULTS.md cites no recipe_hash — run `provenance.py {rel} --stamp`")
        return 1
    if cited == cur:
        print(f"MATCH {rel}: RESULTS numbers are from the current recipe (recipe_hash {cur[:12]}). Reproduce:")
        print("\n".join(l for l in stamp(cell).splitlines() if l.startswith("scripts/") or l.startswith("analysis/")))
        return 0
    print(
        f"DRIFT {rel}: RESULTS cite {cited[:12]} but recipe is now {cur[:12]} — numbers are stale; re-run or re-stamp"
    )
    return 1


def main() -> int:
    args = sys.argv[1:]
    if "--check" in args:
        rest = [a for a in args if a != "--check"]
        return check_all(Path(rest[0]) if rest else ROOT)
    if "--impact" in args:
        pos = [a for a in args if not a.startswith("--")]
        if pos and (Path(pos[0]) / "recipe.yaml").exists():
            st, msg = impact_one(Path(pos[0]).resolve())
            print(f"{st}  {msg}")
            return 0
        return impact_all(Path(pos[0]) if pos else ROOT)
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        sys.exit(__doc__)
    cell = Path(positional[0]).resolve()
    if "--verify" in args:
        return verify(cell)
    if "--stamp" in args:
        run_meta = None
        if "--run" in args:
            i = args.index("--run")
            run_meta = args[i + 1] if i + 1 < len(args) else None
        print(stamp(cell, run_meta))
        return 0
    sys.exit(__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
