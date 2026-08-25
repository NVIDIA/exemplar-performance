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

"""selftest_reproduce.py — the recipe-page contract: THREE steps, the right ones, and nothing else.

Covers the regressions that made recipe pages rot in the first place:
  • the three steps are init → install → run, in that order, parameterized for THIS cell (a page that
    copy/pastes a SIBLING's cell path is the bug we shipped in the -c16/-c32/-c64 pages);
  • `install --recipes <slug>` uses the slug install actually matches on;
  • a cell in a family gets the whole-group commands too, and a SINGLETON does not get a fake group;
  • cross-hardware siblings are NOT a group (you cannot run B200 + GB300 against one profile);
  • a SUPERSEDED cell is never handed a command that runs itself — including when the whole family is
    retired in favour of a SINGLE replacement (the 256k -c16/-c32/-c64 ladder → the 1M-context cell);
  • the generator is idempotent and the banned-primitive lint actually fires;
  • every committed cell page passes the check (this is what `make reproduce-check` runs in CI).
  • family grouping is fail-closed across serving mode, runtime lineage, and nominal workload shape, so a
    generated whole-family command cannot mix aggregate/disaggregate, Dynamo/upstream SGLang, or
    1k/1k–8k/1k–16k/512.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import reproduce as R  # noqa: E402

FAILS: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    if cond:
        print(f"  ok   {name}")
    else:
        FAILS.append(f"{name}{(': ' + detail) if detail else ''}")
        print(f"  FAIL {name} {detail}")


def _cell(root: Path, rel: str, **env) -> Path:
    d = root / rel
    d.mkdir(parents=True, exist_ok=True)
    body = ["envelope:"] + [f"  {k}: {v}" for k, v in env.items()]
    (d / "recipe.yaml").write_text("\n".join(body) + "\n")
    (d / "README.md").write_text(f"# {d.name}\n\n<!-- REPRODUCE:START -->\n<!-- REPRODUCE:END -->\n")
    return d


def _merge_recipe(cell: Path, **sections) -> None:
    """Small fixture helper: merge top-level mapping sections into an existing synthetic cell."""
    p = cell / "recipe.yaml"
    doc = R.yaml.safe_load(p.read_text()) or {}
    for key, values in sections.items():
        current = doc.setdefault(key, {})
        current.update(values)
    p.write_text(R.yaml.safe_dump(doc, sort_keys=False))


def main() -> int:
    print("selftest_reproduce")

    # ── 1. the three steps, parameterized for THIS cell ────────────────────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        c = _cell(
            root,
            "recipes/llm-perf/x/my-cell-c32",
            name="my-slug-c32",
            scenario="llm-perf",
            goal="pareto",
            distribution="synthetic",
            model="m",
            gpu_type="B200",
        )
        steps = R.three_steps(c, root)
        body = "\n".join(steps)
        check("step 1 is init", "scripts/llmb-k8s init" in body)
        check(
            "step 2 installs THIS cell's slug", "scripts/llmb-k8s install <profile> --recipes my-slug-c32" in body, body
        )
        check(
            "step 3 runs THIS cell's path",
            "scripts/llmb-k8s run recipes/llm-perf/x/my-cell-c32 <profile> --teardown --fetch" in body,
            body,
        )
        check(
            "steps are ordered init < install < run",
            body.index("llmb-k8s init") < body.index("llmb-k8s install") < body.index("llmb-k8s run"),
        )
        # THE -c16/-c32/-c64 BUG: the page must never name a sibling's directory.
        check("no sibling path leaks into the steps", "my-cell-c16" not in body and "my-cell-c64" not in body)

        # a singleton must not grow a fabricated group section
        check("singleton has no group block", R.group_block(c, root) == [], str(R.group_block(c, root)))

    # ── 2. a family: same benchmark + same hardware, different points ──────────────────────────────
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        common = dict(scenario="llm-perf", goal="pareto", distribution="workload", model="nemo", gpu_type="GB300")
        a = _cell(root, "recipes/llm-perf/d/cell-c16", name="cell-c16", **common)
        b = _cell(root, "recipes/llm-perf/d/cell-c32", name="cell-c32", **common)
        _cell(root, "recipes/llm-perf/d/cell-c64", name="cell-c64", **common)
        # a cross-HARDWARE sibling: same benchmark, different GPU → NOT runnable on one profile → not a group
        other = dict(common)
        other["gpu_type"] = "B200"
        _cell(root, "recipes/llm-perf/d/cell-b200", name="cell-b200", **other)

        fam = R.group_members(a, root)
        check("family has the 3 same-hardware members", len(fam) == 3, str([f.name for f in fam]))
        check(
            "cross-hardware sibling is excluded from the group",
            all("b200" not in f.name for f in fam),
            str([f.name for f in fam]),
        )
        gb = "\n".join(R.group_block(a, root))
        check("group installs all members in ONE --recipes", "--recipes cell-c16,cell-c32,cell-c64" in gb, gb)
        check("group runs members SERIALLY (whole-node exclusivity)", "for CELL in " in gb and "done" in gb)
        check("group ends with a real compare command", "scripts/llmb-k8s compare " in gb)
        check(
            "the B200 sibling has no group at all",
            R.group_members(root / "recipes/llm-perf/d/cell-b200", root) == [],
        )

        # ── 3. superseded: never told to run itself ────────────────────────────────────────────────
        sup = _cell(root, "recipes/llm-perf/d/cell-base", name="cell-base", **common)
        (sup / "README.md").write_text(
            "# base\n\n> **⚠ SUPERSEDED** use the split cells.\n\n<!-- REPRODUCE:START -->\n<!-- REPRODUCE:END -->\n"
        )
        check(
            "superseded cell is dropped from its family's member list",
            all("base" not in f.name for f in R.group_members(a, root)),
        )
        md = R.reproduce_md(sup, root)
        check(
            "superseded page does NOT offer to run itself",
            "llmb-k8s run recipes/llm-perf/d/cell-base " not in md,
            md,
        )
        check("superseded page points at the replacements", "cell-c16,cell-c32,cell-c64" in md)

        # ── 4. idempotence: write twice → identical bytes, and check() agrees ──────────────────────
        def _write(cell: Path) -> str:
            p = cell / "README.md"
            p.write_text(R.inject(p.read_text(), R.reproduce_md(cell, root)))
            return p.read_text()

        first = _write(a)
        check("generator is idempotent", _write(a) == first)
        check("check() passes right after write", R.check_cell(a, root) == [], str(R.check_cell(a, root)))

        # a hand-edit inside the region must be DETECTED, not silently kept
        (a / "README.md").write_text(first.replace("scripts/llmb-k8s init", "scripts/llmb-k8s init --express"))
        check("stale block is detected", any("STALE" in p for p in R.check_cell(a, root)))

    # ── 4b. MANY cells retired in favour of ONE: the replacement must still be named ───────────────
    # The 256k -c16/-c32/-c64 ladder → the single 1M-context cell. `group_members` used to return []
    # for a family of one, which would have handed each retired page its OWN three steps — a page that
    # contradicts its own banner. A retired cell's replacements are never "not a group".
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        common = dict(scenario="llm-perf", goal="pareto", distribution="workload", model="nemo", gpu_type="GB300")
        new = _cell(root, "recipes/llm-perf/d/cell-1m", name="cell-1m", **common)
        old = []
        for rung in ("c16", "c32", "c64"):
            c = _cell(root, f"recipes/llm-perf/d/cell-{rung}", name=f"cell-{rung}", **common)
            (c / "README.md").write_text(
                f"# {rung}\n\n> **⚠ SUPERSEDED** use cell-1m.\n\n" "<!-- REPRODUCE:START -->\n<!-- REPRODUCE:END -->\n"
            )
            old.append(c)
        for c in old:
            md = R.reproduce_md(c, root)
            check(
                f"{c.name}: retired page names its ONE replacement",
                "--recipes cell-1m" in md and "recipes/llm-perf/d/cell-1m" in md,
                md,
            )
            check(
                f"{c.name}: retired page never offers to run ITSELF",
                f"llmb-k8s run recipes/llm-perf/d/{c.name} " not in md,
                md,
            )
            check(
                f"{c.name}: retired page never names a retired SIBLING",
                not any(o.name in md for o in old if o is not c),
                md,
            )
        # the survivor is now a singleton and must NOT grow a group section naming dead cells
        check(
            "the surviving replacement has no group block",
            R.group_block(new, root) == [],
            str(R.group_block(new, root)),
        )
    # ── 4b. family identity fails closed across deployment/runtime/workload axes ──────────────────
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        common = dict(
            scenario="llm-perf",
            goal="pareto",
            distribution="synthetic",
            model="glm5-fp8",
            gpu_type="B200",
            arch="amd64",
            engine="sglang",
            serving_mode="disaggregated",
            framework="dynamo",
            mode="synthetic",
            launcher="aiperf",
        )

        def rich(
            rel: str,
            name: str,
            *,
            serving_mode="disaggregated",
            image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.4",
            isl=1024,
            osl=1024,
        ):
            env = dict(common)
            env["serving_mode"] = serving_mode
            cell = _cell(root, rel, name=name, **env)
            _merge_recipe(
                cell,
                serving={"stack": "sglang-disagg" if serving_mode == "disaggregated" else "sglang-agg"},
                bench={"endpoint_type": "completions", "synthetic": {"nominal_isl": isl, "nominal_osl": osl}},
                envelope={"provenance": {"image_ref": image}},
            )
            return cell

        a = rich("recipes/llm-perf/glm/dyn-c16", "dyn-c16")
        b = rich(
            "recipes/llm-perf/glm/dyn-c32",
            "dyn-c32",
            image="nvcr.io/nvidia/ai-dynamo/sglang-runtime:1.5@sha256:deadbeef",
        )
        rich("recipes/llm-perf/glm/upstream", "upstream", image="docker.io/lmsysorg/sglang:v0.5.11")
        rich("recipes/llm-perf/glm/agg", "agg", serving_mode="aggregated")
        rich("recipes/llm-perf/glm/8k", "8k", isl=8192)

        fam = R.group_members(a, root)
        check(
            "same runtime repository survives tag/digest rolls",
            [c.name for c in fam] == [a.name, b.name],
            str([c.name for c in fam]),
        )
        check(
            "family excludes aggregate, another runtime, and another workload",
            all(x not in {c.name for c in fam} for x in ("agg", "upstream", "8k")),
            str([c.name for c in fam]),
        )

    # ── 5. the banned-primitive lint actually fires ────────────────────────────────────────────────
    for bad in (
        "kubectl -n ns scale deploy/x --replicas=0",
        "scripts/render.sh $CELL",
        "scripts/fetch_results.sh <run-id>",
        "python3 analysis/llm-perf/aggregate_metrics.py results/RUN",
        "cp cluster-profiles/x.env.example cluster-profiles/x.env",
    ):
        check(f"banned: {bad[:38]}…", bool(R.banned_hits(bad)))
    check(
        "a clean three-step page trips nothing",
        R.banned_hits(
            "scripts/llmb-k8s init\nscripts/llmb-k8s install <p> --recipes s\n"
            "scripts/llmb-k8s run recipes/a/b <p> --teardown --fetch"
        )
        == [],
    )

    # ── 6. every COMMITTED cell page satisfies the contract (what CI enforces) ─────────────────────
    r = subprocess.run([sys.executable, str(ROOT / "scripts/reproduce.py")], capture_output=True, text=True)
    check("make reproduce-check is green on the committed tree", r.returncode == 0, r.stdout + r.stderr)
    cells = R.all_cells()
    check("every cell README carries the markers", all(R.START in (c / "README.md").read_text() for c in cells))
    check(
        "every cell README has the contract's 'How to reproduce' heading",
        all("How to reproduce" in (c / "README.md").read_text() for c in cells),
    )

    # The concrete GA-6 regression: the committed GLM-5 tree must yield six disaggregated
    # runtime/workload families and aggregate singletons — never the old 51-cell cross-product.
    glm = ROOT / "recipes/llm-perf/Glm5/B200_k8s"
    if glm.is_dir():
        agg = sorted((glm / "Agg").glob("**/recipe.yaml"))
        check("GA-6: every GLM aggregate cell is a singleton", all(R.group_members(p.parent) == [] for p in agg))
        expected = {
            ("sglang_dynamo", "1k_1k"): 6,
            ("sglang_dynamo", "8k_1k"): 9,
            ("sglang_dynamo", "16k_512"): 9,
            ("non_dynamo_sglang", "1k_1k"): 6,
            ("non_dynamo_sglang", "8k_1k"): 9,
            ("non_dynamo_sglang", "16k_512"): 9,
        }
        # A runtime family the branch does not carry is SKIPPED-and-NAMED, never a bare StopIteration.
        # Some disaggregated lanes may not be present on this branch. Asserting a family that is absent by
        # design would
        # pin the CATALOG; asserting nothing would hide a real prune. Naming the skip does neither.
        for (runtime, shape), size in expected.items():
            _fam_dir = glm / "Disagg" / runtime / shape
            _found = sorted(_fam_dir.glob("*/recipe.yaml"))
            if not _found:
                print(f"  skip GA-6: {runtime}/{shape} absent on this branch (expected {size} cells)")
                continue
            sample = _found[0].parent
            fam = R.group_members(sample)
            check(
                f"GA-6: {runtime}/{shape} is an isolated {size}-cell family",
                len(fam) == size and all(f"/{runtime}/{shape}/" in f"/{c.as_posix()}/" for c in fam),
                str([c.name for c in fam]),
            )

    if FAILS:
        print(f"\nselftest_reproduce: {len(FAILS)} FAILURE(S)")
        for f in FAILS:
            print(f"  - {f}")
        return 1
    print("selftest_reproduce: all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
