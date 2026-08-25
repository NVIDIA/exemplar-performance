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

"""selftest_run_id.py — offline guards for scripts/run_id.py's CALLER-SUPPLIED run-id handling.

A hand-passed run-id (the optional 3rd positional arg to sweep.sh / run.sh, e.g. `rp3`) is no
longer used verbatim: it is routed through run_id.mint_labeled() so it gets the compact UTC stamp prefix
(rp3 → 260728t0312-rp3), stays DNS-1123 label-safe, is shrunk so the literal Job name <cell>-<kind>-<id>
still fits ≤63, and is IDEMPOTENT (an already-dated id passes through unchanged for resume determinism).

The acceptance contract:
  - custom label → <compact-stamp>-<sanitized-label>, DNS-1123-valid, stamp leads (time-sortable);
  - already-dated ids (all three auto formats) pass through UNCHANGED — no double-prefix;
  - long cell + custom label: the LITERAL <cell>-<kind>-<id> still fits ≤63 (label shrinks / base36 fallback);
  - --at is deterministic (prefix stamp = the given stamp, not now());
  - the run_id.py CLI wires --label through this path (and --job-name --label composes).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import run_id as R  # noqa: E402,N812

AT = "20260728t031200"  # fixed UTC stamp → compact "260728t0312"
STAMP = "260728t0312"

fails: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  [{detail}]" if not cond and detail else ""))
    if not cond:
        fails.append(name)


def dns_ok(s: str) -> bool:
    return bool(R.LABEL_RE.match(s)) and len(s) <= R.MAX


# ── 1. short cell: custom label → <stamp>-<label>, DNS-valid, stamp leads ───────
rid = R.mint_labeled("nemo-b200-256k", "rp3", at=AT, fit_kind="bench")
check("short cell: rp3 → 260728t0312-rp3", rid == f"{STAMP}-rp3", rid)
check("short cell: result is DNS-1123 label-safe", dns_ok(rid), rid)
check("short cell: compact stamp leads (time-sortable)", rid.startswith(STAMP), rid)

# sanitization of a messy label (uppercase / punctuation / spaces)
rid_s = R.mint_labeled("nemo-b200-256k", "B200 Aggr#1", at=AT, fit_kind="bench")
check("messy label sanitized → b200-aggr-1", rid_s == f"{STAMP}-b200-aggr-1", rid_s)
check("sanitized result DNS-valid", dns_ok(rid_s), rid_s)

# no fit_kind → same prefixed form (path-only usage)
check("no-fit: still prefixed", R.mint_labeled("c", "rp3", at=AT) == f"{STAMP}-rp3")

# ── 2. idempotency: already-dated ids pass through UNCHANGED (no double-prefix) ──
for dated in ("260722t0312-a1b2", "20260722t031200", "20260722-031200"):
    out = R.mint_labeled("nemo-b200-256k", dated, at=AT, fit_kind="bench")
    check(f"already-dated passthrough: {dated}", out == dated, out)

# ── 2b. #47: the tight base36-epoch auto-token (long cell names) is ALSO idempotent ──
# Regression for the run-id mismatch: run.sh mints an id and hands it to the bench script, which routes it
# back through mint_labeled(). For a LONG cell name mint() emits the tight `t<b36>` token (not date-led); if
# mint_labeled re-stamps that token, run.sh (fetching /artifacts/<its id>) and the driver (writing
# /artifacts/<bench-time id>) diverge whenever the two mints straddle a wall-clock change — the #47 crash.
long_name = "nemotron-ultra-3-gb300-workload-stable16-pareto"  # 46 chars → tight token, not the dated form
tight = R.mint(long_name, "deadbeef", at=AT, fit_kind="serving")
check(
    "long cell mints the tight t<b36> token",
    tight[0] == "t" and "-" not in tight,
    tight,
)
# Re-passing it as a --label at a DIFFERENT wall-clock time must return it UNCHANGED (this is the fix):
relabelled_later = R.mint_labeled(long_name, tight, at="20260729t201400", fit_kind="serving")
check(
    "#47: tight auto-token passthrough (stamp-drift safe)",
    relabelled_later == tight,
    f"{tight} -> {relabelled_later}",
)
# And an ordinary human label that merely starts with 't' must NOT be mistaken for an auto-token — it is
# still stamped as before (guards the semantic epoch-window check against false positives).
human_t = R.mint_labeled("nemo-b200-256k", "toolbar", at=AT, fit_kind="bench")
check(
    "human label 'toolbar' still stamped (not mistaken for t<b36>)",
    human_t == f"{STAMP}-toolbar",
    human_t,
)


# ── 3. long cell: the LITERAL Job name <cell>-<kind>-<id> must fit ≤63 ──────────
def literal_fits(name: str, kind: str, rid: str) -> int:
    return len(R.sanitize(name)) + 2 + len(kind) + len(rid)  # <name>-<kind>-<id>


# 3a. moderate-long name → label truncated but the leading stamp is kept
name_a = "c" * 41  # budget = 63-41-5-2 = 15
rid_a = R.mint_labeled(name_a, "custombenchlabel", at=AT, fit_kind="bench")
check(
    "long-A: literal <cell>-bench-<id> ≤ 63",
    literal_fits(name_a, "bench", rid_a) <= 63,
    f"{literal_fits(name_a, 'bench', rid_a)}: {rid_a}",
)
check("long-A: stamp still leads", rid_a.startswith(STAMP), rid_a)
check("long-A: DNS-valid", dns_ok(rid_a), rid_a)

# 3b. very-long name → even stamp+1 won't fit → base36-epoch fallback, still fits + sortable
name_b = "c" * 50  # budget = 63-50-5-2 = 6  (< len(stamp)=11)
rid_b = R.mint_labeled(name_b, "custombenchlabel", at=AT, fit_kind="bench")
check(
    "long-B: literal <cell>-bench-<id> ≤ 63",
    literal_fits(name_b, "bench", rid_b) <= 63,
    f"{literal_fits(name_b, 'bench', rid_b)}: {rid_b}",
)
check("long-B: DNS-valid", dns_ok(rid_b), rid_b)
check("long-B: non-empty run-id", len(rid_b) >= 1, rid_b)

# 3c. Extremely tight ids retain recipe entropy instead of the common timestamp prefix. This reproduces
# the Nemotron family shape that gave two different cells the same `ttjn` id.
name_c = "c" * 52  # budget = 63-52-5-2 = 4
tight_a = R.mint(name_c, "054fa5ee8f87e6d7540cdc1860675857", at=AT, fit_kind="bench")
tight_b = R.mint(name_c, "296db695d63c723f621ee88ae5a746fce", at=AT, fit_kind="bench")
check(
    "4-char ids for different recipes retain entropy",
    tight_a != tight_b,
    f"{tight_a} vs {tight_b}",
)
check(
    "both entropy-preserving ids still fit the literal Job name",
    literal_fits(name_c, "bench", tight_a) <= 63 and literal_fits(name_c, "bench", tight_b) <= 63,
)

# ── 4. determinism: same --at → same id (twice), independent of now() ───────────
d1 = R.mint_labeled(name_a, "custombenchlabel", at=AT, fit_kind="bench")
d2 = R.mint_labeled(name_a, "custombenchlabel", at=AT, fit_kind="bench")
check("determinism: identical for a fixed --at", d1 == d2, f"{d1} vs {d2}")

# ── 5. CLI wiring: --label flows through main(); --job-name --label composes ─────
cell = ROOT / "recipes/llm-perf/kvbm/qwen3-0-6b-b200-vllm-agg-kvbm-pareto"
if (cell / "recipe.yaml").is_file():

    def cli(*extra: str) -> str:
        return subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts/run_id.py"),
                str(cell),
                "--at",
                AT,
                *extra,
            ],
            capture_output=True,
            text=True,
        ).stdout.strip()

    out_label = cli("--fit", "bench", "--label", "rp3")
    check("CLI: --label rp3 → 260728t0312-rp3", out_label == f"{STAMP}-rp3", out_label)
    out_dated = cli("--fit", "bench", "--label", "260722t0312-a1b2")
    check(
        "CLI: --label already-dated passthrough",
        out_dated == "260722t0312-a1b2",
        out_dated,
    )
    out_job = cli("--job-name", "bench", "--label", "rp3")
    check(
        "CLI: --job-name --label → stamped job name",
        out_job.endswith(f"-bench-{STAMP}-rp3") and len(out_job) <= 63,
        out_job,
    )
else:
    check("CLI: sample cell present", False, f"missing {cell}")

print(("FAIL: " + ", ".join(fails)) if fails else "OK: run_id caller-supplied run-id handling")
sys.exit(1 if fails else 0)
