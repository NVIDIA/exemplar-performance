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

"""recipe_hash.py <cell> [--check] — a deterministic fingerprint of the FULL recipe.

ONE canonical hash over everything that DETERMINES the benchmark, so if anything that affects a run
changes, the fingerprint changes and two runs with the same recipe_hash provably used the identical
recipe. It covers:
  - the identity envelope (model, gpu_type, arch, engine, serving_mode, framework, scenario,
    distribution, mode, launcher) + the pinned image digest,
  - the full serving + bench (llm-perf) / replay blocks — engine args, the dataset
    sha256, sweep, SLA, etc.,
  - the content hashes of the rendered manifests (so a template change is caught too).
It EXCLUDES maturity/result fields (name, status, results_link, exemplar, run-metadata provenance) so the
fingerprint is STABLE across status promotions and only moves when the benchmark definition really changes.

Record the printed hash alongside a run (RESULTS + the run's metadata): that tags each result with the
exact recipe version that produced it, and `recipe_hash.py <cell>` later tells you if the cell has drifted.

Usage:
  recipe_hash.py <cell-dir>            print the fingerprint (recipe_hash: <sha256>)
  recipe_hash.py <cell-dir> --short    print the 12-char short form
  recipe_hash.py --all [<k8s-root>]    print every cell's fingerprint (used by build_catalog)
"""

import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("recipe_hash: requires pyyaml")

# Envelope fields that are part of the benchmark IDENTITY (not maturity/labels).
ENVELOPE_KEYS = [
    "model",
    "gpu_type",
    "arch",
    "engine",
    "serving_mode",
    "framework",
    "scenario",
    "distribution",
    "mode",
    "launcher",
    "goal",
]
# 'agent' ({placement, arch}) is ALSO a benchmark determinant — colocated vs a separate CPU pool,
# and the agent-node arch, change the measured goodput/cost + deployment. It is hashed CONDITIONALLY (only when
# present) in fingerprint_input, so adding it does NOT churn existing llm-perf hashes (which have no agent block).
# 'goal' IS a benchmark determinant (a subclass of the scenario): it fixes the measurement/exemplar METHOD,
# which is otherwise NOT hashed (exemplar is excluded). Two recipes with the same dataset+sweep but different
# goals measure different things and must have different fingerprints.


def _normalize_serving(serving):
    """The serving block is hashed VERBATIM, but a disagg role's `replicas` defaults to 1 (the 1P1D shape).
    Drop replicas==1 before hashing so a role that sets it explicitly to the default fingerprints identically
    to one that omits it — the same conditional-only-when-!=-default rule the agent block follows. replicas>1
    (a real 1P8D/2P1D topology) is kept, so recipe_hash still moves for a genuinely different server."""
    if not isinstance(serving, dict):
        return serving
    disagg = serving.get("disagg")
    if not isinstance(disagg, dict):
        return serving
    if not any(isinstance(disagg.get(r), dict) and disagg[r].get("replicas") == 1 for r in ("prefill", "decode")):
        return serving  # nothing to strip — leave the object untouched (no churn for existing cells)
    serving = json.loads(json.dumps(serving))  # deep copy; never mutate the caller's recipe
    for r in ("prefill", "decode"):
        role = serving["disagg"].get(r)
        if isinstance(role, dict) and role.get("replicas") == 1:
            del role["replicas"]
    return serving


def fingerprint_input(recipe: dict, cell_dir: Path) -> dict:
    env = recipe.get("envelope") or {}
    prov = env.get("provenance") or {}
    req = env.get("requires") or {}
    subset = {
        "envelope": {k: env.get(k) for k in ENVELOPE_KEYS},
        # the image is what actually runs; pin its digest (not the mutable ref).
        "image": prov.get("image_digest") or prov.get("image_ref"),
        # GPU shape affects the result; tolerations are cluster scheduling, not the benchmark.
        "requires_gpu": req.get("gpu"),
        # the scenario-specific blocks, verbatim (dataset carries its own sha256), with default disagg
        # replicas normalized away so an explicit `replicas: 1` never churns the fingerprint.
        "serving": _normalize_serving(recipe.get("serving")),
        "bench": recipe.get("bench"),
        "replay": recipe.get("replay"),
        # rendered manifests: hash each so a TEMPLATE change is caught even if recipe.yaml didn't move.
        "rendered": {},
    }
    # agent placement/arch is a determinant — include ONLY when present so adding
    # this axis leaves every existing llm-perf fingerprint byte-for-byte unchanged.
    if env.get("agent"):
        subset["envelope"]["agent"] = env["agent"]
    rdir = cell_dir / "rendered"
    if rdir.is_dir():
        for f in sorted(rdir.glob("*.yaml")):
            subset["rendered"][f.name] = hashlib.sha256(f.read_bytes()).hexdigest()
    return subset


def recipe_hash(cell_dir: Path) -> str:
    recipe = yaml.safe_load((cell_dir / "recipe.yaml").read_text()) or {}
    subset = fingerprint_input(recipe, cell_dir)
    canon = json.dumps(subset, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main() -> int:
    args = sys.argv[1:]
    if args and args[0] == "--all":
        root = Path(args[1]) if len(args) > 1 else Path(__file__).resolve().parent.parent
        for rp in sorted((root / "recipes").glob("**/recipe.yaml")):
            print(f"{recipe_hash(rp.parent)}  {rp.parent.relative_to(root)}")
        return 0
    short = "--short" in args
    positional = [a for a in args if not a.startswith("--")]
    if not positional:
        sys.exit("usage: recipe_hash.py <cell-dir> [--short] | --all [<root>]")
    h = recipe_hash(Path(positional[0]))
    print(h[:12] if short else f"recipe_hash: {h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
