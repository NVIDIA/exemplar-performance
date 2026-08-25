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

"""benchmark_id.py <cell> — the STABLE identity of a benchmark (one level coarser than a matrix row).

A matrix row is a *cell* — digest-pinned — so a row's identity is `recipe_hash` (the exact recipe: image digest
+ every serving.extra_args flag + rendered manifests). recipe_hash MOVES when you re-pin the image or toggle a
perf flag. That is correct for reproducibility ("did anything change"), but it means you cannot compare a result
taken before an image bump to one after without the fingerprint declaring drift.

`benchmark_id` is the enduring identity that a row KEEPS across those re-pins — the benchmark itself: WHAT is
measured. It hashes model, gpu_type, arch, engine, serving_mode, framework, scenario, distribution, mode,
launcher, goal, `requires.gpu.count`, `serving.tp`, `serving.max_model_len`/`context_length`, and the
workload+criterion (`bench.dataset.sha256` or the complete `bench.synthetic` block, replay
`trace_sha256`s, or a task-source hash; plus `sweep_concurrency` and `sla`). It REFUSES to invent an
identity when no content discriminator is present. It EXCLUDES the image digest and ALL `serving.extra_args`, plus tuning/provenance
(`gpu_mem_util`, `offload`, `model_revision`, rendered manifests). So a new image or a tuning toggle keeps
`benchmark_id` constant but moves `recipe_hash`.

  Group results by `benchmark_id` to compare across image/flag rolls; use `recipe_hash` to prove two runs
  used the byte-identical setup.

  benchmark_id.py <cell-dir>          print the benchmark_id (benchmark_id: <sha256>)
  benchmark_id.py <cell-dir> --short  12-char short form
  benchmark_id.py --all [<root>]      every cell's benchmark_id
"""

import hashlib
import json
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("benchmark_id: requires pyyaml")

# Envelope fields that identify WHAT is measured (same identity axes as recipe_hash's envelope — model, hw,
# stack, workload distribution, goal). image digest / extra_args / tuning live in recipe_hash ONLY.
IDENTITY_ENVELOPE_KEYS = [
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


class UnknownBenchmarkIdentity(ValueError):
    """Raised when a recipe has no content-addressable workload to identify."""


def benchmark_identity(recipe: dict) -> dict:
    """The stable comparison subset → dict. Pure. EXCLUDES the image digest and ALL serving.extra_args, so an
    image roll or a perf-flag toggle does not change it. Structural serving shape (tp, max_model_len) and the
    deployment scale (gpu.count) ARE included — they define the deployment under test, not a tuning knob."""
    env = recipe.get("envelope") or {}
    req = env.get("requires") or {}
    serving = recipe.get("serving") or {}
    bench = recipe.get("bench") or {}
    replay = recipe.get("replay") or {}
    dataset_sha256 = (bench.get("dataset") or {}).get("sha256") if isinstance(bench, dict) else None
    synthetic = bench.get("synthetic") if isinstance(bench, dict) else None
    synthetic = synthetic if synthetic else None
    replay_traces = [
        r.get("trace_sha256") for r in (replay.get("rungs") or []) if isinstance(r, dict) and r.get("trace_sha256")
    ]
    if not any((dataset_sha256, synthetic, replay_traces)):
        raise UnknownBenchmarkIdentity(
            "workload is UNKNOWN: need bench.dataset.sha256, a non-empty bench.synthetic block, "
            "or replay.rungs[].trace_sha256"
        )
    ident = {
        "envelope": {k: env.get(k) for k in IDENTITY_ENVELOPE_KEYS},
        "gpu_count": (req.get("gpu") or {}).get("count"),
        "tp": serving.get("tp"),
        "max_model_len": serving.get("max_model_len") or serving.get("context_length"),
    }
    # disaggregated SCALE: the per-role worker count (serving.disagg.<role>.replicas) is part of the
    # deployment under test — 1P8D vs 2P1D differ in P:D split even when they share gpu_count/tp (e.g.
    # 1P8D and 3P6D at the same tp both draw 9×tp GPUs). Included ONLY when a role sets replicas != the
    # default 1, so every existing 1P1D/aggregated cell's benchmark_id is byte-for-byte unchanged.
    disagg = serving.get("disagg") or {}
    reps = {
        role: (disagg.get(role) or {}).get("replicas")
        for role in ("prefill", "decode")
        if (disagg.get(role) or {}).get("replicas", 1) != 1
    }
    if reps:
        ident["disagg_replicas"] = reps
    # llm-perf: the workload (dataset sha256), the sweep shape, and the SLA (the measurement criterion).
    if bench:
        ident["bench"] = {
            "dataset_sha256": dataset_sha256,
            "sweep_concurrency": bench.get("sweep_concurrency"),
            "sla": bench.get("sla"),
        }
        if synthetic:
            # A synthetic load has no dataset SHA. Its complete declared generation block is therefore the
            # workload content, not optional provenance: changing ISL/OSL, variance, seed, or request count
            # changes what was measured.
            ident["bench"]["synthetic"] = synthetic
    # replay: the workload is the set of traces — identify by their content hashes (not engine args).
    if replay:
        ident["replay_traces"] = replay_traces
    return ident


def benchmark_id(cell_dir: Path) -> str:
    recipe = yaml.safe_load((cell_dir / "recipe.yaml").read_text()) or {}
    canon = json.dumps(benchmark_identity(recipe), sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def main() -> int:
    try:
        args = sys.argv[1:]
        if args and args[0] == "--all":
            root = Path(args[1]) if len(args) > 1 else Path(__file__).resolve().parent.parent
            for rp in sorted((root / "recipes").glob("**/recipe.yaml")):
                print(f"{benchmark_id(rp.parent)}  {rp.parent.relative_to(root)}")
            return 0
        short = "--short" in args
        positional = [a for a in args if not a.startswith("--")]
        if not positional:
            sys.exit("usage: benchmark_id.py <cell-dir> [--short] | --all [<root>]")
        h = benchmark_id(Path(positional[0]))
        print(h[:12] if short else f"benchmark_id: {h}")
        return 0
    except UnknownBenchmarkIdentity as exc:
        print(f"benchmark_id: UNKNOWN — {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
