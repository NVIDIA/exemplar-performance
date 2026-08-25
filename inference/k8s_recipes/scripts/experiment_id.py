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

"""experiment_id.py <cell> [--stamp <run_meta.json>] — the k8s analog of llmb-tools/compute_mbridge_experiment_id.py.

Training computes experiment_id = sha256(normalized_config + fw_version) and writes it into the run's config so
every result carries a reproducible DEFINITION identity. In this collection that identity already exists as
`recipe_hash` (a normalized content hash over the envelope + image digest + serving/bench/replay + rendered
manifests, excluding volatile fields). So:

    experiment_id == recipe_hash

We keep `recipe_hash` as the internal name (an experiment is a recipe DEFINITION, not an instance) and surface
it as `experiment_id` at the OUTPUT boundary — record.json and, with --stamp, run_meta.json — so k8s and slurm
results join on the same field in the shared results DB. A *run* is one execution instance of an experiment
(its run_id / run-dir); the experiment_id is shared across reruns of the same recipe.

Because experiment_id/recipe_hash includes the image digest and every serving.extra_args flag (mirroring
training's fw_version + full config), it MOVES on an image roll or a perf toggle. To compare results ACROSS
those changes, --stamp also writes `benchmark_id` — the coarser, stable benchmark identity (see benchmark_id.py)
that excludes the image and extra_args. Group results by benchmark_id to compare; use experiment_id/recipe_hash
to prove the exact setup.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recipe_hash as _rh
import benchmark_id as _bid


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if not args:
        sys.exit(__doc__)
    cell = Path(args[0]).resolve()
    if not (cell / "recipe.yaml").is_file():
        sys.exit(f"experiment_id: no recipe.yaml at {cell}")
    eid = _rh.recipe_hash(cell)
    bid = _bid.benchmark_id(cell)

    if "--stamp" in sys.argv:
        i = sys.argv.index("--stamp")
        meta_path = Path(sys.argv[i + 1]) if i + 1 < len(sys.argv) else None
        if not meta_path or not meta_path.is_file():
            sys.exit("experiment_id: --stamp needs an existing run_meta.json path")
        meta = json.loads(meta_path.read_text())
        meta["experiment_id"] = eid  # == recipe_hash; the DB-facing definition identity
        meta.setdefault("recipe_hash", eid)
        meta["benchmark_id"] = bid  # stable benchmark identity (excludes image + extra_args)
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        print(f"stamped experiment_id={eid} benchmark_id={bid} into {meta_path}")
        return 0

    print(f"experiment_id: {eid}   (== recipe_hash)")
    print(f"benchmark_id:  {bid}   (stable across image/flag rolls)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
