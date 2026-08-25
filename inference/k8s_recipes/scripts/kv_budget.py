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

"""kv_budget.py <cell> [--log <file> | -]  — recommend --max-num-seqs from the server's real KV-cache budget.

Design A: the concurrency cap must be the GPU's memory-feasible max, not an arbitrary number — otherwise the
sweep either queues (cap too low) or preempts (cap too high). vLLM prints exactly what we need at startup:

    GPU KV cache size: 1,234,560 tokens
    Maximum concurrency for 262,144 tokens per request: 4.71x

Pipe the server log in (or pass --log). This parses those lines, cross-checks against the cell's
max_model_len, and prints a recommended `--max-num-seqs` (+ whether the committed sweep fits under it).

    kubectl -n <ns> logs <server-pod> | scripts/kv_budget.py <cell> -
    scripts/kv_budget.py <cell> --log server.log

Note: the recommendation is a FLOOR — with prefix caching + shared session prefixes (mooncake traces),
effective concurrency runs higher, so if KV usage stays low mid-sweep you can raise the cap further.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("kv_budget: requires pyyaml")

KV_TOKENS = re.compile(r"GPU KV cache size:\s*([\d,]+)\s*tokens", re.I)
MAX_CONC = re.compile(r"Maximum concurrency for\s*([\d,]+)\s*tokens per request:\s*([\d.]+)\s*x", re.I)
GPU_BLOCKS = re.compile(r"#?\s*GPU blocks:\s*([\d,]+)", re.I)


def num(s: str) -> int:
    return int(s.replace(",", ""))


def main() -> int:
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    cell = Path(args[0]).resolve()
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    serving = recipe.get("serving") or {}
    bench = recipe.get("bench") or {}
    ctx = serving.get("max_model_len")

    # current cap from extra_args
    cur_cap = None
    for a in serving.get("extra_args") or []:
        m = re.search(r"--max-num-seqs[= ]+(\d+)", a)
        if m:
            cur_cap = int(m.group(1))
    sweep = bench.get("sweep_concurrency") or []
    smax = max(sweep) if sweep else None

    # read the log (file, or stdin via "-")
    log = ""
    if "--log" in args:
        log = Path(args[args.index("--log") + 1]).read_text()
    elif "-" in args or not sys.stdin.isatty():
        log = sys.stdin.read()
    else:
        sys.exit("kv_budget: provide the server log via --log <file> or piped stdin (- )")

    kv = KV_TOKENS.search(log)
    mc = MAX_CONC.search(log)
    blocks = GPU_BLOCKS.search(log)

    print(f"cell:            {cell.name}")
    print(f"max_model_len:   {ctx}")
    print(f"current cap:     --max-num-seqs={cur_cap}   sweep max={smax}")

    feasible = None
    if mc:
        req_ctx, factor = num(mc.group(1)), float(mc.group(2))
        feasible = int(factor)  # vLLM's own "Nx" = concurrent full-context requests that fit
        print(f"vLLM reports:    KV holds {factor:.2f}x concurrent requests of {req_ctx:,} tokens")
        if ctx and req_ctx != ctx:
            print(f"  ⚠ vLLM sized for {req_ctx:,} tokens but recipe max_model_len={ctx:,} — check they match")
    elif kv and ctx:
        feasible = num(kv.group(1)) // ctx
        print(f"KV cache size:   {num(kv.group(1)):,} tokens  →  {feasible}x at {ctx:,} tokens/request")
    elif blocks:
        print(f"found '# GPU blocks: {blocks.group(1)}' but no token size — run with the full startup log")

    if feasible is None:
        print(
            "\nCould not find vLLM's KV-cache lines in the log. Grep the server startup for "
            "'GPU KV cache size' / 'Maximum concurrency'."
        )
        return 1

    print(f"\nFULL-CONTEXT FLOOR:  ~{feasible}x concurrent requests AT the full {ctx or 'max'} context.")
    print("  This is the WORST case. Real traces (shorter avg ISL + prefix caching) sustain many more —")
    print("  the cap you set should be well above this floor, then confirmed against live KV usage.")
    if smax is not None:
        if smax > feasible:
            print(
                f"  → sweep max {smax} is above the full-context floor {feasible}: fine IF your trace's real ISL "
                f"is shorter / prefix-cached (watch KV usage stays <~90%); if KV saturates + preempts, lower it."
            )
        else:
            print(
                f"  → sweep max {smax} is within even the worst-case floor {feasible} — safe; extend rungs up to find the ceiling."
            )
    print(
        "  → rule: never sweep above --max-num-seqs (CI enforces); pick the cap from live KV headroom, not this floor."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
