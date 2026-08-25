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

"""python -m llmb_bench <plan|metrics|aiperf|smoke|gate> — CLI for the carved-out runner pieces (Phase 2 M1).

  python -m llmb_bench plan fixed "32 64 128"          → the fixed rung list
  python -m llmb_bench plan adaptive <start> <ratio> <lo> <hi>   → the adaptive candidate grid
  cat server_metrics.prom | python -m llmb_bench metrics         → JSON summary of the /metrics blob
  python -m llmb_bench aiperf <N>                       → the `aiperf profile …` argv for concurrency N (from env)
  <status> on argv + body on stdin: python -m llmb_bench smoke <http_status>   → validate a smoke response

These mirror what the in-pod runner does today; as M1 progresses the sweep loop itself moves here.
"""

from __future__ import annotations

import json
import os
import sys

from . import aiperf as _aiperf
from . import gate as _gate
from . import metrics as _metrics
from . import plan as _plan
from . import smoke as _smoke


def main(argv: list[str]) -> int:
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    cmd, rest = argv[0], argv[1:]
    if cmd == "metrics":
        print(json.dumps(_metrics.parse_metrics(sys.stdin.read())))
        return 0
    if cmd == "aiperf":
        if len(rest) != 1:
            sys.exit("usage: python -m llmb_bench aiperf <N>   (config from the bench env vars)")
        print(" ".join(_aiperf.build_aiperf_args(_aiperf.cfg_from_env(os.environ), int(rest[0]))))
        return 0
    if cmd == "smoke":
        if len(rest) != 1:
            sys.exit("usage: python -m llmb_bench smoke <http_status>   (response body on stdin)")
        ok, reason = _smoke.validate_smoke(rest[0], sys.stdin.read())
        print(reason)
        return 0 if ok else _smoke.SMOKE_FAIL_EXIT
    if cmd == "gate":
        # aiperf profile_export JSON on stdin → the per-rung SLA verdict JSON. Limits/stat from env (as the shell).
        import json as _json

        export = _json.loads(sys.stdin.read() or "{}")
        r = _gate.evaluate_rung(
            export,
            stat=os.environ.get("STOP_STAT", "p50"),
            ttft_limit_ms=float(os.environ.get("TTFT_LIMIT_MS", "10000")),
            tpot_limit_ms=float(os.environ.get("TPOT_LIMIT_MS", "100")),
            aiperf_rc=int(os.environ.get("AIPERF_RC", "0")),
            concurrency=int(os.environ.get("STEP_CONCURRENCY", "0")),
            phase=os.environ.get("STEP_PHASE", ""),
        )
        print(_json.dumps(r, sort_keys=True))
        return 0 if r["passed"] else 0  # verdict is in the JSON; exit stays 0 (the sweep loop reads `passed`)
    if cmd == "plan":
        if rest and rest[0] == "adaptive":
            if len(rest) != 5:
                sys.exit("usage: python -m llmb_bench plan adaptive <start> <ratio> <lo> <hi>")
            grid = _plan.adaptive_grid(int(rest[1]), float(rest[2]), int(rest[3]), int(rest[4]))
        else:
            grid = _plan.fixed_rungs(" ".join(rest[1:] if rest and rest[0] == "fixed" else rest))
        print(" ".join(str(x) for x in grid))
        return 0
    sys.exit(f"llmb_bench: unknown command '{cmd}' (plan | metrics | aiperf | smoke | gate)")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
