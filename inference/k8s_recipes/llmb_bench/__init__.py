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

"""llmb_bench — the aiperf sweep runner, extracted from serving/aiperf/templates/bench-job.yaml.j2.

Harness extraction, milestone M1. Today the ~1,240-line bench-job.yaml.j2 embeds a
`/bin/sh -c` script with inline `python3 - <<PY` heredocs that (a) `pip install`s aiperf in-cluster at
runtime and (b) re-implements the concurrency ramp/binary-search that `scripts/sweep_planner.py` already
owns. This package lifts that orchestration into versioned, unit-tested Python so the template can shrink to
a thin Job spec (`python -m llmb_bench …`) against a pinned bench image (M2), with no runtime apt/pip.

M1 carves the self-contained + error-prone pieces behind a CLI, fully offline + tested:
  - plan    — sweep PLANNING (delegates to sweep_planner, single source)
  - metrics — vLLM /metrics PARSING (sum counters / read gauges)
  - smoke   — the pre-sweep chat-completion gate (HTTP 2xx · JSON · no API error · non-empty choices; exit 67)
  - aiperf  — the `aiperf profile …` argv builder, incl. the workload-cap PRIORITY (request-count > duration >
              full-trace replay) + the warmup/request-count multipliers
  - gate    — the per-rung SLA decision (rc==0 AND TTFT≤limit AND TPOT≤limit at the stop-stat) + gating_ratio;
              THE pass/fail that `max_concurrency_at_sla` and the adaptive search hang on
The remaining glue is the imperative sweep LOOP wiring (already partly in scripts/adaptive_sweep.py) + the
metrics_summary assembly (already externalized in extract_metrics.sh) — thin, and only fully testable live. NOTHING here is wired into the
running template yet: it's the package being built up before the image + template swap (M2/M3) — a coordinated
recipe_hash re-baseline that must land only when NO llm-perf run is in flight.
"""
