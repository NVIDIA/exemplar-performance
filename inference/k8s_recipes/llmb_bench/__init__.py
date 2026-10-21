# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

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
