#!/usr/bin/env bash
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

# shellcheck disable=SC1003,SC1090,SC2012,SC2015,SC2016,SC2034,SC2116,SC2207,SC2221,SC2222,SC2295,SC2317

# new-cell.sh <scenario> <distribution> <model> <hardware> <setup>
#
# Scaffolds recipes/<scenario>/<distribution>/<model>-<hardware>-<setup>/ with a recipe.yaml skeleton
# (schema-appropriate for the scenario) + a RESULTS.md stub. Fill the "# REQUIRED:" blanks, then:
#   scripts/render.sh <cell> && make validate matrix
set -euo pipefail

usage() {
    echo "usage: new-cell.sh <llm-perf> <distribution> <model> <hardware> <setup>" >&2
    exit 2
}
[ "$#" -eq 5 ] || usage
SC="$1"
DIST="$2"
MODEL="$3"
HW="$4"
SETUP="$5"
case "$SC" in
    llm-perf)
        LAUNCHER=aiperf
        MODE=mooncake-trace
        EX_METRIC=max_concurrency_at_sla
        EX_UNIT=concurrency
        EX_CMP="highest sweep_concurrency rung meeting the SLA (metrics_summary.csv)"
        GOAL_LINE='  goal: max-concurrency-sla     # max-concurrency-sla | pareto  (HASHED — fixes the exemplar method)'
        ;;
    *)
        echo "unknown scenario '$SC'" >&2
        usage
        ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CELL="$ROOT/recipes/$SC/$DIST/${MODEL}-${HW}-${SETUP}"
[ -e "$CELL" ] && {
    echo "already exists: $CELL" >&2
    exit 1
}
mkdir -p "$CELL"

{
    cat << EOF
envelope:
  name: ${MODEL}-${HW}-${SETUP}-${DIST}
  model: ${MODEL}
  gpu_type: <GPU>              # REQUIRED: B200 / GB200 / GB300 …
  arch: <amd64|arm64>          # REQUIRED
  engine: vllm                 # REQUIRED: vllm | sglang | trtllm
  serving_mode: aggregated     # REQUIRED: aggregated | disaggregated
  framework: none
  scenario: ${SC}
  distribution: ${DIST}
  mode: ${MODE}                # llm-perf: mooncake-trace|custom-trace|synthetic
  launcher: ${LAUNCHER}
${GOAL_LINE}
  status: planned              # planned | wip | runs | performant | exemplar  (performant/exemplar are human-gated)
  results_link: ./RESULTS.md
  provenance:
    schema_version: k8s.${SC}.v1
    image_digest: <sha256:...>                       # REQUIRED
    image_ref: <registry/image@sha256:...>           # REQUIRED (used by render)
  exemplar:
    metric: ${EX_METRIC}
    unit: ${EX_UNIT}
    reference: null                                  # commit the bar once status=perf-data
    tolerance_pct: 5
    comparison: ${EX_CMP}
EOF

    if [ "$SC" = "llm-perf" ]; then
        cat << 'EOF'

serving:
  stack: vllm-agg
  model_repo: <hf-repo-id>       # REQUIRED
  served_model: <served-name>    # REQUIRED
  tp: 8
  max_model_len: <int>           # REQUIRED
  gpu_mem_util: 0.85
  offload: false
  extra_args: []

bench:
  # TRACE modes (mooncake-trace/custom-trace) need this dataset object. For mode: synthetic REMOVE `dataset`
  # and add a `synthetic:` block instead (isl/osl/+stddev/seed) — aiperf then generates random prompts.
  dataset:
    id: <dataset-id>                    # REQUIRED
    subpath: datasets/<dataset-id>/dataset.jsonl   # REQUIRED: path on the model/data PVC
    sha256: <sha256>                    # REQUIRED: load-bearing in benchmark_id (pins the exact trace)
    type: mooncake_trace                # aiperf dataset type; matches envelope.mode
    seed: 42
    max_isl: <int>                      # REQUIRED: this distribution's context cap (e.g. 262144 for 256k)
  sweep_concurrency: [64, 128, 256]     # REQUIRED
  sla: { ttft_ms: <int>, tpot_ms: <int>, stop_stat: p50 }   # REQUIRED for goal: max-concurrency-sla (drop for pareto)
  connection_reuse: sticky-user-sessions
  request_timeout_s: 3600
  gpu_telemetry: true
EOF
    else
        cat << 'EOF'

serving:
  stack: vllm-agg
  served_model: <served-name>    # REQUIRED
  tp: 8
  api_protocol: openai-compatible

replay:
  dataset: <dataset-id>          # REQUIRED
  schedule_mode: per-task-parallel
  osl_control_mode: force-exact
  tool_timeout_mode: captured-duration-ceiling
  captured_timeout_margin_sec: 5
  supports_min_tokens: true
  guardrails:
    no_internet: network-policy
    external_egress_from_task_pods: false
  acceptance_policy:
    strict_validation: true
    invalid_findings_allowed: 0
  rungs:
    - { concurrency: 8, parallelism: 8, trace: <trace-uri>, accepted: true }   # REQUIRED: >= 1 rung
EOF
    fi
} > "$CELL/recipe.yaml"

REL="${CELL#"$ROOT"/}"
printf '# Results — %s\n\n_TODO: fill in once the run completes._\n' "${MODEL}-${HW}-${SETUP} (${SC}/${DIST})" > "$CELL/RESULTS.md"

if [ "$SC" = "llm-perf" ]; then
    INTERP='Read **TPS/GPU** (capacity), **TTFT/TPOT** vs the SLA (a rung passes only if BOTH are within limits), and **% incomplete/cancelled**.'
else
    INTERP='Read **goodput** (net work vs median-time/cost) and **acceptance** (0 invalid findings under strict replay fidelity).'
fi
# The "How to reproduce" three-step block is NOT written here — scripts/reproduce.py is its single generator
# (it also knows whether this cell belongs to a group, which this script cannot). We emit the page skeleton
# with the markers and let reproduce.py fill them, so a new cell can never be born with a stale/hand-copied
# command list. `make reproduce-check` keeps it that way.
cat > "$CELL/README.md" << 'EOF'
# __TITLE__

_Declarative config: [`recipe.yaml`](recipe.yaml) · Numbers: [`RESULTS.md`](RESULTS.md)._

<!-- REPRODUCE:START -->
<!-- REPRODUCE:END -->

## What to read

__INTERP__

## Cell-specific notes

_TODO: what is unique about THIS cell (hardware shape, image pins, known caveats). Commands do NOT belong
here — the three steps above are the whole user-facing flow; primitives live in docs/CLI.md._
EOF
sed -i.bak "s#__TITLE__#${MODEL} · ${HW} · ${SETUP} (${SC} / ${DIST})#; s#__INTERP__#${INTERP}#" "$CELL/README.md" && rm -f "$CELL/README.md.bak"
python3 "$ROOT/scripts/reproduce.py" --write "$CELL" > /dev/null

echo "created $REL/{recipe.yaml, README.md, RESULTS.md}"
echo "next: fill the '# REQUIRED:' blanks, then:  scripts/render.sh $REL && make validate matrix"
