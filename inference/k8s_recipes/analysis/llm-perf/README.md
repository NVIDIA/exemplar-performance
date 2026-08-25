# analysis/llm-perf — deterministic serving-performance pipeline

These tools provide repeatable processing for fetched benchmark artifacts (see [`../README.md`](../README.md)).

- **aggregate:** `aggregate_metrics.py <run-dir> --gpu-count N` → **`metrics_summary.csv`** (schema v6,
  one row per concurrency rung: TPS/GPU, TPS/User, TTFT/TPOT percentiles, KV/prefix-cache, gpu_stats,
  SLA pass/fail, %incomplete).
- **dashboard:** `sla_compare_dashboard.py --run "<label>=<run-dir>" [...] --out dashboard.html` — the
  canonical SLA-scatter + per-run distributions. Same inputs → same output.
- **exemplar_check:** `exemplar_check.py <cell-dir> <metrics_summary.csv> [--reference N] [--json]` →
  computes the recipe's `envelope.exemplar.metric` (`max_concurrency_at_sla` = highest rung with
  TTFT≤limit AND TPOT≤limit at `bench.sla.stop_stat`, or `tps_per_gpu_at_sla` at that rung) and compares
  it to the committed `reference` bar within `tolerance_pct` (better is fine; >tol below fails). CSV-only
  and deterministic; judges the SLA exactly like the dashboard. `--reference null` → prints the value as
  a **baseline candidate** to commit. This is the cross-CLUSTER reproducibility check (same recipe+GPU,
  different cluster).

The recipe reproduction steps call these tools directly so the same inputs produce the same summaries.
