# Analysis tools

This directory contains the deterministic tools used to summarize completed benchmark runs and compare their results.

## LLM performance results

The [`llm-perf`](llm-perf/) tools process fetched run artifacts into a canonical `metrics_summary.csv` file:

```text
raw run artifacts -> aggregate_metrics.py -> metrics_summary.csv
```

- `aggregate_metrics.py` creates one summary row per concurrency rung.
- `sla_compare_dashboard.py` renders the summary for review.
- `exemplar_check.py` evaluates the metric and tolerance declared by the recipe, when a reference is configured.

Run the analysis through the CLI:

```bash
scripts/llmb-k8s analyze <recipe> results/<run-id>
```

## Comparing results

[`compare.py`](compare.py) compares compatible result records across recipes or runs. It can report same-hardware reproducibility or rank results across hardware configurations.

```bash
scripts/llmb-k8s compare <recipe...>
```

See [`docs/CLI.md`](../docs/CLI.md) for the complete command syntax.
