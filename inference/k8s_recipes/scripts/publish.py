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

"""publish.py <cell> <run-dir> [--set-baseline] [--dry-run] — turn a completed sweep into a published result.

The end-to-end pipeline we validated, as one command: aggregate → metric → provenance stamp → refresh the
RESULTS.md data block → auto-bump wip/planned → runs → rebuild the matrix. `runs` is a factual bar (numbers
exist), so it sets automatically; performant/exemplar are human-gated and never touched here. --set-baseline
also commits the exemplar reference.

DISPATCHES BY SCENARIO (envelope.scenario): llm-perf → aggregate_metrics + exemplar_check (concurrency ceiling
+ SLA table). Everything else is shared.

  scripts/publish.py recipes/llm-perf/256k/nemotron-ultra-3-b200-vllm-agg results/<run-id>
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

try:
    import yaml
except ImportError:
    sys.exit("publish: requires pyyaml")

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import benchmark_id as _bid  # noqa: E402
import goal_handlers as _gh  # noqa: E402  — the GOAL-AWARE spine: one handler per (scenario, goal)
import archive_run as _ar  # noqa: E402  — storage tier split + runs/index.jsonl
import aggregate_cell as _agg  # noqa: E402  — deterministic cross-run aggregate
import charts as _charts  # noqa: E402
import fetch_receipt as _fetch_receipt  # noqa: E402  — the ONE reader of `_fetch_status.json`
import reproduce as _repro  # noqa: E402  — the SINGLE generator of the three-step reproduce block

START, END = "<!-- PUBLISH:START -->", "<!-- PUBLISH:END -->"

# data_provenance → the one-line fidelity note surfaced in the RESULTS.md badge (docs §5), so a reader always
# knows how trustworthy the numbers are.
_PROVENANCE_NOTE = {
    "archived": "real curated CSVs archived from the run-dir — full fidelity.",
    "reconstructed_from_record": "reconstructed from the committed record.json — faithful to the published "
    "summary; raw per-request data is not recoverable (re-run for that).",
    "reconstructed_from_prose": "reconstructed from the committed prose numbers — re-run for full fidelity.",
    "rerun": "a fresh re-run on the current recipe.",
}
_NOTES_STAMP = (
    "> **Non-authoritative.** Human/LLM interpretation only — **no numbers, no graphs** (those live in "
    "`RESULTS.md`, which is 100% code-generated from `record.json` + `aggregate/aggregate.json`). Nothing "
    "here feeds the record, the aggregate, or any published metric.\n"
)


def _num(v, nd=6):
    """A compact number for a generated table/headline: ints stay ints, floats trim to nd sig-ish digits."""
    if v is None:
        return "—"
    try:
        fv = float(v)
    except (TypeError, ValueError):
        return str(v)
    if fv == int(fv):
        return str(int(fv))
    return f"{fv:.{nd}g}"


def read_index(cell: Path) -> list:
    """runs/index.jsonl → run entries (the evolved, append-only run ledger)."""
    idx = cell / "runs" / "index.jsonl"
    out = []
    if idx.is_file():
        for line in idx.read_text().splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except ValueError:
                    pass
    return out


def _provenance_badge(runs: list) -> str:
    """The data_provenance badge line: the distinct fidelity level(s) across this cell's runs + the note."""
    provs = sorted({r.get("data_provenance") for r in runs if r.get("data_provenance")}) or ["unknown"]
    note = " · ".join(_PROVENANCE_NOTE.get(p, "") for p in provs).strip(" ·") or "—"
    return f"> **data_provenance:** `{' , '.join(provs)}` — {note}"


def _band_cell(band: dict | None) -> str:
    if not band:
        return "—"
    mean = band.get("mean")
    if band.get("n", 1) >= 2:
        return f"{_num(mean)} ({_num(band.get('min'))}–{_num(band.get('max'))}, n={band['n']})"
    return _num(mean)


def _aggregate_section(agg: dict, chart_rel: str | None) -> list:
    """The 'Aggregate across N runs' section — the deterministic cross-run band from aggregate.json (docs §4)."""
    hb = agg.get("headline_band") or {}
    n = agg.get("n_runs", 0)
    out = [f"## Aggregate across {n} run(s)", ""]
    per_cluster = hb.get("per_cluster") or {}
    pc = " · ".join(f"{c}: median {_num(v.get('median'))} (n={v.get('n')})" for c, v in per_cluster.items())
    out.append(
        f"**Band:** `{hb.get('metric')}` median **{_num(hb.get('median'))}** "
        f"(min {_num(hb.get('min'))} – max {_num(hb.get('max'))}, spread {hb.get('spread_pct', 0)}%, "
        f"{hb.get('n_runs', n)} run(s))" + (f"  ·  {pc}" if pc else "")
    )
    out.append("")
    rows = agg.get("per_rung_band") or []
    if rows:
        cols = [k for k in rows[0].keys() if k != "concurrency"]
        out.append("| concurrency | " + " | ".join(cols) + " |")
        out.append("|---|" + "---|" * len(cols))
        for r in rows:
            out.append("| " + str(r.get("concurrency")) + " | " + " | ".join(_band_cell(r.get(c)) for c in cols) + " |")
        out.append("")
    if chart_rel:
        out.append(f"![aggregate chart]({chart_rel})")
        out.append("")
    return out


def _provenance_table(record: dict) -> list:
    """A DETERMINISTIC provenance block from the committed record.json (never live git / a run-dir), so
    RESULTS.md regenerates byte-identically. Cites recipe_hash (provenance-check reads it).
    """
    fp = record.get("fingerprint") or {}
    prov = record.get("provenance") or {}
    ds = prov.get("dataset") or {}
    if fp.get("recipe_hash_source") == "launch_attestation":
        hash_note = (
            "_The ledger captured this `recipe_hash` before the benchmark Job was launched; CI verifies "
            "that the launch receipt still matches the published recipe._"
        )
    else:
        hash_note = (
            "_Legacy archive-era recipe-hash evidence: this page predates launch-time attestation and is "
            "not proof of a run-time capture._"
        )
    out = [
        "## Provenance",
        "",
        hash_note,
        "",
        "| field | value |",
        "|---|---|",
        f"| recipe_hash | `{fp.get('recipe_hash')}` |",
        f"| benchmark_id | `{fp.get('benchmark_id')}` |",
        f"| image | `{prov.get('image_digest')}` |",
        f"| dataset | `{ds.get('id')}` · sha256 `{ds.get('sha256')}` |",
        f"| cluster · profile | {prov.get('cluster')} |",
        f"| run_id | {prov.get('run_id')} |",
        f"| gpu_h | {prov.get('gpu_hours')} GPU·h |",
        f"| git commit | `{fp.get('git_commit')}` |",
        f"| date (UTC) | {(prov.get('date_utc') or '')[:10]} |",
        "",
    ]
    # The reproduce commands are NOT repeated here — they LEAD the page (see _reproduce_section). A copy/paste
    # user must not have to scroll past a provenance table to find out how to run this.
    return out


def _reproduce_section(cell: Path) -> list:
    """The three steps, at the TOP of the page, from the one generator (scripts/reproduce.py) that also owns
    every README. PURE + deterministic (a function of the committed recipe.yaml + sibling envelopes).
    """
    return ["## Reproduce", ""] + _repro.three_steps(cell) + [""]


def render_results_md(cell: Path) -> str:
    """PURE + DETERMINISTIC: the FULL RESULTS.md, 100% code-generated from committed `record.json` +
    `aggregate/aggregate.json` + `recipe.yaml` — no run-dir, no live git, NO dependency on NOTES.md. The same
    committed data always yields the same bytes, which is what `make results-check` byte-diffs (docs §3).
    """
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    env = recipe.get("envelope") or {}
    handler = _gh.resolve(env.get("scenario"), env.get("goal"))
    record = json.loads((cell / "record.json").read_text())
    ident = record.get("identity") or {}
    agg = {}
    ap = cell / "aggregate" / "aggregate.json"
    if ap.is_file():
        agg = json.loads(ap.read_text())
    runs = read_index(cell)
    goal = ident.get("goal") or ident.get("scenario") or "chart"
    safe = str(goal).replace("/", "-").replace(" ", "_")
    agg_png = f"aggregate/charts/{safe}.png" if (cell / "aggregate" / "charts" / f"{safe}.png").is_file() else None

    prov = record.get("provenance") or {}
    run_id = prov.get("run_id")
    run_png = (
        f"runs/{run_id}/curated/charts/{safe}.png"
        if run_id and (cell / "runs" / str(run_id) / "curated" / "charts" / f"{safe}.png").is_file()
        else None
    )

    out = [f"# {ident.get('cell') or cell.name} — results", ""]
    out.append(_provenance_badge(runs))
    # Link NOTES.md only if the cell actually HAS one — same rule as the chart links above. An
    # unconditional link to a file we do not guarantee becomes a broken link the moment a cell
    # ships without NOTES.md, and the CI link check is whole-repo, so it then blocks everyone.
    notes_link = (
        " Interpretation (non-authoritative) lives in [`NOTES.md`](NOTES.md)." if (cell / "NOTES.md").is_file() else ""
    )
    out.append(
        "> _RESULTS.md is 100% code-generated from committed `record.json` + `aggregate/aggregate.json`."
        + notes_link
        + "_"
    )
    out.append("")
    # The three steps FIRST — a recipe page answers "how do I recreate this?" before anything else.
    out += _reproduce_section(cell)
    # Aggregate across N runs
    out += _aggregate_section(agg, agg_png)
    # Latest run
    out.append(f"## Latest run — {run_id} ({prov.get('cluster')}, {(prov.get('date_utc') or '')[:10]})")
    out.append("")
    out.append(f"**Headline:** {handler.headline_from_record(record)}")
    out.append("")
    out.append("### Results")
    out.append(handler.table_from_record(record, recipe))
    out.append("")
    out.append("### Chart")
    out.append(_charts.render_ascii(record, runs))
    if run_png:
        out.append("")
        out.append(f"![{ident.get('goal')} chart]({run_png})")
    out.append("")
    # Runs ledger (links to runs/)
    out.append("## Runs")
    out.append("")
    out.append("| run_id | date | cluster | metric | value | data_provenance | curated |")
    out.append("|---|---|---|---|---|---|---|")
    for r in runs:
        cur = r.get("curated") or f"runs/{r.get('run_id')}/curated"
        out.append(
            f"| {r.get('run_id')} | {r.get('date')} | {r.get('cluster')} | {r.get('metric')} | "
            f"{_num(r.get('value'))} | {r.get('data_provenance')} | [{cur}]({cur}) |"
        )
    out.append("")
    # Provenance (deterministic, from the record)
    out += _provenance_table(record)
    return "\n".join(out).rstrip() + "\n"


def split_notes(cell: Path) -> bool:
    """Migrate any hand/LLM narrative ABOVE the old PUBLISH:START marker into NOTES.md (stamped
    non-authoritative), so publish can own RESULTS.md fully. Idempotent: never overwrites an existing NOTES.md,
    never fabricates one when there's no narrative to move. Returns True if a NOTES.md was written.
    """
    rp = cell / "RESULTS.md"
    notes_p = cell / "NOTES.md"
    if notes_p.exists() or not rp.exists():
        return False
    text = rp.read_text()
    above = text.split(START)[0] if START in text else ""
    # drop the H1 title + the seeded placeholder; keep only real prose
    lines = []
    for ln in above.splitlines():
        s = ln.strip()
        if s.startswith("# ") or s in _PLACEHOLDER_LINES:
            continue
        lines.append(ln)
    narrative = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if not narrative:
        return False
    title = f"# {cell.name} — notes (non-authoritative)"
    notes_p.write_text("\n".join([title, "", _NOTES_STAMP, narrative, ""]).rstrip() + "\n")
    return True


def run(*a, **k):
    return subprocess.run([sys.executable, *[str(x) for x in a]], capture_output=True, text=True, **k)


def read_csv(path: Path):
    lines = [l for l in path.read_text().splitlines() if not l.startswith("#")]
    import csv

    return list(csv.DictReader(lines))


def set_reference(recipe_text: str, value) -> str:
    """Write envelope.exemplar.reference = value (format-preserving)."""
    return re.sub(
        r"(^\s*reference:\s*).*$",
        lambda m: m.group(1) + str(value),
        recipe_text,
        count=1,
        flags=re.M,
    )


def bump_status(recipe_text: str) -> tuple[str, str | None]:
    """wip/planned → runs (sticky: never downgrade performant/exemplar). Targeted, format-preserving."""
    m = re.search(r"^(\s*status:\s*)(planned|wip)\b.*$", recipe_text, re.M)
    if not m:
        return recipe_text, None
    old = m.group(2)
    return recipe_text[: m.start()] + m.group(1) + "runs" + recipe_text[m.end() :], old


def identity_mismatch(recipe: dict, run_meta: dict) -> list[str]:
    """Ways a run-dir's run_meta.json disagrees with a cell's recipe identity (empty = it belongs here).
    Pure — guards against publishing another cell's/cluster's results into this cell (Kubernetes API constraint).
    """
    env = recipe.get("envelope") or {}
    serving = recipe.get("serving") or {}
    out: list[str] = []
    rmodel = (run_meta.get("model_name") or "").strip()
    # run_meta records the client-facing model id. A recipe's envelope.model is its stable family identity,
    # while serving.served_model is the exact alias registered at /v1/models. Both explicitly belong to this
    # recipe; accepting either fixes legitimate aliases without fuzzy matching unrelated model names.
    model_aliases = tuple(
        dict.fromkeys(
            value.strip()
            for value in (env.get("model"), serving.get("served_model"))
            if isinstance(value, str) and value.strip()
        )
    )
    if rmodel and model_aliases and rmodel not in model_aliases:
        out.append(f"model {rmodel!r} not in recipe aliases {model_aliases!r}")
    cell_gpu = ((env.get("requires") or {}).get("gpu") or {}).get("count") or serving.get("tp")
    rgpu = run_meta.get("gpu_count")
    try:
        if rgpu and cell_gpu and int(rgpu) != int(cell_gpu):
            out.append(f"gpu_count {rgpu} != recipe {cell_gpu}")
    except (TypeError, ValueError):
        pass
    return out


def variant_overrides(run_dir: Path) -> dict | None:
    """PURE-ish (one/two file reads): the runtime env overrides a run was launched with, or None for a clean
    pinned-config run. Two independent sources, either of which is conclusive:
      • results/<run-id>/_variant.json — written by run.sh BEFORE the deploy, so it exists even for a run that
        crashed, was killed, or never produced numbers;
      • run_meta.json's `overrides` / `variant_id` — the same facts carried inside the run's own provenance.
    A run launched with --env-set/--env-unset served a DIFFERENT configuration than the committed recipe, so
    its recipe_hash (which is deliberately unmoved by a runtime override) does NOT describe it.
    """
    for name, key in (("_variant.json", None), ("run_meta.json", "overrides")):
        p = run_dir / name
        if not p.is_file():
            continue
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        d = d if key is None else d.get(key)
        if isinstance(d, dict) and (d.get("set") or d.get("unset") or d.get("variant_id")):
            return d
    return None


def _variant_refusal(run_dir: Path, ov: dict) -> str:
    parts = [f"-{k}" for k in (ov.get("unset") or [])] + [f"+{k}={v}" for k, v in (ov.get("set") or {}).items()]
    return (
        f"[publish] REFUSING: {run_dir.name} is a VARIANT run "
        f"(variant {ov.get('variant_id') or '?'}: {' '.join(parts) or 'runtime env override'}).\n"
        "  It was launched with --env-set/--env-unset, so the SERVED configuration differs from the "
        "committed recipe.\n"
        "  recipe_hash is deliberately unaffected by a runtime override — which is exactly why this "
        "number must not be filed under it.\n"
        "  There is NO --force for this: a variant number published against a clean recipe_hash is "
        "indistinguishable from a real result.\n"
        "  → Diagnostic? Read it from the run-dir. Keeping the change? Edit the recipe (/change-recipe), "
        "re-render, and re-run WITHOUT overrides."
    )


_PLACEHOLDER_LINES = {"_TODO: fill in once the run completes._"}  # seeded by new-cell.sh
_STALE_HINTS = [
    "no run yet",
    "sweep in progress",
    "numbers land when",
    "top-out",
    "in-flight",
    "tbd",
]


def refresh_narrative(above: str) -> tuple[str, list[str]]:
    """Pure (B2-5). The RESULTS.md narrative ABOVE the PUBLISH markers is human-owned, but a fresh cell seeds a
    placeholder line (`_TODO: fill in once the run completes._`) and agents leave speculative prose that goes
    stale once real numbers land. Strip ONLY the exact new-cell placeholder line(s) — never other prose — and
    return (cleaned_above, stale_phrases) so publish can WARN about the rest instead of silently rewriting a
    human's words."""
    kept = [ln for ln in above.splitlines() if ln.strip() not in _PLACEHOLDER_LINES]
    cleaned = re.sub(r"\n{3,}", "\n\n", "\n".join(kept)).strip()
    cleaned = (cleaned + "\n") if cleaned else ""
    low = cleaned.lower()
    stale = sorted({h for h in _STALE_HINTS if h in low})
    return cleaned, stale


def fetch_incomplete(receipt: dict | None) -> str | None:
    """Return why a fetched run is unsuitable for publication, or ``None`` when usable.

    Runs created before fetch receipts were introduced remain supported. A present receipt must include the
    required verification evidence.
    """
    return _fetch_receipt.blocking_reason(receipt, allow_absent=True)


def main() -> int:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    dry = "--dry-run" in sys.argv
    set_base = "--set-baseline" in sys.argv
    # the invoked profile, threaded from the dispatcher as --cluster=<profile>, is the fallback for the
    # runs.jsonl 'cluster' field when the run metadata lacks it (compare --repro / fleet key on it).
    _cluster_arg = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--cluster=")), "")
    if len(args) < 2:
        sys.exit(__doc__)
    cell, run_dir = Path(args[0]).resolve(), Path(args[1]).resolve()
    recipe_path = cell / "recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text()) or {}
    env = recipe.get("envelope") or {}
    scenario = env.get("scenario")
    sla = ((recipe.get("bench") or {}).get("sla")) or {}

    # Guard: refuse a VARIANT run — one launched through run.sh/deploy.sh with --env-set/--env-unset. Runtime
    # overrides are hash-neutral BY DESIGN (they never touch rendered/*.yaml), so nothing downstream would
    # otherwise notice that the numbers describe a different served configuration than the recipe_hash they
    # would be stamped with. That is the same class of hole as an un-hashed task tarball (provenance
    # reported MATCH after the benchmark itself had changed) — so this one is UNCONDITIONAL: no --force.
    _ov = variant_overrides(run_dir)
    if _ov:
        sys.exit(_variant_refusal(run_dir, _ov))

    # The run directory must belong to this cell. Refuse a mismatch to avoid publishing results under the wrong recipe (even
    # in --dry-run) unless --force.
    _rm = run_dir / "run_meta.json"
    if _rm.exists():
        try:
            _m = json.loads(_rm.read_text())
        except Exception:
            _m = {}
        mism = identity_mismatch(recipe, _m)
        if mism and "--force" not in sys.argv:
            sys.exit(
                f"[publish] REFUSING: {run_dir.name} does not match cell {cell.name} — "
                + "; ".join(mism)
                + ".\n  This run was produced by a different recipe/cluster; publishing "
                "it would write wrong numbers. Use the correct run-dir, or --force to override."
            )

    # Guard: refuse a run-dir whose fetch didn't complete (R2-4 — Teleport stream timeouts leave a partial tree
    # that looks populated). The receipt (written by fetch_results.sh) is authoritative; absent → allow (old runs).
    _fr = run_dir / "_fetch_status.json"
    if _fr.exists():
        try:
            _receipt = json.loads(_fr.read_text())
        except Exception:
            _receipt = None
        why = fetch_incomplete(_receipt)
        if why and "--force" not in sys.argv:
            sys.exit(f"[publish] REFUSING: {run_dir.name} — {why}. Use --force to override.")

    # ── ONE deterministic loop (docs §3), identical across every (scenario, goal) ──────────────────────
    # aggregate raw → archive into curated/ + raw/ → export the record FROM COMMITTED curated → cross-run
    # aggregate → 100%-code-generated RESULTS.md. The keystone: export_record + RESULTS read the committed
    # curated dir, never this scratch run-dir, so publish REGENERATES identically once the run-dir is gone.
    handler = _gh.resolve(scenario, env.get("goal"))
    provenance = "rerun" if (cell / "record.json").exists() else "archived"
    run_meta_path = run_dir / "run_meta.json"

    recipe_text, bumped = bump_status(recipe_path.read_text())

    if dry:
        rows = handler.read_native_rows(run_dir, env.get("goal"))
        metric, value, _s = handler.compute_metric(rows, recipe) if rows else (handler.metric, None, None)
        print(f"[dry-run] would archive {run_dir.name} → runs/<id>/curated + raw (provenance={provenance})")
        print(f"[dry-run] would export record.json + aggregate.json FROM committed curated ({metric}={value})")
        print(f"[dry-run] would render RESULTS.md (100% code-generated) + split narrative → NOTES.md")
        print(
            f"[dry-run] would bump status: {bumped or '(no change)'} → runs" if bumped else "[dry-run] status unchanged"
        )
        print(f"[dry-run] would rebuild the matrix")
        return 0

    if bumped:
        recipe_path.write_text(recipe_text)

    # 1) storage split — curated/ (committed) + raw/ (gitignored), append runs/index.jsonl (with data_provenance)
    archived = _ar.archive_results(cell, run_dir, None, provenance)
    curated = archived["curated"]
    metric, value = archived["entry"]["metric"], archived["entry"]["value"]

    # 2) KEYSTONE: the canonical record — computed from the COMMITTED curated dir, NOT the scratch run-dir.
    run(
        ROOT / "scripts/export_record.py",
        cell,
        curated,
        "--out",
        cell / "record.json",
        f"--cluster={_cluster_arg}",
    )

    # 3) deterministic cross-run aggregate + its chart; per-run chart under the run's curated/charts
    agg = _agg.build(cell)
    (cell / "aggregate").mkdir(exist_ok=True)
    (cell / "aggregate" / "aggregate.json").write_text(json.dumps(agg, indent=2) + "\n")
    _agg.write_charts(cell)
    _ar._write_run_chart(cell, curated)

    # 4) legacy runs.jsonl timing ledger (build_catalog reads it for the median wall/GPU-h) — kept for back-compat
    if run_meta_path.exists():
        try:
            m = json.loads(run_meta_path.read_text())
            secs = m.get("wall_seconds_total")
            if secs is not None:
                with open(cell / "runs.jsonl", "a") as fh:
                    fh.write(
                        json.dumps(
                            {
                                "run_id": m.get("run_id", ""),
                                "date": (m.get("completed_at_utc") or m.get("started_at_utc") or "")[:10],
                                "cluster": m.get("cluster") or m.get("profile") or _cluster_arg or "",
                                "metric": metric,
                                "value": value,
                                "recipe_hash": archived["entry"]["recipe_hash"],
                                "recipe_hash_at_launch": archived["entry"].get("recipe_hash_at_launch"),
                                "benchmark_id": _bid.benchmark_id(cell),
                                "wall_seconds": secs,
                                "gpu_count": m.get("gpu_count"),
                            }
                        )
                        + "\n"
                    )
        except Exception:
            pass  # timing log is best-effort; never block a publish

    # 5) exemplar reference — the ONLY exemplar_check --set path is via the handler (no second direct call).
    if set_base and value is not None:
        handler.set_baseline(cell, curated / handler.summary_csv, value)

    # 6) NOTES split + the 100%-code-generated RESULTS.md (publish OWNS it fully; no dependency on NOTES.md).
    if split_notes(cell):
        print("  ✎ migrated the RESULTS.md narrative into NOTES.md (non-authoritative)")
    (cell / "RESULTS.md").write_text(render_results_md(cell))

    run(ROOT / "scripts/build_catalog.py")  # refresh matrix + catalog
    _print_publish_card(cell, run_dir, metric, value, _cluster_arg, set_base, bumped)
    return 0


def _fmt_dur(secs):
    if not secs:
        return None
    m, s = divmod(int(secs), 60)
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


def rung_coverage(run_meta: dict) -> dict:
    """PURE. Compare the rungs a sweep was CONFIGURED to run against the ones it actually executed.

    A sweep that dies partway still writes a well-formed run_meta.json describing only the rungs that
    ran, so enumerating `sweep_steps` alone reports a truncated run as a clean one. The intended list
    (`concurrencies`) and the executed list are both already recorded — the gap between them is the
    signal, and it must be reported rather than inferred from what happens to be present.

    An early exit is legitimate ONLY where the driver can actually exit early. In `fixed` mode the
    sweep loop has no `break` (serving/aiperf/templates/bench-job.yaml.j2) — it runs every rung in
    CONCURRENCIES — so a rung breaching its SLA can never be why a fixed sweep stopped. Treating a
    breach as exculpatory there disarms this guard on exactly the cells it protects: `goal: pareto`
    cells carry no `bench.sla`, so classify_step falls back to a 100ms TPOT default that real rungs
    routinely exceed (104-125ms measured), which would stamp "by design" on every truncated pareto
    run. Only a sweep-level `sweep_stop_reason` exculpates a fixed sweep; per-step breaches exculpate
    an adaptive sweep, which genuinely stops at its ceiling.

    Returns {intended, executed, missing, complete, verdict}. `complete` is None (UNKNOWN) when a
    sweep plan exists but is unreadable — never True, because an unreadable plan cannot certify
    coverage. A lane that does not sweep at all (writes no run_meta.json) is not UNKNOWN — it has
    no rungs to be incomplete about, and reporting it as UNKNOWN would strip the exemplar verdict
    from every such run.
    """

    def _ints(raw):
        if isinstance(raw, str):
            if not raw.strip():
                return None  # "" is an ABSENT plan (adaptive mode writes it), never an empty one
            raw = raw.replace(",", " ").split()
        if not isinstance(raw, (list, tuple)):
            return None
        out = []
        for item in raw:
            try:
                out.append(int(item))
            except (TypeError, ValueError):
                return None
        return out or None  # an empty list cannot certify "nothing was missed"

    SWEEP_KEYS = ("concurrencies", "executed_sequence", "sweep_steps", "sweep_mode")
    if not any(run_meta.get(k) for k in SWEEP_KEYS):
        # Not a swept run (no rung concept at all) — distinct from a broken plan. Stay silent.
        return {
            "intended": None,
            "executed": None,
            "missing": [],
            "complete": True,
            "verdict": "",
        }

    intended = _ints(run_meta.get("concurrencies"))
    executed = _ints(run_meta.get("executed_sequence"))
    if executed is None:
        executed = [s.get("concurrency") for s in (run_meta.get("sweep_steps") or [])]
        executed = _ints([c for c in executed if c is not None])
    if intended is None or executed is None:
        return {
            "intended": intended,
            "executed": executed,
            "missing": [],
            "complete": None,
            "verdict": "rung coverage UNKNOWN (sweep plan declared but unreadable)",
        }

    missing = [c for c in intended if c not in set(executed)]
    if not missing:
        return {
            "intended": intended,
            "executed": executed,
            "missing": [],
            "complete": True,
            "verdict": "",
        }

    # Why did it stop? Only reasons the DRIVER could have acted on count.
    adaptive = (run_meta.get("sweep_mode") or "").strip().lower() == "adaptive"
    reason = (run_meta.get("sweep_stop_reason") or "").strip()
    if adaptive and not reason:
        # Adaptive genuinely halts at a breach/ceiling, so a per-step reason IS the explanation.
        for step in run_meta.get("sweep_steps") or []:
            if step.get("breaches") or (step.get("stop_reason") or "").strip():
                reason = (
                    step.get("stop_reason") or ""
                ).strip() or f"c={step.get('concurrency')} breached {','.join(step.get('breaches') or [])}"
                break
    if reason:
        return {
            "intended": intended,
            "executed": executed,
            "missing": missing,
            "complete": True,
            "verdict": f"stopped early by design ({reason})",
        }
    return {
        "intended": intended,
        "executed": executed,
        "missing": missing,
        "complete": False,
        "verdict": (
            f"INCOMPLETE — {len(executed)} of {len(intended)} rungs ran; "
            f"{', '.join(f'c={c}' for c in missing)} never ran, and nothing recorded why"
        ),
    }


def _print_publish_card(cell: Path, run_dir: Path, metric, value, cluster_arg, set_base, bumped):
    """Print a visual result card after a successful publish."""
    # Read record.json — written earlier in this publish flow; has result, identity, provenance, detail.
    record = {}
    try:
        record = json.loads((cell / "record.json").read_text())
    except Exception:
        pass
    result = record.get("result") or {}
    identity = record.get("identity") or {}
    prov = record.get("provenance") or {}
    detail = record.get("detail") or {}

    # Read sweep_steps from run_meta.json (more granular than detail.rungs — has breaches, gating_ratio).
    # The SAME file records the rungs the sweep was configured to run, so coverage is checked here
    # rather than inferred from whichever steps happen to be present.
    sweep_steps, run_meta = [], {}
    try:
        run_meta = json.loads((run_dir / "run_meta.json").read_text())
        sweep_steps = run_meta.get("sweep_steps") or []
    except Exception:
        pass
    coverage = rung_coverage(run_meta)

    metric_name = result.get("metric") or metric or "—"
    unit = result.get("unit") or ""
    val_str = f"{_num(value)}  {unit}".strip() if value is not None else "—"

    # Exemplar comparison — only shown when a reference is actually committed in the recipe.
    # The database-wired path (pulling live exemplar targets) is not wired yet; skip gracefully.
    ref = result.get("reference")
    tol = result.get("tolerance_pct")
    if coverage["complete"] is not True and ref is not None:
        # A reference is a FULL-sweep quantity. Scoring a short run against it is not a pass or a
        # fail, it is a category error — the rungs that did not run are exactly the ones that move
        # the metric. Refuse the verdict instead of printing a ✅/❌ nobody can act on.
        exemplar_str = (
            f"⚠ not comparable to ref={_num(ref)} — this run's rung coverage is "
            f"{'UNKNOWN' if coverage['complete'] is None else 'INCOMPLETE'}"
        )
    elif ref is not None and value is not None and tol is not None:
        floor = ref * (1 - tol / 100)
        ok = value >= floor
        icon = "✅" if ok else "❌"
        exemplar_str = f"{icon} ref={_num(ref)}, tol={tol}%  " f"(need ≥{_num(floor)}, got {_num(value)})"
    else:
        exemplar_str = "— (no reference set; use --set-baseline after vetting)"

    # Cluster / date / GPU identity
    cluster = cluster_arg or prov.get("cluster") or "—"
    date_str = (prov.get("date_utc") or prov.get("completed_at_utc") or "")[:10] or "—"
    gpu_count = prov.get("gpu_count") or identity.get("gpu_count") or ""
    gpu_type = identity.get("gpu_type") or ""
    gpu_str = f"{gpu_count}× {gpu_type}".strip() if (gpu_count or gpu_type) else "—"
    cluster_str = f"{cluster}  ·  {date_str}  ·  {gpu_str}"

    # Wall time
    dur = _fmt_dur(prov.get("wall_seconds"))
    gpu_h = prov.get("gpu_hours")
    wall_str = dur or "—"
    if gpu_h:
        wall_str += f"  ({gpu_h:.1f} GPU-h)"

    # Per-rung pass/fail summary
    if sweep_steps:
        parts = []
        for s in sweep_steps:
            c = s.get("concurrency", "?")
            ok = s.get("passed", False)
            brs = s.get("breaches") or []
            tag = "✓" if ok else f"✗({','.join(brs)})" if brs else "✗"
            parts.append(f"c={c} {tag}")
        rung_str = "  ".join(parts)
    elif detail.get("rungs"):
        rung_str = "  ".join(f"c={int(r.get('concurrency', 0))}" for r in detail["rungs"])
    else:
        rung_str = "—"
    # Name what is ABSENT. Listing only the rungs that ran renders a truncated sweep as a clean one.
    if coverage["missing"]:
        rung_str += "  ·  " + "  ".join(f"c={c} ✗" for c in coverage["missing"]) + "  NOT RUN"

    # Status bump note
    status_str = f"status bumped → {bumped}" if bumped else "status unchanged (already at runs)"

    # Assemble rows
    rows = [
        ("metric", metric_name),
        ("value", val_str),
        ("exemplar", exemplar_str),
        ("cluster", cluster_str),
        ("wall", wall_str),
        ("rungs", rung_str),
        ("status", status_str),
    ]
    if coverage["verdict"]:
        rows.insert(2, ("coverage", coverage["verdict"]))

    # Next-step hints (use cell.name, not full path, to keep lines short)
    hints = []
    if ref is None:
        hints.append(f"llmb-k8s publish {cell.name} <run-dir> --set-baseline")
    hints.append(f"llmb-k8s compare --repro {cell.name}  # reproducibility check")

    # Render — box for data rows, plain-text hints below (commands are too long for a fixed-width box)
    k_w = max(len(k) for k, _ in rows) + 2
    content_rows = [f"  {k:<{k_w}} {v}" for k, v in rows]
    header = f"  published  {cell.name}"
    inner_w = max(len(header), max(len(r) for r in content_rows))
    border = "═" * (inner_w + 1)

    print()
    print(f"╔{border}╗")
    print(f"║{header:<{inner_w + 1}}║")
    print(f"╠{border}╣")
    for row in content_rows:
        print(f"║{row:<{inner_w + 1}}║")
    print(f"╚{border}╝")
    print()
    for h in hints:
        print(f"  Next:  {h}")


if __name__ == "__main__":
    raise SystemExit(main())
