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

"""archive_run.py — split a run's artifacts into the committed `curated/` tier + the gitignored `raw/` tier.

The storage half of the deterministic output-data pipeline. Given a
completed run, it lands two tiers under `recipes/<cell>/runs/<run_id>/`:

  curated/   COMMITTED  → datastore later. Small, high-value, regenerable-from:
               metrics_summary.csv | goodput_summary.csv   (native per-rung summary; the aggregator's output)
               rungs.csv                                    (normalized, goal-agnostic per-rung table)
               run_meta.json                                (provenance, with git_commit/git_ref PINNED)
               fingerprint.json                             (per-run repro fingerprint, if present)
               charts/<goal>.png                            (best-effort per-run chart)
  raw/       GITIGNORED → S3 later. The heavy rest: *.prom, smoke_*.json, gpu_stats.csv,
               sweep_decisions.json, report.json, trial_*.jsonl, concurrency_*/ …

and appends one line to `recipes/<cell>/runs/index.jsonl` (the evolved runs.jsonl schema — carries
`data_provenance` so a reconstructed run is never mistaken for a fresh archive). Idempotent: re-archiving the
same run_id overwrites its curated/ deterministically and rewrites (never duplicates) its index line.

Two entry points, one per migration bucket (docs §5):
  archive_run.py <cell> <results-dir> [--run-id ID] [--provenance archived|rerun]
      Bucket A — a real local run-dir still in results/: lift the real curated CSVs (full fidelity).
  archive_run.py <cell> --from-record  [--run-id ID] [--provenance reconstructed_from_record]
      Bucket B — raw GC'd but record.json survives: reconstruct curated/rungs.csv from record.detail.rungs.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "analysis"))
import recipe_hash as _rh  # noqa: E402
import benchmark_id as _bid  # noqa: E402
import goal_handlers as _gh  # noqa: E402

try:
    import yaml
except ImportError:
    sys.exit("archive_run: requires pyyaml")

PROVENANCES = {
    "archived",
    "reconstructed_from_record",
    "reconstructed_from_prose",
    "rerun",
}
# curated file names lifted out of a raw run-dir; everything else is heavy raw.
_CURATED_LIFT = {
    "metrics_summary.csv",
    "goodput_summary.csv",
    "run_meta.json",
    "fingerprint.json",
    "launch_attestation.json",
}
_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def _git(*a):
    try:
        return subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True).stdout.strip() or None
    except Exception:
        return None


def _read_json(p: Path) -> dict:
    if not p.is_file():
        return {}
    try:
        return json.loads(p.read_text())
    except ValueError:
        return {}


def _write_rungs_csv(path: Path, handler, rungs: list) -> None:
    """The normalized, goal-agnostic per-rung table. Columns = handler.rung_keys; cells round-trip byte-for-byte
    (repr for floats, '' for None) so rungs.csv → native → detail.rungs reproduces the record exactly.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(handler.rung_keys)]
    for r in rungs:
        lines.append(",".join(_gh._csv_cell(v) for v in handler.rungs_csv_row(r)))
    path.write_text("\n".join(lines) + "\n")


def _write_run_chart(cell: Path, curated: Path) -> str | None:
    """Best-effort per-run PNG into curated/charts/<goal>.png, rendered from the committed record.json only."""
    import charts as _charts  # noqa: E402 — optional matplotlib inside

    rec_p = cell / "record.json"
    if not rec_p.is_file():
        return None
    record = json.loads(rec_p.read_text())
    ident = record.get("identity") or {}
    goal = ident.get("goal") or ident.get("scenario") or "chart"
    safe = str(goal).replace("/", "-").replace(" ", "_")
    png = curated / "charts" / f"{safe}.png"
    runs = _load_index_runs(cell)
    if _charts.render_png(record, png, runs):
        return f"charts/{safe}.png"
    return None


def _load_index_runs(cell: Path) -> list:
    """The cell's runs as flat dicts (for chart scalar bands): prefer runs/index.jsonl, fall back to legacy
    runs.jsonl. Read-only helper."""
    out: list = []
    for jl in (cell / "runs" / "index.jsonl", cell / "runs.jsonl"):
        if jl.is_file():
            for line in jl.read_text().splitlines():
                line = line.strip()
                if line:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
            if out:
                return out
    return out


def _append_index(cell: Path, entry: dict) -> None:
    """Append (idempotent) one run to runs/index.jsonl, keyed on run_id — a re-archive rewrites its line, never
    duplicates it. Deterministic key order so the committed file is byte-stable."""
    idx = cell / "runs" / "index.jsonl"
    idx.parent.mkdir(parents=True, exist_ok=True)
    keys = [
        "run_id",
        "date",
        "cluster",
        "metric",
        "value",
        "recipe_hash",
        "recipe_hash_at_launch",
        "recipe_hash_source",
        "benchmark_id",
        "wall_seconds",
        "gpu_count",
        "data_provenance",
        "git_commit",
        "curated",
    ]
    line = json.dumps({k: entry.get(k) for k in keys})
    existing = []
    if idx.is_file():
        for ln in idx.read_text().splitlines():
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except ValueError:
                continue
            if obj.get("run_id") != entry.get("run_id"):
                existing.append(ln)
    existing.append(line)
    idx.write_text("\n".join(existing) + "\n")


def _index_entry(
    cell: Path,
    recipe: dict,
    run_meta: dict,
    run_id: str,
    handler,
    rows: list,
    provenance: str,
    recipe_hash: str,
    hash_source: str,
) -> dict:
    metric, value, _src = handler.compute_metric(rows, recipe) if rows else (handler.metric, None, None)
    return {
        "run_id": run_id,
        "date": (run_meta.get("completed_at_utc") or run_meta.get("started_at_utc") or "")[:10] or None,
        "cluster": run_meta.get("cluster") or run_meta.get("profile") or None,
        "metric": metric,
        "value": value,
        # This is intentionally NOT recipe_hash(cell): it is the immutable value
        # captured before the benchmark Job was submitted.
        "recipe_hash": recipe_hash,
        "recipe_hash_at_launch": (recipe_hash if hash_source == "launch_attestation" else None),
        "recipe_hash_source": hash_source,
        "benchmark_id": _bid.benchmark_id(cell),
        "wall_seconds": run_meta.get("wall_seconds_total"),
        "gpu_count": run_meta.get("gpu_count"),
        "data_provenance": provenance,
        "git_commit": run_meta.get("git_commit"),
        "curated": f"runs/{run_id}/curated",
    }


def _launch_hash(results_dir: Path, run_id: str) -> str:
    """Read an immutable launch receipt, failing closed for a real-run archive."""
    candidates = [
        results_dir / "launch_attestation.json",
        results_dir / "curated" / "launch_attestation.json",
    ]
    receipt_path = next((p for p in candidates if p.is_file()), None)
    if receipt_path is None:
        raise ValueError(
            "UNKNOWN launch provenance: launch_attestation.json is absent; refusing to derive recipe_hash at "
            "archive time. Re-run with scripts/run.sh (or a lane launcher) so the hash is captured before apply."
        )
    receipt = _read_json(receipt_path)
    value = receipt.get("recipe_hash")
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise ValueError(f"UNKNOWN launch provenance: {receipt_path} lacks a valid recipe_hash")
    if receipt.get("kind") != "recipe_hash_at_launch" or receipt.get("run_id") != run_id:
        raise ValueError(
            f"UNKNOWN launch provenance: {receipt_path} does not attest run_id {run_id!r} "
            f"(found {receipt.get('run_id')!r})"
        )
    return value


def _pin_git(run_meta: dict, git_commit: str | None, git_ref: str | None) -> dict:
    """Stamp the git identity into the curated run_meta so export_record regenerates the record deterministically
    (no live-git/HEAD dependency). Never overwrites a value already present."""
    rm = dict(run_meta)
    rm.setdefault("git_commit", git_commit or _git("rev-parse", "--short", "HEAD"))
    rm.setdefault("git_ref", git_ref or _git("rev-parse", "--abbrev-ref", "HEAD"))
    return rm


def _handler_for(recipe: dict):
    env = recipe.get("envelope") or {}
    return _gh.resolve(env.get("scenario"), env.get("goal")), env.get("goal")


def launcher_argv(results_dir: Path) -> list | None:
    """The load gen's LITERAL invocation per rung, lifted from aiperf's own output: each concurrency's
    `logs/aiperf/profile_export_aiperf.json` records the exact `cli_command` it ran. A curated PROVENANCE
    field (goes into run_meta, which is NOT hashed) — so capturing the true argv never re-fingerprints a
    recipe. None when absent (reconstructed cells, or non-aiperf generators that don't emit a cli_command).
    """
    import re as _re

    cmds = []
    for pj in sorted(results_dir.glob("concurrency_*/logs/aiperf/profile_export_aiperf.json")):
        try:
            doc = json.loads(pj.read_text()) or {}
            c = (doc.get("input_config") or {}).get("cli_command") or doc.get("cli_command")
        except (ValueError, OSError):
            c = None
        if c:
            m = _re.search(r"concurrency_(\d+)", str(pj))
            cmds.append({"concurrency": int(m.group(1)) if m else None, "command": c})
    return cmds or None


def cpu_info(results_dir: Path) -> dict | None:
    """CPU/memory facts from the run report — useful for CPU-bound workloads. Lifted at
    curation time from report.cpu + report.system_info.memory. Skips 'unknown'/0 placeholders. None if absent.
    """
    for rj in sorted(results_dir.glob("**/report.json")):
        try:
            r = json.loads(rj.read_text()) or {}
        except (ValueError, OSError):
            continue
        cpu = r.get("cpu") or {}
        mem = ((r.get("system_info") or {}).get("memory") or {}).get("total_gb")
        info = {}
        if cpu.get("model") and cpu.get("model") != "unknown":
            info["cpu_model"] = cpu["model"]
        if cpu.get("cores_logical"):
            info["cpu_cores_logical"] = cpu["cores_logical"]
        if mem:
            info["memory_total_gb"] = mem
        if info:
            return info
    return None


def task_source_sha(results_dir: Path) -> str | None:
    """The external task-source.tgz sha256 — the unique id for the exact task BODIES — if the stage
    step left it in the results (archive.sha256 / task_source.sha256). The task IMAGES are already
    digest-pinned by the committed image-digest-manifest, so this is a belt-and-suspenders id for the .tgz.
    """
    for f in sorted(results_dir.glob("**/archive.sha256")) + sorted(results_dir.glob("**/task_source.sha256")):
        try:
            h = f.read_text().split()[0].strip()
            if len(h) == 64:
                return h
        except (OSError, IndexError):
            pass
    return None


def archive_results(cell: Path, results_dir: Path, run_id: str | None, provenance: str) -> dict:
    """Bucket A — split a real results/<run-id> dir into curated/ + raw/."""
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    handler, goal = _handler_for(recipe)
    run_meta = _read_json(results_dir / "run_meta.json")
    run_id = run_id or run_meta.get("run_id") or results_dir.name
    launch_hash = _launch_hash(results_dir, run_id)
    curated = cell / "runs" / run_id / "curated"
    raw = cell / "runs" / run_id / "raw"

    # IDEMPOTENCY GUARD (data-loss bug): if results_dir IS the already-archived cell/runs/<run_id> dir — a
    # re-score/re-publish of a COMMITTED run (e.g. a goal change: max-concurrency-sla → pareto), not a fresh
    # scratch results dir — then `curated`/`raw` below resolve to results_dir's OWN subdirs. The destructive
    # split would then rmtree its own source curated/ (deleting metrics_summary.csv), read zero native rows
    # from the run root (the CSV lives under curated/), and copy the emptied curated/+raw/ back INTO raw/
    # (nesting raw/raw, raw/curated). Detect that and RE-DERIVE from the existing committed curated in place:
    # metrics_summary.csv is untouched, rungs.csv/index are recomputed under the current goal, raw/ is left as-is.
    if results_dir.resolve() == (cell / "runs" / run_id).resolve() and curated.is_dir():
        rows = handler.read_native_rows(curated, goal)
        _write_rungs_csv(curated / "rungs.csv", handler, [handler.rung_from_native(x) for x in rows])
        cur_meta = _read_json(curated / "run_meta.json") or run_meta
        entry = _index_entry(
            cell,
            recipe,
            cur_meta,
            run_id,
            handler,
            rows,
            provenance,
            launch_hash,
            "launch_attestation",
        )
        _append_index(cell, entry)
        return {"run_id": run_id, "curated": curated, "raw": raw, "entry": entry}

    if curated.exists():
        shutil.rmtree(curated)
    if raw.exists():
        shutil.rmtree(raw)
    curated.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)

    # native summary → curated (aggregate first if only raw is present)
    rows = handler.read_native_rows(results_dir, goal)
    summ = results_dir / handler.summary_csv
    if summ.is_file():
        shutil.copy2(summ, curated / handler.summary_csv)
    # normalized rungs.csv (goal-agnostic) — built from the native rows via the handler
    normalized = [handler.rung_from_native(x) for x in rows]
    _write_rungs_csv(curated / "rungs.csv", handler, normalized)
    # run_meta with pinned git + the load gen's invocation: aiperf's literal cli_command per rung.
    # Curated provenance (run_meta is NOT hashed).
    run_meta = _pin_git(run_meta, None, None)
    argv = launcher_argv(results_dir)
    if argv:
        run_meta["launcher_argv"] = argv
    cpu = cpu_info(results_dir)  # CPU/memory block (useful for CPU-bound workloads)
    if cpu:
        run_meta["cpu_info"] = cpu
    tsha = task_source_sha(results_dir)  # unique id for the exact external task bundle (if staged it out)
    if tsha:
        run_meta["task_source_sha256"] = tsha
    (curated / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")
    shutil.copy2(results_dir / "launch_attestation.json", curated / "launch_attestation.json")
    # fingerprint (optional, if the run produced one)
    fp = results_dir / "fingerprint.json"
    if fp.is_file():
        shutil.copy2(fp, curated / "fingerprint.json")

    # everything else (heavy) → raw/, preserving the run-dir layout
    for item in sorted(results_dir.iterdir()):
        if item.name in _CURATED_LIFT:
            continue
        dst = raw / item.name
        if item.is_dir():
            shutil.copytree(item, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(item, dst)

    entry = _index_entry(
        cell,
        recipe,
        run_meta,
        run_id,
        handler,
        rows,
        provenance,
        launch_hash,
        "launch_attestation",
    )
    _append_index(cell, entry)
    return {"run_id": run_id, "curated": curated, "raw": raw, "entry": entry}


def archive_from_record(cell: Path, run_id: str | None, provenance: str) -> dict:
    """Bucket B — raw GC'd, but record.json survives: reconstruct curated/rungs.csv + run_meta.json from the
    committed record (record.detail.rungs is authoritative for the published summary; raw per-request data is
    gone, which is what raw/ + a re-run are for)."""
    recipe = yaml.safe_load((cell / "recipe.yaml").read_text()) or {}
    handler, goal = _handler_for(recipe)
    record = _read_json(cell / "record.json")
    if not record:
        sys.exit(f"archive_run --from-record: {cell} has no record.json to reconstruct from")
    prov = record.get("provenance") or {}
    fp = record.get("fingerprint") or {}
    run_meta = dict(prov.get("run_meta") or {})
    # ensure the fields export_record/aggregate read are present + git PINNED from the record's fingerprint
    run_meta.setdefault("run_id", prov.get("run_id"))
    run_meta.setdefault("cluster", prov.get("cluster"))
    run_meta.setdefault("wall_seconds_total", prov.get("wall_seconds"))
    run_meta.setdefault("gpu_count", prov.get("gpu_count"))
    run_meta = _pin_git(run_meta, fp.get("git_commit"), fp.get("git_ref"))
    run_id = run_id or run_meta.get("run_id") or prov.get("run_id") or "run"
    record_hash = fp.get("recipe_hash")
    if not isinstance(record_hash, str) or not _HASH_RE.fullmatch(record_hash):
        raise ValueError(
            "UNKNOWN reconstructed provenance: record.json has no valid fingerprint.recipe_hash; "
            "refusing to calculate one from the current recipe"
        )

    curated = cell / "runs" / run_id / "curated"
    if curated.exists():
        shutil.rmtree(curated)
    curated.mkdir(parents=True, exist_ok=True)
    rungs = (record.get("detail") or {}).get("rungs") or []
    _write_rungs_csv(curated / "rungs.csv", handler, rungs)
    (curated / "run_meta.json").write_text(json.dumps(run_meta, indent=2) + "\n")

    rows = handler.read_native_rows(curated, goal)  # re-reads the rungs.csv we just wrote (native mapping)
    entry = _index_entry(
        cell,
        recipe,
        run_meta,
        run_id,
        handler,
        rows,
        provenance,
        record_hash,
        "reconstructed_record",
    )
    _append_index(cell, entry)
    return {"run_id": run_id, "curated": curated, "raw": None, "entry": entry}


def main() -> int:
    argv = sys.argv[1:]
    pos = [a for a in argv if not a.startswith("--")]
    if not pos:
        sys.exit(__doc__)
    cell = Path(pos[0]).resolve()
    if not (cell / "recipe.yaml").is_file():
        sys.exit(f"archive_run: {cell} is not a cell (no recipe.yaml)")
    run_id = next((argv[i + 1] for i, a in enumerate(argv) if a == "--run-id"), None)
    from_record = "--from-record" in argv
    prov = next((a.split("=", 1)[1] for a in argv if a.startswith("--provenance=")), None)
    if prov is None:
        i = next((k for k, a in enumerate(argv) if a == "--provenance"), None)
        prov = argv[i + 1] if i is not None and i + 1 < len(argv) else None

    if from_record:
        prov = prov or "reconstructed_from_record"
        if prov not in PROVENANCES:
            sys.exit(f"archive_run: --provenance must be one of {sorted(PROVENANCES)}")
        res = archive_from_record(cell, run_id, prov)
    else:
        if len(pos) < 2:
            sys.exit("archive_run: need a <results-dir> (or use --from-record)")
        results_dir = Path(pos[1]).resolve()
        if not results_dir.is_dir():
            sys.exit(f"archive_run: results dir not found: {results_dir}")
        prov = prov or "archived"
        if prov not in PROVENANCES:
            sys.exit(f"archive_run: --provenance must be one of {sorted(PROVENANCES)}")
        res = archive_results(cell, results_dir, run_id, prov)

    e = res["entry"]
    print(
        f"[archive_run] {cell.name}: run {res['run_id']} → runs/{res['run_id']}/curated  "
        f"({e['data_provenance']}, {e['metric']}={e['value']})"
    )
    print(
        f"[archive_run] index: runs/index.jsonl (+ raw/ {'skipped (from-record)' if res['raw'] is None else 'written, gitignored'})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
